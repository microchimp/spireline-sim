"""tools.fidelity.corpus: the recorded-fight corpus and its regression gate.

Synthetic recordings in the referee's shape. Every folder of recordings is read by its policy; a fight
recorded in several files is kept once with all its source paths, whatever per-step instrumentation it
carries; a fight the game broke off is kept up to its last good step; the recording format and the
flags come from the recording; the judge names the first divergence, every re-anchored step and any
emulator exception; and the gate fails exactly when something that worked in the baseline does not,
or when a baseline fight was not compared.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import corpus as cp  # noqa: E402
from tools.fidelity import fight_compare as fc  # noqa: E402

ID_MAP = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))
NM = fc.names(ID_MAP)
CID, EID = ID_MAP["cards"], ID_MAP["enemies"]
STRIKE, SLUG = "STRIKE_IRONCLAD", "CORPSE_SLUG"


def game_state(hp=27, rnd=1, php=70):
    return {"in_combat": True, "round": rnd, "player": {"hp": php, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "hand": [[STRIKE, False]], "draw": [], "discard": [], "exhaust": [],
            "enemies": [{"name": SLUG, "hp": hp, "max_hp": 27, "block": 0, "powers": []}],
            "rng": {"Shuffle": {"seed": 1, "counter": 1}}}


def emu_state(hp=27, php=70):
    card = {"def_id": CID["StrikeIronclad"], "upgraded": False}
    return {"in_combat": True, "draw_order": [], "rng": {"Shuffle": 1},
            "snapshot": {"player": {"hp": php, "max_hp": 80, "block": 0, "energy": 3, "buffs": []},
                         "piles": {"hand": [card], "discard": [], "exhaust": []},
                         "enemies": [{"def_id": EID["CorpseSlug"], "hp": hp, "block": 0, "buffs": []}]}}


def play(card=STRIKE, **extra):
    return {"kind": "play", "hand_index": 0, "card": card, "target": 0, "target_slot": 0, **extra}


def recording(seed="AAAAAAAAAA", hps=(27, 21, 15), card=STRIKE, choices=(), **extra):
    """A clean game result: the start, then one play per further HP value."""
    steps = [{"action": None, "state": game_state(hps[0])}]
    steps += [{"action": play(card, **extra), "state": game_state(h)} for h in hps[1:]]
    return {"seed": seed, "character": "IRONCLAD", "ascension": 10, "neow_pick": 0, "deck_add": [], "room": [0, 1],
            "choices": list(choices), "steps": steps, "phase_ms": {"neow": 5}, "timing_ms": {"play": 9}}


def broken_off(seed="AAAAAAAAAA"):
    """A fight the game broke off: a play, an end of turn recorded during the turn's setup, two parked plays."""
    r = recording(seed=seed, hps=(27, 21), played=True)
    turn2 = game_state(21, rnd=2)
    r["steps"] += [{"action": {"kind": "end_turn"}, "state": turn2},
                   {"action": play(played=True), "state": turn2}, {"action": play(played=True), "state": turn2}]
    r["choices"] = [{"kind": "cards", "step": 1, "picked": [0]}, {"kind": "cards", "step": 3, "picked": [1]}]
    r["error"] = 'idle_timeout after step 4: {"kind":"play"}'
    return r


class CorpusTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.jobs = self.dir / "jobs"
        self.jobs.mkdir()
        self.corpus = cp.Corpus(self.dir / "corpus")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, results):
        p = self.jobs.parent / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"results": results}), encoding="utf-8")
        return p

    def add(self, *files):
        return self.corpus.add_files(list(files), rand_cards=set(), jobs=self.jobs)

    def fights(self):
        return self.corpus.load_index()["fights"]


class IngestTest(CorpusTestCase):
    def test_a_fight_in_several_files_is_kept_once_with_every_source(self):
        a, b = recording(), recording(seed="BBBBBBBBBB")
        moved = dict(a, timing_ms={"play": 1234}, _id=3, _pid=77, _t=5)  # same fight, other timing, serve tags
        n1 = self.add(self.write("jobs/x.s0.out.json", [a]))
        n2 = self.add(self.write("jobs/x.out.json", [moved, b, {"seed": "C", "error": "stalled", "steps": []}]))
        self.assertEqual((n1["new"], n2["new"], n2["known"], n2["not_kept"]), (1, 1, 1, 1))
        first = self.fights()[0]
        self.assertEqual(len(self.fights()), 2)
        self.assertEqual(first["sources"], ["jobs/x.s0.out.json", "jobs/x.out.json"])
        self.assertEqual((first["steps"], first["plays"], first["build"]), (3, 2, cp.KNOWN_BUILD))
        stored = self.corpus.read(first["id"])
        self.assertNotIn("timing_ms", stored)
        self.assertNotIn("phase_ms", stored)
        self.assertEqual(stored["steps"], a["steps"])

    def test_per_step_instrumentation_is_not_part_of_the_fight(self):
        a = recording()
        timed = json.loads(json.dumps(a))
        for n, s in enumerate(timed["steps"]):
            s.update(cost={"cpu_ms": n}, trace=[["frame", n]], hidden={"x": n}, checksums=[n])
        self.assertEqual(cp.fight_id(a), cp.fight_id(timed))
        self.add(self.write("jobs/a.out.json", [a]), self.write("jobs/frames/t.out.json", [timed]))
        (m,) = self.fights()
        self.assertEqual(m["sources"], ["jobs/a.out.json", "jobs/frames/t.out.json"])
        self.assertTrue(all(set(s) == {"action", "state"} for s in self.corpus.read(m["id"])["steps"]))

    def test_other_steps_choices_or_spec_options_are_another_fight(self):
        a = recording()
        self.assertNotEqual(cp.fight_id(a), cp.fight_id(recording(hps=(27, 21, 14))))
        self.assertNotEqual(cp.fight_id(a), cp.fight_id(recording(choices=[{"kind": "cards", "step": 1, "picked": [0]}])))
        self.assertNotEqual(cp.fight_id(a), cp.fight_id(dict(a, deck_add=["BASH"])))
        self.assertNotEqual(cp.fight_id(a), cp.fight_id(dict(a, enemy_hp_mult=5)))
        self.assertEqual(cp.fight_id(a), cp.fight_id(dict(a, cost={"cpu_ms": 1}, enemy_hp_mult=1, setup=[])))

    def test_recording_format_from_the_fight_or_its_file(self):
        v1 = recording(choices=[{"kind": "cards", "options": [], "picked": []}])
        v2 = recording(choices=[{"kind": "cards", "step": 0, "options": [], "picked": []}])
        v3 = recording(played=True)
        self.assertEqual([cp.recording_format(r) for r in (v1, v2, v3)], [1, 2, 3])
        # A fight with no card choice of its own takes the tags its file's driver wrote.
        plain = recording()
        self.assertEqual(cp.recording_format(plain, cp.file_evidence([plain, v2])), 2)
        self.assertEqual(cp.recording_format(plain, cp.file_evidence([plain, v3])), 3)

    def test_random_enemy_play_before_the_driver_fix_is_flagged_not_dropped(self):
        rand = cp.random_enemy_cards()
        self.assertTrue({"VOLLEY", "SWORD_BOOMERANG"} <= rand)
        self.assertNotIn(STRIKE, rand)
        old, new = recording(card="VOLLEY"), recording(seed="NNNNNNNNNN", card="VOLLEY", played=True)
        self.corpus.add_files([self.write("jobs/old.out.json", [old]), self.write("jobs/new.out.json", [new])],
                              rand_cards=rand, jobs=self.jobs)
        flags = {m["seed"]: m["flags"] for m in self.fights()}
        self.assertEqual(flags, {"AAAAAAAAAA": ["driver-artifact-risk"], "NNNNNNNNNN": []})

    def test_recorded_game_build_wins_over_the_known_one(self):
        h = "0" * 64
        self.assertEqual(cp.build_of({"sts2_sha256": h}, {}), h)
        self.assertEqual(cp.build_of({}, {"game_sha256": h}), h)
        self.assertEqual(cp.build_of({}, {}), cp.KNOWN_BUILD)

    def test_scan_skips_unchanged_files_and_retries_unreadable_ones(self):
        (self.jobs / "a.out.json").write_text(json.dumps({"results": [recording()]}), encoding="utf-8")
        (self.jobs / "b.out.json").write_text('{"results": [', encoding="utf-8")  # still being written
        first = self.corpus.scan(self.jobs, quiet=True)
        self.assertEqual((first["new"], first["files_unreadable"]), (1, 1))
        (self.jobs / "b.out.json").write_text(json.dumps({"results": [recording(seed="BBBBBBBBBB")]}), encoding="utf-8")
        second = self.corpus.scan(self.jobs, quiet=True)
        self.assertEqual((second["files_unchanged"], second["new"], second["fights"]), (1, 1, 2))

    def test_scan_reads_every_folder_by_its_policy(self):
        seeds = {}
        for i, rel in enumerate(["jobs/a.out.json", "jobs/frames/b.out.json", "jobs/frames/s0_frames.out.json",
                                 "jobs-trace/w0/c.out.json", "jobs-nodeless/d.out.json", "jobs-nodeless/fpm_full3s0.out.json",
                                 "jobs-new/e.out.json", "jobs/other/f.out.json"]):
            seeds[rel] = f"{i}" * 10
            self.write(rel, [recording(seed=seeds[rel])])
        self.write("jobs/frames/both.out.json", [recording(seed=seeds["jobs/a.out.json"])])  # also in a stock batch
        got = self.corpus.scan(self.jobs, quiet=True)
        self.assertEqual((got["files_excluded"], got["files_unreviewed"], got["new"]), (2, 2, 6))
        by_seed = {m["seed"]: m for m in self.fights()}
        self.assertEqual(sorted(by_seed), sorted(seeds[r] for r in seeds if "s0" not in r))
        tags = {r: by_seed[s]["tags"] for r, s in seeds.items() if s in by_seed}
        self.assertEqual(tags, {"jobs/a.out.json": [], "jobs/frames/b.out.json": ["settle-trial"],
                                "jobs-trace/w0/c.out.json": [], "jobs-nodeless/d.out.json": ["nodeless-trial"],
                                "jobs-new/e.out.json": ["unreviewed"],
                                "jobs/other/f.out.json": ["unreviewed"]})
        self.assertEqual(sorted(by_seed[seeds["jobs/a.out.json"]]["sources"]), ["jobs/a.out.json", "jobs/frames/both.out.json"])
        excluded = self.corpus.load_index()["excluded"]
        self.assertEqual(sorted(excluded), ["jobs-nodeless/fpm_full3s0.out.json", "jobs/frames/s0_frames.out.json"])
        self.assertIn("no frame sleep", excluded["jobs/frames/s0_frames.out.json"]["reason"])

    def test_a_fight_the_game_broke_off_is_kept_up_to_its_last_good_step(self):
        self.assertEqual(cp.noop_steps(broken_off()["steps"]), [3, 4])
        self.add(self.write("jobs/n.out.json", [broken_off()]))
        (m,) = self.fights()
        rec = self.corpus.read(m["id"])
        self.assertEqual(len(rec["steps"]), 2)  # the start and the play; the end of turn before the parked plays goes
        self.assertEqual(rec["choices"], [{"kind": "cards", "step": 1, "picked": [0]}])
        self.assertNotIn("error", rec)
        self.assertEqual(rec["game_error"], {"error": 'idle_timeout after step 4: {"kind":"play"}', "recorded_steps": 5, "kept": 2})
        self.assertEqual((m["flags"], m["cut"]["kept"]), (["truncated"], 2))
        # Broken off before any good step: nothing to keep.
        early = dict(broken_off(seed="EEEEEEEEEE"), steps=broken_off()["steps"][:1])
        self.assertEqual(cp.prepare(early), (None, "error_before_a_good_step"))

    def test_a_broken_off_prefix_of_a_complete_fight_is_only_its_source(self):
        full = recording(hps=(27, 21, 15), played=True, choices=[{"kind": "cards", "step": 1, "picked": [0]}])
        got = self.add(self.write("jobs/n.out.json", [broken_off()]), self.write("jobs/full.out.json", [full]))
        self.assertEqual((got["new"], got["truncated_known"], got["fights"]), (1, 1, 1))
        self.assertEqual(self.fights()[0]["sources"], ["jobs/full.out.json", "jobs/n.out.json"])
        # A complete fight whose host logged an error after it is the same fight, still not truncated.
        logged = dict(full, error="nodeless: 1 game error(s); first: log: x")
        got = self.add(self.write("jobs/logged.out.json", [logged]))
        self.assertEqual((got["known"], self.fights()[0]["flags"]), (1, []))

    def test_a_fight_that_needs_an_emudump_feature_waits_for_it(self):
        long_fight = dict(recording(), enemy_hp_mult=5)
        self.assertEqual(cp.needs_of(long_fight), ["enemy_hp_mult"])
        self.assertEqual(cp.needs_of(dict(recording(), setup=[{"cmd": "fight"}], use_potions=True)), ["setup"])
        metas = [{"id": "a", "needs": []}, {"id": "b", "needs": ["enemy_hp_mult"]}]
        self.assertEqual(cp.replayable(metas, set()), ([metas[0]], {"enemy_hp_mult": 1}))
        self.assertEqual(cp.replayable(metas, {"enemy_hp_mult"}), (metas, {}))
        fake = self.dir / "fork"
        (fake / "tools" / "emudump").mkdir(parents=True)
        (fake / "tools" / "emudump" / "Fight.cs").write_text('int k = (int?)game["enemy_hp_mult"] ?? 1;', encoding="utf-8")
        self.assertEqual(cp.emudump_features(fake), {"enemy_hp_mult"})

    def test_a_format_1_index_is_migrated_and_rescanned(self):
        self.add(self.write("jobs/a.out.json", [recording()]))
        idx = self.corpus.load_index()
        fid = idx["fights"][0]["id"]
        old_meta = {k: v for k, v in idx["fights"][0].items() if k not in ("spec", "tags", "needs")}
        self.corpus.save_index({"format": 1, "fights": [dict(old_meta, sources=["a.out.json"])],
                                "scanned": {"a.out.json": [1, 2]}})
        got = self.corpus.scan(self.jobs, quiet=True)
        self.assertEqual((got["known"], got.get("new", 0)), (1, 0))
        (m,) = self.fights()
        self.assertEqual((m["id"], m["sources"], m["tags"], m["needs"]), (fid, ["jobs/a.out.json"], [], []))


class JudgeTest(unittest.TestCase):
    def pair(self, game_hps, emu_hps):
        g = recording(hps=game_hps)
        e = {"steps": [{"action": s["action"], "state": emu_state(h)} for s, h in zip(g["steps"], emu_hps)]}
        return g, e

    def test_identical_and_first_divergence(self):
        self.assertEqual(cp.judge_fight(*self.pair((27, 21, 15), (27, 21, 15)), NM), {"fd": None})
        got = cp.judge_fight(*self.pair((27, 21, 15), (27, 21, 14)), NM)
        self.assertEqual((got["fd"], got["field"], got["card"]), (2, ".enemies", STRIKE))

    def test_an_emulator_that_stops_early_diverges_where_it_stopped(self):
        g, e = self.pair((27, 21, 15), (27, 21))
        self.assertEqual(cp.judge_fight(g, e, NM)["field"], "(length)")
        e["error"] = "System.Exception: boom\n   at somewhere"
        got = cp.judge_fight(g, e, NM)
        self.assertEqual((got["fd"], got["field"], got["ferr"]), (2, "(error)", "System.Exception: boom"))

    def test_an_exception_after_the_last_matching_step_is_kept(self):
        g, e = self.pair((27, 21, 15), (27, 21, 15))
        e["error"] = "System.Exception: after the end"
        self.assertEqual(cp.judge_fight(g, e, NM), {"fd": None, "ferr": "System.Exception: after the end"})

    def test_reanchored_verdicts_one_letter_per_transition_with_every_field(self):
        g, e = self.pair((27, 21, 15, 9), (27, 21, 14, 9))
        e["steps"][2]["state"] = emu_state(14, php=69)
        got = cp.judge_reanchor(g, e, NM, None)
        self.assertEqual(got["rv"], "MDM")
        self.assertEqual(got["dp"], {"2": [".enemies", ".player.hp"]})
        e["error"] = "System.Exception: boom"
        self.assertEqual(cp.judge_reanchor(g, e, NM, None)["rerr"], "System.Exception: boom")

    def test_balance_splits_by_steps(self):
        metas = [{"id": str(i), "steps": s} for i, s in enumerate([40, 30, 20, 10, 10, 10])]
        parts = cp.balance(metas, 2)
        loads = sorted(sum(dict((m["id"], m["steps"]) for m in metas)[i] for i in p) for p in parts)
        self.assertEqual(loads, [60, 60])


class ReplayTest(CorpusTestCase):
    def test_a_hung_emudump_is_killed_and_its_fights_fail(self):
        self.add(self.write("jobs/a.out.json", [recording()]))
        fid = self.fights()[0]["id"]
        work = self.dir / "work"
        work.mkdir()
        with mock.patch.object(cp.subprocess, "run", side_effect=subprocess.TimeoutExpired("dotnet", 7)) as run:
            got = cp.replay_part("fight", [fid], str(self.corpus.path), "EmuDump.dll", str(work), timeout=7)
        self.assertEqual(run.call_args.kwargs["timeout"], 7)
        self.assertIn("timed out after 7 s", got["problem"])
        self.assertEqual((got["fights"][fid]["fd"], got["fights"][fid]["ferr"][:23]), (0, "EmuDump fight timed out"))
        self.assertEqual(list(work.iterdir()), [])
        self.assertGreater(cp.part_timeout(2000), cp.part_timeout(20))

    def test_the_run_names_uncommitted_judge_changes(self):
        with mock.patch.object(cp, "git", return_value="") as git:
            info = cp.emulator_id(self.dir / "none.dll")
        status = next(c.args for c in git.call_args_list if c.args[0] == "status")
        self.assertTrue({"src", "tools/emudump", "tools/fidelity", "data/id_map.json"} <= set(status))
        self.assertEqual(len(info["judge_sha256"]), 64)


class GateTest(CorpusTestCase):
    BASE = {"f1": {"n": 5, "fd": None, "rv": "MMMM"},                          # identical
            "f2": {"n": 5, "fd": 3, "rv": "MMDM", "dp": {"3": [".enemies"]}},  # diverged at step 3
            "f3": {"n": 4, "fd": 1, "rv": "DDD"}}

    def test_regressions_and_improvements(self):
        cur = {"f1": {"n": 5, "fd": 4, "rv": "MMMD"},       # identical before: regression (plain and step 4)
               "f2": {"n": 5, "fd": None, "rv": "MMMM"},    # improvement
               "f3": {"n": 4, "fd": 1, "rv": "DDD"},        # unchanged
               "f4": {"n": 3, "fd": None, "rv": "MM"}}      # new, no baseline
        res = cp.compare(self.BASE, cur)
        self.assertEqual([(r["id"], r["plain"], r["steps"]) for r in res["regressions"]], [("f1", [5, 4], [4])])
        self.assertEqual([(r["id"], r["plain"], r["steps"]) for r in res["improvements"]], [("f2", [3, 5], [3])])
        self.assertEqual((res["new"], res["missing"]), (["f4"], []))

    def test_a_step_that_stops_matching_is_a_regression_even_if_the_plain_replay_holds(self):
        cur = dict(self.BASE, f2={"n": 5, "fd": 3, "rv": "MDDM", "dp": {"3": [".enemies"]}})
        self.assertEqual([(r["id"], r["plain"], r["steps"]) for r in cp.compare(self.BASE, cur)["regressions"]],
                         [("f2", None, [2])])
        cur = dict(self.BASE, f2={"n": 5, "fd": 3, "rv": "MM--"})  # no longer re-anchored at all
        self.assertEqual(cp.compare(self.BASE, cur)["regressions"][0]["steps"], [4])

    def test_new_exceptions_and_new_fields_at_a_diverging_step_are_regressions(self):
        throws = dict(self.BASE, f2={"n": 5, "fd": 3, "rv": "MMEM", "dp": {"3": ["(error)"]}})
        more = dict(self.BASE, f2={"n": 5, "fd": 3, "rv": "MMDM", "dp": {"3": [".enemies", ".player.hp"]}})
        late = dict(self.BASE, f1={"n": 5, "fd": None, "rv": "MMMM", "ferr": "System.Exception: late"})
        rerr = dict(self.BASE, f3={"n": 4, "fd": 1, "rv": "DDD", "rerr": "System.Exception: anchor"})
        got = [cp.compare(self.BASE, c)["regressions"] for c in (throws, more, late, rerr)]
        self.assertEqual([[(r["id"], r["steps"], r["fields"], r["errors"]) for r in g] for g in got],
                         [[("f2", [3], {}, [])], [("f2", [3], {3: [".player.hp"]}, [])],
                          [("f1", [], {}, [("plain replay", "System.Exception: late")])],
                          [("f3", [], {}, [("re-anchoring", "System.Exception: anchor")])]])
        # The same field as a bare string (a hand-made baseline) is no regression.
        same = dict(self.BASE, f2={"n": 5, "fd": 3, "rv": "MMDM", "dp": {"3": [".enemies"]}})
        self.assertEqual(cp.compare(dict(self.BASE, f2=dict(self.BASE["f2"], dp={"3": ".enemies"})), same)["regressions"], [])

    def run_doc(self, fights, subset=False):
        return {"format": cp.RESULTS_FORMAT, "label": "t", "fights": fights, "totals": cp.totals(fights), "subset": subset,
                "emulator": {"judge_sha256": "j"}}

    def check(self, fights, subset=False, fmt=cp.RESULTS_FORMAT) -> tuple[int, str]:
        bdir = self.corpus.path / "baselines"
        bdir.mkdir(parents=True, exist_ok=True)
        bp = bdir / "abc.json"
        bp.write_text(json.dumps({"format": fmt, "commit": "abc", "run": "r0", "judge_sha256": "j",
                                  "totals": cp.totals(self.BASE), "fights": self.BASE}), encoding="utf-8")
        out = io.StringIO()
        with redirect_stdout(out):
            code = cp.check(self.corpus, self.run_doc(fights, subset), bp)
        return code, out.getvalue()

    def test_check_exits_non_zero_on_any_regression_and_lists_it(self):
        code, text = self.check(dict(self.BASE, f1={"n": 5, "fd": 2, "field": ".player.hp", "card": "BASH", "rv": "MMMM"}))
        self.assertEqual(code, 1)
        self.assertIn("REGRESSIONS: 1 fights", text)
        self.assertIn("f1", text)
        self.assertIn("diverged at step 2 (.player.hp after BASH)", text)
        code, text = self.check(dict(self.BASE, f3={"n": 4, "fd": 2, "rv": "MDD"}))
        self.assertEqual(code, 0)
        self.assertIn("IMPROVEMENTS: 1 fights", text)
        code, text = self.check(dict(self.BASE, f2={"n": 5, "fd": 3, "rv": "MMEM"}))
        self.assertEqual(code, 1)
        self.assertIn("D>E", text)

    def test_a_full_check_fails_when_baseline_fights_are_missing(self):
        part = {k: v for k, v in self.BASE.items() if k != "f3"}
        code, text = self.check(part)
        self.assertEqual(code, 1)
        self.assertIn("NOT COMPARED: 1 baseline fights", text)
        self.assertIn("1 not compared", text)
        code, text = self.check(part, subset=True)  # --only / --limit: the rest is expected to be absent
        self.assertEqual(code, 0)

    def test_a_baseline_of_an_older_results_format_is_refused(self):
        self.assertEqual(self.check(self.BASE, fmt=1)[0], 2)

    def test_a_baseline_needs_a_full_clean_run(self):
        doc = dict(self.run_doc(self.BASE), label="r", emulator={"commit": "abc", "dirty": False, "emudump_sha256": "e"})
        for bad in ({"subset": True}, {"problems": ["EmuDump fight timed out"]}):
            with self.assertRaises(SystemExit):
                cp.store_baseline(self.corpus, dict(doc, **bad), force=False)
        self.assertEqual(cp.store_baseline(self.corpus, doc, force=False).name, "abc.json")

    def test_baseline_of_the_nearest_ancestor(self):
        bdir = self.corpus.path / "baselines"
        bdir.mkdir(parents=True)
        head = cp.git("rev-parse", "HEAD")
        parent = cp.git("rev-parse", "HEAD~1")
        (bdir / f"{parent}.json").write_text("{}", encoding="utf-8")
        self.assertEqual(cp.baseline_for(self.corpus, None).stem, parent)
        (bdir / f"{head}.json").write_text("{}", encoding="utf-8")
        self.assertEqual(cp.baseline_for(self.corpus, None).stem, head)
        self.assertEqual(cp.baseline_for(self.corpus, parent[:9]).stem, parent)


if __name__ == "__main__":
    unittest.main()
