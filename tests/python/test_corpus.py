"""tools.fidelity.corpus: the recorded-fight corpus and its regression gate.

Synthetic recordings in the referee's shape. A fight recorded in several files is kept once with all
its sources; the recording format and the driver-artifact flag come from the recording; the judge
names the first divergence and every re-anchored step; and the gate fails exactly when a fight or
a step that matched in the baseline no longer does.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import corpus as cp  # noqa: E402
from tools.fidelity import fight_compare as fc  # noqa: E402

ID_MAP = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))
NM = fc.names(ID_MAP)
CID, EID = ID_MAP["cards"], ID_MAP["enemies"]
STRIKE, SLUG = "STRIKE_IRONCLAD", "CORPSE_SLUG"


def game_state(hp=27):
    return {"in_combat": True, "round": 1, "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "hand": [[STRIKE, False]], "draw": [], "discard": [], "exhaust": [],
            "enemies": [{"name": SLUG, "hp": hp, "max_hp": 27, "block": 0, "powers": []}],
            "rng": {"Shuffle": {"seed": 1, "counter": 1}}}


def emu_state(hp=27):
    card = {"def_id": CID["StrikeIronclad"], "upgraded": False}
    return {"in_combat": True, "draw_order": [], "rng": {"Shuffle": 1},
            "snapshot": {"player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3, "buffs": []},
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


class CorpusTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.corpus = cp.Corpus(self.dir / "corpus")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, results):
        p = self.dir / name
        p.write_text(json.dumps({"results": results}), encoding="utf-8")
        return p


class IngestTest(CorpusTestCase):
    def test_a_fight_in_several_files_is_kept_once_with_every_source(self):
        a, b = recording(), recording(seed="BBBBBBBBBB")
        moved = dict(a, timing_ms={"play": 1234}, _id=3, _pid=77, _t=5)  # same fight, other timing, serve tags
        n1 = self.corpus.add_files([self.write("x.s0.out.json", [a])], rand_cards=set())
        n2 = self.corpus.add_files([self.write("x.out.json", [moved, b, {"seed": "C", "error": "stalled", "steps": []}])],
                                   rand_cards=set())
        self.assertEqual((n1["new"], n2["new"], n2["known"], n2["not_clean"]), (1, 1, 1, 1))
        idx = self.corpus.load_index()
        self.assertEqual(len(idx["fights"]), 2)
        first = idx["fights"][0]
        self.assertEqual(first["sources"], ["x.s0.out.json", "x.out.json"])
        self.assertEqual((first["steps"], first["plays"], first["build"]), (3, 2, cp.KNOWN_BUILD))
        stored = self.corpus.read(first["id"])
        self.assertNotIn("timing_ms", stored)
        self.assertNotIn("phase_ms", stored)
        self.assertEqual(stored["steps"], a["steps"])

    def test_other_steps_or_choices_are_another_fight(self):
        a = recording()
        self.assertNotEqual(cp.fight_id(a), cp.fight_id(recording(hps=(27, 21, 14))))
        self.assertNotEqual(cp.fight_id(a), cp.fight_id(recording(choices=[{"kind": "cards", "step": 1, "picked": [0]}])))
        self.assertNotEqual(cp.fight_id(a), cp.fight_id(dict(a, deck_add=["BASH"])))
        self.assertEqual(cp.fight_id(a), cp.fight_id(dict(a, cost={"cpu_ms": 1})))

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
        self.corpus.add_files([self.write("old.out.json", [old]), self.write("new.out.json", [new])], rand_cards=rand)
        flags = {m["seed"]: m["flags"] for m in self.corpus.load_index()["fights"]}
        self.assertEqual(flags, {"AAAAAAAAAA": ["driver-artifact-risk"], "NNNNNNNNNN": []})

    def test_recorded_game_build_wins_over_the_known_one(self):
        h = "0" * 64
        self.assertEqual(cp.build_of({"sts2_sha256": h}, {}), h)
        self.assertEqual(cp.build_of({}, {"game_sha256": h}), h)
        self.assertEqual(cp.build_of({}, {}), cp.KNOWN_BUILD)

    def test_scan_skips_unchanged_files_and_retries_unreadable_ones(self):
        jobs = self.dir / "jobs"
        jobs.mkdir()
        (jobs / "a.out.json").write_text(json.dumps({"results": [recording()]}), encoding="utf-8")
        (jobs / "b.out.json").write_text('{"results": [', encoding="utf-8")  # still being written
        first = self.corpus.scan(jobs, quiet=True)
        self.assertEqual((first["new"], first["files_unreadable"]), (1, 1))
        (jobs / "b.out.json").write_text(json.dumps({"results": [recording(seed="BBBBBBBBBB")]}), encoding="utf-8")
        second = self.corpus.scan(jobs, quiet=True)
        self.assertEqual((second["files_unchanged"], second["new"], second["fights"]), (1, 1, 2))


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
        self.assertEqual((got["fd"], got["field"], got["error"]), (2, "(error)", "System.Exception: boom"))

    def test_reanchored_verdicts_one_letter_per_transition(self):
        g, e = self.pair((27, 21, 15, 9), (27, 21, 14, 9))
        got = cp.judge_reanchor(g, e, NM, None)
        self.assertEqual(got["rv"], "MDM")
        self.assertEqual(got["dp"], {"2": ".enemies"})

    def test_balance_splits_by_steps(self):
        metas = [{"id": str(i), "steps": s} for i, s in enumerate([40, 30, 20, 10, 10, 10])]
        parts = cp.balance(metas, 2)
        loads = sorted(sum(dict((m["id"], m["steps"]) for m in metas)[i] for i in p) for p in parts)
        self.assertEqual(loads, [60, 60])


class GateTest(CorpusTestCase):
    BASE = {"f1": {"n": 5, "fd": None, "rv": "MMMM"},       # identical
            "f2": {"n": 5, "fd": 3, "rv": "MMDM"},          # diverged at step 3
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
        cur = dict(self.BASE, f2={"n": 5, "fd": 3, "rv": "MDDM"})
        self.assertEqual([(r["id"], r["plain"], r["steps"]) for r in cp.compare(self.BASE, cur)["regressions"]],
                         [("f2", None, [2])])
        cur = dict(self.BASE, f2={"n": 5, "fd": 3, "rv": "MM--"})  # no longer re-anchored at all
        self.assertEqual(cp.compare(self.BASE, cur)["regressions"][0]["steps"], [4])

    def run_doc(self, fights):
        return {"label": "t", "fights": fights, "totals": cp.totals(fights)}

    def check(self, fights) -> tuple[int, str]:
        bdir = self.corpus.path / "baselines"
        bdir.mkdir(parents=True, exist_ok=True)
        bp = bdir / "abc.json"
        bp.write_text(json.dumps({"commit": "abc", "run": "r0", "totals": cp.totals(self.BASE), "fights": self.BASE}),
                      encoding="utf-8")
        out = io.StringIO()
        with redirect_stdout(out):
            code = cp.check(self.corpus, self.run_doc(fights), bp)
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
