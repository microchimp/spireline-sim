"""The Act 3 stratum of the fight exam and the practice (rl/act3_starts.py, rl/act3_exam.py, fight_exam.add-act3, practice_mix's act3 share;
the Act 3 week, lane 5: the Test Subject).

The pinned facts: a human Act 3 boss fight is rebuilt as a start with its turn (first or second boss) and the human's own result (the second
boss is survived only by winning the run); the exam fifth is never read; no human run's id reaches a file written here (the run is a salted
hash); the human bar by boss is aggregates only (a cell under 20 runs is not shown); the exam picks its Act 3 cells by quota, clean runs first,
one start a run, writes a third file and leaves the AI exam and the Acts 1 and 2 human stratum alone; the practice reads the Act 3 sets only when
its share is above zero, and with the share at zero draws exactly as before; the traces say how long the Test Subject's forms last and what killed.
A synthetic run record stands in for the corpus; the real corpus is not read here.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import act3_exam as ax  # noqa: E402
import act3_starts as a3  # noqa: E402
import fight_exam as fe  # noqa: E402
import gym_loop as gl  # noqa: E402
import human_starts as hs  # noqa: E402
import practice_mix as pm  # noqa: E402
from test_human_starts import _deal_in_child, _find_ids, _fight, _point, _stats, synthetic_run  # noqa: E402


def stats3(cur, dmg=0, heal=0, max_hp=80, **extra):
    return {"current_hp": cur, "max_hp": max_hp, "current_gold": 99, "damage_taken": dmg, "hp_healed": heal, "player_id": 1, **extra}


MONSTERS = {"TEST_SUBJECT_BOSS": ["MONSTER.TEST_SUBJECT"], "QUEEN_BOSS": ["MONSTER.QUEEN", "MONSTER.TORCH_HEAD_AMALGAM"],
            "AEONGLASS_BOSS": ["MONSTER.AEONGLASS"]}


def act3_run(rid: str, *, first="TEST_SUBJECT_BOSS", second="QUEEN_BOSS", end="win", hp1=70, hp2=40) -> dict:
    """An Ironclad A10 run to Act 3: Act 1 as `synthetic_run`, a 16-floor Act 2 and an Act 3 whose two bosses are floors 48 and 49. `end`:
    "win" (both bosses die), "second" (dies at the second), "first" (dies at the first), "none" (stops before the first)."""
    run = synthetic_run(rid, act2=False)
    a2 = [_point("ancient", _stats(80, 80, 150), [{"model_id": "EVENT.OROBAS", "room_type": "event"}])]
    a2 += [_point("rest_site", _stats(80, 80, 150)) for _ in range(14)]
    a2 += [_fight("boss", "ENCOUNTER.KAISER_CRAB_BOSS", ["MONSTER.KAISER_CRAB"], _stats(60, 80, 200))]
    a3_ = [_point("rest_site", _stats(hp1, 80, 220)) for _ in range(14)]
    if end != "none":
        a3_.append(_fight("boss", f"ENCOUNTER.{first}", MONSTERS[first], stats3(0 if end == "first" else hp2, 60, 6)))
        a3_[-1]["rooms"][-1]["turns_taken"] = 9
    if end in ("win", "second"):
        a3_.append(_fight("boss", f"ENCOUNTER.{second}", MONSTERS[second], stats3(30 if end == "win" else 0, 40, 6)))
        a3_[-1]["rooms"][-1]["turns_taken"] = 7
    run["map_point_history"] = [run["map_point_history"][0], a2, a3_]
    run["acts"] = ["ACT.OVERGROWTH", "ACT.HIVE", "ACT.GLORY"]
    run["win"] = end == "win"
    return run


class RebuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = hs.Tables.load()

    def specs(self, run, fail=None, **kw):
        return hs.rebuild_run(run, self.tables, fail if fail is not None else Counter(), None, acts=(3,), kinds=("boss",), **kw)

    def test_the_two_act_three_bosses_are_starts_with_their_turn_floor_and_the_hp_the_human_walked_in_with(self):
        rid = _find_ids(False, 1, "k")[0]
        specs = self.specs(act3_run(rid))
        self.assertEqual([(s["boss_no"], s["floor"], s["encounter_id"], s["enemy"], s["cat"], s["act"]) for s in specs],
                         [(1, 48, 80, "TEST_SUBJECT_BOSS", "act3-boss", 3), (2, 49, 78, "QUEEN_BOSS", "act3-boss", 3)])
        self.assertEqual((specs[0]["hp"], specs[0]["max_hp"]), (70, 80))  # the floor before: the last rest of Act 3
        self.assertEqual((specs[1]["hp"], specs[1]["max_hp"]), (40, 80))  # the first boss's floor, after its post-fight heal
        self.assertEqual(specs[0]["node_type"], 6)
        self.assertEqual(len(specs[0]["deck"]), len(specs[1]["deck"]))

    def test_acts_one_and_two_are_not_built_and_not_counted_when_act_three_is_asked_and_act_three_when_it_is_not(self):
        rid = _find_ids(False, 1, "k")[0]
        run = act3_run(rid)
        fail, seen = Counter(), Counter()
        hs.rebuild_run(run, self.tables, fail, seen, acts=(3,), kinds=("boss",))
        self.assertEqual(dict(fail), {})
        self.assertEqual(seen["act3-boss"], 2)
        fail = Counter()
        specs = hs.rebuild_run(run, self.tables, fail, None)  # the default: Acts 1 and 2, as before
        self.assertEqual(sorted({s["act"] for s in specs}), [1, 2])
        self.assertEqual(fail["act 3 or later: not built"], 2)
        self.assertTrue(all("boss_no" not in s for s in specs))

    def test_the_first_boss_is_survived_by_going_on_and_the_second_only_by_winning(self):
        rid = _find_ids(False, 1, "k")[0]
        for end, want in (("win", [True, True]), ("second", [True, False]), ("first", [False])):
            specs = self.specs(act3_run(rid, end=end))
            self.assertEqual([s["outcome"]["human_survived"] for s in specs], want, end)
        dead = self.specs(act3_run(rid, end="first"))[0]["outcome"]
        self.assertEqual((dead["human_end_hp"], dead["human_hp_lost"], dead["human_l0"]), (0, 70, hs.DEATH_VALUE))
        won = self.specs(act3_run(rid, end="win"))[1]["outcome"]
        self.assertEqual((won["human_survived"], won["human_turns"]), (True, 7))

    def test_the_outcomes_pass_finds_the_same_floors_and_results(self):
        rid = _find_ids(False, 1, "k")[0]
        for end in ("win", "second", "first"):
            run = act3_run(rid, end=end)
            specs = {s["id"]: s for s in self.specs(run)}
            outs, why, _checks = hs.outcomes_of_run(run, acts=(3,))
            self.assertEqual(sorted(outs), sorted(specs), end)
            self.assertEqual(why, {})
            for sid, s in specs.items():
                self.assertEqual(s["outcome"], outs[sid], end)

    def test_a_boss_on_another_floor_is_refused_by_reason_and_no_run_is_built_without_its_bosses(self):
        rid = _find_ids(False, 1, "k")[0]
        run = act3_run(rid)
        run["map_point_history"][1].append(_point("rest_site", _stats(80, 80, 150)))  # an Act 2 of 17 floors: the bosses are floors 49 and 50
        fail = Counter()
        self.assertEqual(self.specs(run, fail), [])
        self.assertEqual(fail["boss at an unusual floor: the route is not the copy's"], 2)
        self.assertEqual(self.specs(act3_run(rid, end="none")), [])

    def test_the_boss_floors_of_the_acts(self):
        self.assertEqual(hs.BOSS_FLOORS, {1: (17,), 2: (33,), 3: (48, 49)})
        self.assertEqual(hs.BOSS_FLOOR, {1: 17, 2: 33})  # unchanged


class OpeningTests(unittest.TestCase):
    """The copy deals the Test Subject's opening: the real game's state at floor 48, no game played."""

    @classmethod
    def setUpClass(cls):
        cls.tables = hs.Tables.load()
        cls.rid = _find_ids(False, 1, "k")[0]
        cls.specs = hs.rebuild_run(act3_run(cls.rid), cls.tables, Counter(), None, acts=(3,), kinds=("boss",))
        old = os.environ.get("HUMAN_STARTS_ENGINE")
        os.environ["HUMAN_STARTS_ENGINE"] = "old"
        try:
            ctx = mp.get_context("spawn")
            queue = ctx.Queue()
            p = ctx.Process(target=_deal_in_child, args=(queue, cls.specs), daemon=True)
            p.start()
            kind, got = queue.get(timeout=180)
            p.join(30)
        finally:
            if old is None:
                os.environ.pop("HUMAN_STARTS_ENGINE", None)
            else:
                os.environ["HUMAN_STARTS_ENGINE"] = old
        if kind != "ok":
            raise unittest.SkipTest(f"native library unavailable: {got}")
        cls.docs = {s: (d, why) for s, d, why in got}

    def test_the_test_subject_stands_in_its_first_form_at_floor_48(self):
        spec = self.specs[0]
        doc, why = self.docs[spec["id"]]
        self.assertEqual(why, "")
        self.assertEqual((doc["player"]["hp"], doc["player"]["max_hp"], doc["turn"], doc["ascension"], doc["encounter_id"]), (70, 80, 0, 10, 80))
        self.assertEqual([(e["def_id"], e["hp"], e["max_hp"]) for e in doc["enemies"]], [(10007, 111, 111)])

    def test_the_second_boss_opens_too_and_a_record_is_in_the_gyms_schema(self):
        spec = self.specs[1]
        doc, why = self.docs[spec["id"]]
        self.assertEqual((why, doc["encounter_id"], doc["player"]["hp"]), ("", 78, 40))
        rec = hs.start_record(a3.anonymise(dict(spec, run=self.rid)), doc)
        self.assertEqual((rec["cat"], rec["act"], rec["floor"], rec["kind"], rec["node_type"], rec["source"]), ("act3-boss", 3, 49, "boss", 6, "human"))
        self.assertNotIn(self.rid, json.dumps(rec))


class AnonymiseAndScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hs.Tables.load()

    def test_a_run_is_a_salted_hash_and_the_id_never_reaches_a_start(self):
        rid = _find_ids(False, 1, "k")[0]
        spec = a3.anonymise({"id": f"human:{rid}|f48", "run": rid, "seed": rid, "floor": 48})
        self.assertEqual(spec["run"], a3.run_tag(rid))
        self.assertEqual(spec["id"], f"human3:{a3.run_tag(rid)}|f48")
        self.assertEqual((len(spec["run"]), spec["seed"]), (12, spec["run"]))
        self.assertNotIn(rid, json.dumps(spec))
        self.assertEqual(pm.run_of(spec["id"]), spec["run"])  # the practice's exclusion reads the run from the id
        self.assertNotEqual(a3.run_tag(rid), a3.run_tag(rid + "x"))

    def _corpus(self, tmp: Path, n_ok: int, exam_n: int = 1, played_idx=()):
        ok = _find_ids(False, n_ok, "k")
        exam = _find_ids(True, exam_n, "h")
        listed = _find_ids(False, 1, "l")[0]
        runs = []
        for i, r in enumerate(ok):
            runs.append(act3_run(r, first=("TEST_SUBJECT_BOSS", "QUEEN_BOSS", "AEONGLASS_BOSS")[i % 3], second="AEONGLASS_BOSS",
                                 end=("win", "second", "first")[i % 3]))
        runs += [act3_run(e) for e in exam] + [act3_run(listed)]
        p = tmp / "corpus.jsonl.gz"
        with gzip.open(p, "wt", encoding="utf-8") as fh:
            for r in runs:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        (tmp / "ids.txt").write_text(listed + "\n", encoding="utf-8")
        return p, ok, exam, listed

    def test_the_scan_leaves_out_the_exam_fifth_flags_the_loops_runs_names_no_run_and_keeps_the_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            p, ok, exam, listed = self._corpus(t, 6)
            ids = hs.read_exam_ids(t / "ids.txt")
            res = a3.scan(p, ids, 0, played={ok[0]}, cache=t / "cache.json")
            self.assertEqual(res["stats"]["runs_usable"], 6)
            self.assertEqual(res["stats"]["exam_fifth_left_out"], 2)
            self.assertEqual(len(res["specs"]), 10)  # two bosses for the four runs that beat the first, one for the two that died at it
            text = json.dumps(res)
            for rid in ok + exam + [listed]:
                self.assertNotIn(rid, text)
            by_run = Counter(s["run"] for s in res["specs"])
            self.assertEqual(sorted(by_run.values()), [1, 1, 2, 2, 2, 2])
            self.assertEqual(sum(1 for s in res["specs"] if s["loop_played"]), by_run[a3.run_tag(ok[0])])
            self.assertTrue(all(s["id"].startswith("human3:") and s["boss_no"] in (1, 2) for s in res["specs"]))
            self.assertEqual(res["raw"]["boss1|test-subject"]["n"], 2)
            self.assertEqual(res["raw"]["boss1|queen"]["n"], 2)
            self.assertEqual(res["raw"]["boss1|aeonglass"]["n"], 2)
            self.assertEqual(res["raw"]["boss2|aeonglass"]["n"], 4)
            # the cache is read when it is there: the corpus is not needed
            p.unlink()
            again = a3.scan(p, ids, 0, cache=t / "cache.json")
            self.assertEqual(len(again["specs"]), len(res["specs"]))

    def test_the_raw_arrivals_tell_the_boss_turn_the_result_and_the_health_in(self):
        rid = _find_ids(False, 1, "k")[0]
        got = a3._raw_arrivals(act3_run(rid, end="second"))  # noqa: SLF001
        self.assertEqual([(x["boss_no"], x["floor"], x["past"], x["hp_in"], x["turns"]) for x in got], [(1, 48, True, 70, 9), (2, 49, False, 40, 7)])
        got = a3._raw_arrivals(act3_run(rid, end="win"))  # noqa: SLF001
        self.assertEqual([x["past"] for x in got], [True, True])
        got = a3._raw_arrivals(act3_run(rid, end="first"))  # noqa: SLF001
        self.assertEqual([(x["boss_no"], x["past"]) for x in got], [(1, False)])


class HumanBarTests(unittest.TestCase):
    def raw(self, n, past):
        agg: dict = {}
        for i in range(n):
            a3._bump(agg, "boss1|test-subject", {"past": i < past, "hp_in": 60, "max_hp": 80, "damage": 50, "turns": 8, "potions_used": 1})  # noqa: SLF001
        return agg

    def test_the_bar_is_the_share_that_went_past_with_its_interval_and_a_small_cell_is_not_shown(self):
        rows = a3.human_bar_rows(self.raw(1000, 700))
        r = rows[0]
        self.assertEqual((r["turn"], r["boss"], r["n"], r["past"]), ("boss1", "test-subject", 1000, 700))
        self.assertAlmostEqual(r["rate"], 0.7)
        self.assertLess(r["lo"], 0.7)
        self.assertGreater(r["hi"], 0.7)
        self.assertAlmostEqual(r["hp_frac"], 0.75)
        self.assertEqual((r["damage"], r["turns"], r["potions_used"]), (50.0, 8.0, 1.0))
        small = a3.human_bar_rows(self.raw(19, 10))[0]
        self.assertEqual(small, {"turn": "boss1", "boss": "test-subject", "n": 19, "suppressed": True})
        self.assertIn("under 20", a3._tables_text([small], {}, {}))  # noqa: SLF001

    def test_the_spec_aggregates_leave_out_a_cell_under_twenty_and_split_by_the_humans_result(self):
        def spec(past, no=1):
            return {"boss_no": no, "encounter_id": 80, "outcome": {"human_survived": past}, "deck": [1] * 30, "relics": [1] * 18, "potions": [1],
                    "hp": 60, "max_hp": 80}
        agg = a3.spec_aggregates([spec(True)] * 25 + [spec(False)] * 19 + [spec(True, 2)] * 3)
        self.assertEqual(sorted(agg), ["boss1|test-subject|past"])
        self.assertEqual(agg["boss1|test-subject|past"]["n"], 25)
        self.assertAlmostEqual(agg["boss1|test-subject|past"]["hp_frac"], 0.75)


class PickTests(unittest.TestCase):
    def recs(self):
        out = []
        for i in range(40):
            out.append({"id": f"human3:r{i:02d}|f48", "run": f"r{i:02d}", "boss_no": 1, "encounter_id": 80, "loop_played": i % 2 == 0})
        for i in range(30):
            out.append({"id": f"human3:r{i + 20:02d}|f49", "run": f"r{i + 20:02d}", "boss_no": 2, "encounter_id": 80, "loop_played": False})
        return out

    def test_a_cell_takes_its_quota_one_start_a_run_across_cells_and_a_hash_picks(self):
        recs = self.recs()
        quota = {"act3-boss1-80": 10, "act3-boss2-80": 10, "act3-boss1-78": 5}
        a, rep = a3.pick_quota(recs, quota, seed=1)
        b, _ = a3.pick_quota(list(reversed(recs)), quota, seed=1)
        self.assertEqual(sorted(s["id"] for s in a), sorted(s["id"] for s in b))  # the order of the pool does not matter
        self.assertEqual(len({s["run"] for s in a}), len(a))  # one start a run, across both cells
        self.assertEqual(Counter(s["cell"] for s in a), Counter({"act3-boss1-80": 10, "act3-boss2-80": 10}))
        self.assertEqual((rep["act3-boss1-78"]["chosen"], rep["act3-boss1-78"]["short"]), (0, True))
        self.assertEqual(rep["act3-boss1-80"]["pool"], 40)
        c, _ = a3.pick_quota(recs, quota, seed=2)
        self.assertNotEqual(sorted(s["id"] for s in a), sorted(s["id"] for s in c))

    def test_the_preferred_come_first_and_the_others_fill_what_is_short(self):
        recs = self.recs()
        chosen, rep = a3.pick_quota(recs, {"act3-boss1-80": 30}, seed=3, prefer=lambda r: not r["loop_played"])
        self.assertEqual(len(chosen), 30)
        self.assertEqual(sum(1 for s in chosen if s["clean"]), 20)  # 20 of the 40 are clean: all of them, then ten played ones
        self.assertEqual((rep["act3-boss1-80"]["clean_pool"], rep["act3-boss1-80"]["clean"]), (20, 20))
        self.assertTrue(all(not s["loop_played"] for s in chosen if s["clean"]))
        chosen, _ = a3.pick_quota(recs, {"act3-boss1-80": 8}, seed=3, prefer=lambda r: not r["loop_played"])
        self.assertTrue(all(s["clean"] for s in chosen))

    def test_a_skipped_record_is_counted_by_reason(self):
        recs = self.recs()
        _chosen, rep = a3.pick_quota(recs, {"act3-boss1-80": 5}, skip=lambda r: "no outcome" if r["run"] < "r10" else None)
        self.assertEqual(rep["_left_out"], {"no outcome": 10})

    def test_the_quotas_follow_where_the_runs_die_the_test_subject_first(self):
        hq, oq = a3.HUMAN_QUOTA, a3.OWN_QUOTA
        for q in (hq, oq):
            self.assertEqual(sorted(q), sorted(["act3-boss1-80", "act3-boss2-80", "act3-boss1-78", "act3-boss1-73", "act3-boss2-78", "act3-boss2-73"]))
            self.assertGreaterEqual(min(q.values()), a3.MIN_CELL)
            ts = q["act3-boss1-80"] + q["act3-boss2-80"]
            self.assertGreater(ts, 0.4 * sum(q.values()))
            self.assertEqual(max(q, key=q.get), "act3-boss1-80")


def fake_pos(i: int, boss=80, ordinal=1, gen=18) -> dict:
    doc = {"schema": "sts2-emu-snapshot/1", "turn": 0, "encounter_id": boss, "potions": {"max_slots": 2, "slots": [5, 0, 0, 0, 0, 0]},
           "piles": {"hand": [{"def_id": 1}], "draw_pile": {"top": [], "bottom": [], "unordered": [{"def_id": 2}]}}, "enemies": [{"def_id": 10007, "hp": 111}]}
    return {"id": f"{gen}:SEED{i:04d}|b{ordinal}", "gen": gen, "seed": f"SEED{i:04d}", "ordinal": ordinal, "floor": 47 + ordinal, "encounter_id": boss,
            "boss": "TEST_SUBJECT_BOSS", "hp": 60, "max_hp": 80, "relics": [1, 2, 3], "potions": [5], "run_won": False, "run_hp_after": 0,
            "counters_written": 1, "doc": doc}


def write_starts_dir(d: Path, n_human: int = 120, n_own: int = 80) -> None:
    """A starts folder as `act3_starts.py human` and `own` write it, with documents the tests do not look into."""
    human, outcomes = [], {}
    cells = [(1, 80), (1, 78), (1, 73), (2, 80), (2, 78), (2, 73)]
    for i in range(n_human):
        no, eid = cells[i % 6]
        sid = f"human3:{i:012x}|f{47 + no}"
        out = {"human_survived": i % 3 == 0, "human_end_hp": 10, "human_hp_lost": 50, "human_l0": 10.0, "human_l0_unit": 0.6, "human_turns": 8,
               "human_potions_used": 1}
        outcomes[sid] = out
        human.append({"id": sid, "cat": "act3-boss", "kind": "boss", "act": 3, "floor": 47 + no, "seed": f"{i:012x}", "run": f"{i:012x}", "source": "human",
                      "stratum": "human", "node_type": 6, "hp": 70, "max_hp": 80, "encounter_id": eid, "boss_no": no, "loop_played": i % 4 != 0,
                      "split": "train", "kind_same": True, **out, "doc": {"turn": 0, "enemies": [{"def_id": 10007}]}})
    own = []
    for i in range(n_own):
        no, eid = cells[i % 6]
        pos = fake_pos(i, eid, no)
        own.append(a3.own_record(pos))
    a3.write_set(d, a3.HUMAN_STEM, human)
    a3.write_set(d, a3.OWN_STEM, own)
    (d / a3.OUTCOME_FILE).write_text(json.dumps({"rule": "x", "outcomes": outcomes}), encoding="utf-8")


class OwnRecordTests(unittest.TestCase):
    def test_an_arrival_is_a_start_in_the_gyms_schema_and_reads_back_through_the_index(self):
        rec = a3.own_record(fake_pos(7, 80, 2, 17))
        self.assertEqual((rec["id"], rec["cat"], rec["kind"], rec["act"], rec["floor"], rec["source"], rec["stratum"], rec["boss_no"]),
                         ("own3:17:SEED0007|f49", "act3-boss", "boss", 3, 49, "own", "own", 2))
        self.assertEqual((rec["hp"], rec["max_hp"], rec["potions"], rec["relics"], rec["encounter_id"], rec["node_type"]), (60, 80, 1, 3, 80, 6))
        self.assertEqual(rec["split"], "val" if gl.is_val_seed("SEED0007") else "train")
        self.assertEqual(pm.run_of(rec["id"]), "SEED0007")
        for k in ("id", "cat", "kind", "act", "floor", "seed", "ix", "night", "k", "node_type", "hp", "max_hp", "deck_size", "potions", "relics",
                  "encounter_id", "turn", "doc", "split", "kind_same"):
            self.assertIn(k, rec)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(a3.write_set(Path(tmp), a3.OWN_STEM, [rec]), 1)
            gl._START_CACHE.clear()  # noqa: SLF001
            got = gl.load_start(str(Path(tmp) / "own-starts-act3.jsonl"), rec["id"])
            self.assertEqual(got["doc"]["encounter_id"], 80)
            lite = json.loads((Path(tmp) / "own-starts-act3.lite.json").read_text())
            self.assertEqual((len(lite), "doc" in lite[0]), (1, False))


class ExamStratumTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.sd = self.t / "starts"
        self.exam = self.t / "exam"
        self.sd.mkdir()
        self.exam.mkdir()
        write_starts_dir(self.sd)
        for f in ("starts.jsonl", "starts.index.json", "starts.lite.json", "meta.json", "baseline.jsonl", "starts-human.jsonl", "starts-human.index.json",
                  "starts-human.lite.json", "human-meta.json"):
            (self.exam / f).write_text("x:" + f, encoding="utf-8")
        (self.exam / "exclude_runs.json").write_text(json.dumps({"run_seeds": ["old-run"], "start_ids": ["old-start"], "starts": 5, "by_stratum": {"ai": 5}}),
                                                     encoding="utf-8")
        self.q = {"act3-boss1-80": 10, "act3-boss2-80": 8, "act3-boss1-78": 6, "act3-boss1-73": 6, "act3-boss2-78": 4, "act3-boss2-73": 4}

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, **kw):
        return fe.build_act3(self.exam, starts_dir=self.sd, human_quota=self.q, own_quota=self.q, **kw)

    def test_the_cells_are_named_by_the_boss_turn_and_the_boss_and_belong_to_the_act_three_kind(self):
        self.assertEqual(fe.cell_of({"kind": "boss", "act": 3, "encounter_id": 80, "boss_no": 1}), "act3-boss1-80")
        self.assertEqual(fe.cell_of({"kind": "boss", "act": 3, "encounter_id": 80, "boss_no": 2}), "act3-boss2-80")
        self.assertEqual(fe.cell_of({"kind": "boss", "act": 1, "encounter_id": 74}), "act1-boss-74")  # Acts 1 and 2: as before
        self.assertEqual(fe.cell_of({"kind": "elite", "act": 2}), "act2-elite")
        self.assertEqual(fe.kind_of_cell("act3-boss1-80"), "act3-boss")
        self.assertEqual(fe.kind_of_cell("act3-boss2-73"), "act3-boss")
        self.assertEqual(fe.kind_of_cell("act1-boss-74"), "act1-boss")
        self.assertEqual(fe.cell_label("act3-boss1-80"), "act3-boss1-80 (test-subject)")
        self.assertLess(fe.cell_sort("act2-boss-75"), fe.cell_sort("act3-boss1-80"))

    def test_the_stratum_is_a_third_file_with_the_cells_the_outcomes_and_the_halves(self):
        meta = self.build()
        p = fe.exam_paths(self.exam)
        starts = fe.load_act3_exam(self.exam)
        self.assertEqual(len(starts), meta["starts"])
        self.assertEqual(meta["by_stratum"], {"human": 38, "own": 38})
        for c, n in self.q.items():
            self.assertEqual(Counter(s["cell"] for s in starts.values() if s["stratum"] == "human")[c], n, c)
            self.assertEqual(Counter(s["cell"] for s in starts.values() if s["stratum"] == "own")[c], n, c)
        h = next(s for s in starts.values() if s["stratum"] == "human")
        self.assertIn("human_survived", h)  # the human's own result rides on the start
        self.assertEqual(h["half"], fe.half_of(h["id"]))
        self.assertEqual(h["kind_cell"], "act3-boss")
        o = next(s for s in starts.values() if s["stratum"] == "own")
        self.assertNotIn("human_survived", o)
        self.assertTrue(p["act3_starts"].is_file() and p["act3_meta"].is_file())
        got = gl.load_start(str(p["act3_starts"]), h["id"])
        self.assertEqual(got["doc"]["enemies"][0]["def_id"], 10007)  # the opening is in the file, read through its index
        self.assertEqual(meta["human_survived"], sum(1 for s in starts.values() if s["stratum"] == "human" and s.get("human_survived")))

    def test_clean_runs_come_first_and_a_short_cell_is_topped_up_with_played_runs_marked_practised(self):
        meta = self.build()
        starts = fe.load_act3_exam(self.exam)
        human = [s for s in starts.values() if s["stratum"] == "human"]
        lp = {x["id"]: x["loop_played"] for x in json.loads((self.sd / "human-starts-act3.lite.json").read_text())}
        for s in human:
            self.assertEqual(s["clean"], not lp[s["id"]])
            self.assertEqual(s["practised"], 0 if s["clean"] else 1)
        self.assertEqual(meta["practised_starts"], sum(1 for s in starts.values() if not s["clean"]))
        c = meta["by_cell"]["act3-boss1-80"]["human"]
        self.assertEqual(c["clean"], min(c["clean_pool"], c["quota"]))

    def test_one_start_a_run_the_exam_fifth_files_are_left_alone_and_the_practice_is_told_to_drop_the_runs(self):
        before = {f.name: gl.sha256_file(f) for f in self.exam.iterdir() if f.name != "exclude_runs.json"}
        meta = self.build()
        starts = fe.load_act3_exam(self.exam)
        self.assertEqual(len({s["run"] for s in starts.values()}), len(starts))
        after = {f.name: gl.sha256_file(f) for f in self.exam.iterdir() if f.name in before}
        self.assertEqual(before, after)  # the AI exam's and the Acts 1 and 2 human stratum's files are what they were
        ex = json.loads((self.exam / "exclude_runs.json").read_text())
        self.assertIn("old-run", ex["run_seeds"])
        self.assertTrue({s["run"] for s in starts.values()} <= set(ex["run_seeds"]))
        self.assertTrue({s["id"] for s in starts.values()} <= set(ex["start_ids"]))
        self.assertEqual((ex["by_stratum"]["act3-human"], ex["by_stratum"]["act3-own"], ex["starts"]), (38, 38, 5 + 76))
        self.assertTrue((self.exam / "exclude_runs.json.pre-act3.bak").is_file())
        self.assertEqual(meta["runs"], 76)
        # the practice drops them
        for s in starts.values():
            self.assertTrue(pm.excluded(s, frozenset(ex["run_seeds"]) | frozenset(ex["start_ids"])))

    def test_an_existing_stratum_is_a_fixed_ruler_and_the_pick_is_the_same_every_time(self):
        self.build()
        first = sorted(fe.load_act3_exam(self.exam))
        with self.assertRaises(FileExistsError):
            self.build()
        self.build(rebuild=True)
        self.assertEqual(sorted(fe.load_act3_exam(self.exam)), first)
        self.assertTrue(any(f.name.startswith("starts-act3.jsonl.") and f.name.endswith(".bak") for f in self.exam.iterdir()))

    def test_without_an_ai_exam_the_stratum_needs_stand_alone_and_without_starts_it_says_so(self):
        for f in self.exam.iterdir():
            f.unlink()
        with self.assertRaises(FileNotFoundError):
            self.build()
        meta = self.build(stand_alone=True)
        self.assertGreater(meta["starts"], 0)
        empty = self.t / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            fe.build_act3(self.exam, starts_dir=empty, stand_alone=True, rebuild=True)

    def test_the_ai_exams_own_reading_does_not_see_the_stratum(self):
        self.build()
        self.assertEqual(fe.load_human_exam(self.exam), {})  # the Acts 1 and 2 human stratum has no lite list here
        p = fe.exam_paths(self.exam)
        self.assertNotEqual(p["act3_starts"], p["human_starts"])
        self.assertEqual(fe.load_act3_exam(self.t / "nowhere"), {})


class PracticeStratumTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sd = self.root / "starts"
        self.sd.mkdir()
        write_starts_dir(self.sd, 60, 60)
        # a night set with two kinds, so the default draw has something to draw
        night = [{"id": f"night:S{i}|f17", "cat": "act1-boss", "kind": "boss", "act": 1, "floor": 17, "seed": f"S{i}", "run": f"S{i}", "source": "night"}
                 for i in range(20)]
        (self.sd / "starts.lite.json").write_text(json.dumps(night), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def write_mix(self, **act3):
        (self.sd / "mix.json").write_text(json.dumps({"act3": act3}), encoding="utf-8")

    def sampler(self, ex=frozenset()):
        cfg = pm.config(self.sd)
        pools, dropped = pm.load_pools(self.root, ex)
        return pm.Sampler(pools, {"act1-boss": 1.0}, cfg, {}, {}, 1), pools, dropped

    def test_it_is_off_by_default_and_reads_none_of_the_act_three_files(self):
        self.assertEqual(pm.config(self.sd)["act3"], {"share": 0.0, "human_share": 0.5, "own_encounters": [80]})
        _s, pools, dropped = self.sampler()
        self.assertNotIn(("act3-boss", "human"), pools)
        self.assertNotIn(("act3-boss", "ai"), pools)
        self.assertEqual(dict(dropped), {})

    def test_with_the_share_at_zero_the_draws_are_the_same_to_the_byte_and_the_generator_is_not_touched_for_the_stratum(self):
        a, _, _ = self.sampler()
        ra = random.Random(9)
        draws_a = [a.draw(ra) for _ in range(50)]
        self.write_mix(share=0.0, human_share=0.9)  # a mix.json that names the stratum but leaves it off
        b, _, _ = self.sampler()
        rb = random.Random(9)
        draws_b = [b.draw(rb) for _ in range(50)]
        self.assertEqual(draws_a, draws_b)
        self.assertEqual(ra.getstate(), rb.getstate())
        self.assertTrue(all(d["cat"] == "act1-boss" for d in draws_b))
        # the draw the sampler made before the stratum existed: a kind, then a side if it has both, then a start (here: one kind, one side)
        r1, r2 = random.Random(4), random.Random(4)
        b.draw(r1)
        r2.choices(b.cats, b.w)
        r2.random()
        self.assertEqual(r1.getstate(), r2.getstate())

    def test_with_a_share_the_stratum_is_drawn_that_often_human_and_own_by_their_split(self):
        self.write_mix(share=0.5, human_share=0.5, own_encounters=[])
        s, pools, _ = self.sampler()
        self.assertEqual(len(pools[("act3-boss", "human")]), 60)
        self.assertEqual(len(pools[("act3-boss", "ai")]), 60)
        rnd = random.Random(1)
        draws = [s.draw(rnd) for _ in range(2000)]
        n3 = sum(1 for d in draws if d["cat"] == "act3-boss")
        self.assertTrue(900 < n3 < 1100, n3)
        hum = sum(1 for d in draws if d["cat"] == "act3-boss" and d["source"] == "human")
        self.assertTrue(0.4 * n3 < hum < 0.6 * n3, (hum, n3))
        self.assertTrue(all(d["file"].endswith("human-starts-act3.jsonl") for d in draws if d["cat"] == "act3-boss" and d["source"] == "human"))
        self.assertTrue(all(d["file"].endswith("own-starts-act3.jsonl") for d in draws if d["cat"] == "act3-boss" and d["source"] == "own"))
        self.write_mix(share=1.0, human_share=1.0, own_encounters=[])
        s, _, _ = self.sampler()
        self.assertTrue(all(d["source"] == "human" and d["cat"] == "act3-boss" for d in (s.draw(random.Random(i)) for i in range(40))))
        self.write_mix(share=1.0, human_share=0.0, own_encounters=[])
        s, _, _ = self.sampler()
        self.assertTrue(all(d["source"] == "own" for d in (s.draw(random.Random(i)) for i in range(40))))

    def test_our_own_arrivals_are_the_test_subjects_by_default_and_the_exams_runs_are_never_drawn(self):
        self.write_mix(share=1.0, human_share=0.0)
        s, pools, dropped = self.sampler()
        own = pools[("act3-boss", "ai")]
        self.assertTrue(own and len(own) == 20)  # 60 arrivals over six cells: the two Test Subject cells
        self.assertEqual(dropped["act3-own (another boss)"], 40)
        lite = {x["id"]: x for x in json.loads((self.sd / "own-starts-act3.lite.json").read_text())}
        self.assertTrue(all(lite[x["id"]]["encounter_id"] == 80 for x in own))
        ex = frozenset({own[0]["id"], pm.run_of(own[1]["id"])})
        s, pools, dropped = self.sampler(ex)
        self.assertEqual(len(pools[("act3-boss", "ai")]), 18)
        h0 = json.loads((self.sd / "human-starts-act3.lite.json").read_text())[0]
        self.write_mix(share=1.0, human_share=1.0)
        s, pools, dropped = self.sampler(frozenset({h0["run"]}))
        self.assertNotIn(h0["id"], {x["id"] for x in pools[("act3-boss", "human")]})

    def test_the_regret_score_reads_the_act_three_humans_results_too(self):
        outs = json.loads((self.sd / a3.OUTCOME_FILE).read_text())["outcomes"]
        (self.sd / pm.HUMAN_OUTCOMES).write_text(json.dumps({"outcomes": {"human:A|f05": {"human_l0": 50.0}}}), encoding="utf-8")
        vals = pm.read_human_values(self.sd)
        self.assertEqual(len(vals), 1 + len(outs))
        self.assertEqual(vals["human:A|f05"], 50.0)
        self.assertIs(pm.read_human_values(self.sd), vals)  # unchanged files: the cache
        (self.sd / pm.HUMAN_OUTCOMES_ACT3).unlink()
        self.assertEqual(pm.read_human_values(self.sd), {"human:A|f05": 50.0})

    def test_the_loops_time_guess_knows_the_kind(self):
        self.assertIn("act3-boss", gl.MEAN_X16_S)
        self.assertIn("own", pm.SOURCES)


def tr(k, turn, hp, form_hp, intent=(0, 22, 1), boss_strength=0, kind="card"):
    """A trace row: decision k of turn `turn`, the player at `hp`, the Test Subject at `form_hp` of the form's maximum."""
    return [k, kind, turn, hp, 80, 0, 3, 0, 0, 0, [[ax.TS_DEF, form_hp[0], form_hp[1], 0, intent[0], intent[1], intent[2], boss_strength]]]


class TraceTests(unittest.TestCase):
    def story(self):
        """Turn 1 and 2 in the first form (111), turn 3 in the second (212), turns 4 and 5 in the third (313); the player dies on turn 5."""
        return [tr(0, 1, 80, (111, 111)), tr(1, 1, 80, (90, 111), kind="end_turn"),
                tr(2, 2, 70, (90, 111)), tr(3, 2, 70, (8, 111), kind="end_turn"),
                tr(4, 3, 55, (212, 212), (0, 11, 3)), tr(5, 3, 55, (190, 212), (0, 11, 3), kind="end_turn"),
                tr(6, 4, 40, (313, 313), (0, 11, 3)), tr(7, 4, 40, (300, 313), (0, 11, 3), 3, kind="end_turn"),
                tr(8, 5, 20, (300, 313), (0, 45, 1), 6), tr(9, 5, 20, (250, 313), (0, 45, 1), 6, kind="end_turn")]

    def test_the_forms_are_the_bosss_distinct_maximum_health_in_order_and_each_turn_belongs_to_the_form_it_began_in(self):
        f = ax.forms_of(self.story())
        self.assertEqual((f["forms_reached"], f["turns"]), (3, 5))
        self.assertEqual({k: v["turns"] for k, v in f["per_form"].items()}, {1: 2, 2: 1, 3: 2})
        # health lost over a turn: the next turn's first reading less this one's; the last turn is left out (its loss is the death)
        self.assertEqual(f["per_form"][1], {"turns": 2, "hp_lost": 10 + 15, "lost_turns": 2, "strength_in": 0})
        self.assertEqual(f["per_form"][2], {"turns": 1, "hp_lost": 15, "lost_turns": 1, "strength_in": 0})
        self.assertEqual(f["per_form"][3], {"turns": 2, "hp_lost": 20, "lost_turns": 1, "strength_in": 0})
        self.assertEqual(f["last"], {"form": 3, "move": [0, 45, 1], "hp": 20, "boss_strength": 6, "boss_hp": 250, "boss_max": 313})

    def test_a_game_without_the_boss_has_no_story(self):
        f = ax.forms_of([[0, "card", 1, 50, 80, 0, 3, 0, 0, 0, [[999, 10, 10, 0, 0, 5, 1, 0]]]])
        self.assertEqual((f["forms_reached"], f["turns"], f["last"]), (0, 0, None))

    def test_the_summary_counts_forms_reached_turns_in_each_and_what_killed(self):
        rows = [{"trace": self.story(), "won": False, "stratum": "human"}, {"trace": self.story(), "won": False, "stratum": "own"},
                {"trace": self.story()[:4], "won": True, "stratum": "own"}]
        t = ax.trace_summary(rows)
        self.assertEqual(sorted(t), ["all", "human", "own"])
        a = t["all"]
        self.assertEqual((a["games"], a["lost"]), (3, 2))
        self.assertAlmostEqual(a["forms"][1]["share"], 1.0)
        self.assertAlmostEqual(a["forms"][3]["share"], 2 / 3)
        self.assertAlmostEqual(a["forms"][3]["turns_mean"], 2.0)
        self.assertAlmostEqual(a["forms"][3]["hp_lost_per_turn"], 20.0)
        self.assertAlmostEqual(a["forms"][3]["strength_in_mean"], 0.0)
        self.assertEqual(a["death_forms"], {3: 2})
        self.assertAlmostEqual(a["death_boss_strength_mean"], 6.0)
        self.assertAlmostEqual(a["death_boss_hp_share_mean"], 250 / 313)
        self.assertAlmostEqual(a["turns_mean"], (5 + 5 + 2) / 3)
        self.assertEqual(a["death_moves"], [{"form": 3, "intent_type": 0, "magnitude": 45, "hits": 1, "games": 2}])
        self.assertEqual(t["human"]["games"], 1)

    def test_the_grader_reads_a_state_document_into_a_compact_row_and_never_raises(self):
        doc = {"turn": 3, "player": {"hp": 55, "max_hp": 80, "block": 4, "energy": 2, "buffs": [{"id": 0, "amount": 3}, {"id": 2, "amount": 1}]},
               "enemies": [{"def_id": 10007, "hp": 100, "max_hp": 111, "block": 0, "buffs": [{"id": 0, "amount": 6}, {"id": 145, "amount": 3}],
                            "intent": {"type": 0, "magnitude": 22, "hits": 2}}]}
        self.assertEqual(ax.TraceGrader.read(doc), [3, 55, 80, 4, 2, 3, 1, 0, [[10007, 100, 111, 0, 0, 22, 2, 6]]])

        class Boom:
            handle = 1

        g = ax.TraceGrader()
        import types  # noqa: PLC0415

        fake = types.SimpleNamespace(get_state_json=lambda h: (_ for _ in ()).throw(RuntimeError("choice screen")))
        sys.modules.setdefault("sts2_gym", types.ModuleType("sts2_gym"))
        old = getattr(sys.modules["sts2_gym"], "native", None)
        sys.modules["sts2_gym"].native = fake
        try:
            self.assertEqual(g(Boom(), 0, types.SimpleNamespace(kind="card")), {})
        finally:
            if old is not None:
                sys.modules["sts2_gym"].native = old
        self.assertEqual((g.misses, g.rows), (1, []))


def game(start, j, cell, stratum, won, human=None, **kw):
    return {"_task": ["1x", start, j], "arm": "1x", "start": start, "j": j, "cell": cell, "stratum": stratum, "won": won, "capped": False,
            "encounter_id": int(cell.rsplit("-", 1)[1]), "boss_no": int(cell[9]), "start_hp": 60, "start_max_hp": 80, "human_survived": human, "seconds": 20.0,
            "end_hp": 10 if won else 0, **kw}


class PlayPlanTests(unittest.TestCase):
    def starts(self):
        out = {}
        for stratum in ("human", "own"):
            for cell, n in (("act3-boss1-80", 12), ("act3-boss1-78", 6)):
                for i in range(n):
                    sid = f"{stratum}:{cell}:{i}"
                    out[sid] = {"id": sid, "cell": cell, "stratum": stratum}
        return out

    def test_take_names_the_starts_a_cell_and_a_bigger_take_nests_the_smaller(self):
        self.assertEqual(ax.parse_take("act3-boss1-80=4, act3-boss1-78=2"), {"act3-boss1-80": 4, "act3-boss1-78": 2})
        self.assertEqual(ax.parse_take(""), {})
        small = ax.select_starts(self.starts(), {"act3-boss1-80": 4, "act3-boss1-78": 2})
        big = ax.select_starts(self.starts(), {"act3-boss1-80": 8, "act3-boss1-78": 2})
        self.assertEqual(Counter((s["stratum"], s["cell"]) for s in small), Counter({("human", "act3-boss1-80"): 4, ("own", "act3-boss1-80"): 4,
                                                                                      ("human", "act3-boss1-78"): 2, ("own", "act3-boss1-78"): 2}))
        self.assertTrue({s["id"] for s in small} <= {s["id"] for s in big})
        self.assertEqual(len(ax.select_starts(self.starts())), 36)
        self.assertEqual({s["stratum"] for s in ax.select_starts(self.starts(), strata=("human",))}, {"human"})

    def test_the_games_to_play_are_every_start_a_draw_at_a_time_and_a_stop_loses_only_what_has_no_row(self):
        chosen = ax.select_starts(self.starts(), {"act3-boss1-80": 3})
        tasks = ax.tasks_for(chosen, 2, set())
        self.assertEqual(len(tasks), 12)
        self.assertEqual([t[2] for t in tasks], [0] * 6 + [1] * 6)  # draw 0 of every start first
        done = {tasks[0], tasks[7]}
        rest = ax.tasks_for(chosen, 2, done)
        self.assertEqual(len(rest), 10)
        self.assertNotIn(tasks[0], rest)
        self.assertEqual([t[1] for t in ax.tasks_for(chosen, 2, set(), "start")][:2], [chosen[0]["id"]] * 2)
        rows = [{"_task": list(t)} for t in tasks[:3]]
        self.assertEqual(ax.done_keys(rows), set(tasks[:3]))


class ReportTests(unittest.TestCase):
    def rows(self):
        rows = []
        for i in range(40):  # human Test Subject positions: the champion kills 3 of 40 starts (two draws each); the humans passed 28 of 40
            rows += [game(f"h{i}", j, "act3-boss1-80", "human", i < 3 and j == 0, human=i < 28) for j in range(2)]
        for i in range(40):  # ours: no kills
            rows += [game(f"o{i}", j, "act3-boss1-80", "own", False) for j in range(2)]
        for i in range(20):  # the Queen: human positions 5 of 20 starts won on one draw each
            rows += [game(f"q{i}", 0, "act3-boss1-78", "human", i < 5, human=True)]
        rows.append(game("h0", 0, "act3-boss1-80", "human", False, human=True))  # a rerun of a game: the later row stands
        return rows

    def test_kills_by_side_and_cell_with_intervals_and_the_human_bar(self):
        d = ax.report_data(self.rows(), boot=300)
        c = d["cells"]["human|act3-boss1-80"]
        self.assertEqual((c["starts"], c["games"], c["kills"]), (40, 80, 2))  # the rerun replaced h0's first game (a kill) by a loss
        self.assertAlmostEqual(c["rate"], 2 / 80)
        lo, hi = ax.wilson(2, 80)
        self.assertEqual((c["lo"], c["hi"]), (lo, hi))
        self.assertEqual(d["cells"]["own|act3-boss1-80"]["kills"], 0)
        self.assertEqual(d["bosses"]["human|boss1|test-subject"]["games"], 80)
        self.assertEqual(d["bosses"]["human|boss1|queen"]["kills"], 5)
        h = d["human_bar"]["all"]
        self.assertEqual(h["starts"], 60)
        self.assertAlmostEqual(h["human"], (28 + 20) / 60)
        ts = d["human_bar"]["act3-boss1-80"]
        self.assertAlmostEqual(ts["human"], 28 / 40)
        self.assertAlmostEqual(ts["champion"], (0.5 * 2) / 40)  # h1 and h2 won on draw 0 only: half a start each
        self.assertAlmostEqual(ts["diff"], ts["champion"] - ts["human"])
        self.assertLess(ts["lo"], ts["diff"])
        self.assertGreater(ts["hi"], ts["diff"])
        self.assertEqual(d["games"], len({(r["start"], r["j"]) for r in self.rows()}))

    def test_the_text_names_the_cells_and_carries_the_numbers_and_no_run(self):
        text = ax.report_text(ax.report_data(self.rows(), boot=300))
        self.assertIn("human | act3-boss1-80 (test-subject)", text)
        self.assertIn("own | act3-boss1-80 (test-subject)", text)
        self.assertIn("human | act3-boss1-78 (queen)", text)
        self.assertIn("The champion against the human on the same starts", text)
        self.assertIn("champion", text)

    def test_a_capped_game_is_a_loss_and_counted(self):
        rows = [game("a", 0, "act3-boss1-80", "human", True, capped=True), game("b", 0, "act3-boss1-80", "human", True)]
        d = ax.report_data(rows, boot=50)
        self.assertEqual((d["capped"], d["cells"]["human|act3-boss1-80"]["kills"]), (1, 1))

    def test_the_traces_ride_in_the_report(self):
        story = TraceTests().story()
        rows = [game("t", 0, "act3-boss1-80", "own", False, trace=story)]
        d = ax.report_data(rows, boot=50)
        self.assertEqual(d["trace"]["own"]["forms"][3]["games"], 1)
        self.assertIn("killed by form 3 move type 0 45 x 1", ax.report_text(d))


class DeckTests(unittest.TestCase):
    TYPES = {1: "Attack", 2: "Skill", 3: "Power", 4: "Curse"}

    def test_a_deck_is_counted_by_card_type_and_an_upgraded_card_is_a_negative_id(self):
        m = a3.deck_mix([1, -1, 2, 2, 2, 3, 4, 99], self.TYPES)
        self.assertEqual(m, {"size": 8, "attack": 2, "skill": 3, "power": 1, "curse": 1, "?": 1})

    def test_the_table_splits_by_side_boss_and_result_and_drops_a_small_cell(self):
        human = [{"boss_no": 1, "encounter_id": 80, "deck": [1, 2, 2, 3], "outcome": {"human_survived": i < 12}} for i in range(35)]
        own = [(1, 80, i < 3, [1, 1, 1, 2]) for i in range(40)]
        t = a3.decks_table(human, iter(own), self.TYPES)
        self.assertEqual(sorted(t), ["human|boss1|test-subject|all", "human|boss1|test-subject|died", "own|boss1|test-subject|all",
                                     "own|boss1|test-subject|lost"])  # 12 went past and 3 won: under 20, not shown
        h = t["human|boss1|test-subject|all"]
        self.assertEqual((h["n"], h["size"], h["skill"], h["attack"], h["power"]), (35, 4.0, 2.0, 1.0, 1.0))
        self.assertEqual(t["human|boss1|test-subject|died"]["n"], 23)
        o = t["own|boss1|test-subject|lost"]
        self.assertEqual((o["n"], o["skill"], o["attack"]), (37, 1.0, 3.0))


class PureTests(unittest.TestCase):
    def test_the_hash_of_a_run_is_stable_and_the_corpus_default_is_the_old_one(self):
        self.assertEqual(a3.run_tag("abc"), hashlib.sha256((a3.TAG_SALT + "abc").encode()).hexdigest()[:12])
        self.assertEqual(a3.BOSS_ID["ENCOUNTER.TEST_SUBJECT_BOSS"], 80)
        self.assertEqual(sorted(fe.ENCOUNTER_NAMES[b] for b in (73, 78, 80)), ["aeonglass", "queen", "test-subject"])

    def test_wilson(self):
        lo, hi = a3.wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.2)
        self.assertEqual(a3.wilson(0, 0), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
