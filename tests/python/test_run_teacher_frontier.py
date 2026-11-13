"""The frontier grade beside floors, the regrade and the door rows (rl/run_teacher.py; D59 lane B; docs/lab/TEACHER_JOB.md "The frontier grade").

Pinned: one continuation's grade (0 dead before its door, 1 - p2 at the Act 2 door, (1 - p1) x A2 at the Act 1 door, 1 won; x 13.5, floors-
equivalent) and its no-return cases; floors stays the default grade (parser, guard priors, the manifest); `regrade` writes a regraded store under a
new root from the stored continuations (Act 2 exact for every kind, Act 1 deck kinds with A2 read on the option's stored row, Act 1 routes and events
skipped and counted, Act 2 labels only from the Act 2 judge's runs, Act 1 recalibrated per judge on the store's own doors) and leaves the stored store
untouched; a floors regrade reproduces the stored labels byte for byte; the door rows (`rows_door.npz`: per draw the door's row, float16, with A2 and
the floors G) are built at the Act 1 door, stored per run, cut back with the screens they belong to, written into the shard, and make Act 1 routes and
events exact; the constants a grade's unit sets are measured from a store's records.

No game and no emulator: lane A's `Act2Scorer` is a stub whose probability of a row is its first column.
"""

import contextlib
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import run_teacher as rt  # noqa: E402


class StubA2:
    """Lane A's `Act2Scorer` as the teacher sees it: `.score(rows) -> probabilities`, `.sha256`. Here the probability of a row is its first column."""

    sha256 = "ab" * 32

    def __init__(self, nan_for=()):
        self.calls = []
        self.nan_for = tuple(nan_for)

    def score(self, rows):
        rows = np.asarray(rows, dtype="float32")
        self.calls.append(rows.copy())
        out = np.clip(rows[:, 0], 0.0, 1.0).astype("float64")
        for v in self.nan_for:
            out[np.isclose(rows[:, 0], v)] = np.nan
        return out


class FrontierGradeTests(unittest.TestCase):
    """The grade of one continuation, floors-equivalent: 13.5 x (0 dead, 1 - p2 at the Act 2 door, (1 - p1) x A2 at the Act 1 door, 1 won)."""

    def test_each_end_has_its_grade(self):
        s = rt.FRONTIER_SCALE
        self.assertEqual(13.5, s)
        self.assertEqual(0.0, rt.frontier_return("death", 12, None))
        self.assertEqual(0.0, rt.frontier_return("death", 30, None, a2=0.9))
        self.assertAlmostEqual(rt.frontier_return("win", 53, None), s)
        self.assertAlmostEqual(rt.frontier_return("door", 33, 0.25), s * 0.75)  # Act 2 door: 1 - p2
        self.assertAlmostEqual(rt.frontier_return("door", 33, 1.0), 0.0)
        self.assertAlmostEqual(rt.frontier_return("door", 17, 0.2, a2=0.3), s * 0.8 * 0.3)  # Act 1 door: (1 - p1) x A2
        self.assertAlmostEqual(rt.frontier_return("door", 17, 0.0, a2=0.3), s * 0.3)
        self.assertAlmostEqual(rt.frontier_return("door", 17, 0.2, a2=0.3, scale=1.0), 0.8 * 0.3)  # scale 1: the grade itself
        self.assertAlmostEqual(rt.frontier_return("door", 33, 0.25, a2=0.9), s * 0.75)  # A2 is the Act 1 door's: the Act 2 door ignores it

    def test_no_return_for_the_rest(self):
        for end in ("timeout", "step_cap", "?", "next_act"):
            self.assertIsNone(rt.frontier_return(end, 10, 0.5, a2=0.3))
        self.assertIsNone(rt.frontier_return("door", 17, 0.5))  # an Act 1 door with no A2
        self.assertIsNone(rt.frontier_return("door", 17, 0.5, a2=float("nan")))
        self.assertIsNone(rt.frontier_return("door", 17, None, a2=0.3))  # no judge estimate
        self.assertIsNone(rt.frontier_return("door", 48, 0.5, a2=0.3))  # Act 3's door: no grade for it
        self.assertIsNone(rt.frontier_return("door", 33, 1.5))
        self.assertIsNone(rt.frontier_return("door", 17, 0.5, a2=1.5))

    def test_the_judges_death_chance_is_recalibrated_per_act_as_the_floors_leaf_does(self):
        cal = {0: (0.4, 1.2), 1: (-0.3, 0.9)}
        p0, p1 = rt.recalibrate(0.5, 0, cal), rt.recalibrate(0.5, 1, cal)
        self.assertNotAlmostEqual(p0, 0.5)
        self.assertAlmostEqual(rt.frontier_return("door", 17, 0.5, a2=0.4, cal=cal), 13.5 * (1 - p0) * 0.4)
        self.assertAlmostEqual(rt.frontier_return("door", 33, 0.5, cal=cal), 13.5 * (1 - p1))
        self.assertAlmostEqual(rt.frontier_return("door", 33, 0.5, cal={0: (5.0, 1.0)}), 13.5 * 0.5)  # an act with no map keeps the judge's own number

    def test_a_deck_the_act_2_model_likes_beats_one_it_does_not_and_a_death_costs_the_chance_it_held(self):
        self.assertGreater(rt.frontier_return("door", 17, 0.5, a2=0.4), rt.frontier_return("door", 17, 0.5, a2=0.2))
        self.assertGreater(rt.frontier_return("door", 17, 0.3, a2=0.2), rt.frontier_return("door", 17, 0.6, a2=0.2))
        self.assertGreater(rt.frontier_return("door", 17, 0.01, a2=0.01), rt.frontier_return("death", 16, None))

    def test_floors_stays_the_default_grade(self):
        self.assertEqual("floors", rt.DEFAULT_GRADE)
        self.assertEqual(("floors", "frontier"), rt.GRADES)
        a = rt.build_parser().parse_args(["job", "--lab", "L", "--store", "S", "--agent-root", "A"])
        self.assertEqual("floors", a.grade)
        self.assertTrue(a.door_rows)  # the lab's own command line gains the door rows without a flag
        self.assertFalse(rt.build_parser().parse_args(["job", "--lab", "L", "--store", "S", "--agent-root", "A", "--no-door-rows"]).door_rows)
        self.assertEqual(rt.door_return("door", 17, 0.4), rt.draw_return("floors", "door", 17, 0.4))
        self.assertEqual(rt.frontier_return("door", 17, 0.4, a2=0.3), rt.draw_return("frontier", "door", 17, 0.4, a2=0.3))
        with self.assertRaises(ValueError):
            rt.draw_return("other", "door", 17, 0.4)

    def test_the_guards_priors_are_per_grade_and_floors_keeps_its_own(self):
        self.assertEqual(rt.PAIRED_SD_PRIOR, rt.guard_priors("floors"))
        self.assertAlmostEqual(rt.guard_se([5.0, 5.0], "card"), rt.guard_se([5.0, 5.0], "card", grade="floors"))
        self.assertAlmostEqual(rt.guard_se([5.0, 5.0], "card"), math.sqrt((4 * 3.41**2) / 5 / 2), places=6)
        with mock.patch.dict(rt.PAIRED_SD_PRIOR_BY_GRADE, {"frontier": {"card": 1.0}}):
            self.assertAlmostEqual(rt.guard_se([5.0, 5.0], "card", grade="frontier"), math.sqrt((4 * 1.0) / 5 / 2), places=6)
        with mock.patch.dict(rt.PAIRED_SD_PRIOR_BY_GRADE, {"frontier": None}), self.assertRaises(ValueError):
            rt.guard_se([5.0, 5.0], "card", grade="frontier")  # not measured in that unit yet: no guess

    def test_the_act_2_model_file_is_found_by_default_only_when_it_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "a2.pt"
            with mock.patch.object(rt, "DEFAULT_A2", str(model)):
                self.assertEqual((None, None, "auto: no model file yet"), rt.resolve_a2("auto"))
                self.assertEqual((None, None, "auto: no model file yet"), rt.resolve_a2(None))
                model.write_bytes(b"model")
                path, sha, note = rt.resolve_a2("auto")
                self.assertEqual((str(model), rt.file_digest(model), "auto"), (path, sha, note))
                self.assertEqual((None, None, "off"), rt.resolve_a2("none"))
                self.assertEqual((None, None, "off"), rt.resolve_a2(""))
                with self.assertRaises(SystemExit):
                    rt.resolve_a2(str(Path(tmp) / "elsewhere.pt"))
                other = Path(tmp) / "x.pt"
                other.write_bytes(b"x")
                self.assertEqual((str(other), rt.file_digest(other), "given"), rt.resolve_a2(str(other)))

    def test_a_live_frontier_teacher_needs_its_scorer_and_its_measured_constants(self):
        with mock.patch.dict(rt.PAIRED_SD_PRIOR_BY_GRADE, {"frontier": None}), self.assertRaises(SystemExit):
            rt.check_grade("frontier", a2_scorer=StubA2())
        with mock.patch.dict(rt.PAIRED_SD_PRIOR_BY_GRADE, {"frontier": {"card": 1.0}}):
            with self.assertRaises(SystemExit):
                rt.check_grade("frontier", a2_scorer=None)
            rt.check_grade("frontier", a2_scorer=StubA2())
        rt.check_grade("floors", a2_scorer=None)


# ---------------------------------------------------------------------------------------------------------- stored records and stores


def graded(seed, ordinal, kind, *, floor, draws, cal=None, n=None, **over):
    """A stored decision record as the teacher writes it. `draws[option]` is a list of ("door", p), ("death", floor) or ("win", None): the record's G
    are the floors leaf's (`door_return`, the door's floor from the screen's: 17 below it, else 33), rounded to 4 places as stored."""
    door_floor = 17 if floor < 17 else 33
    played = sorted(draws)
    n = n or max(played) + 1
    g, p, hp, ends = {}, {}, {}, Counter()
    for o in played:
        g[str(o)], p[str(o)], hp[str(o)] = [], [], []
        for end, x in draws[o]:
            ends[end] += 1
            if end == "door":
                g[str(o)].append(round(rt.door_return("door", door_floor, x, cal=cal), 4))
                p[str(o)].append(x)
                hp[str(o)].append(50)
            else:
                g[str(o)].append(round(rt.door_return(end, int(x or 0), None), 4))
                p[str(o)].append(None)
                hp[str(o)].append(None)
    k = min(len(v) for v in g.values())
    q = {str(o): round(rt.mean(g[str(o)]), 4) for o in played}
    typ = {"card": "card_reward", "route": "map", "event": "event", "rest": "rest", "shop": "shop", "ancient": "neow", "reward": "rewards",
           "select": "card_select", "bundle": "bundle"}[kind]
    rec = {"v": rt.VERSION, "seed": seed, "ordinal": ordinal, "type": typ, "kind": kind, "act": 0 if floor < 18 else 1, "floor": floor, "hp": 60, "max_hp": 80,
           "n_options": n, "played": played, "player": played[0], "k": k, "screen": None, "lam": 0.0, "v_values": [None] * n, "logp": None, "g": g,
           "ends": dict(ends), "p_death": p, "door_hp": hp, "options": ["x.y"] * n, "teacher": played[0], "teacher_raw": played[0], "q": q,
           "gain": 0.0, "gain_se": None, "gain_se_own": None, "half_picks": [played[0]] * 2, "half_played": [played[0]] * 2, "cpu": 10.0, "seconds": 10.0}
    rec.update(over)
    return rec


def a2_rows(values):
    """Option rows whose first column is the A2 the stub scorer returns for them."""
    rows = np.zeros((len(values), rt.ROWS_WIDTH), dtype="float32")
    rows[:, 0] = values
    rows[:, 1] = np.arange(len(values)) + 1  # something else that tells the rows apart
    return rows


def door_rows_of(values):
    """Door rows (float16): the first column is the A2 the stub scorer returns for a row."""
    rows = np.zeros((len(values), rt.ROWS_WIDTH), dtype="float16")
    rows[:, 0] = values
    rows[:, 2] = np.arange(len(values)) + 1
    return rows


class StoreCase(unittest.TestCase):
    """A synthetic store `<tmp>/run-teacher/14` built the way the job builds one (RunFiles, then `build_shard`)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.src = self.tmp / "run-teacher" / "14"
        self.out = self.tmp / "run-teacher-frontier" / "14"
        self.runs = []

    def manifest(self, name, judge=rt.V2B_JUDGE_SHA256, recal=None):
        base = rt.shard_manifest(job=name, generation=14, identity={"agent": "gen14/x", "agent_sha256": "ab" * 32, "generation": 14, "recipe": "r",
                                                                  "agent_files": {}}, judge=None, lib=None, tree="abc", k=2, k_max=4, max_options=4,
                                 cal=recal or {}, sources={"new": 1})
        base["judge_sha256"] = judge
        return base

    def add_shard(self, name, runs, judge=rt.V2B_JUDGE_SHA256, recal=None):
        """`runs`: {seed: [(record, rows) or (record, rows, door_rows)]}; writes the run files, the shard and a runs.jsonl row per run under `judge`."""
        for seed, screens in runs.items():
            rf = rt.RunFiles(self.src, seed)
            rf.load()
            for item in screens:
                rec, rows = item[0], item[1]
                rf.append(rec, rows, door=item[2] if len(item) > 2 else None)
            self.runs.append({"_task": ["rt-new", seed, 1], "arm": "teacher", "source": "new", "job": name, "judge_sha256": judge, "seed": seed, "ix": 1,
                              "end": "death", "floor": 20, "bosses": []})
        self.write_runs(self.runs)
        return rt.build_shard(self.src, name, list(runs), self.manifest(name, judge, recal))

    @staticmethod
    def write_jsonl(path, rows):
        Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def write_runs(self, rows):
        self.write_jsonl(self.src / "runs.jsonl", rows)

    def read_runs(self):
        return [json.loads(x) for x in (self.src / "runs.jsonl").read_text(encoding="utf-8").splitlines()]

    def records(self, root, name="J1"):
        return [json.loads(x) for x in (root / "shards" / name / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]

    def doors(self, n, p, judge, prefix="R", won=lambda i: i % 5 != 0):
        return [{"_task": ["rt-new", f"{prefix}{i}", i], "arm": "teacher", "source": "new", "judge_sha256": judge, "seed": f"{prefix}{i}", "ix": i,
                 "end": "death", "floor": 20, "bosses": [{"act": 0, "floor": 17, "hp": 50, "max_hp": 80, "p_death": p, "won": won(i)}]} for i in range(n)]


class RegradeTests(StoreCase):
    """`regrade`: a store's continuations graded again under a new root, the stored one untouched."""

    def standard_store(self, judge=rt.V2B_JUDGE_SHA256, recal=None):
        """One run: Act 1 card, route, event and rest screens, then an Act 2 card screen and an Act 2 route screen."""
        s = "RUNAAAAAAA"
        screens = [
            (graded(s, 0, "card", floor=3, draws={0: [("door", 0.2), ("door", 0.4)], 1: [("door", 0.3), ("death", 8)], 2: [("door", 0.1), ("door", 0.1)]}),
             a2_rows([0.10, 0.20, 0.30])),
            (graded(s, 1, "route", floor=3, draws={0: [("door", 0.5), ("door", 0.5)], 1: [("door", 0.2), ("death", 9)]}), a2_rows([0.1, 0.1])),
            (graded(s, 2, "event", floor=5, draws={0: [("door", 0.5), ("door", 0.5)], 1: [("door", 0.2), ("death", 9)]}), a2_rows([0.1, 0.1])),
            (graded(s, 3, "rest", floor=8, draws={0: [("door", 0.6), ("door", 0.5)], 1: [("door", 0.2), ("door", 0.2)]}), a2_rows([0.4, 0.6])),
            (graded(s, 4, "card", floor=20, draws={0: [("door", 0.5), ("door", 0.3)], 1: [("death", 25), ("door", 0.2)]}), a2_rows([0.3, 0.3])),
            (graded(s, 5, "route", floor=21, draws={0: [("death", 22), ("door", 0.9)], 1: [("door", 0.4), ("door", 0.8)]}), a2_rows([0.3, 0.3])),
        ]
        return self.add_shard("J1", {s: screens}, judge=judge, recal=recal)

    def regrade(self, **kw):
        kw.setdefault("scorer", StubA2())
        kw.setdefault("grade", "frontier")
        return rt.regrade_store(self.src, self.out, **kw)

    def test_a_floors_regrade_reproduces_the_stored_labels_byte_for_byte(self):
        cal = {"0": {"a": -0.2, "b": 1.1, "n": 150}, "1": {"a": 0.3, "b": 0.8, "n": 70}}
        c = {0: (-0.2, 1.1), 1: (0.3, 0.8)}  # the labels were made under the shard's recalibration
        s = "RUNAAAAAAA"
        screens = [(graded(s, 0, "card", floor=3, draws={0: [("door", 0.2), ("door", 0.4)], 1: [("door", 0.3), ("death", 8)]}, cal=c), a2_rows([0.1, 0.2])),
                   (graded(s, 1, "route", floor=21, draws={0: [("death", 22), ("door", 0.9)], 1: [("door", 0.4), ("door", 0.8)]}, cal=c), a2_rows([0.3, 0.3])),
                   (graded(s, 2, "route", floor=17, draws={0: [("door", 0.9), ("door", 0.9)], 1: [("door", 0.4), ("win", None)]}, cal=c), a2_rows([0.3, 0.3]))]
        self.add_shard("J1", {s: screens}, recal=cal)
        before = {p.name: p.read_bytes() for p in (self.src / "shards" / "J1").iterdir()}
        res = self.regrade(grade="floors", scorer=None)
        self.assertEqual(0, res["mismatches"])
        self.assertEqual({"regraded": 3}, res["by_reason"])
        self.assertEqual((self.src / "shards" / "J1" / "decisions.jsonl").read_bytes(), (self.out / "shards" / "J1" / "decisions.jsonl").read_bytes())
        self.assertEqual({p.name: p.read_bytes() for p in (self.src / "shards" / "J1").iterdir()}, before)  # the stored store is untouched
        self.assertEqual("floors", json.loads((self.out / "shards" / "J1" / "manifest.json").read_text(encoding="utf-8"))["grade"])
        np.testing.assert_array_equal(np.load(self.src / "shards" / "J1" / "rows.npz")["rows"], np.load(self.out / "shards" / "J1" / "rows.npz")["rows"])
        # a label that does not reproduce is counted, not hidden
        bad = self.records(self.src)
        bad[0]["g"]["0"][0] += 1.0
        (self.src / "shards" / "J1" / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bad), encoding="utf-8")
        self.assertEqual(1, self.regrade(grade="floors", scorer=None)["mismatches"])

    def test_act_2_is_exact_for_every_kind(self):
        self.standard_store()
        self.regrade()
        recs = {(r["kind"], r["floor"]): r for r in self.records(self.out)}
        card2, route2 = recs[("card", 20)], recs[("route", 21)]
        s = 13.5
        # option 0: the door at p 0.5 then 0.3; option 1: a death before the door (0) then the door at 0.2
        self.assertEqual([round(s * 0.5, 4), round(s * 0.7, 4)], card2["g"]["0"])
        self.assertEqual([0.0, round(s * 0.8, 4)], card2["g"]["1"])
        self.assertEqual(round(rt.mean(card2["g"]["0"]), 4), card2["q"]["0"])
        self.assertEqual([0.0, round(s * 0.1, 4)], route2["g"]["0"])  # a route in Act 2 is graded too
        self.assertEqual("frontier", card2["grade"])
        stored = {(r["kind"], r["floor"]): r for r in self.records(self.src)}
        self.assertEqual(stored[("card", 20)]["g"], card2["g_floors"])  # the floors labels stay beside
        self.assertEqual(stored[("card", 20)]["q"], card2["q_floors"])
        self.assertNotIn("a2", card2)  # Act 2 needs no A2

    def test_act_1_deck_kinds_read_the_option_row(self):
        self.standard_store()
        stub = StubA2()
        res = self.regrade(scorer=stub)
        recs = {(r["kind"], r["floor"]): r for r in self.records(self.out)}
        card1 = recs[("card", 3)]
        s = 13.5
        # option 0 (A2 0.10): draws at p 0.2 and 0.4; option 1 (A2 0.20): p 0.3 then a death; option 2 (A2 0.30): p 0.1, 0.1
        self.assertEqual([round(s * 0.8 * 0.10, 4), round(s * 0.6 * 0.10, 4)], card1["g"]["0"])
        self.assertEqual([round(s * 0.7 * 0.20, 4), 0.0], card1["g"]["1"])
        self.assertEqual([round(s * 0.9 * 0.30, 4)] * 2, card1["g"]["2"])
        self.assertEqual({"0": 0.1, "1": 0.2, "2": 0.3}, {o: round(v, 4) for o, v in card1["a2"].items()})
        self.assertEqual(round(rt.mean(card1["g"]["1"]), 4), card1["q"]["1"])
        rest = recs[("rest", 8)]
        self.assertEqual([round(s * 0.4 * 0.4, 4), round(s * 0.5 * 0.4, 4)], rest["g"]["0"])
        self.assertEqual("option_row", card1["a2_from"])
        # the scorer was asked about the option rows stored with the screen, nothing else
        asked = {round(float(x), 4) for c in stub.calls for x in c[:, 0]}
        self.assertEqual({0.1, 0.2, 0.3, 0.4, 0.6}, asked)
        self.assertEqual(rt.ROWS_WIDTH, stub.calls[0].shape[1])
        self.assertEqual(stub.sha256, res["a2_sha256"])
        self.assertEqual({"option_row": 2}, res["a2_from"])

    def test_act_1_routes_and_events_are_skipped_and_counted(self):
        self.standard_store()
        res = self.regrade()
        kinds = [(r["kind"], r["floor"]) for r in self.records(self.out)]
        self.assertNotIn(("route", 3), kinds)
        self.assertNotIn(("event", 5), kinds)
        self.assertIn(("route", 21), kinds)
        self.assertEqual(2, res["by_reason"]["act1_route_event"])
        self.assertEqual(4, res["by_reason"]["regraded"])
        self.assertEqual(6, res["decisions"])
        self.assertEqual({"card": "frontier", "rest": "frontier", "route": "frontier"}, res["grade_by_kind"])
        self.assertNotIn("event", res["grade_by_kind"])  # no Act 1 event kept, no Act 2 event in the store: the kind has no label in the new root

    def test_the_regraded_shard_is_a_shard_the_student_reads(self):
        self.standard_store()
        self.regrade()
        d = self.out / "shards" / "J1"
        z = np.load(d / "rows.npz")
        recs = self.records(self.out)
        self.assertEqual(4, len(recs))
        counts = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["counts"]
        self.assertEqual({"decisions": 4, "rows": len(z["rows"])}, {k: counts[k] for k in ("decisions", "rows")})
        for i, r in enumerate(recs):  # row0 and the decision index were renumbered over the screens kept
            n = int(r["n_options"])
            self.assertEqual([i] * n, z["decision"][r["row0"]:r["row0"] + n].tolist())
        src = {(r["kind"], r["floor"]): r for r in self.records(self.src)}
        stored_rows = np.load(self.src / "shards" / "J1" / "rows.npz")["rows"]
        card = next(r for r in recs if r["kind"] == "card" and r["floor"] == 20)
        np.testing.assert_array_equal(z["rows"][card["row0"]:card["row0"] + 2], stored_rows[src[("card", 20)]["row0"]:src[("card", 20)]["row0"] + 2])
        try:
            import run_student as rs
        except ImportError:  # pragma: no cover
            return
        with mock.patch.dict(rs.TAU_MIN_BY_GRADE, {"frontier": 0.1}):  # the unit's own floor on the temperature, once measured
            ts = rs.load_shards(self.out)
        self.assertEqual(4, len(ts))
        self.assertEqual("frontier", ts.stats["grade"])
        with self.assertRaises(rs.ShardError):
            rs.load_shards(self.out)  # not measured: refused

    def test_the_manifest_names_the_grade_the_scale_the_scorer_and_the_filter(self):
        self.standard_store()
        res = self.regrade()
        man = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        for key in ("grade", "scale", "a2_sha256", "by_reason", "grade_by_kind", "act2_judges", "calibration", "kind", "source"):
            self.assertIn(key, man)
        self.assertEqual(("frontier", 13.5, StubA2.sha256), (man["grade"], man["scale"], man["a2_sha256"]))
        self.assertEqual([rt.V2B_JUDGE_SHA256], man["act2_judges"])
        self.assertEqual(res["by_reason"], man["by_reason"])
        text = json.dumps(man) + (self.out / "shards" / "J1" / "manifest.json").read_text(encoding="utf-8")
        self.assertNotRegex(text, r"[A-Za-z]:[\\/]")
        sm = json.loads((self.out / "shards" / "J1" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(("frontier", 13.5, StubA2.sha256), (sm["grade"], sm["scale"], sm["a2_sha256"]))

    def test_act_2_labels_only_from_runs_the_act_2_judge_made(self):
        self.standard_store(judge=rt.V0_JUDGE_SHA256)
        res = self.regrade()
        kinds = [(r["kind"], r["floor"]) for r in self.records(self.out)]
        self.assertEqual([("card", 3), ("rest", 8)], kinds)  # Act 1 keeps its deck kinds under v0; the Act 2 screens are another judge's numbers
        self.assertEqual(2, res["by_reason"]["act2_judge"])
        rows = self.read_runs()  # the judge of a run is its row's, else the shard's
        rows[0]["judge_sha256"] = rt.V2B_JUDGE_SHA256
        self.write_runs(rows)
        res = self.regrade()
        self.assertEqual(0, res["by_reason"].get("act2_judge", 0))
        self.assertEqual(4, len(self.records(self.out)))

    def test_act_1_is_recalibrated_per_judge_on_the_stores_own_door_outcomes(self):
        self.standard_store()
        mine = self.doors(100, 0.5, rt.V2B_JUDGE_SHA256)
        other = self.doors(100, 0.5, rt.V0_JUDGE_SHA256, prefix="Q", won=lambda i: False)
        self.write_runs(self.read_runs() + mine + other)
        res = self.regrade()
        fit = rt.fit_door_calibration(rt.door_pairs(mine, rt.V2B_JUDGE_SHA256))
        self.assertIn(0, fit)
        p_cal = rt.recalibrate(0.2, 0, rt.calibration_map(fit))
        self.assertNotAlmostEqual(p_cal, 0.2, places=3)
        card1 = next(r for r in self.records(self.out) if r["kind"] == "card" and r["floor"] == 3)
        self.assertAlmostEqual(card1["g"]["0"][0], round(13.5 * (1 - p_cal) * 0.10, 4), places=4)  # v2b's own map; the v0 doors do not enter it
        self.assertEqual(100, res["calibration"][rt.V2B_JUDGE_SHA256[:8]]["0"]["n"])
        self.assertEqual(100, res["calibration"][rt.V0_JUDGE_SHA256[:8]]["0"]["n"])  # each judge has its own

    def test_a_judge_with_no_doors_of_its_own_borrows_another_stores_under_the_same_judge(self):
        s = "RUNAAAAAAA"
        two = {0: [("door", 0.2)] * 2, 1: [("door", 0.2)] * 2}
        self.add_shard("J1", {s: [(graded(s, 0, "card", floor=3, draws=two, n=2), a2_rows([0.1, 0.2]))]}, judge=rt.V0_JUDGE_SHA256)
        plain = self.regrade()
        self.assertEqual({}, plain["calibration"])  # nothing to fit on: the judge's own number stands
        card = self.records(self.out)[0]
        self.assertEqual(round(13.5 * 0.8 * 0.1, 4), card["g"]["0"][0])
        extra = self.tmp / "run-teacher" / "other"
        extra.mkdir(parents=True)
        other = self.doors(100, 0.5, rt.V0_JUDGE_SHA256, prefix="Q", won=lambda i: False)
        self.write_jsonl(extra / "runs.jsonl", other)
        res = rt.regrade_store(self.src, self.out, grade="frontier", scorer=StubA2(), doors_from=[extra])
        fit = rt.fit_door_calibration(rt.door_pairs(other, rt.V0_JUDGE_SHA256))
        p_cal = rt.recalibrate(0.2, 0, rt.calibration_map(fit))
        self.assertGreater(p_cal, 0.2)
        self.assertEqual(round(13.5 * (1 - p_cal) * 0.1, 4), self.records(self.out)[0]["g"]["0"][0])
        self.assertEqual({rt.V0_JUDGE_SHA256[:8]}, set(res["calibration"]))

    def test_a_refused_screen_a_missing_row_and_a_scorer_that_cannot_say_are_skipped_by_reason(self):
        s = "RUNAAAAAAA"
        two = {0: [("door", 0.2)] * 2, 1: [("door", 0.2)] * 2}
        refused = graded(s, 0, "card", floor=3, draws=two, n=2)
        refused.pop("q")
        refused.update(refused="no_return")
        short = graded(s, 1, "card", floor=3, draws={0: [("door", 0.2)] * 2, 1: [("door", 0.2)] * 2, 2: [("door", 0.2)] * 2}, n=3, _ro=[0, 2])
        nan = graded(s, 2, "card", floor=3, draws=two, n=2)
        bundle1 = graded(s, 3, "bundle", floor=3, draws=two, n=2)
        bundle2 = graded(s, 4, "bundle", floor=20, draws=two, n=2)
        ok = graded(s, 5, "card", floor=3, draws=two, n=2)
        self.add_shard("J1", {s: [(refused, None), (short, a2_rows([0.1, 0.1])), (nan, a2_rows([0.77, 0.3])), (bundle1, a2_rows([0.2, 0.2])),
                                  (bundle2, a2_rows([0.2, 0.2])), (ok, a2_rows([0.2, 0.3]))]})
        res = self.regrade(scorer=StubA2(nan_for=[0.77]))
        self.assertEqual({"refused": 1, "no_row": 1, "no_a2": 1, "act1_other_kind": 1, "regraded": 2}, res["by_reason"])
        self.assertEqual([("bundle", 20), ("card", 3)], sorted((r["kind"], r["floor"]) for r in self.records(self.out)))

    def test_the_stored_store_is_never_the_target(self):
        self.standard_store()
        with self.assertRaises(SystemExit):
            rt.regrade_store(self.src, self.src, grade="frontier", scorer=StubA2())
        with self.assertRaises(SystemExit):
            rt.regrade_store(self.src, self.src / "inside", grade="frontier", scorer=StubA2())
        with self.assertRaises(SystemExit):
            rt.regrade_store(self.src, self.out, grade="frontier", scorer=None)  # the frontier grade needs the Act 2 scorer
        with self.assertRaises(SystemExit):
            rt.regrade_store(self.src, self.out, grade="other", scorer=StubA2())
        self.assertFalse(self.out.exists())

    def test_a_dry_run_counts_and_writes_nothing(self):
        self.standard_store()
        res = self.regrade(dry_run=True)
        self.assertEqual(4, res["by_reason"]["regraded"])
        self.assertFalse(self.out.exists())

    def test_the_default_root_is_beside_the_store_under_the_frontier_name(self):
        self.assertEqual(Path("D:/spireline/referee/run-teacher-frontier/14"), rt.frontier_root_for(Path("D:/spireline/referee/run-teacher/14")))
        self.assertEqual(Path("D:/x/run-teacher-frontier/gen13-r0/13"), rt.frontier_root_for(Path("D:/x/run-teacher/gen13-r0/13")))
        with self.assertRaises(SystemExit):
            rt.frontier_root_for(Path("D:/x/elsewhere/14"))

    def test_the_command_runs_it(self):
        self.standard_store()
        stub = StubA2()
        model = self.tmp / "a2.pt"
        model.write_bytes(b"not a model: the scorer is the stub")
        with mock.patch.object(rt, "load_a2_scorer", return_value=stub), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(0, rt.main(["regrade", "--src", str(self.src), "--out", str(self.out), "--a2", str(model)]))
            with self.assertRaises(SystemExit):
                rt.main(["regrade", "--src", str(self.src), "--out", str(self.tmp / "g"), "--a2", str(self.tmp / "missing.pt")])
            self.assertEqual(0, rt.main(["regrade", "--src", str(self.src), "--out", str(self.tmp / "f"), "--grade", "floors"]))
        self.assertTrue((self.out / "manifest.json").is_file())

    def test_the_draws_end_and_floor_come_from_the_record_when_it_says_them_else_from_g(self):
        rec = graded("RUNAAAAAAA", 0, "card", floor=3, draws={0: [("door", 0.2), ("death", 8), ("win", None)], 1: [("door", 0.2)] * 3}, n=2)
        self.assertEqual([("door", 17, 0.2), ("death", 8, None), ("win", 53, None)], rt.draw_facts(rec, 0))
        later = graded("RUNAAAAAAA", 0, "route", floor=17, draws={0: [("door", 0.2)] * 2, 1: [("door", 0.2)] * 2}, n=2)
        self.assertEqual([("door", 33, 0.2)] * 2, rt.draw_facts(later, 0))  # the screen at the Act 1 boss floor is already before the Act 2 door
        said = dict(rec, draw_ends={"0": ["door", "death", "win"]}, draw_floors={"0": [17, 8, 53]})
        self.assertEqual(rt.draw_facts(rec, 0), rt.draw_facts(said, 0))
        odd = dict(rec, draw_ends={"0": ["door", "timeout", "win"]}, draw_floors={"0": [17, 9, 53]})
        self.assertEqual("timeout", rt.draw_facts(odd, 0)[1][0])


# ---------------------------------------------------------------------------------------------------------- the door rows


class DoorRowTests(unittest.TestCase):
    """The row of the state at the Act 1 boss door: the base vector of the state itself and the fight block H of its region, as an option's row is
    built, so lane A's A2 reads it like an option row. One per draw that reaches the door, float16, kept with A2 and the floors G."""

    def fakes(self, base_value=0.25):
        n_base = rt.ROWS_WIDTH - 8
        base = np.full(n_base, base_value, dtype="float32")
        calls = []

        def summary(bases, queries):
            calls.append((len(bases), list(queries)))
            return np.tile(np.arange(8, dtype="float32") / 10.0, (len(bases), 1)) + bases[:, :1]

        nets = SimpleNamespace(base_size=n_base, fight=SimpleNamespace(summary=summary))
        view = SimpleNamespace(act_region=0, boss=1)
        session = SimpleNamespace(view=lambda: view)
        player = SimpleNamespace(policy=SimpleNamespace(nets=nets))
        return base, calls, view, session, player

    def test_the_door_row_is_an_option_row_of_the_state_itself(self):
        base, calls, view, session, player = self.fakes()
        with (mock.patch("run_train.features.encode", return_value=base) as enc, mock.patch("run_train.features.fight_queries", return_value=["q1", "q2"]),
              mock.patch("run_train.features.BASE_SIZE", rt.ROWS_WIDTH - 8)):
            row = rt.door_row_of(player, session)
            enc.assert_called_once_with(view)
        self.assertEqual((rt.ROWS_WIDTH,), row.shape)
        self.assertEqual(np.float16, row.dtype)
        np.testing.assert_array_equal(row[: rt.ROWS_WIDTH - 8], base.astype("float16"))
        self.assertEqual([(1, ["q1", "q2"])], calls)  # one encode, one forward over the region's queries
        # the same bytes `human_prior_infer.option_rows` builds for an option whose afterstate is that state
        import human_prior_infer as hpi

        with (mock.patch("run_train.features.encode_afterstate", return_value=base), mock.patch("run_train.features.fight_queries", return_value=["q1", "q2"]),
              mock.patch("run_train.features.BASE_SIZE", rt.ROWS_WIDTH - 8)):
            want = hpi.option_rows(player.policy.nets, view, [SimpleNamespace(kind="x")])[0]
        np.testing.assert_array_equal(row, want.astype("float16"))

    def run_continuation(self, *, door_floor=17, end="door", rows=True, scorer="stub", grade="floors", p=0.4):
        base, calls, view, session, player = self.fakes()
        order = []
        closed = []
        sess = SimpleNamespace(apply=lambda o: None, close=lambda: (closed.append(1), order.append("close")))
        out = {"end": end, "floor": door_floor if end == "door" else 9, "door": {"hp": 50, "judge": {"p_death": p}} if end == "door" else None,
               "fights": 1, "moves": 1, "decisions": 1}
        door = np.arange(rt.ROWS_WIDTH, dtype="float16").reshape(1, -1)[0] / 1000
        scorer = StubA2() if scorer == "stub" else scorer

        def fake_door_row(pl, s):
            order.append("row")
            self.assertIs(s, sess)
            r = np.zeros(rt.ROWS_WIDTH, dtype="float16")
            r[0] = 0.25
            return r

        import hindsight

        with (mock.patch.object(hindsight, "clone_session", return_value=sess), mock.patch.object(rt, "run_to_door", return_value=out),
              mock.patch.object(rt, "door_row_of", side_effect=fake_door_row) as dr):
            c = rt.continuation(object(), SimpleNamespace(i=0), player, seed="S", ordinal=1, k=0, grade=grade, door_rows=rows, a2_scorer=scorer)
        return c, dr, order, scorer, door

    def test_an_act_1_door_draw_keeps_its_row_its_a2_and_the_floors_g(self):
        c, dr, order, scorer, _ = self.run_continuation()
        self.assertEqual(["row", "close"], order)  # the row is built while the session is open
        self.assertEqual(np.float16, c["door_row"].dtype)
        self.assertEqual((rt.ROWS_WIDTH,), c["door_row"].shape)
        self.assertAlmostEqual(0.25, c["a2"], places=6)
        self.assertEqual(round(rt.door_return("door", 17, 0.4), 4), c["G"])
        self.assertEqual(c["G"], c["g_floors"])
        self.assertEqual(1, len(scorer.calls))
        self.assertEqual((1, rt.ROWS_WIDTH), scorer.calls[0].shape)

    def test_the_grade_chooses_g_and_the_floors_g_stays_beside_it(self):
        c, *_ = self.run_continuation(grade="frontier")
        self.assertEqual(round(rt.frontier_return("door", 17, 0.4, a2=0.25), 4), c["G"])
        self.assertEqual(round(rt.door_return("door", 17, 0.4), 4), c["g_floors"])

    def test_no_row_at_an_act_2_door_a_death_or_when_not_asked(self):
        for kw in ({"door_floor": 33}, {"end": "death"}, {"rows": False}):
            c, dr, *_ = self.run_continuation(**kw)
            dr.assert_not_called()
            self.assertIsNone(c.get("door_row"), kw)
            self.assertIsNone(c.get("a2"), kw)

    def test_without_a_scorer_the_row_is_kept_and_a2_is_not_known(self):
        c, dr, *_ = self.run_continuation(scorer=None)
        dr.assert_called_once()
        self.assertIsNotNone(c["door_row"])
        self.assertIsNone(c["a2"])

    def test_a_frontier_continuation_without_a_scorer_has_no_return(self):
        c, *_ = self.run_continuation(grade="frontier", scorer=None)
        self.assertIsNone(c["G"])

    def test_decide_keeps_every_door_rows_with_its_option_draw_a2_and_floors_g(self):
        def cont(base, option, pl, *, seed, ordinal, k, cal=None, **kw):
            o = option.i
            row = np.full(rt.ROWS_WIDTH, 10 * o + k, dtype="float16")
            door = o == 0 or k == 0
            return {"k": k, "G": 20.0 + o + k if door else 5.0, "end": "door" if door else "death", "floor": 17 if door else 5,
                    "p_death": 0.3 if door else None, "door_hp": 50 if door else None, "fights": 1, "cpu": 0.0, "seconds": 0.0,
                    "door_row": row if door else None, "a2": 0.1 * (o + 1) if door else None, "g_floors": 20.0 + o + k if door else 5.0}

        rd = {"lam": 0.0, "screen": None, "logp": None, "logits": None, "groups": None, "scores": [None] * 2, "v": [None] * 2, "row_options": [0, 1],
              "rows": np.zeros((2, 4), dtype="float32")}
        door_out: list = []
        d = SimpleNamespace(type="card_reward", view=None, options=[SimpleNamespace(kind="card.take", scorable=False, i=i) for i in range(2)])
        view = SimpleNamespace(act_index=0, floor=5, hp=60, max_hp=80)
        with mock.patch.object(rt, "continuation", cont), mock.patch.object(rt, "screen_inputs", return_value=rd):
            rec = rt.decide(None, d, view, SimpleNamespace(index=0), SimpleNamespace(policy=None), seed="RUN", ordinal=7, k=2, max_options=4, k_max=None,
                            rows_out=[], door_out=door_out, a2_scorer=StubA2())
        self.assertEqual(1, len(door_out))
        rows = door_out[0]
        door = rec["door"]
        self.assertEqual([[0, 0], [1, 0], [0, 1]], door["draws"])  # draw-major, the options in order: the draws that reached the door
        self.assertEqual((3, rt.ROWS_WIDTH), rows.shape)
        self.assertEqual([0.0, 10.0, 1.0], [float(r[0]) for r in rows])
        self.assertEqual([0.1, 0.2, 0.1], door["a2"])
        self.assertEqual([20.0, 21.0, 21.0], door["g_floors"])
        self.assertEqual({"0": ["door", "door"], "1": ["door", "death"]}, rec["draw_ends"])
        self.assertEqual({"0": [17, 17], "1": [17, 5]}, rec["draw_floors"])

    def test_a_decision_with_no_door_draw_has_no_door_entry(self):
        def cont(base, option, pl, *, seed, ordinal, k, cal=None, **kw):
            return {"k": k, "G": 5.0, "end": "death", "floor": 5, "p_death": None, "door_hp": None, "fights": 1, "cpu": 0.0, "seconds": 0.0, "door_row": None, "a2": None,
                    "g_floors": 5.0}

        rd = {"lam": 0.0, "screen": None, "logp": None, "logits": None, "groups": None, "scores": [None] * 2, "v": [None] * 2, "row_options": [0, 1],
              "rows": np.zeros((2, 4), dtype="float32")}
        door_out: list = []
        d = SimpleNamespace(type="card_reward", view=None, options=[SimpleNamespace(kind="card.take", scorable=False, i=i) for i in range(2)])
        with mock.patch.object(rt, "continuation", cont), mock.patch.object(rt, "screen_inputs", return_value=rd):
            rec = rt.decide(None, d, SimpleNamespace(act_index=0, floor=5, hp=60, max_hp=80), SimpleNamespace(index=0), SimpleNamespace(policy=None),
                            seed="RUN", ordinal=7, k=2, max_options=4, rows_out=[], door_out=door_out)
        self.assertNotIn("door", rec)
        self.assertEqual([None], door_out)

    def test_a_record_made_without_door_rows_is_the_record_it_always_was(self):
        def cont(base, option, pl, *, seed, ordinal, k, cal=None):  # the old signature: decide passes nothing new when the door rows are off
            return {"k": k, "G": 20.0, "end": "door", "floor": 17, "p_death": 0.3, "door_hp": 50, "fights": 1, "cpu": 0.0, "seconds": 0.0}

        rd = {"lam": 0.0, "screen": None, "logp": None, "logits": None, "groups": None, "scores": [None] * 2, "v": [None] * 2, "row_options": [0, 1], "rows": None}
        d = SimpleNamespace(type="card_reward", view=None, options=[SimpleNamespace(kind="card.take", scorable=False, i=i) for i in range(2)])
        with mock.patch.object(rt, "continuation", cont), mock.patch.object(rt, "screen_inputs", return_value=rd):
            rec = rt.decide(None, d, SimpleNamespace(act_index=0, floor=5, hp=60, max_hp=80), SimpleNamespace(index=0), SimpleNamespace(policy=None),
                            seed="RUN", ordinal=7, k=2, max_options=4)
        for key in ("door", "draw_ends", "draw_floors", "grade"):
            self.assertNotIn(key, rec)


class DoorRowStoreTests(StoreCase):
    """The door rows in the store: one file per run (`<seed>.rows_door.f16`), cut back with the screens, `rows_door.npz` in the shard, routes made exact."""

    def screen(self, seed, ordinal, kind="card", floor=3, a2s=(0.3, 0.5), n_draws=2, **over):
        """A card/route screen of two options, K 2, whose draws all reach the Act 1 door at p 0.2, with door rows of the given A2."""
        draws = {0: [("door", 0.2)] * n_draws, 1: [("door", 0.2)] * n_draws}
        rec = graded(seed, ordinal, kind, floor=floor, draws=draws, n=2, **over)
        pairs = [[o, k] for k in range(n_draws) for o in (0, 1)]
        values = [a2s[o] for k in range(n_draws) for o in (0, 1)]
        rec["door"] = {"draws": pairs, "a2": [0.01] * len(pairs), "g_floors": [rec["g"][str(o)][k] for o, k in pairs]}
        return rec, a2_rows([0.1, 0.1]), door_rows_of(values)

    def test_the_rows_round_trip_and_the_door_pointer_counts_door_rows(self):
        rf = rt.RunFiles(self.src, "AAA")
        rf.load()
        for i in range(3):
            rec, rows, door = self.screen("AAA", i)
            rf.append(rec, rows, door=door)
        got = rt.RunFiles(self.src, "AAA").load()
        self.assertEqual([0, 4, 8], [got[o]["door"]["row0"] for o in sorted(got)])
        self.assertEqual((12 * rt.ROWS_WIDTH * 2), rf.door.stat().st_size)  # float16: two bytes a number

    def test_a_cut_door_file_loses_the_screens_it_cuts_and_both_files_stay_in_step(self):
        rf = rt.RunFiles(self.src, "BBB")
        rf.load()
        for i in range(3):
            rec, rows, door = self.screen("BBB", i)
            rf.append(rec, rows, door=door)
        size = rf.door.stat().st_size
        with open(rf.door, "r+b") as fh:
            fh.truncate(size - 10)
        again = rt.RunFiles(self.src, "BBB")
        self.assertEqual([0, 1], sorted(again.load()))
        self.assertEqual(8 * rt.ROWS_WIDTH * 2, rf.door.stat().st_size)  # cut back to the consistent prefix
        self.assertEqual(4 * rt.ROWS_WIDTH * 4, rf.rows.stat().st_size)  # and the option rows with it
        rf2 = rt.RunFiles(self.src, "BBB")
        rf2.load()
        rec, rows, door = self.screen("BBB", 2)
        rf2.append(rec, rows, door=door)  # a screen appended after the cut takes the cut screen's place
        self.assertEqual([0, 1, 2], sorted(rt.RunFiles(self.src, "BBB").load()))

    def test_a_replay_that_left_the_stored_screens_drops_their_door_rows_too(self):
        rf = rt.RunFiles(self.src, "CCC")
        rf.load()
        for i in range(3):
            rec, rows, door = self.screen("CCC", i)
            rf.append(rec, rows, door=door)
        rf = rt.RunFiles(self.src, "CCC")
        rf.load()
        rf.drop_from(1)
        self.assertEqual([0], sorted(rt.RunFiles(self.src, "CCC").load()))
        self.assertEqual(4 * rt.ROWS_WIDTH * 2, rf.door.stat().st_size)

    def test_a_run_of_an_older_build_has_no_door_file_and_its_screens_load(self):
        rf = rt.RunFiles(self.src, "DDD")
        rf.load()
        rf.append(graded("DDD", 0, "card", floor=3, draws={0: [("door", 0.2)] * 2, 1: [("door", 0.2)] * 2}, n=2), a2_rows([0.1, 0.1]))
        self.assertFalse(rf.door.exists())
        self.assertEqual([0], sorted(rt.RunFiles(self.src, "DDD").load()))
        with self.assertRaises(ValueError):  # the rows must be the ones the record names
            rec, rows, door = self.screen("DDD", 1)
            rt.RunFiles(self.src, "DDD").append(rec, rows, door=door[:3])

    def test_the_shard_holds_rows_door_npz_with_a2_and_the_floors_g_aligned_to_the_decisions(self):
        a = self.screen("AAA", 0)
        old = (graded("AAA", 1, "card", floor=3, draws={0: [("door", 0.2)] * 2, 1: [("door", 0.2)] * 2}, n=2), a2_rows([0.1, 0.1]))
        b = self.screen("AAA", 2, kind="route", a2s=(0.6, 0.7))
        doc = self.add_shard("J1", {"AAA": [a, old, b]})
        d = self.src / "shards" / "J1"
        z = np.load(d / "rows_door.npz")
        self.assertEqual({"door", "decision", "option", "draw", "a2", "g_floors"}, set(z.files))
        self.assertEqual(np.float16, z["door"].dtype)
        self.assertEqual((8, rt.ROWS_WIDTH), z["door"].shape)
        self.assertEqual([0, 0, 0, 0, 2, 2, 2, 2], z["decision"].tolist())  # the shard's decision lines: the old screen (line 1) has none
        self.assertEqual([0, 1, 0, 1, 0, 1, 0, 1], z["option"].tolist())
        self.assertEqual([0, 0, 1, 1, 0, 0, 1, 1], z["draw"].tolist())
        self.assertEqual(8, doc["counts"]["door_rows"])
        self.assertTrue(doc["door_rows"])
        recs = self.records(self.src)
        self.assertEqual([0, None, 4], [r.get("door", {}).get("row0") for r in recs])
        for r in recs:
            self.assertNotIn("_door", r)
        np.testing.assert_array_equal(z["door"][4:6, 0], np.array([0.6, 0.7], dtype="float16"))
        np.testing.assert_allclose(z["g_floors"][:2], [recs[0]["g"]["0"][0], recs[0]["g"]["1"][0]])

    def test_a_shard_with_no_door_rows_says_so_and_writes_no_file(self):
        doc = self.add_shard("J1", {"AAA": [(graded("AAA", 0, "card", floor=3, draws={0: [("door", 0.2)] * 2, 1: [("door", 0.2)] * 2}, n=2), a2_rows([0.1, 0.1]))]})
        self.assertFalse(doc["door_rows"])
        self.assertFalse((self.src / "shards" / "J1" / "rows_door.npz").exists())
        self.assertEqual(0, doc["counts"]["door_rows"])

    def test_a_floors_regrade_of_a_store_with_door_rows_keeps_them_byte_for_byte(self):
        s = "AAA"
        self.add_shard("J1", {s: [self.screen(s, 0), self.screen(s, 1, kind="route"), self.screen(s, 2, kind="card", floor=20, a2s=(0.4, 0.4))]})
        res = rt.regrade_store(self.src, self.out, grade="floors")
        self.assertEqual(0, res["mismatches"])
        a, b = self.src / "shards" / "J1", self.out / "shards" / "J1"
        self.assertEqual((a / "decisions.jsonl").read_bytes(), (b / "decisions.jsonl").read_bytes())
        za, zb = np.load(a / "rows_door.npz"), np.load(b / "rows_door.npz")
        self.assertEqual(set(za.files), set(zb.files))
        for key in za.files:
            np.testing.assert_array_equal(za[key], zb[key], err_msg=key)

    def test_door_rows_make_act_1_routes_and_events_exact_and_are_read_before_the_option_row(self):
        s = "AAA"
        route = self.screen(s, 0, kind="route", a2s=(0.3, 0.5))
        card = self.screen(s, 1, kind="card", a2s=(0.6, 0.8))
        event = self.screen(s, 2, kind="event", floor=5, a2s=(0.1, 0.9))
        self.add_shard("J1", {s: [route, card, event]})
        stub = StubA2()
        res = rt.regrade_store(self.src, self.out, grade="frontier", scorer=stub)
        recs = {r["kind"]: r for r in self.records(self.out)}
        self.assertEqual({"route", "card", "event"}, set(recs))  # routes and events are exact where the door rows are kept
        self.assertEqual(3, res["by_reason"]["regraded"])
        self.assertNotIn("act1_route_event", res["by_reason"])
        self.assertEqual({"door_rows": 3}, res["a2_from"])
        self.assertEqual("door_rows", recs["route"]["a2_from"])
        h = lambda x: float(np.float16(x))  # noqa: E731 -- the door rows are float16: the scorer reads the rounded number
        self.assertEqual([round(13.5 * 0.8 * h(0.3), 4)] * 2, recs["route"]["g"]["0"])  # (1 - p1) x A2 at the draw's own door
        self.assertEqual([round(13.5 * 0.8 * h(0.5), 4)] * 2, recs["route"]["g"]["1"])
        self.assertEqual([round(13.5 * 0.8 * h(0.6), 4)] * 2, recs["card"]["g"]["0"])  # a deck kind too: the door's row, not the option's (0.1)
        self.assertEqual([round(13.5 * 0.8 * h(0.1), 4)] * 2, recs["event"]["g"]["0"])
        self.assertEqual({0.3, 0.5, 0.6, 0.8, 0.1, 0.9}, {round(float(x), 1) for c in stub.calls for x in c[:, 0]})
        # the scorer's own number replaces the one stored at play (lane A's file may be a later fit)
        self.assertEqual([0.3, 0.5, 0.3, 0.5], [round(x, 4) for x in recs["route"]["door"]["a2"]])
        # the regraded shard keeps the door rows of the screens it keeps, renumbered
        z = np.load(self.out / "shards" / "J1" / "rows_door.npz")
        self.assertEqual(12, len(z["door"]))

    def test_a_door_draw_missing_from_the_door_rows_falls_back_to_the_option_row_or_the_skip(self):
        s = "AAA"
        rec, rows, door = self.screen(s, 0, kind="card", a2s=(0.6, 0.8))
        rec["door"]["draws"] = rec["door"]["draws"][:3]  # the door file lost a draw
        rec["door"]["a2"] = rec["door"]["a2"][:3]
        rec["door"]["g_floors"] = rec["door"]["g_floors"][:3]
        route, rows2, door2 = self.screen(s, 1, kind="route", a2s=(0.3, 0.5))
        route["door"]["draws"] = route["door"]["draws"][:3]
        route["door"]["a2"] = route["door"]["a2"][:3]
        route["door"]["g_floors"] = route["door"]["g_floors"][:3]
        self.add_shard("J1", {s: [(rec, rows, door[:3]), (route, rows2, door2[:3])]})
        res = rt.regrade_store(self.src, self.out, grade="frontier", scorer=StubA2())
        self.assertEqual({"option_row": 1}, res["a2_from"])  # the card falls back to its option row
        self.assertEqual(1, res["by_reason"]["act1_route_event"])  # the route has none to read
        self.assertEqual(1, len(self.records(self.out)))


# ---------------------------------------------------------------------------------------------------------- the unit's constants


class UnitConstantsTests(unittest.TestCase):
    """What a grade's unit sets, measured from a store's decision records: the guard's prior (the paired sd of one draw's option difference, by kind) and
    the noise of a centred option value, whose ratio between two grades on the same screens scales TAU_MIN."""

    @staticmethod
    def rec(kind, g, act=0):
        played = sorted(g)
        return {"kind": kind, "act": act, "played": played, "g": {str(o): v for o, v in g.items()}, "q": {str(o): rt.mean(v) for o, v in g.items()}, "k": len(g[played[0]])}

    def test_the_paired_sd_is_the_pooled_sd_of_one_draws_difference_between_two_options(self):
        # two options, K 4: the differences are +1, -1, +1, -1 (sample variance 4 / 3); a second decision of the same kind with differences 0, 0 -> variance 0
        a = self.rec("card", {0: [10.0, 10.0, 10.0, 10.0], 1: [9.0, 11.0, 9.0, 11.0]})
        b = self.rec("card", {0: [5.0, 6.0], 1: [5.0, 6.0]})
        got = rt.paired_sd_by_kind([a, b], min_decisions=1)
        self.assertAlmostEqual(got["card"], math.sqrt((3 * (4 / 3) + 1 * 0.0) / 4), places=4)
        self.assertEqual(4, got["_dof"]["card"])  # 3 degrees of freedom of the first decision, 1 of the second

    def test_a_decision_with_one_draw_or_one_option_adds_nothing_and_a_kind_under_the_cell_minimum_is_left_out(self):
        recs = [self.rec("card", {0: [10.0], 1: [9.0]}), self.rec("rest", {0: [1.0, 2.0, 3.0]}), self.rec("shop", {0: [1.0, 2.0], 1: [2.0, 4.0]})]
        got = rt.paired_sd_by_kind(recs, min_decisions=2)
        self.assertEqual({"_dof"}, set(got))
        got = rt.paired_sd_by_kind(recs, min_decisions=1)
        self.assertEqual({"shop"}, {k for k in got if k != "_dof"})

    def test_more_than_two_options_use_every_pair(self):
        a = self.rec("card", {0: [0.0, 0.0], 1: [1.0, -1.0], 2: [0.0, 0.0]})
        got = rt.paired_sd_by_kind([a], min_decisions=1)
        # pairs (0,1): differences -1, 1 (variance 2); (0,2): 0, 0; (1,2): 1, -1 (variance 2); pooled over three pairs of one degree of freedom each
        self.assertAlmostEqual(got["card"], math.sqrt((2 + 0 + 2) / 3), places=4)

    def test_the_unit_report_compares_two_grades_on_the_same_screens(self):
        floors = [self.rec("card", {0: [20.0, 24.0], 1: [22.0, 22.0]}), self.rec("card", {0: [10.0, 14.0], 1: [12.0, 12.0]})]
        frontier = [self.rec("card", {0: [1.0, 3.0], 1: [2.0, 2.0]}), self.rec("card", {0: [0.0, 2.0], 1: [1.0, 1.0]})]
        rep = rt.unit_constants(floors, frontier, min_decisions=1)
        self.assertAlmostEqual(rep["paired_sd"]["floors"]["card"] / rep["paired_sd"]["frontier"]["card"], 2.0)
        self.assertAlmostEqual(rep["noise_ratio"], rep["noise_sd"]["frontier"] / rep["noise_sd"]["floors"])
        self.assertAlmostEqual(rep["tau_min_suggested"], round(0.25 * rep["noise_ratio"], 3))


if __name__ == "__main__":
    unittest.main()
