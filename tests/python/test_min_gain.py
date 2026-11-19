"""The judge's minimum Act 3 gain as a recorded parameter (2026-11-19, D68 addendum 3; Act 3 week, lane 7): `--min-gain` on `gen0.py train`,
`promote` and `rejudge`, `min_gain` in tools/lab/keep_v2.py (`judge`, `judge_v3`, `route_a`, `route_a_v3`, `seq_call`, `seq_call_v3`,
`futility_routes`, `futility_call`), rl/gen_loop.py (`keep_v2_verdict`, `decide_v2`, `funnel`, `futility_check`, `promote`, `rejudge`,
`run_promote_subprocess`) and rl/gen0.py (the flags and their hand-off).

Constructed data only (keep_v2_fixtures): block A is 4,000 pairs whose deep_f gain at the first Act 3 boss is +0.975 per 100 with the 0.005 bound above 0
(the rule's +1.0 minimum refuses it, +0.5 does not); block B is 2,000 pairs with a gain of +0.7 and a bound below 0 (the minimum is not what stops it).
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import fight_exam as fe  # noqa: E402
import gen0  # noqa: E402
import gen_loop  # noqa: E402
import keep_v2_fixtures as fx  # noqa: E402
import test_gen12_plumbing as tg12  # noqa: E402
from test_gen_loop import stub_candidate  # noqa: E402

from tools.lab import keep_v2 as k2  # noqa: E402

BOOT = 5000
F_B3A = 2  # the frontier's index
# (pairs, candidate level, incumbent level): 0 died in Act 1, 1 beat the Act 1 boss, 2 the Act 2 boss, 3 the first Act 3 boss, 4 won.
# At the frontier B3a a level 3 against a level 2 is +1 deep_f unit (+100 for the pair), the reverse is -100.
A_SPEC = ((39, 3, 2), (3961, 1, 1))  # +39 net in 4,000 pairs: +0.975 per 100, SE about 0.16: the 0.005 bound is about +0.6
B_SPEC = ((60, 3, 2), (46, 2, 3), (1894, 1, 1))  # +14 net in 2,000 pairs: +0.7 per 100, SE about 0.5: the bound is below 0
C_SPEC = ((50, 3, 2), (42, 2, 3), (1908, 1, 1))  # +8 net in 2,000 pairs: +0.4 per 100, under +0.5 too


def arms(spec, prefix: str, seed: int = 21):
    return fx.arms(spec, seed=seed, prefix=prefix)


def judge(spec, prefix: str = "Z", rule: str = "v3", **kw) -> dict:
    _, c, b = arms(spec, prefix)
    return k2.judge(c, b, frontier=F_B3A, rule=rule, boot=BOOT, **kw)


def canon(x) -> str:
    return json.dumps(x, sort_keys=True)


class RuleTests(unittest.TestCase):
    """`keep_v2.judge` on blocks A and B: the minimum moves route A's `min_gain` check and nothing else."""

    @classmethod
    def setUpClass(cls):
        cls.a_default = judge(A_SPEC)
        cls.a_half = judge(A_SPEC, min_gain=0.5)
        cls.b_default = judge(B_SPEC)
        cls.b_half = judge(B_SPEC, min_gain=0.5)

    def test_the_default_is_the_rule_as_it_was_the_minimum_refuses_block_a(self):
        v = self.a_default
        self.assertAlmostEqual(v["deep"]["mean"], 0.975, places=9)
        self.assertGreater(v["deep"]["lower_route"], 0)  # the bound holds ...
        self.assertEqual(v["routes"]["A"]["checks"], {"deep_bound": True, "min_gain": False})  # ... the minimum does not
        self.assertEqual((v["route"], v["promote"], v["extend"], v["blocking"]), ("", False, False, []))
        self.assertEqual((v["min_gain"], v["routes"]["A"]["min_gain"]), (1.0, 1.0))  # the verdict's fields keep their place and value

    def test_no_argument_and_the_default_value_are_the_same_verdict_byte_for_byte(self):
        for rule in ("v3", "v2"):
            with self.subTest(rule=rule):
                self.assertEqual(canon(judge(A_SPEC, rule=rule)), canon(judge(A_SPEC, rule=rule, min_gain=k2.MIN_GAIN)))
                self.assertEqual(canon(judge(B_SPEC, rule=rule)), canon(judge(B_SPEC, rule=rule, min_gain=1.0)))

    def test_half_a_point_promotes_block_a_on_route_a_and_changes_nothing_else(self):
        v, d = self.a_half, self.a_default
        self.assertEqual((v["route"], v["promote"], v["extend"], v["blocking"]), ("A", True, False, []))
        self.assertEqual(v["routes"]["A"]["checks"], {"deep_bound": True, "min_gain": True})
        self.assertEqual((v["min_gain"], v["routes"]["A"]["min_gain"]), (0.5, 0.5))
        for key in ("deep", "floor", "kills", "wins", "crash", "collapse", "alpha"):  # the figures are the data's, not the minimum's
            self.assertEqual(canon(v[key]), canon(d[key]), key)

    def test_the_same_holds_under_rule_v2_which_also_has_route_a(self):
        d, v = judge(A_SPEC, rule="v2"), judge(A_SPEC, rule="v2", min_gain=0.5)
        self.assertEqual((d["route"], d["promote"], d["routes"]["A"]["checks"]["min_gain"]), ("", False, False))
        self.assertEqual((v["route"], v["promote"], v["routes"]["A"]["checks"]["min_gain"]), ("A", True, True))
        self.assertEqual((d["min_gain"], v["min_gain"], v["routes"]["A"]["min_gain"]), (1.0, 0.5, 0.5))

    def test_the_bound_check_is_untouched_block_b_is_not_promoted_at_half_a_point(self):
        for rule in ("v3", "v2"):
            with self.subTest(rule=rule):
                d, v = judge(B_SPEC, rule=rule), judge(B_SPEC, rule=rule, min_gain=0.5)
                self.assertAlmostEqual(v["deep"]["mean"], 0.7, places=9)
                self.assertLess(v["deep"]["lower_route"], 0)
                self.assertEqual((v["routes"]["A"]["checks"]["deep_bound"], v["routes"]["A"]["checks"]["min_gain"]), (False, True))
                self.assertEqual((d["promote"], v["promote"]), (False, False))
                self.assertFalse(v["routes"]["A"]["holds"])
        for min_gain in (0.0, 0.25, 0.5, 0.7):  # not at any minimum the data clears: only the bound decides, and it fails
            self.assertFalse(judge(B_SPEC, min_gain=min_gain)["promote"], min_gain)

    def test_the_extension_logic_uses_the_parameter(self):
        # v3 extends a final, first judgement whose minimum holds and whose bound does not: +0.7 does not reach +1.0 (refused) but reaches +0.5 (extended)
        self.assertEqual((self.b_default["route"], self.b_default["extend"], self.b_default["promote"]), ("", False, False))
        self.assertEqual((self.b_half["route"], self.b_half["extend"], self.b_half["promote"]), ("X", True, False))
        # the conditions of an extension are what they were: not twice, not on a block the funnel stopped, not past a failed guard
        again = judge(B_SPEC, min_gain=0.5, extended=True)
        self.assertEqual((again["route"], again["extend"], again["extended"]), ("", False, True))
        stopped = judge(B_SPEC, min_gain=0.5, final=False)
        self.assertEqual((stopped["route"], stopped["extend"]), ("", False))
        # a gain under the minimum is not extended at the lower minimum either (+0.4 against +0.5), and is at +0.4
        c = judge(C_SPEC, min_gain=0.5)
        self.assertAlmostEqual(c["deep"]["mean"], 0.4, places=9)
        self.assertEqual((c["route"], c["extend"]), ("", False))
        self.assertEqual((judge(C_SPEC, min_gain=0.4)["route"], judge(C_SPEC, min_gain=0.4)["extend"]), ("X", True))

    def test_the_summary_names_the_minimum_after_the_rule_only_when_it_is_not_the_default(self):
        self.assertTrue(k2.summary(self.a_default).startswith("v3 route none | "))
        self.assertTrue(k2.summary(self.a_half).startswith("v3 min gain 0.5 route A | "), k2.summary(self.a_half))
        self.assertTrue(k2.summary(self.b_half).startswith("v3 min gain 0.5 route X (extend) | "), k2.summary(self.b_half))
        v2_default, v2_half = judge(A_SPEC, rule="v2"), judge(A_SPEC, rule="v2", min_gain=0.5)
        self.assertTrue(k2.summary(v2_default).startswith("v2 route none | "))
        self.assertTrue(k2.summary(v2_half).startswith("v2 min gain 0.5 route A | "))
        self.assertNotIn("\t", k2.summary(self.a_half))
        legacy = {k: v for k, v in self.a_default.items() if k != "min_gain"}  # a verdict from before the field's place was used reads as the default
        self.assertTrue(k2.summary(legacy).startswith("v3 route none | "))

    def test_the_value_is_checked(self):
        self.assertEqual((k2.check_min_gain(0), k2.check_min_gain("0.5"), k2.check_min_gain(1)), (0.0, 0.5, 1.0))
        for bad in (-0.1, math.nan, math.inf, "x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                k2.check_min_gain(bad)

    def test_the_three_defaults_are_one_figure(self):
        self.assertEqual((k2.MIN_GAIN, gen_loop.MIN_GAIN_DEFAULT, gen0.MIN_GAIN_DEFAULT), (1.0, 1.0, 1.0))


class FunnelReadingTests(unittest.TestCase):
    """The looks and the futility stop ask for the same minimum the verdict does: a block the funnel refused at +1.0 may be alive at +0.5."""

    N = 2000

    def diffs(self, floor_mean: float, deep_mean: float, deep_se: float):
        floors = fx.std_diffs(floor_mean, 5.0, self.N, seed=3)
        deep = fx.std_diffs(deep_mean, deep_se * math.sqrt(self.N), self.N, seed=4)
        return list(floors), list(deep)

    def test_route_a_is_dead_against_the_minimum_asked_rule_v3(self):
        looks = gen_loop.seq_looks(self.N)
        floors, deep = self.diffs(0.0, 0.7, 0.1)  # deep_f +0.7, SE 0.1: the 99% upper bound is +0.93
        default = k2.seq_call(floors, deep, 3, looks, rule="v3")
        self.assertEqual(default["action"], "refuse")
        self.assertIn("route A is dead", default["reason"])
        self.assertIn("below +1.0", default["reason"])
        half = k2.seq_call(floors, deep, 3, looks, rule="v3", min_gain=0.5)
        self.assertEqual((half["action"], half["routes"]), ("eligible", ["A"]))  # z 7 clears the early z and +0.7 is above +0.5
        self.assertEqual(canon(k2.seq_call(floors, deep, 3, looks, rule="v3")), canon(k2.seq_call(floors, deep, 3, looks, rule="v3", min_gain=1.0)))
        floors2, deep2 = self.diffs(0.0, 0.45, 0.1)
        self.assertEqual(k2.seq_call(floors2, deep2, 3, looks, rule="v3", min_gain=0.5)["action"], "continue")  # alive (upper bound 0.68) but not eligible
        dead = k2.seq_call(floors2, deep2, 3, looks, rule="v3", min_gain=0.8)
        self.assertEqual(dead["action"], "refuse")
        self.assertIn("below +0.8", dead["reason"])

    def test_route_a_is_dead_against_the_minimum_asked_rule_v2(self):
        looks = gen_loop.seq_looks(self.N)
        floors, deep = self.diffs(-1.0, 0.7, 0.1)  # the floor route is dead (99% upper bound -0.74); deep_f's 95% upper bound is +0.86
        self.assertEqual(k2.seq_call(floors, deep, 3, looks)["action"], "refuse")
        self.assertEqual(k2.seq_call(floors, deep, 3, looks, min_gain=0.5)["action"], "continue")
        self.assertEqual(canon(k2.seq_call(floors, deep, 3, looks)), canon(k2.seq_call(floors, deep, 3, looks, min_gain=1.0)))
        up_floors, up_deep = self.diffs(0.0, 0.7, 0.1)
        self.assertNotIn("A", k2.seq_call(up_floors, up_deep, 3, looks)["routes"])  # +0.7 is under +1.0
        self.assertIn("A", k2.seq_call(up_floors, up_deep, 3, looks, min_gain=0.5)["routes"])

    def test_the_futility_stop_asks_for_the_minimum(self):
        deep = {"mean": 0.3, "se": 0.3, "upper95": 0.79}  # the 0.005 critical gain at 1,000 of 3,000 seeds is 0.45
        floor = {"mean": 0.0, "se": 0.3, "upper95": 0.5}
        default = k2.futility_routes(floor, deep, 1000, 3000, routes=("A",))
        half = k2.futility_routes(floor, deep, 1000, 3000, routes=("A",), min_gain=0.5)
        self.assertEqual((default["routes"]["A"]["needs"], default["futile"]), (1.0, True))
        self.assertAlmostEqual(half["routes"]["A"]["needs"], 0.5, places=9)  # max(0.5, 0.45)
        self.assertEqual(half["futile"], False)
        ext_default = k2.futility_routes(floor, deep, 1000, 3000, routes=("A",), extend_possible=True)
        ext_half = k2.futility_routes(floor, deep, 1000, 3000, routes=("A",), extend_possible=True, min_gain=0.5)
        self.assertEqual((ext_default["routes"]["A"]["needs"], ext_default["futile"]), (1.0, True))  # the extension's gain is the parameter itself
        self.assertEqual((ext_half["routes"]["A"]["needs"], ext_half["futile"]), (0.5, False))

    def test_the_funnel_and_the_futility_check_hand_the_minimum_to_the_rule(self):
        seeds, c, b = arms(B_SPEC, "Y")
        seen: list[dict] = []
        real = k2.seq_call

        def spy(*args, **kw):
            seen.append(dict(kw))
            return real(*args, **kw)

        def judge_cb(cr, br):
            keep = gen_loop.keep_v2_verdict(cr, br, [], cand_gen=4, cand_sha="x", block="Y", frontier_info={"index": F_B3A, "name": "B3a"},
                                            boot=300, rule="v3", final=False, min_gain=0.5)
            return gen_loop.compare_arms(cr, br, gen_loop.ALPHA, 300, None, keep=keep)

        with mock.patch.object(k2, "seq_call", spy):
            gen_loop.funnel(seeds, lambda n: (c, b), judge_cb, keep_rule="v3", frontier=F_B3A, min_gain=0.5)
            self.assertTrue(seen and all(kw.get("min_gain") == 0.5 and kw.get("rule") == "v3" for kw in seen), seen[:2])
            seen.clear()
            gen_loop.funnel(seeds, lambda n: (c, b), judge_cb, keep_rule="v3", frontier=F_B3A)
            self.assertTrue(seen and all("min_gain" not in kw for kw in seen))  # the default call is the call it was
        with mock.patch.object(k2, "futility_call", return_value={"futile": False, "routes": {}, "text": ""}) as fc:
            gen_loop.futility_check([0.0] * 5, [0.0] * 5, 100, v2=True, rule="v3", extend_possible=True, min_gain=0.5)
            self.assertEqual(fc.call_args[1]["min_gain"], 0.5)
            gen_loop.futility_check([0.0] * 5, [0.0] * 5, 100, v2=True, rule="v2", min_gain=0.5)
            self.assertEqual(fc.call_args[1]["min_gain"], 0.5)
            gen_loop.futility_check([0.0] * 5, [0.0] * 5, 100, v2=True, rule="v3", extend_possible=True)
            self.assertNotIn("min_gain", fc.call_args[1])


class VerdictPlumbingTests(unittest.TestCase):
    def verdict(self, **kw):
        _, c, b = arms(A_SPEC, "K")
        return gen_loop.keep_v2_verdict(c, b, [], cand_gen=4, cand_sha="x", block="K", frontier_info={"index": F_B3A, "name": "B3a"}, boot=BOOT, rule="v3", **kw), c, b

    def test_the_decision_names_the_minimum_only_when_it_is_not_the_default(self):
        (default, c, b), (half, _, _) = self.verdict(), self.verdict(min_gain=0.5)
        d0, d1 = gen_loop.decide_v2(default), gen_loop.decide_v2(half)
        self.assertNotIn("min_gain", d0)  # a default decision is byte-for-byte what it was
        self.assertEqual(set(d0), {"checks", "wins_co_primary", "promote", "attempt_advisory", "alpha", "rule", "route", "frontier", "review", "extend",
                                   "advisory_checks"})
        self.assertEqual((d0["promote"], d0["route"]), (False, ""))
        self.assertEqual(d1["min_gain"], 0.5)
        self.assertEqual((d1["promote"], d1["route"]), (True, "A"))
        self.assertEqual({k: v for k, v in d1.items() if k not in ("min_gain", "promote", "route", "checks")},
                         {k: v for k, v in d0.items() if k not in ("promote", "route", "checks")})
        self.assertEqual({k: v for k, v in d1["checks"].items() if k != "route"}, {k: v for k, v in d0["checks"].items() if k != "route"})
        self.assertEqual((d0["checks"]["route"], d1["checks"]["route"]), (False, True))
        self.assertEqual(canon(gen_loop.compare_arms(c, b, gen_loop.ALPHA, BOOT, None, keep=default)["decision"]), canon(d0))
        self.assertEqual(default["min_gain"], 1.0)
        self.assertEqual(half["min_gain"], 0.5)

    def test_a_minimum_that_is_not_a_number_of_at_least_zero_is_refused(self):
        for bad in (-1.0, math.nan, math.inf):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.verdict(min_gain=bad)

    def test_min_gain_of_reads_the_rule(self):
        self.assertEqual((gen_loop.min_gain_of(None, "v3"), gen_loop.min_gain_of(0.5, "v3"), gen_loop.min_gain_of(0.5, "v2")), (1.0, 0.5, 0.5))
        self.assertIsNone(gen_loop.min_gain_of(None, "v1"))
        self.assertIsNone(gen_loop.min_gain_of(1.0, "v1"))  # the default is not an error under v1
        with self.assertRaisesRegex(ValueError, "keep rules"):
            gen_loop.min_gain_of(0.5, "v1")  # not a silent no-op
        with self.assertRaises(ValueError):
            gen_loop.min_gain_of(-1, "v3")


class RejudgeAndPromoteTests(unittest.TestCase):
    """`rejudge` and `promote` on block A's stored or stubbed arms: the verdict, the decision, promotion.json and the ledger row."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ("EXAM_DIR", "LOOP_ROOT"):  # the judgement's human-bar line reads the gym's folder: not the real one in a test
            patcher = mock.patch.object(fe, name, self.root / "no-gym")
            patcher.start()
            self.addCleanup(patcher.stop)
        self.agents = self.root / "agent"
        self.agents.mkdir()
        self.seeds, self.c, self.b = arms(A_SPEC, "K")
        (self.root / "seeds").mkdir()
        self.seed_file = self.root / "seeds" / "Etest-r1.json"
        self.seed_file.write_text(json.dumps({"seeds": self.seeds}))
        # the lineage: generation 3 promoted on Eprev with the first Act 3 boss beaten 3 times in 100, so the frontier is B3a (as the rule's lineage reads it)
        prev = [fx.run_row(f"P{i:09d}", 3 if i < 60 else 2 if i < 300 else 1 if i < 1000 else 0, np.random.default_rng(i)) for i in range(2000)]
        self.write_arm(self.agents / "gen3" / "candidate", "Eprev", prev)
        gen_loop.append_promotion(self.agents / "promotions.tsv",
                                  {"cand_gen": 3, "base_gen": 2, "seed_block": "Eprev", "seeds": 2000, "boss_cand": 1000, "boss_base": 900,
                                   "decision": "promoted", "cand_sha": "sha-3", "cand_dir": "candidate", "attempt": 1, "drift": "clear"})
        self.cand = stub_candidate(self.root, "c1", "sha-a", generation=4)
        self.write_arm(self.cand, "Etest-r1", self.c)
        self.write_arm(self.agents / "gen0", "Etest-r1", self.b)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def write_arm(folder: Path, name: str, rows: list[dict]) -> None:
        arm = folder / "arms" / f"{name}-a3-v1"
        arm.mkdir(parents=True)
        (arm / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def rejudge(self, out: str, **kw):
        with mock.patch.object(gen0, "run_block", side_effect=AssertionError("must not play again")):
            return gen_loop.rejudge(self.cand, None, self.seed_file, agent_root=self.agents, note="D68 addendum 3", boot=BOOT, keep_rule="v3",
                                    out=self.root / out, **kw)

    def ledger_row(self, out: str) -> dict:
        return gen_loop.read_promotions(self.root / out / "promotions.tsv")[-1]

    def test_rejudge_refuses_block_a_by_default_and_promotes_it_at_half_a_point(self):
        d0 = self.rejudge("d0")
        self.assertEqual((d0["decision_text"], d0["decision"]["promote"], d0["decision"]["route"]), ("rejected", False, ""))
        self.assertNotIn("min_gain", d0["decision"])
        self.assertEqual((d0["keep_v2"]["min_gain"], d0["keep_v2"]["routes"]["A"]["min_gain"]), (1.0, 1.0))
        row0 = self.ledger_row("d0")
        self.assertTrue(row0["guard"].startswith("v3 route none | "), row0["guard"])
        self.assertFalse((self.root / "d0" / "current.json").exists())
        d1 = self.rejudge("d1", min_gain=0.5)
        self.assertEqual((d1["decision_text"], d1["decision"]["promote"], d1["decision"]["route"], d1["decision"]["min_gain"]), ("promoted", True, "A", 0.5))
        self.assertEqual((d1["keep_v2"]["min_gain"], d1["keep_v2"]["routes"]["A"]["min_gain"], d1["keep_v2"]["frontier"]["name"]), (0.5, 0.5, "B3a"))
        row1 = self.ledger_row("d1")
        self.assertEqual((row1["decision"], row1["rule"], row1["route"]), ("promoted", "v3", "A"))
        self.assertTrue(row1["guard"].startswith("v3 min gain 0.5 route A | "), row1["guard"])  # the note after the rule's name
        self.assertTrue((self.root / "d1" / "current.json").is_file())
        self.assertEqual(json.loads((self.root / "d1" / "promotion.json").read_text())["decision"]["min_gain"], 0.5)
        # the same ledger columns, the same figures: only the minimum's consequences differ
        self.assertEqual(tuple(row1), tuple(row0))
        for key in ("d_floor", "floor_lower", "boss_cand", "boss_base", "d_deep", "deep_lower", "p_deep", "floor_lower95", "k2_cand", "k2_base"):
            self.assertEqual(row1[key], row0[key], key)

    def test_the_default_path_writes_the_promotion_json_it_wrote_before(self):
        none, one = self.rejudge("n0"), self.rejudge("n1", min_gain=1.0)
        for doc in (none, one):
            doc.pop("written")
        self.assertEqual(canon(none), canon(one))
        self.assertEqual(json.loads((self.root / "n0" / "promotion.json").read_text()).keys(), none.keys() | {"written"})
        self.assertEqual(set(none["decision"]), {"checks", "wins_co_primary", "promote", "attempt_advisory", "alpha", "rule", "route", "frontier", "review",
                                                  "extend", "advisory_checks"})
        self.assertEqual(self.ledger_row("n0")["guard"], self.ledger_row("n1")["guard"])

    def test_under_v1_only_the_default_is_accepted(self):
        with self.assertRaisesRegex(ValueError, "keep rules"):
            gen_loop.rejudge(self.cand, None, self.seed_file, agent_root=self.agents, note="n", boot=BOOT, keep_rule="v1", min_gain=0.5, out=self.root / "v1")

    def promote(self, **kw):
        played = iter([self.c, self.b])  # the candidate's arm is played first, then the incumbent's: the stub answers each in turn
        with mock.patch.object(gen_loop, "run_arm", side_effect=lambda *a, **k: next(played)), contextlib.redirect_stdout(io.StringIO()):
            return gen_loop.promote(self.cand, None, self.seed_file, agent_root=self.agents, boot=BOOT, keep_rule="v3", **kw)

    def test_promote_hands_the_minimum_to_the_verdict_and_the_ledger(self):
        doc = self.promote(min_gain=0.5)
        self.assertEqual((doc["decision_text"], doc["decision"]["min_gain"], doc["keep_v2"]["min_gain"]), ("promoted", 0.5, 0.5))
        row = gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]
        self.assertTrue(row["guard"].startswith("v3 min gain 0.5 route A | "), row["guard"])
        self.assertTrue((self.agents / "current.json").is_file())

    def test_promote_by_default_refuses_block_a(self):
        doc = self.promote()
        self.assertEqual((doc["decision_text"], doc["keep_v2"]["min_gain"]), ("rejected", 1.0))
        self.assertNotIn("min_gain", doc["decision"])
        row = gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]
        self.assertTrue(row["guard"].startswith("v3 route none | "), row["guard"])
        self.assertFalse((self.agents / "current.json").exists())


class CommandLineTests(unittest.TestCase):
    PARSE = {"train": ["--marker-dir", "m", "--shards-root", "s", "--agent-root", "a"],
             "promote": ["--cand", "c", "--agent-root", "a"],
             "rejudge": ["--cand", "c", "--seeds-file", "s", "--agent-root", "a", "--note", "n"]}

    def test_every_entry_point_that_reaches_the_judge_has_the_flag_default_one(self):
        p = gen0.build_parser()
        for cmd, extra in self.PARSE.items():
            with self.subTest(cmd=cmd):
                self.assertEqual(p.parse_args([cmd, *extra]).min_gain, 1.0)
                self.assertEqual(p.parse_args([cmd, *extra, "--min-gain", "0.5"]).min_gain, 0.5)
                self.assertEqual(p.parse_args([cmd, *extra, "--min-gain=0"]).min_gain, 0.0)

    def test_a_value_that_is_not_a_number_of_at_least_zero_is_refused_by_the_parser(self):
        p = gen0.build_parser()
        for cmd, extra in self.PARSE.items():
            for bad in ("-0.5", "nan", "inf", "abc"):
                with self.subTest(cmd=cmd, bad=bad), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    p.parse_args([cmd, *extra, "--min-gain", bad])

    def test_the_flag_text_the_lab_command_takes_parses_with_the_trains_flags(self):
        a = gen0.build_parser().parse_args([*["train"], *self.PARSE["train"], "--keep-rule", "v3", "--min-gain", "0.5", "--sequential", "--promote"])
        self.assertEqual((a.keep_rule, a.min_gain, a.sequential, a.promote), ("v3", 0.5, True, True))

    def test_promote_cmd_and_rejudge_cmd_pass_the_minimum_on_and_nothing_at_the_default(self):
        doc = {k: 0 for k in ("decision_text", "floor", "boss_kill", "wins")}
        for argv, want in (([], None), (["--min-gain", "1.0"], None), (["--min-gain", "0.5"], 0.5)):
            a = gen0.build_parser().parse_args(["promote", *self.PARSE["promote"], *argv])
            with mock.patch.object(gen_loop, "promote", return_value=doc) as pr, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(gen0.promote_cmd(a), 0)
            self.assertEqual(pr.call_args[1]["min_gain"], want, argv)
        full = {k: 0 for k in ("decision_text", "n_seeds", "floor", "boss_kill", "wins", "decision")}
        full["rejudged"] = {}
        for argv, want in (([], None), (["--min-gain", "0.5"], 0.5)):
            a = gen0.build_parser().parse_args(["rejudge", *self.PARSE["rejudge"], *argv])
            with mock.patch.object(gen_loop, "rejudge", return_value=full) as rj, mock.patch.object(gen0, "_setup"), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(gen0.rejudge_cmd(a), 0)
            self.assertEqual(rj.call_args[1]["min_gain"], want, argv)

    def test_the_subprocess_of_the_train_job_passes_the_minimum_only_when_it_is_not_the_default(self):
        ran: list[list[str]] = []
        with mock.patch.object(gen_loop.subprocess, "run", lambda cmd, check=False: ran.append(cmd) or SimpleNamespace(returncode=0)):
            for kw in ({}, {"min_gain": None}, {"min_gain": 1.0}, {"min_gain": 0.5}, {"min_gain": 0.25}):
                gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None, keep_rule="v3", **kw)
        self.assertTrue(all("--min-gain" not in cmd for cmd in ran[:3]))
        self.assertEqual([cmd[cmd.index("--min-gain") + 1] for cmd in ran[3:]], ["0.5", "0.25"])
        parsed = gen0.build_parser().parse_args(ran[3][2:])  # the child's own parser reads what the job wrote
        self.assertEqual((parsed.min_gain, parsed.keep_rule), (0.5, "v3"))


class TrainJobTests(unittest.TestCase):
    """`gen0.py train --promote` with the judging child stubbed (the extension tests' harness): the minimum reaches the first block and the second."""

    setUp, tearDown, build = tg12.PlacementTests.setUp, tg12.PlacementTests.tearDown, tg12.PlacementTests.build

    EXTEND = {"seed_block": "E7-day", "decision": {"promote": False, "extend": True, "route": "X"}, "decision_text": "extend"}

    def train(self, *flags: str, verdicts: list[dict | None], rule: str = "v3"):
        self.assertEqual(0, self.build())
        cand = self.agents / "gen7" / "candidate"
        seeds = self.root / "seeds"
        seeds.mkdir()
        calls = []

        def child(*args, **kw):
            calls.append((args, kw))
            verdict = verdicts[len(calls) - 1]
            if verdict is not None:
                (cand / "promotion.json").write_text(json.dumps(verdict))
            return 0

        argv = ["train", "--marker-dir", str(self.marker_dir), "--shards-root", str(self.root), "--agent-root", str(self.agents), "--promote",
                "--emu-python", "python", "--seeds-file", str(seeds / "E{k}-{date}.json"), "--review-guard", "--sequential", "--keep-rule", rule, *flags]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(gen_loop, "train_generation"), mock.patch.object(gen_loop, "run_promote_subprocess", side_effect=child), \
                mock.patch.object(gen_loop, "COMMITTED_SEEDS", seeds), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = gen0.main(argv)
        return rc, calls, out.getvalue() + err.getvalue()

    def test_the_minimum_reaches_the_first_block_and_the_extension_block(self):
        rc, calls, _ = self.train("--min-gain", "0.5", verdicts=[self.EXTEND, None])
        self.assertEqual((rc, len(calls)), (0, 2))
        self.assertEqual([kw["min_gain"] for _, kw in calls], [0.5, 0.5])
        self.assertEqual(calls[1][1]["pool_blocks"], ("E7-day",))  # the second look, judged under the same minimum

    def test_the_default_hands_nothing_on(self):
        rc, calls, _ = self.train(verdicts=[None])
        self.assertEqual((rc, len(calls)), (0, 1))
        self.assertIsNone(calls[0][1]["min_gain"])

    def test_the_default_named_hands_nothing_on_either(self):
        rc, calls, _ = self.train("--min-gain", "1.0", verdicts=[None])
        self.assertEqual((rc, len(calls)), (0, 1))
        self.assertIsNone(calls[0][1]["min_gain"])

    def test_a_minimum_under_rule_v1_stops_before_anything_is_trained(self):
        with mock.patch.dict(os.environ, {gen_loop.KEEP_RULE_FLAG: ""}):
            rc, calls, text = self.train("--min-gain", "0.5", verdicts=[None], rule="v1")
        self.assertEqual((rc, calls), (2, []))
        self.assertIn("keep rules", text)
        self.assertIn("the marker stays", text)


if __name__ == "__main__":
    unittest.main()
