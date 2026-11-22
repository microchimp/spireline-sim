"""The keep rule v3 (D66, 2026-11-18): tools/lab/keep_v2.py (`judge_v3`, `seq_call_v3`, the futility's extension need), rl/gen_loop.py (the
verdict plumbing, the pooled `rejudge` and `promote`, the funnel's pooled prefix) and rl/gen0.py (`--keep-rule v3`, `--pool-block`);
docs/rl/KEEP_TEST_V3.md.

Constructed data only: synthetic paired arms built from counts (`keep_v2_fixtures`) whose floors are then set to an exact mean difference, and
differences with an exact mean and SD. The v3d-like block is `candidate-cheap-v3d`'s block of 2026-11-18 rebuilt from its counts: the Act 1 boss
1,259 against 1,473, the first Act 3 boss and the win 119 against 90 together (deep_f +1.45 per 100, SE about 0.74), floors -1.95."""

from __future__ import annotations

import contextlib
import io
import json
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

BOOT = 20000
# (pairs, candidate level, incumbent level); the levels are keep_v2_fixtures': 0 died in Act 1, 1 beat the Act 1 boss, 2 the Act 2 boss, 3 the
# first Act 3 boss, 4 won. At the frontier B3a deep_f is 1 for level 3 and 2 for a win: +111 +4x2 -88 -1x2 = +29 net in 2,000 pairs = +1.45 per 100.
V3D_SPEC = ((111, 3, 2), (4, 4, 2), (88, 2, 3), (1, 2, 4), (828, 1, 1), (441, 0, 1), (227, 1, 0), (300, 0, 0))
FLAT_SPEC = ((100, 3, 2), (100, 2, 3), (1000, 1, 1), (400, 0, 1), (400, 1, 0))  # no deep_f difference at all
SMALL_SPEC = ((105, 3, 2), (95, 2, 3), (1000, 1, 1), (400, 0, 1), (400, 1, 0))  # +10 net in 2,000 pairs: +0.5 per 100, under the +1.0 the rule asks
LINEAGE = [(586, 800), (1450, 2000), (366, 500)]  # the lineage's Act 1 kills and runs the real drift check of block r1 read
F_B3A = 2  # the frontier's index


def tune_floor(cand: list[dict], base: list[dict], mean: float) -> None:
    """Move candidate floors by one, a run on floor 28 to 46 at a time (so no Act 1 boss is won or lost), until the paired mean floor
    difference is exactly `mean` (to the nearest 1/n)."""
    n = len(cand)
    delta = round(mean * n) - sum(c["floor"] - b["floor"] for c, b in zip(cand, base, strict=True))
    movable = [r for r in cand if 28 <= r["floor"] <= 46 and not r["win"]]
    step = 1 if delta > 0 else -1
    i = 0
    while delta:
        movable[i % len(movable)]["floor"] += step
        delta -= step
        i += 1


def block(spec=V3D_SPEC, floor_mean: float = -1.95, prefix: str = "K", seed: int = 7) -> tuple[list[str], list[dict], list[dict]]:
    seeds, cand, base = fx.arms(spec, seed=seed, prefix=prefix)
    tune_floor(cand, base, floor_mean)
    return seeds, cand, base


def v3(cand, base, **kw) -> dict:
    return k2.judge(cand, base, frontier=F_B3A, rule="v3", boot=kw.pop("boot", BOOT), **kw)


class JudgeV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s1, cls.c1, cls.b1 = block(prefix="K", seed=7)
        cls.s2, cls.c2, cls.b2 = block(prefix="L", seed=8)
        cls.one = v3(cls.c1, cls.b1)
        cls.pooled = v3(cls.c1 + cls.c2, cls.b1 + cls.b2, extended=True)

    def test_the_v3d_like_block_is_extended_where_v2_refused_it(self):
        v = self.one
        self.assertEqual(v["rule"], k2.RULE_V3)
        self.assertEqual(v["kills"]["B1"], [1259, 1473])
        self.assertEqual(v["kills"]["B3a"], [115, 89])
        self.assertEqual(v["kills"]["win"], [4, 1])
        self.assertAlmostEqual(v["deep"]["mean"], 1.45, places=9)
        self.assertTrue(0.65 < v["deep"]["se"] < 0.80, v["deep"]["se"])  # the SE of the real block was 0.72
        self.assertLess(v["deep"]["lower_route"], 0)  # the 0.005 bound is not above 0 ...
        self.assertAlmostEqual(v["floor"]["mean"], -1.95, places=9)
        self.assertEqual(v["routes"]["A"]["checks"], {"deep_bound": False, "min_gain": True})  # ... and no floor margin is asked of route A
        self.assertEqual(sorted(v["routes"]), ["A"])  # route F is gone
        self.assertTrue(v["extend"])
        self.assertEqual(v["route"], "X")
        self.assertFalse(v["promote"])
        self.assertEqual(v["blocking"], [])  # an extension is not a refusal: nothing blocks it
        self.assertFalse(v["checks"]["route"])
        self.assertEqual((v["extended"], v["final"]), (False, True))
        # v2 refused the same arms: the bound, and the floor margin on top
        old = k2.judge(self.c1, self.b1, frontier=F_B3A, boot=BOOT)
        self.assertEqual((old["route"], old["promote"]), ("", False))
        self.assertEqual(old["routes"]["A"]["checks"], {"deep_bound": False, "min_gain": True, "floor_margin": False})
        self.assertNotIn("extend", old)
        # the ledger and the guard line
        cells = k2.ledger_cells(v)
        self.assertEqual((cells["rule"], cells["frontier"], cells["route"]), ("v3", "B3a", "X"))
        self.assertEqual(set(cells), {"rule", "frontier", "route", "k2_cand", "k2_base", "d_deep", "deep_lower", "p_deep", "floor_lower95"})
        line = k2.summary(v)
        self.assertTrue(line.startswith("v3 route X (extend) | frontier B3a | deep +1.45"), line)
        for part in ("floor -1.95 ±", " (99% upper", "collapse below -5.0: ok", "drift skipped (advisory)", "act1 review: kill"):
            self.assertIn(part, line)
        self.assertNotIn("\t", line)
        self.assertEqual(k2.ledger_cells(old)["rule"], "v2")

    def test_the_same_effect_on_4000_pooled_pairs_promotes_and_v2_would_not_have(self):
        v = self.pooled
        self.assertEqual(v["kills"]["B1"], [2518, 2946])
        self.assertAlmostEqual(v["deep"]["mean"], 1.45, places=9)
        self.assertTrue(0.45 < v["deep"]["se"] < 0.60, v["deep"]["se"])
        self.assertGreater(v["deep"]["lower_route"], 0)  # the 0.005 bound is above 0 now
        self.assertLess(v["deep"]["lower_route"], 1.45)
        self.assertEqual((v["route"], v["promote"], v["extend"], v["blocking"]), ("A", True, False, []))
        self.assertAlmostEqual(v["floor"]["mean"], -1.95, places=9)  # floors -1.95: recorded, decide nothing
        self.assertLess(v["floor"]["lower95"], -1.0)
        self.assertEqual(k2.ledger_cells(v)["route"], "A")
        old = k2.judge(self.c1 + self.c2, self.b1 + self.b2, frontier=F_B3A, boot=BOOT)
        self.assertEqual(old["routes"]["A"]["checks"], {"deep_bound": True, "min_gain": True, "floor_margin": False})  # v2's floor margin refused it
        self.assertFalse(old["promote"])

    def test_a_candidate_that_was_extended_is_never_extended_again(self):
        again = v3(self.c1, self.b1, extended=True)  # the same block, but the candidate was extended before (or a second block is pooled)
        self.assertEqual((again["route"], again["extend"], again["promote"], again["blocking"]), ("", False, False, []))
        self.assertTrue(again["extended"])
        # a judgement the funnel stopped early is not final: it refuses, it does not extend
        part = v3(self.c1, self.b1, final=False)
        self.assertEqual((part["route"], part["extend"], part["promote"]), ("", False, False))

    def test_floors_decide_nothing_but_a_collapse(self):
        deep = v3(self.c1 + self.c2, self.b1 + self.b2, extended=True)
        self.assertTrue(deep["checks"]["floor_collapse"])
        self.assertGreater(deep["collapse"]["upper99"], k2.COLLAPSE_FLOORS)
        # a floor 4 below the incumbent's, which v2's margin refuses a hundred times over, still promotes a deeper player
        _, c, b = block(prefix="K", seed=7, floor_mean=-4.0)
        _, c2, b2 = block(prefix="L", seed=8, floor_mean=-4.0)
        low = v3(c + c2, b + b2, extended=True)
        self.assertAlmostEqual(low["floor"]["mean"], -4.0, places=9)
        self.assertEqual((low["route"], low["promote"], low["blocking"]), ("A", True, []))
        # a floor gain promotes nothing: route F is gone (a flat deep_f with a big floor gain)
        _, c3, b3 = block(FLAT_SPEC, floor_mean=3.0, prefix="M", seed=9)
        gain = v3(c3, b3)
        self.assertEqual((gain["route"], gain["promote"], gain["extend"]), ("", False, False))
        self.assertGreater(gain["floor"]["lower_route"], 0)  # today's floor primary would have promoted it
        self.assertEqual(sorted(gain["routes"]), ["A"])

    def test_a_collapse_refuses_as_breakage_and_extends_nothing(self):
        _, c, b = block(prefix="K", seed=7, floor_mean=-6.5)
        v = v3(c, b)
        self.assertLess(v["collapse"]["upper99"], -5.0)
        self.assertTrue(v["collapse"]["collapsed"])
        self.assertEqual((v["route"], v["extend"], v["promote"], v["blocking"]), ("", False, False, ["floor_collapse"]))
        self.assertTrue(v["blocks"])
        self.assertIn("COLLAPSED", k2.summary(v))
        # even a route A that holds does not promote a collapsed player
        _, c2, b2 = block(prefix="L", seed=8, floor_mean=-6.5)
        both = v3(c + c2, b + b2, extended=True)
        self.assertEqual((both["route"], both["promote"], both["blocking"]), ("A", False, ["floor_collapse"]))
        # the line is the 99% upper bound against -5.0, strictly
        self.assertEqual(k2.COLLAPSE_FLOORS, -5.0)
        self.assertEqual(k2.COLLAPSE_ALPHA, 0.01)
        self.assertFalse(k2.floor_collapse({"upper99": -5.0, "mean": -5.5, "n": 2000})["collapsed"])
        self.assertTrue(k2.floor_collapse({"upper99": -5.0001, "mean": -5.5, "n": 2000})["collapsed"])
        # the bound is the bootstrap's one-sided 99%: above the mean by about 2.3 standard errors
        _, c3, b3 = block(prefix="K", seed=7, floor_mean=-4.0)
        ok = v3(c3, b3)
        self.assertAlmostEqual(ok["collapse"]["upper99"] - ok["floor"]["mean"], 2.326 * ok["floor"]["se"], delta=0.15 * ok["floor"]["se"])
        self.assertEqual(ok["blocking"], [])

    def test_a_flat_block_and_a_small_gain_are_refused_not_extended(self):
        _, c, b = block(FLAT_SPEC, floor_mean=0.0, prefix="N", seed=10)
        flat = v3(c, b)
        self.assertAlmostEqual(flat["deep"]["mean"], 0.0, places=9)
        self.assertEqual((flat["route"], flat["extend"], flat["promote"], flat["blocking"]), ("", False, False, []))
        _, c, b = block(SMALL_SPEC, floor_mean=0.0, prefix="O", seed=11)
        small = v3(c, b)
        self.assertAlmostEqual(small["deep"]["mean"], 0.5, places=9)
        self.assertEqual(small["routes"]["A"]["checks"]["min_gain"], False)
        self.assertEqual((small["route"], small["extend"], small["promote"]), ("", False, False))  # under +1.0 per 100 an extension is not earned
        # the point gain of +1.0 earns it (+20 net deep_f units in 2,000 pairs), +0.95 (19 units) does not
        at = v3(*block(((102, 3, 2), (4, 4, 2), (88, 2, 3), (1, 2, 4), (837, 1, 1), (441, 0, 1), (227, 1, 0), (300, 0, 0)), prefix="P", seed=12)[1:])
        self.assertEqual((at["deep"]["mean"], at["route"], at["extend"]), (1.0, "X", True))
        under = v3(*block(((101, 3, 2), (4, 4, 2), (88, 2, 3), (1, 2, 4), (838, 1, 1), (441, 0, 1), (227, 1, 0), (300, 0, 0)), prefix="P", seed=12)[1:])
        self.assertAlmostEqual(under["deep"]["mean"], 0.95, places=9)
        self.assertEqual((under["route"], under["extend"]), ("", False))

    def test_a_drift_that_blocks_under_v2_is_advisory_under_v3(self):
        _, c, b = block(floor_mean=-0.2, prefix="K", seed=7)
        _, c2, b2 = block(floor_mean=-0.2, prefix="L", seed=8)
        rows = (c + c2, b + b2)
        old = k2.judge(*rows, frontier=F_B3A, lineage=LINEAGE, previous_flagged=True, boot=BOOT)  # the real block r1's drift verdict
        self.assertEqual((old["drift"]["verdict"], old["route"]), ("blocked", "A"))
        self.assertEqual((old["promote"], old["blocking"]), (False, ["drift"]))
        new = v3(*rows, lineage=LINEAGE, previous_flagged=True, extended=True)
        self.assertEqual(new["drift"]["verdict"], "blocked")
        self.assertTrue(new["drift"]["advisory"])
        self.assertFalse(new["checks"]["drift"])  # `checks["drift"]` keeps what the check would have said ...
        self.assertTrue(new["promote"])  # ... and the verdict ignores it
        self.assertEqual((new["route"], new["blocking"], new["blocks"]), ("A", [], False))
        self.assertEqual(new["advisory_checks"], ["attempt", "drift"])
        self.assertIn("drift blocked (advisory)", k2.summary(new))
        # flagged (not blocked): `review` may still flag, and the promotion stands
        flagged = v3(*rows, lineage=LINEAGE, extended=True)
        self.assertEqual((flagged["drift"]["verdict"], flagged["review"], flagged["promote"]), ("flagged", True, True))
        # the attempt cap stays advisory beside it
        third = v3(*rows, prior_attempts=2, extended=True)
        self.assertEqual((third["checks"]["attempt"], third["promote"]), (False, True))
        # and a drift that blocks cannot turn an extension into a refusal
        ext = v3(self.c1, self.b1, lineage=LINEAGE, previous_flagged=True)
        self.assertEqual((ext["route"], ext["extend"], ext["blocking"]), ("X", True, []))

    def test_the_crash_guard_and_the_wins_clause_stay_checks(self):
        bad = v3(fx.with_bad(self.c1, 6), self.b1)  # 6 more lost seeds than the incumbent in 2,000 (the allowance is 5)
        self.assertEqual((bad["extend"], bad["promote"], bad["blocking"]), (False, False, ["crash"]))
        # wins: both arms at 10 in 100 and the candidate does not win more
        spec = ((1700, 1, 1), (100, 4, 4), (100, 4, 1), (100, 1, 4))
        _, c, b = fx.arms(spec, seed=3, prefix="W")
        w = v3(c, b)
        self.assertEqual(w["blocking"], ["wins"])
        self.assertFalse(w["promote"])
        self.assertFalse(w["extend"])


def std(mean: float, sd: float, n: int, seed: int = 3) -> list[float]:
    return list(fx.std_diffs(mean, sd, n, seed))


class SeqCallV3Tests(unittest.TestCase):
    LOOKS = [200, 500, 800, 1100, 1400, 1700, 2000]

    def call(self, floor_mean: float, deep_mean: float, n: int, j: int, sd_f: float = 8.8, sd_d: float = 33.0) -> dict:
        return k2.seq_call(std(floor_mean, sd_f, n, 3), std(deep_mean, sd_d, n, 4), j, self.LOOKS, rule="v3")

    def test_the_smoke_look_refuses_nothing_on_these_figures(self):
        for floor, deep in ((-9.0, -9.0), (-2.0, 0.0), (5.0, 5.0)):
            c = self.call(floor, deep, 200, 0)
            self.assertEqual((c["action"], c["smoke_would_refuse"]), ("continue", False), (floor, deep))
        self.assertEqual(self.call(8.0, 30.0, 200, 0)["action"], "continue")  # and never promotes

    def test_a_later_look_refuses_on_a_dead_route_a_or_a_collapse_only(self):
        # route A dead: deep_f's 99% upper bound below +1.0 (800 seeds: SE 1.17, so the mean must be below -1.7)
        dead = self.call(0.5, -2.0, 800, 2)
        self.assertEqual(dead["action"], "refuse")
        self.assertIn("route A is dead", dead["reason"])
        self.assertLess(dead["deep_upper99"], 1.0)
        # the same bound at 95% would have refused a gain of -0.5 here: the 99% one does not
        alive = self.call(0.5, -0.5, 800, 2)
        self.assertLess(alive["deep_upper95"], 1.5)
        self.assertGreater(alive["deep_upper99"], 1.0)
        self.assertEqual(alive["action"], "continue")
        # a floor 3 down with deep_f on its way to an extension: v2 refuses it (both routes dead), v3 plays on
        c = self.call(-3.0, 1.5, 1100, 3)
        self.assertEqual(c["action"], "continue")
        v2 = k2.seq_call(std(-3.0, 8.8, 1100, 3), std(1.5, 33.0, 1100, 4), 3, self.LOOKS)
        self.assertEqual(v2["action"], "refuse")
        # a collapse refuses whatever deep_f says: the 99% upper bound of the floors below -5.0
        gone = self.call(-7.0, 3.0, 800, 2)
        self.assertEqual(gone["action"], "refuse")
        self.assertIn("the floors collapsed", gone["reason"])
        self.assertNotIn("route A is dead", gone["reason"])
        self.assertEqual(self.call(-5.5, 3.0, 800, 2)["action"], "continue")  # 800 seeds: SE 0.31, upper bound -4.8

    def test_an_interim_look_is_eligible_on_route_a_alone_at_the_early_z(self):
        z_e = k2.early_z(7)
        self.assertAlmostEqual(z_e, 3.72, places=2)
        se = 33.0 / 1100 ** 0.5  # 1,100 seeds
        hit = self.call(-3.0, (z_e + 0.05) * se, 1100, 3)  # floors 3 down: v2's floor margin would have kept route A from clearing
        self.assertEqual((hit["action"], hit["routes"]), ("eligible", ["A"]))
        miss = self.call(0.0, (z_e - 0.15) * se, 1100, 3)
        self.assertEqual((miss["action"], miss["routes"]), ("continue", []))
        # a z above the bar but a gain under +1.0 is not eligible (a tight spread): the gain is asked as well
        tight = k2.seq_call(std(0.0, 8.8, 1100, 3), std(0.9, 2.0, 1100, 4), 3, self.LOOKS, rule="v3")
        self.assertGreater(tight["deep_z"], z_e)
        self.assertEqual(tight["action"], "continue")
        # no floor route: a big floor gain alone is not eligible
        floor_only = k2.seq_call(std(6.0, 8.8, 1100, 3), std(0.0, 33.0, 1100, 4), 3, self.LOOKS, rule="v3")
        self.assertEqual((floor_only["action"], floor_only["routes"]), ("continue", []))

    def test_the_last_look_is_final_and_v2_is_untouched(self):
        self.assertEqual(self.call(-9.0, -9.0, 2000, 6)["action"], "final")
        with self.assertRaises(ValueError):
            k2.seq_call([1.0], [1.0], 0, self.LOOKS, rule="v9")
        a = k2.seq_call(std(-0.3, 8.8, 800, 3), std(2.0, 28.0, 800, 4), 2, self.LOOKS)
        b = k2.seq_call(std(-0.3, 8.8, 800, 3), std(2.0, 28.0, 800, 4), 2, self.LOOKS, rule="v2")
        self.assertEqual(a, b)
        self.assertNotIn("deep_upper99", a)


class FutilityV3Tests(unittest.TestCase):
    def check(self, deep_mean: float, n: int, n_block: int, **kw) -> dict:
        return gen_loop.futility_check(std(0.0, 8.8, n), std(deep_mean, 33.0, n, 4), n_block, v2=True, rule="v3", boot=2000, **kw)

    def test_route_a_alone_is_asked(self):
        got = self.check(1.45, 800, 2000, extend_possible=False)
        self.assertEqual(sorted(got["routes"]), ["A"])
        self.assertFalse(got["futile"])

    def test_a_first_block_is_futile_only_when_not_even_an_extension_is_in_reach(self):
        # 800 of 2,000 seeds, deep_f SE 1.17: the 95% upper bound of a mean of -0.5 is +1.4
        first = self.check(-0.5, 800, 2000, extend_possible=True)
        self.assertAlmostEqual(first["routes"]["A"]["needs"], 1.0)  # the gain that earns an extension
        self.assertTrue(first["routes"]["A"]["alive"])
        self.assertFalse(first["futile"])
        # an extended candidate's second block needs what promotes: 2.576 x SE of the whole judgement, scaled to the pooled N
        second = self.check(-0.5, 800, 2000, extend_possible=False)
        self.assertGreater(second["routes"]["A"]["needs"], 1.8)
        self.assertTrue(second["futile"])
        # the pooled N: the same 2,800 pairs played ask less of a judgement of 8,000 pairs (more seeds to come) than of one of 4,000
        small = self.check(0.5, 2800, 4000, extend_possible=False)["routes"]["A"]["needs"]
        big = self.check(0.5, 2800, 8000, extend_possible=False)["routes"]["A"]["needs"]
        self.assertLess(big, small)
        # a hopeless candidate stops even where an extension is possible
        self.assertTrue(self.check(-4.0, 800, 2000, extend_possible=True)["futile"])

    def test_v2s_futility_is_as_it_was(self):
        floors, deep = std(0.0, 8.8, 800), std(1.0, 33.0, 800, 4)
        old = gen_loop.futility_check(floors, deep, 2000, v2=True, boot=2000)
        self.assertEqual(sorted(old["routes"]), ["A", "F"])
        self.assertGreater(old["routes"]["A"]["needs"], 1.0)  # v2: the critical gain (1.9 here), not the extension's 1.0


class FunnelV3Tests(unittest.TestCase):
    def setUp(self):
        _, self.pre_c, self.pre_b = block(prefix="K", seed=7)  # the first block's 2,000 pairs
        self.seeds, self.cand, self.base = block(prefix="L", seed=8)

    def judge(self, cr, br):
        keep = gen_loop.keep_v2_verdict(cr, br, [], cand_gen=4, cand_sha="x", block="L", frontier_info={"index": F_B3A, "name": "B3a"}, boot=500,
                                        rule="v3", pooled=["K"], final=False)
        return gen_loop.compare_arms(cr, br, gen_loop.ALPHA, 500, None, keep=keep)

    def test_the_looks_count_the_pooled_pairs_and_the_rows_returned_hold_them(self):
        out = gen_loop.funnel(self.seeds, lambda n: (self.cand, self.base), self.judge, keep_rule="v3", frontier=F_B3A,
                              prefix=(self.pre_c, self.pre_b))
        self.assertEqual(out["stop"]["action"], "final")
        self.assertEqual((out["stop"]["n"], out["stop"]["pooled_pairs"]), (2000, 4000))  # the block's seeds, and the pairs of the judgement
        self.assertEqual([x["n"] for x in out["looks"]], [2200, 2500, 2800, 3100, 3400, 3700, 4000])
        self.assertEqual(len(out["cand_rows"]), 4000)
        self.assertEqual({r["seed"] for r in out["cand_rows"]}, {r["seed"] for r in out["base_rows"]})
        self.assertTrue(all(x["action"] in ("continue", "final") for x in out["looks"]), [x["action"] for x in out["looks"]])

    def test_a_hopeless_second_block_is_refused_at_the_second_look_on_route_a(self):
        # a candidate that loses deep_f on its second block (-12 per 100): the pooled pairs hold the first block's +1.45, but not for long
        worse = ((30, 3, 2), (270, 2, 3), (1000, 1, 1), (400, 0, 1), (300, 1, 0))
        seeds, c, b = block(worse, floor_mean=0.0, prefix="R", seed=14)
        out = gen_loop.funnel(seeds, lambda n: (c, b), self.judge, keep_rule="v3", frontier=F_B3A, prefix=(self.pre_c, self.pre_b))
        self.assertEqual(out["stop"]["action"], "refuse")
        self.assertIn("route A is dead", out["stop"]["reason"])
        self.assertTrue(500 <= out["stop"]["n"] < 2000, out["stop"]["n"])
        self.assertEqual(len(out["cand_rows"]), out["stop"]["n"] + 2000)

    def test_without_a_prefix_the_funnel_is_as_it_was(self):
        out = gen_loop.funnel(self.seeds, lambda n: (self.cand, self.base), self.judge, keep_rule="v3", frontier=F_B3A)
        self.assertNotIn("pooled_pairs", out["stop"])
        self.assertEqual([x["n"] for x in out["looks"]], gen_loop.seq_looks(2000))


class RulePlumbingTests(unittest.TestCase):
    def test_the_rule_is_a_rule_and_the_helpers_know_the_drift_is_advisory(self):
        self.assertEqual(gen_loop.KEEP_RULES, ("v1", "v2", "v3"))
        self.assertEqual(gen_loop.keep_rule_of("V3"), "v3")
        with tempfile.TemporaryDirectory() as d:  # the old attempt cap is recorded under v3 as under v2 (the round says so and goes on)
            tsv = Path(d) / "promotions.tsv"
            for i in range(2):
                gen_loop.append_promotion(tsv, {"cand_gen": 4, "base_gen": 3, "seed_block": f"E{i}", "seeds": 200, "decision": "rejected",
                                                "cand_sha": f"sha-{i}", "cand_dir": f"c{i}", "attempt": i + 1})
            self.assertEqual(gen_loop.attempts_spent(tsv, 4, None, keep_rule="v3"), 2)
            self.assertEqual(gen_loop.attempts_spent(tsv, 4, None, keep_rule="v2"), 2)
            self.assertIsNone(gen_loop.attempts_spent(tsv, 4, None, keep_rule="v1"))
        checks = {"route": True, "drift": False, "attempt": False, "crash": True}
        self.assertFalse(gen_loop.checks_hold(checks))  # v2's advisory list is the attempt alone
        self.assertTrue(gen_loop.checks_hold(checks, ["attempt", "drift"]))
        self.assertEqual(gen_loop.failed_checks(checks), ["drift"])
        self.assertEqual(gen_loop.failed_checks(checks, ["attempt", "drift"]), [])
        self.assertEqual(gen_loop.row_rule({"rule": "v3"}), "v3")
        self.assertEqual((gen_loop.row_rule({"rule": "v2"}), gen_loop.row_rule({"rule": ""}), gen_loop.row_rule({})), ("v2", "v1", "v1"))
        self.assertEqual(gen_loop.decision_text({"promote": True}), "promoted")
        self.assertEqual(gen_loop.decision_text({"promote": False, "extend": True}), "extend")
        self.assertEqual(gen_loop.decision_text({"promote": False}), "rejected")

    def test_the_decision_carries_the_extension_and_the_advisory_list(self):
        _, c, b = block()
        keep = gen_loop.keep_v2_verdict(c, b, [], cand_gen=4, cand_sha="x", block="K", frontier_info={"index": F_B3A, "name": "B3a"}, boot=2000, rule="v3")
        res = gen_loop.compare_arms(c, b, gen_loop.ALPHA, 2000, None, keep=keep)
        d = res["decision"]
        self.assertEqual((d["promote"], d["extend"], d["route"], d["rule"]), (False, True, "X", k2.RULE_V3))
        self.assertEqual(d["advisory_checks"], ["attempt", "drift"])
        self.assertEqual(gen_loop.decision_text(d), "extend")
        cells = gen_loop.keep_cells(keep, 1)
        self.assertEqual((cells["rule"], cells["route"]), ("v3", "X"))
        self.assertTrue(cells["guard"].startswith("v3 route X (extend)"))
        # v2's decision has neither key
        old = gen_loop.compare_arms(c, b, gen_loop.ALPHA, 2000, None, keep=gen_loop.keep_v2_verdict(
            c, b, [], cand_gen=4, cand_sha="x", block="K", frontier_info={"index": F_B3A, "name": "B3a"}, boot=2000))
        self.assertNotIn("extend", old["decision"])
        self.assertNotIn("advisory_checks", old["decision"])

    def test_extended_before_reads_the_ledger_for_the_candidate_on_another_block(self):
        rows = [{"seed_block": "E1", "cand_sha": "sha-a", "run_head_sha": "", "macro": "", "rule": "v3", "route": "X"},
                {"seed_block": "E2", "cand_sha": "sha-b", "run_head_sha": "", "macro": "", "rule": "v3", "route": "A"},
                {"seed_block": "E3", "cand_sha": "sha-c", "run_head_sha": "", "macro": "", "rule": "v2", "route": "X"}]
        self.assertTrue(gen_loop.extended_before(rows, "sha-a", "", ""))
        self.assertFalse(gen_loop.extended_before(rows, "sha-a", "", "", block="E1"))  # reading the extending block again gives the extension again
        self.assertFalse(gen_loop.extended_before(rows, "sha-b", "", ""))  # promoted, not extended
        self.assertFalse(gen_loop.extended_before(rows, "sha-c", "", ""))  # a v2 row has no extension
        self.assertFalse(gen_loop.extended_before(rows, "sha-a", "h2", ""))  # another run head is another candidate
        self.assertFalse(gen_loop.extended_before(rows, "sha-a", "", "recipe"))
        self.assertFalse(gen_loop.extended_before([], "sha-a"))

    def test_the_cli_names_the_rule_and_the_pooled_blocks(self):
        p = gen0.build_parser()
        for cmd, extra in (("promote", ["--cand", "c", "--agent-root", "a"]), ("rejudge", ["--cand", "c", "--seeds-file", "s", "--agent-root", "a", "--note", "n"]),
                           ("train", ["--marker-dir", "m", "--shards-root", "s", "--agent-root", "a"])):
            self.assertEqual(p.parse_args([cmd, *extra, "--keep-rule", "v3"]).keep_rule, "v3", cmd)
        for cmd, extra in (("promote", ["--cand", "c", "--agent-root", "a"]), ("rejudge", ["--cand", "c", "--seeds-file", "s", "--agent-root", "a", "--note", "n"])):
            self.assertEqual(p.parse_args([cmd, *extra]).pool_block, [], cmd)
            self.assertEqual(p.parse_args([cmd, *extra, "--pool-block", "E1", "--pool-block", "E2"]).pool_block, ["E1", "E2"], cmd)
        ran = []
        with mock.patch.object(gen_loop.subprocess, "run", lambda cmd, check=False: ran.append(cmd) or SimpleNamespace(returncode=0)):
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None, keep_rule="v3", pool_blocks=["E1", "E2"])
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None, keep_rule="v3")
        self.assertEqual([x for i, x in enumerate(ran[0]) if ran[0][i - 1:i] == ["--pool-block"]], ["E1", "E2"])
        self.assertEqual(ran[0][ran[0].index("--keep-rule") + 1], "v3")
        self.assertNotIn("--pool-block", ran[1])


class PooledJudgementTests(unittest.TestCase):
    """`rejudge` and `promote` on saved or stubbed arms: the candidate's block r1 (extended) and its second block r2, the lineage's frontier at B3a."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ("EXAM_DIR", "LOOP_ROOT"):  # the judgement's human-bar line reads the gym's folder: not the real one in a test
            patcher = mock.patch.object(fe, name, self.root / "no-gym")
            patcher.start()
            self.addCleanup(patcher.stop)
        self.agents = self.root / "agent"
        self.agents.mkdir()
        self.s1, self.c1, self.b1 = block(prefix="K", seed=7)
        self.s2, self.c2, self.b2 = block(prefix="L", seed=8)
        (self.root / "seeds").mkdir()
        self.f1, self.f2 = self.root / "seeds" / "Etest-r1.json", self.root / "seeds" / "Etest-r2.json"
        self.f1.write_text(json.dumps({"seeds": self.s1}))
        self.f2.write_text(json.dumps({"seeds": self.s2}))
        # the lineage: generation 3 promoted on Eprev with the first Act 3 boss beaten 3 times in 100 (so the frontier is B3a)
        prev = [fx.run_row(f"P{i:09d}", 3 if i < 60 else 2 if i < 300 else 1 if i < 1000 else 0, np.random.default_rng(i)) for i in range(2000)]
        self.write_arm(self.agents / "gen3" / "candidate", "Eprev", prev)
        gen_loop.append_promotion(self.agents / "promotions.tsv",
                                  {"cand_gen": 3, "base_gen": 2, "seed_block": "Eprev", "seeds": 2000, "boss_cand": 1000, "boss_base": 900,
                                   "decision": "promoted", "cand_sha": "sha-3", "cand_dir": "candidate", "attempt": 1, "drift": "clear"})
        self.cand = stub_candidate(self.root, "c1", "sha-a", generation=4)
        self.write_arm(self.cand, "Etest-r1", self.c1)
        self.write_arm(self.agents / "gen0", "Etest-r1", self.b1)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def write_arm(folder: Path, name: str, rows: list[dict]) -> None:
        arm = folder / "arms" / f"{name}-a3-v1"
        arm.mkdir(parents=True)
        (arm / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def second_block(self) -> None:
        self.write_arm(self.cand, "Etest-r2", self.c2)
        self.write_arm(self.agents / "gen0", "Etest-r2", self.b2)

    def rejudge(self, seeds_file: Path, **kw):
        with mock.patch.object(gen0, "run_block", side_effect=AssertionError("must not play again")):
            return gen_loop.rejudge(self.cand, None, seeds_file, agent_root=self.agents, note="D66", boot=20000, keep_rule="v3", **kw)

    def test_the_first_block_is_rejudged_as_an_extension_and_the_ledger_says_so(self):
        doc = self.rejudge(self.f1)
        self.assertEqual((doc["decision_text"], doc["decision"]["promote"], doc["decision"]["extend"], doc["decision"]["route"]), ("extend", False, True, "X"))
        self.assertEqual((doc["keep_rule"], doc["keep_v2"]["rule"], doc["keep_v2"]["frontier"]["name"]), ("v3", k2.RULE_V3, "B3a"))
        self.assertNotIn("pooled", doc)
        self.assertFalse((self.agents / "current.json").exists())  # an extension promotes nothing
        row = gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]
        self.assertEqual((row["seed_block"], row["decision"], row["rule"], row["route"], row["seeds"]), ("Etest-r1", "extend", "v3", "X", "2000"))
        self.assertEqual(gen_loop.row_rule(row), "v3")
        self.assertIn("v3 route X (extend)", row["guard"])
        rows = gen_loop.read_promotions(self.agents / "promotions.tsv")
        self.assertTrue(gen_loop.extended_before(rows, "sha-a", "", "", block="Etest-r2"))  # the second block is the second look
        self.assertFalse(gen_loop.extended_before(rows, "sha-a", "", "", block="Etest-r1"))
        again = self.rejudge(self.f1)  # the same block read again: the same extension, not a refusal for having been extended
        self.assertEqual(again["decision_text"], "extend")

    def test_pooling_a_block_judges_route_a_on_every_pair_and_names_the_blocks(self):
        self.second_block()
        out = self.root / "dry"
        snapshot = {p: p.read_bytes() for p in sorted(self.root.rglob("*")) if p.is_file()}
        doc = self.rejudge(self.f2, pool_blocks=["Etest-r1"], out=out)
        self.assertEqual((doc["decision_text"], doc["n_seeds"], doc["paired_seeds"]), ("promoted", 4000, 4000))
        k = doc["keep_v2"]
        self.assertEqual((k["route"], k["extended"], k["extend"], k["promote"]), ("A", True, False, True))
        self.assertAlmostEqual(k["deep"]["mean"], 1.45, places=9)
        self.assertGreater(k["deep"]["lower_route"], 0)
        self.assertEqual(k["kills"]["B1"], [2518, 2946])
        pooled = doc["pooled"]
        self.assertEqual([(b["block"], b["role"], b["pairs"]) for b in pooled["blocks"]], [("Etest-r2", "judged", 2000), ("Etest-r1", "pooled", 2000)])
        self.assertEqual((pooled["pairs"], pooled["main_pairs"]), (4000, 2000))
        for b in pooled["blocks"]:  # each side's kills per boss on each block: the figures the report reads
            self.assertEqual((b["kills"]["cand"][0], b["kills"]["base"][0], b["kills"]["cand"][3], b["kills"]["base"][3]), (1259, 1473, 4, 1))
        # the frontier is fixed from the lineage alone: neither block's own row is read as lineage
        self.assertEqual([u["block"] for u in k["frontier"]["used"]], ["Eprev"])
        self.assertEqual(k["frontier"]["name"], "B3a")
        # the dry run: the agent folder is as it was, the copy of the ledger has the pooled row
        after = {p: p.read_bytes() for p in sorted(self.root.rglob("*")) if p.is_file() and out not in p.parents}
        self.assertEqual(after, snapshot)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["current.json", "promotion.json", "promotions.tsv"])
        row = gen_loop.read_promotions(out / "promotions.tsv")[-1]
        self.assertEqual((row["seed_block"], row["seeds"], row["decision"], row["rule"], row["route"]), ("Etest-r2", "4000", "promoted", "v3", "A"))
        self.assertIn("pooled Etest-r2 (2000) + Etest-r1 (2000) = 4000 pairs", row["guard"])
        self.assertEqual(json.loads((out / "promotion.json").read_text())["pooled"]["pairs"], 4000)

    def test_a_pooled_judgement_is_the_second_look_it_never_extends_again(self):
        # a second block with a small gain (+0.55 per 100): the pooled gain is +1.0, its bound is not above 0, and the candidate was extended:
        # refused, not extended twice
        small = ((93, 3, 2), (4, 4, 2), (88, 2, 3), (1, 2, 4), (846, 1, 1), (441, 0, 1), (227, 1, 0), (300, 0, 0))
        _, c2, b2 = block(small, floor_mean=-1.95, prefix="L", seed=8)
        self.write_arm(self.cand, "Etest-r2", c2)
        self.write_arm(self.agents / "gen0", "Etest-r2", b2)
        doc = self.rejudge(self.f2, pool_blocks=["Etest-r1"], out=self.root / "dry")
        k = doc["keep_v2"]
        self.assertEqual((k["deep"]["mean"], k["routes"]["A"]["checks"]), (1.0, {"deep_bound": False, "min_gain": True}))
        self.assertEqual((doc["decision_text"], doc["decision"]["extend"], k["route"], k["extended"]), ("rejected", False, "", True))

    def test_pooling_refuses_what_it_cannot_count_twice(self):
        self.second_block()
        with self.assertRaisesRegex(ValueError, "keep rule v3"):
            gen_loop.rejudge(self.cand, None, self.f2, agent_root=self.agents, note="D66", boot=500, keep_rule="v2", pool_blocks=["Etest-r1"], out=self.root / "d")
        with self.assertRaisesRegex(ValueError, "an arm is missing"):
            self.rejudge(self.f2, pool_blocks=["Etest-none"], out=self.root / "d")
        with self.assertRaisesRegex(ValueError, "seeds are in another block"):
            self.rejudge(self.f2, pool_blocks=["Etest-r2"], out=self.root / "d")  # the block itself pooled with itself
        self.write_arm(self.cand, "Etest-r3", self.c1)  # the first block's rows again under another block's name: every seed is a repeat
        self.write_arm(self.agents / "gen0", "Etest-r3", self.b1)
        with self.assertRaisesRegex(ValueError, "seeds are in another block"):
            self.rejudge(self.f2, pool_blocks=["Etest-r1", "Etest-r3"], out=self.root / "d")

    def test_rejudge_cmd_prints_the_pooled_blocks_and_the_extension(self):
        self.second_block()
        args = SimpleNamespace(cand=str(self.cand), base=None, seeds_file=str(self.f2), agent_root=str(self.agents), config="a3-v1", note="D66",
                               lab=None, review_guard=None, keep_rule="v3", out=str(self.root / "dry"), pool_block=["Etest-r1"])
        buf = io.StringIO()
        with mock.patch.object(gen0, "_setup"), mock.patch.object(gen0, "run_block", side_effect=AssertionError("must not play")):
            with contextlib.redirect_stdout(buf):
                self.assertEqual(gen0.rejudge_cmd(args), 0)
        printed = json.loads(buf.getvalue())
        self.assertEqual((printed["decision_text"], printed["n_seeds"]), ("promoted", 4000))
        self.assertEqual((printed["keep_v2"]["route"], printed["keep_v2"]["extend"], printed["keep_v2"]["extended"]), ("A", False, True))
        self.assertEqual(printed["pooled"]["pairs"], 4000)
        self.assertEqual(printed["keep_v2"]["kills"]["B1"], [2518, 2946])

    # ---------------------------------------------------------------- promote on stub arms

    def promote(self, seeds_file: Path, cand_rows, base_rows, **kw):
        def fake_run_arm(seeds, arm_dir, agent, config, procs, ascension=10, resume=False, **_):
            want = set(seeds)
            return [r for r in (cand_rows if agent.prior_dir else base_rows) if r["seed"] in want]

        with mock.patch.object(gen_loop, "run_arm", fake_run_arm):
            return gen_loop.promote(self.cand, None, seeds_file, agent_root=self.agents, boot=20000, keep_rule="v3", **kw)

    def test_promote_extends_a_first_block_and_promotes_on_the_pooled_pairs_of_the_second(self):
        doc = self.promote(self.f1, self.c1, self.b1)
        self.assertEqual((doc["decision_text"], doc["decision"]["route"]), ("extend", "X"))
        self.assertFalse((self.agents / "current.json").exists())
        rows = gen_loop.read_promotions(self.agents / "promotions.tsv")
        self.assertEqual((rows[-1]["decision"], rows[-1]["rule"], rows[-1]["route"]), ("extend", "v3", "X"))
        # the second block, pooled with the first (the arms of the first are on disk)
        second = self.promote(self.f2, self.c2, self.b2, pool_blocks=["Etest-r1"])
        self.assertEqual((second["decision_text"], second["n_seeds"], second["pooled"]["pairs"]), ("promoted", 4000, 4000))
        self.assertEqual(second["keep_v2"]["frontier"]["name"], "B3a")
        self.assertEqual(second["keep_v2"]["route"], "A")
        cur = json.loads((self.agents / "current.json").read_text())
        self.assertEqual((cur["generation"], Path(cur["dir"]).name, cur["seed_block"]), (4, "c1", "Etest-r2"))
        last = gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]
        self.assertEqual((last["seed_block"], last["seeds"], last["decision"], last["route"]), ("Etest-r2", "4000", "promoted", "A"))
        self.assertIn("= 4000 pairs", last["guard"])

    def test_promote_walks_the_funnel_over_the_pooled_pairs(self):
        doc = self.promote(self.f2, self.c2, self.b2, pool_blocks=["Etest-r1"], sequential=True)  # a second block with the first's arms pooled
        self.assertEqual(doc["decision_text"], "promoted")
        self.assertEqual((doc["sequential"]["stop"]["action"], doc["sequential"]["played"], doc["n_seeds"]), ("final", 2000, 4000))
        self.assertEqual((doc["sequential"]["keep_rule"], doc["sequential"]["frontier"]), ("v3", "B3a"))
        self.assertEqual([x["n"] for x in doc["sequential"]["looks"]][:2], [2200, 2500])  # the pooled pairs at the smoke look and the second
        self.assertEqual(doc["pooled"]["pairs"], 4000)

    def test_pooling_under_another_rule_is_refused_before_a_seed_is_played(self):
        with self.assertRaisesRegex(ValueError, "keep rule v3"):
            gen_loop.promote(self.cand, None, self.f2, agent_root=self.agents, keep_rule="v2", pool_blocks=["Etest-r1"])


class ExtensionRequeueTests(unittest.TestCase):
    """The train job's second block: a candidate the keep rule v3 extended is played on the date's next fresh block and judged pooled."""

    setUp, tearDown, build = tg12.PlacementTests.setUp, tg12.PlacementTests.tearDown, tg12.PlacementTests.build

    def train(self, *flags: str, verdicts: list[dict | None]):
        """`gen0.py train --promote` on the built candidate, the judging child stubbed: it answers with each verdict in turn, writing it as
        the candidate's promotion.json first (None: no record)."""
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
                "--emu-python", "python", "--seeds-file", str(seeds / "E{k}-{date}.json"), "--review-guard", "--sequential", "--keep-rule", *flags]
        out = io.StringIO()
        with mock.patch.object(gen_loop, "train_generation"), mock.patch.object(gen_loop, "run_promote_subprocess", side_effect=child),                 mock.patch.object(gen_loop, "COMMITTED_SEEDS", seeds), contextlib.redirect_stdout(out):
            rc = gen0.main(argv)
        return rc, calls, out.getvalue(), seeds

    EXTEND = {"seed_block": "E7-day", "decision": {"promote": False, "extend": True, "route": "X"}, "decision_text": "extend"}

    def test_an_extended_candidate_is_played_on_the_next_block_and_judged_pooled(self):
        rc, calls, out, seeds = self.train("v3", verdicts=[self.EXTEND, None])
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)
        first, second = calls
        self.assertEqual(first[1]["pool_blocks"], ())
        self.assertEqual(first[1]["keep_rule"], "v3")
        first_block = Path(first[0][3])
        second_block = Path(second[0][3])
        stem = first_block.stem  # E7-<date>
        self.assertEqual(second_block, seeds / f"{stem}-r1.json")  # the date's next -r<j> name, a seed file of its own
        self.assertEqual(len(gen0.load_seeds(second_block)), len(gen0.load_seeds(first_block)))  # as many seeds as the first block
        self.assertFalse(set(gen0.load_seeds(second_block)) & set(gen0.load_seeds(first_block)))
        self.assertEqual(second[1]["pool_blocks"], ("E7-day",))  # the block the extension named
        self.assertEqual((second[1]["sequential"], second[1]["keep_rule"]), (True, "v3"))  # the same funnel and rule
        self.assertIn("extended", out)

    def test_nothing_is_requeued_unless_v3_extended_it(self):
        for rule, verdict in (("v3", {"decision": {"promote": False, "extend": False}}), ("v3", {"decision": {"promote": True, "extend": False}}),
                              ("v2", self.EXTEND)):
            with self.subTest(rule=rule, verdict=verdict):
                self.tearDown()
                self.setUp()
                rc, calls, _, _ = self.train(rule, verdicts=[verdict, None])
                self.assertEqual((rc, len(calls)), (0, 1))
        self.tearDown()
        self.setUp()
        rc, calls, _, _ = self.train("v3", "--no-requeue", verdicts=[self.EXTEND, None])  # switched off
        self.assertEqual((rc, len(calls)), (0, 1))
        self.tearDown()
        self.setUp()
        rc, calls, _, _ = self.train("v3", verdicts=[None, None])  # no record: nothing to read
        self.assertEqual((rc, len(calls)), (0, 1))

    def test_the_next_block_skips_the_names_a_folder_and_the_ledger_hold(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            first = folder / "E18-2026-11-18-r1.json"
            gen0.write_seed_file("E18-2026-11-18-r1", 50, first)
            tsv = folder / "promotions.tsv"
            gen_loop.append_promotion(tsv, {"cand_gen": 18, "seed_block": "E18-2026-11-18-r2", "decision": "rejected"})
            (folder / "E18-2026-11-18-r3.json").write_text("{}")
            got = gen_loop.next_extension_block(first, tsv)
            self.assertEqual(got.name, "E18-2026-11-18-r4.json")  # r2 was judged, r3 is a file
            self.assertEqual(len(gen0.load_seeds(got)), 50)  # as many seeds as the first block
            self.assertEqual(gen0.load_seeds(got), gen0.e_block("E18-2026-11-18-r4", 50))  # the name's own seeds
            self.assertFalse(set(gen0.load_seeds(got)) & set(gen0.load_seeds(first)))
            plain = folder / "E7-2026-11-18.json"  # a first block with no -r suffix extends to -r1
            gen0.write_seed_file("E7-2026-11-18", 10, plain)
            self.assertEqual(gen_loop.next_extension_block(plain, None, n=20).name, "E7-2026-11-18-r1.json")

    def test_extension_of_reads_only_an_extension(self):
        with tempfile.TemporaryDirectory() as d:
            cand = Path(d)
            self.assertIsNone(gen_loop.extension_of(cand))
            for verdict, want in (({"decision": {"promote": False, "extend": True}}, True), ({"decision": {"promote": False}}, False),
                                  ({"decision": {"promote": True, "extend": True}}, False), ({}, False)):
                (cand / "promotion.json").write_text(json.dumps(verdict))
                self.assertEqual(gen_loop.extension_of(cand) is not None, want, verdict)


def nodes_row(seed: str, kind: str) -> dict:
    """A run with a complete node string: "past" went past the first Act 3 boss (died on floor 49), "died" died at it (floor 48), "early" died on floor 30."""
    base = "M" * 15 + "B" + "M" * 15 + "B" + "M" * 14 + "B"  # floors 2 to 48, the bosses on 17, 33 and 48
    nodes, floor = {"past": (base + "B", 49), "died": (base, 48), "early": ("M" * 15 + "B" + "M" * 13, 30)}[kind]
    return {"seed": seed, "floor": floor, "win": False, "end": "death", "crash": False, "nodes": nodes}


class MarginsTests(unittest.TestCase):
    """D68 addendum 6, decision 1 (judge half): every figure of a verdict carries its plus or minus 2 SE; nothing about the decision changes."""

    @classmethod
    def setUpClass(cls):
        cls.s1, cls.c1, cls.b1 = block(prefix="K", seed=7)
        cls.v = v3(cls.c1, cls.b1)

    def test_the_margins_are_the_counts_the_paired_differences_and_their_two_se(self):
        m = self.v["margins"]
        self.assertEqual((m["z"], m["pairs"]), (2.0, 2000))
        self.assertEqual((m["B1"]["cand"], m["B1"]["base"]), (1259, 1473))
        self.assertEqual((m["B3a"]["cand"], m["B3a"]["base"]), (115, 89))
        self.assertEqual((m["wins"]["cand"], m["wins"]["base"]), (4, 1))
        self.assertAlmostEqual(m["B1"]["diff"], 100.0 * (1259 - 1473) / 2000)
        self.assertAlmostEqual(m["B3a"]["diff"], 1.3)
        self.assertAlmostEqual(m["wins"]["diff"], 0.15)
        self.assertAlmostEqual(m["deep_f"]["diff"], 1.45, places=9)
        self.assertAlmostEqual(m["floors"]["diff"], -1.95, places=9)  # floors a run, not per 100
        self.assertEqual(m["floors"]["discordant"]["cand_ahead"] + m["floors"]["discordant"]["base_ahead"] <= 2000, True)
        for name in ("B1", "B2", "B3a", "wins", "deep_f", "floors", "arrivals"):
            x = m[name]
            self.assertAlmostEqual(x["pm2se"], 2.0 * x["se"], places=12, msg=name)
            self.assertGreater(x["se"], 0, name)
        # the SE of deep_f is the one the verdict's own summary gives (SD with n - 1 over the root of n)
        self.assertAlmostEqual(m["deep_f"]["se"], self.v["deep"]["se"], places=9)
        self.assertAlmostEqual(m["floors"]["se"], self.v["floor"]["se"], places=9)
        # the discordant pairs: the candidate beat the first Act 3 boss where the incumbent did not in 111 + 4 pairs, the reverse in 88 + 1
        self.assertEqual(m["B3a"]["discordant"], {"cand_ahead": 115, "base_ahead": 89})
        self.assertEqual(m["wins"]["discordant"], {"cand_ahead": 4, "base_ahead": 1})
        # arrivals at the first Act 3 boss: the fixture's rows have no node string, so a run arrived when its floor reached 48
        want = [sum(k2.arrived_b3a(r) for r in rows) for rows in (self.c1, self.b1)]
        self.assertEqual([m["arrivals"]["cand"], m["arrivals"]["base"]], want)
        self.assertEqual(self.v["margins"], k2.margins(self.c1, self.b1, F_B3A))

    def test_a_rate_per_arrival_has_its_own_binomial_se_and_the_difference_combines_them(self):
        cand = [nodes_row(f"S{i}", k) for i, k in enumerate(["past"] + ["died"] * 3 + ["early"] * 6)]
        base = [nodes_row(f"S{i}", k) for i, k in enumerate(["past"] * 2 + ["died"] * 3 + ["early"] * 5)]
        pa = k2.per_arrival(cand, base)
        self.assertEqual((pa["cand"]["arrivals"], pa["cand"]["passes"]), (4, 1))
        self.assertEqual((pa["base"]["arrivals"], pa["base"]["passes"]), (5, 2))
        self.assertAlmostEqual(pa["cand"]["rate"], 25.0)
        self.assertAlmostEqual(pa["base"]["rate"], 40.0)
        self.assertAlmostEqual(pa["cand"]["se"], 100 * (0.25 * 0.75 / 4) ** 0.5)
        self.assertAlmostEqual(pa["base"]["se"], 100 * (0.4 * 0.6 / 5) ** 0.5)
        self.assertAlmostEqual(pa["diff"], -15.0)
        self.assertAlmostEqual(pa["se"], (pa["cand"]["se"] ** 2 + pa["base"]["se"] ** 2) ** 0.5)
        self.assertAlmostEqual(pa["pm2se"], 2 * pa["se"])
        self.assertEqual(pa["unit"], "per 100 arrivals")
        # the margins' own arrivals count agrees, and an arm with no arrival has no rate
        self.assertEqual((k2.margins(cand, base, 2)["arrivals"]["cand"], k2.margins(cand, base, 2)["arrivals"]["base"]), (4, 5))
        none = k2.per_arrival(cand[4:], base[4:])  # the first four seeds (the arrivals of the candidate) dropped from both arms
        self.assertEqual(none["cand"]["arrivals"], 0)
        self.assertIsNone(none["cand"]["rate"])
        self.assertIsNone(none["diff"])
        # the same reading is in the verdict and, through the keep verdict, at the top of promotion.json
        v = k2.judge(cand, base, frontier=2, rule="v3", boot=500)
        self.assertEqual(v["per_arrival"], pa)

    def test_the_arrival_reading(self):
        self.assertTrue(k2.arrived_b3a({"win": True, "floor": 0}))
        self.assertFalse(k2.arrived_b3a({"crash": True, "floor": 49}))
        self.assertFalse(k2.arrived_b3a({"lost": True, "floor": 49}))
        self.assertTrue(k2.arrived_b3a(nodes_row("a", "died")))
        self.assertFalse(k2.arrived_b3a(nodes_row("a", "early")))
        self.assertTrue(k2.arrived_b3a({"floor": 48}))  # no node string: the cut 48
        self.assertFalse(k2.arrived_b3a({"floor": 47}))

    def test_the_ledger_line_shows_two_se_beside_every_figure_and_the_decision_is_unchanged(self):
        line = k2.summary(self.v)
        for part in ("deep +1.45 ±", "(lower ", "floor -1.95 ±", " (99% upper", "B3a 115/89 +1.30 ±", "wins 4/1 +0.15 ±",
                     "arrivals 115/89 +1.30 ±", "per arrival ", "B1 1259/1473 -10.70 ±", "B2 ", "wins guard off"):
            self.assertIn(part, line)
        self.assertNotIn("\t", line)
        self.assertNotIn("\n", line)
        parts = line.split(" | ")
        for name in ("deep +1.45", "floor -1.95", "B3a ", "wins 4/1", "arrivals ", "per arrival ", "B1 ", "B2 "):
            mine = [x for x in parts if x.startswith(name)]
            self.assertEqual(len(mine), 1, name)
            self.assertIn("±", mine[0], name)  # the margin is on the same part as the figure
        self.assertEqual(line.count("per arrival"), 1)
        # the memo's example ends with B1 and B2, so it reads in the design's order: Act 3 boss, wins, arrivals, per arrival, B1, B2
        order = [line.index(x) for x in ("B3a 115", "wins 4/1", "arrivals 115", "per arrival", "B1 1259", "B2 ")]
        self.assertEqual(order, sorted(order))
        # an old verdict (no margins, no per_arrival) still gives the line, without the margin parts
        old = {k: v for k, v in self.v.items() if k not in ("margins", "per_arrival")}
        old_line = k2.summary(old)
        self.assertNotIn("per arrival", old_line)
        self.assertIn("deep +1.45 ±", old_line)  # deep_f's and the floors' margins come from the bootstrap summary's SE
        # the decision of v3 is what it was: the same arms, the same checks
        self.assertEqual((self.v["route"], self.v["promote"], self.v["extend"]), ("X", False, True))
        # v2 prints them too
        v2 = k2.judge(self.c1, self.b1, frontier=F_B3A, boot=BOOT)
        self.assertIn("deep +1.45 ±", k2.summary(v2))
        self.assertIn("B3a 115/89 +1.30 ±", k2.summary(v2))

    def test_the_funnel_records_per_arrival_at_the_top_of_promotion_json(self):
        keep = dict(self.v)
        out = gen_loop.compare_arms(self.c1, self.b1, keep=keep, boot=500)
        self.assertEqual(out["per_arrival"], keep["per_arrival"])
        self.assertEqual(out["keep_v2"]["margins"], keep["margins"])
        bare = {k: v for k, v in keep.items() if k != "per_arrival"}
        self.assertNotIn("per_arrival", gen_loop.compare_arms(self.c1, self.b1, keep=bare, boot=500))  # a verdict an older judge wrote


if __name__ == "__main__":
    unittest.main()
