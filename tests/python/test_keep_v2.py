"""The keep test v2 (tools/lab/keep_v2.py, rl/gen_loop.py `incumbent_frontier`; docs/rl/KEEP_TEST_V2.md).

Constructed data only: synthetic paired arms built from counts (`keep_v2_fixtures`), differences with an exact mean and SD, and the figures
of the design's section 6. The block-by-block replay of the lab's saved arms skips where the lab's agent folder is absent
(SPIRELINE_REFEREE names it, default D:/spireline/referee); no saved row is committed.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import gen_loop  # noqa: E402
import keep_v2_fixtures as fx  # noqa: E402

from tools.lab import keep_v2 as k2  # noqa: E402
from tools.lab import promotion_guard as pg  # noqa: E402

BOOT = 20000


def row(floor=None, win=False, acts=None):
    r = {"seed": "S", "floor": floor, "win": win}
    if acts is not None:
        r["boss_damage_acts"] = acts
    return r


class BossesBeatenTests(unittest.TestCase):
    def test_bosses_beaten_reads_acts_first_and_floor_cuts_without(self):
        won = [[0, 1.0], [1, 1.0], [2, 1.0]]
        self.assertEqual(k2.bosses_beaten(row(51, win=True)), (True, True, True, True))  # a win beats all four, whatever else the row says
        self.assertEqual(k2.bosses_beaten(row(51, win=True, acts=[[0, 0.3]])), (True, True, True, True))
        # the field where the row has it: a run that ended on floor 35 without beating the Act 2 boss (two such runs in generation 11's arm)
        lost_at_b2 = row(35, acts=[[0, 1.0], [1, 0.8]])
        self.assertEqual(k2.bosses_beaten(lost_at_b2), (True, False, False, False))
        without = {"seed": "S", "floor": 35, "win": False}
        self.assertEqual(k2.bosses_beaten(without), (True, True, False, False))  # the floor cut says the Act 2 boss fell
        # A10 has two Act 3 bosses: beating the first ends the run on floor 48, which the cut (> 48) misses
        self.assertEqual(k2.bosses_beaten(row(48, acts=won)), (True, True, True, False))
        self.assertEqual(k2.bosses_beaten({"seed": "S", "floor": 48, "win": False}), (True, True, False, False))
        # exactly 1.0 is a boss beaten; the first entry of an act decides; an empty field is a run that reached no boss
        self.assertEqual(k2.bosses_beaten(row(20, acts=[[0, 0.999]])), (False, False, False, False))
        self.assertEqual(k2.bosses_beaten(row(20, acts=[[0, 0.4], [0, 1.0]])), (False, False, False, False))
        self.assertEqual(k2.bosses_beaten(row(20, acts=[[0, 1.0], [1, None]])), (True, False, False, False))  # an unknown damage is not a win
        self.assertEqual(k2.bosses_beaten(row(40, acts=[])), (False, False, False, False))
        self.assertEqual(k2.bosses_beaten({"seed": "S", "floor": 0, "win": False, "crash": True}), (False, False, False, False))
        # the floor cuts, 17 / 33 / 48
        for floor, want in ((17, (False, False, False)), (18, (True, False, False)), (33, (True, False, False)), (34, (True, True, False)),
                            (48, (True, True, False)), (49, (True, True, True))):
            self.assertEqual(k2.bosses_beaten({"seed": "S", "floor": floor, "win": False})[:3], want, floor)
        self.assertEqual(k2.bosses_beaten({"seed": "S", "floor": None, "win": False}), (False, False, False, False))

    def test_deep_counts_the_bosses_from_the_frontier_on(self):
        r = row(48, acts=[[0, 1.0], [1, 1.0], [2, 1.0]])  # B1, B2, B3a
        self.assertEqual([k2.deep(r, f) for f in range(4)], [3, 2, 1, 0])
        self.assertEqual([k2.deep(row(51, win=True), f) for f in range(4)], [4, 3, 2, 1])
        self.assertEqual(k2.deep(row(10, acts=[[0, 0.5]]), 0), 0)


class FrontierTests(unittest.TestCase):
    def test_frontier_is_deepest_boss_at_one_and_a_half_per_hundred(self):
        def rec(*kills, n=2000):
            return [(list(kills), n)]

        self.assertEqual(k2.frontier(rec(900, 30, 0, 0)), 1)  # 30 in 2,000 is 1.5 in 100: the Act 2 boss
        self.assertEqual(k2.frontier(rec(900, 29, 0, 0)), 0)  # 29 is not
        self.assertEqual(k2.frontier(rec(900, 60, 30, 0)), 2)  # the ladder: the first Act 3 boss at 1.5 in 100
        self.assertEqual(k2.frontier(rec(900, 60, 29, 0)), 1)
        self.assertEqual(k2.frontier(rec(900, 60, 30, 30)), 3)  # then the win
        self.assertEqual(k2.frontier(rec(0, 0, 0, 30)), 3)  # the deepest boss at the rate, whatever the shallower ones did
        self.assertEqual(k2.frontier([]), 0)  # B1 when there is no block
        self.assertEqual(k2.frontier([([5, 5, 5, 5], 0)]), 0)  # a block with no run says nothing
        # pooled over the lineage's last three blocks, kills and runs added: 10 + 20 in 2 x 1,000 = 1.5 in 100
        self.assertEqual(k2.frontier([([0, 10, 0, 0], 1000), ([0, 20, 0, 0], 1000)]), 1)
        self.assertEqual(k2.frontier([([0, 10, 0, 0], 1000), ([0, 19, 0, 0], 1000)]), 0)
        # only the last three blocks: a fourth, older block is not read
        self.assertEqual(k2.frontier([([0, 500, 0, 0], 1000), ([0, 0, 0, 0], 1000), ([0, 0, 0, 0], 1000), ([0, 0, 0, 0], 1000)]), 0)
        self.assertEqual(k2.frontier([([0, 0, 0, 0], 1000), ([0, 0, 0, 0], 1000), ([0, 0, 0, 0], 1000), ([0, 500, 0, 0], 1000)]), 1)
        # unequal blocks pool by runs, not by rate: generation 11's lineage before E12-r3 (38 + 5 + 5 in 2,400) is 2.0 in 100
        record = k2.frontier_record([([0, 38, 1, 0], 2000), ([0, 5, 0, 0], 200), ([0, 5, 0, 0], 200)])
        self.assertEqual((record["name"], record["pooled_n"], record["pooled_kills"][1], record["rates_per_100"][1]), ("B2", 2400, 48, 2.0))

    def test_block_counts_reads_kills_per_boss(self):
        rows = [row(18, acts=[[0, 1.0]]), row(40, acts=[[0, 1.0], [1, 1.0]]), row(51, win=True), row(5, acts=[[0, 0.2]])]
        self.assertEqual(k2.block_counts(rows), ([3, 2, 1, 1], 4))

    def test_frontier_reads_only_incumbent_rows(self):
        """`gen_loop.incumbent_frontier`: the promoted lineage's own arm on each of its last three blocks. A rejected candidate's arm on a
        block, however good its Act 2 record, is never read; neither is the block being judged."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)

            def arm(rel: str, block: str, n: int, b1: int, b2: int, b3a: int = 0, win: int = 0) -> None:
                rows = []
                for i in range(n):
                    level = 4 if i < win else 3 if i < b3a else 2 if i < b2 else 1 if i < b1 else 0
                    rows.append(fx.run_row(f"{block}-{i}", level, np.random.default_rng(i)))
                p = root / rel / "arms" / f"{block}-a3-v1"
                p.mkdir(parents=True)
                (p / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

            ledger = [
                {"seed_block": "EA", "cand_gen": "1", "cand_dir": "candidate", "decision": "promoted", "seeds": "1000"},
                {"seed_block": "EB", "cand_gen": "2", "cand_dir": "candidate", "decision": "rejected", "seeds": "1000"},
                {"seed_block": "EC", "cand_gen": "2", "cand_dir": "candidate-r1", "decision": "promoted", "seeds": "1000"},
                {"seed_block": "ED", "cand_gen": "3", "cand_dir": "candidate", "decision": "rejected", "seeds": "1000"},
            ]
            arm("gen1/candidate", "EA", 1000, 500, 10)      # the lineage on EA: 1.0 in 100 at the Act 2 boss
            arm("gen1/candidate", "EB", 1000, 500, 10)      # EB: gen 1 stayed incumbent; its arm is gen 1's
            arm("gen2/candidate", "EB", 1000, 600, 400)     # the rejected candidate's arm on EB: 40 in 100, must not count
            arm("gen2/candidate-r1", "EC", 1000, 500, 10)   # EC: the promoted candidate-r1
            arm("gen2/candidate-r1", "ED", 1000, 500, 200)  # ED: the incumbent candidate-r1 again (the block being judged below)
            arm("gen3/candidate", "ED", 1000, 700, 500)     # ED's rejected candidate
            got = gen_loop.incumbent_frontier(root, ledger[:3], "a3-v1", block="ED")
            self.assertEqual((got["name"], got["pooled_n"], got["pooled_kills"][1]), ("B1", 3000, 30))  # 30 in 3,000 is 1.0 in 100
            self.assertEqual([(u["block"], u["arm"]) for u in got["used"]],
                             [("EA", "gen1/candidate"), ("EB", "gen1/candidate"), ("EC", "gen2/candidate-r1")])
            # the same ledger with ED judged: the incumbent's (gen2/candidate-r1) arm on ED is read, not gen3's; the block being judged is not
            got = gen_loop.incumbent_frontier(root, ledger, "a3-v1", block="EE")
            self.assertEqual([(u["block"], u["arm"]) for u in got["used"]],
                             [("EB", "gen1/candidate"), ("EC", "gen2/candidate-r1"), ("ED", "gen2/candidate-r1")])
            self.assertEqual((got["name"], got["pooled_kills"][1]), ("B2", 220))  # 10 + 10 + 200 in 3,000: 7.3 in 100
            got = gen_loop.incumbent_frontier(root, ledger, "a3-v1", block="ED")  # the block being judged is left out
            self.assertEqual([u["block"] for u in got["used"]], ["EA", "EB", "EC"])
            self.assertEqual(gen_loop.incumbent_frontier(root, [], "a3-v1")["name"], "B1")
            # an arm that is not there is named and skipped; no path in the record is absolute
            (root / "gen2" / "candidate-r1" / "arms" / "EC-a3-v1" / "runs.jsonl").unlink()
            got = gen_loop.incumbent_frontier(root, ledger[:3], "a3-v1")
            self.assertEqual(got["missing"], ["gen2/candidate-r1/arms/EC-a3-v1"])
            self.assertEqual(len(got["used"]), 2)
            self.assertNotIn(str(root), json.dumps(got))

    def test_gen0_is_the_first_incumbent_whose_arm_is_read(self):
        """A base that was never beaten is generation 0: the lineage's player on a rejected first block is gen0's arm."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "gen0" / "arms" / "E1-a3-v1"
            p.mkdir(parents=True)
            rows = [fx.run_row(f"s{i}", 2 if i < 40 else 0, np.random.default_rng(i)) for i in range(1000)]
            (p / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            got = gen_loop.incumbent_frontier(root, [{"seed_block": "E1", "cand_gen": "1", "cand_dir": "candidate", "decision": "rejected"}])
            self.assertEqual((got["name"], got["used"][0]["arm"]), ("B2", "gen0"))


def std(mean: float, sd: float, n: int = 2000, seed: int = 3) -> np.ndarray:
    return fx.std_diffs(mean, sd, n, seed)


class RouteTests(unittest.TestCase):
    FLOOR = (-0.3, 8.8)  # l05's floor difference: SE 0.197

    def a(self, deep_mean: float, deep_sd: float = 28.0, floor_mean: float = -0.3, floor_sd: float = 8.8, n: int = 2000) -> dict:
        return k2.route_a(std(floor_mean, floor_sd, n), std(deep_mean, deep_sd, n, 4), boot=BOOT)

    def test_route_a_needs_gain_bound_and_floor_margin(self):
        ok = self.a(3.0)
        self.assertTrue(ok["holds"], ok)
        self.assertGreater(ok["deep_lower"], 0)
        # the gain: significant (tight spread) but under +1.0 per 100 fails the minimum gain only
        small = self.a(0.8, deep_sd=3.0)
        self.assertEqual(small["checks"], {"deep_bound": True, "min_gain": False, "floor_margin": True})
        self.assertFalse(small["holds"])
        # the bound: a +1.2 gain with the spread of l05's block (SE 0.63, z 1.9) does not clear 0.005
        noisy = self.a(1.2)
        self.assertEqual(noisy["checks"], {"deep_bound": False, "min_gain": True, "floor_margin": True})
        # the floor margin: lower 95% bound of the mean floor at least -1.0 (SE 0.197: -0.6 passes, -0.8 does not)
        self.assertTrue(self.a(3.0, floor_mean=-0.6)["checks"]["floor_margin"])
        fail = self.a(3.0, floor_mean=-0.8)
        self.assertEqual(fail["checks"], {"deep_bound": True, "min_gain": True, "floor_margin": False})
        self.assertFalse(fail["holds"])
        self.assertLess(fail["floor_lower95"], -1.0)
        # a collapse (generation 12's -5.91, the pure imitation's -3.14) fails it at once, however big the gain
        self.assertFalse(self.a(8.0, floor_mean=-3.14)["holds"])
        # n = 0 and constant differences hold nothing
        self.assertFalse(k2.route_a([], [], boot=100)["holds"])
        self.assertFalse(k2.route_a([0.0] * 50, [0.0] * 50, boot=100)["holds"])

    def test_route_f_is_the_floor_primary_at_half_alpha(self):
        se = 8.8 / math.sqrt(2000)
        for z, want_f, want_today in ((2.45, False, True), (2.80, True, True), (2.0, False, False)):  # z 2.45: past 0.01 (2.326), short of 0.005 (2.576)
            floor = std(z * se, 8.8, 2000, 5)
            deep = std(0.0, 28.0, 2000, 6)
            self.assertEqual(pg.cluster_bootstrap_lower(floor, 0.01, BOOT)["lower"] > 0, want_today, z)  # today's primary
            self.assertEqual(k2.route_f(floor, deep, boot=BOOT)["holds"], want_f, z)
        # deep_f not worse: a floor gain with Act 2 kills clearly down is not route F; a flat or small loss within the noise is
        floor = std(4 * se, 8.8, 2000, 5)
        down = k2.route_f(floor, std(-3.0, 28.0, 2000, 6), boot=BOOT)  # deep SE 0.63: the 95% upper bound is -1.96
        self.assertEqual(down["checks"], {"floor_bound": True, "deep_not_worse": False})
        self.assertFalse(down["holds"])
        self.assertTrue(k2.route_f(floor, std(-0.7, 28.0, 2000, 6), boot=BOOT)["holds"])  # upper bound +0.34
        self.assertEqual(k2.route_f(floor, std(0.0, 28.0, 2000, 6), boot=BOOT)["checks"]["deep_not_worse"], True)

    def test_bounds_come_from_the_floor_primarys_machinery(self):
        x = std(0.4, 8.0, 700, 9)
        s = k2.diff_summary(x, b=3000)
        for alpha, key in ((k2.ROUTE_ALPHA, "lower_route"), (k2.GUARD_ALPHA, "lower95")):
            self.assertAlmostEqual(s[key], pg.cluster_bootstrap_lower(x, alpha, 3000)["lower"], places=12)
        self.assertAlmostEqual(s["upper95"], -pg.cluster_bootstrap_lower(-x, 0.05, 3000)["lower"], places=12)
        self.assertAlmostEqual(s["p"], pg.cluster_bootstrap_lower(x, 0.05, 3000)["p"], places=12)
        self.assertAlmostEqual(s["se"], float(np.std(x, ddof=1)) / math.sqrt(700), places=12)
        self.assertEqual(k2.diff_summary([], 10)["n"], 0)

    def test_normal_stats_are_the_funnels_seq_stats(self):
        x = std(0.7, 8.0, 400, 2)
        a, b = k2.normal_stats(list(x)), gen_loop.seq_stats(x)
        for key in ("n", "mean", "se", "z"):
            self.assertAlmostEqual(a[key], b[key], places=9)
        self.assertEqual(k2.normal_stats([2.0] * 5)["z"], math.inf)
        self.assertEqual(k2.normal_stats([0.0] * 5)["z"], 0.0)


class GuardTests(unittest.TestCase):
    def test_crash_guard_slack(self):
        seeds, cand, base = fx.arms(((2000, 1, 1),))
        for k_c, k_b, want in ((5, 0, True), (6, 0, False), (8, 3, True), (9, 3, False), (0, 4, True)):  # base + max(2, 0.25% of 2,000 = 5)
            g = k2.crash_guard(fx.with_bad(cand, k_c), fx.with_bad(base, k_b))
            self.assertEqual((g["holds"], g["slack"], g["cand"], g["base"]), (want, 5.0, k_c, k_b), (k_c, k_b))
        short = 200  # a 200-seed block: max(2, 0.5) = 2
        c, b = cand[:short], base[:short]
        self.assertTrue(k2.crash_guard(fx.with_bad(c, 2), b)["holds"])
        self.assertFalse(k2.crash_guard(fx.with_bad(c, 3), b)["holds"])
        self.assertEqual(k2.crash_guard(fx.with_bad(c, 3), b)["slack"], 2)
        # a lost seed counts as a crashed one, and a seed only one arm played is not counted
        self.assertFalse(k2.crash_guard(fx.with_bad(c, 3, lost=True), b)["holds"])
        self.assertTrue(k2.crash_guard(fx.with_bad(c, 3), b[3:])["holds"])  # the three bad seeds have no pair
        # judge: a candidate that crashes 6 more seeds than the incumbent in 2,000 is refused even where a route holds
        seeds, cand, base = fx.arms()
        ok = k2.judge(cand, base, frontier=1, boot=2000)
        self.assertTrue(ok["promote"])
        bad = k2.judge(fx.with_bad(cand, 6), base, frontier=1, boot=2000)
        self.assertTrue(bad["route"])
        self.assertFalse(bad["promote"])
        self.assertEqual(bad["blocking"], ["crash"])

    def test_act1_guards_report_but_never_block(self):
        seeds, cand, base = fx.arms()
        v = k2.judge(cand, base, frontier=1, boot=BOOT)
        a1 = v["act1_review"]
        self.assertTrue(a1["kill"]["fires"])  # 334 incumbent-only kills against 256: today's guard fires
        self.assertTrue(a1["damage"]["fires"])  # and the Act 1 boss damage
        self.assertEqual((a1["kill"]["b"], a1["kill"]["c"], a1["kill"]["cand"], a1["kill"]["base"]), (256, 334, 965, 1043))
        self.assertIn("kill", a1["fires"])
        self.assertIn("damage", a1["fires"])
        self.assertTrue(a1["damage_act2"]["available"])
        self.assertFalse(a1["blocks"])
        self.assertEqual(sorted(v["checks"]), ["attempt", "crash", "drift", "route", "wins"])  # no Act 1 check among them
        self.assertTrue(v["promote"])
        self.assertEqual(v["blocking"], [])
        # the review guard on the same arms blocks (v1's rule) and says why
        kills_c, kills_b, dmg_c, dmg_b = pg.from_rows(cand, base, kill=pg.boss_kill)
        old = pg.judge(kills_c, kills_b, dmg_c, dmg_b)
        self.assertTrue(old["blocks"])
        self.assertEqual(old["blocking"], ["kill", "damage"])
        # the ledger's Act 1 counts are the floor's (a win or a floor past 17), as the drift check reads them
        self.assertEqual(v["drift"]["cand_kills"], 965)
        self.assertEqual(v["drift"]["cand_n"], 2000)
        # without boss_damage_acts the Act 2 damage is unavailable and reported so; nothing else changes
        old_rows = [{k: v for k, v in r.items() if k not in ("boss_damage_acts",)} for r in cand]
        r2 = k2.act1_report(old_rows, base, boot=500)
        self.assertFalse(r2["damage_act2"]["available"])

    def test_wins_co_primary_unchanged(self):
        # inactive below 10 in 100 in either arm: holds whatever the discordance
        seeds, cand, base = fx.arms(((1800, 1, 1), (100, 4, 1), (100, 1, 4)))  # 5% each
        w = k2.wins_clause(cand, base)
        self.assertFalse(w["active"])
        self.assertTrue(w["holds"])
        # active at 10 in 100 in both: the candidate must win more, exact McNemar at one-sided 0.01
        seeds, cand, base = fx.arms(((1700, 1, 1), (100, 4, 4), (100, 4, 1), (100, 1, 4)))  # 10% each, 100 discordant each way
        w = k2.wins_clause(cand, base)
        self.assertEqual((w["active"], w["b"], w["c"], w["holds"]), (True, 100, 100, False))  # equal is not better
        seeds, cand, base = fx.arms(((1560, 1, 1), (100, 4, 4), (240, 4, 1), (100, 1, 4)))
        w = k2.wins_clause(cand, base)
        self.assertEqual((w["active"], w["b"], w["c"], w["holds"]), (True, 240, 100, True))
        self.assertLess(w["p"], 0.01)
        seeds, cand, base = fx.arms(((1560, 1, 1), (100, 4, 4), (100, 4, 1), (240, 1, 4)))  # the incumbent wins more
        w = k2.wins_clause(cand, base)
        self.assertEqual((w["active"], w["holds"]), (True, False))
        # the clause blocks the promotion in `judge` even where a route holds (here far ahead at the Act 2 boss)
        spec = ((1460, 1, 1), (200, 4, 4), (100, 4, 1), (240, 1, 4), (900, 2, 1))
        seeds, cand, base = fx.arms(spec)
        v = k2.judge(cand, base, frontier=1, boot=2000)
        self.assertTrue(v["route"], v["routes"])
        self.assertFalse(v["checks"]["wins"])
        self.assertFalse(v["promote"])
        self.assertEqual(v["blocking"], ["wins"])

    def test_drift_and_attempt_cap_are_the_review_guards(self):
        seeds, cand, base = fx.arms()
        lineage = [(1100, 2000), (1130, 2000), (1120, 2000)]  # the lineage's Act 1 rate 0.56: the candidate's 0.4825 is wholly below
        flagged = k2.judge(cand, base, frontier=1, lineage=lineage, boot=1000)
        self.assertEqual(flagged["drift"]["verdict"], "flagged")
        self.assertTrue(flagged["promote"])  # flagged alone does not block ...
        self.assertTrue(flagged["review"])  # ... it marks the promotion for review
        blocked = k2.judge(cand, base, frontier=1, lineage=lineage, previous_flagged=True, boot=1000)
        self.assertEqual(blocked["drift"]["verdict"], "blocked")
        self.assertFalse(blocked["promote"])
        self.assertEqual(blocked["blocking"], ["drift"])
        third = k2.judge(cand, base, frontier=1, prior_attempts=2, boot=1000)  # a third candidate of the generation
        self.assertFalse(third["promote"])
        self.assertEqual(third["blocking"], ["attempt"])
        self.assertTrue(k2.judge(cand, base, frontier=1, prior_attempts=2, cap=3, boot=1000)["promote"])


class VerdictTests(unittest.TestCase):
    def test_v2_on_l05_like_arms_promotes_by_route_a(self):
        seeds, cand, base = fx.arms()
        v = k2.judge(cand, base, frontier=1, boot=BOOT)
        self.assertEqual(v["kills"], {"B1": [965, 1043], "B2": [107, 49], "B3a": [6, 3], "win": [0, 0]})
        self.assertEqual(v["route"], "A")  # route F does not hold: the floor is down
        self.assertTrue(v["promote"])
        self.assertAlmostEqual(v["deep"]["mean"], 3.05, places=9)  # (107 + 6 - 49 - 3) / 2,000 per 100
        self.assertGreater(v["deep"]["lower_route"], 1.2)  # the design: about +1.4
        self.assertLess(v["deep"]["lower_route"], 1.7)
        self.assertGreater(v["floor"]["lower95"], -1.0)
        self.assertLess(v["floor"]["mean"], 0)
        self.assertLess(v["floor"]["lower_route"], 0)  # the floor primary at 0.005 is not met
        self.assertFalse(v["routes"]["F"]["holds"])
        self.assertEqual((v["frontier"]["index"], v["frontier"]["name"]), (1, "B2"))
        self.assertEqual((v["crash"]["cand"], v["crash"]["base"]), (0, 0))
        self.assertEqual(v["drift"]["verdict"], "skipped")
        self.assertEqual(v["rule"], k2.RULE)
        # the same arms at the Act 1 frontier: deep_B1 is the B1 loss plus the B2 gain, floor-like, and the route is not A
        v1 = k2.judge(cand, base, frontier=0, boot=2000)
        self.assertEqual(v1["frontier"]["name"], "B1")
        self.assertLess(v1["deep"]["mean"], v["deep"]["mean"])
        self.assertFalse(v1["promote"])
        # the ledger cells and the summary line
        cells = k2.ledger_cells(v)
        self.assertEqual((cells["rule"], cells["frontier"], cells["route"], cells["k2_cand"], cells["k2_base"]), ("v2", "B2", "A", 107, 49))
        self.assertEqual(cells["d_deep"], "3.0500")
        self.assertEqual(set(cells), {"rule", "frontier", "route", "k2_cand", "k2_base", "d_deep", "deep_lower", "p_deep", "floor_lower95"})
        line = k2.summary(v)
        self.assertNotIn("\t", line)
        self.assertNotIn("\n", line)
        for part in ("route A", "frontier B2", "K2 107/49", "act1 review: kill, damage"):
            self.assertIn(part, line)

    def test_v2_keeps_every_past_ledger_verdict(self):
        """Section 6 of the design: floor mean, its 95% lower bound, the deep_f gain and SE of the 2,000-seed blocks, as constructed
        differences; the verdict of the routes is the table's. Only `candidate-imit-l05` (route A) changes a verdict."""
        n = 2000
        # block, floor mean, floor lower95, deep mean, deep SE, today's verdict, the route(s) v2 finds ("" = rejected)
        table = (
            ("E1-10-23", 0.41, 0.26, 0.65, 0.36, "promoted", "F"),
            ("E2-10-23", 0.31, 0.15, 0.90, 0.41, "promoted", "F"),
            ("E3-10-23", 0.03, -0.13, -0.15, 0.41, "rejected", ""),
            ("E3-10-23-r1", 0.29, 0.13, 1.05, 0.46, "promoted", "F"),
            ("E4-10-24", -0.20, -0.35, 0.10, 0.42, "rejected", ""),
            ("E4-10-24-r1", -0.13, -0.29, -0.10, 0.47, "rejected", ""),
            ("E5-10-25", 5.00, 4.75, 10.25, 0.86, "promoted", "AF"),
            ("E6-10-25", 1.72, 1.44, 13.05, 1.22, "promoted", "AF"),
            ("E7-10-26", 0.08, -0.21, 2.95, 1.34, "rejected", ""),
            ("E7-10-26-r1", 0.75, 0.44, 6.45, 1.38, "promoted", "AF"),
            ("E8-10-29", 2.04, 1.73, 12.25, 1.38, "promoted", "AF"),
            ("E9-10-29", 1.04, 0.71, 4.85, 1.43, "promoted", "AF"),
            ("E10-10-29", -1.44, -1.76, -1.25, 0.39, "rejected", ""),
            ("E10-10-30-r1", 0.23, -0.11, 0.55, 0.48, "rejected", ""),
            ("E11-11-02", 1.07, 0.78, 0.70, 0.47, "promoted", "F"),
            ("E12-11-02", -5.91, -6.29, -1.05, 0.37, "rejected", ""),
            ("E12-11-05-r3", -0.31, -0.63, 3.05, 0.63, "rejected", "A"),
        )
        for block, fm, lo95, dm, dse, today, want in table:
            floor = std(fm, (fm - lo95) / 1.645 * math.sqrt(n), n, 11)
            deep = std(dm, dse * math.sqrt(n), n, 12)
            st = k2.diff_stats(floor, deep, BOOT)
            a, f = k2.route_a(floor, deep, stats=st), k2.route_f(floor, deep, stats=st)
            got = ("A" if a["holds"] else "") + ("F" if f["holds"] else "")
            self.assertEqual(got, want, f"{block}: {got!r}, the design says {want!r}")
            self.assertEqual(bool(got), today == "promoted" or block == "E12-11-05-r3", block)  # every past verdict stands, the one change is l05
        # the 200-seed blocks were judged by the smoke look only: since D64 it refuses nothing and records what the rule of 2026-11-05 said
        looks = gen_loop.seq_looks(2000)
        for block, fm, lo95, dm, dse in (("E12-11-05-r1", -1.29, -2.42, -1.00, 1.42), ("E12-11-05-r2", -3.14, -4.27, 2.50, 2.06)):
            c = k2.seq_call(std(fm, (fm - lo95) / 1.645 * math.sqrt(200), 200, 13), std(dm, dse * math.sqrt(200), 200, 14), 0, looks)
            self.assertEqual((c["action"], c["smoke_would_refuse"]), ("continue", True), block)
        # the sanity block: the pure imitation would have been refused at 100 seeds, imit-l05 played on
        c = k2.seq_call(std(-4.30, 8.0, 100, 15), std(1.0, 25.0, 100, 16), 0, looks)
        self.assertEqual((c["action"], c["smoke_would_refuse"]), ("continue", True))
        c = k2.seq_call(std(-0.25, 8.0, 100, 15), std(6.0, 24.0, 100, 16), 0, looks)
        self.assertEqual((c["action"], c["smoke_would_refuse"]), ("continue", False))


class SeqCallTests(unittest.TestCase):
    LOOKS = gen_loop.seq_looks(2000)

    def test_the_final_look_is_judged_by_the_test_itself(self):
        c = k2.seq_call(std(-9.0, 8.8, 2000), std(-9.0, 28.0, 2000), 6, self.LOOKS)
        self.assertEqual(c["action"], "final")
        self.assertEqual(k2.seq_call([1.0, 2.0], [1.0, 2.0], 0, [2])["action"], "final")  # a one-look block


# ----------------------------------------------------------------- the lab's saved arms (skipped where they are absent)

REFEREE = Path(os.environ.get("SPIRELINE_REFEREE", "D:/spireline/referee"))
AGENT = REFEREE / "agent"
SECTION_6 = {  # block -> (the frontier, the routes) of the design's section 6 (the 2,000-seed blocks)
    "E1-2026-10-23": ("B1", "F"), "E2-2026-10-23": ("B1", "F"), "E3-2026-10-23": ("B1", ""), "E3-2026-10-23-r1": ("B1", "F"),
    "E4-2026-10-24": ("B1", ""), "E4-2026-10-24-r1": ("B1", ""), "E5-2026-10-25": ("B1", "AF"), "E6-2026-10-25": ("B1", "AF"),
    "E7-2026-10-26": ("B1", ""), "E7-2026-10-26-r1": ("B1", "AF"), "E8-2026-10-29": ("B1", "AF"), "E9-2026-10-29": ("B1", "AF"),
    "E10-2026-10-29": ("B2", ""), "E10-2026-10-30-r1": ("B2", ""), "E11-2026-11-02": ("B2", "F"), "E12-2026-11-02": ("B2", ""),
    "E12-2026-11-05-r3": ("B2", "A"),
}


class RecordedBlockTests(unittest.TestCase):
    """Every judged 2,000-seed block of the lab, from its two arms' saved rows (nothing is played): the frontier the ledger gives and the
    routes of the rule are the design's section 6 verdicts. Skips where the lab's agent folder is absent."""

    def test_v2_reproduces_section_6_from_the_saved_arms(self):
        tsv = AGENT / "promotions.tsv"
        if not tsv.is_file():
            self.skipTest("the lab's promotions.tsv is not here")
        rows = gen_loop.read_promotions(tsv)
        first: dict[str, int] = {}
        last: dict[str, dict] = {}
        for i, r in enumerate(rows):
            first.setdefault(r["seed_block"], i)
            last[r["seed_block"]] = r
        checked = 0
        for block, (want_f, want_route) in SECTION_6.items():
            r = last.get(block)
            cand = AGENT / f"gen{r['cand_gen']}" / (r["cand_dir"] or "candidate") / "arms" / f"{block}-a3-v1" if r else None
            base_gen = r["base_gen"] if r else None
            bases = ([AGENT / "gen0" / "arms" / f"{block}-a3-v1"] if base_gen == "0" else
                     sorted((AGENT / f"gen{base_gen}").glob(f"candidate*/arms/{block}-a3-v1"))) if r else []
            if r is None or not (cand / "runs.jsonl").is_file() or not bases or not (bases[0] / "runs.jsonl").is_file():
                continue
            c, b = gen_loop.arm_rows(cand), gen_loop.arm_rows(bases[0])
            front = gen_loop.incumbent_frontier(AGENT, rows[:first[block]], "a3-v1", block=block)
            self.assertEqual(front["name"], want_f, f"{block}: the frontier from the ledger")
            d = k2.paired_diffs(c, b, front["index"])
            st = k2.diff_stats(d["floor"], d["deep"], BOOT)
            got = ("A" if k2.route_a(d["floor"], d["deep"], stats=st)["holds"] else "") + ("F" if k2.route_f(d["floor"], d["deep"], stats=st)["holds"] else "")
            self.assertEqual(got, want_route, f"{block}: the routes")
            checked += 1
        if checked == 0:
            self.skipTest("none of the lab's saved arms is here")
        self.assertGreaterEqual(checked, 1)


if __name__ == "__main__":
    unittest.main()
