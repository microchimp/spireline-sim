"""Route S in the keep rule v3 (D66 as amended by D68 addendum 6 outcome 4, 2026-11-22): tools/lab/keep_v2.py (`s1_reading`, `route_s1`, `judge_v3(s1=)`,
`seq_call_v3(route_s=True)`, `summary`, `ledger_cells`); docs/rl/KEEP_TEST_V3.md "Route S".

Constructed data only: the synthetic paired arms of `test_keep_v3` (a block whose deep_f gain is +1.45 per 100 with floors -1.95, a flat one, a worse one)
and readings built the way `rl/act3_starts.py arrival-exam` writes them (one of them through `gym_gate.arrival_exam` itself, so the field names agree)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import gym_gate as gg  # noqa: E402
import keep_v2_fixtures as fx  # noqa: E402
from test_keep_v3 import F_B3A, FLAT_SPEC, V3D_SPEC, block, std  # noqa: E402

from tools.lab import keep_v2 as k2  # noqa: E402

BOOT = 4000
RULER, CAND, CHAMP = "a" * 64, "b" * 64, "c" * 64
PAIR = {"ruler_hash": RULER, "candidate": CAND, "champion": CHAMP}
WORSE_SPEC = ((40, 3, 2), (100, 2, 3), (1000, 1, 1), (400, 0, 1), (460, 1, 0))  # deep_f -3.0 per 100 in 2,000 pairs


def result(points: float, se_points: float, *, ruler: str = RULER, cand: str = CAND, champ: str = CHAMP, complete: bool = True, starts: int = 1127,
           test: int = 0, stopped=None) -> dict:
    """An arrival exam's `results.json` as `act3_starts.cmd_arrival_exam` writes it (the fields the rule reads)."""
    return {"rule": "act3-arrival-v1", "config": "a3-v1", "prior": "cpu", "engine": "E2", "ruler": "gen20-candidate-chain", "ruler_hash": ruler,
            "candidate": {"dir": "candidate-fight-v5", "policy_sha256": cand}, "champion": {"dir": "candidate-chain", "policy_sha256": champ},
            "test": test, "draws": list(range(6 * test, 6 * test + 6)), "stopped": stopped,
            "first_boss": {"points": points, "se_points": se_points, "starts": starts, "pairs": 6 * starts, "draws_per_start": 6, "complete": complete}}


def reading(points: float, se_points: float, expect=PAIR, **kw) -> dict:
    return k2.s1_reading(result(points, se_points, **kw), expect)


HOLDS = (1.8, 0.4)  # +1.8 passes per 100 arrivals, SE 0.4: the bound 1.8 - 2.576 x 0.4 = +0.77 is above 0


def v3(cand, base, s1=None, **kw) -> dict:
    return k2.judge(cand, base, frontier=F_B3A, rule="v3", boot=kw.pop("boot", BOOT), s1=s1, **kw)


class ReadingTests(unittest.TestCase):
    def test_the_names_and_the_parameters(self):
        self.assertEqual((k2.ROUTE_S, k2.ROUTE_AS, k2.S1_MIN, k2.S2_FROM_PAIRS, k2.S2_LOOK_ALPHA, k2.S2_FINAL_ALPHA), ("S", "AS", 1.0, 500, 0.01, 0.05))
        self.assertAlmostEqual(k2.S1_Z, 2.5758293035489, places=9)  # the one-sided 0.005 bound, as the exam's own Z995
        self.assertAlmostEqual(k2.S1_Z, gg.Z995, places=12)
        self.assertEqual(k2.route_s_parameters(), {"s1_min": 1.0, "s1_z": k2.S1_Z, "s2_from_pairs": 500, "s2_look_alpha": 0.01, "s2_final_alpha": 0.05})

    def test_a_reading_is_the_exams_figures_and_the_pairs_identity(self):
        r = reading(1.8, 0.4)
        self.assertEqual((r["mean"], r["se"], r["starts"], r["pairs"], r["test"], r["complete"]), (1.8, 0.4, 1127, 6762, 0, True))
        self.assertEqual((r["draws"], r["ruler_hash"], r["candidate"], r["champion"]), ([0, 1, 2, 3, 4, 5], RULER, CAND, CHAMP))
        self.assertTrue(r["match"]["verified"])
        self.assertEqual(r["match"]["reasons"], [])
        self.assertEqual(reading(1.8, 0.4, test=2)["draws"], [12, 13, 14, 15, 16, 17])

    def test_the_field_names_are_the_exams(self):
        """The reading of an exam played through `gym_gate.arrival_exam` (the games' win rows are the exam's own): a +2/6 gain on every start."""
        def game(i: int, j: int, won: bool) -> dict:
            return {"start": f"s{i}", "j": j, "won": won, "capped": False, "engine": "E2", "config": "a3-v1", "start_hp": 60, "end_hp": 30 if won else 0}

        cand = gg.indexed([game(i, j, j < 3 if i % 2 else j < 2) for i in range(40) for j in range(6)])
        inc = gg.indexed([game(i, j, j < 1) for i in range(40) for j in range(6)])
        first = gg.arrival_exam(cand, inc, ruler_hash=RULER)
        out = {"rule": "act3-arrival-v1", "config": "a3-v1", "engine": "E2", "ruler_hash": RULER, "test": 0, "draws": list(range(6)), "stopped": None,
               "candidate": {"dir": "x", "policy_sha256": CAND}, "champion": {"dir": "y", "policy_sha256": CHAMP}, "first_boss": first}
        r = k2.s1_reading(out, PAIR)
        self.assertAlmostEqual(r["mean"], first["points"], places=12)
        self.assertAlmostEqual(r["se"], first["se_points"], places=12)
        self.assertEqual((r["starts"], r["complete"]), (40, True))
        self.assertTrue(k2.route_s1(r)["applied"])


class RouteS1Tests(unittest.TestCase):
    def test_s1_holds_at_the_gain_and_the_bound(self):
        s1 = k2.route_s1(reading(*HOLDS))
        self.assertEqual((s1["applied"], s1["holds"], s1["checks"]), (True, True, {"min_gain": True, "bound": True}))
        self.assertAlmostEqual(s1["lower"], 1.8 - k2.S1_Z * 0.4, places=12)
        self.assertAlmostEqual(s1["pm2se"], 0.8, places=12)
        self.assertEqual((s1["starts"], s1["draws"], s1["ruler_hash"]), (1127, [0, 1, 2, 3, 4, 5], RULER))

    def test_a_mean_under_plus_1_leaves_it_unmet(self):
        s1 = k2.route_s1(reading(0.99, 0.1))  # the bound is far above 0, the gain is not +1.0
        self.assertEqual((s1["applied"], s1["holds"], s1["checks"]), (True, False, {"min_gain": False, "bound": True}))
        self.assertTrue(k2.route_s1(reading(1.0, 0.1))["holds"])  # exactly +1.0 holds

    def test_a_bound_that_is_not_above_0_leaves_it_unmet(self):
        # mean 1.2: the bound is above 0 only while 2.576 SE is under 1.2, that is SE under 0.4658
        self.assertTrue(k2.route_s1(reading(1.2, 0.46))["holds"])
        s1 = k2.route_s1(reading(1.2, 0.47))
        self.assertEqual((s1["applied"], s1["holds"], s1["checks"]), (True, False, {"min_gain": True, "bound": False}))
        self.assertLess(s1["lower"], 0)

    def test_a_reading_for_another_pair_is_not_applied_and_says_why(self):
        for key, got in (("ruler_hash", {"ruler": "d" * 64}), ("candidate", {"cand": "d" * 64}), ("champion", {"champ": "d" * 64})):
            s1 = k2.route_s1(reading(*HOLDS, **got))
            self.assertEqual((s1["applied"], s1["holds"]), (False, False), key)
            self.assertEqual(len(s1["reasons"]), 1, key)
            self.assertIn(f"{key} does not match", s1["reasons"][0])
        # two mismatches: both are said
        self.assertEqual(len(k2.route_s1(reading(*HOLDS, ruler="d" * 64, champ="e" * 64))["reasons"]), 2)
        # a pair that cannot name one of its three is not checked, so the reading is not applied; nor is one that was never checked
        none = k2.route_s1(reading(*HOLDS, expect={"ruler_hash": RULER, "candidate": CAND, "champion": None}))
        self.assertEqual((none["applied"], none["holds"]), (False, False))
        self.assertIn("champion: the pair's is not known", none["reasons"][0])
        self.assertFalse(k2.route_s1(reading(*HOLDS, expect=None))["applied"])
        raw = result(*HOLDS)
        raw["match"] = None
        self.assertFalse(k2.route_s1({**k2.s1_reading(raw), "match": None})["applied"])

    def test_an_incomplete_reading_or_a_candidate_that_is_not_a_fight_swap_is_not_applied(self):
        inc = k2.route_s1(reading(*HOLDS, complete=False, stopped="until"))
        self.assertEqual((inc["applied"], inc["holds"]), (False, False))
        self.assertIn("incomplete", inc["reasons"][0])
        swap = k2.route_s1(reading(*HOLDS), fight_swap=False)
        self.assertEqual((swap["applied"], swap["holds"]), (False, False))
        self.assertIn("not a fight-only swap", swap["reasons"][0])
        self.assertFalse(k2.route_s1(reading(None, None))["applied"])
        self.assertIsNone(k2.route_s1(None))


class JudgeRouteSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.c1, cls.b1 = block(prefix="K", seed=7)  # deep_f +1.45 per 100 (SE 0.74), floors -1.95: v3 extends it
        _, cls.cf, cls.bf = block(FLAT_SPEC, floor_mean=0.0, prefix="N", seed=10)  # no deep_f difference
        _, cls.cw, cls.bw = block(WORSE_SPEC, floor_mean=0.0, prefix="R", seed=14)  # deep_f -3.0
        cls.plain = v3(cls.c1, cls.b1)

    def test_no_reading_means_v3_unchanged(self):
        a = k2.judge(self.c1, self.b1, frontier=F_B3A, rule="v3", boot=BOOT)
        b = k2.judge(self.c1, self.b1, frontier=F_B3A, rule="v3", boot=BOOT, s1=None)
        c = k2.judge_v3(self.c1, self.b1, frontier=F_B3A, boot=BOOT, s1=None)
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertNotIn("route_s", a)
        self.assertEqual((a["route"], a["extend"], a["promote"]), ("X", True, False))  # the same extension v3 gives
        line = k2.summary(a)
        self.assertNotIn("S1", line)
        self.assertNotIn("S2", line)

    def test_s1_and_s2_holding_promote_by_s_where_v3_would_have_extended(self):
        v = v3(self.c1, self.b1, reading(*HOLDS))
        self.assertEqual((v["route"], v["promote"], v["extend"], v["blocking"]), ("S", True, False, []))
        self.assertTrue(v["checks"]["route"])
        self.assertFalse(v["routes"]["A"]["holds"])  # route A's bound is not above 0 ...
        self.assertEqual((self.plain["route"], self.plain["extend"]), ("X", True))  # ... which is what v3 made of it
        rs = v["route_s"]
        self.assertEqual((rs["applies"], rs["holds"], rs["decided_by"], rs["fight_swap"]), (True, True, "S", True))
        self.assertTrue(rs["s1"]["holds"])
        self.assertTrue(rs["s2"]["final_ok"])
        self.assertGreater(rs["s2"]["upper95"], 0)
        self.assertGreater(rs["s2"]["upper99"], rs["s2"]["upper95"])
        self.assertEqual((rs["s2"]["pairs"], rs["s2"]["final_alpha"], rs["s2"]["look_alpha"], rs["s2"]["from_pairs"]), (2000, 0.05, 0.01, 500))
        self.assertEqual(rs["parameters"], k2.route_s_parameters())
        self.assertEqual(rs["rule"], k2.RULE_S)
        self.assertEqual(v["rule"], k2.RULE_V3)  # the ledger's rule column stays v3
        # the verdict's other figures are v3's
        for key in ("deep", "floor", "kills", "collapse", "wins", "crash", "margins", "per_arrival"):
            self.assertEqual(v[key], self.plain[key], key)
        cells = k2.ledger_cells(v)
        self.assertEqual((cells["rule"], cells["route"]), ("v3", "S"))
        self.assertEqual(set(cells), {"rule", "frontier", "route", "k2_cand", "k2_base", "d_deep", "deep_lower", "p_deep", "floor_lower95"})

    def test_a_flat_block_promotes_by_s_and_a_route_a_that_holds_is_as(self):
        flat = v3(self.cf, self.bf, reading(*HOLDS))
        self.assertAlmostEqual(flat["deep"]["mean"], 0.0, places=9)
        self.assertEqual((flat["route"], flat["promote"]), ("S", True))
        self.assertGreater(flat["route_s"]["s2"]["upper95"], 0)
        self.assertEqual(v3(self.cf, self.bf)["route"], "")  # v3 alone refuses it
        _, c2, b2 = block(prefix="L", seed=8)
        both = v3(self.c1 + c2, self.b1 + b2, reading(*HOLDS), extended=True)  # 4,000 pooled pairs: route A holds as well
        self.assertTrue(both["routes"]["A"]["holds"])
        self.assertEqual((both["route"], both["promote"]), ("AS", True))
        self.assertEqual(k2.ledger_cells(both)["route"], "AS")
        self.assertEqual(v3(self.c1 + c2, self.b1 + b2, extended=True)["route"], "A")

    def test_s1_failing_leaves_v3_to_decide(self):
        for label, rd in (("mean under +1.0", reading(0.8, 0.1)), ("bound not above 0", reading(1.2, 0.6))):
            with self.subTest(label):
                ext = v3(self.c1, self.b1, rd)  # v3 extends the +1.45 block, as it did
                self.assertEqual((ext["route"], ext["extend"], ext["promote"]), ("X", True, False))
                self.assertTrue(ext["route_s"]["s1"]["applied"])
                self.assertFalse(ext["route_s"]["s1"]["holds"])
                self.assertEqual((ext["route_s"]["applies"], ext["route_s"]["decided_by"], ext["route_s"]["s2"]), (False, "v3", None))
                flat = v3(self.cf, self.bf, rd)  # and refuses a flat block
                self.assertEqual((flat["route"], flat["promote"], flat["extend"]), ("", False, False))
                again = {k: v for k, v in ext.items() if k != "route_s"}
                self.assertEqual(again, self.plain)  # nothing else about the verdict moved
        # v3's route A still promotes by itself where S1 fails
        _, c2, b2 = block(prefix="L", seed=8)
        a = v3(self.c1 + c2, self.b1 + b2, reading(0.8, 0.1), extended=True)
        self.assertEqual((a["route"], a["promote"]), ("A", True))

    def test_s2s_final_bound_failing_refuses_and_extends_nothing(self):
        v = v3(self.cw, self.bw, reading(*HOLDS))
        self.assertAlmostEqual(v["deep"]["mean"], -3.0, places=9)
        self.assertLess(v["route_s"]["s2"]["upper95"], 0)
        self.assertFalse(v["route_s"]["s2"]["final_ok"])
        self.assertEqual((v["route"], v["promote"], v["extend"], v["blocking"]), ("", False, False, []))
        self.assertFalse(v["checks"]["route"])
        self.assertEqual((v["route_s"]["applies"], v["route_s"]["holds"]), (True, False))
        # the bound is the 95% one-sided: a mean of 1.645 SE below 0 sits on it (n and SE chosen by the arms' own spread)
        ok = v3(self.cf, self.bf, reading(*HOLDS))
        self.assertAlmostEqual(ok["route_s"]["s2"]["upper95"] - ok["deep"]["mean"], 1.645 * ok["deep"]["se"], delta=0.2 * ok["deep"]["se"])

    def test_collapse_crash_and_wins_still_block(self):
        _, c, b = block(prefix="K", seed=7, floor_mean=-6.5)
        col = v3(c, b, reading(*HOLDS))
        self.assertEqual((col["route"], col["promote"], col["blocking"]), ("S", False, ["floor_collapse"]))
        bad = v3(fx.with_bad(self.c1, 6), self.b1, reading(*HOLDS))  # 6 more lost seeds than the incumbent in 2,000 (the allowance is 5)
        self.assertEqual((bad["promote"], bad["blocking"], bad["extend"]), (False, ["crash"], False))
        spec = ((1700, 1, 1), (100, 4, 4), (100, 4, 1), (100, 1, 4))  # both arms win 10 in 100 and the candidate does not win more
        _, cw, bw = fx.arms(spec, seed=3, prefix="W")
        w = v3(cw, bw, reading(*HOLDS))
        self.assertEqual((w["route"], w["promote"], w["blocking"]), ("S", False, ["wins"]))
        # the advisory checks stay advisory: a third attempt and a flagged drift do not stop S
        third = v3(self.c1, self.b1, reading(*HOLDS), prior_attempts=2)
        self.assertEqual((third["checks"]["attempt"], third["promote"], third["route"]), (False, True, "S"))

    def test_a_reading_that_names_another_pair_is_not_applied_and_the_reason_is_recorded(self):
        for key, got in (("ruler_hash", {"ruler": "d" * 64}), ("candidate", {"cand": "d" * 64}), ("champion", {"champ": "d" * 64})):
            v = v3(self.c1, self.b1, reading(*HOLDS, **got))
            self.assertEqual((v["route"], v["extend"], v["promote"]), ("X", True, False), key)  # v3 decides, as without the reading
            rs = v["route_s"]
            self.assertEqual((rs["applies"], rs["decided_by"], rs["s2"]), (False, "v3", None), key)
            self.assertFalse(rs["s1"]["applied"])
            self.assertIn(f"{key} does not match", rs["s1"]["reasons"][0])
            self.assertIn("S1 was not applied", rs["why"])
            self.assertIn(f"S1 not applied ({key} does not match", k2.summary(v))
        swap = v3(self.c1, self.b1, reading(*HOLDS), fight_swap=False)
        self.assertEqual((swap["route"], swap["route_s"]["applies"]), ("X", False))
        self.assertIn("not a fight-only swap", swap["route_s"]["s1"]["reasons"][0])

    def test_the_state_shift_is_the_whole_run_per_arrival_difference_minus_s1(self):
        v = v3(self.c1, self.b1, reading(*HOLDS))
        pa, sh = v["per_arrival"], v["route_s"]["state_shift"]
        self.assertAlmostEqual(sh["diff"], pa["diff"] - 1.8, places=9)
        self.assertAlmostEqual(sh["se"], (pa["se"] ** 2 + 0.4 ** 2) ** 0.5, places=9)
        self.assertAlmostEqual(sh["pm2se"], 2 * sh["se"], places=9)
        self.assertIsNone(v3(self.c1, self.b1, reading(*HOLDS, ruler="d" * 64))["route_s"]["state_shift"])  # not applied: not worked out

    def test_the_line_shows_s1_with_two_se_and_s2s_bound(self):
        v = v3(self.c1, self.b1, reading(*HOLDS))
        line = k2.summary(v)
        self.assertTrue(line.startswith("v3 route S | frontier B3a | deep +1.45"), line)
        self.assertIn("S1 +1.80 ±0.80 per 100 arrivals (1127 starts, 6 draws, ruler aaaaaaaa): holds", line)
        self.assertIn("S2 deep 95% upper +", line)
        self.assertIn("(final bound 0: ok; 99% upper +", line)
        self.assertNotIn("\t", line)
        failing = k2.summary(v3(self.cw, self.bw, reading(*HOLDS)))
        self.assertIn("(final bound 0: FAILS;", failing)
        self.assertTrue(failing.startswith("v3 route none"), failing)
        s1_failed = k2.summary(v3(self.c1, self.b1, reading(0.8, 0.1)))
        self.assertIn("S1 +0.80 ±0.20 per 100 arrivals (1127 starts, 6 draws, ruler aaaaaaaa): does not hold, v3 decides", s1_failed)
        self.assertNotIn("S2", s1_failed)
        self.assertTrue(s1_failed.startswith("v3 route X (extend)"), s1_failed)

    def test_the_rule_is_v3s_alone(self):
        with self.assertRaisesRegex(ValueError, "keep rule v3"):
            k2.judge(self.c1, self.b1, frontier=F_B3A, boot=500, s1=reading(*HOLDS))  # v2
        with self.assertRaisesRegex(ValueError, "keep rule v3"):
            k2.seq_call([1.0, 2.0], [1.0, 2.0], 0, [200, 500], route_s=True)  # v2


class SeqCallRouteSTests(unittest.TestCase):
    LOOKS = [200, 500, 800, 1100, 1400, 1700, 2000]

    def call(self, floor_mean: float, deep_mean: float, n: int, j: int, route_s: bool = True, sd_f: float = 8.8, sd_d: float = 33.0) -> dict:
        return k2.seq_call(std(floor_mean, sd_f, n, 3), std(deep_mean, sd_d, n, 4), j, self.LOOKS, rule="v3", route_s=route_s)

    def test_a_harm_stops_a_look_at_the_99_percent_bound_from_500_pairs_on(self):
        # 800 pairs: deep_f's SE 1.17, so the 99% upper bound is the mean + 2.72
        down = self.call(0.0, -3.0, 800, 2)
        self.assertEqual(down["action"], "refuse")
        self.assertIn("route S", down["reason"])
        self.assertLess(down["deep_upper99"], 0)
        # a mean of -2.0 is down by v3's measure (its 99% bound is under +1.0 and refuses) and not by S's
        self.assertEqual(self.call(0.0, -2.0, 800, 2, route_s=False)["action"], "refuse")
        self.assertEqual(self.call(0.0, -2.0, 800, 2)["action"], "continue")
        # 500 pairs is the first look that can refuse
        self.assertEqual(self.call(0.0, -6.0, 500, 1)["action"], "refuse")
        # under 500 pairs a look refuses nothing (a custom schedule: the rule starts at 500)
        early = k2.seq_call(std(0.0, 8.8, 300, 3), std(-9.0, 33.0, 300, 4), 1, [100, 300, 2000], rule="v3", route_s=True)
        self.assertEqual(early["action"], "continue")
        self.assertIn("refuse from 500 pairs", early["reason"])

    def test_a_collapse_refuses_and_the_smoke_look_does_not(self):
        gone = self.call(-7.0, 3.0, 800, 2)
        self.assertEqual(gone["action"], "refuse")
        self.assertIn("the floors collapsed", gone["reason"])
        self.assertNotIn("deep_f is down", gone["reason"])
        both = self.call(-7.0, -9.0, 800, 2)
        self.assertIn("deep_f is down", both["reason"])
        self.assertIn("the floors collapsed", both["reason"])
        for floor, deep in ((-9.0, -9.0), (-2.0, 0.0), (5.0, 5.0)):
            c = self.call(floor, deep, 200, 0)
            self.assertEqual((c["action"], c["smoke_would_refuse"]), ("continue", False), (floor, deep))

    def test_the_looks_refuse_only_never_promote_early_and_never_ask_a_gain(self):
        z_e = k2.early_z(7)
        se = 33.0 / 1100 ** 0.5
        big = self.call(0.0, (z_e + 0.5) * se, 1100, 3)  # a gain v3 would promote early on route A
        self.assertEqual(self.call(0.0, (z_e + 0.5) * se, 1100, 3, route_s=False)["action"], "eligible")
        self.assertEqual((big["action"], big["routes"]), ("continue", []))
        flat = self.call(0.0, 0.0, 1100, 3)  # and no gain is asked: a flat block plays on (v3 would play on too, its route A is not dead)
        self.assertEqual(flat["action"], "continue")
        for n, j in ((500, 1), (800, 2), (1100, 3), (1400, 4), (1700, 5)):
            self.assertNotEqual(self.call(0.0, 40.0, n, j)["action"], "eligible", n)
        self.assertEqual(self.call(-9.0, -9.0, 2000, 6)["action"], "final")  # the last look is `judge`'s: S2's final bound

    def test_without_route_s_v3_is_as_it_was(self):
        a = k2.seq_call(std(-3.0, 8.8, 1100, 3), std(1.5, 33.0, 1100, 4), 3, self.LOOKS, rule="v3")
        b = k2.seq_call_v3(std(-3.0, 8.8, 1100, 3), std(1.5, 33.0, 1100, 4), 3, self.LOOKS, route_s=False)
        self.assertEqual(a, b)
        self.assertNotIn("route", a)


if __name__ == "__main__":
    unittest.main()
