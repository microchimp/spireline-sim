"""tools/lab/promotion_guard.py (the independent review's D24 guard, docs/PROMOTION_GUARD.md) and its flag in rl/gen_loop.py.

Constructed data only: synthetic paired arms at kill rates 1%, 2%, 5% and 10% on 500 seeds (power and false fires of
the McNemar guard against the exact arithmetic of the review's appendix A), the damage guard on constructed fractions,
the drift check on a declining lineage, the retry cap, and `promote` with the arms stubbed: the existing D24 decision
is unchanged while the flag is off, and the review's guard decides when it is on.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import gen0  # noqa: E402
import gen_loop  # noqa: E402

from tools.lab import promotion_guard as pg  # noqa: E402

RATES = (0.01, 0.02, 0.05, 0.10)
SEEDS_500 = 500


def seeds(n: int) -> list[str]:
    return [f"S{i:09d}" for i in range(n)]


def simulate(rng: np.random.Generator, blocks: int, n: int, p_inc: float, p_cand: float) -> tuple[np.ndarray, np.ndarray]:
    """`blocks` independent paired blocks of n seeds: (candidate kills, incumbent kills) as 0/1 arrays of shape (blocks, n)."""
    return (rng.random((blocks, n)) < p_cand).astype(np.int8), (rng.random((blocks, n)) < p_inc).astype(np.int8)


def fire_rate(cand: np.ndarray, base: np.ndarray, alpha: float) -> float:
    """The share of blocks on which the kill guard fires, by the guard's own rule on the vectorised discordant counts."""
    b = ((cand == 1) & (base == 0)).sum(axis=1)
    c = ((base == 1) & (cand == 0)).sum(axis=1)
    fires = [pg.mcnemar_exact_one_sided(int(ci), int(bi)) < alpha and ci > bi for bi, ci in zip(b, c, strict=True)]
    return sum(fires) / len(fires)


def rows_like(sd: list[str], floor_of, damage=None) -> list[dict]:
    rows = [{"seed": s, "floor": floor_of(i), "win": False, "end": "death", "decisions": 10, "wall": 1.0, "crash": False}
            for i, s in enumerate(sd)]
    if damage is not None:
        for i, r in enumerate(rows):
            r[pg.DAMAGE_FIELD] = damage(i)
    return rows


class ArithmeticTests(unittest.TestCase):
    def test_mcnemar_wilson_and_the_bootstrap_are_the_promotion_codes_own(self):
        for b, c in ((10, 0), (7, 1), (3, 3), (0, 0), (0, 5), (33, 20), (20, 33)):
            self.assertEqual(pg.mcnemar_exact_one_sided(b, c), gen_loop.mcnemar_exact_one_sided(b, c))
        self.assertAlmostEqual(pg.mcnemar_exact_one_sided(7, 1), 9 / 256)
        for k, n in ((0, 10), (3, 10), (40, 2000), (120, 6000), (10, 10)):
            self.assertEqual(pg.wilson(k, n), gen0.wilson(k, n))
        rng = np.random.default_rng(1)
        diffs = list(rng.normal(0.05, 0.3, 300))
        mine = pg.cluster_bootstrap_lower(diffs, 0.05, 2000)
        theirs = gen_loop.cluster_bootstrap_lower(diffs, 0.05, 2000)
        self.assertEqual(mine, theirs)  # the same draws, the same bound: the damage guard resamples exactly as the floor primary

    def test_critical_counts_match_the_review(self):
        self.assertEqual(pg.critical_count(15, 0.01), 13)  # "at 500 seeds ... the 0.01 guard needs 13 of about 15 discordant pairs"
        self.assertEqual(pg.critical_count(10, 0.05), 9)
        self.assertEqual(pg.critical_count(0, 0.05), 1)  # no discordant pair: cannot fire
        self.assertEqual(pg.critical_count(3, 0.05), 4)  # three pairs cannot reach 0.05 one way (1/8)

    def test_exact_guard_power_reproduces_the_reviews_table(self):
        # review section 4.1, guard mode (incumbent at 2%): (seeds, candidate rate, alpha) -> power
        table = {(500, 0.015, 0.01): 0.02, (500, 0.015, 0.05): 0.10, (500, 0.013, 0.01): 0.04, (500, 0.013, 0.05): 0.16,
                 (500, 0.010, 0.01): 0.09, (500, 0.010, 0.05): 0.28, (2000, 0.015, 0.01): 0.11, (2000, 0.015, 0.05): 0.29,
                 (2000, 0.013, 0.01): 0.24, (2000, 0.013, 0.05): 0.49, (2000, 0.010, 0.01): 0.57, (2000, 0.010, 0.05): 0.81,
                 (2000, 0.005, 0.01): 0.98, (2000, 0.005, 0.05): 1.00}
        for (n, p2, alpha), want in table.items():
            self.assertAlmostEqual(pg.guard_power(n, 0.02, p2, alpha), want, delta=0.011, msg=(n, p2, alpha))
        self.assertEqual(pg.guard_power(500, 0.0, 0.0), 0.0)

    def test_coprimary_power_and_the_seeds_for_eighty_percent_reproduce_appendix_a(self):
        # review section 4.1, co-primary mode (one-sided 0.01): power at 500 / 2,000 / 4,000 seeds and the seeds for 80% power
        table = {(0.0135, 0.02): ((0.04, 0.20, 0.44), 7700), (0.02, 0.025): (None, 17600), (0.02, 0.03): ((0.06, 0.35, 0.68), 4800),
                 (0.02, 0.04): ((0.26, 0.91, 1.00), 1400), (0.10, 0.13): ((0.18, 0.73, 0.97), 2300), (0.10, 0.15): ((0.49, 0.99, 1.00), 900)}
        for (p1, p2), (powers, n80) in table.items():
            if powers:
                for n, want in zip((500, 2000, 4000), powers, strict=True):
                    self.assertAlmostEqual(pg.coprimary_power(n, p1, p2), want, delta=0.011, msg=(p1, p2, n))
            self.assertLess(abs(pg.seeds_for_power(p1, p2, 0.8, 0.01) - n80), 100, (p1, p2))
        # the guard at 0.05 needs fewer seeds than at 0.01, and a bigger drop fewer still
        self.assertLess(pg.seeds_for_power(0.02, 0.01, 0.8, 0.05), pg.seeds_for_power(0.02, 0.01, 0.8, 0.01))
        self.assertLess(pg.seeds_for_power(0.02, 0.005, 0.8, 0.05), pg.seeds_for_power(0.02, 0.01, 0.8, 0.05))
        self.assertEqual(pg.seeds_for_power(0.02, 0.02), 0)

    def test_the_table_prints_the_review_cases(self):
        text = pg.table()
        self.assertIn("| 2% -> 1% | 9% | 28% | 57% | 81% |", text)
        self.assertIn("| 2% -> 4% | 25% | 91% | 100% | 1,419 |", text)
        self.assertIn("| 2% | 3.2% | 3.9% |", text)  # false fires of the 0.05 guard with equal 2% arms
        with mock.patch.object(pg, "table", return_value="T\n"), mock.patch("sys.stdout") as out:
            self.assertEqual(pg.main(["--table"]), 0)
            out.write.assert_called_with("T\n")
        self.assertEqual(pg.binom_sf(10, 0, 0.3), 1.0)
        self.assertEqual(pg.binom_sf(10, 11, 0.3), 0.0)
        self.assertAlmostEqual(pg.binom_sf(10, 3, 0.3), sum(pg.binom_pmf(10, j, 0.3) for j in range(3, 11)))
        self.assertAlmostEqual(pg.binom_sf(12, 12, 0.5), 0.5**12)


class SyntheticArmsTests(unittest.TestCase):
    """Paired arms at 1%, 2%, 5% and 10% on 500 seeds, independent kills: the kill guard's false fires stay under its level
    and its power matches the exact mixture of appendix A; the guard sees little at these rates and this block size."""

    BLOCKS = 2000

    def rate_and_bounds(self, rng, p_inc, p_cand, alpha=pg.GUARD_ALPHA, n=SEEDS_500):
        cand, base = simulate(rng, self.BLOCKS, n, p_inc, p_cand)
        got = fire_rate(cand, base, alpha)
        exact = pg.guard_power(n, p_inc, p_cand, alpha)
        sd = (max(exact, 0.005) * (1 - exact) / self.BLOCKS) ** 0.5
        # the module's kill_guard on the first block agrees with the vectorised rule
        sd_names = seeds(n)
        v = pg.kill_guard(dict(zip(sd_names, cand[0].astype(bool), strict=True)), dict(zip(sd_names, base[0].astype(bool), strict=True)), alpha)
        b = int(((cand[0] == 1) & (base[0] == 0)).sum())
        c = int(((base[0] == 1) & (cand[0] == 0)).sum())
        self.assertEqual((v["b"], v["c"], v["pairs"], v["discordant"]), (b, c, n, b + c))
        self.assertEqual(v["fires"], pg.mcnemar_exact_one_sided(c, b) < alpha and c > b)
        return got, exact, sd

    def test_false_fires_with_equal_arms_stay_under_the_level(self):
        rng = np.random.default_rng(20261024)
        for p in RATES:
            got, exact, sd = self.rate_and_bounds(rng, p, p)
            self.assertLessEqual(got, pg.GUARD_ALPHA + 3 * sd, p)  # never above the nominal 5% beyond Monte Carlo noise
            self.assertLess(exact, pg.GUARD_ALPHA, p)  # the exact test is conservative
            self.assertAlmostEqual(got, exact, delta=4 * sd, msg=p)

    def test_power_against_a_halving_matches_the_exact_mixture_and_is_low_at_500_seeds(self):
        rng = np.random.default_rng(20261025)
        powers = {}
        for p in RATES:
            got, exact, sd = self.rate_and_bounds(rng, p, p / 2)
            self.assertAlmostEqual(got, exact, delta=4 * sd, msg=p)
            self.assertGreater(got, pg.GUARD_ALPHA, p)  # a real halving fires more often than noise does
            powers[p] = exact
        self.assertLess(powers[0.01], powers[0.02])  # fewer discordant pairs, less power
        self.assertLess(powers[0.02], powers[0.05])
        self.assertLess(powers[0.05], powers[0.10])
        self.assertAlmostEqual(powers[0.02], 0.28, delta=0.011)  # the review: 28% at 500 seeds, one-sided 0.05
        self.assertLess(powers[0.02], 0.3)  # 500 seeds do not see a halving of a 2% rate
        self.assertGreater(powers[0.10], 0.85)  # at 10% they do

    def test_at_the_promotion_block_size_the_guard_catches_a_halving(self):
        rng = np.random.default_rng(20261026)
        cand, base = simulate(rng, 600, 2000, 0.02, 0.01)
        got = fire_rate(cand, base, 0.05)
        self.assertAlmostEqual(got, pg.guard_power(2000, 0.02, 0.01, 0.05), delta=0.07)
        self.assertGreater(got, 0.7)  # the review: 81% at 2,000 seeds and 0.05, against 57% at 0.01
        self.assertAlmostEqual(pg.guard_power(2000, 0.02, 0.01, 0.01), 0.57, delta=0.011)

    def test_the_guard_fires_only_when_the_candidate_is_worse(self):
        sd = seeds(100)
        cand = dict.fromkeys(sd, False)
        base = dict.fromkeys(sd, False)
        for s in sd[:12]:
            base[s] = True  # twelve incumbent-only kills, no candidate-only: p = 2^-12
        v = pg.kill_guard(cand, base)
        self.assertEqual((v["fires"], v["b"], v["c"], v["cand"], v["base"]), (True, 0, 12, 0, 12))
        self.assertAlmostEqual(v["p_worse"], 2**-12)
        v = pg.kill_guard(base, cand)  # the mirror image: the candidate ahead never fires the guard
        self.assertEqual((v["fires"], v["b"], v["c"]), (False, 12, 0))
        self.assertEqual(pg.kill_guard({}, {})["fires"], False)
        # generation 1's numbers on E1-2026-10-23 (test_gen_loop): 33 candidate-only against 20 incumbent-only: clear
        g1c, g1b = dict.fromkeys(seeds(2000), False), dict.fromkeys(seeds(2000), False)
        for s in seeds(2000)[:33]:
            g1c[s] = True
        for s in seeds(2000)[33:53]:
            g1b[s] = True
        self.assertFalse(pg.kill_guard(g1c, g1b)["fires"])


class DamageGuardTests(unittest.TestCase):
    def test_unavailable_without_the_field_and_never_fires_then(self):
        v = pg.damage_guard(None, None)
        self.assertEqual((v["fires"], v["available"]), (False, False))
        self.assertIn(pg.DAMAGE_FIELD, v["why"])
        sd = seeds(50)
        kc, kb, dc, db = pg.from_rows(rows_like(sd, lambda i: 10), rows_like(sd, lambda i: 10))
        self.assertEqual((dc, db), (None, None))
        self.assertEqual(len(kc), 50)
        half = rows_like(sd, lambda i: 10, damage=lambda i: 0.5)
        half[7].pop(pg.DAMAGE_FIELD)  # one row short of the field: the whole guard is unavailable, not a silent zero
        self.assertEqual(pg.from_rows(half, rows_like(sd, lambda i: 10, damage=lambda i: 0.5))[2], None)

    def test_fires_when_the_candidate_deals_less_boss_damage_and_not_otherwise(self):
        sd = seeds(400)
        rng = np.random.default_rng(3)
        base_d = {s: float(x) for s, x in zip(sd, np.clip(rng.normal(0.5, 0.25, 400), 0, 1), strict=True)}
        worse = {s: max(0.0, v - 0.08) for s, v in base_d.items()}
        v = pg.damage_guard(worse, base_d, boot=2000)
        self.assertTrue(v["fires"] and v["available"])
        self.assertLess(v["diff_mean"], 0)
        self.assertLess(v["diff_upper"], 0)
        self.assertEqual(v["n"], 400)
        same = pg.damage_guard(base_d, base_d, boot=500)
        self.assertFalse(same["fires"])
        self.assertEqual(same["diff_mean"], 0.0)
        better = pg.damage_guard({s: min(1.0, v + 0.08) for s, v in base_d.items()}, base_d, boot=500)
        self.assertFalse(better["fires"])
        self.assertGreater(better["diff_upper"], 0)
        # a small, noisy difference on a block does not fire: the guard is not-worse at one-sided 0.05, not a trip-wire
        noisy = {s: float(np.clip(v + rng.normal(0, 0.3), 0, 1)) for s, v in base_d.items()}
        self.assertFalse(pg.damage_guard(noisy, base_d, boot=500)["fires"])

    def test_the_default_bootstrap_is_the_floor_primarys(self):
        sd = seeds(120)
        rng = np.random.default_rng(5)
        base_d = {s: float(x) for s, x in zip(sd, rng.random(120), strict=True)}
        cand_d = {s: float(x) for s, x in zip(sd, rng.random(120), strict=True)}
        own = pg.damage_guard(cand_d, base_d, boot=1000)
        via = pg.damage_guard(cand_d, base_d, boot=1000, lower=gen_loop.cluster_bootstrap_lower)
        self.assertEqual(own, via)
        self.assertEqual(pg.damage_guard({}, {})["available"], False)


class DriftCheckTests(unittest.TestCase):
    def test_a_declining_lineage_is_flagged_then_blocked(self):
        steady = [(60, 2000), (60, 2000), (60, 2000)]  # a 3% lineage
        v = pg.drift_check(20, 2000, steady)  # the candidate at 1%
        self.assertEqual((v["verdict"], v["flagged"], v["blocks"]), ("flagged", True, False))
        self.assertLess(v["cand_upper"], v["pooled_lower"])
        self.assertEqual((v["pooled_kills"], v["pooled_n"]), (180, 6000))
        again = pg.drift_check(20, 2000, steady, previous_flagged=True)
        self.assertEqual((again["verdict"], again["blocks"]), ("blocked", True))
        clear = pg.drift_check(55, 2000, steady)
        self.assertEqual((clear["verdict"], clear["flagged"], clear["blocks"]), ("clear", False, False))
        self.assertEqual(pg.drift_check(55, 2000, steady, previous_flagged=True)["verdict"], "clear")  # a clear block resets nothing here
        # a halving of a 2% lineage on one 2,000-seed block is flagged; the review's quarter is within noise (docs/PROMOTION_GUARD.md)
        two = [(40, 2000)] * 3
        self.assertEqual(pg.drift_check(20, 2000, two)["verdict"], "flagged")
        self.assertEqual(pg.drift_check(30, 2000, two)["verdict"], "clear")

    def test_only_the_last_three_blocks_are_pooled_and_no_lineage_skips(self):
        v = pg.drift_check(20, 2000, [(5, 2000), (5, 2000), (60, 2000), (60, 2000), (60, 2000)])
        self.assertEqual(v["lineage"], [[60, 2000]] * 3)
        self.assertEqual(v["verdict"], "flagged")
        none = pg.drift_check(20, 2000, [])
        self.assertEqual((none["verdict"], none["flagged"], none["blocks"]), ("skipped", False, False))
        self.assertEqual(pg.drift_check(20, 2000, [(0, 0)])["verdict"], "skipped")
        self.assertEqual(pg.drift_check(0, 0, [(60, 2000)])["verdict"], "skipped")

    def test_the_lineage_comes_from_the_promotions_rows(self):
        rows = [{"seeds": "2000", "decision": "promoted", "boss_cand": "40", "boss_base": "27"},
                {"seeds": "2000", "decision": "rejected", "boss_cand": "31", "boss_base": "44"},
                {"seeds": "2000", "decision": "promoted", "boss_cand": "50", "boss_base": "45"},
                {"seeds": "2000", "decision": "rejected", "boss_cand": "9", "boss_base": "52"},
                {"seeds": "", "decision": "rejected"}]  # an old row without the counts
        self.assertEqual(gen_loop.lineage_blocks(rows), [(44, 2000), (50, 2000), (52, 2000)])  # the lineage's own kills on each block
        self.assertEqual(gen_loop.lineage_blocks(rows[:1]), [(40, 2000)])
        self.assertFalse(gen_loop.previous_drift_flagged(rows))
        self.assertTrue(gen_loop.previous_drift_flagged([*rows, {"drift": "flagged"}]))
        self.assertTrue(gen_loop.previous_drift_flagged([*rows, {"drift": "blocked"}]))
        self.assertFalse(gen_loop.previous_drift_flagged([*rows, {"drift": "clear"}]))
        self.assertFalse(gen_loop.previous_drift_flagged([]))

    def test_a_rejudged_block_counts_once_in_the_lineage_by_its_last_row(self):
        rows = [{"seed_block": "E1", "seeds": "2000", "decision": "promoted", "boss_cand": "40", "boss_base": "27"},
                {"seed_block": "E1", "seeds": "2000", "decision": "rejected", "boss_cand": "40", "boss_base": "27"}]  # E1 rejudged
        self.assertEqual(gen_loop.lineage_blocks(rows), [(27, 2000)])  # one block, read by the rejudged decision (the incumbent stayed)
        more = [*rows, {"seed_block": "E2", "seeds": "2000", "decision": "promoted", "boss_cand": "50", "boss_base": "45"},
                {"seed_block": "E3", "seeds": "2000", "decision": "rejected", "boss_cand": "9", "boss_base": "52"},
                {"seed_block": "E4", "seeds": "2000", "decision": "promoted", "boss_cand": "48", "boss_base": "44"},
                {"seed_block": "E2", "seeds": "2000", "decision": "rejected", "boss_cand": "50", "boss_base": "45"}]  # E2 rejudged later
        # four blocks, three kept; E2 keeps its place among the blocks and its rejudged row; no block twice
        self.assertEqual(gen_loop.lineage_blocks(more), [(45, 2000), (52, 2000), (48, 2000)])
        self.assertEqual(gen_loop.lineage_blocks(more, blocks=4), [(27, 2000), (45, 2000), (52, 2000), (48, 2000)])
        # a later row without the counts does not replace a block's counted row
        self.assertEqual(gen_loop.lineage_blocks([*rows, {"seed_block": "E1", "seeds": "", "decision": "rejected"}]), [(27, 2000)])


class RetryCapTests(unittest.TestCase):
    def test_two_attempts_per_generation(self):
        self.assertEqual(pg.retry_cap(0), {"attempt": 1, "cap": 2, "allowed": True, "blocks": False})
        self.assertEqual(pg.retry_cap(1), {"attempt": 2, "cap": 2, "allowed": True, "blocks": False})
        self.assertEqual(pg.retry_cap(2), {"attempt": 3, "cap": 2, "allowed": False, "blocks": True})
        rows = [{"cand_gen": "2", "cand_sha": "a", "decision": "rejected"}, {"cand_gen": "2", "cand_sha": "a", "decision": "rejected"},
                {"cand_gen": "2", "cand_sha": "b", "decision": "rejected"}, {"cand_gen": "3", "cand_sha": "z", "decision": "rejected"}]
        self.assertEqual(gen_loop.attempts_before(rows, 2, "c"), 2)  # two other candidates (a judged twice counts once)
        self.assertEqual(gen_loop.attempts_before(rows, 2, "a"), 1)  # the same candidate again is not a new attempt
        self.assertEqual(gen_loop.attempts_before(rows, 4, "q"), 0)

    def _runs(self, root: Path, gen: int, n: int, tag: str):
        sink = gen_loop.ShardSink(root / f"job-{tag}", gen, config="a3-v1", base_sims=30, ascension=10, agent={})
        for i in range(n):
            sink.add({"seed": f"R{gen}{tag}{i:05d}"[:10], "generation": gen}, None)
        sink.close()

    def test_the_lab_grants_no_third_budget_with_the_guard_on_and_still_does_with_it_off(self):
        for on in (False, True):
            with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {pg.FLAG: "1" if on else "0"}):
                lab, runs, tsv = Path(d) / "lab", Path(d) / "runs", Path(d) / "promotions.tsv"
                self._runs(runs, 1, 15, "a")
                self.assertEqual(gen_loop.mark_ready(lab, 1, runs, budget=5, tsv=tsv), lab / "train" / "ready-1")
                gen_loop.append_promotion(tsv, {"cand_gen": 2, "decision": "rejected", "seed_block": "E1", "cand_sha": "a"})
                self.assertEqual(gen_loop.mark_ready(lab, 1, runs, budget=5, tsv=tsv), lab / "train" / "ready-1-r1")
                gen_loop.append_promotion(tsv, {"cand_gen": 2, "decision": "rejected", "seed_block": "E1-r1", "cand_sha": "b"})
                third = gen_loop.mark_ready(lab, 1, runs, budget=5, tsv=tsv)
                self.assertEqual(third, None if on else lab / "train" / "ready-1-r2", f"flag on={on}")
                # the explicit argument wins over the variable
                self.assertEqual(gen_loop.mark_ready(lab, 1, runs, budget=5, tsv=tsv, review_guard=True), None)


class PromoteBase(unittest.TestCase):
    """`promote` with the arms stubbed (as test_gen_loop.PromoteEndToEndTests), 400 seeds."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.seeds = seeds(400)
        self.seed_file = self.root / "seeds" / "Etest-1.json"
        self.seed_file.parent.mkdir()
        self.seed_file.write_text(json.dumps({"seeds": self.seeds}))
        self.agents = self.root / "agent"
        self.agents.mkdir()
        self.env = mock.patch.dict(os.environ, {})
        self.env.start()
        os.environ.pop(pg.FLAG, None)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def candidate(self, name: str, sha: str, generation: int = 5) -> Path:
        c = self.root / name
        c.mkdir(parents=True)
        (c / "model.json").write_text(json.dumps({"policy_sha256": sha}))
        (c / "combat_policy.pt").write_bytes(b"x")
        (c / "agent.json").write_text(json.dumps({"generation": generation, "macro": "rules"}))
        return c

    def arms(self, cand_kills: int, base_kills: int, damage=None, sd: list[str] | None = None):
        """Candidate two floors ahead everywhere (the primary passes); the first `cand_kills` seeds killed by the candidate
        only, the next `base_kills` by the incumbent only; `damage` = (cand fraction, base fraction) functions of the index."""
        sd = sd or self.seeds
        base_rows = rows_like(sd, lambda i: 3 + i % 12, damage and damage[1])
        cand_rows = rows_like(sd, lambda i: 5 + i % 12, damage and damage[0])
        for r in cand_rows[:cand_kills]:
            r["floor"] = 18
        for r in base_rows[cand_kills:cand_kills + base_kills]:
            r["floor"] = 18
        return cand_rows, base_rows

    def block(self, name: str) -> tuple[Path, list[str]]:
        """Another E-block, of its own seeds (a block judges one candidate only, and blocks share no seed)."""
        p = self.seed_file.with_name(f"{name}.json")
        sd = [f"{name}-{i:06d}" for i in range(len(self.seeds))]
        p.write_text(json.dumps({"seeds": sd}))
        return p, sd

    def run_promote(self, cand: Path, cand_rows, base_rows, seed_file: Path | None = None, **kw):
        def fake_run_arm(sd, arm_dir, agent, config, procs, ascension=10):
            return cand_rows if agent.prior_dir else base_rows

        with mock.patch.object(gen_loop, "run_arm", fake_run_arm):
            return gen_loop.promote(cand, None, seed_file or self.seed_file, agent_root=self.agents, boot=500, **kw)


class FlagOffTests(PromoteBase):
    def test_the_flag_is_off_unless_set_on(self):
        for v in (None, "", "0", "off", "no", "false", "maybe"):
            self.assertFalse(pg.enabled(v), v)
            self.assertFalse(pg.enabled(env={} if v is None else {pg.FLAG: v}), v)
        for v in ("1", "on", "ON", "true", "yes", " 1 "):
            self.assertTrue(pg.enabled(v), v)
            self.assertTrue(pg.enabled(env={pg.FLAG: v}), v)
        self.assertFalse(gen_loop.review_guard_on())
        self.assertEqual(gen_loop.REVIEW_GUARD_FLAG, pg.FLAG)
        with mock.patch.dict(os.environ, {pg.FLAG: "on"}):
            self.assertTrue(gen_loop.review_guard_on())
            self.assertFalse(gen_loop.review_guard_on(False))  # the explicit argument wins

    def test_with_the_flag_off_the_existing_decision_is_unchanged(self):
        # 8 candidate-only against 20 incumbent-only kills: P(X >= 20 | 28, 1/2) = 0.018, worse at 0.05 but not at D24's 0.01
        cand_rows, base_rows = self.arms(8, 20)
        self.assertLess(gen_loop.mcnemar_exact_one_sided(20, 8), 0.05)
        self.assertGreater(gen_loop.mcnemar_exact_one_sided(20, 8), 0.01)
        before = gen_loop.compare_arms(cand_rows, base_rows, boot=500)
        doc = self.run_promote(self.candidate("gen5c", "sha-a"), cand_rows, base_rows)
        self.assertEqual((doc["decision_text"], doc["decision"]["rule"]), ("promoted", "D24"))
        self.assertEqual(doc["decision"], before["decision"])  # the D24 decision, untouched
        self.assertEqual(set(doc["decision"]["checks"]), {"floor", "boss_kill"})
        self.assertNotIn("review_guard", doc)
        self.assertNotIn("review", doc["decision"])
        rows = gen_loop.read_promotions(self.agents / "promotions.tsv")
        self.assertEqual([(r["decision"], r["attempt"], r["drift"], r["guard"]) for r in rows], [("promoted", "1", "", "off")])
        self.assertTrue((self.agents / "current.json").is_file())
        # the same arms under the review's guard are rejected: the only difference is the flag
        (self.agents / "current.json").unlink()
        block2, sd2 = self.block("Etest-2")
        cand_rows, base_rows = self.arms(8, 20, sd=sd2)
        on = self.run_promote(self.candidate("gen5d", "sha-b"), cand_rows, base_rows, block2, review_guard=True)
        self.assertEqual((on["decision_text"], on["decision"]["rule"]), ("rejected", "D24+" + pg.RULE))
        self.assertEqual(on["review_guard"]["blocking"], ["kill"])
        self.assertFalse((self.agents / "current.json").exists())


class FlagOnTests(PromoteBase):
    def setUp(self):
        super().setUp()
        os.environ[pg.FLAG] = "1"

    def test_the_verdict_and_its_numbers_are_recorded_and_a_clean_candidate_is_promoted(self):
        cand_rows, base_rows = self.arms(30, 20, damage=(lambda i: 0.5 + (i % 5) / 10, lambda i: 0.45 + (i % 5) / 10))
        doc = self.run_promote(self.candidate("gen5c", "sha-a"), cand_rows, base_rows)
        self.assertEqual((doc["decision_text"], doc["decision"]["rule"]), ("promoted", "D24+" + pg.RULE))
        g = doc["review_guard"]
        self.assertEqual(g["rule"], pg.RULE)
        self.assertEqual((g["alpha"], g["blocks"], g["blocking"], g["review"]), (0.05, False, [], False))
        self.assertEqual((g["kill"]["b"], g["kill"]["c"], g["kill"]["pairs"], g["kill"]["fires"]), (30, 20, 400, False))
        self.assertTrue(g["damage"]["available"])
        self.assertAlmostEqual(g["damage"]["diff_mean"], 0.05)
        self.assertFalse(g["damage"]["fires"])
        self.assertEqual(g["drift"]["verdict"], "skipped")  # no lineage block yet
        self.assertEqual(g["attempt"], {"attempt": 1, "cap": 2, "allowed": True, "blocks": False})
        self.assertEqual(set(doc["decision"]["checks"]), {"floor", "boss_kill", "boss_damage", "drift", "attempt"})
        self.assertEqual(doc["decision"]["guard_alpha"], 0.05)
        row = gen_loop.read_promotions(self.agents / "promotions.tsv")[0]
        self.assertEqual((row["decision"], row["attempt"], row["drift"]), ("promoted", "1", "skipped"))
        self.assertIn("kill ok (c 20 b 30", row["guard"])
        self.assertIn("damage ok", row["guard"])
        self.assertIn("attempt 1/2", row["guard"])
        self.assertNotIn("\t", row["guard"])
        self.assertEqual(json.loads((self.agents / "current.json").read_text())["generation"], 5)

    def test_the_damage_guard_rejects_a_candidate_that_hurts_the_boss_less(self):
        cand_rows, base_rows = self.arms(22, 20, damage=(lambda i: 0.3 + (i % 5) / 10, lambda i: 0.5 + (i % 5) / 10))
        doc = self.run_promote(self.candidate("gen5c", "sha-a"), cand_rows, base_rows)
        self.assertEqual(doc["decision_text"], "rejected")
        self.assertEqual(doc["review_guard"]["blocking"], ["damage"])
        self.assertEqual(doc["decision"]["checks"]["boss_damage"], False)
        self.assertEqual(doc["decision"]["checks"]["boss_kill"], True)
        self.assertFalse((self.agents / "current.json").exists())

    def test_the_drift_check_flags_then_blocks_against_the_lineage_in_the_tsv(self):
        tsv = self.agents / "promotions.tsv"
        for i, (block, kc, kb) in enumerate((("E1", 60, 55), ("E2", 62, 60), ("E3", 61, 62))):  # a lineage killing about 15% of 400
            gen_loop.append_promotion(tsv, {"cand_gen": i + 2, "base_gen": i + 1, "seed_block": block, "seeds": 400, "boss_cand": kc,
                                            "boss_base": kb, "decision": "promoted", "cand_sha": f"s{i}", "attempt": 1, "drift": "clear"})
        cand_rows, base_rows = self.arms(16, 18)  # the candidate kills on 16 of 400 (4%): far below the pooled 15%
        doc = self.run_promote(self.candidate("gen5c", "sha-a"), cand_rows, base_rows)
        g = doc["review_guard"]
        self.assertEqual(g["drift"]["lineage"], [[60, 400], [62, 400], [61, 400]])
        self.assertEqual((g["drift"]["verdict"], g["review"], g["blocks"]), ("flagged", True, False))
        self.assertEqual((doc["decision_text"], doc["decision"]["review"]), ("promoted", True))  # flagged for the weekly review, not blocked
        self.assertEqual(gen_loop.read_promotions(tsv)[-1]["drift"], "flagged")
        # the next candidate, flagged too, is blocked
        again = self.candidate("gen6c", "sha-b", generation=6)
        (self.agents / "current.json").unlink()
        block2, sd2 = self.block("Etest-2")
        cand_rows, base_rows = self.arms(16, 18, sd=sd2)
        doc2 = self.run_promote(again, cand_rows, base_rows, block2)
        self.assertEqual((doc2["review_guard"]["drift"]["verdict"], doc2["review_guard"]["drift"]["previous_flagged"]), ("blocked", True))
        self.assertEqual((doc2["decision_text"], doc2["review_guard"]["blocking"]), ("rejected", ["drift"]))
        self.assertEqual(gen_loop.read_promotions(tsv)[-1]["drift"], "blocked")
        self.assertFalse((self.agents / "current.json").exists())

    def test_a_third_attempt_is_refused_before_a_run_is_played(self):
        tsv = self.agents / "promotions.tsv"
        gen_loop.append_promotion(tsv, {"cand_gen": 5, "seed_block": "Ex", "seeds": 400, "decision": "rejected", "cand_sha": "s1"})
        gen_loop.append_promotion(tsv, {"cand_gen": 5, "seed_block": "Ey", "seeds": 400, "decision": "rejected", "cand_sha": "s2"})
        cand = self.candidate("gen5c", "sha-a")
        with mock.patch.object(gen_loop, "run_arm", side_effect=AssertionError("must not play")), \
                self.assertRaisesRegex(ValueError, "already had 2 promotion attempts"):
            gen_loop.promote(cand, None, self.seed_file, agent_root=self.agents, boot=500)
        self.assertEqual(len(gen_loop.read_promotions(tsv)), 2)
        # with the flag off the third attempt is played and judged as before (D24 has no cap)
        cand_rows, base_rows = self.arms(30, 20)
        doc = self.run_promote(cand, cand_rows, base_rows, review_guard=False)
        self.assertEqual((doc["decision"]["rule"], gen_loop.read_promotions(tsv)[-1]["attempt"]), ("D24", "3"))

    def test_rejudge_reads_the_block_under_the_guard_without_counting_an_attempt(self):
        cand = self.candidate("gen5c", "sha-a")
        cand_rows, base_rows = self.arms(8, 20)  # worse at 0.05: rejected under the guard
        for arm_dir, rows in ((cand / "arms" / "Etest-1-a3-v1", cand_rows), (self.agents / "gen0" / "arms" / "Etest-1-a3-v1", base_rows)):
            arm_dir.mkdir(parents=True)
            (arm_dir / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        gen_loop.append_promotion(self.agents / "promotions.tsv", {"cand_gen": 5, "seed_block": "Etest-1", "seeds": 400, "boss_cand": 8,
                                                                  "boss_base": 20, "decision": "promoted", "cand_sha": "sha-a", "attempt": 1})
        with mock.patch.object(gen0, "run_block", side_effect=AssertionError("must not play again")):
            doc = gen_loop.rejudge(cand, None, self.seed_file, agent_root=self.agents, note="review guard", boot=500)
        self.assertEqual((doc["decision_text"], doc["decision"]["rule"]), ("rejected", "D24+" + pg.RULE))
        self.assertEqual(doc["review_guard"]["attempt"]["attempt"], 1)  # its own row does not count as another candidate
        self.assertEqual(doc["review_guard"]["drift"]["verdict"], "skipped")  # its own block is not lineage
        rows = gen_loop.read_promotions(self.agents / "promotions.tsv")
        self.assertEqual([(r["decision"], r["attempt"]) for r in rows], [("promoted", "1"), ("rejected", "1")])

    def test_the_subprocess_command_carries_the_flag(self):
        with mock.patch.object(gen_loop.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("E.json"), Path("a"), 4, None, None, review_guard=True)
            self.assertIn("--review-guard", run.call_args[0][0])
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("E.json"), Path("a"), 4, None, None)
            self.assertNotIn("--review-guard", run.call_args[0][0])
        ap = gen0.build_parser()
        for cmd in (["promote", "--cand", "c", "--agent-root", "a"], ["rejudge", "--cand", "c", "--seeds-file", "s", "--agent-root", "a", "--note", "n"],
                    ["train", "--shards-root", "r", "--agent-root", "a"]):
            self.assertFalse(ap.parse_args(cmd).review_guard)
            self.assertTrue(ap.parse_args([*cmd, "--review-guard"]).review_guard)


if __name__ == "__main__":
    unittest.main()
