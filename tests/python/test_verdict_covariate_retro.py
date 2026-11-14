"""The seed-difficulty retrospective (tools/verdict_covariate_retro.py, D60 item 4): the covariate-adjusted paired estimate, the ceiling of
any difficulty covariate in a paired design, the leave-the-judged-pair-out index and the block report on a constructed agent folder.

Constructed data only: synthetic differences with an exact mean and an exactly orthogonal noise, and a temporary agent folder of arms
written here. No saved row of the lab is read.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

from tools import verdict_covariate_retro as vc  # noqa: E402


def orthogonal_noise(n: int, x: np.ndarray, sd: float, seed: int = 7) -> np.ndarray:
    """Zero-mean noise, exactly orthogonal to the (centered) x and the constant, with the sample SD `sd` (n - 1)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    basis = np.stack([np.ones(n), x - x.mean()], axis=1)
    e = e - basis @ np.linalg.lstsq(basis, e, rcond=None)[0]
    return e * (sd / e.std(ddof=1))


class CupedTests(unittest.TestCase):
    N = 400

    def block(self, gamma: float, delta: float = 0.3, sd: float = 2.0, shift: float = 0.0):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(self.N)
        x = (x - x.mean()) / x.std(ddof=1) + shift  # the block's seeds sit `shift` above the population (mean 0)
        e = orthogonal_noise(self.N, x, sd)
        return delta + gamma * (x - x.mean()) + e, x

    def test_a_covariate_that_explains_the_differences_cuts_the_variance_by_the_exact_share(self):
        gamma, sd = 1.5, 2.0
        d, x = self.block(gamma, sd=sd)
        out = vc.cuped_adjust(d, x, 0.0)
        self.assertTrue(out["used"])
        self.assertAlmostEqual(out["theta"], gamma, places=9)
        n = self.N
        tss = gamma ** 2 * float(((x - x.mean()) ** 2).sum()) + sd ** 2 * (n - 1)
        want = (sd ** 2 * (n - 1) / (n - 2)) / (tss / (n - 1))  # the residual variance (n - 2 df) over the plain one (n - 1)
        self.assertAlmostEqual(out["var_ratio"], want, places=9)
        self.assertLess(out["var_ratio"], 0.65)  # gamma^2 = 2.25 of 6.25 explained: 0.64 of the variance is left
        self.assertLess(out["se"], out["plain_se"])

    def test_the_adjusted_mean_corrects_a_block_whose_seeds_sit_off_the_population_mean(self):
        gamma, delta = 1.5, 0.3
        d, x = self.block(gamma, delta=delta, shift=0.5)
        out = vc.cuped_adjust(d, x, 0.0)
        self.assertAlmostEqual(out["plain_mean"], delta, places=9)  # the sample mean of d: the block's own
        self.assertAlmostEqual(out["mean"], delta - gamma * 0.5, places=9)  # the block's seeds were 0.5 above the population: corrected
        self.assertEqual(len(out["diffs"]), self.N)
        self.assertAlmostEqual(float(np.mean(out["diffs"])), out["mean"], places=9)

    def test_the_adjusted_estimator_has_the_lower_sampling_variance(self):
        rng = np.random.default_rng(11)
        gamma, delta, sd, n, reps = 1.0, 0.2, 1.0, 200, 1500
        plain, adj = [], []
        for _ in range(reps):
            x = rng.standard_normal(n)  # the population mean is 0
            d = delta + gamma * x + sd * rng.standard_normal(n)
            out = vc.cuped_adjust(d, x, 0.0)
            plain.append(out["plain_mean"])
            adj.append(out["mean"])
        self.assertAlmostEqual(float(np.mean(adj)), delta, delta=0.02)  # unbiased
        ratio = float(np.var(adj, ddof=1) / np.var(plain, ddof=1))
        self.assertLess(ratio, 0.65)  # theory: sd^2 / (gamma^2 + sd^2) = 0.5
        self.assertGreater(ratio, 0.35)

    def test_a_useless_covariate_falls_back_to_the_plain_estimate(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(self.N)
        e = orthogonal_noise(self.N, x, 2.0, seed=9)
        d = 0.3 + e  # exactly uncorrelated with x
        out = vc.cuped_adjust(d, x, 0.0)
        self.assertFalse(out["used"])
        self.assertIn("explains nothing", out["reason"])
        self.assertEqual(out["se_ratio"], 1.0)
        self.assertEqual(out["mean"], out["plain_mean"])
        self.assertEqual(out["se"], out["plain_se"])
        self.assertTrue(np.array_equal(out["diffs"], d))

    def test_a_covariate_that_is_noise_is_almost_never_used(self):
        rng = np.random.default_rng(17)
        used = 0
        for _ in range(300):
            x, d = rng.standard_normal(150), rng.standard_normal(150)
            used += bool(vc.cuped_adjust(d, x, 0.0)["used"])
        self.assertLessEqual(used / 300, 0.10)  # |t| >= 2 on pure noise: about 5 in 100

    def test_the_fallbacks_when_the_covariate_is_missing_or_unusable(self):
        d = np.linspace(-1, 1, 100)
        x = np.linspace(0, 1, 100)
        plain = vc.cuped_adjust(d, x, 0.5)
        self.assertTrue(plain["used"])
        for label, args in (("no covariate", (d, None, 0.5)), ("no covariate (mean unknown)", (d, x, None)), ("no covariate (NaN mean)", (d, x, math.nan)),
                            ("missing on some seeds", (d, np.where(np.arange(100) == 3, math.nan, x), 0.5)),
                            ("constant", (d, np.ones(100), 0.5)), ("few seeds", (d[:20], x[:20], 0.5))):
            out = vc.cuped_adjust(*args)
            self.assertFalse(out["used"], label)
            self.assertEqual(out["var_ratio"], 1.0, label)
            self.assertEqual(out["mean"], out["plain_mean"], label)


class CeilingTests(unittest.TestCase):
    def test_equal_responses_leave_nothing_to_remove(self):
        # c = 1.2 u + e, b = 1.2 u + e: the paired difference has no u in it
        out = vc.ceiling_from_moments(1.44 + 1.0, 1.44 + 1.0, 1.44)
        self.assertAlmostEqual(out["share"], 0.0, places=9)
        self.assertAlmostEqual(out["rho"], 1.44 / 2.44, places=9)

    def test_unequal_responses_leave_the_exact_share(self):
        p, q, s2 = 1.3, 1.0, 1.0
        out = vc.ceiling_from_moments(p * p + s2, q * q + s2, p * q)
        self.assertAlmostEqual(out["p"], p, places=9)
        self.assertAlmostEqual(out["q"], q, places=9)
        self.assertAlmostEqual(out["noise_sd"], math.sqrt(s2), places=9)
        self.assertAlmostEqual(out["share"], (p - q) ** 2 / ((p - q) ** 2 + 2 * s2), places=9)  # 0.09 / 2.09

    def test_no_shared_difficulty_and_scale_invariance(self):
        self.assertEqual(vc.ceiling_from_moments(4.0, 4.0, 0.0)["share"], 0.0)
        self.assertEqual(vc.ceiling_from_moments(4.0, 4.0, -0.5)["share"], 0.0)
        a = vc.ceiling_from_moments(2.69, 2.0, 1.3)["share"]
        b = vc.ceiling_from_moments(100 * 2.69, 100 * 2.0, 100 * 1.3)["share"]
        self.assertAlmostEqual(a, b, places=12)

    def test_the_moments_of_simulated_arms_give_the_ceiling_back(self):
        rng = np.random.default_rng(23)
        n, p, q, s = 400_000, 1.5, 1.0, 1.2
        u = rng.standard_normal(n)
        c, b = p * u + s * rng.standard_normal(n), q * u + s * rng.standard_normal(n)
        want = (p - q) ** 2 / ((p - q) ** 2 + 2 * s * s)
        out = vc.ceiling_from_moments(*vc.moments(c, b)[:3])
        self.assertAlmostEqual(out["share"], want, delta=0.01)
        # and an oracle covariate u really does remove that share of the difference's variance
        d = c - b
        adj = vc.cuped_adjust(d, u, 0.0)
        self.assertAlmostEqual(1.0 - adj["var_ratio"], want, delta=0.01)

    def test_the_proportional_model_reads_a_rate_difference_not_a_noise_difference(self):
        # two arms of counts with difficulty scaling the rates: lam_c = 0.06, lam_b = 0.03, difficulty f lognormal with mean 1
        rng = np.random.default_rng(31)
        n, lam_c, lam_b = 600_000, 0.06, 0.03
        f = np.exp(0.8 * rng.standard_normal(n) - 0.32)  # E f = 1
        kc, kb = rng.poisson(lam_c * f), rng.poisson(lam_b * f)
        vc_, vb_, cov, _ = vc.moments(kc, kb)
        out = vc.ceiling_from_moments(vc_, vb_, cov, float(kc.mean()), float(kb.mean()))
        d = kc - kb
        true_share = float(np.var((lam_c - lam_b) * f, ddof=1) / np.var(d, ddof=1))
        self.assertAlmostEqual(out["share_prop"], true_share, delta=0.01)
        self.assertGreater(out["share"], out["share_prop"])  # the additive model reads the rates' noise as response and overstates
        self.assertEqual(out["share_max"], max(out["share"], out["share_prop"]))
        adj = vc.cuped_adjust(d, f, 1.0)  # an oracle covariate takes out the proportional share
        self.assertAlmostEqual(1.0 - adj["var_ratio"], true_share, delta=0.01)

    def test_the_proportional_model_needs_means_and_equal_rates_give_nothing(self):
        self.assertEqual(vc.ceiling_from_moments(2.0, 2.0, 1.0)["share_prop"], 0.0)  # no means given
        self.assertEqual(vc.ceiling_from_moments(2.0, 2.0, 1.0, 0.5, 0.5)["share_prop"], 0.0)  # the same rate
        got = vc.ceiling_from_moments(3.0, 2.0, 1.0, 1.5, 1.0)
        self.assertAlmostEqual(got["share_prop"], (0.5 ** 2 * 1.0) / (1.5 * 1.0 * (3.0 + 2.0 - 2.0)), places=12)

    def test_pooled_share_weights_each_block_by_its_variance_and_seeds(self):
        parts = [{"share_max": 0.10, "var_d": 4.0, "n": 101}, {"share_max": 0.0, "var_d": 4.0, "n": 301}]
        self.assertAlmostEqual(vc.pooled_share(parts), (0.10 * 4.0 * 100) / (4.0 * 100 + 4.0 * 300), places=12)
        self.assertEqual(vc.pooled_share([]), 0.0)

    def test_pooled_moments_are_the_df_weighted_mean(self):
        got = vc.pooled_moments([(2.0, 2.0, 1.0, 101), (4.0, 4.0, 3.0, 301)])
        self.assertAlmostEqual(got[0], (2.0 * 100 + 4.0 * 300) / 400)
        self.assertAlmostEqual(got[2], (1.0 * 100 + 3.0 * 300) / 400)
        self.assertTrue(math.isnan(vc.pooled_moments([])[0]))


def put_arm(path: Path, rows: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def row(seed: str, ix: int, floor: int, acts=None, win=False) -> dict:
    return {"seed": seed, "ix": ix, "floor": floor, "win": win, "boss_damage_acts": acts if acts is not None else []}


class AgentFolderTests(unittest.TestCase):
    """A constructed agent folder: gen1's champion and gen2's candidate on block EX (the judged pair), a third arm on the same seeds."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.agent = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.seeds = [f"S{i:03d}" for i in range(240)]
        rng = np.random.default_rng(2)
        self.hard = rng.integers(0, 2, size=len(self.seeds))  # a seed's difficulty (0 easy, 1 hard)
        champ = [row(s, i, int(24 - 8 * self.hard[i] + rng.integers(-2, 3))) for i, s in enumerate(self.seeds)]
        cand = [row(s, i, int(25 - 8 * self.hard[i] + rng.integers(-2, 3))) for i, s in enumerate(self.seeds)]
        put_arm(self.agent / "gen1" / "candidate" / "arms" / "EX-a3-v1", champ)
        put_arm(self.agent / "gen2" / "candidate-z" / "arms" / "EX-a3-v1", cand)
        self.ledger = ["\t".join(("written", "cand_gen", "base_gen", "seed_block", "seeds", "decision", "cand_dir", "guard", "rule", "frontier", "route"))]
        self.ledger.append("\t".join(("2999-01-01T00:00:00Z", "2", "1", "EX", str(len(self.seeds)), "rejected", "candidate-z", "", "v2", "B1", "")))
        (self.agent / "promotions.tsv").write_text("\n".join(self.ledger) + "\n", encoding="utf-8")

    def third_arm(self, name: str = "other") -> None:
        rng = np.random.default_rng(4)
        put_arm(self.agent / "ab" / name / "arms" / "EX-a3-v1", [row(s, i, int(24 - 8 * self.hard[i] + rng.integers(-2, 3))) for i, s in enumerate(self.seeds)])

    def test_the_arms_of_a_ledger_row_are_found_and_the_pair_is_excluded_from_the_index(self):
        rows = vc.judged_blocks(vc.read_ledger(self.agent / "promotions.tsv"))
        cand, base = vc.arm_dirs(self.agent, rows[0])
        self.assertEqual(cand.parent.parent.name, "candidate-z")
        self.assertEqual(base.parent.parent.name, "candidate")
        self.assertEqual(vc.other_arms_index(self.agent, (cand, base), None, 0), {})  # nothing but the judged pair: no covariate
        self.third_arm()
        index = vc.other_arms_index(self.agent, (cand, base), None, 0)
        self.assertEqual(len(index), len(self.seeds))
        self.assertTrue(all(len(v) == 1 for v in index.values()))

    def test_an_arm_stored_after_the_judgement_is_not_a_covariate_of_it(self):
        self.third_arm()
        rows = vc.judged_blocks(vc.read_ledger(self.agent / "promotions.tsv"))
        cand, base = vc.arm_dirs(self.agent, rows[0])
        self.assertEqual(vc.other_arms_index(self.agent, (cand, base), "2000-01-01T00:00:00Z", 0), {})  # written 2026, the judgement 2000
        self.assertEqual(len(vc.other_arms_index(self.agent, (cand, base), "2999-01-01T00:00:00Z", 0)), len(self.seeds))

    def test_a_block_no_other_arm_has_played_falls_back_and_saves_no_look(self):
        row_ = vc.judged_blocks(vc.read_ledger(self.agent / "promotions.tsv"))[0]
        rep = vc.block_report(self.agent, row_)
        self.assertEqual(rep["n"], len(self.seeds))
        self.assertEqual(rep["covered"]["floor"], 0)
        self.assertFalse(rep["other_arms_on_judged_seeds"])
        for m in vc.METRICS:
            self.assertFalse(rep["adjusted"][m]["used"])
            self.assertEqual(rep["adjusted"][m]["var_ratio"], 1.0)
        self.assertEqual(rep["looks_saved"], 0)
        self.assertEqual(rep["plain_replay"]["stop"], rep["adjusted_replay"]["stop"])
        self.assertEqual(vc.audit_all_blocks(self.agent, vc.read_ledger(self.agent / "promotions.tsv")),
                         {"blocks": 1, "seeds": len(self.seeds), "seeds_with_another_arm": 0})

    def test_a_third_arm_on_the_seeds_is_the_covariate_and_the_report_says_so(self):
        self.third_arm()
        row_ = vc.judged_blocks(vc.read_ledger(self.agent / "promotions.tsv"))[0]
        rep = vc.block_report(self.agent, row_)
        self.assertEqual(rep["covered"]["floor"], len(self.seeds))
        self.assertTrue(rep["other_arms_on_judged_seeds"])
        # the population mean needs seeds outside the block: here there are none, so the estimator falls back rather than guess it
        self.assertFalse(rep["adjusted"]["floor"]["used"])
        self.assertEqual(rep["adjusted"]["floor"]["reason"], "no covariate")
        put_arm(self.agent / "ab" / "wide" / "arms" / "EY-a3-v1", [row(f"T{i:03d}", i, int(20 + (i % 5) - 2)) for i in range(300)])
        rep = vc.block_report(self.agent, row_)  # now the population mean is known (seeds outside the block)
        # the arms differ by a constant everywhere, so seed difficulty explains none of the paired difference: the covariate is not fitted
        self.assertFalse(rep["adjusted"]["floor"]["used"])
        self.assertIn("explains nothing", rep["adjusted"]["floor"]["reason"])
        self.assertEqual(rep["adjusted"]["floor"]["var_ratio"], 1.0)

    def test_the_render_has_no_path_and_no_seed(self):
        row_ = vc.judged_blocks(vc.read_ledger(self.agent / "promotions.tsv"))[0]
        rep = vc.block_report(self.agent, row_)
        text = vc.render([rep], vc.audit_all_blocks(self.agent, vc.read_ledger(self.agent / "promotions.tsv")))
        self.assertIn("EX", text)
        self.assertNotIn(self.tmp.name, text)
        self.assertNotIn("S001", text)
        self.assertIn("0 of them played by any stored arm", text)


if __name__ == "__main__":
    unittest.main()
