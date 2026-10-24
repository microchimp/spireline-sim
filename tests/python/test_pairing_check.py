"""tools/lab/pairing_check.py (docs/PROMOTION_GUARD.md section 4, points 1 and 2) on constructed arms only.

Paired arms share a per-seed latent difficulty and so pass; the same arms with the candidate's seeds shuffled against the
incumbent's (independent) fail the floor bound and sit at the independent discordance; a seed-set mismatch, a config
mismatch and an uneven crash count each fail their own quantity.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.lab import pairing_check as pc  # noqa: E402

N = 1500


def make_rows(seeds, latent, rng, floor_noise=1.5, kill_noise=0.05, kill_cut=0.9):
    rows = []
    for i, (s, u) in enumerate(zip(seeds, latent)):
        floor = int(np.clip(round(4 + 14 * u + rng.normal(0, floor_noise)), 1, 20))
        killed = bool(u + rng.normal(0, kill_noise) > kill_cut)
        rows.append({"seed": s, "ix": i, "win": False, "floor": 18 if killed else min(floor, 17), "crash": False})
    return rows


def arms(paired: bool, seed: int = 7):
    rng = np.random.default_rng(seed)
    seeds = [f"S{i:05d}" for i in range(N)]
    u_base = rng.random(N)
    u_cand = u_base if paired else rng.random(N)
    return seeds, make_rows(seeds, u_cand, rng), make_rows(seeds, u_base, rng)


class PairingCheck(unittest.TestCase):
    def test_paired_arms_pass(self):
        seeds, cand, base = arms(True)
        r = pc.check_pair(cand, base, seed_file=seeds, cand_summary={"config": "a3-v1", "ascension": 10},
                          base_summary={"config": "a3-v1", "ascension": 10}, permutations=300)
        self.assertTrue(r["pass"], pc.failing(r))
        self.assertLess(r["kills"]["discordant"], r["kills"]["expected_independent"])
        self.assertLess(r["kills"]["perm_p_low"], 0.01)
        self.assertLess(r["floors"]["ratio"], pc.SD_RATIO_BOUND)
        self.assertGreater(r["floors"]["pearson_r"], 0.5)

    def test_independent_arms_fail_the_floor_bound(self):
        seeds, cand, base = arms(False)
        r = pc.check_pair(cand, base, seed_file=seeds, cand_summary={"config": "a3-v1"}, base_summary={"config": "a3-v1"}, permutations=300)
        self.assertFalse(r["floors"]["pass"])
        self.assertGreater(r["floors"]["ratio"], 0.95)
        self.assertGreater(r["kills"]["perm_p_low"], 0.05)  # the pairing does not show against the shuffled reference
        self.assertGreater(r["kills"]["ratio"], 0.8)
        self.assertFalse(r["pass"])

    def test_seed_set_and_count_mismatch_fail(self):
        seeds, cand, base = arms(True)
        r = pc.check_pair(cand[:-3], base, seed_file=seeds, cand_summary={"config": "a3-v1"}, base_summary={"config": "a3-v1"}, permutations=50)
        self.assertFalse(r["seeds"]["pass"])
        self.assertEqual((r["seeds"]["only_base"], r["seeds"]["paired"]), (3, N - 3))
        # promotion.json's paired_seeds below n_seeds fails even when the sets are identical
        r = pc.check_pair(cand, base, n_seeds=N, paired_seeds=N - 1, cand_summary={"config": "a3-v1"}, base_summary={"config": "a3-v1"}, permutations=50)
        self.assertFalse(r["seeds"]["pass"])
        # arms equal but not equal to the block's seed file
        r = pc.check_pair(cand, base, seed_file=seeds[:-1], cand_summary={"config": "a3-v1"}, base_summary={"config": "a3-v1"}, permutations=50)
        self.assertFalse(r["seeds"]["pass"])

    def test_config_and_crashes(self):
        seeds, cand, base = arms(True)
        r = pc.check_pair(cand, base, cand_summary={"config": "a3-v1", "ascension": 10}, base_summary={"config": "a3-v1", "ascension": 8}, permutations=50)
        self.assertEqual(r["config"]["mismatch"], {"ascension": [10, 8]})
        self.assertFalse(r["config"]["pass"])
        self.assertEqual(r["config"]["unchecked"], ["sims", "character", "agent_seed"])
        for row in cand[:30]:
            row["crash"] = True  # 2% of the block, all in one arm: neither small nor alike
        r = pc.check_pair(cand, base, cand_summary={"config": "a3-v1"}, base_summary={"config": "a3-v1"}, permutations=50)
        self.assertEqual((r["crashes"]["cand"], r["crashes"]["base"]), (30, 0))
        self.assertFalse(r["crashes"]["pass"])

    def test_sign_test_and_power(self):
        self.assertEqual(pc.sign_test_two_sided(0, 0), 1.0)
        self.assertAlmostEqual(pc.sign_test_two_sided(0, 6), 2 / 64)
        self.assertAlmostEqual(pc.sign_test_two_sided(3, 3), 1.0)
        # the scaled power at factor 1 is the table's independent power; fewer discordant pairs lower it
        rows = pc.power_with_discordance(1.0, n=500, alphas=(0.05,), cases=((0.02, 0.01),))
        self.assertAlmostEqual(rows[0]["scaled"], rows[0]["independent"], places=9)
        self.assertLess(pc.power_with_discordance(0.7, n=500, alphas=(0.05,), cases=((0.02, 0.01),))[0]["scaled"], rows[0]["independent"])

    def test_dirs_and_cli(self):
        seeds, cand, base = arms(True)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for name, rows in (("cand", cand), ("base", base)):
                (t / name).mkdir()
                (t / name / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
                (t / name / "summary.json").write_text(json.dumps({"config": "a3-v1", "ascension": 10}), encoding="utf-8")
            (t / "real").mkdir()  # a real-game file carries event lines and rows without win/crash; the last row of a seed wins
            (t / "real" / "runs.jsonl").write_text('{"ev": "start", "seed": "S00000"}\n{"ev": "end", "seed": "S00000", "floor": 5}\n'
                                                   '{"ev": "end", "seed": "S00000", "floor": 7}\n', encoding="utf-8")
            (t / "seeds.json").write_text(json.dumps({"seeds": seeds}), encoding="utf-8")
            (t / "promotion.json").write_text(json.dumps({"n_seeds": N, "paired_seeds": N}), encoding="utf-8")
            r = pc.check_dirs(t / "cand", t / "base", seeds_file=t / "seeds.json", promotion=t / "promotion.json", permutations=50)
            self.assertTrue(r["pass"])
            out = t / "out.json"
            self.assertEqual(pc.main([str(t / "cand"), str(t / "base"), "--seeds-file", str(t / "seeds.json"), "--json", str(out)]), 0)
            self.assertTrue(json.loads(out.read_text(encoding="utf-8"))["pass"])
            rows = pc.read_rows(t / "real" / "runs.jsonl")
            self.assertEqual([(x["seed"], x["floor"]) for x in rows], [("S00000", 7)])


if __name__ == "__main__":
    unittest.main()
