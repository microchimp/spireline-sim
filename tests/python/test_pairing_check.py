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

    # -- a block the funnel stopped before its last look holds only the seeds played (promotion.json `sequential`)

    STOP = {"action": "refuse", "played": 300, "block_seeds": N}

    def stopped(self):
        seeds, cand, base = arms(True)
        return seeds, cand[:300], base[:300]

    def test_an_early_stop_is_read_as_short_without_the_record_and_as_the_seeds_played_with_it(self):
        seeds, cand, base = self.stopped()
        summ = {"config": "a3-v1", "ascension": 10}
        short = pc.check_pair(cand, base, seed_file=seeds, cand_summary=summ, base_summary=summ, permutations=50)
        self.assertFalse(short["seeds"]["pass"])  # what it did before: 300 paired of 1500
        r = pc.check_pair(cand, base, seed_file=seeds, early_stop=self.STOP, cand_summary=summ, base_summary=summ, permutations=50)
        self.assertTrue(r["seeds"]["pass"], r["seeds"])
        self.assertEqual((r["seeds"]["n_seeds"], r["seeds"]["paired"], r["seeds"]["paired_seeds"]), (300, 300, 300))
        self.assertEqual(r["seeds"]["early_stop"], {"action": "refuse", "played": 300, "block_seeds": N, "seed_file_is_the_block": True, "pass": True})
        self.assertIn("funnel stop on a refuse: the first 300 of the block's 1500 seeds were played", pc._fmt("b", r))

    def test_an_early_stop_does_not_excuse_a_wrong_seed_set(self):
        seeds, cand, base = self.stopped()
        # the arms hold seeds past the first 300 of the file: not what the record says was played
        r = pc.check_pair(cand + arms(True)[1][300:350], base + arms(True)[2][300:350], seed_file=seeds, early_stop=self.STOP, permutations=50)
        self.assertFalse(r["seeds"]["pass"])
        # the arms hold 300 seeds but not the first 300 of the file
        r = pc.check_pair(arms(True)[1][200:500], arms(True)[2][200:500], seed_file=seeds, early_stop=self.STOP, permutations=50)
        self.assertFalse(r["seeds"]["pass"])
        # one arm short of the seeds played
        r = pc.check_pair(cand, base[:290], seed_file=seeds, early_stop=self.STOP, permutations=50)
        self.assertFalse(r["seeds"]["pass"])
        # a seed file that is not the block the record names
        r = pc.check_pair(cand, base, seed_file=seeds[:1200], early_stop=self.STOP, permutations=50)
        self.assertFalse(r["seeds"]["early_stop"]["pass"])
        self.assertFalse(r["seeds"]["pass"])
        # promotion.json's own paired_seeds below the seeds played still fails
        r = pc.check_pair(cand, base, seed_file=seeds, early_stop=self.STOP, paired_seeds=299, permutations=50)
        self.assertFalse(r["seeds"]["pass"])

    def test_the_record_is_taken_from_promotion_json_given_or_found_beside_the_candidates_arm(self):
        seeds, cand, base = self.stopped()
        promo = {"seed_block": "Eblock", "n_seeds": 300, "paired_seeds": 300,
                 "sequential": {"stop": {"action": "refuse", "n": 300}, "played": 300, "block_seeds": N}}
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            arm = t / "cand-gen" / "arms" / "Eblock-a3-v1"
            base_arm = t / "base-gen" / "arms" / "Eblock-a3-v1"
            for d, rows in ((arm, cand), (base_arm, base)):
                d.mkdir(parents=True)
                (d / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
                (d / "summary.json").write_text(json.dumps({"config": "a3-v1", "ascension": 10}), encoding="utf-8")
            (t / "Eblock.json").write_text(json.dumps({"seeds": seeds}), encoding="utf-8")
            (t / "cand-gen" / "promotion.json").write_text(json.dumps(promo), encoding="utf-8")
            # no --promotion: the candidate's own record is found, because it is the record of this block
            r = pc.check_dirs(arm, base_arm, seeds_file=t / "Eblock.json", permutations=50)
            self.assertTrue(r["seeds"]["pass"], r["seeds"])
            self.assertEqual(r["seeds"]["early_stop"]["played"], 300)
            # a record of another block is not taken: the arms then read as short, as before
            (t / "cand-gen" / "promotion.json").write_text(json.dumps({**promo, "seed_block": "Eother"}), encoding="utf-8")
            self.assertFalse(pc.check_dirs(arm, base_arm, seeds_file=t / "Eblock.json", permutations=50)["seeds"]["pass"])
            # given explicitly, it is taken as given; the command line prints the stop
            (t / "p.json").write_text(json.dumps(promo), encoding="utf-8")
            r = pc.check_dirs(arm, base_arm, seeds_file=t / "Eblock.json", promotion=t / "p.json", permutations=50)
            self.assertTrue(r["seeds"]["pass"])
            out = t / "out.json"
            pc.main([str(arm), str(base_arm), "--seeds-file", str(t / "Eblock.json"), "--promotion", str(t / "p.json"), "--json", str(out)])
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["seeds"]["early_stop"]["played"], 300)

    def test_a_block_that_played_everything_is_checked_as_before(self):
        seq = {"stop": {"action": "final", "n": N}, "played": N, "block_seeds": N}
        self.assertIsNone(pc.funnel_stop({"sequential": seq}))
        self.assertIsNone(pc.funnel_stop({"n_seeds": N}))
        self.assertIsNone(pc.funnel_stop(None))
        seeds, cand, base = arms(True)
        r = pc.check_pair(cand, base, seed_file=seeds, early_stop=pc.funnel_stop({"sequential": seq}), cand_summary={"config": "a3-v1"},
                          base_summary={"config": "a3-v1"}, permutations=50)
        self.assertNotIn("early_stop", r["seeds"])
        self.assertTrue(r["seeds"]["pass"])


if __name__ == "__main__":
    unittest.main()
