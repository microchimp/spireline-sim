"""A2c: V_run's DESIGN 5.3 keep conditions (rl/run_train/keep.py) on synthetic data, and the agent.json macro switch."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))

import gen_loop  # noqa: E402
from run_train import keep  # noqa: E402


def synthetic(n_runs: int = 300, rows: int = 6, rate: float = 0.2, seed: int = 1):
    rng = np.random.default_rng(seed)
    run = np.repeat(np.arange(n_runs), rows)
    y = np.repeat((rng.random(n_runs) < rate).astype(np.float64), rows)
    return run, y, rng


class KeepMathTests(unittest.TestCase):
    def conditions(self, p, y, run, train_rate=0.2, rng=None):
        rng = rng or np.random.default_rng(0)
        train_y = np.r_[np.ones(round(4000 * train_rate)), np.zeros(4000 - round(4000 * train_rate))]  # exactly the rate
        v0_p = np.full(len(y), train_rate)  # a V0 that knows only the base rate
        return keep.head_conditions(p, y, run, train_y, v0_p, n_boot=500)

    def test_a_perfect_predictor_passes_every_computable_condition(self):
        run, y, _ = synthetic()
        head = self.conditions(y.copy(), y, run, train_rate=float(y.mean()))
        self.assertGreater(head["brier_skill_vs_base_rate"]["ci"][0], 0.0)
        self.assertGreater(head["brier_skill_vs_v0"]["ci"][0], 0.0)
        self.assertAlmostEqual(head["brier_skill_vs_base_rate"]["value"], 1.0 - 0.0, places=6)
        self.assertTrue(head["log_loss_vs_base_rate"]["pass"])
        self.assertEqual(head["ece_bins_ge_30_positives"]["value"], 0.0)
        verdict = keep.keep_conditions({"act1_boss_kill": head, "win": {"positives": 0, "rows": len(y)}}, {"runs": 300})
        self.assertTrue(verdict["all_computable_pass"])
        self.assertTrue(verdict["heads"]["act1_boss_kill"]["pass"])
        self.assertEqual(verdict["heads"]["win"]["status"], "no_positives")
        self.assertEqual(verdict["not_computed"], ["tier_b_regret_vs_v0", "battery_gain"])
        self.assertEqual(head["tier_b_regret_vs_v0"]["status"], "not_computed")
        self.assertEqual(head["battery_gain"]["status"], "not_computed")

    def test_the_base_rate_fails_its_own_skill_test_and_a_failed_head_blocks_the_verdict(self):
        run, y, _ = synthetic()
        rate = 0.2
        head = self.conditions(np.full(len(y), rate), y, run, train_rate=rate)
        self.assertAlmostEqual(head["brier_skill_vs_base_rate"]["value"], 0.0, places=9)
        self.assertFalse(head["brier_skill_vs_base_rate"]["pass"])
        self.assertFalse(head["log_loss_vs_base_rate"]["pass"])  # equal is not an improvement
        verdict = keep.keep_conditions({"act1_boss_kill": head, "win": None}, {})
        self.assertFalse(verdict["all_computable_pass"])
        self.assertFalse(keep.keep_conditions({"win": None}, {})["all_computable_pass"])  # nothing judged is not a pass

    def test_the_paired_bootstrap_resamples_runs_and_is_repeatable(self):
        run, y, rng = synthetic()
        p = np.clip(y * 0.7 + rng.random(len(y)) * 0.3, 0, 1)
        a = keep.paired_bootstrap_skill(keep.brier_terms(p, y), keep.brier_terms(np.full(len(y), y.mean()), y), run, 2000)
        b = keep.paired_bootstrap_skill(keep.brier_terms(p, y), keep.brier_terms(np.full(len(y), y.mean()), y), run, 2000)
        self.assertEqual(a, b)
        self.assertEqual((a["runs"], a["resamples"]), (300, 2000))
        self.assertLess(a["ci"][0], a["value"])
        self.assertGreater(a["ci"][1], a["value"])
        # a model worse than the base rate on some runs and better on others has an interval around its point value
        worse = keep.paired_bootstrap_skill(keep.brier_terms(1.0 - y, y), keep.brier_terms(np.full(len(y), y.mean()), y), run, 500)
        self.assertLess(worse["ci"][1], 0.0)

    def test_ece_counts_only_bins_with_at_least_30_positives(self):
        p = np.concatenate([np.full(300, 0.15), np.full(100, 0.85), np.full(10, 0.35)])
        y = np.concatenate([np.r_[np.ones(45), np.zeros(255)], np.r_[np.ones(70), np.zeros(30)], np.ones(10)])
        e = keep.expected_calibration_error(p, y)
        self.assertEqual(e["eligible_bins"], 2)  # the 10-row bin (10 positives) is left out
        self.assertAlmostEqual(e["value"], (100 / 400) * 0.15, places=9)
        self.assertEqual([b["eligible"] for b in e["bins"] if b["n"]], [True, False, True])
        none = keep.expected_calibration_error(np.full(50, 0.1), np.r_[np.ones(5), np.zeros(45)])
        self.assertIsNone(none["value"])
        head = keep.head_conditions(np.full(50, 0.1), np.r_[np.ones(5), np.zeros(45)], np.arange(50), np.zeros(10), np.full(50, 0.1), n_boot=50)
        self.assertFalse(head["ece_bins_ge_30_positives"]["pass"])  # no calibrated bin to show: not passed

    def test_v0_is_the_hand_written_arrive_healthy_score_and_a_calibrated_table(self):
        s = keep.v0_score(np.array([1.0, 0.5, 0.5]), np.array([0, 2, 9]), np.array([0, 0, 0]))
        np.testing.assert_allclose(s, [1.0, 0.7, 0.8])  # at most three potions count
        train_s = np.repeat(np.linspace(0.2, 1.2, 10), 100)
        train_y = (train_s > 0.8).astype(np.float64)  # healthier arrivals kill the boss
        p = keep.v0_probability(train_s, train_y, np.array([0.2, 1.2]))
        self.assertLess(p[0], 0.05)
        self.assertGreater(p[1], 0.9)

    def test_the_hook_reads_a_fit_s_valuedata_and_writes_plain_json(self):
        from run_train import models

        rng = np.random.default_rng(2)
        n, width = 600, models.VALUE_INPUT
        run = np.repeat(np.arange(60), 10)
        final = np.repeat(rng.choice([5, 9, 12, 18, 30], 60), 10).astype(np.int64)  # some runs past floor 17, none won
        values = SimpleNamespace(validation=np.repeat(np.arange(60) >= 48, 10), final_floor=final, run_index=run,
                                 v0=np.stack([rng.random(n), rng.integers(0, 3, n), np.zeros(n)], axis=1).astype(np.float32),
                                 inputs=lambda rows: rng.random((len(rows), width)).astype(np.float32))
        record = keep.evaluate_value_net(values, models.RunValueNet())
        json.dumps(record)
        self.assertEqual(record["heads"]["win"]["status"], "no_positives")
        self.assertIn("brier_skill_vs_base_rate", record["heads"]["act1_boss_kill"])
        self.assertIs(record["all_computable_pass"], False)  # an untrained net keeps nothing
        self.assertEqual(record["validation"]["rows"], 120)


class MacroSwitchTests(unittest.TestCase):
    def test_agent_json_gets_vrun_only_when_every_computable_condition_passes_and_the_export_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand = Path(tmp) / "gen4" / "candidate"
            cand.mkdir(parents=True)
            passed, failed = {"all_computable_pass": True}, {"all_computable_pass": False}
            self.assertEqual(gen_loop.macro_for(passed, cand), "rules")  # no run_value.npz yet
            (cand / "run_value.npz").write_bytes(b"x")
            self.assertEqual(gen_loop.macro_for(passed, cand), "vrun")
            self.assertEqual(gen_loop.macro_for(failed, cand), "rules")
            self.assertEqual(gen_loop.macro_for(None, cand), "rules")
            self.assertEqual(gen_loop.macro_for({"status": "no_split", "all_computable_pass": False}, cand), "rules")
            # load_agent reads the macro the way the next promotion's E-block does
            (cand / "agent.json").write_text(json.dumps({"generation": 4, "macro": "vrun"}), encoding="utf-8")
            agent = gen_loop.load_agent(cand)
            self.assertEqual((agent.macro, agent.generation), ("vrun", 4))
            self.assertTrue(agent.macro_npz.endswith("run_value.npz"))


if __name__ == "__main__":
    unittest.main()
