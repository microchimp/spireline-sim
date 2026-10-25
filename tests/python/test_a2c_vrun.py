"""A2c: V_run's DESIGN 5.3 keep conditions (rl/run_train/keep.py) on synthetic data, and the agent.json macro switch."""

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
                                 floor=np.tile(np.arange(1, 11), 60),
                                 v0=np.stack([rng.random(n), rng.integers(0, 3, n), np.zeros(n)], axis=1).astype(np.float32),
                                 inputs=lambda rows: rng.random((len(rows), width)).astype(np.float32))
        record = keep.evaluate_value_net(values, models.RunValueNet())
        json.dumps(record)
        self.assertEqual(record["heads"]["win"]["status"], "no_positives")
        self.assertIn("brier_skill_vs_base_rate", record["heads"]["act1_boss_kill"])
        self.assertIs(record["all_computable_pass"], False)  # an untrained net keeps nothing
        self.assertEqual(record["validation"]["rows"], 120)


class MacroSwitchTests(unittest.TestCase):
    """D33: `vrun` needs a passed keep record and the export; the switch is on without any variable, and SPIRELINE_MACRO_VRUN=rules opts out."""

    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(gen_loop.MACRO_VRUN_FLAG, None)

    def test_agent_json_gets_vrun_when_every_computable_condition_passes_and_the_export_exists_with_no_variable_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand = Path(tmp) / "gen4" / "candidate"
            cand.mkdir(parents=True)
            passed, failed = {"all_computable_pass": True}, {"all_computable_pass": False}
            self.assertEqual(gen_loop.macro_for(passed, cand), "rules")  # no run_value.npz yet
            (cand / "run_value.npz").write_bytes(b"x")
            self.assertEqual(gen_loop.macro_for(passed, cand), "vrun")  # passed, exported, no variable
            self.assertEqual(gen_loop.macro_for(failed, cand), "rules")
            self.assertEqual(gen_loop.macro_for(None, cand), "rules")
            self.assertEqual(gen_loop.macro_for({"status": "no_split", "all_computable_pass": False}, cand), "rules")
            self.assertEqual(gen_loop.macro_for(passed, cand, enabled=False), "rules")  # an explicit flag beats the variable
            # load_agent reads the macro the way the next promotion's E-block does
            (cand / "agent.json").write_text(json.dumps({"generation": 4, "macro": "vrun"}), encoding="utf-8")
            agent = gen_loop.load_agent(cand)
            self.assertEqual((agent.macro, agent.generation), ("vrun", 4))
            self.assertTrue(agent.macro_npz.endswith("run_value.npz"))

    def test_the_variable_is_an_opt_out_that_reads_rules_off_0_false_no_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand = Path(tmp)
            (cand / "run_value.npz").write_bytes(b"x")
            passed = {"all_computable_pass": True}
            for value in ("rules", " RULES ", "off", "0", "false", "no"):
                with self.subTest(value=value), mock.patch.dict(os.environ, {gen_loop.MACRO_VRUN_FLAG: value}):
                    self.assertEqual(gen_loop.macro_for(passed, cand), "rules")
            for value in ("", "1", "on", "true", "yes", "vrun", "2"):  # the old on-values and anything else leave the default
                with self.subTest(value=value), mock.patch.dict(os.environ, {gen_loop.MACRO_VRUN_FLAG: value}):
                    self.assertEqual(gen_loop.macro_for(passed, cand), "vrun")


if __name__ == "__main__":
    unittest.main()


# ------------------------------------------------------------------ amend: the real-game arm plays the scored agent's macro policy

import os  # noqa: E402
import shutil  # noqa: E402
from unittest import mock  # noqa: E402

sys.path.insert(0, str(ROOT / "referee"))
import lockstep  # noqa: E402

try:
    import leakprobes  # noqa: E402

    leakprobes.native_env()
    from sts2_gym import native  # noqa: E402,F401

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001 -- no native library: the lockstep tests are skipped
    HAVE_NATIVE = False


def vrun_agent_dir(root: Path) -> Path:
    """A generation directory whose agent.json says macro vrun, with a loadable run_value.npz (fresh nets)."""
    import torch  # noqa: PLC0415

    from run_train import models  # noqa: PLC0415

    d = root / "gen4" / "candidate"
    d.mkdir(parents=True)
    torch.manual_seed(3)
    npz = models.save(root / "learner.pt", models.FightNet(), models.RunValueNet(), {"generation": 4})
    (d / "run_value.npz").write_bytes(Path(npz).read_bytes())
    (d / "agent.json").write_text(json.dumps({"generation": 4, "macro": "vrun"}), encoding="utf-8")
    return d


def walk(e, steps: int = 8):
    """The lockstep's own macro picks from the run's start to its first fight (EmuLock.choose / the fast-forward's call)."""
    picks = []
    for _ in range(steps):
        d = e._decision()
        if d is None:
            break
        i = e.rules.pick(e.s.view(), d, e.rng).index
        picks.append((d.type, i))
        e.s.apply(d.options[i])
    return picks, e.rng.bit_generator.state


@unittest.skipUnless(HAVE_NATIVE, "needs the native emulator library")
class LockstepMacroPolicyTests(unittest.TestCase):
    """score-real's lockstep (referee/lockstep.py EmuLock) plays the macro policy of the agent SPIRELINE_GEN_DIR names, as
    the emulator arm (gen0.play_emu_run -> Agent.macro_policy) does; generation 0 when the variable is unset."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(gen_loop.ENV_GEN_DIR, None)

    def lock(self):
        e = lockstep.EmuLock("QS2GYXRKWN", sims=2, agent_seed=[11])
        self.addCleanup(e.close)
        return e

    def test_without_a_generation_the_lockstep_plays_the_rules_exactly_as_before(self):
        from run_train import selfplay  # noqa: PLC0415

        new = self.lock()
        old = self.lock()
        old.rules = selfplay.RulesPlayer()  # the player the lockstep hard-coded before the amend
        self.assertIsInstance(new.rules, selfplay.RulesPlayer)
        self.assertEqual(new.rules.eps, 0.0)
        a, b = walk(new), walk(old)
        self.assertGreater(len(a[0]), 0)
        self.assertEqual(a, b)  # same picks, same exploration stream state

    def test_a_vrun_agent_named_by_spireline_gen_dir_makes_the_lockstep_play_its_lookahead(self):
        from run_train.policy import LookaheadPolicy  # noqa: PLC0415

        os.environ[gen_loop.ENV_GEN_DIR] = str(vrun_agent_dir(self.tmp))
        e = self.lock()
        self.assertIsInstance(e.rules, LookaheadPolicy)
        self.assertEqual((e.rules.tau, e.rules.eps), (0.0, 0.0))
        before = e.rng.bit_generator.state
        picks, after = walk(e)
        self.assertGreater(len(picks), 0)
        self.assertTrue(any(p[0] == "map" for p in picks))
        self.assertEqual(after, before)  # tau = eps = 0 is argmax: the lookahead draws nothing from the exploration stream


class KeepAmendTests(unittest.TestCase):
    """The review's changes to run_train/keep.py."""

    def test_ece_eligibility_counts_distinct_positive_runs_not_rows(self):
        # DESIGN 5.3: a bin needs >= 30 wins / boss kills; the rows of one run are correlated, so 40 positive rows of 2 runs
        # are 2 outcomes, not 40
        p, y = np.full(100, 0.45), np.r_[np.ones(40), np.zeros(60)]
        two = np.r_[np.repeat([0, 1], 20), np.arange(2, 62)]
        head = keep.head_conditions(p, y, two, np.r_[np.ones(4), np.zeros(6)], p, n_boot=50)
        self.assertEqual(head["ece_bins_ge_30_positives"]["eligible_bins"], 0)
        self.assertFalse(head["ece_bins_ge_30_positives"]["pass"])
        self.assertEqual((head["positives"], head["positive_runs"]), (40, 2))
        e = keep.expected_calibration_error(p, y, run=two)
        self.assertEqual((e["eligible_bins"], e["bins"][4]["positives"], e["bins"][4]["positive_runs"]), (0, 40, 2))
        thirty = np.r_[np.arange(30), np.arange(10), np.arange(30, 90)]  # the same 40 positive rows over 30 runs
        e = keep.expected_calibration_error(p, y, run=thirty)
        self.assertEqual((e["eligible_bins"], e["bins"][4]["positive_runs"]), (1, 30))
        self.assertAlmostEqual(e["value"], 0.05, places=9)

    def test_the_act1_head_is_judged_on_rows_before_the_boss_room_only(self):
        # Every decision of a boss-killing run from floor 17 on (the boss's rewards, the next map, Act 2) is taken after the
        # kill, so its Act-1 label is fixed by the state; those rows would inflate BSS, log-loss and ECE. The win head keeps all.
        from run_train import models

        rng = np.random.default_rng(4)
        n, width = 600, models.VALUE_INPUT
        run = np.repeat(np.arange(60), 10)
        final = np.repeat(np.tile([5, 9, 30], 20), 10).astype(np.int64)  # every third run kills the boss
        floor = np.where(final > 17, 12, 1) + np.tile(np.arange(10), 60)  # a kill run's rows on floors 12-21, others 1-10
        values = SimpleNamespace(validation=np.repeat(np.arange(60) >= 48, 10), final_floor=final, run_index=run, floor=floor,
                                 v0=np.stack([rng.random(n), rng.integers(0, 3, n), np.zeros(n)], axis=1).astype(np.float32),
                                 inputs=lambda rows: rng.random((len(rows), width)).astype(np.float32))
        record = keep.evaluate_value_net(values, models.RunValueNet())
        head = record["heads"]["act1_boss_kill"]
        # validation: 8 runs x 10 rows + 4 kill runs x 5 rows on floors 12-16; training base rate 16 x 5 / (32 x 10 + 16 x 5)
        self.assertEqual((head["rows"], head["positives"], head["positive_runs"]), (100, 20, 4))
        self.assertAlmostEqual(head["base_rate"], 0.2, places=9)
        self.assertEqual(record["validation"]["act1_head_last_floor"], 16)
        self.assertEqual(record["heads"]["win"]["rows"], 120)

    def test_the_record_says_its_split_also_stopped_the_fit_and_a_vrun_switch_mixes_prior_and_macro(self):
        from run_train import models

        rng = np.random.default_rng(5)
        n = 200
        values = SimpleNamespace(validation=np.repeat(np.arange(20) >= 15, 10), final_floor=np.full(n, 5, np.int64),
                                 run_index=np.repeat(np.arange(20), 10), floor=np.tile(np.arange(1, 11), 20),
                                 v0=np.zeros((n, 3), np.float32),
                                 inputs=lambda rows: rng.random((len(rows), models.VALUE_INPUT)).astype(np.float32))
        v = keep.evaluate_value_net(values, models.RunValueNet())["validation"]
        self.assertIs(v["early_stopping_split"], True)
        self.assertIn("optimistic", v["early_stopping_note"])
        self.assertIn("together", gen_loop.vrun_keep_note("vrun"))
        self.assertIn("rules", gen_loop.vrun_keep_note("rules"))

    def test_a_v0_plus_floor_baseline_is_reported_and_never_gates(self):
        from run_train import models

        rng = np.random.default_rng(6)
        n = 600
        final = np.repeat(np.tile([5, 9, 30], 20), 10).astype(np.int64)
        values = SimpleNamespace(validation=np.repeat(np.arange(60) >= 48, 10), final_floor=final, run_index=np.repeat(np.arange(60), 10),
                                 floor=np.where(final > 17, 12, 1) + np.tile(np.arange(10), 60),
                                 v0=np.stack([rng.random(n), rng.integers(0, 3, n), np.zeros(n)], axis=1).astype(np.float32),
                                 inputs=lambda rows: rng.random((len(rows), models.VALUE_INPUT)).astype(np.float32))
        head = keep.evaluate_value_net(values, models.RunValueNet())["heads"]["act1_boss_kill"]
        info = head["brier_skill_vs_v0_floor_informational"]
        self.assertIs(info["gated"], False)
        self.assertEqual(len(info["ci"]), 2)
        gated = {k: {"pass": True} for k in ("brier_skill_vs_base_rate", "brier_skill_vs_v0", "log_loss_vs_base_rate",
                                              "ece_bins_ge_30_positives")}
        self.assertTrue(keep.passes({**gated, "brier_skill_vs_v0_floor_informational": {"pass": False, "gated": False}}))
