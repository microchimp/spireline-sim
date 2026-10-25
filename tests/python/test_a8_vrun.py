"""A8 (D33): the promotion arms play each agent's own macro; the out-of-fold keep test; the shipped seed fit."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))

import gen_loop  # noqa: E402
from run_train import keep  # noqa: E402


def write_agent(d: Path, generation: int, **meta) -> Path:
    d.mkdir(parents=True)
    (d / "model.json").write_text(json.dumps({"policy_sha256": f"sha{generation}"}))
    (d / "combat_policy.pt").write_bytes(b"x")
    (d / "agent.json").write_text(json.dumps({"generation": generation, **meta}))
    return d


def fresh_npz(path: Path) -> Path:
    import torch  # noqa: PLC0415

    from run_train import models  # noqa: PLC0415

    torch.manual_seed(3)
    path.write_bytes(Path(models.save(path.parent / f"{path.stem}.pt", models.FightNet(), models.RunValueNet(), {"generation": 4})).read_bytes())
    return path


class PromotionArmsTests(unittest.TestCase):
    """`promote` (what `gen0.py promote` runs) builds the candidate arm from its agent.json and the base arm from generation 3's."""

    def test_the_candidate_arm_plays_lookahead_and_the_base_arm_the_rules_in_the_same_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agents = root / "agent"
            cand = write_agent(agents / "gen4" / "candidate", 4, macro="vrun")
            fresh_npz(cand / "run_value.npz")
            gen3 = write_agent(agents / "gen3", 3)  # the incumbent: no macro key
            (agents / "current.json").write_text(json.dumps({"generation": 3, "dir": str(gen3)}))
            seeds = [f"S{i:09d}" for i in range(20)]
            seed_file = root / "seeds" / "Etest-1.json"
            seed_file.parent.mkdir()
            seed_file.write_text(json.dumps({"seeds": seeds}))
            played = []

            def fake_run_arm(seeds_, arm_dir, agent, config, procs, ascension=10):
                played.append((Path(arm_dir).parts[-3], agent))
                return [{"seed": s, "floor": 5 + i % 3, "win": False, "end": "death", "decisions": 10, "wall": 1.0, "crash": False}
                        for i, s in enumerate(seeds_)]

            with mock.patch.object(gen_loop, "run_arm", fake_run_arm):
                gen_loop.promote(cand, None, seed_file, agent_root=agents, boot=200)
            self.assertEqual([name for name, _ in played], ["candidate", "gen3"])
            (_, cand_agent), (_, base_agent) = played
            self.assertEqual((cand_agent.generation, cand_agent.macro, base_agent.generation, base_agent.macro), (4, "vrun", 3, "rules"))
            self.assertEqual(type(cand_agent.macro_policy()).__name__, "LookaheadPolicy")
            self.assertEqual(type(base_agent.macro_policy()).__name__, "RulesPlayer")


def synthetic_window(informative: bool, n_runs: int = 1500, rows: int = 6, seed: int = 11):
    """A window of runs (about one in five kills the Act-1 boss) as `cross_fit_keep` reads it, with a stand-in fold fit whose
    predictions are informative about the kill, or a weak signal that is right for 53.75% of runs and wrong for the rest (a skill of
    zero in expectation)."""
    from types import SimpleNamespace

    rng = np.random.default_rng(seed)
    hashes = np.repeat(rng.random(n_runs), rows)
    kill_run = rng.random(n_runs) < 0.2
    sign_run = np.where(rng.random(n_runs) < 0.5375, 1.0, -1.0)
    y_row, sign_row = np.repeat(kill_run, rows).astype(np.float64), np.repeat(sign_run, rows)
    values = SimpleNamespace(run_hash=hashes, run_index=np.repeat(np.arange(n_runs), rows), floor=np.tile(np.arange(1, rows + 1), n_runs),
                             final_floor=np.where(y_row > 0, 30, 8).astype(np.int64), v0=np.zeros((n_runs * rows, 3), np.float32))

    def fit_fold(fights, vals, config, train_rows, held_rows, fold):
        shift = (0.35 if informative else 0.15 * sign_row[held_rows]) * (y_row[held_rows] - 0.2)
        return {"act1_boss_kill": np.clip(0.2 + shift, 0.001, 0.999), "win": np.zeros(len(held_rows)), "info": {"fold": fold}}

    return values, fit_fold


class OutOfFoldKeepTests(unittest.TestCase):
    """D33: the verdict is the out-of-fold Brier skill against the base rate; the held-out tenth is reported and never gates."""

    TENTH_FAILS = {"all_computable_pass": False, "heads": {"act1_boss_kill": {"brier_skill_vs_base_rate": {"ci": [-0.03, 0.07]}}}}
    TENTH_PASSES = {"all_computable_pass": True}

    def test_an_interval_above_zero_out_of_fold_is_kept_although_the_tenth_does_not_clear(self):
        values, fit_fold = synthetic_window(informative=True)
        record = keep.cross_fit_keep(None, values, None, self.TENTH_FAILS, fit_fold=fit_fold, n_boot=300)
        head = record["heads"]["act1_boss_kill"]
        self.assertGreater(head["brier_skill_vs_base_rate"]["ci"][0], 0.0)
        self.assertIs(record["all_computable_pass"], True)
        self.assertEqual(record["held_out_tenth"]["gating"], False)
        self.assertIs(record["held_out_tenth"]["all_computable_pass"], False)
        self.assertEqual(record["held_out_tenth"]["record"], self.TENTH_FAILS)
        self.assertIn("brier_skill_vs_v0", head)  # reported with the others
        self.assertEqual(record["heads"]["win"]["status"], "no_positives")
        json.dumps(record)
        self.assertEqual([f["fold"] for f in record["validation"]["folds"]], [0, 1, 2, 3, 4])
        self.assertAlmostEqual(sum(f["held_runs"] for f in record["validation"]["folds"]), 1500, delta=0)
        with tempfile.TemporaryDirectory() as tmp:  # the lab's macro decision reads this verdict and the export
            cand = Path(tmp)
            (cand / "run_value.npz").write_bytes(b"x")
            self.assertTrue(gen_loop.vrun_keep_passed(record, cand))

    def test_an_interval_that_includes_zero_out_of_fold_is_not_kept_although_the_tenth_clears(self):
        values, fit_fold = synthetic_window(informative=False)
        record = keep.cross_fit_keep(None, values, None, self.TENTH_PASSES, fit_fold=fit_fold, n_boot=300)
        skill = record["heads"]["act1_boss_kill"]["brier_skill_vs_base_rate"]
        self.assertLessEqual(skill["ci"][0], 0.0)
        self.assertGreaterEqual(skill["ci"][1], 0.0)
        self.assertIs(record["all_computable_pass"], False)
        self.assertIs(record["held_out_tenth"]["all_computable_pass"], True)  # reported, not gating

    def test_folds_are_the_hash_fifths_and_each_run_stops_one_in_ten(self):
        h = np.array([0.0, 0.19, 0.2, 0.55, 0.999999, 0.05, 0.1])
        self.assertEqual(keep.fold_of(h).tolist(), [0, 0, 1, 2, 4, 0, 0])
        self.assertEqual(keep.early_stop_runs(np.array([0.0, 0.005, 0.02, 0.1, 0.105, 0.5])).tolist(), [True, True, False, True, True, True])

    def test_a_fold_with_no_rows_is_skipped_and_the_rest_are_judged(self):
        values, fit_fold = synthetic_window(informative=True, n_runs=300)
        values.run_hash = values.run_hash * 0.8  # nothing in the last fold
        record = keep.cross_fit_keep(None, values, None, None, fit_fold=fit_fold, n_boot=100)
        self.assertIn("skipped", record["validation"]["folds"][4])
        self.assertIsNone(record["held_out_tenth"]["all_computable_pass"])
        self.assertIn("brier_skill_vs_base_rate", record["heads"]["act1_boss_kill"])


class CrossFitFitTests(unittest.TestCase):
    """The default fold fit, on a few real self-play runs (a smoke: the loop fits, predicts and writes plain json)."""

    def test_fit_runs_the_cross_fit_and_keeps_the_tenths_record_beside_it(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_run_train_train import A9, TINY

        from rl.run_train import selfplay, shards, train

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            records = [selfplay.play_run(f"OOF{i:05d}", selfplay.RulesPlayer(eps=0.15), 3, A9, generation=0) for i in range(8)]
            shards.write_shard(d / "r0000.jsonl.gz", records)
            seen = {}

            def judge(fights, values, config, tenth):
                self.assertEqual(len(fights.hashes), len(fights.labels))
                seen["tenth"] = tenth
                seen["record"] = keep.cross_fit_keep(fights, values, config, tenth, n_boot=50)
                return seen["record"]

            out = train.fit([d / "r0000.jsonl.gz"], None, d / "m.pt", generation=0, combat_budget=3, ascension=A9, config=TINY,
                            evaluate=keep.evaluate_value_net, cross_fit=judge)
            self.assertTrue(out.is_file())
            record = seen["record"]
            json.dumps(record)
            done = [f for f in record["validation"]["folds"] if "skipped" not in f]
            self.assertGreaterEqual(len(done), 1)
            self.assertTrue(all(f["fight_epochs"] >= 1 and f["value_epochs"] >= 1 for f in done))
            self.assertIs(record["held_out_tenth"]["record"], seen["tenth"])
            meta = train.models.load_numpy(train.models.npz_for(out)).meta
            self.assertEqual(meta["value"]["keep"]["rule"], keep.OOF_RULE)


if __name__ == "__main__":
    unittest.main()
