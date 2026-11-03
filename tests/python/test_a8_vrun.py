"""A8 (D33): the promotion arms play each agent's own macro; the out-of-fold keep test; the shipped seed fit."""

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


def synthetic_window(informative: bool, n_runs: int = 1500, rows: int = 6, seed: int = 11, win_runs: int = 0):
    """A window of runs (about one in five kills the Act-1 boss) as `cross_fit_keep` reads it, with a stand-in fold fit whose
    predictions are informative about the kill, or a weak signal that is right for 53.75% of runs and wrong for the rest (a skill of
    zero in expectation)."""
    from types import SimpleNamespace

    rng = np.random.default_rng(seed)
    hashes = np.repeat(rng.random(n_runs), rows)
    kill_run = rng.random(n_runs) < 0.2
    sign_run = np.where(rng.random(n_runs) < 0.5375, 1.0, -1.0)
    y_row, sign_row = np.repeat(kill_run, rows).astype(np.float64), np.repeat(sign_run, rows)
    win_idx = np.flatnonzero(kill_run)[:win_runs]  # runs that won: `win_runs` positive runs for the win head
    win_row = np.repeat(np.isin(np.arange(n_runs), win_idx), rows)
    win_sign = np.repeat(np.where(np.arange(n_runs) % 2 == 0, 1.0, -1.0), rows)  # an uninformative win head: the base rate plus noise
    values = SimpleNamespace(run_hash=hashes, run_index=np.repeat(np.arange(n_runs), rows), floor=np.tile(np.arange(1, rows + 1), n_runs),
                             final_floor=np.where(win_row, 999, np.where(y_row > 0, 30, 8)).astype(np.int64), v0=np.zeros((n_runs * rows, 3), np.float32))

    def fit_fold(fights, vals, config, train_rows, held_rows, fold):
        shift = (0.35 if informative else 0.15 * sign_row[held_rows]) * (y_row[held_rows] - 0.2)
        return {"act1_boss_kill": np.clip(0.2 + shift, 0.001, 0.999), "win": np.clip(win_runs / n_runs + 0.01 * win_sign[held_rows], 0.0, 1.0), "info": {"fold": fold}}

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
        self.assertEqual((record["heads"]["win"]["gating"], record["heads"]["win"]["reason"]), (False, "no_positives"))
        self.assertIs(head["gating"], True)
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

    def test_a_head_with_fewer_than_30_positive_runs_is_reported_and_does_not_gate(self):
        values, fit_fold = synthetic_window(informative=True, win_runs=5)
        record = keep.cross_fit_keep(None, values, None, self.TENTH_FAILS, fit_fold=fit_fold, n_boot=300)
        win = record["heads"]["win"]
        self.assertEqual(win["positive_runs"], 5)
        self.assertLessEqual(win["brier_skill_vs_base_rate"]["ci"][0], 0.0)  # the interval crosses (or lies below) zero
        self.assertIs(win["pass"], False)
        self.assertEqual((win["gating"], win["reason"]), (False, "positive_runs_below_30"))
        self.assertIs(record["heads"]["act1_boss_kill"]["pass"], True)
        self.assertIs(record["all_computable_pass"], True)  # kept: the boss-kill head passes, the win head is only reported
        self.assertEqual(record["min_positive_runs"], 30)
        with tempfile.TemporaryDirectory() as tmp:
            cand = Path(tmp)
            (cand / "run_value.npz").write_bytes(b"x")
            self.assertTrue(gen_loop.vrun_keep_passed(record, cand))

    def test_a_head_with_30_positive_runs_gates_again(self):
        values, fit_fold = synthetic_window(informative=True, win_runs=40)
        record = keep.cross_fit_keep(None, values, None, self.TENTH_PASSES, fit_fold=fit_fold, n_boot=300)
        win = record["heads"]["win"]
        self.assertEqual((win["positive_runs"], win["gating"], win["pass"]), (40, True, False))
        self.assertNotIn("reason", win)
        self.assertIs(record["all_computable_pass"], False)  # the uninformative win head now blocks the keep


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


class SeedFitTests(unittest.TestCase):
    """D33: the first vrun candidate ships with the A/B's fit; a later fit replaces it only when it passes the keep test."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import test_gen_loop

        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        rng = np.random.default_rng(3)
        for g in (0, 1):
            sink = gen_loop.ShardSink(root / "runs" / f"job{g}", g, config="a3-v1", base_sims=30, ascension=10, agent={}, part_runs=3)
            for i in range(6):
                sink.add(test_gen_loop.fake_record(f"S{g}{i:08d}"[:10], g), test_gen_loop.fake_run_arrays(rng, 3, 5))
            sink.close()
        cls.root = root
        cls.seed = root / "seedfit" / "run_value.npz"
        cls.seed.parent.mkdir()
        cls.seed.write_bytes(b"SEEDFIT")
        (cls.seed.parent / "run_value.keep.json").write_text(json.dumps({"recipe_out_of_fold": {"value": 0.044}}))
        cls.count = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def train(self, new_passes: bool, seed, env=None, ranking_passes: bool = True):
        """train_generation with the V fit stubbed: it writes run_value.npz (b"NEWFIT") and a keep record that passes or fails."""
        type(self).count += 1
        head = {"positives": 40, "positive_runs": 40, "rows": 900, "pass": new_passes,
                "brier_skill_vs_base_rate": {"value": 0.04 if new_passes else -0.01, "ci": [0.01, 0.07] if new_passes else [-0.03, 0.01],
                                             "runs": 900, "pass": new_passes},
                "brier_skill_vs_v0": {"value": 0.03, "ci": [0.0, 0.06], "pass": True},
                "log_loss_vs_base_rate": {"pass": True}, "ece_bins_ge_30_positives": {"pass": True}}

        def fake_fit(paths, out, **kw):
            Path(out, "run_value.npz").write_bytes(b"NEWFIT")
            Path(out, "run_value.pt").write_bytes(b"PT")
            return {"wall_min": 0.0, "cpu_min": 0.0, "fight": None, "config": None, "value": {
                "runs": 12, "duplicates_dropped": 0, "keep": {"all_computable_pass": new_passes, "heads": {"act1_boss_kill": head}}}}

        with mock.patch.dict(os.environ, env or {}), mock.patch.object(gen_loop, "train_run_value", fake_fit):
            os.environ.pop(gen_loop.MACRO_VRUN_FLAG, None) if not env else None
            return gen_loop.train_generation(1, self.root / "runs", self.root / "agent", device="cpu", epochs=1, max_fights=20,
                                             threads=2, vrun=True, export=False, retrain=self.count, vrun_seed=seed,
                                             ranking_check=lambda *a, **k: {"status": "ok", "pass": ranking_passes, "set": {"digest": "d"},
                                                                            "checks": [{"name": "heal_rate", "pass": ranking_passes}]})

    def test_a_failing_new_fit_keeps_the_seed_fit_and_a_passing_one_replaces_it(self):
        failing = self.train(False, self.seed)
        self.assertEqual((failing / "run_value.npz").read_bytes(), b"SEEDFIT")
        self.assertEqual((failing / "rejected-fit" / "run_value.npz").read_bytes(), b"NEWFIT")
        t = json.loads((failing / "train.json").read_text())
        self.assertEqual((t["vrun_fit"]["source"], t["vrun_fit"]["kept"], t["vrun_keep"]["macro_switch"]["carried"]), ("seed", True, "seed"))
        self.assertEqual(t["vrun_fit"]["sha256"], gen_loop.file_sha256(self.seed))
        self.assertEqual(t["vrun_fit"]["seed"]["record"], {"recipe_out_of_fold": {"value": 0.044}})
        self.assertEqual(t["vrun_fit"]["new_fit_oof"]["brier_skill_vs_base_rate"]["ci"], [-0.03, 0.01])
        meta = json.loads((failing / "agent.json").read_text())
        self.assertEqual((meta["macro"], meta["vrun_fit"]["source"]), ("vrun", "seed"))
        self.assertEqual(gen_loop.load_agent(failing).macro, "vrun")
        passing = self.train(True, self.seed)
        self.assertEqual((passing / "run_value.npz").read_bytes(), b"NEWFIT")
        self.assertFalse((passing / "rejected-fit").exists())
        t = json.loads((passing / "train.json").read_text())
        self.assertEqual((t["vrun_fit"]["source"], t["vrun_fit"]["sha256"]), ("new", gen_loop.file_sha256(passing / "run_value.npz")))
        self.assertEqual(t["vrun_fit"]["new_fit_oof"]["brier_skill_vs_base_rate"]["ci"], [0.01, 0.07])
        self.assertEqual(json.loads((passing / "agent.json").read_text())["macro"], "vrun")

    def test_a_new_fit_that_passes_the_keep_test_but_not_the_ranking_check_keeps_the_seed_fit(self):
        ranked = self.train(True, self.seed, ranking_passes=False)  # D47, D49
        self.assertEqual((ranked / "run_value.npz").read_bytes(), b"SEEDFIT")
        self.assertEqual((ranked / "rejected-fit" / "run_value.npz").read_bytes(), b"NEWFIT")
        t = json.loads((ranked / "train.json").read_text())
        self.assertEqual((t["vrun_fit"]["source"], t["vrun_fit"]["new_fit_passed_keep"]), ("seed", False))
        self.assertFalse(t["vrun_keep"]["ranking"]["pass"])
        self.assertTrue(t["vrun_keep"]["all_computable_pass"])
        self.assertEqual(t["vrun_keep"]["macro_switch"]["passed"], False)

    def test_without_a_seed_a_failing_fit_leaves_the_rules_macro_and_the_opt_out_still_wins(self):
        none = self.train(False, None)
        self.assertEqual(json.loads((none / "agent.json").read_text())["macro"], "rules")
        self.assertEqual(json.loads((none / "train.json").read_text())["vrun_fit"]["source"], "none")
        opted_out = self.train(True, self.seed, env={gen_loop.MACRO_VRUN_FLAG: "rules"})
        self.assertEqual(json.loads((opted_out / "agent.json").read_text())["macro"], "rules")
        self.assertEqual(json.loads((opted_out / "train.json").read_text())["vrun_keep"]["macro_switch"]["enabled"], False)


class TargetMarkerTests(unittest.TestCase):
    """D33 addendum (b): `ready-<gen>-g<target>` trains from generation <gen>'s shards and writes generation <target>."""

    def test_the_target_marker_parses_and_sorts_after_the_plain_and_retrain_forms_of_its_generation(self):
        from tools.lab import jobs

        with tempfile.TemporaryDirectory() as d:
            train = Path(d) / "train"
            train.mkdir()
            for n in ("ready-12", "ready-3-g5", "ready-3-r2", "ready-3", "ready-3-g4", "done-3-g5", "ready-3-g", "ready-3-g5-x", "ready-3-r1-g5"):
                (train / n).write_text("x")
            want = ["ready-3", "ready-3-r2", "ready-3-g4", "ready-3-g5", "ready-12"]
            self.assertEqual([p.name for _, p in gen_loop.ready_markers(train)], want)
            self.assertEqual([p.name for p in jobs.train_markers(Path(d))][:5], want)  # names that fit no form sort last, as before
            self.assertEqual(gen_loop.marker_parts(train / "ready-3-g5"), (3, 0))
            self.assertEqual(gen_loop.marker_parts(train / "ready-3-r2"), (3, 2))
            self.assertEqual((gen_loop.marker_target(train / "ready-3-g5"), gen_loop.marker_target(train / "ready-3-r2"),
                              gen_loop.marker_target(train / "done-3-g5")), (5, None, None))
            self.assertIsNone(gen_loop.marker_parts(train / "done-3-g5"))
            self.assertIsNone(gen_loop._MARKER.fullmatch("done-3-g5"))  # mark_ready's budget count does not see it

    def test_gen0_train_takes_the_target_from_the_marker_and_refuses_a_contradicting_flag(self):
        import gen0

        with tempfile.TemporaryDirectory() as d:
            train = Path(d) / "train"
            train.mkdir()
            (train / "ready-3-g5").write_text("x")
            seen = {}

            def fake_train(gen, shards_root, agent_root, **kw):
                seen.update(gen=gen, **kw)
                return Path(agent_root) / f"gen{kw['candidate_generation'] or gen + 1}" / "candidate"

            with mock.patch.object(gen_loop, "train_generation", fake_train):
                args = ["train", "--marker-dir", str(train), "--shards-root", d, "--agent-root", d]
                self.assertEqual(gen0.main(args), 0)
                self.assertEqual((seen["gen"], seen["candidate_generation"], seen["retrain"]), (3, 5, 0))
                self.assertEqual(gen0.main([*args, "--candidate-generation", "5"]), 0)
                self.assertEqual(gen0.main([*args, "--candidate-generation", "6"]), 2)
                self.assertEqual(gen0.main(["train", "--gen", "3", "--candidate-generation", "7", "--shards-root", d, "--agent-root", d]), 0)
                self.assertEqual((seen["gen"], seen["candidate_generation"]), (3, 7))
                self.assertEqual(gen0.main(["train", "--gen", "3", "--shards-root", d, "--agent-root", d]), 0)
                self.assertIsNone(seen["candidate_generation"])  # the default stays source + 1


class NewGenerationTests(unittest.TestCase):
    """A candidate trained from generation 3's shards and written as generation 5 (the first vrun candidate, D33 addendum (b))."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import test_gen_loop

        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.root = Path(cls.tmp.name)
        rng = np.random.default_rng(5)
        for g in (2, 3):
            sink = gen_loop.ShardSink(root / "runs" / f"job{g}", g, config="a3-v1", base_sims=30, ascension=10, agent={}, part_runs=3)
            for i in range(6):
                sink.add(test_gen_loop.fake_record(f"T{g}{i:08d}"[:10], g), test_gen_loop.fake_run_arrays(rng, 3, 5))
            sink.close()
        cls.seed = root / "seedfit" / "run_value.npz"
        cls.seed.parent.mkdir()
        cls.seed.write_bytes(b"SEEDFIT")
        cls.count = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def train(self, target, new_passes=False):
        type(self).count += 1
        head = {"positives": 40, "positive_runs": 40, "rows": 900, "pass": new_passes,
                "brier_skill_vs_base_rate": {"value": 0.04 if new_passes else -0.01, "ci": [0.01, 0.07] if new_passes else [-0.03, 0.01],
                                             "runs": 900, "pass": new_passes},
                "brier_skill_vs_v0": {"value": 0.03, "ci": [0.0, 0.06], "pass": True},
                "log_loss_vs_base_rate": {"pass": True}, "ece_bins_ge_30_positives": {"pass": True}}

        def fake_fit(paths, out, **kw):
            Path(out, "run_value.npz").write_bytes(b"NEWFIT")
            Path(out, "run_value.pt").write_bytes(b"PT")
            return {"wall_min": 0.0, "cpu_min": 0.0, "fight": None, "config": None, "value": {
                "runs": 12, "duplicates_dropped": 0, "keep": {"all_computable_pass": new_passes, "heads": {"act1_boss_kill": head}}}}

        # a fresh agent folder per call: gen<target>/candidate is written for the first time each time
        agents = self.root / f"agent{self.count}"
        with mock.patch.dict(os.environ), mock.patch.object(gen_loop, "train_run_value", fake_fit):
            os.environ.pop(gen_loop.MACRO_VRUN_FLAG, None)
            return gen_loop.train_generation(3, self.root / "runs", agents, device="cpu", epochs=1, max_fights=20, threads=2, vrun=True,
                                             export=False, vrun_seed=self.seed, candidate_generation=target)

    LINEAGE = {"from_generation": 3, "skips": [4], "decision": "D33 addendum (b)",
               "difference": "macro vrun with the A/B's V_run fit (D33, 2026-10-25); generation 4's two attempts were rules-macro candidates"}

    def test_a_candidate_from_generation_3_shards_is_written_as_generation_5_with_its_lineage(self):
        cand = self.train(5)
        self.assertEqual((cand.parent.name, cand.name), ("gen5", "candidate"))
        t = json.loads((cand / "train.json").read_text())
        a = json.loads((cand / "agent.json").read_text())
        self.assertEqual((t["generation_trained_from"], t["candidate_generation"]), (3, 5))
        self.assertEqual(t["lineage"], self.LINEAGE)
        self.assertEqual((a["generation"], a["trained_from"], a["macro"], a["lineage"]), (5, 3, "vrun", self.LINEAGE))
        self.assertEqual(gen_loop.load_agent(cand).generation, 5)  # what promote writes as cand_gen
        self.assertEqual(json.loads((cand / "model.json").read_text()).get("generation", 5), 5)

    def test_the_default_candidate_is_gen_plus_one_and_has_no_lineage_and_a_lower_target_is_refused(self):
        cand = self.train(None)
        self.assertEqual((cand.parent.name, cand.name), ("gen4", "candidate"))
        self.assertNotIn("lineage", json.loads((cand / "train.json").read_text()))
        self.assertNotIn("lineage", json.loads((cand / "agent.json").read_text()))
        self.assertEqual(json.loads((cand / "agent.json").read_text())["generation"], 4)
        with self.assertRaises(ValueError):
            self.train(3)
        self.assertIsNone(gen_loop.lineage_note(3, 4, "vrun", "seed"))
        self.assertEqual(gen_loop.lineage_note(3, 6, "rules", "none")["skips"], [4, 5])

    def test_the_new_generation_starts_its_attempt_count_at_zero_and_promote_writes_its_number(self):
        from tools.lab import promotion_guard as pg

        rows = [{"cand_gen": "4", "cand_sha": "a", "decision": "rejected"}, {"cand_gen": "4", "cand_sha": "b", "decision": "rejected"}]
        self.assertEqual(gen_loop.attempts_before(rows, 4, "c"), 2)
        self.assertFalse(pg.retry_cap(gen_loop.attempts_before(rows, 4, "c"))["allowed"])  # generation 4's refusal stands
        self.assertEqual(gen_loop.attempts_before(rows, 5, "c"), 0)
        self.assertTrue(pg.retry_cap(gen_loop.attempts_before(rows, 5, "c"))["allowed"])
        with tempfile.TemporaryDirectory() as tmp:
            agents = Path(tmp) / "agent"
            agents.mkdir()
            for r in rows:
                gen_loop.append_promotion(agents / "promotions.tsv", {**r, "seed_block": f"E4-{r['cand_sha']}"})
            cand = write_agent(agents / "gen5" / "candidate", 5, macro="vrun", trained_from=3)
            fresh_npz(cand / "run_value.npz")
            gen3 = write_agent(agents / "gen3", 3)
            (agents / "current.json").write_text(json.dumps({"generation": 3, "dir": str(gen3)}))
            seeds = [f"S{i:09d}" for i in range(20)]
            seed_file = Path(tmp) / "seeds" / "E5-test.json"
            seed_file.parent.mkdir()
            seed_file.write_text(json.dumps({"seeds": seeds}))

            def fake_run_arm(seeds_, arm_dir, agent, config, procs, ascension=10):
                return [{"seed": s, "floor": 5 + i % 3, "win": False, "end": "death", "decisions": 10, "wall": 1.0, "crash": False}
                        for i, s in enumerate(seeds_)]

            with mock.patch.object(gen_loop, "run_arm", fake_run_arm):
                gen_loop.promote(cand, None, seed_file, agent_root=agents, boot=200, review_guard=True)  # generation 5: not refused
            last = gen_loop.read_promotions(agents / "promotions.tsv")[-1]
            self.assertEqual((last["cand_gen"], last["base_gen"], last["seed_block"]), ("5", "3", "E5-test"))


if __name__ == "__main__":
    unittest.main()
