"""The trainer's completed-Q policy target (`--policy-target completed_q`) on synthetic shards."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests.python.test_gen_loop import fake_record, fake_run_arrays, gen0, gen_loop

import train_combat  # noqa: E402  (test_gen_loop puts rl/ on the path)


class CompletedQTargetTests(unittest.TestCase):
    def test_softmax_over_completed_values_on_legal_actions(self):
        nan = np.nan
        visits = np.array([[2, 1, 0, 0], [0, 0, 0, 0], [1, 1, 0, 5]], np.float32)
        legal = np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0]], np.int8)
        q = np.array([[10, 0, nan, nan], [nan, nan, nan, nan], [4, -6, nan, 50]], np.float32)
        got = train_combat.completed_q_target(visits, legal, q, temperature=10.0)
        root = (2 * 10 + 1 * 0) / 3  # the unvisited legal action 2 takes the root's visit-weighted mean value
        want = np.exp([1.0, 0.0, root / 10])
        np.testing.assert_allclose(got[0], [*(want / want.sum()), 0.0], rtol=1e-6)
        np.testing.assert_array_equal(got[1], 0.0)  # no visit: no policy signal, as the visit target leaves it
        want = np.exp([0.4, -0.6])  # the illegal action's visits and value are ignored
        np.testing.assert_allclose(got[2], [*(want / want.sum()), 0.0, 0.0], rtol=1e-6)
        self.assertEqual(got.dtype, np.float32)

    def test_temperature_orders_sharpness(self):
        visits, legal = np.ones((1, 3), np.float32), np.ones((1, 3), np.int8)
        q = np.array([[0.0, 5.0, 10.0]], np.float32)
        sharp, flat = (train_combat.completed_q_target(visits, legal, q, t)[0] for t in (1.0, 100.0))
        self.assertGreater(sharp[2], 0.99)
        self.assertLess(flat.max() - flat.min(), 0.05)


class GenLoopPassesTheTargetTests(unittest.TestCase):
    def test_train_generation_trains_on_completed_q(self):
        with tempfile.TemporaryDirectory() as d:
            root, rng = Path(d), np.random.default_rng(5)
            sink = gen_loop.ShardSink(root / "runs" / "job0", 0, config="a3-v1", base_sims=30, ascension=10, agent={},
                                      part_runs=3)
            for i in range(6):
                sink.add(fake_record(f"Q0{i:08d}"[:10], 0), fake_run_arrays(rng, 3, 4))
            sink.close()
            with mock.patch.object(train_combat, "completed_q_target", wraps=train_combat.completed_q_target) as cq:
                cand = gen_loop.train_generation(0, root / "runs", root / "agent", device="cpu", epochs=1, max_fights=20,
                                                 threads=2, vrun=False, export=False, policy_target="completed_q")
            self.assertEqual(cq.call_count, 1)
            self.assertEqual(cq.call_args.args[3], train_combat.Q_TEMPERATURE)
            t = json.loads((cand / "train.json").read_text())
            self.assertEqual(t["target"], "completed_q")
            self.assertEqual(t["combat"]["policy_target"], "completed_q")
            self.assertEqual(t["combat"]["q_temperature"], train_combat.Q_TEMPERATURE)
            self.assertTrue((cand / "combat_policy.pt").is_file())
            data = gen_loop.build_training_set(gen_loop.window_manifests(root / "runs", 0), max_fights=10)
            self.assertEqual(data["qvals"].shape, data["visits"].shape)  # the window carries the root values through

    def test_train_command_passes_the_option_and_defaults_to_visits(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(gen_loop, "train_generation", return_value=Path(d) / "gen2" / "candidate") as tg:
                self.assertEqual(gen0.main(["train", "--gen", "1", "--shards-root", d, "--agent-root", d]), 0)
                self.assertEqual(tg.call_args.kwargs["policy_target"], "visits")
                self.assertEqual(gen0.main(["train", "--gen", "1", "--shards-root", d, "--agent-root", d,
                                            "--policy-target", "completed_q"]), 0)
                self.assertEqual(tg.call_args.kwargs["policy_target"], "completed_q")


class EpochEndHookTests(unittest.TestCase):
    """`train_combat.train(epoch_end=...)`: None by default, the loop and every number unchanged without it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root, rng = Path(self._tmp.name), np.random.default_rng(5)
        sink = gen_loop.ShardSink(root / "runs" / "job0", 0, config="a3-v1", base_sims=30, ascension=10, agent={}, part_runs=3)
        for i in range(8):
            sink.add(fake_record(f"H0{i:08d}"[:10], 0), fake_run_arrays(rng, 3, 4))
        sink.close()
        self.data = gen_loop.build_training_set(gen_loop.window_manifests(root / "runs", 0), max_fights=1000)
        self.root = root

    def tearDown(self):
        self._tmp.cleanup()

    def fit(self, name, **kw):
        return gen_loop.train_combat_prior(self.data, self.root / name, device="cpu", epochs=3, d_model=16, layers=1, threads=1, seed=3, **kw)

    def test_a_no_op_hook_leaves_the_fit_unchanged_and_sees_every_epoch(self):
        import torch

        calls = []
        plain = self.fit("plain")
        hooked = self.fit("hooked", epoch_end=lambda epoch, stats: calls.append((epoch, dict(stats))))
        self.assertEqual([c[0] for c in calls], [1, 2, 3])
        self.assertEqual(sorted(calls[0][1]), ["policy_train", "policy_val", "value_train", "value_val"])
        self.assertEqual([c["policy_val"] for c in plain["curves"]], [round(s["policy_val"], 4) for _, s in calls])  # the log line's numbers
        self.assertEqual(plain["curves"], hooked["curves"])
        self.assertEqual(plain["report"], hooked["report"])
        for name in ("combat_policy.pt", "combat_net.pt"):
            a = torch.load(self.root / "plain" / name, map_location="cpu", weights_only=True)
            b = torch.load(self.root / "hooked" / name, map_location="cpu", weights_only=True)
            self.assertEqual(sorted(a["model"]), sorted(b["model"]))
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)

    def test_a_hook_that_raises_stops_the_fit_and_the_trainer_restores_its_paths(self):
        before = (train_combat.DATASET, train_combat.CHECKPOINT, train_combat.POLICY_CHECKPOINT)

        def stop(epoch, _stats):
            if epoch == 2:
                raise RuntimeError("stop")

        with self.assertRaises(RuntimeError):
            self.fit("stopped", epoch_end=stop)
        self.assertEqual((train_combat.DATASET, train_combat.CHECKPOINT, train_combat.POLICY_CHECKPOINT), before)
        log = (self.root / "stopped" / "work" / "train_combat.log").read_text(encoding="utf-8")
        self.assertIn("epoch   2", log)
        self.assertNotIn("epoch   3", log)


if __name__ == "__main__":
    unittest.main()
