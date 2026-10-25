"""docs/rl/VRUN_SWEEP.md: the CPU path unchanged bit for bit, the device path, `rows_per_run`, the width of V (CPU, synthetic rows)."""

import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "rl", ROOT / "src", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))

import vrun_golden  # noqa: E402
from test_run_train_train import A9, BufferTests  # noqa: E402

from rl.run_train import models, shards, train  # noqa: E402
from rl.run_train.engine import CombatBudget  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "vrun_golden.npz"


class GoldenTests(unittest.TestCase):
    """`fixtures/vrun_golden.npz` was written by the code before the GPU path existed; the defaults must reproduce it exactly."""

    def test_the_default_config_refits_the_golden_checkpoint_bit_for_bit(self):
        with np.load(GOLDEN) as golden:
            expected = {key: golden[key] for key in golden.files}
        refit = vrun_golden.golden_fit()
        self.assertEqual(sorted(refit), sorted(expected))
        for key, value in expected.items():
            with self.subTest(key=key):
                if "history" in key:
                    np.testing.assert_array_equal(refit[key], value)
                else:
                    self.assertTrue(torch.equal(torch.from_numpy(refit[key]), torch.from_numpy(value)), key)

    def test_the_new_settings_default_to_the_labs_and_the_checkpoint_records_them(self):
        config = train.TrainConfig()
        self.assertEqual((config.device, config.hidden, config.dropout, config.rows_per_run), ("cpu", 256, 0.1, 0))
        with tempfile.TemporaryDirectory() as tmp:
            records = [BufferTests.record(hp, 0) for hp in (20, 21, 22, 23)]
            paths = []
            for i, record in enumerate(records):
                path = Path(tmp) / f"r{i:04d}.jsonl.gz"
                shards.write_shard(path, [record])
                paths.append(path)
            out = train.fit(
                paths,
                None,
                Path(tmp) / "w.pt",
                generation=0,
                combat_budget=CombatBudget(120, 500),
                ascension=A9,
                config=train.TrainConfig(epochs=1, batch=8, hidden=32, dropout=0.3),
            )
            meta = torch.load(out, map_location="cpu", weights_only=True)
            self.assertEqual(meta["meta"]["config"], asdict(train.TrainConfig(epochs=1, batch=8, hidden=32, dropout=0.3)))
            self.assertEqual(tuple(meta["value"]["hidden2.weight"].shape), (32, 32))
            loaded = models.load_numpy(models.npz_for(out))
            self.assertEqual(loaded.value.w1.shape[0], 32)


class RowsPerRunTests(unittest.TestCase):
    LENGTHS = (1, 3, 7, 12, 2)

    def groups(self):
        return np.repeat(np.arange(len(self.LENGTHS)), self.LENGTHS)

    def test_every_run_gives_k_rows_or_all_of_them_and_the_sample_changes_each_epoch(self):
        groups = self.groups()
        rows = np.arange(len(groups))
        generator = torch.Generator().manual_seed(3)
        seen = set()
        for _ in range(6):
            order = train._epoch_order(rows, groups, 4, generator)
            self.assertEqual(len(set(order.tolist())), len(order))  # without replacement
            counts = np.bincount(groups[order], minlength=len(self.LENGTHS))
            self.assertEqual(counts.tolist(), [min(4, length) for length in self.LENGTHS])
            seen.add(tuple(sorted(order.tolist())))
        self.assertGreater(len(seen), 1)

    def test_only_the_given_rows_are_sampled_and_zero_is_the_plain_shuffle(self):
        groups = self.groups()
        train_rows = np.flatnonzero(groups != 3)  # the long run is held out
        order = train._epoch_order(train_rows, groups, 2, torch.Generator().manual_seed(1))
        self.assertTrue(set(order.tolist()) <= set(train_rows.tolist()))
        self.assertEqual(len(order), 1 + 2 + 2 + 2)
        plain = train._epoch_order(train_rows, groups, 0, torch.Generator().manual_seed(1))
        np.testing.assert_array_equal(
            plain,
            train_rows[torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(1)).numpy()],
        )

    def test_the_epoch_loss_is_a_mean_over_the_sampled_rows(self):
        groups = self.groups()
        visited: list[np.ndarray] = []
        parameter = torch.nn.Parameter(torch.zeros(1))
        net = torch.nn.Linear(1, 1)

        def loss_of(rows):  # a batch's mean of the row numbers: its scale does not depend on the batch size
            visited.append(np.asarray(rows).copy())
            return parameter * 0.0 + float(np.mean(rows))

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = parameter

        config = train.TrainConfig(epochs=2, batch=5, rows_per_run=3, lr=0.0)
        stats = train._optimise(
            Net(),
            loss_of,
            np.zeros(len(groups), bool),
            config,
            torch.Generator().manual_seed(0),
            groups=groups,
        )
        del net
        per_epoch = [np.concatenate(visited[:3]), np.concatenate(visited[3:])]
        self.assertEqual(sum(len(part) for part in visited), 2 * sum(min(3, n) for n in self.LENGTHS))
        for epoch, rows in zip(stats["history"], per_epoch, strict=True):
            self.assertEqual(len(rows), sum(min(3, n) for n in self.LENGTHS))
            self.assertAlmostEqual(epoch["train"], float(rows.mean()), places=6)

    def test_v_trains_with_it_and_validates_on_every_held_out_row(self):
        values = vrun_golden.synthetic_value_data()
        values.fill_fight_block(models.NumpyFightNet(models.numpy_params(models.FightNet())))
        torch.manual_seed(2)
        config = train.TrainConfig(epochs=2, batch=16, patience=2, rows_per_run=2)
        stats = train.train_value_net(models.RunValueNet(), values, config, torch.Generator().manual_seed(2))
        self.assertEqual(stats["validation_rows"], int(values.validation.sum()))
        self.assertTrue(np.isfinite([h["train"] for h in stats["history"]]).all())
        with self.assertRaisesRegex(ValueError, "run_index"):
            train.train_value_net(
                models.RunValueNet(),
                train.ValueData(
                    values.bases,
                    values.query_views,
                    values.query_keys,
                    values.targets,
                    values.end_hp,
                    values.validation,
                    fight_block=values.fight_block,
                ),
                config,
                torch.Generator().manual_seed(2),
            )


class DevicePathTests(unittest.TestCase):
    """The resident-rows path, run on a CPU device ("cpu:0" takes the same code as "cuda"): it must learn what the lab's path does."""

    def test_the_device_path_agrees_with_the_lab_path(self):
        base = vrun_golden.GOLDEN_CONFIG
        lab = vrun_golden.golden_fit(train.TrainConfig(**base))
        resident = vrun_golden.golden_fit(train.TrainConfig(**base, device="cpu:0"))
        for key, value in lab.items():
            with self.subTest(key=key):
                np.testing.assert_allclose(resident[key], value, rtol=1e-4, atol=1e-5)

    def test_the_nets_come_back_on_the_cpu_and_the_rows_were_resident(self):
        values = vrun_golden.synthetic_value_data()
        values.fill_fight_block(models.NumpyFightNet(models.numpy_params(models.FightNet())))
        net, fight = models.RunValueNet(hidden=32), models.FightNet()
        config = train.TrainConfig(epochs=2, batch=16, patience=2, device="cpu:0", rows_per_run=3)
        calls = []

        def hook(epoch, module):
            calls.append((epoch, module.training))
            return {"marker": 1.0}

        stats = train.train_value_net(net, values, config, torch.Generator().manual_seed(0), epoch_end=hook)
        fight_stats = train.train_fight_net(fight, vrun_golden.synthetic_fight_data(), config, torch.Generator().manual_seed(0))
        self.assertTrue(stats["resident"] and fight_stats["resident"])
        self.assertEqual(calls, [(0, False), (1, False)][: len(calls)])
        self.assertTrue(all("marker" in h for h in stats["history"]))
        self.assertEqual({p.device.type for p in net.parameters()}, {"cpu"})
        self.assertFalse(net.training)

    def test_a_device_without_room_falls_back_to_batches_and_says_so(self):
        values = vrun_golden.synthetic_value_data()
        values.fill_fight_block(models.NumpyFightNet(models.numpy_params(models.FightNet())))
        config = train.TrainConfig(epochs=1, batch=16, device="cpu:0")
        original = train._resident
        train._resident = lambda *args, **kwargs: None
        try:
            stats = train.train_value_net(models.RunValueNet(), values, config, torch.Generator().manual_seed(0))
        finally:
            train._resident = original
        self.assertFalse(stats["resident"])
        self.assertTrue(np.isfinite(stats["history"][0]["train"]))


class WidthTests(unittest.TestCase):
    def test_a_128_wide_export_plays_as_the_policy_reads_it(self):
        torch.manual_seed(4)
        net = models.RunValueNet(hidden=128, dropout=0.2)
        self.assertEqual(tuple(net.hidden2.weight.shape), (128, 128))
        with tempfile.TemporaryDirectory() as tmp:
            npz = models.save(Path(tmp) / "narrow.pt", models.FightNet(), net, {"generation": 1})
            played = models.load_numpy(npz).value
        self.assertEqual((played.w1.shape[0], played.w2.shape), (128, (128, 128)))
        rows = np.random.default_rng(0).random((9, models.VALUE_INPUT)).astype(np.float32)
        dense, counts = models.value_inputs(rows)
        net.eval()
        with torch.no_grad():
            logits, end_hp = net(torch.from_numpy(dense), torch.from_numpy(counts))
        got_logits, got_end = played.forward(dense, counts)
        np.testing.assert_allclose(got_logits, logits.numpy(), atol=1e-5)
        np.testing.assert_allclose(got_end, end_hp.numpy(), atol=1e-5)
        np.testing.assert_allclose(played.floor_probs(rows).sum(axis=1), 1.0, atol=1e-5)

    def test_the_default_net_is_the_labs(self):
        net = models.RunValueNet()
        self.assertEqual(tuple(net.hidden1.weight.shape), (256, models.VALUE_DENSE + models.EMBED))
        self.assertEqual(net.dropout.p, 0.1)


if __name__ == "__main__":
    unittest.main()
