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
import test_run_train_train as train_tests  # noqa: E402  (the module, so its TestCases are not collected twice)

from rl.run_train import models, shards, train  # noqa: E402
from rl.run_train.engine import CombatBudget  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "vrun_golden.npz"


class GoldenTests(unittest.TestCase):
    """`fixtures/vrun_golden.npz` was written by the code before the GPU path existed; the old path (`rows_per_run=0`, the CPU) must reproduce it exactly."""

    def test_the_old_path_refits_the_golden_checkpoint_bit_for_bit(self):
        with np.load(GOLDEN) as golden:
            expected = {key: golden[key] for key in golden.files}
        refit = vrun_golden.golden_fit(train.TrainConfig(**vrun_golden.GOLDEN_CONFIG, rows_per_run=0))  # the fixture is the old path's
        self.assertEqual(sorted(refit), sorted(expected))
        for key, value in expected.items():
            with self.subTest(key=key):
                if "history" in key:
                    np.testing.assert_array_equal(refit[key], value)
                else:
                    self.assertTrue(torch.equal(torch.from_numpy(refit[key]), torch.from_numpy(value)), key)

    def test_the_new_settings_default_to_the_labs_and_the_checkpoint_records_them(self):
        config = train.TrainConfig()
        self.assertEqual((config.device, config.hidden, config.dropout), ("cpu", 256, 0.1))
        self.assertEqual(config.rows_per_run, 4)  # adopted from the sweep (VRUN_SWEEP.md 1.5); the fixture's old path is rows_per_run=0
        with tempfile.TemporaryDirectory() as tmp:
            records = [train_tests.BufferTests.record(hp, 0) for hp in (20, 21, 22, 23)]
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
                ascension=train_tests.A9,
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

    def test_the_batch_fallback_trains_the_same_nets_as_the_resident_rows(self):
        config = train.TrainConfig(**vrun_golden.GOLDEN_CONFIG, device="cpu:0")
        resident = vrun_golden.golden_fit(config)
        original = train._resident
        train._resident = lambda *args, **kwargs: None
        try:
            fallback = vrun_golden.golden_fit(config)
        finally:
            train._resident = original
        self.assertEqual(sorted(fallback), sorted(resident))
        for key, value in resident.items():
            with self.subTest(key=key):
                np.testing.assert_allclose(fallback[key], value, rtol=1e-6, atol=1e-6)  # FightNet and V, the same values batch by batch


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


# --------------------------------------------------------------------------------------
# rl/vrun_sweep.py
# --------------------------------------------------------------------------------------

import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import vrun_sweep as sweep  # noqa: E402
from run_train import engine as run_train_engine  # noqa: E402
from run_train import keep  # noqa: E402
from run_train import train as rtrain  # noqa: E402
from test_run_train_shards import synthetic_record  # noqa: E402
from test_run_train_train import TAKE, died_on, entry  # noqa: E402

GEN = 3
BUDGET = run_train_engine.CombatBudget(3, 3)  # the module `vrun_sweep` and `run_train.train` share (not `rl.run_train`)


def make_record(i: int, *, kill: bool | None = None, seed: str | None = None) -> dict:
    """Run `i`: three decisions on floors 3, 8 and 12 and a final floor that is a kill (20) or a death on floor 7."""
    rng = np.random.default_rng(1000 + i)
    kill = bool(rng.random() < 0.4) if kill is None else kill
    outcome = {"final_floor": 20, "won": False, "died": True, "end_hp": 0, "end_max_hp": 80, "truncated": False} if kill else died_on(7)
    return synthetic_record(
        seed=seed or f"SW{i:05d}",
        generation=GEN,
        combat={"budget": str(BUDGET), "c": 0.1, "crn_fraction": 0.2},
        outcome=outcome,
        decisions=[entry(TAKE, 0, hp=int(rng.integers(20, 80)), floor=f) for f in (3, 8, 12)],
    )


def write_block(lab: Path, name: str, shard_records: list[list[dict]], *, done: bool, generation: int = GEN) -> Path:
    job = lab / "runs" / name
    directory = job / "shards"
    directory.mkdir(parents=True, exist_ok=True)
    names = []
    for k, records in enumerate(shard_records):
        names.append(f"r{k:04d}.jsonl.gz")
        shards.write_shard(directory / names[-1], records)
    manifest = {"kind": "selfplay-shards", "generation": generation, "base_sims": 3,
                "ascension": 9, "runs": sum(len(r) for r in shard_records), "run_shards": names}
    (directory / "shards.json").write_text(json.dumps(manifest), encoding="utf-8")
    if done:
        (job / "done.rc").write_text("0\n", encoding="utf-8")
    return job


def make_lab(root: Path, runs: int = 120) -> tuple[Path, Path, list[dict]]:
    """A lab with two closed blocks holding overlapping seeds (and copies inside a block) and one block still being written."""
    lab, agent = root / "lab", root / "agent"
    (agent / f"gen{GEN}" / "candidate").mkdir(parents=True)
    (agent / "current.json").write_text(json.dumps({"generation": GEN, "dir": str(agent / f"gen{GEN}" / "candidate")}), encoding="utf-8")
    first = [make_record(i) for i in range(0, 80)]
    second = [make_record(i) for i in range(60, runs)]
    # the copy of seed 70 in the second block differs, so that "the first record is kept" can be told from "the last"
    second[10] = make_record(70, kill=make_record(70)["outcome"]["final_floor"] != 20)
    copies = [make_record(100), make_record(100)]
    write_block(lab, "20261001T0000-selfplay", [first[:40], first[40:]], done=True)
    write_block(lab, "20261002T0000-selfplay", [second[:30], second[30:] + copies], done=True)
    write_block(lab, "20261003T0000-selfplay-gpu", [[make_record(500 + i) for i in range(5)]], done=False)
    return lab, agent, first


class SweepCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.lab, cls.agent, cls.first = make_lab(cls.root)
        cls.out = cls.root / "vrun"
        cls.folder = sweep.build_cache(cls.lab, cls.agent, cls.out)
        cls.manifest = json.loads((cls.folder / "manifest.json").read_text(encoding="utf-8"))
        cls.cache = sweep.load_cache(cls.folder)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_the_census_counts_the_copies_per_block_and_in_total(self):
        census = self.manifest["census"]
        first, second = census["per_block"]
        self.assertEqual((first["records"], first["kept"], first["copies_in_block"], first["copies_of_earlier_blocks"]), (80, 80, 0, 0))
        # 60 runs 60..119 (20 of them seen in block one), plus two extra copies of run 100
        self.assertEqual((second["records"], second["kept"]), (62, 40))
        self.assertEqual((second["copies_in_block"], second["copies_of_earlier_blocks"]), (2, 20))
        self.assertEqual((census["records"], census["unique_seeds"], census["extra_copies"]), (142, 120, 22))
        self.assertEqual((census["seeds_with_copies"], census["max_copies_of_a_seed"]), (21, 3))
        self.assertEqual(self.manifest["runs"], 120)
        self.assertEqual(self.manifest["rows"], 360)
        self.assertEqual([b["manifest"].split("/")[1] for b in self.manifest["blocks"]], ["20261001T0000-selfplay", "20261002T0000-selfplay"])
        self.assertEqual(len(self.manifest["skipped_blocks"]), 1)
        self.assertIn("selfplay-gpu", self.manifest["skipped_blocks"][0]["manifest"])
        self.assertEqual((self.manifest["fight_cap"], self.manifest["fights_exceed_cap"]), (400_000, False))

    def test_the_first_record_per_seed_is_kept(self):
        seeds = list(self.cache.run_seeds)
        self.assertEqual(len(seeds), len(set(seeds)))
        run = seeds.index("SW00070") + 1  # run_index is 1-based
        final = self.cache.values.final_floor[self.cache.values.run_index == run]
        self.assertEqual(set(final.tolist()), {make_record(70)["outcome"]["final_floor"]})
        self.assertEqual(len(final), 3)

    def test_the_key_is_the_sha256_of_the_manifests_shard_names_and_sizes(self):
        blocks, _ = sweep.closed_blocks(sweep.gen_loop.window_manifests(self.lab / "runs", GEN), self.lab, now=time.time())
        self.assertEqual(sweep.cache_key(blocks, self.lab), self.manifest["key"])
        self.assertEqual(self.folder.name, f"cache-{self.manifest['key'][:16]}")
        # a new shard in a closed block is a new cache; the same window is the same one
        self.assertEqual(sweep.build_cache(self.lab, self.agent, self.out), self.folder)
        blocks[0][1]["run_shards"] = [*blocks[0][1]["run_shards"][:1]]
        self.assertNotEqual(sweep.cache_key(blocks, self.lab), self.manifest["key"])

    def test_the_cache_reads_the_window_as_load_buffers_does(self):
        seen, expected = set(), []
        for name in ("20261001T0000-selfplay", "20261002T0000-selfplay"):
            for path in sorted((self.lab / "runs" / name / "shards").glob("r*.jsonl.gz")):
                for record in shards.read_shard(path):
                    if record["seed"] not in seen:
                        seen.add(record["seed"])
                        expected.append(record)
        path = self.root / "unique.jsonl.gz"
        shards.write_shard(path, expected)
        fights, values = rtrain.load_buffers([path], generation=GEN, combat_budget=BUDGET, ascension=9)
        mine = self.cache.values
        np.testing.assert_array_equal(mine.bases.take(np.arange(len(mine.targets))), values.bases.take(np.arange(len(values.targets))))
        for name in ("targets", "end_hp", "final_floor", "floor", "run_index", "v0", "query_keys"):
            np.testing.assert_array_equal(getattr(mine, name), getattr(values, name), err_msg=name)
        np.testing.assert_array_equal(self.cache.fights["labels"], fights.labels)
        np.testing.assert_array_equal(self.cache.fights["encounters"], fights.encounters)
        self.assertEqual(len(mine.query_views), len(values.query_views))
        np.testing.assert_array_equal(mine.run_hash, [rtrain._unit_hash(s) for s in np.repeat(self.cache.run_seeds, 3)])  # noqa: SLF001
        self.assertIsNone(mine.fight_block)  # H is per fold, never cached with the rows

    def test_the_splits_are_disjoint_and_cover_the_stated_hash_ranges(self):
        counts = self.manifest["counts"]
        self.assertEqual(counts["S"]["runs"] + counts["outside_S"]["runs"], counts["all"]["runs"])
        self.assertEqual(counts["S"]["rows"] + counts["outside_S"]["rows"], counts["all"]["rows"])
        self.assertEqual(sum(f["runs"] for f in counts["folds"].values()), counts["outside_S"]["runs"])
        self.assertEqual(sum(f["rows"] for f in counts["folds"].values()), counts["outside_S"]["rows"])
        self.assertGreater(counts["S"]["runs"], 3)
        h = self.cache.hashes
        in_s, lab_split = self.cache.is_s, sweep.in_lab_split(h)
        self.assertTrue(((h[in_s] >= 0.2) & (h[in_s] < 0.3)).all())
        self.assertTrue(((h[~in_s] < 0.2) | (h[~in_s] >= 0.3)).all())
        self.assertTrue((h[lab_split] < 0.1).all())
        self.assertFalse((lab_split & in_s).any())  # the lab's tenth and S do not overlap
        self.assertEqual(counts["S"]["kill_runs"], len({r for r in np.unique(self.cache.values.run_index[in_s & (self.cache.values.final_floor > 17)])}))

    def test_a_folds_fightnet_rows_exclude_that_fold_and_s(self):
        hashes = self.cache.fights["hash"]
        fold = sweep.fold_of(hashes)
        everywhere = np.zeros(len(hashes), int)
        for k in range(sweep.FOLDS):
            rows = sweep.fightnet_rows(self.cache, k)
            self.assertFalse((fold[rows] == k).any())
            self.assertFalse(sweep.in_selection(hashes[rows]).any())
            everywhere[rows] += 1
        outside = ~sweep.in_selection(hashes)
        np.testing.assert_array_equal(everywhere[outside], sweep.FOLDS - 1)  # each fight trains four of the five fits
        np.testing.assert_array_equal(everywhere[~outside], 0)
        np.testing.assert_array_equal(sweep.fightnet_rows(self.cache, None), np.flatnonzero(outside))

    def test_nothing_the_tool_writes_holds_an_absolute_path_or_a_user_name(self):
        for path in self.folder.rglob("*.json"):
            self.assertNoPaths(path.read_text(encoding="utf-8"), self.root)

    def assertNoPaths(self, text: str, root: Path) -> None:  # noqa: N802
        self.assertNotIn(str(root), text)
        self.assertNotIn(root.as_posix(), text)
        self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", text.replace("\\n", " ")), text[:300])
        self.assertNotIn("Users", text)
        self.assertNotIn(os.environ.get("USERNAME", "\0"), text)


class SplitTests(unittest.TestCase):
    def test_hash_ranges_and_folds(self):
        h = np.array([0.0, 0.0999, 0.1, 0.1999999, 0.2, 0.2999999, 0.3, 0.5, 0.7499, 0.9999])
        self.assertEqual(sweep.in_selection(h).tolist(), [False, False, False, False, True, True, False, False, False, False])
        self.assertEqual(sweep.in_lab_split(h).tolist(), [True, True, False, False, False, False, False, False, False, False])
        fold = sweep.fold_of(h)
        self.assertEqual(fold[4], -1)
        self.assertEqual(fold[5], -1)
        self.assertEqual(int(fold[6]), 0)  # the first hash outside S
        self.assertEqual(int(fold[3]), 4)  # the last one, just under 0.2
        self.assertTrue(((fold >= 0) | sweep.in_selection(h)).all())

    def test_every_run_outside_s_is_in_exactly_one_fold_of_similar_size(self):
        h = np.random.default_rng(0).random(50_000)
        fold = sweep.fold_of(h)
        outside = ~sweep.in_selection(h)
        self.assertTrue(((fold[outside] >= 0) & (fold[outside] < sweep.FOLDS)).all())
        self.assertTrue((fold[~outside] == -1).all())
        sizes = np.bincount(fold[outside], minlength=sweep.FOLDS)
        self.assertEqual(sizes.sum(), outside.sum())
        self.assertLess(sizes.max() / sizes.min(), 1.1)
        self.assertAlmostEqual(float(sweep.in_selection(h).mean()), 0.1, delta=0.01)

    def test_the_subsample_arms_are_nested_and_fightnets_validation_is_spread(self):
        h = np.random.default_rng(1).random(20_000)
        a, b, c = (sweep.subsample_runs(h, f) for f in (0.25, 0.5, 0.75))
        self.assertTrue((a <= b).all() and (b <= c).all())
        self.assertAlmostEqual(float(a.mean()), 0.25, delta=0.02)
        self.assertAlmostEqual(float(c.mean()), 0.75, delta=0.02)
        validation = sweep.fight_validation(h)
        self.assertAlmostEqual(float(validation.mean()), 0.1, delta=0.01)
        folds = sweep.fold_of(h[validation])
        self.assertEqual(set(folds[folds >= 0].tolist()), set(range(sweep.FOLDS)))


class SweepMathTests(unittest.TestCase):
    def rows(self, n_runs: int = 200, per_run: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        run = np.repeat(np.arange(n_runs), per_run)
        y = np.repeat((rng.random(n_runs) < 0.3).astype(float), per_run)
        v0 = rng.random(len(run))
        floor = np.tile(np.arange(per_run), n_runs).astype(float)
        return sweep.HeadRows(y, run, v0, floor), rng

    def test_the_keep_table_is_the_keep_functions_with_every_ece_bin(self):
        ev, rng = self.rows(seed=1)
        tr, _ = self.rows(seed=2)
        p = np.clip(0.3 + 0.5 * (ev.y - 0.3) + rng.normal(0, 0.1, len(ev.y)), 0.01, 0.99)
        head = sweep.head_table(p, ev, tr, n_boot=300)
        base = np.full(len(ev.y), tr.y.mean())
        by_hand = 1.0 - ((p - ev.y) ** 2).sum() / ((base - ev.y) ** 2).sum()
        self.assertAlmostEqual(head["brier_skill_vs_base_rate"]["value"], by_hand, places=12)
        v0_p = keep.v0_probability(tr.v0, tr.y, ev.v0)
        self.assertAlmostEqual(head["brier_skill_vs_v0"]["value"], 1.0 - ((p - ev.y) ** 2).sum() / ((v0_p - ev.y) ** 2).sum(), places=12)
        direct = keep.head_conditions(p, ev.y, ev.run, tr.y, v0_p, n_boot=300)
        self.assertEqual(head["brier_skill_vs_base_rate"], direct["brier_skill_vs_base_rate"])
        self.assertEqual(head["ece_bins_ge_30_positives"]["value"], direct["ece_bins_ge_30_positives"]["value"])
        bins = head["ece_bins_ge_30_positives"]["bins"]
        self.assertEqual(len(bins), keep.BINS)  # every bin, eligible or not
        self.assertEqual(sum(b["n"] for b in bins), len(p))
        self.assertFalse(head["brier_skill_vs_v0_floor_informational"]["gated"])
        perfect = sweep.head_table(ev.y.copy(), ev, tr, n_boot=300)
        self.assertAlmostEqual(perfect["brier_skill_vs_base_rate"]["value"], 1.0)
        self.assertTrue(perfect["brier_skill_vs_base_rate"]["pass"])

    def test_expected_floor_error_is_zero_for_the_true_floor_and_counts_a_win_as_w(self):
        final = np.array([3, 17, 20, 52, 53])
        p = np.zeros((5, models.FLOOR_CLASSES))
        p[np.arange(5), final - 1] = 1.0
        np.testing.assert_allclose(sweep.expected_floor_errors(p, final), 0.0, atol=1e-12)
        off = np.zeros_like(p)
        off[np.arange(5), np.minimum(final + 1, 53) - 1] = 1.0  # one floor too far; a win stays a win
        np.testing.assert_allclose(sweep.expected_floor_errors(off, final), [1, 1, 1, 1, 0], atol=1e-12)

    def test_the_floor_baseline_is_least_squares_on_v0_floor_and_a_constant(self):
        rng = np.random.default_rng(3)
        x = rng.random((500, 2))
        final = 2.0 * x[:, 0] + 3.0 * x[:, 1] + 5.0
        fitted = sweep.floor_baseline(x, final, x[:10])
        np.testing.assert_allclose(fitted, final[:10], atol=1e-9)
        noisy = final + rng.normal(0, 1.0, len(final))
        err = np.abs(sweep.floor_baseline(x, noisy, x) - noisy).mean()
        self.assertLess(err, np.abs(noisy - noisy.mean()).mean())  # beats the constant

    def test_arms_are_named_with_their_adoptable_flag(self):
        for name in ("lab", "lab-p10", "lr3e-4", "lr1e-4", "wd1e-3", "wd1e-2", "do0.3", "do0.5", "h64", "h128", "h512", "rows4", "rows8"):
            self.assertTrue(sweep.ARMS[name].adoptable, name)
        for name in ("bce-act1", "logit-v0h", "sub25", "sub50", "sub75", "noH", "lab-cpu"):
            self.assertFalse(sweep.ARMS[name].adoptable, name)
        self.assertEqual(sweep.ARMS["lr1e-4"].delta, {"lr": 1e-4, "epochs": 100, "patience": 10})
        self.assertEqual(sweep.arm_config(sweep.ARMS["h128"], "cuda").hidden, 128)
        self.assertEqual(sweep.arm_config(sweep.ARMS["lab-cpu"], "cuda").device, "cpu")
        combo = sweep.resolve_arm("do0.3+wd1e-3")
        self.assertEqual(combo.delta, {"dropout": 0.3, "weight_decay": 1e-3})
        self.assertTrue(combo.adoptable)
        with self.assertRaises(ValueError):
            sweep.resolve_arm("do0.3+noH")  # a diagnostic is never part of an adoptable recipe

    def test_a_block_is_closed_by_done_rc_or_by_ten_quiet_minutes(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab, now = Path(tmp), time.time()

            def block(name: str, age_s: float, done: bool):
                job = lab / "runs" / name
                directory = job / "shards"
                directory.mkdir(parents=True)
                manifest = directory / "shards.json"
                manifest.write_text("{}", encoding="utf-8")
                os.utime(manifest, (now - age_s, now - age_s))
                if done:
                    (job / "done.rc").write_text("0", encoding="utf-8")
                return directory, {"generation": 3, "run_shards": []}

            blocks = [block("a-done", 30, True), block("b-quiet", 601, False), block("c-writing", 599, False), block("d-fresh", 60, False)]
            kept, skipped = sweep.closed_blocks(blocks, lab, now=now)
            self.assertEqual([(d.parent.name, why) for d, _, why in kept], [("a-done", "done.rc"), ("b-quiet", "age")])
            self.assertEqual([s["manifest"].split("/")[1] for s in skipped], ["c-writing", "d-fresh"])
            kept, _ = sweep.closed_blocks(blocks, lab, now=now, closed_after_s=30 * 60)
            self.assertEqual([d.parent.name for d, _, _ in kept], ["a-done"])

    def test_at_most_two_arms_besides_lab_are_ever_confirmed(self):
        sweep.check_predeclared(set(), ["lab", "do0.3", "wd1e-3"])
        sweep.check_predeclared({"lab", "do0.3"}, ["lab", "do0.3"])  # finishing or re-reading a confirmation is not a new arm
        with self.assertRaises(SystemExit):
            sweep.check_predeclared({"lab", "do0.3", "wd1e-3"}, ["h64"])
        with self.assertRaises(SystemExit):
            sweep.check_predeclared(set(), ["lab", "do0.3", "wd1e-3", "rows4"])
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "confirm").mkdir()
            for name in ("lab.json", "do0.3.json", "do0.3-fold0.json", "lr3e-4-fold2.json", "lr3e-4-fold2.npz"):
                (Path(tmp) / "confirm" / name).write_text("{}", encoding="utf-8")
            self.assertEqual(sweep.confirmed_names(Path(tmp)), {"lab", "do0.3", "lr3e-4"})

    def test_joined_arms_may_not_set_one_setting_twice(self):
        for name in ("h64+h128", "lr3e-4+lr1e-4", "do0.3+do0.5"):
            with self.assertRaises(ValueError, msg=name):
                sweep.resolve_arm(name)
        self.assertEqual(sweep.resolve_arm("lr3e-4+lr3e-4").delta, sweep.ARMS["lr3e-4"].delta)

    def test_a_failed_priority_drop_is_said_not_swallowed(self):
        import contextlib  # noqa: PLC0415
        import io  # noqa: PLC0415

        from sts2_gym import procutil  # noqa: PLC0415

        original = procutil.lower_priority

        def broken(*_args):
            raise OSError("denied")

        procutil.lower_priority = broken
        captured = io.StringIO()
        try:
            with contextlib.redirect_stderr(captured):
                sweep._lower_priority()  # noqa: SLF001
        finally:
            procutil.lower_priority = original
        self.assertIn("could not lower the process priority", captured.getvalue())
        self.assertIn("denied", captured.getvalue())

    def test_the_selfplay_job_is_the_newest_folder_without_a_done_marker_and_a_fresh_stats_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab = Path(tmp)
            stats = {"calls": 100, "uptime_s": 10.0, "total_ms": {"p99": 2.5}}
            for name, done in (("20261001T0000-selfplay", True), ("20261002T0000-selfplay-gpu", False)):
                job = lab / "runs" / name
                job.mkdir(parents=True)
                (job / "prior_server_stats.json").write_text(json.dumps(stats), encoding="utf-8")
                if done:
                    (job / "done.rc").write_text("0", encoding="utf-8")
            self.assertEqual(sweep.running_selfplay_job(lab).name, "20261002T0000-selfplay-gpu")
            self.assertIsNone(sweep.running_selfplay_job(lab, now=time.time() + 3600))  # a stale file: nothing is playing
            reading = sweep.read_server_stats(lab / "runs" / "20261002T0000-selfplay-gpu" / "prior_server_stats.json")
            self.assertEqual((reading["calls"], reading["total_p99_ms"]), (100, 2.5))
            summary = sweep.summarise_samples([{"calls": 100, "uptime_s": 10.0, "total_p99_ms": 2.0}, {"calls": 300, "uptime_s": 20.0, "total_p99_ms": 4.0}])
            self.assertEqual((summary["p99_ms_max"], summary["calls_per_s"]), (4.0, 20.0))


class SweepRunTests(unittest.TestCase):
    """`run`, `ensemble`, `confirm` and `table` end to end on the synthetic window (CPU, a few seconds)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        lab, agent, _ = make_lab(cls.root)
        cls.folder = sweep.build_cache(lab, agent, cls.root / "vrun")
        torch.set_num_threads(1)
        run = ["run", "--cache", str(cls.folder), "--seeds", "1", "--device", "cpu"]
        assert sweep.main([*run, "--arms", "lab,rows4,logit-v0h,bce-act1,sub50,noH,lab-cpu"]) == 0
        assert sweep.main(["ensemble", "--cache", str(cls.folder), "--arm", "lab", "--device", "cpu"]) == 0
        assert sweep.main(["confirm", "--cache", str(cls.folder), "--arms", "lab,rows4", "--seeds", "1", "--device", "cpu"]) == 0
        assert sweep.main(["table", "--cache", str(cls.folder)]) == 0

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_a_fit_json_holds_the_s_keep_table_the_epoch_brier_and_the_minutes(self):
        record = json.loads((self.folder / "fits" / "lab-s0.json").read_text(encoding="utf-8"))
        head = record["keep"]["act1_boss_kill"]
        self.assertEqual(len(head["ece_bins_ge_30_positives"]["bins"]), keep.BINS)
        self.assertEqual(len(record["history"]), record["epochs"])
        self.assertTrue(all({"train", "validation", "s_brier", "s_bss_vs_base_rate"} <= set(e) for e in record["history"]))
        self.assertEqual(record["best_epoch"], 1 + int(np.argmin([e["validation"] for e in record["history"]])))
        self.assertGreaterEqual(record["fit_min"], 0.0)
        self.assertEqual((record["config"]["hidden"], record["config"]["rows_per_run"], record["adoptable"]), (256, 0, True))
        self.assertTrue((self.folder / "fits" / "lab-s0.npz").is_file())
        rows4 = json.loads((self.folder / "fits" / "rows4-s0.json").read_text(encoding="utf-8"))
        self.assertEqual(rows4["config"]["rows_per_run"], 4)  # the lab arm stays the recipe the sweep measured against; rows4 is the change
        self.assertEqual(record["s"]["head_rows"], int(head["rows"]))

    def test_every_arm_wrote_a_fit_and_a_rerun_skips_them(self):
        for name in ("lab", "rows4", "logit-v0h", "bce-act1", "sub50", "noH", "lab-cpu"):
            self.assertTrue((self.folder / "fits" / f"{name}-s0.json").is_file(), name)
        before = (self.folder / "fits" / "lab-s0.json").stat().st_mtime_ns
        self.assertEqual(sweep.main(["run", "--cache", str(self.folder), "--arms", "lab", "--seeds", "1", "--device", "cpu"]), 0)
        self.assertEqual((self.folder / "fits" / "lab-s0.json").stat().st_mtime_ns, before)
        self.assertEqual(sweep.main(["run", "--cache", str(self.folder), "--arms", "rows8", "--seeds", "1", "--device", "cpu", "--budget-min", "0"]), 0)
        self.assertFalse((self.folder / "fits" / "rows8-s0.json").exists())  # the budget was spent before it started

    def test_the_ensemble_averages_five_seeds(self):
        doc = json.loads((self.folder / "fits" / "ens5-lab.json").read_text(encoding="utf-8"))
        self.assertEqual(len(doc["member_brier"]), 5)
        self.assertLessEqual(doc["ensemble_brier"], np.mean(doc["member_brier"]) + 1e-12)  # averaging never loses Brier (convexity)
        self.assertFalse(doc["adoptable"])

    def test_the_confirmation_pools_every_non_s_run_once_and_compares_the_expected_floor(self):
        cache = sweep.load_cache(self.folder)
        doc = json.loads((self.folder / "confirm" / "lab.json").read_text(encoding="utf-8"))
        outside = ~cache.is_s
        self.assertEqual(doc["oof_rows"], int(outside.sum()))
        self.assertEqual(doc["oof_runs"], len(np.unique(cache.values.run_index[outside])))
        seen = np.concatenate([np.load(self.folder / "confirm" / f"lab-fold{k}.npz")["rows"] for k in range(sweep.FOLDS)])
        np.testing.assert_array_equal(np.sort(seen), np.flatnonzero(outside))
        head = doc["keep"]["act1_boss_kill"]
        self.assertEqual(head["rows"], int((outside & cache.head_rows).sum()))
        self.assertEqual(len(head["ece_bins_ge_30_positives"]["bins"]), keep.BINS)
        mae = doc["expected_floor_mae"]
        self.assertEqual(mae["all_rows"]["rows"], doc["oof_rows"])
        self.assertTrue(np.isfinite([mae["all_rows"]["model"], mae["all_rows"]["baseline"]]).all())
        self.assertEqual(mae["w"], 53.0)
        self.assertEqual(len(doc["folds"]), sweep.FOLDS)

    def test_a_fit_on_a_cpu_device_takes_the_resident_path_and_the_hook(self):
        cache = sweep.load_cache(self.folder)
        h = sweep.fit_fightnet_h(cache, "s0", 0, None, "cpu")
        record, p = sweep.s_phase_fit(cache, h, sweep.ARMS["lab"], 0, "cpu:0", n_boot=50)
        reference = json.loads((self.folder / "fits" / "lab-s0.json").read_text(encoding="utf-8"))
        self.assertEqual((record["device"], record["resident"], record["gpu_peak_mb"]), ("cpu:0", True, None))
        self.assertEqual(len(p), record["s"]["head_rows"])
        self.assertTrue(all("s_brier" in e for e in record["history"]))
        for ours, theirs in zip(record["history"][:3], reference["history"][:3], strict=False):
            self.assertAlmostEqual(ours["train"], theirs["train"], delta=1e-3 * abs(theirs["train"]))
            self.assertAlmostEqual(ours["validation"], theirs["validation"], delta=1e-3 * abs(theirs["validation"]))

    def test_the_logistic_diagnostic_reads_v0s_three_inputs_the_floor_and_the_eight_h_numbers(self):
        record = json.loads((self.folder / "fits" / "logit-v0h-s0.json").read_text(encoding="utf-8"))
        self.assertEqual(record["features"], ["hp_fraction", "potions_held", "act_index", "floor", *[f"h{i}" for i in range(8)]])

    def test_the_confirmation_base_rate_is_each_folds_own_training_rate(self):
        cache = sweep.load_cache(self.folder)
        doc = json.loads((self.folder / "confirm" / "lab.json").read_text(encoding="utf-8"))
        head = doc["keep"]["act1_boss_kill"]
        p_all, y_all, b_all = [], [], []
        for fold in range(sweep.FOLDS):
            train_rows = np.flatnonzero((cache.fold != fold) & ~cache.is_s)
            rate = sweep.HeadRows.of(cache, train_rows).y.mean()
            self.assertAlmostEqual(doc["fold_base_rates"][fold], rate, places=12)
            with np.load(self.folder / "confirm" / f"lab-fold{fold}.npz") as z:
                rows, kill = z["rows"], z["kill"]
            is_head = cache.head_rows[rows]
            p_all.append(kill[is_head])
            y_all.append(cache.kill[rows][is_head])
            b_all.append(np.full(int(is_head.sum()), rate))
        p, y, b = np.concatenate(p_all), np.concatenate(y_all), np.concatenate(b_all)
        self.assertAlmostEqual(head["brier_skill_vs_base_rate"]["value"], 1.0 - ((p - y) ** 2).sum() / ((b - y) ** 2).sum(), places=12)
        self.assertAlmostEqual(head["log_loss_vs_base_rate"]["base_rate"], keep.log_loss(b, y), places=12)
        self.assertAlmostEqual(head["base_rate"], b.mean(), places=12)
        self.assertIn("per fold", doc["base_rate"])

    def test_a_third_extra_confirmation_is_refused_before_any_fit(self):
        before = sorted(p.name for p in (self.folder / "confirm").glob("*"))
        with self.assertRaises(SystemExit):
            sweep.main(["confirm", "--cache", str(self.folder), "--arms", "lab,do0.3,wd1e-3", "--seeds", "1", "--device", "cpu"])
        self.assertEqual(sorted(p.name for p in (self.folder / "confirm").glob("*")), before)

    def test_beside_reads_the_servers_p99_while_a_fit_runs_and_gives_a_verdict(self):
        lab = self.root / "lab_beside"
        empty = lab / "runs" / "20261029T0000-selfplay"
        empty.mkdir(parents=True)
        (empty / "done.rc").write_text("0", encoding="utf-8")
        argv = ["beside", "--cache", str(self.folder), "--lab", str(lab), "--device", "cpu", "--before-s", "0.3", "--every-s", "0.05"]
        self.assertEqual(sweep.main(argv), 0)  # nothing is playing: skipped
        self.assertFalse((self.folder / "beside.json").exists())
        for name, p99, under in (("20261030T0000-selfplay-gpu", 2.5, True), ("20261031T0000-selfplay-gpu", 25.0, False)):
            job = lab / "runs" / name
            job.mkdir(parents=True)
            (job / "prior_server_stats.json").write_text(
                json.dumps({"calls": 100, "uptime_s": 10.0, "total_ms": {"p99": p99}}), encoding="utf-8")
            self.assertEqual(sweep.main(argv), 0)
            doc = json.loads((self.folder / "beside.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["job"], name)
            self.assertIsNone(doc["gpu_peak_mb"])  # no CUDA call on a CPU device
            self.assertGreaterEqual(doc["before"]["samples"], 1)
            self.assertGreaterEqual(doc["during"]["samples"], 1)
            self.assertEqual(doc["during"]["p99_ms_max"], p99)
            self.assertIs(doc["verdict"]["server_p99_under_bound"], under)
            self.assertIsNone(doc["verdict"]["client_p99_under_bound"])
            self.assertIsNone(doc["verdict"]["decisions_per_s"])
            self.assertIn("not measurable live", doc["verdict"]["decisions_per_s_note"])
            self.assertGreater(doc["fit_min"], 0)

    def test_a_diagnostic_arm_is_refused_for_confirmation(self):
        for arm in ("noH", "bce-act1", "sub25", "lab-cpu"):
            with self.assertRaises(SystemExit):
                sweep.main(["confirm", "--cache", str(self.folder), "--arms", f"lab,{arm}", "--seeds", "1", "--device", "cpu"])

    def test_the_report_has_both_tables_and_the_parity_row_and_no_path(self):
        text = (self.folder / "report.md").read_text(encoding="utf-8")
        for needle in ("FightNet data: 120 fights against the lab's cap of 400000", "## S table", "| lab | yes |", "| bce-act1 | no |", "## Confirmation table", "| lab |", "Parity", "Data curve", "Ensemble ens5-lab"):
            self.assertIn(needle, text)
        files = [p for p in self.folder.rglob("*") if p.suffix in {".json", ".md"}]
        self.assertGreater(len(files), 20)
        for path in files:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(str(self.root), text, path.name)
            self.assertNotIn(self.root.as_posix(), text, path.name)
            self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", text), path.name)


if __name__ == "__main__":
    unittest.main()
