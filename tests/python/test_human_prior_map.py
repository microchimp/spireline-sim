"""The human prior's map grid (rl/human_prior.py v2, rl/human_prior_infer.py; D58 addendum 13).

Pinned: the numpy tower equals the torch tower (1e-5) on random grids, a v2 shard is the v1 shard plus `grid` and `grid_ix` (every other array
byte for byte), the grid of a row is the act's map with the position of its afterstate (a route option moves it), a net with the tower starts as the plain
net (zero columns), the tower prior round-trips from the net through `export_net` to the numpy scorers, a fit step reaches the tower's weights, and the
refusals (a tower with no grids, a part v1 part v2 folder, `--map-grid` without `--truth`, never over `decisions-v1`).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _path in (HERE, ROOT / "rl", ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import human_prior as hp  # noqa: E402
import human_prior_infer as hpi  # noqa: E402
import human_starts as hs  # noqa: E402
import test_human_prior as thp  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover
    torch = None

from rl.run_train import features  # noqa: E402


def random_grids(n: int, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).random((n, features.MAP_GRID_SIZE)) < 0.3).astype(np.int8)


@unittest.skipIf(torch is None, "torch is not installed")
class TowerTests(unittest.TestCase):
    def test_the_layout_constants_agree(self):
        self.assertEqual((hp.MAP_ROWS, hp.MAP_COLS, hp.MAP_CHANNELS), features.MAP_GRID_SHAPE)
        self.assertEqual((hpi.MAP_ROWS, hpi.MAP_COLS, hpi.MAP_CHANNELS), features.MAP_GRID_SHAPE)
        self.assertEqual(hpi.MAP_GRID_SIZE, features.MAP_GRID_SIZE)

    def test_numpy_tower_equals_the_torch_tower_on_random_inputs(self):
        torch.manual_seed(5)
        tower = hp.map_tower_module().eval()
        # nonzero biases (the default ones are tiny), so a mix-up of the layout or the padding cannot hide; the outputs stay of order 1
        with torch.no_grad():
            for name, p in tower.named_parameters():
                if name.endswith("bias"):
                    p.copy_(torch.randn_like(p) * 0.1)
        params = {f"map.{k}": v.detach().numpy().astype(np.float32) for k, v in tower.state_dict().items()}
        self.assertEqual(sorted(params), sorted(hpi.MAP_KEYS))
        for n, seed in ((1, 1), (7, 2), (40, 3), (600, 4)):
            grid = random_grids(n, seed)
            want = tower(torch.from_numpy(grid)).detach().numpy()
            got = hpi.map_tower(params, grid)
            self.assertEqual(got.shape, (n, hpi.MAP_OUT))
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
        # grids that repeat are computed once and give the same rows
        grid = np.concatenate([random_grids(3, 9)] * 5)
        np.testing.assert_allclose(hpi.map_tower(params, grid), tower(torch.from_numpy(grid)).detach().numpy(), rtol=1e-5, atol=1e-5)
        self.assertGreater(float(np.abs(hpi.map_tower(params, grid)).sum()), 0.0)

    def test_the_tower_reads_the_cell_layout_row_column_channel(self):
        # one cell set: the output moves when the cell moves, and only through the convolutions' reach
        torch.manual_seed(1)
        tower = hp.map_tower_module().eval()
        a = np.zeros((1, 17, 7, 13), dtype=np.int8)
        b = a.copy()
        a[0, 3, 2, 1] = 1
        b[0, 3, 2, 4] = 1  # the same cell, another channel
        outs = [tower(torch.from_numpy(x.reshape(1, -1))).detach().numpy() for x in (a, b)]
        self.assertFalse(np.allclose(outs[0], outs[1]))
        params = {f"map.{k}": v.detach().numpy().astype(np.float32) for k, v in tower.state_dict().items()}
        for x, want in zip((a, b), outs):
            np.testing.assert_allclose(hpi.map_tower(params, x.reshape(1, -1)), want, rtol=1e-5, atol=1e-5)


class ShardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        thp.HumanPriorV1Tests.setUpClass()
        cls.h = thp.HumanPriorV1Tests("test_map_identity_check")
        cls.train_id = thp.HumanPriorV1Tests.train_id
        cls.F = hp.stack().F

    def specs(self):
        specs = self.h.build(self.h.truth())[0]
        for sp in specs:
            sp["run_ix"] = 0
        return specs

    def write(self, tmp: str, specs, map_grid: bool, name: str = "shard-00000.npz") -> Path:
        path = Path(tmp) / name
        hp.write_shard(path, specs, [self.train_id], thp.fake_judge(), {}, None, map_grid=map_grid)
        return path

    def test_a_v2_shard_is_the_v1_shard_plus_the_grid(self):
        specs = self.specs()
        with tempfile.TemporaryDirectory() as tmp:
            v1 = self.write(tmp, specs, False, "v1.npz")
            v2 = self.write(tmp, specs, True, "v2.npz")
            with np.load(v1) as a, np.load(v2) as b:
                self.assertEqual(sorted(set(b.files) - set(a.files)), ["grid", "grid_ix"])
                for key in a.files:
                    self.assertEqual(a[key].tobytes(), b[key].tobytes(), key)  # every v1 array is unchanged
                rows = len(a["dense"])
                self.assertEqual(b["grid_ix"].shape, (rows,))
                self.assertEqual(b["grid"].dtype, np.int8)
                self.assertEqual(b["grid"].shape[1], features.MAP_GRID_SIZE)
                self.assertLess(len(b["grid"]), rows)  # options that share a position share a grid
                self.assertEqual(len({g.tobytes() for g in b["grid"]}), len(b["grid"]))  # and the table holds each once

    def test_a_row_carries_the_grid_of_its_afterstate(self):
        specs = self.specs()
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp, specs, True)
            ds = hp.Dataset.load(Path(tmp), [hs.exam_bucket(self.train_id)], None)
        self.assertTrue(ds.has_grid)
        view_of = hp.spec_view
        start = 0
        for sp in specs:
            view = view_of(sp)
            for j, o in enumerate(sp["options"]):
                row = start + j
                got = ds.grid_rows(np.array([row]))[0]
                if o["k"] == "map.node":
                    # a route option: the position is the room it goes to
                    here = np.argwhere(got.reshape(features.MAP_GRID_SHAPE)[:, :, 12] == 1).tolist()
                    self.assertEqual(here, [[o["node"][1], o["node"][0]]])
                else:
                    np.testing.assert_array_equal(got, features.map_grid(view))
                    self.assertEqual(int(got.reshape(features.MAP_GRID_SHAPE)[sp["position"][1], sp["position"][0], 12]), 1)
            start += len(sp["options"])
        routes = [s for s in specs if s["kind"] == "route"]
        self.assertEqual(len(routes), 2)
        # the options of one route screen differ from each other in their grids
        first = int(ds.dec[[i for i, s in enumerate(specs) if s["kind"] == "route"][0], hp.C["start"]])
        grids = ds.grid_rows(np.arange(first, first + 3))
        self.assertEqual(len({g.tobytes() for g in grids}), 3)

    def test_the_grid_of_a_route_option_matches_option_grids(self):
        # the play side (`option_grids`, what `option_rows(..., with_grid=True)` returns) and the build side (`spec_grids`) agree
        F = self.F
        for sp in self.specs():
            if sp["kind"] != "route":
                continue
            view = hp.spec_view(sp)
            options = [hp._option(o, view, j) for j, o in enumerate(sp["options"])]
            np.testing.assert_array_equal(hpi.option_grids(view, options), np.stack(hp.spec_grids(sp)))
            for opt, g in zip(options, hp.spec_grids(sp)):
                np.testing.assert_array_equal(g, F.map_grid(F.apply_option(view, opt)))

    def test_a_folder_with_and_without_the_grid_is_refused_and_a_v0_spec_cannot_have_one(self):
        specs = self.specs()
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp, specs, True, "shard-00000.npz")
            self.write(tmp, specs, False, "shard-00001.npz")
            with self.assertRaises(SystemExit):
                hp.Dataset.load(Path(tmp), [hs.exam_bucket(self.train_id)], None)
        v0 = [dict(sp) for sp in specs]
        for sp in v0:
            sp.pop("map_nodes")
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "truth"):
            self.write(tmp, v0, True)

    def test_build_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):  # the grid is the true map's
                hp.main(["build", "--map-grid", "--out-dir", tmp])
            with self.assertRaises(SystemExit):  # decisions-v1 is never overwritten
                hp.main(["build", "--map-grid", "--truth", tmp, "--decisions", "decisions-v1", "--out-dir", tmp])

    def test_row_scores_need_the_grid_for_a_tower_prior(self):
        specs = self.specs()
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp, specs, False)
            ds = hp.Dataset.load(Path(tmp), [hs.exam_bucket(self.train_id)], None)
        self.assertFalse(ds.has_grid)
        with self.assertRaisesRegex(ValueError, "grid"):
            hp.row_scores_np(ds, None, lambda d, c, grid=None: np.zeros(len(d)), grid=True)
        with self.assertRaisesRegex(ValueError, "grid"):
            ds.grid_rows(np.arange(2))


@unittest.skipIf(torch is None, "torch is not installed")
class NetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        thp.HumanPriorV1Tests.setUpClass()
        cls.h = thp.HumanPriorV1Tests("test_map_identity_check")
        cls.train_id = thp.HumanPriorV1Tests.train_id

    def v2_dataset(self, tmp: str):
        specs = self.h.build(self.h.truth())[0]
        for sp in specs:
            sp["run_ix"] = 0
            sp["flags"] &= ~hp.F_REPLAY_MISMATCH  # the synthetic run's deck does not replay to its final deck: keep its decisions in a fit
        hp.write_shard(Path(tmp) / "shard-00000.npz", specs, [self.train_id], thp.fake_judge(), {}, None, map_grid=True)
        return hp.Dataset.load(Path(tmp), [hs.exam_bucket(self.train_id)], None)

    def test_a_net_with_the_tower_starts_as_the_plain_net(self):
        torch.manual_seed(3)
        plain = hp.make_net()
        # a judge file in the layout `init_from_judge` reads: the plain net's tensors under value/, plus the heads it drops
        state = {f"value/{k}": v.detach().numpy() for k, v in plain.state_dict().items() if not k.startswith("score.")}
        state["value/floor_head.weight"] = np.zeros((2, 256), np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            judge = Path(tmp) / "judge.npz"
            np.savez(judge, **state)
            net = hp.make_net(map_grid=True)
            hp.init_from_judge(net, judge)
            self.assertEqual(net.hidden1.weight.shape, (256, 350 + 104 + 64))
            self.assertEqual(float(net.hidden1.weight[:, -64:].detach().abs().sum()), 0.0)
            # a net started from a plain prior's weights does the same
            exported = Path(tmp) / "plain.npz"
            hp.export_net(plain, exported, {})
            net2 = hp.make_net(map_grid=True)
            hp.init_from_prior(net2, exported)
            plain.eval()
            net.eval()
            net2.eval()
            rng = np.random.default_rng(2)
            dense = torch.from_numpy(rng.random((9, 350)).astype(np.float32))
            counts = torch.from_numpy((rng.random((9, 1513)) < 0.01).astype(np.float32))
            grid = torch.from_numpy(random_grids(9, 4))
            # (the judge file has no score head, so the nets' score heads are fresh: give them the plain net's)
            net.score.load_state_dict(plain.score.state_dict())
            net2.score.load_state_dict(plain.score.state_dict())
            with torch.no_grad():
                np.testing.assert_allclose(net(dense, counts, grid=grid).numpy(), plain(dense, counts).numpy(), rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(net2(dense, counts, grid=grid).numpy(), plain(dense, counts).numpy(), rtol=1e-5, atol=1e-5)
            with self.assertRaises(ValueError):
                net(dense, counts)

    def test_the_tower_prior_round_trips_to_the_numpy_scorers_and_learns(self):
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as tmp:
            ds = self.v2_dataset(tmp)
            net = hp.make_net(map_grid=True)
            with torch.no_grad():  # a tower that speaks: nonzero columns for it
                net.hidden1.weight[:, -64:] = torch.randn(256, 64) * 0.05
            gt = hp.Gpu(ds, None, torch.device("cpu"), 1.0)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
            net.train()
            before = {k: v.detach().clone() for k, v in net.state_dict().items() if k.startswith("map.")}
            for _ in range(2):
                loss = gt.loss(net, "card", torch.from_numpy(ds.kind_idx["card"]))
                self.assertTrue(torch.isfinite(loss).item())
                opt.zero_grad()
                loss.backward()
                opt.step()
            moved = [k for k, v in net.state_dict().items() if k.startswith("map.") and not torch.equal(v, before[k])]
            self.assertIn("map.conv1.weight", moved)  # the gradient reaches the first convolution
            self.assertIn("map.fc.weight", moved)
            path = Path(tmp) / "weights.npz"
            hp.export_net(net, path, {"kinds": ["card", "ancient", "rest", "shop", "route"]})
            prior = hp.NumpyPrior(path)
            self.assertTrue(prior.map_grid)
            net.eval()
            torch_scores = gt.scores(net)
            np.testing.assert_allclose(torch_scores, hp.row_scores_np(ds, None, prior), rtol=1e-4, atol=1e-4)
            scorer = hpi.PriorScorer.load(path)
            self.assertTrue(scorer.map_grid)
            self.assertTrue(scorer.meta["map_grid"])
            x = hp.assemble_np_v1(ds.dense)
            counts = ds.counts.astype(np.float32)
            got = scorer.logits(x, counts, ds.grid_rows(np.arange(len(x))))
            np.testing.assert_allclose(got, torch_scores, rtol=1e-4, atol=1e-4)
            with self.assertRaisesRegex(ValueError, "map tower"):
                scorer.logits(x, counts)  # a tower prior never scores without grids
            # the grid matters: the same rows with another grid score differently
            shuffled = ds.grid_rows(np.arange(len(x)))[::-1].copy()
            self.assertFalse(np.allclose(scorer.logits(x, counts, shuffled), got))

    def test_a_plain_file_is_unchanged_and_a_half_tower_is_refused(self):
        torch.manual_seed(1)
        with tempfile.TemporaryDirectory() as tmp:
            plain, tower = hp.make_net(), hp.make_net(map_grid=True)
            hp.export_net(plain, Path(tmp) / "plain.npz", {})
            hp.export_net(tower, Path(tmp) / "tower.npz", {})
            p = hpi.PriorScorer.load(Path(tmp) / "plain.npz")
            self.assertFalse(p.map_grid)
            # the tower is ignored when a plain prior is handed a grid, and a plain prior scores as it always did
            rng = np.random.default_rng(1)
            dense = rng.random((4, 350)).astype(np.float32)
            counts = (rng.random((4, 1513)) < 0.01).astype(np.float32)
            np.testing.assert_array_equal(p.logits(dense, counts), p.logits(dense, counts, random_grids(4)))
            with np.load(Path(tmp) / "tower.npz") as z:
                arrays = {k: z[k] for k in z.files}
            half = {k: v for k, v in arrays.items() if k != "map.fc.bias"}
            with self.assertRaisesRegex(ValueError, "part of a map tower"):
                hpi.PriorScorer(half, {"map_grid": True})
            with self.assertRaisesRegex(ValueError, "meta says"):
                hpi.PriorScorer({k: v for k, v in arrays.items() if k != "meta" and not k.startswith("map.")}, {"map_grid": True})

    def test_fit_one_with_the_tower_runs_a_whole_variant(self):
        import types

        torch.manual_seed(2)
        with tempfile.TemporaryDirectory() as tmp:
            ds = self.v2_dataset(tmp)
            dev = torch.device("cpu")
            gt, gv = hp.Gpu(ds, None, dev, 1.0), hp.Gpu(ds, None, dev, 1.0)
            a = types.SimpleNamespace(seed=1, epochs=2, patience=2, batch=8, with_shops=False)
            res = hp.fit_one(gt, gv, "scratch", 1.0, a, ["card", "ancient", "rest", "shop", "route"], Path("unused"), map_grid=True)
            self.assertTrue(res["map_grid"])
            self.assertEqual(res["epochs_run"], 2)
            self.assertTrue(np.isfinite(res["best"]["score"]))
            hp.export_net(res["net"], Path(tmp) / "w.npz", {})
            with np.load(Path(tmp) / "w.npz") as z:
                self.assertTrue(set(hpi.MAP_KEYS) <= set(z.files))
                self.assertTrue(bool(json_meta(z)["map_grid"]))


class RegularisationTests(unittest.TestCase):
    """The two training-time aids of `fit --map-grid`: dropout on the tower's outputs and the left-right mirror of a decision's maps."""

    def test_the_mirror_of_a_grid_is_the_grid_of_the_mirrored_map(self):
        from test_features_map_grid import HAND_MAP, make_view

        nodes = tuple(features.MapNode(6 - n.col, n.row, n.node_type, tuple((6 - c, r) for c, r in n.children)) for n in HAND_MAP)
        for position in ((3, 0), (2, 1), (4, 1), (3, 4)):
            with self.subTest(position=position):
                grid = features.map_grid(make_view(position=position))[None]
                mirrored = features.map_grid(make_view(map_nodes=nodes, position=(6 - position[0], position[1])))[None]
                np.testing.assert_array_equal(hp.mirror_grid(grid), mirrored)
                np.testing.assert_array_equal(hp.mirror_grid(hp.mirror_grid(grid)), grid)  # twice is the map again
        if torch is not None:
            g = torch.from_numpy(random_grids(5, 3))
            np.testing.assert_array_equal(hp.mirror_grid(g).numpy(), hp.mirror_grid(g.numpy()))  # torch and numpy agree

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_tower_dropout_acts_in_training_only_and_the_exported_weights_are_the_same_set(self):
        torch.manual_seed(1)
        net = hp.make_net(map_grid=True, map_dropout=0.5)
        with torch.no_grad():
            net.hidden1.weight[:, -64:] = torch.randn(256, 64) * 0.05
        rng = np.random.default_rng(1)
        dense = torch.from_numpy(rng.random((8, 350)).astype(np.float32))
        counts = torch.from_numpy((rng.random((8, 1513)) < 0.01).astype(np.float32))
        grid = torch.from_numpy(random_grids(8, 2))
        net.eval()
        with torch.no_grad():
            a, b = net(dense, counts, grid=grid), net(dense, counts, grid=grid)
        torch.testing.assert_close(a, b)  # no dropout at inference: the numpy scorer is this net
        net.train()
        with torch.no_grad():
            c, d = net(dense, counts, grid=grid), net(dense, counts, grid=grid)
        self.assertFalse(torch.allclose(c, d))
        self.assertEqual(set(hp.make_net(map_grid=True).state_dict()), set(net.state_dict()))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_the_flip_changes_training_logits_and_never_validation_ones(self):
        torch.manual_seed(4)
        NetTests.setUpClass()
        with tempfile.TemporaryDirectory() as tmp:
            ds = NetTests.v2_dataset(NetTests("test_a_plain_file_is_unchanged_and_a_half_tower_is_refused"), tmp)
            net = hp.make_net(map_grid=True)
            with torch.no_grad():
                net.hidden1.weight[:, -64:] = torch.randn(256, 64) * 0.05
            net.eval()
            gt = hp.Gpu(ds, None, torch.device("cpu"), 1.0)
            didx = torch.arange(len(ds.dec))
            base, _, _ = gt.logits(net, didx, fixed=True)
            gt.flip_p = 1.0
            same, _, _ = gt.logits(net, didx, fixed=True)  # validation (fixed): never flipped
            torch.testing.assert_close(base, same)
            flipped, _, _ = gt.logits(net, didx, fixed=False)
            self.assertFalse(torch.allclose(base, flipped))
            gt.flip_p = 0.0
            plain, _, _ = gt.logits(net, didx, fixed=False)
            torch.testing.assert_close(base, plain)


def json_meta(z) -> dict:
    import json

    return json.loads(str(z["meta"]))


if __name__ == "__main__":
    unittest.main()
