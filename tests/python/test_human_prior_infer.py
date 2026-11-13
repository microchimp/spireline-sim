"""The human prior in numpy (rl/human_prior_infer.py): the map tower and the grids beside the rows (D58 addendum 13).

Pinned: `option_rows` is the same rows, byte for byte, with or without the grid, and the grid is a second array (not a wider row) whose map
step options carry the position of the room they go to; a prior or student with the map tower is scored with the grids and refuses to score
without them, a plain file scores as it always did; the tower's file format (`map_grid` in the meta, the six arrays, shapes checked); a
blended chooser hands a tower scorer its option grids.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import human_prior_infer as hpi  # noqa: E402
import test_features_map_grid as tfm  # noqa: E402
from rl.run_train import features, models, vocab  # noqa: E402
from rl.run_train.features import CardRef, Decision, Option  # noqa: E402


def tower_params(seed: int = 4, scale: float = 0.05) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shapes = {"embed.deck.weight": (64, 1154), "embed.relics.weight": (32, 296), "embed.potions.weight": (8, 63), "hidden1.weight": (256, 454 + 64),
              "hidden1.bias": (256,), "hidden2.weight": (256, 256), "hidden2.bias": (256,), "score.weight": (1, 256), "score.bias": (1,),
              "map.conv1.weight": (16, 13, 3, 3), "map.conv1.bias": (16,), "map.conv2.weight": (32, 16, 3, 3), "map.conv2.bias": (32,),
              "map.fc.weight": (64, 32 * 17 * 7), "map.fc.bias": (64,)}
    return {k: (rng.standard_normal(s) * scale).astype(np.float32) for k, s in shapes.items()}


def plain_params(seed: int = 4) -> dict[str, np.ndarray]:
    p = tower_params(seed)
    p["hidden1.weight"] = p["hidden1.weight"][:, :454]
    return {k: v for k, v in p.items() if not k.startswith("map.")}


def fake_nets() -> types.SimpleNamespace:
    """The judgement's nets as far as `option_rows` reads them: the base width and a fight block of zeros (rows' shape and the grids are what is pinned)."""
    probs = lambda rows: np.full((len(rows), models.FLOOR_CLASSES), 1.0 / models.FLOOR_CLASSES, dtype=np.float32)  # noqa: E731
    return types.SimpleNamespace(base_size=features.BASE_SIZE, fight=types.SimpleNamespace(summary=lambda bases, queries: np.zeros((len(bases), 8), np.float32)),
                                 value=types.SimpleNamespace(floor_probs=probs))


def options_for_view() -> list[Option]:
    """A deck screen's options (a card and a skip) -- the position stays -- and the two rooms a map screen from (3, 0) offers."""
    return [Option("card.take", 0, card=CardRef(472, False)), Option("card.skip", 1)]


MAP_OPTIONS = [Option("map.node", 0, node=(2, 1, vocab.NODE_MONSTER)), Option("map.node", 1, node=(4, 1, vocab.NODE_UNKNOWN))]


class OptionRowsTests(unittest.TestCase):
    def test_the_rows_are_the_same_bytes_with_and_without_the_grid(self):
        view, nets = tfm.make_view(), fake_nets()
        for options in (options_for_view(), MAP_OPTIONS):
            rows = hpi.option_rows(nets, view, options)
            rows2, grid = hpi.option_rows(nets, view, options, with_grid=True)
            self.assertEqual(rows.tobytes(), rows2.tobytes())
            self.assertEqual(rows.shape, (len(options), features.BASE_SIZE + 8))  # 1,863 wide: the grid is not in the row
            self.assertEqual((len(options), hpi.MAP_GRID_SIZE), grid.shape)
            self.assertEqual(np.int8, grid.dtype)

    def test_a_card_option_keeps_the_position_and_a_map_step_moves_it(self):
        view, nets = tfm.make_view(position=(3, 0)), fake_nets()
        _, grid = hpi.option_rows(nets, view, options_for_view(), with_grid=True)
        np.testing.assert_array_equal(grid[0], grid[1])
        np.testing.assert_array_equal(grid[0], features.map_grid(view))
        _, grid = hpi.option_rows(nets, view, MAP_OPTIONS, with_grid=True)
        shaped = grid.reshape(len(MAP_OPTIONS), *features.MAP_GRID_SHAPE)
        self.assertEqual([[1, 2]], np.argwhere(shaped[0, :, :, tfm.HERE] == 1).tolist())  # row 1, column 2
        self.assertEqual([[1, 4]], np.argwhere(shaped[1, :, :, tfm.HERE] == 1).tolist())
        # the reachable bit follows the position: from (4, 1) the monster room's branch is out of reach
        self.assertEqual(1, int(shaped[0, 2, 2, tfm.REACH]))
        self.assertEqual(0, int(shaped[1, 1, 2, tfm.REACH]))
        np.testing.assert_array_equal(grid[0], features.map_grid(features.apply_option(view, MAP_OPTIONS[0])))

    def test_no_options_no_grids(self):
        self.assertEqual((0, hpi.MAP_GRID_SIZE), hpi.option_grids(tfm.make_view(), []).shape)


class ScorerTests(unittest.TestCase):
    def setUp(self):
        self.rows = np.random.default_rng(1).normal(0, 0.4, (6, 1863)).astype(np.float32)
        self.grid = (np.random.default_rng(2).random((6, hpi.MAP_GRID_SIZE)) < 0.25).astype(np.int8)

    def test_a_tower_prior_scores_with_the_grids_and_refuses_without(self):
        p = hpi.PriorScorer(tower_params(), {"map_grid": True})
        self.assertTrue(p.map_grid)
        z = p.row_logits(self.rows, self.grid)
        self.assertEqual((6,), z.shape)
        with self.assertRaisesRegex(ValueError, "map tower"):
            p.row_logits(self.rows)
        with self.assertRaisesRegex(ValueError, "map tower"):
            p.row_logits(self.rows, self.grid[:3])  # one grid per row
        # the tower matters, and the same rows with the same grid give the same logits however they are batched
        self.assertFalse(np.allclose(z, p.row_logits(self.rows, self.grid[::-1].copy())))
        np.testing.assert_allclose(p.row_logits(self.rows[:2], self.grid[:2]), z[:2], rtol=1e-5, atol=1e-5)
        # the first layer's 64 extra columns are what carries the tower: zero them and the grid no longer matters
        flat = dict(tower_params())
        flat["hidden1.weight"] = flat["hidden1.weight"].copy()
        flat["hidden1.weight"][:, 454:] = 0.0
        q = hpi.PriorScorer(flat, {"map_grid": True})
        np.testing.assert_allclose(q.row_logits(self.rows, self.grid), q.row_logits(self.rows, self.grid[::-1].copy()), atol=1e-6)

    def test_a_plain_prior_ignores_a_grid_and_scores_as_it_always_did(self):
        params = plain_params()
        p = hpi.PriorScorer(params)
        self.assertFalse(p.map_grid)
        np.testing.assert_array_equal(p.row_logits(self.rows), p.row_logits(self.rows, self.grid))
        dense, counts = models.value_inputs(self.rows)
        h = np.maximum(np.concatenate([dense, counts[:, :1154] @ params["embed.deck.weight"].T, counts[:, 1154:1450] @ params["embed.relics.weight"].T,
                                       counts[:, 1450:] @ params["embed.potions.weight"].T], axis=1) @ params["hidden1.weight"].T + params["hidden1.bias"], 0)
        h = np.maximum(h @ params["hidden2.weight"].T + params["hidden2.bias"], 0)
        np.testing.assert_allclose(p.row_logits(self.rows), (h @ params["score.weight"].T + params["score.bias"])[:, 0], rtol=1e-5, atol=1e-5)

    def test_the_files_format_is_checked(self):
        with self.assertRaisesRegex(ValueError, "part of a map tower"):
            hpi.PriorScorer({k: v for k, v in tower_params().items() if k != "map.fc.bias"}, {"map_grid": True})
        with self.assertRaisesRegex(ValueError, "meta says"):
            hpi.PriorScorer(tower_params(), {})  # a tower without the flag
        with self.assertRaisesRegex(ValueError, "meta says"):
            hpi.PriorScorer(plain_params(), {"map_grid": True})  # the flag without a tower
        wrong = tower_params()
        wrong["map.fc.weight"] = wrong["map.fc.weight"][:, :-7]
        with self.assertRaisesRegex(ValueError, "wrong shapes"):
            hpi.PriorScorer(wrong, {"map_grid": True})
        narrow = tower_params()
        narrow["hidden1.weight"] = narrow["hidden1.weight"][:, :454]  # a first layer that cannot read the tower's 64 numbers
        with self.assertRaisesRegex(ValueError, "first layer"):
            hpi.PriorScorer(narrow, {"map_grid": True})

    def test_log_probs_take_the_grids_from_the_options(self):
        view, nets = tfm.make_view(position=(3, 0)), fake_nets()
        p = hpi.PriorScorer(tower_params(), {"map_grid": True})
        rows, grid = hpi.option_rows(nets, view, MAP_OPTIONS, with_grid=True)
        np.testing.assert_allclose(p.log_probs(view, MAP_OPTIONS, nets), hpi.log_softmax(p.row_logits(rows, grid)))
        plain = hpi.PriorScorer(plain_params())
        np.testing.assert_allclose(plain.log_probs(view, MAP_OPTIONS, nets), hpi.log_softmax(plain.row_logits(rows)))

    def test_a_student_file_carries_the_tower_and_loads_it_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "student.npz"
            sha = hpi.write_student(path, tower_params(), kinds=("route",), prior_sha256="ab" * 32, fit={"x": 1})
            with np.load(path) as z:
                self.assertTrue(set(hpi.MAP_KEYS) <= set(z.files))
                meta = json.loads(str(z["meta"]))
            self.assertTrue(meta["map_grid"])
            self.assertEqual(["route"], meta["student"]["kinds"])
            s = hpi.StudentScorer.load(path)
            self.assertEqual((sha, True, ("route",)), (s.sha256, s.map_grid, s.kinds))
            np.testing.assert_allclose(s.row_logits(self.rows, self.grid), hpi.PriorScorer(tower_params(), {"map_grid": True}).row_logits(self.rows, self.grid),
                                       rtol=1e-6, atol=1e-6)
            half = {k: v for k, v in tower_params().items() if k != "map.conv2.bias"}
            with self.assertRaisesRegex(ValueError, "part of a map tower"):
                hpi.write_student(Path(tmp) / "half.npz", half, kinds=("route",), prior_sha256="")
            plain = Path(tmp) / "plain.npz"
            hpi.write_student(plain, plain_params(), kinds=("card",), prior_sha256="")
            with np.load(plain) as z:
                self.assertNotIn("map_grid", json.loads(str(z["meta"])))
            self.assertFalse(hpi.StudentScorer.load(plain).map_grid)


class ChooserTests(unittest.TestCase):
    def policy(self, scorer):
        import imitation_chooser as ic

        pol = object.__new__(ic.BlendedPolicy)
        pol.nets, pol.W, pol.pick_mode, pol.skip_bias, pol.tau, pol.eps = fake_nets(), 53.0, ic.PICK_ARGMAX, 0.0, 0.0, 0.0
        pol.prior, pol.student, pol.last = scorer, None, None
        return ic, pol

    def test_the_blended_chooser_hands_a_tower_scorer_its_option_grids(self):
        seen = []

        class Recording(hpi.PriorScorer):
            def row_logits(self, rows, grid=None):
                seen.append((rows.copy(), None if grid is None else grid.copy()))
                return super().row_logits(rows, grid)

        view = tfm.make_view(position=(3, 0))
        ic, pol = self.policy(Recording(tower_params(), {"map_grid": True}))
        decision = Decision("map", tuple(MAP_OPTIONS))
        pick = pol._decide(view, decision, None, "route", 0.5, pol.prior)  # noqa: SLF001
        self.assertIn(pick.index, (0, 1))
        rows, grid = seen[-1]
        want_rows, want_grid = hpi.option_rows(pol.nets, view, MAP_OPTIONS, with_grid=True)
        np.testing.assert_array_equal(rows, want_rows)
        np.testing.assert_array_equal(grid, want_grid)
        # a plain scorer is called as it always was: rows alone
        calls = []

        class Plain(hpi.PriorScorer):
            def row_logits(self, rows, grid=None):
                calls.append(grid)
                return super().row_logits(rows)

        ic, pol = self.policy(Plain(plain_params()))
        pol._decide(view, decision, None, "route", 0.5, pol.prior)  # noqa: SLF001
        self.assertEqual([None], calls)


class MergedStudentTests(unittest.TestCase):
    """One student file that serves each kind with its own net (D59 lane G): the first net's kinds and the kinds kept from another student.
    Nothing is stored under a plain array name, so a build that does not know the format refuses the file instead of playing one net for
    every kind."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.a = self.dir / "a.npz"
        self.b = self.dir / "b.npz"
        hpi.write_student(self.a, plain_params(4), kinds=("card", "rest", "shop", "reward", "route"), prior_sha256="aa" * 32, fit={"who": "a"})
        hpi.write_student(self.b, plain_params(7), kinds=("route", "shop", "event"), prior_sha256="bb" * 32, fit={"who": "b"})
        self.m = self.dir / "m.npz"
        self.sha = hpi.write_merged_student(self.m, [(self.a, ("card", "rest", "shop", "reward")), (self.b, ("route", "event"))])

    @staticmethod
    def inputs(n: int = 6, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, hpi.DENSE_WIDTH)).astype(np.float32), rng.integers(0, 3, (n, hpi.E_SIZE + hpi.F_SIZE + hpi.G_SIZE)).astype(np.float32)

    def test_the_file_names_every_kind_once_and_stores_each_net_under_its_own_prefix(self):
        self.assertEqual(hpi.file_sha256(self.m), self.sha)
        self.assertEqual(("card", "rest", "shop", "reward", "route", "event"), hpi.student_kinds_of(self.m))
        with np.load(self.m, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            self.assertFalse([k for k in z.files if k in hpi._KEYS], "no array under a plain name")  # noqa: SLF001
            for key in hpi._KEYS:  # noqa: SLF001
                np.testing.assert_array_equal(z[f"net0.{key}"], plain_params(4)[key])
                np.testing.assert_array_equal(z[f"net1.{key}"], plain_params(7)[key])
        nets = meta["student"]["nets"]
        self.assertEqual([["card", "rest", "shop", "reward"], ["route", "event"]], [n["kinds"] for n in nets])
        self.assertEqual([hpi.file_sha256(self.a), hpi.file_sha256(self.b)], [n["source_sha256"] for n in nets])
        self.assertEqual("aa" * 32, meta["student"]["prior_sha256"])  # the first net's own record

    def test_each_kind_is_scored_by_its_own_net_and_a_plain_student_by_itself(self):
        s = hpi.load_student(self.m)
        dense, counts = self.inputs()
        a, b = hpi.load_student(self.a), hpi.load_student(self.b)
        for kind, want in (("card", a), ("rest", a), ("shop", a), ("reward", a), ("route", b), ("event", b)):
            np.testing.assert_allclose(s.scorer_for(kind).logits(dense, counts), want.logits(dense, counts), atol=1e-6, err_msg=kind)
        self.assertFalse(np.allclose(s.scorer_for("card").logits(dense, counts), s.scorer_for("route").logits(dense, counts)))
        self.assertTrue(all(s.has(k) for k in ("card", "route", "event")) and not s.has("select"))
        self.assertIs(a, a.scorer_for("card"))  # a plain student is its own scorer for every kind it names
        self.assertEqual(self.sha, s.sha256)

    def test_the_first_net_is_what_a_reader_of_a_plain_prior_gets(self):
        dense, counts = self.inputs()
        np.testing.assert_allclose(hpi.PriorScorer.load(self.m).logits(dense, counts), hpi.load_student(self.a).logits(dense, counts), atol=1e-6)
        self.assertEqual(hpi.PriorScorer.load(self.a).p.keys(), hpi.PriorScorer.load(self.m).p.keys())

    def test_each_net_keeps_its_own_map_tower_and_flags(self):
        tower = self.dir / "tower.npz"
        hpi.write_student(tower, tower_params(), kinds=("route", "event"), prior_sha256="cc" * 32)
        m = self.dir / "mt.npz"
        hpi.write_merged_student(m, [(self.a, ("card", "shop")), (tower, ("route",))])
        s = hpi.load_student(m)
        self.assertEqual((False, True), (s.scorer_for("card").map_grid, s.scorer_for("route").map_grid))
        self.assertEqual(("card", "shop", "route"), hpi.student_kinds_of(m))
        dense, counts = self.inputs()
        with self.assertRaisesRegex(ValueError, "map tower"):  # the kept net still refuses to score without its grids
            s.scorer_for("route").logits(dense, counts)

    def test_what_cannot_be_merged_is_refused_with_the_reason(self):
        out = self.dir / "x.npz"
        cases = [([(self.a, ("card", "shop")), (self.b, ("shop", "event"))], "more than one net"),
                 ([(self.a, ("route",)), (self.b, ("route",))], "more than one net"),
                 ([(self.a, ("event",)), (self.b, ("route",))], "was not fitted for"),
                 ([(self.a, ("card",)), (self.b, ())], "at least one kind"),
                 ([(self.a, ("card",))], "two or more"),
                 ([(self.m, ("card",)), (self.b, ("route",))], "merged student")]
        for parts, why in cases:
            with self.assertRaisesRegex(ValueError, why, msg=str(parts)):
                hpi.write_merged_student(out, parts)
            self.assertFalse(out.exists())
        plain_prior = self.dir / "prior.npz"
        np.savez(plain_prior, meta=np.array(json.dumps({})), **plain_params())
        with self.assertRaisesRegex(ValueError, "not a run student"):
            hpi.write_merged_student(out, [(self.a, ("card",)), (plain_prior, ("route",))])

    def test_a_file_whose_nets_name_a_kind_twice_does_not_load(self):
        with np.load(self.m, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"]))
        meta["student"]["nets"][1]["kinds"] = ["route", "card"]
        bad = self.dir / "bad.npz"
        np.savez(bad, meta=np.array(json.dumps(meta)), **arrays)
        with self.assertRaisesRegex(ValueError, "more than one net"):
            hpi.StudentScorer.load(bad)


if __name__ == "__main__":
    unittest.main()
