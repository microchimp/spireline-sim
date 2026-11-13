"""The student loader's side-block hook (D59 lane F, step 1; `rl/student_sides.py`, `rl/run_student.py`, `rl/human_prior.py`, `rl/human_prior_infer.py`).

Pinned: no flag is the loader as it was (no array, no column); a flag reads its block row for row from the shard's side file (map grid, state gaps, the parity blocks) and a shard that
lacks the file, or holds a malformed one, is refused with the shard and the file named; the three `held-*` blocks (relic effect tags, deck roles, potion effect tags, from a row's counts)
need no file on any side, are the same function of the counts on the teacher's rows, the human rows and the rows at play, and a student that takes them is the prior logit for logit before
its first step; the student's file says which blocks it reads and a scorer refuses to score without a block it cannot make; a fit with the blocks moves the new columns, measures both legs
(the human shards' side columns for leg 2) and writes the blocks into the student file; the command line has `--side-block`, default none.

No game and no real data: synthetic shards, a planted teacher, random side files.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, ROOT / "tests" / "python"):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import human_prior as hp  # noqa: E402
import human_prior_infer as hpi  # noqa: E402
import parity_blocks as pb  # noqa: E402
import run_student as rs  # noqa: E402
import student_sides as ss  # noqa: E402
from run_train import models  # noqa: E402
from test_run_student import (  # noqa: E402
    Synth,
    add_rows_gaps,
    add_rows_map,
    human_dataset,
    human_dataset_v3,
    random_weights,
    run_ids_by_bucket,
    synth_shards,
    write_weights,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

HELD = ("held-relics", "held-roles", "held-potions")


def add_rows_parity(d: Path, seed: int = 17, bad: str | None = None) -> dict[str, np.ndarray]:
    """Write `rows_parity.npz` beside a shard's rows.npz (random blocks, `known` all true) and say so in its manifest."""
    with np.load(d / "rows.npz") as z:
        n = len(z["rows"])
    rng = np.random.default_rng(seed)
    arrays = {name: rng.random((n, pb.WIDTHS[name])).astype(np.float32) for name in pb.BLOCK_NAMES}
    arrays["known"] = np.ones(n, dtype=bool)
    if bad == "shape":
        arrays["relics"] = arrays["relics"][:, :-1]
    elif bad == "nan":
        arrays["relics"][0, 0] = np.nan
    np.savez_compressed(d / pb.SHARD_ROWS_PARITY, **arrays)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    man.update(parity=True, parity_version=pb.PARITY_VERSION, parity_blocks=list(pb.BLOCK_NAMES))
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return arrays


def widened(weights: dict[str, np.ndarray], extra: int, seed: int | None = None) -> dict[str, np.ndarray]:
    """The prior's weights with `extra` more first-layer columns: zero (the plain prior, logit for logit) or random with `seed`."""
    w = dict(weights)
    if seed is None:
        w["hidden1.weight"] = ss.widen_first_layer(weights["hidden1.weight"], extra)
    else:
        rng = np.random.default_rng(seed)
        w["hidden1.weight"] = np.concatenate([weights["hidden1.weight"], (rng.standard_normal((256, extra)) * 0.05).astype(np.float32)], axis=1)
    return w


def side_prior_file(path: Path, weights: dict[str, np.ndarray], flags) -> Path:
    np.savez(path, meta=np.array(json.dumps({"variant": "test-sides", "side_blocks": list(flags)})), **weights)
    return path


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.weights = random_weights()
        cls.prior_path = write_weights(cls.root / "pi_h.npz", cls.weights)
        cls.prior = hpi.PriorScorer.load(cls.prior_path)
        cls.synth = Synth(cls.weights, seed=23)
        cls.shards = synth_shards(cls.root, cls.synth, runs=40, kinds=("card", "route", "shop", "reward"), per_kind=2)
        cls.grid = add_rows_map(cls.shards)
        cls.gaps = add_rows_gaps(cls.shards, version=1)
        man = json.loads((cls.shards / "manifest.json").read_text(encoding="utf-8"))
        man["map_grid"] = True  # what the teacher's own writer says of the file (run_teacher.py reencode / build_shard)
        (cls.shards / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
        cls.parity = add_rows_parity(cls.shards)
        cls.bare = synth_shards(cls.root, cls.synth, runs=8, kinds=("card", "route"), per_kind=1, name="bare")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_no_flag_is_the_loader_as_it_was(self):
        ts = rs.load_shards(self.shards, prior=self.prior)
        self.assertIsNone(ts.sides)
        self.assertEqual((), ts.side_flags)
        self.assertEqual([], rs.public_stats(ts)["side_blocks"])
        again = rs.load_shards(self.shards, prior=self.prior, side_blocks=())
        np.testing.assert_array_equal(ts.dense, again.dense)
        np.testing.assert_array_equal(ts.q, again.q)
        self.assertIsNone(again.sides)

    def test_a_flag_loads_its_block_row_for_row_in_the_blocks_own_order(self):
        ts = rs.load_shards(self.shards, prior=self.prior, side_blocks=("parity-relics", "gaps", "grid"))
        self.assertEqual(("grid", "gaps", "parity-relics"), ts.side_flags)  # the blocks' own order, whatever the order given
        n = len(ts.q)
        self.assertEqual({"grid", "gaps", "parity-relics"}, set(ts.sides))
        for name, width in (("grid", 17 * 7 * 13), ("gaps", 36), ("parity-relics", pb.WIDTHS["relics"])):
            self.assertEqual((n, width), ts.sides[name].shape, name)
        first = int(ts.count[0])
        np.testing.assert_array_equal(ts.sides["gaps"][:first], self.gaps[:first])
        np.testing.assert_array_equal(ts.sides["grid"][:first], self.grid[:first])
        np.testing.assert_array_equal(ts.sides["parity-relics"][:first], self.parity["relics"][:first])
        self.assertEqual(["grid", "gaps", "parity-relics"], rs.public_stats(ts)["side_blocks"])
        cols = ss.columns(ts.sides, ts.side_flags)
        self.assertEqual((n, ss.width(ts.side_flags)), cols.shape)  # the row width the net's first layer is widened by
        self.assertEqual(np.float32, cols.dtype)

    def test_a_block_with_no_flag_is_not_read(self):
        ts = rs.load_shards(self.shards, prior=self.prior, side_blocks=("gaps",))
        self.assertEqual({"gaps"}, set(ts.sides))
        self.assertEqual(36, ss.width(ts.side_flags))

    def test_the_held_blocks_need_no_file_and_are_the_function_of_the_counts(self):
        ts = rs.load_shards(self.bare, prior=self.prior, side_blocks=HELD)
        self.assertEqual(HELD, ts.side_flags)
        want = pb.held_from_counts(ts.counts)
        np.testing.assert_array_equal(ts.sides["held-relics"], want["relics"])
        np.testing.assert_array_equal(ts.sides["held-roles"], want["roles"])
        np.testing.assert_array_equal(ts.sides["held-potions"], want["potions"])
        self.assertEqual((45, 10, 34), tuple(ts.sides[k].shape[1] for k in HELD))
        self.assertEqual(89, ss.width(HELD))

    def test_a_missing_side_file_is_refused_with_the_shard_and_the_file_named(self):
        for flags, name in ((("parity-relics",), "rows_parity.npz"), (("grid",), "rows_map.npz"), (("gaps",), "rows_gaps.npz")):
            with self.assertRaisesRegex(rs.ShardError, name) as ctx:
                rs.load_shards(self.bare, prior=self.prior, side_blocks=flags)
            self.assertIn("bare", str(ctx.exception))

    def test_a_malformed_side_file_is_refused_as_a_shard_error(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        for bad, why in (("shape", "relics"), ("nan", "not finite")):
            d = synth_shards(Path(scratch.name), self.synth, runs=3, kinds=("card",), per_kind=1, name=f"bad-{bad}")
            add_rows_parity(d, bad=bad)
            with self.assertRaisesRegex(rs.ShardError, why):
                rs.load_shards(d, prior=self.prior, side_blocks=("parity-relics",))
        d = synth_shards(Path(scratch.name), self.synth, runs=3, kinds=("card",), per_kind=1, name="nomanifest")
        add_rows_parity(d)
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        man.pop("parity")
        (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
        with self.assertRaisesRegex(rs.ShardError, "parity"):
            rs.load_shards(d, prior=self.prior, side_blocks=("parity-roles",))

    def test_an_unknown_flag_is_refused(self):
        with self.assertRaises(ValueError):
            rs.load_shards(self.shards, prior=self.prior, side_blocks=("nonsense",))

    def test_the_blocks_are_cut_with_the_decisions_the_loader_drops(self):
        # a decision the loader drops (a refused one) must take its rows out of the side blocks too: the arrays stay aligned with ts.dense
        d = Path(self.tmp.name) / "dropping"
        d.mkdir()
        recs, rows = [], []
        for i in range(6):
            r, rr = self.synth.record(f"S{i:08d}", 0, "card")
            recs.append(r)
            rows.append(rr)
        recs[1]["refused"] = "test"
        recs[1]["q"] = None
        from test_run_student import write_shard

        write_shard(d.parent, "dropping", recs, rows)
        add_rows_gaps(d, version=1)
        ts = rs.load_shards(d, prior=self.prior, side_blocks=("gaps",))
        self.assertEqual(5, len(ts))
        self.assertEqual(len(ts.dense), len(ts.sides["gaps"]))
        kept = np.concatenate([np.arange(a, a + 4) for a in (0, 8, 12, 16, 20)])
        full = np.load(d / "rows_gaps.npz")["gaps"]
        np.testing.assert_array_equal(ts.sides["gaps"], full[kept])


@unittest.skipIf(torch is None, "torch is not installed")
class NetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = random_weights()
        cls.rows = np.random.default_rng(1).normal(0, 0.4, (30, rs.ROWS_WIDTH)).astype(np.float32)
        cls.dense, cls.counts = models.value_inputs(cls.rows)

    def test_the_student_net_takes_the_side_columns_and_is_the_prior_logit_for_logit(self):
        flags = ("gaps", "held-relics")
        block = {"gaps": np.random.default_rng(2).random((30, 36)).astype(np.float32), "held-relics": pb.held_from_counts(self.counts)["relics"]}
        cols = ss.columns(block, flags)
        net = rs.make_net(widened(self.weights, ss.width(flags)), 0.0, False, side_blocks=flags).eval()
        plain = rs.make_net(self.weights, 0.0).eval()
        with torch.no_grad():
            got = net(torch.from_numpy(self.dense), torch.from_numpy(self.counts), sides=torch.from_numpy(cols)).numpy()
            want = plain(torch.from_numpy(self.dense), torch.from_numpy(self.counts)).numpy()
            with self.assertRaises(ValueError):
                net(torch.from_numpy(self.dense), torch.from_numpy(self.counts))  # such a net never scores without its columns
        np.testing.assert_allclose(got, want, atol=2e-5)  # zero columns: the plain prior
        self.assertEqual(set(self.weights), set(rs.net_params(net)))
        # random columns are read: another input, another logit; and the numpy scorer agrees with the net
        wide = widened(self.weights, ss.width(flags), seed=5)
        net2 = rs.make_net(wide, 0.0, False, side_blocks=flags).eval()
        scorer = hpi.PriorScorer(wide, {"side_blocks": list(flags)})
        with torch.no_grad():
            got2 = net2(torch.from_numpy(self.dense), torch.from_numpy(self.counts), sides=torch.from_numpy(cols)).numpy()
        np.testing.assert_allclose(got2, scorer.logits(self.dense, self.counts, sides=block), atol=2e-5)
        self.assertFalse(np.allclose(got2, want))


class ScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.weights = random_weights()
        cls.rows = np.random.default_rng(1).normal(0, 0.4, (12, rs.ROWS_WIDTH)).astype(np.float32)
        cls.dense, cls.counts = models.value_inputs(cls.rows)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_a_student_file_says_which_blocks_it_reads_and_a_plain_one_says_none(self):
        flags = ("gaps", "held-relics", "held-potions")
        wide = widened(self.weights, ss.width(flags), seed=7)
        path = self.root / "s.npz"
        hpi.write_student(path, wide, kinds=("card", "route"), prior_sha256="0" * 64, fit={"t": 1}, side_blocks=flags)
        with np.load(path) as z:
            self.assertEqual(list(flags), json.loads(str(z["meta"]))["side_blocks"])
        s = hpi.StudentScorer.load(path)
        self.assertEqual(flags, s.side_blocks)
        plain = self.root / "p.npz"
        hpi.write_student(plain, self.weights, kinds=("card",), prior_sha256="0" * 64)
        with np.load(plain) as z:
            self.assertNotIn("side_blocks", json.loads(str(z["meta"])))
        self.assertEqual((), hpi.StudentScorer.load(plain).side_blocks)

    def test_a_scorer_makes_the_held_blocks_itself_and_refuses_one_it_cannot_make(self):
        flags = ("held-relics", "held-roles")
        wide = widened(self.weights, ss.width(flags), seed=8)
        s = hpi.PriorScorer(wide, {"side_blocks": list(flags)})
        held = pb.held_from_counts(self.counts)
        by_hand = s.logits(self.dense, self.counts, sides={"held-relics": held["relics"], "held-roles": held["roles"]})
        np.testing.assert_allclose(s.logits(self.dense, self.counts), by_hand, atol=1e-6)  # from the counts alone
        np.testing.assert_allclose(s.row_logits(self.rows), by_hand, atol=1e-5)
        g = hpi.PriorScorer(widened(self.weights, 36, seed=9), {"side_blocks": ["gaps"]})
        with self.assertRaisesRegex(ValueError, "gaps"):
            g.logits(self.dense, self.counts)  # a file-backed block must be given
        with self.assertRaisesRegex(ValueError, "gaps"):
            g.row_logits(self.rows)

    def test_the_first_layer_must_be_the_width_the_meta_promises(self):
        with self.assertRaisesRegex(ValueError, "first layer"):
            hpi.PriorScorer(self.weights, {"side_blocks": ["held-relics"]})
        with self.assertRaisesRegex(ValueError, "first layer"):
            hpi.PriorScorer(widened(self.weights, 45, seed=1), {})
        with self.assertRaises(ValueError):
            hpi.PriorScorer(widened(self.weights, 45, seed=1), {"side_blocks": ["nonsense"]})

    def test_option_sides_gives_the_blocks_the_scorer_reads_in_the_blocks_order(self):
        held = pb.held_from_counts(self.counts)
        got = hpi.option_sides(None, [], ("held-roles", "held-relics"), self.rows)
        self.assertEqual((len(self.rows), 55), ss.columns(got, ("held-relics", "held-roles")).shape)
        np.testing.assert_array_equal(got["held-relics"], held["relics"])
        self.assertEqual({"held-relics", "held-roles"}, set(got))


class HumanSideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids = run_ids_by_bucket(30)

    def test_a_dataset_gives_the_side_columns_of_its_rows(self):
        ds = human_dataset_v3({1: self.ids[1][:6]}, grid=True)
        rows = np.arange(0, len(ds.dense), 7)
        got = ds.side_blocks(rows, ("grid", "gaps", "held-relics", "held-potions"))
        np.testing.assert_array_equal(got["grid"], ds.grid_rows(rows))
        np.testing.assert_array_equal(got["gaps"], ds.gaps_rows(rows))
        want = pb.held_from_counts(ds.counts[rows])
        np.testing.assert_array_equal(got["held-relics"], want["relics"])
        np.testing.assert_array_equal(got["held-potions"], want["potions"])
        with self.assertRaisesRegex(ValueError, "parity-relics"):
            ds.side_blocks(rows, ("parity-relics",))  # the human shards carry no file for the full parity blocks
        plain = human_dataset({1: self.ids[1][:4]})
        with self.assertRaisesRegex(ValueError, "gaps"):
            plain.side_blocks(np.arange(3), ("gaps",))
        self.assertEqual({}, plain.side_blocks(np.arange(3), ()))

    def test_row_scores_take_the_side_columns_and_load_human_checks_the_blocks_exist(self):
        flags = ("gaps", "held-roles")
        ds = human_dataset_v3({4: self.ids[4][:8]})
        wide = widened(random_weights(), ss.width(flags), seed=3)
        s = hpi.PriorScorer(wide, {"side_blocks": list(flags)})
        got = hp.row_scores_np(ds, None, s.logits, grid=False, gaps=False, sides=flags)
        side = ds.side_blocks(np.arange(len(ds.dense)), flags)
        x = hp.assemble_np_v1(ds.dense)
        want = s.logits(x, ds.counts.astype(np.float32), sides=side)
        np.testing.assert_allclose(got, want, atol=1e-5)
        with self.assertRaisesRegex(ValueError, "gaps"):
            hp.row_scores_np(human_dataset({4: self.ids[4][:4]}), None, s.logits, grid=False, gaps=False, sides=flags)
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        d = Path(scratch.name)
        plain = human_dataset({1: self.ids[1][:4]})
        np.savez(d / "shard-00000.npz", dec=plain.dec, dense=plain.dense, counts=plain.counts, role=plain.role, ident=plain.ident, run_ids=np.array(plain.run_ids))
        with self.assertRaisesRegex(rs.ShardError, "gaps"):
            rs.load_human(d, (1,), exam_ids=frozenset(), side_blocks=("gaps",))
        with self.assertRaisesRegex(rs.ShardError, "grid"):
            rs.load_human(d, (1,), exam_ids=frozenset(), side_blocks=("grid",))
        with self.assertRaisesRegex(rs.ShardError, "parity-relics"):
            rs.load_human(d, (1,), exam_ids=frozenset(), side_blocks=("parity-relics",))
        self.assertEqual(len(plain.dec), len(rs.load_human(d, (1,), exam_ids=frozenset(), side_blocks=HELD).dec))  # held blocks need nothing of the shards


@unittest.skipIf(torch is None, "torch is not installed")
class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.weights = random_weights()
        cls.prior_path = write_weights(cls.root / "pi_h.npz", cls.weights)
        cls.synth = Synth(cls.weights, seed=29)
        cls.shards = synth_shards(cls.root, cls.synth, runs=60, kinds=("card", "route", "shop", "reward"), per_kind=2)
        add_rows_gaps(cls.shards, version=1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_a_fit_with_side_blocks_moves_the_new_columns_runs_both_legs_and_says_so_in_the_file(self):
        flags = ("gaps", "held-relics", "held-potions")
        ids = run_ids_by_bucket(30)
        human = {1: human_dataset_v3({1: ids[1][:30]}, per_run=1), 4: human_dataset_v3({4: ids[4][:30]}, per_run=1)}
        calls = []

        def fake_load_human(d, fifths, **kw):
            calls.append(kw)
            return human[1 if fifths == rs.HUMAN_TRAIN_FIFTHS else 4]

        with mock.patch.object(rs, "BOOT_DRAWS", 100), mock.patch.object(rs, "LEG1_MIN_DECISIONS", 10), mock.patch.object(rs, "MIN_LAMBDA_DECISIONS", 10), \
                mock.patch.object(rs, "load_human", side_effect=fake_load_human):
            legs = rs.run_fit(self.root, self.prior_path, self.root / "out", human_dir=self.root, exam_ids=frozenset(), device="cpu", log=lambda *_: None,
                              cfg=rs.FitConfig(epochs=2, patience=2, lr=1e-3, beta=0.5, kl=0.05, batch_teacher=64, batch_human=16, dropout=0.0),
                              side_blocks=flags)
        self.assertTrue(all(tuple(c.get("side_blocks") or ()) == ss.parse_flags(flags) for c in calls))  # the human shards are read with the blocks
        out = self.root / "out"
        with np.load(out / rs.STUDENT_OUT) as z:
            self.assertEqual(list(ss.parse_flags(flags)), json.loads(str(z["meta"]))["side_blocks"])
            w = z["hidden1.weight"]
        self.assertEqual(454 + ss.width(flags), w.shape[1])
        self.assertGreater(float(np.abs(w[:, 454:]).max()), 0.0)  # the gradient reached the new columns
        student = hpi.StudentScorer.load(out / rs.STUDENT_OUT)
        self.assertEqual(ss.parse_flags(flags), student.side_blocks)
        self.assertEqual(hpi.file_sha256(self.prior_path), legs["baseline"]["sha256"])  # measured against the plain prior it started from
        fit_doc = json.loads((out / rs.FIT_FILE).read_text())
        self.assertEqual(list(ss.parse_flags(flags)), fit_doc["side_blocks"])
        self.assertIn("card", legs["leg1"]["kinds"])
        self.assertIn("card", legs["leg2"]["kinds"])  # leg 2 scored the student with the human rows' side columns

    def test_a_fit_without_blocks_writes_a_student_that_names_none(self):
        with mock.patch.object(rs, "BOOT_DRAWS", 50):
            rs.run_fit(self.root, self.prior_path, self.root / "plain", device="cpu", log=lambda *_: None,
                       cfg=rs.FitConfig(epochs=1, patience=1, batch_teacher=64, dropout=0.0))
        with np.load(self.root / "plain" / rs.STUDENT_OUT) as z:
            self.assertNotIn("side_blocks", json.loads(str(z["meta"])))
        self.assertEqual([], json.loads((self.root / "plain" / rs.FIT_FILE).read_text())["side_blocks"])

    def test_a_prior_that_already_reads_other_blocks_is_refused(self):
        wide = widened(self.weights, 45, seed=2)
        path = side_prior_file(self.root / "wide.npz", wide, ("held-relics",))
        with self.assertRaisesRegex(rs.ShardError, "side"):
            rs.run_fit(self.root, path, self.root / "x", device="cpu", log=lambda *_: None, side_blocks=("held-potions",))


class CommandLineTests(unittest.TestCase):
    def test_the_command_line_has_the_flag_and_defaults_to_none(self):
        p = rs.build_parser()
        a = p.parse_args(["fit", "--shards-root", "s", "--prior", "p", "--out", "o", "--side-block", "held-relics", "--side-block", "gaps,grid"])
        self.assertEqual(("grid", "gaps", "held-relics"), ss.parse_flags(a.side_block))
        for cmd in (["fit", "--shards-root", "s", "--prior", "p", "--out", "o"], ["check", "--shards-root", "s"],
                    ["round", "--shards-root", "s", "--prior", "p", "--out", "o", "--base", "b", "--agent-root", "r", "--target", "9"]):
            self.assertEqual((), ss.parse_flags(p.parse_args(cmd).side_block), cmd[0])


if __name__ == "__main__":
    unittest.main()
