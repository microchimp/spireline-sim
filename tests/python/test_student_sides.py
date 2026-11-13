"""`rl/student_sides.py`: the side blocks the student's loader can take, one flag per block, default off (D59 lane D, step 5).

Pinned: the blocks, their files, widths and dtypes; no flag means no array (the lab's recipe is unchanged); a flag reads exactly its block, aligned with the shard's rows; a shard that lacks the
file, or whose file has the wrong shape, dtype, version or a number that is not finite, is refused naming the shard (never a silent zero); the parity file's `known` mask; the columns concatenate in
the blocks' own order whatever the order the flags were given in; the widths; a weight matrix is widened with zero columns so the net starts as the plain one; the shard readers of
`run_student` still read a shard that carries the new files.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "rl", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import parity_blocks as pb  # noqa: E402
import student_sides as ss  # noqa: E402


def make_shard(root: Path, n=6, *, grid=True, gaps=True, parity=True, parity_version=None, bad=None) -> Path:
    d = root / "shard"
    d.mkdir()
    rng = np.random.default_rng(3)
    np.savez_compressed(d / "rows.npz", rows=rng.random((n, 1863)).astype(np.float32), decision=np.arange(n, dtype=np.int32) // 2, option=np.arange(n, dtype=np.int16) % 2)
    manifest = {"version": 1}
    if grid:
        g = (rng.random((n, ss.BLOCKS["grid"].width)) < 0.2).astype(np.int8)
        np.savez_compressed(d / "rows_map.npz", grid=g)
        manifest["map_grid"] = True
    if gaps:
        np.savez_compressed(d / "rows_gaps.npz", gaps=rng.random((n, ss.BLOCKS["gaps"].width)).astype(np.float32))
        manifest.update(state_gaps=True, gaps_version=1)
    if parity:
        arrays = {name: rng.random((n, pb.WIDTHS[name])).astype(np.float32) for name in pb.BLOCK_NAMES}
        arrays["known"] = np.array([True] * (n - 1) + [False])
        if bad == "shape":
            arrays["roles"] = arrays["roles"][:, :-1]
        if bad == "nan":
            arrays["relics"][0, 0] = np.nan
        if bad == "dtype":
            arrays["events"] = arrays["events"].astype(np.float64)
        np.savez_compressed(d / "rows_parity.npz", **arrays)
        manifest.update(parity=True, parity_version=parity_version or pb.PARITY_VERSION, parity_blocks=list(pb.BLOCK_NAMES))
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


class FlagTests(unittest.TestCase):
    def test_the_blocks(self):
        self.assertEqual(list(ss.BLOCKS), ["grid", "gaps", "parity-relics", "parity-roles", "parity-events", "parity-potions", "held-relics", "held-roles", "held-potions"])
        self.assertEqual([ss.BLOCKS[k].width for k in ("held-relics", "held-roles", "held-potions")], [45, 10, 34])  # the held halves (D59 lane F): no file, from a row's counts
        self.assertEqual(ss.BLOCKS["grid"].width, 1547)
        self.assertEqual(ss.BLOCKS["gaps"].width, 36)
        self.assertEqual(ss.BLOCKS["parity-relics"].width, pb.WIDTHS["relics"])
        self.assertEqual(ss.BLOCKS["parity-roles"].width, pb.WIDTHS["roles"])
        self.assertEqual(ss.BLOCKS["parity-events"].width, pb.WIDTHS["events"])
        self.assertEqual(ss.BLOCKS["parity-potions"].width, pb.WIDTHS["potions"])
        self.assertEqual(ss.BLOCKS["grid"].file, "rows_map.npz")
        self.assertEqual(ss.BLOCKS["gaps"].file, "rows_gaps.npz")
        self.assertEqual(ss.BLOCKS["parity-roles"].file, "rows_parity.npz")

    def test_default_is_off(self):
        self.assertEqual(ss.parse_flags(None), ())
        self.assertEqual(ss.parse_flags(""), ())
        self.assertEqual(ss.parse_flags([]), ())
        self.assertEqual(ss.DEFAULT_FLAGS, ())

    def test_flags_are_validated_and_ordered_canonically(self):
        self.assertEqual(ss.parse_flags("parity-roles,gaps"), ("gaps", "parity-roles"))
        self.assertEqual(ss.parse_flags(["parity-events", "grid", "grid"]), ("grid", "parity-events"))
        with self.assertRaises(ValueError):
            ss.parse_flags("parity-nonsense")

    def test_the_command_line_helper(self):
        import argparse

        ap = argparse.ArgumentParser()
        ss.add_arguments(ap)
        self.assertEqual(ss.parse_flags(ap.parse_args([]).side_block), ())
        self.assertEqual(ss.parse_flags(ap.parse_args(["--side-block", "gaps", "--side-block", "parity-relics"]).side_block), ("gaps", "parity-relics"))


class ReadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_no_flag_reads_nothing(self):
        d = make_shard(self.tmp)
        self.assertEqual(ss.read_sides(d, (), 6), {})

    def test_a_flag_reads_its_block_aligned_with_the_rows(self):
        d = make_shard(self.tmp)
        out = ss.read_sides(d, ("gaps", "parity-roles"), 6)
        self.assertEqual(set(out), {"gaps", "parity-roles"})
        self.assertEqual(out["gaps"].shape, (6, 36))
        np.testing.assert_array_equal(out["gaps"], np.load(d / "rows_gaps.npz")["gaps"])
        np.testing.assert_array_equal(out["parity-roles"], np.load(d / "rows_parity.npz")["roles"])
        self.assertEqual(out["parity-roles"].dtype, np.float32)

    def test_a_grid_is_read_as_int8_and_columns_are_float32(self):
        d = make_shard(self.tmp)
        out = ss.read_sides(d, ("grid",), 6)
        self.assertEqual(out["grid"].dtype, np.int8)
        cols = ss.columns({"grid": out["grid"], "gaps": np.zeros((6, 36), dtype=np.float32)}, ("grid", "gaps"))
        self.assertEqual(cols.shape, (6, 1547 + 36))
        self.assertEqual(cols.dtype, np.float32)

    def test_a_missing_file_is_refused_naming_the_shard(self):
        d = make_shard(self.tmp, gaps=False)
        with self.assertRaisesRegex(ss.SideError, "shard.*rows_gaps.npz"):
            ss.read_sides(d, ("gaps",), 6)
        d2 = self.tmp / "shard"
        self.assertEqual(ss.read_sides(d2, ("grid",), 6)["grid"].shape, (6, 1547))  # the blocks not asked for are not needed

    def test_wrong_shape_dtype_and_non_finite_are_refused(self):
        for bad, name in (("shape", "parity-roles"), ("nan", "parity-relics"), ("dtype", "parity-events")):
            d = make_shard(Path(tempfile.mkdtemp()), bad=bad)
            self.addCleanup(shutil.rmtree, d.parent, True)
            with self.assertRaises(ss.SideError, msg=bad):
                ss.read_sides(d, (name,), 6)

    def test_a_row_count_that_differs_is_refused(self):
        d = make_shard(self.tmp)
        with self.assertRaises(ss.SideError):
            ss.read_sides(d, ("gaps",), 7)

    def test_a_parity_file_of_another_version_is_refused(self):
        d = make_shard(self.tmp, parity_version=99)
        with self.assertRaisesRegex(ss.SideError, "version"):
            ss.read_sides(d, ("parity-relics",), 6)

    def test_a_shard_without_the_parity_manifest_keys_is_refused(self):
        d = make_shard(self.tmp)
        doc = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        doc.pop("parity")
        (d / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaises(ss.SideError):
            ss.read_sides(d, ("parity-relics",), 6)

    def test_known_rows(self):
        d = make_shard(self.tmp)
        k = ss.read_known(d, 6)
        self.assertEqual(k.dtype, bool)
        self.assertEqual(k.tolist(), [True] * 5 + [False])


class ColumnTests(unittest.TestCase):
    def test_columns_follow_the_blocks_order_whatever_the_flags_order(self):
        a = np.ones((3, ss.BLOCKS["gaps"].width), dtype=np.float32)
        b = np.full((3, ss.BLOCKS["parity-roles"].width), 2, dtype=np.float32)
        cols = ss.columns({"parity-roles": b, "gaps": a}, ss.parse_flags(["parity-roles", "gaps"]))
        self.assertEqual(cols.shape, (3, 36 + pb.WIDTHS["roles"]))
        self.assertTrue(np.all(cols[:, :36] == 1) and np.all(cols[:, 36:] == 2))

    def test_widths(self):
        self.assertEqual(ss.width(()), 0)
        self.assertEqual(ss.width(("gaps", "parity-events")), 36 + pb.WIDTHS["events"])
        self.assertEqual(ss.columns({}, ()), None)

    def test_select_takes_the_sets_rows(self):
        a = np.arange(12, dtype=np.float32).reshape(6, 2)
        s = ss.select({"gaps": a}, np.array([4, 0, 2]))
        np.testing.assert_array_equal(s["gaps"], a[[4, 0, 2]])

    def test_a_weight_matrix_is_widened_with_zero_columns(self):
        w = np.random.default_rng(0).normal(size=(256, 454)).astype(np.float32)
        wide = ss.widen_first_layer(w, 36 + 17)
        self.assertEqual(wide.shape, (256, 454 + 53))
        np.testing.assert_array_equal(wide[:, :454], w)
        self.assertTrue(np.all(wide[:, 454:] == 0))
        x = np.random.default_rng(1).normal(size=(5, 454)).astype(np.float32)
        extra = np.random.default_rng(2).normal(size=(5, 53)).astype(np.float32)
        np.testing.assert_allclose(np.concatenate([x, extra], 1) @ wide.T, x @ w.T, rtol=1e-4, atol=1e-3)  # the net starts as the plain one


class LoaderCompatibilityTests(unittest.TestCase):
    def test_run_students_reader_still_reads_a_shard_that_carries_the_new_files(self):
        try:
            import run_student as rs
        except ImportError:  # pragma: no cover
            self.skipTest("run_student is not importable here")
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        d = make_shard(tmp, grid=False, gaps=False)
        n = 6
        recs = [{"v": rs.VERSION, "seed": "A", "ordinal": i, "n_options": 2, "row0": 2 * i} for i in range(3)]
        (d / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
        sh = rs.read_shard(d)
        self.assertEqual(len(sh["records"]), 3)
        self.assertEqual(sh["rows"].shape, (n, 1863))
        # the shard loader does not read rows_parity.npz: the new module does, by flag
        self.assertEqual(ss.read_sides(d, ("parity-relics",), n)["parity-relics"].shape, (n, pb.WIDTHS["relics"]))


if __name__ == "__main__":
    unittest.main()
