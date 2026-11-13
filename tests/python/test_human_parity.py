"""`rl/human_parity.py`: the held halves of the parity blocks for the human shards (D59 lane D, follow-up).

A synthetic source folder of shards shaped like `human-prior/decisions-v3` (counts, ident, run_ids, ...): the output equals `held_from_counts` of the shard's counts, one output file per shard
in a separate folder (the source is never written), the output holds arrays only (no run ids), a second run does nothing and a resume skips what is finished, a shard with a run of the exam fifth is
refused (the fixed hash rule), a shard of the wrong width is refused, and the manifest counts the rows.
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

import human_parity as hp  # noqa: E402
import human_starts  # noqa: E402
import parity_blocks as pb  # noqa: E402
import parity_potions as pp  # noqa: E402
import parity_relics as pr  # noqa: E402
from run_train import features  # noqa: E402

WIDTH = features.COUNTS_END - features.E_OFFSET


def run_id_in(bucket_is_exam: bool, start: int = 0) -> str:
    i = start
    while True:
        rid = f"synthetic-run-{i}"
        if (human_starts.exam_bucket(rid) == 0) == bucket_is_exam:
            return rid
        i += 1


def make_source(root: Path, shards=(5, 7, 3), exam: bool = False, width: int = WIDTH) -> Path:
    src = root / "decisions-v3"
    src.mkdir()
    rng = np.random.default_rng(1)
    for k, n in enumerate(shards):
        counts = np.zeros((n, max(width, WIDTH)), dtype=np.uint8)[:, :width] if width >= WIDTH else None
        if counts is None:
            np.savez_compressed(src / f"shard-{k:05d}.npz", counts=np.zeros((n, width), dtype=np.uint8), ident=np.arange(n, dtype=np.int32), run_ids=np.array([run_id_in(False, k)]))
            continue
        counts[:, : features.E_SIZE] = (rng.random((n, features.E_SIZE)) < 0.01)
        counts[:, features.F_OFFSET - features.E_OFFSET: features.G_OFFSET - features.E_OFFSET] = (rng.random((n, features.F_SIZE)) < 0.03)
        counts[:, features.G_OFFSET - features.E_OFFSET: features.G_OFFSET - features.E_OFFSET + features.G_SIZE] = (rng.random((n, features.G_SIZE)) < 0.03)
        runs = np.array([run_id_in(exam and k == 1, 100 * k + j) for j in range(2)])
        np.savez_compressed(src / f"shard-{k:05d}.npz", counts=counts, ident=np.arange(n, dtype=np.int32), run_ids=runs, dense=rng.random((n, 113)).astype(np.float32))
    (src / "counts.json").write_text(json.dumps({"rows": sum(shards)}), encoding="utf-8")
    return src


class ReencodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.src = make_source(self.tmp)
        self.out = self.tmp / "decisions-v3-parity"

    def test_the_output_equals_held_from_counts_shard_by_shard(self):
        res = hp.reencode(self.src, self.out)
        self.assertEqual((res["shards"], res["rows"], res["skipped"]), (3, 15, 0))
        for k in range(3):
            want = pb.held_from_counts(np.load(self.src / f"shard-{k:05d}.npz")["counts"])
            z = np.load(self.out / f"shard-{k:05d}.rows_parity.npz")
            self.assertEqual(set(z.files), {"relics_held", "roles_deck", "potions_held"})
            np.testing.assert_array_equal(z["relics_held"], want["relics"])
            np.testing.assert_array_equal(z["roles_deck"], want["roles"])
            np.testing.assert_array_equal(z["potions_held"], want["potions"])
            self.assertEqual(z["relics_held"].shape[1], pr.RELIC_TAG_WIDTH)
            self.assertEqual(z["roles_deck"].shape[1], 10)
            self.assertEqual(z["potions_held"].shape[1], pp.POTION_TAG_WIDTH)
            self.assertEqual(z["relics_held"].dtype, np.float32)

    def test_held_from_counts_equals_held_from_base(self):
        rng = np.random.default_rng(2)
        base = np.zeros((6, features.BASE_SIZE), dtype=np.float32)
        base[:, features.E_OFFSET:features.COUNTS_END] = (rng.random((6, WIDTH)) < 0.02)
        a = pb.held_from_base(base)
        b = pb.held_from_counts(base[:, features.E_OFFSET:features.COUNTS_END].astype(np.uint8))
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])

    def test_the_source_is_never_written_and_the_output_holds_no_run_id(self):
        before = {p.name: p.read_bytes() for p in self.src.iterdir()}
        hp.reencode(self.src, self.out)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.src.iterdir()})
        self.assertFalse(any(p.parent == self.src for p in self.out.iterdir()))
        for p in self.out.glob("*.npz"):
            self.assertFalse(np.load(p).files == [] or "run_ids" in np.load(p).files)
        self.assertNotIn("synthetic-run", "".join(q.read_text(encoding="utf-8") for q in self.out.glob("*.json")))

    def test_a_second_run_does_nothing_and_a_resume_skips_what_is_finished(self):
        hp.reencode(self.src, self.out)
        first = {p.name: p.read_bytes() for p in self.out.glob("*.npz")}
        res = hp.reencode(self.src, self.out)
        self.assertEqual((res["shards"], res["skipped"]), (3, 3))
        self.assertEqual(first, {p.name: p.read_bytes() for p in self.out.glob("*.npz")})
        (self.out / "shard-00001.rows_parity.npz").unlink()
        res = hp.reencode(self.src, self.out)
        self.assertEqual((res["skipped"], res["rows"]), (2, 15))
        self.assertTrue((self.out / "shard-00001.rows_parity.npz").is_file())

    def test_a_file_of_the_wrong_length_is_built_again(self):
        hp.reencode(self.src, self.out)
        np.savez_compressed(self.out / "shard-00000.rows_parity.npz", relics_held=np.zeros((2, pr.RELIC_TAG_WIDTH), dtype=np.float32),
                            roles_deck=np.zeros((2, 10), dtype=np.float32), potions_held=np.zeros((2, pp.POTION_TAG_WIDTH), dtype=np.float32))
        res = hp.reencode(self.src, self.out)
        self.assertEqual(res["skipped"], 2)
        self.assertEqual(np.load(self.out / "shard-00000.rows_parity.npz")["relics_held"].shape[0], 5)

    def test_the_manifest_counts_the_rows(self):
        hp.reencode(self.src, self.out)
        doc = json.loads((self.out / "parity-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((doc["shards"], doc["rows"], doc["parity_version"]), (3, 15, pb.PARITY_VERSION))
        self.assertEqual(doc["blocks"], {"relics_held": pr.RELIC_TAG_WIDTH, "roles_deck": 10, "potions_held": pp.POTION_TAG_WIDTH})

    def test_a_shard_of_the_exam_fifth_is_refused_and_nothing_is_written_for_it(self):
        src = make_source(self.tmp / "x", exam=True) if (self.tmp / "x").mkdir() is None else None
        with self.assertRaisesRegex(AssertionError, "exam fifth"):
            hp.reencode(src, self.tmp / "out2")
        self.assertFalse((self.tmp / "out2" / "shard-00001.rows_parity.npz").exists())

    def test_a_shard_of_the_wrong_width_is_refused(self):
        src = make_source(self.tmp / "y", width=WIDTH - 1) if (self.tmp / "y").mkdir() is None else None
        with self.assertRaises(ValueError):
            hp.reencode(src, self.tmp / "out3")

    def test_the_source_and_the_output_may_not_be_the_same_folder(self):
        with self.assertRaises(ValueError):
            hp.reencode(self.src, self.src)


if __name__ == "__main__":
    unittest.main()
