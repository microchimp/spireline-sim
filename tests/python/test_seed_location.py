"""The loop's generated E-blocks live under the lab folder (`LAB/seeds`), never in the checkout.

The presort's refset score refuses a tree with uncommitted or untracked files, and the train job used to write its block seed
files into `<tree>/referee/seeds`. These tests pin the new rule: a generated block is written beside the lab, the tree stays
clean, and the disjointness checks still see the committed blocks and the lab's generated ones, whichever folder holds them.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))

import gen0  # noqa: E402
import gen_loop  # noqa: E402

GIT = shutil.which("git")


def git(tree: Path, *args: str) -> str:
    cmd = [GIT, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false", *args]
    return subprocess.run(cmd, cwd=tree, check=True, capture_output=True, text=True).stdout


def write_block(path: Path, seeds: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seeds": seeds}), encoding="utf-8")


@unittest.skipUnless(GIT, "needs git")
class GeneratedBlocksLeaveTheTreeCleanTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.tree = self.root / "tree"  # a checkout whose committed folder holds E1
        self.committed = self.tree / "referee" / "seeds"
        write_block(self.committed / "E1-2026-10-23.json", ["AAAAAAAAAA"])
        git(self.tree, "init", "-q")
        git(self.tree, "add", "-A")
        git(self.tree, "commit", "-q", "-m", "committed blocks")
        self.lab = self.root / "lab"
        self.tpl = str(gen_loop.lab_seed_dir(self.lab) / "E{k}-{date}.json")
        (self.root / "agent").mkdir()
        self.tsv = self.root / "agent" / "promotions.tsv"

    def status(self) -> str:
        return git(self.tree, "status", "--porcelain", "--untracked-files=all")

    def test_a_new_block_is_written_under_the_lab_and_the_tree_stays_clean(self):
        p = gen_loop.resolve_e_block(self.tpl, 2, n=20, tsv=self.tsv, cand_sha="aaa", also_dirs=[self.committed])
        self.assertEqual(p.parent, self.lab / "seeds")
        self.assertTrue(p.is_file())
        self.assertEqual(len(gen0.load_seeds(p)), 20)
        self.assertEqual(self.status(), "")  # what the refset score needs
        # the same block again (nothing judged it): reused, not made twice
        self.assertEqual(gen_loop.resolve_e_block(self.tpl, 2, n=20, tsv=self.tsv, cand_sha="aaa", also_dirs=[self.committed]), p)

    def test_a_committed_block_is_reused_from_the_tree_and_a_fresh_r_block_goes_to_the_lab(self):
        kw = {"tsv": self.tsv, "also_dirs": [self.committed]}
        committed = gen_loop.resolve_e_block(self.tpl, 1, n=20, cand_sha="aaa", **kw)
        self.assertEqual(committed, self.committed / "E1-2026-10-23.json")  # unused: used as it is, nothing is created
        self.assertFalse((self.lab / "seeds").exists())
        gen_loop.append_promotion(self.tsv, {"seed_block": "E1-2026-10-23", "cand_sha": "aaa", "decision": "rejected"})
        r1 = gen_loop.resolve_e_block(self.tpl, 1, n=20, cand_sha="bbb", **kw)  # another candidate: a fresh block
        self.assertEqual(r1.parent, self.lab / "seeds")
        self.assertRegex(r1.name, r"^E1-\d{4}-\d\d-\d\d-r1\.json$")
        self.assertEqual(self.status(), "")
        gen_loop.append_promotion(self.tsv, {"seed_block": r1.stem, "cand_sha": "bbb", "decision": "rejected"})
        r2 = gen_loop.resolve_e_block(self.tpl, 1, n=20, cand_sha="ccc", **kw)
        self.assertTrue(r2.name.endswith("-r2.json"))  # the next free name, counting the lab's blocks
        self.assertEqual(self.status(), "")

    def test_the_train_command_hands_the_promotion_a_block_under_the_lab_and_the_trainer_the_lab_folder(self):
        marker_dir = self.lab / "train"
        marker_dir.mkdir(parents=True)
        cand = self.root / "agent" / "gen1" / "candidate"
        cand.mkdir(parents=True)
        (cand / "model.json").write_text(json.dumps({"policy_sha256": "aaa"}), encoding="utf-8")
        before = sorted(p.name for p in gen_loop.COMMITTED_SEEDS.glob("*.json"))
        argv = ["train", "--marker-dir", str(marker_dir), "--shards-root", str(self.root), "--agent-root", str(self.root / "agent"),
                "--promote", "--emu-python", "emu-python", "--seeds-file", self.tpl, "--lab", str(self.lab)]
        for gen, want_in_lab in ((0, False), (1, True)):  # generation 1's E-block is committed; generation 2's is made
            (marker_dir / f"ready-{gen}").write_text("x")
            with mock.patch.object(gen_loop, "train_generation", return_value=cand) as tg, \
                    mock.patch.object(gen_loop, "run_promote_subprocess", return_value=0) as rp:
                self.assertEqual(gen0.main(argv), 0)
            self.assertEqual([Path(p) for p in tg.call_args[1]["seed_dirs"]], [self.lab / "seeds"])
            block = Path(rp.call_args[0][3])
            self.assertEqual(block.parent == self.lab / "seeds", want_in_lab, block)
            self.assertTrue(block.is_file())
            (marker_dir / f"ready-{gen}").unlink()
        self.assertEqual(sorted(p.name for p in gen_loop.COMMITTED_SEEDS.glob("*.json")), before)  # the checkout gained nothing


class KnownSeedSetsReadBothFoldersTests(unittest.TestCase):
    def test_the_union_of_the_committed_and_the_lab_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            lab_seeds = Path(d) / "seeds"
            write_block(lab_seeds / "E7-2026-10-25.json", ["QQQQQQQQQQ"])
            write_block(lab_seeds / "E7-2026-10-25-r1.json", ["RRRRRRRRRR"])
            sets = gen_loop.known_seed_sets(lab_seeds, None, "E7-2026-10-25")  # the block being judged is not its own rival
            self.assertIn("E0-2026-10-23", sets)  # committed
            self.assertEqual(sets["E7-2026-10-25-r1"], ["RRRRRRRRRR"])  # the lab's
            self.assertNotIn("E7-2026-10-25", sets)
            # a caller that names the committed folder as its own (the trainer) adds the lab's folder
            both = gen_loop.known_seed_sets(gen_loop.COMMITTED_SEEDS, None, "", [lab_seeds])
            self.assertEqual({"E0-2026-10-23", "E1-2026-10-23", "E7-2026-10-25", "E7-2026-10-25-r1"} - set(both), set())

    def test_a_promotion_in_the_lab_folder_is_disjoint_from_the_committed_blocks(self):
        shared = gen0.load_seeds(gen_loop.COMMITTED_SEEDS / "E0-2026-10-23.json")[0]
        with tempfile.TemporaryDirectory() as d:
            block = Path(d) / "lab" / "seeds" / "E5-2026-10-25.json"
            write_block(block, [shared, "ZZZZZZZZZZ"])
            with self.assertRaisesRegex(ValueError, "also in E0-2026-10-23"):
                gen_loop.promote(Path(d) / "cand", None, block, agent_root=Path(d) / "agent", lab=Path(d) / "lab")

    def test_a_promotion_sees_a_lab_block_from_another_folder(self):
        with tempfile.TemporaryDirectory() as d:
            lab = Path(d) / "lab"
            write_block(lab / "seeds" / "E4-2026-10-24.json", ["QQQQQQQQQQ", "WWWWWWWWWW"])
            block = Path(d) / "elsewhere" / "E5-2026-10-25.json"
            write_block(block, ["QQQQQQQQQQ", "ZZZZZZZZZZ"])
            with self.assertRaisesRegex(ValueError, "also in E4-2026-10-24"):
                gen_loop.promote(Path(d) / "cand", None, block, agent_root=Path(d) / "agent", lab=lab)

    def test_the_trainer_refuses_a_window_that_holds_a_seed_of_a_lab_block(self):
        import numpy as np
        from tests.python.test_gen_loop import fake_record, fake_run_arrays  # noqa: PLC0415

        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as d:
            write_block(Path(d) / "lab" / "seeds" / "E4-2026-10-24.json", ["QQQQQQQQQQ"])
            sink = gen_loop.ShardSink(Path(d) / "runs" / "job", 0, config="a3-v1", base_sims=30, ascension=10, agent={}, part_runs=2)
            sink.add(fake_record("QQQQQQQQQQ", 0), fake_run_arrays(rng))
            sink.close()
            with self.assertRaisesRegex(ValueError, "also in E4-2026-10-24"):
                gen_loop.train_generation(0, Path(d) / "runs", Path(d) / "agent", device="cpu", epochs=1, vrun=False, export=False,
                                          seed_dirs=[Path(d) / "lab" / "seeds"])
            self.assertFalse((Path(d) / "agent" / "gen1").exists())


if __name__ == "__main__":
    unittest.main()
