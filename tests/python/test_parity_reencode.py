"""`rl/parity_reencode.py`: `rows_parity.npz` for the shards of a teacher store (D59 lane D).

A synthetic store (one shard whose rows are the real base vectors of a real view's options): a run the teacher played is built from its record (no game), the blocks equal
`option_parity` of the screen, a row that does not match the stored base is written as zeros and unknown (nothing is refused), the manifest names the blocks, the result is
idempotent and resumes, and the other files of the shard are untouched.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "rl", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import parity_blocks as pb  # noqa: E402
import parity_reencode as pre  # noqa: E402
from run_train import features, vocab  # noqa: E402
from run_train.features import CardRef, Decision, MapNode, Option, RunView  # noqa: E402

M, E, R, S, B, A = vocab.NODE_MONSTER, vocab.NODE_ELITE, vocab.NODE_REST, vocab.NODE_SHOP, vocab.NODE_BOSS, vocab.NODE_ANCIENT
MAP = (MapNode(3, 0, A, ((2, 1),)), MapNode(2, 1, M, ((2, 2),)), MapNode(2, 2, E, ((3, 3),)), MapNode(3, 3, B, ()))


def cid(name):
    return vocab.card_id_by_name(name)


def make_view():
    return RunView(ascension=10, act_index=0, act_region=vocab.REGION_OVERGROWTH, floor=3, hp=50, max_hp=80, gold=120,
                   deck=(CardRef(472, False), CardRef(131, False), CardRef(30, False)), relics=(vocab.relic_id_by_name("BurningBlood"),), potions=(),
                   max_potion_slots=3, map_nodes=MAP, position=(3, 0), boss=83)


def make_decisions(view):
    d0 = Decision("card_reward", (Option("card.take", 0, card=CardRef(cid("Whirlwind"), False)), Option("card.take", 1, card=CardRef(cid("Inflame"), False)), Option("card.skip", 2)))
    d1 = Decision("neow", (Option("neow.relic", 0, relic=vocab.relic_id_by_name("Lantern")), Option("neow.relic", 1, relic=vocab.relic_id_by_name("Vajra"))))
    return [d0, d1]


class ReencodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.gen = self.tmp / "14"
        self.shard = self.gen / "shards" / "J1"
        self.shard.mkdir(parents=True)
        (self.gen / "decisions.d").mkdir()
        self.view = make_view()
        self.decisions = make_decisions(self.view)
        rows, dec, opt = [], [], []
        for line, d in enumerate(self.decisions):
            for i, o in enumerate(d.options):
                base = features.encode_afterstate(self.view, o)
                rows.append(np.concatenate([base, np.zeros(8, dtype=np.float32)]))
                dec.append(line)
                opt.append(i)
        self.rows = np.stack(rows).astype(np.float32)
        np.savez_compressed(self.shard / "rows.npz", rows=self.rows, decision=np.array(dec, dtype=np.int32), option=np.array(opt, dtype=np.int16))
        (self.shard / "decisions.jsonl").write_text("\n".join(json.dumps({"seed": "AAA", "ordinal": i, "row0": 0}) for i in range(len(self.decisions))) + "\n", encoding="utf-8")
        (self.shard / "manifest.json").write_text(json.dumps({"version": 1, "agent_sha256": "ab" * 32, "map_grid": True}), encoding="utf-8")
        (self.gen / "runs.jsonl").write_text(json.dumps({"seed": "AAA", "source": "new", "ix": 3}) + "\n", encoding="utf-8")
        (self.gen / "decisions.d" / "AAA.record.json").write_text(json.dumps({"decisions": [{"i": 0}, {"i": 1}], "maps": []}), encoding="utf-8")
        self.patches = [mock.patch("run_train.shards.state_view", lambda rec, entry: self.view),
                        mock.patch("run_train.shards.entry_decision", lambda entry: self.decisions[int(entry["i"])])]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def run_cmd(self, *extra):
        import run_teacher as rt

        a = SimpleNamespace(store=str(self.tmp), generation=14, agent_root="", agent="", replay_agent="", lib="", runs_root="", stored_config=rt.STORED_CONFIG, procs=1, shard=[],
                            limit_runs=0)
        for k, v in extra:
            setattr(a, k, v)
        return pre.reencode_cmd(a)

    def test_a_shard_gets_its_blocks_equal_to_option_parity(self):
        self.assertEqual(0, self.run_cmd())
        self.assertTrue(pre.shard_has_parity(self.shard))
        z = np.load(self.shard / "rows_parity.npz")
        self.assertEqual(set(z.files), {"known", *pb.BLOCK_NAMES})
        self.assertTrue(bool(z["known"].all()))
        at = 0
        for d in self.decisions:
            want = pb.option_parity(self.view, list(d.options))
            for name in pb.BLOCK_NAMES:
                np.testing.assert_array_equal(z[name][at:at + len(d.options)], want[name])
            at += len(d.options)
        self.assertEqual(z["relics"].shape, (5, pb.WIDTHS["relics"]))
        doc = json.loads((self.shard / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(doc["parity"])
        self.assertEqual(doc["parity_blocks"], list(pb.BLOCK_NAMES))
        self.assertEqual((doc["parity_rows"], doc["parity_unknown_rows"]), (5, 0))
        self.assertTrue(doc["map_grid"])  # the other manifest keys stay

    def test_the_other_files_of_the_shard_are_untouched(self):
        before = {p.name: p.read_bytes() for p in self.shard.iterdir() if p.name != "manifest.json"}
        self.run_cmd()
        after = {p.name: p.read_bytes() for p in self.shard.iterdir() if p.name not in ("manifest.json", "rows_parity.npz")}
        self.assertEqual(before, after)

    def test_a_second_run_does_nothing_and_a_resume_reads_the_results(self):
        self.run_cmd()
        first = (self.shard / "rows_parity.npz").read_bytes()
        self.assertEqual(0, self.run_cmd())
        self.assertEqual(first, (self.shard / "rows_parity.npz").read_bytes())
        # an interrupted run: the shard lost its file but the run's result stays; the record is gone, yet the shard is built from the result
        (self.shard / "rows_parity.npz").unlink()
        (self.gen / "decisions.d" / "AAA.record.json").unlink()
        self.assertEqual(0, self.run_cmd())
        z = np.load(self.shard / "rows_parity.npz")
        self.assertTrue(bool(z["known"].all()))

    def test_a_row_that_does_not_match_is_unknown_and_zero_and_nothing_is_refused(self):
        rows = self.rows.copy()
        rows[3, 7] += 1.0  # the second screen's first row: its base vector is not what the record gives
        with np.load(self.shard / "rows.npz") as z:
            np.savez_compressed(self.shard / "rows.npz", rows=rows, decision=z["decision"], option=z["option"])
        self.assertEqual(0, self.run_cmd())
        z = np.load(self.shard / "rows_parity.npz")
        np.testing.assert_array_equal(z["known"], [True, True, True, False, False])  # the whole screen: its rows come from one replay
        self.assertEqual(float(np.abs(z["relics"][3:]).sum()), 0.0)
        doc = json.loads((self.shard / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["parity_unknown_rows"], 2)

    def test_limit_runs_builds_results_but_no_shard_file(self):
        self.assertEqual(0, self.run_cmd(("limit_runs", 1)))
        self.assertFalse((self.shard / "rows_parity.npz").exists())
        self.assertTrue((self.gen / "reencode.d" / "AAA.parity.npz").exists())
        self.assertFalse(pre.shard_has_parity(self.shard))

    def test_a_run_with_no_record_is_unknown(self):
        (self.gen / "decisions.d" / "AAA.record.json").unlink()
        self.assertEqual(0, self.run_cmd())
        z = np.load(self.shard / "rows_parity.npz")
        self.assertFalse(bool(z["known"].any()))

    def test_a_held_lock_stops_a_second_run(self):
        (self.gen / pre.LOCK_NAME).write_text(json.dumps({"pid": 4}), encoding="utf-8")
        with mock.patch("tools.lab.winsys.parent_alive", return_value=True):
            self.assertEqual(2, self.run_cmd())
        self.assertFalse(pre.shard_has_parity(self.shard))


if __name__ == "__main__":
    unittest.main()
