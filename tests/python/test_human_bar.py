"""The human fight bar's diagnosis (tools/human_bar.py; D55 addendum 10).

The pinned facts: the plays log and the buffer's fight records are read and nothing under the loop's folder is written; a start's
figures are its latest `n_last` plays that have an outcome (a capped play or one with no value counts for nothing); a death costs the
whole HP going in on both sides; the table's means and paired differences are the arithmetic of those starts; the list of starts the
human survived and the player's latest plays all died in, and the list of starts the player lost 15 HP or more above the human in,
hold the right starts; the markdown carries no start id and hides a cell under 20 runs; the JSON carries the ids and figures.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "tools", ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import human_bar as hb  # noqa: E402
import practice_mix as pm  # noqa: E402
from test_practice_mix import _fake_shard  # noqa: E402

BOSS = "KAISER_CRAB_BOSS"
RARE = "THE_OBSCURA_BOSS"


def sid(i: int, prefix: str = "R") -> str:
    return f"human:{prefix}{i:03d}|f33"


def lite_rec(i: int, enemy: str = BOSS, prefix: str = "R", hp: int = 60) -> dict:
    return {"id": sid(i, prefix), "run": f"{prefix}{i:03d}", "act": 2, "kind": "boss", "enemy": enemy, "hp": hp, "cat": "act2-boss"}


def outcome(survived: bool = True, lost: int = 10) -> dict:
    return {"human_survived": survived, "human_hp_lost": lost if survived else 60, "human_end_hp": 60 - lost if survived else 0,
            "human_l0": float(60 - lost) if survived else -100.0}


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "loop"
        (self.root / "starts").mkdir(parents=True)
        (self.root / "buffer" / "shards").mkdir(parents=True)
        lite = [lite_rec(i) for i in range(30)] + [lite_rec(i, RARE, "Q") for i in range(5)]
        lite.append({**lite_rec(99), "id": "2026-11-01/g9:S099|f33"})  # a night start: not a human one
        outcomes = {r["id"]: outcome() for r in lite}
        outcomes[sid(2)] = outcome(False)  # the human died there
        outcomes[sid(3)] = outcome(True, 10)
        (self.root / "starts" / "human-starts.lite.json").write_text(json.dumps(lite), encoding="utf-8")
        (self.root / "starts" / "human-outcomes.json").write_text(json.dumps({"outcomes": outcomes}), encoding="utf-8")
        dead = lambda i, j=0: {"start": sid(i), "j": j, "root": 40.0, "won": False}  # noqa: E731
        win = lambda i, hp=50, j=0: {"start": sid(i), "j": j, "root": 40.0, "won": True, "end_hp": hp, "held": 0}  # noqa: E731
        first = [dead(0), dead(1), dead(2), dead(5, 0), dead(5, 1), dead(5, 2),
                 {"start": sid(6), "root": 40.0, "won": True, "end_hp": 20, "capped": True}]
        second = [dead(0, 1), dead(2, 1), win(3, 30), win(4, 45), win(5, 50, 3), win(5, 50, 4), win(5, 50, 5), win(5, 50, 6)]
        rest = [win(i) for i in range(7, 30)] + [{"start": sid(i, "Q"), "root": 40.0, "won": True, "end_hp": 50} for i in range(5)] \
            + [{"start": "2026-11-01/g9:S099|f33", "root": 1.0, "won": False}]
        _fake_shard(self.root / "buffer" / "shards" / "s000001-1.npz", first)
        _fake_shard(self.root / "buffer" / "shards" / "s000002-2.npz", second + rest)
        pm.sync_plays(self.root)  # the loop's own step writes the plays log; the diagnosis only reads it

    def tearDown(self):
        self.tmp.cleanup()

    def folder_state(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in sorted(self.root.rglob("*")) if p.is_file()}

    def test_only_reads_and_keeps_the_latest_plays_that_have_an_outcome(self):
        before = self.folder_state()
        doc, md = hb.diagnose(self.root, n_last=4, boot=40)
        self.assertEqual(self.folder_state(), before)  # nothing under the loop's folder was written
        self.assertEqual(doc["starts_played"], 34)  # 29 + 5 human starts; the capped play and the night start count for nothing
        self.assertEqual(doc["plays_used"], 34 + 5)  # the capped play and the night start's play are not used
        self.assertEqual(doc["plays_total"], 7 + 8 + 23 + 5 + 1)
        rows = {r["cell"]: r for r in doc["table"]["by_enemy"]}
        k = rows[f"act2 boss {BOSS}"]
        self.assertEqual((k["starts"], k["runs"], k["plays"]), (29, 29, 34))  # start 5 counts its latest four plays only

    def test_the_table_is_the_arithmetic_of_the_starts_with_a_death_costing_the_whole_hp(self):
        doc, _ = hb.diagnose(self.root, n_last=4, boot=40)
        k = {r["cell"]: r for r in doc["table"]["by_enemy"]}[f"act2 boss {BOSS}"]
        self.assertAlmostEqual(k["player_survival"], 26 / 29)  # starts 0, 1 and 2 died on every play
        self.assertAlmostEqual(k["human_survival"], 28 / 29)  # the human died on start 2
        self.assertAlmostEqual(k["survival_diff"], -2 / 29)
        self.assertAlmostEqual(k["player_hp_lost"], (60 + 60 + 60 + 30 + 15 + 10 + 23 * 10) / 29)
        self.assertAlmostEqual(k["human_hp_lost"], (60 + 28 * 10) / 29)
        self.assertAlmostEqual(k["hp_lost_diff"], (465 - 340) / 29)
        self.assertLessEqual(k["hp_lost_diff_lo"], k["hp_lost_diff"])
        self.assertGreaterEqual(k["hp_lost_diff_hi"], k["hp_lost_diff"])
        self.assertAlmostEqual(k["player_value"], (-100 * 3 + 30 + 45 + 50 + 23 * 50) / 29)
        self.assertAlmostEqual(k["human_value"], (-100 + 28 * 50) / 29)
        ak = {r["cell"]: r for r in doc["table"]["by_act_kind"]}["act2-boss"]
        self.assertEqual((ak["starts"], ak["plays"]), (34, 39))
        self.assertEqual(doc["table"]["all"]["starts"], 34)

    def test_the_two_lists(self):
        doc, _ = hb.diagnose(self.root, n_last=4, min_plays=1, boot=10)
        died = [r["start"] for r in doc["lists"]["human_survived_player_died"]]
        self.assertEqual(died, [sid(0), sid(1)])  # most plays first; start 2 is no entry (the human died there too), start 5 won its latest four
        lost = {r["start"]: r for r in doc["lists"]["player_lost_more"]}
        self.assertEqual(sorted(lost), [sid(0), sid(1), sid(3)])  # 50, 50 and 20 above the human's ten; start 4 is 5 above, start 2 level
        self.assertEqual(lost[sid(3)]["player_hp_lost"], 30.0)
        self.assertEqual(lost[sid(3)]["human_hp_lost"], 10)
        self.assertEqual([r["start"] for r in doc["lists"]["player_lost_more"]][-1], sid(3))  # the biggest gap first
        self.assertEqual(doc["counts"], {"human_survived_player_died": 2, "player_lost_more": 3})
        two, _ = hb.diagnose(self.root, n_last=4, min_plays=2, boot=10)
        self.assertEqual([r["start"] for r in two["lists"]["human_survived_player_died"]], [sid(0)])
        gap = hb.diagnose(self.root, n_last=4, gap_hp=40, boot=10)[0]
        self.assertEqual(sorted(r["start"] for r in gap["lists"]["player_lost_more"]), [sid(0), sid(1)])
        one = hb.diagnose(self.root, n_last=1, boot=10)[0]  # only the latest play of start 0 and 5 and the rest
        self.assertEqual([r["start"] for r in one["lists"]["human_survived_player_died"]], [sid(0), sid(1)])

    def test_the_markdown_has_aggregates_only_and_hides_a_cell_under_twenty_runs(self):
        doc, md = hb.diagnose(self.root, n_last=4, boot=10)
        self.assertNotIn("human:R", md)
        self.assertNotIn("human:Q", md)
        self.assertNotIn("|f33", md)
        self.assertNotIn("R00", md)
        self.assertNotIn("Q00", md)
        self.assertIn(f"| act2 boss {BOSS} | 29 | 34 |", md)
        self.assertNotIn(RARE, md)  # five runs: hidden in the markdown
        self.assertIn("1 further cells are under 20 runs and not shown.", md)
        self.assertIn("Spire Codex public run export for v0.107.1", md)
        self.assertIn("## By enemy (the input to the solver and judge work)", md)
        self.assertIn("## Human survived, every one of the player's latest plays died", md)
        rare = {r["cell"]: r for r in doc["table"]["by_enemy"]}[f"act2 boss {RARE}"]
        self.assertEqual((rare["starts"], rare["runs"]), (5, 5))  # the JSON keeps it
        self.assertIn("human:R000|f33", json.dumps(doc))  # the JSON carries the start ids

    def test_main_writes_both_files_whole_and_a_gone_shard_falls_back_to_the_outcomes_sign(self):
        out = Path(self.tmp.name) / "out"
        self.assertEqual(hb.main(["--root", str(self.root), "--out-dir", str(out), "--boot", "10"]), 0)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["diagnosis.json", "diagnosis.md"])
        doc = json.loads((out / "diagnosis.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["counts"]["player_lost_more"], 3)
        (self.root / "buffer" / "shards" / "s000002-2.npz").unlink()  # retention removed it: the plays log keeps its lines
        doc2, _ = hb.diagnose(self.root, n_last=4, boot=10)
        k = {r["cell"]: r for r in doc2["table"]["by_enemy"]}[f"act2 boss {BOSS}"]
        self.assertAlmostEqual(k["player_survival"], 26 / 29)  # survival still known from the value
        self.assertLess(k["hp_lost_starts"], 29)  # a win with no fight record has no end HP: its HP lost is not guessed
        self.assertNotIn(sid(3), [r["start"] for r in doc2["lists"]["player_lost_more"]])

    def test_no_plays_and_no_outcomes_give_an_empty_diagnosis(self):
        empty = Path(self.tmp.name) / "empty"
        (empty / "starts").mkdir(parents=True)
        doc, md = hb.diagnose(empty)
        self.assertEqual((doc["starts_played"], doc["counts"]), (0, {"human_survived_player_died": 0, "player_lost_more": 0}))
        self.assertIsNone(doc["table"]["all"])
        self.assertIn("0 of 0", md)


if __name__ == "__main__":
    unittest.main()
