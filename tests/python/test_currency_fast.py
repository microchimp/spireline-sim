"""`rl/currency_fast.py`: the readings of the run currency's fast tests and the rule that holds them to a free machine.

Synthetic rows only; nothing is played. Pinned: the paired difference and its standard error, the board's lower bound, the hallway cell's
drinking share, a batch's hand at the Act 1 boss door, the verdict's clauses (each true, false or not yet played), the QUIET.lock rule (the
loop's and the lab score's lines do not hold the job, any other does), and the batch's seeds.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import currency_fast as cf  # noqa: E402


def game(arm, start, j, won, end_hp=30, used=0, held=0):
    return {"arm": arm, "start": start, "j": j, "won": won, "end_hp": end_hp if won else 0, "potions_used": used, "potions_held": held,
            "L0": 0.5 if won else 0.0, "capped": False}


def run(seed, door=None, beat=False, floor=17):
    row = {"seed": seed, "floor": floor, "wall": 30.0, "boss_door": [door] if door else [], "boss_damage_acts": []}
    if door:
        row["boss_damage_acts"] = [[0, 1.0 if beat else 0.4]]
    return row


class ReadingTests(unittest.TestCase):
    def test_paired_is_the_mean_difference_with_its_standard_error(self):
        r = cf.paired([(1, 2), (2, 4), (3, 3), (4, 7)])
        self.assertEqual(r["n"], 4)
        self.assertAlmostEqual(r["diff"], (1 + 2 + 0 + 3) / 4)
        d = [1, 2, 0, 3]
        m = sum(d) / 4
        self.assertAlmostEqual(r["se"], math.sqrt(sum((x - m) ** 2 for x in d) / 3 / 4))
        self.assertEqual(cf.paired([(1, 2)])["se"], None)
        self.assertEqual(cf.paired([])["diff"], None)

    def test_the_board_reading_has_the_difference_in_points_and_its_lower_bound(self):
        rows = []
        for i in range(40):
            rows.append(game("a3-v1", f"S{i}", 0, won=i % 2 == 0))
            rows.append(game("a3-v1-r", f"S{i}", 0, won=i % 4 != 3))
        r = cf.board_reading(rows)
        self.assertEqual(r["pairs"], 40)
        self.assertAlmostEqual(r["win_diff_points"], 100 * (30 - 20) / 40)
        self.assertLess(r["lower95_points"], r["win_diff_points"])
        self.assertAlmostEqual(r["a3-v1"]["win_rate"], 0.5)
        self.assertAlmostEqual(r["a3-v1-r"]["win_rate"], 0.75)

    def test_a_hallway_cell_counts_the_wins_that_began_with_a_potion_and_drank_one(self):
        starts = {"H1": {"cat": "act1-hallway", "hp": 60, "potions": 1}, "H2": {"cat": "act1-hallway", "hp": 50, "potions": 0},
                  "E1": {"cat": "act1-elite", "hp": 70, "potions": 2}}
        rows = [game("a3-v1", "H1", 0, True, end_hp=40, used=1), game("a3-v1", "H1", 1, True, end_hp=45, used=0),
                game("a3-v1", "H2", 0, True, end_hp=50), game("a3-v1", "E1", 0, False),
                game("a3-v1-r", "H1", 0, True, end_hp=40), game("a3-v1-r", "H1", 1, True, end_hp=45), game("a3-v1-r", "H2", 0, False),
                game("a3-v1-r", "E1", 0, True, end_hp=10, used=2)]
        c = cf.cell_reading(rows, starts)
        h, hr = c["act1-hallway"]["a3-v1"], c["act1-hallway"]["a3-v1-r"]
        self.assertEqual((h["games"], h["won_with_potion"], h["drank_share"]), (3, 2, 0.5))
        self.assertEqual(hr["drank_share"], 0.0)
        self.assertAlmostEqual(h["hp_lost"], ((60 - 40) + (60 - 45) + 0) / 3)
        self.assertAlmostEqual(hr["hp_lost"], ((60 - 40) + (60 - 45) + 50) / 3)  # a loss costs the whole start HP
        self.assertAlmostEqual(hr["win_rate"], 2 / 3)
        self.assertEqual(c["act1-elite"]["a3-v1-r"]["potions_used"], 2)

    def test_a_batch_reads_the_hand_at_the_act_one_boss_door_and_the_kill(self):
        a = [run("A", [0, 1, 4, 60, 80], beat=True), run("B", [0, 0, 4, 40, 80], beat=False), run("C", None, floor=9), run("D", [0, 2, 5, 80, 80], True)]
        r = [run("A", [0, 2, 4, 70, 80], beat=True), run("B", [0, 1, 5, 50, 80], beat=True), run("C", None, floor=8), run("D", [0, 2, 5, 80, 80], True)]
        reading = cf.batch_reading(a, r)
        self.assertEqual((reading["a3-v1"]["reached"], reading["a3-v1"]["beat"]), (3, 2))
        self.assertAlmostEqual(reading["a3-v1"]["potions_at_door"], 1.0)
        self.assertAlmostEqual(reading["a3-v1-r"]["potions_at_door"], 5 / 3)
        self.assertEqual(reading["seeds_at_both_doors"], 3)
        self.assertAlmostEqual(reading["potions_at_door_diff"]["diff"], (1 + 1 + 0) / 3)
        self.assertAlmostEqual(reading["act1_boss_kill_per_100_runs_diff"]["diff"], 100 * (3 - 2) / 4)
        self.assertAlmostEqual(reading["a3-v1"]["hp_share_at_door"], (0.75 + 0.5 + 1.0) / 3)
        self.assertAlmostEqual(reading["a3-v1"]["kill_of_reached"], 2 / 3)

    def test_the_verdict_reads_each_clause_and_leaves_an_unplayed_one_open(self):
        batch = {"potions_at_door_diff": {"diff": 0.3, "se": 0.1}, "act1_boss_kill_per_100_runs_diff": {"diff": -1.0, "se": 3.0},
                 "relics_at_door_diff": {"diff": 0.0, "se": 0.1}}
        board = {"lower95_points": -1.5}
        cells = {"act1-hallway": {"a3-v1": {"win_rate": 0.995, "drank_share": 0.6}, "a3-v1-r": {"win_rate": 0.992, "drank_share": 0.3}}}
        v = cf.verdict({"passed": True}, board, cells, batch)
        self.assertTrue(v["passes"])
        self.assertEqual({k: x for k, x in v.items() if k != "passes"}, {
            "audit_passed": True, "potions_at_the_door_up": True, "boss_kill_not_down": True, "relics_at_the_door_not_down": True,
            "board_lower95_above_minus_2": True, "hallway_win_rate_at_least_99": True, "hallway_drinking_down": True})
        down = dict(batch, act1_boss_kill_per_100_runs_diff={"diff": -8.0, "se": 3.0})
        self.assertFalse(cf.verdict({"passed": True}, board, cells, down)["passes"])
        self.assertFalse(cf.verdict({"passed": True}, board, cells, dict(batch, potions_at_door_diff={"diff": -0.1, "se": 0.1}))["passes"])
        self.assertFalse(cf.verdict({"passed": False}, board, cells, batch)["passes"])
        self.assertFalse(cf.verdict({"passed": True}, {"lower95_points": -3.0}, cells, batch)["passes"])
        self.assertFalse(cf.verdict({"passed": True}, board, None, None)["passes"])  # the batch has not been played
        unplayed = cf.verdict({"passed": True}, None, None, batch)
        self.assertNotIn("board_lower95_above_minus_2", unplayed)  # a test not played is not a clause that failed
        self.assertTrue(unplayed["passes"])

    def test_the_batch_seeds_are_ten_distinct_letters_and_digits(self):
        seeds = cf.batch_seeds(200)
        self.assertEqual(len(set(seeds)), 200)
        self.assertTrue(all(len(s) == 10 and s.isalnum() and s.upper() == s for s in seeds))
        self.assertEqual(seeds[:3], cf.batch_seeds(3))


class FreeMachineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ref = Path(self.tmp.name)
        (self.ref / "LAB" / "nightly").mkdir(parents=True)
        (self.ref / "backlog").mkdir()
        self.patches = [mock.patch.object(cf, "QUIET", self.ref / "QUIET.lock"), mock.patch.object(cf, "MAIN_LOCK", self.ref / "backlog" / "main.lock"),
                        mock.patch.object(cf, "LAB", self.ref / "LAB")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def lines(self, *xs):
        (self.ref / "QUIET.lock").write_text("".join(x + "\n" for x in xs), encoding="utf-8")

    def test_the_loop_and_the_lab_score_do_not_hold_the_job_and_any_other_line_does(self):
        self.assertIsNone(cf.blocked_reason())
        self.lines("lab-score 2026-11-06T03:41:57Z pid=1", "gym-loop 2026-11-06T04:00:00Z pid=2")
        self.assertIsNone(cf.blocked_reason())
        self.lines("lab-score 2026-11-06T03:41:57Z pid=1", "board-a 2026-11-06T04:37:39Z pid=3")
        self.assertEqual(cf.blocked_reason(), "quiet:board-a")
        self.lines("b-fast 2026-11-06T05:00:00Z pid=4")
        self.assertIsNone(cf.blocked_reason())  # its own line

    def test_a_night_a_score_block_the_main_lock_and_a_real_game_job_hold_it_too(self):
        (self.ref / "LAB" / "nightly" / "active").write_text("x", encoding="utf-8")
        self.assertEqual(cf.blocked_reason(), "nightly-active")
        (self.ref / "LAB" / "nightly" / "active").unlink()
        (self.ref / "backlog" / "main.lock").write_text("x", encoding="utf-8")
        self.assertEqual(cf.blocked_reason(), "main.lock")
        (self.ref / "backlog" / "main.lock").unlink()
        (self.ref / "LAB" / "status.txt").write_text("2026-11-06T05:00:00Z running job=20261106T0500-nightly-real step=play\n", encoding="utf-8")
        self.assertEqual(cf.blocked_reason(), "a real-game job is running")
        (self.ref / "LAB" / "status.txt").write_text("2026-11-06T05:00:00Z paused(pause) job=- since=x\n", encoding="utf-8")
        self.assertIsNone(cf.blocked_reason())

    def test_the_jobs_line_is_written_and_removed_without_touching_the_others(self):
        self.lines("lab-score 2026-11-06T03:41:57Z pid=1")
        cf.set_line(True)
        text = (self.ref / "QUIET.lock").read_text(encoding="utf-8").splitlines()
        self.assertEqual(text[0], "lab-score 2026-11-06T03:41:57Z pid=1")
        self.assertTrue(text[1].startswith("b-fast 20") and text[1].endswith(f"pid={os.getpid()}"))
        cf.set_line(True)
        self.assertEqual(len((self.ref / "QUIET.lock").read_text(encoding="utf-8").splitlines()), 2)
        cf.set_line(False)
        self.assertEqual((self.ref / "QUIET.lock").read_text(encoding="utf-8").splitlines(), ["lab-score 2026-11-06T03:41:57Z pid=1"])
        self.lines("b-fast 2026-11-06T05:00:00Z pid=4")
        cf.set_line(False)
        self.assertFalse((self.ref / "QUIET.lock").exists())


if __name__ == "__main__":
    unittest.main()
