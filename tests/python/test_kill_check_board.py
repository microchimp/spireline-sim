"""rl/kill_check_board.py: the head-to-head's tasks and figures, on synthetic games (no emulator)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(p))

import kill_check_board as kb  # noqa: E402


def game(config, start, j, won, decisions=10, search_s=1.0, **extra):
    return {"config": config, "start": start, "j": j, "won": won, "capped": False, "L0": 0.8 if won else 0.0, "end_hp": 30 if won else 0,
            "decisions": decisions, "search_s": search_s, "sims": decisions * 30, "seconds": 5.0, "moves_sha": f"{start}{j}", **extra}


class TaskTests(unittest.TestCase):
    STARTS = [{"id": "A|f17", "cat": "act1-boss"}, {"id": "B|f17", "cat": "act1-boss"}]

    def test_every_arm_of_every_game_once_the_two_arms_side_by_side(self):
        tasks = kb.build_tasks(self.STARTS, 2, "v2", "starts.jsonl", "g", set())
        self.assertEqual(8, len(tasks))
        self.assertEqual([("a3-v1", "A|f17", 0), ("a3-v1", "A|f17", 0), ("a3-v1", "B|f17", 0), ("a3-v1", "B|f17", 0)],
                         [(t[0], t[1], t[2]) for t in tasks[:4]])
        self.assertEqual(["a3-v1", "a3-v1-k", "a3-v1", "a3-v1-k"], [t[4]["config"] for t in tasks[:4]])
        info = tasks[0][4]
        self.assertEqual(("v2", "g", False), (info["player"], info["ns"], info["record"]))

    def test_games_with_a_row_are_not_played_again(self):
        done = {("a3-v1", "A|f17", 0), ("a3-v1-k", "A|f17", 0), ("a3-v1", "B|f17", 1)}
        tasks = kb.build_tasks(self.STARTS, 2, "v2", "s.jsonl", "g", done)
        self.assertEqual(5, len(tasks))
        self.assertNotIn(("a3-v1", "B|f17", 1), [(t[4]["config"], t[1], t[2]) for t in tasks])

    def test_the_task_arm_is_the_base_arm_for_the_deadline(self):
        self.assertTrue(all(t[0] == "a3-v1" for t in kb.build_tasks(self.STARTS, 1, "v2", "s", "g", set())))


class FigureTests(unittest.TestCase):
    def rows(self):
        base = [game("a3-v1", "A", 0, True, 10, 2.0), game("a3-v1", "B", 0, False, 12, 3.0), game("a3-v1", "C", 0, False, 8, 1.0),
                game("a3-v1", "D", 0, True, 9, 1.0)]
        kill = [game("a3-v1-k", "A", 0, True, 8, 1.0, lethal_checked=8, lethal_fired=1, lethal_s=0.01, moves_sha="x",
                     lethal_differs=[{"k": 7, "turn": 3, "kill": [2, 0, "card"], "search": [3, 0, "card"], "plays": 2}]),
                game("a3-v1-k", "B", 0, True, 12, 3.0, lethal_checked=12, lethal_fired=0, lethal_s=0.02, moves_sha="B0"),  # won where the base lost
                game("a3-v1-k", "C", 0, False, 8, 1.0, lethal_checked=8, lethal_fired=0, lethal_s=0.01, moves_sha="C0"),
                game("a3-v1-k", "D", 0, False, 9, 1.0, lethal_checked=9, lethal_fired=0, lethal_s=0.01, moves_sha="D0"),  # lost where the base won
                game("a3-v1-k", "E", 0, True)]  # a game the base has no row for: not paired
        return base + kill

    def test_wins_pairs_and_the_discordant_counts(self):
        s = kb.summarize(self.rows())
        p = s["paired"]
        self.assertEqual((4, 2, 2, 1, 1), (p["pairs"], p["wins_base"], p["wins_kill"], p["kill_only"], p["base_only"]))
        self.assertAlmostEqual(0.0, p["win_diff"])
        self.assertEqual(1.0, p["p_two_sided"])
        self.assertEqual(5, s["all_games"]["a3-v1-k"]["games"])  # the unpaired game is in the arm's own table, not the pairs'
        self.assertEqual(4, s["arms"]["a3-v1-k"]["games"])

    def test_search_seconds_per_decision_and_the_fire_counts(self):
        s = kb.summarize(self.rows())
        a, k = s["arms"]["a3-v1"], s["arms"]["a3-v1-k"]
        self.assertAlmostEqual(1000 * 7.0 / 39, a["search_ms_per_decision"], places=2)
        self.assertAlmostEqual(1000 * 6.0 / 37, k["search_ms_per_decision"], places=2)
        self.assertEqual((37, 1, 1), (k["checked"], k["fired"], k["games_with_a_fire"]))
        self.assertAlmostEqual(1 / 37, k["fire_rate"])
        self.assertAlmostEqual(1000 * 0.05 / 37, k["check_ms_per_look"], places=3)
        self.assertNotIn("checked", a)  # the base arm's rows carry no check

    def test_the_decision_where_the_move_differed_is_listed_with_its_game(self):
        d = kb.summarize(self.rows())["differs"]
        self.assertEqual(1, d["count"])
        self.assertEqual(("A", 0, 7, [2, 0, "card"], [3, 0, "card"]),
                         tuple(d["decisions"][0][x] for x in ("start", "j", "k", "kill", "search")))

    def test_a_fired_kill_that_failed_on_the_true_position_is_listed(self):
        rows = [game("a3-v1", "A", 0, True), game("a3-v1-k", "A", 0, True, lethal_checked=3, lethal_fired=1, lethal_s=0.01,
                                                  lethal_wrong=[{"k": 4, "turn": 2, "line": [[0, 0, "card"]]}]),
                game("a3-v1", "B", 0, True), game("a3-v1-k", "B", 0, True, lethal_checked=3, lethal_fired=0, lethal_s=0.01, lethal_wrong=[])]
        s = kb.summarize(rows)
        self.assertEqual(1, s["wrong_kills"]["count"])
        self.assertEqual(("A", 0, 4), tuple(s["wrong_kills"]["decisions"][0][x] for x in ("start", "j", "k")))
        self.assertIn("did not hold on the true position: 1", kb.report_lines(s)[3])

    def test_games_played_move_for_move_alike_are_counted(self):
        s = kb.summarize(self.rows())
        self.assertEqual(3, s["paired"]["games_played_alike"])  # B, C and D share a moves_sha across the arms; A (where the check fired) does not
        same = [game("a3-v1", "A", 0, True), game("a3-v1-k", "A", 0, True)]
        self.assertEqual(1, kb.summarize(same)["paired"]["games_played_alike"])

    def test_a_capped_game_is_a_loss(self):
        rows = [game("a3-v1", "A", 0, True, capped=True), game("a3-v1-k", "A", 0, True)]
        rows[0]["capped"] = True
        p = kb.summarize(rows)["paired"]
        self.assertEqual((0, 1, 1), (p["wins_base"], p["wins_kill"], p["kill_only"]))

    def test_the_exact_test(self):
        self.assertEqual(1.0, kb.exact_two_sided(0, 0))
        self.assertAlmostEqual(2 * (1 / 2**10), kb.exact_two_sided(10, 0))
        self.assertAlmostEqual(1.0, kb.exact_two_sided(5, 5))

    def test_the_lines_read_and_an_empty_run_does_not_break_them(self):
        lines = kb.report_lines(kb.summarize(self.rows()))
        self.assertEqual(4, len(lines))
        self.assertIn("fired 1", lines[1])
        empty = kb.summarize([])
        self.assertEqual(0, empty["paired"]["pairs"])
        kb.report_lines(empty)

    def test_a_game_played_twice_counts_once_and_the_last_row_wins(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "games.jsonl"
            f.write_text("\n".join(json.dumps(r) for r in [game("a3-v1", "A", 0, False), game("a3-v1", "A", 0, True)]) + "\n{cut", encoding="utf-8")
            rows = kb.read_games(f)
        self.assertEqual([True], [r["won"] for r in rows])


if __name__ == "__main__":
    unittest.main()
