"""The engine-move check (tools/gym_engine_move_check.py; D55 addendum 11, item 3): two games files of one player on two engines, matched by
(start, j), read for outcome, end-HP and move-hash agreement. The figures are counted here by hand on a constructed pair; the tool reads
two files and writes nothing."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "tools", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gym_engine_move_check as mc  # noqa: E402


def row(start: str, j: int, won: bool = True, end_hp: int = 30, moves: str = "m0", root: float | None = None, **extra) -> dict:
    out = {"start": start, "j": j, "won": won, "capped": False, "end_hp": end_hp if won else 0, "moves_sha": moves, **extra}
    if root is not None:
        out["root_value"] = root
    return out


def write(path: Path, rows: list[dict], tail: str = "") -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + tail, encoding="utf-8")
    return path


class MoveCheckTests(unittest.TestCase):
    def pair(self):
        """Ten games a and b share. b differs from a in: game 0's moves only; game 1's moves and end HP; game 2's outcome (a won, b lost);
        game 3 is capped in b (a capped game is a loss); a game only in a and one only in b."""
        a = [row("S", j, root=1.0, moves=f"m{j}") for j in range(10)] + [row("OnlyA", 0)]
        b = [row("S", j, root=1.0 + 1e-7, moves=f"m{j}") for j in range(10)] + [row("OnlyB", 0)]
        b[0] = row("S", 0, root=1.0, moves="other")
        b[1] = row("S", 1, end_hp=25, moves="other")
        b[2] = row("S", 2, won=False, moves="other")
        b[3] = {**row("S", 3, moves="m3"), "capped": True}
        return a, b

    def test_the_figures_are_the_counts_of_the_matched_games(self):
        a, b = self.pair()
        res = mc.compare(mc.index(a), mc.index(b))
        self.assertEqual((res["rows_a"], res["rows_b"], res["matched"], res["only_a"], res["only_b"]), (11, 11, 10, 1, 1))
        self.assertEqual(res["outcome_agree"], 8)  # games 2 and 3 differ
        self.assertEqual(res["end_hp_agree"], 8)  # games 1 (25 against 30) and 2 (lost: 0 against 30) differ; game 3 is capped, with the same end HP
        self.assertEqual((res["moves_compared"], res["moves_agree"]), (10, 7))  # 0, 1 and 2 differ
        self.assertEqual(res["moves_differ_only"], 1)  # game 0: the moves alone
        self.assertEqual((res["win_a"], res["win_b"]), (10, 8))
        self.assertAlmostEqual(res["outcome_share"], 0.8)
        self.assertEqual(res["root_value_pairs"], 7)  # game 0 and games 4 to 9 have a root value in both
        self.assertAlmostEqual(res["root_value_max_diff"], 1e-7, places=12)
        self.assertEqual([(d["start"], d["j"]) for d in res["differing"]], [("S", 0), ("S", 1), ("S", 2), ("S", 3)])
        d = {x["j"]: x for x in res["differing"]}
        self.assertEqual(d[1]["end_hp"], [30, 25])
        self.assertEqual(d[2]["outcome"], [True, False])

    def test_a_game_played_twice_keeps_its_last_row_and_a_config_can_be_asked_for(self):
        rows = [row("S", 0, moves="old", config="a3-v1"), row("S", 0, moves="new", config="a3-v1"), row("S", 1, moves="x", config="judge-v0")]
        self.assertEqual(mc.index(rows)[("S", 0)]["moves_sha"], "new")
        self.assertEqual(sorted(mc.index(rows, "a3-v1")), [("S", 0)])
        self.assertEqual(sorted(mc.index(rows, "judge-v0")), [("S", 1)])

    def test_rows_without_a_move_hash_are_not_counted_in_that_agreement(self):
        a = [row("S", 0), row("S", 1)]
        b = [{k: v for k, v in row("S", 0).items() if k != "moves_sha"}, row("S", 1, moves="different")]
        res = mc.compare(mc.index(a), mc.index(b))
        self.assertEqual((res["moves_compared"], res["moves_agree"], res["matched"]), (1, 0, 2))

    def test_engine_tags_come_from_the_rows_else_from_a_gate_files_name(self):
        self.assertEqual(mc.engines([row("S", 0, engine="e2"), row("S", 1, engine="e1")], Path("x.jsonl")), ["e1", "e2"])
        self.assertEqual(mc.engines([row("S", 0)], Path("inc-v2-judge-v0+jb5ecf903-1b3d1f01c90b.jsonl")), ["1b3d1f01c90b"])
        self.assertEqual(mc.engines([row("S", 0)], Path("games-v2.jsonl")), [])

    def test_the_command_prints_the_counts_and_reads_a_file_a_stop_cut_short(self):
        a, b = self.pair()
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            pa = write(t / "inc-v2-cfg-aaaaaaaaaaaa.jsonl", a, tail='{"start": "S", "j"')  # the last line is cut
            pb = write(t / "inc-v2-cfg-bbbbbbbbbbbb.jsonl", b)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = mc.main([str(pa), str(pb), "--list", "2"])
            text = buf.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("engine aaaaaaaaaaaa", text)
            self.assertIn("engine bbbbbbbbbbbb", text)
            self.assertIn("matched starts (start, j): 10; only in A 1, only in B 1", text)
            self.assertIn("outcome agreement:   8 of 10 (80.0%)", text)
            self.assertIn("end-HP agreement:    8 of 10 (80.0%)", text)
            self.assertIn("move-hash agreement: 7 of 10 (70.0%); 1 of the games whose moves differ have the same outcome and end HP", text)
            self.assertIn("differing games (first 2 of 4):", text)
            self.assertIn("S j=0: moves_sha m0 -> other", text)
            self.assertIn("root value: largest difference", text)
            before = {p.name: p.read_bytes() for p in t.iterdir()}
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(mc.main([str(pa), str(pb), "--json"]), 0)
            doc = json.loads(buf.getvalue())
            self.assertEqual((doc["matched"], doc["moves_agree"], doc["engine_a"], doc["engine_b"], doc["differing"]),
                             (10, 7, ["aaaaaaaaaaaa"], ["bbbbbbbbbbbb"], []))
            self.assertEqual({p.name: p.read_bytes() for p in t.iterdir()}, before)  # it writes nothing

    def test_no_matched_game_is_exit_code_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            pa, pb = write(t / "a.jsonl", [row("A", 0)]), write(t / "b.jsonl", [row("B", 0)])
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(mc.main([str(pa), str(pb)]), 2)
            self.assertIn("matched starts (start, j): 0; only in A 1, only in B 1", out.getvalue())
            self.assertIn("outcome agreement:   0 of 0", out.getvalue())


if __name__ == "__main__":
    unittest.main()
