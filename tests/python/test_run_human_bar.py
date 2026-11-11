"""rl/run_human_bar.py: the run-level human bar (D56, D58 C5): floor reached and HP at the Act 1 and Act 2 boss doors, the humans' and the copy's.

Synthetic run records stand in for the corpus (the real corpus is not read here). The pinned facts: the exam fifth is never read (a run of the hash
bucket 0 or listed in the ids file is left out before any figure; a figure that would include one stops with `assert_no_exam`; the fifths asked for
may not include 0), the figures are aggregates with no run id or seed in them, a cell over fewer than 20 runs is hidden, the file is computed once,
and the copy's side (RunRecords and judgement arm rows) has the same shape.
"""

from __future__ import annotations

import ast
import gzip
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import human_starts as hs  # noqa: E402
import run_human_bar as hb  # noqa: E402


def find_ids(bucket: int | None, n: int, prefix: str = "h") -> list[str]:
    """`n` run ids in the hash bucket `bucket` (None: any bucket but 0)."""
    out, i = [], 0
    while len(out) < n:
        rid = f"{prefix}{i:012x}"
        b = hs.exam_bucket(rid)
        if (bucket is None and b != 0) or b == bucket:
            out.append(rid)
        i += 1
    return out


def pt(kind: str, hp: int, max_hp: int = 80, **extra) -> dict:
    return {"map_point_type": kind, "player_stats": [{"current_hp": hp, "max_hp": max_hp, **extra}], "rooms": [{"room_type": kind}]}


def act(n_before_boss: int, door_hp: int, boss_hp_after: int, max_hp: int = 80) -> list[dict]:
    """An act: monsters, the rest site before the boss (whose current_hp is the HP carried into the boss), the boss."""
    return [pt("monster", 60) for _ in range(n_before_boss - 1)] + [pt("rest_site", door_hp, max_hp), pt("boss", boss_hp_after, max_hp, damage_taken=30)]


def human_record(rid: str, a1: tuple[int, int] | None, a2: tuple[int, int] | None = None, win: bool = False, stop_after: int = 4) -> dict:
    """A run that reached the Act 1 door with `a1` = (door HP, HP after) or stopped `stop_after` points in; then the Act 2 door likewise."""
    mph: list[list[dict]] = []
    if a1 is None:
        mph.append([pt("monster", 50) for _ in range(stop_after)])
    else:
        mph.append(act(16, a1[0], a1[1]))
        if a2 is not None:
            mph.append(act(15, a2[0], a2[1]))
            if win:
                mph.append([pt("monster", 40) for _ in range(15)])
        elif not win and stop_after:
            mph.append([pt("monster", 50) for _ in range(stop_after)])
    return {"acts": ["ACT.OVERGROWTH", "ACT.HIVE", "ACT.GLORY"], "ascension": 10, "game_mode": "standard", "was_abandoned": False, "modifiers": [],
            "players": [{"character": "CHARACTER.IRONCLAD"}], "win": win, "run_hash": rid, "seed": "S" + rid, "start_time": 1, "map_point_history": mph}


def write_corpus(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")


class OneRunTests(unittest.TestCase):
    def test_a_run_that_killed_the_act_1_boss_and_died_in_act_2(self):
        row = hb.human_run(human_record("r1", (51, 16), None, stop_after=5))
        self.assertEqual(17 + 5, row["floor"])
        self.assertEqual({1: {"hp": 51, "max_hp": 80, "killed": True}}, row["doors"])  # the rest site's HP is the HP carried into the boss

    def test_a_run_that_died_to_the_act_1_boss_killed_nothing_and_reached_no_second_door(self):
        row = hb.human_run(human_record("r2", (40, 0), None, stop_after=0) | {"win": False})
        self.assertEqual(17, row["floor"])
        self.assertEqual({1: {"hp": 40, "max_hp": 80, "killed": False}}, row["doors"])

    def test_a_win_reached_both_doors_and_killed_both_bosses(self):
        row = hb.human_run(human_record("r3", (60, 20), (70, 30), win=True))
        self.assertEqual(17 + 16 + 15, row["floor"])
        self.assertTrue(row["win"])
        self.assertTrue(row["doors"][1]["killed"] and row["doors"][2]["killed"])
        self.assertEqual(70, row["doors"][2]["hp"])

    def test_a_run_that_stopped_before_the_boss_has_no_door(self):
        row = hb.human_run(human_record("r4", None, stop_after=6))
        self.assertEqual((6, {}), (row["floor"], row["doors"]))

    def test_a_record_without_a_map_is_nothing_and_the_player_id_is_never_read(self):
        self.assertIsNone(hb.human_run({"map_point_history": []}))
        self.assertIsNone(hb.human_run({}))
        text = (ROOT / "rl" / "run_human_bar.py").read_text(encoding="utf-8")
        self.assertNotIn('players"][0]', text)
        self.assertNotIn("player_id", text)
        self.assertNotIn("players[0]", text)


class SummaryTests(unittest.TestCase):
    def rows(self, n: int, door2: int = 0):
        out = []
        for i in range(n):
            a2 = (50 + i % 10, 10) if i < door2 else None
            out.append(hb.human_run(human_record(f"x{i}", (40 + i % 20, 10), a2, stop_after=3)))
        return out

    def test_a_cell_under_twenty_runs_is_hidden_and_named(self):
        doc = hb.summarise(self.rows(30, door2=12))
        self.assertEqual(30, doc["runs"])
        self.assertIsNone(doc["acts"]["2"])  # 12 runs reached the Act 2 door: not shown
        self.assertIn("act2", doc["hidden"])
        self.assertEqual(30, doc["acts"]["1"]["reached"])
        self.assertEqual(30, doc["acts"]["1"]["door_hp"]["n"])
        doc = hb.summarise(self.rows(30, door2=25))
        self.assertEqual(25, doc["acts"]["2"]["reached"])
        self.assertEqual([], doc["hidden"])

    def test_under_twenty_runs_shows_no_figure_at_all(self):
        doc = hb.summarise(self.rows(19))
        self.assertEqual((None, None, None), (doc["runs"], doc["floor"], doc["win_rate"]))
        self.assertIn("runs", doc["hidden"])
        self.assertEqual({"1": None, "2": None}, doc["acts"])

    def test_the_figures_are_the_means_of_the_rows(self):
        rows = [{"floor": 20 + i, "win": i == 0, "doors": {1: {"hp": 30 + i, "max_hp": 60, "killed": i % 2 == 0}}} for i in range(40)]
        doc = hb.summarise(rows)
        self.assertAlmostEqual(sum(20 + i for i in range(40)) / 40, doc["floor"]["mean"], places=3)
        self.assertEqual(0.025, doc["win_rate"])
        a1 = doc["acts"]["1"]
        self.assertEqual((40, 1.0, 20, 0.5), (a1["reached"], a1["reached_share"], a1["killed"], a1["kill_rate"]))
        self.assertAlmostEqual(sum(30 + i for i in range(40)) / 40, a1["door_hp"]["mean"], places=2)
        self.assertAlmostEqual(sum((30 + i) / 60 for i in range(40)) / 40, a1["door_hp"]["share_of_max"], places=3)

    def test_a_copy_without_hp_has_a_door_but_no_door_hp(self):
        rows = [{"floor": 18, "win": False, "doors": {1: {"hp": None, "max_hp": None, "killed": True}}} for _ in range(25)]
        a1 = hb.summarise(rows)["acts"]["1"]
        self.assertEqual((25, 25, None), (a1["reached"], a1["killed"], a1["door_hp"]))


class ExamFifthTests(unittest.TestCase):
    """The corpus fixture has the exam fifth planted two ways: runs of the hash bucket 0, and runs of another bucket listed in the ids file."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.keep = find_ids(None, 50, "k")
        self.hash_exam = find_ids(0, 6, "e")
        self.listed = [i for i in find_ids(None, 5, "l")]
        recs = [human_record(rid, (45, 10), (55, 12) if n % 2 == 0 else None) for n, rid in enumerate(self.keep)]
        # the planted exam runs reach the Act 1 door with 1 HP: a figure that read them would show it
        recs += [human_record(rid, (1, 0)) for rid in self.hash_exam + self.listed]
        self.corpus = self.tmp / "runs.jsonl"
        write_corpus(self.corpus, recs)

    def test_the_exam_fifth_is_left_out_before_any_figure_and_counted(self):
        doc = hb.human_figures(self.corpus, exam_ids=set(self.listed))
        self.assertEqual(50, doc["runs"])
        self.assertEqual(len(self.hash_exam) + len(self.listed), doc["read"]["exam_fifth_left_out"])
        self.assertEqual(45.0, doc["acts"]["1"]["door_hp"]["mean"])  # not pulled down by the planted 1 HP runs
        self.assertEqual(25, doc["acts"]["2"]["reached"])

    def test_a_figure_that_would_hold_an_exam_run_is_refused(self):
        """Were the reader to hand over a run listed in the ids file, the guard stops the figure (the reader never does: the test makes it)."""
        planted = json.dumps(human_record(self.listed[0], (1, 0)), separators=(",", ":"))
        import human_prior as hp  # noqa: PLC0415

        with mock.patch.object(hp, "iter_run_lines", return_value=iter([planted])):
            with self.assertRaises(AssertionError):
                hb.human_figures(self.corpus, exam_ids=set(self.listed))

    def test_a_run_of_the_hash_bucket_zero_handed_over_is_not_counted(self):
        planted = json.dumps(human_record(self.hash_exam[0], (1, 0)), separators=(",", ":"))
        import human_prior as hp  # noqa: PLC0415

        with mock.patch.object(hp, "iter_run_lines", return_value=iter([planted])):
            doc = hb.human_figures(self.corpus, exam_ids=set(self.listed))
        self.assertEqual(1, doc["read"]["other_fifth_left_out"])
        self.assertIsNone(doc["runs"])  # nothing was left to count (and a single run is not a cell)

    def test_the_exam_fifth_cannot_be_asked_for(self):
        with self.assertRaises(ValueError):
            hb.human_figures(self.corpus, exam_ids=set(), fifths=(0, 1))

    def test_the_stored_file_holds_aggregates_only_and_is_computed_once(self):
        doc = hb.human_figures(self.corpus, exam_ids=set(self.listed))
        out = self.tmp / "human-bar" / "run-level.json"
        hb.write_human(out, doc)
        text = out.read_text(encoding="utf-8")
        for rid in self.keep + self.hash_exam + self.listed:
            self.assertNotIn(rid, text)
        self.assertNotIn("S" + self.keep[0], text)  # no seed
        for bad in ("D:/", "C:/", "Users", "spireline"):
            self.assertNotIn(bad, text)
        got = json.loads(text)
        self.assertEqual("Spire Codex public run export for v0.107.1", got["source"])
        self.assertEqual([1, 2, 3, 4], got["fifths"])
        with self.assertRaises(FileExistsError):
            hb.write_human(out, doc)
        hb.write_human(out, doc, force=True)
        self.assertEqual(got["runs"], hb.read_human(out)["runs"])
        self.assertEqual([], [p.name for p in out.parent.iterdir() if p.name.endswith(tuple("0123456789")) and ".tmp" in p.name])

    def test_a_small_fixture_hides_its_cells_in_the_stored_file(self):
        out = self.tmp / "small.json"
        small = self.tmp / "small.jsonl"
        write_corpus(small, [human_record(rid, (45, 10), (55, 12) if n < 5 else None) for n, rid in enumerate(self.keep[:25])])
        hb.write_human(out, hb.human_figures(small, exam_ids=set()))
        got = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(25, got["runs"])
        self.assertIsNone(got["acts"]["2"])  # five runs reached the Act 2 door
        self.assertIn("act2", got["hidden"])

    def test_the_module_reads_only_the_standard_library_at_its_top(self):
        tree = ast.parse((ROOT / "rl" / "run_human_bar.py").read_text(encoding="utf-8"))
        names = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
        self.assertLessEqual(names, set(sys.stdlib_module_names) | {"__future__"})


class CopyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def record(self, floor: int, doors: dict[int, tuple[int, bool]], won: bool = False) -> dict:
        fights = [{"floor": {1: 17, 2: 33}[a], "node_type": 6, "hp_before": hp, "hp_after": 0 if not w else 5, "won": w, "state": {"max_hp": 80}}
                  for a, (hp, w) in doors.items()]
        fights.append({"floor": 4, "node_type": 1, "hp_before": 70, "hp_after": 60, "won": True, "state": {"max_hp": 80}})
        return {"outcome": {"final_floor": floor, "won": won, "died": not won}, "fights": fights}

    def test_a_run_record_gives_the_floor_and_the_hp_each_boss_fight_began_with(self):
        row = hb.copy_run(self.record(34, {1: (55, True), 2: (30, False)}))
        self.assertEqual(34, row["floor"])
        self.assertEqual({1: {"hp": 55, "max_hp": 80, "killed": True}, 2: {"hp": 30, "max_hp": 80, "killed": False}}, row["doors"])
        self.assertIsNone(hb.copy_run({"outcome": {}}))

    def test_recorded_runs_are_read_from_shards_and_capped(self):
        d = self.tmp / "jobs" / "j1" / "shards"
        d.mkdir(parents=True)
        recs = [self.record(20 + i, {1: (40 + i, i % 2 == 0)}) for i in range(30)]
        (d / "r0000.jsonl.gz").write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in recs).encode()))
        doc = hb.copy_from_records([d])
        self.assertEqual((30, "recorded runs"), (doc["runs"], doc["source"]))
        self.assertEqual(30, doc["acts"]["1"]["reached"])
        self.assertAlmostEqual(sum(40 + i for i in range(30)) / 30, doc["acts"]["1"]["door_hp"]["mean"], places=2)
        self.assertEqual([None] * 1, [hb.copy_from_records([d], max_runs=10)["floor"]])  # 10 runs: under a cell

    def test_arm_rows_give_the_floor_and_the_bosses_reached_but_no_hp(self):
        rows = [{"floor": 17, "win": False, "crash": False, "boss_damage_acts": [[0, 0.4]]} for _ in range(12)]
        rows += [{"floor": 30, "win": False, "boss_damage_acts": [[0, 1.0], [1, 0.2]]} for _ in range(13)]
        rows += [{"floor": 12, "win": False, "boss_damage_acts": []}, {"floor": 5, "crash": True}]
        p = self.tmp / "runs.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        doc = hb.copy_from_arms([p, self.tmp / "missing.jsonl"])
        self.assertEqual(26, doc["runs"])  # the crash row is not a run
        a1 = doc["acts"]["1"]
        self.assertEqual((25, 13), (a1["reached"], a1["killed"]))
        self.assertIsNone(a1["door_hp"])
        self.assertIsNone(doc["acts"]["2"])  # 13 reached the second door: hidden


class TableTests(unittest.TestCase):
    def test_the_table_lines_the_figures_up_and_hides_what_is_hidden(self):
        human = hb.summarise([{"floor": 25 + i % 10, "win": False, "doors": {1: {"hp": 50, "max_hp": 80, "killed": True},
                                                                           2: {"hp": 40, "max_hp": 80, "killed": False}}} for i in range(40)])
        copy = hb.summarise([{"floor": 20, "win": False, "doors": {1: {"hp": 30, "max_hp": 80, "killed": False}}} for _ in range(25)])
        text = hb.beside_text(human, {**copy, "source": "recorded runs"}, label="candidate")
        self.assertIn("Spire Codex public run export for v0.107.1", text)
        self.assertIn("| figure | humans | candidate |", text)
        self.assertIn("| Act 1 HP at the door | 50.0 (62.5% of max) | 30.0 (37.5% of max) |", text)
        self.assertIn("| Act 2 HP at the door | 40.0 (50.0% of max) | - (- of max) |", text)  # the copy has no Act 2 cell: hidden
        self.assertIn("| mean floor reached | 29.5 | 20.0 |", text)
        self.assertIn("fewer than 20 runs is not shown", text)

    def test_without_the_stored_human_figures_it_says_how_to_make_them(self):
        self.assertIn("rl/run_human_bar.py compute", hb.beside_text(None, None))

    def test_the_command_line_prints_the_table(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        human = tmp / "h.json"
        hb.write_human(human, hb.summarise([{"floor": 30, "win": False, "doors": {1: {"hp": 50, "max_hp": 80, "killed": True}}} for _ in range(30)]))
        runs = tmp / "runs.jsonl"
        runs.write_text("".join(json.dumps({"floor": 17, "win": False, "boss_damage_acts": [[0, 0.5]]}) + "\n" for _ in range(22)), encoding="utf-8")
        import contextlib  # noqa: PLC0415
        import io  # noqa: PLC0415

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = hb.main(["beside", "--human", str(human), "--records", str(tmp / "nowhere"), "--arms", str(runs)])
        self.assertEqual(0, rc)
        self.assertIn("arm rows (no HP), 22 runs", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
