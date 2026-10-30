"""The fight gym's pure parts (rl/fight_gym.py): seed pairing, start selection, the standard errors, resume, the guard."""

import datetime as dt
import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))

import fight_gym as fg  # noqa: E402
import search_configs as sc  # noqa: E402


def _run(ix: int, floor: int, nodes: str, end: str = "death") -> dict:
    return {"seed": f"S{ix:04d}", "ix": ix, "floor": floor, "end": end, "nodes": nodes, "crash": False}


ACT1 = "M?SM???RT?RM?ERB"  # 16 nodes: index 15 (floor 17) is the Act-1 boss
ACT2 = ACT1 + "AM?E??RTMEM?R?RB"  # elites at floors 21 and 27, the Act-2 boss at floor 33


def _runs() -> list[dict]:
    rows = [_run(i, 17, ACT1) for i in range(40)]
    rows += [_run(100 + i, 33, ACT2) for i in range(20)]
    rows += [_run(200 + i, 24, ACT2[:23]) for i in range(10)]  # reached the floor-21 elite only
    rows += [_run(300, 12, ACT1[:11]), _run(301, 17, ACT1, end="lost"), {**_run(302, 17, ACT1), "crash": True}]
    return rows


def _fake_task(task):
    """A worker task for the guard tests: ("sleep", id, seconds), ("alloc", id, MB, then sleeps), ("ok", id, 0)."""
    kind, tid, x = task
    fg._progress({"k": 7, "turns": 3})
    if kind == "sleep":
        time.sleep(x)
    if kind == "alloc":
        blob = bytearray(int(x * 2**20))
        blob[:: 4096] = b"x" * len(blob[:: 4096])  # touch every page so it is private memory
        time.sleep(60)
    if kind == "raise":
        raise RuntimeError("combat action refused")
    return {"_task": list(task), "ok": True}


def _fake_hard(task):
    return 3.0


class WorkerGuardTests(unittest.TestCase):
    def test_a_task_past_its_deadline_or_over_memory_is_capped_and_the_rest_still_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "rows.jsonl"
            tasks = [("sleep", "a", 120), ("ok", "b", 0), ("alloc", "c", 600), ("raise", "d", 0), ("ok", "e", 0)]
            t0 = time.perf_counter()
            stopped = fg.run_tasks(tasks, _fake_task, target, 2, hard_of=_fake_hard, mem_limit=300 * 2**20,
                                   capped_of=fg.capped_row, poll_s=0.2)
            seconds = time.perf_counter() - t0
            self.assertIsNone(stopped)
            self.assertLess(seconds, 60)  # never waits on the sleeper or the allocator
            rows = {r["_task"][1]: r for r in fg.read_jsonl(target)}
            self.assertEqual(sorted(rows), ["a", "b", "c", "d", "e"])
            self.assertTrue(rows["b"]["ok"] and rows["e"]["ok"])
            self.assertEqual(rows["a"]["cap_reason"], "deadline (killed)")
            self.assertGreaterEqual(rows["a"]["seconds"], 3.0)
            self.assertTrue(rows["c"]["cap_reason"].startswith("memory over 0.3 GB"), rows["c"])
            self.assertTrue(rows["d"]["cap_reason"].startswith("error: RuntimeError"))

    def test_a_capped_game_is_lost_with_how_far_it_got(self):
        row = fg.capped_row(("x16", "S|f17", 1, "act1-boss"), "deadline (killed)", 1234.5, {"k": 88, "turns": 20, "sims": 9000,
                                                                                               "max_hp": 93, "root_value": -40.0})
        self.assertEqual((row["arm"], row["start"], row["j"], row["cat"]), ("x16", "S|f17", 1, "act1-boss"))
        self.assertEqual((row["won"], row["capped"], row["L0"], row["decisions"], row["turns"], row["seconds"]),
                         (False, True, 0.0, 88, 20, 1234.5))
        self.assertEqual(row["root_value"], -40.0)
        self.assertEqual(fg.game_key(row), ("x16", "S|f17", 1))  # a resume skips it
        cont = fg.capped_row(("cont", "S|f17", 4), "memory over 3.0 GB (killed)", 9.0, {})
        self.assertEqual((cont["type"], cont["won"], cont["L0"]), ("cont", False, 0.0))

    def test_deadlines_are_ten_times_the_arms_mean_and_the_kill_comes_later(self):
        self.assertAlmostEqual(fg.deadline_s("x16", "act1-boss"), 742.0)
        self.assertAlmostEqual(fg.deadline_s("x4", "act2-elite"), 350.0)
        self.assertAlmostEqual(fg.task_deadline(("a3-v1", "S|f17", 0, "act1-boss")), 64.0)
        self.assertAlmostEqual(fg.task_deadline(("cont", "S|f17", 0)), 64.0)
        self.assertGreater(fg.hard_deadline(("x16", "S|f17", 0, "act1-boss")), fg.deadline_s("x16", "act1-boss"))
        for (cat, arm), m in fg.MEAN_S.items():
            self.assertGreater(m, 0, (cat, arm))


class SeedPairingTests(unittest.TestCase):
    def test_seeds_depend_on_start_game_and_decision_never_on_the_arm(self):
        self.assertEqual(fg.game_seed("A|f17", 3, "g5"), fg.game_seed("A|f17", 3, "g5"))
        seen = {fg.game_seed(s, j, w) for s in ("A|f17", "B|f17") for j in range(10) for w in ("gload", "gfork", "g0", "g1")}
        self.assertEqual(len(seen), 2 * 10 * 4)
        self.assertNotIn("arm", fg.game_seed.__code__.co_varnames)
        for s in seen:
            self.assertTrue(0 <= s < 1 << 30)

    def test_every_arm_gets_the_same_games(self):
        starts = [{"id": "A|f17"}, {"id": "B|f17"}]
        lists = [fg.todo_games(starts, arm, 10, set()) for arm in fg.ARMS]
        self.assertEqual(lists[0], lists[1])
        self.assertEqual(lists[0], lists[2])
        self.assertEqual(len(lists[0]), 20)

    def test_arms_multiply_every_simulation_count_and_nothing_else(self):
        base = sc.get("a3-v1")
        self.assertIs(fg.arm_settings("a3-v1")["cfg"], base)
        target = sc.Position(kind="target", n_moves=6, hp_frac=0.9, hard=True)
        play = sc.Position(kind="play", n_moves=6, hp_frac=0.2, hard=True)
        for arm, m in (("x4", 4), ("x16", 16)):
            cfg = fg.arm_settings(arm)["cfg"]
            self.assertEqual((cfg.sims, cfg.schedule.high, cfg.schedule.hi), (base.sims * m, base.schedule.high * m, base.schedule.hi * m))
            self.assertEqual((cfg.min_child_sims, cfg.c, cfg.confirm_pairs, cfg.schedule.kind_mult, cfg.schedule.lo),
                             (base.min_child_sims, base.c, base.confirm_pairs, base.schedule.kind_mult, base.schedule.lo))
            self.assertEqual(cfg.sims_for(target), base.sims_for(target) * m)  # no clamp at the top
            self.assertEqual(cfg.sims_for(play), base.sims_for(play) * m)
        self.assertEqual(sc.get("a3-v1").sims_for(target), 120)  # the defaults are untouched


class StartSelectionTests(unittest.TestCase):
    def test_seeded_and_reproducible(self):
        a = fg.select_starts(_runs(), 10, 8, seed=7)
        self.assertEqual(a, fg.select_starts(_runs(), 10, 8, seed=7))
        self.assertNotEqual([s["id"] for s in a], [s["id"] for s in fg.select_starts(_runs(), 10, 8, seed=8)])

    def test_act1_boss_starts_come_from_runs_that_reached_the_boss(self):
        picks = [s for s in fg.select_starts(_runs(), 1000, 0, seed=1) if s["cat"] == "act1-boss"]
        self.assertEqual(len(picks), 70)  # 40 + 20 + 10; not the floor-12 run, the lost run or the crash
        self.assertTrue(all(s["floor"] == 17 and s["kind"] == "boss" and s["act"] == 1 for s in picks))
        self.assertEqual(len({s["id"] for s in picks}), 70)

    def test_act2_starts_are_reached_elites_and_bosses_half_each(self):
        picks = [s for s in fg.select_starts(_runs(), 0, 20, seed=3) if s["act"] == 2]
        self.assertEqual(sum(s["cat"] == "act2-boss" for s in picks), 10)
        self.assertEqual(sum(s["cat"] == "act2-elite" for s in picks), 10)
        for s in picks:
            run = next(r for r in _runs() if r["seed"] == s["seed"])
            self.assertLessEqual(s["floor"], run["floor"])
            self.assertEqual(run["nodes"][s["floor"] - 2], "B" if s["kind"] == "boss" else "E")
        self.assertEqual({s["floor"] for s in picks if s["cat"] == "act2-boss"}, {33})
        self.assertEqual([s["cat"] for s in picks[:2]], ["act2-boss", "act2-elite"])  # interleaved

    def test_a_short_kind_is_filled_from_the_other(self):
        picks = [s for s in fg.select_starts(_runs(), 0, 60, seed=3) if s["act"] == 2]
        self.assertEqual(sum(s["cat"] == "act2-boss" for s in picks), 20)  # only 20 runs reached the Act-2 boss
        self.assertEqual(sum(s["cat"] == "act2-elite" for s in picks), 40)  # 20 x 2 + 10 x 1 = 50 elites on offer


class StandardErrorTests(unittest.TestCase):
    def test_paired_difference_and_both_standard_errors_on_a_fixed_table(self):
        won = {  # (start, j): (a3-v1, x16)
            ("A", 0): (0, 1), ("A", 1): (1, 1), ("B", 0): (0, 0), ("B", 1): (0, 1), ("C", 0): (1, 0), ("C", 1): (0, 1)}
        rows = [{"arm": arm, "start": s, "j": j, "won": v[i]} for (s, j), v in won.items() for i, arm in enumerate(("a3-v1", "x16"))]
        rows.append({"arm": "x16", "start": "D", "j": 0, "won": 1})  # unpaired: left out
        d = fg.paired_diff(rows, "x16", "a3-v1")
        diffs = [1, 0, 0, 1, -1, 1]
        m = sum(diffs) / 6
        se = math.sqrt(sum((x - m) ** 2 for x in diffs) / 5 / 6)
        self.assertAlmostEqual(d["diff"], m)
        self.assertAlmostEqual(d["se_pairs"], se)
        self.assertEqual(d["pairs"], 6)
        per = [0.5, 0.5, 0.0]
        mp = sum(per) / 3
        self.assertAlmostEqual(d["se_starts"], math.sqrt(sum((x - mp) ** 2 for x in per) / 2 / 3))
        self.assertEqual(d["starts"], 3)

    def test_mean_se_edges(self):
        self.assertEqual(fg.mean_se([]), (None, None, 0))
        self.assertEqual(fg.mean_se([2.0]), (2.0, None, 1))
        self.assertEqual(fg.mean_se([1.0, 3.0]), (2.0, 1.0, 2))

    def test_correlations_and_l0(self):
        self.assertAlmostEqual(fg.pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(fg.spearman([1, 2, 3, 4], [1, 4, 9, 100]), 1.0)
        self.assertAlmostEqual(fg.spearman([1, 1, 2], [3, 3, 1]), -1.0)
        self.assertAlmostEqual(fg.mse([0, 1], [1, 1]), 0.5)
        self.assertEqual(fg.l0(False, 50, 2, 80), 0.0)
        self.assertAlmostEqual(fg.l0(True, 40, 0, 80), 0.75)
        self.assertAlmostEqual(fg.l0(True, 75, 2, 80), 1.0)  # capped at one
        r = fg.score_predictions([0.1, 0.5, 0.9, 0.3], {"good": [0.1, 0.5, 0.9, 0.3], "bad": [0.9, 0.5, 0.1, 0.4]}, boot=200)
        self.assertAlmostEqual(r["good"]["pearson"], 1.0)
        self.assertGreater(r["better_share"]["mse"]["good"], 0.9)


class ResumeTests(unittest.TestCase):
    def test_done_games_are_skipped_and_a_cut_line_is_played_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "games.jsonl"
            rows = [{"arm": "x16", "start": "A|f17", "j": 0}, {"arm": "a3-v1", "start": "A|f17", "j": 1}]
            p.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"arm": "x16", "start": "A|f', encoding="utf-8")
            done = fg.done_keys(fg.read_jsonl(p), fg.game_key)
            self.assertEqual(done, {("x16", "A|f17", 0), ("a3-v1", "A|f17", 1)})
            todo = fg.todo_games([{"id": "A|f17"}], "x16", 3, done)
            self.assertEqual(todo, [("A|f17", 1), ("A|f17", 2)])
            self.assertEqual(fg.todo_games([{"id": "A|f17"}], "a3-v1", 2, done), [("A|f17", 0)])


class GuardTests(unittest.TestCase):
    def test_the_night_main_lock_and_other_quiet_lines_block(self):
        noon = dt.datetime(2026, 10, 29, 12, 0, tzinfo=dt.timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp)
            (ref / "backlog").mkdir()
            (ref / "LAB" / "nightly").mkdir(parents=True)
            q = ref / "QUIET.lock"
            with mock.patch.multiple(fg, QUIET=q, MAIN_LOCK=ref / "backlog" / "main.lock", ACTIVE=ref / "LAB" / "nightly" / "active"), \
                    mock.patch.object(fg, "utc", return_value=noon):
                self.assertIsNone(fg.blocked())
                fg.hold(True)
                self.assertTrue(q.read_text().startswith(fg.TAG + " "))
                self.assertIsNone(fg.blocked())  # its own line does not block it
                q.write_text(q.read_text() + "w3-blocks 2026-10-29T11:00Z pid=1\n")
                self.assertIsNone(fg.blocked())
                q.write_text(q.read_text() + "gym-m2 2026-10-29T08:00:00Z pid=2\n")
                self.assertIsNone(fg.blocked())  # over two hours old
                q.write_text(q.read_text() + "gym-m2 2026-10-29T11:30:00Z pid=3\n")
                self.assertEqual(fg.blocked(), "quiet:gym-m2")
                fg.hold(False)
                self.assertNotIn(fg.TAG + " ", q.read_text())
                self.assertIn("pid=3", q.read_text())  # other lines kept
                q.unlink()
                (ref / "backlog" / "main.lock").write_text("x")
                self.assertEqual(fg.blocked(), "main.lock")
                (ref / "backlog" / "main.lock").unlink()
                (ref / "LAB" / "nightly" / "active").write_text("x")
                self.assertEqual(fg.blocked(), "nightly-active")
            with mock.patch.multiple(fg, QUIET=ref / "none", MAIN_LOCK=ref / "none2", ACTIVE=ref / "none3"), \
                    mock.patch.object(fg, "utc", return_value=noon.replace(hour=3, minute=20)):
                self.assertEqual(fg.blocked(), "night-window")


if __name__ == "__main__":
    unittest.main()
