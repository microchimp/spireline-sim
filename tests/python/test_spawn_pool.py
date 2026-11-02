"""rl/spawn_pool.py and gen0.run_block on it: a worker that dies mid-run, a recycle at the run cap, a seed past the wall timeout,
a lost seed's row, resume over existing rows. Real spawned workers, fake tasks (tests/python/pool_fakes.py): no emulator, no game."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
import types
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "rl"):
    sys.path.insert(0, str(p))

import gen0  # noqa: E402
import nightly_real as nr  # noqa: E402
import spawn_pool as sp  # noqa: E402
from tests.python import pool_fakes as pf  # noqa: E402


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.marks = self.tmp / "marks"
        self.marks.mkdir()
        env = unittest.mock.patch.dict(os.environ, {"POOL_FAKE_DIR": str(self.marks), "POOL_FAKE_SLEEP": "0", "POOL_FAKE_TOKEN": "a"})
        env.start()
        self.addCleanup(env.stop)


class PoolTest(Tmp):
    def pool(self, size: int = 1, **kw) -> sp.SupervisedPool:
        pool = sp.SupervisedPool(mp.get_context("spawn"), size, **kw)
        self.addCleanup(pool.terminate)
        return pool

    def finish(self, pool: sp.SupervisedPool, n: int, limit_s: float = 60.0) -> list[sp.Done]:
        out: list[sp.Done] = []
        end = time.time() + limit_s
        while len(out) < n and time.time() < end:
            out.extend(pool.collect())
            pool.wait(0.2)
        return out

    def test_a_worker_that_dies_mid_run_is_reported_and_its_slot_works_again(self):
        pool = self.pool(1)
        first = pool.pids()[0]
        pool.submit("a", pf.die, (3,))
        (d,) = self.finish(pool, 1)
        self.assertEqual(("a", "died"), (d.key, d.status))
        self.assertIn("code 3", d.value)
        self.assertNotEqual(first, pool.pids()[0])  # a fresh worker holds the slot
        pool.submit("b", pf.ok, ("x",))
        (d,) = self.finish(pool, 1)
        self.assertEqual(("b", "ok", "x"), (d.key, d.status, d.value[0]))
        self.assertEqual(1, pool.stats["died"])

    def test_a_run_past_the_wall_timeout_is_killed_and_reported_and_the_slot_works_again(self):
        pool = self.pool(1, run_timeout_s=1.0)
        pool.submit("slow", pf.hang, (120.0,))
        t0 = time.time()
        (d,) = self.finish(pool, 1)
        self.assertEqual("timeout", d.status)
        self.assertLess(time.time() - t0, 30)
        pool.submit("next", pf.ok, ("y",))
        (d,) = self.finish(pool, 1)
        self.assertEqual(("next", "ok"), (d.key, d.status))
        self.assertEqual(1, pool.stats["timed_out"])

    def test_a_task_that_raises_is_reported_and_the_worker_replaced(self):
        pool = self.pool(1)
        pool.submit("r", pf.boom, ())
        (d,) = self.finish(pool, 1)
        self.assertEqual(("raised", "ValueError: a task that raises"), (d.status, d.value))
        pool.submit("n", pf.ok, ())
        self.assertEqual("ok", self.finish(pool, 1)[0].status)

    def test_a_recycle_replaces_a_worker_only_between_runs_and_drops_nothing(self):
        pool = self.pool(1, recycle=2)
        pids = []
        for i in range(5):
            pool.submit(i, pf.ok, ())
            (d,) = self.finish(pool, 1)
            self.assertEqual(("ok", i), (d.status, d.key))  # every run ended with its result
            pids.append(d.value[1])
        self.assertEqual([pids[0], pids[0]], pids[:2])
        self.assertEqual([pids[2], pids[2]], pids[2:4])
        self.assertEqual(3, len(set(pids)))  # replaced after the 2nd and the 4th run
        self.assertEqual(2, pool.stats["recycled"])


class RunBlockTest(Tmp):
    def block(self, seeds, out=None, procs=2, **kw):
        out = out or self.tmp / "arm"
        kw.setdefault("run_timeout_s", 120.0)
        summ = gen0.run_block(seeds, 1, procs, out, agent=pf.FakeAgent(), task=pf.fake_task, initializer=None, **kw)
        rows = [json.loads(x) for x in (out / "runs.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        return summ, rows

    def test_a_worker_that_dies_once_costs_a_rerun_not_the_block(self):
        seeds = ["A1", "DIEONCE1", "A2", "A3"]
        summ, rows = self.block(seeds, run_timeout_s=3.0)
        self.assertEqual(sorted(seeds), sorted(r["seed"] for r in rows))
        self.assertFalse(any(r.get("lost") for r in rows))
        self.assertNotIn("timed_out", summ)
        self.assertEqual(1, summ["pool"]["retried"])
        self.assertEqual((1, 0, 0), (summ["pool"]["died"], summ["pool"]["timed_out"], summ["pool"]["lost"]))

    def test_a_run_past_the_limit_is_recorded_once_as_lost_and_not_run_again(self):
        seeds = ["A1", "HANGONCE1", "A2", "A3"]  # HANGONCE would finish on a second attempt: it must not get one
        summ, rows = self.block(seeds, run_timeout_s=2.0)
        self.assertEqual(sorted(seeds), sorted(r["seed"] for r in rows))  # one row per seed, the hung one once
        row = next(r for r in rows if r["seed"] == "HANGONCE1")
        self.assertEqual(("lost", "timeout", 0, False, 1), (row["end"], row["lost_status"], row["floor"], row["win"], row["attempts"]))
        self.assertEqual((0, 1, 1), (summ["pool"]["retried"], summ["pool"]["timed_out"], summ["pool"]["lost"]))
        (entry,) = summ["timed_out"]  # recorded once, in the block's summary
        self.assertEqual(("HANGONCE1", "arm", 2.0, 0, False), (entry["seed"], entry["arm"], entry["limit_s"], entry["floor"], entry["win"]))
        self.assertGreaterEqual(entry["wall_s"], 2.0)

    def test_the_default_limit_of_an_emulator_run_is_600_seconds(self):
        self.assertEqual(600.0, gen0.RUN_TIMEOUT_S)
        self.assertEqual(600.0, gen0.build_parser().parse_args(["emu", "--seeds-file", "s", "--out", "o"]).run_timeout_s)
        import inspect

        self.assertEqual(gen0.RUN_TIMEOUT_S, inspect.signature(gen0.run_block).parameters["run_timeout_s"].default)

    def test_a_timed_out_seed_counts_as_floor_0_and_no_win_in_the_blocks_numbers(self):
        seeds = ["A1", "A2", "HANGALWAYS1"]  # the fake runs end at floor 5, no win; the hung one counts as a crash row would
        summ, rows = self.block(seeds, run_timeout_s=2.0)
        self.assertEqual((3, 0, 1), (summ["runs"], summ["wins"], summ["lost"]))
        self.assertAlmostEqual(10 / 3, summ["floor"]["mean"])
        self.assertEqual(summ["floor"]["mean"], gen0.engine_block(rows)["floor"]["mean"])
        crash = {"seed": "C", "win": False, "floor": 0, "decisions": 0, "wall": 1.0, "crash": True}  # gen0.real_runs' row of a crash
        self.assertEqual(gen0.engine_block(rows[:2] + [crash])["floor"]["mean"], summ["floor"]["mean"])

    def test_a_seed_that_dies_twice_or_hangs_is_a_lost_row_and_the_block_ends_with_a_row_per_seed(self):
        seeds = ["A1", "DIEALWAYS1", "A2", "HANGALWAYS1", "A3"]
        summ, rows = self.block(seeds, run_timeout_s=2.0)
        self.assertEqual(sorted(seeds), sorted(r["seed"] for r in rows))
        lost = {r["seed"]: r for r in rows if r.get("lost")}
        self.assertEqual({"DIEALWAYS1", "HANGALWAYS1"}, set(lost))
        for r in lost.values():
            self.assertEqual(("lost", 0, False), (r["end"], r["floor"], r["win"]))
        self.assertEqual(2, lost["DIEALWAYS1"]["attempts"])  # a death is run once more
        self.assertEqual(1, lost["HANGALWAYS1"]["attempts"])  # a timeout is not
        self.assertEqual("died", lost["DIEALWAYS1"]["lost_status"])
        self.assertEqual("timeout", lost["HANGALWAYS1"]["lost_status"])
        self.assertEqual(2, summ["lost"])  # engine_block's count
        self.assertEqual(["HANGALWAYS1"], [e["seed"] for e in summ["timed_out"]])
        self.assertEqual(5, summ["runs"])
        self.assertEqual(0, summ["wins"])

    def test_a_recycle_at_the_cap_loses_no_seed(self):
        os.environ["POOL_FAKE_SLEEP"] = "0.05"
        seeds = [f"S{i}" for i in range(9)]
        summ, rows = self.block(seeds, recycle=2)
        self.assertEqual(sorted(seeds), sorted(r["seed"] for r in rows))
        self.assertGreaterEqual(summ["pool"]["recycled"], 3)
        self.assertEqual((0, 0, 0), (summ["pool"]["died"], summ["pool"]["lost"], summ["pool"]["retried"]))

    def test_a_pool_that_cannot_run_anything_stops_instead_of_writing_a_block_of_lost_rows(self):
        with self.assertRaises(RuntimeError) as cm:
            self.block([f"ALLDIE{i}" for i in range(20)])
        self.assertIn("in a row", str(cm.exception))

    def test_a_rerun_with_resume_keeps_the_rows_it_has_and_plays_the_rest(self):
        out = self.tmp / "arm"
        seeds = [f"S{i}" for i in range(6)]
        self.block(seeds[:3], out, resume=True)  # the first pass, token a
        os.environ["POOL_FAKE_TOKEN"] = "b"
        with (out / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write('{"seed": "S5", "ix": 5, "wi')  # the stop cut the last line
        summ, rows = self.block(seeds, out, resume=True)
        self.assertEqual(sorted(seeds), sorted(r["seed"] for r in rows))
        self.assertEqual({"S0": "a", "S1": "a", "S2": "a", "S3": "b", "S4": "b", "S5": "b"}, {r["seed"]: r["token"] for r in rows})
        self.assertEqual(3, summ["resumed_rows"])
        self.assertEqual(6, summ["runs"])

    def test_resume_starts_over_when_the_agent_or_engine_changed_and_without_resume_the_file_is_replaced(self):
        out = self.tmp / "arm"
        seeds = ["S0", "S1", "S2"]
        self.block(seeds, out, resume=True, resume_key={"engine": "old"})
        os.environ["POOL_FAKE_TOKEN"] = "b"
        _, rows = self.block(seeds, out, resume=True, resume_key={"engine": "core2"})
        self.assertEqual({"b"}, {r["token"] for r in rows})
        os.environ["POOL_FAKE_TOKEN"] = "c"
        _, rows = self.block(seeds, out)
        self.assertEqual({"c"}, {r["token"] for r in rows})


class NightlyTest(Tmp):
    def test_the_lock_line_names_its_process_and_still_reads_as_before(self):
        f = self.tmp / "QUIET.lock"
        lock = nr.QuietLock(f)
        self.assertTrue(lock.acquire())
        self.assertRegex(f.read_text(), rf"^lab-nightly \d{{4}}-\d\d-\d\dT\d\d:\d\d:\d\dZ pid={os.getpid()}\n$")
        self.assertEqual("lab-nightly", lock.holder()[0])
        lock.release()
        self.assertFalse(f.exists())

    def test_a_restarted_night_keeps_both_arms_rows_and_resumes_the_real_block(self):
        lab, agent_dir = self.tmp / "lab", self.tmp / "agent"
        date_dir = lab / "nightly" / "2026-10-27"
        (date_dir / "g7-emu").mkdir(parents=True)
        (date_dir / "g7-emu" / "runs.jsonl").write_text('{"seed": "X"}\n')
        (date_dir / "g7").mkdir()
        (date_dir / "g7" / "old.txt").write_text("from the first attempt")
        a = argparse.Namespace(lab=str(lab), runs=5, procs=2, recycle=60, config="a3-v1", stop_after_h=1.0, lib="L", game="G",
                               engine="core2", seeds_also=[])
        ag = {"label": "g7", "generation": 7, "dir": str(agent_dir), "role": "incumbent"}
        seen = {}

        def fake_run(cmd, **kw):
            seen["emu_rows"] = (date_dir / "g7-emu" / "runs.jsonl").is_file()
            seen["real_old"] = (date_dir / "g7" / "old.txt").exists()
            seen["resume"] = "--resume" in cmd  # N3: the real block is resumed in its folder, not cleared and replayed
            seen["no_score_lock"] = "--quiet-lock" not in cmd  # the driver holds lab-nightly itself: the score under it takes no line
            return types.SimpleNamespace(returncode=0)

        with unittest.mock.patch.object(nr.subprocess, "run", side_effect=fake_run):
            nr.play_agent(a, ag, "2026-10-27", date_dir)
        self.assertEqual({"emu_rows": True, "real_old": True, "resume": True, "no_score_lock": True}, seen)


if __name__ == "__main__":
    unittest.main()
