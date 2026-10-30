"""Night streaming (owner, 2026-10-30): the nightly block's emulator arm plays each seed after its real run has ended, beside the
real arm, then the rest at full width (rl/gen0.py RealFeed, StreamOrder, run_block(stream=...), play_streamed; rl/nightly_real.py).
No game, no emulator: the real arm is a fake process or a scripted runs.jsonl, the arm's runs are tests/python/pool_fakes tasks
on real spawned workers."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "rl", ROOT / "referee", ROOT / "src"):
    sys.path.insert(0, str(p))

import gen0  # noqa: E402
import nightly_real as nr  # noqa: E402
import spawn_pool as sp  # noqa: E402
from tests.python import pool_fakes as pf  # noqa: E402

IDLE, BELOW_NORMAL = sp.IDLE_PRIORITY_CLASS, sp.BELOW_NORMAL_PRIORITY_CLASS


def end(seed: str, floor: int = 5) -> dict:
    return {"ev": "end", "seed": seed, "end": "death", "floor": floor, "stats": {"decisions": 3}, "s": 1.0}


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)


# ---------------------------------------------------------------- the feed: which real runs have ended

class FeedTest(Tmp):
    def feed(self) -> gen0.RealFeed:
        return gen0.RealFeed(self.tmp / "runs.jsonl", lambda: True)

    def write(self, text: str, mode: str = "w") -> None:
        with (self.tmp / "runs.jsonl").open(mode, encoding="utf-8", newline="") as fh:
            fh.write(text)

    def test_end_crash_and_hang_lines_end_a_seed_once_in_file_order_and_other_lines_are_ignored(self):
        lines = [{"ev": "ready", "w": 0, "pid": 1}, end("A"), {"ev": "tick", "w": 0, "dec": 200},
                 {"ev": "crash", "w": 1, "seed": "B", "error": "died"},  # a crash row ends the seed's real run
                 {"ev": "crash", "w": 2, "where": "boot", "seed": "X", "error": "no boot"},  # no run: envproto plays it again
                 {"ev": "hang", "w": 3, "seed": "C", "error": "timeout"}, end("A"), {"ev": "sys", "t": 1.0}, {"ev": "start", "seed": "D"}]
        self.write("".join(json.dumps(x) + "\n" for x in lines))
        f = self.feed()
        self.assertEqual(["A", "B", "C"], f.poll())
        self.assertEqual([], f.poll())

    def test_reads_on_from_the_last_read_and_a_half_written_line_waits_for_its_end(self):
        self.write(json.dumps(end("A")) + "\n" + '{"ev": "end", "se')
        f = self.feed()
        self.assertEqual(["A"], f.poll())
        self.write('ed": "B"}\n', "a")
        self.assertEqual(["B"], f.poll())

    def test_a_shorter_file_is_read_again_from_the_top_and_the_final_read_takes_the_unterminated_last_line(self):
        self.write("".join(json.dumps(end(s)) + "\n" for s in "ABC"))
        f = self.feed()
        self.assertEqual(["A", "B", "C"], f.poll())
        self.write(json.dumps(end("A")) + "\n" + json.dumps(end("D")) + "\n")  # a resume's rewrite, then a new row
        self.assertEqual(["D"], f.poll())
        self.write(json.dumps(end("E")), "a")  # the end-of-session rewrite leaves no newline after the last row
        self.assertEqual([], f.poll())
        self.assertEqual(["E"], f.poll(final=True))

    def test_no_file_yet_is_no_seed_yet(self):
        self.assertEqual([], self.feed().poll())


class FakeFeed:
    """A scripted feed: `batches` are what successive polls return; alive turns False after `live_polls` updates."""

    def __init__(self, batches: list[list[str]], live_polls: int, poll_s: float = 0.0):
        self.batches, self.live_polls, self.poll_s = list(batches), live_polls, poll_s
        self.calls, self.finals = 0, 0

    def alive(self) -> bool:
        self.calls += 1
        return self.calls <= self.live_polls

    def poll(self, final: bool = False) -> list[str]:
        self.finals += final
        return self.batches.pop(0) if self.batches else []


class OrderTest(unittest.TestCase):
    TODO = [(0, "A"), (1, "B"), (2, "C"), (3, "D")]

    def drain(self, order: gen0.StreamOrder) -> list[tuple[int, str]]:
        out = []
        while (x := order.pop()) is not None:
            out.append(x)
        return out

    def test_without_a_feed_the_block_order_at_once(self):
        order = gen0.StreamOrder(self.TODO)
        self.assertFalse(order.update(force=True))
        self.assertEqual(self.TODO, self.drain(order))
        self.assertEqual(0, order.left())

    def test_ended_real_runs_first_in_their_order_then_the_rest_in_block_order_each_pair_once(self):
        feed = FakeFeed([["C"], ["Z", "A", "C"], ["D"]], live_polls=2)  # Z is not a seed of the block; C comes twice
        order = gen0.StreamOrder(self.TODO, feed)
        self.assertFalse(order.update(force=True))
        self.assertEqual([(2, "C")], self.drain(order))
        self.assertFalse(order.update())
        self.assertEqual([(0, "A")], self.drain(order))  # while the real arm plays, nothing else: B and D have no real row yet
        self.assertTrue(order.live)
        self.assertTrue(order.update())  # the real arm has ended: its last read (D) first, then B, which it never ended
        self.assertEqual((1, 1), (feed.finals, 0 if order.live else 1))
        self.assertEqual([(3, "D"), (1, "B")], self.drain(order))
        self.assertFalse(order.update())
        self.assertEqual(0, order.left())

    def test_the_feed_is_read_every_poll_s_unless_forced(self):
        now = [100.0]
        feed = FakeFeed([["A"], ["B"]], live_polls=99, poll_s=5.0)
        order = gen0.StreamOrder(self.TODO, feed, clock=lambda: now[0])
        order.update(force=True)
        now[0] += 4.0
        order.update()
        self.assertEqual(1, feed.calls)
        now[0] += 1.0
        order.update()
        self.assertEqual(([(0, "A"), (1, "B")], 2), (self.drain(order), feed.calls))


# ---------------------------------------------------------------- the streamed arm on spawned workers

class Real:
    """A fake real arm, the `alive` of a RealFeed on `path`: each call writes the next scripted line to its runs.jsonl, but with
    `gate` only once the emulator arm (`emu` runs.jsonl) has `need` rows, one more for each seed line it released, so the
    order the arm sees the seeds in is the script's. It has ended (False) once the script is out and the arm caught up."""

    def __init__(self, path: Path, emu: Path, script: list[dict], need: int = 0, gate: bool = True):
        self.path, self.emu, self.script, self.need, self.gate = path, emu, list(script), need, gate

    def __call__(self) -> bool:
        if self.gate and len(read_rows(self.emu)) < self.need:
            return True
        if not self.script:
            return False
        line = self.script.pop(0)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
        if line.get("seed") and line.get("ev") in gen0.REAL_FINISHED:
            self.need += 1
        return True


class StreamBlockTest(Tmp):
    SEEDS = ["S0", "S1", "S2", "S3", "S4", "S5"]
    STAMP = {"agent": {"generation": 7, "label": "fake"}, "sims": 1, "ascension": 10, "character": "IRONCLAD"}

    def block(self, real: Real, procs: int = 2, stream_procs: int = 1, **kw):
        out = self.tmp / "arm"
        feed = gen0.RealFeed(real.path, real, poll_s=0.0)
        kw.setdefault("task", pf.fake_task)
        kw.setdefault("initializer", None)
        kw.setdefault("stream_initializer", None)
        summ = gen0.run_block(self.SEEDS, 1, procs, out, agent=pf.FakeAgent(), run_timeout_s=60.0, stream=feed,
                              stream_procs=stream_procs, stream_floor_gb=kw.pop("floor", 0.0), idle_s=0.01, **kw)
        return summ, read_rows(out / "runs.jsonl")

    def real(self, script: list[dict], need: int = 0, gate: bool = True) -> Real:
        (self.tmp / "real").mkdir(exist_ok=True)
        return Real(self.tmp / "real" / "runs.jsonl", self.tmp / "arm" / "runs.jsonl", script, need, gate)

    def test_each_seed_once_with_its_block_index_beside_the_real_arm_in_its_order_then_the_rest(self):
        script = [end("S3"), {"ev": "crash", "w": 1, "seed": "S1", "error": "died"}, {"ev": "tick", "w": 0},
                  {"ev": "hang", "w": 2, "seed": "S4", "error": "t"}]
        summ, rows = self.block(self.real(script))
        self.assertEqual(sorted(self.SEEDS), sorted(r["seed"] for r in rows))  # every seed exactly once
        self.assertEqual({r["seed"]: self.SEEDS.index(r["seed"]) for r in rows}, {r["seed"]: r["ix"] for r in rows})  # its block index
        self.assertEqual(["S3", "S1", "S4"], [r["seed"] for r in rows[:3]])  # beside: as the real runs ended (a crash, a hang too)
        self.assertEqual({"S0", "S2", "S5"}, {r["seed"] for r in rows[3:]})  # the seeds the real arm never ended, after it
        st = summ["stream"]
        self.assertEqual((1, 3, 0), (st["beside_procs"], st["rows_beside"], st["memory_waits"]))
        self.assertIsNotNone(st["real_ended_s"])
        self.assertEqual(1, len({r["pid"] for r in rows[:3]}))  # one worker beside the real arm
        self.assertEqual((2, 6), (summ["procs"], summ["runs"]))

    def test_a_stopped_block_restarted_in_its_folder_plays_its_real_runs_without_an_emulator_row_first(self):
        arm = self.tmp / "arm"
        arm.mkdir()
        (arm / "resume.json").write_text(json.dumps(self.STAMP), encoding="utf-8")
        held = {"seed": "S0", "ix": 0, "win": False, "floor": 9, "end": "death", "decisions": 4, "wall": 1.0, "token": "before"}
        (arm / "runs.jsonl").write_text(json.dumps(held) + "\n" + '{"seed": "S5", "ix"', encoding="utf-8")  # the stop cut a line
        real = self.real([end("S1")], need=3)  # the restarted real arm ends S1, then stops
        real.path.write_text("".join(json.dumps(end(s)) + "\n" for s in ("S0", "S2", "S4")), encoding="utf-8")  # before the stop
        with unittest.mock.patch.dict(os.environ, {"POOL_FAKE_TOKEN": "after"}):
            summ, rows = self.block(real, resume=True)
        self.assertEqual(sorted(self.SEEDS), sorted(r["seed"] for r in rows))
        self.assertEqual(held, rows[0])  # kept, not played again
        self.assertEqual(["S2", "S4", "S1"], [r["seed"] for r in rows[1:4]])  # no emulator row at the stop: played first
        self.assertEqual({"S3", "S5"}, {r["seed"] for r in rows[4:]})
        self.assertEqual({"after"}, {r["token"] for r in rows[1:]})
        self.assertEqual((1, 3), (summ["resumed_rows"], summ["stream"]["rows_beside"]))

    def test_new_runs_wait_for_memory_headroom_only_while_the_real_arm_plays(self):
        real = self.real([end("S0"), end("S1"), {"ev": "tick"}, {"ev": "tick"}], gate=False)
        with unittest.mock.patch.dict(os.environ, {"SPIRELINE_FAKE_MEM_HEADROOM_GB": "1.0"}):
            summ, rows = self.block(real, floor=3.0)
        self.assertEqual(sorted(self.SEEDS), sorted(r["seed"] for r in rows))
        self.assertEqual((0, 1), (summ["stream"]["rows_beside"], summ["stream"]["memory_waits"]))  # all of them after the real arm

    def test_a_streamed_arm_refuses_a_prior_server(self):
        feed = gen0.RealFeed(self.tmp / "runs.jsonl", lambda: True)
        with self.assertRaises(ValueError):
            gen0.run_block(self.SEEDS, 1, 2, self.tmp / "arm", agent=pf.FakeAgent(), stream=feed, prior_server=object())

    @unittest.skipUnless(sys.platform == "win32", "Windows priority classes")
    def test_the_beside_workers_run_idle_and_every_run_after_the_real_arm_runs_below_normal(self):
        summ, rows = self.block(self.real([end("S2"), end("S0")]), task=pf.prio_task, initializer=gen0._init_worker,
                                stream_initializer=gen0._init_stream_worker)
        self.assertEqual(["S2", "S0"], [r["seed"] for r in rows[:2]])
        self.assertEqual([IDLE, IDLE], [r["prio"] for r in rows[:2]])
        self.assertEqual([BELOW_NORMAL] * 4, [r["prio"] for r in rows[2:]])  # the beside worker raised, the new one started so


class PoolGrowTest(unittest.TestCase):
    def test_grow_starts_the_new_slots_and_reprioritize_moves_every_live_worker(self):
        pool = sp.SupervisedPool(mp.get_context("spawn"), 1, initializer=None)
        try:
            pool.grow(2)
            pool.grow(1)  # never narrows
            pids = pool.pids()
            self.assertEqual(2, len(set(pids)))
            n = pool.reprioritize(BELOW_NORMAL, None)
            if sys.platform == "win32":
                from sts2_gym import procutil

                self.assertEqual(2, n)
                self.assertEqual([BELOW_NORMAL] * 2, [procutil.priority_class(p) for p in pids])
        finally:
            pool.terminate()


# ---------------------------------------------------------------- score_real: the two arms at once, GPU_OK

class FakeProc:
    """envproto play's process: alive for `polls` polls, then exited with `rc`."""
    made: list[FakeProc] = []

    def __init__(self, cmd, env=None, polls: int = 3, rc: int = 0):
        self.cmd, self.polls, self.rc = cmd, polls, rc
        self.returncode, self.killed, self.waited = None, False, False
        FakeProc.made.append(self)

    def poll(self):
        if self.returncode is None:
            self.polls -= 1
            if self.polls < 0:
                self.returncode = self.rc
        return self.returncode

    def wait(self):
        self.waited = True
        if self.returncode is None:
            self.returncode = -9 if self.killed else self.rc
        return self.returncode

    def kill(self):
        self.killed = True


class ScoreStreamTest(Tmp):
    def run_score(self, raise_in_arm: BaseException | None = None, **extra):
        gpu = self.tmp / "GPU_OK"
        gpu.write_text("other-lane\n", encoding="utf-8")
        read = lambda: gpu.read_text(encoding="utf-8") if gpu.is_file() else None  # noqa: E731
        seeds_file = self.tmp / "N-test.json"
        seeds = gen0.write_seed_file("N-test", 2, seeds_file)
        out = self.tmp / "real"
        self.seen: list[tuple[str, str | None]] = []
        self.kw: dict = {}
        FakeProc.made = []

        def fake_block(sd, sims, procs, arm_out, **kw):
            self.kw = {"seeds": list(sd), "procs": procs, **kw}
            self.seen.append(("arm starts; real started", str(len(FakeProc.made))))
            if raise_in_arm is not None:
                raise raise_in_arm
            feed = kw["stream"]
            while feed.alive():
                self.seen.append(("real plays", read()))
            self.seen.append(("real ended, arm plays on", read()))
            arm_out.mkdir(parents=True, exist_ok=True)
            (arm_out / "runs.jsonl").write_text("".join(json.dumps({"seed": s, "ix": i, "win": False, "floor": 3, "decisions": 2, "wall": 1.0})
                                                        + "\n" for i, s in enumerate(sd)), encoding="utf-8")
            return {"runs": len(sd)}

        def fake_popen(cmd, env=None):
            out.mkdir(parents=True, exist_ok=True)
            (out / "runs.jsonl").write_text("".join(json.dumps(end(s, 7)) + "\n" for s in seeds), encoding="utf-8")
            (out / "summary.json").write_text(json.dumps({"divergence_rate": 0.01, "wall_s": 9.0}), encoding="utf-8")
            return FakeProc(cmd, env)

        a = types.SimpleNamespace(out=str(out), game="G", seeds_file=str(seeds_file), runs=2, procs=12, sims=60, config="a3-v1",
                                  ascension=10, character="IRONCLAD", stop_after_h=1.0, lib=None, emu_dir=None, gen_dir=None, only=None,
                                  gpu_ok_file=str(gpu), gpu_ok_tag="lab-nightly", emu_stream=True, emu_stream_procs=3,
                                  emu_stream_floor_gb=2.5, **extra)
        with unittest.mock.patch("subprocess.Popen", fake_popen), unittest.mock.patch.object(gen0, "run_block", fake_block), \
                unittest.mock.patch("subprocess.run", side_effect=AssertionError("the real arm ran through subprocess.run")):
            if raise_in_arm is not None:
                with self.assertRaises(type(raise_in_arm)):
                    gen0.score_real(a)
                return None, read(), out
            rc = gen0.score_real(a)
        return rc, read(), out

    def test_the_line_stands_while_the_real_arm_plays_and_goes_when_it_ends_while_the_arm_plays_on(self):
        rc, after, out = self.run_score()
        self.assertEqual(0, rc)
        self.assertEqual(("arm starts; real started", "1"), self.seen[0])  # the real arm first, then the arm beside it
        plays = [g for what, g in self.seen if what == "real plays"]
        self.assertTrue(plays)
        self.assertEqual({"lab-nightly\nother-lane\n"}, set(plays))
        self.assertEqual(("real ended, arm plays on", "other-lane\n"), self.seen[-1])
        self.assertEqual("other-lane\n", after)
        self.assertEqual((3, 2.5, 12, out / "runs.jsonl"), (self.kw["stream_procs"], self.kw["stream_floor_gb"], self.kw["procs"],
                                                            self.kw["stream"].path))
        self.assertEqual(gen0.load_seeds(self.tmp / "N-test.json")[:2], self.kw["seeds"])  # the same seeds as the arm played first
        self.assertTrue(self.kw["resume"])
        comp = json.loads((out / "compare.json").read_text(encoding="utf-8"))
        for key in ("paired_seeds", "emu", "real", "G", "floor_gap", "divergence_rate", "emu_arm"):
            self.assertIn(key, comp)
        self.assertEqual((2, True, True, "real-emu"), (comp["paired_seeds"], comp["emu_arm"]["played"], comp["emu_arm"]["streamed"],
                                                       comp["emu_arm"]["dir"]))
        self.assertIn("emu_arm", json.loads((out / "summary.json").read_text(encoding="utf-8")))
        self.assertTrue(FakeProc.made[0].waited)

    def test_an_arm_that_raises_waits_for_the_real_arm_then_raises_and_the_line_goes(self):
        _, after, _ = self.run_score(RuntimeError("the pool failed"))
        proc = FakeProc.made[0]
        self.assertEqual((True, False, 0), (proc.waited, proc.killed, proc.returncode))
        self.assertEqual("other-lane\n", after)

    def test_an_interrupt_kills_the_real_arm_and_the_line_goes(self):
        _, after, _ = self.run_score(KeyboardInterrupt())
        self.assertTrue(FakeProc.made[0].killed)
        self.assertEqual("other-lane\n", after)

    def test_without_the_flag_the_arm_is_played_first_through_subprocess_run(self):
        calls = []
        a = types.SimpleNamespace(out=str(self.tmp / "real"), game="G", seeds_file="F.json", runs=2, procs=4, sims=60, config="a3-v1",
                                  ascension=10, character="IRONCLAD", stop_after_h=1.0, lib=None, emu_dir=None, gen_dir=None, only=None)
        with unittest.mock.patch("subprocess.run", lambda cmd, **kw: calls.append("real") or types.SimpleNamespace(returncode=0)), \
                unittest.mock.patch("subprocess.Popen", side_effect=AssertionError("streamed")), \
                unittest.mock.patch.object(gen0, "play_emu_arm", lambda *x, **k: calls.append("arm")):
            gen0.score_real(a)
        self.assertEqual(["arm", "real"], calls)

    def test_the_parser_takes_the_stream_options(self):
        base = ["score", "--game", "G", "--seeds-file", "F", "--out", "O"]
        a = gen0.build_parser().parse_args([*base, "--emu-stream", "--emu-stream-procs", "3", "--emu-stream-floor-gb", "2"])
        self.assertEqual((True, 3, 2.0), (a.emu_stream, a.emu_stream_procs, a.emu_stream_floor_gb))
        b = gen0.build_parser().parse_args(base)
        self.assertEqual((False, 4, 3.0), (b.emu_stream, b.emu_stream_procs, b.emu_stream_floor_gb))


class ReplaceRetryTest(Tmp):
    @unittest.skipUnless(sys.platform == "win32", "a rename over an open file fails on Windows only")
    def test_the_real_arms_resume_rename_waits_out_a_brief_reader(self):
        import envproto

        dst, src = self.tmp / "runs.jsonl", self.tmp / "runs.jsonl.tmp"
        dst.write_text("old\n", encoding="utf-8")
        src.write_text("new\n", encoding="utf-8")
        fh = dst.open("rb")
        with self.assertRaises(PermissionError):
            envproto.replace_retry(src, dst, tries=1)
        t = threading.Timer(0.3, fh.close)
        t.start()
        envproto.replace_retry(src, dst, tries=50, wait_s=0.05)
        t.join()
        self.assertEqual("new\n", dst.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the nightly driver

class NightlyStreamTest(Tmp):
    def run_driver(self, *extra: str, emu_rows: int | None = None, promote: bool = False):
        lab, agent = self.tmp / "lab", self.tmp / "agent"
        if not (agent / "current.json").is_file():
            (agent / "gen3" / "c").mkdir(parents=True)
            (agent / "current.json").write_text(json.dumps({"generation": 3, "dir": str(agent / "gen3" / "c")}))
            if promote:
                (agent / "gen4" / "c").mkdir(parents=True)
                (agent / "promotions.tsv").write_text("written\tcand_gen\tbase_gen\tseed_block\tseeds\tdecision\tcand_sha\tcand_dir\n"
                                                     f"2026-10-25T01:00:00Z\t4\t3\tE\t2000\tpromoted\tsha\t{agent / 'gen4' / 'c'}\n")
        self.events: list[str] = []
        self.cmds: list[list[str]] = []

        def fake_run(cmd, **kw):
            out = Path(cmd[cmd.index("--out") + 1])
            self.cmds.append(cmd)
            self.events.append("start " + out.name)
            out.mkdir(parents=True, exist_ok=True)
            seeds = gen0.load_seeds(cmd[cmd.index("--seeds-file") + 1])
            (out / "runs.jsonl").write_text("\n".join(json.dumps(end(s, 6)) for s in seeds))
            (out / "summary.json").write_text(json.dumps({"divergence_rate": 0.0, "wall_s": 5.0}))
            emu = out.with_name(out.name + "-emu")
            emu.mkdir(exist_ok=True)
            n = len(seeds) if emu_rows is None else emu_rows
            (emu / "runs.jsonl").write_text("".join(json.dumps({"seed": s, "ix": i, "win": False, "floor": 3}) + "\n"
                                                    for i, s in enumerate(seeds[:n])))
            self.events.append("end " + out.name)
            return types.SimpleNamespace(returncode=0)

        argv = ["--lab", str(lab), "--agent-root", str(agent), "--game", "G", "--lib", "L", "--out", str(self.tmp / "job"), "--runs", "4",
                "--date", "2026-10-25", "--quiet-lock", str(self.tmp / "QUIET.lock"), "--gpu-ok-file", str(self.tmp / "GPU_OK"),
                "--since", "2026-10-24T00:00:00Z", *extra]
        with unittest.mock.patch.object(nr.subprocess, "run", side_effect=fake_run):
            rc = nr.main(argv)
        return rc, lab

    def test_each_agent_in_turn_streams_its_arm_beside_its_real_arm(self):
        rc, lab = self.run_driver(promote=True)
        self.assertEqual(0, rc)
        self.assertEqual(["start g3", "end g3", "start g4", "end g4"], self.events)  # the incumbent first; one block at a time
        for cmd in self.cmds:
            self.assertIn("--emu-stream", cmd)
            self.assertEqual(("4", "3"), (cmd[cmd.index("--emu-stream-procs") + 1], cmd[cmd.index("--emu-stream-floor-gb") + 1]))
        for label in ("g3", "g4"):
            self.assertIn("nightly", json.loads((lab / "nightly" / "2026-10-25" / label / "summary.json").read_text()))

    def test_zero_stream_procs_plays_the_arm_first_as_before(self):
        rc, lab = self.run_driver("--emu-stream-procs", "0")
        self.assertEqual(0, rc)
        self.assertNotIn("--emu-stream", self.cmds[0])
        self.assertNotIn("--emu-stream-procs", self.cmds[0])

    def test_a_block_whose_arm_stopped_short_is_left_unfinished_and_a_later_start_finishes_it(self):
        rc, lab = self.run_driver(emu_rows=3)
        folder = lab / "nightly" / "2026-10-25" / "g3"
        self.assertNotIn("nightly", json.loads((folder / "summary.json").read_text()))
        res = json.loads((self.tmp / "job" / "result.json").read_text())
        self.assertEqual(1, res["agents"]["g3"]["emu_missing"])
        self.assertEqual("2026-10-25", nr.unfinished_night(lab))  # the next start resumes this night
        rc, lab = self.run_driver()
        self.assertIn("nightly", json.loads((folder / "summary.json").read_text()))
        self.assertIsNone(nr.unfinished_night(lab))

    def test_emu_missing_counts_the_seeds_without_a_row(self):
        emu = self.tmp / "g3-emu"
        emu.mkdir()
        (emu / "runs.jsonl").write_text(json.dumps({"seed": "A"}) + "\n" + json.dumps({"seed": "B", "lost": True}) + "\n" + '{"seed": "C"')
        self.assertEqual((1, 0, 2), (nr.emu_missing(emu, ["A", "B", "C"]), nr.emu_missing(emu, ["A", "B"]),
                                     nr.emu_missing(self.tmp / "none", ["A", "B"])))


if __name__ == "__main__":
    unittest.main()
