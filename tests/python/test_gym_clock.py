"""The fight-gym daemon's clock (rl/gym_clock.py and its use in rl/gym_loop.py; the gate lane's item 3, 2026-11-11).

The pinned facts: a wait logs its reason when it starts and its duration when it ends, once for a wait that goes on and again when the reason
changes; the yield between steps is timed but not logged; every interval is a row split at 00:00Z; the status file carries `waiting_s_today`,
`playing_s_today` and the waits by reason; past midnight one `daily` line (fights, segments, gates, seconds played and waited by reason) is
written once, also for a day finished while the daemon was down; a step the daemon plays and a nested one (the human bar inside the gate)
are timed without double counting; a card refused is said and counted. The clock is injected: no test waits.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gym_clock as gc  # noqa: E402
import gym_loop as gl  # noqa: E402

T0 = dt.datetime(2026, 11, 12, 10, 0, 0, tzinfo=dt.timezone.utc)


class Clock:
    def __init__(self, t: dt.datetime = T0) -> None:
        self.t = t

    def __call__(self) -> dt.datetime:
        return self.t

    def adv(self, seconds: float) -> None:
        self.t += dt.timedelta(seconds=seconds)


def ledger(tmp: Path, clock: Clock, logs: list[str]) -> gc.TimeLedger:
    return gc.TimeLedger(tmp / "clock.jsonl", logs.append, now=clock, segments=tmp / "segments.jsonl", ledger=tmp / "verdicts.jsonl")


def rows_of(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


class FormatTests(unittest.TestCase):
    def test_durations(self):
        self.assertEqual([gc.dur(x) for x in (0, 37, 60, 245, 3599, 3720, 7384)], ["0s", "37s", "1m00s", "4m05s", "59m59s", "1h02m", "2h03m"])
        self.assertEqual(gc.hours(5400), "1.5 h")


class WaitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.clock = Clock()
        self.logs: list[str] = []
        self.led = ledger(self.t, self.clock, self.logs)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_wait_logs_its_reason_at_the_start_and_its_duration_at_the_end(self):
        self.led.wait("quiet:fix-lab")
        self.clock.adv(860)
        self.led.clear()
        self.assertEqual(self.logs, ["wait start: quiet:fix-lab", "wait end: quiet:fix-lab after 14m20s"])
        (r,) = [x for x in rows_of(self.t / "clock.jsonl") if "kind" in x]
        self.assertEqual((r["kind"], r["what"], r["s"], r["t0"], r["t1"]), ("wait", "quiet:fix-lab", 860.0, "2026-11-12T10:00:00Z", "2026-11-12T10:14:20Z"))

    def test_a_wait_that_goes_on_is_one_wait_and_a_new_reason_ends_the_old_one(self):
        for _ in range(4):  # the daemon's 30-second polls
            self.led.wait("main.lock")
            self.clock.adv(30)
        self.led.wait("night-window")
        self.clock.adv(600)
        self.led.clear()
        self.assertEqual(self.logs, ["wait start: main.lock", "wait end: main.lock after 2m00s", "wait start: night-window", "wait end: night-window after 10m00s"])
        self.assertEqual(self.led.totals()["wait"], {"main.lock": 120.0, "night-window": 600.0})

    def test_clear_with_no_wait_open_logs_nothing_and_stop_is_not_a_wait(self):
        self.led.clear()
        self.assertEqual(self.logs, [])
        d = gl.Daemon.__new__(gl.Daemon)
        d.clock = self.led
        d._block("stop")
        self.assertIsNone(self.led.cur)

    def test_the_yield_between_steps_is_timed_and_not_logged(self):
        self.led.wait("yield", log=False)
        self.clock.adv(150)
        self.led.clear()
        self.assertEqual(self.logs, [])
        self.assertEqual(self.led.totals()["wait"], {"yield": 150.0})

    def test_publish_is_a_wait_with_its_own_lines(self):
        with self.led.span("wait", "publish", log=True):
            self.clock.adv(41)
        self.assertEqual(self.logs, ["wait start: publish", "wait end: publish after 41s"])

    def test_the_totals_and_the_status_fields_count_the_open_interval(self):
        with self.led.span("play", "generate"):
            self.clock.adv(1200)
        self.led.wait("quiet:lab-score")
        self.clock.adv(300)
        f = self.led.status_fields()
        self.assertEqual((f["playing_s_today"], f["waiting_s_today"], f["waits_today"]), (1200.0, 300.0, {"quiet:lab-score": 300.0}))
        self.clock.adv(100)
        self.assertEqual(self.led.status_fields()["waiting_s_today"], 400.0)


class SpanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = Clock()
        self.logs: list[str] = []
        self.led = ledger(Path(self.tmp.name), self.clock, self.logs)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_nested_step_interrupts_the_outer_one_and_nothing_is_counted_twice(self):
        with self.led.span("play", "gate"):
            self.clock.adv(600)
            with self.led.span("play", "human-bar"):
                self.clock.adv(300)
            self.clock.adv(120)
        t = self.led.totals()
        self.assertEqual(t["play"], {"gate": 720.0, "human-bar": 300.0})
        self.assertEqual(t["play_s"], 1020.0)
        self.assertIsNone(self.led.cur)

    def test_a_span_inside_a_wait_resumes_the_wait_without_logging_it_twice(self):
        self.led.wait("quiet:x")
        self.clock.adv(60)
        with self.led.span("play", "starts"):
            self.clock.adv(60)
        self.clock.adv(60)
        self.led.clear()
        self.assertEqual(self.logs, ["wait start: quiet:x", "wait end: quiet:x after 1m00s"])  # the resumed piece is not a new wait

    def test_a_step_that_raises_still_ends_its_interval(self):
        with self.assertRaises(RuntimeError), self.led.span("play", "generate"):
            self.clock.adv(10)
            raise RuntimeError("boom")
        self.assertEqual(self.led.totals()["play_s"], 10.0)
        self.assertIsNone(self.led.cur)

    def test_the_loop_times_nothing_without_a_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = gl.Loop(Path(tmp) / "loop")
            with loop.span("wait", "publish", log=True):
                pass
            self.assertIsNone(loop.clock)
            loop.clock = self.led
            with loop.span("play", "generate"):
                self.clock.adv(5)
            self.assertEqual(self.led.totals()["play_s"], 5.0)


class DayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.clock = Clock(dt.datetime(2026, 11, 12, 23, 50, 0, tzinfo=dt.timezone.utc))
        self.logs: list[str] = []
        self.led = ledger(self.t, self.clock, self.logs)

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_interval_over_midnight_is_two_rows_and_one_wait_in_the_log(self):
        self.led.wait("night-window")
        self.clock.adv(1800)  # to 00:20
        self.led.clear()
        got = [(r["t0"][:10], r["s"]) for r in rows_of(self.t / "clock.jsonl") if "kind" in r]
        self.assertEqual(got, [("2026-11-12", 600.0), ("2026-11-13", 1200.0)])
        self.assertEqual(self.logs[-1], "wait end: night-window after 30m00s")
        self.assertEqual(self.led.totals(dt.date(2026, 11, 12))["wait_s"], 600.0)
        self.assertEqual(self.led.totals(dt.date(2026, 11, 13))["wait_s"], 1200.0)

    def test_the_daily_line_is_written_once_past_midnight_while_the_wait_goes_on(self):
        with open(self.t / "segments.jsonl", "w", encoding="utf-8") as fh:
            for i, (t, n) in enumerate([("2026-11-12T08:00:00Z", 190), ("2026-11-12T22:00:00Z", 200), ("2026-11-13T00:30:00Z", 99)]):
                fh.write(json.dumps({"t": t, "segment": i, "fights": n}) + "\n")
        with open(self.t / "verdicts.jsonl", "w", encoding="utf-8") as fh:
            for t, st, v in [("2026-11-12T09:00:00Z", 1, "reject"), ("2026-11-12T12:00:00Z", 2, "promote"), ("2026-11-13T01:00:00Z", 1, "reject")]:
                fh.write(json.dumps({"t": t, "stage": st, "verdict": v}) + "\n")
        self.clock.t = dt.datetime(2026, 11, 12, 6, 0, 0, tzinfo=dt.timezone.utc)
        with self.led.span("play", "generate"):
            self.clock.adv(3 * 3600)
        self.led.note("gpu-denied:lab-nightly")
        self.led.note("gpu-denied:lab-nightly")
        self.led.wait("quiet:fix-lab")
        self.clock.adv(2 * 3600)  # 11:00
        self.led.wait("main.lock")
        self.clock.t = dt.datetime(2026, 11, 12, 23, 55, 0, tzinfo=dt.timezone.utc)  # the main.lock wait goes on through the day
        self.assertEqual(self.led.roll(), [])  # not midnight yet
        self.clock.t = dt.datetime(2026, 11, 13, 0, 0, 30, tzinfo=dt.timezone.utc)
        lines = self.led.roll()
        self.assertEqual(len(lines), 1)
        line = lines[0]
        self.assertTrue(line.startswith("daily 2026-11-12: 2 segments, 390 fights played; played 3.0 h (generate 3.0 h), waited 15.0 h"), line)
        self.assertIn("quiet:fix-lab 2.0 h", line)
        self.assertIn("main.lock 13.0 h", line)  # 11:00 to midnight: the poll at 23:55 does not break the wait
        self.assertIn("2 gated (stage 1 reject 1, stage 2 promote 1)", line)
        self.assertIn("gpu-denied:lab-nightly x2", line)
        self.assertEqual(self.logs[-1], line)
        self.assertEqual(self.led.roll(), [])  # once
        self.clock.adv(600)
        self.led.clear()
        self.assertEqual(self.logs[-1], "wait end: main.lock after 13h10m")  # the whole wait, not the piece after midnight
        self.assertEqual(self.led.totals()["wait"], {"main.lock": 630.0})  # today's piece only

    def test_a_day_finished_while_the_daemon_was_down_gets_its_line_at_the_next_start_once(self):
        self.led.begin("play", "generate")
        self.clock.adv(600)
        self.led.end()
        self.clock.t = dt.datetime(2026, 11, 14, 9, 0, 0, tzinfo=dt.timezone.utc)  # restarted two days later
        led2 = ledger(self.t, self.clock, self.logs)
        self.assertEqual(led2.pending_days(), [dt.date(2026, 11, 12)])
        lines = led2.roll()
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("daily 2026-11-12: 0 segments, 0 fights played; played 0.2 h (generate 0.2 h), waited 0.0 h"), lines[0])
        led3 = ledger(self.t, self.clock, self.logs)
        self.assertEqual((led3.pending_days(), led3.roll()), ([], []))

    def test_a_restart_keeps_the_days_totals(self):
        self.clock.t = dt.datetime(2026, 11, 12, 6, 0, 0, tzinfo=dt.timezone.utc)
        with self.led.span("play", "generate"):
            self.clock.adv(1000)
        led2 = ledger(self.t, self.clock, self.logs)
        led2.wait("pause")
        self.clock.adv(500)
        self.assertEqual(led2.status_fields(), {"waiting_s_today": 500.0, "playing_s_today": 1000.0, "waits_today": {"pause": 500.0}})

    def test_a_cut_line_in_the_file_is_skipped(self):
        (self.t / "clock.jsonl").write_text('{"t0": "2026-11-12T10:00:00Z", "t1": "2026-11-12T10:10:00Z", "kind": "wait", "what": "pause", "s": 600.0}\n{"t0"', encoding="utf-8")
        led = ledger(self.t, self.clock, self.logs)
        self.assertEqual(led.totals(dt.date(2026, 11, 12))["wait"], {"pause": 600.0})


class DaemonWaitTests(unittest.TestCase):
    """`wait_free`, `refresh_status`, the card refused: the daemon's own use of the clock, with time and the locks faked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.loop = gl.Loop(self.t / "loop")
        self.clock = Clock()
        self.led = ledger(self.t, self.clock, [])
        self.led.log = self.loop.log
        self.loop.clock = self.led
        self.status_dir = self.t / "status"

    def tearDown(self):
        self.tmp.cleanup()

    def daemon(self, **kw) -> gl.Daemon:
        d = gl.Daemon.__new__(gl.Daemon)
        d.a = argparse.Namespace(yield_s=150.0, **kw)
        d.loop = self.loop
        d.clock = self.led
        d.phase, d.commit, d.server_pid, d.trainer_pid = "start", "abc", None, None
        return d

    def test_waiting_between_steps_logs_each_reason_and_its_duration_and_the_status_carries_the_seconds(self):
        d = self.daemon()
        now = [1000.0]
        why = iter(["quiet:lab-score", "quiet:lab-score", "night-window", None])
        statuses: list[dict] = []

        def sleep(n: float) -> None:
            now[0] += n
            self.clock.adv(n)

        def fake_status(phase, **kw):
            statuses.append({"phase": phase, **kw, **self.led.status_fields()})

        d.status = fake_status
        with mock.patch.object(gl.time, "time", lambda: now[0]), mock.patch.object(gl.time, "sleep", sleep), \
                mock.patch.object(gl, "why_blocked", lambda *a, **k: next(why)), mock.patch.object(gl, "hold_quiet", lambda on: None):
            d.wait_free()
        log = (self.loop.root / "loop.log").read_text(encoding="utf-8").splitlines()
        text = "\n".join(log)
        self.assertIn("wait start: quiet:lab-score", text)
        self.assertIn("wait end: quiet:lab-score after 1m00s", text)  # two 30-second polls
        self.assertIn("wait start: night-window", text)
        self.assertIn("wait end: night-window after 30s", text)
        self.assertNotIn("wait start: yield", text)  # the fixed pause is timed, not logged
        self.assertEqual(self.led.totals()["wait"], {"night-window": 30.0, "quiet:lab-score": 60.0, "yield": 150.0})
        self.assertEqual([s["phase"] for s in statuses], ["yield", "waiting", "waiting", "waiting"])
        self.assertEqual(statuses[-1]["waiting_s_today"], 150.0 + 60.0)  # the third poll: the night window is 0 s old
        self.assertIsNone(self.led.cur)  # the wait is over: nothing open

    def test_no_clock_no_change(self):
        d = self.daemon()
        d.clock = None
        d.status = lambda *a, **k: None
        now = [0.0]

        def sleep(n: float) -> None:
            now[0] += n

        with mock.patch.object(gl.time, "time", lambda: now[0]), mock.patch.object(gl.time, "sleep", sleep), \
                mock.patch.object(gl, "why_blocked", lambda *a, **k: None), mock.patch.object(gl, "hold_quiet", lambda on: None):
            d.wait_free()  # nothing blocks: it returns, timing nothing
        self.assertEqual(now[0], 150.0)  # the yield alone
        self.assertEqual(self.led.totals()["wait_s"], 0.0)

    def test_the_status_file_has_the_days_seconds(self):
        d = self.daemon()
        with self.led.span("play", "generate"):
            self.clock.adv(700)
        self.led.wait("pause")
        self.clock.adv(50)
        with mock.patch.object(gl, "STATUS_DIR", self.status_dir):
            d.status("waiting", why="pause")
        doc = json.loads((self.status_dir / "status").read_text(encoding="utf-8"))
        self.assertEqual((doc["phase"], doc["why"], doc["playing_s_today"], doc["waiting_s_today"], doc["waits_today"]),
                         ("waiting", "pause", 700.0, 50.0, {"pause": 50.0}))
        self.assertEqual(doc["incumbent"], "v0")

    def test_a_status_without_a_clock_has_no_seconds(self):
        d = self.daemon()
        d.clock = None
        with mock.patch.object(gl, "STATUS_DIR", self.status_dir):
            d.status("start")
        doc = json.loads((self.status_dir / "status").read_text(encoding="utf-8"))
        self.assertNotIn("waiting_s_today", doc)

    def test_the_watch_thread_rolls_the_day_and_refreshes_the_status(self):
        d = self.daemon()
        calls: list[str] = []
        d.clock = mock.Mock(roll=lambda: calls.append("roll"))
        d.refresh_status = lambda: calls.append("status")
        w = gl.ClockWatch(d)
        waits = iter([False, True])
        w._stop = mock.Mock(wait=lambda s: next(waits))
        w._run()
        self.assertEqual(calls, ["roll", "status"])
        w2 = gl.ClockWatch(d)
        self.assertIsInstance(w2.thread, threading.Thread)
        self.assertTrue(w2.thread.daemon)

    def test_a_card_held_by_another_is_said_and_counted_not_silent(self):
        gpu_ok = self.t / "GPU_OK"
        gpu_ok.write_text("lab-nightly 2026-11-12T03:20:00Z\ngym-loop\n", encoding="utf-8")
        with mock.patch.object(gl, "GPU_OK", gpu_ok), mock.patch.dict(gl.os.environ, {}, clear=False):
            gl.os.environ.pop("GYM_LOOP_NO_GPU", None)
            s = gl.Server(self.loop, self.t, "gate-c0001")
        self.assertIsNone(s.handle)
        self.assertIn("gpu: gate-c0001: the card is held by lab-nightly; the CPU prior plays", (self.loop.root / "loop.log").read_text(encoding="utf-8"))
        self.assertEqual(self.led.notes(T0.date()), {"gpu-denied:lab-nightly": 1})

    def test_the_card_refused_by_the_environment_is_not_a_denial(self):
        with mock.patch.dict(gl.os.environ, {"GYM_LOOP_NO_GPU": "1"}):
            s = gl.Server(self.loop, self.t, "x")
        self.assertIsNone(s.handle)
        self.assertEqual(self.led.notes(T0.date()), {})


if __name__ == "__main__":
    unittest.main()
