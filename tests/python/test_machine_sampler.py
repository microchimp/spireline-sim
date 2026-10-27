"""referee/machine_sampler.py: the one-night machine sampler of the real arm (L3). No game: fake probes and a fake clock; the Windows
readings are read once from this process."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
import machine_sampler as ms  # noqa: E402


class FakeProbes:
    def __init__(self, fail_first: int = 0):
        self.calls, self.fail_first, self.lock = 0, fail_first, threading.Lock()

    def read(self) -> dict:
        with self.lock:
            self.calls += 1
            if self.calls <= self.fail_first:
                raise OSError("counter gone")
            return {k: float(self.calls) for k in ms.FIELDS}


class SamplerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = self.tmp / "machine.jsonl"
        self.logs: list[str] = []

    def lines(self) -> list[dict]:
        return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines()] if self.path.is_file() else []

    def test_a_sample_is_one_json_line_with_the_time_the_fields_and_the_runs_done(self):
        ticks = iter([1_700_000_000.0, 1_700_000_030.0])
        s = ms.MachineSampler(self.path, 30, FakeProbes(), runs_done=lambda: 7, clock=lambda: next(ticks), log=self.logs.append)
        s.sample_once()
        s.sample_once()
        a, b = self.lines()
        self.assertEqual(("2023-11-14T22:13:20Z", 1_700_000_000.0, 7), (a["utc"], a["t"], a["runs_done"]))
        self.assertTrue(set(ms.FIELDS) <= set(a), set(ms.FIELDS) - set(a))
        self.assertEqual(30.0, b["t"] - a["t"])  # epoch seconds, the clock of the run rows' t_end
        self.assertEqual([], self.logs)

    def test_the_thread_samples_every_interval_until_stopped_and_never_after(self):
        probes = FakeProbes()
        s = ms.MachineSampler(self.path, 0.01, probes, log=self.logs.append)
        s.start()
        end = time.time() + 10
        while probes.calls < 4 and time.time() < end:
            time.sleep(0.01)
        s.stop()
        self.assertFalse(s._thread.is_alive())
        n = len(self.lines())
        self.assertGreaterEqual(n, 3)
        self.assertEqual(n + 1, probes.calls)  # the reading taken at start is discarded: the rates start from it
        time.sleep(0.1)
        self.assertEqual(n, len(self.lines()))

    def test_a_failing_read_is_logged_once_skipped_and_the_next_sample_is_written(self):
        s = ms.MachineSampler(self.path, 30, FakeProbes(fail_first=2), log=self.logs.append)
        self.assertEqual([None, None], [s.sample_once(), s.sample_once()])
        self.assertIsNotNone(s.sample_once())
        self.assertEqual(1, len(self.lines()))
        self.assertEqual(1, len(self.logs))  # once, not per sample
        self.assertIn("read", self.logs[0])

    def test_a_write_that_fails_is_logged_once_and_never_raises(self):
        s = ms.MachineSampler(self.tmp, 30, FakeProbes(), log=self.logs.append)  # a directory cannot be opened for append
        self.assertEqual([None, None], [s.sample_once(), s.sample_once()])
        self.assertEqual(1, len(self.logs))

    def test_a_probe_that_fails_at_start_does_not_stop_the_thread(self):
        probes = FakeProbes(fail_first=1)
        s = ms.MachineSampler(self.path, 0.01, probes, log=self.logs.append)
        s.start()
        end = time.time() + 10
        while probes.calls < 3 and time.time() < end:
            time.sleep(0.01)
        s.stop()
        self.assertTrue(self.lines())


@unittest.skipUnless(sys.platform == "win32", "the readings are Windows counters")
class WinProbesTest(unittest.TestCase):
    def test_a_second_reading_holds_every_field(self):
        logs: list[str] = []
        p = ms.WinProbes(Path(tempfile.gettempdir()).drive or None, log=logs.append)
        p.read()
        time.sleep(0.3)
        r = p.read()
        self.assertEqual(set(ms.FIELDS), set(r))
        self.assertGreater(r["avail_mb"], 0)
        self.assertTrue(0 <= r["mem_load_pct"] <= 100)
        self.assertTrue(0 <= r["cpu_pct"] <= 100, r["cpu_pct"])
        self.assertTrue(r["tree_cpu_pct"] >= 0, r["tree_cpu_pct"])  # this process and its children, as a share of the machine
        self.assertEqual(0, r["games"])  # no game process is in this test's tree
        self.assertGreater(r["timer_ms"], 0)


if __name__ == "__main__":
    unittest.main()
