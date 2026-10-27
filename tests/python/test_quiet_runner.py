"""docs/rl/VRUN_SWEEP.md section 7: the quiet-window runner (tools/gpu_lane/quiet_runner.py) and `vrun_sweep.py run --stop-file`.

No GPU and no lab: the open-window predicate on synthetic text, the queue's state machine, the runner against a fake job script in
a temp folder, and the stop file on the synthetic sweep cache the vrun_sweep tests build (CPU, a few seconds)."""

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "rl", ROOT / "src", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import test_vrun_sweep as sweep_tests  # noqa: E402  (the module, so its TestCases are not collected twice)
import vrun_sweep as sweep  # noqa: E402

from tools.gpu_lane import quiet_runner as qr  # noqa: E402

STAMP = "2026-10-27T21:19:41Z"
TAIL = "since=2026-10-26T16:42:44Z tree=5670dce jobs_done=617 fights=75446 runs=600 interrupts=load:43,lock:2,pause:46,user:65"


def status(state: str, step: str | None = None, extra: str = "") -> str:
    """A status.txt line as the lab runner writes it (tools/lab/runner.py `status`)."""
    where = f"job=20261026T1642-{step} step={step} strategy={step}" if step else "job=-"
    return f"{STAMP} {state} {where} {TAIL}" + (f" | {extra}" if extra else "") + "\n"


QUIET = status("paused(pause)", "selfplay", "LAB/PAUSE other_cpu=2%/122s")
LOCK_W4 = "w4-x 2026-10-27T21:00:00Z\n"


class ParseStatusTests(unittest.TestCase):
    def test_every_state_the_lab_writes_while_it_waits_or_runs(self):
        for state in ("paused(pause)", "paused(user)", "paused(load)", "paused(halt)", "paused(lock)", "running"):
            with self.subTest(state=state):
                self.assertEqual(qr.parse_status(status(state, "selfplay"))["state"], state)

    def test_the_step_is_read_before_the_bar_only(self):
        self.assertEqual(qr.parse_status(status("running", "train", "train")), {"state": "running", "step": "train"})
        self.assertEqual(qr.parse_status(status("paused(pause)", "nightly-real", "LAB/PAUSE other_cpu=2%/122s")),
                         {"state": "paused(pause)", "step": "nightly-real"})
        self.assertEqual(qr.parse_status(status("paused(pause)", None, "step=train note"))["step"], None)
        self.assertEqual(qr.parse_status(status("idle")), {"state": "idle", "step": None})

    def test_a_line_without_the_timestamp_and_an_unreadable_one(self):
        self.assertEqual(qr.parse_status("paused(pause) job=x step=train since=y")["state"], "paused(pause)")
        for text in ("", "\n", "garbled line", None, "2026-10-27T21:19:41Z"):
            with self.subTest(text=text):
                self.assertIsNone(qr.parse_status(text)["state"])

    def test_only_the_first_line_counts(self):
        self.assertEqual(qr.parse_status(status("running", "selfplay") + status("paused(pause)")), {"state": "running", "step": "selfplay"})


class WindowTests(unittest.TestCase):
    """Every rule of section 7, on text (None: the file is absent)."""

    def window(self, status_text=QUIET, lock=None, ok=None, off=False):
        return qr.window_open(status_text, lock, ok, off)

    def test_no_lock_and_a_pause_or_lock_pause_opens(self):
        for state in ("paused(pause)", "paused(lock)"):
            with self.subTest(state=state):
                opened, reason = self.window(status(state, "selfplay"))
                self.assertTrue(opened, reason)

    def test_no_lock_and_any_other_state_is_closed(self):
        for state in ("paused(user)", "paused(load)", "paused(halt)", "running"):
            with self.subTest(state=state):
                opened, reason = self.window(status(state, "selfplay"))
                self.assertFalse(opened)
                self.assertIn(state, reason)
        for state in ("idle", "paused(trial)", "paused(disk)", "halted", "stopped(parent)"):
            with self.subTest(state=state):
                self.assertFalse(self.window(status(state))[0])

    def test_the_kill_file_closes_it_whatever_else_holds(self):
        opened, reason = self.window(QUIET, off=True)
        self.assertFalse(opened)
        self.assertIn("GPU_OFF", reason)
        self.assertFalse(self.window(QUIET, LOCK_W4, "w4-x\n", off=True)[0])

    def test_a_running_train_step_closes_it(self):
        for state in ("paused(pause)", "paused(lock)", "running"):
            with self.subTest(state=state):
                opened, reason = self.window(status(state, "train"))
                self.assertFalse(opened)
                self.assertIn("train", reason)
        self.assertFalse(self.window(status("paused(pause)", "train"), LOCK_W4, "w4-x\n")[0])  # consent does not hand over the lab's GPU
        self.assertTrue(self.window(status("paused(pause)", "selfplay"))[0])

    def test_a_lock_with_consent_opens(self):
        opened, reason = self.window(QUIET, LOCK_W4, "w4-x\n")
        self.assertTrue(opened)
        self.assertIn("w4-x", reason)
        self.assertTrue(self.window(QUIET, "w4-x\n", "w4-x 2026-10-27T21:05:00Z\n")[0])  # a UTC after the tag is allowed on either line
        self.assertTrue(self.window(status("idle"), LOCK_W4, "w4-x\n")[0])  # rule 3's consent clause names no state; rule 4 bars four

    def test_a_lock_without_consent_is_closed_whatever_the_tag(self):
        for tag in ("w4-x", "w3-blocks", "lab-nightly"):
            with self.subTest(tag=tag):
                opened, reason = self.window(QUIET, f"{tag} 2026-10-27T21:00:00Z\n", None)
                self.assertFalse(opened)
                self.assertIn(tag, reason)
        self.assertFalse(self.window(status("paused(lock)", "selfplay"), LOCK_W4, None)[0])

    def test_consent_for_another_tag_does_not_count(self):
        opened, reason = self.window(QUIET, LOCK_W4, "w5-y\n")
        self.assertFalse(opened)
        self.assertIn("another tag", reason)
        self.assertFalse(self.window(QUIET, LOCK_W4, "")[0])
        self.assertFalse(self.window(QUIET, LOCK_W4, "\nw4-x\n")[0])  # the first line, not any line

    def test_the_lanes_own_lock_is_closed_even_with_a_consent_file(self):
        for ok in (None, "gpu-lane\n"):
            with self.subTest(ok=ok):
                opened, reason = self.window(QUIET, "gpu-lane 2026-10-27T21:00:00Z\n", ok)
                self.assertFalse(opened)
                self.assertIn("gpu-lane", reason)

    def test_consent_does_not_override_the_barred_states(self):
        for state in ("paused(user)", "paused(load)", "paused(halt)", "running"):
            with self.subTest(state=state):
                self.assertFalse(self.window(status(state, "selfplay"), LOCK_W4, "w4-x\n")[0])

    def test_an_empty_lock_has_no_tag_and_a_missing_or_unreadable_status_is_closed(self):
        self.assertFalse(self.window(QUIET, "", "w4-x\n")[0])
        self.assertFalse(self.window(None)[0])
        self.assertFalse(self.window("garbled")[0])
        self.assertFalse(self.window("")[0])

    def test_reasons_hold_no_paths(self):
        cases = [self.window(QUIET, LOCK_W4, "w4-x\n"), self.window(QUIET, off=True), self.window(QUIET, LOCK_W4),
                 self.window(status("paused(user)")), self.window(None)]
        for _, reason in cases:
            self.assertIsNone(re.search(r"[A-Za-z]:[\\/]|/Users/|/home/", reason), reason)


class QueueTests(unittest.TestCase):
    @staticmethod
    def queue(*states):
        return [{"name": f"j{i}", "args": ["--arms", "lab"], "state": s, "attempts": 0} for i, s in enumerate(states)]

    def test_the_first_queued_or_interrupted_job_in_file_order(self):
        self.assertEqual(qr.next_job(self.queue("done", "queued", "queued")), 1)
        self.assertEqual(qr.next_job(self.queue("done", "interrupted", "queued")), 1)
        self.assertEqual(qr.next_job(self.queue("queued", "interrupted")), 0)
        self.assertEqual(qr.next_job(self.queue("failed", "running", "done")), None)
        self.assertEqual(qr.next_job([]), None)

    def test_the_states_a_job_goes_through(self):
        queue = self.queue("queued", "queued", "queued")
        qr.mark(queue, 0, "running")
        qr.mark(queue, 0, "done")
        qr.mark(queue, 1, "running")
        qr.mark(queue, 1, "failed")
        qr.mark(queue, 2, "running")
        self.assertEqual([j["state"] for j in queue], ["done", "failed", "running"])
        self.assertEqual([j["attempts"] for j in queue], [0, 0, 0])
        self.assertEqual(qr.next_job(queue), None)

    def test_an_interrupted_job_counts_the_attempt_and_is_retried_before_later_jobs(self):
        queue = self.queue("queued", "queued")
        qr.mark(queue, 0, "running")
        qr.mark(queue, 0, "interrupted")
        self.assertEqual((queue[0]["state"], queue[0]["attempts"]), ("interrupted", 1))
        self.assertEqual(qr.next_job(queue), 0)  # the retry goes first
        qr.mark(queue, 0, "running")
        qr.mark(queue, 0, "interrupted")
        self.assertEqual(queue[0]["attempts"], 2)
        qr.mark(queue, 0, "running")
        qr.mark(queue, 0, "done")
        self.assertEqual(qr.next_job(queue), 1)

    def test_an_unknown_state_is_refused(self):
        with self.assertRaises(ValueError):
            qr.mark(self.queue("queued"), 0, "paused")

    def test_the_file_is_rewritten_atomically_and_keeps_what_it_does_not_know(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queue.json"
            self.assertEqual(qr.load_queue(path), [])
            queue = self.queue("queued")
            queue[0]["note"] = "kept"
            qr.save_queue(path, queue)
            qr.save_queue(path, queue)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["queue.json"])  # no temp file left behind
            self.assertEqual(qr.load_queue(path), queue)
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                qr.load_queue(path)
            path.write_text('{"jobs": []}', encoding="utf-8")
            with self.assertRaises(ValueError):
                qr.load_queue(path)


FAKE_JOB = textwrap.dedent('''
    import json, os, sys, time
    from pathlib import Path
    sys.path.insert(0, {src!r})
    from sts2_gym import procutil
    argv = sys.argv[1:]
    stop = Path(argv[argv.index("--stop-file") + 1])
    mode = argv[argv.index("--mode") + 1]
    marker = Path(argv[argv.index("--marker") + 1])
    print("priority", procutil.niceness(), flush=True)
    print("threads", os.environ.get("OMP_NUM_THREADS"), os.environ.get("MKL_NUM_THREADS"), flush=True)
    print("lib", os.environ.get("STS2_LIB_PATH"), flush=True)
    marker.write_text(json.dumps({{"argv": argv, "stop_at_start": stop.exists(), "cwd": os.getcwd()}}), encoding="utf-8")
    if mode == "ok":
        sys.exit(0)
    if mode == "fail":
        sys.stderr.write('Traceback (most recent call last):\\n  File "C:/Users/someone/work/x.py", line 3, in <module>\\n')
        sys.stderr.write('ValueError: boom at C:/Users/someone/work/data.bin\\n')
        sys.exit(1)
    if mode == "wait":  # a fit: stops at an epoch end once the stop file is there
        for _ in range(1200):
            if stop.exists():
                sys.exit(3)
            time.sleep(0.05)
        sys.exit(0)
    if mode == "ignore":
        time.sleep(120)
''')

FAST = ["--idle-s", "0.2", "--busy-s", "0.2"]


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.lab, self.referee, self.repo, self.lane = (self.root / n for n in ("lab", "referee", "repo", "lane"))
        for folder in (self.lab, self.referee, self.repo, self.lane):
            folder.mkdir()
        self.script = self.root / "fake_job.py"
        self.script.write_text(FAKE_JOB.format(src=str(ROOT / "src")), encoding="utf-8")
        self.queue_path, self.log_path = self.lane / "queue.json", self.lane / "quiet.log"
        self.set_status(QUIET)
        self.cache_arg = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def set_status(self, text: str) -> None:
        (self.lab / "status.txt").write_text(text, encoding="utf-8")

    def job(self, name: str, mode: str, state: str = "queued", attempts: int = 0) -> dict:
        return {"name": name, "args": ["--mode", mode, "--marker", str(self.root / f"{name}.marker")], "state": state, "attempts": attempts}

    def write_queue(self, jobs: list[dict]) -> None:
        qr.save_queue(self.queue_path, jobs)

    def queue(self) -> list[dict]:
        return qr.load_queue(self.queue_path)

    def states(self) -> list[tuple[str, str, int]]:
        return [(j["name"], j["state"], j["attempts"]) for j in self.queue()]

    def log_lines(self) -> list[str]:
        if not self.log_path.exists():
            return []
        return [line.split(" ", 1)[1] for line in self.log_path.read_text(encoding="utf-8").splitlines()]

    def run_once(self, *extra: str) -> int:
        argv = ["--lab", str(self.lab), "--referee-dir", str(self.referee), "--repo", str(self.repo), "--gpu-python", sys.executable,
                "--queue", str(self.queue_path), "--log", str(self.log_path), "--once", "--job-script", str(self.script),
                *self.cache_arg, *FAST, *extra]
        with contextlib.redirect_stdout(io.StringIO()):  # the transitions also go to stdout
            return qr.main(argv)

    def close_when_started(self, name: str, text: str) -> threading.Thread:
        marker = self.root / f"{name}.marker"

        def close() -> None:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not marker.exists():
                time.sleep(0.02)
            time.sleep(0.1)
            self.set_status(text)

        thread = threading.Thread(target=close, daemon=True)
        thread.start()
        return thread

    def marker(self, name: str) -> dict:
        return json.loads((self.root / f"{name}.marker").read_text(encoding="utf-8"))

    # ---- the runs
    def test_a_job_in_an_open_window_runs_to_done_at_below_normal_with_two_threads(self):
        self.write_queue([self.job("a", "ok")])
        self.assertEqual(self.run_once(), 0)
        self.assertEqual(self.states(), [("a", "done", 0)])
        self.assertEqual(self.log_lines(), ["window opened: quiet, lab paused(pause)", "job started: a (attempt 1)", "job done: a exit 0"])
        argv = self.marker("a")["argv"]
        self.assertEqual(argv[0], "run")
        self.assertEqual(argv[-4:-1], ["--device", "cuda", "--stop-file"])
        self.assertEqual(Path(argv[-1]), self.lane / "quiet.stop")
        self.assertFalse((self.lane / "quiet.stop").exists())
        job_log = (self.lane / "job-a.log").read_text(encoding="utf-8")
        self.assertIn("priority 10", job_log)  # BelowNormal
        self.assertIn("threads 2 2", job_log)
        self.assertEqual(Path(self.marker("a")["cwd"]).resolve(), self.repo.resolve())

    def test_the_environment_passes_through_and_the_cache_is_prepended(self):
        self.cache_arg = ["--cache", str(self.root / "cache")]
        self.write_queue([self.job("a", "ok")])
        with mock.patch.dict(os.environ, {"STS2_LIB_PATH": "lib-from-the-runner"}):
            self.run_once()
        argv = self.marker("a")["argv"]
        self.assertEqual(argv[:3], ["run", "--cache", str(self.root / "cache")])
        self.assertIn("lib lib-from-the-runner", (self.lane / "job-a.log").read_text(encoding="utf-8"))

    def test_a_failed_job_is_not_retried_and_the_next_one_still_runs(self):
        self.write_queue([self.job("bad", "fail"), self.job("good", "ok")])
        self.run_once()
        self.assertEqual(self.states(), [("bad", "failed", 0), ("good", "done", 0)])
        lines = self.log_lines()
        self.assertEqual(lines[2], "job failed: bad exit 1: ValueError: boom at <path>")  # the type and message; no traceback, no path
        self.run_once()  # a later window: the failed job stays failed
        self.assertEqual(self.states(), [("bad", "failed", 0), ("good", "done", 0)])
        self.assertEqual(sum(line.startswith("job started") for line in self.log_lines()), 2)

    def test_the_window_closing_stops_the_job_interrupts_it_and_a_later_window_retries_it_first(self):
        self.write_queue([self.job("a", "wait"), self.job("b", "ok")])
        closer = self.close_when_started("a", status("paused(user)", "selfplay"))
        t0 = time.monotonic()
        self.run_once()
        closer.join()
        self.assertLess(time.monotonic() - t0, 30)
        self.assertEqual(self.states(), [("a", "interrupted", 1), ("b", "queued", 0)])
        self.assertFalse((self.lane / "quiet.stop").exists())  # touched, seen, deleted
        self.assertEqual(self.log_lines(), ["window opened: quiet, lab paused(pause)", "job started: a (attempt 1)",
                                            "window closed: rule 4: lab paused(user)", "job interrupted: a exit 3"])
        # the owner left: the window reopens, and a (fixed) job a runs before the queued b
        queue = self.queue()
        queue[0]["args"][1] = "ok"
        self.write_queue(queue)
        self.set_status(QUIET)
        self.run_once()
        self.assertEqual(self.states(), [("a", "done", 1), ("b", "done", 0)])
        starts = [line for line in self.log_lines() if line.startswith("job started")]
        self.assertEqual(starts, ["job started: a (attempt 1)", "job started: a (attempt 2)", "job started: b (attempt 1)"])

    def test_each_rule_that_closes_the_window_stops_a_running_job(self):
        closers = {
            "GPU_OFF": (lambda: (self.lab / "GPU_OFF").write_text("", encoding="utf-8"), "rule 1: GPU_OFF present"),
            "train": (lambda: self.set_status(status("paused(pause)", "train")), "rule 2: lab step train"),
            "lock": (lambda: (self.referee / "QUIET.lock").write_text(LOCK_W4, encoding="utf-8"), "rule 3: QUIET.lock w4-x without GPU_OK"),
        }
        for rule, (close, reason) in closers.items():
            with self.subTest(rule=rule):
                self.set_status(QUIET)
                for stale in (self.lab / "GPU_OFF", self.referee / "QUIET.lock", self.log_path, self.root / "a.marker"):
                    stale.unlink(missing_ok=True)
                self.write_queue([self.job("a", "wait")])
                marker = self.root / "a.marker"
                thread = threading.Thread(target=lambda: (self._wait_for(marker), close()), daemon=True)
                thread.start()
                self.run_once()
                thread.join()
                self.assertEqual(self.states(), [("a", "interrupted", 1)])
                self.assertIn(f"window closed: {reason}", self.log_lines())

    @staticmethod
    def _wait_for(path: Path) -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not path.exists():
            time.sleep(0.02)
        time.sleep(0.1)

    def test_a_job_that_ignores_the_stop_file_is_terminated_after_the_wait_and_counts_as_interrupted(self):
        self.write_queue([self.job("a", "ignore")])
        closer = self.close_when_started("a", status("paused(load)", "selfplay"))
        t0 = time.monotonic()
        self.run_once("--stop-wait-s", "1")
        closer.join()
        self.assertLess(time.monotonic() - t0, 60)  # not the job's 120 s sleep
        self.assertEqual(self.states(), [("a", "interrupted", 1)])
        self.assertTrue(self.log_lines()[-1].startswith("job interrupted: a exit "))
        self.assertTrue(self.log_lines()[-1].endswith("(terminated after the wait)"))
        self.assertFalse((self.lane / "quiet.stop").exists())

    def test_a_stale_stop_file_is_deleted_before_a_job_starts(self):
        (self.lane / "quiet.stop").write_text("left by a runner that died\n", encoding="utf-8")
        self.write_queue([self.job("a", "ok")])
        self.run_once()
        self.assertFalse(self.marker("a")["stop_at_start"])
        self.assertEqual(self.states(), [("a", "done", 0)])

    def test_a_closed_window_starts_nothing_and_logs_nothing(self):
        cases = {
            "paused(user)": lambda: self.set_status(status("paused(user)", "selfplay")),
            "GPU_OFF": lambda: (self.lab / "GPU_OFF").write_text("", encoding="utf-8"),
            "no consent": lambda: (self.referee / "QUIET.lock").write_text(LOCK_W4, encoding="utf-8"),
            "wrong consent": lambda: ((self.referee / "QUIET.lock").write_text(LOCK_W4, encoding="utf-8"),
                                      (self.referee / "GPU_OK").write_text("w5-y\n", encoding="utf-8")),
            "no status": lambda: (self.lab / "status.txt").unlink(),
        }
        for name, arrange in cases.items():
            with self.subTest(case=name):
                for stale in (self.lab / "GPU_OFF", self.referee / "QUIET.lock", self.referee / "GPU_OK", self.log_path):
                    stale.unlink(missing_ok=True)
                self.set_status(QUIET)
                arrange()
                self.write_queue([self.job("a", "ok")])
                self.assertEqual(self.run_once(), 0)
                self.assertEqual(self.states(), [("a", "queued", 0)])
                self.assertEqual(self.log_lines(), [])
                self.assertFalse((self.root / "a.marker").exists())

    def test_a_consenting_lock_opens_the_window_and_the_log_names_the_tag(self):
        (self.referee / "QUIET.lock").write_text(LOCK_W4, encoding="utf-8")
        (self.referee / "GPU_OK").write_text("\N{ZERO WIDTH NO-BREAK SPACE}w4-x\r\n", encoding="utf-8", newline="")  # as PowerShell writes it
        self.write_queue([self.job("a", "ok")])
        self.run_once()
        self.assertEqual(self.states(), [("a", "done", 0)])
        self.assertEqual(self.log_lines()[0], "window opened: consent, tag w4-x")

    def test_a_job_left_running_by_a_dead_runner_is_interrupted_at_startup(self):
        self.write_queue([self.job("a", "ok", state="done"), self.job("b", "ok", state="running")])
        self.set_status(status("paused(user)"))  # closed: only the recovery acts
        self.run_once()
        self.assertEqual(self.states(), [("a", "done", 0), ("b", "interrupted", 1)])
        self.assertEqual(self.log_lines(), ["job interrupted: b (the previous runner ended while it ran)"])

    def test_nothing_in_the_logs_holds_a_path_or_a_user_name(self):
        self.write_queue([self.job("bad", "fail"), self.job("a", "wait")])
        closer = self.close_when_started("a", status("paused(user)"))
        self.run_once()
        closer.join()
        texts = [self.log_path.read_text(encoding="utf-8")] + [p.read_text(encoding="utf-8") for p in self.lane.glob("job-*.log")]
        self.assertGreaterEqual(len(texts), 3)
        for text in texts:
            self.assertNotIn(str(self.root), text)
            self.assertNotIn(self.root.as_posix(), text)
            self.assertNotIn("someone", text)
            self.assertNotIn("Traceback", text)
            self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", text), text)

    def test_a_missing_interpreter_or_job_script_is_a_usage_error(self):
        argv = ["--lab", str(self.lab), "--referee-dir", str(self.referee), "--repo", str(self.repo), "--queue", str(self.queue_path),
                "--log", str(self.log_path), "--once"]
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(qr.main([*argv, "--gpu-python", str(self.root / "nope"), "--job-script", str(self.script)]), 2)
            self.assertEqual(qr.main([*argv, "--gpu-python", sys.executable]), 2)  # no rl/vrun_sweep.py under the empty repo

    # ---- tools and wide jobs (PRIOR_SWEEP.md section 6)
    def wide_job(self, name: str, mode: str, procs: object = 8) -> dict:
        return {**self.job(name, mode), "tool": "prior_sweep", "cpu_procs": procs}

    def consent(self) -> None:
        (self.referee / "QUIET.lock").write_text(LOCK_W4, encoding="utf-8")
        (self.referee / "GPU_OK").write_text("w4-x\n", encoding="utf-8")

    def test_cpu_procs_defaults_to_two_and_a_value_that_is_not_a_positive_integer_counts_as_wide(self):
        self.assertEqual(qr.cpu_procs_of({}), 2)
        self.assertEqual(qr.cpu_procs_of({"cpu_procs": 8}), 8)
        for bad in ("8", 0, -1, True, None, 2.5):
            self.assertEqual(qr.cpu_procs_of({"cpu_procs": bad}), float("inf"), bad)
        queue = [{"state": "queued", "cpu_procs": 8}, {"state": "queued"}, {"state": "queued", "cpu_procs": 2}]
        self.assertEqual(qr.next_job(queue), 0)
        self.assertEqual(qr.next_job(queue, wide=False), 1)  # the consent branch passes the wide job over
        self.assertIsNone(qr.next_job([{"state": "queued", "cpu_procs": "x"}], wide=False))
        self.assertTrue(qr.wide_allowed("quiet, lab paused(pause)"))
        self.assertTrue(qr.wide_allowed("quiet, lab paused(lock)"))
        self.assertFalse(qr.wide_allowed("consent, tag w4-x"))

    def test_under_consent_only_jobs_of_two_processes_or_fewer_run_and_the_wide_one_waits(self):
        self.consent()
        self.write_queue([self.wide_job("a", "ok"), self.job("b", "ok")])
        self.run_once()
        self.assertEqual(self.states(), [("a", "queued", 0), ("b", "done", 0)])
        self.assertFalse((self.root / "a.marker").exists())
        self.assertEqual(self.log_lines(), ["window opened: consent, tag w4-x", "job started: b (attempt 1)", "job done: b exit 0"])

    def test_in_a_quiet_window_the_wide_job_runs_in_queue_order(self):
        self.write_queue([self.wide_job("a", "ok"), self.job("b", "ok")])
        self.run_once()
        self.assertEqual(self.states(), [("a", "done", 0), ("b", "done", 0)])
        self.assertEqual([line for line in self.log_lines() if line.startswith("job started")], ["job started: a (attempt 1)", "job started: b (attempt 1)"])

    def test_a_wide_job_is_stopped_when_the_window_turns_to_consent_and_is_not_restarted_under_it(self):
        self.write_queue([self.wide_job("a", "wait")])
        marker = self.root / "a.marker"

        def turn() -> None:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not marker.exists():
                time.sleep(0.02)
            time.sleep(0.1)
            self.consent()

        thread = threading.Thread(target=turn, daemon=True)
        thread.start()
        t0 = time.monotonic()
        self.run_once()
        thread.join()
        self.assertLess(time.monotonic() - t0, 30)
        self.assertEqual(self.states(), [("a", "interrupted", 1)])
        self.assertTrue(any(line.startswith("job interrupted: a exit 3") for line in self.log_lines()))
        self.run_once()  # the consent window is still open: a job of 8 processes does not start
        self.assertEqual(self.states(), [("a", "interrupted", 1)])
        self.assertEqual(sum(line.startswith("job started") for line in self.log_lines()), 1)

    def test_the_tool_picks_the_script_and_the_command_line(self):
        (self.repo / "rl").mkdir()
        for name in ("vrun_sweep.py", "prior_sweep.py"):
            (self.repo / "rl" / name).write_text("", encoding="utf-8")
        args = argparse.Namespace(lab=str(self.lab), referee_dir=str(self.referee), repo=str(self.repo), gpu_python="py", queue=str(self.queue_path),
                                  log=str(self.log_path), cache="CACHE", idle_s=1, busy_s=1)
        runner = qr.QuietRunner(args)
        tail = ["--device", "cuda", "--stop-file", str(self.lane / "quiet.stop")]
        old = {"name": "v", "args": ["--arms", "lab"]}  # no tool: vrun_sweep, args after `run`, the cache prepended
        self.assertEqual(runner.command(old), ["py", str(self.repo / "rl" / "vrun_sweep.py"), "run", "--cache", "CACHE", "--arms", "lab", *tail])
        new = {"name": "p", "tool": "prior_sweep", "args": ["search", "--out", "X"], "cpu_procs": 8}  # args start with the subcommand; no cache
        self.assertEqual(runner.command(new), ["py", str(self.repo / "rl" / "prior_sweep.py"), "search", "--out", "X", *tail])
        self.assertEqual(runner.command({**old, "tool": "vrun_sweep"}), runner.command(old))

    def test_an_unknown_tool_or_a_missing_script_fails_the_job_and_the_next_one_runs(self):
        (self.repo / "rl").mkdir()
        (self.repo / "rl" / "vrun_sweep.py").write_text("", encoding="utf-8")  # no prior_sweep.py in this repo
        self.write_queue([{**self.job("x", "ok"), "tool": "evil"}, self.wide_job("y", "ok", 2), {"name": "z", "args": "no", "state": "queued", "attempts": 0}])
        args = argparse.Namespace(lab=str(self.lab), referee_dir=str(self.referee), repo=str(self.repo), gpu_python=sys.executable,
                                  queue=str(self.queue_path), log=str(self.log_path), cache=None, idle_s=0.1, busy_s=0.1)
        runner = qr.QuietRunner(args)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run(once=True)
        self.assertEqual(self.states(), [("x", "failed", 0), ("y", "failed", 0), ("z", "failed", 0)])
        lines = self.log_lines()
        self.assertIn("job failed: x: unknown tool 'evil'", lines)
        self.assertIn("job failed: y: the prior_sweep script is missing", lines)
        self.assertIn("job failed: z: the queue entry has no args list", lines)


class StopFileTests(unittest.TestCase):
    """`vrun_sweep.py run --stop-file` on the synthetic cache of test_vrun_sweep (CPU)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        lab, agent, _ = sweep_tests.make_lab(cls.root)
        cls.folder = sweep.build_cache(lab, agent, cls.root / "vrun")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def setUp(self) -> None:
        self.stop = self.root / "stop-file"
        self.stop.unlink(missing_ok=True)
        for part in ("fits", "h"):
            for leftover in (self.folder / part).glob("*"):
                leftover.unlink()

    def run_args(self, arms: str, *flags: str) -> list[str]:
        return ["run", "--cache", str(self.folder), "--arms", arms, "--seeds", "1", "--device", "cpu", *flags]

    def fits(self) -> list[str]:
        return sorted(p.name for p in (self.folder / "fits").glob("*")) if (self.folder / "fits").is_dir() else []

    def test_the_exit_code_is_3(self):
        self.assertEqual(sweep.STOP_EXIT, 3)
        self.assertEqual(qr.STOP_EXIT, sweep.STOP_EXIT)

    def test_a_stop_file_present_at_a_fit_boundary_drops_the_fit_and_exits_3(self):
        self.stop.write_text("", encoding="utf-8")
        self.assertEqual(sweep.main(self.run_args("lab,rows4", "--stop-file", str(self.stop))), sweep.STOP_EXIT)
        self.assertEqual(self.fits(), [])
        self.assertFalse((self.folder / "h").exists() and any((self.folder / "h").iterdir()))  # not even the seed's FightNet was started
        self.assertTrue(self.stop.exists())  # the runner's file: the fit never deletes it

    def test_the_command_line_exits_with_3(self):
        self.stop.write_text("", encoding="utf-8")
        done = subprocess.run([sys.executable, str(ROOT / "rl" / "vrun_sweep.py"), *self.run_args("lab", "--stop-file", str(self.stop))],
                              capture_output=True, text=True, check=False, timeout=300)
        self.assertEqual(done.returncode, 3, done.stderr[-500:])
        self.assertEqual(self.fits(), [])
        self.assertNotIn(str(self.root), done.stdout)

    def appear_at_the_first_epoch_end(self):
        """Patch `_optimise` so that the stop file appears just before the first epoch-end hook runs (the V net's and the BCE head's)."""
        real = sweep.train._optimise  # noqa: SLF001
        calls = {"hooked": 0}

        def spy(net, loss_of, validation, config, generator, *, groups=None, epoch_end=None):
            if epoch_end is None:  # FightNet has no hook
                return real(net, loss_of, validation, config, generator, groups=groups)

            def hook(epoch, net):
                calls["hooked"] += 1
                self.stop.write_text("", encoding="utf-8")
                return epoch_end(epoch, net)

            return real(net, loss_of, validation, config, generator, groups=groups, epoch_end=hook)

        return mock.patch.object(sweep.train, "_optimise", spy), calls

    def test_a_fit_stops_at_an_epoch_end_and_writes_nothing(self):
        for arm in ("lab", "bce-act1"):  # the V net and the BCE head both stop between epochs
            with self.subTest(arm=arm):
                self.stop.unlink(missing_ok=True)
                patch, calls = self.appear_at_the_first_epoch_end()
                with patch:
                    code = sweep.main(self.run_args(arm, "--stop-file", str(self.stop)))
                self.assertEqual(code, sweep.STOP_EXIT)
                self.assertEqual(calls["hooked"], 1)  # the first epoch end raised
                self.assertEqual([n for n in self.fits() if n.startswith(arm)], [])  # no JSON, no npz, no leftover temp file

    def test_a_partial_npz_of_the_dropped_fit_is_removed(self):
        (self.folder / "fits").mkdir(exist_ok=True)
        leftover = self.folder / "fits" / "lab-s0.npz"
        leftover.write_bytes(b"half")
        self.stop.write_text("", encoding="utf-8")
        self.assertEqual(sweep.main(self.run_args("lab", "--stop-file", str(self.stop))), sweep.STOP_EXIT)
        self.assertFalse(leftover.exists())

    def test_without_the_flag_or_with_the_file_absent_the_fit_is_written_as_before(self):
        self.assertEqual(sweep.main(self.run_args("lab", "--stop-file", str(self.stop))), 0)  # flag given, file never appears
        self.assertEqual(self.fits(), ["lab-s0.json", "lab-s0.npz"])
        first = json.loads((self.folder / "fits" / "lab-s0.json").read_text(encoding="utf-8"))
        self.assertTrue(all("s_brier" in e for e in first["history"]))
        for leftover in (self.folder / "fits").glob("*"):
            leftover.unlink()
        self.assertEqual(sweep.main(self.run_args("lab")), 0)  # no flag at all
        self.assertEqual(self.fits(), ["lab-s0.json", "lab-s0.npz"])
        # the stop file of an earlier call does not leak into a call without the flag
        self.stop.write_text("", encoding="utf-8")
        self.assertEqual(sweep.main(self.run_args("rows4")), 0)
        self.assertIn("rows4-s0.json", self.fits())


if __name__ == "__main__":
    unittest.main()
