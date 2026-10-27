"""The quiet-window runner: queued GPU sweep jobs run only while the machine is quiet for the lab (docs/rl/VRUN_SWEEP.md section 7).

    python tools/gpu_lane/quiet_runner.py --lab LAB --referee-dir REFEREE --repo FORK --gpu-python PY --queue queue.json --log quiet.log \
        [--cache DIR] [--idle-s 15] [--busy-s 5] [--once]

It takes no lock of its own, never writes `jobs.json`, and never touches `LAB/PAUSE`. Every path comes from the flags.

The window is open when `LAB/GPU_OFF` is absent, the lab's `status.txt` names no `train` step, the status is none of
`paused(user)`, `paused(load)`, `paused(halt)`, `running`, and either (quiet without a claim) `REFEREE/QUIET.lock` is absent and the
status is `paused(pause)` or `paused(lock)`, or (quiet with consent) `QUIET.lock` carries a tag other than `gpu-lane` and the first
line of `REFEREE/GPU_OK` names that tag. The files are read afresh at every check.

The queue is a JSON list of jobs `{"name", "args", "state", "attempts"}` and optionally `"tool"` and `"cpu_procs"`. The runner takes the
first `queued` or `interrupted` job it may run, runs `<gpu python> rl/vrun_sweep.py run <args> --device cuda --stop-file <path>` (tool
`vrun_sweep`, the default) or `<gpu python> rl/prior_sweep.py <args> --device cuda --stop-file <path>` (tool `prior_sweep`: its args name
the subcommand) from the repo at BelowNormal with two CPU threads, and marks it `done` (exit 0), `interrupted` (the stop-file exit code, attempts + 1, retried in a later window) or `failed` (any other
exit; not retried). A job with `cpu_procs` above 2 (default 2) needs the CPU and runs only in a window opened by rule 3's first branch
(reason "quiet..."), never under consent; a consent window takes the first job of 2 processes or fewer, and a wide job that is running
when the window turns to consent is stopped. When a condition fails while a job runs the runner touches the stop file at once, waits for the process (120 s,
then terminates its own child), deletes the stop file and goes back to watching. The log holds one line per transition and no paths.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sts2_gym import procutil  # noqa: E402

STOP_EXIT = 3  # vrun_sweep.STOP_EXIT: the exit code of a fit dropped at a stop file (duplicated: that module imports torch)
OWN_TAG = "gpu-lane"  # a QUIET.lock with this tag is the lane's own quiet machine: the runner stays out
BARRED_STATES = ("paused(user)", "paused(load)", "paused(halt)", "running")  # rule 4
QUIET_STATES = ("paused(pause)", "paused(lock)")  # rule 3, quiet without a claim
STATES = ("queued", "running", "done", "interrupted", "failed")
RUNNABLE = ("queued", "interrupted")
TOOLS = ("vrun_sweep", "prior_sweep")  # the modules under rl/ a job may name; vrun_sweep is the default
DEFAULT_CPU_PROCS = 2
QUIET_PREFIX = "quiet"  # a window reason that starts so is rule 3's first branch (no QUIET.lock): the CPU is free for wide jobs
STOP_WAIT_S = 120.0
STOP_NAME = "quiet.stop"
LOG_LINE_CAP = 200
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


# --------------------------------------------------------------------------------------
# The open-window predicate (pure)
# --------------------------------------------------------------------------------------

_STATE = re.compile(r"^(?:\S+Z\s+)?(?P<state>(?:paused|stopped)\([^)\s]*\)|running|idle|halted|crashed)(?=\s|$)")
_STEP = re.compile(r"(?:^|\s)step=(?P<step>\S+)")


def parse_status(text: str | None) -> dict[str, str | None]:
    """The state (`paused(pause)`, `running`, ...; None when the line is not a status) and the step (`step=train`) of status.txt's first line."""
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    found = _STATE.match(line)
    head = line.split(" | ", 1)[0]  # what follows the bar is free text
    step = _STEP.search(head)
    return {"state": found["state"] if found else None, "step": step["step"] if step else None}


def first_token(text: str | None) -> str:
    """The first word of the first line: a QUIET.lock reads `<tag> <UTC>`, GPU_OK `<tag>`."""
    lines = (text or "").removeprefix("\N{ZERO WIDTH NO-BREAK SPACE}").splitlines()  # the first line as it is: a blank first line names no tag
    words = lines[0].split() if lines else []
    return words[0] if words else ""


def window_open(status_text: str | None, quiet_lock_text: str | None, gpu_ok_text: str | None, gpu_off_exists: bool) -> tuple[bool, str]:
    """Whether the GPU window is open, and the rule that decided it (a short label: no paths, no names but the lock's tag).

    `None` for a text means the file is absent. The rules are VRUN_SWEEP.md section 7's, checked in the order 1, 2, 4, 3.
    """
    if gpu_off_exists:
        return False, "rule 1: GPU_OFF present"
    if status_text is None:
        return False, "no lab status"
    status = parse_status(status_text)
    state, step = status["state"], status["step"]
    if state is None:
        return False, "lab status unreadable"
    if step == "train" or (step or "").startswith("train-"):
        return False, f"rule 2: lab step {step}"
    if state in BARRED_STATES:
        return False, f"rule 4: lab {state}"
    if quiet_lock_text is None:
        if state in QUIET_STATES:
            return True, f"quiet, lab {state}"
        return False, f"rule 3: no QUIET.lock but lab {state}, not paused(pause) or paused(lock)"
    tag = first_token(quiet_lock_text)
    if not tag:
        return False, "rule 3: QUIET.lock without a tag"
    if tag == OWN_TAG:
        return False, f"rule 3: QUIET.lock tagged {OWN_TAG}"
    if gpu_ok_text is None:
        return False, f"rule 3: QUIET.lock {tag} without GPU_OK"
    if first_token(gpu_ok_text) != tag:
        return False, f"rule 3: QUIET.lock {tag}, GPU_OK names another tag"
    return True, f"consent, tag {tag}"


# --------------------------------------------------------------------------------------
# The queue (pure state machine + atomic file)
# --------------------------------------------------------------------------------------


def cpu_procs_of(job: dict[str, Any]) -> float:
    """The CPU processes a job needs (default 2); a value that is not a positive integer counts as unbounded, so the job waits for a quiet window."""
    value = job.get("cpu_procs", DEFAULT_CPU_PROCS)
    return float(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else float("inf")


def tool_of(job: dict[str, Any]) -> str:
    return str(job.get("tool", TOOLS[0]))


def wide_allowed(reason: str) -> bool:
    """Whether a window opened for `reason` leaves the CPU free for jobs of more than 2 processes (quiet without a claim only)."""
    return reason.startswith(QUIET_PREFIX)


def next_job(queue: list[dict[str, Any]], wide: bool = True) -> int | None:
    """Index of the first `queued` or `interrupted` job in file order that may run, else None. With `wide` false (a consent window)
    a job needing more than 2 CPU processes is passed over."""
    for index, job in enumerate(queue):
        if job.get("state") in RUNNABLE and (wide or cpu_procs_of(job) <= DEFAULT_CPU_PROCS):
            return index
    return None


def mark(queue: list[dict[str, Any]], index: int, state: str) -> None:
    """Set job `index` to `state` in place; `interrupted` also counts the attempt."""
    if state not in STATES:
        raise ValueError(f"a job's state is one of {', '.join(STATES)}, not {state!r}")
    job = queue[index]
    job["state"] = state
    if state == "interrupted":
        job["attempts"] = int(job.get("attempts", 0)) + 1


def load_queue(path: Path) -> list[dict[str, Any]]:
    """The queue file's jobs ([] when the file is absent); ValueError when it is not a list of jobs."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    doc = json.loads(text) if text.strip() else []
    if not isinstance(doc, list) or not all(isinstance(job, dict) for job in doc):
        raise ValueError("the queue is a JSON list of job objects")
    return doc


def save_queue(path: Path, queue: list[dict[str, Any]]) -> None:
    """Rewrite the queue file atomically (a temp file in the same folder, then a replace; the replace is retried if a reader holds the file)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(queue, indent=1) + "\n", encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


# --------------------------------------------------------------------------------------
# What reaches a log
# --------------------------------------------------------------------------------------

_DRIVE_PATH = re.compile(r"(?<![\w])[A-Za-z]:[\\/][^\s\"']*")
_UNIX_PATH = re.compile(r"(?<![\w])/(?:home|Users|tmp|var|opt)/[^\s\"']*")


def safe_line(text: str) -> str | None:
    """A line of a job's output as a log may carry it: traceback lines are dropped (they name files), paths are replaced by <path>,
    and the rest is capped. None means: do not log."""
    text = text.rstrip("\r\n")
    if not text.strip() or text.startswith(("Traceback", "  File ", " ", "\t")):
        return None
    return _UNIX_PATH.sub("<path>", _DRIVE_PATH.sub("<path>", text))[:LOG_LINE_CAP]


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:60] or "job"


# --------------------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------------------


class QuietRunner:
    def __init__(self, a: argparse.Namespace, *, sleep: Callable[[float], None] = time.sleep) -> None:
        self.lab, self.referee = Path(a.lab), Path(a.referee_dir)
        self.repo, self.gpu_python = Path(a.repo), str(a.gpu_python)
        self.queue_path, self.log_path = Path(a.queue), Path(a.log)
        self.cache = a.cache
        self.idle_s, self.busy_s = float(a.idle_s), float(a.busy_s)
        self.stop_wait_s = float(getattr(a, "stop_wait_s", STOP_WAIT_S))
        self.job_script = Path(a.job_script) if getattr(a, "job_script", None) else self.repo / "rl" / "vrun_sweep.py"
        self.script_override = bool(getattr(a, "job_script", None))
        self.stop_path = self.queue_path.parent / STOP_NAME
        self.sleep = sleep
        self.was_open: bool | None = None
        self.bad_queue: str | None = None

    # ---- logging
    def log(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    # ---- the window
    @staticmethod
    def _read(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8-sig", errors="replace")  # -sig: PowerShell writes a byte-order mark
        except OSError:
            return None  # absent (or mid-rewrite by its owner, which the next check reads again)

    def check(self) -> tuple[bool, str]:
        """Read the four files afresh and apply the rules; logs the transition (window opened / closed) when there is one."""
        verdict = window_open(self._read(self.lab / "status.txt"), self._read(self.referee / "QUIET.lock"),
                              self._read(self.referee / "GPU_OK"), (self.lab / "GPU_OFF").exists())
        opened, reason = verdict
        if opened and not self.was_open:
            self.log(f"window opened: {reason}")
        elif not opened and self.was_open:
            self.log(f"window closed: {reason}")
        self.was_open = opened
        return verdict

    # ---- the queue
    def read_queue(self) -> list[dict[str, Any]] | None:
        try:
            queue = load_queue(self.queue_path)
        except (OSError, ValueError) as error:  # json.JSONDecodeError is a ValueError
            if self.bad_queue != type(error).__name__:
                self.log(f"queue unreadable: {type(error).__name__}")
            self.bad_queue = type(error).__name__
            return None
        self.bad_queue = None
        return queue

    def set_state(self, name: str, state: str) -> None:
        """Mark the running job called `name` in a fresh read of the queue file and rewrite it (the lane may have edited the file meanwhile)."""
        queue = self.read_queue()
        if queue is None:
            return
        for index, job in enumerate(queue):
            if job.get("name") == name and job.get("state") == "running":
                mark(queue, index, state)
                save_queue(self.queue_path, queue)
                return

    def recover(self) -> None:
        """A job still marked `running` belongs to a runner that died: it counts as interrupted."""
        queue = self.read_queue()
        for job in queue or []:
            if job.get("state") == "running":
                self.set_state(str(job.get("name")), "interrupted")
                self.log(f"job interrupted: {job.get('name')} (the previous runner ended while it ran)")

    # ---- one job
    def script_for(self, job: dict[str, Any]) -> Path:
        """The script a job runs: the test override, else rl/<tool>.py in the repo."""
        return self.job_script if self.script_override else self.repo / "rl" / f"{tool_of(job)}.py"

    def command(self, job: dict[str, Any]) -> list[str]:
        tail = ["--device", "cuda", "--stop-file", str(self.stop_path)]
        if tool_of(job) == "prior_sweep":  # its args start with the subcommand (run, search); it has no sweep cache
            return [self.gpu_python, str(self.script_for(job)), *map(str, job["args"]), *tail]
        cache = ["--cache", str(self.cache)] if self.cache else []
        return [self.gpu_python, str(self.script_for(job)), "run", *cache, *map(str, job["args"]), *tail]

    def environment(self) -> dict[str, str]:
        env = dict(os.environ)  # STS2_LIB_PATH and the rest pass through from the runner's own environment
        env.update(dict.fromkeys(THREAD_ENV, "2"))
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def run_job(self, index: int) -> str:
        """Run job `index` to its end or until the window closes; returns the job's new state."""
        queue = self.read_queue()
        if queue is None or index >= len(queue):
            return "failed"
        job = queue[index]
        name = str(job.get("name", f"job{index}"))
        attempt = int(job.get("attempts", 0)) + 1
        problem = None
        if not isinstance(job.get("args"), list):
            problem = "the queue entry has no args list"
        elif tool_of(job) not in TOOLS:
            problem = f"unknown tool {tool_of(job)[:40]!r}"
        elif not self.script_for(job).is_file():
            problem = f"the {tool_of(job)} script is missing"
        if problem:
            mark(queue, index, "failed")
            save_queue(self.queue_path, queue)
            self.log(f"job failed: {name}: {problem}")
            return "failed"
        wide = cpu_procs_of(job) > DEFAULT_CPU_PROCS
        self.stop_path.unlink(missing_ok=True)  # one left by a runner that died would stop the job at its first epoch
        mark(queue, index, "running")
        save_queue(self.queue_path, queue)
        self.log(f"job started: {name} (attempt {attempt})")
        job_log = self.queue_path.parent / f"job-{slug(name)}.log"
        last: list[str] = []
        try:
            with job_log.open("a", encoding="utf-8") as sink:
                sink.write(f"--- {utc_now()} attempt {attempt}\n")
            proc = subprocess.Popen(self.command(job), cwd=self.repo, env=self.environment(), stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                    bufsize=1, **procutil.popen_priority_kwargs())
        except OSError as error:
            self.set_state(name, "failed")
            self.log(f"job failed: {name}: could not start ({type(error).__name__})")
            return "failed"

        def pump() -> None:  # the child's output, scrubbed, into its log; also keeps a full pipe from blocking the child
            with job_log.open("a", encoding="utf-8") as sink:
                for raw in proc.stdout:
                    line = safe_line(raw)
                    if line is not None:
                        sink.write(line + "\n")
                        sink.flush()
                        last[:] = [line]

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        closing: str | None = None
        interrupt = False
        try:
            while True:
                try:
                    proc.wait(timeout=self.busy_s)
                    break
                except subprocess.TimeoutExpired:
                    pass
                opened, reason = self.check()
                if not opened:
                    closing = reason
                    break
                if wide and not wide_allowed(reason):  # the lock came back with a consent: the CPU is the lane's, not the job's
                    closing = f"window turned to {reason.split(',')[0]}; {cpu_procs_of(job):g} processes need a quiet one"
                    break
        except KeyboardInterrupt:
            closing, interrupt = "runner interrupted", True
        terminated = False
        if closing is not None:
            self.stop_path.write_text(f"{utc_now()}\n", encoding="utf-8")  # touched at once, in the same pass that saw the condition fail
            try:
                proc.wait(timeout=self.stop_wait_s)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                terminated = True
                proc.terminate()  # this handle's own process, never by image name
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            self.stop_path.unlink(missing_ok=True)
        reader.join(timeout=5)
        if not reader.is_alive():
            proc.stdout.close()
        code = proc.returncode
        if code == 0:
            state = "done"
        elif code == STOP_EXIT or terminated:
            state = "interrupted"  # a job the runner ended itself did not fail; it is retried in a later window
        else:
            state = "failed"
        self.set_state(name, state)
        if state == "done":
            self.log(f"job done: {name} exit {code}")
        elif state == "interrupted":
            self.log(f"job interrupted: {name} exit {code}" + (" (terminated after the wait)" if terminated else ""))
        else:
            self.log(f"job failed: {name} exit {code}" + (f": {last[0]}" if last else ""))
        if interrupt:
            raise KeyboardInterrupt
        return state

    # ---- the loop
    def step(self) -> str | None:
        """One check; with the window open and a job queued, run that job. Returns the job's new state, or None when none ran."""
        opened, reason = self.check()
        if not opened:
            return None
        queue = self.read_queue()
        index = None if queue is None else next_job(queue, wide_allowed(reason))
        return None if index is None else self.run_job(index)

    def run(self, once: bool = False) -> None:
        self.recover()
        while True:
            state = self.step()
            if once:
                if state in ("done", "failed"):
                    continue  # the window may still be open: the next job, until none is left or the window closes
                return
            if state in ("done", "failed"):
                continue
            self.sleep(self.idle_s)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lab", required=True, help="the lab folder (status.txt, GPU_OFF)")
    ap.add_argument("--referee-dir", required=True, help="the folder holding QUIET.lock and GPU_OK")
    ap.add_argument("--repo", required=True, help="the fork's root (rl/vrun_sweep.py runs from here)")
    ap.add_argument("--gpu-python", required=True, help="the GPU interpreter")
    ap.add_argument("--queue", required=True, help="the queue file (a JSON list of jobs)")
    ap.add_argument("--log", required=True, help="the transition log")
    ap.add_argument("--cache", default=None, help="the sweep cache folder, prepended to each job's args as --cache")
    ap.add_argument("--idle-s", type=float, default=15.0, help="seconds between checks while no job runs")
    ap.add_argument("--busy-s", type=float, default=5.0, help="seconds between checks while a job runs")
    ap.add_argument("--once", action="store_true", help="one pass: run queued jobs while the window stays open, then exit (tests)")
    ap.add_argument("--job-script", default=None, help=argparse.SUPPRESS)  # tests: a fake script in place of rl/vrun_sweep.py
    ap.add_argument("--stop-wait-s", type=float, default=STOP_WAIT_S, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    runner = QuietRunner(a)
    for what, path in (("--gpu-python", Path(a.gpu_python)), ("the job script", runner.job_script)):
        if not path.is_file():
            print(f"quiet_runner: {what} is not a file", file=sys.stderr)
            return 2
    try:
        runner.run(once=a.once)
    except KeyboardInterrupt:
        runner.stop_path.unlink(missing_ok=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
