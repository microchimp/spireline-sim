"""The fix lab's guards: when the runner may work, and when it must give the machine back.

``Guards.poll()`` answers one question, "may the lab run right now?", as a ``Verdict``. In order:

  halt    ``LAB/HALT`` exists (the runner wrote it after a safety check of the real game failed; the owner removes it;
          with ``stay`` the runner waits for that instead of exiting).
  user    keyboard or mouse input in the last ``idle_s`` seconds (default 60): stop at once and leave (exit 0);
          the Task Scheduler idle trigger starts the runner again at the next idle. With ``stay`` (the runner's
          --always-on, for the logon task) it stops the children and waits instead: nothing would start it again.
  pause   ``LAB/PAUSE`` exists, or ``backlog/main.lock``: no new step starts; a running step gets ``grace`` seconds
          to reach a safe point (its end), then is stopped. ``PAUSE`` holding the word ``now`` stops at once.
          main.lock is the landing lock and guards git state only the presort touches (its tree step checks out the lab
          worktree, its build steps read that tree): ``poll(lock_stops=False)``, which the runner passes while a job other
          than the presort runs, does not return a lock verdict, so that job's steps carry on; the lock is then
          reported in ``readings["lock_ignored"]``. Between jobs ``lock_stops`` stays True: no job of any kind starts.
  load    other processes used more than ``load_pct`` percent of the machine over the last ``load_window`` seconds.
          "Other" is the whole machine's busy time minus the CPU time of this runner's own Job Object, so the
          lab's own game servers (and its practice filler, D48) never count against it, and minus the always-on
          fight loop's process tree (``TreeCpu``; it yields to the night and to QUIET.lock lines by itself).
          Stop at once, wait, resume when the window cools.
  disk    ``LAB`` over ``lab_gb`` or the drive below ``free_gb`` free, after pruning what pruning may remove.
  trial   the ``--trial`` time is up (a running step gets ``grace``).

Every clock and probe is a constructor argument, so the tests drive the guards without a PC.
"""

from __future__ import annotations

import collections
import dataclasses
import os
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from tools.lab import winsys

GB = 1024**3
GUARD_NAMES = ("halt", "user", "pause", "lock", "load", "disk", "parent")


@dataclasses.dataclass
class Verdict:
    kind: str = "ok"  # ok | halt | user | pause | lock | load | disk | trial | parent
    detail: str = ""
    grace: float = 0.0  # seconds a running step may take to reach its end before it is stopped (0: stop now)
    leave: bool = False  # exit the runner (exit 0) instead of waiting

    @property
    def ok(self) -> bool:
        return self.kind == "ok"

    def __str__(self) -> str:
        return self.kind if not self.detail else f"{self.kind} ({self.detail})"


FILETIME_EPOCH = 116444736000000000  # 1601-01-01 to 1970-01-01 in 100 ns


class TreeCpu:
    """The CPU time (100 ns, never decreasing) a process tree that is not the runner's has used since the first call: the always-on
    fight loop's daemon and everything under it (D48: the load guard does not count it, as it does not count the runner's own Job
    Object; the loop yields to the night and to QUIET.lock lines by itself). `who()` gives (pid, latest creation time it may have, in
    epoch seconds, or None); a pid whose process was created later than that is a reused pid, not the loop. A process first seen
    after the first call adds all its CPU (it started since); one that ends between two calls loses only its last interval."""

    def __init__(self, who: Callable[[], tuple[int | None, float | None]], times: Callable[[int], dict] = winsys.tree_times):
        self.who, self.times = who, times
        self.total = 0
        self.last: dict[tuple[int, int], int] | None = None
        self.pid: int | None = None

    def __call__(self) -> int:
        try:
            pid, latest = self.who()
            now = self.times(pid) if pid else {}
            root = next((k for k in now if k[0] == pid), None)
            if root is not None and latest is not None and (root[1] - FILETIME_EPOCH) / 1e7 > latest:
                now = {}  # the pid was reused after the loop wrote its status
        except Exception:  # noqa: BLE001  (a view of the loop refines the guard; it is never a reason the guard fails)
            pid, now = None, {}
        if self.last is None or pid != self.pid:  # first look at this loop: what it used before is not counted
            self.last, self.pid = dict(now), pid
            return self.total
        for k, v in now.items():
            self.total += max(0, v - self.last.get(k, 0))
        self.last = now
        return self.total


class LoadMonitor:
    """The share of the machine that is busy with something other than this runner, over a sliding window."""

    def __init__(self, ours: Callable[[], int], window: float = 120.0, interval: float = 5.0,
                 clock: Callable[[], float] = time.monotonic,
                 times: Callable[[], tuple[int, int, int] | None] = winsys.system_times):
        self.ours, self.window, self.interval, self.clock, self.times = ours, window, interval, clock, times
        self.samples: collections.deque[tuple[float, float, float]] = collections.deque()  # (t_end, fraction, seconds)
        self._prev: tuple[float, tuple[int, int, int], int] | None = None

    def sample(self) -> float | None:
        """Take a sample if `interval` has passed since the last one; the fraction (0..1) of it, or None."""
        now = self.clock()
        if self._prev is not None and now - self._prev[0] < self.interval:
            return None
        cur, mine = self.times(), self.ours()
        if cur is None:
            return None
        got = None
        if self._prev is not None:
            t0, p, pm = self._prev
            total = (cur[1] - p[1]) + (cur[2] - p[2])  # kernel time includes idle time
            if total > 0:
                busy = total - (cur[0] - p[0])
                got = min(1.0, max(0.0, busy - (mine - pm)) / total)
                self.samples.append((now, got, now - t0))
        self._prev = (now, cur, mine)
        while self.samples and self.samples[0][0] < now - self.window:
            self.samples.popleft()
        return got

    def mean(self) -> tuple[float, float]:
        """(duration-weighted mean fraction, seconds covered) over the window."""
        secs = sum(s[2] for s in self.samples)
        return (sum(s[1] * s[2] for s in self.samples) / secs if secs else 0.0), secs

    def sustained(self, threshold: float) -> bool:
        """The mean exceeds `threshold` and the samples cover at least 90% of the window."""
        mean, secs = self.mean()
        return secs >= 0.9 * self.window and mean > threshold


def dir_bytes(path: Path) -> int:
    """Total size of the files under `path` (links not followed)."""
    total = 0
    stack = [path]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


class Guards:
    def __init__(self, lab: Path, lock_file: Path | None, *, disabled: set[str] | None = None, idle_s: float = 60.0,
                 load_pct: float = 50.0, load_window: float = 120.0, lab_gb: float = 30.0, free_gb: float = 50.0,
                 grace: float = 90.0, deadline: float | None = None, clock: Callable[[], float] = time.monotonic,
                 idle_fn: Callable[[], float | None] = winsys.user_idle_seconds, load: LoadMonitor | None = None,
                 prune: Callable[[int], int] | None = None, free_fn: Callable[[Path], int] | None = None,
                 parent_pid: int | None = None, disk_every: float = 60.0, stay: bool = False):
        self.lab, self.lock_file = lab, lock_file
        self.stay = stay  # always-on: user input and HALT pause the lab instead of ending the runner
        self.disabled = disabled or set()
        self.idle_s, self.load_pct, self.lab_gb, self.free_gb, self.grace = idle_s, load_pct, lab_gb, free_gb, grace
        self.deadline, self.clock, self.idle_fn = deadline, clock, idle_fn
        self.load = load
        self.prune = prune
        self.free_fn = free_fn or (lambda p: shutil.disk_usage(p).free)
        self.parent_pid, self.disk_every = parent_pid, disk_every
        self._disk_at = float("-inf")
        self._disk_verdict = Verdict()
        self.readings: dict[str, object] = {}

    def on(self, name: str) -> bool:
        return name not in self.disabled

    def poll(self, lock_stops: bool = True) -> Verdict:
        lab = self.lab
        if self.load is not None and self.on("load"):
            self.load.sample()  # always, so a long pause does not leave the window stale
        if self.on("halt") and (lab / "HALT").exists():
            return Verdict("halt", _first_line(lab / "HALT"), leave=not self.stay)
        if self.on("user"):
            idle = self.idle_fn()
            self.readings["idle_s"] = idle
            if idle is not None and idle < self.idle_s:
                return Verdict("user", f"input {idle:.0f}s ago", leave=not self.stay)
        if self.on("parent") and self.parent_pid and not winsys.parent_alive(self.parent_pid):
            return Verdict("parent", "the process that started the runner is gone", leave=True)
        if self.on("pause") and (lab / "PAUSE").exists():
            now = _first_line(lab / "PAUSE").strip().lower() == "now"
            return Verdict("pause", "LAB/PAUSE", grace=0.0 if now else self.grace)
        held = self.on("lock") and self.lock_file is not None and self.lock_file.exists()
        self.readings["lock_ignored"] = bool(held and not lock_stops)
        if held and lock_stops:
            return Verdict("lock", "backlog main.lock", grace=self.grace)
        if self.deadline is not None and self.clock() >= self.deadline:
            return Verdict("trial", "trial time is up", grace=min(self.grace, 60.0), leave=True)
        if self.on("load") and self.load is not None:
            mean, secs = self.load.mean()
            self.readings["load"] = (round(mean * 100), round(secs))
            if self.load.sustained(self.load_pct / 100.0):
                return Verdict("load", f"other processes {mean * 100:.0f}% of the CPU over {secs:.0f}s")
        if self.on("disk"):
            v = self._disk()
            if not v.ok:
                return v
        return Verdict()

    def _disk(self) -> Verdict:
        now = self.clock()
        if now - self._disk_at < self.disk_every:
            return self._disk_verdict
        self._disk_at = now
        size, free = dir_bytes(self.lab), self.free_fn(self.lab if self.lab.exists() else self.lab.parent)
        self.readings["disk"] = (round(size / GB, 2), round(free / GB, 1))
        need = max(size - int(self.lab_gb * GB), int(self.free_gb * GB) - free, 0)
        if need and self.prune is not None:
            freed = self.prune(need)
            size, free = size - freed, free + freed
            need = max(size - int(self.lab_gb * GB), int(self.free_gb * GB) - free, 0)
        self._disk_verdict = (Verdict("disk", f"lab {size / GB:.1f} GB (limit {self.lab_gb:g}), drive free {free / GB:.1f} GB "
                                              f"(minimum {self.free_gb:g})") if need else Verdict())
        return self._disk_verdict


def read_text_lenient(p: Path) -> str:
    """A small control file as text, whoever wrote it: PowerShell's ``echo now > PAUSE`` writes UTF-16 with a BOM (5.1)
    or UTF-8 with a BOM (7), and an editor may leave NULs. BOMs are dropped and NULs stripped; bad bytes become U+FFFD."""
    with open(p, "rb") as fh:
        raw = fh.read(4096)
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8-sig", errors="replace")
    return text.replace("\x00", "").lstrip("﻿")


def pause_tag(lab: Path) -> str:
    """Who paused: the first word of LAB/PAUSE after an optional leading ``now`` ("" when there is none), as it goes in runner.log."""
    try:
        words = _first_line(lab / "PAUSE").split()
    except OSError:
        return ""
    if words and words[0].lower() == "now":
        words = words[1:]
    return re.sub(r"[^\w.:+-]", "", words[0])[:40] if words else ""


def _first_line(p: Path) -> str:
    try:
        return read_text_lenient(p).strip().splitlines()[0][:200]
    except (OSError, IndexError):
        return ""
