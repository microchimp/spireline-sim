"""The lab's QUIET.lock protocol for a job that takes the machine for hours: the nightly real-game block (`rl/nightly_real.py`, tag
`lab-nightly`) and the daytime real-game score (`gen0.py score --quiet-lock`, tag `lab-score`).

One line per holder in the shared QUIET.lock file: `<tag> <UTC> pid=<n>`. The always-on fight loop (`rl/gym_loop.py`, `why_blocked`)
stops its running step and starts no other while any other tag's line is under 2 h old, so a job that holds a line has the machine
to itself: its games are not stopped by the lab's load guard because the loop was generating beside them.

A holder takes the lock when the file has no line, its own line (a resumed job) or a line older than `stale_h` first; otherwise it
waits, polling every `poll_s`, for up to `wait_h`. While it holds the lock `start_heartbeat` rewrites its line (every 15 minutes, so
the lab runner's 45-minute staleness rule and another's 2 h rule never read a live holder as stale); `release` removes it. Lines
of other holders that are still live are kept as they are when the file is rewritten or the line removed.

The fight loop drops its own line between steps and waits `--yield-s` (150 s) before it starts the next, longer than the 60 s poll
here, so a waiting holder gets in; a loop that died with its line leaves a line that ages out after 2 h, which ends any wait.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

STALE_H = 2.0  # another's lock line under this old is live
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def utc_now(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(time.time() if t is None else t, dt.UTC).strftime(UTC_FORMAT)


def parse_utc(s: str) -> float | None:
    try:
        return dt.datetime.strptime(s, UTC_FORMAT).replace(tzinfo=dt.UTC).timestamp()
    except ValueError:
        return None


class QuietLock:
    """`<tag> <UTC>` in one file. Taken when absent, ours (same tag: a resumed job), or its line is older than `stale_h`; otherwise
    waited for, up to `wait_h`. While held, `heartbeat` rewrites the line so the holder never reads as stale."""

    def __init__(self, path: Path, tag: str, stale_h: float = STALE_H, wait_h: float = 2.0, poll_s: float = 60.0,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep):
        self.path, self.tag, self.stale_h, self.wait_h, self.poll_s = Path(path), tag, stale_h, wait_h, poll_s
        self.clock, self.sleep = clock, sleep
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _lines(self) -> list[str]:
        try:
            return [x.strip() for x in self.path.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
        except OSError:
            return []

    def _age_h(self, line: str) -> float:
        """A line's age in hours: its UTC, else the file's mtime."""
        parts = line.split()
        t = parse_utc(parts[1]) if len(parts) > 1 else None
        if t is None:
            with contextlib.suppress(OSError):
                t = self.path.stat().st_mtime
        return (self.clock() - t) / 3600 if t is not None else 0.0

    def holder(self) -> tuple[str, float] | None:
        """(tag, age in hours) of the first line in the file; None when there is none. The age is the line's UTC, else the file's mtime."""
        lines = self._lines()
        if not lines:
            return None
        parts = lines[0].split()
        return (parts[0] if parts else "", self._age_h(lines[0]))

    def blocked_by(self) -> str | None:
        h = self.holder()
        if h is None or h[0] == self.tag or h[1] >= self.stale_h:
            return None
        return f"{h[0] or 'a lock'} ({h[1]:.1f} h old)"

    def _others(self) -> list[str]:
        """The file's lines that are not ours and still live: kept when our line is written or removed."""
        return [x for x in self._lines() if x.split()[0] != self.tag and self._age_h(x) < self.stale_h]

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        own = f"{self.tag} {utc_now(self.clock())} pid={os.getpid()}"  # the runner clears a line whose pid is gone
        self.path.write_text("".join(x + "\n" for x in [own, *self._others()]), encoding="utf-8")

    def acquire(self) -> bool:
        deadline = self.clock() + self.wait_h * 3600
        while True:
            if self.blocked_by() is None:
                self._write()
                return True
            if self.clock() >= deadline:
                return False
            self.sleep(self.poll_s)

    def holds(self) -> bool:
        return any(x.split()[0] == self.tag for x in self._lines())

    def start_heartbeat(self, every_s: float = 900.0) -> None:
        def beat() -> None:
            while not self._stop.wait(every_s):
                if self.holds():
                    with contextlib.suppress(OSError):
                        self._write()

        self._thread = threading.Thread(target=beat, daemon=True)
        self._thread.start()

    def release(self) -> None:
        self._stop.set()
        if not self.holds():  # never remove another's line
            return
        rest = [x for x in self._lines() if x.split()[0] != self.tag]
        if rest:
            self.path.write_text("".join(x + "\n" for x in rest), encoding="utf-8")
        else:
            self.path.unlink(missing_ok=True)
