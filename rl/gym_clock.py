"""The fight-gym daemon's clock: where its wall time went, by reason (D57 addendum 2 and the gate lane's item 3, 2026-11-11).

The loop once lost three quarters of a week to waits nobody had logged. Every interval the daemon spends is now one of

- **play**: a step that runs games (`generate`, `gate`, `human-bar`, `starts`, `weights`), or
- **wait**: it cannot start or go on (a reason: `quiet:<tag>`, `main.lock`, `nightly-active`, `night-window`, `pause`, `halt`, `publish`;
  `yield`, the fixed pause between steps, is a wait that is counted but not logged line by line).

A wait logs its reason when it starts and its duration when it ends (`wait start: quiet:fix-lab`, `wait end: quiet:fix-lab after 14m20s`);
every interval, played or waited, is a row in `games/clock.jsonl` (`t0`, `t1`, `kind`, `what`, `s`), split at 00:00Z. The status file
carries `waiting_s_today`, `playing_s_today` and `waits_today` (seconds by reason); at 00:00Z `roll` writes one `daily` line to loop.log:
the day's fights and segments (from `games/segments.jsonl`), the gates (from `gate/verdicts.jsonl`), the seconds played and waited by
reason, and the seconds the daemon was in neither (between steps, in compaction and the like). A day with rows and no daily line (the
daemon was down at midnight) gets its line at the next start.

Pure bookkeeping on an injectable clock: no process, no lock file, no emulator.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import threading
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

DAY = dt.timedelta(days=1)


def utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def dur(s: float) -> str:
    """4m05s, 1h02m, 37s."""
    s = int(round(s))
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


def hours(s: float) -> str:
    return f"{s / 3600:.1f} h"


def day_start(t: dt.datetime) -> dt.datetime:
    return t.replace(hour=0, minute=0, second=0, microsecond=0)


class TimeLedger:
    """`games/clock.jsonl`: the daemon's intervals, one row each, and the daily line."""

    def __init__(self, path: Path | str, log: Callable[[str], None] | None = None, now: Callable[[], dt.datetime] = utc,
                 segments: Path | str | None = None, ledger: Path | str | None = None) -> None:
        self.path = Path(path)
        self.log = log or (lambda m: None)
        self.now = now
        self.segments = Path(segments) if segments else None  # games/segments.jsonl: fights and segments of a day
        self.ledger = Path(ledger) if ledger else None  # gate/verdicts.jsonl: the gates of a day
        self.lock = threading.RLock()
        self.cur: dict | None = None  # {kind, what, t0 (this day's piece), start (the interval's first start), log}
        self.rows: list[dict] = []
        self._load()

    # -- rows
    def _load(self) -> None:
        self.rows = []
        try:
            with open(self.path, encoding="utf-8") as fh:
                for x in fh:
                    if x.strip():
                        try:
                            self.rows.append(json.loads(x))
                        except ValueError:
                            pass
        except OSError:
            pass

    def _append(self, row: dict) -> None:
        self.rows.append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row) + "\n")

    def _write_interval(self, kind: str, what: str, t0: dt.datetime, t1: dt.datetime) -> None:
        """One row per UTC day the interval touches."""
        while t0 < t1:
            end = min(t1, day_start(t0) + DAY)
            self._append({"t0": iso(t0), "t1": iso(end), "kind": kind, "what": what, "s": round((end - t0).total_seconds(), 1)})
            t0 = end

    # -- intervals
    def begin(self, kind: str, what: str, log: bool = False) -> None:
        """Start an interval (the open one ends first). A wait with `log` says so in loop.log."""
        with self.lock:
            self.end()
            now = self.now()
            self.cur = {"kind": kind, "what": what, "t0": now, "start": now, "log": bool(log)}
            if log and kind == "wait":
                self.log(f"wait start: {what}")

    def end(self) -> None:
        """End the open interval; a logged wait says how long it lasted."""
        with self.lock:
            c = self.cur
            if c is None:
                return
            self.cur = None
            now = self.now()
            self._write_interval(c["kind"], c["what"], c["t0"], now)
            if c["log"] and c["kind"] == "wait":
                self.log(f"wait end: {c['what']} after {dur((now - c['start']).total_seconds())}")

    def wait(self, what: str, log: bool = True) -> None:
        """Be waiting for `what`: nothing changes when that wait is already open; another reason ends it and starts the new one."""
        with self.lock:
            if self.cur and self.cur["kind"] == "wait" and self.cur["what"] == what:
                return
            self.begin("wait", what, log=log)

    def clear(self) -> None:
        """The wait is over (if one is open): its end is logged."""
        with self.lock:
            if self.cur and self.cur["kind"] == "wait":
                self.end()

    @contextlib.contextmanager
    def span(self, kind: str, what: str, log: bool = False):
        """An interval for the length of the block; the one it interrupts (a gate's play around its human-bar play) goes on after it."""
        with self.lock:
            prev = None if self.cur is None else (self.cur["kind"], self.cur["what"])
            self.begin(kind, what, log=log)
        try:
            yield
        finally:
            with self.lock:
                self.end()
                if prev is not None:
                    self.begin(prev[0], prev[1], log=False)

    def note(self, what: str) -> None:
        """A countable event with no length (`gpu-denied:<holder>`): a row, and a count in the daily line."""
        with self.lock:
            self._append({"t": iso(self.now()), "note": what})

    # -- the day
    def totals(self, day: dt.date | None = None) -> dict:
        """Seconds played and waited on `day` (today by default), by reason, the open interval's share included."""
        with self.lock:
            now = self.now()
            day = day or now.date()
            a = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
            b = a + DAY
            by: dict[tuple[str, str], float] = defaultdict(float)
            for r in self.rows:
                if "kind" in r and r["t0"][:10] == day.isoformat():
                    by[(r["kind"], r["what"])] += float(r["s"])
            c = self.cur
            if c is not None:
                lo, hi = max(c["t0"], a), min(now, b)
                if hi > lo:
                    by[(c["kind"], c["what"])] += (hi - lo).total_seconds()
            return {"play_s": round(sum(v for (k, _), v in by.items() if k == "play"), 1),
                    "wait_s": round(sum(v for (k, _), v in by.items() if k == "wait"), 1),
                    "play": {w: round(v, 1) for (k, w), v in sorted(by.items()) if k == "play"},
                    "wait": {w: round(v, 1) for (k, w), v in sorted(by.items()) if k == "wait"},
                    "notes": self.notes(day)}

    def notes(self, day: dt.date) -> dict:
        out: dict[str, int] = defaultdict(int)
        for r in self.rows:
            if "note" in r and r["t"][:10] == day.isoformat():
                out[r["note"]] += 1
        return dict(sorted(out.items()))

    def status_fields(self) -> dict:
        """What the daemon's status file carries."""
        t = self.totals()
        return {"waiting_s_today": t["wait_s"], "playing_s_today": t["play_s"], "waits_today": t["wait"]}

    def _jsonl(self, p: Path | None) -> list[dict]:
        out: list[dict] = []
        if p is None:
            return out
        try:
            with open(p, encoding="utf-8") as fh:
                for x in fh:
                    if x.strip():
                        try:
                            out.append(json.loads(x))
                        except ValueError:
                            pass
        except OSError:
            pass
        return out

    def daily_line(self, day: dt.date) -> str:
        """`daily 2026-11-10: ...` for a finished day."""
        t = self.totals(day)
        d = day.isoformat()
        seg = [r for r in self._jsonl(self.segments) if str(r.get("t", ""))[:10] == d]
        gates = [r for r in self._jsonl(self.ledger) if str(r.get("t", ""))[:10] == d]
        fights = sum(int(r.get("fights") or 0) for r in seg)
        played = ", ".join(f"{w} {hours(s)}" for w, s in sorted(t["play"].items(), key=lambda x: -x[1]))
        waited = ", ".join(f"{w} {hours(s)}" for w, s in sorted(t["wait"].items(), key=lambda x: -x[1]))
        other = max(0.0, 86400.0 - t["play_s"] - t["wait_s"])
        verdicts: dict[str, int] = defaultdict(int)
        for r in gates:
            verdicts[f"stage {r.get('stage')} {r.get('verdict')}"] += 1
        vtext = ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items()))
        text = (f"daily {d}: {len(seg)} segments, {fights} fights played; played {hours(t['play_s'])}"
                + (f" ({played})" if played else "") + f", waited {hours(t['wait_s'])}" + (f" ({waited})" if waited else "")
                + f", between steps or down {hours(other)}; {len(gates)} gated" + (f" ({vtext})" if vtext else ""))
        if t["notes"]:
            text += "; " + ", ".join(f"{k} x{v}" for k, v in t["notes"].items())
        return text

    def pending_days(self) -> list[dt.date]:
        """Days before today that have interval rows and no `daily` mark."""
        today = self.now().date()
        done = {r["daily"] for r in self.rows if "daily" in r}
        days = sorted({r["t0"][:10] for r in self.rows if "kind" in r} - done)
        return [dt.date.fromisoformat(x) for x in days if dt.date.fromisoformat(x) < today]

    def roll(self) -> list[str]:
        """Past midnight: split the open interval at 00:00Z (its rows stay one a day) and write the daily line of every finished day not
        written yet. Called every few seconds by the daemon's watch thread and at its start; returns the lines written."""
        with self.lock:
            now = self.now()
            c = self.cur
            if c is not None and day_start(c["t0"]) < day_start(now):
                cut = day_start(now)
                self._write_interval(c["kind"], c["what"], c["t0"], cut)
                c["t0"] = cut
            out = []
            for day in self.pending_days():
                line = self.daily_line(day)
                self.log(line)
                self._append({"daily": day.isoformat(), "t": iso(now)})
                out.append(line)
            return out
