"""LAB/alerts/: what the runner raises when a job kind fails twice in a row, and what it lists when a job is killed at its time limit.

Before this a failing kind (score-real failed eight four-hour attempts in a row, 2026-11) left `rc=1` and "backing off after repeated failures"
in runner.log and was tried again after every backoff, with nothing anywhere else that said so. Now:

  LAB/alerts/<kind>.txt     written when the kind's second failure in a row ends; the runner stops picking the kind until the file is
                            removed by hand. It holds the kind, the two job folders, their return codes and durations, and the last three
                            lines of each failed command's log.
  LAB/alerts/timeouts.txt   one entry for every job, report or practice block killed at its time limit, even the first time.

`banner` / `status_prefix` are what the daily readers print at the top (status.txt, clock.md, summary.md). Pure file work: no runner state.
"""

from __future__ import annotations

import datetime as dt
import re
import time
from collections.abc import Sequence
from pathlib import Path

ALERT_DIR = "alerts"
TIMEOUTS = "timeouts.txt"
TAIL_LINES = 3
LINE_CHARS = 300
HOME = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+", re.IGNORECASE)  # a results file holds no user paths


def utc(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(time.time() if t is None else t, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_name(kind: str) -> str:
    """The file name for a kind or strategy (`fights/first-events` style names have no path characters)."""
    return re.sub(r"[^\w.+-]", "_", str(kind)) or "unnamed"


def alerts_dir(lab: Path | str) -> Path:
    return Path(lab) / ALERT_DIR


def alert_path(lab: Path | str, kind: str) -> Path:
    return alerts_dir(lab) / f"{safe_name(kind)}.txt"


def scrub(line: str) -> str:
    """A log line for a results file: user paths made anonymous, long lines cut."""
    line = HOME.sub("<home>", line.rstrip())
    return line if len(line) <= LINE_CHARS else line[: LINE_CHARS - 3] + "..."


def tail(path: Path | str, n: int = TAIL_LINES) -> list[str]:
    """The last `n` non-empty lines of a log (empty when the log is not there)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [scrub(x) for x in [ln for ln in text.splitlines() if ln.strip()][-n:]]


def failed_logs(job_dir: Path, limit: int = 3) -> list[tuple[str, list[str]]]:
    """(command name, last three log lines) of the commands that failed in a job folder, from its job.json; the newest log when it names none."""
    import json  # noqa: PLC0415

    logs = Path(job_dir) / "logs"
    names: list[str] = []
    try:
        data = json.loads((Path(job_dir) / "job.json").read_text(encoding="utf-8"))
        for step in data.get("steps", []):
            names += [c for c, v in (step.get("cmds") or {}).items() if isinstance(v, dict) and v.get("status") == "failed"]
    except (OSError, ValueError):
        pass
    out = [(c, tail(logs / f"{c}.log")) for c in names[:limit]]
    if not any(lines for _, lines in out):
        newest = sorted(logs.glob("*.log"), key=lambda p: p.stat().st_mtime)[-1:] if logs.is_dir() else []
        out = [(p.stem, tail(p)) for p in newest]
    return out


def write_failure(lab: Path | str, kind: str, failures: Sequence[dict], *, blocks: bool = True, now: float | None = None) -> Path:
    """LAB/alerts/<kind>.txt for the kind's failures in a row (each `{job, rc, secs}`, the job folder under LAB/runs). The file says
    whether the kind is stopped (`blocks`) and how to let it run again."""
    path = alert_path(lab, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"ALERT {kind}: {len(failures)} failures in a row, raised {utc(now)}",
             (f"The runner no longer picks {kind}. Remove this file (by hand, after reading the logs) to let it run again."
              if blocks else f"The runner still picks {kind} by its own retry rule; this file stays until it is removed by hand."), ""]
    for f in failures:
        job = str(f.get("job"))
        lines.append(f"job runs/{job}: rc={f.get('rc')} duration={f.get('secs')}s")
        shown = failed_logs(Path(lab) / "runs" / job)
        for cmd, tl in shown:
            lines.append(f"  {cmd} log, last {len(tl)} line(s):")
            lines += [f"    {x}" for x in tl]
        if not shown:
            lines.append("  (no log)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_timeout(lab: Path | str, kind: str, job: str, cmd: str, limit_s: float, ran_s: float, log: Path | str | None = None,
                  now: float | None = None) -> Path:
    """One entry in LAB/alerts/timeouts.txt for a command killed at its time limit (a first time counts)."""
    path = alerts_dir(lab) / TIMEOUTS
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = [f"{utc(now)} {kind} runs/{job} {cmd}: killed at its {limit_s:.0f}s limit (ran {ran_s:.0f}s)"]
    entry += [f"    {x}" for x in (tail(log) if log else [])]
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(entry) + "\n")
    return path


def listing(lab: Path | str) -> list[dict]:
    """The alert files now, oldest first: `{name, kind, first, count}` (`count` the entries of timeouts.txt)."""
    d = alerts_dir(lab)
    out = []
    try:
        files = sorted((p for p in d.glob("*.txt") if p.is_file()), key=lambda p: p.stat().st_mtime)
    except OSError:
        return []
    for p in files:
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        count = sum(1 for x in lines if x and not x.startswith(" ")) if p.name == TIMEOUTS else 1
        out.append({"name": p.name, "kind": p.stem, "first": lines[0] if lines else "", "count": count})
    return out


def status_prefix(lab: Path | str) -> str:
    """The one line status.txt starts with while an alert stands ("" when none): ALERT <kind>[, <kind>] (LAB/alerts)."""
    got = listing(lab)
    if not got:
        return ""
    names = ", ".join(f"{a['kind']}" + (f" x{a['count']}" if a["name"] == TIMEOUTS else "") for a in got)
    return f"ALERT {names} (LAB/alerts)"


MARK_OPEN, MARK_CLOSE = "<!-- lab-alerts -->", "<!-- /lab-alerts -->"


def banner(lab: Path | str) -> str:
    """The alerts as the top of a daily reader (summary.md, clock.md), between markers so it can be replaced; "" when there are none."""
    got = listing(lab)
    if not got:
        return ""
    rows = [f"- {a['name']}: {a['first']}" for a in got]
    return "\n".join([MARK_OPEN, "# ALERTS (LAB/alerts; remove a file by hand when it is dealt with)", "", *rows, MARK_CLOSE]) + "\n\n"


def strip_banner(text: str) -> str:
    """`text` without the alert block `banner` put at its top."""
    a = text.find(MARK_OPEN)
    b = text.find(MARK_CLOSE)
    if a < 0 or b < a:
        return text
    return text[:a] + text[b + len(MARK_CLOSE):].lstrip("\n")


def with_banner(text: str, lab: Path | str) -> str:
    """`text` with the current alerts at the top (a block put there before is replaced, and dropped when no alert stands)."""
    return banner(lab) + strip_banner(text)


def refresh(path: Path | str, lab: Path | str) -> bool:
    """Bring the alert block at the top of a daily reader file up to date; True when the file changed (no file: nothing is made)."""
    p = Path(path)
    try:
        old = p.read_text(encoding="utf-8")
    except OSError:
        return False
    new = with_banner(old, lab)
    if new == old:
        return False
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(new, encoding="utf-8")
    tmp.replace(p)
    return True
