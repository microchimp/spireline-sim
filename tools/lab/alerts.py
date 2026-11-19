"""LAB/alerts/: what the runner raises when a job kind fails twice in a row, and what it lists when a job is killed at its time limit.

Before this a failing kind (score-real failed eight four-hour attempts in a row, 2026-11) left `rc=1` and "backing off after repeated failures"
in runner.log and was tried again after every backoff, with nothing anywhere else that said so. Now:

  LAB/alerts/<kind>.txt     written when the kind's second failure in a row ends; the runner stops picking the kind until the file is
                            removed by hand. It holds the kind, the two job folders, their return codes and durations, and the last three
                            lines of each failed command's log.
  LAB/alerts/timeouts.txt   one entry for every job, report or practice block killed at its time limit, even the first time.
  LAB/alerts/run-student.txt   the run loop's stop (D58): every kind with a human bar failed it in two student rounds running (`write_run_student`);
                            the runner no longer picks the `student` kind until the file is removed by hand.
  LAB/alerts/run-plateau.txt   ten days without a run-loop promotion (`write_run_plateau`), with the last rounds' verdicts; a note, nothing is held.

Three silent failures the 2026-11-18 review of the machine found (210 core-hours of failed retries, a dead holder's QUIET.lock line that kept the
fight loop and the practice filler out) and the two-in-a-row rule above does not see (2026-11-19):

  LAB/alerts/<kind>-repeat.txt    a kind (or fights strategy) failed three times (more than twice) in 24 hours, a success in between or not.
                                  A note: nothing is held (two in a row hold the kind by `<kind>.txt`). Removing the file starts the count again.
  LAB/alerts/<kind>-timeout.txt   a kind killed at its time limit twice in seven days (`timeouts.txt` lists every one). A note; removing it starts the count again.
  LAB/alerts/quiet-lock-<tag>.txt a QUIET.lock line `<tag> <UTC> pid=<n>` more than an hour old whose process is gone. One alert per distinct line;
                                  the line is for a person to remove (the runner clears only its own `lab-*` lines).

`record_failure`, `record_timeout` and `raise_repeats` keep their counts in the runner's state dict (`fail_times`, `timeout_times`,
`alerted_extra`), so a restart does not forget them; `raise_dead_locks` reads the QUIET.lock file and asks whether a process lives.

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


def _round_lines(rounds: Sequence[dict]) -> list[str]:
    """The student rounds' verdicts as lines: round, tag, what was built and which kinds failed which leg (no paths, long lines cut)."""
    lines: list[str] = []
    for r in rounds:
        built = r.get("candidate") or "no candidate"
        lines.append(f"round {r.get('round')} ({r.get('tag')}, {str(r.get('utc', ''))[:10]}): {built}; judged {r.get('judged') or 'no'}")
        for kind, v in sorted((r.get("verdict") or {}).items()):
            if isinstance(v, dict) and not v.get("pass"):
                lines.append(scrub(f"  {kind}: leg 1 {v.get('leg1')}, leg 2 {v.get('leg2')}: {v.get('why') or 'no reason given'}"))
    return lines


def write_run_student(lab: Path | str, rounds: Sequence[dict], *, now: float | None = None) -> Path:
    """LAB/alerts/run-student.txt: every kind with a human bar failed it in the two rounds given (the run loop's stop, D58). The student kind is held
    while it stands; removing it by hand lets the student run again and starts the count over (`jobs.run_loop_tick`)."""
    path = alert_path(lab, "run-student")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"ALERT run-student: every kind failed the human bar in {len(rounds)} student rounds running, raised {utc(now)}",
             "The runner no longer picks the student kind (the teacher keeps labelling). Read LAB/student/verdicts.md, then remove this file by hand",
             "to let the student run again; the count of failing rounds starts over. LAB/student/frozen.json lists the kinds frozen so far.", ""]
    path.write_text("\n".join([*lines, *_round_lines(rounds)]) + "\n", encoding="utf-8")
    return path


def write_run_plateau(lab: Path | str, since: float, days: float, rounds: Sequence[dict], *, now: float | None = None) -> Path:
    """LAB/alerts/run-plateau.txt: no run-loop promotion since `since` (epoch seconds; the last promotion, else the loop's start). A note: nothing is
    held; removing the file starts the count of days again."""
    path = alert_path(lab, "run-plateau")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"ALERT run-plateau: no run-loop promotion in {days:g} days (since {utc(since)[:10]}), raised {utc(now)}",
             "The loop keeps running; this is the note. The last student rounds' verdicts follow (LAB/student/verdicts.md has the tables).",
             "Remove this file by hand when it is dealt with: the days count again from then.", ""]
    path.write_text("\n".join([*lines, *(_round_lines(rounds) or ["(no student round yet)"])]) + "\n", encoding="utf-8")
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


# ---------------------------------------------------------------- the silent failures (2026-11-19)

REPEAT_FAILURES = 3  # more than twice ...
REPEAT_WINDOW_S = 24 * 3600.0  # ... in a day
TIMEOUT_REPEAT = 2  # killed at its limit twice ...
TIMEOUT_WINDOW_S = 7 * 24 * 3600.0  # ... in a week
DEAD_LOCK_MIN_AGE_H = 1.0  # a QUIET.lock line this old, its process gone
KEEP_ROWS = 20
FORGET_DEAD_LOCK_S = 30 * 24 * 3600.0


def repeat_name(kind: str) -> str:
    return f"{kind}-repeat"


def timeout_name(kind: str) -> str:
    return f"{kind}-timeout"


def lock_name(tag: str) -> str:
    return f"quiet-lock-{tag}"


def record_failure(state: dict, kind: str, job: str, rc: int, secs: float, now: float | None = None) -> list[dict]:
    """Note a failed job of `kind` in `state["fail_times"]` (the last day's, at most KEEP_ROWS); returns the kind's rows."""
    now = time.time() if now is None else now
    rows = [r for r in state.setdefault("fail_times", {}).get(kind, []) if now - r["t"] < REPEAT_WINDOW_S]
    rows.append({"t": now, "job": str(job), "rc": rc, "secs": secs})
    state["fail_times"][kind] = rows[-KEEP_ROWS:]
    return state["fail_times"][kind]


def record_timeout(state: dict, kind: str, job: str, cmd: str, limit_s: float, ran_s: float, now: float | None = None) -> list[dict]:
    """Note a command of `kind` killed at its time limit in `state["timeout_times"]` (the last week's); returns the kind's rows."""
    now = time.time() if now is None else now
    rows = [r for r in state.setdefault("timeout_times", {}).get(kind, []) if now - r["t"] < TIMEOUT_WINDOW_S]
    rows.append({"t": now, "job": str(job), "cmd": str(cmd), "limit_s": limit_s, "ran_s": ran_s})
    state["timeout_times"][kind] = rows[-KEEP_ROWS:]
    return state["timeout_times"][kind]


def write_repeat_failures(lab: Path | str, kind: str, rows: Sequence[dict], now: float | None = None) -> Path:
    """LAB/alerts/<kind>-repeat.txt: the failures of the last day, each with its job folder, return code, duration and the tails of its failed logs."""
    path = alert_path(lab, repeat_name(kind))
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"ALERT {repeat_name(kind)}: {kind} failed {len(rows)} times in 24 hours (not necessarily in a row), raised {utc(now)}",
             f"A note: the runner still picks {kind} by its own retry rule (two failures in a row hold it, LAB/alerts/{safe_name(kind)}.txt). A kind that fails,",
             "recovers and fails again spends its core-hours without ever tripping that. Remove this file by hand when it is dealt with: the count starts over.", ""]
    for r in rows:
        lines.append(f"{utc(r['t'])} job runs/{r['job']}: rc={r['rc']} duration={r['secs']}s")
        for cmd, tl in failed_logs(Path(lab) / "runs" / str(r["job"]))[:2]:
            lines.append(f"  {cmd} log, last {len(tl)} line(s):")
            lines += [f"    {x}" for x in tl]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_repeat_timeouts(lab: Path | str, kind: str, rows: Sequence[dict], now: float | None = None) -> Path:
    """LAB/alerts/<kind>-timeout.txt: the commands of the kind killed at their time limit in the last week."""
    path = alert_path(lab, timeout_name(kind))
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"ALERT {timeout_name(kind)}: {kind} was killed at its time limit {len(rows)} times in 7 days, raised {utc(now)}",
             "A note: nothing is held. The limit is too short for the work, or the work hangs; either way each kill spent the whole limit for nothing. "
             f"{TIMEOUTS} lists every kill with its log tail. Remove this file by hand when it is dealt with: the count starts over.", ""]
    for r in rows:
        lines.append(f"{utc(r['t'])} runs/{r['job']} {r['cmd']}: killed at its {float(r['limit_s']):.0f}s limit (ran {float(r['ran_s']):.0f}s)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def dead_quiet_lines(path: Path | str, now: float | None = None, alive=None, min_age_h: float = DEAD_LOCK_MIN_AGE_H) -> list[dict]:
    """The lines of a QUIET.lock file (`<tag> <UTC> pid=<n>`) that name a process which is gone (`alive(pid)` false) and are more than `min_age_h`
    old by their own time: `{tag, utc, pid, age_h, line}`. A line with no pid, no readable time, or a live process is not listed. `alive` is
    `winsys.parent_alive` by default (it says a process it cannot open is gone)."""
    now = time.time() if now is None else now
    if alive is None:
        from tools.lab import winsys  # noqa: PLC0415

        alive = winsys.parent_alive
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        pid = next((int(x[4:]) for x in parts[2:] if x.startswith("pid=") and x[4:].isdigit()), None)
        try:
            t = dt.datetime.strptime(parts[1], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.UTC).timestamp()
        except ValueError:
            continue
        age_h = (now - t) / 3600
        if pid is None or age_h <= min_age_h or alive(pid):
            continue
        out.append({"tag": parts[0], "utc": parts[1], "pid": pid, "age_h": age_h, "line": " ".join(parts)})
    return out


def write_dead_lock(lab: Path | str, d: dict, now: float | None = None) -> Path:
    """LAB/alerts/quiet-lock-<tag>.txt for one dead holder's line (see `dead_quiet_lines`)."""
    path = alert_path(lab, lock_name(d["tag"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"ALERT {lock_name(d['tag'])}: the QUIET.lock line '{scrub(d['line'])}' is {d['age_h']:.1f} h old and process {d['pid']} is gone, raised {utc(now)}",
             "A holder that runs rewrites its line every 15 minutes. This one has not, and nothing runs under that process id, so the job it stood for",
             "died with its line. A line under two hours old keeps the fight loop's step, the practice filler, the teacher and the gates out.",
             "Remove that one line from QUIET.lock by hand (leave the others), then remove this file. The runner clears only its own lab-* lines.", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def raise_repeats(lab: Path | str, state: dict, now: float | None = None) -> list[Path]:
    """Write the repeat-failure and repeat-timeout alerts the counts in `state` call for and are not raised yet; the paths written. An alert
    file removed by hand since it was raised starts its count again. A file that stands without the state knowing (a state lost) is taken as raised."""
    now = time.time() if now is None else now
    done = state.setdefault("alerted_extra", {})
    for name, info in list(done.items()):
        if info.get("type") in ("repeat", "timeout") and not alert_path(lab, name).exists():
            done.pop(name)
            state.get("fail_times" if info["type"] == "repeat" else "timeout_times", {}).pop(info.get("kind"), None)
    out: list[Path] = []
    for kind, rows in list(state.get("fail_times", {}).items()):
        recent = [r for r in rows if now - r["t"] < REPEAT_WINDOW_S]
        name = repeat_name(kind)
        if len(recent) < REPEAT_FAILURES or name in done:
            continue
        done[name] = {"type": "repeat", "kind": kind, "raised": utc(now)}
        if not alert_path(lab, name).exists():
            out.append(write_repeat_failures(lab, kind, recent, now))
    for kind, rows in list(state.get("timeout_times", {}).items()):
        recent = [r for r in rows if now - r["t"] < TIMEOUT_WINDOW_S]
        name = timeout_name(kind)
        if len(recent) < TIMEOUT_REPEAT or name in done:
            continue
        done[name] = {"type": "timeout", "kind": kind, "raised": utc(now)}
        if not alert_path(lab, name).exists():
            out.append(write_repeat_timeouts(lab, kind, recent, now))
    return out


def raise_dead_locks(lab: Path | str, quiet_lock_path: Path | str, state: dict, now: float | None = None, alive=None) -> list[Path]:
    """One alert for each QUIET.lock line whose holder has been gone for more than an hour (`dead_quiet_lines`), once per distinct line (the
    state remembers it for 30 days, so removing the alert file while the line stands does not raise it again every look); the paths written."""
    now = time.time() if now is None else now
    done = state.setdefault("alerted_extra", {})
    for name, info in list(done.items()):
        if info.get("type") == "quiet-lock" and now - float(info.get("t", now)) > FORGET_DEAD_LOCK_S:
            done.pop(name)
    out: list[Path] = []
    for d in dead_quiet_lines(quiet_lock_path, now, alive):
        name = lock_name(d["tag"])
        if done.get(name, {}).get("line") == d["line"]:
            continue
        done[name] = {"type": "quiet-lock", "line": d["line"], "raised": utc(now), "t": now}
        out.append(write_dead_lock(lab, d, now))
    return out


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
