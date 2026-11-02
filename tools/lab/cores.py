"""Cores, not turns (D48): how many of the machine's cores each lab job takes, and the rules of the practice filler.

The runner runs one job at a time from its ladder and rotation (the main lane, as before), and beside it the **filler**: the
`selfplay` kind, run whenever no job needs the cores it would use. A job takes the cores of its kind; the filler gets the rest,
less the cores of the report beside a job and of the always-on fight loop, and runs only while that is at least `floor`. When a
job needs cores the filler holds, the filler is stopped (its Job Object) and started again at the size that is left; a block that
ends starts the next at whatever is free then. The night (nightly-real) takes the whole machine: the filler stops and nothing
else starts.

Settings (LAB/jobs.json, read at every decision; every key optional):

  "cores": {"total": 16, "presort": 16, "fights": 8, "runs": 8, "probes": 4, "mutation": 4, "invariants": 6, "report": 2, "loop": 6}
      the machine's cores (default: the logical processors), the fix-lab job types, the groups-and-kits chain beside a job, and
      the always-on loop while it is stepping (its status file names its pid and phase)
  "<kind>": {"cores": N, ...}
      an agent-era kind's own figure (score-real, train, bench-*); without it, the kind's `procs`, else its command's
      `--procs`, else the whole machine. nightly-real always takes the whole machine.
  "selfplay": {"filler": true, "floor": 4, "explore": ["--explore-tau", "0.1", "--explore-eps", "0.02"],
               "night_margin_min": 30, "quiet_ignore": ["gym-loop", "lab-score"]}
      the filler is on when `enabled` and `filler` are; `explore` is added to its command only when given (default: the plain
      config).

The filler never writes a train marker: its command loses `--lab`, `--shards-root`, `--budget-runs` and `--review-guard` (gen0.py
emu writes LAB/train/ready-<gen> only with --record, --lab and --shards-root), and `--gen-dir` (it plays agent/current.json through
`--agent-root`). Its shards go to its own job folder under LAB/runs (`--out {job}`, `--record`), where training finds them.

It takes no QUIET.lock line, so the night's and the score's drivers (rl/quiet_lock.py, which wait on the first live line) never
wait for it; it yields itself to any live line whose tag is not in `quiet_ignore` (the loop's own and the daytime score's, which it
runs beside), so another session's measurement window still has the machine.

Night: it does not start while LAB/nightly/active is fresh or within `night_margin_min` of the time the night is due
(LAB/nightly/last-ok + nightly-real.every_h, or the retry after a skipped attempt when that is later), and a running block is
stopped two minutes before the night is due, when its marker is up, or when the runner starts the nightly-real job (a block lasts
about 20 minutes, `--block-min 20`, so one started before the margin has ended by then).
"""

from __future__ import annotations

import datetime as dt
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tools.lab import jobs

FILLER_KIND = "selfplay"
DEFAULT_FLOOR = 4
DEFAULT_MARGIN_MIN = 30.0
STOP_BEFORE_S = 120.0  # a running block is stopped this long before the night is due, so the night never finds it running
DEFAULT_QUIET_IGNORE = ("gym-loop", "lab-score")
QUIET_STALE_H = 2.0  # rl/quiet_lock.STALE_H: a line under this old is live
# The fix-lab job types (not kinds of jobs.json). None: the whole machine (presort moves and rebuilds the tree the filler runs from).
JOB_CORES: dict[str, int | None] = {"presort": None, "fights": 8, "runs": 8, "probes": 4, "mutation": 4, "invariants": 6}
REPORT_CORES = 2
LOOP_CORES = 6  # rl/gym_loop.py --procs
LOOP_BUSY = ("start", "starts", "weights", "generate", "gate")  # the loop's status phases that use its processes
# Flags of gen0.py emu the filler drops: flag -> takes a value
DROP = {"--lab": True, "--shards-root": True, "--budget-runs": True, "--review-guard": False, "--gen-dir": True,
        "--explore-tau": True, "--explore-eps": True, "--procs": True, "--server-procs": True}


def _int(v: Any) -> int | None:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None


def _table(doc: dict | None) -> dict:
    t = (doc or {}).get("cores")
    return t if isinstance(t, dict) else {}


def load_doc(lab: Path) -> dict:
    """LAB/jobs.json as a dict ({} when absent or unreadable)."""
    doc = jobs.read_lenient(Path(lab) / "jobs.json")
    return doc if isinstance(doc, dict) else {}


def total_cores(doc: dict | None = None) -> int:
    n = _int(_table(doc).get("total"))
    return n if n else (os.cpu_count() or 16)


def cmd_procs(cmd: Any) -> int | None:
    """The `--procs` of a command list, None when it has none."""
    toks = [str(x) for x in cmd] if isinstance(cmd, (list, tuple)) else []
    for i, t in enumerate(toks):
        if t == "--procs" and i + 1 < len(toks):
            return _int(toks[i + 1])
        if t.startswith("--procs="):
            return _int(t.split("=", 1)[1])
    return None


def job_cores(job: str, params: dict | None, doc: dict | None) -> int:
    """The cores a job of type `job` (with its recorded params) takes, at most the whole machine."""
    total = total_cores(doc)
    if job == "nightly-real":
        return total
    if job in jobs.NEW_KINDS:
        cfg = jobs.config_from(doc)[job]
        n = _int(cfg.get("cores"))
        if n is None:
            p = params or {}
            n = _int(p.get("procs")) or _int(cfg.get("procs")) or cmd_procs(p.get("cmd") or cfg.get("cmd"))
    else:
        t = _table(doc)
        n = _int(t.get(job)) if job in t else JOB_CORES.get(job)
    return total if n is None else max(0, min(total, n))


def report_cores(doc: dict | None) -> int:
    n = _int(_table(doc).get("report"))
    return REPORT_CORES if n is None else n


def loop_cores(doc: dict | None) -> int:
    n = _int(_table(doc).get("loop"))
    return LOOP_CORES if n is None else n


def filler_cfg(doc: dict | None) -> dict:
    return jobs.config_from(doc)[FILLER_KIND]


def filler_on(doc: dict | None) -> bool:
    """The selfplay kind runs as the filler (beside the jobs) instead of as a ladder job."""
    cfg = filler_cfg(doc)
    return bool(cfg.get("enabled") and cfg.get("filler"))


def floor(doc: dict | None) -> int:
    n = _int(filler_cfg(doc).get("floor"))
    return DEFAULT_FLOOR if n is None else max(1, n)


def filler_size(total: int, busy: int, floor_: int) -> int:
    """The cores the filler may use: what is left of `total` after `busy`, or 0 when that is below the floor."""
    left = total - busy
    return left if left >= floor_ else 0


def filler_cmd(cmd: list[str], cores: int, explore: list[str] | None = None) -> list[str]:
    """The selfplay command as the filler runs it: no marker flags, no fixed generation, no exploring flags unless `explore` gives
    them, `--record`, and `--procs` (and `--server-procs` when it plays through the prior server) set to `cores`. ValueError when it
    cannot be a recording block under the current agent (no `emu` or no `--agent-root`)."""
    toks = [str(x) for x in cmd]
    if "emu" not in toks:
        raise ValueError("the selfplay command is not a gen0.py emu block")
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        name = t.split("=", 1)[0]
        if name in DROP:
            i += 1 if ("=" in t or not DROP[name]) else 2
            continue
        out.append(t)
        i += 1
    if not any(x == "--agent-root" or x.startswith("--agent-root=") for x in out):
        raise ValueError("the selfplay command has no --agent-root (the filler plays agent/current.json)")
    if "--record" not in out:
        out.append("--record")
    out += ["--procs", str(int(cores))]
    if any(x == "--prior-server" or x.startswith("--prior-server=") for x in out):
        out += ["--server-procs", str(int(cores))]
    if explore:
        out += [str(x) for x in explore]
    return out


def night_due_at(lab: Path, doc: dict | None) -> float | None:
    """Epoch seconds the night is due (None when nightly-real is off): LAB/nightly/last-ok + every_h (0, long ago, when it never
    ran), or the retry after a skipped attempt (last-skip + jobs.NIGHTLY_RETRY_H) when that is later. The rule of jobs.due."""
    cfg = jobs.config_from(doc)["nightly-real"]
    if not cfg.get("enabled"):
        return None
    nd = Path(lab) / "nightly"
    try:
        t = (nd / "last-ok").stat().st_mtime + float(cfg.get("every_h", 20.0)) * 3600
    except OSError:
        t = 0.0
    try:
        t = max(t, (nd / "last-skip").stat().st_mtime + jobs.NIGHTLY_RETRY_H * 3600)
    except OSError:
        pass
    return t


def night_hold(lab: Path, doc: dict | None, now: float | None = None, starting: bool = True) -> str | None:
    """Why the filler may not start (`starting`) or must stop (a running block); None when it may run. Not while the night's
    marker is fresh; no start within `night_margin_min` of the night's due time; a running block stops STOP_BEFORE_S before it."""
    now = time.time() if now is None else now
    if jobs.nightly_active(lab):
        return "a nightly block is running (LAB/nightly/active)"
    due = night_due_at(lab, doc)
    if due is None:
        return None
    margin = float(filler_cfg(doc).get("night_margin_min", DEFAULT_MARGIN_MIN)) * 60 if starting else STOP_BEFORE_S
    if now >= due - margin:
        when = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(due))
        if not starting:
            return f"the night is due at {when}"
        return f"the night is due at {when}" + ("" if now >= due else f" (no start within {margin / 60:g} min of it)")
    return None


def _line_time(field: str) -> float | None:
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%MZ"):
        try:
            return dt.datetime.strptime(field, fmt).replace(tzinfo=dt.UTC).timestamp()
        except ValueError:
            continue
    return None


def quiet_blocker(path: Path, doc: dict | None, now: float | None = None) -> str | None:
    """The tag of a live QUIET.lock line the filler yields to; None when there is none. A line under QUIET_STALE_H old (its own
    time, else the file's) counts unless its tag is in `quiet_ignore` (default: the loop's and the daytime score's)."""
    now = time.time() if now is None else now
    ignore = filler_cfg(doc).get("quiet_ignore")
    ignore = tuple(ignore) if isinstance(ignore, (list, tuple)) else DEFAULT_QUIET_IGNORE
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        mtime = Path(path).stat().st_mtime
    except OSError:
        return None
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] in ignore:
            continue
        t = (_line_time(parts[1]) if len(parts) > 1 else None) or mtime
        if (now - t) / 3600 < QUIET_STALE_H:
            return parts[0]
    return None


def loop_state(status: Path, alive: Callable[[int], bool] | None = None) -> tuple[int | None, bool]:
    """(pid, stepping) of the always-on loop from its status file (rl/gym_loop.py writes `pid` and `phase`): pid None when there is
    no file or its process is gone; stepping when its phase uses its worker processes."""
    doc = jobs.read_lenient(Path(status))
    if not isinstance(doc, dict):
        return None, False
    pid = _int(doc.get("pid"))
    if not pid or (alive is not None and not alive(pid)):
        return None, False
    return pid, str(doc.get("phase") or "") in LOOP_BUSY
