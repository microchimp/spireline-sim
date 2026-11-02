#!/usr/bin/env python3
"""The lab's clock: where a week of wall time went, and which steps spent it on nothing (reset order D45, file-opens report).

    python tools/lab/clock_report.py [--lab D:/spireline/referee/lab] [--days 7] [--all] [--json out.json]

Reads LAB/runner.log. The runner writes one line per finished command:

    2026-11-02T04:22:04Z groups/groups: rc=0 in 1727.2s cpu=412.3s reads=88420 read_mb=2104.5 write_mb=3.1 out=same

`in` is wall seconds; `cpu` the CPU seconds of the command and its children (a Job Object of its own, tools/lab/winsys.py);
`reads` and `read_mb` the system's read operations and megabytes read (Windows keeps no count of files opened: a small file
costs one or two reads), `write_mb` the megabytes written; `out` whether the output differs from the last run's, by the step's
skip stamp (LAB/stamps/<step>.json, tools/lab/stepcache.py): `changed`, `same`, `skipped` (the command found its inputs unchanged
and ended at once) or `?` (it keeps no stamp, or failed). Lines from before the runner wrote these figures carry only the wall.

A step is the `step/command` of the line with its trailing number made N (batch-03/scen-03 is batch-N/scen-N). Over the last
`--days` (7), a step is flagged when

  IDLE   its wall is more than three times its CPU (it waits: on a disk, on a game, on a lock), at least five minutes in all
  SAME5  its last five runs that could tell all left the output they found (a step that runs and changes nothing)
  LONG   its mean wall, over the runs that did work, is above 20 minutes

The steps whose purpose is the long run itself (BY_DESIGN: training, the nightly real-game block, self-play, scoring, the
benches) are listed with their flags marked `by design` and are not counted as alarms unless they also show IDLE or SAME5.
A run that ended at once because its inputs were unchanged (`out=skipped`) is counted but is neither wall nor CPU.
Nothing here writes except `--json`; tools/fidelity/readiness.py puts the flagged steps in readiness.json under lab.clock.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_LAB = Path("D:/spireline/referee/lab")
DAYS = 7
IDLE_RATIO = 3.0
IDLE_MIN_WALL_S = 300.0
LONG_S = 20 * 60.0
SAME_RUNS = 5
BY_DESIGN = ("train/train", "nightly-real/nightly-real", "selfplay/selfplay", "score-real/score-real", "bench-env/bench-env", "bench-emu/bench-emu")

_LINE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) ([\w.-]+)/([\w.-]+): rc=(-?\d+) in ([\d.]+)s(?: (.*))?$")
_SKIP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) ([\w.-]+)/([\w.-]+): skipped \(")
_NUM_END = re.compile(r"\d+$")


def step_key(step: str, cmd: str) -> str:
    """step/command with the trailing number of each part made N: the batches of a job are one step."""
    return f"{_NUM_END.sub('N', step)}/{_NUM_END.sub('N', cmd)}"


def _ts(s: str) -> float:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


def _num(x: str | None) -> float | None:
    try:
        return float(str(x).removesuffix("s"))
    except (TypeError, ValueError):
        return None


def parse_line(line: str) -> dict[str, Any] | None:
    """One finished command of the log: {t, key, rc, wall, cpu, reads, read_mb, write_mb, out}; None for any other line.
    Figures the line does not carry (older lines, `?`) are None; `skipped` lines (the command was not started) are returned
    with out='not started'."""
    line = line.rstrip("\r\n")
    m = _LINE.match(line)
    if m:
        t, step, cmd, rc, wall, rest = m.groups()
        kv = dict(p.split("=", 1) for p in (rest or "").replace("(FAILED)", "").split() if "=" in p)
        return {"t": _ts(t), "key": step_key(step, cmd), "rc": int(rc), "wall": float(wall), "cpu": _num(kv.get("cpu")),
                "reads": _num(kv.get("reads")), "read_mb": _num(kv.get("read_mb")), "write_mb": _num(kv.get("write_mb")),
                "out": kv.get("out") or "?"}
    s = _SKIP.match(line)
    if s:
        return {"t": _ts(s.group(1)), "key": step_key(s.group(2), s.group(3)), "rc": None, "wall": 0.0, "cpu": None, "reads": None,
                "read_mb": None, "write_mb": None, "out": "not started"}
    return None


def analyse(log_text: str, now: float | None = None, days: int = DAYS, by_design: tuple[str, ...] = BY_DESIGN) -> dict[str, Any]:
    """The clock of the last `days` days of a runner log: per step its runs, wall, CPU, I/O and flags."""
    now = time.time() if now is None else now
    lo = now - days * 86400
    rows = [r for r in (parse_line(x) for x in log_text.splitlines()) if r and lo <= r["t"] <= now]
    steps: dict[str, dict[str, Any]] = {}
    for r in sorted(rows, key=lambda r: r["t"]):
        e = steps.setdefault(r["key"], {"runs": 0, "failed": 0, "not_started": 0, "self_skipped": 0, "wall": 0.0, "work_runs": 0,
                                        "wall_cpu": 0.0, "cpu": 0.0, "cpu_runs": 0, "read_mb": 0.0, "reads": 0.0, "write_mb": 0.0,
                                        "outs": [], "last_utc": ""})
        if r["out"] == "not started":
            e["not_started"] += 1
            continue
        e["runs"] += 1
        e["failed"] += 1 if r["rc"] else 0
        e["last_utc"] = dt.datetime.fromtimestamp(r["t"], dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        if r["out"] == "skipped":  # ended at once on its own stamp: no work, no wall to speak of
            e["self_skipped"] += 1
            continue
        e["work_runs"] += 1
        e["wall"] += r["wall"]
        if r["cpu"] is not None:
            e["cpu_runs"] += 1
            e["cpu"] += r["cpu"]
            e["wall_cpu"] += r["wall"]
        for k in ("reads", "read_mb", "write_mb"):
            e[k] += r[k] or 0.0
        if r["out"] in ("changed", "same"):
            e["outs"].append(r["out"])
    out: dict[str, Any] = {}
    for key, e in steps.items():
        mean = e["wall"] / e["work_runs"] if e["work_runs"] else 0.0
        ratio = (e["wall_cpu"] / e["cpu"]) if e["cpu"] > 0 else None
        last = e["outs"][-SAME_RUNS:]
        flags = []
        if ratio is not None and e["cpu_runs"] and ratio > IDLE_RATIO and e["wall_cpu"] >= IDLE_MIN_WALL_S:
            flags.append("IDLE")
        if len(last) == SAME_RUNS and all(x == "same" for x in last):
            flags.append("SAME5")
        if e["work_runs"] and mean > LONG_S:
            flags.append("LONG")
        design = key in by_design
        alarm = [f for f in flags if not (design and f == "LONG")]
        out[key] = {"runs": e["runs"], "work_runs": e["work_runs"], "self_skipped": e["self_skipped"], "not_started": e["not_started"],
                    "failed": e["failed"], "wall_mean_s": round(mean, 1), "wall_total_h": round(e["wall"] / 3600, 2),
                    "cpu_total_h": round(e["cpu"] / 3600, 2) if e["cpu_runs"] else None,
                    "wall_over_cpu": round(ratio, 2) if ratio is not None else None, "cpu_runs": e["cpu_runs"],
                    "reads": int(e["reads"]) if e["cpu_runs"] else None, "read_mb": round(e["read_mb"], 1) if e["cpu_runs"] else None,
                    "write_mb": round(e["write_mb"], 1) if e["cpu_runs"] else None,
                    "last_outputs": last, "last_utc": e["last_utc"], "flags": flags, "by_design": design, "alarm": bool(alarm)}
    flagged = sorted((k for k, v in out.items() if v["flags"]), key=lambda k: (-out[k]["alarm"], -out[k]["wall_total_h"], k))
    return {"window_days": days, "from_utc": dt.datetime.fromtimestamp(lo, dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "to_utc": dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "runs": sum(v["runs"] for v in out.values()), "steps": len(out),
            "steps_with_cpu": sum(1 for v in out.values() if v["cpu_runs"]),
            "alarms": sum(1 for v in out.values() if v["alarm"]), "flagged": flagged, "by_step": out,
            "wall_total_h": round(sum(v["wall_total_h"] for v in out.values()), 1)}


def _cell(v: Any, fmt: str = "{}") -> str:
    return "-" if v is None else fmt.format(v)


def render(rep: dict[str, Any], show_all: bool = False) -> str:
    """A short table: the flagged steps (all steps with show_all), the worst first."""
    keys = list(rep["by_step"]) if show_all else rep["flagged"]
    keys = sorted(keys, key=lambda k: (-rep["by_step"][k]["alarm"], -rep["by_step"][k]["wall_total_h"], k))
    head = (f"Lab clock {rep['from_utc']} to {rep['to_utc']} ({rep['window_days']} days): {rep['runs']} step runs, {rep['steps']} steps, "
            f"{rep['wall_total_h']} h of wall; {rep['alarms']} flagged"
            + (f"; {rep['steps'] - rep['steps_with_cpu']} steps have no CPU figures yet (the runner writes them since its restart)"
               if rep["steps_with_cpu"] < rep["steps"] else ""))
    cols = ("step", "runs", "no-op", "wall mean s", "wall h", "wall/cpu", "read MB", "reads", "last outputs", "flags")
    rows = []
    for k in keys:
        v = rep["by_step"][k]
        rows.append((k, str(v["work_runs"]), str(v["self_skipped"] + v["not_started"]), f"{v['wall_mean_s']:.0f}", f"{v['wall_total_h']:.1f}",
                     _cell(v["wall_over_cpu"], "{:.1f}"), _cell(v["read_mb"], "{:.0f}"), _cell(v["reads"]),
                     "".join("S" if x == "same" else "C" for x in v["last_outputs"]) or "-",
                     " ".join(v["flags"]) + (" (by design)" if v["by_design"] and v["flags"] else "")))
    if not rows:
        return head + "\nnothing flagged"
    w = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    lines = [head, "  ".join(c.ljust(w[i]) for i, c in enumerate(cols)).rstrip()]
    lines += ["  ".join(r[i].ljust(w[i]) for i in range(len(cols))).rstrip() for r in rows]
    lines.append("flags: IDLE wall over 3x CPU; SAME5 last five runs left the output unchanged (S same, C changed); LONG mean wall over 20 min; "
                 "'no-op' = ended at once on its stamp or not started")
    return "\n".join(lines)


def report_for(lab: Path, now: float | None = None, days: int = DAYS) -> dict[str, Any] | None:
    """The analysis of LAB/runner.log, None when there is no log."""
    try:
        text = (Path(lab) / "runner.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return analyse(text, now, days)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--lab", type=Path, default=DEFAULT_LAB)
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--all", action="store_true", help="every step, not only the flagged ones")
    ap.add_argument("--json", type=Path, help="write the analysis here")
    a = ap.parse_args(argv)
    rep = report_for(a.lab, days=max(1, a.days))
    if rep is None:
        print(f"no runner.log under {a.lab.name}", file=sys.stderr)
        return 2
    print(render(rep, a.all))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
