"""Lab run: heavy work started by hand goes through here, and every core-hour the PC spends is written down with its purpose and its reader.

Why (the 2026-11-18 review of the machine): at least 900 of about 5,000 logged core-hours bought nothing, and the half of the PC's hours that
no log accounts for is work somebody started by hand. A rule kept in a note recurs; this one is kept in code. A **core-hour** is one of the
PC's 16 processor threads busy for an hour (the lab runner's own count: `cpu=` seconds in `LAB/runner.log`, from the Job Object).

    python tools/lab/labrun.py --tag <tag> --purpose "<why>" --consumer "<who reads the result>" [--quiet] [--cores N] -- <command...>
    python tools/lab/labrun.py --report [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--json]

The first form runs the command at BelowNormal priority in a Windows Job Object (the same pieces the lab runner uses, tools/lab/winsys.py;
every process the command starts is in the job, so the CPU seconds and the I/O counted are the whole tree's, and the tree ends with the
wrapper), then appends one JSON line to `LAB/hand-jobs.jsonl`:

    {"tag", "purpose", "consumer", "cmd" (user paths made anonymous), "start", "end" (UTC), "wall_s", "cpu_s", "cores_used" (cpu / wall),
     "cores_declared" (`--cores`, else null), "core_hours", "rc", "quiet", "read_mb", "write_mb"}

`--quiet` is for a measurement that needs the machine to itself: it takes `<tag> <UTC> pid=<Windows pid>` in `QUIET.lock` (the format
`rl/quiet_lock.py` writes and the lab and the fight loop read), keeps it fresh every 15 minutes while the command runs and removes it
after. It refuses at once (exit 75, nothing run) when another holder's line under two hours old stands, the always-on loop's `gym-loop`
apart (it pauses its own step for a line). Whatever the flags, nothing starts while a nightly real-game block runs (`LAB/nightly/active`
fresh, or a `lab-nightly` line in QUIET.lock): exit 75 and the reason; `--allow-night` overrides, and is not for a task.

`--cores N` says how many cores the command is expected to use. It is recorded beside the cores actually used, and a run that used more
than 25 in 100 over it says so on stderr. It does not limit anything.

The command's exit code is labrun's. A command that cannot be started is rc 127 with the reason on stderr (and still written down).

`--report` reads `LAB/runner.log` (the runner's `<step>/<command>: rc=... in ...s cpu=...s` lines: its jobs) and `LAB/hand-jobs.jsonl`
together and prints core-hours by tag, by purpose and by consumer, the failed runs' share of each, and how much of the PC's thread-hours
the two logs account for. A runner line from before 2026-11-03 has no `cpu=`: it is estimated from its wall time at the median cores
measured later for the same step and marked `~`. Not in either log: the always-on fight loop (its own phase log) and anything run outside
the lab and this wrapper, which is the point of the report: the gap shows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import quiet_lock  # noqa: E402 -- rl/ is on sys.path (above)

from tools.lab import cores, jobs, winsys  # noqa: E402

DEFAULT_LAB = "D:/spireline/referee/LAB"
HAND_JOBS = "hand-jobs.jsonl"
REFUSED = 75  # exit code of a run that was not started (BSD's EX_TEMPFAIL)
NOT_STARTED = 127
IGNORED_TAGS = ("gym-loop",)  # the always-on fight loop yields to a QUIET.lock line by itself, so its own line does not block a run
HEARTBEAT_S = 900.0
OVER_DECLARED = 1.25
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
HOME = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+", re.IGNORECASE)  # a results file holds no user paths
RUNNER_STEP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) (\S+?)/(\S+?): rc=(-?\d+) in ([\d.]+)s(?: cpu=([\d.]+)s)?")


def utc(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(time.time() if t is None else t, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_when(s: str) -> float:
    """`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM[:SS]Z` as epoch seconds (UTC); ValueError otherwise."""
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%MZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return dt.datetime.strptime(s, fmt).replace(tzinfo=dt.UTC).timestamp()
        except ValueError:
            continue
    raise ValueError(f"not a date: {s!r} (YYYY-MM-DD or YYYY-MM-DDTHH:MMZ)")


def scrub(text: str, limit: int = 600) -> str:
    """A command line for a results file: user paths made anonymous, a long line cut."""
    text = HOME.sub("<home>", text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def hand_jobs_path(lab: Path | str) -> Path:
    return Path(lab) / HAND_JOBS


# ---------------------------------------------------------------- running a command

def quiet_blockers(path: Path | str, own_tag: str, now: float | None = None, ignore: Sequence[str] = IGNORED_TAGS) -> list[str]:
    """The tags of the live QUIET.lock lines (a line under `quiet_lock.STALE_H` old, by its own time, else the file's) a quiet run must
    not start beside: every tag but the ignored ones; a line carrying our own tag counts (two runs of one tag would share one line)."""
    now = time.time() if now is None else now
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        mtime = p.stat().st_mtime
    except OSError:
        return []
    out: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] in ignore:
            continue
        t = (quiet_lock.parse_utc(parts[1]) if len(parts) > 1 else None) or mtime
        if (now - t) / 3600 < quiet_lock.STALE_H:
            out.append(parts[0])
    return out


def night_block(lab: Path | str, qpath: Path | str, now: float | None = None) -> str | None:
    """Why nothing heavy may start: a nightly real-game block is running (LAB/nightly/active fresh) or holds a `lab-nightly` QUIET.lock line."""
    if jobs.nightly_active(Path(lab)):
        return "a nightly real-game block is running (LAB/nightly/active)"
    if "lab-nightly" in quiet_blockers(qpath, "", now, ignore=()):
        return "the nightly real-game block holds QUIET.lock (a lab-nightly line)"
    return None


def run_command(argv: Sequence[str], *, tag: str, purpose: str, consumer: str, lab: Path | str = DEFAULT_LAB, quiet: bool = False,
                cores_declared: int | None = None, quiet_lock_path: Path | str | None = None, allow_night: bool = False,
                sleep_poll_s: float = 0.2) -> tuple[int, dict[str, Any] | None]:
    """Run `argv` as the module says and write its line. Returns (exit code, the record); the record is None when the run was refused (rc 75:
    nothing was started, nothing is recorded)."""
    lab = Path(lab)
    qpath = Path(quiet_lock_path) if quiet_lock_path else lab.parent / "QUIET.lock"
    if not TAG_RE.match(tag):
        print(f"labrun: tag {tag!r} is not one word (letters, digits, - _ . +): it is the QUIET.lock line's tag", file=sys.stderr)
        return REFUSED, None
    if not allow_night:
        why = night_block(lab, qpath)
        if why:
            print(f"labrun: not started: {why}. Nothing heavy runs beside the real game.", file=sys.stderr)
            return REFUSED, None
    lock = quiet_lock.QuietLock(qpath, tag) if quiet else None
    if lock is not None:
        others = quiet_blockers(qpath, tag)
        if others:
            print(f"labrun: not started: QUIET.lock has a live line from {', '.join(sorted(set(others)))}. A quiet run waits for it to go.", file=sys.stderr)
            return REFUSED, None
        lock.announce()
        lock.start_heartbeat(HEARTBEAT_S)
    winsys.set_below_normal()  # this process too: its own work is a poll loop, and the child inherits the class where the job cannot force it
    t0, start = time.time(), utc()
    rc, cpu, read_mb, write_mb, note = NOT_STARTED, None, None, None, None
    job = winsys.JobObject(kill_on_close=True, priority_class=winsys.BELOW_NORMAL_PRIORITY_CLASS)
    try:
        try:
            proc = winsys.suspended_popen([str(x) for x in argv], job)
        except OSError as e:
            note = f"could not start: {type(e).__name__}: {e}"
            print(f"labrun: {note}", file=sys.stderr)
        else:
            try:
                while True:
                    try:
                        rc = proc.wait(timeout=sleep_poll_s)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            except KeyboardInterrupt:
                job.terminate(130)
                rc = 130
                note = "interrupted"
            acct = job.accounting()
            if acct is not None:
                cpu = round(acct["cpu_secs"], 2)
                read_mb, write_mb = round(acct["read_bytes"] / 1e6, 1), round(acct["write_bytes"] / 1e6, 1)
            job.terminate(0)  # anything the command left running ends with it
    finally:
        job.close()
        if lock is not None:
            lock.release()
    wall = time.time() - t0
    rec: dict[str, Any] = {"tag": tag, "purpose": purpose, "consumer": consumer, "cmd": scrub(" ".join(str(x) for x in argv)),
                           "start": start, "end": utc(), "wall_s": round(wall, 2), "cpu_s": cpu,
                           "cores_used": round(cpu / wall, 2) if cpu is not None and wall > 0 else None,
                           "cores_declared": cores_declared, "core_hours": round(cpu / 3600, 4) if cpu is not None else None,
                           "rc": rc, "quiet": bool(quiet), "read_mb": read_mb, "write_mb": write_mb}
    if note:
        rec["note"] = note
    if cores_declared and rec["cores_used"] is not None and rec["cores_used"] > cores_declared * OVER_DECLARED:
        print(f"labrun: the command used {rec['cores_used']:.1f} cores on average, over the {cores_declared} declared", file=sys.stderr)
    append_record(lab, rec)
    cph = f"{rec['core_hours']:.2f} core-hours" if rec["core_hours"] is not None else "cpu not measured"
    print(f"labrun: {tag} rc={rc}, wall {wall:.1f}s, {cph}; written to {HAND_JOBS}", file=sys.stderr)
    return rc, rec


def append_record(lab: Path | str, rec: dict[str, Any]) -> None:
    """One line, one write: two jobs ending together do not interleave (a line is far under a page)."""
    p = hand_jobs_path(lab)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------- the report

BUILD_STEPS = {"tree", "build", "driver", "native", "restore", "game", "checkout", "publish-native"}
FIDELITY_STEPS = {"batch", "groups", "kits", "targets", "invariants", "mutation", "checks", "corpus", "corpus-import", "runs", "probes",
                  "retest", "refset", "report", "fights"}
PURPOSE_OF = {"teacher": "plain runs / teacher (lab)", "train": "judgements and fits (lab train job)", "selfplay": "self-play practice (lab)",
              "student": "student fits (lab)", "nightly-real": "real game", "score-real": "real game", "bench-env": "benches",
              "bench-emu": "benches"}


def runner_purpose(step: str) -> str:
    """What a runner step is for: the review's table (2026-11-18, analyst C) as a function of the step name."""
    base = re.sub(r"-\d+$", "", step)
    if base in FIDELITY_STEPS:
        return "fidelity and bug hunting"
    if base in BUILD_STEPS:
        return "builds (presort)"
    return PURPOSE_OF.get(base, "other: " + base)


def read_runner(lab: Path | str) -> list[dict[str, Any]]:
    """Every step line of LAB/runner.log, the rolled-over runner.log.1 first (the runner rolls the log at 5 MB): `{t, step, cmd, rc, wall, cpu
    (None before 2026-11-03)}`."""
    out = []
    for name in ("runner.log.1", "runner.log"):
        try:
            text = (Path(lab) / name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            m = RUNNER_STEP.match(line)
            if m:
                out.append({"t": parse_when(m.group(1)), "step": m.group(2), "cmd": m.group(3), "rc": int(m.group(4)), "wall": float(m.group(5)),
                            "cpu": float(m.group(6)) if m.group(6) else None})
    return out


def read_hand(lab: Path | str) -> list[dict[str, Any]]:
    """The lines of LAB/hand-jobs.jsonl that are records (a damaged line is skipped)."""
    try:
        lines = hand_jobs_path(lab).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("tag") and r.get("end"):
            out.append(r)
    return out


def measured_cores(steps: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    """The median cores (cpu / wall) each (step, command) used in the lines that have `cpu=`, for the lines that do not."""
    ratios: dict[tuple[str, str], list[float]] = defaultdict(list)
    for s in steps:
        if s["cpu"] is not None and s["wall"] > 30:
            ratios[(s["step"], s["cmd"])].append(s["cpu"] / s["wall"])
    return {k: statistics.median(v) for k, v in ratios.items()}


def build_report(lab: Path | str, since: float | None = None, until: float | None = None, now: float | None = None) -> dict[str, Any]:
    """The report as data: rows per tag, purpose and consumer, the logged total and the PC's thread-hours over the window."""
    lab = Path(lab)
    now = time.time() if now is None else now
    steps, hand = read_runner(lab), read_hand(lab)
    med = measured_cores(steps)
    doc = cores.load_doc(lab)
    total_threads = cores.total_cores(doc)
    items: list[dict[str, Any]] = []
    for s in steps:
        if (since is not None and s["t"] < since) or (until is not None and s["t"] >= until):
            continue
        est = s["cpu"] is None
        if est:
            ratio = med.get((s["step"], s["cmd"]))
            if ratio is None:
                ratio = med.get((s["step"], s["step"]))
            if ratio is None:
                ratio = float(cores.job_cores(s["step"], None, doc)) if s["step"] in cores.JOB_CORES or s["step"] in jobs.NEW_KINDS else 1.0
            hours = s["wall"] * ratio / 3600
        else:
            hours = s["cpu"] / 3600
        items.append({"source": "lab", "tag": re.sub(r"-\d+$", "", s["step"]), "purpose": runner_purpose(s["step"]), "consumer": "the lab", "hours": hours,
                      "wall_h": s["wall"] / 3600, "failed": s["rc"] != 0, "est": est, "t": s["t"]})
    for r in hand:
        try:
            t = parse_when(str(r["end"]))
        except ValueError:
            continue
        if (since is not None and t < since) or (until is not None and t >= until):
            continue
        ch = r.get("core_hours")
        hours = float(ch) if isinstance(ch, (int, float)) else float(r.get("wall_s") or 0) / 3600  # no CPU figure: one core for the wall time
        items.append({"source": "hand", "tag": str(r["tag"]), "purpose": str(r.get("purpose") or "(none)"),
                      "consumer": str(r.get("consumer") or "(none)"), "hours": hours, "wall_h": float(r.get("wall_s") or 0) / 3600,
                      "failed": r.get("rc") not in (0, None), "est": not isinstance(ch, (int, float)), "t": t})
    lo = since if since is not None else min((i["t"] for i in items), default=now)
    hi = until if until is not None else now
    window_h = max(0.0, (hi - lo) / 3600)
    logged = sum(i["hours"] for i in items)

    def table(key: str) -> list[dict[str, Any]]:
        agg: dict[tuple[str, str], dict[str, Any]] = {}
        for i in items:
            row = agg.setdefault((i[key], i["source"]), {"name": i[key], "source": i["source"], "runs": 0, "failed": 0, "wall_h": 0.0, "core_h": 0.0,
                                                          "failed_core_h": 0.0, "est_core_h": 0.0})
            row["runs"] += 1
            row["wall_h"] += i["wall_h"]
            row["core_h"] += i["hours"]
            if i["failed"]:
                row["failed"] += 1
                row["failed_core_h"] += i["hours"]
            if i["est"]:
                row["est_core_h"] += i["hours"]
        return sorted(agg.values(), key=lambda r: -r["core_h"])

    hand_h = sum(i["hours"] for i in items if i["source"] == "hand")
    return {"lab": "LAB", "since": utc(lo), "until": utc(hi), "window_h": round(window_h, 1), "threads": total_threads,
            "thread_hours": round(window_h * total_threads, 1), "logged_core_h": round(logged, 1), "hand_core_h": round(hand_h, 1),
            "failed_core_h": round(sum(i["hours"] for i in items if i["failed"]), 1), "est_core_h": round(sum(i["hours"] for i in items if i["est"]), 1),
            "share_logged": round(logged / (window_h * total_threads), 3) if window_h > 0 else None,
            "by_tag": table("tag"), "by_purpose": table("purpose"), "by_consumer": table("consumer")}


def render(rep: dict[str, Any]) -> str:
    def block(title: str, rows: list[dict[str, Any]]) -> list[str]:
        out = ["", title, f"  {'name':<44} {'from':<5} {'runs':>6} {'failed':>6} {'wall h':>8} {'core-h':>9} {'of it failed':>12}"]
        for r in rows:
            tilde = "~" if r["est_core_h"] > 0 else " "
            out.append(f"  {r['name'][:44]:<44} {r['source']:<5} {r['runs']:>6} {r['failed']:>6} {r['wall_h']:>8.1f} {r['core_h']:>8.1f}{tilde} "
                       f"{r['failed_core_h']:>12.1f}")
        return out

    share = rep["share_logged"]
    lines = [f"Lab books {rep['since'][:10]} to {rep['until'][:10]} ({rep['window_h']:g} hours; ~ marks core-hours estimated from wall time)",
             f"The PC's {rep['threads']} threads over the window: {rep['thread_hours']:,.0f} thread-hours. Logged: {rep['logged_core_h']:,.0f} core-hours"
             + (f" ({share * 100:.0f} in 100)" if share is not None else "")
             + f", of which {rep['hand_core_h']:,.0f} by hand through labrun, {rep['failed_core_h']:,.0f} in runs that failed, {rep['est_core_h']:,.0f} estimated.",
             "Not in either log: the always-on fight loop (its own phase log) and anything run outside the lab runner and labrun."]
    lines += block("By tag (the runner's step, or the --tag of a hand job)", rep["by_tag"])
    lines += block("By purpose", rep["by_purpose"])
    lines += block("By consumer (who reads the result)", rep["by_consumer"])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- command line

def split_command(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """(options, command): everything before the first `--` and everything after it."""
    args = list(argv)
    if "--" in args:
        i = args.index("--")
        return args[:i], args[i + 1:]
    return args, []


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run heavy work by hand at BelowNormal in a Job Object and write it down "
                                 "(LAB/hand-jobs.jsonl); or report core-hours by tag, purpose and consumer.",
                                 epilog="usage: labrun.py --tag T --purpose P --consumer C [--quiet] [--cores N] -- COMMAND ...    |    labrun.py --report [--since DATE]")
    ap.add_argument("--tag", help="one word: the job's name, and its QUIET.lock tag with --quiet")
    ap.add_argument("--purpose", help="why the work runs (the decision it serves)")
    ap.add_argument("--consumer", help="who reads the result (a lane, a memo, a gate)")
    ap.add_argument("--quiet", action="store_true", help="hold a QUIET.lock line while the command runs (a measurement that needs the machine)")
    ap.add_argument("--cores", type=int, default=None, help="the cores the command is expected to use (recorded; nothing is limited)")
    ap.add_argument("--lab", default=DEFAULT_LAB, help="the lab folder (default: %(default)s)")
    ap.add_argument("--quiet-lock", default=None, help="the shared QUIET.lock (default: next to the lab folder)")
    ap.add_argument("--allow-night", action="store_true", help="start although a nightly real-game block runs (never in a task)")
    ap.add_argument("--report", action="store_true", help="print core-hours by tag, purpose and consumer instead of running a command")
    ap.add_argument("--since", help="with --report: from this UTC date (YYYY-MM-DD or YYYY-MM-DDTHH:MMZ)")
    ap.add_argument("--until", help="with --report: up to this UTC date")
    ap.add_argument("--json", action="store_true", help="with --report: the figures as JSON")
    opts, command = split_command(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(opts)
    if a.report:
        if command:
            ap.error("--report takes no command")
        try:
            rep = build_report(a.lab, parse_when(a.since) if a.since else None, parse_when(a.until) if a.until else None)
        except ValueError as e:
            ap.error(str(e))
        print(json.dumps(rep, indent=1) if a.json else render(rep), end="" if not a.json else "\n")
        return 0
    missing = [n for n in ("tag", "purpose", "consumer") if not getattr(a, n)]
    if missing or not command:
        ap.error("a run needs " + ", ".join("--" + n for n in ("tag", "purpose", "consumer")) + " and a command after `--`"
                 + (f" (missing: {', '.join('--' + m for m in missing)}{'' if command else ' and the command'})" if missing or not command else ""))
    rc, _ = run_command(command, tag=a.tag, purpose=a.purpose, consumer=a.consumer, lab=a.lab, quiet=a.quiet, cores_declared=a.cores,
                        quiet_lock_path=a.quiet_lock, allow_night=a.allow_night)
    return rc


if __name__ == "__main__":
    sys.exit(main())
