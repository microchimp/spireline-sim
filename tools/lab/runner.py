#!/usr/bin/env python3
"""The fix lab's runner: turns idle CPU into the information that makes backlog fixes cheap.

    python tools/lab/runner.py [--lab D:/spireline/referee/lab] [--tree D:/spireline/fork/wt-lab]
        [--game D:/spireline/referee/game-lab] [--trial MINUTES] [--once presort|fights|runs|probes|mutation|invariants]
        [--dry-run] [--skip PREFIX,...] [--no-guard user,load,...] [--always-on]

Started by the Windows scheduled task "Spireline fix lab" (tools/lab/register_task.ps1) from the fork's main
checkout, which it only reads; every job runs with the --tree worktree (detached at the fork's LOCAL main, never
fetched) as cwd. See tools/lab/README.md and referee/RUNBOOK.md "Fix lab".

Jobs, by value: presort (when local main changed: move and rebuild the tree, codex walk, refset score, corpus
check, then groups and kits), fights (one strategy of tools/lab/mix.py per job, chosen from LAB/mix.json: aimed
batches, shrink, first events, groups, kits), runs (the whole-runs strategy), probes, mutation, invariants. Every
finished job adds a row to LAB/yield.tsv (CPU minutes, fights, mismatches, unbuildable, groups, fix-ready groups).
A job is a folder LAB/runs/<UTC yyyymmddThhmm>-<job>/ with job.json (inputs, main sha, steps), logs/,
its outputs and done.rc (the exit code, written last). It is resumable: an interrupted job is resumed at its
first unfinished step if main has not moved, else closed with done.rc 3.

The runner makes no AI calls and no network use. It lowers its own priority to BelowNormal (children inherit
it), lives in a Job Object with KILL_ON_JOB_CLOSE (children die with it, however it ends), and gives the machine
back on user input, on sustained load from other processes, on LAB/PAUSE or backlog/main.lock, and when disk
runs short (tools/lab/guards.py). With --always-on the groups and kits of a finished job run beside the next job (Report, below),
one at a time, at the runner's BelowNormal priority; they stop for the guards, for a presort's tree (at once; after --report-cut-cap stops in a
row the presort waits for the report instead) and for a nightly window.
With --always-on the selfplay kind is the filler when LAB/jobs.json says `selfplay.filler` (D48, tools/lab/cores.py): practice blocks
run beside the job on the cores it leaves (a Job Object of their own that forces BelowNormal on every process in it, the prior
server included), with no train marker, shrunk or stopped when a job needs their cores, stopped for the night and for the guards;
they hold no QUIET.lock line and no main.lock. They run from a frozen copy of the lab tree (tools/lab/frozen.py, made once per
landing, outside the lab folder), not from the tree itself, so a presort that moves the tree does not stop them (D52; --filler-tree off:
the old way, from the tree, stopped for every presort). The load guard counts neither the filler (it is in the runner's Job Object)
nor the always-on fight loop's processes (--loop-status names its daemon).
backlog/main.lock stops only a running presort (the one job that touches git
state) and holds every new job; any other running job carries on under it. Exit 0 when it stopped for any of those or finished a --trial / --once; 1 on
a crash; 2 on bad arguments.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True  # before the runner's own imports: no __pycache__ in the checkout it only reads

from tools.lab import alerts, clock_report, cores, frozen, guards, jobs, mix, plans, prune, retest, winsys  # noqa: E402
from tools.lab import stepcache as sc  # noqa: E402

JOB_TYPES = ("presort", "fights", "runs", "probes", "mutation", "invariants", *jobs.NEW_KINDS)
ROTATION = ("fights", "fights", "probes", "fights", "mutation", "fights", "invariants")  # after presort, by value
REPORT_JOBS = ("presort", "fights", "runs", "probes", "invariants")  # the jobs that end with groups and kits (plans.report_steps)
TREE_ONCE_JOBS = ("probes", "mutation", "invariants")  # job types that run once per presorted tree (last_ok_sha); retest, a fights strategy, too
RAN_ON_TREE = "ran on this tree already"
SEED_LOOK_BACK = 150  # newest job folders read to seed last_ok_sha for a state file from before it existed
REPORT_TIMEOUT_S = 2 * 3600.0  # a report command that runs longer is stopped (the longest groups step seen took 65 minutes)
DEFAULTS = {
    "lab": "D:/spireline/referee/lab", "tree": "D:/spireline/fork/wt-lab", "game": "D:/spireline/referee/game-lab",
    "game_src": "D:/spireline/referee/game", "backlog_dir": "D:/spireline/referee/backlog",
    "codex": "D:/spireline/data/spire-codex/runs-v0.107.1.jsonl.gz", "decompiled": "D:/spireline/referee/decompiled",
    "records": "D:/spireline/referee/jobs", "loop_status": "D:/spireline/logs/gym-loop/status",
}
FILLER_RETRY_S = 120.0  # after a guard, the night, a QUIET.lock line or a failure stopped the filler, it waits this long to start again
FILLER_TICK_S = 5.0  # the filler's rules are read at most this often (jobs.json, the night's markers, QUIET.lock)
FILLER_PRESORT_LOOKAHEAD_S = 3600.0  # no practice block starts when a presort would be chosen within this long (D55 addendum 18)
MAIN_SHA_CACHE_S = 20.0  # the filler's look at the fork's local main (one git call) is repeated at most this often
# The groups and kits beside a job (the report) start only on an idle machine: not while one of these runs, and not under the cores it needs.
REPORT_HOLD_JOBS = ("train", "score-real", "nightly-real")
TRAIN_TAG = "lab-train"  # the QUIET.lock line a running train job (the judgement) holds: the practice loop and the builders yield to it
TRAIN_BEAT_S = 600.0  # ... refreshed this often (the fight loop reads a line over 2 h old as dead; the brief asks 30 minutes at most)
INDEFINITE = 4102444800.0  # 2100-01-01: the backoff of a kind that raised an alert lasts until the alert file is removed
# Stopping the filler asks first: STOP in the block's folder makes `gen0.py emu` write its finished runs to the shards, start no new run and
# end when the runs in flight have; the runner waits this long for that, then kills the Job Object as before. Only a block that wrote
# FILLER_HONORED (a gen0.py that reads STOP) is waited for; an older one is killed at once. A hard kill used to lose every run finished
# since the last 50-run flush (1,828 of the filler's 4,496 by 2026-11-04). rl/gen0.py STOP_FILE and STOP_HONORED_FILE are the same names.
FILLER_STOP_FILE = "STOP"
FILLER_HONORED = "stop-honored"
FILLER_STOP_GRACE_S = 25.0
FORBIDDEN = ("D:/spireline/src",)  # the reviewed upstream checkout: never a lab, tree or game path
RESUME_HOURS = 24.0
CLOCK_EVERY_S = 24 * 3600.0  # LAB/clock.md (tools/lab/clock_report.py over runner.log) is written again after this long
CHILD_ENV = {"PYTHONDONTWRITEBYTECODE": "1", "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_NOLOGO": "1", "PYTHONIOENCODING": "utf-8"}
STALE_LOCK_S = 3 * 900.0  # a `lab-*` QUIET.lock line is refreshed every 15 minutes; three missed refreshes make it stale
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_CONSOLE = 0x00000010


def hidden_console_kw() -> dict[str, Any]:
    """Popen keywords for a job command: a console of its own, hidden. CREATE_NO_WINDOW would leave the command with no
    console at all, so every console program it starts in turn (dotnet, python) opens a visible window of its own; with a
    hidden console they share it instead."""
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    return {"creationflags": CREATE_NEW_CONSOLE, "startupinfo": si}


def utc(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(time.time() if t is None else t, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write(path: Path, text: str) -> None:
    """Write whole or not at all (a reader never sees half a file); retries where Windows briefly locks the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for k in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (k + 1))
    os.replace(tmp, path)


def read_rc(p: Path) -> int | None:
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return None


def read_json(p: Path, default: Any) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    d = DEFAULTS
    ap.add_argument("--lab", default=d["lab"], help="the lab's root (outside every repo)")
    ap.add_argument("--tree", default=d["tree"], help="the worktree the jobs run in (detached at the fork's local main)")
    ap.add_argument("--game", default=d["game"], help="the lab's own game copy (made with referee/linkcopy.py when absent)")
    ap.add_argument("--game-src", default=d["game_src"], help="the verified game copy linkcopy makes --game from")
    ap.add_argument("--fork", default=str(ROOT), help="the fork's main checkout (read only: its local branch names the sha)")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--backlog-dir", default=d["backlog_dir"], help="holds main.lock, backlog-*.json, packets/")
    ap.add_argument("--quiet-lock", default=None, help="the shared QUIET.lock (default: next to the lab folder)")
    ap.add_argument("--backlog", default=None, help="backlog json to use (default: the newest backlog-*.json)")
    ap.add_argument("--codex", default=d["codex"])
    ap.add_argument("--decompiled", default=d["decompiled"])
    ap.add_argument("--records", default=d["records"], help="the recorded fights the invariant checks read")
    ap.add_argument("--trial", type=float, default=None, metavar="MINUTES", help="work for this long, then stop and exit 0")
    ap.add_argument("--once", choices=JOB_TYPES, default=None, help="run one job of this type, then exit")
    ap.add_argument("--always-on", action="store_true",
                    help="for the logon task: keyboard and mouse input and LAB/HALT pause the lab instead of ending the runner")
    ap.add_argument("--report-beside", choices=("auto", "on", "off"), default="auto",
                    help="run the groups and kits of a finished job beside the next job instead of as its last steps; auto: on with --always-on, "
                         "off with --once, --trial and --dry-run")
    ap.add_argument("--presort-report", choices=("stop", "wait"), default="stop",
                    help="a presort checks out, builds and publishes into the tree the running report (groups, kits) reads. stop: the report "
                         "is stopped at once and its jobs wait for the next one (a report is 40 to 70 minutes and nearly always running, so "
                         "waiting for it was most of a presort's wall time); wait: the presort waits for the report to end (the way until 2026-11-04)")
    ap.add_argument("--report-cut-cap", type=int, default=2, metavar="N",
                    help="with --presort-report stop: after N reports stopped by presorts in a row (none ended in between) the next presort "
                         "waits for the running report, so a report that keeps being cut still ends")
    ap.add_argument("--filler", choices=("auto", "on", "off"), default="auto",
                    help="run selfplay as the filler beside the jobs when LAB/jobs.json has selfplay.filler (D48); auto: on with --always-on, "
                         "off with --once, --trial and --dry-run")
    ap.add_argument("--filler-tree", default="auto", metavar="DIR|auto|off",
                    help="where the filler's frozen copies of the lab tree are made (auto: the folder `filler-tree` next to the lab folder); "
                         "off: the filler runs from the lab tree and a presort stops it (before D52)")
    ap.add_argument("--loop-status", default=d["loop_status"],
                    help="the always-on fight loop's status file (its daemon pid and phase): the load guard leaves its processes out")
    ap.add_argument("--rotation", choices=("config", "all"), default="config",
                    help="which fidelity-era kinds (fights, probes, mutation, invariants, runs) the runner may pick: config: LAB/jobs.json `rotation` (absent or "
                         "empty: none); all: every one (a test, a hand run). --once names its kind itself and ignores this")
    ap.add_argument("--reports", choices=("config", "all"), default="config",
                    help="which report steps (checks, corpus-import, groups, kits) run: config: LAB/jobs.json `reports` (a step not enabled there is off); "
                         "all: every one")
    ap.add_argument("--dry-run", action="store_true", help="print the guards, the next job and its plan; change nothing")
    ap.add_argument("--skip", default="", help="comma-separated step or command name prefixes to skip (codex, refset, corpus, ...)")
    ap.add_argument("--no-guard", default="", help=f"comma-separated guards to switch off for a test: {','.join(guards.GUARD_NAMES)}")
    ap.add_argument("--exit-with-parent", action="store_true", help="exit when the process that started the runner is gone")
    ap.add_argument("--fights", type=int, default=300, help="fights in one fights job")
    ap.add_argument("--batch", type=int, default=50, help="fights per scenarios.py call (a safe point between calls)")
    ap.add_argument("--servers", type=int, default=8, help="game servers per fights batch")
    ap.add_argument("--shrink", type=int, default=1000, help="diverging fights shrunk per batch (the default: every one)")
    ap.add_argument("--shrink-cap", type=int, default=160, help="game fights one diverging fight's shrink may cost")
    ap.add_argument("--shrink-budget", type=int, default=6000, help="game fights the shrinking of one job may cost in all")
    ap.add_argument("--shrink-minutes", type=float, default=45.0, help="minutes after which a job starts no new shrink round")
    ap.add_argument("--trace", type=int, default=1000, help="diverging fights traced per batch (the default: every one)")
    ap.add_argument("--trace-minutes", type=float, default=20.0, help="minutes after which a job starts no new trace call")
    ap.add_argument("--runs", type=int, default=40, help="whole runs in one whole-runs job")
    ap.add_argument("--run-batch", type=int, default=10, help="whole runs per runs.py call (a safe point between calls)")
    ap.add_argument("--run-procs", type=int, default=4, help="game sessions per runs.py call")
    ap.add_argument("--run-stop-act", type=int, default=3, help="a whole run stops after this act (3: the whole run; 1 for a short trial)")
    ap.add_argument("--procs", type=int, default=8, help="processes for the corpus check")
    ap.add_argument("--mutants", type=int, default=150)
    ap.add_argument("--presort-gap-min", type=float, default=60.0, metavar="MINUTES",
                    help="a presort for a main that moved waits until the last one ended this long ago (the other jobs keep running on "
                         "the presorted tree meanwhile); 0: presort on every move; --once presort ignores it")
    ap.add_argument("--idle-seconds", type=float, default=60.0, help="user input within this many seconds stops the lab")
    ap.add_argument("--load-pct", type=float, default=50.0, help="other processes' CPU share that pauses the lab ...")
    ap.add_argument("--load-window", type=float, default=120.0, help="... when sustained over this many seconds")
    ap.add_argument("--lab-gb", type=float, default=30.0, help="keep LAB under this size")
    ap.add_argument("--free-gb", type=float, default=50.0, help="keep this much of the drive free")
    ap.add_argument("--grace", type=float, default=90.0, help="seconds a running step gets to reach its end on PAUSE / main.lock / trial end")
    ap.add_argument("--poll", type=float, default=2.0, help="seconds between guard checks while a command runs")
    ap.add_argument("--wait-poll", type=float, default=10.0, help="seconds between guard checks while paused")
    ap.add_argument("--idle-poll", type=float, default=300.0, help="seconds between looks for new work when there is none")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- lab folder, state, job records

DEFAULT_STATE: dict[str, Any] = {
    "version": 1, "last_presorted_sha": None, "tree_sha": None, "last_presort_end": None, "rotation": 0, "next_start": 1, "next_run_start": 1000, "bad_shas": {},
    "backoff": {}, "strategy_cpu": {},  # strategy -> CPU seconds its finished jobs spent (the mix schedules on it)
    "last_ok": {},  # strategy -> epoch seconds its last job finished with rc 0 (mix.json "_every_h"; yield.tsv rows come later, after the report)
    "last_ok_sha": {},  # mutation / invariants / probes / retest -> the tree (sha) its last rc 0 job ran on: each runs once per tree
    "retest_seen": None,  # the group ids marked for a retest when the last retest job finished (None: not known yet)
    "report_pending": [],  # [{name, rc}] finished jobs whose yield row waits for the report beside the next job: a restart resumes them
    "counters": {"jobs_started": {}, "jobs_done": {}, "fights": 0, "runs": 0, "interrupts": {}, "active_seconds": 0},
}


class Lab:
    def __init__(self, root: Path):
        self.root = root
        self.runs = root / "runs"
        self.state_path = root / "state.json"
        self.status_path = root / "status.txt"
        self.log_path = root / "runner.log"

    def load_state(self) -> dict[str, Any]:
        st = json.loads(json.dumps(DEFAULT_STATE))
        got = read_json(self.state_path, {})
        for k, v in got.items():
            if isinstance(v, dict) and isinstance(st.get(k), dict):
                st[k].update(v)
            else:
                st[k] = v
        return st

    def save_state(self, st: dict[str, Any]) -> None:
        atomic_write(self.state_path, json.dumps(st, indent=1, sort_keys=True))

    def runs_without_done(self) -> list[Path]:
        if not self.runs.is_dir():
            return []
        return sorted(d for d in self.runs.iterdir() if d.is_dir() and (d / "job.json").exists() and not (d / "done.rc").exists())


class Run:
    """One job folder: job.json is the record, done.rc the last thing written."""

    def __init__(self, path: Path, data: dict[str, Any]):
        self.dir, self.data = path, data

    @classmethod
    def create(cls, lab: Lab, job: str, sha: str, params: dict[str, Any]) -> Run:
        now = dt.datetime.now(dt.UTC)
        name, n = f"{now:%Y%m%dT%H%M}-{job}", 1
        while (lab.runs / name).exists():  # two jobs in one minute: the seconds (and a counter) tell them apart
            n += 1
            name = f"{now:%Y%m%dT%H%M%S}{'' if n == 2 else n}-{job}"
        d = lab.runs / name
        (d / "logs").mkdir(parents=True)
        r = cls(d, {"version": 1, "job": job, "name": name, "main_sha": sha, "created_utc": utc(), "resumed": 0,
                    "params": params, "steps": [], "outputs": {}})
        r.save()
        return r

    @classmethod
    def load(cls, d: Path) -> Run | None:
        data = read_json(d / "job.json", None)
        return cls(d, data) if isinstance(data, dict) and data.get("job") in JOB_TYPES else None

    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def job(self) -> str:
        return self.data["job"]

    def rec(self, step: str) -> dict[str, Any]:
        for r in self.data["steps"]:
            if r["name"] == step:
                return r
        r = {"name": step, "status": "pending", "cmds": {}}
        self.data["steps"].append(r)
        return r

    def save(self) -> None:
        atomic_write(self.dir / "job.json", json.dumps(self.data, indent=1))

    def finish(self, rc: int, note: str = "") -> None:
        self.data["finished_utc"], self.data["rc"] = utc(), rc
        if note:
            self.data["note"] = note
        self.save()
        atomic_write(self.dir / "done.rc", f"{rc}\n")  # last


class Interrupted(Exception):
    def __init__(self, verdict: guards.Verdict):
        super().__init__(str(verdict))
        self.verdict = verdict


class Halted(Exception):
    pass


class Running:
    def __init__(self, cmd: plans.Cmd, proc: subprocess.Popen, files: list[Any], t0: float, in_job: bool,
                 acct: winsys.JobObject | None = None, stamp0: tuple | None = None, cap: winsys.JobObject | None = None):
        self.cmd, self.proc, self.files, self.t0, self.in_job = cmd, proc, files, t0, in_job
        self.acct = acct  # a Job Object of this command alone, inside the step's: its CPU and I/O, children included (clock accounting)
        self.cap = cap  # the Job Object that holds the command to its memory share (Cmd.mem_share), None for a command with none
        self.stamp0 = stamp0  # the command's skip stamp as it was at the start


def rep_cpu(rep: Report | None) -> float:
    return rep.sj.cpu_time() / 1e7 if rep is not None else 0.0


class Report:
    """The groups and kits of finished jobs, running beside the next job: one chain at a time, groups then kits, in a Job Object of
    their own. It covers every job that finished since the last chain started; each of them gets its yield row when the chain ends."""

    def __init__(self, run: Run, ctx: plans.Ctx, cmds: list[plans.Cmd], covers: list[tuple[Run, int]], sj: winsys.JobObject):
        self.run, self.ctx, self.cmds, self.covers, self.sj = run, ctx, cmds, covers, sj
        self.running: Running | None = None
        self.failed = False
        self.ok: dict[str, bool] = {}
        self.t0 = time.monotonic()


class Filler:
    """One practice block (the selfplay kind) running beside the job: its own job folder, one command, a Job Object of its own that
    forces BelowNormal on every process in it (D48)."""

    def __init__(self, run: Run, ctx: plans.Ctx, step: plans.Step, sj: winsys.JobObject, running: Running, cores_: int, t0: float,
                 sha: str = ""):
        self.run, self.ctx, self.step, self.sj, self.running, self.cores, self.t0 = run, ctx, step, sj, running, cores_, t0
        self.sha = sha  # the presorted build it runs (its frozen copy); "" when it runs from the lab tree


# ---------------------------------------------------------------- the runner

LAB_SOURCES = Path(__file__).resolve().parent


def code_digest() -> dict[str, int]:
    """crc32 of each tools/lab/*.py now (what a long-lived runner process loaded at its start, or what is on disk now)."""
    out: dict[str, int] = {}
    for f in sorted(LAB_SOURCES.glob("*.py")):
        try:
            out[f.name] = zlib.crc32(f.read_bytes())
        except OSError:
            pass
    return out


def changed_sources(before: dict[str, int]) -> str:
    """Names of the lab sources that differ from `before` (changed, added or gone), comma-joined; "" when none."""
    now = code_digest()
    return ", ".join(sorted(k for k in set(before) | set(now) if before.get(k) != now.get(k)))


class Runner:
    def __init__(self, a: argparse.Namespace, clock=time.monotonic):
        self.a, self.clock = a, clock
        self.lab = Lab(Path(a.lab))
        self.fork, self.tree, self.game = Path(a.fork), Path(a.tree), Path(a.game)
        self.skip = tuple(x for x in a.skip.split(",") if x)
        self.t_start = clock()
        self.root_job: winsys.JobObject | None = None
        self.state: dict[str, Any] = self.lab.load_state()
        self.stop_signal: str | None = None
        self.current: tuple[str, str] | None = None  # (job name, step name)
        self.current_strategy: str | None = None
        self.since = utc()
        self._beat = 0.0
        disabled = {x for x in a.no_guard.split(",") if x}
        deadline = self.t_start + a.trial * 60 if a.trial is not None else None
        lock = Path(a.backlog_dir) / "main.lock"
        self.loop_cpu = guards.TreeCpu(self.loop_who)  # the always-on loop's processes: not "other" load (D48)
        self.load = guards.LoadMonitor(lambda: (self.root_job.cpu_time() if self.root_job else 0) + self.loop_cpu(), window=a.load_window,
                                       interval=min(5.0, max(0.2, a.poll)), clock=clock)
        self.guards = guards.Guards(self.lab.root, lock, disabled=disabled, idle_s=a.idle_seconds, load_pct=a.load_pct,
                                    load_window=a.load_window, lab_gb=a.lab_gb, free_gb=a.free_gb, grace=a.grace, deadline=deadline,
                                    clock=clock, load=self.load, prune=self.prune_some,
                                    parent_pid=os.getppid() if a.exit_with_parent else None, stay=a.always_on)
        self.next_strategy: str | None = None  # the strategy choose() picked for the job new_run() makes next
        self.defer_sha: str | None = None  # set by choose() when a presort is deferred: the presorted tree's sha the picked job runs on
        self.last_mix_note = ""
        self.last_loop_note = ""
        self.skip_logged: dict[str, tuple[str, float]] = {}  # name -> (reason, time) of the last "not runnable" line written
        self.code_digest0 = code_digest()  # the lab sources this process loaded; a later change on disk is not in this process
        self.code_stale_logged = ""
        self._mark = (0.0, 0.0)  # (cpu seconds, clock) at the last accounting of the running job
        self.report_beside = a.report_beside == "on" or (a.report_beside == "auto" and a.always_on and a.once is None
                                                          and a.trial is None and not a.dry_run)
        self.report: Report | None = None  # the chain running beside the job
        self.report_cpu_done = 0.0  # CPU seconds of the reports that ended or were stopped
        self.report_due: list[tuple[Run, int]] = []  # finished jobs whose groups and kits have not run yet
        self.report_cuts = 0  # reports a presort stopped since a report last ended (--report-cut-cap)
        self.running_job: str | None = None  # the type of the job execute() is running
        self.guards_ok = True  # the last guard reading allowed work (a main.lock does not stop the report)
        self.lock_stops = True  # main.lock stops what runs now: only a presort (set by execute); between jobs it holds every pick
        self._lock_episode = False  # a "main.lock held: ... continues" line was written for the lock that is up now
        self.filler_mode = a.filler == "on" or (a.filler == "auto" and a.always_on and a.once is None and a.trial is None
                                                 and not a.dry_run)
        self.filler: Filler | None = None  # the practice block running beside the job (D48)
        self.filler_cpu_done = 0.0  # CPU seconds of the filler blocks that ended or were stopped
        self.filler_next = 0.0  # clock time before which no filler block starts (after a stop by a guard, the night, a failure)
        self.filler_note = ""  # the last reason the filler did not start, logged once
        self._filler_tick = float("-inf")
        ft = str(a.filler_tree)
        self.filler_root: Path | None = (None if ft.lower() == "off" else
                                         self.lab.root.parent / "filler-tree" if ft.lower() == "auto" else Path(ft))  # frozen copies (D52)
        self.main_cores = 0  # cores of the job the main lane runs now (0 between jobs)
        self.report_note = ""  # the last reason a due report did not start, logged once
        self._config_note = ""  # the rotation and reports settings last logged (said again only when they change)
        self.train_lock: Any = None  # the QuietLock holding our `lab-train` line while a train job runs (rl/quiet_lock.py)
        self._train_beat = 0.0
        self._load_exempt_logged = False
        self._alert_sig: tuple | None = None
        self._main_sha_seen: tuple[float, str | None, tuple] | None = None

    # ---- plumbing

    def log(self, msg: str) -> None:
        line = f"{utc()} {msg}"
        try:
            print(line, flush=True)
        except (OSError, ValueError, AttributeError):
            pass
        try:
            if not self.lab.root.is_dir():  # a refused or dry run makes no folder
                return
            if self.lab.log_path.exists() and self.lab.log_path.stat().st_size > 5 * 1024 * 1024:
                self.lab.log_path.replace(self.lab.log_path.with_suffix(".log.1"))
            with open(self.lab.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def status(self, state: str, extra: str = "") -> None:
        c = self.state["counters"]
        intr = ",".join(f"{k}:{v}" for k, v in sorted(c["interrupts"].items())) or "-"
        where = f"job={self.current[0]} step={self.current[1]}" if self.current else "job=-"
        main = (self.state.get("tree_sha") or "-")[:7]
        rd = self.guards.readings
        if rd.get("lock_ignored"):
            extra = (extra + " " if extra else "") + "main.lock=held(job continues)"
        if rd.get("load") is not None:
            extra = (extra + " " if extra else "") + f"other_cpu={rd['load'][0]}%/{rd['load'][1]}s"
        if self.current and self.current_strategy:
            where += f" strategy={self.current_strategy}"
        if self.report:
            where += f" report={self.report.run.name}"
        if self.filler:
            where += f" filler={self.filler.run.name}@{self.filler.cores}"
        line = (f"{utc()} {state} {where} since={self.since} tree={main} jobs_done={sum(c['jobs_done'].values())} "
                f"fights={c['fights']} runs={c.get('runs', 0)} interrupts={intr}" + (f" | {extra}" if extra else ""))
        try:
            top = alerts.status_prefix(self.lab.root)  # an alert stands: the first words of the line, for whoever reads only that
        except Exception:  # noqa: BLE001
            top = ""
        line = f"{top} | {line}" if top else line
        try:
            atomic_write(self.lab.status_path, line + "\n")
        except OSError:
            pass
        self._beat = self.clock()

    def heartbeat(self, state: str, extra: str = "") -> None:
        if self.clock() - self._beat >= 20:
            self.status(state, extra)

    def prune_some(self, need: int) -> int:
        busy = {self.current[0]} if self.current else set()
        if self.report:  # the report reads its covered jobs' recordings
            busy |= {r.name for r, _ in self.report.covers}
        if self.filler:
            busy.add(self.filler.run.name)
        freed = prune.prune(self.lab.root, need, busy)
        if freed:
            self.log(f"disk: pruned {freed / guards.GB:.2f} GB of raw outputs")
        return freed

    def main_sha(self) -> str | None:
        r = subprocess.run(["git", "-C", str(self.fork), "rev-parse", "--verify", f"refs/heads/{self.a.branch}"],
                           capture_output=True, text=True, check=False, creationflags=winsys.NO_WINDOW)
        return r.stdout.strip() if r.returncode == 0 else None

    def newest_backlog(self) -> Path | None:
        if self.a.backlog:
            return Path(self.a.backlog)
        files = sorted(Path(self.a.backlog_dir).glob("backlog-*.json"), key=lambda p: p.stat().st_mtime)
        return files[-1] if files else None

    def ctx(self, run: Run | None, job: str, sha: str, params: dict[str, Any], tree: Path | None = None) -> plans.Ctx:
        a = self.a
        name = run.name if run else f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M}-{job}"
        return plans.Ctx(lab=self.lab.root, tree=tree or self.tree, game=self.game, fork=self.fork, py=sys.executable.replace("\\", "/"),
                         sha=sha, job_dir=run.dir if run else self.lab.runs / name, job_name=name, params=params,
                         backlog_dir=Path(a.backlog_dir), backlog=self.newest_backlog(), codex=Path(a.codex),
                         decompiled=Path(a.decompiled), records=Path(a.records), game_src=Path(a.game_src),
                         nuget=Path(os.path.expanduser("~")) / ".nuget" / "packages", skip=self.skip, report_beside=self.report_beside,
                         off=self.reports_off())

    def poll(self) -> guards.Verdict:
        if self.stop_signal:
            return guards.Verdict("signal", self.stop_signal, leave=True)
        exempt = self.train_lock is not None  # a train job that holds the lab-train line is the one the others yield to: load does not pause it
        v = self.guards.poll(self.lock_stops, load_exempt=exempt)
        self.guards_ok = (v.ok or v.kind == "lock") and not self.guards.load_hit  # the filler and the report still give way to load
        if exempt and self.guards.load_hit:
            if not self._load_exempt_logged:
                self._load_exempt_logged = True
                rd = self.guards.readings.get("load") or ("?", "?")
                self.log(f"load guard: other processes {rd[0]}% of the CPU over {rd[1]}s; the train job holds the {TRAIN_TAG} line and goes on "
                         "(the filler and the report give way)")
        else:
            self._load_exempt_logged = False
        if self.report and not self.guards_ok:  # the report gives the machine back at once (it starts over when the lab is free)
            self.kill_report(str(v))
        if self.filler and not self.guards_ok:  # so does the filler (and waits FILLER_RETRY_S before the next block)
            self.stop_filler(str(v))
        if self.guards.readings.get("lock_ignored"):
            if not self._lock_episode:  # once per lock episode, at the first poll that sees it
                self._lock_episode = True
                job = self.current[0] if self.current else "-"
                self.log(f"main.lock held: {job} continues (only a presort stops for it)")
                self.status("running", self.current[1] if self.current else "")
        else:
            self._lock_episode = False
        return v

    # ---- choosing the next job

    def default_params(self, job: str) -> dict[str, Any]:
        a, st = self.a, self.state
        if job == "presort":
            return {"procs_refset": 4, "procs_corpus": a.procs}
        if job == "fights":
            n = max(1, math.ceil(a.fights / max(1, a.batch)))
            return {"batches": n, "batch": a.batch, "start": st["next_start"], "servers": a.servers, "shrink": a.shrink,
                    "shrink_cap": a.shrink_cap, "shrink_budget": a.shrink_budget, "shrink_minutes": a.shrink_minutes, "trace": a.trace,
                    "trace_minutes": a.trace_minutes, "strategy": self.next_strategy}
        if job == "runs":
            return {"batches": max(1, math.ceil(a.runs / max(1, a.run_batch))), "run_batch": a.run_batch, "start": st["next_run_start"],
                    "procs": a.run_procs, "stop_after_act": a.run_stop_act, "strategy": "whole-runs"}
        if job in jobs.NEW_KINDS:
            return jobs.params_for(job, self.lab.root, jobs.load_config(self.lab.root)[job])
        if job == "probes":
            return {"probe_ids": plans.probe_ids(Path(a.backlog_dir))}
        if job == "mutation":
            return {"mutation_source": self.mutation_source(), "mutants": a.mutants}
        return {"seeds": 4 + 4 * (st["counters"]["jobs_done"].get("invariants", 0) % 4), "procs": 4}

    def mutation_source(self) -> str | None:
        """The biggest batch of the newest finished fights job that still has its raw files (path under LAB, no suffix)."""
        for d in sorted(self.lab.runs.glob("*-fights"), reverse=True) if self.lab.runs.is_dir() else []:
            if not (d / "done.rc").exists():
                continue
            best = None
            for v in (d / "fights").glob("*.verdicts.json"):
                base = v.name[: -len(".verdicts.json")]
                files = [d / "fights" / (base + s) for s in (".out.json", ".emu.json", ".reanchor.json")]
                if all(f.exists() for f in files) and not (d / "pruned.json").exists():
                    size = files[0].stat().st_size
                    if best is None or size > best[0]:
                        best = (size, f"runs/{d.name}/fights/{base}")
            if best:
                return best[1]
        return None

    def choose(self, sha: str | None) -> tuple[str, Run | None, str]:
        """(job type, a run to resume or None, why). job '' when there is nothing to run."""
        st, a = self.state, self.a
        self.defer_sha = None
        tree_at = self.deferred_tree(sha) if sha else None  # the presorted tree the jobs run on while main's presort waits
        here = {x for x in (sha, tree_at) if x}  # the shas a job may resume on
        cur = self.lab.runs_without_done()
        for d in reversed(cur):
            run = Run.load(d)
            if run is None or (self.filler is not None and d == self.filler.run.dir):
                continue
            if (run.data.get("params") or {}).get("filler"):  # a practice block left by a runner that stopped: its runs are kept
                if not a.dry_run:
                    self.record(run, 3)
                    run.finish(3, "filler block: ended with the runner that started it")
                continue
            if a.once and run.job != a.once:
                continue
            if self.rotation_hold(run.job) is not None:  # the kind was switched off since the job stopped: it is not resumed
                if not a.dry_run:
                    self.record(run, 3)
                    run.finish(3, f"closed: {run.job} is not in LAB/jobs.json rotation")
                continue
            age_h = (time.time() - (d / "job.json").stat().st_mtime) / 3600
            if sha is None or run.data.get("main_sha") not in here or age_h > RESUME_HOURS:
                if not a.dry_run:
                    self.record(run, 3)
                    run.finish(3, "superseded: main moved or the job was idle too long")
                    self.release_nightly_consent(run.job)
                continue
            return run.job, run, f"resume {run.name}"
        if sha is None:
            return "", None, "cannot read the fork's local main"
        bad = st["bad_shas"].get(sha)
        if bad and time.time() < bad.get("retry_after", 0):
            return "", None, f"build of {sha[:7]} failed {bad['fails']}x; retry after {utc(bad['retry_after'])}"
        if a.once == "presort":
            return "presort", None, "--once presort"
        if st["last_presorted_sha"] != sha or st["tree_sha"] != sha:
            if a.once is not None:
                return "", None, f"--once {a.once} needs the tree presorted at {sha[:7]} first (--once presort)"
            if tree_at is None:
                return "presort", None, f"main moved to {sha[:7]}"
            job, why = self.pick_other(tree_at)  # main moved, but the last presort is recent: its tree still has work to do
            if not job:
                return "presort", None, f"main moved to {sha[:7]}; no other job can run on the tree at {tree_at[:7]}"
            self.defer_sha = tree_at
            return job, None, f"{why} on the tree at {tree_at[:7]}; presort of {sha[:7]} deferred (gap {a.presort_gap_min:g} min)"
        if a.once:
            if a.once == "fights":
                picked, strategy = self.pick_fights(sha)
                if not picked:
                    return "", None, "no fights strategy is runnable"
                self.next_strategy = strategy if picked == "fights" else None
                return picked, None, f"--once, {strategy}"  # the mix may pick whole-runs, its own job type
            why = self.unavailable(a.once, sha)
            if why is not None:
                self.log(f"{a.once}: not runnable ({why})")
            return (a.once, None, "--once") if why is None else ("", None, why)
        job, why = self.pick_other(sha)
        return job, None, why

    def pick_other(self, sha: str) -> tuple[str, str]:
        """(job type, why) of the job after presort: the ladder's, else the fights rotation's, else the self-play filler.
        sha is the tree the job runs on (a presorted tree behind main while a presort is deferred). ("", why) when none can run."""
        st = self.state
        job = self.ladder_top(sha)
        if job:
            self.next_strategy = None
            return job, f"ladder: {job} (priority {jobs.PRIORITY[job]})"
        fights_none = False  # the mix answers the same for every fights slot of one pass: ask once
        allowed = self.rotation_set() if not self.a.once else jobs.FIDELITY_KINDS
        for k in range(len(ROTATION)):
            job = ROTATION[(st["rotation"] + k) % len(ROTATION)]
            if job not in allowed and not (job == "fights" and "runs" in allowed):  # off in LAB/jobs.json rotation: no pick, no log line
                continue
            strategy = None
            if job == "fights":  # the mix says which strategy this slot's job is (a whole run is its own job type)
                if fights_none:
                    continue
                job, strategy = self.pick_fights(sha)
                if not job:
                    fights_none = True
                    continue
                why = None
            else:
                why = self.unavailable(job, sha)
            if why is None:
                st["rotation"] = (st["rotation"] + k + 1) % len(ROTATION)
                self.next_strategy = strategy if job == "fights" else None
                return job, "rotation" + (f", {strategy}" if strategy else "")
            self.note_skip(job, why)
        if not self.filler_owns_selfplay() and self.kind_ready("selfplay", sha):  # the filler: nothing above it could run
            self.next_strategy = None
            return "selfplay", "ladder: selfplay (filler, nothing else ready)"
        return "", "no job is runnable"

    def last_presort_end(self) -> float:
        """Epoch seconds the last presort ended: the state's note, else the newest finished presort folder, else 0 (long ago)."""
        t = self.state.get("last_presort_end")
        if t:
            return float(t)
        ends = [(d / "done.rc").stat().st_mtime for d in (self.lab.runs.glob("*-presort") if self.lab.runs.is_dir() else [])
                if (d / "done.rc").exists()]
        return max(ends, default=0.0)

    def deferred_tree(self, sha: str) -> str | None:
        """The sha of the presorted tree the other jobs may run on, while main's presort waits for the gap since the last one to pass
        (--presort-gap-min): None when main has not moved, the tree is not a good presorted one, the gap has passed, or a
        --once run asks (--once presort ignores the gap; any other --once keeps needing the tree at main)."""
        st, a = self.state, self.a
        tree = st["tree_sha"]
        if a.once is not None or a.presort_gap_min <= 0 or not tree or st["last_presorted_sha"] != tree or tree == sha:
            return None
        return tree if time.time() - self.last_presort_end() < a.presort_gap_min * 60 else None

    def kind_ready(self, kind: str, sha: str) -> bool:
        why = self.unavailable(kind, sha)
        if why is not None:
            self.log(f"{kind}: not runnable ({why})")
        return why is None

    last_nightly_note = ""

    def ladder_top(self, sha: str) -> str:
        """The agent-era kinds that outrank the fights rotation (docs/PLAN.md section 7), highest priority first: nightly-real (1.5: a
        promotion or the weekly clock, the real-game check comes first), score-real (2), train (3), selfplay when the loop
        (selfplay + train + score-real) has not yet used its configured share of the last `share_days` days' CPU (6, `share` and
        `share_days` in LAB/jobs.json, default 7 days; a short window keeps the loop running daily instead of in bursts; an empty window means self-play first; all-time `strategy_cpu` is a statistic only), the
        benches (7, nightly, the lab otherwise idle: a running job was resumed before this point and a ready score-real or
        train came first). Presort (1) is decided before, aimed fights (4), runs and rank (5) and the rest (7) by the rotation and the
        mix. A kind that is off or not ready is passed over quietly (its reason is only logged when it was due)."""
        cfg = jobs.load_config(self.lab.root)
        for kind in ("nightly-real", "score-real", "train"):  # jobs.PRIORITY order
            why = jobs.unready(kind, self.lab.root, cfg[kind])
            if kind == "nightly-real" and why and cfg[kind].get("enabled") and jobs.nightly_wanted(self.lab.root, float(cfg[kind].get("every_h", 20.0))):
                if why != self.last_nightly_note:  # due but held: say why once, not at every pick
                    self.last_nightly_note = why
                    self.log(f"nightly-real: skipped, {why}")
            if why is None and self.kind_ready(kind, sha):
                return kind
        share = float(cfg["selfplay"].get("share") or 0)
        if share > 0 and cfg["selfplay"].get("enabled") and not self.filler_owns_selfplay(cfg):
            days = float(cfg["selfplay"].get("share_days") or mix.WINDOW_DAYS)  # the span the share is judged over, not all time
            loop, total, by = mix.loop_cpu(self.lab.root, days=days)
            note = (f"self-play loop: {100 * loop / total:.1f}% of the last {days:g} days' CPU "
                    f"({', '.join(f'{k} {by.get(k, 0.0):.0f}' for k in jobs.LOOP_KINDS)} of {total:.0f} min) against its share "
                    f"{100 * share:.0f}%" if total > 0 else
                    f"self-play loop: no CPU in the last {days:g} days in LAB/yield.tsv: self-play first")
            if note != self.last_loop_note:
                self.last_loop_note = note
                self.log(note)
            if total <= 0.0 or loop < share * total:
                if self.kind_ready("selfplay", sha):
                    return "selfplay"
        for kind in jobs.BENCHES:
            if jobs.unready(kind, self.lab.root, cfg[kind]) is None and self.kind_ready(kind, sha):
                return kind
        return ""

    def pick_fights(self, sha: str) -> tuple[str, str | None]:
        """(job type, strategy) for a fights slot: the available strategy furthest below its share of CPU time in LAB/mix.json;
        ("", None) when none can run."""
        weights, every_h, notes = mix.load_mix_cadence(self.lab.root, create=not self.a.dry_run)
        note = "; ".join(n for n in notes if not mix.is_idle_note(n))
        if note != self.last_mix_note:
            self.last_mix_note = note
            if note:
                self.log(note)
        for n in notes:  # weight the running code cannot spend stays visible: said at every pick, not only when it changes
            if mix.is_idle_note(n):
                self.log(n)
        stale = changed_sources(self.code_digest0)
        if stale and stale != self.code_stale_logged:
            self.code_stale_logged = stale
            self.log(f"runner code on disk changed since this runner started ({stale}); this process still runs the old code: restart it")
        spent = self.state["strategy_cpu"]
        if self.a.once:
            every_h = {}  # a one-shot run is asked for by hand: the cadence is the rotation's
        chosen, skipped = mix.pick(weights, spent, lambda s: self.unavailable("runs" if s == "whole-runs" else "fights", sha, s),
                                   every_h, self.last_ok() if every_h else None)
        for s, why in skipped.items():
            self.note_skip(s, why)
        if chosen is None:
            return "", None
        self.log(f"mix: {chosen} is next (share wanted/had: {mix.share_table(weights, spent)})")
        return ("runs" if chosen == "whole-runs" else "fights"), chosen

    def last_ok(self) -> dict[str, float]:
        """Strategy -> epoch seconds its last job finished with rc 0, for mix.json's "_every_h": the rows of LAB/yield.tsv, and the
        times `record` kept in the state, which cover a job whose row waits for the report beside the next job (or was lost with a
        restart): without them the pick right after a job would find no row and run the strategy again."""
        got = mix.last_ok(self.lab.root)
        for s, t in (self.state.get("last_ok") or {}).items():
            if isinstance(t, (int, float)) and not isinstance(t, bool):
                got[s] = max(got.get(s, 0.0), float(t))
        return got

    def note_skip(self, name: str, why: str) -> None:
        """Write "<name>: not runnable (<why>)" unless that very line was written for this name within the last hour: a lab with nothing due
        looks again every idle poll, and the same reasons said every five minutes bury the lines that change."""
        now = time.time()
        last = self.skip_logged.get(name)
        if last is not None and last[0] == why and now - last[1] < 3600:
            return
        self.skip_logged[name] = (why, now)
        self.log(f"{name}: not runnable ({why})")

    @staticmethod
    def tree_once_key(job: str, strategy: str | None) -> str | None:
        """The last_ok_sha key of a job that runs once per presorted tree: mutation, invariants and probes, and the retest strategy."""
        if job in TREE_ONCE_JOBS:
            return job
        return "retest" if job == "fights" and strategy == "retest" else None

    def ran_on_tree(self, key: str, sha: str) -> str | None:
        """"ran on this tree already (<sha>)" when the last rc 0 job of this kind ran on the tree the pick is for. A retest only means
        something after the tree changed, except for a group marked for one since the last retest job finished: that one is due."""
        if not sha or (self.state.get("last_ok_sha") or {}).get(key) != sha:
            return None
        if key == "retest" and self.state.get("retest_seen") is not None:
            if set(retest.marked(self.groups_list())) - set(self.state["retest_seen"]):
                return None
        return f"{RAN_ON_TREE} ({sha[:7]})"

    def groups_list(self) -> list:
        doc = read_json(self.lab.root / "groups.json", None)
        return doc.get("groups") if isinstance(doc, dict) and isinstance(doc.get("groups"), list) else []

    def seed_last_ok_sha(self) -> None:
        """A state file from before last_ok_sha: take, for each once-per-tree kind it lacks, the tree of the newest finished (rc 0) job
        of that kind among the newest job folders, so the first start does not run them again on a tree they already ran on."""
        have = self.state.setdefault("last_ok_sha", {})
        want = {*TREE_ONCE_JOBS, "retest"} - set(have)
        if not want or not self.lab.runs.is_dir():
            return
        try:
            names = sorted((d.name for d in self.lab.runs.iterdir() if d.is_dir()), reverse=True)[:SEED_LOOK_BACK]
        except OSError:
            return
        for name in names:
            d = self.lab.runs / name
            if read_rc(d / "done.rc") != 0:
                continue
            run = Run.load(d)
            if run is None:
                continue
            key = self.tree_once_key(run.job, (run.data.get("params") or {}).get("strategy"))
            if key in want and run.data.get("main_sha"):
                have[key] = run.data["main_sha"]
                want.discard(key)
                if not want:
                    break

    # ---- which fidelity-era kinds and report steps run (LAB/jobs.json `rotation` and `reports`, D57 addendum 1)

    def rotation_set(self) -> tuple[str, ...]:
        """The fidelity-era kinds the runner may pick now: every one with --rotation all, else LAB/jobs.json `rotation` (absent or empty: none).
        The settings are logged when they change (and a name that is not a kind is said once)."""
        if self.a.rotation == "all":
            return jobs.FIDELITY_KINDS
        kinds, unknown = jobs.rotation_kinds(cores.load_doc(self.lab.root))
        self.note_config(kinds, unknown)
        return kinds

    def reports_config(self) -> dict[str, dict[str, Any]]:
        if self.a.reports == "all":
            return {s: {"enabled": True, "every_h": None} for s in jobs.REPORT_STEPS}
        return jobs.reports_config(cores.load_doc(self.lab.root))

    def note_config(self, kinds: tuple[str, ...], unknown: list[str] | None = None) -> None:
        """Say once, and again when it changes, which kinds the rotation may pick and which report steps run."""
        rep = self.reports_config() if self.a.reports != "all" else None
        on = [s for s, v in (rep or {}).items() if v["enabled"]]
        note = f"rotation: {', '.join(kinds) or 'none'}; reports: {', '.join(on) or 'none'}" + (f"; not kinds: {', '.join(unknown)}" if unknown else "")
        if note != self._config_note and not self.a.dry_run:
            first = not self._config_note
            self._config_note = note
            self.log(f"LAB/jobs.json {note}" + (" (a kind or step that is not named is off)" if first else ""))

    def rotation_hold(self, job: str) -> str | None:
        """Why the fidelity-era kind `job` is not picked: LAB/jobs.json `rotation` does not name it. Kinds of other families, and a one-shot
        run (--once) that names its own kind, are never held."""
        if job not in jobs.FIDELITY_KINDS or self.a.once:
            return None
        return None if job in self.rotation_set() else "not in LAB/jobs.json rotation (the fidelity-era kinds are off unless it names them)"

    def reports_off(self) -> tuple[str, ...]:
        """The report steps that do not run now: those LAB/jobs.json `reports` does not enable, and `checks` while its `every_h` has not passed
        since the last presort that ran it (`state["last_checks"]`)."""
        cfg = self.reports_config()
        off = [s for s in jobs.REPORT_STEPS if not cfg[s]["enabled"]]
        every = cfg["checks"]["every_h"]
        last = self.state.get("last_checks")
        if "checks" not in off and every and isinstance(last, (int, float)) and time.time() - last < every * 3600:
            off.append("checks")
        return tuple(off)

    def alert_hold(self, key: str) -> str | None:
        """Why `key` (a job type, or the strategy of a fights job) is not picked: it failed twice in a row and LAB/alerts/<key>.txt stands. The
        hold ends when the file is removed by hand (the failure count starts again); the runner logs the lifting once."""
        st = self.state
        held = st.setdefault("alerted", {})
        path = alerts.alert_path(self.lab.root, key)
        if path.exists():
            if key not in held:  # a file the state does not know (a state file lost, a file written by hand): it holds all the same
                held[key] = utc()
            return f"stopped after repeated failures (LAB/alerts/{path.name}); remove the file to let it run again"
        if key in held:
            held.pop(key, None)
            st["backoff"].pop(key, None)
            st.setdefault("fail_streak", {})[key] = 0
            (st.get("fail_recent") or {}).pop(key, None)
            self.lab.save_state(st)
            self.refresh_banners()
            self.log(f"alert {path.name} removed: {key} is picked again")
        return None

    def unavailable(self, job: str, sha: str, strategy: str | None = None, tree: Path | None = None) -> str | None:
        why = self.alert_hold(strategy or job)
        if why is not None:
            return why
        why = self.rotation_hold(job)
        if why is not None:
            return why
        until = self.state["backoff"].get(strategy or job, 0)  # a strategy that keeps failing backs off alone
        if time.time() < until:
            return f"backing off after repeated failures until {utc(until)}"
        key = None if self.a.once else self.tree_once_key(job, strategy)  # a one-shot run is asked for by hand: the cadence is the rotation's
        if key:
            why = self.ran_on_tree(key, sha)
            if why is not None:
                return why
        if job in jobs.NEW_KINDS:  # the input and the switch first (plans never refuse a job that is resumed)
            why = jobs.unready(job, self.lab.root, jobs.load_config(self.lab.root)[job])
            if why is not None:
                return why
        elif job in ("fights", "runs") and jobs.score_active(self.lab.root):
            return "a score-real window is open (LAB/score/active)"
        try:
            params = self.default_params(job)
            if strategy:
                params["strategy"] = strategy
            c = self.ctx(None, job, sha, params, tree)
            return plans.runnable(plans.plan(job, c), c)
        except plans.Unavailable as e:
            return str(e)

    # ---- the filler: practice beside the job (D48, tools/lab/cores.py)

    def loop_who(self) -> tuple[int | None, float | None]:
        """(pid, latest creation time) of the always-on loop's daemon from its status file, for the load guard's TreeCpu."""
        p = Path(self.a.loop_status)
        pid, _ = cores.loop_state(p)
        try:
            return pid, p.stat().st_mtime if pid else None
        except OSError:
            return None, None

    def filler_owns_selfplay(self, cfg: dict | None = None) -> bool:
        """The selfplay kind runs as the filler, so the ladder and the rotation never pick it."""
        if not self.filler_mode:
            return False
        sp = (cfg or jobs.load_config(self.lab.root))["selfplay"]
        return bool(sp.get("enabled") and sp.get("filler"))

    def busy_cores(self, doc: dict) -> int:
        """Cores the filler may not use: the job's, the report's beside it, and the always-on loop's while it steps."""
        busy = self.main_cores + (cores.report_cores(doc) if self.report else 0)
        pid, stepping = cores.loop_state(Path(self.a.loop_status), winsys.parent_alive)
        return busy + (cores.loop_cores(doc) if pid and stepping else 0)

    def filler_room(self, doc: dict) -> int:
        return cores.filler_size(cores.total_cores(doc), self.busy_cores(doc), cores.floor(doc))

    def filler_hold(self, doc: dict, starting: bool) -> str | None:
        """Why the filler may not start (`starting`) or must stop now; None when it may run."""
        if not self.filler_owns_selfplay(jobs.config_from(doc)):
            return "selfplay is not the filler (LAB/jobs.json selfplay.enabled and selfplay.filler)"
        if not self.guards_ok:
            return "a guard holds the lab"
        if self.running_job == "presort" and self.filler_root is None:
            return "a presort moves the tree"
        if self.running_job == "nightly-real":
            return "the night has the machine"
        why = cores.night_hold(self.lab.root, doc, starting=starting)
        if why:
            return why
        tag = cores.quiet_blocker(self.quiet_lock_path(), doc)
        if tag:
            return f"QUIET.lock is held by {tag}"
        if not starting:
            return None
        st = self.state
        if not st.get("tree_sha") or st.get("tree_sha") != st.get("last_presorted_sha"):
            return "the tree is not a presorted build"
        if self.clock() < self.filler_next:
            return "waiting after the last stop"
        why = self.presort_soon()
        if why:
            return why
        disk = self.guards.readings.get("disk")
        if isinstance(disk, tuple) and (disk[0] > 0.9 * self.a.lab_gb or disk[1] < 1.1 * self.a.free_gb):
            return f"the lab is near its disk limit (lab {disk[0]} GB of {self.a.lab_gb:g}, drive free {disk[1]} GB)"
        tree, why = self.filler_tree_for(st["tree_sha"])
        if why:
            return why
        why = self.unavailable("selfplay", st["tree_sha"], tree=tree)
        if why:
            return why
        return None

    def cached_main_sha(self) -> str | None:
        """The fork's local main, looked up at most every MAIN_SHA_CACHE_S (the filler asks at every tick)."""
        now, st = self.clock(), self.state
        mark = (st.get("tree_sha"), st.get("last_presorted_sha"))  # a presort that ended since the look is a reason to look again
        if self._main_sha_seen is None or now - self._main_sha_seen[0] >= MAIN_SHA_CACHE_S or self._main_sha_seen[2] != mark:
            self._main_sha_seen = (now, self.main_sha(), mark)
        return self._main_sha_seen[1]

    def presort_soon(self, horizon_s: float = FILLER_PRESORT_LOOKAHEAD_S) -> str | None:
        """Why a practice block should not start: a presort would be chosen within `horizon_s`. That is when the fork's local main has moved
        past the presorted tree and the last presort ended `--presort-gap-min` ago or more, or will have by then (and a build that failed has
        waited out its retry). The wording is the same until the presort runs, so the log says it once."""
        a, st = self.a, self.state
        sha = self.cached_main_sha()
        if not sha or (st.get("last_presorted_sha") == sha and st.get("tree_sha") == sha):
            return None
        now = time.time()
        due = self.last_presort_end() + max(0.0, a.presort_gap_min) * 60
        bad = st["bad_shas"].get(sha)
        if bad:
            due = max(due, float(bad.get("retry_after", 0)))
        if due - now > horizon_s:
            return None
        return f"a presort of {sha[:7]} is due within {horizon_s / 60:g} min (main moved past the presorted tree)"

    def filler_tree_for(self, sha: str) -> tuple[Path | None, str]:
        """(the tree the filler runs from, why not): the lab tree itself in the old mode, else the frozen copy of the presorted build
        `sha`, made now when there is none and the tree stands at that build with no presort moving it (D52)."""
        if self.filler_root is None:
            return self.tree, ""
        got = frozen.ready(self.filler_root, sha)
        if got is not None:
            return got, ""
        if self.running_job == "presort" or self.unfinished_presort():
            return None, f"no frozen copy of {sha[:7]} yet and a presort is moving the tree"
        t0 = time.monotonic()
        try:
            got = frozen.make(self.tree, self.filler_root, sha, plans.git_head(self.tree))
        except frozen.FreezeError as e:
            return None, str(e)
        except OSError as e:
            self.filler_next = self.clock() + FILLER_RETRY_S  # not a copy attempt every tick
            return None, f"the lab tree could not be copied: {type(e).__name__}"
        files = sum(len(fn) for _, _, fn in os.walk(got))
        self.log(f"filler: froze the lab tree at {sha[:7]} ({files} files, {time.monotonic() - t0:.1f}s)")
        gone = frozen.clean(self.filler_root, [got] + ([self.filler.ctx.tree] if self.filler else []))
        if gone:
            self.log(f"filler: removed the frozen copies of {', '.join(gone)}")
        return got, ""

    def unfinished_presort(self) -> bool:
        """A presort job folder with no done.rc: the tree may be mid-checkout or mid-build (a restart left it, or it is running)."""
        return any(not (d / "done.rc").exists() for d in self.lab.runs.glob("*-presort") if (d / "job.json").exists())

    def filler_stale(self, f: Filler) -> str | None:
        """A running block's build is older than the presorted tree (a presort finished since it started): stop it, the next block plays
        the new build. Not while a presort runs (the state still names the old build until it ends)."""
        st = self.state
        sha = st.get("tree_sha")
        if self.filler_root is None or not f.sha or not sha or sha == f.sha or sha != st.get("last_presorted_sha"):
            return None
        return f"the presort of {sha[:7]} finished; this block plays {f.sha[:7]}"

    def service_filler(self, start: bool = True) -> None:
        """Called wherever the runner waits or loops: close a block that ended, stop one that must give way, start one when the
        rules allow (`start`; not between jobs, so a block never starts just before the pick that would stop it). A fault here
        turns the filler off for this process; it never fails a job."""
        if not self.filler_mode:
            return
        try:
            self._service_filler(start)
        except Halted:
            raise
        except Exception:  # noqa: BLE001
            self.log("filler: fault, off for this runner process: " + traceback.format_exc()[-500:])
            self.filler_mode = False
            self.stop_filler("fault")

    def _service_filler(self, start: bool) -> None:
        f = self.filler
        if f is not None:
            rc = f.running.proc.poll()
            now = self.clock()
            if rc is None and f.running.cmd.timeout and now - f.running.t0 > f.running.cmd.timeout:
                f.sj.terminate()
                rc = f.running.proc.wait(timeout=30)
                f.run.rec(f.step.name)["note"] = f"killed after the {f.running.cmd.timeout:.0f}s limit"
                self.note_timeout(f.run, f.running.cmd, now - f.running.t0)
            if rc is not None:
                self.end_filler(rc)
                f = None
        now = self.clock()
        if now - self._filler_tick < FILLER_TICK_S:
            return
        self._filler_tick = now
        doc = cores.load_doc(self.lab.root)
        if f is not None:
            why = self.filler_hold(doc, starting=False)
            if why:
                self.stop_filler(why)
                return
            why = self.filler_stale(f)
            if why:
                self.stop_filler(why, retry=False)
                return
            room = self.filler_room(doc)
            if room < f.cores:  # a job needs cores the block holds: stop it, the next tick starts one at what is left
                self.stop_filler(f"a job needs its cores ({f.cores} held, {room} left)", retry=False)
            return
        if not start:
            return
        why = self.filler_hold(doc, starting=True)
        size = self.filler_room(doc) if why is None else 0
        if why is None and size <= 0:
            why = f"under {cores.floor(doc)} cores free"
        if why:
            if why != self.filler_note:
                self.filler_note = why
                self.log(f"filler: not started ({why})")
            return
        self.filler_note = ""
        self.start_filler(doc, size)

    def start_filler(self, doc: dict, size: int) -> None:
        sha = self.state["tree_sha"]
        tree, why = self.filler_tree_for(sha)
        if why:
            self.log(f"filler: not started ({why})")
            return
        cfg = cores.filler_cfg(doc)
        params = jobs.params_for("selfplay", self.lab.root, cfg)
        explore = cfg.get("explore") if isinstance(cfg.get("explore"), list) else None
        params["cmd"] = cores.filler_cmd(params["cmd"], size, explore)
        params.update(filler=True, cores=size, procs=size)
        run = Run.create(self.lab, "selfplay", sha, params)
        n = self.state["counters"]["jobs_started"]
        n["selfplay"] = n.get("selfplay", 0) + 1
        self.lab.save_state(self.state)
        ctx = self.ctx(run, "selfplay", sha, params, tree)
        step = plans.plan("selfplay", ctx)[0]
        cmd = step.cmds[0]
        rec = run.rec(step.name)
        why = cmd.skip_reason(ctx, step.name)
        if why:
            rec["status"], rec["note"] = "skipped", why
            run.finish(3, f"filler block not started: {why}")
            self.filler_next = self.clock() + FILLER_RETRY_S
            self.log(f"filler: {run.name} not started ({why})")
            return
        sj = winsys.JobObject(priority_class=winsys.BELOW_NORMAL_PRIORITY_CLASS)
        try:
            running = self.launch(ctx, cmd, sj, run.dir / "logs")
        except OSError as e:
            sj.close()
            rec["status"], rec["note"] = "failed", f"could not start: {type(e).__name__}"
            run.finish(1, rec["note"])
            self.streak(run, 1)
            self.filler_next = self.clock() + FILLER_RETRY_S
            self.log(f"filler: {run.name} {rec['note']}")
            return
        rec["cmds"][cmd.name] = {"status": "running", "argv": cmd.symbolic()}
        rec["status"], rec["started_utc"] = "running", utc()
        run.save()
        self.filler = Filler(run, ctx, step, sj, running, size, self.clock(), sha if self.filler_root is not None else "")
        self.log(f"filler: {run.name} at {size} cores beside {self.running_job or 'idle'}"
                 + (f" (frozen copy of {sha[:7]})" if self.filler_root is not None else ""))

    def end_filler(self, rc: int) -> None:
        """A block ended by itself: its command's record, its account, its row, done.rc."""
        f, self.filler = self.filler, None
        assert f is not None
        run, rec = f.run, f.run.rec(f.step.name)
        failed = self.finish_cmd(run, f.ctx, f.running, rc, rec, self.clock())
        rec["status"], rec["finished_utc"] = ("failed" if failed else "done"), utc()
        self.close_filler(f)
        run.data["outputs"] = self.collect_outputs(run, [f.step], f.ctx)
        rc_job = 1 if failed else 0
        self.streak(run, rc_job)
        c = self.state["counters"]["jobs_done"]
        c["selfplay"] = c.get("selfplay", 0) + (1 if rc_job == 0 else 0)
        self.record(run, rc_job)
        run.finish(rc_job)
        self.lab.save_state(self.state)
        if failed:
            self.filler_next = self.clock() + FILLER_RETRY_S
        self.log(f"filler: {run.name} finished rc={rc_job}")

    def stop_filler(self, why: str, retry: bool = True) -> None:
        """Stop the running block (its Job Object: every process in it, by handle, never by image name). The runs it finished are
        in its shards; the folder is closed with done.rc 0 and the reason."""
        f, self.filler = self.filler, None
        if f is None:
            return
        rec = f.run.rec(f.step.name)
        flushed = ""
        try:
            flushed = self.ask_block_to_finish(f)
            self.kill_group(f.ctx, f.sj, [f.running], rec)
        finally:
            self.close_filler(f)
        rec["status"], rec["note"], rec["finished_utc"] = "stopped", why, utc()
        self.record(f.run, 0)
        f.run.finish(0, f"filler block stopped: {why}")
        if retry:
            self.filler_next = self.clock() + FILLER_RETRY_S
        self.log(f"filler: {f.run.name} stopped ({why})" + (f"; {flushed}" if flushed else ""))

    def ask_block_to_finish(self, f: Filler) -> str:
        """Before the kill: write STOP in the block's folder and wait up to FILLER_STOP_GRACE_S for it to end, so the runs it finished
        are in its shards. "" when the block does not read the file (it never wrote FILLER_HONORED), has ended or the grace is 0;
        else what happened, for the log."""
        d = f.run.dir
        if FILLER_STOP_GRACE_S <= 0 or not (d / FILLER_HONORED).exists() or f.running.proc.poll() is not None:
            return ""
        try:
            (d / FILLER_STOP_FILE).write_text(utc() + "\n", encoding="utf-8")
        except OSError:
            return ""
        t0 = time.monotonic()
        while f.running.proc.poll() is None and time.monotonic() - t0 < FILLER_STOP_GRACE_S:
            time.sleep(0.2)
        took = time.monotonic() - t0
        return (f"the block wrote its finished runs and ended in {took:.1f} s" if f.running.proc.poll() is not None
                else f"the block was still running after {FILLER_STOP_GRACE_S:.0f} s, killed")

    def close_filler(self, f: Filler) -> None:
        cpu = f.sj.cpu_time() / 1e7  # read after the end or the kill, so none of it lands in the job's account
        self.filler_cpu_done += cpu
        f.sj.close()
        f.run.data["cpu_secs"] = round(cpu, 1)
        f.run.data["wall_secs"] = round(self.clock() - f.t0, 1)

    def make_room(self, run: Run) -> None:
        """Before a job runs: its cores are taken from the filler at once (a job never waits for it). The night stops it outright; so
        does a presort when the filler runs from the lab tree (--filler-tree off), which the presort moves."""
        try:
            self.main_cores = cores.job_cores(run.job, run.data.get("params"), cores.load_doc(self.lab.root))
        except Exception:  # noqa: BLE001
            self.main_cores = cores.total_cores()
        if self.filler is None:
            return
        if run.job == "nightly-real" or (run.job == "presort" and self.filler_root is None):
            self.stop_filler("the night starts" if run.job == "nightly-real" else "a presort moves the tree")
            return
        self._filler_tick = float("-inf")
        self.service_filler(start=False)

    # ---- the lab-train line: the judgement holds the machine

    def quiet_module(self):
        """rl/quiet_lock.py of the checkout the runner reads (the protocol of `gen0.py score --quiet-lock`), or None where it is missing."""
        import importlib.util  # noqa: PLC0415

        path = ROOT / "rl" / "quiet_lock.py"
        try:
            spec = importlib.util.spec_from_file_location("lab_quiet_lock", path)
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod
        except Exception:  # noqa: BLE001
            return None

    def take_train_line(self) -> None:
        """While a train job runs, QUIET.lock carries `lab-train <UTC> pid=<runner>`: the gym daemon and the builders that wait on foreign lines
        pause of their own accord; the filler and the report jobs give way to it too. No waiting: the judgement only announces itself. The
        line is rewritten every TRAIN_BEAT_S and removed by `drop_train_line` when the job ends, however it ends."""
        if self.a.dry_run or self.train_lock is not None:
            return
        mod = self.quiet_module()
        if mod is None:
            self.log(f"QUIET.lock: {TRAIN_TAG} line not written (rl/quiet_lock.py is not readable)")
            return
        lock = mod.QuietLock(self.quiet_lock_path(), TRAIN_TAG)
        try:
            lock.announce()
        except OSError as e:
            self.log(f"QUIET.lock: {TRAIN_TAG} line not written ({type(e).__name__})")
            return
        self.train_lock, self._train_beat = lock, self.clock()
        self.log(f"QUIET.lock: {TRAIN_TAG} line written for the train job; the practice loop, the builders, the filler and the report wait for it")

    def service_train_line(self) -> None:
        """Refresh the line every TRAIN_BEAT_S while the job runs (also while it waits out a pause)."""
        lock = self.train_lock
        if lock is None or self.clock() - self._train_beat < TRAIN_BEAT_S:
            return
        self._train_beat = self.clock()
        try:
            lock.announce()
        except OSError as e:
            self.log(f"QUIET.lock: {TRAIN_TAG} line not refreshed ({type(e).__name__})")

    def drop_train_line(self) -> None:
        lock, self.train_lock = self.train_lock, None
        if lock is None:
            return
        try:
            lock.release()
        except OSError as e:
            self.log(f"QUIET.lock: {TRAIN_TAG} line not removed ({type(e).__name__}); the next stale-line sweep clears it")
            return
        self.log(f"QUIET.lock: {TRAIN_TAG} line removed")

    # ---- running a job

    def new_run(self, job: str, sha: str) -> Run:
        params = self.default_params(job)
        if job == "fights":
            self.state["next_start"] += params["batches"]
        elif job == "runs":
            self.state["next_run_start"] += params["batches"]
        self.next_strategy = None
        run = Run.create(self.lab, job, sha, params)
        n = self.state["counters"]["jobs_started"]
        n[job] = n.get(job, 0) + 1
        self.lab.save_state(self.state)
        return run

    def gate(self) -> guards.Verdict | None:
        """Wait out a pause (between steps); a verdict that makes the runner leave comes back."""
        paused = None
        while True:
            v = self.poll()
            if v.ok:
                if paused:
                    self.log(f"resuming (was held by: {paused})")
                return None
            if v.leave:
                return v
            if paused != v.kind:
                paused = v.kind
                tag = guards.pause_tag(self.lab.root) if v.kind == "pause" else ""
                self.log(f"paused: {v}" + (f" ({tag})" if tag else ""))  # the tag names the session that paused, to split pause time
            self.status(f"paused({v.kind})", v.detail)
            time.sleep(self.a.wait_poll)
            self.service_train_line()

    def cpu_secs(self) -> float:
        """CPU seconds of everything the runner has started for its jobs (its Job Object, less the reports that ran beside them,
        which belong to no job's strategy); 0 where there is none (a test)."""
        return max(0.0, self.root_job.cpu_time() / 1e7 - self.report_cpu() - self.filler_cpu()) if self.root_job else 0.0

    def filler_cpu(self) -> float:
        """CPU seconds the filler blocks have used in this process, the running one included (no job's account holds them)."""
        return self.filler_cpu_done + (self.filler.sj.cpu_time() / 1e7 if self.filler else 0.0)

    def report_cpu(self) -> float:
        """CPU seconds the reports have used in this process, the running one included."""
        return self.report_cpu_done + (self.report.sj.cpu_time() / 1e7 if self.report else 0.0)

    def account(self, run: Run) -> None:
        """Add the CPU and wall seconds since the last call to the job's record (cpu_secs, wall_secs in job.json)."""
        cpu, now = self.cpu_secs(), self.clock()
        c0, t0 = self._mark
        run.data["cpu_secs"] = round(float(run.data.get("cpu_secs", 0)) + max(0.0, cpu - c0), 1)
        run.data["wall_secs"] = round(float(run.data.get("wall_secs", 0)) + max(0.0, now - t0), 1)
        self._mark = (cpu, now)

    def execute(self, run: Run) -> guards.Verdict | None:
        """Run the job's steps in order; a verdict that makes the runner leave comes back (the job stays open)."""
        self.current_strategy = (run.data.get("params") or {}).get("strategy")
        self._mark = (self.cpu_secs(), self.clock())
        self.lock_stops = run.job == "presort"  # the landing lock guards git state only the presort touches
        self.running_job = run.job
        self.make_room(run)
        try:
            if run.job == "presort":
                self.report_gives_way_to_presort()
            elif run.job == "nightly-real":
                self.kill_report("a nightly window opens")
            elif run.job in REPORT_HOLD_JOBS:  # it would starve beside the job and die at its time limit: it starts over on an idle machine
                self.kill_report(f"a {run.job} job starts")
            if run.job == "train":
                self.take_train_line()
            return self._execute(run)
        finally:
            self.drop_train_line()  # however the job ended: finished, stopped by a guard or by a signal, a crash
            self.running_job = None
            self.main_cores = 0
            self.lock_stops = True
            self.release_nightly_consent(run.job)  # however the job stopped (finished, interrupted, an exception): its games are gone
            if not (run.dir / "done.rc").exists():  # stopped, not finished: what it used so far is kept for the next stretch
                self.account(run)
                run.save()
            self.current_strategy = None

    def _execute(self, run: Run) -> guards.Verdict | None:
        sha = run.data["main_sha"]
        ctx = self.ctx(run, run.job, sha, run.data["params"])
        steps = plans.plan(run.job, ctx)
        self.current = (run.name, "-")
        self.job_t0 = self.clock()
        stopped = False  # a required step failed
        for step in steps:
            rec = run.rec(step.name)
            if rec["status"] in ("done", "failed", "skipped"):
                stopped = stopped or (rec["status"] == "failed" and step.required)
                continue
            if stopped:
                rec["status"], rec["note"] = "skipped", "an earlier required step failed"
                continue
            v = self.gate()
            if v:
                return v
            self.current = (run.name, step.name)
            v = self.run_step(run, step, rec, ctx)
            if v:
                return v
            stopped = rec["status"] == "failed" and step.required
        run.data["outputs"] = self.collect_outputs(run, steps, ctx)
        core = [run.rec(s.name)["cmds"].get(c.name, {}).get("status") for s in steps for c in s.cmds if c.core]
        failed_all = bool(core) and "done" not in core and "failed" in core  # every core command that ran failed
        self.finish_job(run, 1 if stopped or failed_all else 0, ctx)
        return None

    def gpu_ok_path(self) -> str:
        """The GPU_OK file the nightly's consent line is in: the `--gpu-ok-file` of the nightly-real entry's cmd in LAB/jobs.json (the
        file the driver writes), else its `gpu_ok_file` key, else the default (jobs.consent_path)."""
        return jobs.consent_path(jobs.load_config(self.lab.root).get("nightly-real"))

    def release_nightly_consent(self, job: str) -> None:
        """A nightly-real job that stops leaves no `lab-nightly` line in GPU_OK. The driver removes it in a finally, but a job the
        runner kills (job-object terminate) or loses runs no finally, and the line would stand until the next nightly starts, telling
        the GPU lane it may fit beside a real-game arm that is not running. Called after every stop of such a job; never fails the job."""
        if job != "nightly-real":
            return
        try:
            path = self.gpu_ok_path()
            if jobs.clear_consent(path):
                self.log(f"GPU_OK: removed the {jobs.CONSENT_TAG} line of the stopped nightly-real job ({Path(path).name})")
        except Exception:  # noqa: BLE001
            self.log("GPU_OK not cleaned: " + traceback.format_exc()[-300:])

    def collect_outputs(self, run: Run, steps: list[plans.Step], ctx: plans.Ctx) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in steps:
            for c in s.cmds:
                if run.rec(s.name)["cmds"].get(c.name, {}).get("status") == "done":
                    for role, rel in c.outputs.items():
                        out.setdefault(role, []).append(ctx.expand(rel))
        return out

    def finish_job(self, run: Run, rc: int, ctx: plans.Ctx) -> None:
        st = self.state
        c = st["counters"]
        if run.job == "presort":
            sha = run.data["main_sha"]
            st["last_presort_end"] = time.time()
            if rc == 0:
                st["last_presorted_sha"], st["tree_sha"] = sha, sha
                st["bad_shas"].pop(sha, None)
            else:
                b = st["bad_shas"].setdefault(sha, {"fails": 0})
                b["fails"] += 1
                b["retry_after"] = time.time() + min(6 * 3600, 1800 * 2 ** (b["fails"] - 1))
                st["tree_sha"] = None  # the tree's sources moved but its build did not: nothing may run on it
        self.account(run)  # before the streak: an alert names the job's duration
        self.streak(run, rc)
        c["jobs_done"][run.job] = c["jobs_done"].get(run.job, 0) + (1 if rc == 0 else 0)
        c["active_seconds"] += int(self.clock() - getattr(self, "job_t0", self.clock()))
        cfg = self.reports_config()
        beside = (self.report_beside and rc == 0 and run.job in REPORT_JOBS  # its groups and kits run beside the next job
                  and (cfg["groups"]["enabled"] or cfg["kits"]["enabled"]))  # (no chain to run when both are off)
        if run.job == "presort" and run.rec("checks").get("status") == "done":
            st["last_checks"] = time.time()  # the cadence of `reports.checks.every_h` counts from the presort that ran the checks
        self.record(run, rc, yield_now=not beside)
        run.finish(rc)
        self.lab.save_state(st)
        self.log(f"job {run.name} finished rc={rc}")
        self.current = None
        self.status("idle")
        if beside:
            self.report_due.append((run, rc))
            self.sync_report_pending()
            self.service_report()

    def streak(self, run: Run, rc: int) -> None:
        """A strategy (or job type) that fails twice in a row stops being picked and raises an alert (LAB/alerts/<kind>.txt, tools/lab/alerts.py):
        the backoff lasts until the file is removed by hand (`alert_hold`). One success in between starts the count again. A presort has its own
        per-sha retry, which still decides when it is tried again; its second failure in a row raises the alert too, without holding it."""
        st = self.state
        streak = st.setdefault("fail_streak", {})
        recent = st.setdefault("fail_recent", {})
        key = (run.data.get("params") or {}).get("strategy") or run.job  # a strategy that keeps failing backs off alone
        if rc == 0:
            streak[key] = 0
            recent.pop(key, None)
            st["backoff"].pop(key, None)
            return
        streak[key] = streak.get(key, 0) + 1
        recent[key] = [*(recent.get(key) or []), {"job": run.name, "rc": rc, "secs": run.data.get("wall_secs", 0)}][-2:]
        if streak[key] >= 2:
            self.raise_alert(key, run.job != "presort")

    def raise_alert(self, key: str, blocks: bool) -> None:
        """Write LAB/alerts/<key>.txt for the two failures in `fail_recent`, hold the kind while it stands (`blocks`) and say so once in the log."""
        st = self.state
        path = alerts.alert_path(self.lab.root, key)
        fresh = not path.exists()
        try:
            alerts.write_failure(self.lab.root, key, st["fail_recent"].get(key, []), blocks=blocks)
        except OSError as e:
            self.log(f"ALERT {key}: could not be written ({type(e).__name__})")
        if blocks:
            st.setdefault("alerted", {})[key] = utc()
            st["backoff"][key] = INDEFINITE
        self.refresh_banners()
        if fresh:
            self.log(f"ALERT {key}: failed {st['fail_streak'].get(key, 2)} times in a row; "
                     + (f"no longer picked until LAB/alerts/{path.name} is removed by hand" if blocks else f"see LAB/alerts/{path.name}"))

    def refresh_banners(self) -> None:
        """The alerts at the top of summary.md and clock.md (the files the groups step and the daily clock write): redone when the alert files
        changed, so an alert shows (and goes) without waiting for the next groups run."""
        try:
            sig = tuple((a["name"], a["first"], a["count"]) for a in alerts.listing(self.lab.root))
            if sig == self._alert_sig:
                return
            self._alert_sig = sig
            for name in ("summary.md", "clock.md"):
                alerts.refresh(self.lab.root / name, self.lab.root)
        except Exception:  # noqa: BLE001  (a banner is a convenience)
            self.log("alert banner not refreshed: " + traceback.format_exc()[-300:])

    def note_timeout(self, run: Run, cmd: plans.Cmd, ran_s: float) -> None:
        """A command killed at its time limit goes on LAB/alerts/timeouts.txt (the first time too) and is said once in the log."""
        kind = (run.data.get("params") or {}).get("strategy") or run.job
        try:
            alerts.write_timeout(self.lab.root, kind, run.name, cmd.name, float(cmd.timeout or 0), ran_s, run.dir / "logs" / f"{cmd.name}.log")
        except OSError as e:
            self.log(f"ALERT timeout of {cmd.name}: could not be listed ({type(e).__name__})")
            return
        self.refresh_banners()
        self.log(f"ALERT timeout: {kind}/{cmd.name} of {run.name} was killed at its {float(cmd.timeout or 0):.0f}s limit (LAB/alerts/{alerts.TIMEOUTS})")

    def record(self, run: Run, rc: int, yield_now: bool = True) -> None:
        """A finished (or closed) job's CPU goes to its strategy's account, and its row to LAB/yield.tsv (at once, or when the
        report that groups it has ended: record_yield). Never fails the job."""
        try:
            data = run.data
            strategy = (data.get("params") or {}).get("strategy")
            spent = float(data.get("cpu_secs") or 0) or float(data.get("wall_secs") or 0)
            if strategy:
                cpu = self.state.setdefault("strategy_cpu", {})
                cpu[strategy] = round(float(cpu.get(strategy, 0.0)) + spent, 1)
                if rc == 0:
                    self.state.setdefault("last_ok", {})[strategy] = round(time.time())
            key = self.tree_once_key(run.job, strategy)
            if key and rc == 0 and data.get("main_sha"):
                self.state.setdefault("last_ok_sha", {})[key] = data["main_sha"]  # the tree it ran on: the next pick for this tree skips it
                if key == "retest":
                    self.state["retest_seen"] = retest.marked(self.groups_list())  # what is marked now is not "newly marked" at the next pick
        except Exception:  # noqa: BLE001  (the record is a convenience; the job's own files are what matters)
            self.log("strategy cpu not recorded: " + traceback.format_exc()[-400:])
        if yield_now:
            self.record_yield(run, rc)

    def record_yield(self, run: Run, rc: int, groups_ran: bool | None = None) -> None:
        try:
            data = run.data
            if groups_ran is None:
                groups_ran = any(r.get("name") == "groups" and r.get("status") == "done" for r in data.get("steps", []))
            row = mix.yield_row(run.dir, dict(data, rc=rc), read_json(self.lab.root / "groups.json", None), groups_ran, utc())
            mix.append_yield(self.lab.root, row)
        except Exception:  # noqa: BLE001
            self.log("yield not recorded: " + traceback.format_exc()[-400:])

    # ---- the report: groups and kits beside the next job

    def report_allowed(self) -> bool:
        """A report may start: none runs, a job left groups to do, the guards allow work, no presort is moving the tree, no nightly window is
        open (the real-game arms have the machine to themselves) and the machine is idle enough (`report_hold`)."""
        return (self.report_beside and self.report is None and bool(self.report_due) and self.guards_ok
                and self.running_job not in ("presort", "nightly-real") and not jobs.nightly_active(self.lab.root)
                and self.report_hold() is None)

    def report_hold(self) -> str | None:
        """Why a due report waits for an idle machine (D55 addendum 18): a judgement (train), a score or a nightly job runs, or the cores the
        runner measures free are under the report's own need (the machine less the running job's cores and the always-on loop's while it
        steps). The first time it holds a due report the log says so; it is tried again at every call. A queued report never holds a pick."""
        why = None
        try:
            doc = cores.load_doc(self.lab.root)
            if self.running_job in REPORT_HOLD_JOBS:
                why = f"a {self.running_job} job is running"
            else:
                pid, stepping = cores.loop_state(Path(self.a.loop_status), winsys.parent_alive)
                free = cores.total_cores(doc) - self.main_cores - (cores.loop_cores(doc) if pid and stepping else 0)
                need = cores.report_cores(doc)
                if free < need:
                    why = f"{max(0, free)} cores free, the report needs {need}"
        except Exception:  # noqa: BLE001  (a reading that fails never holds a report: it was never held before)
            why = None
        if self.report is None and self.report_due and self.report_beside:
            note = why or ""
            if note != self.report_note:
                self.report_note = note
                if why:
                    self.log(f"report beside: held ({why}); {len(self.report_due)} job(s) wait, tried again at the next pick")
        return why

    def service_report(self) -> None:
        """Called wherever the runner waits or loops: start the report when one is due, move a running one on. A fault here turns
        the feature off for this process and writes the covered jobs' rows without group counts; it never fails a job."""
        if not self.report_beside:
            return
        try:
            self._service_report()
        except Exception:  # noqa: BLE001
            self.log("report beside: fault, groups and kits run as steps of each job from now on: " + traceback.format_exc()[-500:])
            self.report_beside = False
            self.kill_report("fault")
            self.flush_report_rows()

    def _service_report(self) -> None:
        rep = self.report
        if rep is not None and self.running_job in REPORT_HOLD_JOBS and self.running_job != "nightly-real":
            self.kill_report(f"a {self.running_job} job is running: the report waits for an idle machine")
            return
        if rep is None:
            if self.report_allowed():
                self.start_report()
            return
        r = rep.running
        if r is None:
            self.step_report()
            return
        rc = r.proc.poll()
        now = self.clock()
        if rc is None and r.cmd.timeout and now - r.t0 > r.cmd.timeout:
            winsys.kill_tree_by_pid(r.proc.pid)
            rc = r.proc.wait(timeout=30)
            rep.run.rec(r.cmd.name)["cmds"].setdefault(r.cmd.name, {})["note"] = f"killed after the {r.cmd.timeout:.0f}s limit"
            self.note_timeout(rep.run, r.cmd, now - r.t0)
        if rc is None:
            return
        rec = rep.run.rec(r.cmd.name)
        failed = self.finish_cmd(rep.run, rep.ctx, r, rc, rec, now)
        rec["status"] = "failed" if failed else "done"
        rep.run.save()
        rep.ok[r.cmd.name] = not failed
        rep.failed = rep.failed or failed
        rep.running = None
        self.step_report()

    def start_report(self) -> None:
        covers = self.report_due
        run = covers[-1][0]
        ctx = self.ctx(run, run.job, run.data["main_sha"], run.data["params"])
        cmds = plans.report_cmds(ctx)  # before the covered jobs leave the queue: a fault here keeps them
        self.report = Report(run, ctx, cmds, covers, winsys.JobObject())
        self.report_due = []
        self.log(f"report beside: groups and kits of {', '.join(r.name for r, _ in covers)}, beside the next job")
        self.step_report()

    def step_report(self) -> None:
        """Launch the chain's next command, or end the chain (after a failure the later commands do not run)."""
        rep = self.report
        assert rep is not None
        while rep.running is None:
            if not rep.cmds or rep.failed:
                self.end_report()
                return
            cmd = rep.cmds.pop(0)
            rec = rep.run.rec(cmd.name)
            why = cmd.skip_reason(rep.ctx, cmd.name)
            if why:
                rec["cmds"][cmd.name] = {"status": "skipped", "note": why}
                rec["status"] = "skipped"
                self.log(f"{cmd.name}/{cmd.name}: skipped ({why})")
                continue
            cmd.timeout = cmd.timeout or REPORT_TIMEOUT_S
            try:
                rep.running = self.launch(rep.ctx, cmd, rep.sj, rep.run.dir / "logs")
            except OSError as e:
                rec["cmds"][cmd.name] = {"status": "failed", "rc": 127, "note": f"could not start: {type(e).__name__}"}
                rec["status"] = "failed"
                rep.failed = True
                rep.ok[cmd.name] = False
                self.log(f"{cmd.name}/{cmd.name}: could not start ({type(e).__name__})")
                continue
            rec["cmds"][cmd.name] = {"status": "running", "argv": cmd.symbolic()}
            rec["status"] = "running"
            rec["started_utc"] = utc()
            rep.run.save()

    def end_report(self) -> None:
        rep, self.report = self.report, None
        assert rep is not None
        self.report_cuts = 0  # a report ended: the next presort may stop one again
        self.report_cpu_done += rep_cpu(rep)
        rep.sj.close()
        rep.run.save()
        for run, rc in rep.covers:
            self.record_yield(run, rc, groups_ran=rep.ok.get("groups", False))
        self.sync_report_pending()
        self.log(f"report beside: ended after {time.monotonic() - rep.t0:.0f}s ({'failed' if rep.failed else 'ok'}); "
                 f"{len(rep.covers)} job(s) grouped")

    def kill_report(self, why: str) -> None:
        """Stop the running report; its covered jobs wait for the next one."""
        if self.report is None:
            return
        rep, self.report = self.report, None
        try:
            rec = rep.run.rec(rep.running.cmd.name) if rep.running else {"cmds": {}}
            self.kill_group(rep.ctx, rep.sj, [rep.running] if rep.running else [], rec)
        finally:
            self.report_cpu_done += rep_cpu(rep)  # what it used up to its end stays out of the jobs' accounts
            rep.sj.close()
        for rname in ("groups", "kits"):  # a stopped command is neither done nor failed: it runs in the next report
            r = next((x for x in rep.run.data["steps"] if x["name"] == rname and x.get("status") == "running"), None)
            if r:
                r["status"] = "pending"
        rep.run.save()
        self.report_due = rep.covers + self.report_due
        self.log(f"report beside: stopped ({why}); {len(rep.covers)} job(s) wait for the next report")

    def report_gives_way_to_presort(self) -> None:
        """A presort checks out, builds and publishes into the tree the running report reads (EmuDump's build, the DLL, git), so the two
        cannot run together, and one of them must give way. A report is 40 to 70 minutes (groups 10 to 36, kits up to 48) and the next
        starts the moment one ends, so one is nearly always running and a presort that waited for it waited half an hour: 22 of the
        30 minutes of the presort of 2026-11-04T01:35Z, with the main slot held and only the filler working.

        Default (--presort-report stop): the report is stopped at once; its jobs wait for the next report, which starts when the presort
        ends and so reads the new tree (the groups and kits that did not change skip themselves, tools/lab/stepcache.py). What a
        stopped report had done of its running command is lost, so a report cannot be stopped more than --report-cut-cap times in a row
        by presorts: then the presort waits for it as before, and the count starts again when a report ends."""
        rep = self.report
        if rep is None:
            return
        a = self.a
        if a.presort_report == "wait" or self.report_cuts >= max(0, a.report_cut_cap):
            note = "" if a.presort_report == "wait" else f"; {self.report_cuts} reports were stopped by presorts in a row, this one ends first"
            self.drain_report("the presort moves the tree the report runs from" + note)
            return
        self.report_cuts += 1
        self.kill_report(f"the presort moves the tree the report runs from; it had run {time.monotonic() - rep.t0:.0f}s, "
                         f"{self.report_cuts} of {max(0, a.report_cut_cap)} stops allowed in a row")

    def drain_report(self, why: str) -> None:
        """Wait for the running report to end (a guard that stops it ends the wait)."""
        if self.report is not None:
            self.log(f"report beside: waiting for it to end ({why})")
        while self.report is not None and not self.stop_signal:
            self.poll()
            self.service_report()
            self.service_filler()
            if self.report is not None:
                time.sleep(self.a.poll)

    def flush_report_rows(self) -> None:
        """Rows of jobs whose groups never ran (the runner stops, or the feature failed): written without group counts."""
        due, self.report_due = self.report_due, []
        for run, rc in due:
            self.record_yield(run, rc, groups_ran=False)
        self.sync_report_pending()

    def pending_report_jobs(self) -> list[dict[str, Any]]:
        """The finished jobs whose yield row waits for a report: the chain running now covers some, the queue holds the rest."""
        covered = self.report.covers if self.report is not None else []
        return [{"name": r.name, "rc": rc} for r, rc in [*covered, *self.report_due]]

    def sync_report_pending(self) -> None:
        """Keep that list in the state file whenever it changes: a runner killed or restarted loses its memory, and the jobs' rows (their
        groups and fix-ready counts) would be lost with it; restore_report_pending reads the list back at the next start."""
        pend = self.pending_report_jobs()
        if pend != (self.state.get("report_pending") or []):
            self.state["report_pending"] = pend
            self.lab.save_state(self.state)

    def restore_report_pending(self) -> None:
        """At start: the jobs the last process left with a row owed go back on the report queue (their groups and kits run beside the next
        job); with the report feature off they get their rows now, without group counts. A job that already has its row (killed between the
        row and the state write), or whose folder is gone, is dropped."""
        owed = [e for e in (self.state.get("report_pending") or []) if isinstance(e, dict)]
        if not owed:
            return
        have = {str(r.get("job")) for r in mix.read_yield(self.lab.root)}
        got: list[tuple[Run, int]] = []
        for e in owed:
            name, rc = e.get("name"), e.get("rc")
            run = Run.load(self.lab.runs / name) if isinstance(name, str) and name not in have and isinstance(rc, int) else None
            if run is not None:
                got.append((run, rc))
        self.state["report_pending"] = []
        if got and self.report_beside:
            self.report_due = got + self.report_due
            self.log(f"report beside: {len(got)} job(s) left by the last runner process wait for the next report: "
                     + ", ".join(r.name for r, _ in got))
        else:
            for run, rc in got:
                self.record_yield(run, rc, groups_ran=False)
            if got:
                self.log(f"yield: {len(got)} row(s) left by the last runner process written without group counts (the report beside is off)")
        self.sync_report_pending()
        self.lab.save_state(self.state)

    def run_step(self, run: Run, step: plans.Step, rec: dict[str, Any], ctx: plans.Ctx) -> guards.Verdict | None:
        while True:
            try:
                self.exec_step(run, step, rec, ctx)
                return None
            except Interrupted as e:
                c = self.state["counters"]["interrupts"]
                c[e.verdict.kind] = c.get(e.verdict.kind, 0) + 1
                rec["interrupted"] = rec.get("interrupted", 0) + 1
                rec["status"] = "pending"
                run.save()
                self.lab.save_state(self.state)
                self.log(f"step {step.name} stopped: {e.verdict}")
                if e.verdict.leave:
                    return e.verdict
                v = self.gate()
                if v:
                    return v

    def exec_step(self, run: Run, step: plans.Step, rec: dict[str, Any], ctx: plans.Ctx) -> None:
        rec["status"], rec["started_utc"] = "running", rec.get("started_utc") or utc()
        run.save()
        self.status("running", step.name)
        groups = [[c] for c in step.cmds] if step.mode == "serial" else [list(step.cmds)]
        failed = False
        for group in groups:
            if failed:
                break
            if self.run_group(run, step, group, rec, ctx):
                failed = True
        ran = [v for v in rec["cmds"].values()]
        rec["status"] = "failed" if failed else ("skipped" if ran and all(v["status"] == "skipped" for v in ran) else "done")
        rec["finished_utc"] = utc()
        run.save()

    def run_group(self, run: Run, step: plans.Step, cmds: list[plans.Cmd], rec: dict[str, Any], ctx: plans.Ctx) -> bool:
        """Run `cmds` at once under one Job Object; True when one of them failed."""
        logs = run.dir / "logs"
        todo: list[plans.Cmd] = []
        for cmd in cmds:
            kept = read_rc(logs / f"{cmd.name}.rc")
            if kept is not None and kept in cmd.ok_rcs:
                rec["cmds"].setdefault(cmd.name, {"status": "done", "rc": kept, "note": "kept from before the stop"})
                continue
            why = cmd.skip_reason(ctx, step.name)
            if why:
                rec["cmds"][cmd.name] = {"status": "skipped", "note": why}
                self.log(f"{step.name}/{cmd.name}: skipped ({why})")
                continue
            todo.append(cmd)
        if not todo:
            return False
        v = self.poll()
        if not v.ok:
            raise Interrupted(v)
        sj = winsys.JobObject()
        running: list[Running] = []
        failed = False
        grace_end: float | None = None
        try:
            for cmd in todo:
                try:
                    running.append(self.launch(ctx, cmd, sj, logs))
                    rec["cmds"][cmd.name] = {"status": "running", "argv": cmd.symbolic()}
                except OSError as e:  # the program is not there (dotnet, git ...): a failed command, not a crashed runner
                    rec["cmds"][cmd.name] = {"status": "failed", "rc": 127, "note": f"could not start: {type(e).__name__}", "argv": cmd.symbolic()}
                    (logs / f"{cmd.name}.rc").write_text("127\n")
                    self.log(f"{step.name}/{cmd.name}: could not start ({type(e).__name__})")
                    failed = True
            run.save()
            while running:
                time.sleep(self.a.poll)
                self.service_train_line()
                self.service_report()
                self.service_filler()
                now = self.clock()
                for r in list(running):
                    rc = r.proc.poll()
                    if rc is None and r.cmd.timeout and now - r.t0 > r.cmd.timeout:
                        winsys.kill_tree_by_pid(r.proc.pid)
                        rc = r.proc.wait(timeout=30)
                        rec["cmds"][r.cmd.name]["note"] = f"killed after the {r.cmd.timeout:.0f}s limit"
                        self.note_timeout(run, r.cmd, now - r.t0)
                    if rc is not None:
                        running.remove(r)
                        failed |= self.finish_cmd(run, ctx, r, rc, rec, now)
                self.heartbeat("running", f"{step.name}: {len(running)} running")
                v = self.poll()
                if v.ok:
                    grace_end = None
                elif running:
                    grace_end = grace_end if grace_end is not None else now + v.grace
                    if now >= grace_end:
                        raise Interrupted(v)
        except (Interrupted, KeyboardInterrupt, Halted):
            self.kill_group(ctx, sj, running, rec)
            raise
        finally:
            sj.close()
        run.save()
        return failed

    def launch(self, ctx: plans.Ctx, cmd: plans.Cmd, sj: winsys.JobObject, logs: Path) -> Running:
        argv = [ctx.expand(x) for x in cmd.argv]
        for m in cmd.mkdirs:
            ctx.path(m).mkdir(parents=True, exist_ok=True)
        for g in cmd.cleanup:
            self.rm_glob(ctx, g)
        if cmd.pre:
            cmd.pre(ctx)
        env = dict(os.environ)
        env.update(CHILD_ENV)
        env.update({k: ctx.expand(v) for k, v in cmd.env.items()})
        if cmd.msvc:
            keep = [p for p in env.get("PATH", "").split(os.pathsep) if p and "usr" + os.sep + "bin" not in p.lower().replace("/", os.sep)]
            env["PATH"] = os.pathsep.join([ctx.expand("{vs_installer}").replace("/", os.sep), *keep])
        logs.mkdir(parents=True, exist_ok=True)
        log_path = ctx.path(cmd.stdout) if cmd.stdout else logs / f"{cmd.name}.log"
        marker = f"\n# ---- restart {utc()}: the lines above are an earlier attempt's ----\n".encode()
        had = cmd.append_log and log_path.is_file() and log_path.stat().st_size > 0
        out = open(log_path, "ab" if cmd.append_log else "wb")  # noqa: SIM115
        files = [out]
        if had:
            out.write(marker)
        out.write(f"# {cmd.symbolic()}\n# started {utc()}\n".encode())
        out.flush()
        if cmd.stderr:
            p_err = ctx.path(cmd.stderr)
            p_err.parent.mkdir(parents=True, exist_ok=True)
            err_had = cmd.append_log and p_err.is_file() and p_err.stat().st_size > 0
            err = open(p_err, "ab" if cmd.append_log else "wb")  # noqa: SIM115
            if err_had:
                err.write(marker)
            files.append(err)
        else:
            err = subprocess.STDOUT
        kw: dict[str, Any] = {"cwd": ctx.expand(cmd.cwd), "env": env, "stdout": out, "stderr": err, "stdin": subprocess.DEVNULL}
        if winsys.IS_WIN:
            kw.update(hidden_console_kw())
        in_job = True
        stamp0 = self.stamp_key(cmd.name)
        cap = None
        try:
            cap = self.memory_cap(cmd)
            proc = winsys.suspended_popen(argv, sj, cap, **kw)
        except OSError as e:
            if isinstance(e, FileNotFoundError) or cmd.mem_share > 0:  # a command with a memory cap is never started without it
                if cap:
                    cap.close()
                for f in files:
                    f.close()
                raise
            in_job = False  # could not nest in a job: a plain start, killed by process tree instead
            kw["creationflags"] = kw.get("creationflags", 0) & ~winsys.CREATE_SUSPENDED
            proc = subprocess.Popen(argv, **kw)
        return Running(cmd, proc, files, self.clock(), in_job, self.account_job(proc) if in_job else None, stamp0, cap)

    @staticmethod
    def memory_cap(cmd: plans.Cmd) -> winsys.JobObject | None:
        """The Job Object that holds `cmd`'s processes, children included, to `cmd.mem_share` of the machine's RAM committed together
        (an allocation past it fails in the process that asks, which is how a runaway ends at the cap and not at the machine's
        limit). None for a command with no share. Raises OSError where the share cannot be worked out or the limit not set:
        the caller does not start the command."""
        if cmd.mem_share <= 0:
            return None
        limit = winsys.ram_share(cmd.mem_share)
        if winsys.IS_WIN and not limit:
            raise OSError(f"the machine's RAM is unknown: no memory cap for {cmd.name}")
        return winsys.JobObject(kill_on_close=False, job_memory_limit=limit) if limit else None

    @staticmethod
    def account_job(proc: subprocess.Popen) -> winsys.JobObject | None:
        """A Job Object for this command alone, nested in the step's, so its CPU seconds and I/O are its own (children included)
        even when the step runs several commands at once. A command already gone, or a job that cannot nest, costs only the figures."""
        j = None
        try:
            j = winsys.JobObject(kill_on_close=False)
            if not j.h:
                return None
            j.assign_handle(int(proc._handle))  # type: ignore[attr-defined]
            return j
        except Exception:  # noqa: BLE001  (accounting never stops a job)
            if j is not None:
                j.close()
            return None

    def stamp_key(self, name: str) -> tuple | None:
        """What the command's skip stamp says before and after it runs (LAB/stamps/<name>.json), None where it has none."""
        try:
            st = sc.read_stamp(self.lab.root, name)
            return (st["digest"], st.get("utc"), json.dumps(st["outputs"], sort_keys=True), st.get("changed")) if st else None
        except Exception:  # noqa: BLE001
            return None

    def output_verdict(self, r: Running, ok: bool) -> str:
        """Whether the command's output changed, from its skip stamp: `changed` or `same` (the stamp was written again and the output
        differs or does not), `skipped` (the stamp stands: the command found its inputs unchanged), `?` (no stamp, a failure, or a
        stamp that cannot say)."""
        if not ok:
            return "?"
        after = self.stamp_key(r.cmd.name)
        if after is None:
            return "?"
        if after[:3] == (r.stamp0 or (None,) * 3)[:3]:
            return "skipped"
        return {True: "changed", False: "same", None: "?"}[after[3]]

    @staticmethod
    def accounting_fields(r: Running) -> dict[str, Any]:
        """The figures of a finished command: CPU seconds, read operations and MB read, MB written; empty where unknown."""
        acc = None
        try:
            acc = r.acct.accounting() if r.acct else None
        except Exception:  # noqa: BLE001
            acc = None
        finally:
            if r.acct:
                r.acct.close()
                r.acct = None
        out: dict[str, Any] = {}
        if acc:
            out = {"cpu_secs": round(acc["cpu_secs"], 1), "reads": int(acc["read_ops"]), "read_mb": round(acc["read_bytes"] / 1e6, 1),
                   "write_mb": round(acc["write_bytes"] / 1e6, 1)}
        if r.cap:  # the most the command's processes held together, for a command with a memory cap
            try:
                peak = r.cap.peak_memory()
            except Exception:  # noqa: BLE001
                peak = None
            r.cap.close()
            r.cap = None
            if peak:
                out["peak_mb"] = round(peak["job"] / 2**20)
        return out

    def finish_cmd(self, run: Run, ctx: plans.Ctx, r: Running, rc: int, rec: dict[str, Any], now: float) -> bool:
        for f in r.files:
            f.close()
        cmd = r.cmd
        (run.dir / "logs" / f"{cmd.name}.rc").write_text(f"{rc}\n")
        ok = rc in cmd.ok_rcs
        entry = rec["cmds"].setdefault(cmd.name, {})
        entry.update({"status": "done" if ok else "failed", "rc": rc, "secs": round(now - r.t0, 1)})
        acct = self.accounting_fields(r)
        for g in cmd.after_cleanup:
            self.rm_glob(ctx, g)
        if ok and cmd.post:
            problem = cmd.post(ctx)
            if problem:
                entry["note"] = problem
                self.log(f"{cmd.name}: {problem}")
                if problem.startswith("HALT:"):
                    atomic_write(self.lab.root / "HALT", problem + "\n" + "Remove this file after reading the job's logs to let the lab run again.\n")
                    run.save()
                    raise Halted(problem)
        if ok:
            counters = self.state["counters"]
            for k, n in cmd.counts.items():
                counters[k] = counters.get(k, 0) + n
        out = self.output_verdict(r, ok)
        entry.update({**acct, "out": out})  # the clock's figures stay in job.json beside the seconds
        self.log(f"{rec['name']}/{cmd.name}: rc={rc} in {entry['secs']}s " + self.clock_fields(acct, out) + ("" if ok else " (FAILED)"))
        return not ok

    @staticmethod
    def clock_fields(acct: dict[str, Any], out: str) -> str:
        """The accounting part of a step's log line, key=value for tools/lab/clock_report.py: wall is the `in ...s` before it."""
        peak = f" peak_mb={acct['peak_mb']}" if "peak_mb" in acct else ""  # only a command with a memory cap has one
        if "cpu_secs" in acct:
            return (f"cpu={acct['cpu_secs']}s reads={acct['reads']} read_mb={acct['read_mb']} write_mb={acct['write_mb']} out={out}{peak}")
        return f"cpu=? reads=? read_mb=? write_mb=? out={out}{peak}"

    def kill_group(self, ctx: plans.Ctx, sj: winsys.JobObject, running: list[Running], rec: dict[str, Any]) -> None:
        sj.terminate()
        for r in running:
            try:
                r.proc.kill()
            except OSError:
                pass
            if not r.in_job:
                winsys.kill_tree_by_pid(r.proc.pid)
            try:
                r.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                winsys.kill_tree_by_pid(r.proc.pid)
            for f in r.files:
                f.close()
            if r.acct:
                r.acct.close()
                r.acct = None
            if r.cap:
                r.cap.close()
                r.cap = None
            for g in [*r.cmd.cleanup, *r.cmd.after_cleanup]:
                self.rm_glob(ctx, g)  # a partial output is worse than none
            rec["cmds"].pop(r.cmd.name, None)

    def rm_glob(self, ctx: plans.Ctx, pattern: str) -> None:
        """Remove what a command wrote, only under the job folder, the lab or the game copy's serve/ folder."""
        roots = [ctx.job_dir.resolve(), self.lab.root.resolve(), (ctx.game.parent / "serve").resolve()]
        for m in glob.glob(ctx.expand(pattern)):
            p = Path(m).resolve()
            if not any(p != r and r in p.parents for r in roots):
                continue
            try:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
            except OSError:
                pass

    # ---- top level

    def setup(self) -> int | None:
        """None: go on. An int: the exit code (2 refused arguments, 0 another runner already holds this lab)."""
        for label, p in (("lab", self.lab.root), ("tree", self.tree), ("game", self.game), ("filler-tree", self.filler_root)):
            if p is None:
                continue
            q = str(p.resolve()).replace("\\", "/").lower()
            if any(q.startswith(f.lower()) for f in FORBIDDEN):
                self.log(f"refused: --{label} is under the reviewed upstream checkout")
                return 2
        if self.tree.resolve() == self.fork.resolve():
            self.log("refused: --tree is the fork's main checkout; the lab works in its own worktree")
            return 2
        try:
            self.lab.root.resolve().relative_to(self.fork.resolve())
            self.log("refused: --lab is inside the fork; the lab lives outside every repo")
            return 2
        except ValueError:
            pass
        self.lab.root.mkdir(parents=True, exist_ok=True)
        self.lab.runs.mkdir(exist_ok=True)
        if not winsys.acquire_mutex("SpirelineFixLab-" + format(zlib.crc32(str(self.lab.root.resolve()).lower().encode()), "08x")):
            self.log("another runner already holds this lab; exiting")
            return 0
        winsys.set_below_normal()
        try:
            self.root_job = winsys.JobObject()
            self.root_job.assign_self()
        except OSError as e:
            self.log(f"warning: no Job Object for the runner ({e}); children will not die with it")
            self.root_job = None
        for sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
            if hasattr(signal, sig):
                try:
                    signal.signal(getattr(signal, sig), self.on_signal)
                except ValueError:  # not the main thread (a test)
                    pass
        return None

    def on_signal(self, signum: int, _frame: Any) -> None:
        self.stop_signal = f"signal {signum}"

    def run(self) -> int:
        rc = self.setup()
        if rc is not None:
            return rc
        self.log(f"runner started (trial={self.a.trial}, once={self.a.once}, guards off: {self.a.no_guard or 'none'})")
        self.seed_last_ok_sha()
        self.restore_report_pending()
        crashes = 0
        while True:
            try:
                code = self.one_pass()
            except Halted as e:
                self.status("halted", str(e))
                self.log(f"HALT: {e}")
                if not self.a.always_on:
                    return 0
                self.current = None  # always on: nothing would start the runner again, so it waits at the gate for LAB/HALT to go
                continue
            except KeyboardInterrupt:
                return self.leave(guards.Verdict("signal", "interrupted", leave=True))
            except Exception:  # noqa: BLE001  (a bug in one pass must not end a night's work at once; three in a row end it)
                crashes += 1
                self.log(f"CRASH {crashes}/3 " + traceback.format_exc()[-1200:])
                self.status("crashed", f"{crashes} in a row")
                self.current = None
                if crashes >= 3:
                    return 1
                if self.sleep_idle(min(self.a.idle_poll, 60.0)):
                    return self.leave(self.poll())
                continue
            if code is not None:
                return code
            crashes = 0

    def quiet_lock_path(self) -> Path:
        return Path(self.a.quiet_lock) if self.a.quiet_lock else self.lab.root.parent / "QUIET.lock"

    def clear_stale_quiet_lock(self) -> bool:
        """Remove a stale `lab-*` line from the shared QUIET.lock; True when it did. Called between jobs, when this runner (the
        only one of its lab) has no job running, so a `lab-*` line is a holder that is gone unless its process is alive. Stale:
        older than STALE_LOCK_S (three refresh intervals; its time, else the file's) and either naming no process or naming one
        that is gone (`pid=<n>`, written by rl/nightly_real.py). Other sessions' lines (`w3-blocks`, ...) are never touched."""
        path = self.quiet_lock_path()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            mtime = path.stat().st_mtime
        except OSError:
            return False
        parts = (text.strip().splitlines() or [""])[0].split()
        if not parts or not parts[0].startswith("lab-"):
            return False
        try:
            t = dt.datetime.strptime(parts[1], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.UTC).timestamp() if len(parts) > 1 else mtime
        except ValueError:
            t = mtime
        age = time.time() - t
        pid = next((int(x[4:]) for x in parts[2:] if x.startswith("pid=") and x[4:].isdigit()), None)
        if age < STALE_LOCK_S or (pid is not None and winsys.parent_alive(pid)):
            return False
        try:
            if path.read_text(encoding="utf-8", errors="replace") != text:
                return False  # rewritten while we looked: it is live
            path.unlink()
        except OSError:
            return False
        self.log(f"QUIET.lock: cleared the stale line '{' '.join(parts)}' ({age / 60:.0f} min old; "
                 + (f"process {pid} is gone" if pid is not None else "no process named, none of this lab's jobs is running") + ")")
        return True

    def daily_clock(self) -> None:
        """Write LAB/clock.md, the flagged steps of the last week of this log (tools/lab/clock_report.py), at most once a day and only for a
        runner that stays (not --once, --trial or --dry-run), and say so in the log when a step is flagged. A fault here is a log line."""
        if self.a.once or self.a.trial or self.a.dry_run:
            return
        p = self.lab.root / "clock.md"
        try:
            if p.exists() and time.time() - p.stat().st_mtime < CLOCK_EVERY_S:
                return
            rep = clock_report.report_for(self.lab.root)
            if rep is None:
                return
            atomic_write(p, alerts.with_banner(clock_report.render(rep) + "\n", self.lab.root))
            if rep["alarms"]:
                alarmed = [k for k in rep["flagged"] if rep["by_step"][k]["alarm"]]
                self.log(f"clock: {rep['alarms']} step(s) flagged over the last {rep['window_days']} days (clock.md): " + ", ".join(
                    f"{k} {' '.join(rep['by_step'][k]['flags'])}" for k in alarmed[:5]))
        except Exception as e:  # noqa: BLE001  (a report must not stop the lab)
            self.log(f"clock: report not written ({type(e).__name__})")

    def one_pass(self) -> int | None:
        """Wait out any pause, pick a job, run it. An exit code when the runner should end, else None."""
        a = self.a
        v = self.gate()
        if v:
            return self.leave(v)
        self.clear_stale_quiet_lock()
        self.daily_clock()
        self.refresh_banners()
        self.service_report()
        self.service_filler(start=False)  # between jobs: a block that ended is closed, none starts before the pick
        sha = self.main_sha()
        job, run, why = self.choose(sha)
        if not job:
            self.lab.save_state(self.state)
            self.status("idle", why)
            self.log(f"nothing to do: {why}")
            if a.once:
                return 0
            if self.sleep_idle():
                return self.leave(self.poll())
            return None
        assert sha
        if run is None:
            run = self.new_run(job, self.defer_sha if job != "presort" and self.defer_sha else sha)
        else:
            run.data["resumed"] += 1
            run.save()
        self.since = utc()
        self.log(f"job {run.name}: {why}")
        v = self.execute(run)
        if v:
            return self.leave(v)
        return 0 if a.once else None

    def sleep_idle(self, seconds: float | None = None) -> bool:
        """Sleep up to idle_poll seconds in short slices; True when a guard says leave."""
        end = self.clock() + (self.a.idle_poll if seconds is None else seconds)
        while self.clock() < end:
            time.sleep(min(self.a.wait_poll, 5.0))
            v = self.poll()  # first: a pause or a guard that came during the wait holds the filler before it starts, not after
            if v.leave:
                return True
            self.service_report()
            self.service_filler()  # the lab's idle time is the filler's (start=True): the cores the loop and the report leave
            self.heartbeat("idle")
        return False

    def leave(self, v: guards.Verdict) -> int:
        self.drop_train_line()
        self.kill_report(f"the runner stops: {v}")
        self.stop_filler(f"the runner stops: {v}")
        self.flush_report_rows()
        self.lab.save_state(self.state)
        self.status(f"stopped({v.kind})", v.detail)
        self.log(f"stopping: {v}")
        return 0

    # ---- dry run

    def dry_run(self) -> int:
        a = self.a
        sha = self.main_sha()
        g = self.guards
        g.prune = None  # a dry run reports the disk guard; it never prunes
        v = g.poll()
        mon = guards.LoadMonitor(lambda: 0, window=5, interval=0)
        mon.sample()
        time.sleep(1.0)
        mon.sample()
        load = f"{mon.mean()[0] * 100:.0f}% busy over 1 s (the runner's own children would not count)"
        size = guards.dir_bytes(self.lab.root) / guards.GB if self.lab.root.exists() else 0
        free = shutil.disk_usage(self.lab.root if self.lab.root.exists() else self.lab.root.parent).free / guards.GB
        st = self.state
        print(f"lab {self.lab.root.name}: {len(list(self.lab.runs.iterdir())) if self.lab.runs.is_dir() else 0} job folders, "
              f"{len(self.lab.runs_without_done())} unfinished; state: presorted {str(st['last_presorted_sha'])[:7]}, tree {str(st['tree_sha'])[:7]}, "
              f"rotation {st['rotation']}, next start {st['next_start']}")
        print(f"main {str(sha)[:7]}; user idle {winsys.user_idle_seconds()} s (stop below {a.idle_seconds:g}); "
              f"PAUSE {(self.lab.root / 'PAUSE').exists()}; main.lock {(Path(a.backlog_dir) / 'main.lock').exists()}; "
              f"HALT {(self.lab.root / 'HALT').exists()}; load {load}; lab {size:.2f} GB of {a.lab_gb:g}; drive free {free:.0f} GB (min {a.free_gb:g})")
        print(f"guard verdict now: {v}")
        doc = cores.load_doc(self.lab.root)
        saved, self.filler_mode = self.filler_mode, a.filler != "off"
        try:
            hold = self.filler_hold(doc, starting=True)
            fsize = self.filler_room(doc)
        finally:
            self.filler_mode = saved
        print(f"filler (D48): {'would start at ' + str(fsize) + ' cores' if hold is None and fsize else 'not now: ' + (hold or 'under the floor')}"
              f"; cores: total {cores.total_cores(doc)}, floor {cores.floor(doc)}, busy now {self.busy_cores(doc)}")
        weights, notes = mix.load_mix(self.lab.root, create=False)
        print("mix (wanted share / CPU share had): " + mix.share_table(weights, st["strategy_cpu"]) + ("; " + "; ".join(notes) if notes else ""))
        job, run, why = self.choose(sha)
        if not job:
            print(f"next job: none ({why})")
            return 0
        print(f"next job: {job} ({why})")
        params = run.data["params"] if run else self.default_params(job)
        at = self.defer_sha if job != "presort" and self.defer_sha else sha  # the tree sha while a presort is deferred
        c = self.ctx(run, job, at or "", params)
        try:
            steps = plans.plan(job, c)
        except plans.Unavailable as e:
            print(f"  not runnable: {e}")
            return 0
        print(f"  params: {json.dumps(params)}")
        tree_ready = plans.git_head(self.tree) == at
        for s in steps:
            print(f"  step {s.name} [{s.mode}{', required' if s.required else ''}]")
            for cmd in s.cmds:
                why_skip = cmd.skip_reason(c, s.name)
                if why_skip and not tree_ready and "on this tree" in why_skip and job == "presort":
                    why_skip = why_skip.replace("missing on this tree", "not yet (the tree step makes the tree)")
                print(f"    {cmd.name}: {cmd.symbolic()}" + (f"   <- SKIP: {why_skip}" if why_skip else ""))
        return 0


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    if sys.stdout is None or sys.stderr is None:  # pythonw.exe (the scheduled task): no console, the lab's runner.log has the story
        sink = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
        sys.stdout = sys.stdout or sink
        sys.stderr = sys.stderr or sink
    os.environ.update(CHILD_ENV)
    r = Runner(a)
    if a.dry_run:
        return r.dry_run()
    try:
        return r.run()
    except Exception:  # noqa: BLE001
        r.log("CRASH " + traceback.format_exc()[-1500:])
        r.status("crashed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
