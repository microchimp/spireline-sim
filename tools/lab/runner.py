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
runs short (tools/lab/guards.py). Exit 0 when it stopped for any of those or finished a --trial / --once; 1 on
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

from tools.lab import guards, jobs, mix, plans, prune, winsys  # noqa: E402

JOB_TYPES = ("presort", "fights", "runs", "probes", "mutation", "invariants", *jobs.NEW_KINDS)
ROTATION = ("fights", "fights", "probes", "fights", "mutation", "fights", "invariants")  # after presort, by value
DEFAULTS = {
    "lab": "D:/spireline/referee/lab", "tree": "D:/spireline/fork/wt-lab", "game": "D:/spireline/referee/game-lab",
    "game_src": "D:/spireline/referee/game", "backlog_dir": "D:/spireline/referee/backlog",
    "codex": "D:/spireline/data/spire-codex/runs-v0.107.1.jsonl.gz", "decompiled": "D:/spireline/referee/decompiled",
    "records": "D:/spireline/referee/jobs",
}
FORBIDDEN = ("D:/spireline/src",)  # the reviewed upstream checkout: never a lab, tree or game path
RESUME_HOURS = 24.0
CHILD_ENV = {"PYTHONDONTWRITEBYTECODE": "1", "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_NOLOGO": "1", "PYTHONIOENCODING": "utf-8"}
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
    ap.add_argument("--backlog", default=None, help="backlog json to use (default: the newest backlog-*.json)")
    ap.add_argument("--codex", default=d["codex"])
    ap.add_argument("--decompiled", default=d["decompiled"])
    ap.add_argument("--records", default=d["records"], help="the recorded fights the invariant checks read")
    ap.add_argument("--trial", type=float, default=None, metavar="MINUTES", help="work for this long, then stop and exit 0")
    ap.add_argument("--once", choices=JOB_TYPES, default=None, help="run one job of this type, then exit")
    ap.add_argument("--always-on", action="store_true",
                    help="for the logon task: keyboard and mouse input and LAB/HALT pause the lab instead of ending the runner")
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
    def __init__(self, cmd: plans.Cmd, proc: subprocess.Popen, files: list[Any], t0: float, in_job: bool):
        self.cmd, self.proc, self.files, self.t0, self.in_job = cmd, proc, files, t0, in_job


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
        self.load = guards.LoadMonitor(lambda: self.root_job.cpu_time() if self.root_job else 0, window=a.load_window,
                                       interval=min(5.0, max(0.2, a.poll)), clock=clock)
        self.guards = guards.Guards(self.lab.root, lock, disabled=disabled, idle_s=a.idle_seconds, load_pct=a.load_pct,
                                    load_window=a.load_window, lab_gb=a.lab_gb, free_gb=a.free_gb, grace=a.grace, deadline=deadline,
                                    clock=clock, load=self.load, prune=self.prune_some,
                                    parent_pid=os.getppid() if a.exit_with_parent else None, stay=a.always_on)
        self.next_strategy: str | None = None  # the strategy choose() picked for the job new_run() makes next
        self.defer_sha: str | None = None  # set by choose() when a presort is deferred: the presorted tree's sha the picked job runs on
        self.last_mix_note = ""
        self.last_loop_note = ""
        self.code_digest0 = code_digest()  # the lab sources this process loaded; a later change on disk is not in this process
        self.code_stale_logged = ""
        self._mark = (0.0, 0.0)  # (cpu seconds, clock) at the last accounting of the running job

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
        if rd.get("load") is not None:
            extra = (extra + " " if extra else "") + f"other_cpu={rd['load'][0]}%/{rd['load'][1]}s"
        if self.current and self.current_strategy:
            where += f" strategy={self.current_strategy}"
        line = (f"{utc()} {state} {where} since={self.since} tree={main} jobs_done={sum(c['jobs_done'].values())} "
                f"fights={c['fights']} runs={c.get('runs', 0)} interrupts={intr}" + (f" | {extra}" if extra else ""))
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

    def ctx(self, run: Run | None, job: str, sha: str, params: dict[str, Any]) -> plans.Ctx:
        a = self.a
        name = run.name if run else f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M}-{job}"
        return plans.Ctx(lab=self.lab.root, tree=self.tree, game=self.game, fork=self.fork, py=sys.executable.replace("\\", "/"),
                         sha=sha, job_dir=run.dir if run else self.lab.runs / name, job_name=name, params=params,
                         backlog_dir=Path(a.backlog_dir), backlog=self.newest_backlog(), codex=Path(a.codex),
                         decompiled=Path(a.decompiled), records=Path(a.records), game_src=Path(a.game_src),
                         nuget=Path(os.path.expanduser("~")) / ".nuget" / "packages", skip=self.skip)

    def poll(self) -> guards.Verdict:
        if self.stop_signal:
            return guards.Verdict("signal", self.stop_signal, leave=True)
        return self.guards.poll()

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
            if run is None:
                continue
            if a.once and run.job != a.once:
                continue
            age_h = (time.time() - (d / "job.json").stat().st_mtime) / 3600
            if sha is None or run.data.get("main_sha") not in here or age_h > RESUME_HOURS:
                if not a.dry_run:
                    self.record(run, 3)
                    run.finish(3, "superseded: main moved or the job was idle too long")
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
        for k in range(len(ROTATION)):
            job = ROTATION[(st["rotation"] + k) % len(ROTATION)]
            strategy = None
            if job == "fights":  # the mix says which strategy this slot's job is (a whole run is its own job type)
                job, strategy = self.pick_fights(sha)
                if not job:
                    continue
                why = None
            else:
                why = self.unavailable(job, sha)
            if why is None:
                st["rotation"] = (st["rotation"] + k + 1) % len(ROTATION)
                self.next_strategy = strategy if job == "fights" else None
                return job, "rotation" + (f", {strategy}" if strategy else "")
            self.log(f"{job}: not runnable ({why})")
        if self.kind_ready("selfplay", sha):  # the filler: nothing above it could run
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

    def ladder_top(self, sha: str) -> str:
        """The agent-era kinds that outrank the fights rotation (docs/PLAN.md section 7), highest priority first: score-real (2),
        train (3), selfplay when the loop (selfplay + train + score-real) has not yet used its configured share of the last 7 days' CPU
        (6, `share` in LAB/jobs.json; an empty window means self-play first; all-time `strategy_cpu` is a statistic only), the
        benches (7, nightly, the lab otherwise idle: a running job was resumed before this point and a ready score-real or
        train came first). Presort (1) is decided before, aimed fights (4), runs and rank (5) and the rest (7) by the rotation and the
        mix. A kind that is off or not ready is passed over quietly (its reason is only logged when it was due)."""
        cfg = jobs.load_config(self.lab.root)
        for kind in ("score-real", "train"):
            if jobs.unready(kind, self.lab.root, cfg[kind]) is None and self.kind_ready(kind, sha):
                return kind
        share = float(cfg["selfplay"].get("share") or 0)
        if share > 0 and cfg["selfplay"].get("enabled"):
            loop, total, by = mix.loop_cpu(self.lab.root)  # the loop's share of the last 7 days, not of all time
            note = (f"self-play loop: {100 * loop / total:.1f}% of the last {mix.WINDOW_DAYS} days' CPU "
                    f"({', '.join(f'{k} {by.get(k, 0.0):.0f}' for k in jobs.LOOP_KINDS)} of {total:.0f} min) against its share "
                    f"{100 * share:.0f}%" if total > 0 else
                    f"self-play loop: no CPU in the last {mix.WINDOW_DAYS} days in LAB/yield.tsv: self-play first")
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
        weights, notes = mix.load_mix(self.lab.root, create=not self.a.dry_run)
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
        chosen, skipped = mix.pick(weights, spent, lambda s: self.unavailable("runs" if s == "whole-runs" else "fights", sha, s))
        for s, why in skipped.items():
            self.log(f"{s}: not runnable ({why})")
        if chosen is None:
            return "", None
        self.log(f"mix: {chosen} is next (share wanted/had: {mix.share_table(weights, spent)})")
        return ("runs" if chosen == "whole-runs" else "fights"), chosen

    def unavailable(self, job: str, sha: str, strategy: str | None = None) -> str | None:
        until = self.state["backoff"].get(strategy or job, 0)  # a strategy that keeps failing backs off alone
        if time.time() < until:
            return f"backing off after repeated failures until {utc(until)}"
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
            c = self.ctx(None, job, sha, params)
            return plans.runnable(plans.plan(job, c), c)
        except plans.Unavailable as e:
            return str(e)

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

    def cpu_secs(self) -> float:
        """CPU seconds of everything the runner has started (its Job Object); 0 where there is none (a test)."""
        return self.root_job.cpu_time() / 1e7 if self.root_job else 0.0

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
        try:
            return self._execute(run)
        finally:
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
        streak = st.setdefault("fail_streak", {})
        key = (run.data.get("params") or {}).get("strategy") or run.job  # a strategy that keeps failing backs off alone
        if rc == 0:
            streak[key] = 0
            st["backoff"].pop(key, None)
        elif run.job != "presort":  # a presort has its own per-sha retry
            streak[key] = streak.get(key, 0) + 1
            if streak[key] >= 2:
                st["backoff"][key] = time.time() + min(6 * 3600, 600 * 2 ** (streak[key] - 2))
        c["jobs_done"][run.job] = c["jobs_done"].get(run.job, 0) + (1 if rc == 0 else 0)
        c["active_seconds"] += int(self.clock() - getattr(self, "job_t0", self.clock()))
        self.account(run)
        self.record(run, rc)
        run.finish(rc)
        self.lab.save_state(st)
        self.log(f"job {run.name} finished rc={rc}")
        self.current = None
        self.status("idle")

    def record(self, run: Run, rc: int) -> None:
        """A finished (or closed) job's CPU goes to its strategy's account, and its row to LAB/yield.tsv. Never fails the job."""
        try:
            data = run.data
            strategy = (data.get("params") or {}).get("strategy")
            spent = float(data.get("cpu_secs") or 0) or float(data.get("wall_secs") or 0)
            if strategy:
                cpu = self.state.setdefault("strategy_cpu", {})
                cpu[strategy] = round(float(cpu.get(strategy, 0.0)) + spent, 1)
            groups_ran = any(r.get("name") == "groups" and r.get("status") == "done" for r in data.get("steps", []))
            row = mix.yield_row(run.dir, dict(data, rc=rc), read_json(self.lab.root / "groups.json", None), groups_ran, utc())
            mix.append_yield(self.lab.root, row)
        except Exception:  # noqa: BLE001  (the record is a convenience; the job's own files are what matters)
            self.log("yield not recorded: " + traceback.format_exc()[-400:])

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
                now = self.clock()
                for r in list(running):
                    rc = r.proc.poll()
                    if rc is None and r.cmd.timeout and now - r.t0 > r.cmd.timeout:
                        winsys.kill_tree_by_pid(r.proc.pid)
                        rc = r.proc.wait(timeout=30)
                        rec["cmds"][r.cmd.name]["note"] = f"killed after the {r.cmd.timeout:.0f}s limit"
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
        out = open(ctx.path(cmd.stdout) if cmd.stdout else logs / f"{cmd.name}.log", "wb")  # noqa: SIM115
        files = [out]
        out.write(f"# {cmd.symbolic()}\n# started {utc()}\n".encode())
        out.flush()
        if cmd.stderr:
            p_err = ctx.path(cmd.stderr)
            p_err.parent.mkdir(parents=True, exist_ok=True)
            err = open(p_err, "wb")  # noqa: SIM115
            files.append(err)
        else:
            err = subprocess.STDOUT
        kw: dict[str, Any] = {"cwd": ctx.expand(cmd.cwd), "env": env, "stdout": out, "stderr": err, "stdin": subprocess.DEVNULL}
        if winsys.IS_WIN:
            kw.update(hidden_console_kw())
        in_job = True
        try:
            proc = winsys.suspended_popen(argv, sj, **kw)
        except OSError as e:
            if isinstance(e, FileNotFoundError):
                for f in files:
                    f.close()
                raise
            in_job = False  # could not nest in a job: a plain start, killed by process tree instead
            kw["creationflags"] = kw.get("creationflags", 0) & ~winsys.CREATE_SUSPENDED
            proc = subprocess.Popen(argv, **kw)
        return Running(cmd, proc, files, self.clock(), in_job)

    def finish_cmd(self, run: Run, ctx: plans.Ctx, r: Running, rc: int, rec: dict[str, Any], now: float) -> bool:
        for f in r.files:
            f.close()
        cmd = r.cmd
        (run.dir / "logs" / f"{cmd.name}.rc").write_text(f"{rc}\n")
        ok = rc in cmd.ok_rcs
        entry = rec["cmds"].setdefault(cmd.name, {})
        entry.update({"status": "done" if ok else "failed", "rc": rc, "secs": round(now - r.t0, 1)})
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
        self.log(f"{rec['name']}/{cmd.name}: rc={rc} in {entry['secs']}s" + ("" if ok else " (FAILED)"))
        return not ok

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
        for label, p in (("lab", self.lab.root), ("tree", self.tree), ("game", self.game)):
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

    def one_pass(self) -> int | None:
        """Wait out any pause, pick a job, run it. An exit code when the runner should end, else None."""
        a = self.a
        v = self.gate()
        if v:
            return self.leave(v)
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
            v = self.poll()
            if v.leave:
                return True
            self.heartbeat("idle")
        return False

    def leave(self, v: guards.Verdict) -> int:
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
