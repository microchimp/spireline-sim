#!/usr/bin/env python3
"""Whole-run referee batch: play N seeded runs in the game, replay them in the emulator, compare.

    python referee/runs.py --name <name> --runs 10 [--start 0] [--procs 2] [--game <game copy>]
        [--buff 2000] [--stop-after-act 1] [--avoid Unknown,Shop] [--max-decisions 1500] [--jobs <dir>] [--old-settle]

From the fork's root. Each game process is one run_referee.py session (all its safety checks: known
sts2.dll, no network, scratch profile, real profile untouched) playing a share of the runs
(referee/driver/RunPlay.cs). Outputs: jobs/<name>.out.json (merged, in spec order), <name>.emu.json
(EmuDump run), the comparison (tools/fidelity/run_compare2.py run) saved as <name>.summary.txt, and the same
as data in <name>.runs.json (one record per run with its first divergence; tools/fidelity/lab_groups.py reads it).
Seeds are drawn from --start (start 1 gives the seeds of referee/jobs/batch1). --jobs puts every output in that
folder instead of jobs/runref (their scratch profiles too).

A session that crashes or times out (a game process can exit 0xC0000005) is logged in the summary, and the
batch goes on with what the other sessions played and what the crashed one wrote before it ended: the runs it
never wrote are marked "no result". The merged results are written before anything is judged, so no exit
loses them. Exit 0: every run compared identical; 1: some differ; 3: a session crashed, the rest was compared;
2: no run was recorded at all; 4: a session's safety check failed (a network connection, the real profile
touched, or no report to show neither): nothing was compared (--no-game compares it deliberately).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import serve  # noqa: E402  (referee/serve.py: job_options, the speed defaults every referee pool shares)

PY = sys.executable
JOBS = Path("D:/spireline/referee/jobs/runref")
SEED_CHARS = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
EXIT_NOTHING, EXIT_CRASHED, EXIT_SAFETY = 2, 3, 4


def seeds(n: int, start: int) -> list[str]:
    rng = random.Random(start)
    return ["".join(rng.choice(SEED_CHARS) for _ in range(10)) for _ in range(n)]


def session_job(out: str, part: list[dict], *, old_settle: bool = False) -> dict:
    """One session's job file: serve.py's speed defaults (serve.job_options: the game's turn-flag settle, awaited tweens
    finished at once, no frame padding, a 5% settle audit, the speed patch sets with no sound and still visuals, fast
    reset, no preload) with the run sessions' stall limit. The speed spike played 4 act-1 runs (1,048 decisions) with
    these and with the old settle and got every decision and snapshot equal, in 56.6 s against 193 s (3.4x).
    old_settle plays with the stock settle and pacing (the A/B). The engine's --fixed-fps is not set here: the spike
    proved it on fights only."""
    return {"out": out, **serve.job_options(None, old_settle=old_settle, stall_s=90), "runs": part}


def merge_results(parts: list[list], n: int, procs: int, specs: list[dict]) -> list[dict]:
    """The sessions' results in spec order (session p played specs p, p+procs, ...); a run a session never wrote,
    or a session that wrote nothing, is marked "no result (session ended)"."""
    merged: list = [None] * n
    for p, res in enumerate(parts):
        for k, r in enumerate(res):
            if p + k * procs < n:
                merged[p + k * procs] = r
    return [r if r is not None else {"seed": specs[i]["seed"], "error": "no result (session ended)"} for i, r in enumerate(merged)]


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--game", default="D:/spireline/referee/game-runref")
    ap.add_argument("--buff", type=int, default=2000)
    ap.add_argument("--stop-after-act", type=int, default=3)
    ap.add_argument("--avoid", default="")
    ap.add_argument("--max-decisions", type=int, default=1500)
    ap.add_argument("--potions", action="store_true", help="drink potions in fights")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--no-game", action="store_true", help="reuse jobs/<name>.out.json; replay and compare only")
    ap.add_argument("--old-settle", action="store_true",
                    help="play with the stock settle and frame pacing (before SPEED_PLAN steps 3-4), 3.4x slower: A/B only")
    ap.add_argument("--jobs", type=Path, default=None, help="write every output here instead of jobs/runref")
    a = ap.parse_args()
    jobs = a.jobs.resolve() if a.jobs else JOBS
    jobs.mkdir(parents=True, exist_ok=True)
    out = jobs / f"{a.name}.out.json"
    summary = jobs / f"{a.name}.summary.txt"
    log: list[str] = []

    def say(line: str) -> None:
        print(line)
        log.append(line)

    def finish(code: int, extra: str = "") -> int:
        summary.write_text(("\n".join(log) + "\n" if log else "") + extra, encoding="utf-8")
        return code

    t0 = time.time()
    failed: list[tuple[int, str]] = []
    unsafe = False
    if not a.no_game:
        specs = [{"kind": "run", "seed": s, "character": "IRONCLAD", "ascension": 10, "chooser_seed": a.start + i,
                  "buff_max_hp": a.buff, "avoid": [x for x in a.avoid.split(",") if x],
                  "stop_after_act": a.stop_after_act, "max_decisions": a.max_decisions, "potions": a.potions}
                 for i, s in enumerate(seeds(a.runs, a.start))]
        procs = []
        for p in range(min(a.procs, len(specs))):
            part = specs[p::a.procs]
            job = jobs / f"{a.name}.p{p}.json"
            job.write_text(json.dumps(session_job(str(jobs / f"{a.name}.p{p}.out.json"), part, old_settle=a.old_settle)))
            # With --jobs the sessions' scratch profiles (saves, game logs) live in the job folder too, where the lab's
            # disk pruning finds them; without it run_referee puts them next to the game copy as it always did.
            where = ["--scratch", str(jobs / "scratch" / f"{a.name}-p{p}")] if a.jobs else []
            procs.append(subprocess.Popen([PY, str(HERE / "run_referee.py"), "--game", a.game, "--job", str(job),
                                           "--timeout", str(a.timeout), "--below-normal", *where],
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
            time.sleep(3)  # the first session installs the mod; the second then finds it in place
        reports = [_report(p) for p in procs]
        # What the sessions wrote is merged and saved first: whatever is wrong below, the runs stay on disk.
        merged = merge_results([_read_results(jobs / f"{a.name}.p{p}.out.json") for p in range(len(procs))], len(specs), a.procs, specs)
        out.write_text(json.dumps({"results": merged}), encoding="utf-8")
        for k, r in enumerate(reports):
            say(f"session {k}: " + str({key: r.get(key) for key in ("exit", "seconds", "network_connections", "wer_reports", "real_profile_touched", "scratch")}))
            problem = session_problem(r)
            if problem and problem.startswith("SAFETY"):
                say(f"session {k}: {problem}")
                unsafe = True
            elif problem:
                failed.append((k, problem))
        if unsafe:
            say("a safety check did not hold: stopping, nothing compared (the merged runs are in " + out.name + ")")
            return finish(EXIT_SAFETY)
        # A crashed or timed-out session is logged and the batch goes on: the runs it wrote before it ended are in the
        # merged file, the others marked "no result"; what the other sessions played is compared as usual.
        for k, problem in failed:
            mine = range(k, len(specs), a.procs)
            say(f"session {k} FAILED, the batch goes on: {problem}; {sum(1 for i in mine if 'decisions' in merged[i])} of its {len(mine)} runs were kept")
    elif not out.exists():
        say(f"{out.name} not found")
        return finish(EXIT_NOTHING)
    game_s = time.time() - t0
    if not any(r and "decisions" in r for r in _read_results(out)):
        say("no run was recorded by any session")
        return finish(EXIT_NOTHING)
    emu = jobs / f"{a.name}.emu.json"
    dll = ROOT / "tools/emudump/bin/Release/net9.0/EmuDump.dll"
    with open(emu, "w", encoding="utf-8") as f:
        subprocess.run(["dotnet", str(dll), "run", str(out)], stdout=f, check=True)
    cmp = subprocess.run([PY, "-m", "tools.fidelity.run_compare2", "run", str(out), str(emu), "--json", str(jobs / f"{a.name}.runs.json")],
                         cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    text = cmp.stdout + cmp.stderr + f"game {game_s:.0f} s for {a.name}\n"
    finish(0, text)
    print(text)
    return EXIT_CRASHED if failed else cmp.returncode


def _read_results(path: Path) -> list:
    """A session's results so far; an empty list when the file is missing or was cut off by a crash."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))["results"]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def _report(p: subprocess.Popen) -> dict:
    """run_referee.py prints its report as the last JSON object on stdout; {} when there is none."""
    text = p.communicate()[0]
    start = text.rfind("{\n")
    try:
        r = json.loads(text[start:]) if start >= 0 else {}
    except ValueError:
        r = {}
    return r if isinstance(r, dict) else {}


def session_problem(r: dict) -> str | None:
    """Why a session's report is not clean, or None. A missing or unreadable report, or one whose
    safety fields are absent, is a safety failure: nothing shows the session stayed offline and off
    the real profile. A non-zero exit (a crash, a timeout) is a failed session."""
    if r.get("network_connections") != [] or r.get("real_profile_touched") is not False:
        return (f"SAFETY CHECK FAILED (network_connections {r.get('network_connections', 'missing')}, "
                f"real_profile_touched {r.get('real_profile_touched', 'missing')})")
    if r.get("exit") != 0:
        return f"SESSION FAILED: exit {r.get('exit', 'missing')}, scratch {r.get('scratch')}"
    return None


if __name__ == "__main__":
    sys.exit(main())
