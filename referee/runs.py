#!/usr/bin/env python3
"""Whole-run referee batch: play N seeded runs in the game, replay them in the emulator, compare.

    python referee/runs.py --name <name> --runs 10 [--start 0] [--procs 2] [--game <game copy>]
        [--buff 2000] [--stop-after-act 1] [--avoid Unknown,Shop] [--max-decisions 1500]

From the fork's root. Each game process is one run_referee.py session (all its safety checks: known
sts2.dll, no network, scratch profile, real profile untouched) playing a share of the runs
(referee/driver/RunPlay.cs). Outputs: jobs/<name>.out.json (merged, in spec order), <name>.emu.json
(EmuDump run) and the comparison (tools/fidelity/run_compare2.py run), also saved as <name>.summary.txt.
Seeds are drawn from --start (start 1 gives the seeds of referee/jobs/batch1).
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
PY = sys.executable
JOBS = Path("D:/spireline/referee/jobs/runref")
SEED_CHARS = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def seeds(n: int, start: int) -> list[str]:
    rng = random.Random(start)
    return ["".join(rng.choice(SEED_CHARS) for _ in range(10)) for _ in range(n)]


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
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--no-game", action="store_true", help="reuse jobs/<name>.out.json; replay and compare only")
    a = ap.parse_args()
    JOBS.mkdir(parents=True, exist_ok=True)
    out = JOBS / f"{a.name}.out.json"
    t0 = time.time()
    if not a.no_game:
        specs = [{"kind": "run", "seed": s, "character": "IRONCLAD", "ascension": 10, "chooser_seed": a.start + i,
                  "buff_max_hp": a.buff, "avoid": [x for x in a.avoid.split(",") if x],
                  "stop_after_act": a.stop_after_act, "max_decisions": a.max_decisions}
                 for i, s in enumerate(seeds(a.runs, a.start))]
        procs = []
        for p in range(min(a.procs, len(specs))):
            part = specs[p::a.procs]
            job = JOBS / f"{a.name}.p{p}.json"
            job.write_text(json.dumps({"out": str(JOBS / f"{a.name}.p{p}.out.json"), "fast_reset": True,
                                       "no_preload": True, "fast_patches": True, "stall_s": 90, "runs": part}))
            procs.append(subprocess.Popen([PY, str(HERE / "run_referee.py"), "--game", a.game, "--job", str(job),
                                           "--timeout", str(a.timeout), "--below-normal"],
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
            time.sleep(3)  # the first session installs the mod; the second then finds it in place
        reports = [json.loads(_last_json(p)) for p in procs]
        for r in reports:
            print("session:", {k: r.get(k) for k in ("exit", "seconds", "network_connections", "real_profile_touched")})
            if r.get("network_connections") or r.get("real_profile_touched"):
                print("SAFETY CHECK FAILED: stopping")
                return 1
        merged = [None] * len(specs)
        for p in range(len(procs)):
            path = JOBS / f"{a.name}.p{p}.out.json"
            res = json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else []
            for k, r in enumerate(res):
                merged[p + k * a.procs] = r
        merged = [r if r is not None else {"seed": specs[i]["seed"], "error": "no result (session ended)"}
                  for i, r in enumerate(merged)]
        out.write_text(json.dumps({"results": merged}), encoding="utf-8")
    game_s = time.time() - t0
    emu = JOBS / f"{a.name}.emu.json"
    dll = ROOT / "tools/emudump/bin/Release/net9.0/EmuDump.dll"
    with open(emu, "w", encoding="utf-8") as f:
        subprocess.run(["dotnet", str(dll), "run", str(out)], stdout=f, check=True)
    cmp = subprocess.run([PY, "-m", "tools.fidelity.run_compare2", "run", str(out), str(emu)], cwd=ROOT,
                         capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    text = cmp.stdout + cmp.stderr + f"game {game_s:.0f} s for {a.name}\n"
    (JOBS / f"{a.name}.summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return cmp.returncode


def _last_json(p: subprocess.Popen) -> str:
    """run_referee.py prints its report as the last JSON object on stdout."""
    text = p.communicate()[0]
    start = text.rfind("{\n")
    return text[start:] if start >= 0 else "{}"


if __name__ == "__main__":
    sys.exit(main())
