#!/usr/bin/env python3
"""One fidelity batch end to end: make fights, play them in the game, replay them in the emulator, compare.

    python referee/batch.py --name <name> --fights 100 [--start 0] [--timeout 5400]

Run from the fork's root. Writes D:/spireline/referee/jobs/<name>.{json,out.json,emu.json,summary.txt}
and prints the summary: fights identical, fights diverging grouped by the first differing field,
and errors. Seeds are deterministic from --start, so a batch can be re-run exactly.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = Path(r"D:\spireline\referee\jobs")
GAME = r"D:\spireline\referee\game"
ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--fights", type=int, default=100)
    ap.add_argument("--start", type=int, default=0, help="seed stream offset")
    ap.add_argument("--timeout", type=float, default=5400)
    a = ap.parse_args()

    rng = random.Random(a.start)
    runs = [{"kind": "combat", "seed": "".join(rng.choice(ALPHABET) for _ in range(10)), "character": "IRONCLAD",
             "ascension": 10, "neow": i % 3, "chooser_seed": a.start * 1000 + i, "max_actions": 150}
            for i in range(a.fights)]
    job, out, emu = (JOBS / f"{a.name}{ext}" for ext in (".json", ".out.json", ".emu.json"))
    job.write_text(json.dumps({"out": str(out).replace("\\", "/"), "runs": runs}), encoding="utf-8")

    session = subprocess.run([sys.executable, str(ROOT / "referee" / "run_referee.py"), "--game", GAME, "--job", str(job),
                              "--timeout", str(a.timeout)], capture_output=True, text=True, check=False)
    report = json.loads(session.stdout[session.stdout.index("{"):]) if "{" in session.stdout else {}
    if not out.exists():
        print("referee produced no output:", session.stdout[-1500:], session.stderr[-1500:])
        return 2
    with open(emu, "w", encoding="utf-8") as f:
        subprocess.run(["dotnet", str(ROOT / "tools/emudump/bin/Release/net9.0/EmuDump.dll"), "fight", str(out)],
                       stdout=f, check=False)
    cmp = subprocess.run([sys.executable, "-m", "tools.fidelity.fight_compare", str(out), str(emu)],
                         capture_output=True, text=True, cwd=ROOT, check=False).stdout

    same, groups, errors = 0, collections.defaultdict(list), []
    current = None
    for line in cmp.splitlines():
        if " steps identical" in line:
            same += 1
        elif "error:" in line and not line.startswith(" "):
            errors.append(line[:300])
        elif "first difference at step" in line:
            current = line
        elif line.startswith("    .") and current:
            field = line.split(":")[0].strip()
            groups[field].append(current[:200])
            current = None
    lines = [f"batch {a.name}: {a.fights} fights, session exit {report.get('exit')}, {report.get('seconds')} s, "
             f"network {report.get('network_connections')}, real profile touched {report.get('real_profile_touched')}",
             f"identical: {same}", f"diverging: {sum(len(v) for v in groups.values())}", f"errors: {len(errors)}"]
    for field, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  first field {field}: {len(items)}  e.g. {items[0]}")
    lines += ["  " + e for e in errors[:10]]
    text = "\n".join(lines)
    (JOBS / f"{a.name}.summary.txt").write_text(text + "\n\n" + cmp, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
