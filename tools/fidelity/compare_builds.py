"""Replay every distinct recorded fight with TWO builds of the emulator and compare them.

A fix to the engine is judged against the game by the fights it changes, so this answers "what did
this branch do to everything we have already recorded" in about a minute: how many fights are
identical to the game's with each build, which ones flipped either way, and how many replays
changed at all (including after their first divergence, which the plain comparer cannot judge).

    python -m tools.fidelity.compare_builds --base <EmuDump.dll of the old build> \
        --new <EmuDump.dll of the new build> [--jobs "D:/spireline/referee/jobs/*.out.json"] \
        [--work <scratch dir>] [--procs 4] [--list-changed]

Build the old one from a clean worktree of the old commit
(`cd tools/emudump && dotnet build -c Release --source "$HOME/.nuget/packages"`), never in place.
Recordings with an error or no steps are skipped; fights are de-duplicated on seed, choices and
opening piles, because the job files repeat the same fights across shards and waves.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from .coverage import first_divergence
from .fight_compare import names

ROOT = Path(__file__).resolve().parents[2]


def distinct_fights(pattern: str) -> list:
    seen: set = set()
    fights = []
    for path in sorted(glob.glob(pattern)):
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for r in doc["results"] if isinstance(doc, dict) else doc:
            if not isinstance(r, dict) or not r.get("steps") or r.get("error"):
                continue
            first = r["steps"][0]["state"]
            key = hashlib.md5(json.dumps([r["seed"], r.get("choices"), first["hand"], first["draw"]], sort_keys=True).encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                fights.append(r)
    return fights


def replay(dll: str, tag: str, fights: list, work: Path, procs: int) -> list:
    """EmuDump `fight` over `fights`, split across processes (one emulator per process)."""
    size = -(-len(fights) // procs)
    parts = [fights[i:i + size] for i in range(0, len(fights), size)]
    running = []
    for i, part in enumerate(parts):
        src, dst = work / f"{tag}.{i}.in.json", work / f"{tag}.{i}.out.json"
        src.write_text(json.dumps({"results": part}), encoding="utf-8")
        running.append((dst, subprocess.Popen(["dotnet", dll, "fight", str(src)], stdout=open(dst, "w", encoding="utf-8"))))
    out: list = []
    for (dst, proc), part in zip(running, parts):
        proc.wait()
        doc = json.loads(dst.read_text(encoding="utf-8")) if dst.stat().st_size else {"results": []}
        got = doc.get("results", [])
        out += got + [{"error": "no result"}] * (len(part) - len(got))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="EmuDump.dll of the old build")
    ap.add_argument("--new", required=True, help="EmuDump.dll of the new build")
    ap.add_argument("--jobs", default="D:/spireline/referee/jobs/*.out.json")
    ap.add_argument("--work", default=None, help="scratch directory (default: a temporary one)")
    ap.add_argument("--procs", type=int, default=4, help="processes per build")
    ap.add_argument("--list-changed", action="store_true", help="list every fight whose replay changed")
    a = ap.parse_args()

    work = Path(a.work or tempfile.mkdtemp(prefix="compare_builds_"))
    work.mkdir(parents=True, exist_ok=True)
    nm = names(json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8")))
    fights = distinct_fights(a.jobs)
    print(f"distinct recorded fights: {len(fights)}")
    with cf.ThreadPoolExecutor(2) as pool:
        base_f = pool.submit(replay, a.base, "base", fights, work, a.procs)
        new_f = pool.submit(replay, a.new, "new", fights, work, a.procs)
        base, new = base_f.result(), new_f.result()

    ident_base = ident_new = changed = compared = 0
    flipped_bad, flipped_good, moved = [], [], []
    for g, b, n in zip(fights, base, new):
        if b.get("error") or n.get("error"):
            continue
        compared += 1
        db, dn = first_divergence(g, b, nm), first_divergence(g, n, nm)
        ident_base += db is None
        ident_new += dn is None
        if json.dumps(b.get("steps")) != json.dumps(n.get("steps")):
            changed += 1
            moved.append((g["seed"], db, dn))
        if db is None and dn is not None:
            flipped_bad.append((g["seed"], dn))
        if db is not None and dn is None:
            flipped_good.append((g["seed"], db))
    print(f"compared {compared}: identical to the game with the old build {ident_base}, with the new build {ident_new}")
    print(f"replays that changed at all: {changed}")
    print(f"now diverging (was identical): {len(flipped_bad)} {flipped_bad[:10]}")
    print(f"now identical (was diverging): {len(flipped_good)} {flipped_good[:10]}")
    if a.list_changed:
        for seed, db, dn in moved:
            print(f"  {seed}: first divergence {db} -> {dn}")
    print(f"replays kept in {work}")
    return 1 if flipped_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
