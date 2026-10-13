"""Compare a run start in the referee (the game's own code) with the emulator's.

    python -m tools.fidelity.run_compare <referee out.json> <emudump.json>

Compares the act list, every run and player RNG stream's call counter, and the act-1 map
(coordinates, room types, edges). Exit code 0 when everything matches.
"""

from __future__ import annotations

import argparse
import json
import sys

# Emulator stream names that differ from the game's.
RENAME = {"CombatOrbs": "CombatOrbGeneration"}


def counters(block: dict) -> dict:
    return {RENAME.get(k, k): v["counter"] for k, v in block.items()}


def map_set(points: list) -> dict:
    return {(p["col"], p["row"]): (p["type"], tuple(map(tuple, p["children"]))) for p in points}


def compare(ref: dict, emu: dict) -> list[str]:
    out = []
    if ref["seed"] != emu["seed"]:
        out.append(f"seed: game {ref['seed']} emulator {emu['seed']}")
    if ref["acts"] != emu["acts"]:
        out.append(f"acts: game {ref['acts']} emulator {emu['acts']}")
    for key in ("rng", "player_rng"):
        g, e = counters(ref[key]), counters(emu[key])
        for name in sorted(set(g) | set(e)):
            if g.get(name) != e.get(name):
                out.append(f"{key}.{name}: game {g.get(name)} emulator {e.get(name)}")
    gm, em = map_set(ref["map"]), map_set(emu["map"])
    for coord in sorted(set(gm) | set(em), key=lambda c: (c[1], c[0])):
        if gm.get(coord) != em.get(coord):
            out.append(f"map {coord}: game {gm.get(coord)} emulator {em.get(coord)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("referee")
    ap.add_argument("emulator")
    a = ap.parse_args()
    refs = json.load(open(a.referee, encoding="utf-8"))["results"]
    emus = json.load(open(a.emulator, encoding="utf-8"))["results"]
    bad = 0
    for ref, emu in zip(refs, emus):
        if "error" in ref:
            print("referee error:", ref["error"][:2000])
            return 2
        diffs = compare(ref, emu)
        bad += bool(diffs)
        if diffs:
            print(f"{ref['seed']}: {len(diffs)} differences")
            for d in diffs[:20]:
                print("  " + d)
    print(f"{len(refs)} runs compared, {bad} with differences")
    return 0 if not bad and len(refs) == len(emus) else 1


if __name__ == "__main__":
    sys.exit(main())
