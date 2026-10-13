"""Compare whole runs: the game's run snapshots (referee) with the emulator's, decision by decision.

    python -m tools.fidelity.run_compare2 start <referee out.json> <EmuDump runstart json>
    python -m tools.fidelity.run_compare2 run <referee out.json> <EmuDump run json> [-v]

`start`: the run-start save of each run (any referee result that carries "save") against
`EmuDump runstart`: all three acts' room sets, both relic grab bags, the odds, every RNG counter.
`run`: each recorded run decision (referee/driver/RunPlay.cs) against the emulator's replay of the same
choices (tools/emudump/RunReplay.cs); prints, per run, the first decision whose snapshot or offer differs
and the fields that differ there, then a summary by component. Exit 0 when everything matches.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

from tools.fidelity import run_snapshot as rs


def start(ref_path: str, emu_path: str) -> int:
    refs = [r for r in json.load(open(ref_path, encoding="utf-8"))["results"] if "save" in r]
    emus = {e["seed"]: e for e in json.load(open(emu_path, encoding="utf-8"))["results"]}
    bad = 0
    for r in refs:
        g = rs.from_save(r["save"])
        e = emus.get(r["seed"])
        if e is None:
            print(f"{r['seed']}: no emulator result")
            bad += 1
            continue
        diffs = rs.compare(g, e["snap"])
        if diffs:
            bad += 1
            print(f"{r['seed']}: {len(diffs)} differences")
            for path, gv, ev in diffs[:8]:
                print(f"  {path}: game {rs.short(gv)} emulator {rs.short(ev)}")
    print(f"{len(refs)} run starts compared, {bad} with differences")
    return 0 if not bad and refs else 1


def component(decision: dict | None, path: str) -> str:
    """Which part of the run a divergence belongs to: the screen it happened on, or the field."""
    if decision is None:
        return "run_start"
    return decision.get("screen", "?")


def run(ref_path: str, emu_path: str, verbose: bool) -> int:
    refs = json.load(open(ref_path, encoding="utf-8"))["results"]
    emus = json.load(open(emu_path, encoding="utf-8"))["results"]
    by_comp: collections.Counter = collections.Counter()
    bad = 0
    total_dec = 0
    for ref, emu in zip(refs, emus):
        seed = ref.get("seed")
        if ref.get("kind_run") is None and "decisions" not in ref:
            continue
        decisions = ref.get("decisions") or []
        edec = emu.get("decisions") or []
        total_dec += len(decisions)
        first = None
        for i, d in enumerate(decisions):
            if i >= len(edec):
                first = (i, d, [("replay", "decision recorded", emu.get("error", "emulator stopped"))])
                break
            ed = edec[i]
            diffs = []
            if ed.get("error"):
                diffs.append(("replay", d.get("choice"), ed["error"]))
            if d.get("offer") is not None and ed.get("offer") is not None and d["offer"] != ed["offer"]:
                diffs.append(("offer", d["offer"], ed["offer"]))
            if not diffs and d.get("snap") is not None and ed.get("snap") is not None:
                diffs = rs.compare(rs.from_save(d["snap"]) if "players" in d["snap"] else d["snap"], ed["snap"])
            if diffs:
                first = (i, d, diffs)
                break
        n = len(decisions)
        end = ref.get("end") or ref.get("error", "")[:80]
        if first is None:
            print(f"{seed}: {n} decisions identical (game end: {end})")
            continue
        bad += 1
        i, d, diffs = first
        comp = component(d, diffs[0][0])
        by_comp[comp] += 1
        print(f"{seed}: first divergence at decision {i}/{n} ({comp}, floor {d.get('floor')}, "
              f"choice {rs.short(d.get('choice'), 80)}); game end: {end}")
        for path, gv, ev in diffs[: (20 if verbose else 4)]:
            print(f"  {path}: game {rs.short(gv)} emulator {rs.short(ev)}")
    print(f"{len(refs)} runs, {total_dec} decisions; {bad} runs diverge; first divergence by component: "
          + (", ".join(f"{k} {v}" for k, v in by_comp.most_common()) or "none"))
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("mode", choices=["start", "run"])
    ap.add_argument("referee")
    ap.add_argument("emulator")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    if a.mode == "start":
        return start(a.referee, a.emulator)
    return run(a.referee, a.emulator, a.v)


if __name__ == "__main__":
    sys.exit(main())
