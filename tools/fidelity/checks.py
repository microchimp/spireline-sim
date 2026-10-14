"""Oracle-free fidelity checks over many fights: what must hold whatever the game does.

  determinism   the same config and actions give the same full state hash at every step
                (review condition E11).
  clone         a faithful clone (no resample) taken mid-fight continues exactly like the
                original under the same actions.
  resample      a resampled clone changes hidden state only: everything the player can see
                (neutral state minus draw order and RNG counters) stays identical, the draw
                pile keeps the same cards, and the known top and bottom of the pile hold.
                RNG counters are hidden and may move: resampling reshuffles the unknown middle
                of the draw pile on the re-seeded Shuffle stream.
  snapshot      GetStateJson -> LoadCombatSnapshot -> GetStateJson round-trips.

Each encounter runs in its own child process (one emulator per process, review condition
E10), so a native abort is recorded against its encounter instead of ending the run.

    python -m tools.fidelity.checks --emu <emulator checkout> --encounters all --seeds 4 \
        --json results.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .canon import digest
from .diff import diff, first_divergence
from .trace import Config, Emu, record, replay


def _neutral_visible(doc: dict) -> dict:
    vis = json.loads(json.dumps(doc["neutral"]))
    vis["piles"]["draw"] = sorted(map(tuple, vis["piles"]["draw"]))  # order is hidden, contents are not
    del vis["rng"]  # hidden, and resampling reshuffles on the Shuffle stream
    return vis


def check_encounter(emu: Emu, encounter: int, seeds: int, ascension: int, base_seed: int) -> dict:
    out = {"encounter": encounter, "name": emu.encounter_names.get(encounter, str(encounter)),
           "fights": 0, "steps": 0, "failures": [], "rng_used": Counter()}
    native, cm = emu.native, emu.cm
    for k in range(seeds):
        cfg = Config(seed=base_seed + k, encounter=encounter, ascension=ascension, chooser_seed=1000 + k)
        try:
            first = record(emu, cfg)
        except Exception as exc:  # noqa: BLE001  (Python-level refusal, e.g. an encounter that needs a floor)
            out["failures"].append({"check": "open", "seed": cfg.seed, "error": repr(exc)})
            continue
        out["fights"] += 1
        out["steps"] += len(first["steps"]) - 1
        last = first["steps"][-1]
        for name, n in last["rng"].items():
            if n > 0:
                out["rng_used"][name] += 1

        # determinism
        second = replay(emu, first)
        i = first_divergence(first["steps"], second["steps"], key="full")
        if i is not None:
            out["failures"].append({"check": "determinism", "seed": cfg.seed, "step": i})

        # clone and resample, at one step chosen per fight
        actions = [s["action"] for s in first["steps"][1:]]
        if len(actions) >= 2:
            cut = random.Random(cfg.seed).randrange(1, len(actions))
            combat = emu.open(cfg)
            try:
                for a in actions[:cut]:
                    combat.step(a)
                before = emu.state(combat)
                clone = combat.fork(None)
                resampled = combat.fork(987_654 + k)
                try:
                    rs = emu.state(resampled)
                    d = diff(_neutral_visible(before), _neutral_visible(rs), limit=5)
                    if d:
                        out["failures"].append({"check": "resample-visible", "seed": cfg.seed, "step": cut,
                                                "diff": [list(map(str, x)) for x in d]})
                    try:
                        snap_a = native.get_state_json(combat.handle)["piles"]["draw"]
                        snap_b = native.get_state_json(resampled.handle)["piles"]["draw"]
                    except native.ExportRefused as error:
                        # A step stopped on a screen its document would not replay to: a
                        # finding of its own, and the known ends go unchecked at this cut.
                        out["failures"].append({"check": "export-refused", "seed": cfg.seed, "step": cut,
                                                "why": str(error)})
                    else:
                        if snap_a["known_top"] != snap_b["known_top"] or snap_a["known_bottom"] != snap_b["known_bottom"]:
                            out["failures"].append({"check": "resample-known-ends", "seed": cfg.seed, "step": cut})
                    for a in actions[cut:]:
                        if combat.done:
                            break
                        combat.step(a)
                        clone.step(a)
                        if digest(emu.state(combat)["full"]) != digest(emu.state(clone)["full"]):
                            out["failures"].append({"check": "clone", "seed": cfg.seed, "step": cut})
                            break
                finally:
                    clone.close()
                    resampled.close()
            finally:
                combat.close()

        # snapshot round trip at the opening
        combat = emu.open(cfg)
        try:
            doc = native.get_state_json(combat.handle)
            h = native.load_combat_snapshot(doc, cfg.seed)
            try:
                again = native.get_state_json(h)
                d = diff(doc, again, limit=5)
                if d:
                    out["failures"].append({"check": "snapshot", "seed": cfg.seed,
                                            "diff": [list(map(str, x)) for x in d]})
            finally:
                native.destroy(h)
        finally:
            combat.close()
    out["rng_used"] = dict(out["rng_used"])
    return out


def _worker(args: argparse.Namespace) -> int:
    emu = Emu(args.emu)
    res = check_encounter(emu, args.encounter, args.seeds, args.ascension, args.base_seed)
    print(json.dumps(res))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--emu", default=os.environ.get("STS2_EMU"), help="emulator checkout with out/ built")
    ap.add_argument("--encounters", default="act1", help="act1, all, or comma-separated ids")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=7000)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--json", default=None)
    ap.add_argument("--encounter", type=int, default=None, help=argparse.SUPPRESS)  # worker mode
    args = ap.parse_args(argv)
    if not args.emu:
        raise SystemExit("set --emu or STS2_EMU")
    if args.encounter is not None:
        return _worker(args)

    emu = Emu(args.emu)
    if args.encounters == "all":
        ids = sorted(emu.encounter_names)
    elif args.encounters == "act1":
        ids = list(emu.cm.OVERGROWTH_WEAK) + list(emu.cm.OVERGROWTH_NORMAL)
    else:
        ids = [int(x) for x in args.encounters.split(",")]
    del emu  # the parent runs no emulator; each encounter gets its own process

    def run(eid: int) -> dict:
        cmd = [sys.executable, "-m", "tools.fidelity.checks", "--emu", args.emu, "--encounter", str(eid),
               "--seeds", str(args.seeds), "--ascension", str(args.ascension), "--base-seed", str(args.base_seed)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout,
                               cwd=str(Path(__file__).resolve().parents[2]), check=False)
        except subprocess.TimeoutExpired:
            return {"encounter": eid, "crash": "timeout"}
        if p.returncode != 0 or not p.stdout.strip():
            return {"encounter": eid, "crash": f"exit {p.returncode}", "stderr": p.stderr[-2000:]}
        return json.loads(p.stdout.strip().splitlines()[-1])

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:  # threads only wait on child processes
        results = list(pool.map(run, ids))
    summary = {
        "encounters": len(results),
        "crashed": [r["encounter"] for r in results if "crash" in r],
        "fights": sum(r.get("fights", 0) for r in results),
        "steps": sum(r.get("steps", 0) for r in results),
        "failures_by_check": dict(Counter(f["check"] for r in results for f in r.get("failures", []))),
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "results": results}, indent=1), encoding="utf-8")
    return 0 if not summary["crashed"] and not summary["failures_by_check"] else 1


if __name__ == "__main__":
    sys.exit(main())
