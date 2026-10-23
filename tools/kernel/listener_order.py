"""Does the kernel's listener list over the old state (src/Sts2Emulator/Core2/OldState.cs) keep the game's order?

    python -m tools.kernel.listener_order [--decompiled DIR] [--dll EmuDump.dll] [--work DIR] [--json OUT]

Builds every trigger's scenario of ``tools/fidelity/hook_order.py`` (both listener orders), has ``EmuDump
kernel-order`` list each scenario's listeners as the kernel enumerates them, and checks two things against
``hook_table.game_order_key`` (the rule a trace of the game's own listener lists agreed with on every dispatch):

1. ``sorted``: the kernel's list, keyed by each entry's own kind, creature, index and pile, never goes down;
2. ``pairs``: for every two of the scenario's listeners the kernel lists, the kernel's order is the order of their
   game positions (hook_order.scenario's ``positions``).

A scenario listener the kernel does not list is counted by reason (a power the BuffId rule leaves out or folds, a
potion beyond the slots). Exit 1 on any disagreement or harness error.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.fidelity import hook_order as ho
from tools.fidelity import hook_table as ht

ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "tools/emudump/bin/Release/net9.0/EmuDump.dll"
SCALING = {"scaling": "singleton"}


def scenarios(decompiled: str) -> tuple[list[dict], dict]:
    engine = ht.Engine.load(ht.ENGINE)
    res = ht.check(decompiled, engine, ht.load_json(ht.DISPATCH), ht.load_json(ht.UNMODELLED), ht.read_open(ht.OPEN))
    rows = [ht.Row(**r) for r in res["rows"]]
    out, positions = [], {}
    for trig, spec in ho.TRIGGERS.items():
        kinds = {a["do"] for a in spec["actions"]}
        avoid = frozenset().union(*(ho.AVOID.get(k, frozenset()) for k in kinds))
        ls = ho.listeners_for(engine, rows, set(spec["hooks"]) - avoid, avoid)
        for rev in (False, True):
            sc, pos = ho.scenario(engine, trig, ls, rev)
            out.append(sc)
            positions[sc["name"]] = pos
    return out, positions


def key(e: dict) -> tuple:
    kind = SCALING.get(e["kind"], e["kind"])
    d = {"kind": kind, "creature": e["creature"], "index": e["index"]}
    if e.get("pile"):
        d["pile"] = e["pile"]
    return ht.game_order_key(d)


def find(lkey: str, listeners: list[dict], pos: dict, player_offset: int = 0) -> int | None:
    """The kernel list position of one scenario listener (``kind:emu:side``), or None. A player power is the one the
    scenario put on (at or after ``player_offset``); an enemy power sits on the scenario's one enemy, whatever
    creature number pets (Osty) give it."""
    kind, emu, side = lkey.split(":")
    for i, e in enumerate(listeners):
        if kind == "power" and e["kind"] == "power" and e["buff"] == emu and (
                e.get("enemy") == 0 if side == "enemy" else e["creature"] == 0 and e["index"] >= player_offset):
            return i
        if kind in ("relic", "potion") and e["kind"] == kind and str(e["model"]) == emu \
                and (kind == "relic" or e["index"] == pos["index"]):
            return i
        if kind == "card" and e["kind"] == "card" and str(e["model"]) == emu and e["pile"] == "hand" \
                and e["index"] == pos["index"]:
            return i
        if kind == "enchantment" and e["kind"] == "enchantment" and e["enchantment"] == emu and e["pile"] == "hand" \
                and e["index"] == pos["index"]:
            return i
    return None


def judge(scs: list[dict], positions: dict, results: list[dict]) -> dict:
    out = {"scenarios": 0, "listeners": 0, "sorted_disagreements": 0, "pairs": 0, "pair_disagreements": 0,
           "missing": {}, "errors": {}, "examples": []}
    for res in results:
        name = res["name"]
        if res.get("error"):
            out["errors"][name] = res["error"]
            continue
        out["scenarios"] += 1
        ls = res["listeners"]
        out["listeners"] += len(ls)
        for a, b in zip(ls, ls[1:]):
            if key(b) < key(a):
                out["sorted_disagreements"] += 1
                out["examples"].append(f"{name}: {a} before {b}")
        placed = {}
        for lkey, pos in positions[name].items():
            i = find(lkey, ls, pos, res.get("player_offset", 0))
            if i is None:
                kind, emu, side = lkey.split(":")
                buffs = res["buffs"]["enemy" if side == "enemy" else "player"]
                why = ("power not a game power listener (engine-only, folded or a piece)" if kind == "power" and emu in buffs
                       else "not in the scenario's state" if kind != "power" else "buff not in the state")
                out["missing"].setdefault(why, []).append(f"{name} {lkey}")
                continue
            placed[lkey] = i
        for x, y in itertools.combinations(sorted(placed), 2):
            gx, gy = ht.game_order_key(positions[name][x]), ht.game_order_key(positions[name][y])
            if gx == gy:
                continue
            out["pairs"] += 1
            if (gx < gy) != (placed[x] < placed[y]):
                out["pair_disagreements"] += 1
                out["examples"].append(f"{name}: {x} vs {y}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--decompiled", default=ht.DEFAULT_DECOMPILED)
    ap.add_argument("--dll", default=str(DLL))
    ap.add_argument("--work", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    scs, positions = scenarios(a.decompiled)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(a.work or tmp)
        work.mkdir(parents=True, exist_ok=True)
        sp, op = work / "kernel_order_in.json", work / "kernel_order_out.json"
        sp.write_text(json.dumps({"scenarios": scs}), encoding="utf-8")
        kw = {"creationflags": ho.BELOW_NORMAL} if os.name == "nt" else {}
        r = subprocess.run(["dotnet", a.dll, "kernel-order", str(sp), str(op)], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, env=ho.dotnet_env(), timeout=600, **kw)
        if r.returncode != 0:
            print(f"EmuDump kernel-order failed ({r.returncode}): {r.stdout[-2000:]}")
            return 1
        results = json.loads(op.read_text(encoding="utf-8"))["scenarios"]
    out = judge(scs, positions, results)
    print(f"kernel listener order: {out['scenarios']} scenarios, {out['listeners']} listeners listed; "
          f"sorted by game_order_key: {out['sorted_disagreements']} disagreements; "
          f"{out['pairs']} scenario pairs: {out['pair_disagreements']} disagreements")
    for why, xs in out["missing"].items():
        print(f"  not listed ({why}): {len(xs)}, e.g. {'; '.join(xs[:4])}")
    for name, err in out["errors"].items():
        print(f"  {name}: error {err}")
    for ex in out["examples"][:20]:
        print("  " + ex)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    bad = out["sorted_disagreements"] or out["pair_disagreements"] or out["errors"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
