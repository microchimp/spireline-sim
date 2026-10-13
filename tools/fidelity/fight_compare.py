"""Compare a fight the referee played in the game with the emulator's replay of it, action by action.

    python -m tools.fidelity.fight_compare <referee out.json> <emudump fight.json> --id-map data/id_map.json

Reports the first diverging step per fight with the differing fields: player HP, block, energy
and powers; hand, draw (in order), discard and exhaust piles; living enemies' HP, block and
powers; and each combat RNG stream's call count.
"""

from __future__ import annotations

import argparse
import json
import re
import sys


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).upper()


def power(name: str) -> str:
    """Game power ids end in _POWER; the emulator's BuffId names do not."""
    n = snake(name)
    return n[: -len("_POWER")] if n.endswith("_POWER") else n


def names(id_map: dict) -> dict:
    return {kind: {int(v): snake(k) for k, v in table.items()} for kind, table in id_map.items()}


# Representation, not rules: the emulator keeps a stun as a STUNNED buff, the game as the
# monster's next move (CreatureCmd.Stun). Neither side's power list is compared on it.
NOT_POWERS = {"STUNNED"}


def emu_view(st: dict, nm: dict) -> dict:
    if "snapshot" not in st or not st["in_combat"]:
        return {"in_combat": False}
    s = st["snapshot"]
    card = lambda c: [nm["cards"].get(c["def_id"], c["def_id"]), bool(c["upgraded"])]  # noqa: E731
    powers = lambda bs: sorted([power(b["name"]), b["amount"]] for b in bs if power(b["name"]) not in NOT_POWERS)  # noqa: E731
    p = s["player"]
    return {
        "in_combat": st["in_combat"],
        "player": {"hp": p["hp"], "max_hp": p["max_hp"], "block": p["block"], "energy": p["energy"],
                   "powers": powers(p["buffs"])},
        "hand": [card(c) for c in s["piles"]["hand"]],
        "draw": [[nm["cards"].get(d, d), bool(u)] for d, u in st["draw_order"]],
        "discard": [card(c) for c in s["piles"]["discard"]],
        "exhaust": [card(c) for c in s["piles"]["exhaust"]],
        "enemies": [{"name": nm["enemies"].get(e["def_id"], e["def_id"]), "hp": e["hp"], "block": e["block"],
                     "powers": powers(e["buffs"])} for e in s["enemies"] if e["hp"] > 0],
        "rng": st["rng"],
    }


def game_view(st: dict) -> dict:
    if not st.get("in_combat"):
        return {"in_combat": False}
    p = st["player"]
    return {
        "in_combat": True,
        "player": {"hp": p["hp"], "max_hp": p["max_hp"], "block": p["block"], "energy": p["energy"],
                   "powers": sorted([power(n), a] for n, a in p["powers"])},
        "hand": st["hand"], "draw": st["draw"], "discard": st["discard"], "exhaust": st["exhaust"],
        "enemies": [{"name": e["name"], "hp": e["hp"], "block": e["block"], "powers": sorted([power(n), a] for n, a in e["powers"])}
                    for e in st["enemies"] if e["hp"] > 0],
        "rng": {k: v["counter"] for k, v in st["rng"].items() if k in (
            "Shuffle", "CombatTargets", "MonsterAi", "CombatCardSelection", "CombatCardGeneration",
            "CombatEnergyCosts", "CombatPotionGeneration", "CombatOrbGeneration", "Niche")},
    }


def diff(a, b, path="") -> list:
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b)) for d in diff(a.get(k), b.get(k), f"{path}.{k}")]
    return [] if a == b else [(path, a, b)]


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("referee")
    ap.add_argument("emulator")
    ap.add_argument("--id-map", default="data/id_map.json")
    a = ap.parse_args()
    nm = names(json.load(open(a.id_map, encoding="utf-8")))
    games = json.load(open(a.referee, encoding="utf-8"))["results"]
    emus = json.load(open(a.emulator, encoding="utf-8"))["results"]
    bad = 0
    for g, e in zip(games, emus):
        for side, doc in (("game", g), ("emulator", e)):
            if doc.get("error"):
                print(f"{g.get('seed')}: {side} error: {doc['error'][:400]}")
        steps = list(zip(g.get("steps", []), e.get("steps", [])))
        first = None
        for i, (gs, es) in enumerate(steps):
            d = diff(game_view(gs["state"]), emu_view(es["state"], nm))
            if d:
                first = (i, gs["action"], d)
                break
        if first is None:
            print(f"{g['seed']}: {len(steps)} steps identical (game {len(g.get('steps', []))}, emulator {len(e.get('steps', []))})")
        else:
            bad += 1
            i, act, d = first
            print(f"{g['seed']}: first difference at step {i} after {act}")
            for path, x, y in d[:12]:
                print(f"    {path}: game {x}  emulator {y}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
