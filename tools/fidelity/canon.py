"""Canonical combat state: one engine-neutral document per position, and its hash.

Two levels, kept apart on purpose:

  neutral   what the GAME also has, named the way the game names it (model class names from
            data/id_map.json, the game's RNG stream names). This is what a referee running the
            game's own code is compared on.
  full      neutral plus the emulator's own bookkeeping (relic counters, AI scratch fields,
            per-card flags). Only emulator-vs-emulator checks (determinism, clones) use it.

The draw pile is read in TRUE order through `Sts2_GetPile(0)`, and RNG counters through
`Sts2_RngCallCount`. Both are banned outside verification code (review condition E12); this
module is verification code and must never be imported by search or agent code.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

# Stream ids as Sts2_RngCallCount numbers them (NativeExports.cs: 0 Shuffle .. 7 MonsterAi),
# under the game's own stream names.
RNG_STREAMS = (
    "Shuffle",
    "CombatTargets",
    "CombatCardSelection",
    "CombatCardGeneration",
    "CombatEnergyCosts",
    "CombatOrbGeneration",
    "Niche",
    "MonsterAi",
)
SCHEMA = "spireline-fidelity-combat/0"


@lru_cache(maxsize=1)
def _names(id_map_path: str) -> dict[str, dict[int, str]]:
    raw = json.loads(Path(id_map_path).read_text(encoding="utf-8"))
    return {kind: {int(v): k for k, v in table.items()} for kind, table in raw.items()}


def _name(names: dict, kind: str, value: int) -> str:
    if value in (None, 0, -1):
        return "" if value in (None, 0) else "-"
    return names.get(kind, {}).get(int(value), f"?{kind}:{value}")


def _buffs(names: dict, buffs: list) -> list:
    return [[_name(names, "powers", b["id"]), b["amount"]] for b in buffs]


def _card(names: dict, card: dict) -> list:
    """(class name, upgraded, enchantment, enchant amount): the identity the game shows."""
    return [_name(names, "cards", card["def_id"]), bool(card["upgraded"]), card.get("enchantment", 0),
            card.get("enchant_amount", 0)]


def canonical(handle: int, native, id_map_path: str, *, done: bool = False, won: bool | None = None) -> dict:
    """Both levels for one combat handle. `native` is the emulator's sts2_gym.native module."""
    names = _names(id_map_path)
    # The save document: verification code (this reads the true pile and the RNG counts too),
    # and the one export a step stopped on a screen never stops writing. The fields read here
    # are the agent's document's; only its stopped_step block, unread, adds the true world.
    snap = native.get_save_json(handle)
    player = snap["player"]
    piles = snap["piles"]
    true_draw = [[_name(names, "cards", cid), bool(up)] for cid, up in native.get_pile(handle, native.PILE_DRAW)]
    neutral = {
        "schema": SCHEMA,
        "turn": snap["turn"],
        "done": bool(done),
        "won": won,
        "player": {
            "hp": player["hp"], "max_hp": player["max_hp"], "block": player["block"],
            "energy": player["energy"], "max_energy": player["max_energy"], "gold": player["gold"],
            "stars": player["stars"], "osty_hp": player["osty_hp"], "orb_capacity": player["orb_capacity"],
            "orbs": player["orbs"], "powers": _buffs(names, player["buffs"]),
        },
        "relics": [_name(names, "relics", r["id"]) for r in snap["relics"]],
        "potions": [_name(names, "potions", p) for p in snap["potions"]["slots"][: snap["potions"]["max_slots"]]],
        "piles": {
            "hand": [_card(names, c) for c in piles["hand"]],
            "draw": true_draw,
            "discard": [_card(names, c) for c in piles["discard"]],
            "exhaust": [_card(names, c) for c in piles["exhaust"]],
        },
        "enemies": [
            {"name": _name(names, "enemies", e["def_id"]), "slot": e["slot"], "hp": e["hp"], "max_hp": e["max_hp"],
             "block": e["block"], "powers": _buffs(names, e["buffs"]), "intent": e["intent"]}
            for e in snap["enemies"]
        ],
        "pending_selection": snap.get("pending_selection"),
        "rng": {name: native.rng_call_count(handle, i) for i, name in enumerate(RNG_STREAMS)},
    }
    full = {
        "neutral": neutral,
        "relic_counters": [[_name(names, "relics", r["id"]), r["counter"]] for r in snap["relics"]],
        "counters": snap["counters"],
        "enemy_ai": [e["ai"] for e in snap["enemies"]],
        "cards": {k: piles[k] for k in ("hand", "discard", "exhaust")},
        "draw_detail": piles["draw"],
        "incoming": snap["incoming"],
        "player_extra": {k: v for k, v in player.items() if k not in neutral["player"] and k != "buffs"},
    }
    return {"neutral": neutral, "full": full}


def digest(doc) -> str:
    """sha256 of sorted-key, compact JSON. Hashing reads state only; it never steps an engine."""
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()
