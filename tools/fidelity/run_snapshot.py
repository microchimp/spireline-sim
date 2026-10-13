"""The run snapshot: one canonical shape for the game's run state and the emulator's.

Game side: `from_save(save)` reads the game's own save JSON (SaveManager.ToJson(RunManager.ToSave(null)),
which the referee driver writes at each run decision). Emulator side: `EmuDump runstart` / `EmuDump run`
already write this shape (tools/emudump/RunSnap.cs). `compare(game, emu)` lists the differing fields in
a fixed order, so the first entry is the first field that differs.

Dropped on purpose: timestamps, map drawings, unlock state, discovered_* lists, floor_added_to_deck,
map point history stats (the run history screen's numbers, not run state).
"""

from __future__ import annotations

import re

from tools.fidelity.table_check import norm_enc

PREFIX = re.compile(r"^(CARD|RELIC|POTION|EVENT|ENCOUNTER|ENCHANTMENT|ACT|CHARACTER)\.")
# Emulator relic/potion/card entries that are spelt differently from the game's (none known yet).
ALIASES: dict[str, str] = {}
STREAM_RENAME = {"combat_orbs": "combat_orb_generation"}


def bare(x: str | None) -> str | None:
    if x is None:
        return None
    x = PREFIX.sub("", x)
    return ALIASES.get(x, x)


def enc(x: str | None, emulator: bool = False) -> str | None:
    """Encounter names compared without the _WEAK/_NORMAL/_ELITE/_BOSS suffix (see table_check.norm_enc):
    the emulator keys an encounter by its enum value, which weak and normal variants can share."""
    if not x:
        return None
    return norm_enc(bare(x), emulator=emulator)


def from_save(save: dict) -> dict:
    p = save["players"][0]
    slots = p["max_potion_slot_count"]
    potions = [""] * slots
    for pot in p.get("potions") or []:
        if 0 <= pot["slot_index"] < slots:
            potions[pot["slot_index"]] = bare(pot["id"])
    deck = []
    for c in p["deck"]:
        ench = c.get("enchantment")
        deck.append([bare(c["id"]), c.get("current_upgrade_level", 0),
                     bare(ench["id"]) if ench else None, ench.get("amount", 0) if ench else 0])
    odds = save["odds"]
    snap = {
        "act": save["current_act_index"],
        "floor": sum(len(a) for a in save.get("map_point_history") or []),
        "coord": [save["visited_map_coords"][-1]["col"], save["visited_map_coords"][-1]["row"]]
        if save.get("visited_map_coords") else None,
        "hp": p["current_hp"], "max_hp": p["max_hp"], "gold": p["gold"],
        "potion_slots": slots, "potions": potions,
        "deck": deck,
        "relics": [bare(r["id"]) for r in p["relics"]],
        "relic_props": {bare(r["id"]): r["props"] for r in p["relics"] if r.get("props")},
        "odds": {
            "card_rarity": round(p["odds"]["card_rarity_odds_value"], 6),
            "potion_reward": round(p["odds"]["potion_reward_odds_value"], 6),
            "unknown_monster": round(odds["unknown_map_point_monster_odds_value"], 6),
            "unknown_elite": round(odds["unknown_map_point_elite_odds_value"], 6),
            "unknown_treasure": round(odds["unknown_map_point_treasure_odds_value"], 6),
            "unknown_shop": round(odds["unknown_map_point_shop_odds_value"], 6),
        },
        "rng": dict(save["rng"]["counters"]),
        "player_rng": dict(p["rng"]["counters"]),
        "grab_bag": {k: [bare(x) for x in v] for k, v in p["relic_grab_bag"]["relic_id_lists"].items()},
        "shared_grab_bag": {k: [bare(x) for x in v]
                            for k, v in save["shared_relic_grab_bag"]["relic_id_lists"].items()},
        "events_seen": [bare(x) for x in save.get("events_seen") or []],
        "shop_removals_used": (p.get("extra_fields") or {}).get("card_shop_removals_used", 0),
        "max_energy": p.get("max_energy"),
    }
    acts = []
    for i, a in enumerate(save["acts"]):
        r = a["rooms"]
        act = {
            "events": [bare(x) for x in r["event_ids"]],
            "normal": [bare(x) for x in r["normal_encounter_ids"]],
            "elite": [bare(x) for x in r["elite_encounter_ids"]],
            "boss": bare(r["boss_id"]), "second_boss": bare(r.get("second_boss_id")),
            "ancient": bare(r["ancient_id"]),
        }
        if i == save["current_act_index"]:
            act.update(normal_visited=r["normal_encounters_visited"], elite_visited=r["elite_encounters_visited"],
                       events_visited=r["events_visited"])
        acts.append(act)
    snap["acts"] = acts
    return snap


# Fields in comparison order: where the run is, the player, then the run's hidden state.
ORDER = ["act", "floor", "coord", "hp", "max_hp", "gold", "potion_slots", "potions", "deck", "relics",
         "odds", "rng", "player_rng", "grab_bag", "shared_grab_bag", "acts", "events_seen", "shop_removals_used"]


def _norm(field: str, v, emulator: bool):
    if field == "rng" and isinstance(v, dict):
        return {STREAM_RENAME.get(k, k): n for k, n in v.items()}
    if field == "deck" and isinstance(v, list):
        # The emulator keeps an upgrade as a flag; the game as a level.
        return [[c[0], min(int(c[1]), 1), c[2], c[3]] for c in v]
    if field == "acts" and isinstance(v, list):
        out = []
        for a in v:
            a = dict(a)
            for k in ("normal", "elite"):
                a[k] = [enc(x, emulator) for x in a[k]]
            for k in ("boss", "second_boss"):
                a[k] = enc(a[k], emulator)
            out.append(a)
        return out
    if field in ("grab_bag", "shared_grab_bag") and isinstance(v, dict):
        return {k: x for k, x in v.items() if x}  # a rarity with an empty queue is the same as none
    return v


def _diff(path: str, g, e, out: list):
    if isinstance(g, dict) and isinstance(e, dict):
        for k in list(g) + [k for k in e if k not in g]:
            _diff(f"{path}.{k}", g.get(k), e.get(k), out)
    elif isinstance(g, list) and isinstance(e, list) and g and isinstance(g[0], dict):
        for i in range(max(len(g), len(e))):
            _diff(f"{path}[{i}]", g[i] if i < len(g) else None, e[i] if i < len(e) else None, out)
    elif isinstance(g, list) and isinstance(e, list):
        if g != e:  # the first differing position, with both lengths
            i = next((k for k, (a, b) in enumerate(zip(g, e)) if a != b), min(len(g), len(e)))
            out.append((f"{path}[{i}] (len {len(g)}/{len(e)})", g[i] if i < len(g) else "<end>",
                        e[i] if i < len(e) else "<end>"))
    elif g != e:
        out.append((path, g, e))


def compare(game: dict, emu: dict, fields=ORDER) -> list[tuple[str, object, object]]:
    """(field path, game value, emulator value) for each difference, in ORDER."""
    out: list = []
    for f in fields:
        if f not in game and f not in emu:
            continue
        _diff(f, _norm(f, game.get(f), False), _norm(f, emu.get(f), True), out)
    return out


def short(v, n: int = 160) -> str:
    import json
    s = json.dumps(v)
    return s if len(s) <= n else s[:n] + "..."
