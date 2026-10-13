"""The run snapshot: one canonical shape for the game's run state and the emulator's.

Game side: `from_save(save)` reads the game's own save JSON (SaveManager.ToJson(RunManager.ToSave(null)),
which the referee driver writes at each run decision). Emulator side: `EmuDump runstart` / `EmuDump run`
already write this shape (tools/emudump/RunSnap.cs). `compare(game, emu)` lists the differing fields in
a fixed order, so the first entry is the first field that differs.

Nothing outside ORDER is ever taken as equal: NOT_COMPARED lists the save's fields the comparison
leaves out and why, and `not_compared(game, emu)` names the saved relic and card properties a snapshot
carries that `compare` does not check; run_compare2 prints both with every comparison.
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


# Fields of the game's save the comparison does not check, and why. run_compare2 prints this list.
NOT_COMPARED = {
    "timestamps, num_reloads, game_mode, modifiers, net_id": "bookkeeping, not run state",
    "map drawings, unlock state, discovered_* lists, floor_added_to_deck": "profile and history, not run state",
    "map_point_history beyond its length (the floor)": "the run history screen's numbers",
    "extra_fields (damage_dealt, debuffs_applied, started_with_neow)": "run statistics",
    "character_id, base_orb_slot_count": "the emulator plays the Ironclad only (no orbs)",
    "pre_finished_room": "the room a save resumes into",
    "melted relics": "the game keeps a melted wax relic, inert; the emulator drops it ('relics' compares the "
                     "unmelted ones)",
    "event option identity": "the emulator has no option ids: the game's option index is applied as the "
                             "emulator's action; the enabled indices and the first page's count are compared",
    "a Thieving Hopper's stolen card on a rewards screen": "the emulator returns it to the deck as the fight "
                                                            "ends (known gap); the deck is compared",
    "rest site options the game shows disabled": "the emulator's rest screen lists the enabled ones only",
    "coord and the ancient's events_visited at an act change": "the emulator counts both a decision early; "
                                                                "compared at the next decision",
}

# Saved relic properties the game's hooks read only modulo n (`CardsPlayed % 4` and so on), where
# the emulator keeps just the remainder: both sides are compared modulo n.
PROP_MOD = {("IRON_CLUB", "CardsPlayed"): 4, ("NUNCHAKU", "AttacksPlayed"): 10, ("BOOK_OF_FIVE_RINGS", "CardsAdded"): 5}

# Saved properties declared `[SavedProperty(SerializationCondition.SaveIfNotTypeDefault)]`: the save
# leaves them out at their type's default, so an absent one is that default. Every other compared
# property is AlwaysSave, and an absent one stays None (a difference, never the emulator's value).
SAVED_IF_NOT_DEFAULT = {("JOSS_PAPER", "CardsExhausted"): 0}

# The game's enums its save writes as numbers (CardType; TinkerTime.RiderEffect).
CARD_TYPES = ["None", "Attack", "Skill", "Power", "Status", "Curse", "Quest"]
RIDERS = ["None", "Sapping", "Violence", "Choking", "Energized", "Wisdom", "Chaos", "Expertise", "Curious",
          "Improvement"]

# Card properties RunSnap.CardPropsOf exports; any other card's saved properties are not compared.
CARD_PROPS = {"MAD_SCIENCE": {"TinkerTimeType", "TinkerTimeRider"}, "GUILTY": {"CombatsSeen"}}


def props_of(model: dict) -> dict:
    """A relic's or card's saved properties as {name: value} (ints, bools, strings, model ids, cards)."""
    out = {}
    for items in (model.get("props") or {}).values():
        for it in items or []:
            out[it["name"]] = it["value"]
    return out


def _card_props(deck: list) -> dict:
    """The card properties the emulator keeps (RunSnap.CardPropsOf), as sorted multisets."""
    out: dict = {}
    for c in deck:
        pr = props_of(c)
        cid = bare(c["id"])
        if cid == "MAD_SCIENCE":
            t, r = pr.get("TinkerTimeType", 0), pr.get("TinkerTimeRider", 0)
            out.setdefault(cid, []).append(f"{CARD_TYPES[t] if 0 <= t < len(CARD_TYPES) else t}/"
                                           f"{RIDERS[r] if 0 <= r < len(RIDERS) else r}")
        elif cid == "GUILTY":
            out.setdefault(cid, []).append(pr.get("CombatsSeen", 0))
    return {k: sorted(v) for k, v in out.items()}


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
        # A melted wax relic stays in the game's list (IsMelted, inert); the emulator drops it.
        "relics": [bare(r["id"]) for r in p["relics"] if not props_of(r).get("IsMelted")],
        "melted": [bare(r["id"]) for r in p["relics"] if props_of(r).get("IsMelted")],
        "wax": [bare(r["id"]) for r in p["relics"] if props_of(r).get("IsWax") and not props_of(r).get("IsMelted")],
        "relic_props": {bare(r["id"]): {k: v for k, v in props_of(r).items() if k not in ("IsWax", "IsMelted")}
                        for r in p["relics"] if r.get("props") and not props_of(r).get("IsMelted")},
        "card_props": _card_props(p["deck"]),
        "card_prop_names": sorted({f"{bare(c['id'])}.{k}" for c in p["deck"] for k in props_of(c)}),
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
        "ascension": save.get("ascension"),
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
ORDER = ["act", "floor", "coord", "hp", "max_hp", "max_energy", "gold", "potion_slots", "potions", "deck",
         "card_props", "relics", "wax", "relic_props", "odds", "rng", "player_rng", "grab_bag", "shared_grab_bag",
         "acts", "events_seen", "shop_removals_used", "ascension"]


def _relic_props(game: dict | None, emu: dict | None) -> tuple[dict, dict]:
    """Both sides' relic properties, restricted to what the emulator exports for the relics it holds
    (RunSnap.RelicProps; not_compared names the rest), with PROP_MOD applied. A property the game's
    save does not carry is None, never the emulator's value."""
    game, emu = game or {}, emu or {}
    g: dict = {}
    e: dict = {}
    for relic, props in emu.items():
        for name, ev in props.items():
            gv = (game.get(relic) or {}).get(name, SAVED_IF_NOT_DEFAULT.get((relic, name)))
            m = PROP_MOD.get((relic, name))
            if m:
                gv = gv % m if isinstance(gv, int) else gv
                ev = ev % m if isinstance(ev, int) else ev
            g.setdefault(relic, {})[name] = gv
            e.setdefault(relic, {})[name] = ev
    return g, e


def not_compared(game: dict, emu: dict) -> list[str]:
    """What this game snapshot carries that compare() does not check: saved relic and card properties
    the emulator does not export, and melted relics."""
    out = []
    eprops = emu.get("relic_props") or {}
    for relic, props in (game.get("relic_props") or {}).items():
        for name in props:
            if name not in (eprops.get(relic) or {}):
                out.append(f"relic_props.{relic}.{name}")
    for cp in game.get("card_prop_names") or []:
        cid, name = cp.split(".", 1)
        if name not in CARD_PROPS.get(cid, ()):
            out.append(f"card_props.{cp}")
    if game.get("melted"):
        out.append("melted relics")
    return out


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
        g, e = game.get(f), emu.get(f)
        if f == "relic_props":
            g, e = _relic_props(g, e)
        _diff(f, _norm(f, g, False), _norm(f, e, True), out)
    return out


def short(v, n: int = 160) -> str:
    import json
    s = json.dumps(v)
    return s if len(s) <= n else s[:n] + "..."
