"""Compare a fight the referee played in the game with the emulator's replay of it, action by action.

    python -m tools.fidelity.fight_compare <referee out.json> <emudump fight.json> --id-map data/id_map.json

Reports the first diverging step per fight with the differing fields: player HP, block, energy
and powers; hand, draw (in order), discard and exhaust piles; living enemies' HP, block and
powers; and each combat RNG stream's call count.

The step that ends the fight is compared too, not only whether the fight is still on. Once a won
fight is over the game's snapshot shows what the run holds (player HP and max HP with the after-combat
heals in, the potion belt, the relics, the RNG streams; Player.AfterCombatEnd has cleared the block,
powers and piles), and a lost fight keeps the combat as it ended; game_view and emu_view give both
in the same shape (see "the state after the fight" below).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path


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
NOT_POWERS = {
    "STUNNED",
    # Emulator-internal counters that the game keeps as DynamicVars on another power, not as
    # powers of their own (triage of night03, each checked against the game class):
    "INFERNO_SELF_DAMAGE",         # InfernoPower.SelfDamage (IncrementSelfDamage per play)
    "CRIMSON_MANTLE_SELF_DAMAGE",  # CrimsonMantlePower.SelfDamage
    "THE_BOMB_DAMAGE",             # TheBombPower's Damage var (turns are the power's Amount)
    "PANACHE_ARMED",               # emulator flag for Panache's every-5th-card trigger
    "CLAW_DAMAGE",                 # Claw.BuffFromClawPlay raises the Claws' own damage, no power
    "SLOW_COUNT",                  # SlowPower's "SlowAmount" DynamicVar (cards played this turn), not a power
    # SkittishPower.HasGainedBlockThisTurn (H01): the once-a-turn latch lives in the power's internal
    # Data (GetInternalData<Data>), never in the power list. The game's Skittish is its Amount alone
    # (SkittishAmount: 7 at A8, 6 below), still compared as SKITTISH. The latch shows only as the block
    # it paid, and as the block it withholds from a second hit, or from the next turn's first if it sticks:
    # the block is compared. As a power of its own it sat in the emulator's list and never in the game's,
    # so every step with the latch up differed, and the first of those ended the fight's verdict before
    # the Skittish timing it stands for.
    "SKITTISH_SPENT",
}

# Powers the game stacks as Single, so their Amount is always 1, whose emulator buff keeps a state of its own in the
# amount. SurroundedPower (the Kaiser Crab's, on the player) keeps its Facing in a field; the emulator's SURROUNDED
# buff holds the facing as its amount (1 right, 2 left), so a player facing left differed at every step from the
# first left turn on, and every Kaiser Crab fight's verdict ended there. The facing shows where it acts: in the
# player's HP after the enemy turn.
SINGLE_AMOUNT = {"SURROUNDED"}

# Emulator buff -> the game power with the same rules and amount.
# The tables live in power_aliases.json, which tools/emudump/Reanchor.cs reads too, so writing the
# game's powers back into the emulator and comparing them use one mapping. CrimsonMantlePower.Amount
# is the block per turn (CRIMSON_MANTLE_BLOCK).
_ALIASES = json.loads((Path(__file__).with_name("power_aliases.json")).read_text(encoding="utf-8"))
EMU_RENAME = dict(_ALIASES["emu_rename"])

# Every subclass of the game's TemporaryStrengthPower is one emulator TEMPORARY_STRENGTH buff (and
# TemporaryDexterityPower's, TemporaryFocusPower's likewise). The emulator keeps its magnitude and lets
# the owner set the direction (the player's is a gain taken back at end of turn, an enemy's is a loss
# given back), so the game amounts are summed as they stand. IsPositive=false subclasses (Dark
# Shackles, Mangle...) only sit on enemies.
FOLDS = {sub: buff for buff, subs in _ALIASES["folds"].items() for sub in subs}
TEMP_STRENGTH = {sub for sub, buff in FOLDS.items() if buff == "TEMPORARY_STRENGTH"}


# Enemy moves in one vocabulary. The game records MonsterModel.NextMove's intents (referee/driver/Combat.cs
# Intents: [type] or [type, base damage per hit, hits]); the emulator its Intent (Core/Enemy.cs IntentType
# numbers) and the move's second intent. What both can say: the kind of the move's first intent, and the
# damage-carrying intent's base damage and hits. The game's StatusIntent (a status card into the deck) and
# DebuffStrong are Debuffs to the emulator, and a stun is its STUNNED buff, the game's StunIntent move.
EMU_INTENT = {0: "ATTACK", 1: "DEFEND", 2: "BUFF", 3: "DEBUFF", 4: "OTHER"}
GAME_INTENT = {"Attack": "ATTACK", "DeathBlow": "ATTACK", "Defend": "DEFEND", "Buff": "BUFF", "Debuff": "DEBUFF",
               "DebuffStrong": "DEBUFF", "StatusCard": "DEBUFF", "CardDebuff": "DEBUFF", "Stun": "STUN"}


def game_intent(e: dict):
    """The game enemy's next move as {kind, attack}, or None for a record made before moves were recorded."""
    if "intents" not in e:
        return None
    its = e["intents"]
    kind = GAME_INTENT.get(its[0][0], "OTHER") if its else "NONE"
    attack = next(([i[1], i[2]] for i in its if len(i) >= 3), None)
    return {"kind": kind, "attack": attack}


def emu_intent(e: dict):
    it = e.get("intent")
    if it is None:
        return None
    if any(power(b["name"]) == "STUNNED" for b in e.get("buffs", [])):
        return {"kind": "STUN", "attack": None}
    sec = it.get("secondary")
    if it["type"] == 0 or it.get("carries_damage"):
        attack = [it["magnitude"], it["hits"]]
    elif sec and (sec["type"] == 0 or sec.get("carries_damage")):
        attack = [sec["magnitude"], sec["hits"]]
    else:
        attack = None
    # A move whose number is damage leads with its attack in the game (Core/Enemy.cs Intent.CarriesDamage).
    kind = "ATTACK" if it.get("carries_damage") else EMU_INTENT.get(it["type"], "OTHER")
    return {"kind": kind, "attack": attack}


# Likewise every subclass of TemporaryDexterityPower is one emulator TEMPORARY_DEXTERITY buff
# (AnticipatePower, HelicalDartPower, SpeedPotionPower; checked in the decompile, setup fights reach them):
# power_aliases.json's folds, which game_powers applies.
TEMP_DEXTERITY = {sub for sub, buff in FOLDS.items() if buff == "TEMPORARY_DEXTERITY"}


FOLD_TARGETS = set(FOLDS.values())


def game_powers(raw: list) -> list:
    out, folded = [], collections.Counter()
    for n, a in raw:
        name = power(n)
        if name in FOLDS:
            folded[FOLDS[name]] += a
        elif name in FOLD_TARGETS:
            # The game's own power of the buff's name (BlockNextTurnPower beside SelfFormingClayPower): the emulator holds
            # one sum, so it joins the folded ones instead of standing beside them as a second entry of that name.
            folded[name] += a
        else:
            out.append([name, a])
    out += [[k, v] for k, v in folded.items() if v]
    return sorted(out)


def emu_powers(pairs) -> list:
    """The emulator's buffs ((name, amount) pairs) as the game's power list: internal counters dropped, buffs renamed
    to the game's power, a Single power's amount 1. The one projection every comparer of emulator powers uses."""
    out = []
    for n, a in pairs:
        p = power(n)
        if p in NOT_POWERS:
            continue
        out.append([EMU_RENAME.get(p, p), 1 if p in SINGLE_AMOUNT and a else a])
    return sorted(out)


def emu_view(st: dict, nm: dict) -> dict:
    if not st["in_combat"]:
        return _emu_end_view(st, nm)
    if "snapshot" not in st:
        return {"in_combat": False}
    s = st["snapshot"]
    card = lambda c: [nm["cards"].get(c["def_id"], c["def_id"]), bool(c["upgraded"])]  # noqa: E731
    powers = lambda bs: emu_powers((b["name"], b["amount"]) for b in bs)  # noqa: E731
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
                     "powers": powers(e["buffs"]), "intent": emu_intent(e)} for e in s["enemies"] if e["hp"] > 0],
        "rng": st["rng"],
        # Setup fights only (the referee then records the belt): the potion in each slot, None if empty.
        **({"potions": st["potions"]} if "potions" in st else {}),
    }


# The RNG streams a fight moves (the game's names); the others (UpFront, the map's, the treasure room's) stay put in one.
RNG_STREAMS = ("Shuffle", "CombatTargets", "MonsterAi", "CombatCardSelection", "CombatCardGeneration",
               "CombatEnergyCosts", "CombatPotionGeneration", "CombatOrbGeneration", "Niche")


def game_view(st: dict) -> dict:
    if not st.get("in_combat"):
        return _game_end_view(st)
    p = st["player"]
    return {
        "in_combat": True,
        "player": {"hp": p["hp"], "max_hp": p["max_hp"], "block": p["block"], "energy": p["energy"],
                   "powers": game_powers(p["powers"])},
        "hand": st["hand"], "draw": st["draw"], "discard": st["discard"], "exhaust": st["exhaust"],
        "enemies": [{"name": e["name"], "hp": e["hp"], "block": e["block"], "powers": game_powers(e["powers"]),
                     "intent": game_intent(e)} for e in st["enemies"] if e["hp"] > 0],
        "rng": {k: v["counter"] for k, v in st["rng"].items() if k in RNG_STREAMS},
        **({"potions": st["potions"]} if "potions" in st else {}),
    }


# ---- the state after the fight ----------------------------------------------------------------------
#
# The step that ends a fight used to be compared on one thing: in_combat. A fight whose last hit left the
# player at 68 in the emulator and 76 in the game was scored identical. That step's view now has the same
# fields as a step in combat, so every consumer of game_view/emu_view (the plain replay, the re-anchored
# replay, coverage, the corpus) judges it with no change of its own.
#
# WON (the player alive). The game's Player.AfterCombatEnd clears the player's powers, block and piles inside
# the step and the combat is gone (no roster). What is left to compare is what the run holds: HP and max HP
# with the after-combat heals in (Burning Blood, Meat on the Bone, Chosen Cheese ...), the RNG positions the
# fight left, and, in a record that has them, the potion belt, the relics, the gold and the relics' counters.
# The emulator keeps those on the run, and the run holds no block, powers, piles or roster, so its view has
# them empty: that half of the view checks the game's record, the run's state is the other half.
# LOST. The run is over, so nothing is carried on: the view is the outcome alone (both sides' HP is 0). The rest
# of that step is only where each side stopped the combat, and the two stop in different places: the game's
# record shows the enemy phase cut at the killing hit, the emulator's the phase played out (an enemy's Strength
# gain, a debuff tick, a status card; three Scrolls of Biting fights end at another max HP, Paper Cuts' loss).
# Comparing that leftover called fights different over state nothing reads (7 of the first 59 identical corpus
# fights, 3 more in the rest).

# What only some records have: it counts as a difference where both sides wrote the field. The game's
# snapshot has no gold and no relic counters yet (referee/driver/Setup.cs Decorate), the emulator's dump has both.
OPTIONAL = {".player.gold", ".relic_counters"}


def _game_end_view(st: dict) -> dict:
    p = st.get("player")
    if p is None:  # a state that is no more than the flag: nothing was recorded to compare
        return {"in_combat": False}
    if p["hp"] <= 0:
        return {"in_combat": False, "outcome": "lost"}
    v = {
        "in_combat": False, "outcome": "won",
        "player": {"hp": p["hp"], "max_hp": p["max_hp"], "block": p["block"], "powers": game_powers(p["powers"]),
                   **({"gold": p["gold"]} if "gold" in p else {})},
        "hand": st["hand"], "draw": st["draw"], "discard": st["discard"], "exhaust": st["exhaust"],
        "enemies": [{"name": e["name"], "hp": e["hp"], "block": e["block"], "powers": game_powers(e["powers"]),
                     "intent": game_intent(e)} for e in st["enemies"] if e["hp"] > 0],
        "rng": {k: v["counter"] for k, v in st["rng"].items() if k in RNG_STREAMS},
    }
    for k in ("potions", "relics", "relic_counters"):
        if k in st:
            v[k] = st[k]
    return v


def _emu_end_view(st: dict, nm: dict) -> dict:
    """The emulator's state after the fight: ``final``, which EmuDump writes (Fight.Final)."""
    fin = st.get("final")
    if fin is None:  # an emulator dump that did not say how the fight ended: nothing was recorded to compare
        return {"in_combat": False}
    if not fin["won"]:
        return {"in_combat": False, "outcome": "lost"}
    p = fin["player"]
    v = {
        "in_combat": False, "outcome": "won",
        "player": {"hp": p["hp"], "max_hp": p["max_hp"], "block": 0, "powers": [],
                   **({"gold": p["gold"]} if "gold" in p else {})},
        "hand": [], "draw": [], "discard": [], "exhaust": [], "enemies": [],
        "rng": {k: n for k, n in fin["rng"].items() if k in RNG_STREAMS},
    }
    for k in ("potions", "relics", "relic_counters"):
        if k in fin:
            v[k] = fin[k]
    return v


def _recorded(a, b):
    """Enemy lists without the intents one side did not record (a referee record older than intents)."""
    if not (isinstance(a, list) and isinstance(b, list)):
        return a, b
    if not any(isinstance(x, dict) and "intent" in x for x in a + b):
        return a, b
    def drop(xs, ys):
        return [{k: v for k, v in x.items() if k != "intent"}
                if isinstance(x, dict) and (x.get("intent") is None or i >= len(ys) or not isinstance(ys[i], dict)
                                            or ys[i].get("intent") is None) else x
                for i, x in enumerate(xs)]
    return drop(a, b), drop(b, a)


def diff(a, b, path="") -> list:
    if isinstance(a, dict) and isinstance(b, dict):
        # One side's fight is on and the other's is over: the side that is over is its flag alone here, as it was
        # before the end of a fight had a view (reanchor_compare stops on what such a step shows of the enemies).
        if "in_combat" in a and "in_combat" in b and a["in_combat"] != b["in_combat"]:
            a, b = ({"in_combat": False} if not x["in_combat"] else x for x in (a, b))
        # Both over, one won and the other lost: the rest of the step cannot be compared.
        if "outcome" in a and "outcome" in b and a["outcome"] != b["outcome"]:
            return [(f"{path}.outcome", a["outcome"], b["outcome"])]
        return [d for k in sorted(set(a) | set(b)) if f"{path}.{k}" not in OPTIONAL or (k in a and k in b)
                for d in diff(a.get(k), b.get(k), f"{path}.{k}")]
    a, b = _recorded(a, b)
    return [] if a == b else [(path, a, b)]


def first_difference(g: dict, e: dict, nm: dict):
    """(index, diff list) of the first game step the emulator's replay does not reproduce, or None.

    The one test every comparer of a plain replay uses (corpus, coverage, batch.py, scenarios, refset):
    a replay that stops before the game's last step differs at the first step it lacks, with the
    pseudo-path ``(error)`` when the emulator reported an error and ``(length)`` when it did not. Pairing
    the steps with zip() alone took such a replay for identical.
    """
    gsteps, esteps = g.get("steps", []), e.get("steps", [])
    for i, (gs, es) in enumerate(zip(gsteps, esteps)):
        d = diff(game_view(gs["state"]), emu_view(es["state"], nm))
        if d:
            return i, d
    if len(esteps) < len(gsteps):
        why = "(error)" if e.get("error") else "(length)"
        return len(esteps), [(why, f"{len(gsteps)} steps", f"{len(esteps)} steps")]
    return None


def report(g: dict, e: dict, nm: dict) -> tuple[list, int | None]:
    """One fight's lines of the comparison main prints, and the index of its first differing step.

    The index is None when every game step is reproduced. batch.py calls it per fight as fights finish.
    """
    lines = []
    for side, doc in (("game", g), ("emulator", e)):
        if doc.get("error"):
            lines.append(f"{g.get('seed')}: {side} error: {doc['error'][:400]}")
    fd = first_difference(g, e, nm)
    if fd is not None:
        i, d = fd
        act = g["steps"][i]["action"] if i < len(g.get("steps", [])) else None
        lines.append(f"{g.get('seed')}: first difference at step {i} after {act}")
        lines += [f"    {path}: game {x}  emulator {y}" for path, x, y in d[:12]]
        return lines, i
    n_g, n_e = len(g.get("steps", [])), len(e.get("steps", []))
    lines.append(f"{g.get('seed')}: {min(n_g, n_e)} steps identical (game {n_g}, emulator {n_e})")
    return lines, None


def main() -> int:
    """Compare every fight; exit 0 only when every game fight was reproduced step for step.

    Exit 1 when a fight differs, the emulator reported an error for it, an emulator result is missing or
    belongs to another seed, or nothing was compared.
    """
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("referee")
    ap.add_argument("emulator")
    ap.add_argument("--id-map", default="data/id_map.json")
    a = ap.parse_args()
    nm = names(json.load(open(a.id_map, encoding="utf-8")))
    games = json.load(open(a.referee, encoding="utf-8"))["results"]
    emus = json.load(open(a.emulator, encoding="utf-8"))["results"]
    bad = 0
    for k, g in enumerate(games):
        e = emus[k] if k < len(emus) else None
        if e is None or e.get("seed") not in (None, g.get("seed")):
            # Results are paired by position; one missing or out of order is never a match.
            print(f"{g.get('seed')}: no emulator result" + (f" (position {k} holds seed {e.get('seed')})" if e else ""))
            bad += 1
            continue
        lines, first = report(g, e, nm)
        for line in lines:
            print(line)
        # An emulator error fails the fight even when every step it replayed matched (an exception after
        # the last step, or a fight with no steps at all).
        bad += first is not None or bool(e.get("error"))
    if len(emus) > len(games):
        print(f"{len(emus) - len(games)} emulator results have no game fight")
        bad += 1
    print(f"{len(games)} fights, {bad} not reproduced")
    return 0 if games and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
