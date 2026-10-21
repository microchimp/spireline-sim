#!/usr/bin/env python3
"""Setup fights end to end: make coverage-driven or dense fight specs, play them, replay, compare, shrink.

    python referee/scenarios.py --name <name> --mode dense|coverage --fights 60 [--start 0] [--servers 2]
        [--game D:/spireline/referee/game-setup] [--shrink K] [--shrink-budget 400] [--make-only]
    python referee/scenarios.py --name <name> --mode targets --targets LAB/targets.json --fights N [--start S]
        [--servers K] --game <game copy> --shrink K [--jobs <dir>] [--trace N] [--aim all|bugs|gaps]
    python referee/scenarios.py --name <name> --mode choices --fights N [--edges random|cycle|none|one|max|last|unusual]
        [--features LAB/cache/model_features.json] [...as above]
    python referee/scenarios.py --name <name> --mode analyze --jobs <dir holding batch.py's <name>.*> --game <copy>
        --shrink K --trace N
    python referee/scenarios.py --name <name> --mode runs --runs-from <runs.py batch's <name>.runs.json> --jobs <dir>
        --game <copy> --shrink K [--trace N]

Run from the fork's root. Specs carry a "setup" list (referee/driver/Setup.cs) built from the game's
own content tables (the export, see RUNBOOK "Table check"), limited to what EmuDump can mirror:

- coverage: one encounter (weak, normal, elite or boss of any act, weighted towards the least-seen)
  plus 1-3 of the least-covered relics, potions, powers or upgraded cards.
- dense: an encounter, 10-20 cards from the Ironclad and colourless pools (about 35% upgraded,
  so 20-30 in the deck), 10-20 Ironclad or shared relics, a full potion belt, 0-3 powers on the
  player and 0-2 on the first enemies.
- targets (the fix lab): each spec aims at one target of tools/fidelity/lab_targets.py's list, drawn by
  weight, and stays realistic: the act the Codex profile (aggregate counts, LAB/cache/codex_profile.json)
  says the target is held in or met in (the act item from act 2 on; in act 1 a seed whose act-1 variant
  is known to match, LAB/cache/act1_seeds.json, when one is; no seed twice in a batch), a room and
  encounter of that act by how often real runs meet them, a deck of that act's size with its Strikes and
  Defends (one deck_set item), then cards, relics and potions drawn by how often real runs hold them at
  that act, mixed half and half with how often they are held together with the target. Each card, relic
  and potion is its own item, so the shrinker can tell which ones the divergence needs. Diverging fights are shrunk (--shrink) and
  traced (referee/trace_divergent.py, --trace N, default 40 in this mode) for their first differing event.
  --jobs puts every output in that folder instead of jobs-setup; <name>.aims.json names each spec's target
  (and seed_from: cache for a seed borrowed from a recorded fight, fresh otherwise)
  and <name>.shrink.json every shrink case's status.
  --aim bugs keeps the targets an open backlog item names (lab_targets writes backlog_items); --aim gaps
  weighs the starter items (BURNING_BLOOD, BASH, STRIKE_IRONCLAD, DEFEND_IRONCLAD, ASCENDERS_BANE) 1/20,
  since every real run holds them and their meet share would drown the gaps. --aim all (default) is as before.
- choices (the fix lab): a small deck, 1-3 relics whose pickup opens a card screen (the setup info's
  TransformSelect follow-up), cards that open one (the game commands each model calls, --features), a
  screen-opening card or two in the hand, and the driver's edge answers (spec "choice_edges": none, one, the most
  a screen takes, the last eligible card, the most unusual one; --edges, default one drawn per spec), so a
  difference in how many a screen takes, or which cards it offers, shows as a divergence.
- runs (the fix lab): each whole run of a runs.py batch whose first divergence is in a fight, as a setup fight
  that starts where that fight did (referee/run_repro.py: the run's seed, Neow pick and act, its holdings before
  the fight through player_load, the encounter); judged, shrunk (each loaded card, relic and potion is an item)
  and traced as usual. <name>.repro.json names each spec's run, whether the fight reproduced the run's
  divergence (same, field, other, identical, error), the runs left out and the state a setup cannot put back.
- analyze: no new fights. The finished batch.py batch <name> in --jobs (<name>.json, .out.json, .emu.json,
  .reanchor.json) is judged, shrunk, traced and given verdicts and packs like any other batch, so
  tools/fidelity/lab_groups.py reads it.
--shrink-cap N is the game fights one diverging fight may cost (48), --shrink-minutes M and --trace-minutes M stop
starting shrink rounds and trace calls after M minutes (a per-batch budget), --trace-servers K (up to 8) the
servers for the traced re-runs. A failure in the context packs is a warning: verdicts.json and summary.txt
are written before them.

Every spec uses potions (use_potions). The job goes to serve.py (one game per server), the results to
EmuDump fight and reanchor, and each fight is judged as batch.py does, plus the setup items themselves
(tools/fidelity/setup_compare.py). Diverging fights are shrunk to a minimal setup (referee/shrink.py)
when --shrink is given. Output in D:/spireline/referee/jobs-setup/<name>.*; coverage of setup items in
jobs-setup/coverage-setup.json (local, never committed). Plausibility flags mark specs no real run
could produce (cross_act, debug_room, cross_character, ...), so triage can rank them lower.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
JOBS = Path(r"D:\spireline\referee\jobs-setup")
DEFAULT_JOBS = JOBS
GAME = r"D:\spireline\referee\game-setup"
EXPORT = Path(r"D:\spireline\referee\exports\v0.107.1.json")
EMU_DLL = ROOT / "tools" / "emudump" / "bin" / "Release" / "net9.0" / "EmuDump.dll"
ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
ACTS = ("OVERGROWTH", "UNDERDOCKS", "HIVE", "GLORY")
KINDS = ("weak", "normal", "elite", "boss")
# Powers any character's player can hold through shared cards, potions or relics; the Ironclad's own
# power cards add theirs (Content). Enemy powers are the debuffs player cards put on enemies.
PLAYER_POWERS = ("STRENGTH_POWER", "DEXTERITY_POWER", "VIGOR_POWER", "THORNS_POWER", "PLATING_POWER", "ARTIFACT_POWER",
                 "REGEN_POWER", "RITUAL_POWER", "BUFFER_POWER", "INTANGIBLE_POWER")
ENEMY_POWERS = ("VULNERABLE_POWER", "WEAK_POWER", "STRENGTH_POWER", "POISON_POWER")
OTHER_CHARACTER_POWERS = {"POISON_POWER", "INTANGIBLE_POWER"}  # Silent's; no Ironclad or shared source
RELIC_RARITIES = {"Common", "Uncommon", "Rare", "Shop"}
MIRRORED_FOLLOW_UPS = {"None", "TransformSelect"}


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).upper()


class Content:
    """What specs are drawn from: the game's own tables, cut to what EmuDump mirrors."""

    def __init__(self, export: dict, info: dict, id_map: dict, character: str = "IRONCLAD"):
        ci = export["character_info"][character]
        own_cards, own_relics, own_potions = ci["card_pool"], ci["relic_pool"], ci["potion_pool"]
        legal = lambda c: c.get("multiplayer_constraint", "None") == "None"  # noqa: E731
        self.cards = sorted({c["id"] for pool in (own_cards, "ColorlessCardPool")
                             for c in export["card_pools"][pool]["all"]
                             if legal(c) and c["rarity"] in ("Basic", "Common", "Uncommon", "Rare")})
        self.upgradable = {c["id"] for pool in (own_cards, "ColorlessCardPool") for c in export["card_pools"][pool]["all"]
                           if int(c.get("max_upgrade_level", "0") or 0) > 0}
        follow = info.get("relics", {})
        self.relics = sorted({r["id"] for pool in (own_relics, "SharedRelicPool") for r in export["relic_pools"][pool]["all"]
                              if r["rarity"] in RELIC_RARITIES and follow.get(r["id"]) in MIRRORED_FOLLOW_UPS})
        self.relic_follow = {r: follow.get(r) for r in self.relics}  # what picking the relic up opens (None: nothing)
        self.potions = sorted({p["id"] for pool in (own_potions, "SharedPotionPool") for p in export["potion_pools"][pool]["all"]})
        game_powers = {snake(k) for k in id_map["powers"]}
        buff_keys = set(info.get("buff_keys", [])) - set(info.get("internal_keys", []))
        mirrored = {p for p in game_powers if p.removesuffix("_POWER") in buff_keys}
        card_powers = {c["id"] + "_POWER" for c in export["card_pools"][own_cards]["all"] if c["type"] == "Power"}
        self.player_powers = sorted((set(PLAYER_POWERS) | card_powers) & mirrored)
        self.enemy_powers = sorted(set(ENEMY_POWERS) & mirrored)
        self.encounters = [(e, act, kind) for act in ACTS for kind in KINDS for e in export["acts"][act][kind]]
        others = [c for c in export["character_info"] if c != character]
        pools = export["card_pools"]
        self.other_cards = {c["id"] for o in others for c in pools[export["character_info"][o]["card_pool"]]["all"]}
        self.other_relics = {r["id"] for o in others for r in export["relic_pools"][export["character_info"][o]["relic_pool"]]["all"]}
        self.other_potions = {p["id"] for o in others for p in export["potion_pools"][export["character_info"][o]["potion_pool"]]["all"]}


class NoContent:
    """Stands in for Content where fights were not made from setup items (a batch.py batch): nothing is another character's."""
    other_cards = other_relics = other_potions = frozenset()


def load_content() -> Content:
    info_path = JOBS / "setupinfo.json"
    if not info_path.exists():
        JOBS.mkdir(parents=True, exist_ok=True)
        info_path.write_text(subprocess.run(["dotnet", str(EMU_DLL), "setupinfo"], capture_output=True, text=True,
                                            check=True).stdout, encoding="utf-8")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    export = json.loads(EXPORT.read_text(encoding="utf-8"))["results"][0]
    id_map = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))
    return Content(export, info, id_map)


# ---------------------------------------------------------------- coverage of setup items

def item_keys(spec: dict) -> list[str]:
    keys = []
    for it in spec.get("setup", []):
        cmd, ident = it.get("cmd"), it.get("id")
        if cmd == "fight":
            keys.append(f"encounter:{ident}")
        elif cmd == "card":
            keys.append(("card+:" if it.get("upgraded") else "card:") + ident)
        elif cmd == "power":
            keys.append(f"power:{ident}@{'player' if it.get('target') == 'player' else 'enemy'}")
        elif cmd in ("relic", "potion"):
            keys.append(f"{cmd}:{ident}")
    return keys


def weight(cov: dict, key: str) -> float:
    c = cov.get("items", {}).get(key, {})
    seen, matched = c.get("seen", 0), c.get("matched", 0)
    if seen == 0:
        return 3.0
    return 1.0 if matched < 3 else 0.2


def pick(cov: dict, keys: list[str], k: int, rng: random.Random) -> list[str]:
    """k distinct keys, weighted towards the least covered."""
    pool = [(x, weight(cov, x)) for x in keys]
    out = []
    for _ in range(min(k, len(pool))):
        x = rng.random() * sum(w for _, w in pool)
        for j, (key, w) in enumerate(pool):
            x -= w
            if x <= 0:
                out.append(key)
                pool.pop(j)
                break
    return out


def update_coverage(cov: dict, specs: list, verdicts: list) -> None:
    items = cov.setdefault("items", {})
    for spec, v in zip(specs, verdicts):
        for key in set(item_keys(spec)) | set(v.get("upgraded_plays", [])):
            c = items.setdefault(key, {})
            c["seen"] = c.get("seen", 0) + 1
            if v.get("identical"):
                c["matched"] = c.get("matched", 0) + 1


# ---------------------------------------------------------------- spec generation

def _seed(rng: random.Random) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(10))


def _power_item(rng: random.Random, power: str, target: str) -> dict:
    amount = rng.randint(1, 3)
    if target != "player" and power == "STRENGTH_POWER":
        amount = -amount  # a Strength loss, as Disarm or a Shackling Potion leaves it
    return {"cmd": "power", "id": power, "amount": amount, "target": target}


def encounter_key(content: Content, cov: dict, rng: random.Random) -> tuple:
    keys = [f"encounter:{e}" for e, _, _ in content.encounters]
    name = pick(cov, keys, 1, rng)[0].split(":", 1)[1]
    return next(x for x in content.encounters if x[0] == name)


def placed_encounter(content: Content, cov: dict, rng: random.Random, act1_seeds: dict | None) -> tuple:
    """(encounter, act name, kind, setup prefix, seed or None) of a room no real run holds the setup out of.

    Only an act-1 room is met in the run's own act-1 variant, which the seed decides: such a room gets a seed known to
    start in that variant (act1_seeds, seed -> variant), and when none is known an act 2 or 3 room is drawn instead.
    An act 2 or 3 room gets the setup's `act` command, as a run that has reached that act."""
    for _ in range(40):
        enc, act, kind = encounter_key(content, cov, rng)
        if ACT_NUMBER[act] > 1:
            return enc, act, kind, [{"cmd": "act", "index": ACT_NUMBER[act]}], None
        known = sorted(sd for sd, v in (act1_seeds or {}).items() if v == act)
        if known:
            return enc, act, kind, [], rng.choice(known)
    later = [x for x in content.encounters if ACT_NUMBER[x[1]] > 1]
    enc, act, kind = rng.choice(later)
    return enc, act, kind, [{"cmd": "act", "index": ACT_NUMBER[act]}], None


def coverage_spec(content: Content, cov: dict, rng: random.Random, n: int, act1_seeds: dict | None = None) -> dict:
    enc, act, kind, prefix, seed = placed_encounter(content, cov, rng, act1_seeds)
    cats = {
        "relic": [f"relic:{r}" for r in content.relics],
        "potion": [f"potion:{p}" for p in content.potions],
        "power": [f"power:{p}@player" for p in content.player_powers] + [f"power:{p}@enemy" for p in content.enemy_powers],
        "card+": [f"card+:{c}" for c in content.cards if c in content.upgradable],
    }
    chosen = []
    for _ in range(rng.randint(1, 3)):
        cat = rng.choice(sorted(cats))
        got = pick(cov, [k for k in cats[cat] if k not in chosen], 1, rng)
        chosen += got
    before, after = [], []
    for key in chosen:
        kind_, ident = key.split(":", 1)
        if kind_ == "relic":
            before.append({"cmd": "relic", "id": ident})
        elif kind_ == "potion":
            before.append({"cmd": "potion", "id": ident})
        elif kind_ == "card+":
            before.append({"cmd": "card", "id": ident, "upgraded": True})
        else:
            power, side = ident.split("@")
            after.append(_power_item(rng, power, "player" if side == "player" else "enemy:0"))
    setup = prefix + before + [{"cmd": "fight", "id": enc}] + after
    return _seeded(_spec(rng, n, setup, "coverage", act, kind), seed)


def dense_spec(content: Content, cov: dict, rng: random.Random, n: int, act1_seeds: dict | None = None) -> dict:
    enc, act, kind, prefix, seed = placed_encounter(content, cov, rng, act1_seeds)
    relics = [k.split(":", 1)[1] for k in pick(cov, [f"relic:{r}" for r in content.relics], rng.randint(10, 20), rng)]
    cards = []
    for c in rng.sample(content.cards, rng.randint(10, 20)):
        cards.append({"cmd": "card", "id": c, "upgraded": c in content.upgradable and rng.random() < 0.35})
    mixed = [{"cmd": "relic", "id": r} for r in relics] + cards
    rng.shuffle(mixed)  # relics that act on the deck (eggs, Whetstone) meet cards on both sides of them
    slots = 2 + (2 if "POTION_BELT" in relics else 0)  # A10's two slots (TightBelt), Potion Belt's two more
    potions = [{"cmd": "potion", "id": k.split(":", 1)[1]}
               for k in pick(cov, [f"potion:{p}" for p in content.potions], slots, rng)]
    after = [_power_item(rng, p, "player") for p in rng.sample(content.player_powers, rng.randint(0, 3))]
    for slot in range(2):
        after += [_power_item(rng, p, f"enemy:{slot}") for p in rng.sample(content.enemy_powers, rng.randint(0, 2))]
    setup = prefix + mixed + potions + [{"cmd": "fight", "id": enc}] + after
    return _seeded(_spec(rng, n, setup, "dense", act, kind), seed)


# ---------------------------------------------------------------- choice edges (--mode choices)

CHOICE_FOLLOW = "TransformSelect"  # a relic whose pickup opens a card screen the setup mirror answers
CHOICE_CARDS_FALLBACK = ("ARMAMENTS", "BRAND", "BURNING_PACT", "DUAL_WIELD", "HEADBUTT", "TRUE_GRIT")
SELECT_CMD = "cmd:CardSelectCmd."  # lab_groups.model_features: a model whose class calls a card-screen command
EDGE_NAMES = ("none", "one", "max", "last", "unusual")
# random: each spec draws one edge and keeps it for every screen of its fight (the shrinker replays subsets of a spec,
# and an edge that depends on how many screens came before would move when an item is dropped); cycle: the driver
# walks through the five, one per screen.
EDGE_MODES = ("random", "cycle", *EDGE_NAMES)


class ChoicePools:
    """Which relics, cards and potions open a card-choice screen. Relics by the follow-up the setup info gives their
    pickup; cards and potions by the game commands their model calls (features: lab_groups.model_features, the
    lab's cache/model_features.json), else a short list of the cards every Ironclad deck knows."""

    def __init__(self, content: Content, features: dict | None = None):
        feats = features or {}

        def opens(kind: str, ident: str) -> bool:
            return any(str(f).startswith(SELECT_CMD) for f in feats.get(f"{kind}:{ident}", ()))

        self.relics = [r for r in content.relics if content.relic_follow.get(r) == CHOICE_FOLLOW]
        self.cards = [c for c in content.cards if c not in BASICS and opens("card", c)] \
            or [c for c in CHOICE_CARDS_FALLBACK if c in set(content.cards)]
        self.potions = [p for p in content.potions if opens("potion", p)]


def choice_spec(content: Content, pools: ChoicePools, cov: dict, rng: random.Random, n: int, edges: str = "random",
                act1_seeds: dict | None = None) -> dict:
    """A setup fight built to open card screens, answered with the driver's edge answers (spec "choice_edges": none,
    one, the most, the last eligible, the most unusual card). Small decks, so a screen may offer fewer cards than
    it allows; relics that open a screen on pickup land among the deck's cards; screen-opening cards also start in
    the hand, so they are played early; screen-opening potions fill the belt first."""
    enc, act, kind, prefix, seed = placed_encounter(content, cov, rng, act1_seeds)
    relics = rng.sample(pools.relics, min(rng.randint(1, 3), len(pools.relics)))
    plain = [r for r in content.relics if r not in pools.relics]
    relics += rng.sample(plain, min(rng.randint(0, 2), len(plain)))
    held = rng.sample(pools.cards, min(rng.randint(1, 3), len(pools.cards)))
    fill = [c for c in content.cards if c not in BASICS and c not in held]
    cards = [{"cmd": "card", "id": c, "upgraded": c in content.upgradable and rng.random() < 0.35}
             for c in held + rng.sample(fill, min(rng.randint(0, 8), len(fill)))]
    basics = {"STRIKE_IRONCLAD": rng.randint(1, 5), "DEFEND_IRONCLAD": rng.randint(0, 4), "BASH": rng.randint(0, 1),
              "ASCENDERS_BANE": rng.randint(0, 1)}
    deck = [{"id": b, "upgraded": False} for b in BASICS for _ in range(basics[b])]
    mixed = [{"cmd": "relic", "id": r} for r in relics] + cards
    rng.shuffle(mixed)
    slots = 2 + (2 if "POTION_BELT" in relics else 0)
    belt = rng.sample(pools.potions, min(rng.randint(0, 2), len(pools.potions)))
    belt += rng.sample([p for p in content.potions if p not in belt], 1) if len(belt) < slots and rng.random() < 0.5 else []
    in_hand = [{"cmd": "card", "id": c, "pile": "Hand", "upgraded": c in content.upgradable and rng.random() < 0.35} for c in held]
    setup = prefix + [{"cmd": "deck_set", "cards": deck}] + mixed + [{"cmd": "potion", "id": p} for p in belt[:slots]]
    setup += [{"cmd": "fight", "id": enc}] + in_hand
    spec = _seeded(_spec(rng, n, setup, "choices", act, kind), seed)
    spec["choice_edges"] = rng.choice(EDGE_NAMES) if edges == "random" else edges
    return spec


# ---------------------------------------------------------------- aimed fights (--mode targets)

ACT_NUMBER = {"OVERGROWTH": 1, "UNDERDOCKS": 1, "HIVE": 2, "GLORY": 3}
ROOM_OF_KIND = {"weak": "monster", "normal": "monster", "elite": "elite", "boss": "boss"}
BASICS = ("STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH", "ASCENDERS_BANE")
STARTER_DECK = {"STRIKE_IRONCLAD": 5, "DEFEND_IRONCLAD": 4, "BASH": 1, "ASCENDERS_BANE": 1}
STARTERS = frozenset(("BURNING_BLOOD", *BASICS))  # every run holds them, so their meet share dwarfs any gap
GAP_DOWNWEIGHT = 20.0  # --aim gaps divides a starter target's weight by this
AIMS = ("all", "bugs", "gaps")


def names_backlog(target: dict) -> bool:
    """Whether an open backlog item names this target (lab_targets writes backlog_items, and a reason line before that)."""
    return bool(target.get("backlog_items")) or any(str(r).startswith("open backlog items name it") for r in target.get("reasons", []))


def _hist_draw(rng: random.Random, hist: dict, default: int) -> int:
    """A value drawn from a {value: count} histogram (its keys may be strings)."""
    items = [(int(k), int(v)) for k, v in (hist or {}).items() if int(v) > 0]
    if not items:
        return default
    x = rng.random() * sum(v for _, v in items)
    for k, v in sorted(items):
        x -= v
        if x <= 0:
            return k
    return items[-1][0]


def _weighted(rng: random.Random, pairs: list) -> object | None:
    pairs = [(x, w) for x, w in pairs if w > 0]
    if not pairs:
        return None
    x = rng.random() * sum(w for _, w in pairs)
    for item, w in pairs:
        x -= w
        if x <= 0:
            return item
    return pairs[-1][0]


def _sample(rng: random.Random, pairs: list, k: int) -> list:
    """k distinct items, each drawn by weight from what is left."""
    pool, out = [p for p in pairs if p[1] > 0], []
    for _ in range(min(k, len(pool))):
        got = _weighted(rng, pool)
        out.append(got)
        pool = [p for p in pool if p[0] != got]
    return out


class Aim:
    """What targets mode draws from: the lab's weighted targets (only those the setup can reach) and the
    Codex profile's aggregate counts per act. One Aim makes one batch: it keeps the batch's seeds apart."""

    def __init__(self, targets_doc: dict, profile_doc: dict | None, content: Content, act1_seeds: dict | None = None,
                 aim: str = "all"):
        from tools.fidelity.lab_targets import Profile  # noqa: PLC0415
        self.prof = Profile(profile_doc)
        self.content = content
        self.aim = aim
        self.act1_seeds = dict(act1_seeds or {})
        self.used, self.borrowed = set(), set()  # this batch's seeds; those taken from the cache
        self.enc = {e: (act, kind) for e, act, kind in content.encounters}
        self.cards, self.relics, self.potions = set(content.cards), set(content.relics), set(content.potions)
        self.card_power = {c + "_POWER": c for c in content.cards}
        self.targets, self.skipped = [], collections.Counter()
        for t in targets_doc.get("targets", []):
            w = float(t.get("weight", 0))
            if w > 0 and self.reachable(t["kind"], t["id"]):
                if aim == "bugs" and not names_backlog(t):
                    self.skipped["not_in_backlog"] += 1  # bugs: only what an open backlog item names
                    continue
                if aim == "gaps" and t["id"] in STARTERS:
                    w /= GAP_DOWNWEIGHT  # gaps: the starter items every run holds are met anyway
                self.targets.append((t["kind"], t["id"], w))
            else:
                self.skipped[t["kind"]] += 1

    def reachable(self, kind: str, ident: str) -> bool:
        if kind == "card":
            return ident in self.cards
        if kind == "relic":
            return ident in self.relics
        if kind == "potion":
            return ident in self.potions
        if kind == "encounter":
            return ident in self.enc
        if kind == "enemy":
            return any(ident in self.prof.encounter_monsters(e) for e in self.enc)
        if kind == "power":
            return (ident in self.content.player_powers or ident in self.content.enemy_powers
                    or ident in self.card_power and self.card_power[ident] in self.cards)
        if kind == "act":
            return ident in ACT_NUMBER
        return False

    def encounter_weight(self, e: str) -> float:
        act, kind = self.enc[e]
        return float(self.prof.act(ACT_NUMBER[act]).get("encounters", {}).get(ROOM_OF_KIND[kind], {}).get(e, 0))

    def pick_encounter(self, rng: random.Random, act: int, act_name: str | None = None) -> str:
        """An encounter of the act: a room by how often real runs enter it, then one of its encounters by count."""
        rooms = self.prof.act(act).get("rooms", {}) or {"monster": 7, "elite": 2, "boss": 1}
        room = _weighted(rng, sorted(rooms.items())) or "monster"
        names = [act_name] if act_name else [n for n, a in ACT_NUMBER.items() if a == act]
        cands = [e for e, (an, kind) in sorted(self.enc.items()) if an in names and ROOM_OF_KIND[kind] == room]
        got = _weighted(rng, [(e, self.encounter_weight(e)) for e in cands])
        return got or rng.choice(cands or sorted(e for e, (an, _) in self.enc.items() if an in names))

    def held_item(self, kind: str, ident: str) -> str | None:
        """The profile key of a held target ("card:X"), the power card for a card's power."""
        if kind in ("card", "relic", "potion"):
            return f"{kind}:{ident}"
        if kind == "power" and self.card_power.get(ident) in self.cards:
            return f"card:{self.card_power[ident]}"
        return None

    def place(self, rng: random.Random, kind: str, ident: str) -> tuple[int, str, str]:
        """(act number, act name, encounter) for a target."""
        if kind == "encounter":
            act_name = self.enc[ident][0]
            return ACT_NUMBER[act_name], act_name, ident
        if kind == "enemy":
            cands = [(e, max(1.0, self.encounter_weight(e))) for e in sorted(self.enc) if ident in self.prof.encounter_monsters(e)]
            e = _weighted(rng, cands)
            return ACT_NUMBER[self.enc[e][0]], self.enc[e][0], e
        if kind == "act":
            act = ACT_NUMBER[ident]
            return act, ident, self.pick_encounter(rng, act, ident)
        held = self.held_item(kind, ident)
        shares = self.prof.held_acts(*held.split(":", 1)) if held else {}
        act = _weighted(rng, [(a, s * max(1, self.prof.act(a).get("snapshots", 0))) for a, s in sorted(shares.items())]) or 1
        act_name = None
        if act == 1:
            names = self.prof.act(1).get("act_names", {}) or {"OVERGROWTH": 1, "UNDERDOCKS": 1}
            act_name = _weighted(rng, sorted((n, v) for n, v in names.items() if ACT_NUMBER.get(n) == 1)) or "OVERGROWTH"
        e = self.pick_encounter(rng, act, act_name)
        return act, self.enc[e][0], e

    def seed(self, rng: random.Random, act: int, act_name: str) -> str:
        """A seed whose act 1 is act_name when one is known (act-1 fights are met in the seed's own act 1).

        Each seed is used once per batch (one Aim): cached seeds are drawn without replacement and a fresh one
        is redrawn on a clash, since everything after the play step (shrink's minimal packs, traces, context
        packs, lab_groups) is keyed by seed. A cached seed is noted in borrowed: it names another fight too."""
        got = None
        if act == 1:
            known = sorted(s for s, v in self.act1_seeds.items() if v == act_name and s not in self.used)
            if known:  # a fresh seed starts in either act-1 variant: half the draws would be a cross_act setup
                got = rng.choice(known)
                self.borrowed.add(got)
        while got is None or got in self.used:
            got = _seed(rng)
        self.used.add(got)
        return got

    def draws(self, rng: random.Random, kind: str, pool: list, act: int, k: int, anchor: str | None) -> list:
        """k items of one kind (card, relic, potion) held at the act: half how often real runs hold each, half how
        often it is held together with the anchor (the target)."""
        snaps = max(1, int(self.prof.act(act).get("snapshots", 0)))
        held_anchor = self.prof.held(*anchor.split(":", 1), act) if anchor else 0
        out = []
        for x in pool:
            p = self.prof.held(kind, x, act) / snaps
            if anchor and held_anchor >= 5 and f"{kind}:{x}" != anchor:
                p = 0.5 * p + 0.5 * self.prof.pair(act, f"{kind}:{x}", anchor) / held_anchor
            out.append((x, p))
        return _sample(rng, out, k)


def aimed_spec(aim: Aim, rng: random.Random, n: int) -> tuple[dict, dict]:
    """One realistic spec aimed at one target, and what it aimed at."""
    kind, ident, w = _weighted(rng, [((k, i, w), w) for k, i, w in aim.targets])
    act, act_name, enc = aim.place(rng, kind, ident)
    prof, content = aim.prof, aim.content
    doc = prof.act(act)
    anchor = aim.held_item(kind, ident)
    basics = {b: _hist_draw(rng, (doc.get("basics") or {}).get(b), STARTER_DECK[b]) for b in BASICS}
    size = _hist_draw(rng, doc.get("deck_size"), 10 + 3 * act)
    want_cards = max(0, size - sum(basics.values()))
    pool = [c for c in content.cards if c not in BASICS]
    chosen = aim.draws(rng, "card", pool, act, want_cards, anchor)
    if anchor and anchor.startswith("card:") and anchor[5:] not in chosen:
        chosen = [anchor[5:]] + chosen[: max(0, want_cards - 1)]
    cards = []
    for c in chosen:
        copies, presence = doc.get("copies", {}).get(c, 0), doc.get("cards", {}).get(c, 0)
        n_copies = 1 + (1 if presence and rng.random() < min(1.0, copies / presence - 1) else 0)
        up_rate = doc.get("upgraded", {}).get(c, 0) / copies if copies else 0.3
        for _ in range(n_copies):
            cards.append({"cmd": "card", "id": c, "upgraded": c in content.upgradable and rng.random() < up_rate})
    relics = aim.draws(rng, "relic", content.relics, act, _hist_draw(rng, doc.get("relic_count"), act * 3), anchor)
    if kind == "relic" and ident not in relics:
        relics = [ident] + relics[:-1] if relics else [ident]
    slots = 2 + (2 if "POTION_BELT" in relics else 0)
    potions = aim.draws(rng, "potion", content.potions, act, min(slots, _hist_draw(rng, doc.get("potion_count"), 1)), anchor)
    if kind == "potion" and ident not in potions:
        potions = [ident] + potions[: max(0, slots - 1)]
    mixed = [{"cmd": "relic", "id": r} for r in relics] + cards
    rng.shuffle(mixed)  # relics that act on the deck meet cards on both sides of them, as over a real run
    deck = [{"id": b, "upgraded": False} for b in BASICS for _ in range(basics[b])]
    after = []
    if kind == "power" and not anchor:
        target = "player" if ident in content.player_powers else "enemy:0"
        after.append(_power_item(rng, ident, target))
    setup = ([{"cmd": "act", "index": act}] if act > 1 else []) + [{"cmd": "deck_set", "cards": deck}] + mixed
    setup += [{"cmd": "potion", "id": p} for p in potions] + [{"cmd": "fight", "id": enc}] + after
    spec = _spec(rng, n, setup, "targets", aim.enc[enc][0], aim.enc[enc][1])
    spec["seed"] = aim.seed(rng, act, aim.enc[enc][0])
    return spec, {"seed": spec["seed"], "target": f"{kind}:{ident}", "weight": w, "act": act, "encounter": enc,
                  "seed_from": "cache" if spec["seed"] in aim.borrowed else "fresh"}


def act1_seed_cache(path: Path, games: list | None = None, scan: list | None = None) -> dict:
    """seed -> the act-1 variant its run starts in, from recorded setup fights (their fight.current_act)."""
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
        for f in scan or []:
            try:
                got = json.loads(Path(f).read_text(encoding="utf-8")).get("results", [])
            except (OSError, ValueError):
                continue
            games = list(games or []) + got
    for g in games or []:
        cur = (g or {}).get("fight", {}) or {}
        if g and g.get("seed") and ACT_NUMBER.get(cur.get("current_act")) == 1:
            cache[g["seed"]] = cur["current_act"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")
    return cache


TRACE_CHUNK = 48  # fights per trace_divergent call when the work is cut into safe points
TRACE_SERVERS = 3  # the default cap on game servers for traced re-runs (the lab passes more)


def trace_order(games: list, emus: list, rea: dict, verdicts: list, nm: dict) -> list[str]:
    """The seeds trace_divergent can trace, plausible setups first and the earliest divergence first."""
    import trace_divergent as td  # noqa: PLC0415  (referee/trace_divergent.py)
    found = td.targets(games, emus, rea, nm)
    flagged = {v.get("seed"): bool(v.get("flags")) for v in verdicts}
    return sorted(found, key=lambda s: (flagged.get(s, False), found[s][1], s))


def trace_batch(name: str, game: str, servers: int, n: int, log: Path, seeds: list[str] | None = None,
                deadline: float | None = None, chunk: int = TRACE_CHUNK, max_servers: int = TRACE_SERVERS) -> str:
    """referee/trace_divergent.py on the batch's diverging fights: their first differing event, into the packs.

    With `seeds` (best first) at most `n` of them are traced in calls of `chunk` fights, each call a safe point; no
    new call starts after `deadline` (time.time()), and what earlier calls traced stays."""
    out = JOBS / f"{name}.out.json"
    cmd = [sys.executable, str(ROOT / "referee" / "trace_divergent.py"), str(out), "--game", game,
           "--specs", str(JOBS / f"{name}.json"), "--emu", str(JOBS / f"{name}.emu.json"),
           "--reanchor", str(JOBS / f"{name}.reanchor.json"), "--packs", str(JOBS / f"{name}.packs"),
           "--servers", str(max(1, min(max_servers, servers)))]
    seeds = seeds[:n] if seeds is not None else None
    parts = [seeds[i:i + chunk] for i in range(0, len(seeds), chunk)] if seeds is not None else [None]
    rc, stopped = 0, ""
    with open(log, "w", encoding="utf-8") as fh:
        for k, part in enumerate(parts):
            if k and deadline is not None and time.time() >= deadline:
                stopped = f", stopped at the time limit after {k * chunk} of {len(seeds or [])}"
                break
            args = ["--max", str(n)] if part is None else ["--max", str(len(part))] + [x for s in part for x in ("--seed", s)]
            p = subprocess.run(cmd + args, stdout=fh, stderr=subprocess.STDOUT, text=True, check=False,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            rc = rc or p.returncode
    tdir = JOBS / f"{name}.packs" / "traces"
    traced = len(list(tdir.glob("*.trace.json"))) if tdir.is_dir() else 0
    idx = tdir / "index.md"
    return f"traces: {traced} fights (exit {rc}){stopped}, {idx.relative_to(JOBS).as_posix() if idx.exists() else 'none'}"


def _spec(rng: random.Random, n: int, setup: list, mode: str, act: str, kind: str) -> dict:
    return {"kind": "combat", "seed": _seed(rng), "character": "IRONCLAD", "ascension": 10, "neow": n % 3,
            "chooser_seed": 7000 + n, "max_actions": 150, "setup": setup, "use_potions": True,
            "tag": f"{mode}:{act}:{kind}"}


def _seeded(spec: dict, seed: str | None) -> dict:
    if seed:
        spec["seed"] = seed
    return spec


def spec_flags(spec: dict, content: Content, game: dict | None = None) -> list[str]:
    """Why no real run could hold this setup; empty when it is plausible."""
    flags = set()
    ids = [(it["cmd"], it.get("id")) for it in spec.get("setup", [])]
    if any((c == "card" and i in content.other_cards) or (c == "relic" and i in content.other_relics)
           or (c == "potion" and i in content.other_potions) or (c == "power" and i in OTHER_CHARACTER_POWERS)
           for c, i in ids):
        flags.add("cross_character")
    relics = [i for c, i in ids if c == "relic"]
    if len(relics) != len(set(relics)):
        flags.add("duplicate_relic")
    fight = (game or {}).get("fight") or {}
    if fight.get("kind") in ("elite", "boss"):
        flags.add("debug_room")  # an elite or a boss as the act's first room
    if fight and fight.get("act") != fight.get("current_act"):
        flags.add("cross_act")
    return sorted(flags)


# ---------------------------------------------------------------- play, replay, judge

def play(name: str, specs: list, game: str, servers: int, timeout: float) -> tuple[list, dict]:
    """Specs through serve.py; results in spec order and serve.py's summary."""
    job, out = JOBS / f"{name}.json", JOBS / f"{name}.out.json"
    run_dir = Path(game).parent / "serve" / f"setup-{name}"
    job.write_text(json.dumps({"runs": specs}), encoding="utf-8")
    p = subprocess.run([sys.executable, str(ROOT / "referee" / "serve.py"), "--game", game, "--specs", str(job),
                        "--servers", str(max(1, min(servers, len(specs)))), "--out", str(out), "--run-dir", str(run_dir)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)
    try:
        rep = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        rep = {"tail": p.stdout[-800:]}
    rep["exit"] = p.returncode
    got = json.loads(out.read_text(encoding="utf-8"))["results"] if out.exists() else []
    merged = [got[n] if n < len(got) and got[n] else {"seed": s["seed"], "error": "server produced no result", "steps": []}
              for n, s in enumerate(specs)]
    return merged, rep


def emudump(mode: str, results: list, path: Path, procs: int = 4) -> dict:
    """EmuDump fight or reanchor over results, split over processes (one emulator per process)."""
    k = max(1, min(procs, len(results)))
    size = -(-len(results) // k)
    parts = [results[i:i + size] for i in range(0, len(results), size)]
    running = []
    for i, part in enumerate(parts):
        src, dst = path.with_suffix(f".p{i}.in.json"), path.with_suffix(f".p{i}.json")
        src.write_text(json.dumps({"results": part}), encoding="utf-8")
        running.append((src, dst, len(part), subprocess.Popen(["dotnet", str(EMU_DLL), mode, str(src)],
                                                             stdout=open(dst, "w", encoding="utf-8"))))
    doc: dict = {}
    out: list = []
    for src, dst, n_in, p in running:
        p.wait()
        p.stdout and p.stdout.close()
        part = json.loads(dst.read_text(encoding="utf-8")) if dst.stat().st_size else {"results": []}
        doc = doc or {k_: v for k_, v in part.items() if k_ != "results"}
        got = part.get("results", [])
        out += got + [{"error": "emulator produced no result", "steps": []}] * (n_in - len(got))
        src.unlink()
        dst.unlink()
    doc["results"] = out
    path.write_text(json.dumps(doc), encoding="utf-8")
    return doc


def first_divergence(g: dict, e: dict, nm: dict):
    """(step, field group, action) of the first differing step, or None if every compared step matches."""
    from tools.fidelity import fight_compare as fc  # noqa: PLC0415
    from tools.fidelity.reanchor_compare import group  # noqa: PLC0415
    fd = fc.first_difference(g, e, nm)
    if fd is None:
        return None
    i, d = fd
    path = d[0][0]
    return i, group(path) if path.startswith(".") else path, g["steps"][i].get("action") if i < len(g["steps"]) else None, d


def upgraded_plays(g: dict, matched_steps: set) -> list[str]:
    """card+:X for each re-anchored step that played an upgraded card and matched."""
    out = []
    steps = g.get("steps", [])
    for j in range(1, len(steps)):
        a = steps[j].get("action") or {}
        if a.get("kind") != "play" or j not in matched_steps:
            continue
        hand = steps[j - 1]["state"].get("hand", [])
        hi = a.get("hand_index", -1)
        if 0 <= hi < len(hand) and hand[hi][1]:
            out.append("card+:" + a["card"])
    return out


def judge(specs: list, games: list, emus: list, rea: dict, nm: dict, content: Content) -> list[dict]:
    """One verdict per fight: identical or where it first diverges, mirror gaps, re-anchored steps."""
    from tools.fidelity import reanchor_compare as rc  # noqa: PLC0415
    from tools.fidelity.setup_compare import compare  # noqa: PLC0415
    keys = rc.keys_of(rea)
    verdicts = []
    for spec, g, e, r in zip(specs, games, emus, rea["results"]):
        v = {"seed": spec["seed"], "tag": spec.get("tag"), "flags": spec_flags(spec, content, g)}
        if g.get("error") or e.get("error") or not g.get("steps"):
            v["error"] = str(g.get("error") or e.get("error") or "no steps")[:300]
            verdicts.append(v)
            continue
        sc = compare(g, e)
        v["mirror_gap"] = sc["unsupported"]
        if sc.get("gap_why"):
            v["mirror_gap_why"] = {str(i): m for i, m in sc["gap_why"].items()}  # item -> the mirror's message (lab_groups reads it)
        v["setup_first"] = sc["first"]
        fd = first_divergence(g, e, nm)
        v["identical"] = fd is None
        if fd:
            v["step"], v["field"], act, d = fd
            v["card"] = "start" if act is None else ("END_TURN" if act.get("kind") == "end_turn"
                                                    else ("POTION:" + act.get("potion", "?") if act.get("kind") == "potion" else act.get("card")))
            v["diff"] = [[p, str(x)[:120], str(y)[:120]] for p, x, y in d[:3]]
        an = rc.analyse(g, r, nm, "stop", keys)
        v["transitions"] = an["transitions"]
        v["verified"] = sum(1 for s in an["steps"] if s["j"] > 0 and s["verdict"] == "match")
        v["primary"] = [(s["j"], s["card"], s["paths"][:2]) for s in an["steps"] if s["primary"] and s["j"] > 0][:5]
        v["upgraded_plays"] = upgraded_plays(g, {s["j"] for s in an["steps"] if s["verdict"] == "match"})
        verdicts.append(v)
    return verdicts


def summarise(verdicts: list, minutes: float, rep: dict) -> list[str]:
    ok = [v for v in verdicts if not v.get("error")]
    same = sum(1 for v in ok if v.get("identical"))
    gaps = [v for v in ok if v.get("mirror_gap")]
    div = [v for v in ok if not v.get("identical")]
    vt = sum(v.get("verified", 0) for v in ok)
    trans = sum(v.get("transitions", 0) for v in ok)
    lines = [f"{len(verdicts)} fights: {same} identical, {len(div)} diverging ({len(gaps)} with a mirror gap), "
             f"{len(verdicts) - len(ok)} errors; re-anchored {vt} of {trans} steps match = {vt / max(1, len(ok)):.1f} VT/fight; "
             f"{minutes:.1f} min, {rep.get('s_per_fight_per_server')} s/fight/server, network {rep.get('network_connections')}, "
             f"real profile touched {rep.get('real_profile_touched')}"
             + (f", WER REPORTS {rep.get('wer_reports')}" if rep.get("wer_reports") else "")]
    groups = collections.defaultdict(list)
    for v in div:
        groups[f"{v.get('field')} after {v.get('card')}"].append(v)
    for k, vs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        plain = sum(1 for v in vs if not v["flags"])
        lines.append(f"  {k}: {len(vs)} ({plain} plausible) e.g. {vs[0]['seed']} step {vs[0]['step']} {vs[0].get('diff', [])[:1]}")
    for v in [v for v in verdicts if v.get("error")][:5]:
        lines.append(f"  error {v['seed']}: {v['error'][:200]}")
    for v in gaps[:5]:
        lines.append(f"  mirror gap {v['seed']}: items {v['mirror_gap']}")
    return lines


def load_batch(name: str) -> tuple[list, list, list, dict]:
    """A finished batch.py batch in JOBS: its specs, the game's results, the emulator's replay and re-anchored replay."""
    def read(ext: str) -> dict:
        return json.loads((JOBS / f"{name}{ext}").read_text(encoding="utf-8"))
    specs, games, emus, rea = read(".json")["runs"], read(".out.json")["results"], read(".emu.json")["results"], read(".reanchor.json")
    sizes = (len(specs), len(games), len(emus), len(rea.get("results", [])))
    if len(set(sizes)) != 1:
        raise SystemExit(f"{name}: specs, game, emulator and re-anchor results differ in number {sizes}; not a finished batch")
    return specs, games, emus, rea


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=("dense", "coverage", "targets", "choices", "analyze", "runs", "hypotheses", "retest"), required=True)
    ap.add_argument("--fights", type=int, default=60)
    ap.add_argument("--start", type=int, default=0, help="spec stream offset")
    ap.add_argument("--game", default=GAME)
    ap.add_argument("--servers", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=5400)
    ap.add_argument("--shrink", type=int, default=0, help="shrink up to this many diverging fights")
    ap.add_argument("--shrink-budget", type=int, default=400, help="game fights the shrinker may spend in all")
    ap.add_argument("--shrink-cap", type=int, default=48, help="game fights the shrinker may spend on one diverging fight")
    ap.add_argument("--shrink-minutes", type=float, default=None, help="start no new shrink round after this long")
    ap.add_argument("--make-only", action="store_true", help="write the job file and stop")
    ap.add_argument("--targets", type=Path, help="targets mode: tools/fidelity/lab_targets.py's targets.json (in the lab)")
    ap.add_argument("--aim", choices=AIMS, default="all",
                    help="targets mode: all targets; bugs = only those an open backlog item names; gaps = starter items weigh 1/20")
    ap.add_argument("--edges", choices=EDGE_MODES, default="random", help="choices mode: the driver's choice_edges answers")
    ap.add_argument("--act1-seeds", type=Path, help="choices, dense, coverage: the lab's cache/act1_seeds.json (seed -> its act-1 variant)")
    ap.add_argument("--features", type=Path, help="choices mode: lab_groups' model_features.json (which models open card screens)")
    ap.add_argument("--decompiled", type=Path, default=Path(r"D:\spireline\referee\decompiled"),
                    help="choices mode: the decompile model_features is read from when --features is missing or stale")
    ap.add_argument("--jobs", type=Path, help="write every output here instead of jobs-setup")
    ap.add_argument("--retest", type=Path, help="retest mode: a folder with pending.json (tools/lab/retest.py); every group's spec is played")
    ap.add_argument("--hypotheses", type=Path, help="hypotheses mode: a folder of <source>/*.json setups (tools/lab/hypotheses.py); every one is played")
    ap.add_argument("--runs-from", type=Path, help="runs mode: a runs.py batch's <name>.runs.json (its <name>.out.json beside it)")
    ap.add_argument("--trace", type=int, default=None, help="trace up to N diverging fights (default: 40 in targets mode, else 0)")
    ap.add_argument("--trace-servers", type=int, default=None, help="game servers for the traced re-runs (at most 8; default: --servers, at most 3)")
    ap.add_argument("--trace-minutes", type=float, default=None, help="start no new trace call after this long")
    a = ap.parse_args()
    if a.mode == "targets" and not a.targets:
        ap.error("--mode targets needs --targets")
    if a.mode == "retest" and not a.retest:
        ap.error("--mode retest needs --retest")
    if a.mode == "hypotheses" and not a.hypotheses:
        ap.error("--mode hypotheses needs --hypotheses")
    if a.mode == "runs" and not a.runs_from:
        ap.error("--mode runs needs --runs-from")
    global JOBS  # noqa: PLW0603  (--jobs moves every output of this run)
    if a.jobs:
        JOBS = a.jobs.resolve()

    JOBS.mkdir(parents=True, exist_ok=True)
    content = load_content()
    from tools.fidelity.fight_compare import names  # noqa: PLC0415
    nm = names(json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8")))
    cov_path = JOBS / "coverage-setup.json"
    cov = json.loads(cov_path.read_text(encoding="utf-8")) if cov_path.exists() else {"items": {}}
    rng = random.Random(f"{a.mode}:{a.start}")
    aims, act1_path, repro = [], None, None
    analysing = a.mode == "analyze"
    if analysing:
        specs, games, emus, rea = load_batch(a.name)
    elif a.mode == "targets":
        tdoc = json.loads(a.targets.read_text(encoding="utf-8"))
        lab = a.targets.resolve().parent
        prof_path = lab / tdoc.get("profile", "cache/codex_profile.json")
        prof = json.loads(prof_path.read_text(encoding="utf-8")) if prof_path.exists() else None
        act1_path = lab / "cache" / "act1_seeds.json"
        known = act1_seed_cache(act1_path, scan=sorted(DEFAULT_JOBS.glob("*.out.json")))
        aim = Aim(tdoc, prof, content, known, a.aim)
        if not aim.targets:
            print("no reachable target in", a.targets, f"(aim {a.aim})")
            return 1
        specs = []
        for i in range(a.fights):
            s, m = aimed_spec(aim, rng, a.start * 1000 + i)
            specs.append(s)
            aims.append(m)
        (JOBS / f"{a.name}.aims.json").write_text(json.dumps({"targets": a.targets.name, "aim": a.aim, "skipped": dict(aim.skipped),
                                                              "profile_runs": aim.prof.runs, "aims": aims}, indent=1),
                                                  encoding="utf-8")
    elif a.mode == "choices":
        known1 = act1_seed_cache(a.act1_seeds) if a.act1_seeds else {}
        from tools.fidelity.lab_groups import model_features  # noqa: PLC0415
        feats = model_features(a.decompiled, a.features) if a.features or a.decompiled.is_dir() else {}
        pools = ChoicePools(content, feats)
        if not (pools.relics or pools.cards or pools.potions):
            print("no relic, card or potion that opens a card screen is reachable")
            return 1
        specs = [choice_spec(content, pools, cov, rng, a.start * 1000 + i, a.edges, known1) for i in range(a.fights)]
    elif a.mode == "retest":
        from tools.lab import retest as rt  # noqa: PLC0415
        try:
            specs = rt.to_specs(json.loads((a.retest / "pending.json").read_text(encoding="utf-8")))
        except (OSError, ValueError):
            specs = []
        if not specs:
            print("retest: no group waits for a retest in", a.retest)
            (JOBS / f"{a.name}.verdicts.json").write_text("[]", encoding="utf-8")
            return 0
    elif a.mode == "hypotheses":
        from tools.lab import hypotheses as hyp  # noqa: PLC0415
        setups, problems = hyp.load_setups(a.hypotheses)
        for p_ in problems:
            print("hypotheses:", p_)
        specs = hyp.to_specs(setups)
        if not specs:
            print("hypotheses: no testable setup in", a.hypotheses)
            (JOBS / f"{a.name}.verdicts.json").write_text("[]", encoding="utf-8")  # the batch's output, empty: nothing to judge
            return 0
    elif a.mode == "runs":
        import run_repro  # noqa: PLC0415  (referee/run_repro.py)
        src = a.runs_from.resolve()
        prefix = src.with_name(src.name.removesuffix(".runs.json"))
        specs, skipped = run_repro.specs_from(*run_repro.load(prefix), limit=a.fights)
        repro = {"from": prefix.name, "gaps": list(run_repro.STATE_GAPS), "skipped": skipped,
                 "runs": [{"seed": x["seed"], **x["run_repro"]} for x in specs]}
        (JOBS / f"{a.name}.repro.json").write_text(json.dumps(repro, indent=1), encoding="utf-8")
        if not specs:
            print(f"no run of {prefix.name} diverges first in a fight ({len(skipped)} left out)")
            return 0
    else:
        make = dense_spec if a.mode == "dense" else coverage_spec
        known1 = act1_seed_cache(a.act1_seeds) if a.act1_seeds else {}
        specs = [make(content, cov, rng, a.start * 1000 + i, known1) for i in range(a.fights)]
    if a.make_only:
        (JOBS / f"{a.name}.json").write_text(json.dumps({"runs": specs}), encoding="utf-8")
        print(f"wrote {JOBS / (a.name + '.json')}: {len(specs)} specs")
        return 0

    t0 = time.time()
    if analysing:
        rep = {"network_connections": "n/a (the batch summary above)", "real_profile_touched": "n/a"}
    else:
        games, rep = play(a.name, specs, a.game, a.servers, a.timeout)
        emus = emudump("fight", games, JOBS / f"{a.name}.emu.json")["results"]
        rea = emudump("reanchor", games, JOBS / f"{a.name}.reanchor.json")
    minutes = (time.time() - t0) / 60
    verdicts = judge(specs, games, emus, rea, nm, content)
    if not analysing and repro is None and a.mode not in ("hypotheses", "retest"):  # a batch.py batch keeps its own coverage; run repros are no coverage draw
        update_coverage(cov, specs, verdicts)
        cov_path.write_text(json.dumps(cov, indent=1, sort_keys=True), encoding="utf-8")
    lines = summarise(verdicts, minutes, rep)
    if repro is not None:
        for r, spec, v in zip(repro["runs"], specs, verdicts):
            r["repro"] = run_repro.matched(spec, v)
        (JOBS / f"{a.name}.repro.json").write_text(json.dumps(repro, indent=1), encoding="utf-8")
        lines.append("runs: " + ", ".join(f"{k} {n}" for k, n in sorted(collections.Counter(r["repro"] for r in repro["runs"]).items()))
                     + f" of {len(specs)} run fights (the run's step and field: same, field only, other); {len(repro['skipped'])} runs left out")
    if a.mode == "targets":
        act1_seed_cache(act1_path, games)
        lines.append(f"targets: {len(specs)} fights in {minutes:.1f} min = {len(specs) / max(minutes, 1e-9):.1f} fights/min "
                     f"(play and replay); aim {a.aim}, {len(aim.targets)} reachable targets, skipped {dict(aim.skipped)}")
    if a.mode == "choices":
        screens = sum(1 for g in games for c in g.get("choices", []) if c.get("kind") == "cards")
        edges = collections.Counter(c.get("edge") for g in games for c in g.get("choices", []) if c.get("kind") == "cards")
        lines.append(f"choices: {len(specs)} fights opened {screens} card screens, answered {dict(sorted(edges.items(), key=str))}")

    minimal = {}
    if a.shrink:
        import shrink  # noqa: PLC0415  (referee/shrink.py)
        t_shrink = time.time()
        deadline = t_shrink + 60 * a.shrink_minutes if a.shrink_minutes else None
        cases = shrink.cases_from(specs, games, verdicts, a.shrink, a.shrink_cap)
        run_round = shrink.oracle(lambda n, s: play(n, s, a.game, a.servers, a.timeout)[0],
                                  lambda rs: emudump("fight", rs, JOBS / f"{a.name}.shrink.emu.json")["results"], nm)
        minimal, srep = shrink.shrink_all(cases, run_round, a.name, budget=a.shrink_budget, deadline=deadline)
        lines.append(f"shrink: {srep['summary']}" + (f"; {(time.time() - t_shrink) / 60:.1f} min" if a.mode in ("targets", "choices", "analyze") else ""))
        lines += ["  " + x for x in srep["lines"]]
        mdir = JOBS / f"{a.name}.packs" / "minimal"
        mdir.mkdir(parents=True, exist_ok=True)
        for seed, m in minimal.items():
            (mdir / f"{seed}.json").write_text(json.dumps({"runs": [m["spec"]]}, indent=1), encoding="utf-8")
        (JOBS / f"{a.name}.shrink.json").write_text(json.dumps({"summary": srep["summary"], "fights": srep["fights"],
                                                               "cases": [{k: v for k, v in c.result().items() if k != "spec"}
                                                                         for c in cases]}, indent=1), encoding="utf-8")

    n_trace = a.trace if a.trace is not None else (40 if a.mode == "targets" else 0)
    if n_trace > 0 and any(not v.get("identical") and not v.get("error") for v in verdicts):
        t_trace = time.time()
        seeds = None
        if a.trace_minutes:  # a time limit needs the fights in calls of their own; their order is plausible ones first
            try:
                seeds = trace_order(games, emus, rea, verdicts, nm)
            except Exception as e:  # noqa: BLE001  (one call for all, as without a limit)
                lines.append(f"warning: trace order not computed ({type(e).__name__}); tracing in one call")
        lines.append(trace_batch(a.name, a.game, a.trace_servers or a.servers, n_trace, JOBS / f"{a.name}.trace.log", seeds=seeds,
                                 deadline=t_trace + 60 * a.trace_minutes if a.trace_minutes else None,
                                 max_servers=8 if a.trace_servers else TRACE_SERVERS)
                     + f"; {(time.time() - t_trace) / 60:.1f} min")

    # The batch's own outputs first: a failure in the packs below must not lose them (it cost two 500-fight batches
    # their verdicts once). The summary is written again with the packs' line when they are done.
    (JOBS / f"{a.name}.verdicts.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    summary = JOBS / f"{a.name}.summary.txt"
    head = ""
    if analysing and summary.exists():  # batch.py's own summary stays above this analysis
        head = summary.read_text(encoding="utf-8").split("\n\n# analysis\n")[0].rstrip("\n") + "\n\n# analysis\n"
    summary.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
    try:
        from tools.fidelity import context_pack  # noqa: PLC0415
        context_pack.run(JOBS / f"{a.name}.out.json", JOBS / f"{a.name}.emu.json", JOBS / f"{a.name}.packs",
                         minimal=minimal, flags={v["seed"]: v["flags"] for v in verdicts})
        lines.append(f"context packs: {JOBS / (a.name + '.packs') / 'index.md'}")
    except Exception as e:  # noqa: BLE001  (a warning: the packs are a convenience over files that are already written)
        lines.append(f"warning: context packs not written ({type(e).__name__}: {str(e)[:200]})")
    text = "\n".join(lines)
    summary.write_text(head + text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
