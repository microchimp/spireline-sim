#!/usr/bin/env python3
"""Setup fights end to end: make coverage-driven or dense fight specs, play them, replay, compare, shrink.

    python referee/scenarios.py --name <name> --mode dense|coverage --fights 60 [--start 0] [--servers 2]
        [--game D:/spireline/referee/game-setup] [--shrink K] [--shrink-budget 400] [--make-only]
    python referee/scenarios.py --name <name> --mode targets --targets LAB/targets.json --fights N [--start S]
        [--servers K] --game <game copy> --shrink K [--jobs <dir>] [--trace N]

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


def coverage_spec(content: Content, cov: dict, rng: random.Random, n: int) -> dict:
    enc, act, kind = encounter_key(content, cov, rng)
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
    setup = before + [{"cmd": "fight", "id": enc}] + after
    return _spec(rng, n, setup, "coverage", act, kind)


def dense_spec(content: Content, cov: dict, rng: random.Random, n: int) -> dict:
    enc, act, kind = encounter_key(content, cov, rng)
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
    setup = mixed + potions + [{"cmd": "fight", "id": enc}] + after
    return _spec(rng, n, setup, "dense", act, kind)


# ---------------------------------------------------------------- aimed fights (--mode targets)

ACT_NUMBER = {"OVERGROWTH": 1, "UNDERDOCKS": 1, "HIVE": 2, "GLORY": 3}
ROOM_OF_KIND = {"weak": "monster", "normal": "monster", "elite": "elite", "boss": "boss"}
BASICS = ("STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH", "ASCENDERS_BANE")
STARTER_DECK = {"STRIKE_IRONCLAD": 5, "DEFEND_IRONCLAD": 4, "BASH": 1, "ASCENDERS_BANE": 1}


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

    def __init__(self, targets_doc: dict, profile_doc: dict | None, content: Content, act1_seeds: dict | None = None):
        from tools.fidelity.lab_targets import Profile  # noqa: PLC0415
        self.prof = Profile(profile_doc)
        self.content = content
        self.act1_seeds = dict(act1_seeds or {})
        self.used, self.borrowed = set(), set()  # this batch's seeds; those taken from the cache
        self.enc = {e: (act, kind) for e, act, kind in content.encounters}
        self.cards, self.relics, self.potions = set(content.cards), set(content.relics), set(content.potions)
        self.card_power = {c + "_POWER": c for c in content.cards}
        self.targets, self.skipped = [], collections.Counter()
        for t in targets_doc.get("targets", []):
            if t.get("weight", 0) > 0 and self.reachable(t["kind"], t["id"]):
                self.targets.append((t["kind"], t["id"], float(t["weight"])))
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
            if known and rng.random() < 0.8:
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


def trace_batch(name: str, game: str, servers: int, n: int, log: Path) -> str:
    """referee/trace_divergent.py on the batch's diverging fights: their first differing event, into the packs."""
    out = JOBS / f"{name}.out.json"
    cmd = [sys.executable, str(ROOT / "referee" / "trace_divergent.py"), str(out), "--game", game,
           "--specs", str(JOBS / f"{name}.json"), "--emu", str(JOBS / f"{name}.emu.json"),
           "--reanchor", str(JOBS / f"{name}.reanchor.json"), "--packs", str(JOBS / f"{name}.packs"),
           "--max", str(n), "--servers", str(max(1, min(3, servers)))]
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, check=False,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    idx = JOBS / f"{name}.packs" / "traces" / "index.md"
    traced = sum(1 for line in idx.read_text(encoding="utf-8").splitlines() if line.startswith("- ")) if idx.exists() else 0
    return f"traces: {traced} fights (exit {p.returncode}), {idx.relative_to(JOBS).as_posix() if idx.exists() else 'none'}"


def _spec(rng: random.Random, n: int, setup: list, mode: str, act: str, kind: str) -> dict:
    return {"kind": "combat", "seed": _seed(rng), "character": "IRONCLAD", "ascension": 10, "neow": n % 3,
            "chooser_seed": 7000 + n, "max_actions": 150, "setup": setup, "use_potions": True,
            "tag": f"{mode}:{act}:{kind}"}


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


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=("dense", "coverage", "targets"), required=True)
    ap.add_argument("--fights", type=int, default=60)
    ap.add_argument("--start", type=int, default=0, help="spec stream offset")
    ap.add_argument("--game", default=GAME)
    ap.add_argument("--servers", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=5400)
    ap.add_argument("--shrink", type=int, default=0, help="shrink up to this many diverging fights")
    ap.add_argument("--shrink-budget", type=int, default=400, help="game fights the shrinker may spend in all")
    ap.add_argument("--make-only", action="store_true", help="write the job file and stop")
    ap.add_argument("--targets", type=Path, help="targets mode: tools/fidelity/lab_targets.py's targets.json (in the lab)")
    ap.add_argument("--jobs", type=Path, help="write every output here instead of jobs-setup")
    ap.add_argument("--trace", type=int, default=None, help="trace up to N diverging fights (default: 40 in targets mode, else 0)")
    a = ap.parse_args()
    if a.mode == "targets" and not a.targets:
        ap.error("--mode targets needs --targets")
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
    aims, act1_path = [], None
    if a.mode == "targets":
        tdoc = json.loads(a.targets.read_text(encoding="utf-8"))
        lab = a.targets.resolve().parent
        prof_path = lab / tdoc.get("profile", "cache/codex_profile.json")
        prof = json.loads(prof_path.read_text(encoding="utf-8")) if prof_path.exists() else None
        act1_path = lab / "cache" / "act1_seeds.json"
        known = act1_seed_cache(act1_path, scan=sorted(DEFAULT_JOBS.glob("*.out.json")))
        aim = Aim(tdoc, prof, content, known)
        if not aim.targets:
            print("no reachable target in", a.targets)
            return 1
        specs = []
        for i in range(a.fights):
            s, m = aimed_spec(aim, rng, a.start * 1000 + i)
            specs.append(s)
            aims.append(m)
        (JOBS / f"{a.name}.aims.json").write_text(json.dumps({"targets": a.targets.name, "skipped": dict(aim.skipped),
                                                              "profile_runs": aim.prof.runs, "aims": aims}, indent=1),
                                                  encoding="utf-8")
    else:
        make = dense_spec if a.mode == "dense" else coverage_spec
        specs = [make(content, cov, rng, a.start * 1000 + i) for i in range(a.fights)]
    if a.make_only:
        (JOBS / f"{a.name}.json").write_text(json.dumps({"runs": specs}), encoding="utf-8")
        print(f"wrote {JOBS / (a.name + '.json')}: {len(specs)} specs")
        return 0

    t0 = time.time()
    games, rep = play(a.name, specs, a.game, a.servers, a.timeout)
    emus = emudump("fight", games, JOBS / f"{a.name}.emu.json")["results"]
    rea = emudump("reanchor", games, JOBS / f"{a.name}.reanchor.json")
    minutes = (time.time() - t0) / 60
    verdicts = judge(specs, games, emus, rea, nm, content)
    update_coverage(cov, specs, verdicts)
    cov_path.write_text(json.dumps(cov, indent=1, sort_keys=True), encoding="utf-8")
    lines = summarise(verdicts, minutes, rep)
    if a.mode == "targets":
        act1_seed_cache(act1_path, games)
        lines.append(f"targets: {len(specs)} fights in {minutes:.1f} min = {len(specs) / max(minutes, 1e-9):.1f} fights/min "
                     f"(play and replay); {len(aim.targets)} reachable targets, skipped {dict(aim.skipped)}")

    minimal = {}
    if a.shrink:
        import shrink  # noqa: PLC0415  (referee/shrink.py)
        t_shrink = time.time()
        cases = shrink.cases_from(specs, games, verdicts, a.shrink)
        run_round = shrink.oracle(lambda n, s: play(n, s, a.game, a.servers, a.timeout)[0],
                                  lambda rs: emudump("fight", rs, JOBS / f"{a.name}.shrink.emu.json")["results"], nm)
        minimal, srep = shrink.shrink_all(cases, run_round, a.name, budget=a.shrink_budget)
        lines.append(f"shrink: {srep['summary']}" + (f"; {(time.time() - t_shrink) / 60:.1f} min" if a.mode == "targets" else ""))
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
        lines.append(trace_batch(a.name, a.game, a.servers, n_trace, JOBS / f"{a.name}.trace.log")
                     + f"; {(time.time() - t_trace) / 60:.1f} min")

    from tools.fidelity import context_pack  # noqa: PLC0415
    context_pack.run(JOBS / f"{a.name}.out.json", JOBS / f"{a.name}.emu.json", JOBS / f"{a.name}.packs",
                     minimal=minimal, flags={v["seed"]: v["flags"] for v in verdicts})
    lines.append(f"context packs: {JOBS / (a.name + '.packs') / 'index.md'}")
    (JOBS / f"{a.name}.verdicts.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    text = "\n".join(lines)
    (JOBS / f"{a.name}.summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
