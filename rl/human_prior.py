#!/usr/bin/env python3
"""The human deck prior (milestones 2 and 3 of the 2026-11-05 strategy memo, sections 3.1 to 3.3 and 3.6).

    python rl/human_prior.py donors    [--gens 11,9] [--per-key 64] [--max-runs 8000]
    python rl/human_prior.py build     [--procs 4] [--limit-runs N] [--clean] [--truth DIR]
    python rl/human_prior.py fit       [--out-dir DIR] [--variants gen11,scratch] [--winner-weights 1,2] [--decisions NAME] [--win-input]
    python rl/human_prior.py agreement --model DIR [--baseline run_value.npz] [--decisions NAME]

v1 (2026-11-05, `docs/rl/HUMAN_PRIOR.md` section "v1"): `rl/human_prior_replay.py run` regenerates every run's act maps from its seed on the
copy and replays its run layer for the shop prices; `build --truth` then puts the true map block (and the true distance to the boss)
into every row in place of a donor's, adds the route kind (which room to go to, among the rooms the node the human stood on leads
to), and takes shop shelves and prices from the replay where it was in step; `fit --decisions decisions-v1` fits on those, with
`--win-input` giving the net one extra input, "this run won".

What the public Ironclad Ascension 10 record teaches, one decision at a time: which card a human took (or skipped) from the
offers, which ancient option they chose at an act's start, whether they healed or smithed at a rest site (and which card the
smith took), and what they bought or left in a shop. `build` rebuilds the state each human stood in from the record (the rebuild
`rl/human_starts.py` and `tools/boss_entry_exam.py` do), turns every offered option into its afterstate with the run
judgement's own feature code (`run_train.features.apply_option` / `encode_afterstate`) and stores the 1,863-wide rows as shards.
`fit` trains the prior `pi_h` (an afterstate scorer on the run judgement's trunk; a softmax over the options of a decision against
the human's choice). `agreement` scores a prior (and optionally the run judgement itself) against the human's choice on the
validation fifth.

Splits: the exam fifth (D38: the ids file united with the hash bucket 0) is never read; `assert_no_exam` runs on every shard.
Fifths 1 to 3 train, fifth 4 is the validation set for every choice. Human data is reported as aggregates only (cells under 20
runs are not shown) and the shards stay in the referee folder, never in a repository.

The record has no map. Each decision gets a donor map: the map block (and the distance-to-boss scalar) of one of the run
judgement's own self-play states in the same region and row of the act (the exam's donors, `donors`), drawn afresh every epoch at
fit time and from a fixed seed at validation. The fight block H (FightNet on the afterstate) is computed once, with the first
donor of the key.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import math
import os
import sys
import time
import types
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CORPUS = Path("D:/spireline/data/spire-codex/runs-v0.107.1.jsonl.gz")
EXAM_IDS = CORPUS.with_name("human-exam-fifth-ids.txt")
OUT_DIR = Path("D:/spireline/referee/human-prior")
RUNS_ROOT = Path("D:/spireline/referee/LAB/runs")
GEN11 = Path("D:/spireline/referee/agent/gen11/candidate/run_value.npz")
REF_COUNTS = Path("D:/spireline/referee/backlog/scratch-strategy/counts.json")
W_WON = 53.0  # the lookahead macro's worth of a won run
MIN_CELL = 20  # D13: a cell over fewer runs is not shown
EXAM_BUCKET = 0
VAL_FIFTH = 4
TRAIN_FIFTHS = (1, 2, 3)
SPIRE_CODEX = "Spire Codex public run export for v0.107.1"

REGION = {"ACT.OVERGROWTH": 1, "ACT.UNDERDOCKS": 2, "ACT.HIVE": 3, "ACT.GLORY": 4}
START_HP, START_MAX_HP, START_GOLD = 64, 80, 99  # the run judgement's own Act 1 start state at the first ancient
KIND = {"card": 0, "rest": 1, "ancient": 2, "shop": 3, "event": 4, "route": 5}  # event rows only with `build --events`, route rows only in v1 shards (docs/rl/HUMAN_PRIOR.md)
KIND_NAME = {v: k for k, v in KIND.items()}
ROLE_HEAL, ROLE_SMITH, ROLE_OTHER = 0, 1, 2
OFFER_CARD, OFFER_RELIC, OFFER_POTION = 0, 1, 2
# `dec` columns, one int32 row per decision
COLS = ("kind", "fifth", "act", "floor", "region", "row", "boss", "run", "won", "final", "start", "n", "chosen", "aux", "aux2", "flags",
        "offer", "price", "gold", "hp", "max_hp")
C = {name: i for i, name in enumerate(COLS)}
F_SPLIT, F_PRICE_UNVERIFIED, F_UNAVAILABLE, F_REPLAY_MISMATCH, F_BOSS_GUESSED = 1, 2, 4, 8, 16
F_GATE_UNKNOWN = 32  # an event page with an option whose lock turns on a price the record does not carry (the option is listed)
F_SHELF_ONLY = 64  # v1 shop rows: the copy's shelf and prices match the record's, but the replay had left the record's path before the visit
# the dense part stored per row: A (22), the first 21 of B (room and special slots), D (25), X (12) and H (8)
N_DENSE = 88
S_A, S_B, S_D, S_X, S_H = slice(0, 22), slice(22, 43), slice(43, 68), slice(68, 80), slice(80, 88)
# v1 shards (the true map): the same, with the map block C (25) between B and D, and the distance to the boss kept in A
N_DENSE_V1 = 113
S1_A, S1_B, S1_C, S1_D, S1_X, S1_H = slice(0, 22), slice(22, 43), slice(43, 68), slice(68, 93), slice(93, 105), slice(105, 113)
REST_RELICS = {"SHOVEL": "rest.dig", "GIRYA": "rest.lift", "MEAT_CLEAVER": "rest.cook", "PUMPKIN_CANDLE": "rest.kindle",
               "PAELS_GROWTH": "rest.clone"}
REST_CHOICE_KIND = {"DIG": "rest.dig", "LIFT": "rest.lift", "COOK": "rest.cook", "KINDLE": "rest.kindle", "CLONE": "rest.clone"}
SPLIT_SCREEN = 6  # a card-reward point listing six (nine, ...) offers is two (three, ...) merged screens of three
MAX_SCREEN = 4  # screens of more than four offers (other than the six) are dropped
PRICE_TOLERANCE = 10  # gold: the memo's shop check
PRICE_GATE = 0.9


# ------------------------------------------------------------------------------------------------------------- the run stack


_STACK: dict = {}


def stack() -> types.SimpleNamespace:
    """The fork's feature code. The native library is loaded when `HP_LIB` names a build or one is found where the loader looks (a
    test session already has it); otherwise a stub stands in for `sts2_gym.native`, which only the live-state builders read (the exam
    tool does the same), so a build needs no engine. The fight network and tables are cached."""
    if _STACK:
        return _STACK["ns"]
    import sts2_gym  # noqa: PLC0415 -- the package loads its library lazily

    lib = os.environ.get("HP_LIB", "")
    if lib:
        os.environ["STS2_LIB_PATH"] = lib
    try:
        from run_train import features, models, vocab  # noqa: PLC0415
    except (FileNotFoundError, RuntimeError, OSError):
        stub = types.ModuleType("sts2_gym.native")
        sts2_gym.native = stub
        sys.modules["sts2_gym.native"] = stub
        from run_train import features, models, vocab  # noqa: PLC0415
    from sts2_gym import names  # noqa: PLC0415

    ns = types.SimpleNamespace(F=features, M=models, V=vocab, names=names)
    _STACK["ns"] = ns
    return ns


def below_normal() -> None:
    """BelowNormal priority through the Windows priority class (no POSIX call)."""
    import human_starts as hs  # noqa: PLC0415

    hs.below_normal()


class Tables:
    """Slug -> emulator id for cards, relics, potions and enchantments; rarities and the colourless pool for shop prices."""

    def __init__(self) -> None:
        import human_starts as hs  # noqa: PLC0415

        s = stack()
        im = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))
        self.cards = {hs.slug(k): v for k, v in im["cards"].items()}
        self.relics = {hs.slug(k): v for k, v in im["relics"].items()}
        self.potions = {hs.slug(k): v for k, v in im["potions"].items()}
        self.enchants = {hs.slug(n): int(i) for i, n in s.names.enchantments().items() if int(i) != 0}
        gen = s.V._GENERATED
        rel = s.names._definitions(gen / "Relics.g.cs", "RelicDef")
        pot = s.names._definitions(gen / "Potions.g.cs", "PotionDef")
        self.relic_rarity = {i: f.get("Rarity", "").rpartition(".")[2] for i, f in rel.items()}
        self.potion_rarity = {i: f.get("Rarity", "").rpartition(".")[2] for i, f in pot.items()}
        self.colorless = frozenset(s.V.card_pools()["Colorless"])
        self.boss_of = {slug: bid for slug, (bid, _region) in s.V.BOSSES.items()}
        self.region_bosses: dict[int, list[int]] = defaultdict(list)
        for _slug, (bid, region) in s.V.BOSSES.items():
            self.region_bosses[region].append(bid)
        rid = s.names.relics()
        by_name = {v: k for k, v in rid.items()}
        self.membership = by_name.get("MembershipCard")
        self.courier = by_name.get("TheCourier")

    def card_ref(self, c: tuple):
        """(slug, level, enchantment slug, amount) -> CardRef, or None for a card or enchantment the copy does not know."""
        F = stack().F
        cid = self.cards.get(c[0])
        if cid is None:
            return None
        ench = amount = 0
        if c[2]:
            ench = self.enchants.get(c[2])
            if ench is None:
                return None
            amount = c[3]
        return F.CardRef(cid, c[1] >= 1, ench, amount)


# ----------------------------------------------------------------------------------------------------------- shop prices


def round_even(x: float) -> int:
    """.NET's Math.Round / Godot's Mathf.RoundToInt: half to even (the copy's `RoundToEven`)."""
    return int(round(x))


def merchant_modifier(price: int, relics: Iterable[int], t: Tables) -> int:
    """The Courier's 20 percent off, then Membership Card's half (the copy's `EnterShop` order)."""
    held = set(relics)
    if t.courier in held:
        price = int(price * 0.8)
    if t.membership in held:
        price //= 2
    return price


def shop_price(offer: int, ident: int, relics: Iterable[int], t: Tables) -> tuple[int, int, int]:
    """(expected, low, high) price in gold of one shop offer, from the copy's rules (`RunRewardGenerator.EnterShop`): a card's base by
    rarity (50 / 75 / 150, colourless x1.15) jittered by 0.95 to 1.05; a relic's by rarity (175 / 225 / 275, shop 200) by 0.85 to 1.15; a
    potion's (50 / 75 / 100) by 0.95 to 1.05; the Courier and Membership Card then take their share. One coloured card is on sale at
    half and the record does not say which: the check allows for it, the expected price here does not."""
    s = stack()
    if offer == OFFER_CARD:
        try:
            rarity = s.V.card(ident).rarity
        except KeyError:
            rarity = "Common"
        base = {"Rare": 150, "Uncommon": 75}.get(rarity, 50)
        if ident in t.colorless:
            base = round_even(base * 1.15)
        spread = 0.05
    elif offer == OFFER_RELIC:
        base = {"Common": 175, "Uncommon": 225, "Rare": 275, "Shop": 200}.get(t.relic_rarity.get(ident, ""), 200)
        spread = 0.15
    else:
        base = {"Rare": 100, "Uncommon": 75}.get(t.potion_rarity.get(ident, ""), 50)
        spread = 0.05
    return tuple(merchant_modifier(round_even(base * m), relics, t) for m in (1.0, 1 - spread, 1 + spread))  # type: ignore[return-value]


def removal_price(removals_before: int, relics: Iterable[int], t: Tables) -> int:
    """The card-removal service at Ascension 10 (Inflation): 100 and 50 more for every removal already bought this run."""
    return merchant_modifier(100 + 50 * removals_before, relics, t)


def price_check(bought: list[tuple[int, int, bool]], removal: int | None, spent: int, relics: list[int], t: Tables) -> dict:
    """One shop visit's rebuilt spend against the record's gold spent. `bought` lists (offer kind, id, colourless?) per purchase.
    Returns the expected total with the sale on the best-fitting bought coloured card (or on none), the rules' band, and flags."""
    mid = lo = hi = 0
    coloured_cards: list[int] = []
    for offer, ident, colorless in bought:
        p, a, b = shop_price(offer, ident, relics, t)
        mid, lo, hi = mid + p, lo + a, hi + b
        if offer == OFFER_CARD and not colorless:
            coloured_cards.append(p)
    if removal is not None:
        mid, lo, hi = mid + removal, lo + removal, hi + removal
    options = [0] + [p - p // 2 for p in coloured_cards]  # the sale takes this much off one coloured card
    best = min(options, key=lambda cut: abs(spent - (mid - cut)))
    pred = mid - best
    cut_lo = min(options)
    return {"pred": pred, "strict": mid, "within": abs(spent - pred) <= PRICE_TOLERANCE, "strict_within": abs(spent - mid) <= PRICE_TOLERANCE,
            "in_band": (lo - max(options) - 1) <= spent <= (hi - cut_lo + 1), "error": spent - pred}


# --------------------------------------------------------------------------------------------------------------- the corpus


def iter_run_lines(corpus: Path, exam: frozenset, stats: Counter, limit: int = 0, ascensions: tuple[int, ...] = (10,)) -> Iterator[str]:
    """The raw JSON line of every usable run (Ironclad, Ascension 10 or the levels in `ascensions`, standard, finished, no modifiers,
    vanilla acts) that is not in the exam fifth. The same filters as `human_starts.read_runs`; the exam fifth's lines are dropped here,
    before any worker sees them."""
    import human_starts as hs  # noqa: PLC0415

    opener = gzip.open if str(corpus).endswith(".gz") else open
    asc = tuple(ascensions)
    with opener(corpus, "rt", encoding="utf-8") as fh:
        for line in fh:
            stats["lines"] += 1
            if '"character":"CHARACTER.IRONCLAD"' not in line:
                continue
            if asc == (10,):
                if '"ascension":10,' not in line:
                    continue
            elif not any(f'"ascension":{a},' in line for a in asc):
                continue
            try:
                r = json.loads(line)
            except ValueError:
                stats["unparsable"] += 1
                continue
            if not hs.human_ok(r, asc):
                continue
            stats["ironclad_a10"] += 1
            if hs.is_exam(hs.run_id(r), exam):
                stats["exam_fifth_left_out"] += 1
                continue
            if not hs.vanilla_acts(r):
                stats["modded_acts"] += 1
                continue
            stats["runs_usable"] += 1
            yield line
            if limit and stats["runs_usable"] >= limit:
                return


# -------------------------------------------------------------------------------------------- decisions from one human run


class Drop(Exception):
    """A decision that cannot be built; the message is the reason (counted, never patched)."""


def _boss_of_act(mph: list, a: int, t: Tables) -> int | None:
    for pt in mph[a]:
        if pt.get("map_point_type") == "boss":
            model = ((pt.get("rooms") or [{}])[0].get("model_id")) or ""
            return t.boss_of.get(model.split(".", 1)[-1])
    return None


class NotAChoice(Exception):
    """A step of the route with only one room to go to: not a decision."""


def run_exclusion(truth: dict | None) -> str | None:
    """Why a run's true map cannot stand (the run is left out whole), or None: a Golden Compass makes an act's map one column, a record
    whose route is not a path through the regenerated map is a `map_mismatch` (the copy and the game disagree on the run), a map the
    copy could not draw at all is a `map_crash`."""
    if truth is None:
        return None
    if truth.get("maps_crash"):
        return "map_crash"
    if truth.get("compass"):
        return "golden_compass"
    if any(a.get("status") != "ok" for a in truth.get("acts") or []):
        return "map_mismatch"
    return None


def _truth_position(truth: dict | None, a: int, i: int) -> tuple[int, int] | None:
    """The node (col, row) the human stood on at point `i` of act `a`, or None when there is no true map or the record cannot tell
    which of several paths of the same room types it took."""
    if truth is None or a >= len(truth.get("acts") or []):
        return None
    ent = truth["acts"][a]
    cols = ent.get("cols") or []
    if ent.get("status") != "ok" or i >= len(cols) or cols[i] < 0:
        return None
    return (int(cols[i]), i)


def act_map(ent: dict) -> dict:
    """An act's true map from a truth record: the `F.MapNode`s, the children of every node (sorted), the types and the boss node."""
    s = stack()
    kids: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for c, rw, c2, r2 in ent["edges"]:
        kids[(c, rw)].append((c2, r2))
    nodes = tuple(s.F.MapNode(c, rw, ty, tuple(kids.get((c, rw), ()))) for c, rw, ty in ent["nodes"])
    boss = next(((c, rw) for c, rw, ty in ent["nodes"] if ty == s.V.NODE_BOSS), None)
    return {"nodes": nodes, "kids": {k: sorted(v) for k, v in kids.items()}, "type": {(c, rw): ty for c, rw, ty in ent["nodes"]}, "boss": boss}


def _route_options(amap: dict, prev_pos: tuple[int, int], cur_pos: tuple[int, int]) -> tuple[list[dict], int]:
    """The options of the step from `prev_pos` to `cur_pos` (the rooms the node leads to) and the index of the one the human took."""
    kids = amap["kids"].get(prev_pos, [])
    if len(kids) < 2:
        raise NotAChoice
    if cur_pos not in kids:
        raise Drop("route step is not along an edge of the map")
    return [{"k": "map.node", "node": [c, rw, amap["type"][(c, rw)]]} for c, rw in kids], kids.index(cur_pos)


def _floor_state(r: dict, f: int, gains: dict, removed: dict, deck: list, potions: list, t: Tables) -> tuple[list, list, list]:
    """(relic ids, deck refs, potion ids) walking into floor `f`; a card, relic or enchantment the copy does not know is a Drop."""
    import human_starts as hs  # noqa: PLC0415

    relic_ids = []
    for name in hs.relics_at(r, f, gains, removed):
        if name not in t.relics:
            raise Drop("relic unknown to the copy")
        relic_ids.append(t.relics[name])
    refs = []
    for c in deck:
        ref = t.card_ref(c)
        if ref is None:
            raise Drop("deck card or enchantment unknown to the copy")
        refs.append(ref)
    return relic_ids, refs, [t.potions[p] for p in potions if p in t.potions]


def run_decisions(r: dict, t: Tables, stats: Counter, seen: Counter, price_stats: Counter, events: bool = False, truth: dict | None = None) -> list[dict]:
    """Every card reward, ancient, rest-site and shop decision of one run as plain-data specs: the state the human stood in and the
    options with the chosen one. `stats` counts the drops by reason, `seen` the screens the corpus holds (counts.json's definitions),
    `price_stats` the shop price check.

    With `truth` (v1: the run's record from `rl/human_prior_replay.py`) every spec carries the act's true map and the node the human
    stood on (a decision whose node the record cannot tell is dropped), the shop prices come from the replay where it is in step, and a
    route decision is built at every step with more than one room to go to. A run whose true map cannot stand is left out whole.

    With `events`, also one decision per option page of each event visit whose choice the option table resolves (`rl/event_keys.py`);
    the others are dropped and counted."""
    import human_starts as hs  # noqa: PLC0415

    rid = hs.run_id(r)
    fifth = hs.exam_bucket(rid)
    mph = r.get("map_point_history") or []
    flat = [(a, i, pt) for a, act in enumerate(mph) for i, pt in enumerate(act)]
    won = bool(r.get("win"))
    final = len(flat)
    names_of_acts = r.get("acts") or []
    exclusion = run_exclusion(truth)
    if truth is not None:
        stats["runs_read"] += 1
        stats[f"runs_read|fifth{fifth}"] += 1
        if exclusion:
            stats[f"runs_left_out|{exclusion}"] += 1
            stats[f"runs_left_out|{exclusion}|fifth{fifth}"] += 1
        else:
            stats["runs_true_map"] += 1
            stats[f"runs_true_map|fifth{fifth}"] += 1
            stats["runs_position_ambiguous"] += int(any(not a.get("unique", True) for a in truth["acts"]))
            stats["runs_with_boots"] += int(truth.get("boots") is not None)
    maps: dict[int, dict] = {}
    shop_visits = {int(v["floor"]): v for v in (truth or {}).get("shops") or []}
    gains: dict[str, int] = {}
    removed: dict[str, int] = {}
    for f, (_, _, pt) in enumerate(flat, 1):
        ps = (pt.get("player_stats") or [{}])[0]
        for c in ps.get("relic_choices") or []:
            if c.get("was_picked"):
                gains.setdefault(hs._cid(c["choice"]), f)
        for c in ps.get("bought_relics") or []:
            gains.setdefault(hs._cid(c), f)
        for c in ps.get("relics_removed") or []:
            removed[hs._cid(c)] = f
    deck: list = [("ASCENDERS_BANE", 0, None, 0)] + [("STRIKE_IRONCLAD", 0, None, 0)] * 5 + [("DEFEND_IRONCLAD", 0, None, 0)] * 4 + [("BASH", 0, None, 0)]
    potions: list[str] = []
    bosses = [_boss_of_act(mph, a, t) for a in range(len(mph))]
    specs: list[dict] = []
    prev: dict = {}
    removals = 0
    for f, (a, i, pt) in enumerate(flat, 1):
        ps = (pt.get("player_stats") or [{}])[0]
        ptype = pt.get("map_point_type")
        act = min(a, 2)
        region = REGION.get(names_of_acts[a]) if a < len(names_of_acts) else None
        want: list[str] = []
        cc = ps.get("card_choices") or []
        if cc and ptype != "shop":
            seen[f"act{act + 1}|card"] += 1
            want.append("card")
        if ps.get("ancient_choice"):
            seen[f"act{act + 1}|ancient"] += 1
            want.append("ancient")
        if ps.get("rest_site_choices"):
            seen[f"act{act + 1}|rest"] += 1
            want.append("rest")
        if ptype == "shop":
            seen[f"act{act + 1}|shop"] += 1
            want.append("shop")
        if events and _is_event_visit(pt, ptype) and ps.get("event_choices"):
            seen[f"act{act + 1}|event"] += 1
            want.append("event")
        if exclusion:
            for kind in want:
                stats[f"drop|act{act + 1}|{kind}|run left out: {exclusion}"] += 1
            continue
        boss = bosses[a] if a < len(bosses) else None
        guessed = boss is None
        if guessed and region is not None:
            pool = t.region_bosses[region]
            boss = pool[int(hashlib.sha256(f"{rid}|{a}".encode()).hexdigest()[:8], 16) % len(pool)]
        amap = None
        if truth is not None:
            if a not in maps:
                maps[a] = act_map(truth["acts"][a])
            amap = maps[a]
        if truth is not None and i >= 1 and region is not None and prev:
            _route_decision(specs, r, t, stats, truth, amap, a, i, f, act, region, boss, guessed, won, final, fifth, rid, prev, gains, removed, deck, potions)
        if want and region is not None:
            try:
                relic_ids, refs, pot_ids = _floor_state(r, f, gains, removed, deck, potions, t)
                if ptype == "ancient" or "ancient" in want:
                    hp, mx, gold = (START_HP, START_MAX_HP, START_GOLD) if f == 1 else (ps.get("current_hp"), prev.get("max_hp", ps.get("max_hp")),
                                                                                       prev.get("current_gold", ps.get("current_gold")))
                elif "rest" in want:
                    hp, mx, gold = prev.get("current_hp", ps.get("current_hp")), prev.get("max_hp", ps.get("max_hp")), prev.get("current_gold", ps.get("current_gold"))
                elif "shop" in want:
                    hp, mx, gold = ps.get("current_hp"), ps.get("max_hp"), int(ps.get("current_gold") or 0) + int(ps.get("gold_spent") or 0)
                else:
                    hp, mx, gold = ps.get("current_hp"), ps.get("max_hp"), ps.get("current_gold")
                if hp is None or mx is None or gold is None or not 0 < int(mx):
                    raise Drop("hp or gold missing in the record")
                if prev:  # continuity checks of the rebuild against the record's own fields (counted, not used)
                    if "rest" in want and ptype == "rest_site":
                        stats["check|rest_hp_n"] += 1
                        stats["check|rest_hp_ok"] += int(prev.get("current_hp", -1) + int(ps.get("hp_healed") or 0) == ps.get("current_hp"))
                    if "shop" in want:
                        stats["check|shop_gold_n"] += 1
                        stats["check|shop_gold_ok"] += int(prev.get("current_gold", -1) == gold)
                base = {"run": rid, "fifth": fifth, "act": act, "floor": f, "region": region, "row": min(i, 16), "boss": boss,
                        "won": won, "final": final, "hp": int(min(hp, mx)), "max_hp": int(mx), "gold": int(gold),
                        "deck": [(x.card_id, int(x.upgraded), x.enchantment, x.enchant_amount) for x in refs], "relics": relic_ids,
                        "potions": pot_ids, "flags": F_BOSS_GUESSED if guessed else 0}
                if truth is not None:
                    pos = _truth_position(truth, a, i)
                    if pos is None:
                        raise Drop("map position ambiguous (several paths of the same room types)")
                    base.update(map_nodes=amap["nodes"], position=pos)
                walk_in = base
                if "event" in want:  # an event's state is the one walking in (the floor before's), whatever else the point holds
                    h2, m2, g2 = prev.get("current_hp", ps.get("current_hp")), prev.get("max_hp", ps.get("max_hp")), prev.get("current_gold", ps.get("current_gold"))
                    if h2 is None or m2 is None or g2 is None or not 0 < int(m2):
                        stats[f"drop|act{act + 1}|event|hp or gold missing in the record"] += 1
                        want = [k for k in want if k != "event"]
                    else:
                        walk_in = dict(base, hp=int(min(h2, m2)), max_hp=int(m2), gold=int(g2))
                for kind in want:
                    try:
                        specs.extend(_kind_specs(kind, walk_in if kind == "event" else base, ps, pt, t, stats, price_stats, removals,
                                                 shop_visits.get(f) if truth is not None else None))
                    except Drop as why:
                        stats[f"drop|act{act + 1}|{kind}|{why}"] += 1
            except Drop as why:
                for kind in want:
                    stats[f"drop|act{act + 1}|{kind}|{why}"] += 1
        elif want:
            for kind in want:
                stats[f"drop|act{act + 1}|{kind}|act region unknown"] += 1
        if ptype == "shop" and ps.get("cards_removed"):
            removals += 1
        hs.apply_point(deck, ps)
        hs.apply_potions(potions, pt, ps)
        prev = ps
    if specs and not hs.final_deck_matches(deck, r):
        stats["runs_replay_mismatch"] += 1
        for sp in specs:
            sp["flags"] |= F_REPLAY_MISMATCH
    return specs


def _route_decision(specs: list, r: dict, t: Tables, stats: Counter, truth: dict, amap: dict, a: int, i: int, f: int, act: int, region: int, boss: int,
                    guessed: bool, won: bool, final: int, fifth: int, rid: str, prev: dict, gains: dict, removed: dict, deck: list, potions: list) -> None:
    """The route decision before floor `f` (point `i` of act `a`): which room to go to from the node the human stood on. The state is the
    one after floor `f - 1` (the HP, gold, deck, relics and potions walking out of it); the options are the rooms the node leads to
    (the afterstate is the step: the room's type pending, one floor on, the position moved). A step with one room to go to is not a
    decision; Winged Boots (free steps to any room of the next row) are not modelled, so the floors after the pick-up are left out."""
    tag = f"route|act{act + 1}"
    stats[f"{tag}|steps"] += 1
    prev_pos, cur_pos = _truth_position(truth, a, i - 1), _truth_position(truth, a, i)
    try:
        if prev_pos is not None and len(amap["kids"].get(prev_pos, [])) < 2:
            raise NotAChoice
        if truth.get("boots") is not None and f > int(truth["boots"]):
            raise Drop("route: after Winged Boots were picked up (free steps are not modelled)")
        if prev_pos is None or cur_pos is None:
            raise Drop("route: position ambiguous (several paths of the same room types)")
        options, chosen = _route_options(amap, prev_pos, cur_pos)
        relic_ids, refs, pot_ids = _floor_state(r, f, gains, removed, deck, potions, t)
        hp, mx, gold = prev.get("current_hp"), prev.get("max_hp"), prev.get("current_gold")
        if hp is None or mx is None or gold is None or not 0 < int(mx):
            raise Drop("hp or gold missing in the record")
    except NotAChoice:
        stats[f"{tag}|single_room"] += 1
        return
    except Drop as why:
        stats[f"drop|act{act + 1}|route|{why}"] += 1
        return
    stats[f"{tag}|choices"] += 1
    if len({amap["type"][tuple(o["node"][:2])] for o in options}) > 1:
        stats[f"{tag}|type_choices"] += 1
    base = {"run": rid, "fifth": fifth, "act": act, "floor": f - 1, "region": region, "row": min(i - 1, 16), "boss": boss, "won": won, "final": final,
            "hp": int(min(hp, mx)), "max_hp": int(mx), "gold": int(gold), "deck": [(x.card_id, int(x.upgraded), x.enchantment, x.enchant_amount) for x in refs],
            "relics": relic_ids, "potions": pot_ids, "flags": F_BOSS_GUESSED if guessed else 0, "map_nodes": amap["nodes"], "position": prev_pos}
    specs.append(dict(base, kind="route", options=options, chosen=chosen, aux=int(amap["boss"][0]) if amap["boss"] else 0, aux2=0, offer=0, price=0))


def _view_args(sp: dict) -> dict:
    return {"ascension": 10, "act_index": sp["act"], "act_region": sp["region"], "floor": sp["floor"], "hp": sp["hp"], "max_hp": sp["max_hp"],
            "gold": sp["gold"], "relics": tuple(sp["relics"]), "potions": tuple(sp["potions"]), "boss": sp["boss"]}


def _kind_specs(kind: str, base: dict, ps: dict, pt: dict, t: Tables, stats: Counter, price_stats: Counter, removals: int, visit: dict | None = None) -> list[dict]:
    """The specs of one kind at one map point. Options are plain dicts: `{"k": option kind, ...}`; `chosen` indexes them."""
    import human_starts as hs  # noqa: PLC0415

    if kind == "card":
        return _card_specs(base, ps, t, stats)
    if kind == "ancient":
        opts = ps.get("ancient_choice") or []
        out, chosen = [], []
        for n, c in enumerate(opts):
            rel = t.relics.get(str(c.get("TextKey") or ""))
            if rel is None or ((c.get("title") or {}).get("table") not in (None, "relics")):
                raise Drop("ancient option is not a relic the copy knows")
            out.append({"k": "neow.relic", "relic": rel})
            if c.get("was_chosen"):
                chosen.append(n)
        if len(chosen) != 1 or not 1 < len(out) <= 4:
            raise Drop("ancient screen without one chosen option")
        return [dict(base, kind="ancient", options=out, chosen=chosen[0], aux=0, aux2=0, offer=0, price=0)]
    if kind == "rest":
        return _rest_specs(base, ps, t)
    if kind == "event":
        return _event_specs(base, ps, pt, t, stats)
    return _shop_specs(base, ps, pt, t, stats, price_stats, removals, visit, v1="map_nodes" in base)


def _is_event_visit(pt: dict, ptype: str) -> bool:
    """A "?" room that turned out to be an event (not an ancient, not Neow)."""
    mid = str((pt.get("rooms") or [{}])[0].get("model_id") or "")
    return ptype != "ancient" and mid.startswith("EVENT.") and mid != "EVENT.NEOW"


def _event_gate_state(base: dict, t: Tables) -> dict:
    """What the events' lock rules read (`event_keys.available`) from the state the human walked in with."""
    V = stack().V
    deck = [(V.card(c[0]), bool(c[1])) for c in base["deck"]]
    starter = {t.relics.get("BURNING_BLOOD"), t.relics.get("BLACK_BLOOD")}
    return {"gold": base["gold"], "potions": len(base["potions"]), "relics": sum(1 for r in base["relics"] if r not in starter),
            "n_deck": len(deck), "n_attack": sum(f.card_type == "Attack" for f, _ in deck), "n_skill": sum(f.card_type == "Skill" for f, _ in deck),
            "n_power": sum(f.card_type == "Power" for f, _ in deck), "n_upgradable": sum(V.card_upgradable(f.card_id, up) for f, up in deck),
            "n_exhaust": sum(f.is_exhaust(up) for f, up in deck)}


def _event_specs(base: dict, ps: dict, pt: dict, t: Tables, stats: Counter) -> list[dict]:
    """One decision per option page of an event visit: the options are the page's displayed indices that the state leaves on offer, as
    `event.option` afterstates (event id, index, step = the page's place in the visit), the chosen one the index the option table gives
    the recorded key. A key the table cannot place (a rolled page, a name no source knows) or a page with one option is dropped and
    counted by reason; the other pages of the visit still count."""
    import event_keys as ek  # noqa: PLC0415

    tab = ek.table()
    act = f"act{base['act'] + 1}"
    name = str((pt.get("rooms") or [{}])[0].get("model_id") or "").removeprefix("EVENT.")
    event = tab.by_name.get(name)
    if event is None:
        stats[f"drop|{act}|event|event of another mod, not in the table"] += 1
        return []
    gate = _event_gate_state(base, t)
    out = []
    for step, c in enumerate(ps.get("event_choices") or []):
        key = str((c.get("title") or {}).get("key") or "")
        idx = tab.resolve(name, key)
        if idx is None:
            rolled = any(r.get("dynamic") for r in tab.rows_for(name, key))
            why = "page with rolled options" if rolled else "name the table does not place"
            stats[f"drop|{act}|event|{why}"] += 1
            continue
        shown = tab.shown_for(name, key)
        if shown is None or idx not in shown:
            stats[f"drop|{act}|event|page not in the table"] += 1
            continue
        if len(shown) < 2:
            stats[f"drop|{act}|event|page with one option"] += 1
            continue
        avail = {i: (True if i == idx else (ek.available(name, i, gate) if step == 0 else True)) for i in shown}
        options = [i for i in shown if avail[i] is not False]
        if len(options) < 2:
            stats[f"drop|{act}|event|page with one option on offer"] += 1
            continue
        flags = base["flags"] | (F_GATE_UNKNOWN if any(avail[i] is None for i in options) else 0)
        out.append(dict(base, kind="event", options=[{"k": "event.option", "event": [event["engine_id"], i, min(step, 3)], "id": i} for i in options],
                        chosen=options.index(idx), aux=event["engine_id"], aux2=step, offer=0, price=0, flags=flags))
        stats[f"event_decisions|{act}"] += 1
    return out


def _card_specs(base: dict, ps: dict, t: Tables, stats: Counter) -> list[dict]:
    import human_starts as hs  # noqa: PLC0415

    cc = ps.get("card_choices") or []
    n = len(cc)
    if n >= SPLIT_SCREEN and n % 3 == 0:  # two or more merged screens of three offers each, in order
        parts = [(cc[j:j + 3], F_SPLIT) for j in range(0, n, 3)]
    elif 1 <= n <= MAX_SCREEN:
        parts = [(cc, 0)]
    else:
        raise Drop(f"card screen of {n} offers")
    out = []
    for part, flag in parts:
        offers = []
        for c in part:
            card = c.get("card") or {}
            ident = t.cards.get(hs._cid(card.get("id") or ""))
            if ident is None:
                raise Drop("offered card unknown to the copy")
            offers.append({"k": "card.take", "card": [ident, int(int(card.get("current_upgrade_level") or 0) >= 1)]})
        picked = [j for j, c in enumerate(part) if c.get("was_picked")]
        if len(picked) > 1:
            stats["drop|card|two cards picked on one screen"] += 1
            continue
        chosen = picked[0] if picked else len(part)
        out.append(dict(base, kind="card", options=offers + [{"k": "card.skip"}], chosen=chosen, aux=int(not picked), aux2=0, offer=0, price=0,
                        flags=base["flags"] | flag))
    return out


def _rest_specs(base: dict, ps: dict, t: Tables) -> list[dict]:
    import human_starts as hs  # noqa: PLC0415

    s = stack()
    choice = str((ps.get("rest_site_choices") or [""])[0]).upper()
    relic_names = {n for n, i in t.relics.items() if i in set(base["relics"])}
    others = sorted((REST_RELICS[n] for n in REST_RELICS if n in relic_names))
    deck = [s.F.CardRef(c[0], bool(c[1]), c[2], c[3]) for c in base["deck"]]
    distinct: list[tuple] = []
    for c in sorted(deck):
        if s.V.card_upgradable(c.card_id, c.upgraded) and not any(d[0] == c for d in distinct):
            distinct.append((c, None))
    options: list[dict] = [{"k": "rest.heal", "role": ROLE_HEAL}]
    for c, _ in distinct:
        options.append({"k": "rest.smith", "role": ROLE_SMITH, "card": [c.card_id, int(c.upgraded), c.enchantment, c.enchant_amount]})
    for k in others:
        options.append({"k": k, "role": ROLE_OTHER})
    chosen, aux = -1, 0
    if choice == "HEAL":
        chosen, aux = 0, ROLE_HEAL
    elif choice == "SMITH":
        aux = ROLE_SMITH
        target = hs._cid((ps.get("upgraded_cards") or [None])[0] or "") if ps.get("upgraded_cards") else ""
        tid = t.cards.get(target)
        for j, o in enumerate(options):
            if o["k"] == "rest.smith" and o["card"][0] == tid and not o["card"][1]:
                chosen = j
                break
        if not distinct:
            raise Drop("smith with no upgradable card")
    elif choice in REST_CHOICE_KIND:
        aux = ROLE_OTHER
        want = REST_CHOICE_KIND[choice]
        for j, o in enumerate(options):
            if o["k"] == want:
                chosen = j
        if chosen < 0:
            raise Drop(f"rest choice {choice} not offered by the relics held")
    else:
        raise Drop(f"rest choice {choice or 'none'} not modelled")
    return [dict(base, kind="rest", options=options, chosen=chosen, aux=aux, aux2=0, offer=0, price=0)]


def replay_prices(visit: dict | None, offers: list[dict]) -> tuple[str, list[int | None]]:
    """The copy's own prices for a shop visit's offers, and how far they can be trusted.

    `visit` is the replay's record of the visit (`rl/human_prior_replay.py`): the copy's shelf with a price on every item, whether the
    replay was clean on entry (nothing before it had left the record's path), whether the shelf equals the record's, and whether the
    prices of what the human bought add up to the gold the record says was spent. Returns `(grade, prices)`, `prices` one per offer
    (None where the copy's shelf has no such item):
      `strict`      the probe's criterion: clean on entry, the shelf equal, the bought items' prices exact, no price-changing relic bought;
      `shelf_only`  the same evidence without the clean path (the shelf and the prices match anyway);
      otherwise     the reason in one word (`no_replay`, `moving`, `shelf_differs`, `price_not_exact`, `offer_missing`)."""
    if visit is None:
        return "no_replay", [None] * len(offers)
    if visit.get("moving"):
        return "moving", [None] * len(offers)
    if not visit.get("shelf_ok"):
        return "shelf_differs", [None] * len(offers)
    if visit.get("purchase") and not visit.get("exact"):
        return "price_not_exact", [None] * len(offers)
    pools = {OFFER_CARD: [list(c) for c in visit["cards"]], OFFER_RELIC: [list(c) for c in visit["relics"]], OFFER_POTION: [list(c) for c in visit["potions"]]}
    prices: list[int | None] = []
    for o in offers:
        pool = pools[o["offer"]]
        hit = next((k for k, e in enumerate(pool) if e[0] == o["id"] and (o["offer"] != OFFER_CARD or bool(e[2]) == bool(o["up"]))), None)
        if hit is None:
            hit = next((k for k, e in enumerate(pool) if e[0] == o["id"]), None)
        prices.append(int(pool.pop(hit)[1]) if hit is not None else None)
    if any(p is None for p in prices):
        return "offer_missing", [None] * len(offers)
    return ("strict" if visit.get("clean") else "shelf_only"), prices


def _shop_specs(base: dict, ps: dict, pt: dict, t: Tables, stats: Counter, price_stats: Counter, removals: int, visit: dict | None = None,
                v1: bool = False) -> list[dict]:
    import human_starts as hs  # noqa: PLC0415

    s = stack()
    gold = base["gold"]
    relics = base["relics"]
    bought_relics = Counter(hs._cid(c) for c in ps.get("bought_relics") or [])
    bought_potions = Counter(hs._cid(c) for c in ps.get("bought_potions") or [])
    slots = s.F.max_potion_slots(10, relics)
    free = len(base["potions"]) < slots
    offers: list[dict] = []
    # purchases the visit made, for the price check: (offer kind, id, colourless?)
    spent_items: list[tuple[int, int, bool]] = []
    # the record lists the cards left on the shelf in `card_choices` and the cards bought in `cards_gained` (a bought colourless card
    # also in `bought_colorless`); relics and potions are listed whole with `was_picked`
    shelf = [(c.get("card") or {}, False) for c in ps.get("card_choices") or []]
    took_cards = [{"id": c["id"], "current_upgrade_level": c.get("current_upgrade_level")} for c in ps.get("cards_gained") or []]
    have = Counter(hs._cid(c["id"]) for c in took_cards)
    for c in ps.get("bought_colorless") or []:
        if have[hs._cid(c)] == 0:
            took_cards.append({"id": c})
    shelf += [(c, True) for c in took_cards]
    for card, took in shelf:
        name = hs._cid(card.get("id") or "")
        ident = t.cards.get(name)
        if ident is None:
            stats["drop|shop|offered card unknown to the copy"] += 1
            continue
        if took:
            spent_items.append((OFFER_CARD, ident, ident in t.colorless))
        offers.append({"offer": OFFER_CARD, "id": ident, "bought": took, "up": int(int(card.get("current_upgrade_level") or 0) >= 1)})
    for c in ps.get("relic_choices") or []:
        name = hs._cid(c.get("choice") or "")
        ident = t.relics.get(name)
        if ident is None:
            stats["drop|shop|offered relic unknown to the copy"] += 1
            continue
        took = bought_relics[name] > 0
        if took:
            bought_relics[name] -= 1
            spent_items.append((OFFER_RELIC, ident, False))
        offers.append({"offer": OFFER_RELIC, "id": ident, "bought": took, "up": 0})
    for c in ps.get("potion_choices") or []:
        name = hs._cid(c.get("choice") or "")
        ident = t.potions.get(name)
        if ident is None:
            stats["drop|shop|offered potion unknown to the copy"] += 1
            continue
        took = bought_potions[name] > 0
        if took:
            bought_potions[name] -= 1
            spent_items.append((OFFER_POTION, ident, False))
        offers.append({"offer": OFFER_POTION, "id": ident, "bought": took, "up": 0})
    spent = int(ps.get("gold_spent") or 0)
    removed = bool(ps.get("cards_removed"))
    rem_price = removal_price(removals, relics, t) if removed else None
    chk = price_check(spent_items, rem_price, spent, relics, t)
    price_stats["visits"] += 1
    price_stats["within10_all"] += chk["within"]
    price_stats["strict_within10_all"] += chk["strict_within"]
    if spent > 0:
        price_stats["visits_with_spend"] += 1
        price_stats["within10_spend"] += chk["within"]
        price_stats["strict_within10_spend"] += chk["strict_within"]
        price_stats["in_band_spend"] += chk["in_band"]
        has_relic = any(o == OFFER_RELIC for o, _, _ in spent_items)
        key = "relic" if has_relic else "no_relic"
        price_stats[f"visits_with_spend|{key}"] += 1
        price_stats[f"within10_spend|{key}"] += chk["within"]
        price_stats[f"in_band_spend|{key}"] += chk["in_band"]
        err = chk["error"]
        price_stats[f"err_bin|{max(-5, min(5, round(err / 10)))}"] += 1
    unmatched = sum(bought_relics.values()) + sum(bought_potions.values())
    if unmatched:
        stats["shop_visits_with_purchase_not_in_offers"] += 1
    grade, copy_prices = replay_prices(visit, offers) if v1 else ("rules", [None] * len(offers))
    if v1:
        stats[f"shop_visit|{grade}"] += 1
        stats[f"shop_visit|act{base['act'] + 1}|{grade}"] += 1
        stats[f"shop_visit_purchase|{grade}"] += int(bool(spent_items or removed))
    out = []
    for o, quoted in zip(offers, copy_prices):
        mid, _, _ = shop_price(o["offer"], o["id"], relics, t)
        price = quoted if quoted is not None else mid
        avail = price <= gold and (o["offer"] != OFFER_POTION or free)
        if o["bought"]:
            avail = True
        opt = {"k": {OFFER_CARD: "shop.card", OFFER_RELIC: "shop.relic", OFFER_POTION: "shop.potion"}[o["offer"]], "price": price, "id": o["id"], "up": o["up"]}
        flags = base["flags"] | (0 if avail else F_UNAVAILABLE)
        if v1 and grade != "strict":
            flags |= F_PRICE_UNVERIFIED | (F_SHELF_ONLY if grade == "shelf_only" else 0)
        out.append(dict(base, kind="shop", options=[{"k": "shop.leave"}, opt], chosen=int(o["bought"]), aux=int(avail), aux2=int(chk["within"]),
                        offer=o["offer"], price=price, flags=flags))
    if out:
        stats[f"shop_visits_built|act{base['act'] + 1}"] += 1
    return out


# ------------------------------------------------------------------------------------------------------------- featurizing


def _option(sp_opt: dict, view, index: int):
    F = stack().F
    k = sp_opt["k"]
    if k == "card.take":
        return F.Option(k, index, card=F.CardRef(sp_opt["card"][0], bool(sp_opt["card"][1])))
    if k == "card.skip":
        return F.Option(k, index)
    if k == "neow.relic":
        return F.Option(k, index, relic=sp_opt["relic"])
    if k == "rest.heal":
        return F.Option(k, 0)
    if k == "rest.smith":
        card = F.CardRef(sp_opt["card"][0], bool(sp_opt["card"][1]), sp_opt["card"][2], sp_opt["card"][3])
        di = next(j for j, c in enumerate(view.deck) if c == card)
        return F.Option(k, 1, deck_index=di)
    if k.startswith("rest."):
        return F.Option(k, stack().V.REST_OPTION_IDS[k])
    if k == "event.option":
        return F.Option(k, sp_opt["event"][1], event=tuple(sp_opt["event"]))
    if k == "shop.leave":
        return F.Option(k, index)
    if k == "shop.card":
        return F.Option(k, index, card=F.CardRef(sp_opt["id"], bool(sp_opt["up"])), price=sp_opt["price"])
    if k == "shop.relic":
        return F.Option(k, index, relic=sp_opt["id"], price=sp_opt["price"])
    if k == "shop.potion":
        return F.Option(k, index, potion=sp_opt["id"], price=sp_opt["price"])
    if k == "map.node":
        return F.Option(k, index, node=(int(sp_opt["node"][0]), int(sp_opt["node"][1]), int(sp_opt["node"][2])))
    raise ValueError(k)


def spec_view(sp: dict):
    """The run view of a spec. A v1 spec carries the act's true map (`map_nodes`) and the node the human stood on (`position`); a v0
    spec has an empty map (the map block and the distance to the boss come from a donor at fit time)."""
    F = stack().F
    return F.RunView(**_view_args(sp), deck=tuple(F.CardRef(c[0], bool(c[1]), c[2], c[3]) for c in sp["deck"]),
                     max_potion_slots=F.max_potion_slots(10, sp["relics"]), map_nodes=sp.get("map_nodes", ()), position=sp.get("position", (0, 0)))


def spec_rows(sp: dict) -> list[np.ndarray]:
    """The 1,855-wide afterstate base vector of every option of a spec: with the true map (v1), or an empty one (v0; the map block and
    the distance-to-boss scalar are put in at fit time from a donor)."""
    F = stack().F
    view = spec_view(sp)
    rows = []
    for j, o in enumerate(sp["options"]):
        opt = _option(o, view, j)
        rows.append(F.encode(view) if opt.kind in F.UNCHANGED_KINDS else F.encode_afterstate(view, opt))
    return rows


def pack_dense(bases: np.ndarray, h: np.ndarray) -> np.ndarray:
    """A base vectors (rows x 1855) and their fight blocks -> the stored dense part (rows x 88)."""
    F = stack().F
    out = np.zeros((len(bases), N_DENSE), dtype=np.float32)
    out[:, S_A] = bases[:, 0:22]
    out[:, S_A.start + 15] = 0.0  # the distance to the boss comes from the donor
    out[:, S_B] = bases[:, F.B_OFFSET:F.B_OFFSET + 21]
    out[:, S_D] = bases[:, F.D_OFFSET:F.D_OFFSET + F.D_SIZE]
    out[:, S_X] = bases[:, F.X_OFFSET:F.X_OFFSET + F.X_SIZE]
    out[:, S_H] = h
    return out


def pack_dense_v1(bases: np.ndarray, h: np.ndarray) -> np.ndarray:
    """A base vectors (rows x 1855) with the true map and their fight blocks -> the stored dense part of a v1 shard (rows x 113):
    A with the distance to the boss, the first 21 of B, the map block C, D, X and H."""
    F = stack().F
    out = np.zeros((len(bases), N_DENSE_V1), dtype=np.float32)
    out[:, S1_A] = bases[:, 0:22]
    out[:, S1_B] = bases[:, F.B_OFFSET:F.B_OFFSET + 21]
    out[:, S1_C] = bases[:, F.C_OFFSET:F.C_OFFSET + F.C_SIZE]
    out[:, S1_D] = bases[:, F.D_OFFSET:F.D_OFFSET + F.D_SIZE]
    out[:, S1_X] = bases[:, F.X_OFFSET:F.X_OFFSET + F.X_SIZE]
    out[:, S1_H] = h
    return out


def pack_counts(bases: np.ndarray) -> np.ndarray:
    F = stack().F
    counts = bases[:, F.E_OFFSET:F.COUNTS_END]
    if counts.size and (counts.min() < 0 or counts.max() > 255 or not np.array_equal(counts, np.round(counts))):
        raise ValueError("a count is not a small whole number: the uint8 store would lose it")
    return counts.astype(np.uint8)


def load_donors(path: Path):
    """The donor table: keys (region, row) -> (n, map block (n x 25), distance-to-boss scalar (n))."""
    z = np.load(path, allow_pickle=False)
    return {"keys": z["keys"].astype(np.int32), "n": z["n"].astype(np.int32), "c": z["c"], "a15": z["a15"]}


def donor_index(donors: dict) -> dict[tuple[int, int], int]:
    return {(int(r), int(w)): k for k, (r, w) in enumerate(donors["keys"])}


_WORK: dict = {}


def _worker_state():
    """The tables, the run judgement, the donors and the exam ids, loaded once per worker process."""
    if not _WORK:
        import human_starts as hs  # noqa: PLC0415

        s = stack()
        _WORK["t"] = Tables()
        _WORK["judge"] = s.M.load_numpy(os.environ.get("HP_JUDGE", str(GEN11)))
        donors = load_donors(Path(os.environ["HP_DONORS"])) if os.environ.get("HP_DONORS") else None
        _WORK["donors"] = donors
        _WORK["dindex"] = donor_index(donors) if donors else {}
        _WORK["truth_dir"] = os.environ.get("HP_TRUTH", "")
        _WORK["exam"] = hs.read_exam_ids(os.environ.get("HP_EXAM_IDS", str(EXAM_IDS)))
    return _WORK


def load_truth_chunk(path: str | Path) -> dict[str, dict]:
    """The truth records of one chunk (`rl/human_prior_replay.py`), by run id."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        doc = json.load(fh)
    if len(doc["run_ids"]) != len(doc["runs"]):
        raise ValueError(f"truth chunk {path}: run ids and records differ in number")
    return {rid: run for rid, run in zip(doc["run_ids"], doc["runs"])}


def build_chunk(job: dict) -> dict:
    """A chunk of raw run lines -> one shard file. Runs in a pool worker. Returns the counters and the run ids written."""
    import human_starts as hs  # noqa: PLC0415

    w = _worker_state()
    t, judge, donors, dindex, exam = w["t"], w["judge"], w["donors"], w["dindex"], w["exam"]
    stats, seen, price_stats = Counter(), Counter(), Counter()
    specs: list[dict] = []
    run_ids: list[str] = []
    truth_by_id = load_truth_chunk(job["truth"]) if job.get("truth") else None
    for line in job["lines"]:
        r = json.loads(line)
        rid = hs.run_id(r)
        if hs.is_exam(rid, exam):
            raise AssertionError(f"human_prior: run {rid} belongs to the exam fifth and may not be read")
        if truth_by_id is not None and rid not in truth_by_id:
            raise AssertionError(f"human_prior: run {rid} has no record in the truth chunk {job['truth']}")
        got = run_decisions(r, t, stats, seen, price_stats, bool(job.get("events")), truth_by_id[rid] if truth_by_id is not None else None)
        if got:
            run_ids.append(rid)
            for sp in got:
                sp["run_ix"] = len(run_ids) - 1
            specs.extend(got)
    hs.assert_no_exam(run_ids, exam)
    path = Path(job["out"])
    n = write_shard(path, specs, run_ids, judge, dindex, donors)
    stats["decisions"] = len(specs)
    stats["rows"] = n
    return {"stats": dict(stats), "seen": dict(seen), "price": dict(price_stats), "runs": len(run_ids), "path": str(path)}


def write_shard(path: Path, specs: list[dict], run_ids: list[str], judge, dindex: dict, donors: dict | None) -> int:
    """Featurize `specs` and write them as one npz shard (atomically). Returns the number of rows. Specs with a true map (`map_nodes`)
    make a v1 shard: the map block is stored with the row and the fight block is computed from the true map; the others make a v0
    shard (no map block, a donor's at fit time)."""
    F = stack().F
    v1 = bool(specs) and all("map_nodes" in sp for sp in specs)
    if any("map_nodes" in sp for sp in specs) and not v1:
        raise ValueError("write_shard: some specs carry a true map and some do not")
    all_rows: list[np.ndarray] = []
    dec = np.zeros((len(specs), len(COLS)), dtype=np.int32)
    role: list[int] = []
    ident: list[int] = []
    start = 0
    for d, sp in enumerate(specs):
        rows = spec_rows(sp)
        all_rows.extend(rows)
        dec[d, C["kind"]] = KIND[sp["kind"]]
        dec[d, C["fifth"]] = sp["fifth"]
        dec[d, C["act"]] = sp["act"]
        dec[d, C["floor"]] = sp["floor"]
        dec[d, C["region"]] = sp["region"]
        dec[d, C["row"]] = sp["row"]
        dec[d, C["boss"]] = sp["boss"]
        dec[d, C["run"]] = sp["run_ix"]
        dec[d, C["won"]] = int(sp["won"])
        dec[d, C["final"]] = sp["final"]
        dec[d, C["start"]] = start
        dec[d, C["n"]] = len(rows)
        dec[d, C["chosen"]] = sp["chosen"]
        dec[d, C["aux"]] = sp["aux"]
        dec[d, C["aux2"]] = sp["aux2"]
        dec[d, C["flags"]] = sp["flags"]
        dec[d, C["offer"]] = sp["offer"]
        dec[d, C["price"]] = sp["price"]
        dec[d, C["gold"]] = sp["gold"]
        dec[d, C["hp"]] = sp["hp"]
        dec[d, C["max_hp"]] = sp["max_hp"]
        for o in sp["options"]:
            if o["k"] == "map.node":  # a route option: the room's type as its role, its column as its id
                role.append(int(o["node"][2]))
                ident.append(int(o["node"][0]))
                continue
            role.append(o.get("role", 0))
            ident.append(int(o.get("id", o.get("relic", (o.get("card") or [0])[0]))))
        start += len(rows)
    if all_rows:
        bases = np.stack(all_rows).astype(np.float32)
    else:
        bases = np.zeros((0, F.BASE_SIZE), dtype=np.float32)
    # the fight block H, once per (region, boss), with the first donor's distance to the boss for the row key
    h = np.zeros((len(bases), F.FIGHT_BLOCK_SIZE), dtype=np.float32)
    if len(bases):
        reg = np.repeat(dec[:, C["region"]], dec[:, C["n"]])
        bos = np.repeat(dec[:, C["boss"]], dec[:, C["n"]])
        row = np.repeat(dec[:, C["row"]], dec[:, C["n"]])
        a15 = np.zeros(len(bases), dtype=np.float32)
        if donors is not None and not v1:
            for key in np.unique(np.stack([reg, row], 1), axis=0):
                k = dindex.get((int(key[0]), int(key[1])))
                if k is not None and donors["n"][k] > 0:
                    a15[(reg == key[0]) & (row == key[1])] = donors["a15"][k, 0]
        for rg, bs in {(int(a), int(b)) for a, b in zip(reg, bos)}:
            sel = np.nonzero((reg == rg) & (bos == bs))[0]
            view = F.RunView(ascension=10, act_index=0, act_region=rg, floor=1, hp=1, max_hp=1, gold=0, deck=(), relics=(), potions=(),
                             max_potion_slots=2, map_nodes=(), position=(0, 0), boss=bs)
            queries = F.fight_queries(view)
            b = bases[sel].copy()
            if not v1:
                b[:, 15] = a15[sel]
            h[sel] = judge.fight.summary(b[:, :judge.base_size], queries)
    if v1:
        dense = pack_dense_v1(bases, h)
    else:
        dense = pack_dense(bases, h) if len(bases) else np.zeros((0, N_DENSE), np.float32)
    counts = pack_counts(bases) if len(bases) else np.zeros((0, F.COUNTS_END - F.E_OFFSET), np.uint8)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, dec=dec, dense=dense, counts=counts, role=np.array(role, dtype=np.int8), ident=np.array(ident, dtype=np.int32),
                            run_ids=np.array(run_ids, dtype="U32"), cols=np.array(COLS))
    os.replace(tmp, path)
    return len(bases)


# ------------------------------------------------------------------------------------------------------------------ donors


def build_donors(runs_root: Path, gens: list[int], per_key: int, max_runs: int, out: Path, seed: int = 20261105) -> dict:
    """Donor maps from the run judgement's own self-play states, by (region, row of the act): the map-ahead block and the distance to
    the boss of up to `per_key` states (at most two per run and key, so one run's map does not fill a key)."""
    import random  # noqa: PLC0415

    s = stack()
    F = s.F
    shards_mod = __import__("run_train.shards", fromlist=["x"])
    files: list[Path] = []
    for job in sorted(Path(runs_root).glob("*selfplay*")):
        meta = job / "shards" / "shards.json"
        if not meta.is_file():
            continue
        try:
            g = int(json.loads(meta.read_text(encoding="utf-8")).get("generation", -1))
        except (ValueError, OSError):
            continue
        if g in gens:
            files.extend(sorted((job / "shards").glob("r*.jsonl.gz")))
    random.Random(seed).shuffle(files)
    pool: dict[tuple[int, int], list] = defaultdict(list)
    runs = 0
    for fpath in files:
        with gzip.open(fpath, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                runs += 1
                taken: Counter = Counter()
                cache: dict[int, tuple] = {}
                for d in rec["decisions"]:
                    st = d["state"]
                    key = (int(st["act_region"]), int(st["position"][1]))
                    if len(pool[key]) >= per_key or taken[key] >= 2:
                        continue
                    if st["map"] not in cache:
                        cache[st["map"]] = shards_mod._nodes_from_json(rec["maps"][st["map"]]["nodes"])
                    nodes = cache[st["map"]]
                    pos = (int(st["position"][0]), int(st["position"][1]))
                    c = F._map_ahead(nodes, pos).copy()
                    a15 = (F._boss_row(nodes) - pos[1]) / F.MAP_ROWS
                    pool[key].append((c, a15))
                    taken[key] += 1
                if runs >= max_runs:
                    break
        if runs >= max_runs:
            break
    keys = sorted(pool)
    c_arr = np.zeros((len(keys), per_key, F.C_SIZE), dtype=np.float32)
    a_arr = np.zeros((len(keys), per_key), dtype=np.float32)
    n_arr = np.zeros(len(keys), dtype=np.int32)
    for k, key in enumerate(keys):
        for j, (c, a15) in enumerate(pool[key]):
            c_arr[k, j], a_arr[k, j] = c, a15
        n_arr[k] = len(pool[key])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keys=np.array(keys, dtype=np.int32), n=n_arr, c=c_arr, a15=a_arr)
    cover = {f"region{r}": {str(w): int(n_arr[k]) for k, (rr, w) in enumerate(keys) if rr == r} for r in (1, 2, 3, 4)}
    return {"runs_read": runs, "gens": gens, "per_key": per_key, "keys": len(keys), "coverage": cover}


# ------------------------------------------------------------------------------------------------------------------- build


def contract_report(built: dict, reference: dict) -> dict:
    """The built counts against counts.json (`decision_kinds_training_fifths`): per kind and act the screens the build saw, the
    reference's, and the decisions written. `seen` and the reference use the same definitions (card-reward screens off the shop, rest
    points with a choice, ancient screens, shop visits)."""
    ref = reference["decision_kinds_training_fifths"]
    names = {"card": "card_reward_screens", "rest": "rest_points_with_choice", "ancient": "ancient_screens", "shop": "points:shop"}
    out: dict = {}
    for kind, key in names.items():
        for act in (1, 2, 3):
            want = int(ref[f"act{act}"].get(key, 0))
            got = int(built["seen"].get(f"act{act}|{kind}", 0))
            out[f"{kind}|act{act}"] = {"reference": want, "seen": got, "equal": want == got}
    return out


def merge_counts(results: list[dict]) -> tuple[Counter, Counter, Counter]:
    stats, seen, price = Counter(), Counter(), Counter()
    for r in results:
        stats.update(r["stats"])
        seen.update(r["seen"])
        price.update(r["price"])
    return stats, seen, price


def summarize(out_dir: Path, results: list[dict], parent: Counter) -> dict:
    stats, seen, price = merge_counts(results)
    built = Counter()
    by_fifth = Counter()
    won_decisions = Counter()
    flags = Counter()
    run_sets: dict[tuple, set] = defaultdict(set)
    for r in results:
        with np.load(r["path"], allow_pickle=False) as z:
            dec, run_ids = z["dec"], z["run_ids"]
        for d in dec:
            kind, act = KIND_NAME[int(d[C["kind"]])], int(d[C["act"]]) + 1
            built[f"{kind}|act{act}"] += 1
            by_fifth[f"{kind}|fifth{int(d[C['fifth']])}"] += 1
            if int(d[C["won"]]):
                won_decisions[kind] += 1
            for bit, name in ((F_SPLIT, "split_screen"), (F_PRICE_UNVERIFIED, "price_unverified"), (F_UNAVAILABLE, "unavailable"),
                              (F_REPLAY_MISMATCH, "replay_mismatch"), (F_BOSS_GUESSED, "boss_guessed"),
                              (F_GATE_UNKNOWN, "gate_unknown"), (F_SHELF_ONLY, "shelf_only")):
                if int(d[C["flags"]]) & bit:
                    flags[f"{kind}|{name}"] += 1
            run_sets[(kind, act)].add(str(run_ids[int(d[C["run"]])]))
    drops = {k.removeprefix("drop|"): v for k, v in sorted(stats.items()) if k.startswith("drop|")}
    checks = {k.removeprefix("check|"): v for k, v in sorted(stats.items()) if k.startswith("check|")}
    rebuild_checks = {"rest_hp_matches_previous_floor_plus_heal": round(checks.get("rest_hp_ok", 0) / max(checks.get("rest_hp_n", 0), 1), 4),
                      "shop_gold_matches_previous_floor": round(checks.get("shop_gold_ok", 0) / max(checks.get("shop_gold_n", 0), 1), 4),
                      "rest_hp_n": checks.get("rest_hp_n", 0), "shop_gold_n": checks.get("shop_gold_n", 0)}
    doc = {"written_by": "rl/human_prior.py build (aggregates only)", "source": SPIRE_CODEX, "parent": dict(parent),
           "seen": dict(sorted(seen.items())), "built": dict(sorted(built.items())), "built_by_fifth": dict(sorted(by_fifth.items())),
           "winner_decisions": dict(won_decisions), "flags": dict(sorted(flags.items())), "drops": drops,
           "runs_replay_mismatch": int(stats.get("runs_replay_mismatch", 0)), "runs_with_decisions": sum(r["runs"] for r in results),
           "rows": int(stats.get("rows", 0)), "rebuild_checks": rebuild_checks, "shop_price_check": price_summary(price),
           "cells_runs": {f"{k[0]}|act{k[1]}": (len(v) if len(v) >= MIN_CELL else f"<{MIN_CELL}") for k, v in sorted(run_sets.items())}}
    if stats.get("runs_read"):
        doc["true_map"] = truth_summary(stats)
    return doc


def truth_summary(stats: Counter) -> dict:
    """v1's counts: the runs with a true map and those left out (by reason and fifth), the route steps and decisions per act, and the
    shop visits by how far the replay vouches for them."""
    def pick(prefix: str) -> dict:
        return {k.removeprefix(prefix): v for k, v in sorted(stats.items()) if k.startswith(prefix)}

    route: dict = {}
    for act in (1, 2, 3):
        tag = f"route|act{act}|"
        route[f"act{act}"] = {name: int(stats.get(tag + name, 0)) for name in ("steps", "single_room", "choices", "type_choices")}
        route[f"act{act}"]["dropped"] = {k.split("|", 3)[3]: v for k, v in sorted(stats.items()) if k.startswith(f"drop|act{act}|route|")}
    visits = {g: int(stats.get(f"shop_visit|{g}", 0)) for g in ("strict", "shelf_only", "shelf_differs", "price_not_exact", "moving", "offer_missing", "no_replay")}
    return {"runs_read": int(stats.get("runs_read", 0)), "runs_true_map": int(stats.get("runs_true_map", 0)),
            "runs_left_out": {k: v for k, v in pick("runs_left_out|").items() if "|" not in k},
            "runs_left_out_by_fifth": {k: v for k, v in pick("runs_left_out|").items() if "|" in k},
            "runs_read_by_fifth": pick("runs_read|"), "runs_true_map_by_fifth": pick("runs_true_map|"),
            "runs_position_ambiguous_somewhere": int(stats.get("runs_position_ambiguous", 0)), "runs_with_winged_boots": int(stats.get("runs_with_boots", 0)),
            "route": route,
            "shop_visits_by_grade": visits, "shop_visits_by_grade_and_act": {a: {g: int(stats.get(f"shop_visit|{a}|{g}", 0)) for g in visits} for a in ("act1", "act2", "act3")},
            "shop_visits_with_a_purchase_by_grade": pick("shop_visit_purchase|")}


def price_summary(price: Counter) -> dict:
    v, vs = price.get("visits", 0), price.get("visits_with_spend", 0)

    def rate(num, den):
        return round(num / den, 4) if den else None

    return {"visits": v, "visits_with_spend": vs, "tolerance_gold": PRICE_TOLERANCE,
            "within10_all": rate(price.get("within10_all", 0), v), "strict_within10_all": rate(price.get("strict_within10_all", 0), v),
            "within10_spend": rate(price.get("within10_spend", 0), vs), "strict_within10_spend": rate(price.get("strict_within10_spend", 0), vs),
            "in_rules_band_spend": rate(price.get("in_band_spend", 0), vs),
            "within10_spend_with_relic": rate(price.get("within10_spend|relic", 0), price.get("visits_with_spend|relic", 0)),
            "within10_spend_without_relic": rate(price.get("within10_spend|no_relic", 0), price.get("visits_with_spend|no_relic", 0)),
            "visits_with_spend_with_relic": price.get("visits_with_spend|relic", 0),
            "in_band_spend_with_relic": rate(price.get("in_band_spend|relic", 0), price.get("visits_with_spend|relic", 0)),
            "in_band_spend_without_relic": rate(price.get("in_band_spend|no_relic", 0), price.get("visits_with_spend|no_relic", 0)),
            "error_bins_gold_per_10": {k.split("|", 1)[1]: n for k, n in sorted(price.items()) if k.startswith("err_bin|")},
            "gate": PRICE_GATE, "gate_share_used": "within10_spend"}


def mark_price_unverified(out_dir: Path, verified: bool) -> None:
    """Shop rows of an unverified price check are marked (flag bit) so the fit leaves them out."""
    if verified:
        return
    for p in sorted(Path(out_dir).glob("shard-*.npz")):
        with np.load(p, allow_pickle=False) as zf:
            z = dict(zf)
        shop = z["dec"][:, C["kind"]] == KIND["shop"]
        z["dec"][shop, C["flags"]] |= F_PRICE_UNVERIFIED
        tmp = Path(str(p) + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **z)
        os.replace(tmp, p)


def parse_ascensions(text: str) -> tuple[int, ...]:
    """'10' -> (10,); '11-20' -> (11, ..., 20); '11,13' -> (11, 13). The default build reads Ascension 10 only."""
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.extend(range(lo, hi + 1))
        elif part:
            out.append(int(part))
    if not out:
        raise ValueError(f"no Ascension level in {text!r}")
    return tuple(out)


def cmd_build(a: argparse.Namespace) -> int:
    import multiprocessing as mp  # noqa: PLC0415

    import human_starts as hs  # noqa: PLC0415
    from spawn_pool import SupervisedPool  # noqa: PLC0415

    if a.procs > 4:
        raise SystemExit("human_prior: at most 4 processes")
    below_normal()
    out_dir = Path(a.out_dir)
    dec_dir = out_dir / (a.decisions or ("decisions-v1" if a.truth else "decisions"))
    if a.clean and dec_dir.is_dir():
        for p in dec_dir.glob("shard-*.npz*"):
            p.unlink()
    dec_dir.mkdir(parents=True, exist_ok=True)
    truth_dir = Path(a.truth) if a.truth else None
    donors_path = out_dir / "donors.npz"
    if truth_dir is None:
        if not donors_path.is_file():
            print("human_prior: no donors.npz: run `donors` first (the map block would be empty)", file=sys.stderr)
            return 2
        os.environ["HP_DONORS"] = str(donors_path)
    elif not truth_dir.is_dir():
        print(f"human_prior: no truth folder {truth_dir}: run rl/human_prior_replay.py first", file=sys.stderr)
        return 2
    os.environ["HP_JUDGE"] = str(a.judge)
    os.environ["HP_EXAM_IDS"] = str(a.exam_ids)
    if a.lib:
        os.environ["HP_LIB"] = str(a.lib)
    exam = hs.read_exam_ids(a.exam_ids)
    t0 = time.time()
    parent: Counter = Counter()
    results: list[dict] = []
    pool = SupervisedPool(mp.get_context("spawn"), max(1, a.procs), initializer=below_normal, run_timeout_s=1800.0)
    pending: list[tuple[int, dict]] = []
    retried: set[int] = set()
    chunk: list[str] = []
    ix = 0
    chunks: dict[int, dict] = {}

    def submit_ready() -> None:
        while pending and pool.free():
            k, job = pending.pop(0)
            pool.submit(k, build_chunk, (job,))

    def drain() -> None:
        for d in pool.collect():
            if d.status == "ok":
                results.append(d.value)
                if len(results) % 20 == 0:
                    print(f"human_prior: {len(results)} chunks done ({time.time() - t0:.0f} s)", flush=True)
            elif d.key not in retried:
                retried.add(d.key)
                pending.append((d.key, chunks[d.key]))
            else:
                parent["chunks_lost"] += 1
                print(f"human_prior: chunk {d.key} lost: {d.status} {d.value}", file=sys.stderr)

    try:
        for line in iter_run_lines(Path(a.corpus), exam, parent, a.limit_runs, parse_ascensions(getattr(a, "ascensions", "10"))):
            chunk.append(line)
            if len(chunk) >= a.chunk_runs:
                job = {"lines": chunk, "out": str(dec_dir / f"shard-{ix:05d}.npz"), "events": bool(a.events),
                       "truth": str(truth_dir / f"truth-{ix:05d}.json.gz") if truth_dir else ""}
                chunks[ix] = job
                pending.append((ix, job))
                ix += 1
                chunk = []
            submit_ready()
            drain()
            while len(pending) > 2 * a.procs:  # do not read far ahead of the workers
                pool.wait(0.2)
                submit_ready()
                drain()
        if chunk:
            job = {"lines": chunk, "out": str(dec_dir / f"shard-{ix:05d}.npz"), "events": bool(a.events),
                    "truth": str(truth_dir / f"truth-{ix:05d}.json.gz") if truth_dir else ""}
            chunks[ix] = job
            pending.append((ix, job))
        while pending or pool.inflight():
            submit_ready()
            pool.wait(0.3)
            drain()
    finally:
        pool.close()
    summary = summarize(dec_dir, results, parent)
    pc = summary["shop_price_check"]
    verified = pc["within10_spend"] is not None and pc["within10_spend"] >= PRICE_GATE
    if truth_dir is None:
        summary["shop_price_check"]["verified"] = bool(verified)
        mark_price_unverified(dec_dir, verified)
    else:  # v1: each visit is graded on its own against the replay (the rules' band above is the old check, kept for the comparison)
        summary["shop_price_check"]["verified"] = "per visit, against the replay (flags on the rows)"
    ref = json.loads(Path(a.reference).read_text(encoding="utf-8")) if Path(a.reference).is_file() else None
    if ref and not a.limit_runs:
        summary["contract"] = contract_report(summary, ref)
    summary["seconds"] = round(time.time() - t0, 1)
    (dec_dir / "counts.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("built", "drops", "shop_price_check", "seconds")}, indent=1))
    return 0


def cmd_donors(a: argparse.Namespace) -> int:
    below_normal()
    t0 = time.time()
    info = build_donors(Path(a.runs_root), [int(x) for x in a.gens.split(",")], a.per_key, a.max_runs, Path(a.out_dir) / "donors.npz")
    info["seconds"] = round(time.time() - t0, 1)
    (Path(a.out_dir) / "donors.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "coverage"}))
    for r, cov in info["coverage"].items():
        print(r, cov)
    return 0


# ---------------------------------------------------------------------------------------------------------- the data for a fit

DENSE_WIDTH = 350  # A 22, B 258, C 25, D 25, X 12, H 8: the run judgement's dense input
HP_CAPS = np.array([1.0, 6.0, 6.0, 6.0], dtype=np.float32)  # models.HP_CAPS: the HP scalars are clamped from above


def donor_keys_for(donors: dict, regions: np.ndarray, rows: np.ndarray, min_n: int = 8) -> tuple[np.ndarray, Counter]:
    """For each decision the donor table key to draw from: its own (region, row), or the nearest row of the region with enough donors
    (Act 3's late rows are sparse), or -1 (a zero map block) when the region has none. Returns the key indices and a coverage count."""
    index = donor_index(donors)
    n = donors["n"]
    by_region: dict[int, list[int]] = defaultdict(list)
    for (rg, rw), k in index.items():
        if n[k] >= min_n:
            by_region[rg].append(rw)
    out = np.full(len(regions), -1, dtype=np.int64)
    cover: Counter = Counter()
    cache: dict[tuple[int, int], int] = {}
    for d, (rg, rw) in enumerate(zip(regions.tolist(), rows.tolist())):
        key = (rg, rw)
        if key not in cache:
            k = index.get(key)
            if k is not None and n[k] >= min_n:
                cache[key] = k
                cover["own"] += 1
            else:
                rws = by_region.get(rg)
                if rws:
                    near = min(rws, key=lambda x: (abs(x - rw), x))
                    cache[key] = index[(rg, near)]
                    cover["borrowed"] += 1
                else:
                    cache[key] = -1
                    cover["none"] += 1
        out[d] = cache[key]
    return out, cover


class Dataset:
    """The shards' decisions of some fifths, in memory: the rows (dense part, counts), the decisions and, per kind, the padded
    matrix of each decision's row indices (-1 pads), the roles of the rows and the donor keys."""

    def __init__(self, dec: np.ndarray, dense: np.ndarray, counts: np.ndarray, role: np.ndarray, ident: np.ndarray, run_ids: list[str]):
        self.dec, self.dense, self.counts, self.role, self.ident, self.run_ids = dec, dense, counts, role, ident, run_ids
        self.kind_idx: dict[str, np.ndarray] = {}
        self.rows_mat: dict[str, np.ndarray] = {}
        self.dkey: np.ndarray | None = None
        self.coverage: Counter = Counter()
        self.has_map = dense.shape[1] == N_DENSE_V1  # v1 shards carry the true map block with every row; v0's come from a donor

    @classmethod
    def load(cls, dec_dir: Path | Iterable[tuple[Path, bool]], fifths: Iterable[int], donors: dict | None = None, kinds: Iterable[str] | None = None):
        """The decisions of `fifths` (and, with `kinds`, only those kinds) from the shards of `dec_dir`. `dec_dir` may also be a list of
        `(folder, won_only)` parts (the strong sets of `rl/strong_sets.py`): the parts' decisions are joined, and a won-only part keeps
        the decisions of winning runs."""
        fifths = set(fifths)
        kind_ids = [KIND[k] for k in kinds] if kinds is not None else None
        parts = [(Path(dec_dir), False)] if isinstance(dec_dir, (str, Path)) else [(Path(d), bool(w)) for d, w in dec_dir]
        decs, denses, counts_, roles, idents = [], [], [], [], []
        run_ids: list[str] = []
        row_off = 0
        shards = [(p, won_only) for d, won_only in parts for p in sorted(d.glob("shard-*.npz"))]
        for p, won_only in shards:
            with np.load(p, allow_pickle=False) as z:
                dec = z["dec"]
                keep = np.isin(dec[:, C["fifth"]], list(fifths))
                if kind_ids is not None:
                    keep &= np.isin(dec[:, C["kind"]], kind_ids)
                if won_only:
                    keep &= dec[:, C["won"]] == 1
                if not keep.any():
                    continue
                d = dec[keep].copy()
                starts, ns = d[:, C["start"]], d[:, C["n"]]
                sel = np.concatenate([np.arange(s, s + n) for s, n in zip(starts, ns)]) if len(d) else np.zeros(0, dtype=np.int64)
                new_start = np.concatenate([[0], np.cumsum(ns)[:-1]]) if len(d) else np.zeros(0, dtype=np.int64)
                d[:, C["start"]] = new_start + row_off
                d[:, C["run"]] += len(run_ids)
                run_ids.extend(str(x) for x in z["run_ids"])
                decs.append(d)
                denses.append(z["dense"][sel])
                counts_.append(z["counts"][sel])
                roles.append(z["role"][sel])
                idents.append(z["ident"][sel])
                row_off += len(sel)
        if not decs:
            raise SystemExit(f"human_prior: no shard in {[str(d) for d, _ in parts]} holds fifths {sorted(fifths)}")
        if len({d.shape[1] for d in denses}) != 1:
            raise SystemExit(f"human_prior: the shards of {dec_dir} mix the v0 and the v1 layout")
        ds = cls(np.concatenate(decs), np.concatenate(denses), np.concatenate(counts_), np.concatenate(roles), np.concatenate(idents), run_ids)
        ds._index(donors)
        return ds

    def _index(self, donors: dict | None) -> None:
        dec = self.dec
        for kind, k in KIND.items():
            self.kind_idx[kind] = np.nonzero(dec[:, C["kind"]] == k)[0]
        if donors is not None and not self.has_map:
            self.dkey, self.coverage = donor_keys_for(donors, dec[:, C["region"]], dec[:, C["row"]])
        else:
            self.dkey = np.full(len(dec), -1, dtype=np.int64)

    def matrix(self, idx: np.ndarray) -> np.ndarray:
        """The padded (decisions x options) matrix of row indices for the decisions `idx`."""
        n = self.dec[idx, C["n"]]
        k = int(n.max()) if len(idx) else 1
        m = np.full((len(idx), k), -1, dtype=np.int64)
        st = self.dec[idx, C["start"]]
        for j in range(k):
            ok = n > j
            m[ok, j] = st[ok] + j
        return m


def assemble_np(dense: np.ndarray, a15: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Stored dense rows (n x 88), the donors' distance-to-boss scalar (n) and map block (n x 25) -> the judge's 350-wide dense input."""
    n = len(dense)
    x = np.zeros((n, DENSE_WIDTH), dtype=np.float32)
    x[:, 0:22] = dense[:, S_A]
    x[:, 15] = a15
    x[:, 0:4] = np.minimum(x[:, 0:4], HP_CAPS)
    x[:, 22:43] = dense[:, S_B]
    x[:, 280:305] = c
    x[:, 305:330] = dense[:, S_D]
    x[:, 330:342] = dense[:, S_X]
    x[:, 342:350] = dense[:, S_H]
    return x


def assemble_np_v1(dense: np.ndarray) -> np.ndarray:
    """Stored v1 dense rows (n x 113: the true map block and distance to the boss are in them) -> the judge's 350-wide dense input."""
    n = len(dense)
    x = np.zeros((n, DENSE_WIDTH), dtype=np.float32)
    x[:, 0:22] = dense[:, S1_A]
    x[:, 0:4] = np.minimum(x[:, 0:4], HP_CAPS)
    x[:, 22:43] = dense[:, S1_B]
    x[:, 280:305] = dense[:, S1_C]
    x[:, 305:330] = dense[:, S1_D]
    x[:, 330:342] = dense[:, S1_X]
    x[:, 342:350] = dense[:, S1_H]
    return x


def fixed_pick(didx: np.ndarray, n: np.ndarray) -> np.ndarray:
    """A deterministic donor index per decision (validation and agreement): a hash of the decision's position."""
    h = (didx.astype(np.int64) * 1103515245 + 12345) % 2147483648
    return h % np.maximum(n, 1)


def row_donors(ds: Dataset, donors: dict, didx: np.ndarray, rows_per_dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row of the decisions `didx` (rows laid out decision by decision): the fixed donor's distance scalar and map block."""
    key = ds.dkey[didx]
    n = np.where(key >= 0, donors["n"][np.maximum(key, 0)], 0)
    pick = fixed_pick(didx, n)
    a15 = np.where(key >= 0, donors["a15"][np.maximum(key, 0), pick], 0.0).astype(np.float32)
    c = np.where((key >= 0)[:, None], donors["c"][np.maximum(key, 0), pick], 0.0).astype(np.float32)
    return np.repeat(a15, rows_per_dec), np.repeat(c, rows_per_dec, axis=0)


def row_scores_np(ds: Dataset, donors: dict | None, score_fn, chunk: int = 20000, extra: np.ndarray | None = None) -> np.ndarray:
    """Score every row of the dataset with `score_fn(dense350, counts_f32) -> (n,)`: with each decision's fixed donor (a v0 dataset), or
    with the row's own true map (a v1 one). `extra` (one number per row) is passed on as a third argument, for a prior with an extra input."""
    out = np.zeros(len(ds.dense), dtype=np.float32)
    n_dec = len(ds.dec)
    # rows are decision-contiguous in decision order
    if not ds.has_map:
        rows_per = ds.dec[:, C["n"]]
        a15_all, c_all = row_donors(ds, donors, np.arange(n_dec), rows_per)
    for s in range(0, len(ds.dense), chunk):
        e = min(s + chunk, len(ds.dense))
        x = assemble_np_v1(ds.dense[s:e]) if ds.has_map else assemble_np(ds.dense[s:e], a15_all[s:e], c_all[s:e])
        cnt = ds.counts[s:e].astype(np.float32)
        out[s:e] = score_fn(x, cnt) if extra is None else score_fn(x, cnt, extra[s:e])
    return out


def _bool_unavailable(ds: Dataset, shops_ok: bool, drop_mismatch: bool, verified_only: bool = True, shelf_only_ok: bool = False) -> np.ndarray:
    """Which decisions a fit and a validation leave out: replay mismatches, and (shops) unavailable offers and prices not verified
    (`verified_only`; `shelf_only_ok` lets through the rows whose shelf and prices matched the replay off the record's clean path)."""
    f = ds.dec[:, C["flags"]]
    bad = np.zeros(len(f), dtype=bool)
    if drop_mismatch:
        bad |= (f & F_REPLAY_MISMATCH) != 0
    shop = ds.dec[:, C["kind"]] == KIND["shop"]
    bad |= shop & ((f & F_UNAVAILABLE) != 0)
    if not shops_ok:
        bad |= shop
    elif verified_only:
        unverified = (f & F_PRICE_UNVERIFIED) != 0
        if shelf_only_ok:
            unverified &= (f & F_SHELF_ONLY) == 0
        bad |= shop & unverified
    return bad


def _lse(s: np.ndarray) -> np.ndarray:
    mx = s.max(1, keepdims=True)
    return mx[:, 0] + np.log(np.exp(s - mx).sum(1))


def metrics_from_scores(ds: Dataset, scores: np.ndarray, valid: np.ndarray) -> dict:
    """Negative log-likelihood and top-1 agreement per kind and act from row scores (higher = preferred), over the decisions in `valid`.
    Cards and ancients: a softmax over the options. Rests: the kind (heal / smith / other) by the best option's role, the kind's
    marginal likelihood, and the smith target among the smith options (when the record names it). Shops: buy against leave."""
    out: dict = {}
    dec = ds.dec

    def put(name: str, nll: np.ndarray, hit: np.ndarray, act: np.ndarray) -> None:
        if not len(nll):
            return
        out[name] = {"n": int(len(nll)), "nll": float(np.mean(nll)), "top1": float(np.mean(hit)),
                     "by_act": {f"act{a + 1}": {"n": int((act == a).sum()), "nll": float(np.mean(nll[act == a])) if (act == a).any() else None,
                                                "top1": float(np.mean(hit[act == a])) if (act == a).any() else None} for a in (0, 1, 2)}}

    for kind in ("card", "ancient", "rest", "shop", "route"):
        idx = ds.kind_idx[kind]
        idx = idx[valid[idx]]
        if not len(idx):
            continue
        m = ds.matrix(idx)
        ok = m >= 0
        s = np.where(ok, scores[np.maximum(m, 0)], -np.inf)
        act = dec[idx, C["act"]]
        chosen = dec[idx, C["chosen"]]
        lse = _lse(s)
        top = s.argmax(1)
        if kind in ("card", "ancient", "route"):
            nll = lse - s[np.arange(len(idx)), chosen]
            hit = (top == chosen).astype(np.float64)
            put(kind, nll, hit, act)
            if kind == "route":
                rl = np.where(ok, ds.role[np.maximum(m, 0)], -1)
                differs = (np.where(ok, rl, 99).min(1) != rl.max(1))  # the rooms are not all of one type
                put("route_type_choices", nll[differs], hit[differs], act[differs])
                out["route_coin_top1"] = float(np.mean(1.0 / dec[idx, C["n"]]))
            if kind == "card":
                skipped = dec[idx, C["aux"]] != 0
                put("card_take_screens", nll[~skipped], hit[~skipped], act[~skipped])
                put("card_skip_screens", nll[skipped], hit[skipped], act[skipped])
                whole = (dec[idx, C["flags"]] & F_SPLIT) == 0
                put("card_unsplit_screens", nll[whole], hit[whole], act[whole])
                pred_skip = top == (dec[idx, C["n"]] - 1)
                out["card_skip_rate"] = {"predicted": float(pred_skip.mean()), "human": float(skipped.mean())}
                n_opt = dec[idx, C["n"]]
                out["card_coin_top1"] = float(np.mean(1.0 / n_opt))
        elif kind == "rest":
            role = np.where(ok, ds.role[np.maximum(m, 0)], -1)
            aux = dec[idx, C["aux"]]
            sk = np.where(role == aux[:, None], s, -np.inf)
            nll_kind = lse - _lse(sk)
            top_role = np.take_along_axis(role, top[:, None], 1)[:, 0]
            put("rest_kind", nll_kind, (top_role == aux).astype(np.float64), act)
            smith = (aux == ROLE_SMITH) & (chosen >= 0)
            if smith.any():
                ss = np.where(role == ROLE_SMITH, s, -np.inf)[smith]
                ch = chosen[smith]
                nll_t = _lse(ss) - ss[np.arange(len(ch)), ch]
                put("rest_smith_target", nll_t, (ss.argmax(1) == ch).astype(np.float64), act[smith])
                top3 = (np.argsort(-ss, axis=1)[:, :3] == ch[:, None]).any(1).astype(np.float64)
                put("rest_smith_target_top3", nll_t, top3, act[smith])
        else:  # shop: row 0 leave, row 1 buy
            z = s[:, 1] - s[:, 0]
            lab = chosen > 0
            nll = np.logaddexp(0.0, np.where(lab, -z, z))
            put("shop", nll, ((z > 0) == lab).astype(np.float64), act)
    return out


def score_total(m: dict) -> float:
    """The early-stopping score: the sum of the mean validation NLL of the kinds the fit trains on."""
    return float(sum(m[k]["nll"] for k in ("card", "ancient", "rest_kind", "rest_smith_target", "shop", "route") if k in m))


# ------------------------------------------------------------------------------------------------------------------- the net


def make_net(hidden: int = 256, dropout: float = 0.1, win: bool = False):
    """The run judgement's trunk (RunValueNet's embeddings and two layers) with one logit in place of its floor and end-HP heads.

    With `win` the net takes one more input per row, "this run won" (1 or 0): a column of weights added to the first layer's
    pre-activation (`won_w`), which starts at zero so the net starts as the plain one. At play the input is 1."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    base = stack().M.RunValueNet

    class PriorNet(base):  # type: ignore[valid-type,misc]
        uses_win = bool(win)

        def __init__(self) -> None:
            super().__init__(hidden, dropout)
            del self.floor_head, self.end_hp_head
            self.score = nn.Linear(hidden, 1)
            if win:
                self.won_w = nn.Parameter(torch.zeros(hidden))

        def forward(self, dense, counts, won=None):  # noqa: ANN001
            joined = torch.cat([dense, self.embed(counts)], dim=1)
            pre = self.hidden1(joined)
            if win:
                if won is None:
                    raise ValueError("this prior takes the won input")
                pre = pre + won[:, None] * self.won_w[None, :]
            h = self.dropout(torch.relu(pre))
            h = torch.relu(self.hidden2(h))
            return self.score(h).squeeze(1)

    PriorNet.__qualname__ = PriorNet.__name__
    return PriorNet()


def init_from_judge(net, npz: Path) -> None:
    """Copy the judge's trunk (`value/` embeddings and hidden layers) into `net`; the score head (and the won input) stay at their fresh initialisation."""
    import torch  # noqa: PLC0415

    with np.load(npz, allow_pickle=False) as z:
        state = {k.removeprefix("value/"): torch.from_numpy(z[k]) for k in z.files if k.startswith("value/")}
    state = {k: v for k, v in state.items() if not k.startswith(("floor_head", "end_hp_head"))}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if unexpected or set(missing) - {"won_w"} != {"score.weight", "score.bias"}:
        raise ValueError(f"the judge's trunk does not fit: missing {missing}, unexpected {unexpected}")


def init_from_prior(net, npz: Path) -> None:
    """Start from a fitted prior's exported weights (`export_net`'s `weights.npz`: the layers by their torch names): every tensor of `net`."""
    import torch  # noqa: PLC0415

    with np.load(npz, allow_pickle=False) as z:
        state = {k: torch.from_numpy(z[k]) for k in z.files if k != "meta"}
    state.pop("extra.won.weight", None)
    net.load_state_dict(state, strict=True)


class NumpyPrior:
    """The prior in numpy (the exported weights.npz): one logit per row of the judge's dense and counts inputs.

    A prior with an extra input (v1's win variant) lists it in `meta["extra_inputs"]`: each entry names a weight vector (`weight_key`,
    one number per unit of the first layer) that is added, times the input, to the first layer's pre-activation, and the input value
    the stored `hidden1.bias` already includes (`folded_at`). The call's third argument is the input per row; left out, the rows
    are scored at `folded_at`, the play setting, so a loader that knows nothing of extra inputs scores them right too."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as z:
            self.p = {k: z[k] for k in z.files if k != "meta"}
            self.meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        self.extra_inputs = list(self.meta.get("extra_inputs") or [])

    def __call__(self, dense: np.ndarray, counts: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
        p = self.p
        e = np.concatenate([counts[:, :1154] @ p["embed.deck.weight"].T, counts[:, 1154:1450] @ p["embed.relics.weight"].T,
                            counts[:, 1450:] @ p["embed.potions.weight"].T], axis=1)
        pre = np.concatenate([dense, e], axis=1) @ p["hidden1.weight"].T + p["hidden1.bias"]
        if extra is not None and self.extra_inputs:
            for spec in self.extra_inputs:
                pre = pre + (np.asarray(extra, dtype=np.float32) - np.float32(spec["folded_at"]))[:, None] * p[spec["weight_key"]][None, :]
        h = np.maximum(pre, 0.0)
        h = np.maximum(h @ p["hidden2.weight"].T + p["hidden2.bias"], 0.0)
        return (h @ p["score.weight"].T + p["score.bias"])[:, 0]


WON_INPUT = {"name": "won", "weight_key": "extra.won.weight", "added_to": "hidden1 pre-activation (times the input)", "value_at_play": 1.0,
             "folded_at": 1.0, "values_in_training": "1 for a decision of a run that won, 0 for one of a run that did not"}


def export_net(net, path: Path, meta: dict) -> None:
    """Write the net as `weights.npz` (the layers by their torch names, and `meta` as JSON). A net with the won input stores its weight
    vector as `extra.won.weight` and folds it into `hidden1.bias` at the play setting (input 1), so a loader that ignores `extra_inputs`
    scores at play; `meta["extra_inputs"]` documents the one addition."""
    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    if "won_w" in arrays:
        w = arrays.pop("won_w")
        arrays["hidden1.bias"] = (arrays["hidden1.bias"] + w * np.float32(WON_INPUT["folded_at"])).astype(np.float32)
        arrays[WON_INPUT["weight_key"]] = w
        meta = dict(meta, extra_inputs=[WON_INPUT])
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    os.replace(tmp, path)



# ------------------------------------------------------------------------------------------------------------------- the fit


class Gpu:
    """The rows, decisions and (v0) donors on the card, and the loss of a batch of decisions of one kind. A v1 dataset carries its own
    map block with every row, so no donor is drawn; a net with the won input gets the decision's run outcome (or a fixed value)."""

    def __init__(self, ds: Dataset, donors: dict | None, dev, winner_weight: float):
        import torch  # noqa: PLC0415

        self.t = torch
        self.ds, self.dev = ds, dev
        self.has_map = ds.has_map
        self.dense = torch.from_numpy(ds.dense).to(dev)
        self.counts = torch.from_numpy(ds.counts).to(dev)
        if not self.has_map:
            self.dc = torch.from_numpy(donors["c"]).to(dev)
            self.da = torch.from_numpy(donors["a15"]).to(dev)
            self.dn = torch.from_numpy(donors["n"].astype(np.int64)).to(dev)
        self.dkey = torch.from_numpy(ds.dkey).to(dev)
        dec = ds.dec
        self.won = torch.from_numpy(dec[:, C["won"]].astype(np.float32)).to(dev)
        self.set_winner_weight(winner_weight)
        self.chosen = torch.from_numpy(dec[:, C["chosen"]].astype(np.int64)).to(dev)
        self.aux = torch.from_numpy(dec[:, C["aux"]].astype(np.int64)).to(dev)
        self.role = torch.from_numpy(ds.role.astype(np.int64)).to(dev)
        self.start = torch.from_numpy(dec[:, C["start"]].astype(np.int64)).to(dev)
        self.n = torch.from_numpy(dec[:, C["n"]].astype(np.int64)).to(dev)
        self.caps = torch.from_numpy(HP_CAPS).to(dev)
        self.mats: dict[str, object] = {}

    def set_winner_weight(self, winner_weight: float) -> None:
        self.weight = 1.0 + (winner_weight - 1.0) * self.won

    def matrix(self, didx):
        """Padded (B x K) row indices of the decisions `didx` (a tensor), -1 for pads."""
        t = self.t
        n = self.n[didx]
        k = int(n.max().item())
        j = t.arange(k, device=self.dev)[None, :]
        m = self.start[didx][:, None] + j
        return t.where(j < n[:, None], m, t.full_like(m, -1))

    def logits(self, net, didx, fixed: bool, win: float | None = None):
        """The logit of every option of the decisions `didx`. `fixed`: the same donor for a decision every time (v0). `win`: the value of
        the won input for every row (a net that takes it); None gives each decision its run's outcome."""
        t = self.t
        m = self.matrix(didx)
        valid = m >= 0
        owner = t.arange(len(didx), device=self.dev)[:, None].expand_as(m)[valid]
        rows = m[valid]
        d = self.dense[rows]
        if self.has_map:
            x = t.cat([d[:, 0:22], d[:, 22:43], t.zeros(len(rows), 237, device=self.dev), d[:, 43:68], d[:, 68:93], d[:, 93:105], d[:, 105:113]], dim=1)
        else:
            key = self.dkey[didx]
            n = t.where(key >= 0, self.dn[key.clamp(min=0)], t.zeros_like(key))
            if fixed:
                pick = ((didx * 1103515245 + 12345) % 2147483648) % n.clamp(min=1)
            else:
                pick = (t.rand(len(didx), device=self.dev) * n.clamp(min=1)).long().clamp(max=(n - 1).clamp(min=0))
            kk = key.clamp(min=0)
            has = (key >= 0).float()
            a15 = self.da[kk, pick] * has
            c = self.dc[kk, pick] * has[:, None]
            # B: the stored first 21 slots, then zeros; assemble in place
            x = t.cat([d[:, 0:22], d[:, 22:43], t.zeros(len(rows), 237, device=self.dev), c[owner], d[:, 43:68], d[:, 68:80], d[:, 80:88]], dim=1)
            x[:, 15] = a15[owner]
        x[:, 0:4] = t.minimum(x[:, 0:4], self.caps)
        cnt = self.counts[rows].float()
        if getattr(net, "uses_win", False):
            flag = self.won[didx][owner] if win is None else t.full((len(rows),), float(win), device=self.dev)
            z = net(x, cnt, flag)
        else:
            z = net(x, cnt)
        out = t.full(m.shape, -1e9, device=self.dev)
        out[valid] = z
        return out, m, valid

    def loss(self, net, kind: str, didx):
        t = self.t
        z, m, valid = self.logits(net, didx, fixed=False)
        w = self.weight[didx]
        ch = self.chosen[didx]
        if kind in ("card", "ancient", "route"):
            nll = t.logsumexp(z, 1) - z.gather(1, ch[:, None])[:, 0]
            return (nll * w).sum() / w.sum()
        if kind == "rest":
            role = t.where(valid, self.role[m.clamp(min=0)], t.full_like(m, -1))
            aux = self.aux[didx]
            lse = t.logsumexp(z, 1)
            same = role == aux[:, None]
            nll_kind = lse - t.logsumexp(z.masked_fill(~same, -1e9), 1)
            loss = (nll_kind * w).sum() / w.sum()
            has = (aux == ROLE_SMITH) & (ch >= 0)
            if has.any():
                zs = z.masked_fill(role != ROLE_SMITH, -1e9)
                nll_t = t.logsumexp(zs, 1) - z.gather(1, ch.clamp(min=0)[:, None])[:, 0]
                loss = loss + (nll_t * w * has).sum() / (w * has).sum().clamp(min=1e-6)
            return loss
        zz = z[:, 1] - z[:, 0]
        lab = ch.float()
        bce = t.nn.functional.softplus(-zz) * lab + t.nn.functional.softplus(zz) * (1 - lab)
        return (bce * w).sum() / w.sum()

    def scores(self, net, chunk_dec: int = 20000, win: float | None = None) -> np.ndarray:
        """Every row's logit with the fixed donors (decision-contiguous rows, like the dataset's)."""
        t = self.t
        out = np.zeros(len(self.ds.dense), dtype=np.float32)
        n_dec = len(self.ds.dec)
        net.eval()
        with t.no_grad():
            for s in range(0, n_dec, chunk_dec):
                didx = t.arange(s, min(s + chunk_dec, n_dec), device=self.dev)
                z, m, valid = self.logits(net, didx, fixed=True, win=win)
                out[m[valid].cpu().numpy()] = z[valid].float().cpu().numpy()
        return out


def fit_one(gt: Gpu, gv: Gpu, init: str, winner_weight: float, a: argparse.Namespace, kinds: list[str], judge: Path, win: bool = False) -> dict:
    """One variant: train with early stopping on the validation fifth. Returns its state, history and best metrics. `win` gives the net
    the won input (validated with each decision's own run outcome as the input)."""
    import torch  # noqa: PLC0415

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    net = make_net(win=win).to(dev)
    if init == "gen11":
        init_from_judge(net, judge)
        net.to(dev)
    elif init.startswith("prior:"):  # a warm start from a fitted prior's weights (`strong_sets.py fit --init-from`)
        init_from_prior(net, Path(init[len("prior:"):]))
        net.to(dev)
    lr = 3e-4 if init == "gen11" or init.startswith("prior:") else 1e-3
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    ds_train, ds_val = gt.ds, gv.ds
    gt.set_winner_weight(winner_weight)
    verified_only = not getattr(a, "with_shops", False)
    valid_val = ~_bool_unavailable(ds_val, "shop" in kinds, True, verified_only)
    valid_tr = ~_bool_unavailable(ds_train, "shop" in kinds, True, verified_only)
    pools = {k: ds_train.kind_idx[k][valid_tr[ds_train.kind_idx[k]]] for k in kinds}
    history, best, best_state, since = [], None, None, 0
    t0 = time.time()
    for epoch in range(a.epochs):
        net.train()
        batches = []
        for k in kinds:
            perm = rng.permutation(pools[k])
            batches += [(k, perm[i:i + a.batch]) for i in range(0, len(perm), a.batch)]
        order = rng.permutation(len(batches))
        total, nb = 0.0, 0
        for bi in order:
            k, idx = batches[bi]
            loss = gt.loss(net, k, torch.from_numpy(idx).to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            nb += 1
        m = metrics_from_scores(ds_val, gv.scores(net), valid_val)
        sc = score_total(m)
        history.append({"epoch": epoch + 1, "train_loss": total / max(nb, 1), "val_score": sc, "val_card_nll": m["card"]["nll"],
                        "val_card_top1": m["card"]["top1"], "val_rest_kind_top1": m.get("rest_kind", {}).get("top1"),
                        "val_ancient_top1": m.get("ancient", {}).get("top1"), "val_route_top1": m.get("route", {}).get("top1"),
                        "val_shop_nll": m.get("shop", {}).get("nll"), "seconds": round(time.time() - t0, 1)})
        print(f"  [{init} x{winner_weight:g}{' win' if win else ''}] epoch {epoch + 1}: train {total / max(nb, 1):.4f} val score {sc:.4f} "
              f"card top1 {m['card']['top1']:.4f} route top1 {m.get('route', {}).get('top1', float('nan')):.4f} ({time.time() - t0:.0f} s)", flush=True)
        if best is None or sc < best["score"] - 1e-4:
            best = {"score": sc, "epoch": epoch + 1, "metrics": m}
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= a.patience:
                break
    net.load_state_dict(best_state)
    return {"net": net, "init": init, "winner_weight": winner_weight, "win": win, "best": best, "history": history,
            "epochs_run": len(history), "seconds": round(time.time() - t0, 1)}


def read_counts_file(dec_dir: Path) -> dict:
    p = Path(dec_dir) / "counts.json"
    if not p.is_file():
        raise SystemExit(f"human_prior: {p} is missing: run `build` first")
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_fit(a: argparse.Namespace) -> int:
    import torch  # noqa: PLC0415

    t0 = time.time()
    out_root = Path(a.out_dir)
    dec_dir = out_root / (a.decisions or "decisions")
    counts = read_counts_file(dec_dir)
    donors = load_donors(out_root / "donors.npz") if (out_root / "donors.npz").is_file() else None
    ds_tr = Dataset.load(dec_dir, TRAIN_FIFTHS, donors)
    ds_va = Dataset.load(dec_dir, [VAL_FIFTH], donors)
    if donors is None and not ds_tr.has_map:
        raise SystemExit("human_prior fit: a v0 dataset needs donors.npz")
    shops_ok = bool(a.with_shops or ds_tr.has_map or counts.get("shop_price_check", {}).get("verified") is True)
    kinds = ["card", "ancient", "rest"] + (["shop"] if shops_ok else []) + (["route"] if len(ds_tr.kind_idx["route"]) else [])
    print(f"human_prior fit: {dec_dir.name}, {'true' if ds_tr.has_map else 'donor'} maps, kinds {kinds}", flush=True)
    import human_starts as hs  # noqa: PLC0415

    exam = hs.read_exam_ids(EXAM_IDS)
    hs.assert_no_exam(ds_tr.run_ids, exam)
    hs.assert_no_exam(ds_va.run_ids, exam)
    print(f"human_prior fit: train {len(ds_tr.dec):,} decisions {len(ds_tr.dense):,} rows; validation {len(ds_va.dec):,} decisions "
          f"({time.time() - t0:.0f} s); donor keys {dict(ds_tr.coverage)}", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt, gv = Gpu(ds_tr, donors, dev, 1.0), Gpu(ds_va, donors, dev, 1.0)
    results = []
    for init in a.variants.split(","):
        for ww in [float(x) for x in a.winner_weights.split(",")]:
            results.append(fit_one(gt, gv, init, ww, a, kinds, Path(a.judge), win=bool(a.win_input)))
    best = min(results, key=lambda r: r["best"]["score"])
    out = out_root / a.name
    out.mkdir(parents=True, exist_ok=True)
    verified_only = not a.with_shops
    base_meta = {"kinds": kinds, "feature_version": stack().F.FEATURE_VERSION, "map": "true" if ds_tr.has_map else "donor", "source": SPIRE_CODEX}
    variants = []
    for r in results:
        name = f"{r['init']}-w{r['winner_weight']:g}" + ("-win" if r["win"] else "")
        export_net(r["net"], out / f"weights-{name}.npz", dict(base_meta, variant=name))
        variants.append({"name": name, "init": r["init"], "winner_weight": r["winner_weight"], "win_input": r["win"], "best_epoch": r["best"]["epoch"],
                         "epochs_run": r["epochs_run"], "val_score": r["best"]["score"], "val_metrics": r["best"]["metrics"],
                         "history": r["history"], "seconds": r["seconds"]})
    chosen = f"{best['init']}-w{best['winner_weight']:g}" + ("-win" if best["win"] else "")
    export_net(best["net"], out / "weights.npz", dict(base_meta, variant=chosen, judge=str(a.judge), inputs="judge dense 350 + counts 1513; one logit per option"))
    valid_tr = ~_bool_unavailable(ds_tr, shops_ok, True, verified_only)
    valid_va = ~_bool_unavailable(ds_va, shops_ok, True, verified_only)
    trained = {k: (int(valid_tr[ds_tr.kind_idx[k]].sum()) if k in kinds else 0) for k in KIND}
    present = {k: int(len(ds_tr.kind_idx[k])) for k in KIND}
    mismatch = int(((ds_tr.dec[:, C["flags"]] & F_REPLAY_MISMATCH) != 0).sum())
    fit_doc = {"written_by": "rl/human_prior.py fit (aggregates only)", "source": SPIRE_CODEX, "decisions_folder": dec_dir.name, "chosen_variant": chosen,
               "chosen_reason": "lowest validation score (sum of the mean NLL of the kinds fitted) on fifth 4", "map": base_meta["map"],
               "win_input": bool(a.win_input), "kinds_fitted": kinds, "shops_in_fit": shops_ok,
               "shops_in_fit_rule": "verified rows only" if (shops_ok and verified_only) else ("all rows" if shops_ok else "none"),
               "train_fifths": list(TRAIN_FIFTHS), "validation_fifth": VAL_FIFTH,
               "trained_on_decisions": trained, "in_training_fifths": present,
               "validation_decisions": {k: (int(valid_va[ds_va.kind_idx[k]].sum()) if k in kinds else 0) for k in KIND},
               "left_out_replay_mismatch_decisions_in_training_fifths": mismatch,
               "left_out_note": "unavailable shop offers and runs whose deck replay disagrees with the final deck are always left out; shop rows are in the fit "
                                "only when their prices are verified", "donor_keys": dict(ds_tr.coverage), "variants": variants,
               "validation_top1_chosen": {k: {"top1": v["top1"], "n": v["n"], "by_act": {a_: x["top1"] for a_, x in v["by_act"].items()}}
                                          for k, v in best["best"]["metrics"].items() if isinstance(v, dict) and "top1" in v},
               "wall_seconds": round(time.time() - t0, 1), "torch": torch.__version__}
    (out / "fit.json").write_text(json.dumps(fit_doc, indent=1), encoding="utf-8")
    print(json.dumps({"chosen": chosen, "validation_top1": fit_doc["validation_top1_chosen"], "wall_seconds": fit_doc["wall_seconds"]}, indent=1))
    return 0


def route_baselines(ds: Dataset, valid: np.ndarray, train: Dataset | None = None) -> dict:
    """What a route choice is worth without a model, on the route decisions of `ds` in `valid`: a coin over the rooms; the room whose
    column is nearest the boss's (every room of a step is one row nearer the boss, so the column is all that tells them apart; ties
    share the credit); and, with the training decisions, the room type the humans took most often per act (the rate of being taken
    when offered, in the same act), with the credit shared between equal rates. Top-1 and (where there is a probability) the mean
    negative log-likelihood, over all route decisions, those where the rooms differ in type, and per act."""
    idx = ds.kind_idx["route"]
    idx = idx[valid[idx]]
    if not len(idx):
        return {}
    dec = ds.dec
    m = ds.matrix(idx)
    ok = m >= 0
    rows = np.maximum(m, 0)
    col = np.where(ok, ds.ident[rows], 0)
    role = np.where(ok, ds.role[rows], -1)
    n = dec[idx, C["n"]]
    ch = dec[idx, C["chosen"]]
    boss_col = dec[idx, C["aux"]]
    act = dec[idx, C["act"]]
    ar = np.arange(len(idx))
    differs = np.where(ok, role, 99).min(1) != role.max(1)
    uniform = (1.0 / n, np.log(n))
    dist = np.where(ok, np.abs(col - boss_col[:, None]), 99)
    best = dist.min(1)
    ties = (dist == best[:, None]).sum(1)
    near = (np.where(dist[ar, ch] == best, 1.0 / ties, 0.0), None)
    models: dict[str, tuple] = {"uniform": uniform, "nearest_to_boss_column": near}
    if train is not None and len(train.kind_idx["route"]):
        tidx = train.kind_idx["route"]
        tidx = tidx[~_bool_unavailable(train, False, True)[tidx]]
        tm = train.matrix(tidx)
        tok = tm >= 0
        trole = np.where(tok, train.role[np.maximum(tm, 0)], 0)
        tact = train.dec[tidx, C["act"]]
        tch = train.dec[tidx, C["chosen"]]
        offered = np.zeros((3, 9))
        taken = np.zeros((3, 9))
        for j in range(tm.shape[1]):
            sel = tok[:, j]
            np.add.at(offered, (tact[sel], trole[sel, j]), 1)
        np.add.at(taken, (tact, trole[np.arange(len(tidx)), tch]), 1)
        rate = (taken + 0.5) / (offered + 1.0)
        score = np.where(ok, rate[act[:, None], np.maximum(role, 0)], 0.0)
        top = score.max(1)
        tie = (np.isclose(score, top[:, None]) & ok).sum(1)
        p_hit = np.where(np.isclose(score[ar, ch], top), 1.0 / tie, 0.0)
        prob = score / score.sum(1, keepdims=True)
        models["type_frequency_by_act"] = (p_hit, -np.log(prob[ar, ch]))

    def cell(mask: np.ndarray, hit: np.ndarray, nll: np.ndarray | None) -> dict | None:
        if int(mask.sum()) < MIN_CELL:
            return None
        return {"n": int(mask.sum()), "top1": float(hit[mask].mean()), "nll": float(nll[mask].mean()) if nll is not None else None}

    out: dict = {"decisions": int(len(idx)), "decisions_with_rooms_of_different_types": int(differs.sum()),
                 "nearest_to_boss_column_means": "the room whose column is nearest the boss node's; ties share the credit"}
    for name, (hit, nll) in models.items():
        out[name] = {"all": cell(np.ones(len(idx), bool), hit, nll), "rooms_differ_in_type": cell(differs, hit, nll),
                     "by_act": {f"act{k + 1}": cell(act == k, hit, nll) for k in (0, 1, 2)}}
    return out


def cmd_agreement(a: argparse.Namespace) -> int:
    """The prior's (and optionally the judgement's) top-1 agreement per kind and act on the validation fifth (never the exam fifth). A
    v1 dataset adds the route kind (with its baselines) and the decisions of winning runs apart; a prior with the won input is
    scored at its play setting (input 1), with each run's own outcome, and on the winning runs' decisions."""
    import human_starts as hs  # noqa: PLC0415

    out_root = Path(a.out_dir)
    dec_dir = out_root / (a.decisions or "decisions")
    donors = load_donors(out_root / "donors.npz") if (out_root / "donors.npz").is_file() else None
    counts = read_counts_file(dec_dir)
    ds = Dataset.load(dec_dir, [VAL_FIFTH], donors)
    hs.assert_no_exam(ds.run_ids, hs.read_exam_ids(EXAM_IDS))
    doc: dict = {"written_by": "rl/human_prior.py agreement (aggregates only)", "source": SPIRE_CODEX, "fifth": VAL_FIFTH, "decisions_folder": dec_dir.name,
                 "true_map": ds.has_map, "decisions": {k: int(len(ds.kind_idx[k])) for k in KIND}}
    won_dec = ds.dec[:, C["won"]] == 1
    won_row = np.repeat(won_dec, ds.dec[:, C["n"]])
    shops_ok = False
    verified_only = not a.with_shops
    valid = ~_bool_unavailable(ds, shops_ok, True, verified_only)
    if a.model:
        model = Path(a.model)
        weights = model / "weights.npz" if model.is_dir() else model
        prior = NumpyPrior(weights)
        shops_ok = "shop" in (prior.meta.get("kinds") or [])
        valid = ~_bool_unavailable(ds, shops_ok, True, verified_only)
        doc.update(model=str(weights.name), variant=prior.meta.get("variant"), map_in_training=prior.meta.get("map", "donor"))
        play = np.ones(len(ds.dense), dtype=np.float32) if prior.extra_inputs else None
        scores = row_scores_np(ds, donors, prior, extra=play)
        doc["prior"] = metrics_from_scores(ds, scores, valid)
        if prior.extra_inputs:
            doc["won_input"] = "prior = every row scored at the play setting (the input is 1); prior_input_as_recorded = each decision's own run outcome"
            doc["prior_input_as_recorded"] = metrics_from_scores(ds, row_scores_np(ds, donors, prior, extra=won_row.astype(np.float32)), valid)
        v_win, v_other = valid & won_dec, valid & ~won_dec
        doc["prior_winning_runs"] = metrics_from_scores(ds, scores, v_win)
        doc["prior_other_runs"] = metrics_from_scores(ds, scores, v_other)
    valid = ~_bool_unavailable(ds, shops_ok, True, verified_only)
    if len(ds.kind_idx["route"]):
        train = Dataset.load(dec_dir, TRAIN_FIFTHS, donors, kinds=["route"]) if not a.no_train_baseline else None
        if train is not None:
            hs.assert_no_exam(train.run_ids, hs.read_exam_ids(EXAM_IDS))
        doc["route_baselines"] = route_baselines(ds, valid, train)
    if a.baseline:
        s = stack()
        nets = s.M.load_numpy(a.baseline)

        def v_fn(dense, cnt):
            logits, _ = nets.value.forward(dense, cnt)
            return s.M.expected_floors(s.M.softmax(logits), W_WON).astype(np.float32)

        doc["baseline_judge"] = {"file": Path(a.baseline).name, "choice": "argmax of expected final floors (W=53) over the same options",
                                 "metrics": metrics_from_scores(ds, row_scores_np(ds, donors, v_fn), valid)}
    shop_idx = ds.kind_idx["shop"]
    shop_idx = shop_idx[valid[shop_idx]] if shops_ok else shop_idx[:0]
    if len(shop_idx):
        bought = float(np.mean(ds.dec[shop_idx, C["chosen"]] > 0))
        doc["shop_majority_rate"] = {"n": int(len(shop_idx)), "bought": bought, "top1_of_always_leaving": 1.0 - bought}
    doc["price_check_verified"] = counts.get("shop_price_check", {}).get("verified")
    text = json.dumps(doc, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0



def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("donors")
    d.add_argument("--runs-root", default=str(RUNS_ROOT))
    d.add_argument("--gens", default="11,9")
    d.add_argument("--per-key", type=int, default=64)
    d.add_argument("--max-runs", type=int, default=8000)
    d.add_argument("--out-dir", default=str(OUT_DIR))
    b = sub.add_parser("build")
    b.add_argument("--corpus", default=str(CORPUS))
    b.add_argument("--exam-ids", default=str(EXAM_IDS))
    b.add_argument("--out-dir", default=str(OUT_DIR))
    b.add_argument("--judge", default=str(GEN11), help="the run judgement whose FightNet fills the fight block H")
    b.add_argument("--reference", default=str(REF_COUNTS))
    b.add_argument("--procs", type=int, default=4)
    b.add_argument("--chunk-runs", type=int, default=40)
    b.add_argument("--ascensions", default="10", help="the Ascension levels read: 10 (the default), a list 11,12 or a range 11-20 (the strong sets, docs/rl/HUMAN_PRIOR.md)")
    b.add_argument("--limit-runs", type=int, default=0)
    b.add_argument("--lib", default="", help="a native build to load (STS2_LIB_PATH); by default none is loaded")
    b.add_argument("--clean", action="store_true")
    b.add_argument("--events", action="store_true", help="also build event rows (kind 4) from the option table; the fit does not train them yet")
    b.add_argument("--truth", default="", help="v1: the folder of truth-NNNNN.json.gz chunks (rl/human_prior_replay.py); the true map, the "
                   "route decisions and the replay's shop prices replace the donors")
    b.add_argument("--decisions", default="", help="the shard folder under --out-dir (default decisions, or decisions-v1 with --truth)")
    f = sub.add_parser("fit")
    f.add_argument("--out-dir", default=str(OUT_DIR))
    f.add_argument("--judge", default=str(GEN11))
    f.add_argument("--variants", default="gen11,scratch")
    f.add_argument("--winner-weights", default="1,2")
    f.add_argument("--epochs", type=int, default=30)
    f.add_argument("--patience", type=int, default=3)
    f.add_argument("--batch", type=int, default=256)
    f.add_argument("--seed", type=int, default=20261105)
    f.add_argument("--name", default="pi_h-v0", help="the folder under --out-dir the prior is written to")
    f.add_argument("--with-shops", action="store_true", help="fit on shop rows even when their prices are unverified (a v0 dataset's all are)")
    f.add_argument("--decisions", default="", help="the shard folder under --out-dir (default decisions; v1: decisions-v1)")
    f.add_argument("--win-input", action="store_true", help="give the net the extra input 'this run won' (1 for the rows of winning runs)")
    g = sub.add_parser("agreement")
    g.add_argument("--model", default="", help="a fitted prior folder (weights.npz) or file; optional when --baseline is given")
    g.add_argument("--out-dir", default=str(OUT_DIR))
    g.add_argument("--baseline", default="", help="a run_value.npz whose expected floors choose, scored on the same decisions")
    g.add_argument("--out", default="")
    g.add_argument("--decisions", default="", help="the shard folder under --out-dir (default decisions; v1: decisions-v1)")
    g.add_argument("--with-shops", action="store_true", help="score the shop rows whose prices are not verified too")
    g.add_argument("--no-train-baseline", action="store_true", help="skip the type-frequency route baseline (it reads the training fifths)")
    a = ap.parse_args(argv)
    if a.cmd == "donors":
        return cmd_donors(a)
    if a.cmd == "build":
        return cmd_build(a)
    return cmd_fit(a) if a.cmd == "fit" else cmd_agreement(a)


if __name__ == "__main__":
    raise SystemExit(main())
