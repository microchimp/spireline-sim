"""The run currency: a fight's end state priced by the frozen run head, in the search's units (training design section 3, brief 3a).

Today a fight is searched in HP: a win is worth its end HP plus 10 for each potion still held, a death -100. A potion is then worth 10 HP
in a hallway and at a boss alike, and the AI drinks in 60% of the Act 1 hallway fights it wins holding one (humans 23%) and reaches the
Act 1 boss with 0.8 potions (humans 1.6). Here the terminal value of a fight is what the run head (`run_train.models`, generation 11's
V_run, frozen) says the run is worth from the post-fight state:

    T         the head's expected final floor of the post-fight state, when the fight is won
    T_lost    the fight's own floor, when it is lost (the run ends there, exactly)
    T_full    the head's value with full HP and every starting potion held
    u         (T - T_lost) / (T_full - T_lost), clipped to [0, U_MAX]: 0 is a death, 1 a full-HP win holding everything
    v         DEATH_VALUE + u (max HP at the start - DEATH_VALUE): the search's units

Within one fight v is an affine function of the expected final floor, so the search maximizes the expected final floor; the spread from a
death to a full-HP win is today's (-100 to max HP), so the exploration constant, the confirmation's arithmetic and the board's statistics
keep their meaning. The post-fight state is the fight start's run view with HP, max HP and the potions still held (by identity) replaced by
their end values, and the end-of-combat heal of Burning Blood (6) or Black Blood (12) added to the HP of a win (the run layer applies it,
the combat does not); deck, relics and gold are as at the start (what a fight changes in them is not carried).

Where the view comes from. A run in progress has the exact view (`gen0.play_emu_run`: `RunSession.view()`), and a start record from a
replay carries it (`run_view`, `view_to_json`). A start record without one (the board's, the exam's, a human start) has a snapshot document
of the fight: HP, max HP, gold, relics, potions and the whole deck are in it, the map is not. Its view is built from the document
(`contexts_of_start`) with the map-ahead block and the rows to the boss taken from the mean over the copy's Act 1 maps by rows to the
boss and node type (`MapTable`, data/run_currency_map_mean.json); an act's boss the document does not name is every boss of the act's
region, the value the mean over them. The map is the same before and after a fight, so its error moves T_lost-to-T_full together and
touches only the head's second-order terms.

A judge leaf (`judge_client.estimate`) has the end HP and a potion count, not their identity: the count form averages the head over the
subsets of the starting potions of that size, and a fractional count interpolates between its two neighbours (Jensen's error of composing a
mean through V is the known cost; a judge trained on u removes it, brief 3b).

The head can be wrong: `audit` is the table of its marginals (10 more HP at 30, 60 and 90% HP, each potion kind, 50 gold, by act and
floor band, and the HP-to-potion ratio against today's 10 HP) and its stop rule (design 3.4): a potion worth under 3 or over 40 HP in an
Act 1 hallway fight, or V not increasing in HP in some act and floor band. A fight whose T_full - T_lost is under `MIN_SPAN` floors has no
usable scale and is priced as today (`FightCurrency.fallback`).

This module is numpy-only: it loads the head through `run_train.models.load_numpy` and never the emulator library or torch, so the search's
hot path is a dictionary lookup (a miss is one batched forward of the head's two nets over the contexts).

    python rl/run_currency.py audit  --starts FILE [--per-cell 60] [--out DIR]
    python rl/run_currency.py map-table [--seeds 300] [--out data/run_currency_map_mean.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_train import features, models, vocab  # noqa: E402
from run_train.features import CardRef, MapNode, RunView  # noqa: E402

DEATH_VALUE = -100.0  # live_search.DEATH_VALUE (a test holds the two equal)
POTION_VALUE = 10.0  # today's potion, in HP: live_search's and L0's
W_WON = 53.0  # the value of a won run, in floors (vocab.WON_FLOOR; Agent.W)
FROZEN_HEAD_SHA256 = "f0a199fb82c2e4071973fee89fd433cf9dd27cfa6c1cacb70f83184293cb38ee"  # generation 11's V_run (D44); 12 and 13 carry it
HEAD_ENV = "SPIRELINE_RUN_CURRENCY_HEAD"  # a head to price with instead of the frozen one (its sha is recorded, not pinned)
REFEREE_ENV = "FIGHT_GYM_REFEREE"
MIN_SPAN = 0.25  # floors between a lost fight and a full-HP win under which the head gives the fight no scale
U_MAX = 1.5  # a win can be worth more than the start's full state (max HP gained in the fight); never more than this
END_HEALS = {"BurningBlood": 6, "BlackBlood": 12}  # the run layer's heal after a won combat (RunRewardGenerator), not in the combat sim
BOSS_FLOORS = (17, 33, 48)  # run_fields.ACT_BOSS_FLOORS
NODE_MONSTER, NODE_ELITE, NODE_BOSS = 1, 2, 6  # run_train.vocab
AUDIT_POTION_HP = (3.0, 40.0)  # design 3.4: a potion outside this many HP is mispriced (Act 1 hallway fights)
AUDIT_HP_FRACTIONS = (0.3, 0.6, 0.9)
AUDIT_GRID = tuple(round(0.1 * i, 1) for i in range(2, 11))  # the HP curve's points, as a fraction of max HP
AUDIT_TOLERANCE = 0.01  # floors a band's mean curve may dip between two HP points before V is called not increasing
MAP_TABLE_PATH = ROOT / "data" / "run_currency_map_mean.json"


# ------------------------------------------------------------------------------------------------------------------- the head


@dataclass(frozen=True)
class Head:
    """The run head, numpy: both nets and what a won run is worth. `floors` is its expected final floor of a batch of contexts."""

    nets: models.NumpyNets
    w: float = W_WON
    sha256: str = ""

    @classmethod
    def load(cls, path: str | Path, w: float = W_WON) -> Head:
        path = Path(path)
        return cls(models.load_numpy(path), w, file_sha256(path))

    def floors(self, ctxs: Sequence[Ctx], table: MapTable | None = None) -> np.ndarray:
        """Expected final floor of each context, (n,) float64. Rows are grouped by region and boss, which fix the fight block's queries."""
        if not ctxs:
            return np.zeros(0)
        out = np.zeros(len(ctxs))
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, c in enumerate(ctxs):
            groups[(c.view.act_region, c.view.boss)].append(i)
        for rows in groups.values():
            bases = np.stack([encode_ctx(ctxs[i], table) for i in rows])[:, : self.nets.base_size]
            block = self.nets.fight.summary(bases, features.fight_queries(ctxs[rows[0]].view))
            probs = self.nets.value.floor_probs(np.concatenate([bases, block], axis=1))
            out[rows] = models.expected_floors(probs, self.w)
        return out


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def referee_root() -> Path:
    return Path(os.environ.get(REFEREE_ENV, "D:/spireline/referee"))


def default_head_path() -> Path:
    """The head the currency prices with: `$SPIRELINE_RUN_CURRENCY_HEAD`, else generation 11's V_run fit."""
    named = os.environ.get(HEAD_ENV)
    return Path(named) if named else referee_root() / "agent" / "gen11" / "candidate" / "run_value.npz"


@lru_cache(maxsize=4)
def _load_head(path: str, mtime_ns: int, pinned: bool) -> Head:
    head = Head.load(path)
    if pinned and head.sha256 != FROZEN_HEAD_SHA256:
        raise ValueError(f"{path} is not the frozen run head (sha256 {head.sha256[:8]}, expected {FROZEN_HEAD_SHA256[:8]}): the currency "
                         f"is frozen at generation 11's V_run until a refit passes the ranking check, the audit and the exam (design 3.4)")
    return head


def default_head() -> Head:
    """The process's head, loaded once. The frozen one is pinned by its sha256; a head named in the environment is taken as it is."""
    path = default_head_path()
    return _load_head(str(path.resolve()), path.stat().st_mtime_ns, HEAD_ENV not in os.environ)


# ---------------------------------------------------------------------------------------------------------------- the contexts


@dataclass(frozen=True)
class Ctx:
    """One run state the head is asked about: a view, and the node type of the fight it stands at (for the map block of a view without
    a map). A fight whose act boss is unknown has one context per candidate boss, and the value is their mean."""

    view: RunView
    node_type: int | None = None

    @property
    def exact(self) -> bool:
        return bool(self.view.map_nodes)


class MapTable:
    """The mean map-ahead block (block C) by rows to the boss and node type, over the copy's Act 1 maps.

    `block(rows_to_boss, node_type)` is the mean of `features.map_ahead` over every node of that type that many rows below the boss; a
    (distance, type) the maps never had is the nearest distance's of that type."""

    def __init__(self, blocks: dict[tuple[int, int], np.ndarray]):
        if not blocks:
            raise ValueError("an empty map table")
        self.blocks = {k: np.asarray(v, dtype=np.float32) for k, v in blocks.items()}
        self._by_type: dict[int, list[int]] = defaultdict(list)
        for d, t in self.blocks:
            self._by_type[t].append(d)

    def block(self, rows_to_boss: int, node_type: int | None) -> np.ndarray:
        t = node_type if node_type in self._by_type else (NODE_MONSTER if NODE_MONSTER in self._by_type else next(iter(self._by_type)))
        d = min(self._by_type[t], key=lambda x: (abs(x - rows_to_boss), x))
        return self.blocks[(d, t)]

    @classmethod
    def from_json(cls, doc: dict) -> MapTable:
        return cls({(int(d), int(t)): v for d, per in doc["blocks"].items() for t, v in per.items()})

    def to_json(self) -> dict:
        out: dict[str, dict[str, list[float]]] = {}
        for (d, t), v in sorted(self.blocks.items()):
            out.setdefault(str(d), {})[str(t)] = [round(float(x), 6) for x in v]
        return {"what": "mean map-ahead block by rows to the boss and node type", "blocks": out}


@lru_cache(maxsize=1)
def default_table() -> MapTable:
    return MapTable.from_json(json.loads(MAP_TABLE_PATH.read_text(encoding="utf-8")))


def rows_to_boss(view: RunView) -> int:
    """Floors between the view's floor and its act's boss floor (the rows left on the map)."""
    return max(0, BOSS_FLOORS[min(max(view.act_index, 0), len(BOSS_FLOORS) - 1)] - view.floor)


def encode_ctx(ctx: Ctx, table: MapTable | None = None) -> np.ndarray:
    """The base vector of a context. A view with a map is `features.encode` exactly; one without gets the mean map block of its rows to
    the boss and node type, and the scalar that counts those rows."""
    x = features.encode(ctx.view)
    if not ctx.exact:
        d = rows_to_boss(ctx.view)
        x[features.A_OFFSET + 15] = d / features.MAP_ROWS
        x[features.C_OFFSET : features.C_OFFSET + features.C_SIZE] = (table or default_table()).block(d, ctx.node_type)
    return x


def view_to_json(view: RunView) -> dict:
    """A run view as a plain document (the start records' `run_view`); `view_from_json` is its inverse."""
    return {"ascension": view.ascension, "act_index": view.act_index, "act_region": view.act_region, "floor": view.floor, "hp": view.hp,
            "max_hp": view.max_hp, "gold": view.gold, "deck": [[c.card_id, int(c.upgraded), c.enchantment, c.enchant_amount] for c in view.deck],
            "relics": list(view.relics), "potions": list(view.potions), "max_potion_slots": view.max_potion_slots,
            "map": [[n.col, n.row, n.node_type, [list(ch) for ch in n.children]] for n in view.map_nodes],
            "position": list(view.position), "boss": view.boss}


def view_from_json(doc: dict) -> RunView:
    return RunView(ascension=int(doc["ascension"]), act_index=int(doc["act_index"]), act_region=int(doc["act_region"]), floor=int(doc["floor"]),
                   hp=int(doc["hp"]), max_hp=int(doc["max_hp"]), gold=int(doc["gold"]),
                   deck=tuple(CardRef(int(c[0]), bool(c[1]), int(c[2]), int(c[3])) for c in doc["deck"]),
                   relics=tuple(int(r) for r in doc["relics"]), potions=tuple(int(p) for p in doc["potions"]),
                   max_potion_slots=int(doc["max_potion_slots"]),
                   map_nodes=tuple(MapNode(int(n[0]), int(n[1]), int(n[2]), tuple((int(c), int(r)) for c, r in n[3])) for n in doc.get("map") or []),
                   position=(int(doc["position"][0]), int(doc["position"][1])), boss=int(doc["boss"]))


def doc_cards(doc: dict) -> list[dict]:
    """Every card of a snapshot document, in all its piles (the draw pile keeps a known top, a known bottom and an unordered middle)."""
    out: list[dict] = []
    for pile in (doc.get("piles") or {}).values():
        if isinstance(pile, dict):
            for part in pile.values():
                out.extend(part)
        else:
            out.extend(pile)
    return out


def regions_of(act: int, encounter_id: int | None) -> list[int]:
    """The regions an act's fight can be in: Act 1 has two, told apart by the encounter when only one pool holds it."""
    if act >= 3:
        return [vocab.REGION_GLORY]
    if act == 2:
        return [vocab.REGION_HIVE]
    pools = vocab.encounter_pools()
    both = [vocab.REGION_OVERGROWTH, vocab.REGION_UNDERDOCKS]
    holding = [r for r in both if encounter_id is not None and any(encounter_id in ids for ids in pools[r].values())]
    return holding or both


def contexts_of_start(start: dict) -> list[Ctx]:
    """The run states a fight start stands in. A record with a `run_view` (a replay's) has the one exact context; otherwise the view is
    built from the snapshot document (see the module docstring) with a context for every region and boss still possible."""
    node_type = start.get("node_type")
    node = int(node_type) if node_type is not None else None
    if start.get("run_view"):
        return [Ctx(view_from_json(start["run_view"]), node)]
    doc = start["doc"]
    player = doc["player"]
    act = int(start.get("act") or 1)
    floor = int(start["floor"])
    slots = doc.get("potions") or {}
    max_slots = int(slots.get("max_slots", 2))
    potions = tuple(int(p) for p in (slots.get("slots") or [])[:max_slots] if p)
    deck = tuple(CardRef(int(c["def_id"]), bool(c.get("upgraded")), int(c.get("enchantment") or 0), int(c.get("enchant_amount") or 0))
                 for c in doc_cards(doc))
    relics = tuple(int(r["id"]) for r in doc.get("relics") or [])
    encounter = doc.get("encounter_id", start.get("encounter_id"))
    pools = vocab.encounter_pools()
    out = []
    for region in regions_of(act, encounter):
        bosses = [int(encounter)] if node == NODE_BOSS and encounter in pools[region]["boss"] else list(pools[region]["boss"])
        for boss in bosses:
            view = RunView(ascension=int(doc.get("ascension", 10)), act_index=act - 1, act_region=region, floor=floor, hp=int(player["hp"]),
                           max_hp=int(player["max_hp"]), gold=int(player.get("gold", 0)), deck=deck, relics=relics, potions=potions,
                           max_potion_slots=max_slots, map_nodes=(), position=(0, 0), boss=boss)
            out.append(Ctx(view, node))
    return out


def end_heal_of(relics: Iterable[int]) -> int:
    """HP the run layer adds after a won combat: Burning Blood's 6 or Black Blood's 12 (Black Blood replaces Burning Blood)."""
    held = set(relics)
    return max((heal for name, heal in END_HEALS.items() if vocab.relic_id_by_name(name) in held), default=0)


# ----------------------------------------------------------------------------------------------------------------- the currency


class FightCurrency:
    """One fight's terminal value. Built at the fight's start from its contexts (usually one) and the head.

    `u(hp, max_hp, potions)` is the normalized value of a won fight, `__call__` the search's units; `potions` is a tuple of potion ids held
    at the end, or a number (a judge leaf's expected count). Memoized by key, so a search's hot path is a dictionary lookup."""

    def __init__(self, head, ctxs: Sequence[Ctx], *, table: MapTable | None = None, floor: int | None = None):
        if not ctxs:
            raise ValueError("a fight currency needs at least one context")
        self.head, self.ctxs, self.table = head, tuple(ctxs), table
        v0 = ctxs[0].view
        self.floor = float(v0.floor if floor is None else floor)
        self.max_hp0 = int(v0.max_hp)
        self.hp0 = int(v0.hp)
        self.potions0 = tuple(sorted(v0.potions))
        self.end_heal = end_heal_of(v0.relics)
        self._memo: dict[tuple, float] = {}
        self._count_memo: dict[tuple, float] = {}
        self.stats: Counter = Counter()
        self.t_lost = self.floor
        self.t_full = self._t(self.max_hp0, self.max_hp0, self.potions0)
        self.span = self.t_full - self.t_lost
        self.fallback = not self.span >= MIN_SPAN  # also true for a NaN

    # -- the head's value of a state

    def _states(self, hp: int, max_hp: int, potions: tuple[int, ...]) -> list[Ctx]:
        return [dataclasses.replace(c, view=dataclasses.replace(c.view, hp=hp, max_hp=max_hp, potions=potions)) for c in self.ctxs]

    def _key(self, hp: int, max_hp: int, potions: Iterable[int]) -> tuple[int, int, tuple[int, ...]]:
        max_hp = int(max_hp)
        return (int(min(max_hp, hp + self.end_heal)), max_hp, tuple(sorted(potions)))

    def _t(self, hp: int, max_hp: int, potions: Iterable[int]) -> float:
        """The head's expected final floor of the post-fight state (the end-of-combat heal added), mean over the contexts."""
        key = self._key(hp, max_hp, potions)
        got = self._memo.get(key)
        if got is None:
            self.stats["evals"] += 1
            got = self._memo[key] = float(np.mean(self.head.floors(self._states(*key), self.table)))
        return got

    def prefill(self, states: Iterable[tuple[int, int, Iterable[int]]]) -> None:
        """Evaluate many (hp, max HP, potions) states in one batch (the head's cost is mostly per call)."""
        todo: list[tuple[int, int, tuple[int, ...]]] = []
        seen = set()
        for hp, max_hp, potions in states:
            key = self._key(hp, max_hp, potions)
            if key not in self._memo and key not in seen:
                seen.add(key)
                todo.append(key)
        if not todo:
            return
        ctxs = [c for key in todo for c in self._states(*key)]
        values = self.head.floors(ctxs, self.table).reshape(len(todo), len(self.ctxs)).mean(axis=1)
        self.stats["evals"] += len(todo)
        self._memo.update(zip(todo, (float(v) for v in values), strict=True))

    def _t_count(self, hp: int, max_hp: int, n: int) -> float:
        """T with `n` of the starting potions held, averaged over which ones."""
        n = max(0, min(n, len(self.potions0)))
        key = (hp, max_hp, n)
        got = self._count_memo.get(key)
        if got is None:
            subsets = sorted(set(itertools.combinations(self.potions0, n)))
            got = self._count_memo[key] = float(np.mean([self._t(hp, max_hp, s) for s in subsets]))
        return got

    # -- the normalized value and the search's units

    def u(self, hp: float, max_hp: float, potions: tuple[int, ...] | float) -> float:
        """(T - T_lost) / (T_full - T_lost) of a won fight ending with `hp` of `max_hp` and `potions` held, clipped to [0, U_MAX]."""
        hp, max_hp = int(round(hp)), int(round(max_hp))
        if isinstance(potions, int | float | np.integer | np.floating):
            lo = int(math.floor(potions))
            t = self._t_count(hp, max_hp, lo)
            frac = float(potions) - lo
            if frac > 1e-9:
                t = (1.0 - frac) * t + frac * self._t_count(hp, max_hp, lo + 1)
        else:
            t = self._t(hp, max_hp, potions)
        raw = (t - self.t_lost) / self.span
        self.stats["priced"] += 1
        if raw < 0.0:
            self.stats["clipped_low"] += 1
        elif raw > U_MAX:
            self.stats["clipped_high"] += 1
        return float(min(U_MAX, max(0.0, raw)))

    def __call__(self, hp: float, max_hp: float, potions: tuple[int, ...] | float) -> float:
        """The value of a won fight in the search's units; a fight with no usable scale is priced as today (HP + 10 a potion)."""
        if self.fallback:
            self.stats["fallback"] += 1
            n = len(potions) if isinstance(potions, tuple | list) else float(potions)
            return float(hp) + POTION_VALUE * n
        return DEATH_VALUE + self.u(hp, max_hp, potions) * (self.max_hp0 - DEATH_VALUE)

    def summary(self) -> dict:
        """What the fight's pricing did, for a row: head evaluations, states priced, clips, and whether the fight fell back."""
        return {"evals": int(self.stats["evals"]), "priced": int(self.stats["priced"]), "clipped_low": int(self.stats["clipped_low"]),
                "clipped_high": int(self.stats["clipped_high"]), "fallback": bool(self.fallback), "t_lost": round(self.t_lost, 3),
                "t_full": round(float(self.t_full), 3)}

    @classmethod
    def for_view(cls, view: RunView, head: Head | None = None, *, table: MapTable | None = None) -> FightCurrency:
        """The currency of the fight a run stands at, from the run's exact view (`RunSession.view()` at the fight's first decision)."""
        return cls(head or default_head(), [Ctx(view, None)], table=table)

    @classmethod
    def for_start(cls, start: dict, head: Head | None = None, *, table: MapTable | None = None) -> FightCurrency:
        return cls(head or default_head(), contexts_of_start(start), table=table)


_CURRENCIES: dict[str, FightCurrency] = {}


def currency_for_start(start: dict, head: Head | None = None) -> FightCurrency:
    """A start's currency, kept for the process (a start is played on several draws, and the memo is the same for all of them)."""
    key = f"{start.get('id')}|{start.get('floor')}"
    got = _CURRENCIES.get(key)
    if got is None:
        if len(_CURRENCIES) >= 64:
            _CURRENCIES.pop(next(iter(_CURRENCIES)))
        got = _CURRENCIES[key] = FightCurrency.for_start(start, head)
    return got


# ---------------------------------------------------------------------------------------------------------------------- the audit


def band_of(view: RunView) -> str:
    """The floor band of a view within its act: early (11 or more floors to the boss), mid (6 to 10) or late."""
    d = rows_to_boss(view)
    return "early" if d >= 11 else "mid" if d >= 6 else "late"


def kind_of(node_type: int | None) -> str:
    return {NODE_MONSTER: "hallway", NODE_ELITE: "elite", NODE_BOSS: "boss"}.get(node_type or 0, "fight")


def _weighted_median(pairs: Sequence[tuple[float, int]]) -> float:
    pairs = sorted(pairs)
    total = sum(n for _, n in pairs)
    acc = 0
    for value, n in pairs:
        acc += n
        if acc * 2 >= total:
            return value
    return pairs[-1][0]


def audit(head, starts: Sequence[dict], *, per_cell: int = 60, table: MapTable | None = None, kinds: Sequence[int] | None = None) -> dict:
    """The head's marginal table over fight starts (design 3.3) and its stop rule (design 3.4).

    Per start (its first context, so a start with no known boss is read with its first candidate): V over the HP curve (`AUDIT_GRID` of
    max HP, potions as held), 10 more HP at 30, 60 and 90% of max HP, each potion kind alone against an empty belt at the start's HP, and
    50 gold. Per cell (act and kind) the means; the HP-to-potion ratio is the mean potion marginal over the mean HP marginal at 60%,
    against today's 10 HP. The rule (`stop` lists what trips it): in Act 1 hallway fights a potion worth under 3 or over 40 HP (the
    median over the potion kinds the sampled belts held, weighted by how often they were held; every kind's own figure is reported), and
    every cell and (act, floor band) whose mean HP curve falls by more than `AUDIT_TOLERANCE` floors between two points."""
    kinds = sorted(vocab.potion_ids()) if kinds is None else list(kinds)
    by_cell: dict[str, list[Ctx]] = defaultdict(list)
    for s in starts:
        ctx = contexts_of_start(s)[0]
        by_cell[f"act{ctx.view.act_index + 1}-{kind_of(ctx.node_type)}"].append(ctx)
    states: list[Ctx] = []
    recs: dict[str, list[dict]] = defaultdict(list)
    for cell, ctxs in sorted(by_cell.items()):
        for ctx in ctxs[:per_cell]:
            v = dataclasses.replace(ctx.view, max_potion_slots=max(ctx.view.max_potion_slots, 3))
            rec: dict = {"band": band_of(v), "act": v.act_index + 1, "held": list(v.potions), "ix": {}}

            def add(tag, view, rec=rec, node=ctx.node_type):
                rec["ix"][tag] = len(states)
                states.append(Ctx(view, node))

            for f in AUDIT_GRID:
                add(("curve", f), dataclasses.replace(v, hp=max(1, round(f * v.max_hp))))
            for f in AUDIT_HP_FRACTIONS:
                hp = max(1, min(v.max_hp - 10, round(f * v.max_hp)))
                add(("hp10", f, 0), dataclasses.replace(v, hp=hp, potions=()))
                add(("hp10", f, 1), dataclasses.replace(v, hp=hp + 10, potions=()))
            add(("none",), dataclasses.replace(v, potions=()))
            for k in kinds:
                add(("potion", k), dataclasses.replace(v, potions=(k,)))
            add(("gold", 0), dataclasses.replace(v, potions=()))
            add(("gold", 1), dataclasses.replace(v, potions=(), gold=v.gold + 50))
            recs[cell].append(rec)
    values = head.floors(states, table)
    cells: dict[str, dict] = {}
    curves_by_name: dict[str, list[np.ndarray]] = defaultdict(list)
    stop: list[str] = []
    for cell, rs in recs.items():
        curves = np.array([[values[r["ix"][("curve", f)]] for f in AUDIT_GRID] for r in rs])
        hp10 = {f: np.array([values[r["ix"][("hp10", f, 1)]] - values[r["ix"][("hp10", f, 0)]] for r in rs]) for f in AUDIT_HP_FRACTIONS}
        pot = {k: float(np.mean([values[r["ix"][("potion", k)]] - values[r["ix"][("none",)]] for r in rs])) for k in kinds}
        gold = np.array([values[r["ix"][("gold", 1)]] - values[r["ix"][("gold", 0)]] for r in rs])
        hp_marginal = float(hp10[0.6].mean()) / 10.0  # floors per HP at 60% of max HP
        mean_potion = float(np.mean(list(pot.values())))
        held = Counter(p for r in rs for p in r["held"])
        cells[cell] = {
            "states": len(rs), "curve_mean": [round(float(x), 4) for x in curves.mean(axis=0)],
            "floors_per_10hp": {f"{int(round(f * 100))}": round(float(m.mean()), 4) for f, m in hp10.items()},
            "floors_per_potion_mean": round(mean_potion, 4), "floors_per_50_gold": round(float(gold.mean()), 4),
            "potion_hp_equivalent": round(mean_potion / hp_marginal, 2) if hp_marginal > 0 else None,
            "potion_hp_equivalent_today": POTION_VALUE,
        }
        if cell == "act1-hallway":
            eq = {k: pot[k] / hp_marginal for k in kinds} if hp_marginal > 0 else {}
            weighted = [(eq[k], n) for k, n in held.items() if k in eq]
            med = _weighted_median(weighted) if weighted else None
            cells[cell]["potion_hp_equivalent_by_kind"] = {str(k): round(x, 2) for k, x in sorted(eq.items())}
            cells[cell]["potion_hp_equivalent_median_held"] = None if med is None else round(med, 2)
            if hp_marginal <= 0:
                stop.append("act1-hallway: V does not rise with HP at 60% of max HP, so a potion has no price in HP")
            elif med is None:
                stop.append("act1-hallway: no potion held in the sampled starts to price")
            elif not AUDIT_POTION_HP[0] <= med <= AUDIT_POTION_HP[1]:
                stop.append(f"act1-hallway: a potion (median over the kinds held) is worth {med:.1f} HP, outside "
                            f"{AUDIT_POTION_HP[0]:g} to {AUDIT_POTION_HP[1]:g}")
        curves_by_name[cell].extend(curves)
        for r, curve in zip(rs, curves, strict=True):
            curves_by_name[f"act{r['act']}-{r['band']}"].append(curve)
    mono: dict[str, dict] = {}
    for name, curves in sorted(curves_by_name.items()):
        mean = np.array(curves).mean(axis=0)
        worst = float(np.min(np.diff(mean)))
        mono[name] = {"states": len(curves), "curve_mean": [round(float(x), 4) for x in mean], "worst_step": round(worst, 4),
                      "increasing": worst >= -AUDIT_TOLERANCE}
        if worst < -AUDIT_TOLERANCE:
            stop.append(f"{name}: V is not increasing in HP (the mean curve falls {-worst:.3f} floors between two points)")
    return {"head_sha256": getattr(head, "sha256", ""), "states_evaluated": len(states), "hp_grid": list(AUDIT_GRID), "cells": cells,
            "monotone": mono, "stop": stop, "passed": not stop}


def audit_markdown(rep: dict) -> str:
    lines = [f"run head {str(rep['head_sha256'])[:8]}: {rep['states_evaluated']} states evaluated; {'PASSED' if rep['passed'] else 'STOP'}", "",
             "| cell | states | floors per 10 HP at 30 / 60 / 90% | floors per potion | HP a potion is worth (today 10) | floors per 50 gold |",
             "|---|---|---|---|---|---|"]
    for cell, c in sorted(rep["cells"].items()):
        hp = c["floors_per_10hp"]
        lines.append(f"| {cell} | {c['states']} | {hp['30']:.3f} / {hp['60']:.3f} / {hp['90']:.3f} | {c['floors_per_potion_mean']:.3f} | "
                     f"{c['potion_hp_equivalent']} | {c['floors_per_50_gold']:.3f} |")
    lines += ["", "| act and band | states | worst step of the mean HP curve (floors) | increasing |", "|---|---|---|---|"]
    for name, m in sorted(rep["monotone"].items()):
        lines.append(f"| {name} | {m['states']} | {m['worst_step']:+.4f} | {m['increasing']} |")
    lines += ["", "stop reasons: " + ("; ".join(rep["stop"]) if rep["stop"] else "none")]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------------------------- the map table


def build_map_table(seeds: Sequence[str], ascension: int = 10) -> MapTable:
    """Mean block C by rows to the boss and node type over the Act 1 maps of `seeds` (the run engine; no fight is played)."""
    from run_train import engine  # noqa: PLC0415 -- the emulator library

    sums: dict[tuple[int, int], np.ndarray] = {}
    counts: Counter = Counter()
    for seed in seeds:
        env = engine.new_run_env(seed, ascension, "IRONCLAD", 1)
        try:
            env.reset()
            view = features.view_from_engine(env, ascension)
        finally:
            env.close()
        boss_row = features._boss_row(view.map_nodes)  # noqa: SLF001
        for node in view.map_nodes:
            if node.node_type not in (NODE_MONSTER, NODE_ELITE, NODE_BOSS):
                continue
            block = features.map_ahead(dataclasses.replace(view, position=(node.col, node.row)))
            key = (boss_row - node.row, node.node_type)
            sums[key] = sums.get(key, 0) + block.astype(np.float64)
            counts[key] += 1
    return MapTable({k: v / counts[k] for k, v in sums.items()})


# -------------------------------------------------------------------------------------------------------------------------- CLI


def _read_starts(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def cmd_audit(a: argparse.Namespace) -> int:
    starts = _read_starts(Path(a.starts))
    rep = audit(Head.load(a.head) if a.head else default_head(), starts, per_cell=a.per_cell)
    text = audit_markdown(rep)
    print(text)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        (out / "audit.md").write_text(text, encoding="utf-8")
    return 0 if rep["passed"] else 2


def cmd_map_table(a: argparse.Namespace) -> int:
    seeds = [f"MAPT{i:05d}" for i in range(a.seeds)]
    table = build_map_table(seeds)
    out = Path(a.out)
    out.write_text(json.dumps(table.to_json(), indent=1), encoding="utf-8")
    print(f"{len(table.blocks)} (rows to boss, node type) blocks from {len(seeds)} maps -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--starts", required=True, help="a start file with snapshot documents (the exam's, the board's)")
    p.add_argument("--per-cell", type=int, default=60)
    p.add_argument("--head", default="", help="a head to audit instead of the frozen one")
    p.add_argument("--out", default="")
    p = sub.add_parser("map-table")
    p.add_argument("--seeds", type=int, default=300)
    p.add_argument("--out", default=str(MAP_TABLE_PATH))
    a = ap.parse_args(argv)
    return {"audit": cmd_audit, "map-table": cmd_map_table}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
