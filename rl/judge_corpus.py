"""The judge's corpus (D58 C1, D57 addendum 7; 2026-11-11): fight positions with labels, strata and sampling weights, in a store the judge
lane trains from.

    python rl/judge_corpus.py plan      [--store DIR]    # the human pool, its features, the sampling order (reads starts, never the corpus)
    python rl/judge_corpus.py work      [--store DIR] ...  # one process: the next fights of the plan (claim files; BelowNormal)
    python rl/judge_corpus.py run       [--store DIR] [--procs 8]   # a bounded supervisor: starts the work processes, restarts a dead one, ends with the plan
    python rl/judge_corpus.py compact   [--store DIR]    # the finished fights as shards + the manifest (safe to run while fights are still being built)
    python rl/judge_corpus.py selfplay  [--store DIR]    # the loop's buffer as self-play shards
    python rl/judge_corpus.py coverage  [--store DIR]    # coverage.md and the manifest's counts
    python rl/judge_corpus.py status    [--store DIR]    # fights done, errors, CPU so far

What it holds. Two sources of fight positions with labels:
  human      the fights of the human fight starts (`human_starts.py`: the four training fifths of the public run export; the exam fifth and
             the fight exam's own human starts never) replayed by the teacher (player v3 under `a3-v1@h1e4b16`, the CPU prior): the fight
             is played once, then three turns are drawn (early, middle, late third of the fight's decisions that have two or more distinct
             moves, one each by a hash coin) and the door position (the first decision, the human's own position) is kept as well. Every
             position can be reached again by replaying the teacher's moves (`replay`). Labels: the human's own result of that fight
             (survived, HP, potions: the sidecar `human-outcomes.json`), the teacher's own result of this very game, and for boss and
             elite turns the search settled on a ladder (`converge.py`): the game's own search is the first rung (x16 at a boss, x4 at an
             elite), then x32 (x8) and, only where the rule has not held, x64 (x16) is the cap; the rule is `converge.rule_holds` with
             the paired comparison (the two most visited moves played in the same sampled worlds). A turn the rule never held for keeps
             the capped rung's values and is marked `soft`. A hallway turn is labelled by the teacher's own search at its budget (`played`).
  selfplay   the always-on loop's replay buffer: every usable position of the x16 teacher's fights (the loop's own starts, human starts and
             pre-loss starts), labelled with the search as played (visits and values per move) and the fight's outcome.
Strata (D57 addendum 7): kind, act, enemy (a boss a cell), turn stage, deck type (`deck_tag`), relics and cards (presence), health band,
potion state. The human starts are drawn by sequential Poisson sampling: each start has a boost (`boost_weights`: the rarest of its relics,
cards and enemy against a minimum support) and a priority `u / boost`; a cell's fights are taken in priority order, so any prefix is a
usable sample and the weight of a start, 1 / inclusion probability (`inclusion`), brings the oversampled strata back to the human pool's.

Every position row is one JSON line in `shards/<source>-<kind>-act<A>-<NNNN>.jsonl`; its tokens (the judge's input: `types`, `ids`, `owners`,
`feats`) are the same row of `<shard>.npz`. `Corpus` reads them back (`iter_positions`, `arrays`: the npz layout `judge.Positions` reads).

The corpus rules (D13/D38/D55) hold: derived game positions and aggregates only; no per-player id exists in any file (the corpus's single-player
runs carry none); no run record leaves the corpus folder; the exam fifth is never an input (`human_starts.assert_no_exam`, and the loader
refuses any human row of it); nothing here is copied from the corpus folder but the derived positions, as the human starts already are.
Attribution: Spire Codex public run export for v0.107.1. A cell with fewer than `MIN_CELL` runs or positions is not published.
"""

from __future__ import annotations

import argparse
import base64
import bisect
import contextlib
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCHEMA = "judge-corpus-v1"
NAMESPACE = "judge-corpus-v1"
TAG = "judge-corpus"
STORE = Path(os.environ.get("JUDGE_CORPUS_STORE", "D:/spireline/referee/judge/corpus"))
LOOP = Path(os.environ.get("JUDGE_CORPUS_LOOP", "D:/spireline/referee/gym-loop"))
REFEREE = Path(os.environ.get("JUDGE_CORPUS_REFEREE", "D:/spireline/referee"))
ATTRIBUTION = "Spire Codex public run export for v0.107.1"
MIN_CELL = 20  # a cell under this many runs or positions is never published
HELD_OUT = 5  # one run in five is held out, by the hash of its run id (`split_of`)
KINDS = ("boss", "elite", "hallway")
LADDER_KINDS = ("boss", "elite")  # the kinds whose turns are searched on the ladder; a hallway turn keeps the teacher's own search
STAGES = ("early", "middle", "late")
BANDS = ("healthy", "hurt", "critical")
BAND_EDGES = (1 / 3, 2 / 3)  # HP fraction below the first: critical; below the second: hurt
ARCHETYPES = ("starter", "power", "exhaust", "block", "attack", "mixed")
ORDER_BLOCK = 100  # the plan's order is cells in turn, this many fights of each at a time (the cells run side by side, so the first hours give all four of the dear cells)
CELLS = ("act2-boss", "act2-elite", "act1-boss", "act1-elite", "act2-hallway", "act1-hallway")  # the order the plan draws from
# fights to build in each cell (human starts): a boss or elite fight gives about 3.85 labelled positions (the door and three turns), so the Act 2 bosses
# (all 3,700 of the pool) and 1,500 Act 1 bosses give 20,000 boss turns, 5,400 elite fights 20,700; a hallway fight gives about 3.7 positions, so 2,800 give
# 10,400. Measured: an Act 2 boss fight costs 60 to 110 CPU-s on this shared machine, so the quotas stop at the brief's counts; raise them to extend
QUOTA = {"act2-boss": 3900, "act1-boss": 1500, "act2-elite": 2700, "act1-elite": 2700, "act2-hallway": 1400, "act1-hallway": 1400}
SAMPLE_RATE = {"boss": 0.5, "elite": 0.2, "hallway": 0.65}  # the share of a kind's pool the plan aims at; fixed, so a longer quota extends the same order
MIN_SUPPORT_STARTS = 200  # a relic, card or enemy asks for this many starts in the sample (`boost_weights`, with the sampling rate); measured: 50 to 100 are reached
BOOST_CAP = 1000.0
LADDER_RELS = (2, 4)  # rungs searched after the game's own: x32 then x64 at a boss, x8 then x16 at an elite (the cap is the last)
DUEL_PAIRS = {2: 128, 4: 256}  # shared worlds of the paired comparison at each rung
SHARD_ROWS = 4000
# the token layout (combat_tokens; a native test pins these to it)
T_GLOBAL, T_HAND, T_DRAW, T_DISCARD, T_EXHAUST, T_POTION, T_HAND_LOCKED, T_RELIC = 1, 2, 3, 4, 5, 8, 13, 14
F_HP, F_MAGNITUDE = 0, 13
DECK_TYPES = (T_HAND, T_HAND_LOCKED, T_DRAW, T_DISCARD, T_EXHAUST)


# ---------------------------------------------------------------------------------------------------------- digests, splits, stages


def digest(*parts: Any) -> int:
    """A fixed 64-bit number from the parts (the corpus's own namespace)."""
    text = "|".join(str(p) for p in (NAMESPACE, *parts))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def split_of(run: str) -> str:
    """"heldout" for one run in HELD_OUT (the first 32 bits of the sha256 of the run id), else "train": the judge's split, by run (a
    run's fights share a deck, so a split by start would put one deck on both sides). Human and self-play rows use the same rule."""
    return "heldout" if int(hashlib.sha256(str(run).encode()).hexdigest()[:8], 16) % HELD_OUT == 0 else "train"


def stage_of(index: int, n: int) -> str:
    """The stage of the `index`th of `n` eligible decisions: the fight's decisions in thirds, early, middle, late."""
    return STAGES[min(2, 3 * index // max(1, n))]


def pick_stage_turns(start_id: str, eligible: Sequence[int], per_stage: int = 1) -> list[dict]:
    """Per stage, `per_stage` decisions of the eligible ones (two or more distinct root moves) in that third, each by a hash coin of the
    fight (fixed: the same fight always gives the same turns): [{k, stage, stage_n}], in decision order. A stage with no decision gives none."""
    elig = sorted(eligible)
    out: list[dict] = []
    for stage in STAGES:
        members = [k for i, k in enumerate(elig) if stage_of(i, len(elig)) == stage]
        pool = list(members)
        for j in range(per_stage):
            if not pool:
                break
            u = digest("turn", start_id, stage, j) / 2**64
            out.append({"k": pool.pop(int(u * len(pool))), "stage": stage, "stage_n": len(members)})
    return sorted(out, key=lambda p: p["k"])


def health_band(frac: float) -> str:
    return "critical" if frac < BAND_EDGES[0] else "hurt" if frac < BAND_EDGES[1] else "healthy"


def potion_state(n: int) -> str:
    return "none" if n <= 0 else "one" if n == 1 else "two+"


# ---------------------------------------------------------------------------------------------------------- deck type and token features


def deck_tag(counts: Mapping[int, int], cards: Mapping[int, Any]) -> str:
    """A simple archetype tag from the deck's dominant cards. Over the cards that are not basic, curse, status or quest (the deck's
    "core"): `starter` with three or fewer; `power` with at least three powers making a quarter of the core; `exhaust` with at least three
    exhaust cards making 30%; `block` with at least three cards that give block making 35%; `attack` with attacks making 55%; else
    `mixed`. The first that holds, in that order. `cards` is `names.cards()` (card id -> CardInfo)."""
    core = power = exhaust = block = attack = 0
    for cid, n in counts.items():
        info = cards.get(int(cid))
        if info is None or info.card_type in ("Curse", "Status", "Quest") or info.rarity == "Basic":
            continue
        core += n
        power += n * (info.card_type == "Power")
        attack += n * (info.card_type == "Attack")
        exhaust += n * bool(info.exhaust)
        block += n * (info.base_block > 0)
    if core <= 3:
        return "starter"
    if power >= 3 and power * 4 >= core:
        return "power"
    if exhaust >= 3 and exhaust * 10 >= core * 3:
        return "exhaust"
    if block >= 3 and block * 100 >= core * 35:
        return "block"
    if attack * 100 >= core * 55:
        return "attack"
    return "mixed"


def vocab_inverse() -> tuple[dict[int, int], dict[int, int]]:
    """(vocabulary index -> card id, vocabulary index -> relic id), as `combat_tokens.Vocab.build` numbers them (by sorted id, from 1);
    built here from the same tables so the loader and the coverage need no native library (a native test pins the two together)."""
    from sts2_gym import names  # noqa: PLC0415 -- parses the emulator's source tables only

    return ({i + 1: cid for i, cid in enumerate(sorted(names.cards()))}, {i + 1: rid for i, rid in enumerate(sorted(names.relics()))})


def token_features(types: np.ndarray, ids: np.ndarray, feats: np.ndarray, inv_card: Mapping[int, int], inv_relic: Mapping[int, int],
                   cards: Mapping[int, Any]) -> dict:
    """What a position's tokens say for the strata: HP fraction (the global token's), potions held, relics (ids), the deck (card id ->
    copies over the hand, draw, discard and exhaust piles; a card choice screen's candidates are not counted twice) and its tag."""
    types = np.asarray(types)
    ids = np.asarray(ids)
    glob = np.nonzero(types == T_GLOBAL)[0]
    hp_frac = float(feats[glob[0], F_HP]) if len(glob) else None
    potions = int(((types == T_POTION) & (np.asarray(feats)[:, F_MAGNITUDE] > 0.5)).sum())
    relics = sorted({inv_relic[int(i)] for i in ids[types == T_RELIC] if int(i) in inv_relic})
    deck: Counter = Counter(inv_card[int(i)] for i in ids[np.isin(types, DECK_TYPES)] if int(i) in inv_card)
    return {"hp_frac": hp_frac, "potions": potions, "relics": relics, "deck": {str(k): v for k, v in sorted(deck.items())},
            "deck_n": int(sum(deck.values())), "arch": deck_tag(deck, cards)}


# ---------------------------------------------------------------------------------------------------------- the human pool: boosts, order, weights


def feature_keys(relics: Iterable[int], deck: Iterable[int], enemy: str, arch: str) -> set[str]:
    """The features of one start that a minimum support is kept for: each relic and card in it, its enemy and its deck tag."""
    return {f"relic:{r}" for r in relics} | {f"card:{c}" for c in deck} | {f"enemy:{enemy}", f"arch:{arch}"}


def boost_weights(features: Sequence[set[str]], support: int = MIN_SUPPORT_STARTS, cap: float = BOOST_CAP, rate: float = 1.0) -> list[float]:
    """Per start, 1 or the rarest of its features' `support / (rate * the starts in the pool that have it)`, whichever is larger: with `rate` the
    share of the pool that will be sampled, a feature of n starts then gives about `rate * boost * n` starts to the sample, so a boost of
    `support / (rate * n)` is what takes a feature to `support` starts in the sample (a feature found in 20 starts of a pool and a support of
    100 gives 5 / rate; one in more than `support / rate` starts gets 1). The sampling order (`order_key`) puts the boosted starts first."""
    count: Counter = Counter(f for fs in features for f in fs)
    return [min(cap, max([1.0] + [support / (max(rate, 1e-9) * count[f]) for f in fs])) for fs in features]


def order_key(start_id: str, boost: float) -> float:
    """The start's place in its cell's sampling order, smaller first: `u / boost`, u a fixed uniform of the start id (sequential Poisson
    sampling: the first n of a cell are a sample whose inclusion probabilities follow the boosts, whatever n is)."""
    u = (digest("order", start_id) + 1) / (2**64 + 1)
    return u / boost


def inclusion(boosts: Sequence[float], n: int) -> list[float]:
    """The inclusion probability of each start when `n` of the cell are sampled: `min(1, lam * boost)` with lam fixed so that they add up
    to `n` (all 1 when n covers the cell). The start's training weight is its inverse."""
    if n >= len(boosts):
        return [1.0] * len(boosts)
    if n <= 0:
        return [0.0] * len(boosts)
    lo, hi = 0.0, 1.0
    while sum(min(1.0, hi * b) for b in boosts) < n:
        hi *= 2.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if sum(min(1.0, mid * b) for b in boosts) < n:
            lo = mid
        else:
            hi = mid
    return [min(1.0, hi * b) for b in boosts]


def plan_order(cells: Mapping[str, Sequence[str]], quota: Mapping[str, int], block: int = ORDER_BLOCK,
               cell_order: Sequence[str] = CELLS) -> list[str]:
    """The work order: `block` fights of each cell in turn (`cell_order`), each cell's own sampling order, until every cell's quota is
    met. Any prefix is a balanced sample of the cells that are still running."""
    out: list[str] = []
    taken = dict.fromkeys(cell_order, 0)
    while any(taken[c] < min(quota.get(c, 0), len(cells.get(c, ()))) for c in cell_order):
        for c in cell_order:
            want = min(quota.get(c, 0), len(cells.get(c, ())))
            take = min(block, want - taken[c])
            if take > 0:
                out += list(cells[c][taken[c] : taken[c] + take])
                taken[c] += take
    return out


# ---------------------------------------------------------------------------------------------------------- labels (pure)


def root_value(children: Sequence[dict]) -> float | None:
    """The search's value at the root: the visit-weighted mean of its moves' mean values (the search's `value_sum` over its visits)."""
    num = den = 0.0
    for c in children:
        if c.get("v") is not None and c.get("n", 0) > 0:
            num += c["n"] * c["v"]
            den += c["n"]
    return None if den <= 0 else num / den


def label_of(rec: dict, tier: str, rel: float | str, rungs: Sequence[dict] = ()) -> dict:
    """A position's search label from a search record (`converge.search_record`): the move, the visit distribution with each move's mean
    value and error, the root value, the best move's value and death rate, the rung's duel when it had one, and a short line for every
    rung searched (the move and the visits it chose, the cost, the duel). `tier`: settled, soft or played."""
    ch = rec["children"]
    visits = sum(c["n"] for c in ch)
    p_death = None
    num = sum(c["n"] * c["pd"] for c in ch if c.get("pd") is not None)
    den = sum(c["n"] for c in ch if c.get("pd") is not None)
    if den > 0:
        p_death = num / den
    best = rec["best"]
    out = {"tier": tier, "rel": rel, "budget": rec.get("budget"), "sims": rec.get("sims"), "visits": visits,
           "move": {k: best[k] for k in ("a", "t", "kind", "card")}, "root_value": root_value(ch), "best_value": best.get("v"), "p_death": p_death,
           "children": [{k: c[k] for k in ("a", "t", "kind", "card", "n", "v", "se", "pd")} for c in ch]}
    if rec.get("duel"):
        out["duel"] = rec["duel"]
    if rungs:
        out["rungs"] = [{"rel": r["rel"], "best": [r["best"]["a"], r["best"]["t"], r["best"]["kind"]], "n": r["best"]["n"], "sims": r.get("sims"),
                         "cpu_s": r.get("cpu_s"), "capped": r.get("capped"), "from_game": r.get("from_game", False),
                         "top2": [[c["a"], c["t"], c["kind"], c["n"], c["v"], c["se"]] for c in r["children"][:2]],  # the unpaired reading, for comparison
                         **({"duel": r["duel"], "duel_cpu_s": r.get("duel_cpu_s")} if r.get("duel") else {})} for r in rungs]
    return out


def settle(game_rec: dict, search: Callable[[int], dict], rels: Sequence[int] = LADDER_RELS, gap_se: float = 2.0, se_mode: str = "paired") -> dict:
    """The screen and the cap for one turn. The game's own search (the teacher's budget) is the first rung; then each rung of `rels` in turn
    (`search(rel)` returns its record, with a `duel`): the turn is settled at the first rung whose move equals the rung below's and whose
    top two moves' gap is at least `gap_se` standard errors (`converge.rule_holds`), and no deeper rung is searched. A turn the rule never
    held for is `soft`: the label is the last rung that was not capped. Returns {tier, rel, rec, rungs, cpu_s}."""
    import converge  # noqa: PLC0415

    first = {**game_rec, "rel": 1, "from_game": True}
    rungs = [first]
    turn = {"rungs": [{**game_rec, "rel": "game"}, first]}
    label_at = 1
    for rel in rels:
        rec = search(rel)
        rungs.append(rec)
        turn["rungs"].append(rec)
        if not rec.get("capped"):
            label_at = rel
        verdict = converge.rule_holds(turn, rel, gap_se, se_mode)
        if verdict:
            return {"tier": "settled", "rel": rel, "rec": rec, "rungs": rungs, "cpu_s": _cpu(rungs)}
    rec = next(r for r in rungs if r["rel"] == label_at)
    return {"tier": "soft", "rel": label_at, "rec": rec, "rungs": rungs, "cpu_s": _cpu(rungs)}


def _cpu(rungs: Sequence[dict]) -> float:
    return round(sum((r.get("cpu_s") or 0.0) + (r.get("duel_cpu_s") or 0.0) for r in rungs if not r.get("from_game")), 3)


def moves_string(moves: Sequence[Sequence[int]]) -> str:
    return ";".join(f"{a},{t}" for a, t in moves)


def moves_of(text: str) -> list[list[int]]:
    return [[int(x) for x in m.split(",")] for m in text.split(";")] if text else []


# ---------------------------------------------------------------------------------------------------------- the store: shards and tokens


def pack_tokens(items: Sequence[Mapping[str, np.ndarray]]) -> bytes:
    """Several positions' tokens as one compressed npz: `lens` and the flat `types`, `ids`, `owners`, `feats` (judge_data's layout)."""
    lens = np.array([len(i["types"]) for i in items], dtype=np.uint16)

    def cat(name: str, dtype: Any, width: int | None = None) -> np.ndarray:
        if not items:
            return np.zeros((0, width) if width else (0,), dtype)
        return np.concatenate([np.asarray(i[name]).astype(dtype) for i in items])

    buf = io.BytesIO()
    np.savez_compressed(buf, lens=lens, types=cat("types", np.uint8), ids=cat("ids", np.int32), owners=cat("owners", np.uint8),
                        feats=cat("feats", np.float32, 18))
    return buf.getvalue()


def unpack_tokens(data: bytes | Path) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data) as z:
        return {k: z[k] for k in z.files}


def b64_tokens(item: Mapping[str, np.ndarray]) -> str:
    return base64.b64encode(pack_tokens([item])).decode("ascii")


def tokens_of_b64(text: str) -> dict[str, np.ndarray]:
    z = unpack_tokens(base64.b64decode(text))
    return {k: z[k] for k in ("types", "ids", "owners", "feats")}


def write_shard(stem: Path, rows: Sequence[dict], toks: Sequence[Mapping[str, np.ndarray]]) -> None:
    """One shard: `<stem>.jsonl` (a row a line) and `<stem>.npz` (the tokens of the same rows, in order), each written whole."""
    if len(rows) != len(toks):
        raise ValueError("a shard needs one token set a row")
    stem.parent.mkdir(parents=True, exist_ok=True)
    tmp = stem.with_name(stem.name + ".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8", newline="\n")
    tmpz = stem.with_name(stem.name + ".npz.tmp")
    tmpz.write_bytes(pack_tokens(toks))
    os.replace(tmpz, stem.with_name(stem.name + ".npz"))
    os.replace(tmp, stem.with_name(stem.name + ".jsonl"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=False), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------------- weights


WEIGHT_MODES = ("pool", "cell", "none", "balanced")


def row_weight(row: Mapping[str, Any], mode: str = "pool") -> float:
    """A row's training weight. `pool`: 1 / the inclusion probability of its start (the human rows; 1 for a self-play row), which brings the
    oversampled strata back to the human pool's mix; `cell`: the same divided by the mean of its (source, kind, act) cell's, so a cell
    keeps its own size but not its oversampling (the compaction divides); `none`: 1. The compaction stores `w` as the first, the cell's
    normalised figure as `w_cell`; `Corpus.arrays(weight="balanced")` starts from `w_cell` and gives every (kind, act) cell of its selection the same
    total weight (a self-play buffer is half Act 1 boss positions)."""
    if mode == "none":
        return 1.0
    if mode == "cell":
        return float(row.get("w_cell", 1.0))
    if mode == "pool":
        return float(row.get("w", 1.0))
    if mode == "balanced":
        raise ValueError("the balanced weight needs the whole selection: `Corpus.arrays(weight='balanced')` makes it")
    raise ValueError(f"unknown weight mode {mode!r}; one of {', '.join(WEIGHT_MODES)}")


# ---------------------------------------------------------------------------------------------------------- the loader


@dataclass
class Position:
    """One corpus position: its row (labels, strata, weights), its tokens and its shard."""

    row: dict
    tokens: dict[str, np.ndarray]
    shard: str

    @property
    def weight(self) -> float:
        return float(self.row.get("w", 1.0))


class Corpus:
    """A corpus store read back: `iter_rows` / `iter_positions` by source, kind, act, split and tier; `arrays` as the npz layout
    `judge.Positions` reads (flat tokens, per-position labels, groups by start, the held-out flags by run). The human rows of the exam
    fifth are refused (the hash rule of `human_starts.exam_bucket`); the build enforces the ids file as well."""

    def __init__(self, root: Path | str = STORE):
        self.root = Path(root)
        mp = self.root / "manifest.json"
        self.manifest = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}

    def shards(self, src: str | None = None, kind: str | None = None, act: int | None = None) -> list[Path]:
        out = []
        for p in sorted((self.root / "shards").glob("*.jsonl")):
            m = re.fullmatch(r"(?P<src>[a-z]+)-(?P<kind>[a-z]+)-act(?P<act>\d)-\d+", p.stem)
            if not m or (src and m["src"] != src) or (kind and m["kind"] != kind) or (act is not None and int(m["act"]) != act):
                continue
            out.append(p)
        return out

    @staticmethod
    def _keep(row: Mapping[str, Any], split: str, tiers: Sequence[str] | None, stages: Sequence[str] | None) -> bool:
        return ((split == "all" or row["split"] == split) and (tiers is None or row["label"]["tier"] in tiers)
                and (stages is None or row["stage"] in stages))

    def iter_rows(self, src: str | None = None, kind: str | None = None, act: int | None = None, split: str = "all",
                  tiers: Sequence[str] | None = None, stages: Sequence[str] | None = None) -> Iterator[tuple[dict, str, int]]:
        """(row, shard stem, row index in its shard)."""
        from human_starts import exam_bucket  # noqa: PLC0415

        if split not in ("all", "train", "heldout"):
            raise ValueError(f"unknown split {split!r}")
        for p in self.shards(src, kind, act):
            with open(p, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    row = json.loads(line)
                    if row["src"] == "human" and exam_bucket(row["run"]) == 0:
                        raise AssertionError(f"{row['pid']}: a run of the exam fifth is in the corpus")
                    if row["split"] != split_of(row["run"]):
                        raise AssertionError(f"{row['pid']}: its split is not the run's")
                    if self._keep(row, split, tiers, stages):
                        yield row, p.stem, i

    def iter_positions(self, **kw: Any) -> Iterator[Position]:
        cache: dict[str, dict] = {}
        for row, stem, i in self.iter_rows(**kw):
            if stem not in cache:
                cache = {stem: self._tokens(stem)}
            z = cache[stem]
            off = z["offsets"][i]
            n = int(z["lens"][i])
            yield Position(row, {k: z[k][off : off + n] for k in ("types", "ids", "owners", "feats")}, stem)

    def _tokens(self, stem: str) -> dict:
        z = unpack_tokens(self.root / "shards" / (stem + ".npz"))
        z["offsets"] = np.concatenate([[0], np.cumsum(z["lens"].astype(np.int64))])[:-1]
        return z

    def arrays(self, src: str | None = None, kind: str | None = None, act: int | None = None, split: str = "all",
               tiers: Sequence[str] | None = None, label: str = "auto", weight: str = "pool") -> dict[str, np.ndarray]:
        """The positions as arrays in `judge_data`'s layout: `lens`, flat `types`/`ids`/`owners`/`feats`, `labels` [n, 4] (won, end HP over
        max HP when won, potions held when won, L0), `meta` (k, turn, hp, max_hp, root value, best value, best P(death), sims: judge_data.META),
        `group` (the start's index), `start_ids`, `start_heldout` (by run), `start_cat`/`cats`, plus `weight` and `run`.
        `label`: `teacher` (the teacher's own result of the game: all rows), `human` (the human's result of the fight: human rows only) or
        `auto` (the teacher's for self-play rows, the human's for human rows)."""
        if label not in ("auto", "teacher", "human"):
            raise ValueError(f"unknown label {label!r}")
        lens, types, ids, owners, feats, meta, labs, grp, wts, runs, cell_of = [], [], [], [], [], [], [], [], [], [], []
        start_ix: dict[str, int] = {}
        start_info: dict[str, dict] = {}
        cats: dict[str, int] = {}
        for pos in self.iter_positions(src=src, kind=kind, act=act, split=split, tiers=tiers):
            r = pos.row
            lab = label_row(r, label)
            if lab is None:
                continue
            gi = start_ix.setdefault(r["start"], len(start_ix))
            start_info.setdefault(r["start"], {"cat": f"act{r['act']}-{r['kind']}", "run": r["run"]})
            cats.setdefault(start_info[r["start"]]["cat"], len(cats))
            lens.append(len(pos.tokens["types"]))
            for lst, name in ((types, "types"), (ids, "ids"), (owners, "owners"), (feats, "feats")):
                lst.append(pos.tokens[name])
            s = r["label"]
            meta.append([r["k"], r["turn"], r["hp"], r["max_hp"], _nan(s.get("root_value")), _nan(s.get("best_value")), _nan(s.get("p_death")),
                         _nan(s.get("sims"))])
            labs.append(lab)
            grp.append(gi)
            wts.append(row_weight(r, "cell" if weight == "balanced" else weight))
            runs.append(r["run"])
            cell_of.append(f"act{r['act']}-{r['kind']}")
        if weight == "balanced" and wts:  # every (kind, act) cell of the selection carries the same total weight; the mean weight stays 1
            n_cell = Counter(cell_of)
            wts = [w * len(wts) / (len(n_cell) * n_cell[c]) for w, c in zip(wts, cell_of, strict=True)]
        names_ = sorted(start_ix, key=start_ix.get)
        cat_names = sorted(cats)
        return {"lens": np.array(lens, np.uint16), "types": _cat(types, np.uint8), "ids": _cat(ids, np.int32), "owners": _cat(owners, np.uint8),
                "feats": np.concatenate(feats).astype(np.float32) if feats else np.zeros((0, 18), np.float32),
                "meta": np.array(meta, np.float32).reshape(-1, 8), "labels": np.array(labs, np.float32).reshape(-1, 4),
                "group": np.array(grp, np.int32), "weight": np.array(wts, np.float32), "run": np.array(runs),
                "start_ids": np.array(names_), "start_heldout": np.array([split_of(start_info[s]["run"]) == "heldout" for s in names_], dtype=bool),
                "start_cat": np.array([cat_names.index(start_info[s]["cat"]) for s in names_], np.int16), "cats": np.array(cat_names)}


def _nan(x: Any) -> float:
    return float("nan") if x is None else float(x)


def _cat(parts: Sequence[np.ndarray], dtype: Any) -> np.ndarray:
    return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)


def label_row(row: Mapping[str, Any], which: str = "auto") -> list[float] | None:
    """[won, end HP over max HP (0 when lost), potions held at the end (0 when lost), L0] of a row, from the teacher's game or the human's
    fight; None when the row has none of that kind (a self-play row has no human result)."""
    src = row["src"]
    if which == "auto":
        which = "human" if src == "human" else "teacher"
    if which == "human":
        h = row.get("human")
        if h is None:
            return None
        won = bool(h["survived"])
        mx = max(1, int(row.get("door_max_hp") or row["max_hp"]))
        return [float(won), (h["end_hp"] / mx) if won else 0.0, float(h.get("potions_held_end") or 0) if won else 0.0, float(h.get("l0_unit") or 0.0)]
    t = row.get("teacher")
    if t is None:
        return None
    won = bool(t["won"])
    return [float(won), (t["end_hp"] / max(1, t["max_hp"])) if won else 0.0, float(t.get("potions_held") or 0) if won else 0.0, float(t.get("L0") or 0.0)]


# ---------------------------------------------------------------------------------------------------------- coverage (aggregates; no cell under MIN_CELL)


def count_table(rows: Iterable[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], Any]) -> dict[Any, int]:
    out: Counter = Counter()
    for r in rows:
        out[key(r)] += 1
    return dict(out)


def shown(n: int) -> str:
    """A count as published: under MIN_CELL it is not shown."""
    return str(n) if n >= MIN_CELL else f"<{MIN_CELL}"


def coverage_md(rows: Sequence[Mapping[str, Any]], minimums: Mapping[str, int], names_of: Mapping[str, Mapping[int, str]] | None = None) -> str:
    """The coverage table as markdown from position rows (aggregates only, no cell under MIN_CELL; runs per cell are counted too, so a
    cell of many positions from a handful of runs is not shown as strong). `minimums` are the cell minimums the build chose; every
    dimension's table says which cells meet them."""
    names_of = names_of or {}
    rows = list(rows)
    lines: list[str] = []

    def runs_in(sel: Iterable[Mapping[str, Any]]) -> int:
        return len({r["run"] for r in sel})

    srcs = sorted({r["src"] for r in rows})
    for s in srcs:
        sub = [r for r in rows if r["src"] == s]
        lines.append(f"\n## Source: {s} ({len(sub)} positions, {runs_in(sub)} runs)")
        cells = [f"act{a}-{k}" for a in (1, 2) for k in KINDS]
        _emit(lines, sub, "By kind and act", lambda r: f"act{r['act']}-{r['kind']}", cells, minimums.get("cell"))
        _emit(lines, [r for r in sub if r["kind"] == "boss"], "By boss", lambda r: r["enemy"], None, minimums.get("enemy"))
        _emit(lines, [r for r in sub if r["kind"] == "elite"], "By elite", lambda r: r["enemy"], None, minimums.get("enemy"))
        doors = sum(1 for r in sub if r["door"])
        stages = [*STAGES, *(["door"] if any(r["stage"] == "door" for r in sub) else [])]
        _emit(lines, sub, "By turn stage", lambda r: r["stage"], stages, None)
        lines.append(f"\nDoor positions (the fight's first decision; a sampled early turn can be one): {shown(doors)}")
        _emit(lines, sub, "By deck type", lambda r: r["arch"], ARCHETYPES, minimums.get("arch"))
        _emit(lines, sub, "By health band", lambda r: health_band(r["hp_frac"]), BANDS, None)
        _emit(lines, sub, "By potions held", lambda r: potion_state(r["potions"]), ("none", "one", "two+"), None)
        _emit(lines, sub, "By label tier", lambda r: r["label"]["tier"], ("settled", "soft", "played"), None)
        rel: Counter = Counter()
        crd: Counter = Counter()
        relic_runs: dict[int, set] = defaultdict(set)
        card_runs: dict[int, set] = defaultdict(set)
        for r in sub:
            for x in r["relics"]:
                rel[x] += 1
                relic_runs[x].add(r["run"])
            for x in r["deck"]:
                crd[int(x)] += 1
                card_runs[int(x)].add(r["run"])
        for label, cnt, truns, nm in (("relic", rel, relic_runs, names_of.get("relic", {})), ("card", crd, card_runs, names_of.get("card", {}))):
            mn = minimums.get(label) or 0
            ok = [k for k, v in cnt.items() if v >= max(mn, MIN_CELL) and len(truns[k]) >= MIN_CELL]
            lines.append(f"\n**By {label} presence** ({len(cnt)} {label}s appear; {len(ok)} in at least {max(mn, MIN_CELL)} positions and {MIN_CELL} runs;"
                         f" {len(cnt) - len(ok)} under that and not listed)\n")
            lines.append(f"| {label} | positions | runs |")
            lines.append("| --- | --- | --- |")
            for k in sorted(ok, key=lambda x: -cnt[x]):
                lines.append(f"| {nm.get(k, k)} | {cnt[k]} | {len(truns[k])} |")
    return "\n".join(lines)


def _emit(lines: list[str], sub: Sequence[Mapping[str, Any]], title: str, keyfn: Callable[[Mapping[str, Any]], Any], order: Sequence[Any] | None,
          minimum: int | None) -> None:
    groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for r in sub:
        groups[keyfn(r)].append(r)
    keys = list(order) if order is not None else sorted(groups, key=str)
    if not groups:
        return
    lines.append(f"\n**{title}**\n")
    lines.append("| cell | positions | runs |" + (" meets minimum |" if minimum else ""))
    lines.append("| --- | --- | --- |" + (" --- |" if minimum else ""))
    for k in keys:
        g = groups.get(k, [])
        n, nr = len(g), len({r["run"] for r in g})
        if n < MIN_CELL or nr < MIN_CELL:
            lines.append(f"| {k} | <{MIN_CELL} | <{MIN_CELL} |" + (" no |" if minimum else ""))
        else:
            lines.append(f"| {k} | {n} | {nr} |" + (f" {'yes' if n >= minimum else 'no'} |" if minimum else ""))
