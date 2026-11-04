"""The practice mix: which starts the always-on loop's fights are drawn from (D38; the 2026-11-04 memo, sections 3.2 and 5).

The loop's generation used to draw a kind by `starts/weights.json`, then a start uniformly from the nights' AI runs. This module
is what `gym_loop.sample_tasks` now draws with. Pure Python (the shard reader imports numpy when it is called), so the rules are
tested without the emulator.

Sources. `night`: the nights' AI runs (`starts.jsonl`). `human`: fights rebuilt from human runs (`human-starts.jsonl`, built by
`rl/human_starts.py`). `preloss`: the position three decisions before the end of a practice fight the AI lost
(`preloss-starts.jsonl`, written by `gym_loop.gen_game_steps` and merged after the segment). A draw picks the kind by the weights
(unchanged), then the source by a configurable share per kind (`HUMAN_SHARE`; the rest is the AI side, night and preloss starts
together), then a start by the rule below.

Regret weighting (prioritized level replay, Jiang et al. 2021; both terms are needed, either alone did worse than uniform there):
  score    of a start = the mean, over its latest `n_last` plays in the buffer, of how far the played outcome fell short of the
           teacher's root value at the opening: max(0, root - played), both in the search's own currency (end HP plus ten a held
           potion; a death is -100). A start nobody has played takes the mean score of the ones that were played (so it neither
           jumps the queue nor is buried), a pre-loss start the score of the loss it comes from (`score0`).
  rank     the start's rank by score within its kind and side (ties share their mean rank), weight (1 / rank) ** (1 / temperature).
  stale    (now - the segment it was last drawn in) / the sum of those, so starts never drawn come up.
  draw     probability = (1 - staleness) * rank weight + staleness * staleness weight, within the kind and side.

Exclusion. Never drawn: any start whose id, run id or seed is listed in `<loop root>/exam/exclude_runs.json` (the fight exam's
file, read at every draw, no file is no exclusion). Accepted shapes: a list of strings, or a dict whose lists of strings and
whose dict keys are the ids (`{"runs": [...], "starts": [...]}`, `{"runs": {"RUN": {...}}}`, `{"RUN": true}`).

Settings live in `<loop root>/starts/mix.json` when it exists (merged over `DEFAULTS`, read at every segment).
"""

from __future__ import annotations

import json
import math
import os
import random
from bisect import bisect_left
from collections import Counter, defaultdict
from itertools import accumulate
from pathlib import Path

DEATH_VALUE = -100.0  # live_search.DEATH_VALUE: what the search scores a death
POTION_VALUE = 10.0  # live_search's and L0's potion value
KINDS = ("act1-hallway", "act1-elite", "act1-boss", "act2-hallway", "act2-elite", "act2-boss")
SOURCES = ("night", "human", "preloss")
FILES = {"night": "starts", "human": "human-starts", "preloss": "preloss-starts"}  # stem of the set, its index and its lite list
DEFAULTS = {
    # the share of a kind's draws that come from human runs; kinds with no human starts (or no AI starts) take the other side whole
    "human_share": {"act1-hallway": 0.25, "act1-elite": 0.25, "act1-boss": 0.5, "act2-hallway": 0.5, "act2-elite": 0.5, "act2-boss": 0.5},
    "regret": {"temperature": 1.4, "staleness": 0.1, "n_last": 4, "mode": "shortfall"},
    "preloss": {"back": 3, "max": 4000, "min_decisions": 5},  # decisions before the end; most kept; shortest lost fight used
}


def _read_json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(p: Path, obj) -> None:
    p = Path(p)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8", newline="\n")
    os.replace(tmp, p)


def _iter_jsonl(p: Path):
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except ValueError:
                        pass  # a line cut by a stop
    except OSError:
        return


def config(starts_dir: Path) -> dict:
    """DEFAULTS with `starts/mix.json` merged over them, one level down."""
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    user = _read_json(Path(starts_dir) / "mix.json", {}) or {}
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
    return cfg


# ----------------------------------------------------------------------------------------------------------------- ids and exclusion


def run_of(sid: str) -> str:
    """The run a start comes from: `night:SEED|f17` and `human:HASH|f17` give SEED and HASH; a pre-loss start (`preloss:` plus its
    parent's id and the game and decision) gives its parent's run."""
    return str(sid).split("|")[0].rsplit(":", 1)[-1]


ID_KEYS = ("id", "run", "seed", "start", "run_id", "run_hash")


def _collect(x, out: set) -> None:
    """Every id in the exam's file, whatever its shape: strings in lists, dicts in lists (their id-like fields), the keys of a
    dict of runs (`{"RUN": {...}}`), lists inside a dict, and the keys whose value is true."""
    if isinstance(x, str):
        out.add(x)
    elif isinstance(x, list):
        for y in x:
            if isinstance(y, dict):
                out.update(y[k] for k in ID_KEYS if isinstance(y.get(k), str))
            else:
                _collect(y, out)
    elif isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, list):
                _collect(v, out)
            elif isinstance(v, dict):
                if v and all(isinstance(y, (dict, bool, type(None))) for y in v.values()):
                    out.update(str(y) for y in v)  # a dict of runs: the keys are the ids
                else:
                    _collect(v, out)
            elif v is True:
                out.add(str(k))


def read_exclusion(root: Path) -> frozenset[str]:
    """The ids in `<root>/exam/exclude_runs.json`; empty when there is no file or it cannot be read."""
    out: set[str] = set()
    _collect(_read_json(Path(root) / "exam" / "exclude_runs.json", None), out)
    return frozenset(out)


def excluded(s: dict, ex: frozenset[str] | set[str]) -> bool:
    """Whether a start (or its run) is listed: by its id, its seed, its run id, or the run its id names."""
    if not ex:
        return False
    return (s["id"] in ex or s.get("seed") in ex or s.get("run") in ex or run_of(s["id"]) in ex
            or (s.get("parent") is not None and (s["parent"] in ex or run_of(s["parent"]) in ex)))


# ------------------------------------------------------------------------------------------------------------------------ plays


def root_value(visits, q) -> float | None:
    """The search's value at a position: the visit-weighted mean of the root children's values (None for a forced move, where
    nothing was searched)."""
    tot = acc = 0.0
    for v, x in zip(visits, q):
        v, x = float(v), float(x)
        if v > 0 and math.isfinite(x):
            tot += v
            acc += v * x
    return acc / tot if tot > 0 else None


def played_value(meta: dict) -> float | None:
    """What a fight scored in the search's currency: a death is -100, a win its end HP plus ten a held potion (the fight's own
    count when the row has it, else the potions it began with less the ones it used). A capped fight has no outcome."""
    if meta.get("capped") or not meta.get("value_ok", True):
        return None
    if not meta.get("won"):
        return DEATH_VALUE
    held = meta.get("potions_held")
    if held is None:
        held = max(0, int(meta.get("start_potions") or 0) - int(meta.get("potions_used") or 0))
    return float(meta.get("end_hp") or 0) + POTION_VALUE * int(held)


def shard_plays(path: Path) -> list[dict]:
    """The plays of one buffer shard: for each fight, the start, the game, the teacher's root value at the fight's first searched
    decision and the outcome. Reads only the arrays it needs (the shard's tokens stay on disk)."""
    import numpy as np  # noqa: PLC0415

    out = []
    with np.load(path) as z:
        metas = json.loads(str(z["fights_meta"]))
        fight = z["fight"]
        visits, qvals, ok = z["visits"], z["qvals"], z["value_ok"]
    if not len(fight):
        return out
    starts = np.flatnonzero(np.r_[True, fight[1:] != fight[:-1]])
    ends = np.r_[starts[1:], len(fight)]
    for a, b in zip(starts, ends):
        m = metas[int(fight[a])]
        root = None
        for i in range(int(a), min(int(b), int(a) + 3)):
            root = root_value(visits[i], qvals[i])
            if root is not None:
                break
        played = played_value({**m, "value_ok": bool(ok[a])})
        out.append({"start": m["start"], "j": int(m["j"]), "root": root, "played": played})
    return out


def sync_plays(root: Path, log=None) -> list[dict]:
    """The plays log (`starts/plays.jsonl`) brought up to date with the buffer's shards (each shard is read once, in name order,
    which is segment order), and returned whole. Shards the buffer's retention has deleted keep their lines."""
    root = Path(root)
    path = root / "starts" / "plays.jsonl"
    rows = list(_iter_jsonl(path))
    done = {r["shard"] for r in rows if "shard" in r}
    new = 0
    shards = root / "buffer" / "shards"
    if shards.is_dir():
        with open(path, "a", encoding="utf-8") as fh:
            for p in sorted(shards.glob("s*.npz")):
                if ".tmp" in p.name or p.name in done:
                    continue
                try:
                    plays = shard_plays(p)
                except (OSError, ValueError, KeyError, EOFError) as error:  # a shard being written or cut: read again next time
                    if log:
                        log(f"plays: {p.name} unreadable ({type(error).__name__})")
                    continue
                for pl in plays:
                    pl["shard"] = p.name
                    fh.write(json.dumps(pl, separators=(",", ":")) + "\n")
                    rows.append(pl)
                fh.write(json.dumps({"shard": p.name, "done": True}) + "\n")
                rows.append({"shard": p.name, "done": True})
                new += 1
    if len(rows) > 400_000:  # a bounded log: the newest 250,000 lines
        rows = rows[-250_000:]
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")
        os.replace(tmp, path)
    return [r for r in rows if "start" in r]


def scores(plays: list[dict], n_last: int = 4, mode: str = "shortfall") -> dict[str, float]:
    """Each played start's regret score: the mean over its latest `n_last` plays (the log is in play order) of the gap between
    the teacher's root value at the opening and the outcome. `shortfall` counts only the plays that fell short (max(0, root -
    played)); `absolute` counts the gap either way. A play without a root value or an outcome counts for nothing."""
    by: dict[str, list[float]] = defaultdict(list)
    for p in plays:
        if p.get("root") is None or p.get("played") is None:
            continue
        gap = p["root"] - p["played"]
        by[p["start"]].append(abs(gap) if mode == "absolute" else max(0.0, gap))
    return {sid: sum(g[-n_last:]) / len(g[-n_last:]) for sid, g in by.items()}


# ----------------------------------------------------------------------------------------------------------------------- weights


def rank_weights(values: list[float], temperature: float) -> list[float]:
    """(1 / rank) ** (1 / temperature) for each value, the highest value being rank 1; values that tie share their mean rank, so a
    pool whose scores are all equal is uniform. Not normalised."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    out = [0.0] * len(values)
    i = 0
    p = 1.0 / max(temperature, 1e-9)
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + 1 + j + 1) / 2.0
        w = mean_rank ** -p
        for t in range(i, j + 1):
            out[order[t]] = w
        i = j + 1
    return out


def pool_weights(items: list[dict], score: dict[str, float], last_seg: dict[str, int], seg: int, cfg: dict) -> list[float]:
    """The probability of each start of a pool (kind and side): the rank weight of its regret score and the staleness weight,
    mixed `1 - staleness : staleness`. Unplayed starts take the mean of the played ones' scores (a pre-loss start its own
    `score0`)."""
    n = len(items)
    if not n:
        return []
    played = [score[s["id"]] for s in items if s["id"] in score]
    fill = sum(played) / len(played) if played else 0.0
    vals = []
    for s in items:
        v = score.get(s["id"])
        if v is None:
            v = s.get("score0")
        vals.append(fill if v is None else float(v))
    rw = rank_weights(vals, float(cfg["temperature"]))
    stale = [max(0, seg - int(last_seg.get(s["id"], 0))) + 1e-9 for s in items]
    sr, ss = sum(rw), sum(stale)
    rho = min(1.0, max(0.0, float(cfg["staleness"])))
    return [(1 - rho) * a / sr + rho * b / ss for a, b in zip(rw, stale)]


def quartiles(xs: list[float]) -> list[float] | None:
    """[q1, median, q3] by linear interpolation, None for no values."""
    if not xs:
        return None
    xs = sorted(xs)

    def q(p: float) -> float:
        h = (len(xs) - 1) * p
        lo = math.floor(h)
        hi = min(lo + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (h - lo)

    return [round(q(0.25), 2), round(q(0.5), 2), round(q(0.75), 2)]


# -------------------------------------------------------------------------------------------------------------------------- pools


def load_pools(root: Path, ex: frozenset[str] | set[str]) -> tuple[dict[tuple[str, str], list[dict]], Counter]:
    """The draw pools {(kind, side): [start]} with side `human` or `ai` (night and pre-loss starts together), every excluded
    start left out. Each entry: id, source, the set's file, and a pre-loss start's score0. Also the number of starts left out
    by source."""
    sd = Path(root) / "starts"
    pools: dict[tuple[str, str], list[dict]] = defaultdict(list)
    dropped: Counter = Counter()
    for source, stem in FILES.items():
        for s in _read_json(sd / f"{stem}.lite.json", []) or []:
            if excluded(s, ex):
                dropped[source] += 1
                continue
            src = s.get("source") or source
            pools[(s["cat"], "human" if src == "human" else "ai")].append(
                {"id": s["id"], "source": src, "file": str(sd / f"{stem}.jsonl"), "score0": s.get("score0")})
    return pools, dropped


class Sampler:
    """A segment's draws: a kind by `weights`, a side by the kind's human share, a start by the regret rule. The pools' weights are
    worked out once (rank and staleness as they stand when the segment begins); each `draw` is a binary search. A kind with starts
    on one side only draws from that side."""

    def __init__(self, pools: dict, weights: dict[str, float], cfg: dict, score: dict[str, float], last_seg: dict[str, int], seg: int) -> None:
        self.pools, self.cfg, self.score = pools, cfg, score
        self.cats = [c for c in KINDS if weights.get(c, 0) > 0 and (pools.get((c, "ai")) or pools.get((c, "human")))]
        if not self.cats:
            raise RuntimeError("no start mix: run `gym_loop.py starts` and `weights` first")
        self.w = [weights[c] for c in self.cats]
        self.cum = {key: list(accumulate(pool_weights(items, score, last_seg, seg, cfg["regret"]))) for key, items in pools.items() if items}

    def draw(self, rnd: random.Random) -> dict:
        """One draw: the pool entry (id, source, file) with its kind and, when it has one, its score."""
        c = rnd.choices(self.cats, self.w)[0]
        ai, hu = self.pools.get((c, "ai")) or [], self.pools.get((c, "human")) or []
        share = float(self.cfg["human_share"].get(c, 0.0))
        side = "human" if hu and (not ai or rnd.random() < share) else "ai"
        items, cw = (hu, self.cum[(c, "human")]) if side == "human" else (ai, self.cum[(c, "ai")])
        s = items[min(bisect_left(cw, rnd.random() * cw[-1]), len(items) - 1)]
        return {**s, "cat": c, "score": self.score.get(s["id"], s.get("score0"))}


def mix_summary(drawn: list[dict]) -> dict:
    """What a segment drew: the count per kind and source, and the quartiles of the drawn starts' scores."""
    mix: dict[str, Counter] = defaultdict(Counter)
    for d in drawn:
        mix[d["cat"]][d["source"]] += 1
    sc = [d["score"] for d in drawn if d.get("score") is not None]
    return {"by_kind": {c: dict(v) for c, v in sorted(mix.items())}, "scored": len(sc), "drawn": len(drawn), "regret_quartiles": quartiles(sc)}


def mix_line(m: dict) -> str:
    """The segment log's share of the line: `mix act1-boss night 10 human 8; act2-boss human 5 night 2; regret q1/med/q3 ...`."""
    parts = []
    for c, v in m["by_kind"].items():
        parts.append(c + " " + " ".join(f"{src} {v[src]}" for src in SOURCES if v.get(src)))
    q = m["regret_quartiles"]
    reg = "/".join(f"{x:g}" for x in q) if q else "none"
    return f"mix {'; '.join(parts)}; regret q1/med/q3 {reg} ({m['scored']} of {m['drawn']} drawn starts scored)"


# --------------------------------------------------------------------------------------------------------------------- pre-loss


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


def preloss_start(parent: dict, sid: str, j: int, k: int, doc: dict, root: float | None, player: str, made: str) -> dict:
    """The start made from the position at decision `k` of game `j` of start `sid`, a fight the AI lost. It keeps the parent's
    kind, run and split; its score0 is the loss it comes from: the root value of that position against the death."""
    pots = doc.get("potions") or {}
    held = sum(1 for x in (pots.get("slots") or [])[: int(pots.get("max_slots", 0))] if x)
    pl = doc.get("player") or {}
    return {"id": f"preloss:{sid}|j{j}|k{k}", "cat": parent["cat"], "kind": parent["kind"], "act": parent["act"],
            "floor": parent["floor"], "seed": parent["seed"], "run": parent.get("run") or parent["seed"], "ix": parent.get("ix", -1),
            "night": "preloss", "source": "preloss", "parent": sid, "parent_player": player, "k": int(k),
            "node_type": parent["node_type"], "hp": pl.get("hp"), "max_hp": pl.get("max_hp"), "deck_size": len(doc_cards(doc)),
            "potions": held, "relics": len(doc.get("relics") or []), "encounter_id": doc.get("encounter_id"),
            "turn": int(doc.get("turn", 0)), "kind_same": True, "replay_same": None, "split": parent.get("split"),
            "score0": None if root is None else round(max(0.0, float(root) - DEATH_VALUE), 3), "made": made, "doc": doc}


def merge_preloss(root: Path, max_keep: int) -> dict:
    """Move the pre-loss starts the segment's games wrote (`starts/preloss-new/*.json`, one atomic file a start) into
    `preloss-starts.jsonl`, its byte index and its lite list. Keeps the newest `max_keep` (the set is rewritten when it is a
    quarter over). Returns {merged, kept, dropped}."""
    sd = Path(root) / "starts"
    inbox = sd / "preloss-new"
    files = sorted((p for p in inbox.glob("*.json")), key=lambda p: (p.stat().st_mtime_ns, p.name)) if inbox.is_dir() else []
    path, idx_p, lite_p = sd / "preloss-starts.jsonl", sd / "preloss-starts.index.json", sd / "preloss-starts.lite.json"
    index = _read_json(idx_p, {}) or {}
    lite = _read_json(lite_p, []) or []
    merged = 0
    if files:
        with open(path, "ab") as fh:
            for p in files:
                try:
                    rec = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    p.unlink(missing_ok=True)
                    continue
                if rec["id"] not in index:
                    fh.seek(0, 2)
                    index[rec["id"]] = fh.tell()
                    fh.write((json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8"))
                    lite.append({k: v for k, v in rec.items() if k != "doc"})
                    merged += 1
                p.unlink(missing_ok=True)
    dropped = 0
    if len(lite) > int(max_keep * 1.25):
        keep = {r["id"] for r in lite[-int(max_keep):]}
        dropped = len(lite) - len(keep)
        tmp = path.with_name(path.name + ".tmp")
        new_index: dict[str, int] = {}
        with open(path, "rb") as src, open(tmp, "wb") as dst:
            for sid, off in sorted(index.items(), key=lambda kv: kv[1]):
                if sid in keep:
                    src.seek(off)
                    new_index[sid] = dst.tell()
                    dst.write(src.readline())
        os.replace(tmp, path)
        index, lite = new_index, [r for r in lite if r["id"] in keep]
    if merged or dropped:
        _write_json(idx_p, index)
        _write_json(lite_p, lite)
    return {"merged": merged, "kept": len(lite), "dropped": dropped}
