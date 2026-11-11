"""Labels searched to convergence on human positions: the doubling schedule and the stopping rule (D55 addendum 15; 2026-11-09).

The question: instead of a fixed search depth, search a position until the move is stable and clearly the best, so a training set holds
solutions we know are good. Before building such a corpus this measures what convergence costs and how often the teacher
(`a3-v1@h1e4b16`: x1 at hallways, x4 at elites, x16 at bosses) is already converged. "Converged" means stable under the current judge
(the combat prior and the rollouts as they are), not perfect play.

The positions are human A10 players' fights (the fight exam's human stratum, `exam/starts-human.jsonl`). `select` chooses the fights
(fixed seed), `run` plays each with the teacher (player v2, the CPU prior, one process each), keeps turns chosen uniformly among the
fight's decisions that have two or more distinct root moves (a decision with one legal move has nothing to converge), and searches each
turn on a ladder of budgets: the kind's teacher budget times 1/2, 1, 2, 4, 8, 16 (the 1/2 rung only gives the 1x rung something to be
compared with; boss 8, 16, ... 256; elite 2, 4, ... 64; hallway 1/2, 1, ... 16, in the recipe's units where a3-v1 is 1 and the budget
multiplies the simulations of a decision). Every rung is one search from the same position with its own fixed seed; every simulation
starts from a resampling clone, so no rung sees the hidden order or the enemies' rolls. A turn is re-found by replaying the fight's moves
(the fight is a function of its start, its draw and its seeds), checked against the observation's hash at the time it was played.

The rule (`rule_holds`), applied after the fact: converged at a rung when its move equals the move of the rung below and the two most
visited moves' mean values differ by at least `GAP_SE` standard errors (the search's own `value_stderr`: the spread of the per-simulation
returns over the visits, each simulation being one determinization and one playout; the gap's error is the two errors added in
quadrature). The standard errors treat the simulations of a move as independent draws, which tree search does not make them, so they
are the closest estimate the search exposes and not a guarantee.

    python rl/converge.py select --out DIR
    python rl/converge.py run --out DIR [--shard LABEL] [--kinds boss,elite,hallway] [--limit N]
    python rl/converge.py calibrate --out DIR [--n N]    # the stored boss turns timed again on this build: the CPU factor and an identity check
    python rl/converge.py report --out DIR [--md FILE]

Several `run` processes share one folder: a fight is claimed by a file made exclusively, so each process takes the next fight nobody has.
Lab code: nothing here changes the engine, the search, the loop or the night. Rows go to the lane folder (`turns/*.json`), not the repo.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import search_configs  # noqa: E402 -- pure (no native code)

NAMESPACE = "converge-v1"
SELECT_SEED = 20261109
LOOP = Path(os.environ.get("CONVERGE_LOOP", "D:/spireline/referee/gym-loop"))
EXAM = LOOP / "exam"
PLAYER = LOOP / "players" / "v2"
LANE = Path(os.environ.get("CONVERGE_LANE", "D:/spireline/referee/backlog/converged-labels"))
GAME_NS = "x"  # the exam's namespace: the teacher's game seeds are the exam's own
GAME_J = 0  # the draw of every teacher game
TEACHER_CONFIG = "a3-v1@h1e4b16"
KINDS = ("boss", "elite", "hallway")  # the order of play: bosses first, so a partial result is still the boss table
TEACHER_MULT = {"hallway": 1, "elite": 4, "boss": 16}  # the recipe a3-v1@h1e4b16
RELS = (0.5, 1, 2, 4, 8, 16)  # a rung's multiple of the kind's teacher budget; 0.5 is the reference the 1x rung is compared with
REPORT_RELS = (1, 4, 16)  # the tables' columns: the teacher's budget, four times it, sixteen times it
GAP_SE = 2.0  # the rule's gap, in standard errors
SE_MODE = "pooled"  # how the gap's error is estimated (`gap_stats`); "own" is the strict reading
SE_MODES = ("pooled", "own", "paired")
DUEL_SALT = "duel"  # the seeds of the paired comparison's worlds (`duel_seed`)
CPU_CAP_S = 300.0  # a search past this many CPU seconds stops there and is marked capped
MIN_ROOT_MOVES = 2  # a decision with fewer distinct root moves is not a turn
REFEREE = Path(os.environ.get("CONVERGE_REFEREE", "D:/spireline/referee"))
QUIET_STALE_H = 2.0  # another holder's QUIET.lock line under this old is live
QUIET_IGNORE = ("gym-loop", "lab-score", "w3-blocks")  # the lines this job runs beside (the lab's own quiet_ignore, tools/lab/cores.py)
PAUSE_POLL_S = 60.0
LOSS_MIN_VISITS = 20  # the teacher's move needs this many visits at the converged rung for its value to be read
MATERIAL_HP = 5.0  # an overturn that costs this much value (HP) or more at the converged rung is called material
QUOTAS = {"act1-boss": 100, "act2-boss": 100, "act1-elite": 50, "act2-elite": 50, "act1-hallway": 50, "act2-hallway": 50}  # turns
HUMAN_META_POOLS = {  # exam/human-meta.json by_cell pool counts summed per act and kind (the human fights the stratum was drawn from)
    "act1-boss": 7197, "act1-elite": 17957, "act1-hallway": 2107, "act2-boss": 3317, "act2-elite": 7695, "act2-hallway": 1963}


# ---------------------------------------------------------------------------------------------------------- seeds and the sample


def _digest(*parts: Any) -> int:
    text = "|".join(str(p) for p in (NAMESPACE, *parts))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def search_seed(start: str, j: int, k: int, rel: float) -> int:
    """The determinization seed of the rung `rel` at decision k of game j: fixed, and different for every rung (a 30-bit number)."""
    return _digest("search", start, int(j), int(k), f"{rel:g}") % (1 << 30)


def order_key(start_id: str, salt: str = "order") -> str:
    return hashlib.sha256(f"{SELECT_SEED}:{salt}:{start_id}".encode()).hexdigest()


def select_fights(lite: Sequence[dict], quotas: dict[str, int] | None = None) -> list[dict]:
    """The fights to play and how many turns each gives. `lite` is the exam's human stratum (`starts-human.lite.json`).

    A kind-and-act cell wants `quota` turns. Its fights are chosen by hash order, a cell with several bosses stratified over them (an equal
    share each, the remainder to the first by id); a cell with fewer fights than turns takes them all and gives a second turn (a different
    decision of the same fight) to the first fights in hash order until the quota is met. Bosses come first in the result, the acts mixed
    by hash order within a kind."""
    quotas = quotas or QUOTAS
    out: list[dict] = []
    for kc, quota in quotas.items():
        pool = [s for s in lite if s["kind_cell"] == kc]
        by_cell: dict[str, list[dict]] = defaultdict(list)
        for s in pool:
            by_cell[s["cell"]].append(s)
        cells = sorted(by_cell)
        chosen: list[dict] = []
        if len(pool) <= quota:
            chosen = list(pool)
        else:
            base, rem = divmod(quota, len(cells))
            wants = {c: base + (1 if i < rem else 0) for i, c in enumerate(cells)}
            short = 0
            for c in cells:
                items = sorted(by_cell[c], key=lambda s: order_key(s["id"]))
                take = items[: wants[c]]
                short += wants[c] - len(take)
                chosen += take
            if short:  # a cell smaller than its share: the rest of the pool fills it, by hash order
                ids = {s["id"] for s in chosen}
                rest = sorted((s for s in pool if s["id"] not in ids), key=lambda s: order_key(s["id"]))
                chosen += rest[:short]
        chosen.sort(key=lambda s: order_key(s["id"]))
        extra = max(0, quota - len(chosen))
        for i, s in enumerate(chosen):
            out.append({"start": s["id"], "kind": s["kind"], "act": int(s["act"]), "kind_cell": kc, "cell": s["cell"],
                        "enemy": s.get("enemy"), "turns": 2 if i < extra else 1})
    kind_rank = {k: i for i, k in enumerate(KINDS)}
    out.sort(key=lambda f: (kind_rank[f["kind"]], order_key(f["start"], "play")))
    return out


def pick_turns(start_id: str, eligible: Sequence[int], n_turns: int) -> list[int]:
    """`n_turns` distinct decision indices, each uniform among the eligible ones left, by a hash coin of the fight (fixed seed)."""
    pool = list(eligible)
    out: list[int] = []
    for i in range(n_turns):
        if not pool:
            break
        u = _digest("turn", start_id, i) / 2**64
        out.append(pool.pop(int(u * len(pool))))
    return out


# ---------------------------------------------------------------------------------------------------------- the ladder


def multiplier(kind: str, rel: float) -> float:
    """The rung's budget in the recipe's units (a3-v1 is 1)."""
    return TEACHER_MULT[kind] * rel


def config_at(kind: str, rel: float) -> search_configs.SearchConfig:
    """a3-v1 with the kind's budget multiplied: at rel 1 the very simulations the teacher's `a3-v1@h1e4b16` gives that kind of fight."""
    mult = multiplier(kind, rel)
    mult = int(mult) if float(mult).is_integer() else float(mult)
    base = search_configs.get("a3-v1")
    return dataclasses.replace(base, name=f"a3-v1@{kind}x{mult:g}", stakes=(mult, mult, mult))


def ladder(kind: str) -> list[tuple[float, float]]:
    """(rel, budget in recipe units) per rung, lowest first."""
    return [(r, multiplier(kind, r)) for r in RELS]


# ---------------------------------------------------------------------------------------------------------- records (pure)


def _num(x: float | None) -> float | None:
    return None if x is None or math.isnan(x) or math.isinf(x) else round(float(x), 6)


def child_record(c: Any) -> dict:
    """A root move's statistics as JSON (`live_search.ChildStat`): a missing statistic (no visits, one visit) is null."""
    return {"a": int(c.action), "t": int(c.target), "kind": c.kind, "card": c.card_def_id, "merged": [int(m) for m in c.merged_actions],
            "n": int(c.visits), "v": _num(c.value_mean), "se": _num(c.value_stderr), "pd": _num(c.p_death)}


def search_record(res: Any, *, max_sims: int | None, seed: int, cpu_s: float | None, load: float | None, pos_kind: str,
                  rel: float | str, budget: float | None) -> dict:
    """One search's row: its move, its root moves' statistics (most visited first), and what it cost."""
    children = sorted((child_record(c) for c in res.children), key=lambda c: (-c["n"], c["a"], c["t"]))
    capped = max_sims is not None and res.sims + res.select_mapping_failures < max_sims
    return {"rel": rel, "budget": budget, "seed": seed, "max_sims": max_sims, "sims": int(res.sims), "failures": int(res.select_mapping_failures),
            "capped": bool(capped), "cpu_s": None if cpu_s is None else round(cpu_s, 3), "load": load, "pos_kind": pos_kind,
            "n_moves": len(children), "best": child_record(res.best), "children": children}


def same_move(a: dict, b: dict) -> bool:
    """Whether two records' moves are the same thing to do (`live_search._same_move` on records): one action, or copies of it that a
    search merged, at one target."""
    if (a["kind"], a["t"]) != (b["kind"], b["t"]):
        return False
    return a["a"] == b["a"] or b["a"] in a["merged"] or a["a"] in b["merged"]


def top_two(rec: dict) -> tuple[dict, dict] | None:
    """The two most visited root moves of a search record, or None with fewer than two."""
    ch = sorted(rec["children"], key=lambda c: (-c["n"], c["a"], c["t"]))
    return (ch[0], ch[1]) if len(ch) >= 2 else None


def gap_stats(rec: dict, se_mode: str = SE_MODE) -> dict | None:
    """The top two moves' value gap (first minus second), its standard error, and the visits; the gap or its error is None where it
    cannot be estimated (a move with no value, or too few visits for a spread).

    `se_mode` "paired" (2026-11-11, the judge corpus): the gap and its error come from the rung's `duel`, the top two moves each played in the
    same sampled worlds (`duel_pair`), so the spread the unpaired readings below carry (the draw's and the enemies' luck, which both moves
    share) cancels in the difference; a rung with no duel has no gap in this mode. The visits and values beside it stay the search's.

    `se_mode` "pooled": the two moves' returns are taken to spread alike (the spread of a return is mostly the draw's and the enemies'
    luck, which both moves share), the spread is pooled over the two moves' degrees of freedom, and the error is
    `s * sqrt(1/n1 + 1/n2)`; it exists whenever one of the two moves has two visits. "own": each move's own error
    (`value_stderr`), added in quadrature; a move with one visit has none. The search gives the prior's favourite most of its visits,
    so the runner-up often has one or a few, and "own" then cannot say anything about it."""
    pair = top_two(rec)
    if pair is None:
        return None
    a, b = pair
    out = {"n1": a["n"], "n2": b["n"], "v1": a["v"], "v2": b["v"], "gap": None, "se": None, "z": None}
    if se_mode == "paired":
        d = rec.get("duel")
        if d and d.get("n", 0) >= 2 and d.get("mean") is not None and d.get("se") is not None:
            out["gap"], out["se"] = d["mean"], d["se"]
            if d["se"] > 0:
                out["z"] = d["mean"] / d["se"]
        return out
    if a["v"] is None or b["v"] is None:
        return out
    out["gap"] = a["v"] - b["v"]
    if se_mode == "own":
        if a["se"] is not None and b["se"] is not None:
            out["se"] = math.hypot(a["se"], b["se"])
    elif se_mode == "pooled":
        parts = [(m["n"] - 1, m["se"] ** 2 * m["n"]) for m in (a, b) if m["n"] >= 2 and m["se"] is not None]
        dof = sum(d for d, _ in parts)
        if dof > 0:
            var = sum(d * v for d, v in parts) / dof
            out["se"] = math.sqrt(var * (1.0 / a["n"] + 1.0 / b["n"]))
    else:
        raise ValueError(f"unknown se_mode {se_mode!r}")
    if out["se"] is not None and out["se"] > 0:
        out["z"] = out["gap"] / out["se"]
    return out


def rung_of(turn: dict, rel: float | str) -> dict | None:
    for r in turn["rungs"]:
        if r["rel"] == rel:
            return r
    return None


def rule_holds(turn: dict, rel: float, gap_se: float = GAP_SE, se_mode: str = SE_MODE) -> bool | None:
    """The stopping rule at rung `rel`: the move equals the move at `rel / 2` and the top-two gap is at least `gap_se` standard errors.
    None when it cannot be told (a missing or capped rung: a search cut short is not the rung's search)."""
    r, prev = rung_of(turn, rel), rung_of(turn, rel / 2)
    if r is None or prev is None or r["capped"] or prev["capped"]:
        return None
    if not same_move(r["best"], prev["best"]):
        return False
    g = gap_stats(r, se_mode)
    if g is None or g["gap"] is None or g["se"] is None:
        return False
    return g["gap"] >= gap_se * g["se"]


def convergence(turn: dict, gap_se: float = GAP_SE, se_mode: str = SE_MODE) -> dict:
    """The first rung (from the teacher's budget up) at which the rule holds, the move there, and the CPU spent to reach it.
    `rel` is None for a turn the rule never held for; its move is then the top rung's and its CPU the whole ladder's.
    `cpu_to` sums every rung up to and including the converged one, the 1/2 reference too: a doubling run pays for all of them."""
    rels = [r for r in RELS if r >= 1]
    holds = {r: rule_holds(turn, r, gap_se, se_mode) for r in rels}
    conv = next((r for r in rels if holds[r]), None)
    last = conv if conv is not None else max(r["rel"] for r in turn["rungs"] if r["rel"] != "game")
    cpu = sum(r["cpu_s"] or 0.0 for r in turn["rungs"] if r["rel"] != "game" and r["rel"] <= last)
    rec = rung_of(turn, last)
    return {"rel": conv, "holds": holds, "last": last, "cpu_to": cpu, "own_cpu": (rec or {}).get("cpu_s"), "move": (rec or {}).get("best")}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def quantile(xs: Sequence[float], q: float) -> float | None:
    """The q-quantile by linear interpolation (the numpy default); None for no data."""
    s = sorted(xs)
    if not s:
        return None
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _share(k: int, n: int) -> dict:
    return {"k": k, "n": n, "p": (k / n if n else None), "ci": wilson(k, n)}


def _mean(xs: Sequence[float | None]) -> float | None:
    ys = [x for x in xs if x is not None]
    return statistics.fmean(ys) if ys else None


def group_summary(turns: Sequence[dict], gap_se: float = GAP_SE, se_mode: str = SE_MODE) -> dict:
    """The tables' figures for one group of turns (a kind and act, or a kind)."""
    n = len(turns)
    conv = [convergence(t, gap_se, se_mode) for t in turns]

    def agree(r1: float | str, r2: float | str) -> dict:
        pairs = [(rung_of(t, r1), rung_of(t, r2)) for t in turns]
        pairs = [(a, b) for a, b in pairs if a and b and not a["capped"] and not b["capped"]]
        return _share(sum(same_move(a["best"], b["best"]) for a, b in pairs), len(pairs))

    rule_at = {}
    for r in REPORT_RELS:
        known = [v for v in (c["holds"][r] for c in conv) if v is not None]
        rule_at[r] = _share(sum(1 for v in known if v), len(known))
    by = {r: _share(sum(1 for c in conv if c["rel"] is not None and c["rel"] <= r), n) for r in REPORT_RELS}
    cpu_to = [c["cpu_to"] for c in conv]
    cpu_conv = [c["cpu_to"] for c in conv if c["rel"] is not None]
    own_conv = [c["own_cpu"] for c in conv if c["rel"] is not None and c["own_cpu"] is not None]
    # the overturn: the converged move (the top rung's for a turn that never converged) against the teacher's move in the game
    over_game, over_game_conv, over_r1, over_r1_conv, losses, unmeasured = [], [], [], [], [], 0
    for t, c in zip(turns, conv, strict=True):
        game, r1 = rung_of(t, "game"), rung_of(t, 1)
        if c["move"] is None:
            continue
        if game:
            diff = not same_move(game["best"], c["move"])
            over_game.append(diff)
            if c["rel"] is not None:
                over_game_conv.append(diff)
            if diff:
                top = rung_of(t, c["last"])
                mine = next((ch for ch in top["children"] if same_move(ch, game["best"])), None)
                if mine and mine["n"] >= LOSS_MIN_VISITS and mine["v"] is not None and c["move"]["v"] is not None:
                    losses.append(c["move"]["v"] - mine["v"])
                else:
                    unmeasured += 1
        if r1 and not r1["capped"]:
            diff = not same_move(r1["best"], c["move"])
            over_r1.append(diff)
            if c["rel"] is not None:
                over_r1_conv.append(diff)
    loads = [r["load"] for t in turns for r in t["rungs"] if r["rel"] != "game" and r.get("load") is not None]
    kinds: dict[str, int] = defaultdict(int)
    for t in turns:
        kinds[t.get("pos_kind")] += 1
    unconv = []
    for t, c in zip(turns, conv, strict=True):
        if c["rel"] is None:
            g = gap_stats(rung_of(t, c["last"]), se_mode)
            if g and g["gap"] is not None:
                unconv.append(g)
    rung_cpu = {r: [x["cpu_s"] for t in turns for x in [rung_of(t, r)] if x and x["cpu_s"] is not None] for r in RELS}
    return {
        "turns": n, "fights": len({t["start"] for t in turns}),
        "agree_1_4": agree(1, 4), "agree_4_16": agree(4, 16), "agree_1_16": agree(1, 16), "agree_game_1": agree("game", 1),
        "rule_at": rule_at, "converged_by": by, "never": _share(sum(1 for c in conv if c["rel"] is None), n),
        "never_small_gap": _share(sum(1 for g in unconv if abs(g["gap"]) < 1.0), len(unconv)),
        "cpu_to_median": quantile(cpu_to, 0.5), "cpu_to_p90": quantile(cpu_to, 0.9), "cpu_to_mean": _mean(cpu_to),
        "cpu_to_conv_median": quantile(cpu_conv, 0.5), "cpu_to_conv_p90": quantile(cpu_conv, 0.9),
        "own_cpu_conv_median": quantile(own_conv, 0.5), "own_cpu_conv_p90": quantile(own_conv, 0.9),
        "overturn_game": _share(sum(over_game), len(over_game)), "overturn_game_converged": _share(sum(over_game_conv), len(over_game_conv)),
        "overturn_1x": _share(sum(over_r1), len(over_r1)), "overturn_1x_converged": _share(sum(over_r1_conv), len(over_r1_conv)),
        "loss_median": quantile(losses, 0.5), "loss_mean": _mean(losses), "loss_n": len(losses), "loss_unmeasured": unmeasured,
        "overturn_material": _share(sum(1 for x in losses if x >= MATERIAL_HP), len(over_game)),
        "rung_cpu_mean": {str(r): _mean(v) for r, v in rung_cpu.items()}, "rung_cpu_median": {str(r): quantile(v, 0.5) for r, v in rung_cpu.items()},
        "load_median": quantile(loads, 0.5), "load_min": min(loads) if loads else None, "load_max": max(loads) if loads else None,
        "capped_rungs": sum(1 for t in turns for r in t["rungs"] if r["capped"]), "pos_kinds": dict(kinds),
        "eligible_mean": _mean([t["eligible"] for t in turns]), "game_cpu_mean": _mean([t["game"]["cpu_s"] for t in turns if t.get("game")]),
        "replay_bad": sum(1 for t in turns if not t.get("replay_ok", True)),
    }


def group_key(t: dict) -> str:
    return f"act{t['act']}-{t['kind']}"


def corpus_cost(per_turn: dict[str, float | None], weights: dict[str, float], turns: int = 10000) -> dict:
    """Core-hours for `turns` turns drawn with `weights` (group -> share, normalised here): each turn costs its group's figure
    (`per_turn`, CPU seconds). A group with a weight and no figure is left out of the mix."""
    use = {g: w for g, w in weights.items() if w > 0 and per_turn.get(g) is not None}
    total = sum(use.values())
    if total <= 0:
        return {"core_hours": None, "per_turn_s": None}
    per = sum(w / total * per_turn[g] for g, w in use.items())
    return {"core_hours": per * turns / 3600.0, "per_turn_s": per}


def cost_table(groups: dict[str, dict], pools: dict[str, int] | None = None, turns: int = 10000, cpu_scale: dict[str, float] | None = None) -> dict:
    """The 10,000-turn corpus priced at the human positions' own mix of kinds (the pool counts: fights; and the pool counts times the
    mean eligible decisions per fight, the mix by turns) and for each kind alone (its acts at their pool counts). Per mix, the price of the
    doubling run (each turn pays up to its converged rung) and of every turn at a fixed rung (1x, 4x, 16x of the kind's teacher budget).
    A mix over every kind is left out while a kind and act has no turns; a kind alone needs only its own turns. `cpu_scale` multiplies a
    kind's CPU-s ({"boss": 0.4}): the bosses were measured on an older engine build than the other kinds, and a mixed corpus is priced on one."""
    pools = pools or HUMAN_META_POOLS
    scale = cpu_scale or {}

    def sc(g: str) -> float:
        return scale.get(g.split("-")[1], 1.0)

    cpu_to = {g: (None if v["cpu_to_mean"] is None else v["cpu_to_mean"] * sc(g)) for g, v in groups.items()}
    fixed = {r: {g: (None if v["rung_cpu_mean"].get(str(r)) is None else v["rung_cpu_mean"][str(r)] * sc(g)) for g, v in groups.items()}
             for r in REPORT_RELS}
    mixes: dict[str, dict[str, float]] = {
        "own mix (fights)": {g: float(pools[g]) for g in groups if g in pools},
        "own mix (turns)": {g: float(pools[g]) * (groups[g].get("eligible_mean") or 0.0) for g in groups if g in pools},
    }
    for kind, name in (("boss", "bosses alone"), ("elite", "elites alone"), ("hallway", "hallways alone")):
        mixes[name] = {g: float(pools[g]) for g in groups if g in pools and g.endswith("-" + kind)}
    out = {}
    for name, w in mixes.items():
        if name.startswith("own mix") and any(g not in groups for g in pools):  # a mix over every kind needs turns of every kind
            continue
        if not w:
            continue
        row = {"to convergence": corpus_cost(cpu_to, w, turns)}
        for r in REPORT_RELS:
            row[f"fixed {r}x"] = corpus_cost(fixed[r], w, turns)
        out[name] = {"weights": w, **row}
    return out


def summarize(turns: Sequence[dict], pools: dict[str, int] | None = None, gap_se: float = GAP_SE, se_mode: str = SE_MODE,
              cpu_scale: dict[str, float] | None = None) -> dict:
    by_group: dict[str, list[dict]] = defaultdict(list)
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for t in turns:
        by_group[group_key(t)].append(t)
        by_kind[t["kind"]].append(t)
    groups = {g: group_summary(ts, gap_se, se_mode) for g, ts in sorted(by_group.items())}
    kinds = {k: group_summary(ts, gap_se, se_mode) for k, ts in sorted(by_kind.items())}
    return {"gap_se": gap_se, "se_mode": se_mode, "turns": len(turns), "groups": groups, "kinds": kinds,
            "cost": cost_table(groups, pools, cpu_scale=cpu_scale), "cost_as_measured": cost_table(groups, pools) if cpu_scale else None,
            "cpu_scale": cpu_scale or {}}


# ---------------------------------------------------------------------------------------------------------- the machine's load


def system_times() -> tuple[int, int, int] | None:
    """(idle, kernel, user) of the whole machine in 100 ns ticks (kernel includes idle); None off Windows."""
    if sys.platform != "win32":
        return None
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None
    return tuple((f.dwHighDateTime << 32) | f.dwLowDateTime for f in (idle, kernel, user))  # type: ignore[return-value]


def busy_fraction(a: tuple[int, int, int] | None, b: tuple[int, int, int] | None) -> float | None:
    """The share of the machine's logical CPUs that was busy between two `system_times` readings."""
    if a is None or b is None:
        return None
    total = (b[1] - a[1]) + (b[2] - a[2])
    return None if total <= 0 else round(max(0.0, min(1.0, 1.0 - (b[0] - a[0]) / total)), 3)


# ---------------------------------------------------------------------------------------------------------- playing (native)

_LAST: list = [None]


def _install_capture() -> None:
    """Keep the result of the search that just returned, for the recorder (`fight_gym.play_steps` calls its grader right after the
    search with no step between). The search is unchanged; the wrapper only reads."""
    import live_search  # noqa: PLC0415

    if getattr(live_search.search_steps, "_converge", False):
        return
    inner = live_search.search_steps

    def search_steps(handle, **kw):
        res = yield from inner(handle, **kw)
        _LAST[0] = res
        return res

    search_steps._converge = True  # type: ignore[attr-defined]
    live_search.search_steps = search_steps


def obs_hash(world: Any) -> str:
    return hashlib.sha256(bytes(world.obs)).hexdigest()[:16]


def info_of(start: dict, world: Any) -> dict:
    return {"node_type": start["node_type"], "floor": start["floor"], "max_hp": world.max_hp()}


class Recorder:
    """`play_steps`'s grader for the teacher's game: keeps, at every decision, the move played, how many distinct root moves there were,
    the observation's hash and the search's statistics. It never changes a move."""

    def __init__(self, start: dict) -> None:
        self.start = start
        self.rows: list[dict] = []

    def __call__(self, world: Any, k: int, best: Any) -> dict:
        res, _LAST[0] = _LAST[0], None
        if res is None or res.best is not best:
            raise RuntimeError("converge: the search result of this decision was not captured (play_steps changed?)")
        pos = search_configs.position_of(world, info_of(self.start, world))
        self.rows.append({"k": k, "n_moves": len(res.children), "pos_kind": pos.kind, "move": [int(best.action), int(best.target)],
                          "obs": obs_hash(world),
                          "search": search_record(res, max_sims=None, seed=-1, cpu_s=None, load=None, pos_kind=pos.kind, rel="game",
                                                  budget=None)})
        return {"k": k}


def play_teacher(start: dict, extra: dict) -> tuple[dict, list[dict]]:
    """The teacher's game on a start (`a3-v1@h1e4b16`, draw GAME_J): the game's row and the recorder's rows, one per decision."""
    import fight_gym as fg  # noqa: PLC0415
    import live_search  # noqa: PLC0415

    _install_capture()
    rec = Recorder(start)
    cfg = search_configs.get(TEACHER_CONFIG)
    world = fg._fresh_world(start["doc"], start["id"], GAME_J, GAME_NS)  # noqa: SLF001
    c0, w0 = time.process_time(), time.perf_counter()
    try:
        row = live_search._drive(fg.play_steps(world, start, GAME_J, cfg, extra, GAME_NS, grader=rec), extra.get("prior_fn"))  # noqa: SLF001
    finally:
        world.close()
    row = {k: row[k] for k in ("won", "end_hp", "max_hp", "turns", "decisions", "sims", "capped", "moves_sha", "potions_held", "potions_used", "L0") if k in row}
    row["cpu_s"] = round(time.process_time() - c0, 2)
    row["wall_s"] = round(time.perf_counter() - w0, 2)
    return row, rec.rows


def replay_to(start: dict, moves: Sequence[Sequence[int]], k: int) -> Any:
    """The game's world at decision k: the start's fresh world with the first k moves stepped (the game is a function of its seeds)."""
    import fight_gym as fg  # noqa: PLC0415

    world = fg._fresh_world(start["doc"], start["id"], GAME_J, GAME_NS)  # noqa: SLF001
    try:
        step_moves(world, moves[:k])
    except BaseException:
        world.close()
        raise
    return world


def step_moves(world: Any, moves: Sequence[Sequence[int]]) -> None:
    import live_search  # noqa: PLC0415

    for action, target in moves:
        live_search._step(world, int(action), int(target))  # noqa: SLF001


def duel_seed(start_id: str, k: int, rel: float, i: int) -> int:
    """The seed of round i of the paired comparison at decision k, rung `rel`: one world's seed, shared by both moves (1 to 2^30 - 1)."""
    return _digest(DUEL_SALT, start_id, int(k), f"{rel:g}", int(i)) % ((1 << 30) - 1) + 1


def duel_stats(diffs: Sequence[float], deaths: Sequence[float] = ()) -> dict:
    """The paired comparison's figures from the rounds' value differences (first move minus second): the rounds, the mean, its standard
    error (None with fewer than two rounds) and, when given, the mean difference in the chance of dying (first minus second)."""
    n = len(diffs)
    mean = math.fsum(diffs) / n if n else None
    se = None
    if n >= 2:
        se = math.sqrt(math.fsum((d - mean) ** 2 for d in diffs) / (n - 1) / n)
    return {"n": n, "mean": _num(mean), "se": _num(se), "p_death_diff": _num(math.fsum(deaths) / len(deaths)) if deaths else None}


def duel_pair(world: Any, a: dict, b: dict, pairs: int, start_id: str, k: int, rel: float, potion_value: float = 10.0) -> dict:
    """The two moves `a` and `b` (child records) played in the same sampled worlds: each round draws one world (a resampling clone with
    the round's seed, so the hidden order and the enemies' rolls are the same for both) and plays each move there, then the search's own
    rollout policy to the fight's end, with the same random stream for both; the value is the end HP plus `potion_value` a potion held, a
    death -100 (`live_search._outcome`). The rounds' differences a minus b give a mean and a standard error with none of the spread the
    draw adds: the rule's gap when `se_mode` is "paired". A move the world does not offer (a draw-pile choice that no longer shows its
    card) is left out of its round. This is the search's own confirmation round (`live_search._Search._confirm`) for the two moves the
    rule is about, whatever confirmation the config asks for (a3-v1's is off)."""
    import combat_mcts  # noqa: PLC0415
    import live_search  # noqa: PLC0415

    root = combat_mcts.Combat(world.handle)  # a view of the handle: never closed here (the world owns it)
    root.encounter_id = getattr(world, "encounter_id", -1)
    children, _max_hp = live_search._root_moves(root)  # noqa: SLF001

    def find(rec: dict):
        for c in children:
            n = c.node
            if (n.kind, n.target) == (rec["kind"], rec["t"]) and (n.action == rec["a"] or n.action in rec["merged"] or rec["a"] in c.merged):
                return c
        return None

    ca, cb = find(a), find(b)
    if ca is None or cb is None:
        return {**duel_stats([]), "missing": True}
    diffs: list[float] = []
    deaths: list[float] = []
    for i in range(pairs):
        seed = duel_seed(start_id, k, rel, i)
        oa = live_search._playout(root, ca, seed, potion_value)  # noqa: SLF001
        ob = live_search._playout(root, cb, seed, potion_value)  # noqa: SLF001
        if oa is None or ob is None:
            continue
        diffs.append(oa.value - ob.value)
        deaths.append(float(oa.died) - float(ob.died))
    return duel_stats(diffs, deaths)


def search_rung(world: Any, start: dict, kind: str, rel: float, k: int, extra: dict, cap_s: float = CPU_CAP_S,
                config: search_configs.SearchConfig | None = None, pairs: int = 0) -> dict:
    """One rung's search of the position on `world`: the kind's budget times `rel`, the rung's own seed, the CPU prior, the clock the
    process's CPU time (so the cap is CPU seconds). Returns the rung's row. `config` replaces the rung's config (tests). With `pairs` > 0
    the rung's two most visited moves are then compared in that many shared worlds (`duel_pair`) and the row carries the result as
    `duel` (its CPU in `duel_cpu_s`, apart from the search's `cpu_s`); a capped search is not compared."""
    import live_search  # noqa: PLC0415

    cfg = config or config_at(kind, rel)
    pos = search_configs.position_of(world, info_of(start, world))
    kw = cfg.search_kwargs(pos, env_prior=False)
    kw.update(extra)
    kw["time_budget_s"] = cap_s
    seed = search_seed(start["id"], GAME_J, k, rel)
    t0, c0 = system_times(), time.process_time()
    res = live_search._drive(live_search.search_steps(world.handle, seed=seed, clock=time.process_time, **kw), extra.get("prior_fn"))  # noqa: SLF001
    cpu = time.process_time() - c0
    rec = search_record(res, max_sims=kw.get("max_sims"), seed=seed, cpu_s=cpu, load=busy_fraction(t0, system_times()), pos_kind=pos.kind,
                        rel=rel, budget=multiplier(kind, rel))
    if pairs > 0 and not rec["capped"]:
        pair = top_two(rec)
        if pair is not None:
            d0 = time.process_time()
            rec["duel"] = duel_pair(world, pair[0], pair[1], pairs, start["id"], k, rel)
            rec["duel_cpu_s"] = round(time.process_time() - d0, 3)
    return rec


def run_fight(fight: dict, extra: dict, starts_path: str, log: Callable[[str], None] = print,
              pause: Callable[[str], Any] = lambda label: 0.0) -> list[dict]:
    """Play one fight with the teacher and search its chosen turns on the ladder: one dict per turn. `pause(label)` is called before the
    game and before every rung (a wait for the quiet lock, if any)."""
    import fight_gym as fg  # noqa: PLC0415
    import gym_loop as gl  # noqa: PLC0415

    start = gl.load_start(starts_path, fight["start"])
    pause(f"{fight['kind_cell']} game")
    game, dec = play_teacher(start, extra)
    eligible = [d["k"] for d in dec if d["n_moves"] >= MIN_ROOT_MOVES]
    ks = pick_turns(start["id"], eligible, int(fight["turns"]))
    moves = [d["move"] for d in dec]
    turns = []
    for n, k in enumerate(ks):
        world = replay_to(start, moves, k)
        try:
            ok = obs_hash(world) == dec[k]["obs"]
            rungs = [{**dec[k]["search"], "rel": "game", "budget": TEACHER_MULT[fight["kind"]], "pos_kind": dec[k]["pos_kind"],
                      "seed": fg.game_seed(start["id"], GAME_J, f"{GAME_NS}{k}")}]
            if ok:
                for rel, _budget in ladder(fight["kind"]):
                    pause(f"{fight['kind_cell']} rung {rel:g}x")
                    rungs.append(search_rung(world, start, fight["kind"], rel, k, extra))
        finally:
            world.close()
        turns.append({"turn": f"{start['id']}#{n}", "start": start["id"], "n": n, "kind": fight["kind"], "act": fight["act"],
                      "cell": fight["cell"], "enemy": fight.get("enemy"), "k": k, "decisions": len(dec), "eligible": len(eligible),
                      "pos_kind": dec[k]["pos_kind"], "n_moves": dec[k]["n_moves"], "replay_ok": ok, "game": game, "rungs": rungs})
        log(f"{fight['kind_cell']} turn {n} at decision {k} of {len(dec)} (eligible {len(eligible)}) replay_ok={ok} ladder cpu "
            f"{sum(r['cpu_s'] or 0 for r in rungs if r['rel'] != 'game'):.1f}s")
    return turns


# ---------------------------------------------------------------------------------------------------------- the quiet lock


def quiet_reason(now: dt.datetime | None = None, referee: Path | None = None, ignore: Sequence[str] | None = None) -> str | None:
    """Why this job must not compute now, or None: a night block is running, or another holder's QUIET.lock line is live (under two
    hours old; the lines the lab runs beside are not holders, nor the tags in `ignore`: a caller's own tag). The measurement pauses at fight
    and rung boundaries while this says so."""
    referee = referee or REFEREE
    now = now or dt.datetime.now(dt.UTC)
    if (referee / "LAB" / "nightly" / "active").exists():
        return "nightly-active"
    try:
        lines = [x.strip() for x in (referee / "QUIET.lock").read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
    except OSError:
        return None
    for line in lines:
        parts = line.split()
        if parts[0] in QUIET_IGNORE or parts[0] in (ignore or ()):
            continue
        try:
            stamp = parts[1].rstrip("Z")
            t = dt.datetime.fromisoformat(stamp if len(stamp) > 16 else stamp + ":00").replace(tzinfo=dt.UTC)
        except (IndexError, ValueError):
            return "quiet:" + parts[0]  # a line with no readable time is live
        if (now - t).total_seconds() < QUIET_STALE_H * 3600:
            return "quiet:" + parts[0]
    return None


def pause_point(label: str, log: Callable[[str], None] = print, poll_s: float | None = None, out: Path | None = None,
                ignore: Sequence[str] | None = None) -> float:
    """Wait while `quiet_reason` says so; returns the seconds waited. Called between fights and between rungs, never inside a search or
    the teacher's game (a game's clock is the wall clock). Each wait is logged to `out/pauses.log` with its start and end (UTC)."""
    reason = quiet_reason(ignore=ignore)
    if reason is None:
        return 0.0
    t0 = time.time()
    stamp = "%Y-%m-%dT%H:%M:%SZ"
    start = dt.datetime.now(dt.UTC).strftime(stamp)
    log(f"pause ({reason}) at {label}")
    while quiet_reason(ignore=ignore) is not None:
        time.sleep(PAUSE_POLL_S if poll_s is None else poll_s)
    waited = time.time() - t0
    log(f"resume after {waited:.0f}s")
    if out is not None:
        end = dt.datetime.now(dt.UTC).strftime(stamp)
        with open(out / "pauses.log", "a", encoding="utf-8") as fh:
            fh.write(f"{start} {end} {waited:.0f}s {reason} {label}" + chr(10))
    return waited


# ---------------------------------------------------------------------------------------------------------- the lane's files


def fight_file(out: Path, start_id: str) -> Path:
    return out / "turns" / (hashlib.sha1(start_id.encode()).hexdigest()[:16] + ".json")  # noqa: S324 -- a file name


def claim(out: Path, start_id: str) -> bool:
    """Take a fight for this process (a file made exclusively, holding the pid); a claim whose process is gone is taken over."""
    from sts2_gym import procutil  # noqa: PLC0415

    d = out / "claims"
    d.mkdir(parents=True, exist_ok=True)
    p = d / (hashlib.sha1(start_id.encode()).hexdigest()[:16] + ".claim")  # noqa: S324
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(p.read_text().strip())
            except (OSError, ValueError):
                return False
            if procutil.pid_alive(pid):
                return False
            with contextlib.suppress(OSError):
                p.unlink()
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(str(os.getpid()))
        return True
    return False


def write_atomic(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    os.replace(tmp, path)


def read_turns(out: Path) -> list[dict]:
    turns: list[dict] = []
    for p in sorted((out / "turns").glob("*.json")) if (out / "turns").is_dir() else []:
        turns += json.loads(p.read_text(encoding="utf-8"))
    return turns


# ---------------------------------------------------------------------------------------------------------- the engine calibration

CALIBRATION_RELS = (1, 2, 4)  # the rungs timed again (x16, x32, x64 at a boss): the dearer ones cost the same per simulation


def search_alike(old: dict, new: dict) -> bool:
    """Whether two records of one search played alike: the same simulations and the same root moves with the same visits and values."""
    keys = [(c["a"], c["t"], c["kind"], c["n"], c["v"]) for c in old["children"]]
    return old["sims"] == new["sims"] and keys == [(c["a"], c["t"], c["kind"], c["n"], c["v"]) for c in new["children"]]


def calibration_summary(rows: Sequence[dict]) -> dict:
    """What the calibration found: the ratio of the current build's CPU-s to the stored rows' over the same games and searches, and how many
    of them played alike (the engine change was meant to change no game). `game_cpu_ratio` is whole fights, `rung_cpu_ratio` the rungs
    (the cost the corpus is priced on); the busy shares are the machine's during the old and the new timings."""
    rungs = [x for r in rows for x in r["rungs"]]

    def ratio(new: Sequence[float], old: Sequence[float]) -> float | None:
        return sum(new) / sum(old) if old and sum(old) > 0 else None

    return {"turns": len(rows), "games_alike": sum(1 for r in rows if r["game_alike"]), "rungs": len(rungs),
            "rungs_alike": sum(1 for x in rungs if x["alike"]),
            "game_cpu_ratio": ratio([r["game_new_cpu"] for r in rows], [r["game_old_cpu"] for r in rows]),
            "rung_cpu_ratio": ratio([x["new_cpu"] for x in rungs], [x["old_cpu"] for x in rungs]),
            "rung_cpu_ratio_by_rel": {str(rel): ratio([x["new_cpu"] for x in rungs if x["rel"] == rel], [x["old_cpu"] for x in rungs if x["rel"] == rel])
                                      for rel in CALIBRATION_RELS},
            "load_old_median": quantile([x["old_load"] for x in rungs if x["old_load"] is not None], 0.5),
            "load_new_median": quantile([x["new_load"] for x in rungs if x["new_load"] is not None], 0.5)}


def calibrate_turn(t: dict, extra: dict, starts_path: str) -> dict:
    """Play the turn's fight again on this build and search its turn at `CALIBRATION_RELS`: the stored rows' CPU-s beside the new ones, and
    whether the game and each search came out alike."""
    import gym_loop as gl  # noqa: PLC0415

    start = gl.load_start(starts_path, t["start"])
    game, dec = play_teacher(start, extra)
    old = t["game"]
    alike = all(game[key] == old[key] for key in ("moves_sha", "decisions", "sims", "won", "end_hp"))
    world = replay_to(start, [d["move"] for d in dec], t["k"])
    rungs = []
    try:
        for rel in CALIBRATION_RELS:
            was, now = rung_of(t, rel), search_rung(world, start, t["kind"], rel, t["k"], extra)
            rungs.append({"rel": rel, "old_cpu": was["cpu_s"], "new_cpu": now["cpu_s"], "old_load": was["load"], "new_load": now["load"],
                          "alike": search_alike(was, now)})
    finally:
        world.close()
    return {"turn": t["turn"], "act": t["act"], "kind": t["kind"], "k": t["k"], "game_alike": alike, "game_old_cpu": old["cpu_s"],
            "game_new_cpu": game["cpu_s"], "rungs": rungs}


# ---------------------------------------------------------------------------------------------------------- commands


def cmd_select(a: argparse.Namespace) -> int:
    lite = json.loads((EXAM / "starts-human.lite.json").read_text(encoding="utf-8"))
    fights = select_fights([s for s in lite if s.get("stratum", "human") == "human"])
    out = Path(a.out)
    write_atomic(out / "sample.json", {"rule": NAMESPACE, "seed": SELECT_SEED, "quotas": QUOTAS, "fights": fights})
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for f in fights:
        tally[f["kind_cell"]][0] += 1
        tally[f["kind_cell"]][1] += f["turns"]
    for kc, (nf, nt) in sorted(tally.items()):
        print(f"{kc}: {nf} fights, {nt} turns")
    print(f"{len(fights)} fights, {sum(f['turns'] for f in fights)} turns -> {out / 'sample.json'}")
    return 0


def setup_native() -> None:
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from sts2_gym import engine_select, procutil  # noqa: PLC0415

    engine_select.select_engine("core2")
    procutil.lower_priority()


def cmd_run(a: argparse.Namespace) -> int:
    setup_native()
    out = Path(a.out)
    sample = json.loads((out / "sample.json").read_text(encoding="utf-8"))["fights"]
    kinds = set(a.kinds.split(","))
    fights = [f for f in sample if f["kind"] in kinds]
    if a.limit:
        fights = fights[: a.limit]
    import gen_loop  # noqa: PLC0415

    extra = gen_loop.load_agent(PLAYER).search_extra()
    starts_path = str(EXAM / "starts-human.jsonl")
    done = 0
    def say(m: str) -> None:
        print(f"[{a.shard}] {m}", flush=True)

    def pause(label: str) -> float:
        return pause_point(label, say, out=out)

    for i, fight in enumerate(fights):
        if fight_file(out, fight["start"]).is_file():
            continue
        pause("between fights")
        if not claim(out, fight["start"]):
            continue
        t0, c0 = time.perf_counter(), time.process_time()
        try:
            turns = run_fight(fight, extra, starts_path, log=say, pause=pause)
        except Exception:  # noqa: BLE001 -- a fight that fails is recorded and the process goes on
            (out / "errors").mkdir(parents=True, exist_ok=True)
            (out / "errors" / (fight_file(out, fight["start"]).stem + ".txt")).write_text(traceback.format_exc()[-3000:], encoding="utf-8")
            print(f"[{a.shard}] {fight['kind_cell']} fight {i + 1}/{len(fights)} FAILED, see errors/", flush=True)
            continue
        write_atomic(fight_file(out, fight["start"]), turns)
        done += 1
        print(f"[{a.shard}] done {fight['kind_cell']} fight {i + 1}/{len(fights)}: cpu {time.process_time() - c0:.1f}s "
              f"wall {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[{a.shard}] finished, {done} fights", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------------- the report


def cmd_calibrate(a: argparse.Namespace) -> int:
    """Time `--n` stored boss turns again on this build (half of them from each act, by hash order) and write `calibration.json`."""
    setup_native()
    out = Path(a.out)
    boss = sorted((t for t in read_turns(out) if t["kind"] == "boss" and t["n"] == 0 and t["replay_ok"]), key=lambda t: order_key(t["start"], "calibrate"))
    pick = [t for act in (1, 2) for t in [x for x in boss if x["act"] == act][: a.n // 2]]
    import gen_loop  # noqa: PLC0415

    extra = gen_loop.load_agent(PLAYER).search_extra()
    starts_path = str(EXAM / "starts-human.jsonl")
    rows = []
    for t in pick:
        pause_point("calibrate", out=out)
        rows.append(calibrate_turn(t, extra, starts_path))
        print(f"calibrate {len(rows)}/{len(pick)}: game alike {rows[-1]['game_alike']}, rungs alike {[x['alike'] for x in rows[-1]['rungs']]}", flush=True)
    summary = calibration_summary(rows)
    write_atomic(out / "calibration.json", {"summary": summary, "rows": rows})
    print(json.dumps(summary, indent=1))
    return 0


def cpu_scale_of(out: Path) -> dict[str, float] | None:
    """The boss CPU factor from `calibration.json` (current build over the stored rows), or None without one."""
    path = out / "calibration.json"
    if not path.is_file():
        return None
    ratio = json.loads(path.read_text(encoding="utf-8"))["summary"].get("rung_cpu_ratio")
    return {"boss": ratio} if ratio else None


def fmt_share(s: dict) -> str:
    if not s["n"]:
        return "-"
    lo, hi = s["ci"]
    return f"{100 * s['p']:.0f}% ({s['k']}/{s['n']}; {100 * lo:.0f}-{100 * hi:.0f})"


def _f(x: float | None, nd: int = 1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def tables(summary: dict, strict: dict | None = None) -> str:
    """The report's tables as markdown, from `summarize`'s output; `strict` (the same turns under `se_mode="own"`) adds a compact table of
    the figures the error estimate moves."""
    lines: list[str] = []
    order = [g for k in KINDS for g in sorted(summary["groups"]) if g.endswith(k)]
    rows = [(g, summary["groups"][g]) for g in order] + [(f"all {k}", summary["kinds"][k]) for k in KINDS if k in summary["kinds"]]

    def table(title: str, head: list[str], body: Callable[[dict], list[str]]) -> None:
        lines.append(f"\n**{title}**\n")
        lines.append("| group | turns | " + " | ".join(head) + " |")
        lines.append("| --- | --- | " + " | ".join("---" for _ in head) + " |")
        for name, g in rows:
            lines.append(f"| {name} | {g['turns']} | " + " | ".join(body(g)) + " |")

    table("Agreement of the move (share of turns; 95% interval)", ["1x vs 4x", "4x vs 16x", "1x vs 16x", "game vs 1x (seed repeat)"],
          lambda g: [fmt_share(g["agree_1_4"]), fmt_share(g["agree_4_16"]), fmt_share(g["agree_1_16"]), fmt_share(g["agree_game_1"])])
    table("The rule holds at the rung (move equals the half rung's and the gap is at least 2 SE)", ["at 1x", "at 4x", "at 16x"],
          lambda g: [fmt_share(g["rule_at"][r]) for r in REPORT_RELS])
    table("Converged by the rung (first rung the rule held at, up to it)", ["by 1x", "by 4x", "by 16x", "never"],
          lambda g: [fmt_share(g["converged_by"][r]) for r in REPORT_RELS] + [fmt_share(g["never"])])
    table("CPU-s to convergence (all rungs up to the converged one; turns never converged pay the whole ladder)",
          ["median", "p90", "mean", "converged only: median", "p90", "own rung: median", "p90"],
          lambda g: [_f(g["cpu_to_median"]), _f(g["cpu_to_p90"]), _f(g["cpu_to_mean"]), _f(g["cpu_to_conv_median"]),
                     _f(g["cpu_to_conv_p90"]), _f(g["own_cpu_conv_median"]), _f(g["own_cpu_conv_p90"])])
    table("Overturn of the teacher's move by the converged move (the top rung's for a turn never converged)",
          ["vs the game's move: all turns", "converged turns only", "vs the 1x rung's move: all", "converged only", "value lost (HP; median, mean, n measured)",
           "overturns losing 5 HP or more"],
          lambda g: [fmt_share(g["overturn_game"]), fmt_share(g["overturn_game_converged"]), fmt_share(g["overturn_1x"]),
                     fmt_share(g["overturn_1x_converged"]), f"{_f(g['loss_median'], 2)}, {_f(g['loss_mean'], 2)}, {g['loss_n']}",
                     fmt_share(g["overturn_material"])])
    table("CPU-s of one search at each rung (mean over turns), the fight's CPU-s, and the machine's busy share while the rungs ran",
          [f"{r:g}x" for r in RELS] + ["game CPU-s (whole fight)", "load median (min-max)", "capped rungs"],
          lambda g: [_f(g["rung_cpu_mean"][str(r)], 2) for r in RELS] + [_f(g["game_cpu_mean"]),
                     f"{_pct(g['load_median'])} ({_pct(g['load_min'])}-{_pct(g['load_max'])})", str(g["capped_rungs"])])
    if strict is not None:
        s_rows = [(g, strict["groups"][g]) for g in order if g in strict["groups"]] + [(f"all {k}", strict["kinds"][k]) for k in KINDS if k in strict["kinds"]]
        lines.append("\n**The same turns with each move's own error only (a move with one visit has none, so the rule cannot hold on it)**\n")
        lines.append("| group | turns | rule holds at 1x | at 4x | at 16x | never converged | CPU-s to convergence: median | p90 | overturn of the game's move |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for name, g in s_rows:
            cells = [fmt_share(g["rule_at"][r]) for r in REPORT_RELS] + [fmt_share(g["never"]), _f(g["cpu_to_median"]), _f(g["cpu_to_p90"]),
                                                                       fmt_share(g["overturn_game"])]
            lines.append(f"| {name} | {g['turns']} | " + " | ".join(cells) + " |")
        lines.append("\nCorpus of 10,000 turns at convergence under that reading, core-hours: " + "; ".join(
            f"{name} {_f(c['to convergence']['core_hours'], 1)}" for name, c in strict["cost"].items()))
    lines.append("\n**Cost of a 10,000-turn corpus, core-hours (CPU time)**\n")
    lines.append("| mix | at convergence (doubling) | every turn at 1x | at 4x | at 16x |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, c in summary["cost"].items():
        lines.append(f"| {name} | " + " | ".join(_f(c[k]["core_hours"], 1) for k in ("to convergence", "fixed 1x", "fixed 4x", "fixed 16x")) + " |")
    scale = summary.get("cpu_scale") or {}
    if scale and summary.get("cost_as_measured"):
        lines.append("\nCPU-s of " + ", ".join(f"{k} x {v:.3f}" for k, v in scale.items()) + " (the bosses were timed on an older engine build than the other"
                     " kinds; the factor is `calibration.json`'s ratio of the same rungs timed on the current build). As measured, unscaled: "
                     + "; ".join(f"{name} {_f(c['to convergence']['core_hours'], 1)}" for name, c in summary["cost_as_measured"].items()
                                 if not name.startswith("own mix")) + " (at convergence).")
    lines.append("\nBudgets in the recipe's units: 1x / 4x / 16x of the kind's teacher budget are hallway x1 / x4 / x16, elite x4 / x16 / x64, boss x16 / x64 / x256.")
    return "\n".join(lines)


def pools_of_meta(path: Path) -> dict[str, int] | None:
    """The pool counts per act and kind from the exam's `human-meta.json` (None if the file is missing)."""
    if not path.is_file():
        return None
    pools: dict[str, int] = defaultdict(int)
    for c, v in json.loads(path.read_text(encoding="utf-8")).get("by_cell", {}).items():
        pools["-".join(c.split("-")[:2])] += int(v["pool"])
    return dict(pools)


def cmd_report(a: argparse.Namespace) -> int:
    out = Path(a.out)
    turns = read_turns(out)
    pools = pools_of_meta(EXAM / "human-meta.json")
    scale = cpu_scale_of(out)
    summary, strict = summarize(turns, pools, cpu_scale=scale), summarize(turns, pools, se_mode="own", cpu_scale=scale)
    write_atomic(out / "summary.json", summary)
    write_atomic(out / "summary-own-error.json", strict)
    (out / "turns.jsonl").write_text("".join(json.dumps(t) + "\n" for t in turns), encoding="utf-8")
    md = tables(summary, strict)
    if a.md:
        Path(a.md).write_text(md, encoding="utf-8")
    print(md)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--out", default=str(LANE))
    p = sub.add_parser("run")
    p.add_argument("--out", default=str(LANE))
    p.add_argument("--shard", default="0", help="a label for this process's log lines")
    p.add_argument("--kinds", default=",".join(KINDS))
    p.add_argument("--limit", type=int, default=0, help="only the first N fights of the sample (a pilot)")
    p = sub.add_parser("calibrate")
    p.add_argument("--out", default=str(LANE))
    p.add_argument("--n", type=int, default=24, help="boss turns timed again on this build (half from each act)")
    p = sub.add_parser("report")
    p.add_argument("--out", default=str(LANE))
    p.add_argument("--md")
    a = ap.parse_args(argv)
    return {"select": cmd_select, "run": cmd_run, "report": cmd_report, "calibrate": cmd_calibrate}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
