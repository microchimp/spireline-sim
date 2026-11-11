"""The run-level teacher and its first measurement (D56; D57 item 3; docs/rl/RUN_SELF_PLAY.md).

The player's run chooser (`vrun+pi_h`: `argmax_o V(o) + lambda log pi_h(o)`, V the run-value network with FightNet's fight block)
looks one step ahead through a network. The teacher looks further: at a run-level screen it plays each shortlisted option forward on
resampled clones of the run (`hindsight.clone_session`: everything the player has not been shown is redrawn; what is visible, the map,
the boss, the shown intents and the known cards, is kept), the same K draws for every option (common random numbers), with the
player's own run chooser and fight search (its stakes schedule), to the next boss door. The leaf at the door is the fight judge's
estimate of the boss fight from that exact position (P(death)), with a constant value for surviving it (`BEYOND`, the mean final floor
of the player's runs that beat that boss); a death on the way counts its floor. The option's value Q is the mean over the K draws, in
floors (the run-value network's unit), and the teacher takes `argmax_o Q(o) + lambda log pi_h(o)` on the screens the player blends
(the same guard), `argmax_o Q(o)` elsewhere. What it plays is guarded: its pick over the player's only when the measured gain is more
than two standard errors (`guarded`, `guard_se`); its raw pick and every draw are recorded for the student. Why the door and not the act's end: the door-leaf rescoring of the 11-05 labels
(`relabel` below; the memo, section 3) halves the noise of an option difference and saves the boss fight, most of a continuation's CPU.

No foresight: the continuations never read the run's own seed past the decision (the clone is resampled); the teacher's streams come
from (game seed, decision ordinal, k) in a namespace of their own and never touch the run's streams, so a run whose teacher agrees
with the player at every screen is the player's run exactly.

    python rl/run_teacher.py play    --out DIR --arm player|teacher [--seeds-from RUNS.jsonl] [--runs 50] [--procs 4] [--k 4] [--max-options 4]
    python rl/run_teacher.py report  --out DIR
    python rl/run_teacher.py relabel --labels DIR     # stored hindsight labels rescored with the door leaf (no play)

`play` holds its own QUIET.lock line (`run-teacher`), stops on another live line (not the always-on loop's `gym-loop`), on main.lock and
while a night runs, and resumes where it stopped (a run with a row is never played again). Lab code: nothing here changes the engine,
the search, the lab's self-play, the loop or the night. No human record is read anywhere in this file.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

VERSION = 1
NAMESPACE = "run-teacher-v1"  # the continuations' seeds: no game, agent or search seed of any other job comes from it
TAG = "run-teacher"  # this job's QUIET.lock line
AGENT_SEED = 20261023  # gen0.AGENT_SEED: the run's own streams, as the lab's arms play them
ACT_BOSS_FLOORS = (17, 33, 48)
LAST_FLOOR = ACT_BOSS_FLOORS[1]  # the teacher decides screens below the Act 2 boss floor (Act 3's fidelity is barely verified, D38)
# The value of surviving a boss, in final floors: the mean final floor of generation 13's runs that got past that act's boss on
# E14-2026-11-08 (2,600 runs: 1,516 past the Act 1 boss, mean 30.93; 241 past the Act 2 boss, mean 42.30; agent/gen13/candidate-stakes/arms).
BEYOND = {0: 30.93, 1: 42.30}
WON_FLOOR = 53.0  # vocab.WON_FLOOR
DEFAULT_K = 4
DEFAULT_MAX_OPTIONS = 4
CONT_WALL_CAP_S = 900.0  # a continuation still going after this long is stopped (a timeout: the decision falls back to the player)
CONT_DECISION_CAP = 4000
RUN_WALL_CAP_S = 4 * 3600.0
# decision type -> kind (rl/run_train/features.Decision.type)
KIND_OF_TYPE = {"map": "route", "card_reward": "card", "rest": "rest", "shop": "shop", "neow": "ancient", "event": "event",
                "rewards": "reward", "card_select": "select", "bundle": "bundle"}
TEACHER_KINDS = tuple(KIND_OF_TYPE.values())
REF = Path(os.environ.get("RUN_TEACHER_REFEREE", "D:/spireline/referee"))
QUIET = REF / "QUIET.lock"
MAIN_LOCK = REF / "backlog" / "main.lock"
ACTIVE = REF / "LAB" / "nightly" / "active"
DEFAULT_AGENT = "D:/spireline/referee/agent/gen13/candidate-stakes"
DEFAULT_JUDGE = "D:/spireline/referee/agent/judge/v0/judge.pt"
DEFAULT_SEEDS_FROM = "D:/spireline/referee/agent/gen13/candidate-stakes/arms/E14-2026-11-08-a3-v1/runs.jsonl"
NEG = -1e9  # the value of an option the teacher did not play (never taken)
# The guard on what the teacher plays (`guarded`): it takes its own pick over the player's only when the measured gain exceeds Z standard
# errors. A decision's own variance of the paired difference (K - 1 degrees of freedom) is pooled with a prior one of NU0 degrees of freedom:
# the paired sd of one draw's option difference under the door leaf, by kind, from the 11-05 labels rescored (`relabel`; card 3.41, rest
# 2.76, route 4.83, shop 2.94, all 3.11 floors).
Z = 2.0
NU0 = 4.0
PAIRED_SD_PRIOR = {"card": 3.41, "rest": 2.76, "route": 4.83, "shop": 2.94}
PAIRED_SD_DEFAULT = 3.11


# ---------------------------------------------------------------------------------------------------------- seeds


def _digest(*parts: Any) -> int:
    text = "|".join(str(p) for p in (NAMESPACE, *parts))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def resample_seed(seed: str, ordinal: int, k: int) -> int:
    """Draw k's resampled clone of decision `ordinal` of run `seed`: the same for every option (common random numbers)."""
    return int(_digest("resample", seed, int(ordinal), int(k)) % (2**31 - 1))


def stream_key(seed: str, ordinal: int, k: int, stream: int) -> list[int]:
    """The entropy of draw k's streams: 0 the fights' search seeds, 1 the run chooser's (never drawn from at argmax)."""
    return [_digest("stream", seed, int(ordinal), int(k), int(stream)), int(ordinal), int(k), int(stream)]


# ---------------------------------------------------------------------------------------------------------- the leaf and the pick


def act_of_floor(floor: int) -> int:
    """The 0-based act of a floor: up to 17 Act 1, 18 to 33 Act 2, later Act 3 (`macro_recipe.act_of_floor` less one)."""
    return 0 if floor <= ACT_BOSS_FLOORS[0] else 1 if floor <= ACT_BOSS_FLOORS[1] else 2


def door_return(end: str, floor: int, p_death: float | None, beyond: dict[int, float] = BEYOND, won_floor: float = WON_FLOOR) -> float | None:
    """G of one continuation, in final floors: a death on the way is its floor; at the boss door, P(death) x the boss's floor plus
    P(survive) x the value of surviving that boss (`beyond`); a won run its floor. None for any other end (timeout, step cap, a door
    with no estimate, a door past the acts `beyond` knows): such a draw has no return and the decision falls back to the player."""
    if end == "death":
        return float(floor)
    if end == "win":
        return float(won_floor)
    if end == "door":
        act = act_of_floor(int(floor))
        if p_death is None or act not in beyond or not (0.0 <= float(p_death) <= 1.0):
            return None
        p = float(p_death)
        return p * float(floor) + (1.0 - p) * float(beyond[act])
    return None


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def sample_var(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def teacher_index(q: dict[int, float], n: int, player: int, *, screen: str | None = None, lam: float = 0.0,
                  logp: Sequence[float] | None = None, logits: Sequence[float] | None = None, groups: Sequence[int] | None = None) -> int:
    """The teacher's option among `n`: `q` holds the value of each option it played (the others are never taken). On a blended
    screen (`screen` card/ancient/rest with lam > 0) it is the player's own blend with Q in place of V (`imitation_chooser.blend_pick`
    and `blend_rest`, the guard included, the prior read over all the screen's options); elsewhere the highest Q, ties to the player's
    own pick, then the first."""
    if not q:
        return player
    if screen in ("card", "ancient", "rest") and lam > 0 and not math.isinf(lam) and logp is not None:
        from imitation_chooser import blend_pick, blend_rest  # noqa: PLC0415

        values = [float(q[i]) if i in q else NEG for i in range(n)]
        if screen == "rest" and logits is not None and groups is not None:
            pick = blend_rest(values, logits, groups, lam).index
        else:
            pick = blend_pick(values, logp, lam).index
        return pick if pick in q else player
    if screen in ("card", "ancient", "rest") and math.isinf(lam):
        return player  # the prior alone decides: Q cannot move it
    top = max(q.values())
    tied = [i for i in sorted(q) if q[i] >= top - 1e-9]
    return player if player in tied else tied[0]


def paired_se(a: Sequence[float], b: Sequence[float]) -> float | None:
    """The standard error of mean(a - b) over paired draws; None with fewer than two."""
    d = [x - y for x, y in zip(a, b, strict=True)]
    if len(d) < 2:
        return None
    return math.sqrt(sample_var(d) / len(d))


def guard_se(diffs: Sequence[float], kind: str | None, nu0: float = NU0) -> float:
    """The standard error of a mean paired difference over K draws, its variance pooled with the kind's prior (`PAIRED_SD_PRIOR`) of
    `nu0` degrees of freedom, so that two draws that happen to agree do not make a tiny error."""
    k = len(diffs)
    prior = PAIRED_SD_PRIOR.get(kind or "", PAIRED_SD_DEFAULT) ** 2
    own = sample_var(diffs) if k >= 2 else 0.0
    var = ((k - 1) * own + nu0 * prior) / ((k - 1) + nu0) if k >= 1 else prior
    return math.sqrt(var / max(1, k))


def guarded(raw: int, player: int, gain: float, se: float, z: float = Z) -> int:
    """What the teacher plays: its own pick when the measured gain over the player's pick exceeds z standard errors, else the player's."""
    return raw if raw != player and gain > z * se else player


def split_halves(g: dict[int, list[float]]) -> tuple[dict[int, float], dict[int, float]] | None:
    """The option values from the first and the second half of the draws (K even, at least 2), for the teacher's self-agreement."""
    k = min((len(v) for v in g.values()), default=0)
    if k < 2:
        return None
    h = k // 2
    return ({o: mean(v[:h]) for o, v in g.items()}, {o: mean(v[h : 2 * h]) for o, v in g.items()})


# ---------------------------------------------------------------------------------------------------------- the statistics


def pooled(vs: Iterable[tuple[float, int]]) -> float | None:
    num = den = 0.0
    for v, df in vs:
        if df > 0 and not math.isnan(v):
            num += v * df
            den += df
    return num / den if den else None


def advantage_variances(g: Sequence[Sequence[float]]) -> list[tuple[float, int]]:
    """Per option, the variance over draws of its draw less the options' mean on the same draw, with its degrees of freedom."""
    n = min((len(r) for r in g), default=0)
    if n < 2 or len(g) < 2:
        return []
    mk = [mean([g[o][k] for o in range(len(g))]) for k in range(n)]
    return [(sample_var([g[o][k] - mk[k] for k in range(n)]), n - 1) for o in range(len(g))]


def noise_and_spread(gs: Sequence[Sequence[Sequence[float]]]) -> dict[str, float | None]:
    """Over decisions (each a matrix G[option][draw]): the pooled noise sd of one draw's centred option value, the true spread of the
    centred option values between options (their observed variance less the noise / K), the signal-to-noise of one decision's values
    at its K, and the shrinkage weight w = spread^2 / (spread^2 + noise^2 / K) an empirical-Bayes teacher puts on its own values."""
    gs = [g for g in gs if len(g) >= 2 and min(len(r) for r in g) >= 2]
    if not gs:
        return {"decisions": 0, "noise_sd": None, "spread_sd": None, "snr": None, "w": None, "k": None}
    va = pooled(x for g in gs for x in advantage_variances(g))
    k = mean([min(len(r) for r in g) for g in gs])
    obs = [m - mean([mean(r) for r in g]) for g in gs for m in (mean(r) for r in g)]
    tau2 = max(0.0, sample_var(obs) - (va or 0.0) / k) if len(obs) > 2 else 0.0
    noise2 = (va or 0.0) / k
    return {"decisions": len(gs), "noise_sd": None if va is None else round(math.sqrt(va), 3), "spread_sd": round(math.sqrt(tau2), 3),
            "snr": None if not noise2 else round(math.sqrt(tau2 / noise2), 3), "w": None if not (tau2 + noise2) else round(tau2 / (tau2 + noise2), 3),
            "k": round(k, 2)}


def lambda_fit(decisions: Sequence[Sequence[tuple[float, float, float]]]) -> dict[str, float | int | None]:
    """Lambda measured: how many floors the teacher's play-outs credit to one unit of a prior's log-probability beyond what V sees.
    Each decision is a list of (Q, V, log p) over the options the teacher played; within a decision each is centred, and
    `lambda = sum (A_Q - A_V) l / sum l^2` (no intercept: the centring removes it), with a standard error clustered by decision
    (the sandwich form, so one decision's options are not counted as independent)."""
    num = den = 0.0
    cen: list[tuple[list[float], list[float]]] = []
    for d in decisions:
        if len(d) < 2:
            continue
        q, v, lp = (list(x) for x in zip(*d, strict=True))
        mq, mv, ml = mean(q), mean(v), mean(lp)
        y = [(a - mq) - (b - mv) for a, b in zip(q, v, strict=True)]
        ll = [x - ml for x in lp]
        cen.append((y, ll))
        num += sum(a * b for a, b in zip(y, ll, strict=True))
        den += sum(b * b for b in ll)
    if not cen or den <= 0:
        return {"decisions": len(cen), "lambda": None, "se": None}
    lam = num / den
    meat = sum(sum(b * (a - lam * b) for a, b in zip(y, ll, strict=True)) ** 2 for y, ll in cen)
    return {"decisions": len(cen), "lambda": round(lam, 4), "se": round(math.sqrt(meat) / den, 4)}


# ---------------------------------------------------------------------------------------------------------- relabel (no play)


def relabel(recs: Sequence[dict], beyond: dict[int, float] | None = None) -> dict:
    """Stored hindsight labels (`rl/hindsight.py`, continuations to the act's end with the judge's estimate recorded at each boss
    door) rescored with the door leaf: each continuation that reached the door is valued by `door_return` at the judge's P(death)
    there instead of the boss fight it went on to play. `beyond` defaults to the labels' own survivors' mean head value per act.
    Returns the noise and spread under both leaves, overall and by kind, the judge's calibration at the door, and the split-half
    agreement of the best option (draws 0-1 against 2-3)."""
    recs = [r for r in recs if not r.get("refused") and r.get("continuations")]
    if beyond is None:
        acc: dict[int, list[float]] = defaultdict(list)
        for r in recs:
            for c in r["continuations"]:
                if c.get("end") == "next_act" and c.get("G") is not None:
                    acc[int(r["act"])].append(float(c["G"]))
        beyond = {a: mean(v) for a, v in acc.items() if v}

    def g_of(r: dict, mode: str) -> list[list[float]] | None:
        cell: dict[tuple[int, int], float] = {}
        for c in r["continuations"]:
            g = c.get("G")
            if mode == "door" and c.get("door") and (c["door"].get("judge") or {}).get("p_death") is not None:
                g = door_return("door", int(c["door"]["floor"]), float(c["door"]["judge"]["p_death"]), beyond)
            if g is None:
                return None
            cell[(int(c["o"]), int(c["k"]))] = float(g)
        try:
            return [[cell[(o, k)] for k in range(int(r["k"]))] for o in r["played"]]
        except KeyError:
            return None

    out: dict[str, Any] = {"labels": len(recs), "beyond": {str(a): round(v, 3) for a, v in sorted(beyond.items())}}
    cal = [(float(c["door"]["judge"]["p_death"]), int(c.get("end") == "death" and int(c.get("floor", -1)) == int(c["door"]["floor"])))
           for r in recs for c in r["continuations"] if c.get("door") and (c["door"].get("judge") or {}).get("p_death") is not None]
    if cal:
        base = mean([y for _, y in cal])
        brier = mean([(p - y) ** 2 for p, y in cal])
        out["door_judge"] = {"n": len(cal), "p_death_mean": round(mean([p for p, _ in cal]), 4), "boss_death_rate": round(base, 4),
                             "brier": round(brier, 4), "brier_base": round(base * (1 - base), 4),
                             "skill": round(1 - brier / (base * (1 - base)), 4) if 0 < base < 1 else None}
    for mode in ("played", "door"):
        block: dict[str, Any] = {}
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            groups["all"].append(r)
            groups[f"{r['kind']}/a{int(r['act']) + 1}"].append(r)
        for name, rs in sorted(groups.items()):
            gs = [g for g in (g_of(r, mode) for r in rs) if g is not None]
            row = noise_and_spread(gs)
            agree = n = 0
            for g in gs:
                h = split_halves(dict(enumerate(g)))
                if h is None:
                    continue
                a, b = h
                n += 1
                agree += int(max(a, key=a.get) == max(b, key=b.get))
            row["split_half_agree"] = None if not n else round(agree / n, 3)
            row["chance"] = round(mean([1 / len(g) for g in gs]), 3) if gs else None
            if row["decisions"] >= 20:
                block[name] = row
        out[mode] = block
    return out


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            with contextlib.suppress(ValueError):
                r = json.loads(line)  # a line cut by a stop does not parse and is dropped
                if isinstance(r, dict):
                    rows.append(r)
    return rows


# ---------------------------------------------------------------------------------------------------------- the play (native)


def run_to_door(s: Any, player: Any, rng: Any, explore: Any, deadline: float) -> dict:
    """Play `s` on under the fixed player until it enters a boss fight (the door: the fight's first decision, not played), dies or
    wins. Returns {end, floor, door, fights, moves, decisions}; `end` is "door", "death", "win", "timeout" or "step_cap"."""
    import hindsight  # noqa: PLC0415
    from run_train import engine  # noqa: PLC0415

    out: dict[str, Any] = {"end": "?", "floor": 0, "door": None, "fights": 0, "moves": 0, "decisions": 0}
    try:
        while True:
            if time.perf_counter() > deadline:
                raise hindsight.Timeout("the continuation outlived its wall cap")
            if out["decisions"] + out["moves"] >= CONT_DECISION_CAP:
                out["end"] = "step_cap"
                break
            d = s.decision()
            if d is None and s.phase != engine.rc.PHASE_COMBAT:
                out["end"] = "win" if s.won else "death"
                break
            if d is not None:
                pick = player.policy.pick(s.view(), d, explore)
                s.apply(d.options[pick.index])
                out["decisions"] += 1
                continue
            if s.info["node_type"] == engine.vocab.NODE_BOSS:
                out["door"] = hindsight.door_of(s, player)
                out["end"] = "door"
                break
            out["fights"] += 1
            out["moves"] += player.play_fight(s, rng, deadline)
    except hindsight.Timeout:
        out["end"] = "timeout"
    out["floor"] = int(s.info["floor"])
    return out


def continuation(base: Any, option: Any, player: Any, *, seed: str, ordinal: int, k: int) -> dict:
    """Draw k of one option: fork `base` on the decision's resample seed for k, apply the option, play on to the door."""
    import hindsight  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    t0, c0 = time.perf_counter(), time.process_time()
    rng = np.random.default_rng(stream_key(seed, ordinal, k, 0))
    explore = np.random.default_rng(stream_key(seed, ordinal, k, 1))
    c = hindsight.clone_session(base, resample_seed(seed, ordinal, k))
    try:
        c.apply(option)
        out = run_to_door(c, player, rng, explore, time.perf_counter() + CONT_WALL_CAP_S)
    finally:
        c.close()
    p = ((out["door"] or {}).get("judge") or {}).get("p_death")
    g = door_return(out["end"], out["floor"], p)
    return {"k": k, "G": None if g is None else round(g, 4), "end": out["end"], "floor": out["floor"], "p_death": p,
            "door_hp": (out["door"] or {}).get("hp"), "fights": out["fights"], "cpu": round(time.process_time() - c0, 3),
            "seconds": round(time.perf_counter() - t0, 3)}


def shortlist(options: Sequence[Any], scores: Sequence[float | None], limit: int, keep: int) -> list[int]:
    """The options the teacher plays: all playable ones when there are `limit` or fewer, else the player's own pick, the unchanged
    options (skip, leave, proceed) and the player's best-scored others, up to `limit` (the unchanged ones may take it past)."""
    import hindsight  # noqa: PLC0415

    allowed = [i for i, o in enumerate(options) if o.kind != "unknown"]
    if len(allowed) <= limit:
        return allowed
    chosen = {keep} | {i for i in allowed if options[i].kind in hindsight.UNCHANGED_KINDS}
    rest = sorted((i for i in allowed if i not in chosen), key=lambda i: -(scores[i] if scores[i] is not None else -1e9))
    for i in rest:
        if len(chosen) >= limit:
            break
        chosen.add(i)
    return sorted(chosen)


def screen_inputs(policy: Any, view: Any, decision: Any) -> dict[str, Any]:
    """The player's own reading of a screen: its lambda there, the screen the blend decides (None: V alone), the prior's
    log-probabilities and logits over the options, the rest groups, and the player's score of each option (V + lambda log pi_h on a
    blended screen, V elsewhere; None for an option V cannot score)."""
    import imitation_chooser as ic  # noqa: PLC0415
    from human_prior_infer import log_softmax, option_rows  # noqa: PLC0415
    from run_train import models  # noqa: PLC0415
    from run_train import policy as run_policy  # noqa: PLC0415

    view = decision.view or view
    lam = policy.lam_at(view.floor) if hasattr(policy, "lam_at") else 0.0
    screen = ic.screen_of(decision) if lam > 0 else None
    options = list(decision.options)
    out: dict[str, Any] = {"lam": lam, "screen": screen, "logp": None, "logits": None, "groups": None, "scores": [None] * len(options),
                           "v": [None] * len(options)}
    idx = [i for i, o in enumerate(options) if o.scorable]
    if idx:
        probs = run_policy.floor_probs(policy.nets, view, [options[i] for i in idx])
        for i, v in zip(idx, models.expected_floors(probs, policy.W).tolist(), strict=True):
            out["v"][i] = float(v)
            out["scores"][i] = float(v)
    if screen is not None:
        rows = option_rows(policy.nets, view, options)
        logits = [float(x) for x in policy.prior.row_logits(rows)]
        logp = [float(x) for x in log_softmax(logits)]
        out.update(logp=logp, logits=logits, groups=[ic.rest_group(o) for o in options] if screen == "rest" else None)
        if not math.isinf(lam):
            out["scores"] = [None if v is None else v + lam * lp for v, lp in zip(out["v"], logp, strict=True)]
    return out


def decide(s: Any, decision: Any, view: Any, pick: Any, player: Any, *, seed: str, ordinal: int, k: int, max_options: int) -> dict:
    """Play the screen's shortlisted options K times each to the door and return the decision's record: the teacher's index,
    the player's, each draw's G and what the screen's reading was."""
    t0, c0 = time.perf_counter(), time.process_time()
    view = decision.view or view
    rd = screen_inputs(player.policy, view, decision)
    chosen = shortlist(decision.options, rd["scores"], max_options, pick.index)
    draws: dict[int, list[dict]] = {i: [] for i in chosen}
    for kk in range(k):  # draw-major: the same draw for every option before the next
        for oi in chosen:
            draws[oi].append(continuation(s, decision.options[oi], player, seed=seed, ordinal=ordinal, k=kk))
    g = {o: [d["G"] for d in rows] for o, rows in draws.items()}
    ok = all(x is not None for v in g.values() for x in v)
    rec: dict[str, Any] = {"v": VERSION, "seed": seed, "ordinal": ordinal, "type": decision.type, "kind": KIND_OF_TYPE.get(decision.type),
                           "act": int(view.act_index), "floor": int(view.floor), "hp": int(view.hp), "max_hp": int(view.max_hp),
                           "n_options": len(decision.options), "played": chosen, "player": int(pick.index), "k": k,
                           "screen": rd["screen"], "lam": rd["lam"], "v_values": rd["v"], "logp": rd["logp"],
                           "g": {str(o): v for o, v in g.items()},
                           "ends": dict(Counter(d["end"] for rows in draws.values() for d in rows)),
                           "p_death": {str(o): [d["p_death"] for d in rows] for o, rows in draws.items()},
                           "door_hp": {str(o): [d["door_hp"] for d in rows] for o, rows in draws.items()},
                           "options": [o.kind for o in decision.options]}
    if not ok:
        rec.update(teacher=int(pick.index), refused="no_return")
    elif int(pick.index) not in g:
        rec.update(teacher=int(pick.index), refused="player_unplayed")
    else:
        q = {o: mean(v) for o, v in g.items()}
        kw = {"screen": rd["screen"], "lam": rd["lam"], "logp": rd["logp"], "logits": rd["logits"], "groups": rd["groups"]}
        p = int(pick.index)
        kind = KIND_OF_TYPE.get(decision.type)

        def score_gain(t: int, vals: dict[int, float]) -> float:
            """The teacher's score of its pick over the player's: Q's difference, plus the prior's on a blended screen."""
            extra = 0.0
            if rd["screen"] is not None and rd["logp"] is not None and not math.isinf(rd["lam"]):
                extra = rd["lam"] * (rd["logp"][t] - rd["logp"][p])
            return vals[t] - vals[p] + extra

        raw = teacher_index(q, len(decision.options), p, **kw)
        gain = score_gain(raw, q) if raw != p else 0.0
        se = guard_se([a - b for a, b in zip(g[raw], g[p], strict=True)], kind) if raw != p else None
        played = guarded(raw, p, gain, se if se is not None else math.inf)
        rec.update(teacher=played, teacher_raw=raw, q={str(o): round(x, 4) for o, x in q.items()})
        rec["gain"] = round(gain, 4)
        rec["gain_se"] = None if se is None else round(se, 4)
        own = paired_se(g[raw], g[p]) if raw != p else None
        rec["gain_se_own"] = None if own is None else round(own, 4)
        halves = split_halves(g)
        if halves is not None:
            hp = []
            for h, sl in zip(halves, (slice(0, k // 2), slice(k // 2, 2 * (k // 2))), strict=True):
                r_h = teacher_index(h, len(decision.options), p, **kw)
                if r_h == p:
                    hp.append((r_h, p))
                    continue
                se_h = guard_se([a - b for a, b in zip(g[r_h][sl], g[p][sl], strict=True)], kind)
                hp.append((r_h, guarded(r_h, p, score_gain(r_h, h), se_h)))
            rec["half_picks"] = [x[0] for x in hp]
            rec["half_played"] = [x[1] for x in hp]
    rec["cpu"] = round(time.process_time() - c0, 3)
    rec["seconds"] = round(time.perf_counter() - t0, 3)
    return rec


def eligible(kind: str | None, view: Any, decision: Any, kinds: Sequence[str]) -> bool:
    """Whether the teacher decides this screen: a kind it covers, below the Act 2 boss floor, at least two playable options."""
    if kind is None or kind not in kinds or int(view.floor) >= LAST_FLOOR:
        return False
    return sum(1 for o in decision.options if o.kind != "unknown") >= 2


_W: dict[str, Any] = {}


def _init_worker(out: str, agent_dir: str, judge: str, k: int, max_options: int, kinds: list[str], lib: str) -> None:
    os.environ["STS2_LIB_PATH"] = lib or str(ROOT / "out")
    os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    from sts2_gym import procutil  # noqa: PLC0415

    procutil.lower_priority()  # BelowNormal (Windows priority class)
    import hindsight  # noqa: PLC0415

    player = hindsight.build_player(agent_dir, config="a3-v1", judge_path=judge or None, own_schedule=True)
    with contextlib.suppress(Exception):
        import torch  # noqa: PLC0415

        torch.set_num_threads(1)
    _W.update(out=Path(out), player=player, k=int(k), max_options=int(max_options), kinds=tuple(kinds))


def play_run(task: tuple) -> dict:
    """One whole run of `task` = ("rt", arm, seed, ix): the player's own run (`gen0.play_emu_run`'s loop and streams, argmax), or the
    same run with the teacher deciding every eligible screen. Each teacher decision is appended to `decisions.d/<seed>.jsonl`."""
    import hindsight  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from run_train import engine  # noqa: PLC0415

    _, arm, seed, ix = task
    player = _W["player"]
    t0, c0 = time.perf_counter(), time.process_time()
    rng = np.random.default_rng([AGENT_SEED, int(ix)])
    explore = np.random.default_rng([AGENT_SEED, int(ix), 1])
    s = hindsight.session_for(seed, player.agent, "a3-v1", own_schedule=True)
    sink = _W["out"] / "decisions.d" / f"{seed}.jsonl"
    if arm == "teacher":
        sink.parent.mkdir(parents=True, exist_ok=True)
        sink.write_text("", encoding="utf-8")  # a run with no row is played again from its start
    row: dict[str, Any] = {"_task": list(task), "arm": arm, "seed": seed, "ix": int(ix), "end": "?", "floor": 0, "decisions": 0,
                           "screens": Counter(), "teacher_screens": 0, "changed": 0, "teacher_cpu": 0.0, "bosses": []}
    boss: dict | None = None
    ordinal = 0
    try:
        while True:
            if time.perf_counter() - t0 > RUN_WALL_CAP_S:
                row["end"] = "timeout"
                break
            d = s.decision()
            if boss is not None and (d is not None or s.phase != engine.rc.PHASE_COMBAT):
                boss["won"] = bool(d is not None or s.won)
                row["bosses"].append(boss)
                boss = None
            if d is None and s.phase != engine.rc.PHASE_COMBAT:
                row["end"] = "win" if s.won else "death"
                break
            if d is not None:
                view = s.view()
                pick = player.policy.pick(view, d, explore)
                kind = KIND_OF_TYPE.get(d.type)
                index = pick.index
                if sum(1 for o in d.options if o.kind != "unknown") >= 2:
                    row["screens"][f"{kind}/a{int(view.act_index) + 1}"] += 1
                if arm == "teacher" and eligible(kind, view, d, _W["kinds"]):
                    rec = decide(s, d, view, pick, player, seed=seed, ordinal=ordinal, k=_W["k"], max_options=_W["max_options"])
                    index = int(rec["teacher"])
                    row["teacher_screens"] += 1
                    row["changed"] += int(index != pick.index)
                    row["teacher_cpu"] += rec["cpu"]
                    with open(sink, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                ordinal += 1
                s.apply(d.options[index])
                row["decisions"] += 1
                continue
            if boss is None and s.info["node_type"] == engine.vocab.NODE_BOSS:
                info = s.info
                boss = {"act": int(s.act_index), "floor": int(info["floor"]), "hp": int(info["hp"]), "max_hp": int(info["max_hp"])}
            player.play_fight(s, rng, None)
        row["floor"] = int(s.info["floor"])
    finally:
        s.close()
    row["screens"] = dict(row["screens"])
    row["teacher_cpu"] = round(row["teacher_cpu"], 2)
    row["cpu"] = round(time.process_time() - c0, 2)
    row["wall"] = round(time.perf_counter() - t0, 2)
    return row


def capped_row(task: tuple, reason: str, seconds: float, prog: dict) -> dict:
    return {"_task": list(task), "arm": task[1], "seed": task[2], "ix": task[3], "capped": True, "cap_reason": reason, "wall": round(seconds, 1)}


# ---------------------------------------------------------------------------------------------------------- the machine guards


def utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _quiet_lines() -> list[str]:
    try:
        return [x.strip() for x in QUIET.read_text(encoding="utf-8").splitlines() if x.strip()]
    except OSError:
        return []


def _age_h(line: str) -> float:
    try:
        stamp = dt.datetime.strptime(line.split()[1], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except (IndexError, ValueError):
        return 0.0
    return (utc() - stamp).total_seconds() / 3600.0


def blocked() -> str | None:
    """Why the job must not play now: the night runs, main.lock stands, or another live QUIET line does (any tag but ours and the
    always-on loop's `gym-loop`, which pauses its step while ours stands)."""
    if ACTIVE.exists():
        return "nightly-active"
    if MAIN_LOCK.exists():
        return "main.lock"
    return _foreign_line()


def _foreign_line() -> str | None:
    """"quiet:<tag>" for the first live QUIET line that is neither ours nor the always-on loop's, else None."""
    for x in _quiet_lines():
        tag = x.split()[0]
        if tag not in (TAG, "gym-loop") and _age_h(x) < 2.0:
            return "quiet:" + tag
    return None


def night_only() -> str | None:
    """The guard of a job that shares the machine (`play --share`, the coordinator's rule of 2026-11-11 for this measurement): it holds
    no QUIET line and stops for nobody's, only for a night (a CPU-s figure does not depend on the sharing; a wall-clock figure does)."""
    return "nightly-active" if ACTIVE.exists() else None


def must_stop() -> str | None:
    """Why a running job must stop now: `blocked()` less main.lock, which only keeps a job from starting (a landing lets a running
    job finish, as the always-on loop's guards do: a teacher run lost to a landing is about twenty minutes of four processes)."""
    if ACTIVE.exists():
        return "nightly-active"
    return _foreign_line()


def hold(on: bool) -> None:
    """Take (or drop) this job's QUIET.lock line (`run-teacher <UTC> pid=<Windows PID>`); other lines are kept as they are."""
    keep = [x for x in _quiet_lines() if x.split()[0] != TAG]
    if on:
        keep.append(f"{TAG} {utc().strftime('%Y-%m-%dT%H:%M:%SZ')} pid={os.getpid()}")
    if keep:
        QUIET.write_text("".join(x + "\n" for x in keep), encoding="utf-8", newline="\n")
    else:
        QUIET.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------------------- the report


def _paired(a: dict[str, float], b: dict[str, float]) -> dict | None:
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return None
    d = [a[x] - b[x] for x in keys]
    return {"n": len(keys), "mean": round(mean(d), 3), "se": round(math.sqrt(sample_var(d) / len(d)), 3)}


def report(out: Path) -> dict:
    """The measurement's aggregates: CPU-s a run by arm, the share of screens the teacher changes by kind and act (raw, at two
    standard errors, and the split-half self-agreement), the noise and spread of its values, and the arms' outcomes paired by seed."""
    runs = [r for r in read_jsonl(Path(out) / "runs.jsonl") if not r.get("capped")]
    recs = [r for f in sorted((Path(out) / "decisions.d").glob("*.jsonl")) for r in read_jsonl(f)]
    done_teacher = {r["seed"] for r in runs if r["arm"] == "teacher"}
    recs = [r for r in recs if r["seed"] in done_teacher]
    res: dict[str, Any] = {"runs": dict(Counter(r["arm"] for r in runs))}
    for arm in ("player", "teacher"):
        rs = [r for r in runs if r["arm"] == arm]
        if not rs:
            continue
        cpu = sorted(r["cpu"] for r in rs)
        res[arm] = {"runs": len(rs), "cpu_mean": round(mean(cpu), 1), "cpu_median": cpu[len(cpu) // 2], "cpu_p90": cpu[int(0.9 * (len(cpu) - 1))],
                    "teacher_cpu_mean": round(mean([r.get("teacher_cpu", 0.0) for r in rs]), 1),
                    "wall_mean": round(mean([r["wall"] for r in rs]), 1), "floor_mean": round(mean([r["floor"] for r in rs]), 2),
                    "act1_boss_won": sum(1 for r in rs for b in r["bosses"] if b["act"] == 0 and b.get("won")),
                    "act2_boss_won": sum(1 for r in rs for b in r["bosses"] if b["act"] == 1 and b.get("won")),
                    "act1_door_hp_mean": round(mean([b["hp"] for r in rs for b in r["bosses"] if b["act"] == 0]), 1) if any(b["act"] == 0 for r in rs for b in r["bosses"]) else None,
                    "act2_door_hp_mean": round(mean([b["hp"] for r in rs for b in r["bosses"] if b["act"] == 1]), 1) if any(b["act"] == 1 for r in rs for b in r["bosses"]) else None,
                    "teacher_screens_mean": round(mean([r.get("teacher_screens", 0) for r in rs]), 1),
                    "screens_mean": round(mean([sum(r.get("screens", {}).values()) for r in rs]), 1)}
    by = {arm: {r["seed"]: r for r in runs if r["arm"] == arm} for arm in ("player", "teacher")}
    res["paired"] = {
        "floor": _paired({s: r["floor"] for s, r in by["teacher"].items()}, {s: r["floor"] for s, r in by["player"].items()}),
        "act1_boss": _paired({s: float(any(b["act"] == 0 and b.get("won") for b in r["bosses"])) for s, r in by["teacher"].items()},
                             {s: float(any(b["act"] == 0 and b.get("won") for b in r["bosses"])) for s, r in by["player"].items()}),
        "act2_boss": _paired({s: float(any(b["act"] == 1 and b.get("won") for b in r["bosses"])) for s, r in by["teacher"].items()},
                             {s: float(any(b["act"] == 1 and b.get("won") for b in r["bosses"])) for s, r in by["player"].items()}),
    }
    cells: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        if r.get("refused"):
            continue
        cells["all"].append(r)
        cells[f"{r['kind']}/a{int(r['act']) + 1}"].append(r)
        cells[f"act{int(r['act']) + 1}"].append(r)
    table: dict[str, Any] = {}
    for name, rs in sorted(cells.items()):
        changed = [r for r in rs if r["teacher"] != r["player"]]
        raw = [r for r in rs if r.get("teacher_raw", r["teacher"]) != r["player"]]
        halves = [r for r in rs if r.get("half_picks")]
        played_h = [r for r in rs if r.get("half_played")]
        gs = [[r["g"][str(o)] for o in r["played"]] for r in rs]
        ns = noise_and_spread(gs)
        table[name] = {"screens": len(rs), "changed": round(len(changed) / len(rs), 3), "changed_raw": round(len(raw) / len(rs), 3),
                       "split_half_agree": round(mean([float(r["half_picks"][0] == r["half_picks"][1]) for r in halves]), 3) if halves else None,
                       "split_half_agree_played": round(mean([float(r["half_played"][0] == r["half_played"][1]) for r in played_h]), 3) if played_h else None,
                       "chance": round(mean([1 / len(r["played"]) for r in rs]), 3),
                       "gain_mean": round(mean([r["gain"] for r in rs]), 3), "cpu_mean": round(mean([r["cpu"] for r in rs]), 1),
                       "options_played": round(mean([len(r["played"]) for r in rs]), 2), **{f"values_{k}": v for k, v in ns.items()}}
    res["screens"] = table
    lam_cells: dict[str, list] = defaultdict(list)
    for r in recs:
        if r.get("refused") or r.get("logp") is None or not r.get("screen"):
            continue
        rows = [(mean(r["g"][str(o)]), r["v_values"][o], r["logp"][o]) for o in r["played"] if r["v_values"][o] is not None]
        lam_cells["all"].append(rows)
        lam_cells[f"{r['screen']}/a{int(r['act']) + 1}"].append(rows)
        lam_cells[f"act{int(r['act']) + 1}"].append(rows)
    res["lambda"] = {name: lambda_fit(rows) for name, rows in sorted(lam_cells.items())}
    res["refused"] = sum(1 for r in recs if r.get("refused"))
    return res


# ---------------------------------------------------------------------------------------------------------- commands


def seeds_from(path: Path, n: int) -> list[tuple[str, int]]:
    """The first `n` runs (by index) of a lab arm's runs.jsonl: the game seeds and indices the measurement plays."""
    rows = [r for r in read_jsonl(path) if r.get("seed") and r.get("ix") is not None]
    rows.sort(key=lambda r: int(r["ix"]))
    seen, out = set(), []
    for r in rows:
        if r["seed"] not in seen:
            seen.add(r["seed"])
            out.append((str(r["seed"]), int(r["ix"])))
        if len(out) >= n:
            break
    return out


def play_cmd(a: argparse.Namespace) -> int:
    import fight_gym  # noqa: PLC0415 -- its supervised pool (deadlines, a stop poll)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "runs.jsonl"
    done = {tuple(r["_task"]) for r in read_jsonl(target) if not r.get("capped")}
    pairs = seeds_from(Path(a.seeds_from), a.runs)
    arms = [x.strip() for x in a.arm.split(",") if x.strip()]
    tasks = [("rt", arm, seed, ix) for seed, ix in pairs for arm in arms if ("rt", arm, seed, ix) not in done]
    mf = out / "manifest.json"
    if not mf.is_file():
        mf.write_text(json.dumps({"version": VERSION, "created_utc": utc().strftime("%Y-%m-%dT%H:%M:%SZ"), "agent": Path(a.agent).name,
                                  "agent_generation_dir": "/".join(Path(a.agent).parts[-2:]), "judge": "/".join(Path(a.judge).parts[-2:]) if a.judge else None,
                                  "k": a.k, "max_options": a.max_options, "kinds": list(TEACHER_KINDS), "last_floor": LAST_FLOOR,
                                  "beyond": BEYOND, "seeds_from": "/".join(Path(a.seeds_from).parts[-4:]), "runs": len(pairs),
                                  "leaf": "the boss door: P(death) from the judge x the boss floor + P(survive) x beyond[act]; a death on the way its floor",
                                  "procs": a.procs, "share": bool(a.share)}, indent=1), encoding="utf-8")
    print(f"run-teacher: {len(tasks)} runs to play ({', '.join(arms)}), K {a.k}, shortlist {a.max_options}, {a.procs} processes"
          f"{', sharing the machine (no QUIET line)' if a.share else ''}", flush=True)
    start_guard, stop_guard = (night_only, night_only) if a.share else (blocked, must_stop)
    while tasks:
        why = start_guard()
        if why:
            print(f"run-teacher: waiting ({why}) {utc():%H:%M:%SZ}", flush=True)
            time.sleep(60)
            continue
        if not a.share:
            hold(True)
        last = [time.monotonic()]

        def refresh() -> None:
            if not a.share and time.monotonic() - last[0] > 600:
                last[0] = time.monotonic()
                hold(True)

        try:
            reason = fight_gym.run_tasks(tasks, play_run, target, a.procs, hard_of=lambda t: RUN_WALL_CAP_S + 600, capped_of=capped_row,
                                         init=_init_worker, init_args=(str(out), a.agent, a.judge, a.k, a.max_options, list(TEACHER_KINDS),
                                                                       str(ROOT / "out")),
                                         stop=stop_guard, refresh=refresh)
        finally:
            if not a.share:
                hold(False)
        done = {tuple(r["_task"]) for r in read_jsonl(target) if not r.get("capped")}
        tasks = [t for t in tasks if t not in done]
        if reason is None:
            break
        print(f"run-teacher: stopped ({reason}) with {len(tasks)} left {utc():%H:%M:%SZ}", flush=True)
    (out / "report.json").write_text(json.dumps(report(out), indent=1), encoding="utf-8")
    print("run-teacher: done", flush=True)
    return 0


def report_cmd(a: argparse.Namespace) -> int:
    res = report(Path(a.out))
    (Path(a.out) / "report.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


def relabel_cmd(a: argparse.Namespace) -> int:
    import hindsight  # noqa: PLC0415

    res = relabel(hindsight.read_labels(Path(a.labels)))
    text = json.dumps(res, indent=1)
    if a.write:
        Path(a.write).write_text(text, encoding="utf-8")
    print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("play", help="play the measurement's runs (resumable)")
    pl.add_argument("--out", required=True)
    pl.add_argument("--arm", default="player,teacher", help="player, teacher, or both (comma-separated)")
    pl.add_argument("--agent", default=DEFAULT_AGENT)
    pl.add_argument("--judge", default=DEFAULT_JUDGE)
    pl.add_argument("--seeds-from", default=DEFAULT_SEEDS_FROM)
    pl.add_argument("--runs", type=int, default=50)
    pl.add_argument("--procs", type=int, default=4)
    pl.add_argument("--k", type=int, default=DEFAULT_K)
    pl.add_argument("--max-options", type=int, default=DEFAULT_MAX_OPTIONS)
    pl.add_argument("--share", action="store_true", help="share the machine: no QUIET line, stop only for a night")
    rp = sub.add_parser("report", help="the measurement's aggregates")
    rp.add_argument("--out", required=True)
    rl = sub.add_parser("relabel", help="stored hindsight labels rescored with the door leaf")
    rl.add_argument("--labels", required=True)
    rl.add_argument("--write", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return {"play": play_cmd, "report": report_cmd, "relabel": relabel_cmd}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
