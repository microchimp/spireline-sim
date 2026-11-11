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
    python rl/run_teacher.py job     --lab LAB --store DIR --agent-root DIR [--source auto|new|stored] [--procs 4] [--block-min 60]
    python rl/run_teacher.py label   --lab LAB --store DIR --agent-root DIR   # `job --source stored`: round 0 from the stored exploring runs

`play` is the measurement (D56): it holds its own QUIET.lock line (`run-teacher`), stops on another live line (not the always-on loop's
`gym-loop`), on main.lock and while a night runs, and resumes where it stopped (a run with a row is never played again).

`job` is the same teacher as a lab job (docs/lab/TEACHER_JOB.md): it plays the current player (`agent/current.json`) under the filler's
rules (`tools/lab/jobs.teacher_hold`: BelowNormal, judgement first, none within an hour of a due presort, a daily core-hour budget), and
writes the **decision shards** the student learns from into `<store>/<generation>/` (shards/<job>/decisions.jsonl, rows.npz, manifest.json;
jobs/<job>/shards: the run records through `gen_loop.RunRecorder`, so the run value refits from the same runs). Two sources: `new` plays whole
runs with the guarded teacher deciding every screen it covers; `stored` replays the lab's stored exploring runs exactly (the same streams,
checked against their records) and labels some of their screens, the run itself unchanged. Adaptive K: K 2 draws of every shortlisted option
first, two more only where the top two options are within two standard errors. P(death) at the door is recalibrated per act on the player's
own door outcomes (`fit_door_calibration`). A killed run is replayed from its start and reuses the screens it had already labelled.

Lab code: nothing here changes the engine, the search, the lab's self-play, the loop or the night. No human record is read anywhere in this
file.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import shutil
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
# The job (`job`): adaptive K, the door's recalibration, the stored-run source, the store.
JOB_K = 2  # draws of every shortlisted option before the first look
JOB_K_MAX = 4  # draws at most: two more when the top two options are within Z standard errors
JOB_K_STEP = 2
CAL_MIN_DOORS = 60  # a door calibration per act needs this many of the player's own door outcomes; fewer keeps the judge's P(death)
CAL_RIDGE = 20.0  # pseudo-observations pulling the fitted (intercept, slope) towards (0, 1)
CAL_CLIP = 1e-3  # P(death) is clipped to [CAL_CLIP, 1 - CAL_CLIP] before its logit
LABEL_RATE = 0.2  # source `stored`: the chance a covered screen of a replayed run is labelled (a hash of the seed and the ordinal)
LABEL_PER_RUN = 5  # source `stored`: screens labelled from one replayed run (the replay stops there)
STORED_TARGET = 10000  # source `auto`: labelled screens from stored runs before the job turns to new play
STORED_CONFIG = "a3-v1@h1e4b16"  # the search schedule of the stored generation 13 runs (`config` of their rows)
STORED_FROM, STORED_TO = "20261105", "20261108"  # the stored runs of round 0: job folders of these dates (6,889 runs of generation 13)
ROWS_WIDTH = 1863  # human_prior_infer.option_rows
SHARD_DECISIONS, SHARD_ROWS, SHARD_MANIFEST = "decisions.jsonl", "rows.npz", "manifest.json"
DEFAULT_LAB = "D:/spireline/referee/LAB"
DEFAULT_STORE = "D:/spireline/referee/run-teacher"
DEFAULT_AGENT_ROOT = "D:/spireline/referee/agent"
DEFAULT_RUNS_ROOT = "D:/spireline/referee/lab/runs"
SEED_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # gen0.ALPHABET: the lab's game seeds
NEW_IX0 = 1_000_000  # the agent streams' index of the job's new runs (the lab's runs count from 0)


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


def door_return(end: str, floor: int, p_death: float | None, beyond: dict[int, float] = BEYOND, won_floor: float = WON_FLOOR,
                cal: dict[int, tuple[float, float]] | None = None) -> float | None:
    """G of one continuation, in final floors: a death on the way is its floor; at the boss door, P(death) x the boss's floor plus
    P(survive) x the value of surviving that boss (`beyond`); a won run its floor. None for any other end (timeout, step cap, a door
    with no estimate, a door past the acts `beyond` knows): such a draw has no return and the decision falls back to the player.
    With `cal` (`fit_door_calibration`) the judge's P(death) is first mapped through its act's recalibration."""
    if end == "death":
        return float(floor)
    if end == "win":
        return float(won_floor)
    if end == "door":
        act = act_of_floor(int(floor))
        if p_death is None or act not in beyond or not (0.0 <= float(p_death) <= 1.0):
            return None
        p = recalibrate(float(p_death), act, cal)
        return p * float(floor) + (1.0 - p) * float(beyond[act])
    return None


# ---------------------------------------------------------------------------------------------------------- the door's recalibration


def _logit(p: float) -> float:
    p = min(max(float(p), CAL_CLIP), 1.0 - CAL_CLIP)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def recalibrate(p: float, act: int, cal: dict[int, tuple[float, float]] | None) -> float:
    """The judge's P(death) `p` at an act's boss door mapped through that act's calibration (a, b): sigmoid(a + b logit p). An act with
    no calibration (too few of the player's door outcomes yet) keeps `p`."""
    if not cal or act not in cal:
        return float(p)
    a, b = cal[act]
    return _sigmoid(a + b * _logit(p))


def fit_door_calibration(pairs: Iterable[tuple[int, float, int]], min_n: int = CAL_MIN_DOORS, ridge: float = CAL_RIDGE) -> dict[int, dict[str, Any]]:
    """Per act, a logistic recalibration of the judge's P(death) at the boss door on the player's own door outcomes: `pairs` are
    (act, the judge's P(death) there, 1 if the player died in that boss fight else 0). Newton's method on the log-likelihood with a ridge of
    `ridge` pseudo-observations pulling (intercept a, slope b) towards (0, 1), the judge as it stands. An act with fewer than `min_n` doors
    is left out (identity). Each entry: a, b, n, deaths, the mean P(death) before and after, and the Brier score before and after."""
    by_act: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for act, p, y in pairs:
        if p is not None and 0.0 <= float(p) <= 1.0 and y in (0, 1):
            by_act[int(act)].append((_logit(p), int(y)))
    out: dict[int, dict[str, Any]] = {}
    for act, xs in sorted(by_act.items()):
        if len(xs) < min_n:
            continue
        a, b = 0.0, 1.0
        for _ in range(50):
            ga = gb = haa = hab = hbb = 0.0
            for x, y in xs:
                q = _sigmoid(a + b * x)
                r = q - y
                w = q * (1.0 - q)
                ga += r
                gb += r * x
                haa += w
                hab += w * x
                hbb += w * x * x
            ga += ridge * a
            gb += ridge * (b - 1.0)
            haa += ridge
            hbb += ridge
            det = haa * hbb - hab * hab
            if det <= 1e-12:
                break
            da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
            a, b = a - da, b - db
            if abs(da) + abs(db) < 1e-9:
                break
        before = [_sigmoid(x) for x, _ in xs]
        after = [_sigmoid(a + b * x) for x, _ in xs]
        ys = [y for _, y in xs]
        out[act] = {"a": round(a, 5), "b": round(b, 5), "n": len(xs), "deaths": sum(ys),
                    "p_mean_before": round(mean(before), 4), "p_mean_after": round(mean(after), 4), "death_rate": round(mean(ys), 4),
                    "brier_before": round(mean([(q - y) ** 2 for q, y in zip(before, ys, strict=True)]), 4),
                    "brier_after": round(mean([(q - y) ** 2 for q, y in zip(after, ys, strict=True)]), 4)}
    return out


def calibration_map(doc: dict[Any, dict[str, Any]] | None) -> dict[int, tuple[float, float]]:
    """The (a, b) of each act of a `fit_door_calibration` result (or of calibration.json's `acts`), keyed by act number."""
    return {int(a): (float(v["a"]), float(v["b"])) for a, v in (doc or {}).items() if isinstance(v, dict) and "a" in v and "b" in v}


def door_pairs(runs: Iterable[dict]) -> list[tuple[int, float, int]]:
    """The player's own door outcomes of a store's run rows: for each boss the run reached, (act, the judge's P(death) at the door,
    1 if the run lost that fight)."""
    out = []
    for r in runs:
        if r.get("capped"):
            continue
        for b in r.get("bosses") or []:
            if b.get("p_death") is not None and "won" in b:
                out.append((int(b["act"]), float(b["p_death"]), 0 if b["won"] else 1))
    return out


def label_door_pairs(labels: Iterable[dict]) -> list[tuple[int, float, int]]:
    """The same pairs from stored hindsight labels (`rl/hindsight.py`): a continuation that reached its boss door with a judge estimate and
    played the boss on, died there or did not. An optional bootstrap for the first days, while the store has few doors of its own."""
    out = []
    for r in labels:
        if r.get("refused"):
            continue
        for c in r.get("continuations") or []:
            door = c.get("door")
            p = ((door or {}).get("judge") or {}).get("p_death")
            if p is not None and c.get("end") is not None:
                out.append((int(r["act"]), float(p), int(c["end"] == "death" and int(c.get("floor", -1)) == int(door["floor"]))))
    return out


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


def top_two(g: dict[int, list[float]], extra: dict[int, float] | None = None) -> tuple[int, int, float, float, list[float]] | None:
    """The two best options by score (the mean return plus `extra`, the prior's share on a blended screen), their gap, and the paired
    draws of the difference: (best, second, gap, mean difference, per-draw differences). None with fewer than two options."""
    extra = extra or {}
    if len(g) < 2:
        return None
    score = {o: mean(v) + extra.get(o, 0.0) for o, v in g.items()}
    ranked = sorted(g, key=lambda o: (-score[o], o))
    a, b = ranked[0], ranked[1]
    diffs = [(x + extra.get(a, 0.0)) - (y + extra.get(b, 0.0)) for x, y in zip(g[a], g[b], strict=False)]
    return a, b, score[a] - score[b], mean(diffs), diffs


def wants_more_draws(g: dict[int, list[float]], kind: str | None, extra: dict[int, float] | None = None, z: float = Z) -> bool:
    """Adaptive K (D57 part 3): after the first draws, two more only where the top two options are within `z` standard errors of each
    other (the guard's own error, `guard_se`: the paired variance pooled with the kind's prior). Options that are clearly apart, or a
    screen with fewer than two options valued, stop where they are."""
    t = top_two(g, extra)
    if t is None:
        return False
    _, _, gap, _, diffs = t
    return gap < z * guard_se(diffs, kind)


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


def continuation(base: Any, option: Any, player: Any, *, seed: str, ordinal: int, k: int, cal: dict[int, tuple[float, float]] | None = None) -> dict:
    """Draw k of one option: fork `base` on the decision's resample seed for k, apply the option, play on to the door. `cal` recalibrates
    the judge's P(death) at the door (`door_return`); the draw's `p_death` stays the judge's own number."""
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
    g = door_return(out["end"], out["floor"], p, cal=cal)
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


def screen_inputs(policy: Any, view: Any, decision: Any, rows: bool = False) -> dict[str, Any]:
    """The player's own reading of a screen: its lambda there, the screen the blend decides (None: V alone), the prior's
    log-probabilities and logits over the options, the rest groups, and the player's score of each option (V + lambda log pi_h on a
    blended screen, V elsewhere; None for an option V cannot score). With `rows` also the judgement's input rows of every option
    (`human_prior_infer.option_rows`, float32 (n, 1863)): what the decision shard stores for the student."""
    import numpy as np  # noqa: PLC0415

    import imitation_chooser as ic  # noqa: PLC0415
    from human_prior_infer import log_softmax, option_rows  # noqa: PLC0415
    from run_train import models  # noqa: PLC0415
    from run_train import policy as run_policy  # noqa: PLC0415

    view = decision.view or view
    lam = policy.lam_at(view.floor) if hasattr(policy, "lam_at") else 0.0
    screen = ic.screen_of(decision) if lam > 0 else None
    options = list(decision.options)
    out: dict[str, Any] = {"lam": lam, "screen": screen, "logp": None, "logits": None, "groups": None, "scores": [None] * len(options),
                           "v": [None] * len(options), "rows": None}
    idx = [i for i, o in enumerate(options) if o.scorable]
    if idx:
        probs = run_policy.floor_probs(policy.nets, view, [options[i] for i in idx])
        for i, v in zip(idx, models.expected_floors(probs, policy.W).tolist(), strict=True):
            out["v"][i] = float(v)
            out["scores"][i] = float(v)
    if screen is not None or rows:
        mat = option_rows(policy.nets, view, options)
        if rows:
            out["rows"] = np.ascontiguousarray(mat, dtype=np.float32)
        if screen is not None:
            logits = [float(x) for x in policy.prior.row_logits(mat)]
            logp = [float(x) for x in log_softmax(logits)]
            out.update(logp=logp, logits=logits, groups=[ic.rest_group(o) for o in options] if screen == "rest" else None)
            if not math.isinf(lam):
                out["scores"] = [None if v is None else v + lam * lp for v, lp in zip(out["v"], logp, strict=True)]
    return out


def prior_extra(rd: dict[str, Any], options: Iterable[int]) -> dict[int, float]:
    """The prior's share of an option's score on a screen the player blends (lambda x log pi_h), else nothing: what the adaptive look adds
    to the options' mean returns to rank them as the pick does."""
    if rd.get("screen") is None or rd.get("logp") is None or math.isinf(rd["lam"]):
        return {}
    return {o: rd["lam"] * rd["logp"][o] for o in options}


def decide(s: Any, decision: Any, view: Any, pick: Any, player: Any, *, seed: str, ordinal: int, k: int, max_options: int,
           k_max: int | None = None, k_step: int = JOB_K_STEP, rows_out: list | None = None,
           cal: dict[int, tuple[float, float]] | None = None) -> dict:
    """Play the screen's shortlisted options K times each to the door and return the decision's record: the teacher's index,
    the player's, each draw's G and what the screen's reading was. With `k_max` above `k` the draws are taken adaptively: `k` first, then
    `k_step` more at a time while the top two options are within Z standard errors (`wants_more_draws`), up to `k_max`; the record's `k` is
    the number of draws taken. With `rows_out` the screen's option rows (`screen_inputs`) are appended to it."""
    t0, c0 = time.perf_counter(), time.process_time()
    view = decision.view or view
    rd = screen_inputs(player.policy, view, decision, rows=rows_out is not None)
    if rows_out is not None:
        rows_out.append(rd["rows"])
    chosen = shortlist(decision.options, rd["scores"], max_options, pick.index)
    kind = KIND_OF_TYPE.get(decision.type)
    extra = prior_extra(rd, chosen)
    top = max(int(k), int(k_max or 0))
    draws: dict[int, list[dict]] = {i: [] for i in chosen}
    taken = 0
    while True:
        upto = min(top, taken + (int(k) if taken == 0 else int(k_step)))
        for kk in range(taken, upto):  # draw-major: the same draw for every option before the next
            for oi in chosen:
                draws[oi].append(continuation(s, decision.options[oi], player, seed=seed, ordinal=ordinal, k=kk, cal=cal))
        taken = upto
        if taken >= top:
            break
        g_now = {o: [d["G"] for d in rows] for o, rows in draws.items()}
        if any(x is None for v in g_now.values() for x in v) or int(pick.index) not in g_now or not wants_more_draws(g_now, kind, extra):
            break
    k = taken
    g = {o: [d["G"] for d in rows] for o, rows in draws.items()}
    ok = all(x is not None for v in g.values() for x in v)
    rec: dict[str, Any] = {"v": VERSION, "seed": seed, "ordinal": ordinal, "type": decision.type, "kind": kind,
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

        def score_gain(t: int, vals: dict[int, float]) -> float:
            """The teacher's score of its pick over the player's: Q's difference, plus the prior's on a blended screen."""
            extra_gain = 0.0
            if rd["screen"] is not None and rd["logp"] is not None and not math.isinf(rd["lam"]):
                extra_gain = rd["lam"] * (rd["logp"][t] - rd["logp"][p])
            return vals[t] - vals[p] + extra_gain

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
    done_teacher = {r["seed"] for r in runs if r["arm"] in ("teacher", "label")}
    recs = [r for r in recs if r["seed"] in done_teacher]
    res: dict[str, Any] = {"runs": dict(Counter(r["arm"] for r in runs))}
    for arm in ("player", "teacher", "label"):
        rs = [r for r in runs if r["arm"] == arm]
        if not rs:
            continue
        for r in rs:
            r.setdefault("bosses", [])
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


# ---------------------------------------------------------------------------------------------------------- the job: seeds, the store
# (D56, D57 part 3; docs/lab/TEACHER_JOB.md.) The store of a generation is `<store>/<generation>/`:
#   runs.jsonl                  one row per finished run (a run with a row is never played again; a capped row is not a finished run)
#   decisions.d/<seed>.jsonl    a run's decision records as they are made, `<seed>.rows.f32` their option rows (float32, 1,863 wide),
#                               `<seed>.record.json` the run's RunRecord (gen_loop.RunRecorder) once it ended; a killed run's files are
#                               what its replay reuses
#   shards/<job>/               the decision shard: decisions.jsonl, rows.npz, manifest.json (what rl/run_student.py loads)
#   jobs/<job>/                 job.json, result.json, shards/ (the RunRecords of the job's runs in the lab's own shard layout, for V)
#   calibration.json            the door's recalibration, refitted at every job start
#   manifest.json               the store's index: the settings, the totals
# and `<store>/ledger.jsonl` is the CPU ledger of the daily budget (tools/lab/jobs.py).


def new_seed(generation: int, n: int) -> str:
    """The game seed of the generation's n-th new run: ten characters of the lab's seed alphabet from the job's own namespace, so no
    run of any other job (the lab's blocks, the exam, the gym) shares it."""
    digest = hashlib.sha256(f"{NAMESPACE}|seed|{int(generation)}|{int(n)}".encode()).digest()
    return "".join(SEED_ALPHABET[b % len(SEED_ALPHABET)] for b in digest[:10])


def coin(seed: str, ordinal: int) -> float:
    """A fixed number in [0, 1) for a (run, decision) of the stored-run source: which screens of a replayed run are labelled."""
    return _digest("coin", seed, int(ordinal)) / 2.0**64


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON: {type(o).__name__}")


def write_json(path: Path, doc: Any) -> None:
    """Whole or not at all: a temporary file beside it, then a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, default=_json_default), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def clean_path(value: Any) -> Any:
    """A manifest or result value without absolute paths (the last two parts of a path stay)."""
    if isinstance(value, str) and (value[1:3] in (":/", ":\\") or value.startswith("/")):
        return "/".join(Path(value.replace("\\", "/")).parts[-2:])
    if isinstance(value, dict):
        return {k: clean_path(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_path(v) for v in value]
    return value


class RunFiles:
    """One run's decision records and option rows in the store, written as the run goes so that a killed run loses nothing of its labelled
    screens: a record is appended after its rows, `row0` its first row in the run's own row file. `load` reads back the records whose rows
    are all there (in order) and cuts both files to that prefix."""

    def __init__(self, gen_dir: Path | str, seed: str) -> None:
        d = Path(gen_dir) / "decisions.d"
        self.recs, self.rows, self.record = d / f"{seed}.jsonl", d / f"{seed}.rows.f32", d / f"{seed}.record.json"
        self.n_rows = 0
        self.by_ordinal: dict[int, dict] = {}

    def load(self) -> dict[int, dict]:
        width = ROWS_WIDTH * 4
        size = self.rows.stat().st_size if self.rows.is_file() else 0
        have = size // width
        keep: list[dict] = []
        end = 0
        for r in read_jsonl(self.recs):
            if r.get("row0") is not None:
                last = int(r["row0"]) + int(r["n_options"])
                if int(r["row0"]) != end or last > have:
                    break
                end = last
            keep.append(r)
        text = "".join(json.dumps(r) + "\n" for r in keep)
        if self.recs.is_file() and self.recs.read_text(encoding="utf-8", errors="replace") != text:
            self.recs.write_text(text, encoding="utf-8", newline="\n")  # cut to the consistent prefix
        if self.rows.is_file() and size != end * width:
            with open(self.rows, "r+b") as fh:
                fh.truncate(end * width)
        self.n_rows = end
        self.by_ordinal = {int(r["ordinal"]): r for r in keep}
        return self.by_ordinal

    def append(self, rec: dict, rows: Any) -> dict:
        """Store a decision record with its rows (a float32 array (n_options, 1863), or None when none could be built); returns the stored record."""
        import numpy as np  # noqa: PLC0415

        self.recs.parent.mkdir(parents=True, exist_ok=True)
        r = dict(rec)
        if rows is not None:
            arr = np.ascontiguousarray(rows, dtype="<f4")
            if arr.shape != (int(rec["n_options"]), ROWS_WIDTH):
                raise ValueError(f"rows {arr.shape} for {rec['n_options']} options")
            with open(self.rows, "ab") as fh:
                fh.write(arr.tobytes())
            r["row0"] = self.n_rows
            self.n_rows += len(arr)
        else:
            r["row0"] = None
        with open(self.recs, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(r, default=_json_default) + "\n")
        self.by_ordinal[int(r["ordinal"])] = r
        return r

    def drop_from(self, ordinal: int) -> None:
        """Forget the records of this decision and after (a replay that left them: they are computed again)."""
        keep = [r for o, r in sorted(self.by_ordinal.items()) if o < ordinal]
        end = max((int(r["row0"]) + int(r["n_options"]) for r in keep if r.get("row0") is not None), default=0)
        self.recs.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8", newline="\n")
        if self.rows.is_file():
            with open(self.rows, "r+b") as fh:
                fh.truncate(end * ROWS_WIDTH * 4)
        self.n_rows, self.by_ordinal = end, {int(r["ordinal"]): r for r in keep}


def done_rows(gen_dir: Path | str) -> list[dict]:
    """The finished runs of a store: the rows of runs.jsonl that are not capped."""
    return [r for r in read_jsonl(Path(gen_dir) / "runs.jsonl") if not r.get("capped")]


def sharded_seeds(gen_dir: Path | str, but: str | None = None) -> set[str]:
    """The runs whose decisions an earlier job's shard already holds (the shard `but`, which is being rebuilt, aside)."""
    out: set[str] = set()
    for m in sorted((Path(gen_dir) / "shards").glob(f"*/{SHARD_MANIFEST}")):
        if m.parent.name == but:
            continue
        with contextlib.suppress(OSError, ValueError):
            out.update(str(s) for s in json.loads(m.read_text(encoding="utf-8")).get("seeds", []))
    return out


def build_shard(gen_dir: Path | str, name: str, seeds: Sequence[str], manifest: dict) -> dict:
    """Write the decision shard `shards/<name>/` from the finished runs `seeds` (in that order): `decisions.jsonl` one record per screen
    exactly as the teacher made it plus `row0` (the screen's first row in `rows.npz`), `rows.npz` (`rows` float32 (rows, 1863) as
    `human_prior_infer.option_rows` built them at play, `decision` int32 the record's line, `option` int16 the option's index) and
    `manifest.json` (`manifest` plus the counts and the runs). Rebuilt whole when it exists; returns the manifest written."""
    import numpy as np  # noqa: PLC0415

    gen_dir = Path(gen_dir)
    recs_out: list[dict] = []
    rows_parts, dec_parts, opt_parts = [], [], []
    n_rows = 0
    by_kind: Counter = Counter()
    refused: Counter = Counter()
    extended = 0
    for seed in seeds:
        rf = RunFiles(gen_dir, seed)
        recs = rf.load()
        mat = np.fromfile(rf.rows, dtype="<f4").reshape(-1, ROWS_WIDTH) if rf.rows.is_file() and rf.n_rows else np.zeros((0, ROWS_WIDTH), dtype="<f4")
        for ordinal in sorted(recs):
            r = dict(recs[ordinal])
            local = r.pop("row0", None)
            line = len(recs_out)
            if local is not None:
                n = int(r["n_options"])
                rows_parts.append(mat[int(local) : int(local) + n])
                dec_parts.append(np.full(n, line, dtype=np.int32))
                opt_parts.append(np.arange(n, dtype=np.int16))
                r["row0"] = n_rows
                n_rows += n
            else:
                r["row0"] = None
            recs_out.append(r)
            by_kind[f"{r.get('kind')}/a{int(r.get('act', 0)) + 1}"] += 1
            if r.get("refused"):
                refused[r["refused"]] += 1
            extended += int(int(r.get("k", 0)) > int(manifest.get("k", 0)))
    shards = gen_dir / "shards"
    tmp, final = shards / f"{name}.tmp", shards / name
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    (tmp / SHARD_DECISIONS).write_text("".join(json.dumps(r, default=_json_default) + "\n" for r in recs_out), encoding="utf-8", newline="\n")
    np.savez_compressed(tmp / "rows.tmp.npz",
                        rows=np.concatenate(rows_parts).astype(np.float32) if rows_parts else np.zeros((0, ROWS_WIDTH), dtype=np.float32),
                        decision=np.concatenate(dec_parts) if dec_parts else np.zeros(0, dtype=np.int32),
                        option=np.concatenate(opt_parts) if opt_parts else np.zeros(0, dtype=np.int16))
    os.replace(tmp / "rows.tmp.npz", tmp / SHARD_ROWS)
    doc = {**manifest, "seeds": list(seeds), "runs": len(seeds),
           "counts": {"decisions": len(recs_out), "rows": n_rows, "runs": len(seeds), "refused": int(sum(refused.values())),
                      "k_extended": extended, "by_kind_act": dict(sorted(by_kind.items()))}}
    write_json(tmp / SHARD_MANIFEST, clean_path(doc))
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    return doc


def file_digest(path: Path | str | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def agent_identity(agent_dir: Path | str, agent: Any) -> dict[str, Any]:
    """What the shard's manifest says the player was: the generation folder, the sha256s of what plays (the combat prior's own, the run
    head, the human prior) and one sha256 over them and the recipe."""
    meta: dict[str, Any] = {}
    with contextlib.suppress(OSError, ValueError):
        meta = json.loads((Path(agent_dir) / "model.json").read_text(encoding="utf-8"))
    hp = getattr(agent, "human_prior", None)
    hp = hp if hp and Path(str(hp)).is_file() else (Path(agent_dir) / "human_prior.npz" if (Path(agent_dir) / "human_prior.npz").is_file() else None)
    files = {"prior_sha256": meta.get("policy_sha256"), "run_head_sha256": file_digest(getattr(agent, "macro_npz", None)),
             "human_prior_sha256": file_digest(hp)}
    recipe = agent.recipe() if hasattr(agent, "recipe") else None
    whole = hashlib.sha256((json.dumps(files, sort_keys=True) + "|" + str(recipe)).encode()).hexdigest()
    return {"agent": "/".join(Path(str(agent_dir)).parts[-2:]), "generation": getattr(agent, "generation", None), "recipe": recipe,
            "agent_files": files, "agent_sha256": whole}


def shard_manifest(*, job: str, generation: int, identity: dict, judge: str | None, lib: str | None, tree: str | None, k: int, k_max: int,
                   max_options: int, cal: dict | None, sources: dict[str, int]) -> dict:
    """The manifest of a job's decision shard (before its counts): who played (agent, judge, engine sha256s), K and the shortlist, the leaf's
    B, the guard's Z and priors, the door's recalibration, the namespace of the seeds."""
    return {"version": VERSION, "kind": "run-teacher-shard", "created_utc": utc().strftime("%Y-%m-%dT%H:%M:%SZ"), "job": job,
            "generation": generation, **identity,
            "judge": "/".join(Path(judge).parts[-2:]) if judge else None, "judge_sha256": file_digest(judge),
            "engine": tree, "engine_sha256": file_digest(Path(lib) / "Sts2Emulator.dll") if lib else None,
            "k": k, "k_max": k_max, "adaptive": k_max > k, "m": max_options, "max_options": max_options,
            "beyond": BEYOND, "leaf_b": BEYOND, "last_floor": LAST_FLOOR, "guard": {"z": Z, "nu0": NU0, "paired_sd_prior": PAIRED_SD_PRIOR,
                                                                                     "paired_sd_default": PAIRED_SD_DEFAULT},
            "recal": cal or {}, "leaf": "the boss door: P(death) from the judge (recalibrated per act where the store has enough of the player's own "
                                         "door outcomes) x the boss floor + P(survive) x beyond[act]; a death on the way its floor",
            "seeds_namespace": NAMESPACE, "sources": sources, "kinds": list(TEACHER_KINDS),
            "human_corpus": "not read"}


# ---------------------------------------------------------------------------------------------------------- the job: tasks and sources


def new_tasks(generation: int, done: set[str], limit: int) -> list[tuple]:
    """The next `limit` new runs of the generation that have no finished row: ("rt-new", seed, ix); a run left half done comes first
    (it has the lowest index) and is replayed with the screens it had."""
    out: list[tuple] = []
    n = 0
    while len(out) < limit:
        seed = new_seed(generation, n)
        if seed not in done:
            out.append(("rt-new", seed, new_ix(n)))
        n += 1
    return out


def new_ix(n: int) -> int:
    return NEW_IX0 + int(n)


def stored_tasks(rows: Sequence[dict], done: set[str], limit: int, date_from: str = STORED_FROM, date_to: str = STORED_TO) -> list[tuple]:
    """The stored exploring runs (`hindsight.exploring_runs` rows) of the job folders dated `date_from` to `date_to` (YYYYMMDD, inclusive) that
    have no finished row, in a fixed shuffle (a hash of the seed): ("rt-label", seed, ix, row)."""
    pick = [r for r in rows if date_from <= str(r.get("_job", ""))[:8] <= date_to and r.get("seed") and r.get("ix") is not None
            and r["seed"] not in done]
    pick.sort(key=lambda r: (coin(str(r["seed"]), -1), str(r["seed"])))
    return [("rt-label", str(r["seed"]), int(r["ix"]), r) for r in pick[:limit]]


def choose_source(source: str, stored_screens: int, stored_left: int, target: int = STORED_TARGET) -> str:
    """`auto` is round 0 first: the stored runs while they have not yielded `target` screens and there are some left, then new play."""
    if source != "auto":
        return source
    return "stored" if stored_screens < target and stored_left > 0 else "new"


# ---------------------------------------------------------------------------------------------------------- the job: the workers

_J: dict[str, Any] = {}


def _init_job_worker(cfg: dict) -> None:
    os.environ["STS2_LIB_PATH"] = cfg.get("lib") or str(ROOT / "out")
    os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    from sts2_gym import procutil  # noqa: PLC0415

    procutil.lower_priority()  # BelowNormal (Windows priority class)
    import hindsight  # noqa: PLC0415

    player = hindsight.build_player(cfg["agent_dir"], config="a3-v1", judge_path=cfg.get("judge") or None, own_schedule=True)
    with contextlib.suppress(Exception):
        import torch  # noqa: PLC0415

        torch.set_num_threads(1)
    _J.clear()
    _J.update(cfg, player=player, replayers={}, cal_map=calibration_map(cfg.get("cal")), gen_dir=Path(cfg["gen_dir"]))


def _replayer(explore: tuple[float, float]) -> Any:
    """The exploring player that played a stored run (its streams and picks are redrawn by `hindsight.replay_run`)."""
    import hindsight  # noqa: PLC0415

    cache = _J["replayers"]
    if explore not in cache:
        cache[explore] = hindsight.build_player(_J["replay_agent"], config="a3-v1", judge_path=None, explore=explore, own_schedule=True)
    return cache[explore]


def _gate(est_s: float, seed: str) -> str | None:
    """Why this worker must not start a run now: the block's time is over, the STOP file stands, or the day's budget (the ledger, with this
    run's estimate reserved and every reservation before it) has no room. The reservation is written first, so two workers that look at
    once do not both take the last room: the earlier line wins and the other retracts its own."""
    from tools.lab import jobs  # noqa: PLC0415

    if time.time() > _J["deadline"]:
        return "block-end"
    if _J.get("stop_file") and Path(_J["stop_file"]).exists():
        return "stop-file"
    ledger = Path(_J["ledger"])
    jobs.append_ledger(ledger, {"ev": "start", "job": _J["job"], "seed": seed, "est": round(est_s, 1)})
    rows = jobs.read_ledger(ledger)
    mine = max((i for i, r in enumerate(rows) if r.get("ev") == "start" and r.get("job") == _J["job"] and r.get("seed") == seed), default=len(rows) - 1)
    now = time.time()
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    if jobs.teacher_spent_s(rows[: mine + 1], day, now) / 3600.0 > float(_J["budget_h"]):
        jobs.append_ledger(ledger, {"ev": "end", "job": _J["job"], "seed": seed, "cpu": 0.0, "retracted": True})
        return "budget"
    return None


def _skip_row(task: tuple, reason: str) -> dict:
    return {"_task": list(task[:3]), "arm": "label" if task[0] == "rt-label" else "teacher", "seed": task[1], "ix": task[2], "capped": True,
            "cap_reason": reason, "wall": 0.0}


def _taken_pick(pick: Any, index: int) -> Any:
    """The player's Pick as the run recorder keeps it when the teacher took another option: its index, no probabilities of the player's own."""
    import dataclasses  # noqa: PLC0415

    return pick if int(index) == int(pick.index) else dataclasses.replace(pick, index=int(index), p_chosen=1.0, p_floor=None)


def _reusable(have: dict | None, seed: str, d: Any, view: Any, pick: Any) -> bool:
    """A stored record stands for this screen when the replay is at the same place the record was made: the same run, screen type, floor,
    number of options and the player's own pick."""
    return (have is not None and have.get("seed") == seed and have.get("type") == d.type and int(have.get("floor", -1)) == int(view.floor)
            and int(have.get("n_options", -1)) == len(d.options) and int(have.get("player", -1)) == int(pick.index))


def _screen(files: RunFiles, ordinal: int, d: Any, view: Any, pick: Any, s: Any, seed: str, counters: dict) -> dict:
    """The teacher's record of this screen: the stored one when the run is being replayed past it, else played (K draws, adaptive) and stored."""
    have = files.by_ordinal.get(ordinal)
    if _reusable(have, seed, d, view, pick):
        counters["reused"] += 1
        counters["reused_cpu"] += float(have.get("cpu") or 0.0)
        return have
    if have is not None or any(o >= ordinal for o in files.by_ordinal):
        files.drop_from(ordinal)  # the replay left what was stored: from here on it is played again
    rows_out: list = []
    rec = decide(s, d, view, pick, _J["player"], seed=seed, ordinal=ordinal, k=_J["k"], max_options=_J["max_options"], k_max=_J["k_max"],
                 rows_out=rows_out, cal=_J["cal_map"])
    return files.append(rec, rows_out[0] if rows_out else None)


def new_run(task: tuple) -> dict:
    """One whole new run: `task` = ("rt-new", seed, ix). The player's own run loop (`gen0.play_emu_run`'s streams and chooser) with the guarded
    teacher deciding every covered screen below the Act 2 boss; the run's RunRecord is kept (`gen_loop.RunRecorder`) and the P(death) at each
    boss door recorded beside the fight's outcome, for the door's recalibration."""
    import numpy as np  # noqa: PLC0415

    import gen0  # noqa: PLC0415
    import gen_loop  # noqa: PLC0415
    import hindsight  # noqa: PLC0415
    from run_train import engine  # noqa: PLC0415
    from tools.lab import jobs  # noqa: PLC0415

    _, seed, ix = task[:3]
    why = _gate(_J["est_new_s"], seed)
    if why:
        return _skip_row(task, why)
    player = _J["player"]
    policy = player.policy
    t0, c0 = time.perf_counter(), time.process_time()
    files = RunFiles(_J["gen_dir"], seed)
    files.load()
    rng = np.random.default_rng([AGENT_SEED, int(ix)])
    explore = np.random.default_rng([AGENT_SEED, int(ix), 1])
    cfg = player.agent.search_config("a3-v1")
    rec = gen_loop.RunRecorder(seed, player.agent.generation, policy.describe(), cfg.sims, cfg.c, hindsight.ASCENSION)
    s = hindsight.session_for(seed, player.agent, "a3-v1", own_schedule=True)
    row: dict[str, Any] = {"_task": list(task[:3]), "arm": "teacher", "source": "new", "job": _J["job"], "seed": seed, "ix": int(ix), "end": "?",
                           "floor": 0, "decisions": 0, "screens": Counter(), "teacher_screens": 0, "changed": 0, "teacher_cpu": 0.0, "bosses": [],
                           "k_extended": 0}
    counters = {"reused": 0, "reused_cpu": 0.0}
    boss: dict | None = None
    ordinal = moves = 0
    try:
        while True:
            if time.perf_counter() - t0 > RUN_WALL_CAP_S:
                row["end"] = "timeout"
                break
            d = s.decision()
            if rec.fight is not None and (d is not None or s.phase != engine.rc.PHASE_COMBAT):
                rec.close_fight(s, won=d is not None or s.won)
            if boss is not None and (d is not None or s.phase != engine.rc.PHASE_COMBAT):
                boss["won"] = bool(d is not None or s.won)
                row["bosses"].append(boss)
                boss = None
            if d is None and s.phase != engine.rc.PHASE_COMBAT:
                row["end"] = "win" if s.won else "death"
                break
            if d is not None:
                view = s.view()
                if _J.get("stop_floor") and int(view.floor) >= int(_J["stop_floor"]):
                    row["end"] = "stopped"
                    break
                pick = policy.pick(view, d, explore)
                kind = KIND_OF_TYPE.get(d.type)
                index = pick.index
                if sum(1 for o in d.options if o.kind != "unknown") >= 2:
                    row["screens"][f"{kind}/a{int(view.act_index) + 1}"] += 1
                if eligible(kind, view, d, _J["kinds"]):
                    r = _screen(files, ordinal, d, view, pick, s, seed, counters)
                    index = int(r["teacher"])
                    row["teacher_screens"] += 1
                    row["changed"] += int(index != pick.index)
                    row["teacher_cpu"] += float(r.get("cpu") or 0.0)
                    row["k_extended"] += int(int(r.get("k", 0)) > int(_J["k"]))
                rec.decision(view, d, _taken_pick(pick, index))
                ordinal += 1
                s.apply(d.options[index])
                row["decisions"] += 1
                continue
            if boss is None and s.info["node_type"] == engine.vocab.NODE_BOSS:
                info = s.info
                boss = {"act": int(s.act_index), "floor": int(info["floor"]), "hp": int(info["hp"]), "max_hp": int(info["max_hp"])}
                if player.judge is not None:
                    boss["p_death"] = ((hindsight.door_of(s, player) or {}).get("judge") or {}).get("p_death")
            if rec.fight is None:
                rec.open_fight(s, s.combat())
            moves += player.play_fight(s, rng, None)
        row["floor"] = int(s.info["floor"])
        if row["end"] in ("win", "death"):
            _, calls, _ = gen0.prior_path_of(player.agent, None, None)
            record = rec.finish(s, moves, won=row["end"] == "win", prior_path="cpu", prior_calls=calls)
            write_json(files.record, record)
    finally:
        s.close()
    row["screens"] = dict(row["screens"])
    row["teacher_cpu"] = round(row["teacher_cpu"], 2)
    row["reused_screens"] = counters["reused"]
    row["cpu"] = round(time.process_time() - c0 + counters["reused_cpu"], 2)
    row["wall"] = round(time.perf_counter() - t0, 2)
    jobs.append_ledger(_J["ledger"], {"ev": "end", "job": _J["job"], "seed": seed, "cpu": row["cpu"], "screens": row["teacher_screens"]})
    return row


def label_run(task: tuple) -> dict:
    """One stored exploring run, replayed exactly (`hindsight.replay_run`: its streams, checked against its record) and labelled at some of its
    covered screens (`LABEL_RATE` of them by a hash of the seed and the ordinal, at most `LABEL_PER_RUN`; the replay stops at the last): the
    run itself is the stored run, the teacher only reads it. `task` = ("rt-label", seed, ix, row of the stored run)."""
    import numpy as np  # noqa: PLC0415

    import hindsight  # noqa: PLC0415
    from tools.lab import jobs  # noqa: PLC0415

    _, seed, ix, record = task
    why = _gate(_J["est_label_s"], seed)
    if why:
        return _skip_row(task, why)
    t0, c0 = time.perf_counter(), time.process_time()
    files = RunFiles(_J["gen_dir"], seed)
    files.load()
    player = _J["player"]
    replayer = _replayer(tuple(record["_explore"]))
    scratch = np.random.default_rng(0)  # the argmax player draws nothing from it
    counters = {"reused": 0, "reused_cpu": 0.0}
    row: dict[str, Any] = {"_task": ["rt-label", seed, int(ix)], "arm": "label", "source": "stored", "job": _J["job"], "seed": seed, "ix": int(ix),
                           "stored_job": record.get("_job"), "teacher_screens": 0, "changed": 0, "teacher_cpu": 0.0, "k_extended": 0, "screens": {}}
    taken: Counter = Counter()

    def hook(pt: Any) -> bool:
        if int(pt.view.floor) >= LAST_FLOOR:
            return True
        kind = KIND_OF_TYPE.get(pt.decision.type)
        if not eligible(kind, pt.view, pt.decision, _J["kinds"]) or coin(seed, pt.ordinal) >= _J["label_rate"]:
            return False
        pick = player.policy.pick(pt.view, pt.decision, scratch)  # what the current player would take here
        r = _screen(files, pt.ordinal, pt.decision, pt.view, pick, pt.session, seed, counters)
        taken[f"{kind}/a{int(pt.view.act_index) + 1}"] += 1
        row["teacher_screens"] += 1
        row["changed"] += int(int(r["teacher"]) != int(pick.index))
        row["teacher_cpu"] += float(r.get("cpu") or 0.0)
        row["k_extended"] += int(int(r.get("k", 0)) > int(_J["k"]))
        return row["teacher_screens"] >= _J["per_run"]

    res = hindsight.replay_run(seed, int(ix), replayer, record, hook, config="a3-v1", own_schedule=True)
    row["screens"] = dict(taken)
    row["replay"] = res
    row["teacher_cpu"] = round(row["teacher_cpu"], 2)
    row["reused_screens"] = counters["reused"]
    row["floor"] = int(res.get("floor", 0))
    row["end"] = res.get("end")
    row["cpu"] = round(time.process_time() - c0 + counters["reused_cpu"], 2)
    row["wall"] = round(time.perf_counter() - t0, 2)
    jobs.append_ledger(_J["ledger"], {"ev": "end", "job": _J["job"], "seed": seed, "cpu": row["cpu"], "screens": row["teacher_screens"]})
    return row


def job_task(task: tuple) -> dict:
    return new_run(task) if task[0] == "rt-new" else label_run(task)


def job_capped_row(task: tuple, reason: str, seconds: float, prog: dict) -> dict:
    return {**_skip_row(task, reason), "wall": round(seconds, 1)}


# ---------------------------------------------------------------------------------------------------------- the job: one block


def drop_skips(path: Path) -> None:
    """Take the rows the workers wrote for runs they declined to start (the block's time over, the budget spent, a STOP file) out of runs.jsonl:
    they are not outcomes and the runs are asked for again at the next job."""
    rows = read_jsonl(path)
    keep = [r for r in rows if r.get("cap_reason") not in ("block-end", "budget", "stop-file")]
    if len(keep) != len(rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8", newline="\n")


def consolidate(gen_dir: Path, job_name: str, ctx: dict, agent: Any) -> dict:
    """Write the decision shard of every run that finished and is in no earlier shard (including runs an earlier killed job finished), and
    their RunRecords in the lab's shard layout (`jobs/<job>/shards`, which the run-value refit reads). Returns the shard's manifest."""
    import gen_loop  # noqa: PLC0415

    rows = done_rows(gen_dir)
    old = sharded_seeds(gen_dir, but=job_name)
    seeds = list(dict.fromkeys(r["seed"] for r in rows if r["seed"] not in old))  # in the order the runs finished
    sources = dict(Counter(r.get("source", "?") for r in rows if r["seed"] in set(seeds)))
    manifest = shard_manifest(job=job_name, generation=ctx["generation"], identity=ctx["identity"], judge=ctx["judge"], lib=ctx["lib"], tree=ctx["tree"],
                              k=ctx["k"], k_max=ctx["k_max"], max_options=ctx["max_options"], cal=ctx["cal_doc"], sources=sources)
    doc = build_shard(gen_dir, job_name, seeds, manifest)
    recs = [(s, RunFiles(gen_dir, s).record) for s in seeds]
    recs = [(s, p) for s, p in recs if p.is_file()]
    if recs:
        cfg = agent.search_config("a3-v1")
        sink = gen_loop.ShardSink(gen_dir / "jobs" / job_name, agent.generation, config=cfg.name, base_sims=cfg.sims, ascension=10,
                                  agent=agent.describe())
        for _, p in recs:
            sink.add(json.loads(p.read_text(encoding="utf-8")), None)
        sink.close()
    return doc


def store_manifest(gen_dir: Path, ctx: dict) -> dict:
    """The store's index (`manifest.json`): the settings of the latest job and the totals over every shard."""
    shards = []
    for m in sorted((gen_dir / "shards").glob(f"*/{SHARD_MANIFEST}")):
        with contextlib.suppress(OSError, ValueError):
            shards.append(json.loads(m.read_text(encoding="utf-8")))
    totals = Counter()
    for sh in shards:
        for key in ("decisions", "rows", "runs", "refused", "k_extended"):
            totals[key] += int((sh.get("counts") or {}).get(key, 0))
    last = shards[-1] if shards else {}
    doc = {"version": VERSION, "kind": "run-teacher-store", "generation": ctx["generation"], "updated_utc": utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
           "shards": [sh.get("job") for sh in shards], "totals": dict(totals),
           "latest": {k: last.get(k) for k in ("agent", "agent_sha256", "judge", "judge_sha256", "engine", "engine_sha256", "k", "k_max", "m", "beyond",
                                               "guard", "recal", "created_utc")}}
    write_json(gen_dir / SHARD_MANIFEST, clean_path(doc))
    return doc


def stored_screens_done(gen_dir: Path) -> int:
    return sum(int(r.get("teacher_screens", 0)) for r in done_rows(gen_dir) if r.get("source") == "stored")


def job_block(a: argparse.Namespace, cfg: dict, *, hold_fn: Any, store: Path, generation: int, agent_dir: str, name: str) -> dict:
    """One block of the job: adopt what an earlier killed block finished, refit the door's calibration, play new or stored runs on `procs`
    processes until the block's time, the budget or a guard ends it (the runs in flight when the time ends finish), and write the shard."""
    import fight_gym  # noqa: PLC0415
    import gen_loop  # noqa: PLC0415
    import hindsight  # noqa: PLC0415
    from tools.lab import jobs  # noqa: PLC0415

    t_start = time.time()
    gen_dir = store / str(generation)
    job_dir = gen_dir / "jobs" / name
    job_dir.mkdir(parents=True, exist_ok=True)
    target = gen_dir / "runs.jsonl"
    ledger = jobs.teacher_ledger(store)
    agent = gen_loop.load_agent(agent_dir)
    res: dict[str, Any] = {"job": name, "generation": generation, "started_utc": utc().strftime("%Y-%m-%dT%H:%M:%SZ"), "runs": 0, "decisions": 0}
    jobs.append_ledger(ledger, {"ev": "job_start", "job": name})
    try:
        # the door's recalibration on the player's own door outcomes so far
        pairs = door_pairs(done_rows(gen_dir))
        if a.labels_calibration and Path(a.labels_calibration).is_dir():
            pairs += label_door_pairs(hindsight.read_labels(Path(a.labels_calibration)))
        cal_doc = fit_door_calibration(pairs)
        write_json(gen_dir / "calibration.json", {"version": VERSION, "fit_utc": res["started_utc"], "min_doors": CAL_MIN_DOORS, "ridge": CAL_RIDGE,
                                                  "doors": len(pairs), "acts": {str(k): v for k, v in cal_doc.items()}})
        done = {r["seed"] for r in done_rows(gen_dir)}
        mode = a.source
        stored_rows: list[dict] = []
        if mode in ("auto", "stored"):
            stored_rows = hindsight.exploring_runs(Path(a.runs_root), generation, config=a.stored_config)
            mode = choose_source(mode, stored_screens_done(gen_dir), len(stored_tasks(stored_rows, done, 1, a.stored_from, a.stored_to)))
        tasks = (stored_tasks(stored_rows, done, a.max_runs, a.stored_from, a.stored_to) if mode == "stored"
                 else new_tasks(generation, done, a.max_runs))
        res.update(source=mode, tasks=len(tasks))
        lib = str(a.lib or ROOT / "out")
        worker_cfg = {"agent_dir": str(agent_dir), "replay_agent": str(a.replay_agent or agent_dir), "judge": a.judge, "lib": lib, "gen_dir": str(gen_dir),
                      "ledger": str(ledger), "job": name, "k": a.k, "k_max": a.k_max, "max_options": a.max_options, "cal": cal_doc,
                      "deadline": t_start + a.block_min * 60.0, "budget_h": float(cfg.get("core_hours_per_day", 24.0)),
                      "stop_file": str(job_dir / "STOP"), "stop_floor": a.stop_floor, "label_rate": a.label_rate, "per_run": a.per_run,
                      "kinds": kinds_of(a.kinds),
                      "est_new_s": jobs.TEACHER_EST_CPU_S, "est_label_s": 60.0 + 45.0 * a.per_run}
        if mode == "stored" and gen_loop.load_agent(worker_cfg["replay_agent"]).generation != generation:
            raise SystemExit("the stored runs are of generation %d: --replay-agent must be that generation's player" % generation)
        why = hold_fn(True)
        if why:
            res["held"] = why
            tasks = []
        capped0 = sum(1 for r in read_jsonl(target) if r.get("capped"))
        if tasks:
            reason = fight_gym.run_tasks(tasks, job_task, target, a.procs, hard_of=lambda t: RUN_WALL_CAP_S + 600, capped_of=job_capped_row,
                                         init=_init_job_worker, init_args=(worker_cfg,),
                                         stop=lambda: "STOP file" if (job_dir / "STOP").exists() else hold_fn(False))
            drop_skips(target)
            res["capped"] = sum(1 for r in read_jsonl(target) if r.get("capped")) - capped0  # runs lost to a worker that died or ran too long
            if reason:
                res["stopped"] = reason
            done = {r["seed"] for r in done_rows(gen_dir)}
            tasks = [t for t in tasks if t[1] not in done]
        ctx = {"generation": generation, "identity": agent_identity(agent_dir, agent), "judge": a.judge, "lib": lib, "tree": _git_head(),
               "k": a.k, "k_max": a.k_max, "max_options": a.max_options, "cal_doc": {str(k): v for k, v in cal_doc.items()}}
        manifest = consolidate(gen_dir, name, ctx, agent)
        store_manifest(gen_dir, ctx)
        mine = [r for r in done_rows(gen_dir) if r.get("job") == name]
        res.update(runs=len(mine), decisions=sum(int(r.get("teacher_screens", 0)) for r in mine), shard_decisions=manifest["counts"]["decisions"],
                   shard_rows=manifest["counts"]["rows"], cpu_s=round(sum(float(r.get("cpu", 0.0)) for r in mine), 1),
                   wall_s=round(time.time() - t_start, 1), tasks_left=len(tasks))
        res["decisions_s"] = round(res["decisions"] / res["wall_s"], 4) if res["wall_s"] else None
        res["budget_left_h"] = round(jobs.teacher_budget_left_h(cfg, jobs.read_ledger(ledger)), 2)
        write_json(gen_dir / "report.json", report(gen_dir))
    finally:
        jobs.append_ledger(ledger, {"ev": "job_end", "job": name})
        write_json(job_dir / "job.json", clean_path({**res, "finished_utc": utc().strftime("%Y-%m-%dT%H:%M:%SZ")}))
    return res


def _git_head() -> str | None:
    import subprocess  # noqa: PLC0415

    with contextlib.suppress(Exception):
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip() or None
    return None


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


def read_json_file(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def take_store_lock(store: Path) -> str | None:
    """Take the store's job lock (`<store>/job.lock`: this process's pid and the time): None when it is ours now, else why not (the job that holds
    it is still running). A lock whose process is gone is taken over. Two blocks on one store would play the same runs twice."""
    from tools.lab import winsys  # noqa: PLC0415

    lock = Path(store) / "job.lock"
    doc = read_json_file(lock)
    if isinstance(doc, dict):
        pid = int(doc.get("pid") or 0)
        if pid and pid != os.getpid() and winsys.parent_alive(pid):
            return f"another teacher job holds the store (pid {pid})"
    write_json(lock, {"pid": os.getpid(), "t": utc().strftime("%Y-%m-%dT%H:%M:%SZ")})
    return None


def drop_store_lock(store: Path) -> None:
    lock = Path(store) / "job.lock"
    doc = read_json_file(lock)
    if isinstance(doc, dict) and int(doc.get("pid") or 0) == os.getpid():
        lock.unlink(missing_ok=True)


def kinds_of(text: str) -> tuple[str, ...]:
    """The screen kinds the job covers: `all`, a comma list, or `none` (the player's own runs, recorded: a cheap source of door outcomes)."""
    if text.strip() in ("", "all"):
        return TEACHER_KINDS
    if text.strip() == "none":
        return ()
    names = tuple(x.strip() for x in text.split(",") if x.strip())
    bad = [x for x in names if x not in TEACHER_KINDS]
    if bad:
        raise SystemExit(f"unknown screen kind {bad[0]!r}; one of {', '.join(TEACHER_KINDS)}")
    return names


def job_cmd(a: argparse.Namespace) -> int:
    """The teacher as a lab job: one block of runs (or, with --loop, block after block) under the filler's rules; the shard, the run records,
    the store's manifest and `result.json` are written whether the block played or was held."""
    from tools.lab import jobs  # noqa: PLC0415

    with contextlib.suppress(Exception):
        from sts2_gym import procutil  # noqa: PLC0415

        procutil.lower_priority()  # BelowNormal (Windows priority class); the workers lower themselves
    lab, store = Path(a.lab), Path(a.store)
    cur = read_json_file(Path(a.agent_root) / "current.json") or {}
    agent_dir = a.agent or cur.get("dir")
    if not agent_dir:
        print("run-teacher job: no player (agent/current.json has no `dir`; give --agent)", flush=True)
        return 2
    if a.agent:
        import gen_loop  # noqa: PLC0415

        generation = int(gen_loop.load_agent(agent_dir).generation)
    else:
        generation = int(cur["generation"])
    cfg = dict(jobs.config_from(jobs.read_lenient(lab / "jobs.json"))[jobs.TEACHER])
    cfg["store"] = str(store)
    if a.budget_h is not None:
        cfg["core_hours_per_day"] = float(a.budget_h)
    if a.share:
        def hold_fn(starting: bool) -> str | None:
            if (lab / "HALT").exists() or (lab / "PAUSE").exists():
                return "LAB/HALT or LAB/PAUSE is up"
            return "nightly-active" if jobs.nightly_active(lab) or ACTIVE.exists() else None
    else:
        def hold_fn(starting: bool) -> str | None:
            return jobs.teacher_hold(lab, cfg, starting=starting)
    mode = "stored" if a.cmd == "label" else a.source
    a.source = mode
    while True:
        name = a.name or utc().strftime("%Y%m%dT%H%M") + "-teacher"
        why = hold_fn(True) or take_store_lock(store)
        if why:
            print(f"run-teacher job: held ({why}) {utc():%H:%M:%SZ}", flush=True)
            res = {"job": name, "generation": generation, "runs": 0, "decisions": 0, "held": why}
        else:
            print(f"run-teacher job {name}: generation {generation}, source {mode}, K {a.k}..{a.k_max}, {a.procs} processes, block {a.block_min:g} min"
                  f"{', sharing the machine' if a.share else ''}", flush=True)
            try:
                res = job_block(a, cfg, hold_fn=hold_fn, store=store, generation=generation, agent_dir=str(agent_dir), name=name)
            finally:
                drop_store_lock(store)
            print("run-teacher job: " + json.dumps(clean_path(res)), flush=True)
        if a.result:
            write_json(Path(a.result), clean_path(res))
        if not a.loop:
            return 1 if res.get("runs", 0) == 0 and res.get("capped", 0) >= 3 else 0  # a job that lost every run it tried has failed (the runner's alert reads rc)
        a.name = None
        time.sleep(60 if res.get("runs", 0) == 0 else 5)


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
    for name, help_ in (("job", "the teacher as a lab job: shards in the store, under the filler's rules"),
                        ("label", "round 0: replay the stored exploring runs and label their screens (job --source stored)")):
        jp = sub.add_parser(name, help=help_)
        jp.add_argument("--lab", default=DEFAULT_LAB, help="the lab folder (jobs.json, the night's markers, HALT/PAUSE)")
        jp.add_argument("--store", default=DEFAULT_STORE, help="the store: <store>/<generation>/ holds runs, shards and the manifest")
        jp.add_argument("--agent-root", default=DEFAULT_AGENT_ROOT, help="where agent/current.json names the player")
        jp.add_argument("--agent", default="", help="a player's folder instead of current.json's")
        jp.add_argument("--replay-agent", default="", help="source stored: the player that played the stored runs (default: the player)")
        jp.add_argument("--judge", default=DEFAULT_JUDGE)
        jp.add_argument("--lib", default="", help="the engine's native folder (default: this tree's out/)")
        jp.add_argument("--source", choices=("auto", "new", "stored"), default="auto",
                        help="new: whole runs the teacher plays; stored: replay the stored exploring runs; auto: stored until it has yielded STORED_TARGET screens")
        jp.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT, help="the lab's job folders (the stored exploring runs)")
        jp.add_argument("--stored-from", default=STORED_FROM)
        jp.add_argument("--stored-to", default=STORED_TO)
        jp.add_argument("--stored-config", default=STORED_CONFIG)
        jp.add_argument("--label-rate", type=float, default=LABEL_RATE)
        jp.add_argument("--per-run", type=int, default=LABEL_PER_RUN)
        jp.add_argument("--procs", type=int, default=4)
        jp.add_argument("--block-min", type=float, default=60.0, help="no run starts after this long; the runs in flight finish")
        jp.add_argument("--budget-h", type=float, default=None, help="core-hours a UTC day (default: jobs.json teacher.core_hours_per_day, 24)")
        jp.add_argument("--k", type=int, default=JOB_K, help="draws of every option before the first look")
        jp.add_argument("--k-max", type=int, default=JOB_K_MAX, help="draws at most (equal to --k: fixed K)")
        jp.add_argument("--max-options", type=int, default=DEFAULT_MAX_OPTIONS)
        jp.add_argument("--max-runs", type=int, default=400, help="runs asked of one block (the budget and the block's time end it earlier)")
        jp.add_argument("--kinds", default="all", help="screen kinds the teacher covers: all, a comma list, or none (plain player runs)")
        jp.add_argument("--stop-floor", type=int, default=0, help="end a new run at this floor (a smoke test); 0: play it out")
        jp.add_argument("--share", action="store_true", help="the machine is shared by order: honour only HALT, PAUSE and the night")
        jp.add_argument("--loop", action="store_true", help="block after block; wait while held")
        jp.add_argument("--result", default="", help="write the block's result.json here")
        jp.add_argument("--name", default="", help="the job's name (default: the UTC minute and -teacher)")
        jp.add_argument("--labels-calibration", default="", help="stored hindsight labels (a labels.d's parent) whose door outcomes seed the calibration")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return {"play": play_cmd, "report": report_cmd, "relabel": relabel_cmd, "job": job_cmd, "label": job_cmd}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
