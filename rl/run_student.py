#!/usr/bin/env python3
"""The run student `pi_s` and its gate (D56, D57 part 3; lane 3 of docs/rl/RUN_SELF_PLAY.md section 10; docs/rl/RUN_STUDENT.md).

    python rl/run_student.py check  --shards-root DIR [--prior weights.npz] [--map-grid]
    python rl/run_student.py fit    --shards-root DIR --prior weights.npz --out DIR [--human-shards DIR] [--beta 1 --kl 0.1 --epochs 8]
                                    [--map-grid [--teacher-prior weights.npz]]
    python rl/run_student.py candidate --out DIR --base CURRENT --agent-root DIR --target N
    python rl/run_student.py round  --shards-root DIR --prior weights.npz --out DIR --human-shards DIR --base CURRENT --agent-root DIR --target N

The run teacher (`rl/run_teacher.py`) plays each option of a run-level screen forward on resampled clones of the run and records, per
screen, a decision record and the options' judgement rows (`rows.npz`: the 1,863-wide vectors `human_prior_infer.option_rows` builds at
play). The student learns from those records, and from the human choices the human prior was fitted on, and takes the prior's place kind by
kind where it passes its two held-out legs.

**The fit.** The net is the human prior's (`PriorNet`: the run judgement's trunk with one logit per option row), started from the prior's
weights. A step takes a batch of teacher decisions and a batch of human decisions of one kind and minimises

    CE(softmax((Q + lambda log pi_h) / tau_T) over the played options, softmax of pi_s over the same options)
      + beta x CE(human choice, pi_s)  +  kl x KL(pi_h || pi_s)   (over all the screen's options, on both batches)

`Q` is the teacher's mean play-out value of an option (floors) and `lambda` the lambda the teacher decided that screen with (the player's
lambda where the player blends the prior, 0 elsewhere); `tau_T` is the sd of the mean of the K draws of a centred option value, pooled by kind
and act (`run_teacher.noise_and_spread`), so a decision the play-outs cannot separate gives a flat target. Options off the shortlist have no
`Q` and carry no target. The human term is the prior's own loss per kind (card, ancient, rest, shop, route; event as a softmax), over
the training fifths (1 to 3), the exam fifth never read (`check_no_exam`); kinds the human record has no fit for (reward, select) have only
the teacher's term.

**The legs**, on data the fit never saw. *Leg 1*, teacher imitation: a fifth of the runs by a hash of the game seed (`split_of`) is held
out; per kind the student's cross-entropy to the teacher's target against the prior's, paired by decision, the gain with a 95% interval by
resampling whole runs (passes when the lower bound is above 0). *Leg 2*, the human bar: per kind with human choices, fifth 4's log-loss
of the student against the prior's, on won runs (at most +0.002 nats worse) and all runs (at most +0.02), the margins of D55 addendum 15
(`strong_sets.TIE_NATS`, `LIMIT_B_NATS`); a kind with no human fit hands over on leg 1 alone. *Lambda* per kind and act is measured on the
leg-1 fifth with `run_teacher.lambda_fit` (how many floors of the teacher's value the student's log-probability explains beyond the
judgement's V), used as `V + lambda log pi_s`; no knob.

A kind that passes both legs, with a lambda, is given to the student in the candidate (`candidate`): recipe `vrun+pi_s:...`
(`macro_recipe.student_name`), the human prior's blend kept on every other kind, the legs stored beside the student
(run_student_legs.json) for `screens.run_student`.

**The map grid** (D58 addendum 13, `--map-grid`): the prior is the v2 human prior (a map tower over block M, the whole act map: `docs/rl/HUMAN_PRIOR.md`
"v2"), the teacher's rows each come with their grid (`rows_map.npz` beside `rows.npz`; `run_teacher.py reencode` adds it to older shards), the human
term reads the v2 decision shards (`decisions-v2`) and the student carries the tower, started from the prior's. The flag and the prior must agree: a
fit with it refuses a prior without the tower and shards without the grid; a fit without it refuses a prior that has the tower. The target's
`lambda log pi_h` term is the prior the teacher decided with (`--teacher-prior`, default the fit's prior): the recorded log-probabilities must be
that prior's own; the student starts from, is anchored to (KL) and is measured against `--prior`.

**The state gaps** (D58 addenda 14 and 16, `--state-gaps`): the prior is the v3 human prior (block G's columns close the first layer's input: the relic
counters, the "?" rooms' odds, the merchant's removals bought and owed, and a known bit for each part: `docs/rl/HUMAN_PRIOR.md` "v3"), the teacher's rows each
come with their block G (`rows_gaps.npz` beside `rows.npz`, written from the engine's own state by replay; `run_teacher.py reencode` adds it to older shards),
the human term reads the v3 decision shards (`decisions-v3`, where the counters are unknown: known bit 0) and the student carries the columns, started from
the prior's. The two flags are independent (a v3 prior may carry the tower too: then both are given) and each is checked as `--map-grid` is: the flag and
the prior must agree, a shard set without the side file (or with one written for another block G version or another row count) is refused.

Lab code: nothing here changes the engine, the search, the lab's self-play, the loop or the night. Teacher shards hold game seeds, never a
human record; the human term reads only the prior's shards of fifths 1 to 4 and reports aggregates only (cells under 20 runs hidden).
Source of the human record: Spire Codex public run export for v0.107.1.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import human_prior as hp  # noqa: E402
import human_starts as hs  # noqa: E402
import macro_recipe  # noqa: E402
from human_prior_infer import PriorScorer, file_sha256, log_softmax  # noqa: E402

VERSION = 1
SPIRE_CODEX = hp.SPIRE_CODEX
KINDS = macro_recipe.STUDENT_KINDS
KIND_ID = {k: i for i, k in enumerate(KINDS)}
HUMAN_KINDS = ("card", "ancient", "rest", "shop", "route", "event")  # the kinds the human record has choices for
ROWS_WIDTH = 1863  # human_prior_infer.option_rows: the base vector and the fight block
DENSE_WIDTH, COUNTS_WIDTH = 350, 1513
DECISIONS_FILE, ROWS_FILE, MANIFEST_FILE = "decisions.jsonl", "rows.npz", "manifest.json"
ROWS_GAPS_FILE = "rows_gaps.npz"  # block G of each row of rows.npz (`gaps` float32 (rows, GAPS_WIDTH)), written by the teacher from the state gaps on (manifest `state_gaps`)
GAPS_WIDTH = 36  # features.GAPS_SIZE (a test holds them equal)
GAPS_VERSION = 1  # features.GAPS_VERSION
ROWS_MAP_FILE = "rows_map.npz"  # the map grid of each row of rows.npz (`grid` int8 (rows, GRID_WIDTH)), written by the teacher from the map grid on
GRID_WIDTH = 17 * 7 * 13  # features.MAP_GRID_SIZE (a test holds them equal)
LEGS_FILE, FIT_FILE, STUDENT_OUT = "legs.json", "fit.json", "student.npz"

# the legs and the hand-over rule
TEST_FIFTH = 0  # leg 1: the runs whose seed hash falls here (a fifth) are held out of the fit
SPLIT_SALT = "spireline-run-student-split|"
VAL_EVERY = 8  # of the other runs one in this many stops the fit early (a tenth of all)
LEG1_MIN_DECISIONS = 100  # held-out decisions a kind needs for a leg-1 verdict
TIE_NATS = 0.002  # leg 2, won runs: the student may be worse than the prior by at most this (D55 addendum 15's tie margin)
LIMIT_NATS = 0.02  # leg 2, all runs: ... by at most this
MIN_CELL = hp.MIN_CELL  # a human cell over fewer runs is not shown (D13)
MIN_LAMBDA_DECISIONS = 30  # held-out decisions a lambda cell needs; fewer falls back to the kind's pooled, then the kind has no lambda
LAMBDA_CAP = 2.0  # a measured lambda above this is clipped (the recipes' base is 0.5)
TAU_MIN = 0.25  # floors: the target's temperature never goes below this
# The same floor in each grade's own unit (D59): a store is graded one way (`run_teacher.py regrade`), and a temperature measured in one grade's unit is held back by
# that grade's floor. The frontier's is the floors' scaled by the ratio of the noise sd of one draw's centred option value in the two grades on the same screens
# (`run_teacher.unit_constants`, `tau_min_suggested`): 0.568 / 2.010 (the two regraded stores, 13,551 screens, 2026-11-13, A2 `a2.npz` sha256 a5d3da24) x 0.25 = 0.071. A grade
# whose unit is not measured (None) is refused, not fitted with a floors number.
FRONTIER_TAU_MIN = 0.071
TAU_MIN_BY_GRADE: dict[str, float | None] = {"floors": TAU_MIN, "frontier": FRONTIER_TAU_MIN}
GRADES = ("floors", "frontier")
TAU_CELL_MIN = 20  # decisions a (kind, act) cell needs for its own noise estimate; fewer takes the kind's, then all
BOOT_DRAWS = 2000
HUMAN_TRAIN_FIFTHS = (1, 2, 3)
HUMAN_TEST_FIFTH = 4


class ShardError(ValueError):
    """A decision shard the loader refuses: it is not read, never repaired."""


# ---------------------------------------------------------------------------------------------------------- lane 2's functions
# `lambda_fit` and `noise_and_spread` are run_teacher's (rl/run_teacher.py, lane 2): the student measures lambda and the targets' temperature
# with the teacher's own arithmetic, imported at the call, so the two can never disagree.


def teacher_module() -> Any:
    import run_teacher  # noqa: PLC0415

    return run_teacher


def teacher_fn(name: str) -> Any:
    """`run_teacher.<name>` (`lambda_fit`, `noise_and_spread`)."""
    return getattr(teacher_module(), name)


# ---------------------------------------------------------------------------------------------------------- the split


def tau_min_for(grade: str) -> float:
    """The floor of the targets' temperature in `grade`'s unit; a grade whose unit is not measured is refused (`ShardError`)."""
    if grade not in TAU_MIN_BY_GRADE:
        raise ShardError(f"unknown grade {grade!r}; one of {', '.join(GRADES)}")
    if grade == "floors":
        return TAU_MIN  # the module's own, so that a test or a probe that sets it is read
    got = TAU_MIN_BY_GRADE[grade]
    if got is None:
        raise ShardError(f"TAU_MIN is not measured for the {grade} grade (run_teacher.py units on a regraded store, then set TAU_MIN_BY_GRADE): no fit on it yet")
    return float(got)


def seed_hash(seed: str, salt: str = SPLIT_SALT) -> int:
    return int(hashlib.sha256((salt + str(seed)).encode("utf-8")).hexdigest()[:8], 16)


def split_of(seed: str) -> str:
    """Which part a run is in, by a fixed hash of its game seed: `test` (a fifth, leg 1's held-out runs), `val` (a tenth of all runs: the fit
    stops early on it) or `train`. A run is in one part whatever else is in the data."""
    h = seed_hash(seed)
    if h % 5 == TEST_FIFTH:
        return "test"
    return "val" if (h // 5) % VAL_EVERY == 0 else "train"


# ---------------------------------------------------------------------------------------------------------- the targets


def teacher_target(q: Sequence[float], logp: Sequence[float] | None, lam: float, tau: float) -> np.ndarray:
    """The teacher's distribution over the played options: `softmax((Q + lambda log pi_h) / tau)`. `q` and `logp` (the prior's
    log-probabilities of the same options; None with lambda 0) are aligned. tau is in floors (> 0)."""
    if not tau > 0:
        raise ValueError(f"the target's temperature is positive, not {tau}")
    q = np.asarray(q, dtype=np.float64)
    score = q if not lam else q + float(lam) * np.asarray(logp, dtype=np.float64)
    z = (score - score.max()) / tau
    e = np.exp(z)
    return e / e.sum()


def tau_table(g_by_cell: Mapping[tuple[str, int], Sequence[Sequence[Sequence[float]]]]) -> dict[str, Any]:
    """The noise sd (floors) of one draw's centred option value, pooled by (kind, act) where a cell has `TAU_CELL_MIN` decisions, else by kind,
    else over everything: {"cells": {"kind/aN": sd}, "kinds": {...}, "all": sd}. `g_by_cell[(kind, act)]` lists each decision's G[option][draw]."""
    ns = teacher_fn("noise_and_spread")
    by_kind: dict[str, list] = defaultdict(list)
    out: dict[str, Any] = {"cells": {}, "kinds": {}, "all": None, "decisions": {}}
    for (kind, act), gs in g_by_cell.items():
        by_kind[kind].extend(gs)
        r = ns(gs)
        out["decisions"][f"{kind}/a{act + 1}"] = int(r["decisions"])
        if r["decisions"] >= TAU_CELL_MIN and r["noise_sd"]:
            out["cells"][f"{kind}/a{act + 1}"] = float(r["noise_sd"])
    for kind, gs in by_kind.items():
        r = ns(gs)
        if r["decisions"] >= TAU_CELL_MIN and r["noise_sd"]:
            out["kinds"][kind] = float(r["noise_sd"])
    r = ns([g for gs in by_kind.values() for g in gs])
    out["all"] = float(r["noise_sd"]) if r["noise_sd"] else None
    return out


def noise_sd_for(table: Mapping[str, Any], kind: str, act: int) -> float | None:
    got = table["cells"].get(f"{kind}/a{act + 1}") or table["kinds"].get(kind) or table["all"]
    return None if got is None else float(got)


# ---------------------------------------------------------------------------------------------------------- the shards


@dataclasses.dataclass
class TeacherSet:
    """The teacher's decisions of every shard read, complete ones only (every option has a row, at least two options have a value).
    Per decision arrays (`n` of them) and per row arrays (`rows` of them, decision after decision, option order)."""

    seeds: list[str]  # run ids: game seeds of the teacher's runs (never a human record)
    run: np.ndarray  # (n,) int32 index into `seeds`
    kind: np.ndarray  # (n,) int8 index into KINDS
    act: np.ndarray  # (n,) int8 0-based
    floor: np.ndarray  # (n,) int16
    k: np.ndarray  # (n,) int16 draws per option
    lam: np.ndarray  # (n,) float32 the lambda the teacher decided the screen with (0 where the player does not blend the prior)
    teacher: np.ndarray  # (n,) int16 the option the guarded teacher took
    player: np.ndarray  # (n,) int16 the option the player took
    start: np.ndarray  # (n,) int64 first row
    count: np.ndarray  # (n,) int32 rows (= options)
    q: np.ndarray  # (rows,) float32 the teacher's value of a played option, NaN for the others
    v: np.ndarray  # (rows,) float32 the judgement's V of the option
    dense: np.ndarray  # (rows, 350) float32
    counts: np.ndarray  # (rows, 1513) float32
    g: list  # per decision, G[played option][draw] (floors), for the noise of a draw
    logp_rec: np.ndarray  # (rows,) float32 the log-probabilities the teacher recorded on the screens it blends (NaN elsewhere)
    stats: dict[str, Any]
    manifests: list[dict[str, Any]]
    zh: np.ndarray | None = None  # (rows,) float32 the prior's logits, once `attach_prior` ran
    tau: np.ndarray | None = None  # (n,) float32 the target's temperature
    noise: dict[str, Any] | None = None
    grid: np.ndarray | None = None  # (rows, GRID_WIDTH) int8 block M of each row, when every shard read has `rows_map.npz` (a map-grid fit needs it)
    zt: np.ndarray | None = None  # (rows,) float32 the logits of the prior the teacher decided with, when it is not `zh`'s (`attach_prior(teacher_prior=)`)
    gaps: np.ndarray | None = None  # (rows, GAPS_WIDTH) float32 block G of each row, when every shard read has `rows_gaps.npz` (a state-gaps fit needs it)

    def __len__(self) -> int:
        return len(self.kind)

    def kind_name(self, i: int) -> str:
        return KINDS[int(self.kind[i])]

    def run_split(self) -> np.ndarray:
        """The part of each run (`test`, `val`, `train`)."""
        return np.array([split_of(s) for s in self.seeds])

    def split_idx(self) -> dict[str, np.ndarray]:
        part = self.run_split()[self.run] if len(self) else np.array([])
        return {name: np.flatnonzero(part == name) for name in ("train", "val", "test")}

    def matrix(self, idx: np.ndarray) -> np.ndarray:
        """Padded (decisions x options) row indices of the decisions `idx`, -1 for pads."""
        idx = np.asarray(idx, dtype=np.int64)
        width = int(self.count[idx].max()) if len(idx) else 1
        m = np.full((len(idx), width), -1, dtype=np.int64)
        for j in range(width):
            ok = self.count[idx] > j
            m[ok, j] = self.start[idx][ok] + j
        return m

    def logits_of(self, scorer: PriorScorer) -> np.ndarray:
        """`scorer`'s logit of every row (a scorer with the map tower is given each row's grid, one with the state gaps each row's block G, and either refuses a
        set that lacks what it reads)."""
        out = np.zeros(len(self.dense), dtype=np.float32)
        if scorer.map_grid and self.grid is None:
            raise ShardError("this prior has a map tower and the shards have no map grid (rows_map.npz): run `run_teacher.py reencode`")
        if getattr(scorer, "state_gaps", False) and self.gaps is None:
            raise ShardError("this prior reads the state gaps and the shards have none (rows_gaps.npz): run `run_teacher.py reencode`")
        for s in range(0, len(self.dense), 20000):
            out[s:s + 20000] = scorer.logits(self.dense[s:s + 20000], self.counts[s:s + 20000], None if self.grid is None else self.grid[s:s + 20000],
                                             self.gaps[s:s + 20000] if getattr(scorer, "state_gaps", False) else None)
        return out

    def attach_prior(self, prior: PriorScorer, tol: float = 1e-3, teacher_prior: PriorScorer | None = None) -> None:
        """Score every row with the prior (`zh`) and check that the log-probabilities the teacher recorded on the screens it blends are the
        teacher's prior's (a shard played with another prior is refused): the target `Q + lambda log pi_h` must use the prior the teacher used.
        That is `prior`, unless `teacher_prior` names the one the teacher played with (a map-grid fit starts from the v2 prior while the shards were
        played with the plain one): the target then uses its log-probabilities (`zt`) and `prior` is the student's start, anchor and baseline."""
        self.zh = self.logits_of(prior)
        self.zt = None if teacher_prior is None else self.logits_of(teacher_prior)
        zt = self.zh if self.zt is None else self.zt
        worst, checked = 0.0, 0
        have = ~np.isnan(self.logp_rec)
        for i in range(len(self)):
            a, b = int(self.start[i]), int(self.start[i] + self.count[i])
            if have[a:b].all():
                checked += 1
                worst = max(worst, float(np.abs(log_softmax(zt[a:b]) - self.logp_rec[a:b]).max()))
        if worst > tol:
            raise ShardError(f"the log-probabilities the teacher recorded differ from {'the teacher prior' if teacher_prior is not None else 'this prior'}'s by up to "
                             f"{worst:.4f}: the shards were played with another human prior than the one given")
        self.stats["logp_check"] = {"max_abs_difference": round(worst, 6), "decisions_checked": checked}

    def set_tau(self) -> dict[str, Any]:
        """The target's temperature per decision: the pooled noise sd of one draw (`tau_table`) over the square root of the decision's K."""
        cells: dict[tuple[str, int], list] = defaultdict(list)
        for i in range(len(self)):
            cells[(self.kind_name(i), int(self.act[i]))].append(self.g[i])
        self.noise = tau_table(cells)
        floor = tau_min_for(str(self.stats.get("grade", "floors")))
        sd = np.array([noise_sd_for(self.noise, self.kind_name(i), int(self.act[i])) or floor for i in range(len(self))], dtype=np.float64)
        self.tau = np.maximum(sd / np.sqrt(np.maximum(self.k, 1)), floor).astype(np.float32)
        return self.noise

    def targets(self, lam_of: Mapping[tuple[str, int], float] | None = None) -> np.ndarray:
        """(rows,) the teacher's distribution over each decision's played options (zeros elsewhere); the lambda of a decision is the
        teacher's own (`lam`), or `lam_of[(kind, act)]` where given. Needs `attach_prior` and `set_tau`."""
        if self.zh is None or self.tau is None:
            raise ValueError("targets need the prior's logits (attach_prior) and the temperature (set_tau)")
        t = np.zeros(len(self.q), dtype=np.float32)
        for i in range(len(self)):
            a, b = int(self.start[i]), int(self.start[i] + self.count[i])
            q = self.q[a:b]
            played = ~np.isnan(q)
            lam = float(self.lam[i]) if lam_of is None else float(lam_of.get((self.kind_name(i), int(self.act[i])), self.lam[i]))
            logp = log_softmax((self.zh if self.zt is None else self.zt)[a:b]) if lam else None
            t[a:b][played] = teacher_target(q[played], None if logp is None else logp[played], lam, float(self.tau[i]))
        return t


def find_shards(root: Path | str) -> list[Path]:
    """The folders under `root` (or `root` itself) that hold a `decisions.jsonl`, in name order."""
    root = Path(root)
    if (root / DECISIONS_FILE).is_file():
        return [root]
    return sorted(p.parent for p in root.rglob(DECISIONS_FILE))


def _clean(value: Any) -> Any:
    """A manifest value without absolute paths (the last two parts of a path stay)."""
    if isinstance(value, str) and (value[1:3] in (":/", ":\\") or value.startswith("/")):
        return "/".join(Path(value.replace("\\", "/")).parts[-2:])
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def read_shard(d: Path | str) -> dict[str, Any]:
    """One shard folder: its manifest, decision records, and the rows with their decision and option. Refuses (ShardError) a missing file, a
    schema version it does not know, rows of another width, non-finite numbers, indices out of range, an option twice in a decision, a
    `row0` that does not point at the decision's first row, and counts in the manifest that the data does not have."""
    d = Path(d)
    for name in (DECISIONS_FILE, ROWS_FILE, MANIFEST_FILE):
        if not (d / name).is_file():
            raise ShardError(f"{d.name}: no {name}")
    try:
        manifest = json.loads((d / MANIFEST_FILE).read_text(encoding="utf-8"))
    except ValueError as e:
        raise ShardError(f"{d.name}: manifest.json is not JSON ({e})") from e
    if not isinstance(manifest, dict):
        raise ShardError(f"{d.name}: manifest.json is not an object")
    lines = [ln for ln in (d / DECISIONS_FILE).read_text(encoding="utf-8").splitlines() if ln.strip()]
    recs: list[dict] = []
    tail_cut = 0
    for i, ln in enumerate(lines):
        try:
            r = json.loads(ln)
        except ValueError as e:
            if i == len(lines) - 1:
                tail_cut = 1  # the last line of a shard cut by a stop
                break
            raise ShardError(f"{d.name}: line {i + 1} of {DECISIONS_FILE} is not JSON") from e
        if not isinstance(r, dict) or r.get("v") != VERSION:
            raise ShardError(f"{d.name}: line {i + 1} is not a version {VERSION} decision record")
        recs.append(r)
    with np.load(d / ROWS_FILE, allow_pickle=False) as z:
        missing = [k for k in ("rows", "decision", "option") if k not in z.files]
        if missing:
            raise ShardError(f"{d.name}: {ROWS_FILE} lacks {missing}")
        rows, dec, opt = z["rows"], z["decision"].astype(np.int64), z["option"].astype(np.int64)
    if rows.ndim != 2 or rows.shape[1] != ROWS_WIDTH or rows.dtype.kind != "f":
        raise ShardError(f"{d.name}: rows are {rows.shape} {rows.dtype}, not (n, {ROWS_WIDTH}) floats")
    if not (len(dec) == len(opt) == len(rows)):
        raise ShardError(f"{d.name}: rows, decision and option differ in length")
    if len(rows) and not np.isfinite(rows).all():
        raise ShardError(f"{d.name}: rows hold a number that is not finite")
    if len(dec) and (dec.min() < 0 or dec.max() >= len(recs)):
        raise ShardError(f"{d.name}: a row names a decision outside the {len(recs)} records")
    if len(opt) and opt.min() < 0:
        raise ShardError(f"{d.name}: a row names a negative option")
    n_opts = np.array([int(r.get("n_options", -1)) for r in recs], dtype=np.int64)
    if len(dec) and (opt >= n_opts[dec]).any():
        raise ShardError(f"{d.name}: a row names an option past its decision's n_options")
    order = np.lexsort((opt, dec))
    sd, so = dec[order], opt[order]
    if len(sd) > 1 and ((sd[1:] == sd[:-1]) & (so[1:] == so[:-1])).any():
        raise ShardError(f"{d.name}: an option has two rows in one decision")
    counts = manifest.get("counts")
    if isinstance(counts, dict):
        for key, got in (("decisions", len(recs) + tail_cut), ("rows", len(rows))):
            if isinstance(counts.get(key), int) and counts[key] != got:
                raise ShardError(f"{d.name}: the manifest counts {counts[key]} {key}, the shard has {got}")
    first = np.searchsorted(sd, np.arange(len(recs)), side="left")
    for i, r in enumerate(recs):
        if "row0" in r and r["row0"] is not None and first[i] < len(sd) and sd[first[i]] == i and int(r["row0"]) != int(order[first[i]]):
            raise ShardError(f"{d.name}: record {i + 1}'s row0 {r['row0']} is not its first row {int(order[first[i]])}")
    grid = None
    if (d / ROWS_MAP_FILE).is_file():  # the map grid of each row, in rows.npz's order
        with np.load(d / ROWS_MAP_FILE, allow_pickle=False) as zg:
            if "grid" not in zg.files:
                raise ShardError(f"{d.name}: {ROWS_MAP_FILE} lacks grid")
            grid = zg["grid"]
        if grid.ndim != 2 or grid.shape != (len(rows), GRID_WIDTH) or grid.dtype != np.int8:
            raise ShardError(f"{d.name}: the map grid is {grid.shape} {grid.dtype}, not ({len(rows)}, {GRID_WIDTH}) int8 (one row of block M per row)")
        if len(grid) and (grid.min() < 0 or grid.max() > 1):
            raise ShardError(f"{d.name}: the map grid holds a number that is not 0 or 1")
    gaps = None
    if (d / ROWS_GAPS_FILE).is_file():  # block G of each row, in rows.npz's order
        if manifest.get("gaps_version") not in (None, GAPS_VERSION):
            raise ShardError(f"{d.name}: block G is version {manifest.get('gaps_version')}, this build reads version {GAPS_VERSION}")
        with np.load(d / ROWS_GAPS_FILE, allow_pickle=False) as zg:
            if "gaps" not in zg.files:
                raise ShardError(f"{d.name}: {ROWS_GAPS_FILE} lacks gaps")
            gaps = zg["gaps"]
        if gaps.ndim != 2 or gaps.shape != (len(rows), GAPS_WIDTH) or gaps.dtype != np.float32:
            raise ShardError(f"{d.name}: block G is {gaps.shape} {gaps.dtype}, not ({len(rows)}, {GAPS_WIDTH}) float32 (one row of block G per row: the row count differs)")
        if len(gaps) and not np.isfinite(gaps).all():
            raise ShardError(f"{d.name}: block G holds a number that is not finite")
    return {"name": d.name, "manifest": manifest, "records": recs, "rows": rows, "order": order, "sorted_decision": sd, "sorted_option": so,
            "tail_cut": tail_cut, "grid": grid, "gaps": gaps}


_PROVENANCE_KEYS = ("version", "agent", "agent_sha256", "judge", "judge_sha256", "engine", "engine_sha256", "k", "m", "max_options", "beyond",
                    "created_utc", "runs", "scale", "a2_sha256")


def load_shards(root: Path | str, *, kinds: Sequence[str] = KINDS, prior: PriorScorer | None = None, map_grid: bool = False,
                teacher_prior: PriorScorer | None = None, state_gaps: bool = False) -> TeacherSet:
    """Every shard under `root`, complete decisions only. A decision is dropped (and counted by reason in `stats["dropped"]`) when the
    teacher refused it, has no values, is of a kind the student does not take (`kinds`), has fewer than two options with a value, an option
    without a row, or a lambda of infinity; any other malformation refuses the shard. With `prior` the rows are scored (`attach_prior`).
    With `map_grid` every shard must hold its `rows_map.npz` (else `ShardError` naming the shards that lack it) and the set carries the
    grids; without it the grids are not read into the set. A prior with the map tower (`prior.map_grid`) needs `map_grid`, and the flag needs
    such a prior: the two are checked against each other. `state_gaps` is the same for block G (`rows_gaps.npz`, a prior with `state_gaps`)."""
    from run_train import models  # noqa: PLC0415 -- numpy only

    shards = find_shards(root)
    if not shards:
        raise ShardError(f"no decision shard (a folder with {DECISIONS_FILE}) under {Path(root).name}")
    for who, scorer in (("the prior", prior), ("the teacher prior", teacher_prior)):
        if scorer is not None and scorer.map_grid and not map_grid:
            raise ShardError(f"{who} has a map tower: fit with --map-grid (a fit without it refuses a map-grid prior)")
    if map_grid and prior is not None and not prior.map_grid:
        raise ShardError("--map-grid: the prior has no map tower (give the v2 human prior)")
    for who, scorer in (("the prior", prior), ("the teacher prior", teacher_prior)):
        if scorer is not None and getattr(scorer, "state_gaps", False) and not state_gaps:
            raise ShardError(f"{who} reads the state gaps: fit with --state-gaps (a fit without it refuses a state-gaps prior)")
    if state_gaps and prior is not None and not getattr(prior, "state_gaps", False):
        raise ShardError("--state-gaps: the prior does not read the state gaps (give the v3 human prior)")
    if state_gaps:
        lacking = [d.name for d in shards if not (d / ROWS_GAPS_FILE).is_file()]
        if lacking:
            raise ShardError(f"--state-gaps: {len(lacking)} of {len(shards)} shards have no {ROWS_GAPS_FILE} ({', '.join(lacking[:4])}{', ...' if len(lacking) > 4 else ''}): "
                             "run `rl/run_teacher.py reencode` on the store")
    if map_grid:
        lacking = [d.name for d in shards if not (d / ROWS_MAP_FILE).is_file()]
        if lacking:
            raise ShardError(f"--map-grid: {len(lacking)} of {len(shards)} shards have no {ROWS_MAP_FILE} ({', '.join(lacking[:4])}{', ...' if len(lacking) > 4 else ''}): "
                             "run `rl/run_teacher.py reencode` on the store")
    seeds: list[str] = []
    seed_ix: dict[str, int] = {}
    cols: dict[str, list] = {k: [] for k in ("run", "kind", "act", "floor", "k", "lam", "teacher", "player", "start", "count")}
    q_rows: list[np.ndarray] = []
    v_rows: list[np.ndarray] = []
    lp_rows: list[np.ndarray] = []
    dense_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    grid_parts: list[np.ndarray] = []
    gaps_parts: list[np.ndarray] = []
    g_all: list = []
    dropped: Counter = Counter()
    kept: Counter = Counter()
    manifests: list[dict] = []
    grades: dict[str, set[str]] = defaultdict(set)  # grade -> the shards (and records) that carry it
    n_rows = 0
    for d in shards:
        sh = read_shard(d)
        grade = sh["manifest"].get("grade", "floors")  # a shard from before the grades is graded floors
        if grade not in GRADES:
            raise ShardError(f"{sh['name']}: unknown grade {grade!r}; one of {', '.join(GRADES)}")
        grades[grade].add(sh["name"])
        manifests.append({"shard": sh["name"], "grade": grade, **{k: _clean(sh["manifest"][k]) for k in _PROVENANCE_KEYS if k in sh["manifest"]}})
        dropped["truncated_tail"] += sh["tail_cut"]
        sd, so, order, rows = sh["sorted_decision"], sh["sorted_option"], sh["order"], sh["rows"]
        lo = np.searchsorted(sd, np.arange(len(sh["records"])), side="left")
        hi = np.searchsorted(sd, np.arange(len(sh["records"])), side="right")
        for i, r in enumerate(sh["records"]):
            kind = r.get("kind")
            if r.get("grade") is not None:
                if r["grade"] not in GRADES:
                    raise ShardError(f"{sh['name']}: record {i + 1} has an unknown grade {r['grade']!r}")
                grades[r["grade"]].add(sh["name"])
            if r.get("refused") or r.get("q") is None:
                dropped["refused"] += 1
                continue
            if kind not in kinds:
                dropped["kind_not_taken"] += 1
                continue
            n = int(r["n_options"])
            played = [int(o) for o in r["played"]]
            if len(set(played)) != len(played) or any(not 0 <= o < n for o in played):
                raise ShardError(f"{sh['name']}: record {i + 1} lists played options {played} outside its {n} options")
            try:
                q = {int(o): float(x) for o, x in r["q"].items()}
                gm = [[float(x) for x in r["g"][str(o)]] for o in played if str(o) in r["g"]]
            except (TypeError, ValueError, AttributeError) as e:
                raise ShardError(f"{sh['name']}: record {i + 1} has values that are not numbers") from e
            if any(o not in played for o in q) or not all(math.isfinite(x) for x in q.values()):
                raise ShardError(f"{sh['name']}: record {i + 1}'s q is not finite values of played options")
            priced = [o for o in played if o in q]
            if len(priced) < 2:
                dropped["fewer_than_two_priced"] += 1
                continue
            idx = order[lo[i]:hi[i]]
            if len(idx) != n or not np.array_equal(so[lo[i]:hi[i]], np.arange(n)):
                dropped["incomplete_rows"] += 1
                continue
            lam = float(r["lam"]) if r.get("screen") is not None else 0.0
            if not math.isfinite(lam):
                dropped["lambda_infinite"] += 1
                continue
            seed = str(r["seed"])
            if seed not in seed_ix:
                seed_ix[seed] = len(seeds)
                seeds.append(seed)
            vv = r.get("v_values") or [None] * n
            lp = r.get("logp")
            qr = np.full(n, np.nan, dtype=np.float32)
            for o, x in q.items():
                qr[o] = x
            cols["run"].append(seed_ix[seed])
            cols["kind"].append(KIND_ID[kind])
            cols["act"].append(min(max(int(r["act"]), 0), 2))
            cols["floor"].append(int(r["floor"]))
            cols["k"].append(min((len(x) for x in gm), default=int(r.get("k", 1))))
            cols["lam"].append(lam)
            cols["teacher"].append(int(r.get("teacher", r["player"])))
            cols["player"].append(int(r["player"]))
            cols["start"].append(n_rows)
            cols["count"].append(n)
            q_rows.append(qr)
            v_rows.append(np.array([np.nan if x is None else float(x) for x in vv], dtype=np.float32))
            lp_rows.append(np.array(lp, dtype=np.float32) if lp is not None and len(lp) == n else np.full(n, np.nan, dtype=np.float32))
            sel = rows[idx]
            dense, cnt = models.value_inputs(sel)
            dense_parts.append(dense)
            count_parts.append(cnt)
            if map_grid:
                grid_parts.append(sh["grid"][idx])
            if state_gaps:
                gaps_parts.append(sh["gaps"][idx])
            g_all.append(gm)
            kept[kind] += 1
            n_rows += n
    if len(grades) > 1:
        raise ShardError("the shards carry two grades (" + "; ".join(f"{g}: {len(s)} shard(s)" for g, s in sorted(grades.items()))
                         + "): a store is graded one way, its Q in one unit; regrade into a root of its own (run_teacher.py regrade) and fit one root at a time")
    if not cols["run"]:
        raise ShardError(f"no complete decision in {len(shards)} shard(s) (dropped: {dict(dropped)})")
    ts = TeacherSet(
        seeds=seeds, run=np.array(cols["run"], dtype=np.int32), kind=np.array(cols["kind"], dtype=np.int8), act=np.array(cols["act"], dtype=np.int8),
        floor=np.array(cols["floor"], dtype=np.int16), k=np.array(cols["k"], dtype=np.int16), lam=np.array(cols["lam"], dtype=np.float32),
        teacher=np.array(cols["teacher"], dtype=np.int16), player=np.array(cols["player"], dtype=np.int16),
        start=np.array(cols["start"], dtype=np.int64), count=np.array(cols["count"], dtype=np.int32), q=np.concatenate(q_rows),
        v=np.concatenate(v_rows), dense=np.concatenate(dense_parts), counts=np.concatenate(count_parts), g=g_all,
        logp_rec=np.concatenate(lp_rows),
        stats={"shards": len(shards), "decisions": len(cols["run"]), "rows": n_rows, "runs": len(seeds), "kept": dict(kept), "dropped": dict(dropped),
               "map_grid": bool(map_grid), "state_gaps": bool(state_gaps), "grade": next(iter(grades), "floors")},
        manifests=manifests, grid=np.concatenate(grid_parts).astype(np.int8) if map_grid else None,
        gaps=np.concatenate(gaps_parts).astype(np.float32) if state_gaps else None)
    ts.set_tau()
    if prior is not None:
        ts.attach_prior(prior, teacher_prior=teacher_prior)
    return ts


def public_stats(ts: TeacherSet) -> dict[str, Any]:
    """The counts of a loaded set for a report: no seeds, nothing per run."""
    split = ts.split_idx()
    kinds = Counter(KINDS[int(k)] for k in ts.kind)
    return {"shards": ts.stats["shards"], "decisions": len(ts), "rows": int(len(ts.q)), "runs": len(ts.seeds), "kept_by_kind": dict(kinds),
            "dropped": ts.stats["dropped"], "decisions_by_split": {k: int(len(v)) for k, v in split.items()},
            "runs_by_split": dict(Counter(ts.run_split().tolist())), "noise_sd": ts.noise, "provenance": ts.manifests,
            "logp_check": ts.stats.get("logp_check"), "map_grid": ts.grid is not None, "state_gaps": ts.gaps is not None, "grade": ts.stats.get("grade", "floors")}


# ---------------------------------------------------------------------------------------------------------- the human data


def default_exam_ids() -> frozenset[str]:
    """The exam fifth's run ids: the ids files beside the corpus united (the earlier and the 2026-11-06 one), whichever exist."""
    paths = [hs.EXAM_IDS, hs.EXAM_IDS.with_name("human-exam-fifth-ids-2026-11-06.txt")]
    ids: set[str] = set()
    found = 0
    for p in paths:
        if Path(p).is_file():
            ids |= set(hs.read_exam_ids(p))
            found += 1
    if not found:
        raise SystemExit("run_student: no exam ids file beside the corpus: the exam fifth cannot be kept out")
    return frozenset(ids)


def check_no_exam(ds: Any, exam_ids: frozenset[str]) -> int:
    """Raise unless no run the dataset's shards list is in the exam fifth (`human_starts.assert_no_exam` on every run id) and no decision is
    in fifth 0 (the exam bucket). Returns the runs checked. The human term and both legs call this on every dataset they load."""
    if len(ds.dec) and (ds.dec[:, hp.C["fifth"]] == hs.EXAM_BUCKET).any():
        raise AssertionError("run_student: a decision of the exam fifth (bucket 0) is in the dataset")
    return hs.assert_no_exam(ds.run_ids, exam_ids)


def load_human(dec_dir: Path | str, fifths: Sequence[int], *, exam_ids: frozenset[str] | None = None, kinds: Sequence[str] | None = None,
               map_grid: bool = False, state_gaps: bool = False):
    """The prior's decisions of `fifths` (1 to 4 only: asking for the exam bucket is refused before any shard is read) from the shards of
    `dec_dir`, checked against the exam fifth. v1 shards only (the true map in every row): a donor-map dataset is refused. With `map_grid`
    the shards must also hold block M (v2: `human_prior.py build --truth --map-grid`), with `state_gaps` block G (v3: `build --truth --state-gaps`)."""
    bad = [f for f in fifths if f == hs.EXAM_BUCKET or f not in (1, 2, 3, 4)]
    if bad:
        raise AssertionError(f"run_student: fifth(s) {bad} asked for; only 1 to 4 are readable, bucket 0 is the exam")
    ds = hp.Dataset.load(Path(dec_dir), fifths, None, kinds=list(kinds) if kinds is not None else None)
    if not ds.has_map:
        raise ShardError("the human shards are v0 (donor maps): the student reads v1 shards, whose rows carry the true map")
    if map_grid and not ds.has_grid:
        raise ShardError("--map-grid: the human shards have no map grid (use the v2 shards, human-prior/decisions-v2)")
    if state_gaps and not ds.has_gaps:
        raise ShardError("--state-gaps: the human shards have no block G (use the v3 shards, human-prior/decisions-v3)")
    check_no_exam(ds, default_exam_ids() if exam_ids is None else exam_ids)
    return ds


# ---------------------------------------------------------------------------------------------------------- the net and the fit


@dataclasses.dataclass
class FitConfig:
    epochs: int = 8
    patience: int = 2
    batch_teacher: int = 128
    batch_human: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    clip: float = 1.0
    beta: float = 1.0  # weight of the human choices' cross-entropy
    kl: float = 0.1  # weight of KL(pi_h || pi_s)
    seed: int = 20261111


def make_net(params: Mapping[str, np.ndarray], dropout: float = 0.1, state_gaps: bool = False):
    """The prior's net (the run judgement's embeddings, two layers and one logit; the parameter names are `weights.npz`'s) loaded with
    `params` (a prior's weights): the student before any step is the prior, logit for logit. `state_gaps`: the prior's first layer ends in block G's columns
    (they are in `hidden1.weight` like every other input; the flag makes the net take the extra input)."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    e_deck, e_rel, e_pot = (params[k].shape for k in ("embed.deck.weight", "embed.relics.weight", "embed.potions.weight"))
    hidden = params["hidden1.weight"].shape[0]
    wide = params["hidden1.weight"].shape[1]
    has_map = "map.conv1.weight" in params  # the v2 prior's map tower (`human_prior.map_tower_module`): hidden1 then reads its 64 outputs too

    class Embed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.deck = nn.Linear(e_deck[1], e_deck[0], bias=False)
            self.relics = nn.Linear(e_rel[1], e_rel[0], bias=False)
            self.potions = nn.Linear(e_pot[1], e_pot[0], bias=False)

        def forward(self, counts):  # noqa: ANN001
            a, b = e_deck[1], e_deck[1] + e_rel[1]
            return torch.cat([self.deck(counts[:, :a]), self.relics(counts[:, a:b]), self.potions(counts[:, b:])], dim=1)

    class StudentNet(nn.Module):
        uses_win = False
        uses_map = has_map
        uses_gaps = bool(state_gaps)

        def __init__(self) -> None:
            super().__init__()
            self.embed = Embed()
            if has_map:
                self.map = hp.map_tower_module()
            self.hidden1 = nn.Linear(wide, hidden)
            self.hidden2 = nn.Linear(hidden, hidden)
            self.score = nn.Linear(hidden, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, dense, counts, grid=None, gaps=None):  # noqa: ANN001
            parts = [dense, self.embed(counts)]
            if has_map:
                if grid is None:
                    raise ValueError("this net reads the map grid")
                parts.append(self.map(grid))
            if state_gaps:
                if gaps is None:
                    raise ValueError("this net reads the state gaps")
                parts.append(gaps.float())
            h = self.dropout(torch.relu(self.hidden1(torch.cat(parts, dim=1))))
            return self.score(torch.relu(self.hidden2(h))).squeeze(1)

    net = StudentNet()
    net.load_state_dict({k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in params.items()}, strict=True)
    return net


def net_params(net) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}


def read_prior_params(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != "meta"}


def human_loss_from_z(gpu, kind: str, didx, z, m, valid):  # noqa: ANN001
    """The prior's loss of a batch of one kind given the batch's logits `z` (padded -1e9; `Gpu.logits`): `Gpu.loss`'s formulas, with the
    event kind (a softmax over the page's options, as card, ancient and route)."""
    t = gpu.t
    w = gpu.weight[didx]
    ch = gpu.chosen[didx]
    if kind in ("card", "ancient", "route", "event"):
        nll = t.logsumexp(z, 1) - z.gather(1, ch[:, None])[:, 0]
        return (nll * w).sum() / w.sum()
    if kind == "rest":
        role = t.where(valid, gpu.role[m.clamp(min=0)], t.full_like(m, -1))
        aux = gpu.aux[didx]
        lse = t.logsumexp(z, 1)
        nll_kind = lse - t.logsumexp(z.masked_fill(role != aux[:, None], -1e9), 1)
        loss = (nll_kind * w).sum() / w.sum()
        has = (aux == hp.ROLE_SMITH) & (ch >= 0)
        if has.any():
            zs = z.masked_fill(role != hp.ROLE_SMITH, -1e9)
            nll_t = t.logsumexp(zs, 1) - z.gather(1, ch.clamp(min=0)[:, None])[:, 0]
            loss = loss + (nll_t * w * has).sum() / (w * has).sum().clamp(min=1e-6)
        return loss
    zz = z[:, 1] - z[:, 0]  # shop: row 0 leave, row 1 buy
    lab = ch.float()
    bce = t.nn.functional.softplus(-zz) * lab + t.nn.functional.softplus(zz) * (1 - lab)
    return (bce * w).sum() / w.sum()


def _kl(t, z_s, z_h):  # noqa: ANN001
    """Mean over decisions of KL(softmax(z_h) || softmax(z_s)) over the padded options (pads are -1e9 in both)."""
    ls, lh = t.log_softmax(z_s, 1), t.log_softmax(z_h, 1)
    return (lh.exp() * (lh - ls)).sum(1).mean()


def fit(ts: TeacherSet, prior_params: Mapping[str, np.ndarray], *, human=None, cfg: FitConfig | None = None, device: str | None = None,
        lam_of: Mapping[tuple[str, int], float] | None = None, log=print, state_gaps: bool = False) -> dict[str, Any]:
    """Fit the student on the teacher's `train` runs (stopping early on the `val` runs' cross-entropy) and, with `human` (a v1 `hp.Dataset` of
    the training fifths), the human choices. Returns {"params", "history", "best_epoch", "seconds", "config"}; the weights are the best epoch's."""
    import torch  # noqa: PLC0415

    cfg = cfg or FitConfig()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    net = make_net(prior_params, cfg.dropout, state_gaps).to(dev)
    frozen = make_net(prior_params, 0.0, state_gaps).to(dev).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    parts = ts.split_idx()
    train, val = parts["train"], parts["val"]
    if not len(train):
        raise ValueError("no training decision: the shards' runs are all held out")
    if ts.zh is None or ts.tau is None:
        raise ValueError("fit needs the prior's logits and temperatures on the set (load_shards(prior=...) or attach_prior)")
    target = torch.from_numpy(ts.targets(lam_of)).to(dev)
    dense = torch.from_numpy(ts.dense).to(dev)
    counts = torch.from_numpy(ts.counts).to(dev)
    zh_rows = torch.from_numpy(ts.zh).to(dev)
    played_rows = torch.from_numpy(~np.isnan(ts.q)).to(dev)
    tower = bool(getattr(net, "uses_map", False))
    if tower and (ts.grid is None or (human is not None and cfg.beta > 0 and not human.has_grid)):
        raise ShardError("the prior has a map tower: the teacher's shards need rows_map.npz and the human shards the map grid (decisions-v2)")
    grid = torch.from_numpy(ts.grid).to(dev) if tower else None
    if state_gaps and (ts.gaps is None or (human is not None and cfg.beta > 0 and not human.has_gaps)):
        raise ShardError("the prior reads the state gaps: the teacher's shards need rows_gaps.npz and the human shards block G (decisions-v3)")
    gaps = torch.from_numpy(ts.gaps).to(dev) if state_gaps else None

    def batch_logits(model, idx: np.ndarray):
        m = torch.from_numpy(ts.matrix(idx)).to(dev)
        valid = m >= 0
        rows = m[valid]
        z = torch.full(m.shape, -1e9, device=dev)
        kw = {}
        if tower:
            kw["grid"] = grid[rows]
        if state_gaps:
            kw["gaps"] = gaps[rows]
        z[valid] = model(dense[rows], counts[rows], **kw)
        return z, m, valid

    def teacher_terms(idx: np.ndarray, train_mode: bool):
        net.train(train_mode)
        z, m, valid = batch_logits(net, idx)
        pm = torch.where(valid, played_rows[m.clamp(min=0)], torch.zeros_like(valid))
        tm = torch.where(pm, target[m.clamp(min=0)], torch.zeros_like(z))
        logq = torch.log_softmax(z.masked_fill(~pm, -1e9), 1)
        ce = -(tm * logq).sum(1).mean()
        zh = torch.where(valid, zh_rows[m.clamp(min=0)], torch.full_like(z, -1e9))
        return ce, _kl(torch, z, zh)

    pools: dict[str, np.ndarray] = {}
    gpu = None
    if human is not None and cfg.beta > 0:
        gpu = hp.Gpu(human, None, dev, 1.0)
        ok = ~hp._bool_unavailable(human, True, True, True)  # noqa: SLF001 -- the prior's own rule: replay mismatches, unverified shops out
        pools = {k: human.kind_idx[k][ok[human.kind_idx[k]]] for k in HUMAN_KINDS if k in human.kind_idx and ok[human.kind_idx[k]].any()}
    kinds_h = list(pools)
    weights_h = np.array([max(1, len(pools[k]) // cfg.batch_human) for k in kinds_h], dtype=np.float64)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: list[dict[str, Any]] = []
    best, best_state, since, t0 = None, None, 0, time.time()
    for epoch in range(cfg.epochs):
        perm = rng.permutation(train)
        tot = {"ce": 0.0, "kl": 0.0, "human": 0.0, "steps": 0}
        for i in range(0, len(perm), cfg.batch_teacher):
            ce, klt = teacher_terms(perm[i:i + cfg.batch_teacher], True)
            loss = ce + cfg.kl * klt
            if gpu is not None:
                kind = kinds_h[int(rng.choice(len(kinds_h), p=weights_h / weights_h.sum()))]
                didx = torch.from_numpy(rng.choice(pools[kind], size=min(cfg.batch_human, len(pools[kind])), replace=False)).to(dev)
                net.train()
                z, m, valid = gpu.logits(net, didx, False)
                with torch.no_grad():
                    zh, _, _ = gpu.logits(frozen, didx, True)
                hl = human_loss_from_z(gpu, kind, didx, z, m, valid)
                loss = loss + cfg.beta * hl + cfg.kl * _kl(torch, z, zh)
                tot["human"] += float(hl.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip)
            opt.step()
            tot["ce"] += float(ce.item())
            tot["kl"] += float(klt.item())
            tot["steps"] += 1
        with torch.no_grad():
            ev = val if len(val) else train
            vce = float(np.mean([float(teacher_terms(ev[j:j + 512], False)[0].item()) for j in range(0, len(ev), 512)]))
        n = max(tot["steps"], 1)
        history.append({"epoch": epoch + 1, "train_ce": round(tot["ce"] / n, 5), "train_kl": round(tot["kl"] / n, 5),
                        "train_human": round(tot["human"] / n, 5) if gpu is not None else None, "val_ce": round(vce, 5),
                        "seconds": round(time.time() - t0, 1)})
        log(f"  epoch {epoch + 1}: teacher ce {tot['ce'] / n:.4f} kl {tot['kl'] / n:.4f} human {tot['human'] / n:.4f} val ce {vce:.4f}")
        if best is None or vce < best - 1e-5:
            best, since = vce, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            since += 1
            if since >= cfg.patience:
                break
    net.load_state_dict(best_state)
    return {"params": net_params(net), "history": history, "best_epoch": 1 + int(np.argmin([h["val_ce"] for h in history])),
            "seconds": round(time.time() - t0, 1), "config": dataclasses.asdict(cfg), "device": str(dev),
            "val_decisions": int(len(val)), "train_decisions": int(len(train))}


# ---------------------------------------------------------------------------------------------------------- leg 1


def _ce_rows(ts: TeacherSet, idx: np.ndarray, logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per decision of `idx`: the cross-entropy of the teacher's target (over the played options) to the softmax of `logits` over the same options."""
    m = ts.matrix(idx)
    ok = m >= 0
    mm = np.maximum(m, 0)
    played = ok & ~np.isnan(ts.q[mm])
    z = np.where(played, logits[mm].astype(np.float64), -np.inf)
    top = z.max(1, keepdims=True)
    lse = top[:, 0] + np.log(np.exp(z - top).sum(1))
    logq = np.where(played, z - lse[:, None], 0.0)
    return -(np.where(played, target[mm].astype(np.float64) * logq, 0.0)).sum(1)


def _boot(runs: np.ndarray, diff: np.ndarray, draws: int = BOOT_DRAWS) -> dict[str, float]:
    import strong_sets as ss  # noqa: PLC0415

    b = ss.paired_bootstrap(runs, diff, len(np.unique(runs)), draws=draws)
    return {"mean": round(b["mean"], 5), "lo95": round(b["lo95"], 5), "hi95": round(b["hi95"], 5), "runs": b["runs"]}


def leg1(ts: TeacherSet, idx: np.ndarray, student: PriorScorer, prior: PriorScorer, target: np.ndarray | None = None) -> dict[str, Any]:
    """Teacher imitation on the decisions `idx` (the held-out runs): per kind, per kind and act, and over all, the cross-entropy to the
    teacher's target of the student and of the prior (the same decisions), the gain (prior minus student, nats, positive is better), its 95%
    interval by resampling runs, the target's own entropy (the floor of any cross-entropy) and the decisions and runs."""
    target = ts.targets() if target is None else target
    zs = ts.logits_of(student)
    zp = ts.zh if ts.zh is not None else ts.logits_of(prior)
    out: dict[str, Any] = {"kinds": {}, "cells": {}, "test_decisions": int(len(idx))}

    def cell(sel: np.ndarray) -> dict[str, Any] | None:
        if not len(sel):
            return None
        ce_s, ce_p = _ce_rows(ts, sel, zs, target), _ce_rows(ts, sel, zp, target)
        m = ts.matrix(sel)
        ok = m >= 0
        mm = np.maximum(m, 0)
        tt = np.where(ok, target[mm].astype(np.float64), 0.0)
        ent = -(np.where(tt > 0, tt * np.log(np.maximum(tt, 1e-300)), 0.0)).sum(1)
        runs = ts.run[sel]
        return {"n": int(len(sel)), "runs": int(len(np.unique(runs))), "ce_student": round(float(ce_s.mean()), 5), "ce_prior": round(float(ce_p.mean()), 5),
                "ce_floor": round(float(ent.mean()), 5), "gain": _boot(runs, ce_p - ce_s)}

    kind_of = ts.kind[idx]
    for kid, kind in enumerate(KINDS):
        sel = idx[kind_of == kid]
        c = cell(sel)
        if c is not None:
            out["kinds"][kind] = c
        for act in (0, 1, 2):
            sa = sel[ts.act[sel] == act]
            if len(sa) >= MIN_CELL:
                out["cells"][f"{kind}/a{act + 1}"] = cell(sa)
    out["all"] = cell(idx)
    return out


# ---------------------------------------------------------------------------------------------------------- leg 2


def human_per_decision(ds: Any, scores: np.ndarray, valid: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Per decision of each kind the human's negative log-likelihood under row `scores`, with the run, act and whether the run won:
    card, ancient and rest (`strong_sets.per_decision`: the rest kind's and the smith target's choices pooled as the kind "rest"), shop
    (buy against leave), route and event (a softmax over the options)."""
    import strong_sets as ss  # noqa: PLC0415

    dec, c = ds.dec, hp.C
    out: dict[str, dict[str, np.ndarray]] = {}
    base = ss.per_decision(ds, scores, valid)
    for kind in ("card", "ancient"):
        if kind in base:
            idx = ds.kind_idx[kind]
            idx = idx[valid[idx]]
            out[kind] = {"run": base[kind]["run"], "act": base[kind]["act"], "nll": base[kind]["nll"], "won": dec[idx, c["won"]].astype(np.int8)}
    parts = []
    idx = ds.kind_idx["rest"]
    idx = idx[valid[idx]]
    if "rest_kind" in base:
        parts.append({"run": base["rest_kind"]["run"], "act": base["rest_kind"]["act"], "nll": base["rest_kind"]["nll"],
                      "won": dec[idx, c["won"]].astype(np.int8)})
    if "rest_smith_target" in base:
        aux, ch = dec[idx, c["aux"]], dec[idx, c["chosen"]]
        sm = (aux == hp.ROLE_SMITH) & (ch >= 0)
        parts.append({"run": base["rest_smith_target"]["run"], "act": base["rest_smith_target"]["act"], "nll": base["rest_smith_target"]["nll"],
                      "won": dec[idx[sm], c["won"]].astype(np.int8)})
    if parts:
        out["rest"] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    for kind in ("route", "event", "shop"):
        if kind not in ds.kind_idx:
            continue
        idx = ds.kind_idx[kind]
        idx = idx[valid[idx]]
        if not len(idx):
            continue
        m = ds.matrix(idx)
        ok = m >= 0
        s = np.where(ok, scores[np.maximum(m, 0)], -np.inf)
        chosen = dec[idx, c["chosen"]]
        if kind == "shop":
            z = s[:, 1] - s[:, 0]
            lab = chosen > 0
            nll = np.logaddexp(0.0, np.where(lab, -z, z))
        else:
            nll = hp._lse(s) - s[np.arange(len(idx)), chosen]  # noqa: SLF001
        out[kind] = {"run": dec[idx, c["run"]], "act": dec[idx, c["act"]], "nll": nll, "won": dec[idx, c["won"]].astype(np.int8)}
    return out


def leg2(ds: Any, student: PriorScorer, prior: PriorScorer, *, exam_ids: frozenset[str] | None = None) -> dict[str, Any]:
    """The human bar on `ds` (fifth 4, never the exam fifth): per kind with human choices, the student's mean log-loss minus the prior's on
    the won runs and on all runs (nats; negative is better), each with a 95% interval by resampling runs, the decisions and runs; a cell over
    fewer than `MIN_CELL` runs is not shown. `no_human_fit` lists the kinds the data has no decisions for."""
    check_no_exam(ds, default_exam_ids() if exam_ids is None else exam_ids)
    valid = ~hp._bool_unavailable(ds, True, True, True)  # noqa: SLF001
    if (student.map_grid or prior.map_grid) and not ds.has_grid:
        raise ShardError("leg 2: the scorers have a map tower and the human shards have no map grid (use decisions-v2)")
    if (student.state_gaps or prior.state_gaps) and not ds.has_gaps:
        raise ShardError("leg 2: the scorers read the state gaps and the human shards have no block G (use decisions-v3)")
    ps = human_per_decision(ds, hp.row_scores_np(ds, None, student.logits, grid=student.map_grid, gaps=student.state_gaps), valid)
    pp = human_per_decision(ds, hp.row_scores_np(ds, None, prior.logits, grid=prior.map_grid, gaps=prior.state_gaps), valid)
    out: dict[str, Any] = {"measured": True, "heldout_fifth": HUMAN_TEST_FIFTH, "source": SPIRE_CODEX, "kinds": {}, "no_human_fit": [], "hidden": []}

    def cell(rec_s: dict, rec_p: dict, keep: np.ndarray) -> dict[str, Any] | None:
        runs = np.unique(rec_s["run"][keep])
        if len(runs) < MIN_CELL or keep.sum() < MIN_CELL:
            return None
        diff = rec_s["nll"][keep] - rec_p["nll"][keep]
        b = _boot(rec_s["run"][keep], diff)
        return {"n": int(keep.sum()), "runs": int(len(runs)), "diff": b["mean"], "lo95": b["lo95"], "hi95": b["hi95"],
                "ll_student": round(float(rec_s["nll"][keep].mean()), 5), "ll_prior": round(float(rec_p["nll"][keep].mean()), 5)}

    for kind in HUMAN_KINDS:
        if kind not in ps:
            out["no_human_fit"].append(kind)
            continue
        rs_, rp_ = ps[kind], pp[kind]
        everyone = np.ones(len(rs_["nll"]), dtype=bool)
        won = rs_["won"] == 1
        c = {"won": cell(rs_, rp_, won), "all": cell(rs_, rp_, everyone)}
        if c["all"] is None:
            out["hidden"].append(kind)
            out["no_human_fit"].append(kind)
            continue
        out["kinds"][kind] = c
    out["no_human_fit"] += [k for k in KINDS if k not in HUMAN_KINDS]
    return out


# ---------------------------------------------------------------------------------------------------------- lambda


def measure_lambda(ts: TeacherSet, idx: np.ndarray, student: PriorScorer) -> dict[str, Any]:
    """Lambda per kind and act on the decisions `idx` (the held-out runs): `run_teacher.lambda_fit` over each decision's played options'
    (Q, V, log pi_s), the student's log-probability taken over all the screen's options as it is at play. A (kind, act) cell with fewer than
    `MIN_LAMBDA_DECISIONS` decisions falls back to the kind's pooled over acts; none for the kind with fewer; Act 3 takes Act 2's (the
    teacher decides no Act 3 screen). Clipped to [0, LAMBDA_CAP]. `chosen[kind]` is the recipe's triple."""
    fitter = teacher_fn("lambda_fit")
    zs = ts.logits_of(student)
    cells: dict[tuple[str, int], list] = defaultdict(list)
    for i in idx:
        a, b = int(ts.start[i]), int(ts.start[i] + ts.count[i])
        q, v = ts.q[a:b], ts.v[a:b]
        lp = log_softmax(zs[a:b])
        keep = ~np.isnan(q) & ~np.isnan(v)
        if keep.sum() >= 2:
            cells[(ts.kind_name(i), int(ts.act[i]))].append([(float(x), float(y), float(z)) for x, y, z in zip(q[keep], v[keep], lp[keep], strict=True)])
    out: dict[str, Any] = {"cells": {}, "kinds": {}, "chosen": {}}
    chosen: dict[str, list[float]] = {}
    for kind in KINDS:
        per_act = {a: cells.get((kind, a), []) for a in (0, 1)}
        pooled = [d for a in (0, 1) for d in per_act[a]]
        kf = fitter(pooled) if pooled else {"decisions": 0, "lambda": None, "se": None}
        out["kinds"][kind] = kf
        triple: list[float | None] = []
        for a in (0, 1):
            cf = fitter(per_act[a]) if per_act[a] else {"decisions": 0, "lambda": None, "se": None}
            out["cells"][f"{kind}/a{a + 1}"] = cf
            use = cf if cf["decisions"] >= MIN_LAMBDA_DECISIONS and cf["lambda"] is not None else kf
            ok = use["decisions"] >= MIN_LAMBDA_DECISIONS and use["lambda"] is not None
            triple.append(None if not ok else min(max(float(use["lambda"]), 0.0), LAMBDA_CAP))
        if all(x is not None for x in triple):
            chosen[kind] = [round(triple[0], 4), round(triple[1], 4), round(triple[1], 4)]  # Act 3 takes Act 2's
    out["chosen"] = chosen
    out["rule"] = (f"cell if at least {MIN_LAMBDA_DECISIONS} held-out decisions, else the kind's pooled, else none; clipped to [0, {LAMBDA_CAP}]; "
                   "Act 3 = Act 2")
    return out


# ---------------------------------------------------------------------------------------------------------- the verdict


def human_fit_of(legs: Mapping[str, Any], kind: str) -> bool:
    """Whether a kind has a human fit to hold the student to: a kind of `HUMAN_KINDS` that the held-out human data had decisions for."""
    return kind in HUMAN_KINDS and kind not in ((legs.get("leg2") or {}).get("no_human_fit") or [])


def kind_verdict(legs: Mapping[str, Any], kind: str) -> dict[str, Any]:
    """Whether the student may take `kind` over: leg 1 (the held-out teacher imitation: at least `LEG1_MIN_DECISIONS` decisions and the
    gain's lower bound above 0), leg 2 where the kind has a human fit (won runs within +`TIE_NATS`, all runs within +`LIMIT_NATS`; unmeasured
    fails), and a measured lambda. `why` says what failed."""
    l1 = ((legs.get("leg1") or {}).get("kinds") or {}).get(kind)
    l2 = ((legs.get("leg2") or {}).get("kinds") or {}).get(kind)
    why: list[str] = []
    ok1 = False
    if not l1:
        why.append("leg 1: no held-out decisions of this kind")
    elif l1["n"] < LEG1_MIN_DECISIONS:
        why.append(f"leg 1: {l1['n']} held-out decisions, needs {LEG1_MIN_DECISIONS}")
    elif not l1["gain"]["lo95"] > 0:
        why.append(f"leg 1: the gain's lower bound {l1['gain']['lo95']:+.4f} is not above 0")
    else:
        ok1 = True
    ok2: bool | None = None
    if human_fit_of(legs, kind):
        if not l2 or not l2.get("won") or not l2.get("all"):
            ok2 = False
            why.append("leg 2: the human bar is not measured on the won runs" if (legs.get("leg2") or {}).get("measured") else "leg 2: not measured")
        else:
            ok2 = l2["won"]["diff"] <= TIE_NATS + 1e-12 and l2["all"]["diff"] <= LIMIT_NATS + 1e-12
            if not ok2:
                why.append(f"leg 2: won {l2['won']['diff']:+.4f} (at most +{TIE_NATS:g}), all {l2['all']['diff']:+.4f} (at most +{LIMIT_NATS:g})")
    lam = ((legs.get("lambda") or {}).get("chosen") or {}).get(kind)
    if lam is None:
        why.append("no measured lambda")
    return {"leg1": ok1, "leg2": ok2, "lambda": lam, "pass": bool(ok1 and ok2 is not False and lam is not None), "why": "; ".join(why)}


def passing_kinds(legs: Mapping[str, Any], base_blends: Sequence[str] = ("card", "ancient", "rest")) -> dict[str, list[float]]:
    """The kinds the candidate gives the student, with their lambda per act: those that pass `kind_verdict`, less a kind the base decides
    by `vrun` alone whose measured lambdas are all 0 (the student there would be `vrun` again)."""
    out: dict[str, list[float]] = {}
    for kind in KINDS:
        v = kind_verdict(legs, kind)
        if not v["pass"]:
            continue
        if kind not in base_blends and not any(x > 0 for x in v["lambda"]):
            continue
        out[kind] = [float(x) for x in v["lambda"]]
    return out


# ---------------------------------------------------------------------------------------------------------- one round


def utc() -> str:
    import datetime as dt  # noqa: PLC0415

    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_legs(ts: TeacherSet, student_path: Path | str, prior_path: Path | str, *, human_test=None, baseline_kind: str = "human prior",
             exam_ids: frozenset[str] | None = None) -> dict[str, Any]:
    """Both legs and lambda for a written student file against `prior_path`, on the held-out runs (leg 1 and lambda) and fifth 4 (leg 2:
    `human_test`, a loaded v1 dataset, or None: not measured, and no kind with human choices can pass)."""
    from human_prior_infer import StudentScorer  # noqa: PLC0415

    student = StudentScorer.load(student_path)
    prior = PriorScorer.load(prior_path)
    test = ts.split_idx()["test"]
    legs: dict[str, Any] = {"version": VERSION, "written": utc(), "source": SPIRE_CODEX, "student_sha256": file_sha256(student_path),
                            "baseline": {"sha256": file_sha256(prior_path), "kind": baseline_kind},
                            "rule": {"leg1": f"gain lower bound above 0 on at least {LEG1_MIN_DECISIONS} held-out decisions of the kind",
                                     "leg2": f"human log-loss against the baseline's at most +{TIE_NATS} (won runs) and +{LIMIT_NATS} (all runs); "
                                             "none for a kind with no human fit"}}
    legs["leg1"] = leg1(ts, test, student, prior)
    legs["leg2"] = leg2(human_test, student, prior, exam_ids=exam_ids) if human_test is not None else {"measured": False, "no_human_fit": []}
    legs["lambda"] = measure_lambda(ts, test, student)
    legs["verdict"] = {k: kind_verdict(legs, k) for k in KINDS}
    legs["kinds_passing"] = passing_kinds(legs)
    return legs


def run_fit(shards_root: Path | str, prior_path: Path | str, out: Path | str, *, human_dir: Path | str | None = None, cfg: FitConfig | None = None,
            device: str | None = None, exam_ids: frozenset[str] | None = None, log=print, map_grid: bool = False,
            teacher_prior_path: Path | str | None = None, state_gaps: bool = False) -> dict[str, Any]:
    """Load the shards, fit the student from the prior's weights, write `student.npz`, `fit.json` and `legs.json` into `out`. Returns the legs.
    `map_grid`: the prior is the v2 prior (a map tower), the shards need their `rows_map.npz` and the human shards the map grid; the student
    carries the tower. `teacher_prior_path`: the prior the teacher decided with, when it is not `prior_path`. `state_gaps`: the prior is the v3 prior (block
    G's columns), the shards need their `rows_gaps.npz` and the human shards block G; the student carries the columns."""
    import human_prior_infer as hpi  # noqa: PLC0415

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prior = PriorScorer.load(prior_path)
    teacher_prior = PriorScorer.load(teacher_prior_path) if teacher_prior_path else None
    log(f"loading the shards under {Path(shards_root).name}")
    ts = load_shards(shards_root, prior=prior, map_grid=map_grid, teacher_prior=teacher_prior, state_gaps=state_gaps)
    log(f"  {len(ts)} decisions of {len(ts.seeds)} runs; dropped {ts.stats['dropped']}")
    human_train = human_test = None
    if human_dir is not None:
        log("loading the human decisions (fifths 1 to 3 to fit on, fifth 4 for leg 2)")
        human_train = load_human(human_dir, HUMAN_TRAIN_FIFTHS, exam_ids=exam_ids, map_grid=map_grid, state_gaps=state_gaps)
        human_test = load_human(human_dir, (HUMAN_TEST_FIFTH,), exam_ids=exam_ids, map_grid=map_grid, state_gaps=state_gaps)
    res = fit(ts, read_prior_params(prior_path), human=human_train, cfg=cfg, device=device, log=log, state_gaps=state_gaps)
    kinds = sorted({ts.kind_name(i) for i in range(len(ts))}, key=KINDS.index)
    sha = hpi.write_student(out / STUDENT_OUT, res["params"], kinds=kinds, prior_sha256=file_sha256(prior_path),
                            fit={"best_epoch": res["best_epoch"], "config": res["config"], "train_decisions": res["train_decisions"]}, state_gaps=state_gaps)
    legs = run_legs(ts, out / STUDENT_OUT, prior_path, human_test=human_test, exam_ids=exam_ids)
    fit_doc = {"version": VERSION, "written": utc(), "student_sha256": sha, "prior_sha256": file_sha256(prior_path), "data": public_stats(ts),
               "map_grid": bool(map_grid), "state_gaps": bool(state_gaps), "teacher_prior_sha256": file_sha256(teacher_prior_path) if teacher_prior_path else None,
               "fit": {k: res[k] for k in ("history", "best_epoch", "seconds", "config", "device", "train_decisions", "val_decisions")},
               "human": None if human_train is None else {"train_fifths": list(HUMAN_TRAIN_FIFTHS), "test_fifth": HUMAN_TEST_FIFTH,
                                                          "train_decisions": int(len(human_train.dec))}}
    (out / FIT_FILE).write_text(json.dumps(fit_doc, indent=1), encoding="utf-8")
    (out / LEGS_FILE).write_text(json.dumps(legs, indent=1), encoding="utf-8")
    return legs


def build_student_candidate(out: Path | str, base: Path | str, agent_root: Path | str, target: int, *, why: str, prior_dir: Path | str | None = None,
                            tag: str = "student", export: bool = True, kinds: Sequence[str] | None = None, human_prior: Path | str | None = None,
                            lam: float | None = None) -> Path:
    """The candidate for a fitted student (`run_fit`'s folder `out`): `gen_loop.build_candidate` with the student on the kinds that passed
    (or `kinds`, a subset of them), at their lambdas, the base's human prior, stakes schedule and hold kept; the legs are stored beside the
    student (run_student_legs.json) for `screens.run_student`. Refuses when no kind passed. A base that plays `vrun` alone has no human prior
    to keep: `human_prior` and `lam` give the blend the other screens are decided by."""
    import gen_loop  # noqa: PLC0415

    out = Path(out)
    legs = json.loads((out / LEGS_FILE).read_text(encoding="utf-8"))
    if legs.get("student_sha256") != file_sha256(out / STUDENT_OUT):
        raise ValueError("legs.json is not for this student.npz")
    base_agent = gen_loop.load_agent(base)
    blended = macro_recipe.is_blended(base_agent.macro) or human_prior is not None
    blends = (("card", "ancient", "rest") if blended else ()) + tuple(k for k, _ in base_agent.student_lams)
    take = passing_kinds(legs, blends)
    if kinds is not None:
        take = {k: v for k, v in take.items() if k in set(kinds)}
    if not take:
        raise ValueError("no kind passed both legs with a measured lambda: no candidate")
    # the legs were measured against the prior the fit started from: it must be the base's own prior (or the student it replaces)
    named = base_agent.student or (str(human_prior) if human_prior is not None else None) or base_agent.human_prior
    if (human_prior is None) != (lam is None):
        raise ValueError("the blend needs both the human prior's weights and lambda")
    if (file_sha256(Path(named)) if named else None) != legs["baseline"]["sha256"]:
        raise ValueError("the legs were measured against another prior than the base's (its student's if it has one, else its human prior's)")
    extra = {} if human_prior is None else {"human_prior": human_prior, "lam": lam}
    cand = gen_loop.build_candidate(Path(prior_dir or base_agent.path), base, agent_root, target, why=why, export=export, tag=tag,
                                    student=out / STUDENT_OUT, student_lams=take, **extra)
    (cand / gen_loop.STUDENT_LEGS_FILE).write_text(json.dumps(legs, indent=1), encoding="utf-8")
    return cand


# ---------------------------------------------------------------------------------------------------------- the command line


def _priority() -> None:
    with contextlib.suppress(Exception):
        hs.below_normal()


def cmd_check(a: argparse.Namespace) -> int:
    prior = PriorScorer.load(a.prior) if a.prior else None
    ts = load_shards(a.shards_root, prior=prior, map_grid=a.map_grid, state_gaps=a.state_gaps)
    print(json.dumps(public_stats(ts), indent=1))
    return 0


def _fit_args(a: argparse.Namespace) -> FitConfig:
    return FitConfig(epochs=a.epochs, patience=a.patience, beta=a.beta, kl=a.kl, lr=a.lr, seed=a.seed)


def cmd_fit(a: argparse.Namespace) -> int:
    _priority()
    legs = run_fit(a.shards_root, a.prior, a.out, human_dir=a.human_shards, cfg=_fit_args(a), device=a.device, map_grid=a.map_grid,
                   teacher_prior_path=a.teacher_prior, state_gaps=a.state_gaps)
    print(json.dumps({"kinds_passing": legs["kinds_passing"], "verdict": {k: v["pass"] for k, v in legs["verdict"].items()}}, indent=1))
    return 0


def cmd_candidate(a: argparse.Namespace) -> int:
    cand = build_student_candidate(a.out, a.base, a.agent_root, a.target, why=a.why, tag=a.tag, export=not a.no_export, prior_dir=a.prior_dir,
                                   human_prior=a.human_prior, lam=a.lam)
    print(cand.as_posix())
    return 0


def cmd_round(a: argparse.Namespace) -> int:
    _priority()
    legs = run_fit(a.shards_root, a.prior, a.out, human_dir=a.human_shards, cfg=_fit_args(a), device=a.device, map_grid=a.map_grid,
                   teacher_prior_path=a.teacher_prior, state_gaps=a.state_gaps)
    print(json.dumps({"kinds_passing": legs["kinds_passing"]}, indent=1))
    if a.no_build:
        return 0
    if not legs["kinds_passing"]:
        print("no kind passed both legs: no candidate is built")
        return 3
    cand = build_student_candidate(a.out, a.base, a.agent_root, a.target, why=a.why, tag=a.tag, export=not a.no_export, prior_dir=a.prior_dir,
                                   human_prior=a.human_prior, lam=a.lam)
    print(cand.as_posix())
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="load the shards and print their counts (no fit)")
    c.add_argument("--shards-root", required=True)
    c.add_argument("--prior")
    c.add_argument("--map-grid", action="store_true", help="the shards must hold rows_map.npz (and the prior, if given, the map tower)")
    c.add_argument("--state-gaps", action="store_true", help="the shards must hold rows_gaps.npz (and the prior, if given, read the state gaps)")
    c.set_defaults(fn=cmd_check)

    def fit_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--shards-root", required=True, help="the teacher's shards (run-teacher/<generation>): folders with decisions.jsonl")
        p.add_argument("--prior", required=True, help="the human prior the teacher played (weights.npz or a candidate's human_prior.npz)")
        p.add_argument("--out", required=True)
        p.add_argument("--human-shards", help="the prior's v1 decision shards (human-prior/decisions-v1; with --map-grid the v2 ones, decisions-v2); without them leg 2 is not measured")
        p.add_argument("--map-grid", action="store_true", help="the map grid: the prior is the v2 prior (map tower), the shards need rows_map.npz and the human shards the grid")
        p.add_argument("--state-gaps", action="store_true", help="the state gaps: the prior is the v3 prior (block G's columns), the shards need rows_gaps.npz and the human shards "
                       "block G (decisions-v3)")
        p.add_argument("--teacher-prior", help="the human prior the teacher played with, when it is not --prior (a map-grid fit starts from the v2 prior; the target uses this one's log-probabilities)")
        p.add_argument("--beta", type=float, default=FitConfig.beta)
        p.add_argument("--kl", type=float, default=FitConfig.kl)
        p.add_argument("--epochs", type=int, default=FitConfig.epochs)
        p.add_argument("--patience", type=int, default=FitConfig.patience)
        p.add_argument("--lr", type=float, default=FitConfig.lr)
        p.add_argument("--seed", type=int, default=FitConfig.seed)
        p.add_argument("--device")

    def cand_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--base", required=True, help="the incumbent: a generation folder or current.json")
        p.add_argument("--agent-root", required=True)
        p.add_argument("--target", type=int, required=True, help="the candidate's generation number")
        p.add_argument("--why", default="run student: the teacher's labels and the human choices (D56)")
        p.add_argument("--tag", default="student")
        p.add_argument("--prior-dir", help="the combat prior export (default: the base's own folder)")
        p.add_argument("--no-export", action="store_true")
        p.add_argument("--human-prior", help="a base that plays vrun alone: the human prior's weights the other screens are blended with")
        p.add_argument("--lambda", dest="lam", type=float, help="... and its lambda")

    f = sub.add_parser("fit", help="fit the student and measure both legs and lambda")
    fit_opts(f)
    f.set_defaults(fn=cmd_fit)
    k = sub.add_parser("candidate", help="build the candidate for a fitted student")
    k.add_argument("--out", required=True)
    cand_opts(k)
    k.set_defaults(fn=cmd_candidate)
    r = sub.add_parser("round", help="fit, measure the legs, and build the candidate when a kind passed")
    fit_opts(r)
    cand_opts(r)
    r.add_argument("--no-build", action="store_true")
    r.set_defaults(fn=cmd_round)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
