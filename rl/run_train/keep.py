"""DESIGN 5.3's keep conditions for V_run, computed on the validation split (held-out runs) of a fit.

A head is one yes/no outcome V's floor distribution can be read as: the Act-1 boss kill (final floor past 17, or a win) and the
win (final floor 53). For a head with positives the conditions are:

  * Brier skill score > 0 against the base-rate model (the training split's rate), the paired bootstrap CI over runs excluding 0;
  * the same against V0, the hand-written "arrive healthy" value (below), fitted only as far as its score-to-probability table;
  * log-loss lower than the base rate's;
  * ECE <= 3 pp, over the equal-width bins that hold positives of at least 30 distinct runs (DESIGN counts wins / boss kills,
    and the rows of one run are correlated outcomes of one run; no such bin: not passed).

Tier-B regret against V0 and the battery gain need the lab's decision sets and are `not_computed`. A head with no positives in
the validation split is recorded as `no_positives` and judges nothing. `all_computable_pass` is true when at least one head has
positives and every computable condition of every such head passes. The maths takes plain arrays; `evaluate_value_net` reads a
fit's `ValueData` and trained net.
"""

from __future__ import annotations

from typing import Any

import numpy as np

BINS = 10  # ECE: equal-width bins on [0, 1]
MIN_POSITIVES = 30
ECE_MAX = 0.03
BOOTSTRAP = 2000
CI = (2.5, 97.5)
ACT1_LAST_FLOOR = 17  # the same cut as gen_loop.boss_kill
NOT_COMPUTED = {"status": "not_computed", "why": "needs the lab's Tier-B decision sets / the paired battery, which no fit has"}


def brier_terms(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.asarray(p, np.float64) - y) ** 2


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    q = np.clip(np.asarray(p, np.float64), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def v0_score(hp_frac: np.ndarray, potions: np.ndarray, act: np.ndarray) -> np.ndarray:
    """V0, the hand-written "arrive healthy" value: HP fraction, a tenth for each potion held (at most three), the act index."""
    return np.asarray(hp_frac, np.float64) + 0.1 * np.minimum(np.asarray(potions, np.float64), 3.0) + np.asarray(act, np.float64)


def v0_probability(train_score: np.ndarray, train_y: np.ndarray, score: np.ndarray, bins: int = BINS) -> np.ndarray:
    """V0's score as a probability: the training rows' outcome rate in V0-score quantile bins (smoothed one row towards the base
    rate), so the baseline is calibrated and only its ranking is hand-written."""
    base = float(np.mean(train_y)) if len(train_y) else 0.0
    edges = np.unique(np.quantile(train_score, np.linspace(0, 1, bins + 1)[1:-1])) if len(train_score) else np.array([])
    tb = np.searchsorted(edges, train_score, side="right")
    table = np.array([(train_y[tb == b].sum() + base) / ((tb == b).sum() + 1.0) for b in range(len(edges) + 1)])
    return table[np.searchsorted(edges, score, side="right")]


def paired_bootstrap_skill(err_model: np.ndarray, err_ref: np.ndarray, run: np.ndarray, n_boot: int = BOOTSTRAP,
                           seed: int = 20261024) -> dict[str, Any]:
    """Brier skill score 1 - sum(err_model) / sum(err_ref), with a percentile CI from resampling whole runs (both models'
    errors move together: the bootstrap is paired)."""
    _, inv = np.unique(run, return_inverse=True)
    s_model, s_ref = np.bincount(inv, weights=err_model), np.bincount(inv, weights=err_ref)
    point = 1.0 - s_model.sum() / s_ref.sum() if s_ref.sum() > 0 else float("nan")
    n = len(s_model)
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)
    den = counts @ s_ref
    boot = 1.0 - (counts @ s_model) / np.where(den > 0, den, np.nan)
    lo, hi = np.nanpercentile(boot, CI) if np.isfinite(boot).any() else (float("nan"), float("nan"))
    return {"value": float(point), "ci": [float(lo), float(hi)], "runs": int(n), "resamples": int(n_boot)}


def expected_calibration_error(p: np.ndarray, y: np.ndarray, run: np.ndarray | None = None, bins: int = BINS,
                               min_positives: int = MIN_POSITIVES) -> dict[str, Any]:
    """ECE over the equal-width bins whose positive rows come from at least `min_positives` distinct runs (`run`: each row's
    run id; none = every row its own run), weighted by bin size among those bins."""
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    run = np.arange(len(p)) if run is None else np.asarray(run)
    idx = np.minimum((p * bins).astype(np.int64), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        n, pos = int(m.sum()), int(y[m].sum())
        pos_runs = int(np.unique(run[m & (y > 0)]).size)
        rows.append({"bin": b, "n": n, "positives": pos, "positive_runs": pos_runs, "mean_pred": float(p[m].mean()) if n else None,
                     "freq": pos / n if n else None, "eligible": pos_runs >= min_positives})
    ok = [r for r in rows if r["eligible"]]
    value = None
    if ok:
        total = sum(r["n"] for r in ok)
        value = float(sum(r["n"] / total * abs(r["mean_pred"] - r["freq"]) for r in ok))
    return {"value": value, "eligible_bins": len(ok), "bins": rows}


def head_conditions(p: np.ndarray, y: np.ndarray, run: np.ndarray, train_y: np.ndarray, v0_p: np.ndarray, *,
                    n_boot: int = BOOTSTRAP) -> dict[str, Any]:
    """The conditions for one head on the validation rows: model probability `p`, label `y`, `run` id of each row; `train_y` is
    the training split's labels (the base rate), `v0_p` V0's probability on the validation rows."""
    y = np.asarray(y, np.float64)
    base = np.full(len(y), float(np.mean(train_y)) if len(train_y) else 0.0)
    vs_base = paired_bootstrap_skill(brier_terms(p, y), brier_terms(base, y), run, n_boot)
    vs_v0 = paired_bootstrap_skill(brier_terms(p, y), brier_terms(v0_p, y), run, n_boot)
    for d in (vs_base, vs_v0):
        d["pass"] = bool(d["ci"][0] > 0.0 and d["value"] > 0.0)
    ll_model, ll_base = log_loss(p, y), log_loss(base, y)
    ece = expected_calibration_error(p, y, run)
    return {"positives": int(y.sum()), "positive_runs": int(np.unique(np.asarray(run)[y > 0]).size), "rows": int(len(y)), "base_rate": float(base[0]) if len(base) else None,
            "brier_skill_vs_base_rate": vs_base, "brier_skill_vs_v0": vs_v0,
            "log_loss_vs_base_rate": {"model": ll_model, "base_rate": ll_base, "pass": bool(ll_model < ll_base)},
            "ece_bins_ge_30_positives": {**ece, "max": ECE_MAX, "pass": bool(ece["value"] is not None and ece["value"] <= ECE_MAX)},
            "tier_b_regret_vs_v0": dict(NOT_COMPUTED), "battery_gain": dict(NOT_COMPUTED)}


def passes(head: dict[str, Any]) -> bool:
    keys = ("brier_skill_vs_base_rate", "brier_skill_vs_v0", "log_loss_vs_base_rate", "ece_bins_ge_30_positives")
    return all(head[k]["pass"] for k in keys)


def keep_conditions(heads: dict[str, dict | None], validation: dict[str, Any]) -> dict[str, Any]:
    """The `vrun_keep` record: each head's conditions (none, or no positives: `no_positives`) and the verdict."""
    out: dict[str, Any] = {"validation": validation, "heads": {}}
    judged = []
    for name, h in heads.items():
        if h is None or h["positives"] == 0:
            out["heads"][name] = {"status": "no_positives", "positives": 0, "rows": 0 if h is None else int(h["rows"])}
            continue
        h["pass"] = passes(h)
        out["heads"][name] = h
        judged.append(h["pass"])
    out["all_computable_pass"] = bool(judged) and all(judged)
    out["not_computed"] = ["tier_b_regret_vs_v0", "battery_gain"]
    return out


def evaluate_value_net(values: Any, net: Any) -> dict[str, Any]:
    """`run_train.train.fit`'s `evaluate` hook: the keep record of a trained V on the held-out runs of its `ValueData`."""
    import torch  # noqa: PLC0415

    from . import models  # noqa: PLC0415

    held = np.asarray(values.validation, bool)
    val, trn = np.flatnonzero(held), np.flatnonzero(~held)
    final = np.asarray(values.final_floor, np.int64)
    run = np.asarray(values.run_index, np.int64)
    if not len(val) or not len(trn):
        return {"status": "no_split", "validation": {"rows": int(len(val)), "train_rows": int(len(trn))}, "all_computable_pass": False}
    first_boss = models.floor_class(ACT1_LAST_FLOOR + 1)
    labels = {"act1_boss_kill": (final > ACT1_LAST_FLOOR).astype(np.float64), "win": (final >= models.FLOOR_CLASSES).astype(np.float64)}
    probs = {k: np.zeros(len(val)) for k in labels}
    net.eval()
    with torch.no_grad():
        for start in range(0, len(val), 8192):
            rows = val[start:start + 8192]
            logits, _ = net(*[torch.from_numpy(a) for a in models.value_inputs(values.inputs(rows))])
            p = torch.softmax(logits, dim=1).numpy().astype(np.float64)
            probs["act1_boss_kill"][start:start + len(rows)] = p[:, first_boss:].sum(axis=1)
            probs["win"][start:start + len(rows)] = p[:, models.FLOOR_CLASSES - 1]
    v0 = v0_score(values.v0[:, 0], values.v0[:, 1], values.v0[:, 2])
    heads: dict[str, dict | None] = {}
    for name, y in labels.items():
        if y[val].sum() == 0:
            heads[name] = {"positives": 0, "rows": int(len(val))}
            continue
        heads[name] = head_conditions(probs[name], y[val], run[val], y[trn], v0_probability(v0[trn], y[trn], v0[val]))
    info = {"rows": int(len(val)), "runs": int(len(np.unique(run[val]))), "train_rows": int(len(trn)),
            "train_runs": int(len(np.unique(run[trn]))), "split": "held-out tenth of runs by seed hash (run_train.train._held_out)"}
    return keep_conditions(heads, info)
