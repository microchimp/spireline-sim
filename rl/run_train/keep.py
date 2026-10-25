"""DESIGN 5.3's keep conditions for V_run.

Since D33 (2026-10-25) the verdict is the out-of-fold one (`cross_fit_keep`): the Act-1 boss-kill head's Brier skill against
the base rate, pooled over a 5-fold cross-fit by run on the whole deduplicated window (the folds are the hash fifths of the
run seeds; each fold's rows are predicted by nets fitted on the other four), kept when the 95% paired-bootstrap interval is
above zero. Against V0, the log-loss and the ECE are reported with their pass flags on the same rows, not gating. The held-out
tenth below (~37 kill runs, an interval that crosses zero at this skill size) is still computed by `evaluate_value_net` and
reported in the record under `held_out_tenth`; it does not gate.

The tenth's conditions, computed on the validation split (held-out runs) of a fit:

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
# The Act-1 head is judged on rows before the boss room. A run's decisions on floor 17 (the boss's rewards, the next map) and
# past it are all taken after the kill, so their label is fixed by the state and would inflate every condition.
ACT1_HEAD_LAST_FLOOR = ACT1_LAST_FLOOR - 1
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
                    n_boot: int = BOOTSTRAP, base_p: np.ndarray | None = None) -> dict[str, Any]:
    """The conditions for one head on the validation rows: model probability `p`, label `y`, `run` id of each row; `train_y` is
    the training split's labels (the base rate), `v0_p` V0's probability on the validation rows. `base_p` (one base rate per row)
    replaces the constant from `train_y`: the pooled out-of-fold rows each carry their own fold's training rate."""
    y = np.asarray(y, np.float64)
    base = np.asarray(base_p, np.float64) if base_p is not None else np.full(len(y), float(np.mean(train_y)) if len(train_y) else 0.0)
    vs_base = paired_bootstrap_skill(brier_terms(p, y), brier_terms(base, y), run, n_boot)
    vs_v0 = paired_bootstrap_skill(brier_terms(p, y), brier_terms(v0_p, y), run, n_boot)
    for d in (vs_base, vs_v0):
        d["pass"] = bool(d["ci"][0] > 0.0 and d["value"] > 0.0)
    ll_model, ll_base = log_loss(p, y), log_loss(base, y)
    ece = expected_calibration_error(p, y, run)
    return {"positives": int(y.sum()), "positive_runs": int(np.unique(np.asarray(run)[y > 0]).size), "rows": int(len(y)), "base_rate": float(np.mean(base)) if len(base) else None,
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
    floor = np.asarray(values.floor, np.int64)  # each row's own floor (ValueData.floor)
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
    scope = {"act1_boss_kill": floor <= ACT1_HEAD_LAST_FLOOR, "win": np.ones(len(floor), bool)}
    heads: dict[str, dict | None] = {}
    for name, y in labels.items():
        sv, st = scope[name][val], scope[name][trn]
        yv, yt = y[val][sv], y[trn][st]
        if yv.sum() == 0:
            heads[name] = {"positives": 0, "rows": int(sv.sum())}
            continue
        heads[name] = head_conditions(probs[name][sv], yv, run[val][sv], yt, v0_probability(v0[trn][st], yt, v0[val][sv]))
        # informational, never gated: V0 plus the floor's progress through the act (V reads the floor, V0 does not), so the
        # skill over V0 can be told apart from "knows how far into the act the run is"
        v0f = v0 + floor / ACT1_LAST_FLOOR
        pf = v0_probability(v0f[trn][st], yt, v0f[val][sv])
        heads[name]["brier_skill_vs_v0_floor_informational"] = {
            **paired_bootstrap_skill(brier_terms(probs[name][sv], yv), brier_terms(pf, yv), run[val][sv]), "gated": False,
            "baseline": "V0 + floor / 17, through the same training-row quantile table as V0"}
    info = {"act1_head_last_floor": ACT1_HEAD_LAST_FLOOR, "rows": int(len(val)), "runs": int(len(np.unique(run[val]))), "train_rows": int(len(trn)),
            "train_runs": int(len(np.unique(run[trn]))), "split": "held-out tenth of runs by seed hash (run_train.train._held_out)",
            "early_stopping_split": True,
            "early_stopping_note": ("the same held-out tenth chose the fit's best epoch (train._optimise, patience 3), so these "
                                    "conditions are mildly optimistic")}
    return keep_conditions(heads, info)


# ------------------------------------------------------------------ D33: the out-of-fold keep test

FOLDS = 5  # the sweep's cross-fitting folds (VRUN_SWEEP.md 1.1), here over every run of the deduplicated window
OOF_RULE = ("D33 (2026-10-25): kept when the out-of-fold Act-1 boss-kill Brier skill against the base rate has its 95% paired-bootstrap "
            "interval above zero; V0, log-loss and ECE are reported, the held-out tenth is reported")


def fold_of(hashes: np.ndarray, folds: int = FOLDS) -> np.ndarray:
    """Each run's fold 0..folds-1: its seed hash (`train._unit_hash`) in equal fifths."""
    return np.minimum((np.asarray(hashes, np.float64) * folds).astype(np.int64), folds - 1)


def early_stop_runs(hashes: np.ndarray) -> np.ndarray:
    """The runs a fold's fits stop on: the lowest hundredth of every hash tenth, one run in ten, spread over the folds alike."""
    return np.mod(np.asarray(hashes, np.float64) * 10.0, 1.0) < 0.1


def keep_record_oof(heads: dict[str, dict | None], validation: dict[str, Any], tenth: dict[str, Any] | None) -> dict[str, Any]:
    """The `vrun_keep` record under D33: a head is kept on the Brier skill against the base rate alone; its other conditions carry
    `pass` flags (`all_conditions_pass`) without gating; `tenth` (`evaluate_value_net`'s record) is reported, never gating."""
    out: dict[str, Any] = {"rule": OOF_RULE, "validation": validation, "heads": {}}
    judged = []
    for name, h in heads.items():
        if h is None or h["positives"] == 0:
            out["heads"][name] = {"status": "no_positives", "positives": 0, "rows": 0 if h is None else int(h["rows"])}
            continue
        h["all_conditions_pass"] = passes(h)
        h["pass"] = bool(h["brier_skill_vs_base_rate"]["pass"])
        out["heads"][name] = h
        judged.append(h["pass"])
    out["all_computable_pass"] = bool(judged) and all(judged)
    out["gating"] = "brier_skill_vs_base_rate"
    out["held_out_tenth"] = {"gating": False, "all_computable_pass": None if tenth is None else tenth.get("all_computable_pass"),
                             "record": tenth}
    out["not_computed"] = ["tier_b_regret_vs_v0", "battery_gain"]
    return out


def fit_fold_nets(fights: Any, values: Any, config: Any, train_rows: np.ndarray, held_rows: np.ndarray, fold: int) -> dict[str, Any]:
    """Fit FightNet on the fights of the other folds, fill H for every row from it, fit V on `train_rows` (stopping on one run in
    ten of them) and return the fold's rows' P(Act-1 boss kill) and P(win) with the fit's numbers."""
    import dataclasses  # noqa: PLC0415
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from . import models, train  # noqa: PLC0415

    t0 = time.perf_counter()
    fh = np.asarray(fights.hashes, np.float64)
    frows = np.flatnonzero(fold_of(fh) != fold)
    fstop = early_stop_runs(fh[frows])
    if not fstop.any():
        fstop = train._held_out(fh[frows], 0.1)
    fdata = train.FightData(bases=fights.bases.select(frows), encounters=fights.encounters[frows], weak=fights.weak[frows],
                            labels=fights.labels[frows], validation=fstop, hashes=fh[frows])
    torch.manual_seed(1000 + fold)
    generator = torch.Generator().manual_seed(1000 + fold)
    fight_net = models.FightNet()
    fight_stats = train.train_fight_net(fight_net, fdata, config, generator)
    filled = dataclasses.replace(values, fight_block=None)
    filled.fill_fight_block(models.NumpyFightNet(models.numpy_params(fight_net)))
    vh = np.asarray(values.run_hash, np.float64)[train_rows]
    vstop = early_stop_runs(vh)
    if not vstop.any():
        vstop = train._held_out(vh, 0.1)
    data = dataclasses.replace(
        filled, bases=filled.bases.select(train_rows), query_keys=filled.query_keys[train_rows], targets=filled.targets[train_rows],
        end_hp=filled.end_hp[train_rows], validation=vstop, fight_block=filled.fight_block[train_rows],
        final_floor=filled.final_floor[train_rows], floor=filled.floor[train_rows], run_index=filled.run_index[train_rows],
        v0=filled.v0[train_rows], run_hash=filled.run_hash[train_rows], runs=int(len(np.unique(filled.run_index[train_rows]))))
    value_net = models.RunValueNet(hidden=config.hidden, dropout=config.dropout)
    value_stats = train.train_value_net(value_net, data, config, generator)
    first_boss = models.floor_class(ACT1_LAST_FLOOR + 1)
    kill, win = np.zeros(len(held_rows)), np.zeros(len(held_rows))
    value_net.eval()
    with torch.no_grad():
        for start in range(0, len(held_rows), 8192):
            rows = held_rows[start:start + 8192]
            logits, _ = value_net(*[torch.from_numpy(a) for a in models.value_inputs(filled.inputs(rows))])
            p = torch.softmax(logits, dim=1).numpy().astype(np.float64)
            kill[start:start + len(rows)] = p[:, first_boss:].sum(axis=1)
            win[start:start + len(rows)] = p[:, models.FLOOR_CLASSES - 1]
    return {"act1_boss_kill": kill, "win": win,
            "info": {"fight_epochs": fight_stats.get("epochs"), "value_epochs": value_stats.get("epochs"),
                     "value_best_validation": value_stats.get("best_validation"), "fit_min": round((time.perf_counter() - t0) / 60, 2)}}


def cross_fit_keep(fights: Any, values: Any, config: Any, tenth: dict[str, Any] | None = None, *, fit_fold: Any = None,
                   n_boot: int = BOOTSTRAP, folds: int = FOLDS) -> dict[str, Any]:
    """`run_train.train.fit`'s `cross_fit` hook: the D33 keep record. Each fold's runs are predicted by nets fitted without them
    (`fit_fold(fights, values, config, train_rows, held_rows, fold)`, `fit_fold_nets` by default); the pooled rows are judged
    like the tenth's (floor <= 16 rows for the Act-1 head), each row against its own fold's training base rate and V0 table."""
    final = np.asarray(values.final_floor, np.int64)
    floor = np.asarray(values.floor, np.int64)
    run = np.asarray(values.run_index, np.int64)
    row_fold = fold_of(values.run_hash, folds)
    from . import models  # noqa: PLC0415

    labels = {"act1_boss_kill": (final > ACT1_LAST_FLOOR).astype(np.float64), "win": (final >= models.FLOOR_CLASSES).astype(np.float64)}
    scope = {"act1_boss_kill": floor <= ACT1_HEAD_LAST_FLOOR, "win": np.ones(len(floor), bool)}
    probs = {k: np.full(len(final), np.nan) for k in labels}
    v0 = v0_score(values.v0[:, 0], values.v0[:, 1], values.v0[:, 2])
    fit_fold = fit_fold or fit_fold_nets
    done, fold_info = [], []
    for f in range(folds):
        held, trn = np.flatnonzero(row_fold == f), np.flatnonzero(row_fold != f)
        if not len(held) or len(np.unique(run[trn])) < 2:
            fold_info.append({"fold": f, "skipped": "no held-out rows, or fewer than two training runs", "rows": int(len(held))})
            continue
        got = fit_fold(fights, values, config, trn, held, f)
        for name in labels:
            probs[name][held] = got[name]
        done.append((held, trn))
        fold_info.append({"fold": f, "held_runs": int(len(np.unique(run[held]))), "held_rows": int(len(held)), "train_rows": int(len(trn)),
                          **got.get("info", {})})
    heads: dict[str, dict | None] = {}
    for name, y in labels.items():
        parts: dict[str, list] = {"p": [], "y": [], "run": [], "base": [], "v0": []}
        for held, trn in done:
            sh, st = scope[name][held], scope[name][trn]
            if not sh.any():
                continue
            yt = y[trn][st]
            parts["p"].append(probs[name][held][sh])
            parts["y"].append(y[held][sh])
            parts["run"].append(run[held][sh])
            parts["base"].append(np.full(int(sh.sum()), float(yt.mean()) if len(yt) else 0.0))
            parts["v0"].append(v0_probability(v0[trn][st], yt, v0[held][sh]))
        cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in parts.items()}
        if not len(cat["y"]) or cat["y"].sum() == 0:
            heads[name] = {"positives": 0, "rows": int(len(cat["y"]))}
            continue
        heads[name] = head_conditions(cat["p"], cat["y"], cat["run"], np.zeros(0), cat["v0"], n_boot=n_boot, base_p=cat["base"])
    pooled = np.concatenate([held for held, _ in done]) if done else np.zeros(0, np.int64)
    info = {"split": f"{folds}-fold cross-fit by run seed hash over the whole deduplicated window (D33)", "folds": fold_info,
            "act1_head_last_floor": ACT1_HEAD_LAST_FLOOR, "rows": int(len(pooled)), "runs": int(len(np.unique(run[pooled]))),
            "base_rate": "per fold: the rate over that fold's training rows (floor <= 16 for the Act-1 head)"}
    return keep_record_oof(heads, info, tenth)
