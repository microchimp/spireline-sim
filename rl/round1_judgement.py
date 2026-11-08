"""refit-judgement (docs/rl/ROUND.md; memo 3.4 and 3.5 step 3): the run judgement V refit on the new fighter's exploring runs and the
human chosen-option rows, with a source input (human or self), the human weight picked on held-out self-play runs (`select_human_weight`:
the lowest clears-Act-2 log-loss among the weights that keep clears Act 1 no worse than the incumbent's; D55 addendum 12), the kept fit the
average of its seeds' weights, and a held-out report (clears Act 1 and Act 2 log-loss, the selection, the D33 keep record, the guards, the
label leg). The ranking check (keep test (2), its skip guard) and the label bound ((3)) are reported by `guards_report` and decide `kept`
only when `judgement.guards` is "gate".

    python rl/round1.py refit-judgement --config round1.json --round-dir DIR

The source input is a column of first-layer weights times 1 for a human row and 0 for a self row; at play the input is 0 ("self"), so the
exported `run_value.npz` is the plain network (the column is stored beside it in `source_input.npz`). FightNet is the incumbent's
(`judgement.fight_net` "keep"): H of a human row was computed by it at build time, so self and human rows read one H.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

import round1_data as rd

CLEARS = ((16, 1, 16), (32, 18, 32))  # (last floor of the act, first and last floor of the rows its head is scored on)


def human_rows(human_dir: Path, fifths: list[int], kinds: list[str]) -> dict[str, np.ndarray]:
    """The chosen option's row of every human decision of `kinds` in `fifths`: dense (350), counts (1513) and the floor class of the
    run's final floor (a won run: the won class)."""
    import human_prior as hp  # noqa: PLC0415
    import human_starts as hs  # noqa: PLC0415
    from run_train import models  # noqa: PLC0415

    ds = hp.Dataset.load(human_dir, fifths, None, kinds)
    hs.assert_no_exam(ds.run_ids, hs.read_exam_ids(hp.EXAM_IDS))
    ok = ~hp._bool_unavailable(ds, False, True, True)  # noqa: SLF001
    idx = np.concatenate([ds.kind_idx[k][ok[ds.kind_idx[k]]] for k in kinds if k in ds.kind_idx]) if kinds else np.zeros(0, np.int64)
    dec = ds.dec
    idx = idx[dec[idx, hp.C["chosen"]] >= 0]
    rows = dec[idx, hp.C["start"]] + dec[idx, hp.C["chosen"]]
    final = np.where(dec[idx, hp.C["won"]] > 0, models.FLOOR_CLASSES, np.clip(dec[idx, hp.C["final"]], 1, models.FLOOR_CLASSES))
    return {"dense": hp.assemble_np_v1(ds.dense[rows]).astype(np.float16), "counts": ds.counts[rows].astype(np.float16),
            "cls": (final - 1).astype(np.int64), "n_decisions": np.int64(len(idx))}


def source_net_class() -> Any:
    from run_train import models  # noqa: PLC0415

    torch = __import__("torch")

    class SourceNet(models.RunValueNet):  # type: ignore[misc,valid-type]
        def __init__(self, hidden: int = 256, dropout: float = 0.1) -> None:
            super().__init__(hidden, dropout)
            self.src_w = torch.nn.Parameter(torch.zeros(hidden))

        def forward(self, dense, counts, src=None):  # noqa: ANN001
            pre = self.hidden1(torch.cat([dense, self.embed(counts)], dim=1))
            if src is not None:
                pre = pre + src[:, None] * self.src_w[None, :]
            h = torch.relu(self.hidden2(self.dropout(torch.relu(pre))))
            return self.floor_head(h), torch.sigmoid(self.end_hp_head(h)).squeeze(1)

    return SourceNet


def clears_logloss(logits: np.ndarray, final: np.ndarray, floor: np.ndarray) -> dict[str, float]:
    """Log-loss of the two clears heads read off the floor-class distribution: P(final > 16) on rows at floors 1-16, P(final > 32) on rows at
    floors 18-32 (the keep test's row ranges)."""
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    out = {}
    for name, (last, lo, hi) in zip(("clears_act1", "clears_act2"), CLEARS, strict=True):
        sel = (floor >= lo) & (floor <= hi)
        if sel.sum() < 20:
            out[name] = float("nan")
            continue
        q = np.clip(p[sel][:, last:].sum(1), 1e-6, 1 - 1e-6)
        y = (final[sel] > last).astype(float)
        out[name] = float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())
    return out


def train_one(cfg: dict, state: dict, self_d: dict, hum: dict, hw: float, seed: int, dev: Any) -> dict:
    """One fit from the incumbent's weights: rows of self (weight 1) and human (weight `hw`), cross-entropy on the floor class plus
    end_hp_weight x MSE on the self rows' end-HP; returns the final state dict (CPU) and the held-out clears log-loss."""
    import torch  # noqa: PLC0415

    J = cfg["judgement"]
    net = source_net_class()(256, 0.1)
    net.load_state_dict({**state, "src_w": torch.zeros(256)}, strict=True)
    net.to(dev)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero(~self_d["val"])
    n_self, n_h = len(tr), len(hum["cls"])
    D = torch.from_numpy(np.concatenate([self_d["dense"][tr], hum["dense"]])).to(dev)
    C = torch.from_numpy(np.concatenate([self_d["counts"][tr], hum["counts"]])).to(dev)
    tgt = torch.from_numpy(np.concatenate([self_d["target"][tr], np.eye(53, dtype=np.float32)[hum["cls"]]])).to(dev)
    hp_t = torch.from_numpy(np.concatenate([self_d["end_hp"][tr], np.zeros(n_h, np.float32)])).to(dev)
    src = torch.from_numpy(np.concatenate([np.zeros(n_self, np.float32), np.ones(n_h, np.float32)])).to(dev)
    w = torch.from_numpy(np.concatenate([np.ones(n_self, np.float32), np.full(n_h, hw, np.float32)])).to(dev)
    wh = torch.from_numpy(np.concatenate([np.ones(n_self, np.float32), np.zeros(n_h, np.float32)])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=J["lr"], weight_decay=J["weight_decay"])
    N = n_self + n_h
    for _ in range(J["epochs"]):
        net.train()
        perm = rng.permutation(N)
        for i in range(0, N, J["batch"]):
            b = torch.from_numpy(perm[i:i + J["batch"]]).to(dev)
            logits, end = net(D[b].float(), C[b].float(), src[b])
            ce = -(tgt[b] * torch.log_softmax(logits, 1)).sum(1)
            mse = (end - hp_t[b]) ** 2
            loss = ((ce + J["end_hp_weight"] * mse * wh[b]) * w[b]).sum() / w[b].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval()
    net.cpu()
    return {"state": {k: v.detach().clone() for k, v in net.state_dict().items()}, "hw": hw, "seed": seed}


def nanmean(xs: list[float]) -> float:
    ok = [x for x in xs if not np.isnan(x)]
    return float(np.mean(ok)) if ok else float("nan")


def held_out_score(state: dict, self_d: dict) -> dict[str, float]:
    import torch  # noqa: PLC0415

    net = source_net_class()(256, 0.1)
    net.load_state_dict(state, strict=True)
    net.eval()
    va = np.flatnonzero(self_d["val"])
    outs = []
    with torch.no_grad():
        for i in range(0, len(va), 8192):
            b = va[i:i + 8192]
            outs.append(net(torch.from_numpy(self_d["dense"][b]).float(), torch.from_numpy(self_d["counts"][b]).float())[0].numpy())
    logits = np.concatenate(outs) if outs else np.zeros((0, 53), np.float32)
    ll = clears_logloss(logits, self_d["final"][va], self_d["floor"][va])
    ll["mean"] = nanmean(list(ll.values()))
    return ll


def average_states(states: list[dict]) -> dict:
    import torch  # noqa: PLC0415

    return {k: torch.stack([s[k].float() for s in states]).mean(0) for k in states[0]}


def self_arrays(values: Any, fight: Any, chunk: int = 8192) -> dict[str, np.ndarray]:
    """V's self-play rows as the net reads them (dense 350, counts 1513) with their targets, from a ValueData whose H is filled."""
    from run_train import models  # noqa: PLC0415

    values.fill_fight_block(fight)
    dense, counts = [], []
    for s in range(0, len(values.targets), chunk):
        d, c = models.value_inputs(values.inputs(np.arange(s, min(s + chunk, len(values.targets)))))
        dense.append(d.astype(np.float16))
        counts.append(c.astype(np.float16))
    return {"dense": np.concatenate(dense), "counts": np.concatenate(counts), "target": values.targets.astype(np.float32),
            "end_hp": values.end_hp.astype(np.float32), "val": np.asarray(values.validation, bool),
            "final": np.asarray(values.final_floor, np.int64), "floor": np.asarray(values.floor, np.int64)}


def _finite(x: Any) -> bool:
    return x is not None and not np.isnan(x)


def select_human_weight(grid: list[dict], incumbent: dict, rule: str = "clears_act2", act1_tolerance: float = 0.0) -> dict:
    """The selection step over the human-weight grid. `grid` rows carry `human_weight`, `clears_act1`, `clears_act2` and `mean` (held-out
    log-loss of the seed-averaged fits, lower is better); `incumbent` the same figures for the fit the round starts from.

    Rule "clears_act2" (D55 addendum 12): among the weights whose clears-Act-1 log-loss is not worse than the incumbent's by more than
    `act1_tolerance`, the lowest clears-Act-2 log-loss; if no weight keeps clears Act 1, the one with the lowest clears-Act-1 log-loss
    (`fallback` says so). Rule "mean" is the first draft's lowest mean of both heads. A figure that could not be computed (NaN: under 20
    held-out rows) ranks last. Returns `chosen`, the whole `grid` of weights, `table` (every weight's figures and whether it was eligible),
    the `incumbent`'s figures, the `rule`, a `fallback` (None, or why the rule could not choose) and a `reason` in words."""
    if not grid:
        raise ValueError("the human-weight grid is empty")
    ceiling = incumbent.get("clears_act1")
    table = []
    for g in grid:
        ok = _finite(g["clears_act1"]) and (not _finite(ceiling) or g["clears_act1"] <= ceiling + act1_tolerance + 1e-12)
        table.append({"human_weight": g["human_weight"], "clears_act1": g["clears_act1"], "clears_act2": g["clears_act2"], "mean": g["mean"],
                      "eligible": bool(ok)})

    def worst_last(key: str):
        return lambda r: (not _finite(r[key]), r[key] if _finite(r[key]) else 0.0)

    fallback = None
    if rule == "mean":
        pick = min(table, key=worst_last("mean"))
        reason = f"lowest mean held-out clears log-loss ({pick['mean']:.4f})"
    else:
        pool = [r for r in table if r["eligible"]]
        if not pool:
            pick = min(table, key=worst_last("clears_act1"))
            fallback = "no weight keeps clears-Act-1 log-loss within the incumbent's; the lowest clears-Act-1 log-loss is kept"
            reason = fallback
        else:
            pick = min(pool, key=worst_last("clears_act2"))
            if not _finite(pick["clears_act2"]):
                fallback = "clears-Act-2 log-loss could not be computed on the held-out runs; the first eligible weight is kept"
                pick = pool[0]
            reason = fallback or (f"lowest held-out clears-Act-2 log-loss ({pick['clears_act2']:.4f}) among the {len(pool)} of {len(table)} "
                                  f"weights with clears Act 1 not worse than the incumbent's ({ceiling if _finite(ceiling) else 'n/a'})")
    return {"rule": rule, "chosen": pick["human_weight"], "grid": [g["human_weight"] for g in grid], "table": table,
            "incumbent": {k: incumbent.get(k) for k in ("clears_act1", "clears_act2", "mean")}, "act1_tolerance": act1_tolerance,
            "fallback": fallback, "reason": reason}


def guards_report(keep_doc: dict, leg: dict, J: dict) -> dict:
    """The keep test's guards, as numbers (D55 addendum 12: reported, and they decide `kept` only when `J["guards"]` is "gate").

    * `ranking`: keep test (2), the ranking check of `run_train.ranking` (rest rate, heal rate, human agreement): its verdict and each check;
    * `skip_guard`: the candidate's card skip rate against the incumbent's on the ranking set, a fall of more than `skip_guard_points` per
      100 fails unless the label leg favours the candidate (the strategy memo's (2)); None figures when the ranking check did not run;
    * `label_bound`: keep test (3), the new judgement with the new prior against the incumbent's blend on the held-out labelled decisions,
      the lower end of the paired 95% interval against `label_bound_floors` (a floor a decision); the memo's section 4: at 240 decisions this
      refuses a level candidate about 83 times in 100, which is why it is reported and not a gate.
    `pass` is the guards together (a figure that is not there passes): what `kept` needs under "gate"."""
    mode = J["guards"]
    rk = keep_doc.get("ranking")
    ranking = {"ran": rk is not None and rk.get("status") not in (None, "not_run"), "status": (rk or {}).get("status"),
               "pass": None if rk is None else bool(rk.get("pass")),
               "checks": [{k: c.get(k) for k in ("name", "candidate", "incumbent", "diff", "limit", "pass")} for c in (rk or {}).get("checks") or []]}
    row = (leg or {}).get("all") or {}
    diff = row.get("new_judgement_new_prior_minus_incumbent_blend") or {}
    bound = float(J["label_bound_floors"])
    lo = diff.get("lo95")
    label_bound = {"n": row.get("n"), "diff_floors": diff.get("mean"), "se": diff.get("se"), "lo95": lo, "bound": bound,
                   "ok": None if lo is None else bool(lo > bound), "reading": "report only unless guards is gate"}
    favours = None if lo is None else bool(lo > 0)
    skip = (((rk or {}).get("shard") or {}).get("skip_rate") or {})
    fall = None if skip.get("incumbent") is None or skip.get("candidate") is None else 100.0 * (skip["incumbent"] - skip["candidate"])
    limit = float(J["skip_guard_points"])
    skip_guard = {"fall_points": fall, "limit_points": limit, "n": skip.get("n"), "label_leg_favours": favours,
                  "ok": None if fall is None else bool(fall <= limit or favours)}
    parts = [ranking["pass"], label_bound["ok"], skip_guard["ok"]]
    return {"mode": mode, "ranking": ranking, "skip_guard": skip_guard, "label_bound": label_bound,
            "pass": all(p for p in parts if p is not None), "decides_kept": mode == "gate"}


def refit(cfg: dict, *, values: Any, fight_net: Any, value_net_state: dict, hum: dict, out_dir: Path, meta: dict) -> dict:
    """The fit: for each human weight of the grid, `seeds` fits from the incumbent; `select_human_weight` picks the weight (its rule is
    `judgement.weight_rule`) and its seeds' weights are averaged. Writes run_value.pt/.npz (plain RunValueNet) and source_input.npz;
    returns the report (`selection`: the chosen weight, the grid and every weight's figures)."""
    import torch  # noqa: PLC0415

    from run_train import models  # noqa: PLC0415

    J = cfg["judgement"]
    t0 = time.time()
    torch.set_num_threads(int(J["threads"]))
    dev = torch.device("cuda" if torch.cuda.is_available() and J.get("device", "cuda") == "cuda" else "cpu")
    from run_train import models as m  # noqa: PLC0415

    fight = m.NumpyFightNet(m.numpy_params(fight_net))
    self_d = self_arrays(values, fight)
    base = {**value_net_state, "src_w": torch.zeros(256)}
    incumbent_score = held_out_score(base, self_d)
    grid: list[dict] = []
    fits: dict[float, list[dict]] = {}
    for hw in J["human_weights"]:
        fits[hw] = [train_one(cfg, base, self_d, hum, float(hw), int(s), dev) for s in J["seeds"]]
        scores = [held_out_score(f["state"], self_d) for f in fits[hw]]
        grid.append({"human_weight": hw, "seeds": [s["mean"] for s in scores], "mean": nanmean([s["mean"] for s in scores]),
                     "clears_act1": nanmean([s["clears_act1"] for s in scores]),
                     "clears_act2": nanmean([s["clears_act2"] for s in scores])})
    selection = select_human_weight(grid, incumbent_score, J.get("weight_rule", "clears_act2"), float(J.get("act1_tolerance", 0.0)))
    best = next(g for g in grid if g["human_weight"] == selection["chosen"])
    kept_state = average_states([f["state"] for f in fits[best["human_weight"]]])
    final_score = held_out_score(kept_state, self_d)
    value = models.RunValueNet(256, 0.1)
    value.load_state_dict({k: v for k, v in kept_state.items() if k != "src_w"}, strict=True)
    value.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = models.save(out_dir / "run_value.pt", fight_net, value, {**meta, "refit": "round1", "human_weight": best["human_weight"],
                                                                     "seeds": list(J["seeds"])})
    np.savez(out_dir / "source_input.npz", src_w=kept_state["src_w"].numpy().astype(np.float32),
             note=np.array("first-layer weights times 1 for a human row, 0 for self; play is 0 (not applied in run_value.npz)"))
    return {"npz": str(npz), "value": value, "human_weight": best["human_weight"], "grid": grid, "selection": selection,
            "incumbent_heldout": incumbent_score,
            "kept_heldout": final_score, "self_rows": int(len(self_d["target"])), "self_rows_heldout": int(self_d["val"].sum()),
            "human_rows": int(len(hum["cls"])), "human_decisions": int(hum["n_decisions"]), "device": str(dev),
            "wall_seconds": round(time.time() - t0, 1)}
