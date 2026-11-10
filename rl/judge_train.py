#!/usr/bin/env python3
"""The fight judge trained on the corpus (D58 C2, D57 item 1; judge v2).

    python rl/judge_train.py train --out DIR/judge.pt [--store CORPUS] [--init AGENT/combat_net.pt] [--compare A.pt B.pt]   (GPU venv)
    python rl/judge_train.py eval  --judge A.pt [--judge B.pt ...] [--store CORPUS] [--out eval.json] [--md eval.md]
    python rl/judge_train.py summary [--store CORPUS]                                                       (counts the trainer would read)

What it does. `judge.JudgeNet` (the combat network's trunk with the judge head: win logit, end-HP-fraction logit, potions held) is trained
on the judge corpus (`judge_corpus.py`, `JUDGE_CORPUS.md`): fight positions of the human starts (the teacher's replay with the human's own
result of the fight and, at boss and elite turns, the search settled on a ladder of budgets) and of the always-on loop's self-play (the
x16 teacher's search as played and the fight's outcome). Four soft labels train the three outputs; each loss term is a weighted mean over
the rows that carry its label, and `LossCfg` holds its weight:

  teacher outcome   the very game the position was in (won, end HP, potions held): the judge's old target, one draw from the distribution.
  human outcome     the human's own result of that fight (human rows only); a label of the fight from the door, not of the turn. A human
                    plays the fights better than the teacher, so the same position has two outcome distributions: the judge head reads a
                    SOURCE input (`JudgeNet.src_emb`: row 0, the teacher, is zero and is what the search always asks; row 1 is added for the
                    human reading), and only the human reading is trained to this label.
  search value      the judge's own value V = P(win) x (end HP + 10 x potions) + P(death) x -100 (what the search reads at a leaf,
                    `judge_client.estimate`) against the search's root value, in HP units (Huber on V / 50); a settled turn weighs most,
                    a soft turn less, a turn kept as played least (`tier_w`).
  search win        the win logit against 1 - the search's death share at the root (rows that have one).
  moves (aux.)      the policy head (already in `JudgeNet`: one logit a action token) against the search's visit distribution over the
                    action slots; it only shapes the trunk, the judge never reads it, and `--w-policy 0` leaves it out.

Rows weigh by the corpus's sampling weight (`--weights pool`: the human rows by 1 / inclusion probability, which brings the oversampled
strata back to the pool's mix, self-play 1; `cell`, `none`, `balanced` as `judge_corpus.row_weight`), clipped at `--wclip` times the mean
and renormalised. The split is the corpus's: by RUN, one in five held out (`split` in the rows; the exam fifth is never inside the corpus
and the loader refuses it), the validation tenth for the epoch and the win calibration by run as well (`judge.val_of(run)`).

`eval` and the end of `train` score judges on the same held-out rows: per cell (act x kind) and per source, the L0 against the teacher's
outcome and the human's (Pearson, Spearman, win Brier, calibration error), and V against the search's settled and soft root values.
Lab code: nothing here changes the engine, the search or the loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import judge as jd  # noqa: E402
import judge_corpus as jc  # noqa: E402
from combat_tokens import MAX_ACTIONS, TYPE_PAD, VOCAB  # noqa: E402
from judge_client import DEATH_VALUE, POTION_VALUE  # noqa: E402

VERSION = "judge-v2"
CELLS = ("act1-boss", "act1-elite", "act1-hallway", "act2-boss", "act2-elite", "act2-hallway")
TIER_CODE = {"played": 0, "soft": 1, "settled": 2}
TIERS = tuple(TIER_CODE)
SPLIT_CODE = {"train": 0, "val": 1, "heldout": 2}
STAGE_CODE = {"early": 0, "middle": 1, "late": 2, "door": 3}
STORE = jc.STORE
HEADLINE_CELLS = ("act1-boss", "act2-boss")  # the board's cells


# ---------------------------------------------------------------------------------------------------------- reading the corpus


def cell_of(row: dict) -> str:
    return f"act{row['act']}-{row['kind']}"


def policy_target(children: list[dict]) -> np.ndarray | None:
    """The search's visit distribution over the action slots (the policy head's target: token 1 + a is action a): each child's visits
    added to its slot (a target's moves share a slot), normalised; None when there is nothing to learn from."""
    out = np.zeros(MAX_ACTIONS, np.float64)
    for c in children or []:
        a = int(c["a"])
        if 0 <= a < MAX_ACTIONS and (c.get("n") or 0) > 0:
            out[a] += c["n"]
    total = out.sum()
    return None if total <= 0 else (out / total).astype(np.float16)


def extract(row: dict) -> dict:
    """One corpus row as the trainer's columns (pure): the teacher's and the human's outcome labels (with whether the row has one),
    the search's root value, best value, death share and tier, the sampling weights, the policy target, and the strata."""
    lab = row["label"]
    teach = jc.label_row(row, "teacher")
    human = jc.label_row(row, "human") if row.get("human") else None

    def num(x):
        return float("nan") if x is None else float(x)

    pol = policy_target(lab.get("children"))
    return {"y_teacher": teach or [0.0, 0.0, 0.0, 0.0], "has_teacher": teach is not None, "y_human": human or [0.0, 0.0, 0.0, 0.0],
            "has_human": human is not None, "root_value": num(lab.get("root_value")), "best_value": num(lab.get("best_value")),
            "p_death": num(lab.get("p_death")), "tier": TIER_CODE.get(lab.get("tier"), 0), "w": float(row.get("w", 1.0)),
            "w_cell": float(row.get("w_cell", 1.0)), "pol": pol, "cell": cell_of(row), "stage": STAGE_CODE.get(row.get("stage"), 3),
            "door": bool(row.get("door")), "tag": f"{row['src']}:{row.get('sub', '')}",
            "meta": [row["k"], row["turn"], row["hp"], row["max_hp"], num(lab.get("root_value")), num(lab.get("best_value")),
                     num(lab.get("p_death")), num(lab.get("sims"))]}


def read_corpus(store: Path | str = STORE, require_usable: bool = True, max_per_cell: int | None = None, srcs: tuple[str, ...] = ("human", "selfplay"),
                log=lambda m: print(m, flush=True)) -> tuple[dict[str, np.ndarray], dict]:
    """Every position of the store as arrays (and a report of what was read). Needs the manifest's `usable_for_training: true` unless
    `require_usable` is False (a dry run). `max_per_cell` stops each (source, kind, act) after that many rows (a smoke test)."""
    corpus = jc.Corpus(store)
    man = corpus.manifest
    if require_usable and not man.get("usable_for_training"):
        raise RuntimeError("the corpus manifest does not say usable_for_training: true (the human boss and elite positions are not labelled yet)")
    cols: dict[str, list] = {k: [] for k in ("lens", "types", "ids", "owners", "feats", "meta", "y_teacher", "has_teacher", "y_human", "has_human",
                                              "root_value", "best_value", "p_death", "tier", "w", "w_cell", "pol", "has_pol", "cell", "stage", "door",
                                              "tag", "split", "run", "start")}
    t0 = time.perf_counter()
    for src in srcs:
        for kind in jc.KINDS:
            for act in (1, 2):
                n = 0
                for pos in corpus.iter_positions(src=src, kind=kind, act=act):
                    r = pos.row
                    e = extract(r)
                    cols["lens"].append(len(pos.tokens["types"]))
                    for name in ("types", "ids", "owners", "feats"):
                        cols[name].append(pos.tokens[name])
                    for name in ("meta", "y_teacher", "has_teacher", "y_human", "has_human", "root_value", "best_value", "p_death", "tier", "w",
                                 "w_cell", "cell", "stage", "door", "tag"):
                        cols[name].append(e[name])
                    cols["pol"].append(e["pol"] if e["pol"] is not None else np.zeros(MAX_ACTIONS, np.float16))
                    cols["has_pol"].append(e["pol"] is not None)
                    cols["split"].append(r["split"])
                    cols["run"].append(r["run"])
                    cols["start"].append(r["start"])
                    n += 1
                    if max_per_cell and n >= max_per_cell:
                        break
                if n:
                    log(f"read {src}-{kind}-act{act}: {n} rows ({time.perf_counter() - t0:.0f} s)")
    if not cols["lens"]:
        raise RuntimeError(f"no positions in {store}")
    tag_names = sorted(set(cols["tag"]))
    cell_names = list(CELLS)
    start_names = sorted(set(cols["start"]))
    start_ix = {s: i for i, s in enumerate(start_names)}
    run_of_start: dict[str, str] = {}
    for s, r in zip(cols["start"], cols["run"], strict=True):
        run_of_start[s] = r
    cell_of_start: dict[str, str] = {}
    for s, c in zip(cols["start"], cols["cell"], strict=True):
        cell_of_start[s] = c
    run_val = {r: jd.val_of(r) for r in set(cols["run"])}
    split = np.array([SPLIT_CODE["heldout"] if sp == "heldout" else (SPLIT_CODE["val"] if run_val[r] else SPLIT_CODE["train"])
                      for sp, r in zip(cols["split"], cols["run"], strict=True)], np.int8)
    arrays = {
        "lens": np.array(cols["lens"], np.uint16), "types": np.concatenate(cols["types"]).astype(np.uint8),
        "ids": np.concatenate(cols["ids"]).astype(np.int32), "owners": np.concatenate(cols["owners"]).astype(np.uint8),
        "feats": np.concatenate(cols["feats"]).astype(np.float32), "meta": np.array(cols["meta"], np.float64).reshape(-1, 8),
        "y_teacher": np.array(cols["y_teacher"], np.float32).reshape(-1, 4), "has_teacher": np.array(cols["has_teacher"], bool),
        "y_human": np.array(cols["y_human"], np.float32).reshape(-1, 4), "has_human": np.array(cols["has_human"], bool),
        "root_value": np.array(cols["root_value"], np.float32), "best_value": np.array(cols["best_value"], np.float32),
        "p_death": np.array(cols["p_death"], np.float32), "tier": np.array(cols["tier"], np.int8), "w": np.array(cols["w"], np.float32),
        "w_cell": np.array(cols["w_cell"], np.float32), "pol": np.stack(cols["pol"]).astype(np.float16), "has_pol": np.array(cols["has_pol"], bool),
        "cell": np.array([cell_names.index(c) for c in cols["cell"]], np.int8), "stage": np.array(cols["stage"], np.int8),
        "door": np.array(cols["door"], bool), "tag": np.array([tag_names.index(t) for t in cols["tag"]], np.int16), "split": split,
        "group": np.array([start_ix[s] for s in cols["start"]], np.int32), "start_ids": np.array(start_names),
        "start_run": np.array([run_of_start[s] for s in start_names]), "tags": np.array(tag_names), "cells": np.array(cell_names),
    }
    report = {"rows": int(len(arrays["lens"])), "starts": len(start_names), "runs": len(set(cols["run"])),
              "by_tag": dict(Counter(cols["tag"])), "by_cell": dict(Counter(cols["cell"])), "by_tier": dict(Counter(TIERS[t] for t in cols["tier"])),
              "by_split": dict(Counter(int(s) for s in split)), "human_rows": int(sum(cols["has_human"])), "seconds": round(time.perf_counter() - t0, 1),
              "manifest": {k: man.get(k) for k in ("written", "usable_for_training", "complete", "engine_tag", "player")}}
    return arrays, report


def weights_of(a: dict[str, np.ndarray], mode: str = "pool", clip: float | None = 20.0, human_mult: float = 1.0) -> np.ndarray:
    """The rows' training weights, mean 1 over the TRAINING rows. `pool`: the corpus's `w`; `cell`: `w_cell`; `none`: 1; `balanced`: `w_cell`
    scaled so every (kind, act) cell has the same total. A weight over `clip` times the mean is cut to it first (a rare stratum's weight can
    be large); the human rows are multiplied by `human_mult` before the clip."""
    if mode not in jc.WEIGHT_MODES:
        raise ValueError(f"unknown weight mode {mode!r}")
    n = len(a["lens"])
    if mode == "none":
        w = np.ones(n, np.float64)
    elif mode == "pool":
        w = a["w"].astype(np.float64)
    else:  # cell, balanced
        w = a["w_cell"].astype(np.float64)
        if mode == "balanced":
            counts = np.bincount(a["cell"], minlength=len(a["cells"])).astype(np.float64)
            present = (counts > 0).sum()
            w = w * n / (present * np.maximum(counts[a["cell"]], 1.0))
    human = np.array([str(t).startswith("human:") for t in a["tags"]])[a["tag"]] if len(a["tags"]) else np.zeros(n, bool)
    w = np.where(human, w * human_mult, w)
    train = a["split"] == SPLIT_CODE["train"]
    mean = w[train].mean() if train.any() else w.mean()
    w = w / max(mean, 1e-12)
    if clip:
        w = np.minimum(w, clip)
        mean = w[train].mean() if train.any() else w.mean()
        w = w / max(mean, 1e-12)
    return w.astype(np.float32)


# ---------------------------------------------------------------------------------------------------------- the data on a device


class CorpusData:
    """The corpus arrays on a device in compact dtypes (tokens: uint8 types and owners, int32 ids, float32 features), with the batch the
    judge reads, the targets of the loss, and the split (0 train, 1 validation, 2 held out) of each position."""

    def __init__(self, a: dict[str, np.ndarray], device, weights: np.ndarray | None = None):
        self.a = a
        self.device = device
        self.n = len(a["lens"])
        self.lens = a["lens"].astype(np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lens)[:-1]])
        self.split = a["split"].astype(np.int64)
        self.group = a["group"].astype(np.int64)
        self.cats = [str(c) for c in a["cells"]]
        self.start_cat = a["cell"].astype(np.int64)  # per position here (the cell of its fight)
        self.tags = [str(t) for t in a["tags"]]
        self.meta = a["meta"].astype(np.float64)
        self.weights = weights if weights is not None else np.ones(self.n, np.float32)
        d = device
        self.t_types = torch.from_numpy(a["types"]).to(d)
        self.t_ids = torch.from_numpy(a["ids"]).to(d)
        self.t_owners = torch.from_numpy(a["owners"]).to(d)
        self.t_feats = torch.from_numpy(a["feats"]).to(d)
        self.t_lens = torch.from_numpy(self.lens).to(d)
        self.t_off = torch.from_numpy(self.offsets).to(d)
        f = {k: torch.from_numpy(np.asarray(a[k], np.float32)).to(d) for k in ("y_teacher", "y_human", "root_value", "p_death", "pol")}
        m = {k: torch.from_numpy(np.asarray(a[k], bool)).to(d) for k in ("has_teacher", "has_human", "has_pol")}
        self.f, self.m = f, m
        self.t_tier = torch.from_numpy(a["tier"].astype(np.int64)).to(d)
        self.t_w = torch.from_numpy(np.asarray(self.weights, np.float32)).to(d)
        self.t_max_hp = torch.from_numpy(np.maximum(1.0, self.meta[:, 3]).astype(np.float32)).to(d)

    def batch(self, idx: np.ndarray) -> tuple[dict, None]:
        i = torch.from_numpy(idx).to(self.device)
        lens = self.t_lens[i]
        width = int(lens.max())
        ar = torch.arange(width, device=self.device)
        valid = ar[None, :] < lens[:, None]
        pos = torch.where(valid, self.t_off[i][:, None] + ar[None, :], torch.zeros_like(ar)[None, :])
        types = torch.where(valid, self.t_types[pos].long(), torch.full_like(pos, TYPE_PAD))
        feats = self.t_feats[pos] * valid[..., None]
        return ({"types": types, "ids": self.t_ids[pos].long() * valid, "owners": self.t_owners[pos].long() * valid, "feats": feats,
                 "mask": ~valid}, None)

    def targets(self, idx: np.ndarray) -> dict[str, torch.Tensor]:
        i = torch.from_numpy(idx).to(self.device)
        rv = self.f["root_value"][i]
        pd = self.f["p_death"][i]
        return {"w": self.t_w[i], "y_teacher": self.f["y_teacher"][i], "m_teacher": self.m["has_teacher"][i], "y_human": self.f["y_human"][i],
                "m_human": self.m["has_human"][i], "root_value": torch.nan_to_num(rv), "m_value": ~torch.isnan(rv), "p_death": torch.nan_to_num(pd),
                "m_pd": ~torch.isnan(pd), "tier": self.t_tier[i], "max_hp": self.t_max_hp[i], "pol": self.f["pol"][i], "m_pol": self.m["has_pol"][i]}


# ---------------------------------------------------------------------------------------------------------- the loss


@dataclass
class LossCfg:
    w_teacher: float = 1.0  # the teacher's outcome of the game (win, end HP, potions)
    w_human: float = 0.5  # the human's outcome of the fight (human rows)
    w_value: float = 1.0  # the judge's value in HP units against the search's root value
    w_search_win: float = 0.5  # the win logit against 1 - the search's death share
    w_policy: float = 0.1  # the policy head against the search's visits (auxiliary)
    hp_w: float = 1.0
    pot_w: float = 0.25
    huber_scale: float = 50.0  # HP units per unit of the value loss
    tier_w: tuple[float, float, float] = (0.25, 0.5, 1.0)  # played, soft, settled: how much a root value counts
    extras: dict = field(default_factory=dict)


def wmean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The weighted mean of x (0 when the weights add up to nothing)."""
    s = w.sum()
    return (x * w).sum() / s.clamp(min=1e-8) if float(s) > 0 else x.sum() * 0.0


def outcome_loss(out: torch.Tensor, y: torch.Tensor, w: torch.Tensor, hp_w: float = 1.0, pot_w: float = 0.25) -> torch.Tensor:
    """`judge.judge_loss` with a weight a row (and 0 weight for a row without the label): win BCE over the rows, the end-HP BCE and the potions
    Huber over the rows that won (weighted by their weights); with unit weights it is `judge.judge_loss`."""
    won = y[:, 0]
    bce = nn.functional.binary_cross_entropy_with_logits
    loss = wmean(bce(out[:, 0], won, reduction="none"), w)
    ww = w * won
    loss = loss + hp_w * wmean(bce(out[:, 1], y[:, 1], reduction="none"), ww)
    return loss + pot_w * wmean(nn.functional.smooth_l1_loss(out[:, 2], y[:, 2], reduction="none"), ww)


def value_of(out: torch.Tensor, max_hp: torch.Tensor) -> torch.Tensor:
    """The judge's value at a leaf in the search's units (`judge_client.estimate`): P(win) x (end HP + 10 x potions) + P(death) x -100."""
    p = torch.sigmoid(out[:, 0])
    hp = torch.sigmoid(out[:, 1]) * max_hp
    return p * (hp + POTION_VALUE * out[:, 2]) + (1.0 - p) * DEATH_VALUE


def policy_loss(pol_logits: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Cross entropy of the policy head's softmax over the action tokens against the search's visit distribution (rows in `w`)."""
    logp = torch.log_softmax(pol_logits.float(), dim=-1)
    t = target.float()
    ce = -(t * logp.masked_fill(t <= 0, 0.0)).sum(-1)
    return wmean(ce, w)


def corpus_loss(out: torch.Tensor, pol: torch.Tensor | None, t: dict[str, torch.Tensor], cfg: LossCfg,
                out_h: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """The batch's loss and its parts (floats): each term averages over the rows that have its label, by the rows' weights. `out` is the
    judge's rows as the teacher's (the outcome of the game, the search's value and death share: what the search reads at a leaf); `out_h`
    the same rows read with the human source, trained to the human's outcome (None: `out`)."""
    w = t["w"]
    parts: dict[str, torch.Tensor] = {}
    if cfg.w_teacher:
        parts["teacher"] = cfg.w_teacher * outcome_loss(out, t["y_teacher"], w * t["m_teacher"], cfg.hp_w, cfg.pot_w)
    if cfg.w_human:
        parts["human"] = cfg.w_human * outcome_loss(out if out_h is None else out_h, t["y_human"], w * t["m_human"], cfg.hp_w, cfg.pot_w)
    tier_w = torch.tensor(cfg.tier_w, device=out.device, dtype=out.dtype)[t["tier"]]
    if cfg.w_value:
        v = value_of(out, t["max_hp"])
        err = nn.functional.smooth_l1_loss(v / cfg.huber_scale, t["root_value"] / cfg.huber_scale, reduction="none")
        parts["value"] = cfg.w_value * wmean(err, w * t["m_value"] * tier_w)
    if cfg.w_search_win:
        soft = (1.0 - t["p_death"]).clamp(0.0, 1.0)
        parts["search_win"] = cfg.w_search_win * wmean(nn.functional.binary_cross_entropy_with_logits(out[:, 0], soft, reduction="none"),
                                                       w * t["m_pd"] * tier_w)
    if cfg.w_policy and pol is not None:
        parts["policy"] = cfg.w_policy * policy_loss(pol, t["pol"], w * t["m_pol"])
    total = sum(parts.values())
    return total, {k: float(v.detach()) for k, v in parts.items()}


# ---------------------------------------------------------------------------------------------------------- calibration and prediction


def predict(model: jd.JudgeNet, data: CorpusData, idx: np.ndarray, batch: int = 2048, human: bool = False) -> np.ndarray:
    """The judge's raw rows [n, 3] on the positions `idx`: as the teacher's (what the search asks) or, with `human`, as the human's outcome
    (the source input; a checkpoint without it reads the same either way)."""
    model.eval()
    outs = []
    with torch.inference_mode():
        for at in range(0, len(idx), batch):
            x, _ = data.batch(idx[at : at + batch])
            src = torch.ones(len(x["types"]), dtype=torch.long, device=data.device) if human else None
            outs.append(model.judge_out(**x, src=src).float().cpu().numpy())
    return np.concatenate(outs).astype(np.float64) if outs else np.zeros((0, jd.JUDGE_OUT))


def win_target(data: CorpusData, idx: np.ndarray) -> np.ndarray:
    """The win label calibration is fitted to: the teacher's outcome of the game (the reading the search asks for)."""
    return data.a["y_teacher"][idx, 0].astype(np.float64)


def fit_platt(model: jd.JudgeNet, data: CorpusData, idx: np.ndarray) -> dict:
    """Platt scaling of the win logit on the validation positions (weighted by the rows' weights): the scale and shift that minimise the
    log loss against `win_target` (they apply to both readings, the teacher's and the human's)."""
    model.win_cal.copy_(torch.tensor([1.0, 0.0]))
    raw = torch.from_numpy(predict(model, data, idx)[:, 0]).double()
    y = torch.from_numpy(win_target(data, idx)).double()
    w = torch.from_numpy(data.weights[idx].astype(np.float64))
    ab = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([ab], lr=0.5, max_iter=200)

    def ll(a_, b_):
        return wmean(nn.functional.binary_cross_entropy_with_logits(raw * a_ + b_, y, reduction="none"), w)

    def closure():
        opt.zero_grad()
        loss = ll(ab[0], ab[1])
        loss.backward()
        return loss

    before = float(ll(torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)))
    opt.step(closure)
    after = float(ll(ab[0].detach(), ab[1].detach()))
    model.win_cal.copy_(ab.detach().float())
    return {"scale": round(float(ab[0]), 4), "shift": round(float(ab[1]), 4), "val_log_loss": [round(before, 5), round(after, 5)]}


# ---------------------------------------------------------------------------------------------------------- evaluation


def value_np(rows: np.ndarray, max_hp: np.ndarray) -> np.ndarray:
    p = 1.0 / (1.0 + np.exp(-rows[:, 0]))
    hp = 1.0 / (1.0 + np.exp(-rows[:, 1])) * max_hp
    return p * (hp + POTION_VALUE * rows[:, 2]) + (1.0 - p) * DEATH_VALUE


def _scores(l0, y, p) -> dict:
    sc = jd.score(l0, y[:, 3], p, y[:, 0])
    win = sc.pop("win")
    return sc | {"win_brier": win["brier"], "win_ece": win["ece"], "win_rate": jd._r(float(y[:, 0].mean())), "mean_p_win": jd._r(float(np.mean(p)))}  # noqa: SLF001


def _value_scores(v, root) -> dict:
    ok = ~np.isnan(root)
    if ok.sum() < 3:
        return {"n": int(ok.sum())}
    v, root = np.asarray(v, np.float64)[ok], np.asarray(root, np.float64)[ok]
    return {"n": int(ok.sum()), "pearson": jd._r(jd.pearson(v, root)), "spearman": jd._r(jd.spearman(v, root)),  # noqa: SLF001
            "mae": jd._r(float(np.mean(np.abs(v - root)))), "bias": jd._r(float(np.mean(v - root)))}


def subsets(data: CorpusData, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Named sets of held-out positions: all, each cell (act x kind), each source tag, and the boss cells by source."""
    out = {"all": np.ones(len(idx), bool)}
    for ci, c in enumerate(data.cats):
        m = data.start_cat[idx] == ci
        if m.any():
            out[c] = m
    tag = data.a["tag"][idx]
    for ti, name in enumerate(data.tags):
        m = tag == ti
        if m.any():
            out[f"src {name}"] = m
            for c in HEADLINE_CELLS:
                mc = m & (data.start_cat[idx] == data.cats.index(c))
                if mc.any():
                    out[f"{c} | {name}"] = mc
    return out


def evaluate_judge(rows: np.ndarray, data: CorpusData, idx: np.ndarray, min_n: int = 50, rows_h: np.ndarray | None = None) -> dict:
    """One judge's rows on the positions `idx` (raw output rows), scored per subset: against the teacher's outcome (L0, win), the human's
    outcome (human rows; read with the human source, `rows_h`), and the search's root value (settled and soft turns; and the turns kept as
    played, apart)."""
    a = data.a
    max_hp = np.maximum(1.0, data.meta[idx, 3])
    p, l0 = jd.rows_to_win_l0(rows, max_hp)
    p_h, l0_h = jd.rows_to_win_l0(rows if rows_h is None else rows_h, max_hp)
    v = value_np(rows, max_hp)
    root = a["root_value"][idx].astype(np.float64)
    tier = a["tier"][idx]
    ht, hh = a["has_teacher"][idx], a["has_human"][idx]
    yt, yh = a["y_teacher"][idx], a["y_human"][idx]
    out: dict[str, dict] = {}
    for name, m in subsets(data, idx).items():
        if m.sum() < min_n:
            continue
        rep: dict = {"positions": int(m.sum()), "starts": int(len(np.unique(data.group[idx][m])))}
        mt = m & ht
        if mt.sum() >= min_n:
            rep["teacher"] = _scores(l0[mt], yt[mt], p[mt])
        mh = m & hh
        if mh.sum() >= min_n:
            rep["human"] = _scores(l0_h[mh], yh[mh], p_h[mh])
        ms = m & (tier >= TIER_CODE["soft"])
        if ms.sum() >= min_n:
            rep["search_settled"] = _value_scores(v[ms], np.where(ms, root, np.nan)[ms])
        mp = m & (tier == TIER_CODE["played"])
        if mp.sum() >= min_n:
            rep["search_played"] = _value_scores(v[mp], np.where(mp, root, np.nan)[mp])
        out[name] = rep
    return out


def evaluate_paths(paths: list[str], data: CorpusData, split: int = 2, models: dict | None = None, log=lambda m: print(m, flush=True), min_n: int = 50) -> dict:
    """Each checkpoint on the same positions of one split: {path: {sha256, version, scores: evaluate_judge}}, and the search's own root value
    beside them (self-play and settled rows: its Pearson and Spearman against the teacher's outcome, the reference the judge is read against)."""
    idx = np.flatnonzero(data.split == split)
    out: dict = {"split": split, "positions": int(len(idx)), "judges": {}}
    for path in paths:
        model, blob = jd.load(path, data.device) if models is None or path not in models else (models[path], {})
        rows = predict(model, data, idx)
        rows_h = predict(model, data, idx, human=True) if data.a["has_human"][idx].any() else rows
        out["judges"][str(path)] = {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None,
                                    "version": blob.get("version"), "scores": evaluate_judge(rows, data, idx, min_n, rows_h)}
        log(f"scored {path}")
    ok = ~np.isnan(data.a["root_value"][idx])
    ref = {}
    for name, m in subsets(data, idx).items():
        mm = m & ok & data.a["has_teacher"][idx]
        if mm.sum() >= min_n:
            ref[name] = jd.score(data.a["root_value"][idx][mm], data.a["y_teacher"][idx][mm, 3])
    out["search_root_value"] = ref
    return out


def md_tables(report: dict, labels: dict[str, str] | None = None, cells: tuple[str, ...] | None = None) -> str:
    """Held-out tables in markdown: per subset, the judges side by side (Pearson / Spearman / calibration error of L0 and win against the
    teacher's outcome, the human's outcome, and V against the search's settled values)."""
    labels = labels or {}
    names = list(report["judges"])
    head = [labels.get(n, Path(n).parent.name) for n in names]
    sets = [s for s in next(iter(report["judges"].values()))["scores"]]
    if cells:
        sets = [s for s in sets if s in cells or s == "all"]

    def f(x):
        return "-" if x is None else f"{x:.3f}"

    lines = []
    for key, title, cols in (("teacher", "against the teacher's outcome of the game (L0)", ("pearson", "spearman", "win_ece", "win_brier")),
                             ("human", "against the human's outcome of the fight (L0; human rows)", ("pearson", "spearman", "win_ece", "win_brier")),
                             ("search_settled", "V against the search's settled and soft root values (HP units)", ("pearson", "spearman", "mae", "bias")),
                             ("search_played", "V against the loop's search as played (HP units)", ("pearson", "spearman", "mae", "bias"))):
        lines += [f"\n**{title}**\n", "| subset | positions | " + " | ".join(f"{h} {c}" for c in cols for h in head) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in cols for _ in head) + " |"]
        for s in sets:
            rep0 = report["judges"][names[0]]["scores"][s]
            if key not in rep0:
                continue
            cells_ = []
            for c in cols:
                for n in names:
                    cells_.append(f(report["judges"][n]["scores"].get(s, {}).get(key, {}).get(c)))
            lines.append(f"| {s} | {rep0[key].get('n', rep0['positions'])} | " + " | ".join(cells_) + " |")
    ref = {s: r for s, r in report.get("search_root_value", {}).items() if s in sets}
    if ref:
        lines += ["\n**the search's own root value (rows that have one) against the teacher's outcome (L0), for reference**\n",
                  "| subset | positions | pearson | spearman |", "| --- | --- | --- | --- |"]
        lines += [f"| {s} | {r['n']} | {f(r['pearson'])} | {f(r['spearman'])} |" for s, r in ref.items()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------- training


def make_model(a: argparse.Namespace, device):
    init_blob = torch.load(a.init, map_location="cpu", weights_only=True) if a.init else None
    start = torch.load(a.init_judge, map_location="cpu", weights_only=True) if a.init_judge else None
    ref = start or init_blob
    dims = {"n_buckets": int(ref["n_buckets"]) if ref else 28, "d_model": int(ref.get("d_model", 128)) if ref else a.d_model,
            "layers": int(ref.get("layers", 2)) if ref else a.layers, "heads": int(ref.get("heads", 4)) if ref else 4}
    model = jd.JudgeNet(VOCAB, dims["n_buckets"], d_model=dims["d_model"], layers=dims["layers"], heads=dims["heads"], dropout=a.dropout)
    if start is not None:
        missing, unexpected = model.load_state_dict(start["model"], strict=False)
        assert set(missing) <= {"win_cal"} and not unexpected, (missing, unexpected)
        model.win_cal.copy_(torch.tensor([1.0, 0.0]))  # the calibration is refitted at the end
    elif init_blob is not None:
        missing, unexpected = model.load_state_dict(init_blob["model"], strict=False)
        assert all(k.startswith("judge.") or k == "win_cal" for k in missing) and not unexpected, (missing, unexpected)
    return model.to(device), dims


def loss_over(model: jd.JudgeNet, data: CorpusData, idx: np.ndarray, cfg: LossCfg, batch: int = 2048) -> float:
    model.eval()
    tot = wt = 0.0
    with torch.inference_mode():
        for at in range(0, len(idx), batch):
            part = idx[at : at + batch]
            x, _ = data.batch(part)
            out, out_h, pol = model.judge_out_all(**x)
            t = data.targets(part)
            loss, _ = corpus_loss(out, pol, t, cfg, out_h)
            w = float(t["w"].sum())
            tot += float(loss.detach()) * w
            wt += w
    return tot / max(wt, 1e-8)


def train(a: argparse.Namespace, data: CorpusData, device, log=lambda m: print(m, flush=True)) -> tuple[jd.JudgeNet, dict]:
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model, dims = make_model(a, device)
    cfg = LossCfg(w_teacher=a.w_teacher, w_human=a.w_human, w_value=a.w_value, w_search_win=a.w_search_win, w_policy=a.w_policy,
                  tier_w=tuple(a.tier_w))
    head = [p for n, p in model.named_parameters() if n.startswith(("judge.", "src_emb."))]
    trunk = [p for n, p in model.named_parameters() if not n.startswith(("judge.", "src_emb."))]
    for p in trunk:
        p.requires_grad_(a.trunk_lr > 0)
    groups = [{"params": head, "lr": a.lr}] + ([{"params": [p for p in trunk if p.requires_grad], "lr": a.lr * a.trunk_lr}] if a.trunk_lr > 0 else [])
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.wd)
    train_idx = np.flatnonzero(data.split == SPLIT_CODE["train"])
    val_idx = np.flatnonzero(data.split == SPLIT_CODE["val"])
    if not len(train_idx) or not len(val_idx):
        raise RuntimeError(f"train {len(train_idx)} and validation {len(val_idx)} positions: both are needed")
    steps = a.epochs * math.ceil(len(train_idx) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(a.seed)
    best, best_state, hist = math.inf, None, []
    t0 = time.perf_counter()
    for epoch in range(a.epochs):
        model.train()
        order = rng.permutation(train_idx)
        tot, wt, parts_sum = 0.0, 0.0, Counter()
        for at in range(0, len(order), a.batch):
            part = order[at : at + a.batch]
            x, _ = data.batch(part)
            out, out_h, pol = model.judge_out_all(**x)
            t = data.targets(part)
            loss, parts = corpus_loss(out, pol, t, cfg, out_h)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            w = float(t["w"].sum())
            tot += float(loss.detach()) * w
            wt += w
            for k, v in parts.items():
                parts_sum[k] += v * w
        vl = loss_over(model, data, val_idx, cfg)
        hist.append({"epoch": epoch + 1, "train_loss": round(tot / max(wt, 1e-8), 5), "val_loss": round(vl, 5),
                     "parts": {k: round(v / max(wt, 1e-8), 4) for k, v in parts_sum.items()}, "s": round(time.perf_counter() - t0, 1)})
        log(json.dumps(hist[-1]))
        if vl < best:
            best = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if a.pause:
            time.sleep(a.pause)  # the card is shared with the lab's trainer: a breath between epochs
    model.load_state_dict(best_state)
    cal = fit_platt(model, data, val_idx)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    info = {"dims": dims, "cfg": asdict(cfg), "best_val_loss": round(best, 5), "log": hist, "win_calibration": cal, "train": int(len(train_idx)),
            "val": int(len(val_idx)), "state": best_state, "seconds": round(time.perf_counter() - t0, 1)}
    return model, info


def counts_for(report: dict, manifest_path: Path) -> dict:
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else None
    return {"rows": report["rows"], "starts": report["starts"], "runs": report["runs"], "by_tag": report["by_tag"], "by_cell": report["by_cell"],
            "by_tier": report["by_tier"], "human_rows": report["human_rows"], "manifest": report["manifest"], "manifest_sha256": sha}


def cmd_train(a: argparse.Namespace) -> int:
    from sts2_gym import procutil  # noqa: PLC0415

    procutil.set_priority_class(procutil.BELOW_NORMAL_PRIORITY_CLASS)  # the CPU side of a GPU job yields to the lab
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    arrays, report = read_corpus(a.store, require_usable=not a.allow_unusable, max_per_cell=a.max_per_cell)
    print(json.dumps({k: report[k] for k in ("rows", "starts", "runs", "by_tag", "by_split", "human_rows", "seconds")}), flush=True)
    w = weights_of(arrays, a.weights, a.wclip, a.human_mult)
    data = CorpusData(arrays, device, w)
    model, info = train(a, data, device)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = {"kind": jd.KIND, "version": a.version, "model": info["state"], **info["dims"], "outputs": ["win_logit", "hp_frac_logit", "potions"],
            "trained": {"corpus": counts_for(report, Path(a.store) / "manifest.json"), "positions": report["rows"], "train": info["train"],
                        "val": info["val"], "heldout": int((data.split == 2).sum()), "init": Path(a.init).parent.name if a.init else None,
                        "init_judge": Path(a.init_judge).parent.name if a.init_judge else None, "epochs": a.epochs, "lr": a.lr, "trunk_lr": a.trunk_lr,
                        "wd": a.wd, "dropout": a.dropout, "batch": a.batch, "weights": a.weights, "wclip": a.wclip, "human_mult": a.human_mult,
                        "loss": info["cfg"], "best_val_loss": info["best_val_loss"], "seed": a.seed, "win_calibration": info["win_calibration"]}}
    torch.save(blob, out)
    paths = [str(out), *a.compare]
    rep = evaluate_paths(paths, data, 2, models={str(out): model}, log=lambda m: print(m, flush=True))
    full = {"version": a.version, "trained": blob["trained"], "log": info["log"], "heldout": rep, "seconds": info["seconds"]}
    out.with_suffix(".report.json").write_text(json.dumps(full, indent=1), encoding="utf-8")
    labels = {str(out): a.version, **{p: Path(p).parent.name for p in a.compare}}
    out.with_suffix(".eval.md").write_text(md_tables(rep, labels), encoding="utf-8")
    sc = rep["judges"][str(out)]["scores"]
    print(json.dumps({"heldout_all": sc.get("all"), "seconds": info["seconds"], "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}), flush=True)
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    arrays, report = read_corpus(a.store, require_usable=not a.allow_unusable, max_per_cell=a.max_per_cell)
    data = CorpusData(arrays, device, weights_of(arrays, "none", None))
    split = {"train": 0, "val": 1, "heldout": 2}[a.split]
    rep = evaluate_paths(a.judge, data, split)
    rep["corpus"] = counts_for(report, Path(a.store) / "manifest.json")
    labels = {p: Path(p).parent.name for p in a.judge}
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    md = md_tables(rep, labels)
    if a.md:
        Path(a.md).write_text(md, encoding="utf-8")
    print(md)
    return 0


def cmd_summary(a: argparse.Namespace) -> int:
    arrays, report = read_corpus(a.store, require_usable=False, max_per_cell=a.max_per_cell)
    print(json.dumps(report, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("train", "eval", "summary"))
    ap.add_argument("--store", default=str(STORE))
    ap.add_argument("--out")
    ap.add_argument("--md")
    ap.add_argument("--init", default=None, help="the agent's combat network the trunk starts from (as v0 and v1)")
    ap.add_argument("--init-judge", default=None, help="a judge checkpoint to continue from instead")
    ap.add_argument("--compare", nargs="*", default=[], help="train: judge checkpoints scored beside the new one on the same held-out positions")
    ap.add_argument("--judge", action="append", default=[], help="eval: each checkpoint to score (repeatable)")
    ap.add_argument("--split", choices=("train", "val", "heldout"), default="heldout")
    ap.add_argument("--version", default=VERSION)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--trunk-lr", type=float, default=0.1, help="the trunk's learning rate as a share of --lr (0 freezes it)")
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pause", type=float, default=0.5)
    ap.add_argument("--weights", choices=jc.WEIGHT_MODES, default="pool")
    ap.add_argument("--wclip", type=float, default=20.0, help="a weight is cut at this many times the mean (0: not cut)")
    ap.add_argument("--human-mult", type=float, default=1.0, help="human rows' weights are multiplied by this before the clip")
    ap.add_argument("--w-teacher", type=float, default=LossCfg.w_teacher)
    ap.add_argument("--w-human", type=float, default=LossCfg.w_human)
    ap.add_argument("--w-value", type=float, default=LossCfg.w_value)
    ap.add_argument("--w-search-win", type=float, default=LossCfg.w_search_win)
    ap.add_argument("--w-policy", type=float, default=LossCfg.w_policy)
    ap.add_argument("--tier-w", type=float, nargs=3, default=list(LossCfg.tier_w), metavar=("PLAYED", "SOFT", "SETTLED"))
    ap.add_argument("--max-per-cell", type=int, default=None, help="smoke test: read only this many rows of each source, kind and act")
    ap.add_argument("--allow-unusable", action="store_true", help="read a corpus whose manifest is not usable_for_training yet (a dry run)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "train" and not a.out:
        ap.error("train needs --out")
    if a.cmd == "eval" and not a.judge:
        ap.error("eval needs at least one --judge")
    return {"train": cmd_train, "eval": cmd_eval, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
