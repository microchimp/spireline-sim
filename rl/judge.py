#!/usr/bin/env python3
"""The fight judge v0 (D37 part 2, D41): a fight position in, the win chance and the end HP out; trained on the card.

    python rl/judge.py train --data DATA/positions.npz --out JUDGE.pt [--init AGENT/combat_net.pt] [--epochs 12]   (GPU venv)
    python rl/judge.py m3 --judge JUDGE.pt --m2 M2_DIR                                                      (emulator venv)
    python rl/judge.py eval --data DATA/positions.npz --judge A.pt [--judge B.pt ...] [--unseen-by MANIFEST]   (GPU venv or CPU)

What it is. `JudgeNet` is the combat network's trunk (`combat_net.CombatNet`: the same tokens, embeddings and encoder the
agent's value head reads) with a judge head on the global token: three numbers, the win chance (a logit), the end HP as a
fraction of max HP if the fight is won (a logit) and the potions still held at the end if won (softplus). It starts from the
agent's own network (`--init`, the value head's trunk) and learns from fights already played out by a3-v1 (`judge_data.py`):
every decision position of a game is labelled with that game's outcome. `estimate` turns the three numbers into the search's
value at a leaf, in its units (`live_search`: a win is end HP plus 10 per potion held, a death -100), and into L0.

The held-out fifth (`judge_data.split_of`, by start id) is never seen in training; a tenth of the training starts (by another
hash) chooses the epoch. `train` reports, on the held-out fifth: Pearson and Spearman of the judge's L0 against the game's L0,
the win chance's Brier score, log loss and calibration error (ten equal-width bins), per category, next to two other
estimates of the same positions: the agent's old value head (M3's "value head") and the search's own root value (the
play-outs a3-v1 ran at that decision). `m3` scores the judge on M3's 200 leaves (the board's Act-1 boss positions at the end of
the first turn, each with ten fresh a3-v1 games as its target), next to the old head and 30 play-outs.

`eval` scores any number of judge checkpoints on the same positions of one split (the held-out fifth by default), in all and by
category, start source and (with `--unseen-by`) on the starts of runs a given manifest's judge never trained on: how v1 is compared
with v0. The npz may come from `judge_data.py buffer` (v1: the loop's buffer, held out by run) or `build` (v0); an npz with
`start_run` takes its validation tenth by run as well.

`LocalJudge` is the judge on the CPU (one torch thread): the tests', the lockstep gym's `--judge-path cpu`, and the remote
judge's fallback. The prior server serves the same checkpoint on the card (`prior_server.py`, judge mode).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from combat_net import CombatNet  # noqa: E402
from combat_tokens import N_FEATURES, TYPE_PAD, VOCAB, bucket_centres  # noqa: E402
from judge_client import JUDGE_OUT, l0_of  # noqa: E402

KIND = "judge"
VERSION = "judge-v0"  # the default stamp of `train`; `--version judge-v1` names a retrain


class JudgeNet(CombatNet):
    """`CombatNet` plus the judge head on the global token (slot 0). The value and policy heads are kept so the agent's
    network loads into it whole; the judge never reads them."""

    def __init__(self, vocab, n_buckets: int, d_model: int = 128, layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__(vocab, n_buckets, d_model=d_model, layers=layers, heads=heads, dropout=dropout)
        self.judge = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, JUDGE_OUT))
        # The win logit's calibration, scale and shift (Platt scaling), fitted on the validation starts after training; (1, 0)
        # while training. A buffer, so the checkpoint, the CPU judge and the server all apply it.
        self.register_buffer("win_cal", torch.tensor([1.0, 0.0]))

    @classmethod
    def from_blob(cls, blob: dict) -> JudgeNet:
        model = cls(VOCAB, int(blob["n_buckets"]), d_model=int(blob["d_model"]), layers=int(blob["layers"]), heads=int(blob["heads"]))
        missing, unexpected = model.load_state_dict(blob["model"], strict=False)
        if unexpected or set(missing) - {"win_cal"}:  # a checkpoint from before the calibration reads as uncalibrated
            raise ValueError(f"judge checkpoint keys: missing {missing}, unexpected {unexpected}")
        return model

    def judge_out(self, types, ids, owners, feats, mask) -> torch.Tensor:
        """[B, 3]: win logit, end-HP-fraction logit (given a win), potions held at the end (given a win, softplus applied)."""
        return self._rows(self._trunk(types, ids, owners, feats, mask)[:, 0])

    def judge_out_policy(self, types, ids, owners, feats, mask) -> tuple[torch.Tensor, torch.Tensor]:
        """`judge_out` and the policy head's [B, MAX_ACTIONS] logits from one pass of the trunk (judge_train's auxiliary move target)."""
        hidden = self._trunk(types, ids, owners, feats, mask)
        return self._rows(hidden[:, 0]), self._policy_logits(hidden, types)

    def _rows(self, summary: torch.Tensor) -> torch.Tensor:
        out = self.judge(summary)
        win = out[:, :1] * self.win_cal[0] + self.win_cal[1]
        return torch.cat([win, out[:, 1:2], nn.functional.softplus(out[:, 2:3])], dim=1)


def load(path: str | Path, device: str = "cpu") -> tuple[JudgeNet, dict]:
    blob = torch.load(str(path), map_location="cpu", weights_only=True)
    if blob.get("kind") != KIND:
        raise ValueError(f"{Path(path).name} is not a judge checkpoint")
    return JudgeNet.from_blob(blob).eval().to(device), blob


def batch_tensors(tokens: list, device="cpu") -> dict:
    """`combat_net.collate` for Tokens (or anything with types/ids/owners/feats arrays)."""
    width = max(len(t.types) for t in tokens)
    n = len(tokens)
    types = np.zeros((n, width), np.int64)
    ids = np.zeros((n, width), np.int64)
    owners = np.zeros((n, width), np.int64)
    feats = np.zeros((n, width, N_FEATURES), np.float32)
    for i, t in enumerate(tokens):
        m = len(t.types)
        types[i, :m], ids[i, :m], owners[i, :m], feats[i, :m] = t.types, t.ids, t.owners, t.feats
    tt = torch.from_numpy(types).to(device)
    return {"types": tt, "ids": torch.from_numpy(ids).to(device), "owners": torch.from_numpy(owners).to(device),
            "feats": torch.from_numpy(feats).to(device), "mask": tt == TYPE_PAD}


class LocalJudge:
    """The judge on the CPU, one torch thread: token sequences in, the raw [n, 3] rows out (`judge_client.estimate` reads them)."""

    def __init__(self, path: str | Path):
        torch.set_num_threads(1)
        self.path = str(path)
        self.model, blob = load(path)
        self.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def rows(self, tokens: list) -> np.ndarray:
        if not tokens:
            return np.zeros((0, JUDGE_OUT), np.float32)
        with torch.inference_mode():
            return self.model.judge_out(**batch_tensors(tokens)).numpy().astype(np.float64)


_LOCAL: dict[str, LocalJudge] = {}


def local(path: str | Path) -> LocalJudge:
    key = str(path)
    if key not in _LOCAL:
        _LOCAL[key] = LocalJudge(key)
    return _LOCAL[key]


# ---------------------------------------------------------------------------------------------------------- metrics


def pearson(x, y) -> float | None:
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rank(x) -> np.ndarray:
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    xs = x[order]
    i = 0
    while i < len(x):  # ties share their mean rank
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y) -> float | None:
    return pearson(_rank(x), _rank(y))


def calibration(p, y, bins: int = 10) -> dict:
    """Ten equal-width bins of the predicted chance: the expected calibration error (|mean p - rate|, weighted by count), the
    largest bin gap, and the table."""
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table, ece, worst = [], 0.0, 0.0
    for b in range(bins):
        m = which == b
        if not m.any():
            continue
        gap = abs(p[m].mean() - y[m].mean())
        ece += m.mean() * gap
        worst = max(worst, gap) if m.sum() >= 30 else worst
        table.append([round(float(edges[b]), 1), int(m.sum()), round(float(p[m].mean()), 3), round(float(y[m].mean()), 3)])
    return {"ece": round(float(ece), 4), "max_gap_n30": round(float(worst), 4), "bins": table}


def score(pred_l0, true_l0, pred_win=None, won=None) -> dict:
    out = {"n": len(true_l0), "pearson": _r(pearson(pred_l0, true_l0)), "spearman": _r(spearman(pred_l0, true_l0)),
           "mse": _r(float(np.mean((np.asarray(pred_l0) - np.asarray(true_l0)) ** 2)))}
    if pred_win is not None and won is not None:
        p = np.clip(np.asarray(pred_win, np.float64), 1e-6, 1 - 1e-6)
        y = np.asarray(won, np.float64)
        out["win"] = {"pearson": _r(pearson(p, y)), "brier": _r(float(np.mean((p - y) ** 2))),
                      "log_loss": _r(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))), **calibration(p, y)}
    return out


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), 4)


# ---------------------------------------------------------------------------------------------------------- data


def val_of(start_id: str) -> bool:
    """A tenth of the training starts, by a hash other than the held-out split's: the epoch is chosen on them."""
    return int(hashlib.sha256(("val|" + start_id).encode()).hexdigest()[:8], 16) % 10 == 0


class Positions:
    """positions.npz on a device: flat tokens with offsets, labels, meta and which split each position is in."""

    def __init__(self, path: str | Path, device):
        z = np.load(str(path))
        self.lens = z["lens"].astype(np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lens)[:-1]])
        self.labels = z["labels"].astype(np.float32)  # won, hp_frac, potions, L0
        self.meta = z["meta"].astype(np.float64)
        self.group = z["group"].astype(np.int64)
        start_ids = [str(s) for s in z["start_ids"]]
        held = z["start_heldout"].astype(bool)
        runs = [str(s) for s in z["start_run"]] if "start_run" in z.files else start_ids  # v1's npz: the validation tenth by run
        val = np.array([val_of(s) for s in runs])
        self.split = np.where(held[self.group], 2, np.where(val[self.group], 1, 0))  # 0 train, 1 val, 2 held out
        self.start_cat = z["start_cat"][self.group]
        self.cats = [str(c) for c in z["cats"]]
        self.start_ids = start_ids
        self.start_runs = runs
        self.start_src = z["start_src"][self.group] if "start_src" in z.files else None
        self.srcs = [str(c) for c in z["srcs"]] if "srcs" in z.files else []
        self.device = device
        self.t_types = torch.from_numpy(z["types"].astype(np.int64)).to(device)
        self.t_ids = torch.from_numpy(z["ids"].astype(np.int64)).to(device)
        self.t_owners = torch.from_numpy(z["owners"].astype(np.int64)).to(device)
        self.t_feats = torch.from_numpy(z["feats"].astype(np.float32)).to(device)
        self.t_lens = torch.from_numpy(self.lens).to(device)
        self.t_off = torch.from_numpy(self.offsets).to(device)
        self.t_labels = torch.from_numpy(self.labels).to(device)

    def batch(self, idx: np.ndarray) -> tuple[dict, torch.Tensor]:
        i = torch.from_numpy(idx).to(self.device)
        lens = self.t_lens[i]
        width = int(lens.max())
        ar = torch.arange(width, device=self.device)
        valid = ar[None, :] < lens[:, None]
        pos = torch.where(valid, self.t_off[i][:, None] + ar[None, :], torch.zeros_like(ar)[None, :])
        types = torch.where(valid, self.t_types[pos], torch.full_like(pos, TYPE_PAD))
        feats = self.t_feats[pos] * valid[..., None]
        return ({"types": types, "ids": self.t_ids[pos] * valid, "owners": self.t_owners[pos] * valid, "feats": feats,
                 "mask": ~valid}, self.t_labels[i])


def judge_loss(out: torch.Tensor, y: torch.Tensor, hp_w: float = 1.0, pot_w: float = 0.25) -> torch.Tensor:
    won = y[:, 0]
    bce = nn.functional.binary_cross_entropy_with_logits
    loss = bce(out[:, 0], won)
    w = won / won.sum().clamp(min=1.0)
    loss = loss + hp_w * (bce(out[:, 1], y[:, 1], reduction="none") * w).sum()
    loss = loss + pot_w * (nn.functional.smooth_l1_loss(out[:, 2], y[:, 2], reduction="none") * w).sum()
    return loss


def fit_calibration(model: JudgeNet, data: Positions, idx: np.ndarray) -> dict:
    """Platt scaling of the win logit on the validation starts' positions: the scale and shift that minimise its log loss there."""
    model.win_cal.copy_(torch.tensor([1.0, 0.0]))
    raw = torch.from_numpy(predict_rows(model, data, idx)[:, 0]).double()
    y = torch.from_numpy(data.labels[idx, 0]).double()
    ab = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([ab], lr=0.5, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(raw * ab[0] + ab[1], y)
        loss.backward()
        return loss

    before = float(nn.functional.binary_cross_entropy_with_logits(raw, y))
    opt.step(closure)
    after = float(nn.functional.binary_cross_entropy_with_logits(raw * ab[0].detach() + ab[1].detach(), y))
    model.win_cal.copy_(ab.detach().float())
    return {"scale": round(float(ab[0]), 4), "shift": round(float(ab[1]), 4), "val_log_loss": [round(before, 5), round(after, 5)]}


def predict_rows(model: JudgeNet, data: Positions, idx: np.ndarray, batch: int = 1024) -> np.ndarray:
    model.eval()
    outs = []
    with torch.inference_mode():
        for at in range(0, len(idx), batch):
            x, _ = data.batch(idx[at : at + batch])
            outs.append(model.judge_out(**x).float().cpu().numpy())
    return np.concatenate(outs).astype(np.float64) if outs else np.zeros((0, JUDGE_OUT))


def _old_head(init: str, data: Positions, idx: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """The agent's value head on the same positions (M3's "value head"): P(win) and its L0, as fight_gym._value_head reads it."""
    blob = torch.load(init, map_location="cpu", weights_only=True)
    model = CombatNet(VOCAB, blob["n_buckets"], d_model=blob.get("d_model", 128), layers=blob.get("layers", 4), heads=blob.get("heads", 4))
    model.load_state_dict(blob["model"])
    model = model.eval().to(device)
    c = bucket_centres(int(blob["n_buckets"]), int(blob["width"]))
    wins, l0s = [], []
    with torch.inference_mode():
        for at in range(0, len(idx), 1024):
            part = idx[at : at + 1024]
            x, _ = data.batch(part)
            v, _ = model(**x)
            p = torch.softmax(v.float(), dim=1).cpu().numpy()
            max_hp = np.maximum(1.0, data.meta[part, 3])
            wins.append(1.0 - p[:, 0])
            l0s.append((p[:, 1:] * (0.5 + 0.5 * np.minimum(1.0, c[None, 1:] / max_hp[:, None]))).sum(1))
    return np.concatenate(wins), np.concatenate(l0s)


def _cat_scores(l0, y, p) -> dict:
    sc_ = score(l0, y[:, 3], p, y[:, 0])
    return {k: v for k, v in sc_.items() if k != "win"} | {"win_ece": sc_["win"]["ece"], "win_brier": sc_["win"]["brier"],
                                                            "win_rate": _r(float(y[:, 0].mean())), "mean_p_win": _r(float(np.mean(p)))}


def evaluate(model, data: Positions, split: int, init: str | None = None, starts: np.ndarray | None = None) -> dict:
    """The model on the positions of one split (0 train, 1 validation, 2 held out), optionally only those of the starts `starts`
    marks (a bool per start). `by_cat` and `by_src` need 50 positions a cell."""
    pos = data.split == split
    if starts is not None:
        pos &= np.asarray(starts, bool)[data.group]
    idx = np.flatnonzero(pos)
    rows = predict_rows(model, data, idx)
    max_hp = np.maximum(1.0, data.meta[idx, 3])
    p, l0 = rows_to_win_l0(rows, max_hp)
    y = data.labels[idx]
    rep = {"positions": int(len(idx)), "starts": int(len(np.unique(data.group[idx]))), "judge": score(l0, y[:, 3], p, y[:, 0])}
    won = y[:, 0] > 0.5
    if won.any():
        rep["judge"]["hp_frac_mae_if_won"] = _r(float(np.mean(np.abs(1 / (1 + np.exp(-rows[won, 1])) - y[won, 1]))))
    rv = data.meta[idx, 4]
    ok = ~np.isnan(rv)
    if ok.sum() >= 3:
        rep["search_root_value"] = score(rv[ok], y[ok, 3])
    pdeath = data.meta[idx, 6]
    okp = ~np.isnan(pdeath)
    if okp.sum() >= 3:
        rep["search_best_p_win"] = score(1 - pdeath[okp], y[okp, 3], 1 - pdeath[okp], y[okp, 0])
    if init:
        vw, vl = _old_head(init, data, idx, data.device)
        rep["old_value_head"] = score(vl, y[:, 3], vw, y[:, 0])
    rep["by_cat"] = {}
    for ci, cat in enumerate(data.cats):
        m = data.start_cat[idx] == ci
        if m.sum() >= 50:
            rep["by_cat"][cat] = {"positions": int(m.sum()), "starts": int(len(np.unique(data.group[idx][m]))), "judge": _cat_scores(l0[m], y[m], p[m])}
    if data.start_src is not None:
        rep["by_src"] = {}
        for si, name in enumerate(data.srcs):
            m = data.start_src[idx] == si
            if m.sum() >= 50:
                rep["by_src"][name] = {"positions": int(m.sum()), "starts": int(len(np.unique(data.group[idx][m]))), "judge": _cat_scores(l0[m], y[m], p[m])}
    return rep


def rows_to_win_l0(rows: np.ndarray, max_hp: np.ndarray, potion_value: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    p = 1 / (1 + np.exp(-rows[:, 0]))
    hp = 1 / (1 + np.exp(-rows[:, 1]))
    return p, l0_of(p, hp, rows[:, 2], max_hp, potion_value)


def cmd_train(a) -> int:
    from sts2_gym import procutil  # noqa: PLC0415

    procutil.set_priority_class(procutil.BELOW_NORMAL_PRIORITY_CLASS)  # the CPU side of a GPU job yields to the lab
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    t0 = time.perf_counter()
    data = Positions(a.data, device)
    init_blob = torch.load(a.init, map_location="cpu", weights_only=True) if a.init else None
    dims = {"n_buckets": int(init_blob["n_buckets"]) if init_blob else 28, "d_model": int(init_blob.get("d_model", 128)) if init_blob else a.d_model,
            "layers": int(init_blob.get("layers", 2)) if init_blob else a.layers, "heads": int(init_blob.get("heads", 4)) if init_blob else 4}
    model = JudgeNet(VOCAB, dims["n_buckets"], d_model=dims["d_model"], layers=dims["layers"], heads=dims["heads"], dropout=a.dropout)
    if init_blob:
        missing, unexpected = model.load_state_dict(init_blob["model"], strict=False)
        assert all(k.startswith("judge.") or k == "win_cal" for k in missing) and not unexpected, (missing, unexpected)
    model.to(device)
    # The judge head learns at lr; the trunk (the agent's, when warm-started) at lr x trunk_lr, or not at all (0): a few hundred
    # starts' positions overfit a whole trunk within an epoch, while the agent's trunk already reads positions well.
    head = [p for n, p in model.named_parameters() if n.startswith("judge.")]
    trunk = [p for n, p in model.named_parameters() if not n.startswith(("judge.", "head.", "policy."))]
    for p in trunk:
        p.requires_grad_(a.trunk_lr > 0)
    groups = [{"params": head, "lr": a.lr}] + ([{"params": trunk, "lr": a.lr * a.trunk_lr}] if a.trunk_lr > 0 else [])
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.wd)
    train_idx = np.flatnonzero(data.split == 0)
    val_idx = np.flatnonzero(data.split == 1)
    steps = a.epochs * math.ceil(len(train_idx) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(a.seed)
    best, best_state, log = math.inf, None, []
    for epoch in range(a.epochs):
        model.train()
        order = rng.permutation(train_idx)
        tot = 0.0
        for at in range(0, len(order), a.batch):
            x, y = data.batch(order[at : at + a.batch])
            loss = judge_loss(model.judge_out(**x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
        model.eval()
        with torch.inference_mode():
            vl = 0.0
            for at in range(0, len(val_idx), 2048):
                x, y = data.batch(val_idx[at : at + 2048])
                vl += float(judge_loss(model.judge_out(**x), y)) * len(y)
        vl /= max(1, len(val_idx))
        log.append({"epoch": epoch + 1, "train_loss": round(tot / len(train_idx), 5), "val_loss": round(vl, 5),
                    "s": round(time.perf_counter() - t0, 1)})
        print(json.dumps(log[-1]), flush=True)
        if vl < best:
            best = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            time.sleep(a.pause)  # the card is shared with the lab's trainer: a breath between epochs
    model.load_state_dict(best_state)
    cal = fit_calibration(model, data, val_idx)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    held = evaluate(model, data, 2, a.init)
    blob = {"kind": KIND, "version": a.version, "model": best_state, **dims, "outputs": ["win_logit", "hp_frac_logit", "potions"],
            "trained": {"data_sha256": hashlib.sha256(Path(a.data).read_bytes()).hexdigest(), "positions": int(len(data.lens)),
                        "train": int(len(train_idx)), "val": int(len(val_idx)), "heldout": held["positions"],
                        "init": Path(a.init).parent.name if a.init else None, "epochs": a.epochs, "lr": a.lr, "trunk_lr": a.trunk_lr, "wd": a.wd,
                        "dropout": a.dropout, "batch": a.batch,
                        "best_val_loss": round(best, 5), "seed": a.seed, "win_calibration": cal}}
    if a.manifest:  # the data's window and counts (judge_data.py buffer's manifest), kept with the weights
        man = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
        blob["trained"]["buffer"] = {k: man.get(k) for k in ("window", "segments", "shards", "fights", "starts", "runs", "heldout_runs",
                                                            "positions", "by_cat", "by_src", "skipped_fights", "excluded", "sha256")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, a.out)
    report = {"version": a.version, "trained": blob["trained"], "log": log, "heldout": held}
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"heldout_judge": held["judge"] | {"win": {k: v for k, v in held["judge"]["win"].items() if k != "bins"}},
                      "old_value_head": {k: v for k, v in held.get("old_value_head", {}).items() if k != "win"},
                      "search_root_value": held.get("search_root_value"), "seconds": round(time.perf_counter() - t0, 1)}))
    return 0


# ---------------------------------------------------------------------------------------------------------- M3's leaves


def m3_scores(judge_path: str, m2: Path) -> dict:
    """The judge on M3's 200 leaves against their ten fresh a3-v1 games, beside the old value head and 30 play-outs (as M3)."""
    import combat_mcts  # noqa: PLC0415
    import fight_gym as fg  # noqa: PLC0415
    from prior_client import encode_position  # noqa: PLC0415

    from sts2_gym import native  # noqa: PLC0415

    judge = local(judge_path)
    leaves = fg.read_jsonl(m2 / "m3_leaves.jsonl")
    m3 = fg.read_jsonl(m2 / "m3.jsonl")
    preds = {r["start"]: r for r in m3 if r.get("type") == "pred" and not r.get("capped")}
    conts: dict[str, list] = {}
    for r in m3:
        if r.get("type") == "cont" and not r.get("capped"):
            conts.setdefault(r["start"], []).append(r)
    ids, rows, max_hps = [], [], []
    for leaf in leaves:
        s = leaf["start"]
        if s not in preds or not conts.get(s):
            continue
        root = combat_mcts.Combat(native.load_combat_snapshot(leaf["doc"], fg.game_seed(s, 0, "m3pred")))
        try:
            view = root.fork(None)
            try:
                rows.append(judge.rows([encode_position(view.handle, False)])[0])
                max_hps.append(view.max_hp())
            finally:
                view.close()
        finally:
            root.close()
        ids.append(s)
    rows = np.array(rows)
    p, l0 = rows_to_win_l0(rows, np.array(max_hps, np.float64))
    tl = [float(np.mean([c["L0"] for c in conts[s]])) for s in ids]
    tw = [float(np.mean([c["won"] for c in conts[s]])) for s in ids]
    out = {"leaves": len(ids), "target_L0": _r(float(np.mean(tl))), "target_win": _r(float(np.mean(tw)))}
    for name, l0x, wx in (("judge", l0, p), ("old_value_head", [preds[s]["vh_L0"] for s in ids], [preds[s]["vh_win"] for s in ids]),
                          ("playouts_30", [preds[s]["po_L0"] for s in ids], [preds[s]["po_win"] for s in ids] if "po_win" in preds[ids[0]] else None)):
        sc = score(l0x, tl)
        if wx is not None:
            sc["win_vs_rate"] = {"pearson": _r(pearson(wx, tw)), "spearman": _r(spearman(wx, tw)),
                                 "mse": _r(float(np.mean((np.asarray(wx) - np.asarray(tw)) ** 2)))}
        out[name] = sc | {"mean_L0": _r(float(np.mean(l0x)))}
    return out


def cmd_m3(a) -> int:
    import fight_gym as fg  # noqa: PLC0415
    import gen0  # noqa: PLC0415

    gen0.select_engine(fg._meta(Path(a.m2))["engine"])  # noqa: SLF001
    rep = m3_scores(a.judge, Path(a.m2))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep))
    return 0


def cmd_eval(a) -> int:
    """Score each --judge on the same positions of --split (held out by default); `--unseen-by MANIFEST` adds the starts of runs
    that manifest's training did not contain (its `train` and `heldout` ids are "SEED|fNN")."""
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    data = Positions(a.data, device)
    split = {"train": 0, "val": 1, "heldout": 2}[a.split]
    unseen = None
    if a.unseen_by:
        man = json.loads(Path(a.unseen_by).read_text(encoding="utf-8"))
        seen = {str(i).split("|")[0] for i in man["train"] + man["heldout"]}
        unseen = np.array([r not in seen for r in data.start_runs], dtype=bool)
    out = {"data": str(Path(a.data).name), "split": a.split, "judges": {}}
    for path in a.judge:
        model, blob = load(path, device)
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        rep = {"sha256": sha, "version": blob.get("version"), "all": evaluate(model, data, split, a.init)}
        if unseen is not None:
            rep["unseen_by"] = {"starts": int(unseen.sum()), **evaluate(model, data, split, a.init, unseen)}
        out["judges"][str(path)] = rep
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({p: {"all": {k: v for k, v in r["all"]["judge"].items() if k != "win"} | {"win_ece": r["all"]["judge"]["win"]["ece"]}}
                      for p, r in out["judges"].items()}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("train", "m3", "eval"))
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--init", default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--trunk-lr", type=float, default=0.1, help="the trunk's learning rate as a share of --lr (0 freezes it)")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pause", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--judge", action="append", default=[], help="m3: the checkpoint; eval: each checkpoint to score (repeatable)")
    ap.add_argument("--m2")
    ap.add_argument("--version", default=VERSION, help="train: the stamp kept in the checkpoint")
    ap.add_argument("--manifest", default=None, help="train: judge_data.py buffer's manifest.json, kept in the checkpoint")
    ap.add_argument("--split", choices=("train", "val", "heldout"), default="heldout")
    ap.add_argument("--unseen-by", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "m3":
        a.judge = a.judge[0] if a.judge else None
    return {"train": cmd_train, "m3": cmd_m3, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
