"""refit-judgement (docs/rl/ROUND.md; memo 3.4 and 3.5 step 3): the run judgement V refit on the new fighter's exploring runs and the
human chosen-option rows, with a source input (human or self), the human weight picked on held-out self-play runs, the kept fit the
average of its seeds' weights, and a held-out report (clears Act 1 and Act 2 log-loss, the D33 keep record, the label leg).

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


def refit(cfg: dict, *, values: Any, fight_net: Any, value_net_state: dict, hum: dict, out_dir: Path, meta: dict) -> dict:
    """The fit: for each human weight, `seeds` fits from the incumbent; the weight with the lowest mean held-out clears log-loss is kept
    and its seeds' weights averaged. Writes run_value.pt/.npz (plain RunValueNet) and source_input.npz; returns the report."""
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
    best = min(grid, key=lambda g: (bool(np.isnan(g["mean"])), g["mean"]))
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
    return {"npz": str(npz), "value": value, "human_weight": best["human_weight"], "grid": grid, "incumbent_heldout": incumbent_score,
            "kept_heldout": final_score, "self_rows": int(len(self_d["target"])), "self_rows_heldout": int(self_d["val"].sum()),
            "human_rows": int(len(hum["cls"])), "human_decisions": int(hum["n_decisions"]), "device": str(dev),
            "wall_seconds": round(time.time() - t0, 1)}
