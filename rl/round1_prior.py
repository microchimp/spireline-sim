"""refit-prior (docs/rl/ROUND.md): the deck prior pi_h refit on the human decisions, the player's own blended choices and the labels'
clear winners, with a KL anchor to the current prior, and its held-out report (human fifth 4, the player's held-out runs, the label leg).

    python rl/round1.py refit-prior --config round1.json --round-dir DIR

The loss is  w_human * CE(human) + w_agent * CE(agent choice) + w_label * CE(label clear winner)  +  kl_beta * KL(old || new)  with the KL
taken over every decision of every batch (the old prior's softmax over the same options). Every weight is a config key (prior.weights,
prior.kl_beta, prior.lr ...). Torch is imported here only; the human corpus is read through `human_prior.Dataset` (training fifths for the
fit, fifth 4 for early stopping; `assert_no_exam` on both).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import round1_data as rd

SPIRE_CODEX = "Spire Codex public run export for v0.107.1"


class RowSet:
    """Decisions as padded groups of option rows: the judgement's inputs (dense 350, counts 1513) kept as float16."""

    def __init__(self) -> None:
        self.dense: list[np.ndarray] = []
        self.counts: list[np.ndarray] = []
        self.start: list[int] = []
        self.n: list[int] = []
        self.chosen: list[int] = []
        self.meta: list[tuple[str, int]] = []  # (screen, act)
        self._rows = 0

    def add(self, rows: np.ndarray, chosen: int, screen: str, act: int) -> None:
        from run_train import models  # noqa: PLC0415

        d, c = models.value_inputs(np.asarray(rows, dtype=np.float32))
        self.dense.append(d.astype(np.float16))
        self.counts.append(c.astype(np.float16))
        self.start.append(self._rows)
        self.n.append(len(d))
        self.chosen.append(int(chosen))
        self.meta.append((screen, int(act)))
        self._rows += len(d)

    def __len__(self) -> int:
        return len(self.n)

    def freeze(self) -> "RowSet":
        self.D = np.concatenate(self.dense) if self.dense else np.zeros((0, 350), np.float16)
        self.C = np.concatenate(self.counts) if self.counts else np.zeros((0, 1513), np.float16)
        self.S, self.N, self.Ch = (np.asarray(x, dtype=np.int64) for x in (self.start, self.n, self.chosen))
        return self


def sets_from_items(items: list[rd.Item], label_winners: bool = False) -> RowSet:
    """Agent items: the chosen option is the target. Label items (`label_winners`): the clear winner among the played options."""
    rs = RowSet()
    for it in items:
        if label_winners:
            if getattr(it, "winner", None) is None:
                continue
            rs.add(it.rows[it.played], it.played.index(it.winner), it.screen, it.act)
        elif it.chosen >= 0:
            rs.add(it.rows, it.chosen, it.screen, it.act)
    return rs.freeze()


class SetGpu:
    def __init__(self, rs: RowSet, dev: Any) -> None:
        import torch  # noqa: PLC0415

        self.t, self.dev, self.size = torch, dev, len(rs)
        self.D, self.C = torch.from_numpy(rs.D).to(dev), torch.from_numpy(rs.C).to(dev)
        self.S, self.N, self.Ch = (torch.from_numpy(x).to(dev) for x in (rs.S, rs.N, rs.Ch))

    def logits(self, net: Any, idx: Any) -> tuple[Any, Any]:
        t = self.t
        n = self.N[idx]
        k = int(n.max().item())
        j = t.arange(k, device=self.dev)[None, :]
        valid = j < n[:, None]
        rows = (self.S[idx][:, None] + j)[valid]
        z = net(self.D[rows].float(), self.C[rows].float())
        out = t.full(valid.shape, -1e9, device=self.dev)
        out[valid] = z
        return out, valid

    def ce(self, net: Any, idx: Any) -> tuple[Any, Any, Any]:
        z, valid = self.logits(net, idx)
        ch = self.Ch[idx]
        return self.t.logsumexp(z, 1) - z.gather(1, ch[:, None])[:, 0], z, valid


def kl_old_new(t: Any, z_new: Any, z_old: Any, valid: Any) -> Any:
    """Per decision KL(old || new) over its options."""
    lp_new = t.log_softmax(z_new.masked_fill(~valid, -1e9), 1)
    lp_old = t.log_softmax(z_old.masked_fill(~valid, -1e9), 1)
    return t.where(valid, lp_old.exp() * (lp_old - lp_new), t.zeros_like(lp_new)).sum(1)


def load_net(path: Path | str, dev: Any) -> Any:
    import torch  # noqa: PLC0415

    import human_prior as hp  # noqa: PLC0415

    net = hp.make_net()
    with np.load(path, allow_pickle=False) as z:
        net.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files if k != "meta" and not k.startswith("extra.")})
    return net.to(dev)


def set_metrics(net: Any, sg: SetGpu, rs: RowSet, batch: int = 4096) -> dict:
    """NLL and top-1 of the chosen option on a set, in all and by screen."""
    t = sg.t
    nll, hit = [], []
    net.eval()
    with t.no_grad():
        for s in range(0, sg.size, batch):
            idx = t.arange(s, min(s + batch, sg.size), device=sg.dev)
            loss, z, valid = sg.ce(net, idx)
            nll.append(loss.float().cpu().numpy())
            hit.append((z.argmax(1) == sg.Ch[idx]).float().cpu().numpy())
    nll_a, hit_a = (np.concatenate(x) if x else np.zeros(0) for x in (nll, hit))
    out: dict[str, Any] = {"n": int(len(nll_a))}
    if len(nll_a):
        out.update(nll=float(nll_a.mean()), top1=float(hit_a.mean()))
        for scr in ("card", "rest"):
            sel = np.array([m[0] == scr for m in rs.meta])
            if sel.sum() >= 20:
                out[scr] = {"n": int(sel.sum()), "nll": float(nll_a[sel].mean()), "top1": float(hit_a[sel].mean())}
    return out


def fit(cfg: dict, *, human_dir: Path, train_sets: dict[str, RowSet], val_sets: dict[str, RowSet], old_path: Path, out_dir: Path,
        dev_name: str | None = None) -> dict:
    """The refit proper. `train_sets` / `val_sets` map "agent" and "label" to RowSets; the human decisions come from `human_dir`."""
    import torch  # noqa: PLC0415

    import human_prior as hp  # noqa: PLC0415
    import human_starts as hs  # noqa: PLC0415

    P = cfg["prior"]
    t0 = time.time()
    dev = torch.device(dev_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(P["seed"])
    rng = np.random.default_rng(P["seed"])
    kinds = list(P["kinds"])
    ds_tr = hp.Dataset.load(human_dir, P["human_fifths_train"], None, kinds)
    ds_va = hp.Dataset.load(human_dir, [P["human_fifth_val"]], None, kinds)
    exam = hs.read_exam_ids(hp.EXAM_IDS)
    hs.assert_no_exam(ds_tr.run_ids, exam)
    hs.assert_no_exam(ds_va.run_ids, exam)
    gt, gv = hp.Gpu(ds_tr, None, dev, 1.0), hp.Gpu(ds_va, None, dev, 1.0)
    valid_tr = ~hp._bool_unavailable(ds_tr, False, True, True)  # noqa: SLF001
    valid_va = ~hp._bool_unavailable(ds_va, False, True, True)  # noqa: SLF001
    net, old = load_net(old_path, dev), load_net(old_path, dev)
    old.eval()
    for p in old.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    W = P["weights"]
    sg = {k: SetGpu(v, dev) for k, v in train_sets.items() if len(v)}
    vg = {k: SetGpu(v, dev) for k, v in val_sets.items() if len(v)}
    pools = {k: ds_tr.kind_idx[k][valid_tr[ds_tr.kind_idx[k]]] for k in kinds if k in ds_tr.kind_idx}

    def human_val(n: Any) -> dict:
        return hp.metrics_from_scores(ds_va, gv.scores(n), valid_va)

    def val_score(n: Any) -> tuple[float, dict]:
        m = human_val(n)
        s = hp.score_total(m)
        extra = {}
        for k, g in vg.items():
            extra[k] = set_metrics(n, g, val_sets[k])
            if "nll" in extra[k]:
                s += extra[k]["nll"]
        return s, {"human": m, **extra}

    before_score, before = val_score(net)
    best, best_state, since, history = {"score": before_score, "epoch": 0}, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0, []
    batch = P["batch"]
    for epoch in range(P["epochs"]):
        net.train()
        jobs = []
        for k, pool in pools.items():
            perm = rng.permutation(pool)
            jobs += [("human", k, perm[i:i + batch]) for i in range(0, len(perm), batch)]
        for k, g in sg.items():
            perm = rng.permutation(g.size)
            jobs += [(k, k, perm[i:i + batch]) for i in range(0, len(perm), batch)]
        total, kl_sum, nb = 0.0, 0.0, 0
        for ji in rng.permutation(len(jobs)):
            src, kind, idx = jobs[ji]
            didx = torch.from_numpy(np.asarray(idx)).to(dev)
            if src == "human":
                loss = W["human"] * gt.loss(net, kind, didx)
                z, _, valid = gt.logits(net, didx, fixed=False)
                with torch.no_grad():
                    z_old, _, _ = gt.logits(old, didx, fixed=False)
            else:
                ce, z, valid = sg[src].ce(net, didx)
                loss = W[src] * ce.mean()
                with torch.no_grad():
                    z_old, _ = sg[src].logits(old, didx)
            kl = kl_old_new(torch, z, z_old, valid).mean()
            loss = loss + P["kl_beta"] * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total, kl_sum, nb = total + float(loss.item()), kl_sum + float(kl.item()), nb + 1
        net.eval()
        sc, m = val_score(net)
        history.append({"epoch": epoch + 1, "train_loss": total / max(nb, 1), "train_kl": kl_sum / max(nb, 1), "val_score": sc})
        if sc < best["score"] - 1e-4:
            best, best_state, since = {"score": sc, "epoch": epoch + 1}, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            since += 1
            if since >= P["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    after_score, after = val_score(net)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"kinds": kinds, "source": SPIRE_CODEX, "variant": "round1-refit", "refit_of": str(old_path), "config": P}
    wpath = out_dir / "weights.npz"
    hp.export_net(net, wpath, meta)
    return {"weights": str(wpath), "sha256": rd_sha(wpath), "best_epoch": best["epoch"], "epochs_run": len(history), "history": history,
            "val_score_before": before_score, "val_score_after": after_score,
            "heldout": {"before": before, "after": after}, "human_train_decisions": {k: int(len(v)) for k, v in pools.items()},
            "set_sizes": {k: len(v) for k, v in train_sets.items()}, "wall_seconds": round(time.time() - t0, 1), "device": str(dev)}


def rd_sha(path: Path | str) -> str:
    import hashlib  # noqa: PLC0415

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_leg_report(items: list[rd.Item], nets: Any, old_prior: Path, new_prior: Path, lam: float, grid: list[float]) -> dict:
    """Held-out label leg of the new prior against the old one, under the judgement `nets`: the mean centred label value A of the picks
    of the old blend, the new blend (at `lam` and at each grid lambda), the judgement alone and the new prior alone, with paired
    differences from the old blend (standard error, 95% interval)."""
    from human_prior_infer import PriorScorer  # noqa: PLC0415
    from run_train import models  # noqa: PLC0415

    if not items:
        return {"n": 0, "note": "no held-out label decisions"}
    old, new = PriorScorer.load(old_prior), PriorScorer.load(new_prior)
    vals = [list(models.expected_floors(nets.value.floor_probs(it.rows), 53.0)) for it in items]
    lo = [list(old.row_logits(it.rows)) for it in items]
    ln = [list(new.row_logits(it.rows)) for it in items]
    arms: dict[str, Any] = {"old_blend": (vals, lo, lam), "new_blend": (vals, ln, lam), "judgement_only": (vals, ln, 0.0),
                            "new_prior_only": (vals, ln, math.inf)}
    for g in grid:
        arms[f"new_lambda_{g:g}"] = (vals, ln, float(g))
    return rd.leg_picks(items, arms)


def choose_lambda(cfg: dict, leg: dict) -> tuple[float, str]:
    """The blend weight the candidate carries: `prior.lambda` under rule 'fixed'; under 'labels' the grid value with the best held-out
    label value among the new blends (all cards and rests), but only if it beats the fixed lambda's blend by more than one standard
    error; otherwise the fixed one. Returns (lambda, reason)."""
    P = cfg["prior"]
    fixed = float(P["lambda"])
    if P["lambda_rule"] == "fixed" or "all" not in leg:
        return fixed, "prior.lambda_rule is fixed" if P["lambda_rule"] == "fixed" else "no label leg to choose on"
    row = leg["all"]
    best = max(P["lambda_grid"], key=lambda g: (row[f"new_lambda_{g:g}"][0] if row.get(f"new_lambda_{g:g}") else -1e9))
    d = row.get(f"new_lambda_{best:g}_minus_old_blend", {})
    ref = row.get("new_blend_minus_old_blend", {})
    gain = (d.get("mean") or 0.0) - (ref.get("mean") or 0.0)
    return (float(best), f"labels: lambda {best:g} is {gain:+.3f} floors over the fixed one") if gain > (d.get("se") or 1e9) else (
        fixed, "labels: no grid lambda beats the fixed one by a standard error")


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)
