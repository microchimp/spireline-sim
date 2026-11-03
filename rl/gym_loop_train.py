"""The always-on loop's trainer (D40, gym build item L1): train_combat's network and heads, continuously, on the card.

    D:/spireline/venv-gpu/Scripts/python.exe rl/gym_loop_train.py --root ROOT --base AGENT_DIR --parent-pid PID
                                             [--every-min 30] [--max-epochs 4] [--window 300000]

Started by the daemon (`rl/gym_loop.py run`), in its Job Object, at BelowNormal; it ends with the daemon (the parent's PID and its
STOP file are read every chunk). It reads the replay buffer's shards (`ROOT/buffer/shards`), keeps the newest `--window` rows in
memory, and fits `combat_net.CombatNet` from the weights it last saved (at the first start: the incumbent's `combat_policy.pt`):

- the value head on the fight's outcome as `train_combat.train` scores it (soft cross-entropy on the end-HP bucket, death the first;
  a fight stopped by a guard has no value label);
- the policy head on the teacher search's root visits (train_combat's visit target over the legal actions); where the exact solver
  finished the decision with an exact value and the search's move was worse than the solver's (expected solve: `q_played` below
  `value`), on the solver's move instead.

Rows from the held-out run seeds (gym_loop.is_val_seed) are never trained on; each checkpoint is scored on them.

**Freshness guard.** Every row that arrives adds `--max-epochs` rows' worth of training samples to a budget; training spends it, so
on unchanged data the trainer stops after `--max-epochs` epochs of what was added and idles (the idle time is logged), however long
generation takes. **Checkpoints** every `--every-min` minutes of wall time when anything was trained since the last: a folder
`ROOT/checkpoints/cNNNN` the players load as an agent (model.json, combat_policy.pt, combat_net.pt) and its train.json.

**The card and the locks.** It trains on the card only while GPU_OK's first line is absent or one of the loop's (its own line,
`gym-loop-train`, is there only while it trains; it is dropped when it idles or pauses). Otherwise, during the night (the lab's
night running or due), it trains on the CPU on one thread ("the night is training time, at low CPU"); at any other time it waits.
It also waits while LAB/PAUSE or LAB/HALT exists and while another live QUIET.lock line stands (the night's own excepted).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gym_loop as gl  # noqa: E402 -- pure at import

MAX_ACTIONS = gl.MAX_ACTIONS
CHUNK_S = 20.0  # training between two looks at the locks, the buffer and the parent


def _log(root: Path, msg: str) -> None:
    with open(root / "trainer" / "trainer.log", "a", encoding="utf-8") as fh:
        fh.write(f"{gl.stamp()} {gl.clean(msg)}\n")


# ---------------------------------------------------------------------------------------------------------- the buffer


class Buffer:
    """The newest `window` rows of the replay buffer as flat arrays (each row's tokens one after another, with offsets)."""

    KEYS = ("visits", "legal", "label", "value_ok", "val", "exact", "exact_value", "q_played", "solver_best", "played")

    def __init__(self, shards: Path, window: int) -> None:
        self.dir, self.window = Path(shards), int(window)
        self.parts: list[tuple[str, dict]] = []
        self.known: set[str] = set()
        self.rows = 0
        self.flat: dict | None = None

    def refresh(self) -> int:
        """Load the shards that appeared since the last call; drop the oldest beyond the window. Returns the rows added."""
        import numpy as np  # noqa: PLC0415

        names = sorted(p.name for p in self.dir.glob("s*.npz") if ".tmp" not in p.name)
        added = 0
        for name in names:
            if name in self.known:
                continue
            self.known.add(name)
            try:
                with np.load(self.dir / name) as z:
                    a = {k: z[k] for k in z.files if k not in ("fights_meta",)}
            except (OSError, ValueError, EOFError):
                continue
            self.parts.append((name, a))
            n = int(a["lengths"].size)
            self.rows += n
            added += n
        while len(self.parts) > 1 and self.rows - int(self.parts[0][1]["lengths"].size) >= self.window:
            self.rows -= int(self.parts.pop(0)[1]["lengths"].size)
        if added:
            self.flat = None
        return added

    def arrays(self) -> dict:
        import numpy as np  # noqa: PLC0415

        if self.flat is None and self.parts:
            ps = [p for _, p in self.parts]
            flat = {k: np.concatenate([p[k] for p in ps]) for k in ("types", "ids", "owners", "feats", "lengths", *self.KEYS)}
            flat["offsets"] = np.concatenate([[0], np.cumsum(flat["lengths"])[:-1]]).astype(np.int64)
            flat["train_idx"] = np.flatnonzero(flat["val"] == 0)
            flat["val_idx"] = np.flatnonzero(flat["val"] != 0)
            self.flat = flat
        return self.flat or {}


def batch_tensors(flat: dict, rows, device, width: int, n_buckets: int, solver_override: bool = True):
    """Padded inputs (as `combat_net.collate` pads them) and the targets of `rows`, built with array indexing."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from combat_tokens import TYPE_PAD  # noqa: PLC0415

    rows = np.asarray(rows)
    lens = flat["lengths"][rows].astype(np.int64)
    w = int(lens.max())
    ar = np.arange(w)[None, :]
    valid = ar < lens[:, None]
    pos = np.where(valid, flat["offsets"][rows][:, None] + ar, 0)
    types = np.where(valid, flat["types"][pos], TYPE_PAD).astype(np.int64)
    ids = np.where(valid, flat["ids"][pos], 0).astype(np.int64)
    owners = np.where(valid, flat["owners"][pos], 0).astype(np.int64)
    feats = np.where(valid[:, :, None], flat["feats"][pos], 0.0).astype(np.float32)
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(device)  # noqa: E731
    types_t = t(types)
    inputs = {"types": types_t, "ids": t(ids), "owners": t(owners), "feats": t(feats), "mask": types_t == TYPE_PAD}
    label = flat["label"][rows].astype(np.int64)
    bucket = np.where(label <= 0, 0, np.minimum(n_buckets - 1, 1 + (label - 1) // width))
    legal = flat["legal"][rows].astype(np.float32)
    visits = flat["visits"][rows].astype(np.float32) * legal
    if solver_override:
        sb = flat["solver_best"][rows].astype(np.int64)
        worse = (flat["exact"][rows] > 0) & (sb >= 0) & np.isfinite(flat["q_played"][rows]) & \
                (flat["exact_value"][rows] - flat["q_played"][rows] > 1e-6)
        ok = worse & (legal[np.arange(len(rows)), np.clip(sb, 0, MAX_ACTIONS - 1)] > 0)
        if ok.any():
            one = np.zeros_like(visits)
            one[np.arange(len(rows)), np.clip(sb, 0, MAX_ACTIONS - 1)] = 1.0
            visits = np.where(ok[:, None], one, visits)
    total = visits.sum(axis=1, keepdims=True)
    policy = visits / np.maximum(total, 1e-9)
    return inputs, {"bucket": t(bucket), "value_ok": t(flat["value_ok"][rows].astype(np.float32)), "legal": t(legal),
                    "policy": t(policy.astype(np.float32)), "has_policy": t((total[:, 0] > 0).astype(np.float32)),
                    "most": t(np.argmax(visits, axis=1).astype(np.int64))}


def load_net(path: Path):
    """A `combat_policy.pt` as a `CombatNet` on the CPU, with its shape (width, bucket count) for `batch_tensors`."""
    import torch  # noqa: PLC0415
    from combat_net import CombatNet  # noqa: PLC0415
    from combat_tokens import VOCAB  # noqa: PLC0415

    blob = torch.load(path, map_location="cpu", weights_only=True)
    m = CombatNet(VOCAB, blob["n_buckets"], d_model=blob["d_model"], layers=blob["layers"], heads=blob["heads"])
    m.load_state_dict(blob["model"])
    return m, {k: blob[k] for k in ("n_buckets", "width", "max_hp", "d_model", "layers", "heads")}


def evaluate_model(flat: dict, model, shape: dict, device, limit: int = 20000) -> dict:
    """Value NLL, policy cross-entropy and top-1 agreement with the teacher on the newest `limit` held-out rows of `flat`."""
    import torch  # noqa: PLC0415

    idx = flat.get("val_idx")
    if idx is None or not len(idx):
        return {}
    idx = idx[-limit:]
    model.train(False)
    tot = {"value": 0.0, "policy": 0.0, "agree": 0.0, "n": 0}
    with torch.no_grad():
        for s in range(0, len(idx), 512):
            rows = idx[s: s + 512]
            inputs, tg = batch_tensors(flat, rows, device, shape["width"], shape["n_buckets"], solver_override=False)
            vl, pl, ag = losses(model, inputs, tg)
            tot["value"] += float(vl) * len(rows)
            tot["policy"] += float(pl) * len(rows)
            tot["agree"] += float(ag) * len(rows)
            tot["n"] += len(rows)
    n = max(1, tot["n"])
    return {"val_rows": tot["n"], "val_value_nll": tot["value"] / n, "val_policy_ce": tot["policy"] / n,
            "val_top1_agree": tot["agree"] / n}


def score_pair(root: Path, cand_dir: Path, base_dir: Path, window: int = 300_000, limit: int = 20000) -> dict:
    """A checkpoint and an incumbent on the same held-out rows (the newest `limit` of the buffer's newest `window` rows), on the
    CPU on one thread: the gate pre-filter's figures for a checkpoint whose own train.json does not carry them for this
    incumbent and split (a checkpoint from before the figures named whose they were, or an incumbent promoted since)."""
    import torch  # noqa: PLC0415

    torch.set_num_threads(1)
    buf = Buffer(Path(root) / "buffer" / "shards", window)
    buf.refresh()
    flat = buf.arrays()
    out: dict = {}
    for tag, d in (("", Path(cand_dir)), ("base_", Path(base_dir))):
        m, shape = load_net(d / "combat_policy.pt")
        got = evaluate_model(flat, m, shape, "cpu", limit)
        if not got:
            raise RuntimeError("no held-out rows in the buffer")
        out.update({f"{tag}{k}": v for k, v in got.items() if k != "val_rows" or not tag})
        out[f"{tag}sha256"] = gl.sha256_file(d / "combat_policy.pt")
    out["split"] = gl.val_split_id()
    return out


def losses(model, inputs: dict, tg: dict):
    """train_combat.train's two terms: soft cross-entropy on the value (one-hot outcome bucket) and on the policy target."""
    import torch  # noqa: PLC0415

    value_logits, policy_logits = model(**inputs)
    vl = torch.nn.functional.cross_entropy(value_logits, tg["bucket"], reduction="none")
    value_loss = (vl * tg["value_ok"]).sum() / tg["value_ok"].sum().clamp(min=1.0)
    logp = torch.log_softmax(policy_logits.masked_fill(tg["legal"] == 0, float("-inf")), dim=1)
    per = -(tg["policy"] * logp.nan_to_num(0.0)).sum(dim=1)
    policy_loss = (per * tg["has_policy"]).sum() / tg["has_policy"].sum().clamp(min=1.0)
    top = policy_logits.masked_fill(tg["legal"] == 0, float("-inf")).argmax(dim=1)
    agree = ((top == tg["most"]).float() * tg["has_policy"]).sum() / tg["has_policy"].sum().clamp(min=1.0)
    return value_loss, policy_loss, agree


# ---------------------------------------------------------------------------------------------------------- the trainer


def trainer_mode(root: Path) -> str:
    """gpu | cpu | paused:<why> | stop."""
    if (root / "trainer" / "STOP").exists() or (root / "STOP").exists():
        return "stop"
    if gl.HALT.exists() or gl.PAUSE.exists():
        return "paused:pause"
    tag = gl.other_quiet(allow=("lab-nightly",))
    if tag:
        return f"paused:quiet:{tag}"
    if not os.environ.get("GYM_LOOP_NO_GPU") and gl.gpu_allowed():
        return "gpu"
    if gl.night_on():
        return "cpu"
    first = (gl._lines(gl.GPU_OK) or ["?"])[0]  # noqa: SLF001
    return f"paused:gpu_ok:{first.split()[0][:30]}"


class Trainer:
    def __init__(self, a: argparse.Namespace) -> None:
        import torch  # noqa: PLC0415
        from combat_net import CombatNet  # noqa: PLC0415
        from combat_tokens import VOCAB  # noqa: PLC0415

        self.a = a
        self.root = Path(a.root)
        self.dir = self.root / "trainer"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ckdir = self.root / "checkpoints"
        self.ckdir.mkdir(parents=True, exist_ok=True)
        self.torch = torch
        torch.set_num_threads(1)
        torch.manual_seed(a.seed)
        state_path = self.dir / "state.pt"
        blob = torch.load(Path(a.base) / "combat_policy.pt", map_location="cpu", weights_only=True)
        self.shape = {k: blob[k] for k in ("n_buckets", "width", "max_hp", "d_model", "layers", "heads")}
        self.model = CombatNet(VOCAB, blob["n_buckets"], d_model=blob["d_model"], layers=blob["layers"], heads=blob["heads"])
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=a.lr, weight_decay=0.01)
        self.st = {"steps": 0, "samples": 0, "budget": 0.0, "ck": 0, "idle_s": {}, "train_s": {"gpu": 0.0, "cpu": 0.0},
                   "gpu_busy_s": 0.0, "since_ck": 0, "base": str(Path(a.base).name), "started": gl.stamp(), "rows_seen": 0,
                   "known": []}
        if state_path.is_file():
            saved = torch.load(state_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(saved["model"])
            self.opt.load_state_dict(saved["opt"])
            self.st.update(saved["st"])
            _log(self.root, f"resumed at step {self.st['steps']} (checkpoint c{self.st['ck']:04d})")
        else:
            self.model.load_state_dict(blob["model"])
            _log(self.root, f"started from {Path(a.base).name}")
        self.base_meta = json.loads((Path(a.base) / "model.json").read_text(encoding="utf-8"))
        self.buffer = Buffer(self.root / "buffer" / "shards", a.window)
        self.counted = set(self.st.get("known") or [])  # shards whose rows are in the budget already (a restart loads them again)
        self.device = "cpu"
        self.last_ck = time.time()
        self.has_gpu_line = False
        self.rng = __import__("numpy").random.default_rng(a.seed + self.st["steps"])

    # -- devices and the card's line
    def to(self, device: str) -> None:
        if device == self.device:
            return
        self.model.to(device)
        for s in self.opt.state.values():
            for k, v in s.items():
                if hasattr(v, "to"):
                    s[k] = v.to(device)
        self.device = device

    def gpu_line(self, on: bool) -> bool:
        if on and not self.has_gpu_line:
            self.has_gpu_line = gl.take_gpu(gl.TRAIN_TAG)
        elif not on and self.has_gpu_line:
            gl.drop_gpu(gl.TRAIN_TAG)
            self.has_gpu_line = False
        return self.has_gpu_line

    # -- the steps
    def refresh(self) -> int:
        """Load new shards; each shard's training rows enter the freshness budget once, whatever restarts happen."""
        import numpy as np  # noqa: PLC0415

        added = self.buffer.refresh()
        fresh = 0
        for name, part in self.buffer.parts:
            if name not in self.counted:
                self.counted.add(name)
                fresh += int(np.sum(part["val"] == 0))
        if fresh:
            self.st["budget"] += self.a.max_epochs * fresh
            self.st["rows_seen"] += fresh
            self.st["known"] = sorted(self.counted)
        return added

    def train_chunk(self, device: str) -> int:
        torch = self.torch
        flat = self.buffer.arrays()
        idx = flat.get("train_idx")
        if idx is None or len(idx) < self.a.batch:
            return 0
        self.to(device)
        torch.set_num_threads(1)
        self.model.train(True)
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < CHUNK_S and self.st["budget"] >= self.a.batch:
            rows = idx[self.rng.integers(0, len(idx), self.a.batch)]
            inputs, tg = batch_tensors(flat, rows, device, self.shape["width"], self.shape["n_buckets"])
            vl, pl, _ = losses(self.model, inputs, tg)
            loss = vl + self.a.policy_weight * pl
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            n += 1
            self.st["budget"] -= self.a.batch
            self.st["samples"] += self.a.batch
            self.st["steps"] += 1
            self.st["since_ck"] += 1
            v, p = float(vl.detach()), float(pl.detach())
            self.st["ema_value"] = 0.98 * self.st.get("ema_value", v) + 0.02 * v
            self.st["ema_policy"] = 0.98 * self.st.get("ema_policy", p) + 0.02 * p
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.st["train_s"][device if device == "cpu" else "gpu"] += dt
        if device == "cuda":
            self.st["gpu_busy_s"] += dt
        return n

    def evaluate(self, model=None, limit: int = 20000) -> dict:
        return evaluate_model(self.buffer.arrays(), model or self.model, self.shape, self.device, limit)

    def incumbent(self) -> tuple[str, Path]:
        """The daemon's incumbent now (its state.json), not the one this process was started beside: a promotion does not restart
        the trainer, and the gate's pre-filter compares each checkpoint with the incumbent it would meet."""
        st = gl.read_json(self.root / "state.json", {}) or {}
        name = st.get("incumbent")
        d = ((st.get("versions") or {}).get(name) or {}).get("dir") if name else None
        if d and (Path(d) / "combat_policy.pt").is_file():
            return str(name), Path(d)
        return Path(self.a.base).name, Path(self.a.base)

    def base_eval(self) -> dict:
        """The incumbent's network on the same held-out rows, for comparison (once per checkpoint, small). The figures carry whose
        they are (`base_version`, `base_sha256` of its combat_policy.pt) and which split they were scored on."""
        name, d = self.incumbent()
        m, _ = load_net(d / "combat_policy.pt")
        m.to(self.device)
        return {"base_version": name, "base_sha256": gl.sha256_file(d / "combat_policy.pt"),
                **{f"base_{k}": v for k, v in self.evaluate(m).items() if k != "val_rows"}}

    def checkpoint(self) -> str:
        import combat_prior  # noqa: PLC0415

        torch = self.torch
        existing = [int(p.name[1:]) for p in self.ckdir.glob("c[0-9][0-9][0-9][0-9]")]
        n = max([self.st["ck"], *existing]) + 1
        name = f"c{n:04d}"
        tmp = self.ckdir / f".tmp-{name}"
        tmp.mkdir(parents=True, exist_ok=True)
        cpu_state = {k: v.detach().to("cpu") for k, v in self.model.state_dict().items()}
        blob = {"model": cpu_state, **self.shape}
        torch.save(blob, tmp / "combat_policy.pt")
        torch.save(blob, tmp / "combat_net.pt")
        meta = dict(self.base_meta)
        meta["policy_sha256"] = combat_prior._sha256_file(tmp / "combat_policy.pt")  # noqa: SLF001
        meta["gym_loop"] = {"checkpoint": name, "base": self.st["base"], "steps": self.st["steps"]}
        (tmp / "model.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        ev = {**self.evaluate(), **self.base_eval()}
        info = {"checkpoint": name, "t": gl.stamp(), "steps": self.st["steps"], "samples": self.st["samples"],
                "rows_seen": self.st["rows_seen"], "window_rows": self.buffer.rows, "train_s": self.st["train_s"],
                "idle_s": self.st["idle_s"], "device": self.device, "ema_value": self.st.get("ema_value"),
                "ema_policy": self.st.get("ema_policy"), "split": gl.val_split_id(), **ev}
        (tmp / "train.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        os.replace(tmp, self.ckdir / name)
        self.st["ck"] = n
        self.st["since_ck"] = 0
        self.save_state()
        with open(self.dir / "checkpoints.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(info) + "\n")
        _log(self.root, f"checkpoint {name}: step {self.st['steps']}, val top-1 {ev.get('val_top1_agree')}, "
                        f"value nll {ev.get('val_value_nll')}")
        return name

    def save_state(self) -> None:
        torch = self.torch
        tmp = self.dir / "state.pt.tmp"
        torch.save({"model": {k: v.detach().to("cpu") for k, v in self.model.state_dict().items()},
                    "opt": _opt_cpu(self.opt.state_dict()), "st": self.st}, tmp)
        os.replace(tmp, self.dir / "state.pt")

    def status(self, mode: str) -> None:
        gl.write_json(self.dir / "status.json", {"t": gl.stamp(), "pid": os.getpid(), "mode": mode, "device": self.device,
                                                 "rows": self.buffer.rows, "budget": self.st["budget"], "steps": self.st["steps"],
                                                 "samples": self.st["samples"], "checkpoint": self.st["ck"],
                                                 "train_s": self.st["train_s"], "idle_s": self.st["idle_s"]})

    def run(self) -> int:
        from tools.lab import winsys  # noqa: PLC0415

        winsys.set_below_normal()
        last_status = 0.0
        try:
            while True:
                if self.a.parent_pid and not winsys.parent_alive(self.a.parent_pid):
                    _log(self.root, "the daemon is gone: stopping")
                    break
                mode = trainer_mode(self.root)
                if mode == "stop":
                    break
                self.refresh()
                trained = 0
                if mode in ("gpu", "cpu"):
                    dev = "cpu" if mode == "cpu" else ("cuda" if self.torch.cuda.is_available() else None)
                    if dev == "cuda" and not self.gpu_line(True):
                        mode, dev = "paused:gpu_ok", None
                    if dev is not None and self.st["budget"] >= self.a.batch:
                        trained = self.train_chunk(dev)
                if not trained:
                    self.gpu_line(False)
                    why = "fresh" if mode in ("gpu", "cpu") else mode
                    if why == "fresh" and len(self.buffer.arrays().get("train_idx", [])) < self.a.batch:
                        why = "no-data"
                    self.st["idle_s"][why] = self.st["idle_s"].get(why, 0.0) + 15.0
                    time.sleep(15.0)
                if self.st["since_ck"] > 0 and time.time() - self.last_ck >= self.a.every_min * 60:
                    self.checkpoint()
                    self.last_ck = time.time()
                if time.time() - last_status > 30:
                    self.status(mode)
                    last_status = time.time()
        finally:
            self.gpu_line(False)
            self.save_state()
            self.status("stopped")
        return 0


def _opt_cpu(sd: dict) -> dict:
    out = {"state": {}, "param_groups": sd["param_groups"]}
    for k, s in sd["state"].items():
        out["state"][k] = {n: (v.detach().to("cpu") if hasattr(v, "to") else v) for n, v in s.items()}
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["score"]:  # the gate pre-filter's figures for one checkpoint (see `score_pair`); one JSON line on stdout
        sp = argparse.ArgumentParser(prog="gym_loop_train.py score")
        sp.add_argument("--root", required=True)
        sp.add_argument("--candidate", required=True)
        sp.add_argument("--base", required=True)
        sp.add_argument("--window", type=int, default=300_000)
        s = sp.parse_args(argv[1:])
        print(json.dumps(score_pair(Path(s.root), Path(s.candidate), Path(s.base), s.window)))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--base", required=True, help="the incumbent's folder: its combat_policy.pt is the first weights")
    ap.add_argument("--parent-pid", type=int, default=0)
    ap.add_argument("--every-min", type=float, default=30.0)
    ap.add_argument("--max-epochs", type=float, default=4.0)
    ap.add_argument("--window", type=int, default=300_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    return Trainer(a).run()


if __name__ == "__main__":
    raise SystemExit(main())
