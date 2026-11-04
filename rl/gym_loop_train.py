"""The always-on loop's trainer (D40, gym build item L1): train_combat's network and heads, continuously, on the card.

    D:/spireline/venv-gpu/Scripts/python.exe rl/gym_loop_train.py --root ROOT --base AGENT_DIR --parent-pid PID
                                             [--every-min 30] [--max-epochs 4] [--window 300000] [--reset-every 0]
    D:/spireline/venv-gpu/Scripts/python.exe rl/gym_loop_train.py replay --root ROOT --base AGENT_DIR --out DIR [--arm SPEC ...]

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
generation takes (`--max-epochs` is the pass cap, any number; 4 today, and the replay below tests 8 and 16). **Checkpoints** every
`--every-min` minutes of wall time when anything was trained since the last: a folder `ROOT/checkpoints/cNNNN` the players load as
an agent (model.json, combat_policy.pt, combat_net.pt) and its train.json. **Resets** (`--reset-every N`, off at 0): after every
N-th checkpoint the last linear layer of the value head and of the policy head is re-initialised and their Adam moments dropped,
the encoder kept (primacy-bias resets, Nikishin 2022).

**The card and the locks.** It trains on the card only while GPU_OK's first line is absent or one of the loop's (its own line,
`gym-loop-train`, is there only while it trains; it is dropped when it idles or pauses). Otherwise, during the night (the lab's
night running or due), it trains on the CPU on one thread ("the night is training time, at low CPU"); at any other time it waits.
It also waits while LAB/PAUSE or LAB/HALT exists and while another live QUIET.lock line stands (the night's own excepted).

**`replay`** (brief 1b of the training design, section 1.4): trains arms offline on the stored shards in arrival order from one base,
one shard per checkpoint interval, with the live trainer's own step and budget (`--max-epochs` passes over each shard's training
rows when it arrives, rows drawn uniformly from everything that has arrived so far), and scores every arm after every shard on the
same held-out rows (the newest `--limit` of the window's, as the gate pre-filter scores them). Nothing under ROOT is written; it
uses the card only while GPU_OK is absent or one of the loop's, and yields while the live trainer holds it; one thread.
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


def train_step(model, opt, flat: dict, idx, rng, device, shape: dict, batch: int, policy_weight: float) -> tuple[float, float]:
    """One optimiser step on `batch` rows drawn uniformly from `idx` (the trainer's step, and the replay's): returns the value and
    policy losses of the batch."""
    import torch  # noqa: PLC0415

    rows = idx[rng.integers(0, len(idx), batch)]
    inputs, tg = batch_tensors(flat, rows, device, shape["width"], shape["n_buckets"])
    vl, pl, _ = losses(model, inputs, tg)
    loss = vl + policy_weight * pl
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(vl.detach()), float(pl.detach())


def head_outputs(model) -> list[tuple[str, object]]:
    """The last linear layer of the value head and of the policy head: what a reset re-initialises (the encoder is never one)."""
    from torch import nn  # noqa: PLC0415

    out = []
    for name in ("head", "policy"):
        linears = [m for m in getattr(model, name).modules() if isinstance(m, nn.Linear)]
        out.append((name, linears[-1]))
    return out


def reset_heads(model, opt=None, seed: int = 0) -> list[str]:
    """Re-initialise the heads' last linear layers as `nn.Linear` does at construction (uniform within +-1/sqrt(fan_in), drawn from
    a generator of its own so the training RNG is untouched) and drop their optimizer moments; everything else is left as it is.
    Returns the names of the parameters reset."""
    import torch  # noqa: PLC0415

    g = torch.Generator().manual_seed(int(seed))
    names = []
    with torch.no_grad():
        for name, layer in head_outputs(model):
            bound = 1.0 / math.sqrt(layer.in_features)
            for pname, p in (("weight", layer.weight), ("bias", layer.bias)):
                p.copy_(torch.empty(p.shape).uniform_(-bound, bound, generator=g).to(p.device))
                if opt is not None:
                    opt.state.pop(p, None)  # Adam starts again on this tensor (moments and step count)
                names.append(f"{name}.{pname}")
    return names


def prepare_eval(flat: dict, shape: dict, device, limit: int = 20000) -> list:
    """The batches `evaluate_model` scores, built once for many models: the newest `limit` held-out rows in 512s with their targets."""
    idx = flat.get("val_idx")
    if idx is None or not len(idx):
        return []
    idx = idx[-limit:]
    out = []
    for s in range(0, len(idx), 512):
        rows = idx[s: s + 512]
        inputs, tg = batch_tensors(flat, rows, device, shape["width"], shape["n_buckets"], solver_override=False)
        out.append((len(rows), inputs, tg))
    return out


def eval_prepared(model, prepared: list) -> dict:
    """`evaluate_model`'s figures on batches from `prepare_eval` (the same sums in the same order)."""
    import torch  # noqa: PLC0415

    model.train(False)
    tot = {"value": 0.0, "policy": 0.0, "agree": 0.0, "n": 0}
    with torch.no_grad():
        for n, inputs, tg in prepared:
            vl, pl, ag = losses(model, inputs, tg)
            tot["value"] += float(vl) * n
            tot["policy"] += float(pl) * n
            tot["agree"] += float(ag) * n
            tot["n"] += n
    n = max(1, tot["n"])
    return {"val_rows": tot["n"], "val_value_nll": tot["value"] / n, "val_policy_ce": tot["policy"] / n,
            "val_top1_agree": tot["agree"] / n}


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
            v, p = train_step(self.model, self.opt, flat, idx, self.rng, device, self.shape, self.a.batch, self.a.policy_weight)
            n += 1
            self.st["budget"] -= self.a.batch
            self.st["samples"] += self.a.batch
            self.st["steps"] += 1
            self.st["since_ck"] += 1
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
        every = int(getattr(self.a, "reset_every", 0) or 0)
        if every > 0 and n % every == 0:  # the checkpoint above keeps the heads it was scored with; training goes on from fresh ones
            self.st["resets"] = int(self.st.get("resets", 0)) + 1
            reset_heads(self.model, self.opt, seed=self.a.seed * 100_003 + self.st["resets"])
            _log(self.root, f"reset: the heads' last layers after {name} (reset {self.st['resets']})")
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


# ---------------------------------------------------------------------------------------------------------- the offline replay

# Section 1.4's six arms: the pass cap 4 (two seeds, for the run-to-run spread), 8 and 16, and 8 and 16 with the heads' last layers
# reset every 24 checkpoints. NAME:PASSES[:RESET_EVERY[:SEED]].
DEFAULT_ARMS = ("cap4a:4:0:0", "cap4b:4:0:1", "cap8:8", "cap16:16", "cap8r24:8:24", "cap16r24:16:24")


def parse_arm(spec: str) -> dict:
    parts = str(spec).split(":")
    if not 2 <= len(parts) <= 4 or not parts[0]:
        raise ValueError(f"arm {spec!r}: expected NAME:PASSES[:RESET_EVERY[:SEED]]")
    return {"name": parts[0], "epochs": float(parts[1]), "reset_every": int(parts[2]) if len(parts) > 2 and parts[2] else 0,
            "seed": int(parts[3]) if len(parts) > 3 and parts[3] else 0}


def card_free() -> bool:
    """The replay may use the card now: GPU_OK is absent or leads with one of the loop's lines, and the live trainer is not on it."""
    lines = gl._lines(gl.GPU_OK)  # noqa: SLF001
    return gl.gpu_allowed(lines) and gl.TRAIN_TAG not in lines


def replay_arm(flat: dict, shards: list[dict], prepared: list, base_dir: Path, arm: dict, device: str, *, batch: int = 256,
               lr: float = 1e-4, policy_weight: float = 1.0, free=None, say=None) -> list[dict]:
    """One arm: the base weights, then the shards in arrival order. Each shard is one checkpoint interval: its training rows add
    `arm["epochs"]` passes to the budget, the budget is spent in steps of `batch` rows drawn uniformly from the training rows that
    have arrived, and the model is scored on the held-out batches. Every `reset_every` checkpoints the heads' last layers are reset
    after the score. Returns one dict per checkpoint (checkpoint 0 is the base). `free()` is asked every 16 steps; training waits
    while it says no."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    torch.manual_seed(arm["seed"])
    model, shape = load_net(Path(base_dir) / "combat_policy.pt")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(arm["seed"])
    train_idx = flat["train_idx"]
    out = [{"arm": arm["name"], "ck": 0, "shard": None, "rows": 0, "train_rows": 0, "steps": 0, "samples": 0, "resets": 0,
            "reset_after": False, "card_s": 0.0, **eval_prepared(model, prepared)}]
    budget, steps, resets, card_s, ck, since, seen_train, rows_seen = 0.0, 0, 0, 0.0, 0, 0, 0, 0
    ema_v = ema_p = None
    for sh in shards:
        seen_train += sh["train"]
        rows_seen += sh["rows"]
        budget += arm["epochs"] * sh["train"]
        idx = train_idx[:seen_train]
        model.train(True)
        t0 = time.perf_counter()
        n = 0
        while budget >= batch and len(idx) >= batch:
            if free is not None and n % 16 == 0:
                while not free():
                    time.sleep(15.0)
            v, p = train_step(model, opt, flat, idx, rng, device, shape, batch, policy_weight)
            budget -= batch
            steps += 1
            since += 1
            n += 1
            ema_v = v if ema_v is None else 0.98 * ema_v + 0.02 * v
            ema_p = p if ema_p is None else 0.98 * ema_p + 0.02 * p
        if device == "cuda":
            torch.cuda.synchronize()
        card_s += time.perf_counter() - t0
        if not since:  # nothing trained since the last checkpoint: the live trainer writes none
            continue
        since = 0
        ck += 1
        reset = bool(arm["reset_every"] > 0 and ck % arm["reset_every"] == 0)
        row = {"arm": arm["name"], "ck": ck, "shard": sh["name"], "t": sh.get("t"), "rows": rows_seen, "train_rows": seen_train,
               "steps": steps, "samples": steps * batch, "resets": resets, "reset_after": reset, "card_s": round(card_s, 2),
               "ema_value": ema_v, "ema_policy": ema_p, **eval_prepared(model, prepared)}
        out.append(row)
        if say:
            say(f"{arm['name']} ck {ck} rows {rows_seen} steps {steps} policy CE {row['val_policy_ce']:.4f} "
                f"value NLL {row['val_value_nll']:.4f} top-1 {row['val_top1_agree']:.4f}{' (reset after)' if reset else ''}")
        if reset:
            resets += 1
            reset_heads(model, opt, seed=arm["seed"] * 100_003 + resets)
    return out


def summarize(arms: dict[str, list[dict]], spec: dict[str, dict]) -> dict:
    """Section 1.4's table from the arms' checkpoint rows: held-out policy CE, value NLL and top-1 agreement at the end and at
    matched arrival points (a quarter, a half, three quarters of the checkpoints), the mean of the last 8 checkpoints' CE, and the
    adoption rule: the arm with the lowest CE at the end is adopted if it beats cap 4 (the mean of the cap-4 arms) by more than the
    cap-4 arms differ; a reset arm only if it also beats the same cap without resets by more than that."""
    def at(rows, frac):
        trained = [r for r in rows if r["ck"] > 0]
        return trained[max(0, min(len(trained) - 1, int(round(frac * len(trained))) - 1))]

    table = {}
    for name, rows in arms.items():
        trained = [r for r in rows if r["ck"] > 0]
        if not trained:
            continue
        last = trained[-1]
        table[name] = {
            "passes": spec[name]["epochs"], "reset_every": spec[name]["reset_every"], "seed": spec[name]["seed"],
            "checkpoints": last["ck"], "steps": last["steps"], "card_s": last["card_s"],
            "base_policy_ce": rows[0]["val_policy_ce"],
            "end": {k: last[k] for k in ("val_policy_ce", "val_value_nll", "val_top1_agree")},
            "last8_policy_ce": sum(r["val_policy_ce"] for r in trained[-8:]) / len(trained[-8:]),
            "at": {f"{int(f * 100)}%": {"ck": at(rows, f)["ck"], "rows": at(rows, f)["rows"], "policy_ce": at(rows, f)["val_policy_ce"],
                                        "value_nll": at(rows, f)["val_value_nll"], "top1": at(rows, f)["val_top1_agree"]}
                   for f in (0.25, 0.5, 0.75, 1.0)},
            "end_is_just_after_reset": bool(last.get("reset_after")),
        }
    out = {"arms": table}
    cap4 = [n for n, s in spec.items() if n in table and s["epochs"] == 4 and not s["reset_every"]]
    if cap4:
        ends = [table[n]["end"]["val_policy_ce"] for n in cap4]
        ref, spread = sum(ends) / len(ends), (max(ends) - min(ends))
        rule = {"cap4_arms": cap4, "cap4_end_ce": ref, "cap4_spread": spread, "gain": {}}
        for n, t in table.items():
            if n not in cap4:
                rule["gain"][n] = ref - t["end"]["val_policy_ce"]
        ranked = sorted((n for n in table if n not in cap4), key=lambda n: table[n]["end"]["val_policy_ce"])
        verdict = "cap4"
        for n in ranked:
            if rule["gain"][n] <= spread:
                break  # this one and every worse one fail to beat cap 4 by more than the cap-4 arms differ
            if spec[n]["reset_every"]:  # resets only where they beat the same cap without them
                twin = [m for m in ranked if spec[m]["epochs"] == spec[n]["epochs"] and not spec[m]["reset_every"]]
                if twin and table[n]["end"]["val_policy_ce"] >= table[twin[0]]["end"]["val_policy_ce"] - spread:
                    continue
            verdict = n
            break
        rule.update(best_at_end=ranked[0] if ranked else None, adopt=verdict)
        out["rule"] = rule
    return out


def replay(root: Path, base: Path, out_dir: Path, arms: list[dict], *, window: int = 300_000, limit: int = 20000, batch: int = 256,
           lr: float = 1e-4, policy_weight: float = 1.0, device: str = "auto", say=print) -> dict:
    """Section 1.4: every arm over the newest `window` rows of ROOT's buffer in arrival order from BASE's weights, scored after
    every shard on the same held-out rows. Per-arm rows go to OUT_DIR/<arm>.jsonl (an arm already finished there is kept, so a
    rerun after an interruption continues with the next), the table to OUT_DIR/summary.json. ROOT is only read."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    torch.set_num_threads(1)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = Buffer(Path(root) / "buffer" / "shards", window)
    buf.refresh()
    flat = buf.arrays()
    if not len(flat.get("train_idx", [])) or not len(flat.get("val_idx", [])):
        raise RuntimeError("the buffer has no training or no held-out rows")
    shards = []
    for name, part in buf.parts:
        stamp = name.rsplit("-", 1)[-1].split(".")[0]
        shards.append({"name": name, "rows": int(part["lengths"].size), "train": int(np.sum(part["val"] == 0)),
                       "t": int(stamp) if stamp.isdigit() else None})
    buf.parts = []  # the flat arrays hold every row; the shards' own copies are not needed again (the replay runs beside a busy loop)
    _, shape = load_net(Path(base) / "combat_policy.pt")
    prepared = prepare_eval(flat, shape, device, limit)
    say(f"replay: {len(shards)} shards, {buf.rows} rows ({len(flat['train_idx'])} training, {len(flat['val_idx'])} held-out, "
        f"{sum(n for n, _, _ in prepared)} scored), device {device}, {len(arms)} arms")
    results, spec = {}, {a["name"]: a for a in arms}
    for arm in arms:
        f = out_dir / f"{arm['name']}.jsonl"
        if f.is_file():
            results[arm["name"]] = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
            say(f"{arm['name']}: kept from an earlier run")
            continue
        rows = replay_arm(flat, shards, prepared, Path(base), arm, device, batch=batch, lr=lr, policy_weight=policy_weight,
                          free=card_free if device == "cuda" else None, say=say)
        tmp = f.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        os.replace(tmp, f)
        results[arm["name"]] = rows
    summary = {"base": Path(base).name, "window": window, "rows": buf.rows, "shards": len(shards), "heldout_rows": limit,
               "split": gl.val_split_id(), "device": device, "torch": torch.__version__, "batch": batch, "lr": lr,
               **summarize(results, spec)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["replay"]:  # section 1.4's arms on the stored buffer; prints the summary as one JSON document
        rp = argparse.ArgumentParser(prog="gym_loop_train.py replay")
        rp.add_argument("--root", required=True, help="the loop's folder: only buffer/shards is read")
        rp.add_argument("--base", required=True, help="the folder whose combat_policy.pt every arm starts from")
        rp.add_argument("--out", required=True, help="where the arms' rows and summary.json go (not under the loop's folder)")
        rp.add_argument("--arm", action="append", default=[], help="NAME:PASSES[:RESET_EVERY[:SEED]] (default: the six of section 1.4)")
        rp.add_argument("--window", type=int, default=300_000)
        rp.add_argument("--limit", type=int, default=20000, help="the newest held-out rows each arm is scored on")
        rp.add_argument("--batch", type=int, default=256)
        rp.add_argument("--lr", type=float, default=1e-4)
        rp.add_argument("--policy-weight", type=float, default=1.0)
        rp.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        r = rp.parse_args(argv[1:])
        if Path(r.out).resolve().is_relative_to(Path(r.root).resolve()):
            raise SystemExit("--out must be outside the loop's folder")
        from tools.lab import winsys  # noqa: PLC0415

        winsys.set_below_normal()
        arms = [parse_arm(s) for s in (r.arm or DEFAULT_ARMS)]
        print(json.dumps(replay(Path(r.root), Path(r.base), Path(r.out), arms, window=r.window, limit=r.limit, batch=r.batch,
                                lr=r.lr, policy_weight=r.policy_weight, device=r.device, say=lambda m: print(m, flush=True))))
        return 0
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
    ap.add_argument("--reset-every", type=int, default=0,
                    help="re-initialise the heads' last linear layers after every N-th checkpoint (0: never)")
    a = ap.parse_args(argv)
    return Trainer(a).run()


if __name__ == "__main__":
    raise SystemExit(main())
