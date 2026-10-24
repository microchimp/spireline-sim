#!/usr/bin/env python3
"""Section 9.2 of docs/rl/GPU_INFERENCE.md: the GPU server's logits against the CPU prior's, on recorded positions.

What it does: samples positions (a fixed seed) from the newest `combat-g<gen>-*.npz` parts of a self-play shard folder,
rebuilds each position's `Tokens` from the part's concatenated arrays (`gen_loop.write_part` stores the four token arrays
end to end, `lengths` gives the offsets), computes the CPU logits the way `combat_prior.PolicyNet` does (one unpadded
sequence, `policy_logits`, one torch thread), and gets the GPU logits from a real `prior_server.py` over its pipe:
once from several client threads (mixed buckets, padding) and once one position at a time (a bucket of one). It then
reports, over the action slots, the largest and the 99.9th-percentile |dlogit|, the argmax flips among positions whose
top-2 gap exceeds 1e-5 (the gate: none), and the same on the float64 `legal_weights` probabilities.

Run it under the GPU venv's python (the server needs CUDA; the CPU logits are computed in the same interpreter unless
`--cpu-logits FILE` brings logits dumped by `--dump-cpu FILE` under the emulator venv's python, which is the torch the
workers use):

    python rl/prior_tolerance.py --shards-glob "<lab>/runs/*-selfplay/shards/combat-g3-*.npz" \
        --policy <gen dir>/combat_policy.pt --lib <native library dir> --json OUT.json

The pure parts (rebuild, compare, the paragraph) import no torch. The JSON holds numbers only, no paths.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shlex
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

TOKEN_KEYS = ("types", "ids", "owners", "feats")  # gen_loop.TOKEN_KEYS (not imported: gen_loop is the lab's, this runs bare)
GAP = 1e-5  # the gate's top-2 gap below which an argmax flip is not counted (docs section 5)
LOGIT_TOLERANCE = 1e-4  # the gate's bound on max |dlogit| at the action slots


# ------------------------------------------------------------------ positions


def newest_parts(pattern: str, pool: int) -> list[str]:
    """The newest parts (by modification time, name as the tie-break) holding at least `pool` positions between them."""
    files = [f for f in glob.glob(pattern) if ".tmp" not in os.path.basename(f)]  # write_part's temporary files
    files.sort(key=lambda f: (os.path.getmtime(f), f), reverse=True)
    out, have = [], 0
    for f in files:
        with np.load(f, allow_pickle=False) as blob:
            n = int(blob["lengths"].size)
        if n:
            out.append(f)
            have += n
        if have >= pool:
            break
    return out


def rebuild_tokens(part: dict, rows, make=None) -> list:
    """The `Tokens` of the given sample rows of one part. `part` is `gen_loop.read_part`'s dict: `types`, `ids`, `owners`
    and `feats` hold every sample's tokens end to end and `lengths` says how many each has, so sample i is the slice from
    `lengths[:i].sum()`. `make(types, ids, owners, feats)` builds one (default: `combat_tokens.Tokens`)."""
    if make is None:
        from combat_tokens import Tokens as make  # noqa: PLC0415, N813 -- native; only the real tool needs it
    lengths = np.asarray(part["lengths"], dtype=np.int64)
    ends = np.cumsum(lengths)
    out = []
    for i in rows:
        lo, hi = int(ends[i] - lengths[i]), int(ends[i])
        out.append(make(part["types"][lo:hi].copy(), part["ids"][lo:hi].copy(), part["owners"][lo:hi].copy(),
                        part["feats"][lo:hi].copy()))
    return out


def load_positions(pattern: str, n: int, seed: int, pool: int):
    """(tokens list, legal [n, 32] int8, info dict): `n` positions drawn without replacement, with a fixed seed, from the newest
    parts. The order is the draw's order, so the same arguments always give the same positions."""
    parts = newest_parts(pattern, max(pool, n))
    if not parts:
        raise SystemExit(f"no parts match {pattern!r}")
    index = []  # (part number, row)
    sizes = []
    for k, f in enumerate(parts):
        with np.load(f, allow_pickle=False) as blob:
            size = int(blob["lengths"].size)
        sizes.append(size)
        index.extend((k, r) for r in range(size))
    if len(index) < n:
        raise SystemExit(f"only {len(index)} positions in {len(parts)} parts, {n} wanted")
    pick = np.random.default_rng(seed).choice(len(index), size=n, replace=False)
    chosen = [index[i] for i in pick]
    by_part: dict[int, list[int]] = {}
    for k, r in chosen:
        by_part.setdefault(k, []).append(r)
    cache: dict[tuple[int, int], tuple] = {}
    for k, rows in by_part.items():
        with np.load(parts[k], allow_pickle=False) as blob:
            part = {key: blob[key] for key in (*TOKEN_KEYS, "lengths", "legal")}
        for r, tok in zip(rows, rebuild_tokens(part, rows), strict=True):
            cache[(k, r)] = (tok, part["legal"][r].copy())
    tokens = [cache[c][0] for c in chosen]
    legal = np.stack([cache[c][1] for c in chosen]).astype(np.int8)
    lens = np.array([len(t) for t in tokens])
    info = {"n": n, "seed": seed, "parts": len(parts), "pool": len(index),
            "tokens_min": int(lens.min()), "tokens_mean": round(float(lens.mean()), 2), "tokens_max": int(lens.max())}
    return tokens, legal, info


def positions_digest(tokens: list, legal: np.ndarray) -> str:
    """sha256 over the sampled positions' tokens (length, types, ids, owners, feats) and legal rows, in order: it names the
    positions a dumped set of CPU logits belongs to."""
    h = hashlib.sha256()
    for tok in tokens:
        h.update(len(tok).to_bytes(4, "little"))
        for key in TOKEN_KEYS:
            h.update(np.ascontiguousarray(getattr(tok, key), dtype=np.float32 if key == "feats" else np.int64).tobytes())
    h.update(np.ascontiguousarray(legal, dtype=np.int8).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ the comparison


def _top2_gap(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per row: the gap between the largest and second-largest value among the masked entries (nan with fewer than two)."""
    masked = np.where(mask, values.astype(np.float64), -np.inf)
    top = np.sort(masked, axis=1)[:, -2:]
    gap = top[:, 1] - top[:, 0]
    return np.where(np.isfinite(top[:, 0]), gap, np.nan)


def _flips(cpu: np.ndarray, gpu: np.ndarray, mask: np.ndarray) -> dict:
    """Argmax flips among the masked slots: counted where the CPU's top-2 gap exceeds GAP; the rest are listed apart."""
    gap = _top2_gap(cpu, mask)
    ok = np.nan_to_num(gap, nan=0.0) > GAP
    a = np.argmax(np.where(mask, cpu.astype(np.float64), -np.inf), axis=1)
    b = np.argmax(np.where(mask, gpu.astype(np.float64), -np.inf), axis=1)
    differ = a != b
    return {"positions": int(ok.sum()), "flips": int((differ & ok).sum()), "below_gap_positions": int((~ok).sum()),
            "below_gap_flips": int((differ & ~ok).sum()), "below_gap_single_slot": int(np.isnan(gap).sum())}


def _dist(diff: np.ndarray) -> dict:
    if diff.size == 0:
        return {"max": None, "p99.9": None, "mean": None, "n": 0}
    return {"max": float(diff.max()), "p99.9": float(np.percentile(diff, 99.9)), "mean": float(diff.mean()), "n": int(diff.size)}


def compare(cpu: np.ndarray, gpu: np.ndarray, legal: np.ndarray) -> dict:
    """CPU logits against GPU logits, both [n, 32] (the action slots hold a logit, the others -inf).

    `mismatch_finite`: slots finite in one and not the other (must be 0). `logit`: |dlogit| over the slots finite in both;
    `logit_per_position_max`: the same, one maximum per position. `flips_action_slots` / `flips_legal_slots`: argmax flips
    among the positions whose top-2 gap exceeds 1e-5, over every action slot and over the recorded legal slots only (the
    softmax the search reads is over the legal slots). `prob`: |dp| of `prior_protocol.legal_weights` (float64, the recorded
    legal row) and its argmax flips among positions whose top-2 probability gap exceeds 1e-5.
    """
    from prior_protocol import legal_weights  # noqa: PLC0415

    cpu = np.asarray(cpu, dtype=np.float32)
    gpu = np.asarray(gpu, dtype=np.float32)
    fin_c, fin_g = np.isfinite(cpu), np.isfinite(gpu)
    both = fin_c & fin_g
    with np.errstate(invalid="ignore"):  # inf - inf at the slots that are no action
        diff = np.where(both, np.abs(cpu.astype(np.float64) - gpu.astype(np.float64)), 0.0)
    legal_mask = (np.asarray(legal) != 0) & both
    pc, pg, gaps_ok, flips = [], [], [], 0
    pdiff = []
    for i in range(cpu.shape[0]):
        idx = np.flatnonzero(np.asarray(legal[i]) != 0)
        a, b = legal_weights(cpu[i], idx), legal_weights(gpu[i], idx)
        pdiff.append(np.abs(a - b))
        pc.append(a)
        pg.append(b)
    pdiff_all = np.concatenate(pdiff) if pdiff else np.zeros(0)
    pflip_pos = pflip = 0
    for a, b in zip(pc, pg, strict=True):
        if a.size < 2:
            continue
        top = np.sort(a)[-2:]
        if top[1] - top[0] > GAP:
            pflip_pos += 1
            pflip += int(np.argmax(a) != np.argmax(b))
    return {"positions": int(cpu.shape[0]),
            "mismatch_finite": int((fin_c != fin_g).sum()),
            "mismatch_finite_positions": int((fin_c != fin_g).any(axis=1).sum()),
            "logit": _dist(diff[both]),
            "logit_per_position_max": _dist(diff.max(axis=1)),
            "logit_legal_slots": _dist(diff[legal_mask]),
            "flips_action_slots": _flips(cpu, gpu, both),
            "flips_legal_slots": _flips(cpu, gpu, legal_mask),
            "prob": {**_dist(pdiff_all), "flips": pflip, "flip_positions": pflip_pos}}


def paragraph(result: dict) -> str:
    """The sentence block for section 9.2 of the doc, numbers only."""
    p, t = result["positions"], result["torch"]
    lines = [f"{p['n']:,} positions (seed {p['seed']}, drawn from the newest {p['parts']} parts of a {p['pool']:,}-position "
             f"pool, {p['tokens_min']}-{p['tokens_max']} tokens, mean {p['tokens_mean']}); CPU logits by torch {t['cpu']}, "
             f"server on torch {t['server']}."]
    for name in ("batched", "alone"):
        r = result[name]
        if r is None:
            continue
        s = result["server"][name]
        lines.append(
            f"- {name} ({s['description']}): max |dlogit| {r['logit']['max']:.3g}, 99.9th percentile {r['logit']['p99.9']:.3g} "
            f"over {r['logit']['n']:,} action slots (per-position max: mean {r['logit_per_position_max']['mean']:.3g}); "
            f"slots finite in one path only: {r['mismatch_finite']}; argmax flips {r['flips_action_slots']['flips']} of "
            f"{r['flips_action_slots']['positions']:,} positions with a top-2 gap over 1e-5 "
            f"({r['flips_legal_slots']['flips']} of {r['flips_legal_slots']['positions']:,} over the legal slots; "
            f"{r['flips_action_slots']['below_gap_flips']} flips among the {r['flips_action_slots']['below_gap_positions']} "
            f"positions with no gap over 1e-5 (ties or near-ties; positions with fewer than two action slots are listed apart), not counted); float64 probabilities: max |dp| {r['prob']['max']:.3g}, 99.9th "
            f"percentile {r['prob']['p99.9']:.3g}, flips {r['prob']['flips']} of {r['prob']['flip_positions']:,}.")
    lines.append(f"Gate (max |dlogit| <= {LOGIT_TOLERANCE:g}, no flips above the gap, no finiteness mismatch): "
                 f"{'PASS' if result['gate']['pass'] else 'FAIL'}.")
    return "\n".join(lines)


def gate(result_parts: list[dict]) -> dict:
    ok = True
    worst = 0.0
    for r in result_parts:
        if r is None:
            continue
        worst = max(worst, r["logit"]["max"] or 0.0)
        ok = ok and (r["logit"]["max"] or 0.0) <= LOGIT_TOLERANCE and r["flips_action_slots"]["flips"] == 0 \
            and r["flips_legal_slots"]["flips"] == 0 and r["mismatch_finite"] == 0
    return {"pass": bool(ok), "max_dlogit": worst, "bound": LOGIT_TOLERANCE, "gap": GAP}


# ------------------------------------------------------------------ the CPU logits


def cpu_logits(policy: str, tokens: list) -> np.ndarray:
    """The logits `combat_prior.PolicyNet` computes for one unpadded sequence at a time, [n, 32] float32."""
    import torch  # noqa: PLC0415
    from combat_prior import PolicyNet  # noqa: PLC0415

    net = PolicyNet(policy)  # one torch thread, eval, fp32
    out = np.empty((len(tokens), 32), dtype=np.float32)
    with torch.inference_mode():
        for i, tok in enumerate(tokens):
            out[i] = net.model.policy_logits(**net._batch([tok]))[0].numpy()  # noqa: SLF001 -- the net's own batch builder
    return out


# ------------------------------------------------------------------ the server's logits


def _client(handle, policy_sha: str, vocab_sha: str, card_sha: str):
    from multiprocessing import connection  # noqa: PLC0415

    import prior_protocol as proto  # noqa: PLC0415

    family = "AF_PIPE" if sys.platform == "win32" else "AF_UNIX"
    conn = connection.Client(proto.pipe_address(handle.endpoint), family=family, authkey=handle.authkey)
    conn.send_bytes(proto.pack_hello(policy_sha, vocab_sha, card_sha))
    status, reason = proto.unpack_hello_reply(conn.recv_bytes())
    if status != proto.STATUS_OK:
        raise RuntimeError(f"HELLO refused: status {status} {reason}")
    return conn


def server_logits(handle, policy: str, tokens: list, threads: int) -> np.ndarray:
    """The server's logits for every position, asked by `threads` client threads (each its own connection, positions handed
    out in order), [n, 32] float32."""
    import combat_prior  # noqa: PLC0415
    import prior_protocol as proto  # noqa: PLC0415

    shas = (hashlib.sha256(Path(policy).read_bytes()).hexdigest(), combat_prior.vocab_sha256(), proto.card_table_sha256())
    out = np.empty((len(tokens), 32), dtype=np.float32)
    counter, lock, errors = [0], threading.Lock(), []

    def work(k: int) -> None:
        try:
            conn = _client(handle, *shas)
            while True:
                with lock:
                    i = counter[0]
                    counter[0] += 1
                if i >= len(tokens):
                    break
                conn.send_bytes(proto.pack_prior(i + 1, tokens[i]))
                status, got, row = proto.unpack_reply(conn.recv_bytes())
                if status != proto.REPLY_OK or got != i + 1:
                    raise RuntimeError(f"reply status {status}, request {got} for {i + 1}")
                out[i] = row
            conn.send_bytes(proto.pack_bye())
            conn.close()
        except Exception as error:  # noqa: BLE001 -- type and message only
            errors.append(f"{type(error).__name__}: {error}"[:200])

    pool = [threading.Thread(target=work, args=(k,), daemon=True) for k in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    if errors:
        raise RuntimeError(errors[0])
    return out


def run_server_phase(a, policy: str, tokens: list, threads: int, name: str, workdir: Path) -> tuple[np.ndarray, dict]:
    """Start a server, ask it for every position with `threads` clients, stop it; (logits, its facts)."""
    import prior_launch  # noqa: PLC0415

    handle = prior_launch.start_prior_server(
        sys.executable, policy, lib_path=os.environ.get("STS2_LIB_PATH"), kill_file=workdir / "GPU_OFF",
        stats_path=workdir / f"stats-{name}.json", extra_args=tuple(shlex.split(a.prior_server_args or "")))
    if handle is None:
        raise SystemExit("the prior server did not start")
    started = time.perf_counter()
    try:
        logits = server_logits(handle, policy, tokens, threads)
    finally:
        handle.stop()
    wall = time.perf_counter() - started
    stats = handle.read_stats()
    hist = {int(k): v for k, v in stats.get("batch_hist", {}).items()}
    batches = sum(hist.values())
    calls = sum(k * v for k, v in hist.items())
    facts = {"description": f"{threads} client threads" if threads > 1 else "one position at a time, a bucket of one",
             "threads": threads, "calls": stats.get("calls"), "batches": stats.get("batches"),
             "overlong": stats.get("overlong"), "mean_batch": round(calls / batches, 3) if batches else None,
             "batch_hist": {str(k): v for k, v in sorted(hist.items())}, "wall_s": round(wall, 1),
             "ready_s": stats.get("ready_s"), "buckets_captured": stats.get("buckets_captured")}
    return logits, facts


# ------------------------------------------------------------------ command line


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Section 9.2: GPU server logits against the CPU prior's, on recorded positions.")
    p.add_argument("--shards-glob", required=True, help="the parts, e.g. '<lab>/runs/*-selfplay/shards/combat-g3-*.npz'")
    p.add_argument("--policy", required=True, help="combat_policy.pt of the generation that played the parts")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261025)
    p.add_argument("--pool", type=int, default=40000, help="draw from the newest parts holding at least this many positions")
    p.add_argument("--threads", type=int, default=16, help="client threads of the batched phase")
    p.add_argument("--lib", help="STS2_LIB_PATH (the native library directory)")
    p.add_argument("--prior-server-args", help="extra flags for prior_server.py, shlex-split")
    p.add_argument("--json", help="write the result here")
    p.add_argument("--dump-cpu", metavar="FILE", help="compute the CPU logits, save them (npy) and stop; no server, no CUDA")
    p.add_argument("--cpu-logits", metavar="FILE", help="use CPU logits dumped by --dump-cpu (the emulator venv's torch)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.lib:
        os.environ["STS2_LIB_PATH"] = a.lib
    os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")
    import torch  # noqa: PLC0415

    tokens, legal, info = load_positions(a.shards_glob, a.n, a.seed, a.pool)
    print(f"positions: {info}", flush=True)
    digest = positions_digest(tokens, legal)
    t0 = time.perf_counter()
    cpu_torch = torch.__version__
    if a.cpu_logits:
        cpu = np.load(a.cpu_logits)
        meta = json.loads(Path(a.cpu_logits + ".json").read_text(encoding="utf-8"))
        cpu_torch = meta["torch"]
        if cpu.shape != (a.n, 32) or meta.get("n") != a.n or meta.get("seed") != a.seed:
            raise SystemExit("the dumped CPU logits are not for these positions (n or seed differ): dump them again with --dump-cpu")
        if meta.get("digest") != digest:
            raise SystemExit("the dumped CPU logits are for other positions (the position digest differs or is missing, e.g. new "
                             "parts arrived or --pool changed): dump them again with --dump-cpu")
    else:
        cpu = cpu_logits(a.policy, tokens)
    print(f"cpu logits: {time.perf_counter() - t0:.1f} s (torch {cpu_torch})", flush=True)
    if a.dump_cpu:
        np.save(a.dump_cpu, cpu)
        Path(a.dump_cpu + ".json").write_text(json.dumps({"n": a.n, "seed": a.seed, "torch": torch.__version__, "digest": digest}), encoding="utf-8")
        return 0
    result = {"positions": info, "torch": {"cpu": cpu_torch, "server": torch.__version__},
              "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "policy_sha256": hashlib.sha256(Path(a.policy).read_bytes()).hexdigest()[:16], "server": {}}
    with tempfile.TemporaryDirectory(prefix="spireline-tol-") as tmp:
        workdir = Path(tmp)
        gpu_b, result["server"]["batched"] = run_server_phase(a, a.policy, tokens, a.threads, "batched", workdir)
        gpu_1, result["server"]["alone"] = run_server_phase(a, a.policy, tokens, 1, "alone", workdir)
    result["batched"] = compare(cpu, gpu_b, legal)
    result["alone"] = compare(cpu, gpu_1, legal)
    fin = np.isfinite(gpu_b) & np.isfinite(gpu_1)
    with np.errstate(invalid="ignore"):
        result["batched_vs_alone"] = _dist(np.abs(gpu_b.astype(np.float64) - gpu_1.astype(np.float64))[fin])
    result["gate"] = gate([result["batched"], result["alone"]])
    text = paragraph(result)
    print(text)
    print(f"batched against alone (the bucket-to-bucket spread on the server): max {result['batched_vs_alone']['max']:.3g}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({**result, "paragraph": text}, indent=1), encoding="utf-8")
    return 0 if result["gate"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
