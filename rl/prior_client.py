"""The worker side of the batched prior server: `RemotePrior`, the drop-in for `combat_prior.LivePrior`.

What it is: a frozen dataclass a self-play worker pickles in with its agent. On a call it encodes the position as
`LivePrior` does, sends the tokens to the prior server (`rl/prior_server.py`, one pipe connection per process, opened on
first use) and turns the 32 logits it gets back into probabilities over the legal actions with
`prior_protocol.legal_weights`, the code `PolicyNet.weights` runs. It also holds the per-process counters and latency
samples (`stats`) a run reads to record which prior played it.

Why: the server runs the forward on the GPU; the worker keeps the emulator, the encode and the softmax, and imports no
torch (`docs/rl/GPU_INFERENCE.md` sections 2, 3 and 6).

What it must not do: import torch (only a process that falls back pays for it, through `combat_prior`), call `recv()` or
`send()` (they unpickle; frames travel as bytes), raise out of a prior call because of the transport, or retry: the first
failure of any kind (connect, auth, HELLO refused, timeout, EOF, bad frame, wrong request id) makes this process use the
in-process CPU prior for the rest of its life, with one stderr line carrying the exception type and message only.
"""

from __future__ import annotations

import ctypes
import itertools
import json
import os
import random
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import connection
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prior_protocol as proto
from prior_protocol import legal_weights

RESERVOIR = 4096

_LOCK = threading.RLock()
_CONNS: dict[tuple[int, str], connection.Connection] = {}
_COUNTS = {"server": 0, "cpu": 0}
_FALLBACK: list[str | None] = [None]
_LATENCY: list[float] = []
_LATENCY_SEEN = [0]
_RNG = random.Random(0)
_REQUEST_IDS = itertools.count(1)
_LOCAL: dict[tuple[str, bool], object] = {}
_HASHES: list[tuple[str, str] | None] = [None]


def stats() -> dict:
    """Calls answered by the server and by the CPU fallback, why this process fell back (None if it has not), latency in ms."""
    with _LOCK:
        samples = np.array(_LATENCY, dtype=np.float64)
        seen = _LATENCY_SEEN[0]
        out = {"server": _COUNTS["server"], "cpu": _COUNTS["cpu"], "fallback_reason": _FALLBACK[0]}
    if samples.size:
        p50, p99 = (round(float(x), 3) for x in np.percentile(samples, [50, 99]))
    else:
        p50 = p99 = None
    out["latency_ms"] = {"p50": p50, "p99": p99, "n": seen}
    return out


def reset_stats() -> None:
    """Zero the counters and the latency samples (the sticky fallback and the connections are kept)."""
    with _LOCK:
        _COUNTS["server"] = _COUNTS["cpu"] = 0
        _LATENCY.clear()
        _LATENCY_SEEN[0] = 0


def reset_state() -> None:
    """Forget everything, connections and the sticky fallback included. For tests only."""
    with _LOCK:
        for conn in _CONNS.values():
            _close(conn)
        _CONNS.clear()
        _FALLBACK[0] = None
        _LOCAL.clear()
    reset_stats()


def _close(conn) -> None:
    try:
        conn.close()
    except Exception:  # noqa: BLE001, S110 -- closing a connection that is already broken
        pass


def _sample(milliseconds: float) -> None:
    """Reservoir sampling (algorithm R): a bounded, unbiased sample of every latency seen."""
    with _LOCK:
        _LATENCY_SEEN[0] += 1
        if len(_LATENCY) < RESERVOIR:
            _LATENCY.append(milliseconds)
        else:
            slot = _RNG.randrange(_LATENCY_SEEN[0])
            if slot < RESERVOIR:
                _LATENCY[slot] = milliseconds


def _fall_back(error: BaseException) -> None:
    reason = proto.describe_error(error)
    with _LOCK:
        if _FALLBACK[0] is None:
            _FALLBACK[0] = reason
            print(f"prior_client: using the CPU prior from now on ({reason})", file=sys.stderr, flush=True)
        for conn in _CONNS.values():
            _close(conn)
        _CONNS.clear()


def _within(seconds: float, call):
    """Run `call()` and give up after `seconds`: a server that accepts and then says nothing must not hang a worker."""
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = call()
        except BaseException as error:  # noqa: BLE001
            box["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise TimeoutError(f"no answer within {seconds:g} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


class HelloRefused(Exception):
    """The server answered HELLO with a status other than ok."""


def _hashes() -> tuple[str, str]:
    """(vocab_sha256, card_table_sha256) of this build, computed once per process."""
    if _HASHES[0] is None:
        from combat_prior import vocab_sha256  # noqa: PLC0415 -- imports no torch

        _HASHES[0] = (vocab_sha256(), proto.card_table_sha256())
    return _HASHES[0]


def _open(prior: RemotePrior):
    family = "AF_PIPE" if sys.platform == "win32" else "AF_UNIX"
    address = proto.pipe_address(prior.endpoint)
    vocab, card = _hashes()
    hello = proto.pack_hello(prior.policy_sha256, vocab, card)

    def connect():
        conn = connection.Client(address, family=family, authkey=prior.authkey)
        try:
            conn.send_bytes(hello)
            if not conn.poll(prior.timeout_s):
                raise TimeoutError("no HELLO answer")
            status, reason = proto.unpack_hello_reply(conn.recv_bytes())
        except BaseException:
            _close(conn)
            raise
        if status != proto.STATUS_OK:
            _close(conn)
            raise HelloRefused(f"HELLO refused, status {status} ({proto.STATUS_NAMES.get(status, '?')}): {reason}")
        return conn

    return _within(prior.timeout_s + 2.0, connect)


def _connection(prior: RemotePrior):
    key = (os.getpid(), prior.endpoint)
    conn = _CONNS.get(key)
    if conn is None:
        conn = _CONNS[key] = _open(prior)
    return conn


def _local(prior: RemotePrior):
    """The in-process CPU prior of the fallback; `combat_prior` (and so torch, on first use) is imported only here."""
    key = (prior.policy, prior.probe)
    found = _LOCAL.get(key)
    if found is None:
        from combat_prior import LivePrior  # noqa: PLC0415

        found = _LOCAL[key] = LivePrior(prior.policy, prior.probe)
    return found


@dataclass(frozen=True)
class RemotePrior:
    """`live_search`'s `prior_fn(handle, legal)`, answered by the prior server and, failing that, in process.

    `policy` and `probe` are what `LivePrior` takes (the fallback is that prior); `endpoint`, `authkey` and the policy
    file's sha256 say which server to ask and what it must be serving. The handle is only read, as for `LivePrior`.
    """

    policy: str
    probe: bool
    endpoint: str
    authkey: bytes
    policy_sha256: str
    timeout_s: float = 5.0

    def warm(self) -> None:
        """Open the connection now (HELLO included), so the first search does not spend its budget on it."""
        with _LOCK:
            if _FALLBACK[0] is None:
                try:
                    _connection(self)
                except Exception as error:  # noqa: BLE001
                    _fall_back(error)

    def _ask(self, tokens) -> np.ndarray:
        with _LOCK:
            conn = _connection(self)
            req_id = next(_REQUEST_IDS) & 0xFFFFFFFF
            frame = proto.pack_prior(req_id, tokens)
            started = time.perf_counter()
            conn.send_bytes(frame)
            if not conn.poll(self.timeout_s):
                raise TimeoutError(f"no reply within {self.timeout_s:g} s")
            status, got, logits = proto.unpack_reply(conn.recv_bytes())
            elapsed = (time.perf_counter() - started) * 1000.0
            if got != req_id:
                raise proto.ProtocolError(f"reply for request {got}, expected {req_id}")
            if status != proto.REPLY_OK:
                raise proto.ProtocolError(f"server answered status {status}")
            _sample(elapsed)
            return logits

    def __call__(self, handle: int, legal: Sequence[int]) -> np.ndarray:
        if _FALLBACK[0] is None:
            from combat_tokens import encode  # noqa: PLC0415 -- loads the emulator library, which a worker has anyway

            from sts2_gym import native  # noqa: PLC0415

            obs = (ctypes.c_int * native.OBS_SIZE)()
            view = native.clone(handle, False, 0, obs)
            native.destroy(view)
            tokens = encode(handle, obs, native.encounter_id(handle), probe=self.probe)
            try:
                row = self._ask(tokens)
            except Exception as error:  # noqa: BLE001 -- any transport failure falls back; the search goes on
                _fall_back(error)
            else:
                with _LOCK:
                    _COUNTS["server"] += 1
                return legal_weights(row, list(legal))
        weights = _local(self)(handle, legal)
        with _LOCK:
            _COUNTS["cpu"] += 1
        return weights


# ---------------------------------------------------------------------------------------------- bench client

# Mirrors of combat_tokens' type numbers, for synthetic positions; a test pins them to the real ones.
_T_GLOBAL, _T_HAND, _T_DRAW, _T_DISCARD, _T_EXHAUST, _T_ENEMY, _T_BUFF = 1, 2, 3, 4, 5, 6, 7
_T_ENEMY_BUFF, _T_END_TURN, _T_RELIC, _T_INTENT, _T_INTENT_NEXT = 9, 10, 14, 15, 16
_FILLER_TYPES = (_T_DRAW, _T_DISCARD, _T_EXHAUST, _T_ENEMY, _T_BUFF, _T_ENEMY_BUFF, _T_RELIC, _T_INTENT, _T_INTENT_NEXT)


@dataclass
class SyntheticTokens:
    types: np.ndarray
    ids: np.ndarray
    owners: np.ndarray
    feats: np.ndarray

    def __len__(self) -> int:
        return len(self.types)


def synthetic_tokens(rng: np.random.Generator, count: int) -> SyntheticTokens:
    """A valid, real-shaped position of `count` tokens: a global token, a block of action tokens, then other tokens."""
    count = max(count, 4)
    actions = int(rng.integers(2, min(count - 1, 12) + 1))
    types = np.empty(count, dtype=np.int64)
    types[0] = _T_GLOBAL
    types[1 : 1 + actions - 1] = _T_HAND
    types[actions] = _T_END_TURN
    types[1 + actions :] = rng.choice(_FILLER_TYPES, count - 1 - actions)
    return SyntheticTokens(
        types,
        rng.integers(0, 3, count).astype(np.int64),
        rng.integers(0, 2, count).astype(np.int64),
        rng.random((count, proto.N_FEATURES), dtype=np.float32),
    )


def bench_client(endpoint: str, authkey: bytes, shas: tuple[str, str, str], count: int, seed: int) -> list[float]:
    """Send `count` synthetic requests back to back over one connection; return each round trip in ms.

    Prints `connected`, then waits for a line on stdin before it starts timing, so the selftest can start every client
    together. Used by `prior_server.py --selftest`.
    """
    rng = np.random.default_rng(seed)
    pool = [synthetic_tokens(rng, int(n)) for n in rng.integers(16, 41, 64)]
    family = "AF_PIPE" if sys.platform == "win32" else "AF_UNIX"
    conn = connection.Client(proto.pipe_address(endpoint), family=family, authkey=authkey)
    conn.send_bytes(proto.pack_hello(*shas))
    status, reason = proto.unpack_hello_reply(conn.recv_bytes())
    if status != proto.STATUS_OK:
        raise HelloRefused(f"status {status}: {reason}")
    print("connected", flush=True)
    sys.stdin.readline()
    times = []
    for i in range(count):
        frame = proto.pack_prior(i + 1, pool[i % len(pool)])
        started = time.perf_counter()
        conn.send_bytes(frame)
        status, got, row = proto.unpack_reply(conn.recv_bytes())
        times.append((time.perf_counter() - started) * 1000.0)
        if status != proto.REPLY_OK or got != i + 1:
            raise proto.ProtocolError(f"reply status {status}, request {got} for {i + 1}")
        legal_weights(row, [1, 2, 3])
    conn.send_bytes(proto.pack_bye())
    conn.close()
    return times


def main(argv: Sequence[str] | None = None) -> None:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="The synthetic client of prior_server.py --selftest.")
    parser.add_argument("--bench", required=True, metavar="ENDPOINT")
    parser.add_argument("--authkey-env", required=True)
    parser.add_argument("--shas", required=True, help="policy,vocab,card_table sha256, comma separated")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    authkey = bytes.fromhex(os.environ.pop(args.authkey_env).strip())
    shas = tuple(args.shas.split(","))
    try:
        times = bench_client(args.bench, authkey, shas, args.n, args.seed)
    except Exception as error:  # noqa: BLE001 -- type and message only, as everywhere
        print(f"prior_client: {proto.describe_error(error)}", file=sys.stderr, flush=True)
        sys.exit(1)
    print(json.dumps({"n": len(times), "ms": [round(t, 3) for t in times]}), flush=True)


if __name__ == "__main__":
    main()
