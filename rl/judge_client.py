"""The search's side of the fight judge (rl/judge.py): a leaf's three numbers turned into the search's value, and the judge asked
for many leaves at once, on the card through the prior server in judge mode or on the CPU.

What it is. `estimate(row, max_hp)` reads the judge's row (win logit, end-HP-fraction logit, potions held at the end) as the
search's leaf value in its units (`live_search`: a win is worth its end HP plus `potion_value` per potion held, a death
`DEATH_VALUE`), with the death chance and the end HP if won. `RemoteJudge` sends a batch of positions to a prior server that
serves a judge checkpoint (`prior_server.py` loads either kind: a judge's reply row carries its three numbers in its first
three slots) as one PRIORS frame; on any transport failure it falls back, for the rest of the process, to the judge on the CPU
(`judge.LocalJudge`), with one stderr line. Its state is its own: a judge failure never touches the prior's connection or its
fallback.

What it must not do: import torch (only a process that falls back pays for it), unpickle anything, or raise out of a batch
because of the transport.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prior_protocol as proto  # noqa: E402

JUDGE_OUT = 3  # win logit, end-HP-fraction logit (given a win), potions held at the end (given a win)
DEATH_VALUE = -100.0  # live_search.DEATH_VALUE
POTION_VALUE = 10.0  # live_search's and L0's potion value

_LOCK = threading.RLock()
_CONNS: dict = {}
_STATE = {"fallback": None, "server": 0, "cpu": 0, "batches": 0, "ms": 0.0}
_IDS = iter(range(1, 1 << 31))


class Estimate(NamedTuple):
    value: float  # the search's value at the leaf (HP units)
    p_death: float
    hp_if_win: float
    potions_if_win: float


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def l0_of(p_win, hp_frac, potions, max_hp, potion_value: float = POTION_VALUE):
    """The L0 the judge expects: P(win) x (0.5 + 0.5 min(1, end HP fraction + potion_value x potions / max HP))."""
    p_win, hp_frac, potions, max_hp = (np.asarray(x, np.float64) for x in (p_win, hp_frac, potions, max_hp))
    return p_win * (0.5 + 0.5 * np.minimum(1.0, hp_frac + potion_value * potions / np.maximum(1.0, max_hp)))


def estimate(row: Sequence[float], max_hp: int, potion_value: float = POTION_VALUE, terminal=None) -> Estimate:
    """The judge's row as the search's leaf value: P(win) x (end HP + potion_value x potions) + P(death) x DEATH_VALUE.

    With a `terminal` function (the run currency, rl/run_currency.py `FightCurrency`) the won branch is its value of (end HP if won, max HP,
    expected potions held if won) instead of HP + potion_value x potions: the leaf composes the judge's chances through the same pricing a
    rolled-out win gets. The judge gives a potion count, not which potions, so the currency averages over the starting potions (a mean of
    the judge's outputs put through the head is Jensen's error; a judge trained on the currency's u, brief 3b, removes it)."""
    p = float(_sigmoid(row[0]))
    hp = float(_sigmoid(row[1])) * max(1, int(max_hp))
    pots = max(0.0, float(row[2]))
    won = float(terminal(hp, max_hp, pots)) if terminal is not None else hp + potion_value * pots
    return Estimate(p * won + (1.0 - p) * DEATH_VALUE, 1.0 - p, hp, pots)


def stats() -> dict:
    with _LOCK:
        s = dict(_STATE)
    s["mean_batch_ms"] = round(s.pop("ms") / s["batches"], 3) if s["batches"] else None
    return s


def reset_state() -> None:
    """Forget the connections and the fallback. For tests only."""
    with _LOCK:
        for c in _CONNS.values():
            try:
                c.close()
            except Exception:  # noqa: BLE001, S110
                pass
        _CONNS.clear()
        _STATE.update({"fallback": None, "server": 0, "cpu": 0, "batches": 0, "ms": 0.0})


def encode_leaf(handle: int):
    """The tokens the judge reads at a leaf: the prior's encoding without the end-turn probe (the judge is trained on it)."""
    from prior_client import encode_position  # noqa: PLC0415

    return encode_position(handle, False)


@dataclass(frozen=True)
class CpuJudge:
    """The judge on the CPU, in process: a batch of handles in, their rows out."""

    path: str

    def warm(self) -> None:
        import judge  # noqa: PLC0415

        judge.local(self.path)

    def rows(self, handles: Sequence[int]) -> np.ndarray:
        import judge  # noqa: PLC0415

        out = judge.local(self.path).rows([encode_leaf(h) for h in handles])
        with _LOCK:
            _STATE["cpu"] += len(handles)
        return out


@dataclass(frozen=True)
class RemoteJudge:
    """The judge on the card: one PRIORS frame per batch to a prior server serving the judge checkpoint `path`."""

    path: str
    endpoint: str
    authkey: bytes = field(repr=False)
    policy_sha256: str  # the judge checkpoint's sha256: the server's HELLO checks it
    timeout_s: float = 10.0

    def _conn(self):
        import prior_client  # noqa: PLC0415

        key = (os.getpid(), self.endpoint)
        conn = _CONNS.get(key)
        if conn is None:
            conn = _CONNS[key] = prior_client._open(self)  # noqa: SLF001 -- the same HELLO, with the judge's sha256
        return conn

    def warm(self) -> None:
        with _LOCK:
            if _STATE["fallback"] is None:
                try:
                    self._conn()
                except Exception as error:  # noqa: BLE001
                    self._fall_back(error)

    def _fall_back(self, error: BaseException) -> None:
        with _LOCK:
            if _STATE["fallback"] is None:
                _STATE["fallback"] = proto.describe_error(error)
                print(f"judge_client: using the CPU judge from now on ({_STATE['fallback']})", file=sys.stderr, flush=True)
            for c in _CONNS.values():
                try:
                    c.close()
                except Exception:  # noqa: BLE001, S110
                    pass
            _CONNS.clear()

    def _ask(self, tokens: list) -> np.ndarray:
        with _LOCK:
            conn = self._conn()
            req = next(_IDS) & 0xFFFFFFFF
            t0 = time.perf_counter()
            conn.send_bytes(proto.pack_priors(req, tokens))
            if not conn.poll(self.timeout_s):
                raise TimeoutError(f"no reply within {self.timeout_s:g} s")
            status, got, rows = proto.unpack_priors_reply(conn.recv_bytes())
            if got != req or status != proto.REPLY_OK or rows.shape[0] != len(tokens):
                raise proto.ProtocolError(f"judge reply status {status}, request {got} for {req}, {rows.shape[0]} rows")
            _STATE["ms"] += (time.perf_counter() - t0) * 1000.0
            _STATE["batches"] += 1
            _STATE["server"] += len(tokens)
            return rows[:, :JUDGE_OUT].astype(np.float64)

    def rows(self, handles: Sequence[int]) -> np.ndarray:
        if not handles:
            return np.zeros((0, JUDGE_OUT))
        tokens = [encode_leaf(h) for h in handles]
        if _STATE["fallback"] is None:
            try:
                return self._ask(tokens)
            except Exception as error:  # noqa: BLE001 -- any transport failure falls back; the searches go on
                self._fall_back(error)
        import judge  # noqa: PLC0415

        out = judge.local(self.path).rows(tokens)
        with _LOCK:
            _STATE["cpu"] += len(handles)
        return out


def leaf_fn_of(judge):
    """`live_search`'s `leaf_fn(handle)` for the sequential driver: the judge's raw row for one leaf (the search reads it with
    `estimate`)."""

    def leaf_fn(handle: int) -> np.ndarray:
        return judge.rows([handle])[0]

    return leaf_fn
