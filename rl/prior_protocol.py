"""The wire format between the self-play workers and the batched prior server, and the shared softmax.

What it is: the frames of `docs/rl/GPU_INFERENCE.md` section 3 (HELLO, PRIOR, REPLY, BYE) as pack and
unpack functions over `struct` and numpy, the `ProtocolError` every malformed frame raises, the hash of the card
attribute table, and `legal_weights`, the float64 softmax over the legal actions that `combat_prior.PolicyNet.weights`
used to hold inline (lifted out so the CPU prior and the remote prior turn logits into probabilities with the same
code).

Why: a worker answers a prior call by sending the position's tokens and getting the policy head's 32 logits back; the
softmax stays on the worker so the two paths differ only by where the logits were computed.

What it must not do: import torch or pickle anything (frames are plain bytes, a client never calls `recv()`), import
`combat_tokens` at module load (that loads the emulator DLL; `card_table_sha256` does it lazily), or trust a frame: a
bad magic, kind or length is a `ProtocolError`, never an index error or a short read.
"""

from __future__ import annotations

import hashlib
import os
import struct
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

MAGIC = b"SPR1"
PROTO = 1

HELLO, PRIOR, BYE = 1, 2, 3

STATUS_OK, STATUS_POLICY, STATUS_VOCAB, STATUS_PROTO, STATUS_CARD_TABLE = 0, 1, 2, 3, 4
STATUS_NAMES = {
    STATUS_OK: "ok",
    STATUS_POLICY: "policy mismatch",
    STATUS_VOCAB: "vocab mismatch",
    STATUS_PROTO: "proto mismatch",
    STATUS_CARD_TABLE: "card-table mismatch",
}
REPLY_OK, REPLY_BAD_REQUEST = 0, 1

# Mirrors of combat_tokens.MAX_ACTIONS and N_FEATURES; a test pins them to the real ones, so this module
# can stay free of the emulator DLL.
MAX_ACTIONS = 32
N_FEATURES = 18

_SHA_LEN = 64
_FRAME = struct.Struct("<4sHH")  # magic, kind, proto (HELLO) or status (HELLO reply)
_PRIOR_HEAD = struct.Struct("<4sHIHH")  # magic, kind, req_id, n_tokens, flags
_REPLY_HEAD = struct.Struct("<4sHHI")  # magic, kind, status, req_id
_KIND = struct.Struct("<4sH")
_PER_TOKEN = 1 + 4 + 1 + 4 * N_FEATURES
_HELLO_SIZE = _FRAME.size + 3 * _SHA_LEN
_REPLY_SIZE = _REPLY_HEAD.size + 4 * MAX_ACTIONS


class ProtocolError(Exception):
    """A frame that is not one of ours, or is cut short or too long."""


def pipe_address(endpoint: str) -> str:
    """The address `multiprocessing.connection` takes: on Windows a bare pipe name gets the local pipe prefix."""
    if sys.platform == "win32" and not endpoint.startswith("\\\\"):
        return "\\\\.\\pipe\\" + endpoint
    return endpoint


def describe_error(error: BaseException) -> str:
    """`Type: message` for a log line: no traceback, and the user's home directory is not written out."""
    text = f"{type(error).__name__}: {error}"
    for home in {str(Path.home()), os.environ.get("USERPROFILE", "")}:
        if home:
            text = text.replace(home, "~")
    return text[:300].replace("\n", " ")


def _sha(value: str, what: str) -> bytes:
    data = value.encode("ascii", "replace") if isinstance(value, str) else bytes(value)
    if len(data) != _SHA_LEN:
        raise ProtocolError(f"{what} must be {_SHA_LEN} ascii characters, got {len(data)}")
    return data


def peek_kind(data: bytes) -> int:
    """The kind of a frame, after checking its magic."""
    if len(data) < _KIND.size:
        raise ProtocolError(f"frame of {len(data)} bytes has no header")
    magic, kind = _KIND.unpack_from(data)
    if magic != MAGIC:
        raise ProtocolError("bad magic")
    return kind


def _check(data: bytes, kind: int, what: str) -> None:
    got = peek_kind(data)
    if got != kind:
        raise ProtocolError(f"{what} expected kind {kind}, got {got}")


def pack_hello(policy_sha256: str, vocab_sha256: str, card_table_sha256: str, proto: int = PROTO) -> bytes:
    return b"".join(
        (
            _FRAME.pack(MAGIC, HELLO, proto),
            _sha(policy_sha256, "policy_sha256"),
            _sha(vocab_sha256, "vocab_sha256"),
            _sha(card_table_sha256, "card_table_sha256"),
        )
    )


def unpack_hello(data: bytes) -> tuple[int, str, str, str]:
    """(proto, policy_sha256, vocab_sha256, card_table_sha256). The proto is not checked here: the server answers a mismatch."""
    _check(data, HELLO, "HELLO")
    if len(data) != _HELLO_SIZE:
        raise ProtocolError(f"HELLO of {len(data)} bytes, expected {_HELLO_SIZE}")
    _, _, proto = _FRAME.unpack_from(data)
    body = data[_FRAME.size :]
    shas = [body[i * _SHA_LEN : (i + 1) * _SHA_LEN].decode("ascii", "replace") for i in range(3)]
    return proto, shas[0], shas[1], shas[2]


def pack_hello_reply(status: int, reason: str = "") -> bytes:
    return _FRAME.pack(MAGIC, HELLO, status) + reason.encode("utf-8")


def unpack_hello_reply(data: bytes) -> tuple[int, str]:
    _check(data, HELLO, "HELLO reply")
    if len(data) < _FRAME.size:
        raise ProtocolError("HELLO reply cut short")
    _, _, status = _FRAME.unpack_from(data)
    return status, data[_FRAME.size :].decode("utf-8", "replace")


def pack_bye() -> bytes:
    return _KIND.pack(MAGIC, BYE)


def pack_prior(req_id: int, tokens, flags: int = 0) -> bytes:
    """A PRIOR frame for `tokens` (anything with .types .ids .owners .feats of one length)."""
    count = len(tokens.types)
    if not 0 < count < 1 << 16:
        raise ProtocolError(f"{count} tokens do not fit a frame")
    feats = np.ascontiguousarray(tokens.feats, dtype="<f4")
    if feats.shape != (count, N_FEATURES) or len(tokens.ids) != count or len(tokens.owners) != count:
        raise ProtocolError("token arrays differ in length or feats is not [T, N_FEATURES]")
    return b"".join(
        (
            _PRIOR_HEAD.pack(MAGIC, PRIOR, req_id & 0xFFFFFFFF, count, flags),
            np.ascontiguousarray(tokens.types, dtype="u1").tobytes(),
            np.ascontiguousarray(tokens.ids, dtype="<i4").tobytes(),
            np.ascontiguousarray(tokens.owners, dtype="u1").tobytes(),
            feats.tobytes(),
        )
    )


def unpack_prior(data: bytes) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """(req_id, types u8[T], ids i32[T], owners u8[T], feats f32[T, 18], flags); read-only views on `data`."""
    _check(data, PRIOR, "PRIOR")
    if len(data) < _PRIOR_HEAD.size:
        raise ProtocolError("PRIOR header cut short")
    _, _, req_id, count, flags = _PRIOR_HEAD.unpack_from(data)
    if count == 0:
        raise ProtocolError("PRIOR with no tokens")
    expected = _PRIOR_HEAD.size + count * _PER_TOKEN
    if len(data) != expected:
        raise ProtocolError(f"PRIOR of {len(data)} bytes, expected {expected} for {count} tokens")
    at = _PRIOR_HEAD.size
    types = np.frombuffer(data, "u1", count, at)
    at += count
    ids = np.frombuffer(data, "<i4", count, at)
    at += 4 * count
    owners = np.frombuffer(data, "u1", count, at)
    at += count
    feats = np.frombuffer(data, "<f4", count * N_FEATURES, at).reshape(count, N_FEATURES)
    return req_id, types, ids, owners, feats, flags


def pack_reply(status: int, req_id: int, logits: np.ndarray | None = None) -> bytes:
    """A REPLY frame; `logits` (float32, MAX_ACTIONS) is ignored, and sent as zeros, unless the status is ok."""
    if status == REPLY_OK:
        row = np.ascontiguousarray(logits, dtype="<f4")
        if row.shape != (MAX_ACTIONS,):
            raise ProtocolError(f"logits shape {row.shape}, expected ({MAX_ACTIONS},)")
        body = row.tobytes()
    else:
        body = bytes(4 * MAX_ACTIONS)
    return _REPLY_HEAD.pack(MAGIC, PRIOR, status, req_id & 0xFFFFFFFF) + body


def unpack_reply(data: bytes) -> tuple[int, int, np.ndarray]:
    """(status, req_id, logits f32[MAX_ACTIONS])."""
    _check(data, PRIOR, "REPLY")
    if len(data) != _REPLY_SIZE:
        raise ProtocolError(f"REPLY of {len(data)} bytes, expected {_REPLY_SIZE}")
    _, _, status, req_id = _REPLY_HEAD.unpack_from(data)
    return status, req_id, np.frombuffer(data, "<f4", MAX_ACTIONS, _REPLY_HEAD.size)


def legal_weights(row: np.ndarray, legal: Sequence[int]) -> np.ndarray:
    """One position's probabilities over its legal actions, in their order, as float64.

    `row` is the policy head's logits at the action slots. A legal index outside the row, or whose logit is not
    finite, gets no mass; the softmax subtracts the finite maximum; a position with no finite score on any legal
    action gets a uniform prior. This is the per-row code `combat_prior.PolicyNet.weights` ran before it was lifted
    here, unchanged.
    """
    index = np.asarray(legal, dtype=np.int64)
    scores = np.full(len(index), -np.inf)
    inside = index < row.shape[0]
    scores[inside] = row[index[inside]]
    finite = np.isfinite(scores)
    if not finite.any():
        return np.full(len(index), 1.0 / max(1, len(index)))
    scores = np.where(finite, scores, -np.inf)
    weights = np.exp(scores - scores[finite].max())
    return weights / weights.sum()


def card_table_sha256() -> str:
    """sha256 over the card attribute table's bytes (and shape) and its vocabulary size.

    `vocab_sha256` does not cover the table the network composes cards from; two builds with the same vocabulary but
    another table would read other cards than they learned. Imports `combat_tokens` here, not at module load: that
    loads the emulator DLL (`STS2_LIB_PATH`).
    """
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from combat_tokens import card_attribute_table  # noqa: PLC0415

    table, vocab_size = card_attribute_table()
    table = np.ascontiguousarray(table)
    digest = hashlib.sha256()
    digest.update(table.tobytes())
    digest.update(struct.pack("<qqq", table.shape[0], table.shape[1], int(vocab_size)))
    return digest.hexdigest()
