"""The human deck prior `pi_h` in numpy, for play (milestone 4 of the 2026-11-05 strategy memo, section 3.3).

`rl/human_prior.py` fits `pi_h` on the GPU and exports its weights (`weights.npz`). Block and self-play processes run on the CPU
without torch, so this module scores a decision's options from that file with numpy alone, the way `run_train.policy` scores the run
judgement `V`:

    prior = PriorScorer.load("human_prior.npz")
    logp  = prior.log_probs(view, options, nets)        # (n,) log-probabilities over the options, a softmax of the logits

`nets` is the run judgement's `NumpyNets` (its FightNet fills the fight block H, the value net reads the same row). The row of an
option is the judgement's own: `features.encode_afterstate(view, option)` (1,855 numbers; at play the view holds the true map, so the
map-ahead block and the distance to the boss are the run's own, not a donor's) with H appended, then `models.value_inputs` splits it
into the 350 dense numbers and the counts the net reads. `PriorNet` of the fit is `RunValueNet`'s embeddings and two 256-wide layers
with one logit in place of the floor and end-HP heads; `logits` is that forward pass weight for weight (dropout off).

The weights are not fitted here and nothing here imports torch or the fit module. `sha256` of the file names the prior in a candidate's
macro (`imitation_chooser.macro_name`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# the counts blocks of a row: E (deck cards), F (relics), G (potions); the fit's embeddings read them in this order
E_SIZE, F_SIZE, G_SIZE = 1154, 296, 63
DENSE_WIDTH = 350
_KEYS = ("embed.deck.weight", "embed.relics.weight", "embed.potions.weight", "hidden1.weight", "hidden1.bias", "hidden2.weight",
         "hidden2.bias", "score.weight", "score.bias")


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=4)
def _load_cached(path: str, mtime_ns: int, size: int) -> PriorScorer:
    return PriorScorer.load(path)


def load(path: Path | str) -> PriorScorer:
    """`PriorScorer.load`, once per file version in this process (a policy is built for every run of a block)."""
    p = Path(path)
    st = p.stat()
    return _load_cached(str(p.resolve()), st.st_mtime_ns, st.st_size)


def log_softmax(x: np.ndarray) -> np.ndarray:
    """Log-probabilities of a vector of logits (a softmax over the options of one decision)."""
    x = np.asarray(x, dtype=np.float64)
    top = x.max()
    return x - (top + np.log(np.exp(x - top).sum()))


class PriorScorer:
    """`pi_h`: one logit per option's afterstate row."""

    def __init__(self, params: dict[str, np.ndarray], meta: dict[str, Any] | None = None, sha256: str = ""):
        missing = [k for k in _KEYS if k not in params]
        if missing:
            raise ValueError(f"the prior's weights lack {missing}")
        self.p = {k: np.asarray(params[k], dtype=np.float32) for k in _KEYS}
        self.meta = meta or {}
        self.sha256 = sha256
        want = (self.p["embed.deck.weight"].shape[1], self.p["embed.relics.weight"].shape[1], self.p["embed.potions.weight"].shape[1])
        if want != (E_SIZE, F_SIZE, G_SIZE):
            raise ValueError(f"the prior reads counts of widths {want}, this build has {(E_SIZE, F_SIZE, G_SIZE)}")
        width = DENSE_WIDTH + sum(self.p[k].shape[0] for k in ("embed.deck.weight", "embed.relics.weight", "embed.potions.weight"))
        if self.p["hidden1.weight"].shape[1] != width:
            raise ValueError(f"the prior's first layer reads {self.p['hidden1.weight'].shape[1]} numbers, the judgement's input is {width}")

    @classmethod
    def load(cls, path: Path | str) -> PriorScorer:
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            params = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        return cls(params, meta, file_sha256(path))

    def logits(self, dense: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """The logit of each row of the judgement's assembled inputs (`models.value_inputs`): `dense` (n, 350), `counts` (n, 1513)."""
        p = self.p
        counts = np.asarray(counts, dtype=np.float32)
        emb = np.concatenate([counts[:, :E_SIZE] @ p["embed.deck.weight"].T,
                              counts[:, E_SIZE:E_SIZE + F_SIZE] @ p["embed.relics.weight"].T,
                              counts[:, E_SIZE + F_SIZE:] @ p["embed.potions.weight"].T], axis=1)
        h = np.maximum(np.concatenate([np.asarray(dense, dtype=np.float32), emb], axis=1) @ p["hidden1.weight"].T + p["hidden1.bias"], 0.0)
        h = np.maximum(h @ p["hidden2.weight"].T + p["hidden2.bias"], 0.0)
        return (h @ p["score.weight"].T + p["score.bias"])[:, 0]

    def row_logits(self, rows: np.ndarray) -> np.ndarray:
        """The logits of rows of the judgement's 1,863-wide vector (the base vector and H): the shared input of V and the prior."""
        from run_train import models  # noqa: PLC0415 -- numpy only

        dense, counts = models.value_inputs(rows)
        return self.logits(dense, counts)

    def log_probs(self, view: Any, options: Sequence[Any], nets: Any) -> np.ndarray:
        """Log-probabilities of a decision's options: the softmax over their logits."""
        return log_softmax(self.row_logits(option_rows(nets, view, options)))


def option_rows(nets: Any, view: Any, options: Sequence[Any]) -> np.ndarray:
    """The judgement's input rows of a decision's options, (n, 1863): each afterstate's base vector with the fight block H that
    `nets.fight` (the judgement's FightNet) gives it. `nets.value.floor_probs(rows)` values the same rows."""
    from run_train import features  # noqa: PLC0415 -- numpy only

    if nets.base_size != features.BASE_SIZE:
        raise ValueError("the prior reads feature version 2 rows; this judgement's nets read an older vector")
    bases = np.stack([features.encode_afterstate(view, option) for option in options])
    block = nets.fight.summary(bases, features.fight_queries(view))
    return np.concatenate([bases, block], axis=1)
