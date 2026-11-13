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

A prior or student fitted with the map grid (D58 addendum 13; `map_grid: true` in its meta) carries a map tower: two 3x3 convolutions over block M
(`features.map_grid`, the whole act map: 17 rows of 7 columns, 13 numbers a cell), a linear layer to 64 numbers, concatenated to the present input
before `hidden1`. `PriorScorer` runs it in numpy (`map_tower`); `option_rows(..., with_grid=True)` returns each option's grid beside the rows (a second
array, never a wider row) and `row_logits(rows, grid)` / `log_probs` score with it. A file without a tower scores as it always did.

A prior or student fitted with the state gaps (D58 addenda 14 and 16; `state_gaps: true` in its meta) reads block G (`features.gaps_block`: relic counters, the
"?" rooms' state, the merchant's removals, three known bits; `GAPS_SIZE` numbers) as the last columns of its first layer, after the map tower's when there is one.
`option_gaps(view, options)` returns each option's block G (a third array beside the rows and the grids), `row_logits(rows, grid, gaps)` / `log_probs` score with
it; the view's `gaps` come from the engine (`features.gaps_from_engine`) and a view without them (the live coach's) writes zeros and known bits 0.

The weights are not fitted here and nothing here imports torch or the fit module. `sha256` of the file names the prior in a candidate's
macro (`imitation_chooser.macro_name`).

The run student `pi_s` (`rl/run_student.py`) is the same net refitted on the run teacher's labels, in the same file format; its file also
says which decision kinds it was fitted for (`meta["student"]["kinds"]`) and what it started from. `StudentScorer` / `load_student` read
such a file with the same scoring as `PriorScorer` (a plain prior's file is refused there: it names no kinds), `write_student` writes one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# the counts blocks of a row: E (deck cards), F (relics), G (potions); the fit's embeddings read them in this order
E_SIZE, F_SIZE, G_SIZE = 1154, 296, 63
DENSE_WIDTH = 350
_KEYS = ("embed.deck.weight", "embed.relics.weight", "embed.potions.weight", "hidden1.weight", "hidden1.bias", "hidden2.weight",
         "hidden2.bias", "score.weight", "score.bias")
# the map tower's arrays (`human_prior.MapTower`'s torch names): two 3x3 convolutions (padding 1) and a linear layer to MAP_OUT numbers
MAP_KEYS = ("map.conv1.weight", "map.conv1.bias", "map.conv2.weight", "map.conv2.bias", "map.fc.weight", "map.fc.bias")
MAP_ROWS, MAP_COLS, MAP_CHANNELS = 17, 7, 13  # features.MAP_GRID_SHAPE (a test holds them equal)
MAP_CONV1, MAP_CONV2, MAP_OUT = 16, 32, 64
MAP_GRID_SIZE = MAP_ROWS * MAP_COLS * MAP_CHANNELS
GAPS_SIZE = 36  # features.GAPS_SIZE: block G of one row, float32 (a test holds them equal)


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


STUDENT_META = "student"  # the key of a student file's meta block
STUDENT_FORMAT = 1


def log_softmax(x: np.ndarray) -> np.ndarray:
    """Log-probabilities of a vector of logits (a softmax over the options of one decision)."""
    x = np.asarray(x, dtype=np.float64)
    top = x.max()
    return x - (top + np.log(np.exp(x - top).sum()))


def _conv3x3(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """A 3x3 convolution with padding 1 (torch's `Conv2d(c, o, 3, padding=1)`, a cross-correlation): `x` (n, c, h, w) -> (n, o, h, w)."""
    n, c, h, w = x.shape
    o = weight.shape[0]
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    patches = np.stack([padded[:, :, i:i + h, j:j + w] for i in range(3) for j in range(3)], axis=2).reshape(n, c * 9, h * w)
    out = np.matmul(weight.reshape(o, c * 9), patches) + bias[None, :, None]
    return out.reshape(n, o, h, w)


def map_tower(p: dict[str, np.ndarray], grid: np.ndarray, chunk: int = 512) -> np.ndarray:
    """The map tower on block M grids, `grid` (n, MAP_GRID_SIZE) of 0 and 1: two 3x3 convolutions (16 then 32 channels, ReLU), the flattened
    (channel, row, column) maps through a linear layer to 64 numbers, ReLU: (n, 64). Equal to the torch tower to float rounding. Grids that
    repeat (the options of one screen share a position unless they are a map step) are computed once."""
    g = np.ascontiguousarray(np.asarray(grid, dtype=np.int8)).reshape(-1, MAP_GRID_SIZE)
    if len(g) > 1:
        keys = g.view(np.dtype((np.void, g.shape[1])))[:, 0]
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        if len(first) < len(g):
            return map_tower(p, g[first], chunk)[inverse.reshape(-1)]
    out = np.zeros((len(g), MAP_OUT), dtype=np.float32)
    for s in range(0, len(g), chunk):
        x = g[s:s + chunk].astype(np.float32).reshape(-1, MAP_ROWS, MAP_COLS, MAP_CHANNELS).transpose(0, 3, 1, 2)
        x = np.maximum(_conv3x3(x, p["map.conv1.weight"], p["map.conv1.bias"]), 0.0)
        x = np.maximum(_conv3x3(x, p["map.conv2.weight"], p["map.conv2.bias"]), 0.0)
        out[s:s + chunk] = np.maximum(x.reshape(len(x), -1) @ p["map.fc.weight"].T + p["map.fc.bias"], 0.0)
    return out


def map_keys_in(keys: Any) -> tuple[str, ...]:
    """The map tower's arrays among `keys`: all of `MAP_KEYS` or none (a file with some is refused)."""
    have = tuple(k for k in MAP_KEYS if k in keys)
    if have and len(have) != len(MAP_KEYS):
        raise ValueError(f"the weights hold part of a map tower: {sorted(set(MAP_KEYS) - set(have))} are missing")
    return have


class PriorScorer:
    """`pi_h`: one logit per option's afterstate row (with the map tower, one per row and its map grid)."""

    def __init__(self, params: dict[str, np.ndarray], meta: dict[str, Any] | None = None, sha256: str = ""):
        missing = [k for k in _KEYS if k not in params]
        if missing:
            raise ValueError(f"the prior's weights lack {missing}")
        self.p = {k: np.asarray(params[k], dtype=np.float32) for k in _KEYS}
        self.map_grid = bool(map_keys_in(params))
        self.p.update({k: np.asarray(params[k], dtype=np.float32) for k in MAP_KEYS if self.map_grid})
        self.meta = meta or {}
        if bool(self.meta.get("map_grid")) != self.map_grid:
            raise ValueError(f"the file's meta says map_grid={bool(self.meta.get('map_grid'))} but its weights {'hold' if self.map_grid else 'lack'} the map tower")
        self.state_gaps = bool(self.meta.get("state_gaps"))  # block G's columns close the first layer's input
        self.side_blocks: tuple[str, ...] = _side_flags(self.meta.get("side_blocks"))  # the student loader's side blocks (rl/student_sides.py), after block G's columns
        self.sha256 = sha256
        want = (self.p["embed.deck.weight"].shape[1], self.p["embed.relics.weight"].shape[1], self.p["embed.potions.weight"].shape[1])
        if want != (E_SIZE, F_SIZE, G_SIZE):
            raise ValueError(f"the prior reads counts of widths {want}, this build has {(E_SIZE, F_SIZE, G_SIZE)}")
        width = DENSE_WIDTH + sum(self.p[k].shape[0] for k in ("embed.deck.weight", "embed.relics.weight", "embed.potions.weight"))
        if self.map_grid:
            shapes = {"map.conv1.weight": (MAP_CONV1, MAP_CHANNELS, 3, 3), "map.conv1.bias": (MAP_CONV1,), "map.conv2.weight": (MAP_CONV2, MAP_CONV1, 3, 3),
                      "map.conv2.bias": (MAP_CONV2,), "map.fc.weight": (MAP_OUT, MAP_CONV2 * MAP_ROWS * MAP_COLS), "map.fc.bias": (MAP_OUT,)}
            wrong = {k: self.p[k].shape for k, want in shapes.items() if self.p[k].shape != want}
            if wrong:
                raise ValueError(f"the map tower's arrays have the wrong shapes (this build reads a {MAP_ROWS}x{MAP_COLS}x{MAP_CHANNELS} grid): {wrong}")
            width += MAP_OUT
        if self.state_gaps:
            if int(self.meta.get("gaps_version", 1)) != 1:
                raise ValueError(f"the prior reads block G version {self.meta.get('gaps_version')}, this build writes version 1")
            width += GAPS_SIZE
        if self.side_blocks:
            import student_sides  # noqa: PLC0415

            width += student_sides.width(self.side_blocks)
        if self.p["hidden1.weight"].shape[1] != width:
            raise ValueError(f"the prior's first layer reads {self.p['hidden1.weight'].shape[1]} numbers, the judgement's input is {width}")

    @classmethod
    def load(cls, path: Path | str) -> PriorScorer:
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            params = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        return cls(params, meta, file_sha256(path))

    def logits(self, dense: np.ndarray, counts: np.ndarray, grid: np.ndarray | None = None, gaps: np.ndarray | None = None,
               sides: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        """The logit of each row of the judgement's assembled inputs (`models.value_inputs`): `dense` (n, 350), `counts` (n, 1513). A prior with
        the map tower needs each row's block M `grid` (n, MAP_GRID_SIZE) and refuses to score without it; one without ignores `grid`. A prior with the
        state gaps likewise needs each row's block G `gaps` (n, GAPS_SIZE) and ignores it otherwise. A prior with side blocks (`side_blocks` in its meta, the student
        loader's: `rl/student_sides.py`) reads their columns after block G's: `sides` gives each block (name -> (n, width)); the held blocks it makes from `counts` itself
        when they are not given, a block with a file (the grid, the gaps, the parity blocks) it refuses to score without."""
        p = self.p
        counts = np.asarray(counts, dtype=np.float32)
        emb = np.concatenate([counts[:, :E_SIZE] @ p["embed.deck.weight"].T,
                              counts[:, E_SIZE:E_SIZE + F_SIZE] @ p["embed.relics.weight"].T,
                              counts[:, E_SIZE + F_SIZE:] @ p["embed.potions.weight"].T], axis=1)
        parts = [np.asarray(dense, dtype=np.float32), emb]
        if self.map_grid:
            if grid is None or len(grid) != len(counts):
                raise ValueError("this prior has a map tower: score its rows with their block M grids (option_rows(..., with_grid=True))")
            parts.append(map_tower(p, grid))
        if self.state_gaps:
            if gaps is None or len(gaps) != len(counts):
                raise ValueError("this prior reads the state gaps: score its rows with their block G (option_gaps)")
            parts.append(np.asarray(gaps, dtype=np.float32))
        if self.side_blocks:
            import student_sides  # noqa: PLC0415

            given = dict(sides or {})
            for name, block in student_sides.derive(counts, [f for f in self.side_blocks if f not in given]).items():
                given[name] = block
            lacking = [f for f in self.side_blocks if f not in given]
            if lacking:
                raise ValueError(f"this scorer reads the side blocks {list(self.side_blocks)}: {lacking[0]} was not given (option_sides builds them at play)")
            wrong = [f for f in self.side_blocks if len(given[f]) != len(counts)]
            if wrong:
                raise ValueError(f"the side block {wrong[0]} has {len(given[wrong[0]])} rows for {len(counts)} rows")
            parts.append(student_sides.columns(given, self.side_blocks))
        h = np.maximum(np.concatenate(parts, axis=1) @ p["hidden1.weight"].T + p["hidden1.bias"], 0.0)
        h = np.maximum(h @ p["hidden2.weight"].T + p["hidden2.bias"], 0.0)
        return (h @ p["score.weight"].T + p["score.bias"])[:, 0]

    def row_logits(self, rows: np.ndarray, grid: np.ndarray | None = None, gaps: np.ndarray | None = None,
                   sides: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        """The logits of rows of the judgement's 1,863-wide vector (the base vector and H): the shared input of V and the prior. `grid`: the
        rows' block M grids, which a prior with the map tower needs; `gaps`: their block G, which a prior with the state gaps needs; `sides`: the side blocks a
        student with `side_blocks` reads (the held blocks it makes from the rows)."""
        from run_train import models  # noqa: PLC0415 -- numpy only

        dense, counts = models.value_inputs(rows)
        return self.logits(dense, counts, grid, gaps, sides)

    def log_probs(self, view: Any, options: Sequence[Any], nets: Any) -> np.ndarray:
        """Log-probabilities of a decision's options: the softmax over their logits."""
        rows = option_rows(nets, view, options)
        grid = option_grids(view, options) if self.map_grid else None
        gaps = option_gaps(view, options) if self.state_gaps else None
        sides = option_sides(view, options, self.side_blocks, rows) if self.side_blocks else None
        return log_softmax(self.row_logits(rows, grid, gaps, sides))


class StudentScorer(PriorScorer):
    """`pi_s`: a `PriorScorer` (one logit per option's afterstate row) whose file names the kinds it was fitted for."""

    def __init__(self, params: dict[str, np.ndarray], meta: dict[str, Any] | None = None, sha256: str = ""):
        super().__init__(params, meta, sha256)
        block = self.meta.get(STUDENT_META)
        if not isinstance(block, dict) or block.get("format") != STUDENT_FORMAT or not block.get("kinds"):
            raise ValueError(f"not a run student file: its meta has no `{STUDENT_META}` block of format {STUDENT_FORMAT} naming kinds")
        self.kinds: tuple[str, ...] = tuple(str(k) for k in block["kinds"])
        self.prior_sha256: str = str(block.get("prior_sha256") or "")

    def has(self, kind: str) -> bool:
        return kind in self.kinds


@lru_cache(maxsize=4)
def _load_student_cached(path: str, mtime_ns: int, size: int) -> StudentScorer:
    return StudentScorer.load(path)


def load_student(path: Path | str) -> StudentScorer:
    """`StudentScorer.load`, once per file version in this process."""
    p = Path(path)
    st = p.stat()
    return _load_student_cached(str(p.resolve()), st.st_mtime_ns, st.st_size)


def student_kinds_of(path: Path | str) -> tuple[str, ...]:
    """The kinds a student file was fitted for, read from its meta alone (no weights loaded); ValueError for a file that is not a student's."""
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    block = meta.get(STUDENT_META)
    if not isinstance(block, dict) or block.get("format") != STUDENT_FORMAT or not block.get("kinds"):
        raise ValueError(f"{Path(path).name} is not a run student file")
    return tuple(str(k) for k in block["kinds"])


def write_student(path: Path | str, params: dict[str, np.ndarray], *, kinds: Sequence[str], prior_sha256: str, fit: dict[str, Any] | None = None,
                  extra_meta: dict[str, Any] | None = None, state_gaps: bool = False, side_blocks: Sequence[str] = ()) -> str:
    """Write a student file (the prior's nine arrays and a meta block naming the kinds, the prior it started from and the fit's record),
    tmp then rename. Returns the file's sha256."""
    import os  # noqa: PLC0415

    missing = [k for k in _KEYS if k not in params]
    if missing:
        raise ValueError(f"the student's weights lack {missing}")
    if not kinds:
        raise ValueError("a student file names at least one kind")
    tower = map_keys_in(params)  # a student fitted with the map grid carries the tower's weights and says so (`map_grid: true`)
    meta = {**(extra_meta or {}), STUDENT_META: {"format": STUDENT_FORMAT, "kinds": [str(k) for k in kinds], "prior_sha256": prior_sha256,
                                                  "fit": fit or {}}}
    if tower:
        meta["map_grid"] = True
    if state_gaps:  # the first layer's last GAPS_SIZE columns read block G (no array of its own)
        meta["state_gaps"] = True
        meta["gaps_version"] = 1
    if side_blocks:  # the student loader's side blocks (`rl/student_sides.py`): the first layer's last columns, in the blocks' own order, after block G's
        meta["side_blocks"] = list(_side_flags(side_blocks))
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, meta=np.array(json.dumps(meta, sort_keys=True)), **{k: np.asarray(params[k], dtype=np.float32) for k in (*_KEYS, *tower)})
    os.replace(tmp, path)
    return file_sha256(path)


def option_rows(nets: Any, view: Any, options: Sequence[Any], with_grid: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """The judgement's input rows of a decision's options, (n, 1863): each afterstate's base vector with the fight block H that
    `nets.fight` (the judgement's FightNet) gives it. `nets.value.floor_probs(rows)` values the same rows. With `with_grid` it returns
    `(rows, grid)`, `grid` the options' block M, int8 (n, MAP_GRID_SIZE) (`features.afterstate_map_grid`): a second array, the rows themselves
    are the same bytes either way."""
    from run_train import features  # noqa: PLC0415 -- numpy only

    if nets.base_size != features.BASE_SIZE:
        raise ValueError("the prior reads feature version 2 rows; this judgement's nets read an older vector")
    bases = np.stack([features.encode_afterstate(view, option) for option in options])
    block = nets.fight.summary(bases, features.fight_queries(view))
    rows = np.concatenate([bases, block], axis=1)
    if not with_grid:
        return rows
    return rows, option_grids(view, options)


def option_grids(view: Any, options: Sequence[Any]) -> np.ndarray:
    """Block M of each option's afterstate, int8 (n, MAP_GRID_SIZE): the view's whole-act map with the position of the state after the option
    (a map step moves it; every other option leaves it)."""
    from run_train import features  # noqa: PLC0415 -- numpy only

    if features.MAP_GRID_SIZE != MAP_GRID_SIZE:
        raise ValueError("the map tower's grid and the features' block M differ in size")
    if not len(options):
        return np.zeros((0, MAP_GRID_SIZE), dtype=np.int8)
    return np.stack([features.afterstate_map_grid(view, option) for option in options])


def _side_flags(spec: Any) -> tuple[str, ...]:
    """The side blocks a meta names, validated and in the blocks' own order (`student_sides.parse_flags`); () for none."""
    if not spec:
        return ()
    import student_sides  # noqa: PLC0415

    return student_sides.parse_flags([str(x) for x in spec])


def option_sides(view: Any, options: Sequence[Any], flags: Sequence[str], rows: np.ndarray) -> dict[str, np.ndarray]:
    """The student's side blocks of each option's afterstate at play, name -> float32 (n, width): the map grid (`option_grids`), the state gaps (`option_gaps`) and the full parity
    blocks (`parity_blocks.option_parity`) from the view and the options; the held blocks from `rows` (the options' judgement rows: their counts are the afterstates')."""
    import student_sides  # noqa: PLC0415
    from run_train import models  # noqa: PLC0415 -- numpy only

    flags = student_sides.parse_flags(list(flags))
    out: dict[str, np.ndarray] = {}
    if any(student_sides.BLOCKS[f].derived for f in flags):
        out.update(student_sides.derive(models.value_inputs(np.asarray(rows))[1], flags))
    if "grid" in flags:
        out["grid"] = option_grids(view, options).astype(np.float32)
    if "gaps" in flags:
        out["gaps"] = option_gaps(view, options)
    full = [f for f in flags if student_sides.BLOCKS[f].file == "rows_parity.npz"]
    if full:
        import parity_blocks  # noqa: PLC0415

        built = parity_blocks.option_parity(view, options)
        for f in full:
            out[f] = built[student_sides.BLOCKS[f].key]
    return out


def option_gaps(view: Any, options: Sequence[Any]) -> np.ndarray:
    """Block G of each option's afterstate, float32 (n, GAPS_SIZE) (`features.afterstate_gaps`): the view's relic counters, "?" rooms and removals, with what
    each option changes of them (a merchant's removal, a relic's starting counter)."""
    from run_train import features  # noqa: PLC0415 -- numpy only

    if features.GAPS_SIZE != GAPS_SIZE:
        raise ValueError("the prior's block G and the features' differ in size")
    if not len(options):
        return np.zeros((0, GAPS_SIZE), dtype=np.float32)
    return np.stack([features.afterstate_gaps(view, option) for option in options])
