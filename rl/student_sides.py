"""The side blocks the student's loader can take, one flag per block, default off (D59 lane D, step 5; `docs/rl/HUMAN_PRIOR.md` "v4").

The teacher's shards hold, beside `rows.npz` (the judgement's 1,863-wide input of each option's afterstate), side files written for the student and the prior to read: the map grid
(`rows_map.npz`, block M), the state gaps (`rows_gaps.npz`, block G) and the information-parity blocks (`rows_parity.npz`: relic effect tags, deck roles, event payoffs, potion effect tags). D58 addendum 23
parked M and G out of the student's input; the parity blocks are new. This module is the one place that knows the blocks, their files, widths and dtypes, so a fit can take any of them
behind **one flag each**, none by default (the lab's recipe is unchanged):

    flags = student_sides.parse_flags(args.side_block)             # a tuple of block names in the blocks' own order; () = none
    sides = student_sides.read_sides(shard_dir, flags, n_rows)      # {name: array aligned with rows.npz}, each validated, a missing or wrong file refused
    cols  = student_sides.columns(student_sides.select(sides, idx), flags)   # float32 (len(idx), width(flags)), or None for no flags
    w1    = student_sides.widen_first_layer(w1, student_sides.width(flags))  # zero columns: the net starts as the plain one

`run_student.read_shard` already reads M and G; the one-line hook (a call to `read_sides` where it builds a shard's dict, and the net's first layer widened by `width(flags)`) is landed
last and through the orchestrator, because lane B edits that file (BRIEF-lane-D, rules). Nothing here fits a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

import parity_blocks as pb


class SideError(Exception):
    """A side block a flag asks for that the shard cannot give: a missing file, a wrong shape, dtype or version, a number that is not finite."""


@dataclass(frozen=True)
class SideBlock:
    name: str
    file: str  # the shard's side file
    key: str  # the array in it
    width: int
    dtype: str  # the dtype the file holds
    version_key: str = ""  # the manifest key naming the block's layout version ("" none)
    version: int | None = None
    flag_key: str = ""  # the manifest key that says the file is complete ("" none)


# the order here is the order of the columns `columns` builds
BLOCKS: dict[str, SideBlock] = {
    "grid": SideBlock("grid", "rows_map.npz", "grid", 17 * 7 * 13, "int8", flag_key="map_grid"),
    "gaps": SideBlock("gaps", "rows_gaps.npz", "gaps", 36, "float32", "gaps_version", 1, "state_gaps"),
    "parity-relics": SideBlock("parity-relics", pb.SHARD_ROWS_PARITY, "relics", pb.WIDTHS["relics"], "float32", "parity_version", pb.PARITY_VERSION, "parity"),
    "parity-roles": SideBlock("parity-roles", pb.SHARD_ROWS_PARITY, "roles", pb.WIDTHS["roles"], "float32", "parity_version", pb.PARITY_VERSION, "parity"),
    "parity-events": SideBlock("parity-events", pb.SHARD_ROWS_PARITY, "events", pb.WIDTHS["events"], "float32", "parity_version", pb.PARITY_VERSION, "parity"),
    "parity-potions": SideBlock("parity-potions", pb.SHARD_ROWS_PARITY, "potions", pb.WIDTHS["potions"], "float32", "parity_version", pb.PARITY_VERSION, "parity"),
}
DEFAULT_FLAGS: tuple[str, ...] = ()


def parse_flags(spec: str | Iterable[str] | None) -> tuple[str, ...]:
    """Block names from a comma list or a list of names, validated and in the blocks' own order (a name twice counts once); None or empty is no block."""
    if spec is None:
        return ()
    names = [x.strip() for x in spec.split(",")] if isinstance(spec, str) else [str(x).strip() for x in spec]
    names = [x for x in names if x]
    unknown = [x for x in names if x not in BLOCKS]
    if unknown:
        raise ValueError(f"unknown side block {unknown[0]!r}: one of {', '.join(BLOCKS)}")
    return tuple(name for name in BLOCKS if name in names)


def add_arguments(parser) -> None:  # noqa: ANN001
    """`--side-block NAME` (repeatable or a comma list), default none: a driver adds it to its parser and passes `parse_flags(args.side_block)`."""
    parser.add_argument("--side-block", action="append", default=[], metavar="NAME",
                        help=f"a side block to give the student's input (one of {', '.join(BLOCKS)}); default none")


def width(flags: Sequence[str]) -> int:
    return sum(BLOCKS[f].width for f in parse_flags(list(flags)))


def _manifest(d: Path) -> dict:
    try:
        doc = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SideError(f"{d.name}: no readable manifest.json ({e})") from e
    if not isinstance(doc, dict):
        raise SideError(f"{d.name}: manifest.json is not an object")
    return doc


def read_sides(d: Path | str, flags: Sequence[str], n_rows: int) -> dict[str, np.ndarray]:
    """The blocks named by `flags` for the shard folder `d`, each aligned with its rows.npz (`n_rows` rows). No flag: `{}` (nothing is read). A block whose file is missing, whose manifest
    does not say it is there at this layout version, or whose array has another shape, dtype or a number that is not finite is `SideError`."""
    flags = parse_flags(list(flags))
    if not flags:
        return {}
    d = Path(d)
    manifest = _manifest(d)
    out: dict[str, np.ndarray] = {}
    files: dict[str, Mapping[str, np.ndarray]] = {}
    for name in flags:
        b = BLOCKS[name]
        if not (d / b.file).is_file():
            raise SideError(f"{d.name}: side block {name!r} needs {b.file}, which the shard lacks (run_teacher.py reencode / parity_reencode.py adds it)")
        if b.flag_key and not manifest.get(b.flag_key):
            raise SideError(f"{d.name}: the manifest does not say {b.flag_key} for {b.file}")
        if b.version_key and manifest.get(b.version_key) not in (None, b.version):
            raise SideError(f"{d.name}: {b.file} is layout version {manifest.get(b.version_key)}, this build reads version {b.version}")
        if b.file not in files:
            with np.load(d / b.file, allow_pickle=False) as z:
                files[b.file] = {k: z[k] for k in z.files}
        arrays = files[b.file]
        if b.key not in arrays:
            raise SideError(f"{d.name}: {b.file} lacks {b.key}")
        a = arrays[b.key]
        if a.ndim != 2 or a.shape != (n_rows, b.width) or a.dtype != np.dtype(b.dtype):
            raise SideError(f"{d.name}: {name} is {a.shape} {a.dtype}, not ({n_rows}, {b.width}) {b.dtype} (one row per row of rows.npz)")
        if a.dtype.kind == "f" and len(a) and not np.isfinite(a).all():
            raise SideError(f"{d.name}: {name} holds a number that is not finite")
        out[name] = a
    return out


def read_known(d: Path | str, n_rows: int) -> np.ndarray:
    """The parity file's `known` mask (bool, one per row): False where a row's blocks could not be built (the run could not be replayed, or the rows built again differed)."""
    d = Path(d)
    f = d / pb.SHARD_ROWS_PARITY
    if not f.is_file():
        raise SideError(f"{d.name}: no {pb.SHARD_ROWS_PARITY}")
    with np.load(f, allow_pickle=False) as z:
        if "known" not in z.files:
            raise SideError(f"{d.name}: {pb.SHARD_ROWS_PARITY} lacks known")
        k = z["known"]
    if k.shape != (n_rows,) or k.dtype != np.bool_:
        raise SideError(f"{d.name}: known is {k.shape} {k.dtype}, not ({n_rows},) bool")
    return k


def select(sides: Mapping[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    """The rows `idx` of every block (the rows a fit's set keeps)."""
    return {k: v[idx] for k, v in sides.items()}


def columns(sides: Mapping[str, np.ndarray], flags: Sequence[str]) -> np.ndarray | None:
    """The blocks named by `flags`, concatenated in the blocks' own order as float32 (n, width(flags)); None for no flag."""
    flags = parse_flags(list(flags))
    if not flags:
        return None
    return np.concatenate([np.asarray(sides[f], dtype=np.float32) for f in flags], axis=1)


def widen_first_layer(weight: np.ndarray, extra: int) -> np.ndarray:
    """A first-layer weight matrix (out, in) with `extra` zero columns appended: a net with the side blocks as input starts exactly as the plain one, as block G's columns did."""
    w = np.asarray(weight)
    return np.concatenate([w, np.zeros((w.shape[0], int(extra)), dtype=w.dtype)], axis=1)
