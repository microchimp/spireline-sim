"""Synthetic paired arms for the keep test v2's tests (tools/lab/keep_v2.py): no saved row is committed, only counts.

An arm is described by groups `(count, candidate level, incumbent level)`, a level being how deep the run got: 0 died before the Act 1 boss,
1 beat it (died in Act 2), 2 beat the Act 2 boss (died in Act 3), 3 also beat the first Act 3 boss (A10 has two; floor 48, which the
floor cuts miss), 4 won. The rows carry the fields the rule reads (`floor`, `win`, `crash`, `boss_damage`, `boss_damage_acts`).
"""

from __future__ import annotations

import numpy as np

# `candidate-imit-l05` against generation 11 on E12-2026-11-05-r3 (2,000 paired seeds), rebuilt from the block's counts:
# Act 1 boss 709 both / 256 candidate only / 334 incumbent only / 701 neither; Act 2 boss 10 both / 97 candidate only / 39 incumbent only;
# first Act 3 boss 6 (candidate) and 3 (incumbent), all among those who beat the Act 2 boss. deep at the Act 2 boss: +3.05 per 100.
L05_SPEC = (
    (10, 2, 2), (91, 2, 1), (6, 3, 1), (36, 1, 2), (3, 1, 3), (563, 1, 1),  # both beat the Act 1 boss (709)
    (256, 1, 0), (334, 0, 1), (701, 0, 0),
)
FLOOR_PARAMS = {0: (15.8, 2.7, 3, 17), 1: (29.2, 4.1, 18, 33), 2: (42.1, 3.7, 34, 47)}  # level -> mean, SD, lowest, highest floor


def floor_of(level: int, rng: np.random.Generator) -> int:
    if level == 3:
        return 48
    if level == 4:
        return 51
    mean, sd, lo, hi = FLOOR_PARAMS[level]
    return int(np.clip(np.rint(rng.normal(mean, sd)), lo, hi))


def run_row(seed: str, level: int, rng: np.random.Generator) -> dict:
    """One run that got `level` deep."""
    acts: list[list] = []
    if level == 0:
        acts = [[0, round(float(rng.uniform(0.2, 0.8)), 4)]]
    elif level == 1:
        acts = [[0, 1.0], [1, round(float(rng.uniform(0.2, 0.8)), 4)]]
    elif level == 2:
        acts = [[0, 1.0], [1, 1.0], [2, round(float(rng.uniform(0.2, 0.8)), 4)]]
    else:  # 3: the first Act 3 boss beaten, the second not; 4: the win
        acts = [[0, 1.0], [1, 1.0], [2, 1.0]]
    return {"seed": seed, "floor": floor_of(level, rng), "win": level == 4, "end": "win" if level == 4 else "death", "decisions": 10,
            "wall": 1.0, "crash": False, "boss_damage": acts[0][1], "boss_damage_acts": acts}


def crash_row(seed: str, lost: bool = False) -> dict:
    """A seed whose worker crashed (`crash`) or was lost (`lost`): floor 0, no win, as `gen0` writes them."""
    if lost:
        return {"seed": seed, "floor": 0, "win": False, "end": "lost", "decisions": 0, "wall": 1.0, "lost": True, "boss_damage": 0.0}
    return {"seed": seed, "floor": 0, "win": False, "end": "crash", "decisions": 0, "wall": 1.0, "crash": True}


def arms(spec=L05_SPEC, seed: int = 7, prefix: str = "K") -> tuple[list[str], list[dict], list[dict]]:
    """(seeds, candidate rows, incumbent rows) for the groups of `spec`, the pairs shuffled so the groups are not in blocks."""
    rng = np.random.default_rng(seed)
    pairs = [(c, b) for count, c, b in spec for _ in range(count)]
    order = rng.permutation(len(pairs))
    seeds = [f"{prefix}{i:09d}" for i in range(len(pairs))]
    cand = [run_row(s, pairs[i][0], rng) for s, i in zip(seeds, order, strict=True)]
    base = [run_row(s, pairs[i][1], rng) for s, i in zip(seeds, order, strict=True)]
    return seeds, cand, base


def with_bad(rows: list[dict], k: int, lost: bool = False) -> list[dict]:
    """`rows` with the first `k` replaced by crashed (or lost) seeds."""
    return [crash_row(r["seed"], lost) for r in rows[:k]] + list(rows[k:])


def std_diffs(mean: float, sd: float, n: int, seed: int = 3) -> np.ndarray:
    """`n` values whose sample mean and sample SD (n - 1) are exactly `mean` and `sd`."""
    x = np.random.default_rng(seed).normal(0, 1, n)
    return (x - x.mean()) / x.std(ddof=1) * sd + mean
