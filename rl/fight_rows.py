"""The fights inside plain runs as fight-trainer rows (D65, lane R): the row format's constants, the grade rule, and the reader that turns a
plain store's combat part into rows of the fight trainer's buffer. Pure: numpy only, no native code.

    run_teacher.py job --kinds none --combat-rows elite-boss ...        # the data step keeps the searched decisions of elite and boss fights
    gym_loop_train.py --root R --base B --extra-rows-root STORE/<gen>   # the fight trainer reads them beside its own shards

**What is recorded.** The run worker keeps, per searched combat decision with two or more moves, `prior_loop`'s row (the root's token
encoding, the legal mask, the search's root visits, the root's mean values, the fight's final HP as the value label, `fights` the run's
index in the part) and three more columns, written only when rows are asked for:

- `fight_kind` (int8): 0 hallway, 1 elite, 2 boss (`KIND_NAMES`; `macro_recipe.FIGHT_KINDS` order);
- `grade` (int16): the stakes multiplier the fight's search ran at (a3-v1@h1e4b16: hallway 1, elite 4, boss 16);
- `grade_sims` (int32): the simulations the config asked for at that decision (30 at a plain play and 120 at a target or a choice
  screen, times the multiplier); `sims` (already there) is what the search spent.

**The grade rule, first version.** The fight loop's own teacher searches at x16 (`fight_gym` arm x16), so a plain row searched at
grade 16 (a boss fight) is a policy and value row; a row searched shallower (an elite at x4, a hallway at x1) is a value row only, its
policy target being a shallower search's than the one the network is trained to imitate. The rule lives in `row_weights` and the weights
are per-row fields (`w_policy`, `w_value`) of the trainer's buffer, so a later rule (a share of the x4 policy, a weight by simulations)
changes this one function or the trainer's `--policy-min-grade`, not the part format.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

KIND_NAMES = ("hallway", "elite", "boss")  # the code of a row's `fight_kind` is its index here (macro_recipe.FIGHT_KINDS: hall, elite, boss)
COMBAT_ROWS = ("none", "elite-boss", "all")  # run_teacher job --combat-rows
KINDS_OF = {"none": (), "elite-boss": ("elite", "boss"), "all": ("hall", "elite", "boss")}  # the search config's fight kinds a mode records
ROW_EXTRA_KEYS = ("fight_kind", "grade", "grade_sims")  # the columns of a part beyond `prior_loop`'s; present only in a part that holds fight rows
GYM_GRADE = 16  # the stakes multiplier of the fight loop's own teacher (fight_gym arm x16): the grade of every row in its shards
POLICY_MIN_GRADE = 16  # a row's policy target is learned from when its grade is at least this
MAX_ACTIONS = 32  # combat_tokens.MAX_ACTIONS (a test keeps them equal)


def row_weights(grade, min_policy_grade: int = POLICY_MIN_GRADE) -> tuple[np.ndarray, np.ndarray]:
    """(policy weight, value weight) per row from its grade: the policy is learned from a row searched at `min_policy_grade` or deeper
    and every row teaches the value (its fight's final HP does not depend on how deep the search was)."""
    g = np.asarray(grade)
    return (g >= int(min_policy_grade)).astype(np.float32), np.ones(g.shape, dtype=np.float32)


def part_to_buffer(blob: dict, *, is_val: Callable[[str], bool], min_policy_grade: int = POLICY_MIN_GRADE) -> dict | None:
    """The rows of a combat part in the fight trainer's shard layout (`gym_loop.samples_arrays` plus `label`, `value_ok`, `val`), with the
    weights of `row_weights` and `src` 1 (a plain run's row), `grade`. `is_val(seed)` says whether a run's rows are held out: the split
    is by the run's seed, the `fights` column being the run's index in the part, so a run's rows are never on both sides. None for a part
    with no rows or whose rows carry no grade (a self-play part: its search ran at no recorded grade)."""
    if "lengths" not in blob or "grade" not in blob or int(blob["lengths"].size) == 0:
        return None
    n = int(blob["lengths"].size)
    seeds = [str(s) for s in blob["seeds"]]
    run_val = np.array([bool(is_val(s)) for s in seeds], dtype=np.int8)
    grade = blob["grade"].astype(np.int16)
    w_policy, w_value = row_weights(grade, min_policy_grade)
    return {
        "types": blob["types"].astype(np.int16), "ids": blob["ids"].astype(np.int32), "owners": blob["owners"].astype(np.int16),
        "feats": blob["feats"].astype(np.float32), "lengths": blob["lengths"].astype(np.int32),
        "visits": blob["visits"].astype(np.float32), "legal": blob["legal"].astype(np.int8),
        "label": blob["labels"].astype(np.int16), "value_ok": np.ones(n, dtype=np.int8), "val": run_val[blob["fights"]],
        "exact": np.zeros(n, dtype=np.int8), "exact_value": np.full(n, np.nan, dtype=np.float32),
        "q_played": np.full(n, np.nan, dtype=np.float32), "solver_best": np.full(n, -1, dtype=np.int16),
        "played": np.full(n, -1, dtype=np.int16), "w_policy": w_policy, "w_value": w_value,
        "src": np.ones(n, dtype=np.int8), "grade": grade,
    }


def part_paths(root: Path | str) -> list[Path]:
    """The combat parts of a plain store: `root` a generation folder (`<store>/<generation>`: `jobs/*/shards/combat-g*-*.npz`) or the store
    itself (`<generation>/jobs/*/shards/...`). A part still being written (`.tmp`) is not one."""
    root = Path(root)
    found = list(root.glob("jobs/*/shards/combat-g*-*.npz")) + list(root.glob("*/jobs/*/shards/combat-g*-*.npz"))
    return sorted(p for p in found if ".tmp" not in p.name)
