"""Training the value models: FightNet first, then V (DESIGN.md 5.2-5.4).

`fit(shards, prev_checkpoint, out, ...)` reads self-play records, keeps the replay buffers of
5.4, fits FightNet on the fights, fills every V row's fight block H with the new FightNet
-- through the numpy inference the policy runs, so training and serving compute H alike --
and then fits V on the decisions. It writes the checkpoint's `.pt` and `.npz` (5.1).

Rows are rebuilt from the stored pre-decision views, never read off the engine (section 4):

- a **fight row** is the view at fight start, its encounter and weak flag, and its outcome
  class (`models.fight_class`: HP at the start against HP after the fight, and death --
  never a step reward);
- a **V row** is the afterstate of the option a decision chose
  (`features.encode_afterstate`), with its bootstrapped floor target
  `y_t = (1 - beta) onehot(final floor) + beta p_gen(x_{t+1})` and its end-HP target (5.2).
  A decision whose chosen option is unscorable -- the rules' enchant, an unknown screen --
  has no afterstate and gives no row, and the rows around it bootstrap through the next
  state that has a value (erratum O4).

Torch is imported here and nowhere a worker or the service loads.
"""

from __future__ import annotations

import copy
import hashlib
import math
import operator
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn import functional

from sts2_gym.seeds import canonicalize

from . import FEATURE_VERSION, features, models
from .engine import CombatBudget
from .evaluate import EVAL_SEEDS, TEST_SEEDS, load_seeds
from .shards import RunRecord, entry_decision, read_shard, state_view

if TYPE_CHECKING:
    from .features import RunView

__all__ = [
    "CombatBudget",
    "FightData",
    "SparseRows",
    "TrainConfig",
    "ValueData",
    "bootstrap_targets",
    "eligible_records",
    "end_hp_target",
    "fight_rows",
    "fit",
    "load_buffers",
    "train_fight_net",
    "train_value_net",
    "value_rows",
]


@dataclass(frozen=True)
class TrainConfig:
    """The training settings; every default is DESIGN.md 5.2-5.4's (and 5.9's)."""

    value_generations: int = 8  # V's buffer: the decisions of the last 8 generations
    fight_cap: int = 400_000  # FightNet's buffer: the newest 400k fights of the era
    beta: float = 0.3  # the bootstrap blend (FINDINGS, "the value target")
    end_hp_weight: float = 0.25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 1024
    epochs: int = 30
    patience: int = 3  # epochs without a better validation loss before stopping
    validation: float = 0.1  # of runs, held out whole: their decisions and fights
    threads: int = 2
    # Beyond the lab's old recipe (docs/rl/VRUN_SWEEP.md 3); with rows_per_run=0 the "cpu" path is unchanged bit for bit.
    device: str = "cpu"  # anything else ("cuda", "cuda:1"; "cpu:0" too, which tests use) trains with the rows resident on it
    hidden: int = 256  # RunValueNet's width, both layers
    dropout: float = 0.1  # RunValueNet's, before the second layer
    # rows_per_run 4: the sweep (VRUN_SWEEP.md 1.5) measured out-of-fold Brier skill 0.044 against 0.027 for 0, best epoch 9-20 against 1-2
    rows_per_run: int = 4  # k > 0: each epoch samples k rows of every run (fewer if it has fewer); 0: every row, every epoch


# --------------------------------------------------------------------------------------
# Targets and rows: DESIGN.md 5.2
# --------------------------------------------------------------------------------------


def _chosen_scorable(entry: dict[str, Any]) -> bool:
    return entry["options"][entry["chosen"]].get("scorable", True)


def bootstrap_targets(record: RunRecord, beta: float) -> list[tuple[int, np.ndarray]]:
    """Return (decision index, floor target) for every decision that gives a V row.

    `y_t = (1 - beta) onehot(final floor) + beta p_gen(x_{t+1})`, where `p_gen(x_{t+1})`
    is the floor distribution the generating network stored for the next state that has
    one (`p_floor`). A decision without a scorable choice gives no row and stores no
    distribution, so the rows before it bootstrap through the state after it (erratum O4).
    The last such decision of a run, and every decision of a policy without a network (the
    rules policy stores none), take beta = 0.
    """
    decisions = record["decisions"]
    outcome = np.zeros(models.FLOOR_CLASSES)
    outcome[models.floor_class(record["outcome"]["final_floor"])] = 1.0
    following: list[float] | None = None  # the next stored distribution, walking back
    targets: list[tuple[int, np.ndarray]] = []
    for t in range(len(decisions) - 1, -1, -1):
        entry = decisions[t]
        if _chosen_scorable(entry):
            target = (
                outcome.copy()
                if following is None
                else (1.0 - beta) * outcome + beta * np.asarray(following, np.float64)
            )
            targets.append((t, target))
        if entry.get("p_floor") is not None:
            following = entry["p_floor"]
    targets.reverse()
    return targets


def end_hp_target(record: RunRecord) -> float:
    """Return the end-HP target: hp / max hp at the end of Act 1, 0 for a death (5.2)."""
    outcome = record["outcome"]
    if outcome["died"]:
        return 0.0
    return outcome["end_hp"] / max(1, outcome["end_max_hp"])


def fight_rows(record: RunRecord) -> list[tuple[np.ndarray, int, bool, int]]:
    """Return (base vector at fight start, encounter, weak, class) for each fight."""
    return [
        (
            features.encode(state_view(record, fight)),
            int(fight["encounter"]),
            bool(fight["weak"]),
            models.fight_class(fight["hp_before"], fight["hp_after"]),
        )
        for fight in record["fights"]
    ]


def value_rows(
    record: RunRecord,
    beta: float,
) -> list[tuple[np.ndarray, RunView, np.ndarray, float]]:
    """Return (afterstate base vector, view, floor target, end-HP target) per V row."""
    end_hp = end_hp_target(record)
    rows = []
    for t, target in bootstrap_targets(record, beta):
        entry = record["decisions"][t]
        view = state_view(record, entry)
        chosen = entry_decision(entry).options[entry["chosen"]]
        rows.append((features.encode_afterstate(view, chosen), view, target, end_hp))
    return rows


# --------------------------------------------------------------------------------------
# Storage: sparse rows, and the two buffers (5.4)
# --------------------------------------------------------------------------------------


def _positions(
    indptr: np.ndarray,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (the stored positions of `rows`, in order; each position's row; lengths)."""
    starts, lengths = indptr[rows], indptr[rows + 1] - indptr[rows]
    total = int(lengths.sum())
    owners = np.repeat(np.arange(len(rows)), lengths)
    offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return np.repeat(starts, lengths) + offsets, owners, lengths


@dataclass(frozen=True)
class SparseRows:
    """Rows of one width, kept by their non-zero entries (CSR).

    A feature vector is mostly zeros -- block E alone is 1154 wide -- and a buffer holds
    hundreds of thousands of them, so rows are stored sparse and made dense one batch at a
    time (`take`). Every stored value is the float32 the dense row held.
    """

    indptr: np.ndarray  # int64, one more than the rows
    indices: np.ndarray  # int32
    values: np.ndarray  # float32
    width: int = features.BASE_SIZE

    @classmethod
    def build(
        cls,
        vectors: Iterable[np.ndarray],
        width: int = features.BASE_SIZE,
    ) -> SparseRows:
        indptr, indices, values = [0], [], []
        for vector in vectors:
            dense = np.asarray(vector, dtype=np.float32)
            nonzero = np.flatnonzero(dense)
            indices.append(nonzero.astype(np.int32))
            values.append(dense[nonzero])
            indptr.append(indptr[-1] + len(nonzero))
        return cls(
            np.asarray(indptr, dtype=np.int64),
            np.concatenate(indices) if indices else np.zeros(0, np.int32),
            np.concatenate(values) if values else np.zeros(0, np.float32),
            width,
        )

    @classmethod
    def concatenate(
        cls,
        parts: Sequence[SparseRows],
        width: int = features.BASE_SIZE,
    ) -> SparseRows:
        indptr, offset = [np.zeros(1, np.int64)], 0
        for part in parts:
            indptr.append(part.indptr[1:] + offset)
            offset += int(part.indptr[-1])
        return cls(
            np.concatenate(indptr),
            np.concatenate([p.indices for p in parts] or [np.zeros(0, np.int32)]),
            np.concatenate([p.values for p in parts] or [np.zeros(0, np.float32)]),
            width,
        )

    def __len__(self) -> int:
        return len(self.indptr) - 1

    def take(self, rows: np.ndarray) -> np.ndarray:
        """Return the given rows, dense, as float32."""
        rows = np.asarray(rows, dtype=np.int64)
        positions, owners, _ = _positions(self.indptr, rows)
        dense = np.zeros((len(rows), self.width), dtype=np.float32)
        dense[owners, self.indices[positions]] = self.values[positions]
        return dense

    def select(self, rows: np.ndarray) -> SparseRows:
        """Return the given rows, still sparse."""
        rows = np.asarray(rows, dtype=np.int64)
        positions, _, lengths = _positions(self.indptr, rows)
        indptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(lengths)])
        return SparseRows(
            indptr,
            self.indices[positions],
            self.values[positions],
            self.width,
        )


@dataclass(frozen=True)
class FightData:
    """FightNet's buffer: one row per fight."""

    bases: SparseRows
    encounters: np.ndarray  # int64
    weak: np.ndarray  # float32
    labels: np.ndarray  # int64, `models.fight_class`
    validation: np.ndarray  # bool: the fights of the held-out tenth of runs
    hashes: np.ndarray | None = None  # float64: each fight's run hash (`keep.cross_fit_keep` folds the fights by it)


@dataclass
class ValueData:
    """V's buffer: one row per decision with a scorable choice.

    `query_views[query_keys[i]]` is a view with row i's region and boss, which is all the
    fight queries read (`features.fight_queries`); `fight_block` is H, filled by
    `fill_fight_block` once FightNet is trained.
    """

    bases: SparseRows
    query_views: list[RunView]
    query_keys: np.ndarray  # int64
    targets: np.ndarray  # float32, (rows, 18)
    end_hp: np.ndarray  # float32
    validation: np.ndarray  # bool: the held-out tenth of runs
    fight_block: np.ndarray | None = None
    runs: int = 0
    generations: tuple[int, ...] = ()
    duplicates_dropped: int = 0  # records `load_buffers` left out because their (generation, seed) was already read
    # What the keep conditions (`keep.evaluate_value_net`) read, one entry per row: the run's final floor, the run's index,
    # the row's own floor, and V0's inputs (HP fraction, potions held, act index)
    final_floor: np.ndarray | None = None
    floor: np.ndarray | None = None
    run_index: np.ndarray | None = None
    v0: np.ndarray | None = None
    run_hash: np.ndarray | None = None  # float64: `_unit_hash` of each row's run seed

    def fill_fight_block(self, fight: models.NumpyFightNet) -> None:
        """Compute H for every row with `fight`, exactly as the policy does (3.3, 5.3)."""
        block = np.zeros((len(self.bases), features.FIGHT_BLOCK_SIZE), np.float32)
        for key, view in enumerate(self.query_views):
            rows = np.flatnonzero(self.query_keys == key)
            queries = features.fight_queries(view)
            for start in range(0, len(rows), 8192):
                part = rows[start : start + 8192]
                block[part] = fight.summary(self.bases.take(part), queries)
        self.fight_block = block

    def inputs(self, rows: np.ndarray) -> np.ndarray:
        """Return the given rows as V's 1851-wide input: base vector and H.

        Raises:
            RuntimeError: Before `fill_fight_block`.

        """
        if self.fight_block is None:
            raise RuntimeError("H is filled by the trained FightNet first (5.3)")
        return np.concatenate([self.bases.take(rows), self.fight_block[rows]], axis=1)


def _unit_hash(key: str) -> float:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") / 2**64


def _held_out(hashes: np.ndarray, fraction: float) -> np.ndarray:
    """Return the validation mask: the groups whose hash falls under `fraction`.

    A group keeps its side from one generation to the next, since its hash never changes.
    A buffer too small to hold out a tenth still holds out one group when it has two, and
    never holds out everything.
    """
    held = hashes < fraction
    if not held.any() and len(np.unique(hashes)) >= 2:
        held = hashes == hashes.min()
    if held.all():
        held = np.zeros_like(held)
    return held


@cache
def _reserved_seeds() -> dict[str, str]:
    """Return every held-out seed, canonical, and the file that reserves it (5.6)."""
    return {
        canonicalize(seed): path.name
        for path in (EVAL_SEEDS, TEST_SEEDS)
        for seed in load_seeds(path)
    }


def _in_era(record: RunRecord, budget: str, ascension: int) -> bool:
    """Whether a record was played in the current data era (5.4, 5.7, erratum E6).

    A change of combat budget, ascension or `FEATURE_VERSION` starts a new era.
    """
    return (
        record["combat"]["budget"] == budget
        and record["ascension"] == ascension
        and record["feature_version"] == FEATURE_VERSION
    )


def eligible_records(
    shards: Iterable[Path | str],
    *,
    generation: int,
    combat_budget: CombatBudget | int,
    ascension: int,
    dedupe: bool = True,
) -> Iterator[tuple[int, int, RunRecord]]:
    """Yield (shard order, line, record) of every run record that counts for generation `generation` (5.4).

    A record counts when it was played at `generation` or before, was not truncated, is of the current era
    (`_in_era`), and its (generation, seed) was not read before (a resumed block can hold copies). Records come in the
    order the shards are given in. This is the filter `load_buffers` applies (it keeps its own loop to count the copies).

    Raises:
        ValueError: If any record, of any era, is a run of an eval or final-test seed (5.6).

    """
    budget = str(CombatBudget.of(combat_budget))
    reserved = _reserved_seeds()
    seen: set[tuple[int, str]] = set()
    for order, path in enumerate(shards):
        for line, record in enumerate(read_shard(Path(path))):
            seed = canonicalize(record["seed"])
            if seed in reserved:
                raise ValueError(
                    f"{path}: run {record['seed']!r} is seed {seed} of {reserved[seed]}, "
                    "which is held out of training (DESIGN.md 5.6)",
                )
            played = record.get("generation")
            if (
                played is None
                or played > generation
                or record["outcome"]["truncated"]
                or not _in_era(record, budget, ascension)
            ):
                continue
            if dedupe:  # the same rule as load_buffers: one record per (generation, seed); off for a census of the copies
                if (played, seed) in seen:
                    continue
                seen.add((played, seed))
            yield order, line, record


def load_buffers(
    shards: Iterable[Path | str],
    *,
    pairs: Iterable[Path | str] = (),
    generation: int,
    combat_budget: CombatBudget | int,
    ascension: int,
    config: TrainConfig = TrainConfig(),  # noqa: B008 -- frozen
) -> tuple[FightData, ValueData]:
    """Read the replay buffers the shards hold for training generation `generation` (5.4).

    Only records of the current era count (`_in_era`), from generation `generation` or
    before, and never a truncated run (1.4). `pairs` are paired fight records
    (`deckfights.pair_records`): FightNet rows only. V keeps the decisions of the last
    `value_generations` generations; FightNet keeps every fight, capped at the newest
    `fight_cap` (newest by generation, then by the order the shards are given in).
    A (generation, seed) is one run: shards that hold a seed more than once (a self-play block
    the lab resumed before its writer was idempotent) give the first copy that counts, so a
    truncated copy never stands in for a complete one, and the later copies are dropped and
    counted in `ValueData.duplicates_dropped`.

    Raises:
        ValueError: If any record, of any era, is a run of an eval or final-test seed:
            those are never trained on (5.6), so a shard holding one is a leak.

    """
    budget = str(CombatBudget.of(combat_budget))
    reserved = _reserved_seeds()
    fights: list[_FightChunk] = []
    value = _ValueBuilder()
    seen: set[tuple[int, str]] = set()
    duplicates = 0
    for order, path in enumerate(shards):
        for line, record in enumerate(read_shard(Path(path))):
            seed = canonicalize(record["seed"])
            if seed in reserved:
                raise ValueError(
                    f"{path}: run {record['seed']!r} is seed {seed} of {reserved[seed]}, "
                    "which is held out of training (DESIGN.md 5.6)",
                )
            played = record.get("generation")
            if (
                played is None
                or played > generation
                or record["outcome"]["truncated"]
                or not _in_era(record, budget, ascension)
            ):
                continue
            if (played, seed) in seen:
                duplicates += 1
                continue
            seen.add((played, seed))
            if rows := fight_rows(record):
                key = (played, order, line)
                fights.append(_FightChunk.of(key, record["seed"], rows))
            if played > generation - config.value_generations:
                value.add(record, value_rows(record, config.beta))
    for order, path in enumerate(pairs):
        for line, record in enumerate(read_shard(Path(path))):
            seed = canonicalize(record["seed"])
            if seed in reserved:
                raise ValueError(
                    f"{path}: pairs of run {record['seed']!r} are seed {seed} of "
                    f"{reserved[seed]}, which is held out of training (DESIGN.md 5.6)",
                )
            played = record.get("generation")
            if played is None or played > generation or not _in_era(record, budget, ascension):
                continue
            if rows := fight_rows(record):
                # After the generation's own runs: the newest fights of that generation.
                key = (played, 1_000_000 + order, line)
                fights.append(_FightChunk.of(key, record["seed"], rows))
    values = value.finish(config)
    values.duplicates_dropped = duplicates
    return _fight_data(fights, config), values


@dataclass(frozen=True)
class _FightChunk:
    """One run's fights, sparse, with the key that orders runs from oldest to newest."""

    key: tuple[int, int, int]  # (generation, shard order, line in the shard)
    bases: SparseRows
    encounters: np.ndarray
    weak: np.ndarray
    labels: np.ndarray
    hashes: np.ndarray  # each fight's validation hash: its run's

    @classmethod
    def of(cls, key: tuple[int, int, int], seed: str, rows: list) -> _FightChunk:
        return cls(
            key,
            SparseRows.build(row[0] for row in rows),
            np.array([row[1] for row in rows], dtype=np.int64),
            np.array([row[2] for row in rows], dtype=np.float32),
            np.array([row[3] for row in rows], dtype=np.int64),
            # By run: the fights of V's held-out runs are FightNet's held-out fights.
            np.full(len(rows), _unit_hash(seed)),
        )

    def last(self, count: int) -> _FightChunk:
        """Return this run's last `count` fights, its newest."""
        keep = np.arange(len(self.labels) - count, len(self.labels))
        return _FightChunk(
            self.key,
            self.bases.select(keep),
            self.encounters[keep],
            self.weak[keep],
            self.labels[keep],
            self.hashes[keep],
        )


def _fight_data(chunks: list[_FightChunk], config: TrainConfig) -> FightData:
    """Keep the newest `fight_cap` fights, oldest first; hold out those of held-out runs."""
    kept: list[_FightChunk] = []
    room = config.fight_cap
    for chunk in sorted(chunks, key=operator.attrgetter("key"), reverse=True):
        if room <= 0:
            break
        if len(chunk.labels) > room:
            chunk = chunk.last(room)
        kept.append(chunk)
        room -= len(chunk.labels)
    kept.reverse()

    def joined(field: str, dtype: type) -> np.ndarray:
        return np.concatenate(
            [getattr(chunk, field) for chunk in kept] or [np.zeros(0, dtype)],
        ).astype(dtype, copy=False)

    return FightData(
        bases=SparseRows.concatenate([chunk.bases for chunk in kept]),
        encounters=joined("encounters", np.int64),
        weak=joined("weak", np.float32),
        labels=joined("labels", np.int64),
        validation=_held_out(joined("hashes", np.float64), config.validation),
        hashes=joined("hashes", np.float64),
    )


class _ValueBuilder:
    """Accumulates V rows run by run, sparse as they arrive."""

    def __init__(self) -> None:
        self.parts: list[SparseRows] = []
        self.keys: list[int] = []
        self.views: dict[tuple[int, int], int] = {}
        self.query_views: list[RunView] = []
        self.targets: list[np.ndarray] = []
        self.end_hp: list[float] = []
        self.hashes: list[float] = []
        self.final_floor: list[int] = []
        self.floor: list[int] = []
        self.run_index: list[int] = []
        self.v0: list[tuple[float, int, int]] = []
        self.runs = 0
        self.generations: set[int] = set()

    def add(self, record: RunRecord, rows: list) -> None:
        if not rows:
            return
        self.runs += 1
        self.generations.add(int(record["generation"]))
        self.parts.append(SparseRows.build(row[0] for row in rows))
        run_hash = _unit_hash(record["seed"])
        for _, view, target, end_hp in rows:
            key = (view.act_region, view.boss)
            if key not in self.views:
                self.views[key] = len(self.query_views)
                self.query_views.append(view)
            self.keys.append(self.views[key])
            self.targets.append(target)
            self.end_hp.append(end_hp)
            self.hashes.append(run_hash)
            self.final_floor.append(int(record["outcome"]["final_floor"]))
            self.floor.append(int(view.floor))
            self.run_index.append(self.runs)
            self.v0.append((view.hp / max(1, view.max_hp), len(view.potions), view.act_index))

    def finish(self, config: TrainConfig) -> ValueData:
        n = len(self.keys)
        return ValueData(
            bases=SparseRows.concatenate(self.parts),
            query_views=self.query_views,
            query_keys=np.array(self.keys, dtype=np.int64),
            targets=(
                np.array(self.targets, dtype=np.float32)
                if n
                else np.zeros((0, models.FLOOR_CLASSES), np.float32)
            ),
            end_hp=np.array(self.end_hp, dtype=np.float32),
            validation=_held_out(np.array(self.hashes), config.validation),
            runs=self.runs,
            generations=tuple(sorted(self.generations)),
            final_floor=np.array(self.final_floor, dtype=np.int64),
            floor=np.array(self.floor, dtype=np.int64),
            run_index=np.array(self.run_index, dtype=np.int64),
            v0=np.array(self.v0, dtype=np.float32).reshape(n, 3),
            run_hash=np.array(self.hashes, dtype=np.float64),
        )


# --------------------------------------------------------------------------------------
# Optimisation: DESIGN.md 5.3
# --------------------------------------------------------------------------------------


def _device_of(config: TrainConfig) -> torch.device | None:
    """Return the device a fit trains on, or None for the lab's own CPU path (`config.device == "cpu"`)."""
    return None if config.device == "cpu" else torch.device(config.device)


def _epoch_order(
    train_rows: np.ndarray,
    groups: np.ndarray | None,
    per_group: int,
    generator: torch.Generator,
) -> np.ndarray:
    """Return an epoch's rows in the order they are visited.

    Every row, shuffled, by default. With `per_group` k > 0, k rows of every group (fewer if it has fewer), drawn without
    replacement by the same generator, then shuffled: each run gets the same say however long it is.
    """
    if per_group <= 0 or groups is None:
        return train_rows[torch.randperm(len(train_rows), generator=generator).numpy()]
    keys = torch.rand(len(train_rows), generator=generator).numpy()
    group = groups[train_rows]
    by_group = np.lexsort((keys, group))  # grouped, and random within a group
    sorted_group = group[by_group]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(sorted_group)) + 1])
    sizes = np.diff(np.concatenate([starts, [len(sorted_group)]]))
    rank = np.arange(len(sorted_group)) - np.repeat(starts, sizes)
    chosen = train_rows[by_group[rank < per_group]]
    return chosen[torch.randperm(len(chosen), generator=generator).numpy()]


def _optimise(
    net: torch.nn.Module,
    loss_of: Callable[[Any], torch.Tensor],
    validation: np.ndarray,
    config: TrainConfig,
    generator: torch.Generator,
    *,
    groups: np.ndarray | None = None,
    epoch_end: Callable[[int, torch.nn.Module], dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Fit `net` with AdamW on a cosine schedule, early-stopped on validation loss.

    At most `config.epochs` epochs; training stops after `patience` epochs without a
    better validation loss and keeps the best epoch's weights. With nothing held out (a
    buffer of one run) it trains every epoch and keeps the last.

    On the CPU `loss_of` takes a numpy array of rows. On another device (`config.device`; the net is already there) it takes a
    tensor of rows on that device, the epoch's loss is summed there and read once, and the shuffle stays on the CPU generator.
    `groups` (each row's run) with `config.rows_per_run` > 0 samples that many rows of every run per epoch. `epoch_end(epoch,
    net)`, if given, runs after each epoch's validation with the net in evaluation mode; the numbers it returns join the
    epoch's history entry.
    """
    device = _device_of(config)
    train_rows, held = np.flatnonzero(~validation), np.flatnonzero(validation)
    held_rows = held if device is None else torch.from_numpy(held).to(device)
    optimiser = torch.optim.AdamW(
        net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=config.epochs,
    )
    best, best_state, stale = math.inf, copy.deepcopy(net.state_dict()), 0
    history: list[dict[str, float]] = []
    for _ in range(config.epochs):
        net.train()
        order = _epoch_order(train_rows, groups, config.rows_per_run, generator)
        if device is None:
            total = 0.0
            for start in range(0, len(order), config.batch):
                rows = order[start : start + config.batch]
                loss = loss_of(rows)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += loss.item() * len(rows)
        else:
            order_on_device = torch.from_numpy(order).to(device)
            running = torch.zeros((), dtype=torch.float64, device=device)
            for start in range(0, len(order), config.batch):
                rows = order_on_device[start : start + config.batch]
                loss = loss_of(rows)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                running += loss.detach().double() * len(rows)
            total = float(running)
        schedule.step()
        epoch = {"train": total / max(1, len(order))}
        history.append(epoch)
        if not len(held):
            continue
        net.eval()
        with torch.no_grad():
            if device is None:
                epoch["validation"] = sum(
                    float(loss_of(held[start : start + config.batch]))
                    * len(held[start : start + config.batch])
                    for start in range(0, len(held), config.batch)
                ) / len(held)
            else:
                summed = torch.zeros((), dtype=torch.float64, device=device)
                for start in range(0, len(held), config.batch):
                    part = held_rows[start : start + config.batch]
                    summed += loss_of(part).double() * len(part)
                epoch["validation"] = float(summed) / len(held)
            if epoch_end is not None:
                epoch.update(epoch_end(len(history) - 1, net))
        if epoch["validation"] < best:
            best, best_state, stale = (
                epoch["validation"],
                copy.deepcopy(net.state_dict()),
                0,
            )
        else:
            stale += 1
            if stale >= config.patience:
                break
    if len(held):
        net.load_state_dict(best_state)
    net.eval()
    return {
        "rows": len(validation),
        "validation_rows": len(held),
        "epochs": len(history),
        "best_validation": best if math.isfinite(best) else None,
        "history": history,
    }


def _resident(
    count: int,
    make: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    chunk: int = 16384,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the dense and count inputs of rows 0..count-1 as float32 tensors on `device`, built `chunk` rows at a time.

    The whole set lives on the device (about 2.2 GB for 300k rows of V's 1,863 numbers) so no batch is densified on the CPU.
    Returns None if the device runs out of memory, after freeing what it took: the caller then densifies per batch.
    """
    dense = counts = None
    try:
        for start in range(0, count, chunk):
            part_dense, part_counts = make(np.arange(start, min(count, start + chunk)))
            if dense is None:
                dense = torch.empty((count, part_dense.shape[1]), dtype=torch.float32, device=device)
                counts = torch.empty((count, part_counts.shape[1]), dtype=torch.float32, device=device)
            dense[start : start + len(part_dense)] = torch.from_numpy(part_dense).to(device)
            counts[start : start + len(part_counts)] = torch.from_numpy(part_counts).to(device)
    except torch.cuda.OutOfMemoryError:
        dense = counts = None
        torch.cuda.empty_cache()
    return None if dense is None else (dense, counts)


def _staged(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Return a CPU array's batch on `device` through pinned memory (the fallback when the rows are not resident)."""
    tensor = torch.from_numpy(array)
    return tensor.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else tensor.to(device)


def _release(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()  # the cache would otherwise keep the rows' memory from a server sharing the GPU


def train_fight_net(
    net: torch.nn.Module,
    data: FightData,
    config: TrainConfig,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Fit FightNet: cross-entropy against each fight's class (5.3).

    On a `config.device` other than the CPU the net trains there (moved before the optimiser is built) with the rows resident,
    and comes back on the CPU. `stats["resident"]` says whether the rows fitted on the device.
    """
    labels = torch.from_numpy(data.labels)
    device = _device_of(config)

    if device is None:

        def loss_of(rows: np.ndarray) -> torch.Tensor:
            dense, counts = models.fight_inputs(
                data.bases.take(rows),
                data.encounters[rows],
                data.weak[rows],
            )
            logits = net(torch.from_numpy(dense), torch.from_numpy(counts))
            return functional.cross_entropy(logits, labels[rows])

        return _optimise(net, loss_of, data.validation, config, generator)

    def make(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return models.fight_inputs(data.bases.take(rows), data.encounters[rows], data.weak[rows])

    net.to(device)
    try:
        held = _resident(len(data.labels), make, device)
        labels_on_device = labels.to(device)

        def loss_of_device(rows: torch.Tensor) -> torch.Tensor:
            if held is not None:
                dense, counts = held[0][rows], held[1][rows]
            else:
                dense, counts = (_staged(a, device) for a in make(rows.cpu().numpy()))
            return functional.cross_entropy(net(dense, counts), labels_on_device[rows])

        stats = _optimise(net, loss_of_device, data.validation, config, generator)
        stats["resident"] = held is not None
    finally:
        held = None
        net.to("cpu")
        _release(device)
    return stats


def train_value_net(
    net: torch.nn.Module,
    data: ValueData,
    config: TrainConfig,
    generator: torch.Generator,
    *,
    epoch_end: Callable[[int, torch.nn.Module], dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Fit V: soft cross-entropy on the floor logits plus 0.25 x MSE on end HP (5.3).

    `config.rows_per_run` > 0 samples that many rows of every run per epoch (it needs `data.run_index`); the loss stays a
    mean over the sampled rows. On a `config.device` other than the CPU the net trains there (moved before the optimiser is
    built) with the dense rows resident, H included, and comes back on the CPU; `stats["resident"]` says whether they fitted.
    `epoch_end(epoch, net)` runs after each epoch's validation (see `_optimise`).

    Raises:
        ValueError: For `rows_per_run` without `data.run_index`.

    """
    targets, end_hp = torch.from_numpy(data.targets), torch.from_numpy(data.end_hp)
    device = _device_of(config)
    groups = None
    if config.rows_per_run > 0:
        if data.run_index is None:
            raise ValueError("rows_per_run samples rows of each run, and this data has no run_index")
        groups = data.run_index

    if device is None:

        def loss_of(rows: np.ndarray) -> torch.Tensor:
            dense, counts = models.value_inputs(data.inputs(rows))
            logits, predicted = net(torch.from_numpy(dense), torch.from_numpy(counts))
            log_p = functional.log_softmax(logits, dim=1)
            floors = -(targets[rows] * log_p).sum(dim=1).mean()
            end = functional.mse_loss(predicted, end_hp[rows])
            return floors + config.end_hp_weight * end

        return _optimise(net, loss_of, data.validation, config, generator, groups=groups, epoch_end=epoch_end)

    def make(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return models.value_inputs(data.inputs(rows))

    net.to(device)
    try:
        held = _resident(len(data.targets), make, device)
        targets_on_device, end_hp_on_device = targets.to(device), end_hp.to(device)

        def loss_of_device(rows: torch.Tensor) -> torch.Tensor:
            if held is not None:
                dense, counts = held[0][rows], held[1][rows]
            else:
                dense, counts = (_staged(a, device) for a in make(rows.cpu().numpy()))
            logits, predicted = net(dense, counts)
            log_p = functional.log_softmax(logits, dim=1)
            floors = -(targets_on_device[rows] * log_p).sum(dim=1).mean()
            end = functional.mse_loss(predicted, end_hp_on_device[rows])
            return floors + config.end_hp_weight * end

        stats = _optimise(
            net,
            loss_of_device,
            data.validation,
            config,
            generator,
            groups=groups,
            epoch_end=epoch_end,
        )
        stats["resident"] = held is not None
    finally:
        held = None
        net.to("cpu")
        _release(device)
    return stats


def fit(
    shards: Iterable[Path | str],
    prev_checkpoint: Path | str | None,
    out: Path | str,
    *,
    pairs: Iterable[Path | str] = (),
    generation: int,
    combat_budget: CombatBudget | int,
    ascension: int,
    config: TrainConfig = TrainConfig(),  # noqa: B008 -- frozen
    evaluate: Callable[[ValueData, torch.nn.Module], dict[str, Any]] | None = None,
    cross_fit: Callable[[FightData, ValueData, TrainConfig, dict[str, Any] | None], dict[str, Any]] | None = None,
) -> Path:
    """Train FightNet then V on the buffers of generation `generation`; write `out` (5.3).

    `out` is the `.pt` checkpoint, and its `.npz` export is written beside it, last.
    `prev_checkpoint` (a `.pt`) is the learner to warm-start from, or None for fresh
    weights. Torch is seeded with the generation and runs on `config.threads` threads,
    restored afterwards. Returns `out`. A shard holding a run of an eval or final-test
    seed raises `ValueError` (`load_buffers`).

    Raises:
        ValueError: If either buffer is empty: no record of this era, generation window and
            outcome to train on.

    """
    out = Path(out)
    budget = str(CombatBudget.of(combat_budget))
    threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        fights, values = load_buffers(
            shards,
            pairs=pairs,
            generation=generation,
            combat_budget=combat_budget,
            ascension=ascension,
            config=config,
        )
        if not len(fights.labels) or not len(values.targets):
            raise ValueError(
                f"no training data for generation {generation} at {budget}, ascension "
                f"{ascension}: {len(fights.labels)} fights and {len(values.targets)} "
                "decisions in the buffers",
            )
        torch.manual_seed(generation)
        generator = torch.Generator().manual_seed(generation)
        fight_net = models.FightNet()
        value_net = models.RunValueNet(hidden=config.hidden, dropout=config.dropout)
        if prev_checkpoint is not None:
            models.load_torch(prev_checkpoint, fight_net, value_net)
        fight_stats = train_fight_net(fight_net, fights, config, generator)
        values.fill_fight_block(models.NumpyFightNet(models.numpy_params(fight_net)))
        value_stats = train_value_net(value_net, values, config, generator)
        keep_doc = evaluate(values, value_net) if evaluate is not None else None  # the held-out tenth (keep.evaluate_value_net)
        if cross_fit is not None:  # D33: the out-of-fold fits the keep verdict rests on; it gets the tenth's record to report
            keep_doc = cross_fit(fights, values, config, keep_doc)
        if keep_doc is not None:
            value_stats = {**value_stats, "keep": keep_doc}
        models.save(
            out,
            fight_net,
            value_net,
            {
                "generation": generation,
                "era": {"combat_budget": budget, "ascension": ascension},
                "warm_start": None if prev_checkpoint is None else str(prev_checkpoint),
                "fight": fight_stats,
                "value": {
                    **value_stats,
                    "runs": values.runs,
                    "generations": list(values.generations),
                    "duplicates_dropped": values.duplicates_dropped,
                },
                "config": asdict(config),
            },
        )
        return out
    finally:
        torch.set_num_threads(threads)
