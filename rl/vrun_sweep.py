"""Train V_run harder on the GPU and judge it (docs/rl/VRUN_SWEEP.md; GPU piece 2).

    vrun_sweep.py cache   --lab LAB --agent-root AGENT --out VRUN      the deduplicated window, cached
    vrun_sweep.py run     --cache DIR --arms a,b|all --seeds 3 --device cuda [--budget-min N] [--stop-file PATH]
    vrun_sweep.py ensemble --cache DIR --arm a                          5 seeds, probabilities averaged
    vrun_sweep.py confirm --cache DIR --arms lab,a[,b] --seeds 1 --device cuda
    vrun_sweep.py table   --cache DIR                                   the report's two tables
    vrun_sweep.py beside  --cache DIR --lab LAB                         one fit beside a running self-play block

Every fit reads the cache (`cache`): the window's closed blocks, run records deduplicated by seed, V's rows without H, and
FightNet's fights with their run hash. Runs are split by `train._unit_hash` of the seed: S = [0.2, 0.3) selects (arms are
early-stopped and ranked on it), and the rest is cross-fitted in five folds (`confirm`). H is not cached with the data: FightNet
is refit per seed (S phase, on every run outside S) or per fold (confirmation, on the other four folds), so no run's H comes from
a FightNet that trained on its own fights. Fits use at most 2 CPU threads at BelowNormal. Outputs hold numbers and paths
relative to the cache folder, never absolute paths.

`run --stop-file PATH` is the quiet-window runner's brake (VRUN_SWEEP.md section 7): the file is looked for at every epoch end and
at every fit boundary, and when it exists the fit in hand is dropped (nothing of it is written) and the process exits with code 3.
"""

from __future__ import annotations

import os as _os  # noqa: E402

if __name__ == "__main__":  # D31: run as a command this is self-play, promotion or training: old engine, whatever STS2_ENGINE says
    _os.environ.pop("STS2_ENGINE", None)  # sts2_gym.native reads it once, when the imports below load it (NATIVE_KERNEL.md section 3)

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")  # numpy's BLAS stays within the two threads a fit may use

import argparse
import contextlib
import ctypes
import dataclasses
import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

_RL = Path(__file__).resolve().parent
for _p in (_RL, _RL.parent / "src", _RL.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_loop  # noqa: E402
import torch  # noqa: E402
from run_train import keep, models, shards, train  # noqa: E402
from run_train.engine import CombatBudget  # noqa: E402
from sts2_gym.seeds import canonicalize  # noqa: E402

CACHE_FORMAT = "vrun-sweep-cache/1"
S_RANGE = (0.2, 0.3)  # the selection split, by run-seed hash
LAB_RANGE = (0.0, 0.1)  # the lab's own held-out tenth (left untouched)
FOLDS = 5
THREADS = 2
CLOSED_AFTER_MIN = 10.0  # a block whose manifest has not changed for this long, with no done marker, counts as closed
FIRST_BOSS = models.floor_class(keep.ACT1_LAST_FLOOR + 1)  # the first floor class past the Act 1 boss
EXPECTED_FLOOR_W = 53.0
STOP_EXIT = 3  # `run --stop-file`: the exit code of a fit dropped because the stop file appeared


class StopFit(Exception):
    """The stop file exists: the fit in hand is dropped (raised at an epoch end or a fit boundary, caught by `cmd_run`)."""


_STOP: dict[str, Path | None] = {"file": None}


def check_stop() -> None:
    """Raise `StopFit` when `run --stop-file` was given and that file exists; a no-op otherwise."""
    path = _STOP["file"]
    if path is not None and path.exists():
        raise StopFit("the stop file exists")


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------


def in_selection(hashes: np.ndarray) -> np.ndarray:
    return (hashes >= S_RANGE[0]) & (hashes < S_RANGE[1])


def in_lab_split(hashes: np.ndarray) -> np.ndarray:
    return (hashes >= LAB_RANGE[0]) & (hashes < LAB_RANGE[1])


def fold_of(hashes: np.ndarray) -> np.ndarray:
    """Return each run's fold 0..4 over the hashes outside S, and -1 for a hash inside S."""
    h = np.asarray(hashes, np.float64)
    fold = np.clip((FOLDS * ((h - S_RANGE[1]) % 1.0) / 0.9).astype(np.int64), 0, FOLDS - 1)  # 0.9 = the hash mass outside S
    return np.where(in_selection(h), -1, fold)


def fight_validation(hashes: np.ndarray) -> np.ndarray:
    """FightNet's early-stopping runs: the lowest tenth of every hash tenth, spread over S-free folds alike."""
    return np.mod(np.asarray(hashes, np.float64) * 10.0, 1.0) < 0.1


def subsample_runs(hashes: np.ndarray, fraction: float) -> np.ndarray:
    """Return the runs a data-curve arm trains on: a hash-chosen `fraction` of them, nested (25% inside 50% inside 75%)."""
    return np.mod(np.asarray(hashes, np.float64) * 1_000_003.0, 1.0) < fraction


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON")


def save_json(path: Path, doc: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)


def peak_rss_mb() -> float | None:
    """Return this process's peak resident set in MB: psutil if present, else the Windows counter through ctypes."""
    with contextlib.suppress(ImportError):
        import psutil  # noqa: PLC0415

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak:
            return peak / 2**20
    if sys.platform != "win32":
        return None
    from ctypes import wintypes  # noqa: PLC0415

    class Counters(ctypes.Structure):
        _fields_ = [  # noqa: RUF012
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    if not kernel.K32GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return counters.PeakWorkingSetSize / 2**20


def _lower_priority() -> None:
    try:
        from sts2_gym import procutil  # noqa: PLC0415

        procutil.lower_priority()
    except Exception as error:  # noqa: BLE001 -- a failed drop must not stop a fit, but it must not pass unseen either
        print(f"vrun_sweep: could not lower the process priority ({type(error).__name__}: {error}); running at normal priority",
              file=sys.stderr)


# --------------------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------------------


def closed_blocks(
    manifests: list[tuple[Path, dict]],
    lab: Path,
    *,
    now: float,
    closed_after_s: float = CLOSED_AFTER_MIN * 60,
) -> tuple[list[tuple[Path, dict, str]], list[dict]]:
    """Split the window's blocks into closed ones (with why) and skipped ones.

    A block is closed when its job folder has `done.rc`, or when its `shards.json` has not changed for `closed_after_s`.
    """
    kept, skipped = [], []
    for directory, manifest in manifests:
        job, path = directory.parent, directory / "shards.json"
        relative = path.relative_to(lab).as_posix()
        if (job / "done.rc").is_file():
            kept.append((directory, manifest, "done.rc"))
        elif now - path.stat().st_mtime > closed_after_s:
            kept.append((directory, manifest, "age"))
        else:
            skipped.append({"manifest": relative, "why": "no done.rc and the manifest changed in the last "
                            f"{closed_after_s / 60:g} minutes"})
    return kept, skipped


def cache_key(blocks: list[tuple[Path, dict, str]], lab: Path) -> str:
    """sha256 over the sorted (manifest path relative to the lab, shard names, shard sizes) of the blocks."""
    entries = []
    for directory, manifest, _ in blocks:
        names = list(manifest.get("run_shards", []))
        sizes = [(directory / name).stat().st_size for name in names]
        entries.append([(directory / "shards.json").relative_to(lab).as_posix(), names, sizes])
    return hashlib.sha256(json.dumps(sorted(entries)).encode("utf-8")).hexdigest()


@dataclass
class Census:
    """What the window's run records held: per block, and the copies of seeds."""

    blocks: list[dict]
    seen: dict[str, int] = field(default_factory=dict)  # canonical seed -> index of the block that first had it
    copies: dict[str, int] = field(default_factory=dict)  # canonical seed -> number of extra records

    def see(self, block: int, seed: str) -> bool:
        """Count one eligible record; return True if it is the seed's first (to be kept)."""
        row = self.blocks[block]
        row["records"] += 1
        key = canonicalize(seed)
        if key in self.seen:
            row["copies_in_block" if self.seen[key] == block else "copies_of_earlier_blocks"] += 1
            self.copies[key] = self.copies.get(key, 0) + 1
            return False
        self.seen[key] = block
        row["kept"] += 1
        return True

    def summary(self) -> dict[str, Any]:
        records = sum(b["records"] for b in self.blocks)
        return {
            "per_block": self.blocks,
            "records": records,
            "unique_seeds": len(self.seen),
            "extra_copies": sum(self.copies.values()),
            "seeds_with_copies": len(self.copies),
            "max_copies_of_a_seed": 1 + max(self.copies.values(), default=0),
        }


def split_counts(values: train.ValueData) -> dict[str, Any]:
    """Row, run, Act-1 kill and head counts of S, the lab's split and the five folds."""
    run = np.asarray(values.run_index)
    final, floor, hashes = np.asarray(values.final_floor), np.asarray(values.floor), np.asarray(values.run_hash)
    runs, first = np.unique(run, return_index=True)
    run_hash, run_final = hashes[first], final[first]
    row_fold = fold_of(hashes)

    def counts(row_mask: np.ndarray, run_mask: np.ndarray) -> dict[str, int]:
        head = row_mask & (floor <= keep.ACT1_HEAD_LAST_FLOOR)
        return {
            "runs": int(run_mask.sum()),
            "kill_runs": int((run_mask & (run_final > keep.ACT1_LAST_FLOOR)).sum()),
            "rows": int(row_mask.sum()),
            "head_rows": int(head.sum()),
            "positive_head_rows": int((head & (final > keep.ACT1_LAST_FLOOR)).sum()),
        }

    out = {
        "all": counts(np.ones(len(run), bool), np.ones(len(runs), bool)),
        "S": counts(in_selection(hashes), in_selection(run_hash)),
        "lab_split": counts(in_lab_split(hashes), in_lab_split(run_hash)),
        "outside_S": counts(~in_selection(hashes), ~in_selection(run_hash)),
    }
    run_fold = fold_of(run_hash)
    out["folds"] = {str(k): counts(row_fold == k, run_fold == k) for k in range(FOLDS)}
    return out


def build_cache(lab: Path, agent_root: Path, out: Path, *, closed_after_min: float = CLOSED_AFTER_MIN,
                now: float | None = None) -> Path:
    """Build `out/cache-<key>/` from the current window's closed blocks; return its folder (an existing one is kept)."""
    t0 = time.perf_counter()
    lab, agent_root, out = Path(lab), Path(agent_root), Path(out)
    now = time.time() if now is None else now
    agent = gen_loop.load_agent(agent_root=agent_root)
    generation = agent.generation
    current = gen_loop.read_json(agent_root / "current.json")
    if int(current["generation"]) != generation:
        raise ValueError("current.json and the agent directory it names disagree on the generation")
    blocks, skipped = closed_blocks(gen_loop.window_manifests(lab / "runs", generation), lab, now=now,
                                    closed_after_s=closed_after_min * 60)
    if not blocks:
        raise ValueError(f"no closed block of generations {generation - gen_loop.GEN_WINDOW + 1}..{generation} under the lab")
    key = cache_key(blocks, lab)
    folder = out / f"cache-{key[:16]}"
    if (folder / "manifest.json").is_file():
        print(f"cache {folder.name} exists ({len(blocks)} blocks); nothing to build")
        return folder
    first = blocks[0][1]
    budget, ascension = int(first["base_sims"]), int(first["ascension"])
    paths, block_of = [], []
    for index, (directory, manifest, _) in enumerate(blocks):
        for name in manifest.get("run_shards", []):
            paths.append(directory / name)
            block_of.append(index)
    census = Census([{"manifest": (d / "shards.json").relative_to(lab).as_posix(), "generation": int(m["generation"]),
                      "manifest_runs": int(m.get("runs", 0)), "closed_by": why, "shards": len(m.get("run_shards", [])),
                      "records": 0, "kept": 0, "copies_in_block": 0, "copies_of_earlier_blocks": 0}
                     for d, m, why in blocks])
    value, chunks, seeds = train._ValueBuilder(), [], []  # noqa: SLF001
    config = train.TrainConfig(threads=THREADS)
    for order, line, record in train.eligible_records(paths, generation=generation,  # dedupe off: the census counts the copies
                                                      combat_budget=CombatBudget(budget, budget), ascension=ascension, dedupe=False):
        if not census.see(block_of[order], record["seed"]):
            continue
        if rows := train.fight_rows(record):
            chunks.append(train._FightChunk.of((int(record["generation"]), order, line), record["seed"], rows))  # noqa: SLF001
        before = value.runs
        value.add(record, train.value_rows(record, config.beta))
        if value.runs > before:
            seeds.append(str(record["seed"]))
    values = value.finish(config)
    fights = train.SparseRows.concatenate([c.bases for c in chunks])
    arrays = {
        "indptr": values.bases.indptr, "indices": values.bases.indices, "values": values.bases.values,
        "query_keys": values.query_keys, "targets": values.targets, "end_hp": values.end_hp,
        "final_floor": values.final_floor, "floor": values.floor, "run_index": values.run_index, "v0": values.v0,
        "run_hash": values.run_hash, "run_seeds": np.array(seeds),
        "f_indptr": fights.indptr, "f_indices": fights.indices, "f_values": fights.values,
        "f_encounters": np.concatenate([c.encounters for c in chunks]), "f_weak": np.concatenate([c.weak for c in chunks]),
        "f_labels": np.concatenate([c.labels for c in chunks]), "f_hash": np.concatenate([c.hashes for c in chunks]),
    }
    folder.mkdir(parents=True, exist_ok=True)
    save_npz(folder / "data.npz", **arrays)
    manifest = {
        "format": CACHE_FORMAT, "key": key, "generation": generation, "base_sims": budget, "ascension": ascension,
        "blocks": census.blocks, "skipped_blocks": skipped, "census": census.summary(),
        "counts": split_counts(values), "bases_width": values.bases.width, "rows": len(values.targets), "runs": values.runs,
        "fights": int(len(arrays["f_labels"])), "fight_cap": config.fight_cap,
        "fights_exceed_cap": bool(len(arrays["f_labels"]) > config.fight_cap), "generations": list(values.generations),
        "query_views": [{"state": shards.view_to_json(v, 0), "map": shards.map_to_json(v)} for v in values.query_views],
        "built": gen_loop.utc_now(),
        "build": {"wall_min": round((time.perf_counter() - t0) / 60, 2), "peak_rss_mb": peak_rss_mb()},
    }
    save_json(folder / "manifest.json", manifest)  # last: a folder with a manifest is a whole cache
    return folder


@dataclass
class Cache:
    """A loaded cache: V's rows (no H), FightNet's fights, and the splits read off the run hashes."""

    folder: Path
    manifest: dict
    values: train.ValueData
    fights: dict[str, np.ndarray]
    run_seeds: np.ndarray

    @cached_property
    def hashes(self) -> np.ndarray:
        return self.values.run_hash

    @cached_property
    def is_s(self) -> np.ndarray:
        return in_selection(self.hashes)

    @cached_property
    def fold(self) -> np.ndarray:
        return fold_of(self.hashes)

    @cached_property
    def head_rows(self) -> np.ndarray:
        return self.values.floor <= keep.ACT1_HEAD_LAST_FLOOR

    @cached_property
    def kill(self) -> np.ndarray:
        return (self.values.final_floor > keep.ACT1_LAST_FLOOR).astype(np.float64)

    @cached_property
    def v0_score(self) -> np.ndarray:
        return keep.v0_score(self.values.v0[:, 0], self.values.v0[:, 1], self.values.v0[:, 2])


def load_cache(folder: Path) -> Cache:
    folder = Path(folder)
    manifest = gen_loop.read_json(folder / "manifest.json")
    if manifest.get("format") != CACHE_FORMAT:
        raise ValueError(f"{folder.name} is not a {CACHE_FORMAT} cache")
    with np.load(folder / "data.npz", allow_pickle=False) as data:
        a = {name: data[name] for name in data.files}
    views = [shards.view_from_json(v["state"], [v["map"]]) for v in manifest["query_views"]]
    n = len(a["targets"])
    values = train.ValueData(
        bases=train.SparseRows(a["indptr"], a["indices"], a["values"], int(manifest["bases_width"])),
        query_views=views, query_keys=a["query_keys"], targets=a["targets"], end_hp=a["end_hp"],
        validation=np.zeros(n, bool), runs=int(manifest["runs"]), generations=tuple(manifest["generations"]),
        final_floor=a["final_floor"], floor=a["floor"], run_index=a["run_index"], v0=a["v0"], run_hash=a["run_hash"])
    fights = {name[2:]: a[name] for name in a if name.startswith("f_") and name != "f_indptr"}
    fights["bases"] = train.SparseRows(a["f_indptr"], a["f_indices"], a["f_values"], int(manifest["bases_width"]))
    return Cache(folder, manifest, values, fights, a["run_seeds"])


def select_values(values: train.ValueData, rows: np.ndarray, validation: np.ndarray,
                  fight_block: np.ndarray | None = None) -> train.ValueData:
    """Return the given rows as their own `ValueData`; `validation` is per selected row; H from `fight_block` or the source."""
    rows = np.asarray(rows, np.int64)
    block = values.fight_block if fight_block is None else fight_block
    return dataclasses.replace(
        values, bases=values.bases.select(rows), query_keys=values.query_keys[rows], targets=values.targets[rows],
        end_hp=values.end_hp[rows], validation=np.asarray(validation, bool), fight_block=None if block is None else block[rows],
        final_floor=values.final_floor[rows], floor=values.floor[rows], run_index=values.run_index[rows], v0=values.v0[rows],
        run_hash=values.run_hash[rows], runs=int(len(np.unique(values.run_index[rows]))))


# --------------------------------------------------------------------------------------
# FightNet per seed / per fold, and H
# --------------------------------------------------------------------------------------


def fightnet_rows(cache: Cache, excluded_fold: int | None) -> np.ndarray:
    """Return the fights FightNet trains on: every fight outside S, and outside fold `excluded_fold` if given."""
    h = cache.fights["hash"]
    keep_mask = ~in_selection(h)
    if excluded_fold is not None:
        keep_mask &= fold_of(h) != excluded_fold
    return np.flatnonzero(keep_mask)


def fit_fightnet_h(cache: Cache, tag: str, seed: int, excluded_fold: int | None, device: str) -> np.ndarray:
    """Fit FightNet on `fightnet_rows`, fill H for every cached row from it; cached under `h/<tag>.npz`."""
    path = cache.folder / "h" / f"{tag}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            return data["fight_block"]
    rows = fightnet_rows(cache, excluded_fold)
    f = cache.fights
    data = train.FightData(bases=f["bases"].select(rows), encounters=f["encounters"][rows], weak=f["weak"][rows],
                           labels=f["labels"][rows], validation=fight_validation(f["hash"][rows]))
    config = train.TrainConfig(threads=THREADS, device=device)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    net = models.FightNet()
    t0 = time.perf_counter()
    stats = train.train_fight_net(net, data, config, generator)
    fit_min = (time.perf_counter() - t0) / 60
    t1 = time.perf_counter()
    filled = dataclasses.replace(cache.values, fight_block=None)
    filled.fill_fight_block(models.NumpyFightNet(models.numpy_params(net)))
    fill_min = (time.perf_counter() - t1) / 60
    save_npz(path, fight_block=filled.fight_block)
    save_json(path.with_suffix(".json"), {
        "tag": tag, "seed": seed, "excluded_fold": excluded_fold, "fights": int(len(rows)),
        "validation_fights": int(stats["validation_rows"]), "epochs": stats["epochs"], "best_validation": stats["best_validation"],
        "resident": stats.get("resident"), "fit_min": round(fit_min, 2), "fill_min": round(fill_min, 2), "device": device})
    return filled.fight_block


# --------------------------------------------------------------------------------------
# The keep table on a set of rows
# --------------------------------------------------------------------------------------


@dataclass
class HeadRows:
    """The Act-1 head's rows (floor <= 16) of one side of a comparison."""

    y: np.ndarray
    run: np.ndarray
    v0: np.ndarray
    floor: np.ndarray

    @classmethod
    def of(cls, cache: Cache, rows: np.ndarray) -> HeadRows:
        rows = rows[cache.head_rows[rows]]
        return cls(cache.kill[rows], cache.values.run_index[rows], cache.v0_score[rows], cache.values.floor[rows].astype(np.float64))


def head_table(p: np.ndarray, ev: HeadRows, tr: HeadRows, *, n_boot: int = keep.BOOTSTRAP) -> dict[str, Any]:
    """`keep.head_conditions` for probabilities `p` on `ev`, the base rate and V0's table from `tr`, plus V0 + floor/17 (informational)."""
    v0_p = keep.v0_probability(tr.v0, tr.y, ev.v0)
    head = keep.head_conditions(p, ev.y, ev.run, tr.y, v0_p, n_boot=n_boot)
    pf = keep.v0_probability(tr.v0 + tr.floor / keep.ACT1_LAST_FLOOR, tr.y, ev.v0 + ev.floor / keep.ACT1_LAST_FLOOR)
    head["brier_skill_vs_v0_floor_informational"] = {
        **keep.paired_bootstrap_skill(keep.brier_terms(p, ev.y), keep.brier_terms(pf, ev.y), ev.run, n_boot),
        "gated": False, "baseline": "V0 + floor / 17, through the same training-row quantile table as V0"}
    return head


def head_p(net: torch.nn.Module, dense: torch.Tensor, counts: torch.Tensor, batch: int = 8192) -> torch.Tensor:
    """P(Act-1 boss kill) from the floor softmax: the mass on the classes past floor 17, float64."""
    out = []
    with torch.no_grad():
        for start in range(0, len(dense), batch):
            logits, _ = net(dense[start:start + batch], counts[start:start + batch])
            out.append(torch.softmax(logits, dim=1)[:, FIRST_BOSS:].sum(dim=1).double())
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.float64)


def value_tensors(values: train.ValueData, rows: np.ndarray, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    dense, counts = models.value_inputs(values.inputs(rows))
    return torch.from_numpy(dense).to(device), torch.from_numpy(counts).to(device)


def expected_floor_errors(p: np.ndarray, final: np.ndarray, w: float = EXPECTED_FLOOR_W) -> np.ndarray:
    """Return |V_W - final floor| per row for floor distributions `p`: what the policy's ranking reads."""
    return np.abs(models.expected_floors(p, w) - np.asarray(final, np.float64))


def floor_baseline(train_x: np.ndarray, train_final: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Least squares of the final floor on [V0 score, floor, 1], fitted on the training rows, applied to `x`."""
    design = np.column_stack([train_x, np.ones(len(train_x))])
    coef, *_ = np.linalg.lstsq(design, np.asarray(train_final, np.float64), rcond=None)
    return np.column_stack([x, np.ones(len(x))]) @ coef


# --------------------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    name: str
    delta: dict[str, Any] = field(default_factory=dict)  # the TrainConfig changes against the lab's
    adoptable: bool = False
    kind: str = "v"  # "v": a V fit; "bce": a binary head on the kill label; "logit": a logistic regression
    device: str | None = None  # a device the arm insists on ("cpu" for the parity row)
    max_seeds: int | None = None
    subsample: float | None = None  # train on this fraction of the training runs
    zero_h: bool = False


ARMS: dict[str, Arm] = {a.name: a for a in (
    Arm("lab", {}, True),
    Arm("lab-p10", {"patience": 10}, True),
    Arm("lab-cpu", {}, False, device="cpu", max_seeds=1),
    Arm("lr3e-4", {"lr": 3e-4, "epochs": 100, "patience": 10}, True),
    Arm("lr1e-4", {"lr": 1e-4, "epochs": 100, "patience": 10}, True),
    Arm("wd1e-3", {"weight_decay": 1e-3}, True),
    Arm("wd1e-2", {"weight_decay": 1e-2}, True),
    Arm("do0.3", {"dropout": 0.3}, True),
    Arm("do0.5", {"dropout": 0.5}, True),
    Arm("h64", {"hidden": 64}, True),
    Arm("h128", {"hidden": 128}, True),
    Arm("h512", {"hidden": 512}, True),
    Arm("rows4", {"rows_per_run": 4}, True),
    Arm("rows8", {"rows_per_run": 8}, True),
    Arm("bce-act1", {}, False, kind="bce"),
    Arm("logit-v0h", {}, False, kind="logit"),
    Arm("sub25", {}, False, subsample=0.25),
    Arm("sub50", {}, False, subsample=0.50),
    Arm("sub75", {}, False, subsample=0.75),
    Arm("noH", {}, False, zero_h=True),
)}


def resolve_arm(name: str) -> Arm:
    """Return the named arm; `a+b` combines the config changes of adoptable arms (the sweep's `best+`)."""
    if name in ARMS:
        return ARMS[name]
    parts = name.split("+")
    if len(parts) > 1 and all(part in ARMS and ARMS[part].adoptable and ARMS[part].kind == "v" for part in parts):
        delta: dict[str, Any] = {}
        for part in parts:
            for key, value in ARMS[part].delta.items():
                if key in delta and delta[key] != value:
                    raise ValueError(f"{name!r} sets {key} twice ({delta[key]!r} and {value!r}); join arms that change different settings")
                delta[key] = value
        return Arm(name, delta, True)
    raise ValueError(f"unknown arm {name!r}; arms: {', '.join(ARMS)}, or adoptable arms joined with +")


def arm_config(arm: Arm, device: str) -> train.TrainConfig:
    # every arm is a change against the lab's recipe as the sweep measured it (rows_per_run=0), whatever TrainConfig's default has become
    base = train.TrainConfig(threads=THREADS, rows_per_run=0)
    return dataclasses.replace(base, device=arm.device or device, **arm.delta)


# --------------------------------------------------------------------------------------
# One fit in the S phase
# --------------------------------------------------------------------------------------


def _fit_logit(cache: Cache, h: np.ndarray, train_rows: np.ndarray, ev_rows: np.ndarray) -> tuple[np.ndarray, dict]:
    """Logistic regression of the kill label on V0's three inputs (HP fraction, potions held, act index), the floor and the 8 H
    numbers (ridge 1e-3, Newton steps)."""
    def features(rows: np.ndarray) -> np.ndarray:
        return np.column_stack([cache.values.v0[rows], cache.values.floor[rows], h[rows]]).astype(np.float64)

    x_train, y = features(train_rows), cache.kill[train_rows]
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0) + 1e-9
    design = np.column_stack([(x_train - mean) / scale, np.ones(len(y))])
    w, ridge = np.zeros(design.shape[1]), 1e-3
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-design @ w))
        gradient = design.T @ (p - y) + ridge * w
        hessian = (design * (p * (1 - p))[:, None]).T @ design + ridge * np.eye(len(w))
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    z = np.column_stack([(features(ev_rows) - mean) / scale, np.ones(len(ev_rows))]) @ w
    return 1.0 / (1.0 + np.exp(-z)), {"features": ["hp_fraction", "potions_held", "act_index", "floor", *[f"h{i}" for i in range(h.shape[1])]]}


def _fit_bce(cache: Cache, h: np.ndarray, train_rows: np.ndarray, ev_rows: np.ndarray, config: train.TrainConfig,
             seed: int) -> tuple[np.ndarray, dict]:
    """A small MLP head on the same inputs, trained with BCE on the kill label over the floor <= 16 rows."""
    device = torch.device(config.device)
    both = np.concatenate([train_rows, ev_rows])
    sub = select_values(cache.values, both, np.r_[np.zeros(len(train_rows), bool), np.ones(len(ev_rows), bool)], h)
    chunk = train._resident(len(both), lambda r: models.value_inputs(sub.inputs(r)), device)  # noqa: SLF001
    if chunk is None:
        raise RuntimeError("the BCE head's rows do not fit on the device")
    x = torch.cat(chunk, dim=1)
    del chunk
    y = torch.from_numpy(cache.kill[both]).float().to(device)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(64, 1)).to(device)

    def loss_of(rows: Any) -> torch.Tensor:
        index = torch.as_tensor(rows, device=device)
        return torch.nn.functional.binary_cross_entropy_with_logits(net(x[index]).squeeze(1), y[index])

    stats = train._optimise(net, loss_of, sub.validation, config, generator,  # noqa: SLF001
                            epoch_end=lambda _epoch, _net: check_stop() or {})
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(x[len(train_rows):]).squeeze(1)).double().cpu().numpy()
    del x
    return p, {"epochs": stats["epochs"], "best_validation": stats["best_validation"], "history": stats["history"]}


def s_phase_fit(cache: Cache, h: np.ndarray, arm: Arm, seed: int, device: str, *, n_boot: int = keep.BOOTSTRAP) -> tuple[dict, np.ndarray]:
    """Fit `arm` at `seed` on the runs outside S (early-stopped on S); return its JSON record and P(kill) on S's head rows."""
    config = arm_config(arm, device)
    outside, s_mask = ~cache.is_s, cache.is_s
    if arm.subsample is not None:
        outside = outside & subsample_runs(cache.hashes, arm.subsample)
    train_rows, s_rows = np.flatnonzero(outside), np.flatnonzero(s_mask)
    head_train = train_rows[cache.head_rows[train_rows]]
    head_s = s_rows[cache.head_rows[s_rows]]
    ev, tr = HeadRows.of(cache, s_rows), HeadRows.of(cache, train_rows)
    base_rate = float(tr.y.mean())
    block = np.zeros_like(h) if arm.zero_h else h
    on_cuda = torch.device(config.device).type == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    record: dict[str, Any] = {"arm": arm.name, "seed": seed, "adoptable": arm.adoptable, "kind": arm.kind, "device": config.device,
                              "cache_key": cache.manifest["key"], "config": dataclasses.asdict(config)}
    if arm.kind == "logit":
        p, extra = _fit_logit(cache, block, head_train, head_s)
        record.update(extra)
    elif arm.kind == "bce":
        p, extra = _fit_bce(cache, block, head_train, head_s, config, seed)
        record.update(extra)
        record["best_epoch"] = int(np.argmin([e["validation"] for e in extra["history"]])) + 1
    else:
        rows = np.concatenate([train_rows, s_rows])
        data = select_values(cache.values, rows, np.r_[np.zeros(len(train_rows), bool), np.ones(len(s_rows), bool)], block)
        dense, counts = value_tensors(dataclasses.replace(cache.values, fight_block=block), head_s, config.device)
        y_t = torch.from_numpy(ev.y).to(config.device)
        base = float(((base_rate - ev.y) ** 2).mean())

        def hook(_epoch: int, net: torch.nn.Module) -> dict[str, float]:
            check_stop()
            brier = float(((head_p(net, dense, counts) - y_t) ** 2).mean())
            return {"s_brier": brier, "s_bss_vs_base_rate": 1.0 - brier / base}

        torch.manual_seed(1000 + seed)
        generator = torch.Generator().manual_seed(1000 + seed)
        net = models.RunValueNet(hidden=config.hidden, dropout=config.dropout)
        stats = train.train_value_net(net, data, config, generator, epoch_end=hook)
        net.eval()
        p = head_p(net, dense.cpu(), counts.cpu()).numpy()
        history = stats["history"]
        record.update(epochs=stats["epochs"], best_validation=stats["best_validation"], history=history,
                      resident=stats.get("resident"), train_rows=int(len(train_rows)),
                      train_runs=int(len(np.unique(cache.values.run_index[train_rows]))))
        record["best_epoch"] = int(np.argmin([e["validation"] for e in history])) + 1
    record["fit_min"] = round((time.perf_counter() - t0) / 60, 3)
    record["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1) if on_cuda else None
    record["keep"] = {"act1_boss_kill": head_table(p, ev, tr, n_boot=n_boot)}
    record["s"] = {"head_rows": int(len(ev.y)), "positives": int(ev.y.sum()), "runs": int(len(np.unique(ev.run))), "base_rate": base_rate}
    return record, p


# --------------------------------------------------------------------------------------
# run / ensemble
# --------------------------------------------------------------------------------------


def fit_paths(cache: Cache, arm: str, seed: int) -> tuple[Path, Path]:
    return cache.folder / "fits" / f"{arm}-s{seed}.json", cache.folder / "fits" / f"{arm}-s{seed}.npz"


def ensure_fit(cache: Cache, arm: Arm, seed: int, device: str, h_cache: dict[int, np.ndarray]) -> dict:
    """Return the fit's record, fitting it (and the seed's FightNet) if its JSON does not exist yet."""
    json_path, npz_path = fit_paths(cache, arm.name, seed)
    if json_path.is_file():
        return gen_loop.read_json(json_path)
    try:
        check_stop()  # a fit boundary
        if seed not in h_cache:
            h_cache.clear()  # one seed's H at a time
            h_cache[seed] = fit_fightnet_h(cache, f"s{seed}", seed, None, device)  # a finished FightNet stays cached: it is not the arm's fit
            check_stop()  # the boundary between FightNet and the arm's fit
        record, p = s_phase_fit(cache, h_cache[seed], arm, seed, device)
    except StopFit:
        for partial in (npz_path, npz_path.with_name(npz_path.name + ".tmp"), json_path.with_name(json_path.name + ".tmp")):
            partial.unlink(missing_ok=True)  # the fit has no JSON (checked above), so anything under its names is a leftover
        raise
    record["preds"] = npz_path.name
    save_npz(npz_path, p=p.astype(np.float32))
    save_json(json_path, record)
    return record


def _summary_line(record: dict) -> str:
    head = record["keep"]["act1_boss_kill"]
    return (f"{record['arm']} s{record['seed']}: best epoch {record.get('best_epoch')}, S BSS vs base "
            f"{head['brier_skill_vs_base_rate']['value']:+.3f} vs V0 {head['brier_skill_vs_v0']['value']:+.3f}, "
            f"{record['fit_min']:.2f} min")


def cmd_run(args: argparse.Namespace) -> int:
    cache = load_cache(Path(args.cache))
    names = [a for a in ARMS if a != "ens5"] if args.arms == "all" else [a for a in args.arms.split(",") if a]
    arms = [resolve_arm(n) for n in names]
    started, h_cache = time.perf_counter(), {}
    _STOP["file"] = Path(args.stop_file) if getattr(args, "stop_file", None) else None
    try:
        for seed in range(args.seeds):
            for arm in arms:
                if arm.max_seeds is not None and seed >= arm.max_seeds:
                    continue
                if fit_paths(cache, arm.name, seed)[0].is_file():
                    continue
                if args.budget_min is not None and (time.perf_counter() - started) / 60 >= args.budget_min:
                    print(f"budget of {args.budget_min:g} min spent; stopping before {arm.name} s{seed}")
                    return 0
                print(_summary_line(ensure_fit(cache, arm, seed, args.device, h_cache)), flush=True)
    except StopFit:
        print(f"stop file seen: the fit in hand is dropped, nothing of it written; exit {STOP_EXIT}", flush=True)
        return STOP_EXIT
    finally:
        _STOP["file"] = None
    return 0


def cmd_ensemble(args: argparse.Namespace) -> int:
    cache = load_cache(Path(args.cache))
    arm, h_cache, members, preds = resolve_arm(args.arm), {}, [], []
    if arm.kind != "v":
        raise SystemExit("an ensemble averages V fits")
    for seed in range(5):
        members.append(ensure_fit(cache, arm, seed, args.device, h_cache))
        with np.load(fit_paths(cache, arm.name, seed)[1], allow_pickle=False) as data:
            preds.append(data["p"].astype(np.float64))
    s_rows = np.flatnonzero(cache.is_s)
    ev, tr = HeadRows.of(cache, s_rows), HeadRows.of(cache, np.flatnonzero(~cache.is_s))
    mean_p = np.mean(preds, axis=0)
    head = head_table(mean_p, ev, tr)
    member_brier = [float(((p - ev.y) ** 2).mean()) for p in preds]
    base_brier = float(((tr.y.mean() - ev.y) ** 2).mean())
    doc = {"arm": f"ens5-{arm.name}", "of": arm.name, "seeds": list(range(5)), "adoptable": False, "keep": {"act1_boss_kill": head},
           "member_brier": member_brier, "ensemble_brier": float(((mean_p - ev.y) ** 2).mean()), "base_rate_brier": base_brier,
           "member_bss_vs_base_rate": [1 - b / base_brier for b in member_brier],
           "ensemble_bss_vs_base_rate": 1 - float(((mean_p - ev.y) ** 2).mean()) / base_brier,
           "cache_key": cache.manifest["key"]}
    seed_variance = float(np.mean([(p - mean_p) ** 2 for p in preds]))
    doc["seed_variance"] = seed_variance  # mean over rows and seeds of (p_k - mean p)^2 = mean member Brier - ensemble Brier
    doc["variance_share"] = seed_variance / float(np.mean(member_brier)) if np.mean(member_brier) > 0 else None
    doc["variance_share_note"] = "seed variance as a share of the mean member Brier: what averaging the 5 seeds removes"
    save_json(cache.folder / "fits" / f"ens5-{arm.name}.json", doc)
    print(f"ens5-{arm.name}: BSS vs base rate {doc['ensemble_bss_vs_base_rate']:+.3f}; members {np.mean(doc['member_bss_vs_base_rate']):+.3f}")
    return 0


# --------------------------------------------------------------------------------------
# confirm: 5-fold cross-fit over the runs outside S
# --------------------------------------------------------------------------------------


def _confirm_fold(cache: Cache, h: np.ndarray, arm: Arm, fold: int, device: str) -> dict:
    """Fit `arm` on the runs outside S and fold `fold` (early-stopped on S); predict every row of the fold."""
    config = arm_config(arm, device)
    fold_of_rows = cache.fold
    train_rows = np.flatnonzero((fold_of_rows != fold) & ~cache.is_s)
    s_rows, fold_rows = np.flatnonzero(cache.is_s), np.flatnonzero(fold_of_rows == fold)
    data = select_values(cache.values, np.concatenate([train_rows, s_rows]),
                         np.r_[np.zeros(len(train_rows), bool), np.ones(len(s_rows), bool)], h)
    torch.manual_seed(2000 + fold)
    generator = torch.Generator().manual_seed(2000 + fold)
    net = models.RunValueNet(hidden=config.hidden, dropout=config.dropout)
    t0 = time.perf_counter()
    stats = train.train_value_net(net, data, config, generator)
    fit_min = (time.perf_counter() - t0) / 60
    kill, efloor = np.zeros(len(fold_rows)), np.zeros(len(fold_rows))
    net.eval()
    with torch.no_grad():
        for start in range(0, len(fold_rows), 8192):
            rows = fold_rows[start:start + 8192]
            dense, counts = models.value_inputs(np.concatenate([cache.values.bases.take(rows), h[rows]], axis=1))
            logits, _ = net(torch.from_numpy(dense), torch.from_numpy(counts))
            p = torch.softmax(logits, dim=1).numpy().astype(np.float64)
            kill[start:start + len(rows)] = p[:, FIRST_BOSS:].sum(axis=1)
            efloor[start:start + len(rows)] = models.expected_floors(p, EXPECTED_FLOOR_W)
    save_npz(cache.folder / "confirm" / f"{arm.name}-fold{fold}.npz", rows=fold_rows, kill=kill, efloor=efloor)
    return {"fold": fold, "epochs": stats["epochs"], "best_epoch": int(np.argmin([e["validation"] for e in stats["history"]])) + 1,
            "best_validation": stats["best_validation"], "fit_min": round(fit_min, 3), "train_rows": int(len(train_rows))}


def pooled_confirmation(cache: Cache, arm: Arm, fold_info: list[dict], *, n_boot: int = keep.BOOTSTRAP) -> dict[str, Any]:
    """Pool the out-of-fold predictions of `arm`: the keep table on the floor <= 16 rows, and the expected-floor error."""
    v, kill = cache.values, cache.kill
    rows_all, p_all, e_all, v0p_all, v0fp_all, base_all, rate_all, fold_rates = [], [], [], [], [], [], [], []
    for fold in range(FOLDS):
        with np.load(cache.folder / "confirm" / f"{arm.name}-fold{fold}.npz", allow_pickle=False) as z:
            rows, p, e = z["rows"], z["kill"], z["efloor"]
        train_rows = np.flatnonzero((cache.fold != fold) & ~cache.is_s)
        tr = HeadRows.of(cache, train_rows)
        fold_rates.append(float(tr.y.mean()))
        rate_all.append(np.full(len(rows), fold_rates[-1]))
        head = cache.head_rows[rows]
        v0_rows, floor_rows = cache.v0_score[rows], v.floor[rows].astype(np.float64)
        v0p = keep.v0_probability(tr.v0, tr.y, v0_rows)
        v0fp = keep.v0_probability(tr.v0 + tr.floor / keep.ACT1_LAST_FLOOR, tr.y, v0_rows + floor_rows / keep.ACT1_LAST_FLOOR)
        baseline = floor_baseline(np.column_stack([cache.v0_score[train_rows], v.floor[train_rows]]), v.final_floor[train_rows],
                                  np.column_stack([v0_rows, floor_rows]))
        rows_all.append(rows)
        p_all.append(p)
        e_all.append(e)
        v0p_all.append(v0p)
        v0fp_all.append(v0fp)
        base_all.append(baseline)
    rows, p, e = np.concatenate(rows_all), np.concatenate(p_all), np.concatenate(e_all)
    v0p, v0fp, baseline = np.concatenate(v0p_all), np.concatenate(v0fp_all), np.concatenate(base_all)
    head = cache.head_rows[rows]
    y, run = kill[rows][head], v.run_index[rows][head]
    base_rows = np.concatenate(rate_all)[head]  # each out-of-fold row's base rate is its fold's training rate
    table = keep.head_conditions(p[head], y, run, base_rows, v0p[head], n_boot=n_boot)  # (the base entries are replaced below)
    vs_base = keep.paired_bootstrap_skill(keep.brier_terms(p[head], y), keep.brier_terms(base_rows, y), run, n_boot)
    vs_base["pass"] = bool(vs_base["ci"][0] > 0.0 and vs_base["value"] > 0.0)
    ll_model, ll_base = keep.log_loss(p[head], y), keep.log_loss(base_rows, y)
    table["brier_skill_vs_base_rate"] = vs_base
    table["log_loss_vs_base_rate"] = {"model": ll_model, "base_rate": ll_base, "pass": bool(ll_model < ll_base)}
    table["base_rate"] = float(base_rows.mean())
    table["brier_skill_vs_v0_floor_informational"] = {
        **keep.paired_bootstrap_skill(keep.brier_terms(p[head], y), keep.brier_terms(v0fp[head], y), run, n_boot), "gated": False,
        "baseline": "V0 + floor / 17, through each fold's training-row quantile table"}
    final = v.final_floor[rows].astype(np.float64)
    model_err, base_err = np.abs(e - final), np.abs(baseline - final)
    mae = {"w": EXPECTED_FLOOR_W, "baseline": "least squares of the final floor on [V0 score, floor, 1], fitted on each fold's training rows"}
    for scope, mask in (("all_rows", np.ones(len(rows), bool)), ("floor_le_16_rows", head)):
        mae[scope] = {"rows": int(mask.sum()), "model": float(model_err[mask].mean()), "baseline": float(base_err[mask].mean()),
                      "skill_vs_baseline": keep.paired_bootstrap_skill(model_err[mask], base_err[mask], v.run_index[rows][mask], n_boot)}
    return {"arm": arm.name, "adoptable": arm.adoptable, "cache_key": cache.manifest["key"], "folds": fold_info,
            "oof_rows": int(len(rows)), "oof_runs": int(len(np.unique(v.run_index[rows]))), "keep": {"act1_boss_kill": table},
            "expected_floor_mae": mae, "fold_base_rates": fold_rates,
            "base_rate": "per fold: the rate over that fold's training rows (outside S and the fold, floor <= 16), one constant for each "
                         "out-of-fold row of the fold"}


MAX_OTHER_CONFIRMED = 2  # besides `lab`: the arms pre-declared from the S ranking (VRUN_SWEEP 1.2)


def confirmed_names(folder: Path) -> set[str]:
    """The arms with any confirmation output (a pooled JSON or a fold's) in the cache."""
    directory = Path(folder) / "confirm"
    return {path.stem.rsplit("-fold", 1)[0] for path in directory.glob("*.json")} if directory.is_dir() else set()


def check_predeclared(existing: set[str], requested: list[str]) -> None:
    """Refuse a confirmation that would leave more than `lab` plus two other arms confirmed in this cache."""
    others = (set(existing) | set(requested)) - {"lab"}
    if len(others) > MAX_OTHER_CONFIRMED:
        raise SystemExit(f"only lab and {MAX_OTHER_CONFIRMED} other arms are confirmed (VRUN_SWEEP 1.2); this would confirm "
                         f"{', '.join(sorted(others))}")


def cmd_confirm(args: argparse.Namespace) -> int:
    if args.seeds != 1:
        raise SystemExit("a confirmation is one seed per arm (--seeds 1)")
    cache = load_cache(Path(args.cache))
    arms = [resolve_arm(n) for n in args.arms.split(",") if n]
    for arm in arms:
        if arm.name != "lab" and not arm.adoptable:
            raise SystemExit(f"{arm.name} is not adoptable; only lab and the adoptable arms are confirmed")
        if arm.kind != "v" or arm.device:
            raise SystemExit(f"{arm.name} is not a V fit")
    check_predeclared(confirmed_names(cache.folder), [arm.name for arm in arms])
    info: dict[str, list[dict]] = {arm.name: [] for arm in arms}
    for fold in range(FOLDS):
        pending = [a for a in arms if not (cache.folder / "confirm" / f"{a.name}-fold{fold}.json").is_file()]
        h = fit_fightnet_h(cache, f"fold{fold}", 100 + fold, fold, args.device) if pending else None
        for arm in arms:
            meta_path = cache.folder / "confirm" / f"{arm.name}-fold{fold}.json"
            if arm in pending:
                meta = _confirm_fold(cache, h, arm, fold, args.device)
                save_json(meta_path, meta)
                print(f"{arm.name} fold {fold}: best epoch {meta['best_epoch']}, {meta['fit_min']:.2f} min", flush=True)
            info[arm.name].append(gen_loop.read_json(meta_path))
    for arm in arms:
        doc = pooled_confirmation(cache, arm, info[arm.name])
        save_json(cache.folder / "confirm" / f"{arm.name}.json", doc)
        head = doc["keep"]["act1_boss_kill"]
        print(f"{arm.name}: OOF BSS vs base {head['brier_skill_vs_base_rate']['value']:+.3f} {head['brier_skill_vs_base_rate']['ci']} "
              f"vs V0 {head['brier_skill_vs_v0']['value']:+.3f} {head['brier_skill_vs_v0']['ci']}")
    return 0


# --------------------------------------------------------------------------------------
# table
# --------------------------------------------------------------------------------------


def _ms(values: list[float], digits: int = 3) -> str:
    if not values:
        return "-"
    return f"{np.mean(values):+.{digits}f} ({min(values):+.{digits}f}..{max(values):+.{digits}f})" if len(values) > 1 else f"{values[0]:+.{digits}f}"


def _bins_text(head: dict) -> str:
    bins = head["ece_bins_ge_30_positives"]["bins"]
    return " ".join(f"{b['bin']}:{b['n']}/{b['positives']}/{b['mean_pred']:.3f}/{b['freq']:.3f}{'*' if b['eligible'] else ''}"
                    for b in bins if b["n"])


def s_row(fits: list[dict]) -> str:
    """One markdown row for an arm over its seeds (mean, and min..max for the skill scores)."""
    heads = [f["keep"]["act1_boss_kill"] for f in fits]
    ece = [h["ece_bins_ge_30_positives"] for h in heads]
    epochs = [f["best_epoch"] for f in fits if f.get("best_epoch") is not None]
    val = [f["best_validation"] for f in fits if f.get("best_validation") is not None]
    ece_vals = [e["value"] for e in ece if e["value"] is not None]
    model_ll = np.mean([h["log_loss_vs_base_rate"]["model"] for h in heads])
    base_ll = np.mean([h["log_loss_vs_base_rate"]["base_rate"] for h in heads])
    epoch_text = f"{np.mean(epochs):.1f}" if epochs else "-"
    val_text = f"{np.mean(val):.4f}" if val else "-"
    ece_text = f"{np.mean(ece_vals) * 100:.1f} pp" if ece_vals else "none"
    adoptable = "yes" if fits[0]["adoptable"] else "no"
    return (f"| {fits[0]['arm']} | {adoptable} | {len(fits)} | {epoch_text} | {val_text} | "
            f"{_ms([h['brier_skill_vs_base_rate']['value'] for h in heads])} | {_ms([h['brier_skill_vs_v0']['value'] for h in heads])} | "
            f"{model_ll:.4f} / {base_ll:.4f} | {ece_text} ({np.mean([e['eligible_bins'] for e in ece]):.1f} bins) | "
            f"{np.mean([f['fit_min'] for f in fits]):.2f} |")


def confirmation_pass(doc: dict, fits: list[dict]) -> tuple[bool, str]:
    """Section 1.5's rule: both skill CIs above 0, log-loss under the base rate, ECE <= 3 pp, and every S seed positive."""
    head = doc["keep"]["act1_boss_kill"]
    three = [f for f in fits if f["seed"] < 3]
    checks = {
        "BSS vs base CI > 0": head["brier_skill_vs_base_rate"]["ci"][0] > 0,
        "BSS vs V0 CI > 0": head["brier_skill_vs_v0"]["ci"][0] > 0,
        "log-loss < base": head["log_loss_vs_base_rate"]["pass"],
        "ECE <= 3 pp": head["ece_bins_ge_30_positives"]["pass"],
        "each of the 3 S seeds BSS > 0": len(three) == 3 and all(f["keep"]["act1_boss_kill"]["brier_skill_vs_base_rate"]["value"] > 0 for f in three),
        "adoptable": bool(doc.get("adoptable")),
    }
    return all(checks.values()), ", ".join(f"{k}: {'ok' if v else 'FAIL'}" for k, v in checks.items())


def build_report(cache: Cache) -> str:
    fits_dir = cache.folder / "fits"
    by_arm: dict[str, list[dict]] = {}
    for path in sorted(fits_dir.glob("*-s[0-9]*.json")):
        record = gen_loop.read_json(path)
        by_arm.setdefault(record["arm"], []).append(record)
    order = {name: i for i, name in enumerate(ARMS)}
    names = sorted(by_arm, key=lambda n: (order.get(n, len(order)), n))
    cap, fights = cache.manifest.get("fight_cap", train.TrainConfig().fight_cap), cache.manifest["fights"]
    fight_note = (f"FightNet data: {fights} fights against the lab's cap of {cap}: "
                  + ("the same fights the lab trains FightNet on (the cap drops none)" if fights <= cap
                     else "above the cap, so the lab drops the oldest and this sweep does not"))
    lines = ["# Report (numbers only)", "", f"cache {cache.folder.name}, generation {cache.manifest['generation']}, "
             f"{cache.manifest['runs']} runs, {cache.manifest['rows']} rows", "", fight_note, "",
             "## S table (selection split, hash in [0.2, 0.3); head = floor <= 16 rows; skill = Brier skill, mean (min..max) over seeds)", "",
             "| arm | adoptable | seeds | best epoch | val loss | BSS vs base rate | BSS vs V0 | log-loss model / base | ECE (eligible bins) | fit min |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [s_row(by_arm[n]) for n in names]
    lines += ["", "ECE bins per fit (bin:rows/positive rows/mean prediction/frequency; * = eligible, 30 positive runs):", ""]
    lines += [f"- {f['arm']} s{f['seed']}: {_bins_text(f['keep']['act1_boss_kill'])}" for n in names for f in by_arm[n]]
    for ens in sorted(fits_dir.glob("ens5-*.json")):
        doc = gen_loop.read_json(ens)
        h = doc["keep"]["act1_boss_kill"]
        lines += ["", f"Ensemble {doc['arm']}: BSS vs base rate {h['brier_skill_vs_base_rate']['value']:+.3f} {h['brier_skill_vs_base_rate']['ci']}, "
                  f"vs V0 {h['brier_skill_vs_v0']['value']:+.3f}; members {np.mean(doc['member_bss_vs_base_rate']):+.3f}; "
                  f"variance share {doc['variance_share']}"]
    lab, cpu = by_arm.get("lab", []), by_arm.get("lab-cpu", [])
    if lab and cpu:
        gpu0 = next((f for f in lab if f["seed"] == 0), lab[0])
        lines += ["", "Parity (seed 0; judged against the 3-seed spread, not bit for bit): "
                  + "; ".join(f"{f['arm']} best epoch {f.get('best_epoch')}, BSS vs base "
                              f"{f['keep']['act1_boss_kill']['brier_skill_vs_base_rate']['value']:+.3f}, vs V0 "
                              f"{f['keep']['act1_boss_kill']['brier_skill_vs_v0']['value']:+.3f}, {f['fit_min']:.2f} min"
                              for f in (gpu0, cpu[0]))]
    curve = [(n, by_arm[n]) for n in ("sub25", "sub50", "sub75", "lab") if n in by_arm]
    if curve:
        lines += ["", "Data curve (BSS vs base rate on S): " + ", ".join(
            f"{n} {_ms([f['keep']['act1_boss_kill']['brier_skill_vs_base_rate']['value'] for f in fs])}" for n, fs in curve)]
    confirmed = sorted((cache.folder / "confirm").glob("*.json")) if (cache.folder / "confirm").is_dir() else []
    confirmed = [p for p in confirmed if "-fold" not in p.name]
    if confirmed:
        lines += ["", "## Confirmation table (5-fold cross-fit, pooled out-of-fold floor <= 16 rows)", "",
                  "| arm | BSS vs base rate [CI] | BSS vs V0 [CI] | log-loss model / base | ECE | expected-floor MAE model / V0+floor | rule 1.5 |",
                  "|---|---|---|---|---|---|---|"]
        details = []
        for path in confirmed:
            doc = gen_loop.read_json(path)
            h = doc["keep"]["act1_boss_kill"]
            ok, why = confirmation_pass(doc, by_arm.get(doc["arm"], []))
            e = h["ece_bins_ge_30_positives"]
            mae = doc["expected_floor_mae"]["all_rows"]
            ece_text = f"{e['value'] * 100:.1f} pp" if e["value"] is not None else "none"
            lines.append(
                f"| {doc['arm']} | {h['brier_skill_vs_base_rate']['value']:+.3f} [{h['brier_skill_vs_base_rate']['ci'][0]:+.3f}, "
                f"{h['brier_skill_vs_base_rate']['ci'][1]:+.3f}] | {h['brier_skill_vs_v0']['value']:+.3f} "
                f"[{h['brier_skill_vs_v0']['ci'][0]:+.3f}, {h['brier_skill_vs_v0']['ci'][1]:+.3f}] | "
                f"{h['log_loss_vs_base_rate']['model']:.4f} / {h['log_loss_vs_base_rate']['base_rate']:.4f} | "
                f"{ece_text} ({e['eligible_bins']} bins) | "
                f"{mae['model']:.3f} / {mae['baseline']:.3f} | {'PASS' if ok else 'fail'} |")
            details.append(f"- {doc['arm']} bins: {_bins_text(h)}; rule: {why}")
        lines += ["", *details]
    return "\n".join(lines) + "\n"


def cmd_table(args: argparse.Namespace) -> int:
    cache = load_cache(Path(args.cache))
    text = build_report(cache)
    (cache.folder / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


# --------------------------------------------------------------------------------------
# beside: one fit while a self-play block plays
# --------------------------------------------------------------------------------------


def running_selfplay_job(lab: Path, *, now: float | None = None, fresh_s: float = 60.0) -> Path | None:
    """Return the newest self-play job folder with no `done.rc` whose server stats file changed within `fresh_s`."""
    now = time.time() if now is None else now
    for job in sorted((p for p in (Path(lab) / "runs").glob("*selfplay*") if p.is_dir()), reverse=True):
        stats = job / "prior_server_stats.json"
        if not (job / "done.rc").exists() and stats.is_file() and now - stats.stat().st_mtime <= fresh_s:
            return job
    return None


def read_server_stats(path: Path) -> dict | None:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        return {"t": time.time(), "calls": int(doc["calls"]), "uptime_s": float(doc["uptime_s"]),
                "total_p99_ms": float(doc["total_ms"]["p99"])}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def summarise_samples(samples: list[dict]) -> dict[str, Any]:
    if not samples:
        return {"samples": 0}
    out: dict[str, Any] = {"samples": len(samples), "p99_ms_mean": float(np.mean([s["total_p99_ms"] for s in samples])),
                           "p99_ms_max": float(max(s["total_p99_ms"] for s in samples))}
    if len(samples) > 1 and samples[-1]["uptime_s"] > samples[0]["uptime_s"]:
        out["calls_per_s"] = (samples[-1]["calls"] - samples[0]["calls"]) / (samples[-1]["uptime_s"] - samples[0]["uptime_s"])
    return out


def cmd_beside(args: argparse.Namespace) -> int:
    cache = load_cache(Path(args.cache))
    job = running_selfplay_job(Path(args.lab))
    if job is None:
        print("no self-play job is running (no folder without done.rc with a fresh server stats file); skipping the beside check")
        return 0
    stats_path = job / "prior_server_stats.json"
    h = fit_fightnet_h(cache, "s0", 0, None, args.device)
    phase, samples, stop = {"name": "before"}, [], threading.Event()

    def sample() -> None:
        while not stop.is_set():
            if (s := read_server_stats(stats_path)) is not None:
                samples.append({**s, "phase": phase["name"]})
            stop.wait(args.every_s)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    time.sleep(args.before_s)
    phase["name"] = "during"
    on_cuda = torch.device(args.device).type == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    record, _ = s_phase_fit(cache, h, ARMS["lab"], 0, args.device)
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if on_cuda else None
    phase["name"] = "after"
    stop.set()
    thread.join()
    before = summarise_samples([s for s in samples if s["phase"] == "before"])
    during = summarise_samples([s for s in samples if s["phase"] == "during"])
    bound = 20.0
    doc = {"job": job.name, "device": args.device, "fit_min": record["fit_min"],
           "gpu_peak_mb": None if peak_mb is None else round(peak_mb, 1), "before": before, "during": during,
           "p99_bound_ms": bound,
           "verdict": {
               "server_p99_under_bound": bool(during["p99_ms_max"] < bound) if during.get("samples") else None,
               "server_p99_max_during_ms": during.get("p99_ms_max"),
               "client_p99_under_bound": None,
               "client_p99_note": "not measurable live: the server's stats hold no client latency; read the block's summary.json "
                                  "when it ends (the client latency and prior_server_degraded at the 20 ms bound)",
               "decisions_per_s": None,
               "decisions_per_s_note": "not measurable live: the server's calls per second (before and during above) is only a proxy; "
                                       "compare the block's decisions per second in its summary.json against the 9.1 reference"},
           "samples": [{k: v for k, v in s.items() if k != "t"} for s in samples]}
    save_json(cache.folder / "beside.json", doc)
    print(json.dumps({k: v for k, v in doc.items() if k != "samples"}, indent=1))
    return 0


# --------------------------------------------------------------------------------------
# cache command and main
# --------------------------------------------------------------------------------------


def cmd_cache(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    folder = build_cache(Path(args.lab), Path(args.agent_root), Path(args.out), closed_after_min=args.closed_after_min)
    manifest = gen_loop.read_json(folder / "manifest.json")
    print(json.dumps({"cache": folder.name, "census": {k: v for k, v in manifest["census"].items() if k != "per_block"},
                      "per_block": manifest["census"]["per_block"], "skipped_blocks": manifest["skipped_blocks"],
                      "counts": manifest["counts"], "rows": manifest["rows"], "runs": manifest["runs"], "fights": manifest["fights"]},
                     indent=1))
    print(f"peak RSS of the build: {manifest['build']['peak_rss_mb']} MB; this call: {(time.perf_counter() - t0) / 60:.1f} min wall "
          f"(build {manifest['build']['wall_min']} min)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("cache")
    p.add_argument("--lab", required=True)
    p.add_argument("--agent-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--closed-after-min", type=float, default=CLOSED_AFTER_MIN)
    p.set_defaults(func=cmd_cache)
    p = sub.add_parser("run")
    p.add_argument("--cache", required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--budget-min", type=float, default=None)
    p.add_argument("--stop-file", default=None, help="drop the fit in hand and exit 3 when this path exists (checked at every epoch end)")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("ensemble")
    p.add_argument("--cache", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=cmd_ensemble)
    p = sub.add_parser("confirm")
    p.add_argument("--cache", required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=cmd_confirm)
    p = sub.add_parser("table")
    p.add_argument("--cache", required=True)
    p.set_defaults(func=cmd_table)
    p = sub.add_parser("beside")
    p.add_argument("--cache", required=True)
    p.add_argument("--lab", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--before-s", type=float, default=20.0)
    p.add_argument("--every-s", type=float, default=5.0, help="seconds between reads of the server's stats file")
    p.set_defaults(func=cmd_beside)
    args = parser.parse_args(argv)
    _lower_priority()
    torch.set_num_threads(THREADS)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
