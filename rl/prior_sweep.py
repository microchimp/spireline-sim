"""The combat prior's recipe sweep on the idle GPU (docs/rl/PRIOR_SWEEP.md; GPU lane, week 4 item 2).

    prior_sweep.py build  --lab LAB --agent-root AGENT --out PRIOR [--build-seeds 8,9]   evaluation set + shared training sets
    prior_sweep.py run    --out PRIOR --arms a,b|all --seeds 0,1 --device cuda [--stop-file P] [--budget-min N]
    prior_sweep.py run    --out PRIOR --confirm ARM --seeds 10,11,12 --device cuda          fresh fits into ARM/confirm-s<seed>/
    prior_sweep.py search --out PRIOR --arms a,b,c|auto --procs 8 --reps 24                 dev-set search with each prior
    prior_sweep.py table  --out PRIOR                                                       report.md: cells, main effects, A/B reading
    prior_sweep.py queue  --out PRIOR --queue QUEUE.json --repo FORK                        the sweep's jobs for the quiet-window runner

`build` carves the evaluation set out of the current window first: every run whose seed hash (`run_train.train._unit_hash`) falls in
[0.0, 0.1), never sampled into a training set. The training sets are `gen_loop.build_training_set` on the other runs, as the lab builds
it (6,400 fights, and 12,800 for the data arm, build seed 7; further build seeds for `lab-bs<N>`). `run` fits one arm and torch seed at a
time through `gen_loop.train_combat_prior` (2 CPU threads, BelowNormal) into a temporary folder renamed to `<arm>/s<seed>/` on success;
the stop file is looked at every epoch end and the folder is removed (exit 3). Outputs hold numbers and names, never absolute paths.
"""

from __future__ import annotations

import os as _os

if __name__ == "__main__":  # D31: run as a command this is self-play-grade work: old engine, whatever STS2_ENGINE says
    _os.environ.pop("STS2_ENGINE", None)

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")  # numpy's BLAS stays within the two threads a fit may use

import argparse
import hashlib
import itertools
import json
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

for _p in (Path(__file__).resolve().parents[1] / "src", Path(__file__).resolve().parent):  # sts2_gym, then the rl modules
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import gen_loop  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "prior-sweep/1"
FIT_FORMAT = "prior-sweep-fit/1"
EVAL_RANGE = (0.0, 0.1)  # the evaluation runs: seed hash in [0.0, 0.1)
BUILD_SEED = 7  # the shared training sets' build seed (the lab's own rule is seed = generation)
EXTRA_BUILD_SEEDS = (8, 9)  # `lab-bs<N>`: the lab's cell on a training set rebuilt with seed N
SIZES = (6400, 12800)
FIGHT_KEY = 4096  # gen_loop.build_training_set: a fight is (run, fight_ix) = run * 4096 + fight_ix
TARGETS = {"visits": ("visits", None), "cq1": ("completed_q", 1.0), "cq3": ("completed_q", 3.0)}
EPOCHS = (16, 32)
D_MODELS = (128, 256)
LAYERS = (2, 4)
LAB_ARM = "visits-16-128-2"  # the lab's cell
DATA_ARM = "data12800"
EXTRA_TORCH_SEEDS = (2, 3)
CONFIRM_SEEDS = (10, 11, 12)
THREADS = 2
STOP_EXIT = 3  # quiet_runner.STOP_EXIT: a fit dropped because the stop file appeared
EVAL_BATCH = 512
EXCEEDS = 2.0  # an effect "moves the prior" when it exceeds this many standard errors
TOKEN_KEYS = gen_loop.TOKEN_KEYS
ROW_KEYS = gen_loop.ROW_KEYS
SEED_RULES = {
    "build_seed": "the shared training set is built once with the build seed in the recipe (the lab builds with seed = generation "
                  "and resamples its fights each generation)",
    "torch_seed": "torch.manual_seed(seed) sets the weight initialisation; the train/validation split (85/15 by run) and the epoch "
                  "order come from a generator seeded 0 inside train_combat, so a torch seed varies the initialisation only",
}


class StopFit(Exception):
    """The stop file exists: the fit in hand is dropped (raised at an epoch end or between phases, caught by `cmd_run`)."""


# --------------------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    name: str
    target: str
    epochs: int
    d_model: int
    layers: int
    size: int = 6400
    build_seed: int = BUILD_SEED

    @property
    def policy_target(self) -> str:
        return TARGETS[self.target][0]

    @property
    def q_temperature(self) -> float | None:
        return TARGETS[self.target][1]

    @property
    def data_name(self) -> str:
        return f"train-{self.size}" + ("" if self.build_seed == BUILD_SEED else f"-bs{self.build_seed}")


_CELL = re.compile(r"^(visits|cq1|cq3)-(\d+)-(\d+)-(\d+)$")
_BUILD = re.compile(r"^lab-bs(\d+)$")


def cell_name(target: str, epochs: int, d_model: int, layers: int) -> str:
    return f"{target}-{epochs}-{d_model}-{layers}"


def grid_names() -> list[str]:
    """The 24 cells, target-major, then epochs, `d_model`, layers (the lab's cell first)."""
    return [cell_name(t, e, d, n) for t, e, d, n in itertools.product(TARGETS, EPOCHS, D_MODELS, LAYERS)]


def resolve_arm(name: str) -> Arm:
    """`{target}-{epochs}-{d_model}-{layers}`, `lab-bs<N>` (the lab's cell, build seed N) or `data12800` (the lab's cell on 12,800 fights)."""
    if found := _CELL.match(name):
        target, epochs, d_model, layers = found[1], int(found[2]), int(found[3]), int(found[4])
        if epochs not in EPOCHS or d_model not in D_MODELS or layers not in LAYERS:
            raise ValueError(f"{name!r} is not a cell of the grid (epochs {EPOCHS}, d_model {D_MODELS}, layers {LAYERS})")
        return Arm(name, target, epochs, d_model, layers)
    if found := _BUILD.match(name):
        return Arm(name, "visits", EPOCHS[0], D_MODELS[0], LAYERS[0], build_seed=int(found[1]))
    if name == DATA_ARM:
        return Arm(name, "visits", EPOCHS[0], D_MODELS[0], LAYERS[0], size=12800)
    raise ValueError(f"unknown arm {name!r}: a cell like {LAB_ARM}, lab-bs<N> or {DATA_ARM}")


def single_factor_cells() -> list[str]:
    """The lab's cell's neighbours that differ in one factor, in factor order (target levels, epochs, d_model, layers)."""
    lab = resolve_arm(LAB_ARM)
    others = [cell_name(t, lab.epochs, lab.d_model, lab.layers) for t in TARGETS if t != lab.target]
    return [*others, cell_name(lab.target, EPOCHS[1], lab.d_model, lab.layers), cell_name(lab.target, lab.epochs, D_MODELS[1], lab.layers),
            cell_name(lab.target, lab.epochs, lab.d_model, LAYERS[1])]


def all_order(seeds: Sequence[int] = (0, 1), extra_build_seeds: Sequence[int] = EXTRA_BUILD_SEEDS) -> list[tuple[str, int]]:
    """Every (arm, torch seed) of the sweep in the order `run --arms all` fits them: the lab's cell, its extra torch seeds and the
    `lab-bs<N>` cells, the single-factor cells, the rest of the grid, then the data arm; an interrupted sweep keeps the readable part."""
    seeds = list(seeds)
    order: list[tuple[str, int]] = [(LAB_ARM, s) for s in seeds]
    order += [(LAB_ARM, s) for s in EXTRA_TORCH_SEEDS if s not in seeds]
    order += [(f"lab-bs{n}", 0) for n in extra_build_seeds]
    singles = single_factor_cells()
    order += [(name, s) for name in singles for s in seeds]
    order += [(name, s) for name in grid_names() if name != LAB_ARM and name not in singles for s in seeds]
    order += [(DATA_ARM, s) for s in seeds]
    return order


def build_seeds_of(out: Path) -> list[int]:
    """The extra build seeds `build` made training sets for (the defaults when nothing is built yet)."""
    try:
        sets = gen_loop.read_json(Path(out) / "manifest.json")["sets"]
    except (OSError, ValueError, KeyError):
        return list(EXTRA_BUILD_SEEDS)
    return sorted(int(m[1]) for name in sets if (m := re.match(r"^train-\d+-bs(\d+)$", name)))


def fit_dir(out: Path, arm: str, seed: int, confirm: bool = False) -> Path:
    return Path(out) / arm / (f"confirm-s{seed}" if confirm else f"s{seed}")


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON")


def _clean(value: Any) -> Any:
    """NaN and infinity are not JSON: they become null."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def save_json(path: Path, doc: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_clean(doc), indent=1, default=_json_default, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as blob:
        return {k: blob[k] for k in blob.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lower_priority() -> None:
    try:
        from sts2_gym import procutil  # noqa: PLC0415

        procutil.lower_priority()
    except Exception as error:  # noqa: BLE001 -- a failed drop must not stop a fit, but it must not pass unseen either
        print(f"prior_sweep: could not lower the process priority ({type(error).__name__}); running at normal priority", file=sys.stderr)


def check_stop(path: Path | None) -> None:
    if path is not None and Path(path).exists():
        raise StopFit("the stop file exists")


def stop_hook(path: Path | None):
    """The `epoch_end` hook of `train_combat.train`: raises `StopFit` when the stop file exists."""

    def hook(_epoch: int, _stats: dict[str, float]) -> None:
        check_stop(path)

    return hook


# --------------------------------------------------------------------------------------
# build: the evaluation set and the training sets
# --------------------------------------------------------------------------------------


def runs_in_range(seeds: np.ndarray, bounds: tuple[float, float] = EVAL_RANGE) -> np.ndarray:
    """Per run (a part's `seeds`): whether the hash of its seed falls in [lo, hi)."""
    from run_train import train  # noqa: PLC0415

    hashes = np.array([train._unit_hash(str(s)) for s in seeds], dtype=np.float64)  # noqa: SLF001
    return (hashes >= bounds[0]) & (hashes < bounds[1])


def take_rows(blob: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    """The samples of `blob` where `keep` (one flag per sample); per-run arrays (`seeds`, `prior_path`) and scalars are left whole,
    so a run keeps its index in the part and ids stay comparable between the evaluation and training copies."""
    tokens = np.repeat(keep, blob["lengths"])
    out = {}
    for key, value in blob.items():
        if key in TOKEN_KEYS:
            out[key] = value[tokens]
        elif key in ROW_KEYS or key == "fights":
            out[key] = value[keep]
        else:
            out[key] = value
    return out


def split_window(manifests: list[tuple[Path, dict]], tmp: Path, bounds: tuple[float, float] = EVAL_RANGE
                 ) -> tuple[list[tuple[Path, dict]], list[tuple[Path, dict]]]:
    """Write each combat part twice under `tmp`: the samples of the evaluation runs, and those of every other run. Both copies keep
    every part (an empty one stays), so `build_training_set` numbers the runs the same way on either side and its dedupe by
    (generation, seed) acts on whole seeds. Returns the two manifest lists to hand to `build_training_set`."""
    sides: dict[str, list[tuple[Path, dict]]] = {"eval": [], "train": []}
    for index, (directory, manifest) in enumerate(manifests):
        names = list(manifest.get("combat_parts", []))
        folders = {side: tmp / side / f"{index:03d}" for side in sides}
        for folder in folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            blob = gen_loop.read_part(directory / name)
            if "lengths" in blob:
                in_eval = runs_in_range(blob["seeds"], bounds)[blob["fights"]] if blob["lengths"].size else np.zeros(0, dtype=bool)
                copies = {"eval": take_rows(blob, in_eval), "train": take_rows(blob, ~in_eval)}
            else:
                copies = {"eval": blob, "train": blob}
            for side, copy in copies.items():
                with (folders[side] / name).open("wb") as handle:
                    np.savez_compressed(handle, **copy)
        for side, folder in folders.items():
            sides[side].append((folder, {"generation": int(manifest["generation"]), "combat_parts": names}))
    return sides["eval"], sides["train"]


def set_counts(data: dict[str, np.ndarray]) -> dict[str, Any]:
    runs = data["fights"]
    fights = runs.astype(np.int64) * FIGHT_KEY + data["_fight_ix"].astype(np.int64)
    return {"samples": int(data["lengths"].size), "runs": int(np.unique(runs).size), "fights": int(np.unique(fights).size)}


def eval_arrays(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """What the evaluation set keeps: the tokens, lengths, legal masks, visits and values, and the run and fight ids."""
    out = {k: data[k] for k in (*TOKEN_KEYS, "lengths", "legal", "visits", "qvals")}
    out["runs"] = data["fights"].astype(np.int64)
    out["fights"] = data["fights"].astype(np.int64) * FIGHT_KEY + data["_fight_ix"].astype(np.int64)
    return out


def window_key(blocks: list[tuple[Path, dict, str]], lab: Path, extra_build_seeds: Sequence[int]) -> str:
    """sha256 over the sorted (manifest path relative to the lab, combat part names, part sizes) of the blocks and the build rules."""
    entries = []
    for directory, manifest, _ in blocks:
        names = list(manifest.get("combat_parts", []))
        entries.append([(directory / "shards.json").relative_to(lab).as_posix(), names, [(directory / n).stat().st_size for n in names]])
    rules = {"eval_range": list(EVAL_RANGE), "build_seed": BUILD_SEED, "sizes": list(SIZES), "extra_build_seeds": list(extra_build_seeds)}
    return hashlib.sha256(json.dumps([sorted(entries), rules]).encode("utf-8")).hexdigest()


def build(lab: Path, agent_root: Path, out: Path, *, extra_build_seeds: Sequence[int] = EXTRA_BUILD_SEEDS,
          closed_after_min: float | None = None, now: float | None = None) -> dict[str, Any]:
    """Build `out`'s evaluation set and training sets from the current window's closed blocks; returns the manifest."""
    import vrun_sweep  # noqa: PLC0415

    t0 = time.perf_counter()
    lab, agent_root, out = Path(lab), Path(agent_root), Path(out)
    generation = int(gen_loop.read_json(agent_root / "current.json")["generation"])
    blocks, skipped = vrun_sweep.closed_blocks(
        gen_loop.window_manifests(lab / "runs", generation), lab, now=time.time() if now is None else now,
        closed_after_s=(vrun_sweep.CLOSED_AFTER_MIN if closed_after_min is None else closed_after_min) * 60)
    if not blocks:
        raise ValueError(f"no closed block of generations {generation - gen_loop.GEN_WINDOW + 1}..{generation} under the lab")
    key = window_key(blocks, lab, extra_build_seeds)
    done = out / "manifest.json"
    if done.is_file():
        old = gen_loop.read_json(done)
        if old.get("key") != key:
            raise ValueError("the window or the build rules changed since this folder was built; use a new --out")
        print("the evaluation set and training sets exist; nothing to build")
        return old
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out, prefix="split-") as scratch:
        eval_mans, train_mans = split_window([(d, m) for d, m, _ in blocks], Path(scratch))
        evaluation = gen_loop.build_training_set(eval_mans, max_fights=2**31 - 1, seed=0)
        save_npz(out / "eval.npz", eval_arrays(evaluation))
        manifest: dict[str, Any] = {
            "format": FORMAT, "key": key, "generation": generation, "build_seed": BUILD_SEED, "eval_range": list(EVAL_RANGE),
            "blocks": [{"manifest": (d / "shards.json").relative_to(lab).as_posix(), "generation": int(m["generation"]),
                        "closed_by": why, "parts": len(m.get("combat_parts", []))} for d, m, why in blocks],
            "skipped_blocks": skipped,
            "eval": {**set_counts(evaluation), "runs_distinct_in_window": int(evaluation["_runs_distinct"][0]),
                     "sha256": sha256_file(out / "eval.npz")},
            "sets": {},
        }
        jobs = [(f"train-{size}", size, BUILD_SEED) for size in SIZES]
        jobs += [(f"train-{SIZES[0]}-bs{n}", SIZES[0], n) for n in extra_build_seeds]
        for name, size, seed in jobs:
            data = gen_loop.build_training_set(train_mans, max_fights=size, seed=seed)
            save_npz(out / f"{name}.npz", {k: v for k, v in data.items() if not k.startswith("_")})
            manifest["sets"][name] = {
                "max_fights": size, "build_seed": seed, **set_counts(data), "fights_available": int(data["_fights_available"][0]),
                "samples_by_generation": {str(int(g)): int(n) for g, n in data["_samples_by_generation"]},
                "runs_dropped_as_copies": int(data["_runs_dropped"][0]), "sha256": sha256_file(out / f"{name}.npz")}
    manifest["build"] = {"wall_s": round(time.perf_counter() - t0, 1)}
    save_json(done, manifest)  # last: a folder with a manifest is a whole build
    return manifest


def cmd_build(a: argparse.Namespace) -> int:
    manifest = build(Path(a.lab), Path(a.agent_root), Path(a.out), extra_build_seeds=[int(x) for x in a.build_seeds.split(",") if x])
    counts = {"generation": manifest["generation"], "blocks": len(manifest["blocks"]), "eval": manifest["eval"],
              "sets": {k: {c: v[c] for c in ("samples", "runs", "fights", "fights_available")} for k, v in manifest["sets"].items()},
              "wall_s": manifest["build"]["wall_s"]}
    print(json.dumps(counts, indent=1))
    return 0


# --------------------------------------------------------------------------------------
# The measures
# --------------------------------------------------------------------------------------


def completed_q(visits: np.ndarray, legal: np.ndarray, qvals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The searches' own values per decision, as `train_combat.completed_q_target` and `a2c_target.heldout` define them: a visited
    legal action keeps its mean value, an unvisited legal one takes the root value (the visit-weighted mean of the visited ones), an
    illegal one is -inf. Returns (completed Q, the rows that have a visited legal action: the others carry no signal)."""
    ok = legal > 0
    q = np.asarray(qvals, dtype=np.float64)
    seen = ok & (np.where(ok, visits, 0.0) > 0) & np.isfinite(q)
    w = np.where(seen, np.where(ok, visits, 0.0), 0.0).astype(np.float64)
    keep = w.sum(axis=1) > 0
    root = (w * np.where(seen, q, 0.0)).sum(axis=1, keepdims=True) / np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
    return np.where(ok, np.where(seen, q, root), -np.inf), keep


def regret_measures(probs: np.ndarray, legal: np.ndarray, visits: np.ndarray, qvals: np.ndarray) -> dict[str, float]:
    """Measures 1 and 2 of PRIOR_SWEEP.md section 4 on the rows with a visited legal action, in the values' own units (HP):
    the expected completed-Q regret of the net's policy (max over legal actions of completed Q minus the policy-weighted mean:
    it rewards sharpness, so it ranks within one target level only), the top-1 regret of the net's argmax (comparable across
    targets), the share of rows where that argmax is the most-visited action, and the policy's entropy in nats."""
    cq, keep = completed_q(visits, legal, qvals)
    ok = legal > 0
    p = np.where(ok, np.nan_to_num(np.asarray(probs, dtype=np.float64)), 0.0)
    best = np.where(keep, cq.max(axis=1), 0.0)
    expected = best - (p * np.where(ok, cq, 0.0)).sum(axis=1)
    top = np.where(ok, p, -1.0).argmax(axis=1)
    top_regret = best - np.where(keep, cq[np.arange(len(top)), top], 0.0)
    agree = top == np.where(ok, visits, -1.0).argmax(axis=1)
    entropy = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)).sum(axis=1)
    if not keep.any():
        raise ValueError("no row of the evaluation set has a visited legal action")
    return {"rows": int(keep.sum()), "exp_q_regret": float(expected[keep].mean()), "top1_q_regret": float(top_regret[keep].mean()),
            "top1_most_visited": float(agree[keep].mean()), "entropy": float(entropy[keep].mean())}


def load_policy_net(path: Path, device: str):
    import torch  # noqa: PLC0415
    from combat_net import CombatNet  # noqa: PLC0415
    from combat_tokens import VOCAB  # noqa: PLC0415

    ck = torch.load(path, map_location=device, weights_only=True)
    model = CombatNet(VOCAB, ck["n_buckets"], d_model=ck["d_model"], layers=ck["layers"], heads=ck["heads"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval()


def policy_probs(model, data: dict[str, np.ndarray], device: str, batch: int = EVAL_BATCH) -> np.ndarray:
    """The net's policy over the action slots of every row of `data` (legal slots only; the end-turn probe channels blanked, as trained)."""
    import torch  # noqa: PLC0415
    from combat_net import collate  # noqa: PLC0415
    from combat_tokens import F_ENDTURN_DEATH, F_ENDTURN_KNOWN, F_ENDTURN_MEAN, F_ENDTURN_WORST, Tokens  # noqa: PLC0415

    offsets = np.concatenate([[0], np.cumsum(data["lengths"])])
    legal = data["legal"] > 0
    out = []
    with torch.no_grad():
        for start in range(0, len(legal), batch):
            stop = min(len(legal), start + batch)
            states = [Tokens(*(data[k][offsets[i]:offsets[i + 1]] for k in TOKEN_KEYS)) for i in range(start, stop)]
            tensors = collate(states, device)
            for channel in (F_ENDTURN_MEAN, F_ENDTURN_WORST, F_ENDTURN_DEATH, F_ENDTURN_KNOWN):
                tensors["feats"][:, :, channel] = 0.0
            logits = model(**tensors)[1].masked_fill(torch.from_numpy(~legal[start:stop]).to(device), float("-inf"))
            out.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(out)


def eval_measures(policy: Path, evaluation: dict[str, np.ndarray], device: str) -> dict[str, float]:
    model = load_policy_net(policy, device)
    return regret_measures(policy_probs(model, evaluation, device), evaluation["legal"], evaluation["visits"], evaluation["qvals"])


def native_problem() -> str | None:
    """Why measure 3 cannot run here (the emulator library is not found), or None. Checked before the first fit so a fit is not lost to it."""
    import gen0  # noqa: PLC0415

    gen0._setup()  # noqa: SLF001
    lib = os.environ.get("STS2_LIB_PATH", "")
    if not lib or not (Path(lib) / "Sts2Emulator.dll").exists():
        return "STS2_LIB_PATH does not name a folder holding the emulator library; measure 3 needs it (--no-devset skips the measure)"
    return None


def devset_measures(prior_dir: Path, devset: Path, stop: Path | None = None) -> dict[str, Any]:
    """Measure 3: the net alone on the labelled dev-set positions, through `a2c_target._eval_position`'s net-alone path (no search):
    expected and top-1 regret in HP and top-1 agreement with the label, population-weighted; the per-position values ride along
    (weight, expected, top-1) for the confirmation's paired interval."""
    import a2c_target  # noqa: PLC0415
    import a3_devset  # noqa: PLC0415
    import gen0  # noqa: PLC0415

    gen0._setup()  # noqa: SLF001
    a2c_target.DEVSET = Path(devset)
    rows = [r for r in a3_devset.read_jsonl(Path(devset) / "labels.jsonl") if "error" not in r]
    weights = a3_devset._weights(rows, json.loads((Path(devset) / "manifest.json").read_text(encoding="utf-8")))  # noqa: SLF001
    per: dict[str, list[float]] = {}
    for index, row in enumerate(rows):
        if index % 20 == 0:
            check_stop(stop)
        net = a2c_target._eval_position(index, row, {"net": str(prior_dir)}, 0)[0]["net"]["net"]  # noqa: SLF001
        per[row["pid"]] = [float(net["exp_regret"] or 0.0), float(net["top1_regret"] or 0.0), float(net["agree"])]
    n = len(per)
    mean = {i: sum(weights[p] * v[i] for p, v in per.items()) / n for i in range(3)}
    return {"positions": n, "exp_regret": mean[0], "top1_regret": mean[1], "agree": mean[2],
            "per_position": {p: [round(weights[p], 6), v[0], v[1]] for p, v in per.items()}}


# --------------------------------------------------------------------------------------
# run: one fit
# --------------------------------------------------------------------------------------


def fit_validation(doc: dict[str, Any]) -> dict[str, Any]:
    """The trainer's per-epoch validation losses (the 15% of runs `train_combat` holds out for checkpoint selection) boiled down."""
    curves = doc.get("curves") or []
    out: dict[str, Any] = {"report": {k: v for k, v in (doc.get("report") or {}).items() if v is not None}}
    if curves:
        best_policy = min(curves, key=lambda c: c["policy_val"])
        best_value = min(curves, key=lambda c: c["value_val"])
        out.update({"policy_val": best_policy["policy_val"], "policy_val_epoch": best_policy["epoch"], "value_val": best_value["value_val"],
                    "value_val_epoch": best_value["epoch"], "last_policy_val": curves[-1]["policy_val"], "last_value_val": curves[-1]["value_val"]})
    return out


def recipe_of(arm: Arm, seed: int, data: dict[str, np.ndarray]) -> dict[str, Any]:
    return {"target": arm.target, "policy_target": arm.policy_target, "q_temperature": arm.q_temperature, "epochs": arm.epochs,
            "d_model": arm.d_model, "layers": arm.layers, "heads": 4, "batch": 64, "lr": 3e-4, "threads": THREADS, "max_fights": arm.size,
            "build_seed": arm.build_seed, "torch_seed": seed, "split_seed": 0, "train_samples": int(data["lengths"].size),
            "train_set": arm.data_name, "seed_rules": SEED_RULES}


def fit_one(out: Path, arm: Arm, seed: int, data: dict[str, np.ndarray], evaluation: dict[str, np.ndarray], *, device: str, generation: int,
            stop: Path | None, devset: Path | None, confirm: bool = False) -> dict[str, Any]:
    """Fit `arm` at torch seed `seed` into a temporary folder and rename it to `<arm>/s<seed>/` once its measures are written;
    anything else (the stop file, an error, Ctrl-C) removes the folder. Returns the measures."""
    import torch  # noqa: PLC0415

    final = fit_dir(out, arm.name, seed, confirm)
    tmp = final.with_name("tmp-" + final.name)
    shutil.rmtree(tmp, ignore_errors=True)  # the leftover of a process that was killed
    shutil.rmtree(final, ignore_errors=True)  # a folder without measures.json is not a fit
    on_gpu = str(device).startswith("cuda") and torch.cuda.is_available()
    try:
        if on_gpu:
            torch.cuda.reset_peak_memory_stats()
        doc = gen_loop.train_combat_prior(data, tmp, device=device, epochs=arm.epochs, d_model=arm.d_model, layers=arm.layers,
                                          threads=THREADS, seed=seed, policy_target=arm.policy_target, q_temperature=arm.q_temperature,
                                          epoch_end=stop_hook(stop))
        peak = round(torch.cuda.max_memory_allocated() / 2**20, 1) if on_gpu else None
        (tmp / "work" / "train.npz").unlink(missing_ok=True)  # the fit's copy of its data
        (tmp / "combat_net.pt").unlink(missing_ok=True)  # the value-best checkpoint: the prior is the policy head's
        gen_loop.write_prior_model_json(tmp, generation + 1)
        check_stop(stop)
        evaluated = eval_measures(tmp / "combat_policy.pt", evaluation, device)
        check_stop(stop)
        dev = devset_measures(tmp, devset, stop) if devset is not None else None
        measures = {"format": FIT_FORMAT, "arm": arm.name, "seed": seed, "confirm": confirm, "recipe": recipe_of(arm, seed, data),
                    "fit_min": round(float(doc["wall_min"]), 3), "gpu_peak_mb": peak, "eval": evaluated, "devset": dev,
                    "validation": fit_validation(doc), "curves": doc.get("curves")}
        save_json(tmp / "measures.json", measures)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    os.replace(tmp, final)
    return measures


def _summary(m: dict[str, Any]) -> str:
    e, d = m["eval"], m["devset"]
    dev = f", dev exp/top1 {d['exp_regret']:.3f}/{d['top1_regret']:.3f}" if d else ""
    return (f"{m['arm']} s{m['seed']}: eval exp {e['exp_q_regret']:.3f} top1 {e['top1_q_regret']:.3f} agree {e['top1_most_visited']:.3f}"
            f"{dev}, {m['fit_min']:.2f} min, peak {m['gpu_peak_mb']} MB")


def cmd_run(a: argparse.Namespace) -> int:
    out = Path(a.out)
    manifest = gen_loop.read_json(out / "manifest.json")  # no build: no run
    seeds = [int(x) for x in a.seeds.split(",") if x]
    if a.confirm:
        jobs = [(a.confirm, s) for s in seeds]
    elif a.arms == "all":
        jobs = all_order(seeds, build_seeds_of(out))
    elif a.arms:
        jobs = [(n, s) for n in a.arms.split(",") if n for s in seeds]
    else:
        print("prior_sweep run: give --arms or --confirm", file=sys.stderr)
        return 2
    arms = {name: resolve_arm(name) for name, _ in jobs}
    stop = Path(a.stop_file) if a.stop_file else None
    devset = None if a.no_devset else Path(a.devset)
    if devset is not None and (problem := native_problem()):
        print(f"prior_sweep run: {problem}", file=sys.stderr)
        return 2
    started = time.perf_counter()
    training: tuple[str, dict[str, np.ndarray]] | None = None
    evaluation: dict[str, np.ndarray] | None = None
    try:
        for name, seed in jobs:
            if (fit_dir(out, name, seed, bool(a.confirm)) / "measures.json").is_file():
                continue
            if a.budget_min is not None and (time.perf_counter() - started) / 60 >= a.budget_min:
                print(f"budget of {a.budget_min:g} min spent; stopping before {name} s{seed}")
                return 0
            check_stop(stop)  # a fit boundary
            arm = arms[name]
            if training is None or training[0] != arm.data_name:
                path = out / f"{arm.data_name}.npz"
                if not path.is_file():
                    print(f"prior_sweep run: {arm.data_name}.npz is not built (build --build-seeds)", file=sys.stderr)
                    return 2
                training = None  # one training set at a time
                training = (arm.data_name, load_npz(path))
            if evaluation is None:
                evaluation = load_npz(out / "eval.npz")
            print(_summary(fit_one(out, arm, seed, training[1], evaluation, device=a.device, generation=int(manifest["generation"]),
                                   stop=stop, devset=devset, confirm=bool(a.confirm))), flush=True)
    except StopFit:
        print(f"stop file seen: the fit in hand is dropped, nothing of it written; exit {STOP_EXIT}", flush=True)
        return STOP_EXIT
    return 0


# --------------------------------------------------------------------------------------
# search: measure 4
# --------------------------------------------------------------------------------------


def cmd_search(a: argparse.Namespace) -> int:
    out = Path(a.out)
    if Path(a.quiet_lock).exists():
        print("prior_sweep search: QUIET.lock exists; the search needs the CPU and runs only with no lock", file=sys.stderr)
        return 2
    names = auto_search_arms(out) if a.arms == "auto" else [n for n in a.arms.split(",") if n]
    if not names:
        print("prior_sweep search: no arm to search with", file=sys.stderr)
        return 2
    paths = {n: fit_dir(out, n, a.seed) for n in names}
    missing = [n for n, p in paths.items() if not (p / "combat_policy.pt").is_file()]
    if missing:
        print(f"prior_sweep search: no fit for {', '.join(missing)}", file=sys.stderr)
        return 2
    import a2c_target  # noqa: PLC0415
    import a3_devset  # noqa: PLC0415
    import gen0  # noqa: PLC0415

    gen0._setup()  # noqa: SLF001
    folder = out / "search"
    folder.mkdir(parents=True, exist_ok=True)
    rows = [r for r in a3_devset.read_jsonl(a2c_target.DEVSET / "labels.jsonl") if "error" not in r]
    t0 = time.time()
    got = a3_devset._pool([(i, r, {n: str(p) for n, p in paths.items()}, a.reps) for i, r in enumerate(rows)],  # noqa: SLF001
                          a2c_target._eval_position, a.procs, folder / "devset.jsonl")  # noqa: SLF001
    base = LAB_ARM if LAB_ARM in names else names[0]
    report = a2c_target.report(argparse.Namespace(out=str(folder)), base=base)
    report["paired_vs_base"] = report.pop("paired_vs_visits")
    doc = {"format": "prior-sweep-search/1", "arms": names, "base": base, "torch_seed": a.seed, "reps": a.reps, "procs": a.procs,
           "wall_s": round(time.time() - t0), "errors": sum("error" in g for g in got), "report": report}
    save_json(out / "search.json", doc)
    return 0


# --------------------------------------------------------------------------------------
# table: cells, main effects, the data arm, the confirmation
# --------------------------------------------------------------------------------------

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
        20: 2.086, 25: 2.060, 30: 2.042, 60: 2.000, 120: 1.980}


def t_crit(df: float) -> float:
    """Two-sided 95% Student t for `df` degrees of freedom, from a table, at the largest tabulated df not above it (the wider side)."""
    if df < 1:
        return float("inf")
    return T975[max(k for k in T975 if k <= df)] if df < 1000 else 1.96


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")


def pooled_variance(groups: Sequence[Sequence[float]]) -> tuple[float, int]:
    """The within-group (between-seed) variance pooled over groups, and its degrees of freedom."""
    ss = sum(float(((np.asarray(g, dtype=np.float64) - np.mean(g)) ** 2).sum()) for g in groups)
    df = sum(len(g) - 1 for g in groups)
    return (ss / df if df > 0 else float("nan")), df


def effect_model(cells: Sequence[tuple[dict[str, Any], Sequence[float]]], factors: Sequence[str], *, interactions: bool = True,
                 sigma2: tuple[float, int] | None = None) -> dict[str, Any]:
    """Least-squares fit of the cell means on the factors' centred indicator codes (the first level seen of a factor is its reference;
    a two-level factor codes -0.5/+0.5, so its coefficient is the second level minus the first averaged over the other factors)
    and, when the residual degrees of freedom of the means allow, their two-way interactions. Cells are weighted by their seed count.
    The error is the between-seed variance pooled over the cells (or `sigma2` = (variance, df) given): Var(beta) = sigma2 (X'WX)^-1.
    An effect "moves the prior" when it exceeds `EXCEEDS` standard errors."""
    levels = {f: list(dict.fromkeys(c[f] for c, _ in cells)) for f in factors}
    names, columns, owner = ["intercept"], [np.ones(len(cells))], [None]
    for f in factors:
        for level in levels[f][1:]:
            indicator = np.array([1.0 if c[f] == level else 0.0 for c, _ in cells])
            names.append(f"{f}:{level}")
            columns.append(indicator - indicator.mean())
            owner.append(f)

    def design(with_interactions: bool) -> tuple[list[str], np.ndarray]:
        cols, labels = list(columns), list(names)
        if with_interactions:
            for i, j in itertools.combinations(range(1, len(columns)), 2):
                if owner[i] != owner[j]:
                    cols.append(columns[i] * columns[j])
                    labels.append(f"{names[i]} x {names[j]}")
        return labels, np.column_stack(cols)

    labels, x = design(interactions)
    if x.shape[1] >= len(cells) or np.linalg.matrix_rank(x) < x.shape[1]:
        labels, x = design(False)  # not enough residual degrees of freedom among the means: main effects alone
        interactions = False
    if x.shape[1] > len(cells) or np.linalg.matrix_rank(x) < x.shape[1]:
        return {"estimable": False, "cells": len(cells)}
    n = np.array([len(v) for _, v in cells], dtype=np.float64)
    ybar = np.array([float(np.mean(v)) for _, v in cells])
    xtwx = x.T @ (n[:, None] * x)
    beta = np.linalg.solve(xtwx, x.T @ (n * ybar))
    var, df = sigma2 if sigma2 is not None else pooled_variance([v for _, v in cells])
    inverse = np.linalg.inv(xtwx)
    se = np.sqrt(var * np.diag(inverse)) if df > 0 else np.full(len(beta), float("nan"))
    effects = [{"name": label, "effect": float(b), "se": float(s), "exceeds": bool(abs(b) > EXCEEDS * s) if np.isfinite(s) else False}
               for label, b, s in zip(labels, beta, se, strict=True) if label != "intercept"]
    return {"estimable": True, "cells": len(cells), "interactions": interactions, "residual_variance": var, "residual_df": df,
            "sigma": float(np.sqrt(var)) if df > 0 else None, "effects": effects}


def effect_of(model: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((e for e in model.get("effects", []) if e["name"] == name), None)


def split_arm(name: str) -> dict[str, Any] | None:
    found = _CELL.match(name)
    return {"target": found[1], "epochs": int(found[2]), "d_model": int(found[3]), "layers": int(found[4])} if found else None


def load_fits(out: Path) -> tuple[dict[str, dict[int, dict]], dict[str, dict[int, dict]]]:
    """Every fit's measures.json under `out`: ({arm: {seed: measures}}, the confirmation fits likewise)."""
    fits: dict[str, dict[int, dict]] = {}
    confirms: dict[str, dict[int, dict]] = {}
    for path in sorted(Path(out).glob("*/*/measures.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        (confirms if m.get("confirm") else fits).setdefault(m["arm"], {})[int(m["seed"])] = m
    return fits, confirms


def metric(m: dict[str, Any], name: str) -> float | None:
    """A per-fit number by its table name."""
    e, d, v = m["eval"], m.get("devset") or {}, m.get("validation") or {}
    table = {"exp_q_regret": e["exp_q_regret"], "top1_q_regret": e["top1_q_regret"], "top1_most_visited": e["top1_most_visited"],
             "entropy": e["entropy"], "dev_exp": d.get("exp_regret"), "dev_top1": d.get("top1_regret"), "policy_val": v.get("policy_val"),
             "value_val": v.get("value_val"), "fit_min": m.get("fit_min"), "gpu_peak_mb": m.get("gpu_peak_mb")}
    return table[name]


def values_of(fits: dict[int, dict], name: str) -> list[float]:
    return [x for m in fits.values() if (x := metric(m, name)) is not None]


def welch_interval(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """mean(a) - mean(b) with the between-seed (Welch) t-interval; `excludes_zero` when it does not contain 0."""
    ma, sa = mean_sd(a)
    mb, sb = mean_sd(b)
    va, vb = sa**2 / len(a), sb**2 / len(b)
    se = float(np.sqrt(va + vb))
    if not np.isfinite(se):
        return {"diff": ma - mb, "se": None, "df": None, "lo": None, "hi": None, "excludes_zero": False}
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)) if se > 0 else len(a) + len(b) - 2
    half = t_crit(df) * se
    return {"diff": ma - mb, "se": se, "df": float(df), "lo": ma - mb - half, "hi": ma - mb + half,
            "excludes_zero": bool(ma - mb - half > 0 or ma - mb + half < 0)}


def paired_bootstrap(a: dict[str, float], b: dict[str, float], weights: dict[str, float], n: int = 2000, seed: int = 1) -> tuple[float, float, float]:
    """The weighted mean of a - b over positions with a 95% bootstrap interval (a3_devset.paired_boot's statistic)."""
    keys = sorted(a)
    w = np.array([weights[k] for k in keys])
    d = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(keys), size=(n, len(keys)))
    means = np.sort((w[pick] * d[pick]).sum(axis=1) / w[pick].sum(axis=1))
    return float((w * d).sum() / w.sum()), float(means[int(0.025 * n)]), float(means[int(0.975 * n)])


def seed_mean_positions(fits: dict[int, dict], column: int) -> tuple[dict[str, float], dict[str, float]]:
    """Per dev-set position the mean over the fits' seeds of the net-alone regret (column 1 expected, 2 top-1), and the weights."""
    per: dict[str, list[float]] = {}
    weights: dict[str, float] = {}
    for m in fits.values():
        for pid, row in ((m.get("devset") or {}).get("per_position") or {}).items():
            per.setdefault(pid, []).append(row[column])
            weights[pid] = row[0]
    return {p: float(np.mean(v)) for p, v in per.items()}, weights


def cell_values(fits: dict[str, dict[int, dict]], name: str) -> list[tuple[dict[str, Any], list[float]]]:
    """The grid cells that have fits, as (factor levels, per-seed values of `name`) for `effect_model`."""
    cells = []
    for arm in grid_names():
        if arm in fits and (values := values_of(fits[arm], name)):
            cells.append((split_arm(arm), values))
    return cells


def main_effects(fits: dict[str, dict[int, dict]]) -> dict[str, Any]:
    """Measure 2 across all cells; measure 1 within each target level (its own residual: the levels' regrets are not on one scale)."""
    out: dict[str, Any] = {"top1_q_regret": effect_model(cell_values(fits, "top1_q_regret"), ["target", "epochs", "d_model", "layers"]),
                           "exp_q_regret": {}}
    for target in TARGETS:
        cells = [(c, v) for c, v in cell_values(fits, "exp_q_regret") if c["target"] == target]
        out["exp_q_regret"][target] = effect_model(cells, ["epochs", "d_model", "layers"]) if len(cells) > 1 else {"estimable": False, "cells": len(cells)}
    return out


def choose_cell(effects: dict[str, Any]) -> dict[str, Any]:
    """PRIOR_SWEEP.md 5.2: the better level of every factor whose measure-2 effect exceeds twice its standard error, the lab's level
    otherwise; a factor the within-target measure 1 contradicts beyond twice its error stays at the lab's level."""
    lab = split_arm(LAB_ARM)
    m2 = effects["top1_q_regret"]
    chosen, notes = dict(lab), []
    if not m2.get("estimable"):
        return {"cell": LAB_ARM, "notes": ["measure 2's model is not estimable yet"], "moves": []}
    moves = [e for e in m2["effects"] if e["exceeds"] and " x " not in e["name"]]
    targets = [e for e in moves if e["name"].startswith("target:") and e["effect"] < 0]
    if targets:
        chosen["target"] = min(targets, key=lambda e: e["effect"])["name"].split(":")[1]
    for e in moves:
        factor, level = e["name"].split(":")
        if factor == "target" or e["effect"] >= 0:  # a positive effect: the second level is worse, the lab's level (the first) stays
            continue
        chosen[factor] = int(level)
    for factor in ("epochs", "d_model", "layers"):
        if chosen[factor] != lab[factor]:
            within = effect_of(effects["exp_q_regret"].get(chosen["target"], {}), f"{factor}:{chosen[factor]}")
            if within and within["exceeds"] and within["effect"] > 0:
                notes.append(f"{factor} {chosen[factor]} is contradicted by measure 1 within {chosen['target']}; kept at {lab[factor]}")
                chosen[factor] = lab[factor]
    return {"cell": cell_name(chosen["target"], chosen["epochs"], chosen["d_model"], chosen["layers"]), "notes": notes,
            "moves": [e["name"] for e in moves]}


def best_target_cell(fits: dict[str, dict[int, dict]]) -> str | None:
    """The target level with the lowest mean top-1 regret over its cells, as the cell at the lab's other levels (None: no fits)."""
    means = {}
    for target in TARGETS:
        values = [x for arm, seeds in fits.items() if (c := split_arm(arm)) and c["target"] == target for x in values_of(seeds, "top1_q_regret")]
        if values:
            means[target] = float(np.mean(values))
    if not means:
        return None
    lab = split_arm(LAB_ARM)
    return cell_name(min(means, key=means.get), lab["epochs"], lab["d_model"], lab["layers"])


def auto_search_arms(out: Path) -> list[str]:
    """The three nets of measure 4: the lab's cell, the chosen cell, and the best target level's cell when it is another one."""
    fits, _ = load_fits(out)
    arms = [LAB_ARM, choose_cell(main_effects(fits))["cell"], best_target_cell(fits) or LAB_ARM]
    return [a for i, a in enumerate(arms) if a not in arms[:i] and (fit_dir(out, a, 0) / "combat_policy.pt").is_file()]


def seed_spread(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    mean, sd = mean_sd(values)
    return {"n": len(values), "mean": mean, "sd": None if not np.isfinite(sd) else sd, "min": float(min(values)), "max": float(max(values))}


def spreads(fits: dict[str, dict[int, dict]]) -> dict[str, Any]:
    """The torch-seed spread (the lab's cell at build seed 7, torch seeds 0..3) beside the build-seed spread (torch seed 0, build seeds 7, 8, 9)."""
    out: dict[str, Any] = {}
    for name in ("exp_q_regret", "top1_q_regret", "dev_exp", "dev_top1", "policy_val"):
        torch_seeds = values_of(fits.get(LAB_ARM, {}), name)
        build = [x for arm, seed in ((LAB_ARM, 0), *((a, 0) for a in sorted(fits) if _BUILD.match(a))) if seed in fits.get(arm, {})
                 and (x := metric(fits[arm][seed], name)) is not None]
        out[name] = {"torch_seeds": seed_spread(torch_seeds), "build_seeds": seed_spread(build)}
    return out


def data_reading(fits: dict[str, dict[int, dict]]) -> dict[str, Any]:
    """The data arm (12,800 fights, 16 epochs) against the lab's cell at 32 epochs on 6,400 (equal gradient steps) and at 16 epochs;
    the error is the between-seed variance pooled over the visits-target cells (the data arm is a visits-target fit)."""
    out: dict[str, Any] = {}
    for name in ("exp_q_regret", "top1_q_regret"):
        data = values_of(fits.get(DATA_ARM, {}), name)
        var, df = pooled_variance([values_of(s, name) for a, s in fits.items() if a.startswith("visits-") and _CELL.match(a)])
        row: dict[str, Any] = {"data12800": seed_spread(data), "residual_df": df}
        for ref in (cell_name("visits", 32, 128, 2), LAB_ARM):
            other = values_of(fits.get(ref, {}), name)
            if not data or not other:
                continue
            diff = float(np.mean(data) - np.mean(other))
            se = float(np.sqrt(var * (1 / len(data) + 1 / len(other)))) if df > 0 and np.isfinite(var) else None
            row[f"vs_{ref}"] = {"diff": diff, "se": se, "exceeds": bool(se is not None and abs(diff) > EXCEEDS * se)}
        out[name] = row
    return out


def confirmation(confirms: dict[str, dict[int, dict]], chosen: str) -> dict[str, Any]:
    """PRIOR_SWEEP.md 5.3 on the fresh fits only: the chosen cell against the lab's. It passes when its mean top-1 regret (and, when it
    shares the lab's target, its expected regret) is lower with the between-seed t-interval above 0 excluded, and its dev-set net-alone
    regret is not worse beyond the paired bootstrap interval."""
    if chosen == LAB_ARM:
        return {"status": "the chosen cell is the lab's cell; nothing to confirm"}
    if chosen not in confirms or LAB_ARM not in confirms:
        return {"status": "waiting for confirm fits", "have": {a: sorted(s) for a, s in confirms.items()}}
    new, lab = confirms[chosen], confirms[LAB_ARM]
    same_target = split_arm(chosen)["target"] == split_arm(LAB_ARM)["target"]
    rows: dict[str, Any] = {}
    for name in ("top1_q_regret", "exp_q_regret", "dev_top1", "dev_exp"):
        a, b = values_of(new, name), values_of(lab, name)
        if len(a) > 1 and len(b) > 1:
            rows[name] = welch_interval(a, b)
    paired: dict[str, Any] = {}
    for name, column in (("dev_exp", 1), ("dev_top1", 2)):
        (pa, weights), (pb, _) = seed_mean_positions(new, column), seed_mean_positions(lab, column)
        if pa and pb:
            mean, lo, hi = paired_bootstrap(pa, pb, weights)
            paired[name] = {"diff": mean, "lo": lo, "hi": hi, "worse": bool(lo > 0)}
    better = {n: bool(r["excludes_zero"] and r["diff"] < 0) for n, r in rows.items()}
    checks = {"top1_q_regret_better": better.get("top1_q_regret", False),
              "exp_q_regret_better_within_target": better.get("exp_q_regret", False) if same_target else None,
              "dev_not_worse": not any(p["worse"] for p in paired.values()) if paired else None}
    passed = checks["top1_q_regret_better"] and checks["exp_q_regret_better_within_target"] is not False and checks["dev_not_worse"] is not False
    return {"status": "pass" if passed else "fail", "chosen": chosen, "lab": LAB_ARM, "seeds": {chosen: sorted(new), LAB_ARM: sorted(lab)},
            "welch": rows, "dev_paired": paired, "checks": checks, "same_target": same_target}


def fmt(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def fmt_range(values: Sequence[float], digits: int = 3) -> str:
    if not values:
        return "-"
    return f"{np.mean(values):.{digits}f} [{min(values):.{digits}f}, {max(values):.{digits}f}]"


CELL_COLUMNS = (("top1_q_regret", "top-1 regret"), ("exp_q_regret", "expected regret"), ("top1_most_visited", "top-1 = most visited"),
                ("entropy", "entropy"), ("dev_top1", "dev top-1"), ("dev_exp", "dev expected"), ("policy_val", "val policy"),
                ("fit_min", "fit min"), ("gpu_peak_mb", "GPU MB"))


def cell_table(fits: dict[str, dict[int, dict]]) -> list[str]:
    lines = ["| arm | fits | " + " | ".join(label for _, label in CELL_COLUMNS) + " |", "|---|---|" + "---|" * len(CELL_COLUMNS)]
    order = [a for a in (*grid_names(), *sorted(a for a in fits if _BUILD.match(a)), DATA_ARM) if a in fits]
    for arm in order:
        cells = [fmt_range(values_of(fits[arm], name), 1 if name == "gpu_peak_mb" else 3) for name, _ in CELL_COLUMNS]
        lines.append(f"| {arm} | {len(fits[arm])} | " + " | ".join(cells) + " |")
    return lines


def effects_table(model: dict[str, Any]) -> list[str]:
    if not model.get("estimable"):
        return [f"not estimable yet ({model.get('cells', 0)} cells)"]
    lines = [f"cells {model['cells']}, interactions {'in' if model['interactions'] else 'out'}, between-seed variance {fmt(model['residual_variance'], 5)} "
             f"on {model['residual_df']} df", "", "| effect | estimate | SE | beyond 2 SE |", "|---|---|---|---|"]
    lines += [f"| {e['name']} | {fmt(e['effect'], 4)} | {fmt(e['se'], 4)} | {'yes' if e['exceeds'] else 'no'} |" for e in model["effects"]]
    return lines


def build_report(out: Path) -> tuple[str, dict[str, Any]]:
    fits, confirms = load_fits(out)
    effects = main_effects(fits)
    chosen = choose_cell(effects)
    summary = {"effects": effects, "chosen": chosen, "spreads": spreads(fits), "data_arm": data_reading(fits),
               "confirmation": confirmation(confirms, chosen["cell"]), "search_arms": auto_search_arms(out), "fits": sum(len(s) for s in fits.values())}
    lines = ["# Combat prior recipe sweep", "", f"Fits: {summary['fits']} (confirmation fits: {sum(len(s) for s in confirms.values())}). Regrets in HP, "
             "on the fixed evaluation set (runs whose seed hash is in [0.0, 0.1)); the dev-set columns are the net alone on the labelled positions.", "",
             "## Cells (mean [min, max] over torch seeds)", "", *cell_table(fits), "", "## Main effects", "",
             "Codes are centred: a two-level factor is the second level minus the first, averaged over the other factors; target is each completed-Q "
             "level minus visits. The error is the between-seed variance pooled over the cells.", "",
             "### Measure 2 (top-1 regret, across targets)", "", *effects_table(effects["top1_q_regret"])]
    for target, model in effects["exp_q_regret"].items():
        lines += ["", f"### Measure 1 (expected regret) within target {target}", "", *effects_table(model)]
    lines += ["", f"## Chosen cell: {chosen['cell']}", "", f"Factors beyond 2 SE on measure 2: {', '.join(chosen['moves']) or 'none'}.", *[f"- {n}" for n in chosen["notes"]],
              "", "## Seed spreads (torch seeds beside build seeds)", "", "| measure | torch seeds n / mean / sd / range | build seeds n / mean / sd / range |", "|---|---|---|"]
    for name, both in summary["spreads"].items():
        def one(s: dict[str, Any]) -> str:
            return "-" if not s.get("n") else f"{s['n']} / {fmt(s['mean'])} / {fmt(s['sd'])} / {fmt(s['min'])}-{fmt(s['max'])}"

        lines.append(f"| {name} | {one(both['torch_seeds'])} | {one(both['build_seeds'])} |")
    lines += ["", "## Data arm (12,800 fights, 16 epochs)", ""]
    for name, row in summary["data_arm"].items():
        lines.append(f"- {name}: data12800 mean {fmt(row['data12800'].get('mean'))} over {row['data12800'].get('n', 0)} fits; " + "; ".join(
            f"{k} diff {fmt(v['diff'])} (SE {fmt(v['se'])}, beyond 2 SE: {'yes' if v['exceeds'] else 'no'})" for k, v in row.items() if k.startswith("vs_")))
    lines += ["", "## Confirmation (fresh fits)", "", f"status: {summary['confirmation']['status']}"]
    for name, r in summary["confirmation"].get("welch", {}).items():
        lines.append(f"- {name}: chosen minus lab {fmt(r['diff'], 4)}, interval [{fmt(r['lo'], 4)}, {fmt(r['hi'], 4)}], excludes 0: {'yes' if r['excludes_zero'] else 'no'}")
    for name, r in summary["confirmation"].get("dev_paired", {}).items():
        lines.append(f"- {name} paired: {fmt(r['diff'], 4)} [{fmt(r['lo'], 4)}, {fmt(r['hi'], 4)}], worse: {'yes' if r['worse'] else 'no'}")
    lines += ["", "Nothing is adopted by this sweep (PRIOR_SWEEP.md 5.4)."]
    return "\n".join(lines) + "\n", summary


def cmd_table(a: argparse.Namespace) -> int:
    out = Path(a.out)
    text, summary = build_report(out)
    (out / "report.md").write_text(text, encoding="utf-8")
    save_json(out / "table.json", summary)
    print(text)
    return 0


# --------------------------------------------------------------------------------------
# queue
# --------------------------------------------------------------------------------------


def queue_jobs(out: Path, seeds: Sequence[int] = (0, 1)) -> list[dict[str, Any]]:
    """The sweep's jobs for the runner in the order of `all_order`, minus the fits that exist; the search job (8 processes) last."""
    jobs = []
    for arm, seed in all_order(seeds, build_seeds_of(out)):
        if (fit_dir(out, arm, seed) / "measures.json").is_file():
            continue
        jobs.append({"name": f"prior-{arm}-s{seed}", "tool": "prior_sweep", "args": ["run", "--out", str(out), "--arms", arm, "--seeds", str(seed)],
                     "cpu_procs": 2, "state": "queued", "attempts": 0})
    jobs.append({"name": "prior-search", "tool": "prior_sweep", "args": ["search", "--out", str(out), "--arms", "auto", "--procs", "8", "--reps", "24"],
                 "cpu_procs": 8, "state": "queued", "attempts": 0})
    return jobs


def cmd_queue(a: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(a.repo)))
    from tools.gpu_lane import quiet_runner  # noqa: PLC0415

    path = Path(a.queue)
    queue = quiet_runner.load_queue(path)
    have = {j.get("name") for j in queue}
    added = [j for j in queue_jobs(Path(a.out), [int(x) for x in a.seeds.split(",") if x]) if j["name"] not in have]
    quiet_runner.save_queue(path, [*queue, *added])
    print(f"queued {len(added)} jobs ({sum(j['cpu_procs'] > 2 for j in added)} needing a quiet window); the queue holds {len(queue) + len(added)}")
    return 0


# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--lab", required=True)
    p.add_argument("--agent-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--build-seeds", default=",".join(map(str, EXTRA_BUILD_SEEDS)), help="further build seeds for the lab-bs<N> arms")
    p.set_defaults(func=cmd_build)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--arms", default="", help="a comma list of arms, or all")
    p.add_argument("--confirm", default="", help="fit this arm at fresh seeds into <arm>/confirm-s<seed>/")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stop-file", default=None, help="drop the fit in hand and exit 3 when this path exists (checked at every epoch end)")
    p.add_argument("--budget-min", type=float, default=None)
    p.add_argument("--devset", default=None, help="the dev-set folder (default: the lab's agent/devset)")
    p.add_argument("--no-devset", action="store_true", help="skip measure 3")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("search")
    p.add_argument("--out", required=True)
    p.add_argument("--arms", default="auto", help="a comma list (the lab's cell first), or auto: lab, chosen, best target")
    p.add_argument("--seed", type=int, default=0, help="the torch seed whose fit is searched with")
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--reps", type=int, default=24)
    p.add_argument("--quiet-lock", default=None, help="refuse to run when this file exists (default: the referee's QUIET.lock)")
    p.add_argument("--device", default="cpu", help=argparse.SUPPRESS)  # the runner appends --device and --stop-file to every job
    p.add_argument("--stop-file", default=None, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_search)
    p = sub.add_parser("table")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_table)
    p = sub.add_parser("queue")
    p.add_argument("--out", required=True)
    p.add_argument("--queue", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--seeds", default="0,1")
    p.set_defaults(func=cmd_queue)
    args = parser.parse_args(argv)
    if args.command in ("run", "search", "build"):
        referee = ROOT.parents[1] / "referee"
        if args.command == "run" and args.devset is None:
            args.devset = str(referee / "agent" / "devset")
        if args.command == "search" and args.quiet_lock is None:
            args.quiet_lock = str(referee / "QUIET.lock")
        _lower_priority()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
