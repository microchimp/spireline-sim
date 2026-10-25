"""The golden fit of `run_train.train` on the CPU: a small synthetic buffer, fixed seeds, no data from the lab.

`golden_fit(config)` trains FightNet, fills H, and trains V on synthetic rows; `fixtures/vrun_golden.npz` holds the result of the
code before the GPU path was added (`python tests/python/vrun_golden.py <out.npz>` wrote it), and `test_vrun_sweep` refits at the
default `TrainConfig` and compares every parameter and every per-epoch loss for equality. Only the API the old code had is used,
so the same module runs against either side of the change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "rl", ROOT / "src", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))

from test_run_train_features import make_view  # noqa: E402

from rl.run_train import features, models, train  # noqa: E402

GOLDEN_CONFIG = {"epochs": 6, "batch": 32, "patience": 3}


def synthetic_value_data(runs: int = 30, per_run: int = 7, seed: int = 20261025) -> train.ValueData:
    """Return `runs` x `per_run` random sparse rows, run ids, floors and targets; one run in five is held out."""
    rng = np.random.default_rng(seed)
    n = runs * per_run
    vectors = []
    for _ in range(n):
        row = np.zeros(features.BASE_SIZE, np.float32)
        at = rng.choice(features.BASE_SIZE, 40, replace=False)
        row[at] = rng.random(40).astype(np.float32)
        vectors.append(row)
    run_index = np.repeat(np.arange(runs), per_run)
    final = rng.integers(3, 54, runs)
    targets = np.zeros((n, models.FLOOR_CLASSES), np.float32)
    targets[np.arange(n), final[run_index] - 1] = 1.0
    return train.ValueData(
        bases=train.SparseRows.build(vectors),
        query_views=[make_view()],
        query_keys=np.zeros(n, np.int64),
        targets=targets,
        end_hp=rng.random(n).astype(np.float32),
        validation=(run_index % 5 == 0),
        runs=runs,
        final_floor=final[run_index].astype(np.int64),
        floor=np.tile(np.arange(1, per_run + 1), runs).astype(np.int64),
        run_index=run_index.astype(np.int64),
        v0=rng.random((n, 3)).astype(np.float32),
    )


def synthetic_fight_data(n: int = 96, seed: int = 11) -> train.FightData:
    rng = np.random.default_rng(seed)
    vectors = []
    for _ in range(n):
        row = np.zeros(features.BASE_SIZE, np.float32)
        at = rng.choice(features.BASE_SIZE, 30, replace=False)
        row[at] = rng.random(30).astype(np.float32)
        vectors.append(row)
    return train.FightData(
        bases=train.SparseRows.build(vectors),
        encounters=rng.choice([8, 2, 11], n).astype(np.int64),
        weak=(rng.random(n) < 0.5).astype(np.float32),
        labels=rng.integers(0, 6, n).astype(np.int64),
        validation=(np.arange(n) % 5 == 0),
    )


def golden_fit(config: train.TrainConfig | None = None) -> dict[str, np.ndarray]:
    """Fit FightNet, fill H, fit V; return every parameter ("fight/..", "value/..") and the per-epoch losses."""
    config = config or train.TrainConfig(**GOLDEN_CONFIG)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fights, values = synthetic_fight_data(), synthetic_value_data()
        torch.manual_seed(7)
        generator = torch.Generator().manual_seed(7)
        fight_net, value_net = models.FightNet(), models.RunValueNet()
        fight_stats = train.train_fight_net(fight_net, fights, config, generator)
        values.fill_fight_block(models.NumpyFightNet(models.numpy_params(fight_net)))
        value_stats = train.train_value_net(value_net, values, config, generator)
    finally:
        torch.set_num_threads(threads)
    out: dict[str, np.ndarray] = {}
    for prefix, net in (("fight/", fight_net), ("value/", value_net)):
        out.update({prefix + k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()})
    for prefix, stats in (("fight_history/", fight_stats), ("value_history/", value_stats)):
        for key in ("train", "validation"):
            out[prefix + key] = np.array([h.get(key, np.nan) for h in stats["history"]], np.float64)
    return out


if __name__ == "__main__":
    np.savez(sys.argv[1], **golden_fit())
