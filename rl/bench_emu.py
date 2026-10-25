"""Emulator throughput on this machine: per core and whole machine, as JSON on stdout.

    python rl/bench_emu.py --n 1 8 16 --seconds 60

For each worker count N a `spawn` pool of N processes starts (each lowered to BelowNormal /
nice 10, one emulator per process), waits at a barrier, and then runs each metric for a fixed
wall time `--seconds`, all workers at once:

  steps   env steps/s, a random legal policy over whole runs (as agent_probe does)
  clones  env.clone(resample_hidden=True) + close per second, from a mid-run state
  sims    MCTS simulations/s at the mcts_probe defaults (sims 150, depth 8, rollout 12), each
          search of 150 starting from a fresh run advanced 10 steps

Totals are the sum over workers of operations / wall time. No host or user names are printed.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

from sts2_gym import procutil  # noqa: E402
from sts2_gym.engine_select import select_engine  # noqa: E402

METRICS = ("steps", "clones", "sims")
SETTINGS = {"sims": 150, "depth": 8, "rollout": 12, "ascension": 8, "advance_steps": 10}


def _env(seed: str):
    from sts2_gym import Sts2RunEnv

    return Sts2RunEnv(seed=seed, max_floors=64, max_episode_steps=4000, ascension=8)


def _steps(deadline: float, who: int) -> tuple[int, float]:
    import numpy as np

    rng = np.random.default_rng(who)
    done_steps, start, run = 0, time.perf_counter(), 0
    while time.perf_counter() < deadline:
        env = _env(f"B{who:03d}{run:06d}")
        env.reset()
        for _ in range(4000):
            legal = np.flatnonzero(env.action_masks())
            if legal.size == 0 or time.perf_counter() >= deadline:
                break
            _o, _r, term, trunc, _i = env.step(int(rng.choice(legal)))
            done_steps += 1
            if term or trunc:
                break
        env.close()
        run += 1
    return done_steps, time.perf_counter() - start


def _clones(deadline: float, who: int) -> tuple[int, float]:
    import numpy as np

    env = _env(f"C{who:03d}000000")
    env.reset()
    rng = np.random.default_rng(who)
    for _ in range(10):
        legal = np.flatnonzero(env.action_masks())
        env.step(int(rng.choice(legal)))
    n, start = 0, time.perf_counter()
    while time.perf_counter() < deadline:
        env.clone(resample_hidden=True).close()
        n += 1
    elapsed = time.perf_counter() - start
    env.close()
    return n, elapsed


def _sims(deadline: float, who: int) -> tuple[int, float]:
    import mcts_probe as mp_
    import numpy as np

    rng = np.random.default_rng(who)
    n, start, search = 0, time.perf_counter(), 0
    while time.perf_counter() < deadline:
        env = _env(f"S{who:03d}{search:06d}")
        env.reset()
        root_handle = mp_.Handle(env._run_handle)
        for _ in range(SETTINGS["advance_steps"]):
            legal = root_handle.legal()
            if legal.size == 0:
                break
            root_handle.step(int(legal[0]))
        root = mp_.Node(root_handle.legal())
        for sim in range(SETTINGS["sims"]):
            if time.perf_counter() >= deadline:
                break
            mp_.simulate(root_handle, root, sim, SETTINGS["depth"], SETTINGS["rollout"], rng)
            n += 1
        env.close()
        search += 1
    return n, time.perf_counter() - start


_RUN = {"steps": _steps, "clones": _clones, "sims": _sims}


def _worker(who: int, seconds: float, metrics: tuple[str, ...], barrier, queue) -> None:
    procutil.lower_priority(10)
    out = {}
    try:
        _clones(time.perf_counter() + 0.5, who)  # load the native library, warm up
        for metric in metrics:
            barrier.wait()
            count, elapsed = _RUN[metric](time.perf_counter() + seconds, who)
            out[metric] = (count, elapsed)
    except Exception as error:  # report, do not hang the barrier of the others
        barrier.abort()
        out["error"] = repr(error)
    queue.put((who, out))


def bench(n: int, seconds: float, metrics: tuple[str, ...]) -> dict:
    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(n), ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(i, seconds, metrics, barrier, queue)) for i in range(n)]
    for p in procs:
        p.start()
    results = [queue.get() for _ in procs]
    for p in procs:
        p.join()
    errors = [r[1]["error"] for r in results if "error" in r[1]]
    row: dict = {"workers": n, "seconds": seconds}
    if errors:
        row["errors"] = errors
    for metric in metrics:
        per = [r[1][metric][0] / r[1][metric][1] for r in results if metric in r[1]]
        row[{"steps": "steps_per_s", "clones": "clones_per_s", "sims": "sims_per_s"}[metric]] = {
            "total": round(sum(per), 1),
            "per_worker_mean": round(sum(per) / len(per), 1) if per else None,
        }
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--n", type=int, nargs="+", default=[1])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    ap.add_argument("--engine", default="old", help="the emulator engine of the workers: old (default), core2 or core2:F08,F10 "
                    "(docs/rebuild/NATIVE_KERNEL.md section 10); the environment's STS2_ENGINE is ignored")
    a = ap.parse_args(argv)
    # D31: the workers load the library, and each reads STS2_ENGINE then; set it here, before the pool spawns, and only from
    # --engine, so an inherited core2 never reaches a benchmark that did not ask for it.
    select_engine(a.engine)
    rows = [bench(n, a.seconds, tuple(a.metrics)) for n in a.n]
    print(json.dumps({"settings": SETTINGS, "engine": a.engine, "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                      "runs": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
