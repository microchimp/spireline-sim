#!/usr/bin/env python3
"""The measurements of docs/rl/GPU_INFERENCE.md sections 9.1 and 9.3, run under the emulator venv's python.

    throughput  9.1 rows: one FIXED seed list for every row, `arm:procs` rows, ABAB interleaved repeats, the server's CPU in
                every CPU-time denominator, server calls/s, mean batch, client p50/p99, the machine's load around each row.
    fullrun     9.1 full-run A/B: `gen0.run_block` on the CPU path at --procs against `run_block_served` at --server-procs,
                the same seed stream, runs/h, decisions/h, mean floor, peak commit charge of the machine.
    ram         9.1 RAM case: a server block whose server is killed (its own process, never by name) after --after-s;
                peak commit before and after the kill.
    quality     9.3: the CPU path against the server on the same seeds, identity (run, record and search level), the paired
                floor difference with two-sided cluster-bootstrap intervals, the pre-registered equivalence verdict (90% interval
                within +-0.30 floor, boss kills by exact two-sided McNemar not significant, sims per decision and
                decisions per run within 2%), sims per decision and decisions per run.

    python rl/prior_bench.py throughput --seeds-file F --seeds 48 --rows "noprior:16,cpu:16,server:16,server:20,server:24" \
        --decisions 100 --repeat 2 --gpu-python <venv-gpu python> --json OUT.json

Every command prints a markdown table (numbers only) and writes its JSON with --json. Results hold no host names, user names
or absolute paths: paths are flags and are never recorded. The `load` figures are the machine's processor load percentage
from the OS before and after a row (a label, not a measurement of the row). The pure parts import no native code.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (HERE, ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

ARMS = ("noprior", "cpu", "server")
ARM_LABELS = {"noprior": "no prior", "cpu": "prior, CPU path (today)", "server": "prior, server"}
BOOT_DRAWS = 10_000
BOOT_SEED = 20261025
TOKEN_KEYS = ("types", "ids", "owners", "feats")
IDENTITY_KEYS = ("decisions", "combat_decisions", "sims", "floor", "end")
ACT1_LAST_FLOOR = 17  # gen_loop.boss_kill: a run past the Act 1 boss floor killed it


# ------------------------------------------------------------------ pure parts: rows, order, arithmetic


def parse_rows(text: str) -> list[tuple[str, int]]:
    """`"noprior:16,cpu:16,server:16"` -> [("noprior", 16), ...]. The arm is one of ARMS, the count a positive integer."""
    rows = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        arm, _, procs = part.partition(":")
        if arm not in ARMS or not procs.isdigit() or int(procs) < 1:
            raise ValueError(f"bad row {part!r}: want arm:procs with arm in {', '.join(ARMS)}")
        rows.append((arm, int(procs)))
    if not rows:
        raise ValueError("no rows")
    return rows


def abab_order(rows: list, repeat: int) -> list[tuple[int, str, int]]:
    """Every row once per repeat, the rows in the given order each time (a, b, c, a, b, c): (repeat, arm, procs)."""
    return [(rep, arm, procs) for rep in range(repeat) for arm, procs in rows]


def row_metrics(tasks: list[dict], wall_s: float, server: dict | None = None) -> dict:
    """One row's numbers from its tasks (`dec`, `sims`, `cpu` seconds of the worker's process) and the block's wall time.

    `server` (`cpu_s`, `calls`, `batches` over the block) adds the server's CPU to the denominator of the per-CPU-second
    figures: dec per CPU-s = decisions / (worker CPU + server CPU).
    """
    dec = sum(t["dec"] for t in tasks)
    sims = sum(t["sims"] for t in tasks)
    worker_cpu = sum(t["cpu"] for t in tasks)
    server_cpu = float(server["cpu_s"]) if server else 0.0
    denom = worker_cpu + server_cpu
    out = {"tasks": len(tasks), "decisions": dec, "sims": sims, "block_wall_s": wall_s,
           "dec_s": dec / wall_s if wall_s else None, "sims_s": sims / wall_s if wall_s else None,
           "worker_cpu_s": worker_cpu, "server_cpu_s": server_cpu if server else None,
           "dec_per_cpu_s": dec / denom if denom else None, "sims_per_cpu_s": sims / denom if denom else None,
           "server_cpu_share": server_cpu / denom if (server and denom) else None,
           "server_calls_s": server["calls"] / wall_s if (server and wall_s) else None,
           "mean_batch": server["calls"] / server["batches"] if (server and server.get("batches")) else None}
    return out


def mean_metrics(rows: list[dict]) -> dict:
    """The mean of each numeric figure over the repeats of one row (None where no repeat has it)."""
    out: dict = {}
    for key in rows[0]:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        out[key] = sum(vals) / len(vals) if vals else None
    return out


def mcnemar_two_sided(b: int, c: int) -> float:
    """Exact two-sided McNemar: b and c are the discordant pairs. Twice the smaller binomial tail, capped at 1; no discordant
    pair gives 1."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def bootstrap_ci(diffs, draws: int = BOOT_DRAWS, seed: int = BOOT_SEED, level: float = 0.95) -> dict:
    """Two-sided percentile interval of the mean of paired differences, resampling whole seeds (clusters = pairs)."""
    x = np.asarray(diffs, dtype=np.float64)
    n = x.size
    if n == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "se": None, "draws": draws}
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    step = max(1, 4_000_000 // n)
    for s in range(0, draws, step):
        e = min(draws, s + step)
        means[s:e] = x[rng.integers(0, n, size=(e - s, n))].mean(axis=1)
    tail = (1.0 - level) / 2.0
    se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {"n": int(n), "mean": float(x.mean()), "lo": float(np.quantile(means, tail)), "hi": float(np.quantile(means, 1 - tail)),
            "se": se, "draws": draws}


EQUIV_BOUND = 0.30  # floors: 0.29 is the smallest gain the judge has ever promoted, so a transport effect inside +-0.3 is invisible to it
EQUIV_LEVEL = 0.90  # a two-sided 90% interval inside the bound is two one-sided tests at 5%


SERVED_MIN = 0.99  # the server arm must be at least this share `server` runs ...
CPU_ARM_MIN = 1.0  # ... and the CPU arm all `cpu` runs: otherwise a dead server makes two CPU arms "equivalent"


def equivalence_verdict(ci90: dict, boss_p: float, sims_ratio: float | None, decisions_ratio: float | None,
                        served_share: float | None, cpu_share: float | None, dropped_pairs: int | None,
                        bound: float = EQUIV_BOUND, alpha: float = 0.05, ratio_tol: float = 0.02) -> dict:
    """Section 9.3's pre-registered rule. Equivalent when (a) the two-sided 90% cluster-bootstrap interval of the paired mean
    floor difference lies within +-`bound`, (b) the two-sided exact McNemar test on Act-1 boss kills is not significant at
    `alpha`, (c) sims per decision and decisions per run (server over CPU path) are each within `ratio_tol` of 1, (d) the server
    arm is at least 99% `server` runs and the CPU arm 100% `cpu` runs, and (e) no pair was lost to a crash or a missing run. A
    figure that was not measured (None) fails its condition. The point estimate, its SE, the 95% interval and the identity
    figures are information and carry no threshold."""
    floor_ok = bool(ci90.get("n")) and ci90["lo"] >= -bound and ci90["hi"] <= bound
    boss_ok = boss_p >= alpha
    ratios_ok = all(r is not None and abs(r - 1.0) <= ratio_tol + 1e-9 for r in (sims_ratio, decisions_ratio))
    paths_ok = (served_share is not None and cpu_share is not None and served_share + 1e-9 >= SERVED_MIN
                and cpu_share + 1e-9 >= CPU_ARM_MIN)
    pairs_ok = dropped_pairs == 0
    return {"equivalent": bool(floor_ok and boss_ok and ratios_ok and paths_ok and pairs_ok), "floor_ok": bool(floor_ok),
            "boss_ok": bool(boss_ok), "ratios_ok": bool(ratios_ok), "paths_ok": bool(paths_ok), "pairs_ok": bool(pairs_ok),
            "served_share": served_share, "cpu_share": cpu_share, "dropped_pairs": dropped_pairs, "bound": bound,
            "alpha": alpha, "ratio_tol": ratio_tol}


def boss_killed(row: dict) -> bool:
    """gen_loop.boss_kill: the run won, or ended past the Act 1 boss floor."""
    return bool(row.get("win")) or int(row.get("floor") or 0) > ACT1_LAST_FLOOR


def identical_rows(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in IDENTITY_KEYS)


def sample_digests(part: dict) -> dict[str, list[tuple[str, int]]]:
    """seed -> [(digest, fight index)] for the combat samples of one part (`gen_loop.read_part`'s dict), in the order the
    run took them. The digest covers the sample's tokens, the root visit counts and the simulation count, so two runs have
    the same digest at the same index exactly when the position and the search that played it were the same."""
    lengths = np.asarray(part["lengths"], dtype=np.int64)
    out: dict[str, list[tuple[str, int]]] = {}
    if lengths.size == 0:
        return out
    ends = np.cumsum(lengths)
    seeds, owner, fight_ix = part["seeds"], part["fights"], part["fight_ix"]
    for i in range(lengths.size):
        lo, hi = int(ends[i] - lengths[i]), int(ends[i])
        h = hashlib.blake2b(digest_size=12)
        for key in TOKEN_KEYS:
            h.update(np.ascontiguousarray(part[key][lo:hi]).tobytes())
        h.update(np.ascontiguousarray(part["visits"][i]).tobytes())
        h.update(int(part["sims"][i]).to_bytes(8, "little", signed=True))
        out.setdefault(str(seeds[int(owner[i])]), []).append((h.hexdigest(), int(fight_ix[i])))
    return out


TIMING_KEYS = ("seconds",)  # wall-clock fields of a record's fights: never part of an identity


def _strip(item: dict) -> dict:
    return {k: v for k, v in item.items() if k not in TIMING_KEYS}


def _floor_of_fight(record: dict | None, fight: int | None):
    fights = (record or {}).get("fights") or []
    return fights[fight].get("floor") if fight is not None and 0 <= fight < len(fights) else None


def first_divergence(rec_a: dict | None, samples_a: list, rec_b: dict | None, samples_b: list) -> dict | None:
    """None when the two runs are the same decision for decision; else where they first differ: the first combat search
    whose position or visit counts differ (its sample index, fight and floor), else the first RunRecord decision, else the
    first fight. A search that differs in visit counts only may still pick the same move: `level` says which kind it was."""
    for j in range(min(len(samples_a), len(samples_b))):
        if samples_a[j][0] != samples_b[j][0]:
            fight = samples_a[j][1]
            return {"level": "search", "index": j, "fight": fight, "floor": _floor_of_fight(rec_a, fight)}
    if len(samples_a) != len(samples_b):
        j = min(len(samples_a), len(samples_b))
        longer = samples_a if len(samples_a) > j else samples_b
        return {"level": "search-count", "index": j, "fight": longer[j][1], "floor": _floor_of_fight(rec_a or rec_b, longer[j][1])}
    for field, level in (("decisions", "decision"), ("fights", "fight")):
        la, lb = (rec_a or {}).get(field) or [], (rec_b or {}).get(field) or []
        for j in range(min(len(la), len(lb))):
            if _strip(la[j]) != _strip(lb[j]):
                return {"level": level, "index": j, "fight": j if field == "fights" else None, "floor": la[j].get("floor")}
        if len(la) != len(lb):
            j = min(len(la), len(lb))
            return {"level": level + "-count", "index": j, "fight": None, "floor": (la if len(la) > j else lb)[j].get("floor")}
    return None


def quality_stats(rows_cpu: list[dict], rows_srv: list[dict], divergence: dict[str, dict | None] | None = None,
                  draws: int = BOOT_DRAWS) -> dict:
    """Section 9.3's numbers from the two arms' rows (paired on seed; a crashed or missing run on either side leaves the pair out of the statistics and fails the verdict).

    `divergence` maps a seed to `first_divergence`'s answer (None: identical), when records were kept.
    """
    cpu = {r["seed"]: r for r in rows_cpu if not r.get("crash")}
    srv = {r["seed"]: r for r in rows_srv if not r.get("crash")}
    seeds = sorted(set(cpu) & set(srv))
    n = len(seeds)
    same_row = [s for s in seeds if identical_rows(cpu[s], srv[s])]
    out: dict = {"paired_seeds": n, "crashed_pairs_dropped": len({r["seed"] for r in rows_cpu} & {r["seed"] for r in rows_srv}) - n,
                 "unpaired_seeds": len({r["seed"] for r in rows_cpu} ^ {r["seed"] for r in rows_srv}),
                 "identical_rows": len(same_row), "identical_rows_fraction": len(same_row) / n if n else None,
                 "identity_keys": list(IDENTITY_KEYS)}
    if divergence is not None:
        diverged = {s: divergence[s] for s in seeds if divergence.get(s) is not None}
        hist: dict[str, int] = {}
        levels: dict[str, int] = {}
        for d in diverged.values():
            hist[str(d["floor"])] = hist.get(str(d["floor"]), 0) + 1
            levels[d["level"]] = levels.get(d["level"], 0) + 1
        out.update({"identical_decision_for_decision": n - len(diverged),
                    "identical_decision_for_decision_fraction": (n - len(diverged)) / n if n else None,
                    "diverged": len(diverged), "first_diverging_floor_histogram": dict(sorted(hist.items(), key=lambda kv: (kv[0] == "None", int(kv[0]) if kv[0] != "None" else 0))),
                    "first_divergence_level": levels})
    d = [float(srv[s]["floor"]) - float(cpu[s]["floor"]) for s in seeds]
    ci = bootstrap_ci(d, draws)  # the 95% interval, information
    ci90 = bootstrap_ci(d, draws, level=EQUIV_LEVEL)  # the interval the rule reads
    out["floor"] = {"cpu_mean": float(np.mean([cpu[s]["floor"] for s in seeds])) if n else None,
                    "server_mean": float(np.mean([srv[s]["floor"] for s in seeds])) if n else None, "diff_ci": ci,
                    "diff_ci90": ci90}
    b = sum(1 for s in seeds if boss_killed(srv[s]) and not boss_killed(cpu[s]))
    c = sum(1 for s in seeds if boss_killed(cpu[s]) and not boss_killed(srv[s]))
    out["boss_kill"] = {"cpu": sum(boss_killed(cpu[s]) for s in seeds), "server": sum(boss_killed(srv[s]) for s in seeds),
                        "server_only": b, "cpu_only": c, "p_two_sided": mcnemar_two_sided(b, c)}
    sims_c, sims_s = sum(cpu[s]["sims"] for s in seeds), sum(srv[s]["sims"] for s in seeds)
    comb_c, comb_s = sum(cpu[s]["combat_decisions"] for s in seeds), sum(srv[s]["combat_decisions"] for s in seeds)
    dec_c, dec_s = sum(cpu[s]["decisions"] for s in seeds), sum(srv[s]["decisions"] for s in seeds)
    spd_c, spd_s = (sims_c / comb_c if comb_c else None), (sims_s / comb_s if comb_s else None)
    out["sims_per_decision"] = {"cpu": spd_c, "server": spd_s, "ratio": spd_s / spd_c if spd_c and spd_s else None}
    out["decisions_per_run"] = {"cpu": dec_c / n if n else None, "server": dec_s / n if n else None,
                                "ratio": dec_s / dec_c if dec_c else None}
    ratios = [out["sims_per_decision"]["ratio"], out["decisions_per_run"]["ratio"]]
    served = sum(1 for r in rows_srv if r.get("prior_path") == "server") / len(rows_srv) if rows_srv else None
    cpu_share = sum(1 for r in rows_cpu if r.get("prior_path") == "cpu") / len(rows_cpu) if rows_cpu else None
    out["arm_paths"] = {"server_arm_server_share": served, "cpu_arm_cpu_share": cpu_share}
    out["verdict"] = equivalence_verdict(ci90, out["boss_kill"]["p_two_sided"], *ratios, served, cpu_share,
                                         out["crashed_pairs_dropped"] + out["unpaired_seeds"])
    return out


# ------------------------------------------------------------------ the machine


def machine_load() -> float | None:
    """The processors' load percentage (all CPUs, averaged) from the OS; None when it cannot be read. A label, not a metric."""
    cmd = "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average"
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd], capture_output=True, text=True,
                             timeout=60, check=False).stdout.strip()
        return float(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def commit_charge_gb() -> float | None:
    """The machine's committed memory in GiB (the OS performance counter's commit total); None off Windows."""
    if os.name != "nt":
        return None
    from ctypes import wintypes  # noqa: PLC0415

    class Perf(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                    ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
                    ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                    ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t), ("HandleCount", wintypes.DWORD),
                    ("ProcessCount", wintypes.DWORD), ("ThreadCount", wintypes.DWORD)]

    info = Perf()
    info.cb = ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
        return None
    return info.CommitTotal * info.PageSize / 2**30


class CommitSampler(threading.Thread):
    """Samples the machine's commit charge every `every_s` seconds on a thread: `samples` is [(seconds since start, GiB)]."""

    def __init__(self, every_s: float = 10.0) -> None:
        super().__init__(name="commit-sampler", daemon=True)
        self.every_s, self.samples, self._stop_flag, self.t0 = every_s, [], threading.Event(), time.monotonic()

    def _take(self) -> None:
        gb = commit_charge_gb()
        if gb is not None:
            self.samples.append((time.monotonic() - self.t0, gb))

    def run(self) -> None:
        while True:
            self._take()
            if self._stop_flag.wait(self.every_s):
                return

    def finish(self) -> None:
        self._stop_flag.set()
        self.join(timeout=30)
        self._take()

    def peak(self, lo: float = 0.0, hi: float = float("inf")) -> float | None:
        vals = [g for t, g in self.samples if lo <= t <= hi]
        return max(vals) if vals else None

    def at(self, t: float) -> float | None:
        before = [g for s, g in self.samples if s <= t]
        return before[-1] if before else None


# ------------------------------------------------------------------ shared setup


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gen-dir", help="a generation directory or a current.json (default: the agent folder's current.json)")
    p.add_argument("--config", default="a3-v1", help="the search config (rl/search_configs.py)")
    p.add_argument("--lib", help="STS2_LIB_PATH (the native library directory)")
    p.add_argument("--json", help="write the result here")


def add_server(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpu-python", help="the interpreter that runs rl/prior_server.py (default: $SPIRELINE_GPU_PYTHON)")
    p.add_argument("--prior-server-args", help="extra flags for rl/prior_server.py, shlex-split (use / in paths)")


def setup(a: argparse.Namespace):
    """(gen0, gen_loop, agent) with STS2_LIB_PATH set."""
    if getattr(a, "lib", None):
        os.environ["STS2_LIB_PATH"] = a.lib
    import gen0  # noqa: PLC0415

    gen0._setup()  # noqa: SLF001
    import gen_loop  # noqa: PLC0415

    agent = gen_loop.load_agent(a.gen_dir or (gen_loop.DEFAULT_AGENT_ROOT / "current.json"))
    if not agent.prior_dir:
        raise SystemExit("the agent has no combat prior to measure")
    return gen0, gen_loop, agent


def server_opts(a: argparse.Namespace, server_procs: int | None):
    import prior_launch  # noqa: PLC0415

    return prior_launch.ServerOpts(mode="require", gpu_python=getattr(a, "gpu_python", None) or os.environ.get(prior_launch.GPU_PYTHON_ENV),
                                   server_procs=server_procs, extra_args=tuple(shlex.split(getattr(a, "prior_server_args", None) or "")))


def seed_list(gen0, a: argparse.Namespace, count: int) -> list[str]:
    if getattr(a, "stream", None):
        return gen0.stream_seeds(a.stream, count)
    seeds = gen0.load_seeds(a.seeds_file)
    if len(seeds) < count:
        raise SystemExit(f"{a.seeds_file} holds {len(seeds)} seeds, {count} wanted")
    return seeds[:count]


def write_json(path: str | None, doc: dict) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")


def num(x, nd: int = 1) -> str:
    return "-" if x is None else f"{x:,.{nd}f}"


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def agent_facts(agent) -> dict:
    return agent.describe()


# ------------------------------------------------------------------ throughput (9.1)

_BARRIER = None


def _bench_init(barrier) -> None:
    global _BARRIER  # noqa: PLW0603
    _BARRIER = barrier
    import gen0  # noqa: PLC0415

    gen0._init_worker()  # noqa: SLF001


def _bench_task(args: tuple) -> dict:
    """One run to `n` combat decisions in a worker. A warm task also zeroes the client counters and waits at the barrier, so
    every worker takes exactly one warm task (imports, model load, the connection) before the timed block."""
    seed, ix, agent, config, n, warm = args
    import gen0  # noqa: PLC0415

    gen0._setup()  # noqa: SLF001
    import gen_loop  # noqa: PLC0415

    gen_loop.replay_note = lambda *a, **k: False  # a bench plays hooked runs on purpose; the replay warning is noise here
    t0, c0 = time.perf_counter(), time.process_time()
    row = gen0.play_emu_run(seed, ix, config, hook=lambda k, root, session, pos: k >= n, agent=agent)
    out = {"dec": row["decisions"], "sims": row["sims"], "wall": time.perf_counter() - t0, "cpu": time.process_time() - c0,
           "floor": row["floor"], "acts": row["combat_decisions"], "prior_path": row.get("prior_path"),
           "calls": row.get("prior_calls"), "prior_latency_ms": row.get("prior_latency_ms"), "pid": os.getpid()}
    if warm:
        if agent.prior_server:
            import prior_client  # noqa: PLC0415

            prior_client.reset_stats()
        if _BARRIER is not None:
            _BARRIER.wait(900)
    return out


def wait_snapshot(handle, calls: int, timeout_s: float = 20.0) -> tuple[dict, bool]:
    """The server's stats file once it has counted `calls` requests (it is rewritten every 5 s): (stats, exact). Not exact
    when the count never arrived; the last file read is returned then."""
    end, last = time.monotonic() + timeout_s, {}
    while time.monotonic() < end:
        last = handle.read_stats()
        if last.get("calls") == calls:
            return last, True
        time.sleep(0.5)
    return last, False


def run_row(a: argparse.Namespace, gen0, agent, arm: str, procs: int, seeds: list[str], workdir: Path, tag: str) -> dict:
    import prior_launch  # noqa: PLC0415

    run_agent = dataclasses.replace(agent, prior_dir=None) if arm == "noprior" else agent
    handle = None
    base, exact, load0, load1 = {}, True, None, None
    ctx = mp.get_context("spawn")
    try:
        if arm == "server":
            opts = server_opts(a, procs)
            kill = workdir / prior_launch.KILL_FILE_NAME
            prior_launch.resolve_prior_server("require", opts.gpu_python, kill)
            handle = prior_launch.start_prior_server(opts.gpu_python, Path(agent.prior_dir) / "combat_policy.pt",
                                                     lib_path=os.environ.get("STS2_LIB_PATH"), kill_file=kill,
                                                     stats_path=workdir / f"stats-{tag}.json", extra_args=opts.extra_args)
            if handle is None:
                raise prior_launch.PriorServerError("the server did not start")
            run_agent = dataclasses.replace(agent, prior_server=handle.endpoint, prior_authkey=handle.authkey)
        with ctx.Pool(procs, initializer=_bench_init, initargs=(ctx.Barrier(procs),)) as pool:
            warm = pool.map(_bench_task, [(seeds[0], 0, run_agent, a.config, 2, True)] * procs, chunksize=1)
            if handle is not None:
                base, exact = wait_snapshot(handle, sum((w["calls"] or {}).get("server", 0) for w in warm))
            load0 = machine_load()
            t0 = time.perf_counter()
            tasks = pool.map(_bench_task, [(s, i, run_agent, a.config, a.decisions, False) for i, s in enumerate(seeds)], chunksize=1)
            wall = time.perf_counter() - t0
            load1 = machine_load()
    finally:
        if handle is not None:
            handle.stop()
    server = None
    if handle is not None:
        final = handle.read_stats()
        server = {"cpu_s": final.get("host_cpu_s", 0.0) - base.get("host_cpu_s", 0.0),
                  "calls": final.get("calls", 0) - base.get("calls", 0), "batches": final.get("batches", 0) - base.get("batches", 0),
                  "baseline_exact": exact, "ready_s": final.get("ready_s"), "overlong": final.get("overlong"),
                  "replay_ms": final.get("replay_ms"), "queue_wait_ms": final.get("queue_wait_ms")}
    metrics = row_metrics(tasks, wall, server)
    lat = gen0.aggregate_latency(tasks) if arm == "server" else None
    paths: dict[str, int] = {}
    for t in tasks:
        paths[str(t["prior_path"])] = paths.get(str(t["prior_path"]), 0) + 1
    served_share = paths.get("server", 0) / len(tasks) if (arm == "server" and tasks) else None
    fallback_calls = sum((t["calls"] or {}).get("cpu") or 0 for t in tasks) if arm == "server" else None
    return {"arm": arm, "procs": procs, **metrics, "latency_ms": lat, "server": server, "prior_paths": paths,
            "served_share": served_share, "fallback_calls": fallback_calls, "load_before": load0, "load_after": load1, "workers_used": len({t["pid"] for t in tasks})}


def table_throughput(rows: list[dict]) -> str:
    """The 9.1 table: one line per (arm, procs), the mean over the repeats; then one line per repeat with load."""
    keys = list(dict.fromkeys((r["arm"], r["procs"]) for r in rows))
    out = ["| arm | procs | dec/s (wall, total) | dec per CPU-s incl. server | sims per CPU-s incl. server | server CPU share "
           "| server calls/s | mean batch | p50 / p99 call latency (ms) | tasks tagged server |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, procs in keys:
        group = [r for r in rows if (r["arm"], r["procs"]) == (arm, procs)]
        m = mean_metrics([{k: v for k, v in r.items() if isinstance(v, (int, float)) or v is None} for r in group])
        lats = [r["latency_ms"] for r in group if r.get("latency_ms") and r["latency_ms"].get("p50") is not None]
        lat = "-" if not lats else f"{sum(x['p50'] for x in lats) / len(lats):.2f} / {sum(x['p99'] for x in lats) / len(lats):.2f}"
        share = "-" if m["server_cpu_share"] is None else f"{100 * m['server_cpu_share']:.1f}%"
        out.append(f"| {ARM_LABELS[arm]} | {procs} | {num(m['dec_s'], 2)} | {num(m['dec_per_cpu_s'], 2)} | {num(m['sims_per_cpu_s'], 0)} "
                   f"| {share} | {num(m['server_calls_s'], 0)} | {num(m['mean_batch'], 2)} | {lat} | {pct(m['served_share'])} |")
    out += ["", "| repeat | arm | procs | dec/s | load before / after (%) |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['repeat'] + 1} | {r['arm']} | {r['procs']} | {num(r['dec_s'], 2)} | {num(r['load_before'], 0)} / {num(r['load_after'], 0)} |")
    return "\n".join(out)


def throughput_warnings(rows: list[dict]) -> list[str]:
    """One line per server row that is not a clean server row: tasks not tagged `server` or calls that fell back to the CPU
    prior (the row then measures a mix), or a server-CPU baseline that is not exact."""
    out = []
    for r in rows:
        if r["arm"] != "server":
            continue
        tag = f"repeat {r['repeat'] + 1} server:{r['procs']}"
        share = r.get("served_share")
        if (share is not None and share < 1.0) or r.get("fallback_calls"):
            out.append(f"WARNING: {tag}: {pct(share)} of tasks tagged server, {r.get('fallback_calls') or 0} calls fell back to the "
                       "CPU prior: not a clean server row")
        if r.get("server") and not r["server"].get("baseline_exact", True):
            out.append(f"WARNING: {tag}: the server CPU baseline snapshot was not exact (the stats file never showed the warm-up "
                       "call count): server CPU is approximate")
    return out


def cmd_throughput(a: argparse.Namespace) -> int:
    gen0, _, agent = setup(a)
    rows = parse_rows(a.rows)
    seeds = seed_list(gen0, a, a.seeds)
    order = abab_order(rows, a.repeat)
    results = []
    with tempfile.TemporaryDirectory(prefix="spireline-bench-") as tmp:
        for rep, arm, procs in order:
            print(f"row {len(results) + 1}/{len(order)}: repeat {rep + 1} {arm}:{procs} ...", file=sys.stderr, flush=True)
            r = run_row(a, gen0, agent, arm, procs, seeds, Path(tmp), f"{rep}-{arm}{procs}")
            r["repeat"] = rep
            results.append(r)
            print(f"  dec/s {num(r['dec_s'], 2)}  load {num(r['load_before'], 0)}/{num(r['load_after'], 0)}  paths {r['prior_paths']}",
                  file=sys.stderr, flush=True)
    table = table_throughput(results)
    warnings = throughput_warnings(results)
    print(table)
    for line in warnings:
        print(line)
    write_json(a.json, {"tool": "prior_bench throughput", "agent": agent_facts(agent), "config": a.config, "seeds": len(seeds),
                        "decisions_per_seed": a.decisions, "rows": a.rows, "repeat": a.repeat, "order": [f"{r}:{x}:{p}" for r, x, p in order],
                        "results": results, "warnings": warnings, "table": table})
    return 0


# ------------------------------------------------------------------ fullrun and ram (9.1)


def block_facts(summary: dict) -> dict:
    wall_h = summary["wall_s"] / 3600.0
    floor = summary.get("floor") or {}
    return {"runs": summary["runs"], "wall_s": summary["wall_s"], "procs": summary["procs"],
            "runs_per_h": summary["runs"] / wall_h if wall_h else None,
            "decisions_per_h": summary["decisions"] / wall_h if wall_h else None, "mean_floor": floor.get("mean"),
            "decisions_per_s": summary.get("decisions_per_s"), "crashes": summary.get("crashes"),
            "prior_paths": summary.get("prior_paths"),
            "prior_server": None if "prior_server" not in summary else {
                "kind": summary["prior_server"].get("kind"), "restarts": summary["prior_server"].get("restarts"),
                "stats": {k: v for k, v in (summary["prior_server"].get("stats") or {}).items() if k != "device"}},
            "prior_latency_ms": summary.get("prior_latency_ms"), "prior_server_degraded": summary.get("prior_server_degraded")}


def cmd_fullrun(a: argparse.Namespace) -> int:
    gen0, _, agent = setup(a)
    seeds = seed_list(gen0, a, a.n)
    out_dir = Path(a.out)
    result = {}
    for arm in a.order.split(","):
        sampler = CommitSampler(a.sample_s)
        before = commit_charge_gb()
        sampler.start()
        print(f"fullrun {arm}: {a.block_min:g} min ...", file=sys.stderr, flush=True)
        try:
            if arm == "cpu":
                summary = gen0.run_block(seeds, a.config, a.procs, out_dir / arm, block_min=a.block_min, agent=agent)
            elif arm == "server":
                summary = gen0.run_block_served(seeds, a.config, a.procs, out_dir / arm, opts=server_opts(a, a.server_procs),
                                                agent=agent, block_min=a.block_min)
            else:
                raise SystemExit(f"--order names cpu and server, not {arm!r}")
        finally:
            sampler.finish()
        peak = sampler.peak()
        result[arm] = {**block_facts(summary), "commit_before_gb": before, "peak_commit_gb": peak,
                       "peak_commit_over_before_gb": None if peak is None or before is None else peak - before,
                       "commit_samples": len(sampler.samples)}
    cpu, srv = result.get("cpu"), result.get("server")
    lines = ["| arm | procs | runs | runs/h | decisions/h | mean floor | peak commit (GB) | prior paths |", "|---|---|---|---|---|---|---|---|"]
    for arm, r in result.items():
        lines.append(f"| {arm} | {r['procs']} | {r['runs']} | {num(r['runs_per_h'], 0)} | {num(r['decisions_per_h'], 0)} | "
                     f"{num(r['mean_floor'], 2)} | {num(r['peak_commit_gb'], 1)} | {r['prior_paths']} |")
    if cpu and srv and cpu["decisions_per_h"] and srv["decisions_per_h"]:
        result["server_over_cpu_decisions"] = srv["decisions_per_h"] / cpu["decisions_per_h"]
        lines.append(f"\nserver / CPU path: decisions per hour x{result['server_over_cpu_decisions']:.3f}")
    table = "\n".join(lines)
    print(table)
    write_json(a.json, {"tool": "prior_bench fullrun", "agent": agent_facts(agent), "config": a.config, "block_min": a.block_min,
                        "sample_s": a.sample_s, "result": result, "table": table})
    return 0


def cmd_ram(a: argparse.Namespace) -> int:
    import prior_launch  # noqa: PLC0415

    gen0, _, agent = setup(a)
    seeds = seed_list(gen0, a, a.n)
    out_dir = Path(a.out)
    opts = server_opts(a, a.procs)
    kill_file = out_dir / prior_launch.KILL_FILE_NAME
    prior_launch.resolve_prior_server("require", opts.gpu_python, kill_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    handle = prior_launch.start_prior_server(opts.gpu_python, Path(agent.prior_dir) / "combat_policy.pt",
                                             lib_path=os.environ.get("STS2_LIB_PATH"), kill_file=kill_file,
                                             stats_path=out_dir / "prior_server_stats.json", extra_args=opts.extra_args)
    if handle is None:
        raise prior_launch.PriorServerError("the server did not start")
    try:
        return _ram_block(a, gen0, agent, seeds, out_dir, handle)
    finally:
        handle.stop()  # run_block stops it too; this covers a failure before or around the block


def _ram_block(a: argparse.Namespace, gen0, agent, seeds: list[str], out_dir: Path, handle) -> int:
    sampler = CommitSampler(a.sample_s)
    kills: list[float] = []
    done = threading.Event()

    def killer() -> None:
        if done.wait(a.after_s):
            return
        first = handle.proc
        kills.append(time.monotonic() - sampler.t0)
        first.terminate()  # this handle's own process, never an image name
        if a.kills > 1:  # the job restarts a dead server once: kill that one too, so the block cannot recover
            end = time.monotonic() + 120
            while time.monotonic() < end and handle.proc is first and not done.is_set():
                time.sleep(0.2)
            if handle.proc is not first and handle.proc is not None:
                handle.proc.terminate()

    before = commit_charge_gb()
    sampler.start()
    thread = threading.Thread(target=killer, name="server-killer", daemon=True)
    thread.start()
    try:
        summary = gen0.run_block(seeds, a.config, a.procs, out_dir / "block", block_min=a.block_min, agent=agent,
                                 prior_server=handle, server_procs=a.procs)
    finally:
        done.set()
        sampler.finish()
        thread.join(timeout=30)
    kill_t = kills[0] if kills else None
    result = {**block_facts(summary), "after_s": a.after_s, "killed_at_s": kill_t, "commit_before_gb": before,
              "peak_commit_before_kill_gb": sampler.peak(0, kill_t) if kill_t is not None else sampler.peak(),
              "peak_commit_after_kill_gb": sampler.peak(kill_t) if kill_t is not None else None,
              "commit_at_kill_gb": sampler.at(kill_t) if kill_t is not None else None, "commit_samples": len(sampler.samples),
              "timeline": [[round(t, 1), round(g, 3)] for t, g in sampler.samples[:: max(1, len(sampler.samples) // 200)]]}
    table = (f"| procs | killed at (s) | commit before block (GB) | peak before kill (GB) | peak after kill (GB) | prior paths |\n"
             f"|---|---|---|---|---|---|\n| {a.procs} | {num(kill_t, 0)} | {num(before, 2)} | {num(result['peak_commit_before_kill_gb'], 2)} "
             f"| {num(result['peak_commit_after_kill_gb'], 2)} | {result['prior_paths']} |")
    print(table)
    write_json(a.json, {"tool": "prior_bench ram", "agent": agent_facts(agent), "config": a.config, "result": result, "table": table})
    return 0


# ------------------------------------------------------------------ quality (9.3)


def load_arm(gen_loop, arm_dir: Path) -> tuple[list[dict], dict[str, dict], dict[str, list]]:
    """(rows, record by seed, sample digests by seed) of one finished block that recorded its shards."""
    from run_train import shards  # noqa: PLC0415

    rows = gen_loop.arm_rows(arm_dir)
    sdir = arm_dir / "shards"
    manifest = gen_loop.read_json(sdir / "shards.json")
    records = {r["seed"]: r for name in manifest["run_shards"] for r in shards.read_shard(sdir / name)}
    samples: dict[str, list] = {}
    for name in manifest["combat_parts"]:
        samples.update(sample_digests(gen_loop.read_part(sdir / name)))
    return rows, records, samples


def cmd_quality(a: argparse.Namespace) -> int:
    gen0, gen_loop, agent = setup(a)
    seeds = seed_list(gen0, a, a.seeds)
    out_dir = Path(a.out)
    record = not a.no_record
    print(f"quality: cpu arm at {a.procs} procs ...", file=sys.stderr, flush=True)
    s_cpu = gen0.run_block(seeds, a.config, a.procs, out_dir / "cpu", agent=agent, record=record)
    print(f"quality: server arm at {a.server_procs} procs ...", file=sys.stderr, flush=True)
    s_srv = gen0.run_block_served(seeds, a.config, a.procs, out_dir / "server", opts=server_opts(a, a.server_procs), agent=agent,
                                  record=record)
    rows_cpu, rows_srv = gen_loop.arm_rows(out_dir / "cpu"), gen_loop.arm_rows(out_dir / "server")
    divergence = None
    if record:
        _, rec_c, smp_c = load_arm(gen_loop, out_dir / "cpu")
        _, rec_s, smp_s = load_arm(gen_loop, out_dir / "server")
        divergence = {s: first_divergence(rec_c.get(s), smp_c.get(s, []), rec_s.get(s), smp_s.get(s, []))
                      for s in {r["seed"] for r in rows_cpu} & {r["seed"] for r in rows_srv}}
    stats = quality_stats(rows_cpu, rows_srv, divergence, a.boot)
    paths = {"cpu": s_cpu.get("prior_paths"), "server": s_srv.get("prior_paths")}
    lines = ["| statistic | value |", "|---|---|", f"| paired seeds | {stats['paired_seeds']} |",
             f"| identical rows ({', '.join(IDENTITY_KEYS)}) | {stats['identical_rows']} ({100 * (stats['identical_rows_fraction'] or 0):.1f}%) |"]
    if divergence is not None:
        lines.append(f"| identical decision for decision (tokens, visits, record) | {stats['identical_decision_for_decision']} "
                     f"({100 * (stats['identical_decision_for_decision_fraction'] or 0):.1f}%) |")
        lines.append(f"| first diverging floor (histogram) | {stats['first_diverging_floor_histogram']} |")
        lines.append(f"| first divergence by level | {stats['first_divergence_level']} |")
    ci, ci90, v = stats["floor"]["diff_ci"], stats["floor"]["diff_ci90"], stats["verdict"]
    if ci["n"]:
        lines += [f"| mean floor, CPU / server | {stats['floor']['cpu_mean']:.3f} / {stats['floor']['server_mean']:.3f} |",
                  f"| paired floor difference (server - CPU), 95% cluster-bootstrap CI | {ci['mean']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}] |",
                  f"| standard error | {ci['se']:.3f} |",
                  f"| 90% cluster-bootstrap CI (the rule: within +-{v['bound']:.2f}) | [{ci90['lo']:+.3f}, {ci90['hi']:+.3f}] -> {v['floor_ok']} |"]
    bk = stats["boss_kill"]
    lines += [f"| Act-1 boss kills, CPU / server | {bk['cpu']} / {bk['server']} |",
              f"| discordant pairs (server only / CPU only), McNemar two-sided p | {bk['server_only']} / {bk['cpu_only']}, {bk['p_two_sided']:.4f} |",
              f"| sims per combat decision, CPU / server (ratio) | {num(stats['sims_per_decision']['cpu'], 2)} / {num(stats['sims_per_decision']['server'], 2)} ({num(stats['sims_per_decision']['ratio'], 4)}) |",
              f"| decisions per run, CPU / server (ratio) | {num(stats['decisions_per_run']['cpu'], 1)} / {num(stats['decisions_per_run']['server'], 1)} ({num(stats['decisions_per_run']['ratio'], 4)}) |",
              f"| both ratios within 2% | {v['ratios_ok']} |", f"| boss kills not significant at 0.05 | {v['boss_ok']} |",
              f"| pairs lost to a crash / to a missing run (both must be 0) | {stats['crashed_pairs_dropped']} / {stats['unpaired_seeds']} -> {v['pairs_ok']} |",
              f"| runs tagged server in the server arm (>= 99%) / tagged cpu in the CPU arm (100%) | {pct(v['served_share'])} / {pct(v['cpu_share'])} -> {v['paths_ok']} |",
              f"| **equivalent** (90% CI within +-{v['bound']:.2f}, McNemar p >= 0.05, ratios within 2%, paths as tagged, no lost pair) | **{v['equivalent']}** |",
              f"| prior paths | {paths} |"]
    table = "\n".join(lines)
    print(table)
    write_json(a.json, {"tool": "prior_bench quality", "agent": agent_facts(agent), "config": a.config, "recorded": record,
                        "stats": stats, "prior_paths": paths, "table": table})
    return 0


# ------------------------------------------------------------------ command line


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Measurements of docs/rl/GPU_INFERENCE.md sections 9.1 and 9.3.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("throughput", help="9.1 rows: one fixed seed list, ABAB repeats, server CPU in the denominators")
    t.add_argument("--seeds-file", required=True)
    t.add_argument("--seeds", type=int, default=48, help="the first N seeds of the file, the same for every row (default 48: 2 per worker on average at 24 workers)")
    t.add_argument("--rows", default="noprior:16,cpu:16,server:16,server:20,server:24", help="arm:procs list; arm is noprior, cpu or server")
    t.add_argument("--decisions", type=int, default=100, help="combat decisions per seed before the run is ended")
    t.add_argument("--repeat", type=int, default=2)
    add_common(t)
    add_server(t)
    t.set_defaults(func=cmd_throughput)

    f = sub.add_parser("fullrun", help="9.1 full-run A/B: run_block on the CPU path against the server, same seed stream")
    f.add_argument("--seeds-file")
    f.add_argument("--stream", help="seeds from gen0.stream_seeds(NAME) instead of a file")
    f.add_argument("--n", type=int, default=4000, help="seeds in the stream or file prefix (more than the block can play)")
    f.add_argument("--block-min", type=float, default=15.0)
    f.add_argument("--procs", type=int, default=16, help="workers on the CPU path")
    f.add_argument("--server-procs", type=int, required=True, help="workers on the server path (the best row of the throughput table)")
    f.add_argument("--order", default="cpu,server", help="the arms in the order they run")
    f.add_argument("--sample-s", type=float, default=10.0, help="seconds between commit-charge samples")
    f.add_argument("--out", required=True, help="a work directory for the two blocks")
    add_common(f)
    add_server(f)
    f.set_defaults(func=cmd_fullrun)

    r = sub.add_parser("ram", help="9.1 RAM case: a server block whose server is killed after --after-s")
    r.add_argument("--seeds-file")
    r.add_argument("--stream")
    r.add_argument("--n", type=int, default=4000)
    r.add_argument("--procs", type=int, required=True, help="workers (the server count under test)")
    r.add_argument("--block-min", type=float, default=4.0)
    r.add_argument("--after-s", type=float, default=60.0)
    r.add_argument("--kills", type=int, default=2, help="2: kill the restarted server as well, so no worker finds a server again")
    r.add_argument("--sample-s", type=float, default=10.0)
    r.add_argument("--out", required=True)
    add_common(r)
    add_server(r)
    r.set_defaults(func=cmd_ram)

    q = sub.add_parser("quality", help="9.3: the CPU path against the server on the same seeds")
    q.add_argument("--seeds-file", required=True)
    q.add_argument("--seeds", type=int, default=1000)
    q.add_argument("--procs", type=int, default=16, help="workers on the CPU path")
    q.add_argument("--server-procs", type=int, required=True)
    q.add_argument("--boot", type=int, default=BOOT_DRAWS)
    q.add_argument("--no-record", action="store_true", help="rows only: run-level identity, no first-divergence search")
    q.add_argument("--out", required=True)
    add_common(q)
    add_server(q)
    q.set_defaults(func=cmd_quality)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd in ("fullrun", "ram") and not (args.seeds_file or args.stream):
        args.stream = "prior-bench"
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
