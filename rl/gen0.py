"""Generation 0 (PLAN.md section 11 round 6): seed blocks, the emulator driver, and the paired result maths.

Generation 0 is the PLAY agent with no learned prior: IS-MCTS in fights (`live_search.search_live`, every simulation forks
the position with resampling), the rules policy elsewhere. The same agent plays in the real game through
`referee/envproto.py play`; this file plays it on the emulator alone and compares the two on paired seeds.

    python rl/gen0.py seeds --name E0-2026-10-23 --n 500 --out referee/seeds/E0-2026-10-23.json
    python rl/gen0.py emu --seeds-file F --sims 60 --procs 16 --out DIR [--n N] [--block-min 20]
    python rl/gen0.py report --emu DIR [--emu DIR2] --real DIR --out summary.json

The pure parts (seed derivation, Wilson and paired intervals, the result writer) import no native code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
AGENT_SEED = 20261023  # the base agent seed of generation 0's blocks (an agent seed, never a game seed)
RUN_WALL_CAP_S = 1800.0
Z = 1.959964

# ------------------------------------------------------------------ seeds


def e_block(name: str, n: int) -> list[str]:
    """`n` game seeds, a pure function of the block's name: sha256("<name>:<i>") mapped onto 10 characters."""
    out: list[str] = []
    for i in range(n):
        digest = hashlib.sha256(f"{name}:{i}".encode()).digest()
        out.append("".join(ALPHABET[b % len(ALPHABET)] for b in digest[:10]))
    return out


def load_seeds(path: str | Path) -> list[str]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    seeds = doc["seeds"] if isinstance(doc, dict) else doc
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds in the block")
    return list(seeds)


def write_seed_file(name: str, n: int, out: str | Path) -> list[str]:
    seeds = e_block(name, n)
    doc = {"name": name, "n": n, "rule": "sha256(name:i) bytes 0..9 each mod 32 into " + ALPHABET, "seeds": seeds}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return seeds


# ------------------------------------------------------------------ the maths


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """The Wilson score interval for k successes in n trials."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    # the endpoints are exact at k = 0 and k = n; the formula leaves rounding dust there
    return (0.0 if k == 0 else max(0.0, c - h), 1.0 if k == n else min(1.0, c + h))


def mean_ci(xs: list[float], z: float = Z) -> dict:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    m = sum(xs) / n
    if n < 2:
        return {"n": n, "mean": m, "lo": None, "hi": None}
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = z * sd / math.sqrt(n)
    return {"n": n, "mean": m, "lo": m - h, "hi": m + h}


def paired(a: dict[str, float], b: dict[str, float]) -> dict:
    """Mean of a - b over the seeds both have, with a normal interval on the paired differences."""
    seeds = sorted(set(a) & set(b))
    return {**mean_ci([a[s] - b[s] for s in seeds]), "seeds": len(seeds)}


def engine_block(runs: list[dict]) -> dict:
    """One engine's table: wins with a Wilson interval, mean floor with a normal interval, crashes, rates."""
    n = len(runs)
    wins = sum(1 for r in runs if r["win"])
    crashes = sum(1 for r in runs if r.get("crash"))
    dec = sum(r["decisions"] for r in runs)
    wall = sum(r["wall"] for r in runs)
    lo, hi = wilson(wins, n)
    return {"runs": n, "wins": wins, "win_rate": wins / n if n else None, "win_lo": lo, "win_hi": hi,
            "floor": mean_ci([r["floor"] for r in runs]), "crashes": crashes, "decisions": dec,
            "decisions_per_s_per_proc": dec / wall if wall else None, "run_wall_s": wall,
            "ai_state_assumed": sum(r.get("ai_state_assumed") or 0 for r in runs)}


def compare(emu: list[dict], real: list[dict], real_summary: dict | None = None) -> dict:
    """Both engines' tables on the seeds they share, and G = W_emu - W_real and the floor gap, paired."""
    shared = {r["seed"] for r in emu} & {r["seed"] for r in real}
    e = [r for r in emu if r["seed"] in shared]
    g = [r for r in real if r["seed"] in shared]
    out = {"paired_seeds": len(shared), "emu": engine_block(e), "real": engine_block(g),
           "G": paired({r["seed"]: float(r["win"]) for r in e}, {r["seed"]: float(r["win"]) for r in g}),
           "floor_gap": paired({r["seed"]: float(r["floor"]) for r in e}, {r["seed"]: float(r["floor"]) for r in g})}
    if real_summary:
        out["divergence_rate"] = real_summary.get("divergence_rate")
        out["diverged_decisions"] = real_summary.get("diverged_decisions")
        out["real_decisions_per_s_wall"] = real_summary.get("decisions_per_s")
    return out


def result_json(runs: list[dict], wall_s: float, generation: int = 0) -> dict:
    """The yield keys of tools/lab/mix.py AGENT_COLUMNS for a block of runs."""
    t = engine_block(runs)
    return {"runs": t["runs"], "decisions": t["decisions"], "decisions_s": round(t["decisions"] / wall_s, 2) if wall_s else None,
            "generation": generation, "wins": t["wins"], "mean_floor": round(t["floor"]["mean"], 3) if t["runs"] else None,
            "crashes": t["crashes"]}


def real_runs(runs_jsonl: str | Path) -> list[dict]:
    """envproto play's runs.jsonl as the per-run rows of the emulator driver (a crash is a loss on floor 0)."""
    rows = []
    for line in Path(runs_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("ev") == "end":
            win = str(e.get("end")).lower() in ("win", "victory", "won")
            rows.append({"seed": e["seed"], "win": win, "floor": e.get("floor") or 0, "decisions": (e.get("stats") or {}).get("decisions", 0),
                         "wall": e.get("s", 0.0), "crash": False,
                         "ai_state_assumed": e.get("assumed", 0)})
        elif e.get("ev") == "crash":
            rows.append({"seed": e["seed"], "win": False, "floor": 0, "decisions": 0, "wall": e.get("s", 0.0), "crash": True,
                         "error": e.get("error")})
    return rows


# ------------------------------------------------------------------ the emulator driver


def _setup() -> None:
    lib = os.environ.get("STS2_LIB_PATH")
    if not lib:
        lib = next((str(d) for d in (ROOT / "out", ROOT.parent / "sts2" / "out") if (d / "Sts2Emulator.dll").exists()), "")
        if lib:
            os.environ["STS2_LIB_PATH"] = lib
    os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")
    for p in (ROOT / "rl", ROOT / "src", ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _init_worker() -> None:
    _setup()
    from sts2_gym import procutil  # noqa: PLC0415

    procutil.lower_priority()  # BelowNormal on Windows


def play_emu_run(seed: str, ix: int, sims: int | str, ascension: int = 10, agent_seed: int = AGENT_SEED,
                 hook=None) -> dict:
    """One whole run on the emulator under generation 0: search_live in fights, the rules policy elsewhere.

    `sims` is a flat budget or a search config name (`search_configs`). `hook(k, root, session, position)` is called
    at every combat decision k (before its search, on the oracle root it must only fork); a true return ends the run
    there ("hooked"). The hook never touches the agent's random streams, so the run replays exactly.
    """
    import numpy as np  # noqa: PLC0415

    _setup()
    import live_search  # noqa: PLC0415
    import search_configs  # noqa: PLC0415
    from run_train import engine, selfplay  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    t0 = time.perf_counter()
    rng = np.random.default_rng([agent_seed, ix])
    explore = np.random.default_rng([agent_seed, ix, 1])
    cfg = search_configs.get(sims) if isinstance(sims, str) else search_configs.flat(int(sims))
    rules = selfplay.RulesPlayer()
    s = engine.RunSession(seed, ascension, engine.CombatBudget(cfg.sims, cfg.sims))
    decisions = assumed = k = sims_spent = 0
    end = "?"
    try:
        env = s.env
        while True:
            if time.perf_counter() - t0 > RUN_WALL_CAP_S:
                end = "timeout"
                break
            d = s.decision()
            if d is None and s.phase != engine.rc.PHASE_COMBAT:
                end = "win" if s.won else "death"
                break
            if d is not None:
                view = s.view()
                s.apply(d.options[rules.pick(view, d, explore).index])
                decisions += 1
                continue
            root = s.combat(fork_combat=True).combat_fork(None)
            try:
                pos = search_configs.position_of(root, s.info) if (cfg.schedule or hook) else None
                if hook is not None and hook(k, root, s, pos):
                    end = "hooked"
                    break
                k += 1
                res = live_search.search_live(root.handle, seed=int(rng.integers(1 << 30)), **cfg.search_kwargs(pos))
                sims_spent += res.sims
            finally:
                native.destroy(root.handle) if not hasattr(root, "close") else root.close()
            assumed += bool(res.ai_state_assumed)
            st = native.run_step(s.handle, res.best.action, res.best.target, env._run_obs_buf, env._run_rew_buf,
                                 env._run_terminal_buf, env._run_truncated_buf)
            if st != 0:
                raise RuntimeError(f"combat action refused (status {st})")
            decisions += 1
        floor = int(s.info["floor"])
        return {"seed": seed, "ix": ix, "win": end == "win", "floor": floor, "end": end, "decisions": decisions,
                "wall": round(time.perf_counter() - t0, 2), "ai_state_assumed": assumed, "crash": False,
                "combat_decisions": k, "sims": sims_spent, "config": cfg.name}
    finally:
        s.close()


def _run_safe(seed: str, ix: int, sims: int | str, ascension: int) -> dict:
    t0 = time.perf_counter()
    try:
        return play_emu_run(seed, ix, sims, ascension)
    except Exception as e:  # noqa: BLE001 -- an engine or agent defect is a crash of that run: a loss, reported
        return {"seed": seed, "ix": ix, "win": False, "floor": 0, "end": "crash", "decisions": 0, "wall": round(time.perf_counter() - t0, 2),
                "ai_state_assumed": 0, "crash": True, "error": f"{type(e).__name__}: {e}"[:300]}


def run_block(seeds: list[str], sims: int | str, procs: int, out: Path, *, ascension: int = 10, block_min: float | None = None,
              start_ix: int = 0) -> dict:
    """Play the seeds on a spawn pool of `procs` single-emulator processes. With `block_min`, stop starting runs after that
    many minutes (the seeds are then a stream) and let the ones in flight finish. Each run is appended to runs.jsonl."""
    out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    rows: list[dict] = []
    t0 = time.time()
    deadline = t0 + block_min * 60 if block_min else None
    runs_file = out / "runs.jsonl"
    runs_file.write_text("", encoding="utf-8")
    with ctx.Pool(procs, initializer=_init_worker, maxtasksperchild=20) as pool, open(runs_file, "a", encoding="utf-8") as fh:
        pending: list = []
        nxt = 0
        while nxt < len(seeds) or pending:
            while nxt < len(seeds) and len(pending) < procs and (deadline is None or time.time() < deadline):
                pending.append(pool.apply_async(_run_safe, (seeds[nxt], start_ix + nxt, sims, ascension)))
                nxt += 1
            if deadline is not None and time.time() >= deadline:
                nxt = len(seeds)
            still = []
            for p in pending:
                if p.ready():
                    r = p.get()
                    rows.append(r)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                else:
                    still.append(p)
            pending = still
            time.sleep(0.5)
    wall = time.time() - t0
    summary = {"sims": sims, "procs": procs, "ascension": ascension, "agent_seed": AGENT_SEED, "wall_s": round(wall, 1),
               **engine_block(rows), "decisions_per_s": round(sum(r["decisions"] for r in rows) / wall, 2) if wall else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "result.json").write_text(json.dumps(result_json(rows, wall)), encoding="utf-8")
    return summary


def stream_seeds(job: str, n: int = 100000) -> list[str]:
    """A self-play block's seeds: a stream named by the job, never the E-block's (names differ, so the seeds do)."""
    return e_block(f"SP-{job}", n)


def env_bench_result(summary: dict) -> dict:
    """envproto bench's summary.json as the yield keys: the whole-machine and per-process decision rate, memory per run."""
    return {"runs": summary.get("runs_ended"), "decisions": summary.get("decisions"), "decisions_s": summary.get("decisions_per_s"),
            "dec_s": summary.get("decisions_per_s_per_proc"), "mem_mb_per_run": summary.get("mem_mb_mean"),
            "mean_floor": summary.get("mean_floor"),
            "crashes": len(summary.get("crashes") or []) + len(summary.get("hangs") or [])}


def emu_bench_result(doc: dict) -> dict:
    """bench_emu's JSON (the last worker count) as the yield keys: totals per second."""
    row = (doc.get("runs") or [{}])[-1]
    tot = lambda k: (row.get(k) or {}).get("total")  # noqa: E731
    return {"steps_s": tot("steps_per_s"), "clones_s": tot("clones_per_s"), "sims_s": tot("sims_per_s")}


def bench_cmd(a: argparse.Namespace) -> int:
    """bench-env / bench-emu as one command that exits 0 and leaves {out}/result.json."""
    import subprocess  # noqa: PLC0415

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if a.lib:
        env["STS2_LIB_PATH"] = a.lib
    env.setdefault("STS2_ALLOW_STALE_NATIVE", "1")
    if a.cmd == "bench-env":
        cmd = [sys.executable, str(ROOT / "referee" / "envproto.py"), "bench", "--game", a.game, "--out", str(out), "--procs", str(a.procs),
               "--seconds", str(a.seconds), "--recycle", str(a.recycle)]
        rc = subprocess.run(cmd, env=env, check=False).returncode
        summ = out / "summary.json"
        res = env_bench_result(json.loads(summ.read_text(encoding="utf-8"))) if summ.exists() else {}
    else:
        cmd = [sys.executable, str(ROOT / "rl" / "bench_emu.py"), "--n", str(a.procs), "--seconds", str(a.seconds)]
        p = subprocess.run(cmd, env=env, check=False, capture_output=True, text=True)
        rc = p.returncode
        try:
            doc = json.loads(p.stdout)
        except ValueError:
            doc = {}
        (out / "bench.json").write_text(p.stdout, encoding="utf-8")
        res = emu_bench_result(doc) if doc else {}
    (out / "result.json").write_text(json.dumps(res), encoding="utf-8")
    return rc if res else (rc or 1)


def score_real(a: argparse.Namespace) -> int:
    """The real-game block: envproto `play` (lockstep on) over the seed block, then result.json with the yield keys."""
    import subprocess  # noqa: PLC0415

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if a.lib:
        env["STS2_LIB_PATH"] = a.lib
    env.setdefault("STS2_ALLOW_STALE_NATIVE", "1")
    cmd = [sys.executable, str(ROOT / "referee" / "envproto.py"), "play", "--game", a.game, "--out", str(out), "--procs", str(a.procs),
           "--runs", str(a.runs), "--sims", str(a.sims), "--seeds-file", a.seeds_file, "--agent-seed", str(AGENT_SEED),
           "--ascension", str(a.ascension), "--stop-after-h", str(a.stop_after_h)]
    rc = subprocess.run(cmd, env=env, check=False).returncode
    rows = real_runs(out / "runs.jsonl") if (out / "runs.jsonl").exists() else []
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8")) if (out / "summary.json").exists() else {}
    res = result_json(rows, float(summ.get("wall_s") or 0.0)) if rows else {"runs": 0}
    res["divergences"] = summ.get("diverged_decisions")
    (out / "result.json").write_text(json.dumps(res), encoding="utf-8")
    if a.emu_dir and rows:
        emu = [json.loads(x) for x in (Path(a.emu_dir) / "runs.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        (out / "compare.json").write_text(json.dumps(compare(emu, rows, summ), indent=1), encoding="utf-8")
    return rc


# ------------------------------------------------------------------ command line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seeds")
    s.add_argument("--name", required=True)
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--out", required=True)
    e = sub.add_parser("emu")
    e.add_argument("--seeds-file")
    e.add_argument("--stream", help="self-play: seeds from the stream of this job name instead of a file")
    e.add_argument("--n", type=int, default=0)
    e.add_argument("--sims", type=int, default=60)
    e.add_argument("--config", help="a named search config (rl/search_configs.py), instead of --sims")
    e.add_argument("--procs", type=int, default=16)
    e.add_argument("--ascension", type=int, default=10)
    e.add_argument("--block-min", type=float, default=None)
    e.add_argument("--out", required=True)
    e.add_argument("--lib", help="directory of the native library (STS2_LIB_PATH)")
    sc = sub.add_parser("score")
    sc.add_argument("--game", required=True)
    sc.add_argument("--seeds-file", required=True)
    sc.add_argument("--runs", type=int, default=500)
    sc.add_argument("--procs", type=int, default=12)
    sc.add_argument("--sims", type=int, default=60)
    sc.add_argument("--ascension", type=int, default=10)
    sc.add_argument("--stop-after-h", type=float, default=3.5)
    sc.add_argument("--lib")
    sc.add_argument("--emu-dir", help="the emulator baseline to pair with (its runs.jsonl)")
    sc.add_argument("--out", required=True)
    for name in ("bench-env", "bench-emu"):
        b = sub.add_parser(name)
        b.add_argument("--game", default="")
        b.add_argument("--procs", type=int, default=16)
        b.add_argument("--seconds", type=float, default=60.0)
        b.add_argument("--recycle", type=int, default=0)
        b.add_argument("--lib")
        b.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--emu", action="append", required=True, help="DIR:LABEL, e.g. out/emu60:60")
    r.add_argument("--real", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "seeds":
        print(len(write_seed_file(a.name, a.n, a.out)), "seeds")
        return 0
    if a.cmd in ("bench-env", "bench-emu"):
        return bench_cmd(a)
    if a.cmd == "score":
        return score_real(a)
    if a.cmd == "emu":
        if a.lib:
            os.environ["STS2_LIB_PATH"] = a.lib
        if a.stream:
            seeds = stream_seeds(a.stream)
        else:
            seeds = load_seeds(a.seeds_file)
        if a.n:
            seeds = seeds[: a.n]
        summ = run_block(seeds, a.config or a.sims, a.procs, Path(a.out), ascension=a.ascension, block_min=a.block_min)
        print(json.dumps(summ))
        return 0
    real = real_runs(Path(a.real) / "runs.jsonl")
    rs = json.loads((Path(a.real) / "summary.json").read_text(encoding="utf-8"))
    doc: dict = {"real": {"dir": "real", "sims": rs.get("sims"), "wall_s": rs.get("wall_s")}}
    for spec in a.emu:
        d, _, label = spec.rpartition(":")
        emu = [json.loads(x) for x in (Path(d) / "runs.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        doc[f"emu_{label}_vs_real"] = compare(emu, real, rs)
        doc[f"emu_{label}"] = engine_block(emu)
    Path(a.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
