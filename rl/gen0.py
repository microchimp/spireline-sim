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
import collections
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
AGENT_SEED = 20261023  # the base agent seed of generation 0's blocks (an agent seed, never a game seed)
RUN_WALL_CAP_S = 1800.0
RUN_TIMEOUT_S = 600.0  # the dispatcher's wall limit per emulator run (was the in-run cap plus 5 minutes, 2,100 s): the slowest of 3,999 finished
# runs took 123 s, and a hung run was killed, run again and killed again while the other workers idled (2026-11-02 sweep, tails 1,290 to 2,808 s)
REVIEW_GUARD_HELP = ("decide with the review's promotion guard (tools/lab/promotion_guard.py, docs/PROMOTION_GUARD.md); "
                     "default: the SPIRELINE_REVIEW_GUARD variable, off")  # emu, train, promote and rejudge: one reading of the flag
SEQUENTIAL_HELP = ("play the E-block as the funnel (D51, gen_loop.seq_call): a 200-seed smoke block, then 300 seeds at a time, stopping at a "
                   "look that refuses (or promotes early); a block not stopped plays every seed and is judged as without this flag")
# The lab asks a running block to finish by writing STOP_FILE in the block's folder (tools/lab/runner.py FILLER_STOP_FILE; a test keeps the
# names equal). A block that reads it says so by writing STOP_HONORED_FILE first, so a runner never waits for one that cannot.
STOP_FILE = "STOP"
STOP_HONORED_FILE = "stop-honored"
WORKER_RECYCLE = 20  # a worker process is replaced after this many finished runs (between runs only)
Z = 1.959964

# ------------------------------------------------------------------ seeds

import run_fields  # noqa: E402 -- rl/ is on sys.path whenever gen0 is imported
import spawn_pool  # noqa: E402

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
    lost = sum(1 for r in runs if r.get("lost"))
    out = {"runs": n, "wins": wins, "win_rate": wins / n if n else None, "win_lo": lo, "win_hi": hi,
           "floor": mean_ci([r["floor"] for r in runs]), "crashes": crashes, "decisions": dec,
           "decisions_per_s_per_proc": dec / wall if wall else None, "run_wall_s": wall,
           "ai_state_assumed": sum(r.get("ai_state_assumed") or 0 for r in runs)}
    if lost:
        out["lost"] = lost  # seeds whose worker died or timed out on every attempt: counted as losses on floor 0
    fields = run_fields.aggregate(runs)  # mean boss damage, card choices by floor, skip rate: only for rows that carry them
    if fields:
        out["run_fields"] = fields
    return out


def compare(emu: list[dict], real: list[dict], real_summary: dict | None = None) -> dict:
    """Both engines' tables on the seeds they share, and G = W_emu - W_real and the floor gap, paired."""
    shared = {r["seed"] for r in emu} & {r["seed"] for r in real}
    e = [r for r in emu if r["seed"] in shared]
    g = [r for r in real if r["seed"] in shared]
    out = {"paired_seeds": len(shared), "emu": engine_block(e), "real": engine_block(g),
           # the rows each arm holds and those with no partner: a copy arm that plays a prefix of the block (`score --emu-runs`) leaves
           # the rest of the real arm unpaired, and its tables above are over the paired seeds only
           "rows": {"emu": len(emu), "real": len(real), "emu_unpaired": len({r["seed"] for r in emu} - shared),
                    "real_unpaired": len({r["seed"] for r in real} - shared)},
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


RUN_FIELD_KEYS = ("boss_damage", "boss_damage_acts", "cards", "nodes", "rests")  # rl/run_fields.py: what an envproto end event adds to its row


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
                         "ai_state_assumed": e.get("assumed", 0), **{k: e[k] for k in RUN_FIELD_KEYS if k in e}})
        elif e.get("ev") == "crash":
            rows.append({"seed": e["seed"], "win": False, "floor": 0, "decisions": 0, "wall": e.get("s", 0.0), "crash": True,
                         "error": e.get("error")})
    return rows


# ------------------------------------------------------------------ the emulator driver


def session_kwargs(agent, character: str = "IRONCLAD") -> dict:
    """The `RunSession` keyword arguments of an agent's run: its character and the mask version its agent.json names
    (nothing for version 1, so a version-1 run is built exactly as it always was)."""
    return {**({} if character == "IRONCLAD" else {"character": character}),
            **({} if agent.mask_version == 1 else {"mask_version": agent.mask_version})}


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
                 hook=None, agent=None, record: bool = False, character: str = "IRONCLAD") -> dict:
    """One whole run on the emulator under the agent: search_live in fights, the macro policy elsewhere.

    `sims` is a flat budget or a search config name (`search_configs`). `agent` is a `gen_loop.Agent` (none: generation 0,
    no prior, the rules policy). With a prior the search is PUCT over the prior's probabilities; the agent's random
    streams are the same either way. `hook(k, root, session, position)` is called at every combat decision k (before its
    search, on the oracle root it must only fork); a true return ends the run there ("hooked"). The hook never touches the
    agent's random streams, so the run replays exactly. With `record`, a finished run also returns its RunRecord and its
    combat samples under `_shard` (`gen_loop.RunRecorder`); recording only reads the position, so the run is unchanged.
    """
    import numpy as np  # noqa: PLC0415

    _setup()
    import gen_loop  # noqa: PLC0415
    import live_search  # noqa: PLC0415
    import run_fields  # noqa: PLC0415
    import search_configs  # noqa: PLC0415
    from run_train import engine  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    t0 = time.perf_counter()
    rng = np.random.default_rng([agent_seed, ix])
    explore = np.random.default_rng([agent_seed, ix, 1])
    cfg = search_configs.get(sims) if isinstance(sims, str) else search_configs.flat(int(sims))
    agent = agent or gen_loop.Agent()
    if hook is not None and agent.prior_server:
        gen_loop.replay_note("server", "this hooked replay")  # a replay is exact on the CPU path only
    extra = agent.search_extra()
    rules = agent.macro_policy()
    prior_before = _prior_counts(agent)
    rec = None
    if record:
        from combat_tokens import encode  # noqa: PLC0415

        rec = gen_loop.RunRecorder(seed, agent.generation, rules.describe(), cfg.sims, cfg.c, ascension)
    s = engine.RunSession(seed, ascension, engine.CombatBudget(cfg.sims, cfg.sims), **session_kwargs(agent, character))
    decisions = assumed = k = sims_spent = 0
    fields = run_fields.RunFields()  # the row's boss damage, card choices, map nodes and rest choices
    slugs = run_fields.card_slugs()
    fight_boss: bool | None = None  # None: no fight open; else whether the open fight is a boss fight
    end = "?"
    try:
        env = s.env
        while True:
            if time.perf_counter() - t0 > RUN_WALL_CAP_S:
                end = "timeout"
                break
            d = s.decision()
            if rec is not None and rec.fight is not None and (d is not None or s.phase != engine.rc.PHASE_COMBAT):
                rec.close_fight(s, won=d is not None or s.won)  # the fight ended: the run goes on, or it ended in a win/death
            if fight_boss is not None and (d is not None or s.phase != engine.rc.PHASE_COMBAT):
                if fight_boss:
                    fields.boss_over(won=d is not None or s.won)
                fight_boss = None
            if d is None and s.phase != engine.rc.PHASE_COMBAT:
                end = "win" if s.won else "death"
                break
            if d is not None:
                view = s.view()
                pick = rules.pick(view, d, explore)
                if rec is not None:
                    rec.decision(view, d, pick)
                fields.emu_macro(view.floor, d.options[pick.index], slugs)
                s.apply(d.options[pick.index])
                decisions += 1
                continue
            if fight_boss is None:
                fight_boss = s.info["node_type"] == engine.vocab.NODE_BOSS
            if fight_boss:  # the combat observation opens the run observation: the boss's HP at this decision
                fields.boss_enemies(s.act_index, run_fields.obs_enemies(env._run_obs_buf, native.OBS_ENEMY_OFFSET,
                                                                       native.OBS_ENEMY_SLOT_SIZE, native.MAX_ENEMIES))
            if rec is not None and rec.fight is None:
                rec.open_fight(s, s.combat())
            root = s.combat(fork_combat=True).combat_fork(None)
            try:
                pos = search_configs.position_of(root, s.info) if (cfg.schedule or hook) else None
                if hook is not None and hook(k, root, s, pos):
                    end = "hooked"
                    break
                k += 1
                tokens = encode(root.handle, root.obs, root.encounter_id, probe=False) if rec is not None else None
                kw = cfg.search_kwargs(pos, env_prior=False)  # the agent argument alone decides the prior
                kw.update(extra)
                res = live_search.search_live(root.handle, seed=int(rng.integers(1 << 30)), **kw)
                sims_spent += res.sims
                if rec is not None:
                    rec.sample(root, res, tokens)
            finally:
                native.destroy(root.handle) if not hasattr(root, "close") else root.close()
            assumed += bool(res.ai_state_assumed)
            st = native.run_step(s.handle, res.best.action, res.best.target, env._run_obs_buf, env._run_rew_buf,
                                 env._run_terminal_buf, env._run_truncated_buf)
            if st != 0:
                raise RuntimeError(f"combat action refused (status {st})")
            decisions += 1
        floor = int(s.info["floor"])
        prior_path, prior_calls, prior_latency = prior_path_of(agent, prior_before, _prior_counts(agent))
        row = {"seed": seed, "ix": ix, "win": end == "win", "floor": floor, "end": end, "decisions": decisions,
               "wall": round(time.perf_counter() - t0, 2), "ai_state_assumed": assumed, "crash": False,
               "combat_decisions": k, "sims": sims_spent, "config": cfg.name, "generation": agent.generation,
               "character": character, "prior_path": prior_path, "prior_calls": prior_calls}
        if prior_latency is not None:
            row["prior_latency_ms"] = prior_latency
        row.update(fields.row())
        if rec is not None and end in ("win", "death"):
            row["_shard"] = {"record": rec.finish(s, k, won=end == "win", prior_path=prior_path, prior_calls=prior_calls),
                             "samples": rec.samples.arrays()}
        return row
    finally:
        s.close()


def _prior_counts(agent) -> dict | None:
    """This process's prior-client counters when the agent's prior is a `RemotePrior`; None on the in-process path (which does
    not count, and must not need `prior_client`)."""
    if not (agent.prior_dir and agent.prior_server):
        return None
    import prior_client  # noqa: PLC0415

    return prior_client.stats()


def prior_path_of(agent, before: dict | None, after: dict | None) -> tuple[str, dict, dict | None]:
    """(prior_path, prior_calls, latency) of one run from the client counters read before and after it. "none": the agent has
    no prior. "cpu": the in-process prior, or every call fell back. "server": every call was answered by the server (a run that
    made none counts as server unless the process had already fallen back). "mixed": both server and cpu answers. "uniform":
    any call at all was answered by the uniform prior of the memory floor (`prior_client`), whatever else answered the run: the
    run's tag is its weakest prior, which is what the training set's exclude switch needs; the counts show the split, and a
    run that made none follows a process that had already fallen to uniform. `prior_calls` has a "uniform" count only on a
    run with uniform calls. `latency` is the process's reservoir (cumulative over its runs) with its pid, or None."""
    if not agent.prior_dir:
        return "none", {"server": 0, "cpu": 0}, None
    if before is None or after is None:
        return "cpu", {"server": 0, "cpu": None}, None
    srv, cpu = after["server"] - before["server"], after["cpu"] - before["cpu"]
    uni = after.get("uniform", 0) - before.get("uniform", 0)
    if uni:
        path = "uniform"
    elif srv and cpu:
        path = "mixed"
    elif cpu:
        path = "cpu"
    elif srv:
        path = "server"
    elif after.get("uniform"):
        path = "uniform"
    else:
        path = "cpu" if after.get("fallback_reason") else "server"
    lat = after.get("latency_ms")
    return path, {"server": srv, "cpu": cpu, **({"uniform": uni} if uni else {})}, ({**lat, "pid": os.getpid()} if lat else None)


def aggregate_latency(rows: list[dict]) -> dict:
    """The block's client latency from the rows: each worker process's last (fullest) reservoir, weighted by its call count.
    `p50` and `p99` are count-weighted means over the processes, `p99_max` the worst process's p99."""
    best: dict[Any, dict] = {}
    for r in rows:
        lat = r.get("prior_latency_ms")
        if lat and lat.get("n") and lat.get("p50") is not None and lat.get("p99") is not None:
            if lat.get("pid") not in best or lat["n"] > best[lat.get("pid")]["n"]:
                best[lat.get("pid")] = lat
    n = sum(lat["n"] for lat in best.values())
    if not n:
        return {"p50": None, "p99": None, "p99_max": None, "n": 0, "processes": 0}
    return {"p50": round(sum(lat["p50"] * lat["n"] for lat in best.values()) / n, 3),
            "p99": round(sum(lat["p99"] * lat["n"] for lat in best.values()) / n, 3),
            "p99_max": round(max(lat["p99"] for lat in best.values()), 3), "n": n, "processes": len(best)}


def _run_safe(seed: str, ix: int, sims: int | str, ascension: int, agent=None, record: bool = False,
              character: str = "IRONCLAD") -> dict:
    t0 = time.perf_counter()
    try:
        return play_emu_run(seed, ix, sims, ascension, agent=agent, record=record, character=character)
    except Exception as e:  # noqa: BLE001 -- an engine or agent defect is a crash of that run: a loss, reported
        return {"seed": seed, "ix": ix, "win": False, "floor": 0, "end": "crash", "decisions": 0, "wall": round(time.perf_counter() - t0, 2),
                "ai_state_assumed": 0, "crash": True, "error": f"{type(e).__name__}: {e}"[:300],
                "boss_damage": 0.0}  # a crash is a loss on floor 0: the boss was not reached


def lost_row(seed: str, ix: int, status: str, reason: str, attempts: int, wall: float) -> dict:
    """The row of a seed whose worker died or timed out on every attempt: a loss on floor 0 (like a crash), marked `lost`."""
    return {"seed": seed, "ix": ix, "win": False, "floor": 0, "end": "lost", "decisions": 0, "wall": wall, "ai_state_assumed": 0,
            "lost": True, "lost_status": status, "lost_reason": reason[:200], "attempts": attempts, "boss_damage": 0.0}


def resume_rows(out: Path, stamp: dict, seeds: list[str]) -> list[dict]:
    """The rows of `out/runs.jsonl` that a rerun of the block keeps: one per seed of `seeds` (the first), a half-written last line
    dropped. The file is rewritten with just those rows. `out/resume.json` holds the block's identity (agent, config, ascension,
    character, engine): rows written under another identity are not kept, and the file starts empty. A file with no identity
    (written before this existed) is kept when all its seeds belong to `seeds`."""
    out.mkdir(parents=True, exist_ok=True)
    runs_file, stamp_file = out / "runs.jsonl", out / "resume.json"
    want = json.loads(json.dumps(stamp))
    kept: list[dict] = []
    if runs_file.is_file():
        try:
            have = json.loads(stamp_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            have = None
        raw = []
        for line in runs_file.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue  # a line cut by the stop
            if isinstance(r, dict) and r.get("seed") is not None:
                raw.append(r)
        wanted = set(seeds)
        if have == want or (have is None and raw and all(r["seed"] in wanted for r in raw)):
            seen: set[str] = set()
            for r in raw:
                if r["seed"] in wanted and r["seed"] not in seen:
                    seen.add(r["seed"])
                    kept.append(r)
        elif raw:
            print("resume: runs.jsonl was written for another agent or setting; it starts empty", file=sys.stderr, flush=True)
    tmp = runs_file.with_name("runs.jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    os.replace(tmp, runs_file)
    stamp_file.write_text(json.dumps(want), encoding="utf-8")
    return kept


def run_block(seeds: list[str], sims: int | str, procs: int, out: Path, *, ascension: int = 10, character: str = "IRONCLAD",
              block_min: float | None = None, start_ix: int = 0, agent=None, record: bool = False,
              prior_server=None, server_procs: int | None = None, latency_p99_ms: float = 20.0,
              run_timeout_s: float | None = RUN_TIMEOUT_S, recycle: int = WORKER_RECYCLE, resume: bool = False,
              resume_key: dict | None = None, task=None, initializer=_init_worker, stop_file: Path | str | None = None) -> dict:
    """Play the seeds on a spawn pool of `procs` single-emulator processes (`spawn_pool.SupervisedPool`). With `block_min`, stop
    starting runs after that many minutes (the seeds are then a stream) and let the ones in flight finish. Each run is appended
    to runs.jsonl.

    No seed is lost with its worker. A run that is still going after `run_timeout_s` (None: no limit) has its worker killed and
    is written at once as a row with `end: "lost"`, `lost: true`, floor 0 and no win, and not run again (a hang replays, so a second
    attempt only doubles the wait while the other workers idle); summary.json then has `timed_out`, one entry per such seed (seed,
    index, the arm's folder name, the limit, the seconds it ran, floor 0, no win). A worker that exits mid-run is replaced and its
    seed is run again once, then written the same way, so the block always ends with one row per seed. A worker is replaced after `recycle`
    runs, only between runs. With `resume`, runs.jsonl is kept (see `resume_rows`) and its seeds are not played again; the
    summary then has `resumed_rows`. `resume_key` adds to the identity a resumed file must match (the engine). `task` and
    `initializer` are what a worker runs and starts with (tests replace them; both must be importable in the worker).
    With `record`, the finished runs' records and combat samples go to `out/shards` (`gen_loop.ShardSink`), tagged with the
    agent's generation. The sink writes a part when it is full or its oldest run has waited `gen_loop.FLUSH_AGE_S`.

    With `stop_file`, a file of that name appearing is a request to finish (the lab's filler is stopped for a job's cores, a QUIET.lock
    line or the night): the finished runs are written to the shards at once, no new run starts, the runs in flight finish (written as
    each ends) and the block returns as after its time limit, with `stopped` in the summary. A kill that follows loses only the runs
    still in flight. A stop file left from an earlier block is removed at the start; `STOP_HONORED_FILE` beside it says this block
    reads it.

    With `prior_server` (a started `prior_launch.ServerHandle`, see `run_block_served`) the workers' combat prior is that
    server: the pool has `server_procs` workers (default `procs`), a dead server is restarted once on the same endpoint, and the
    server is stopped at the end. summary.json then carries the server's stats, the client latency and `prior_server_degraded`.
    When the restarted server dies too (the second death) the block drains and shrinks: it stops dispatching, waits for the runs
    in flight (their workers fall back as `prior_client` decides: the CPU prior, or a uniform one under the memory floor),
    terminates that pool and goes on, on a fresh pool of `procs` plain CPU-path workers whose agent has no server. Nothing is
    replayed or skipped: the next seed is the next in the stream. summary.json then has `pool_shrunk` = {"from", "to",
    "at_run", "drained_s"}: the two pool widths, the runs finished when the second death was seen (the ones in flight then
    finish after it) and the seconds the drain took; `procs` is the width the block started with."""
    _setup()
    import gen_loop  # noqa: PLC0415

    task = task or _run_safe  # looked up now, so a test can replace gen0._run_safe
    agent = agent or gen_loop.Agent()
    nprocs = server_procs if (prior_server is not None and server_procs) else procs
    if prior_server is not None:
        agent = dataclasses.replace(agent, prior_server=prior_server.endpoint, prior_authkey=prior_server.authkey)
    cfg0 = None
    sink = None
    if record:
        import search_configs  # noqa: PLC0415

        cfg0 = search_configs.get(sims) if isinstance(sims, str) else search_configs.flat(int(sims))
        sink = gen_loop.ShardSink(out, agent.generation, config=cfg0.name, base_sims=cfg0.sims, ascension=ascension,
                                  agent=agent.describe())
    out.mkdir(parents=True, exist_ok=True)
    stop_path = Path(stop_file) if stop_file else None
    if stop_path is not None:
        stop_path.unlink(missing_ok=True)
        (stop_path.parent / STOP_HONORED_FILE).write_text(f"{os.getpid()}\n", encoding="utf-8")
    stopped: dict | None = None
    ctx = mp.get_context("spawn")
    rows: list[dict] = []
    t0 = time.time()
    deadline = t0 + block_min * 60 if block_min else None
    runs_file = out / "runs.jsonl"
    stamp = {"agent": agent.describe(), "sims": sims, "ascension": ascension, "character": character, **(resume_key or {})}
    held = resume_rows(out, stamp, seeds) if resume else []  # the rows a rerun keeps (and writes back); else the file starts empty
    if not resume:
        runs_file.write_text("", encoding="utf-8")
    rows.extend(held)
    gave_up = False
    held_seeds = {r["seed"] for r in held}
    todo = [(ix, s) for ix, s in enumerate(seeds, start_ix) if s not in held_seeds]  # (index in the stream, seed)
    if held:
        print(f"resume: {runs_file.name} holds {len(held)} rows; those {len(held)} seeds are not played again", file=sys.stderr, flush=True)
    resumed = None
    if sink is not None and sink.done:  # a resumed block: the job replays its stream; the seeds its shards hold are not played again
        before = len(todo)
        todo = [(ix, s) for ix, s in todo if s not in sink.done]
        resumed = {"runs_held": len(sink.done), "seeds_skipped": before - len(todo)}
        print(f"resume: the shards hold {resumed['runs_held']} runs; {resumed['seeds_skipped']} of the {len(seeds)} seeds are not "
              "played again", file=sys.stderr, flush=True)
    pool_shrunk = None
    pool = None
    attempts: dict[int, int] = {}  # run index -> failed attempts so far (a died or timed-out worker)
    retry: collections.deque = collections.deque()  # (index, seed) to run again, before any new seed
    counts = {"retried": 0, "lost": 0, "fails_in_a_row": 0}
    timed_out: list[dict] = []  # the seeds whose run outlived `run_timeout_s`: one entry each, in the summary
    fail_limit = max(8, 2 * nprocs)  # this many failed attempts with no success between: the pool itself is broken, not the seeds
    new_decisions = 0
    pools_stats: list[dict] = []

    def new_pool(width: int) -> spawn_pool.SupervisedPool:
        return spawn_pool.SupervisedPool(ctx, width, initializer=initializer, recycle=recycle, run_timeout_s=run_timeout_s)

    try:
        pool = new_pool(nprocs)
        with open(runs_file, "a", encoding="utf-8") as fh:

            def write_row(r: dict) -> None:
                nonlocal new_decisions
                shard = r.pop("_shard", None)
                if sink is not None and shard is not None:
                    sink.add(shard["record"], shard["samples"])
                rows.append(r)
                new_decisions += r.get("decisions") or 0
                fh.write(json.dumps(r) + "\n")
                fh.flush()

            def settle(d: spawn_pool.Done) -> None:
                """One task's end: the row of a finished run; a died one is run again once, then written as `lost`; a timed-out
                one is written as `lost` at once (and listed in `timed_out`), never run again."""
                ix, seed = d.key
                if d.status == "ok":
                    counts["fails_in_a_row"] = 0
                    write_row(d.value)
                    return
                counts["fails_in_a_row"] += 1
                attempts[ix] = attempts.get(ix, 0) + 1
                again = attempts[ix] < 2 and d.status != "timeout"
                print(f"run {seed} (index {ix}): {d.status} after {d.wall_s:g} s ({d.value}); "
                      + ("run again" if again else "written as lost"), file=sys.stderr, flush=True)
                if again:
                    counts["retried"] += 1
                    retry.append((ix, seed))
                else:
                    counts["lost"] += 1
                    if d.status == "timeout":
                        timed_out.append({"seed": seed, "ix": ix, "arm": out.name, "limit_s": run_timeout_s, "wall_s": d.wall_s,
                                          "floor": 0, "win": False, "attempts": attempts[ix]})
                    write_row(lost_row(seed, ix, d.status, str(d.value), attempts[ix], d.wall_s))
                if counts["fails_in_a_row"] >= fail_limit:
                    raise RuntimeError(f"the pool failed {fail_limit} attempts in a row with no run finishing between: last {d.status}: "
                                       f"{d.value}; stopped instead of writing lost rows for the rest of the block")

            nxt = 0
            while nxt < len(todo) or retry or pool.inflight():
                if stop_path is not None and stopped is None and stop_path.exists():
                    stopped = {"at_run": len(rows), "in_flight": pool.inflight(), "after_s": round(time.time() - t0, 1)}
                    nxt = len(todo)  # nothing new starts; a run that died is not run again
                    retry.clear()
                    if sink is not None:
                        sink.flush()  # the runs finished so far reach the shards before anything else can go wrong
                    print(f"stop requested: {stopped['at_run']} runs done, {stopped['in_flight']} in flight, finishing them",
                          file=sys.stderr, flush=True)
                if prior_server is not None and not gave_up and not prior_server.alive() and not prior_server.restart_once():
                    gave_up = True  # the second death: drain, then go on with plain CPU-path workers
                    print("prior-server: the server is gone and cannot be restarted; the block drains its runs in flight and goes "
                          f"on, on {procs} CPU-path workers", file=sys.stderr, flush=True)
                    at_run, t_drain = len(rows), time.time()
                    while pool.inflight():  # no new tasks; the workers in flight fall back under prior_client's memory floor
                        for d in pool.collect():
                            settle(d)
                        if pool.inflight():
                            pool.wait(0.5)
                    drained_s = round(time.time() - t_drain, 1)
                    pools_stats.append(pool.stats)
                    pool.terminate()
                    agent = dataclasses.replace(agent, prior_server=None, prior_authkey=None)
                    pool_shrunk = {"from": nprocs, "to": procs, "at_run": at_run, "drained_s": drained_s}
                    nprocs = procs
                    pool = new_pool(nprocs)
                while pool.free() and (retry or (nxt < len(todo) and (deadline is None or time.time() < deadline))):
                    if retry:
                        ix, seed = retry.popleft()
                    else:
                        ix, seed = todo[nxt]
                        nxt += 1
                    pool.submit((ix, seed), task, (seed, ix, sims, ascension, agent, record, character))
                if deadline is not None and time.time() >= deadline:
                    nxt = len(todo)
                for d in pool.collect():
                    settle(d)
                if sink is not None:
                    if stopped is not None:
                        if sink.pending:
                            sink.flush()  # draining: every run that ends is written at once
                    else:
                        sink.tick()
                pool.wait(0.5)
        pools_stats.append(pool.stats)
    finally:
        if pool is not None:
            pool.terminate()  # a pool that ended its work has no live worker left to stop; one that failed does
        if prior_server is not None:
            prior_server.stop()  # close its stdin, wait 5 s, terminate
    wall = time.time() - t0
    shard_doc = sink.close() if sink is not None else None
    pool_stats = {k: sum(st[k] for st in pools_stats) for k in pools_stats[0]} if pools_stats else {}
    summary = {"sims": sims, "procs": procs, "ascension": ascension, "character": character, "agent_seed": AGENT_SEED,
               "wall_s": round(wall, 1),
               "config": cfg0.name if cfg0 else (sims if isinstance(sims, str) else f"flat-{sims}"), "agent": agent.describe(),
               **engine_block(rows), "decisions_per_s": round(new_decisions / wall, 2) if wall else None}
    summary["run_timeout_s"] = run_timeout_s
    summary["pool"] = {**pool_stats, "retried": counts["retried"], "lost": counts["lost"]}
    if timed_out:
        summary["timed_out"] = timed_out
    if held:
        summary["resumed_rows"] = len(held)
    if stopped is not None:
        summary["stopped"] = {**stopped, "runs_written": len(rows) - len(held), "by": STOP_FILE}
    if shard_doc is not None:
        summary["shards"] = {"runs": shard_doc["runs"], "samples": shard_doc["samples"], "generation": shard_doc["generation"]}
    if resumed is not None:
        summary["resumed"] = resumed
    tags = [r["prior_path"] for r in rows if "prior_path" in r]
    summary["prior_paths"] = dict(sorted(Counter(tags).items()))
    if prior_server is not None:
        lat = aggregate_latency(rows)
        summary["procs"] = pool_shrunk["from"] if pool_shrunk else nprocs  # the width the block started with
        if pool_shrunk:
            summary["pool_shrunk"] = pool_shrunk
        summary["prior_server"] = {"kind": prior_server.kind, "restarts": prior_server.restarts, "stats": prior_server.read_stats()}
        summary["prior_latency_ms"] = lat
        summary["prior_server_degraded"] = pool_shrunk is not None or (bool(tags) and (
            tags.count("server") < 0.99 * len(tags) or (lat["p99"] is not None and lat["p99"] > latency_p99_ms)))
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "result.json").write_text(json.dumps({**result_json(rows, wall, agent.generation), "character": character}),
                                     encoding="utf-8")
    return summary


def run_block_served(seeds: list[str], sims: int | str, procs: int, out: Path, *, opts, agent=None, **kw) -> dict:
    """`run_block` with the prior server that `opts` (`prior_launch.ServerOpts`) asks for: resolved (`off` when `auto` has no GPU
    python or finds the kill file), started for the agent's combat prior (twice under `auto`, once under `require`), and
    handed to the block. The kill file is `<lab>/GPU_OFF` with a lab, else `<out>/GPU_OFF`. `require` that cannot start raises."""
    _setup()
    import gen_loop  # noqa: PLC0415
    import prior_launch  # noqa: PLC0415

    agent = agent or gen_loop.Agent()
    out = Path(out)
    handle = None
    if opts.mode != "off" and not agent.prior_dir:
        prior_launch.log("the agent has no combat prior to serve: CPU path")
    elif opts.mode != "off":
        kill = Path(opts.lab or out) / prior_launch.KILL_FILE_NAME
        if prior_launch.resolve_prior_server(opts.mode, opts.gpu_python, kill) == "on":
            out.mkdir(parents=True, exist_ok=True)
            for _ in range(1 if opts.mode == "require" else 2):
                handle = prior_launch.start_prior_server(
                    opts.gpu_python, Path(agent.prior_dir) / "combat_policy.pt", lib_path=os.environ.get("STS2_LIB_PATH"),
                    kill_file=kill, stats_path=out / "prior_server_stats.json", extra_args=opts.extra_args)
                if handle is not None:
                    break
            if handle is None and opts.mode == "require":
                raise prior_launch.PriorServerError("--prior-server require: the server did not start")
            if handle is None:
                prior_launch.log("auto: the server did not start: CPU path")
    try:
        return run_block(seeds, sims, procs, out, agent=agent, prior_server=handle, server_procs=opts.server_procs,
                         latency_p99_ms=opts.latency_p99_ms, **kw)
    finally:
        if handle is not None:
            handle.stop()  # run_block stops it too; this covers a failure before its loop


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
        cmd = [sys.executable, str(ROOT / "rl" / "bench_emu.py"), "--n", str(a.procs), "--seconds", str(a.seconds),
               "--engine", a.engine]  # bench_emu.py takes its engine from the flag alone, so the environment never picks it
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


def exclude_prior_paths(text: str | None) -> tuple[str, ...]:
    """`--exclude-prior-paths cpu,mixed` as a tuple of tags (validated where the training set is built)."""
    return tuple(t.strip() for t in (text or "").split(",") if t.strip())


def train_cmd(a: argparse.Namespace) -> int:
    """`train --gen N`, or with --marker-dir the lowest LAB/train/ready-<gen> marker (the lab consumes the marker when the
    command succeeds). Writes AGENT/gen<N+1>/candidate; with --promote then plays the promotion block."""
    _setup()
    import gen_loop  # noqa: PLC0415

    gen, retrain, target, attempt = a.gen, 0, getattr(a, "candidate_generation", None), 1
    if a.marker_dir:
        marks = gen_loop.ready_markers(Path(a.marker_dir))
        if not marks:
            print("no ready-<gen> marker", file=sys.stderr)
            return 2
        gen, retrain = marks[0][0], gen_loop.marker_parts(marks[0][1])[1]
        named = gen_loop.marker_target(marks[0][1])  # ready-<gen>-g<target> names the candidate generation
        if named is not None and target is not None and target != named:
            print(f"--candidate-generation {target} contradicts the marker {marks[0][1].name}", file=sys.stderr)
            return 2
        target = named if named is not None else target
        attempt = gen_loop.marker_attempt(marks[0][1])  # ready-<gen>-g<target>-a<n>: the built candidate gen<target>/candidate-a<n>
    if gen is None:
        print("--gen or --marker-dir is needed", file=sys.stderr)
        return 2
    cand_gen = gen + 1 if target is None else target
    spent = gen_loop.attempts_spent(Path(a.agent_root) / "promotions.tsv", cand_gen, True if a.review_guard else None)
    if spent is not None:  # the review guard's cap: `promote` would refuse this candidate, so nothing is trained (rc 0: a finished round)
        from tools.lab import jobs  # noqa: PLC0415

        done = jobs.consume_marker(marks[0][1]) if a.marker_dir else None
        print(f"nothing trained: generation {cand_gen} already had {spent} judged candidates (the review guard caps promotion attempts at "
              f"{gen_loop.review_guard_module().ATTEMPT_CAP}), so a further candidate could not be judged"
              + (f"; {marks[0][1].name} renamed {done.name}" if done else ""))
        return 0
    built = gen_loop.built_candidate(Path(a.agent_root), cand_gen, retrain, attempt)  # `gen0.py build` assembled it: judged as it is, never trained over
    if attempt > 1 and built is None:  # an attempt marker is written by `build`; training would write over the first attempt's folder
        print(f"{marks[0][1].name} names attempt {attempt} of generation {cand_gen}, but gen{cand_gen}/"
              f"{gen_loop.candidate_dirname(0, attempt)} is not a built candidate: nothing trained", file=sys.stderr)
        return 2
    if built is not None and (built / "promotion.json").is_file():
        from tools.lab import jobs  # noqa: PLC0415

        done = jobs.consume_marker(marks[0][1]) if a.marker_dir else None
        print(f"nothing trained or judged: gen{cand_gen}/{built.name} was built (not trained) and its promotion.json says it was "
              "judged already" + (f"; {marks[0][1].name} renamed {done.name}" if done else ""))
        return 0
    if built is not None:
        cand = built
        print(f"gen{cand_gen}/{built.name} was built, not trained (agent.json `{gen_loop.BUILT_KEY}`): it is judged as it is", flush=True)
    else:
        cap = a.max_fights or gen_loop.MAX_FIGHTS
        cand = gen_loop.train_generation(gen, Path(a.shards_root), Path(a.agent_root), device=a.device, epochs=a.epochs,
                                         max_fights=cap * (retrain + 1), threads=a.threads, vrun=not a.no_vrun,
                                         retrain=retrain, policy_target=a.policy_target,  # a retrain draws on the larger window
                                         exclude_prior_paths=exclude_prior_paths(getattr(a, "exclude_prior_paths", None)),
                                         seed_dirs=[gen_loop.lab_seed_dir(a.lab)] if a.lab else [],  # the lab's generated E-blocks too
                                         vrun_seed=getattr(a, "vrun_seed", None), candidate_generation=target)
    target = gen + 1 if target is None else target
    print(json.dumps({"candidate": str(cand.parent.name) + "/" + cand.name, "generation": target}))
    if a.promote:
        if not (a.emu_python and a.seeds_file):
            print("--promote needs --emu-python and --seeds-file", file=sys.stderr)
            return 2
        sha, head = gen_loop.candidate_identity(gen_loop.load_agent(cand))  # the prior's sha and the run head's: attempt 2 changes only the head
        seeds_file = gen_loop.resolve_e_block(a.seeds_file, target, tsv=Path(a.agent_root) / "promotions.tsv", cand_sha=sha, head_sha=head) \
            if "{k}" in a.seeds_file else Path(a.seeds_file)
        import prior_launch  # noqa: PLC0415

        return gen_loop.run_promote_subprocess(a.emu_python, cand, None, seeds_file, Path(a.agent_root), a.procs, a.lib,
                                               Path(a.lab) if a.lab else None, Path(a.shards_root), review_guard=a.review_guard,
                                               prior_server_flags=tuple(prior_launch.ServerOpts.from_args(a).flags()),
                                               sequential=bool(getattr(a, "sequential", False)))  # off unless the train flags say
    return 0


def build_cmd(a: argparse.Namespace) -> int:
    """`build`: generation N's candidate assembled from a combat prior export and a generation's V_run fit and macro, nothing trained
    (`gen_loop.build_candidate`); with --marker-dir it then writes `ready-<base generation>-g<N>`, so the lab's train job judges it
    on the promotion path (`train` sees the `built` candidate and skips training). With `--attempt K` (K > 1) the same generation
    number is tried again beside the earlier attempts: gen<N>/candidate-a<K> and the marker `ready-<base generation>-g<N>-a<K>`
    (their `done-` form once consumed); the first attempt's folder, promotion.json and markers are never touched."""
    _setup()
    import gen_loop  # noqa: PLC0415

    base = a.base or str(Path(a.agent_root) / "current.json")
    attempt = int(getattr(a, "attempt", 1) or 1)
    marker = None
    if a.marker_dir:  # checked before anything is written: a marker that already exists (or was consumed) is not written again
        base_gen = gen_loop.load_agent(base).generation
        name = f"ready-{base_gen}-g{a.candidate_generation}" + (f"-a{attempt}" if attempt > 1 else "")
        taken = [p.name for p in (Path(a.marker_dir) / name, Path(a.marker_dir) / ("done-" + name.removeprefix("ready-"))) if p.exists()]
        if taken:
            print(f"{', '.join(taken)} exists in {Path(a.marker_dir).name}: not built", file=sys.stderr)
            return 2
        marker = Path(a.marker_dir) / name
    cand = gen_loop.build_candidate(a.prior, base, a.agent_root, a.candidate_generation, why=a.why, export=not a.no_export,
                                    vrun_fit=getattr(a, "vrun_fit", None) or None, attempt=attempt)
    meta = gen_loop.read_json(cand / "agent.json")
    out = {"candidate": f"{cand.parent.name}/{cand.name}", "generation": meta["generation"], "built_on": meta["built_on"],
           "policy_sha256": meta[gen_loop.BUILT_KEY]["prior"]["policy_sha256"], "vrun_fit": meta["vrun_fit"]}
    if marker is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(gen_loop.utc_now() + "\n", encoding="utf-8")
        out["marker"] = marker.name
    print(json.dumps(out))
    return 0


def vrun_fit_cmd(a: argparse.Namespace) -> int:
    """`vrun-fit`: V_run fit on the runs one combat prior played (`gen_loop.fit_run_value_matched`), at BelowNormal priority;
    prints the window and the keep verdict. Nothing is promoted or written into the agent folder: `build --vrun-fit OUT` does that.
    The keep verdict includes the ranking check (D47, D49) against `--incumbent` (default: the generation of `--gen-dir`);
    `--ranking-only` repeats just that check on a fit already in OUT."""
    _setup()
    import gen_loop  # noqa: PLC0415

    from sts2_gym import procutil  # noqa: PLC0415

    procutil.lower_priority()  # BelowNormal on Windows: the fit yields to the lab
    incumbent = a.incumbent or a.gen_dir
    if not incumbent:
        raise SystemExit("the ranking check needs the head the fit would replace: --incumbent (a run_value.npz, a generation "
                         "directory or a current.json)")
    seed_dirs = [gen_loop.lab_seed_dir(a.lab)] if a.lab else ()
    if a.ranking_only:
        doc = gen_loop.rerank_vrun_fit(Path(a.out), incumbent, Path(a.shards_root), seed_dirs=seed_dirs)
    else:
        sha = a.prior_sha or gen_loop.load_agent(a.gen_dir).describe()["prior_sha256"]
        if not sha:
            raise SystemExit(f"{a.gen_dir} plays no combat prior")
        gens = [int(x) for x in a.gens.split(",")] if a.gens else None
        doc = gen_loop.fit_run_value_matched(Path(a.shards_root), sha, Path(a.out), gens=gens,
                                             beta=gen_loop.MATCHED_BETA if a.beta is None else a.beta, threads=a.threads,
                                             device=a.device, seed_dirs=seed_dirs, incumbent=incumbent,
                                             exploring_only=bool(getattr(a, "exploring_only", False)))
    head = ((doc["keep"].get("heads") or {}).get("act1_boss_kill") or {}).get("brier_skill_vs_base_rate") or {}
    rk = doc["keep"].get("ranking") or {}
    print(json.dumps({"out": Path(a.out).name, "window": doc["window"]["runs_by_generation"], "blocks": doc["window"].get("blocks"),
                      "exploring_only": doc["window"].get("exploring_only", False), "beta": doc["beta"], "kept": doc["kept"],
                      "brier_skill": head.get("value"), "ci": head.get("ci"), "wall_min": doc["run_value"].get("wall_min"),
                      "ranking": {"status": rk.get("status"), "pass": rk.get("pass"), "why": rk.get("why"),
                                  "checks": [{k: c.get(k) for k in ("name", "candidate", "incumbent", "diff", "pass")}
                                             for c in rk.get("checks") or []]}}))
    return 0 if doc["kept"] else 3


def ranking_check_cmd(a: argparse.Namespace) -> int:
    """`ranking-check`: the V_run keep test's ranking check (D47, D49) on its own: CANDIDATE's options ranked against INCUMBENT's on the
    fixed set of held-out decisions, plus the human leg; writes ranking_set.json and ranking.json into OUT. A candidate or incumbent is
    a run_value.npz, a generation directory, a fit folder or a current.json. Exit 3 when the check does not pass."""
    _setup()
    import gen_loop  # noqa: PLC0415
    from run_train import ranking  # noqa: PLC0415

    from sts2_gym import procutil  # noqa: PLC0415

    procutil.lower_priority()  # BelowNormal on Windows: the check yields to the lab
    fit_json = Path(a.candidate) / gen_loop.VRUN_FIT_JSON  # a fit folder: the runs it was fitted on are left out of the set
    window = gen_loop.fit_window_seeds(gen_loop.read_json(fit_json), a.runs_root) if fit_json.is_file() else ()
    rec = ranking.run_check(gen_loop.incumbent_head(a.candidate), gen_loop.incumbent_head(a.incumbent), Path(a.runs_root), Path(a.out),
                            exclude_seeds=window, human=None if a.no_human else ranking.human_leg)
    print(json.dumps({"pass": rec["pass"], "set": rec["set"]["digest"], "wall_s": rec["wall_s"],
                      "checks": [{k: c.get(k) for k in ("name", "candidate", "incumbent", "diff", "se", "limit", "n", "pass")} for c in rec["checks"]]}))
    return 0 if rec["pass"] else 3


def promote_cmd(a: argparse.Namespace) -> int:
    if a.lib:
        os.environ["STS2_LIB_PATH"] = a.lib
    _setup()
    import gen_loop  # noqa: PLC0415
    import prior_launch  # noqa: PLC0415

    opts = prior_launch.ServerOpts.from_args(a)  # the arms play on the CPU path unless --prior-server says otherwise
    doc = gen_loop.promote(Path(a.cand), Path(a.base) if a.base else None, Path(a.seeds_file), agent_root=Path(a.agent_root),
                           procs=a.procs, config=a.config, ascension=a.ascension, lab=Path(a.lab) if a.lab else None,
                           lab_runs=[Path(x) for x in a.lab_runs] if a.lab_runs else None,
                           review_guard=True if a.review_guard else None, sequential=bool(a.sequential),
                           smoke=a.smoke_seeds, batch=a.batch_seeds,
                           **({"server_opts": opts} if opts.mode != "off" else {}))
    print(json.dumps({k: doc[k] for k in ("decision_text", "floor", "boss_kill", "wins")}, indent=1))
    return 0


def rejudge_cmd(a: argparse.Namespace) -> int:
    _setup()
    import gen_loop  # noqa: PLC0415

    doc = gen_loop.rejudge(Path(a.cand), Path(a.base) if a.base else None, Path(a.seeds_file), agent_root=Path(a.agent_root),
                           config=a.config, note=a.note, lab=Path(a.lab) if a.lab else None,
                           review_guard=True if a.review_guard else None)
    out = {k: doc[k] for k in ("decision_text", "n_seeds", "floor", "boss_kill", "wins", "decision")}
    if "seeds" in doc["rejudged"]:  # a block the funnel stopped early: say which seeds were judged
        out["seeds_judged"] = doc["rejudged"]["seeds"]["text"]
    print(json.dumps(out, indent=1))
    return 0


def select_engine(engine: str) -> None:
    """Set the emulator engine of this process and every process it starts.

    STS2_ENGINE, read by sts2_gym.native when it loads (docs/rebuild/NATIVE_KERNEL.md section 3): "old" clears it, whatever
    the environment said, so a command that did not ask for core2 runs old.
    """
    if engine == "old":
        os.environ.pop("STS2_ENGINE", None)
    else:
        os.environ["STS2_ENGINE"] = engine


def emu_arm_match(emu_dir: str | Path | None, agent: dict, engine: str = "old") -> tuple[bool, str]:
    """Whether a finished emulator arm in `emu_dir` was played by `agent` (an `Agent.describe()` dict) on `engine`: its summary.json
    names the same generation, combat-prior sha and macro policy, and the same engine (none named: old). Returns (matches, why); a
    missing dir, runs file, summary or agent name is a mismatch."""
    if not emu_dir:
        return False, "no emulator dir given"
    d = Path(emu_dir)
    if not (d / "runs.jsonl").is_file() or not (d / "summary.json").is_file():
        return False, "no finished emulator arm in the dir"
    try:
        summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "unreadable summary.json"
    named = summ.get("agent")
    if summ.get("engine", "old") != engine:
        return False, f"played on another engine ({summ.get('engine', 'old')!r}, scored {engine!r})"
    if not isinstance(named, dict):
        return False, "summary.json names no agent"
    named = {"macro": "rules", **named}  # a summary from before the macro switch was played by the rules
    for key in ("generation", "prior_sha256", "macro"):
        if named.get(key) != agent.get(key, "rules" if key == "macro" else None):
            return False, f"names another agent ({key} {named.get(key)!r}, scored {agent.get(key)!r})"
    return True, "names the scored agent"


def emu_arm_seeds(seeds: list[str], runs: int, emu_runs: int | None = None) -> list[str]:
    """The seeds the copy arm plays: the first `runs` of the block (the real arm's), cut to the first `emu_runs` of those. A fixed
    prefix, so the set the two arms share is the same on every night, every resume and every reader; each seed keeps its place in
    the block as its run index (the agent's RNG is [agent seed, index]). `emu_runs` None: every seed the real arm plays; one larger
    than the block is the whole block; under 1 is refused (a copy arm of no seeds pairs nothing)."""
    block = list(seeds)[:runs]
    if emu_runs is None:
        return block
    if emu_runs < 1:
        raise ValueError(f"--emu-runs {emu_runs}: the copy arm plays at least one seed")
    return block[:emu_runs]


def play_emu_arm(a: argparse.Namespace, agent, out: Path) -> dict:
    """Play the scored agent on the emulator over the real block's seeds, config and process count into `out`. With `emu_runs` only
    the first that many seeds of the block (`emu_arm_seeds`); a resumed arm keeps its rows on those seeds."""
    if a.lib:
        os.environ["STS2_LIB_PATH"] = a.lib
    engine = getattr(a, "engine", "old")
    select_engine(engine)  # the arm's spawn workers load the library on this engine
    seeds = emu_arm_seeds(load_seeds(a.seeds_file), a.runs, getattr(a, "emu_runs", None))
    # resume: a stopped arm is rerun into the same folder and keeps its finished seeds (the identity file guards the agent, the
    # config and the engine); run_timeout_s None (not given) is the dispatcher's default, RUN_TIMEOUT_S
    timeout = getattr(a, "run_timeout_s", None)
    summ = run_block(seeds, a.config or a.sims, a.procs, out, ascension=a.ascension, character=a.character, agent=agent, resume=True,
                     resume_key={"engine": engine}, **({"run_timeout_s": timeout} if timeout else {}))
    if engine != "old":  # an old arm's summary stays as every self-play block writes it
        summ = {**summ, "engine": engine}
        (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    return summ


GPU_OK_FILE = "D:/spireline/referee/GPU_OK"  # the lines of the lanes that allow a fit on the GPU beside the real-game arm


def gpu_ok_add(path: str | Path, tag: str) -> None:
    """Put the line `tag` first in the file `path` (created if absent): the other lines stay, an older line equal to `tag` goes.
    Best effort: a file that cannot be written costs the GPU lane its window, never the block."""
    p = Path(path)
    try:
        have = p.read_text(encoding="utf-8").splitlines() if p.is_file() else []
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text("".join(x + "\n" for x in [tag, *(h for h in have if h.strip() and h.strip() != tag)]), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as e:
        print(f"gpu ok: {p.name} not written ({type(e).__name__})", file=sys.stderr, flush=True)


def gpu_ok_remove(path: str | Path, tag: str) -> None:
    """Remove the lines equal to `tag` from the file `path`, the other lines kept; the file is deleted when nothing else is left in it."""
    p = Path(path)
    try:
        if not p.is_file():
            return
        rest = [h for h in p.read_text(encoding="utf-8").splitlines() if h.strip() and h.strip() != tag]
        if not rest:
            p.unlink()
            return
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text("".join(x + "\n" for x in rest), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as e:
        print(f"gpu ok: {p.name} not cleaned ({type(e).__name__})", file=sys.stderr, flush=True)


QUIET_TAG = "lab-score"  # the QUIET.lock line of the daytime real-game score (`score --quiet-lock`); the night's is lab-nightly
QUIET_SKIPPED_RC = 75  # `score --quiet-lock` when the lock stayed taken for the whole wait: nothing was played (EX_TEMPFAIL)


def score_real(a: argparse.Namespace) -> int:
    """`play_score`, under the shared QUIET.lock when `--quiet-lock FILE` is given (the lab's score-real job; the nightly driver holds
    its own line and does not pass it, so the two never wait on each other).

    The line `<tag> <UTC> pid=<n>` (default tag lab-score) is taken before the first process of either arm, rewritten every 15
    minutes while the block plays and removed at the end, however it ends; the fight loop yields to it (rl/gym_loop.py why_blocked),
    so it no longer generates beside the games and the lab's load guard no longer stops them. While another's line is under 2 h old
    the score waits (`--quiet-wait-h`, default 2; rl/quiet_lock.py). When the lock stays taken that long, nothing is played:
    result.json gets {"skipped": ...} and the exit code is QUIET_SKIPPED_RC, so the lab job fails and keeps its LAB/score/pending
    marker instead of consuming it (the nightly driver's skip does the same through its own marker)."""
    path = getattr(a, "quiet_lock", None)
    if not path:
        return play_score(a)
    import quiet_lock  # noqa: PLC0415

    wait_h = getattr(a, "quiet_wait_h", None)
    wait_h = 2.0 if wait_h is None else float(wait_h)
    lock = quiet_lock.QuietLock(Path(path), getattr(a, "quiet_tag", None) or QUIET_TAG, wait_h=wait_h)
    if not lock.acquire():
        why = f"QUIET.lock stayed taken by {lock.blocked_by()} for {wait_h:g} h"
        print(f"score: skipped: {why}", file=sys.stderr, flush=True)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps({"skipped": why}), encoding="utf-8")
        return QUIET_SKIPPED_RC
    lock.start_heartbeat()
    try:
        return play_score(a)
    finally:
        lock.release()


def play_score(a: argparse.Namespace) -> int:
    """The real-game block: envproto `play` (lockstep on) over the seed block, then result.json with the yield keys.

    The block is paired with an emulator arm of the same agent (`--gen-dir`): `--emu-dir` is used as given when its summary.json
    names that agent, otherwise the arm is played first on the same seeds, config and process count into `<out>-emu/`. The
    real summary.json and compare.json record which arm paired and whether it was played.

    With `resume` (a stopped block rerun into the same folder) the folder is left as it is and envproto play gets `--resume` and the
    agent's description for its identity file: it keeps the finished rows and plays the missing seeds, so the summary, result.json and
    compare.json below are computed over every row of the block. The emulator arm resumes in either case (play_emu_arm).

    With `gpu_ok_tag` the line TAG is first in the GPU_OK file (`gpu_ok_file`) exactly while the game's side plays: written after the
    emulator arm has returned (the arm plays through the prior server on the GPU) and removed when the subprocess ends, however it ends."""
    import subprocess  # noqa: PLC0415

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _setup()
    import gen_loop  # noqa: PLC0415

    agent = gen_loop.load_agent(getattr(a, "gen_dir", None) or None)
    only = getattr(a, "only", None)
    engine = getattr(a, "engine", "old")
    if only:  # a probe of chosen seeds: the game's side only, no emulator arm to pair with
        ok, why, emu_dir = True, "only: no emulator arm", None
    else:
        ok, why = emu_arm_match(a.emu_dir, agent.describe(), engine)
        emu_dir = Path(a.emu_dir) if ok else out.with_name(out.name + "-emu")
        if not ok:
            print(f"emulator arm: {why}; playing it into {emu_dir.name}", file=sys.stderr)
            play_emu_arm(a, agent, emu_dir)
    arm = {"dir": None if only else emu_dir.name if not ok else str(a.emu_dir).replace("\\", "/"), "played": not ok, "agent": agent.describe(),
           "reason": "played: " + why if not ok else why, "engine": engine}
    emu_runs = getattr(a, "emu_runs", None)
    if emu_runs is not None and not only:  # the copy arm played a prefix of the block: the figures pair on the seeds both arms hold
        arm["emu_runs"] = min(emu_runs, a.runs)
        arm["partial"] = emu_runs < a.runs
    env = dict(os.environ)
    if a.lib:
        env["STS2_LIB_PATH"] = a.lib
    env.setdefault("STS2_ALLOW_STALE_NATIVE", "1")
    if getattr(a, "gen_dir", None):
        env["SPIRELINE_GEN_DIR"] = str(a.gen_dir)  # search_configs adds the generation's prior to every search
    cmd = [sys.executable, str(ROOT / "referee" / "envproto.py"), "play", "--game", a.game, "--out", str(out), "--procs", str(a.procs),
           "--runs", str(a.runs), "--sims", a.config or str(a.sims), "--seeds-file", a.seeds_file, "--agent-seed", str(AGENT_SEED),
           "--ascension", str(a.ascension), "--stop-after-h", str(a.stop_after_h), "--character", a.character, "--engine", engine]
    if getattr(a, "recycle", 0):
        cmd += ["--recycle", str(a.recycle)]
    if getattr(a, "sample_machine", 0):
        cmd += ["--sample-machine", str(a.sample_machine)]  # the machine sampler of envproto play (referee/machine_sampler.py)
    if only:
        cmd += ["--only", only]
    if getattr(a, "resume", False):
        cmd += ["--resume", "--agent-identity", json.dumps(agent.describe())]
    tag, gpu_file = getattr(a, "gpu_ok_tag", None), getattr(a, "gpu_ok_file", None) or GPU_OK_FILE
    if tag:
        gpu_ok_add(gpu_file, tag)
    try:
        rc = subprocess.run(cmd, env=env, check=False).returncode
    finally:
        if tag:
            gpu_ok_remove(gpu_file, tag)
    rows = real_runs(out / "runs.jsonl") if (out / "runs.jsonl").exists() else []
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8")) if (out / "summary.json").exists() else {}
    res = result_json(rows, float(summ.get("wall_s") or 0.0)) if rows else {"runs": 0}
    res["divergences"] = summ.get("diverged_decisions")
    res["character"] = a.character
    res["engine"] = engine
    res["emu_arm"] = arm
    (out / "result.json").write_text(json.dumps(res), encoding="utf-8")
    if summ:
        agg = run_fields.aggregate(rows)
        (out / "summary.json").write_text(json.dumps({**summ, "emu_arm": arm, **({"run_fields": agg} if agg else {})}, indent=1),
                                          encoding="utf-8")
    if rows and not only:
        emu = [json.loads(x) for x in (emu_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        (out / "compare.json").write_text(json.dumps({**compare(emu, rows, summ), "emu_arm": arm}, indent=1), encoding="utf-8")
    return rc


# ------------------------------------------------------------------ command line


def add_prior_server_flags(p: argparse.ArgumentParser) -> None:
    """The prior server's flags (docs/rl/GPU_INFERENCE.md section 7): all default to the CPU path."""
    p.add_argument("--prior-server", choices=("off", "auto", "require"), default="off",
                   help="serve the combat prior from one batched GPU process: auto falls back to the CPU path, require fails")
    p.add_argument("--gpu-python", help="the interpreter that runs rl/prior_server.py (default: $SPIRELINE_GPU_PYTHON)")
    p.add_argument("--server-procs", type=int, default=None, help="worker count on the server path (default --procs)")
    p.add_argument("--prior-server-args", help="extra flags for rl/prior_server.py, shlex-split (use / in paths)")
    p.add_argument("--latency-p99-ms", type=float, default=20.0, help="client p99 call latency above which the block is marked degraded")


def build_parser() -> argparse.ArgumentParser:
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
    e.add_argument("--character", default="IRONCLAD", help="the run's character (the game's character id)")
    e.add_argument("--block-min", type=float, default=None)
    e.add_argument("--run-timeout-s", type=float, default=RUN_TIMEOUT_S,
                   help="a run still going after this many seconds has its worker killed and is written as lost (floor 0, no win); "
                   "it is not run again, and the block's summary lists it under timed_out")
    e.add_argument("--out", required=True)
    e.add_argument("--lib", help="directory of the native library (STS2_LIB_PATH)")
    e.add_argument("--gen-dir", help="a generation directory (or a current.json) whose combat prior and macro policy play")
    e.add_argument("--agent-root", help="the agent folder: its current.json names the promoted generation when --gen-dir is absent")
    e.add_argument("--record", action="store_true", help="write the run records and combat samples to OUT/shards, tagged with the generation")
    e.add_argument("--explore-tau", type=float, default=None,
                   help="self-play only (needs --record): sample the vrun macro's choices at this softmax temperature in floors "
                   "(DESIGN.md 5.5); evaluation always plays the argmax agent")
    e.add_argument("--explore-eps", type=float, default=0.0, help="with --explore-tau: the uniform share of the macro's sampling")
    e.add_argument("--lab", help="the lab folder: write LAB/train/ready-<gen> once the generation's self-play budget is spent")
    e.add_argument("--shards-root", help="where the lab's job directories (with shards) are, to count the budget")
    e.add_argument("--budget-runs", type=int, default=2000)
    e.add_argument("--review-guard", action="store_true", help=REVIEW_GUARD_HELP)  # the cap on the ready- markers it writes (mark_ready)
    add_prior_server_flags(e)
    t = sub.add_parser("train", help="train generation N's candidate from the last 3 generations' shards. Since D33 (2026-10-25) a "
                       "candidate whose V_run is kept plays the value-head macro (agent.json macro vrun, no environment variable); "
                       "SPIRELINE_MACRO_VRUN=rules writes macro rules instead")
    t.add_argument("--gen", type=int)
    t.add_argument("--policy-target", choices=("visits", "completed_q"), default="visits",
                   help="the combat policy's target: root visits, or the completed-Q softmax (train_combat)")
    t.add_argument("--marker-dir", help="LAB/train: take the lowest ready-<gen> marker instead of --gen")
    t.add_argument("--candidate-generation", type=int, default=None,
                   help="the candidate's generation number (default gen + 1); a number that skips ones writes gen<N>/candidate with a "
                   "lineage note and starts its own promotion attempt count. A ready-<gen>-g<N> marker names it too")
    t.add_argument("--shards-root", required=True)
    t.add_argument("--agent-root", required=True)
    t.add_argument("--device", default=None, help="cuda or cpu (default: cuda when torch sees one)")
    t.add_argument("--epochs", type=int, default=16)
    t.add_argument("--max-fights", type=int, default=None, help="fights for the combat net at the first try (default "
                   "gen_loop.MAX_FIGHTS; x2 and x3 at the retrains)")
    t.add_argument("--threads", type=int, default=6)
    t.add_argument("--no-vrun", action="store_true")
    t.add_argument("--vrun-seed", help="a V_run fit (run_value.npz) the candidate carries unless its own new fit passes the keep test "
                   "(the A/B's fit for the first vrun candidate; <stem>.keep.json beside it is recorded in train.json)")
    t.add_argument("--exclude-prior-paths", help="comma list of prior paths (none, cpu, server, mixed, uniform) whose runs the training set drops")
    add_prior_server_flags(t)  # passed through to --promote's subprocess; the promotion arms are CPU unless these say otherwise
    t.add_argument("--promote", action="store_true", help="then run `promote` against the current champion in the emulator venv")
    t.add_argument("--emu-python", help="the emulator venv's python for --promote")
    t.add_argument("--seeds-file", help="the E-block for --promote; with {k} (and {date}) the block of the candidate's "
                   "generation, created if it does not exist. Put it under the lab folder (LAB/seeds/E{k}-{date}.json), "
                   "not in the checkout: a file written into the tree makes the presort's refset score refuse it. "
                   "Committed blocks (referee/seeds) are reused and checked whichever folder the template names")
    t.add_argument("--procs", type=int, default=16)
    t.add_argument("--lib")
    t.add_argument("--lab")
    t.add_argument("--review-guard", action="store_true", help=REVIEW_GUARD_HELP)
    t.add_argument("--sequential", action="store_true", help=SEQUENTIAL_HELP)
    bd = sub.add_parser("build", help="assemble generation N's candidate from a combat prior export and a generation's V_run fit and "
                        "macro, nothing trained (D44 step A); with --marker-dir the lab's train job then judges it as it is")
    bd.add_argument("--prior", required=True, help="a combat prior export: model.json, combat_policy.pt, combat_net.pt (the fight "
                    "loop's players/v<N>)")
    bd.add_argument("--base", help="the generation whose V_run fit, macro and settings the candidate keeps (a generation directory or "
                    "a current.json; default: --agent-root's current.json)")
    bd.add_argument("--agent-root", required=True)
    bd.add_argument("--candidate-generation", type=int, required=True, help="the candidate's generation number (gen<N>/candidate)")
    bd.add_argument("--why", required=True, help="the decision the candidate is built for (agent.json and train.json record it)")
    bd.add_argument("--marker-dir", help="LAB/train: write ready-<base generation>-g<N> (-a<K> from the second attempt) once the candidate is complete")
    bd.add_argument("--attempt", type=int, default=1, help="the candidate's attempt at this generation number: from 2 it is written as "
                    "gen<N>/candidate-a<K> with the marker ready-<base generation>-g<N>-a<K>, beside the first attempt, which stays as it is")
    bd.add_argument("--no-export", action="store_true", help="no ONNX files (they need torch's exporter and the onnx package)")
    bd.add_argument("--vrun-fit", help="a fighter-matched V_run fit (`vrun-fit`'s OUT) to carry instead of the base's; it must "
                    "have passed the keep test")
    vf = sub.add_parser("vrun-fit", help="fit V_run on the runs one combat prior played (the fighter-matched recipe, reset order "
                        "step C): FightNet, the value head and the D33 keep record into OUT; nothing else is trained")
    vf.add_argument("--shards-root", required=True, help="the lab's job directories (LAB/runs)")
    who = vf.add_mutually_exclusive_group(required=True)
    who.add_argument("--prior-sha", help="the combat prior's policy sha256 whose runs make the window")
    who.add_argument("--gen-dir", help="a generation directory (or a current.json): its prior's runs make the window")
    vf.add_argument("--gens", help="comma list: only these generations' blocks of that prior")
    vf.add_argument("--exploring-only", action="store_true", help="only the blocks an exploring agent played (tau or eps in the manifest's "
                    "agent block; D47's rule for generation 12's second attempt); default: every block of that prior, argmax ones too")
    vf.add_argument("--beta", type=float, default=None, help="the bootstrap blend (default: 0, pure Monte Carlo)")
    vf.add_argument("--out", required=True)
    vf.add_argument("--incumbent", help="the head the fit would replace, for the ranking check (D47, D49): a run_value.npz, a generation "
                    "directory or a current.json (default: --gen-dir)")
    vf.add_argument("--ranking-only", action="store_true", help="repeat only the ranking check on the fit already in OUT")
    vf.add_argument("--lab", help="the lab folder: its generated E-blocks (LAB/seeds) are checked against the window too")
    vf.add_argument("--device", default=None)
    vf.add_argument("--threads", type=int, default=4)
    rk = sub.add_parser("ranking-check", help="the V_run keep test's ranking check (D47, D49): a candidate head's options ranked against "
                        "the incumbent's on the fixed set of held-out decisions, and against the human choice on the held-out fifth")
    rk.add_argument("--candidate", required=True, help="a run_value.npz, a generation directory, a fit folder or a current.json")
    rk.add_argument("--incumbent", required=True, help="the head it would replace (the same forms)")
    rk.add_argument("--runs-root", default="D:/spireline/referee/LAB/runs", help="the lab's job directories (the set is drawn from "
                    "their self-play shards)")
    rk.add_argument("--out", required=True, help="ranking_set.json and ranking.json go here (beside the keep document)")
    rk.add_argument("--no-human", action="store_true", help="skip the human leg (the record then cannot pass)")
    pr = sub.add_parser("promote", help="play candidate and base on a fresh E-block and test the promotion rule")
    pr.add_argument("--cand", required=True)
    pr.add_argument("--base", help="the base generation directory (default: the champion in --agent-root, else generation 0)")
    pr.add_argument("--seeds-file", default=str(ROOT / "referee" / "seeds" / "E1-2026-10-23.json"))
    pr.add_argument("--agent-root", required=True)
    pr.add_argument("--procs", type=int, default=16)
    pr.add_argument("--config", default="a3-v1")
    pr.add_argument("--ascension", type=int, default=10)
    pr.add_argument("--lib")
    pr.add_argument("--lab", help="the lab folder: write LAB/score/pending after a promotion")
    pr.add_argument("--lab-runs", action="append", help="a folder of self-play job directories (the lab's runs folder): "
                    "the E-block must not contain their seeds; repeatable")
    pr.add_argument("--review-guard", action="store_true", help=REVIEW_GUARD_HELP)
    pr.add_argument("--sequential", action="store_true", help=SEQUENTIAL_HELP)
    pr.add_argument("--smoke-seeds", type=int, default=200, help="with --sequential: the seeds of the smoke block, the first look")
    pr.add_argument("--batch-seeds", type=int, default=300, help="with --sequential: the seeds added between the later looks")
    add_prior_server_flags(pr)
    rj = sub.add_parser("rejudge", help="apply the current promotion rule to a candidate's finished arms on its own block")
    rj.add_argument("--cand", required=True)
    rj.add_argument("--base")
    rj.add_argument("--seeds-file", required=True)
    rj.add_argument("--agent-root", required=True)
    rj.add_argument("--config", default="a3-v1")
    rj.add_argument("--note", required=True, help="why the block is judged again (the decision that changed the rule)")
    rj.add_argument("--lab", help="the lab folder: write LAB/score/pending after a promotion")
    rj.add_argument("--review-guard", action="store_true", help=REVIEW_GUARD_HELP)
    sc = sub.add_parser("score")
    sc.add_argument("--game", required=True)
    sc.add_argument("--seeds-file", required=True)
    sc.add_argument("--runs", type=int, default=500)
    sc.add_argument("--procs", type=int, default=12)
    sc.add_argument("--sims", type=int, default=60)
    sc.add_argument("--config", help="a named search config (rl/search_configs.py), instead of --sims")
    sc.add_argument("--ascension", type=int, default=10)
    sc.add_argument("--character", default="IRONCLAD", help="the run's character (the game's character id)")
    sc.add_argument("--stop-after-h", type=float, default=3.5)
    sc.add_argument("--run-timeout-s", type=float, default=None,
                    help="the emulator arm's wall limit per run (default: RUN_TIMEOUT_S, 600 s); past it the worker is killed and "
                    "the seed written as lost (floor 0, no win), not run again")
    sc.add_argument("--quiet-lock", default=None, metavar="FILE",
                    help="hold a line in this shared QUIET.lock file from before the first process until the end, so the fight loop "
                    "yields (the lab's score-real job; the nightly driver holds its own line and passes nothing)")
    sc.add_argument("--quiet-tag", default=QUIET_TAG, help="the line's tag with --quiet-lock (default: %(default)s)")
    sc.add_argument("--quiet-wait-h", type=float, default=2.0,
                    help="with --quiet-lock: wait this long for another's line under 2 h old, then play nothing (exit code 75, "
                    "result.json {skipped})")
    sc.add_argument("--lib")
    sc.add_argument("--emu-runs", type=int, default=None,
                    help="the emulator (copy) arm plays only the first N of the block's seeds (default: all of --runs); compare.json, "
                    "the summary and the readers pair on the seeds both arms hold")
    sc.add_argument("--emu-dir", help="an emulator arm to pair with, used only when its summary.json names the agent scored (--gen-dir); otherwise the arm is played first into OUT-emu")
    sc.add_argument("--only", help="play only these seeds of the file (comma-separated), each keeping its place in the full file as its "
                    "run index (the agent's RNG is [agent seed, index]); the game's side only, no emulator arm")
    sc.add_argument("--gen-dir", help="play this generation's combat prior (SPIRELINE_GEN_DIR for the search configs)")
    sc.add_argument("--out", required=True)
    sc.add_argument("--engine", choices=("old", "core2"), default="old",
                    help="the emulator's engine, for the lockstep and the emulator arm (docs/rebuild/NATIVE_KERNEL.md); default old")
    sc.add_argument("--recycle", type=int, default=0, help="restart each game process after this many runs (0: never)")
    sc.add_argument("--sample-machine", type=float, default=0.0,
                    help="the game's side only: every this many seconds one line of machine readings goes to machine.jsonl in OUT (0: off)")
    sc.add_argument("--gpu-ok-tag", default=None,
                    help="while the game's side plays (never during the emulator arm) keep this line first in the GPU_OK file: "
                    "the GPU lane may run fits beside it; removed when the play ends")
    sc.add_argument("--gpu-ok-file", default=GPU_OK_FILE, help="the file --gpu-ok-tag writes to")
    sc.add_argument("--resume", action="store_true",
                    help="a stopped block rerun into OUT keeps the real game's finished rows (envproto play --resume) and plays the "
                    "seeds it has no row for; the emulator arm resumes either way")
    for name in ("bench-env", "bench-emu"):
        b = sub.add_parser(name)
        b.add_argument("--game", default="")
        b.add_argument("--procs", type=int, default=16)
        b.add_argument("--seconds", type=float, default=60.0)
        b.add_argument("--recycle", type=int, default=0)
        b.add_argument("--lib")
        b.add_argument("--out", required=True)
        if name == "bench-emu":
            b.add_argument("--engine", choices=("old", "core2"), default="old",
                           help="the emulator's engine for the workers (docs/rebuild/NATIVE_KERNEL.md section 10); default old")
    r = sub.add_parser("report")
    r.add_argument("--emu", action="append", required=True, help="DIR:LABEL, e.g. out/emu60:60")
    r.add_argument("--real", required=True)
    r.add_argument("--out", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    # D31: core2 only where it is asked for (score and bench-emu with --engine core2); every other command runs old, whatever
    # the environment says, so the lab's self-play, the promotion arms and the train job stay on the old engine.
    select_engine(getattr(a, "engine", "old") if a.cmd in ("score", "bench-emu") else "old")
    if a.cmd in ("score", "emu") and a.character != "IRONCLAD":
        sys.path.insert(0, str(ROOT / "referee"))
        import lockstep  # noqa: PLC0415

        lockstep.require_character(a.character)  # exit at once, not as a crash row per run
    if a.cmd == "seeds":
        print(len(write_seed_file(a.name, a.n, a.out)), "seeds")
        return 0
    if a.cmd in ("bench-env", "bench-emu"):
        return bench_cmd(a)
    if a.cmd == "score":
        return score_real(a)
    if a.cmd == "train":
        return train_cmd(a)
    if a.cmd == "build":
        return build_cmd(a)
    if a.cmd == "vrun-fit":
        return vrun_fit_cmd(a)
    if a.cmd == "ranking-check":
        return ranking_check_cmd(a)
    if a.cmd == "promote":
        return promote_cmd(a)
    if a.cmd == "rejudge":
        return rejudge_cmd(a)
    if a.cmd == "emu":
        if a.lib:
            os.environ["STS2_LIB_PATH"] = a.lib
        if a.stream:
            seeds = stream_seeds(a.stream)
        else:
            seeds = load_seeds(a.seeds_file)
        if a.n:
            seeds = seeds[: a.n]
        _setup()
        import gen_loop  # noqa: PLC0415

        agent = gen_loop.load_agent(a.gen_dir, a.agent_root) if (a.gen_dir or a.agent_root) else None
        if a.explore_tau is not None or a.explore_eps:
            if not a.record:
                raise SystemExit("--explore-tau/--explore-eps are for recorded self-play (--record) only")
            agent = gen_loop.exploring_agent(agent, a.explore_tau or 0.0, a.explore_eps)
        import prior_launch  # noqa: PLC0415

        opts = prior_launch.ServerOpts.from_args(a)
        stop_file = Path(a.out) / STOP_FILE if a.record else None  # recorded self-play: the lab may ask the block to finish (the filler)
        if opts.mode == "off":
            summ = run_block(seeds, a.config or a.sims, a.procs, Path(a.out), ascension=a.ascension, character=a.character,
                             block_min=a.block_min, agent=agent, record=a.record, run_timeout_s=a.run_timeout_s, stop_file=stop_file)
        else:
            summ = run_block_served(seeds, a.config or a.sims, a.procs, Path(a.out), opts=opts, ascension=a.ascension,
                                    character=a.character, block_min=a.block_min, agent=agent, record=a.record,
                                    run_timeout_s=a.run_timeout_s, stop_file=stop_file)
        if a.record and a.lab and a.shards_root:
            gen_loop.mark_ready(Path(a.lab), agent.generation if agent else 0, Path(a.shards_root), a.budget_runs,
                                tsv=Path(a.agent_root) / "promotions.tsv" if a.agent_root else None,
                                review_guard=True if a.review_guard else None)  # the train job's reading of the cap, not only the variable
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
