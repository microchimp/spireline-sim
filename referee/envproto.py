#!/usr/bin/env python3
"""The real game, headless, as a training environment (prototype; driver/Env.cs).

    python referee/envproto.py smoke  --game <copy> --out <dir>                    one random run, printed
    python referee/envproto.py bench  --game <copy> --out <dir> --procs 8 --seconds 600
    python referee/envproto.py record --game <copy> --out <dir> --seeds S1 S2 S3   the normal driver (seeded chooser)
    python referee/envproto.py replay --game <copy> --out <dir>                    its picks through the channel
    python referee/envproto.py play   --agent play --game <copy> --out <dir> --runs 50 --procs 4 --sims 60

One game process per GameEnv, which plays whole runs back to back (reset() starts the next one, no restart).
Each pending choice with two or more legal actions comes as {"kind", "obs", "actions"}; step(i) answers with the
index into "actions". The channel is the game's own stdin and stdout pipes (no socket, no file). The referee's
session checks hold: known sts2.dll, override.cfg, scratch APPDATA, the real profile untouched, and a netstat
check every few seconds over every game process tree (the session fails on any connection).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import queue
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_referee as rr  # noqa: E402
import serve  # noqa: E402

PREFIX = "\x1eENV ".encode()
BELOW_NORMAL = 0x4000


class GameDied(Exception):
    pass


def below_normal_self() -> None:
    """BelowNormal priority for this (Python) process, through the Windows priority classes."""
    k32 = ctypes.WinDLL("kernel32")
    k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL)


class GameEnv:
    """One headless game. reset(seed) -> first decision message; step(a) -> (message, done)."""

    def __init__(self, game: Path, scratch: Path, *, fixed_fps: int = 60, install: bool = True, stall_s: int = 300,
                 trace: bool = False, boot_timeout: float = 180, command: list[str] | None = None):
        self.game = Path(game).resolve()
        self.digest = rr.check_build(self.game)
        if install:
            rr.install(self.game, HERE / "driver" / "bin" / "Release" / "net9.0")
        self.scratch = Path(scratch).resolve()
        for d in ("appdata", "localappdata"):
            (self.scratch / d).mkdir(parents=True, exist_ok=True)
        rr.write_settings(self.scratch, boot_opts=True)
        jopts = serve.job_options({}, stall_s=stall_s)
        jopts.update(env=True, out=str(self.scratch / "out.json").replace("\\", "/"))
        if trace:
            jopts["env_trace"] = True
        job = self.scratch / "job.json"
        job.write_text(json.dumps(jopts), encoding="utf-8")
        args = command or [*rr.game_command(self.game), *serve.engine_args(fixed_fps)]
        self.proc = rr.popen_no_wer(args, cwd=self.game, env=rr.game_env(self.scratch, job), stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        self.pid = self.proc.pid
        self._q: queue.Queue = queue.Queue()
        self._log = open(self.scratch / "game-stdout.log", "wb")
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        self.t_boot = time.time()
        m = self._recv(boot_timeout)
        if m.get("t") != "ready":
            raise GameDied(f"expected ready, got {m}")
        self.boot_s = time.time() - self.t_boot
        self.last: dict | None = None

    def _reader(self) -> None:
        for raw in iter(self.proc.stdout.readline, b""):
            i = raw.find(PREFIX)
            if i < 0:
                self._log.write(raw)
                continue
            if i > 0:
                self._log.write(raw[:i] + b"\n")
            try:
                self._q.put(json.loads(raw[i + len(PREFIX):]))
            except ValueError as e:
                self._q.put({"t": "garbled", "error": str(e)})
        self._log.flush()
        self._q.put(None)

    def _send(self, o: dict) -> None:
        try:
            self.proc.stdin.write((json.dumps(o) + "\n").encode())
            self.proc.stdin.flush()
        except OSError as e:
            raise GameDied(f"stdin: {e}")

    def _recv(self, timeout: float = 120) -> dict:
        try:
            m = self._q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"no message in {timeout} s")
        if m is None:
            raise GameDied(f"game exited ({self.proc.poll()})")
        return m

    def reset(self, seed: str, spec: dict | None = None, hash: bool = False, record: Path | None = None,
              timeout: float = 120) -> tuple[dict, bool]:
        cmd = {"cmd": "reset", "spec": {"seed": seed, "potions": True, "stop_after_act": 3, "max_decisions": 3000,
                                        **(spec or {})}, "hash": hash}
        if record:
            cmd["record"] = str(record).replace("\\", "/")
        self._send(cmd)
        return self._next(timeout)

    def step(self, a: int, timeout: float = 120) -> tuple[dict, bool]:
        self._send({"a": int(a)})
        return self._next(timeout)

    def _next(self, timeout: float) -> tuple[dict, bool]:
        m = self._recv(timeout)
        if m.get("t") == "bad_action":
            raise ValueError(f"illegal action: {m}")
        self.last = m
        return m, m.get("t") == "end"

    def close(self) -> int | None:
        try:
            self._send({"cmd": "quit"})
            self.proc.wait(30)
        except (GameDied, subprocess.TimeoutExpired, OSError):
            subprocess.run(["taskkill", "/PID", str(self.pid), "/T", "/F"], capture_output=True, check=False)
        self._thread.join(10)
        if not self._thread.is_alive():
            self._log.close()
        return self.proc.poll()


def random_agent(msg: dict, rng: random.Random) -> int:
    return rng.randrange(len(msg["actions"]))


# ---------------------------------------------------------------- bench: N games in a spawn pool


def bench_worker(wid: int, game: str, out: str, seconds: float, q, step_timeout: float, recycle: int = 0) -> None:
    below_normal_self()
    rng = random.Random(1000 + wid)
    t_end = time.time() + seconds
    k = 0
    restarts = 0
    while time.time() < t_end:
        try:
            env = GameEnv(Path(game), Path(out) / "scratch" / f"w{wid:02d}r{restarts}")
        except Exception as e:  # boot failure
            q.put({"ev": "crash", "w": wid, "where": "boot", "error": repr(e)[:300], "t": time.time()})
            restarts += 1
            continue
        q.put({"ev": "ready", "w": wid, "pid": env.pid, "boot_s": round(env.boot_s, 1), "t": time.time()})
        dec = 0
        runs_here = 0
        try:
            while time.time() < t_end:
                seed = f"E{wid:02d}X{k:05d}"
                k += 1
                t0 = time.time()
                m, done = env.reset(seed, timeout=step_timeout)
                n = 0
                slow = (0.0, None)
                while not done:
                    ts, kind = time.time(), m.get("kind")
                    m, done = env.step(random_agent(m, rng), timeout=step_timeout)
                    if time.time() - ts > slow[0]:
                        slow = (round(time.time() - ts, 2), kind)
                    n += 1
                    dec += 1
                    if dec % 200 == 0:
                        q.put({"ev": "tick", "w": wid, "dec": 200, "t": time.time()})
                    if time.time() > t_end + 120:
                        break
                q.put({"ev": "end", "w": wid, "seed": seed, "end": m.get("end"), "floor": m.get("floor"), "asked": n,
                       "ms": m.get("ms"), "mem_mb": m.get("mem_mb"), "dec": dec % 200, "t": time.time(), "s": time.time() - t0,
                       "stale": m.get("stale"), "slowest_step": slow, "fight_errors": m.get("fight_errors"),
                       "probe": m.get("probe")})
                dec -= dec % 200
                runs_here += 1
                if recycle and runs_here >= recycle:
                    q.put({"ev": "recycle", "w": wid, "t": time.time()})
                    break  # a fresh game: the session slows down sharply once private memory nears ~1.6 GB
                if m.get("stale"):
                    break  # the game retires itself (Env.cs); a fresh one follows
        except TimeoutError as e:
            q.put({"ev": "hang", "w": wid, "seed": seed, "error": str(e), "t": time.time()})
        except (GameDied, ValueError) as e:
            q.put({"ev": "crash", "w": wid, "seed": seed, "error": str(e)[:300], "t": time.time()})
        rc = env.close()
        restarts += 1
        q.put({"ev": "closed", "w": wid, "rc": rc, "t": time.time()})


class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


_cpu_prev: tuple | None = None


def sysstat() -> dict:
    """System-wide physical memory load and CPU busy fraction since the last call (leak probe)."""
    global _cpu_prev
    k32 = ctypes.WinDLL("kernel32")
    ms = _MemStatus()
    ms.dwLength = ctypes.sizeof(ms)
    k32.GlobalMemoryStatusEx(ctypes.byref(ms))
    ft = [ctypes.c_ulonglong() for _ in range(3)]
    k32.GetSystemTimes(*[ctypes.byref(x) for x in ft])
    idle, kern, user = (x.value for x in ft)  # kernel time includes idle
    busy = None
    if _cpu_prev:
        di, dk, du = idle - _cpu_prev[0], kern - _cpu_prev[1], user - _cpu_prev[2]
        busy = round(100 * (1 - di / (dk + du)), 1) if dk + du else None
    _cpu_prev = (idle, kern, user)
    return {"mem_load_pct": ms.dwMemoryLoad, "avail_phys_mb": ms.ullAvailPhys >> 20, "avail_commit_mb": ms.ullAvailPageFile >> 20,
            "cpu_busy_pct": busy}


def bench(a) -> int:
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    game = Path(a.game).resolve()
    rr.check_build(game)
    rr.install(game, HERE / "driver" / "bin" / "Release" / "net9.0")
    before = rr.real_profile_state()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=bench_worker, args=(w, str(game), str(out), a.seconds, q, a.step_timeout, a.recycle))
             for w in range(a.procs)]
    for p in procs:
        p.start()
    events = []
    pids: set[int] = set()
    seen: set[str] = set()
    last_net = 0.0
    last_sys = 0.0
    evlog = open(out / "events.jsonl", "w", encoding="utf-8")
    while any(p.is_alive() for p in procs) or not q.empty():
        try:
            e = q.get(timeout=1)
            events.append(e)
            evlog.write(json.dumps(e) + "\n")
            evlog.flush()
            if e["ev"] == "ready":
                pids.add(e["pid"])
        except queue.Empty:
            pass
        if time.time() - last_sys > 10:
            last_sys = time.time()
            e = {"ev": "sys", "t": last_sys, **sysstat()}
            events.append(e)
            evlog.write(json.dumps(e) + "\n")
            evlog.flush()
        if pids and time.time() - last_net > 3:
            last_net = time.time()
            for pid, conns in rr.connections_by_pid(sorted(pids)).items():
                if conns:
                    seen |= conns
                    print(f"NETWORK: pid {pid}: {sorted(conns)}", file=sys.stderr)
    for p in procs:
        p.join()
    wer = set()
    for pid, rep in rr.wer_reports_by_pid(sorted(pids)).items():
        wer |= rep
    s = summarize(events, a.procs)
    s.update(network_connections=sorted(seen), wer_reports=sorted(wer), real_profile_touched=before != rr.real_profile_state())
    (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps(s, indent=1))
    return 0 if not seen and not wer and not s["real_profile_touched"] else 1


def summarize(events: list, procs: int) -> dict:
    ready = [e for e in events if e["ev"] == "ready"]
    if not ready:
        return {"error": "no game became ready", "events": len(events)}
    t_warm = max(e["t"] for e in ready[:procs]) + 30  # every game up, plus 30 s of JIT warm-up
    dec_events = [e for e in events if e["ev"] in ("tick", "end")]
    t_last = max(e["t"] for e in events)
    dec = sum(e["dec"] for e in dec_events if e["t"] >= t_warm)
    ends = [e for e in events if e["ev"] == "end"]
    by_end: dict[str, int] = {}
    for e in ends:
        k = str(e["end"]).split(":")[0]
        by_end[k] = by_end.get(k, 0) + 1
    span = max(t_last - t_warm, 1e-9)
    full_runs = [e for e in ends if e["t"] >= t_warm]
    mem = [e["mem_mb"] for e in ends if e.get("mem_mb")]
    first_mem: dict[int, int] = {}
    last_mem: dict[int, int] = {}
    for e in ends:
        if e.get("mem_mb"):
            first_mem.setdefault(e["w"], e["mem_mb"])
            last_mem[e["w"]] = e["mem_mb"]
    return {
        "procs": procs, "window_s": round(span, 1), "decisions": dec, "decisions_per_s": round(dec / span, 1),
        "decisions_per_s_per_proc": round(dec / span / procs, 2),
        "runs_ended": len(ends), "runs_per_hour": round(len(full_runs) / span * 3600, 1), "ends": by_end,
        "mean_asked_per_run": round(sum(e["asked"] for e in ends) / max(len(ends), 1), 1),
        "mean_floor": round(sum(e["floor"] or 0 for e in ends) / max(len(ends), 1), 2),
        "boot_s_mean": round(sum(e["boot_s"] for e in ready) / len(ready), 1), "boots": len(ready),
        "crashes": [e for e in events if e["ev"] == "crash"], "hangs": [e for e in events if e["ev"] == "hang"],
        "retired_stale": sum(1 for e in ends if e.get("stale")),
        "mem_mb_max": max(mem) if mem else None, "mem_mb_mean": round(sum(mem) / len(mem)) if mem else None,
        "mem_growth_mb_mean": round(sum(last_mem[w] - first_mem[w] for w in last_mem) / max(len(last_mem), 1)),
    }


# ---------------------------------------------------------------- fidelity: record (seeded chooser) and replay


def record(a) -> int:
    """Each seed in a fresh game with no channel: the normal driver's seeded choices, traced with state hashes."""
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    game = Path(a.game).resolve()
    rr.check_build(game)
    rr.install(game, HERE / "driver" / "bin" / "Release" / "net9.0")
    bad = 0
    for seed in a.seeds:
        scratch = out / "scratch" / f"rec-{seed}"
        (scratch / "appdata").mkdir(parents=True, exist_ok=True)
        (scratch / "localappdata").mkdir(parents=True, exist_ok=True)
        jopts = serve.job_options({}, stall_s=300)
        spec = {"kind": "run", "seed": seed, "potions": True, "stop_after_act": 3, "max_decisions": 3000,
                "buff_max_hp": a.buff}
        jopts.update(env_trace=True, runs=[spec], out=str(out / f"rec-{seed}.json").replace("\\", "/"))
        job = scratch / "job.json"
        job.write_text(json.dumps(jopts), encoding="utf-8")
        cmd = [sys.executable, str(HERE / "run_referee.py"), "--game", str(game), "--job", str(job), "--scratch", str(scratch),
               "--no-install", "--below-normal", "--timeout", "900"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        rep = json.loads(r.stdout[r.stdout.index("{"):]) if "{" in r.stdout else {}
        ok = r.returncode == 0 and rep.get("network_connections") == [] and rep.get("real_profile_touched") is False
        bad += not ok
        res = json.loads((out / f"rec-{seed}.json").read_text(encoding="utf-8"))
        run = res[0] if isinstance(res, list) else res.get("results", [res])[0]
        print(seed, "ok" if ok else f"session FAILED {rep}", run.get("end"), "trace", len(run.get("env_trace", [])),
              sorted({t[0] for t in run.get("env_trace", [])}))
    return 1 if bad else 0


def replay(a) -> int:
    """The recorded runs' picks through the channel, back to back in one game, after warm-up runs of random play."""
    out = Path(a.out).resolve()
    recs = sorted(out.glob("rec-*.json"))
    env = GameEnv(Path(a.game), out / "scratch" / "replay")
    rng = random.Random(7)
    seen: set[str] = set()
    for i in range(a.warmup):  # the session's state after other runs must not change a run
        m, done = env.reset(f"WARM{i}")
        while not done:
            m, done = env.step(random_agent(m, rng))
    seen |= rr.connections(env.pid)
    results = []
    for rec in recs:
        res = json.loads(rec.read_text(encoding="utf-8"))
        run = res["results"][0] if "results" in res else res
        trace = run["env_trace"]
        m, done = env.reset(run["seed"] if "seed" in run else rec.stem[4:], hash=True, spec={"buff_max_hp": a.buff},
                            record=out / f"replay-{rec.stem[4:]}.json")
        same = 0
        first = None
        for j, (kind, n, idx, h) in enumerate(trace):
            if done:
                first = first or f"#{j}: the replay ended early ({m.get('end')})"
                break
            if m["kind"] != kind or len(m["actions"]) != n or m.get("h") != h:
                first = first or f"#{j}: {kind}/{n}/{h} vs {m['kind']}/{len(m['actions'])}/{m.get('h')}"
            else:
                same += 1
            m, done = env.step(idx)
        if not done:
            first = first or "the replay asked more decisions than recorded"
        rep_run = json.loads((out / f"replay-{rec.stem[4:]}.json").read_text(encoding="utf-8")) if done else {}
        dumps_equal = done and strip_timing(rep_run) == strip_timing(run)
        results.append({"seed": rec.stem[4:], "decisions": len(trace), "hash_equal": same, "first_diff": first,
                        "end": run.get("end"), "replay_end": m.get("end") if done else None, "dump_equal": dumps_equal})
        print(json.dumps(results[-1]))
    seen |= rr.connections(env.pid)
    env.close()
    ok = all(r["first_diff"] is None and r["dump_equal"] for r in results) and not seen
    (out / "replay.json").write_text(json.dumps({"results": results, "network_connections": sorted(seen)}, indent=1),
                                     encoding="utf-8")
    print("replay", "OK" if ok else "DIFFERS", "network", sorted(seen))
    return 0 if ok else 1


def strip_timing(r: dict) -> dict:
    """The run's record without what differs by design: timing, costs, the trace itself."""
    return serve.drop_keys(r, {"timing_ms", "cost", "env_trace", "boot_ms", "settle_audit"})


def smoke(a) -> int:
    out = Path(a.out).resolve()
    env = GameEnv(Path(a.game), out / "scratch" / "smoke")
    print("boot", round(env.boot_s, 1), "s, pid", env.pid)
    rng = random.Random(1)
    kinds: dict[str, int] = {}
    for r in range(a.runs):
        t0 = time.time()
        m, done = env.reset(f"SMOKE{r}")
        n = 0
        while not done:
            kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
            if n == 0 and r == 0:
                print("first decision:", json.dumps(m)[:600])
            m, done = env.step(random_agent(m, rng))
            n += 1
        print(f"run {r}: {m} in {time.time() - t0:.1f} s, {n / (time.time() - t0):.1f} decisions/s")
    print("kinds", kinds, "network", rr.connections(env.pid))
    print("exit", env.close())
    return 0


def play_worker(wid: int, game: str, out: str, seeds: list, sims: int, q, step_timeout: float, ascension: int,
                agent_seed: int = 0, run_ids: list | None = None, stop_at: float = 0.0) -> None:
    """One game process plays its seeds under the PLAY agent in lockstep (referee/lockstep.py)."""
    below_normal_self()
    import lockstep  # noqa: PLC0415
    env = None
    restarts = 0
    for n_seed, seed in enumerate(seeds):
        if stop_at and time.time() > stop_at:  # the block's time limit: no new run; a worker's seeds go in order, so the
            break  # runs done are a prefix of each worker's share and stay paired with the emulator's
        run_ix = run_ids[n_seed] if run_ids else n_seed
        if env is None:
            try:
                env = GameEnv(Path(game), Path(out) / "scratch" / f"p{wid:02d}r{restarts}", install=False)
            except Exception as e:  # boot failure
                q.put({"ev": "crash", "w": wid, "seed": seed, "where": "boot", "error": repr(e)[:300]})
                restarts += 1
                continue
        t0 = time.time()
        try:
            m, ls = lockstep.play_run(_Timed(env, step_timeout), seed, ascension=ascension, sims=sims,
                                 agent_seed=[agent_seed, run_ix])
            ev = {"ev": "end", "w": wid, "seed": seed, "end": m.get("end"), "floor": m.get("floor"), "asked": m.get("asked"),
                  "s": round(time.time() - t0, 1), "stats": ls.stats, "lost": ls.lost, "divergences": ls.log.records,
                  "agent_seed": ls.agent_seed, "assumed": len(ls.assumed)}
            if m.get("stale"):
                q.put(ev)
                q.put({"ev": "net", "w": wid, "conns": sorted(rr.connections(env.pid))})
                env.close()
                env, restarts = None, restarts + 1
                continue
        except (TimeoutError, GameDied, ValueError) as e:
            ev = {"ev": "crash", "w": wid, "seed": seed, "error": f"{type(e).__name__}: {e}"[:300], "s": round(time.time() - t0, 1), "agent_seed": [agent_seed, run_ix],
                  "divergences": getattr(getattr(e, "lockstep", None), "log", None) and e.lockstep.log.records or []}
            env.close()
            env, restarts = None, restarts + 1
        except Exception as e:  # noqa: BLE001 -- an agent/emulator bug is a crash of that run, reported
            ev = {"ev": "crash", "w": wid, "seed": seed, "where": "agent", "error": f"{type(e).__name__}: {e}"[:300],
                  "s": round(time.time() - t0, 1), "agent_seed": [agent_seed, run_ix],
                  "divergences": getattr(getattr(e, "lockstep", None), "log", None) and e.lockstep.log.records or []}
            env.close()  # the game is mid-run; a fresh one for the next seed
            env, restarts = None, restarts + 1
        q.put(ev)
    if env is not None:
        q.put({"ev": "net", "w": wid, "conns": sorted(rr.connections(env.pid))})
        env.close()
    q.put({"ev": "done", "w": wid})


class _Timed:
    """A GameEnv whose reset and step carry the play mode's step timeout."""

    def __init__(self, env, timeout):
        self.env, self.timeout = env, timeout

    def reset(self, *a, **k):
        return self.env.reset(*a, timeout=self.timeout, **k)

    def step(self, a):
        return self.env.step(a, timeout=self.timeout)


def play(a) -> int:
    """`--agent play`: N runs on --procs game processes, the PLAY agent in lockstep with the emulator."""
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    seeds = [f"A1S{i:04d}" for i in range(a.runs)]
    if a.seeds_file:  # a committed seed block (rl/gen0.py): the first --runs of it, in order
        doc = json.loads(Path(a.seeds_file).read_text(encoding="utf-8"))
        seeds = (doc["seeds"] if isinstance(doc, dict) else doc)[: a.runs]
    agent_seed = a.agent_seed if a.agent_seed is not None else int.from_bytes(os.urandom(6), "big")  # not a game seed
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    stop_at = time.time() + a.stop_after_h * 3600 if a.stop_after_h else 0.0
    live = open(out / "runs.jsonl", "w", encoding="utf-8")  # one line per finished run as it ends; rewritten whole at the end
    procs = [ctx.Process(target=play_worker, args=(w, a.game, str(out), seeds[w::a.procs], a.sims, q, a.step_timeout, a.ascension,
                                       agent_seed, list(range(a.runs))[w::a.procs], stop_at))
             for w in range(a.procs)]
    t0 = time.time()
    for p in procs:
        p.start()
    events, done, nets = [], 0, set()
    while done < len(procs):
        try:
            e = q.get(timeout=5)
        except queue.Empty:
            if not any(p.is_alive() for p in procs):
                break
            continue
        if e["ev"] == "done":
            done += 1
        elif e["ev"] == "net":
            nets |= set(e["conns"])
        else:
            events.append(e)
            live.write(json.dumps(e) + chr(10))
            live.flush()
            print(json.dumps({k: v for k, v in e.items() if k != "divergences"})[:300], flush=True)
    for p in procs:
        p.join(30)
    wall = time.time() - t0
    live.close()
    (out / "runs.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    ends = [e for e in events if e["ev"] == "end"]
    dec = sum(e["stats"]["decisions"] for e in ends)
    rank = sum(e["stats"]["ranked"] for e in ends)
    divs = [dict(d, seed=e["seed"]) for e in events if e["ev"] in ("end", "crash") for d in e.get("divergences", [])]
    first = {}
    for d in divs:
        first.setdefault(d["type"], []).append((d["decision"], d["floor"], d.get("reason") or d["diff"][:2]))
    summary = {"agent_seed": agent_seed, "sims": a.sims, "runs": len(ends), "crashes": sum(e["ev"] == "crash" for e in events), "wins": sum(1 for e in ends if e["end"] in ("win", "victory")),
               "ends": sorted({str(e["end"]) for e in ends}),
               "mean_floor": round(sum(e["floor"] or 0 for e in ends) / max(1, len(ends)), 2), "decisions": dec,
               "decisions_per_s": round(dec / wall, 2), "wall_s": round(wall, 1), "ranked_share": round(rank / max(1, dec), 3),
               "diverged_decisions": sum(e["stats"]["diverged"] for e in ends),
               "divergence_rate": round(sum(e["stats"]["diverged"] for e in ends) / max(1, dec), 4),
               "reanchors": sum(e["stats"]["reanchors"] for e in ends), "reanchor_failed": sum(e["stats"]["reanchor_failed"] for e in ends),
               "first_loss_by_reason": _count(e["stats"]["first_loss"]["reason"].split(":")[0] + ":" + (e["stats"]["first_loss"]["reason"].split(":")[1:2] or [""])[0] for e in ends if e["stats"]["first_loss"]),
               "first_divergences_by_type": {k: v[:5] for k, v in first.items()},
               "gaps": sum(e["stats"].get("gaps", 0) for e in ends), "resumed": sum(e["stats"].get("resumed", 0) for e in ends),
               "losses_by_reason": _merge(e["stats"].get("losses", {}) for e in ends),
               "reanchor_fail_reasons": _merge(e["stats"].get("reanchor_fail_reasons", {}) for e in ends),
               "reanchor_by_plan": _merge(e["stats"].get("reanchor_by_plan", {}) for e in ends),
               "divergences_by_type": _count(f"{d['type']}:{'gap' if d.get('gap') else 'state'}:{(d.get('reason') or ','.join(x[0] for x in d['diff'][:3]))[:40]}" for d in divs),
               "run_s_total": round(sum(e["s"] for e in ends), 1),
               "decisions_per_s_per_proc": round(dec / max(1e-9, sum(e["s"] for e in ends)), 2),
               "ranked_decisions_per_s_per_proc": round(rank / max(1e-9, sum(e["s"] for e in ends)), 3),
               "network_connections": sorted(nets), "errors": [e for e in events if e["ev"] == "crash"][:10]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "divergences.jsonl").write_text("\n".join(json.dumps(d) for d in divs), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 1 if nets else 0


def _merge(dicts) -> dict:
    o: dict = {}
    for d in dicts:
        for k, v in d.items():
            o[k] = o.get(k, 0) + v
    return o


def _count(it) -> dict:
    o: dict = {}
    for x in it:
        o[x] = o.get(x, 0) + 1
    return o


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("mode", choices=["smoke", "bench", "record", "replay", "play"])
    ap.add_argument("--game", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--seeds", nargs="*", default=["FID1", "FID2", "FID3"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--buff", type=int, default=0, help="record/replay: buff_max_hp, so random play reaches later floors")
    ap.add_argument("--agent", choices=["play"], default="play", help="play: lockstep PLAY agent (mode play)")
    ap.add_argument("--stop-after-h", type=float, default=0.0, help="play: start no new run after this many hours (0: no limit)")
    ap.add_argument("--seeds-file", default=None, help="play: JSON seed block (default: A1S0000...)")
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--agent-seed", type=int, default=None, help="play: the agent's own seed (default: random, recorded); never derived from a game seed")
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--step-timeout", type=float, default=180)
    ap.add_argument("--recycle", type=int, default=0, help="bench: restart each game after this many runs (0: never)")
    a = ap.parse_args()
    return {"smoke": smoke, "bench": bench, "record": record, "replay": replay, "play": play}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
