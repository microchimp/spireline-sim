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
import contextlib
import ctypes
import faulthandler
import json
import multiprocessing as mp
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import perfcounters  # noqa: E402
import run_referee as rr  # noqa: E402
import serve  # noqa: E402

PREFIX = "\x1eENV ".encode()
BELOW_NORMAL = 0x4000


class GameDied(Exception):
    pass


BOOT_TIMEOUT_S = 180  # GameEnv's wait for the game's `ready` message, one try
BOOT_SAMPLE_S = 30.0  # a start still waiting after this long takes a reading of its process, and again each time
BOOT_BUDGET_S = 1200.0  # --boot-budget-s default: a start that fails is retried on a fresh process for this long in all, before its seed gets a row
BOOT_PAUSE_S = 5.0  # between a failed start and the next try
BOOT_TRIES_MAX = 12  # no more tries than this for one seed, whatever the budget
BOOT_SLOW_S = 30.0  # a first try that worked but took longer than this is reported too


class _IoCounters(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


def process_probe(pid: int) -> dict:
    """What process `pid` has used so far: `cpu_s` (kernel plus user seconds), `read_mb`, `read_ops`, `write_mb` (what it has read and written,
    all devices). {} off Windows or when the process cannot be opened. It tells a start that is working (CPU and reads rising) from one
    that is waiting (neither), which is what a stalled start needs to be read by."""
    if sys.platform != "win32":
        return {}
    try:
        from ctypes import wintypes  # noqa: PLC0415

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.GetProcessTimes.argtypes = [wintypes.HANDLE, *[ctypes.POINTER(wintypes.FILETIME)] * 4]
        k32.GetProcessIoCounters.argtypes = [wintypes.HANDLE, ctypes.POINTER(_IoCounters)]
        h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return {}
        out: dict = {}
        try:
            ft = [wintypes.FILETIME() for _ in range(4)]
            if k32.GetProcessTimes(h, *[ctypes.byref(x) for x in ft]):
                out["cpu_s"] = round(sum((x.dwHighDateTime << 32) | x.dwLowDateTime for x in ft[2:]) / 1e7, 2)
            io = _IoCounters()
            if k32.GetProcessIoCounters(h, ctypes.byref(io)):
                out.update(read_mb=round(io.read_bytes / 1e6, 1), read_ops=io.read_ops, write_mb=round(io.write_bytes / 1e6, 1))
        finally:
            k32.CloseHandle(h)
        return out
    except Exception:  # noqa: BLE001 -- a reading that fails is no reading, never the start's failure
        return {}


def boot_game(make, budget_s: float, *, pause_s: float = BOOT_PAUSE_S, max_tries: int = BOOT_TRIES_MAX,
              clock=None, sleep=None, say=None):
    """Start a game process; a start that fails is retried on a fresh process, not charged to the run that wanted it.

    `make(k)` starts try `k` (0-based) and returns the GameEnv, or raises (GameEnv has killed the process of a start that failed by then).
    After a failed try the next begins `pause_s` later while fewer than `budget_s` seconds have passed since the first began and fewer
    than `max_tries` have run; a `budget_s` of 0 or less is the one try. `say(line)` gets a dict for every try that failed
    (`try`, `ok` False, `final` whether it was the last, `s` its seconds, `error`, and the exception's `boot_probe`: the process's cpu
    seconds and bytes read when it was given up, a reading every BOOT_SAMPLE_S before that, the bytes its stdout log and its scratch
    profile held) and for a try that worked after a failure or took over BOOT_SLOW_S. Returns (env or None, tries, the last error or
    None). Only Exception is caught: a SystemExit (an unknown build) ends the worker as it always did. `clock` and `sleep` default to
    time.time and time.sleep (looked up at the call)."""
    clock = clock or time.time
    sleep = sleep or time.sleep
    t0 = clock()
    k = 0
    while True:
        t_try = clock()
        try:
            env = make(k)
        except Exception as e:  # noqa: BLE001 -- a start that failed in any way
            again = budget_s > 0 and clock() - t0 < budget_s and k + 1 < max_tries
            if say is not None:
                say({"try": k + 1, "ok": False, "final": not again, "s": round(clock() - t_try, 1), "error": repr(e)[:200],
                     **getattr(e, "boot_probe", {})})
            if not again:
                return None, k + 1, e
            sleep(pause_s)
            k += 1
            continue
        s = clock() - t_try
        if say is not None and (k > 0 or s > BOOT_SLOW_S):
            say({"try": k + 1, "ok": True, "final": True, "s": round(s, 1), "boot_s": round(getattr(env, "boot_s", s), 1),
                 "waited_s": round(clock() - t0, 1)})
        return env, k + 1, None


def below_normal_self() -> None:
    """BelowNormal priority for this (Python) process, through the Windows priority classes."""
    k32 = ctypes.WinDLL("kernel32")
    k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL)


def reset_spec(seed: str, character: str = "IRONCLAD", spec: dict | None = None, extra_actions: bool = False,
               full_select: bool = False) -> dict:
    """The run spec a reset sends: the seed, the character (the game's character id; the driver's default is IRONCLAD) and
    `extra_actions` (card-reward alternatives, "none" on a selection that allows 0, potion actions outside fights; the driver's
    default is off, and it changes the sequence of asked decisions: play and bench turn it on, record/replay/smoke leave it off)
    and `full_select` (a fight card screen offers its real min..max: picks one at a time and a "done" action; off by default so
    old records replay, on in play; the key is present only when on)."""
    out = {"seed": seed, "character": character, "extra_actions": bool(extra_actions), "potions": True, "stop_after_act": 3,
           "max_decisions": 3000, **(spec or {})}
    if full_select:
        out["full_select"] = True
    return out


def write_record_meta(out: Path, seed: str, character: str, extra_actions: bool, full_select: bool = False) -> None:
    """What a record was made with, for replay to use (a sidecar: replay globs rec-*.json)."""
    (Path(out) / f"meta-{seed}.json").write_text(json.dumps({"character": character, "extra_actions": bool(extra_actions),
                                                                      "full_select": bool(full_select)}), encoding="utf-8")


def read_record_meta(out: Path, seed: str, character: str = "IRONCLAD") -> dict:
    """The stored character, extra_actions and full_select of a record; a record without a sidecar (or a key) was made with the flag off."""
    p = Path(out) / f"meta-{seed}.json"
    m = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {"character": m.get("character", character), "extra_actions": bool(m.get("extra_actions", False)),
            "full_select": bool(m.get("full_select", False))}


class GameEnv:
    """One headless game. reset(seed) -> first decision message; step(a) -> (message, done)."""

    def __init__(self, game: Path, scratch: Path, *, fixed_fps: int = 60, install: bool = True, stall_s: int = 300,
                 trace: bool = False, boot_timeout: float = BOOT_TIMEOUT_S, command: list[str] | None = None):
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
        self.last: dict | None = None
        self.proc = None
        self.boot_trace: list[dict] = []
        try:
            self.proc = rr.popen_no_wer(args, cwd=self.game, env=rr.game_env(self.scratch, job), stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            self.pid = self.proc.pid
            self._q: queue.Queue = queue.Queue()
            self._log = open(self.scratch / "game-stdout.log", "wb")
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
            self.t_boot = time.time()
            m = self._wait_ready(boot_timeout)
            if m.get("t") != "ready":
                raise GameDied(f"expected ready, got {m}")
            self.boot_s = time.time() - self.t_boot
        except BaseException as e:
            self._boot_failed(e)  # a start that failed leaves no game process behind
            raise

    def _wait_ready(self, timeout: float) -> dict:
        """The game's first message, waited for in slices of BOOT_SAMPLE_S; every slice that passes without one adds a reading of the
        process to `boot_trace`. The error after `timeout` is the one `_recv` gives for the whole wait."""
        t_end = self.t_boot + timeout
        while True:
            try:
                return self._recv(max(0.0, min(t_end - time.time(), BOOT_SAMPLE_S)))
            except TimeoutError:
                if time.time() >= t_end - 0.05:
                    raise TimeoutError(f"no message in {timeout} s") from None
                self.boot_trace.append({"s": round(time.time() - self.t_boot, 1), **process_probe(self.pid)})

    def _boot_failed(self, e: BaseException) -> None:
        """The readings of a start that failed go on the exception as `boot_probe` (what the process had used when it was given up, the
        trace of readings, the bytes its stdout log and its scratch profile hold, its exit code if it had one), then its process tree is
        killed, its reader joined and its log closed. A process that never started has nothing to kill."""
        probe: dict = {"trace": list(getattr(self, "boot_trace", []))}
        proc = getattr(self, "proc", None)
        try:
            if proc is not None:
                probe.update(process_probe(proc.pid))
                code = proc.poll()
                if code is None and isinstance(e, GameDied):  # the pipe closed before the exit was reaped
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        code = proc.wait(2)
                probe["exit"] = code
            with contextlib.suppress(OSError):
                probe["log_b"] = (self.scratch / "game-stdout.log").stat().st_size
            with contextlib.suppress(OSError):  # the game's own files in its profile: more than the settings file means its engine got going
                probe["scratch_files"] = sum(len(f) for _r, _d, f in os.walk(self.scratch / "appdata")) + sum(
                    len(f) for _r, _d, f in os.walk(self.scratch / "localappdata"))
            if proc is not None and proc.poll() is None:
                self._kill_tree()
            thread = getattr(self, "_thread", None)
            if thread is not None:
                thread.join(10)
            log = getattr(self, "_log", None)
            if log is not None and (thread is None or not thread.is_alive()):
                log.close()
            if proc is not None and (thread is None or not thread.is_alive()):  # the process is dead and its reader done: release the pipes
                for pipe in (proc.stdin, proc.stdout):
                    with contextlib.suppress(OSError, ValueError):
                        pipe.close()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(5)
        except Exception:  # noqa: BLE001 -- the report of a failed start never replaces the failure
            pass
        with contextlib.suppress(AttributeError, TypeError):
            e.boot_probe = probe

    def _kill_tree(self) -> None:
        try:
            subprocess.run(["taskkill", "/PID", str(self.pid), "/T", "/F"], capture_output=True, check=False, timeout=60)
        except (subprocess.TimeoutExpired, OSError):
            with contextlib.suppress(OSError):
                self.proc.kill()

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
              timeout: float = 120, character: str = "IRONCLAD", extra_actions: bool = False,
              full_select: bool = False) -> tuple[dict, bool]:
        cmd = {"cmd": "reset", "spec": reset_spec(seed, character, spec, extra_actions, full_select), "hash": hash}
        if record:
            cmd["record"] = str(record).replace("\\", "/")
        self._send(cmd)
        return self._next(timeout)

    def step(self, a: int, timeout: float = 120) -> tuple[dict, bool]:
        self._send({"a": int(a)})
        return self._next(timeout)

    anchor_cmd = True  # False once the driver has shown it does not know the anchor_state command (an old driver)

    def anchor_state(self, timeout: float = 120) -> dict | None:
        """The open decision's hidden state (driver command anchor_state: stream counters, the draw pile in order, the deck's and
        relics' holdings), for the lockstep's emulator side only (lockstep.Lockstep's mid-fight re-anchor; nothing else may read
        it). The decision stays open: the next `step` answers it. None from a driver that does not know the command, or from a run
        whose reset did not ask for the hidden state: the driver answers bad_action and sends the decision again, both are read
        and dropped (the decision is the one the caller holds), and this process is not asked again."""
        if not self.anchor_cmd:
            return None
        self._send({"cmd": "anchor_state"})
        m = self._recv(timeout)
        if m.get("t") == "anchor_state":
            state = m.get("state")
            return state if isinstance(state, dict) else None
        if m.get("t") == "bad_action":
            again = self._recv(timeout)
            if again.get("t") != "decide":
                raise GameDied(f"anchor_state: after bad_action, {again.get('t')} instead of the decision")
            self.anchor_cmd = False
            return None
        raise GameDied(f"anchor_state: unexpected {m.get('t')}")

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
            self._kill_tree()
        self._thread.join(10)
        if not self._thread.is_alive():
            self._log.close()
        return self.proc.poll()


def random_agent(msg: dict, rng: random.Random) -> int:
    return rng.randrange(len(msg["actions"]))


def extras_of(m: dict) -> list[tuple[int, str]]:
    """The actions of a decision that the driver offers beyond a screen's own options (lockstep.extra_kind, and a card
    reward's skip): (index, "<kind>@<decision kind>"), e.g. (3, "potion_use@map")."""
    import lockstep  # noqa: PLC0415

    out = []
    for i, a in enumerate(m.get("actions", [])):
        k = lockstep.extra_kind(a) or ("card_skip" if a == "skip" and m.get("kind") == "card_reward" else None)
        if k:
            out.append((i, f"{k}@{m['kind']}"))
    return out


def held_potions(m: dict) -> int:
    return sum(1 for p in (m.get("obs") or {}).get("potions", []) if p)


# ---------------------------------------------------------------- bench: N games in a spawn pool


def bench_worker(wid: int, game: str, out: str, seconds: float, q, step_timeout: float, recycle: int = 0,
                 character: str = "IRONCLAD", max_runs: int = 0) -> None:
    below_normal_self()
    rng = random.Random(1000 + wid)
    t_end = time.time() + seconds
    k = 0
    restarts = 0
    while time.time() < t_end and not (max_runs and k >= max_runs):
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
            while time.time() < t_end and not (max_runs and k >= max_runs):
                seed = f"E{wid:02d}X{k:05d}"
                k += 1
                t0 = time.time()
                m, done = env.reset(seed, timeout=step_timeout, character=character, extra_actions=True)
                n = 0
                slow = (0.0, None)
                offered: dict = {}
                taken: dict = {}
                effect: dict = {}
                while not done:
                    ts, kind = time.time(), m.get("kind")
                    pick = random_agent(m, rng)
                    ex = extras_of(m)
                    for i, key in ex:
                        offered[key] = offered.get(key, 0) + 1
                        if i == pick:
                            taken[key] = taken.get(key, 0) + 1
                    before = held_potions(m)
                    took = next((key for i, key in ex if i == pick), None)
                    m, done = env.step(pick, timeout=step_timeout)
                    if took and took.startswith(("potion_", "card_alt")) and not done and held_potions(m) != before:
                        effect[took] = effect.get(took, 0) + 1
                    if time.time() - ts > slow[0]:
                        slow = (round(time.time() - ts, 2), kind)
                    n += 1
                    dec += 1
                    if dec % 200 == 0:
                        q.put({"ev": "tick", "w": wid, "dec": 200, "t": time.time()})
                    if time.time() > t_end + 120:
                        break
                q.put({"ev": "end", "w": wid, "seed": seed, "character": character, "end": m.get("end"), "floor": m.get("floor"), "asked": n,
                       "ms": m.get("ms"), "mem_mb": m.get("mem_mb"), "dec": dec % 200, "t": time.time(), "s": time.time() - t0,
                       "stale": m.get("stale"), "slowest_step": slow, "fight_errors": m.get("fight_errors"),
                       "probe": m.get("probe"), "alt_checked": m.get("alt_checked"), "alt_mismatch": m.get("alt_mismatch"), "extras": {"offered": offered, "taken": taken, "effect": effect}})
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


def sysstat(perf: perfcounters.PerfSampler | None = None, acct: perfcounters.ParentAcct | None = None,
            qsize: int | None = None) -> dict:
    """System-wide physical memory load and CPU busy fraction since the last call (leak probe), the typeperf
    counters of the window (perfcounters.py) and the bench parent's own accounting for it."""
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
    out = {"mem_load_pct": ms.dwMemoryLoad, "avail_phys_mb": ms.ullAvailPhys >> 20, "avail_commit_mb": ms.ullAvailPageFile >> 20,
           "cpu_busy_pct": busy}
    if perf is not None:
        out.update(perf.drain())
    if acct is not None:
        out.update(acct.snap(qsize))
    return out


def bench(a) -> int:
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    game = Path(a.game).resolve()
    rr.check_build(game)
    rr.install(game, HERE / "driver" / "bin" / "Release" / "net9.0")
    before = rr.real_profile_state()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=bench_worker, args=(w, str(game), str(out), a.seconds, q, a.step_timeout, a.recycle, a.character,
                                                  a.max_runs // a.procs + (1 if w < a.max_runs % a.procs else 0) if a.max_runs else 0))
             for w in range(a.procs)]
    for p in procs:
        p.start()
    events = []
    pids: set[int] = set()
    seen: set[str] = set()
    last_net = 0.0
    last_sys = 0.0
    evlog = open(out / "events.jsonl", "w", encoding="utf-8")
    perf = perfcounters.PerfSampler(pids)
    perf.start()
    acct = perfcounters.ParentAcct()
    while any(p.is_alive() for p in procs) or not q.empty():
        t_a = time.perf_counter()
        try:
            e = q.get(timeout=1)
        except queue.Empty:
            e = None
        t_b = time.perf_counter()
        acct.wait += t_b - t_a
        if e is not None:
            events.append(e)
            evlog.write(json.dumps(e) + "\n")
            evlog.flush()
            acct.events += 1
            if e["ev"] == "ready":
                pids.add(e["pid"])
        if time.time() - last_sys > 10:
            last_sys = time.time()
            perf.poll()
            try:
                qs = q.qsize()
            except NotImplementedError:
                qs = None
            e = {"ev": "sys", "t": last_sys, **sysstat(perf, acct, qs)}
            events.append(e)
            evlog.write(json.dumps(e) + "\n")
            evlog.flush()
        t_c = time.perf_counter()
        acct.handle += t_c - t_b
        if pids and time.time() - last_net > 3:
            last_net = time.time()
            for pid, conns in rr.connections_by_pid(sorted(pids)).items():
                if conns:
                    seen |= conns
                    print(f"NETWORK: pid {pid}: {sorted(conns)}", file=sys.stderr)
            acct.net += time.perf_counter() - t_c
    perf.stop()
    for p in procs:
        p.join()
    wer = set()
    for pid, rep in rr.wer_reports_by_pid(sorted(pids)).items():
        wer |= rep
    s = summarize(events, a.procs)
    s.update(character=a.character, network_connections=sorted(seen), wer_reports=sorted(wer), real_profile_touched=before != rr.real_profile_state())
    (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k != "per_minute"}, indent=1))
    return 0 if not seen and not wer and not s["real_profile_touched"] else 1


def parent_summary(sysev: list) -> dict:
    """The bench parent's accounting over the whole window: CPU seconds, time blocked on the result queue, time
    handling results and in the netstat check, and the largest backlog seen."""
    sums = {k: round(sum(e.get(f"parent_{k}_s") or 0.0 for e in sysev), 2) for k in ("cpu", "wait", "handle", "net", "window")}
    win = sums.pop("window")
    qs = [e["parent_qsize"] for e in sysev if e.get("parent_qsize") is not None]
    return {"cpu_s": sums["cpu"], "wait_s": sums["wait"], "handle_s": sums["handle"], "net_s": sums["net"], "window_s": win,
            "cpu_share": round(sums["cpu"] / win, 3) if win else None, "wait_share": round(sums["wait"] / win, 3) if win else None,
            "qsize_max": max(qs) if qs else None}


_SUMMED = ("parent_cpu_s", "parent_wait_s", "parent_handle_s", "parent_net_s", "parent_window_s", "parent_events")


def per_minute(events: list) -> list:
    """One row per minute of the window: asked decisions per second, runs ended, and the sys counters (mean of the
    10 s windows in it; parent_* are sums; <field>_max is the largest 2 s sample seen)."""
    if not events:
        return []
    t0 = min(e["t"] for e in events)
    rows: dict[int, dict] = {}
    for e in events:
        m = int((e["t"] - t0) // 60)
        r = rows.setdefault(m, {"dec": 0, "runs": 0, "sys": []})
        if e["ev"] in ("tick", "end"):
            r["dec"] += e["dec"]
        if e["ev"] == "end":
            r["runs"] += 1
        if e["ev"] == "sys":
            r["sys"].append(e)
    out = []
    for m in sorted(rows):
        r = rows[m]
        row = {"minute": m, "dec_per_s": round(r["dec"] / 60, 1), "runs": r["runs"], "sys_windows": len(r["sys"])}
        keys = sorted({k for e in r["sys"] for k, v in e.items() if isinstance(v, (int, float)) and k not in ("t", "perf_samples")})
        for k in keys:
            vs = [e[k] for e in r["sys"] if isinstance(e.get(k), (int, float))]
            if k in _SUMMED:
                row[k] = round(sum(vs), 2)
            elif k.endswith("_max"):
                row[k] = round(max(vs), 2)
            else:
                row[k] = round(sum(vs) / len(vs), 2)
        out.append(row)
    return out


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
    sysev = [e for e in events if e["ev"] == "sys"]
    return {
        "parent": parent_summary(sysev), "per_minute": per_minute(events),
        "alt_checks": {"checked": sum(e.get("alt_checked") or 0 for e in ends), "mismatch": sum(e.get("alt_mismatch") or 0 for e in ends)},
        "extras": {k: _merge((e.get("extras") or {}).get(k, {}) for e in ends) for k in ("offered", "taken", "effect")},
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
        spec = {"kind": "run", **reset_spec(seed, a.character, extra_actions=a.extra_actions, full_select=a.full_select), "buff_max_hp": a.buff}
        jopts.update(env_trace=True, runs=[spec], out=str(out / f"rec-{seed}.json").replace("\\", "/"))
        job = scratch / "job.json"
        job.write_text(json.dumps(jopts), encoding="utf-8")
        cmd = [sys.executable, str(HERE / "run_referee.py"), "--game", str(game), "--job", str(job), "--scratch", str(scratch),
               "--no-install", "--below-normal", "--timeout", "900"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        rep = json.loads(r.stdout[r.stdout.index("{"):]) if "{" in r.stdout else {}
        ok = r.returncode == 0 and rep.get("network_connections") == [] and rep.get("real_profile_touched") is False
        bad += not ok
        write_record_meta(out, seed, a.character, a.extra_actions, a.full_select)
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
        meta = read_record_meta(out, rec.stem[4:], a.character)  # the stored flag, not the command line's
        m, done = env.reset(run["seed"] if "seed" in run else rec.stem[4:], hash=True, spec={"buff_max_hp": a.buff},
                            record=out / f"replay-{rec.stem[4:]}.json", character=meta["character"],
                            extra_actions=meta["extra_actions"], full_select=meta["full_select"])
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
        results.append({"seed": rec.stem[4:], "extra_actions": meta["extra_actions"], "full_select": meta["full_select"], "decisions": len(trace), "hash_equal": same, "first_diff": first,
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
        m, done = env.reset(f"SMOKE{r}", character=a.character)
        n = 0
        while not done:
            kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
            if n == 0 and r == 0:
                print("first decision:", json.dumps(m)[:600])
                o = m.get("obs", {})  # what the character starts with: the deck size and the relics
                print("start", json.dumps({"character": a.character, "deck": len(o.get("deck", [])), "relics": o.get("relics"),
                                           "hp": o.get("max_hp")}))
            m, done = env.step(random_agent(m, rng))
            n += 1
        print(f"run {r}: {m} in {time.time() - t0:.1f} s, {n / (time.time() - t0):.1f} decisions/s")
    print("kinds", kinds, "network", rr.connections(env.pid))
    print("exit", env.close())
    return 0


def _fields_row(ls: Any) -> dict:
    """The run's record fields for the end event, fenced like the hooks: a fault is reported as `fields_error`, never as a crash."""
    fields = getattr(ls, "fields", None)
    err = getattr(ls, "fields_error", None)
    if fields is None:
        return {}
    try:
        row = dict(fields.row())
    except Exception as e:  # noqa: BLE001
        row, err = {}, err or f"row: {type(e).__name__}: {e}"
    if err:
        row["fields_error"] = err
    return row


def _row_extra(env, restarts: int, timed: _Timed | None = None) -> dict:
    """What an end or crash row of a play worker adds: `t_end` (epoch seconds, to join to the machine sampler's lines), `boot_s` (the
    game process's start-to-ready seconds, for the process that played the run), `restarts` (how often this worker has restarted its
    game so far) and `streak` (the run's longest identical (screen, action) run, see _Timed)."""
    row: dict = {"t_end": round(time.time(), 1), "restarts": restarts}
    if env is not None:
        row["boot_s"] = round(env.boot_s, 1) if isinstance(getattr(env, "boot_s", None), (int, float)) else None
    if timed is not None:
        row["streak"] = dict(timed.streak)
    return row


def play_worker(wid: int, game: str, out: str, seeds: list, sims: int, q, step_timeout: float, ascension: int,
                agent_seed: int = 0, run_ids: list | None = None, stop_at: float = 0.0, character: str = "IRONCLAD",
                recycle: int = 0, run_deadline_s: float = 0.0, wrapup_deadline_s: float = 0.0,
                boot_budget_s: float = BOOT_BUDGET_S) -> None:
    """One game process plays its seeds under the PLAY agent in lockstep (referee/lockstep.py). `recycle` > 0 restarts the game
    after that many runs (a long block's game process otherwise grows; 0: never). `run_deadline_s` > 0 (the main loop's
    --run-deadline-s): every run arms faulthandler to write the stack of all threads to `out/scratch/worker-<wid>.stack` two minutes
    (at least 60 s) before the main loop would terminate this worker, and disarms it when the run ends: a worker that hangs in a
    native call or a loop leaves the frame it sat in there. `wrapup_deadline_s` > 0 (--wrapup-deadline-s): the same for the wrap-up
    (a `wrapup` event starts the main loop's clock for it; the dump comes at `max(60, limit - 60)` s into it into the same file and
    is cancelled once `done` is put): the stale path's cleanup after a run and everything after the last run, which no run clock
    covers.

    A game process that does not come up (no `ready` in BOOT_TIMEOUT_S, or the process gone first) is killed and started again on a fresh
    scratch folder (`boot_game`), for `boot_budget_s` seconds in all; the seed is charged a crash row (where: boot) only when every try
    failed, and then the worker's next seed gets one try, not the budget, until a start works (a game that cannot start at all costs a
    try per seed, not the budget per seed). Each failed try puts a `boot` event (see `boot_game`); the run clock is started again once a
    retried start works, so the minutes spent starting do not come out of the run's deadline."""
    below_normal_self()
    import lockstep  # noqa: PLC0415
    stack = None
    if run_deadline_s > 0 or wrapup_deadline_s > 0:
        (Path(out) / "scratch").mkdir(parents=True, exist_ok=True)
        stack = open(Path(out) / "scratch" / f"worker-{wid}.stack", "a", encoding="utf-8")
    arm_run = stack is not None and run_deadline_s > 0
    arm_wrap = stack is not None and wrapup_deadline_s > 0

    def wrap_begin():  # the rest is cleanup (game close, connection listing): the main loop's wrap-up clock starts (RunDeadline)
        q.put({"ev": "wrapup", "w": wid, "t": time.time()})
        if arm_wrap:
            faulthandler.dump_traceback_later(max(60, wrapup_deadline_s - 60), repeat=False, file=stack)

    env = None
    restarts = 0
    runs_here = 0
    boot_failed = False  # the last seed's start used up its tries: the next seed gets one
    for n_seed, seed in enumerate(seeds):
        if stop_at and time.time() > stop_at:  # the block's time limit: no new run; a worker's seeds go in order, so the
            break  # runs done are a prefix of each worker's share and stay paired with the emulator's
        run_ix = run_ids[n_seed] if run_ids else n_seed
        q.put({"ev": "start", "w": wid, "seed": seed, "t": time.time()})  # the main loop's run clock (RunDeadline) starts here
        if arm_run:
            faulthandler.dump_traceback_later(max(60, run_deadline_s - 120), repeat=False, file=stack)
        if recycle and env is not None and runs_here >= recycle:  # a fresh game process every `recycle` runs
            q.put({"ev": "net", "w": wid, "conns": sorted(rr.connections(env.pid))})
            env.close()
            env, restarts, runs_here = None, restarts + 1, 0
        if env is None:
            runs_here = 0
            first = restarts  # try k boots in scratch folder p<wid>r<first + k>

            def say(line, _first=first, _seed=seed):
                q.put({"ev": "boot", "w": wid, "seed": _seed, "restarts": _first + line["try"] - 1, "t": time.time(), **line})

            env, tries, err = boot_game(lambda k: GameEnv(Path(game), Path(out) / "scratch" / f"p{wid:02d}r{first + k}", install=False),
                                        0.0 if boot_failed else boot_budget_s, pause_s=BOOT_PAUSE_S, max_tries=BOOT_TRIES_MAX, say=say)
            restarts = first + tries - 1  # the folder of the last try
            if env is None:  # every try failed: this seed's crash row, and the next seed one try
                boot_failed = True
                q.put({"ev": "crash", "w": wid, "seed": seed, "where": "boot", "error": repr(err)[:300], "boot_tries": tries,
                       **_row_extra(None, restarts)})
                restarts += 1
                if arm_run:
                    faulthandler.cancel_dump_traceback_later()
                continue
            boot_failed = False
            if tries > 1:  # minutes were spent starting: the run's own clocks begin again
                q.put({"ev": "start", "w": wid, "seed": seed, "t": time.time()})
                if arm_run:
                    faulthandler.dump_traceback_later(max(60, run_deadline_s - 120), repeat=False, file=stack)
        runs_here += 1
        t0 = time.time()
        timed = _Timed(env, step_timeout, character)
        try:
            m, ls = lockstep.play_run(timed, seed, ascension=ascension, sims=sims,
                                 agent_seed=[agent_seed, run_ix], character=character)
            ev = {"ev": "end", "w": wid, "seed": seed, "character": character, "end": m.get("end"), "floor": m.get("floor"), "asked": m.get("asked"),
                  "s": round(time.time() - t0, 1), "stats": ls.stats, "lost": ls.lost, "divergences": ls.log.records,
                  "agent_seed": ls.agent_seed, "assumed": len(ls.assumed), **_fields_row(ls), **_row_extra(env, restarts, timed)}
            if m.get("stalled"):  # ended by the stall guard (lockstep.StallGuard): the answer the game kept ignoring
                ev["stalled"] = m["stalled"]
            if m.get("fight_errors"):  # why a fight stopped (the driver's FightLoop: max_actions, idle_timeout after step ...)
                ev["fight_errors"] = m["fight_errors"]
            if m.get("stale"):
                if arm_run:
                    faulthandler.cancel_dump_traceback_later()
                q.put(ev)
                wrap_begin()  # its cleanup is unclocked otherwise (no run is in flight); the next run's start event ends this clock
                q.put({"ev": "net", "w": wid, "conns": sorted(rr.connections(env.pid))})
                env.close()
                env, restarts = None, restarts + 1
                if arm_wrap:
                    faulthandler.cancel_dump_traceback_later()
                continue
        except (TimeoutError, GameDied, ValueError) as e:
            ev = {"ev": "crash", "w": wid, "seed": seed, "error": f"{type(e).__name__}: {e}"[:300], "s": round(time.time() - t0, 1), "agent_seed": [agent_seed, run_ix], **_row_extra(env, restarts, timed),
                  "divergences": getattr(getattr(e, "lockstep", None), "log", None) and e.lockstep.log.records or []}
            env.close()
            env, restarts = None, restarts + 1
        except Exception as e:  # noqa: BLE001 -- an agent/emulator bug is a crash of that run, reported
            ev = {"ev": "crash", "w": wid, "seed": seed, "where": "agent", "error": f"{type(e).__name__}: {e}"[:300],
                  "s": round(time.time() - t0, 1), "agent_seed": [agent_seed, run_ix], **_row_extra(env, restarts, timed),
                  "divergences": getattr(getattr(e, "lockstep", None), "log", None) and e.lockstep.log.records or []}
            env.close()  # the game is mid-run; a fresh one for the next seed
            env, restarts = None, restarts + 1
        if arm_run:  # the run is over (end row, or either crash branch): nothing to dump any more
            faulthandler.cancel_dump_traceback_later()
        q.put(ev)
    wrap_begin()  # every result is put: what is left (the final connection listing, the game close, the done) is the wrap-up
    if env is not None:
        q.put({"ev": "net", "w": wid, "conns": sorted(rr.connections(env.pid))})
        env.close()
    q.put({"ev": "done", "w": wid})
    if arm_wrap:
        faulthandler.cancel_dump_traceback_later()
    if stack is not None:
        stack.close()


RUN_DEADLINE_S = 1800.0  # --run-deadline-s default: a run that has not ended this long after its worker started it is given up
WRAPUP_DEADLINE_S = 300.0  # --wrapup-deadline-s default: a worker that said `wrapup` and not `done` this long after is stopped
DEADLINE_RESPAWN_CAP = 3  # --deadline-respawn-cap default: replacement workers one worker slot gets after run-deadline closes and deaths
POLL_S = 1.0  # the main loop's wait for the next event: also how soon a worker whose process is gone is seen when nothing else happens


def remaining_after(seeds: list, run_ids: list, seed) -> tuple[list, list]:
    """The seeds and run ids of a worker's share after `seed`, the one its run deadline closed it on: what a replacement worker plays.
    The closed seed keeps its crash row and is not played again, and each later seed keeps the run id (so the agent seed) it has in the
    block. `seed` last in the share, or not in it: nothing is left."""
    if seed not in seeds:
        return [], []
    k = seeds.index(seed)
    return list(seeds[k + 1:]), list(run_ids[k + 1:])


class RunDeadline:
    """The play loop's per-worker run clock (`--run-deadline-s`; 0: off).

    A worker puts a `start` event when a run begins and an `end` or `crash` event when it is over; nothing the worker does between
    the two is bounded from outside (a call into the native library that never returns, a hung `rr.connections`), and a worker that
    never says `done` holds the block open for ever. `seen` keeps, per worker, the seed and start time of its current run and the
    time of its last event; `overdue` names the workers whose current run is older than the deadline; `close` gives the crash row
    of such a worker's seed and stops listening to the worker. A worker with no run in flight (past `stop_at`, or between runs) is
    never overdue: it ends by itself.

    The wrap-up has its own clock (`wrapup_s`, `--wrapup-deadline-s`; 0: off): a worker puts `wrapup` when its last run's result is
    put (and before the cleanup of a stale run), then `done`; `overdue_wrapup` names the workers whose `wrapup` is older than the
    limit with no `done` (a new `start` ends the clock: the worker went on to another run). `close_wrapup` closes such a worker
    without a crash row: its runs all have rows.

    A worker whose process is gone without `done` (the emulator's fail-fast kills the whole process and posts nothing) is found by the
    play loop, which asks `close_dead` for the crash row (where: died) of the run it had in flight, if any, and where the rest of
    its share begins. `finished` holds the workers whose `done` is in."""

    def __init__(self, deadline_s: float, agent_seed: int, run_index: dict[int, dict[str, int]], wrapup_s: float = 0.0):
        self.deadline_s = float(deadline_s or 0)
        self.wrapup_s = float(wrapup_s or 0)
        self.wrapping: dict[int, float] = {}  # worker -> time of its `wrapup` event, until its next start or its done
        self.agent_seed = agent_seed
        self.run_index = run_index  # worker -> {seed: its run index in the block}, for the agent seed of a deadline row
        self.current: dict[int, tuple[str, float]] = {}  # worker -> (seed, start time) of the run in flight
        self.last_event: dict[int, float] = {}
        self.last_row: dict[int, str] = {}  # worker -> the seed of its last end or crash row
        self.finished: set[int] = set()  # workers that said done
        self.closed: set[int] = set()

    def seen(self, e: dict, now: float) -> bool:
        """Note one event of a worker. False for an event of a worker that was closed: its rows are not taken."""
        w = e["w"]
        if w in self.closed:
            return False
        self.last_event[w] = now
        if e["ev"] == "start":
            self.current[w] = (e["seed"], float(e.get("t", now)))
            self.wrapping.pop(w, None)
        elif e["ev"] == "wrapup":
            self.wrapping[w] = float(e.get("t", now))
        elif e["ev"] in ("end", "crash", "done"):
            self.current.pop(w, None)
            if e["ev"] == "done":
                self.wrapping.pop(w, None)
                self.finished.add(w)
            elif "seed" in e:
                self.last_row[w] = e["seed"]
        return True

    def overdue(self, now: float) -> list[int]:
        if self.deadline_s <= 0:
            return []
        return [w for w, (_seed, t) in self.current.items() if now - t > self.deadline_s]

    def overdue_wrapup(self, now: float) -> list[int]:
        if self.wrapup_s <= 0:
            return []
        return [w for w, t in self.wrapping.items() if now - t > self.wrapup_s]

    def close_wrapup(self, w: int, now: float) -> int:
        """Stop listening to a worker stuck in its wrap-up; the seconds it was in it (no row: every run of it has one)."""
        t = self.wrapping.pop(w)
        self.closed.add(w)
        return int(round(now - t))

    def _crash_row(self, w: int, seed: str, n: int, now: float, where: str, error: str) -> dict:
        """The crash row of a run the main loop gave up: `n` seconds since the worker began it."""
        row = {"ev": "crash", "w": w, "seed": seed, "where": where, "error": error,
               "s": float(n), "t_end": round(now, 1), "divergences": []}
        ix = self.run_index.get(w, {}).get(seed)
        if ix is not None:
            row["agent_seed"] = [self.agent_seed, ix]  # as the worker writes it (a resume checks it)
        return row

    def close(self, w: int, now: float) -> dict:
        """The crash row of the overdue worker's seed; the worker is closed (its later events are ignored)."""
        seed, t = self.current.pop(w)
        self.closed.add(w)
        n = int(round(now - t))
        return self._crash_row(w, seed, n, now, "deadline", f"worker deadline: no result after {n} s")

    def close_dead(self, w: int, now: float, exitcode: int | None) -> tuple[dict | None, str | None]:
        """Close a worker whose process has exited without `done`: (the crash row of its run in flight, the seed its share goes on after).
        The row (where: died, the exit code in the text and as `exitcode`) is None when no run was in flight (the worker died between
        runs, or after its last row). The seed is the one in flight, else the worker's last row; None when it had none: its whole share
        is left. The worker is closed (its later events are ignored)."""
        self.closed.add(w)
        self.wrapping.pop(w, None)
        cur = self.current.pop(w, None)
        if cur is None:
            return None, self.last_row.get(w)
        seed, t = cur
        what = "worker process exited " + ("(exit code unknown)" if exitcode is None else str(exitcode))
        if isinstance(exitcode, int) and exitcode > 255:
            what += f" (0x{exitcode:x})"
        row = self._crash_row(w, seed, int(round(now - t)), now, "died", what)
        row["exitcode"] = exitcode
        return row, seed


STACK_LINES = 80  # the most lines of a hung worker's stack a deadline row keeps
_STACK_FRAME = re.compile(r'^\s*File "(?P<file>.*)", line (?P<line>\d+) in (?P<func>.*?)\s*$')
_STACK_THREAD = re.compile(r"^\s*(?:Current thread|Thread)\s+(?P<id>0x[0-9a-fA-F]+)")


def reduce_stack(text: str, limit: int = STACK_LINES) -> list[str]:
    """The last faulthandler dump in `text` (after the last "Timeout (" marker; the whole text without one) as a list of lines that
    hold no path: a thread header becomes `Thread <id>`, a frame `basename:line in function`; any other line is kept only if it names
    no directory. Over `limit` lines, each thread keeps its innermost frames so that every thread still shows (the main thread's
    frame is the one that matters and faulthandler lists it last)."""
    _head, mark, tail = text.rpartition("Timeout (")
    lines = tail.splitlines()[1:] if mark else text.splitlines()  # the rest of the marker line is no stack
    threads: list[list[str]] = []
    for raw in lines:
        t = _STACK_THREAD.match(raw)
        f = _STACK_FRAME.match(raw)
        if t:
            threads.append([f"Thread {t.group('id')}"])
        elif f:
            if not threads:
                threads.append(["Thread ?"])
            base = re.split(r'[\\/]', f.group('file'))[-1]
            threads[-1].append(f"{base}:{f.group('line')} in {f.group('func')}")
        elif raw.strip() and "/" not in raw and "\\" not in raw and threads:
            threads[-1].append(raw.strip())
    if sum(len(t) for t in threads) > limit:
        keep = max(1, (limit - len(threads)) // max(1, len(threads)))
        threads = [t[:1 + keep] for t in threads]
    return [x for t in threads for x in t][:limit]


def read_stack_dump(path: Path) -> list[str] | None:
    """The reduced stack of the hung worker's last faulthandler dump in `path`; None when there is no file or no dump in it."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return reduce_stack(text) or None


class _Timed:
    """A GameEnv whose reset and step carry the play mode's step timeout. It also counts, for the run it plays, the longest run of
    identical consecutive (screen kind, action) pairs: a run-layer loop that answers one screen the same way again and again shows as
    a streak of hundreds where a normal run has a handful."""

    def __init__(self, env, timeout, character: str = "IRONCLAD"):
        self.env, self.timeout, self.character = env, timeout, character
        self._pair, self._n = None, 0
        self.streak: dict = {"n": 0, "screen": None, "action": None}

    def reset(self, *a, **k):
        k.setdefault("character", self.character)
        k.setdefault("extra_actions", True)  # play runs in lockstep with the extra actions on
        k.setdefault("full_select", True)  # and with fight card screens at their real min..max (driver/Selector.cs)
        self._pair, self._n, self.streak = None, 0, {"n": 0, "screen": None, "action": None}
        return self.env.reset(*a, timeout=self.timeout, **k)

    def step(self, a):
        pair = ((getattr(self.env, "last", None) or {}).get("kind"), int(a))  # the screen this action answers, and the action
        self._n = self._n + 1 if pair == self._pair else 1
        self._pair = pair
        if self._n > self.streak["n"]:
            self.streak = {"n": self._n, "screen": pair[0], "action": pair[1]}
        return self.env.step(a, timeout=self.timeout)

    def anchor_state(self):
        """GameEnv.anchor_state with the step timeout (None from an env without it)."""
        ask = getattr(self.env, "anchor_state", None)
        return ask(timeout=self.timeout) if ask is not None else None


def select_runs(seeds: list, runs: int, only: str | None) -> tuple[list, list]:
    """(seeds, run indices) a play block starts. The agent's RNG is [agent_seed, run index] and the index is the seed's place in the
    block, so `only` (comma-separated seeds) keeps each chosen seed's place in the FULL file, not in the subset: the same agent path
    as the block's own run. Without `only`: the first `runs` seeds, indices 0...runs-1."""
    if not only:
        seeds = seeds[:runs]
        return seeds, list(range(len(seeds)))
    want = [x.strip() for x in only.split(",") if x.strip()]
    missing = [x for x in want if x not in seeds]
    if missing:
        raise ValueError(f"--only: not in the seeds file: {','.join(missing)}")
    ids = sorted({seeds.index(x) for x in want})
    return [seeds[i] for i in ids], ids


def select_engine(engine: str) -> None:
    """Set the emulator engine of this process and the workers it spawns.

    STS2_ENGINE, read by sts2_gym.native when it loads: "old" clears it, whatever the environment said.
    """
    if engine == "old":
        os.environ.pop("STS2_ENGINE", None)
    else:
        os.environ["STS2_ENGINE"] = engine


def _read_rows(path: Path) -> list[dict]:
    """The whole lines of a runs file that parse as a row with a seed; a line cut by a stop, or a blank one, is dropped."""
    rows: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for line in text.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("seed") is not None:
            rows.append(r)
    return rows


def _divergence_rows(events: list[dict]) -> list[dict]:
    return [dict(d, seed=e["seed"]) for e in events if e["ev"] in ("end", "crash") for d in e.get("divergences", [])]


def resume_block(out: Path, stamp: dict, ids: dict[str, int], agent_seed: int) -> tuple[list[dict], float]:
    """What a `play --resume` keeps of an earlier session of the block in `out`: (finished rows, wall seconds the earlier sessions ran).

    A kept row is the first `end` or `crash` row of a seed of this block (`ids`: seed -> its place in the block, the run index of the
    agent's RNG [agent seed, index]); a boot failure is no run and is played again, and so is a row whose recorded agent seed is not
    [agent seed, that seed's index], so a kept row is always the row a fresh block would have played. A `deadline` or `died` row (the
    run was given up: its worker was stopped, or its process was gone) is kept like any crash row, and its seed is not played again.
    `out/resume.json` holds the
    block's identity (stamp: agent, sims or config, ascension, character, engine, agent seed). Rows written under another identity
    are not kept (a stderr line) and the block starts empty; a file with rows and no identity (written before this existed) is kept
    when every row's seed belongs to the block. A half-written last line is dropped. runs.jsonl and divergences.jsonl are rewritten
    with the kept rows only; resume.json is written. `out/wall.json` holds the seconds the block has run over all its sessions."""
    out.mkdir(parents=True, exist_ok=True)
    runs_file, stamp_file, wall_file = out / "runs.jsonl", out / "resume.json", out / "wall.json"
    want = json.loads(json.dumps(stamp))
    try:
        have = json.loads(stamp_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        have = None
    raw = _read_rows(runs_file)
    same = have == want or (have is None and bool(raw) and all(r["seed"] in ids for r in raw))
    kept: list[dict] = []
    if same:
        seen: set[str] = set()
        for r in raw:
            s = r["seed"]
            if s not in ids or s in seen or r.get("ev") not in ("end", "crash") or r.get("where") == "boot":
                continue
            if r.get("agent_seed", [agent_seed, ids[s]]) != [agent_seed, ids[s]]:
                print(f"resume: the row of {s} carries another agent seed or run index; it is played again", file=sys.stderr, flush=True)
                continue
            seen.add(s)
            kept.append(r)
    elif raw or have is not None:
        print("resume: runs.jsonl was written for another agent or setting; it starts empty", file=sys.stderr, flush=True)
    before = 0.0
    if same:
        try:
            before = float(json.loads(wall_file.read_text(encoding="utf-8"))["wall_s"])
        except (OSError, ValueError, KeyError, TypeError):
            before = 0.0
    tmp = runs_file.with_name("runs.jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    os.replace(tmp, runs_file)
    (out / "divergences.jsonl").write_text("\n".join(json.dumps(d) for d in _divergence_rows(kept)), encoding="utf-8")
    stamp_file.write_text(json.dumps(want), encoding="utf-8")
    write_wall(out, before)
    return kept, before


def write_wall(out: Path, seconds: float) -> None:
    (out / "wall.json").write_text(json.dumps({"wall_s": round(seconds, 1)}), encoding="utf-8")


def gen_dir_tail() -> str | None:
    """The agent a block without an explicit identity was played by: the last two parts of the generation folder SPIRELINE_GEN_DIR
    names (the way gen_loop.Agent.describe writes `dir`), None for generation 0."""
    v = os.environ.get("SPIRELINE_GEN_DIR")
    return "/".join(Path(v).parts[-2:]) if v else None


def _start_sampler(a, out: Path, events: list, resume: bool):
    """The machine sampler of `--sample-machine` (referee/machine_sampler.py), started; None when off or when it cannot start: a
    sampler fault is never the block's. A fresh block clears the file of an earlier one, a resumed block appends to it."""
    every = float(getattr(a, "sample_machine", 0) or 0)
    if every <= 0:
        return None
    try:
        import machine_sampler  # noqa: PLC0415

        if not resume:
            (out / "machine.jsonl").unlink(missing_ok=True)
        s = machine_sampler.MachineSampler(out / "machine.jsonl", every, None,  # the readings are opened by the sampler's own thread
                                           runs_done=lambda: sum(1 for e in list(events) if e["ev"] in ("end", "crash")),
                                           make_probes=lambda: machine_sampler.WinProbes(out.drive or None))
        s.start()
        return s
    except Exception as e:  # noqa: BLE001
        print(f"machine sampler not started ({type(e).__name__})", file=sys.stderr, flush=True)
        return None


def play(a) -> int:
    """`--agent play`: N runs on --procs game processes, the PLAY agent in lockstep with the emulator.

    With `a.resume` the block keeps the finished rows of an earlier session (`resume_block`) and plays only the seeds it has no
    row for, each with the run id and agent seed it has in a fresh block; the summary counts all rows and sums the sessions' wall.

    A worker that goes silent is closed one of two ways, and its seeds go on to a replacement worker either way (a new id; at most
    --deadline-respawn-cap per original slot, both ways together): its run clock runs out (where: deadline), or its process is gone
    without `done` (where: died; the emulator's fail-fast kills the process and nothing is posted). The loop checks the processes on
    every pass and drains the queue once more after it sees an exit, so a worker that posted its rows and `done` and then exited is
    never read as dead; a process that exits after its last row or in its wrap-up is left alone (no row, no replacement)."""
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    select_engine(getattr(a, "engine", "old"))  # before the workers spawn: each loads the library on it
    import lockstep  # noqa: PLC0415

    lockstep.require_character(a.character)  # the emulator side must be able to start it: exit before any game boots
    # The workers boot their games with install=False, so this tree's driver goes into the game copy once, here: a block must
    # never play whatever driver an earlier bench or record left in a shared copy.
    rr.check_build(Path(a.game).resolve())
    rr.install(Path(a.game).resolve(), HERE / "driver" / "bin" / "Release" / "net9.0")
    seeds = [f"A1S{i:04d}" for i in range(max(a.runs, 0) if not getattr(a, "only", None) else 0)]
    if a.seeds_file:  # a committed seed block (rl/gen0.py): the first --runs of it, in order
        doc = json.loads(Path(a.seeds_file).read_text(encoding="utf-8"))
        seeds = doc["seeds"] if isinstance(doc, dict) else doc
    elif getattr(a, "only", None):
        raise ValueError("--only needs --seeds-file")
    seeds, run_ids = select_runs(seeds, a.runs, getattr(a, "only", None))
    agent_seed = a.agent_seed if a.agent_seed is not None else int.from_bytes(os.urandom(6), "big")  # not a game seed
    resume = bool(getattr(a, "resume", False))
    held: list[dict] = []
    wall_before = 0.0
    if resume:  # the identity includes the agent seed: a block without --agent-seed has a new random one and starts empty
        shutil.rmtree(out / "scratch", ignore_errors=True)  # the game working copies of the stopped session
        raw_id = getattr(a, "agent_identity", None)
        stamp = {"agent": json.loads(raw_id) if raw_id else gen_dir_tail(), "sims": a.sims, "ascension": a.ascension,
                 "character": a.character, "engine": getattr(a, "engine", "old"), "agent_seed": agent_seed}
        held, wall_before = resume_block(out, stamp, dict(zip(seeds, run_ids)), agent_seed)
        have_rows = {r["seed"] for r in held}
        todo = [(s, i) for s, i in zip(seeds, run_ids) if s not in have_rows]  # run ids stay those of the full seed list
        seeds, run_ids = [s for s, _ in todo], [i for _, i in todo]
        if held:
            print(f"resume: runs.jsonl holds {len(held)} rows; those seeds are not played again", file=sys.stderr, flush=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    # the time limit counts the seconds the block has already run, so an interrupted block is not given its hours again
    stop_at = time.time() + a.stop_after_h * 3600 - wall_before if a.stop_after_h else 0.0
    live = open(out / "runs.jsonl", "a" if resume else "w", encoding="utf-8")  # one line per finished run as it ends; rewritten whole at the end
    wids = [w for w in range(a.procs) if not resume or seeds[w::a.procs]]
    deadline_s = getattr(a, "run_deadline_s", RUN_DEADLINE_S)
    wrapup_s = getattr(a, "wrapup_deadline_s", WRAPUP_DEADLINE_S)
    boot_budget_s = getattr(a, "boot_budget_s", BOOT_BUDGET_S)

    def make_proc(w, w_seeds, w_ids):
        return ctx.Process(target=play_worker, args=(w, a.game, str(out), w_seeds, a.sims, q, a.step_timeout, a.ascension,
                                                     agent_seed, w_ids, stop_at, a.character, getattr(a, "recycle", 0), deadline_s, wrapup_s,
                                                     boot_budget_s))

    share = {w: (seeds[w::a.procs], run_ids[w::a.procs]) for w in wids}  # worker -> (seeds, run ids) it was handed
    procs = [make_proc(w, *share[w]) for w in wids]
    proc_of = dict(zip(wids, procs))
    # a worker closed for its run deadline, or found dead (its process gone without `done`), gets a replacement for the rest of its share
    # (a new worker id, so the closed one's late events stay ignored and its scratch folders p<NN>r<k> stay apart): slot_of names the
    # original slot, respawns counts per slot (the one cap of both ways), respawned counts per way (the summary)
    slot_of = {w: w for w in wids}
    respawns: dict[int, int] = {}
    respawned = {"deadline": 0, "died": 0}
    respawn_cap = getattr(a, "deadline_respawn_cap", DEADLINE_RESPAWN_CAP)
    next_wid = a.procs
    unplayed_after_cap = 0
    watch = RunDeadline(deadline_s, agent_seed, {w: dict(zip(*share[w])) for w in wids}, wrapup_s)
    t0 = time.time()
    beat = t0
    for p in procs:
        p.start()
    events, done, nets = list(held), 0, set()
    wrapup_terminated = 0
    boots: list[dict] = []  # the `boot` events of this session: every start that failed, and a retried or slow one that worked
    boot_file: Any = None

    def stop(w):  # the one way a stuck worker is stopped: through its own Process object, never by name or pid
        proc_of[w].terminate()
        proc_of[w].join(10)

    def hand_on(w, rest_seeds, rest_ids, kind):
        """Hand the rest of closed worker `w`'s share (`kind`: deadline or died) to a replacement worker: nothing when none is left or
        the block's time limit has passed, nothing but a count (unplayed_after_cap) when the slot has had its replacements."""
        nonlocal next_wid, unplayed_after_cap
        if not rest_seeds or (stop_at and time.time() > stop_at):  # its share is over, or the block's time limit: nothing to hand on
            return
        slot = slot_of[w]
        if respawns.get(slot, 0) >= respawn_cap:  # a bad seed must not loop: the rest of the slot stays unplayed and the summary says so
            unplayed_after_cap += len(rest_seeds)
            print(f"worker {w}: slot {slot} has had {respawn_cap} replacement workers; its {len(rest_seeds)} remaining seeds stay unplayed", flush=True)
            return
        respawns[slot] = respawns.get(slot, 0) + 1
        respawned[kind] += 1
        new_w, next_wid = next_wid, next_wid + 1  # the old process is gone (stopped or dead), so at most --procs workers live
        share[new_w], slot_of[new_w] = (rest_seeds, rest_ids), slot
        watch.run_index[new_w] = dict(zip(rest_seeds, rest_ids))
        proc_of[new_w] = make_proc(new_w, rest_seeds, rest_ids)
        procs.append(proc_of[new_w])
        proc_of[new_w].start()
        print(f"replacement worker {new_w} for worker {w} ({kind}; slot {slot}, {respawns[slot]} of {respawn_cap}): {len(rest_seeds)} remaining seeds", flush=True)

    def handle(e):  # one event of the queue
        nonlocal done, boot_file
        fresh = watch.seen(e, time.time())
        if e["ev"] == "net":
            nets.update(e["conns"])
        elif not fresh or e["ev"] in ("start", "wrapup"):
            return  # a closed worker's late row, or the run and wrap-up clocks' own events
        elif e["ev"] == "boot":  # a start's timing line: boot.jsonl and the log, never a row of the block
            boots.append(e)
            if boot_file is None:
                boot_file = open(out / "boot.jsonl", "a" if resume else "w", encoding="utf-8")
            boot_file.write(json.dumps(e) + chr(10))
            boot_file.flush()
            print("boot " + json.dumps(e)[:600], flush=True)
        elif e["ev"] == "done":
            done += 1
        else:
            events.append(e)
            live.write(json.dumps(e) + chr(10))
            live.flush()
            print(json.dumps({k: v for k, v in e.items() if k != "divergences"})[:300], flush=True)

    def drain():  # every event already in the queue
        while True:
            try:
                e = q.get(timeout=0.1)
            except queue.Empty:
                return
            handle(e)

    def reap_dead():
        """Workers whose process has exited without `done` and that no clock closed: they died (an emulator fail-fast kills the process
        and posts nothing). What a worker put before its exit may still be in the queue, so the queue is drained once more after the exit
        is seen, and only a worker still without its `done` after that is read as dead. Its run in flight gets a crash row (where: died)
        and the rest of its share goes to a replacement. A worker with no run in flight and nothing left to play (it exited after its last
        row or in its wrap-up) is finished with no row; one that died between runs gets no row but its remaining seeds are handed on."""
        nonlocal done
        gone = [w for w, p in proc_of.items() if w not in watch.closed and w not in watch.finished and not p.is_alive()]
        if not gone:
            return
        drain()
        for w in gone:
            if w in watch.finished:  # its done was in the queue: it ended by itself
                continue
            now = time.time()
            code = getattr(proc_of[w], "exitcode", None)
            proc_of[w].join(10)
            row, after = watch.close_dead(w, now, code)
            done += 1
            if after is None:
                rest_seeds, rest_ids = list(share[w][0]), list(share[w][1])  # it posted no result: its whole share is left
            else:
                rest_seeds, rest_ids = remaining_after(*share[w], after)
            if row is None:
                if not rest_seeds:
                    print(f"worker {w}: process exited ({code}) without done after its last result; nothing is left to play", file=sys.stderr, flush=True)
                    continue
                print(f"worker {w}: process exited ({code}) without done between runs; its {len(rest_seeds)} remaining seeds go on", flush=True)
            else:
                events.append(row)
                live.write(json.dumps(row) + chr(10))
                live.flush()
                print(f"worker {w}: {row['error']} (seed {row['seed']}); counted as done", flush=True)
            hand_on(w, rest_seeds, rest_ids, "died")

    sampler = _start_sampler(a, out, events, resume)
    while done < len(procs):
        if resume and time.time() - beat >= 30:  # the seconds run so far, so a stop leaves the block's wall behind
            beat = time.time()
            write_wall(out, wall_before + beat - t0)
        reap_dead()  # first: a worker that is gone is dead, whatever its clocks say
        for w in watch.overdue(time.time()):  # a run that outlived --run-deadline-s: its worker is stopped, its seed has a crash row
            row = watch.close(w, time.time())
            stop(w)
            stack = read_stack_dump(out / "scratch" / f"worker-{w}.stack")  # the worker's own faulthandler dump (play_worker)
            if stack:  # the scratch folder is removed after the block, so the frame goes into the row (no directory in it)
                row["stack"] = stack
            events.append(row)
            live.write(json.dumps(row) + chr(10))
            live.flush()
            print(f"worker {w}: {row['error']} (seed {row['seed']}); worker terminated, counted as done", flush=True)
            done += 1
            hand_on(w, *remaining_after(*share[w], row["seed"]), "deadline")
        for w in watch.overdue_wrapup(time.time()):  # a worker in its wrap-up past --wrapup-deadline-s: stopped, no row (its runs have theirs)
            n = watch.close_wrapup(w, time.time())
            stop(w)
            wrapup_terminated += 1
            print(f"worker {w} wrap-up deadline after {n} s: terminated", file=sys.stderr, flush=True)
            done += 1
        if done >= len(procs):
            break
        try:
            e = q.get(timeout=POLL_S)
        except queue.Empty:
            continue  # the next pass looks at the processes again
        handle(e)
    for p in procs:
        p.join(30)
    if sampler is not None:
        sampler.stop()
    session = time.time() - t0
    wall = session + wall_before  # the block's wall: the sessions' sum (wall_before: 0.0 without --resume)
    live.close()
    if boot_file is not None:
        boot_file.close()
    (out / "runs.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    ends = [e for e in events if e["ev"] == "end"]
    dec = sum(e["stats"]["decisions"] for e in ends)
    rank = sum(e["stats"]["ranked"] for e in ends)
    divs = _divergence_rows(events)
    first = {}
    for d in divs:
        first.setdefault(d["type"], []).append((d["decision"], d["floor"], d.get("reason") or d["diff"][:2]))
    summary = {"agent_seed": agent_seed, "character": a.character, "engine": getattr(a, "engine", "old"), "sims": a.sims, "runs": len(ends), "crashes": sum(e["ev"] == "crash" for e in events), "wrapup_terminated": wrapup_terminated,
               "deadline_respawns": respawned["deadline"], "died_respawns": respawned["died"], "unplayed_after_cap": unplayed_after_cap,
               "boot_retries": sum(1 for b in boots if not b["ok"] and not b["final"]), "boot_gave_up": sum(1 for b in boots if not b["ok"] and b["final"]), "wins": sum(1 for e in ends if e["end"] in ("win", "victory")),
               "ends": sorted({str(e["end"]) for e in ends}),
               "mean_floor": round(sum(e["floor"] or 0 for e in ends) / max(1, len(ends)), 2), "decisions": dec,
               "decisions_per_s": round(dec / max(wall, 1e-9), 2), "wall_s": round(wall, 1), "ranked_share": round(rank / max(1, dec), 3),
               "diverged_decisions": sum(e["stats"]["diverged"] for e in ends),
               "divergence_rate": round(sum(e["stats"]["diverged"] for e in ends) / max(1, dec), 4),
               "reanchors": sum(e["stats"]["reanchors"] for e in ends), "reanchor_failed": sum(e["stats"]["reanchor_failed"] for e in ends),
               "first_loss_by_reason": _count(e["stats"]["first_loss"]["reason"].split(":")[0] + ":" + (e["stats"]["first_loss"]["reason"].split(":")[1:2] or [""])[0] for e in ends if e["stats"]["first_loss"]),
               "first_divergences_by_type": {k: v[:5] for k, v in first.items()},
               "gaps": sum(e["stats"].get("gaps", 0) for e in ends), "resumed": sum(e["stats"].get("resumed", 0) for e in ends),
               "losses_by_reason": _merge(e["stats"].get("losses", {}) for e in ends),
               "reanchor_fail_reasons": _merge(e["stats"].get("reanchor_fail_reasons", {}) for e in ends),
               "reanchor_by_plan": _merge(e["stats"].get("reanchor_by_plan", {}) for e in ends),
               **_provenance(ends),
               "hidden": _hidden(ends),
               "divergences_by_type": _count(f"{d['type']}:{'gap' if d.get('gap') else 'state'}:{(d.get('reason') or ','.join(x[0] for x in d['diff'][:3]))[:40]}" for d in divs),
               "run_s_total": round(sum(e["s"] for e in ends), 1),
               "decisions_per_s_per_proc": round(dec / max(1e-9, sum(e["s"] for e in ends)), 2),
               "ranked_decisions_per_s_per_proc": round(rank / max(1e-9, sum(e["s"] for e in ends)), 3),
               "network_connections": sorted(nets), "errors": [e for e in events if e["ev"] == "crash"][:10]}
    if boots:  # the starts that failed or were slow, with their readings (the full lines are in boot.jsonl)
        summary["boot"] = {"failed_tries": sum(not b["ok"] for b in boots), "failed_s": round(sum(b["s"] for b in boots if not b["ok"]), 1),
                           "slowest_s": max(b["s"] for b in boots), "lines": boots[:20]}
    if resume:
        summary.update(resumed_rows=len(held), wall_s_before=round(wall_before, 1), wall_s_session=round(session, 1))  # wall_s: both
        write_wall(out, wall)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "divergences.jsonl").write_text("\n".join(json.dumps(d) for d in divs), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 1 if nets else 0


def _provenance(ends) -> dict:
    """What the block's re-anchors were seeded from, summed over the finished runs (the lockstep's per-run stats): how many held
    by the game's hidden state ("hidden"), by the observation alone, by moved streams only, or by none of them; and the streams
    the game's counters named against the streams the emulator moved; and the walks of suspended emulators (walks, floors walked,
    rules picks, fights resolved unplayed). The four ways are always present, so a block that ran
    on observation anchoring reads "hidden": 0 rather than looking like one that did not report."""
    st = [e["stats"] for e in ends]
    ways = {k: 0 for k in ("hidden", "observation", "streams", "none")}
    for k, v in _merge(s.get("anchored_with", {}) for s in st).items():
        ways[k] = ways.get(k, 0) + v
    return {"anchored_with": ways,
            "streams_named": sum(s.get("streams_named", 0) for s in st),
            "streams_moved": sum(s.get("streams_moved", 0) for s in st),
            "streams_unmoved_names": _merge(s.get("streams_unmoved_names", {}) for s in st),
            "walk": {k: sum(s.get("walk", {}).get(k, 0) for s in st) for k in ("walks", "floors", "picks", "fights")}}


def _hidden(ends) -> dict:
    """The lockstep's second, unranked comparison (referee/hidden_diff.py), summed over the finished runs' stats: the decisions it
    looked at, the ones with a difference, the divergence records that carry one (`records`; the ranked records are the same
    with or without it), the runs that had a difference, and by field the records carrying it and the unbroken runs of
    differing decisions that began with it. Nothing here enters `diverged_decisions`, the divergences or the ranking."""
    st = [e["stats"].get("hidden") or {} for e in ends]
    return {"checked": sum(s.get("checked", 0) for s in st), "diverged_decisions": sum(s.get("diverged", 0) for s in st),
            "records": sum(s.get("records", 0) for s in st), "runs_with_a_difference": sum(1 for s in st if s.get("diverged")),
            "errors": sum(s.get("errors", 0) for s in st),
            "by_field": _merge(s.get("by_field", {}) for s in st), "parts_by_field": _merge(s.get("parts_by_field", {}) for s in st)}


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


def parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--run-deadline-s", type=float, default=RUN_DEADLINE_S,
                    help="play: a worker whose current run has no result this many seconds after it began is terminated and its seed gets a "
                         "crash row (where: deadline); the block closes with the other workers' rows (0: off)")
    ap.add_argument("--deadline-respawn-cap", type=int, default=DEADLINE_RESPAWN_CAP,
                    help="play: a worker closed for its run deadline, or found dead (its process gone without done: a crash row, where: died), "
                         "is replaced by a new worker (a new id) that plays the rest of its seeds; at most this many replacements per original "
                         "worker slot, deadline closes and deaths together, then the rest stays unplayed (summary: deadline_respawns, "
                         "died_respawns, unplayed_after_cap; 0: none)")
    ap.add_argument("--wrapup-deadline-s", type=float, default=WRAPUP_DEADLINE_S,
                    help="play: a worker that has put its last result (event wrapup) and has not said done this many seconds later is "
                         "terminated: no crash row (its runs have theirs), one stderr line, counted as wrapup_terminated in the summary (0: off)")
    ap.add_argument("--seeds-file", default=None, help="play: JSON seed block (default: A1S0000...)")
    ap.add_argument("--only", default=None, help="play: only these seeds of --seeds-file (comma-separated), each keeping its place in the full file as its run index")
    ap.add_argument("--resume", action="store_true",
                    help="play: keep the finished rows of an earlier session of this block (same identity: agent, sims or config, ascension, "
                         "character, engine, --agent-seed) and play only the seeds without a row, each with its fresh-block run id and agent seed")
    ap.add_argument("--agent-identity", default=None,
                    help="play --resume: the agent's description as JSON (gen_loop.Agent.describe) for the identity file; default: "
                         "SPIRELINE_GEN_DIR's last two parts")
    ap.add_argument("--sims", type=lambda v: int(v) if v.isdigit() else v, default=60,
                    help="simulations per combat decision, or a search config name (rl/search_configs.py)")
    ap.add_argument("--agent-seed", type=int, default=None, help="play: the agent's own seed (default: random, recorded); never derived from a game seed")
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--character", default="IRONCLAD", help="the run's character (the game's character id)")
    ap.add_argument("--step-timeout", type=float, default=180)
    ap.add_argument("--boot-budget-s", type=float, default=BOOT_BUDGET_S,
                    help="play: a game process that does not come up (no ready in 180 s, or the process gone first) is killed and started "
                         "again on a fresh scratch folder for this many seconds in all before its seed gets a crash row (where: boot); "
                         "0: one try. Each failed try is a line of boot.jsonl and the log (summary: boot_retries, boot_gave_up)")
    ap.add_argument("--engine", choices=["old", "core2"], default="old",
                    help="play: the emulator's engine in the lockstep (docs/rebuild/NATIVE_KERNEL.md); every other mode runs old")
    ap.add_argument("--extra-actions", action="store_true", help="record: record with the extra actions on (replay uses the stored flag)")
    ap.add_argument("--full-select", action="store_true", help="record: record with fight card screens at their real min..max (replay uses the stored flag)")
    ap.add_argument("--max-runs", type=int, default=0, help="bench: stop after this many runs in all (0: run for --seconds)")
    ap.add_argument("--recycle", type=int, default=0, help="bench, play: restart each game after this many runs (0: never)")
    ap.add_argument("--sample-machine", type=float, default=0.0,
                    help="play: every this many seconds append one line of machine readings (cpu, memory, disk, timer, game processes, "
                         "runs done) to machine.jsonl in --out (0: off)")
    return ap


def main() -> int:
    a = parser().parse_args()
    select_engine(a.engine if a.mode == "play" else "old")
    return {"smoke": smoke, "bench": bench, "record": record, "replay": replay, "play": play}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
