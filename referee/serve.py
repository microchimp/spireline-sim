#!/usr/bin/env python3
"""Fight-server dispatcher: N long-lived game processes fed from one shared inbox.

    python referee/serve.py --game <game copy> --specs <job.json> [--servers 4] [--repeat 1] [--out results.json]

--specs is a job file (or a bare JSON list) whose "runs" are fight specs; its fast_reset / no_preload / fast_patches /
idle_frames / stall_s keys apply to every server. Each server is one headless game on its own scratch
profile (same safety checks as run_referee.py: known sts2.dll sha256, override.cfg, APPDATA in scratch,
no network connections, real profile untouched). The dispatcher writes one spec file per fight into
<run-dir>/inbox; whichever server is free claims the oldest one (driver/Server.cs), plays it and appends
a JSON line to its own <run-dir>/out/s<NN>.jsonl. Results are merged into --out (same format as a
runs-mode out file, in spec order) and results.jsonl (arrival order).

A server that dies or hangs is restarted; the spec it had claimed goes back to the inbox once, and is
recorded as stalled if it kills a server a second time. One hot spare (--spares) is booted and held
(hold.<pid> in the inbox) so a restart costs no boot time. At the end STOP goes into the inbox.
--verify-against <out.json> compares every result with a reference from "runs" mode, ignoring timing.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_referee as rr  # noqa: E402

TIMING_KEYS = {"timing_ms", "phase_ms", "cost", "cost_setup"}  # per-run timing and CPU/frame cost, never compared
META_KEYS = {"_id", "_pid", "_t"}  # added by the server to each result line


class Server:
    def __init__(self, sid: int, spare: bool):
        self.sid, self.spare = sid, spare
        self.name = f"s{sid:02d}"
        self.proc: subprocess.Popen | None = None
        self.pid = 0
        self.t_spawn = 0.0
        self.t_ready: float | None = None  # unix seconds, stamped by the driver
        self.t_exit: float | None = None
        self.rc: int | None = None
        self.offset = 0
        self.lines: list[dict] = []
        self.killed = ""
        self.connections: set[str] = set()
        self.log = None

    @property
    def alive(self) -> bool:
        return self.t_exit is None


def read_lines(path: Path, offset: int) -> tuple[list[dict], int]:
    """New complete JSON lines of an append-only file; a half-written last line waits for next time."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except FileNotFoundError:
        return [], offset
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    out = []
    for raw in data[:end].split(b"\n"):
        if raw.strip():
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                out.append({"error": "unparsable result line"})
    return out, offset + end + 1


def strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in TIMING_KEYS and k not in META_KEYS}


def drop_keys(x, keys: set):
    """x without the named keys at any depth: fields a newer driver adds that an older reference lacks."""
    if isinstance(x, dict):
        return {k: drop_keys(v, keys) for k, v in x.items() if k not in keys}
    if isinstance(x, list):
        return [drop_keys(v, keys) for v in x]
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--game", type=Path, required=True)
    ap.add_argument("--specs", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=1, help="play the spec list this many times")
    ap.add_argument("--servers", type=int, default=4)
    ap.add_argument("--spares", type=int, default=1, choices=(0, 1), help="hot spares booted and held")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="merged results, runs-mode format (default <run-dir>/results.json)")
    ap.add_argument("--stall-s", type=int, default=None, help="driver watchdog: seconds without progress (default 20)")
    ap.add_argument("--idle-frames", type=float, default=None)
    ap.add_argument("--preload", action="store_true", help="leave the game's asset preloader on (default off, as batch.py)")
    ap.add_argument("--slow-reset", action="store_true", help="return to the main menu between fights (default fast_reset)")
    ap.add_argument("--boot-timeout", type=float, default=180)
    ap.add_argument("--timeout", type=float, default=7200, help="whole run, seconds")
    ap.add_argument("--max-restarts", type=int, default=None, help="give up after this many restarts (default 3*servers+3)")
    ap.add_argument("--net-interval", type=float, default=1.0, help="seconds between netstat calls (one call for all servers)")
    ap.add_argument("--normal-priority", action="store_true", help="do not lower the games to BelowNormal")
    ap.add_argument("--no-install", action="store_true")
    ap.add_argument("--build", type=Path, default=HERE / "driver" / "bin" / "Release" / "net9.0")
    ap.add_argument("--no-patches", action="store_true", help="play without the step 2 speed patches (driver/Patches.cs)")
    ap.add_argument("--verify-against", type=Path, default=None, help="runs-mode out file to compare every result with")
    ap.add_argument("--verify-ignore", default="", help="comma-separated keys dropped at any depth before --verify-against "
                    "compares (fields the reference predates, e.g. move,intents)")
    a = ap.parse_args()

    game = a.game.resolve()
    digest = rr.check_build(game)
    job_in = json.loads(a.specs.read_text(encoding="utf-8"))
    runs = (job_in["runs"] if isinstance(job_in, dict) else job_in) * a.repeat
    total, n_act = len(runs), max(1, a.servers)
    jopts = {k: v for k, v in (job_in.items() if isinstance(job_in, dict) else []) if k in ("idle_frames", "stall_s")}
    jopts["fast_reset"] = not a.slow_reset
    jopts["no_preload"] = not a.preload
    jopts["fast_patches"] = not a.no_patches  # SPEED_PLAN step 2 (driver/Patches.cs)
    if a.stall_s is not None:
        jopts["stall_s"] = a.stall_s
    if a.idle_frames is not None:
        jopts["idle_frames"] = a.idle_frames
    stall_s = int(jopts.get("stall_s", 20))
    max_restarts = a.max_restarts if a.max_restarts is not None else 3 * n_act + 3

    run_dir = (a.run_dir or game.parent / "serve" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    inbox, outdir = run_dir / "inbox", run_dir / "out"
    for d in (inbox, outdir):
        d.mkdir(parents=True, exist_ok=True)
    if not a.no_install:
        for attempt in range(30):  # a running game holds the mod DLL open; wait for it
            try:
                rr.install(game, a.build)
                break
            except PermissionError:
                time.sleep(2)
        else:
            print("could not install the driver: mod DLL locked", file=sys.stderr)
            return 2

    before = rr.real_profile_state()
    t_start = time.time()
    servers: list[Server] = []
    results: dict[str, dict] = {}  # spec id -> result line
    arrival: list[dict] = []
    attempts: collections.Counter = collections.Counter()
    ids = [f"{i:06d}" for i in range(total)]
    fed = 0
    restarts = 0
    requeued = 0
    next_sid = 0
    seen_net: set[str] = set()
    events: list[str] = []

    def log(msg: str) -> None:
        line = f"[{time.time() - t_start:7.1f}s] {msg}"
        events.append(line)
        print(line, flush=True)

    def spawn(spare: bool) -> Server:
        nonlocal next_sid
        s = Server(next_sid, spare)
        next_sid += 1
        scratch = run_dir / "scratch" / s.name
        (scratch / "appdata").mkdir(parents=True, exist_ok=True)
        (scratch / "localappdata").mkdir(parents=True, exist_ok=True)
        rr.write_settings(scratch, boot_opts=True)
        job = scratch / "job.json"
        job.write_text(json.dumps({"out": str(outdir / f"{s.name}.jsonl").replace("\\", "/"),
                                   "inbox": str(inbox).replace("\\", "/"), **jopts}), encoding="utf-8")
        s.t_spawn = time.time()
        s.proc, s.log = rr.launch(game, scratch, job, below_normal=not a.normal_priority)
        s.pid = s.proc.pid
        if spare:
            (inbox / f"hold.{s.pid}").write_text("", encoding="utf-8")
        servers.append(s)
        log(f"{s.name} started ({'spare' if spare else 'active'}) pid {s.pid}")
        return s

    def feed() -> None:
        nonlocal fed
        window = max(8, 4 * n_act)
        pending = sum(1 for f in os.listdir(inbox) if f.endswith(".json"))
        while fed < total and pending < window:
            tmp = inbox / f"{ids[fed]}.tmp"
            tmp.write_text(json.dumps(runs[fed]), encoding="utf-8")
            os.replace(tmp, inbox / f"{ids[fed]}.json")
            fed += 1
            pending += 1

    def collect(s: Server) -> None:
        new, s.offset = read_lines(outdir / f"{s.name}.jsonl", s.offset)
        for r in new:
            s.lines.append(r)
            rid = r.get("_id")
            if rid is None:
                where = f" on spec {r['_fatal']}" if r.get("_fatal") is not None else ""
                log(f"{s.name} server error{where}: {str(r.get('error'))[:300]}")
            elif rid not in results:
                results[rid] = r
                arrival.append(r)

    def check_ready(s: Server) -> None:
        if s.t_ready is None:
            try:
                s.t_ready = int((inbox / "hb" / f"{s.pid}.ready").read_text()) / 1000
                log(f"{s.name} ready after {s.t_ready - s.t_spawn:.1f}s")
            except (FileNotFoundError, ValueError):
                pass

    def handle_exit(s: Server, rc: int) -> None:
        nonlocal restarts, requeued
        s.t_exit, s.rc = time.time(), rc
        collect(s)
        if s.log:
            s.log.close()
        for hb in ((inbox / f"hold.{s.pid}"), (inbox / "hb" / str(s.pid))):
            hb.unlink(missing_ok=True)
        stopping = (inbox / "STOP").exists()
        lost = []
        for taken in sorted(inbox.glob(f"*.json.taken.{s.pid}")):
            rid = taken.name.split(".json.taken.")[0]
            if rid in results:  # appended before the process died; only the delete was lost
                taken.unlink()
                continue
            attempts[rid] += 1
            if attempts[rid] == 1:
                os.replace(taken, inbox / f"{rid}.json")  # keeps its age, so it is claimed first
                requeued += 1
                lost.append(f"{rid} re-queued")
            else:
                spec = runs[int(rid)]
                results[rid] = {"seed": spec.get("seed"), "steps": [], "_id": rid,
                                "error": "stalled: the game failed this fight twice (no progress, crash or run-start exception)"}
                arrival.append(results[rid])
                taken.unlink()
                lost.append(f"{rid} recorded as stalled")
        if not stopping:
            restarts += 1
            log(f"{s.name} died (exit {rc}{', ' + s.killed if s.killed else ''}); {'; '.join(lost) or 'no spec in flight'}")
        elif rc != 0:
            log(f"{s.name} ended after STOP with exit {rc}{', ' + s.killed if s.killed else ''}")

    def kill(s: Server, why: str) -> None:
        s.killed = why
        subprocess.run(["taskkill", "/PID", str(s.pid), "/T", "/F"], capture_output=True, check=False)

    rc_final = 0
    last_net = 0.0
    try:
        feed()
        for _ in range(n_act):
            spawn(spare=False)
        while len(results) < total:
            now = time.time()
            if now - t_start > a.timeout:
                log("run timeout")
                rc_final = 1
                break
            feed()
            live = [s for s in servers if s.alive]
            for s in live:
                collect(s)
                check_ready(s)
            if now - last_net >= a.net_interval and live:  # one netstat call for every server
                last_net = now
                for pid, conns in rr.connections_by_pid([s.pid for s in live]).items():
                    seen_net |= conns
                    for s in live:
                        if s.pid == pid:
                            s.connections |= conns
            for s in live:
                code = s.proc.poll()
                if code is not None:
                    handle_exit(s, code)
                    continue
                if s.t_ready is None and now - s.t_spawn > a.boot_timeout:
                    kill(s, "boot timeout")
                elif s.t_ready is not None:
                    try:
                        age = now - (inbox / "hb" / str(s.pid)).stat().st_mtime
                    except FileNotFoundError:
                        age = 0
                    if age > 2 * stall_s + 5:  # the driver's own watchdog should have fired by now
                        kill(s, f"no heartbeat for {age:.0f}s")
            # keep N active servers: promote the spare first, else boot a replacement
            live = [s for s in servers if s.alive]
            if restarts > max_restarts:
                log(f"giving up: {restarts} restarts")
                rc_final = 1
                break
            actives = [s for s in live if not s.spare]
            while len(actives) < n_act:
                spare = next((s for s in live if s.spare), None)
                if spare is not None:
                    spare.spare = False
                    (inbox / f"hold.{spare.pid}").unlink(missing_ok=True)
                    log(f"{spare.name} promoted from spare")
                    live = [s for s in servers if s.alive]
                    actives = [s for s in live if not s.spare]
                else:
                    actives.append(spawn(spare=False))
            remaining = total - len(results)
            if (a.spares and not any(s.spare for s in live) and all(s.t_ready for s in actives)
                    and remaining >= 2 * n_act + 2):
                spawn(spare=True)
            time.sleep(0.25)
    finally:
        # STOP: servers finish what they hold, see the empty inbox and exit 0; spares and stragglers are killed.
        (inbox / "STOP").write_text("", encoding="utf-8")
        for s in servers:
            if s.alive and s.spare:
                kill(s, "spare not needed")
        deadline = time.time() + 45
        for s in servers:
            if not s.alive:
                continue
            try:
                s.proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                kill(s, "did not exit after STOP")
                s.proc.wait()
            collect(s)
            handle_exit(s, s.proc.returncode)
        for pid, conns in rr.connections_by_pid([s.pid for s in servers if s.alive]).items():
            seen_net |= conns

    t_end = time.time()
    after = rr.real_profile_state()
    merged = []
    for i, rid in enumerate(ids):
        r = results.get(rid)
        merged.append(strip(r) if r else {"seed": runs[i].get("seed"), "steps": [], "error": "no result"})
    out = a.out or run_dir / "results.json"
    out.write_text(json.dumps({"results": merged}), encoding="utf-8")
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in arrival), encoding="utf-8")

    # Per-server and total timing.
    print()
    print(f"{'server':7} {'boot s':>7} {'fights':>6} {'first->last s':>13} {'wall s/fight':>12} {'game ms/fight':>13}  end")
    t_ready_min = min((s.t_ready for s in servers if s.t_ready), default=t_start)
    for s in servers:
        done = [r for r in s.lines if r.get("_id") is not None]
        boot = f"{s.t_ready - s.t_spawn:.1f}" if s.t_ready else "-"
        span = (done[-1]["_t"] / 1000 - s.t_ready) if done and s.t_ready else 0
        game_ms = [r["timing_ms"]["play"] + r["timing_ms"]["reset"] for r in done if "timing_ms" in r]
        print(f"{s.name:7} {boot:>7} {len(done):>6} {span:>13.1f} {span / len(done) if done else 0:>12.2f} "
              f"{sum(game_ms) / len(game_ms) if game_ms else 0:>13.0f}  "
              f"{'spare ' if s.spare else ''}exit {s.rc}{' (' + s.killed + ')' if s.killed else ''}")
    stalls = sum(1 for r in results.values() if str(r.get("error", "")).startswith("stalled"))
    errors = sum(1 for r in results.values() if r.get("error") and not str(r.get("error")).startswith("stalled"))
    wall, post_boot = t_end - t_start, t_end - t_ready_min
    first_boot = min((s.t_ready - s.t_spawn for s in servers if s.t_ready), default=0)
    summary = {
        "fights": total, "results": len(results), "servers": n_act, "wall_s": round(wall, 1),
        "wall_after_first_ready_s": round(post_boot, 1), "fights_per_min": round(60 * len(results) / max(1e-9, wall), 1),
        "fights_per_min_after_boot": round(60 * len(results) / max(1e-9, post_boot), 1),
        "s_per_fight_per_server": round(post_boot * n_act / max(1, len(results)), 2),
        "first_boot_s": round(first_boot, 1), "restarts": restarts, "requeued": requeued, "stalled_results": stalls,
        "errors": errors, "sts2_dll_sha256": digest, "network_connections": sorted(seen_net),
        "real_profile_touched": before != after, "run_dir": run_dir.name,
    }
    if a.verify_against:
        ref = json.loads(a.verify_against.read_text(encoding="utf-8"))["results"]
        same = same_steps = 0
        first_bad = []
        ignore = {k for k in a.verify_ignore.split(",") if k}
        for i, r in enumerate(merged):
            g = ref[i % len(ref)]
            if ignore:
                r, g = drop_keys(r, ignore), drop_keys(g, ignore)
            same_steps += r.get("steps") == g.get("steps")
            if strip(r) == strip(g):
                same += 1
            elif len(first_bad) < 5:
                keys = [k for k in set(r) | set(g) if strip(r).get(k) != strip(g).get(k)]
                first_bad.append({"spec": i, "differing_keys": sorted(keys)})
        summary["verify"] = {"results_equal": same, "steps_equal": same_steps, "of": total, "first_mismatches": first_bad}
    print(json.dumps(summary, indent=1))
    (run_dir / "summary.json").write_text(json.dumps({**summary, "events": events}, indent=1), encoding="utf-8")
    ok = rc_final == 0 and len(results) == total and not seen_net and before == after
    if a.verify_against and summary["verify"]["results_equal"] != total:
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
