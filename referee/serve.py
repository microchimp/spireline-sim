#!/usr/bin/env python3
"""Fight-server dispatcher: N long-lived game processes, each fed through its own inbox.

    python referee/serve.py --game <game copy> --specs <job.json> [--servers 4] [--repeat 1] [--out results.json]

--specs is a job file (or a bare JSON list) whose "runs" are fight specs; its idle_frames / stall_s / settle /
settle_audit / fast_tweens / sleep_usec / trace_frames / step_costs / idle_probe keys apply to every server
(fast_reset / no_preload come from --slow-reset / --preload; the speed patch sets fast_patches, silent and
still_visuals are on unless --no-patches; --job-opt KEY=JSON sets any job key, last wins). Driver job keys:
  settle        "flags" (default here and in batch.py): the game's own turn flags, held two checks one frame
                apart; "frames": the old rule, six stable frames (driver/Settle.cs).
  settle_audit  0..1 (default 0.05): share of fights, picked by seed hash, whose every flag settle is re-checked
                with the old rule at stock pacing; the result's "settle_audit" lists the steps (an object per
                disagreement).
  fast_tweens   true (default): awaited tweens finish at once (driver/Patches.cs).
  sleep_usec    0 (default): no per-frame padding (Godot's low-processor sleep, stock 6900); only safe with
                the flag settle.
  trace_frames  per-frame trace of each step's settle in the step's "frame_trace" (diagnostics).
  step_costs    frames and main-thread megacycles per step in the step's "cost".
  idle_probe    N: idle N frames after step 0 and record a quiet frame's cost in "idle_probe".
  idle_timeout_s  seconds (and 600 frames) a settle waits before the fight ends with idle_timeout (default 10).
  silent        true (default): the referee never starts a sound (driver/Patches.cs).
  still_visuals true (default): visual-only nodes do no per-frame work (driver/Patches.cs).
  profile       true: frame-cost profile per fight (driver/Lean.cs; diagnostics, set with --job-opt).
The game processes start with run_referee.DOTNET_ENV (DOTNET_TieredPGO=0) unless --no-dotnet-tuning.
--old-settle runs the old settle: settle "frames", stock tweens and stock frame pacing, no audit. job_options()
holds these defaults, so batch.py's pool gets the same ones.
Each server is one headless game on its own scratch
profile (same safety checks as run_referee.py: known sts2.dll sha256, override.cfg, APPDATA in scratch,
no network connections, real profile untouched). The dispatcher keeps one spec file waiting in each
server's <run-dir>/inbox/s<NN>; the server claims it (driver/Server.cs), plays it and appends a JSON
line to its own <run-dir>/out/s<NN>.jsonl, and the dispatcher hands it the next. One consumer per
inbox, because a claim is a rename and on Windows a rename is not exclusive: two servers renaming
the same file can both succeed (MoveFileEx opens the source, then renames through that handle; a
second open made before the first rename landed renames the file again), and the loser then answered
"bad spec file" for a fight the other played (night04: 4 of 600). Results are merged into --out (same
format as a runs-mode out file, in spec order) and results.jsonl (arrival order).

A server that dies or hangs is restarted; the spec it had claimed goes back to the front of the queue
once, and is recorded as stalled if it kills a server a second time. One hot spare (--spares) is booted
and held (hold.<pid> in its inbox) so a restart costs no boot time. At the end STOP goes into every inbox.
A server whose fight left the game's turn flags set writes a "_retire" line and exits 5 (driver/Server.cs);
it is replaced like a dead one, with nothing to re-queue.

Nodeless servers ("nodeless": true): their fallback and audit replays on the full host come back to the pool
(job key "nodeless_full": "pool", a result line with "_full") and are played as "<id>f" with "host": "full",
first by the --full-lane servers (default about 15%, at least 1) and by any server whose own queue is empty;
the pool merges both lines into what one process would have returned (driver/Nodeless.cs). --full-here
keeps the replays in each server's own process.
--verify-against <out.json> compares every result with a reference from "runs" mode, ignoring timing and
the diagnostics above (settle_audit, idle_probe, per-step cost and trace).

The dispatcher is the importable Pool class (submit / poll / close); the command line is one user of
it and batch.py, which keeps one pool for a whole batch, is the other.
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
META_KEYS = {"_id", "_pid", "_t", "_full"}  # added by the server to each result line
STALLED = "stalled: the game failed this fight twice (no progress, crash or run-start exception)"
FULL_SUFFIX = "f"  # id of a spec's full-host replay (fallback or audit of a nodeless fight): "<id>f"


def default_full_lane(servers: int) -> int:
    """Servers that prefer full-host replays in a nodeless pool. On the 240 sp-nodeless specs 5.4% of fights
    were fallbacks and 5% audits, each a full-host fight at ~4x a nodeless fight's CPU: ~15% of the servers' time."""
    return 0 if servers < 2 else max(1, round(0.15 * servers))


def first_diff(a: dict, b: dict) -> str | None:
    """Nodeless.FirstDiff: the first result key, timing, cost and server keys aside, where two results differ."""
    for k in list(a) + [k for k in b if k not in a]:
        if k in TIMING_KEYS or k in META_KEYS:
            continue
        if a.get(k) != b.get(k):
            return k
    return None


class Server:
    def __init__(self, sid: int, spare: bool, inbox: Path):
        self.sid, self.spare = sid, spare
        self.name = f"s{sid:02d}"
        self.inbox = inbox / self.name  # only this server claims from it
        self.proc: subprocess.Popen | None = None
        self.pid = 0
        self.t_spawn = 0.0
        self.t_ready: float | None = None  # unix seconds, stamped by the driver
        self.t_exit: float | None = None
        self.rc: int | None = None
        self.offset = 0
        self.lines: list[dict] = []
        self.killed = ""
        self.retiring = ""  # the server's own "_retire" line: a fight left the game state stale (driver/Server.cs)
        self.lane = "main"  # "full": takes the pool's full-host replays before anything else
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


DIAG_KEYS = {"settle_audit", "idle_probe", "checksum_cost"}  # driver diagnostics: kept in the merged output, never verified
# per-step instrumentation: step_costs, trace_frames (Settle.cs), trace mode (Trace.cs), the game's checksums (ChecksumTap.cs)
DIAG_STEP_KEYS = {"cost", "frame_trace", "trace", "hidden", "checksums", "net"}


def comparable(r: dict) -> dict:
    """What --verify-against compares: strip() without the driver's diagnostics (strip() itself keeps them,
    because the merged output is read for them, e.g. batch.py's settle-audit summary)."""
    out = {k: v for k, v in strip(r).items() if k not in DIAG_KEYS}
    if isinstance(out.get("steps"), list):
        out["steps"] = [{k: v for k, v in st.items() if k not in DIAG_STEP_KEYS} if isinstance(st, dict) else st
                        for st in out["steps"]]
    return out


# Driver job keys a job file may set for every server (the module docstring has what they do).
JOB_KEYS = ("idle_frames", "stall_s", "settle", "settle_audit", "fast_tweens", "sleep_usec",
            "trace_frames", "step_costs", "idle_probe", "idle_timeout_s", "pick_timing",
            # the nodeless host (driver/Nodeless.cs), opt-in: off unless a job file sets them
            "nodeless", "nodeless_on_error", "nodeless_audit", "health", "error_log", "frame_sleep_usec")


def job_options(job_in, *, slow_reset: bool = False, preload: bool = False, no_patches: bool = False,
                stall_s: int | None = None, idle_frames: float | None = None, old_settle: bool = False,
                settle_audit: float | None = None, job_opt=()) -> dict:
    """The job keys every server gets: a job file's JOB_KEYS, the speed defaults (SPEED_PLAN steps 3-4; keys
    in the job file win over them), then the command-line choices, and last every job_opt "KEY=JSON" (so
    one can turn a default off). serve.py's command line and batch.py's pool both build their servers'
    options here, so the two cannot drift apart.

    The speed patch sets (driver/Patches.cs) are on unless no_patches: fast_patches (SPEED_PLAN step 2),
    silent (no sound is started) and still_visuals (visual-only nodes do no per-frame work)."""
    jopts = {k: v for k, v in (job_in.items() if isinstance(job_in, dict) else []) if k in JOB_KEYS}
    if old_settle:
        for k in ("settle_audit", "sleep_usec"):
            jopts.pop(k, None)
        jopts.update(settle="frames", fast_tweens=False)
    else:
        for k, v in (("settle", "flags"), ("fast_tweens", True), ("sleep_usec", 0), ("settle_audit", 0.05)):
            jopts.setdefault(k, v)
    if settle_audit is not None:
        jopts["settle_audit"] = settle_audit
    jopts["fast_reset"] = not slow_reset
    jopts["no_preload"] = not preload
    for key in ("fast_patches", "silent", "still_visuals"):
        jopts[key] = not no_patches
    if stall_s is not None:
        jopts["stall_s"] = stall_s
    if idle_frames is not None:
        jopts["idle_frames"] = idle_frames
    for kv in job_opt:
        k, _, v = kv.partition("=")
        jopts[k] = json.loads(v)
    return jopts


def _taskkill(s: Server) -> None:
    subprocess.run(["taskkill", "/PID", str(s.pid), "/T", "/F"], capture_output=True, check=False)


class Pool:
    """The dispatcher: N long-lived game servers, each fed through its own inbox, as long as the caller wants.

    submit(specs) queues fights and returns their ids (in submission order, so id order is spec order);
    poll() runs one dispatcher tick and returns the fights finished since the last call as (id, result
    line) pairs, in arrival order; close() writes STOP, lets the servers finish what they hold and
    returns the last results. The caller sleeps between polls and decides when it is done.

    The safety checks live here, so every user of the pool gets them: H1 the known sts2.dll (at
    construction), H2 one netstat call per net_interval for every live server (seen_net; the same tick lists
    any WerFault.exe about a game process in seen_wer, a crash report attempt even without a connection;
    the games start with an error mode that keeps Windows Error Reporting away, run_referee.launch), H3 the real
    profile compared between start() and close() (profile_touched). The rules: a server that dies or
    hangs is replaced (the hot spare first); the spec it had claimed goes back to the front of the queue
    once and is recorded as stalled if it kills a second server (specs it had not claimed go back too,
    not counted); more than max_restarts restarts sets `failed`. Each active server has `depth` specs
    waiting in its inbox, so it starts the next fight as soon as one ends.
    launch / connections / profile_state / check_build / kill are injectable for tests.
    """

    def __init__(self, game: Path, run_dir: Path, *, servers: int = 4, spares: int = 1, jopts: dict | None = None,
                 boot_timeout: float = 180, max_restarts: int | None = None, net_interval: float = 1.0,
                 hold_timeout: float = 60, dotnet_tuning: bool = True, full_lane: int | None = None,
                 route_full: bool | None = None,
                 below_normal: bool = True, depth: int = 1, echo=print, launch=rr.launch, connections=rr.connections_by_pid,
                 profile_state=rr.real_profile_state, check_build=rr.check_build, kill=_taskkill,
                 wer_reports=rr.wer_reports_by_pid):
        self.game = game
        self.digest = check_build(game)
        self.n = max(1, servers)
        self.spares = spares
        self.jopts = dict(jopts if jopts is not None else job_options(None))
        # Full-host replays of nodeless fights (driver/Nodeless.cs "nodeless_full": "pool"): a fallback or an
        # audit comes back as a line with "_full", and the spec is played again with "host": "full" by a server
        # of the full lane (any server takes it when its own queue is empty), so nodeless servers never run
        # the full host's scenes, and a server whose fight left stale state can retire without losing it.
        nodeless = bool(self.jopts.get("nodeless"))
        self.route_full = nodeless if route_full is None else (route_full and nodeless)
        if self.route_full:
            self.jopts["nodeless_full"] = "pool"
        else:
            self.jopts.pop("nodeless_full", None)
        lane = default_full_lane(self.n) if full_lane is None else full_lane
        self.full_lane = max(0, min(lane, self.n - 1)) if self.route_full else 0
        self.fqueue: collections.deque[str] = collections.deque()  # full-host replay ids, not yet in an inbox
        self.fjobs: dict[str, dict] = {}  # replay id -> {"rid", "why", "reason", "line"} until its result is merged
        self.full_replays: collections.Counter = collections.Counter()  # by why: fallback, audit
        self.audit_diffs = 0
        self.retired = 0
        self.stall_s = int(self.jopts.get("stall_s", 20))
        # The driver rewrites its status file every frame (driver/Watchdog.cs), so an old file means no frame
        # has ended. That alone is not a hang: a starved server can take minutes over one frame, and the
        # driver's own watchdog tells a blocked main thread from a starved one. This is the backstop for a
        # driver whose watchdog cannot run.
        self.hb_limit = 10 * self.stall_s
        self.boot_timeout = boot_timeout
        self.hold_timeout = hold_timeout
        self.max_restarts = max_restarts if max_restarts is not None else 3 * self.n + 3
        self.net_interval = net_interval
        self.below_normal = below_normal
        self.dotnet_tuning = dotnet_tuning  # run_referee.DOTNET_ENV in the games' environment
        self.depth = max(1, depth)
        self.window = 2 * self.n * self.depth  # unclaimed specs a caller feeding the pool keeps ready
        self._echo, self._launch, self._connections, self._wer_reports = echo, launch, connections, wer_reports
        self._profile_state, self._kill_fn = profile_state, kill
        self.run_dir = Path(run_dir).resolve()
        self.inbox, self.outdir = self.run_dir / "inbox", self.run_dir / "out"
        for d in (self.inbox, self.outdir):
            d.mkdir(parents=True, exist_ok=True)
        if any(self.inbox.iterdir()) or any(self.outdir.iterdir()):
            # A used run folder would mix in: its s<NN>.jsonl lines are read from offset 0 and their
            # ids ("000000"...) are this run's ids too, and a leftover STOP ends every server.
            raise ValueError(f"run folder {self.run_dir.name} was used before; give a new one")
        self.servers: list[Server] = []
        self.specs: dict[str, dict] = {}  # id -> spec, in submission order
        self.queue: collections.deque[str] = collections.deque()  # submitted, not yet in a server's inbox
        self.results: dict[str, dict] = {}  # id -> result line
        self.arrival: list[dict] = []
        self.attempts: collections.Counter = collections.Counter()
        self.restarts = self.requeued = 0
        self.seen_net: set[str] = set()
        self.seen_wer: set[str] = set()  # WerFault.exe about a game process (a crash report attempt)
        self.events: list[str] = []
        self.more = 0  # specs the caller still means to submit (only decides whether a hot spare pays)
        self.failed = ""  # why the pool gave up; empty while it is healthy
        self.t_start = time.time()
        self.t_end: float | None = None
        self.before = self.after = None
        self.started = self.closed = self.stopping = False
        self._next_id = self._next_sid = 0
        self._last_net = 0.0
        self._new: list[tuple[str, dict]] = []

    # -- caller API ---------------------------------------------------------------------------------
    def submit(self, specs: list) -> list[str]:
        ids = []
        for spec in specs:
            rid = f"{self._next_id:06d}"
            self._next_id += 1
            self.specs[rid] = spec
            self.queue.append(rid)
            ids.append(rid)
        return ids

    @property
    def unclaimed(self) -> int:
        """Specs no server has taken yet: queued here (full-host replays too) plus waiting in the servers' inboxes."""
        return len(self.queue) + len(self.fqueue) + sum(len(self._pending(s)) for s in self.servers if s.alive)

    @property
    def outstanding(self) -> int:
        """Submitted specs without a result yet."""
        return len(self.specs) - len(self.results)

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        self.before = self._profile_state()
        self.t_start = time.time()
        for _ in range(self.n):
            self._spawn(spare=False)
        self._feed()

    def poll(self) -> list[tuple[str, dict]]:
        """One dispatcher tick: feed the inboxes, collect results, check the network, replace dead or
        hung servers, keep N active plus the spare. Returns what finished since the last call."""
        if not self.started:
            self.start()
        if self.closed or self.failed:
            return self._drain()
        now = time.time()
        self._feed()
        live = [s for s in self.servers if s.alive]
        for s in live:
            self._collect(s)
            self._check_ready(s)
        if now - self._last_net >= self.net_interval and live:  # one netstat call for every server
            self._last_net = now
            for pid, conns in self._connections([s.pid for s in live]).items():
                self.seen_net |= conns
                for s in live:
                    if s.pid == pid:
                        s.connections |= conns
            self._check_wer(live)
        for s in live:
            code = s.proc.poll()
            if code is not None:
                self._handle_exit(s, code)
                continue
            if s.t_ready is None and now - s.t_spawn > self.boot_timeout:
                self._kill(s, "boot timeout")
            elif s.t_ready is not None:
                try:
                    age = now - (s.inbox / "hb" / str(s.pid)).stat().st_mtime
                except FileNotFoundError:
                    age = 0
                if age > self.hb_limit:
                    self._kill(s, f"no heartbeat for {age:.0f}s")
        # keep N active servers: promote the spare first, else boot a replacement
        live = [s for s in self.servers if s.alive]
        if self.restarts > self.max_restarts:
            self.failed = f"giving up: {self.restarts} restarts"
            self.log(self.failed)
            return self._drain()
        actives = [s for s in live if not s.spare]
        while len(actives) < self.n:
            spare = next((s for s in live if s.spare), None)
            if spare is not None:
                spare.spare = False
                (spare.inbox / f"hold.{spare.pid}").unlink(missing_ok=True)
                self.log(f"{spare.name} promoted from spare")
                live = [s for s in self.servers if s.alive]
                actives = [s for s in live if not s.spare]
            else:
                actives.append(self._spawn(spare=False))
        if (self.spares and not any(s.spare for s in live) and all(s.t_ready for s in actives)
                and self.outstanding + self.more >= 2 * self.n + 2):
            self._spawn(spare=True)
        self._feed()
        return self._drain()

    def close(self, abandon: bool = False) -> list[tuple[str, dict]]:
        """STOP: servers finish what they hold, see their empty inbox and exit 0; spares and stragglers
        are killed. abandon=True first withdraws the specs no server has claimed (renamed to
        *.withdrawn), so an aborted run does not play them. Returns what finished since the last poll.

        Withdrawing renames files in a server's inbox, and a rename is not exclusive on Windows (two renames
        of one file can both succeed), so it happens only once the server is not claiming: its status file
        reads "hold", stamped after the hold file was written (driver/Server.cs checks the hold file before
        it writes its status and before it claims), or it is dead. A server that does not hold within
        hold_timeout (it is still playing a fight) is killed first; this is the abandon path."""
        if self.closed:
            return self._drain()
        self.closed = self.stopping = True
        if not self.started:
            self.t_end = time.time()
            return self._drain()
        live = [s for s in self.servers if s.alive]
        if abandon:
            self.queue.clear()
            self.fqueue.clear()
            t_hold_ms = int(time.time() * 1000)
            for s in live:  # hold first, so no server is claiming while its specs are withdrawn
                (s.inbox / f"hold.{s.pid}").write_text("", encoding="utf-8")
            waiting = [s for s in live if self._pending(s)]  # an empty inbox has nothing to withdraw
            deadline = time.time() + self.hold_timeout
            while waiting:
                waiting = [s for s in waiting if s.proc.poll() is None and not self._holds(s, t_hold_ms)]
                if not waiting:
                    break
                if time.time() > deadline:
                    for s in waiting:
                        self._kill(s, "did not hold for the abandon")
                        s.proc.wait()
                    break
                time.sleep(0.05)
            for s in live:
                n = 0
                for f in self._pending(s):
                    try:
                        os.replace(f, f.with_name(f.name + ".withdrawn"))
                        n += 1
                    except OSError:
                        pass  # claimed by a server that died meanwhile; its exit is handled below
                if n:
                    self.log(f"{s.name} {'killed' if s.killed else 'holds'}, {n} specs withdrawn")
        for s in live:
            (s.inbox / "STOP").write_text("", encoding="utf-8")
        for s in self.servers:
            if s.alive and s.spare:
                self._kill(s, "spare not needed")
        deadline = time.time() + 45
        for s in self.servers:
            if not s.alive:
                continue
            try:
                s.proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                self._kill(s, "did not exit after STOP")
                s.proc.wait()
            self._collect(s)
            self._handle_exit(s, s.proc.returncode)
        for fid in list(self.fjobs):  # full-host replays nobody played (an abandoned or early close)
            self._unreplayed(fid, "not replayed: the pool closed first")
        # Every server, exited ones too: a child the game started (crashpad_handler) can outlive it.
        for pid, conns in self._connections([s.pid for s in self.servers if s.pid]).items():
            self.seen_net |= conns
            for s in self.servers:
                if s.pid == pid:
                    s.connections |= conns
        self._check_wer(self.servers)
        self.t_end = time.time()
        self.after = self._profile_state()
        return self._drain()

    @property
    def profile_touched(self) -> bool:
        return self.started and self.after is not None and self.before != self.after

    def merged(self, ids: list[str] | None = None) -> list[dict]:
        """Results in spec order (runs-mode out format: timing and server keys stripped)."""
        out = []
        for rid in (self.specs if ids is None else ids):
            r = self.results.get(rid)
            out.append(strip(r) if r else {"seed": self.specs[rid].get("seed"), "steps": [], "error": "no result"})
        return out

    def table(self) -> list[str]:
        """Per-server timing, one line each."""
        lines = [f"{'server':7} {'boot s':>7} {'fights':>6} {'first->last s':>13} {'wall s/fight':>12} {'game ms/fight':>13}  end"]
        for s in self.servers:
            done = [r for r in s.lines if r.get("_id") is not None]
            boot = f"{s.t_ready - s.t_spawn:.1f}" if s.t_ready else "-"
            span = (done[-1]["_t"] / 1000 - s.t_ready) if done and s.t_ready else 0
            game_ms = [r["timing_ms"]["play"] + r["timing_ms"]["reset"] for r in done if "timing_ms" in r]
            lines.append(f"{s.name:7} {boot:>7} {len(done):>6} {span:>13.1f} {span / len(done) if done else 0:>12.2f} "
                         f"{sum(game_ms) / len(game_ms) if game_ms else 0:>13.0f}  "
                         f"{'spare ' if s.spare else ''}{'full lane ' if s.lane == 'full' else ''}exit {s.rc}"
                         f"{' (' + s.killed + ')' if s.killed else ''}{' (retired: ' + s.retiring + ')' if s.retiring else ''}")
        return lines

    def summary(self) -> dict:
        t_end = self.t_end or time.time()
        t_ready_min = min((s.t_ready for s in self.servers if s.t_ready), default=self.t_start)
        stalls = sum(1 for r in self.results.values() if str(r.get("error", "")).startswith("stalled"))
        errors = sum(1 for r in self.results.values() if r.get("error") and not str(r.get("error")).startswith("stalled"))
        n_res = len(self.results)
        wall, post_boot = t_end - self.t_start, t_end - t_ready_min
        first_boot = min((s.t_ready - s.t_spawn for s in self.servers if s.t_ready), default=0)
        return {
            "fights": len(self.specs), "results": n_res, "servers": self.n, "wall_s": round(wall, 1),
            "wall_after_first_ready_s": round(post_boot, 1), "fights_per_min": round(60 * n_res / max(1e-9, wall), 1),
            "fights_per_min_after_boot": round(60 * n_res / max(1e-9, post_boot), 1),
            "s_per_fight_per_server": round(post_boot * self.n / max(1, n_res), 2),
            "first_boot_s": round(first_boot, 1), "restarts": self.restarts, "retired": self.retired,
            "requeued": self.requeued, "stalled_results": stalls, "errors": errors,
            **({"full_lane": self.full_lane, "full_replays": dict(self.full_replays), "audit_diffs": self.audit_diffs}
               if self.route_full else {}),
            "sts2_dll_sha256": self.digest,
            "network_connections": sorted(self.seen_net), "wer_reports": sorted(self.seen_wer),
            "real_profile_touched": self.profile_touched, "run_dir": self.run_dir.name,
        }

    def write_run_files(self, summary: dict) -> None:
        """results.jsonl (arrival order) and summary.json (with the event log) in the run folder."""
        (self.run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.arrival), encoding="utf-8")
        (self.run_dir / "summary.json").write_text(json.dumps({**summary, "events": self.events}, indent=1), encoding="utf-8")

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t_start:7.1f}s] {msg}"
        self.events.append(line)
        if self._echo:
            self._echo(line, flush=True)

    # -- dispatcher internals -----------------------------------------------------------------------
    def _drain(self) -> list[tuple[str, dict]]:
        new, self._new = self._new, []
        return new

    def _finish(self, rid: str, r: dict) -> None:
        self.results[rid] = r
        self.arrival.append(r)
        self._new.append((rid, r))

    @staticmethod
    def _holds(s: Server, since_ms: int) -> bool:
        """The server's status file says it holds (does not claim), written at or after since_ms (unix ms)."""
        try:
            words = (s.inbox / "hb" / str(s.pid)).read_text(encoding="utf-8").split()
        except OSError:
            return False
        try:
            return len(words) >= 2 and words[0] == "hold" and int(words[1]) >= since_ms
        except ValueError:
            return False

    @staticmethod
    def _pending(s: Server) -> list[Path]:
        """Spec files waiting in a server's inbox, oldest id first."""
        try:
            return sorted(s.inbox / f for f in os.listdir(s.inbox) if f.endswith(".json"))
        except FileNotFoundError:
            return []

    def _feed(self) -> None:
        """Top every active server's inbox up to `depth` waiting specs, from the front of a queue: full-lane
        servers first, from the full-host replays, then (theirs empty) from the main queue; the others the
        other way round, so no work waits while a server idles."""
        if self.stopping or not (self.queue or self.fqueue):
            return
        self._assign_lanes()
        actives = [s for s in self.servers if s.alive and not s.spare]
        for s in sorted(actives, key=lambda s: s.lane != "full"):
            for _ in range(self.depth - len(self._pending(s))):
                order = (self.fqueue, self.queue) if s.lane == "full" or not self.full_lane else (self.queue, self.fqueue)
                q = next((q for q in order if q), None)
                if q is None:
                    return
                rid = q.popleft()
                tmp = s.inbox / f"{rid}.tmp"
                tmp.write_text(json.dumps(self._spec_for(rid)), encoding="utf-8")
                os.replace(tmp, s.inbox / f"{rid}.json")

    def _assign_lanes(self) -> None:
        """Keep `full_lane` active servers in the full lane: those already in it, else the oldest others (a lane
        is only the dispatcher's preference; every server runs the same job)."""
        actives = [s for s in self.servers if s.alive and not s.spare]
        n_full = sum(s.lane == "full" for s in actives)
        for s in actives:
            if n_full >= self.full_lane:
                break
            if s.lane != "full":
                s.lane = "full"
                n_full += 1
                self.log(f"{s.name} takes the full-host lane")

    def _spec_for(self, rid: str) -> dict:
        """The spec file for an id: a submitted spec, or its full-host replay ("host": "full", driver/Nodeless.cs)."""
        job = self.fjobs.get(rid)
        return self.specs[rid] if job is None else {**self.specs[job["rid"]], "host": "full"}

    def _route_full(self, rid: str, line: dict) -> None:
        """A nodeless server left a full-host replay to the pool ("_full"): queue it, ahead of the lane's other work."""
        why = (line.get("_full") or {}).get("why", "fallback")
        fid = rid + FULL_SUFFIX
        self.fjobs[fid] = {"rid": rid, "why": why, "reason": (line.get("_full") or {}).get("reason"), "line": line}
        self.full_replays[why] += 1
        self.fqueue.append(fid)

    def _merge_full(self, fid: str, full: dict) -> None:
        """A full-host replay is back: the result Nodeless.Play would have returned in one process. Fallback: the
        full host's result with cost.fallback and cost.nodeless_attempt. Audit: the nodeless result, with
        cost.audit saying whether the full host agrees (Nodeless.Audit)."""
        job = self.fjobs.pop(fid)
        rid, first = job["rid"], job["line"]
        if job["why"] == "audit":
            res = {k: v for k, v in first.items() if k != "_full"}
            diff = first_diff(res, full)
            if diff is not None:
                self.audit_diffs += 1
                self.log(f"nodeless audit: {res.get('seed')} differs from the full host at '{diff}'")
            fc = full.get("cost") or {}
            res.setdefault("cost", {})["audit"] = {"same": diff is None, "first_diff": diff, "full_error": full.get("error"),
                                                   "full_cpu_ms": fc.get("cpu_ms"), "full_errors": fc.get("errors")}
        else:
            res = {**full, "_id": rid}
            res.setdefault("seed", self.specs[rid].get("seed"))  # a "bad spec file" line has none
            cost = res.setdefault("cost", {})
            cost["fallback"] = job["reason"]
            wasted = {k: v for k, v in (first.get("cost") or {}).items() if k != "health"}
            if wasted:
                cost["nodeless_attempt"] = wasted
        self._finish(rid, res)

    def _unreplayed(self, fid: str, why: str) -> None:
        """A full-host replay that will not be played (stalled twice, or the pool closed): an audit keeps its nodeless
        result and says so; a fallback has no valid result, so it is an error."""
        job = self.fjobs.pop(fid)
        rid, first = job["rid"], job["line"]
        if rid in self.results:
            return
        if job["why"] == "audit":
            res = {k: v for k, v in first.items() if k != "_full"}
            res.setdefault("cost", {})["audit"] = {"same": None, "first_diff": None, "full_error": why}
        else:
            res = {"seed": first.get("seed") or self.specs[rid].get("seed"), "steps": [], "_id": rid,
                   "error": why if why == STALLED else f"nodeless fallback ({job['reason']}) {why}"}
        self._finish(rid, res)

    def _spawn(self, spare: bool) -> Server:
        s = Server(self._next_sid, spare, self.inbox)
        self._next_sid += 1
        s.inbox.mkdir(parents=True, exist_ok=True)
        scratch = self.run_dir / "scratch" / s.name
        (scratch / "appdata").mkdir(parents=True, exist_ok=True)
        (scratch / "localappdata").mkdir(parents=True, exist_ok=True)
        rr.write_settings(scratch, boot_opts=True)
        job = scratch / "job.json"
        job.write_text(json.dumps({"out": str(self.outdir / f"{s.name}.jsonl").replace("\\", "/"),
                                   "inbox": str(s.inbox).replace("\\", "/"), **self.jopts}), encoding="utf-8")
        s.t_spawn = time.time()
        s.proc, s.log = self._launch(self.game, scratch, job, below_normal=self.below_normal,
                                     dotnet_tuning=self.dotnet_tuning)
        s.pid = s.proc.pid
        if spare:  # never fed while it holds, so its inbox stays empty
            (s.inbox / f"hold.{s.pid}").write_text("", encoding="utf-8")
        self.servers.append(s)
        self.log(f"{s.name} started ({'spare' if spare else 'active'}) pid {s.pid}")
        return s

    def _collect(self, s: Server) -> None:
        new, s.offset = read_lines(self.outdir / f"{s.name}.jsonl", s.offset)
        for r in new:
            s.lines.append(r)
            rid = r.get("_id")
            if rid is None:
                if r.get("_retire"):
                    s.retiring = str(r["_retire"])
                    self.log(f"{s.name} retires after spec {r.get('_after')}: game state left stale ({s.retiring})")
                    continue
                where = f" on spec {r['_fatal']}" if r.get("_fatal") is not None else ""
                self.log(f"{s.name} server error{where}: {str(r.get('error'))[:300]}")
            elif rid in self.fjobs:
                self._merge_full(rid, r)
            elif rid not in self.results and rid in self.specs:
                if "seed" not in r:  # the driver's "bad spec file" line cannot know it; the dispatcher does
                    r["seed"] = self.specs[rid].get("seed")
                if r.get("_full"):
                    self._route_full(rid, r)
                    if self.stopping:  # STOP is out: no server takes new work any more
                        self.fqueue.remove(rid + FULL_SUFFIX)
                        self._unreplayed(rid + FULL_SUFFIX, "not replayed: the pool was closing")
                else:
                    self._finish(rid, r)

    def _check_ready(self, s: Server) -> None:
        if s.t_ready is None:
            try:
                s.t_ready = int((s.inbox / "hb" / f"{s.pid}.ready").read_text()) / 1000
                self.log(f"{s.name} ready after {s.t_ready - s.t_spawn:.1f}s")
            except (FileNotFoundError, ValueError):
                pass

    def _check_wer(self, servers) -> None:
        """Log (once each) any WerFault.exe about these servers' game trees: Windows Error Reporting started for
        a crash. The error mode the games start with should make this impossible; this is the check that says so."""
        for reports in self._wer_reports([s.pid for s in servers if s.pid]).values():
            for line in sorted(reports - self.seen_wer):
                self.log(f"WER: {line}: Windows Error Reporting started for a game process")
            self.seen_wer |= reports

    def _handle_exit(self, s: Server, rc: int) -> None:
        s.t_exit, s.rc = time.time(), rc
        self._check_wer([s])
        self._collect(s)
        if s.log:
            s.log.close()
        for hb in ((s.inbox / f"hold.{s.pid}"), (s.inbox / "hb" / str(s.pid))):
            hb.unlink(missing_ok=True)
        stopping = self.stopping
        lost, back = [], []
        for taken in sorted(s.inbox.glob(f"*.json.taken.{s.pid}")):
            rid = taken.name.split(".json.taken.")[0]
            replay = rid.endswith(FULL_SUFFIX)
            if (rid in self.results) or (replay and rid not in self.fjobs):
                taken.unlink()  # appended (or merged) before the process died; only the delete was lost
                continue
            self.attempts[rid] += 1
            if self.attempts[rid] == 1:
                taken.unlink()
                back.append(rid)  # to the front of the queue, so it is played next
                self.requeued += 1
                lost.append(f"{rid} re-queued")
            else:
                if replay:
                    self._unreplayed(rid, STALLED)
                else:
                    self._finish(rid, {"seed": self.specs.get(rid, {}).get("seed"), "steps": [], "_id": rid, "error": STALLED})
                taken.unlink()
                lost.append(f"{rid} recorded as stalled")
        for f in self._pending(s):  # handed to it but never claimed: back too, not an attempt
            f.unlink()
            back.append(f.name[: -len(".json")])
        self.queue.extendleft(reversed([r for r in back if not r.endswith(FULL_SUFFIX)]))
        self.fqueue.extendleft(reversed([r for r in back if r.endswith(FULL_SUFFIX)]))
        if s.retiring and not stopping:
            # Planned: the driver wrote its last result, then exited for a fresh process (driver/Server.cs).
            # Counted as a restart too, so a fight that poisons every server still ends the run.
            self.retired += 1
            self.restarts += 1
            self.log(f"{s.name} retired (exit {rc}); {'; '.join(lost) or 'no spec in flight'}")
        elif not stopping:
            self.restarts += 1
            self.log(f"{s.name} died (exit {rc}{', ' + s.killed if s.killed else ''}); {'; '.join(lost) or 'no spec in flight'}")
        elif rc != 0:
            self.log(f"{s.name} ended after STOP with exit {rc}{', ' + s.killed if s.killed else ''}")

    def _kill(self, s: Server, why: str) -> None:
        s.killed = why
        self._kill_fn(s)


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
    ap.add_argument("--stall-s", type=int, default=None, help="driver watchdog: seconds (and 600 frames) without progress (default 20)")
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
    ap.add_argument("--no-patches", action="store_true",
                    help="play without the speed patches (driver/Patches.cs: fast_patches, silent, still_visuals)")
    ap.add_argument("--no-dotnet-tuning", action="store_true", help="start the games without run_referee.DOTNET_ENV")
    ap.add_argument("--job-opt", action="append", default=[], metavar="KEY=JSON",
                    help="extra job key for every server, e.g. profile=true (driver/Lean.cs); repeatable")
    ap.add_argument("--verify-against", type=Path, default=None, help="runs-mode out file to compare every result with")
    ap.add_argument("--settle-audit", type=float, default=None,
                    help="share of fights whose flag settles are re-checked against the old six-frame rule (default 0.05)")
    ap.add_argument("--old-settle", action="store_true",
                    help="six-stable-frame settle, awaited tweens and stock frame pacing (before SPEED_PLAN steps 3-4)")
    ap.add_argument("--verify-ignore", default="", help="comma-separated keys dropped at any depth before --verify-against "
                    "compares (fields the reference predates, e.g. move,intents)")
    ap.add_argument("--full-lane", type=int, default=None, help="nodeless pool: servers that take the full-host replays "
                    "(fallbacks, audits) first (default about 15%% of --servers, at least 1)")
    ap.add_argument("--full-here", action="store_true", help="nodeless pool: each server replays its own fallbacks and "
                    "audits on the full host in its own process (before the full-host lane)")
    a = ap.parse_args()

    game = a.game.resolve()
    job_in = json.loads(a.specs.read_text(encoding="utf-8"))
    runs = (job_in["runs"] if isinstance(job_in, dict) else job_in) * a.repeat
    total = len(runs)
    jopts = job_options(job_in, slow_reset=a.slow_reset, preload=a.preload, no_patches=a.no_patches,
                        stall_s=a.stall_s, idle_frames=a.idle_frames, old_settle=a.old_settle,
                        settle_audit=a.settle_audit, job_opt=a.job_opt)
    run_dir = a.run_dir or game.parent / "serve" / time.strftime("%Y%m%d-%H%M%S")
    pool = Pool(game, run_dir, servers=a.servers, spares=a.spares, jopts=jopts, boot_timeout=a.boot_timeout,
                max_restarts=a.max_restarts, net_interval=a.net_interval, below_normal=not a.normal_priority,
                dotnet_tuning=not a.no_dotnet_tuning, full_lane=a.full_lane, route_full=not a.full_here)
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

    ids = pool.submit(runs)
    rc_final = 0
    try:
        pool.start()
        while len(pool.results) < total:
            if time.time() - pool.t_start > a.timeout:
                pool.log("run timeout")
                rc_final = 1
                break
            pool.poll()
            if pool.failed:
                rc_final = 1
                break
            time.sleep(0.25)
    finally:
        pool.close()

    merged = pool.merged(ids)
    out = a.out or pool.run_dir / "results.json"
    out.write_text(json.dumps({"results": merged}), encoding="utf-8")
    print()
    print("\n".join(pool.table()))
    summary = pool.summary()
    if a.verify_against:
        ref = json.loads(a.verify_against.read_text(encoding="utf-8"))["results"]
        same = same_steps = 0
        first_bad = []
        ignore = {k for k in a.verify_ignore.split(",") if k}
        for i, r in enumerate(merged):
            g = ref[i % len(ref)]
            if ignore:
                r, g = drop_keys(r, ignore), drop_keys(g, ignore)
            cr, cg = comparable(r), comparable(g)
            same_steps += cr.get("steps") == cg.get("steps")
            if cr == cg:
                same += 1
            elif len(first_bad) < 5:
                keys = [k for k in set(cr) | set(cg) if cr.get(k) != cg.get(k)]
                first_bad.append({"spec": i, "differing_keys": sorted(keys)})
        summary["verify"] = {"results_equal": same, "steps_equal": same_steps, "of": total, "first_mismatches": first_bad}
    print(json.dumps(summary, indent=1))
    pool.write_run_files(summary)
    ok = rc_final == 0 and len(pool.results) == total and not pool.seen_net and not pool.profile_touched
    if a.verify_against and summary["verify"]["results_equal"] != total:
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
