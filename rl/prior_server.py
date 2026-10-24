#!/usr/bin/env python3
"""The batched prior server: the combat policy head's logits for many workers' positions, one forward per batch.

What it is: one process that loads a `combat_policy.pt`, owns it on the GPU as one captured CUDA graph per shape
bucket (batch size x token length), and answers the self-play workers' PRIOR frames (`prior_protocol`, over a named
pipe on Windows and an AF_UNIX socket elsewhere) with the 32 logits at the action slots. Design and numbers:
`docs/rl/GPU_INFERENCE.md`. Run it under the GPU venv's python with `STS2_LIB_PATH` set (it needs `combat_tokens`).

Why: an eager forward on the GPU costs about as much host CPU as the CPU forward it replaces; a graph replay costs a
fraction of it. `--selftest N --clients C` measures that on the real checkpoint (the go/no-go of section 9.0).

What it must not do: see the emulator, the search or a seed; unpickle anything (frames are bytes, the challenge uses
`send_bytes`/`recv_bytes`); listen on TCP or answer remote clients; run more than one torch thread or let a NORMAL
priority thread spin (the sync blocks); log a traceback or a path under the user profile (exception type and message
only); survive a thread failure (any exception in any thread is one stderr line, the stats file, `os._exit(1)`: never
a half-alive server); idle out while its parent lives (it exits when its parent is gone, its stdin is closed, or the
kill file appears); or be killed by image name (use the kill file or close its stdin).

`--device cpu` exists for the tests alone. It runs the same bucketing and padding code without graphs, leaves the
fused attention path as the CPU prior has it, and runs a batch of one at its own length, so one request is bit-exact
with `combat_prior.LivePrior`.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"  # before torch: no NORMAL-priority thread may spin
os.environ["MKL_NUM_THREADS"] = "1"

import time

_T0 = time.perf_counter()

import argparse
import collections
import hashlib
import json
import multiprocessing
import queue
import secrets
import subprocess
import sys
import tempfile
import threading
from multiprocessing import connection
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import combat_prior
import prior_protocol as proto
from combat_net import CombatNet
from combat_tokens import (
    MAX_ACTIONS,
    MAX_ENEMIES,
    N_FEATURES,
    N_TYPES,
    TYPE_BUFF,
    TYPE_END_TURN,
    TYPE_ENEMY,
    TYPE_ENEMY_BUFF,
    TYPE_GLOBAL,
    TYPE_INTENT,
    TYPE_INTENT_NEXT,
    TYPE_PAD,
    TYPE_RELIC,
    VOCAB,
)

from sts2_gym import procutil

BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
TOKEN_STEP = 8
# Recorded root positions of the generation-3 window are 16-57 tokens long (p50 26, p99 42, max 57 over 10,152 samples), so
# 64 covers them with 64 graphs; 256 would capture four times as many and reserve ~2.2 GB of GPU memory for nothing.
DEFAULT_MAX_TOKENS = 64
HARD_MAX_TOKENS = 1024  # longer than this is a bad request, not an eager forward
MAX_FRAME = 1 << 17
# One float32 row per token: the 18 features, then type, id, owner. Every integer here is far below 2**24, so the
# float is exact, and a batch is one staging buffer, one host-to-device copy and one device-to-host copy.
WIDTH = N_FEATURES + 3
C_TYPE, C_ID, C_OWNER = N_FEATURES, N_FEATURES + 1, N_FEATURES + 2
PIPE_REJECT_REMOTE_CLIENTS = 0x00000008
FAULT_ENV = "SPIRELINE_PRIOR_SERVER_FAULT"

if sys.platform == "win32":
    import _winapi

    class LocalPipeListener(connection.PipeListener):
        """`multiprocessing.connection.PipeListener` whose pipe refuses clients on other machines.

        The standard-library listener does not set PIPE_REJECT_REMOTE_CLIENTS, so its pipe is reachable over the
        network by default. `_new_handle` is the standard library's, with that one flag added.
        """

        def _new_handle(self, first=False):
            flags = _winapi.PIPE_ACCESS_DUPLEX | _winapi.FILE_FLAG_OVERLAPPED
            if first:
                flags |= _winapi.FILE_FLAG_FIRST_PIPE_INSTANCE
            return _winapi.CreateNamedPipe(
                self._address,
                flags,
                _winapi.PIPE_TYPE_MESSAGE
                | _winapi.PIPE_READMODE_MESSAGE
                | _winapi.PIPE_WAIT
                | PIPE_REJECT_REMOTE_CLIENTS,
                _winapi.PIPE_UNLIMITED_INSTANCES,
                connection.BUFSIZE,
                connection.BUFSIZE,
                _winapi.NMPWAIT_WAIT_FOREVER,
                _winapi.NULL,
            )


def _summary(values) -> dict:
    if not len(values):
        return {"mean": None, "p50": None, "p99": None, "n": 0}
    data = np.asarray(values, dtype=np.float64)
    p50, p99 = np.percentile(data, [50, 99])
    return {"mean": round(float(data.mean()), 4), "p50": round(float(p50), 4), "p99": round(float(p99), 4), "n": int(data.size)}


def _rss_mb() -> float | None:
    """Resident set (working set on Windows) of this process in MiB."""
    try:
        import psutil  # noqa: PLC0415

        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        kernel, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / 2**20
        return None
    import resource  # noqa: PLC0415

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Request:
    """One PRIOR frame waiting for the batcher: its tokens in, its logits row out."""

    __slots__ = ("types", "ids", "owners", "feats", "n", "logits", "event", "t_enq", "t_done")

    def __init__(self, types, ids, owners, feats):
        self.types, self.ids, self.owners, self.feats = types, ids, owners, feats
        self.n = len(types)
        self.logits = None
        self.event = threading.Event()
        self.t_enq = time.perf_counter()
        self.t_done = 0.0


class PriorServer:
    """The model, its buckets, the listener and the threads."""

    def __init__(self, policy, device, *, max_batch=128, max_tokens=DEFAULT_MAX_TOKENS, deadline_us=0, stats_path=None,
                 kill_file=None):
        self.policy = str(policy)
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.max_batch = max(1, int(max_batch))
        self.token_sizes = list(range(TOKEN_STEP, (max(int(max_tokens), 1) + TOKEN_STEP - 1) // TOKEN_STEP * TOKEN_STEP + 1, TOKEN_STEP))
        self.max_tokens = self.token_sizes[-1]
        sizes = [b for b in BATCH_SIZES if b <= self.max_batch]
        if sizes[-1] != self.max_batch:
            sizes.append(self.max_batch)
        self.batch_sizes = sizes
        self.deadline = max(0, int(deadline_us)) / 1e6
        self.stats_path = stats_path
        self.kill_file = kill_file
        self.fault = os.environ.get(FAULT_ENV)
        self.queue: queue.SimpleQueue = queue.SimpleQueue()
        self.lock = threading.Lock()
        self.calls = self.batches = self.overlong = self.connections = self.refused = 0
        self.hist: collections.Counter = collections.Counter()
        self.queue_wait = collections.deque(maxlen=20000)
        self.replay = collections.deque(maxlen=20000)
        self.total = collections.deque(maxlen=20000)
        self.buckets: dict = {}
        self.capture_s = 0.0
        self.ready_s = None
        self.address = None
        self.listener = None
        self.authkey = b""
        self.stop_stdin = threading.Event()
        self.stage = None

    # ------------------------------------------------------------------ startup

    def load(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if self.cuda:
            # the fused eval-mode encoder path is neither capturable nor the reference numerics; the CPU test device
            # keeps it so that it plays exactly as the CPU prior does
            torch.backends.mha.set_fastpath_enabled(False)
        blob = torch.load(self.policy, map_location="cpu", weights_only=True)
        model = CombatNet(
            VOCAB,
            blob["n_buckets"],
            d_model=blob.get("d_model", 128),
            layers=blob.get("layers", 4),
            heads=blob.get("heads", 4),
        )
        model.load_state_dict(blob["model"])
        self.model = model.eval().to(self.device)
        self.policy_sha = hashlib.sha256(Path(self.policy).read_bytes()).hexdigest()
        self.vocab_sha = combat_prior.vocab_sha256()
        self.card_sha = proto.card_table_sha256()
        limits = np.full(N_TYPES, 1 << 30, dtype=np.int64)
        for kind, table in (
            (TYPE_BUFF, model.buff_emb),
            (TYPE_ENEMY_BUFF, model.buff_emb),
            (TYPE_GLOBAL, model.encounter_emb),
            (TYPE_RELIC, model.relic_emb),
            (TYPE_ENEMY, model.enemy_emb),
            (TYPE_INTENT, model.intent_emb),
            (TYPE_INTENT_NEXT, model.intent_emb),
        ):
            limits[kind] = table.num_embeddings
        self.id_limits = limits
        cells = self.max_batch * self.max_tokens * WIDTH
        self.stage = torch.zeros(cells, pin_memory=self.cuda)
        self.stage_np = self.stage.numpy()
        self.host_out = torch.zeros(self.max_batch, MAX_ACTIONS, pin_memory=self.cuda)
        if self.cuda:
            self.static_in = torch.zeros(cells, device=self.device)
            self.event = torch.cuda.Event(blocking=True)  # the default sync spins, and charges the wait as CPU

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        types = packed[..., C_TYPE].long()
        ids = packed[..., C_ID].long()
        owners = packed[..., C_OWNER].long()
        feats = packed[..., :N_FEATURES].contiguous()
        return self.model.policy_logits(types, ids, owners, feats, types == TYPE_PAD)

    def capture(self) -> None:
        """One CUDA graph per (batch, tokens) bucket, warmed first on a side stream; all before READY."""
        started = time.perf_counter()
        if self.cuda:
            pool = torch.cuda.graph_pool_handle()
            side = torch.cuda.Stream()
            for batch in self.batch_sizes:
                for tokens in self.token_sizes:
                    inp = self.static_in[: batch * tokens * WIDTH].view(batch, tokens, WIDTH)
                    inp.zero_()
                    inp[:, 0, C_TYPE] = TYPE_GLOBAL  # a valid sequence, never an all-PAD one
                    inp[:, 1:, C_TYPE] = TYPE_END_TURN
                    torch.cuda.synchronize()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side), torch.no_grad():
                        for _ in range(3):
                            self.forward(inp)
                    torch.cuda.current_stream().wait_stream(side)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool), torch.no_grad():
                        out = self.forward(inp)
                    self.buckets[(batch, tokens)] = (graph, inp, out)
            torch.cuda.synchronize()
        self.capture_s = time.perf_counter() - started

    # ------------------------------------------------------------------ the batcher

    def _fill(self, batch: list[Request], rows: int, tokens: int) -> np.ndarray:
        view = self.stage_np[: rows * tokens * WIDTH].reshape(rows, tokens, WIDTH)
        view.fill(0.0)
        for i, request in enumerate(batch):
            row, n = view[i], request.n
            row[:n, :N_FEATURES] = request.feats
            row[:n, C_TYPE] = request.types
            row[:n, C_ID] = request.ids
            row[:n, C_OWNER] = request.owners
        if len(batch) < rows:
            view[len(batch) :] = view[0]  # spare rows are copies of a real row, never all-PAD
        return view

    def _bucket_of(self, batch: list[Request]) -> tuple[int, int]:
        longest = max(request.n for request in batch)
        if not self.cuda and len(batch) == 1:
            return 1, longest  # the test device: one request, unpadded, exactly as the CPU prior runs it
        rows = next(b for b in self.batch_sizes if b >= len(batch))
        return rows, (longest + TOKEN_STEP - 1) // TOKEN_STEP * TOKEN_STEP

    def _logits(self, rows: int, tokens: int) -> np.ndarray:
        """The staged batch's logits as a host array [rows, MAX_ACTIONS] (valid until the next call)."""
        if self.cuda:
            graph, inp, out = self.buckets[(rows, tokens)]
            inp.copy_(self.stage[: rows * tokens * WIDTH].view(rows, tokens, WIDTH), non_blocking=True)
            graph.replay()
            host = self.host_out[:rows]
            host.copy_(out, non_blocking=True)
            self.event.record()
            self.event.synchronize()
            return host.numpy()
        with torch.inference_mode():
            return self.forward(self.stage[: rows * tokens * WIDTH].view(rows, tokens, WIDTH)).numpy()

    def _eager_alone(self, request: Request) -> None:
        packed = np.zeros((1, request.n, WIDTH), dtype=np.float32)
        packed[0, :, :N_FEATURES] = request.feats
        packed[0, :, C_TYPE] = request.types
        packed[0, :, C_ID] = request.ids
        packed[0, :, C_OWNER] = request.owners
        with torch.inference_mode():
            request.logits = self.forward(torch.from_numpy(packed).to(self.device))[0].cpu().numpy().copy()

    def run(self, batch: list[Request]) -> None:
        """Answer `batch`: fill the bucket, replay, hand each request its row, wake its connection thread."""
        if self.fault == "batcher":
            raise RuntimeError("injected fault (batcher)")
        started = time.perf_counter()
        normal = [r for r in batch if r.n <= self.max_tokens]
        for request in batch:
            if request.n > self.max_tokens:
                self._eager_alone(request)
        replay_ms = None
        if normal:
            rows, tokens = self._bucket_of(normal)
            self._fill(normal, rows, tokens)
            logits = self._logits(rows, tokens)
            for i, request in enumerate(normal):
                request.logits = logits[i].copy()
            replay_ms = (time.perf_counter() - started) * 1000.0
        done = time.perf_counter()
        for request in batch:
            request.t_done = done
            request.event.set()
        with self.lock:
            self.calls += len(batch)
            self.overlong += len(batch) - len(normal)
            if normal:
                self.batches += 1
                self.hist[len(normal)] += 1
                self.replay.append(replay_ms)
            for request in batch:
                self.queue_wait.append((started - request.t_enq) * 1000.0)
                self.total.append((done - request.t_enq) * 1000.0)

    def batcher(self) -> None:
        while True:
            batch = [self.queue.get()]  # blocks: no GPU work on an empty queue
            if self.deadline:
                limit = time.perf_counter() + self.deadline
            while len(batch) < self.max_batch:
                try:
                    if self.deadline:
                        left = limit - time.perf_counter()
                        if left <= 0:
                            break
                        batch.append(self.queue.get(timeout=left))
                    else:
                        batch.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            self.run(batch)

    # ------------------------------------------------------------------ connections

    def _valid(self, request: Request) -> bool:
        if request.n > HARD_MAX_TOKENS or int(request.types.max()) >= N_TYPES or int(request.owners.max()) > MAX_ENEMIES:
            return False
        return int(request.ids.min()) >= 0 and bool((request.ids < self.id_limits[request.types]).all())

    def _hello(self, data: bytes) -> tuple[int, str]:
        try:
            version, policy, vocab, card = proto.unpack_hello(data)
        except proto.ProtocolError as error:
            return proto.STATUS_PROTO, f"bad HELLO: {error}"
        for status, name, got, want in (
            (proto.STATUS_PROTO, "proto", version, proto.PROTO),
            (proto.STATUS_POLICY, "policy_sha256", policy, self.policy_sha),
            (proto.STATUS_VOCAB, "vocab_sha256", vocab, self.vocab_sha),
            (proto.STATUS_CARD_TABLE, "card_table_sha256", card, self.card_sha),
        ):
            if got != want:
                return status, f"{name} differs (server {str(want)[:12]}, client {str(got)[:12]})"
        return proto.STATUS_OK, ""

    def serve_connection(self, conn) -> None:
        """Own one connection end to end: challenge, HELLO, then request, wait, reply until it closes."""
        try:
            try:
                connection.deliver_challenge(conn, self.authkey)
                connection.answer_challenge(conn, self.authkey)
                status, reason = self._hello(conn.recv_bytes(MAX_FRAME))
                conn.send_bytes(proto.pack_hello_reply(status, reason))
                if status != proto.STATUS_OK:
                    with self.lock:
                        self.refused += 1
                    return
                with self.lock:
                    self.connections += 1
                while True:
                    data = conn.recv_bytes(MAX_FRAME)
                    try:
                        if proto.peek_kind(data) == proto.BYE:
                            return
                        req_id, types, ids, owners, feats, _flags = proto.unpack_prior(data)
                    except proto.ProtocolError:
                        conn.send_bytes(proto.pack_reply(proto.REPLY_BAD_REQUEST, 0))  # status 1, then close
                        return
                    request = Request(types, ids, owners, feats)
                    if not self._valid(request):
                        conn.send_bytes(proto.pack_reply(proto.REPLY_BAD_REQUEST, req_id))
                        return
                    if self.fault == "connection":
                        raise RuntimeError("injected fault (connection)")
                    self.queue.put(request)
                    request.event.wait()
                    conn.send_bytes(proto.pack_reply(proto.REPLY_OK, req_id, request.logits))
            except (EOFError, OSError, ValueError, proto.ProtocolError, multiprocessing.AuthenticationError):
                return  # that client is gone or wrong; the server carries on
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def accept_loop(self) -> None:
        while True:
            conn = self.listener.accept()
            self.guard(lambda c=conn: self.serve_connection(c), "connection")

    def guard(self, target, name: str) -> threading.Thread:
        """Run `target` in a daemon thread whose failure takes the whole process down."""

        def run() -> None:
            try:
                target()
            except BaseException as error:  # noqa: BLE001
                self.fatal(error)

        thread = threading.Thread(target=run, name=name, daemon=True)
        thread.start()
        return thread

    def serve(self, endpoint: str, authkey: bytes) -> None:
        self.address, self.authkey = proto.pipe_address(endpoint), authkey
        if sys.platform == "win32":
            self.listener = LocalPipeListener(self.address)
        else:
            self.listener = connection.Listener(self.address, family="AF_UNIX")
        self.guard(self.batcher, "batcher")
        self.guard(self.accept_loop, "acceptor")
        if self.stats_path:
            self.guard(self._stats_loop, "stats")
        self.ready_s = time.perf_counter() - _T0

    # ------------------------------------------------------------------ stats and exit

    def stats(self) -> dict:
        with self.lock:
            snapshot = {
                "calls": self.calls,
                "batches": self.batches,
                "batch_hist": {str(k): v for k, v in sorted(self.hist.items())},
                "overlong": self.overlong,
                "connections": self.connections,
                "hello_refused": self.refused,
                "queue_wait_ms": _summary(list(self.queue_wait)),
                "replay_ms": _summary(list(self.replay)),
                "total_ms": _summary(list(self.total)),
            }
        snapshot.update(
            host_cpu_s=round(time.process_time(), 3),
            uptime_s=round(time.perf_counter() - _T0, 3),
            restarts=0,
            device=str(self.device),
            buckets_captured=len(self.buckets),
            capture_s=round(self.capture_s, 3),
            ready_s=None if self.ready_s is None else round(self.ready_s, 3),
        )
        return snapshot

    def write_stats(self) -> None:
        if not self.stats_path:
            return
        tmp = f"{self.stats_path}.tmp"
        Path(tmp).write_text(json.dumps(self.stats(), indent=1), encoding="utf-8")
        os.replace(tmp, self.stats_path)

    def _stats_loop(self) -> None:
        while True:
            time.sleep(5)
            self.write_stats()

    def finish(self, code: int) -> None:
        """Flush the stats and leave at once: nothing a server thread holds is worth waiting for."""
        try:
            self.write_stats()
        except Exception:  # noqa: BLE001, S110
            pass
        if self.address and sys.platform != "win32":
            try:
                os.unlink(self.address)
            except OSError:
                pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)

    def fatal(self, error: BaseException) -> None:
        sys.stderr.write(f"prior_server: {proto.describe_error(error)}\n")
        sys.stderr.flush()
        self.finish(1)

    def watch(self, parent_pid: int | None) -> None:
        """Poll the exit conditions once a second: parent gone, stdin closed, kill file present."""
        gone = procutil.watch_process(parent_pid) if parent_pid else None
        while True:
            time.sleep(1.0)
            if self.stop_stdin.is_set() or (gone and gone()) or (self.kill_file and os.path.exists(self.kill_file)):
                self.finish(0)


def read_stdin_until_eof(server: PriorServer) -> None:
    stream = getattr(sys.stdin, "buffer", None)
    if stream is None:
        return
    while stream.read(4096):
        pass
    server.stop_stdin.set()


# ---------------------------------------------------------------------------------------------- the selftest


def _measure(server: PriorServer, requests: list[Request], runs: int, eager: bool) -> tuple[float, float]:
    """(host CPU ms, wall ms) per batch over `runs` batches of `requests`, through the bucket or an eager forward."""
    rows = len(requests)
    started, cpu = time.perf_counter(), time.process_time()
    for _ in range(runs):
        if eager:
            tokens = (max(r.n for r in requests) + TOKEN_STEP - 1) // TOKEN_STEP * TOKEN_STEP
            view = server._fill(requests, rows, tokens)  # noqa: SLF001
            with torch.no_grad():
                server.forward(torch.from_numpy(view).to(server.device)).cpu().numpy()
        else:
            for request in requests:
                request.event.clear()
            server.run(requests)
    return (time.process_time() - cpu) * 1000.0 / runs, (time.perf_counter() - started) * 1000.0 / runs


def selftest(server: PriorServer, calls: int, clients: int) -> None:
    from prior_client import synthetic_tokens  # noqa: PLC0415

    print(f"device: {server.device}  buckets captured: {len(server.buckets)}  capture: {server.capture_s:.1f} s")
    rng = np.random.default_rng(0)
    for rows in (1, 4, 8, 32):
        if rows > server.max_batch:
            continue
        requests = []
        for n in rng.integers(16, 41, rows):
            t = synthetic_tokens(rng, int(n))
            requests.append(Request(t.types.astype(np.uint8), t.ids.astype(np.int32), t.owners.astype(np.uint8), t.feats))
        _measure(server, requests, 20, False)
        cpu, wall = _measure(server, requests, 1000 if server.cuda else 200, False)
        label = "graph replay" if server.cuda else "bucket forward (cpu)"
        print(f"{label} batch {rows}: host CPU {cpu:.3f} ms/batch, wall {wall:.3f} ms/batch")
        if server.cuda:
            _measure(server, requests, 10, True)
            cpu, wall = _measure(server, requests, 200, True)
            print(f"eager forward batch {rows}: host CPU {cpu:.3f} ms/batch, wall {wall:.3f} ms/batch")

    key = secrets.token_hex(32)
    endpoint = f"spireline-prior-selftest-{os.getpid()}-{secrets.token_hex(4)}"
    if sys.platform != "win32":
        endpoint = str(Path(tempfile.mkdtemp()) / "prior.sock")
    server.serve(endpoint, bytes.fromhex(key))
    shas = ",".join((server.policy_sha, server.vocab_sha, server.card_sha))
    env = dict(os.environ, SPR_SELFTEST_KEY=key)
    procs = [
        subprocess.Popen(  # noqa: S603
            [sys.executable, str(HERE / "prior_client.py"), "--bench", endpoint, "--authkey-env", "SPR_SELFTEST_KEY",
             "--shas", shas, "--n", str(calls), "--seed", str(i)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
        )
        for i in range(clients)
    ]
    for proc in procs:
        if proc.stdout.readline().strip() != "connected":
            raise RuntimeError("a selftest client did not connect")
    with server.lock:
        calls0, batches0, hist0 = server.calls, server.batches, dict(server.hist)
    started, cpu = time.perf_counter(), time.process_time()
    for proc in procs:
        proc.stdin.write("go\n")
        proc.stdin.flush()
    times: list[float] = []
    for proc in procs:
        out, _ = proc.communicate()
        times.extend(json.loads(out.strip().splitlines()[-1])["ms"])
    wall, cpu = time.perf_counter() - started, time.process_time() - cpu
    with server.lock:
        done, batches = server.calls - calls0, server.batches - batches0
        hist = {k: v - hist0.get(k, 0) for k, v in sorted(server.hist.items()) if v - hist0.get(k, 0)}
    lat = np.asarray(times)
    print(f"saturation: {clients} clients x {calls} requests back to back")
    print(f"calls/s: {done / wall:.0f}")
    print(f"batches/s: {batches / wall:.0f}")
    print(f"mean batch: {done / max(batches, 1):.2f}")
    print(f"host CPU per batch: {cpu * 1000.0 / max(batches, 1):.3f} ms (whole server process, all threads)")
    print(f"latency p50/p99: {np.percentile(lat, 50):.2f} / {np.percentile(lat, 99):.2f} ms")
    print(f"time to READY: {server.ready_s:.1f} s")
    rss = _rss_mb()
    print(f"RSS: {rss:.0f} MiB" if rss is not None else "RSS: unknown")
    if server.cuda:
        print(f"torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved() / 2**20:.0f} MiB")
    print(f"batch histogram (requests per batch: batches): {hist}")


# ---------------------------------------------------------------------------------------------- command line


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="The batched combat prior server (docs/rl/GPU_INFERENCE.md).")
    parser.add_argument("--policy", required=True, help="combat_policy.pt to serve")
    parser.add_argument("--endpoint", help="pipe name (Windows) or socket path (elsewhere)")
    parser.add_argument("--authkey-env", metavar="NAME", help="environment variable holding the key as hex digits; it is deleted once read")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="cpu is for the tests alone")
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="longest sequence with a captured graph; longer ones take an eager forward and are counted")
    parser.add_argument("--deadline-us", type=int, default=0, help="timed wait for a fuller batch; 0 (default) never waits")
    parser.add_argument("--parent-pid", type=int, help="exit when this process is gone")
    parser.add_argument("--kill-file", help="exit when this file exists")
    parser.add_argument("--priority", choices=("normal", "below"), default="normal")
    parser.add_argument("--stats", help="JSON file, rewritten every 5 s and at exit")
    parser.add_argument("--selftest", type=int, metavar="N", help="N requests per synthetic client, then print the numbers")
    parser.add_argument("--clients", type=int, default=16, help="synthetic clients for --selftest")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.selftest is None and not (args.endpoint and args.authkey_env):
        print("prior_server: --endpoint and --authkey-env are required unless --selftest is given", file=sys.stderr)
        return 2
    authkey = None
    if args.authkey_env:
        value = os.environ.pop(args.authkey_env, None)
        if not value:
            print(f"prior_server: environment variable {args.authkey_env} is not set", file=sys.stderr)
            return 2
        try:
            authkey = bytes.fromhex(value.strip())
        except ValueError:
            print(f"prior_server: environment variable {args.authkey_env} is not hexadecimal", file=sys.stderr)
            return 2
    procutil.set_priority_class(
        procutil.NORMAL_PRIORITY_CLASS if args.priority == "normal" else procutil.BELOW_NORMAL_PRIORITY_CLASS
    )
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("prior_server: --device cuda but no CUDA device is available; refusing to start", file=sys.stderr)
        return 2
    if args.device == "cpu":
        print("prior_server: WARNING --device cpu is for tests only; it is a net loss in a real job", file=sys.stderr, flush=True)
    server = PriorServer(
        args.policy, args.device, max_batch=args.max_batch, max_tokens=args.max_tokens,
        deadline_us=args.deadline_us, stats_path=args.stats, kill_file=args.kill_file,
    )
    server.load()
    server.capture()
    if args.selftest is not None:
        selftest(server, args.selftest, args.clients)
        server.finish(0)
    server.serve(args.endpoint, authkey)
    server.guard(lambda: read_stdin_until_eof(server), "stdin")
    print(f"READY {args.endpoint}", flush=True)
    try:
        server.watch(args.parent_pid)
    except KeyboardInterrupt:
        server.finish(0)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 -- startup failures: type and message, no traceback
        print(f"prior_server: {proto.describe_error(error)}", file=sys.stderr, flush=True)
        sys.exit(1)
