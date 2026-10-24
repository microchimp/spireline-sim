"""Start, watch, restart and stop the batched prior server of one self-play job (docs/rl/GPU_INFERENCE.md sections 2, 6, 7).

This module imports no torch and no native code: `gen0.py` (the emulator venv) starts the server in the GPU venv's python as a
child process and talks to it only through its stdin, its stdout (the `READY <endpoint>` line) and the files it names.

    handle = start_prior_server(gpu_python, policy, lib_path=lib, kill_file=lab / "GPU_OFF", stats_path=out / "prior_stats.json")
    ...                                    # the workers connect to handle.endpoint with handle.authkey
    handle.stop()                          # close stdin, wait 5 s, terminate

The authkey is a fresh 32 random bytes per server. It travels in the child's environment variable `SPIRELINE_PRIOR_AUTHKEY`
as 64 lowercase hex digits (an environment variable cannot hold arbitrary bytes) and never on a command line.
"""

from __future__ import annotations

import os
import queue
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = ROOT / "rl" / "prior_server.py"
AUTHKEY_ENV = "SPIRELINE_PRIOR_AUTHKEY"
MODE_ENV = "SPIRELINE_PRIOR_SERVER"  # "off" overrides any flag
GPU_PYTHON_ENV = "SPIRELINE_GPU_PYTHON"
KILL_FILE_NAME = "GPU_OFF"
MODES = ("off", "auto", "require")
STOP_WAIT_S = 5.0
LOG_LINE_CAP = 200
LOG_LINES_CAP = 40  # server output lines logged per server; the rest is read and dropped


class PriorServerError(RuntimeError):
    """`--prior-server require` could not be honoured."""


def log(message: str) -> None:
    print(f"prior-server: {message}", file=sys.stderr, flush=True)


def describe_error(error: BaseException) -> str:
    """Exception type and message only, the way every log of this project names a failure."""
    return f"{type(error).__name__}: {error}"[:LOG_LINE_CAP]

_DRIVE_PATH = re.compile(r"(?<![\w])[A-Za-z]:[\/][^\s\"']*")
_UNIX_PATH = re.compile(r"(?<![\w])/(?:home|Users|tmp|var|opt)/[^\s\"']*")


def safe_child_line(text: str) -> str | None:
    """A line of the server's output as a log may carry it: traceback lines are dropped (they name files under the user
    profile), every drive-letter or home path is replaced by <path>, and the rest is capped. None means: do not log."""
    if text.startswith(("Traceback", "  File ", " ", "	")):
        return None
    text = _DRIVE_PATH.sub("<path>", text)
    text = _UNIX_PATH.sub("<path>", text)
    return text[:LOG_LINE_CAP]



# ------------------------------------------------------------------ the decision


def resolve_prior_server(mode: str, gpu_python: str | os.PathLike | None, kill_file: str | os.PathLike | None) -> str:
    """`"off"` or `"on"` for a `--prior-server` mode; raises `PriorServerError` when `require` cannot be met.

    `off` is off, and so is any mode when SPIRELINE_PRIOR_SERVER=off. `auto` is off with one log line when there is no GPU python
    or the kill file exists, else on. `require` is on, or an error when it has no GPU python or the kill file exists.
    """
    if mode not in MODES:
        raise ValueError(f"--prior-server is one of {', '.join(MODES)}, not {mode!r}")
    if mode == "off":
        return "off"
    if os.environ.get(MODE_ENV, "").strip().lower() == "off":
        log(f"{MODE_ENV}=off overrides --prior-server {mode}: CPU path")
        return "off"
    problem = None
    if not gpu_python or not Path(gpu_python).is_file():
        problem = "no GPU python (--gpu-python or " + GPU_PYTHON_ENV + ")"
    elif kill_file and Path(kill_file).exists():
        problem = "the kill file exists"
    if problem is None:
        return "on"
    if mode == "require":
        raise PriorServerError(f"--prior-server require: {problem}")
    log(f"auto: {problem}: CPU path")
    return "off"


@dataclass(frozen=True)
class ServerOpts:
    """The `--prior-server*` flags of a self-play or promotion command."""

    mode: str = "off"
    gpu_python: str | None = None
    server_procs: int | None = None
    extra_args: tuple[str, ...] = ()
    latency_p99_ms: float = 20.0
    lab: str | None = None

    @classmethod
    def from_args(cls, a: Any) -> ServerOpts:
        extra = getattr(a, "prior_server_args", None)
        return cls(mode=getattr(a, "prior_server", None) or "off",
                   gpu_python=getattr(a, "gpu_python", None) or os.environ.get(GPU_PYTHON_ENV) or None,
                   server_procs=getattr(a, "server_procs", None),
                   extra_args=tuple(shlex.split(extra)) if extra else (),
                   latency_p99_ms=float(getattr(a, "latency_p99_ms", None) or 20.0),
                   lab=getattr(a, "lab", None) or None)

    def flags(self) -> list[str]:
        """The command-line flags that rebuild these options in another `gen0.py` process (the promotion subprocess)."""
        out: list[str] = []
        if self.mode == "off":
            return out  # the default command line stays the default command line
        out += ["--prior-server", self.mode]
        if self.gpu_python:
            out += ["--gpu-python", self.gpu_python]
        if self.server_procs:
            out += ["--server-procs", str(self.server_procs)]
        if self.extra_args:
            out += ["--prior-server-args=" + shlex.join(self.extra_args)]  # one token: a value like "--foo" survives argparse
        if self.latency_p99_ms != 20.0:
            out += ["--latency-p99-ms", str(self.latency_p99_ms)]
        return out


# ------------------------------------------------------------------ the server process


def make_endpoint() -> tuple[str, str, str | None]:
    """(endpoint, kind, temp dir to remove): a named pipe on Windows, a unix socket path elsewhere (the tests)."""
    nonce = secrets.token_hex(4)
    if os.name == "nt":
        return rf"\\.\pipe\spireline-prior-{os.getpid()}-{nonce}", "pipe", None
    tmp = tempfile.mkdtemp(prefix="spireline-prior-")
    return str(Path(tmp) / f"p-{os.getpid()}-{nonce}.sock"), "unix", tmp


class ServerHandle:
    """One running prior server: its endpoint and key, the child process, and the restart budget (one)."""

    def __init__(self, cmd: list[str], env: dict[str, str], endpoint: str, kind: str, authkey: bytes, stats_path: Path,
                 ready_timeout_s: float, tmpdir: str | None) -> None:
        self._cmd, self._env, self._ready_timeout_s, self._tmpdir = cmd, env, ready_timeout_s, tmpdir
        self.endpoint, self.kind, self.authkey, self.stats_path = endpoint, kind, authkey, Path(stats_path)
        self.proc: subprocess.Popen | None = None
        self.restarts = 0

    def __repr__(self) -> str:  # never the key or the endpoint name
        return f"ServerHandle(kind={self.kind!r}, restarts={self.restarts}, alive={self.alive()})"

    def _launch(self) -> bool:
        """Start the child and wait for its READY line; False (child reaped, reason logged) when it never comes."""
        try:
            proc = subprocess.Popen(self._cmd, env=self._env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        except OSError as error:
            log(f"cannot start the server: {describe_error(error)}")
            return False
        self.proc = proc
        lines: queue.Queue[str | None] = queue.Queue()

        def pump() -> None:  # reads the child's output to its end, so a full pipe never blocks the server
            logged = 0
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    text = line.rstrip()
                    if text.startswith("READY"):
                        lines.put(text)
                    elif text and logged < LOG_LINES_CAP:
                        safe = safe_child_line(text)
                        if safe is not None:
                            logged += 1
                            log("server: " + safe)
            except (OSError, ValueError):
                pass
            finally:
                try:
                    proc.stdout.close()  # type: ignore[union-attr]
                except (OSError, ValueError):
                    pass
                lines.put(None)

        threading.Thread(target=pump, name="prior-server-output", daemon=True).start()
        try:
            return self._await_ready(proc, lines)
        except Exception as error:  # noqa: BLE001 -- a launch that fails for any reason leaves the block on the CPU path
            log(f"the server launch failed: {describe_error(error)}")
            self._reap()
            return False

    def _await_ready(self, proc: subprocess.Popen, lines: queue.Queue) -> bool:
        deadline = time.monotonic() + self._ready_timeout_s
        while True:
            try:
                got = lines.get(timeout=max(0.0, min(0.5, deadline - time.monotonic())))
            except queue.Empty:
                if time.monotonic() >= deadline:
                    log(f"no READY line within {self._ready_timeout_s:g} s: stopping the server")
                    self._reap()
                    return False
                continue
            if got is None:
                code: int | None = proc.returncode
                if proc.poll() is None:
                    try:
                        code = proc.wait(timeout=STOP_WAIT_S)
                    except (subprocess.TimeoutExpired, OSError):
                        code = None  # alive with its output closed: reaped below
                why = " (the server refuses to start without CUDA, exit 2)" if code == 2 else ""
                log(f"the server exited with code {code} before READY{why}")
                self._reap()
                return False
            return True

    def _reap(self) -> None:
        proc = self.proc
        if proc is None:
            return
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        if proc.poll() is None:
            try:
                proc.wait(timeout=STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=STOP_WAIT_S)
                except subprocess.TimeoutExpired:
                    proc.kill()  # this handle's own process, never an image name
                    proc.wait()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        """Close the server's stdin (it exits on EOF), wait 5 s, then terminate. Safe to call twice."""
        self._reap()
        if self._tmpdir is not None:
            for p in Path(self._tmpdir).glob("*"):
                try:
                    p.unlink()
                except OSError:
                    pass
            try:
                Path(self._tmpdir).rmdir()
            except OSError:
                pass
            self._tmpdir = None

    def restart_once(self) -> bool:
        """Start the server again on the same endpoint and key. At most once; False when the budget is spent or it fails."""
        if self.restarts >= 1:
            return False
        self.restarts += 1
        self._reap()
        if self.kind == "unix":
            try:
                Path(self.endpoint).unlink()
            except OSError:
                pass
        ok = self._launch()
        log("restarted the server" if ok else "the restart failed")
        return ok

    def read_stats(self) -> dict[str, Any]:
        """The server's stats file ({} when it wrote none or it is unreadable)."""
        import json  # noqa: PLC0415

        try:
            doc = json.loads(self.stats_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return doc if isinstance(doc, dict) else {}


def start_prior_server(gpu_python: str | os.PathLike, policy_path: str | os.PathLike, *, lib_path: str | os.PathLike | None,
                       kill_file: str | os.PathLike, stats_path: str | os.PathLike, extra_args: Sequence[str] = (),
                       ready_timeout_s: float = 600.0, server_script: str | os.PathLike | None = None,
                       priority: str = "normal") -> ServerHandle | None:
    """Start `rl/prior_server.py` under `gpu_python` and wait for its READY line. None (the reason logged) when it does not come.

    `extra_args` follow the standard flags, so a later `--device cpu` or `--max-batch N` wins. The child gets STS2_LIB_PATH and the
    authkey in its environment, a PIPE for stdin (closing it stops the server) and a PIPE for stdout. It starts at the parent's
    priority class and sets its own from `--priority`; it is never started detached from the parent's Job Object.
    """
    endpoint, kind, tmpdir = make_endpoint()
    authkey = secrets.token_bytes(32)
    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(gpu_python), str(server_script or SERVER_SCRIPT), "--policy", str(policy_path), "--endpoint", endpoint,
           "--authkey-env", AUTHKEY_ENV, "--device", "cuda", "--max-batch", "128", "--deadline-us", "0",
           "--parent-pid", str(os.getpid()), "--kill-file", str(kill_file), "--priority", priority,
           "--stats", str(stats_path), *map(str, extra_args)]
    env = dict(os.environ)
    env[AUTHKEY_ENV] = authkey.hex()
    if lib_path:
        env["STS2_LIB_PATH"] = str(lib_path)
    handle = ServerHandle(cmd, env, endpoint, kind, authkey, stats_path, ready_timeout_s, tmpdir)
    if handle._launch():  # noqa: SLF001 -- the module's own class
        return handle
    handle.stop()
    return None
