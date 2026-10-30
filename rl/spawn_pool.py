"""A spawn pool that cannot lose a task: the emulator arm's workers (rl/gen0.py `run_block`).

`multiprocessing.Pool` hands a task to a worker and then only waits for its result. A worker that exits while the task runs
(a native crash, the memory killer, a stray terminate) is replaced, but the task is not handed out again and its result never
comes: the dispatcher's `AsyncResult.ready()` stays false for ever (CPython issue 22393). This pool keeps one pipe and one
task per worker, so it always knows which task a dead worker held, and it has what the stdlib pool lacks:

- a death is reported (`Done.status == "died"`), the worker replaced, and the task is the caller's to run again;
- a per-run wall timeout kills the worker (this pool's own child, by its Process object) and reports `"timeout"`;
- a worker is recycled after `recycle` finished runs, only while idle (between runs), never with a task in flight;
- `grow` widens it, `shrink` narrows it between runs, and `reprioritize` moves its workers to another Windows priority
  class (the streamed emulator arm of rl/gen0.py plays on a few Idle workers beside the real game, more while some of its
  game slots stand empty, then on the full width at BelowNormal).

The retry policy (once, then a `lost` row) belongs to the caller. Everything is polled from one thread: no locks.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import wait as _conn_wait
from typing import Any

STOP_JOIN_S = 10.0  # a retired worker is given this long to exit on its own before it is terminated
GRAVE_S = 30.0
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000  # the Windows priority classes (as in sts2_gym.procutil)
IDLE_PRIORITY_CLASS = 0x00000040


def set_priority_class_of(pid: int | None, priority_class: int) -> bool:
    """Put process `pid` (a child of this process) in the Windows `priority_class`. False off Windows or when it cannot be set."""
    if pid is None or sys.platform != "win32":
        return False
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetPriorityClass.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = k32.OpenProcess(0x0200 | 0x1000, False, int(pid))  # PROCESS_SET_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        return bool(k32.SetPriorityClass(handle, priority_class))
    finally:
        k32.CloseHandle(handle)


def _worker_main(conn, initializer) -> None:
    """The worker's loop: one task at a time from the pipe; `None` (or a closed pipe) ends it."""
    if initializer is not None:
        initializer()
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        fn, args = msg
        try:
            res = ("ok", fn(*args))
        except BaseException as e:  # noqa: BLE001 -- the task is reported; this worker is replaced (its state is not trusted)
            res = ("raised", f"{type(e).__name__}: {e}"[:300])
        try:
            conn.send(res)
        except (EOFError, OSError):
            return
        if res[0] != "ok":
            return


@dataclass
class Done:
    """One task's end. `status`: "ok" (`value` is the task's return value), "died" (the worker exited; `value` says how),
    "timeout" (it ran past the wall timeout and was killed) or "raised" (the task raised; `value` is the error)."""
    key: Any
    status: str
    value: Any = None
    wall_s: float = 0.0
    pid: int | None = None


class _Slot:
    def __init__(self) -> None:
        self.proc = None
        self.conn = None
        self.task: tuple[Any, float] | None = None  # (key, start time) of the task in flight
        self.runs = 0  # finished tasks of this worker process


class SupervisedPool:
    def __init__(self, ctx, size: int, *, initializer: Callable[[], None] | None = None, recycle: int = 0,
                 run_timeout_s: float | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.ctx, self.initializer, self.recycle, self.run_timeout_s, self.clock = ctx, initializer, recycle, run_timeout_s, clock
        self.stats = {"started": 0, "recycled": 0, "died": 0, "timed_out": 0, "raised": 0}
        self._grave: list[tuple[Any, float]] = []  # recycled workers still exiting: (process, retired at)
        self.slots = [_Slot() for _ in range(size)]
        for s in self.slots:
            self._start(s)

    # ---- workers

    def _start(self, slot: _Slot) -> None:
        parent, child = self.ctx.Pipe()
        proc = self.ctx.Process(target=_worker_main, args=(child, self.initializer), daemon=True)
        proc.start()
        child.close()  # the parent's copy: with it closed, a dead worker reads as EOF here
        slot.proc, slot.conn, slot.task, slot.runs = proc, parent, None, 0
        self.stats["started"] += 1

    def _kill(self, slot: _Slot) -> None:
        """Stop this slot's worker now (it is the pool's own child), close its pipe."""
        proc = slot.proc
        if proc.is_alive():
            proc.terminate()
            proc.join(5.0)
            if proc.is_alive():
                proc.kill()
                proc.join(5.0)
        else:
            proc.join(1.0)
        try:
            slot.conn.close()
        except OSError:
            pass

    def _replace(self, slot: _Slot) -> None:
        self._kill(slot)
        self._start(slot)

    def _retire(self, slot: _Slot) -> None:
        """Between runs only (the slot has no task): ask the worker to leave, start its successor, reap the old one later."""
        assert slot.task is None
        try:
            slot.conn.send(None)
        except (OSError, ValueError):
            pass
        self._grave.append((slot.proc, self.clock()))
        try:
            slot.conn.close()
        except OSError:
            pass
        self.stats["recycled"] += 1
        self._start(slot)

    def _reap(self) -> None:
        keep = []
        for proc, t in self._grave:
            if not proc.is_alive():
                proc.join(0)
            elif self.clock() - t > GRAVE_S:
                proc.terminate()
                keep.append((proc, t))
            else:
                keep.append((proc, t))
        self._grave = keep

    # ---- tasks

    def free(self) -> int:
        return sum(s.task is None for s in self.slots)

    def inflight(self) -> int:
        return sum(s.task is not None for s in self.slots)

    def pids(self) -> list[int | None]:
        return [s.proc.pid for s in self.slots]

    def grow(self, size: int) -> None:
        """Widen the pool to `size` workers (never narrows): the new slots' workers start now, with the current `initializer`."""
        while len(self.slots) < size:
            slot = _Slot()
            self.slots.append(slot)
            self._start(slot)

    def shrink(self, size: int) -> int:
        """Narrow toward `size` workers: idle ones above it leave (asked to exit and reaped later, as a recycled one); busy ones are
        left to end their run, for a later call. Returns how many left."""
        gone = 0
        while len(self.slots) > max(size, 0):
            slot = next((s for s in reversed(self.slots) if s.task is None), None)
            if slot is None:
                break
            self.slots.remove(slot)
            try:
                slot.conn.send(None)
            except (OSError, ValueError):
                pass
            self._grave.append((slot.proc, self.clock()))
            try:
                slot.conn.close()
            except OSError:
                pass
            gone += 1
        return gone

    def reprioritize(self, priority_class: int, initializer: Callable[[], None] | None) -> int:
        """Put every live worker in the Windows `priority_class` (they are this pool's own children) and start later ones (a
        replaced or recycled worker, a slot `grow` adds) with `initializer`. Returns how many workers were changed (0 off Windows)."""
        self.initializer = initializer
        return sum(set_priority_class_of(s.proc.pid, priority_class) for s in self.slots if s.proc is not None and s.proc.is_alive())

    def submit(self, key: Any, fn: Callable[..., Any], args: tuple = ()) -> None:
        slot = next((s for s in self.slots if s.task is None), None)
        if slot is None:
            raise RuntimeError("SupervisedPool.submit: no free worker")
        for attempt in (0, 1):
            try:
                slot.conn.send((fn, args))
                break
            except (OSError, ValueError):
                if attempt:
                    raise
                self._replace(slot)  # the idle worker was gone: a fresh one takes the task
        slot.task = (key, self.clock())

    def _done(self, slot: _Slot, status: str, value: Any) -> Done:
        key, t0 = slot.task  # type: ignore[misc]
        pid = slot.proc.pid
        slot.task = None
        return Done(key, status, value, round(self.clock() - t0, 2), pid)

    def _poll_slot(self, slot: _Slot) -> Done | None:
        died = False
        try:
            if slot.conn.poll():
                kind, val = slot.conn.recv()
                d = self._done(slot, kind, val)
                if kind == "ok":
                    slot.runs += 1
                    if self.recycle and slot.runs >= self.recycle:
                        self._retire(slot)
                else:
                    self.stats["raised"] += 1
                    self._replace(slot)
                return d
        except (EOFError, OSError):
            died = True
        if not died and slot.proc.is_alive():
            if self.run_timeout_s is not None and self.clock() - slot.task[1] > self.run_timeout_s:  # type: ignore[index]
                d = self._done(slot, "timeout", f"no result after {self.run_timeout_s:g} s: the worker was killed")
                self.stats["timed_out"] += 1
                self._replace(slot)
                return d
            return None
        slot.proc.join(1.0)
        code = slot.proc.exitcode
        d = self._done(slot, "died", f"the worker exited with code {code} while the run was in flight")
        self.stats["died"] += 1
        self._replace(slot)
        return d

    def collect(self) -> list[Done]:
        """Every task that ended since the last call (non-blocking); dead idle workers are replaced."""
        out: list[Done] = []
        for slot in self.slots:
            if slot.task is None:
                if not slot.proc.is_alive():
                    self._replace(slot)
                continue
            d = self._poll_slot(slot)
            if d is not None:
                out.append(d)
        self._reap()
        return out

    def wait(self, timeout: float) -> None:
        """Sleep up to `timeout` seconds, less when a busy worker has answered or exited."""
        busy = [s for s in self.slots if s.task is not None]
        if not busy:
            time.sleep(min(timeout, 0.05))
            return
        _conn_wait([s.conn for s in busy] + [s.proc.sentinel for s in busy], timeout)

    # ---- the end

    def close(self) -> None:
        """Stop politely: idle workers are told to leave and given a moment, the rest are terminated."""
        for s in self.slots:
            if s.task is None:
                try:
                    s.conn.send(None)
                except (OSError, ValueError):
                    pass
        end = self.clock() + STOP_JOIN_S
        for s in self.slots:
            s.proc.join(max(0.0, end - self.clock()))
        self.terminate()

    def terminate(self) -> None:
        for s in self.slots:
            if s.proc.is_alive():
                s.proc.terminate()
        for s in self.slots:
            s.proc.join(5.0)
            try:
                s.conn.close()
            except OSError:
                pass
        for proc, _ in self._grave:
            if proc.is_alive():
                proc.terminate()
        self._grave = []
