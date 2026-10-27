"""The one-night machine sampler for `envproto.py play --sample-machine SECONDS` (the real arm's slow windows).

L1 found machine-wide slow windows in the nightly's real arm (8-13x the normal seconds per decision for 37-50 minutes, every worker in
the same wall-clock minutes, CPU low inside the window) and nothing on disk to say why. A thread of the play process appends one JSON
line to `machine.jsonl` in the block's folder every SECONDS; the morning after, its lines join to the run rows by time (`t` here and
`t_end` in a run row are both epoch seconds).

A line holds: t (epoch s), utc, cpu_pct (whole machine), tree_cpu_pct (this process and everything it started, as a share of the whole
machine), avail_mb, mem_load_pct, pages_input_per_s, disk_queue, disk_read_bps, disk_write_bps, disk_latency_ms (the drive the block's
game copies live on), timer_ms (the system-wide timer resolution now in force), games (game processes alive in the tree), runs_done.

Only what the interpreter has: ctypes into kernel32 (memory, system and process times, the process snapshot), ntdll (timer resolution) and
pdh (the performance counters, read in process: no typeperf, no PowerShell, no child process). psutil is not in the referee venv. A rate
(cpu, disk) needs two readings, so the thread takes one at start and discards it; a reading that fails is None in its line and is logged
once, and a failing sample never reaches the block.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path
from typing import Any

GAME_EXE = "slaythespire2"  # the game's process image name (lowercase prefix)
FIELDS = ("cpu_pct", "tree_cpu_pct", "avail_mb", "mem_load_pct", "pages_input_per_s", "disk_queue", "disk_read_bps", "disk_write_bps",
          "disk_latency_ms", "timer_ms", "games")


class MachineSampler:
    """Appends `probes.read()` + the time + `runs_done()` to `path` every `interval` seconds on a daemon thread until `stop()`."""

    def __init__(self, path: Path, interval: float, probes: Any, runs_done: Callable[[], int] = lambda: 0,
                 clock: Callable[[], float] = time.time, log: Callable[[str], None] | None = None):
        self.path, self.interval, self.probes, self.runs_done, self.clock = Path(path), float(interval), probes, runs_done, clock
        self._log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self._logged: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _once(self, key: str, e: BaseException) -> None:
        if key not in self._logged:  # one line per kind of failure, never one per sample
            self._logged.add(key)
            self._log(f"machine sampler: {key} failed ({type(e).__name__}); that reading is skipped")

    def sample_once(self) -> dict | None:
        """One line: written and returned, or None when the reading or the write failed (logged once)."""
        try:
            t = self.clock()
            row = {"t": round(t, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)), **self.probes.read(),
                   "runs_done": int(self.runs_done())}
        except Exception as e:  # noqa: BLE001 -- a sampler fault is never the block's
            self._once("read", e)
            return None
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as e:
            self._once("write", e)
            return None
        return row

    def _run(self) -> None:
        try:
            self.probes.read()  # the rates start from this reading
        except Exception as e:  # noqa: BLE001
            self._once("first read", e)
        while not self._stop.wait(self.interval):
            self.sample_once()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="machine-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)


# ---------------------------------------------------------------- the Windows readings (ctypes only)

class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class _ProcEntry(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


class _PdhValue(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]


class WinProbes:
    """`read()` -> the FIELDS of a line. `drive` ("D:") is the drive of the game copies; None skips the disk fields."""

    def __init__(self, drive: str | None = None, root_pid: int | None = None, log: Callable[[str], None] | None = None):
        k32 = self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        k32.Process32FirstW.argtypes = k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcEntry)]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(ctypes.c_ulonglong)] * 4
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.root = root_pid or os.getpid()
        self.ncpu = os.cpu_count() or 1
        self._log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self._logged: set[str] = set()
        self._sys_prev: tuple[int, int, int] | None = None
        self._tree_prev: dict[int, int] = {}
        self._tree_t: float | None = None
        self._pdh = None
        self._counters: dict[str, Any] = {}
        self._pdh_open(drive)

    def _once(self, key: str, why: str) -> None:
        if key not in self._logged:
            self._logged.add(key)
            self._log(f"machine sampler: {key} unavailable ({why}); None in the lines")

    # ---- performance counters (pdh.dll, in process)
    def _pdh_open(self, drive: str | None) -> None:
        paths = {"pages_input_per_s": r"\Memory\Pages Input/sec"}
        if drive and len(drive) == 2 and drive[1] == ":":
            d = drive.upper()
            paths.update(disk_queue=rf"\LogicalDisk({d})\Current Disk Queue Length", disk_read_bps=rf"\LogicalDisk({d})\Disk Read Bytes/sec",
                         disk_write_bps=rf"\LogicalDisk({d})\Disk Write Bytes/sec", disk_latency_ms=rf"\LogicalDisk({d})\Avg. Disk sec/Transfer")
        try:
            pdh = ctypes.WinDLL("pdh")
            pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
            pdh.PdhGetFormattedCounterValue.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(_PdhValue)]
            q = ctypes.c_void_p()
            if pdh.PdhOpenQueryW(None, None, ctypes.byref(q)) != 0:
                raise OSError("PdhOpenQuery")
            self._pdh, self._query = pdh, q
        except (OSError, AttributeError) as e:
            self._once("performance counters", type(e).__name__)
            return
        for field, path in paths.items():
            h = ctypes.c_void_p()
            if pdh.PdhAddEnglishCounterW(q, path, None, ctypes.byref(h)) == 0:
                self._counters[field] = h
            else:
                self._once(field, "counter not found")

    def _pdh_read(self) -> dict:
        out: dict[str, float | None] = {k: None for k in ("pages_input_per_s", "disk_queue", "disk_read_bps", "disk_write_bps", "disk_latency_ms")}
        if self._pdh is None or not self._counters:
            return out
        self._pdh.PdhCollectQueryData(self._query)
        for field, h in self._counters.items():
            v = _PdhValue()
            if self._pdh.PdhGetFormattedCounterValue(h, 0x200, None, ctypes.byref(v)) == 0 and v.CStatus in (0, 1):  # PDH_FMT_DOUBLE; valid
                out[field] = round(v.doubleValue * (1000.0 if field == "disk_latency_ms" else 1.0), 3)
        return out

    # ---- kernel32 / ntdll
    def _memory(self) -> dict:
        ms = _MemStatus()
        ms.dwLength = ctypes.sizeof(ms)
        if not self.k32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            raise OSError("GlobalMemoryStatusEx")
        return {"avail_mb": int(ms.ullAvailPhys >> 20), "mem_load_pct": int(ms.dwMemoryLoad)}

    def _cpu(self) -> float | None:
        ft = [ctypes.c_ulonglong() for _ in range(3)]
        if not self.k32.GetSystemTimes(*[ctypes.byref(x) for x in ft]):
            raise OSError("GetSystemTimes")
        idle, kern, user = (x.value for x in ft)  # kernel time includes idle
        prev, self._sys_prev = self._sys_prev, (idle, kern, user)
        if prev is None:
            return None
        busy = (kern - prev[1]) + (user - prev[2])
        return round(100 * (1 - (idle - prev[0]) / busy), 1) if busy > 0 else None

    def _tree(self) -> tuple[float | None, int]:
        """(CPU share of the whole machine used by this process and all it started since the last call, game processes alive)."""
        snap = self.k32.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
        if snap in (None, wintypes.HANDLE(-1).value):
            raise OSError("CreateToolhelp32Snapshot")
        kids: dict[int, list[tuple[int, str]]] = {}
        e = _ProcEntry()
        e.dwSize = ctypes.sizeof(e)
        try:
            ok = self.k32.Process32FirstW(snap, ctypes.byref(e))
            while ok:
                kids.setdefault(e.th32ParentProcessID, []).append((e.th32ProcessID, e.szExeFile.lower()))
                ok = self.k32.Process32NextW(snap, ctypes.byref(e))
        finally:
            self.k32.CloseHandle(snap)
        members, games, todo = {self.root}, 0, [self.root]
        while todo:
            for pid, exe in kids.get(todo.pop(), []):
                if pid not in members:
                    members.add(pid)
                    todo.append(pid)
                    games += exe.startswith(GAME_EXE)
        now, cur = time.monotonic(), {}
        for pid in members:
            h = self.k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                continue
            try:
                t = [ctypes.c_ulonglong() for _ in range(4)]
                if self.k32.GetProcessTimes(h, *[ctypes.byref(x) for x in t]):
                    cur[pid] = t[2].value + t[3].value  # kernel + user, 100 ns
            finally:
                self.k32.CloseHandle(h)
        prev, t0, self._tree_prev, self._tree_t = self._tree_prev, self._tree_t, cur, now
        if t0 is None or now <= t0:
            return None, games
        used = sum(max(0, v - prev.get(pid, 0)) for pid, v in cur.items())  # a process new since the last call counts from zero
        return round(100 * used / ((now - t0) * 1e7 * self.ncpu), 1), games

    def _timer_ms(self) -> float:
        lo, hi, cur = (wintypes.ULONG() for _ in range(3))
        if ctypes.WinDLL("ntdll").NtQueryTimerResolution(ctypes.byref(lo), ctypes.byref(hi), ctypes.byref(cur)) != 0:
            raise OSError("NtQueryTimerResolution")
        return round(cur.value / 1e4, 3)  # 100 ns units

    def read(self) -> dict:
        row: dict[str, Any] = {k: None for k in FIELDS}
        for name, fn in (("memory", lambda: row.update(self._memory())), ("cpu", lambda: row.update(cpu_pct=self._cpu())),
                         ("process tree", lambda: row.update(zip(("tree_cpu_pct", "games"), self._tree()))),
                         ("counters", lambda: row.update(self._pdh_read())), ("timer", lambda: row.update(timer_ms=self._timer_ms()))):
            try:
                fn()
            except Exception as e:  # noqa: BLE001 -- one failing reading is None in the line, the others stay
                self._once(name, type(e).__name__)
        return row
