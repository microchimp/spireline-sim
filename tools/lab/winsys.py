"""Windows process facilities for the fix-lab runner, by ctypes (the venv has no psutil).

Everything here degrades to "unknown" (None / no-op) off Windows, so the runner's logic and its tests
import on any platform. Nothing here reads another process's memory.

- ``user_idle_seconds``      GetLastInputInfo: seconds since the last keyboard or mouse input in this session.
- ``set_below_normal``       BELOW_NORMAL_PRIORITY_CLASS for this process (children inherit it).
- ``JobObject``              a Job Object: KILL_ON_JOB_CLOSE (members die with the last handle), TerminateJobObject,
                             the CPU time of every process that ever ran in it, the pids still in it.
- ``suspended_popen``        Popen that starts the child suspended, puts it in a job, then resumes it (no grandchild
                             can escape the job in the gap).
- ``system_times``           GetSystemTimes (idle, kernel, user), for the busy share of the machine.
- ``parent_alive``           whether the process that started this one still runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

IS_WIN = os.name == "nt"

BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
CREATE_SUSPENDED = 0x00000004
CREATE_NO_WINDOW = 0x08000000
NO_WINDOW = CREATE_NO_WINDOW if os.name == "nt" else 0  # creationflags for helper calls: no console flashes under pythonw
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectBasicAccountingInformation = 1
JobObjectBasicProcessIdList = 3
JobObjectExtendedLimitInformation = 9
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
WAIT_OBJECT_0 = 0

if IS_WIN:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _u32 = ctypes.WinDLL("user32", use_last_error=True)
    _nt = ctypes.WinDLL("ntdll")

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

    class _BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

    class _EXT_LIMIT(ctypes.Structure):
        _fields_ = [("Basic", _BASIC_LIMIT), ("Io", _IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _ACCOUNTING(ctypes.Structure):
        _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                    ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                    ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                    ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]

    class _FILETIME(ctypes.Structure):
        _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]

    _k32.GetCurrentProcess.restype = wintypes.HANDLE
    _k32.GetTickCount.restype = wintypes.DWORD
    _k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _k32.SetPriorityClass.restype = wintypes.BOOL
    _k32.GetPriorityClass.argtypes = [wintypes.HANDLE]
    _k32.GetPriorityClass.restype = wintypes.DWORD
    _k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _k32.SetInformationJobObject.restype = wintypes.BOOL
    _k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                               ctypes.POINTER(wintypes.DWORD)]
    _k32.QueryInformationJobObject.restype = wintypes.BOOL
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.AssignProcessToJobObject.restype = wintypes.BOOL
    _k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.TerminateJobObject.restype = wintypes.BOOL
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.CloseHandle.restype = wintypes.BOOL
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _k32.WaitForSingleObject.restype = wintypes.DWORD
    _k32.GetSystemTimes.argtypes = [ctypes.POINTER(_FILETIME)] * 3
    _k32.GetSystemTimes.restype = wintypes.BOOL
    _nt.NtResumeProcess.argtypes = [wintypes.HANDLE]
    _nt.NtResumeProcess.restype = ctypes.c_long


def user_idle_seconds() -> float | None:
    """Seconds since the last keyboard or mouse input in this session (None when it cannot be told)."""
    if not IS_WIN:
        return None
    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(LASTINPUTINFO)
    if not _u32.GetLastInputInfo(ctypes.byref(info)):
        return None
    ticks = (_k32.GetTickCount() - info.dwTime) & 0xFFFFFFFF  # both are 32-bit millisecond counters that wrap
    return ticks / 1000.0


def set_below_normal() -> bool:
    """BelowNormal priority class for this process; children inherit it. False when it could not be set."""
    if not IS_WIN:
        try:
            os.setpriority(os.PRIO_PROCESS, 0, 10)  # not meant for the Windows PC; keeps the tests honest elsewhere
            return True
        except (AttributeError, OSError):
            return False
    return bool(_k32.SetPriorityClass(_k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS))


def priority_class() -> int | None:
    """This process's priority class (Windows), None elsewhere."""
    return int(_k32.GetPriorityClass(_k32.GetCurrentProcess())) if IS_WIN else None


def system_times() -> tuple[int, int, int] | None:
    """(idle, kernel, user) in 100 ns units summed over all CPUs; kernel includes idle."""
    if not IS_WIN:
        return None
    i, k, u = _FILETIME(), _FILETIME(), _FILETIME()
    if not _k32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u)):
        return None
    return (i.hi << 32) | i.lo, (k.hi << 32) | k.lo, (u.hi << 32) | u.lo


def parent_alive(pid: int | None = None) -> bool:
    """Whether the process `pid` (default: this one's parent) is still running. True when it cannot be told."""
    if not IS_WIN:
        return True
    pid = pid or os.getppid()
    if pid <= 0:
        return True
    h = _k32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False  # gone (the parent a process was started under is always openable by it)
    try:
        return _k32.WaitForSingleObject(h, 0) != WAIT_OBJECT_0
    finally:
        _k32.CloseHandle(h)


class JobObject:
    """A Job Object. With ``kill_on_close`` every member dies when the last handle to it closes, which is also
    what happens when its owner is killed from outside. Jobs nest (Windows 8 and later): a process that is
    already in one (this runner, or a Task Scheduler instance) can be put in a second, child job."""

    def __init__(self, kill_on_close: bool = True):
        self.h = None
        if not IS_WIN:
            return
        self.h = _k32.CreateJobObjectW(None, None)
        if not self.h:
            raise OSError(ctypes.get_last_error(), "CreateJobObject failed")
        if kill_on_close:
            ext = _EXT_LIMIT()
            ext.Basic.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not _k32.SetInformationJobObject(self.h, JobObjectExtendedLimitInformation, ctypes.byref(ext),
                                                ctypes.sizeof(ext)):
                err = ctypes.get_last_error()
                self.close()
                raise OSError(err, "SetInformationJobObject(KILL_ON_JOB_CLOSE) failed")

    def assign_handle(self, process_handle: int) -> None:
        if self.h and not _k32.AssignProcessToJobObject(self.h, process_handle):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")

    def assign_self(self) -> None:
        if self.h:
            self.assign_handle(_k32.GetCurrentProcess())

    def terminate(self, code: int = 1) -> None:
        """Kill every process in the job (and so everything they started)."""
        if self.h:
            _k32.TerminateJobObject(self.h, code)

    def cpu_time(self) -> int:
        """User + kernel time (100 ns) of every process that has run in the job, finished ones included."""
        if not self.h:
            return 0
        acc = _ACCOUNTING()
        if not _k32.QueryInformationJobObject(self.h, JobObjectBasicAccountingInformation, ctypes.byref(acc),
                                              ctypes.sizeof(acc), None):
            return 0
        return int(acc.TotalUserTime) + int(acc.TotalKernelTime)

    def pids(self) -> list[int]:
        """Process ids still in the job."""
        if not self.h:
            return []
        n = 512

        class _IDS(ctypes.Structure):
            _fields_ = [("Assigned", wintypes.DWORD), ("InList", wintypes.DWORD), ("Ids", ctypes.c_size_t * n)]

        ids = _IDS()
        if not _k32.QueryInformationJobObject(self.h, JobObjectBasicProcessIdList, ctypes.byref(ids), ctypes.sizeof(ids), None):
            return []
        return [int(ids.Ids[i]) for i in range(min(ids.InList, n))]

    def close(self) -> None:
        if self.h:
            _k32.CloseHandle(self.h)
            self.h = None


def suspended_popen(argv: list[str], job: JobObject | None, **kw: Any) -> subprocess.Popen:
    """Popen that puts the child in `job` before it runs a single instruction (created suspended, assigned, then
    resumed), so nothing it starts can be outside the job. Without a job, or off Windows, a plain Popen."""
    if not (IS_WIN and job and job.h):
        return subprocess.Popen(argv, **kw)
    kw["creationflags"] = kw.get("creationflags", 0) | CREATE_SUSPENDED
    p = subprocess.Popen(argv, **kw)
    try:
        job.assign_handle(int(p._handle))  # type: ignore[attr-defined]
    except OSError:
        p.kill()
        raise
    _nt.NtResumeProcess(int(p._handle))  # type: ignore[attr-defined]
    return p


_MUTEXES: list[Any] = []


def acquire_mutex(name: str) -> bool:
    """Take a named (per-session) mutex for the life of this process; False when another process holds it.
    The handle is never closed here, so the mutex goes with the process, however it ends. True off Windows."""
    if not IS_WIN:
        return True
    _k32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    _k32.CreateMutexW.restype = wintypes.HANDLE
    h = _k32.CreateMutexW(None, False, "Local\\" + name)
    err = ctypes.get_last_error()
    if not h:
        return True  # cannot tell: do not block the lab on it
    _MUTEXES.append(h)
    return err != 183  # ERROR_ALREADY_EXISTS


def kill_tree_by_pid(pid: int) -> None:
    """Last resort for a process that is in no job: taskkill /T /F."""
    if IS_WIN:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False, creationflags=NO_WINDOW)
    else:  # pragma: no cover
        try:
            os.kill(pid, 9)
        except OSError:
            pass


if __name__ == "__main__":  # a quick look at what this PC reports
    print("idle s", user_idle_seconds(), "| priority class", priority_class(), "| times", system_times(), file=sys.stderr)
