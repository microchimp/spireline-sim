"""Process helpers that differ between POSIX and Windows: priority, parent liveness, child listing.

One place for the calls the rl/ code used to make with `os.nice`, `os.getppid` and `ps`, none of
which behave the same (or exist) on Windows. Windows has no nice values and no reparenting of
orphans, so priority goes through the priority classes and liveness through a process handle.
"""

from __future__ import annotations

import os
import subprocess
import sys

IS_WINDOWS = sys.platform == "win32"

BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
IDLE_PRIORITY_CLASS = 0x00000040
NORMAL_PRIORITY_CLASS = 0x00000020

_SYNCHRONIZE = 0x00100000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_TH32CS_SNAPPROCESS = 0x2
_INVALID_HANDLE = -1

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.GetCurrentProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _k32.GetPriorityClass.argtypes = [wintypes.HANDLE]
    _k32.GetPriorityClass.restype = wintypes.DWORD
    _k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    for _f in (_k32.Process32FirstW, _k32.Process32NextW):
        _f.argtypes = [wintypes.HANDLE, ctypes.c_void_p]

    class _ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_wchar * 260),
        ]


def priority_class() -> int:
    """This process's Windows priority class (Windows only)."""
    return int(_k32.GetPriorityClass(_k32.GetCurrentProcess()))


def lower_priority(level: int = 10) -> None:
    """Run at `level` niceness at least (POSIX), or BelowNormal at most (Windows).

    Children inherit it on both. Idempotent: a process already lower is left as it is.
    """
    if not IS_WINDOWS:
        current = os.nice(0)
        if current < level:
            os.nice(level - current)
        return
    if level <= 0 or priority_class() in (BELOW_NORMAL_PRIORITY_CLASS, IDLE_PRIORITY_CLASS):
        return
    if not _k32.SetPriorityClass(_k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS):
        raise OSError(ctypes.get_last_error(), "SetPriorityClass failed")


def popen_priority_kwargs(level: int = 10) -> dict:
    """`subprocess` keyword arguments that start the child at a lowered priority."""
    if IS_WINDOWS:
        return {"creationflags": BELOW_NORMAL_PRIORITY_CLASS}
    return {"preexec_fn": lambda: os.nice(level)}


def parent_pid() -> int:
    """The pid to hand to `parent_alive` from a child: the pid of the process that started it."""
    return os.getppid()


def pid_alive(pid: int) -> bool:
    """Whether process `pid` is running, without signalling it.

    `os.kill(pid, 0)` is the POSIX probe, but on Windows any signal other than the two console
    events calls TerminateProcess, so signal 0 would kill the process it asks about.
    """
    if not IS_WINDOWS:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass  # alive, under another user
        return True
    return parent_alive(pid)


def parent_alive(pid: int) -> bool:
    """Whether process `pid` (the one a child was started by) is still running.

    POSIX: an orphan is reparented, so the parent is gone once `getppid` no longer returns it.
    Windows: nothing is reparented, so ask the process itself (a pid reused later by another
    process is not told apart from the parent, as with any pid check).
    """
    if not IS_WINDOWS:
        return os.getppid() == pid
    handle = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not _k32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == _STILL_ACTIVE
    finally:
        _k32.CloseHandle(handle)


def child_processes(pid: int) -> list[tuple[int, str]]:
    """(pid, command line or image name) of the direct children of `pid`; for tests.

    POSIX uses `ps`; Windows walks a Toolhelp snapshot (image name only, no command line).
    """
    if not IS_WINDOWS:
        found = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid=,command="], capture_output=True, text=True, check=True
        )
        out = []
        for line in found.stdout.splitlines():
            child, parent, command = line.split(None, 2)
            if int(parent) == pid:
                out.append((int(child), command))
        return out
    snap = _k32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if snap in (None, _INVALID_HANDLE, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_last_error(), "CreateToolhelp32Snapshot failed")
    out = []
    try:
        entry = _ProcessEntry()
        entry.dwSize = ctypes.sizeof(_ProcessEntry)
        ok = _k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            if entry.th32ParentProcessID == pid:
                out.append((int(entry.th32ProcessID), entry.szExeFile))
            ok = _k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        _k32.CloseHandle(snap)
    return out


def niceness() -> int:
    """This process's niceness: `os.nice(0)`, or the Windows priority class mapped onto it.

    BelowNormal reads 10 and Idle 19 (what `lower_priority(10)` asks for), anything else 0.
    """
    if not IS_WINDOWS:
        return os.nice(0)
    cls = priority_class()
    if cls == IDLE_PRIORITY_CLASS:
        return 19
    return 10 if cls == BELOW_NORMAL_PRIORITY_CLASS else 0
