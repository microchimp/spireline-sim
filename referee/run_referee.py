#!/usr/bin/env python3
"""Run one referee session: the user's own copy of the game, headless, driven by our mod.

    python referee/run_referee.py --game <game copy> --job job.json [--scratch <dir>]

What it does, per Spireline security addendum 1, section 3:
  H1  refuses unless sts2.dll is the known v0.107.1 build (sha256 below);
  H2  launches with --force-steam off and an override.cfg that blanks the Sentry DSN, and records
      every network connection the game process and its children (crashpad_handler.exe) open
      (there should be none). The game starts with the error mode GAME_ERROR_MODE, which keeps Windows
      Error Reporting (WerFault.exe, a connection to Microsoft) from running for a crash of it; any
      WerFault.exe about the game's tree is reported as "wer_reports";
  H3  points APPDATA/LOCALAPPDATA at a scratch folder, so saves, logs and settings land there,
      and checks the real profile folder was not created or touched.
The driver mod is copied into <game>/mods/SpirelineReferee and is inert without the job variable.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import re
import sys
import threading
import time
from pathlib import Path

KNOWN_BUILDS = {  # sts2.dll sha256 -> build
    "a1f9e653f1e28e4076558fee1e60d218619cb7e057b887c6417f62c62c6d7a52": "v0.107.1 / 23811903",
}
HERE = Path(__file__).resolve().parent
OVERRIDE = '[sentry]\n\nconfig/dsn=""\n'


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def process_table() -> dict[int, tuple[int, str]]:
    """Every process on the machine: pid -> (parent pid, exe name), from one Toolhelp32 snapshot."""
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.Process32FirstW.argtypes = k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if snap is None or snap == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_last_error(), "CreateToolhelp32Snapshot failed")
    table: dict[int, tuple[int, str]] = {}
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            table[entry.th32ProcessID] = (entry.th32ParentProcessID, entry.szExeFile)
            ok = k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)
    return table


def creation_time(pid: int) -> int | None:
    """A running process's creation time (FILETIME ticks), or None when it cannot be read (gone, or no access)."""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        t = [wintypes.FILETIME() for _ in range(4)]
        if not k32.GetProcessTimes(h, *[ctypes.byref(x) for x in t]):
            return None
        return (t[0].dwHighDateTime << 32) | t[0].dwLowDateTime
    finally:
        k32.CloseHandle(h)


def process_trees(roots, table: dict[int, tuple[int, str]], created=None) -> dict[int, tuple[int, str]]:
    """Every root pid and all its descendants (the game's crashpad_handler, anything they start):
    pid -> (root pid, exe name). A root that has exited keeps its entry with name "". Windows keeps a
    child's parent pid after the parent exits, so a lingering child is still found.

    Windows also reuses pids: a process whose parent exited long ago still names that parent's pid, and a game
    process can be given the same number later (2026-10-16: Discord, running since 13 Oct, whose parent pid had
    been reused by a game server, was counted as the game's child and halted the lab). A real child is never
    older than its parent, so a "child" created before its parent is not one. When either time cannot be read
    the child is kept, so the check never gets weaker for a process it cannot inspect. `created` (pid -> time) is
    given with the live process table (creation_time); without it (a made-up table) no time is compared."""
    owner: dict[int, tuple[int, str]] = {int(r): (int(r), table.get(int(r), (0, ""))[1]) for r in roots}
    children: dict[int, list[int]] = {}
    for pid, (ppid, _name) in table.items():
        if pid != ppid:
            children.setdefault(ppid, []).append(pid)
    born: dict[int, int | None] = {}

    def when(pid: int) -> int | None:
        if pid not in born:
            born[pid] = created(pid) if created else None
        return born[pid]

    for root in [int(r) for r in roots]:
        stack = [root]
        while stack:
            parent = stack.pop()
            for child in children.get(parent, []):
                if child in owner:  # also cuts a cycle made by a reused parent pid
                    continue
                c, p = when(child), when(parent)
                if c is not None and p is not None and c < p:
                    continue  # older than its "parent": that parent pid was reused, so it is not a descendant
                owner[child] = (root, table[child][1])
                stack.append(child)
    return owner


def parse_netstat(out: str, owner: dict[int, tuple[int, str]]) -> dict[int, set[str]]:
    """`netstat -ano` lines of the given pids, grouped by root pid. A child's lines name the child."""
    found: dict[int, set[str]] = {root: set() for root, _ in owner.values()}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] not in ("TCP", "UDP") or not parts[-1].isdigit():
            continue
        pid = int(parts[-1])
        if pid in owner:
            root, name = owner[pid]
            endpoint = " ".join(parts[:4] if parts[0] == "TCP" else parts[:3])
            found[root].add(endpoint if pid == root else f"{endpoint} ({name} pid {pid})")
    return found


def connections_by_pid(pids) -> dict[int, set[str]]:
    """One `netstat -ano` call for any number of game pids: pid -> the TCP/UDP endpoints of the game
    process and of every process it started (crashpad_handler.exe, the crash reporter Sentry starts)."""
    owner = process_trees(pids, process_table(), created=creation_time)
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, check=False,
                         creationflags=subprocess.CREATE_NO_WINDOW).stdout
    return parse_netstat(out, owner)


def connections(pid: int) -> set[str]:
    return connections_by_pid([pid])[pid]


# Second line of defence: a Windows Error Reporting attempt for a game process is visible even when it
# opens no connection. WerFault.exe names the crashed process in its command line
# (`WerFault.exe -u -p <pid> -s <handle>`, or `-pss -s <n> -p <pid> -ip <pid>`).
WER_EXE = "werfault.exe"
_WER_PID = re.compile(r"(?<![\w-])-(?:p|ip)\s+(\d+)\b")
_WER_COMMAND_CACHE: dict[int, str] = {}  # WerFault pid -> command line (fixed for the life of the process)


def werfault_named_pids(command_line: str) -> set[int]:
    """The pids a WerFault.exe command line names (-p and -ip)."""
    return {int(x) for x in _WER_PID.findall(command_line or "")}


def command_lines(pids) -> dict[int, str]:
    """pid -> command line of those processes, from WMI through PowerShell (a process that is gone, or
    whose line cannot be read, is missing). Only called when a WerFault.exe exists."""
    ids = sorted({int(p) for p in pids})
    if not ids:
        return {}
    query = " or ".join(f"ProcessId={i}" for i in ids)
    script = (f"Get-CimInstance Win32_Process -Filter '{query}' | "
              "ForEach-Object { '{0}|{1}' -f $_.ProcessId, $_.CommandLine }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True,
                             text=True, check=False, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    lines: dict[int, str] = {}
    for line in out.splitlines():
        pid, _, cmd = line.partition("|")
        if pid.strip().isdigit():
            lines[int(pid)] = cmd.strip()
    return lines


def wer_reports_by_pid(pids, table: dict[int, tuple[int, str]] | None = None,
                       read_command_lines=command_lines, cache: dict[int, str] | None = None) -> dict[int, set[str]]:
    """Every WerFault.exe that exists now and is about a game process: pid -> report lines. A WerFault is about
    a game process when it was started by one (it is in that process's tree) or its command line names one."""
    live = table is None
    table = process_table() if live else table
    cache = _WER_COMMAND_CACHE if cache is None else cache
    owner = process_trees(pids, table, created=creation_time if live else None)
    found: dict[int, set[str]] = {root: set() for root, _ in owner.values()}
    wer = {pid: ppid for pid, (ppid, name) in table.items() if name.lower() == WER_EXE}
    for gone in [w for w in cache if w not in wer]:
        del cache[gone]  # a pid can be reused by another WerFault
    if not wer:
        return found
    todo = [w for w, ppid in wer.items() if ppid not in owner and w not in cache]
    if todo:
        got = read_command_lines(todo)
        cache.update({w: got.get(w, "") for w in todo})
    for w, ppid in wer.items():
        hits = {p for p in werfault_named_pids(cache.get(w, "")) if p in owner}
        if ppid in owner:
            hits.add(ppid)
        for p in hits:
            root, name = owner[p]
            found[root].add(f"WerFault.exe pid {w} for game pid {p}" + ("" if p == root else f" ({name} of pid {root})"))
    return found


def wer_reports(pid: int) -> set[str]:
    return wer_reports_by_pid([pid])[pid]


def check_build(game: Path) -> str:
    """H1: the sha256 of sts2.dll, or SystemExit if it is not a known build."""
    digest = sha256(game / "data_sts2_windows_x86_64" / "sts2.dll")
    allowed = set(KNOWN_BUILDS) | set(filter(None, os.environ.get(
        "SPIRELINE_REFEREE_KNOWN_SHA256", "").split(",")))
    if digest not in allowed:
        print(f"refusing: unknown sts2.dll {digest}", file=sys.stderr)
        raise SystemExit(2)
    return digest


def write_settings(scratch: Path, boot_opts: bool = True) -> None:
    """A fresh profile with every volume at 0 (FMOD plays even headless) and mod loading agreed to:
    the game skips mods until the player has seen its mods warning. Missing fields take defaults.

    boot_opts adds two boot-time savers (both read from settings.save, the profile's SettingsSave):
      skip_intro_logo: true  -> NGame.cs:679 `skipLogo` is true, so LaunchMainMenu (NGame.cs:975-996)
                                skips the logo animation, its 0.8 s fade in and the fade out.
      fps_limit: 0           -> NGame.cs:745 InitializeGraphicsPreferences sets Engine.MaxFps to it
                                (the default is 60, SettingsSave.cs:24-25); 0 is uncapped.
    Property names are the JsonPropertyName values in SettingsSave.cs (lines 24 and 82).
    """
    settings = scratch / "appdata" / "SlayTheSpire2" / "default" / "1" / "settings.save"
    settings.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": 5, "volume_master": 0.0, "volume_bgm": 0.0, "volume_sfx": 0.0,
        "volume_ambience": 0.0, "fullscreen": False,
        "mod_settings": {"mods_enabled": True, "mod_list": []},
    }
    if boot_opts:
        body.update({"skip_intro_logo": True, "fps_limit": 0})
    settings.write_text(json.dumps(body, indent=2), encoding="utf-8")


def real_profile_state():
    """Every file under the real profile folder with its mtime, to prove a session left it alone."""
    real_profile = Path(os.environ["APPDATA"]) / "SlayTheSpire2"
    return real_profile.exists() and sorted((str(p), p.stat().st_mtime) for p in real_profile.rglob("*"))


# .NET runtime settings for the game process (CoreCLR reads DOTNET_* at startup; they change how
# code is compiled and memory collected, never what the game computes). Measured on r20v2, one
# server, A/B in the same conditions:
#   DOTNET_TieredPGO=0  the game's code is huge and each fight reaches new paths, so with PGO the
#                       tiering thread was still instrumenting and recompiling 100-200 ms per fight
#                       after 20 fights; without it ~40 ms, and fight CPU fell ~8%.
DOTNET_ENV = {"DOTNET_TieredPGO": "0"}


def game_env(scratch: Path, job: Path, dotnet_tuning: bool = True) -> dict[str, str]:
    """The game's environment: scratch profile folders, the job file and (by default) DOTNET_ENV."""
    env = dict(os.environ, APPDATA=str(scratch / "appdata"), LOCALAPPDATA=str(scratch / "localappdata"),
               SPIRELINE_REFEREE_JOB=str(job.resolve()))
    if dotnet_tuning:
        env.update(DOTNET_ENV)
    return env


# Error mode of every game process (kernel32 SetErrorMode flags). A process that crashes with an
# unhandled exception normally starts WerFault.exe, Windows Error Reporting, which sends the crash
# report to Microsoft: a network connection from the game tree (addendum 1, H2; incident 2026-10-15,
# a WerFault of a crashed game process opened TCP :443 and the lab halted). With SEM_NOGPFAULTERRORBOX
# set, the unhandled-exception filter ends the process without calling WER. A child inherits its
# parent's error mode unless CreateProcess gets CREATE_DEFAULT_ERROR_MODE, which nothing here may pass.
SEM_FAILCRITICALERRORS = 0x0001
SEM_NOGPFAULTERRORBOX = 0x0002
SEM_NOOPENFILEERRORBOX = 0x8000
GAME_ERROR_MODE = SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
CREATE_DEFAULT_ERROR_MODE = 0x04000000
_ERROR_MODE_LOCK = threading.Lock()  # SetErrorMode is process-wide: one spawn at a time sets and restores it


def _kernel32():
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.SetErrorMode.restype = wintypes.UINT
    k32.SetErrorMode.argtypes = [wintypes.UINT]
    k32.GetErrorMode.restype = wintypes.UINT
    k32.GetErrorMode.argtypes = []
    return k32


@contextlib.contextmanager
def game_error_mode():
    """While the block runs, this process has GAME_ERROR_MODE on top of its own error mode, so a child
    started inside it inherits that; the launcher's own mode is put back afterwards (a crash of the
    launcher itself reports as it always did)."""
    if os.name != "nt":
        yield
        return
    with _ERROR_MODE_LOCK:
        k32 = _kernel32()
        before = k32.GetErrorMode()
        k32.SetErrorMode(before | GAME_ERROR_MODE)
        try:
            yield
        finally:
            k32.SetErrorMode(before)


def popen_no_wer(args, **kw) -> subprocess.Popen:
    """subprocess.Popen for a game process: it starts with GAME_ERROR_MODE, so a crash of it or of anything
    it starts is not handed to Windows Error Reporting. The only way to start a game process."""
    if kw.get("creationflags", 0) & CREATE_DEFAULT_ERROR_MODE:
        raise ValueError("CREATE_DEFAULT_ERROR_MODE would give the game process the default error mode back "
                         "(and its crashes to Windows Error Reporting)")
    with game_error_mode():
        return subprocess.Popen(args, **kw)


def game_command(game: Path, windowed: bool = False) -> list[str]:
    """The game's command line: the one place it is written (launch here, envproto.GameEnv through popen_no_wer)."""
    return [str(game / "SlayTheSpire2.exe")] + ([] if windowed else ["--headless"]) + ["--force-steam", "off"]


def launch(game: Path, scratch: Path, job: Path, below_normal: bool = False, dotnet_tuning: bool = True,
           windowed: bool = False, command: list[str] | None = None, extra_args: list[str] | None = None):
    """Start the game headless (or, with windowed, in a normal window) on a scratch profile;
    returns (process, log file). CREATE_NO_WINDOW only hides a console; the game's window still shows.
    The process gets GAME_ERROR_MODE (popen_no_wer). `command` replaces the game's command line: tests only.
    `extra_args` go last on the command line (engine flags: serve.py's `--fixed-fps 60`, serve.engine_args)."""
    env = game_env(scratch, job, dotnet_tuning)
    log = open(scratch / "game-stdout.log", "wb")
    flags = subprocess.CREATE_NO_WINDOW | (subprocess.BELOW_NORMAL_PRIORITY_CLASS if below_normal else 0)
    args = command or game_command(game, windowed)
    args = args + list(extra_args or [])
    proc = popen_no_wer(args, cwd=game, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    return proc, log


def install(game: Path, build: Path) -> None:
    """override.cfg (blank Sentry DSN) and the driver mod, copied only when they changed."""
    cfg = game / "override.cfg"
    if not cfg.exists() or cfg.read_text(encoding="utf-8") != OVERRIDE:
        cfg.write_text(OVERRIDE, encoding="utf-8")
    mod = game / "mods" / "SpirelineReferee"
    mod.mkdir(parents=True, exist_ok=True)
    # Parallel sessions share the game copy: copy the mod only when it changed, since a
    # running game holds the loaded DLL open.
    for src, dst in ((build / "SpirelineReferee.dll", mod / "SpirelineReferee.dll"),
                     (HERE / "driver" / "SpirelineReferee.json", mod / "SpirelineReferee.json")):
        if not dst.exists() or sha256(src) != sha256(dst):
            shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--game", type=Path, required=True)
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--build", type=Path, default=HERE / "driver" / "bin" / "Release" / "net9.0")
    ap.add_argument("--no-install", action="store_true", help="the caller already installed the mod and override.cfg")
    ap.add_argument("--legacy-boot-settings", action="store_true",
                    help="do not write skip_intro_logo / fps_limit into settings.save (to time the old boot)")
    ap.add_argument("--below-normal", action="store_true", help="run the game at BelowNormal priority")
    ap.add_argument("--no-dotnet-tuning", action="store_true", help="start the game without DOTNET_ENV")
    ap.add_argument("--windowed", action="store_true",
                    help="launch in a window (no --headless), to check headless changes nothing; every other "
                         "safety step is the same (sound off, fullscreen off, scratch profile, no network)")
    a = ap.parse_args()

    game = a.game.resolve()
    digest = check_build(game)

    scratch = (a.scratch or game.parent / "scratch" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    (scratch / "appdata").mkdir(parents=True, exist_ok=True)
    (scratch / "localappdata").mkdir(parents=True, exist_ok=True)
    if not a.no_install:
        install(game, a.build)
    # A job with "normal_settings" keeps the profile's defaults: no skip_intro_logo, no fps_limit 0.
    normal = json.loads(a.job.read_text(encoding="utf-8")).get("normal_settings") is True
    write_settings(scratch, boot_opts=not (a.legacy_boot_settings or normal))

    before = real_profile_state()
    t0 = time.time()
    proc, _log = launch(game, scratch, a.job, below_normal=a.below_normal, dotnet_tuning=not a.no_dotnet_tuning,
                        windowed=a.windowed)
    seen: set[str] = set()
    wer: set[str] = set()
    while proc.poll() is None and time.time() - t0 < a.timeout:
        seen |= connections(proc.pid)
        wer |= wer_reports(proc.pid)
        time.sleep(0.5)
    if proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
        rc = "timeout"
    else:
        rc = proc.returncode
    seen |= connections(proc.pid)  # a child the game started (crashpad_handler) can outlive it
    wer |= wer_reports(proc.pid)
    for line in sorted(wer):
        print(f"WER: {line}: Windows Error Reporting started for a game process", file=sys.stderr)
    after = real_profile_state()
    report = {
        "exit": rc, "seconds": round(time.time() - t0, 1), "sts2_dll_sha256": digest,
        "network_connections": sorted(seen), "wer_reports": sorted(wer), "real_profile_touched": before != after,
        "scratch": scratch.name, **({"windowed": True} if a.windowed else {}),
    }
    (scratch / "session.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0 if rc == 0 and not seen and before == after else 1


if __name__ == "__main__":
    sys.exit(main())
