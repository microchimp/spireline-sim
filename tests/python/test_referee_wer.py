"""referee/run_referee.py: a game process starts with an error mode that keeps Windows Error Reporting away.

Incident 2026-10-15 (addendum 1, H2): a game process crashed, Windows started WerFault.exe for it, WerFault
opened TCP :443 to Microsoft to send the crash report, and the referee's network check halted the lab. The
fix is in our launcher only: `launch` starts the game with SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX |
SEM_NOOPENFILEERRORBOX (inherited from the launcher for the length of the spawn, then put back), and nothing
may pass CREATE_DEFAULT_ERROR_MODE. The second line of defence lists any WerFault.exe about a game tree.

These tests never crash a process in the default error mode. The one that crashes a child on purpose does it
through `launch`, and the child checks its own error mode first: if the mode is not there it exits with 99
instead of crashing, so a regression of the fix cannot send a crash report.
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))

import run_referee as rr  # noqa: E402
import serve  # noqa: E402

FAKE = Path(__file__).resolve().parent / "fixtures" / "fake_fight_server.py"
# A venv python.exe on Windows is a launcher that starts the real interpreter as a child; the base
# interpreter is the process `launch` starts, so its pid and error mode are the ones the test sees.
PYTHON = getattr(sys, "_base_executable", sys.executable)
WINDOWS = os.name == "nt"
ACCESS_VIOLATION = 0xC0000005
POLL_S = float(os.environ.get("SPIRELINE_WER_POLL_S", "20"))

# The child of the crash test: prints its own error mode, refuses to crash without it, then reads address 0
# (ctypes.c_int.from_address does no SEH-guarded call, so this is a real access violation, not an OSError).
CRASH_CHILD = """
import ctypes, sys
mode = ctypes.WinDLL("kernel32").GetErrorMode()
print("errormode", mode, flush=True)
if mode & 0x0002 == 0:
    sys.exit(99)
ctypes.c_int.from_address(0).value
sys.exit(7)
"""


def get_error_mode() -> int:
    import ctypes

    return ctypes.WinDLL("kernel32").GetErrorMode()


@contextlib.contextmanager
def default_error_mode():
    """This test process with the error mode a lab runner has (0), whatever the harness that started the tests
    set: the fix must not depend on what the launcher inherited. Only the test process; a game process is
    still started by `launch`."""
    import ctypes

    k32 = ctypes.WinDLL("kernel32")
    before = k32.GetErrorMode()
    k32.SetErrorMode(0)
    try:
        yield
    finally:
        k32.SetErrorMode(before)


class LaunchFlagsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "job.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def launch_recorded(self, **kw):
        seen = {}

        def popen(args, **popen_kw):
            seen["args"], seen["kw"] = args, popen_kw
            seen["mode"] = get_error_mode() if WINDOWS else None
            return mock.Mock(pid=1)

        with mock.patch.object(rr.subprocess, "Popen", popen):
            proc, log = rr.launch(self.dir, self.dir, self.dir / "job.json", **kw)
        log.close()
        return seen

    @unittest.skipUnless(WINDOWS, "Windows error modes")
    def test_the_game_is_spawned_inside_the_error_mode_and_the_launchers_own_is_put_back(self):
        with default_error_mode():
            seen = self.launch_recorded(below_normal=True)
            self.assertEqual(seen["mode"], rr.GAME_ERROR_MODE)
            self.assertEqual(get_error_mode(), 0)
        with default_error_mode():  # a launcher that already has some bits keeps them
            import ctypes

            ctypes.WinDLL("kernel32").SetErrorMode(0x0004)
            self.assertEqual(self.launch_recorded()["mode"], rr.GAME_ERROR_MODE | 0x0004)
            self.assertEqual(get_error_mode(), 0x0004)

    @unittest.skipUnless(WINDOWS, "Windows error modes")
    def test_the_error_mode_is_put_back_when_the_spawn_fails(self):
        with default_error_mode(), mock.patch.object(rr.subprocess, "Popen", side_effect=OSError("no such file")):
            with self.assertRaises(OSError):
                rr.popen_no_wer(["nothing"])
            self.assertEqual(get_error_mode(), 0)

    def test_no_launch_passes_create_default_error_mode(self):
        for kw in ({}, {"below_normal": True}, {"windowed": True}):
            flags = self.launch_recorded(**kw)["kw"]["creationflags"]
            self.assertEqual(flags & rr.CREATE_DEFAULT_ERROR_MODE, 0, kw)
            self.assertTrue(flags & subprocess.CREATE_NO_WINDOW)
        with self.assertRaises(ValueError):
            rr.popen_no_wer(["nothing"], creationflags=rr.CREATE_DEFAULT_ERROR_MODE | subprocess.CREATE_NO_WINDOW)

    def test_the_game_command_is_unchanged(self):
        args = self.launch_recorded()["args"]
        self.assertEqual([Path(args[0]).name, *args[1:]], ["SlayTheSpire2.exe", "--headless", "--force-steam", "off"])
        self.assertNotIn("--headless", self.launch_recorded(windowed=True)["args"])

    def test_nothing_else_in_the_referee_or_the_lab_starts_the_game_or_asks_for_the_default_error_mode(self):
        """One launch site (run_referee.launch -> popen_no_wer); serve.py reaches it through its `launch` argument."""
        offenders = []
        for path in [*(ROOT / "referee").glob("*.py"), *(ROOT / "tools").rglob("*.py")]:
            text = path.read_text(encoding="utf-8")
            if path.name != "run_referee.py" and re.search(r"CREATE_DEFAULT_ERROR_MODE|0x0?4000000\b|SlayTheSpire2\.exe", text):
                offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [])
        src = (ROOT / "referee" / "run_referee.py").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"subprocess\.Popen\(", src)), 1)  # the one inside popen_no_wer
        self.assertIs(serve.Pool.__init__.__kwdefaults__["launch"], rr.launch)


class WerReportTest(unittest.TestCase):
    GAME, CRASHPAD, OTHER_GAME = 100, 101, 300
    WER_CHILD, WER_NAMING, WER_OTHER, WER_UNREADABLE = 400, 401, 402, 403
    TABLE = {
        GAME: (50, "SlayTheSpire2.exe"),
        CRASHPAD: (GAME, "crashpad_handler.exe"),
        OTHER_GAME: (50, "SlayTheSpire2.exe"),
        WER_CHILD: (GAME, "WerFault.exe"),  # started by the game process itself
        WER_NAMING: (900, "WerFault.exe"),  # started by the Windows service: only its command line names the pid
        WER_OTHER: (901, "WerFault.exe"),  # about some other program
        WER_UNREADABLE: (902, "WerFault.exe"),  # command line could not be read
    }
    LINES = {
        WER_NAMING: r'C:\Windows\system32\WerFault.exe -pss -s 524 -p 101 -ip 101',
        WER_OTHER: r"C:\Windows\system32\WerFault.exe -u -p 5555 -s 88",
    }

    def reader(self, calls):
        def read(pids):
            calls.append(sorted(pids))
            return {p: self.LINES[p] for p in pids if p in self.LINES}

        return read

    def test_command_lines_name_the_crashed_pid(self):
        named = rr.werfault_named_pids
        self.assertEqual(named(r'"C:\Windows\system32\WerFault.exe" -u -p 1234 -s 4567'), {1234})
        self.assertEqual(named("WerFault.exe -pss -s 524 -p 99 -ip 77 -x 5"), {99, 77})
        self.assertEqual(named("WerFault.exe -pss -s 5"), set())
        self.assertEqual(named(""), set())

    def test_a_werfault_of_a_game_tree_is_reported_under_its_root(self):
        calls: list = []
        got = rr.wer_reports_by_pid([self.GAME, self.OTHER_GAME], self.TABLE, self.reader(calls), cache={})
        self.assertEqual(got[self.GAME], {f"WerFault.exe pid {self.WER_CHILD} for game pid {self.GAME}",
                                          f"WerFault.exe pid {self.WER_NAMING} for game pid {self.CRASHPAD} "
                                          f"(crashpad_handler.exe of pid {self.GAME})"})
        self.assertEqual(got[self.OTHER_GAME], set())
        # the child of the game needs no command line; the others are read in one call
        self.assertEqual(calls, [[self.WER_NAMING, self.WER_OTHER, self.WER_UNREADABLE]])

    def test_a_game_that_has_exited_keeps_its_reports_and_command_lines_are_read_once(self):
        table = {k: v for k, v in self.TABLE.items() if k not in (self.GAME, self.CRASHPAD, self.WER_CHILD)}
        table[self.WER_NAMING] = (900, "WerFault.exe")
        calls: list = []
        cache: dict = {}
        line = {self.WER_NAMING: r"WerFault.exe -u -p 100 -s 1"}
        read = lambda pids: (calls.append(sorted(pids)), {p: line[p] for p in pids if p in line})[1]  # noqa: E731
        for _ in range(3):
            got = rr.wer_reports_by_pid([self.GAME], table, read, cache)
            self.assertEqual(got[self.GAME], {f"WerFault.exe pid {self.WER_NAMING} for game pid {self.GAME}"})
        self.assertEqual(len(calls), 1)
        table.pop(self.WER_NAMING)  # the WerFault is gone: its cache entry goes too (the pid can be reused)
        self.assertEqual(rr.wer_reports_by_pid([self.GAME], table, read, cache)[self.GAME], set())
        self.assertNotIn(self.WER_NAMING, cache)

    def test_no_werfault_means_no_command_line_is_read(self):
        table = {self.GAME: (50, "SlayTheSpire2.exe")}
        got = rr.wer_reports_by_pid([self.GAME], table, lambda pids: self.fail("read"), cache={})
        self.assertEqual(got, {self.GAME: set()})

    @unittest.skipUnless(WINDOWS, "WMI")
    def test_command_lines_reads_a_live_process(self):
        lines = rr.command_lines([os.getpid(), 2**22 + 3])
        self.assertIn(os.getpid(), lines)
        self.assertNotIn(2**22 + 3, lines)
        self.assertIn("python", lines[os.getpid()].lower())


class PoolWerTest(unittest.TestCase):
    """serve.Pool: a WerFault about a server is in the summary and the log; the network verdict is untouched."""

    def pool(self, tmp: Path, wer):
        def fake_launch(game, scratch, job, below_normal=False, dotnet_tuning=True):
            log = open(scratch / "game-stdout.log", "wb")
            return subprocess.Popen([PYTHON, str(FAKE), str(job)], stdout=log, stderr=subprocess.STDOUT), log

        return serve.Pool(tmp / "game", tmp / "run", servers=1, spares=0, launch=fake_launch, echo=None,
                          connections=lambda pids: {int(p): set() for p in pids}, profile_state=lambda: [],
                          check_build=lambda game: "fake-digest", kill=lambda s: s.proc.kill(), wer_reports=wer)

    def run_one(self, wer):
        with tempfile.TemporaryDirectory() as t:
            p = self.pool(Path(t), wer)
            try:
                p.submit([{"kind": "combat", "seed": "SEED000001", "n_steps": 1, "fake": {"sleep": 0.3}}])
                t0 = time.time()
                while p.outstanding and time.time() - t0 < 60 and not p.failed:
                    p.poll()
                    time.sleep(0.02)
                p.close()
                return p.summary(), list(p.events), p.seen_net
            finally:
                p.close(abandon=True)

    def test_a_werfault_is_reported_once_and_does_not_count_as_a_connection(self):
        report = "WerFault.exe pid 4242 for game pid 1"
        summary, events, seen_net = self.run_one(lambda pids: {int(p): {report} for p in pids})
        self.assertEqual(summary["wer_reports"], [report])
        self.assertEqual(summary["network_connections"], [])
        self.assertEqual(seen_net, set())
        self.assertEqual(sum(report in e for e in events), 1)

    def test_no_werfault_gives_an_empty_list(self):
        summary, events, _ = self.run_one(lambda pids: {int(p): set() for p in pids})
        self.assertEqual(summary["wer_reports"], [])
        self.assertFalse([e for e in events if "WER" in e])


@unittest.skipUnless(WINDOWS, "Windows Error Reporting")
class CrashTest(unittest.TestCase):
    def test_a_crashing_game_process_never_reaches_windows_error_reporting(self):
        with tempfile.TemporaryDirectory() as t, default_error_mode():
            tmp = Path(t)
            (tmp / "job.json").write_text("{}", encoding="utf-8")
            t0 = time.time()
            proc, log = rr.launch(tmp, tmp, tmp / "job.json", below_normal=True, command=[PYTHON, "-c", CRASH_CHILD])
            pid = proc.pid
            self.assertEqual(get_error_mode(), 0, "the launcher's own error mode is put back")
            connections: set[str] = set()
            wer: set[str] = set()
            children: set[int] = set()
            exit_s = None
            while time.time() - t0 < POLL_S:
                table = rr.process_table()
                children |= {w for w, (pp, name) in table.items() if name.lower() == "werfault.exe" and pp == pid}
                wer |= rr.wer_reports_by_pid([pid], table)[pid]
                if exit_s is None:
                    connections |= rr.connections_by_pid([pid])[pid]  # while the child runs (netstat takes a moment)
                    if proc.poll() is not None:
                        exit_s = time.time() - t0
                        connections |= rr.connections_by_pid([pid])[pid]  # and once more after it is gone
                time.sleep(0.2)
            connections |= rr.connections_by_pid([pid])[pid]
            log.close()
            printed = (tmp / "game-stdout.log").read_text(encoding="utf-8", errors="replace")
            mode = int(re.search(r"errormode (\d+)", printed).group(1))
            code = proc.returncode & 0xFFFFFFFF

        self.assertNotEqual(code, 99, "the child did not inherit the error mode and so did not crash")
        self.assertEqual(code, ACCESS_VIOLATION, hex(code))
        self.assertEqual(mode & rr.GAME_ERROR_MODE, rr.GAME_ERROR_MODE, hex(mode))
        self.assertIsNotNone(exit_s)
        self.assertLess(exit_s, 10, "a crash without WER ends the process at once")
        self.assertEqual(children, set())
        self.assertEqual(wer, set())
        self.assertEqual(connections, set())
        print(f"\ncrash test: exit {hex(code)} after {exit_s:.1f} s, child error mode {hex(mode)}, WerFault seen "
              f"{sorted(wer | children) or 'no'}, connections {sorted(connections)}, polled {POLL_S:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
