"""referee/run_referee.py: the network check covers the game process and every process it starts.

The game starts crashpad_handler.exe (Sentry's crash reporter) as its child; its connections count
as the game's. Pure functions only: a process table and a netstat listing in the shapes Windows gives.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))

import run_referee as rr  # noqa: E402

GAME, CRASHPAD, GRANDCHILD, OTHER, OTHER_CHILD = 100, 200, 300, 400, 500
TABLE = {
    1: (0, "System"),
    GAME: (50, "SlayTheSpire2.exe"),
    CRASHPAD: (GAME, "crashpad_handler.exe"),
    GRANDCHILD: (CRASHPAD, "WerFault.exe"),
    OTHER: (50, "SlayTheSpire2.exe"),
    OTHER_CHILD: (OTHER, "crashpad_handler.exe"),
    600: (1, "svchost.exe"),
}
NETSTAT = """
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       600
  TCP    10.0.0.2:50001         34.120.1.1:443         ESTABLISHED     200
  TCP    10.0.0.2:50002         34.120.1.2:443         SYN_SENT        300
  UDP    0.0.0.0:5353           *:*                                    100
  TCP    10.0.0.2:50003         34.120.1.3:443         ESTABLISHED     500
  TCP    [::]:445               [::]:0                 LISTENING       4
"""


class ProcessTreeTests(unittest.TestCase):
    def test_children_and_grandchildren_belong_to_their_root(self):
        owner = rr.process_trees([GAME, OTHER], TABLE)
        self.assertEqual(owner[CRASHPAD], (GAME, "crashpad_handler.exe"))
        self.assertEqual(owner[GRANDCHILD], (GAME, "WerFault.exe"))
        self.assertEqual(owner[OTHER_CHILD], (OTHER, "crashpad_handler.exe"))
        self.assertNotIn(600, owner)

    def test_an_exited_game_still_owns_its_lingering_child(self):
        table = {pid: v for pid, v in TABLE.items() if pid != GAME}
        owner = rr.process_trees([GAME], table)
        self.assertEqual(owner[GAME], (GAME, ""))
        self.assertEqual(owner[CRASHPAD][0], GAME)

    def test_a_reused_parent_pid_cycle_ends(self):
        table = {7: (8, "a.exe"), 8: (7, "b.exe")}
        self.assertEqual(set(rr.process_trees([7], table)), {7, 8})

    # 2026-10-16: Discord (running since 13 Oct) named a parent pid that a game server had been given since; it was
    # counted as the game's child and halted the lab. A child older than its parent is not its child.
    APP, APP_HELPER = 700, 800

    def reused(self):
        table = dict(TABLE)
        table[self.APP] = (GAME, "Discord.exe")          # its real parent (an updater) had pid GAME, exited days ago
        table[self.APP_HELPER] = (self.APP, "Discord.exe")
        born = {GAME: 1000, CRASHPAD: 1001, GRANDCHILD: 1002, OTHER: 1000, OTHER_CHILD: 1003,
                self.APP: 10, self.APP_HELPER: 11}
        return table, born

    def test_a_process_older_than_the_game_is_not_its_child(self):
        table, born = self.reused()
        owner = rr.process_trees([GAME], table, created=born.get)
        self.assertNotIn(self.APP, owner)
        self.assertNotIn(self.APP_HELPER, owner)
        self.assertEqual(owner[CRASHPAD], (GAME, "crashpad_handler.exe"))  # real children are still found
        self.assertEqual(owner[GRANDCHILD], (GAME, "WerFault.exe"))

    def test_an_unreadable_time_keeps_the_child(self):
        table, born = self.reused()
        born[self.APP] = None  # no access: never weaker than before
        self.assertIn(self.APP, rr.process_trees([GAME], table, created=born.get))

    def test_without_times_nothing_changes(self):
        table, _ = self.reused()
        self.assertIn(self.APP, rr.process_trees([GAME], table))

    def test_the_live_creation_time_reads_this_process(self):
        import os
        here = rr.creation_time(os.getpid())
        self.assertIsNotNone(here)
        self.assertGreater(here, 0)
        self.assertIsNone(rr.creation_time(0xFFFFFFF0))  # no such process


class NetstatTests(unittest.TestCase):
    def test_child_connections_count_for_the_game(self):
        found = rr.parse_netstat(NETSTAT, rr.process_trees([GAME, OTHER], TABLE))
        self.assertEqual(found[GAME], {
            "UDP 0.0.0.0:5353 *:*",
            "TCP 10.0.0.2:50001 34.120.1.1:443 ESTABLISHED (crashpad_handler.exe pid 200)",
            "TCP 10.0.0.2:50002 34.120.1.2:443 SYN_SENT (WerFault.exe pid 300)",
        })
        self.assertEqual(found[OTHER], {"TCP 10.0.0.2:50003 34.120.1.3:443 ESTABLISHED (crashpad_handler.exe pid 500)"})

    def test_unrelated_processes_are_ignored(self):
        found = rr.parse_netstat(NETSTAT, rr.process_trees([GAME], {GAME: (50, "SlayTheSpire2.exe")}))
        self.assertEqual(found, {GAME: {"UDP 0.0.0.0:5353 *:*"}})


if __name__ == "__main__":
    unittest.main()
