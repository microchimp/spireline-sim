"""rl/quiet_lock.py (the QUIET.lock protocol shared by the nightly driver and `gen0.py score --quiet-lock`) and the daytime score's use
of it: the line is written, rewritten and removed; the wait rule; the fight loop yields to it and its own line ages out. No game, no
emulator: `gen0.play_score` is replaced."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
import unittest.mock as mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "rl", ROOT / "src"):
    sys.path.insert(0, str(p))

import fight_gym as fg  # noqa: E402
import gen0  # noqa: E402
import gym_loop as gl  # noqa: E402
import quiet_lock as ql  # noqa: E402

LINE = r"^lab-score \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ pid=%d\n$"


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.f = self.tmp / "QUIET.lock"

    def lock(self, now: str = "2026-11-02T12:00:00Z", **kw):
        t = [ql.parse_utc(now)]
        slept: list[float] = []

        def sleep(s):
            slept.append(s)
            t[0] += s

        lk = ql.QuietLock(self.f, "lab-score", clock=lambda: t[0], sleep=sleep, poll_s=60, **kw)
        return lk, t, slept


class LineTest(Tmp):
    def test_the_line_is_written_with_the_tag_and_our_pid_rewritten_by_the_heartbeat_and_removed(self):
        lk, t, _ = self.lock()
        self.assertTrue(lk.acquire())
        self.assertRegex(self.f.read_text(), LINE % os.getpid())
        self.assertIn(" 2026-11-02T12:00:00Z ", self.f.read_text())
        lk.start_heartbeat(every_s=0.05)
        t[0] += 900
        for _ in range(100):  # the heartbeat thread rewrites the line with the clock's new time
            if "2026-11-02T12:15:00Z" in self.f.read_text():
                break
            time.sleep(0.05)
        self.assertRegex(self.f.read_text(), LINE % os.getpid())
        self.assertIn("2026-11-02T12:15:00Z", self.f.read_text())
        lk.release()
        self.assertFalse(self.f.exists())
        time.sleep(0.2)
        self.assertFalse(self.f.exists())  # the heartbeat stopped with the release: it does not bring the line back

    def test_other_live_lines_are_kept_through_a_write_and_a_release_and_stale_ones_are_dropped(self):
        self.f.write_text("old-gate 2026-11-02T08:00:00Z pid=1\nw3-blocks 2026-11-02T11:30:00Z pid=2\n")  # the first is 4 h old
        lk, _, _ = self.lock()
        self.assertTrue(lk.acquire())
        lines = self.f.read_text().splitlines()
        self.assertTrue(lines[0].startswith("lab-score "))
        self.assertEqual(["w3-blocks 2026-11-02T11:30:00Z pid=2"], lines[1:])
        lk._write()  # a heartbeat
        self.assertEqual(2, len(self.f.read_text().splitlines()))
        lk.release()
        self.assertEqual(["w3-blocks 2026-11-02T11:30:00Z pid=2"], self.f.read_text().splitlines())

    def test_release_never_removes_another_holders_line(self):
        lk, _, _ = self.lock()
        self.f.write_text("w4-y 2026-11-02T11:59:00Z pid=3\n")
        lk.release()
        self.assertEqual("w4-y 2026-11-02T11:59:00Z pid=3\n", self.f.read_text())


class WaitTest(Tmp):
    def test_waits_while_another_is_under_two_hours_old_then_gives_up(self):
        self.f.write_text("w4-readout 2026-11-02T11:30:00Z pid=1\n")
        lk, t, slept = self.lock(wait_h=0.5)
        self.assertFalse(lk.acquire())
        self.assertAlmostEqual(1800, sum(slept), delta=60)  # polled for the whole wait
        self.assertEqual("w4-readout 2026-11-02T11:30:00Z pid=1\n", self.f.read_text())  # the other's line is untouched
        self.assertIn("w4-readout", lk.blocked_by())

    def test_the_fight_loop_drops_its_line_between_steps_and_the_score_takes_the_lock_before_the_loop_starts_again(self):
        with mock.patch.object(fg, "QUIET", self.f):
            gl.hold_quiet(True)  # a step is running: the loop's line is fresh and first
            lk = ql.QuietLock(self.f, "lab-score", poll_s=60, wait_h=2.0, sleep=lambda s: gl.hold_quiet(False))  # the step ends in the first wait
            self.assertEqual("gym-loop", lk.holder()[0])
            self.assertIn("gym-loop", lk.blocked_by())
            self.assertTrue(lk.acquire())
            self.assertTrue(self.f.read_text().startswith("lab-score "))
            # the loop's next start sees the line (it waits --yield-s, 150 s, longer than the 60 s poll, so the score got in first)
            self.assertEqual("quiet:lab-score", gl.why_blocked(self.tmp / "loop", starting=True))
            self.assertEqual("quiet:lab-score", gl.why_blocked(self.tmp / "loop", starting=False))  # a running step stops at its next poll
            lk.release()
            self.assertFalse(self.f.exists())
            self.assertIsNone(gl.other_quiet())

    def test_the_loops_line_is_rewritten_while_it_steps_and_ages_out_when_it_dies(self):
        with mock.patch.object(fg, "QUIET", self.f):
            old = "gym-loop 2026-11-02T10:10:00Z pid=99"  # 1 h 50 min old at 12:00
            self.f.write_text(old + "\n")
            gl.hold_quiet(True)  # the step's stop poll rewrites its line (every 30 minutes): it is fresh again
            self.assertNotEqual(old + "\n", self.f.read_text())
            self.assertIn("gym-loop", ql.QuietLock(self.f, "lab-score", wait_h=2.0).blocked_by())
            # a loop that died with its line: the line is not rewritten and the rule frees the lock 2 h after its stamp
            self.f.write_text(old + "\n")
            lk, t, slept = self.lock(now="2026-11-02T12:00:00Z", wait_h=2.0)
            self.assertTrue(lk.acquire())
            self.assertAlmostEqual(600, sum(slept), delta=60)  # 10 minutes until the line reads 2 h old
            self.assertTrue(self.f.read_text().startswith("lab-score "))
            self.assertNotIn("gym-loop", self.f.read_text())


class ScoreTest(Tmp):
    def args(self, *extra):
        return gen0.build_parser().parse_args(["score", "--game", "G", "--seeds-file", "F", "--out", str(self.tmp / "out"), *extra])

    def test_the_flag_defaults_to_no_lock_and_the_tag_to_lab_score(self):
        a = self.args()
        self.assertEqual((None, "lab-score", 2.0), (a.quiet_lock, a.quiet_tag, a.quiet_wait_h))
        with mock.patch.object(gen0, "play_score", return_value=0) as play:
            self.assertEqual(0, gen0.score_real(a))
        play.assert_called_once_with(a)
        self.assertFalse(self.f.exists())

    def test_the_score_holds_its_line_while_it_plays_and_removes_it_after_however_it_ends(self):
        a = self.args("--quiet-lock", str(self.f))
        seen = []

        def play(args):
            seen.append(self.f.read_text())
            return 7

        with mock.patch.object(gen0, "play_score", side_effect=play):
            self.assertEqual(7, gen0.score_real(a))
        self.assertRegex(seen[0], LINE % os.getpid())  # taken before the first process of either arm
        self.assertFalse(self.f.exists())
        with mock.patch.object(gen0, "play_score", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                gen0.score_real(a)
        self.assertFalse(self.f.exists())

    def test_the_score_keeps_to_the_wait_rule_and_plays_nothing_when_the_lock_stays_taken(self):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.f.write_text(f"w4-readout {now} pid=1\n")
        a = self.args("--quiet-lock", str(self.f), "--quiet-wait-h", "0")
        with mock.patch.object(gen0, "play_score") as play:
            self.assertEqual(gen0.QUIET_SKIPPED_RC, gen0.score_real(a))
        play.assert_not_called()
        self.assertIn("w4-readout", json.loads((self.tmp / "out" / "result.json").read_text())["skipped"])
        self.assertEqual(f"w4-readout {now} pid=1\n", self.f.read_text())

    def test_the_score_takes_a_lock_whose_holder_is_stale_or_ours(self):
        for line in ("w4-readout 2026-01-01T00:00:00Z pid=1", f"lab-score {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} pid=2"):
            self.f.write_text(line + "\n")
            with mock.patch.object(gen0, "play_score", return_value=0) as play:
                self.assertEqual(0, gen0.score_real(self.args("--quiet-lock", str(self.f), "--quiet-wait-h", "0")), line)
            play.assert_called_once()
            self.assertFalse(self.f.exists())


if __name__ == "__main__":
    unittest.main()
