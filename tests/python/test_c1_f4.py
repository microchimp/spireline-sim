"""F4: a non-Ironclad emulator start exits at once, before any worker, when the native library cannot start it."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT))

import envproto  # noqa: E402
import gen0  # noqa: E402
import lockstep  # noqa: E402


class CharacterSupport(unittest.TestCase):
    def test_support_is_judged_by_the_native_binding(self):
        self.assertIsNone(lockstep.character_supported("IRONCLAD", NS()))
        why = lockstep.character_supported("SILENT", NS())
        self.assertIn("run_reset_character", why)
        self.assertIn("SILENT", why)
        self.assertIsNone(lockstep.character_supported("SILENT", NS(run_reset_character=lambda *a: 0)))
        with self.assertRaises(SystemExit) as cm:
            lockstep.require_character("DEFECT", NS())
        self.assertIn("run_reset_character", str(cm.exception))
        lockstep.require_character("IRONCLAD", NS())  # no exit

    def test_play_and_gen0_exit_before_any_worker(self):
        msg = "--character SILENT: the native emulator library has no run_reset_character"
        with mock.patch.object(lockstep, "character_supported", lambda c, n=None: None if c == "IRONCLAD" else msg), \
                mock.patch.object(envproto.mp, "get_context", side_effect=AssertionError("a worker pool was started")), \
                mock.patch.object(gen0, "score_real", side_effect=AssertionError("score ran")), \
                tempfile.TemporaryDirectory() as d:
            a = envproto.parser().parse_args(["play", "--game", "g", "--out", d, "--character", "SILENT"])
            with self.assertRaises(SystemExit) as cm:
                envproto.play(a)
            self.assertEqual(str(cm.exception), msg)
            for cmd in (["score", "--game", "g", "--out", d, "--seeds-file", "s"], ["emu", "--seeds-file", "s", "--out", d]):
                with self.assertRaises(SystemExit) as cm:
                    gen0.main(cmd + ["--character", "SILENT"])
                self.assertEqual(str(cm.exception), msg)


if __name__ == "__main__":
    unittest.main()
