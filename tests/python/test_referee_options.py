"""referee/serve.py and run_referee.py: the job keys and environment every game server starts with.

The speed patch sets (fast_patches, silent, still_visuals) are on by default and each can be turned off
alone for an A/B run; the .NET runtime settings go into the game's environment unless switched off.
"""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))

import run_referee  # noqa: E402
import serve  # noqa: E402


def args(**kw) -> argparse.Namespace:
    base = dict(slow_reset=False, preload=False, no_patches=False, stall_s=None, idle_frames=None, job_opt=[])
    base.update(kw)
    return argparse.Namespace(**base)


class JobOptionsTest(unittest.TestCase):
    def test_patch_sets_on_by_default(self):
        o = serve.job_options({"runs": []}, args())
        for key in ("fast_patches", "fast_reset", "no_preload"):
            self.assertIs(o[key], True, key)

    def test_no_patches_turns_every_patch_set_off(self):
        o = serve.job_options([], args(no_patches=True))
        self.assertIs(o["fast_patches"], False)

    def test_job_opt_overrides_one_default(self):
        o = serve.job_options({}, args(job_opt=["fast_patches=false", "profile=true", 'exp=["spine"]']))
        self.assertIs(o["fast_patches"], False)
        self.assertIs(o["profile"], True)
        self.assertEqual(o["exp"], ["spine"])

    def test_specs_file_keys_and_flags(self):
        o = serve.job_options({"idle_frames": 8, "stall_s": 30, "out": "x"}, args(stall_s=40))
        self.assertEqual((o["idle_frames"], o["stall_s"]), (8, 40))
        self.assertNotIn("out", o)


class GameEnvTest(unittest.TestCase):
    def test_dotnet_settings_and_scratch_profile(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in run_referee.DOTNET_ENV:
                os.environ.pop(k, None)
            scratch = Path("C:/scratch/s00")
            env = run_referee.game_env(scratch, Path("job.json"))
            self.assertEqual(env["DOTNET_TieredPGO"], "0")
            self.assertEqual(env["APPDATA"], str(scratch / "appdata"))
            self.assertTrue(env["SPIRELINE_REFEREE_JOB"].endswith("job.json"))
            plain = run_referee.game_env(scratch, Path("job.json"), dotnet_tuning=False)
            for k in run_referee.DOTNET_ENV:
                self.assertNotIn(k, plain)


if __name__ == "__main__":
    unittest.main()
