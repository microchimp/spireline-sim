"""referee/serve.py and run_referee.py: the job keys and environment every game server starts with.

The speed patch sets (fast_patches, silent, still_visuals) are on by default and each can be turned off
alone for an A/B run; the settle defaults (SPEED_PLAN steps 3-4) are serve.py's and batch.py's alike;
the .NET runtime settings go into the game's environment unless switched off.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))

import run_referee  # noqa: E402
import serve  # noqa: E402


class JobOptionsTest(unittest.TestCase):
    def test_patch_sets_on_by_default(self):
        o = serve.job_options({"runs": []})
        for key in ("fast_patches", "silent", "still_visuals", "fast_reset", "no_preload"):
            self.assertIs(o[key], True, key)

    def test_no_patches_turns_every_patch_set_off(self):
        o = serve.job_options([], no_patches=True)
        self.assertEqual([o["fast_patches"], o["silent"], o["still_visuals"]], [False] * 3)

    def test_job_opt_overrides_one_default(self):
        o = serve.job_options({}, job_opt=["silent=false", "profile=true", 'exp=["spine"]', 'settle="frames"'])
        self.assertIs(o["silent"], False)
        self.assertIs(o["still_visuals"], True)
        self.assertIs(o["profile"], True)
        self.assertEqual(o["exp"], ["spine"])
        self.assertEqual(o["settle"], "frames")

    def test_specs_file_keys_and_flags(self):
        o = serve.job_options({"idle_frames": 8, "stall_s": 30, "idle_timeout_s": 15, "out": "x"}, stall_s=40)
        self.assertEqual((o["idle_frames"], o["stall_s"], o["idle_timeout_s"]), (8, 40, 15))
        self.assertNotIn("out", o)

    def test_settle_defaults_and_the_old_settle(self):
        o = serve.job_options(None)
        self.assertEqual((o["settle"], o["fast_tweens"], o["sleep_usec"], o["settle_audit"]), ("flags", True, 0, 0.05))
        self.assertEqual(serve.job_options({"settle_audit": 0.5})["settle_audit"], 0.5)  # the specs file wins
        old = serve.job_options({"sleep_usec": 0, "settle_audit": 0.5}, old_settle=True)
        self.assertEqual((old["settle"], old["fast_tweens"]), ("frames", False))
        self.assertNotIn("sleep_usec", old)
        self.assertNotIn("settle_audit", old)


class ComparableTest(unittest.TestCase):
    def test_verify_ignores_every_kind_of_instrumentation(self):
        step = {"action": None, "state": {"hp": 80}}
        bare = {"seed": "S", "steps": [step]}
        full = {"seed": "S", "settle_audit": [0], "idle_probe": {}, "checksum_cost": {"ms": 1}, "timing_ms": {}, "_id": "0",
                "steps": [{**step, "cost": {}, "frame_trace": [], "trace": [], "hidden": {}, "checksums": [], "net": []}]}
        self.assertEqual(serve.comparable(full), serve.comparable(bare))
        self.assertNotEqual(serve.comparable({**bare, "steps": [{**step, "state": {"hp": 79}}]}), serve.comparable(bare))

    def test_a_traced_rerun_still_reproduces_its_batch(self):
        # "trace": true turns on trace mode (Trace.cs) and the checksum tap's full states (ChecksumTap.cs).
        import trace_divergent  # noqa: PLC0415

        batch = [{"action": None, "state": {"enemies": [{"hp": 10}]}}]
        traced = [{"action": None, "state": {"enemies": [{"hp": 10, "move": "BITE", "intents": []}]},
                   "trace": [], "hidden": {}, "checksums": [["x", None, 1]], "net": [{"state": {"cost": 1}}]}]
        self.assertEqual(trace_divergent.strip_new(traced), trace_divergent.strip_new(batch))


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
