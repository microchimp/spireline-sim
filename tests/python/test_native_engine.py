"""The emulator engine option (docs/rebuild/NATIVE_KERNEL.md piece 1, D31): old by default, core2 only where asked.

The library side runs in a fresh interpreter per case, because sts2_gym.native reads STS2_ENGINE once, when it loads.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT))
import envproto  # noqa: E402
import gen0  # noqa: E402

LIB = ROOT / "out" / "Sts2Emulator.dll"

PROBE = r"""
import ctypes, json, sys
sys.path.insert(0, sys.argv[1])
from sts2_gym import native
out = {"engine": native.engine(), "create": native.create(7)}
h = native.run_create()
obs = (ctypes.c_int * native.RUN_OBS_SIZE)()
out["run"] = h
out["reset"] = native.run_reset(h, "QS2GYXRKWN", obs, 10)
# A walk of first legal actions into and through the first fights: under core2 every combat step goes through the
# kernel, whose ids and handler table come from the files embedded in the library.
rew, term, trunc = (ctypes.c_float * 1)(), (ctypes.c_int * 1)(), (ctypes.c_int * 1)()
# At every combat step the run is also cloned (resampled) and its fight forked (resampled) and both copies take the
# same step: the search's paths, which carry the kernel (piece 2) and refuse nothing.
combat_steps, refused, forks = 0, 0, 0
cobs = (ctypes.c_int * native.OBS_SIZE)()
MAXA = 32
crew = (ctypes.c_float * 1)()
for i in range(300):
    mask = native.run_action_mask(h, native.RUN_MAX_ACTIONS)
    legal = [a for a in range(native.RUN_MAX_ACTIONS) if mask[a]]
    if not legal or term[0]:
        break
    in_combat = native.run_phase(h) == 0
    combat_steps += in_combat
    if in_combat and i % 3 == 0:
        c = native.run_clone(h, True, i, (ctypes.c_int * native.RUN_OBS_SIZE)())
        refused += native.run_step(c, legal[0], -1, (ctypes.c_int * native.RUN_OBS_SIZE)(), rew, term, trunc) < 0
        term[0] = 0
        native.run_destroy(c)
        f = native.run_fork_combat(h, True, i, cobs)
        fmask = native.valid_actions(f, MAXA)
        native.step(f, [a for a in range(MAXA) if fmask[a]][0], cobs, crew)
        native.destroy(f)
        forks += 1
    refused += native.run_step(h, legal[0], -1, obs, rew, term, trunc) < 0
out["combat_steps"], out["refused"], out["forks"] = combat_steps, refused, forks
# A combat handle of its own, through its fight's end or 200 steps.
ch, handle_steps = out["create"], 0
if ch >= 0:
    native.reset(ch, cobs)
    for _ in range(200):
        cmask = native.valid_actions(ch, MAXA)
        legal = [a for a in range(MAXA) if cmask[a]]
        handle_steps += 1
        if not legal or native.step(ch, legal[0], cobs, crew):
            break
out["handle_steps"] = handle_steps
print(json.dumps(out))
"""


def _probe(engine: str | None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("STS2_ENGINE", "STS2_LIB_PATH")}
    env["STS2_ALLOW_STALE_NATIVE"] = "1"
    if engine is not None:
        env["STS2_ENGINE"] = engine
    with tempfile.TemporaryDirectory() as d:
        script = Path(d) / "probe.py"
        script.write_text(PROBE, encoding="utf-8")
        return subprocess.run([sys.executable, "-I", str(script), str(ROOT / "src")], env=env, capture_output=True,
                              text=True, timeout=120, check=False)


@unittest.skipUnless(LIB.exists(), "no native build in out/")
class LibraryEngineTests(unittest.TestCase):
    def test_unset_is_old_and_combat_handles_work(self):
        r = _probe(None)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        doc = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(doc["engine"], "old")
        self.assertGreaterEqual(doc["create"], 0)
        self.assertGreaterEqual(doc["run"], 0)
        self.assertGreater(doc["combat_steps"], 0)
        self.assertGreater(doc["forks"], 0)
        self.assertEqual(doc["refused"], 0)

    def test_core2_builds_the_shipped_kernel_and_runs_handles_clones_and_forks(self):
        r = _probe("core2")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        doc = json.loads(r.stdout.strip().splitlines()[-1])
        shipped = json.loads((ROOT / "tools" / "fidelity" / "migrated_families.json").read_text(encoding="utf-8"))["migrated"]
        self.assertEqual(doc["engine"], "core2:" + ",".join(shipped))
        self.assertGreaterEqual(doc["create"], 0)  # a combat handle steps through the kernel's gate (piece 2)
        self.assertGreater(doc["handle_steps"], 0)
        self.assertGreaterEqual(doc["run"], 0)
        self.assertGreater(doc["combat_steps"], 0)
        self.assertGreater(doc["forks"], 0)
        self.assertEqual(doc["refused"], 0)

    def test_a_refused_spec_fails_the_import(self):
        r = _probe("core3")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("unknown engine 'core3'", r.stderr)


class ScoreEngineTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get("STS2_ENGINE")

    def tearDown(self):
        if self._env is None:
            os.environ.pop("STS2_ENGINE", None)
        else:
            os.environ["STS2_ENGINE"] = self._env

    @staticmethod
    def _score(**kw: str) -> list:
        out = Path(tempfile.mkdtemp()) / "probe"
        a = SimpleNamespace(out=str(out), game="G", seeds_file="F.json", runs=500, procs=4, sims=60, config="a3-v1", ascension=10,
                            character="IRONCLAD", stop_after_h=1.0, lib=None, emu_dir=None, gen_dir=None, only="S3", **kw)
        cmds = []
        with mock.patch("subprocess.run", lambda cmd, **_: cmds.append(cmd) or SimpleNamespace(returncode=0)):
            gen0.score_real(a)
        return cmds[0]

    def test_score_passes_the_engine_to_the_lockstep(self):
        cmd = self._score(engine="core2")
        self.assertEqual(cmd[cmd.index("--engine") + 1], "core2")

    def test_score_defaults_to_old(self):
        cmd = self._score()
        self.assertEqual(cmd[cmd.index("--engine") + 1], "old")
        a = gen0.build_parser().parse_args(["score", "--game", "G", "--seeds-file", "F", "--out", "O"])
        self.assertEqual(a.engine, "old")

    def test_only_score_selects_core2_and_every_other_command_clears_it(self):
        os.environ["STS2_ENGINE"] = "core2"
        with mock.patch.object(gen0, "bench_cmd", return_value=0):
            gen0.main(["bench-emu", "--out", "O"])
        self.assertNotIn("STS2_ENGINE", os.environ)  # the lab's paths run old whatever the environment said
        with mock.patch.object(gen0, "score_real", return_value=0):
            gen0.main(["score", "--game", "G", "--seeds-file", "F", "--out", "O", "--engine", "core2"])
        self.assertEqual(os.environ.get("STS2_ENGINE"), "core2")

    def test_an_emulator_arm_on_another_engine_does_not_pair(self):
        d = Path(tempfile.mkdtemp())
        (d / "runs.jsonl").write_text("", encoding="utf-8")
        agent = {"generation": 0, "prior_sha256": None, "macro": "rules"}
        (d / "summary.json").write_text(json.dumps({"agent": agent}), encoding="utf-8")
        self.assertTrue(gen0.emu_arm_match(d, agent)[0])
        ok, why = gen0.emu_arm_match(d, agent, "core2")
        self.assertFalse(ok)
        self.assertIn("another engine", why)
        (d / "summary.json").write_text(json.dumps({"agent": agent, "engine": "core2"}), encoding="utf-8")
        self.assertTrue(gen0.emu_arm_match(d, agent, "core2")[0])

    def test_the_lockstep_takes_the_engine_for_play_only(self):
        a = envproto.parser().parse_args(["play", "--game", "G", "--out", "O", "--engine", "core2"])
        self.assertEqual(a.engine, "core2")
        self.assertEqual(envproto.parser().parse_args(["play", "--game", "G", "--out", "O"]).engine, "old")
        envproto.select_engine("core2")
        self.assertEqual(os.environ.get("STS2_ENGINE"), "core2")
        envproto.select_engine("old")
        self.assertNotIn("STS2_ENGINE", os.environ)


if __name__ == "__main__":
    unittest.main()
