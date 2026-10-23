"""Leak test L4 on the run readouts (Spireline DESIGN.md 7.6): what is behind a map node does not
reach any agent-visible readout. The engine keeps it (it resolves a node from it); the binding
refuses to hand it out unless a generation test asks. Probes in referee/leakprobes.py."""

import ctypes
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))

import leakprobes  # noqa: E402

try:
    leakprobes.native_env()
    from sts2_gym import native  # noqa: E402

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    HAVE_NATIVE = False


@unittest.skipUnless(HAVE_NATIVE, "needs the native emulator library")
class RunReadoutsHideEncountersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = leakprobes.play_to("L2A", lambda s, d: d is not None and d.type == "map")

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def test_the_hidden_generation_lists_are_refused_unless_a_generation_test_asks(self):
        for list_id in (11, 12, 13):
            with self.assertRaises(PermissionError, msg=f"list {list_id} was handed out"):
                native.run_state_list(self.s.handle, list_id, 256)
        self.assertTrue(native.run_state_list(self.s.handle, 11, 256, hidden=True))

    def test_passes_before_no_leak_every_other_readout_is_the_same_in_all_resampled_clones(self):
        """At a map decision the encounters behind the nodes differ from clone to clone (L2). Any
        readout that carried them would differ too: the observation, the info block and every
        state list but the hidden three must not."""
        seen: dict[str, set] = {}
        for k in range(120):
            obs = (ctypes.c_int * native.RUN_OBS_SIZE)()
            h = native.run_clone(self.s.handle, True, 1000 + k, obs)
            try:
                out = {"obs": tuple(obs), "info": tuple(native.run_info(h))}
                for list_id in range(0, 26):
                    if list_id in (11, 12, 13):
                        continue
                    try:
                        out[f"list{list_id}"] = native.run_state_list(h, list_id, 4096)
                    except RuntimeError:  # an id the engine does not export
                        continue
                for key, value in out.items():
                    seen.setdefault(key, set()).add(value)
            finally:
                native.run_destroy(h)
        varying = sorted(k for k, v in seen.items() if len(v) > 1)
        self.assertEqual(varying, [], f"readouts that move with the hidden encounters: {varying}")
        control = set()
        for k in range(20):
            h = leakprobes.resampled_clone(self.s, 7000 + k)
            try:
                control.add(native.run_state_list(h, 11, 256, hidden=True))
            finally:
                native.run_destroy(h)
        self.assertGreater(len(control), 1, "the control: the hidden list itself must vary across clones")


if __name__ == "__main__":
    unittest.main()
