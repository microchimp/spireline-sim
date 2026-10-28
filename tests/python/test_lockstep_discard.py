"""D32/D34: the lockstep's emulator side runs the run mask at version 3, so the referee's discard_potion and use_potion on a
card-reward screen with a full belt are stepped, not named gaps. A refusal (a potion with no use outside a fight, a use
under mask version 2, or a version-1 run) stays a named gap."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import leakprobes  # noqa: E402
import lockstep  # noqa: E402

try:
    leakprobes.native_env()
    import mask_screens  # noqa: E402
    from run_train import engine  # noqa: E402
    from sts2_gym import native  # noqa: E402

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    HAVE_NATIVE = False

BELT = ("BLOCK_POTION", "BLOOD_POTION", "FRUIT_JUICE")


def belt_of(e):
    return [e.slug["potions"].get(p, f"?{p}") if p else None for p in native.run_state_list(e.s.handle, 2, native.RUN_POTION_SLOTS)][:3]


@unittest.skipUnless(HAVE_NATIVE, "needs the native library (run API 30)")
class DiscardOnTheCardRewardTests(unittest.TestCase):
    def bare(self, seed, mask_version):
        """An EmuLock-shaped object over a run built at this mask version, on a card reward with the full belt."""
        s = engine.RunSession(seed, 0, engine.CombatBudget(4, 4), mask_version=mask_version)
        self.addCleanup(s.close)
        e = lockstep.EmuLock.__new__(lockstep.EmuLock)
        e.s, e.native = s, native
        e.slug = {k: {v: __import__("run_train.vocab", fromlist=["x"]).game_slug(n) for n, v in m.items()}
                  for k, m in lockstep.EmuLock._id_map().items()}
        mask_screens.to_card_reward(s)
        mask_screens.hold_belt(s, *BELT)
        return e

    def emu(self, seed):
        e = lockstep.EmuLock(seed, 0, sims=4)
        self.addCleanup(e.s.close)
        mask_screens.to_card_reward(e.s)
        mask_screens.hold_belt(e.s, *BELT)
        return e

    def test_the_emulator_side_runs_at_mask_version_four(self):
        self.assertEqual(lockstep.MASK_VERSION, 4)
        e = lockstep.EmuLock("D32LOCK0", 0, sims=4)
        try:
            self.assertEqual(e.s.mask_version, 4)
        finally:
            e.s.close()

    def test_a_discard_label_on_a_card_reward_with_a_full_belt_is_stepped_not_a_gap(self):
        e = self.emu("D32LOCK1")
        out = e.apply("card_reward", [["discard_potion", 1, "BLOOD_POTION"], "skip"], 0)
        self.assertIsNone(out)
        self.assertEqual(belt_of(e), ["BLOCK_POTION", None, "FRUIT_JUICE"])
        # the referee's slot may differ from the emulator's: the slot is found by potion
        self.assertIsNone(e.apply("card_reward", [["discard_potion", 0, "FRUIT_JUICE"]], 0))
        self.assertEqual(belt_of(e), ["BLOCK_POTION", None, None])

    def test_a_use_label_on_a_card_reward_with_a_blood_potion_is_stepped_not_a_gap(self):
        e = self.emu("D34LOCK2")
        e.s.env.debug_set_player(hp=30, max_hp=80)
        out = e.apply("card_reward", [["use_potion", 1, "BLOOD_POTION"], "skip"], 0)
        self.assertIsNone(out)
        self.assertEqual(belt_of(e), ["BLOCK_POTION", None, "FRUIT_JUICE"])
        self.assertEqual(e.s.info["hp"], 46)  # the potion healed 20% of the max hp (16)
        # the referee's slot may differ from the emulator's: the slot is found by potion
        self.assertIsNone(e.apply("card_reward", [["use_potion", 0, "FRUIT_JUICE"]], 0))
        self.assertEqual(belt_of(e), ["BLOCK_POTION", None, None])
        self.assertEqual(e.s.info["max_hp"], 85)  # +5 max hp

    def test_a_use_label_for_a_combat_only_potion_stays_a_named_gap_with_the_emulators_reason(self):
        e = self.emu("D34LOCK3")
        out = e.apply("card_reward", [["use_potion", 0, "BLOCK_POTION"]], 0)
        self.assertTrue(out.startswith('no_emulator_option:card_reward:["use_potion", 0, "BLOCK_POTION"]:'), out)
        self.assertIn("Usage", out)
        self.assertEqual(belt_of(e), list(BELT))

    def test_a_version_two_run_still_names_a_use_as_a_gap_with_the_thrown_away_reason(self):
        e = self.bare("D34LOCK4", 2)
        out = e.apply("card_reward", [["use_potion", 1, "BLOOD_POTION"]], 0)
        self.assertTrue(out.startswith('no_emulator_option:card_reward:["use_potion", 1, "BLOOD_POTION"]:'), out)
        self.assertIn("thrown away", out)
        self.assertEqual(belt_of(e), list(BELT))

    def test_a_version_one_run_still_names_the_closed_belt(self):
        e = self.bare("D32LOCK3", 1)
        out = e.apply("card_reward", [["discard_potion", 0, "BLOCK_POTION"]], 0)
        self.assertTrue(out.startswith("no_emulator_option:card_reward:"), out)
        self.assertIn("belt is closed", out)
        self.assertEqual(belt_of(e), list(BELT))


if __name__ == "__main__":
    unittest.main()
