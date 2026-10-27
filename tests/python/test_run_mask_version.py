"""D32/D34: the run action mask has versions (Sts2Run_SetMaskVersion, run API 30). Version 1, the default, is today's mask;
version 2 adds the potion discard on the card reward, deck-select, bundle-select and Crystal Sphere screens; version 3 adds the
use of the AnyTime potions there. The C# tests
(RunMaskVersionTests, RunMaskGoldenTests) cover every screen; these drive a real run to a card reward through the binding."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import leakprobes  # noqa: E402

try:
    leakprobes.native_env()
    import mask_screens  # noqa: E402
    from run_train import engine  # noqa: E402
    from sts2_gym import native, run_env  # noqa: E402

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    HAVE_NATIVE = False

BELT = ("BLOCK_POTION", "BLOOD_POTION", "FRUIT_JUICE")


@unittest.skipUnless(HAVE_NATIVE, "needs the native library (run API 31)")
class MaskVersionBindingTests(unittest.TestCase):
    def test_the_binding_requires_api_31_and_names_the_versions(self):
        self.assertEqual(native._REQUIRED_RUN_NATIVE_API_VERSION, 31)
        self.assertEqual((native.RUN_MASK_VERSION_DEFAULT, native.RUN_MASK_VERSIONS), (1, (1, 2, 3)))

    def test_an_unknown_handle_or_version_is_refused(self):
        h = native.run_create()
        try:
            native.run_set_mask_version(h, 3)
            native.run_set_mask_version(h, 2)
            native.run_set_mask_version(h, 1)
            for bad in (0, 4, -1):
                with self.assertRaises(ValueError):
                    native.run_set_mask_version(h, bad)
        finally:
            native.run_destroy(h)
        with self.assertRaises(ValueError):
            native.run_set_mask_version(255, 2)

    def test_version_two_adds_exactly_the_discard_slots_on_a_card_reward_and_version_one_is_unchanged(self):
        s = engine.RunSession("D32MASK1", 0, engine.CombatBudget(4, 4))
        try:
            mask_screens.to_card_reward(s)
            mask_screens.hold_belt(s, *BELT)
            one = mask_screens.run_mask(s.handle)
            self.assertFalse(any(one[native.RUN_USE_POTION_ACTION:]), "version 1 lists no potion action on a card reward")
            self.assertTrue(any(one[:4]), "the card reward's own options are on the mask")
            native.run_set_mask_version(s.handle, 2)
            two = mask_screens.run_mask(s.handle)
            expected = list(one)
            for slot in range(len(BELT)):
                expected[native.RUN_DISCARD_POTION_ACTION + slot] = 1
            self.assertEqual(two, expected)
            native.run_set_mask_version(s.handle, 1)
            self.assertEqual(mask_screens.run_mask(s.handle), one)
        finally:
            s.close()

    def test_version_three_adds_exactly_the_use_slots_of_the_usable_potions_to_version_twos_mask(self):
        s = engine.RunSession("D34MASK1", 0, engine.CombatBudget(4, 4))
        try:
            mask_screens.to_card_reward(s)
            mask_screens.hold_belt(s, *BELT)
            native.run_set_mask_version(s.handle, 2)
            two = mask_screens.run_mask(s.handle)
            native.run_set_mask_version(s.handle, 3)
            three = mask_screens.run_mask(s.handle)
            expected = list(two)
            expected[native.RUN_USE_POTION_ACTION + 1] = 1  # Blood Potion
            expected[native.RUN_USE_POTION_ACTION + 2] = 1  # Fruit Juice (the Block Potion is combat-only)
            self.assertEqual(three, expected)
            # the gym's mask still hides every potion action from a policy
            self.assertFalse(s.env.action_masks()[native.RUN_USE_POTION_ACTION:].any())
        finally:
            s.close()

    def test_a_session_built_with_version_two_and_its_clone_carry_it_and_the_gym_mask_still_hides_it(self):
        s = engine.RunSession("D32MASK2", 0, engine.CombatBudget(4, 4), mask_version=2)
        try:
            mask_screens.to_card_reward(s)
            mask_screens.hold_belt(s, *BELT)
            self.assertTrue(all(mask_screens.run_mask(s.handle)[native.RUN_DISCARD_POTION_ACTION + k] for k in range(3)))
            self.assertEqual(engine._legal(s.handle), [i for i in engine._legal(s.handle) if i < native.RUN_USE_POTION_ACTION])
            self.assertFalse(s.env.action_masks()[native.RUN_USE_POTION_ACTION:].any())
            copy = run_env.Sts2RunEnv.clone(s.env, resample_hidden=True, seed=5)
            try:
                self.assertTrue(mask_screens.run_mask(copy._run_handle)[native.RUN_DISCARD_POTION_ACTION])
                self.assertEqual(copy._mask_version, 2)
            finally:
                copy.close()
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
