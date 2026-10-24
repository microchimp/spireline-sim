"""C2b: Sts2Run_ResetAs (a run as any of the five characters) and the out-of-fight potion actions, through
src/sts2_gym/native.py and referee/lockstep.py. The binding tests need the native library (run API 26) and skip without
it; the lockstep mapping tests run on a scripted emulator."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))

import leakprobes  # noqa: E402
import lockstep  # noqa: E402

try:
    leakprobes.native_env()
    from run_train import engine  # noqa: E402
    from sts2_gym import native  # noqa: E402

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    HAVE_NATIVE = False

# character: (max hp, start relic, start deck)
KITS = {
    "IRONCLAD": (80, "BURNING_BLOOD", ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"]),
    "SILENT": (70, "RING_OF_THE_SNAKE", ["STRIKE_SILENT"] * 5 + ["DEFEND_SILENT"] * 5 + ["NEUTRALIZE", "SURVIVOR"]),
    "DEFECT": (75, "CRACKED_CORE", ["STRIKE_DEFECT"] * 4 + ["DEFEND_DEFECT"] * 4 + ["ZAP", "DUALCAST"]),
    "REGENT": (75, "DIVINE_RIGHT", ["STRIKE_REGENT"] * 4 + ["DEFEND_REGENT"] * 4 + ["FALLING_STAR", "VENERATE"]),
    "NECROBINDER": (66, "BOUND_PHYLACTERY", None),
}


def slugs():
    from run_train import vocab  # noqa: PLC0415

    ids = lockstep.EmuLock._id_map()
    return {k: {v: vocab.game_slug(n) for n, v in m.items()} for k, m in ids.items()}


@unittest.skipUnless(HAVE_NATIVE, "needs the native library")
class ResetAsBindingTests(unittest.TestCase):
    def test_the_binding_starts_a_run_as_the_silent(self):
        s = engine.RunSession("C2BSEED1", 0, engine.CombatBudget(8, 8), character="SILENT")
        try:
            view, slug = s.view(), slugs()
            self.assertEqual(sorted(slug["cards"][c.card_id] for c in view.deck), sorted(KITS["SILENT"][2]))
            self.assertEqual([slug["relics"][r] for r in view.relics], ["RING_OF_THE_SNAKE"])
            self.assertEqual((view.hp, view.max_hp), (70, 70))
        finally:
            s.close()

    def test_each_character_starts_with_its_own_hp_relic_and_deck_size(self):
        slug = slugs()
        for name, (max_hp, relic, deck) in KITS.items():
            with self.subTest(character=name):
                s = engine.RunSession("C2BSEED2", 0, engine.CombatBudget(8, 8), character=name)
                try:
                    view = s.view()
                    self.assertEqual((view.hp, view.max_hp), (max_hp, max_hp))
                    self.assertEqual([slug["relics"][r] for r in view.relics], [relic])
                    names = sorted(slug["cards"][c.card_id] for c in view.deck)
                    self.assertEqual(names, sorted(deck) if deck else names)
                    self.assertEqual(len(names), len(deck) if deck else 10)
                finally:
                    s.close()

    def test_ironclad_through_reset_as_is_the_run_reset_start(self):
        a, b = native.run_create(), native.run_create()
        try:
            obs_a, obs_b = (native.ctypes.c_int * native.RUN_OBS_SIZE)(), (native.ctypes.c_int * native.RUN_OBS_SIZE)()
            self.assertEqual(native.run_reset(a, "C2BSEED3", obs_a, 0), 0)
            self.assertEqual(native.run_reset_character(b, "C2BSEED3", obs_b, 0, "IRONCLAD"), 0)
            self.assertEqual(list(obs_a), list(obs_b))
        finally:
            native.run_destroy(a)
            native.run_destroy(b)

    def test_an_unknown_character_or_ascension_is_refused(self):
        h = native.run_create()
        try:
            obs = (native.ctypes.c_int * native.RUN_OBS_SIZE)()
            with self.assertRaises(ValueError):
                native.run_reset_character(h, "C2BSEED4", obs, 0, "WATCHER")
            self.assertEqual(native.run_reset_character(h, "C2BSEED4", obs, 99, "SILENT"), -2)
            self.assertEqual(native.run_reset_character(255, "C2BSEED4", obs, 0, "SILENT"), -1)
            self.assertEqual(native.run_reset_character(h, "C2BSEED4", obs, 0, "character.defect"), 0)
        finally:
            native.run_destroy(h)

    def test_the_reason_binding_names_an_empty_slot_and_the_mask_lists_no_potion_actions_on_an_empty_belt(self):
        s = engine.RunSession("C2BSEED5", 0, engine.CombatBudget(8, 8))
        try:
            self.assertEqual(native.run_potion_action_reason(s.handle, native.RUN_USE_POTION_ACTION), "the belt slot is empty")
            self.assertEqual(native.run_potion_action_reason(s.handle, native.RUN_DISCARD_POTION_ACTION + 2), "the belt slot is empty")
            with self.assertRaises(ValueError):
                native.run_potion_action_reason(s.handle, 3)
            mask = native.run_action_mask(s.handle, native.RUN_MAX_ACTIONS)
            self.assertFalse(any(mask[native.RUN_USE_POTION_ACTION:native.RUN_DISCARD_POTION_ACTION + native.RUN_POTION_SLOTS]))
        finally:
            s.close()

    def test_the_potion_action_constants_agree_across_the_layers(self):
        self.assertEqual(native.RUN_USE_POTION_ACTION, engine.USE_POTION_ACTION)
        self.assertEqual(native.RUN_DISCARD_POTION_ACTION, engine.DISCARD_POTION_ACTION)
        self.assertEqual((lockstep.USE_POTION_ACTION, lockstep.DISCARD_POTION_ACTION, lockstep.POTION_SLOTS),
                         (native.RUN_USE_POTION_ACTION, native.RUN_DISCARD_POTION_ACTION, native.RUN_POTION_SLOTS))

    def test_a_screens_legal_actions_hide_the_use_and_discard_actions(self):
        real = engine.native.run_action_mask
        mask = [0] * native.RUN_MAX_ACTIONS
        for i in (3, 242, 243, 248, 250, 253):
            mask[i] = 1
        engine.native.run_action_mask = lambda handle, n: mask
        try:
            self.assertEqual(engine._legal(0), [3])
        finally:
            engine.native.run_action_mask = real


class CharacterSupportTests(unittest.TestCase):
    def test_ironclad_needs_nothing_and_the_other_four_need_the_binding(self):
        stub = SimpleNamespace()
        self.assertIsNone(lockstep.character_supported("IRONCLAD", stub))
        self.assertIn("run_reset_character", lockstep.character_supported("SILENT", stub))
        with self.assertRaises(SystemExit):
            lockstep.require_character("SILENT", stub)

    def test_the_five_characters_the_game_offers_are_supported_and_others_are_not(self):
        stub = SimpleNamespace(run_reset_character=lambda *a: 0,
                               RUN_CHARACTERS=("IRONCLAD", "SILENT", "REGENT", "NECROBINDER", "DEFECT"))
        for name in stub.RUN_CHARACTERS:
            self.assertIsNone(lockstep.character_supported(name, stub), name)
        self.assertIsNone(lockstep.character_supported("CHARACTER.SILENT", stub))
        self.assertIn("WATCHER", lockstep.character_supported("WATCHER", stub))
        lockstep.require_character("DEFECT", stub)
        with self.assertRaises(SystemExit):
            lockstep.require_character("WATCHER", stub)

    @unittest.skipUnless(HAVE_NATIVE, "needs the native library")
    def test_the_real_binding_supports_all_five(self):
        for name in native.RUN_CHARACTERS:
            self.assertIsNone(lockstep.character_supported(name), name)


def potion_emu(belt, legal, reasons=None):
    """An EmuLock without the native library over a scripted belt (emulator potion ids -> slugs below) and mask."""
    e = lockstep.EmuLock.__new__(lockstep.EmuLock)
    taken = []
    e.s = SimpleNamespace(handle=7, _step=taken.append)
    e.slug = {"cards": {}, "relics": {}, "potions": {11: "FRUIT_JUICE", 12: "BLOCK_POTION", 13: "BLOOD_POTION"}}
    e.native = SimpleNamespace(
        run_state_list=lambda h, k, n: (belt + [0] * n)[:n],
        run_action_mask=lambda h, n: [1 if i in legal else 0 for i in range(n)],
        run_potion_action_reason=lambda h, a: (reasons or {}).get(a, ""),
    )
    return e, taken


class PotionLabelMappingTests(unittest.TestCase):
    def test_a_use_label_becomes_the_use_action_of_the_slot_holding_that_potion(self):
        e, taken = potion_emu([11, 12], legal={242, 248, 249})
        labels = [["use_potion", 0, "FRUIT_JUICE"], ["map", 1, 0]]
        self.assertIsNone(e.apply("map", labels, 0))
        self.assertEqual(taken, [242])

    def test_a_discard_label_becomes_the_discard_action_and_the_slot_is_found_by_potion_when_the_sides_differ(self):
        e, taken = potion_emu([11, 12], legal={242, 248, 249})
        self.assertIsNone(e.apply("shop", [["discard_potion", 1, "BLOCK_POTION"]], 0))
        self.assertIsNone(e.apply("shop", [["discard_potion", 0, "BLOCK_POTION"]], 0))  # the referee's slot differs
        self.assertEqual(taken, [249, 249])

    def test_a_refused_use_carries_the_emulators_reason_and_steps_nothing(self):
        why = "the potion's Usage is not AnyTime: the game offers no use button for it outside a fight"
        e, taken = potion_emu([11, 12], legal={242, 248, 249}, reasons={243: why})
        out = e.apply("rest", [["use_potion", 1, "BLOCK_POTION"]], 0)
        self.assertTrue(out.startswith("no_emulator_option:rest:"), out)  # the screen kind, as for any gap
        self.assertIn('"use_potion"', out)
        self.assertTrue(out.endswith(why), out)
        self.assertEqual(taken, [])

    def test_a_potion_the_emulators_belt_lacks_is_a_named_gap(self):
        e, taken = potion_emu([11], legal={242, 248})
        out = e.apply("event", [["discard_potion", 0, "BLOOD_POTION"]], 0)
        self.assertEqual(out, 'no_emulator_option:event:["discard_potion", 0, "BLOOD_POTION"]:'
                              "the emulator's belt holds no BLOOD_POTION")
        self.assertEqual(taken, [])

    def test_an_engine_refusal_is_a_gap_not_a_crash(self):
        e, taken = potion_emu([11], legal={242})

        def refuse(action):
            raise RuntimeError(f"the engine refused action {action} in phase 2")

        e.s._step = refuse
        out = e.apply("map", [["use_potion", 0, "FRUIT_JUICE"]], 0)
        self.assertTrue(out.startswith("no_emulator_option:map:"), out)
        self.assertIn("refused action 242", out)

    def test_card_alternatives_stay_without_an_emulator_option(self):
        e, taken = potion_emu([11], legal={242})
        out = e.apply("card_reward", ["alt:SACRIFICE"], 0)
        self.assertEqual(out, f"no_emulator_option:card_reward:{json.dumps('alt:SACRIFICE')}")
        self.assertEqual(taken, [])

    def test_a_potion_label_is_never_paired_as_a_screen_option(self):
        e, _ = potion_emu([11], legal={242})
        e.dec = SimpleNamespace(options=(), view=None)
        self.assertEqual(e._pair("map", [["use_potion", 0, "FRUIT_JUICE"]]), [None])


if __name__ == "__main__":
    unittest.main()
