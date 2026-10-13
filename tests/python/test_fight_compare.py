"""tools.fidelity.fight_compare: power lists that mean the same thing on both sides compare equal.

The cases are the representation differences found by the night03 triage, each checked against
the game's classes: emulator-only counters are hidden, renamed buffs map to the game power, and
every TemporaryStrengthPower subclass folds into one TEMPORARY_STRENGTH holding the summed amount.
A real difference in an amount must still show up.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity.fight_compare import (  # noqa: E402
    EMU_RENAME, NOT_POWERS, diff, emu_intent, game_intent, game_powers, power)


def emu_powers(bs):
    return sorted([EMU_RENAME.get(power(n), power(n)), a] for n, a in bs if power(n) not in NOT_POWERS)


class PowerListTest(unittest.TestCase):
    def test_temporary_strength_subclasses_sum(self):
        game = game_powers([["STRENGTH_POWER", -9], ["DARK_SHACKLES_POWER", 9], ["MANGLE_POWER", 10]])
        self.assertEqual(game, [["STRENGTH", -9], ["TEMPORARY_STRENGTH", 19]])
        self.assertEqual(game, emu_powers([["Strength", -9], ["TemporaryStrength", 19]]))

    def test_setup_strike_is_temporary_strength(self):
        self.assertEqual(game_powers([["SETUP_STRIKE_POWER", 7]]), emu_powers([["TemporaryStrength", 7]]))

    def test_emulator_counters_hidden(self):
        emu = emu_powers([["Inferno", 2], ["InfernoSelfDamage", 3], ["PanacheArmed", 1], ["ClawDamage", 4],
                          ["TheBombDamage", 40], ["Stunned", 1]])
        self.assertEqual(emu, [["INFERNO", 2]])

    def test_renames(self):
        emu = emu_powers([["CrimsonMantleBlock", 4], ["NextTurnEnergy", 2], ["NextTurnDraw", 1]])
        game = game_powers([["CRIMSON_MANTLE_POWER", 4], ["ENERGY_NEXT_TURN_POWER", 2], ["DRAW_CARDS_NEXT_TURN_POWER", 1]])
        self.assertEqual(emu, game)

    def test_real_amount_difference_still_shows(self):
        self.assertTrue(diff(game_powers([["DARK_SHACKLES_POWER", 9]]), emu_powers([["TemporaryStrength", 8]])))


def g_enemy(intents, name="NIBBIT"):
    e = {"name": name, "hp": 10, "block": 0, "powers": []}
    if intents is not None:
        e["intents"] = intents
    return {**{k: v for k, v in e.items() if k != "intents"}, "intent": game_intent(e)}


def e_enemy(type_, mag=0, hits=1, carries=False, secondary=None, buffs=(), name="NIBBIT"):
    raw = {"intent": {"type": type_, "magnitude": mag, "hits": hits, "carries_damage": carries, "secondary": secondary},
           "buffs": [{"name": b} for b in buffs]}
    return {"name": name, "hp": 10, "block": 0, "powers": [], "intent": emu_intent(raw)}


class IntentTest(unittest.TestCase):
    """The game's MoveState intents and the emulator's Intent, mapped to one vocabulary (night04 moves)."""

    def test_same_moves_compare_equal(self):
        pairs = [
            ([["Attack", 7, 1], ["Defend"]], e_enemy(0, 7)),                 # Nibbit SLICE_MOVE
            ([["Attack", 2, 4]], e_enemy(0, 2, 4)),                          # Seapunk SPINNING_KICK_MOVE
            ([["Buff"], ["Defend"]], e_enemy(2)),                            # Seapunk BUBBLE_BURP_MOVE
            ([["StatusCard", 2]], e_enemy(3, 2)),                            # Leaf Slime STICKY_SHOT
            ([["DebuffStrong"]], e_enemy(3, 1)),                             # Shrinker Beetle SHRINKER_MOVE
            ([["Stun"]], e_enemy(4, buffs=["Stunned"])),                     # a stunned Corpse Slug
            ([["Attack", 9, 1], ["Debuff"]], e_enemy(3, 9, carries=True)),  # damage carried by a debuff
        ]
        for game, emu in pairs:
            self.assertEqual(diff([g_enemy(game)], [emu]), [], game)

    def test_wrong_move_shows(self):
        self.assertTrue(diff([g_enemy([["Attack", 13, 1]])], [e_enemy(2)]))          # Butt vs Hiss
        self.assertTrue(diff([g_enemy([["Attack", 8, 1]])], [e_enemy(0, 14)]))       # Chomp vs Stomp
        self.assertTrue(diff([g_enemy([["Attack", 3, 2]])], [e_enemy(0, 3, 1)]))     # hits differ

    def test_record_without_intents_is_not_compared(self):
        self.assertEqual(diff([g_enemy(None)], [e_enemy(0, 99)]), [])
        self.assertTrue(diff([dict(g_enemy(None), hp=9)], [e_enemy(0, 99)]))


if __name__ == "__main__":
    unittest.main()
