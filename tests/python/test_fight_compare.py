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

from tools.fidelity.fight_compare import diff, game_powers, power, EMU_RENAME, NOT_POWERS  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
