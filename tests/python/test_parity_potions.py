"""Block P4 of the information-parity side blocks: what each potion does (`rl/parity_potions.py`, D59 lane D).

Pinned: every potion id has a row; the well-known potions carry the expected tags; the immediate energy, block, Strength, Dexterity, HP and max HP of a drink are the table's, measured by drinking
each potion in a fight on the engine; the encoding's layout; held and offered potions are tag sums; an option's afterstate (a potion bought or taken is held and is the option's potion).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "rl", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import parity_potions as pp  # noqa: E402
from sts2_gym import names  # noqa: E402


def pid(name):
    return {v: k for k, v in names.potions().items()}[name]


class TableTests(unittest.TestCase):
    def test_every_potion_has_a_row(self):
        ids = set(names.potions())
        have = set(pp.potion_rows())
        self.assertEqual(len(ids), 63)
        self.assertEqual(sorted(ids - have), [])
        self.assertEqual(sorted(have - ids), [])

    def test_names_are_the_declared_ones(self):
        for p, (effects, flags) in pp.potion_rows().items():
            self.assertTrue(set(effects) <= set(pp.EFFECTS), p)
            self.assertTrue(set(flags) <= set(pp.FLAGS), (p, set(flags) - set(pp.FLAGS)))

    def test_well_known_potions(self):
        r = pp.potion_rows()
        self.assertEqual(r[pid("BlockPotion")][0], {"block": 12})
        self.assertEqual(r[pid("EnergyPotion")][0], {"energy": 2})
        self.assertEqual(r[pid("StrengthPotion")][0], {"strength": 2})
        self.assertEqual(r[pid("DexterityPotion")][0], {"dexterity": 2})
        self.assertEqual(r[pid("FirePotion")][0], {"damage": 20})
        self.assertIn("targets_enemy", r[pid("FirePotion")][1])
        self.assertEqual(r[pid("FruitJuice")][0], {"max_hp": 5})
        self.assertIn("any_time", r[pid("FruitJuice")][1])
        self.assertIn("automatic", r[pid("FairyInABottle")][1])
        self.assertEqual(r[pid("SwiftPotion")][0], {"draw": 3})

    def test_rarities_come_from_the_engine_table(self):
        r = pp.potion_rarities()
        self.assertEqual(len(r), 63)
        self.assertEqual(r[pid("BlockPotion")], "Common")
        self.assertEqual(r[pid("FairyInABottle")], "Rare")


class EncodingTests(unittest.TestCase):
    def test_layout(self):
        self.assertEqual(pp.POTION_TAG_WIDTH, len(pp.EFFECTS) + len(pp.FLAGS) + len(pp.RARITIES))
        v = pp.potion_tags(pid("BlockPotion"))
        self.assertEqual(v.shape, (pp.POTION_TAG_WIDTH,))
        self.assertEqual(v.dtype, np.float32)
        self.assertAlmostEqual(float(v[pp.EFFECTS.index("block")]), 12 / pp.SCALES["block"])
        self.assertEqual(float(v[len(pp.EFFECTS) + len(pp.FLAGS) + pp.RARITIES.index("Common")]), 1.0)

    def test_matrix_range_and_sum(self):
        m = pp.potion_tag_matrix()
        self.assertEqual(m.shape, (64, pp.POTION_TAG_WIDTH))
        self.assertTrue(np.all(np.abs(m) <= 1.0))
        np.testing.assert_allclose(pp.held_tags((pid("BlockPotion"), pid("EnergyPotion"))), pp.potion_tags(pid("BlockPotion")) + pp.potion_tags(pid("EnergyPotion")), atol=1e-7)
        np.testing.assert_array_equal(pp.potion_tags(0), np.zeros(pp.POTION_TAG_WIDTH, dtype=np.float32))
        np.testing.assert_array_equal(pp.potion_tags(999), np.zeros(pp.POTION_TAG_WIDTH, dtype=np.float32))


class MeasuredTests(unittest.TestCase):
    def test_a_drink_shows_what_the_table_says(self):
        bad = []
        for p, (effects, flags) in pp.potion_rows().items():
            name = names.potions()[p]
            if {"chooses", "targets_enemy", "automatic", "random"} & set(flags) or name in pp.DRINK_CHECK_EXEMPT:
                continue
            got = pp.measure_drink(p)
            if got is None:
                continue
            for key, field in (("energy", "energy"), ("block", "block"), ("strength", "strength"), ("dexterity", "dexterity"), ("max_hp", "max_hp"), ("heal", "hp")):
                want, have = effects.get(key, 0), got[field]
                if want and have != want:
                    bad.append((name, key, want, have))
                elif not want and have > 0 and key != "heal":
                    bad.append((name, key, 0, have))
        self.assertEqual(bad, [])

    def test_the_drinkable_potions_are_mostly_measured(self):
        measured = sum(1 for p in pp.potion_rows() if pp.measure_drink(p) is not None)
        self.assertGreaterEqual(measured, 50)


if __name__ == "__main__":
    unittest.main()
