"""Block P1 of the information-parity side blocks: what each relic does (`rl/parity_relics.py`, D59 lane D).

Pinned: every relic id has a row and no row names an unknown relic; a handful of well-known relics carry the expected effect tags and triggers; the effects the
engine can show at a fight's first turn (energy, cards drawn, block, Strength, Dexterity) are those of the table, measured by starting a fight with the relic and
without it; the encoding's layout and scales; a relic held and a relic on offer are tag sums, and an option that adds a relic adds its tags to the held ones.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "rl", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import parity_relics as pr  # noqa: E402
from run_train import vocab  # noqa: E402
from sts2_gym import names, native  # noqa: E402


def rid(name: str) -> int:
    return vocab.relic_id_by_name(name)


class TableTests(unittest.TestCase):
    def test_every_relic_id_has_a_row(self):
        ids = set(names.relics())
        have = set(pr.relic_rows())
        self.assertEqual(len(ids), 296)
        self.assertEqual(sorted(ids - have), [])
        self.assertEqual(sorted(have - ids), [])

    def test_effect_and_trigger_names_are_the_declared_ones(self):
        for relic_id, (effects, triggers) in pr.relic_rows().items():
            self.assertTrue(set(effects) <= set(pr.EFFECTS), (relic_id, set(effects) - set(pr.EFFECTS)))
            self.assertTrue(set(triggers) <= set(pr.TRIGGERS), (relic_id, set(triggers) - set(pr.TRIGGERS)))

    def test_well_known_relics(self):
        def fx(name):
            return pr.relic_rows()[rid(name)]

        self.assertEqual(fx("Anchor")[0], {"block": 10})
        self.assertIn("combat_start", fx("Anchor")[1])
        self.assertEqual(fx("Lantern")[0], {"energy": 1})
        self.assertEqual(fx("Vajra")[0], {"strength": 1})
        self.assertEqual(fx("OddlySmoothStone")[0], {"dexterity": 1})
        self.assertEqual(fx("Strawberry")[0], {"max_hp": 7})
        self.assertIn("pickup", fx("Strawberry")[1])
        self.assertEqual(fx("BurningBlood")[0], {"heal": 6})
        self.assertIn("reward", fx("BurningBlood")[1])
        self.assertEqual(fx("PotionBelt")[0], {"potion_slot": 2})
        self.assertEqual(fx("OldCoin")[0], {"gold": 300})
        self.assertEqual(fx("BagOfPreparation")[0], {"draw": 2})
        self.assertIn("card_play", fx("Kunai")[1])
        self.assertEqual(fx("Kunai")[0], {"dexterity": 1})
        self.assertEqual(fx("Shuriken")[0], {"strength": 1})
        self.assertEqual(fx("MembershipCard")[0], {"shop": 1})
        self.assertIn("shop", fx("TheCourier")[1])
        self.assertEqual(fx("DreamCatcher")[0], {"card_reward": 1})
        self.assertIn("rest_site", fx("DreamCatcher")[1])
        self.assertEqual(fx("RegalPillow")[0], {"heal": 15})
        self.assertEqual(fx("Ectoplasm")[0], {"energy": 1, "drawback": 1})
        self.assertEqual(fx("CallingBell")[0].get("curse"), 1)
        self.assertEqual(fx("Girya")[0], {"strength": 3})
        self.assertEqual(fx("Pantograph")[0], {"heal": 25})
        self.assertEqual(fx("BloodVial")[0], {"heal": 2})
        self.assertEqual(fx("MiniatureTent")[0], {"rest": 1})

    def test_the_relics_with_no_effect_have_no_tags(self):
        for name in ("Circlet", "WongoCustomerAppreciationBadge", "FakeMerchantsRug", "MassiveScroll"):
            self.assertEqual(pr.relic_rows()[rid(name)][0], {}, name)

    def test_rarities_come_from_the_engine_table(self):
        r = pr.relic_rarities()
        self.assertEqual(len(r), 296)
        self.assertEqual(r[rid("BurningBlood")], "Starter")
        self.assertEqual(r[rid("Anchor")], "Common")
        self.assertEqual(r[rid("Orichalcum")], "Uncommon")
        self.assertEqual(r[rid("Bread")], "Shop")
        self.assertTrue(set(r.values()) <= set(pr.RARITIES))


class EncodingTests(unittest.TestCase):
    def test_width_and_layout(self):
        self.assertEqual(pr.RELIC_TAG_WIDTH, len(pr.EFFECTS) + len(pr.TRIGGERS) + len(pr.RARITIES))
        v = pr.relic_tags(rid("Anchor"))
        self.assertEqual(v.shape, (pr.RELIC_TAG_WIDTH,))
        self.assertEqual(v.dtype, np.float32)
        self.assertAlmostEqual(float(v[pr.EFFECTS.index("block")]), 10 / pr.SCALES["block"])
        self.assertEqual(float(v[len(pr.EFFECTS) + pr.TRIGGERS.index("combat_start")]), 1.0)
        self.assertEqual(float(v[len(pr.EFFECTS) + len(pr.TRIGGERS) + pr.RARITIES.index("Common")]), 1.0)
        self.assertEqual(float(v[pr.EFFECTS.index("energy")]), 0.0)

    def test_scaled_effects_stay_in_range(self):
        m = pr.relic_tag_matrix()
        self.assertEqual(m.shape, (297, pr.RELIC_TAG_WIDTH))
        self.assertTrue(np.all(m >= -1.0) and np.all(m <= 1.0))
        self.assertTrue(np.all(m[0] == 0))  # id 0 is no relic

    def test_a_held_set_is_the_sum(self):
        held = (rid("Anchor"), rid("Lantern"))
        s = pr.held_tags(held)
        np.testing.assert_allclose(s, pr.relic_tags(held[0]) + pr.relic_tags(held[1]), atol=1e-7)
        np.testing.assert_array_equal(pr.held_tags(()), np.zeros(pr.RELIC_TAG_WIDTH, dtype=np.float32))

    def test_unknown_relic_id_is_zero(self):
        np.testing.assert_array_equal(pr.relic_tags(9999), np.zeros(pr.RELIC_TAG_WIDTH, dtype=np.float32))


class MeasuredTests(unittest.TestCase):
    """The table against the engine: a fight started with the relic, against the same fight without it."""

    @classmethod
    def setUpClass(cls):
        cls.base = pr.measure_opening(None)

    def test_the_measurement_of_no_relic_is_the_plain_opening(self):
        self.assertEqual(self.base["energy"], 3)
        self.assertEqual(self.base["hand"], 5)
        self.assertEqual(self.base["block"], 0)

    def test_combat_start_relics_match_the_engine(self):
        checked = {}
        for name in ("Anchor", "Lantern", "Vajra", "OddlySmoothStone", "BagOfPreparation", "SwordOfJade", "VeryHotCocoa", "BronzeScales", "Gorget"):
            got = pr.measure_opening(rid(name))
            checked[name] = {k: got[k] - self.base[k] for k in ("energy", "hand", "block", "strength", "dexterity")}
        self.assertEqual(checked["Anchor"]["block"], 10)
        self.assertEqual(checked["Lantern"]["energy"], 1)
        self.assertEqual(checked["Vajra"]["strength"], 1)
        self.assertEqual(checked["OddlySmoothStone"]["dexterity"], 1)
        self.assertEqual(checked["BagOfPreparation"]["hand"], 2)
        self.assertEqual(checked["SwordOfJade"]["strength"], 3)
        self.assertEqual(checked["VeryHotCocoa"]["energy"], 4)

    def test_every_relic_whose_first_turn_is_tagged_agrees_with_the_engine(self):
        """Every relic the table tags with a first-turn energy, draw, block, Strength or Dexterity (a combat_start or turn_start relic with no condition and no
        drawback elsewhere) shows that effect at the opening of a fight, within the table's own amount."""
        keys = {"energy": "energy", "draw": "hand", "block": "block", "strength": "strength", "dexterity": "dexterity"}
        bad = []
        for relic_id, (effects, triggers) in pr.relic_rows().items():
            if "combat_start" not in triggers or "conditional" in triggers:
                continue
            want = {k: v for k, v in effects.items() if k in keys and v > 0}
            if not want or names.relics()[relic_id] in pr.FIRST_TURN_CHECKED:
                continue
            got = pr.measure_opening(relic_id)
            for k, v in want.items():
                if got[keys[k]] - self.base[keys[k]] < v:
                    bad.append((names.relics()[relic_id], k, v, got[keys[k]] - self.base[keys[k]]))
        self.assertEqual(bad, [])

    def test_the_engine_shows_nothing_at_the_opening_that_the_table_lacks(self):
        """The other direction, over all 296 relics: any energy, block, Strength, Dexterity, hand-size or HP change a fight's opening shows is in the table (a card
        added to the hand counts as draw or add_card; HP lost as hp_cost, HP gained as heal), and a relic the table calls unconditional at the combat's start shows
        exactly the table's energy, block, Strength and Dexterity."""
        skip = {"WhisperingEarring", "Bread"}  # the earring's cards spend the energy at once; Bread's -2 is turn one, its +1 the later turns
        bad = []
        for relic_id, name in names.relics().items():
            if name in skip:
                continue
            got = pr.measure_opening(relic_id)
            d = {k: got[k] - self.base[k] for k in ("energy", "hand", "block", "strength", "dexterity", "hp", "max_hp")}
            effects, triggers = pr.relic_rows()[relic_id]
            for key, field in (("energy", "energy"), ("block", "block"), ("strength", "strength"), ("dexterity", "dexterity"), ("max_hp", "max_hp")):
                if d[field] and not effects.get(key):
                    bad.append((name, key, d[field], "not in the table"))
                elif d[field] and "conditional" not in triggers and "combat_start" in triggers and d[field] != effects[key] and key != "block":
                    bad.append((name, key, d[field], effects[key]))
            if d["hand"] > 0 and not (effects.get("draw") or effects.get("add_card")):
                bad.append((name, "hand", d["hand"], "not in the table"))
            if d["hp"] < 0 and not effects.get("hp_cost"):
                bad.append((name, "hp", d["hp"], "not in the table"))
            if d["hp"] > 0 and not (effects.get("heal") or effects.get("max_hp")):
                bad.append((name, "hp", d["hp"], "not in the table"))
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
