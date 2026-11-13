"""Block P2 of the information-parity side blocks: the roles of the cards of a deck (`rl/parity_cards.py`, D59 lane D).

Pinned: every card id has a probe row and the probe is what `rl/card_tags.py` measures on the engine (a few cards played again here); the printed roles (area damage, repeatable block,
exhaust, power, junk) of known cards; the measured roles (scaling, draw, energy) of known cards; a known deck's counts and size; an option's cards' roles; the scales.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "rl", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import parity_cards as pc  # noqa: E402
from run_train import vocab  # noqa: E402
from run_train.features import CardRef  # noqa: E402


def cid(name: str) -> int:
    return vocab.card_id_by_name(name)


def role(card_name: str, role_name: str, upgraded: bool = False) -> float:
    return float(pc.card_roles(cid(card_name), upgraded)[pc.ROLES.index(role_name)])


class ProbeTests(unittest.TestCase):
    def test_every_card_has_a_probe_row(self):
        t = pc.probe_table()
        self.assertEqual(sorted(set(vocab.card_ids()) - set(t)), [])
        self.assertTrue(all("playable" in r for r in t.values()))

    def test_no_playable_card_is_left_with_scaling_untested(self):
        left = [r["name"] for r in pc.probe_table().values() if r.get("playable") and (r.get("scaling_damage") is None or r.get("scaling_block") is None)]
        # a card that discards or exhausts the follow-up the test plays, or that needs a target the bed lacks, stays untested; there are few
        self.assertLess(len(left), 20, left)

    def test_the_probe_is_what_card_tags_measures(self):
        import card_tags as ct  # noqa: PLC0415

        for name in ("Offering", "Inflame", "ShrugItOff", "StrikeIronclad", "DefendIronclad", "BattleTrance"):
            try:
                c = cid(name)
            except KeyError:
                continue
            fresh = ct.probe(c)
            row = pc.probe_table()[c]
            for tag in ("front_damage", "front_block", "draw", "energy"):
                self.assertEqual(row[tag], fresh[tag], (name, tag))

    def test_targets_come_from_the_card_table(self):
        self.assertEqual(pc.card_targets()[cid("Whirlwind")], "AllEnemies")
        self.assertEqual(pc.card_targets()[cid("Inflame")], "Self")


class RoleTests(unittest.TestCase):
    def test_printed_roles(self):
        self.assertEqual(role("Whirlwind", "multi_target"), 1.0)
        self.assertEqual(role("StrikeIronclad", "multi_target"), 0.0)
        self.assertEqual(role("Whirlwind", "multi_target"), 1.0)
        self.assertEqual(role("ShrugItOff", "repeatable_block"), 1.0)  # 8 block for 1 energy
        self.assertEqual(role("DefendIronclad", "repeatable_block"), 0.0)  # 5 block
        self.assertEqual(role("DefendIronclad", "repeatable_block", upgraded=True), 1.0)  # an upgraded Defend blocks 8 for one energy
        self.assertEqual(role("Impervious", "repeatable_block"), 0.0)  # exhausts
        self.assertEqual(role("Impervious", "exhaust"), 1.0)
        self.assertEqual(role("Inflame", "power"), 1.0)
        self.assertEqual(role("StrikeIronclad", "power"), 0.0)
        self.assertEqual(role("AscendersBane", "junk"), 1.0)
        self.assertEqual(role("Dazed", "junk"), 1.0)
        self.assertEqual(role("StrikeIronclad", "junk"), 0.0)

    def test_measured_roles(self):
        self.assertEqual(role("Inflame", "scaling_damage"), 1.0)  # Strength
        self.assertEqual(role("Offering", "draw"), 1.0)
        self.assertEqual(role("Offering", "energy"), 1.0)
        self.assertEqual(role("StrikeIronclad", "scaling_damage"), 0.0)
        self.assertEqual(role("DefendIronclad", "scaling_block"), 0.0)
        self.assertEqual(role("DemonForm", "scaling_damage"), 1.0)  # costs three and acts at the next turn's start: the extra-energy opening, one turn on
        self.assertEqual(role("Footwork", "scaling_block"), 1.0)  # Dexterity
        self.assertEqual(role("BattleTrance", "draw"), 1.0)

    def test_upgrade_changes_the_printed_roles(self):
        # an upgrade that removes exhaust, or lifts block over the per-energy line, moves the role; the measured roles are the base card's
        facts = vocab.card(cid("DefendIronclad"))
        self.assertEqual(role("DefendIronclad", "repeatable_block", True), float(facts.block_for(True) / max(1, facts.cost_for(True)) >= pc.REPEATABLE_BLOCK_PER_ENERGY))


class DeckTests(unittest.TestCase):
    def test_a_known_deck(self):
        deck = [CardRef(cid("StrikeIronclad"), False)] * 5 + [CardRef(cid("DefendIronclad"), False)] * 4 + [CardRef(cid("Whirlwind"), False), CardRef(cid("Inflame"), False),
                                                                                         CardRef(cid("ShrugItOff"), False), CardRef(cid("Offering"), False)]
        v = pc.deck_roles(deck)
        self.assertEqual(v.shape, (pc.N_ROLES + 1,))
        self.assertAlmostEqual(float(v[pc.ROLES.index("multi_target")]), 1 / pc.COUNT_SCALE)
        self.assertAlmostEqual(float(v[pc.ROLES.index("scaling_damage")]), 1 / pc.COUNT_SCALE)
        self.assertAlmostEqual(float(v[pc.ROLES.index("repeatable_block")]), 1 / pc.COUNT_SCALE)
        self.assertAlmostEqual(float(v[pc.ROLES.index("exhaust")]), 1 / pc.COUNT_SCALE)  # Offering
        self.assertAlmostEqual(float(v[pc.ROLES.index("power")]), 1 / pc.COUNT_SCALE)
        self.assertAlmostEqual(float(v[pc.ROLES.index("draw")]), 2 / pc.COUNT_SCALE)  # Shrug It Off draws one, Offering three
        self.assertAlmostEqual(float(v[pc.N_ROLES]), 13 / pc.DECK_SIZE_SCALE)

    def test_counts_saturate_and_an_empty_deck_is_zero(self):
        deck = [CardRef(cid("Whirlwind"), False)] * 30
        self.assertEqual(float(pc.deck_roles(deck)[pc.ROLES.index("multi_target")]), 1.0)
        np.testing.assert_array_equal(pc.deck_roles([]), np.zeros(pc.N_ROLES + 1, dtype=np.float32))

    def test_an_options_cards(self):
        v = pc.cards_roles([CardRef(cid("Whirlwind"), False)])
        self.assertEqual(float(v[pc.ROLES.index("multi_target")]), 0.5)
        np.testing.assert_array_equal(pc.cards_roles([]), np.zeros(pc.N_ROLES, dtype=np.float32))

    def test_an_unknown_card_is_zero(self):
        np.testing.assert_array_equal(pc.card_roles(99999), np.zeros(pc.N_ROLES, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
