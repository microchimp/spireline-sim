"""tools/fidelity/table_check.py static-data checks: cards, relics, potions, powers, monsters.

Synthetic game and emulator tables, no game and no dotnet.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.fidelity import table_check as tc  # noqa: E402


def card(damage=6, up_damage=9, kw="", up_kw="", cost=1, up_cost=1, block=None, up_block=None, **extra):
    base = {"vars": f"Damage={damage}" + (f";Block={block}" if block is not None else ""), "keywords": kw,
            "cost": str(cost), "star_cost": "-1"}
    up = {"vars": f"Damage={up_damage}" + (f";Block={up_block}" if up_block is not None else ""),
          "keywords": up_kw, "cost": str(up_cost), "star_cost": "-1"}
    return {"cost_x": "False", "detail": {"base": base, "upgraded": up}, **extra}


def emu_card(**kw):
    d = {"upgradable": True, "base_damage": 6, "base_block": 0, "upgrade_damage": 3, "upgrade_block": 0,
         "upgrade_cost": 0, "keywords": [], "keywords_upgraded": [], "star_cost": -1}
    d.update(kw)
    return d


class CardStatic(unittest.TestCase):
    def setUp(self):
        tc.lines.clear()
        tc.counts.clear()

    def test_equal_cards_report_nothing(self):
        tc.check_cards_static({"A": card()}, {"A": emu_card()})
        self.assertEqual(tc.counts, {})

    def test_damage_and_upgrade_change(self):
        tc.check_cards_static({"A": card(damage=7, up_damage=12)}, {"A": emu_card()})
        self.assertEqual(tc.counts["card_static"], 2)
        self.assertIn("base damage A: game 7 vs emulator 6", tc.lines[0])
        self.assertIn("upgrade damage change A: game 5 vs emulator 3", tc.lines[1])

    def test_osty_damage_var_counts_as_damage(self):
        g = card()
        g["detail"]["base"]["vars"] = "OstyDamage=6"
        g["detail"]["upgraded"]["vars"] = "OstyDamage=9"
        tc.check_cards_static({"POKE": g}, {"POKE": emu_card()})
        self.assertEqual(tc.lines, [])
        tc.check_cards_static({"POKE": g}, {"POKE": emu_card(base_damage=7)})
        self.assertEqual(len(tc.lines), 1)

    def test_keywords_and_upgrade_cost(self):
        g = card(kw="Exhaust", up_kw="", cost=2, up_cost=1)
        tc.check_cards_static({"A": g}, {"A": emu_card(keywords=["Exhaust"], keywords_upgraded=["Exhaust"], upgrade_cost=0)})
        text = "\n".join(tc.lines)
        self.assertIn("keywords upgraded A: game [] vs emulator ['Exhaust']", text)
        self.assertIn("upgrade cost change A: game -1 vs emulator 0", text)

    def test_not_upgradable_skips_upgrade_fields(self):
        tc.check_cards_static({"A": card()}, {"A": emu_card(upgradable=False)})
        self.assertEqual(tc.counts, {})

    def test_missing_card_and_detail_error(self):
        tc.check_cards_static({"A": card(), "B": {"detail": {"error": "boom"}}}, {"B": emu_card()})
        text = "\n".join(tc.lines)
        self.assertIn("1 game cards not in emulator: A", text)
        self.assertIn("B: game detail unavailable (boom)", text)


class OtherEntities(unittest.TestCase):
    def setUp(self):
        tc.lines.clear()
        tc.counts.clear()

    def test_relic_potion_power(self):
        game = {
            "all_relics": {"R": {"rarity": "Common", "upon_pickup": "True", "allowed_in_shops": "True"}},
            "all_potions": {"P": {"rarity": "Rare", "target": "Self"}},
            "powers": {"X_POWER": {"type": "Buff", "stack_type": "Counter"}},
        }
        emu = {
            "relics": {"R": {"rarity": "Rare", "upon_pickup": False, "allowed_in_shops": True}},
            "all_potions": {"P": {"rarity": "Rare", "target": "AnyEnemy"}},
            "powers": {"X_POWER": {"is_buff": False, "stack_type": "Counter"}},
        }
        tc.check_simple_entities(game, emu)
        self.assertEqual(tc.counts, {"relic_static": 2, "potion_static": 1, "power_static": 1})

    def test_foul_potion_target_is_the_combat_one(self):
        game = {"all_potions": {"FOUL_POTION": {"rarity": "Common", "target": "TargetedNoCreature"}}}
        emu = {"all_potions": {"FOUL_POTION": {"rarity": "Common", "target": "AllEnemies"}}}
        tc.check_simple_entities(game, emu)
        self.assertEqual(tc.counts, {})
        emu["all_potions"]["FOUL_POTION"]["target"] = "Self"
        tc.check_simple_entities(game, emu)
        self.assertEqual(tc.counts, {"potion_static": 1})

    def test_monster_numbers_the_export_cannot_see(self):
        mv = lambda d: {"id": "M", "intents": [{"type": "Attack", "damage": d, "repeats": "1"}]}  # noqa: E731
        game = {"monsters": {
            "WATERFALLGIANT": {"min_hp": "1", "max_hp": "1", "moves": [mv("15")]},
            "THEFORGOTTEN": {"min_hp": "1", "max_hp": "1", "moves": [mv("ERR InvalidOperationException")]},
        }}
        emu = {"monsters": {
            "WATERFALLGIANT": {"min_hp": 1, "max_hp": 1, "attacks": ["15x1", "20x1", "25x1"]},
            "THEFORGOTTEN": {"min_hp": 1, "max_hp": 1, "attacks": ["13x1"]},
        }}
        tc.check_simple_entities(game, emu)
        self.assertEqual(tc.counts, {})

    def test_monsters(self):
        game = {"monsters": {
            "AXE_BOT": {"min_hp": "70", "max_hp": "78", "moves": [
                {"id": "A", "intents": [{"type": "Attack", "damage": "12", "repeats": "1"}]},
                {"id": "B", "intents": [{"type": "Defend"}]}]},
            "BROKEN": {"min_hp": "ERR X", "max_hp": "1", "error": "NullReferenceException"},
        }}
        emu = {"monsters": {
            "AXEBOT": {"min_hp": 70, "max_hp": 78, "attacks": ["12x1", "9x2"]},
            "BROKEN": {"min_hp": 1, "max_hp": 1, "attacks": []},
        }}
        tc.check_simple_entities(game, emu)
        text = "\n".join(tc.lines)
        self.assertIn("attack moves (damage x hits) AXEBOT: game ['12x1'] vs emulator ['12x1', '9x2']", text)
        self.assertIn("BROKEN: game moves unavailable (NullReferenceException)", text)


if __name__ == "__main__":
    unittest.main()
