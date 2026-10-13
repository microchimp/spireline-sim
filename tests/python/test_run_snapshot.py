"""tools.fidelity.run_snapshot: the game's save and the emulator's export reach one shape, and
compare() names the first field that differs.

The save below is a cut-down run-start save in the game's own field names (SerializableRun,
SerializablePlayer): prefixes are stripped, potions are placed by slot, upgrade levels fold to the
emulator's flag, encounters compare without their variant suffix, and an emptied rarity queue is
the same as a missing one.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import run_snapshot as rs  # noqa: E402

SAVE = {
    "current_act_index": 0,
    "map_point_history": [[{"map_point_type": "ancient"}]],
    "visited_map_coords": [{"col": 3, "row": 0}],
    "odds": {"unknown_map_point_monster_odds_value": 0.1, "unknown_map_point_elite_odds_value": -1,
             "unknown_map_point_treasure_odds_value": 0.02, "unknown_map_point_shop_odds_value": 0.03},
    "rng": {"counters": {"up_front": 411, "combat_orbs": 0}},
    "shared_relic_grab_bag": {"relic_id_lists": {"common": ["RELIC.ANCHOR"], "rare": []}},
    "events_seen": [],
    "players": [{
        "current_hp": 64, "max_hp": 80, "gold": 99, "max_energy": 3, "max_potion_slot_count": 2,
        "potions": [{"id": "POTION.FIRE_POTION", "slot_index": 1}],
        "deck": [{"id": "CARD.BASH", "current_upgrade_level": 1},
                 {"id": "CARD.STRIKE_IRONCLAD", "enchantment": {"id": "ENCHANTMENT.SHARP", "amount": 2}}],
        "relics": [{"id": "RELIC.BURNING_BLOOD"}],
        "odds": {"card_rarity_odds_value": -0.05, "potion_reward_odds_value": 0.4},
        "rng": {"counters": {"rewards": 0}},
        "relic_grab_bag": {"relic_id_lists": {"common": ["RELIC.VAJRA"]}},
    }],
    "acts": [{"rooms": {"event_ids": ["EVENT.TRIAL"], "normal_encounter_ids": ["ENCOUNTER.CORPSE_SLUGS_WEAK"],
                        "elite_encounter_ids": [], "boss_id": "ENCOUNTER.CEREMONIAL_BEAST_BOSS",
                        "second_boss_id": None, "ancient_id": "EVENT.NEOW", "normal_encounters_visited": 0,
                        "elite_encounters_visited": 0, "events_visited": 1}}],
}

EMU = {
    "act": 0, "floor": 1, "coord": [3, 0], "hp": 64, "max_hp": 80, "gold": 99, "potion_slots": 2,
    "potions": ["", "FIRE_POTION"],
    "deck": [["BASH", 1, None, 0], ["STRIKE_IRONCLAD", 0, "SHARP", 2]],
    "relics": ["BURNING_BLOOD"],
    "odds": {"card_rarity": -0.05, "potion_reward": 0.4, "unknown_monster": 0.1, "unknown_elite": -1,
             "unknown_treasure": 0.02, "unknown_shop": 0.03},
    "rng": {"up_front": 411, "combat_orbs": 0}, "player_rng": {"rewards": 0},
    "grab_bag": {"common": ["VAJRA"]}, "shared_grab_bag": {"common": ["ANCHOR"]},
    "events_seen": [], "shop_removals_used": 0,
    "acts": [{"events": ["TRIAL"], "normal": ["CORPSE_SLUGS"], "elite": [], "boss": "CEREMONIAL_BEAST",
              "second_boss": None, "ancient": "NEOW", "normal_visited": 0, "elite_visited": 0,
              "events_visited": 1}],
}


class RunSnapshotTest(unittest.TestCase):
    def test_same_run_compares_equal(self):
        self.assertEqual(rs.compare(rs.from_save(SAVE), EMU), [])

    def test_first_difference_is_named(self):
        emu = copy.deepcopy(EMU)
        emu["gold"] = 98
        emu["rng"]["up_front"] = 410
        diffs = rs.compare(rs.from_save(SAVE), emu)
        self.assertEqual(diffs[0], ("gold", 99, 98))
        self.assertEqual(diffs[1], ("rng.up_front", 411, 410))

    def test_list_difference_names_position(self):
        emu = copy.deepcopy(EMU)
        emu["acts"][0]["normal"] = ["TOADPOLES"]
        (path, game, mine), = rs.compare(rs.from_save(SAVE), emu)
        self.assertTrue(path.startswith("acts[0].normal[0]"))
        self.assertEqual((game, mine), ("CORPSESLUGS", "TOADPOLES"))

    def test_upgrade_level_is_a_flag(self):
        save = copy.deepcopy(SAVE)
        save["players"][0]["deck"][0]["current_upgrade_level"] = 2
        self.assertEqual(rs.compare(rs.from_save(save), EMU), [])


if __name__ == "__main__":
    unittest.main()


class RunCompareTest(unittest.TestCase):
    """tools.fidelity.run_compare2: offers compare as what the screen lets the player take."""

    def test_reward_offers_compare_as_sets_without_returned_stolen_cards(self):
        from tools.fidelity.run_compare2 import norm_offer
        game = [{"type": "card", "cards": [["CARD.ANGER", 0]]}, {"type": "gold", "amount": 12},
                {"type": "special_card", "card": ["STRIKE_IRONCLAD", 0]}]
        emu = [{"type": "gold", "amount": 12}, {"type": "card", "cards": [["ANGER", 0]]}]
        self.assertEqual(norm_offer("rewards", game, False), norm_offer("rewards", emu, True))

    def test_rest_offers_compare_enabled_options_only(self):
        from tools.fidelity.run_compare2 import norm_offer
        self.assertEqual(norm_offer("rest", [["HEAL", True], ["SMITH", False]], False),
                         norm_offer("rest", [["HEAL", True]], True))

    def test_a_replay_error_is_a_divergence(self):
        from tools.fidelity.run_compare2 import compare_decision
        diffs = compare_decision({"screen": "map", "choice": [0, 1]}, {"error": "map point (0, 1) not offered"}, None)
        self.assertEqual(diffs[0][0], "replay")

    def test_act_change_ignores_where_the_emulator_already_stands(self):
        from tools.fidelity.run_compare2 import compare_decision
        g = rs.from_save(SAVE)
        g["coord"] = None
        e = copy.deepcopy(EMU)
        e["acts"][0]["events_visited"] = 2
        save = copy.deepcopy(SAVE)
        save["visited_map_coords"] = []
        self.assertEqual(compare_decision({"screen": "act_change", "snap": save}, {"snap": e}, None), [])
        self.assertNotEqual(compare_decision({"screen": "map", "snap": save}, {"snap": e}, None), [])
