"""referee/hidden_diff.py: the lockstep's second, unranked comparison, on synthetic observations (no native library).

The readers: the game's `obs.combat` (player block and powers, per enemy hp, block and powers), the copy's observation buffer
(block at +2 of an enemy slot and at 2 for the player, powers in the (BuffId, amount) slots), and the relics (the driver's
anchor-state list against the copy's counters and melted list)."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT))

import hidden_diff as hd  # noqa: E402

# The buffer's geometry, small and made up: the player's buffs at 10 (3 slots), the enemies from 20, 12 wide (hp, max, block,
# intent, magnitude, then 3 buff slots of 2), 3 enemy slots.
GEOM = dict(player_buffs_at=10, player_buff_slots=3, enemies_at=20, enemy_size=11, enemy_slots=3, enemy_buff_slots=3)
NAMES = {0: "Strength", 2: "Vulnerable", 3: "Weak", 7: "Ritual", 9: "DemonForm"}


def buffer(player_block=0, player=(), enemies=()):
    """A buffer holding the player's block and buffs [(id, amount)] and enemies [(hp, max, block, [(id, amount)])]."""
    obs = [0] * 60
    obs[2] = player_block
    for k, (i, a) in enumerate(player):
        obs[10 + 2 * k], obs[11 + 2 * k] = i, a
    for k, (hp, mx, block, buffs) in enumerate(enemies):
        at = 20 + 11 * k
        obs[at], obs[at + 1], obs[at + 2] = hp, mx, block
        for j, (i, a) in enumerate(buffs):
            obs[at + 5 + 2 * j], obs[at + 6 + 2 * j] = i, a
    return obs


def emu(obs):
    return hd.emu_view(obs, names=NAMES, **GEOM)


def game(player_block=0, player=(), enemies=()):
    """A referee observation: player block and powers, enemies [(hp, block, [(power, amount)])]."""
    return {"combat": {"player": {"hp": 70, "block": player_block, "powers": [list(p) for p in player]},
                       "enemies": [{"name": "X", "hp": hp, "block": b, "powers": [list(p) for p in ps]} for hp, b, ps in enemies]}}


class FightTests(unittest.TestCase):
    def diff(self, g, e):
        return hd.diff(hd.game_view(g), emu(e))

    def test_two_sides_that_agree_have_no_difference(self):
        g = game(4, [("STRENGTH_POWER", 2)], [(30, 6, [("VULNERABLE_POWER", 2), ("RITUAL_POWER", 3)]), (12, 0, [])])
        e = buffer(4, [(0, 2)], [(30, 40, 6, [(2, 2), (7, 3)]), (12, 20, 0, [])])
        self.assertEqual(self.diff(g, e), [])

    def test_an_enemys_block_that_differs_is_named_by_its_place(self):
        g = game(enemies=[(30, 0, []), (12, 5, [])])
        e = buffer(enemies=[(30, 40, 0, []), (12, 20, 0, [])])
        self.assertEqual(self.diff(g, e), [[".combat.enemies[1].block", 5, 0]])

    def test_a_power_that_differs_in_amount_or_is_missing_on_one_side(self):
        g = game(enemies=[(30, 0, [("VULNERABLE_POWER", 2), ("RITUAL_POWER", 3)])])
        e = buffer(enemies=[(30, 40, 0, [(2, 1)])])
        self.assertEqual(self.diff(g, e), [[".combat.enemies[0].powers.RITUAL", 3, None],
                                           [".combat.enemies[0].powers.VULNERABLE", 2, 1]])

    def test_the_players_block_and_powers(self):
        g = game(7, [("DEMON_FORM_POWER", 2)], [(30, 0, [])])
        e = buffer(3, [(9, 3), (3, 1)], [(30, 40, 0, [])])
        self.assertEqual(self.diff(g, e), [[".combat.player.block", 7, 3], [".combat.player.powers.DEMON_FORM", 2, 3],
                                           [".combat.player.powers.WEAK", None, 1]])

    def test_strength_is_id_zero_and_an_empty_slot_is_not_a_power(self):
        e = emu(buffer(player=[(0, -2)]))
        self.assertEqual(e["player"]["powers"], {"STRENGTH": -2})
        self.assertEqual(emu(buffer())["player"]["powers"], {})

    def test_the_games_names_are_the_fight_comparers_names(self):
        # the folds and renames tools/fidelity/fight_compare.py applies: the game's Setup Strike is the copy's Temporary Strength
        g = game(enemies=[(30, 0, [("SETUP_STRIKE_POWER", 3)])])
        self.assertEqual(hd.game_view(g)["enemies"][0]["powers"], {"TEMPORARY_STRENGTH": 3})

    def test_a_corpse_has_no_block_or_powers_to_compare(self):
        g = game(enemies=[(0, 0, []), (12, 0, [])])
        e = buffer(enemies=[(0, 40, 9, [(2, 4)]), (12, 20, 0, [])])
        self.assertEqual(self.diff(g, e), [])

    def test_a_roster_of_another_size_is_one_entry_the_ranked_comparison_already_reports(self):
        g = game(enemies=[(30, 0, [])])
        e = buffer(enemies=[(30, 40, 0, []), (12, 20, 0, [])])
        self.assertEqual(self.diff(g, e), [[".combat.enemies.count", 1, 2]])

    def test_full_buff_slots_may_hide_a_power_so_a_game_only_power_is_not_a_difference_there(self):
        g = game(player=[("STRENGTH_POWER", 1), ("WEAK_POWER", 1), ("DEMON_FORM_POWER", 1), ("RITUAL_POWER", 1)])
        full = buffer(player=[(0, 1), (3, 1), (9, 1)])  # three slots, all taken
        self.assertEqual(self.diff(g, full), [])
        roomy = buffer(player=[(0, 1), (3, 1)])  # a slot free: the game's Demon Form and Ritual should show
        self.assertEqual([x[0] for x in self.diff(g, roomy)], [".combat.player.powers.DEMON_FORM", ".combat.player.powers.RITUAL"])
        # a power the copy shows with another amount is a difference even when its slots are full
        wrong = buffer(player=[(0, 2), (3, 1), (9, 1)])
        self.assertEqual(self.diff(g, wrong), [[".combat.player.powers.STRENGTH", 1, 2]])

    def test_a_field_the_stream_does_not_carry_is_not_compared(self):
        g = {"combat": {"enemies": [{"name": "X", "hp": 30}]}}  # a driver from before block and powers
        e = buffer(7, [(0, 2)], [(30, 40, 9, [(2, 4)])])
        self.assertEqual(self.diff(g, e), [])
        self.assertIsNone(hd.game_view({"hp": 3}))  # outside a fight

    def test_values_are_compared_through_the_observation_bound(self):
        g = game(enemies=[(30, 999999999, [])])
        e = buffer(enemies=[(30, 40, 2 ** 15, [])])
        self.assertEqual(self.diff(g, e), [])


class RelicTests(unittest.TestCase):
    WRITES = staticmethod(lambda relics: [(r["id"], int((r.get("ints") or {}).get("AttacksPlayed", 0))) for r in relics
                                          if r["id"] == "PEN_NIB"])

    def diff(self, game_relics, counters, melted=()):
        return hd.relic_diff(game_relics, counters, list(melted), self.WRITES)

    def test_a_counter_that_differs_is_named_with_both_values(self):
        got = self.diff([{"id": "PEN_NIB", "ints": {"AttacksPlayed": 8}}], {"PEN_NIB": 9})
        self.assertEqual(got, [[".relics.PEN_NIB.counter", 8, 9]])
        self.assertEqual(self.diff([{"id": "PEN_NIB", "ints": {"AttacksPlayed": 8}}], {"PEN_NIB": 8}), [])

    def test_a_relic_the_copy_does_not_hold_is_the_ranked_relic_lists_business(self):
        self.assertEqual(self.diff([{"id": "PEN_NIB", "ints": {"AttacksPlayed": 8}}], {}), [])

    def test_the_melted_flag_is_compared_as_a_list_and_a_melted_relics_counter_is_not(self):
        melted = {"id": "PEN_NIB", "ints": {"AttacksPlayed": 8}, "bools": {"IsMelted": True}}
        self.assertEqual(self.diff([melted], {}, []), [[".relics.melted", ["PEN_NIB"], []]])
        self.assertEqual(self.diff([melted], {}, ["PEN_NIB"]), [])
        self.assertEqual(self.diff([melted], {"PEN_NIB": 99}, ["PEN_NIB"]), [])  # the copy's counter of a melted relic is nobody's

    def test_a_library_that_cannot_say_which_relics_are_melted_compares_no_flag(self):
        melted = {"id": "SAI", "bools": {"IsMelted": True}}
        self.assertEqual(hd.relic_diff([melted], {}, None, self.WRITES), [])

    def test_no_anchor_state_relics_is_no_difference(self):
        self.assertEqual(hd.relic_diff(None, {"PEN_NIB": 3}, [], self.WRITES), [])


class KindTests(unittest.TestCase):
    def test_the_summarys_names(self):
        k = hd.kind
        self.assertEqual(k(".combat.enemies[3].block"), "enemy_block")
        self.assertEqual(k(".combat.enemies[0].powers.VULNERABLE"), "enemy_power:VULNERABLE")
        self.assertEqual(k(".combat.player.block"), "player_block")
        self.assertEqual(k(".combat.player.powers.DEMON_FORM"), "player_power:DEMON_FORM")
        self.assertEqual(k(".combat.enemies.count"), "roster")
        self.assertEqual(k(".relics.PEN_NIB.counter"), "relic_counter:PEN_NIB")
        self.assertEqual(k(".relics.melted"), "relic_melted")
        self.assertEqual(k(".something.else"), "other")


if __name__ == "__main__":
    unittest.main()
