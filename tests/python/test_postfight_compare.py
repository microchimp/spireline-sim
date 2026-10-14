"""The step that ends a fight is compared like any other (tools.fidelity fight_compare and its callers).

The comparer used to see nothing of that step but ``in_combat``: the game's snapshot after the last enemy dies
and the emulator's dump of it both reduced to ``{"in_combat": False}``, so a fight the game left at 76 hit
points and the emulator at 68 was scored identical (reference set fight G009: the emulator takes a hit the
game's Thorns prevent). These tests build such a fight from synthetic steps in the shapes the referee and
EmuDump write, break the emulator's end of it on purpose, and require every judge to say so:
fight_compare.report, coverage.first_divergence, reanchor_compare.analyse and the corpus's judges.
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import corpus as cp  # noqa: E402
from tools.fidelity import coverage  # noqa: E402
from tools.fidelity import fight_compare as fc  # noqa: E402
from tools.fidelity import reanchor_compare as rc  # noqa: E402

ID_MAP = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))
NM = fc.names(ID_MAP)
CID, EID = ID_MAP["cards"], ID_MAP["enemies"]
STRIKE, SLUG = "STRIKE_IRONCLAD", "CORPSE_SLUG"
PLAY = {"kind": "play", "hand_index": 0, "card": STRIKE, "target": 0, "target_slot": 0, "played": True}
END = {"kind": "end_turn"}

STREAMS = ("Shuffle", "CombatTargets", "MonsterAi", "CombatCardSelection", "CombatCardGeneration",
           "CombatEnergyCosts", "CombatPotionGeneration", "CombatOrbGeneration", "Niche")
# The stream the referee names but a fight never moves; it must not matter.
OTHER = {"UpFront": {"seed": 5, "counter": 423}}


def game_rng(**counters):
    return {**{k: {"seed": 1, "counter": counters.get(k, 0)} for k in STREAMS}, **OTHER}


def game_fight(hp=27, php=70):
    return {"in_combat": True, "round": 1,
            "player": {"hp": php, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "hand": [[STRIKE, False]], "draw": [], "discard": [], "exhaust": [],
            "enemies": [{"name": SLUG, "hp": hp, "max_hp": 27, "block": 0, "powers": []}],
            "rng": game_rng()}


def game_won(hp=76, max_hp=80, rng=None, **extra):
    """The game's snapshot after a won fight: Player.AfterCombatEnd has emptied the player and the combat is gone."""
    return {"in_combat": False, "round": 3,
            "player": {"hp": hp, "max_hp": max_hp, "block": 0, "energy": 1, "powers": []},
            "hand": [], "draw": [], "discard": [], "exhaust": [], "enemies": [],
            "rng": game_rng(**(rng or {})), **extra}


def game_lost(**residue):
    """A lost fight: the game stops the combat where the player died, the piles and the roster stay as they were."""
    st = game_fight(hp=14, php=0)
    st["in_combat"] = False
    st["hand"], st["discard"] = [[STRIKE, False], [STRIKE, False]], [["DEFEND_IRONCLAD", False]]
    st["enemies"][0]["powers"] = [["STRENGTH", 6]]
    st.update(residue)
    return st


def emu_fight(hp=27, php=70):
    card = {"def_id": CID["StrikeIronclad"], "upgraded": False}
    return {"in_combat": True, "draw_order": [], "rng": {k: 0 for k in STREAMS},
            "snapshot": {"player": {"hp": php, "max_hp": 80, "block": 0, "energy": 3, "buffs": []},
                         "piles": {"hand": [card], "discard": [], "exhaust": []},
                         "enemies": [{"def_id": EID["CorpseSlug"], "hp": hp, "block": 0, "buffs": []}]}}


def emu_won(hp=76, max_hp=80, rng=None, gold=99, **final):
    """EmuDump's dump of the step that won the fight (Fight.Final): the run's state, all streams, gold."""
    return {"in_combat": False, "run_phase": "RelicReward",
            "final": {"won": True, "player": {"hp": hp, "max_hp": max_hp, "gold": gold},
                      "rng": {**{k: 0 for k in STREAMS}, "UpFront": 423, "Niche": 0, **(rng or {})}, **final}}


def emu_lost(hp=0, max_hp=80, **residue):
    """EmuDump's dump of a lost fight: that, and the combat the emulator finished the enemy phase of."""
    st = emu_fight(hp=14, php=hp)
    st["in_combat"] = False
    st["final"] = {"won": False}
    st["snapshot"]["player"]["max_hp"] = max_hp  # Scrolls of Biting: another max HP than the game's at the end
    st["snapshot"]["piles"]["discard"] = [{"def_id": CID["DefendIronclad"], "upgraded": False}]
    st["snapshot"]["enemies"][0]["buffs"] = [{"name": "Strength", "amount": 9}]  # the game's still has 6
    st.update(residue)
    return st


def steps_of(*pairs):
    """(action, game state, emulator state) rows to the game's and the emulator's step lists."""
    return ({"seed": "T", "steps": [{"action": a, "state": g} for a, g, _ in pairs]},
            {"seed": "T", "steps": [{"action": a, "state": e} for a, _, e in pairs]})


def winning_fight(emu_end):
    """A start, two plays and a killing blow after which the game stands at 76 of 80."""
    return steps_of((None, game_fight(27), emu_fight(27)),
                    (PLAY, game_fight(21), emu_fight(21)),
                    (PLAY, game_fight(15), emu_fight(15)),
                    (PLAY, game_won(76), emu_end))


class EndViewTest(unittest.TestCase):
    """game_view / emu_view of the step that ends the fight."""

    def d(self, game, emu):
        return fc.diff(fc.game_view(game), fc.emu_view(emu, NM))

    def test_matching_end_states_are_identical(self):
        self.assertEqual(self.d(game_won(), emu_won()), [])

    def test_the_hp_the_fight_ends_at_is_compared(self):
        # G009: the emulator takes a hit the game's Thorns prevent, 68 where the game has 76.
        self.assertEqual(self.d(game_won(76), emu_won(68)), [(".player.hp", 76, 68)])

    def test_max_hp_is_compared(self):
        self.assertEqual(self.d(game_won(max_hp=81), emu_won(max_hp=80)), [(".player.max_hp", 81, 80)])

    def test_the_rng_positions_the_fight_leaves_are_compared(self):
        self.assertEqual(self.d(game_won(rng={"Shuffle": 14}), emu_won(rng={"Shuffle": 14})), [])
        self.assertEqual(self.d(game_won(rng={"Shuffle": 14}), emu_won(rng={"Shuffle": 19})), [(".rng.Shuffle", 14, 19)])
        # a stream the fight never moves is not part of the view
        self.assertEqual(self.d(game_won(), emu_won(rng={"UpFront": 9999})), [])

    def test_the_belt_and_the_relics_are_compared_where_the_record_has_them(self):
        g = game_won(potions=["FIRE_POTION", None], relics=["BURNING_BLOOD"])
        ok = emu_won(potions=["FIRE_POTION", None], relics=["BURNING_BLOOD"], relic_counters=[0])
        self.assertEqual(self.d(g, ok), [])
        no_potion = emu_won(potions=[None, None], relics=["BURNING_BLOOD"], relic_counters=[0])
        self.assertEqual(self.d(g, no_potion), [(".potions", ["FIRE_POTION", None], [None, None])])
        no_relic = emu_won(potions=["FIRE_POTION", None], relics=[], relic_counters=[])
        self.assertEqual(self.d(g, no_relic), [(".relics", ["BURNING_BLOOD"], [])])
        # a record without the extras (the emulator then writes none either) compares neither
        self.assertEqual(self.d(game_won(), emu_won()), [])

    def test_gold_and_relic_counters_count_once_both_sides_have_them(self):
        self.assertEqual(self.d(game_won(), emu_won(gold=432)), [])  # the game's snapshot has no gold yet
        g = game_won()
        g["player"]["gold"] = 99
        self.assertEqual(self.d(g, emu_won(gold=432)), [(".player.gold", 99, 432)])
        self.assertEqual(self.d(g, emu_won(gold=99)), [])
        # the same for the relics' counters
        g = game_won(relics=["PEN_NIB"], relic_counters=[3])
        self.assertEqual(self.d(g, emu_won(relics=["PEN_NIB"], relic_counters=[4])), [(".relic_counters", [3], [4])])
        self.assertEqual(self.d(game_won(relics=["PEN_NIB"]), emu_won(relics=["PEN_NIB"], relic_counters=[4])), [])

    def test_what_the_game_clears_is_empty_in_both(self):
        # Player.AfterCombatEnd: a snapshot that still shows block, powers or piles is not a torn-down fight
        left = game_won()
        left["player"]["block"], left["player"]["powers"] = 5, [["THORNS_POWER", 15]]
        left["discard"] = [["DEFEND_IRONCLAD", False]]
        paths = [p for p, _, _ in self.d(left, emu_won())]
        self.assertEqual(paths, [".discard", ".player.block", ".player.powers"])

    def test_one_side_still_fighting_is_judged_as_before(self):
        # a fight over on one side only: the side that is over is its flag alone, so the step reads as it did when
        # the end of a fight had no view (reanchor_compare's stop on the enemies reads these paths)
        for game, emu in ((game_fight(), emu_won()), (game_fight(), emu_lost())):
            got = self.d(game, emu)
            self.assertEqual(got, fc.diff(fc.game_view(game), {"in_combat": False}))
            self.assertIn(".in_combat", [p for p, _, _ in got])
            self.assertIn(".enemies", [p for p, _, _ in got])
        got = self.d(game_won(), emu_fight())
        self.assertEqual(got, fc.diff({"in_combat": False}, fc.emu_view(emu_fight(), NM)))
        self.assertIn(".in_combat", [p for p, _, _ in got])

    def test_a_win_against_a_loss_is_the_only_difference_reported(self):
        self.assertEqual(self.d(game_lost(), emu_won()), [(".outcome", "lost", "won")])
        self.assertEqual(self.d(game_won(), emu_lost()), [(".outcome", "won", "lost")])

    def test_a_lost_fight_compares_that_it_was_lost_not_where_each_side_stopped_the_combat(self):
        # the game froze the combat at the killing hit (Strength 6, one card in the discard, max HP 77); the
        # emulator finished the enemy phase (Strength 9, a status card, max HP 80): neither is state of anything
        g = game_lost()
        g["player"]["max_hp"] = 77
        self.assertEqual(self.d(g, emu_lost(max_hp=80)), [])
        # a player the emulator kept alive is the difference
        self.assertEqual(self.d(game_lost(), emu_won()), [(".outcome", "lost", "won")])

    def test_an_emulator_dump_that_wrote_nothing_after_the_fight_is_no_match(self):
        # the old comparison could not tell: an emulator that said nothing was identical to any end
        self.assertNotEqual(self.d(game_won(), {"in_combat": False}), [])

    def test_records_that_hold_no_more_than_the_flag_compare_nothing(self):
        self.assertEqual(self.d({"in_combat": False}, {"in_combat": False}), [])

    def test_an_overkill_is_still_a_loss(self):
        self.assertEqual(self.d(game_lost(), emu_lost(hp=-4)), [])


class EveryJudgeSeesTheEndTest(unittest.TestCase):
    """The deliberately broken fight: the emulator is 8 hit points short after the killing blow."""

    def setUp(self):
        self.ok = winning_fight(emu_won(76))
        self.bad = winning_fight(emu_won(68))

    def test_the_intact_fight_is_identical_everywhere(self):
        g, e = self.ok
        lines, first = fc.report(g, e, NM)
        self.assertIsNone(first)
        self.assertIsNone(coverage.first_divergence(g, e, NM))
        self.assertEqual(cp.judge_fight(g, e, NM), {"fd": None})
        an = rc.analyse(g, e, NM)
        self.assertEqual([s["verdict"] for s in an["steps"]], ["match"] * 4)

    def test_report_names_the_last_step_and_the_hit_points(self):
        g, e = self.bad
        lines, first = fc.report(g, e, NM)
        self.assertEqual(first, 3)
        self.assertEqual(lines[0], "T: first difference at step 3 after {'kind': 'play', 'hand_index': 0, 'card': "
                                   "'STRIKE_IRONCLAD', 'target': 0, 'target_slot': 0, 'played': True}")
        self.assertIn(".player.hp: game 76  emulator 68", lines[1])

    def test_coverage_blames_the_card_that_ended_the_fight(self):
        g, e = self.bad
        self.assertEqual(coverage.first_divergence(g, e, NM), 3)
        cov = {"cards": {}}
        got = coverage.update(cov, [g], [e], NM, "abc")
        self.assertEqual(got["new_suspects"], 1)
        self.assertEqual(cov["cards"][STRIKE]["suspect_commit"], "abc")
        self.assertEqual(cov["cards"][STRIKE]["verified"], 2)  # the two plays before it

    def test_the_plain_judge_names_the_step_and_the_field(self):
        g, e = self.bad
        got = cp.judge_fight(g, e, NM)
        self.assertEqual((got["fd"], got["field"], got["path"], got["card"]), (3, ".player.hp", ".player.hp", STRIKE))

    def test_the_reanchored_judge_marks_the_last_step_a_primary_divergence(self):
        g, e = self.bad
        an = rc.analyse(g, e, NM)
        self.assertEqual([s["verdict"] for s in an["steps"]], ["match", "match", "match", "diverge"])
        last = an["steps"][-1]
        self.assertEqual((last["paths"], last["primary"], an["first_divergence"]), ([".player.hp"], True, 3))
        got = cp.judge_reanchor(g, e, NM, None)
        self.assertEqual((got["rv"], got["dp"]), ("MMD", {"3": [".player.hp"]}))
        self.assertEqual(rc.summarise([an])["totals"]["diverging_primary"], 1)

    def test_a_fight_that_ends_in_a_loss_in_both_stays_identical_despite_the_leftover(self):
        g, e = steps_of((None, game_fight(27), emu_fight(27)),
                        (END, game_lost(), emu_lost()))
        self.assertIsNone(coverage.first_divergence(g, e, NM))
        self.assertEqual(rc.analyse(g, e, NM)["steps"][-1]["verdict"], "match")

    def test_a_fight_one_side_ends_early_or_late_is_still_the_in_combat_difference(self):
        g, e = steps_of((None, game_fight(27), emu_fight(27)),
                        (PLAY, game_won(76), emu_fight(15)))
        self.assertEqual(coverage.first_divergence(g, e, NM), 1)
        self.assertIn(".in_combat", rc.analyse(g, e, NM)["steps"][-1]["paths"])
        g, e = steps_of((None, game_fight(27), emu_fight(27)),
                        (PLAY, game_fight(21), emu_won(76)))
        self.assertEqual(coverage.first_divergence(g, e, NM), 1)
        self.assertIn(".in_combat", rc.analyse(g, e, NM)["steps"][-1]["paths"])

    def test_gold_alone_cannot_break_a_fight_the_game_did_not_record_it_for(self):
        g, e = self.ok
        e2 = copy.deepcopy(e)
        e2["steps"][-1]["state"]["final"]["player"]["gold"] = 12345
        self.assertIsNone(coverage.first_divergence(g, e2, NM))


if __name__ == "__main__":
    unittest.main()
