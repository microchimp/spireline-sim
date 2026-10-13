"""tools.fidelity.invariants: each check fires on a state built to break it, and not on a sound fight.

The fights are game-shaped recordings written by hand (the shape referee/driver/Combat.cs records), so the
checks are tested without the game or the emulator. Card ids come from the real tables: STRIKE_IRONCLAD,
DEFEND_IRONCLAD, BASH and INFLAME (a Power) exist in data/id_map.json and the emulator's card table.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity.invariants import (  # noqa: E402
    Judge,
    check_game,
    game_frame,
    triage,
)

RNG = {k: {"seed": 1, "counter": 0} for k in (
    "Shuffle", "CombatTargets", "MonsterAi", "CombatCardSelection", "CombatCardGeneration",
    "CombatEnergyCosts", "CombatPotionGeneration", "CombatOrbGeneration", "Niche")}


def state(hand=(), draw=(), discard=(), exhaust=(), hp=60, energy=3, block=0, powers=(), enemies=None, round_=1) -> dict:
    card = lambda n: [n, False]  # noqa: E731
    return {
        "in_combat": True, "round": round_,
        "player": {"hp": hp, "max_hp": 80, "block": block, "energy": energy, "powers": [list(p) for p in powers]},
        "hand": [card(c) for c in hand], "draw": [card(c) for c in draw],
        "discard": [card(c) for c in discard], "exhaust": [card(c) for c in exhaust],
        "enemies": enemies if enemies is not None else [{"name": "SEAPUNK", "hp": 40, "max_hp": 47, "block": 0, "powers": []}],
        "rng": copy.deepcopy(RNG),
    }


def step(st: dict, kind: str = "end_turn", **kw: object) -> dict:
    action = {"kind": kind, **kw}
    return {"action": action, "state": st}


def play(card, index, st, played=True):
    return step(st, "play", hand_index=index, card=card, target=0, target_slot=0, played=played)


def fight(*steps: dict) -> dict:
    return {"seed": "TESTSEED01", "steps": [{"action": None, "state": steps[0]}, *steps[1:]]}


def checks(f):
    return {c for _, c, _ in check_game(JUDGE, f)}


JUDGE = Judge()
S, D, B = "STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"


class SoundFight(unittest.TestCase):
    def test_a_plain_fight_has_no_violation(self):
        a = state(hand=[S, D, B], draw=[S, D], discard=[])
        b = state(hand=[S, D], draw=[S, D], discard=[B], energy=1)
        b["enemies"][0]["hp"] = 32
        c = state(hand=[S, D], draw=[S, D], discard=[B], energy=0)
        self.assertEqual(check_game(JUDGE, fight(a, play(B, 2, b), play(D, 1, c))), [])


class FrameChecks(unittest.TestCase):
    def one(self, **kw: object) -> set:
        return checks(fight(state(**kw)))

    def test_hp(self):
        self.assertIn("hp_range", self.one(hp=81))
        self.assertIn("hp_range", self.one(hp=-1))

    def test_enemy_hp(self):
        e = [{"name": "SEAPUNK", "hp": 50, "max_hp": 47, "block": 0, "powers": []}]
        self.assertIn("hp_range", self.one(enemies=e))

    def test_energy(self):
        self.assertIn("energy", self.one(energy=-1))
        self.assertIn("energy", self.one(energy=99))

    def test_block(self):
        self.assertIn("block", self.one(block=-3))

    def test_zero_power(self):
        self.assertIn("power_amount", self.one(powers=[("VULNERABLE_POWER", 0)]))

    def test_negative_power(self):
        self.assertIn("power_amount", self.one(powers=[("VULNERABLE_POWER", -1)]))
        self.assertNotIn("power_amount", self.one(powers=[("STRENGTH_POWER", -2)]))

    def test_duplicate_power(self):
        self.assertIn("power_dup", self.one(powers=[("WEAK_POWER", 1), ("WEAK_POWER", 2)]))

    def test_instanced_power_may_repeat(self):
        self.assertNotIn("power_dup", self.one(powers=[("THE_BOMB_POWER", 3), ("THE_BOMB_POWER", 2)]))

    def test_hand_limit(self):
        self.assertIn("hand_limit", self.one(hand=[S] * 11))

    def test_unknown_card(self):
        self.assertIn("hand_limit", self.one(hand=["NOT_A_CARD"]))

    def test_no_enemies(self):
        self.assertIn("no_enemies", self.one(enemies=[]))


class TransitionChecks(unittest.TestCase):
    def two(self, a, b, action=None):
        return checks(fight(a, action or step(b)))

    def test_rng_never_falls(self):
        a, b = state(), state()
        a["rng"]["Shuffle"]["counter"] = 5
        b["rng"]["Shuffle"]["counter"] = 4
        self.assertIn("rng_monotone", self.two(a, b))

    def test_turn_moves_by_one(self):
        self.assertIn("turn_monotone", self.two(state(round_=1), state(round_=3)))
        self.assertIn("turn_monotone", self.two(state(round_=2), state(round_=1)))

    def test_dead_enemy_stays_dead(self):
        dead = [{"name": "SEAPUNK", "hp": 0, "max_hp": 47, "block": 0, "powers": []}]
        self.assertIn("dead_stay_dead", self.two(state(enemies=dead), state()))

    def test_refused_play_changes_nothing(self):
        a = state(hand=[S], energy=3)
        b = state(hand=[S], energy=2)
        self.assertIn("refused_play", checks(fight(a, play(S, 0, b, played=False))))

    def test_action_names_the_card_in_hand(self):
        a = state(hand=[S, D], energy=3)
        b = state(hand=[S], discard=[D], energy=2)
        self.assertIn("action_card", checks(fight(a, play(S, 1, b))))

    def test_hand_order_kept_by_a_plain_play(self):
        a = state(hand=[S, D, B, S], energy=3)
        ok = state(hand=[S, D, S], discard=[B], energy=1)
        bad = state(hand=[D, S, S], discard=[B], energy=1)
        self.assertNotIn("hand_order", checks(fight(a, play(B, 2, ok))))
        self.assertIn("hand_order", checks(fight(a, play(B, 2, bad))))

    def test_draw_order_kept_without_a_shuffle(self):
        a = state(hand=[B], draw=[S, D, B])
        ok = state(hand=[B, S], draw=[D, B])
        bad = state(hand=[B, S], draw=[B, D])
        self.assertNotIn("draw_order", self.two(a, ok))
        self.assertIn("draw_order", self.two(a, bad))

    def test_draw_order_may_change_with_a_shuffle(self):
        a = state(draw=[S, D, B])
        b = state(draw=[B, D, S])
        b["rng"]["Shuffle"]["counter"] = 3
        self.assertNotIn("draw_order", self.two(a, b))

    def test_exhaust_only_grows(self):
        self.assertIn("exhaust_grows", self.two(state(exhaust=[S]), state(hand=[S])))
        self.assertNotIn("exhaust_grows", checks(fight(state(exhaust=[S], hand=["EXHUME"]), play("EXHUME", 0, state(hand=[S], energy=2)))))


class Conservation(unittest.TestCase):
    def test_a_card_cannot_vanish(self):
        a = state(hand=[S, D], energy=3)
        b = state(hand=[S], energy=2)  # the Defend is nowhere
        self.assertIn("conservation", checks(fight(a, play(D, 1, b))))

    def test_a_card_cannot_be_duplicated(self):
        a = state(hand=[S, D], energy=3)
        b = state(hand=[S], discard=[D, D], energy=2)
        self.assertIn("conservation", checks(fight(a, play(D, 1, b))))

    def test_a_played_power_leaves_the_piles(self):
        a = state(hand=["INFLAME", S], energy=3)
        b = state(hand=[S], energy=2, powers=[("STRENGTH_POWER", 2)])
        self.assertNotIn("conservation", checks(fight(a, play("INFLAME", 0, b))))

    def test_a_creator_may_add_its_card(self):
        a = state(hand=["ANGER", S], energy=3)
        b = state(hand=[S], discard=["ANGER", "ANGER"], energy=3)
        self.assertNotIn("conservation", checks(fight(a, play("ANGER", 0, b))))

    def test_a_non_creator_may_not(self):
        a = state(hand=[B, S], energy=3)
        b = state(hand=[S], discard=[B, "ANGER"], energy=1)
        self.assertIn("conservation", checks(fight(a, play(B, 0, b))))

    def test_an_upgrade_keeps_the_count(self):
        a = state(hand=[B, S], energy=3)
        b = state(hand=[B, S], energy=3)
        b["hand"][1][1] = True
        self.assertNotIn("conservation", checks(fight(a, play("ARMAMENTS", 0, b))))

    def test_an_auto_played_card_in_flight_is_a_recording_issue(self):
        a = state(hand=[S], draw=[S, D], powers=[("HELLRAISER_POWER", 1)])
        b = state(hand=[S], draw=[D], powers=[("HELLRAISER_POWER", 1)])
        found = checks(fight(a, step(b)))
        self.assertIn("conservation_inflight", found)
        self.assertNotIn("conservation", found)


class Frames(unittest.TestCase):
    def test_a_fight_left_in_combat_with_no_enemy_is_not_a_frame_failure(self):
        f = game_frame(step({"in_combat": False}))
        self.assertFalse(f.in_combat)
        self.assertEqual(JUDGE.frame(f), [])


class Triage(unittest.TestCase):
    def test_a_known_design_reason_is_named(self):
        verdict, why = triage({"relation": "pad", "card": "MIND_BLAST", "detail": ".Item1.Enemies[0].Hp: 16 | 15"})
        self.assertIn("by design", verdict)
        self.assertIn("draw pile", why)

    def test_the_same_relation_on_another_card_is_not_excused(self):
        verdict, _ = triage({"relation": "pad", "card": "STRIKE_IRONCLAD", "detail": ".Item1.Enemies[0].Hp: 16 | 15"})
        self.assertEqual(verdict, "unclassified")

    def test_an_exception_is_never_excused(self):
        verdict, _ = triage({"relation": "exception", "card": "pad", "detail": "NullReferenceException"})
        self.assertEqual(verdict, "unclassified")


if __name__ == "__main__":
    unittest.main()
