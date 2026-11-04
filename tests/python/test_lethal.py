"""rl/lethal.py, the kill check at the root of the fight search: the enumerator on synthetic positions, then the emulator's rules."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(p))

import lethal  # noqa: E402


class Move(NamedTuple):
    key: tuple
    action: int
    target: int
    kind: str


# name -> (cost, damage, HP the player loses, cards drawn)
CARDS = {"strike": (1, 6, 0, 0), "strike2": (1, 6, 0, 0), "bash": (2, 8, 0, 0), "defend": (1, 0, 0, 0), "blood": (0, 8, 3, 0),
         "pommel": (1, 9, 0, 1), "recoil": (0, 20, 5, 0)}
POTIONS = {"fire": 20}


class Toy:
    """A one-turn fight: enemies' HP, the player's HP and energy, a hand, a draw pile, potions. `fork(seed)` reshuffles the draw
    pile by the seed (the resample); `fork(None)` is a faithful copy. Counts the copies not yet closed."""

    open_count = 0

    def __init__(self, enemies, hp=50, energy=3, hand=(), draw=(), potions=()):
        self.enemies, self.hp_, self.energy = list(enemies), hp, energy
        self.hand, self.draw, self.potions = list(hand), list(draw), list(potions)
        self.done = False
        self.won_ = False
        Toy.open_count += 1

    def fork(self, seed):
        c = Toy(self.enemies, self.hp_, self.energy, self.hand, self.draw, self.potions)
        c.done, c.won_ = self.done, self.won_
        if seed is not None:
            random.Random(seed).shuffle(c.draw)
        return c

    def close(self):
        Toy.open_count -= 1

    def hp(self):
        return max(0, self.hp_)

    def won(self):
        return self.won_

    def settle(self):
        if self.hp_ <= 0:
            self.done, self.won_ = True, False
        elif all(e <= 0 for e in self.enemies):
            self.done, self.won_ = True, True


class ToyRules:
    def moves(self, w):
        out, seen = [], set()
        for i, name in enumerate(w.hand):
            if name not in seen and CARDS[name][0] <= w.energy:
                seen.add(name)
                out.append(Move(("card", name), i, -1, "card"))
        out += [Move(("potion", p), len(w.hand) + 1 + j, -1, "potion") for j, p in enumerate(dict.fromkeys(w.potions))]
        out.append(Move(("end_turn",), len(w.hand), -1, "end_turn"))
        return out

    def play(self, w, m):
        if m.kind == "card":
            name = m.key[1]
            cost, dmg, self_dmg, draws = CARDS[name]
            w.hand.remove(name)
            w.energy -= cost
            w.hp_ -= self_dmg
            for _ in range(draws):
                if w.draw:
                    w.hand.append(w.draw.pop(0))
            w.enemies[0] -= dmg
        elif m.kind == "potion":
            w.potions.remove(m.key[1])
            w.enemies[0] -= POTIONS[m.key[1]]
        w.settle()

    def enemy_hp(self, w):
        return sum(max(0, e) for e in w.enemies)

    def state_key(self, w):
        return (tuple(w.enemies), w.hp_, w.energy, tuple(sorted(w.hand)), tuple(w.draw), tuple(sorted(w.potions)))

    def match(self, w, m):
        return next((x for x in self.moves(w) if x.key == m.key), None)


class ToyRulesNoKey(ToyRules):
    """The same rules without a state key: every order of the cards is explored."""

    state_key = None


RULES = ToyRules()


class Clock:
    """A fake clock that moves `tick` seconds each time it is read."""

    def __init__(self, tick):
        self.t, self.tick = 0.0, tick

    def __call__(self):
        self.t += self.tick
        return self.t


class FindLethalTests(unittest.TestCase):
    def setUp(self):
        Toy.open_count = 0

    def tearDown(self):
        self.assertEqual(0, Toy.open_count, "a copy was left open: the emulator's handle pool would fill")

    def look(self, toy, rules=None, **limits):
        try:
            return lethal.find_lethal(toy, rules or RULES, lethal.Limits(**limits))
        finally:
            toy.close()

    def test_a_kill_is_found_and_its_first_move_named(self):
        rep = self.look(Toy([12], energy=2, hand=["defend", "strike", "strike"]))
        self.assertEqual("kill", rep.reason)
        self.assertEqual(["strike", "strike"], [m.key[1] for m in rep.kill.path])
        self.assertEqual((0, 0.0), (rep.kill.potions, rep.kill.hp_lost))
        self.assertEqual(lethal.VERIFY_WORLDS, rep.kill.verified)

    def test_the_kill_may_need_the_card_that_costs_most(self):
        rep = self.look(Toy([14], energy=3, hand=["strike", "defend", "bash", "strike2"]))
        self.assertIn("bash", [m.key[1] for m in rep.kill.path])
        self.assertEqual(14, sum(CARDS[m.key[1]][1] for m in rep.kill.path))

    def test_no_kill_is_none_and_says_the_whole_tree_was_tried(self):
        rep = self.look(Toy([40], energy=3, hand=["strike", "strike2", "bash", "defend"]))
        self.assertIsNone(rep.kill)
        self.assertEqual("none", rep.reason)
        self.assertGreater(rep.sequences, 0)
        self.assertEqual(rep.record()["fired"], False)

    def test_end_turn_is_never_part_of_a_sequence(self):
        rep = self.look(Toy([12], energy=2, hand=["strike", "strike2"]))
        self.assertTrue(all(m.kind != "end_turn" for m in rep.kill.path))
        none = self.look(Toy([13], energy=2, hand=["strike", "strike2"]))  # one HP short: ending the turn is not a way to win here
        self.assertIsNone(none.kill)

    def test_a_death_is_not_a_kill(self):
        # the card kills the enemy and the player together: the fight is lost, so it is no kill
        rep = self.look(Toy([20], hp=5, energy=1, hand=["recoil"]))
        self.assertIsNone(rep.kill)
        self.assertEqual("none", rep.reason)
        alive = self.look(Toy([20], hp=6, energy=1, hand=["recoil"]))
        self.assertEqual((1, 5.0), (len(alive.kill.path), alive.kill.hp_lost))

    def test_the_kill_that_costs_the_least_hp_is_chosen(self):
        # blood kills alone for 3 HP; two strikes kill for none, and are found after it
        rep = self.look(Toy([8], energy=2, hand=["blood", "strike", "strike2"]))
        self.assertEqual("kill", rep.reason)
        self.assertEqual(0.0, rep.kill.hp_lost)
        self.assertEqual(["strike", "strike2"], [m.key[1] for m in rep.kill.path])
        self.assertGreaterEqual(rep.kills_seen, 2)

    def test_the_only_kill_may_cost_hp(self):
        rep = self.look(Toy([8], energy=0, hand=["blood", "strike"]))
        self.assertEqual((["blood"], 3.0), ([m.key[1] for m in rep.kill.path], rep.kill.hp_lost))

    def test_a_kill_with_no_potion_beats_a_shorter_one_with_a_potion(self):
        toy = Toy([20], energy=4, hand=["bash", "strike", "strike2"], potions=["fire"])
        rep = self.look(toy)
        self.assertEqual(0, rep.kill.potions)
        self.assertEqual(3, len(rep.kill.path))
        self.assertTrue(all(m.kind == "card" for m in rep.kill.path))

    def test_a_potion_is_used_when_it_is_the_only_kill(self):
        rep = self.look(Toy([26], energy=3, hand=["bash", "strike"], potions=["fire"]))
        self.assertEqual(1, rep.kill.potions)
        self.assertIn("fire", [m.key[1] for m in rep.kill.path])

    def test_the_sequence_cap_stops_the_enumeration_and_the_report_says_so(self):
        names = [f"c{i}" for i in range(7)]
        for n in names:
            CARDS[n] = (0, 1, 0, 0)
        self.addCleanup(lambda: [CARDS.pop(n) for n in names])
        rep = self.look(Toy([1000], energy=3, hand=list(names)), ToyRulesNoKey(), max_steps=10**6)  # 7! orderings of free cards: far more than the cap
        self.assertEqual(("cap", lethal.MAX_SEQUENCES), (rep.reason, rep.sequences))
        self.assertIsNone(rep.kill)
        small = self.look(Toy([1000], energy=3, hand=list(names)), ToyRulesNoKey(), max_sequences=5, max_steps=10**6)
        self.assertEqual(("cap", 5), (small.reason, small.sequences))

    def test_the_step_cap_stops_it_too(self):
        names = [f"s{i}" for i in range(7)]
        for n in names:
            CARDS[n] = (0, 1, 0, 0)
        self.addCleanup(lambda: [CARDS.pop(n) for n in names])
        rep = self.look(Toy([1000], energy=3, hand=list(names)), ToyRulesNoKey(), max_steps=30)
        self.assertEqual("steps", rep.reason)
        self.assertLessEqual(rep.steps, 30 + len(names))

    def free_cards(self, n, prefix):
        names = [f"{prefix}{i}" for i in range(n)]
        for name in names:
            CARDS[name] = (0, 1, 0, 0)
        self.addCleanup(lambda: [CARDS.pop(name) for name in names])
        return names

    def test_the_same_cards_in_another_order_are_explored_once(self):
        hand = self.free_cards(5, "p")
        big = lethal.Limits(max_sequences=10**6, max_steps=10**6)
        with_key = self.look(Toy([1000], energy=3, hand=list(hand)), **vars(big))
        no_key = self.look(Toy([1000], energy=3, hand=list(hand)), rules=ToyRulesNoKey(), **vars(big))
        self.assertEqual(("none", "none"), (with_key.reason, no_key.reason))
        self.assertGreater(with_key.pruned, 0)
        self.assertEqual(0, no_key.pruned)
        self.assertLess(with_key.steps, no_key.steps // 3)  # 5! orders against 2^5 sets of cards
        self.assertLess(with_key.sequences, no_key.sequences)

    def test_pruning_does_not_lose_a_kill(self):
        hand = self.free_cards(5, "q")
        for rules in (RULES, ToyRulesNoKey()):
            toy = Toy([4], energy=3, hand=[*hand, "strike"])
            rep = lethal.find_lethal(toy, rules)
            toy.close()
            self.assertEqual("kill", rep.reason)
            self.assertGreaterEqual(sum(CARDS[m.key[1]][1] for m in rep.kill.path), 4)

    def test_the_open_copy_cap_stops_it_and_closes_every_copy(self):
        hand = self.free_cards(6, "o")
        rep = self.look(Toy([1000], energy=3, hand=list(hand)), max_open=4, max_steps=10**6)
        self.assertEqual("open", rep.reason)
        self.assertIsNone(rep.kill)

    def test_a_failure_in_the_rules_leaves_no_copy_open(self):
        hand = self.free_cards(5, "f")

        class Failing(ToyRules):
            plays = 0

            def play(self, w, m):
                Failing.plays += 1
                if Failing.plays == 9:
                    raise RuntimeError("combat handle pool is full")
                super().play(w, m)

        toy = Toy([1000], energy=3, hand=list(hand))
        try:
            with self.assertRaises(RuntimeError):
                lethal.find_lethal(toy, Failing())
        finally:
            toy.close()
        # tearDown: no copy left open

    def test_the_time_cap_stops_it_and_the_clock_is_the_callers(self):
        names = [f"t{i}" for i in range(7)]
        for n in names:
            CARDS[n] = (0, 1, 0, 0)
        self.addCleanup(lambda: [CARDS.pop(n) for n in names])
        toy = Toy([1000], energy=3, hand=list(names))
        try:
            rep = lethal.find_lethal(toy, ToyRulesNoKey(), lethal.Limits(max_seconds=0.05), clock=Clock(0.01))
        finally:
            toy.close()
        self.assertEqual("time", rep.reason)
        self.assertLess(rep.sequences, lethal.MAX_SEQUENCES)

    def test_a_kill_that_needs_a_draw_the_player_cannot_see_is_not_trusted(self):
        # pommel strikes for 9 and draws; 15 HP is a kill only if the drawn card is the strike
        def top(seed):
            pile = ["strike", "defend"]
            random.Random(seed).shuffle(pile)
            return pile[0]

        seed = next(s for s in range(1, 200) if top(s) == "strike" and any(top(s + i) == "defend" for i in range(1, 4)))
        toy = Toy([15], energy=2, hand=["pommel"], draw=["strike", "defend"])
        rep = self.look(toy, seed=seed)
        self.assertIsNone(rep.kill)
        self.assertEqual("unverified", rep.reason)
        self.assertGreaterEqual(rep.rejected, 1)
        self.assertEqual(rep.record()["fired"], False)

    def test_a_kill_that_holds_whatever_is_drawn_is_trusted(self):
        # the same shape, but every card in the draw pile finishes the enemy
        rep = self.look(Toy([15], energy=2, hand=["pommel"], draw=["strike", "strike"]))
        self.assertEqual("kill", rep.reason)
        self.assertEqual(2, len(rep.kill.path))

    def test_the_record_is_small_and_names_the_line(self):
        rep = self.look(Toy([12], energy=2, hand=["strike", "strike2"]))
        rec = rep.record()
        self.assertTrue(rec["fired"])
        self.assertEqual("card", rec["line"][0][2])
        self.assertEqual(2, rec["plays"])
        for key in ("reason", "sequences", "steps", "kills_seen", "rejected", "seconds", "hp_lost", "potions", "verified"):
            self.assertIn(key, rec)

    def test_the_position_is_only_forked_never_stepped(self):
        toy = Toy([12], energy=2, hand=["strike", "strike2"])
        before = (list(toy.enemies), toy.hp_, toy.energy, list(toy.hand), toy.done)
        lethal.find_lethal(toy, RULES)
        self.assertEqual(before, (toy.enemies, toy.hp_, toy.energy, toy.hand, toy.done))
        toy.close()

    def test_the_answer_is_a_function_of_the_position_not_of_the_run(self):
        a = self.look(Toy([14], energy=3, hand=["strike", "bash", "defend", "strike2"]))
        b = self.look(Toy([14], energy=3, hand=["strike", "bash", "defend", "strike2"]))
        self.assertEqual([m.key for m in a.kill.path], [m.key for m in b.kill.path])
        self.assertEqual((a.sequences, a.steps), (b.sequences, b.steps))


# ---------------------------------------------------------------------------------------------------------- the emulator's rules

try:
    import ctypes

    import combat_mcts
    import live_search
    import search_configs

    from sts2_gym import native
    from sts2_gym.env import ENCOUNTER_IDS

    NATIVE = True
except (ImportError, OSError, RuntimeError):  # no built library: the synthetic tests above still run
    NATIVE = False

BLUDGEON, STRIKE, DEFEND, BASH = 47, 472, 131, 30  # 32 damage for 3 energy; the starter's cards


def arena(seed: int, deck: list[int], encounter: str = "nibbits-weak"):
    """A real encounter on its own handle (the player at 64 of 80 HP), as the live-search tests build one."""
    handle = native.create(seed)
    obs = (ctypes.c_int * native.OBS_SIZE)()
    native.reset_arena(handle, deck, obs, encounter_id=ENCOUNTER_IDS[encounter], relic_ids=[], potion_ids=[], player_hp=64, player_max_hp=80)
    return combat_mcts.Combat(handle)


QUICK = {"seed": 5, "max_sims": 24, "min_child_sims": 1, "confirm_pairs": 0, "time_budget_s": 120.0}


def drive(*args, **kw):
    return live_search._drive(live_search.search_steps(*args, **kw), None)  # noqa: SLF001


@unittest.skipUnless(NATIVE, "no emulator library")
class LiveRulesTests(unittest.TestCase):
    def two_bludgeon_turns(self):
        """Ten Bludgeons against one Nibbit (44 HP): one is played and the turn ended, so the next hand holds a kill."""
        fight = arena(1, [BLUDGEON] * 10)
        self.addCleanup(fight.close)
        fight.step(0)
        fight.step(max(fight.legal()))  # End Turn is the last action
        self.assertFalse(fight.done)
        return fight

    def test_a_real_kill_is_found_verified_and_true_on_the_real_position(self):
        fight = self.two_bludgeon_turns()
        rep = lethal.find_lethal(fight, live_search._LIVE_RULES)  # noqa: SLF001
        self.assertEqual("kill", rep.reason)
        self.assertEqual(1, len(rep.kill.path))
        true_world = fight.fork(None)  # the oracle copy: the real draw order and rolls
        try:
            live_search._step(true_world, rep.kill.path[0].action, rep.kill.path[0].target)  # noqa: SLF001
            self.assertTrue(true_world.done and true_world.won())
        finally:
            true_world.close()

    def test_the_position_the_check_was_asked_about_is_left_as_it_was(self):
        fight = self.two_bludgeon_turns()
        legal, enemy = fight.legal(), list(fight.obs[live_search._ENEMIES:live_search._ENEMIES + live_search._ENEMY])  # noqa: SLF001
        lethal.find_lethal(fight, live_search._LIVE_RULES)  # noqa: SLF001
        self.assertEqual(legal, fight.legal())
        self.assertEqual(enemy, list(fight.obs[live_search._ENEMIES:live_search._ENEMIES + live_search._ENEMY]))  # noqa: SLF001
        self.assertFalse(fight.done)

    def test_search_live_answers_a_kill_with_no_simulations_and_says_so(self):
        fight = self.two_bludgeon_turns()
        res = live_search.search_live(fight.handle, lethal_check=True, **QUICK)
        self.assertEqual(0, res.sims)
        self.assertTrue(res.kill_check["fired"])
        self.assertEqual("kill", res.kill_check["reason"])
        self.assertEqual((res.best.action, res.best.target), tuple(res.kill_check["line"][0][:2]))
        self.assertEqual([(res.best.action, res.best.target, res.best.kind)], res.line)
        self.assertNotIn("true_world_kill", res.kill_check)  # the oracle replay is the shadow measurement's, not the check's
        plain = live_search.search_live(fight.handle, **QUICK)
        self.assertIsNone(plain.kill_check)  # off: no record, no look

    def test_a_fired_check_in_shadow_mode_records_the_searchs_own_move(self):
        fight = self.two_bludgeon_turns()
        from unittest import mock

        with mock.patch.dict("os.environ", {live_search.LETHAL_SHADOW_ENV: "1"}):
            res = live_search.search_live(fight.handle, lethal_check=True, **QUICK)
        self.assertTrue(res.kill_check["fired"])
        self.assertEqual(0, res.sims)  # the answer is still the kill's
        self.assertEqual(QUICK["max_sims"], res.kill_check["search_sims"])
        self.assertFalse(res.kill_check["differs"])  # a Bludgeon is the only sensible play, and the search plays it
        self.assertIs(True, res.kill_check["true_world_kill"])  # and the kill holds on the position's real draw order and rolls
        self.assertEqual([res.best.action, res.best.target, res.best.kind], res.kill_check["search_move"])

    def test_with_no_kill_the_search_is_what_it_was(self):
        fight = arena(2, [STRIKE] * 5 + [DEFEND] * 4 + [BASH], "toadpoles")
        self.addCleanup(fight.close)
        plain = live_search.search_live(fight.handle, **QUICK)
        checked = live_search.search_live(fight.handle, lethal_check=True, **QUICK)
        self.assertFalse(checked.kill_check["fired"])
        self.assertEqual("none", checked.kill_check["reason"])
        self.assertEqual(plain.best, checked.best)
        self.assertEqual(plain.sims, checked.sims)
        self.assertEqual([c.visits for c in plain.children], [c.visits for c in checked.children])
        self.assertEqual(plain.line, checked.line)

    def test_search_steps_does_the_same(self):
        fight = self.two_bludgeon_turns()
        res = drive(fight.handle, lethal_check=True, **QUICK)
        self.assertEqual((0, True), (res.sims, res.kill_check["fired"]))
        other = arena(2, [STRIKE] * 5 + [DEFEND] * 4 + [BASH], "toadpoles")
        self.addCleanup(other.close)
        self.assertEqual(live_search.search_live(other.handle, **QUICK).best, drive(other.handle, lethal_check=True, **QUICK).best)

    def test_one_legal_move_is_not_looked_at(self):
        fight = arena(3, [BLUDGEON])
        self.addCleanup(fight.close)
        fight.step(0)  # the only card is played: only End Turn is left
        res = live_search.search_live(fight.handle, lethal_check=True, **QUICK)
        self.assertIsNone(res.kill_check)

    def test_every_kill_the_check_claims_is_true_on_the_real_fight(self):
        """Fights played forward, the check asked at every decision: each kill it names ends the real fight with the player alive."""
        claims = 0
        types = live_search._card_types()  # noqa: SLF001
        hand_at, slot = live_search._HAND, live_search._CARD  # noqa: SLF001
        for seed in range(8):
            fight = arena(seed, [STRIKE] * 5 + [BASH] * 2 + [DEFEND] * 3)
            self.addCleanup(fight.close)
            for _ in range(60):
                if fight.done:
                    break
                rep = lethal.find_lethal(fight, live_search._LIVE_RULES)  # noqa: SLF001
                if rep.kill is not None:
                    claims += 1
                    true_world = fight.fork(None)
                    try:
                        for move in rep.kill.path:
                            live_search._step(true_world, move.action, move.target)  # noqa: SLF001
                        self.assertTrue(true_world.done and true_world.won(), (seed, rep.record()))
                    finally:
                        true_world.close()
                    first = rep.kill.path[0]
                    live_search._step(fight, first.action, first.target)  # noqa: SLF001
                    continue
                hand = live_search._hand_size(fight.obs)  # noqa: SLF001
                attacks = [a for a in fight.legal() if a < hand and types.get(fight.obs[hand_at + a * slot]) == "Attack"]
                fight.step(attacks[0] if attacks else hand)
        self.assertGreater(claims, 0, "no fight reached a kill: the test saw nothing")


class ConfigTests(unittest.TestCase):
    def test_a3_v1_k_is_a3_v1_with_the_check_on_and_nothing_else_changed(self):
        import dataclasses

        import search_configs as sc

        base, k = sc.get("a3-v1"), sc.get("a3-v1-k")
        self.assertTrue(k.lethal_check)
        self.assertFalse(base.lethal_check)
        self.assertEqual(dataclasses.replace(base, name="a3-v1-k", lethal_check=True), k)

    def test_only_the_k_config_passes_the_option_on(self):
        import search_configs as sc

        pos = sc.Position("play", 4, 0.9, False)
        self.assertTrue(sc.get("a3-v1-k").search_kwargs(pos)["lethal_check"])
        for name in ("a3-v1", "gen0-60", "judge-v0", "flat-30"):
            self.assertNotIn("lethal_check", sc.get(name).search_kwargs(pos), name)  # every other config's arguments are what they were
        rest = {k: v for k, v in sc.get("a3-v1-k").search_kwargs(pos).items() if k != "lethal_check"}
        self.assertEqual(rest, sc.get("a3-v1").search_kwargs(pos))

    @unittest.skipUnless(NATIVE, "no emulator library")
    def test_the_option_reaches_search_live_and_the_pool_does_not_take_it(self):
        import inspect

        self.assertIn("lethal_check", inspect.signature(live_search.search_live).parameters)
        self.assertIn("lethal_check", inspect.signature(live_search.search_steps).parameters)
        self.assertNotIn("lethal_check", live_search._SEARCH_SETTINGS)  # noqa: SLF001 -- ParallelSearcher's searches are unchanged


if __name__ == "__main__":
    unittest.main()
