"""The search's Python tidy: every helper `rl/live_search.py` and `rl/combat_mcts.py` now computes another way gives the old form's result.

The old forms are kept below, as they were before the tidy (a per-index read of the observation, numpy for the rollout's weights and
`Generator.choice` for its pick), and each new helper is held to them on fixed and random inputs: random observations laid out as the
emulator lays them out, the positions of real fights played forward at random, and whole rollouts and searches (same moves, same
random draws, same end position, same generator state). A numpy upgrade that changed `choice` or `sum` would fail here first.
"""

from __future__ import annotations

import contextlib
import ctypes
import random
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))

import combat_mcts  # noqa: E402
import live_search  # noqa: E402
from combat_tokens import MAX_POTIONS, POTION_OFFSET  # noqa: E402

from sts2_gym import names, native  # noqa: E402
from sts2_gym.env import ENCOUNTER_IDS  # noqa: E402

STRIKE, DEFEND, BASH, TRUE_GRIT, BLUDGEON, POMMEL_STRIKE, SECRET_WEAPON = 472, 131, 30, 517, 47, 358, 416
BLOCK_POTION = 5
DECK = [STRIKE] * 4 + [DEFEND] * 3 + [BASH, TRUE_GRIT, BLUDGEON, POMMEL_STRIKE, SECRET_WEAPON]
ENCOUNTERS = ("nibbits-weak", "toadpoles", "fogmog", "construct-menagerie")

# ------------------------------------------------------------------------------------------------------ the old forms


def old_hand_size(obs) -> int:
    return sum(1 for slot in range(live_search._MAX_HAND) if obs[live_search._HAND + slot * live_search._CARD])


def old_living_targets(obs) -> list[int]:
    reviving = live_search._reviving()
    targets = []
    for index in range(native.MAX_ENEMIES):
        base = live_search._ENEMIES + index * live_search._ENEMY
        if obs[base] <= 0:
            continue
        buffs = range(base + 5, base + 5 + 2 * live_search._ENEMY_BUFFS, 2)
        if any(obs[buff] == reviving and obs[buff + 1] > 0 for buff in buffs):
            continue
        targets.append(index)
    return targets


def old_legal(mask) -> list[int]:
    return [action for action in range(live_search.MAX_ACTIONS) if mask[action]]


def old_legal_moves(obs, mask) -> list:
    legal = old_legal(mask)
    moves = []
    if obs[live_search._SELECTION_KIND]:
        count = obs[live_search._SELECTION_COUNT]
        for action in legal:
            if action >= count:
                moves.append(live_search._Move(("skip",), action, -1, "skip", None))
            elif action < live_search._MAX_CANDIDATES:
                card = live_search._card_at(obs, live_search._SELECTION + action * live_search._CARD)
                moves.append(live_search._Move(("select", card), action, -1, "select", card[0]))
            else:
                moves.append(live_search._Move(("select-index", action), action, -1, "select", None))
        return moves

    hand = old_hand_size(obs)
    targets = old_living_targets(obs)
    aimed_cards, aimed_potions = live_search._aimed_cards(), live_search._aimed_potions()
    for action in legal:
        if action < hand:
            card = live_search._card_at(obs, live_search._HAND + action * live_search._CARD)
            aims = targets if card[0] in aimed_cards and targets else [-1]
            moves.extend(live_search._Move(("card", card, t), action, t, "card", card[0]) for t in aims)
        elif action == hand:
            moves.append(live_search._Move(("end_turn",), action, -1, "end_turn", None))
        else:
            potion = obs[POTION_OFFSET + 2 * (action - hand - 1)]
            aims = targets if potion in aimed_potions and targets else [-1]
            moves.extend(live_search._Move(("potion", potion, t), action, t, "potion", None) for t in aims)
    return moves


def old_tempo_weight(card_type: str, *, needed: bool) -> float:
    if card_type == "Attack":
        return combat_mcts.W_ATTACK
    if card_type == "Power":
        return combat_mcts.W_POWER
    if card_type == "Skill":
        return combat_mcts.W_SKILL_NEEDED if needed else combat_mcts.W_SKILL_IDLE
    return combat_mcts.W_OTHER


def old_rollout_weights(obs, legal: list[int]) -> np.ndarray:
    weights = np.ones(len(legal), dtype=np.float64)
    if not obs[live_search._SELECTION_KIND]:
        hand = old_hand_size(obs)
        types = live_search._card_types()
        needed = combat_mcts.incoming_damage(obs) > obs[2]
        for i, action in enumerate(legal):
            if action < hand:
                card_type = types.get(obs[live_search._HAND + action * live_search._CARD], "")
                weights[i] = old_tempo_weight(card_type, needed=needed)
            else:
                weights[i] = combat_mcts.W_OTHER
    return weights / weights.sum()


def old_rollout_target(obs, action: int) -> int:
    if obs[live_search._SELECTION_KIND]:
        return -1
    hand = old_hand_size(obs)
    if action < hand:
        aimed = obs[live_search._HAND + action * live_search._CARD] in live_search._aimed_cards()
    elif action > hand:
        aimed = obs[POTION_OFFSET + 2 * (action - hand - 1)] in live_search._aimed_potions()
    else:
        return -1
    targets = old_living_targets(obs) if aimed else []
    if not targets:
        return -1

    def health(index):
        base = live_search._ENEMIES + index * live_search._ENEMY
        return obs[base] + obs[base + 2]

    return min(targets, key=lambda i: (health(i), i))


def old_rollout(world, rng) -> None:
    for _ in range(live_search.STEP_CAP):
        if world.done:
            return
        legal = old_legal(native.valid_actions(world.handle, live_search.MAX_ACTIONS))
        if not legal:
            return
        if len(legal) > 1:
            action = int(rng.choice(legal, p=old_rollout_weights(world.obs, legal)))
        else:
            action = int(rng.choice(legal))
        live_search._step(world, action, old_rollout_target(world.obs, action))


def old_heuristic_prior(world) -> dict:
    mask = native.valid_actions(world.handle, live_search.MAX_ACTIONS)
    legal = old_legal(mask)
    raw = np.asarray(old_rollout_weights(world.obs, legal), dtype=np.float64)
    weights = dict(zip(legal, (raw / raw.sum()).tolist()))
    moves = old_legal_moves(world.obs, mask)
    aims = Counter(move.action for move in moves)
    prior: dict = {}
    for move in moves:
        share = weights.get(move.action, 0.0) / aims[move.action]
        prior[move.key] = prior.get(move.key, 0.0) + share
    return prior


def old_held(obs) -> int:
    return sum(1 for slot in range(MAX_POTIONS) if obs[POTION_OFFSET + 2 * slot])


# ------------------------------------------------------------------------------------------------------ inputs


def random_weights(rnd: random.Random) -> list[float]:
    """Weights as the rollout makes them (the tempo constants) or arbitrary positive floats, 1 to 32 of them."""
    n = rnd.randint(1, 32)
    if rnd.random() < 0.7:
        pool = [combat_mcts.W_ATTACK, combat_mcts.W_POWER, combat_mcts.W_SKILL_NEEDED, combat_mcts.W_SKILL_IDLE, combat_mcts.W_OTHER]
        return [rnd.choice(pool) for _ in range(n)]
    return [rnd.uniform(0.01, 9.0) for _ in range(n)]


def random_obs(rnd: random.Random):
    """An observation laid out as the emulator lays it out, filled with plausible numbers; a ctypes array, as the search reads."""
    obs = (ctypes.c_int * native.OBS_SIZE)()
    cards = list(names.cards())
    reviving = live_search._reviving()
    obs[0], obs[1], obs[2], obs[3] = rnd.randint(1, 80), 80, rnd.randint(0, 30), rnd.randint(0, 4)
    holes = rnd.random() < 0.3
    for slot in range(rnd.randint(0, native.OBS_MAX_HAND)):
        base = native.OBS_HAND_OFFSET + slot * native.OBS_CARD_SLOT_SIZE
        if holes and rnd.random() < 0.3:
            continue  # an empty slot among full ones: the count is of non-empty slots
        obs[base] = rnd.choice(cards)
        obs[base + 1], obs[base + 2], obs[base + 3] = rnd.randint(0, 1), rnd.randint(0, 3), rnd.randint(0, 5)
    for index in range(native.MAX_ENEMIES):
        base = native.OBS_ENEMY_OFFSET + index * native.OBS_ENEMY_SLOT_SIZE
        obs[base] = rnd.choice([0, -1, rnd.randint(1, 120)])
        obs[base + 1] = rnd.randint(1, 120)
        obs[base + 2] = rnd.randint(0, 40)
        obs[base + 3] = rnd.choice([0, 1, 2, 3])
        obs[base + 4] = rnd.randint(0, 30)
        for buff in range(native.OBS_MAX_ENEMY_BUFFS):
            if rnd.random() < 0.3:
                obs[base + 5 + 2 * buff] = reviving if rnd.random() < 0.25 else rnd.randint(1, 60)
                obs[base + 6 + 2 * buff] = rnd.randint(-1, 5)
    for slot in range(MAX_POTIONS):
        obs[POTION_OFFSET + 2 * slot] = rnd.choice([0, 0, BLOCK_POTION, rnd.randint(1, 40)])
    if rnd.random() < 0.25:
        obs[live_search._SELECTION_KIND] = rnd.randint(1, 3)
        obs[live_search._SELECTION_COUNT] = rnd.randint(0, 14)
        for i in range(native.OBS_MAX_SELECTION_CANDIDATES):
            base = live_search._SELECTION + i * native.OBS_CARD_SLOT_SIZE
            obs[base] = rnd.choice(cards)
    return obs


def random_mask(rnd: random.Random):
    mask = (ctypes.c_int * live_search.MAX_ACTIONS)()
    for action in range(live_search.MAX_ACTIONS):
        mask[action] = 1 if rnd.random() < rnd.choice([0.1, 0.4, 0.9]) else 0
    return mask


class Fight:
    """A real encounter on its own handle, played forward at random to reach positions."""

    def __init__(self, seed: int, encounter: str, deck=DECK, potions=(BLOCK_POTION,)):
        self.handle = native.create(seed)
        self.obs = (ctypes.c_int * native.OBS_SIZE)()
        self.reward = (ctypes.c_float * 1)()
        native.reset_arena(
            self.handle, deck, self.obs, encounter_id=ENCOUNTER_IDS[encounter], relic_ids=[], potion_ids=list(potions),
            player_hp=80, player_max_hp=80,
        )
        self.done = False

    def close(self) -> None:
        native.destroy(self.handle)

    def positions(self, rnd: random.Random, steps: int):
        """Yield (obs, mask) of the position now, then step a random legal action; up to `steps` positions."""
        for _ in range(steps):
            mask = native.valid_actions(self.handle, live_search.MAX_ACTIONS)
            legal = old_legal(mask)
            if self.done or not legal:
                return
            yield self.obs, mask
            self.done = native.step(self.handle, rnd.choice(legal), self.obs, self.reward, -1)


# ------------------------------------------------------------------------------------------------------ the tests


class NumpyArithmeticTests(unittest.TestCase):
    def test_numpy_sum_is_numpys_sum_to_the_last_bit(self):
        rnd = random.Random(1)
        for _ in range(60000):
            weights = random_weights(rnd)
            self.assertEqual(live_search._numpy_sum(weights), float(np.array(weights, dtype=np.float64).sum()), weights)

    def test_numpy_sum_of_every_length(self):
        rnd = random.Random(2)
        for n in range(1, 140):
            weights = [rnd.uniform(0.1, 5.0) for _ in range(n)]
            self.assertEqual(live_search._numpy_sum(weights), float(np.array(weights, dtype=np.float64).sum()), n)

    def test_normalized_is_numpys_division_by_the_sum(self):
        rnd = random.Random(3)
        for _ in range(5000):
            weights = random_weights(rnd)
            array = np.array(weights, dtype=np.float64)
            self.assertEqual(live_search._normalized(weights), (array / array.sum()).tolist())

    def test_draw_index_is_choices_pick_and_leaves_the_generator_where_choice_does(self):
        rnd = random.Random(4)
        for trial in range(20000):
            weights = random_weights(rnd)
            n = len(weights)
            if n == 1:
                continue  # a lone move is not drawn for (see below)
            seed = rnd.randrange(2**31)
            legal = rnd.sample(range(live_search.MAX_ACTIONS), n)
            old_rng, new_rng = np.random.default_rng(seed), np.random.default_rng(seed)
            array = np.array(weights, dtype=np.float64)
            old = int(old_rng.choice(legal, p=array / array.sum()))
            new = legal[live_search._draw_index(weights, new_rng)]
            self.assertEqual(old, new, (trial, weights))
            self.assertEqual(old_rng.bit_generator.state, new_rng.bit_generator.state, (trial, weights))

    def test_choice_of_one_draws_nothing(self):
        rng = np.random.default_rng(5)
        before = rng.bit_generator.state
        self.assertEqual(int(rng.choice([7])), 7)
        self.assertEqual(rng.bit_generator.state, before)


class ObservationReadTests(unittest.TestCase):
    def test_reads_on_random_observations_match_the_old_per_index_reads(self):
        rnd = random.Random(6)
        for trial in range(4000):
            obs = random_obs(rnd)
            for form in (obs, list(obs)):  # the search reads a ctypes buffer; the tests hand over lists
                self.assertEqual(live_search._hand_size(form), old_hand_size(form), trial)
                self.assertEqual(live_search._living_targets(form), old_living_targets(form), trial)
                for action in range(0, 16):
                    self.assertEqual(live_search._rollout_target(form, action), old_rollout_target(form, action), (trial, action))
            mask = random_mask(rnd)
            self.assertEqual(live_search._legal_actions(mask), old_legal(mask))
            self.assertEqual(live_search._legal_moves(obs, mask), old_legal_moves(obs, mask), trial)
            self.assertEqual(live_search._moves_of(obs, old_legal(mask)), old_legal_moves(obs, mask), trial)
            legal = old_legal(mask)
            np.testing.assert_array_equal(live_search.live_rollout_weights(mock.Mock(obs=obs), legal), old_rollout_weights(obs, legal))

    def test_a_reviving_enemy_with_hp_is_turned_away_in_a_snapshot_that_carries_both(self):
        rnd = random.Random(7)
        reviving = live_search._reviving()
        for _ in range(300):
            obs = random_obs(rnd)
            index = rnd.randrange(native.MAX_ENEMIES)
            base = native.OBS_ENEMY_OFFSET + index * native.OBS_ENEMY_SLOT_SIZE
            obs[base] = rnd.randint(1, 50)
            slot = base + 5 + 2 * rnd.randrange(native.OBS_MAX_ENEMY_BUFFS)
            obs[slot], obs[slot + 1] = reviving, rnd.randint(0, 3)
            self.assertEqual(live_search._living_targets(obs), old_living_targets(obs))
            self.assertEqual(index in live_search._living_targets(obs), obs[slot + 1] <= 0 and not any(
                obs[b] == reviving and obs[b + 1] > 0 for b in range(base + 5, base + 5 + 2 * native.OBS_MAX_ENEMY_BUFFS, 2) if b != slot
            ))

    def test_potions_held_are_counted_as_before(self):
        rnd = random.Random(8)
        for _ in range(500):
            obs = random_obs(rnd)
            world = mock.Mock(obs=obs, done=False)
            world.won.return_value = True
            world.hp.return_value = 30
            self.assertEqual(live_search._outcome(world, 10.0).value, 30 + 10.0 * old_held(obs))

    def test_combat_mcts_legal_is_the_old_list(self):
        for seed in range(3):
            fight = Fight(seed, "toadpoles")
            try:
                world = combat_mcts.Combat(fight.handle)
                for _obs, mask in fight.positions(random.Random(seed), 12):
                    self.assertEqual(world.legal(), old_legal(mask))
            finally:
                fight.close()


class RealPositionTests(unittest.TestCase):
    def test_every_read_on_the_positions_of_fights_played_at_random(self):
        seen_selection = seen_potion_move = 0
        for seed, encounter in enumerate(ENCOUNTERS * 3):
            fight = Fight(seed, encounter)
            try:
                for obs, mask in fight.positions(random.Random(seed), 60):
                    legal = old_legal(mask)
                    self.assertEqual(live_search._legal_moves(obs, mask), old_legal_moves(obs, mask))
                    np.testing.assert_array_equal(
                        live_search.live_rollout_weights(mock.Mock(obs=obs), legal), old_rollout_weights(obs, legal)
                    )
                    for action in legal:
                        self.assertEqual(live_search._rollout_target(obs, action), old_rollout_target(obs, action))
                    seen_selection += bool(obs[live_search._SELECTION_KIND])
                    seen_potion_move += any(m.kind == "potion" for m in old_legal_moves(obs, mask))
            finally:
                fight.close()
        self.assertGreater(seen_potion_move, 0)  # the positions include a potion to play
        self.assertGreater(seen_selection, 0)  # and an open selection (True Grit, Armaments)

    def test_the_heuristic_prior_is_the_old_prior_to_the_last_bit(self):
        for seed, encounter in enumerate(ENCOUNTERS * 2):
            fight = Fight(seed, encounter)
            try:
                for _obs, _mask in fight.positions(random.Random(seed), 40):
                    world = combat_mcts.Combat(fight.handle).fork(None)  # a faithful copy: its observation is filled in
                    try:
                        expected = old_heuristic_prior(world)
                        self.assertEqual(live_search._heuristic_prior(world), expected)
                        legal = old_legal(native.valid_actions(world.handle, live_search.MAX_ACTIONS))
                        moves = live_search._moves_of(world.obs, legal)
                        self.assertEqual(live_search._heuristic_prior(world, legal, moves), expected)
                    finally:
                        world.close()
            finally:
                fight.close()

    def test_a_rollout_plays_the_same_game_with_the_same_draws(self):
        for seed, encounter in enumerate(ENCOUNTERS * 4):
            fight = Fight(seed, encounter)
            try:
                root = combat_mcts.Combat(fight.handle)
                for resample in (seed + 11, seed + 12):
                    old_world, new_world = root.fork(resample), root.fork(resample)
                    old_rng, new_rng = np.random.default_rng(seed), np.random.default_rng(seed)
                    try:
                        old_rollout(old_world, old_rng)
                        live_search._rollout(new_world, new_rng)
                        self.assertEqual(bytes(old_world.obs), bytes(new_world.obs), (seed, encounter))
                        self.assertEqual(old_world.done, new_world.done)
                        self.assertEqual(old_rng.bit_generator.state, new_rng.bit_generator.state)
                        self.assertEqual(native.get_state_json(old_world.handle), native.get_state_json(new_world.handle))
                    finally:
                        old_world.close()
                        new_world.close()
            finally:
                fight.close()

    def test_a_search_with_and_without_a_prior_makes_the_old_choices(self):
        # The tidy changes how the tree and the rollout read the position, not what the search does: the old rollout and prior put back
        # in place of the new give the same statistics for the same seed.
        def uniform(_handle, legal):
            return np.ones(len(legal))

        for seed, encounter, prior in ((0, "toadpoles", None), (1, "nibbits-weak", uniform), (2, "fogmog", uniform), (3, "toadpoles", uniform)):
            results = []
            for old in (False, True):
                fight = Fight(seed, encounter)
                try:
                    with contextlib.ExitStack() as stack:
                        if old:
                            stack.enter_context(mock.patch.object(live_search, "_rollout", old_rollout))
                            stack.enter_context(mock.patch.object(live_search, "_heuristic_prior", lambda world, *_: old_heuristic_prior(world)))
                        results.append(live_search.search_live(
                            fight.handle, time_budget_s=600.0, min_child_sims=2, max_sims=40, confirm_pairs=3, seed=seed,
                            prior_fn=prior, prior_min_visits=3,
                        ))
                finally:
                    fight.close()
            new_result, old_result = results
            self.assertEqual(new_result.children, old_result.children)
            self.assertEqual(new_result.best, old_result.best)
            self.assertEqual(new_result.line, old_result.line)
            self.assertEqual(new_result.sims, old_result.sims)


if __name__ == "__main__":
    unittest.main()
