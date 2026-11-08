# The search's Python tidy (D55 addendum 9, item 2; 2026-11-09)

Lab code only: `rl/live_search.py` (the search) and `Combat.legal` in `rl/combat_mcts.py`. The search computes what it computed before and
plays the same moves with the same random draws; only how it reads the position and draws its pick changed. The check is the fight exam's
human stratum played before and after on one engine build, 400 starts by four draws, all 1,600 games alike in outcome, end HP and move
hash (section 4).

## 1. Why

A run costs about 55 CPU-seconds (the 2026-11-07 profile: the emulator's copy and step about 54%, the judgement network 15%, the search's
own Python about 30%). The Python share was spread thin: the rollout built a numpy array of weights, normalised it and called
`Generator.choice` at every step of every playout (about 25 microseconds of numpy against a 150-microsecond emulator step), read the
observation one `ctypes` index at a time (the hand size alone, three times a step), and a tree step read the same legal actions and moves
two or three times (once to merge by key, once more for the heuristic prior, and for the network's request).

## 2. What changed

| Where | Before | After |
| --- | --- | --- |
| `_hand_size`, `_outcome`'s potions held | a generator over ten or six `ctypes` indices | one strided slice and a `count(0)` |
| `_living_targets`, `_aimed_target` | per-enemy, per-buff index reads | one slice of the six enemy slots (`_living_in`), buff ids and amounts by stride |
| `_legal_actions` (`Combat.legal`) | `mask[i]` for 32 indices | one slice of the mask, then `itertools.compress` |
| rollout weights | `np.ones`, a Python loop writing into it, `weights / weights.sum()` | a list of floats (`_tempo_weights`); `live_rollout_weights` is the numpy form of the same list |
| rollout pick | `rng.choice(legal, p=weights)` | `_draw_index`: the same cumulative sums, the same one uniform from `rng.random()` |
| rollout target | `min(targets, key=...)` over a fresh enemy read | a loop over the one enemy slice read for the living check |
| tree step | `_distinct(_world_moves(world))`, then `_node_legal` and `_legal_moves` again for the prior | `_legal_actions` and `_moves_of` once; the merge, `_heuristic_prior` and the network request share them |
| heuristic prior | `live_rollout_weights` as a numpy array, `_checked_weights` (a second normalisation), `_prior_by_key` | the same two normalisations on Python floats (`_normalized` twice) and `_prior_of_moves` over the shared moves |

## 3. What is shared, and what each share relies on

- **`legal` and `all_moves` of a tree step** (`_simulate`, passed to `_tree_choice`, `_heuristic_prior`, `_node_prior_steps`): one read of
  the valid-action mask and of the observation per node, used by the merge by key, the heuristic prior and the network's request. They
  rely on the world not being stepped between those uses: the prior is read and the choice made before `_step_into` plays the move. The
  network request gets `list(legal)`, a list of its own, as a fresh read gave it.
- **`ids` (the hand slots' def ids) and `hand` of a rollout step**: read once, used by the weights and the target. They rely on the
  observation not changing between the pick and the target; `_step` comes after both.
- **`needed` (the enemies' announced attack against the player's block)**: looked up at the first Skill in a position's legal actions and
  not at all when there is none. Only a Skill's weight reads it, and `incoming_damage` has no side effect.
- **The enemy slice of a step**: one read serves the living check (HP, Reviving) and the lowest HP-plus-block pick. It relies on `obs`
  being the buffer the last `_step` filled.
- **Process-wide tables** (`_card_types`, `_aimed_cards`, `_aimed_potions`, `_reviving`): `functools.cache`, as before. They rely on the
  game's name tables being fixed for the process.
- **`_numpy_sum`, `_normalized`, `_draw_index`** are not caches but twins of numpy's arithmetic: `ndarray.sum` for up to 128 float64
  (seven or fewer in order, eight or more as eight interleaved running sums combined pairwise), `cumsum`, the division by the last sum, and
  `searchsorted(side="right")` of the one uniform that `Generator.choice(p=...)` draws. A lone legal move draws nothing in `choice`, so
  none here. They rely on numpy's current implementation; `tests/python/test_search_tidy.py` holds them to numpy on 60,000 random weight
  lists and to `Generator.choice` on 20,000 (the index and the generator's state afterwards), so a numpy that changed either fails the
  suite before it changes a game.

## 4. Proof and measurements

**Identity.** The fight exam's human stratum (400 starts, four draws, player `v2`, config `a3-v1`, the CPU prior so the network's batching
cannot move a number) played before and after on one engine build, the two games files compared with `tools/gym_engine_move_check.py`:
outcome 1,600 of 1,600, end HP 1,600 of 1,600, move hash 1,600 of 1,600 (also the root values: largest difference 0.0; the simulations,
1,814,963, and the decisions, 43,967, add up alike). No game was capped and no search met its clock. The same 1,600 games from the
server-prior run the fight loop made on an earlier engine build agree in all three as well, so the arithmetic of the pick is not a hair
away from numpy's: nothing in the games tells the two apart. The profile's 20 starts (below) and the teacher timing's 20 starts also played
alike in every move hash.

**Cost.** 20 fight-exam starts (6 hallway and 6 elite under `a3-v1`, 8 bosses under `a3-v1@h1e4b16`; AI and human strata alternating), one
process, `time.process_time` per fight, `profile_fights.py` in the results folder. Before and after run side by side so that the machine
load is common; the load was other lab jobs (CPU 41-81% of 8 cores / 16 threads, 6-18 other Python processes).

| Run | Before | After | Change |
| --- | --- | --- | --- |
| 20 starts, pair 1 (load 63-80%) | 1013.8 CPU-s | 892.0 CPU-s | -12.0% |
| 20 starts, pair 2 (load 41-81%) | 907.4 CPU-s | 813.5 CPU-s | -10.3% |
| teacher config (K 2, x16, all 20 starts) | 1207.3 CPU-s | 1084.5 CPU-s | -10.2% |

A fight saves 4.7 to 6.1 CPU-seconds; a run, at the profile's 55.4 CPU-s and the same mix, about 6.2 (11.2% of it). The profile of the 20
starts puts the search's own Python (everything but the emulator's calls and the network) at 24.1% of the time before and 15.1% after, so
about two fifths of it went. What is left is spread thin (`_simulate`, `_moves_of`, `_draw_index`, `_legal_actions`, `_tempo_weights`,
`_puct_choice`, each 1-2% of the profile), the emulator's `step` is 76% of it.

## 5. Tests (`tests/python/test_search_tidy.py`)

The old forms are kept in the test file. Held to them: the reads on 4,000 random observations (ctypes buffers and lists), the positions of
twelve real fights played forward at random (hand, selection, potions, Reviving), the heuristic prior to the last bit, whole rollouts
(final observation, game state, generator state), and whole searches with and without a prior (children, best, line, simulations).
`tests/python/test_live_search.py` is unchanged except that its "rolls out with the corrected weights" spy now watches `_tempo_weights`,
where the rollout reads its weights.
