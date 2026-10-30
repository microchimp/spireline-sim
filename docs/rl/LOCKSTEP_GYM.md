# The lockstep fight gym: K fights a process, the network in one batch (gym build item P1, 2026-10-30)

`rl/gym_lockstep.py`, wired as `rl/fight_gym.py play --lockstep K`. Status words as in GPU_INFERENCE.md: **measured**,
**estimated**, **decided**.

## 1. What it does

- **One process plays K fights at once.** Each fight is `fight_gym.play_steps`: the M2 gym's game loop, whose searches are
  `live_search.search_steps`. A search runs unchanged and in its own order; wherever it needs the prior (the root's prior on a
  resampled probe, a tree node's on its `prior_min_visits`-th visit) it yields the request `(handle, legal)` instead of calling
  the prior, and is sent the answer there. The process advances each of its K fights to its next request (or its end), answers
  every pending request in one batch, and goes round again.
- **Nothing is batched within one search** (no virtual loss; GPU_INFERENCE.md section 8). Across fights nothing is shared but
  the network, so a fight plays as it plays alone (tests pin it, section 4).
- **Who answers the batch** (`--prior`):
  - `cpu`: the agent's CPU prior (`combat_prior.LivePrior`), row by row. Bit for bit the sequential gym: the check against M2,
    and the path for the night.
  - `cpu-stacked`: the same network on the CPU in one padded forward per batch. Not bit-exact with a batch of one.
  - `server`: the batched prior server (`rl/prior_server.py`, CUDA graphs) through one PRIORS frame per batch (section 3), so
    the card sees K times the processes' rows at once. The client's fallback (the CPU prior, sticky) is unchanged.
- **Each fight has its own clock** (`OwnClock`): the time its own steps ran plus its share of each batch it was in. The fight's
  deadline and its searches' safety budgets read that clock, so they mean what they mean in the sequential gym. A row's
  `seconds` is that time; `wall_s` is the wall time from its first step to its end (about K times longer).
- **Grading** (`--grade-wall-s S`): before each move is stepped, one expected solve (`native.solve_expected`, the only solver
  output a grade may read) at horizon 0 under an S-second wall, on a fresh solver. The row's `grades` hold, per decision, the
  solve's status, whether horizon 0 finished, whether the value is exact (the fight ends within the turn: a last turn), the
  value, the value of the move the search chose (mapped onto the solver's keys as `coverage.py regret` does) and the solver's
  best move; `graded` sums them per game. The move played is the search's, always.
- **Worlds** (`--worlds reload|memory`, default `reload`). `memory` loads each start once per process and forks each seed's
  world from it in memory. **Decided not the default:** the load seed survives a resampling fork (the fork reshuffles the draw
  pile from the order the load dealt; measured on 6 starts, every trajectory differed by load seed), so seeds j >= 1 would be
  other worlds than M2's and the fight board's squares would no longer pair, for a saving of 3.1 ms a game (a reload against a
  0.07 ms fork, measured; 0.05% of a 6.4 s game). Seed 0 is the same world either way.

## 2. Guards (carried over from the M2 gym's commit 2b2fbe99)

- A fight past its deadline (ten times its arm's mean seconds, read between decisions on its own clock) stops there: capped
  and lost, with its turns, decisions and seconds. The other fights go on.
- The parent reads two shared numbers per process without a message: the serial of the fight being advanced and a heartbeat
  (one beat per advance). A process whose heartbeat stops for `--hang-s` (default 180 s; a native step that never returns, as
  F18's) or whose private memory passes `--mem-gb` (default 3) is killed by its own handle. What it sent before is read from
  its pipe first; the fight it was advancing gets a capped row with how far it got; its other unfinished fights go back to the
  queue and are played from their start in a fresh process (an outcome depends on its seeds only). A fight sent back three
  times is capped.
- A process holds at most 100 fights: the emulator's pool holds 256 combats, and a fight waiting on the network holds two (its
  world, and a simulation's fork or the root's probe).
- The job waits while the night is on or due, while main.lock exists, or while another live QUIET.lock line stands
  (`fight_gym.blocked`); it takes no QUIET.lock line itself (a two-core job). The server path takes GPU_OK's first line only
  while it runs, and only when the file is empty or absent.

## 3. The PRIORS frame

`prior_protocol.pack_priors` / `unpack_priors` and their reply: one request carries many rows.

```
PRIORS  (client -> server)
  magic, kind=4 (u16), req_id (u32), rows R (u16), flags (u16)
  sizes   u16[R]                  each row's token count
  types   u8[sum]  ids i32[sum]  owners u8[sum]  feats f32[sum*18]   the rows one after another, field by field
REPLY   (server -> client)
  magic, kind=4, status (u16), req_id (u32), rows R (u16), logits f32[R*32]
```

The server enqueues a frame's rows together (queue items are lists of requests; a PRIOR frame is a list of one), the batcher
drains whole items and replays at most `--max-batch` rows at a time, and the reply goes out once every row has its logits.
`RemotePrior.batch` sends one frame and turns each row into probabilities with `legal_weights`, as a single call does.

## 4. Tests (`tests/python/test_gym_lockstep.py`)

On the CPU path, with arena fights and the fixture prior (`prior_min_visits` 0: every new node asks), on core2 in spawned
processes as the gym plays: K = 1 plays the sequential gym's games (`fight_gym.run_tasks`) and K = 4 plays K = 1's (won, end
HP, turns, decisions, simulations, root values, the moves' hash, M3's leaf); grading at 0.5 s records a grade per decision and
changes no move; `--worlds memory` gives seed 0's games; F18's fixture as a fight is killed as hung and the two other fights are
replayed to the same rows; stand-in fights pin the hang, memory and raise guards; the PRIORS frame round-trips and a batch through
a CPU-device server matches the CPU prior within 1e-6 (a batch of one bit-exact). `test_solver_hindsight_pins` probes
`rl.gym_lockstep`.

## 5. Measured (2026-10-30, results dir `lab/results/fight-gym/2026-10-30-gpu/`)

M2's 200 Act-1 boss starts at seed 0 (so every game pairs with M2's seed-0 game), agent gen7/candidate-r1, core2, 2 processes
at BelowNormal while M2's resume ran its 6 workers beside them. The wall rates move with that load; the robust comparison is
each game's own time on the same starts (`seconds`, the time its own steps ran plus its share of the batches).

- **Bit for bit.** The CPU path at K = 1 plays all 200 of M2's a3-v1 seed-0 games: won, end HP, turns, decisions, simulations,
  potions, first move, root value and L0 alike (200/200). Every other row too: the GPU path at K = 8 and 64, the stacked CPU
  forward, grading on, all identical to the CPU path game for game; x16 on the GPU path plays M2's x16 seed-0 games (50/50).
  The first run did not: the lockstep processes loaded the emulator library before choosing the engine and played on the old
  one, which changed 7 of 200 games (fixed on this branch before the measurement; a test pins the engine). The original sequential gym, played in one
  process in start order, reproduces M2's 200 (the control).
- **Fights per hour** (a3-v1; wall rate on 2 processes, and own seconds per game on the same starts as the CPU K = 1 row):

  | row | fights/h | per core/h | own s/game (same starts) | network s/game | CPU s/game |
  |---|---|---|---|---|---|
  | CPU, K = 1 (the before) | 1,819 | 910 | 3.89 | 0.50 | 3.93 |
  | CPU stacked forward, K = 64 | 1,936 | 968 | 3.68 (x1.06) | 0.21 | 3.68 |
  | GPU server, K = 8 (100 starts) | 1,824 | 912 | 4.09 -> 3.79 (x1.08) | 0.12 | 3.68 |
  | GPU server, K = 64 | 1,999 | 1,000 | 3.57 (x1.09) | 0.08 | 3.51 |
  | GPU server, K = 64, graded at 1 s (100 starts) | 1,168 | 584 | 3.76 -> 6.00 against ungraded | 0.10 | 5.95 |

  The M2 CPU workers played 6.4 s per game at 6 workers (about 560 per core per hour) on a busier machine.
- **x16** (leaf switch off): own time on the 16 starts both rows played 53.4 s -> 50.0 s a game (x1.07); the GPU row on 50
  starts took 64 s of wall per game per process (M2's workers: 74 s). Network share of the CPU path: 10%.
- **Where the CPU goes** (cProfile, one process, 12 a3-v1 games, CPU prior): rollouts 67% (their emulator steps included), the
  tree's descent steps 10%, the network's forward 10.5%, its encode 1.4%, forks 0.7%; the emulator's step is 54% of all. So the
  GPU can take at most the network's ~12%: the gain above is that, nearly all of it.
- **The card.** Server replays at K = 64 x 2 processes: 1,287 replays for 37,162 rows (mean 29 rows; 293 replays of a full 64),
  6.4 ms a replay, busy 2.3% of the wall; nvidia-smi read 16% mean utilisation with the desktop. Numerics: section 9.2's tool
  at 64 client threads on 4,000 recorded g7 positions, max |dlogit| 2.9e-6 (gate 1e-4), 0 argmax flips: PASS.
- **Grading** (horizon 0, 1 s wall): 2,859 decisions; horizon 0 finished on 93.2%, the value exact on 68 (2.4%, the last turns
  of 32 of 100 games), regret 0 on all 68. It costs 2.4 s a game, 82% of it in the 192 solves (6.7%) that reach the wall; a
  0.1 s wall would keep most of horizon 0 for a fraction of that (not measured).
- **Stop rule** (gym-build.md P1): x16 with the leaf switch off gains +7%, under +20%. Landed for the batch plumbing; the real
  gain waits on the value head replacing the rollouts (P4), which are two thirds of the CPU.
