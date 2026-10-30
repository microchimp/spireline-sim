# The fight judge v0: the leaf estimate on the card (D37 part 2, D41, 2026-10-30)

`rl/judge.py`, `rl/judge_client.py`, `rl/judge_data.py`; the search's side in `rl/live_search.py` (`leaf="judge"`), the
config `search_configs.JUDGE_V0`, the gym's arm `judge-v0` (`rl/fight_gym.py play --lockstep K --judge PATH`). Status words as
in GPU_INFERENCE.md: **measured**, **estimated**, **decided**.

## 1. What it is

- **The judge**: a fight position in (the tokens the agent's network reads, without the end-turn probe), three numbers out: the
  win chance, the end HP as a fraction of max HP if the fight is won, and the potions still held at the end if won. The network
  is the agent's combat network (`CombatNet`: the same embeddings and encoder its value head reads) with a judge head on the
  global token, started from the agent's own weights (gen7/candidate-r1, the board's agent).
- **Its labels**: every decision position of a fight played out by a3-v1 (the board's player, with the board's agent), labelled
  with that fight's outcome. A position's label is one draw from the outcome distribution there; enough of them learn the
  distribution.
- **In the search**: where a simulation leaves the tree, the leaf's value is the judge's estimate instead of a play-out to the
  fight's end: P(win) x (end HP + 10 x potions) + P(death) x -100, the play-out's own units. The search asks for it by a
  request (`LeafAsk(handle)`), so a lockstep process gathers its K fights' leaves and the judge answers them in one batch on the
  card (a second prior server, serving the judge checkpoint; section 4). The confirmation still plays out; a3-v1 and every
  other config are unchanged (their calls carry no `leaf`).

## 2. Data without contamination

- **The exam** is the fight board's 200 Act-1 boss starts (2026-10-29 M2, replayed from the 2026-10-27 g7 night).
- **Training starts** come from three other nights (2026-10-26 g5, 2026-10-29 g7, 2026-10-30 g9): runs replayed exactly with
  their night's agent and engine (`fight_gym._replay_task`), keeping every hallway, elite and boss fight of Acts 1 and 2 on the
  run's path, each as the position at its first combat decision. `judge_data.py starts` refuses the board's night and drops
  any start whose run seed is a board start's.
- **The check** (`judge_data.py check`, and `test_judge.ContaminationTests` on the committed manifest
  `tests/python/fixtures/judge_v0_manifest.json`): no shared start id, run seed or position document (sha256) with the exam,
  no start from the board's night.
- **Held out**: one start in five by the sha256 of its id (`split_of`), every seed and position of a start on one side. The
  epoch is chosen on a tenth of the training starts (another hash); the held-out fifth is read once, for the report.

**Measured** (lab folder `lab/results/judge/2026-10-30-v0/`): 150 runs a night, 450 replays (439 the same as the night's run;
the others left it after a later engine took another macro choice, still positions the agent reached), 3,539 starts: Act 1
hallway 2,231, elite 244, boss 385; Act 2 hallway 527, elite 95, boss 57. The check: 0 shared ids, 0 shared run seeds, 0 shared
positions, 0 from the board's night; 2,791 training starts and 748 held out. Played by a3-v1 with gen7/candidate-r1 on core2,
lockstep K = 32 on 6 processes with the prior on the card: bosses 4 seeds, elites 3, hallway fights 2, so 8,301 games and
131,359 positions (93,368 to train, 9,665 to choose the epoch, 28,326 held out). Rates: Act-1 boss 5,601 games an hour, Act-1
hallway 17,153 (the six stages took 66 minutes of computing; the 450 replays about 22).

## 3. Training

`python rl/judge.py train` (GPU venv): AdamW, the head at 5e-4 and the agent's trunk at a tenth of it, weight decay 0.1,
dropout 0.3, 10 epochs of one-cycle, batch 256; the epoch with the lowest validation loss (epoch 5) is kept, and then the win
logit is Platt-scaled on the validation positions (scale 0.77, shift 0.03: the network was overconfident on new decks). About a
minute on the card. **Decided** on the validation loss alone, among five settings: a whole trunk at the head's rate overfits
within an epoch (on the first 385 boss starts its validation loss rose from epoch 1), a frozen trunk is close behind (0.7628
against 0.7603).

**Measured** on the 28,326 held-out positions (each against the outcome of the one game it was in), beside the agent's old value
head and the search's own root value at that decision (a3-v1's play-outs, about 35 simulations):

| estimate | Pearson (L0) | Spearman (L0) | MSE (L0) | win Brier | win calibration error |
|---|---|---|---|---|---|
| judge v0 | 0.872 | 0.926 | 0.0285 | 0.075 | 0.027 |
| old value head | 0.860 | 0.910 | 0.0315 | 0.077 | 0.026 |
| search root value (play-outs) | 0.886 | 0.938 | (other units) | 0.085 (its death share) | 0.115 |

By kind (judge, Pearson / calibration error): Act-1 hallway 0.860 / 0.005, Act-1 elite 0.638 / 0.074, Act-1 boss 0.685 /
0.025, Act-2 hallway 0.797 / 0.049, Act-2 elite 0.676 / 0.063, Act-2 boss 0.470 / 0.129 (57 starts: too few). The judge still
under-reads the middle of the range (predicted 0.25 where 0.34 won).

**M3's 200 leaves** (the board's Act-1 boss positions at the end of the first turn, each against ten fresh a3-v1 games; the same
measurement as M3's):

| estimate | Pearson (L0) | Spearman (L0) | MSE (L0) | mean L0 (target 0.225) |
|---|---|---|---|---|
| judge v0 (one call) | 0.750 | 0.756 | 0.028 | 0.234 |
| old value head (one call) | 0.676 | 0.691 | 0.037 | 0.236 |
| 30 play-outs | 0.705 | 0.797 | 0.070 | 0.046 |

So one call of the judge reads a boss position better than the old head on every count, and better than 30 play-outs on the
linear correlation and the error; the play-outs still rank the positions better (their own scale is wrong: they read 0.05 where
the player gets 0.23).

## 4. The search with the judge

**The cost of a judged simulation, and JUDGE_SIMS_X (measured).** On the judge data's 94 held-out Act-1 boss starts x 3 seeds
(6 processes, K = 32, prior and judge on the card): a3-v1 0.168 s a decision (35 simulations); judge-v0 at 4 x a3-v1's
simulations 0.233 s (x1.39), at 3 x 0.164 s (x0.97). **Decided:** 3, a3-v1's wall time per decision. A judged simulation
costs about a quarter of a played-out one: the descent and the leaf's encode (0.1 ms) on the CPU, and the leaf's share of a
batch on the card.

**The board (measured, `lab/results/fight-gym/2026-10-30-judge/`).** The 200 Act-1 boss starts x 10 seeds, judge-v0 against
a3-v1 on the same starts and seeds, both played now on the same engine (a3-v1 replayed on today's engine: 1,980 of its 2,000
games are M2's; the other 20 changed with the F08 enemy-HP fixes). 6 processes, K = 64, the prior and the judge each on a
prior server on the card:

| arm | wins | mean end HP / max HP | L0 | own s a game | s a decision | decisions a game | simulations a decision |
|---|---|---|---|---|---|---|---|
| a3-v1 | 36.9% | 0.080 | 0.225 | 4.47 | 0.162 | 27.6 | 31 |
| judge-v0 | 44.1% | 0.121 | 0.282 | 4.41 | 0.132 | 33.4 | 92 |

Paired over the 2,000 games: wins +7.2 points (SE 0.9 over the pairs, 1.5 over the 200 starts), L0 +0.057 (SE 0.005), end HP
+0.041 of max HP (SE 0.003). For scale, M2's x4 (four times a3-v1's simulations and time) won +4.0 and x16 +9.6. Against
M2's own a3-v1 rows the gain reads +7.3.

**Speed and the card (measured, same conditions).** 2,000 games on 6 processes: a3-v1 4,772 fights an hour, the card 4.8%
busy (nvidia-smi's mean utilisation over the run; idle 0.0%); judge-v0 4,802 fights an hour, the card 32.6% busy. The judge's
batches average 39 leaves a process a round; the judge server replayed 130,378 batches of 41 rows (0.93 ms each) and the prior
server 123,065 of 10. 200 games (seed 0): 6 processes a3-v1 5,725 an hour (card 8.6%), judge-v0 3,899 (30.4%); 2 processes
a3-v1 2,389 (3.1%), judge-v0 1,556 (35.2%): with few games a process the judge's batches are small (7 to 26 leaves) and its
round trips cost more than its play-outs saved, so the judge wants many fights a process. Per decision the judge's searches
ran at a3-v1's time or under (0.132 s against 0.162 at K = 64 on 6 processes); its games are longer (33 decisions against 28:
it loses later or not at all), so fights an hour are equal, not higher.

## 5. Tests (`tests/python/test_judge.py`)

Start selection (fights on the path, the floor reached, excluded runs), the held-out split, the contamination counts, the
positions' round trip and labels; the committed manifest shares nothing with the exam and matches the lab's files when they are
here; the row read as the search's value and the L0 it implies; a CPU-device server in judge mode answers a batch as the CPU
judge does (within 1e-5; a batch of one within 1e-6), a refused HELLO falls back to the CPU judge without touching the prior's
fallback; a policy checkpoint is not a judge; JUDGE_V0 is a3-v1 with judged leaves and JUDGE_SIMS_X times the simulations; the
search with JUDGE_V0's settings asks the judge at its leaves, rolls nothing out there and returns a legal move on a fixture
fight (and refuses to run without a judge); the steps yield `LeafAsk`; the lockstep gym plays a judge arm's games on the CPU
judge path and a3-v1's recorded games carry one position per decision; and the trained v0 judge clears its floors on a fixed
sample of held-out positions (`tests/python/fixtures/judge_v0_heldout.npz`; skipped where the checkpoint is not on the
machine).

## 6. Next (v1)

- **More data on the exam's kind.** 385 Act-1 boss starts (57 Act-2 bosses) carry the judge; positions overfit by start within
  epochs. Every night adds 2,000 runs: the always-on loop's generation (GYM_LOOP.md) is the source, with the board's night kept
  out as here.
- **The leaves the search asks about.** The judge learned the positions a3-v1 played, and the search asks it about every
  position its tree reaches, blunders included. Label tree positions too (the search's own leaves, played out), or train on the
  judge-v0 player's games (a second round: the player that uses the judge relabels).
- **Bigger, or the judge inside the tree.** A wider trunk on more data; the judge's value as the tree's prior for which move to
  try (it is called at every leaf already); and the time: at equal time per decision the judge did 3 x the simulations, so the
  same card can buy x16's simulations at a fraction of x16's play-outs.
- **Seams (D37).** Potions used and HP lost are its outputs here only for the leaf; the run planner's use needs the judge's drift
  on decks it has not seen, measured against the player's replays first.
