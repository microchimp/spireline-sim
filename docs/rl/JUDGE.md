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

## 7. Judge v1: retrained on the fight-gym loop's buffer (2026-11-08, training design 1.3's fallback)

Board A (judge v0 against x16 on 400 paired boss games an arm) failed the adoption rule: j480 3.4 points under x16, j960 2.5 under.
The rule's fallback is a judge retrained on the loop's x16 fights. `judge_data.py buffer` builds the positions from the loop's
replay buffer, `judge.py train` trains (unchanged apart from `--version` and `--manifest`, which keep the data's window in the
checkpoint), `judge.py eval` scores any number of checkpoints on the same positions. Files: `referee/agent/judge/v1/` (`judge.pt`,
`REPORT.md`, `eval-heldout.json`, `data-manifest.json`); v0's folder is untouched. Tests: `tests/python/test_judge_buffer.py`.

- **Data** (**measured**): the buffer's 183 shards, segments 1 to 185, 2026-10-30T20:19Z to 2026-11-08T11:07Z, the x16 teacher's
  fights (players v0 and c0001 to c0175) in every start kind the loop draws (night replays, human starts, pre-loss starts): 20,550
  fights of 6,840 starts of 2,654 runs, **416,279 positions** (one per decision, tokens as the root prior reads them, labelled
  with the fight's won / end HP fraction / potions held / L0; a shard from before `potions_held` gives the potions it began with
  less those it used, right in 97.5 in 100). Left out: the exam's runs (`exam/exclude_runs.json`), the M2 board's 200 Act 1 boss
  starts and board A's start lists, capped fights and fights of cut segments (no value label): 3,227 fights; the check on the output
  finds no shared start id or run. Held out **by run** (`split_of` on the run seed, one in five: 526 runs, 88,624 positions; the
  validation tenth by run too): v0 split by start, which puts one run's deck on both sides.
- **Training** (**measured**): the agent network of v0's init (gen7/candidate-r1), trunk at 0.03 of the head's rate, weight decay
  0.1, dropout 0.3, 10 one-cycle epochs of batch 256, best epoch by validation loss, win logit Platt-scaled; chosen among seven
  settings by validation loss alone. 3.5 minutes on the card, BelowNormal.
- **Held-out, v1 against v0, the same 88,624 positions** (Pearson / Spearman of the L0, win calibration error): all 0.820 / 0.860 /
  0.028 against v0's 0.802 / 0.849 / 0.083. Act 1 boss 0.717 / 0.714 / 0.033 against 0.673 / 0.690 / 0.114; Act 2 boss 0.615 / 0.600 /
  0.045 against 0.662 / 0.613 / 0.116 (53 held-out starts); Act 2 elite 0.591 against 0.471 (still over-read: mean P(win) 0.748 for a
  rate of 0.679); human starts 0.702 against 0.662. On the 77,169 positions of runs v0 never saw: 0.811 against 0.789. What changed
  most is the level: v0, trained on a3-v1's fights, read x16's boss positions 11 points low at Act 1 and 12 at Act 2; v1 reads
  them within 1 point. Rank agreement moved little, and the search's own root value still ranks better (0.832 / 0.902).

## 8. Judge v2: trained on the corpus (D58 C2, D57 item 1; 2026-11-11)

`rl/judge_train.py` trains the judge on the judge corpus (`JUDGE_CORPUS.md`: the human starts' fight positions with the human's own result and the
search settled on a ladder of budgets, and the loop's self-play positions) and scores any number of checkpoints on the same held-out positions.
`judge.py` gains the policy-and-source readings it needs (`judge_out_policy`, `judge_out_all`, the `src_emb` row); `judge.py train` and the v0 and v1
checkpoints are unchanged in what they compute. Tests: `tests/python/test_judge_train.py` (24: the reader on a synthetic store, the weights, every
loss term, the search calibration, the source input, a short training run on a tiny network, the held-out scoring). Folder for the checkpoints:
`referee/agent/judge/v2/` (the corpus-trained judge: 599 human fights and the self-play half, trained 2026-11-11), `v2-sp/` (the interim, self-play half of the corpus only).
v0 and v1 are untouched.

**The loss.** Four labels train the judge's three outputs (win logit, end-HP logit, potions) and each term is a weighted mean over the rows that
carry its label (`LossCfg`): the **teacher's outcome** of the game the position was in (the old target); the **human's outcome** of the fight
(human rows); the **search value**: the judge's own leaf value V = P(win) x (end HP + 10 x potions) + P(death) x -100 (`judge_client.estimate`,
what the search reads) against the search's root value, Huber on V / 50, a settled turn counting 1, a soft turn 0.5, a turn kept as played 0.25
(`tier_w`); the **search win**: the win logit against 1 - the search's death share (settled and soft turns); and, as an auxiliary, the **moves**:
the policy head `JudgeNet` already carries against the search's visit distribution over the action slots (token 1 + a is action a, checked on the
corpus), which only shapes the trunk. Rows weigh by the corpus's sampling weight (`pool`: 1 / inclusion probability for the human rows, 1 for
self-play; clipped at 20 times the mean and renormalised); the split is the corpus's, **by run** (one in five held out; the exam fifth is not in
the corpus and the loader refuses it), the validation tenth for the epoch and the win calibration by run too.

**Two things the self-play measurement forced.** (1) *The search's values are on another scale than the games' results.* The loop's root values
(rollouts under a weaker policy than the teacher's) sit 29 HP under the realised result of the games at bosses (mean -65.9 against -36.7), 23 at
elites (-13.5 against +9.9) and 10 at hallways; a third of all root values are exactly -100. Trained on both, the judge splits the difference: the
win head learned to say 0.75 logits too low (the Platt step had to add it back) and the outcome correlations fell. `calibrate_search` puts the root
value and the death share on the outcomes' scale first, by a monotone map (equal-count bins, adjacent violators pooled, equal values one knot) per
label tier and kind fitted on the TRAINING rows only; the search's ranking of positions is kept, its level is the games'. (2) *A human plays
the same position better than the teacher.* One win head cannot be calibrated to both outcome distributions, so the judge head reads a **source**
input (`JudgeNet.src_emb`: row 0, the teacher and every use in the search, is zero and never moves; row 1 is added for the human reading), and only
the human reading is trained to the human's outcome. A checkpoint without the row (v0, v1) loads and reads unchanged.

**Self-play half, measured** (the corpus's 917,068 self-play positions of 20,795 starts and 7,392 runs, from 359 buffer shards of the loop to 2026-11-10, less 3,227 fights of the exam's, the boards' and the M2 board's runs and 27 capped ones;
655,811 to train, 68,794 for the epoch and the calibration, **192,463 held out** by run; 538 s on the card, BelowNormal, 10 epochs of batch 512,
the agent's network of v0 and v1 as the trunk's start, head 5e-4, trunk 0.3 of it, weight decay 0.1, dropout 0.3). Held-out settings (Pearson of the
judge's L0 against the teacher's outcome of the game; all, Act 1 boss, Act 2 boss; Spearman all):

| | all | Act 1 boss | Act 2 boss | Spearman all | note |
|---|---|---|---|---|---|
| v0 | 0.781 | 0.650 | 0.431 | 0.834 | a3-v1's fights |
| v1 | 0.809 | 0.684 | 0.610 | 0.860 | the buffer to 11-08, 416k positions |
| trunk 0.1, raw root values | 0.824 | 0.713 | 0.674 | 0.864 | |
| trunk 0.1, no search terms | 0.830 | 0.719 | 0.688 | 0.877 | the raw root values cost about 0.006 |
| trunk 0.1, no move term | 0.821 | 0.707 | 0.666 | 0.862 | the move term is worth about 0.003 |
| trunk 0.3, raw root values | 0.830 | 0.726 | 0.694 | 0.867 | lowest validation loss of the raw settings |
| trunk 1.0, raw root values | 0.835 | 0.737 | 0.701 | 0.873 | validation loss rises from epoch 5 |
| balanced weights (equal cells), trunk 0.1 | 0.812 | 0.688 | 0.641 | 0.855 | the small Act 2 cells overfit |
| **v2-sp: trunk 0.3, calibrated root values** | **0.835** | **0.738** | 0.686 | **0.875** | the setting kept; win calibration error 0.009 |

v2-sp against v1 and v0 per cell (Pearson / Spearman / win calibration error; held-out positions):

| cell | positions | v2-sp | v1 | v0 |
|---|---|---|---|---|
| all | 192,463 | 0.835 / 0.875 / 0.009 | 0.809 / 0.860 / 0.040 | 0.781 / 0.834 / 0.113 |
| Act 1 boss | 89,534 | 0.738 / 0.741 / 0.012 | 0.684 / 0.694 / 0.048 | 0.650 / 0.671 / 0.134 |
| Act 1 elite | 18,821 | 0.717 / 0.820 / 0.019 | 0.680 / 0.820 / 0.045 | 0.691 / 0.799 / 0.059 |
| Act 1 hallway | 32,700 | 0.891 / 0.931 / 0.005 | 0.889 / 0.927 / 0.006 | 0.897 / 0.922 / 0.006 |
| Act 2 boss | 18,126 | 0.686 / 0.676 / 0.028 | 0.610 / 0.615 / 0.087 | 0.431 / 0.417 / 0.284 |
| Act 2 elite | 10,145 | 0.646 / 0.721 / 0.037 | 0.648 / 0.729 / 0.042 | 0.594 / 0.658 / 0.134 |
| Act 2 hallway | 23,137 | 0.837 / 0.895 / 0.021 | 0.835 / 0.902 / 0.022 | 0.818 / 0.882 / 0.082 |

Boss rows are where the gain is (Act 1 +0.054, Act 2 +0.076 over v1 in Pearson), elites and hallways are level. The v2-sp judge's V against the loop's
search as played reads Pearson 0.868 (v1 0.822) and 22 HP above it at the mean, as v1 does (21.9): the raw root values read low, which is what the calibration measured. These
positions are the loop's own, not the humans': **the human positions and the settled labels are what the corpus build adds, and v2-sp has seen neither.**
The held-out half is the loop's self-play of runs none of the three judges trained on (v1's held-out runs are the same runs; v0 was trained on three nights, some
of whose runs fall in it).

**v2 on the corpus (2026-11-11 01:37Z, the manifest's `usable_for_training: true`).** The corpus's human build had been paused behind another lane's
`run-teacher` QUIET line (workers wait on every line but the loop's); once it was dropped the build ran, and the manifest said usable with **599 human fights
of 537 runs** (2,270 positions: boss Act 1 / Act 2 380 / 385, elite 375 / 384, hallway 376 / 370;
1,821 to train, 449 held out by run) beside the 917,068 self-play positions: **919,338 positions, 657,433 to train, 68,993 for the epoch and the
calibration, 192,912 held out**. Tiers of the human boss and elite rows: 442 + 352 settled, 323 + 407 soft; hallway rows and every self-play row `played`.
The human share is small (0.25% of the rows, about 2% of the weight after the corpus's weights `1 / inclusion probability`, clipped at 4 times the mean: the
design lane's clip), so this v2 is v2-sp plus a first look at the humans' positions; the retrain as v2b is for `complete: true`, hours of CPU away.
The search calibration on the human rows (training rows only, 12 and 14 bins since a group has few rows): soft turns read -43 HP at the root against +2.6 realised,
settled turns -51 against -30. Same recipe as v2-sp (the one command below, `--wclip 4`); 1,228 s on the card; `referee/agent/judge/v2/judge.pt`, sha256
`83440687669c864b24db2a91969fe16b72ec135b2cd5ccda60ceb9c8e455048c`; best validation loss 1.0886; win calibration scale 0.946, shift -0.082.
For the board it is `judge-board.pt` beside it (sha256 `94eeca3c92efd80853e65e546178104eb8f06ddc2f3d82c223bf57f167df9fbb`): the same checkpoint without the source
row, which the board's older engine tree (the copy board A and the v1 board played on) cannot load; row 0 is zero, so the two read identically (checked on 64
human boss positions: difference 0.0).

Held-out, v2 against v2-sp, v1 and v0 (Pearson / Spearman / win calibration error against the teacher's outcome; all 192,912 positions by run):

| cell | positions | v2 | v2-sp | v1 | v0 |
|---|---|---|---|---|---|
| all | 192,912 | 0.832 / 0.873 / 0.012 | 0.835 / 0.875 / 0.009 | 0.809 / 0.860 / 0.040 | 0.781 / 0.834 / 0.113 |
| Act 1 boss | 89,618 | 0.731 / 0.735 / 0.016 | 0.738 / 0.741 / 0.012 | 0.684 / 0.694 / 0.048 | 0.650 / 0.671 / 0.135 |
| Act 1 elite | 18,879 | 0.703 / 0.812 / 0.023 | 0.717 / 0.820 / 0.019 | 0.680 / 0.820 / 0.045 | 0.691 / 0.799 / 0.060 |
| Act 1 hallway | 32,790 | 0.891 / 0.927 / 0.004 | 0.892 / 0.931 / 0.005 | 0.889 / 0.927 / 0.006 | 0.897 / 0.922 / 0.006 |
| Act 2 boss | 18,221 | 0.685 / 0.673 / 0.025 | 0.686 / 0.675 / 0.027 | 0.609 / 0.615 / 0.088 | 0.431 / 0.417 / 0.285 |
| Act 2 elite | 10,214 | 0.654 / 0.725 / 0.034 | 0.645 / 0.720 / 0.037 | 0.648 / 0.729 / 0.042 | 0.593 / 0.658 / 0.134 |
| Act 2 hallway | 23,190 | 0.831 / 0.894 / 0.019 | 0.837 / 0.895 / 0.021 | 0.834 / 0.902 / 0.022 | 0.818 / 0.882 / 0.082 |

On the self-play positions v2 and v2-sp are level (the humans' rows are too few to move them, and v2's Act 1 boss and elite are 0.007 and 0.014 lower; not tested against a second seed). On the **human positions held out** (449, 84 Act 1 boss and 95 Act 2 boss: small, read as direction): the judge's V against the search's settled
and soft root values (306 turns) Pearson / Spearman 0.790 / 0.849 against v2-sp 0.776 / 0.846, v1 0.680 / 0.744, v0 0.648 / 0.666, at Act 2 boss 0.767 / 0.788 against 0.712 / 0.751,
0.669 / 0.652 and 0.345 / 0.393; against the human's own outcome of the fight (read with the source input) 0.646 / 0.776 against 0.643 / 0.751, 0.610 / 0.727, 0.570 / 0.666, win
calibration error 0.078 against 0.111, 0.139, 0.269. Every judge reads these positions about 30 HP above the search's settled value (the search's level is low, as measured).

**To train v2 the moment the manifest says `usable_for_training: true`** (GPU venv; about 10 to 15 minutes; leaves the gym trainer running):

    D:/spireline/venv-gpu/Scripts/python.exe -I rl/judge_train.py train --version judge-v2 \
        --init D:/spireline/referee/agent/gen7/candidate-r1/combat_net.pt --out D:/spireline/referee/agent/judge/v2/judge.pt \
        --compare D:/spireline/referee/agent/judge/v0/judge.pt D:/spireline/referee/agent/judge/v1/judge.pt

It writes `judge.pt` (the corpus's manifest sha256 and counts, the weights, the loss settings and the search calibration inside), `judge.report.json`
and `judge.eval.md` (the held-out tables of the three judges). `eval` scores any checkpoints again (`judge_train.py eval --judge A.pt --judge B.pt`);
`summary` counts what the trainer would read. A retrain as v2b when the manifest says `complete: true` is the same command with another `--out`.

**The board, played 2026-11-11** (rule 1.3 of the training design, board A's starts and draws: Act 1 boss the fight board's 200 starts x draws 0 and 1, Act 2 boss board A's
100 starts x draws 0 to 3, so 400 paired games an arm in each act; x16 is board A's own games, v1 and v0 the v1 board's and board A's rows on the same games). The driver is
`D:/spireline/referee/backlog/scratch-board-a/v2/board_v2.py` (the v1 board's on the same engine tree; Act 2 first; no timing sample; `JUDGE_BOARD_PATH`, `JUDGE_BOARD_OUT`,
`JUDGE_BOARD_HOLD=1` for a `judge-train` QUIET line, none by default so as not to pause the corpus build), with `analyze.py`, `analyze2.py`, `compare_prev.py` and `make_summary.py`;
`summary.md` there has the tables. It played `judge-board.pt`, six processes, K 16, BelowNormal, 02:02Z to 10:49Z (25,379 s of play, stopped once for the 03:10Z night
window; one j960 Act 1 fight hit the 3,600 s cap and counts as a loss, no search hit the safety clock).

| per 100, SE clustered by start | x16 | j480 (v2) | j960 (v2) |
|---|---|---|---|
| Act 1 boss (400 games an arm) | 48.8 +- 3.3 | 52.5 +- 3.3 | 52.8 +- 3.3 |
| Act 2 boss (400 games an arm) | 19.5 +- 3.2 | 16.8 +- 2.9 | 18.5 +- 3.1 |

Paired differences against x16, points per 100: diff (SE over starts); one-sided 95% lower bound; games only the arm won / only x16 won; one-sided McNemar p that the arm is better.

| | Act 1 | Act 2 | both (800 pairs) |
|---|---|---|---|
| j480 - x16 | +3.8 (2.0); +0.5; 33/18; p 0.024 | -2.8 (2.1); -6.2; 19/30; p 0.957 | +0.5 (1.5); -1.9; 52/48; p 0.382 |
| j960 - x16 | +4.0 (1.8); +1.0; 32/16; p 0.015 | -1.0 (2.3); -4.7; 24/28; p 0.756 | +1.5 (1.5); -0.9; 56/44; p 0.136 |

v2 against v1 on the same games: j480 +2.4 pooled (Act 1 +3.8, p 0.014; Act 2 +1.0), j960 +2.1 pooled (Act 1 +2.8, Act 2 +1.5; p 0.039 pooled); against v0: j480 +3.9, j960 +4.0
pooled (p 0.001). Seconds a boss fight (own clock, indicative: the machine was full): Act 1 x16 66.5, j480 47.9 (0.72), j960 110.1 (1.66); Act 2 x16 79.7, j480 76.3 (0.96), j960 161.5 (2.03).

**Verdict by rule 1.3: pooled, j480** (j960 is +1.5 with p 0.136, short of +3 with p < 0.05; j480's one-sided 95% lower bound against x16 is -1.9 by the start-clustered SE,
-1.6 by the pair SE, just above the rule's -2). By act: **Act 1 boss, j960** (+4.0, p 0.015); **Act 2 boss, keep x16** (j960 -1.0, j480 -2.8 with bound -6.2). The pooled pass is
marginal and comes from Act 1; the Act 2 boss, where v0 and v1 lost most, is still not won (v0 -5.2, v1 -3.8, v2 -2.8 for j480). The judge was trained on 599 human fights, 0.25% of its
positions; the corpus has since grown to 4,680 fights (`complete` is not set), and the retrain as v2b on it is the one command above. Nothing is switched: the decision is the
orchestrator's.

### 8.1 Judge v2b: the same recipe on the complete corpus (2026-11-12)

The corpus's human build finished (`manifest.json` `complete: true`, sha256 `96bc0bfcaa6c1ed18f43cc8a7912682aad7a5c3c231e87fd496c7b507ae5c9d2`; `JUDGE_CORPUS.md`), and addendum 10 of D58 asked for the retrain before any switch,
read per act. v2b is the one command above with `--version judge-v2b`, `--out referee/agent/judge/v2b/judge.pt` and `--compare` naming v0, v1, v2-sp and v2 (the trainer takes any version string;
the recipe, `--wclip 4`, the init network, the seed and every setting are v2's). Trained 2026-11-12 07:11Z to 07:22Z on the idle card (GPU venv, BelowNormal): **551 s** of training (the whole command 11 minutes with the 81 s corpus read and the scoring), 10 epochs, best validation
loss 1.1271 (epoch 10; v2's 1.0886 was on another validation split and another mix, not comparable), win calibration scale 0.997, shift -0.085. `judge.pt` sha256
`445f576cf54ec924131fbf76599f95903c9eddd321ceba6b2cb4177a5085d4ad`; `judge-board.pt` beside it (the same checkpoint without `src_emb.weight`, for the board's older engine tree), sha256
`62bcaf0dbeaf8c42c846f074d289c882376f0aed7ae27849e4f61b2725aab643`; the two read identically on 64 human boss positions (largest difference 0.0). v0, v1, v2-sp and v2 are untouched.

**What the trainer read** (positions; the corpus's split by run): 968,159 in all, 692,392 to train, 72,959 for the epoch and the calibration, 202,808 held out.

| source | cell | train | validation | held out |
|---|---|---|---|---|
| human (13,371 fights of 6,418 runs: 9,571 / 1,088 / 2,712 fights by split) | boss | 14,443 | 1,640 | 4,060 |
| | elite | 14,607 | 1,625 | 4,241 |
| | hallway | 7,531 | 900 | 2,044 |
| | all | 36,581 | 4,165 | 10,345 |
| self-play | boss | 371,546 | 35,713 | 107,660 |
| | elite | 94,923 | 10,152 | 28,966 |
| | hallway | 189,342 | 22,929 | 55,837 |
| | all | 655,811 | 68,794 | 192,463 |

By act the human rows are Act 1 / Act 2 boss 4,154 / 10,289 train, 466 / 1,174 validation, 1,185 / 2,875 held out; elite 7,366 / 7,241, 791 / 834, 2,098 / 2,143; hallway 3,680 / 3,851, 459 / 441, 1,059 / 985. The human share of the
rows is 5.3% (v2: 0.25%); labels by tier: 20,273 settled, 20,343 soft, 927,543 played. **The search calibration on the human rows** (training rows only; the search's root value and death share put on the outcomes'
scale by a monotone map per tier and kind, `calibrate_search`): settled boss turns 7,950 rows, root value -70.9 before and -46.8 after against -46.7 realised, win share 0.235 to 0.431 against 0.432; settled elite 6,599 rows, -20.3 to +0.9 against +1.0, win share 0.616 to 0.771 against 0.772;
soft boss 6,493 rows, -57.8 to +1.4 against +1.4, win share 0.343 to 0.814 against 0.814; soft elite 8,008 rows, -25.6 to +14.8 against +14.8, win share 0.587 to 0.891 against 0.891. The self-play rows' calibration is v2's (boss 300,654 rows,
-65.9 to -36.8 against -36.7; elite 77,856, -13.5 to +9.8 against +9.9; hallway 166,997, +31.4 to +41.6 against +41.8).

**Held-out, v2b against v2, v2-sp, v1 and v0** (Pearson / Spearman / win calibration error against the teacher's outcome of the game; all 202,808 positions by run; the held-out half is the corpus's: v2's 192,912 positions were 192,463 self-play and 449 human, these are the same 192,463 and 10,345 human, so v2, v2-sp, v1 and v0 are scored again here on the same positions):

| cell | positions | v2b | v2 | v2-sp | v1 | v0 |
|---|---|---|---|---|---|---|
| all | 202,808 | 0.831 / 0.873 / 0.012 | 0.828 / 0.869 / 0.013 | 0.832 / 0.872 / 0.010 | 0.805 / 0.856 / 0.041 | 0.774 / 0.827 / 0.118 |
| Act 1 boss | 90,719 | 0.730 / 0.735 / 0.017 | 0.731 / 0.735 / 0.016 | 0.737 / 0.740 / 0.012 | 0.684 / 0.694 / 0.048 | 0.649 / 0.670 / 0.135 |
| Act 1 elite | 20,919 | 0.713 / 0.807 / 0.022 | 0.702 / 0.804 / 0.021 | 0.718 / 0.812 / 0.017 | 0.681 / 0.815 / 0.043 | 0.688 / 0.793 / 0.064 |
| Act 1 hallway | 33,759 | 0.894 / 0.928 / 0.004 | 0.891 / 0.927 / 0.004 | 0.892 / 0.931 / 0.005 | 0.888 / 0.927 / 0.006 | 0.897 / 0.921 / 0.006 |
| Act 2 boss | 21,001 | 0.703 / 0.696 / 0.018 | 0.677 / 0.669 / 0.036 | 0.680 / 0.672 / 0.029 | 0.604 / 0.613 / 0.102 | 0.417 / 0.416 / 0.309 |
| Act 2 elite | 12,288 | 0.659 / 0.729 / 0.048 | 0.662 / 0.728 / 0.030 | 0.656 / 0.726 / 0.031 | 0.654 / 0.732 / 0.041 | 0.606 / 0.669 / 0.135 |
| Act 2 hallway | 24,122 | 0.835 / 0.898 / 0.019 | 0.832 / 0.893 / 0.018 | 0.837 / 0.894 / 0.019 | 0.834 / 0.901 / 0.021 | 0.818 / 0.881 / 0.081 |

On the whole held-out set (95% self-play positions) v2b is level with v2 and v2-sp (all 0.831 against 0.828 and 0.832, Act 1 boss 0.730 against 0.731 and 0.737; the loop's night-start positions alone 0.849 against 0.850 and 0.853:
the human rows, 5% of the training rows, cost the loop's own positions at most 0.004) and well ahead of v1 and v0; **Act 2 boss is where it moves, Pearson 0.703 against v2's 0.677 and v2-sp's 0.680, win calibration error 0.018 against 0.036 and 0.029.** Act 2 elite's
calibration error is the one cell that got worse (0.048 against 0.030; Pearson level at 0.659 against 0.662). The scores of v2 here differ from section 8's in the third place only because the human half of the held-out
set grew.

**The human held-out block** (positions from Spire Codex public run export for v0.107.1's training fifths, held out by run: **10,345 positions of 2,712 fights**; boss 4,060 (1,185 Act 1, 2,875 Act 2), elite 4,241, hallway 2,044;
8,301 are boss and elite turns, every one with a settled or soft search value). The judge's V read with the human source input against **the human's own outcome of the fight** (L0; Pearson / Spearman / win calibration error):

| cell | positions | v2b | v2 | v2-sp | v1 | v0 |
|---|---|---|---|---|---|---|
| all human positions | 10,345 | 0.663 / 0.726 / 0.014 | 0.628 / 0.700 / 0.054 | 0.608 / 0.662 / 0.122 | 0.573 / 0.652 / 0.150 | 0.523 / 0.593 / 0.286 |
| Act 1 boss | 1,185 | 0.547 / 0.588 / 0.055 | 0.509 / 0.562 / 0.103 | 0.486 / 0.533 / 0.189 | 0.489 / 0.533 / 0.193 | 0.419 / 0.484 / 0.328 |
| Act 2 boss | 2,875 | 0.515 / 0.522 / 0.029 | 0.451 / 0.463 / 0.093 | 0.433 / 0.438 / 0.261 | 0.360 / 0.392 / 0.372 | 0.253 / 0.344 / 0.629 |
| Act 1 elite | 2,098 | 0.668 / 0.713 / 0.016 | 0.646 / 0.687 / 0.047 | 0.646 / 0.658 / 0.034 | 0.627 / 0.693 / 0.042 | 0.610 / 0.671 / 0.108 |
| Act 2 elite | 2,143 | 0.658 / 0.685 / 0.015 | 0.634 / 0.663 / 0.037 | 0.615 / 0.630 / 0.091 | 0.600 / 0.638 / 0.072 | 0.535 / 0.576 / 0.212 |
| Act 1 hallway | 1,059 | 0.796 / 0.859 / 0.015 | 0.788 / 0.854 / 0.027 | 0.793 / 0.850 / 0.020 | 0.786 / 0.852 / 0.023 | 0.805 / 0.847 / 0.014 |
| Act 2 hallway | 985 | 0.734 / 0.770 / 0.027 | 0.726 / 0.747 / 0.016 | 0.704 / 0.736 / 0.043 | 0.721 / 0.749 / 0.032 | 0.685 / 0.735 / 0.096 |

The human's win rate on these positions is 0.85 (0.74 Act 1 boss, 0.76 Act 2 boss); the mean win probability v2b gives is 0.84 (0.71, 0.74), v2's 0.87 (0.77, 0.76), v2-sp's 0.74, v1's 0.72, v0's 0.56 (Act 2 boss 0.14). The gain
of v2b over v2 in Pearson against the human's outcome is **+0.035 over all 10,345 positions (95% interval by bootstrap over the 2,712 fights +0.026 to +0.043)**, +0.059 at the 4,060 boss positions (+0.044 to +0.074; 1,050 fights),
+0.064 at Act 2 boss (+0.046 to +0.083; 742 fights) and +0.038 at Act 1 boss (+0.009 to +0.069; 308 fights); against v1 +0.089 (+0.077 to +0.103) all, +0.155 at Act 2 boss (+0.126 to +0.183). The same positions read with the teacher
reading (the one the search asks, against the teacher's outcome of the game the position was in; Pearson / Spearman / win calibration error):

| cell | positions | v2b | v2 | v2-sp | v1 | v0 |
|---|---|---|---|---|---|---|
| all human positions | 10,345 | 0.770 / 0.804 / 0.023 | 0.746 / 0.783 / 0.047 | 0.753 / 0.789 / 0.038 | 0.718 / 0.771 / 0.075 | 0.654 / 0.710 / 0.211 |
| Act 1 boss | 1,185 | 0.712 / 0.711 / 0.044 | 0.700 / 0.698 / 0.058 | 0.702 / 0.701 / 0.052 | 0.671 / 0.691 / 0.071 | 0.587 / 0.624 / 0.214 |
| Act 2 boss | 2,875 | 0.668 / 0.678 / 0.064 | 0.618 / 0.626 / 0.111 | 0.630 / 0.637 / 0.091 | 0.576 / 0.604 / 0.203 | 0.395 / 0.469 / 0.468 |

**V against the search's settled and soft root values** (HP units, raw: the search's level is low by the calibration above, so every judge but v0 sits 25 to 33 HP above; Pearson / Spearman / mean bias):

| cell | positions | v2b | v2 | v2-sp | v1 | v0 |
|---|---|---|---|---|---|---|
| all human turns | 8,301 | 0.784 / 0.851 / +31.2 | 0.793 / 0.843 / +26.9 | 0.784 / 0.843 / +27.9 | 0.705 / 0.750 / +25.7 | 0.668 / 0.682 / +1.7 |
| Act 1 boss | 1,185 | 0.730 / 0.831 / +32.5 | 0.749 / 0.831 / +30.3 | 0.747 / 0.833 / +30.3 | 0.624 / 0.719 / +30.1 | 0.672 / 0.698 / +10.8 |
| Act 2 boss | 2,875 | 0.763 / 0.824 / +31.2 | 0.783 / 0.801 / +24.5 | 0.766 / 0.808 / +27.0 | 0.700 / 0.691 / +14.0 | 0.490 / 0.508 / -19.5 |
| Act 1 elite | 2,098 | 0.749 / 0.795 / +28.4 | 0.743 / 0.779 / +25.6 | 0.726 / 0.769 / +27.1 | 0.598 / 0.617 / +32.5 | 0.676 / 0.686 / +15.2 |
| Act 2 elite | 2,143 | 0.726 / 0.784 / +33.1 | 0.734 / 0.777 / +29.4 | 0.732 / 0.772 / +28.3 | 0.637 / 0.673 / +32.5 | 0.652 / 0.667 / +12.0 |

Against the search's own values v2b is level with v2 and v2-sp (all 0.784 against 0.793 and 0.784; Act 2 boss 0.763 against 0.783 and 0.766: differences not tested; the bias is 2 to 7 HP higher than v2's, 31.2 against 26.9 over all turns): the human rows moved the judge toward what the humans
achieved, not toward the search's settled moves. Against the loop's search as played (160,068 held-out positions, Pearson) v2b reads 0.858 against v2's 0.863, v2-sp's 0.868, v1's 0.823 and v0's 0.850.

**The board, played 2026-11-12** (the per-act rule of D58 addendum 10; the same engine tree, starts, draws and arms as the v2 board: Act 1 boss the fight board's 200 starts x draws 0 and 1, Act 2 boss board A's 100 starts x draws 0 to 3,
400 paired games an arm in each act, K 16, x16 is board A's own games, v2, v1 and v0 the earlier boards' rows on the same games). It played `judge-board.pt`; the driver is
`D:/spireline/referee/backlog/scratch-board-a/v2b/board_v2b.py` (the v2 board's with two changes: the process count follows the brief's rule, 3 while the after-corpus measurement worker or a `lab-train` line ran and 6 otherwise,
re-read every two minutes, and it stops for any foreign QUIET line but the lab's own `lab-*`, `run-teacher`, `gym-loop`, `lab-score`, `w3-blocks` and `judge-corpus`), with `analyze.py`, `analyze2.py`, `compare_prev.py` and `make_summary.py`; `summary.md` there has
the tables; results `D:/spireline/referee/lab/results/fight-gym/2026-11-12-judge-v2b-board/`. A pilot of 16 games (4 starts, both acts, both arms; those games count in the board), then the main phase, Act 2 first: 07:23Z to 14:00Z with no QUIET line of its own, 3 processes to 09:50Z
and 6 after (23,821 s of play in all; this job took about 27 core-hours, 26% of the machine on average; no game capped, no search hit the safety clock).

| per 100, SE clustered by start | x16 | j480 (v2b) | j960 (v2b) |
|---|---|---|---|
| Act 1 boss (400 games an arm) | 48.8 +- 3.3 | 51.7 +- 3.3 | 53.2 +- 3.2 |
| Act 2 boss (400 games an arm) | 19.5 +- 3.2 | 18.8 +- 3.0 | 19.5 +- 3.0 |

Paired differences against x16, points per 100: diff (SE over starts); one-sided 95% lower bound; games only the arm won / only x16 won; one-sided McNemar p that the arm is better.

| | Act 1 | Act 2 | both (800 pairs) |
|---|---|---|---|
| j480 - x16 | +3.0 (2.0); -0.3; 32/20; p 0.063 | -0.8 (2.4); -4.6; 27/30; p 0.702 | +1.1 (1.5); -1.4; 59/50; p 0.222 |
| j960 - x16 | +4.5 (2.0); +1.2; 36/18; p 0.010 | +0.0 (2.2); -3.7; 28/28; p 0.553 | +2.2 (1.5); -0.2; 64/46; p 0.052 |

v2b against v2 on the same games (diff; SE; v2b-only / v2-only; one-sided p that v2b is better): j480 +0.6 pooled (1.0; 34/29; p 0.307; Act 1 -0.8, Act 2 +2.0, p 0.100), j960 +0.8 pooled (1.0; 35/29; p 0.266; Act 1 +0.5, Act 2 +1.0).
Against v1: j480 +3.0 pooled (1.2; 59/35; p 0.009; Act 1 +3.0, Act 2 +3.0), j960 +2.9 (1.3; 55/32; p 0.009); against v0: j480 +4.5 (1.4; p 0.001), j960 +4.8 (1.6; p 0.000). Seconds a boss fight (own clock, indicative: this board ran at 3 then 6 processes,
the v2 board's on a full machine, board A's x16 at 12 processes with the lab paused, so only the ratio of the two judge arms is steady): Act 1 x16 66.5, j480 33.6, j960 71.5; Act 2 79.7, 55.1, 115.9; j960 costs 2.1 times j480 in both acts.

**Verdict by rule 1.3, read per act (j960 if it beats x16 by 3 or more with one-sided McNemar p under 0.05; else j480 if its one-sided 95% lower bound against x16 is above -2; else keep x16): Act 1 boss, j960** (+4.5, p 0.010; j480 +3.0 has its bound at
-0.3, so the rule passes it too, at half of x16's seconds); **Act 2 boss, keep x16** (j960 +0.0, p 0.553; j480 -0.8 with bound -4.6, short of -2); **pooled, for the record, j480** (j960 +2.2 with p 0.052 misses both of its tests by a hair; j480's lower bound -1.4 is above -2, by the
start-clustered SE and by the pair SE alike). That is the v2 board's verdict in all three places (Act 1 j960, Act 2 x16, pooled j480), so **the per-act result replicates**, with the pooled j480 margin wider (-1.4 against -1.9) and j960 closer (+2.2 against +1.5). v2b's board games do not differ detectably
from v2's (+0.6 and +0.8 pooled): the human fights, from 599 to 13,371, moved the judge's reading of human positions, clearly at Act 2 boss, and did not move the board's kill rates by more than the noise of 800 pairs. The Act 2 boss, where an Ascension 10 run is decided, is still not
won by any judge arm (v0 -5.2, v1 -3.8, v2 -2.8, v2b -0.8 for j480; v2b +0.0 for j960), so a switch for the Act 2 boss has no support from this board; a switch for the Act 1 boss does. Nothing is switched: the decision is the orchestrator's.

**Unverified / left:** one training seed (no second seed was trained, so a difference between v2b and v2 of this size, such as the held-out gain at Act 2 boss of +0.026 Pearson, cannot be told from seed noise); the human-block intervals are by fight and not by run (a run's fights share a deck); the seconds are indicative; the board's Act 2 starts are 100 starts
(the exam's act2-boss cell), so the Act 2 verdict rests on 100 start clusters.

**In the lab's teacher from 2026-11-12T15:48Z** (D58 addendum 24): the daily teacher job asks `judge-board.pt` for P(death) at every boss door (`docs/lab/TEACHER_JOB.md`, "The judge"); the door's recalibration is kept per judge, so v2b's starts from the judge's own number and is not fitted on v0's pairs. The player's loader never feeds the source row, so `judge.pt` and `judge-board.pt` read identically (the same value, P(death) and end HP on eight engine-built states).
