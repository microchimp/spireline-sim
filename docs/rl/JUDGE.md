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
`referee/agent/judge/v2/` (the corpus-trained judge, **not trained yet: see Status**), `v2-sp/` (the interim, self-play half of the corpus only).
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

**Status (2026-11-11 00:30Z).** The corpus's human build (17,001 fights, `judge_corpus.py run`) is stalled: its workers pause while another lane's
`run-teacher` QUIET line stands (the run-self-play lane's measurement, from 21:34Z; the line is also what holds any heavy job of this lane by the
brief's rule), so the manifest still says `usable_for_training: false` and v2 on the full corpus has not been trained, and the board not played.
v2-sp (`referee/agent/judge/v2-sp/judge.pt`, sha256 41429179...) is trained and measured, and board-ready.

**To train v2 the moment the manifest says `usable_for_training: true`** (GPU venv; about 10 to 15 minutes; leaves the gym trainer running):

    D:/spireline/venv-gpu/Scripts/python.exe -I rl/judge_train.py train --version judge-v2 \
        --init D:/spireline/referee/agent/gen7/candidate-r1/combat_net.pt --out D:/spireline/referee/agent/judge/v2/judge.pt \
        --compare D:/spireline/referee/agent/judge/v0/judge.pt D:/spireline/referee/agent/judge/v1/judge.pt

It writes `judge.pt` (the corpus's manifest sha256 and counts, the weights, the loss settings and the search calibration inside), `judge.report.json`
and `judge.eval.md` (the held-out tables of the three judges). `eval` scores any checkpoints again (`judge_train.py eval --judge A.pt --judge B.pt`);
`summary` counts what the trainer would read. A retrain as v2b when the manifest says `complete: true` is the same command with another `--out`.

**The board** (rule 1.3 of the training design, board A's starts and draws): `D:/spireline/referee/backlog/scratch-board-a/v2/board_v2.py` (the v1
board's driver on the same engine tree, so v2's games pair with board A's x16 and v0 rows and with v1's; no timing sample; `JUDGE_BOARD_PATH`
names the checkpoint, `JUDGE_BOARD_OUT` the results folder, `JUDGE_BOARD_HOLD=1` takes a `judge-train` QUIET line, none by default so as not to pause
the corpus build) then `analyze.py`, `analyze2.py`, `compare_prev.py` (v2 against v1 and v0) and `make_summary.py`. A pilot of eight games with a
self-play checkpoint ran end to end (394 s at six processes beside training and the corpus builder).
