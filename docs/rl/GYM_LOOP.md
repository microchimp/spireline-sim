# The always-on fight-gym loop (D40, gym build item L1, 2026-10-30)

`rl/gym_loop.py` (the daemon, generation, the buffer, the gate, the board) and `rl/gym_loop_train.py` (the trainer, in the GPU
venv). Status words as in GPU_INFERENCE.md: **measured**, **estimated**, **decided**.

## 1. What runs

One daemon process, BelowNormal, holding a named mutex (one loop at a time) and a Job Object with kill-on-close, so every process it
starts (P1's lockstep processes, the prior server, the trainer) ends with it however it ends. Its folder (default
`D:/spireline/referee/gym-loop`, beside LAB, not in it, so the lab's disk guard does not count it):

| path | what |
|---|---|
| `starts/` | the training start set (`starts.jsonl` with documents, its byte index, `starts.lite.json`), `meta.json`, `weights.json` |
| `buffer/shards/` | the replay buffer, one compressed shard per generation segment; `buffer/index.jsonl` lists them |
| `buffer/incoming/` | one file per finished generation fight until its segment is compacted |
| `checkpoints/cNNNN/` | the trainer's checkpoints: `model.json`, `combat_policy.pt`, `combat_net.pt`, `train.json` (a player loads it as an agent) |
| `players/vN/` | promoted versions (copied out of `checkpoints/`, so pruning never removes one) |
| `gate/` | the gate's games per candidate and the incumbent's (per incumbent, config and engine), `verdicts.jsonl` |
| `games/` | generation rows per segment, `segments.jsonl`, `steps.jsonl` (wall, prior path, server and process stats) |
| `board/` | each promoted version's 2,000 board games as `fp-vN` rows |
| `trainer/` | the trainer's `state.pt` (weights, optimiser, counters), `status.json`, `checkpoints.jsonl`, its log |
| `state.json`, `loop.log`, `gpu.jsonl` | the promotion state and seeds, the log, the card's split once a minute |

Its PID and phase are in `D:/spireline/logs/gym-loop/status`. Stop it with `gym_loop.py stop` (its STOP file; after five minutes its
own PID). It is resumable: a restart compacts a killed segment's finished fights, reloads the state, the trainer reloads `state.pt`,
and the gate's games on disk are kept.

**Moving it to a newer engine** (first done 2026-11-02, D44 step E: tag 438479d64595 to aaa0d2e4a460): stop it at a verdict line,
check out the new fork main in its worktree, publish the worktree's own `out/` and Release EmuDump (restore, NativeAOT publish,
EmuDump build), check `gym_loop.engine_tag()` changed, start it again. No code change: the gate's files are per engine tag, so
the incumbent's games are played once more on the new engine at the next gate (stage 1's 400, then stage 2's 1,600 when a
candidate reaches it).

## 2. The locks (decided)

- **Start a step** (a generation segment, a gate, the start set, the weights pass) only when none of these holds: the loop's STOP
  file, LAB/HALT, LAB/nightly/active, 03:10-03:45Z, LAB/PAUSE, `backlog/main.lock`, another QUIET.lock line under 2 h old (not
  `w3-blocks`, not the loop's own). While a step runs it holds the line `gym-loop <UTC> pid=<Windows PID>`.
- **Stop a running step at once** on all of these but main.lock: a landing lets a running step finish, as the lab's guards let a
  running job (the loop runs from its own worktree and touches no git state). A stopped step loses the fights in flight only.
- **Between steps** the daemon drops its line and waits `--yield-s` (150 s): the waiting gates poll QUIET.lock every 60 s, so a
  waiting gate takes the lock there and the daemon waits until it is done. A step is about `--segment-min` (20) minutes, so a
  waiting gate waits at most that long.
- **GPU_OK.** The prior server and the trainer use the card only while GPU_OK is absent or its first line is one of the loop's
  (`gym-loop`, `gym-loop-train`); each holds its line only while it uses the card. Another first line (a gate, the night's
  `lab-nightly`) takes the card: the server is not started (the CPU prior plays) and the trainer leaves the card.
- **The night is training time.** Generation stops for the night. The trainer goes on: on the card when GPU_OK allows it, else on
  the CPU on one thread while the night runs or is due; it waits while LAB/PAUSE or HALT exists or another QUIET.lock line (not the
  night's) stands.

## 3. Generation

- **Start mix** (gym-build P2, step 1). Every Act 1 and Act 2 fight (hallway, elite, boss; not events) that `--runs-per-night`
  runs (250) of each finished night's emulator arm entered, replayed exactly with the night's agent (`fight_gym._replay_task`), at
  the fight's first decision. The nights are found, not listed (`nights_available`: every `LAB/nightly/<date or date-and-time>/<agent>-emu` with its
  summary.json, the newest `--max-nights`, 14); each night's runs are its own seeded draw (`night_seed`, by the night's label), a
  run once replayed is kept in `replays-<date>-<agent>.jsonl`, and only the picked runs of the nights taken are assembled (streamed;
  `assemble_starts`). Until 2026-11-02 the set was 50 runs of four nights (10-26 g5, 10-27 g7, 10-29 g7, 10-30 g9), 1,478 starts.
  Runs behind M2's starts (the board's 200 Act-1 bosses and their runs) are excluded, so no board deck is trained on. A tenth of
  run seeds is the trainer's held-out set.
- **The set follows the nights.** Between steps the daemon compares `meta.json`'s `spec` (the nights and runs a night it was built
  from) with the nights there are now (`starts_stale`): a night finished since, the window moving on, or another `--runs-per-night`
  rebuilds the set, which replays only the new runs (a new night: about 250 runs, ten minutes of six processes). The buffer, the
  checkpoints, the gate's games and the promotion state are not touched. An unreadable night is skipped and recorded in the meta.
- **Weights.** Per kind, the fights a run enters (from every night's runs) times the mean L0 a fight loses (a death loses all of
  it), the loss from 50 training starts per kind x 2 games at a3-v1 with the incumbent (`weights.json`). They are computed again
  when the set is rebuilt or the incumbent changes (`weights_stale`); the incumbent's games are kept across sets, so a rebuilt set
  plays no new game and a new incumbent plays its own (a kind is topped up from its starts in a fixed order). A segment draws kinds by
  these weights, then a source, then a start (next bullet), then the start's next unplayed game j (seed namespace `L`, so no
  generation game is a board game).
- **Sources and regret** (2026-11-04, `rl/practice_mix.py`; settings in `starts/mix.json`, read every segment). Three sources:
  `night` (the AI runs above), `human` (`starts/human-starts.jsonl`, rebuilt by `rl/human_starts.py` from the four training fifths
  of the human corpus; coverage in [HUMAN_STARTS.md](HUMAN_STARTS.md)) and `preloss` (below). A kind's draw is human with the
  kind's share: 0.5 for Act 2 hallway, elite and boss and for the Act 1 boss, 0.25 for the Act 1 hallway and elite (a kind with
  starts on one side only draws from it). Within a kind and side, a start's probability is `0.9` x the rank weight of its regret
  score + `0.1` x its staleness weight (prioritized level replay: either term alone did worse than uniform). The score is the mean,
  over the start's latest four plays in the buffer, of `max(0, root - played)`: the teacher's visit-weighted root value at the
  fight's first searched decision less the outcome, both in the search's currency (end HP plus ten a potion held; a death is -100);
  the rank weight is `(1 / rank) ** (1 / 1.4)` (equal scores share a rank); the staleness weight is the segments since the start was
  last drawn (`state.json` `last_seg`). A start nobody has played takes the mean score of those played; plays come from the buffer's
  shards through `starts/plays.jsonl` (each shard read once, kept after retention removes it). Never drawn: any start whose id,
  run or seed is in `exam/exclude_runs.json` under the loop folder (the fight exam's file; read at every segment; absent: nothing
  excluded). Each segment's log line carries `mix <kind> <source> <n> ...` and the quartiles of the drawn starts' scores, and
  `games/segments.jsonl` the same as data.
- **Human-shortfall regret** (2026-11-08, D55 addendum 10; `regret.mode` in `starts/mix.json`, default `shortfall`). The corpus's per-floor
  record gives every human start the human's own result (`starts/human-outcomes.json`, `rl/human_starts.py outcomes`; fields and coverage
  in [HUMAN_STARTS.md](HUMAN_STARTS.md)). Mode `human-shortfall`: for a start with an entry the score is the mean over its latest
  `n_last` plays of `max(0, human_l0 - played)`, both in the search's currency (end HP plus ten a held potion, a death -100); the
  teacher's root is not needed. Mode `max`: the larger of that and the root shortfall, play by play. Starts without an entry keep the
  root rule; a mode the code does not know scores as `shortfall`. `sample_tasks` reads the sidecar only when the mode asks for it (a long-
  lived process reads it again only when the file changes). **The daemon imports `practice_mix` and `gym_loop` once, at its start: a running
  daemon has the mode only if it was started from a tree that contains it; writing `mix.json` for a daemon without it changes nothing.**
  `tools/human_bar.py` reads the loop's plays against the sidecar and writes `referee/backlog/human-bar/diagnosis.{json,md}`: the
  human-bar table by act, kind and enemy on the human starts played so far, the starts the human survived where the player's latest plays
  all died, and those where the player lost 15 HP or more above the human (the loop draws by regret, so its table reads lower than an
  unselected sample; the exam's human stratum is the unselected reading).
- **Pre-loss starts.** When a practice fight is lost (not capped), the position three decisions before its last decision is exported
  with `native.get_state_json` (the public state; never the draw order) and kept as a start of the same kind, run and split, with
  the loss as its first score (`score0`: the teacher's value there against the death), until it has been played. The copy can
  export any position a decision is asked at; it refuses only a step stopped on a screen (`ExportRefused`), and then no start is
  made (counted as `refused` in `segments.jsonl`'s `preloss`). A fight shorter than five decisions, and a fight that began at a
  pre-loss start, make none. Files are written one a start under `starts/preloss-new/` by the game processes and merged after the
  segment into `starts/preloss-starts.jsonl` (index and lite list beside it); the newest 4,000 are kept. `--no-preloss` turns it off.
- **Teacher.** The newest checkpoint (the incumbent before the first) under x16, in P1's lockstep processes (`--k` 2 a process,
  six processes, the prior on the card through P1's server when allowed). The x16 rate is P1's: about 55 s of a process's own time
  per Act-1 boss game.
- **Labels** (`gen_game_steps`, `Recorder`). Per decision: the tokens the root prior reads (the probe copy, `encode_position`),
  the root's visits and mean values per action (`gen_loop.root_targets`, the self-play layout), the legal mask, the move played,
  and the exact solver's grade at horizon 0 under `--exact-wall-s` (0.25 s; `gym_lockstep.Grader`, expected solve only). The
  fight's outcome is the value label (end HP; 0 for a death); a fight a guard stopped has none. The recorder reads the search's
  result through a wrapper around `live_search.search_steps` installed in the generation's own processes only; it never changes a
  move (a test plays the same fights recorded, unrecorded and in the sequential gym: same moves).
- **Cut segments** (measured on the first evening: other workers' short QUIET.lock holds stopped four segments in 40 minutes). A stop loses the fights in flight, and the long ones are the likelier to be in flight, so the fights a cut segment finished lean towards quick deaths: their value labels are dropped (the policy targets stay), and K is 2 so a stop loses few fights (at 8, an x16 boss fight took about ten minutes of wall time).
- **Buffer.** Shards are kept newest first while within 3,000,000 rows, 21 days and 20 GB (`retain`); older ones are deleted.

## 4. Training

`rl/gym_loop_train.py`: `combat_net.CombatNet` with the incumbent's shape, from the incumbent's weights (later its own `state.pt`),
AdamW at 1e-4, batches of 256 from the newest 300,000 rows (train split only). Losses as `train_combat.train`: soft cross-entropy on
the end-HP bucket (value) and on the visit distribution over legal actions (policy); where the exact solver's value is exact and
the search's move was worse, the solver's move is the policy target. **Freshness:** each new row adds `--max-epochs` (4) rows of
budget; with the budget spent the trainer idles and counts the idle time by reason. **Checkpoints** every 30 minutes when it trained,
each scored on the held-out rows (value NLL, policy cross-entropy, top-1 agreement with the teacher), beside the incumbent's network
on the same rows. The incumbent is the daemon's incumbent at the checkpoint (read from `state.json`, not the one the trainer was
started beside: a promotion does not restart it), and `train.json` names it (`base_version`, `base_sha256` of its
`combat_policy.pt`) and the held-out split (`split`, `gym_loop.val_split_id()`); before 2026-11-03 it did not, and the base
figures of c0020 to c0048 were an older version's (the trainer's start-up base) although v2 was the incumbent: a policy CE gap of 0.06 that was never v2's. The held-out rows
are the newest 20,000 held-out rows of the window, so they move a little with each checkpoint: only a candidate figure and an
incumbent figure scored in the same call compare.

## 5. The gate and the board (decided before the first checkpoint)

`PROMOTION_RULE` (`gym-loop-promote-v2`: v1 was fixed before the first checkpoint, v2 on 2026-11-01, D43): the board's 200 Act-1 boss starts, game j with the gym's seeds (namespace `g`), the
player (`--player-config judge-v0 --judge D:/spireline/referee/agent/judge/v0/judge.pt`, the defaults since the judge landed (D41): JUDGE_V0's search with the judge v0 at its leaves, the judge on its own server on the card beside the prior's; the incumbent's games are kept per config and judge; measured: the gate's judge-v0 games of version 0 are the board's judge-v0 games move for move),
the candidate and the incumbent on the same games. A capped game is lost. Stage 1, one pair of seed indices (400 pairs): go on only if the candidate
wins at least 1.0 point more. v1 always played j = 0..1, the pair each incumbent had been picked on, so the incumbent's wins there
were a lucky overcount (version 2: 196 of 400 there, 746 of its other 1,600) and an equal candidate needed a true +3.4 points. v2
plays a pair the incumbent was not picked on: `state.json`'s `gate_pair` (0 is j = 0..1, 1 is j = 2..3, up to 4), moved on by one
(wrapping) at each promotion. The incumbent's games on it are copied from its promotion gate's file (`gate/cNNNN-...jsonl`, rows
marked `reused_from`) into its kept `inc-vN-...` file, or played once when that file lacks them. A state from before the change
starts on pair 1. Verdict lines carry `rule`, `stage1_pair` and `stage1_js`. Stage 2 (unchanged), j = 0..9 (2,000 pairs, the board's games): promote only if the exact one-sided McNemar test
on the discordant pairs gives p < 0.05 and the paired mean L0 difference is not negative. Only the newest ungated checkpoint is
scored; older ones are marked superseded. A promoted checkpoint becomes `players/vN`, the incumbent, and its board row "Fight player,
version N": `board_commit` rebuilds `docs/paper/fight-board.html` in its own Spireline worktree (`D:/spireline/wt-board`) with the
standing arms and every version, keeps the page's log lines and adds one, commits only the page and pushes to main. The page's own arms are kept in its order (`page_arms`), each with its games file (`BOARD_GAMES`, or `board/games.json` in the loop's folder for an arm added later); an arm with no known games stops the step, so a rebuild never drops a row.

**The pre-filter** (D49 item 3, 2026-11-03; `PREFILTER`, `prefilter_pick`; `--prefilter best|incumbent|off`, `--prefilter-margin`,
`--prefilter-floor-h`). It decides who enters the gate; `PROMOTION_RULE` above is untouched. Each checkpoint without a verdict is
decided once, from its `train.json`: the held-out policy cross-entropy (the policy is what the gate's player reads) against the
incumbent's on the same rows; the gap is the incumbent's minus the checkpoint's. It is gated if the gap exceeds the margin (0)
and, in mode `best` (the default), the gap of every checkpoint already gated and rejected against this incumbent; otherwise it
is marked `skipped` in `state.json`'s `gated`, put in the window, and the log says `prefilter: cNNNN skipped: held-out policy CE
x against vN's y on the same R rows (gap g): why`. **Floor:** when six hours have passed since the last gate's verdict, the
skipped checkpoint with the best gap in the window is gated, so the board keeps moving and the rule can be checked against
what the gate says. If the figures do not name this incumbent and split (a checkpoint from before they did, or an incumbent
promoted since), the incumbent is scored once with the checkpoint on the current held-out rows (`gym_loop_train.py score`, the
CPU, one thread, about 80 s) and stored as the checkpoint's `prefilter.json`; if even that fails the checkpoint goes to the gate.
A new incumbent starts a new record (`state.json`'s `prefilter`: last gate, window, figures, best rejected gap). Why mode
`best`: measured on the day it was written, all 78 checkpoints gated against v2 had a held-out policy CE below v2's (gap 0.003 to
0.013 from c0049 on) and none was promoted, so a bare 'below the incumbent's' skips nothing; replayed on c0049 to c0100 the rule
gates 22 of 53 (the floor never fires while the gap keeps setting records).

The power, estimated: with about a quarter of pairs discordant, a +3-point candidate passes stage 1 four times in five and stage 2
(SE about 1.1 points on 2,000 pairs) about 85% of the time; a candidate with no gain reaches stage 2 a third of the time and is
promoted 5% of the time. A gate costs about 0.4 machine-hours (stage 1) plus 1.6 (stage 2) at a3-v1; the incumbent's 2,000 games are
played once per incumbent.

## 6. What is measured

- `games/segments.jsonl`: fights and samples per segment, by kind; fights per hour is their sum over the wall time.
- `gpu.jsonl`: the card's busy share per process from Windows' GPU Engine counters, split into the loop's inference (its server),
  its training (the trainer), the lab's Python processes and the rest.
- `gate/verdicts.jsonl` and `trainer/checkpoints.jsonl`: each checkpoint's scores.

## 7. The fight exam (D55, item 3.1 of the 2026-11-04 train-and-test memo): a second ruler beside the gate

`rl/fight_exam.py` builds and plays a fixed, stratified, held-out set of fight starts and prints what a candidate does to the
champion's results on it. It is a parallel instrument: it never reads or writes the gate's games, it never changes
`PROMOTION_RULE`, and its `compare` prints figures and decides nothing. The module's docstring is the specification; this is the
short version and the working notes.

- **The set.** About 1,100 starts from the gym's start records (the nights' replays): act (1, 2) by kind (hallway, elite, boss),
  a boss a cell of its own, up to 100 a cell, drawn by a seeded hash, clean starts only (the run is in the trainer's held-out
  split, or the generation never played the start: `state.json`'s `next_j`). The board's 200 starts and every run behind them are
  out. Cells the clean pool cannot fill are listed short in `exam/meta.json` and in every report. `--extra-starts PATH` takes a
  second file in the same record schema as a stratum (the human starts), up to 100 a cell, and puts its runs on the exclusion
  list. The exam keeps its own copy of every start's document (`exam/starts.jsonl`, the gym's format and byte index): the gym
  rebuilds its start set with each night and drops the old nights, the exam does not move.
- **The human stratum** (2026-11-08, D55 addendum 10). `fight_exam.py add-human` adds up to 50 clean human starts a cell (act by kind, a
  boss per boss; 400 in all: equal quotas of 30 or 31 over the 13 cells, the remainder to cells in a fixed hash order, at most one start
  a run, the same seeded hash as the rest) as a **second file** in the exam folder (`starts-human.jsonl` with its index and lite list,
  `human-meta.json`); *clean* is the run's never having been played by the loop (`state.json` `next_j` and `starts/plays.jsonl`). Each
  start carries the human's result from the sidecar (survived, HP left, HP lost, potions, turns, value). **The AI exam is not rewritten**:
  `starts.*`, `meta.json`, the baseline, the halves and every reading of the champion on them stay byte for byte (the files are hashed
  before and after; the hashes are in `human-meta.json`; a test and `report` before and after prove it). The stratum's runs are merged into
  `exclude_runs.json` (the old file kept once as `exclude_runs.json.pre-human.bak`), so the practice drops them from its next segment.
  `play` and `compare` play the stratum with the rest (`--stratum all|ai|human`; its rows go in the same games file with
  `stratum: "human"`; every reading of the AI exam leaves them out). `report` and `compare` print the exam's own lines first, unchanged,
  then the human line: survival and HP lost of the human against the player's (the mean over its four draws) by act and kind and by boss,
  the paired difference (player minus human) with a 95% bootstrap interval over starts resampled within cells, and counts; `compare` adds
  the candidate against the champion on the stratum. No calibration flag is applied to it (the human is not a draw). **The exam is run by
  hand** (no lab job and no gate calls it): the stratum adds its 1,600 games to a full `play` (about 25 minutes at the loop's speed) and
  its half A (about 800 games) to `compare`. `--allow-quiet TAG` lets a run ignore a named QUIET.lock tag (the lab's daytime score).
- **First reading** (2026-11-08, champion v2 at a3-v1, four draws a start, 400 starts, 1,600 games in 1,928 s of wall time with the lab's
  daytime score running beside it). Survival, player against human on the same starts: Act 1 hallway 100.0 / 100.0, elite 77.4 / 80.6
  (-3.2 points, interval -14.5 to +8.1), boss 50.1 / 71.4 (-21.2, -26.9 to -15.8); Act 2 hallway 98.4 / 100.0, elite 71.0 / 77.4
  (-6.5, -17.7 to +4.8), boss 52.5 / 83.5 (-31.0, -38.7 to -22.8); all 62.0 / 79.8 (-17.8, -21.3 to -14.4). HP lost (a death costs the
  whole HP going in, both sides): Act 1 boss 44.0 / 39.2, Act 2 boss 49.5 / 40.2, all 38.6 / 33.2. Every boss is under the human's;
  the widest gaps are Kaiser Crab (44.2 / 80.0), the Insatiable (52.4 / 87.1) and Soul Fysh (53.3 / 83.3).
- **For the practice sampler.** `exam/exclude_runs.json` lists `run_seeds` (every run behind an exam start) and `start_ids`. The
  daemon's sampler (`sample_tasks`) picks any start of `starts.lite.json` and does not read it yet: **until it skips them the
  practice can play exam starts, and every exam start it plays is no longer clean** (the teacher's moves at that position become
  training data). Skipping the start ids is the minimum; skipping the runs is the board's own rule (here that is about 38 in 100 of
  the runs of the nine nights, because a run gives one exam start of each kind it reached).
- **Draws.** Four a start (`DRAWS`), game j in the seed namespace `x` (the gate's is `g`, generation's `L`, the weights' `W`);
  seeds never depend on the player, so a candidate plays the champion's shuffles and per-decision search seeds.
- **Halves.** `half_of(id)`: sha256 of `fight-exam-half:` and the id, first 32 bits below 2^31 is A. The gate side reads half A
  (`compare` plays half A only); half B is the orchestrator's weekly read (`play --half B`).
- **Scores.** Per game: win (capped is lost), L0, HP lost, potions spent, turns, enemy HP removed, energy spent on cards, block
  held at the end of a turn above the attack the intents announce (wasted block). The last three come from an observer that reads
  the position before each decision (`Observer`; the moves are the gym's, a test pins it). Per start: the mean over draws.
- **Baseline.** `baseline` asks the judge (`agent/judge/v0/judge.pt`, on the CPU) for P(win) and L0 at the first decision of every
  start and draw: a property of the start and the shuffles. `compare` prints the paired win difference raw and adjusted (a
  regression of the per-start differences on P(win) and P(win)(1 - P(win)), moved to the exam's mean baseline), each with a
  bootstrap interval over starts resampled within cells. With common draws the start's difficulty already cancels in a paired
  difference, so the adjustment mostly corrects half A's baseline imbalance and removes the part of the differences that
  difficulty explains; it is not AIVAT (chance after the first decision is not corrected).
- **Calibration flags.** Starts the champion won on all four draws or none are kept and weighted zero, listed in every report. A
  flag made from the champion's draws and used on the same draws flatters a candidate when difficulties are skewed (the cross-fit
  test shows it), so `compare` also prints the win difference with flags from two draws and the difference on the other two.
- **Run it.** `select` (pauses the loop for about a minute), then `play --player v2` (the calibration, about 4,400 games, the
  champion's `a3-v1` games under the loop's lockstep processes and prior server), `report --write-calibration`, later
  `compare c0150`. A run takes the QUIET.lock line `fight-exam` (refreshed every ten minutes), starts only with no other tag but
  the loop's own and `w3-blocks`, never while `LAB/nightly/active` exists or in the night window, stops at once on any of them and at
  `--until`, and is resumable (a game with a row is never played twice). Games go to `exam/games-<player>.jsonl`.
