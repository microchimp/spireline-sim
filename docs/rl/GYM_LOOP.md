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

- **Start mix** (gym-build P2, step 1). Every Act 1 and Act 2 fight (hallway, elite, boss; not events) that 50 runs of each night's
  emulator arm entered (2026-10-26 g5, 10-27 g7, 10-29 g7, 10-30 g9), replayed exactly with the night's agent
  (`fight_gym._replay_task`), at the fight's first decision. Runs behind M2's starts (the board's 200 Act-1 bosses and their runs)
  are excluded, so no board deck is trained on. A tenth of run seeds is the trainer's held-out set.
- **Weights.** Per kind, the fights a run enters (from every night's runs) times the mean L0 a fight loses (a death loses all of
  it), the loss from 50 training starts per kind x 2 games at a3-v1 with the incumbent (`weights.json`). A segment draws kinds by
  these weights, then a start uniformly in the kind, then the start's next unplayed game j (seed namespace `L`, so no generation
  game is a board game).
- **Teacher.** The newest checkpoint (the incumbent before the first) under x16, in P1's lockstep processes (`--k` 8 a process,
  six processes, the prior on the card through P1's server when allowed). The x16 rate is P1's: about 55 s of a process's own time
  per Act-1 boss game.
- **Labels** (`gen_game_steps`, `Recorder`). Per decision: the tokens the root prior reads (the probe copy, `encode_position`),
  the root's visits and mean values per action (`gen_loop.root_targets`, the self-play layout), the legal mask, the move played,
  and the exact solver's grade at horizon 0 under `--exact-wall-s` (0.25 s; `gym_lockstep.Grader`, expected solve only). The
  fight's outcome is the value label (end HP; 0 for a death); a fight a guard stopped has none. The recorder reads the search's
  result through a wrapper around `live_search.search_steps` installed in the generation's own processes only; it never changes a
  move (a test plays the same fights recorded, unrecorded and in the sequential gym: same moves).
- **Buffer.** Shards are kept newest first while within 3,000,000 rows, 21 days and 20 GB (`retain`); older ones are deleted.

## 4. Training

`rl/gym_loop_train.py`: `combat_net.CombatNet` with the incumbent's shape, from the incumbent's weights (later its own `state.pt`),
AdamW at 1e-4, batches of 256 from the newest 300,000 rows (train split only). Losses as `train_combat.train`: soft cross-entropy on
the end-HP bucket (value) and on the visit distribution over legal actions (policy); where the exact solver's value is exact and
the search's move was worse, the solver's move is the policy target. **Freshness:** each new row adds `--max-epochs` (4) rows of
budget; with the budget spent the trainer idles and counts the idle time by reason. **Checkpoints** every 30 minutes when it trained,
each scored on the held-out rows (value NLL, policy cross-entropy, top-1 agreement with the teacher), beside the incumbent's network
on the same rows.

## 5. The gate and the board (decided before the first checkpoint)

`PROMOTION_RULE` (`gym-loop-promote-v1`): the board's 200 Act-1 boss starts, game j with the gym's seeds (namespace `g`), the
player (`--player-config judge-v0 --judge D:/spireline/referee/agent/judge/v0/judge.pt`, the defaults since the judge landed (D41): JUDGE_V0's search with the judge v0 at its leaves, the judge on its own server on the card beside the prior's; the incumbent's games are kept per config and judge; measured: the gate's judge-v0 games of version 0 are the board's judge-v0 games move for move),
the candidate and the incumbent on the same games. A capped game is lost. Stage 1, j = 0..1 (400 pairs): go on only if the candidate
wins at least 1.0 point more. Stage 2, j = 0..9 (2,000 pairs, the board's games): promote only if the exact one-sided McNemar test
on the discordant pairs gives p < 0.05 and the paired mean L0 difference is not negative. Only the newest ungated checkpoint is
scored; older ones are marked superseded. A promoted checkpoint becomes `players/vN`, the incumbent, and its board row "Fight player,
version N": `board_commit` rebuilds `docs/paper/fight-board.html` in its own Spireline worktree (`D:/spireline/wt-board`) with the
standing arms and every version, keeps the page's log lines and adds one, commits only the page and pushes to main. The page's own arms are kept in its order (`page_arms`), each with its games file (`BOARD_GAMES`, or `board/games.json` in the loop's folder for an arm added later); an arm with no known games stops the step, so a rebuild never drops a row.

The power, estimated: with about a quarter of pairs discordant, a +3-point candidate passes stage 1 four times in five and stage 2
(SE about 1.1 points on 2,000 pairs) about 85% of the time; a candidate with no gain reaches stage 2 a third of the time and is
promoted 5% of the time. A gate costs about 0.4 machine-hours (stage 1) plus 1.6 (stage 2) at a3-v1; the incumbent's 2,000 games are
played once per incumbent.

## 6. What is measured

- `games/segments.jsonl`: fights and samples per segment, by kind; fights per hour is their sum over the wall time.
- `gpu.jsonl`: the card's busy share per process from Windows' GPU Engine counters, split into the loop's inference (its server),
  its training (the trainer), the lab's Python processes and the rest.
- `gate/verdicts.jsonl` and `trainer/checkpoints.jsonl`: each checkpoint's scores.
