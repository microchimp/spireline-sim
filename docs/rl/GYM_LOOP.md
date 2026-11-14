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
| `games/` | generation rows per segment, `segments.jsonl`, `steps.jsonl` (wall, prior path, server and process stats), `clock.jsonl` (every interval the daemon played or waited, by reason: section 2) |
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
candidate reaches it). **The move check** (D51 did it by hand on the 400 stage-1 pairs: outcome 400 of 400, end HP 400 of 400, move hash
398 of 400): `python tools/gym_engine_move_check.py gate/inc-vN-CFG-OLDTAG.jsonl gate/inc-vN-CFG-NEWTAG.jsonl [--list 10]` matches the
incumbent's games on the two tags by (start, j) and prints the matched count, the agreement of outcome, end HP and move hash, the
games whose moves differ though outcome and end HP agree, and the root value's largest and median difference (float noise shows as
about 1e-7). On the D51 pair (tags aaa0d2e4a460 and 1b3d1f01c90b, all 2,000 games) it reads: outcome 1998 of 2000, end HP 1995, move hash 1973.

## 2. The locks (decided)

- **Start a step** (a generation segment, a gate, the start set, the weights pass) only when none of these holds: the loop's STOP
  file, LAB/HALT, LAB/nightly/active, 03:10-03:45Z, LAB/PAUSE, `backlog/main.lock`, another QUIET.lock line under 2 h old (not
  `w3-blocks`, not the loop's own, not a tag in `--quiet-allow`: by default `run-teacher`, `judge-corpus`, `judge-train`, the builders' measurements the loop plays beside; a judgement's `lab-train` line holds it). While a step runs it holds the line `gym-loop <UTC> pid=<Windows PID>`.
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

**Every wait is logged and timed** (2026-11-11, the gate lane; `rl/gym_clock.py`). The loop once lost three quarters of a week to waits nobody
had logged. Now every interval of the daemon's day is a **play** (a step that works: `generate`, `gate`, `human-bar`, `starts`, `weights`,
`prefilter`) or a **wait** with a reason: the reasons of `why_blocked` (`quiet:<tag>`, `main.lock`, `nightly-active`, `night-window`, `pause`,
`halt`), `publish` (the board page's build and push after a promotion) and `yield` (the fixed pause between steps). A wait writes
`wait start: <reason>` to loop.log when it begins and `wait end: <reason> after 14m20s` when it ends (one pair for a wait that goes on, a new
pair when the reason changes; `yield` is timed but not logged line by line). Every interval is a row in `games/clock.jsonl` (`t0`, `t1`,
`kind`, `what`, `s`; split at 00:00Z). The status file (`D:/spireline/logs/gym-loop/status`) carries `waiting_s_today`, `playing_s_today` and
`waits_today` (seconds by reason), refreshed every 30 s by a watch thread, also in the middle of a twenty-minute step. At 00:00Z loop.log gets
one line, `daily 2026-11-12: 71 segments, 14203 fights played; played 17.9 h (generate 15.2 h, gate 2.4 h, ...), waited 4.1 h (quiet:lab-score
2.0 h, yield 1.8 h, ...), between steps or down 2.0 h; 11 gated (stage 1 reject 9, stage 2 reject 2); gpu-denied:lab-nightly x1`; a day finished
while the daemon was down gets its line at the next start. A card refused (`GPU_OK` held by another, so the CPU prior plays) is logged
(`gpu: <step>: the card is held by <tag>; the CPU prior plays`) and counted in the daily line. What the daily line shows as "between steps or
down" is whatever is in neither list: retention and pruning between steps, and the time the daemon was not running.

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
- **Run-teacher starts** (D60 item 5, lane L; `rl/teacher_starts.py`, `--teacher-store`, off unless the daemon is given the store). A fourth source,
  `teacher`: every Act 1 elite, Act 1 boss, Act 2 elite and Act 2 boss (floors up to 33) that a finished run of the run teacher's store
  (`D:/spireline/referee/run-teacher/<generation>/`, generations 14 and 15 by `--teacher-gens`) entered, at the fight's first decision, in the
  start format of the nights' (`starts/teacher-starts.jsonl`, its byte index and lite list, `teacher-meta.json`, the replayed rows in
  `teacher-replays.jsonl`); ids `teacher:<seed>|f<floor>`. The builder replays each run as the teacher played it (the player's pick drawn at every
  screen so its streams run as they did, the stored teacher pick where the screen has a record, the player's own fight search), **on the old
  engine** (the teacher's job sets none; the same runs on core2 diverged in 2 of 3), and checks every screen with a record and every fight start
  against the run's RunRecord (floor, options, HP, deck, the player's pick): at the first difference the run's later fights are not used. A
  start's position is a snapshot document, played by the gym on core2 like any other. Exclusions as the nights' (the board's runs, the exam's
  `exclude_runs.json`; the sampler leaves them out at every draw too). The daemon checks the store between steps (`Daemon.teacher_step`): a build
  when there is no set, or `--teacher-min-new` (25) finished runs have come since the last (each run is replayed once). The mix: `teacher_share`
  per kind in `starts/mix.json` (none by default; the teacher takes its share of a kind's draws first and the human share splits the rest), the
  segment log's `mix` line shows `teacher <n>`. Nothing in the teacher's store is written (its records are read, never loaded through `RunFiles.load`,
  which cuts files). `python rl/teacher_starts.py census --store STORE` counts the starts a store would give without playing.
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
the search's move was worse, the solver's move is the policy target. **Freshness:** each new row adds `--max-epochs` (4 by default; the daemon has run at 8 since 2026-11-08) rows of
budget; with the budget spent the trainer idles and counts the idle time by reason. **Checkpoints** every 30 minutes when it trained,
each scored on the held-out rows (value NLL, policy cross-entropy, top-1 agreement with the teacher), beside the incumbent's network
on the same rows. The incumbent is the daemon's incumbent at the checkpoint (read from `state.json`, not the one the trainer was
started beside: a promotion does not restart it), and `train.json` names it (`base_version`, `base_sha256` of its
`combat_policy.pt`) and the held-out split (`split`, `gym_loop.val_split_id()`); before 2026-11-03 it did not, and the base
figures of c0020 to c0048 were an older version's (the trainer's start-up base) although v2 was the incumbent: a policy CE gap of 0.06 that was never v2's. The held-out rows
are the newest 20,000 held-out rows of the window, so they move a little with each checkpoint: only a candidate figure and an
incumbent figure scored in the same call compare.

## 5. The gate and the board (decided before the first checkpoint)

**Since 2026-11-11 the rule is v3 (`gym-loop-promote-v3`, D57 item 2; the block "Gate rule v3" below). The next paragraph describes v1 and v2,
whose stage 2 survives unchanged as path (a).**

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

**Gate rule v3** (D57 item 2, 2026-11-11; `PROMOTION_RULE`, `decide`, `pool_members`, `promotion_guard`; the statistics in `rl/gym_gate.py`,
the replay in `referee/backlog/gym-gate/replay.py`). Between 2026-10-31 and 11-09 v2 refused 175 candidates at +0.0 to +1.1 points on 2,000
games (SE 0.65): a gain of a point cannot show p < 0.05 on 2,000 pairs, so the loop stood still while its students improved. The owner's
decision: "not worse, and the trend across candidates positive", the lab's judgement still deciding what ships.

- **Pairs and errors.** A pair is (board start, seed j); its difference is the candidate's win minus the incumbent's (a capped game is lost):
  -1, 0 or +1. Standard errors are **start-clustered** (the ten seeds of a start share its difficulty; about 5% above the pair-level error on
  the ledger's games): with C starts, per-start sums S_c of n_c pairs, n pairs, mean m: SE = sqrt(C/(C-1) * sum_c (S_c - m n_c)^2) / n.
  z = 1.645 is the one-sided 95% point.
- **Stage 1 is a screen** (400 pairs of the rotating seed pair, as before): it stops a candidate **clearly worse**, when the one-sided 95%
  *upper* bound of its difference, m + 1.645 SE, is under -2.0 points (about -4.5 points at the 1.5-point SE of 400 pairs). Everything else
  goes to stage 2. *The brief said "lower bound below -2.0"; read literally that stops every candidate under about +0.5 points; on the
  ledger's 166 gated candidates v2's +1.0 floor stopped 132, the screen as built stops 2 and the literal reading would stop 106, most level
  candidates, which is v2's starvation again. `screen.bound` = "lower" is that reading, one constant.*
- **Stage 2 promotes** (2,000 pairs) when (a) **alone**: v2's test unchanged, the exact one-sided McNemar p < 0.05 and the paired L0 not
  lower (`single.min_points` can ask for a floor on top; 0, because the brief's "+1.5 points" would have refused c0019 and c0185, which v2
  promoted at +1.30 and +1.35, p 0.025), or (b) **not worse and the trend positive**: the candidate's lower bound m - 1.645 SE is above -1.0
  point *and* the pooled paired difference of the last five stage-2 candidates of this incumbent, this one included (at least three,
  `pool.min`), is positive at one-sided p < 0.10. Pooled **per start** (2026-11-11, `gym_gate.pooled`): a candidate's difference at a start is the mean
  of its own pairs there (its seeds); a start's one number u_s is the mean of that over the pool's candidates that played the start (a start
  some did not play is the mean over those that did; all of them, in the gate); then the one-sided paired test over starts: P = mean of u_s,
  SE = sd(u_s) / sqrt(starts), z = P / SE, p = 1 - Phi(z). The five candidates are paired against the same incumbent games, so their differences
  share the incumbent's draws (they correlate about 0.5); the spread of their per-start average carries that: five candidates cut the SE by
  about a fifth (0.67 to 0.55 points), not by sqrt 5. Treating them as independent samples (their own SEs combined as sqrt(sum SE_i^2) / 5)
  understates the SE by 1.7 on average (1.2 to 1.9 on the ledger's pools of 2 to 5) and is not what the gate does. **The pool** is the ledger's stage-2 rows with the same
  incumbent (a promotion restarts it) and the **same rule version** (a v1 or v2 row was picked on a +1.0 stage-1 floor and its 2,000 pairs carry
  that selection), whatever the engine (each member's difference is taken on its own engine's games); a member whose games are gone is left
  out. Verdict lines and ledger rows carry the figures (`diff_lb`, `pool_k`, `pool_diff`, `pool_se`, `pool_p`, `pool_prior`, `path`):

      gate c0004: stage 2 promote: cand 908 inc 896 of 2000 (+0.6 points, SE 0.6) p 0.169; lower bound -0.33 points; pool of 3: +0.40 points (SE 0.27), p 0.072; promote by path pool

- **Guards.** At most one promotion in six hours (`guard_h`): a candidate that passes inside the window is recorded as a reject with `guard`
  and `guarded_path` (its games and the pool are untouched, so the next candidate may promote). A promotion's human comparison (the exam's
  human stratum, four draws, `a3-v1`, the CPU prior) plays once, only after the guard (`--human-bar promotion`, the new default; `on`: after every
  stage-2 verdict as before, `off`: never) and goes to the ledger and loop.log.
- **The replay** (the 34 stage-2 verdicts since 2026-10-31, `replay.py`, one process, nothing played). Which would have promoted: c0001 (2026-10-31
  00:11Z), c0019 (10-31 14:30Z) and c0185 (11-09 12:00Z), all by (a): the three v2 promoted, and nothing else. The pooled path never fires on
  that history: the 18 candidates of v2's long stretch (11-01 to 11-09) pooled +0.1 to +0.7 points at SE 0.55 (p 0.11 to 0.42), and the 11
  candidates since v3 pool -0.5 to 0.0. On the 1,600 pairs of each candidate that its stage-1 screen did not see (v1 and v2 sent a candidate on only
  above +1.0 points on those 400, which flatters its 2,000) only c0001 would pass; c0019's +1.30 becomes +1.19 and c0185's +1.35 becomes
  +0.50. **So the v2 stretch's candidates were not measurably improving**; a rule can only promote what the games show. **False promotion**
  (sign flips by start on the ledger's own games, the same sign for every candidate so their correlation is kept): a look promotes with
  probability 0.11 under no difference (path (b) alone 0.09); some promotion in the first five candidates 0.18, over a four-candidate stretch
  0.14, eleven 0.29, eighteen 0.32 (the looks repeat each other: the candidates are nearly one player). Promoting a level candidate costs a restart of the pool
  and nothing else; (b)'s lower bound keeps a worse one out. **Power** (a true gain per candidate on the real variances, promoted within 3 / 5 /
  10 candidates): +0.5 points 0.39 / 0.47 / 0.61; +1.0 points 0.70 / 0.78 / 0.87; +1.5 points 0.91 / 0.95 / 0.98. The pool minimum of three
  (the brief had none): 0.18 against 0.20 false promotion in the first five candidates, 0.78 against 0.80 power at +1.0; a wash, kept because a
  trend is more than one candidate. **The pooling check** (2026-11-11, `replay_pool.py`, the 38 stage-2 verdicts now in the ledger: the 34 above and
  four more rejects of the v3 incumbent): the pooled p of every look is the same to 1e-15 whether the pool is averaged pair by pair then
  clustered by start (as first landed) or per start (as now written); on this board's balanced design (200 starts x 10 seeds) they are one statistic. The
  independent-candidate version would have fired the pooled path on nine looks (c0018, c0019, c0109, c0120, c0164, c0181, c0182, c0183, c0185), seven
  of them candidates the gate refused; the gate's own pooled path fires on c0019 alone (which (a) also promotes).
- **Cost.** Almost every gated candidate now goes to stage 2 (164 of the 166 would have): its 1,600 games are about 9.5 minutes of six processes on
  the fast engine, so a gate is about 12 minutes where a stage-1 reject was 2.6; at one gated candidate an hour (32 since 11-09 12:00Z, 23 a day) the
  loop spends about a fifth of its time on the gate (4% before). The incumbent's 2,000 games are still played once per incumbent and engine. The
  pool's earlier members are read from their games files (a second per member).
- **Statistical power is the lever left.** Pooling five candidates buys a fifth of the SE; 2,000 pairs cannot show a point. Doubling the seeds
  of stage 2 (j up to 19, the incumbent's extra 2,000 played once) halves the variance for about 10 more minutes a candidate, but the board page
  and `promote`'s `fp-vN` rows assume the board's 2,000 games, so it is a change of its own (not made here).

**The gate's cadence** (D60 item 5, lane L; `--gate-every-min`, default 0 = every checkpoint, as before). The trainer writes a checkpoint every
`--checkpoint-min` (30) and the gate used to be offered the newest one each time it was free. With `--gate-every-min 120` a checkpoint is offered
only when it was made 120 minutes or more after the checkpoint the gate last played to a verdict (a reject or a promotion, read from `state.json`
and the checkpoint's own `train.json`, so a restart changes nothing; a skip or a supersede is no anchor); the ones in between are never decided by
the pre-filter and the next gate supersedes them. One checkpoint in four reaches the gate at 30-minute checkpoints; the pre-filter and its six-hour
floor act on what is offered. Measured on the 36 candidates after v3's promotion (2026-11-09 12:01Z to 11-11 07:34Z, `FIGHT-LOOP-L.md`): the trainer
spent 34.6 of 38.3 hours idle on the freshness budget and trained 393 s in all, so a longer gap between gated checkpoints adds no training;
more passes per row (`--max-epochs`) does.

**The plateau alarm** (D57 addendum 2; `gym_gate.plateau_update`, `Daemon.plateau`, `--plateau-alert on|off` default on, `--alerts-dir`
default `LAB/alerts`). A loop with no promotion for 5 days or 25 gated candidates (ledger rows with a verdict of reject or promote since the last
promotion; the first verdict's time when there has been none) is a failure to report the same day, not a wait. The daemon writes
`LAB/alerts/gym-plateau.txt` (first line `ALERT gym-plateau: no promotion for 5.3 days and 31 gated candidates (incumbent v3, last promotion ...)`,
then the last ten verdicts: candidate, stage, verdict, points on that stage's pairs) and a loop.log line `plateau: ...`; it checks at its start,
after every verdict and every ten minutes, rewrites the file as the figures move (no new log line) and removes it, with a log line, at the next
promotion. The lab's status files already print `LAB/alerts/` (`tools/lab/alerts.py`: `ALERT gym-plateau (LAB/alerts)`). **At the first start of
this code the alarm trips at once**: 32 gated candidates since v3's promotion (2026-11-09 12:00Z), the ten latest between +0.05 and +0.75 points.

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

**The human line** (D55 addendum 11, 2026-11-08; `human_bar`, `human_bar_adopt`, `HUMAN_BAR`; daemon flag `--human-bar promotion|on|off`; since 2026-11-11
the default is `promotion`: only a promotion that passed the guard plays it, on the CPU prior (`HUMAN_BAR[prior]`, the prior the weekly readings were taken with);
`on` is the original: after every stage-2 verdict). After a **stage-2** verdict (in mode `on`: promote or reject; never after a stage-1 reject, a stopped or an
incomplete gate) the daemon plays the candidate on the fight exam's human stratum (400 clean human fight starts, four draws, 1,600 games, the exam's own configuration `a3-v1`
whatever the gate's `--player-config` is, `exam/starts-human.jsonl`) with the gate's own processes (`play_tasks` with
`fight_exam:exam_game_steps`: `--procs`, BelowNormal, its QUIET line, the same stop reasons as the gate's play: STOP, the night, PAUSE,
HALT, another line), writes the games to `exam/games-<checkpoint>.jsonl` (resumable: a game with a row is not played again) and reads them
beside the incumbent's (`exam/games-<incumbent>.jsonl`, rows of config `a3-v1`) and the human's own results:

    gate c0190: human bar: cand 55.0 / inc 62.0 / human 79.8 in 100 on 400 starts; cand-inc -7.0 [-11.2, -2.8]; HP lost 41.2 / 38.6 / 33.2

Survival in points per 100 and HP lost (a death costs the whole HP going in, on all three sides) are on the starts both players have
games on; the difference is paired by start on the draws both have, a 95% bootstrap interval over starts resampled within cells. The
reading goes into the verdict dict as `human` and so into `state.json`'s `gated[cand]` and `gate/verdicts.jsonl` (the dict has the
numbers, counts, the engine tags of both sides, `same_engine`, `complete` and the line); the loop log carries the line. **It is
report-only**: `PROMOTION_RULE` and `decide` do not read it and a failure of the play or the reading (recorded as `human: "error: ..."`)
never changes or loses the verdict. A play a stop cuts short is recorded as `human: "stopped"` and the verdict goes on (the games played
stay in the file; `fight_exam.py play --player-dir <checkpoint folder> --stratum human` finishes them, `report --player <name>` reads
them). With no incumbent games the line gives the candidate's side alone and says so; with the incumbent's games on another engine tag
(its human games were played before an engine move) it ends `; engines differ (cand X, inc Y)` and the difference is then not a clean
comparison (to make it one, keep the old `exam/games-vN.jsonl` under another name and play the incumbent's stratum again on the
current engine with `fight_exam.py play --player vN --stratum human`).
On a **promotion** the candidate's human games are copied to `exam/games-<new version>.jsonl` (relabelled, `reused_from` the checkpoint)
so the new incumbent's stratum games are found under its name. Cost, measured: the champion's 1,600 stratum games took 1,928 s at six
processes (the daytime score running beside them), so a stage-2 candidate's gate gets about 32 minutes longer at `--procs 6` (a stage-2
gate was 20 to 25 minutes per arm), plus about a second for the reading; stage-1 rejects, most gates, add nothing. The lab's keep test
prints the same kind of line without playing (docs/rl/KEEP_TEST_V2.md).

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
  the candidate against the champion on the stratum. No calibration flag is applied to it (the human is not a draw). **The full exam is
  run by hand** (no lab job calls it): the stratum adds its 1,600 games to a full `play` (about 25 minutes at the loop's speed) and
  its half A (about 800 games) to `compare`. **The gate calls the stratum alone** (2026-11-08, D55 addendum 11): after a stage-2 verdict
  the daemon plays the candidate on it and prints the human line, report-only (section 5, "The human line"). `play` takes
  `--player-dir DIR` (a checkpoint folder; its games go to `games-<folder name>.jsonl`) as well as `--player NAME-OR-FOLDER`.
  `--allow-quiet TAG` lets a run ignore a named QUIET.lock tag (the lab's daytime score). The lab's keep test v2 prints one report-only line
  naming the judged player's fight network and its latest reading on the stratum (docs/rl/KEEP_TEST_V2.md); `tools/gym_engine_move_check.py`
  compares two games files of one player on two engines (outcome, end HP and move hash per start).
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
