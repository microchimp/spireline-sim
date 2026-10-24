# tools/lab: the fix lab's runner

Spends idle (or, for a week away, all of the) PC time on the work that makes backlog fixes cheap (aimed real-game fights,
whole runs, shrinking, grouping, fix kits). No AI calls, no network. The shared design and the other parts' CLIs are in
`referee/RUNBOOK.md`, section "Fix lab". This folder is the runner and the Windows task.

| File | What |
|---|---|
| `runner.py` | CLI, lab folder, job records, the main loop, supervising child processes |
| `plans.py` | the jobs as data: steps and symbolic commands (`{tree}`, `{lab}`, `{job}` ...), one plan per strategy |
| `jobs.py` | the agent-era job kinds (selfplay, train, score-real, bench-env, bench-emu): defaults, `LAB/jobs.json`, markers |
| `mix.py` | the strategy mix (`LAB/mix.json`), the scheduler that follows it, the yield record (`LAB/yield.tsv`) |
| `guards.py` | PAUSE / main.lock / user input / load / disk / HALT as one `Verdict` |
| `prune.py` | what may be deleted, in what order, and what never (recordings the corpus already holds go first) |
| `corpus_import.py` | adds a job's exact and divergent fights to the regression corpus (last step of a fights job) |
| `winsys.py` | ctypes: GetLastInputInfo, priority class, Job Objects, GetSystemTimes, mutex |
| `register_task.ps1` | registers "Spireline fix lab" (`-Trigger Idle` or `Logon`, `-WhatIf`, `-Unregister`) |

## CLI

```
python tools/lab/runner.py [--lab D:/spireline/referee/lab] [--tree D:/spireline/fork/wt-lab] [--game D:/spireline/referee/game-lab]
    [--trial MINUTES] [--once presort|fights|runs|probes|mutation|invariants] [--dry-run]
    [--skip PREFIX,...] [--no-guard user,load,disk,pause,lock,halt,parent] [--exit-with-parent] [--always-on]
    [--fights 300] [--batch 50] [--servers 8] [--procs 8] [--mutants 150]
    [--shrink 1000] [--shrink-cap 160] [--shrink-budget 6000] [--shrink-minutes 45] [--trace 1000] [--trace-minutes 20]
    [--runs 40] [--run-batch 10] [--run-procs 4] [--run-stop-act 3]
    [--idle-seconds 60] [--load-pct 50] [--load-window 120] [--lab-gb 30] [--free-gb 50] [--grace 90]
    [--fork <main checkout>] [--branch main] [--game-src D:/spireline/referee/game] [--backlog-dir ...] [--backlog FILE]
    [--codex DATASET] [--decompiled DIR] [--records DIR] [--poll 2] [--wait-poll 10] [--idle-poll 300]
```

Run it from the fork's main checkout (it only reads it; `--fork` defaults to the checkout holding `runner.py`). Jobs run
with `--tree` as cwd. It refuses a `--tree` that is the main checkout, a `--lab` inside the fork, and any path under
`D:/spireline/src`. Exit 0: stopped by a guard (user input, HALT; not with `--always-on`), `--trial` over, `--once` done,
nothing to do with `--once`, or another runner holds this lab; 1: crash; 2: refused arguments.

- `--dry-run`: guard readings, the mix, next job, its plan with every command's skip reason. Creates and changes nothing.
- `--trial M`: work for M minutes, let a running step finish within `--grace` (at most 60 s), exit 0.
- `--once J`: one job of type J (a `fights` needs the tree presorted at main first: `--once presort`), then exit.
- `--skip`: prefixes of step or command names. Steps: `tree restore build driver native game checks corpus groups
  kits corpus-import targets batch-NN runs-NN probes mutation invariants`. Commands: `worktree-add checkout restore-* build-emudump
  build-driver publish-native game-copy codex-s0..3 refset corpus groups kits corpus-import targets scen-NN batch-NN analyze-NN runs-NN
  probes mut-prepare mut-schemata mut-report meta invariants`.
- `--no-guard`: switch guards off for a trial on a busy PC. Never in the task.
- `--always-on`: for the logon task. Keyboard or mouse input and `LAB/HALT` stop the children and make the runner wait
  (status `paused(user)` / `paused(halt)`) instead of exiting 0, since nothing would start it again. PAUSE, main.lock,
  load and disk already waited. `--trial` and signals still end it.
- `--servers 8` game servers per batch (the lab's CPU is free); `--shrink` and `--trace` are per batch and mean "every
  diverging fight"; a job's `--shrink-cap` (game fights one diverging fight may cost), `--shrink-budget` (game fights in
  all), `--shrink-minutes` and `--trace-minutes` are split over its batches, so one job cannot run for hours.
- Game speed (the speed spike, `referee/design/real-game-speed.md`): the games the lab starts (scenarios.py, batch.py and
  serve.py) run with `--fixed-fps 60` on top of the flag settle, fast tweens, `silent` and `still_visuals`, and `runs.py`
  sessions with the same job defaults (3.4x faster than the stock settle on whole runs). Per-action game checksums and
  every recorded step were equal on 50 of 50 fights. The host stays the full one (nodeless is the specs file's choice).

## Jobs

`presort` runs when local `main` differs from `state.last_presorted_sha` (or the tree is not at it): `tree` (worktree add /
`checkout --detach --force`), `restore` (offline, only when `obj/project.assets.json` is missing), `build` (EmuDump),
`driver` (referee driver), `native` (`dotnet publish ... win-x64`), `game` (linkcopy), `checks` (codex walk in 4 shards +
refset score, in parallel; the refset score and the corpus results are copied into the job folder under the roles
`lab_groups` reads), `corpus` (`corpus.py check --no-scan`), `groups`, `kits`. A failed `tree` or `build` fails the job
and blocks everything on that sha (retry after 30 min, doubling); the checks' own exit 1 (a regression) is a result, not a failure.

After a presort, a rotation by value: `fights, fights, probes, fights, mutation, fights, invariants`. A job type that cannot
run (tool or input missing, backing off after two failed jobs in a row) is skipped with a log line and the rotation moves on.

**A fights slot is one strategy** (`mix.py`), picked when the job is made: the available strategy furthest below its share
of the CPU time the fights slots have spent (its `strategy` is in `job.json`'s params and in `runner.log`):

| strategy | job | what it runs | weight |
|---|---|---|---|
| `aimed-bugs` | `fights` | `scenarios.py --mode targets --aim bugs`: targets an open backlog item names | 30 |
| `aimed-gaps` | `fights` | `--mode targets --aim gaps`: what real runs meet and little covers, starter items weigh 1/20 | 25 |
| `choice-edges` | `fights` | `--mode choices`: small decks, relics/cards/potions that open card screens, the driver's `choice_edges` answers (none, one, max, last, unusual) | 10 |
| `whole-runs` | `runs` | `runs.py`: `--runs` 40 seeded runs, `--run-batch` 10 per call, `--run-procs` 4 sessions; a crashed session is logged, the other sessions' runs and the crashed one's finished runs are kept | 10 |
| `dense` | `fights` | `--mode dense` | 10 |
| `under-tested` | `fights` | `--mode coverage` | 10 |
| `random` | `fights` | `batch.py` (random decks, its streaming pool), then `scenarios.py --mode analyze` (judge, shrink, trace, verdicts) | 5 |

`LAB/mix.json` (written with these weights when missing, read at every choice, BOM and NULs tolerated) holds
`{"aimed-bugs": 30, ...}`; 0 or absent never runs; a bad file falls back to the defaults and says so once in the log.
A strategy whose tool, input or driver key is missing on the tree (`--aim`, `choices`, `choice_edges`, `batch.py`, a
backlog for aimed-bugs, ...) is passed over with the reason, and one that fails twice in a row backs off alone.

- A `fights` job: `lab_targets.py` for the aimed ones, then `ceil(--fights/--batch)` batches, one call each (with
  the batch's shrink and trace), so there is a safe point every minute or two. A batch's `summary.txt` is parsed: any
  network connection or a touched real profile writes `LAB/HALT` and stops the lab. The batch's `serve/` folders are
  removed after it. A job made before strategies existed (no `strategy` in its params) resumes as it was planned.
- A `runs` job: `runs.py` batches; each session's report is checked for the network and the real profile (a failure
  halts the lab), a crashed session only gets a note. Its scratch profiles live in the job folder (`runs/scratch`).
  After each batch, `repro-NN` (when the tree has `referee/run_repro.py`): `scenarios.py --mode runs` plays every run
  whose first divergence is in a fight again as a setup fight from the run's save just before it (seed, Neow pick, act,
  `player_load` of deck, relics, potions, HP, gold), then judges, shrinks (each loaded card, relic and potion is an
  item) and traces it as a setup batch `runs/<name>-repro.*`, so the run's mismatch gets a minimal repro in the fight
  groups; `<name>-repro.repro.json` says per run whether the fight diverged as the run did and what a setup cannot
  restore (RNG stream positions, floor and map coordinate, potion slot positions, run history), and the run's symptom
  names it (`repro`).
- `probes`: `referee/probe.py run --only <ids>` for ids that `backlog/packets/grades.json` lists under `probes` for items
  missing a rule. Otherwise skipped (probes.py takes a list of ids, not a queue).
- `mutation`: `mutate prepare/schemata/report` on the biggest raw batch of the newest fights job; `mutation.json` is the reach data.
- `invariants`: `invariants meta` (chooser seeds grow by 4 each pass) and `invariants run` over the recorded fights.

## Yield

`LAB/yield.tsv`, one row when a job finishes (or is closed as superseded), so the next week's mix can be set from data:

```
utc  job  strategy  main  rc  wall_min  cpu_min  fights  mismatches  unbuildable  groups  fix_ready  screen_groups
```

`cpu_min` is the CPU time of everything the runner started for the job (its Job Object), summed over the stretches of a
resumed job; `fights` are the verdicts the job wrote (for whole runs, the runs); `mismatches` are those that diverge;
`unbuildable` are fights the emulator's setup mirror could not build (a mirror gap) or that gave no result (an error),
and runs with no result; `groups` are the root-mechanism groups of `lab_groups.py` with a symptom from this job (read
from the `by_job` count of each group) and `fix_ready` those not host-suspect with a symptom from the job that has a
traced first event and no mirror gap. `screen_groups` are the groups with a symptom from the job that is a card screen the
setup mirror could not answer as the game was answered (what `choice-edges` looks for: a relic that allows up to N picks
treated as exactly N, a card the game picked that the emulator does not offer); they are never fix-ready by that
definition (the setup has a mirror gap), so this column is how that strategy's findings are counted. `NA` where the job's
groups step did not finish. The rows of the other job types carry the job type as strategy. The runner keeps each strategy's CPU seconds in `state.json` (`strategy_cpu`).

## Agent-era job kinds and the priority ladder (`jobs.py`)

Five more job types, one command each, configured as data (defaults in `jobs.DEFAULTS`, overridden per kind in an optional
`LAB/jobs.json`: `enabled`, `cmd` (symbolic, `{python}` `{tree}` `{lab}` `{job}` expand), `cwd`, `timeout_min`, `needs`, and per
kind `share`, `weekly`, `weekly_days`, `every_h`, `procs`). Priority ladder, each tick the highest ready job whose guards pass:
presort (1), `score-real` (2), `train` (3), aimed fights (4), runs/rank (5), `selfplay` (6), the rest and the nightly benches (7).
The fights rotation and `mix.json` still decide within tiers 4, 5 and 7; the new kinds are chosen before it (score-real, train,
selfplay while the loop is under its `share` of the last 7 days' CPU time, due benches) or after it (selfplay as the filler when nothing else can run).

**The `selfplay.share` is judged for the loop, over the last 7 days.** The loop is the kinds that exist only because self-play does:
`selfplay`, `train` (and the promotion it runs) and `score-real` (`jobs.LOOP_KINDS`); the benches, presort and the fights strategies are not
in it. The runner adds up their `cpu_min` and the lab's whole `cpu_min` over the rows of `LAB/yield.tsv` whose `utc` is in the last 7
days (`mix.loop_cpu`); self-play gets the slot ahead of the rotation while loop < share x whole. A window with no row (no file, or all rows
older) means self-play first, and the runner log says so; the log also says the loop's share at each change. The all-time
`strategy_cpu` in `state.json` stays as a statistic and still drives the fights mix; it no longer decides the self-play slot.

| kind | priority | ready when | default | writes |
|---|---|---|---|---|
| `selfplay` | 6 | `enabled` (filler; `share` > 0 puts it ahead of the rotation until the loop has used that share of the last 7 days' CPU, see below) | off; placeholder command | yield: runs, decisions, decisions_s, generation |
| `train` | 3 | `LAB/train/ready-<gen>` exists (oldest first; renamed `done-<gen>` on success); 60 min cap | on, marker-gated | yield: generation, promoted, loss |
| `score-real` | 2 | `LAB/score/pending` exists, or `weekly` and 7 days since `LAB/score/last-ok`; 240 min cap | on, marker-gated; weekly off | yield: runs, wins, mean_floor, crashes, decisions_s, divergences. `LAB/score/active` exists while it runs; fights and runs jobs are not started then (a marker older than 5 h is ignored) |
| `bench-env` | 7 | `enabled` and `every_h` (20) since `LAB/bench-env/last-ok` | off | yield: dec_s, crashes, mem_mb_per_run; readiness.json `bench.env` (next to LAB) |
| `bench-emu` | 7 | same; needs `{tree}/rl/bench_emu.py` | off | yield: steps_s, clones_s, sims_s; readiness.json `bench.emu` |

A command reports through `{job}/result.json` (an object; keys named like the yield columns). All guards (PAUSE, HALT, main.lock,
user input, load, disk, parent) apply to these kinds as to every job, between steps and while the command runs. The runner is
sequential, so a job never runs beside another; `train` therefore does not overlap `selfplay` here. New `yield.tsv` columns come after
`screen_groups`; an older file is widened with NA, and old rows keep their values.

## Lab folder

```
LAB/state.json status.txt runner.log targets.json groups.json summary.md coverage-setup.json coverage-random.json
    mix.json yield.tsv   PAUSE HALT (when present)
LAB/runs/<UTC>-<job>/job.json done.rc logs/<command>.log|.rc pruned.json
    presort:  codex/s0..3.jsonl|.err   (logs/refset.log, logs/corpus.log)
    fights:   fights/<name>.{json,out.json,emu.json,reanchor.json,verdicts.json,summary.txt}, <name>.packs/{index.md,minimal/,traces/}, shrink files
    runs:     runs/<name>.{out.json,emu.json,runs.json,summary.txt}, runs/scratch/
    probes:   probes/     mutation: mut/, mutation.json     invariants: meta.json, invariants.json
```

`job.json`: `{version, job, name, main_sha, created_utc, finished_utc, rc, resumed, note, cpu_secs, wall_secs, params, steps:[{name,
status (pending|running|done|failed|skipped), interrupted, started_utc, finished_utc, note, cmds:{<name>:{status, rc, secs, argv
(symbolic), note}}}], outputs:{role:[path relative to the job folder]}}`. Roles: `codex_walk`, `codex_walk_err`,
`refset_score` (`refset/score.json`, a copy of the stored score), `corpus` (`corpus/results.json`, a copy of the
corpus run's results), `fights` (verdicts), `runs` (`runs.json`), `probes`, `mutation_reach`, `metamorphic`, `invariants`.
`done.rc`: 0 ok, 1 required step failed or every core command failed, 3 superseded.

`PAUSE` and `HALT` are read whoever wrote them: PowerShell's `echo now > PAUSE` writes UTF-16 with a BOM.

## The Windows task

`powershell -ExecutionPolicy Bypass -File tools\lab\register_task.ps1 -WhatIf` prints the definition and changes nothing;
without `-WhatIf` it registers "Spireline fix lab" for the current user (Interactive, Limited, no password): idle trigger,
RunOnlyIfIdle 10 min, StopOnIdleEnd, RestartOnIdle, priority 7, no WakeToRun, not on battery, 12 h limit, MultipleInstances
IgnoreNew; action `pythonw.exe tools\lab\runner.py --lab ... --tree ... --game ... --exit-with-parent` in the fork's main
checkout. `-Unregister` removes it. GetLastInputInfo only sees the logged-on user's input, which is why the task is
Interactive. Task Scheduler's idle test is its own (input, CPU, disk) and may count a busy lab against itself: read
`LAB/runner.log` after the first nights before trusting it. The runner's own input check is the backstop either way.

`-Trigger Logon` is the always-on version for a week away: starts at the user's logon, no idle condition, no stop at the end
of idle, priority 7, no execution time limit, MultipleInstances IgnoreNew (a second start is ignored), restart on failure
every 5 minutes up to 3 times, still Interactive and Limited with no password, and the runner gets `--always-on` (and not
`--exit-with-parent`: a parent that Task Scheduler's service does not let the user's token open would read as gone and end a
week's work at once; the Job Object that holds the task already kills every child when it stops) and `--lab-gb 100` (a week
of unattended data; `-LabGb N` sets it, in either mode). The idle mode stays the default.

## Tests

`python -m unittest tests.python.test_lab_runner tests.python.test_lab_mix` (about 20 s): guards with fake clocks (also PAUSE
files written as PowerShell writes them, the always-on stay mode), load window maths, pruning (what stays, failed writes),
plans on a fake tree (one per strategy and the reasons one cannot run), the mix file, the scheduler's CPU shares, the yield
rows, the runner on trivial commands (done.rc last, resume, supersede, PAUSE now / with grace, HALT, backoff, always-on
waiting out a HALT), a real Job Object test (kill a process from outside: its grandchild dies), and the task script under
`-WhatIf` for both triggers.
