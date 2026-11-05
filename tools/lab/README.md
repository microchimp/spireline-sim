# tools/lab: the fix lab's runner

Spends idle (or, for a week away, all of the) PC time on the work that makes backlog fixes cheap (aimed real-game fights,
whole runs, shrinking, grouping, fix kits). No AI calls, no network. The shared design and the other parts' CLIs are in
`referee/RUNBOOK.md`, section "Fix lab". This folder is the runner and the Windows task.

| File | What |
|---|---|
| `runner.py` | CLI, lab folder, job records, the main loop, supervising child processes |
| `plans.py` | the jobs as data: steps and symbolic commands (`{tree}`, `{lab}`, `{job}` ...), one plan per strategy |
| `jobs.py` | the agent-era job kinds (selfplay, train, score-real, nightly-real, bench-env, bench-emu): defaults, `LAB/jobs.json`, markers |
| `mix.py` | the strategy mix (`LAB/mix.json`), the scheduler that follows it, the yield record (`LAB/yield.tsv`) |
| `guards.py` | PAUSE / main.lock / user input / load / disk / HALT as one `Verdict`; `TreeCpu`, the always-on loop's CPU the load guard leaves out |
| `frozen.py` | D52: the frozen copy of the lab tree the practice filler runs from (one per presorted build, outside the lab folder) |
| `cores.py` | D48: the cores each job takes (`LAB/jobs.json` `cores`), and the practice filler's rules (its command, the night, QUIET.lock, the floor) |
| `promotion_guard.py` | the review's promotion guard (McNemar at 0.05, boss damage, lineage drift, retry cap) and its power tables; `rl/gen_loop.py` calls it behind `SPIRELINE_REVIEW_GUARD` (off), see `docs/PROMOTION_GUARD.md` |
| `keep_v2.py` | the keep test v2 (promotion on the deepest boss the lineage reaches: frontier, deep_f, routes A and F, crash guard, the funnel's call at each look); `rl/gen_loop.py` calls it behind `--keep-rule v2` / `SPIRELINE_KEEP_RULE` (default v1), see `docs/rl/KEEP_TEST_V2.md` |
| `pairing_check.py` | `docs/PROMOTION_GUARD.md` section 4 points 1 and 2 on two arms' `runs.jsonl` (seed sets, config, crashes, discordant kills, floor-difference SD), with a permutation reference; read-only |
| `prune.py` | what may be deleted, in what order, and what never (recordings the corpus already holds go first) |
| `corpus_import.py` | adds a job's exact and divergent fights to the regression corpus (last step of a fights job) |
| `winsys.py` | ctypes: GetLastInputInfo, priority class, Job Objects (optionally forcing one priority class on every member, or a memory limit), the machine's RAM, GetSystemTimes, a process tree's CPU, mutex |
| `stepcache.py` | the skip-by-inputs stamps (`LAB/stamps/<step>.json`): digests of a step's inputs, the engine digest, `skip_reason` / `write_stamp` |
| `register_task.ps1` | registers "Spireline fix lab" (`-Trigger Idle` or `Logon`, `-WhatIf`, `-Unregister`) |

## CLI

```
python tools/lab/runner.py [--lab D:/spireline/referee/lab] [--tree D:/spireline/fork/wt-lab] [--game D:/spireline/referee/game-lab]
    [--trial MINUTES] [--once presort|fights|runs|probes|mutation|invariants] [--dry-run]
    [--skip PREFIX,...] [--no-guard user,load,disk,pause,lock,halt,parent] [--exit-with-parent] [--always-on]
    [--fights 300] [--batch 50] [--servers 8] [--procs 8] [--mutants 150]
    [--shrink 1000] [--shrink-cap 160] [--shrink-budget 6000] [--shrink-minutes 45] [--trace 1000] [--trace-minutes 20]
    [--runs 40] [--run-batch 10] [--run-procs 4] [--run-stop-act 3]
    [--report-beside auto|on|off] [--filler auto|on|off] [--filler-tree auto|off|DIR] [--loop-status D:/spireline/logs/gym-loop/status] [--idle-seconds 60] [--load-pct 50] [--load-window 120] [--lab-gb 30] [--free-gb 50] [--grace 90]
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
- `--presort-gap-min M` (default 60): one presort per batch of main moves. When main moved, the presort waits until the last
  one ended M minutes ago; meanwhile the ladder and the fights rotation keep running on the presorted tree (their job
  records name that tree's sha). It does not wait when no other job can run on the tree, when the tree is not a good presorted
  one (a failed presort), or with `--once presort`; `0` presorts on every move. The end of the last presort is `state.last_presort_end`
  (an older state: the newest finished presort folder).
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

`presort` runs when local `main` differs from `state.last_presorted_sha` (or the tree is not at it), at most once per
`--presort-gap-min` unless nothing else can run: `tree` (worktree add /
`checkout --detach --force`), `restore` (offline, only when `obj/project.assets.json` is missing), `build` (EmuDump),
`driver` (referee driver), `native` (`dotnet publish ... win-x64`), `game` (linkcopy), `checks` (codex walk in 4 shards +
refset score, in parallel; the refset score and the corpus results are copied into the job folder under the roles
`lab_groups` reads), `corpus` (`corpus.py check --no-scan`), `groups`, `kits`. A failed `tree` or `build` fails the job
and blocks everything on that sha (retry after 30 min, doubling); the checks' own exit 1 (a regression) is a result, not a failure.

After a presort, a rotation by value: `fights, fights, probes, fights, mutation, fights, invariants`. A job type that cannot
run (tool or input missing, backing off after two failed jobs in a row) is skipped with a log line and the rotation moves on.
`probes`, `mutation` and `invariants`, and the `retest` strategy, run **once per presorted tree**: `state.json`'s `last_ok_sha` keeps, for
each, the tree its last rc 0 job ran on, and a pick on that tree skips it ("ran on this tree already (<sha>)", written once an hour at
most). A retest is due again on the same tree only for a group marked for one since the last retest job finished (`retest_seen` in the
state). `--once` ignores the rule; a state file from before it is seeded from the newest finished job folders. When no slot of a pass has
a job the lab is idle: "nothing to do", the filler starts beside idle, and the next look is `--idle-poll` (5 minutes) later. The mix is
asked once per pass, not once per fights slot.

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
An optional `"_every_h": {"aimed-bugs": 24}` gives a strategy a cadence (the only `_` key that is not a note): while its last
job that finished rc 0 ended less than that many hours ago it is passed over as "ran within the last 24 h (due again ...)", and the
slot goes to the others or, with none due, the lab idles and the filler has the machine. The time is the newest of `yield.tsv`'s
rows (columns `utc`, `strategy`, `rc`) and `state.json`'s `last_ok` (set when the job finishes: the row of a fights job waits for
the report beside the next job). A strategy with no finished job is due; `--once` runs ignore the cadence.

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

## Skip by inputs, and the report beside the next job

A step that is a function of files records a digest of those files beside its output (`LAB/stamps/<step>.json`, written when the
step succeeds); the next time the step runs it first digests its inputs, and when the digest is the one in the stamp **and every
output the stamp names is still there with the size and time it was written**, the step is skipped, its earlier output stands, and one
line says so (`step skipped: ...` in the step's log; the runner's own `skipped (inputs unchanged ...)` for the presort checks). A
missing, pruned or rewritten output, a changed input, a changed digest of the code, `--force` (groups and kits) all run the step. No
timer: a step that changes rarely but not never (the groups file) reruns the day an input changes.

| step | where it skips | inputs (digest of) |
|---|---|---|
| `codex-s0..3` | the runner (`plans.check_parts`) | the engine (`stepcache.engine_digest`: `src/Sts2Emulator`, `tools/emudump`, the invariant checks EmuDump compiles in, the data files and build settings it copies), the Codex dataset (size, time), the decompile folder, the shard |
| `corpus` | the runner | the engine, `corpus.py` with the fidelity comparators and data files, the corpus index (the fights) |
| `refset` | the runner | the engine, `refset_score.py` with the comparators and `referee/refset.py`, the reference set's manifest and records |
| `groups` | the tool (`lab_groups.py`, `--force` to rerun) | every file a source is read from (the batches' `<name>.*` files and packs, runs, corpus / refset / codex / invariants / mutation outputs of every finished job: size and time), the newest backlog, `id_map.json`, `sweep_map.json`, `FINDINGS.md`, the host gaps, `retests.json`, the Codex profile, the decompile's model files, and the grouping code |
| `kits` | the tool (`packet.py --groups`, `--force` to rerun) | the top items, the groups those items would take (the first three matches, whole, and how many more: a group no item matches does not count), the engine and the xunit tests, the Python tools, the evidence files by content (ledger, findings, attempts, `id_map`, `sweep_map`, corpus index) and the evidence folders by their entries (probes, traces, minimal packs, reference set, Codex walk, corpus runs) |

The tree's sha is not an input: a landing that changes none of the above (lab code, `rl/`, docs, the xunit tests) leaves every digest
as it was, so only the steps whose inputs moved run. The groups and kits skip inside the tools, so they take effect when the tree is
at a commit that has them (the next presort); the three presort checks skip in `plans.py`, which the runner process loads when it
starts. Not tracked: the corpus check's nearest-ancestor baseline (the lab reads its `results.json`, which does not depend on it);
raw recordings a group's examples point to (a kit built before their pruning stands); folders searched deeper than one level for kits.

**The report beside the next job.** With `--always-on` (or `--report-beside on`) a job that ends well hands its groups and kits to the
runner's *report*: the job is finished at once (done.rc, state), the next job starts, and one chain per time (groups, then kits; a
failed groups stops the chain) runs in a Job Object of its own beside it, at the runner's BelowNormal priority. The chain covers every
job that finished since the last one started; each gets its `yield.tsv` row when the chain ends (group columns NA when groups did not
finish). The list of jobs whose row is owed is kept in `state.json` (`report_pending`) and put back on the queue when the runner starts
again (a restart for a landing used to drop it, and with it the rows). It stops at once on any guard except a held `main.lock` (user, pause, load, disk, halt: the covered jobs wait for the next
chain), gives way to a presort (the tools run from the tree the presort checks out, builds and publishes into: the report is stopped at once and its jobs wait for the next one, which starts when the presort ends and reads the new tree; a report is 40 to 70 minutes and nearly always running, so letting it end was most of a presort's wall time. After `--report-cut-cap` stops by presorts in a row (default 2) the next presort waits for the report; `--presort-report wait` is the old way), is never started while a nightly window is open
(`LAB/nightly/active`) and is stopped when a nightly-real job starts. The report's CPU is kept out of the jobs' CPU accounts (the
mix schedules on them). A fault in the machinery turns it off for the process and the groups and kits are steps of each job again.
Off for `--once`, `--trial` and `--dry-run`. `corpus-import` stays a step of the fights job (prune checks the groups file's time).

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
| `selfplay` | 6 | `enabled` (with `filler`: beside the jobs, see D48 below; else the bottom of the ladder, `share` > 0 puts it ahead of the rotation until the loop has used that share of the last 7 days' CPU, see below) | off; placeholder command | yield: runs, decisions, decisions_s, generation |
| `train` | 3 | `LAB/train/ready-<gen>` exists (oldest first; renamed `done-<gen>` on success); 60 min cap | on, marker-gated | yield: generation, promoted, loss |
| `score-real` | 2 | `LAB/score/pending` exists, or `weekly` and 7 days since `LAB/score/last-ok`; 240 min cap | on, marker-gated; weekly off | yield: runs, wins, mean_floor, crashes, decisions_s, divergences. `LAB/score/active` exists while it runs; fights and runs jobs are not started then (a marker older than 5 h is ignored) |
| `nightly-real` | 2 | `enabled`, `every_h` (20) since `LAB/nightly/last-ok`; held (reason in runner.log) while `LAB/HALT` or `LAB/PAUSE` exists, `LAB/nightly/active` or `LAB/score/active` is fresh, or the last try was skipped under 1 h ago; 600 min cap | off; placeholder command | runs `rl/nightly_real.py`: per night a fresh 2,000-seed block (`N-<date>[-<agent>]` in `LAB/seeds/`) for the incumbent and each generation promoted since the last night, the lockstep with the emulator arm on `--engine core2`, `--recycle 60`, under `QUIET.lock` (`lab-nightly <UTC>`, refreshed while it runs; waits up to 2 h, then skips). Folder `LAB/nightly/<date>/<agent>/` (+ sibling `<agent>-emu/`); summary.json `nightly` counts a crash as a loss; readiness.json `nightly`; yield: runs, wins, mean_floor, crashes, divergences |
| `bench-env` | 7 | `enabled` and `every_h` (20) since `LAB/bench-env/last-ok` | off | yield: dec_s, crashes, mem_mb_per_run; readiness.json `bench.env` (next to LAB) |
| `bench-emu` | 7 | same; needs `{tree}/rl/bench_emu.py` | off | yield: steps_s, clones_s, sims_s; readiness.json `bench.emu` |

Ranking input of the nightly blocks (each folder is already ranked in place; to rank one again or join its arm by hand): `python tools/lab/rank.py LAB/nightly/<date>/<agent> --out <dir> --emu-dir LAB/nightly/<date>/<agent>-emu` (by_family.md/json are written beside top20.md; weights unchanged).

A command reports through `{job}/result.json` (an object; keys named like the yield columns). All guards (PAUSE, HALT, main.lock,
user input, load, disk, parent) apply to these kinds as to every job, between steps and while the command runs. The runner is
sequential, so a job never runs beside another job of the ladder or the rotation (the filler below is the exception). New `yield.tsv`
columns come after `screen_groups`; an older file is widened with NA, and old rows keep their values.

## Practice as the filler, cores not turns (D48, `cores.py`)

With `--always-on` (or `--filler on`) and `selfplay.enabled` and `selfplay.filler` in `LAB/jobs.json`, self-play is no longer a ladder
job: it runs beside whatever job runs, on the cores that job leaves. Each job type has a `cores` figure (a kind's own `cores`, else its
`procs`, else its command's `--procs`; the fix-lab types from the top-level `cores` map; nightly-real always the whole machine;
presort 4, its codex shards, since D52). The filler's size is `cores.total`
less the job's cores, the report's (`cores.report`, 2) while one runs, and the always-on loop's (`cores.loop`, 6) while its status file
says it is stepping; it runs only at `selfplay.floor` (4) or more. Before a job runs, the filler gives up the cores the job needs at once
(stopped through its Job Object, then started again at what is left); a block that ends starts the next at whatever is free then.
When the runner has no job to run (the ladder and the mix give nothing) the filler starts in the idle wait, on the cores the loop and
the report leave ("filler: <name> at N cores beside idle"), under the same holds; the first job to become ready takes its cores first.
Each block is a `LAB/runs/<UTC>-selfplay` job folder (`params.filler`), its command the selfplay command without `--lab`,
`--shards-root`, `--budget-runs`, `--review-guard`, `--gen-dir` and exploring flags, with `--record` and `--procs` / `--server-procs`
set to its size, plus `selfplay.explore` when given: it plays `agent/current.json`, writes shards to its own folder and never a train
marker. Its Job Object forces BelowNormal on every process in it (the prior server included). It holds no QUIET.lock line and no
main.lock; it stops for every guard but main.lock, for a live QUIET.lock line whose tag is not in `selfplay.quiet_ignore`
(`gym-loop`, `lab-score`), when `LAB/nightly/active` appears, two minutes before the night is due (`LAB/nightly/last-ok` + `every_h`, or the retry
after a skipped try) and when a nightly-real job starts; it does not start within `selfplay.night_margin_min` (30) of the due time,
within 120 s of a stop, near the disk limits, or on a tree that is not a presorted build. A stopped block is asked first: the runner writes `STOP` in the block's folder and waits up to 25 s
(`runner.FILLER_STOP_GRACE_S`) for a block that wrote `stop-honored` (a `gen0.py emu --record` that reads the file) to put its finished
runs in the shards and end, then kills the Job Object; only the runs in flight are lost (before, every run since the last 50-run flush
was: 1,828 of the filler's 4,496 by 2026-11-04). A block that is killed hard still loses at most `gen_loop.FLUSH_AGE_S` (120 s) of finished
runs, since a part is also written when its oldest run has waited that long. A stopped block is closed with done.rc 0 and
the reason (its finished runs are in its shards); a block left by a runner that died is closed (done.rc 3), never resumed. Its CPU stays
out of the jobs' accounts. The load guard counts neither the filler (it is in the runner's Job Object) nor the always-on loop's
process tree (`--loop-status`), so neither pauses a job.

**It runs from a frozen copy of the lab tree (D52, `frozen.py`), not from the tree.** A presort checks out the new main, builds it and
publishes into `out/`, so a filler running from the tree had to stop for every presort (36 minutes or more with the machine at 7% of
its CPU, several times a day). The copy is made once per landing, when the tree stands at a presorted build and no presort is moving it
(`git rev-parse HEAD` equals the state's tree sha, no presort job folder without done.rc): the tracked files and the published `out/`,
about 1,400 files and 160 MB, two to ten seconds, into `<lab folder>/../filler-tree/<sha12>/` (`--filler-tree DIR` moves it) through a
`.part` folder renamed once whole, its `FROZEN` marker written last. A presort leaves the filler running (it takes `cores.presort`, 4);
the first block after a runner restart that finds no copy while a presort is moving the tree waits for the presort. When a presort ends
on a new sha the block still playing the old build is stopped at once (no 120 s wait), the next block starts from the new copy and the
old copy is removed when no block uses it. A presort that fails leaves a running block on its copy and starts no new one (the tree is
not a build). `--filler-tree off`: the D48 way, the filler runs from the tree and every presort stops it (with `cores.presort` 16 in
`LAB/jobs.json`, the D48 table, a presort also leaves the filler no cores).

## Lab folder

```
LAB/state.json status.txt runner.log targets.json groups.json summary.md coverage-setup.json coverage-random.json
    mix.json yield.tsv   PAUSE HALT (when present)
LAB/stamps/<step>.json   what a step that skips on unchanged inputs last ran with (tools/lab/stepcache.py)
LAB/cache/groups/jobs/<job>.json, sym/<aa>/<key>.json.gz   the grouping step's cache (tools/lab/groupcache.py): per finished job its
    sources, per source its symptoms; deleting the folder restores the old behaviour (one slow run), lab_groups.py --force clears it
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

`PAUSE` and `HALT` are read whoever wrote them: PowerShell's `echo now > PAUSE` writes UTF-16 with a BOM. A tag in it (its first word after an optional `now`, e.g. `echo "w4-macro 2026-10-25T07:01:00Z" > PAUSE`) is logged on the `paused:` line, `paused: pause (LAB/PAUSE) (w4-macro)`, so pause time can be split by session.

## The lab's clock

Every finished command adds a line to `LAB/runner.log` (and its figures to `job.json`): `groups/groups: rc=0 in 1727.2s cpu=412.3s
reads=88420 read_mb=2104.5 write_mb=3.1 out=same`. `in` is wall seconds; `cpu` the CPU seconds of the command and everything it
started (a Job Object of its own inside the step's); `reads`, `read_mb`, `write_mb` the system's I/O counters for it (read operations,
megabytes read and written; Windows keeps no count of files opened, a small file costs one or two reads); `out` whether the output
differs from the last run's, from the step's skip stamp (`changed`, `same`, `skipped` when the command found its inputs unchanged and
ended at once, `?` for a command that keeps no stamp or failed). `cpu=?` marks a command the Job Object could not be attached to.

A command with `mem_share` (plans.Cmd; the mutation job's `mut-prepare` and `mut-schemata` have 0.5) is held to that share of the
machine's RAM, read from the OS, by a Job Object memory limit on it and everything it starts: an allocation past the limit fails in
the process that asks (a runaway mutant allocates without end; before the cap its engine host held tens of GB for the 120 s stall
timer and exhausted the machine's virtual memory, 2026-11-05). The command's line then ends `peak_mb=N`, the most the processes held
together (a request the limit refused counts in it, so a peak above the share means something was refused), and `job.json` has the
same figure. A command whose cap cannot be set is not started (rc 127, `could not start`). `tools/fidelity/mutate.py` also puts
each replay host in a job of its own with `MEM_SHARE / (hosts + 1)` of the RAM (`host_cap`), so one runaway ends in its host as an
error the judge counts like a crash, and no neighbour is starved.

`python tools/lab/clock_report.py [--lab LAB] [--days 7] [--all] [--json FILE]` reads a week of that log and lists the steps whose wall
is more than three times their CPU (IDLE: they wait), whose output was the same five runs running (SAME5), or whose mean wall is above
20 minutes (LONG); the lab's own long work (training, the nightly block, self-play, scoring, the benches) is marked and not counted.
The runner writes `LAB/clock.md` (the table) between jobs once a day, and a `clock:` line in runner.log when a step is flagged;
tools/fidelity/readiness.py puts the flagged steps in readiness.json under `lab.clock`. Lines from before the runner wrote the figures
carry only the wall, so only LONG can fire on them.

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
