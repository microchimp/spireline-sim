# tools/lab: the fix lab's runner

Spends idle PC time on the work that makes backlog fixes cheap (aimed real-game fights, shrinking, grouping, fix
kits). No AI calls, no network. The shared design and the other parts' CLIs are in `referee/RUNBOOK.md`, section
"Fix lab". This folder is the runner and the Windows task.

| File | What |
|---|---|
| `runner.py` | CLI, lab folder, job records, the main loop, supervising child processes |
| `plans.py` | the jobs as data: steps and symbolic commands (`{tree}`, `{lab}`, `{job}` ...) |
| `guards.py` | PAUSE / main.lock / user input / load / disk / HALT as one `Verdict` |
| `prune.py` | what may be deleted, in what order, and what never |
| `winsys.py` | ctypes: GetLastInputInfo, priority class, Job Objects, GetSystemTimes, mutex |
| `register_task.ps1` | registers "Spireline fix lab" (`-WhatIf`, `-Unregister`) |

## CLI

```
python tools/lab/runner.py [--lab D:/spireline/referee/lab] [--tree D:/spireline/fork/wt-lab] [--game D:/spireline/referee/game-lab]
    [--trial MINUTES] [--once presort|fights|probes|mutation|invariants] [--dry-run]
    [--skip PREFIX,...] [--no-guard user,load,disk,pause,lock,halt,parent] [--exit-with-parent]
    [--fights 300] [--batch 50] [--servers 6] [--shrink 8] [--procs 8] [--mutants 150]
    [--idle-seconds 60] [--load-pct 50] [--load-window 120] [--lab-gb 30] [--free-gb 50] [--grace 90]
    [--fork <main checkout>] [--branch main] [--game-src D:/spireline/referee/game] [--backlog-dir ...] [--backlog FILE]
    [--codex DATASET] [--decompiled DIR] [--records DIR] [--poll 2] [--wait-poll 10] [--idle-poll 300]
```

Run it from the fork's main checkout (it only reads it; `--fork` defaults to the checkout holding `runner.py`). Jobs run
with `--tree` as cwd. It refuses a `--tree` that is the main checkout, a `--lab` inside the fork, and any path under
`D:/spireline/src`. Exit 0: stopped by a guard (user input, HALT), `--trial` over, `--once` done, nothing to do with
`--once`, or another runner holds this lab; 1: crash; 2: refused arguments.

- `--dry-run`: guard readings, next job, its plan with every command's skip reason. Creates and changes nothing.
- `--trial M`: work for M minutes, let a running step finish within `--grace` (at most 60 s), exit 0.
- `--once J`: one job of type J (a `fights` needs the tree presorted at main first: `--once presort`), then exit.
- `--skip`: prefixes of step or command names. Steps: `tree restore build driver native game checks corpus groups
  kits targets batch-NN probes mutation invariants`. Commands: `worktree-add checkout restore-* build-emudump build-driver
  publish-native game-copy codex-s0..3 refset corpus groups kits targets scen-NN probes mut-prepare mut-schemata
  mut-report meta invariants`.
- `--no-guard`: switch guards off for a trial on a busy PC. Never in the task.

## Jobs

`presort` runs when local `main` differs from `state.last_presorted_sha` (or the tree is not at it): `tree` (worktree add /
`checkout --detach --force`), `restore` (offline, only when `obj/project.assets.json` is missing), `build` (EmuDump),
`driver` (referee driver), `native` (`dotnet publish ... win-x64`), `game` (linkcopy), `checks` (codex walk in 4 shards +
refset score, in parallel), `corpus` (`corpus.py check --no-scan`), `groups`, `kits`. A failed `tree` or `build` fails the job
and blocks everything on that sha (retry after 30 min, doubling); the checks' own exit 1 (a regression) is a result, not a failure.

After a presort, a rotation by value: `fights, fights, probes, fights, mutation, fights, invariants`. A job type that cannot
run (tool or input missing, backing off after two failed jobs in a row) is skipped with a log line and the rotation moves on.

- `fights`: `lab_targets.py` (if present), then `ceil(--fights/--batch)` batches, one `scenarios.py` call each, so there is a
  safe point every minute or two; mode `targets` when the tree's `scenarios.py` has it and `lab_targets.py` exists, else
  `coverage`; `--jobs <job>/fights` when the tree's `scenarios.py` takes it, else the same call through a one-line shim
  that sets its output folder (existing modes unchanged). A batch's `summary.txt` is parsed: any network connection or a
  touched real profile writes `LAB/HALT` and stops the lab. The batch's `serve/setup-<name>*` folders are removed after it.
- `probes`: `referee/probe.py run --only <ids>` for ids that `backlog/packets/grades.json` lists under `probes` for items
  missing a rule. Otherwise skipped (probes.py takes a list of ids, not a queue).
- `mutation`: `mutate prepare/schemata/report` on the biggest raw batch of the newest fights job; `mutation.json` is the reach data.
- `invariants`: `invariants meta` (chooser seeds grow by 4 each pass) and `invariants run` over the recorded fights.

## Lab folder

```
LAB/state.json status.txt runner.log targets.json groups.json summary.md coverage-setup.json   PAUSE HALT (when present)
LAB/runs/<UTC>-<job>/job.json done.rc logs/<command>.log|.rc pruned.json
    presort:  codex/s0..3.jsonl|.err   (logs/refset.log, logs/corpus.log)
    fights:   fights/<name>.{json,out.json,emu.json,reanchor.json,verdicts.json,summary.txt}, <name>.packs/{index.md,minimal/}, shrink files
    probes:   probes/     mutation: mut/, mutation.json     invariants: meta.json, invariants.json
```

`job.json`: `{version, job, name, main_sha, created_utc, finished_utc, rc, resumed, note, params, steps:[{name, status
(pending|running|done|failed|skipped), interrupted, started_utc, finished_utc, note, cmds:{<name>:{status, rc, secs, argv
(symbolic), note}}}], outputs:{role:[path relative to the job folder]}}`. Roles: `codex_walk`, `codex_walk_err`,
`refset_score` (`refset/score.json`, a copy of the stored score), `corpus` (`corpus/results.json`, a copy of the
corpus run's results), `fights` (verdicts), `probes`, `mutation_reach`, `metamorphic`, `invariants`.
`done.rc`: 0 ok, 1 required step failed or every core command failed, 3 superseded.

## The Windows task

`powershell -ExecutionPolicy Bypass -File tools\lab\register_task.ps1 -WhatIf` prints the definition and changes nothing;
without `-WhatIf` it registers "Spireline fix lab" for the current user (Interactive, Limited, no password): idle trigger,
RunOnlyIfIdle 10 min, StopOnIdleEnd, RestartOnIdle, priority 7, no WakeToRun, not on battery, 12 h limit, MultipleInstances
IgnoreNew; action `pythonw.exe tools\lab\runner.py --lab ... --tree ... --game ... --exit-with-parent` in the fork's main
checkout. `-Unregister` removes it. GetLastInputInfo only sees the logged-on user's input, which is why the task is
Interactive. Task Scheduler's idle test is its own (input, CPU, disk) and may count a busy lab against itself: read
`LAB/runner.log` after the first nights before trusting it. The runner's own input check is the backstop either way.

## Tests

`python -m unittest tests.python.test_lab_runner` (about 12 s): guards with fake clocks, load window maths, pruning (what stays),
plans on a fake tree, the runner on trivial commands (done.rc last, resume, supersede, PAUSE now / with grace, HALT, backoff),
a real Job Object test (kill a process from outside: its grandchild dies), and the task script under `-WhatIf`.
