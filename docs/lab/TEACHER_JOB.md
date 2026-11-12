# The teacher job (`rl/run_teacher.py job`, `tools/lab/jobs.py` kind `teacher`)

Written 2026-11-11 (lane 2 of the learning-system build plan, D56 and D57 part 3; checkpoint C4). The data it makes is read by the run
student (`rl/run_student.py`, `docs/rl/RUN_STUDENT.md`); the teacher itself is `docs/rl/RUN_SELF_PLAY.md` section 3. No human record is read
anywhere in the teacher or its job.

## What it does, in plain words

At each run-level screen of a self-play run (which card, which route, rest or smith, what to buy, which event option, which relic) the
teacher plays each of the best few options forward on copies of the run, a few times each, to the next boss door, and writes down how many
floors each option was worth and how sure that number is. The student learns from those marks. This job makes that a daily routine: it
plays runs with the current player (`agent/current.json`) on the idle part of the machine, under the rules the practice filler kept, with
a budget of 24 core-hours a day, and keeps what it writes in a store the student reads. Like a coach who watches some of the team's
games every night, replays each decision both ways, and keeps notes.

## The one command

From a tree that does not move under it (a worktree detached at the landed commit with `out/` built by `publish-wt.sh`, as the gym loop's
`wt-gym-loop-run`; the lab tree moves at every presort), with the emulator Python:

```
D:/spireline/src/sts2/.venv/Scripts/python.exe rl/run_teacher.py job --lab D:/spireline/referee/LAB --store D:/spireline/referee/run-teacher \
    --agent-root D:/spireline/referee/agent --judge D:/spireline/referee/agent/judge/v2b/judge-board.pt --source new --procs 4 --loop
```

**The judge** is the fight judge the teacher asks for P(death) at every boss door. It is **v2b** (`agent/judge/v2b/judge-board.pt`, the default of
`--judge` since D58 addendum 24; before that v0, `agent/judge/v0/judge.pt`, and round 0's labels and the first blocks of round 1 were v0's). v2b reads human
positions better than any judge before it and beats v0's arms on the boards (`docs/rl/JUDGE.md` section 8.1). Of v2b's two files the teacher uses
`judge-board.pt`, the checkpoint without the source row (`src_emb`, the human-outcome input): the player's loader never feeds a source (None is the
teacher, whose row 0 is zero), so `judge.pt` and `judge-board.pt` read identically (the same estimate, value, P(death) and end HP, on eight states built from the engine, and on 64 human boss positions, `docs/rl/JUDGE.md` section 8) and the board file
also loads on any engine tree. The manifest of every shard names the judge (`judge`, `judge_sha256`), and the door's recalibration is kept per judge
(section "P(death) recalibrated per act").

**Round 0** (labelling the stored generation 13 exploring runs; a one-off, **not under the daily budget**, which applies from round 1, the daily
job above) is the same program with `label`, which takes `--procs N` (a spawn pool, one engine per process, BelowNormal; the pool's usual cap of
six is lifted for this job; `--mem-gb 4` kills and retries a worker over that much private memory):

```
D:/spireline/src/sts2/.venv/Scripts/python.exe rl/run_teacher.py label --lab D:/spireline/referee/LAB     --store D:/spireline/referee/run-teacher/gen13-r0 --agent-root D:/spireline/referee/agent     --judge D:/spireline/referee/agent/judge/v0/judge.pt --procs 12 --share
```

(round 0 was run with v0, as written here; a `label` run today names `--judge` as it wants and defaults to v2b.)

`--share` because the machine is shared by order (without it the job waits on any live QUIET.lock line, a corpus builder's included). It is one
block of up to 12 hours over the stored runs in a fixed shuffle, 5 screens at most from each (`--per-run`), stopping new runs when the store holds
10,000 labelled screens (`--target-screens`, 0: no limit; the runs in flight finish), and it writes the shard of the runs finished so far every
30 minutes (`--shard-every-min`, parts `<job>-p001`, ...; each run is in one part only), so the student can be started on a part while the
block goes on. Its data is under `gen13-r0/13/` (`<store>/<generation>/`); `run_student.py ... --shards-root D:/spireline/referee/run-teacher`
reads it and the daily job's together. A stopped or killed round 0 is started again with the same command: finished runs are not played again.

The daily job reads the player from `agent/current.json`, plays one block (`--block-min`, default 60: no run starts after that; the runs in flight
finish), writes the shard, waits while a rule holds it, and starts the next block, day after day (`--loop`; without it, one block and exit).
`--source new` (the default of `job`) is round 1 and on: whole runs the guarded teacher plays. `--source stored` (the command `label`) is round 0,
above. `--source auto` is both in one store: the stored generation 13 exploring runs of 2026-11-05 to 11-08 (6,889 runs), replayed exactly and
labelled, until 10,000 screens have come from the store, then new play (not beside a separate round 0 store: it would label the same stored runs
again). `--share` (a hand-started job only) honours HALT, PAUSE and the night, and ignores QUIET.lock lines, judgement first
and the presort rule: for a machine shared by order.

## The rules (the filler's, in `tools/lab/jobs.teacher_hold`)

One function of the lab folder, used by the runner's pick (`jobs.unready`) and by the job's own poll every ten seconds, so they agree:

| rule | at the start | while it runs |
|---|---|---|
| `LAB/HALT` or `LAB/PAUSE` up | holds | stops |
| a night running (`LAB/nightly/active`), or due within `night_margin_min` (30) | holds | stops two minutes before it is due |
| a live `QUIET.lock` line that is not in `quiet_ignore` (default `gym-loop`, `lab-score`): `lab-train`, `w3-blocks`, `judge-corpus` and every other tag | holds | stops |
| judgement first: a train marker (`LAB/train/ready-*`) or `LAB/score/pending` waiting | holds | a running block ends within its `block_min` |
| a presort due within `presort_horizon_min` (60): main moved past the presorted tree and the last presort ended `presort_gap_min` ago or will have | holds | carries on |
| the day's core-hour budget (`core_hours_per_day`, 24, UTC day) spent | holds | the workers start no run the budget has no room for |

BelowNormal: the workers lower themselves with the Windows priority class (`procutil.lower_priority`), the main process too; never
`os.nice`. It holds **no QUIET.lock line** (the existing filler kinds hold none; the always-on loop and the nightly's driver wait on the
first live line, and a job that holds none never makes them wait). At most `--procs` (4) processes. One job at a time on a store
(`<store>/job.lock`, taken over when its process is gone).

**The budget** is a ledger (`<store>/ledger.jsonl`, one line per event). A worker writes a `start` line with the run's estimated CPU
before it plays, and an `end` line with the CPU it used. A day's spend is the CPU of the runs that ended that UTC day plus the estimate
of each run in flight (not those of a job that wrote its end, nor of a start older than two hours: a killed job's leftover). A worker
checks the spend with its own reservation and every reservation before it; when that is over the budget it retracts its line and the run
is asked for again by the next job. So the budget is not overspent by the runs running when it ran out, and two workers looking at once do
not both take the last room. A killed run costs the CPU its partial work used, which the ledger does not see until the run is replayed
and ends (the replayed run's `end` line carries the CPU of its stored screens).

When the runner schedules it (kind `teacher`, priority 8, the lowest: after everything else the lab has) the rules above are its
`unready`. The ladder hook in `tools/lab/runner.py` (`ladder_top`: after the benches, `jobs.unready("teacher", ...)`) is in since lane 4
(`docs/lab/RUN_LOOP.md`, which also has this fragment with the `student` kind's): the runner picks the teacher when `LAB/jobs.json` enables it and
nothing above it is ready. The job can still be started by hand with the command above, which keeps to the same rules by itself.

## The jobs.json fragment (disabled; the orchestrator merges it)

```json
{
 "teacher": {
  "_note": "2026-11-11 (lane 2, D56/D57): the run-level teacher job. Off; the one command in docs/lab/TEACHER_JOB.md starts it by hand until the runner's ladder picks the kind (lane 4).",
  "enabled": false,
  "core_hours_per_day": 24,
  "block_min": 60,
  "procs": 4,
  "cores": 4,
  "timeout_min": 240,
  "presort_horizon_min": 60,
  "quiet_ignore": ["gym-loop", "lab-score"],
  "store": "D:/spireline/referee/run-teacher",
  "cwd": "{tree}",
  "cmd": [
   "{python}", "rl/run_teacher.py", "job", "--lab", "{lab}", "--store", "D:/spireline/referee/run-teacher",
   "--agent-root", "D:/spireline/referee/agent", "--judge", "D:/spireline/referee/agent/judge/v2b/judge-board.pt", "--lib", "{tree}/out",
   "--source", "new", "--procs", "4", "--block-min", "60", "--budget-h", "24", "--result", "{job}/result.json"
  ],
  "needs": ["{tree}/rl/run_teacher.py", "{tree}/out/Sts2Emulator.dll"]
 }
}
```

`result.json` carries `runs`, `decisions` (teacher screens), `decisions_s` and `generation`, the yield columns the lab reads, and the
block's CPU, shard counts and the budget left.

## The store (`D:/spireline/referee/run-teacher/<generation>/`)

```
runs.jsonl                  one row per finished run (a run with a row is never played again; a capped row is not a finished run)
decisions.d/<seed>.jsonl    a run's decision records as they are made; <seed>.rows.f32 their option rows, <seed>.rows_map.i8 their map grids
                            (int8, 1,547 a row, written before the rows), <seed>.rows_gaps.f32 their block G (float32, 36 a row); <seed>.record.json the run's RunRecord
shards/<job>/               the decision shard: decisions.jsonl, rows.npz, rows_map.npz, rows_gaps.npz, manifest.json
reencode.d/<seed>.npz       `reencode`'s per-run results (rows, grids and options built again), kept so a stopped reencode resumes;
                            <seed>.gaps.npz the same for block G (rows, gaps, options)
jobs/<job>/                 job.json, shards/ (the RunRecords of the job's runs in the lab's own run-shard layout)
calibration.json            the door's recalibration, refitted at every job start on the pairs recorded under the judge the job runs with
                            (`judge_sha256`, `doors` used, `doors_other_judges` left out, the fitted `acts`)
manifest.json, report.json  the store's index and totals; the measurement's report over every decision in the store
```

and `<store>/ledger.jsonl` (the budget), `<store>/job.lock`. A killed run's screens are on disk: the next job **replays** the run from its
start (the run is a function of its seed, the player and the teacher's picks, all stored) and plays only the screens it does not have.

**The decision shard** is the one the build plan fixes (`referee/backlog/run-self-play/BUILD-PLAN.md`, read by `run_student.load_shards`):

* `decisions.jsonl`: one record per screen exactly as the measurement writes it (`seed, ordinal, type, kind, act, floor, hp, max_hp,
  n_options, played, player, teacher, teacher_raw, k, screen, lam, v_values, logp, g, q, gain, gain_se, half_picks, half_played, ends,
  p_death, door_hp, options, cpu`, and `v`, `gain_se_own`, `refused`) plus `row0`, the screen's first row in `rows.npz`. `k` is the draws
  taken at that screen (2 or 4 under adaptive K); `p_death` is the judge's own number at the door, `g` the returns under the recalibrated
  one.
* `rows.npz`: `rows` float32 (sum of options, 1863) as `human_prior_infer.option_rows` builds them at play, for every option the judgement can
  score, on every screen (a screen the player does not blend included), `decision` int32 the record's line, `option` int16 the option's
  index. An option the rules policy decides (`select.enchant`, ...) has no row; a screen with one is in the shard for its marks (`q`, `g`)
  but the student's loader counts it incomplete and does not learn from it. A screen the teacher could not read (an error in its play-out)
  is a record with `refused` set and the player's own pick, so it never costs the run.
* `rows_map.npz` (from D58 addendum 13): `grid` int8 (sum of options, 1547), block M of each row's afterstate (`features.map_grid`: the whole act
  map, 17 rows of 7 columns of 13 numbers, the position's reachable set and the position itself), row for row in `rows.npz`'s order; the
  manifest says `map_grid: true`. The student reads it with `run_student.py ... --map-grid` (`docs/rl/RUN_STUDENT.md`). A shard whose runs were
  played before the grid was stored has no such file (`map_grid: false`): `reencode` writes it (below).
* `rows_gaps.npz` (from D58 addenda 14 and 16): `gaps` float32 (sum of options, 36), block G of each row's afterstate (`features.afterstate_gaps`: the relic counters
  the engine reads (`vocab.COUNTER_RELICS`, over their scales), the "?" rooms visited this act and the odds of a fight, a treasure room and a shop, the merchant's
  removal quote and the removals owed, and three known bits), row for row in `rows.npz`'s order; the manifest says `state_gaps: true` and `gaps_version: 1`. The
  numbers are read off the engine's own state (state lists 25, 26 and 27) at the decision, so a shard played before has them only from a replay: `reencode` writes
  the file (below). The student reads it with `run_student.py ... --state-gaps` (`docs/rl/RUN_STUDENT.md`); a file for another block G version, another row count
  or with a number that is not finite is refused.
* `manifest.json`: the agent (`agent`, `agent_sha256` over the combat prior's, the run head's and the human prior's sha256s and the recipe),
  the judge and engine (`judge`, `judge_sha256`, `engine`, `engine_sha256` of the native library), `k`, `k_max`, `m`, the leaf's B
  (`beyond`: 30.93 and 42.30), the guard's Z, NU0 and priors, the door's recalibration (`recal`), the namespace of the seeds, `seeds` (the
  runs in the shard), the sources (`new` and `stored` runs) and the `counts` the loader checks (`decisions`, `rows`).

A finished run is in one shard only (the job that finished it, or the next job's, if the first was killed before it wrote one); a shard
is rebuilt whole when its job runs again.

**`reencode`: the map grid and the state gaps of an older shard.**

```
python rl/run_teacher.py reencode --store D:/spireline/referee/run-teacher/gen13-r0 --generation 13 [--sides grid,gaps] [--procs 4] [--lib <native folder>] [--agent <player folder>]
```

adds `rows_map.npz` and `rows_gaps.npz` (`--sides` limits it to one) to every shard of `<store>/<generation>` that lacks them. It takes the store's job lock (it refuses while a teacher job runs),
checks that the player it loads (`agent/current.json`, or `--agent`) is the one the shards name (`agent_sha256`), and builds the rows of every
labelled screen again: a stored exploring run (source `stored`) is **replayed exactly** (`hindsight.replay_run`: the run's own seed and streams,
checked against its record; the replay stops after the last labelled screen, and its fights are the stored run's own, no search of the teacher's
is run), a run the teacher played (source `new`) is read from its RunRecord (`decisions.d/<seed>.record.json`), which holds every state and option
list, so no game is played. The rows built again must **equal the stored `rows.npz` rows exactly** (the same options, the same float32 numbers):
a shard where any screen differs is refused as a whole (`ReencodeError` names the run, the screen and the largest difference) and nothing is
written for it; otherwise `rows_map.npz` is written (tmp, then rename) and the manifest gains `map_grid: true`. It is idempotent (a shard that has
its files is skipped) and resumes (`reencode.d/` keeps each finished run). A run is replayed once for whatever side files its shard lacks; a run whose grid result
is already in `reencode.d/` (an earlier reencode of the map grid) is replayed again only for its block G. Block G is the engine's state at the screen, so it
needs an engine view that holds all three parts (a native library older than state lists 26 and 27 is refused for it, `--lib`), and a teacher-played run's
RunRecord holds block G's source only if the run was played by a build that recorded it (`state.gaps` in the record): an older record cannot give it and the
run is reported, not built with zeros that would pass for unknown parts. Workers are spawn processes at BelowNormal, one engine each, at most
`--procs` (default 4); a replay costs about the run's own CPU to its last labelled screen (5 to 20 CPU-s). The stored runs replay on the engine
the labelling used when `--lib` names that build's native folder (the rows decide whether it did: a differing row refuses the shard).

## How a run is played

**New runs** (`--source new`): the game seed is the generation's n-th seed of the job's own namespace (`run-teacher-v1`: ten characters of
the lab's seed alphabet, never a seed of the lab's blocks, the exam or the gym), the agent's streams `[AGENT_SEED, 1,000,000 + n]`. The run
is the player's own run loop (`gen0.play_emu_run`'s) with the **guarded** teacher deciding every covered screen below the Act 2 boss: it
takes its own pick only when the measured gain over the player's pick is over two standard errors, so about three screens in a hundred
change (measurement, 1,089 screens), and every other screen is the player's. A teacher that agrees with the player everywhere is the
player's run (a test). The run's RunRecord goes through `gen_loop.RunRecorder` into `jobs/<job>/shards` in the lab's layout, so the run
value refits from these runs (`gen0.py vrun-fit --shards-root <store>/<generation>/jobs`).

**Stored runs** (`--source stored`, `label`): the lab's stored exploring runs of generation 13 (`hindsight.exploring_runs`: the job folders
of 2026-11-05 to 11-08 that named `--explore-tau`) are replayed with `hindsight.replay_run`: the same streams, the exploring player's
picks redrawn, every decision checked against the stored row (cards, rests, map path, final floor); a replay that leaves its record stops
there. At each covered screen a hash of the seed and the ordinal decides whether to label it (`--label-rate` 0.2, at most `--per-run` 5
screens; the replay stops after the last): the run is the stored run, the teacher only reads it; the record's `player` is the argmax
player's pick at that state. Two stored runs replayed in the smoke matched their records through 70 and 78 decisions to the last floor.

**Adaptive K** (`--k` 2, `--k-max` 4): two draws of every shortlisted option (up to four options, `--max-options`) first; two more only
where the top two options (by mean return plus the prior's share on a blended screen) are within two standard errors, the guard's own
error (the paired variance pooled with the kind's prior, `guard_se`). The draws are the same for every option (common random numbers: the
seed and the streams are functions of the run, the decision and the draw only), and the first two of a four are the draws a fixed K 4
takes, so a shard's records are comparable with the measurement's. With the guard's prior an error at K 2 is about 2.2 floors, so two
errors are 4.4 floors and only decisions with a gap over that stop at two draws; the smoke and the first day's `k_extended` count say how
many that is.

**P(death) recalibrated per act**: at each boss the run reaches, the job records the judge's P(death) at the door beside the fight's
outcome. At every job start the pairs of the store (act, judge's P, died) fit, per act, `P' = sigmoid(a + b logit P)` with a ridge of 20
pseudo-observations towards the judge as it stands, and only from 60 doors of that act; the leaf `G = P' x boss floor + (1 - P') x B[act]`
uses it from then on, and the shard's manifest says which map played. `--labels-calibration DIR` seeds the pairs from the hindsight
labels' door outcomes for the first days, only when the labels' own manifest names the judge the job runs with. Plain player runs (`--kinds none`,
about 20 to 40 CPU-s a run) are a cheap source of doors.

**The recalibration is per judge** (D58 addendum 24). A P(death) is one judge's own number, so a map `(a, b)` fitted on v0's pairs says nothing about
v2b's, and the fit reads only the pairs recorded under the judge the job runs with. Every run row now carries `judge_sha256`, the sha256 of the judge
file that made its door numbers (`None` for a job with no judge); a row with no such field was written before the field existed and is v0's
(`run_teacher.V0_JUDGE_SHA256`: every shard manifest of the store to that day names v0). A judge with fewer than 60 doors of an act in the store
starts from the ridge alone, that is, keeps its own number at that act until it has them. A job's start line says what it used, for example
`door recalibration under judge v2b/judge-board.pt (62bcaf0d): doors used by act none, 58 doors recorded under other judges left out`, and
the block's `result.json` carries the same as `recal` (`judge_sha256`, `doors_by_act`, `doors_other_judges`, `fitted_acts`). Two files that read
identically (v2b's `judge.pt` and `judge-board.pt`) are two judges here, by hash: their pairs are kept apart.

## Cost (the measurement's figures, and the first smoke's)

The measurement's final figures (K 4, M 4; `referee/backlog/run-self-play/BUILD-PLAN.md`): 1,383 CPU-s a run, 43.8 screens a run, 33.5 CPU-s
a screen on 24 runs played with the machine held, and 2,342 a run, 48.7 screens, 48 a screen on 26 runs played while it was shared (about
1.45 times the CPU-s on a loaded machine). At 24 core-hours a day that is about 60 teacher runs and 2,700 screens a day on a held machine and
about 37 runs and 1,800 screens on a shared one. Round 0's
stored runs cost the same per screen (the continuations are the cost).

**The first smoke** (2026-11-11, a scratch lab folder, 2 processes beside the corpus builder, a judge board and the practice loop; BelowNormal):
two new runs of the current player stopped at floor 9, K 2 first and up to 4 (`--stop-floor 9`): 31 teacher screens, 1,237 CPU-s on them,
**39.9 CPU-s a screen** against the measurement's 33.5 held and 48 shared (the smoke's screens are floors 1 to 9, whose play-outs run
furthest to the door, on a machine that was shared), 621 CPU-s a run to floor 9 (a whole run is 1,400 to 1,700). Adaptive K: 27 of the 31 screens went on to four
draws (87%), so it saved about 6% of the CPU here, not more: at K 2 the guard's error is 2.2 floors and only a gap over 4.4 floors stops
the draws. The teacher changed 1 screen of 31 (3.2%; measurement 2.8%), its raw pick differed on 52% (39%). One run lost its first attempt to
`select.enchant` (the judgement has no row for an option the rules policy decides: fixed, see above) and was replayed from its seven stored
screens to the same place. `label` on two stored exploring runs: the replays matched their records at every decision checked, and labelled 4
screens at floors 1 to 2 for 271 CPU-s (68 a screen). The shard of both jobs loaded in `run_student.load_shards`: 34 decisions kept of 35
(the one with an option that has no row is counted incomplete), 106 rows, 4 runs, 7 kinds. A plain player run (`--kinds none`) is 38 to 48
CPU-s here, with the door's P(death) recorded.

## Checks and failure modes

* `python rl/run_student.py check --shards-root <store>/<generation>` loads every shard under it with lane 3's loader (the smoke's shard
  passed it).
* `report.json` in the store (and `python rl/run_teacher.py report --out <store>/<generation>`) is the measurement's report over every
  decision: CPU-s a run and a screen by arm, changed share by kind and act, split-half agreement, noise and spread, lambda.
* A guard that holds the job is said on the job's output (`held (<reason>)`) and in `result.json`; a job the runner starts that finds
  itself held exits 0 with `held` in its result.
* A worker that dies is a capped row (`cap_reason`) and the run is played again, from its stored screens, by the next job. The runner's
  alert after two failed jobs in a row (`LAB/alerts/teacher.txt`) applies to the kind like any other.
* The shard never holds an absolute path, a host or user name, or a human record; the manifest's paths are cut to their last two parts.

## Files

`rl/run_teacher.py` (the job, the shard, the calibration, adaptive K, the stored-run source, `reencode`), `tools/lab/jobs.py` (kind `teacher`, the rules,
the ledger), `tests/python/test_run_teacher.py`, `tests/python/test_lab_jobs_teacher.py`, this file.
