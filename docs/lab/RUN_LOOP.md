# The run loop in the lab (`teacher` -> `student` -> `train`, `tools/lab/jobs.py`, `runner.py`, `alerts.py`, `rl/run_human_bar.py`)

Written 2026-11-11 (lane 4 of the learning-system build plan, D56, D57 part 3, D58 C4 and C5). The teacher is `docs/lab/TEACHER_JOB.md`, the student
`docs/rl/RUN_STUDENT.md`, the design `docs/rl/RUN_SELF_PLAY.md`. Human figures: Spire Codex public run export for v0.107.1, aggregates of the training
fifths; the exam fifth is never read and no cell under 20 runs is shown.

## What it does, in plain words

The lab's runner already ran a judge for each new player (the `train` job). This lane makes the runner also *make* the new players, by itself.
The teacher plays the current player and writes down how many floors each choice was worth. When about ten thousand of those notes have piled up, the
student (a copy of the human prior, taught from the notes) is fitted and tested, and if it is good enough on any kind of decision a candidate player is
built from it and handed to the judge. If the judge promotes it, the next teacher block plays the new player, and the round starts over. Like a
school that sets its own homework, marks it, runs the exam and, when a pupil passes, promotes the pupil to teach the next year: with a few stop signs
so that it halts itself when it goes wrong instead of burning the machine.

## The chain

```
teacher job (priority 8, the lab's last slot; round 0 by hand)   writes shards under referee/run-teacher/<generation>/shards/<job>
   |   10,000 new labelled screens since the last fit (counted by `rl/run_student.py check`), or the weekly fit (>= 1,000 new)
   v
student job (priority 3.5, right after train)                    fit + legs + candidate; writes LAB/train/ready-<gen>-g<N>-student
   |   judgement first: a waiting train marker holds the student, and the marker is what train takes next
   v
train job (as for any candidate)                                 the lab's judgement: smoke look, futility, sequential, keep test v2
   |   a promotion rewrites agent/current.json
   v
the next teacher block reads agent/current.json at its start     the new generation's self-play and labels follow without a builder
```

* **The ladder** (`runner.ladder_top`): nightly-real, score-real, train, **student**, the self-play share, the benches, **teacher**. The teacher is
  last (the filler's slot): a bench that is ready, a train marker, a due student all come first. A running teacher block ends within its `block_min`
  (60) when a marker appears (lane 2's rule), so the student waits at most one block.
* **The student job's trigger** (`jobs.student_due`): `new_screens` (10,000) new labelled screens since the last fit, **or** the weekly fit (`weekly_days`
  7 since the last fit, or since the loop started when there has been none) with at least `weekly_min_screens` (1,000) new. The count is two-stage:
  the shard manifests' `counts.decisions` (cheap; read at every pick through two fixed-depth globs, because the store holds thousands of per-run files) say
  when `rl/run_student.py check --shards-root D:/spireline/referee/run-teacher` is worth asking, and `check`'s `decisions` (what the loader keeps) is the
  figure the trigger uses. `check` runs the lab tree's own `rl/run_student.py` (the entry's `cwd`, `{tree}`: its `out/` is rebuilt at a presort; the runner's
  checkout is not, and its library went stale at an engine landing, which silently held the loop from 11-12 to 11-13; `check_cmd` overrides). It is asked again
  only when a shard changed. A student whose recorded screens meet the trigger but which is held (a failed `check`, a QUIET line, a waiting marker) says why once
  per reason in `LAB/runner.log` (`student: due but held (...)`), the teacher the same (`runner.note_held`). The fit's own count is stored as the baseline (`LAB/student/state.json`
  `last_fit`): a round that built nothing still counts as the fit, so the next one wants 10,000 *more*.
* **The student job's holds** (`jobs.student_hold`, also the runner's `unready`): the filler's rules (HALT, PAUSE, the night or a start within 30 min of
  it, a live QUIET.lock line outside `quiet_ignore`: `gym-loop` and `lab-score`), judgement first (any `LAB/train/ready-*` marker: one candidate is judged
  at a time), `LAB/alerts/run-student.txt` standing, and an incumbent it can start from (`agent/current.json` -> its student, else its human prior). It
  holds no QUIET.lock line (the existing filler kinds hold none). One process on the card, `cores` 4, BelowNormal (the runner's own class and
  `run_student._priority`).
* **The command** is `tools/lab/student_job.py`, run by the GPU Python: `rl/run_student.py`'s `run_fit` and `build_student_candidate` (lane 3) with three
  differences and nothing else: a kind in `LAB/student/frozen.json` is left out of the loader (so out of the fit and the candidate); it writes
  `result.json` (the per-kind verdicts, the candidate's folder under the agent root, the screens the fit used) and `run-bar.md`; and it exits 3 when no kind
  passed both legs (a finished round: the kind's `ok_rcs` is `[0, 3]`, so the runner does not count it a failure; every other nonzero code is one).
  The runner, once the process has exited cleanly, writes the marker **`LAB/train/ready-<gen>-g<N>-student`** (`N` = the incumbent's generation + 1; a
  second round at the same number is `-student-r2`, and so on: the first tag with no candidate folder and no marker, ready or done) and records the
  round; the marker is the candidate `gen<N>/candidate-student` that `gen0.py train --marker-dir` judges as any tagged candidate (`screen:student`, lane 3).
* **No builder.** A promotion is `current.json`; `teacher` and `student` both read it when they start. The student's first fit needs the 10,000
  screens of round 0 (hand-started, `D:/spireline/referee/run-teacher/gen13-r0/13/`) and the daily job's; `--shards-root` is the store root, so both are read.

## The stops

| stop | trigger | what it does | where it is written | reset |
|---|---|---|---|---|
| a kind is frozen | the kind has a human fit and its leg 2 (the human bar: won runs within +0.002, all runs within +0.02 of the prior) failed in **2 rounds running** (a pass resets the count; a round that did not measure the bar for the kind, for example a cell under 20 won runs, which lane 3 reports as leg 2 `False`, leaves the count as it was: the job marks it `leg2_measured` false) | the kind is left out of the fit and the candidate; the other kinds go on | **`LAB/student/frozen.json`** (`kinds: {kind: {since, round, failures, why}}`), the line in `LAB/student/verdicts.md`, the count in `LAB/student/state.json` `fails` | the orchestrator deletes the kind's entry (or the file): its count starts again at the next round |
| the student stops | **every** kind with a measured human bar failed it in 2 rounds running | the runner no longer picks `student` (the teacher goes on labelling) | `LAB/alerts/run-student.txt` (the two rounds' verdicts per kind), so also the alerts line at the top of status.txt, clock.md and summary.md | remove the file by hand: the student runs again and the count of failing rounds starts over (`run_loop_tick` notices the removal) |
| plateau | **10 days** without a run-loop promotion (a student candidate the train job promoted); the clock starts at the loop's first tick and at each promotion | a note only: nothing is held | `LAB/alerts/run-plateau.txt` with the last three rounds' verdicts | remove the file by hand: the ten days count again from then |
| a student job crashes | any exit code but 0 and 3, twice in a row | the runner's existing alert: `student` is no longer picked | `LAB/alerts/student.txt` (the runner's own, as for every kind) | remove the file |
| judgement first | a train marker is up | the student (and the teacher's next block) wait for the judge | -- | the marker is consumed |

A kind with no human record (`reward`, `select`) never fails the human bar and so is never frozen; it is judged on leg 1 and the lab's judgement. The
fidelity-drop stop of the memo (a real-game block's fidelity falling) is **not built** here: the plan's lane 4 does not list it.

`tools/lab/jobs.run_loop_tick` (called by the runner at every pass and after each teacher, student or train job; a no-op while neither kind is
enabled, and in `--once`, `--trial` and `--dry-run`) starts the loop's clock, notices an alert removed by hand, raises the plateau note, writes the daily
line and refreshes the status text.

## The daily line and the status files

`LAB/run-loop.md`: one line per closed UTC day, the last 90 kept:

```
2026-11-12 teacher: 37 runs, 1,648 screens, 22.9 core-hours (budget 24/day); student: 0 rounds, no fit yet, frozen -; last run-loop promotion none
```

from the teacher's ledgers (`<store>/ledger.jsonl` and the hand-started round 0's `<store>/gen13-r0/ledger.jsonl`: round 0's labelling is in the day's
figures but not under the daily budget). The last seven lines are also at the bottom of `LAB/clock.md` (the runner's daily report), and `LAB/status.txt`
carries today's figures after its line while the loop is on: `| run-loop today: teacher 12 runs 530 screens 8.4 core-h; student 0 rounds, frozen -`.
Each student round and each judgement adds a block to **`LAB/student/verdicts.md`**: the legs per kind and, beside them, the run-level human bar.

## The run-level human bar (C5): `rl/run_human_bar.py`

What the humans reach at the doors of the Act 1 and Act 2 bosses, to read a candidate against: floor reached (map points entered; the floor a run died
on, or won through) and, for each of the two bosses, the share of runs that reached it, the share of those that killed it, and the HP the run carried
into the fight (the last point before the boss: the rest site's own figure), in HP and as a share of max HP. Computed once from the four non-exam
fifths of the corpus (`python rl/run_human_bar.py compute`: about six minutes, one BelowNormal process), aggregates only, stored in
**`D:/spireline/referee/human-bar/run-level.json`** (`--force` to redo; no run id, no seed, no path in it; a cell under 20 runs is `null`).

The stored figures (11,961 Ironclad A10 runs of the four non-exam fifths, computed 2026-11-11; 2,968 exam-fifth runs left out before any figure):

| figure | Act 1 boss | Act 2 boss |
|---|---|---|
| runs that reached the boss | 8,840 (73.9% of all runs) | 4,559 (38.1%) |
| of those, killed it | 6,528 (73.8%) | 3,452 (75.7%) |
| HP carried into the fight, mean (median; middle half) | 54.2 (53; 43 to 65) | 61.0 (60; 48 to 73) |
| the same as a share of max HP | 66.7% | 72.2% |

Floor reached over all runs: mean 27.1, median 24, 10th to 90th percentile 8 to 49; 11.4% won. (The copy's generation 13: mean floor about 24.8 on the
judgement arms, Act 1 boss killed by 65% of those reaching it.)

`beside` / `beside_text` put the copy's figures next to them: from RunRecords (the teacher job's recorded runs under `<store>/<generation>/jobs/*/shards`,
which carry the HP at the door) after a student round, and from a candidate's judgement arms (`arms/*/runs.jsonl`: floor and boss reached and killed,
no HP) after the judgement. The exam fifth is never read: `human_prior.iter_run_lines` drops it before this code sees a record (the ids file and the hash
bucket 0) and `human_starts.assert_no_exam` stops a figure that would hold one.

## The two jobs.json fragments (disabled) and the one restart

Both kinds are in `tools/lab/jobs.py` `DEFAULTS` (off), so a lab without the fragments changes nothing. The fragments write the entries out, for the
orchestrator to merge into the live `LAB/jobs.json` (top level, beside `selfplay`, `train` ...):

```json
{
 "teacher": {
  "_note": "2026-11-11 (lane 2 + lane 4, D56/D57): the run-level teacher job; the runner's ladder picks it after the benches. Off until the orchestrator switches it on; round 0 (label) is run by hand.",
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
 },
 "student": {
  "_note": "2026-11-11 (lane 4, D56/D57/D58): the run student: fit, legs and candidate when 10,000 new labelled screens exist (or weekly), marker ready-<gen>-g<N>-student for the train job. Off until the orchestrator switches it on. State, freeze and verdicts under LAB/student/.",
  "enabled": false,
  "cwd": "{tree}",
  "timeout_min": 180,
  "procs": 4,
  "cores": 4,
  "gpu": true,
  "new_screens": 10000,
  "weekly_days": 7.0,
  "weekly_min_screens": 1000,
  "quiet_ignore": ["gym-loop", "lab-score"],
  "ok_rcs": [0, 3],
  "shards_root": "D:/spireline/referee/run-teacher",
  "human_shards": "D:/spireline/referee/human-prior/decisions-v1",
  "agent_root": "D:/spireline/referee/agent",
  "check_timeout_s": 900,
  "cmd": [
   "D:/spireline/venv-gpu/Scripts/python.exe", "tools/lab/student_job.py", "--lab", "{lab}",
   "--shards-root", "D:/spireline/referee/run-teacher", "--prior", "{student_prior}", "--teacher-prior", "{student_teacher_prior}",
   "--human-shards", "D:/spireline/referee/human-prior/decisions-v1", "--out", "{job}/student",
   "--base", "D:/spireline/referee/agent/current.json", "--agent-root", "D:/spireline/referee/agent",
   "--target", "{student_target}", "--tag", "{student_tag}", "--records", "D:/spireline/referee/run-teacher",
   "--result", "{job}/result.json"
  ],
  "needs": ["{tree}/tools/lab/student_job.py", "{tree}/rl/run_student.py", "D:/spireline/venv-gpu/Scripts/python.exe"]
 }
}
```

`{student_prior}`, `{student_teacher_prior}`, `{student_target}` and `{student_tag}` are filled when the job is made (`jobs.student_params`): the
incumbent's student file (else its human prior) from `agent/current.json`; the incumbent's human prior (`agent.json` `human_prior.file`), which the teacher
decided with, when the fit starts from the student's file (a start from the human prior itself has none: the token and `--teacher-prior` before it are left
out); the incumbent's generation + 1; and the first free tag. Nothing else of `jobs.json` changes: the fragments touch
neither `selfplay` nor `train` (whose `--marker-dir` already takes `ready-<gen>-g<N>-<tag>` markers).

**The edit the orchestrator makes** (outside a night; `LAB/nightly/active` absent): back up `LAB/jobs.json` as `jobs.json.<date>-run-loop.bak`, add the two
keys above at the top level (the two `_note`s say what they are; add a dated sentence to the file's own `_note`), then set `"enabled": true` in each when
the loop is to run: `teacher` once round 0's hand-started job has ended (they would otherwise share the machine; round 0 holds 12 processes), `student`
when the first 10,000 labelled screens exist (it is harmless earlier: it waits for them). Then **restart the runner once** (the `Spireline fix lab`
scheduled task): the runner reads `jobs.json` at every choice, but the new ladder, the loop's upkeep and the `student` kind are code, which a running
runner has not loaded. The restart brings the code in from the fork's main checkout; the lab tree (`--tree`) gets `tools/lab/student_job.py` at the
next presort, which a moved main brings, and the student job's `needs` hold it back until then. Nothing else is restarted (the gym daemon and the
hand-started round 0 are not touched).

## Files and tests

`tools/lab/jobs.py` (the `student` kind, the hold, the trigger, the stops, the daily line; `_filler_basics` now shared by the teacher's hold),
`tools/lab/runner.py` (the ladder: `student` after `train`, `teacher` after the benches; `run_loop_tick`; the status and clock text),
`tools/lab/plans.py` (one line: a kind's `ok_rcs` from its params), `tools/lab/alerts.py` (`write_run_student`, `write_run_plateau`),
`tools/lab/student_job.py` (the command), `rl/run_human_bar.py`; tests `tests/python/test_lab_jobs_student.py` (the kind, the trigger and its reset,
the holds, the params and the tag, the finish and every stop with its trigger and reset, the plateau, the daily line, the chain through the runner on a
scratch lab, the fragments), `test_lab_student_job.py` (the command with lane 3's functions stood in for), `test_lab_alerts_run_loop.py` (the two alert
files), `test_run_human_bar.py` (the human pass on a fixture with the exam fifth planted two ways and refused, a cell under 20 hidden, the copy's side).

## Not built, not verified

* **No real fit has run**: the student job's command was run only with stand-ins (the scratch chain) and lane 3's functions stubbed; the first real
  round waits for the first 10,000 labelled screens. After a student is **promoted**, the incumbent's prior for the next fit is the student, and lane
  3's loader checks the teacher's recorded log-probabilities against the prior it is given (`TeacherSet.attach_prior`): the teacher's are the human
  prior's (its pull is `lambda * logp` of pi_h), which a student prior's are not. The job therefore names the incumbent's human prior as the teacher's
  (`--teacher-prior`, `{student_teacher_prior}`, 2026-11-13: the first unattended run refused its store by up to 4.7 nats without it), and the fit starts
  from the student's file.
* The fidelity-drop stop of the memo is not built (not in the plan's lane 4).
* A freeze leaves the kind out of the fit's data and the candidate; the net is one network, so a frozen kind's weights are still the prior's through the
  KL term, not removed.
* The copy's door HP beside a judgement comes from the arms' rows, which carry no HP; the HP side comes from the teacher's recorded runs, after the
  student round, of the incumbent.
