# The run student `pi_s` and its gate (`rl/run_student.py`)

Written 2026-11-11 (lane 3 of the learning-system build plan, D56 and D57 part 3; checkpoint C4). Source of the human record the
human term reads: Spire Codex public run export for v0.107.1, through the human prior's own shards; aggregates only, no cell under 20
runs is shown, the exam fifth is never read. The teacher's shards hold game seeds of the copy's own runs, never a human record.

## What it is, in plain words

Today the run-level choices (which card, which relic at an ancient, heal or smith, which route, what to buy, which event option) are made
by the run judgement `V` plus, on card, ancient and rest screens, half a vote from the human prior `pi_h`: `V + 0.5 log pi_h`. Routes,
shops, events, relic rewards and deck selections get `V` alone. Nothing improves `pi_h` from the copy's own play, and the average human
caps it.

The run teacher (`rl/run_teacher.py`, lane 2) plays each option of a screen forward, K times each on resampled clones of the run, and
writes down how many floors each option was worth (`Q`) and how sure that number is. The student is the prior's own net, **started from
the prior's weights and taught those numbers** (and still shown the human choices, so it does not forget them). Where it predicts the
teacher better than the prior does and is not worse than the prior on held-out human choices, the candidate gives the student that kind
of decision. Like a trainee who learns from the senior's marks but is still examined against the textbook.

## The data it reads (lane 2's shard, fixed in `referee/backlog/run-self-play/BUILD-PLAN.md`)

A shard folder holds `decisions.jsonl` (one record per screen: `seed, ordinal, type, kind, act, floor, hp, max_hp, n_options, played,
player, teacher, teacher_raw, k, screen, lam, v_values, logp, g, q, ...` and `row0`), `rows.npz` (`rows` float32 (rows, 1863) as
`human_prior_infer.option_rows` builds them at play, `decision` int32 the record's line, `option` int16 the option's index) and
`manifest.json`. `load_shards(root)` takes every folder under `root` that has a `decisions.jsonl`.

* **Refused** (`ShardError`, nothing is read past it): a missing file; a record that is not version 1; a bad line anywhere but the last (a
  last line cut by a stop is dropped and counted); rows of another width or not finite; a row naming a decision or option that is not there;
  an option with two rows; a `row0` that is not the decision's first row; manifest counts the data does not have; a played option out of
  range; a `q` that is not finite or names an option not played. With a prior given, the log-probabilities the teacher recorded on the
  screens it blended must equal the prior's own on those rows (tolerance 0.001): a shard played with another prior is refused.
* **Dropped and counted by reason** (`stats["dropped"]`): the teacher refused the screen; a kind the student does not take (`bundle`);
  fewer than two options with a value; an option with no row (the judgement cannot score it); a lambda of infinity.
* What stays is a decision whose every option has a row and at least two options a `Q`. A decision with an unscorable option is not
  learned from, which is also when the policy cannot use the student at play (`student_kind_of` needs every option scorable).

The measurement's 1,343 records (`referee/backlog/run-self-play/measure/decisions.d/`, 50 runs) are in this shape without the rows; the
tests give them synthetic rows and check the loader, the targets and the temperature on their real shapes (a rest screen of 17 options
with 4 played, shops of 29, selections with an unscorable option).

## The fit

The net is `PriorNet`'s: the run judgement's embeddings, two 256-wide layers and one logit per option row, loaded from the prior's
`weights.npz` (before any step it is the prior, logit for logit; a test holds it). A step takes a batch of teacher decisions and a batch
of human decisions of one kind, and minimises

```
CE( softmax((Q + lambda log pi_h) / tau_T)  over the played options ,  softmax(pi_s) over the same options )
  + beta * CE( human choice , pi_s )                                   (the prior's own loss per kind)
  + kl   * KL( pi_h || pi_s )        over all the screen's options, on both batches
```

* **`Q`** is the mean of the K play-outs (floors). **`lambda`** is the lambda the teacher decided the screen with: the player's lambda
  where the player blends the prior (`screen` set in the record: card, ancient, rest), **0 elsewhere**. The measurement's records carry
  `lam` 0.5 on every kind, but only the blended screens apply it, so the loader reads it only where `screen` is set.
* **`tau_T`** is the sd of the mean of a decision's K draws of a centred option value: the pooled noise sd of one draw
  (`run_teacher.noise_and_spread`, by kind and act when a cell has 20 decisions, else the kind's, else all) over the square root of the
  decision's K, never under 0.25 floors. The measurement's figures (noise sd 2.1 in Act 1, 1.6 in Act 2; spread between options 0.9 and
  0.7) make `tau_T` about 1 floor at K 4 and the targets soft: a decision whose options the play-outs cannot tell apart gives a flat
  target, and the student is not taught a pick the teacher itself could not repeat.
* **Options off the shortlist** (the teacher plays at most four) have no `Q`: they carry no target and sit outside the softmax of the
  cross-entropy; they are in the KL term, which keeps their logits near the prior's.
* **The human term** is `human_prior.Gpu`'s loss for card, ancient, rest (kind and smith target), shop (buy or leave per offer) and route,
  with the event kind as a softmax over the page's options, over the v1 shards (true map in every row) of fifths 1 to 3, replay
  mismatches and unverified shops out as the prior's own fit leaves them. Kinds the human record has no fit for (`reward`, `select`)
  have only the teacher's term. `check_no_exam` runs on every dataset the human term or leg 2 loads: it refuses the exam bucket as a
  request (`load_human` of fifth 0 never reads a shard), a decision in bucket 0, and any run in the ids files or the hash bucket.
* **Defaults**: `beta` 1, `kl` 0.1, AdamW 3e-4, weight decay 1e-4, dropout 0.1, gradient clip 1, batches of 128 teacher and 256 human
  decisions, 8 epochs, early stopping at patience 2 on the cross-entropy of the held-out `val` runs. They are first values, tuned on
  nothing; the first real fit's legs decide whether they move (a larger `beta` or `kl` holds the human bar, a smaller one lets the
  student follow the teacher further).
* **Splits of the teacher's runs, by a hash of the game seed** (`split_of`): a fifth (`test`) is held out for leg 1 and lambda, a tenth
  (`val`) stops the fit early, the rest train. No run is in two parts.

## The legs and lambda (on data the fit never saw)

**Leg 1, teacher imitation** (`leg1`): on the `test` runs, per kind (and per act where a cell has 20 decisions), the cross-entropy of the
teacher's target under the student and under the prior, the gain (prior minus student, nats), its 95% interval by resampling whole
runs, and the target's own entropy (no cross-entropy can go below it, so it bounds the headroom). A kind passes leg 1 with at least 100
held-out decisions and the gain's lower bound above 0.

**Leg 2, the human bar** (`leg2`): on fifth 4 (never trained on, never the exam), per kind that has human choices, the student's mean
log-loss minus the prior's, on the **won runs** and on **all runs**, each with a 95% interval by runs. The margins are D55 addendum 15's
(`strong_sets.TIE_NATS`, `LIMIT_B_NATS`, a test holds them equal): the student may be worse than the prior by at most **0.002 nats on
the won runs and 0.02 on all runs**, and may be better. A cell over fewer than 20 runs is not shown (hidden cell: the kind has no human
fit). **A kind with no human fit hands over on leg 1 alone**: `reward` and `select` always (no human choice exists), and any kind the
held-out human data has no decisions for (events, when the shards were built without `--events`). A kind that does have a human fit and
whose leg 2 was not measured (no `--human-shards`) does **not** pass.

**Lambda per kind and act** (`measure_lambda`): `run_teacher.lambda_fit` over each held-out decision's played options' `(Q, V,
log pi_s)`: how many floors of the teacher's value the student's log-probability explains beyond the judgement `V`, with the standard error
clustered by decision. The student's log-probability is over all the screen's options, as at play. A (kind, act) cell with fewer than 30
decisions takes the kind's pooled value, a kind with fewer than 30 has none (it cannot pass); the value is clipped to [0, 2]; Act 3 takes
Act 2's (the teacher decides no Act 3 screen, D38). The recipe's `lambda` is this number, not a knob. For reading it: the measurement's
`lambda_fit` for `pi_h` itself (floors per unit of log-probability beyond `V`) was 0.06 over all decisions, 0.23 (se 0.05) for rest in
Act 1, near 0 for cards, against the 0.5 the recipe uses today.

**The verdict** (`kind_verdict`, `passing_kinds`): a kind is handed over when it passes leg 1, leg 2 where it has a human fit, and has a
lambda. A kind the base decides by `V` alone whose measured lambdas are all 0 is left out (the student there would be `vrun` again);
card, ancient and rest at lambda 0 are kept, because they replace the prior's blend by `vrun`, which is a change.

## The recipe and the policy

```
vrun+pi_s:lambda=<L>:pi_h=<8 hex>[:act<N>=<L>...]:pi_s=<8 hex>:<kind>=<l1>/<l2>/<l3>...[:pick=...]
```

It is the human-prior recipe (`macro_recipe.macro_name`) with the student's sha256 (8 hex) and one `<kind>=` part per kind the student has
taken over, in the order `card, ancient, rest, route, shop, event, reward, select`, each with its lambda in Acts 1, 2 and 3. A recipe
without the parts is the recipe it always was, byte for byte; `parse_macro`, `parse_act_lambdas` and `parse_pick` read the blended part of
a student recipe as they read a plain one's; `is_blended` is true for both. The ledger counts attempts per recipe, so each student (and each
lambda) starts at attempt 1.

`imitation_chooser.BlendedPolicy(..., student=, student_lams=)` decides a kind the student has taken over by `V + lambda_act log pi_s` with
the same blend, guard and sampling as the prior's, over every option of the screen when the judgement scores them all; lambda 0 for the
kind and act is `vrun` (`LookaheadPolicy.pick`); every other screen is decided as before. The guard's probability is
`min(0.02, 0.25 / n)` for a screen of `n` options (a shop of 29 offers: 0.0086, so no option is ruled out for sitting under a fixed 2 in
100). The run row's `decided` entry says `s:<kind>` where the student decided.

## The candidate and `screen:student`

`run_student.py candidate` (`build_student_candidate`) calls `gen_loop.build_candidate(..., student=, student_lams=)`: the combat prior,
V fit, human prior and its lambdas, pick mode, stakes schedule and potion hold are **the base's**, the student is copied as
`run_student.npz`, `agent.json` gets a `student` block (file, sha256, kinds and lambdas, the kinds the file was fitted for), and the legs
are stored beside it as `run_student_legs.json`. The loader (`load_agent`) holds the folder to the recipe: the student's bytes by the 8
hex and the full sha256, the kinds against `agent.json`, the kinds against what the file was fitted for. The base may be a vrun
generation (then give `--human-prior` and `--lambda`), a blended one or an earlier student's (its blend is kept, its student replaced).

`screens.kind_of` reads a candidate that differs from the incumbent in the student (and the recipe naming it) alone as `screen:student`;
`run_student` reads the stored legs for the kinds the recipe names and **rejects** when any fails the verdict above (a kind the legs do not
cover is a rejection); it skips, saying why and playing the arms, when the legs are missing, are not for these bytes, or were measured
against another baseline than the incumbent's (its student's if it has one, else its human prior). **A candidate with a student is never
read by `screen:prior`** (a student is fitted to the teacher, so "better than the prior on the humans' choices by 0.002" would fail it by
design) **or `screen:fight`**: it has the student's screen or none. The ledger's rule column reads `screen:student gain=+0.034 won=+0.001
all=+0.008` (the decision-weighted leg-1 gain, the worst leg-2 differences). After the screen the judgement is the lab's (keep test v2,
futility from 500, sequential); the run-level human bar is reported beside it.

## The V refit

The run judgement's refit from a window of runs already exists (nothing is built here for it; `gen0.py train --shards-root` fits V_run as part of a
trained candidate): `python rl/gen0.py vrun-fit --shards-root <LAB/runs> --gen-dir <current.json>
--out <dir>` (the D33 out-of-fold keep test over five FightNet and five V fits) and `gen0.py build --vrun-fit <dir>`. It is a candidate of its
own and not part of a student candidate: lambda is measured against the incumbent's `V`, so a student and a new `V` in one candidate
would have a lambda for a judgement it is not played with.

## The first real fit (one command, when 10,000 screens exist)

```
D:/spireline/venv-gpu/Scripts/python.exe rl/run_student.py round \
    --shards-root D:/spireline/referee/run-teacher/<generation> \
    --prior <the incumbent's human_prior.npz> \
    --human-shards D:/spireline/referee/human-prior/decisions-v1 \
    --out D:/spireline/referee/run-student/<generation>-r1 \
    --base D:/spireline/referee/agent/current.json --agent-root D:/spireline/referee/agent --target <N+1>
```

`round` is `fit` (the shards, the fit, `student.npz`, `fit.json`, `legs.json`) then `candidate` (when a kind passed: exit 3 and no candidate
when none did; `--no-build` stops after the legs). About ten minutes of the card for the fit (the prior's own took 171 s on about 200,000
decisions) and under 30 CPU-minutes for the legs. It runs at BelowNormal; take a `run-student <UTC> pid=<Windows PID>` line in
`referee/QUIET.lock` first when it is run by hand (the lab's `student` job, lane 4, holds its own). `check --shards-root DIR` prints the
counts (decisions by kind, split, dropped reasons) without a fit; that is what the job's "10,000 new screens" test reads. The lab's
`train` job then judges the candidate through `ready-<generation>-g<N>-student`.

## Not built, not verified

* **No fit on real data**: lane 2's data does not exist yet. Every figure above that is a measurement is the design lane's 50-run
  measurement's; the fit, the legs and the screen are tested on synthetic shards (a planted teacher) and on the measurement's records
  with synthetic rows. The one-minute smoke fit on the card is reported in the HANDOFF line.
* The shard's manifest key names are lane 2's to fix; the loader checks `counts.decisions` and `counts.rows` when present and keeps a
  cleaned copy of the provenance keys it knows (`agent`, `judge`, `engine`, `k`, `max_options`, `beyond`, ...).
* The shop and event kinds mix two shapes: the teacher's decision is a whole shop screen with up to 29 options, the human record's a
  buy-or-leave per offer (shop) or a page of two or three options (event). Both train one logit per afterstate row; whether the mix helps
  or hurts is what leg 2 measures on the first fit.
* `beta`, `kl`, the epoch count, the 100-decision floor of leg 1 and the guard's scaling are first values.
* `run_teacher.lambda_fit` and `noise_and_spread` are local copies (marked) until `rl/run_teacher.py` is on main; `teacher_fn` returns the
  teacher's own the moment it imports, and a test holds the two equal while both exist.

## Tests

`tests/python/test_run_student.py` (targets and temperature, the loader and each refusal, the measurement's records, the split, the exam
refusals, the net equal to the prior, the fit on a planted teacher, leg 1, the KL term, leg 2, the verdict's margins, lambda recovering a
planted value, a whole round, the command line), `test_imitation_chooser.py` (the recipe grammar and the byte-for-byte plain recipes, the
student's screens and guard, the policy with a student), `test_screens.py` (`screen:student`: kinds, verdicts, skips, the promote path,
never `screen:prior`), `test_gen_loop.py` (`build_candidate` with a student, the loader's refusals, `build_student_candidate`).

The one-minute smoke fit on the card (synthetic shards written by a scratch script, 4,800 decisions of 400 runs and four synthetic human
fifths; the GPU venv) ran the whole `fit` command: load, fit (7 s on the card), both legs, lambda, `legs.json`; no path or run id in
its files.
