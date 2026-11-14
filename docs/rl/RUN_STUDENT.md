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

## The map rows (`--map-grid`, D58 addendum 13)

The prior and the student see the whole act map, not a summary: each row also comes with block M (`features.map_grid`: 17 rows of 7 columns of
13 numbers: the node type, the three edges to the next row, whether the cell is on a path from the position to the boss, whether it is the
position), and the net has a map tower for it (two 3x3 convolutions, a linear layer to 64, concatenated before the first layer; the first layer
reads 518 numbers instead of 454). The base vector the run judgement reads is unchanged. `HUMAN_PRIOR.md` section "v2: the map grid" has the
encoding and the v2 prior's agreement.

* **Data.** The teacher's shards carry `rows_map.npz` beside `rows.npz` (`grid` int8, one row of 1,547 a row; `docs/lab/TEACHER_JOB.md`);
  `run_teacher.py reencode` adds it to shards played before. The human shards are the v2 ones (`human-prior/decisions-v2`: `grid` and `grid_ix` beside
  each shard's rows).
* **The flag and the prior must agree.** A fit with `--map-grid` needs the v2 prior (it refuses one without the tower), every shard with its
  `rows_map.npz` (`ShardError` names the shards that lack it and says to run `reencode`) and human shards with the grid. A fit without the flag refuses a
  prior that has the tower. A `rows_map.npz` that is not one 0-or-1 int8 row per row of `rows.npz` is refused.
* **`--teacher-prior`.** The target is `Q + lambda log pi_h` with the human prior the teacher *decided with*, whose log-probabilities are in the
  shard (`logp`; the loader checks them within 0.001). The v2 prior is not that prior, so a map-grid fit names the plain prior the teacher played
  with (`--teacher-prior human_prior.npz`): the target uses its log-probabilities, and `--prior` (the v2 prior) is where the student starts, what its
  KL term anchors it to and the baseline both legs measure it against. Without `--teacher-prior` the check refuses the shards (the recorded
  log-probabilities are not the v2 prior's).
* **The student file** carries the tower's six arrays and `map_grid: true` in its meta (`human_prior_infer.write_student`); `StudentScorer` runs the
  tower in numpy and refuses to score a row without its grid. At play `option_rows(..., with_grid=True)` gives the grids beside the rows
  (`imitation_chooser.BlendedPolicy` passes them to a scorer that has the tower; a plain scorer is called as it always was).
* **The legs and the recipe are unchanged**: leg 1 on the held-out teacher runs, leg 2 on fifth 4 (from the v2 shards), lambda per kind and act, the
  verdict, the candidate. A candidate built from a map-grid fit has the v2 prior as the student's baseline: `build_student_candidate` refuses a base
  whose human prior is another file (the legs were measured against the v2 prior), so a base on the plain prior takes `--human-prior <v2> --lambda L`.

```
D:/spireline/venv-gpu/Scripts/python.exe rl/run_student.py round --map-grid \
    --shards-root D:/spireline/referee/run-teacher/<generation> \
    --prior <the v2 prior's weights.npz> --teacher-prior <the incumbent's human_prior.npz> \
    --human-shards D:/spireline/referee/human-prior/decisions-v2 \
    --out D:/spireline/referee/run-student/<generation>-r1-map [--no-build] ...
```

**The first map-grid fit (2026-11-12; `referee/run-student/gen13-r1-map`, beside the summary-rows fit `gen13-r1`).** Round 0's store (`gen13-r0`: 2,188 runs, 10,058 screens, 41,691
rows; `rows_map.npz` added by `reencode`: 2,177 runs replayed, every row built again equal to its stored row, 5.0 core-hours of CPU, 3 hours on 4 processes), the v2b prior
(`HUMAN_PRIOR.md` "v2": the tower with dropout and the mirror, as good as v1 on the humans) as `--prior`, the plain prior the teacher played with as `--teacher-prior`, the v2 human
shards, the recipe's defaults; 3.5 minutes in all (the fit 7 s on the card, seven epochs run, best epoch five). Both fits read the same 9,967 decisions of 2,176 runs and the same 1,971 held-out
decisions, so their cross-entropies to the teacher's target compare directly (nats, lower is better; the floor is the target's own entropy):

| kind | held-out decisions | summary-rows student (prior v0) | map-grid student (prior v2b) | floor |
|---|---|---|---|---|
| card | 515 | 1.2816 | 1.2771 | 1.0146 |
| ancient | 117 | 1.0915 | 1.0971 | 0.7485 |
| rest | 159 | 0.9508 | 0.9642 | 0.7241 |
| route | 486 | 0.7586 | 0.7665 | 0.5331 |
| shop | 143 | 1.2936 | 1.3425 | 0.9304 |
| event | 214 | 0.6784 | 0.6759 | 0.4712 |
| reward | 254 | 0.7232 | 0.7485 | 0.5015 |
| select | 83 | 1.4592 | 1.4398 | 1.0447 |
| all | 1,971 | 0.9856 | 0.9935 | 0.7267 |

The whole map does not help the student imitate the teacher on this data: it is within 0.02 nats of the summary-rows student on every kind but shop (0.05 worse) and reward (0.025 worse),
0.008 worse over all. The legs' verdicts are not comparable, because each is against its own prior: against v0, which was never fitted on routes or shops, the summary-rows student passes
route, shop and event; against v2b, which was, the map-grid student fails leg 2 on every kind with a human fit (route won +0.028, all +0.022; card won +0.024; ancient, rest, shop likewise)
and only event (no human fit) passes, on leg 1 alone at lambda 0.93. Its candidate is `gen14/candidate-mapgrid` (built with `--human-prior <v2b> --lambda 0.5`, which replaces the base's human prior
as well as adding the student: judge it as that, not as `screen:student`); no marker was written.

## The state gaps (`--state-gaps`, D58 addenda 14 and 16)

The state a choice is scored on also holds three things a player tracks that the base vector leaves out (block G, `features.gaps_block`, 36 numbers: the counters of the
relics whose counter the engine reads, the "?" rooms visited this act and the odds the next is a fight, a treasure room or a shop, the merchant's removal quote and the
removals owed, and a **known** bit for each of the three parts). The base vector (1,855, feature version 2) and the value net's input are unchanged: block G is an
extra block only the prior and the student read, as the first layer's last 36 columns (after the map tower's 64 when there is one). `HUMAN_PRIOR.md` section "v3: the
state gaps" has the encoding, how a human record gives two of the three parts, and the v3 prior.

* **Data.** The teacher's shards carry `rows_gaps.npz` beside `rows.npz` (`gaps` float32, 36 a row; `docs/lab/TEACHER_JOB.md`); `run_teacher.py reencode` adds it to shards
  played before, by exact replay. The human shards are the v3 ones (`human-prior/decisions-v3`: `gaps` and `gaps_ix` beside each shard's rows; on a human row the counters are
  unknown, their known bit is 0).
* **The flag and the prior must agree**, as for the map grid: a fit with `--state-gaps` needs a prior that reads block G, every shard with its `rows_gaps.npz` and human shards
  with block G; without the flag a prior that reads it is refused. A `rows_gaps.npz` of another block G version (`gaps_version`), another row count, another dtype or with a
  number that is not finite is refused (`ShardError` names the shard). The flags are independent: `--map-grid --state-gaps` is the v3 prior with both.
* **The student file** says `state_gaps: true` and `gaps_version` in its meta (the columns are `hidden1`'s, no array of their own); `StudentScorer` refuses to score a row
  without its block G, and at play `imitation_chooser` passes it (`human_prior_infer.option_gaps`) to a scorer that reads it. A live (real game) view has no engine state: block G
  is then zeros with every known bit 0.
* **The legs and the recipe are unchanged.**

```
D:/spireline/venv-gpu/Scripts/python.exe rl/run_student.py round --map-grid --state-gaps \
    --shards-root D:/spireline/referee/run-teacher/<generation> \
    --prior <the v3 prior's weights.npz> --teacher-prior <the incumbent's human_prior.npz> \
    --human-shards D:/spireline/referee/human-prior/decisions-v3 \
    --out D:/spireline/referee/run-student/<generation>-r1-gaps [--no-build] ...
```

## The grade (D59: floors, or the frontier)

The teacher's `Q` is in the unit of its **grade** (`docs/lab/TEACHER_JOB.md`, "The frontier grade, the regrade and the door rows"): `floors` (every shard so far: final floors, the judge's chance of dying at a boss
door times the boss's floor plus the chance of surviving times `BEYOND`) or `frontier` (the chance of killing the Act 2 boss, x 13.5, in a store that `run_teacher.py regrade` wrote under `run-teacher-frontier/`). A shard's
manifest says which (`grade`; one with no such key is floors', so no shard written before D59 needs changing, and a record may say it too). **A store is graded one way**:

* `load_shards` refuses shards that carry two grades (a shard's manifest, or a record's own `grade`): the target `softmax((Q + lambda log pi_h) / tau)` would mix two units, and the temperature, lambda and the legs'
  paired differences with it. The error names the grades and the number of shards of each; fit one root at a time (`--shards-root D:/spireline/referee/run-teacher/<g>` or `.../run-teacher-frontier/<g>`, never their parent).
  An unknown grade is refused too. The loaded set names its grade (`ts.stats["grade"]`, `public_stats`, each shard's provenance entry).
* **`TAU_MIN`** (the floor of the target's temperature, 0.25 floors, above) is the grade's own unit's: `TAU_MIN_BY_GRADE` (`floors` 0.25, the value `TAU_MIN` holds and every floors store reads unchanged; `frontier`
  0.071; a grade whose unit is `None` is refused with that said, not fitted with a floors number). The frontier's is the floors' scaled by the ratio of the noise sd of one draw's centred option value in the two
  grades **on the same screens**: `python rl/run_teacher.py units --store <the regraded generation folder>` reports both (a regraded record keeps the floors `g` as `g_floors`) and `tau_min_suggested` = 0.25 x
  (frontier noise sd / floors noise sd), so that the same share of a typical temperature is held back. Measured on the two regraded stores pooled (13,551 screens, 2026-11-13, A2 `a2.npz` sha256 `a5d3da24`):
  0.568 / 2.010 x 0.25 = 0.071 (Act 1 alone 0.194 / 1.996 would give 0.024, Act 2 alone 1.234 / 2.065 0.149: the temperature of an Act 1 decision is its noise over the root of K, about 0.1, so the floor binds there
  only when K is 4 and the noise small).
* **`PAIRED_SD_PRIOR`** (the guard's prior sd of a paired difference, by kind, `run_teacher.py`; used only where the teacher plays, `guard_se`) has a floors table (card 3.41, rest 2.76, route 4.83, shop 2.94, default 3.11:
  unchanged) and a frontier one, `FRONTIER_PAIRED_SD_PRIOR` in `PAIRED_SD_PRIOR_BY_GRADE`, measured by the same `units` command: per kind **and door act** (the unit differs by about six between Act 1 and Act 2: card 0.32 and
  2.09, rest 0.26 and 1.88, shop 0.33 and 2.18), the pooled sd of one draw's difference between two played options (each pair of options of a decision: the sample variance of its K differences on K - 1 degrees of freedom,
  weighted by those degrees; a cell under 20 decisions left out), and `_all` (0.93) for a cell with none. The measuring function reads any store: on generation 14's floors store it is the figure to compare the 11-05
  table with (card 3.62, rest 2.17, route 4.84, shop 3.16); on the regraded stores, the frontier's (`docs/lab/TEACHER_JOB.md`, "The frontier's constants, measured").
* The student's legs, lambda and recipe are unchanged; lambda is measured against V, which is still floors (`V + lambda log pi_s` at play), so on a frontier store it is the new value per nat beyond V in the
  frontier's unit: read it as the design says (`OBJECTIVE.md` section 6, risk 7) before it is given to a candidate.
* **`--target-lambda-scale S`** (D59 lane G, default 1; `fit` and `round`): the teacher's lambda in the target `Q + lambda log pi_h` is the player's, in floors per nat, and the regrade does not rescale it, so on a
  frontier store, where Q's differences between options are about a fifth of floors' in Act 1, the prior term is most of an Act 1 target (a sharpened human prior). S multiplies the lambda of every decision
  (`TeacherSet.scale_lambda`), for the fit and for leg 1's target alike; `fit.json` records `target_lambda_scale` and `legs.json` `target.lambda_scale` when it is not 1 (a fit without the flag writes neither, byte
  for byte as before). The grades' noise ratio, 0.283 (`run_teacher.py units`' `noise_ratio`, the one `FRONTIER_TAU_MIN` uses), is the scale that puts the term in the frontier's unit; it makes
  the Q-driven target LOOK0.md's fit (iv) used. The lambda the legs then measure is still against V in floors (the unit caveat above: in Act 1 it is 0 for card, rest and reward for that reason).
* **The lambda in the store** (D59 lane H): `run_teacher.py regrade --lambda-scale S` writes the scaled lambda into every record (`lam`; the stored one stays as `lam_floors`) and both manifests name it, so a fit on
  the regraded store needs no `--target-lambda-scale` (the loader records `lambda_scale` in `fit.json` and `legs.json` `target.store_lambda_scale`, refuses a store whose shards are in two lambda units, and refuses
  `--target-lambda-scale` on top of a scaled store). The scale is the grades' noise ratio on the stores being fitted together (`run_teacher.py units --store A --store B` pools them and prints
  `lambda_scale_suggested`: 0.283, the first value, unless the measured ratio differs from it by more than a fifth); the regrade's own result carries the same measurement (`unit`: noise and spread ratios, pooled and by
  door act) beside the scale it used. The ratio is a mixture of the acts' (Act 1 about 0.13, Act 2 about 0.65 on the regraded generation 14 and 15 stores), so it moves with the share of Act 2 screens.
* **A start that is a merged student** (D59 lane H; `--prior` a file `human_prior_infer.write_merged_student` wrote, as the incumbent of generation 15): the fit starts from the file's first net
  (`read_prior_params`), and every baseline is per kind: `load_baseline` reads a merged file as a `StudentScorer`, `TeacherSet.logits_of` scores a kind's rows with the net that serves the kind (so the KL anchor on teacher
  rows and leg 1's prior column are that net's), and leg 2 reads the human log-loss of the baseline per kind the same way. The first net is no longer the baseline of route and event. (The KL anchor on the human rows
  still reads the start net, the first one.) A kind the incumbent plays through the human prior's blend (card, ancient and rest when the student does not serve them) is still measured against the student file's net,
  not the head the incumbent plays.

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

**Heads kept from the incumbent** (D59 lane G; `candidate --keep route,event --kinds card,rest,shop,reward --lambda-floor card=0.5`, `build_student_candidate(keep=, kinds=, lambda_floor=)`,
`build_candidate(student_keep=)`): the incumbent's own student serves some kinds at the lambdas its recipe gives them; a new fit can take other kinds without the candidate losing those heads. A student is one
net scoring every kind, so the two nets go in one **merged student file** (`human_prior_infer.write_merged_student`): net 0 (the new fit's) serves its kinds, net 1 (the incumbent's student file, bytes
unchanged) the kept ones; the arrays sit under `net<i>.` and nothing under a plain name, so a build that does not know the format fails to read the file instead of scoring every kind with net 0 (a plain
`PriorScorer` on the file is net 0; `read_prior_params`, which a fit starts from, likewise). `StudentScorer.scorer_for(kind)` is the net that serves a kind (itself for a plain student) and
`BlendedPolicy` decides each student kind with it; the recipe still names one file by one sha256 (that of the merged file) and every kind in `STUDENT_KINDS` order, agent.json's `student` block gets `kept`
(the incumbent's file, its sha256, the kinds and their lambdas) and `fitted_for` is the merged file's kinds. A kind the incumbent's student does not serve, one the new student takes too, and a base with no
`run_student_legs.json` for its student's bytes are refused. The stored legs are the new fit's with the kept kinds' leg 1, leg 2 and lambda entries copied from the incumbent's own legs (`kept_legs`; `kept`
says where each came from and what it was measured against), so `screen:student` reads every kind the recipe names and holds a kept head to the legs it passed, not to ones measured on another grade's
store. `--lambda-floor KIND=VALUE` (repeatable) raises a taken kind's lambda to at least VALUE in each act; the measured one stays in the legs and is recorded beside the floor (`student.floored` in
agent.json, `lambda.floored` in the legs).

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
* Lambda and the temperature use `run_teacher.lambda_fit` and `run_teacher.noise_and_spread` themselves (imported at the call; a test holds that
  they are those functions).

## Tests

`tests/python/test_run_student.py` (the grade: two grades refused, `TAU_MIN` per grade, a store with no grade is floors'; the map grid: `rows_map.npz` in the loader, the refusals both ways, the teacher's prior for the target, the student
net and file with the tower, a whole map-grid fit and both legs; targets and temperature, the loader and each refusal, the measurement's records, the split, the exam
refusals, the net equal to the prior, the fit on a planted teacher, leg 1, the KL term, leg 2, the verdict's margins, lambda recovering a
planted value, a whole round, the command line), `test_imitation_chooser.py` (the recipe grammar and the byte-for-byte plain recipes, the
student's screens and guard, the policy with a student), `test_screens.py` (`screen:student`: kinds, verdicts, skips, the promote path,
never `screen:prior`), `test_gen_loop.py` (`build_candidate` with a student, the loader's refusals, `build_student_candidate`; `StudentKeepTests`: kept heads, floors and the stored legs),
`test_human_prior_infer.py` (`MergedStudentTests`: the merged file's format and refusals, each kind scored by its own net), `test_imitation_chooser.py` (a merged student decides each kind by its net),
`test_run_student.py` (`TargetLambdaScaleTests`: the scale, its record, the flag, a fit starting from a merged student's first net).

The one-minute smoke fit on the card (synthetic shards written by a scratch script, 4,800 decisions of 400 runs and four synthetic human
fifths; the GPU venv) ran the whole `fit` command: load, fit (7 s on the card), both legs, lambda, `legs.json`; no path or run id in
its files.
