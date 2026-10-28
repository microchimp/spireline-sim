# The combat prior's training recipe, swept on the idle GPU (GPU lane, week 4 item 2, D36)

Design, 2026-10-27, revised the same day after the independent Opus review (section 8). Status words as in
`GPU_INFERENCE.md`.

## 1. The question

The lab fits the combat prior (`CombatNet`, the policy head over the action tokens) each generation with one recipe:
root-visit targets, 16 epochs, `d_model` 128, 2 layers, on 6,400 fights sampled from the window (`gen_loop.MAX_FIGHTS`),
2.6-2.7 GPU-minutes (measured, generations 5-7). Week 3's A2c-target round measured completed-Q targets against visits at
that one size: the dev-set search regret was lower at T = 1 and T = 3 with a CI crossing 0, and the net alone was better
at T = 1. Nothing has measured the net's size, the epochs or the data amount. **Which factors of the recipe move the prior's
quality, by how much against the fit-to-fit spread, and is there a cell worth a paired E-block A/B against the lab's recipe?**

## 2. Data (measured 2026-10-27)

The current window is generation 7's (generations 5-6, 5 closed blocks, every run played on the GPU prior server). The
manifests count 690,342 combat samples; generation 7's own `train.json` counted 551,873 samples after the (generation,
seed) dedupe and 49,825 fights, which are the figures that matter. **A fixed evaluation set is carved out of the window
before any training set is built:** the fights of a tenth of the runs (by run seed hash), never sampled into any training
set; every arm is scored on it, so arms of different data amounts or epochs are judged on the same rows. The training
sets are then built as the lab builds them (`build_training_set`, `max_fights` 6,400 and, for the data arm, 12,800,
build seed 7) from the remaining runs and saved once as `train.npz` files shared by the arms. `train_combat`'s own
held-out split (15% of fights by run, `rng(0)`) stays what it is: the checkpoint-selection split, never a measure.

## 3. Arms

| factor | levels |
|---|---|
| policy target | root visits (the lab's), completed-Q at T = 1, completed-Q at T = 3 (the informative level: at a 14 HP median Q spread, T = 0.5 and T = 1 differ only on near-ties) |
| epochs | 16 (the lab's), 32 |
| `d_model` | 128 (the lab's), 256 |
| layers | 2 (the lab's), 4 |

The full factorial is 24 cells, **two torch seeds each** (48 fits). Beside it: the lab's cell with **two more torch seeds
and two more build seeds** (the split and the per-epoch order come from `rng(0)` in `train_combat`, so a torch seed varies
the initialisation only; the lab's own fits also resample the fights with `seed = gen`, so the fit-to-fit spread the
lab sees is wider than the torch-seed spread: both are measured), and **the data arm**: the lab's cell on 12,800 fights
at 16 epochs, read against the lab's cell on 6,400 fights at 32 epochs (equal gradient steps). Order: the lab's cell
first, then its extra seeds, then the single-factor cells, then the rest, then the data arm, so an interrupted sweep still
has the readable part. Every fit is a queue job of the runner (`tool` `prior_sweep`, 2 CPU threads, BelowNormal);
`train_combat`'s epoch loop gets the stop-file hook so a fit stops between epochs when a window closes; **a fit writes into
a temporary folder and renames it on success** (the trainer saves `combat_policy.pt` on every improving epoch, so "nothing
written" needs the temporary folder), exit 3 and the folder removed when stopped. The first fit of each size is timed and
its minutes decide how many fits a window's budget takes (no timing exists yet for `d_model` 256 or 4 layers).

## 4. Measures

1. **Expected completed-Q regret on the evaluation set** (the searches' own values: for each decision, the max over legal
   actions of completed Q minus the net's policy-weighted mean, in HP). **It is not neutral across targets:** softmax(Q/T)
   has regret 0 as T goes to 0 while any spread visit distribution has regret above 0, so it rewards sharpness toward
   argmax Q and would hand the target factor to the sharpest level by construction. **It ranks size, epochs and data within
   one target level only.** Also reported: the top-1 regret, the top-1 agreement with the most-visited action, the policy
   entropy and the validation policy and value losses.
2. **Top-1 completed-Q regret on the evaluation set**, the measure that is comparable across targets (the net's argmax
   against the search's best completed Q).
3. **Dev-set net alone** (`rl/a2c_target.py`'s path on the 200 labelled A3 positions, `agent/devset`): expected and
   top-1 regret in HP, population-weighted. No search: cheap for every net.
4. **Dev-set search** (`a3-v1` with the prior, 24 seeds per position, 8 emulator processes) for three nets only: the lab's
   cell, the chosen cell (section 5) and the best target level by measure 2 when it is another cell; reported with
   **milliseconds per search**, since a 256-wide or 4-layer prior is not matched wall-clock at matched simulations, and
   with the server's milliseconds per call for that net (its `--selftest`). This is the week-3 instrument, underpowered
   for a 0.05 HP gap; it is reported, not decided on. It needs the CPU: it runs only in a window with no `QUIET.lock` and
   the lab paused (never under the consent branch) and never with the runner's 2-thread jobs beside it.
5. Fit minutes and the GPU peak memory per cell.

## 5. The reading and the rule (pre-registered)

1. **Factor main effects with the between-seed residual as the error.** On measures 1 (within each target level) and 2
   (across targets): a linear model of the cell means on the four factors (and their two-way interactions when the
   residual degrees of freedom allow), the residual being the between-seed variance pooled over cells (24 cells x 1 df,
   plus the lab's cell's extra seeds). A factor "moves the prior" when its effect exceeds twice the residual's standard
   error; the table reports every effect with that error, and the build-seed spread beside the torch-seed spread.
2. **The chosen cell** is the one the main effects point to (the better level of every factor that moves the prior; the
   lab's level otherwise), not the minimum over the 24 cell means (which would name a winner when nothing matters).
3. **Confirmation on fresh fits:** the chosen cell and the lab's cell are re-fit with **three fresh torch seeds each**, and
   the decision reads those six fits only: the chosen cell passes when its mean top-1 regret (measure 2) and, within its
   target level, its expected regret (measure 1) are better than the lab's cell's with the between-seed t-interval
   excluding 0, and its dev-set net-alone regret is not worse beyond that measure's paired interval.
4. **Nothing is adopted by this sweep.** A cell that passes goes to a paired E-block A/B as a lab job once the week-4
   lane has resumed the lab: both arms built from the same live sample, carrying the same V_run fit, differing in the
   prior's recipe only, both on the server transport, under the promotion judge's statistic and guard; the block is sized
   from the predicted gap. Adoption is then a reviewed change of `train_generation`'s defaults with a HANDOFF line. If
   no cell passes, the report says so with the main-effects table.

**Recorded now for the A/B:** the full recipe of each arm including the build seed rule (`seed = gen`, fights resampled)
and the torch seed rule; that one fit per arm tests two fits, not two recipes, so the A/B's arms are fit once each as the
lab does and the sweep's seed spread is the stated caveat; for a 256-wide or 4-layer cell, the server's milliseconds per
call and self-play's runs per hour (measured on one 15-minute block before the A/B), since the A/B's arms are matched on
simulations, not wall-clock.

## 6. Code (fork, reviewed; the lab's default path unchanged)

- `rl/prior_sweep.py`: `build` (the evaluation set, the shared `train.npz` and the data arm's, their shas), `run --arms
  ... --seeds ... --device cuda --stop-file --budget-min` (one fit per call through `gen_loop.train_combat_prior` with
  `threads=2`, into a temporary folder renamed on success; measures 1-3 written beside the net as JSON), `search --arms
  ...` (measure 4 through `a2c_target`'s functions, with ms per search), `table` (the main-effects table with the
  between-seed error, the cell table, the confirmation table, the data-arm reading).
- `rl/train_combat.py`: an `epoch_end` hook (None by default; the loop unchanged without it).
- `tools/gpu_lane/quiet_runner.py`: queue jobs carry `tool` (`vrun_sweep` or `prior_sweep`) and `cpu_procs` (default 2);
  under the consent branch only jobs with `cpu_procs` <= 2 run; the search jobs carry 8 and run only in the no-lock
  paused window.
- Tests on synthetic shards: the evaluation set is disjoint from every training set; the grid and its order; the
  temporary-folder rename and the stop file between epochs; the measures on synthetic predictions (incl. the sharpness
  property of measure 1: a softmax(Q/T) net's regret falls with T, the test's reason for ranking within a level); the
  main-effects arithmetic on synthetic cells; the queue's `tool` and `cpu_procs`; no absolute path in any written JSON.

## 7. Cost, where it runs, and risks

Design and review ~0.8 round (spent), implementation and its review ~1.0, the fits GPU time in quiet windows (the lab is
paused under `w4-unstick`, so the runner's rule 3 opens the window now), the dev-set searches ~45 CPU-minutes in all in an
open window. Fits: 48 + 4 + 2 + 6 = 60, timed from the first of each size. Disk: each fit's work folder (its `train.npz`
copy) is removed after the fit; the kept checkpoints are ~3 MB each. Risks: the window is one generation's static play;
the evaluation set's searches carry the generation-7 prior's own biases (every arm is judged on the same rows, so the
comparison is fair, the absolute numbers are not transferable); measure 1's sharpness bias is contained by reading it
within a target level; two seeds per cell resolve main effects, not cells, which is why the confirmation re-fits decide.

## 8. Review log

- **2026-10-27, independent Opus review of the first draft: CHANGES**, two blockers and four required findings, all
  applied: expected Q-regret rewards sharpness, so it ranks within a target level and the targets are read by top-1 regret
  and the search check (1); the rule reads factor main effects against the between-seed residual and confirms the chosen
  cell on fresh fits instead of picking the minimum cell (2); a fixed evaluation set carved before any build, the trainer's
  split kept for checkpoint selection only, the data arm read at equal steps (3); the seed rules recorded and build-seed
  fits added (4); what the E-block A/B needs recorded now (5); the temporary-folder fit for the stop file, `threads=2`, the
  dev-set search kept out of consent windows (6); T = 3 instead of T = 0.5, the 15% split, the disk figure (7). The
  sample count is now the deduplicated one from `train.json`; the GPU-hours figure is replaced by timing the first fits.

## 9. Implementation notes (2026-10-28)

`rl/prior_sweep.py` (`build`, `run`, `search`, `table`, `queue`), `train_combat.train`'s `epoch_end` hook (passed through
`gen_loop.train_combat_prior`), and the runner's `tool` / `cpu_procs` fields are in; `tests/python/test_prior_sweep.py` covers them.
Where the code makes a choice the sections above leave open:

- **Build.** The window is the closed blocks of generations 5-7's window (`vrun_sweep.closed_blocks`). The combat parts are split by
  run (`_unit_hash` of the seed in [0.0, 0.1) is the evaluation set) into two sets of copies before `build_training_set` runs, so its
  dedupe by (generation, seed) acts on whole seeds on either side. The two build seeds of the `lab-bs<N>` cells (8 and 9) are built
  together with the shared sets (`build --build-seeds`), so `run` never needs the lab. `build_training_set` now also returns
  `_fight_ix` (an underscore key: the saved sets do not carry it) for the evaluation set's fight ids.
- **Seeds.** The torch seed is the fit's `--seeds` entry; the `lab-bs<N>` cells are fit at torch seed 0 so the build-seed spread
  (torch seed 0 at build seeds 7, 8, 9) is not mixed with the torch-seed spread (build seed 7 at torch seeds 0-3). The lab's own rule
  moves both at once (`seed = gen`). Every `measures.json` records both rules.
- **Error model.** Codes are centred indicators (a two-level factor reads second level minus first, averaged over the other factors;
  target reads each completed-Q level minus visits). The residual is the between-seed variance pooled over the cells of the model:
  the 24 cells and the lab's extra seeds (26 df) for measure 2, the 8 cells of the level (8 df, 11 for visits) for measure 1.
  Two-way interactions are in when the means leave a spare degree of freedom.
- **The chosen cell** (5.2): the target level with a measure-2 effect below -2 SE (the lower), and each other factor's second level
  when its measure-2 effect is below -2 SE; a factor that measure 1 within the chosen target level contradicts beyond 2 SE stays at
  the lab's level, with a note. The confirmation reads measure 1 only when the chosen cell keeps the lab's target (across targets it
  is not comparable); the dev-set check is a paired bootstrap over positions of the seed-mean net-alone regret, "worse" meaning the
  interval of chosen minus lab lies above 0.
- **Search.** `search --arms auto` takes the lab's cell, the chosen cell and the best target level's cell from `table`'s logic. The
  server's milliseconds per call (`--selftest`) is not measured by this tool.
- **Runner.** A prior_sweep job's args start with its subcommand; a job with `cpu_procs` above 2 runs only in a window whose reason
  starts with "quiet", and is stopped like a closed window when the window turns to consent. `run` needs `STS2_LIB_PATH` in the
  runner's environment (measure 3 loads the emulator) and exits 2 before any fit without it.

Timing fits (generation 7 window, 16 epochs, torch seed 0, 2 CPU threads, BelowNormal, one RTX 3080), minutes include the data write:

| cell | fit min | GPU peak MB | eval top-1 | eval expected | dev top-1 | dev expected |
|---|---|---|---|---|---|---|
| visits-16-128-2 | 3.54 | 707 | 5.587 | 7.997 | 0.329 | 1.377 |
| visits-16-256-2 | 3.51 | 842 | 5.621 | 7.913 | 0.278 | 1.302 |
| visits-16-128-4 | 4.47 | 807 | 5.621 | 7.955 | 0.262 | 1.314 |
| visits-16-256-4 | 5.34 | 1047 | 5.647 | 7.870 | 0.198 | 1.322 |

Width costs almost nothing at 2 CPU threads and depth costs about a minute per two layers. Scaling the 16-epoch minutes by the epochs,
the 48 factorial fits come to about 5 hours, and the whole sweep (54 fits, then 6 confirmation fits) to about 5.5 hours of quiet
windows. The lab's recorded 2.6-2.7 minutes for its cell is on another timing basis (these include the data write and the
evaluation passes); the difference was not investigated.

## 10. Results (measured 2026-10-28; fits 2026-10-27 23:02Z to 2026-10-28 17:59Z)

**The lab's recipe is the chosen cell. No factor of the recipe improves the prior at 6,400 fights; completed-Q targets and four
layers make it worse; nothing passes the rule of section 5, nothing goes to an E-block A/B.** The one factor with a favourable sign,
twice the fights per fit, did not confirm on fresh seeds. Full tables: `agent/gpu-lane/prior/report.md`, `table.json` and
`confirm-data.json` (the referee folder, not the repo); every fit's `measures.json` beside its net.

### 10.1 What ran

63 fits (the 48 factorial, 2 extra torch seeds and 2 build seeds of the lab's cell, the data arm's 2, then 9 confirmation fits),
about 6.4 GPU-hours in 23 windows of the runner over 19 hours of wall clock: 18 consent windows (the kernel lane's `rebuild-s3`,
`rebuild-s4`, `rebuild-s4fix` and `rebuild-base` tags, the week-5 lane's `lab-nightly`, each with `GPU_OK`) and 5 quiet windows
(`paused(lock)` 3, `paused(pause)` 2). 22 interruptions at a window's close (a lock without consent, a landing's `main.lock`, the
lab resuming), every one resumed at the next window, 0 failed fits. Fit minutes (2 CPU threads, BelowNormal, one RTX 3080, the
data write and the evaluation passes included; the lab's 2.6-2.7 for its cell is on another basis):

| 16 epochs | 128/2 | 256/2 | 128/4 | 256/4 | | 32 epochs | 128/2 | 256/2 | 128/4 | 256/4 | | data arm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| min | 3.1-3.2 | 3.5-3.6 | 4.4-4.5 | 5.3-5.5 | | min | 6.1-6.6 | 6.9-8.9 | 9.2-12.2 | 10.9-11.3 | | 6.2 |
| GPU MB | 707 | 842 | 806 | 1,050 | | GPU MB | 707 | 842 | 806 | 1,050 | | 1,274 |

### 10.2 Cells (means over torch seeds; regrets in HP; dev = the net alone on the 200 labelled positions)

| cell | fits | top-1 regret | expected regret | dev top-1 | dev expected |
|---|---|---|---|---|---|
| **visits-16-128-2 (the lab's)** | 4 | 5.620 [5.587, 5.649] | 8.001 [7.983, 8.021] | 0.328 | 1.384 |
| visits-16-128-4 | 2 | 5.617 | 7.964 | 0.216 | 1.317 |
| visits-16-256-2 | 2 | 5.593 | 7.903 | 0.239 | 1.289 |
| visits-16-256-4 | 2 | 5.630 | 7.862 | 0.224 | 1.324 |
| visits-32-128-2 | 2 | 5.583 | 7.927 | 0.261 | 1.317 |
| visits-32-128-4 | 2 | 5.608 | 7.955 | 0.238 | 1.333 |
| visits-32-256-2 | 2 | 5.570 | 8.016 | 0.376 | 1.385 |
| visits-32-256-4 | 2 | 5.632 | 8.019 | 0.236 | 1.404 |
| cq1-16-128-2 | 2 | 5.655 | 8.134 | 0.242 | 1.470 |
| cq1-16-128-4 | 2 | 5.702 | 8.028 | 0.206 | 1.363 |
| cq1-16-256-2 | 2 | 5.623 | 7.999 | 0.265 | 1.467 |
| cq1-16-256-4 | 2 | 5.682 | 8.054 | 0.378 | 1.495 |
| cq1-32-128-2 | 2 | 5.618 | 8.089 | 0.382 | 1.494 |
| cq1-32-128-4 | 2 | 5.675 | 8.168 | 0.201 | 1.492 |
| cq1-32-256-2 | 2 | 5.630 | 8.065 | 0.392 | 1.549 |
| cq1-32-256-4 | 2 | 5.710 | 8.062 | 0.289 | 1.537 |
| cq3-16-128-2 | 2 | 5.663 | 8.622 | 0.407 | 1.831 |
| cq3-16-128-4 | 2 | 5.691 | 8.540 | 0.588 | 1.756 |
| cq3-16-256-2 | 2 | 5.674 | 8.411 | 0.458 | 1.796 |
| cq3-16-256-4 | 2 | 5.694 | 8.463 | 0.459 | 1.796 |
| cq3-32-128-2 | 2 | 5.628 | 8.577 | 0.386 | 1.820 |
| cq3-32-128-4 | 2 | 5.684 | 8.564 | 0.573 | 1.803 |
| cq3-32-256-2 | 2 | 5.688 | 8.519 | 0.454 | 1.815 |
| cq3-32-256-4 | 2 | 5.735 | 8.772 | 0.740 | 1.982 |
| lab's cell, build seeds 8, 9 | 1 each | 5.594, 5.636 | 7.970, 7.987 | 0.459, 0.320 | 1.383, 1.367 |
| data12800 (the lab's cell on 12,800 fights) | 2 | 5.540 [5.529, 5.551] | 7.929 | 0.280 | 1.340 |

Seed spreads of the lab's cell: top-1 regret sd 0.029 over 4 torch seeds and 0.026 over 3 build seeds; expected regret 0.016 and
0.014; dev top-1 0.025 and 0.078; dev expected 0.031 and 0.008. The build-seed spread (the lab's own `seed = gen` rule resamples
the fights) is no wider than the torch-seed spread on the evaluation set.

### 10.3 Main effects (5.1; centred codes, the error is the between-seed variance pooled over the cells)

Measure 2, top-1 regret across targets (24 cells, interactions in, residual variance 0.00068 on 26 df, SE 0.009 per main effect):

| effect | estimate (HP) | SE | beyond 2 SE |
|---|---|---|---|
| target: completed-Q, T = 1, minus visits | +0.055 | 0.009 | **yes, worse** |
| target: completed-Q, T = 3, minus visits | +0.076 | 0.009 | **yes, worse** |
| epochs 32 minus 16 | -0.007 | 0.007 | no |
| d_model 256 minus 128 | +0.010 | 0.007 | no |
| layers 4 minus 2 | +0.043 | 0.007 | **yes, worse** |
| epochs 32 x d_model 256 | +0.037 | 0.015 | yes, worse |
| the other eight two-way interactions | within 2 SE | | no |

Measure 1, expected regret within a target level (8 cells each; visits also carries the lab's extra seeds):

| level | epochs 32 | d_model 256 | layers 4 | interactions beyond 2 SE |
|---|---|---|---|---|
| visits (10 df, SE 0.025) | +0.047 | -0.013 | -0.012 | epochs x d_model +0.177 (SE 0.049): the 32-epoch 256-wide cells overfit |
| cq1 (8 df, SE 0.025) | +0.042 | **-0.060** | +0.006 | none |
| cq3 (8 df, SE 0.029) | **+0.099** | -0.034 | +0.053 | every pair positive (+0.135 to +0.219): the long, wide and deep cq3 cells are worse |

**The chosen cell (5.2) is `visits-16-128-2`, the lab's own:** the target factor points to visits, the layers factor to 2, and epochs and
width move nothing beyond 2 SE on measure 2 (within the visits level, measure 1 moves on none of them either). So 5.3 has nothing to
confirm and 5.4 sends nothing to an A/B.

### 10.4 The data arm (2 fits, then a post-hoc confirmation on 3 fresh seeds)

At 2 fits the data arm read -0.080 HP top-1 regret against the lab's cell (SE 0.024, beyond 2 SE) and -0.044 against 32 epochs on
6,400 fights, the equal-steps reading (SE 0.027, within). Because the pre-registered confirmation (5.3) had nothing to confirm, the
same reading was applied post hoc to the data arm: three fresh torch seeds (10, 11, 12) each of the data arm, the lab's cell and the
32-epoch cell, read on those nine fits only with 5.3's rule (Welch 95% t-intervals; the dev-set check a paired bootstrap over
positions of the seed-mean regret).

| data arm against | top-1 regret | expected regret (within visits) | dev top-1, paired | dev expected, paired | 5.3 |
|---|---|---|---|---|---|
| the lab's cell (6,400 fights, 16 epochs) | -0.034 [-0.127, +0.060] | **-0.049 [-0.081, -0.016]** | +0.058 [-0.039, +0.170] | +0.004 [-0.033, +0.050] | **fail** (top-1 crosses 0) |
| 32 epochs on 6,400 (equal steps) | -0.027 [-0.116, +0.061] | +0.010 [-0.062, +0.081] | +0.115 [+0.005, +0.264], worse | +0.037 [-0.013, +0.100] | fail |

Fresh-seed top-1 regrets: the lab's cell 5.638, 5.577, 5.576; the data arm 5.574, 5.567, 5.548; 32 epochs 5.628, 5.574, 5.569. The
2-fit -0.080 was half a seed draw: with the between-seed sd of 0.03 HP, three fits a side resolve about 0.09 HP on top-1 regret, and
the data arm's lead (0.03 HP) sits under it. Its expected regret within the target does confirm (-0.05 HP: sharper toward the
search's best action), and its dev set is not worse. Doubling the epochs on the same fights does nothing (top-1 -0.006
[-0.095, +0.082]): what the data arm has, it has from the fights, not the steps.

### 10.5 What was not run

- **Measure 4, the dev-set search for three nets:** not run. The three nets of the design resolve to the lab's cell alone (the chosen
  cell and the best target level are both the lab's), and no other cell differs from it beyond the instrument's resolution (0.05 HP,
  section 4); the queued job was pointed at the lab's cell against the data arm and then removed when the data arm failed to confirm
  (45 CPU-minutes would read noise). The server's milliseconds per call for a wider or deeper net was not measured for the same
  reason: no such net is a candidate.
- **The E-block A/B (5.4):** not queued; no cell passes.

### 10.6 Reading

1. **The recipe is not the lever.** Four factors, 24 cells, 2 seeds each: the lab's cell is at the plateau. Completed-Q targets cost
   0.055 (T = 1) and 0.076 HP (T = 3) of top-1 regret on the search's own values, on the population of the window's decisions with an
   SE of 0.009; four layers cost 0.043; 32 epochs and width 256 buy nothing, and together they overfit (the trainer's 15% checkpoint
   split does not prevent it). The between-cell differences within the visits level (0.06 HP) are the size of the seed spread.
2. **Week 3's dev-set reading does not hold on the population.** Week 3 saw the net alone better with completed-Q targets at T = 1 on the
   200 dev positions; here `cq1-16-128-2`'s dev top-1 is 0.242 against 0.328 with seed ranges that overlap (sd 0.025-0.078), while the
   evaluation set (every decision of a tenth of the window's runs) says the opposite with 6x the precision. The dev set's 200 positions
   cannot resolve a 0.05 HP effect; the population can.
3. **More fights per fit is the only factor with a favourable sign, and it is small.** -0.03 HP top-1 (unconfirmed), -0.05 expected
   (confirmed within the target), at twice the fit time (6.2 against 3.1 minutes) and 1.3 GB of GPU memory. It is below the E-block's
   resolution, so it is not worth a block now; it is the cheapest lab-side change if a block is ever spent on the prior
   (`gen_loop.MAX_FIGHTS` 12,800).
4. **Where the prior's quality comes from.** With the recipe flat, the prior's 5.6 HP top-1 / 8.0 HP expected regret against the
   search's own completed Q on this window is a property of the data: one static generation's self-play, labels from searches the
   prior itself guided. A better prior comes from the loop (fresher, more diverse windows, more fights), not from the fit.

**Nothing is adopted (5.4).** Lane spend for item 2: design and review 0.8 rounds, implementation and review 1.0, the fits in the
runner's windows (GPU only), the report 0.4.
