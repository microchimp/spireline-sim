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
