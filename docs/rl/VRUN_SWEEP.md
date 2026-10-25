# Training V_run harder on the idle GPU (GPU lane, piece 2, D30)

Design, 2026-10-25, revised the same day after the independent Opus review (section 9). Status words as in
`GPU_INFERENCE.md`: **measured**, **estimated**, **decided**. Piece 3 (sweeps in the quiet windows) hangs off section 7 and
starts only once this piece has a measured result (decided, D30).

## 0. The question

V_run is the run-level value (`run_train.models.RunValueNet`): the deck, relics and potions as count embeddings plus the
dense state (act, floor, HP, map ahead, the 8-number fight block H that FightNet predicts for the fights ahead), a
1,863-wide input, 256 -> dropout 0.1 -> 256 -> 53 floor classes and an end-HP sigmoid, trained each generation from
fresh weights on the window's run records with a floor target and 0.25 x the end-HP MSE (AdamW 1e-3, weight decay 1e-4,
batch 1024, cosine over 30 epochs, patience 3, a held-out tenth of runs by seed hash, CPU, 6 threads). **The target is
pure Monte Carlo today:** the `beta` 0.3 blend reads `p_floor` from the records, and under the rules macro every record
stores none (measured by the review: null in all 19,048 records), so the one-hot final floor is the whole target.

**V_run fails its keep conditions (DESIGN 5.3).** On the generation-4 candidate's fit (measured, `gen4/candidate/train.json`:
289,532 rows, 10,562 run records, stopped at epoch 4 with the best validation loss at epoch 1 of 30) the week-3 re-evaluation
on the Act-1 boss-kill head (rows before the boss room, floor <= 16) gave a Brier skill score of 0.048 against the base
rate (CI -0.030 to 0.104) and 0.046 against V0 (CI -0.031 to 0.102): both fail; log-loss 0.181 against 0.214 and ECE 0.6 pp
over 3 eligible bins pass. So the macro policy stays `rules`.

**Two facts the review found change what can be measured:**

- **The window holds duplicate run records.** Block `20261024T0432-selfplay` has 4,865 records for 1,115 manifest runs
  (950 seeds appear up to 8 times as identical copies); the generation 1-3 window has 19,048 records for 15,298 distinct
  seeds. The lab's fits and the keep test's run bootstrap counted copies as runs (the CIs above are too narrow). A lab
  finding for the week-3 lane (told); this sweep dedupes by seed.
- **A tenth of the runs holds only ~38 Act-1 kills** (2.3-3.4% of runs by seed; the 5.9% of the week-3 report was a row
  rate before the floor cut). The standard error of a Brier skill score on 38 positives is about 0.043, so a true skill of
  0.08 would clear 0 on such a split about a third of the time: a "nothing passes" verdict on a tenth would mostly measure
  the protocol. Confirmation needs the whole non-selection set (section 1).

The fit's best epoch being its first is the lead. Hypotheses, each with an arm:

- **H1 optimisation and regularisation.** 1e-3 on 1,863 mostly-sparse inputs with weak regularisation finds its best point
  in one pass and then memorises; a smaller rate, more weight decay, more dropout or a narrower layer should give a later
  best epoch and a better out-of-fold Brier.
- **H2 correlated rows.** A run gives ~27 rows with one shared outcome (a run that kills the boss ~39 rows before it); the
  effective sample is the run count. Sampling k rows per run per epoch gives every run the same say without down-weighting
  the long runs' kills (the review's point against 1/rows weights).
- **H3 the event is weakly predictable (the null).** At Ascension 10 the Act-1 kill may depend little on what the state
  before floor 17 shows, given this agent. A logistic model on V0's inputs + floor + H is the diagnostic that separates H3
  from H1/H5: if it has no skill either, the signal is not in the input.
- **H4 the floor distribution is not the event.** P(kill) is read off a 53-class one-hot target as the mass above floor
  17. A diagnostic binary head (BCE on the kill label, floor <= 16 rows) says whether a direct target finds more; it is
  **never adoptable** (the policy reads the floor distribution, and the keep test certifies what the policy reads).
- **H5 too little data.** 13-15k deduplicated runs; a training-run subsample curve (25/50/75/100% of the training runs on
  fixed evaluation runs) says how skill moves with data, and a 5-seed ensemble says how much is variance (diagnostic: the
  policy loads one net).

## 1. Protocol (decided here, after the review)

1. **One cached, deduplicated dataset.** The current window's run shards (`gen_loop.window_manifests(LAB/runs, gen)`; closed
   blocks only, a block with no `done` and a changing manifest is skipped) are read exactly as `run_train.train.load_buffers`
   reads them, then **deduplicated by seed** (the first record per seed kept, the copy count recorded), and the `ValueData`
   (sparse rows, targets, end HP, run index, each row's floor, the run's final floor, V0's inputs, the seeds and their
   hashes) is saved under `D:/spireline/referee/agent/gpu-lane/vrun/cache-<key>/` as npz. **The cache key is the list of
   manifests with every shard name and size**, so a window that moves (a promotion, a new block) is a new cache. H is not
   cached: FightNet is refit per fold (next item), because a FightNet trained on every run leaks the evaluation runs'
   boss fights into their rows (the review's blocker 1).
2. **Splits by run seed hash** (`train._unit_hash`), never the lab's own tenth:
   - **S, the selection split: hash in [0.2, 0.3)** (~1,500 runs, ~38 kills). Arms are early-stopped and ranked on S. The
     lab's split [0, 0.1) is left untouched so the lab's next fit is a fresh test of whatever is adopted.
   - **Confirmation by 5-fold cross-fitting over the non-S runs** (hash outside [0.2, 0.3), ~13.7k runs, ~370 kills, SE ~0.014
     estimated): for each fold, FightNet is fit on the other four folds, H is filled for all rows from that FightNet, V is fit
     on the other four folds with early stopping on S, and the fold's rows get out-of-fold predictions; the keep statistics
     are computed once on the pooled out-of-fold rows. Cost per confirmed arm: 5 FightNet fits + 5 H fills + 5 V fits.
   - Confirmation runs for `lab` and for **at most two arms pre-declared from the S ranking** (and nothing else: the other
     arms' confirmation columns do not exist, so they cannot be read).
3. **The keep conditions, as the lab computes them** (`run_train.keep.head_conditions`): Act-1 boss-kill head on rows with
   floor <= 16; Brier skill against the training base rate and against V0 (calibrated through its training-row quantile
   table) with the paired bootstrap over runs (2,000 draws, 2.5/97.5 percentiles); log-loss against the base rate; ECE over
   bins with >= 30 distinct positive runs, **every bin reported** (the gen-4 fit passed on 3 bins while bins 3-6 were
   overconfident); the informational V0 + floor baseline. The win head has no positives and judges nothing.
4. **What the policy reads, also measured.** `LookaheadPolicy` ranks options by V_W = sum_k k p_k + W p_53, not by
   P(kill). Each confirmed arm also reports the mean absolute error of its expected floor against the run's final floor,
   on the out-of-fold rows, beside V0 + floor's regression on the same rows.
5. **Adoption rule (the task's, made exact).** The **adoptable set** is a `TrainConfig` recipe: learning rate, weight decay,
   dropout, hidden width, row sampling, epochs and patience. An arm is adopted only if, on the pooled out-of-fold rows, both
   Brier skill scores (against the base rate and against V0) have bootstrap CIs whose lower bounds are above 0, log-loss
   beats the base rate and ECE <= 3 pp, **and** each of its 3 seeds has positive skill on S. Adoption lands as a reviewed
   commit that changes `TrainConfig`'s defaults, with a HANDOFF line. **Adoption does not flip the macro by itself:** today
   `train_generation` switches the candidate's macro to `vrun` whenever a lab fit passes the keep conditions; that switch
   would then judge the prior and V_run together in the next promotion without DESIGN 5.3's Tier-B regret and battery
   conditions. The adoption commit therefore puts the macro switch behind an explicit setting (an `agent.json` field or
   environment flag, default off), and the flip waits for a paired rules-against-vrun A/B on one E-block with the same
   prior and an owner decision. Once vrun plays, `p_floor` appears in the records and the `beta` blend comes alive, so this
   sweep's verdict holds for the Monte Carlo target only.
6. **Parity.** `lab` runs once on the CPU (the lab's path) and once on the GPU with the same seed. Dropout on CUDA draws
   from its own generator, so the two fits differ by more than cuBLAS rounding; the parity row is judged against the
   3-seed spread, not bit for bit. Every arm's number is from the GPU fit.
7. **Three seeds per arm**; the table shows the mean and spread of the S keep statistics, the best epoch, the S Brier of
   the head per epoch (logged), and the fit minutes. Fit time per arm is **estimated**, not measured, until the first run.

## 2. Arms

| group | arm | change against the lab's config |
|---|---|---|
| baseline | `lab` | none (CPU and GPU parity); `lab-p10` the same with patience 10 (so the lr arms differ by the rate alone) |
| H1 | `lr3e-4`, `lr1e-4` | learning rate; epochs 100, patience 10 |
| H1 | `wd1e-3`, `wd1e-2` | weight decay |
| H1 | `do0.3`, `do0.5` | dropout before hidden2 |
| H1 | `h64`, `h128`, `h512` | hidden width (both layers) |
| H2 | `rows4`, `rows8` | k rows sampled per run per epoch (every run the same say; kills not down-weighted) |
| H4 (diagnostic only) | `bce-act1` | a binary head on the kill label, floor <= 16 rows; BSS only, never adoptable |
| H3 (diagnostic only) | `logit-v0h` | logistic regression on V0's inputs + floor + H (is the signal in the input at all?) |
| H5 (diagnostic) | `sub25`, `sub50`, `sub75` | the training runs subsampled, evaluation runs fixed: the data curve |
| H5 (diagnostic) | `ens5` | 5 seeds of the best S arm, probabilities averaged: the variance share |
| diagnostic | `noH` | H zeroed: does the fight block carry anything? |
| combination | `best+` | the two or three best single changes together, chosen on S; a candidate for confirmation |

Dropped from the first draft, with the review's reasons: `beta0`/`beta0.5` (the blend does nothing without `p_floor`),
`allruns` (only generations 1-3 have shards, so it equals `lab`), `warm` (its source trained on the evaluation runs and
duplicates `fit(prev_checkpoint)`), `runw` (1/rows weights down-weight the kill runs' long pre-boss stretch about 3x).
Order: `lab` and `lab-p10` (parity, the patience control), H1, H2, the diagnostics, then `ens5` and `best+` from the S
ranking, then confirmation of `lab` plus the one or two pre-declared arms.

## 3. Code (fork, reviewed before landing; the lab's default path unchanged bit for bit)

- `rl/run_train/train.py`: `TrainConfig` gains `device: str = "cpu"`, `hidden: int = 256`, `dropout: float = 0.1`,
  `rows_per_run: int = 0` (0: every row, today's `.mean()` loss unchanged). On a device other than the CPU: the net moves to
  the device before the optimiser is built; the deduplicated dense rows stay resident on the GPU (about 2.2 GB dense for
  290k x 1,863 float32, or CSR) so no per-batch densify on the CPU, which the review names as the real bottleneck; the
  shuffle generator stays on the CPU; the loss is accumulated on the device and read once per epoch, never `.item()` per
  batch; evaluation (`keep.evaluate_value_net` calls `.numpy()`) and saving run on the CPU copy of the net, and
  `load_torch` gets `map_location="cpu"`. The `device="cpu"` path is today's code: a test compares a fit against a golden
  checkpoint produced by the parent commit on a small synthetic buffer, bit for bit.
- `rl/run_train/models.py`: `RunValueNet(hidden=256, dropout=0.1)`; the checkpoint meta records them; `NumpyValueNet`
  reads shapes from the weights, so a 128-wide export plays as the policy reads it (a test exports and plays one).
- `rl/vrun_sweep.py`: `cache` (dedupe, splits, save; prints the duplicate census and the peak RSS of the build), `run
  --arms ... --seeds 3 --device cuda --budget-min N` (fits from the cache, one JSON per fit with the S keep table, the per-epoch
  S Brier, the best epoch and the fit minutes; stops between fits when the budget is spent: what piece 3's runner calls),
  `confirm --arms lab,<a>,<b>` (the 5-fold cross-fit with FightNet per fold and the pooled out-of-fold keep table and
  expected-floor error), `table`. Arms are named in the file. Diagnostic arms are marked `adoptable=False` in code and in
  the table.
- Tests (CPU, synthetic buffers): the splits are disjoint and cover the stated hash ranges; dedupe keeps the first record
  per seed and counts the rest; `rows_per_run` samples k rows of every run and leaves the loss's scale as the mean;
  the golden bit-for-bit test above; a cache round trip; the fold assignment covers every non-S run exactly once and
  FightNet's training rows of a fold never include that fold; the table's arithmetic; no absolute path in any JSON the
  tool writes (paths are relative to the cache folder).
- No new input features (DESIGN 5.2); `noH` is an ablation, not a model.

## 4. Where it runs, and what it must not cost

The GPU is no longer idle: self-play and the promotion arms run on the GPU inference server since 2026-10-25 01:40Z
(`GPU_INFERENCE.md`), and self-play at 20 workers peaks at 29.8 GB of commit (31.2 GB on a double server death) against
31.9 GB of RAM. So, before any fit runs beside self-play: the cache build's peak RSS is measured (estimated 3-5 GB for the
sparse rows and targets of ~400k rows) and **one fit runs while a self-play block plays, with the server's live stats
read every 5 s: its p99 total time per batch and the block's client p99 must stay under the 20 ms degradation bound**, and
the block's decisions per second within 5% of the 9.1 figure. If either moves, the sweep runs only when self-play is paused
(a quiet window, piece 3, or the lab's own `train` step holding the GPU, in which case the sweep waits). Fits use at most
2 CPU threads at BelowNormal. The sweep writes only under `agent/gpu-lane/vrun/` and the fork worktree; it reads the
lab's shards and the current generation's checkpoint; it never touches `jobs.json`, `current.json`, `promotions.tsv` or a
`genN/` folder.

## 5. The report: measured 2026-10-25 (cache `756c32dd`, generation 3's window)

**Data.** 10 closed blocks of generations 1-3: 19,048 records, **15,298 runs after deduplication by seed** (3,750 copies, all
in block `20261024T0432-selfplay`), 421,893 rows, 411 Act-1 kill runs, 106,688 fights (under the lab's 400k cap, so FightNet
sees the lab's data). S (hash in [0.2, 0.3)): 1,529 runs, 38 kills, 41,589 head rows. Outside S: 13,769 runs, 373 kills,
370,829 head rows, in five folds of 58-95 kills. Cache build 5.5 min at 2 threads, peak RSS 1.6 GB. A V fit takes 0.1-0.4
GPU-minutes (4.1x faster than the CPU fit of the same recipe); FightNet plus the H fill ~0.55 min per seed or fold.

**S ranking (every arm, 3 seeds; Brier skill on S, mean and seed range).** The lab recipe: -0.011 (-0.039..+0.007), best
epoch 1 in every seed (the CPU fit of the same recipe -0.004: the parity gap is inside the seed spread). Every H1 arm stays at
or below +0.003: `lr3e-4` -0.016, `lr1e-4` -0.003 (best epoch 12, still no skill), `wd1e-3` -0.012, `wd1e-2` -0.015, `do0.3`
-0.010, `do0.5` -0.009, `h64` -0.014, `h128` -0.015, `h512` +0.003. **H2: `rows4` +0.030 (+0.028..+0.032), best epoch 10;
`rows8` +0.021 (+0.012..+0.026), best epoch 7; `rows4+h512` +0.030 (+0.029..+0.031).** Diagnostics: `bce-act1` +0.050
(+0.049..+0.051); `logit-v0h` +0.039 (+0.034..+0.042): a logistic model on V0's inputs, the floor and H has more skill on S
than the lab's net; `noH` -0.001 (H carries nothing for the lab fit); the data curve `sub25` -0.049, `sub50` -0.086, `sub75`
-0.024, `lab` -0.011. The pre-declared confirmation arms, from this ranking: `rows4` and `rows8` (`rows4+h512` ties `rows4`
and adds nothing).

**Confirmation (5-fold cross-fit over the 13,769 non-S runs, FightNet and H refit per fold, early stopping on S, pooled
out-of-fold floor <= 16 rows; base rate per fold's training rows; V0 calibrated per fold):**

| arm | BSS vs base rate [CI] | BSS vs V0 [CI] | log-loss model / base | ECE (eligible bins) | expected-floor MAE model / V0+floor | best epoch per fold | rule 1.5 |
|---|---|---|---|---|---|---|---|
| `lab` | +0.027 [+0.002, +0.051] | +0.025 [+0.001, +0.049] | 0.1495 / 0.1661 | 1.0 pp (8) | 2.699 / 3.215 | 1, 1, 2, 1, 2 | fail (S seeds -0.039..+0.007) |
| **`rows4`** | **+0.044 [+0.028, +0.060]** | **+0.042 [+0.026, +0.058]** | 0.1526 / 0.1661 | 1.6 pp (7) | 2.702 / 3.215 | 17, 8, 13, 9, 20 | **PASS** |
| `rows8` | +0.042 [+0.026, +0.059] | +0.040 [+0.024, +0.057] | 0.1521 / 0.1661 | 1.6 pp (7) | 2.705 / 3.215 | 7, 5, 4, 8, 10 | PASS |

**Reading.** (1) **V_run has a small, real pre-kill skill once the duplicates are gone and the evaluation is large
enough**: even the lab's recipe clears 0 out of fold (+0.027, lower bound +0.002); the week-3 failure (0.048, CI crossing 0)
was a 38-kill evaluation on a duplicated window, not evidence of no skill. (2) **H2 holds**: sampling four rows per run per
epoch moves the skill to +0.044 against both baselines (lower bounds +0.028 and +0.026), moves the best epoch from 1-2 to
8-20 (the fit learns instead of memorising the long runs), and every one of its three S seeds is positive. (3) **H1 does
not**: no learning rate, weight decay, dropout or width changes the S skill. (4) **H3 is rejected for this data**: the input
carries signal (the logistic diagnostic +0.039, the direct binary head +0.050 on S), and (5) **H4 is partly supported**: the
direct kill target still finds more than the 53-class floor target does, so the keep condition's event is not what the net is
best at; the policy's own quantity, the expected floor, is nevertheless better than the V0 + floor regression (MAE 2.70
against 3.22, a 16% reduction) for every recipe. (6) **H5**: the data curve is steep and noisy (sub25 -0.049 to lab -0.011
on S); more runs will help, and the confirmation's 13.7k runs are what made the verdict possible.

**Verdict: `rows4` is adopted** under rule 1.5 (both out-of-fold CIs clear 0, log-loss and ECE pass, three positive S seeds):
`TrainConfig.rows_per_run` defaults to 4 in the adoption commit. The macro stays `rules`: the switch to `vrun` is behind an
explicit setting (`SPIRELINE_MACRO_VRUN`, default off) and waits for a paired rules-against-vrun A/B on one E-block and an
owner decision (the Tier-B regret and battery conditions of DESIGN 5.3 are still not computed). The lab's own keep test on
its tenth (~37 kills) will keep reading a CI that crosses 0 at this skill size; the out-of-fold table above is the evidence.

**Not measured.** The `beside` check of section 4 (a fit beside a running self-play block): no self-play job ran during the
sweep (the lab was in presort). It stays a precondition for running fits while self-play plays; the sweep's fits ran with the
GPU otherwise idle.

## 6. Risks

- **The null holds.** If no arm's pooled CI clears 0, nothing is adopted; the lane reports V_run has no measurable
  pre-kill skill at this data size, the macro stays `rules`, the data curve says what more runs would buy, and DESIGN's V0
  remains the hand-written value.
- **Selection on S, confirmation out of fold** bounds the winner's curse; only pre-declared arms are confirmed.
- **The GPU fit differs from the CPU fit** (dropout's generator, cuBLAS); the parity row shows whether that moves the keep
  numbers beyond the seed spread.
- **A stale cache** against the window the lab trains on: the key covers it; if a promotion moves the window before adoption
  lands, `lab` and the adopted arm are re-run on the new cache first.
- **Memory beside self-play** (section 4) is measured before the sweep shares the machine with it.
- The cache and the JSONs hold no paths or user names; results hold numbers only.

## 7. Piece 3: the quiet-window runner (after this piece has a measured result)

`tools/gpu_lane/quiet_runner.py`: a loop that runs the next queued arm (`vrun_sweep.py run --arms <one>`) only while
**another lane's quiet window is open with that lane's explicit consent**: a `QUIET.lock` whose tag is not `gpu-lane` **and**
a line from that lane in HANDOFF (or a `GPU_OK` marker beside its lock) saying the GPU may be used in its window (a quiet
lock is that lane's quiet machine for its measurements, not consent); the lab's status reads `paused(pause)` or
`paused(lock)` with the window's tag (never `paused(user)`, the owner at the PC, nor `paused(load)`); no `train` step is
running; `LAB/GPU_OFF` is absent. It checks those conditions **between epochs**, not between fits (a 100-epoch fit runs
minutes), aborts and discards the fit when the window closes or the lab resumes, re-checks right after a fit starts, never
takes a lock of its own, never writes `jobs.json`, logs to `agent/gpu-lane/quiet.log` and leaves a HANDOFF line per window
used. Designed in one page after this sweep's table, reviewed, then run.

## 8. Budget

Design and review ~2.5 Opus-equivalent rounds (spent); one Sonnet implementation round and its review ~1.5; the sweep is
GPU time (its per-fit minutes are measured first); the report ~0.5. The usage page, not a round count, is the cap (D30):
every lane stops at a 68% reading.

## 9. Review log

- **2026-10-25, independent Opus review of the first draft: CHANGES**, three blockers and nine required findings, all
  applied above. Blockers: the confirmation split leaked through FightNet's H and the warm start (now FightNet per fold,
  `warm` dropped); a tenth of runs holds ~38 kills, far too few (confirmation is 5-fold cross-fitting over ~13.7k runs);
  the window holds duplicate run records (deduplicated by seed in the cache; the lab finding told to the week-3 lane).
  Required: the lab's own split is not reused for selection (S is [0.2, 0.3)); `beta` arms and `allruns` change nothing
  and are replaced by the data curve and the logistic diagnostic; the adoptable set is named, at most two pre-declared
  arms are confirmed, `ens5` is a diagnostic, `warm_start` is dropped; the expected-floor error is measured beside P(kill)
  and adoption no longer flips the macro by itself; the GPU and RAM are shared with the inference server and are measured
  before the sweep runs beside self-play; the CUDA path's details (device before optimiser, CPU evaluation and save,
  CPU shuffle generator, no per-batch `.item()`, rows resident on the GPU, a golden bit-for-bit test, `rows_per_run`
  instead of weights); the cache key; the piece-3 runner's consent, pause kinds, epoch-level checks and `GPU_OFF`. Smalls:
  the 1,863-wide input, every ECE bin reported, `lab-p10`, per-epoch S Brier, the generator state after FightNet, no
  absolute paths in `train.json`. The draft's unsupported figures (1,615 held-out runs, ~14k runs, ~85 kills, the beta
  blend, ~1 GPU-minute per arm) were corrected or marked estimated.
