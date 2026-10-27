# The combat prior's training recipe, swept on the idle GPU (GPU lane, week 4 item 2, D36)

Design, 2026-10-27. One page, Opus-reviewed before any fit (section 8). Status words as in `GPU_INFERENCE.md`.

## 1. The question

The lab fits the combat prior (`CombatNet`, the policy head over the action tokens) each generation with one recipe:
root-visit targets, 16 epochs, `d_model` 128, 2 layers, on 6,400 fights sampled from the window (`gen_loop.MAX_FIGHTS`),
about 2.7 GPU-minutes (measured, generation 5's `train.json`). Week 3's A2c-target round measured completed-Q targets
against visits at one size and found the dev-set search regret lower at T = 1 and T = 3 but with a CI crossing 0, and the
net alone better at T = 1 (A2c-target report). Nothing has measured the size of the net, the epochs or the data amount.
**Which recipe gives the best prior, judged by what the search gets from it, and does any beat the lab's recipe by a margin
the held-out searches can resolve?**

## 2. Data (measured 2026-10-27)

The current window: generation 7's, generations 5-6, 5 closed blocks, 7,243 runs, 690,342 combat samples, every run played
on the GPU prior server. The training set is built once, exactly as the lab's `train_generation` builds it
(`build_training_set(manifests, max_fights=6400, seed=7)`: fights sampled, the held-out tenth of fights by run index is
`train_combat`'s own), saved as one `train.npz` shared by every arm (as `rl/a2c_target.py train` does). One arm doubles the
sample (`max_fights` 12,800) to read the data lever. The window is static while the lab is paused; its sha is recorded.

## 3. Arms (2 seeds each; estimated 50 fits, ~6 GPU-hours in quiet windows through the runner)

| factor | levels |
|---|---|
| policy target | root visits (the lab's), completed-Q at T = 1, completed-Q at T = 0.5 |
| epochs | 16 (the lab's), 32 |
| `d_model` | 128 (the lab's), 256 |
| layers | 2 (the lab's), 4 |
| data (the lab's recipe only) | 6,400 fights (the lab's), 12,800 |

The full factorial of the first four is 24 arms; with the data arm, 25; two seeds each (the A2c-target round's warning:
two fits of one recipe differ more than most recipes do). The lab's recipe `visits-16-128-2` is the baseline. Fits run in
the order baseline first, then the single-factor changes, then the rest, so an interrupted sweep still has the readable
part. Each fit is a queue job of the runner (`tool` `prior_sweep`); `train_combat`'s epoch loop gets the stop-file hook so
a fit stops between epochs when a window closes (exit 3, nothing written, retried).

## 4. Measures (every arm) and the search check (three arms)

1. **Held-out expected completed-Q regret** (the A2c-target round's criterion, favouring neither target): on the held-out
   fights' searches, the max over legal actions of completed Q minus the net's policy-weighted mean, in HP; also the top-1
   regret, the top-1 agreement with the most-visited action and the policy entropy. Paired bootstrap over held-out runs of
   (arm minus baseline), 2,000 draws, fixed seed.
2. **Dev-set net alone** (`rl/a2c_target.py`'s path on the 200 labelled A3 positions, `agent/devset`): expected regret and
   top-1 regret in HP, population-weighted, paired bootstrap against the baseline. No search: cheap for 50 nets.
3. **Dev-set search** (`a3-v1` with the prior, 24 seeds per position, 8 emulator processes, ~15 minutes per arm): only for
   the baseline, the best arm by measure 1 and the best by measure 2 (when different). This is the week-3 instrument and is
   underpowered for a 0.05 HP gap (the A2c-target report), so it is reported, not decided on.
4. Fit minutes, the validation policy and value losses, the GPU peak memory.

## 5. The rule (pre-registered)

The **winner** is the arm with the lowest mean held-out expected regret over its two seeds, provided both of its seeds beat
the baseline's worse seed and the paired bootstrap CI of (arm minus baseline) lies below 0; its dev-set net-alone regret
must not be worse than the baseline's beyond that measure's CI. The winner is **not adopted by this sweep**: it goes to a
paired E-block A/B (a lab job: the winner's recipe trained on the live window against the lab's recipe, both arms on the
server transport, the promotion judge's statistic and guard) once the week-4 lane has resumed the lab; adoption means a
reviewed change of `train_generation`'s defaults after that A/B, with a HANDOFF line. If no arm meets the rule the report
says so, with the per-factor reading (which factor moves the regret, by how much, against the seed spread).

## 6. Code (fork, reviewed; the lab's default path unchanged)

- `rl/prior_sweep.py`: `build` (the shared `train.npz` and the data arm's), `run --arms ... --seeds 2 --device cuda
  --stop-file` (one fit per call through `gen_loop.train_combat_prior`, the held-out measure and the dev-set net-alone
  measure written beside the net as JSON), `search --arms ...` (measure 3 through `a2c_target`'s functions), `table`;
  arms named in the file; `--budget-min` and the stop file for the runner.
- `rl/train_combat.py`: an `epoch_end` hook (None by default; the loop unchanged without it) so the stop file is honoured.
- `tools/gpu_lane/quiet_runner.py`: queue jobs carry `tool` (`vrun_sweep` or `prior_sweep`); the runner picks the module.
- Tests on synthetic shards: the grid and its order, the saved set is reused, the held-out and dev-set arithmetic on
  synthetic predictions, the stop file between epochs, the queue's `tool` field, no absolute path in any written JSON.

## 7. Cost, where it runs, and risks

Design and review ~0.6 round, implementation and its review ~1.0, the fits are GPU time in quiet windows (the GPU is
idle now; the lab is paused under `w4-unstick`, so rule 3 of the runner's spec opens the window; the dev-set searches take
the CPU for ~45 minutes in all and run only in an open window too). Risks: two seeds resolve only large differences (the
rule asks both seeds to agree; the E-block A/B is the decider); the window is one generation's play and static; the held-out
regret is the searches' own value and favours neither target but does favour nets that imitate the search's visit
distribution, which is why the dev-set net-alone measure is read beside it; 50 fits write ~50 x 3 MB of checkpoints under
`agent/gpu-lane/prior/` (the fork holds none of them).

## 8. Review log

(filled in after the review)
