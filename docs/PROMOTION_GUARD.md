# The promotion guard: D24 as the independent review of 2026-10-24 revised it

The lab promotes a candidate generation when it beats the incumbent on a fresh 2,000-seed emulator E-block. Decision
D24 made the Act-1 boss-kill check a *not-worse guard* while either arm kills the boss in under 10% of runs, because a
co-primary at a 1 to 2% kill rate has no power. The independent review (section 4 of its 2026-10-24 report, copied in
the `review/input-2026-10-24` branch under `docs/review-input/d24-guard-input.md`) computed that guard's power, found it
"nearly free of false fires and nearly free of power", and proposed a better one. This document is that guard: the rule
in words, its power, how to switch it on, and what the PC must check first.

**Status: implemented behind a flag that is off.** The code is `tools/lab/promotion_guard.py` (pure functions, standard
library plus numpy for the one bootstrap) and the flag-gated call in `rl/gen_loop.py` (`promote`, `rejudge`,
`mark_ready`). With the flag off the lab decides exactly as before; `tests/python/test_promotion_guard.py` checks that.

## 1. The rule in words

What does not change:

- **Primary.** The paired mean floor on the block, cluster bootstrap over seeds, one-sided alpha 0.01: the lower bound
  of the candidate's floor advantage must be above 0.
- **Co-primaries at 10%.** Once both arms kill the Act-1 boss in at least 10% of runs the boss kill is a co-primary
  (exact McNemar, one-sided 0.01, the candidate ahead); wins join as a co-primary once both arms win at least 10%.

What the review's guard adds while either arm is under 10% (the "guard mode" of D24), each computed from the paired
per-seed outcomes of the two arms on the block:

1. **The boss-kill guard at one-sided 0.05** (D24 ran it at 0.01). On the discordant pairs (seeds where exactly one
   arm killed the boss), the exact one-sided McNemar test: with `b` candidate-only and `c` incumbent-only kills,
   `p = P(X >= c)`, `X ~ Binomial(b + c, 1/2)`. The guard fires, and the promotion is refused, when `p < 0.05` and
   `c > b`. The review's argument: a false fire costs one retrial on a fresh block, a missed collapse costs generations,
   so a guard should lean toward firing.
2. **The boss-damage paired guard.** Damage dealt to the Act-1 boss as a fraction of its HP, 0 when the boss was not
   reached, paired by seed, with the floor primary's cluster bootstrap (the same resampling code and seed), not-worse at
   one-sided 0.05: the guard fires when even the upper bound of candidate minus incumbent is below 0. It measures what
   the kill check is for (the ability to win the boss fight) with the power of a continuous variable, long before kills
   reach 10%. It needs one field, `boss_damage`, in every row of both arms' `runs.jsonl`; while any row lacks it the
   guard reports itself unavailable and never fires.
3. **The drift check across generations.** The candidate's kill rate on its block against the promoted lineage's pooled
   rate over its last three blocks (6,000 seeds), unpaired, Wilson 95% intervals. *Flagged* when the candidate's upper
   bound is below the pooled lower bound: the promotion goes through but is marked for the weekly review (`review` in
   the decision, `flagged` in the tsv). *Blocked*, and the promotion refused, when that happens twice in a row. The
   lineage's kills on a block are the promoted candidate's where the block promoted it, else the incumbent's.
4. **The retry cap.** At most two promotion attempts per generation. The lab grants no third self-play budget (no
   `ready-<gen>-r2` marker) after a generation's second rejection, `promote` refuses a third candidate before any run is
   played, and every tsv row records the attempt number. The review's arithmetic: with two attempts per generation the
   expected false promotions over 20 generations are about 0.4, not 0.2, and without a cap they grow without bound.

The decision with the flag on: promote when the primary (and the co-primaries, where active) pass and no guard blocks:
the kill guard does not fire, the damage guard does not fire, the drift check is not blocked, and the attempt is within
the cap. The verdict and every number it used go into the candidate's `promotion.json` under `review_guard`, and three
columns join `promotions.tsv`: `attempt`, `drift` (clear / flagged / blocked / skipped; empty while the flag is off) and
`guard` (one line: each part's verdict and its key number; `off` while the flag is off). The decision's `rule` reads
`D24+review-2026-10-24-4.3` instead of `D24`.

Two things the review said that are not code: re-judging a block under a changed rule is post hoc and applies from the
next block (so the flag is flipped between generations, and `rejudge --review-guard` exists only for an owner decision
that says so); and promotions at the margin are subject to the winner's curse, which is why the headline number stays
the weekly real-game block.

## 2. Power

The arithmetic of the review's appendix A, re-run by `python -m tools.lab.promotion_guard --table` (the tests check
these tables against the review's numbers): for `n` paired seeds with kill rates `p1` (incumbent) and `p2` (candidate),
discordance `d = p1 + p2 - 2 p1 p2` under independence of the two arms' kills on a seed, the number of discordant pairs
`m ~ Binomial(n, d)`, the incumbent-only count `~ Binomial(m, (p1 - p1 p2) / d)`, the guard fires at the smallest `k` with
`P(X >= k | m, 1/2) <= alpha`, and the power is the mixture over `m`. Seeds for 80% power use the normal approximation
on the discordant pairs. Real arms are positively correlated (D24 saw 53 discordant pairs where independence predicts
66), which lowers discordance and so the true power: these are upper bounds.

Guard mode: the probability that a candidate whose kill rate fell is stopped (independent arms).

| incumbent -> candidate | 500 seeds, 0.01 | 500 seeds, 0.05 | 2,000 seeds, 0.01 | 2,000 seeds, 0.05 | seeds for 80% power, 0.01 | seeds for 80% power, 0.05 |
|---|---|---|---|---|---|---|
| 2% -> 1.5% | 2% | 10% | 11% | 29% | 13,732 | 8,447 |
| 2% -> 1.3% | 4% | 16% | 24% | 49% | 6,570 | 4,034 |
| 2% -> 1% | 9% | 28% | 57% | 81% | 2,879 | 1,758 |
| 2% -> 0.5% | 31% | 61% | 98% | 100% | 990 | 591 |
| 1% -> 0.5% | 3% | 12% | 25% | 52% | 5,799 | 3,542 |
| 5% -> 4% | 4% | 15% | 19% | 42% | 8,600 | 5,293 |
| 5% -> 2.5% | 34% | 62% | 96% | 99% | 1,127 | 688 |
| 10% -> 8% | 9% | 26% | 43% | 70% | 4,099 | 2,523 |
| 10% -> 5% | 72% | 90% | 100% | 100% | 543 | 331 |

Co-primary mode: the probability that a real improvement passes (one-sided 0.01).

| incumbent -> candidate | 500 seeds | 2,000 seeds | 4,000 seeds | seeds for 80% power |
|---|---|---|---|---|
| 1.35% -> 2% | 4% | 20% | 44% | 7,748 |
| 2% -> 2.5% | 2% | 9% | 19% | 17,603 |
| 2% -> 3% | 6% | 35% | 68% | 4,843 |
| 2% -> 4% | 25% | 91% | 100% | 1,419 |
| 5% -> 7.5% | 20% | 81% | 99% | 1,864 |
| 10% -> 13% | 18% | 73% | 97% | 2,262 |
| 10% -> 15% | 49% | 99% | 100% | 871 |

False fires of the kill guard with equal arms (the exact test is conservative; the nominal level is 0.05).

| kill rate in both arms | 500 seeds | 2,000 seeds |
|---|---|---|
| 1% | 2.3% | 3.7% |
| 2% | 3.2% | 3.9% |
| 5% | 3.7% | 4.3% |
| 10% | 4.0% | 4.5% |

Reading the tables:

- The 500-run figure is the real-game scoring block; the promotion block is 2,000 seeds. At 500 seeds no setting of the
  kill guard sees anything short of a collapse; at 2,000 the 0.05 guard catches a halving four times in five and a
  one-third drop half the time, against 57% and 24% at 0.01.
- The 80% column is why there is a damage guard: 80% power against a one-quarter drop in a 2% kill rate needs 8,400
  seeds at 0.05 and 13,700 at 0.01, four to seven promotion blocks per generation. No binary guard at these rates can
  be paid for; a continuous boss-fight measure can.
- The compounding case (six generations each losing a quarter of the kill rate, 2% to 0.4%), with the power falling as
  the rate falls: the 0.05 kill guard never fires along the way with probability 0.30, the 0.01 guard 0.71 (the review
  quoted 1/8 and 1/2 with the power held at the first generation's). The drift check, simulated on the same path with
  2,000-seed blocks and a lineage that starts at 2%, flags at least once with probability 0.84 and blocks at least once
  with probability 0.25 (20,000 simulated lineages); it flags a one-step halving from 2% two times in three. Real,
  positively correlated arms lower all of these.
- The cost of leaning toward firing: with equal arms the kill guard fires in about 4% of generations and the damage
  guard in 5%, so about one promotion in eleven is a spurious retrial on a fresh block; the cap bounds the retrials.

## 3. How to flip the flag

The flag is named after the review that proposed the guard. It is read where the decision is made, so it can be set in
either of two ways, and it is **off** unless set to `1`, `on`, `true` or `yes`:

- **The environment variable `SPIRELINE_REVIEW_GUARD=1`** in the environment of the lab's runner (the shell or the
  scheduled task that starts `tools/lab/runner.py`). The runner passes its environment to every job, the `train` job's
  promote subprocess inherits it, and the self-play job that writes the `ready-<gen>` markers sees it too, so one
  variable switches the decision, the tsv columns and the retry cap together.
- **`--review-guard`** on `python rl/gen0.py train --promote ...`, `promote` and `rejudge` (the `train` job's command in
  `LAB/jobs.json` can carry it). The explicit switch wins over the variable; without it the variable decides.

Flip it between generations, never while a block is being judged (the review: rule changes apply from the next block).
Flip it back by unsetting the variable and dropping the switch; nothing else changes, and the tsv rows written while it
was on keep their columns.

What the owner sees afterwards: `promotion.json` carries `review_guard` with the four verdicts and their numbers,
`promotions.tsv` the three new columns, and a flagged drift reads `review: true` in the decision: that promotion goes on
the weekly review's list. After a generation's second rejection the lab stops granting it budgets: self-play goes on at
that generation, and the next step (a retrain by hand, or a search-side change) is the owner's decision.

## 4. What the PC must check before flipping it

Everything above assumes the two arms are *paired*: every seed is played by both arms, and a seed means the same game in
both. The cloud cannot see the PC's blocks; these checks belong there, on a finished promotion block (`arms/<block>-<config>/runs.jsonl`
of the candidate and of the incumbent), before the flag goes on.

1. **The real per-seed pairing of the two arms.** The seed sets of the two `runs.jsonl` files are identical and equal to
   the block's seed file; `paired_seeds` in `promotion.json` equals `n_seeds` (the code refuses a partial arm, but check
   the arms were not played on different configs or ascensions: `summary.json` of each arm names them). Both arms were
   played on the same engine build (the same tree sha in the job record), since a seed is the same game only under the
   same engine. A crashed run counts as floor 0 and no kill in whichever arm it crashed; the crash counts in
   `promotion.json`'s two summaries should be small and alike.
2. **The pairing shows in the discordance.** Under real pairing the arms are positively correlated: the discordant pairs
   `b + c` on the block are *below* `n (p1 + p2 - 2 p1 p2)` from the two arms' kill rates (D24: 53 against 66). Equal
   or above it means the arms behaved as if independent, and the pairing (or the engine's determinism per seed) is in
   doubt. The same check on the floors: the paired standard deviation of the floor differences should be well below
   the square root of the sum of the two arms' floor variances.
3. **The `boss_damage` field.** The damage guard is inert until `rl/gen0.py`'s run row (`play_emu_run`) carries
   `boss_damage`: the damage dealt to the Act-1 boss as a fraction of its starting HP, 0 when the run ended before the
   boss, 1 when it killed it. That is a PC change: it needs the engine's boss HP at the fight's end, and the native
   library to test it. The flag can be flipped without it; the kill guard, the drift check and the cap work from the
   fields the rows already have.
4. **The lineage in `promotions.tsv`.** The drift check pools the last three rows' `seeds`, `boss_cand` and `boss_base`;
   rows written since 2026-10-23 have them, older rows are skipped. With fewer than one usable row the check reports
   `skipped` and does not block.
5. **Both venvs reach the module.** `promote` runs in the emulator venv and imports `tools/lab/promotion_guard.py` from
   the fork's root (numpy only); the self-play job calls `mark_ready` in the same venv. `python rl/gen0.py promote --help`
   shows `--review-guard` once the tree is at this change.

Once the PC confirms 1 and 2 on a real block, the flag can go on; 3 is the one further change that makes the damage
guard informative.

## 5. Where the review's points are

| review | code |
|---|---|
| 4.2, run the McNemar guard at one-sided 0.05 | `promotion_guard.kill_guard`, `GUARD_ALPHA`; `gen_loop.decide_review` uses it in guard mode |
| 4.3.1, the drift check over the lineage's last three blocks | `promotion_guard.drift_check`; `gen_loop.lineage_blocks`, `previous_drift_flagged`; the `drift` column |
| 4.3.2, the boss-damage paired guard, cluster bootstrap as the floor primary | `promotion_guard.damage_guard` (`cluster_bootstrap_lower` is the primary's, checked equal in the tests); `boss_damage` |
| side finding (a), cap attempts at two and record the attempt | `promotion_guard.retry_cap`; `gen_loop.mark_ready`, `promote`, `attempts_before`; the `attempt` column |
| appendix A, the power arithmetic | `guard_power`, `coprimary_power`, `seeds_for_power`, `table` (`--table`) |
| 4.2, rule changes apply from the next block | the flag is read per judgement; `rejudge --review-guard` only on an owner decision |
