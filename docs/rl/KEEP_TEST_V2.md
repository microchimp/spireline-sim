# The keep test v2 (promotion on the deepest boss the lineage reaches)

The design, its figures and the retroactive verdicts are in Spireline's `docs/design/2026-11-05-keep-test-v2.md`. This page is the rule,
the switches and the ledger columns as built. The default is unchanged: `--keep-rule v1` (D24's paired mean floor, plus the review
guard when `--review-guard` is on) stays the rule until the lab's train job is told otherwise.

## The rule

`tools/lab/keep_v2.py` (pure, like `promotion_guard.py`: no file read, no run played). A run beats up to four bosses, **B1, B2, B3a** (the
first of A10's two Act 3 bosses) and **B3b** (the win), read from `boss_damage_acts` (the first entry of each act, fraction exactly 1.0; a
win beats all four) and, for a row without the field (blocks before 2026-10-25), from the floor cuts 17 / 33 / 48.

* **Frontier f** = the deepest boss the promoted lineage beats in at least 1.5 of 100 runs, pooled over its last three judged blocks (the
  blocks `lineage_blocks` pools for the drift check; the candidate's arm where it was promoted, the incumbent's where it stayed, never a
  rejected candidate's); B1 when none. `gen_loop.incumbent_frontier(agent_root, rows, config)` reads those arms and fixes f before a seed is
  played. The ladder moves f to B3a, then to the win, by itself; the record is `keep_v2.frontier` in `promotion.json`.
* **deep_f** of a run = how many of the bosses from f on it beats; the paired difference is in points per 100 runs.
* **Route A (deeper)**: deep_f's gain has a cluster-bootstrap lower bound at one-sided 0.005 above 0, a point gain of at least +1.0 per
  100, and the floor guard holds (the paired mean floor's one-sided 95% lower bound is at least -1.0 floors).
* **Route F (further)**: today's floor primary at one-sided 0.005, and deep_f not worse (its one-sided 95% upper bound is not below 0).
* **Promote** when route A or route F holds and the guards hold: the wins co-primary (as before, once both arms win 10 in 100), the crash
  guard (the candidate's crashed or lost seeds at most the incumbent's plus max(2, 0.25% of the block)), the drift check and the attempt cap
  of `promotion_guard` (unchanged). The Act 1 kill guard, the Act 1 and Act 2 damage guards and D24's Act 1 boss co-primary are not checks:
  they are reported as `act1_review` and block nothing (the measure is unconditional, so an Act 1 death is already charged to it).
* **Funnel** (looks as D51: 200, 500, then every 300): `keep_v2.seq_call` on the normal statistics of the floor and deep_f differences.
  Smoke look: refuse when the floor mean is below -1.0 and deep_f's mean is not above 0, or the floor's 95% upper bound is below -1.0.
  Later looks: refuse only when route F is dead (floor 99% upper bound below 0) and route A is dead (that bound below -1.0, or deep_f's 95%
  upper bound below +1.0). Interim looks: eligible when a route's z reaches 3.72 (0.001 split over five looks and two routes) with that
  route's other conditions; `promote` then asks `keep_v2.judge` on the seeds played and promotes only if the route that cleared the z holds
  there. Last look: the test on all the seeds.

## Switches

| where | how |
|---|---|
| `rl/gen0.py train`, `promote`, `rejudge` | `--keep-rule {v1,v2}`; default the variable `SPIRELINE_KEEP_RULE`, else v1 (`gen_loop.keep_rule_of`; any other value is an error) |
| the train job's child (`run_promote_subprocess`) | `--keep-rule` is passed on only when named; the child otherwise reads the variable |
| the cap before training | under v2 `attempts_spent` applies the cap whatever `--review-guard` says: the rule carries the cap |
| `rl/gen0.py rejudge --out DIR` | a dry run: the agent folder is only read; `promotion.json`, a copy of `promotions.tsv` with the new row and (on a promotion) `current.json` go to DIR. Refused together with `--lab` (no lab marker). A re-judge spends no attempt, as before |

First live use (orchestrator, outside a night and not while the lab's train job runs), no `--out`:
`gen0.py rejudge --cand <agent>/gen12/candidate-imit-l05 --seeds-file <lab>/seeds/E12-2026-11-05-r3.json --agent-root <agent> --review-guard
--keep-rule v2 --note "keep test v2"` (no `--lab`: the night is the real-game check). Dry run of the same on the saved arms: route A,
frontier B2, Act 2 boss 107 against 49, deep_B2 +3.05 (0.005 bound +1.50), floor 95% lower bound -0.63, crash 0 / 0, drift clear.

## The ledger and the record

`promotions.tsv` gains nine columns after `macro`: `rule` (`v2`), `frontier` (`B2`), `route` (`A`, `F`, `AF` or empty), `k2_cand` and `k2_base`
(Act 2 boss kills, written whatever the verdict), `d_deep` and `deep_lower` (per 100 runs; the 0.005 bound), `p_deep`, `floor_lower95`. Rows
written before them, and every row judged by v1, read empty there, which is v1 (`gen_loop.row_rule`); `append_promotion` rewrites an older
header. `boss_cand`, `boss_base` and `p_boss` stay Act 1, so `lineage_blocks` (the drift check) is untouched. `promotion.json` gains
`keep_v2` (both routes' numbers, the frontier with its pooled counts and the arms it read, the crash guard, the drift check, the attempt, the
Act 1 and damage figures and `act1_review`) and `keep_rule`; its `sequential` key gains `keep_rule` and `frontier`.

## The human bar (report-only; D55 addendum 11, 2026-11-08)

Every automatic judgement of a fight player prints the human line. Under v2, `promote` and `rejudge` add one key to `promotion.json`,
`human_bar` (`line`, and `network`, `gym_player`, `reading_of`, `date` and the figures where there is a reading), and `gen0.py promote` and
`rejudge` print its `line` beside the verdict. The line names the judged player's **fight network** (the first 12 hex digits of the sha256 of
its combat prior's `combat_policy.pt`, `Agent.describe()["prior_sha256"]`) and the **gym player** that holds the same bytes (`players/<name>`,
or a checkpoint with games in the exam folder), then that player's latest reading on the fight exam's human stratum, from its games file
(`exam/games-<name>.jsonl`, rows of config `a3-v1`): survival and HP lost of the player and of the human on the same starts, the starts and
games counted, the engine tag and the date the games file was last written (rows carry no time):

    human bar: fight network 64d057e174ac (gym player v2): survival 62.0 / human 79.8 in 100, HP lost 38.6 / 33.2 on 400 of 400 starts, 1600 games, engine e911430a94a1, games file written 2026-11-08

or `human bar: no human-stratum games for fight network 64d057e174ac (gym player v1)` when that player has none. **No game is played**:
the lab's candidates share one fight network (generations 11, 12 and 13 all carry the bytes of the gym's version 2, sha256 `64d057e1...`),
so the reading is the gym player's, played by the fight loop's gate or by hand. A network no gym player holds gets the **gym incumbent's**
reading, labelled as the incumbent's (`human bar: fight network X (no gym player holds these bytes); the gym incumbent v2's reading, not
this network's: ...`), and the base agent's network is named in the same line (` | base ...`) when it is another. The line is read from
`fight_exam.LOOP_ROOT` / `EXAM_DIR` (the loop's folder, default `D:/spireline/referee/gym-loop`); a failure to read it gives
`human bar: not read (ErrorName)` and never touches the verdict. No rule, guard, check or ledger column reads it (the ledger row is as before:
the line lives in `promotion.json` and the command output only). v1 judgements are unchanged.

## The futility stop (D55 addendum 18, 2026-11-09)

A judgement used to play every seed of its block unless a refusal or an early promotion ended it: `gen14/candidate-prior-act2` stood at
71 against 76 Act 2 boss kills on 800 common seeds with 2,200 to play (about ten hours of 16 processes) to confirm no promotion. The funnel
now also stops a block that cannot reach one.

**The rule.** At a look at `futility.min_seeds` (default **500** paired seeds; the first landing, 2026-11-09, had 1,000) or more that would
play on (the looks of the funnel: 200, then every 300; the refusals and the early promotion are asked first and are as they were; the last look
is judged as ever), the keep test asks, for each route it can promote by, whether the 95% upper bound of the candidate's gain (the paired
cluster bootstrap of the routes' own lower bounds, `keep_v2.diff_summary`) reaches the gain that route needs on the whole block:

* **route A** (the Act 2 boss and deeper, deep_f per 100 runs): `max(+1.0, z(0.995) x SE x sqrt(n / N))`, the rule's own minimum gain or the
  gain whose 0.005 lower bound clears 0 on all N seeds, whichever is more;
* **route F** (the floors): `z(0.995) x SE x sqrt(n / N)` (D24's alpha 0.01 under `--keep-rule v1`, which has this route alone);

where SE is the standard error of the n seeds played and N the block's seeds (the spread seen so far, the seeds still to come all counted). A
route whose bound does not reach its gain, or is not above 0, is dead. When **no** route is alive the judgement stops: decision `rejected`, the
decision's `rule` ends `+D51-funnel+futility@<seeds>`, `sequential.stop` is `{action: futility, rule: futility@<seeds>, n, futility: the
routes' bounds}`, the ledger row's `guard` cell ends `| funnel futility@<seeds> of <N> seeds`, both arms' rows to that look are kept, the keep
test v2 runs on the seeds played as for a refusal (`keep_v2` and the usual summary), the train job exits 0 so the lab's marker becomes `done-`,
and the job log carries `futility look at <n> of <N> seeds: route A (...): 95% upper bound +x falls short of the +y it needs; route F (...)
...; no route can reach it: the judgement stops` at every look it asked. A route that is alive keeps the judgement going; nothing is asked
before `min_seeds`. `promotions.tsv` keeps its 31 columns in their order (the column `rule` stays the keep rule, `v2`; `row_rule` reads it);
`sequential.futility` in `promotion.json` records the switch.

**Switch.** On by default. `gen0.py train` / `promote`: `--no-futility`, `--futility-min-seeds N` (the train job passes them to its promote child
only when named, so the lab's live command needs no change); `rl/round1_config.json` `futility.enabled` (true) and `futility.min_seeds`
(500), which the round's block step passes to `promote`; in code `gen_loop.promote(futility=, futility_min_seeds=)`, `gen_loop.FUTILITY_ON`,
`FUTILITY_MIN_SEEDS`.

**What it costs and saves** (the normal model of `tests/python/test_futility.py::ModelTests`: 3,000 seeds in the funnel's looks, per-seed SD
7.74 floors and 41 per 100 on deep_f; 200,000 paths; "lost" is the chance that the block would have promoted at its last look and the stop
ended it first; the first look at 1,000 / at 500):

| true gain (floors, deep_f per 100) | P(promote at the last look) | lost to the stop | stopped | seeds played, mean |
|---|---|---|---|---|
| 0, 0 (level) | 0.010 | 0.0001 / 0.0002 | 74% / 75% | 2,080 / 1,963 |
| -0.5, -2 | 0 | 0 / 0 | 100% / 100% | 1,137 / 717 |
| +0.2, 0 | 0.127 | 0.0008 / 0.0019 | 35% / 36% | 2,607 / 2,535 |
| +0.3, 0 | 0.330 | 0.0018 / 0.0043 | 17% / 18% | 2,811 / 2,761 |
| +0.5, 0 | 0.833 | 0.0021 / 0.0058 | 2% / 2% | 2,978 / 2,960 |
| +1.07, 0 (generation 11's gain) | 1.000 | 0.0000 / 0.0000 | 0% / 0% | 3,000 / 3,000 |
| 0, +1.0 | 0.114 | 0.0008 / 0.0017 | 38% / 39% | 2,580 / 2,507 |
| 0, +2.0 | 0.542 | 0.0022 / 0.0059 | 8% / 9% | 2,908 / 2,872 |
| 0, +3.0 | 0.924 | 0.0014 / 0.0045 | 1% / 1% | 2,991 / 2,979 |
| 0, +5.0 | 1.000 | 0.0000 / 0.0002 | 0% / 0% | 3,000 / 3,000 |

The stop loses at most six promotions in a thousand blocks at any gain from 500 seeds (two in a thousand from 1,000), and none where the test
finds the gain for certain; a level candidate ends at 1,963 of 3,000 seeds on average (a third saved), a harmful one at the look after 500. **Replay
of the lab's recorded blocks** (rows read, nothing played; the test skips where the folders are absent): none of the five promoted blocks that
have their arms (E3-r1, E5, E7-r1, E8 and E12-r3 under v2) stops for futility from 1,000 or from 500 seeds; the rejected E14-2026-11-07 (2,000
seeds played) stops at 1,100 and E4-2026-10-24 at 1,100 (500 when the first look is 500). `gen14/candidate-prior-act2`, read on its 1,100 common
seeds (floor +0.06, deep_B2 -1.6 per 100): route A is dead from 800 seeds (95% upper bound +1.25 against +1.61; +0.18 against +1.68 at 1,100)
and route F alive by a hair (+0.51 against +0.42 at 1,100), so by this rule it stops at about 1,700 seeds if the means hold, not at 800 or 1,100.

## The screens before the arms (D55 addendum 18)

`rl/screens.py`, asked by `gen_loop.promote` before an arm is played (so by `train --promote`, and by `gen0.py promote`). It compares the
candidate's folder with the incumbent's by part: **fight** (`combat_policy.pt`, `combat_net.pt`, `combat_net.onnx`), **prior**
(`human_prior.npz`), **value** (the run judgement: `run_value.*`, `fight_net.*`) and **settings** (agent.json: the macro and recipe with the prior's
tag taken out, mask version, the prior's lambda, lambda by act and guard, the stakes schedule, the potion hold). Generation, dates, lineage and
notes are bookkeeping. A screen runs only when one part differs and it is the fight network or the prior; anything else plays the arms as before.

* **`screen:fight`** (only the fight network differs). The candidate plays the fight exam's human stratum (`fight_exam.py play --player-dir
  <candidate> --stratum human --allow-quiet lab-train`: 400 starts, four draws, the machinery of the human-bar line; its games go to
  `exam/games-lab-g<gen>-<folder>-<sha8>.jsonl` and are resumable). The incumbent's side is the games of the gym player that holds its
  network's bytes (`fight_exam.gym_player_of`, the lab's human-bar line). The candidate is rejected when its survival is below the incumbent's, on
  the same starts and draws, by **more than two standard errors** (the bootstrap's over starts, resampled within cells, 400 draws). The screen is
  not applied (and says why, the arms are played) when the exam has no human stratum, the incumbent's network has no human-stratum games, the
  candidate's games did not all get played (the run was blocked or stopped: exit 3) or the two sides' engine tags differ (a difference across
  engines is not a difference of networks). It takes the exam's mutex and a `fight-exam` QUIET.lock line like any exam run; the judgement's own
  `lab-train` line does not hold it (`--allow-quiet`).
* **`screen:prior`** (only the human prior differs). Both priors are scored on the held-out human choices of fifth 4 of the all-player Ascension 10
  pool, the leg `rl/strong_sets.py` selects priors by (D55 addendum 15 item 3; `strong_sets.heldout_sets`, `per_decision`, `pooled`, so the numbers
  are the ones `strong_sets evaluate` prints; the exam fifth is asserted absent; CPU numpy, seconds). The **primary** is the log-loss on the
  held-out **won** runs: the candidate must be better than the incumbent's prior by more than the tie margin (`strong_sets.TIE_NATS`, 0.002 nats).
  The **guard** is the log-loss on **all** held-out runs: the candidate may be worse by at most `strong_sets.LIMIT_B_NATS` (0.02 nats). It is
  rejected when the primary fails or the guard fails. Both differences (candidate minus incumbent, nats) go in the summary and, for a rejection, in the
  ledger's rule column as `screen:prior won=-0.0xx all=+0.0xx`. Checked on the lab's folders: `gen14/candidate-best-players` (prior 56afab28)
  against `gen13/candidate-stakes` (2ebfe6d7) differs in the prior only and passes (won -0.0079, all +0.0152 on the whole fifth: 384 won of 3,601
  runs; the half splits give won -0.008 to -0.013, all +0.017 to +0.018). It is not applied (the arms are played) when the held-out set or either
  `human_prior.npz` is missing.

A rejection is a rejected judgement and plays no arm: `promotion.json` has `decision_text` rejected, `n_seeds` 0, the decision's `rule`
`screen:fight` / `screen:prior` and `screen` with the figures (survival of both, the difference, its SE and z; or both log-losses per kind); the
ledger row has `seeds` 0, an empty `seed_block` (the block stays fresh), the screen's rule in its `rule` column (`screen:fight`, or `screen:prior won=... all=...`) and its line in `guard`; the attempt
is spent as for any rejection (`attempts_before` counts the candidate). A screen that passes goes on to the arms with its line added to the row's
`guard` cell and `screen` in `promotion.json`; one that is skipped says why in the job log (`screen:fight: not applied (...)`) and in `screen`; one
that fails with an error is logged and recorded (`screen.error`) and never fails the judgement. `--no-screen` on `train` and `promote` switches
the screens off. Checked on the lab's folders: `gen14/candidate-prior-act2` against `gen13/candidate-stakes` differs in settings only (lambda by
act: no screen).

## Tests

`tests/python/test_keep_v2.py` (the rule, the frontier, the routes, the guards, section 6's verdicts from constructed differences, and
the lab's saved arms where they exist), `test_funnel.py` (`V2RuleTests`, `PromoteV2Tests`, including the human-bar line), `test_gen_loop.py`
(`KeepRuleV2Tests`, `test_train_passes_keep_rule_to_promote`, the line through `rejudge` and its command); the line's pieces in
`test_fight_exam.py` (`HumanBarReadingTests`); synthetic arms in `tests/python/keep_v2_fixtures.py` (counts only; no saved row is committed).
`tests/python/test_futility.py` (the futility stop: the routes' bounds, a level candidate stopping at 1,000 seeds, a better one and one better on floors only or on deep_f only never stopping, the ledger line's columns, the switches, the model table, the recorded blocks).
`tests/python/test_screens.py` (which screen a pair of folders gets, the fight screen's two-SE rule on constructed games, the prior screen's primary and guard with the best-players case and the lab's real held-out set where present, `promote` rejecting before any arm, the ledger row, the flags).
