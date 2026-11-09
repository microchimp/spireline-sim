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

A judgement used to play every seed of its block unless a refusal or an early promotion ended it: tonight's `gen14/candidate-prior-act2` stood at
71 against 76 Act 2 boss kills on 800 common seeds with 2,200 to play (about ten hours of 16 processes) to confirm no promotion. The funnel
now also stops a block that cannot reach one.

**The rule.** At a look at `futility.min_seeds` (default 1,000) paired seeds or more that would play on (the looks of the funnel: 200, then
every 300; the refusals and the early promotion are asked first and are as they were; the last look is judged as ever), the keep test asks,
for each route it can promote by, whether the 95% upper bound of the candidate's gain (the paired cluster bootstrap of the routes' own lower
bounds, `keep_v2.diff_summary`) reaches the gain that route needs on the whole block:

* **route A** (the Act 2 boss and deeper, deep_f per 100 runs): `max(+1.0, z(0.995) x SE x sqrt(n / N))`, the rule's own minimum gain or the
  gain whose 0.005 lower bound clears 0 on all N seeds, whichever is more;
* **route F** (the floors): `z(0.995) x SE x sqrt(n / N)` (D24's alpha 0.01 under `--keep-rule v1`, which has this route alone);

where SE is the standard error of the n seeds played and N the block's seeds (the spread seen so far, the seeds still to come all counted). A
route whose bound does not reach its gain, or is not above 0, is dead. When **no** route is alive the judgement stops: decision `rejected`, the
decision's `rule` ends `+D51-funnel+futility@<seeds>`, `sequential.stop` is `{action: futility, rule: futility@<seeds>, n, futility: the
routes' bounds}`, the ledger row's `guard` cell ends `| funnel futility@<seeds> of <N> seeds`, both arms' rows to that look are kept, the keep
test v2 runs on the seeds played as for a refusal (`keep_v2` and the usual summary), the train job exits 0 so the lab's marker becomes `done-`,
and the job log carries `futility look at <n> of <N> seeds: route A (...): 95% upper bound +x falls short of the +y it needs; route F (...)
...; no route can reach it: the judgement stops` at every look it asked (the last of them, if it stopped). A route that is alive keeps the
judgement going; nothing is asked before `min_seeds`. `promotions.tsv` keeps its 31 columns in their order (the column `rule` stays the keep
rule, `v2`; `row_rule` reads it); `sequential.futility` in `promotion.json` records the switch.

**Switch.** On by default. `gen0.py train` / `promote`: `--no-futility`, `--futility-min-seeds N` (the train job passes them to its promote child
only when named, so the lab's live command needs no change); `rl/round1_config.json` `futility.enabled` (true) and `futility.min_seeds`
(1000), which the round's block step passes to `promote`; in code `gen_loop.promote(futility=, futility_min_seeds=)`, `gen_loop.FUTILITY_ON`,
`FUTILITY_MIN_SEEDS`.

**What it costs and saves** (the normal model of `tests/python/test_futility.py::ModelTests`: 3,000 seeds in the funnel's looks, per-seed SD
7.74 floors and 41 per 100 on deep_f; 200,000 paths; "lost" is the chance that the block would have promoted at its last look and the stop
ended it first):

| true gain (floors, deep_f per 100) | P(promote at the last look) | lost to the stop | stopped | seeds played, mean |
|---|---|---|---|---|
| 0, 0 (level) | 0.010 | 0.0001 | 74% | 2,080 |
| -0.5, -2 | 0 | 0 | 100% | 1,137 |
| +0.2, 0 | 0.127 | 0.0008 | 35% | 2,607 |
| +0.3, 0 | 0.330 | 0.0018 | 17% | 2,811 |
| +0.5, 0 | 0.833 | 0.0021 | 2% | 2,978 |
| +1.07, 0 (generation 11's gain) | 1.000 | 0.0000 | 0% | 3,000 |
| 0, +1.0 | 0.114 | 0.0008 | 38% | 2,580 |
| 0, +2.0 | 0.542 | 0.0022 | 8% | 2,908 |
| 0, +3.0 | 0.924 | 0.0014 | 1% | 2,991 |
| 0, +5.0 | 1.000 | 0.0000 | 0% | 3,000 |

The stop loses at most two promotions in a thousand blocks at any gain, and none where the test finds the gain for certain; a level candidate
ends at 2,080 of 3,000 seeds on average (a third saved). **Replay of the lab's recorded blocks** (rows read, nothing played; the test skips
where the folders are absent): none of the five promoted blocks that have their arms (E3-r1, E5, E7-r1, E8 and E12-r3 under v2) stops
for futility at 1,000 or at 500 seeds; the rejected E14-2026-11-07 (2,000 seeds played) stops at 1,100 and E4-2026-10-24 at 1,100.
Tonight's `gen14/candidate-prior-act2`, read on its 1,100 common seeds (floor +0.06, deep_B2 -1.6 per 100): route A is dead (95% upper
bound +0.18 against +1.68) and route F alive by a hair (+0.51 against +0.42), so by this rule it stops at about 1,700 seeds if the means
hold, not at 1,100.

## Tests

`tests/python/test_keep_v2.py` (the rule, the frontier, the routes, the guards, section 6's verdicts from constructed differences, and
the lab's saved arms where they exist), `test_funnel.py` (`V2RuleTests`, `PromoteV2Tests`, including the human-bar line), `test_gen_loop.py`
(`KeepRuleV2Tests`, `test_train_passes_keep_rule_to_promote`, the line through `rejudge` and its command); the line's pieces in
`test_fight_exam.py` (`HumanBarReadingTests`); synthetic arms in `tests/python/keep_v2_fixtures.py` (counts only; no saved row is committed).
`tests/python/test_futility.py` (the futility stop: the routes' bounds, a level candidate stopping at 1,000 seeds, a better one and one better on floors only or on deep_f only never stopping, the ledger line's columns, the switches, the model table, the recorded blocks).
