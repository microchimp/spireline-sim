# The keep rule v3 (D66: the run judge decides on Act 3 kills and wins; floors only catch breakage)

Owner, 2026-11-18 (D66): "The honest fix is to play this brain on 2,000 fresh seeds and decide on Act 3 kills and wins alone, and to change
the judge so floors only catch breakage." v3 is a mode of the keep test v2's module (`tools/lab/keep_v2.py`: `judge(..., rule="v3")`,
`seq_call(..., rule="v3")`), so the frontier, deep_f, the bootstrap, the funnel's looks and the ledger columns are those of
[KEEP_TEST_V2.md](KEEP_TEST_V2.md); v2 is unchanged and stays selectable. Constants: `RULE_V3` = `keep-v3-2026-11-18`, `LEDGER_RULE_V3` = `v3`.

## The rule

* **Route A (deeper) is the only route to a promotion** (for a fight-only swap there is also route S, below, when `--route-s auto` is on). deep_f's gain per 100 runs (frontier f as v2, paired by seed, cluster bootstrap) has
  its one-sided 0.005 lower bound above 0 and a point gain of at least +1.0. There is **no floor margin** and no route F: a floor gain
  promotes nothing.
* **Floors catch breakage only.** The block is refused when the paired mean floor difference's one-sided 99% upper bound is below -5.0
  floors (`checks["floor_collapse"]`, `collapse` in the verdict). Anything above that is recorded (`floor`, the ledger's `floor_lower95`) and
  decides nothing: floors of -1.95 neither refuse nor extend-block a deeper player.
* **Guards that stay:** the crash guard (as v2), the wins co-primary (as v2: once both arms win 10 in 100; a win is already inside deep_f for
  a frontier of B3a).
* **Advisory, recorded, never blocking:** the drift check (the candidate's Act 1 kill rate against the lineage, `promotion_guard.drift_check`;
  `drift["advisory"]` is true, `checks["drift"]` keeps what it would have said, `review` still marks a flagged drift) and the attempt cap
  (as since D65 addendum 2). `decision["advisory_checks"]` names them (`["attempt", "drift"]`); `gen_loop.checks_hold` and `failed_checks`
  take that list.
* **Extension instead of refusal.** A judgement of every seed of a block whose route A has a point gain of at least +1.0 but a 0.005 bound
  not above 0, for a candidate not extended before, with no guard failing (no collapse, no crash, no wins failure), is **extend**:
  `decision["extend"]` true, route `X`, `promote` false; promotion.json's and the ledger's `decision` read `extend` (`gen_loop.decision_text`).
  Nothing is promoted and no attempt is refused. A block the funnel stopped early is not final and does not extend.
* **The pooled judgement is the second look.** The candidate is played on a second fresh block and judged by route A on all the pairs of both
  blocks (its arms `arms/<block>-<config>` and the base's for the same blocks). A candidate that was extended (a ledger row of its identity
  with rule v3 and route X on another block, `gen_loop.extended_before`) or whose judgement pools a block is not extended again: it is
  promoted or refused. Example, from the two blocks of `candidate-cheap-v3d` (deep_f +1.45 per 100, SE 0.72 on 2,000 pairs: extend; the same
  effect on 4,000 pooled pairs, SE about 0.52: the 0.005 bound is above 0, promote).
* **The funnel** (`keep_v2.seq_call_v3`, `gen_loop.funnel(keep_rule="v3")`): the smoke look as D64 (nothing refused on the figures; a
  crashing candidate is still refused); a later look refuses when route A is dead (deep_f's one-sided **99%** upper bound below +1.0 per
  100: not even an extension is in reach) or the floors have collapsed (99% upper bound below -5.0); an interim look is eligible to promote
  by route A's z reaching 3.72 (as v2: 0.001 split over five looks and two routes, kept) with a gain of at least +1.0, and `promote` then asks the
  verdict on the seeds played as in v2. The final look is `judge`'s: promote, extend or refuse.
* **The futility stop** (`FUTILITY_MIN_SEEDS` = 500, on route A only): the 95% upper bound of deep_f's gain against the gain route A needs to
  promote on the whole judgement, its N counting the pooled pairs when a block extends another. On a block that may still be extended (a first
  block of this candidate) the need is +1.0, the gain that earns an extension (`futility_routes(extend_possible=True)`): the block could not
  promote alone, and stopping it for that would defeat the extension.

## Switches

| where | how |
|---|---|
| `rl/gen0.py train`, `promote`, `rejudge` | `--keep-rule {v1,v2,v3}`; default the variable `SPIRELINE_KEEP_RULE`, else v1 (`gen_loop.KEEP_RULES`) |
| the train job's child (`run_promote_subprocess`) | `--keep-rule` is passed on when named; `pool_blocks` become `--pool-block` |
| `rl/gen0.py rejudge --pool-block <block>` (repeatable) | pool the named blocks' finished arms with the seed file's block: the judgement is route A on every pair; the frontier and the lineage leave all the blocks out; promotion.json names each block, its pairs and each side's kills per boss under `pooled`; the ledger row is the seed file's block with the pooled pairs in `seeds` and "pooled ... = N pairs" in `guard`. Refused under v1/v2, for a block with a missing arm or a seed another block holds, and for a block the funnel stopped |
| `rl/gen0.py promote --pool-block <block>` | the same pooling for a candidate being played on its second block (the funnel's looks, its refusals and its futility count the pooled pairs; promotion.json `n_seeds` is the pairs of the whole judgement) |
| `rl/gen0.py rejudge --out DIR` | a dry run, as v2: the agent folder is read only |
| `rl/gen0.py train`, `promote`, `rejudge` `--min-gain <float>` | route A's smallest deep_f gain per 100 runs (2026-11-19, D68 addendum 3), default 1.0, at least 0. It is a parameter of `keep_v2.judge`, `seq_call`, `futility_routes` and the functions under them (`min_gain=`), never a changed global: the judgement, the funnel's refusal of a dead route A and early promotion, the futility stop and the extension (a minimum that holds with a bound that does not) all use the one value. The bound check (0.005, one-sided, above 0) is not changed. Recorded: `keep_v2.min_gain` and `keep_v2.routes.A.min_gain` always; `decision.min_gain` and `min gain <value>` after the rule's name in the ledger's `guard` cell (`v3 min gain 0.5 route A \| ...`) only when it is not 1.0, so a default judgement writes the promotion.json and the row it always wrote. An error under v1. The train job passes it to its judging child, and to the extension's second block |
| `rl/gen0.py train`, `promote`, `rejudge` `--route-s {auto,off}` | route S (below). Default `off`: v3 exactly as above. `auto` is the keep rule v3's (an error under v1/v2, before anything is trained or judged). The train job passes it on to its judging child (`run_promote_subprocess(route_s=)`, only when it is `auto`) |
| `rl/gen0.py rejudge --route-s auto --s1 <results.json>` | rejudge a candidate under route S with an arrival exam's reading already played (`judge-design/S1-v5.sh` writes one), without playing S1 again: the reading is checked against the champion's ruler hash and the two fight networks (a mismatch is not applied, the reason is recorded) and its test is entered in the ruler's ledger. Without `--s1` the candidate's own `s1.json` is read, else v3 decides |
| `rl/gen0.py rejudge --route-s auto --s1 ... --complete-block` (with `--procs`, `--ascension`, `--lib` and the prior-server flags) | a block the funnel stopped before its last look has too few pairs for S2's final bound: play the seeds that have no row (the funnel's own resume, `run_arm(resume=True)`: the stored rows are kept and nothing is played twice) and judge all of them under S2. Without it the rejudgement stops with an error that says how many pairs are missing; with a dry run (`--out`) it is refused, since it plays games |

## The ledger

The nine v2 columns, with `rule` = `v3`, `route` = `A`, `X` (an extension) or empty and `decision` = `promoted`, `extend` or `rejected`. A
pooled row has the second block's name and the pooled pairs in `seeds`; the lineage's drift check reads the ledger's Act 1 columns, so a
pooled row counts the first block's Act 1 kills a second time (the check is advisory under v3). `gen_loop.row_rule` reads `v3`.

## Route S: the boss-door route for fight-only swaps (D66 as amended by D68 addendum 6 outcome 4, 2026-11-22)

**Why.** A fight brain's steps are small and whole runs cannot see them: v3's route A needs 8,500 pairs (18 h) to see a true +1.0 per 100 runs at its bound, and
a step the fight gate measured in Acts 1-2 does not show in deep_f (fitted: 0.00 +/- 0.17 per 100 runs per fight point). Route S judges the fight brain where it acts, at
the first Act 3 boss, and keeps the whole-run judge as the breakage check. The design and the figures are in the judge-design memo (`DESIGN.md` sections 3 and 5); the
owner adopted it on 2026-11-22 (D68 addendum 6 outcome 4), which amends D66: a fight-only swap can be promoted without a whole-run Act 3 gain.

**When it applies.** Only with `--route-s auto`, and only to a candidate whose `agent.json` says `built.swap: fight` (the field `rl/gym_wiring.py` writes for the champion with only its fight
network replaced; `gen_loop.fight_only_swap` reads it and nothing else). Every other candidate, and every judgement with the flag `off` (the default everywhere), is judged by v3 as above.

**S1, the arrival exam** (`rl/act3_starts.py arrival-exam`; the reading is `gym_gate.arrival_exam`'s). The candidate's fight network plays against the champion's from the champion's own
first-Act-3-boss arrivals (the ruler: its first-boss arrivals in its judged base arms on blocks it was not selected on, rebuilt to the boss door and kept only when the replay matches the row;
`gym-loop/arrival/<champion folder name>/`), 6 paired draws a start, config `a3-v1`, the CPU prior. The reading is the per-start mean survival difference in passes per 100 arrivals, its SE over the
starts. **S1 holds when the mean is at least `S1_MIN` (+1.0 per 100 arrivals) and the mean minus `S1_Z` SE (2.576, the one-sided 0.005 bound) is above 0** (`keep_v2.route_s1`). A reading whose
ruler hash, candidate or champion (the sha256 of the fight network) does not match the pair it is applied to, one that is incomplete, and a candidate that is not a fight-only swap are **not applied**:
the reason is recorded (`keep_v2.route_s.s1.reasons`, the ledger's `guard` cell: `S1 not applied (...)`) and v3 decides.

**S1 inside `promote`** (`gen_loop.s1_stage`, after the cheap fight screen and before any arm is played). The champion's ruler is used when its `meta.json` names this champion and its starts file hashes
to the meta's hash (`act3_starts.load_ruler`), else built there (`arrivals`, 12 processes, about 1.6 h). The test index k is one more than any test played against the ruler (the ledger
`s1-ledger.json` beside it, and any draws a games file in its `exam` folder holds), so no draw (6k to 6k + 5) is used by two candidates; a resumed or repeated judgement of the same candidate keeps its own k, and a
reading that stands in the candidate's folder (`s1.json`, for this ruler, champion and candidate) is not played again. 12 processes. Both children inherit the train job's priority (BelowNormal), hold no QUIET.lock line and are
passed the lab's own tag with `--allow-quiet lab-train` (the exam would otherwise read the runner's line as another's and stop); the gym daemon is not touched. A stop or a block (the night window, PAUSE, HALT, a
landing, another's line) ends the stage with exit status 3: the job fails, the runner retries it, and the ruler's replays and the exam's games resume. The reading goes to `<candidate>/s1.json` and to promotion.json as `s1`.
A champion that has no arms to build a ruler from, or a candidate that holds the champion's own fight network, is S1-unavailable: v3 decides and `route_s_mode.why` says so.

**S2, the whole-run breakage check** (only when S1 holds). The funnel over the block (the D51 looks, 2,000 pairs) with its looks **refusing only** (`seq_call_v3(route_s=True)`): from `S2_FROM_PAIRS` (500)
pairs on, a look refuses when deep_f's one-sided 99% upper bound (`S2_LOOK_ALPHA` 0.01) is below 0 or the floors have collapsed (the existing collapse test). **No futility stop, no early promotion, no extension.**
The last look is `judge`'s: it refuses when deep_f's one-sided 95% upper bound (`S2_FINAL_ALPHA` 0.05) is below 0, and otherwise promotes if the collapse, crash and wins guards hold. The champion's stored arm on a block the
candidate never played is reused only after `BASE_REUSE_SEEDS` (20) of its seeds replay with the same outcome rows (`gen_loop.OUTCOME_FIELDS`: index, win, floor, end, crash, lost, nodes, cards, rests, boss damage) on the current tree
(`base_arm_reuse`; the champion replays the first 20 seeds of the block it holds and the candidate does not, each keeping its index in the block, into `<candidate>/arms-check/`, with the arms' own prior transport);
if one differs the stored arm is moved to `<champion>/arms-stale/` and the champion plays the block afresh. The check is recorded in promotion.json as `base_reuse` (nothing to check when the candidate played the rows beside the champion).

**What is recorded.** promotion.json: `s1` (the reading: mean and SE in passes per 100 arrivals, starts, draws, test, ruler hash, the two networks, `match`), `route_s_mode` (what was asked, whether it applied, why not),
`state_shift` (the whole-run pass rate per 100 arrivals' difference minus S1's, with its SE), `base_reuse`, `sequential.route` = `S`, and under `keep_v2.route_s` the parameters (`s1_min`, `s1_z`, `s2_from_pairs`,
`s2_look_alpha`, `s2_final_alpha`), the S1 record, S2's bounds (deep_f's 95% and 99% upper bounds, the final check) and which rule decided. The ledger's `rule` column stays `v3`; `route` is `S`, or `AS` when route A holds too
(a refused S2 is the empty route: no extension); the `guard` cell shows `S1 +x.xx ±y.yy per 100 arrivals (N starts, 6 draws, ruler <8 hex>): holds` and `S2 deep 95% upper +z.zz (final bound 0: ok; 99% upper +w.ww)`. The ledger's
columns are unchanged.

**Parameters** (named constants of `tools/lab/keep_v2.py`, recorded with every verdict): `S1_MIN` 1.0, `S1_Z` 2.576, `S2_FROM_PAIRS` 500, `S2_LOOK_ALPHA` 0.01, `S2_FINAL_ALPHA` 0.05; in `rl/gen_loop.py`: `S1_PROCS` 12, `BASE_REUSE_SEEDS` 20.

**Rejudging a judged candidate under S** (`rl/gen0.py rejudge --route-s auto --s1 <results.json> [--complete-block]`, see the switches above). A block the funnel stopped (v5's `futility@1100`) is completed from its stored rows and judged by S2 on every pair.

**The undo.** Remove `--route-s auto` from the train job's command in `LAB/jobs.json` (or set `off`): the next judgement is v3 as above. Nothing in the verdicts of v3 changes with the flag off: no key is added, the child's command line is as it was. A candidate promoted by route S stays promoted;
its `s1.json`, `s1/` folder and the ledger beside the ruler are only records.

**Not verified.** That the champion's stored arms (played with the prior on the GPU server) replay bit for bit through the server on this tree: if they do not, the check refuses the reuse and the champion plays the block afresh, which costs time and breaks nothing.

## Run it

Dry run (orchestrator reads it, then the live command without `--out`):
`gen0.py rejudge --cand <agent>/gen18/candidate-cheap-v3d --base <agent>/gen17/candidate-cheap-v2 --seeds-file <lab>/seeds/E18-2026-11-18-r2.json
--pool-block E18-2026-11-18-r1 --agent-root <agent> --review-guard --keep-rule v3 --note "D66: pooled r1+r2 under keep rule v3" --out DIR`.
The lab's train job is switched to `--keep-rule v3` by editing `LAB/jobs.json` (the orchestrator's step).
