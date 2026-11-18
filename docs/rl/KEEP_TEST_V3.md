# The keep rule v3 (D66: the run judge decides on Act 3 kills and wins; floors only catch breakage)

Owner, 2026-11-18 (D66): "The honest fix is to play this brain on 2,000 fresh seeds and decide on Act 3 kills and wins alone, and to change
the judge so floors only catch breakage." v3 is a mode of the keep test v2's module (`tools/lab/keep_v2.py`: `judge(..., rule="v3")`,
`seq_call(..., rule="v3")`), so the frontier, deep_f, the bootstrap, the funnel's looks and the ledger columns are those of
[KEEP_TEST_V2.md](KEEP_TEST_V2.md); v2 is unchanged and stays selectable. Constants: `RULE_V3` = `keep-v3-2026-11-18`, `LEDGER_RULE_V3` = `v3`.

## The rule

* **Route A (deeper) is the only route to a promotion.** deep_f's gain per 100 runs (frontier f as v2, paired by seed, cluster bootstrap) has
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

## The ledger

The nine v2 columns, with `rule` = `v3`, `route` = `A`, `X` (an extension) or empty and `decision` = `promoted`, `extend` or `rejected`. A
pooled row has the second block's name and the pooled pairs in `seeds`; the lineage's drift check reads the ledger's Act 1 columns, so a
pooled row counts the first block's Act 1 kills a second time (the check is advisory under v3). `gen_loop.row_rule` reads `v3`.

## Run it

Dry run (orchestrator reads it, then the live command without `--out`):
`gen0.py rejudge --cand <agent>/gen18/candidate-cheap-v3d --base <agent>/gen17/candidate-cheap-v2 --seeds-file <lab>/seeds/E18-2026-11-18-r2.json
--pool-block E18-2026-11-18-r1 --agent-root <agent> --review-guard --keep-rule v3 --note "D66: pooled r1+r2 under keep rule v3" --out DIR`.
The lab's train job is switched to `--keep-rule v3` by editing `LAB/jobs.json` (the orchestrator's step).
