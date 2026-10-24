# Kit ranking-provenance-1 (Python only; the lockstep review's recommendations 5, 6 and 11 and the lone-End-Turn bucket; batch `cloud/ranking-1`)

## Header
- Base: fork `main` at `b163b4cc` or later, branch `cloud/ranking-1`; one commit per item below.
- Cloud half: each item with its test (fails before, passes after), the Python suite green (`tests/python`, no new failing name), a PR to `main`. Local half: the landing session re-runs the ranking on the stored blocks (`agent/gen0/real-s2-w3`, `agent/gen3/real-s3`) and compares the tables.
- Allowed-file footprint: `tools/lab/rank.py`, `tools/lab/by_family.py`, `referee/lockstep.py` (item 3 only: the record context), `referee/envproto.py` (summary only), their tests under `tests/python/`. No C#, nothing under `Core2/`, `tools/kernel/` or `docs/rebuild/`.
- Expected corpus footprint: none (no engine change); byte proof identical.
- Run-first counts: Linux cloud per `docs/CLOUD_ENV.md`; PC at `26c19ef2`: Sts2Emulator.Tests 6,937 / 0 / 7, EmuDump.Tests 334, Python 3,008 tests with no new failing name.
- Stop rules: stop and report if a change would alter any existing group's weight under the span-weighted ranking (item 1 adds weights beside it, it does not replace it); if `turn_advanced` is absent from the records you test with (it is written by `Lockstep._track` since main `d0187715`; synthesise it in tests); if a reader of `ranked.json` outside `tools/lab` breaks.

## Symptom
`docs/reviews/2026-10-24-lockstep-and-ranking.md` section 3: the agent weight is span-to-next-anchor (bridge geometry), so group weights are not additive and a fix order that follows it moves with the bridge; `emu_outcome` provenance is implicit (now `rank --emu-dir`, but the report does not say which runs lacked an arm); the lone-End-Turn records (about 24% of block S2 before main `26c19ef2`) sit in F06 because `by_family` reads only `acted.type`, although every record now carries `turn_advanced`.

## Expected behaviour
1. `rank` publishes three weights per group beside the current one: divergence count, span-weighted (today's), and span capped at five decisions; `top20.md` shows the rank under each and a rank-stability line (Spearman or Kendall between the three orders over the top 20).
2. `rank`'s summary and `top20.md` state the outcome rule per run: how many run ends were joined from the emulator arm, how many fell back to the proxy, and the arm folder; `by_family.md` repeats the line (it already reads the same arm).
3. `by_family`: a record with `turn_advanced == true` whose diff is a hand (with or without hp) and whose `acted.type` is `play` is filed under a new rule R12d/R13d "lone End Turn" (coarse family "turn boundary", evidence `shape`) ahead of R12b/R13b; the table shows the bucket's weight; records without `turn_advanced` keep today's placement.
4. Housekeeping (recommendation 11): the golden test compares normalised text (done on main; keep it), and `by_family.md`'s header names the rank-table version (the rule ids and the records' context keys) so a snapshot can be told from a stale one.

## Scenario tests
`tests/python/test_lab_rank.py`: the three weights on a synthetic block (a group with one long span ranks first by span, last by count); the stability line's value on a constructed order; the outcome-rule line with and without an arm. `tests/python/test_lab_by_family.py`: a hand diff after a play with `turn_advanced` true lands in the lone-End-Turn bucket, the same record without the key keeps F06, a record with `turn_advanced` false after a play keeps F06.

## Fix sketch
`rank.weigh` gains a `mode` (count / span / span5) and `rank.write` emits the three; `top20.md`'s table gets three rank columns and a stability line; the summary dict gets `outcome_rule_runs: {joined, proxy}`. `by_family.RULES` gains R12d/R13d before R12b/R13b with the `turn_advanced` test; the header line adds the table version.

## Confidence
High: every item is a reporting or placement change described in the review with the data already in the records; no game rule.
