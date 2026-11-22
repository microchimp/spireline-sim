"""The keep test v2 (docs/rl/KEEP_TEST_V2.md; the design is Spireline's docs/design/2026-11-05-keep-test-v2.md).

Pure functions over the paired per-seed outcomes of a candidate and the incumbent on one E-block, like `promotion_guard.py`
(no file read, no run played). `rl/gen_loop.py` calls `judge` and `seq_call` only when the keep rule is v2 (``--keep-rule v2``
or ``SPIRELINE_KEEP_RULE=v2``; the default is v1, today's floor primary, untouched).

The rule. A run beats up to four bosses: B1, B2, B3a (the first Act 3 boss) and B3b (the win); since 2026-11-19 a boss is beaten when the run
went past it (`went_past`), not when its health bar read 1.0 (`bosses_beaten_loose`). The **frontier** f is a boss, the
deepest the promoted lineage beats in at least 1.5 of 100 runs over its last three judged blocks (B1 when none), fixed before the
block is played. **deep_f** of a run is how many of the bosses from f on it beats, paired by seed, in points per 100 runs.
Two routes, one alpha (0.005 one-sided):

  route A, deeper   the paired deep_f gain's cluster-bootstrap lower bound at 0.005 is above 0, its point gain is at least
                    +1.0 per 100, and the floor guard holds: the paired mean floor's one-sided 95% lower bound is at least
                    -1.0 floors.
  route F, further  today's floor primary at one-sided 0.005 (the lower bound above 0) and deep_f not worse (its one-sided
                    95% upper bound is not below 0).

Promote when route A or route F holds and the guards hold: the wins co-primary (once both arms win 10 in 100), the crash guard
(the candidate's crashed or lost seeds at most the incumbent's plus max(2, 0.25% of the block)), the drift check (the review
guard's, unchanged). The attempt cap (two candidates per generation under one recipe) is recorded and gates nothing since D65 addendum 2
(`checks["attempt"]` keeps the value the cap gave; `attempt_advisory`). The Act 1 kill guard and the damage guards (Act 1 and Act 2 boss) are *reported*
(`act1_review`) and block nothing; so is D24's Act 1 boss co-primary (the rule's measure is unconditional: an Act 1 death is
already charged to it).

The funnel (`seq_call`) keeps D51's looks and judges each look on normal statistics (SD / root n) of the floor and deep_f differences:
the smoke look refuses a floor collapse that deep_f does not offset, a later look refuses only when both routes are dead, and an
interim look is eligible to promote when a route's z reaches the early z (the 0.001 split over the interim looks and the two
routes); the caller then asks `judge` on the seeds played.

Keep rule v3 (D66, 2026-11-18; docs/rl/KEEP_TEST_V3.md) is a mode of this module (``judge(..., rule="v3")``, ``seq_call(..., rule="v3")``;
``--keep-rule v3``): route A (deeper) is the only route to a promotion and it carries no floor margin, route F is gone, the floors refuse
only a collapse (the paired mean floor difference's one-sided 99% upper bound below -5.0 floors), the drift check is recorded and decides
nothing, and a block whose deep_f gain is at least +1.0 per 100 but whose 0.005 bound is not above 0 is *extended* (route "X"), not refused:
the candidate is played on a second fresh block and judged on both pooled. Rule v2 is unchanged.

The +1.0 is `MIN_GAIN`, the default of the `min_gain` argument of the functions that ask it (since 2026-11-19, D68 addendum 3: `--min-gain` on
`gen0.py train`, `promote` and `rejudge`). A caller that names another minimum gets it in the verdict (`min_gain`, `routes["A"]["min_gain"]`), in the
funnel's looks and in the futility stop; the module's constant is never changed.

Route S (D68 addendum 6 outcome 4; docs/rl/KEEP_TEST_V3.md) is v3's second route to a promotion, for a candidate that is a fight-only swap of the
champion (``judge_v3(..., s1=<reading>)``; with no reading v3 is exactly as above). S1 is the arrival exam's reading (`rl/act3_starts.py arrival-exam`
through `s1_reading`): the candidate's fight network against the champion's on the champion's own first-Act-3-boss arrivals, in passes per 100
arrivals; it holds when the mean is at least `S1_MIN` and the mean minus `S1_Z` SE is above 0 (`route_s1`). S2 is the whole-run judgement as the
breakage check: the funnel's looks refuse only (`seq_call_v3(route_s=True)`: from `S2_FROM_PAIRS` pairs on, deep_f's 99% upper bound below 0 or a
floor collapse), and the final judgement on every pair refuses when deep_f's 95% upper bound is below 0 and otherwise promotes if the collapse, crash and
wins guards hold. Under S there is no futility stop, no early promotion and no extension. The ledger's route is "S" ("AS" when route A holds too).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

from tools.lab import promotion_guard as pg

RULE = "keep-v2-2026-11-05"
LEDGER_RULE = "v2"  # the `rule` column of promotions.tsv; an older row has none and reads as v1
ROUTE_ALPHA = 0.005  # one-sided level of both routes
GUARD_ALPHA = 0.05  # one-sided level of the floor guard, of deep_f's upper bound and of the reported Act 1 guards
FLOOR_MARGIN = 1.0  # route A's floor guard: the 95% lower bound of the mean floor difference is at least minus this
MIN_GAIN = 1.0  # route A's smallest deep_f gain, per 100 runs: the default of every `min_gain` argument below (2026-11-19, D68 addendum 3: a recorded parameter, `--min-gain`)
FRONTIER_RATE = 0.015  # a boss is the frontier when the lineage beats it in at least this share of runs
LINEAGE_BLOCKS = 3  # blocks of the promoted lineage pooled for the frontier (and by the drift check)
CRASH_SLACK = (2, 0.0025)  # the candidate may crash or lose this many seeds more than the incumbent: max(2, 0.25% of the block)
EARLY_ALPHA = 0.001  # all the alpha early promotions may spend, split over the interim looks and the two routes
REFUSE_ALPHA = 0.01  # an interim look refuses when the floor's one-sided 99% upper bound is below 0 (route F dead)
SMOKE_MEAN = -1.0  # D64: no longer a refusal; the smoke look records `smoke_would_refuse` for a floor mean below this when deep_f's mean is not above 0
WIN_GATE = 0.10  # wins join the co-primaries once both arms reach this rate (gen_loop.WIN_GATE)
WIN_ALPHA = 0.01  # the wins co-primary's alpha (gen_loop.ALPHA)
BOSSES = ("B1", "B2", "B3a", "win")
ACT_BOSS_FLOORS = (17, 33, 48)  # rl/run_fields.ACT_BOSS_FLOORS: the usual boss floors (a row with `nodes` names its own, `boss_floors`)
BOSS_READING = "strict-2026-11-19"  # how a boss counts as beaten (`went_past`); the verdicts carry it as `boss_reading`
BOOT_B = pg.BOOT_B
BOOT_SEED = pg.BOOT_SEED
MARGIN_Z = 2.0  # the margin every printed figure carries: plus or minus this many standard errors (D68 addendum 6 decision 1)

RULE_V3 = "keep-v3-2026-11-18"
LEDGER_RULE_V3 = "v3"  # the `rule` column of a promotions.tsv row judged by the keep rule v3
RULE_VERSIONS = ("v2", "v3")  # the `rule` argument of `judge` and `seq_call`
COLLAPSE_FLOORS = -5.0  # v3: the floors refuse only a collapsed player: the paired mean floor difference's one-sided 99% upper bound below this
COLLAPSE_ALPHA = 0.01  # the one-sided level of that bound
EXTEND_ROUTE = "X"  # v3: the ledger's route for a block that is extended, not decided
ADVISORY_V3 = ("attempt", "drift")  # v3: checks recorded with their value that decide nothing (the attempt cap since D65 addendum 2, the drift check since D66)

# Route S, the boss-door route for fight-only swaps (D66 as amended by D68 addendum 6 outcome 4, 2026-11-22; docs/rl/KEEP_TEST_V3.md "Route S")
RULE_S = "route-s-2026-11-22"
ROUTE_S = "S"  # the ledger's route when S1 and S2 hold ("AS" when route A holds as well)
ROUTE_AS = "AS"
S1_MIN = 1.0  # S1: the arrival exam's mean gain is at least this many passes per 100 arrivals ...
S1_Z = NormalDist().inv_cdf(1 - ROUTE_ALPHA)  # ... and the mean minus this many SE (2.576, the one-sided 0.005 bound) is above 0
S2_FROM_PAIRS = 500  # S2's looks refuse only from this many whole-run pairs on (the D51 schedule's second look)
S2_LOOK_ALPHA = 0.01  # a look refuses when deep_f's one-sided 99% upper bound is below 0 (or the floors collapse)
S2_FINAL_ALPHA = 0.05  # the final look (2,000 pairs) refuses when deep_f's one-sided 95% upper bound is below 0
S1_KEYS = ("ruler_hash", "candidate", "champion")  # what a reading must name the same as the pair it is applied to


def check_min_gain(value: Any) -> float:
    """`value` as a float for a `min_gain` argument: finite and not negative (0 asks the bound alone). The one place the parameter is checked."""
    x = float(value)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"min_gain must be a finite number of at least 0 (per 100 runs), not {value!r}")
    return x


# ------------------------------------------------------------------ bosses beaten, the frontier, deep_f


def boss_floors(nodes: Any) -> list[int]:
    """The floors the run's boss fights were on, in order, from its node string (`nodes`, one letter per room entered): floor 1 is the
    first room and has no letter, so the k-th letter is floor k + 2 and a 'B' at index i is a boss on floor i + 2. A boss the run never
    entered is not listed. Acts 1 to 3 have 17, 16 and 15 floors (bosses on 17, 33 and 48; the second Act 3 boss on 49), but one run in
    about four hundred that reaches Act 2 has an Act 2 of 18 floors, whose boss is on 35 and whose Act 3 boss is on 50: the string, not
    the cuts, says where each boss is. Only a complete string says so (`complete_nodes`)."""
    if not isinstance(nodes, str):
        return []
    return [i + 2 for i, ch in enumerate(nodes) if ch == "B"]


def complete_nodes(row: Mapping[str, Any]) -> bool:
    """Whether the row's node string covers every room of the run: the floor is one more than its length. True of every emulator arm row
    of generations 15 to 18 (29,698 of 29,698); not of the real game's rows, whose strings leave rooms out (the stream asks no map decision
    when the boss is the only node left), nor of a row without the field."""
    nodes = row.get("nodes")
    return isinstance(nodes, str) and int(row.get("floor") or 0) == len(nodes) + 1


def went_past(row: Mapping[str, Any], act: int) -> bool:
    """The one strict rule (2026-11-19): the run went past the first boss of act `act` (0 = Act 1, 1 = Act 2, 2 = the first Act 3 boss).
    A win went past all of them. A crashed or lost seed went past none. Otherwise the run's floor is above the boss's floor: the boss's
    floor is read from the row's own node string (`boss_floors`) when that string is complete (a boss never entered was not passed), else
    it is the cut 17 / 33 / 48, and then a fight the row records as lost (the act's first `boss_damage_acts` entry below 1.0, or None)
    is not passed whatever the floor says. The fraction of the boss's health taken is NOT read as a kill: it reaches 1.0 at the first Act 3
    boss without the boss dying; a row ending on floor 48 beat nothing there, a row on floor 49 or later did. A row ended by a deadline
    reads like any other: by its floor."""
    if row.get("win"):
        return True
    if row.get("crash") or row.get("lost"):
        return False
    floor = int(row.get("floor") or 0)
    if complete_nodes(row):
        floors = boss_floors(row["nodes"])
        return act < len(floors) and floor > floors[act]
    acts = row.get("boss_damage_acts")
    if isinstance(acts, list):
        for entry in acts:
            if int(entry[0]) == act:
                if entry[1] is None or entry[1] < 1.0:
                    return False
                break
    return floor > ACT_BOSS_FLOORS[act]


def bosses_beaten(row: Mapping[str, Any]) -> tuple[bool, bool, bool, bool]:
    """(B1, B2, B3a, win) of one run under the strict reading (`went_past`): a boss is beaten when the run went past it. A win beats all
    four. Since 2026-11-19: the judge used to read a boss as beaten when `boss_damage_acts` had a first entry of 1.0 for its act
    (`bosses_beaten_loose`), which is the same at Acts 1 and 2 (but for the odd long Act 2) and three times too high at the first Act 3
    boss, where two runs in three that took the whole of the boss's first form still die there."""
    if row.get("win"):
        return (True, True, True, True)
    return (went_past(row, 0), went_past(row, 1), went_past(row, 2), False)


def bosses_beaten_loose(row: Mapping[str, Any]) -> tuple[bool, bool, bool, bool]:
    """The reading before 2026-11-19, kept for comparison tables only (`tools/lab/ledger_strict.py`): (B1, B2, B3a, win) where the row
    has `boss_damage_acts` the first entry of each act at exactly 1.0, and the floor cuts 17 / 33 / 48 where it has not; a win beats all
    four. Nothing decides on it."""
    if row.get("win"):
        return (True, True, True, True)
    acts = row.get("boss_damage_acts")
    if isinstance(acts, list):
        first: dict[int, Any] = {}
        for entry in acts:
            first.setdefault(int(entry[0]), entry[1])
        return (first.get(0) == 1.0, first.get(1) == 1.0, first.get(2) == 1.0, False)
    floor = int(row.get("floor") or 0)
    return (floor > ACT_BOSS_FLOORS[0], floor > ACT_BOSS_FLOORS[1], floor > ACT_BOSS_FLOORS[2], False)


def deep(row: Mapping[str, Any], f: int) -> int:
    """deep_f of one run: how many of the bosses from `f` on (0 = B1 ... 3 = the win) it beats."""
    return sum(bosses_beaten(row)[f:])


def frontier_record(counts: Sequence[tuple[Sequence[int], int]], blocks: int = LINEAGE_BLOCKS, rate: float = FRONTIER_RATE) -> dict[str, Any]:
    """The frontier from the lineage's last `blocks` judged blocks, oldest first, each `(kills per boss [B1, B2, B3a, win], runs)`:
    the deepest boss whose pooled kill rate is at least `rate`, B1 (index 0) when none. The pooled counts are kept in the record."""
    recent = [([int(x) for x in k], int(n)) for k, n in list(counts)[-blocks:] if int(n) > 0]
    pooled_n = sum(n for _, n in recent)
    pooled = [sum(k[j] for k, _ in recent) for j in range(len(BOSSES))]
    rates = [pooled[j] / pooled_n if pooled_n else 0.0 for j in range(len(BOSSES))]
    index = max([0] + [j for j in range(len(BOSSES)) if rates[j] >= rate])
    return {"index": index, "name": BOSSES[index], "rate": rate, "blocks": len(recent), "pooled_n": pooled_n, "pooled_kills": pooled,
            "rates_per_100": [round(100 * x, 3) for x in rates], "block_counts": [[list(k), n] for k, n in recent]}


def frontier(counts: Sequence[tuple[Sequence[int], int]], blocks: int = LINEAGE_BLOCKS, rate: float = FRONTIER_RATE) -> int:
    """The frontier's index (0 = B1, 1 = B2, 2 = B3a, 3 = the win): the deepest boss the lineage beats in `rate` of its runs,
    pooled over its last `blocks` judged blocks (`frontier_record`); B1 when none or when there is no block yet."""
    return int(frontier_record(counts, blocks, rate)["index"])


def block_counts(rows: Sequence[Mapping[str, Any]]) -> tuple[list[int], int]:
    """(kills per boss [B1, B2, B3a, win], runs) of one arm's rows: the input of `frontier`."""
    kills = [0, 0, 0, 0]
    for r in rows:
        for j, beat in enumerate(bosses_beaten(r)):
            kills[j] += bool(beat)
    return kills, len(rows)


# ------------------------------------------------------------------ paired differences


def paired_rows(cand_rows: Sequence[Mapping[str, Any]], base_rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    """The seeds both arms played (sorted) and each arm's row by seed."""
    c = {r["seed"]: r for r in cand_rows}
    b = {r["seed"]: r for r in base_rows}
    return sorted(set(c) & set(b)), c, b


def paired_diffs(cand_rows: Sequence[Mapping[str, Any]], base_rows: Sequence[Mapping[str, Any]], f: int = 0) -> dict[str, Any]:
    """Per paired seed: the floor difference (candidate minus incumbent) and the deep_f difference in points per 100 runs, with the
    seeds in the order they are listed."""
    seeds, c, b = paired_rows(cand_rows, base_rows)
    return {"seeds": seeds,
            "floor": [float(c[s].get("floor") or 0) - float(b[s].get("floor") or 0) for s in seeds],
            "deep": [100.0 * (deep(c[s], f) - deep(b[s], f)) for s in seeds]}


# ------------------------------------------------------------------ margins (D68 addendum 6, decision 1): every figure with its plus or minus 2 SE


def arrived_b3a(row: Mapping[str, Any]) -> bool:
    """Whether the run arrived at the first Act 3 boss (its fight began, or it went past): a win did; a crashed or lost seed did not; a row with
    a complete node string arrived when the string has a third boss room; any other row when its floor reached the cut 48 (the first Act 3 boss's
    floor at the usual map lengths). The same reading `went_past(row, 2)` is made on: passes are arrivals that went past."""
    if row.get("win"):
        return True
    if row.get("crash") or row.get("lost"):
        return False
    if complete_nodes(row):
        return len(boss_floors(row["nodes"])) >= 3
    return int(row.get("floor") or 0) >= ACT_BOSS_FLOORS[2]


def _finite(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def paired_margin(cand: Sequence[float], base: Sequence[float], *, scale: float = 100.0, z: float = MARGIN_Z, counts: bool = True) -> dict[str, Any]:
    """One measure's margin from the arms' per-seed values (same order, one pair a seed): `cand` and `base` (the totals when `counts`, else the
    means), `diff` the paired mean difference (candidate minus incumbent) times `scale` (100: per 100 runs; 1 for floors, a run), `se` its standard
    error (SD of the paired differences with n - 1, over the root of n: the SE the funnel reads), `pm2se` z x se (the plus or minus the ledger prints),
    and `discordant` the pairs the candidate is ahead in and those the incumbent is. A figure that cannot be worked out (under two pairs) is None."""
    n = len(cand)
    diffs = [float(a) - float(b) for a, b in zip(cand, base, strict=True)]
    st = normal_stats([scale * d for d in diffs])
    se = st["se"]
    return {"cand": sum(cand) if counts else (math.fsum(cand) / n if n else None), "base": sum(base) if counts else (math.fsum(base) / n if n else None),
            "diff": _finite(st["mean"]), "se": _finite(se), "pm2se": _finite(z * se),
            "discordant": {"cand_ahead": sum(1 for d in diffs if d > 0), "base_ahead": sum(1 for d in diffs if d < 0)}}


def margins(cand_rows: Sequence[Mapping[str, Any]], base_rows: Sequence[Mapping[str, Any]], frontier: int = 0, z: float = MARGIN_Z) -> dict[str, Any]:
    """The margins of a judgement (`promotion.json` `keep_v2.margins`): over the seeds both arms played, for B1, B2, B3a (the boss beaten: a win
    beats all), wins, deep_f at `frontier`, floors and arrivals at the first Act 3 boss, the counts of each arm, the paired difference per 100
    runs (floors: floors a run), its SE, plus or minus `z` SE and the discordant pairs (`paired_margin`). Nothing here decides anything."""
    seeds, c, b = paired_rows(cand_rows, base_rows)
    beaten_c = {s: bosses_beaten(c[s]) for s in seeds}
    beaten_b = {s: bosses_beaten(b[s]) for s in seeds}
    out: dict[str, Any] = {"z": z, "pairs": len(seeds), "unit": "per 100 runs; floors a run"}
    for j, name in enumerate(("B1", "B2", "B3a", "wins")):
        out[name] = paired_margin([int(beaten_c[s][j]) for s in seeds], [int(beaten_b[s][j]) for s in seeds], z=z)
    out["deep_f"] = paired_margin([deep(c[s], frontier) for s in seeds], [deep(b[s], frontier) for s in seeds], z=z)
    out["floors"] = paired_margin([float(c[s].get("floor") or 0) for s in seeds], [float(b[s].get("floor") or 0) for s in seeds], scale=1.0, z=z,
                                  counts=False)
    out["arrivals"] = paired_margin([int(arrived_b3a(c[s])) for s in seeds], [int(arrived_b3a(b[s])) for s in seeds], z=z)
    return out


def _arrival_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arrived = [r for r in rows if arrived_b3a(r)]
    passes = sum(1 for r in arrived if went_past(r, 2))
    n = len(arrived)
    p = passes / n if n else None
    return {"arrivals": n, "passes": passes, "rate": None if p is None else 100.0 * p,
            "se": None if p is None else 100.0 * math.sqrt(p * (1.0 - p) / n)}


def per_arrival(cand_rows: Sequence[Mapping[str, Any]], base_rows: Sequence[Mapping[str, Any]], z: float = MARGIN_Z) -> dict[str, Any]:
    """Each arm's pass rate at the first Act 3 boss per 100 arrivals, with its binomial SE, over the seeds both arms played (`promotion.json`
    `per_arrival`): the arm's arrivals, the passes among them (the run went past the boss; a win did), the rate and its SE per 100 arrivals, and the
    candidate's rate minus the incumbent's with its SE (the two arms' SEs combined as independent: each arm's arrivals are its own runs) and plus or
    minus `z` SE. One arm with no arrival has no rate."""
    seeds, c, b = paired_rows(cand_rows, base_rows)
    ca, ba = _arrival_arm([c[s] for s in seeds]), _arrival_arm([b[s] for s in seeds])
    if ca["rate"] is None or ba["rate"] is None:
        diff = se = pm = None
    else:
        diff, se = ca["rate"] - ba["rate"], math.hypot(ca["se"], ba["se"])
        pm = z * se
    return {"z": z, "pairs": len(seeds), "cand": ca, "base": ba, "diff": diff, "se": se, "pm2se": pm, "unit": "per 100 arrivals"}


# ------------------------------------------------------------------ the bootstrap and the two routes


def _bootstrap_means(diffs: Sequence[float], b: int, seed: int):
    """The resampled means of `promotion_guard.cluster_bootstrap_lower` (whole seeds resampled; the same draws for the same inputs)."""
    import numpy as np  # noqa: PLC0415 -- the one part that is not standard library

    x = np.asarray(diffs, dtype=np.float64)
    n = x.size
    rng = np.random.default_rng(seed)
    means = np.empty(b)
    step = max(1, 4_000_000 // n)
    for s in range(0, b, step):
        e = min(b, s + step)
        means[s:e] = x[rng.integers(0, n, size=(e - s, n))].mean(axis=1)
    return x, means


def diff_summary(diffs: Sequence[float], b: int = BOOT_B, seed: int = BOOT_SEED, *, upper99: bool = False) -> dict[str, float]:
    """Seeds, mean, SE (SD with n - 1 over the root of n) and the cluster bootstrap's bounds of the mean of paired differences, all from
    one resampling: `lower_route` (one-sided 0.005 lower bound), `lower95` and `upper95` (one-sided 95%), and `p` (the share of
    resampled means at or below 0, as `promotion_guard.cluster_bootstrap_lower`). `lower_route` and `lower95` equal that function's
    `lower` at ROUTE_ALPHA and GUARD_ALPHA for the same inputs. With `upper99` (the keep rule v3's collapse check) the one-sided 99% upper
    bound of the same resampling is added as `upper99`; without it the summary is exactly what v2 always had."""
    import numpy as np  # noqa: PLC0415

    n = len(diffs)
    if n == 0:
        nan = float("nan")
        return {"n": 0, "mean": nan, "se": nan, "lower_route": nan, "lower95": nan, "upper95": nan, "p": 1.0, "boot": 0,
                **({"upper99": nan} if upper99 else {})}
    x, means = _bootstrap_means(diffs, b, seed)
    se = float(x.std(ddof=1)) / math.sqrt(n) if n > 1 else math.inf
    out = {"n": int(n), "mean": float(x.mean()), "se": se, "lower_route": float(np.quantile(means, ROUTE_ALPHA)),
           "lower95": float(np.quantile(means, GUARD_ALPHA)), "upper95": float(np.quantile(means, 1 - GUARD_ALPHA)),
           "p": float((np.sum(means <= 0) + 1) / (b + 1)), "boot": int(b)}
    if upper99:
        out["upper99"] = float(np.quantile(means, 1 - COLLAPSE_ALPHA))
    return out


def diff_stats(floor_diffs: Sequence[float], deep_diffs: Sequence[float], b: int = BOOT_B, seed: int = BOOT_SEED, *,
               floor_upper99: bool = False) -> dict[str, dict[str, float]]:
    """`diff_summary` of the floor and of the deep_f differences (two resamplings; both routes read them). `floor_upper99` adds the floor's
    99% upper bound (the keep rule v3's collapse check)."""
    return {"floor": diff_summary(floor_diffs, b, seed, upper99=floor_upper99), "deep": diff_summary(deep_diffs, b, seed)}


def route_a(floor_diffs: Sequence[float], deep_diffs: Sequence[float], *, boot: int = BOOT_B, seed: int = BOOT_SEED,
            stats: Mapping[str, Mapping[str, float]] | None = None, min_gain: float = MIN_GAIN) -> dict[str, Any]:
    """Route A, deeper: deep_f's bootstrap lower bound at ROUTE_ALPHA above 0, its mean at least `min_gain` (MIN_GAIN unless a caller says
    otherwise), and the floor guard (the floor's 95% lower bound at least -FLOOR_MARGIN). `stats` from `diff_stats` skips the resampling."""
    st = stats or diff_stats(floor_diffs, deep_diffs, boot, seed)
    fl, dp = st["floor"], st["deep"]
    checks = {"deep_bound": bool(dp["lower_route"] > 0), "min_gain": bool(dp["mean"] >= min_gain),
              "floor_margin": bool(fl["lower95"] >= -FLOOR_MARGIN)}
    return {"holds": all(checks.values()), "checks": checks, "alpha": ROUTE_ALPHA, "min_gain": min_gain, "floor_margin": FLOOR_MARGIN,
            "d_deep": dp["mean"], "deep_lower": dp["lower_route"], "p_deep": dp["p"], "floor_lower95": fl["lower95"]}


def route_f(floor_diffs: Sequence[float], deep_diffs: Sequence[float], *, boot: int = BOOT_B, seed: int = BOOT_SEED,
            stats: Mapping[str, Mapping[str, float]] | None = None) -> dict[str, Any]:
    """Route F, further: today's floor primary at ROUTE_ALPHA (the bootstrap lower bound above 0, half of D24's 0.01) and deep_f not
    worse (its one-sided 95% upper bound not below 0)."""
    st = stats or diff_stats(floor_diffs, deep_diffs, boot, seed)
    fl, dp = st["floor"], st["deep"]
    checks = {"floor_bound": bool(fl["lower_route"] > 0), "deep_not_worse": bool(dp["upper95"] >= 0)}
    return {"holds": all(checks.values()), "checks": checks, "alpha": ROUTE_ALPHA, "d_floor": fl["mean"], "floor_lower": fl["lower_route"],
            "p_floor": fl["p"], "deep_upper95": dp["upper95"]}


def route_a_v3(floor_diffs: Sequence[float], deep_diffs: Sequence[float], *, boot: int = BOOT_B, seed: int = BOOT_SEED,
               stats: Mapping[str, Mapping[str, float]] | None = None, min_gain: float = MIN_GAIN) -> dict[str, Any]:
    """Route A under the keep rule v3 (D66): deep_f's bootstrap lower bound at ROUTE_ALPHA above 0 and its mean at least `min_gain`, and nothing
    else: no floor margin. (`floor_lower95` is recorded, as the ledger's column.) `stats` from `diff_stats` skips the resampling."""
    st = stats or diff_stats(floor_diffs, deep_diffs, boot, seed)
    fl, dp = st["floor"], st["deep"]
    checks = {"deep_bound": bool(dp["lower_route"] > 0), "min_gain": bool(dp["mean"] >= min_gain)}
    return {"holds": all(checks.values()), "checks": checks, "alpha": ROUTE_ALPHA, "min_gain": min_gain, "floor_margin": None,
            "d_deep": dp["mean"], "deep_lower": dp["lower_route"], "p_deep": dp["p"], "floor_lower95": fl["lower95"]}


def floor_collapse(floor: Mapping[str, float]) -> dict[str, Any]:
    """The keep rule v3's floor check: floors catch breakage only. `floor` is `diff_summary(..., upper99=True)` of the floor differences;
    the player is collapsed when the one-sided 99% upper bound of the paired mean floor difference is below COLLAPSE_FLOORS (-5.0 floors).
    Anything above that is recorded and decides nothing."""
    up = float(floor["upper99"])
    return {"collapsed": bool(up < COLLAPSE_FLOORS), "upper99": up, "limit": COLLAPSE_FLOORS, "alpha": COLLAPSE_ALPHA, "mean": float(floor["mean"]),
            "n": int(floor["n"])}


# ------------------------------------------------------------------ route S (D68 addendum 6 outcome 4): S1, the arrival exam's reading, and S2


def _short(x: Any) -> str:
    return str(x)[:12] if x is not None else "none"


def s1_match(reading: Mapping[str, Any], expect: Mapping[str, Any] | None) -> dict[str, Any]:
    """Whether a reading names the pair it is applied to: its ruler's hash, its candidate (the sha256 of the fight network it played) and its
    champion must equal `expect`'s (`ruler_hash`, `candidate`, `champion`). A key the caller cannot name cannot be checked, and an unchecked
    reading is not applied. `reasons` says each failure in words; `expect` is kept in the record."""
    reasons: list[str] = []
    if expect is None:
        reasons.append("no pair identity was given: the reading cannot be checked against the pair")
    else:
        for key in S1_KEYS:
            want, got = expect.get(key), reading.get(key)
            if want is None:
                reasons.append(f"{key}: the pair's is not known, so the reading cannot be checked")
            elif got != want:
                reasons.append(f"{key} does not match (the reading names {_short(got)}, the pair {_short(want)})")
    return {"verified": not reasons, "reasons": reasons, "expect": {k: (expect or {}).get(k) for k in S1_KEYS}}


def s1_reading(result: Mapping[str, Any], expect: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """S1's reading from an arrival exam's results (`rl/act3_starts.py arrival-exam`, its `results.json`; `gym_gate.arrival_exam`'s figures) in the shape
    the rule reads: `mean` and `se` in passes per 100 arrivals (the per-start mean survival difference, candidate minus champion, unweighted over the
    starts, its SE over the starts), `starts`, `pairs`, `draws` (the draw numbers played), `test`, `complete` (every start has all its draws),
    the ruler's hash and the sha256 of the candidate's and of the champion's fight network the exam was played for, and `match`
    (`s1_match` against `expect`, the pair's own ruler hash and networks)."""
    fb = result.get("first_boss") or {}
    reading = {"rule": result.get("rule"), "config": result.get("config"), "engine": result.get("engine"), "mean": fb.get("points"),
               "se": fb.get("se_points"), "starts": fb.get("starts"), "pairs": fb.get("pairs"), "draws": list(result.get("draws") or ()),
               "draws_per_start": fb.get("draws_per_start"), "test": result.get("test"), "complete": bool(fb.get("complete")),
               "stopped": result.get("stopped"), "ruler_hash": result.get("ruler_hash"),
               "candidate": (result.get("candidate") or {}).get("policy_sha256"), "champion": (result.get("champion") or {}).get("policy_sha256"),
               "candidate_dir": (result.get("candidate") or {}).get("dir"), "champion_dir": (result.get("champion") or {}).get("dir")}
    reading["match"] = s1_match(reading, expect)
    return reading


def route_s1(reading: Mapping[str, Any] | None, *, fight_swap: bool = True, s1_min: float = S1_MIN, z: float = S1_Z) -> dict[str, Any] | None:
    """S1: the arrival exam's reading holds when its mean is at least `s1_min` passes per 100 arrivals and the mean minus `z` SE (2.576, the one-sided
    0.005 bound) is above 0. A reading that is not applied (`applied` false, the words in `reasons`) never holds: one that does not name the pair it
    is for (`s1_match`), a candidate that is not a fight-only swap (`fight_swap`), one that is incomplete or has no mean or SE. None for no reading."""
    if reading is None:
        return None
    reasons: list[str] = []
    if not fight_swap:
        reasons.append("the candidate is not a fight-only swap of the champion")
    match = reading.get("match")
    if not match:
        reasons.append("the reading was not checked against the pair (`s1_reading(result, expect)` does that)")
    elif not match.get("verified"):
        reasons += list(match.get("reasons") or ["the reading does not match the pair"])
    mean, se = reading.get("mean"), reading.get("se")
    known = all(isinstance(x, (int, float)) and math.isfinite(x) for x in (mean, se))
    if not known:
        reasons.append("the reading has no mean or no SE")
    if not reading.get("complete"):
        reasons.append("the reading is incomplete: not every start has all its draws")
    lower = float(mean) - z * float(se) if known else None
    checks = {"min_gain": bool(known and mean >= s1_min), "bound": bool(known and lower > 0)}
    applied = not reasons
    return {"applied": applied, "holds": bool(applied and all(checks.values())), "checks": checks, "mean": mean if known else None,
            "se": se if known else None, "pm2se": MARGIN_Z * float(se) if known else None, "lower": lower, "min": s1_min, "z": z,
            "starts": reading.get("starts"), "pairs": reading.get("pairs"), "draws": list(reading.get("draws") or ()),
            "test": reading.get("test"), "ruler_hash": reading.get("ruler_hash"), "candidate": reading.get("candidate"),
            "champion": reading.get("champion"), "reasons": reasons}


def route_s_parameters() -> dict[str, Any]:
    """The route's named parameters, as `promotion.json` records them under `keep_v2.route_s.parameters`."""
    return {"s1_min": S1_MIN, "s1_z": S1_Z, "s2_from_pairs": S2_FROM_PAIRS, "s2_look_alpha": S2_LOOK_ALPHA, "s2_final_alpha": S2_FINAL_ALPHA}


def state_shift(per_arr: Mapping[str, Any], s1: Mapping[str, Any]) -> dict[str, Any] | None:
    """The whole-run pass rate per 100 arrivals' difference (candidate minus champion, `per_arrival`) minus S1's, with the two SEs combined as
    independent: how far the whole run's arrivals and fights moved from what the fight brain does on the champion's own arrivals. None when either
    figure cannot be worked out."""
    a, sa, b, sb = per_arr.get("diff"), per_arr.get("se"), s1.get("mean"), s1.get("se")
    if any(x is None or not math.isfinite(x) for x in (a, sa, b, sb)):
        return None
    se = math.hypot(sa, sb)
    return {"whole_run_per_arrival": a, "s1": b, "diff": a - b, "se": se, "pm2se": MARGIN_Z * se, "unit": "per 100 arrivals"}


# ------------------------------------------------------------------ the guards


def _bad(row: Mapping[str, Any]) -> bool:
    return bool(row.get("crash") or row.get("lost"))


def crash_guard(cand_rows: Sequence[Mapping[str, Any]], base_rows: Sequence[Mapping[str, Any]], slack: tuple[int, float] = CRASH_SLACK) -> dict[str, Any]:
    """The crash guard on the seeds both arms played: the candidate's crashed or lost seeds (`crash`, `lost`) at most the incumbent's
    plus max(slack[0], slack[1] * seeds). A crash is a floor-0 loss in the floor, so without this guard a candidate could buy
    nothing by crashing but could hide a fault in a large block."""
    seeds, c, b = paired_rows(cand_rows, base_rows)
    cand_bad, base_bad = sum(_bad(c[s]) for s in seeds), sum(_bad(b[s]) for s in seeds)
    allowed = max(slack[0], slack[1] * len(seeds))
    return {"holds": bool(cand_bad <= base_bad + allowed), "cand": cand_bad, "base": base_bad, "slack": allowed, "n": len(seeds)}


def act_damage(row: Mapping[str, Any], act: int) -> float | None:
    """The damage dealt to the boss of act `act` (0-based) as a fraction of its HP: 1.0 for a win, the first entry of that act in
    `boss_damage_acts`, 0.0 when the act's boss was not reached; None when the row carries no `boss_damage_acts` (an older row)."""
    if row.get("win"):
        return 1.0
    acts = row.get("boss_damage_acts")
    if not isinstance(acts, list):
        return None
    for entry in acts:
        if int(entry[0]) == act:
            return None if entry[1] is None else float(entry[1])
    return 0.0


def act1_report(cand_rows: Sequence[Mapping[str, Any]], base_rows: Sequence[Mapping[str, Any]], *, alpha: float = GUARD_ALPHA,
                boot: int = BOOT_B, seed: int = BOOT_SEED) -> dict[str, Any]:
    """Today's Act 1 kill guard and damage guard on the arms, and the same damage guard on the Act 2 boss: report only. `fires` names
    those that would have fired; nothing here blocks (`blocks` is always False). The Act 1 counts are the ledger's (`boss_cand`,
    `boss_base`: a win or a floor past 17), so the drift check reads what it always read."""
    kills_c, kills_b, dmg_c, dmg_b = pg.from_rows(cand_rows, base_rows, kill=pg.boss_kill)
    kill = pg.kill_guard(kills_c, kills_b, alpha)
    damage = pg.damage_guard(dmg_c, dmg_b, alpha, boot, seed)
    seeds, c, b = paired_rows(cand_rows, base_rows)
    a2c = {s: act_damage(c[s], 1) for s in seeds}
    a2b = {s: act_damage(b[s], 1) for s in seeds}
    known = bool(seeds) and all(v is not None for v in (*a2c.values(), *a2b.values()))
    damage2 = pg.damage_guard({s: float(v) for s, v in a2c.items()} if known else None, {s: float(v) for s, v in a2b.items()} if known else None,
                              alpha, boot, seed)
    fires = [name for name, part in (("kill", kill), ("damage", damage), ("damage_act2", damage2)) if part.get("fires")]
    return {"kill": kill, "damage": damage, "damage_act2": damage2, "fires": fires, "blocks": False,
            "note": "reported only: v2 blocks nothing on the Act 1 kill, the Act 1 damage or the Act 2 damage"}


def wins_clause(cand_rows: Sequence[Mapping[str, Any]], base_rows: Sequence[Mapping[str, Any]], gate: float = WIN_GATE,
                alpha: float = WIN_ALPHA) -> dict[str, Any]:
    """The wins co-primary of D24, unchanged: active once both arms win at least `gate` of their runs, and then the exact McNemar on
    the discordant pairs must say the candidate wins more, at one-sided `alpha`. Inactive: holds."""
    seeds, c, b = paired_rows(cand_rows, base_rows)
    only_c = sum(1 for s in seeds if c[s].get("win") and not b[s].get("win"))
    only_b = sum(1 for s in seeds if b[s].get("win") and not c[s].get("win"))
    n = max(1, len(seeds))
    cw, bw = sum(bool(c[s].get("win")) for s in seeds) / n, sum(bool(b[s].get("win")) for s in seeds) / n
    p = pg.mcnemar_exact_one_sided(only_c, only_b)
    active = cw >= gate and bw >= gate
    return {"active": bool(active), "holds": bool(p < alpha and only_c > only_b) if active else True, "cand": cw, "base": bw, "b": only_c,
            "c": only_b, "p": p, "alpha": alpha}


# ------------------------------------------------------------------ the verdict on one block


def judge(cand_rows: Sequence[Mapping[str, Any]], base_rows: Sequence[Mapping[str, Any]], *, frontier: int = 0,
          frontier_info: Mapping[str, Any] | None = None, lineage: Sequence[tuple[int, int]] = (), previous_flagged: bool = False,
          prior_attempts: int = 0, cap: int = pg.ATTEMPT_CAP, boot: int = BOOT_B, seed: int = BOOT_SEED, rule: str = "v2",
          extended: bool = False, final: bool = True, min_gain: float = MIN_GAIN, s1: Mapping[str, Any] | None = None,
          fight_swap: bool = True) -> dict[str, Any]:
    """The keep test v2 on one block (`rule` "v3": the keep rule v3, `judge_v3`; `extended` and `final` are read by v3 alone, so are `s1` (route S's
    reading, `s1_reading`) and `fight_swap`; `min_gain` is route A's
    smallest deep_f gain per 100 runs, MIN_GAIN by default, and is recorded in the verdict): both routes, the wins clause, the crash guard and the drift check, with the Act 1 guards reported and
    the attempt cap recorded (D65 addendum 2: `checks["attempt"]` is what the old cap said, `attempt_advisory` is true, nothing refuses on
    it). `frontier` is the frontier's index (`frontier_record` for the lineage); `lineage` the promoted lineage's
    (Act 1 kills, runs) on its last blocks for the drift check, `previous_flagged` and `prior_attempts` as for
    `promotion_guard.judge`. `promote` is true when route A or F holds and every gating check holds; `route` reads "A", "F", "AF" or "";
    `blocks` is true when a check other than the routes and the advisory attempt fails (the promotion would be refused even on a route)."""
    if rule == "v3":
        return judge_v3(cand_rows, base_rows, frontier=frontier, frontier_info=frontier_info, lineage=lineage, previous_flagged=previous_flagged,
                        prior_attempts=prior_attempts, cap=cap, boot=boot, seed=seed, extended=extended, final=final, min_gain=min_gain,
                        s1=s1, fight_swap=fight_swap)
    if rule != "v2":
        raise ValueError(f"unknown keep rule {rule!r} (the rules of this module are {', '.join(RULE_VERSIONS)})")
    if s1 is not None:
        raise ValueError("route S is the keep rule v3's (rule='v3')")
    d = paired_diffs(cand_rows, base_rows, frontier)
    st = diff_stats(d["floor"], d["deep"], boot, seed)
    ra, rf = route_a(d["floor"], d["deep"], stats=st, min_gain=min_gain), route_f(d["floor"], d["deep"], stats=st)
    route = ("A" if ra["holds"] else "") + ("F" if rf["holds"] else "")
    wins = wins_clause(cand_rows, base_rows)
    crash = crash_guard(cand_rows, base_rows)
    a1 = act1_report(cand_rows, base_rows, boot=boot, seed=seed)
    drift = pg.drift_check(a1["kill"]["cand"], a1["kill"]["pairs"], lineage, previous_flagged)
    attempt = pg.retry_cap(prior_attempts, cap)
    seeds, c, b = paired_rows(cand_rows, base_rows)
    kills = {name: [sum(bosses_beaten(c[s])[j] for s in seeds), sum(bosses_beaten(b[s])[j] for s in seeds)] for j, name in enumerate(BOSSES)}
    checks = {"route": bool(route), "wins": bool(wins["holds"]), "crash": bool(crash["holds"]), "drift": not drift["blocks"],
              "attempt": bool(attempt["within_cap"])}
    blocking = [k for k in pg.failed_checks(checks) if k != "route"]
    return {"rule": RULE, "boss_reading": BOSS_READING, "frontier": {**dict(frontier_info or {}), "index": int(frontier), "name": BOSSES[int(frontier)]},
            "route": route, "routes": {"A": ra, "F": rf}, "floor": st["floor"], "deep": st["deep"], "kills": kills,
            "alpha": ROUTE_ALPHA, "floor_margin": FLOOR_MARGIN, "min_gain": min_gain, "wins": wins, "crash": crash, "drift": drift,
            "attempt": attempt, "act1_review": a1, "checks": checks, "promote": pg.checks_hold(checks), "blocks": bool(blocking),
            "blocking": blocking, "attempt_advisory": True, "review": drift["verdict"] == "flagged",
            "margins": margins(cand_rows, base_rows, frontier), "per_arrival": per_arrival(cand_rows, base_rows)}


def judge_v3(cand_rows: Sequence[Mapping[str, Any]], base_rows: Sequence[Mapping[str, Any]], *, frontier: int = 0,
             frontier_info: Mapping[str, Any] | None = None, lineage: Sequence[tuple[int, int]] = (), previous_flagged: bool = False,
             prior_attempts: int = 0, cap: int = pg.ATTEMPT_CAP, boot: int = BOOT_B, seed: int = BOOT_SEED, extended: bool = False,
             final: bool = True, min_gain: float = MIN_GAIN, s1: Mapping[str, Any] | None = None, fight_swap: bool = True) -> dict[str, Any]:
    """The keep rule v3 (D66) on the paired arms of one judgement (one block, or the pooled blocks of an extended candidate: the rows are
    all the pairs). Route A alone promotes: deep_f's gain per 100 runs has its one-sided 0.005 bound above 0 and a point gain of at least
    `min_gain` (+1.0 by default), with no floor margin; route F is gone. The floors decide nothing but a collapse: `checks["floor_collapse"]` fails when the paired
    mean floor difference's one-sided 99% upper bound is below -5.0 floors. The crash guard and the wins co-primary hold as in v2. The
    drift check and the attempt cap are recorded (`checks["drift"]`, `checks["attempt"]` keep what each would have said, `advisory_checks`
    names them) and decide nothing; `review` still flags a flagged drift.

    A candidate is *extended*, not refused, when `final` (every seed of the block is played; a block the funnel stopped is not) and not
    `extended` (it was extended before, or this judgement already pools a second block) and route A's point gain is at least `min_gain` while its
    0.005 bound is not above 0, and no guard fails (no collapse, no crash, no wins failure): `extend` is true, `route` reads "X", and
    `promote` is false. Anything else that does not promote is a refusal. `route` reads "A" when route A holds (also when a guard blocks the
    promotion, as in v2), "X" for an extension, "" otherwise.

    Route S (D68 addendum 6 outcome 4): with `s1` (a reading from `s1_reading`, checked against the pair) and `fight_swap` (the candidate is a fight-only
    swap), S1 is asked (`route_s1`). When it holds the decision is S's: the collapse, crash and wins guards as above, and S2's final bound, deep_f's
    one-sided 95% upper bound not below 0 (`S2_FINAL_ALPHA`); there is no extension. `route` reads "S" ("AS" when route A holds too) and the verdict
    carries `route_s` (the parameters, the S1 reading, the S2 bounds, the state shift). When S1 does not hold, or the reading is not applied (its reasons are
    recorded in `route_s`), v3 decides as it always did. With no `s1` the verdict is exactly v3's and has no `route_s`."""
    d = paired_diffs(cand_rows, base_rows, frontier)
    st = diff_stats(d["floor"], d["deep"], boot, seed, floor_upper99=True)
    ra = route_a_v3(d["floor"], d["deep"], stats=st, min_gain=min_gain)
    collapse = floor_collapse(st["floor"])
    wins = wins_clause(cand_rows, base_rows)
    crash = crash_guard(cand_rows, base_rows)
    a1 = act1_report(cand_rows, base_rows, boot=boot, seed=seed)
    drift = {**pg.drift_check(a1["kill"]["cand"], a1["kill"]["pairs"], lineage, previous_flagged), "advisory": True}
    attempt = pg.retry_cap(prior_attempts, cap)
    seeds, c, b = paired_rows(cand_rows, base_rows)
    kills = {name: [sum(bosses_beaten(c[s])[j] for s in seeds), sum(bosses_beaten(b[s])[j] for s in seeds)] for j, name in enumerate(BOSSES)}
    pa = per_arrival(cand_rows, base_rows)
    s1_rec = route_s1(s1, fight_swap=fight_swap)
    s_active = bool(s1_rec and s1_rec["holds"])  # S1 holds: S decides, and nothing is extended
    s2 = None
    if s_active:  # S2's final bound: deep_f's one-sided 95% upper bound (the bootstrap's, as route A's bounds) must not be below 0
        ds = diff_summary(d["deep"], boot, seed, upper99=True)
        s2 = {"pairs": int(ds["n"]), "deep_mean": ds["mean"], "deep_se": ds["se"], "upper95": ds["upper95"], "upper99": ds["upper99"],
              "final_alpha": S2_FINAL_ALPHA, "final_ok": bool(ds["upper95"] >= 0), "look_alpha": S2_LOOK_ALPHA, "from_pairs": S2_FROM_PAIRS,
              "note": "the funnel's looks refuse only (deep_f's 99% upper bound below 0 or a floor collapse, from 500 pairs on); the final look refuses when "
                      "the 95% upper bound is below 0 and otherwise promotes if the guards hold"}
    s_holds = bool(s2 and s2["final_ok"])
    checks = {"route": bool(ra["holds"] or s_holds), "floor_collapse": not collapse["collapsed"], "wins": bool(wins["holds"]), "crash": bool(crash["holds"]),
              "drift": not drift["blocks"], "attempt": bool(attempt["within_cap"])}
    gating = {k: v for k, v in checks.items() if k not in ADVISORY_V3}
    blocking = [k for k, v in gating.items() if not v and k != "route"]
    extend = bool(final and not extended and not s_active and not ra["holds"] and ra["checks"]["min_gain"] and not ra["checks"]["deep_bound"]
                  and not blocking)
    route = (ROUTE_AS if ra["holds"] else ROUTE_S) if s_holds else "A" if ra["holds"] else EXTEND_ROUTE if extend else ""
    out = {"rule": RULE_V3, "boss_reading": BOSS_READING, "frontier": {**dict(frontier_info or {}), "index": int(frontier), "name": BOSSES[int(frontier)]},
           "route": route, "routes": {"A": ra}, "floor": st["floor"], "deep": st["deep"], "kills": kills, "alpha": ROUTE_ALPHA,
           "min_gain": min_gain, "collapse": collapse, "wins": wins, "crash": crash, "drift": drift, "attempt": attempt, "act1_review": a1,
           "checks": checks, "promote": all(gating.values()), "extend": extend, "extended": bool(extended), "final": bool(final),
           "blocks": bool(blocking), "blocking": blocking, "attempt_advisory": True, "advisory_checks": list(ADVISORY_V3),
           "review": drift["verdict"] == "flagged", "margins": margins(cand_rows, base_rows, frontier), "per_arrival": pa}
    if s1_rec is not None:  # a verdict with no reading is exactly v3's
        why = ("S1 holds: S decides (S2's final bound, the guards; no extension)" if s_active
               else "S1 did not hold: v3 decides" if s1_rec["applied"] else "S1 was not applied: v3 decides (" + "; ".join(s1_rec["reasons"]) + ")")
        out["route_s"] = {"rule": RULE_S, "parameters": route_s_parameters(), "fight_swap": bool(fight_swap), "s1": s1_rec, "applies": s_active, "s2": s2,
                          "holds": s_holds, "state_shift": state_shift(pa, s1_rec) if s1_rec["applied"] else None, "decided_by": "S" if s_active else "v3",
                          "why": why}
    return out


def ledger_cells(verdict: Mapping[str, Any]) -> dict[str, Any]:
    """The nine promotions.tsv columns a v2 row adds after `macro`: rule, frontier, route (A, F, AF or empty; v3: A, X for an extension, S or AS
    when route S decided, or empty), k2_cand and k2_base (the Act 2 boss's kills, the headline, written whatever the verdict), d_deep and deep_lower
    (per 100 runs; the 0.005 bound), p_deep, floor_lower95. The columns are fixed: route S's figures (S1's reading with plus or minus 2 SE, S2's
    bound) are in the row's `guard` cell (`summary`) and in promotion.json (`keep_v2.route_s`)."""
    k2 = verdict["kills"]["B2"]
    return {"rule": LEDGER_RULE_V3 if verdict.get("rule") == RULE_V3 else LEDGER_RULE, "frontier": verdict["frontier"]["name"], "route": verdict["route"], "k2_cand": k2[0], "k2_base": k2[1],
            "d_deep": f"{verdict['deep']['mean']:.4f}", "deep_lower": f"{verdict['deep']['lower_route']:.4f}", "p_deep": f"{verdict['deep']['p']:.5f}",
            "floor_lower95": f"{verdict['floor']['lower95']:.4f}"}


def _pm(se: float | None, z: float = MARGIN_Z, places: int = 2) -> str:
    """` ±x` for a figure whose standard error is `se` (the margin is `z` SE); nothing when the SE cannot be worked out."""
    return f" ±{z * se:.{places}f}" if se is not None and math.isfinite(se) else ""


def margin_parts(verdict: Mapping[str, Any]) -> list[str]:
    """The ledger line's margin parts, in the order of the design (D68 addendum 6, decision 1): the Act 3 boss kills, the wins, the arrivals at the
    first Act 3 boss, the pass rate per 100 arrivals, then B1 and B2 -- each with its counts (candidate/incumbent), its paired difference per 100
    runs (per 100 arrivals for the pass rate) and its plus or minus 2 SE. A verdict without margins (one an older judge wrote) has none."""
    m, pa = verdict.get("margins"), verdict.get("per_arrival")
    parts: list[str] = []

    def one(label: str, key: str) -> None:
        x = m.get(key) if m else None
        if x and x.get("diff") is not None:
            parts.append(f"{label} {x['cand']}/{x['base']} {x['diff']:+.2f}{_pm(x.get('se'), (m or {}).get('z', MARGIN_Z))}")

    one("B3a", "B3a")
    one("wins", "wins")
    one("arrivals", "arrivals")
    if pa and pa.get("diff") is not None:
        parts.append(f"per arrival {pa['cand']['rate']:.1f}/{pa['base']['rate']:.1f} {pa['diff']:+.1f}{_pm(pa.get('se'), pa.get('z', MARGIN_Z), 1)}")
    one("B1", "B1")
    one("B2", "B2")
    return parts


def route_s_parts(verdict: Mapping[str, Any]) -> list[str]:
    """The ledger line's route S parts (none for a verdict with no S1 reading): S1's mean in passes per 100 arrivals with plus or minus 2 SE, its starts,
    draws and ruler, whether it holds (or why it was not applied), and, when S1 held, S2's final bound (deep_f's 95% upper bound, not below 0) and the
    99% upper bound the looks read."""
    rs = verdict.get("route_s")
    if not rs:
        return []
    s1, s2 = rs["s1"], rs.get("s2")
    if not s1["applied"]:
        return [f"S1 not applied ({'; '.join(s1['reasons'])})"]
    parts = [f"S1 {s1['mean']:+.2f}{_pm(s1.get('se'))} per 100 arrivals ({s1['starts']} starts, {len(s1['draws'])} draws, ruler {str(s1['ruler_hash'])[:8]}): "
             + ("holds" if s1["holds"] else "does not hold, v3 decides")]
    if s2:
        parts.append(f"S2 deep 95% upper {s2['upper95']:+.2f} (final bound 0: {'ok' if s2['final_ok'] else 'FAILS'}; 99% upper {s2['upper99']:+.2f})")
    return parts


def summary(verdict: Mapping[str, Any]) -> str:
    """One line for the `guard` cell of promotions.tsv (no tabs): the route, the frontier, deep_f and the floor guard, then the crash,
    wins, drift and attempt checks, then the Act 1 guards that would have fired (reported only), then the margins (`margin_parts`). Every figure
    carries its plus or minus 2 SE (since D68 addendum 6): deep_f and the floors in place, the rest in the margin parts."""
    dp, fl, k2 = verdict["deep"], verdict["floor"], verdict["kills"]["B2"]
    r, w, cr, d, a = verdict["route"] or "none", verdict["wins"], verdict["crash"], verdict["drift"], verdict["attempt"]
    mg = verdict.get("min_gain", MIN_GAIN)
    gain = f" min gain {mg:g}" if mg != MIN_GAIN else ""  # the note after the rule's name, only when the minimum is not the default
    margins_ = margin_parts(verdict)
    if verdict.get("rule") == RULE_V3:
        co = verdict["collapse"]
        parts = [f"v3{gain} route {r}" + (" (extend)" if verdict["extend"] else ""), f"frontier {verdict['frontier']['name']}",
                 f"deep {dp['mean']:+.2f}{_pm(dp.get('se'))} (lower {dp['lower_route']:+.2f}) K2 {k2[0]}/{k2[1]}", *route_s_parts(verdict),
                 f"floor {fl['mean']:+.2f}{_pm(fl.get('se'))} (99% upper {co['upper99']:+.2f}, collapse below {co['limit']:+.1f}: {'COLLAPSED' if co['collapsed'] else 'ok'})",
                 f"crash {'ok' if cr['holds'] else 'FAILS'} ({cr['cand']}/{cr['base']})",
                 "wins guard " + (("ok" if w["holds"] else "FAILS") if w["active"] else "off"),
                 f"drift {d['verdict']} (advisory)" + (f" ({d['cand_rate']:.4f} vs pooled {d['pooled_rate']:.4f})" if "cand_rate" in d else ""),
                 f"attempt {a['attempt']}/{a['cap']}{pg.attempt_note(a)}",
                 "act1 review: " + (", ".join(verdict["act1_review"]["fires"]) if verdict["act1_review"]["fires"] else "none"), *margins_]
        return " | ".join(parts).replace("\t", " ")
    parts = [f"v2{gain} route {r}", f"frontier {verdict['frontier']['name']}",
             f"deep {dp['mean']:+.2f}{_pm(dp.get('se'))} (lower {dp['lower_route']:+.2f}) K2 {k2[0]}/{k2[1]}",
             f"floor lower95 {fl['lower95']:+.2f} (mean {fl['mean']:+.2f}{_pm(fl.get('se'))})",
             f"crash {'ok' if cr['holds'] else 'FAILS'} ({cr['cand']}/{cr['base']})",
             "wins guard " + (("ok" if w["holds"] else "FAILS") if w["active"] else "off"),
             f"drift {d['verdict']}" + (f" ({d['cand_rate']:.4f} vs pooled {d['pooled_rate']:.4f})" if "cand_rate" in d else ""),
             f"attempt {a['attempt']}/{a['cap']}{pg.attempt_note(a)}",
             "act1 review: " + (", ".join(verdict["act1_review"]["fires"]) if verdict["act1_review"]["fires"] else "none"), *margins_]
    return " | ".join(parts).replace("\t", " ")


# ------------------------------------------------------------------ the funnel's call at one look


def early_z(n_looks: int, alpha_early: float = EARLY_ALPHA, routes: int = 2) -> float:
    """The z a route must reach at an interim look to promote early: the standard normal quantile at 1 - alpha_early / (routes x
    interim looks), 3.72 for the seven looks and two routes. Infinite when there is no interim promote look (under three looks)."""
    k = n_looks - 2
    return NormalDist().inv_cdf(1 - alpha_early / (routes * k)) if k > 0 else math.inf


def normal_stats(diffs: Sequence[float]) -> dict[str, float]:
    """Seeds, mean, standard error (SD with n - 1 over the root of n) and z = mean / se: `gen_loop.seq_stats`, over again."""
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean": math.nan, "se": math.inf, "z": 0.0}
    mean = math.fsum(diffs) / n
    if n < 2:
        return {"n": n, "mean": mean, "se": math.inf, "z": 0.0}
    var = math.fsum((x - mean) ** 2 for x in diffs) / (n - 1)
    se = math.sqrt(var) / math.sqrt(n)
    z = mean / se if se > 0 else (math.inf if mean > 0 else -math.inf if mean < 0 else 0.0)
    return {"n": n, "mean": mean, "se": se, "z": z}


def seq_call_v3(floor_diffs: Sequence[float], deep_diffs: Sequence[float], j: int, looks: Sequence[int], *,
                alpha_early: float = EARLY_ALPHA, min_gain: float = MIN_GAIN, route_s: bool = False) -> dict[str, Any]:
    """The funnel's call at look `j` of `looks` under the keep rule v3 (normal statistics, as `seq_call`). The smoke look (j = 0) refuses
    nothing on these figures (D64; `funnel` still refuses a crashing candidate there); a later look refuses when route A is dead (deep_f's
    one-sided 99% upper bound below `min_gain`, +1.0 per 100 by default: not even an extension is in reach) or the floors have collapsed (their 99% upper bound
    below -5.0); an interim look is eligible to promote when route A's z reaches the early z (`early_z`, 3.72 for seven looks) with a gain of at
    least +1.0. Nothing else about the floors is asked: no floor route, no floor margin. The last look is `final`.

    `route_s` (S2 of route S, after S1 held): the looks refuse only. From `S2_FROM_PAIRS` (500) pairs on a look refuses when deep_f's one-sided 99% upper
    bound (`S2_LOOK_ALPHA`) is below 0 or the floors have collapsed (as above); it is never eligible to promote early and never asks route A's gain, so
    nothing here stops a candidate that is not dead. The smoke look and the last look are as above; the last look's decision is `judge`'s (S2's final bound)."""
    fl, dp = normal_stats(floor_diffs), normal_stats(deep_diffs)
    n_looks = len(looks)
    nd = NormalDist()
    z95, z99 = nd.inv_cdf(1 - GUARD_ALPHA), nd.inv_cdf(1 - COLLAPSE_ALPHA)
    z_e = early_z(n_looks, alpha_early)
    m, s, d, t = fl["mean"], fl["se"], dp["mean"], dp["se"]
    out: dict[str, Any] = {**fl, "look": j + 1, "of": n_looks, "upper99": m + z99 * s, "lower99": m - z99 * s, "lower95": m - z95 * s,
                           "upper95": m + z95 * s, "deep_mean": d, "deep_se": t, "deep_z": dp["z"], "deep_upper95": d + z95 * t,
                           "deep_upper99": d + z99 * t, "z_early": z_e, "routes": []}
    if j >= n_looks - 1:
        return {**out, "action": "final", "reason": "every seed played: the keep rule v3 on all of them"}
    if j == 0:  # D64: the smoke look refuses nothing on the floor or on deep_f; the figures are recorded
        return {**out, "action": "continue", "smoke_would_refuse": False,
                "reason": "smoke block: refuses nothing on these figures (D64; v3 asks the floors for a collapse and deep_f for route A at the later looks)"}
    if route_s:  # S2: the looks refuse only, from 500 pairs on
        out["route"] = ROUTE_S
        if dp["n"] < S2_FROM_PAIRS:
            return {**out, "action": "continue", "reason": f"route S: the looks refuse from {S2_FROM_PAIRS} pairs on ({dp['n']} played)"}
        harm = d + nd.inv_cdf(1 - S2_LOOK_ALPHA) * t < 0
        collapsed = out["upper99"] < COLLAPSE_FLOORS
        if harm or collapsed:
            why = []
            if harm:
                why.append(f"deep_f is down (route S: its 99% upper bound {d + nd.inv_cdf(1 - S2_LOOK_ALPHA) * t:+.2f} is below 0)")
            if collapsed:
                why.append(f"the floors collapsed (99% upper bound {out['upper99']:+.2f} below {COLLAPSE_FLOORS:+.1f})")
            return {**out, "action": "refuse", "reason": " and ".join(why)}
        return {**out, "action": "continue", "reason": "route S: the looks refuse only, and neither a harm nor a collapse is shown"}
    collapsed = out["upper99"] < COLLAPSE_FLOORS
    a_dead = out["deep_upper99"] < min_gain
    if collapsed or a_dead:
        why = []
        if collapsed:
            why.append(f"the floors collapsed (99% upper bound {out['upper99']:+.2f} below {COLLAPSE_FLOORS:+.1f})")
        if a_dead:
            why.append(f"route A is dead (deep_f's 99% upper bound {out['deep_upper99']:+.2f} below {min_gain:+.1f})")
        return {**out, "action": "refuse", "reason": " and ".join(why)}
    if dp["z"] >= z_e and d >= min_gain:
        return {**out, "action": "eligible", "routes": ["A"],
                "reason": f"route A clears the early z {z_e:.2f} (deep_f z {dp['z']:.2f}, gain {d:+.2f} per 100)"}
    return {**out, "action": "continue", "reason": "route A is not dead and is not eligible"}


def seq_call(floor_diffs: Sequence[float], deep_diffs: Sequence[float], j: int, looks: Sequence[int], *,
             alpha_early: float = EARLY_ALPHA, rule: str = "v2", min_gain: float = MIN_GAIN, route_s: bool = False) -> dict[str, Any]:
    """The funnel's call at look `j` (0-based) of `looks` on the seeds played, normal statistics throughout: `action` is "refuse",
    "eligible" (a route clears the early z and its other conditions; `routes` names them; the caller still asks `judge`), "continue"
    or "final" (the last look: `judge` on all the seeds). With m, s, z the floor differences' mean, SE and z, and d, t, u the deep_f
    differences' mean, SE and z:

      smoke look (j = 0)  never refuses or promotes on these figures (D64); `smoke_would_refuse` records the old rule: m < -1.0 and d is not
                          above 0, or m + 1.645 s < -1.0 (no route can pass). The funnel (`gen_loop.funnel`) still refuses a crashing candidate here.
      later looks         refuse when route F is dead (m + 2.326 s < 0) and route A is dead (m + 2.326 s < -1.0, or d + 1.645 t < 1.0).
      interim looks       eligible when route F's z reaches z_e with d + 1.645 t >= 0, or route A's u reaches z_e with d >= 1.0 and
                          m - 1.645 s >= -1.0; z_e = `early_z` (3.72 for seven looks).

    `rule` "v3" is `seq_call_v3`. `min_gain` is the 1.0 above (route A's smallest gain, MIN_GAIN by default). `route_s` (v3 only) is S2's looks
    (`seq_call_v3(route_s=True)`): they refuse only.
    """
    if rule == "v3":
        return seq_call_v3(floor_diffs, deep_diffs, j, looks, alpha_early=alpha_early, min_gain=min_gain, route_s=route_s)
    if rule != "v2":
        raise ValueError(f"unknown keep rule {rule!r} (the rules of this module are {', '.join(RULE_VERSIONS)})")
    if route_s:
        raise ValueError("route S is the keep rule v3's (rule='v3')")
    fl, dp = normal_stats(floor_diffs), normal_stats(deep_diffs)
    n_looks = len(looks)
    nd = NormalDist()
    z95, z99 = nd.inv_cdf(1 - GUARD_ALPHA), nd.inv_cdf(1 - REFUSE_ALPHA)
    z_e = early_z(n_looks, alpha_early)
    m, s, d, t = fl["mean"], fl["se"], dp["mean"], dp["se"]
    out: dict[str, Any] = {**fl, "look": j + 1, "of": n_looks, "upper99": m + z99 * s, "lower99": m - z99 * s, "lower95": m - z95 * s,
                           "upper95": m + z95 * s, "deep_mean": d, "deep_se": t, "deep_z": dp["z"], "deep_upper95": d + z95 * t,
                           "z_early": z_e, "routes": []}
    if j >= n_looks - 1:
        return {**out, "action": "final", "reason": "every seed played: the keep test on all of them"}
    if j == 0:  # D64: the smoke look refuses nothing on the floor; it records what the old rule would have said
        would = bool((m < SMOKE_MEAN and not d > 0) or m + z95 * s < -FLOOR_MARGIN)
        return {**out, "action": "continue", "smoke_would_refuse": would,
                "reason": "smoke block: the floor refuses nothing (D64); its figures are recorded"
                          + (f"; the old rule would have refused at floor mean {m:+.3f} with deep_f mean {d:+.2f} per 100" if would else "")}
    f_dead = out["upper99"] < 0
    a_dead = out["upper99"] < -FLOOR_MARGIN or out["deep_upper95"] < min_gain
    if f_dead and a_dead:
        return {**out, "action": "refuse", "reason": f"route F is dead (floor 99% upper bound {out['upper99']:+.3f} below 0) and route A is dead "
                                                      f"(that bound below {-FLOOR_MARGIN:+.1f} or deep_f's 95% upper bound {out['deep_upper95']:+.2f} "
                                                      f"below {min_gain:+.1f})"}
    eligible = []
    if fl["z"] >= z_e and out["deep_upper95"] >= 0:
        eligible.append("F")
    if dp["z"] >= z_e and d >= min_gain and out["lower95"] >= -FLOOR_MARGIN:
        eligible.append("A")
    if eligible:
        return {**out, "action": "eligible", "routes": eligible,
                "reason": f"route {''.join(eligible)} clears the early z {z_e:.2f} (floor z {fl['z']:.2f}, deep_f z {dp['z']:.2f})"}
    return {**out, "action": "continue", "reason": "neither route is dead and none is eligible"}


# ------------------------------------------------------------------ the futility stop (D55 addendum 18, 2026-11-09)

FUTILITY_MIN_SEEDS = 500  # the futility stop never looks before this many paired seeds (gen_loop.FUTILITY_MIN_SEEDS is the same figure)


def critical_gain(se: float, n: int, n_block: int, alpha: float) -> float:
    """The gain (mean paired difference) the final test needs: the one-sided `alpha` lower bound clears 0 on all `n_block` seeds when the
    mean is at least z(1 - alpha) x SE of the whole block, the block's SE being the SE of the `n` seeds played (`se`, SD over root n)
    scaled to `n_block` seeds. This is the smallest gain any promotion on that measure can have, and the least the futility stop asks
    of a route: the per-seed spread is the one seen so far, the seeds still to come are all counted."""
    return NormalDist().inv_cdf(1 - alpha) * se * math.sqrt(n / n_block)


def futility_routes(floor: Mapping[str, float], deep: Mapping[str, float] | None, n: int, n_block: int, *, floor_alpha: float = ROUTE_ALPHA,
                    routes: Sequence[str] = ("A", "F"), measure: str = "deep_f", extend_possible: bool = False,
                    min_gain: float = MIN_GAIN) -> dict[str, Any]:
    """Whether any route can still promote, from the summaries (`diff_summary`: mean, se, upper95) of the floor and deep_f differences
    on `n` of `n_block` paired seeds. For each route the 95% upper bound of the candidate's gain is set against the gain the route
    needs to promote on all the seeds:

      route A (deeper)   deep_f's upper95 against max(`min_gain`, the critical gain of deep_f at ROUTE_ALPHA): the rule asks a point gain of
                         `min_gain` (+1.0 per 100 by default) and a lower bound above 0. With `extend_possible` (the keep rule v3, a block that
                         may still be extended: this candidate was not extended before and no second block is pooled) the need is `min_gain` alone,
                         the gain that earns an extension: a block that cannot promote on its own is not futile while it can extend
      route F (further)  the floor's upper95 against the critical gain of the floor at `floor_alpha` (ROUTE_ALPHA for the keep test v2;
                         D24's 0.01 for v1, which has this route alone)

    A route is alive when its upper95 reaches the gain it needs (and is above 0: no route promotes on a gain that is not positive).
    `futile` is true when every route asked for is dead; a bound that cannot be worked out (under two seeds) keeps its route alive.
    The check reads the gains only. The guards, the wins co-primary and the rest are further conditions of a promotion, so keeping a
    route alive here never promotes anything: it only declines to stop. The caller asks nothing before FUTILITY_MIN_SEEDS."""
    out: dict[str, Any] = {"n": int(n), "block": int(n_block), "routes": {}}
    for name in routes:
        if name == "A":
            if deep is None:
                continue
            need = min_gain if extend_possible else max(min_gain, critical_gain(deep["se"], n, n_block, ROUTE_ALPHA))
            got, mean, label = deep["upper95"], deep["mean"], f"{measure} per 100 runs"
        else:
            need, got, mean, label = critical_gain(floor["se"], n, n_block, floor_alpha), floor["upper95"], floor["mean"], "floors"
        known = math.isfinite(need) and math.isfinite(got)
        out["routes"][name] = {"measure": label, "mean": float(mean), "upper95": float(got), "needs": float(need),
                               "alive": True if not known else bool(got > 0 and got >= need)}
    out["futile"] = bool(out["routes"]) and not any(r["alive"] for r in out["routes"].values())
    out["text"] = "; ".join(f"route {k} ({r['measure']}): 95% upper bound {r['upper95']:+.2f} {'reaches' if r['alive'] else 'falls short of'} "
                            f"the {r['needs']:+.2f} it needs" for k, r in out["routes"].items())
    return out


def futility_call(floor_diffs: Sequence[float], deep_diffs: Sequence[float] | None, n_block: int, *, floor_alpha: float = ROUTE_ALPHA,
                  routes: Sequence[str] = ("A", "F"), measure: str = "deep_f", boot: int = BOOT_B, seed: int = BOOT_SEED,
                  extend_possible: bool = False, min_gain: float = MIN_GAIN) -> dict[str, Any]:
    """`futility_routes` on the paired differences of the seeds played: the bounds are the cluster bootstrap's (`diff_summary`, the
    method of the routes' own lower bounds). `deep_diffs` None: the floor route alone (the v1 rule)."""
    fl = diff_summary(floor_diffs, boot, seed)
    dp = diff_summary(deep_diffs, boot, seed) if deep_diffs is not None and "A" in routes else None
    return futility_routes(fl, dp, len(floor_diffs), n_block, floor_alpha=floor_alpha, routes=routes, measure=measure, extend_possible=extend_possible,
                           min_gain=min_gain)
