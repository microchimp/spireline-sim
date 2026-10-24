"""The promotion guard the independent review of 2026-10-24 proposed (its section 4.3; docs/PROMOTION_GUARD.md).

Pure functions over the paired per-seed outcomes of a candidate and the incumbent on one E-block. Nothing here reads a
file or plays a run. `rl/gen_loop.py` calls `judge` only when the review-guard flag is set (``SPIRELINE_REVIEW_GUARD``,
or ``--review-guard`` on ``gen0.py promote``; off by default), and the D24 decision path is untouched otherwise.

The four parts, each a dict with a verdict and the numbers it used:

  kill_guard     the Act-1 boss-kill not-worse guard: the exact one-sided McNemar test on the discordant pairs at
                 one-sided 0.05 (D24 ran it at 0.01). Fires when the incumbent-only kills are significantly more than
                 the candidate-only kills.
  damage_guard   the boss-damage paired guard: damage dealt to the Act-1 boss as a fraction of its HP (0 when the boss
                 was not reached), paired by seed, the cluster bootstrap of the floor primary, not-worse at one-sided
                 0.05. Fires when even the upper bound of candidate minus incumbent is below 0. Needs ``boss_damage``
                 in every row of both arms; without it the guard reports itself unavailable and never fires.
  drift_check    the candidate's kill rate on its block against the promoted lineage's pooled rate over its last three
                 blocks (unpaired; Wilson 95% intervals): flagged for the weekly review when the candidate's upper
                 bound is below the pooled lower bound; blocked when that happens twice in a row.
  retry_cap      at most two promotion attempts per generation.

`judge` composes them. ``blocks`` is true when any part says the promotion must not go through: the kill or the damage
guard fires, the drift check blocks, or the attempt is over the cap. A flagged drift alone does not block: it marks the
promotion for the weekly review (``review``).

The power arithmetic of the review's appendix A is here too (`guard_power`, `coprimary_power`, `seeds_for_power`), so
the tables in docs/PROMOTION_GUARD.md can be re-run: ``python -m tools.lab.promotion_guard --table``.
"""

from __future__ import annotations

import functools
import math
import os
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any

FLAG = "SPIRELINE_REVIEW_GUARD"  # the environment variable (and the name of the config switch); off unless set to on
ON_VALUES = ("1", "on", "true", "yes")
RULE = "review-2026-10-24-4.3"  # the review's section that proposed this guard
GUARD_ALPHA = 0.05  # the guards' one-sided level (the review: a guard should lean toward firing)
ATTEMPT_CAP = 2  # promotion attempts per generation
LINEAGE_BLOCKS = 3  # blocks of the promoted lineage pooled by the drift check
DAMAGE_FIELD = "boss_damage"  # the runs.jsonl field the damage guard needs: damage to the Act-1 boss / its HP, 0 if not reached
ACT1_LAST_FLOOR = 17  # as rl/gen_loop.py: a run past floor 17 killed the Act 1 boss
Z95 = 1.959964
BOOT_B = 20000  # as rl/gen_loop.py BOOT_B
BOOT_SEED = 20261023  # as rl/gen_loop.py cluster_bootstrap_lower


def enabled(value: str | None = None, env: Mapping[str, str] | None = None) -> bool:
    """Whether the flag is on: `value`, or the environment variable, is one of ON_VALUES (case does not matter)."""
    v = value if value is not None else (os.environ if env is None else env).get(FLAG, "")
    return str(v).strip().lower() in ON_VALUES


# ------------------------------------------------------------------ the arithmetic


def mcnemar_exact_one_sided(b: int, c: int) -> float:
    """P(X >= b), X ~ Binomial(b + c, 1/2): the exact one-sided McNemar test, as rl/gen_loop.py. No discordant pair: 1."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / 2**n


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval for k successes in n trials (as rl/gen0.py)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (0.0 if k == 0 else max(0.0, c - h), 1.0 if k == n else min(1.0, c + h))


def cluster_bootstrap_lower(diffs: Sequence[float], alpha: float = GUARD_ALPHA, b: int = BOOT_B, seed: int = BOOT_SEED) -> dict[str, float]:
    """One-sided (1 - alpha) lower bound of the mean of paired differences, resampling whole seeds: the floor primary's
    bootstrap (rl/gen_loop.cluster_bootstrap_lower), the same draws for the same inputs."""
    import numpy as np  # noqa: PLC0415 -- the one part that is not standard library

    x = np.asarray(diffs, dtype=np.float64)
    n = x.size
    if n == 0:
        return {"n": 0, "mean": float("nan"), "lower": float("nan"), "p": 1.0, "boot": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(b)
    step = max(1, 4_000_000 // n)
    for s in range(0, b, step):
        e = min(b, s + step)
        means[s:e] = x[rng.integers(0, n, size=(e - s, n))].mean(axis=1)
    return {"n": int(n), "mean": float(x.mean()), "lower": float(np.quantile(means, alpha)),
            "p": float((np.sum(means <= 0) + 1) / (b + 1)), "boot": int(b)}


# ------------------------------------------------------------------ the four parts


def boss_kill(row: Mapping[str, Any]) -> bool:
    """Act-1 boss killed: the run won, or ended past the Act 1 boss floor (as rl/gen_loop.boss_kill)."""
    return bool(row.get("win")) or int(row.get("floor") or 0) > ACT1_LAST_FLOOR


def kill_guard(cand: Mapping[str, bool], base: Mapping[str, bool], alpha: float = GUARD_ALPHA) -> dict[str, Any]:
    """The not-worse boss-kill guard on the seeds both arms played. `b` = candidate-only kills, `c` = incumbent-only kills;
    `p_worse` = P(X >= c), X ~ Binomial(b + c, 1/2). Fires when p_worse < alpha and c > b."""
    seeds = sorted(set(cand) & set(base))
    b = sum(1 for s in seeds if cand[s] and not base[s])
    c = sum(1 for s in seeds if base[s] and not cand[s])
    p_worse = mcnemar_exact_one_sided(c, b)
    return {"fires": bool(p_worse < alpha and c > b), "alpha": alpha, "pairs": len(seeds), "cand": sum(bool(cand[s]) for s in seeds),
            "base": sum(bool(base[s]) for s in seeds), "b": b, "c": c, "discordant": b + c, "p_worse": p_worse}


def damage_guard(cand: Mapping[str, float] | None, base: Mapping[str, float] | None, alpha: float = GUARD_ALPHA,
                 boot: int = BOOT_B, seed: int = BOOT_SEED,
                 lower: Callable[..., dict[str, float]] | None = None) -> dict[str, Any]:
    """The boss-damage paired guard. `cand` and `base` map seed to the damage dealt to the Act-1 boss as a fraction of
    its HP (0 when not reached); None when the arms' rows lack the field, and the guard is then unavailable and does not
    fire. `diff_upper` is the one-sided (1 - alpha) upper bound of candidate minus incumbent from the cluster bootstrap
    (`lower`, default the floor primary's, on incumbent minus candidate); the guard fires when it is below 0."""
    if cand is None or base is None:
        return {"fires": False, "available": False, "why": f"no {DAMAGE_FIELD} field in the rows", "alpha": alpha, "n": 0}
    seeds = sorted(set(cand) & set(base))
    if not seeds:
        return {"fires": False, "available": False, "why": "no paired seeds", "alpha": alpha, "n": 0}
    worse = [float(base[s]) - float(cand[s]) for s in seeds]  # positive: the candidate dealt less
    r = (lower or cluster_bootstrap_lower)(worse, alpha, boot, seed)
    cand_mean = sum(float(cand[s]) for s in seeds) / len(seeds)
    base_mean = sum(float(base[s]) for s in seeds) / len(seeds)
    return {"fires": bool(r["lower"] > 0), "available": True, "alpha": alpha, "n": len(seeds), "cand_mean": cand_mean,
            "base_mean": base_mean, "diff_mean": cand_mean - base_mean, "diff_upper": -float(r["lower"]), "p_worse": float(r["p"]),
            "boot": int(r["boot"])}


def drift_check(cand_kills: int, cand_n: int, lineage: Sequence[tuple[int, int]], previous_flagged: bool = False,
                z: float = Z95, blocks: int = LINEAGE_BLOCKS) -> dict[str, Any]:
    """The candidate's kill rate on its block against the pooled rate of the lineage's last `blocks` blocks, each a
    (kills, runs) pair, oldest first. Flagged when the candidate's upper bound is below the pooled lower bound; blocked
    when the previous candidate was flagged too. Skipped (clear) while the lineage has no block."""
    recent = [(int(k), int(n)) for k, n in list(lineage)[-blocks:] if int(n) > 0]
    pooled_k, pooled_n = sum(k for k, _ in recent), sum(n for _, n in recent)
    out: dict[str, Any] = {"cand_kills": int(cand_kills), "cand_n": int(cand_n), "lineage": [list(x) for x in recent],
                           "pooled_kills": pooled_k, "pooled_n": pooled_n, "previous_flagged": bool(previous_flagged), "z": z}
    if pooled_n == 0 or cand_n <= 0:
        return {**out, "verdict": "skipped", "flagged": False, "blocks": False, "why": "no lineage block to compare with"}
    cand_lo, cand_hi = wilson(cand_kills, cand_n, z)
    pool_lo, pool_hi = wilson(pooled_k, pooled_n, z)
    flagged = cand_hi < pool_lo
    verdict = "blocked" if flagged and previous_flagged else "flagged" if flagged else "clear"
    return {**out, "verdict": verdict, "flagged": flagged, "blocks": verdict == "blocked", "cand_rate": cand_kills / cand_n,
            "cand_lower": cand_lo, "cand_upper": cand_hi, "pooled_rate": pooled_k / pooled_n, "pooled_lower": pool_lo,
            "pooled_upper": pool_hi}


def retry_cap(prior_attempts: int, cap: int = ATTEMPT_CAP) -> dict[str, Any]:
    """Attempt number `prior_attempts` + 1 of a generation, and whether the cap allows it."""
    attempt = int(prior_attempts) + 1
    return {"attempt": attempt, "cap": int(cap), "allowed": attempt <= cap, "blocks": attempt > cap}


# ------------------------------------------------------------------ the composed verdict


def from_rows(cand_rows: Sequence[Mapping[str, Any]], base_rows: Sequence[Mapping[str, Any]],
              kill: Callable[[Mapping[str, Any]], bool] = boss_kill, damage_field: str = DAMAGE_FIELD
              ) -> tuple[dict[str, bool], dict[str, bool], dict[str, float] | None, dict[str, float] | None]:
    """(candidate kills, incumbent kills, candidate damage, incumbent damage) by seed from two arms' runs.jsonl rows,
    on the seeds both played. The damage maps are None when any paired row of either arm lacks `damage_field`."""
    c = {r["seed"]: r for r in cand_rows}
    b = {r["seed"]: r for r in base_rows}
    seeds = sorted(set(c) & set(b))
    kills_c = {s: bool(kill(c[s])) for s in seeds}
    kills_b = {s: bool(kill(b[s])) for s in seeds}
    if seeds and all(c[s].get(damage_field) is not None and b[s].get(damage_field) is not None for s in seeds):
        return kills_c, kills_b, {s: float(c[s][damage_field]) for s in seeds}, {s: float(b[s][damage_field]) for s in seeds}
    return kills_c, kills_b, None, None


def judge(cand_kills: Mapping[str, bool], base_kills: Mapping[str, bool], cand_damage: Mapping[str, float] | None = None,
          base_damage: Mapping[str, float] | None = None, *, lineage: Sequence[tuple[int, int]] = (),
          previous_flagged: bool = False, prior_attempts: int = 0, alpha: float = GUARD_ALPHA, cap: int = ATTEMPT_CAP,
          boot: int = BOOT_B, seed: int = BOOT_SEED) -> dict[str, Any]:
    """The review's guard on one block: the four parts and whether, together, they block the promotion."""
    kill = kill_guard(cand_kills, base_kills, alpha)
    damage = damage_guard(cand_damage, base_damage, alpha, boot, seed)
    drift = drift_check(kill["cand"], kill["pairs"], lineage, previous_flagged)
    attempt = retry_cap(prior_attempts, cap)
    blocking = [name for name, part in (("kill", kill), ("damage", damage), ("drift", drift), ("attempt", attempt))
                if part.get("fires") or part.get("blocks")]
    return {"rule": RULE, "alpha": alpha, "kill": kill, "damage": damage, "drift": drift, "attempt": attempt,
            "blocks": bool(blocking), "blocking": blocking, "review": drift["verdict"] == "flagged"}


def summary(verdict: Mapping[str, Any]) -> str:
    """One line for promotions.tsv (no tabs): each part's verdict and its key number."""
    k, d, r, a = verdict["kill"], verdict["damage"], verdict["drift"], verdict["attempt"]
    kill = f"kill {'FIRES' if k['fires'] else 'ok'} (c {k['c']} b {k['b']} p {k['p_worse']:.3f})"
    dmg = (f"damage {'FIRES' if d['fires'] else 'ok'} (diff {d['diff_mean']:+.4f} upper {d['diff_upper']:+.4f})" if d.get("available")
           else "damage n/a")
    drift = f"drift {r['verdict']}" + (f" ({r['cand_rate']:.4f} vs pooled {r['pooled_rate']:.4f})" if "cand_rate" in r else "")
    return " | ".join([kill, dmg, drift, f"attempt {a['attempt']}/{a['cap']}"]).replace("\t", " ")


# ------------------------------------------------------------------ power (the review's appendix A)


def _log_binom_pmf(n: int, k: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


def binom_pmf(n: int, k: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    lp = _log_binom_pmf(n, k, p)
    return 0.0 if lp == -math.inf else math.exp(lp)


def binom_sf(n: int, k: int, p: float) -> float:
    """P(X >= k), X ~ Binomial(n, p): the pmf at k, then the terms above it by the ratio of consecutive terms."""
    if k <= 0:
        return 1.0
    if k > n or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    term = binom_pmf(n, k, p)
    total = term
    ratio = p / (1.0 - p)
    for j in range(k, n):
        term *= (n - j) / (j + 1) * ratio
        total += term
        if term < 1e-18 * total:
            break
    return min(1.0, total)


@functools.lru_cache(maxsize=None)
def critical_count(m: int, alpha: float) -> int:
    """The smallest k with P(X >= k | m, 1/2) <= alpha: the discordant pairs one way that make the test fire. m + 1 when no k does."""
    tail = 0.0
    term = 0.5**m  # P(X = m)
    for k in range(m, -1, -1):
        tail += term
        if tail > alpha:
            return k + 1
        term *= k / (m - k + 1)  # P(X = k - 1) from P(X = k)
    return 0


def guard_power(n: int, p_inc: float, p_cand: float, alpha: float = GUARD_ALPHA) -> float:
    """P(the kill guard fires) with n paired seeds, the incumbent killing at `p_inc` and the candidate at `p_cand`, the two
    arms' kills independent on a seed (appendix A: discordance d = p1 + p2 - 2 p1 p2, m ~ Binomial(n, d), the incumbent-only
    count ~ Binomial(m, (p1 - p1 p2) / d), fires at the critical count; the mixture over m). Real arms are positively
    correlated, which lowers discordance and so the true power."""
    p1, p2 = float(p_inc), float(p_cand)
    d = p1 + p2 - 2 * p1 * p2
    if d <= 0.0:
        return 0.0
    q = (p1 - p1 * p2) / d
    power = 0.0
    for m in range(n + 1):
        pm = binom_pmf(n, m, d)
        if pm < 1e-15:
            if m > n * d:
                break
            continue
        k = critical_count(m, alpha)
        power += pm * binom_sf(m, k, q)
    return min(1.0, power)


def coprimary_power(n: int, p_inc: float, p_cand: float, alpha: float = 0.01) -> float:
    """P(a real improvement passes the co-primary) = the guard's power with the arms swapped (the candidate ahead)."""
    return guard_power(n, p_cand, p_inc, alpha)


def seeds_for_power(p_inc: float, p_cand: float, power: float = 0.8, alpha: float = 0.01) -> int:
    """Paired seeds for `power` against the one-sided test at `alpha`: the normal approximation on the discordant pairs
    (appendix A). Independent arms; the direction tested is the one the rates have."""
    p1, p2 = float(p_inc), float(p_cand)
    d = p1 + p2 - 2 * p1 * p2
    if d <= 0.0 or p1 == p2:
        return 0
    q = (max(p1, p2) - p1 * p2) / d  # the share of discordant pairs that go the way the rates differ
    nd = statistics.NormalDist()
    z_a, z_b = nd.inv_cdf(1 - alpha), nd.inv_cdf(power)
    m = ((z_a * 0.5 + z_b * math.sqrt(q * (1 - q))) / (q - 0.5)) ** 2
    return math.ceil(m / d)


# ------------------------------------------------------------------ the tables of docs/PROMOTION_GUARD.md

GUARD_CASES = ((0.02, 0.015), (0.02, 0.013), (0.02, 0.01), (0.02, 0.005), (0.01, 0.005), (0.05, 0.04), (0.05, 0.025), (0.10, 0.08), (0.10, 0.05))
COPRIMARY_CASES = ((0.0135, 0.02), (0.02, 0.025), (0.02, 0.03), (0.02, 0.04), (0.05, 0.075), (0.10, 0.13), (0.10, 0.15))


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _rate(p: float) -> str:
    return f"{100 * p:g}%"


def table() -> str:
    """The three tables, as Markdown."""
    lines = ["Guard mode: the probability that a candidate whose kill rate fell is stopped (independent arms).", "",
             "| incumbent -> candidate | 500 seeds, 0.01 | 500 seeds, 0.05 | 2,000 seeds, 0.01 | 2,000 seeds, 0.05 | seeds for 80% power, 0.01 | seeds for 80% power, 0.05 |",
             "|---|---|---|---|---|---|---|"]
    for p1, p2 in GUARD_CASES:
        lines.append(f"| {_rate(p1)} -> {_rate(p2)} | {_pct(guard_power(500, p1, p2, 0.01))} | {_pct(guard_power(500, p1, p2, 0.05))} | "
                     f"{_pct(guard_power(2000, p1, p2, 0.01))} | {_pct(guard_power(2000, p1, p2, 0.05))} | "
                     f"{seeds_for_power(p1, p2, 0.8, 0.01):,} | {seeds_for_power(p1, p2, 0.8, 0.05):,} |")
    lines += ["", "Co-primary mode: the probability that a real improvement passes (one-sided 0.01).", "",
              "| incumbent -> candidate | 500 seeds | 2,000 seeds | 4,000 seeds | seeds for 80% power |", "|---|---|---|---|---|"]
    for p1, p2 in COPRIMARY_CASES:
        lines.append(f"| {_rate(p1)} -> {_rate(p2)} | {_pct(coprimary_power(500, p1, p2))} | {_pct(coprimary_power(2000, p1, p2))} | "
                     f"{_pct(coprimary_power(4000, p1, p2))} | {seeds_for_power(p1, p2, 0.8, 0.01):,} |")
    lines += ["", "False fires of the kill guard with equal arms (the exact test is conservative; the nominal level is 0.05).", "",
              "| kill rate in both arms | 500 seeds | 2,000 seeds |", "|---|---|---|"]
    for p in (0.01, 0.02, 0.05, 0.10):
        lines.append(f"| {_rate(p)} | {100 * guard_power(500, p, p, 0.05):.1f}% | {100 * guard_power(2000, p, p, 0.05):.1f}% |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import sys  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", action="store_true", help="print the power tables of docs/PROMOTION_GUARD.md")
    a = ap.parse_args(argv)
    if a.table:
        sys.stdout.write(table())
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
