#!/usr/bin/env python3
"""Verdicts corrected for seed difficulty: the retrospective (D60 item 4, lane K). Read-only; nothing here plays a game.

    python -I tools/verdict_covariate_retro.py [--agent-root D:/spireline/referee/agent] [--rule v2] [--json PATH]

The question: can a judgement reach its verdict on fewer paired runs by cancelling seed difficulty (CUPED: regress the paired
difference on a covariate known before the judgement, here the mean of the *other* stored arms' outcomes on the same seed)?
For every judged block of the ledger (`promotions.tsv`, rows of the keep rule `--rule`) this prints

  * the premise audit: how many of the block's seeds any stored arm other than the judged pair has played (arms stored before the
    judgement, `stored_before`). The covariate is the mean of those arms' outcomes on the seed, so it exists only where this is
    above 0;
  * the plain paired figures at every look of the funnel and the covariate-adjusted ones (`cuped_adjust`; where the covariate is
    missing it falls back to the plain estimate, and `looks saved` is 0 by construction);
  * the funnel's replay on the stored arms (`keep_v2.seq_call` on the floor and deep_f differences at the ledger's frontier) for the
    plain and the adjusted differences: the look at which each would stop;
  * the ceiling any difficulty covariate could reach in a paired design (`ceiling_from_moments`): the share of the paired
    difference's variance that a perfectly measured, one-dimensional seed difficulty would explain, from the two arms' own variances
    and covariance, for the floor, deep_f and the Act 2 boss kill.

The estimator is here, tested, and not wired into the keep test: it lands only if the retrospective shows a gain (the lane's rule).
Aggregates only: no seed is printed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_AGENT_ROOT = Path(os.environ.get("SPIRELINE_REFEREE", "D:/spireline/referee")) / "agent"
MIN_SEEDS = 30  # `cuped_adjust` needs this many seeds with a covariate
MIN_ABS_T = 2.0  # ... and the covariate's slope on the differences at least this many standard errors from 0 (else the plain estimate)
METRICS = ("floor", "deep", "b2")


# ------------------------------------------------------------------ the estimator


def cuped_adjust(d: Sequence[float], x: Sequence[float] | None, mu_x: float | None, *, min_abs_t: float = MIN_ABS_T) -> dict[str, Any]:
    """The covariate-adjusted paired estimate (CUPED) of the mean of the paired differences `d`, against the covariate `x` of the same
    seeds, whose mean over the seeds' population is `mu_x` (known before the judgement, from other seeds' stored arms).

    The adjusted per-seed values are d_i - theta (x_i - mu_x) with theta the least-squares slope of d on x; their mean is
    mean(d) - theta (mean(x) - mu_x), unbiased when mu_x is the population mean (x is the same number for both arms, so it only
    corrects a block whose seeds are easier or harder than the population). Standard error: the residual SD (n - 2 degrees of
    freedom: theta is fitted) over the root of n, so the figure is never optimistic about the fitted slope. The adjusted
    differences are returned in `diffs` (the funnel reads them as it reads plain ones).

    Falls back to the plain estimate (`used` False, `diffs` = d, `se_ratio` 1.0, `reason` says why) when there is no covariate or
    it is missing on any seed, fewer than `MIN_SEEDS` seeds, the covariate is constant, or its slope is below `min_abs_t` standard
    errors (a covariate that explains nothing is not fitted on noise)."""
    dd = np.asarray(d, dtype=np.float64)
    n = int(dd.size)
    mean = float(dd.mean()) if n else math.nan
    se = float(dd.std(ddof=1)) / math.sqrt(n) if n > 1 else math.inf
    base = {"n": n, "plain_mean": mean, "plain_se": se}

    def fallback(reason: str, **extra: Any) -> dict[str, Any]:
        return {**base, "used": False, "reason": reason, "diffs": dd, "theta": 0.0, "r": extra.get("r", math.nan), "mean": mean, "se": se,
                "se_ratio": 1.0, "var_ratio": 1.0}

    if x is None or mu_x is None or not math.isfinite(float(mu_x)):
        return fallback("no covariate")
    xx = np.asarray(x, dtype=np.float64)
    if xx.shape != dd.shape or not np.all(np.isfinite(xx)):
        return fallback("covariate missing on some seeds")
    if n < MIN_SEEDS:
        return fallback(f"under {MIN_SEEDS} seeds")
    xc, dc = xx - xx.mean(), dd - dd.mean()
    sxx, sdd = float(xc @ xc), float(dc @ dc)
    if sxx <= 1e-12 * max(1.0, float(np.abs(xx).max()) ** 2 * n) or sdd <= 0.0:
        return fallback("covariate or differences constant")
    sxd = float(xc @ dc)
    r = sxd / math.sqrt(sxx * sdd)
    t = abs(r) * math.sqrt((n - 2) / max(1e-300, 1.0 - r * r)) if abs(r) < 1 else math.inf
    if t < min_abs_t:
        return fallback(f"covariate explains nothing (|t| {t:.2f} < {min_abs_t})", r=r)
    theta = sxd / sxx
    adj = dd - theta * (xx - float(mu_x))
    rss = float(((dc - theta * xc) ** 2).sum())
    se_adj = math.sqrt(rss / (n - 2)) / math.sqrt(n)
    ratio = se_adj / se if se > 0 else 1.0
    return {**base, "used": True, "reason": "adjusted", "diffs": adj, "theta": theta, "r": r, "mean": float(adj.mean()), "se": se_adj,
            "se_ratio": ratio, "var_ratio": ratio * ratio}


# ------------------------------------------------------------------ the ceiling of any difficulty covariate in a paired design


def ceiling_from_moments(var_c: float, var_b: float, cov: float, mean_c: float | None = None, mean_b: float | None = None) -> dict[str, float]:
    """What the best seed-difficulty covariate could take out of the paired difference's variance, from the two arms' per-seed
    variances and covariance across seeds (and, for the second model, their means).

    Additive model: c = p u + e_c and b = q u + e_b with u the seed's difficulty (unit variance, shared by the arms) and the arms' own
    noise e of one common SD. The three moments fix p, q and the noise (var_c - var_b = p^2 - q^2, cov = p q). The paired difference
    c - b = (p - q) u + (e_c - e_b): pairing already removes the shared difficulty, and an oracle that measured u exactly would take
    out only (p - q)^2, the part of the difference that depends on how the two arms respond to difficulty differently. `share` is
    (p - q)^2 over the difference's variance. For a count of rare events (a boss kill) the arms' noise is not of one size (it grows
    with the rate), and this model reads that as a difference in response, so it overstates.

    Proportional model (with the means): difficulty scales the arms' expected outcomes, E[c | seed] = mean_c f(seed) and
    E[b | seed] = mean_b f(seed), E f = 1, the arms' noise independent given the seed. Then cov = mean_c mean_b Var f and the
    difference's conditional mean varies with the seed by (mean_c - mean_b)^2 Var f: `share_prop` is that over the difference's
    variance, (mean_c - mean_b)^2 cov / (mean_c mean_b var_d).

    `share_max` is the larger of the two: the most any covariate (a noisy measure of the difficulty does less) can cut the variance by,
    whichever model holds. No shared difficulty (cov <= 0) gives 0."""
    vd = var_c + var_b - 2.0 * cov
    out = {"share": 0.0, "share_prop": 0.0, "share_max": 0.0, "p": 0.0, "q": 0.0, "noise_sd": math.sqrt(max(var_c, var_b, 0.0)),
           "rho": 0.0 if cov <= 0 else math.nan, "var_d": max(vd, 0.0)}
    if cov <= 0.0 or vd <= 0.0 or var_c <= 0.0 or var_b <= 0.0:
        return out
    a = var_c - var_b
    root = math.sqrt(a * a + 4.0 * cov * cov)
    p, q = math.sqrt((a + root) / 2.0), math.sqrt((root - a) / 2.0)
    share = min(1.0, (p - q) ** 2 / vd)
    prop = 0.0
    if mean_c is not None and mean_b is not None and mean_c > 0.0 and mean_b > 0.0:
        prop = min(1.0, (mean_c - mean_b) ** 2 * cov / (mean_c * mean_b * vd))
    return {**out, "share": share, "share_prop": prop, "share_max": max(share, prop), "p": p, "q": q,
            "noise_sd": math.sqrt(max(0.0, var_c - p * p)), "rho": cov / math.sqrt(var_c * var_b)}


def pooled_share(parts: Sequence[Mapping[str, float]], key: str = "share_max") -> float:
    """The share of the paired difference's variance a covariate could remove over several blocks: the blocks' removable variances
    (`key` times `var_d`) over their variances, each weighted by its seeds (`n`)."""
    num = sum(float(p[key]) * float(p["var_d"]) * (float(p["n"]) - 1) for p in parts)
    den = sum(float(p["var_d"]) * (float(p["n"]) - 1) for p in parts)
    return num / den if den > 0 else 0.0


def moments(c: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, int]:
    """(variance of c, variance of b, their covariance, n) over paired seeds, n - 1 in the denominators."""
    cc, bb = np.asarray(c, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = int(cc.size)
    if n < 2:
        return math.nan, math.nan, math.nan, n
    cov = float(((cc - cc.mean()) * (bb - bb.mean())).sum() / (n - 1))
    return float(cc.var(ddof=1)), float(bb.var(ddof=1)), cov, n


def pooled_moments(parts: Sequence[tuple[float, float, float, int]]) -> tuple[float, float, float, int]:
    """The moments of several blocks pooled about each block's own means (degrees of freedom summed)."""
    good = [p for p in parts if p[3] >= 2 and all(math.isfinite(v) for v in p[:3])]
    df = sum(p[3] - 1 for p in good)
    if df <= 0:
        return math.nan, math.nan, math.nan, 0
    return (sum(p[0] * (p[3] - 1) for p in good) / df, sum(p[1] * (p[3] - 1) for p in good) / df,
            sum(p[2] * (p[3] - 1) for p in good) / df, df + len(good))


# ------------------------------------------------------------------ the ledger, the arms, the other arms on a seed


def read_ledger(tsv: Path) -> list[dict[str, str]]:
    """The rows of promotions.tsv by column name."""
    lines = [x for x in Path(tsv).read_text(encoding="utf-8").splitlines() if x.strip()]
    head = lines[0].split("\t")
    return [dict(zip(head, x.split("\t"), strict=False)) for x in lines[1:]]


def judged_blocks(rows: Sequence[Mapping[str, str]], rule: str = "v2") -> list[dict[str, str]]:
    """One ledger row per judged block of the keep rule `rule` ("v2": the rows with `rule` v2; "all": every row), the last row of a block
    winning (a block re-judged under a newer rule keeps the newer row), oldest first."""
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        if rule == "all" or (r.get("rule") or "") == rule:
            out[r["seed_block"]] = dict(r)
    return sorted(out.values(), key=lambda r: r["written"])


def arm_dirs(agent: Path, row: Mapping[str, str], config: str = "a3-v1") -> tuple[Path | None, Path | None]:
    """(the candidate's arm folder, the incumbent's) of a ledger row: the candidate's under `gen<cand_gen>/<cand_dir>/arms`, the
    incumbent's the one other folder of that block under the base generation (`gen<base_gen>/*/arms`, `gen0/arms`). None when absent."""
    block = f"{row['seed_block']}-{config}"
    cand = Path(agent) / f"gen{row['cand_gen']}" / (row.get("cand_dir") or "candidate") / "arms" / block
    if row["base_gen"] == "0":
        bases = [Path(agent) / "gen0" / "arms" / block]
    else:
        bases = sorted((Path(agent) / f"gen{row['base_gen']}").glob(f"candidate*/arms/{block}"))
    base = next((b for b in bases if (b / "runs.jsonl").is_file() and b.resolve() != cand.resolve()), None)
    return (cand if (cand / "runs.jsonl").is_file() else None, base)


def read_arm(arm_dir: Path) -> dict[str, dict[str, Any]]:
    """An arm's rows by seed."""
    out: dict[str, dict[str, Any]] = {}
    for x in (Path(arm_dir) / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            out[r["seed"]] = r
    return out


def outcome(row: Mapping[str, Any], frontier: int) -> dict[str, float]:
    """One run's three figures: the floor, deep_f at `frontier` (bosses beaten from it on, 0 to 3) and the Act 2 boss kill (0 or 1)."""
    from tools.lab import keep_v2 as k2  # noqa: PLC0415

    beaten = k2.bosses_beaten(row)
    return {"floor": float(row.get("floor") or 0), "deep": float(sum(beaten[frontier:])), "b2": float(beaten[1])}


def stored_before(path: Path, when: str) -> bool:
    """Whether the file was last written before the UTC stamp `when` (`2026-11-13T22:00:33Z`): an arm after the judgement began is not
    a covariate of it."""
    t = datetime.fromtimestamp(Path(path).stat().st_mtime, tz=UTC)
    return t < datetime.strptime(when, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def other_arms_index(agent: Path, exclude: Sequence[Path | None], when: str | None, frontier: int) -> dict[str, list[dict[str, float]]]:
    """Every stored arm's outcomes by seed over the agent folder's `runs.jsonl` files, except the judged pair (`exclude`: arm folders)
    and, with `when`, any file written after that stamp (no leakage from later arms): seed -> one outcome dict per arm that played it."""
    skip = {Path(p).resolve() for p in exclude if p is not None}
    index: dict[str, list[dict[str, float]]] = {}
    for f in sorted(Path(agent).rglob("runs.jsonl")):
        if f.parent.resolve() in skip or (when is not None and not stored_before(f, when)):
            continue
        try:
            rows = [json.loads(x) for x in f.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
        except (OSError, ValueError):
            continue
        for r in rows:
            if isinstance(r, dict) and r.get("seed") is not None:
                index.setdefault(str(r["seed"]), []).append(outcome(r, frontier))
    return index


def difficulty(seeds: Sequence[str], index: Mapping[str, Sequence[Mapping[str, float]]], metric: str) -> tuple[np.ndarray, float]:
    """The covariate of each seed (the mean of the other arms' `metric` on it; NaN when no arm has played it) and the population mean
    (the mean over every seed of `index` outside `seeds`; NaN when none)."""
    mine = set(seeds)
    x = np.array([float(np.mean([o[metric] for o in index[s]])) if index.get(s) else math.nan for s in seeds])
    pop = [float(np.mean([o[metric] for o in v])) for s, v in index.items() if s not in mine and v]
    return x, (float(np.mean(pop)) if pop else math.nan)


# ------------------------------------------------------------------ one block


def paired_outcomes(cand: Mapping[str, Mapping[str, Any]], base: Mapping[str, Mapping[str, Any]], frontier: int) -> dict[str, Any]:
    """The seeds both arms played, in the block's own order (`ix`, the seed's place in the block's file), with each arm's outcomes."""
    seeds = sorted(set(cand) & set(base), key=lambda s: (int(cand[s].get("ix", 0)), s))
    out: dict[str, Any] = {"seeds": seeds, "ix": [int(cand[s].get("ix", i)) for i, s in enumerate(seeds)]}
    for arm, rows in (("c", cand), ("b", base)):
        figures = [outcome(rows[s], frontier) for s in seeds]
        for m in METRICS:
            out[f"{m}_{arm}"] = np.array([f[m] for f in figures])
    return out


def looks_of(n_block: int) -> list[int]:
    """The funnel's looks of a block of `n_block` seeds (`gen_loop.seq_looks`: 200, 500, 800, ...)."""
    import gen_loop  # noqa: PLC0415

    return gen_loop.seq_looks(int(n_block))


def replay(floor_d: Sequence[float], deep_d: Sequence[float], looks: Sequence[int], stored: int) -> dict[str, Any]:
    """The funnel's plain statistics at each look reached by the `stored` paired seeds (`keep_v2.seq_call` on the first n differences)
    and the first look whose action is not "continue" ("refuse" or "eligible"), None when none is reached."""
    from tools.lab import keep_v2 as k2  # noqa: PLC0415

    table, stop = [], None
    for j, n in enumerate(looks):
        if n > stored:
            break
        call = k2.seq_call(list(floor_d[:n]), list(deep_d[:n]), j, looks)
        table.append({"n": n, "floor_z": call["z"], "deep_z": call["deep_z"], "floor_mean": call["mean"], "floor_se": call["se"], "action": call["action"]})
        if stop is None and call["action"] in ("refuse", "eligible"):
            stop = {"n": n, "look": j + 1, "action": call["action"]}
    return {"looks": table, "stop": stop}


def actual_stop(row: Mapping[str, str]) -> str:
    """The funnel's recorded stop in a ledger row's guard text: `promote at 1400`, `refuse at 500`, `futility at 2600` or `final`."""
    import re  # noqa: PLC0415

    g = row.get("guard") or ""
    m = re.search(r"funnel (promote|refuse) at (\d+) of (\d+) seeds", g) or re.search(r"(futility)@(\d+) of (\d+) seeds", g)
    return f"{m.group(1)} at {m.group(2)}" if m else "final"


def block_report(agent: Path, row: Mapping[str, str]) -> dict[str, Any] | None:
    """The retrospective of one judged block, None when an arm is not stored."""
    cand_dir, base_dir = arm_dirs(agent, row)
    if cand_dir is None or base_dir is None:
        return None
    from tools.lab import keep_v2 as k2  # noqa: PLC0415

    f = k2.BOSSES.index(row["frontier"]) if row.get("frontier") in k2.BOSSES else 0
    pair = paired_outcomes(read_arm(cand_dir), read_arm(base_dir), f)
    n = len(pair["seeds"])
    index = other_arms_index(agent, (cand_dir, base_dir), row["written"], f)
    covered = {m: int(sum(1 for s in pair["seeds"] if index.get(s))) for m in METRICS}
    n_block = 2000
    import re  # noqa: PLC0415

    mm = re.search(r"of (\d+) seeds", row.get("guard") or "")
    if mm:
        n_block = int(mm.group(1))
    looks = looks_of(max(n_block, n))
    d = {"floor": pair["floor_c"] - pair["floor_b"], "deep": 100.0 * (pair["deep_c"] - pair["deep_b"]), "b2": 100.0 * (pair["b2_c"] - pair["b2_b"])}
    adjusted: dict[str, dict[str, Any]] = {}
    for m in METRICS:
        x, mu = difficulty(pair["seeds"], index, m)
        adjusted[m] = cuped_adjust(d[m], x, mu)
    plain = replay(d["floor"], d["deep"], looks, n)
    adj = replay(adjusted["floor"]["diffs"], adjusted["deep"]["diffs"], looks, n)
    mom = {m: moments(pair[f"{m}_c"], pair[f"{m}_b"]) for m in METRICS}
    ceilings = {m: {**ceiling_from_moments(*mom[m][:3], float(np.mean(pair[f"{m}_c"])), float(np.mean(pair[f"{m}_b"]))), "n": mom[m][3]} for m in METRICS}
    return {"block": row["seed_block"], "written": row["written"], "cand_gen": row["cand_gen"], "base_gen": row["base_gen"], "decision": row["decision"],
            "frontier": row.get("frontier", ""), "route": row.get("route", ""), "n": n, "n_block": n_block, "actual_stop": actual_stop(row),
            "covered": covered, "other_arms_on_judged_seeds": covered["floor"] > 0,
            "adjusted": {m: {k: v for k, v in a.items() if k != "diffs"} for m, a in adjusted.items()},
            "plain_replay": plain, "adjusted_replay": adj,
            "looks_saved": (plain["stop"]["look"] - adj["stop"]["look"]) if plain["stop"] and adj["stop"] else 0,
            "moments": {m: {"var_c": v[0], "var_b": v[1], "cov": v[2], "n": v[3]} for m, v in mom.items()},
            "ceiling": ceilings}


def audit_all_blocks(agent: Path, rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """The premise audit over every ledger row's block: seeds judged and seeds any other stored arm has also played."""
    seeds_total = seeds_covered = blocks = 0
    seen: set[str] = set()
    for row in rows:
        if row["seed_block"] in seen:
            continue
        seen.add(row["seed_block"])
        cand_dir, base_dir = arm_dirs(agent, row)
        if cand_dir is None or base_dir is None:
            continue
        pair = sorted(set(read_arm(cand_dir)) & set(read_arm(base_dir)))
        index = other_arms_index(agent, (cand_dir, base_dir), None, 0)
        blocks += 1
        seeds_total += len(pair)
        seeds_covered += sum(1 for s in pair if index.get(s))
    return {"blocks": blocks, "seeds": seeds_total, "seeds_with_another_arm": seeds_covered}


# ------------------------------------------------------------------ the report


def render(reports: Sequence[Mapping[str, Any]], audit: Mapping[str, Any]) -> str:
    """The retrospective as markdown tables (aggregates only)."""
    out = [f"Premise audit: {audit['blocks']} judged blocks, {audit['seeds']} judged seeds, {audit['seeds_with_another_arm']} of them played by any "
           "stored arm other than the judged pair.", ""]
    out += ["| block | n | frontier | recorded stop | plain replay stop | adjusted replay stop | covered | adjusted variance ratio floor / deep / B2 | looks saved |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        ps, ad = r["plain_replay"]["stop"], r["adjusted_replay"]["stop"]
        ratios = " / ".join(f"{r['adjusted'][m]['var_ratio']:.3f}" for m in METRICS)
        out.append(f"| {r['block']} | {r['n']} | {r['frontier'] or '-'} | {r['actual_stop']} | {ps['action'] + ' at ' + str(ps['n']) if ps else 'none'} | "
                   f"{ad['action'] + ' at ' + str(ad['n']) if ad else 'none'} | {r['covered']['floor']} of {r['n']} | {ratios} | {r['looks_saved']} |")
    out += ["", "Ceiling of any seed-difficulty covariate on the paired difference's variance (`ceiling_from_moments`; each cell is the additive model / the "
                "proportional model; the additive one overstates for rare kills), and the arms' seed correlation rho:", "",
            "| block | rho floor | rho deep_f | rho B2 | ceiling floor | ceiling deep_f | ceiling B2 |", "|---|---|---|---|---|---|---|"]

    def cell(c: Mapping[str, float]) -> str:
        return f"{100 * c['share']:.1f}% / {100 * c['share_prop']:.1f}%"

    for r in reports:
        c = r["ceiling"]
        out.append(f"| {r['block']} | " + " | ".join(f"{c[m]['rho']:.3f}" for m in METRICS) + " | " + " | ".join(cell(c[m]) for m in METRICS) + " |")
    parts = {m: [r["ceiling"][m] for r in reports] for m in METRICS}
    pooled = {m: {"share": pooled_share(parts[m], "share"), "share_prop": pooled_share(parts[m], "share_prop")} for m in METRICS}
    out.append("| pooled over the blocks | | | | " + " | ".join(cell(pooled[m]) for m in METRICS) + " |")
    out += ["", "Plain funnel z by look (floor / deep_f):", ""]
    for r in reports:
        out.append(f"- {r['block']}: " + "; ".join(f"{t['n']}: {t['floor_z']:+.2f} / {t['deep_z']:+.2f}" for t in r["plain_replay"]["looks"]))
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--agent-root", default=str(DEFAULT_AGENT_ROOT))
    ap.add_argument("--rule", default="v2", choices=("v2", "all"), help="the ledger rows to judge (default the keep test v2's)")
    ap.add_argument("--json", default=None, help="write the figures here")
    a = ap.parse_args(argv)
    agent = Path(a.agent_root)
    ledger = read_ledger(agent / "promotions.tsv")
    reports = [r for r in (block_report(agent, row) for row in judged_blocks(ledger, a.rule)) if r is not None]
    audit = audit_all_blocks(agent, judged_blocks(ledger, "all"))
    text = render(reports, audit)
    print(text)
    if a.json:
        Path(a.json).write_text(json.dumps({"audit": audit, "blocks": reports}, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                                encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
