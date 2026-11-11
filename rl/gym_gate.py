"""The fight gym's gate rule v3 and its plateau alarm (D57 item 2 and D57 addendum 2, 2026-11-11). Pure functions: no games, no emulator.

`gym_loop.decide` applies the rule; this module holds the statistics it uses and the alarm's arithmetic.

**Why v3.** Between 2026-10-31 and 11-09 the v2 gate refused 175 candidates at +0.0 to +1.1 points on 2,000 games (SE 0.65): a gain of
a point cannot show p < 0.05 on 2,000 pairs, so the loop stood still while its students improved. The owner's decision (D57): "not worse,
and the trend across candidates positive", the lab's judgement still deciding what ships.

**The rule.** Every game pair is (board start s, seed j); a pair's difference d is the candidate's win minus the incumbent's (a capped
game is lost), so d is -1, 0 or +1. The standard error of a mean of d is the *start-clustered* one (the ten seeds of a start share the
start's difficulty; the clustered SE is about 5% above the pair-level one on the ledger's own games): with C starts, per-start sums S_c
of n_c pairs each, n pairs in all, mean m: SE = sqrt( C/(C-1) * sum_c (S_c - m n_c)^2 ) / n. Z = 1.645, the one-sided 95% point.

- *Stage 1* (400 pairs of one seed pair, as v2) is a screen: it rejects a candidate that is **clearly worse**, when the one-sided 95%
  upper bound of its difference, m + Z SE, is below -2.0 points. (The brief's wording was "lower bound below -2.0"; read literally that
  rejects every candidate under about +0.5 points, most level ones, which is the opposite of a screen that rejects only the clearly
  worse. `screen.bound` "lower" is that reading, kept as a constant.) Everything else goes on to stage 2.
- *Stage 2* (2,000 pairs, the board's games) promotes when either
  (a) **alone**: the old rule, unchanged: the exact one-sided McNemar test on the discordant pairs gives p < 0.05 and the paired mean L0
      difference is not negative (`single`; `min_points` is a floor on top, 0: the brief's "+1.5 points" would have refused the +1.35 points of
      c0185 that v2 promoted at p 0.025, so (a) is v2's rule and adds nothing to it), or
  (b) **not worse and the trend positive**: the candidate's lower bound m - Z SE is above -1.0 point, *and* the pooled paired difference of
      the last five stage-2 candidates of this incumbent, this one included, is positive with one-sided p < 0.10. The pooled difference is
      taken **per start**: a candidate's difference at start s is the mean of its own pairs there, the start's one number u_s is the mean of
      that over the pool's candidates that played s, and the test is the one-sided paired test over starts: P = mean of u_s, SE = sd(u_s) /
      sqrt(starts), z = P / SE, p = 1 - Phi(z). It is the starts' own spread of the candidates' average, so the candidates' shared incumbent
      draws (their differences correlate about 0.5) and their near-identical play are accounted for (five candidates cut the SE by about a
      fifth, not by sqrt 5; combining the five candidates' own SEs as independent would understate it by about 1.7). A pool needs `min` = 3
      candidates before it speaks: a trend is more than one point (replayed: a false promotion in the first five candidates of an incumbent 18% against 20% with no minimum, the
      power to promote a +1 point family in five candidates 78% against 80%: close to a wash, kept for the meaning).
  The pool is the ledger's stage-2 rows with the same incumbent (a promotion resets it) and the same rule version (a row written under v1 or
  v2 was picked on a +1.0 stage-1 floor; its 2,000 pairs carry that selection), whatever the engine (each member's difference is taken
  on its own engine's games).
- *Guards*: at most one promotion in six hours; a candidate that passes inside the window is a reject with `guard` set.

**False promotion** (`referee/backlog/gym-gate/replay.py`, on the ledger's own games): under no difference at all a look promotes about one time
in nine (the pooled test's 10% and the single test's 5%, overlapping), and because the candidates are nearly the same player the looks
repeat each other: the chance of some false promotion is 14% over a four-candidate stretch, 29% over eleven and 32% over eighteen. Promoting
a level candidate costs only a restart of the pool; (b)'s lower bound keeps a worse one out. Replayed on the 34 stage-2 verdicts since 2026-10-31
the rule promotes exactly what v2 promoted (c0001, c0019, c0185, all by (a)) and nothing else: the pooled path never fires, because the
pooled gain of the 18 candidates of v2's long stretch was +0.1 to +0.7 points (SE 0.55) and, on the 1,600 pairs that stage 1 did not
select on, about zero.

**The plateau alarm** (D57 addendum 2): no promotion for 5 days or 25 gated candidates is a failure of the loop: `plateau_update` writes
LAB/alerts/gym-plateau.txt (days, candidates, the last ten verdicts in points) and a log line, and removes the file at the next promotion.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path

Z95 = 1.6448536269514722  # one-sided 95%


def upper_tail(z: float) -> float:
    """P(Z > z) for a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def win(row: dict) -> int:
    """1 for a won game that was not capped (a capped game is lost, the gate's rule), else 0."""
    return 1 if row.get("won") and not row.get("capped") else 0


def pair_diffs(cand: Iterable[dict], inc: Iterable[dict], want: Callable[[int], bool] | None = None) -> dict[tuple[str, int], int]:
    """{(start, j): candidate win - incumbent win} over the pairs both arms played (j accepted by `want`, all by default)."""
    a = {(r["start"], int(r["j"])): win(r) for r in cand if want is None or want(int(r["j"]))}
    b = {(r["start"], int(r["j"])): win(r) for r in inc if want is None or want(int(r["j"]))}
    return {k: a[k] - b[k] for k in sorted(set(a) & set(b))}


def cluster_mean_se(d: dict[tuple[str, int], float]) -> tuple[float | None, float | None, int, int]:
    """(mean, start-clustered standard error, pairs, starts) of a pair-keyed difference. The SE is None with fewer than two starts."""
    n = len(d)
    if not n:
        return None, None, 0, 0
    m = sum(d.values()) / n
    sums: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for (start, _j), x in d.items():
        sums[start] = sums.get(start, 0.0) + x
        sizes[start] = sizes.get(start, 0) + 1
    c = len(sums)
    if c < 2:
        return m, None, n, c
    ss = sum((sums[s] - m * sizes[s]) ** 2 for s in sums)
    return m, math.sqrt(c / (c - 1) * ss) / n, n, c


def screen_stage1(d: dict[tuple[str, int], float], screen: dict) -> dict:
    """Stage 1's screen on the stage-1 pairs' differences: `reject` when the candidate is clearly worse. `screen`: `bound` ("upper": the
    one-sided 95% upper bound m + z SE under `points`; "lower": m - z SE under `points`, the literal reading), `z`, `points`."""
    m, se, n, c = cluster_mean_se(d)
    if m is None or se is None:
        return {"reject": False, "diff": m, "se": se, "bound": None, "n": n, "starts": c}
    z = float(screen.get("z", Z95))
    bound = m + z * se if screen.get("bound", "upper") == "upper" else m - z * se
    return {"reject": 100.0 * bound < float(screen["points"]), "diff": m, "se": se, "bound": bound, "n": n, "starts": c}


def pooled(members: list[dict[tuple[str, int], float]]) -> dict:
    """The pooled paired difference of `members` (each a candidate's pair differences, candidate minus the shared incumbent), per start.

    The estimator: each candidate's difference at a start is the mean of its own pairs there (its seeds); a start's one number is the mean of
    that over **the candidates that played the start** (all of them, in the gate: every stage-2 candidate plays the board's 200 starts x 10
    seeds; a start some did not play is the mean over those that did, and a start nobody played is not in the test); then the one-sided
    paired test over starts: P = mean of the start numbers, SE = sd(start numbers) / sqrt(starts), z = P / SE, p = 1 - Phi(z).
    The five candidates are paired against the same incumbent games, so their differences share the incumbent's draws (correlation about
    0.5); averaging them per start first and taking the spread of those averages across starts carries that correlation by construction
    (five candidates cut the SE by about a fifth, not by sqrt 5). The estimator that treats the candidates as independent samples (the
    candidates' own SEs combined as sqrt(sum SE_i^2) / k) understates the SE by about 1.7 and fires on a shared incumbent draw. The form
    this function held before (a mean pair by pair over the pairs every member played, then the start-clustered SE) is the same statistic
    when every member played the same seeds at every start, as the gate's members do, and differs only where a start was not played in full.

    {k, pairs, starts, partial, diff, se, z, p}: `pairs` the pairs every member played (the pairs the first estimator used; kept as a
    diagnostic), `starts` the starts in the test, `partial` how many of them some member did not play, `p` the one-sided p of 'the pooled
    difference is not positive' (1.0 with fewer than two starts)."""
    if not members:
        return {"k": 0, "pairs": 0, "starts": 0, "partial": 0, "diff": None, "se": None, "z": None, "p": 1.0}
    common = set(members[0])
    for x in members[1:]:
        common &= set(x)
    cells: dict[str, list[float]] = {}
    for x in members:
        sums: dict[str, float] = {}
        sizes: dict[str, int] = {}
        for (start, _j), v in x.items():
            sums[start] = sums.get(start, 0.0) + v
            sizes[start] = sizes.get(start, 0) + 1
        for start, tot in sums.items():
            cells.setdefault(start, []).append(tot / sizes[start])
    per_start = {s: math.fsum(v) / len(v) for s, v in cells.items()}
    c = len(per_start)
    partial = sum(1 for v in cells.values() if len(v) < len(members))
    out = {"k": len(members), "pairs": len(common), "starts": c, "partial": partial}
    if not c:
        return {**out, "diff": None, "se": None, "z": None, "p": 1.0}
    mean = math.fsum(per_start.values()) / c
    if c < 2:
        return {**out, "diff": mean, "se": None, "z": None, "p": 1.0}
    se = math.sqrt(math.fsum((v - mean) ** 2 for v in per_start.values()) / (c - 1) / c)
    if se <= 1e-12 * max(1.0, abs(mean)):  # every start says the same: no spread, so the sign is the whole answer
        return {**out, "diff": mean, "se": se, "z": None, "p": 0.0 if mean > 1e-12 else 1.0}
    z = mean / se
    return {**out, "diff": mean, "se": se, "z": z, "p": upper_tail(z)}


def stage2(own: dict[tuple[str, int], float], earlier: list[dict[tuple[str, int], float]], mcnemar_p: float, l0_diff: float | None,
           rule: dict) -> dict:
    """Stage 2 of the v3 rule for the candidate whose pair differences are `own`, with the pool's earlier members (oldest first, this
    candidate not among them). Returns {ok, path, why, lb, pool, ...}: `path` is "single" (a), "pool" (b) or None."""
    single, pool = rule["single"], rule["pool"]
    m, se, n, _c = cluster_mean_se(own)
    lb = None if m is None or se is None else m - float(pool.get("z_lower", Z95)) * se
    a_ok = (mcnemar_p < float(single["alpha"]) and ((l0_diff or 0.0) >= 0.0 or not single.get("l0_not_worse", True))
            and m is not None and 100.0 * m >= float(single.get("min_points", 0.0)))
    members = [*earlier[-(int(pool["k"]) - 1):], own] if int(pool["k"]) > 1 else [own]
    pl = pooled(members)
    not_worse = lb is not None and 100.0 * lb > float(pool["not_worse_points"])
    pool_ready = pl["k"] >= int(pool["min"])
    trend = pl["diff"] is not None and pl["diff"] > 0 and pl["p"] < float(pool["alpha"])
    b_ok = not_worse and pool_ready and trend
    if a_ok:
        why, path = None, "single"
    elif b_ok:
        why, path = None, "pool"
    elif not not_worse:
        why, path = f"not worse fails: lower bound {100 * (lb or 0.0):+.2f} points is not above {pool['not_worse_points']:+.1f}", None
    elif not pool_ready:
        why, path = f"pool of {pl['k']} under the {pool['min']} candidates a trend needs", None
    else:
        why, path = f"pooled difference {100 * (pl['diff'] or 0.0):+.2f} points of {pl['k']}, p {pl['p']:.3f} not under {pool['alpha']}", None
    return {"ok": a_ok or b_ok, "path": path, "why": why, "diff_se": se, "diff_lb": lb, "pairs": n, "pool": pl,
            "single_ok": a_ok, "pool_ok": b_ok}


# ---------------------------------------------------------------------------------------------------------- the plateau alarm


PLATEAU = {"days": 5.0, "candidates": 25, "last": 10, "file": "gym-plateau.txt"}


def parse_stamp(s) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def plateau_state(ledger: list[dict], promotions: list[dict], now: dt.datetime, cfg: dict | None = None) -> dict:
    """Days since the last promotion (the first verdict's time when there has been none), the gated candidates since (ledger rows with a
    verdict of reject or promote), the last ten verdicts in points, and whether either limit is reached."""
    cfg = cfg or PLATEAU
    since = parse_stamp(promotions[-1].get("t")) if promotions else None
    if since is None and ledger:
        since = parse_stamp(ledger[0].get("t"))
    if since is None:
        since = now
    rows = [r for r in ledger if r.get("verdict") in ("reject", "promote") and (parse_stamp(r.get("t")) or since) > since]
    days = max(0.0, (now - since).total_seconds() / 86400.0)
    last = [{"candidate": r.get("candidate"), "stage": r.get("stage"), "verdict": r.get("verdict"), "points": round(100 * (r.get("win_diff") or 0.0), 2),
             "pairs": r.get("pairs")} for r in [x for x in ledger if x.get("verdict") in ("reject", "promote")][-int(cfg["last"]):]]
    return {"days": days, "candidates": len(rows), "since": since.strftime("%Y-%m-%dT%H:%M:%SZ"), "last": last,
            "tripped": days >= float(cfg["days"]) or len(rows) >= int(cfg["candidates"])}


def plateau_text(st: dict, incumbent: str, now: dt.datetime, cfg: dict | None = None) -> str:
    cfg = cfg or PLATEAU
    lines = [f"ALERT gym-plateau: no promotion for {st['days']:.1f} days and {st['candidates']} gated candidates "
             f"(incumbent {incumbent}, last promotion {st['since']}), raised {now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
             f"The fight loop's gate has promoted nothing for {cfg['days']:g} days or {cfg['candidates']} gated candidates (D57 addendum 2: a failure, "
             "reported to the owner the same day with a diagnosis lane, not a wait). The file is removed by the daemon at the next promotion.",
             "", f"Last {len(st['last'])} verdicts (candidate, stage, verdict, candidate minus incumbent in points on that stage's pairs):"]
    lines += [f"  {x['candidate']} stage {x['stage']} {x['verdict']} {x['points']:+.2f} points ({x['pairs']} pairs)" for x in st["last"]]
    return "\n".join(lines) + "\n"


def plateau_update(ledger: list[dict], promotions: list[dict], incumbent: str, alert_dir: Path, log: Callable[[str], None],
                   now: dt.datetime, cfg: dict | None = None) -> str:
    """Write, refresh or remove LAB/alerts/gym-plateau.txt. Returns "raised" (written now, logged), "standing" (rewritten, no new
    line), "cleared" (removed: a promotion since, logged) or "quiet"."""
    cfg = cfg or PLATEAU
    path = Path(alert_dir) / cfg["file"]
    st = plateau_state(ledger, promotions, now, cfg)
    if st["tripped"]:
        existed = path.is_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(plateau_text(st, incumbent, now, cfg), encoding="utf-8", newline="\n")
        tmp.replace(path)
        if not existed:
            log(f"plateau: no promotion for {st['days']:.1f} days and {st['candidates']} gated candidates (since {st['since']}); wrote {path.parent.name}/{path.name}")
            return "raised"
        return "standing"
    if path.is_file():
        path.unlink()
        log(f"plateau: cleared ({st['days']:.1f} days and {st['candidates']} gated candidates since the last promotion, {st['since']}); removed {path.parent.name}/{path.name}")
        return "cleared"
    return "quiet"


def read_ledger(path: Path) -> list[dict]:
    """gate/verdicts.jsonl, a line cut by a stop skipped."""
    out: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for x in fh:
                if x.strip():
                    try:
                        out.append(json.loads(x))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out
