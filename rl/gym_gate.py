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


# ---------------------------------------------------------------------------------------------------------- rule v4 (D57 addendum 9)
#
# Rule v4 judges the candidate on the fight exam's human stratum (400 starts of Acts 1 and 2) under the player's own search, paired by
# (start, draw j) with the incumbent, at a resolution that sees about +1 point. Everything here is pure: rows in, numbers out. A game row
# carries `start`, `j`, `won`, `capped`, `start_hp`, `end_hp` (a capped game is lost, as in v3). The statistics are v3's: the paired
# difference per (start, j), its mean, the start-clustered SE (`cluster_mean_se`), one-sided bounds at z 1.645 (95) and z 1.282 (90).

Z90 = 1.2815515655446004  # one-sided 90%


def hp_lost(row: dict) -> float:
    """HP lost in a fight: the start HP less the end HP, the whole start HP when the game was lost (or capped)."""
    hp0 = float(row.get("start_hp") or 0.0)
    return hp0 - float(row.get("end_hp") or 0.0) if win(row) else hp0


def indexed(rows: Iterable[dict], want: Callable[[int], bool] | None = None, starts: set | None = None) -> dict[tuple[str, int], dict]:
    """{(start, j): row} (the last row of a key wins), j accepted by `want`, starts in `starts` when given."""
    return {(r["start"], int(r["j"])): r for r in rows
            if (want is None or want(int(r["j"]))) and (starts is None or r["start"] in starts)}


def exam_pairs(cand: dict, inc: dict) -> tuple[dict, dict, list]:
    """(survival, HP lost, keys) pair differences, candidate minus incumbent, over the (start, j) both played: {key: diff} twice."""
    keys = sorted(set(cand) & set(inc))
    surv = {k: float(win(cand[k]) - win(inc[k])) for k in keys}
    hp = {k: hp_lost(cand[k]) - hp_lost(inc[k]) for k in keys}
    return surv, hp, keys


def exam_read(cand: dict, inc: dict, cells: dict[str, str] | None = None) -> dict:
    """The paired reading of two indexed game sets: survival and HP lost of each side on the pairs both played, the difference, its
    clustered SE and bounds (`ub95` = m + Z95 SE, `lb90` = m - Z90 SE; fractions for survival, HP for HP lost); `by_cell` (start id to
    cell name in `cells`) the same per cell, with no bounds."""
    surv, hp, keys = exam_pairs(cand, inc)
    m, se, n, c = cluster_mean_se(surv)
    hm, hse, _n, _c = cluster_mean_se(hp)
    out = {"pairs": n, "starts": c, "cand_wins": sum(win(cand[k]) for k in keys), "inc_wins": sum(win(inc[k]) for k in keys),
           "win_diff": m, "win_se": se, "ub95": None if m is None or se is None else m + Z95 * se,
           "lb90": None if m is None or se is None else m - Z90 * se, "hp_diff": hm, "hp_se": hse,
           "cand_hp_lost": None if not keys else sum(hp_lost(cand[k]) for k in keys) / n,
           "inc_hp_lost": None if not keys else sum(hp_lost(inc[k]) for k in keys) / n,
           "cand_rate": None if not n else sum(win(cand[k]) for k in keys) / n,
           "inc_rate": None if not n else sum(win(inc[k]) for k in keys) / n}
    if cells:
        by: dict[str, dict] = {}
        for cell in sorted({cells[k[0]] for k in keys if k[0] in cells}):
            sub = [k for k in keys if cells.get(k[0]) == cell]
            cm, cse, cn, cc = cluster_mean_se({k: surv[k] for k in sub})
            hm2, hse2, _a, _b = cluster_mean_se({k: hp[k] for k in sub})
            by[cell] = {"pairs": cn, "starts": cc, "cand_rate": sum(win(cand[k]) for k in sub) / cn,
                        "inc_rate": sum(win(inc[k]) for k in sub) / cn, "win_diff": cm, "win_se": cse,
                        "cand_hp_lost": sum(hp_lost(cand[k]) for k in sub) / cn, "inc_hp_lost": sum(hp_lost(inc[k]) for k in sub) / cn,
                        "hp_diff": hm2, "hp_se": hse2}
        out["by_cell"] = by
    return out


def v4_bar(read: dict, rule: dict) -> dict:
    """Stage 2 and 3's test on a reading: `ok` (promote) when the survival gain is at least `promote.points`, its 90% lower bound is above
    zero and HP lost a fight is not worse by more than `promote.hp_worse` (point estimate); `near` when the gain is at least
    `near_points`."""
    p = rule["promote"]
    m, lb, hp = read["win_diff"], read["lb90"], read["hp_diff"]
    if m is None or lb is None or hp is None:
        return {"ok": False, "near": False, "why": "no pairs to judge"}
    gain_ok, lb_ok, hp_ok = 100.0 * m >= float(p["points"]), lb > 0.0, hp <= float(p["hp_worse"])
    why = "; ".join(x for x, bad in ((f"gain {100 * m:+.2f} points is under the {p['points']:+.2f} needed", not gain_ok),
                                     (f"the 90% lower bound {100 * lb:+.2f} points is not above 0", not lb_ok),
                                     (f"HP lost a fight is worse by {hp:+.2f}, over the {p['hp_worse']:+.1f} allowed", not hp_ok)) if bad)
    return {"ok": gain_ok and lb_ok and hp_ok, "near": 100.0 * m >= float(rule["near_points"]), "why": why or None}


def v4_decide(cand: dict, inc: dict, stage: int, rule: dict, cells: dict[str, str] | None = None) -> dict:
    """Rule v4 on the indexed games of the candidate and the incumbent, restricted by the caller to the stage's draws. Stage 1: `reject`
    when the 95% UPPER bound of the survival gain is under `screen.points`, else `continue` (to stage 2). Stages 2 and 3: `promote` when
    `v4_bar` passes; else `continue` (stage 2 only: on to stage 3) when the gain is at least `near_points`; else `reject`. `incomplete`
    when fewer pairs than the stage's `starts x draws` were played. Returns the reading with verdict, stage and why."""
    read = exam_read(cand, inc, cells)
    need = int(rule["starts"]) * int(rule["stage_draws"][stage - 1])
    if read["pairs"] < need:
        return {"verdict": "incomplete", "stage": stage, "need": need, **read}
    if stage == 1:
        bad = read["ub95"] is not None and 100.0 * read["ub95"] < float(rule["screen"]["points"])
        why = (f"stage 1: the 95% upper bound {100 * read['ub95']:+.2f} points is under the {rule['screen']['points']:+.1f} that marks "
               f"a candidate clearly worse") if bad else None
        return {"verdict": "reject" if bad else "continue", "stage": 1, "why": why, **read}
    bar = v4_bar(read, rule)
    if bar["ok"]:
        return {"verdict": "promote", "stage": stage, "why": None, "path": "exam", **read}
    last = stage >= len(rule["stage_draws"])
    if bar["near"] and not last:
        return {"verdict": "continue", "stage": stage, "path": None, "why": f"stage {stage}: {bar['why']}; the gain is over "
                f"{rule['near_points']:+.1f}, so stage {stage + 1}", **read}
    return {"verdict": "reject", "stage": stage, "why": f"stage {stage}: {bar['why']}", "path": None, **read}


def act3_guard(cand: dict, inc: dict, rule: dict, cells: dict[str, str] | None = None) -> dict:
    """The Act 3 guard on the indexed games of one draw: `fail` when the 95% UPPER bound of the survival difference (candidate minus
    incumbent) is under `act3_guard.points`, as the stage-1 screen does; reported by cell. No pairs: no fail."""
    g = rule["act3_guard"]
    read = exam_read(cand, inc, cells)
    ub = None if read["win_diff"] is None or read["win_se"] is None else read["win_diff"] + float(g.get("z", Z95)) * read["win_se"]
    fail = ub is not None and 100.0 * ub < float(g["points"])
    return {"fail": fail, "pairs": read["pairs"], "win_diff": read["win_diff"], "win_se": read["win_se"], "ub95": ub,
            "cand_rate": read["cand_rate"], "inc_rate": read["inc_rate"], "by_cell": read.get("by_cell", {}),
            "why": (f"Act 3 guard: the 95% upper bound {100 * ub:+.2f} points of the difference is under {g['points']:+.1f}"
                    if fail else None)}


def pick_act3_gate(lite: list[dict], n: int, quotas: dict[str, int], seed: int) -> list[str]:
    """The Act 3 guard's fixed subsample: `n` start ids of the Act 3 exam's lite records, drawn once by a seeded hash. `quotas` {boss id
    (the cell name's last part, e.g. "80" for the Test Subject): starts wanted}, split over the boss's cells in proportion to their
    sizes; what the quotas leave of `n` is spread evenly over the other bosses. Within a cell the order is the hash of (seed, start id).
    Returns the ids, by boss and cell."""
    import hashlib  # noqa: PLC0415

    cells: dict[str, list[str]] = {}
    for s in lite:
        cells.setdefault(s["cell"], []).append(s["id"])
    for c in cells:
        cells[c].sort(key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    bosses = sorted({c.rsplit("-", 1)[-1] for c in cells})
    want: dict[str, int] = {b: int(quotas[b]) for b in bosses if b in quotas}
    rest = [b for b in bosses if b not in want]
    left = max(0, n - sum(want.values()))
    for i, b in enumerate(rest):
        want[b] = left // len(rest) + (1 if i < left % len(rest) else 0)
    out: list[str] = []
    for b in bosses:
        cs = sorted(c for c in cells if c.rsplit("-", 1)[-1] == b)
        total = sum(len(cells[c]) for c in cs)
        take = {c: int(round(want[b] * len(cells[c]) / total)) for c in cs}
        take[cs[-1]] = want[b] - sum(take[c] for c in cs[:-1])  # the last cell takes the rounding
        for c in cs:
            out += cells[c][:max(0, min(take[c], len(cells[c])))]
    return out


def identity_sample(keys: Iterable[tuple[str, int]], n: int, seed: str) -> list[tuple[str, int]]:
    """`n` of the (start, j) keys by a seeded hash (all of them when there are fewer): the games an identity check replays."""
    import hashlib  # noqa: PLC0415

    return sorted(sorted(set(keys), key=lambda k: hashlib.sha256(f"{seed}:{k[0]}:{k[1]}".encode()).hexdigest())[:n])


def identity_same(stored: dict, replay: dict) -> bool:
    """Whether a replayed game is the stored one: the same outcome (a capped game is lost), end HP and move hash."""
    return (win(stored) == win(replay) and int(stored.get("end_hp") or 0) == int(replay.get("end_hp") or 0)
            and stored.get("moves_sha") == replay.get("moves_sha"))


def identity_check(stored: dict[tuple[str, int], dict], replay: dict[tuple[str, int], dict]) -> dict:
    """{n, same, ok, bad}: `replay` {(start, j): row} against the stored rows of the same keys; ok only when every key was replayed and
    every one is the same. `bad` lists up to five differing keys."""
    bad = [k for k in sorted(stored) if k not in replay or not identity_same(stored[k], replay[k])]
    return {"n": len(stored), "same": len(stored) - len(bad), "ok": bool(stored) and not bad, "bad": [list(k) for k in bad[:5]]}


# ---------------------------------------------------------------------------------------------------------- rule v5 (D68 addendum 5 decision 4)
#
# Rule v5 reads two parts of its own ruler (`gym-loop/gate5/`): part `h`, human starts of Acts 1 and 2 that the fight exam did not take, and
# part `a3`, the two Act 3 bosses (human openings and our own arrivals). Within a part a cell counts by a fixed weight (our runs' deaths there
# per 100 runs), the estimate is the stratified one, and a candidate passes by one of two routes, each a gain in its own part with the other
# part not clearly worse. Everything here is pure: rows in, numbers out.

Z975 = 1.959963984540054  # one-sided 97.5%: each of the two routes, so the rule as a whole stays near 5 in 100
MARGIN_Z = 2.0  # every printed figure carries its plus or minus this many standard errors (D68 addendum 6 decision 1)
BOSS_FLAG_POINTS = -3.0  # a boss cell whose difference is this many points or worse is flagged (a flag, never a refusal; D68 addendum 6 decision 1)


def strat_read(cand: dict, inc: dict, cells: dict[str, str], weights: dict[str, float]) -> dict:
    """The paired reading of one part: per start the mean over the draws both played of candidate minus incumbent survival (and HP lost);
    per cell the mean of those start means and their sample variance; the part's figure the weighted sum over the cells present (weights
    renormalised over them; a cell with no weight is reported, not counted), its SE sqrt(sum w^2 var / n). `cand`/`inc`: {(start, j): row};
    `cells`: start id -> cell. Fractions for survival, HP for HP lost. A cell of one start adds no variance (flagged in `thin`)."""
    keys = sorted(k for k in set(cand) & set(inc) if k[0] in cells)
    per: dict[str, dict[str, list]] = {}
    for k in keys:
        per.setdefault(cells[k[0]], {}).setdefault(k[0], []).append(k)
    by: dict[str, dict] = {}
    for c, starts in sorted(per.items()):
        d_s, h_s = [], []
        cw = iw = n = 0
        ch = ih = 0.0
        for ks in starts.values():
            d_s.append(sum(win(cand[k]) - win(inc[k]) for k in ks) / len(ks))
            h_s.append(sum(hp_lost(cand[k]) - hp_lost(inc[k]) for k in ks) / len(ks))
            cw += sum(win(cand[k]) for k in ks)
            iw += sum(win(inc[k]) for k in ks)
            ch += sum(hp_lost(cand[k]) for k in ks)
            ih += sum(hp_lost(inc[k]) for k in ks)
            n += len(ks)
        m = len(d_s)
        dm, hm = sum(d_s) / m, sum(h_s) / m
        dv = sum((x - dm) ** 2 for x in d_s) / (m - 1) if m > 1 else 0.0
        hv = sum((x - hm) ** 2 for x in h_s) / (m - 1) if m > 1 else 0.0
        by[c] = {"starts": m, "pairs": n, "weight": float(weights.get(c, 0.0)), "win_diff": dm, "win_se": math.sqrt(dv / m),
                 "win_pm2se": MARGIN_Z * math.sqrt(dv / m), "hp_diff": hm, "hp_se": math.sqrt(hv / m), "hp_pm2se": MARGIN_Z * math.sqrt(hv / m),
                 "cand_rate": cw / n, "inc_rate": iw / n, "cand_hp_lost": ch / n, "inc_hp_lost": ih / n, "_dv": dv, "_hv": hv}
    counted = {c: x for c, x in by.items() if x["weight"] > 0}
    wsum = sum(x["weight"] for x in counted.values())
    out = {"pairs": len(keys), "starts": sum(x["starts"] for x in by.values()), "cells": len(by),
           "cand_wins": sum(win(cand[k]) for k in keys), "inc_wins": sum(win(inc[k]) for k in keys),
           "by_cell": {c: {k: v for k, v in x.items() if not k.startswith("_")} for c, x in by.items()}}
    if not counted or wsum <= 0:
        return {**out, "win_diff": None, "win_se": None, "win_pm2se": None, "hp_diff": None, "hp_se": None, "hp_pm2se": None, "cand_rate": None,
                "inc_rate": None, "cand_hp_lost": None, "inc_hp_lost": None, "thin": []}
    w = {c: x["weight"] / wsum for c, x in counted.items()}

    def wsum_of(key: str) -> float:
        return sum(w[c] * x[key] for c, x in counted.items())

    win_se = math.sqrt(sum(w[c] ** 2 * x["_dv"] / x["starts"] for c, x in counted.items()))
    hp_se = math.sqrt(sum(w[c] ** 2 * x["_hv"] / x["starts"] for c, x in counted.items()))
    return {**out, "win_diff": wsum_of("win_diff"), "win_se": win_se, "win_pm2se": MARGIN_Z * win_se,
            "hp_diff": wsum_of("hp_diff"), "hp_se": hp_se, "hp_pm2se": MARGIN_Z * hp_se,
            "cand_rate": wsum_of("cand_rate"), "inc_rate": wsum_of("inc_rate"), "cand_hp_lost": wsum_of("cand_hp_lost"),
            "inc_hp_lost": wsum_of("inc_hp_lost"), "thin": sorted(c for c, x in counted.items() if x["starts"] < 2)}


def boss_flags(reads: dict[str, dict], points: float = BOSS_FLAG_POINTS) -> list[dict]:
    """The boss cells of the parts' readings (`strat_read`; a cell whose name holds "boss": act1-boss, act3-boss1-80, ...) whose survival difference
    is `points` or worse, in points: [{part, cell, points, pm2se, starts, pairs}], parts and cells in name order. A flag is information on the
    verdict, never a refusal: nothing in `v5_decide` reads it. A cell with no pairs is not flagged."""
    out = []
    for part in sorted(reads):
        for cell, x in sorted((reads[part].get("by_cell") or {}).items()):
            if "boss" in cell and x.get("win_diff") is not None and 100.0 * x["win_diff"] <= points:
                out.append({"part": part, "cell": cell, "points": 100.0 * x["win_diff"],
                            "pm2se": None if x.get("win_se") is None else 100.0 * MARGIN_Z * x["win_se"], "starts": x.get("starts"), "pairs": x.get("pairs")})
    return out


def flag_text(flags: list[dict]) -> str:
    """The flags as one clause of a stage line ("" when there are none): each cell with its difference and plus or minus 2 SE, in points."""
    if not flags:
        return ""
    return ("FLAG, a boss cell at " + f"{BOSS_FLAG_POINTS:+.0f}" + " points or worse (information, not a refusal): "
            + "; ".join(f"part {f['part']} {f['cell']} {f['points']:+.2f}" + ("" if f["pm2se"] is None else f" ±{f['pm2se']:.2f}") + " points" for f in flags))


def v5_worse(read: dict, part: dict, z: float = Z95) -> dict:
    """`worse` when the part's one-sided 95% upper bound of the survival difference is under its `worse` floor (points): the candidate is
    clearly worse there. No pairs: not worse (nothing to judge)."""
    m, se = read.get("win_diff"), read.get("win_se")
    if m is None or se is None:
        return {"worse": False, "ub95": None}
    ub = m + z * se
    return {"worse": 100.0 * ub < float(part["worse"]), "ub95": ub}


def v5_route(read: dict, route: dict) -> dict:
    """One route's own test on its part: the gain at least `points`, its one-sided lower bound at `z` above zero, HP lost a fight not worse
    by more than `hp_worse` (point estimate). {ok, lb, why}."""
    m, se, hp = read.get("win_diff"), read.get("win_se"), read.get("hp_diff")
    if m is None or se is None or hp is None:
        return {"ok": False, "lb": None, "why": "no pairs to judge"}
    lb = m - float(route["z"]) * se
    level = 100.0 * (1.0 - upper_tail(float(route["z"])))
    bad = [x for x, b in ((f"gain {100 * m:+.2f} points is under the {route['points']:+.2f} needed", 100.0 * m < float(route["points"])),
                          (f"the {level:.1f}% lower bound {100 * lb:+.2f} points is not above 0", lb <= 0.0),
                          (f"HP lost a fight is worse by {hp:+.2f}, over the {route['hp_worse']:+.1f} allowed", hp > float(route["hp_worse"])))
           if b]
    return {"ok": not bad, "lb": lb, "why": "; ".join(bad) or None}


def v5_decide(reads: dict[str, dict], stage: int, rule: dict) -> dict:
    """Rule v5 on the two parts' readings at a stage. Stage 1: `reject` when either part is clearly worse (`v5_worse`), else `continue`.
    Stages 2 and 3: `promote` by the first route (in `rule["routes"]` order) whose own test passes while no other part is clearly worse;
    else `continue` (to the next stage; never after the last) when no part is clearly worse and some part's gain reaches its `near` points;
    else `reject`. Returns {verdict, stage, path (the route), why, worse: {part: ...}, routes: {route: ...}}."""
    parts = rule["parts"]
    worse = {p: v5_worse(reads[p], parts[p], float(rule.get("z_worse", Z95))) for p in parts}
    bad = [p for p in parts if worse[p]["worse"]]
    flags = boss_flags(reads, float(rule.get("flag_points", BOSS_FLAG_POINTS)))  # information on the verdict: nothing below reads it
    why_bad = "; ".join(f"part {p} is clearly worse (95% upper bound {100 * worse[p]['ub95']:+.2f} points, under {parts[p]['worse']:+.1f})"
                        for p in bad)
    if stage == 1:
        return {"verdict": "reject" if bad else "continue", "stage": 1, "path": None, "why": f"stage 1: {why_bad}" if bad else None,
                "worse": worse, "routes": {}, "flags": flags}
    routes = {}
    for name, r in rule["routes"].items():
        own = v5_route(reads[r["part"]], r)
        others = [p for p in parts if p != r["part"] and worse[p]["worse"]]
        routes[name] = {**own, "ok": own["ok"] and not others, "own_ok": own["ok"], "blocked_by": others}
    for name in rule["routes"]:
        if routes[name]["ok"]:
            return {"verdict": "promote", "stage": stage, "path": name, "why": None, "worse": worse, "routes": routes, "flags": flags}
    last = stage >= max(len(p["stage_draws"]) for p in parts.values())
    near = [p for p in parts if reads[p].get("win_diff") is not None and 100.0 * reads[p]["win_diff"] >= float(parts[p]["near"])]

    def route_why(n: str, x: dict) -> str:
        bits = [x["why"]] if x["why"] else []
        if x["blocked_by"]:
            bits.append(f"part {', '.join(x['blocked_by'])} is clearly worse")
        return f"route {n}: " + ("; ".join(bits) or "passed")

    why = "; ".join(route_why(n, x) for n, x in routes.items())
    if near and not bad and not last:
        return {"verdict": "continue", "stage": stage, "path": None, "worse": worse, "routes": routes, "flags": flags,
                "why": f"stage {stage}: {why}; part {', '.join(near)} gains at least its near bar, so stage {stage + 1}"}
    return {"verdict": "reject", "stage": stage, "path": None, "worse": worse, "routes": routes, "flags": flags, "why": f"stage {stage}: {why}"}


# ---------------------------------------------------------------------------------------------------------- the arrival exam (judge build 1; a report, no rule)
#
# S1 of the judge design (DESIGN.md sections 3 and 4): the candidate's fight network plays the champion's own first-Act-3-boss arrivals against the
# champion's (`rl/act3_starts.py` builds the ruler and plays the games; the reading is here). One reading is the per-start mean of the survival
# difference (candidate minus champion) over the draws of the test, averaged unweighted over the starts, its SE taken over the starts. The two bounds the
# design names are printed and no rule reads them: S1 would hold at a gain of at least +1.0 passes per 100 arrivals with the one-sided 0.005 lower
# bound above 0, and a 95% upper bound below 0 would count as a reversal for the gate.

Z995 = 2.5758293035489004  # one-sided 99.5%: S1's hold bound (0.005)
ARRIVAL_DRAWS = 6  # draws a start in one test: test k plays draws 6k to 6k + 5
ARRIVAL_HOLD_POINTS = 1.0  # passes per 100 arrivals S1 would ask of a gain


def arrival_js(test: int = 0, draws: int = ARRIVAL_DRAWS) -> tuple[int, ...]:
    """The draws test `test` plays: `draws` x test to `draws` x test + `draws` - 1 (6k to 6k + 5), so every test's draws are new."""
    if test < 0 or draws < 1:
        raise ValueError("a test index is 0 or more and a test plays at least one draw")
    return tuple(range(draws * test, draws * test + draws))


def arrival_cell(per: dict[str, list[tuple[int, int]]]) -> dict:
    """The reading of a set of starts: `per` {start: [(candidate win, champion win, ...) per draw]} as (cand, inc) pairs of 0/1; the per-start mean of
    the difference, the unweighted mean over the starts, the SE over the starts (n - 1), both survival rates (per-start means, unweighted)."""
    starts = len(per)
    if not starts:
        return {"starts": 0, "pairs": 0, "diff": None, "se": None, "pm2se": None, "points": None, "se_points": None, "pm2se_points": None,
                "cand_rate": None, "inc_rate": None}
    means = [math.fsum(c - i for c, i in v) / len(v) for v in per.values()]
    mean = math.fsum(means) / starts
    se = math.sqrt(math.fsum((x - mean) ** 2 for x in means) / (starts - 1) / starts) if starts > 1 else None
    return {"starts": starts, "pairs": sum(len(v) for v in per.values()), "diff": mean, "se": se, "pm2se": None if se is None else MARGIN_Z * se,
            "points": 100.0 * mean, "se_points": None if se is None else 100.0 * se, "pm2se_points": None if se is None else 100.0 * MARGIN_Z * se,
            "cand_rate": math.fsum(math.fsum(c for c, _i in v) / len(v) for v in per.values()) / starts,
            "inc_rate": math.fsum(math.fsum(i for _c, i in v) / len(v) for v in per.values()) / starts}


def arrival_exam(cand: dict, inc: dict, *, ids=None, test: int = 0, draws: int = ARRIVAL_DRAWS, ruler_hash: str | None = None,
                 cells: dict[str, str] | None = None, hold_points: float = ARRIVAL_HOLD_POINTS) -> dict:
    """S1's reading on the indexed games ({(start, j): row}) of the candidate and the champion: the pairs both played, j in test `test`'s draws
    (`arrival_js`), of the starts in `ids` (all by default). The statistic is the per-start mean survival difference (candidate minus champion; a
    capped game is lost) averaged unweighted over the starts, its SE over the starts; `points`, `se_points` and `pm2se_points` are the same in passes
    per 100 arrivals. `lb995` is the one-sided 0.005 lower bound and `ub95` the 95% upper bound (fractions); `reads` says what each would mean (S1 holds:
    a gain of at least `hold_points` with `lb995` above 0; a reversal: `ub95` below 0): a report, nothing reads it. `complete` is true when every start
    of `ids` has all its draws. `by_cell` (start id to cell name in `cells`) is the same reading per cell. `ruler_hash` is carried through."""
    js = arrival_js(test, draws)
    want = set(js)
    keys = sorted(k for k in set(cand) & set(inc) if k[1] in want and (ids is None or k[0] in ids))
    per: dict[str, list[tuple[int, int]]] = {}
    for start, j in keys:
        per.setdefault(start, []).append((win(cand[(start, j)]), win(inc[(start, j)])))
    out = {"report_only": True, "statistic": "per-start mean survival difference (candidate minus champion), unweighted over the starts, SE over the starts",
           "test": test, "draws": list(js), "draws_per_start": draws, "ruler_hash": ruler_hash, **arrival_cell(per)}
    out["complete"] = bool(per) and all(len(v) == draws for v in per.values()) and (ids is None or len(per) == len(set(ids)))
    m, se = out["diff"], out["se"]
    out["lb995"] = None if m is None or se is None else m - Z995 * se
    out["ub95"] = None if m is None or se is None else m + Z95 * se
    out["hold_points"] = hold_points
    out["reads"] = {"hold": bool(out["lb995"] is not None and 100.0 * m >= hold_points and out["lb995"] > 0.0),
                    "reversal": bool(out["ub95"] is not None and out["ub95"] < 0.0)}
    if cells:
        by: dict[str, dict[str, list[tuple[int, int]]]] = {}
        for start, v in per.items():
            by.setdefault(cells.get(start, "?"), {})[start] = v
        out["by_cell"] = {c: arrival_cell(v) for c, v in sorted(by.items())}
    return out


def _arrival_line(label: str, x: dict) -> str:
    if x.get("diff") is None:
        return f"{label}: no pairs"
    pm = "" if x["pm2se_points"] is None else f", ±{x['pm2se_points']:.2f} (2 SE)"
    se = "n/a" if x["se_points"] is None else f"{x['se_points']:.2f}"
    return (f"{label}: {x['starts']} starts, {x['pairs']} games; the candidate passes {100 * x['cand_rate']:.1f} in 100, the champion {100 * x['inc_rate']:.1f}; "
            f"difference {x['points']:+.2f} passes per 100 arrivals (SE {se}{pm})")


def arrival_report(result: dict) -> str:
    """The text of an arrival exam's result (`rl/act3_starts.py arrival-exam`): the players, the ruler's hash, the draws, the reading with its SE and
    plus or minus 2 SE, both bounds, and what each would mean; second-boss starts and each cell on their own lines (for the record)."""
    r = result["first_boss"]
    js = result["draws"]
    cand, champ = result["candidate"], result["champion"]
    lines = ["Arrival exam (S1 of the judge design; a report: no rule reads it)",
             f"candidate {cand['dir']} (fight network {cand['policy_sha256'][:12]}) against champion {champ['dir']} ({champ['policy_sha256'][:12]})",
             f"ruler {result['ruler']}, hash {result['ruler_hash']}; config {result['config']} on the {result['prior'].upper()} prior, engine {result['engine']}; "
             f"test {result['test']}: draws {js[0]} to {js[-1]} ({len(js)} a start, paired)",
             _arrival_line("first Act 3 boss (the statistic)", r)]
    if r.get("diff") is not None:
        lb = "n/a" if r["lb995"] is None else f"{100 * r['lb995']:+.2f}"
        ub = "n/a" if r["ub95"] is None else f"{100 * r['ub95']:+.2f}"
        lines += [f"0.005 lower bound {lb} passes per 100; S1 would hold at a gain of at least {r['hold_points']:+.1f} with that bound above 0: "
                  f"{'yes' if r['reads']['hold'] else 'no'}",
                  f"95% upper bound {ub} passes per 100; a reversal when it is below 0: {'yes' if r['reads']['reversal'] else 'no'}"]
    if not r.get("complete"):
        lines.append(f"INCOMPLETE: not every start has its {len(js)} draws for both players ({r['pairs']} paired games of {r['starts']} starts)")
    if result.get("stopped"):
        lines.append(f"stopped: {result['stopped']}")
    for cell, x in sorted((r.get("by_cell") or {}).items()):
        lines.append("  " + _arrival_line(cell, x))
    s = result.get("second_boss_record")
    if s and s.get("starts"):
        lines.append(_arrival_line("second Act 3 boss (for the record)", s))
        for cell, x in sorted((s.get("by_cell") or {}).items()):
            lines.append("  " + _arrival_line(cell, x))
    return "\n".join(lines)
