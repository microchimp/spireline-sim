"""The pairing check of docs/PROMOTION_GUARD.md section 4 (points 1 and 2) on two arms' runs.jsonl.

Read-only: it reads two arm folders (``runs.jsonl`` and ``summary.json`` each), optionally the block's seed file and the
block's ``promotion.json``, and prints, per block, every quantity the section names with its pass/fail:

  seeds     the two arms' seed sets are identical (and equal to the seed file); paired_seeds == n_seeds
  config    the arms' summaries agree on config, ascension, character, agent seed and sims
  crashes   small (each at most 1% of the block) and alike (a two-sided sign test on the two counts at 0.05)
  kills     the discordant pairs b + c are below n (p1 + p2 - 2 p1 p2), the count independent arms would give
  floors    the SD of the paired floor differences is at most 0.9 times sqrt(var1 + var2), and the correlation positive

Beside each bound it gives a permutation reference (the candidate's seeds shuffled against the incumbent's, which keeps both
marginals and breaks the pairing), so a count that is below the bound by chance alone shows as such, and the same two
independence assumptions the power tables rest on, re-run with the observed discordance (``power_with_discordance``).
The engine build (a third condition of section 4.1) is in no runs row; it comes from the job record and is not checked here.

    python -I tools/lab/pairing_check.py CAND_ARM_DIR BASE_ARM_DIR [--seeds-file F] [--promotion promotion.json] [--json OUT]

A block the funnel stopped early (D51; promotion.json carries ``sequential`` with a stop before the last look) holds only the
seeds played, the first N of the block in file order: the arms are then expected to hold exactly those N (the seed set equal to the
seed file's first N, ``paired_seeds`` equal to N), and the output says so. The record is promotion.json; with ``--seeds-file`` and no
``--promotion`` the candidate's own ``promotion.json`` (two folders above its arm folder) is used when it is for the same block.
Every quantity is then that of the seeds played, which is fewer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tools.lab import promotion_guard as pg  # noqa: E402

CRASH_SHARE = 0.01  # a crash count is "small" at or below this share of the block
SD_RATIO_BOUND = 0.9  # "well below": the paired SD at most this share of the independent one (correlation about 0.19 at equal variances)
SUMMARY_KEYS = ("config", "sims", "ascension", "character", "agent_seed")
PERMUTATIONS = 2000
PERM_SEED = 20261025
Z_ONE_SIDED_05 = 1.645


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """The run rows of a runs.jsonl: lines with a seed (a real-game file also has event lines), the last row of a seed wins."""
    by_seed: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or "seed" not in row or ("ev" in row and row["ev"] != "end"):
            continue
        by_seed[row["seed"]] = row
    out = list(by_seed.values())
    return out


def load_seed_file(path: str | Path) -> list[str]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(doc["seeds"] if isinstance(doc, dict) else doc)


def _crashed(row: Mapping[str, Any]) -> bool:
    return bool(row.get("crash"))


def funnel_stop(promotion: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The funnel's early stop of a promotion.json (`{"action", "played", "block_seeds"}`), or None for a block that played every
    seed or was not played as the funnel. The same reading as `rl/gen_loop.funnel_stop_record` (a test holds them together); this
    module reads no repo code beyond the guard's, so it carries its own."""
    seq = (promotion or {}).get("sequential")
    if not isinstance(seq, dict) or not isinstance(seq.get("stop"), dict):
        return None
    action, played, total = seq["stop"].get("action"), seq.get("played"), seq.get("block_seeds")
    if action in (None, "final") or not isinstance(played, int) or not isinstance(total, int) or not 0 < played < total:
        return None
    return {"action": action, "played": played, "block_seeds": total}


def sign_test_two_sided(a: int, b: int) -> float:
    """Two-sided exact sign test on two counts (1 when both are 0)."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def guard_power_with_discordance(n: int, d: float, q: float, alpha: float) -> float:
    """pg.guard_power with the discordance rate `d` and the incumbent-only share `q` given instead of derived from the rates."""
    if d <= 0.0:
        return 0.0
    power = 0.0
    for m in range(n + 1):
        pm = pg.binom_pmf(n, m, d)
        if pm < 1e-15:
            if m > n * d:
                break
            continue
        power += pm * pg.binom_sf(m, pg.critical_count(m, alpha), q)
    return min(1.0, power)


def power_with_discordance(factor: float, n: int = 2000, alphas: Sequence[float] = (0.05, 0.01),
                           cases: Sequence[tuple[float, float]] = ((0.02, 0.015), (0.02, 0.013), (0.02, 0.01), (0.02, 0.005))
                           ) -> list[dict[str, Any]]:
    """The guard's power of the doc's table under independence against the same with the discordance scaled by `factor`
    (observed discordant pairs over the independent expectation), the incumbent-only share unchanged."""
    out = []
    for p1, p2 in cases:
        d = p1 + p2 - 2 * p1 * p2
        q = (p1 - p1 * p2) / d
        for a in alphas:
            out.append({"incumbent": p1, "candidate": p2, "n": n, "alpha": a, "independent": pg.guard_power(n, p1, p2, a),
                        "scaled": guard_power_with_discordance(n, min(1.0, d * factor), q, a), "factor": factor})
    return out


def check_pair(cand_rows: Sequence[Mapping[str, Any]], base_rows: Sequence[Mapping[str, Any]], *,
               cand_summary: Mapping[str, Any] | None = None, base_summary: Mapping[str, Any] | None = None,
               seed_file: Sequence[str] | None = None, n_seeds: int | None = None, paired_seeds: int | None = None,
               early_stop: Mapping[str, Any] | None = None, permutations: int = PERMUTATIONS, rng_seed: int = PERM_SEED) -> dict[str, Any]:
    """Every section 4.1 / 4.2 quantity for one block. `n_seeds` and `paired_seeds` are promotion.json's (default: the seed
    file's length and the number of seeds both arms played). `early_stop` (`funnel_stop` of promotion.json) says the funnel ended
    the block before its last look: the arms are then expected to hold the first `played` seeds of the seed file, and the seed
    file must be the block of the record (`block_seeds` long)."""
    c = {r["seed"]: r for r in cand_rows}
    b = {r["seed"]: r for r in base_rows}
    both = sorted(set(c) & set(b))
    n = len(both)
    out: dict[str, Any] = {}

    # 4.1 the seed sets, the counts, the configuration
    played = int(early_stop["played"]) if early_stop else None
    expected_n = (n_seeds if n_seeds is not None else played if played is not None
                  else (len(seed_file) if seed_file is not None else n))
    seeds = {"cand_n": len(c), "base_n": len(b), "paired": n, "only_cand": len(set(c) - set(b)), "only_base": len(set(b) - set(c)),
             "identical": set(c) == set(b), "n_seeds": expected_n,
             "paired_seeds": paired_seeds if paired_seeds is not None else n}
    if early_stop:  # a stop before the last look: the seeds played, not the block, are what both arms hold
        seeds["early_stop"] = {"action": early_stop["action"], "played": played, "block_seeds": early_stop["block_seeds"],
                               "pass": bool(played == expected_n)}
    if seed_file is not None:
        sf = set(seed_file[:played] if played is not None else seed_file)
        seeds["equal_to_seed_file"] = set(c) == sf and set(b) == sf
        if early_stop:
            seeds["early_stop"]["seed_file_is_the_block"] = len(seed_file) == early_stop["block_seeds"]
            seeds["early_stop"]["pass"] = bool(seeds["early_stop"]["pass"] and seeds["early_stop"]["seed_file_is_the_block"])
    seeds["pass"] = bool(seeds["identical"] and seeds["paired_seeds"] == expected_n and seeds.get("equal_to_seed_file", True)
                         and seeds.get("early_stop", {}).get("pass", True))
    out["seeds"] = seeds

    cs, bs = dict(cand_summary or {}), dict(base_summary or {})
    diffs = {k: [cs.get(k), bs.get(k)] for k in SUMMARY_KEYS if k in cs and k in bs}  # a key one summary lacks cannot be compared
    mismatch = {k: v for k, v in diffs.items() if v[0] != v[1]}
    unchecked = [k for k in SUMMARY_KEYS if k not in diffs]
    out["config"] = {"compared": diffs, "mismatch": mismatch, "unchecked": unchecked, "pass": not mismatch and bool(diffs)}

    crash_c = sum(_crashed(r) for r in c.values())
    crash_b = sum(_crashed(r) for r in b.values())
    p_sign = sign_test_two_sided(crash_c, crash_b)
    out["crashes"] = {"cand": crash_c, "base": crash_b, "bound_each": CRASH_SHARE * max(n, 1), "p_alike": p_sign,
                      "pass": bool(max(crash_c, crash_b) <= CRASH_SHARE * n and p_sign >= 0.05)}
    if n == 0:
        out["pass"] = False
        return out

    kc = np.array([pg.boss_kill(c[s]) for s in both], dtype=bool)
    kb = np.array([pg.boss_kill(b[s]) for s in both], dtype=bool)
    fc = np.array([float(c[s].get("floor") or 0) for s in both])
    fb = np.array([float(b[s].get("floor") or 0) for s in both])
    # a crashed run counts as floor 0 and no kill (section 4.1)
    for arr_f, arr_k, rows in ((fc, kc, c), (fb, kb, b)):
        for i, s in enumerate(both):
            if _crashed(rows[s]):
                arr_f[i] = 0.0
                arr_k[i] = False

    # 4.2 the kills
    p1, p2 = float(kb.mean()), float(kc.mean())
    bb = int(np.sum(kc & ~kb))  # candidate-only kills
    cc = int(np.sum(kb & ~kc))  # incumbent-only kills
    both_k = int(np.sum(kc & kb))
    disc = bb + cc
    d_ind = p1 + p2 - 2 * p1 * p2
    expected = n * d_ind
    sd = math.sqrt(n * d_ind * (1 - d_ind)) if 0 < d_ind < 1 else 0.0
    rng = np.random.default_rng(rng_seed)
    perm_disc = np.empty(permutations)
    perm_ratio = np.empty(permutations)
    var_sum = float(fc.var(ddof=1) + fb.var(ddof=1)) if n > 1 else 0.0
    for i in range(permutations):
        idx = rng.permutation(n)
        perm_disc[i] = np.sum(kc[idx] != kb)
        perm_ratio[i] = (np.std(fc[idx] - fb, ddof=1) / math.sqrt(var_sum)) if var_sum > 0 else 1.0
    phi_den = math.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    phi = ((both_k / n) - p1 * p2) / phi_den if phi_den > 0 else float("nan")
    out["kills"] = {"n": n, "p_incumbent": p1, "p_candidate": p2, "cand": int(kc.sum()), "base": int(kb.sum()), "both": both_k,
                    "b_cand_only": bb, "c_base_only": cc, "discordant": disc, "expected_independent": expected,
                    "sd_independent": sd, "ratio": disc / expected if expected > 0 else float("nan"),
                    "z": (disc - expected) / sd if sd > 0 else float("nan"), "phi": phi,
                    "perm_mean": float(perm_disc.mean()), "perm_p_low": float((np.sum(perm_disc <= disc) + 1) / (permutations + 1)),
                    "bound": expected, "pass": bool(disc < expected),
                    "significantly_below": bool(sd > 0 and (disc - expected) / sd <= -Z_ONE_SIDED_05)}
    if bb + cc > 0:
        out["kills"]["base_only_share"] = cc / (bb + cc)
        out["kills"]["base_only_share_independent"] = (p1 * (1 - p2)) / d_ind if d_ind > 0 else float("nan")

    # 4.2 the floors
    diff = fc - fb
    sd_diff = float(diff.std(ddof=1)) if n > 1 else float("nan")
    sd_ind = math.sqrt(var_sum)
    r = float(np.corrcoef(fc, fb)[0, 1]) if fc.std() > 0 and fb.std() > 0 else float("nan")
    out["floors"] = {"mean_cand": float(fc.mean()), "mean_base": float(fb.mean()), "sd_cand": float(fc.std(ddof=1)),
                     "sd_base": float(fb.std(ddof=1)), "sd_diff": sd_diff, "sd_independent": sd_ind,
                     "ratio": sd_diff / sd_ind if sd_ind > 0 else float("nan"), "bound_ratio": SD_RATIO_BOUND, "pearson_r": r,
                     "perm_ratio_mean": float(perm_ratio.mean()), "perm_ratio_p5": float(np.quantile(perm_ratio, 0.05)),
                     "perm_p_low": float((np.sum(perm_ratio <= sd_diff / sd_ind) + 1) / (permutations + 1)) if sd_ind > 0 else 1.0,
                     "pass": bool(sd_ind > 0 and sd_diff / sd_ind <= SD_RATIO_BOUND and r > 0)}
    out["pass"] = bool(all(out[k]["pass"] for k in ("seeds", "config", "crashes", "kills", "floors")))
    return out


def failing(result: Mapping[str, Any]) -> list[str]:
    return [k for k in ("seeds", "config", "crashes", "kills", "floors") if k in result and not result[k]["pass"]]


def _fmt(name: str, r: Mapping[str, Any]) -> str:
    s, k, f, cr = r["seeds"], r.get("kills", {}), r.get("floors", {}), r["crashes"]
    stop = s.get("early_stop")
    stopped = (f" [funnel stop on a {stop['action']}: the first {stop['played']} of the block's {stop['block_seeds']} seeds were played, "
               "and every quantity below is that of those seeds]") if stop else ""
    lines = [f"{name}: {'PASS' if r['pass'] else 'FAIL ' + ','.join(failing(r))}",
             f"  seeds   paired {s['paired']} of {s['n_seeds']} (paired_seeds {s['paired_seeds']}), only-cand {s['only_cand']} only-base {s['only_base']}, "
             f"identical {s['identical']}, seed file {s.get('equal_to_seed_file', 'n/a')}  {'ok' if s['pass'] else 'FAIL'}{stopped}",
             f"  config  mismatch {r['config']['mismatch'] or 'none'}  {'ok' if r['config']['pass'] else 'FAIL'}",
             f"  crashes cand {cr['cand']} base {cr['base']} (sign p {cr['p_alike']:.2f})  {'ok' if cr['pass'] else 'FAIL'}"]
    if k:
        lines.append(f"  kills   cand {k['cand']} base {k['base']} (p {k['p_candidate']:.4f}/{k['p_incumbent']:.4f}); b {k['b_cand_only']} c {k['c_base_only']} "
                     f"b+c {k['discordant']} vs bound {k['bound']:.1f} (ratio {k['ratio']:.2f}, z {k['z']:+.2f}, perm p {k['perm_p_low']:.3f}, phi {k['phi']:.2f})  "
                     f"{'ok' if k['pass'] else 'FAIL'}")
        lines.append(f"  floors  SD diff {f['sd_diff']:.3f} vs sqrt(v1+v2) {f['sd_independent']:.3f} (ratio {f['ratio']:.3f}, bound {f['bound_ratio']}, "
                     f"r {f['pearson_r']:+.3f}, perm ratio {f['perm_ratio_mean']:.3f})  {'ok' if f['pass'] else 'FAIL'}")
    return "\n".join(lines)


def check_dirs(cand_dir: str | Path, base_dir: str | Path, *, seeds_file: str | Path | None = None,
               promotion: str | Path | None = None, **kw: Any) -> dict[str, Any]:
    cand_dir, base_dir = Path(cand_dir), Path(base_dir)
    summ = lambda d: json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").is_file() else {}  # noqa: E731
    seed_file = load_seed_file(seeds_file) if seeds_file else None
    if promotion is None and seeds_file:  # the candidate's own record, when it is the record of this block
        parents = cand_dir.resolve().parents
        own = parents[1] / "promotion.json" if len(parents) > 1 else None
        if own is not None and own.is_file():
            try:
                same = json.loads(own.read_text(encoding="utf-8")).get("seed_block") == Path(seeds_file).stem
            except (OSError, ValueError):
                same = False
            promotion = own if same else None
    promo = json.loads(Path(promotion).read_text(encoding="utf-8")) if promotion else {}
    return check_pair(read_rows(cand_dir / "runs.jsonl"), read_rows(base_dir / "runs.jsonl"), cand_summary=summ(cand_dir),
                      base_summary=summ(base_dir), seed_file=seed_file, n_seeds=promo.get("n_seeds"),
                      paired_seeds=promo.get("paired_seeds"), early_stop=funnel_stop(promo), **kw)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cand_dir")
    ap.add_argument("base_dir")
    ap.add_argument("--seeds-file")
    ap.add_argument("--promotion")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = check_dirs(a.cand_dir, a.base_dir, seeds_file=a.seeds_file, promotion=a.promotion)
    print(_fmt(Path(a.cand_dir).name, res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
