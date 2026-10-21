"""Summarise serve.py run dirs for the speed spike: per-fight cost, phases, spike timers, checksum equality.

    python an.py <run_dir> [<reference run_dir>]
"""
import json
import statistics as st
import sys
from pathlib import Path


COSTS_ALL = False
WARM = []


def load(d):
    rs = {}
    for f in sorted((Path(d) / "out").glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                o = json.loads(line)
                if "_retire" not in o and "seed" in o:
                    rs.setdefault(f.name, []).append(o)
    # the first fight of each server pays JIT warm-up: dropped from costs unless it is the only one
    warm = [o for v in rs.values() for o in (v[1:] if len(v) > 1 else v)]
    rs = sorted([o for v in rs.values() for o in v], key=lambda o: str(o.get("_id")))
    if not COSTS_ALL:
        globals()["WARM"] = warm
    summ = json.load(open(Path(d) / "summary.json")) if (Path(d) / "summary.json").exists() else {}
    return rs, summ


def sums(rs):
    """Per-seed checksum list and per-step state list (no timing), for game-vs-game fidelity."""
    out = {}
    for r in rs:
        key = (r.get("seed"), r.get("chooser_seed"), json.dumps(r.get("setup"), sort_keys=True)[:200])
        cks = [c[2] for s in r.get("steps", []) for c in (s.get("checksums") or [])]
        states = [json.dumps({"a": s.get("action"), "s": s.get("state")}, sort_keys=True) for s in r.get("steps", [])]
        dec = json.dumps(r.get("decisions") or r.get("log") or r.get("choices"), sort_keys=True, default=str)
        out[key] = (cks, states, dec, r.get("error"))
    return out


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(st.median(xs), 1) if xs else None


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 1) if xs else None


def main():
    d = sys.argv[1]
    rs, summ = load(d)
    w = WARM or rs
    steps = [max(0, len(r.get("steps", [])) - 1) for r in rs]
    cost = [r.get("cost") or {} for r in w]
    tm = [r.get("timing_ms") or {} for r in w]
    ph = [r.get("phase_ms") or {} for r in w]
    print(f"{d}: {len(rs)} results, errors {sum(1 for r in rs if r.get('error'))}, steps/fight {mean(steps)}, "
          f"wall {summ.get('wall_s') or summ.get('seconds')} s, summary keys {sorted(k for k in summ if 'per' in k or 'min' in k)[:6]}")
    print("  cpu_ms mean", mean([c.get("cpu_ms") for c in cost]), "main_cpu_ms mean", mean([c.get("main_cpu_ms") for c in cost]),
          "median", med([c.get("cpu_ms") for c in cost]), "frames mean", mean([c.get("frames") for c in cost]), "gen2", sum(c.get("gen2_gcs", 0) or 0 for c in cost),
          "host", {h: sum(1 for c in cost if c.get("host") == h) for h in {c.get("host") for c in cost}})
    print("  play_ms mean", mean([t.get("play") for t in tm]), "reset_ms mean", mean([t.get("reset") for t in tm]))
    keys = sorted({k for p in ph for k in p})
    print("  phase_ms mean", {k: mean([p.get(k) for p in ph]) for k in keys})
    sp = [c.get("spike") for c in cost if c.get("spike")]
    if sp:
        print("  spike mean/fight", {k: mean([s.get(k) for s in sp]) for k in sp[0]})
    sc = [s.get("cost") for r in w for s in r.get("steps", []) if s.get("cost")]
    if sc:
        print("  per step: frames mean", mean([c.get("frames") for c in sc]), "mcyc mean", mean([c.get("mcyc") for c in sc]),
              "mcyc median", med([c.get("mcyc") for c in sc]), "n", len(sc))
    bp = [r.get("branch_probe") or (r.get("fight") or {}).get("branch_probe") for r in rs]
    bp = [b for b in bp if b]
    if bp:
        print("  branch_probe", {k: (mean([b.get(k) for b in bp]) if isinstance(bp[0].get(k), (int, float)) else bp[0].get(k)) for k in bp[0]})
    ck = sum(len(v[0]) for v in sums(rs).values())
    print("  checksums", ck)
    if len(sys.argv) > 2:
        a, b = sums(rs), sums(load(sys.argv[2])[0])
        same_ck = sum(1 for k in a if k in b and a[k][0] == b[k][0] and a[k][0])
        same_st = sum(1 for k in a if k in b and a[k][1] == b[k][1])
        same_dec = sum(1 for k in a if k in b and a[k][2] == b[k][2])
        diff = [k[0] for k in a if k in b and (a[k][1] != b[k][1] or a[k][0] != b[k][0])]
        print(f"  vs {sys.argv[2]}: checksums identical {same_ck}/{len(a)}, step states identical {same_st}/{len(a)}, "
              f"choices identical {same_dec}/{len(a)}, missing {sum(1 for k in a if k not in b)}; differ {diff[:8]}")


main()
