"""Readiness score for full Ironclad A10 runs, from data already on disk (no AI, no game runs).

    python -m tools.fidelity.readiness [--referee D:/spireline/referee] [--json out.json] [--top 5]

Headline: the share of Codex runs (real recorded runs the walker replays) that the emulator replays identically to
the end, from the newest refset score that has a codex walk. Sub-scores, each with its source file and date:
  codex      runs identical to the end / runs walked; judged-runs match rate; first-divergence floor (median, 10th
             percentile) and the deepest act reached among the stopped runs (act from floor: 1-17, 18-34, 35+, approximate)
  corpus     fights identical in the newest corpus baseline
  lab        open fix groups (non-host-suspect) and the host-suspect count
  data       differences between the game export and the emulator tables (newest audit/data-checks table-check text)
  hypotheses audit setups confirmed / refuted / open (LAB/hypotheses.json when present; setups with no verdict are open)
Blockers: the groups of LAB/groups.json by estimated run impact (host-suspect ones marked), then the codex stops by
the runs they ended. Missing inputs give a null sub-score and a note, never a guess.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

DEFAULT_REFEREE = Path("D:/spireline/referee")
ACT_FLOORS = (17, 34)  # last floor of act 1 and act 2 (approximate)


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _day(p: Path) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(p.stat().st_mtime))


def _rel(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.name


def act_of(floor: int) -> int:
    return 1 if floor <= ACT_FLOORS[0] else 2 if floor <= ACT_FLOORS[1] else 3


def pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def codex_score(root: Path):
    best = None
    for p in (root / "refset").glob("v*/scores/*.json"):
        d = _load(p)
        c = d.get("codex") if isinstance(d, dict) else None
        if not isinstance(c, dict) or "runs" not in c:
            continue
        key = (d.get("scored") or "", p.name)
        if best is None or key > best[0]:
            best = (key, p, d)
    if best is None:
        return None, {"note": "no refset score with a codex walk"}
    _, p, d = best
    c = d["codex"]
    runs, complete = c.get("runs") or 0, c.get("complete") or 0
    per_run = c.get("per_run") or {}
    stopped = sorted(v[1] for v in per_run.values() if v and v[0] != "complete" and isinstance(v[1], int))
    out = {
        "source": _rel(p, root), "date": (d.get("scored") or _day(p))[:10], "label": d.get("label"),
        "runs": runs, "identical_to_end": complete, "identical_pct": pct(complete, runs),
        "judged": c.get("judged"), "judged_matching_pct": c.get("judged_matching_pct"),
        "stopped": len(stopped), "components_at_zero": c.get("components_at_zero") or [],
        "stops_by_class": c.get("stops_by_class") or {}, "stops": c.get("stops") or {},
    }
    if stopped:
        out["first_divergence_floor_median"] = statistics.median(stopped)
        out["first_divergence_floor_p10"] = stopped[int(0.1 * (len(stopped) - 1))]
        acts = {1: 0, 2: 0, 3: 0}
        for f in stopped:
            acts[act_of(f)] += 1
        out["stopped_by_act"] = {str(k): v for k, v in acts.items()}
        out["deepest_act_stopped"] = max(k for k, v in acts.items() if v)
    return out, None


def corpus_score(root: Path):
    best = None
    for p in (root / "corpus" / "baselines").glob("*.json"):
        d = _load(p)
        t = d.get("totals") if isinstance(d, dict) else None
        if isinstance(t, dict) and t.get("fights"):
            key = d.get("made") or ""
            if best is None or key > best[0]:
                best = (key, p, d)
    if best is None:
        return None, {"note": "no corpus baseline"}
    _, p, d = best
    t = d["totals"]
    return {"source": _rel(p, root), "date": (d.get("made") or _day(p))[:10], "fights": t["fights"],
            "identical": t.get("identical", 0), "identical_pct": pct(t.get("identical", 0), t["fights"]),
            "steps_verified_pct": pct(t.get("reanchor_matched", 0), t.get("reanchored", 0))}, None


def lab_score(root: Path):
    p = root / "lab" / "groups.json"
    d = _load(p)
    if not isinstance(d, dict) or "groups" not in d:
        return None, [], {"note": "no lab groups.json"}
    gs = d["groups"]
    host = [g for g in gs if g.get("host_suspect")]
    real = [g for g in gs if not g.get("host_suspect")]
    blockers = sorted(gs, key=lambda g: -(g.get("impact_runs_est") or 0))
    return {"source": _rel(p, root), "date": (d.get("generated_utc") or _day(p))[:10], "groups": len(gs),
            "open_groups": len(real), "host_suspect": len(host),
            "open_runs_impact_max": max((g.get("impact_runs_est") or 0 for g in real), default=0)}, blockers, None


def data_score(root: Path):
    best = None
    for p in (root / "audit" / "data-checks").glob("*.txt"):
        m = None
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            mm = re.match(r"TOTAL:\s*(\d+) difference", ln)
            if mm:
                m = int(mm.group(1))
        if m is not None and (best is None or (p.stat().st_mtime, p.name) > (best[0], best[1].name)):
            best = (p.stat().st_mtime, p, m)
    if best is None:
        return None, {"note": "no table-check output"}
    return {"source": _rel(best[1], root), "date": _day(best[1]), "mismatches": best[2]}, None


def hyp_score(root: Path):
    ids = {}
    files = list((root / "hypotheses").glob("*/*.json"))
    for p in files:
        d = _load(p)
        for h in (d if isinstance(d, list) else [d] if isinstance(d, dict) else []):
            if isinstance(h, dict) and h.get("id"):
                ids[h["id"]] = bool(h.get("testable", True))
    if not ids:
        return None, {"note": "no hypotheses"}
    lp = root / "lab" / "hypotheses.json"
    res = _load(lp) if lp.exists() else None
    res = res if isinstance(res, dict) else {}
    st = {i: (res.get(i) or {}).get("status") for i in ids}
    conf = sum(1 for v in st.values() if v == "confirmed")
    ref = sum(1 for v in st.values() if v == "refuted")
    return {"source": _rel(lp, root) if res else "hypotheses/*/*.json (no verdicts file yet)",
            "date": _day(lp) if res else max(_day(p) for p in files),
            "total": len(ids), "confirmed": conf, "refuted": ref, "open": len(ids) - conf - ref,
            "not_testable": sum(1 for t in ids.values() if not t)}, None


def build(root: Path, top: int = 5) -> dict:
    notes = {}
    out = {}
    for name, fn in (("codex", codex_score), ("corpus", corpus_score), ("data", data_score), ("hypotheses", hyp_score)):
        v, n = fn(root)
        out[name] = v
        if n:
            notes[name] = n["note"]
    lab, groups, n = lab_score(root)
    out["lab"] = lab
    if n:
        notes["lab"] = n["note"]
    cx = out["codex"]
    out["headline_pct"] = cx["identical_pct"] if cx else None
    out["blockers"] = [{"kind": "group", "id": g.get("id"), "title": (g.get("title") or "")[:90],
                        "runs_est": g.get("impact_runs_est"), "host_suspect": bool(g.get("host_suspect")),
                        "backlog": g.get("backlog_ids") or []} for g in groups[:top]]
    if cx:
        out["codex_stop_causes"] = [{"cause": k, "runs": v} for k, v in
                                    sorted(cx["stops"].items(), key=lambda kv: -kv[1])[:top]]
    out["notes"] = notes
    return out


def render(r: dict) -> str:
    L = []
    h = r["headline_pct"]
    L.append(f"READINESS (Ironclad A10, Codex runs replayed identically to the end): {h if h is not None else 'n/a'}%")
    L.append("")
    rows = []
    c = r["codex"]
    if c:
        rows.append(("codex identical", f"{c['identical_to_end']}/{c['runs']} = {c['identical_pct']}%", c["source"], c["date"]))
        rows.append(("codex judged match", f"{c['judged_matching_pct']}% of {c['judged']}", c["source"], c["date"]))
        if "first_divergence_floor_median" in c:
            sa = c["stopped_by_act"]
            rows.append(("first-divergence floor",
                         f"median {c['first_divergence_floor_median']}, p10 {c['first_divergence_floor_p10']}; "
                         f"deepest act {c['deepest_act_stopped']} (stops a1/a2/a3 {sa['1']}/{sa['2']}/{sa['3']})",
                         c["source"], c["date"]))
    k = r["corpus"]
    if k:
        rows.append(("corpus identical", f"{k['identical']}/{k['fights']} fights = {k['identical_pct']}%", k["source"], k["date"]))
    b = r["lab"]
    if b:
        rows.append(("lab open groups", f"{b['open_groups']} open, {b['host_suspect']} host-suspect (of {b['groups']})",
                     b["source"], b["date"]))
    d = r["data"]
    if d:
        rows.append(("data-check mismatches", str(d["mismatches"]), d["source"], d["date"]))
    y = r["hypotheses"]
    if y:
        rows.append(("audit hypotheses", f"{y['confirmed']} confirmed, {y['refuted']} refuted, {y['open']} open of {y['total']}",
                     y["source"], y["date"]))
    w = max((len(x[0]) for x in rows), default=0)
    for a, v, s, dt in rows:
        L.append(f"{a:<{w}}  {v}  [{s}, {dt}]")
    for k2, v in r["notes"].items():
        L.append(f"note: {k2}: {v}")
    L.append("")
    L.append("Top blockers by run impact (lab groups; HOST? = may be a host artefact):")
    for i, g in enumerate(r["blockers"], 1):
        L.append(f"{i}. {g['id']} ~{g['runs_est']} runs{' HOST?' if g['host_suspect'] else ''} "
                 f"{','.join(g['backlog'])} | {g['title']}")
    if r.get("codex_stop_causes"):
        L.append("Top codex stop causes (runs ended):")
        for i, s in enumerate(r["codex_stop_causes"], 1):
            L.append(f"{i}. {s['cause']} {s['runs']}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--referee", type=Path, default=DEFAULT_REFEREE)
    ap.add_argument("--json", type=Path, help="write the result as JSON here")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args(argv)
    r = build(a.referee, a.top)
    print(render(r))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(r, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
