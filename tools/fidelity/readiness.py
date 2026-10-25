"""Readiness score for full Ironclad A10 runs, from data already on disk (no AI, no game runs).

    python -m tools.fidelity.readiness [--referee D:/spireline/referee] [--json out.json] [--top 5]

Headline: the share of Codex runs (real recorded runs the walker replays) that the emulator replays identically to
the end, from the newest refset score that has a codex walk. Sub-scores, each with its source file and date:
  codex       runs identical to the end / runs walked (and / runs in the set), judged-runs match rate, the stops by
              class, and a full-run estimate: the chance a run still replays identically at the end of act 1, act 2 and
              the deepest floor with enough runs (Kaplan-Meier over the per-run floors: a run that ends in the game
              before a floor is not counted against it; every stop class counts as a divergence, as in the headline). The
              headline alone favours short runs. A stopped run diverged on the floor it stopped on. Act ends are
              floors 17, 33 and 48 (ActModel rooms + 2 per act: 17, 16 and 15 floors).
  corpus      fights identical in the newest corpus baseline
  lab         open fix groups (not host-suspect, not stale) by estimated run exposure, and the host-suspect count; and
              lab.utilisation: CPU minutes the lab's jobs used over the wall minutes the owner left it (last 7 days,
              minus the runner's user / PAUSE / HALT / main.lock spans, per core), against the 90% target, by strategy
  data        differences between the game export and the emulator tables (newest audit/data-checks table-check text)
  agent       the newest agent block's per-arm boss damage, skip rate and cards taken (promotion.json; n/a for rows older than the fields)
  hypotheses  audit setups confirmed / refuted / not testable / open (LAB/hypotheses.json; setups with no verdict are open)
Blockers: the Codex stop causes by the runs they ended (measured), and the open lab groups by the runs that hold their
ingredients (an estimate of exposure, not of stops). Missing inputs give a null sub-score and a note, never a guess.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from itertools import accumulate
from pathlib import Path

DEFAULT_REFEREE = Path("D:/spireline/referee")
# Floors per act: ActModel.GetNumberOfFloors is rooms + 2 (the ancient floor and the boss). Overgrowth and Underdocks have 15
# rooms (17 floors), Hive 14 (16), Glory 13 (15). The Codex walker counts floors across acts, the ancient floor being the first of
# each act (a stop there is at floor 1 or 18). The data agree: complete runs peak at floors 17, 33 and 48-49 (a win can record
# one floor more than the act 3 boss).
ACT_FLOORS = (17, 16, 15)
ACT_ENDS = tuple(accumulate(ACT_FLOORS))  # last floor of act 1, 2 and 3: (17, 33, 48)
ACT_LABEL = {k: f"end of act {i}" for i, k in enumerate(ACT_ENDS, 1)}
MIN_AT_RISK = 100  # the full-run estimate stops where fewer runs than this are still being followed
IMPACT_BUCKETS = ((1000, ">=1000"), (100, "100-999"), (1, "1-99"), (0, "0"))


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
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
    return 1 if floor <= ACT_ENDS[0] else 2 if floor <= ACT_ENDS[1] else 3


def pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def _num(x):
    return int(x) if isinstance(x, float) and x == int(x) else x


def _runs(per_run) -> list[tuple[bool, int]]:
    """(complete, floors) per walked run; entries that are not [status, floors, ...] are skipped."""
    out = []
    for v in (per_run or {}).values():
        if isinstance(v, list) and len(v) >= 2 and isinstance(v[1], int):
            out.append((v[0] == "complete", v[1]))
    return out


def survival_curve(runs: list[tuple[bool, int]]) -> dict[int, tuple[float, int]]:
    """Kaplan-Meier chance that a run still replays identically through floor k, given the game run lasts that long:
    {floor: (pct, runs still followed at that floor)}, from floor 0 (a stop before the first floor) to the deepest floor seen.

    The walker counts a floor before walking it, so floors_walked of a stopped run is the floor it stopped on (an event at
    that floor: the run was followed there and diverged). A complete run of L floors is followed through floor L and then
    leaves (the game run ended; that says nothing about the emulator). Every stop class (EMULATOR, WALKER, PROFILE, FIGHT)
    is an event, as in the headline; the 'judged' figure leaves some of them out."""
    ev, cen = Counter(), Counter()
    for complete, f in runs:
        (cen if complete else ev)[f] += 1
    if not (ev or cen):
        return {}
    top = max(list(ev) + list(cen))
    s, curve = 1.0, {}
    for k in range(0, top + 1):
        at = sum(n for f, n in ev.items() if f >= k) + sum(n for f, n in cen.items() if f >= k)
        if at:
            s *= 1 - ev.get(k, 0) / at
        curve[k] = (round(100.0 * s, 1), at)
    return curve


def survival(runs: list[tuple[bool, int]], min_at_risk: int | None = None) -> dict:
    """{floor: [pct, at_risk]} of survival_curve for the act ends and for the deepest floor with at least `min_at_risk`
    runs still followed (the estimate is not shown where fewer are)."""
    curve = survival_curve(runs)
    if not curve:
        return {}
    min_at_risk = MIN_AT_RISK if min_at_risk is None else min_at_risk
    out = {}
    for k in ACT_ENDS:
        if k in curve and curve[k][1] >= min_at_risk:
            out[k] = list(curve[k])
    deep = [k for k, (_, at) in curve.items() if k >= 1 and at >= min_at_risk]
    if deep and max(deep) not in out:
        out[max(deep)] = list(curve[max(deep)])
    return dict(sorted(out.items()))


def codex_scores(root: Path) -> list[tuple[tuple, Path, dict]]:
    """Every refset score that has a codex walk, oldest first."""
    found = []
    for p in (root / "refset").glob("v*/scores/*.json"):
        d = _load(p)
        c = d.get("codex") if isinstance(d, dict) else None
        if isinstance(c, dict) and "runs" in c and "complete" in c:
            found.append(((d.get("scored") or "", p.name), p, d))
    return sorted(found, key=lambda t: t[0])


def codex_score(root: Path):
    found = codex_scores(root)
    if not found:
        return None, {"note": "no refset score with a codex walk"}
    _, p, d = found[-1]
    c = d["codex"]
    runs, complete = c.get("runs") or 0, c.get("complete") or 0
    allruns = _runs(c.get("per_run"))
    stopped = sorted(f for ok, f in allruns if not ok)
    emu = d.get("emulator") if isinstance(d.get("emulator"), dict) else {}
    out = {
        "source": _rel(p, root), "date": (d.get("scored") or _day(p))[:10], "label": d.get("label"),
        "commit": emu.get("commit") or (c.get("commit") or {}).get("commit"), "dirty": emu.get("dirty"),
        "not_quotable": d.get("not_quotable") or [],
        "runs": runs, "identical_to_end": complete, "identical_pct": pct(complete, runs),
        "runs_in_set": c.get("runs_in_set"), "not_walked": c.get("gap"),
        "identical_pct_of_set": pct(complete, c["runs_in_set"]) if c.get("runs_in_set") else None,
        "judged": c.get("judged"), "judged_matching_pct": c.get("judged_matching_pct"),
        "stopped": runs - complete, "components_at_zero": c.get("components_at_zero") or [],
        "stops_by_class": c.get("stops_by_class") or {}, "stops": c.get("stops") or {},
    }
    if stopped:
        out["first_divergence_floor_median"] = _num(statistics.median(stopped))
        out["first_divergence_floor_p10"] = stopped[int(0.1 * (len(stopped) - 1))]
        acts = Counter(act_of(f) for f in stopped)
        out["stopped_by_act"] = {str(k): acts.get(k, 0) for k in (1, 2, 3)}
    done = sorted(f for ok, f in allruns if ok)
    if done:
        out["complete_run_floors_median"] = _num(statistics.median(done))
        out["complete_runs_by_act"] = {str(k): sum(1 for f in done if act_of(f) == k) for k in (1, 2, 3)}
    sv = survival(allruns)
    if sv:
        out["full_run_estimate"] = [{"floor": k, "identical_pct": v[0], "runs_followed": v[1]} for k, v in sv.items()]
    # the scores before it: how the headline has moved
    byday = {}
    for _, _, s in found:  # the newest score of each day
        byday[(s.get("scored") or "")[:10]] = {"date": (s.get("scored") or "")[:10], "label": s.get("label"),
                                              "identical": s["codex"]["complete"], "runs": s["codex"]["runs"],
                                              "identical_pct": pct(s["codex"]["complete"], s["codex"]["runs"])}
    hist = list(byday.values())
    out["history"] = hist[-5:]
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
    return {"source": _rel(p, root), "date": (d.get("made") or _day(p))[:10], "commit": d.get("commit"),
            "fights": t["fights"], "identical": t.get("identical", 0), "identical_pct": pct(t.get("identical", 0), t["fights"]),
            "steps_verified_pct": pct(t.get("reanchor_matched", 0), t.get("reanchored", 0))}, None


def agent_score(root: Path):
    """The newest agent block (the latest promotion.json under agent/gen*/*/): its two arms' runs, mean boss damage, skip rate, cards
    taken per run and the cards taken by floor, from the per-arm run_fields the tables carry. None where the block's rows predate
    the fields (unknown, not 0); no agent folder: no section and no note."""
    docs = []
    for p in sorted((root / "agent").glob("gen*/*/promotion.json")):
        d = _load(p)
        if isinstance(d, dict) and isinstance(d.get("cand"), dict):
            docs.append((str(d.get("written") or ""), str(p), p, d))
    if not docs:
        return None, None
    written, _, p, d = max(docs, key=lambda x: x[:2])
    arms = {}
    for name in ("cand", "base"):
        s = (d.get(name) or {}).get("summary") or {}
        rf = s.get("run_fields") or {}
        bd, cd = rf.get("boss_damage") or {}, rf.get("cards") or {}
        arms[name] = {"runs": s.get("runs"), "boss_damage_mean": bd.get("mean"), "boss_damage_n": bd.get("n"),
                      "skip_rate": cd.get("skip_rate"), "cards_taken_per_run": cd.get("taken_per_run"),
                      "cards_taken_by_floor": cd.get("taken_by_floor")}
    known = any(a["boss_damage_mean"] is not None or a["skip_rate"] is not None for a in arms.values())
    return ({"source": _rel(p, root), "date": written[:10] or _day(p), "block": d.get("seed_block"), "decision": ("promoted" if d["decision"].get("promote") else "rejected") if isinstance(d.get("decision"), dict) else d.get("decision"),
             "arms": arms},
            None if known else {"note": "the latest agent block's rows predate the boss_damage and cards fields"})


def lab_score(root: Path):
    p = root / "lab" / "groups.json"
    d = _load(p)
    if not isinstance(d, dict) or not isinstance(d.get("groups"), list):
        return None, [], {"note": "no lab groups.json"}
    gs = [g for g in d["groups"] if isinstance(g, dict)]
    host = [g for g in gs if g.get("host_suspect")]
    stale = [g for g in gs if g.get("stale") and not g.get("host_suspect")]
    real = [g for g in gs if not g.get("host_suspect") and not g.get("stale")]
    est = [max(0, g.get("impact_runs_est") or 0) for g in real]
    buckets = {name: 0 for _, name in IMPACT_BUCKETS}
    for e in est:  # IMPACT_BUCKETS runs from the highest floor down; a group goes in the first one it reaches
        buckets[next(name for lo, name in IMPACT_BUCKETS if e >= lo)] += 1
    prof = (d.get("inputs") or {}).get("profile_runs")
    ranked = sorted(real, key=lambda g: (-(g.get("impact_runs_est") or 0), -(g.get("count") or 0), str(g.get("id"))))
    return {"source": _rel(p, root), "date": (d.get("generated_utc") or _day(p))[:10], "commit": d.get("main_sha"),
            "groups": len(gs), "open_groups": len(real), "host_suspect": len(host), "stale": len(stale),
            "open_by_impact_runs": buckets, "profile_runs": prof,
            "open_with_backlog_id": sum(1 for g in real if g.get("backlog_ids")),
            "open_runs_impact_max": max(est, default=0)}, ranked, None


UTIL_DAYS = 7
UTIL_TARGET_PCT = 90.0
OWNER_KINDS = ("user", "pause", "halt", "lock")  # runner.log pauses that are the owner's or another session's, not the lab's
_LOG_LINE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) (.*)$")
_PAUSED = re.compile(r"^paused: (\w+)")


def _ts(s: str) -> float:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


def held_spans(log_text: str, end: float) -> list[tuple[str, float, float]]:
    """(kind, start, end) of every span the runner was held, from runner.log: `paused: <kind>` to `resuming` (or the next
    `runner started`); a `HALT:` line holds until the next start or resume. A span still open at the end of the log runs to `end`."""
    spans: list[tuple[str, float, float]] = []
    cur: tuple[str, float] | None = None
    for line in log_text.splitlines():
        m = _LOG_LINE.match(line)
        if not m:
            continue
        t, msg = _ts(m.group(1)), m.group(2)
        pm = _PAUSED.match(msg)
        if pm:
            if cur and cur[0] == pm.group(1):
                continue  # logged again (a restarted runner reports the pause it finds)
            if cur:
                spans.append((cur[0], cur[1], t))
            cur = (pm.group(1), t)
        elif msg.startswith("HALT:"):
            if cur is None:
                cur = ("halt", t)
        elif msg.startswith("resuming") or msg.startswith("runner started"):
            if cur:
                spans.append((cur[0], cur[1], t))
                cur = None
    if cur:
        spans.append((cur[0], cur[1], max(end, cur[1])))
    return spans


def _union_min(spans: list[tuple[float, float]], lo: float, hi: float) -> float:
    """Minutes covered by the spans, clipped to [lo, hi], overlaps counted once."""
    total, last = 0.0, lo
    for a, b in sorted((max(a, lo), min(b, hi)) for a, b in spans if b > lo and a < hi):
        a = max(a, last)
        if b > a:
            total += b - a
            last = b
    return total / 60.0


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def lab_utilisation(root: Path, now: float | None = None, cores: int | None = None, days: int = UTIL_DAYS):
    """(utilisation, note): CPU minutes the lab's jobs used over the wall minutes the owner left it, last `days` days.
    CPU minutes are the cpu_min of the finished jobs in LAB/yield.tsv; available wall minutes are the window minus the
    runner's owner-use spans (user, PAUSE, HALT, main.lock; machine load is reported but not subtracted), times `cores`
    (a job's cpu_min sums its processes). All-time counters from LAB/state.json ride along. Null and a note without a yield."""
    lab = root / "lab"
    now = time.time() if now is None else now
    cores = max(1, cores or os.cpu_count() or 1)
    try:
        rows = list(csv.DictReader((lab / "yield.tsv").read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    except OSError:
        return None, "no lab yield.tsv: the lab has no finished job to measure"
    lo = now - days * 86400
    try:
        log = (lab / "runner.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        log = ""
    spans = held_spans(log, now)
    owner = _union_min([(a, b) for k, a, b in spans if k in OWNER_KINDS], lo, now)
    held = {k: round(_union_min([(a, b) for kk, a, b in spans if kk == k], lo, now), 1) for k in sorted({k for k, _, _ in spans})}
    by: dict[str, dict] = {}
    for r in rows:
        try:
            t = _ts(r["utc"])
        except (KeyError, ValueError):
            continue
        if lo <= t <= now:
            e = by.setdefault(r.get("strategy") or "?", {"jobs": 0, "cpu_min": 0.0, "wall_min": 0.0})
            e["jobs"] += 1
            e["cpu_min"] += _f(r.get("cpu_min"))
            e["wall_min"] += _f(r.get("wall_min"))
    window = days * 1440.0
    avail = max(0.0, window - owner)
    cap = avail * cores
    cpu = sum(e["cpu_min"] for e in by.values())
    for e in by.values():
        e["share_pct"] = round(100 * e["cpu_min"] / cpu, 1) if cpu else 0.0
        e["pct_of_available"] = round(100 * e["cpu_min"] / cap, 2) if cap else 0.0
        e["cpu_min"], e["wall_min"] = round(e["cpu_min"], 1), round(e["wall_min"], 1)
    pct_ = round(100 * cpu / cap, 1) if cap else None
    st = _load(lab / "state.json")
    st = st if isinstance(st, dict) else {}
    ctr = st.get("counters") or {}
    from tools.lab import mix as labmix
    weights, mix_notes = labmix.load_mix(lab, create=False)
    return {"window_days": days, "from_utc": dt.datetime.fromtimestamp(lo, dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "to_utc": dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "cores": cores,
            "cpu_min": round(cpu, 1), "wall_min": window, "owner_min": round(owner, 1), "available_wall_min": round(avail, 1),
            "held_min": held, "pct": pct_, "target_pct": UTIL_TARGET_PCT,
            "meets_target": (pct_ >= UTIL_TARGET_PCT) if pct_ is not None else None,
            "by_strategy": dict(sorted(by.items(), key=lambda kv: -kv[1]["cpu_min"])),
            "mix_idle_notes": [n for n in mix_notes if labmix.is_idle_note(n)],
            "idle_with_weight": sorted(k for k, w in weights.items() if w > 0 and not by.get(k, {}).get("cpu_min")),
            "all_time": {"cpu_min": round(sum(_f(v) for v in (st.get("strategy_cpu") or {}).values()) / 60, 1),
                         "active_min": round(_f(ctr.get("active_seconds")) / 60, 1), "interrupts": ctr.get("interrupts") or {}},
            "source": "lab/yield.tsv, lab/runner.log, lab/state.json"}, None


def data_score(root: Path):
    best = None
    for p in (root / "audit" / "data-checks").glob("*.txt"):
        total, by = None, {}
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            mm = re.match(r"TOTAL:\s*(\d+) difference", ln)
            if mm:
                total = int(mm.group(1))
            mm = re.match(r"(?!TOTAL)(\w+):\s*(\d+) difference\(s\)", ln)
            if mm and int(mm.group(2)):
                by[mm.group(1)] = int(mm.group(2))
        if total is None and by:
            total = sum(by.values())
        if total is None:
            continue
        nm = re.match(r"\d{4}-\d{2}-\d{2}", p.name)
        day = nm.group(0) if nm else _day(p)
        key = (day, p.stat().st_mtime, p.name)
        if best is None or key > best[0]:
            best = (key, p, total, by)
    if best is None:
        return None, {"note": "no table-check output"}
    (day, _, _), p, total, by = best
    return {"source": _rel(p, root), "date": day, "mismatches": total, "by_check": by}, None


def hyp_score(root: Path):
    from tools.lab import hypotheses as hyp
    from tools.lab import mix as labmix

    hdir, lab = root / "hypotheses", root / "lab"
    setups, problems = hyp.load_setups(hdir)
    if not setups:
        return None, {"note": "no hypotheses"}
    res = hyp.read_results(lab)
    st = Counter()
    for s in setups:
        r = res.get(s["id"]) or {}
        v = r.get("status")
        if v in ("confirmed", "refuted"):
            st[v] += 1
        elif not s.get("testable") or (v == "not testable" and r.get("attempts", 0) >= hyp.MAX_ATTEMPTS):
            st["not_testable"] += 1
        else:
            st["open"] += 1
    lp = lab / "hypotheses.json"
    files = list(hdir.glob("*/*.json"))
    weights, mix_notes = labmix.load_mix(lab, create=False)  # the lab's own reading, defaults included
    weight = weights.get("hypotheses", 0.0)
    return {"source": _rel(lp, root) if lp.exists() else "hypotheses/*/*.json (no verdicts file yet)",
            "date": _day(lp) if lp.exists() else max(_day(p) for p in files),
            "total": len(setups), "confirmed": st["confirmed"], "refuted": st["refuted"],
            "not_testable": st["not_testable"], "open": st["open"], "bad_files": len(problems),
            "lab_mix_weight": weight, "lab_mix_notes": [n for n in mix_notes if not labmix.is_idle_note(n)]}, None  # the idle share is in lab.utilisation


def build(root: Path, top: int = 5, now: float | None = None, cores: int | None = None, days: int = UTIL_DAYS) -> dict:
    notes = {}
    out = {}
    for name, fn in (("codex", codex_score), ("corpus", corpus_score), ("data", data_score), ("hypotheses", hyp_score), ("agent", agent_score)):
        v, n = fn(root)
        out[name] = v
        if n:
            notes[name] = n["note"]
    if out.get("agent") is None:
        out.pop("agent", None)  # no agent block yet: no section, as before
    lab, groups, n = lab_score(root)
    out["lab"] = lab
    if n:
        notes["lab"] = n["note"]
    util, un = lab_utilisation(root, now, cores, days)
    if lab is not None:
        lab["utilisation"] = util
    elif util is not None:
        out["lab_utilisation"] = util  # no groups.json to hang it on
    if un and lab is not None:  # without a lab folder the "lab" note already says so
        notes["lab_utilisation"] = un
    cx = out["codex"]
    out["headline_pct"] = cx["identical_pct"] if cx else None
    out["blockers"] = []
    if cx:
        runs = cx["runs"] or 0
        out["blockers"] = [{"cause": k, "class": k.split(":")[0], "runs": v, "pct_of_runs": pct(v, runs)}
                           for k, v in sorted(cx["stops"].items(), key=lambda kv: (-kv[1], kv[0]))[:top]]
    out["lab_blockers"] = [{"id": g.get("id"), "title": (g.get("title") or "")[:90], "runs_est": g.get("impact_runs_est"),
                            "divergences": g.get("count"), "backlog": g.get("backlog_ids") or [],
                            "relation": g.get("relation")} for g in groups[:top]]
    # cautions that change how far a number can be trusted
    if cx:
        if cx.get("dirty"):
            notes["codex_dirty"] = "the codex score was made on a dirty tree"
        if cx.get("not_quotable"):
            notes["codex_not_quotable"] = "; ".join(str(x) for x in cx["not_quotable"])[:200]
    commits = {k: out[k]["commit"] for k in ("codex", "corpus", "lab") if out.get(k) and out[k].get("commit")}
    if len(set(commits.values())) > 1:
        notes["commits_differ"] = ", ".join(f"{k} {v[:8]}" for k, v in commits.items())
    y = out["hypotheses"]
    if y and y["lab_mix_notes"]:
        notes["lab_mix"] = "; ".join(y["lab_mix_notes"])[:200]
    if y and y["open"] and not y["lab_mix_weight"] and not (y["confirmed"] or y["refuted"]):
        notes["hypotheses_idle"] = "the lab mix gives the hypotheses strategy no weight, so no setup is being run"
    out["notes"] = notes
    return out


def render(r: dict) -> str:
    L = []
    h = r["headline_pct"]
    c = r["codex"]
    L.append(f"READINESS (Ironclad A10): {h if h is not None else 'n/a'}% of Codex runs replayed identically to the end"
             + (f" ({c['identical_to_end']}/{c['runs']})" if c else ""))
    L.append("")
    rows = []
    if c:
        s, dt = c["source"], c["date"]
        rows.append(("codex identical", f"{c['identical_to_end']}/{c['runs']} = {c['identical_pct']}%"
                     + (f" ({c['identical_pct_of_set']}% of the {c['runs_in_set']} runs in the set)" if c.get("identical_pct_of_set") else ""), s, dt))
        rows.append(("codex judged match", f"{c['judged_matching_pct']}% of {c['judged']} (complete runs and EMULATOR stops only)", s, dt))
        for e in c.get("full_run_estimate") or []:
            lab = f" ({ACT_LABEL[e['floor']]})" if e["floor"] in ACT_LABEL else ""
            rows.append((f"full run to floor {e['floor']}{lab}", f"{e['identical_pct']}% still identical ({e['runs_followed']} runs followed)", s, dt))
        if "first_divergence_floor_median" in c:
            sa = c["stopped_by_act"]
            rows.append(("codex stops", f"{c['stopped']} runs; first-divergence floor median {c['first_divergence_floor_median']}, "
                         f"p10 {c['first_divergence_floor_p10']}; act 1/2/3 {sa['1']}/{sa['2']}/{sa['3']}", s, dt))
        if "complete_run_floors_median" in c:
            ca = c["complete_runs_by_act"]
            rows.append(("codex identical runs", f"median {c['complete_run_floors_median']} floors long; end in act 1/2/3 "
                         f"{ca['1']}/{ca['2']}/{ca['3']}", s, dt))
        if c.get("stops_by_class"):
            rows.append(("stops by class", ", ".join(f"{k} {v}" for k, v in sorted(c["stops_by_class"].items(), key=lambda kv: -kv[1])), s, dt))
        if c.get("history") and len(c["history"]) > 1:
            rows.append(("codex trend", "identical share (the run sets differ in size) " + " -> ".join(
                f"{x['identical_pct']}% = {x['identical']}/{x['runs']} ({x['date'][5:]})" for x in c["history"]), s, dt))
    k = r["corpus"]
    if k:
        rows.append(("corpus identical", f"{k['identical']}/{k['fights']} fights = {k['identical_pct']}% ({k['steps_verified_pct']}% of steps verified)",
                     k["source"], k["date"]))
    b = r["lab"]
    if b:
        bk = b["open_by_impact_runs"]
        rows.append(("lab open groups", f"{b['open_groups']} open by runs held: " + ", ".join(f"{n} {v}" for n, v in bk.items())
                     + f"; {b['stale']} stale", b["source"], b["date"]))
        rows.append(("lab host-suspect", f"{b['host_suspect']} of {b['groups']} groups (may be host artefacts, left out of the open count)",
                     b["source"], b["date"]))
        u = b.get("utilisation")
        if u and u["pct"] is not None:
            top_s = ", ".join(f"{k} {v['cpu_min']:.0f}" for k, v in list(u["by_strategy"].items())[:4])
            idle = f"; idle with weight: {', '.join(u['idle_with_weight'])}" if u["idle_with_weight"] else ""
            rows.append(("lab utilisation", f"{u['pct']}% of {u['cores']} cores over {u['window_days']} days, owner time ({u['owner_min']:.0f} min) "
                         f"left out (target {u['target_pct']:.0f}%); CPU min by strategy: {top_s}{idle}", u["source"], u["to_utc"][:10]))
    d = r["data"]
    if d:
        rows.append(("data-check mismatches", f"{d['mismatches']}" + (" (" + ", ".join(f"{a} {n}" for a, n in d["by_check"].items()) + ")" if d["by_check"] else ""),
                     d["source"], d["date"]))
    y = r["hypotheses"]
    if y:
        rows.append(("audit hypotheses", f"{y['confirmed']} confirmed, {y['refuted']} refuted, {y['not_testable']} not testable, "
                     f"{y['open']} open of {y['total']}", y["source"], y["date"]))
    ag = r.get("agent")
    if ag:
        def cell(a):
            f = lambda v, nd=3: "n/a" if v is None else f"{v:.{nd}f}"  # noqa: E731
            return f"boss damage {f(a['boss_damage_mean'])}, skip rate {f(a['skip_rate'])}, cards/run {f(a['cards_taken_per_run'], 2)}"
        rows.append(("agent latest block", f"{ag['block']} ({ag['decision']}): candidate {cell(ag['arms']['cand'])}; "
                     f"incumbent {cell(ag['arms']['base'])}", ag["source"], ag["date"]))
    nt = r.get("nightly")
    if nt and nt.get("agents"):
        def ncell(g, a):
            f = lambda v, nd=3: "n/a" if v is None else f"{v:.{nd}f}"  # noqa: E731
            return (f"{g}: {a.get('runs')} runs, win {f(a.get('win_rate'))}, floor {f(a.get('mean_floor'), 2)}, boss kills {a.get('boss_kills')}, "
                    f"divergence {f(a.get('divergence_rate'), 4)}, {a.get('crashes')} crashes")
        rows.append(("agent nightly (real game)", "; ".join(ncell(g, a) for g, a in sorted(nt["agents"].items())), "lab/nightly", nt.get("date")))
    w = max((len(x[0]) for x in rows), default=0)
    for a, v, s, dt in rows:
        L.append(f"{a:<{w}}  {v}  [{s}, {dt}]")
    for k2, v in r["notes"].items():
        L.append(f"note: {k2}: {v}")
    L.append("")
    L.append("Top blockers by run impact (measured: runs the Codex walk ended on, of all runs walked):")
    for i, g in enumerate(r["blockers"], 1):
        L.append(f"{i}. {g['cause']}  {g['runs']} runs ({g['pct_of_runs']}%)")
    L.append("Top open lab groups by runs that hold their ingredients (an estimate of exposure; host-suspect and stale left out):")
    for i, g in enumerate(r["lab_blockers"], 1):
        L.append(f"{i}. {g['id']} ~{g['runs_est']} runs, {g['divergences'] or 0} seen {','.join(g['backlog'])} | {g['title']}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--referee", type=Path, default=DEFAULT_REFEREE)
    ap.add_argument("--json", type=Path, help="write the result as JSON here")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--cores", type=int, default=None, help="logical cores the lab's CPU minutes are measured against (default: this machine's)")
    ap.add_argument("--days", type=int, default=UTIL_DAYS, help="days the lab utilisation is measured over (default: 7; 2 for the last 48 hours)")
    a = ap.parse_args(argv)
    r = build(a.referee, a.top, cores=a.cores, days=max(1, a.days))
    if a.json and a.json.exists():  # `nightly` is written by the lab's nightly-real job (tools/lab/jobs.py), not derived: keep it and show it
        with contextlib.suppress(ValueError, OSError):
            old = json.loads(a.json.read_text(encoding="utf-8")).get("nightly")
            if old:
                r["nightly"] = old
    print(render(r))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        if a.json.exists():  # `bench` is written by hand (rl/bench_emu.py), not derived: keep it
            with contextlib.suppress(ValueError, OSError):
                r.setdefault("bench", json.loads(a.json.read_text(encoding="utf-8")).get("bench"))
            if r.get("bench") is None:
                r.pop("bench", None)
        a.json.write_text(json.dumps(r, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
