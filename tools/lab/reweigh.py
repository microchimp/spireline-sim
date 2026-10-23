"""The weekly re-weighting of the fix lab's strategy mix (docs/PLAN.md section 7, rule D1).

A fights strategy (a key of LAB/mix.json) that produced no fix-ready group in the window drops to half its weight. Selfplay
(LAB/jobs.json `selfplay.share`) absorbs the difference, so every unchanged strategy keeps its absolute share of CPU time:

    s' = 1 - (1 - s) * F' / F      F, F': the fights weights before and after; s: the selfplay share; capped at 0.9

The window is the WINDOW_DAYS before --now (default: now): a yield.tsv row counts when  now - 14 days < utc <= now. A strategy
is judged only with at least MIN_JOBS jobs or MIN_CPU_MIN CPU minutes in the window; with less evidence it keeps its weight
("not judged"). A judged strategy whose rows carry no fix_ready figure at all (every value NA) is not judged either: an
unknown is not a zero. The agent jobs (selfplay, train, score-real) are not fights strategies and are never halved.

    python tools/lab/reweigh.py --lab <LAB folder> [--now 2026-10-24T00:00:00Z] [--apply]

The dry run (the default) prints the table and writes nothing. --apply writes LAB/mix.json (its notes kept, a dated `_reweigh`
note added) and LAB/jobs.json's selfplay share, each after a copy to <name>.<date>.bak (a second run on the same day writes
<name>.<date>.2.bak, so an earlier backup is never overwritten). Both writes are atomic (a temp file, then replace). A malformed
yield.tsv row is skipped with a note; a malformed mix.json or jobs.json, or a jobs.json without a selfplay share, stops the run
before anything is written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.lab import mix  # noqa: E402

WINDOW_DAYS = 14
MIN_JOBS = 3
MIN_CPU_MIN = 60.0
SELFPLAY_CAP = 0.9
REQUIRED = ("utc", "strategy", "cpu_min", "fix_ready")


class ConfigError(Exception):
    """A file the rule needs cannot be used; nothing is written."""


# ---------------------------------------------------------------- reading

def parse_utc(text: object) -> dt.datetime | None:
    """An ISO time as an aware UTC datetime (a naive one is taken as UTC); None when it is not a time."""
    s = str(text or "").strip()
    if s[-1:] in ("Z", "z"):
        s = s[:-1] + "+00:00"
    try:
        when = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return when.astimezone(dt.timezone.utc)


def _number(text: str) -> float | None:
    try:
        v = float(text.strip())
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _count(text: str) -> int | None:
    try:
        return int(text.strip())
    except ValueError:
        return None


def _shown(x: float) -> str:
    return f"{x:g}"


def read_json_doc(p: Path, what: str) -> dict | None:
    """The object in a JSON file, read as the runner reads its lab files (BOM and NUL bytes dropped); None when missing."""
    try:
        raw = p.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as e:
        raise ConfigError(f"{what} unreadable ({type(e).__name__})") from None
    try:
        doc = json.loads(raw.decode("utf-8", errors="replace").lstrip("﻿").replace("\x00", ""))
    except ValueError:
        raise ConfigError(f"{what} is not JSON") from None
    if not isinstance(doc, dict):
        raise ConfigError(f"{what} is not an object")
    return doc


def read_rows(text: str | None, start: dt.datetime, now: dt.datetime) -> tuple[list[dict], int, list[str]]:
    """(the valid rows of yield.tsv in the window, the number of rows outside it, a note per malformed row).

    The window is half open: start < utc <= now. A row is malformed when its cell count is not the header's, its utc is not a
    time, its cpu_min is not a number, its fix_ready is neither a whole number nor NA, or it has no strategy."""
    if text is None:
        return [], 0, []
    lines = text.lstrip("﻿").splitlines()
    if not lines:
        return [], 0, []
    head = lines[0].split("\t")
    missing = [c for c in REQUIRED if c not in head]
    if missing:
        raise ConfigError(f"yield.tsv has no {', '.join(missing)} column")
    rows: list[dict] = []
    outside = 0
    notes: list[str] = []
    for n, ln in enumerate(lines[1:], start=2):
        if not ln.strip():
            continue
        cells = ln.split("\t")
        if len(cells) != len(head):
            notes.append(f"yield.tsv line {n} skipped: {len(cells)} cells, the header has {len(head)}")
            continue
        r = dict(zip(head, cells))
        when = parse_utc(r["utc"])
        if when is None:
            notes.append(f"yield.tsv line {n} skipped: utc {r['utc']!r} is not a time")
            continue
        if not (start < when <= now):
            outside += 1
            continue
        strategy = r["strategy"].strip()
        cpu = _number(r["cpu_min"])
        if not strategy:
            notes.append(f"yield.tsv line {n} skipped: no strategy")
            continue
        if cpu is None or cpu < 0:
            notes.append(f"yield.tsv line {n} skipped: cpu_min {r['cpu_min']!r} is not a number")
            continue
        fr_text = r["fix_ready"].strip()
        fix: int | None = None
        if fr_text != "NA":
            fix = _count(fr_text)
            if fix is None or fix < 0:
                notes.append(f"yield.tsv line {n} skipped: fix_ready {r['fix_ready']!r} is not a count")
                continue
        rows.append({"utc": when, "strategy": strategy, "cpu": cpu, "fix_ready": fix, "line": n})
    return rows, outside, notes


def mix_weights(doc: dict) -> tuple[dict[str, float], list[str]]:
    """The usable weights of the fights strategies in mix.json, and a note for each key left as it is."""
    weights: dict[str, float] = {}
    notes: list[str] = []
    for k, v in doc.items():
        if str(k).startswith("_"):
            continue
        if k not in mix.STRATEGIES:
            notes.append(f"mix.json: unknown key {k!r} left as it is (not a fights strategy)")
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v < 0:
            notes.append(f"mix.json: {k} has no usable weight; left as it is")
        else:
            weights[k] = float(v)
    return weights, notes


# ---------------------------------------------------------------- the rule

def new_share(share: float, old_total: float, new_total: float) -> tuple[float, bool]:
    """The selfplay share after the fights weights went from old_total to new_total, and whether the cap bound.

    s' = 1 - (1 - s) * F'/F keeps every unchanged strategy's absolute share of CPU time; capped at SELFPLAY_CAP."""
    if new_total == old_total:
        return share, False
    raw = 1 - (1 - share) * new_total / old_total
    return round(min(SELFPLAY_CAP, max(0.0, raw)), 4), raw > SELFPLAY_CAP


def plan(lab: Path, now: dt.datetime) -> dict:
    """Everything the run would do, read from the lab folder; writes nothing. Raises ConfigError when a file cannot be used."""
    lab = Path(lab)
    notes: list[str] = []
    mdoc = read_json_doc(lab / "mix.json", "mix.json")
    mix_existed = mdoc is not None
    if mdoc is None:
        mdoc = {"_note": mix.NOTE, **mix.DEFAULT_MIX}
        notes.append("mix.json is missing: the default mix is the starting point")
    weights, wnotes = mix_weights(mdoc)
    notes += wnotes
    if sum(weights.values()) <= 0:
        raise ConfigError("mix.json gives no fights strategy a weight")
    absent = [s for s in mix.STRATEGIES if s not in weights]
    if absent:
        notes.append("no usable weight in mix.json (weight 0, not in the table): " + ", ".join(absent))

    start = now - dt.timedelta(days=WINDOW_DAYS)
    try:
        yt = (lab / "yield.tsv").read_text(encoding="utf-8")
    except FileNotFoundError:
        yt = None
        notes.append("yield.tsv is missing: no evidence, every strategy keeps its weight")
    except OSError as e:
        raise ConfigError(f"yield.tsv unreadable ({type(e).__name__})") from None
    rows, outside, rnotes = read_rows(yt, start, now)
    notes += rnotes

    jdoc = read_json_doc(lab / "jobs.json", "jobs.json")
    sp = (jdoc or {}).get("selfplay")
    share: float | None = None
    if isinstance(sp, dict):
        v = sp.get("share")
        if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1:
            share = float(v)
    if share is None:
        notes.append("jobs.json has no usable selfplay share: the difference cannot be placed; --apply refuses")

    table: list[dict] = []
    for s in mix.STRATEGIES:
        if s not in weights:
            continue
        mine = [r for r in rows if r["strategy"] == s]
        n = len(mine)
        cpu = sum(r["cpu"] for r in mine)
        known = [r["fix_ready"] for r in mine if r["fix_ready"] is not None]
        fr = sum(known) if known else None
        old = weights[s]
        if n < MIN_JOBS and cpu < MIN_CPU_MIN:
            new, verdict = old, f"not judged ({n} jobs, {cpu:.1f} CPU min; needs {MIN_JOBS} jobs or {MIN_CPU_MIN:g} CPU min)"
        elif fr is None:
            new, verdict = old, "not judged (no fix_ready figure in the window)"
        elif old == 0:
            new, verdict = old, "kept (weight 0)"
        elif fr == 0:
            new, verdict = old / 2, f"halved (fix_ready 0 in {n} jobs)"
        else:
            new, verdict = old, f"kept (fix_ready {fr} in {n} jobs)"
        table.append({"strategy": s, "jobs": n, "cpu": cpu, "fix_ready": fr, "old": old, "new": new, "verdict": verdict})

    old_total = sum(r["old"] for r in table)
    new_total = sum(r["new"] for r in table)
    share_new, capped = (None, False) if share is None else new_share(share, old_total, new_total)

    stamp = now.strftime("%Y-%m-%d")
    new_mix = dict(mdoc)
    for r in table:
        if r["new"] != r["old"]:
            new_mix[r["strategy"]] = int(r["new"]) if float(r["new"]).is_integer() else round(r["new"], 4)
    summary = "; ".join(f"{r['strategy']}: {r['verdict']}, weight {_shown(r['old'])}->{_shown(r['new'])}" for r in table)
    sp_part = "selfplay share unknown" if share is None else f"selfplay share {share:.4g}->{share_new:.4g}"
    note = f"{stamp} (window {start:%Y-%m-%d} to {now:%Y-%m-%d}, {len(rows)} rows): {summary}; {sp_part}"
    if rnotes:
        note += f"; {len(rnotes)} malformed yield.tsv row(s) skipped"
    prev = mdoc.get("_reweigh")
    new_mix["_reweigh"] = note + (f" | earlier: {prev}" if isinstance(prev, str) and prev else "")

    new_jobs = None
    if jdoc is not None and isinstance(sp, dict) and share_new is not None:
        new_jobs = dict(jdoc)
        new_jobs["selfplay"] = {**sp, "share": share_new}

    return {"now": now, "start": start, "rows": len(rows), "outside": outside, "table": table, "F": old_total, "F2": new_total,
            "share_old": share, "share_new": share_new, "capped": capped, "notes": notes, "stamp": stamp,
            "mix_existed": mix_existed, "new_mix": new_mix, "jobs_existed": jdoc is not None, "new_jobs": new_jobs}


# ---------------------------------------------------------------- output

def render(p: dict) -> str:
    out = [f"window {p['start']:%Y-%m-%d %H:%M}Z to {p['now']:%Y-%m-%d %H:%M}Z ({WINDOW_DAYS} days): "
           f"{p['rows']} rows in the window, {p['outside']} outside it"]
    out.append(f"{'strategy':<14} {'jobs':>5} {'cpu_min':>8} {'fix_ready':>9} {'old':>6} {'new':>6}  verdict")
    for r in p["table"]:
        fr = "NA" if r["fix_ready"] is None else str(r["fix_ready"])
        out.append(f"{r['strategy']:<14} {r['jobs']:>5} {r['cpu']:>8.1f} {fr:>9} {_shown(r['old']):>6} {_shown(r['new']):>6}  {r['verdict']}")
    if p["share_old"] is None:
        out.append("selfplay share: unknown (jobs.json has no usable selfplay.share); --apply refuses")
    else:
        cap = "; cap 0.9 bound" if p["capped"] else ""
        out.append(f"selfplay share: {p['share_old']:.4f} -> {p['share_new']:.4f} "
                   f"(fights weight {_shown(p['F'])} -> {_shown(p['F2'])}{cap})")
    out += [f"note: {n}" for n in p["notes"]]
    return "\n".join(out)


# ---------------------------------------------------------------- writing

def backup(path: Path, stamp: str) -> str:
    """Copy path to <name>.<stamp>.bak, or <name>.<stamp>.2.bak, .3.bak ... when that exists; the backup's file name."""
    n = 1
    while True:
        dst = path.with_name(f"{path.name}.{stamp}.bak" if n == 1 else f"{path.name}.{stamp}.{n}.bak")
        if not dst.exists():
            break
        n += 1
    shutil.copy2(path, dst)
    return dst.name


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def dump(doc: dict) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def apply(lab: Path, p: dict) -> list[str]:
    """Back up and rewrite mix.json and jobs.json from a plan. Returns the messages (file names only)."""
    if p["share_new"] is None or p["new_jobs"] is None:
        raise ConfigError("the selfplay share is not known")
    lab = Path(lab)
    mpath, jpath = lab / "mix.json", lab / "jobs.json"
    msgs: list[str] = []
    if p["mix_existed"]:
        msgs.append(f"backup {backup(mpath, p['stamp'])}")
    msgs.append(f"backup {backup(jpath, p['stamp'])}")
    write_atomic(mpath, dump(p["new_mix"]))
    write_atomic(jpath, dump(p["new_jobs"]))
    msgs.append(f"wrote mix.json and jobs.json (selfplay share {p['share_old']:.4f} -> {p['share_new']:.4f})")
    return msgs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The weekly re-weighting of the fix lab's strategy mix (docs/PLAN.md section 7, D1).")
    ap.add_argument("--lab", required=True, type=Path, help="the lab folder (mix.json, jobs.json, yield.tsv)")
    ap.add_argument("--now", default=None, help="the UTC instant the window ends at, e.g. 2026-10-24T00:00:00Z (default: now)")
    ap.add_argument("--apply", action="store_true", help="write mix.json and jobs.json, after backups (default: a dry run)")
    args = ap.parse_args(argv)
    if args.now is None:
        now = dt.datetime.now(dt.timezone.utc)
    else:
        now = parse_utc(args.now)
        if now is None:
            ap.error(f"--now {args.now!r} is not a time")
    try:
        p = plan(args.lab, now)
        print(render(p))
        if args.apply:
            for m in apply(args.lab, p):
                print(m)
        else:
            print("dry run: nothing written (--apply writes mix.json and jobs.json, after backups)")
    except ConfigError as e:
        print(f"reweigh: {e}; nothing written", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"reweigh: could not write ({type(e).__name__}); the backups made so far are kept", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
