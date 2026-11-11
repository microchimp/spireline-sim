"""The run-level human bar (D56, D58 C5): floor reached and HP at the Act 1 and Act 2 boss doors, the humans' against the copy's.

    python rl/run_human_bar.py compute [--out D:/spireline/referee/human-bar/run-level.json] [--limit N] [--force]
    python rl/run_human_bar.py beside --records ROOT [--arms DIR ...] [--max-runs 400]

`compute` reads the human records once (the four fifths that are not the exam fifth: `human_starts.exam_bucket` 1 to 4; the ids file and the
hash rule both keep the exam fifth out of the read, and `assert_no_exam` stops a figure that would include one) and stores the aggregates:
floor reached (map points entered: the floor a run died on, or won through), and for each of the Acts 1 and 2 how many runs reached the boss,
how many killed it, and HP at the door (the HP the run carried into the boss fight: the last point before it, the rest site's own figure), in
HP and as a share of max HP. It stores aggregates only: no run, no seed, no run id; a cell over fewer than 20 runs is hidden (None). It never
overwrites its output unless `--force` (it is computed once).

`beside` puts the copy's figures next to them: from the recorded runs of the copy (RunRecords: the teacher job's runs and the lab's self-play
shards, `copy_from_records`), which carry the door HP, and from a judgement arm's `runs.jsonl` rows (`copy_from_arms`), which carry floor and
whether each boss was reached and killed, but not the HP. The table is the "beside every verdict" of the memo (section 4.4 item 4): the lab's
student job writes it under `LAB/student/verdicts.md` after a round and after the judgement.

Source of the human figures: Spire Codex public run export for v0.107.1. Nothing from the corpus folder is copied anywhere else.
Module level imports are the standard library only, so the lab's runner can import the pure functions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPIRE_CODEX = "Spire Codex public run export for v0.107.1"
MIN_CELL = 20  # D13: a cell over fewer runs is not shown
VERSION = 1
FIFTHS = (1, 2, 3, 4)  # the corpus's fifths that are not the exam fifth (bucket 0)
ACTS = (1, 2)
BOSS_FLOOR = {1: 17, 2: 33}  # the boss floors of Acts 1 and 2 on the copy (rl/run_fields.ACT_BOSS_FLOORS)
BOSS_NODE = 6  # run_train.vocab.NODE_BOSS: a RunRecord fight's node_type
OUT = Path("D:/spireline/referee/human-bar/run-level.json")


# ---------------------------------------------------------------------------------------------------------------------- one run


def _quantile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(len(s) - 1, lo + 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def human_run(r: Mapping[str, Any]) -> dict[str, Any] | None:
    """One human run's figures: `floor` (map points entered), `win`, and for each of Acts 1 and 2 present in the record its boss door:
    `{"hp", "max_hp", "killed"}`. The HP carried into the boss fight is the last point before the boss, which for the record is the rest
    site's own `current_hp`. None when the record has no map history. The record's player id is never read."""
    mph = r.get("map_point_history")
    if not isinstance(mph, list) or not mph:
        return None
    floor = sum(len(a) for a in mph if isinstance(a, list))
    win = bool(r.get("win"))
    doors: dict[int, dict[str, Any]] = {}
    for a, act in enumerate(mph[:2], start=1):
        if not isinstance(act, list) or len(act) < 2 or act[-1].get("map_point_type") != "boss":
            continue
        prev = (act[-2].get("player_stats") or [{}])[0]
        hp, mx = prev.get("current_hp"), prev.get("max_hp")
        if not isinstance(hp, int) or not isinstance(mx, int) or mx <= 0:
            continue
        doors[a] = {"hp": hp, "max_hp": mx, "killed": a < len(mph) and bool(mph[a])}  # the next act has points: the boss died
    return {"floor": floor, "win": win, "doors": doors}


# ---------------------------------------------------------------------------------------------------------------------- aggregates


def _door_cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """HP at the door over runs that had one; None under MIN_CELL runs."""
    if len(rows) < MIN_CELL:
        return None
    hp = [float(x["hp"]) for x in rows]
    frac = [float(x["hp"]) / float(x["max_hp"]) for x in rows if x.get("max_hp")]
    return {"n": len(rows), "mean": round(sum(hp) / len(hp), 2), "p25": round(_quantile(hp, 0.25), 1), "p50": round(_quantile(hp, 0.5), 1),
            "p75": round(_quantile(hp, 0.75), 1), "share_of_max": round(sum(frac) / len(frac), 4) if frac else None}


def summarise(runs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The aggregate of `human_run`-shaped dicts (a copy run has the same keys; its `doors` may lack `hp`). Cells over fewer than MIN_CELL
    runs are None and named in `hidden`."""
    runs = list(runs)
    n = len(runs)
    hidden: list[str] = []
    out: dict[str, Any] = {"runs": n if n >= MIN_CELL else None}
    if n < MIN_CELL:
        hidden.append("runs")
        out.update(floor=None, win_rate=None, acts={str(a): None for a in ACTS}, hidden=hidden)
        return out
    floors = [float(r["floor"]) for r in runs]
    out["floor"] = {"mean": round(sum(floors) / n, 3), "p10": round(_quantile(floors, 0.1), 1), "p50": round(_quantile(floors, 0.5), 1),
                    "p90": round(_quantile(floors, 0.9), 1)}
    out["win_rate"] = round(sum(1 for r in runs if r.get("win")) / n, 4)
    acts: dict[str, Any] = {}
    for a in ACTS:
        reached = [r["doors"][a] for r in runs if a in (r.get("doors") or {})]
        if len(reached) < MIN_CELL:
            acts[str(a)] = None
            hidden.append(f"act{a}")
            continue
        killed = sum(1 for d in reached if d.get("killed"))
        with_hp = [d for d in reached if isinstance(d.get("hp"), (int, float))]
        cell: dict[str, Any] = {"reached": len(reached), "reached_share": round(len(reached) / n, 4), "killed": killed,
                                "kill_rate": round(killed / len(reached), 4), "door_hp": _door_cell(with_hp)}
        if with_hp and cell["door_hp"] is None:
            hidden.append(f"act{a}.door_hp")
        acts[str(a)] = cell
    out["acts"] = acts
    out["hidden"] = hidden
    return out


def human_figures(corpus: Path | str | None = None, exam_ids: Iterable[str] | None = None, *, limit: int = 0,
                  fifths: Sequence[int] = FIFTHS) -> dict[str, Any]:
    """The human records' run-level figures, from the fifths in `fifths` (never 0, the exam fifth). One pass over the corpus."""
    for p in (ROOT / "rl", ROOT / "src", ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import human_prior as hp  # noqa: PLC0415
    import human_starts as hs  # noqa: PLC0415

    if hs.EXAM_BUCKET in fifths:
        raise ValueError("the exam fifth is never read")
    corpus = Path(corpus) if corpus is not None else hs.CORPUS
    exam = frozenset(exam_ids) if exam_ids is not None else hs.read_exam_ids()
    stats: Counter = Counter()
    rows: list[dict[str, Any]] = []
    for line in hp.iter_run_lines(corpus, exam, stats):  # the exam fifth's lines are dropped here, before this code sees a record
        r = json.loads(line)
        rid = hs.run_id(r)
        if hs.exam_bucket(rid) not in fifths:
            stats["other_fifth_left_out"] += 1
            continue
        hs.assert_no_exam([rid], exam)
        row = human_run(r)
        if row is None:
            stats["no_map_history"] += 1
            continue
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    doc = summarise(rows)
    keep = ("lines", "ironclad_a10", "runs_usable", "exam_fifth_left_out", "modded_acts", "other_fifth_left_out", "no_map_history")
    doc["read"] = {k: int(stats[k]) for k in keep if k in stats}
    return doc


def write_human(path: Path | str, doc: Mapping[str, Any], *, force: bool = False) -> Path:
    """The stored figures (aggregates only; atomic write). Refuses to replace an existing file without `force`: it is computed once."""
    p = Path(path)
    if p.exists() and not force:
        raise FileExistsError(f"{p.name} exists: the human figures are computed once (--force to redo)")
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {"version": VERSION, "source": SPIRE_CODEX, "written": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "character": "Ironclad", "ascension": 10, "fifths": list(FIFTHS), **doc,
            "notes": {"floor": "map points entered (the floor a run died on, or won through)",
                      "door_hp": "the HP carried into the boss fight (the last point before the boss)",
                      "kill_rate": "of the runs that reached that boss"}}
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def read_human(path: Path | str = OUT) -> dict[str, Any] | None:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


# ---------------------------------------------------------------------------------------------------------------------- the copy's runs


def copy_run(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """One RunRecord (the lab's run shard, `run_train.shards`) as the same figures: floor from the outcome, a door per boss fight of the
    Act 1 and Act 2 boss floors with the HP the fight began with (`hp_before`) and whether it was won."""
    out = record.get("outcome")
    if not isinstance(out, dict) or not isinstance(out.get("final_floor"), int):
        return None
    doors: dict[int, dict[str, Any]] = {}
    for f in record.get("fights") or []:
        if f.get("node_type") != BOSS_NODE:
            continue
        for a, fl in BOSS_FLOOR.items():
            if f.get("floor") == fl and isinstance(f.get("hp_before"), int):
                mx = (f.get("state") or {}).get("max_hp")
                doors[a] = {"hp": f["hp_before"], "max_hp": mx if isinstance(mx, int) and mx > 0 else None, "killed": bool(f.get("won"))}
    return {"floor": int(out["final_floor"]), "win": bool(out.get("won")), "doors": doors}


def copy_from_records(shard_dirs: Iterable[Path | str], *, max_runs: int = 400) -> dict[str, Any]:
    """The copy's figures from run shards (`r*.jsonl.gz`) under the given folders, newest folders first, at most `max_runs` runs."""
    rows: list[dict[str, Any]] = []
    dirs = sorted({Path(d) for d in shard_dirs}, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for d in dirs:
        for p in sorted(d.glob("r*.jsonl.gz"), reverse=True):
            try:
                text = gzip.decompress(p.read_bytes()).decode("utf-8")
            except (OSError, EOFError, ValueError):
                continue
            for line in text.splitlines():
                if not line:
                    continue
                try:
                    row = copy_run(json.loads(line))
                except ValueError:
                    continue
                if row is not None:
                    rows.append(row)
                if len(rows) >= max_runs:
                    return {**summarise(rows), "source": "recorded runs"}
    return {**summarise(rows), "source": "recorded runs"}


def copy_from_arms(runs_jsonl: Iterable[Path | str]) -> dict[str, Any]:
    """The copy's figures from a judgement arm's `runs.jsonl` rows: floor, and for each boss reached (`boss_damage_acts`, 0-based act) whether
    it died (fraction 1.0). The rows do not carry HP, so the door's HP is absent (None) here."""
    rows: list[dict[str, Any]] = []
    for path in runs_jsonl:
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not isinstance(r, dict) or not isinstance(r.get("floor"), int) or r.get("crash"):
                continue
            doors: dict[int, dict[str, Any]] = {}
            for ent in r.get("boss_damage_acts") or []:
                if isinstance(ent, (list, tuple)) and len(ent) == 2 and ent[0] in (0, 1):
                    doors[int(ent[0]) + 1] = {"hp": None, "max_hp": None, "killed": float(ent[1]) >= 1.0}
            rows.append({"floor": int(r["floor"]), "win": bool(r.get("win")), "doors": doors})
    return {**summarise(rows), "source": "arm rows (no HP)"}


# ---------------------------------------------------------------------------------------------------------------------- the table


def _f(x: Any, fmt: str = "{:.1f}") -> str:
    return "-" if x is None else fmt.format(x)


def _pct(x: Any) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def beside_text(human: Mapping[str, Any] | None, copy: Mapping[str, Any] | None, label: str = "copy") -> str:
    """The run-level bar as a small table: the human records' figures and the copy's, floor reached, then Act 1 and Act 2. A cell under 20
    runs shows '-'. Without the stored human figures it says how to compute them."""
    if not human:
        return ("run-level human bar: the human figures are not computed yet (python rl/run_human_bar.py compute); "
                f"{label}: " + (f"{(copy or {}).get('runs') or 0} runs" if copy else "no runs") + "\n")
    c = copy or {}
    lines = [f"run-level human bar ({SPIRE_CODEX}; fifths 1 to 4, Ironclad A10; {label}: {c.get('source', 'no runs')}, {c.get('runs') or 0} runs)",
             "", f"| figure | humans | {label} |", "|---|---|---|"]
    hf, cf = human.get("floor") or {}, c.get("floor") or {}
    lines.append(f"| mean floor reached | {_f(hf.get('mean'))} | {_f(cf.get('mean'))} |")
    lines.append(f"| median floor reached | {_f(hf.get('p50'))} | {_f(cf.get('p50'))} |")
    for a in ACTS:
        ha, ca = (human.get("acts") or {}).get(str(a)) or {}, (c.get("acts") or {}).get(str(a)) or {}
        hd, cd = ha.get("door_hp") or {}, ca.get("door_hp") or {}
        lines.append(f"| Act {a} boss reached | {_pct(ha.get('reached_share'))} | {_pct(ca.get('reached_share'))} |")
        lines.append(f"| Act {a} boss killed, of those reaching it | {_pct(ha.get('kill_rate'))} | {_pct(ca.get('kill_rate'))} |")
        lines.append(f"| Act {a} HP at the door | {_f(hd.get('mean'))} ({_pct(hd.get('share_of_max'))} of max) | "
                     f"{_f(cd.get('mean'))} ({_pct(cd.get('share_of_max'))} of max) |")
    lines += ["", "A cell over fewer than 20 runs is not shown (-). Humans: aggregates of the training fifths, computed once; "
                  "the exam fifth is never read."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------- the command line


def _below_normal() -> None:
    try:
        import ctypes  # noqa: PLC0415

        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)
    except (AttributeError, OSError):
        pass


def cmd_compute(a: argparse.Namespace) -> int:
    _below_normal()
    out = Path(a.out)
    if out.exists() and not a.force:
        print(f"{out.name} exists: the human figures are computed once (--force to redo)")
        return 0
    doc = human_figures(limit=a.limit)
    write_human(out, doc, force=a.force)
    print(json.dumps({k: doc[k] for k in ("runs", "floor", "win_rate", "hidden", "read") if k in doc}, indent=1))
    return 0


def cmd_beside(a: argparse.Namespace) -> int:
    root = Path(a.records)
    dirs = list(root.glob("*/jobs/*/shards")) + list(root.glob("*/*/jobs/*/shards")) + list(root.glob("jobs/*/shards"))
    copy = copy_from_records(dirs, max_runs=a.max_runs) if dirs else None
    if a.arms:
        arms = copy_from_arms(Path(p) for p in a.arms)
        copy = arms if copy is None or not copy.get("runs") else copy
    print(beside_text(read_human(a.human), copy))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute", help="the human figures, once")
    c.add_argument("--out", default=str(OUT))
    c.add_argument("--limit", type=int, default=0, help="stop after this many runs (a trial)")
    c.add_argument("--force", action="store_true")
    c.set_defaults(fn=cmd_compute)
    b = sub.add_parser("beside", help="print the table of the human figures and the copy's")
    b.add_argument("--human", default=str(OUT))
    b.add_argument("--records", required=True, help="the teacher store root (its jobs/<job>/shards hold the copy's RunRecords)")
    b.add_argument("--arms", nargs="*", help="judgement arms' runs.jsonl (floor and boss rows only; used when there are no recorded runs)")
    b.add_argument("--max-runs", type=int, default=400)
    b.set_defaults(fn=cmd_beside)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
