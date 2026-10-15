"""The fix lab's strategy mix and its yield record.

Strategies (what a fights-slot job spends the CPU on), with the default weights of the first mix:

  aimed-bugs    aimed fights at the entities open backlog items name (scenarios.py --mode targets --aim bugs)   30
  aimed-gaps    aimed fights at what real runs meet and little covers, starter items weighed 1/20 (--aim gaps)  25
  choice-edges  small-deck setup fights whose card screens are answered with edge answers (--mode choices)      10
  whole-runs    whole seeded runs (referee/runs.py)                                                              10
  dense         full decks, relics and a full belt (--mode dense)                                                10
  under-tested  the least-covered cards, relics, potions and powers (--mode coverage)                            10
  random        batch.py's random decks, shrunk and traced by --mode analyze                                      5

``LAB/mix.json`` holds the weights, an object of strategy to number (a missing strategy weighs 0 and never runs; keys
starting with "_" are notes). The owner edits it between jobs; it is read at every choice. The weights are shares of
the CPU time the fights slots of the rotation spend: the next job is the available strategy furthest below its
share (least CPU seconds spent per unit of weight; a tie goes to the heavier, then to the name).

``LAB/yield.tsv`` has one row per finished job (tab separated, header first):

  utc  job  strategy  main  rc  wall_min  cpu_min  fights  mismatches  unbuildable  groups  fix_ready  screen_groups

  fights       fights judged (whole runs: runs recorded or not)
  mismatches   fights (runs) that diverge from the game, mirror gaps included
  unbuildable  fights the emulator's setup mirror could not build (a mirror gap) or that produced no comparable result
               (an error), runs with no result
  groups       distinct root-mechanism groups (tools/fidelity/lab_groups.py) with a symptom from this job
  fix_ready    of those, groups that are not host-suspect and have a symptom from this job with a traced first event
               and no mirror gap: the ones a fixer can start from
  screen_groups  of those, groups with a symptom from this job that is a card screen the setup mirror could not answer as
               the game was answered (a count or eligibility difference, what choice-edges looks for). Never fix-ready
               (the setup has a mirror gap), but a precise finding: the screen, the relic or card that opened it.

NA where a column cannot be known (the job's groups step did not finish; an older lab_groups without by_job).
"""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Callable
from pathlib import Path

STRATEGIES = ("aimed-bugs", "aimed-gaps", "choice-edges", "whole-runs", "dense", "under-tested", "random")
DEFAULT_MIX = {"aimed-bugs": 30, "aimed-gaps": 25, "choice-edges": 10, "whole-runs": 10, "dense": 10, "under-tested": 10, "random": 5}
YIELD_COLUMNS = ("utc", "job", "strategy", "main", "rc", "wall_min", "cpu_min", "fights", "mismatches", "unbuildable", "groups", "fix_ready",
                 "screen_groups")
NOTE = "weights of the CPU time the fights slots spend, per strategy; 0 or absent: never. Read at every job."


def default_text() -> str:
    return json.dumps({"_note": NOTE, **DEFAULT_MIX}, indent=1) + "\n"


def parse_mix(text: str) -> tuple[dict[str, float], list[str]]:
    """(weights, notes) from the text of mix.json; the defaults and a note when it cannot be used."""
    notes: list[str] = []
    try:
        doc = json.loads(text.lstrip("﻿").replace("\x00", ""))
    except ValueError as e:
        return dict(DEFAULT_MIX), [f"mix.json is not JSON ({e}); using the default mix"]
    if not isinstance(doc, dict):
        return dict(DEFAULT_MIX), ["mix.json is not an object; using the default mix"]
    weights: dict[str, float] = {}
    for k, v in doc.items():
        if str(k).startswith("_"):
            continue
        if k not in STRATEGIES:
            notes.append(f"mix.json: unknown strategy {k!r} ignored")
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v < 0:
            notes.append(f"mix.json: weight of {k} is not a number of at least 0; {k} will not run")
        else:
            weights[k] = float(v)
    if not any(w > 0 for w in weights.values()):
        return dict(DEFAULT_MIX), notes + ["mix.json gives no strategy a weight; using the default mix"]
    return weights, notes


def load_mix(lab: Path, create: bool = True) -> tuple[dict[str, float], list[str]]:
    """The mix from LAB/mix.json; the file is written with the defaults when it is missing (the owner edits it), unless
    `create` is off (a dry run changes nothing)."""
    p = Path(lab) / "mix.json"
    try:
        text = p.read_bytes().decode("utf-8", errors="replace")
    except FileNotFoundError:
        if create:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(default_text(), encoding="utf-8")
            except OSError:
                pass
        return dict(DEFAULT_MIX), []
    except OSError as e:
        return dict(DEFAULT_MIX), [f"mix.json unreadable ({type(e).__name__}); using the default mix"]
    return parse_mix(text)


def pick(weights: dict[str, float], spent: dict[str, float], unavailable: Callable[[str], str | None]
         ) -> tuple[str | None, dict[str, str]]:
    """The strategy to run next: of those with weight whose `unavailable(strategy)` is None, the one with the least CPU
    seconds spent per unit of weight (ties: heavier, then name). Also {strategy: why not} for the others."""
    skipped: dict[str, str] = {}
    best: tuple[float, float, str] | None = None
    for s in STRATEGIES:
        w = weights.get(s, 0.0)
        if w <= 0:
            continue
        why = unavailable(s)
        if why is not None:
            skipped[s] = why
            continue
        key = (float(spent.get(s, 0.0)) / w, -w, s)
        if best is None or key < best:
            best = key
    return (best[2] if best else None), skipped


def share_table(weights: dict[str, float], spent: dict[str, float]) -> str:
    """One line for the log: each strategy's wanted share and the share of CPU it has had."""
    total_w = sum(w for w in weights.values() if w > 0) or 1.0
    total_s = sum(spent.get(s, 0.0) for s in STRATEGIES) or 1.0
    return ", ".join(f"{s} {100 * weights.get(s, 0) / total_w:.0f}%/{100 * spent.get(s, 0) / total_s:.0f}%" for s in STRATEGIES if weights.get(s, 0) > 0)


# ---------------------------------------------------------------- the yield of one job

def _read(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def job_counts(job_dir: Path) -> dict[str, int]:
    """fights, mismatches and unbuildable from the job's verdict files (fights jobs) and runs.json files (whole runs)."""
    fights = mismatches = unbuildable = 0
    for vf in sorted(Path(job_dir).rglob("*.verdicts.json")):
        verdicts = _read(vf)
        if not isinstance(verdicts, list):
            continue
        for v in verdicts:
            if not isinstance(v, dict):
                continue
            fights += 1
            if v.get("error"):
                unbuildable += 1
            else:
                mismatches += 0 if v.get("identical") else 1
                unbuildable += 1 if v.get("mirror_gap") else 0
    for rf in sorted(Path(job_dir).rglob("*.runs.json")):
        doc = _read(rf)
        for r in (doc or {}).get("runs", []) if isinstance(doc, dict) else []:
            if not isinstance(r, dict):
                continue
            fights += 1
            if not r.get("recorded"):
                unbuildable += 1
            elif r.get("first"):
                mismatches += 1
    return {"fights": fights, "mismatches": mismatches, "unbuildable": unbuildable}


def job_groups(groups_doc: object, job_name: str) -> tuple[int, int, int] | None:
    """(groups with a symptom from the job, of those the fix-ready ones, of those the card-screen ones) from groups.json;
    None when it cannot say."""
    if not isinstance(groups_doc, dict) or not isinstance(groups_doc.get("groups"), list):
        return None
    if groups_doc["groups"] and not any("by_job" in g for g in groups_doc["groups"] if isinstance(g, dict)):
        return None  # written by a lab_groups that does not count per job
    n = ready = screens = 0
    for g in groups_doc["groups"]:
        mine = (g.get("by_job") or {}).get(job_name) if isinstance(g, dict) else None
        if mine:
            n += 1
            ready += 1 if not g.get("host_suspect") and mine.get("ready", 0) > 0 else 0
            screens += 1 if mine.get("gap", 0) > 0 else 0
    return n, ready, screens


def yield_row(job_dir: Path, data: dict, groups_doc: object, groups_ran: bool, utc: str) -> dict[str, object]:
    """The yield.tsv row of a finished job (job.json's `data`)."""
    counts = job_counts(job_dir)
    g = job_groups(groups_doc, data.get("name", "")) if groups_ran else None
    params = data.get("params") or {}
    return {"utc": utc, "job": data.get("name", ""), "strategy": params.get("strategy") or data.get("job", ""),
            "main": str(data.get("main_sha", ""))[:7], "rc": data.get("rc", ""),
            "wall_min": round(float(data.get("wall_secs", 0)) / 60, 1), "cpu_min": round(float(data.get("cpu_secs", 0)) / 60, 1),
            **counts, "groups": g[0] if g else "NA", "fix_ready": g[1] if g else "NA", "screen_groups": g[2] if g else "NA"}


def append_yield(lab: Path, row: dict[str, object]) -> None:
    """Add a row to LAB/yield.tsv (the header first when the file is new). A file written with fewer columns (the same
    ones, fewer at the end) is rewritten with the new header and NA in the old rows; one with other columns is kept
    aside as yield.tsv.old and a new file is started."""
    p = Path(lab) / "yield.tsv"
    head = "\t".join(YIELD_COLUMNS)
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    if lines and lines[0] != head:
        old = lines[0].split("\t")
        if list(YIELD_COLUMNS[: len(old)]) == old:
            pad = "\t".join(["NA"] * (len(YIELD_COLUMNS) - len(old)))
            with open(p, "w", encoding="utf-8", newline="") as fh:
                fh.write("\n".join([head] + [ln + "\t" + pad for ln in lines[1:] if ln.strip()]) + "\n")
        else:
            p.replace(p.with_name("yield.tsv.old"))
            lines = []
    with open(p, "a", encoding="utf-8", newline="") as fh:
        if not lines:
            fh.write(head + "\n")
        fh.write("\t".join(str(row.get(c, "")) for c in YIELD_COLUMNS) + "\n")


def read_yield(lab: Path) -> list[dict[str, str]]:
    p = Path(lab) / "yield.tsv"
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    cols = lines[0].split("\t") if lines else []
    return [dict(zip(cols, ln.split("\t"))) for ln in lines[1:] if ln.strip()]


def now_utc() -> str:
    return dt.datetime.fromtimestamp(time.time(), dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
