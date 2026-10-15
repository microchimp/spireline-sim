"""Disk pruning for the fix lab: what may go, in what order, and what never goes.

``prune(lab, need, busy)`` frees at least `need` bytes (or everything it may) from finished jobs, oldest first,
tier by tier, and returns the bytes freed. A job that is running (`busy`) is never touched.

  tier A  scratch: serve/, chunks/, work/, scratch/ folders, shrink replays (<name>.shrinkN.json, .shrinkN.out.json), a whole-runs
          session's copy of its runs (<name>.pN.out.json), split inputs, logs over 20 MB cut to their tail.
  tier B  raw outputs of fights that MATCHED: in `<name>.out.json`, `.emu.json` and `.reanchor.json` the entry of a fight
          whose verdict is identical (and has no mirror gap) becomes the stub {"seed", "pruned": true, "steps": []}, so
          every list stays aligned with `<name>.verdicts.json`, which is never touched.
  tier C  raw outputs of whole jobs older than `old_days` once a later groups.json has read them: the .out/.emu/.reanchor
          files, specs, codex walk shards, whatever tier B left.

Never removed, at any tier: minimal packs (`*.packs/minimal/`), groups.json, summary.md, targets.json, state.json,
status.txt, job.json, done.rc, verdicts, summaries, logs' heads, and the fix kits (they live outside the lab).
Every removal goes through ``protected`` first.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

SCRATCH_DIRS = {"serve", "chunks", "work", "scratch", "wt"}
SCRATCH_FILE_PATTERNS = (".shrink.emu.json",)
# a shrink round's job and result files (<name>.shrink3.json, .shrink3.out.json) and a whole-runs session's copy of what
# runs.py merged into <name>.out.json (<name>.p0.json, .p0.out.json)
SCRATCH_RE = re.compile(r"\.shrink\d+(\.out)?\.json$|\.p\d+(\.out)?\.json$")
KEEP_NAMES = {"groups.json", "summary.md", "targets.json", "state.json", "status.txt", "job.json", "done.rc", "pruned.json",
              "coverage-setup.json", "yield.tsv", "mix.json", "coverage-random.json", "PAUSE", "HALT"}
KEEP_SUFFIXES = (".verdicts.json", ".summary.txt", ".runs.json", ".shrink.json")
RAW_SUFFIXES = (".out.json", ".emu.json", ".reanchor.json")
LOG_CUT = 20 * 1024 * 1024
LOG_TAIL = 256 * 1024


def protected(p: Path) -> bool:
    """True for anything pruning must never remove."""
    parts = {x.lower() for x in p.parts}
    if "minimal" in parts or "kits" in parts or "packets" in parts:
        return True
    n = p.name.lower()
    return n in KEEP_NAMES or n.endswith(KEEP_SUFFIXES) or n.startswith("minimal")


def finished_jobs(lab: Path) -> list[Path]:
    """Job folders with a done.rc, oldest first (the folder name starts with its UTC time)."""
    runs = lab / "runs"
    if not runs.is_dir():
        return []
    return sorted(d for d in runs.iterdir() if d.is_dir() and (d / "done.rc").exists())


def _size(p: Path) -> int:
    try:
        if p.is_file():
            return p.stat().st_size
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except OSError:
        return 0


def _remove(p: Path) -> int:
    if protected(p):
        return 0
    n = _size(p)
    try:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    except OSError:
        return 0
    return n


def tier_a(job: Path) -> int:
    freed = 0
    for d in sorted((x for x in job.rglob("*") if x.is_dir() and x.name in SCRATCH_DIRS), key=lambda x: -len(x.parts)):
        if d.exists():
            freed += _remove(d)
    for f in list(job.rglob("*")):
        if f.is_file() and not protected(f):
            if f.name.endswith(SCRATCH_FILE_PATTERNS) or SCRATCH_RE.search(f.name) or (f.name.endswith(".in.json") and ".p" in f.name):
                freed += _remove(f)
                continue
            try:  # a file another process holds or just removed is left as it is
                if f.suffix == ".log" and f.stat().st_size > LOG_CUT:
                    before = f.stat().st_size
                    with open(f, "rb") as fh:
                        fh.seek(-LOG_TAIL, os.SEEK_END)
                        tail = fh.read()
                    f.write_bytes(b"[cut by the lab's disk pruning; tail follows]\n" + tail)
                    freed += before - f.stat().st_size
            except OSError:
                continue
    return freed


def _matched(v: dict) -> bool:
    return v.get("identical") is True and not v.get("mirror_gap") and not v.get("setup_first") and not v.get("error")


def tier_b(job: Path) -> int:
    """Stub the raw entries of matched fights; one pruned.json note per job."""
    freed, notes = 0, []
    for vf in sorted(job.rglob("*.verdicts.json")):
        base = vf.name[: -len(".verdicts.json")]
        try:
            verdicts = json.loads(vf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(verdicts, list):
            continue
        drop = {i for i, v in enumerate(verdicts) if isinstance(v, dict) and _matched(v)}
        if not drop:
            continue
        for suffix in RAW_SUFFIXES:
            f = vf.with_name(base + suffix)
            if not f.exists():
                continue
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
                res = doc.get("results")
            except (OSError, ValueError, AttributeError):
                continue
            if not isinstance(res, list) or len(res) != len(verdicts):
                continue  # not aligned with the verdicts: leave it alone
            for i in drop:
                if not (isinstance(res[i], dict) and res[i].get("pruned")):
                    res[i] = {"seed": (res[i] or {}).get("seed") if isinstance(res[i], dict) else None, "pruned": True, "steps": []}
            tmp = f.with_name(f.name + ".tmp")
            try:
                before = f.stat().st_size
                tmp.write_text(json.dumps(doc), encoding="utf-8")
                os.replace(tmp, f)
                gained = before - f.stat().st_size
            except OSError:  # disk full, a lock, a vanished file: this file is left as it was, the rest go on
                try:
                    tmp.unlink()
                except OSError:
                    pass
                continue
            freed += max(0, gained)
            notes.append({"file": f.relative_to(job).as_posix(), "tier": "B", "stubbed": len(drop), "bytes_freed": gained})
    if notes:
        _note(job, notes)
    return freed


def tier_c(job: Path) -> int:
    freed, notes = 0, []
    for f in sorted(job.rglob("*")):
        if not f.is_file() or protected(f) or "logs" in f.parts:
            continue
        raw = f.name.endswith(RAW_SUFFIXES) or (f.suffix == ".jsonl" and "codex" in f.parts) or \
            (f.suffix == ".json" and f.name.endswith(".json") and f.parent.name == "fights" and f.name.count(".") == 1)
        if raw:
            n = _remove(f)
            if n:
                freed += n
                notes.append({"file": f.relative_to(job).as_posix(), "tier": "C", "bytes_freed": n})
    if notes:
        _note(job, notes)
    return freed


def _note(job: Path, notes: list[dict]) -> None:
    p = job / "pruned.json"
    try:
        old = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = []
    try:
        p.write_text(json.dumps(old + notes, indent=1), encoding="utf-8")
    except OSError:
        pass  # the note is a convenience; the pruning it describes has happened


def prune(lab: Path, need: int, busy: set[str] | None = None, old_days: float = 7.0, now: float | None = None) -> int:
    """Free about `need` bytes from finished jobs, oldest first, cheapest tier first; the bytes freed."""
    busy = busy or set()
    now = time.time() if now is None else now
    jobs = [j for j in finished_jobs(lab) if j.name not in busy]
    freed = 0
    for tier in (tier_a, tier_b):
        for j in jobs:
            freed += tier(j)
            if freed >= need:
                return freed
    groups = lab / "groups.json"
    g_time = groups.stat().st_mtime if groups.exists() else None
    for j in jobs:
        done = (j / "done.rc").stat().st_mtime
        if g_time is not None and g_time > done and now - done > old_days * 86400:
            freed += tier_c(j)
            if freed >= need:
                break
    return freed
