#!/usr/bin/env python3
"""The presort's codex walk, reused when it cannot have changed.

    python tools/lab/walk_reuse.py <source codex folder> <job codex folder>      (the copy the presort runs)

The walk is four EmuDump processes of about 22 minutes each (90 of a presort's 112 CPU minutes). Its result is a
function of the EmuDump build, the codex file and the decompile it reads, and of nothing else on the tree. The
build is a function of the git trees in WALK_TREES, so a main move that leaves those trees alone (a lab-code or
docs landing, a change to the C# tests) gives the same walk. ``find_source`` finds an earlier presort whose walk
finished and whose trees equal the new sha's; ``copy_shards`` puts its shard files into the new job's folder
(modification times kept), so the job holds the walk's files exactly as if it had run, groups reads them as before,
and the pruning of the older folder cannot take them away. The refset score and the corpus check are not reused:
they take under two minutes together and read data that moves on its own (new records).

Standard library only: the runner starts this file from its own code folder, not from the tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# What EmuDump.dll is built from; tools/fidelity/refset_score.py keeps the same list (EMU_SOURCES) for its own
# codex reuse, and tests/python/test_lab_runner.py fails if the two drift.
WALK_TREES = ("src/Sts2Emulator", "src/Sts2Emulator.Tests/Support/Invariants", "tools/emudump")
SHARDS = 4
JOB_NAME = re.compile(r"^\d{8}T\d{4}-presort$")
SCAN_LIMIT = 80  # presort folders looked at, newest first
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


_TREES: dict[tuple[str, str], tuple[str, ...]] = {}  # (fork, sha) -> tree ids: a commit's trees never change; a failure is not kept


def trees(fork: str, sha: str) -> tuple[str, ...] | None:
    """The git tree ids of WALK_TREES at a commit of the fork; None when the commit or a path is not there."""
    if (fork, sha) in _TREES:
        return _TREES[(fork, sha)]
    out = []
    for p in WALK_TREES:
        try:
            r = subprocess.run(["git", "-C", fork, "rev-parse", "--verify", "-q", f"{sha}:{p}"], capture_output=True, text=True,
                               check=False, creationflags=CREATE_NO_WINDOW)
        except OSError:
            return None
        t = r.stdout.strip()
        if r.returncode != 0 or not t:
            return None
        out.append(t)
    _TREES[(fork, sha)] = tuple(out)
    return _TREES[(fork, sha)]


def data_floor(*paths: Path) -> float:
    """The newest modification time of the walk's data: a file, or a folder and its direct entries. A walk older than
    this may have read other data. Inf when one of them cannot be read (then nothing is reused)."""
    newest = 0.0
    for p in paths:
        try:
            newest = max(newest, p.stat().st_mtime)
            if p.is_dir():
                with os.scandir(p) as it:
                    for e in it:
                        newest = max(newest, e.stat().st_mtime)
        except OSError:
            return float("inf")
    return newest


def argv_sig(argvs: list[str]) -> str:
    return hashlib.sha1("\n".join(argvs).encode("utf-8")).hexdigest()[:16]


def _done(rec: object) -> bool:
    return isinstance(rec, dict) and rec.get("status") == "done" and rec.get("rc") == 0


def job_walk_sig(doc: dict) -> str | None:
    """The signature of the walk commands a finished job ran: the one its params name (a job that reused a walk), else
    the one of the argvs its four shard commands recorded."""
    sig = (doc.get("params") or {}).get("walk_sig")
    if sig:
        return str(sig)
    for s in doc.get("steps") or []:
        if s.get("name") == "checks":
            cmds = s.get("cmds") or {}
            argvs = [(cmds.get(f"codex-s{i}") or {}).get("argv") for i in range(SHARDS)]
            if all(argvs):
                return argv_sig(argvs)
    return None


def has_walk(job: Path, doc: dict, floor: float) -> bool:
    """The job holds a finished walk: its four shard commands (or its reuse command) done with rc 0, every shard file
    there, none older than the data floor."""
    cmds: dict = {}
    for s in doc.get("steps") or []:
        if s.get("name") == "checks":
            cmds = s.get("cmds") or {}
    if not (all(_done(cmds.get(f"codex-s{i}")) for i in range(SHARDS)) or _done(cmds.get("codex-reuse"))):
        return False
    try:
        return all((job / "codex" / f"s{i}.jsonl").stat().st_mtime >= floor and (job / "codex" / f"s{i}.jsonl").stat().st_size > 0
                   for i in range(SHARDS))
    except OSError:
        return False


def find_source(runs: Path, fork: Path, sha: str, sig: str, floor: float) -> str | None:
    """The folder name of the newest presort whose walk can stand for the one at `sha`: same EmuDump trees, same walk
    commands, finished after the data last changed. None when there is none (the walk then runs)."""
    want = trees(str(fork), sha)
    if want is None or floor == float("inf") or not runs.is_dir():
        return None
    names = sorted((d.name for d in runs.iterdir() if d.is_dir() and JOB_NAME.match(d.name)), reverse=True)[:SCAN_LIMIT]
    for name in names:
        try:
            doc = json.loads((runs / name / "job.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        s = doc.get("main_sha")
        if not s or trees(str(fork), s) != want or job_walk_sig(doc) != sig:
            continue
        if has_walk(runs / name, doc, floor):
            return name
    return None


def copy_shards(src: Path, dst: Path) -> int:
    """Copy the four shard files (and their .err files, when there are some) from src to dst, times kept, each file
    appearing whole or not at all. 0 when all four shards were copied, else 2."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in range(SHARDS):
        for ext in ("jsonl", "err"):
            f = src / f"s{i}.{ext}"
            if not f.is_file():
                if ext == "jsonl":
                    print(f"shard {i} is missing in the source")
                    return 2
                continue
            part = dst / f"s{i}.{ext}.part"
            shutil.copy2(f, part)
            os.replace(part, dst / f"s{i}.{ext}")
            n += 1
    print(f"copied {n} files of the earlier walk")
    return 0


def main(argv: list[str] | None = None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 2:
        print(__doc__)
        return 2
    return copy_shards(Path(a[0]), Path(a[1]))


if __name__ == "__main__":
    sys.exit(main())
