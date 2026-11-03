"""A frozen copy of the lab tree for the practice filler (D52).

The practice filler (tools/lab/cores.py) used to run from the lab tree, and a presort moves that tree: it checks out the new main,
builds it and publishes into `out/` (a DLL the filler's emulator processes hold). So the filler stopped for the length of every
presort, 36 minutes or more of single-threaded steps with the machine at 7% of its CPU, and the lab presorts several times a day.

The filler now runs from a copy of the tree made once per landing, at the moment the tree stands at a presorted build and no presort
is moving it: the tracked files (`rl/`, `src/`, `tools/`, `data/`, `referee/`, `docs/`; not the 86 MB of test fixtures, which nothing
reads at run time) and the published `out/` (the build of that sha). Nothing writes into the copy but the filler's own processes
(the runner's children have PYTHONDONTWRITEBYTECODE). The copy lives outside the lab folder and the tree, one folder per sha:

    <root>/<sha12>/            the copy; `FROZEN` inside it holds the full sha and is written last
    <root>/<sha12>.part        a copy being made (never used; removed by the next clean-up)

A copy without its marker, or with a marker for another sha, is not a copy: it is made again. Folders of other shas are removed
when no running block uses them. About 1,400 files and 160 MB, two to ten seconds.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Iterable
from pathlib import Path

MARK = "FROZEN"
SKIP_DIRS = frozenset({"bin", "obj", "__pycache__", ".git", "node_modules"})  # build output the published out/ replaces, VCS, caches
SKIP_TOP = frozenset({"tests"})  # fixtures only a test reads
NEEDS = ("rl/gen0.py", "rl/gen_loop.py", "out/Sts2Emulator.dll")  # what a block cannot start without (the selfplay kind's own needs)
KEEP_AGE_S = 3600.0  # a `.part` folder this old is a copy that never finished


class FreezeError(Exception):
    """The tree cannot be copied now; the message says why (it is logged as the reason the filler did not start)."""


def folder(root: Path | str, sha: str) -> Path:
    return Path(root) / sha[:12]


def ready(root: Path | str, sha: str) -> Path | None:
    """The finished copy of `sha`, or None: its marker holds the sha and the files a block starts from are there."""
    d = folder(root, sha)
    try:
        if (d / MARK).read_text(encoding="utf-8").strip() != sha:
            return None
    except OSError:
        return None
    return d if all((d / n).is_file() for n in NEEDS) else None


def _ignore(top: Path):
    def ign(path: str, names: list[str]) -> list[str]:
        return [n for n in names if n in SKIP_DIRS or (Path(path) == top and n in SKIP_TOP)]
    return ign


def make(tree: Path | str, root: Path | str, sha: str, head: str) -> Path:
    """Copy `tree` (which must stand at `sha`: `head` is what `git rev-parse HEAD` says there) to <root>/<sha12>. The copy is built in
    a `.part` folder and renamed once whole, so a reader sees all of it or none. FreezeError when the tree is not at `sha` or lacks
    the files a block needs; OSError when the copy fails (the partial folder is removed)."""
    src, root = Path(tree), Path(root)
    if not sha or head != sha:
        raise FreezeError(f"the lab tree is at {(head or '?')[:7]}, not at the presorted build {sha[:7]}")
    missing = [n for n in NEEDS if not (src / n).is_file()]
    if missing:
        raise FreezeError("the lab tree has no " + ", ".join(missing))
    dst, part = folder(root, sha), Path(str(folder(root, sha)) + ".part")
    root.mkdir(parents=True, exist_ok=True)
    for old in (part, dst):
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
            if old.exists():
                raise FreezeError(f"an earlier copy at {old.name} cannot be removed")
    try:
        shutil.copytree(src, part, ignore=_ignore(src))
        (part / MARK).write_text(sha + "\n", encoding="utf-8")  # last: a folder with this file is whole
        os.replace(part, dst)
    except OSError:
        shutil.rmtree(part, ignore_errors=True)
        raise
    return dst


def clean(root: Path | str, keep: Iterable[Path | str] = (), now: float | None = None) -> list[str]:
    """Remove the copies under `root` that are not in `keep` (the one a block runs from, the one about to) and any `.part` folder
    older than KEEP_AGE_S. A folder that cannot be removed (a process still holds a file) stays for the next call. Returns the names
    removed."""
    root = Path(root)
    now = time.time() if now is None else now
    keep_names = {Path(k).name for k in keep}
    gone: list[str] = []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return gone
    for d in entries:
        if not d.is_dir() or d.name in keep_names:
            continue
        if d.name.endswith(".part"):
            try:
                if now - d.stat().st_mtime < KEEP_AGE_S:
                    continue
            except OSError:
                continue
        shutil.rmtree(d, ignore_errors=True)
        if not d.exists():
            gone.append(d.name)
    return gone
