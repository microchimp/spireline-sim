"""Skip a lab step whose inputs have not changed (reset order D44 step D).

A step that is a pure function of files (the codex walk, the corpus check, the groups, the fix kits) records a digest of
those files beside its output. The next time, when the digest is the same and the output it recorded is still there as
written, the step is skipped and the earlier output stands; any other case runs the step. The decision is by inputs
only: no timer, no counter.

    stamp = LAB/stamps/<step>.json
        {"step", "digest", "parts": {name: short digest}, "outputs": {label: [size, mtime_ns]}, "job", "utc", "rc",
         "content": {label: digest of what the output holds}, "changed": whether that differs from the previous stamp's}

An output is named by a label: its path under the lab folder by default, so a stamp holds no absolute path. A file the
step reads is hashed by content when it is small or can change without changing its size (code, the fix list, the
retest file), and by (size, mtime_ns) when it is big and written once (a finished job's recordings, the Codex
dataset). Nothing here writes except `write_stamp`; the checks are pure, so a dry run can ask them.

`content` and `changed` are for the lab's clock (tools/lab/clock_report.py): a step that runs and leaves the same output as the
time before it has told nobody anything. They play no part in the skip decision. `changed` is True for a first stamp and for an
output that differs, False when every output is as it was, None when that cannot be told (a stamp from before this field, an
output too big to hash).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

STAMP_DIR = "stamps"
SKIP_DIRS = ("bin", "obj", ".git", "__pycache__", "node_modules")
MISSING = "missing"


def sha_hex(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def digest(parts: Mapping[str, Any]) -> str:
    """One digest of named parts (each a string, number, list or dict): the same parts give the same digest, in any order."""
    return sha_hex(json.dumps(dict(parts), sort_keys=True, separators=(",", ":"), default=str))


def file_hash(path: Path | str) -> str:
    """Content digest of a file, MISSING when it cannot be read."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return MISSING
    return h.hexdigest()


def stat_sig(path: Path | str) -> str:
    """'size:mtime_ns' of a file or folder, MISSING when absent: for a file that is big and written once."""
    try:
        st = os.stat(path)
    except OSError:
        return MISSING
    return f"{st.st_size}:{st.st_mtime_ns}"


def walk_files(root: Path | str, suffixes: tuple[str, ...] = (), skip_dirs: tuple[str, ...] = SKIP_DIRS) -> list[Path]:
    """Files under root (sorted), without build output and VCS folders; `suffixes` keeps only those endings (all when empty)."""
    out: list[Path] = []
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in skip_dirs)
        out += [Path(dp) / f for f in sorted(fn) if not suffixes or f.endswith(suffixes)]
    return out


def files_hash(files: Iterable[Path | str], base: Path | str | None = None) -> str:
    """Digest of a set of files by content: their names (relative to base when given) and the content digest of each."""
    parts = []
    for f in sorted(str(x).replace("\\", "/") for x in files):
        name = f
        if base is not None:
            b = str(base).replace("\\", "/").rstrip("/") + "/"
            name = f[len(b):] if f.startswith(b) else f
        parts.append([name, file_hash(f)])
    return digest({"files": parts})


def stats_hash(files: Iterable[Path | str], base: Path | str | None = None) -> str:
    """Digest of a set of files by (name, size, mtime_ns): for recordings that a finished job wrote once."""
    parts = []
    for f in sorted(str(x).replace("\\", "/") for x in files):
        name = f
        if base is not None:
            b = str(base).replace("\\", "/").rstrip("/") + "/"
            name = f[len(b):] if f.startswith(b) else f
        parts.append([name, stat_sig(f)])
    return digest({"stats": parts})


def tree_hash(root: Path | str, suffixes: tuple[str, ...] = (), skip_dirs: tuple[str, ...] = SKIP_DIRS) -> str:
    """Content digest of every file under root (build output and VCS folders left out)."""
    return files_hash(walk_files(root, suffixes, skip_dirs), root)


# ---------------------------------------------------------------- stamps

def stamp_path(lab: Path | str, step: str) -> Path:
    return Path(lab) / STAMP_DIR / f"{step}.json"


def read_stamp(lab: Path | str, step: str) -> dict[str, Any] | None:
    try:
        doc = json.loads(stamp_path(lab, step).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and isinstance(doc.get("digest"), str) and isinstance(doc.get("outputs"), dict) else None


def label_of(lab: Path | str, path: Path | str) -> str:
    """An output's label: its path under the lab folder (forward slashes), else its file name. The path is read as written first: the
    lab's runs folder is a junction to another drive (2026-11-03), and a path resolved through it no longer lies under the lab, which
    cut every job output's label down to its file name, so no stamp could find its output again and no check was ever skipped."""
    p, base = Path(path), Path(lab)
    try:
        return Path(os.path.abspath(p)).relative_to(os.path.abspath(base)).as_posix()
    except ValueError:
        pass
    try:
        return p.resolve().relative_to(base.resolve()).as_posix()
    except (ValueError, OSError):
        return p.name


def where_now(lab: Path | str, stamp: Mapping[str, Any], label: str) -> Path:
    """Where the output a stamp calls `label` is now: under the lab folder. A label that is a bare file name, in a stamp that names its
    job, was written by the runner of 2026-11-03 before label_of kept the junction's name: its file is the one of that name in the
    job's folder (none or several: the lab folder, where it will not be found and the step runs)."""
    p = Path(lab) / label
    job = stamp.get("job")
    if "/" in label or not job or p.exists():
        return p
    hits = [Path(dp) / label for dp, _, fn in os.walk(Path(lab) / "runs" / str(job)) if label in fn]
    return hits[0] if len(hits) == 1 else p


def outputs_of(lab: Path | str, outputs: Iterable[Path | str] | Mapping[str, Path | str]) -> dict[str, Path]:
    if isinstance(outputs, Mapping):
        return {k: Path(v) for k, v in outputs.items()}
    return {label_of(lab, p): Path(p) for p in outputs}


def _sig(path: Path) -> list[int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


def skip_reason(lab: Path | str, step: str, want: str, outputs: Iterable[Path | str] | Mapping[str, Path | str] | None = None) -> str | None:
    """Why the step need not run, or None when it must. It need not run when its last stamp holds this digest and every output
    the stamp recorded is still there with the size and time it was recorded with. `outputs` names where the outputs are now
    (label -> path); by default they are under the lab folder, at the labels the stamp holds. Pure: reads, never writes."""
    stamp = read_stamp(lab, step)
    if stamp is None or stamp["digest"] != want:
        return None
    if outputs is None:
        now = {k: where_now(lab, stamp, k) for k in stamp["outputs"]}
    else:
        now = outputs_of(lab, outputs)
    if set(now) != set(stamp["outputs"]):
        return None
    for k, p in now.items():
        if _sig(p) != stamp["outputs"][k]:
            return None  # an output is gone or was written again since: the stamp does not describe what is there
    where = f" from {stamp['job']}" if stamp.get("job") else ""
    return f"inputs unchanged since {stamp.get('utc', '?')} (digest {want[:12]}); the earlier output{where} stands"


CONTENT_CAP = 512 << 20  # an output bigger than this is not read to tell whether it changed


def content_hash(path: Path | str) -> str | None:
    """A short content digest of a file, None for a folder, a file over CONTENT_CAP, or one that cannot be read."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) > CONTENT_CAP:
            return None
        return file_hash(path)[:24]
    except OSError:
        return None


def changed_since(previous: Mapping[str, Any] | None, now: Mapping[str, str | None]) -> bool | None:
    """Whether the outputs' content differs from the previous stamp's: True when there was no stamp or an output differs, False when
    all are as they were, None when that cannot be told."""
    if previous is None:
        return True
    old = previous.get("content")
    if not isinstance(old, dict):
        return None
    if set(old) != set(now):
        return True
    unknown = False
    for k, v in now.items():
        if v is None or old.get(k) is None:
            unknown = True
        elif v != old[k]:
            return True
    return None if unknown else False


def write_stamp(lab: Path | str, step: str, want: str, outputs: Iterable[Path | str] | Mapping[str, Path | str], job: str = "",
                parts: Mapping[str, str] | None = None, rc: int | None = None, content: Mapping[str, str] | None = None) -> None:
    """Record that `outputs` (written just now) came from inputs with this digest. An output that is missing is recorded as
    such by leaving the stamp out: nothing is skipped on a step that wrote nothing. `content` gives the digest of what an output
    holds, by label, where the file itself carries something that changes every run (a time, a sha): the others are hashed."""
    sigs: dict[str, list[int]] = {}
    held: dict[str, str | None] = {}
    outs = outputs_of(lab, outputs)
    for k, p in outs.items():
        s = _sig(p)
        if s is None:
            remove_stamp(lab, step)
            return
        sigs[k] = s
    for k, p in outs.items():
        held[k] = (content or {}).get(k) or content_hash(p)
    doc = {"step": step, "digest": want, "parts": dict(parts or {}), "outputs": sigs, "job": job, "rc": rc,
           "content": held, "changed": changed_since(read_stamp(lab, step), held),
           "utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
    path = stamp_path(lab, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def remove_stamp(lab: Path | str, step: str) -> None:
    try:
        stamp_path(lab, step).unlink()
    except OSError:
        pass


def refresh_outputs(lab: Path | str, step: str, outputs: Iterable[Path | str] | Mapping[str, Path | str]) -> None:
    """Re-record the outputs' size and time in an existing stamp (a skipped step that touched its output)."""
    stamp = read_stamp(lab, step)
    if stamp is not None:
        write_stamp(lab, step, stamp["digest"], outputs, stamp.get("job", ""), stamp.get("parts"), stamp.get("rc"), stamp.get("content"))


# ---------------------------------------------------------------- the engine

ENGINE_TREES = ("src/Sts2Emulator", "tools/emudump")
ENGINE_FILES = ("data/id_map.json", "data/game_version.json", "tools/fidelity/power_aliases.json", "tools/fidelity/migrated_families.json",
                "Directory.Build.props", "Directory.Build.targets", "Directory.Packages.props", "global.json", "nuget.config", "NuGet.Config")
ENGINE_LINKED = "src/Sts2Emulator.Tests/Support/Invariants"  # EmuDump compiles these files in (tools/emudump/EmuDump.csproj)
_engine_cache: dict[str, tuple[str, str]] = {}


def _engine_key(tree: Path) -> str:
    """A cheap key of the files the engine digest covers (names, sizes, times): the digest is recomputed only when it moves."""
    parts = []
    for rel in ENGINE_TREES + (ENGINE_LINKED,):
        for f in walk_files(tree / rel):
            parts.append((str(f), _sig(f)))
    parts += [(rel, _sig(tree / rel)) for rel in ENGINE_FILES]
    return digest({"k": parts})


def engine_digest(tree: Path | str) -> str:
    """Digest of what the emulator and EmuDump are built from at `tree`: the emulator and EmuDump sources, the invariant checks
    EmuDump compiles in, the data files it copies, the build settings. A landing that changes none of them (lab code, rl/, docs,
    the xunit tests) leaves it as it was, so the checks that replay the engine need not run again."""
    t = Path(tree)
    key = str(t.resolve())
    k = _engine_key(t)
    got = _engine_cache.get(key)
    if got and got[0] == k:
        return got[1]
    files = [f for rel in ENGINE_TREES + (ENGINE_LINKED,) for f in walk_files(t / rel)] + [t / rel for rel in ENGINE_FILES if (t / rel).is_file()]
    d = files_hash(files, t)
    _engine_cache[key] = (k, d)
    return d


def tools_digest(tree: Path | str, rels: Iterable[str] = ("tools/fidelity", "referee")) -> str:
    """Digest of the Python and JSON tools a check runs (their code is an input), by content, without caches and tests."""
    t = Path(tree)
    files: list[Path] = []
    for rel in rels:
        files += walk_files(t / rel, (".py", ".json"), SKIP_DIRS + ("runs", "jobs", "serve", "driver"))
    return files_hash(files, t)


def decompiled_sig(decompiled: Path | str) -> str:
    """The decompile folder is fixed for a game version: its own time and entry count tell a new one."""
    try:
        return f"{os.stat(decompiled).st_mtime_ns}:{len(os.listdir(decompiled))}"
    except OSError:
        return MISSING


def when_unchanged(lab_of: Callable[[Any], Path], step: str, digest_of: Callable[[Any], str]) -> Callable[[Any], str | None]:
    """A `Cmd.when` for a step: its skip reason, or None to run. The digest function is trusted not to raise; if it does, the step runs."""
    def check(c: Any) -> str | None:
        try:
            return skip_reason(lab_of(c), step, digest_of(c))
        except Exception:  # noqa: BLE001  (a step that cannot tell whether its inputs changed runs)
            return None
    return check
