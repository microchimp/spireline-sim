"""A cache for the lab's grouping step (tools/fidelity/lab_groups.py): what the step learns from each finished job.

The step opens about 90,000 different small files per run, on the hard disk the jobs also write to: 340 to 440 s when the
files are still in memory, 800 to 3,900 s when they are not (docs/diagnostics/2026-11-02-file-opens.md). A finished job's
files are never written again, so the step need read each of them once in its life. This module keeps, under
LAB/cache/groups/ (beside LAB/stamps/, which decides whether the step runs at all):

    jobs/<job>.json        per finished job: the list of its sources (what lab_sources found by sniffing its json files),
                           keyed by the job's directory listing (name, size and time of every .json/.jsonl file in it), and
                           per source its key, pool size, symptom count and the name of its symptom blob
    sym/<aa>/<key>.json.gz the symptoms the step's reader extracted from one source, named by the source's key

The source key is the digest of: the code the readers run (lab_groups.py, lab_targets.py, hook_table.py, scenarios.py and the
comparers it judges with, the id map: READER_CODE), the source's label and time, and the name, size and time of every file the
reader looks at. A listing of all 205,000 files of the finished jobs is one `os.scandir` pass (1.5 s on the lab's disk), so a
run with a warm cache opens one file per job that has symptoms and no source file at all.

Rules, each one tested (tests/python/test_lab_groupcache.py):

* A key that misses reads the source again and replaces the entry; a changed file or changed code is a miss.
* Nothing is trusted on a size-and-time match for a file newer than two seconds (a file a writer may still be at): such a
  source, or a job holding such a file, is read fresh and not stored. The listing is taken before the reading, so a file that
  changes during it carries the old size and time into the entry and misses the next time.
* A cache file that does not parse, is cut short, or does not say what the entry says is deleted and rebuilt, never fatal;
  a cache that cannot be written (no space, no folder) costs only the speed. Deleting LAB/cache/groups/ restores the old
  behaviour; `lab_groups.py --force` clears it, `--no-cache` leaves it alone.
* A job's file is written as soon as its last source is read, so a run that is stopped (the runner gives the machine back
  on any input or load) keeps the progress it made.
* Only a finished job (done.rc) is cached; the runner's own unfinished job, and sources named on the command line, are read.

A listing reports what a directory entry holds, which can lag for a file a writer still has open; that is the second case of
the two-second rule, and finished jobs are not written again. Nothing here runs a game or an emulator.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
import re
import time
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.lab import stepcache as sc

FORMAT = 1
RACY_NS = 2 * 10**9  # a file whose time is within this of the listing is not trusted (a writer may still be at it)
JOB_SUFFIXES = (".json", ".jsonl")  # the files the finding of a job's sources depends on
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
GC_GRACE_S = 3600.0  # a blob or temp file younger than this is left alone by the clean-up (another run may be writing it)
# What the readers' output depends on besides the source's own files, relative to the fork's root. The grouping that follows
# the reading is not among them: a change there needs no new reading, but this list is the same as the step's skip digest
# (lab_groups.GROUPS_CODE) plus the comparers `derived_verdicts` judges with, so one rule covers both.
READER_CODE = ("tools/fidelity/lab_groups.py", "tools/fidelity/lab_targets.py", "tools/fidelity/hook_table.py", "tools/lab/stepcache.py",
               "tools/lab/groupcache.py", "referee/scenarios.py", "referee/refset.py", "tools/fidelity/fight_compare.py",
               "tools/fidelity/reanchor_compare.py", "tools/fidelity/setup_compare.py", "tools/fidelity/coverage.py", "data/id_map.json")


def norm(p: Path | str) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


class Listing:
    """Snapshots of folder listings: name -> (size, mtime_ns, is_dir) of each entry, one `os.scandir` per folder, kept for the
    life of the object. `sig` gives the 'size:mtime_ns' of stepcache.stat_sig without a stat call of its own."""

    def __init__(self, now_ns: int | None = None):
        self.now_ns = time.time_ns() if now_ns is None else now_ns  # taken before any listing: the 'two seconds' are counted from here
        self.dirs: dict[str, dict[str, tuple[int, int, bool]] | None] = {}
        self.scanned = 0
        self.seconds = 0.0

    def entries(self, d: Path | str) -> dict[str, tuple[int, int, bool]] | None:
        """The folder's entries, None when it cannot be listed."""
        k = norm(d)
        if k in self.dirs:
            return self.dirs[k]
        t0 = time.perf_counter()
        out: dict[str, tuple[int, int, bool]] | None = {}
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        st = e.stat()
                        out[e.name] = (st.st_size, st.st_mtime_ns, e.is_dir(follow_symlinks=False))
                    except OSError:
                        continue
        except OSError:
            out = None
        self.dirs[k] = out
        self.scanned += 1
        self.seconds += time.perf_counter() - t0
        return out

    def names(self, d: Path | str) -> list[str]:
        """sorted(os.listdir(d)), [] where the folder cannot be listed."""
        e = self.entries(d)
        return sorted(e) if e else []

    def sig(self, path: Path | str) -> str:
        """stepcache.stat_sig(path), from the folder's listing when it has been taken."""
        d, n = os.path.split(str(path))
        e = self.dirs.get(norm(d))
        if e is not None and n in e and not e[n][2]:
            size, mt, _ = e[n]
            return f"{size}:{mt}"
        return sc.stat_sig(path)  # a folder not listed (yet), a name the listing has not seen, or a folder (whose size the listing gives as 0)

    def stat_files(self, files: Iterable[Path | str]) -> list[tuple[str, str, int]]:
        """(path with forward slashes, 'size:mtime_ns' as stepcache.stat_sig makes it, mtime_ns or 0) for each file, from the listing
        where the folder has been listed (one split and one lookup per folder run, not per file)."""
        out: list[tuple[str, str, int]] = []
        last_d, ents = None, None
        for f in files:
            name = str(f)
            d, n = os.path.split(name)
            if d != last_d:
                last_d, ents = d, self.dirs.get(norm(d))
            got = ents.get(n) if ents is not None else None
            if got is not None and not got[2]:  # a folder's size is 0 in a listing and 4096 to os.stat: ask for folders
                out.append((name.replace("\\", "/"), f"{got[0]}:{got[1]}", got[1]))
            else:
                st = None
                with contextlib.suppress(OSError):
                    st = os.stat(name)
                out.append((name.replace("\\", "/"), f"{st.st_size}:{st.st_mtime_ns}" if st else sc.MISSING, st.st_mtime_ns if st else 0))
        return out

    @staticmethod
    def rows_hash(rows: Iterable[tuple[str, str, int]], base: Path | str | None = None) -> str:
        """stepcache.stats_hash over the files of `stat_files` rows: the same digest."""
        parts = []
        prefix = str(base).replace("\\", "/").rstrip("/") + "/" if base is not None else None
        for name, sig, _mt in sorted(rows, key=lambda r: r[0]):
            parts.append([name[len(prefix):] if prefix and name.startswith(prefix) else name, sig])
        return sc.digest({"stats": parts})

    def mtime_ns(self, path: Path | str) -> int:
        """The file's time (0 when unknown): the 'two seconds' check."""
        d, n = os.path.split(str(path))
        e = self.dirs.get(norm(d))
        if e is not None and n in e:
            return e[n][1]
        with contextlib.suppress(OSError):
            return os.stat(path).st_mtime_ns
        return 0

    def walk_files(self, root: Path | str, skip_dirs: tuple[str, ...] = sc.SKIP_DIRS) -> list[Path]:
        """stepcache.walk_files(root) from the listing: the files under root (build output and VCS folders left out)."""
        out: list[Path] = []
        stack = [Path(root)]
        while stack:
            d = stack.pop()
            ents = self.entries(d) or {}
            out += [d / n for n in sorted(ents) if not ents[n][2]]
            stack += [d / n for n in sorted(ents, reverse=True) if ents[n][2] and n not in skip_dirs]
        return out

    def stats_hash(self, files: Iterable[Path | str], base: Path | str | None = None) -> str:
        """stepcache.stats_hash(files, base), the same digest, from the listing."""
        parts = []
        for f in sorted(str(x).replace("\\", "/") for x in files):
            name = f
            if base is not None:
                b = str(base).replace("\\", "/").rstrip("/") + "/"
                name = f[len(b):] if f.startswith(b) else f
            parts.append([name, self.sig(f)])
        return sc.digest({"stats": parts})

    def json_files(self, root: Path | str) -> list[tuple[str, int, int]]:
        """(path under root with forward slashes, size, mtime_ns) of every .json and .jsonl file under root, sorted: what the
        finding of a job's sources depends on."""
        out: list[tuple[str, int, int]] = []
        top = Path(root)
        stack: list[tuple[Path, str]] = [(top, "")]
        while stack:
            d, pre = stack.pop()
            for n, (size, mt, isdir) in (self.entries(d) or {}).items():
                if isdir:
                    stack.append((d / n, pre + n + "/"))
                elif n.endswith(JOB_SUFFIXES):
                    out.append((pre + n, size, mt))
        out.sort()
        return out


@dataclass
class Stats:
    jobs: int = 0
    jobs_hit: int = 0  # source list taken from the cache
    jobs_found: int = 0  # source list found again (new job, changed file, new code)
    jobs_uncached: int = 0  # not finished, or a file newer than two seconds
    sources: int = 0
    sources_hit: int = 0
    sources_read: int = 0
    sources_uncached: int = 0
    symptoms_hit: int = 0
    symptoms_read: int = 0
    corrupt: int = 0
    write_errors: int = 0
    blobs_written: int = 0
    files_removed: int = 0
    load_s: float = 0.0
    store_s: float = 0.0

    def line(self, listing: Listing | None = None) -> str:
        scan = f"listed {listing.scanned} folders in {listing.seconds:.1f}s; " if listing else ""
        return (f"cache: {scan}jobs {self.jobs} (source list from cache {self.jobs_hit}, found {self.jobs_found}, uncached {self.jobs_uncached}); "
                f"sources {self.sources} (from cache {self.sources_hit}, read {self.sources_read}, uncached {self.sources_uncached}); "
                f"symptoms {self.symptoms_hit} from cache, {self.symptoms_read} read; cache load {self.load_s:.1f}s, store {self.store_s:.1f}s"
                + (f"; corrupt files deleted {self.corrupt}" if self.corrupt else "")
                + (f"; write errors {self.write_errors}" if self.write_errors else "")
                + (f"; old cache files removed {self.files_removed}" if self.files_removed else ""))


@dataclass
class Token:
    """The answer to a lookup: a hit with its symptoms, or a miss that says whether the result may be stored."""
    kind: str
    path: Path
    when: str
    hit: bool = False
    symptoms: list[dict] = field(default_factory=list)
    pool: int = 0
    label: str | None = None
    key: str | None = None
    cacheable: bool = False
    entry: JobEntry | None = None
    idx: int = -1


@dataclass
class JobEntry:
    name: str
    job_dir: Path
    path: Path  # the cache file
    job_key: str
    sources: list[dict]
    writable: bool = True  # False: a file in the job is newer than two seconds
    dirty: bool = False
    todo: set[int] = field(default_factory=set)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Whole or not at all, with a temp name of this process's own (two runs never share one); retries where Windows briefly
    locks the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    err: OSError | None = None
    for k in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            err = e
            time.sleep(0.05 * (k + 1))
    with contextlib.suppress(OSError):
        tmp.unlink()
    if err:
        raise err


class GroupCache:
    def __init__(self, lab: Path, root: Path, directory: Path | None = None, enabled: bool = True, refresh: bool = False,
                 now_ns: int | None = None, code_files: Iterable[str] = READER_CODE,
                 files_of: Callable[[str, Path], list[Path]] | None = None, listing: Listing | None = None):
        """`root`: the fork's checkout the readers run from (their code is an input). `files_of(kind, path)`: the files a source's
        reader looks at (lab_groups.source_files over this cache's listing). `refresh`: look nothing up, store everything."""
        self.lab = Path(lab)
        self.root = Path(root)
        self.dir = Path(directory) if directory else self.lab / "cache" / "groups"
        self.enabled = enabled
        self.refresh = refresh
        self.listing = listing or Listing(now_ns)
        self.files_of = files_of
        self.stats = Stats()
        self.code = sc.files_hash([self.root / c for c in code_files], self.root) if enabled else ""
        self.by_source: dict[tuple[str, str], tuple[JobEntry, int]] = {}
        self.entries: dict[str, JobEntry] = {}
        self.jobs_seen: set[str] = set()
        self.warned = False
        self._rows: dict[tuple[str, str], list[tuple[str, str, int]]] = {}

    def rows_of(self, kind: str, path: Path | str) -> list[tuple[str, str, int]]:
        """The files a source's reader looks at, with their signatures and times (listed once per run, however often asked)."""
        k = (kind, str(path))
        if k not in self._rows:
            self._rows[k] = self.listing.stat_files(self.files_of(kind, Path(path))) if self.files_of else []
        return self._rows[k]

    # ------------------------------------------------------------------ small things

    def _warn(self, what: str) -> None:
        self.stats.write_errors += 1
        if not self.warned:
            self.warned = True
            print(f"groupcache: {what}; going on without it")

    def _drop(self, path: Path) -> None:
        """A cache file that cannot be trusted: removed, and counted."""
        self.stats.corrupt += 1
        with contextlib.suppress(OSError):
            path.unlink()

    def clear(self) -> None:
        """Remove the whole cache (--force)."""
        if not self.dir.is_dir():
            return
        for p in sorted(self.dir.rglob("*"), reverse=True):
            with contextlib.suppress(OSError):
                p.rmdir() if p.is_dir() else p.unlink()
        with contextlib.suppress(OSError):
            self.dir.rmdir()

    def job_file(self, name: str) -> Path:
        stem = name if SAFE_NAME.match(name) else "x-" + hashlib.sha1(name.encode("utf-8", "replace")).hexdigest()[:24]
        return self.dir / "jobs" / f"{stem}.json"

    def blob_file(self, blob: str) -> Path:
        return self.dir / "sym" / blob[:2] / f"{blob}.json.gz"

    def racy(self, mtime_ns: int) -> bool:
        return mtime_ns > self.listing.now_ns - RACY_NS  # also a time in the future: a clock that cannot be trusted

    # ------------------------------------------------------------------ the source list of a job

    def _read_job(self, path: Path) -> dict | None:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            self._drop(path)
            return None
        ok = (isinstance(doc, dict) and doc.get("format") == FORMAT and isinstance(doc.get("code"), str)
              and isinstance(doc.get("job_key"), str) and isinstance(doc.get("sources"), list)
              and all(isinstance(s, dict) and isinstance(s.get("kind"), str) and isinstance(s.get("path"), str) for s in doc["sources"]))
        if not ok:
            self._drop(path)
            return None
        return doc

    def job_sources(self, job: Path, when: str, discover: Callable[[], list[tuple[str, Path, str]]]) -> list[tuple[str, Path, str]]:
        """(kind, path, when) of the sources of a finished job: from the cache when the job's json files and the code are as they
        were, else `discover()` (the sniffing) and the result remembered."""
        self.stats.jobs += 1
        self.jobs_seen.add(job.name)
        if not self.enabled:
            self.stats.jobs_uncached += 1
            return discover()
        t0 = time.perf_counter()
        files = self.listing.json_files(job)
        jk = sc.digest({"files": files})
        newest = max((f[2] for f in files), default=0)
        writable = not self.racy(newest)
        path = self.job_file(job.name)
        old = None if self.refresh or not writable else self._read_job(path)
        self.stats.load_s += time.perf_counter() - t0
        if old is not None and old["code"] == self.code and old["job_key"] == jk:
            sources = old["sources"]
            self.stats.jobs_hit += 1
        else:
            found = discover()
            if not writable:
                self.stats.jobs_uncached += 1
                return found
            self.stats.jobs_found += 1
            carried = {(s["kind"], s["path"]): s for s in (old["sources"] if old and old["code"] == self.code else [])}
            sources = []
            for kind, p, _w in found:
                rel_ = self._rel(job, p)
                prev = carried.get((kind, rel_), {})
                sources.append({"kind": kind, "path": rel_, **{k: prev[k] for k in ("label", "when", "key", "pool", "n", "blob", "sha") if k in prev}})
        entry = JobEntry(job.name, job, path, jk, sources, writable=writable, dirty=old is None or old["job_key"] != jk or old["code"] != self.code)
        out: list[tuple[str, Path, str]] = []
        last: dict[tuple[str, str], int] = {}  # a source named twice is read (and kept) once
        for i, s in enumerate(sources):
            p = self._abs(job, s["path"])
            out.append((s["kind"], p, when))
            last[(s["kind"], str(p))] = i
        for k, i in last.items():
            self.by_source[k] = (entry, i)
        entry.todo = set(last.values())
        self.entries[job.name] = entry
        if not sources and entry.dirty:  # a job with no source: nothing to read, the empty list is the entry
            self._write_job(entry)
        return out

    @staticmethod
    def _rel(job: Path, p: Path) -> str:
        """A source's path as the cache keeps it: under the job with forward slashes, else as it is."""
        try:
            return Path(p).relative_to(job).as_posix()
        except ValueError:
            return str(p)

    @staticmethod
    def _abs(job: Path, s: str) -> Path:
        return Path(s) if os.path.isabs(s) else job / s

    # ------------------------------------------------------------------ the symptoms of a source

    def lookup(self, kind: str, path: Path, when: str) -> Token:
        """The cached symptoms of a source (`hit`), else a token that says whether `store` may keep what the reader makes of it."""
        tok = Token(kind, Path(path), when)
        self.stats.sources += 1
        found = self.by_source.get((kind, str(path))) if self.enabled else None
        if found is None or self.files_of is None:
            self.stats.sources_uncached += 1
            return tok
        entry, idx = found
        tok.entry, tok.idx = entry, idx
        s = entry.sources[idx]
        if not entry.writable or os.path.isabs(s["path"]):  # a warm job, or a source outside the job (its listing does not cover it)
            self.stats.sources_uncached += 1
            return tok
        rows = self.rows_of(kind, path)
        if any(self.racy(r[2]) for r in rows):  # a file a writer may still be at
            self.stats.sources_uncached += 1
            return tok
        base = str(self.lab).replace("\\", "/").rstrip("/") + "/"
        sigs = [[r[0][len(base):] if r[0].startswith(base) else r[0], r[1]] for r in rows]
        ident = [FORMAT, self.code, kind, f"{entry.name}/{s['path']}", when, sigs]
        key = hashlib.sha256(json.dumps(ident, separators=(",", ":")).encode("utf-8")).hexdigest()
        tok.key, tok.cacheable = key, True
        complete = isinstance(s.get("n"), int) and isinstance(s.get("pool"), int) and isinstance(s.get("label"), str)
        if self.refresh or s.get("key") != key or not complete:
            return tok
        got = self._load_blob(s)
        if got is None:
            return tok
        tok.hit, tok.symptoms, tok.pool, tok.label = True, got, s["pool"], s["label"]
        self.stats.sources_hit += 1
        self.stats.symptoms_hit += len(got)
        self._done(entry, idx)
        return tok

    def _load_blob(self, s: dict) -> list[dict] | None:
        n = s["n"]
        if n == 0:
            return []
        blob = s.get("blob")
        if not isinstance(blob, str) or not re.fullmatch(r"[0-9a-f]{40}", blob):
            return None
        p = self.blob_file(blob)
        t0 = time.perf_counter()
        try:
            data = p.read_bytes()
            if hashlib.sha256(data).hexdigest()[:24] != s.get("sha"):  # not the bytes the entry was written with
                self._drop(p)
                return None
            doc = json.loads(gzip.decompress(data))
        except FileNotFoundError:
            return None
        except (OSError, EOFError, zlib.error, ValueError):  # gzip.BadGzipFile is an OSError; a cut file raises EOFError
            self._drop(p)
            return None
        finally:
            self.stats.load_s += time.perf_counter() - t0
        if not isinstance(doc, list) or len(doc) != n or not all(isinstance(x, dict) for x in doc):
            self._drop(p)
            return None
        return doc

    def store(self, tok: Token, symptoms: list[dict], pool: int, label: str) -> None:
        """Remember what the reader made of the source (call before the symptoms are changed in any way)."""
        if tok.hit:
            return
        self.stats.symptoms_read += len(symptoms)
        entry = tok.entry
        if not (tok.cacheable and tok.key and entry is not None and entry.writable):  # counted as uncached by the lookup
            if entry is not None:
                self._done(entry, tok.idx)
            return
        self.stats.sources_read += 1
        t0 = time.perf_counter()
        s = entry.sources[tok.idx]
        blob, sha = None, ""
        try:
            if symptoms:
                blob = tok.key[:40]
                data = gzip.compress(json.dumps(symptoms, separators=(",", ":")).encode("utf-8"), compresslevel=3)
                atomic_write_bytes(self.blob_file(blob), data)
                sha = hashlib.sha256(data).hexdigest()[:24]
                self.stats.blobs_written += 1
        except (OSError, TypeError, ValueError) as e:
            self._warn(f"a symptom file was not written ({type(e).__name__})")
            blob = None
            s.pop("key", None)
        else:
            s.update({"label": label, "when": tok.when, "key": tok.key, "pool": int(pool), "n": len(symptoms)})
            if blob:
                s["blob"], s["sha"] = blob, sha
            else:
                s.pop("blob", None)
                s.pop("sha", None)
            entry.dirty = True
        self.stats.store_s += time.perf_counter() - t0
        self._done(entry, tok.idx)

    def _done(self, entry: JobEntry, idx: int) -> None:
        entry.todo.discard(idx)
        if not entry.todo and entry.dirty and entry.writable:
            self._write_job(entry)

    def _write_job(self, entry: JobEntry) -> None:
        t0 = time.perf_counter()
        doc = {"format": FORMAT, "job": entry.name, "code": self.code, "job_key": entry.job_key, "sources": entry.sources}
        try:
            atomic_write_bytes(entry.path, json.dumps(doc, separators=(",", ":")).encode("utf-8"))
            entry.dirty = False
        except (OSError, TypeError, ValueError) as e:
            self._warn(f"a job file was not written ({type(e).__name__})")
        self.stats.store_s += time.perf_counter() - t0

    # ------------------------------------------------------------------ clean-up

    def finish(self, complete: bool = True) -> None:
        """Flush what is left and, when every finished job was seen, remove the files of jobs that are gone and the symptom
        files no job refers to (older than an hour: another run may be writing them)."""
        if not self.enabled:
            return
        for entry in self.entries.values():
            if entry.dirty and entry.writable and not entry.todo:
                self._write_job(entry)
        if not complete:
            return
        now = time.time()
        if not self.jobs_seen:
            return
        keep_jobs = {self.job_file(n).name for n in self.jobs_seen}
        sources = [s for e in self.entries.values() for s in e.sources]
        for name in self.jobs_seen - set(self.entries):  # a job this run left uncached (a warm file): its file stands, and its symptom files
            with contextlib.suppress(OSError, ValueError, AttributeError, TypeError):
                sources += json.loads(self.job_file(name).read_text(encoding="utf-8")).get("sources", [])
        keep_blobs = {s["blob"] + ".json.gz" for s in sources if isinstance(s, dict) and isinstance(s.get("blob"), str)}
        for sub, keep, suffix in (("jobs", keep_jobs, ".json"), ("sym", keep_blobs, ".json.gz")):
            root = self.dir / sub
            if not root.is_dir():
                continue
            for p in list(root.rglob("*")):
                if not p.is_file():
                    continue
                with contextlib.suppress(OSError):
                    old_enough = now - p.stat().st_mtime > GC_GRACE_S
                    if p.name.endswith(".tmp") and old_enough:
                        p.unlink()
                    elif p.name.endswith(suffix) and p.name not in keep and old_enough:
                        p.unlink()
                        self.stats.files_removed += 1
