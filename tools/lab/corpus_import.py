"""Import the lab's real-game fights into the regression corpus (tools/fidelity/corpus.py).

    python tools/lab/corpus_import.py --lab <LAB> [--job <name>] [--corpus <dir>] [--budget-gb 2] [--max-fights 100000] [--no-diverge] [--dry-run] [--no-marker]

A job's recordings are ``fights/<name>.out.json`` with ``<name>.verdicts.json`` beside it (one verdict per result,
same order). From each:

- a fight the emulator matched exactly (identical, no mirror gap, no setup-first, no error) is imported plain;
- a fight it diverged on is imported tagged ``lab-diverged`` (an expected failure: the corpus baseline stores its
  divergence, and ``check`` flags it only when it gets worse) unless ``--no-diverge``;
- a fight with an error, a mirror gap, a setup that came first, or that the game broke off is left out: the
  verdict is not a clean statement about the emulator.

Fights are kept once per setup (corpus.spec_id: seed, character, ascension, Neow pick, deck, room, setup ...):
one already in the corpus, or earlier in this import, is skipped, matches before divergences. The corpus stores
them gzip-compressed; ``--budget-gb`` caps the size of its fights folder (the estimate is the compressed size) and
``--max-fights`` the number of fights in the corpus (a full replay costs time, not disk: 100000 is about ten minutes
of `corpus.py run` on 16 processes). A fight over either cap is not imported and its file is not marked covered.
Shrink replays (``*.shrinkN.out.json``) and a whole-runs session's copies are not recordings of their own.

After an import the job folder holds ``corpus-imported.json``: per out file, its size and mtime, the corpus path and
whether every result of it is now represented in the corpus (stored, or its setup stored) with those setups' ids.
tools/lab/prune.py deletes such a file first, once groups.json has read it and the corpus still holds the setups.
``--no-marker`` (and ``--dry-run``) leave no marker: use it for an import into a scratch corpus.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fidelity import corpus as cp  # noqa: E402
from tools.lab import prune as pr  # noqa: E402

MARKER = "corpus-imported.json"
DIVERGED_TAG = "lab-diverged"
GB = 1024 ** 3


def kind(v) -> str:
    """"match", "diverge" or "" (not a clean statement about the emulator) for one verdict."""
    if not isinstance(v, dict) or v.get("error") or v.get("mirror_gap") or v.get("setup_first"):
        return ""
    if v.get("identical") is True:
        return "match"
    if v.get("identical") is False and v.get("field"):
        return "diverge"
    return ""


def gz_size(rec: dict) -> int:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as g:
        g.write(cp.canon(rec).encode("utf-8"))
    return len(buf.getvalue())


def recording_pairs(job: Path) -> list:
    """(out file, verdicts) of every recording in a job folder, shrink replays and run-session copies left out."""
    pairs = []
    for f in sorted(job.rglob("*.out.json")):
        if pr.SCRATCH_RE.search(f.name) or any(p in pr.SCRATCH_DIRS for p in f.relative_to(job).parts):
            continue
        vf = f.with_name(f.name[: -len(".out.json")] + ".verdicts.json")
        try:
            verdicts = json.loads(vf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(verdicts, list):
            pairs.append((f, verdicts))
    return pairs


def import_job(job: Path, corpus: cp.Corpus, budget: int | None = None, diverge: bool = True, dry: bool = False,
               rand_cards: set | None = None, state: dict | None = None, mark: bool = True,
               max_fights: int | None = None) -> dict:
    """Import one job's fights; counts. With ``dry`` nothing is written (corpus or job folder); ``mark=False`` writes no marker.

    ``state`` ({"have": spec ids, "used": bytes, "n": fights}) carries the dedupe set and the budgets spent across a dry run's jobs.
    """
    job = Path(job)
    state = state if state is not None else {}
    if "have" not in state:
        state["have"] = corpus.specs() if corpus.index_path.exists() else set()
        state["used"] = state["n"] = 0
    have = state["have"]
    cands, loaded = [], []
    counts = {"results": 0, "left_out": 0, "pruned": 0, "duplicate": 0, "over_budget": 0, "match": 0, "diverge": 0}
    for f, verdicts in recording_pairs(job):
        try:
            results = json.loads(f.read_text(encoding="utf-8"))["results"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not isinstance(results, list) or len(results) != len(verdicts):
            continue
        loaded.append((f, results))
        for r, v in zip(results, verdicts):
            counts["results"] += 1
            if isinstance(r, dict) and r.get("pruned"):
                counts["pruned"] += 1
                continue
            k = kind(v)
            rec, why = cp.prepare(r) if k else (None, "")
            if rec is None or why != "clean" or (k == "diverge" and not diverge):
                counts["left_out"] += 1
                continue
            cands.append((0 if k == "match" else 1, f, k, rec, r))
    cands.sort(key=lambda c: c[0])  # matches before divergences; the sort is stable
    chosen: dict = {}  # (file, kind) -> [results]
    used = 0
    base = corpus.size() + state["used"]
    stored = (len(corpus.load_index()["fights"]) if corpus.index_path.exists() else 0) + state["n"]
    for _, f, k, rec, r in cands:
        sid = cp.spec_id(rec)
        if sid in have:
            counts["duplicate"] += 1
            continue
        n = gz_size(rec)
        if budget is not None and base + used + n > budget or max_fights is not None and stored + counts["match"] + counts["diverge"] >= max_fights:
            counts["over_budget"] += 1
            continue
        have.add(sid)
        used += n
        chosen.setdefault((f, k), []).append(r)
        counts[k] += 1
    counts["bytes"] = used
    if dry:
        state["used"] += used
        state["n"] += counts["match"] + counts["diverge"]
        return counts
    with tempfile.TemporaryDirectory() as td:
        labels, files = {}, []
        for n, ((f, k), rs) in enumerate(sorted(chosen.items(), key=lambda x: (str(x[0][0]), x[0][1]))):
            tmp = Path(td) / f"{n}.out.json"
            tmp.write_text(json.dumps({"results": rs}), encoding="utf-8")
            labels[str(tmp)] = (f"lab/{job.name}/{f.relative_to(job).as_posix()}", DIVERGED_TAG if k == "diverge" else None)
            files.append(tmp)
        if files:
            corpus.add_files(files, rand_cards=rand_cards, quiet=True, labels=labels)
    if mark:
        _mark(job, corpus, loaded)
    return counts


def _mark(job: Path, corpus: cp.Corpus, loaded: list) -> None:
    """Write the job's marker: which out files the corpus now fully represents."""
    specs = corpus.specs() if corpus.index_path.exists() else set()
    files = {}
    for f, results in loaded:
        covered, ids = True, set()
        for r in results:
            if isinstance(r, dict) and r.get("pruned"):
                continue
            rec, why = cp.prepare(r)
            sid = cp.spec_id(rec) if rec is not None and why == "clean" else None
            if sid is None or sid not in specs:
                covered = False
                break
            ids.add(sid)
        st = f.stat()
        files[f.relative_to(job).as_posix()] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "covered": covered,
                                                "specs": sorted(ids) if covered else []}
    doc = {"corpus": str(corpus.path), "files": files}
    tmp = job / (MARKER + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, job / MARKER)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lab", type=Path, help="the lab root: import every finished job without a marker (or --job)")
    ap.add_argument("--job", help="a job folder name under <lab>/runs, or a path to one")
    ap.add_argument("--corpus", type=Path, default=cp.DEFAULT_CORPUS)
    ap.add_argument("--budget-gb", type=float, help="cap on the size of the corpus fights folder")
    ap.add_argument("--max-fights", type=int, help="cap on the number of fights in the corpus")
    ap.add_argument("--no-diverge", action="store_true", help="matches only")
    ap.add_argument("--dry-run", action="store_true", help="count only; write nothing")
    ap.add_argument("--no-marker", action="store_true", help="import, but leave the jobs unmarked (a scratch corpus)")
    ap.add_argument("--all", action="store_true", help="with --lab: also jobs that already have a marker")
    a = ap.parse_args(argv)
    if a.job and Path(a.job).is_dir():
        jobs = [Path(a.job)]
    elif a.lab and a.job:
        jobs = [a.lab / "runs" / a.job]
    elif a.lab:
        jobs = [j for j in pr.finished_jobs(a.lab) if a.all or not (j / MARKER).exists()]
    else:
        ap.error("give --lab or --job")
    jobs = [j for j in jobs if j.is_dir()]
    corpus = cp.Corpus(a.corpus)
    budget = None if a.budget_gb is None else int(a.budget_gb * GB)
    total: dict = {}
    state: dict = {}
    for j in jobs:
        c = import_job(j, corpus, budget, not a.no_diverge, a.dry_run, state=state, mark=not a.no_marker, max_fights=a.max_fights)
        print(f"{j.name}: {c}")
        for k, v in c.items():
            total[k] = total.get(k, 0) + v
    print("corpus-import:", "would add" if a.dry_run else "added", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
