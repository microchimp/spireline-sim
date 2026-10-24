"""Diff two emulator builds' RNG draws, fight by fight: same counts are not enough.

    python -m tools.fidelity.rng_diff --a <old EmuDump.dll> --b <new EmuDump.dll> [--corpus DIR] [--only ID ...]
        [--limit N] [--procs 8] [--show 20]     the corpus fights this checkout's EmuDump can replay (tools/fidelity/corpus.py)
    python -m tools.fidelity.rng_diff --a <dll> --b <dll> --fights <referee out.json> [...]    fights from files
    python -m tools.fidelity.rng_diff --logs <a.json> <b.json>   two `EmuDump fight --rng-log` outputs

`EmuDump fight --rng-log` records every draw of every combat stream as [stream, index, caller] in the
step it happened (tools/emudump/Engines/Old/OldRngLog.cs; index is the stream's counter after the draw). Per fight and
stream the draws are compared in order by (step, caller, index). A fight differs when a draw moved (the
same stream count, another caller, step or stream position: what a per-stream count diff cannot see) or
one build makes a draw the other does not (count). The first difference per fight is printed with both
sides' step, caller and index.

What it cannot see: the caller is two engine frames, not the creature or card a draw is for, so two
enemies' rolls swapped under the same caller look the same. corpus.py identity (the full-state hash)
catches those; a structure-only refactor needs both.

Which fights are compared. From the corpus: every fight `corpus.py identity` replays, the ones with a console
setup or potions among them (47 draw sites, by caller, occur in no other fight on 2026-10-14: potion rolls, the
creation of Inklet, Fossil Stalker, Exoskeleton, Scrolls, Decimillipede and others). A fight whose spec needs a
feature this checkout's EmuDump source does not apply (corpus.FEATURES; enemy_hp_mult, 80 fights, on that date) is
not replayed: its count and the feature are printed as NOT COMPARED and repeated in the last line, never dropped in
silence, and its draws are unchecked. From --fights: a record with no steps or no ascension is counted the same way.
--limit and --only narrow the set on request and are not reported as left out.

EmuDump runs with DOTNET_TieredCompilation=0 and DOTNET_JITMinOpts=1 so the JIT inlines nothing:
otherwise a caller can vanish from the stack in one build and not the other. Exit 1 when any fight
differs or a replay failed on one side only.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

NO_INLINING = {"DOTNET_TieredCompilation": "0", "DOTNET_JITMinOpts": "1", "DOTNET_TieredPGO": "0"}
# The EmuDump flags whose output carries a caller label for each RNG draw (tools/emudump/Engines/Old/OldRngLog.cs, OldTrace.cs).
# The label is read off the stack, which tiered JIT inlining changes from run to run, so a run with one of them uses NO_INLINING.
# Defined once, here: tools/fidelity/packet.py and referee/trace_divergent.py import it.
LABELLED_FLAGS = ("--rng-log", "--trace", "--trace-to")
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)


def draws(fight: dict) -> dict:
    """{stream: [(step, caller, index), ...]} in draw order. The index is the stream's position after the draw,
    so a stream that one build resets, copies or advances outside the log differs too."""
    out: dict = collections.defaultdict(list)
    for j, s in enumerate(fight.get("steps", [])):
        for stream, index, caller in s.get("rng_log", []):
            out[stream].append((j, caller, index))
    return dict(out)


def diff_fight(a: dict, b: dict) -> dict | None:
    """None when both builds made the same draws at the same places; else the first difference."""
    if bool(a.get("error")) != bool(b.get("error")):
        return {"kind": "error", "a": (a.get("error") or "")[:160], "b": (b.get("error") or "")[:160]}
    da, db = draws(a), draws(b)
    first = None
    counts = {}
    for stream in sorted(set(da) | set(db)):
        xa, xb = da.get(stream, []), db.get(stream, [])
        if len(xa) != len(xb):
            counts[stream] = (len(xa), len(xb))
        for n, (p, q) in enumerate(zip(xa, xb)):
            if p != q:
                cand = {"stream": stream, "draw": n + 1, "a": p, "b": q}
                break
        else:
            if len(xa) == len(xb):
                continue
            n = min(len(xa), len(xb))
            cand = {"stream": stream, "draw": n + 1, "a": xa[n] if n < len(xa) else None, "b": xb[n] if n < len(xb) else None}
        step = min(x[0] for x in (cand["a"], cand["b"]) if x)
        if first is None or step < first["at_step"]:
            first = dict(cand, at_step=step)
    if first is None:
        return None
    first["kind"] = "count" if counts else "moved"
    first["counts"] = counts
    return first


def compare(fights_a: list, fights_b: list) -> dict:
    res = {"fights": len(fights_a), "same": 0, "moved": [], "count": [], "error": [], "missing": 0}
    for k, a in enumerate(fights_a):
        b = fights_b[k] if k < len(fights_b) else None
        if b is None:
            res["missing"] += 1
            continue
        d = diff_fight(a, b)
        if d is None:
            res["same"] += 1
        else:
            res[d["kind"]].append((a.get("seed"), d))
    return res


def run_emudump(dll: str, fights: list, work: Path, procs: int, tag: str) -> list:
    """`EmuDump fight --rng-log --compact` over fights, split across processes; results in input order."""
    if not fights:
        return []
    k = max(1, min(procs, len(fights)))
    size = -(-len(fights) // k)
    env = dict(os.environ, **NO_INLINING)
    jobs = []
    for i in range(0, len(fights), size):
        part = fights[i:i + size]
        src, dst = work / f"{tag}.{i}.in.json", work / f"{tag}.{i}.out.json"
        src.write_text(json.dumps({"results": part}), encoding="utf-8")
        jobs.append((src, dst, len(part)))

    def one(job):
        src, dst, n = job
        timeout = 120 + 0.3 * sum(len(f.get("steps", [])) for f in json.loads(src.read_text(encoding="utf-8"))["results"])
        try:
            with dst.open("wb") as so:
                subprocess.run(["dotnet", dll, "fight", str(src), "--rng-log", "--compact"], stdout=so, stderr=subprocess.DEVNULL,
                               env=env, timeout=timeout, check=False, creationflags=BELOW_NORMAL)
            got = json.loads(dst.read_text(encoding="utf-8")).get("results", [])
        except (subprocess.TimeoutExpired, ValueError, OSError) as ex:
            got = [{"error": f"EmuDump {type(ex).__name__}"}] * n
        for f in (src, dst):
            f.unlink(missing_ok=True)
        return got + [{"error": "no result"}] * (n - len(got))

    with ThreadPoolExecutor(len(jobs)) as ex:
        parts = list(ex.map(one, jobs))
    return [r for p in parts for r in p]


def load_fights(a) -> tuple[list, dict]:
    """(the fights to replay, {reason: count} of the fights left out). Nothing is left out unsaid: report() prints it."""
    left: collections.Counter = collections.Counter()
    if a.fights:
        out = []
        for p in a.fights:
            for r in json.loads(Path(p).read_text(encoding="utf-8"))["results"]:
                if not isinstance(r, dict):
                    left["not a fight record"] += 1
                elif not r.get("steps"):
                    left["no recorded steps"] += 1
                elif r.get("ascension") is None:
                    left["no ascension"] += 1
                else:
                    out.append(r)
        return out[: a.limit or None], dict(left)
    from tools.fidelity.corpus import DEFAULT_CORPUS, Corpus, emudump_features, replayable  # noqa: PLC0415
    c = Corpus(Path(a.corpus or DEFAULT_CORPUS))
    metas = [m for m in c.load_index()["fights"] if not a.only or m["id"] in a.only or m["seed"] in a.only]
    metas, skipped = replayable(metas, emudump_features())  # as corpus.py identity: setup fights are fights
    for needs, n in skipped.items():
        left[f"needs {needs}, which this checkout's EmuDump source does not apply"] += n
    return [c.read(m["id"]) for m in metas[: a.limit or None]], dict(left)


def report(res: dict, show: int, left: dict | None = None) -> int:
    for kind, title in (("moved", "draws moved (same counts)"), ("count", "draws added or removed"), ("error", "replay error on one side")):
        items = res[kind]
        print(f"{title}: {len(items)} fights")
        for seed, d in items[:show]:
            if kind == "error":
                print(f"  {seed}: a: {d['a'] or 'ok'} | b: {d['b'] or 'ok'}")
                continue
            def side(x):
                return "no draw" if x is None else f"step {x[0]} {x[1]} (index {x[2]})"
            counts = "; counts " + ", ".join(f"{s} {x}->{y}" for s, (x, y) in d["counts"].items()) if d["counts"] else ""
            print(f"  {seed}: {d['stream']} draw {d['draw']}: a {side(d['a'])} | b {side(d['b'])}{counts}")
    bad = len(res["moved"]) + len(res["count"]) + len(res["error"]) + res["missing"]
    not_compared = ""
    if left:
        not_compared = f", NOT COMPARED {sum(left.values())} ({'; '.join(f'{n} {why}' for why, n in sorted(left.items()))})"
        print(f"NOT COMPARED: {sum(left.values())} fights left out of this check, so their draws are unchecked: "
              + "; ".join(f"{n} {why}" for why, n in sorted(left.items())))
    print(f"rng_diff: {'FAIL' if bad or not res['fights'] else 'ok'}: {res['fights']} fights, {res['same']} draw for draw the same, "
          f"{len(res['moved'])} moved, {len(res['count'])} counts changed, {len(res['error'])} errors, {res['missing']} missing"
          f"{not_compared}")
    return 1 if bad or not res["fights"] else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--a", help="EmuDump.dll of the old build")
    ap.add_argument("--b", help="EmuDump.dll of the new build")
    ap.add_argument("--logs", nargs=2, help="two EmuDump fight --rng-log outputs instead of running builds")
    ap.add_argument("--corpus")
    ap.add_argument("--fights", nargs="*", help="referee out.json files instead of the corpus")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--procs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--show", type=int, default=20)
    ap.add_argument("--work")
    a = ap.parse_args(argv)
    if a.logs:
        fa, fb = (json.loads(Path(p).read_text(encoding="utf-8"))["results"] for p in a.logs)
        return report(compare(fa, fb), a.show)
    if not (a.a and a.b):
        ap.error("--a and --b (two EmuDump.dll builds), or --logs")
    fights, left = load_fights(a)
    work = Path(a.work or tempfile.mkdtemp(prefix="rng_diff_"))
    work.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(2) as ex:
        ra = ex.submit(run_emudump, a.a, fights, work, a.procs, "a")
        rb = ex.submit(run_emudump, a.b, fights, work, a.procs, "b")
        res = compare(ra.result(), rb.result())
    return report(res, a.show, left)


if __name__ == "__main__":
    sys.exit(main())
