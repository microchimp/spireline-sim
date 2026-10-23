"""Byte proof: two EmuDump builds write the same bytes for every corpus fight.

The proof a refactor of the oracle tools (the engine boundary, referee/design/rebuild-plan.md section 1)
owes: the old engine's output must not move by a byte. For each corpus fight and each mode below it
runs the A build and the B build (EmuDump.dll, the JIT build) over the same input and compares what each
wrote for that fight, as text, byte for byte. Nothing is judged and nothing is normalised except the
text of an `error` field, which holds a stack trace with build paths and line numbers: it is compared up to
its first line (and counted, so it shows).

    python tools/fidelity/byte_proof.py --a <A EmuDump.dll> --b <B EmuDump.dll> [--modes fight,netstate]
        [--procs N] [--only ID_OR_SEED ...] [--limit N] [--show N] [--work DIR] [--corpus DIR]

Exit 0 only when every fight of every mode is identical on both sides and no run failed.
A mode is a row of MODES: the EmuDump command and its flags. Later rounds add reanchor, hookprobe, refset and codex.

The fights are batched as corpus.hash_part batches them (`balance`, two parts per process), one EmuDump
process per part and side, in a spawn pool at BELOW_NORMAL priority.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pool workers import this module by path
    sys.path.insert(0, str(ROOT))

from tools.fidelity import corpus as C  # noqa: E402

# mode -> (EmuDump command, flags). The input file goes between the two.
MODES: dict[str, tuple[str, list[str]]] = {
    "fight": ("fight", ["--compact"]),
    "netstate": ("fight", ["--compact", "--netstate"]),
    "state-hash": ("fight", ["--compact", "--state-hash=fields"]),
    "rng-log": ("fight", ["--compact", "--rng-log"]),
}

_ERROR = re.compile(r'"error":"((?:[^"\\]|\\.)*)"')
_NEWLINE = re.compile(r"\\[rn]")
_DECODER = json.JSONDecoder()


def split_results(raw: bytes) -> list[str]:
    """The text of each element of the output's `results` array, as written (no re-serialising)."""
    text = raw.decode("utf-8")
    start = text.find('"results"')
    if start < 0:
        raise ValueError("no results array in the output")
    i = text.index("[", start) + 1
    out = []
    while True:
        while text[i] in " \t\r\n":
            i += 1
        if text[i] == "]":
            return out
        _, end = _DECODER.raw_decode(text, i)
        out.append(text[i:end])
        i = end
        while text[i] in " \t\r\n":
            i += 1
        if text[i] == ",":
            i += 1


def first_line_of_errors(piece: str) -> tuple[str, int]:
    """`piece` with each error text cut at its first line (a stack trace names build paths); how many were cut."""
    cut = 0

    def one(m):
        nonlocal cut
        first = _NEWLINE.split(m.group(1), maxsplit=1)[0]
        cut += first != m.group(1)
        return f'"error":"{first}"'

    return _ERROR.sub(one, piece), cut


def first_difference(x: str, y: str) -> int:
    n = min(len(x), len(y))
    i = next((k for k in range(n) if x[k] != y[k]), n)
    return i


def compare_outputs(ids: list, raw_a: bytes | None, raw_b: bytes | None) -> dict:
    """Per fight, whether the two outputs are identical. Both None-able: a side that wrote nothing is a problem, not a pass."""
    res = {"same": 0, "differ": [], "problems": [], "error_cut": 0}
    pieces = {}
    for side, raw in (("a", raw_a), ("b", raw_b)):
        if raw is None:
            res["problems"].append(f"{side}: no output")
            continue
        try:
            pieces[side] = split_results(raw)
        except (ValueError, IndexError) as ex:
            res["problems"].append(f"{side}: output is not a results array ({type(ex).__name__}: {ex})")
            continue
        if len(pieces[side]) != len(ids):
            res["problems"].append(f"{side}: {len(pieces[side])} results for {len(ids)} fights")
            del pieces[side]
    if len(pieces) < 2:
        return res
    for fid, pa, pb in zip(ids, pieces["a"], pieces["b"]):
        if pa == pb:
            res["same"] += 1
            res["error_cut"] += first_line_of_errors(pa)[1] > 0
            continue
        (pa, ca), (pb, cb) = first_line_of_errors(pa), first_line_of_errors(pb)
        if pa == pb:
            res["same"] += 1
            res["error_cut"] += 1
            continue
        at = first_difference(pa, pb)
        res["differ"].append({"id": fid, "at": at, "a": pa[max(0, at - 40):at + 60], "b": pb[max(0, at - 40):at + 60],
                              "len_a": len(pa), "len_b": len(pb),
                              "sha_a": hashlib.sha256(pa.encode()).hexdigest()[:12], "sha_b": hashlib.sha256(pb.encode()).hexdigest()[:12]})
    return res


def command_for(emudump) -> list:
    """The JIT build through dotnet, a NativeAOT .exe as it is; a .py file (a fake EmuDump in the tests) through Python."""
    if str(emudump).lower().endswith(".py"):
        return [sys.executable, str(emudump)]
    return C.emudump_command(emudump)


def run_side(emudump: str, command: str, flags: list, src: Path, dst: Path, timeout: float) -> tuple[bytes | None, str | None]:
    """One EmuDump over one input file: its stdout, or None and why it failed."""
    try:
        with dst.open("wb") as so:
            rc = subprocess.run([*command_for(emudump), command, str(src), *flags], stdout=so, stderr=subprocess.DEVNULL,
                                check=False, creationflags=C.BELOW_NORMAL, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout:.0f} s"
    if rc != 0:
        return None, f"exit {rc}"
    return dst.read_bytes(), None


def prove_batch(mode: str, ids: list, games: list, emudump_a: str, emudump_b: str, work: str, timeout: float | None = None) -> dict:
    """One worker: both builds over these fights in one mode, compared per fight."""
    command, flags = MODES[mode]
    timeout = timeout or 2 * C.part_timeout(sum(len(g.get("steps", [])) for g in games))
    tag = hashlib.sha256((mode + "|".join(map(str, ids))).encode()).hexdigest()[:10]
    src = Path(work) / f"bp-{tag}.in.json"
    src.write_text(json.dumps({"results": games}), encoding="utf-8")
    raws, why = {}, []
    t0 = time.time()
    for side, dll in (("a", emudump_a), ("b", emudump_b)):
        dst = Path(work) / f"bp-{tag}.{side}.out.json"
        raws[side], problem = run_side(dll, command, flags, src, dst, timeout)
        if problem:
            why.append(f"{side}: EmuDump {command} {problem}")
        with contextlib.suppress(OSError):
            dst.unlink()
    with contextlib.suppress(OSError):
        src.unlink()
    res = compare_outputs(ids, raws["a"], raws["b"])
    res["problems"] = why + [p for p in res["problems"] if not any(p.startswith(w[:2]) for w in why)]
    res["seconds"] = time.time() - t0
    res["mode"] = mode
    return res


def prove_corpus_batch(mode: str, ids: list, corpus_dir: str, emudump_a: str, emudump_b: str, work: str) -> dict:
    c = C.Corpus(Path(corpus_dir))
    return prove_batch(mode, ids, [c.read(i) for i in ids], emudump_a, emudump_b, work)


def prove(corpus: "C.Corpus", ids: list, modes: list, a: Path, b: Path, procs: int, work: Path) -> dict:
    """Every mode over every fight. Returns {mode: totals}."""
    metas = {m["id"]: m for m in corpus.load_index()["fights"]}
    parts = C.balance([metas[i] for i in ids], 2 * procs)
    totals = {m: {"fights": len(ids), "same": 0, "differ": [], "problems": [], "error_cut": 0, "seconds": 0.0} for m in modes}
    with ProcessPoolExecutor(max_workers=procs, mp_context=get_context("spawn"), initializer=C._below_normal) as ex:
        futures = [ex.submit(prove_corpus_batch, m, p, str(corpus.path), str(a), str(b), str(work)) for m in modes for p in parts]
        for f in futures:
            r = f.result()
            t = totals[r["mode"]]
            t["same"] += r["same"]
            t["differ"] += r["differ"]
            t["problems"] += r["problems"]
            t["error_cut"] += r["error_cut"]
            t["seconds"] += r["seconds"]
    return totals


def report(totals: dict, metas: dict, show: int) -> int:
    bad = 0
    for mode, t in totals.items():
        for d in t["differ"][:show]:
            seed = metas.get(d["id"], {}).get("seed")
            print(f"  [{mode}] {d['id']} {seed}: differs at char {d['at']} (lengths {d['len_a']} / {d['len_b']})")
            print(f"      a: ...{d['a']}")
            print(f"      b: ...{d['b']}")
        for p in t["problems"][:show]:
            print(f"  [{mode}] problem: {p}")
        n = len(t["differ"]) + len(t["problems"])
        bad += n + (t["fights"] == 0)
        print(f"byte_proof: {mode}: {t['fights']} fights, {t['same']} identical, {len(t['differ'])} differ, {len(t['problems'])} run problems"
              f" ({t['error_cut']} with an error text compared to its first line)")
    print(f"byte_proof: {'FAIL' if bad else 'ok'}: {sum(t['same'] for t in totals.values())} fight-mode pairs identical, "
          f"{sum(len(t['differ']) for t in totals.values())} differ")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, default=C.DEFAULT_CORPUS)
    ap.add_argument("--a", required=True, help="EmuDump.dll of the base build")
    ap.add_argument("--b", required=True, help="EmuDump.dll of the new build")
    ap.add_argument("--modes", default=",".join(MODES), help=f"comma list of {', '.join(MODES)}")
    ap.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--only", nargs="*", help="fight ids or seeds")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=10, help="differing fights to print per mode")
    ap.add_argument("--work", type=Path, help="folder for the temporary inputs and outputs (default: under the corpus's runs/)")
    a = ap.parse_args(argv)
    modes = [m for m in a.modes.split(",") if m]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise SystemExit(f"unknown mode(s) {unknown}; known: {list(MODES)}")
    corpus = C.Corpus(a.corpus)
    idx = corpus.load_index()["fights"]
    only = set(a.only or [])
    metas = [m for m in idx if not only or m["id"] in only or m["seed"] in only]
    metas, skipped = C.replayable(metas[: a.limit or None], C.emudump_features())
    ids = [m["id"] for m in metas]
    if skipped:
        print(f"skipped, this EmuDump does not apply what they need: {skipped}")
    work = a.work or corpus.path / "runs" / f"byteproof-{time.strftime('%Y%m%d-%H%M%S')}"
    work.mkdir(parents=True, exist_ok=True)
    totals = prove(corpus, ids, modes, Path(a.a), Path(a.b), a.procs, work)
    with contextlib.suppress(OSError):
        work.rmdir()
    return report(totals, {m["id"]: m for m in idx}, a.show)


if __name__ == "__main__":
    raise SystemExit(main())
