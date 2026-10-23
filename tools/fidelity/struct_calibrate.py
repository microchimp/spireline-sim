"""Calibrate struct_compare on an engine against the legacy comparers (referee/design/rebuild-plan.md section 1, "One comparer").

    python tools/fidelity/struct_calibrate.py corpus --emudump <EmuDump.dll> --out <calib.json> [--procs 8] [--limit N]
    python tools/fidelity/struct_calibrate.py traced --emudump <EmuDump.dll> --out <calib.json> <traced out.json> ...

Each fight is replayed once (`EmuDump fight --compact --game-view`, plus --netstate for traced fights) and judged by
fight_compare.first_difference, netstate_diff.diff_fight (traced) and struct_compare.compare_fight on the same output.
The contingency table per legacy comparer counts fights flagged by both (at the same step, struct earlier, struct later),
by the legacy comparer only, by struct_compare only, and by neither. A fight the legacy comparer flags that struct_compare
does not flag at that step or earlier is a calibration failure to explain. --gate-against <earlier output> runs the census
gate: it fails when a fight was compared on fewer fields (schema paths) here than in the earlier output (engine B against A). Writes nothing into the repo; the output holds
fight ids, seeds, steps and field paths only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fidelity import corpus as C  # noqa: E402
from tools.fidelity import fight_compare as fc  # noqa: E402
from tools.fidelity import netstate_diff as nd  # noqa: E402
from tools.fidelity import struct_compare as sc  # noqa: E402


def _judge(fid: str, g: dict, e: dict, nm: dict, rows: list, traced: bool) -> dict:
    out: dict = {"id": fid, "seed": g.get("seed")}
    fd = fc.first_difference(g, e, nm)
    out["fc"] = None if fd is None else fd[0]
    if fd is not None:
        out["fc_paths"] = [p for p, _, _ in fd[1][:8]]
    if traced:
        r = nd.diff_fight(g, e, False)
        out["nd"] = r.get("step")
        if "step" in r:
            out["nd_paths"] = [f[0] for f in r["fields"][:8]]
        else:
            out["nd_verdict"] = r.get("verdict", "")[:120]
    try:
        s = sc.compare_fight(g, e, rows)
        out.update(sc=s["first"], sc_paths=[p for p, _, _ in s["at"][:8]], sc_vals=[[p, x, y] for p, x, y in s["at"][:4]],
                   census=s["census"], leaves=s["leaves"], allowed=s["allowed"])
    except Exception as ex:  # noqa: BLE001 - a comparer crash is a finding, reported per fight
        out["sc"], out["sc_error"] = -1, f"{type(ex).__name__}: {ex}"[:300]
    if e.get("error"):
        out["ferr"] = str(e["error"]).splitlines()[0][:160]
    return out


def _part(args) -> list:
    ids, games, emudump, work, traced = args
    C._below_normal()
    nm = fc.names(json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8")))
    rows = sc.load_rows()
    src, dst, err = (Path(work) / f"cal-{ids[0]}{x}" for x in (".in.json", ".out.json", ".err.txt"))
    src.write_text(json.dumps({"results": games}), encoding="utf-8")
    flags = ["--compact", "--game-view"] + (["--netstate"] if traced else [])
    with dst.open("wb") as so, err.open("wb") as se:
        rc = subprocess.run([*C.emudump_command(emudump), "fight", str(src), *flags], stdout=so, stderr=se, check=False,
                            creationflags=C.BELOW_NORMAL, timeout=C.part_timeout(sum(len(g["steps"]) for g in games))).returncode
    res = json.loads(dst.read_text(encoding="utf-8")).get("results", []) if rc == 0 else []
    out = []
    for n, (fid, g) in enumerate(zip(ids, games)):
        e = res[n] if n < len(res) else {"error": f"EmuDump exit {rc}", "steps": []}
        out.append(_judge(fid, g, e, nm, rows, traced))
    for f in (src, dst, err):
        with contextlib.suppress(OSError):
            f.unlink()
    return out


def table(results: list, key: str) -> dict:
    t = {"both_same_step": 0, "both_struct_earlier": 0, "both_struct_later": 0, "legacy_only": 0, "struct_only": 0, "neither": 0}
    for r in results:
        lg, s = r.get(key), r.get("sc")
        if lg is not None and s is not None:
            t["both_same_step" if s == lg else "both_struct_earlier" if s < lg else "both_struct_later"] += 1
        elif lg is not None:
            t["legacy_only"] += 1
        elif s is not None:
            t["struct_only"] += 1
        else:
            t["neither"] += 1
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("mode", choices=("corpus", "traced"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--emudump", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--corpus", type=Path, default=C.DEFAULT_CORPUS)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gate-against", type=Path, help="an earlier output (engine A): fail any fight compared on fewer fields here")
    a = ap.parse_args()
    work = a.work or a.out.parent
    work.mkdir(parents=True, exist_ok=True)
    jobs = []
    if a.mode == "corpus":
        c = C.Corpus(a.corpus)
        metas = c.load_index()["fights"][: a.limit or None]
        for ids in C.balance(metas, a.procs * 4):
            jobs.append((ids, [c.read(i) for i in ids], a.emudump, str(work), False))
    else:
        fights = []
        for f in a.files:
            for k, r in enumerate(json.loads(Path(f).read_text(encoding="utf-8"))["results"]):
                if r.get("ascension") is not None and any("net" in s for s in r.get("steps", [])):
                    fights.append((f"{Path(f).name}#{k}", r))
        fights = fights[: a.limit or None]
        per = max(1, len(fights) // (a.procs * 2))
        for i in range(0, len(fights), per):
            chunk = fights[i:i + per]
            jobs.append(([x[0] for x in chunk], [x[1] for x in chunk], a.emudump, str(work), True))
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(a.procs, len(jobs) or 1)) as pool:
        results = [r for part in pool.imap_unordered(_part, jobs) for r in part]
    results.sort(key=lambda r: r["id"])
    tables = {"fight_compare": table(results, "fc")}
    if a.mode == "traced":
        tables["netstate_diff"] = table(results, "nd")
    allowed: dict = {}
    for r in results:
        for k, v in (r.get("allowed") or {}).items():
            allowed[k] = allowed.get(k, 0) + v
    doc = {"mode": a.mode, "fights": len(results), "seconds": round(time.time() - t0), "allowed_sha": sc.rows_sha(),
           "tables": tables, "allowed_fired": allowed, "sc_errors": sum("sc_error" in r for r in results), "results": results}
    a.out.write_text(json.dumps(doc), encoding="utf-8")
    print(json.dumps({k: v for k, v in doc.items() if k != "results"}, indent=1))
    if a.gate_against:
        base = {r["id"]: r for r in json.loads(a.gate_against.read_text(encoding="utf-8"))["results"] if "census" in r}
        bad = sc.census_gate(base, {r["id"]: r for r in results if "census" in r})
        for fid, missing in list(bad.items())[:20]:
            print(f"census gate: {fid}: not compared on {', '.join(missing[:6])}")
        print(f"census gate: {len(bad)} of {len(base)} fights compared on fewer fields than {a.gate_against.name}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
