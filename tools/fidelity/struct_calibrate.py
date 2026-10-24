"""Calibrate struct_compare on an engine against the legacy comparers (referee/design/rebuild-plan.md section 1, "One comparer").

    python tools/fidelity/struct_calibrate.py corpus --emudump <EmuDump.dll> --out <calib.json> [--procs 8] [--limit N]
    python tools/fidelity/struct_calibrate.py traced --emudump <EmuDump.dll> --out <calib.json> <traced out.json> ...
    python tools/fidelity/struct_calibrate.py traced --engine core2 --switches F08,F10 --gate-against <old calib.json> ...

Each fight is replayed once (`EmuDump fight --compact --game-view`, plus --netstate for traced fights) and judged by
fight_compare.first_difference, netstate_diff.diff_fight (traced) and struct_compare.compare_fight on the same output.
The contingency table per legacy comparer counts fights flagged by both (at the same step, struct earlier, struct later),
by the legacy comparer only, by struct_compare only, and by neither. A fight the legacy comparer flags that struct_compare
does not flag at that step or earlier is a calibration failure to explain. --gate-against <earlier output> runs the census
gate: it fails when a fight was compared on fewer fields (schema paths) here than in the earlier output (engine B against A). Writes nothing into the repo; the output holds
fight ids, seeds, steps and field paths only.

--switches F08,F10 (core2 only; `none` for no switch, `all` for F01-F17): the switch set under test. The DLL reads its switches from the
migrated_families.json beside it, so the set is a scratch copy of the EmuDump build with that file rewritten (fuzz_core2.dll_for, the
same recipe the fuzzer uses; removed afterwards, the build itself is never touched), and struct_compare judges core2 with the set as
`migrated` (rows of a migrated family are no longer inherited from old). With --gate-against the run is also measured against the
earlier (old engine) output, fight by fight, against the game: better, same or worse by the first differing step of struct_compare
(then the number of fields differing there) and of netstate_diff. struct_compare is read twice for such a run: `sc` with the migrated
rows as above, and `sc_shared` with every old row inherited, the same rows old was judged by, so the verdict is a behaviour measure
and the stricter rows cannot make a fight look worse. The gate also fails when any fight is worse.

A comparer crash (struct_compare raising) has its own cell, struct_crash: not a flag, not a pass.
--self: the shadow of the engine against itself. The engine's own game view stands as the record and struct_compare judges
it against that same output: every fight must land in 'neither' with no difference (it checks the comparer's own
mechanics: flatten, rows, census), and exits 1 otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import math
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
from tools.fidelity import fuzz_core2 as FZ  # noqa: E402 - the switch set and its scratch build (one recipe for fuzz and calibrate)
from tools.fidelity import fight_compare as fc  # noqa: E402
from tools.fidelity import netstate_diff as nd  # noqa: E402
from tools.fidelity import struct_compare as sc  # noqa: E402


def self_record(g: dict, e: dict) -> dict:
    """The engine's own game view standing as the record (the shadow of an engine against itself). Its snapshot is each
    step's state and its net the step's net checkpoint; the actions are the record's; there are no checksums, so no step is
    left out of the net comparison. struct_compare of this record against the same engine output must find nothing."""
    steps = []
    for gs, es in zip(g["steps"], e.get("steps", [])):
        v = es.get("game_view") or {}
        st = {"action": gs.get("action"), "state": v.get("snapshot")}
        if gs.get("net") and v.get("net") is not None:
            st["net"] = [{"ctx": "self", "state": v["net"]}]
        steps.append(st)
    return {"seed": g.get("seed"), "steps": steps}


def _judge(fid: str, g: dict, e: dict, nm: dict, rows: list, traced: bool, self_check: bool = False, engine: str = "old",
           migrated: list | None = None) -> dict:
    out: dict = {"id": fid, "seed": g.get("seed")}
    fd = None if self_check else fc.first_difference(g, e, nm)
    out["fc"] = None if fd is None else fd[0]
    if fd is not None:
        out["fc_paths"] = [p for p, _, _ in fd[1][:8]]
    if traced and not self_check:
        r = nd.diff_fight(g, e, False)
        out["nd"] = r.get("step")
        if "step" in r:
            out["nd_paths"] = [f[0] for f in r["fields"][:8]]
        else:
            out["nd_verdict"] = r.get("verdict", "")[:120]
    if self_check:
        g = self_record(g, e)
    try:
        s = sc.compare_fight(g, e, rows, engine, migrated)
        out.update(sc=s["first"], sc_n=len(s["at"]), sc_paths=[p for p, _, _ in s["at"][:8]], sc_vals=[[p, x, y] for p, x, y in s["at"][:4]],
                   census=s["census"], waived=s["waived"], leaves=s["leaves"], allowed=s["allowed"],
                   surplus=s["surplus"])
        if migrated:  # the same rows old is judged by (nothing migrated): the verdict against old compares like with like
            t = sc.compare_fight(g, e, rows, engine, [])
            out.update(sc_shared=t["first"], sc_shared_n=len(t["at"]))
    except Exception as ex:  # noqa: BLE001 - a comparer crash is a finding, reported per fight
        out["sc"], out["sc_error"] = -1, f"{type(ex).__name__}: {ex}"[:300]
    if e.get("error"):
        out["ferr"] = str(e["error"]).splitlines()[0][:160]
    return out


def _part(args) -> list:
    ids, games, emudump, work, traced, self_check, engine, migrated = args
    C._below_normal()
    nm = fc.names(json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8")))
    rows = sc.load_rows()
    tag = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in ids[0])
    src, dst, err = (Path(work) / f"cal-{tag}{x}" for x in (".in.json", ".out.json", ".err.txt"))
    src.write_text(json.dumps({"results": games}), encoding="utf-8")
    flags = ["--compact", "--game-view"] + (["--netstate"] if traced else []) + ["--engine", engine]
    with dst.open("wb") as so, err.open("wb") as se:
        rc = subprocess.run([*C.emudump_command(emudump), "fight", str(src), *flags], stdout=so, stderr=se, check=False,
                            creationflags=C.BELOW_NORMAL, timeout=C.part_timeout(sum(len(g["steps"]) for g in games))).returncode
    res = json.loads(dst.read_text(encoding="utf-8")).get("results", []) if rc == 0 else []
    out = []
    for n, (fid, g) in enumerate(zip(ids, games)):
        e = res[n] if n < len(res) else {"error": f"EmuDump exit {rc}", "steps": []}
        out.append(_judge(fid, g, e, nm, rows, traced, self_check, engine, migrated))
    for f in (src, dst, err):
        with contextlib.suppress(OSError):
            f.unlink()
    return out


def traced_fight_id(f: str, k: int) -> str:
    """A fight id unique across traced files (they all share the name traced.out.json): the last three path parts and the index."""
    return "/".join(Path(f).parts[-3:]) + f"#{k}"


def table(results: list, key: str) -> dict:
    t = {"both_same_step": 0, "both_struct_earlier": 0, "both_struct_later": 0, "legacy_only": 0, "struct_only": 0, "neither": 0,
         "struct_crash": 0}
    for r in results:
        lg, s = r.get(key), r.get("sc")
        if s == -1:  # a comparer crash is its own cell: not a flag, not a pass
            t["struct_crash"] += 1
        elif lg is not None and s is not None:
            t["both_same_step" if s == lg else "both_struct_earlier" if s < lg else "both_struct_later"] += 1
        elif lg is not None:
            t["legacy_only"] += 1
        elif s is not None:
            t["struct_only"] += 1
        else:
            t["neither"] += 1
    return t


def resolve_switches(engine: str, text: str | None) -> list[str] | None:
    """The switch set of `--switches` (fuzz_core2.parse_switches: `F08,F10`, `none`, `all`), or None when the flag is absent (the
    build's own migrated_families.json stands). A switch set is core2's: old has no switches, so the flag with another engine is
    refused rather than ignored (a run that looked switched on and was not would read as 'no change')."""
    if text is None:
        return None
    if engine != "core2":
        raise ValueError(f"--switches needs --engine core2 (got --engine {engine})")
    return FZ.parse_switches(text)


def standing(r: dict, key: str, n_key: str | None = None) -> tuple:
    """How far a fight got against the game, as an ordering (greater is better): the step of its first difference (no difference
    is past every step), then fewer fields differing at that step. A result without the field count (an older output) has none:
    the step alone orders it, and a tie on the step is 'same'."""
    step = r.get(key)
    if step is None:
        return (math.inf, 0)
    return (step, -(r.get(n_key) or 0) if n_key else 0)


def verdict(old: dict, new: dict, key: str, n_key: str | None = None, new_key: str | None = None, new_n_key: str | None = None) -> str:
    a, b = standing(old, key, n_key), standing(new, new_key or key, new_n_key or n_key)
    if n_key and (old.get(n_key) is None or new.get(new_n_key or n_key) is None):
        a, b = a[:1], b[:1]  # one side has no field count: the step alone
    return "better" if b > a else "worse" if b < a else "same"


def compare_runs(base: dict, mine: dict) -> dict:
    """Each fight of two runs of one traced set (A = the earlier run, B = this one) against the game: the verdict of B relative to
    A by struct_compare (`sc_shared` of B when it has one, the rows A was judged by, else `sc`) and by netstate_diff (`nd`).
    {"fights": {id: {"sc": [A step, B step, verdict], "nd": [...]}}, "counts": {"sc": {better, same, worse}, "nd": {...}},
    "missing": [ids of A that B lacks]}."""
    fights: dict = {}
    counts = {k: {"better": 0, "same": 0, "worse": 0} for k in ("sc", "nd")}
    for fid, a in base.items():
        b = mine.get(fid)
        if b is None or "census" not in a or "census" not in b:
            continue
        sk = "sc_shared" if "sc_shared" in b else "sc"
        v_sc = verdict(a, b, "sc", "sc_n", sk, sk + "_n")
        entry = {"sc": [a.get("sc"), b.get(sk), v_sc]}
        counts["sc"][v_sc] += 1
        if "nd" in a or "nd" in b:
            v_nd = verdict(a, b, "nd")
            entry["nd"] = [a.get("nd"), b.get("nd"), v_nd]
            counts["nd"][v_nd] += 1
        fights[fid] = entry
    return {"fights": fights, "counts": counts, "missing": sorted(fid for fid in base if fid not in mine)}


def exposed(mine: dict) -> int:
    """Fights whose rows of the migrated families (compared on core2 from the switch on) flag a step earlier than the shared rows do:
    what the switch exposes that old's waivers hid. Informational, not a gate."""
    return sum(1 for r in mine.values() if "sc_shared" in r and r.get("sc") is not None
               and (r["sc_shared"] is None or r["sc"] < r["sc_shared"]))


def run_jobs(jobs: list, procs: int) -> list:
    """The jobs over a spawn pool (one emulator per process), their results in one list."""
    with mp.get_context("spawn").Pool(min(procs, len(jobs) or 1)) as pool:
        return [r for part in pool.imap_unordered(_part, jobs) for r in part]


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("mode", choices=("corpus", "traced"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--emudump", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--corpus", type=Path, default=C.DEFAULT_CORPUS)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--self", dest="self_check", action="store_true",
                    help="shadow of the engine against itself: its own game view stands as the record; every fight must be identical")
    ap.add_argument("--engine", choices=("old", "core2"), default="old",
                    help="the engine EmuDump runs and whose allowed rows apply (core2 also gets old's: it runs over old's state)")
    ap.add_argument("--switches", help="core2 only: the switch set under test (F01,F02,... / none / all); the DLL is a scratch copy of "
                    "the build with that set, and core2 is judged with it as the migrated families (default: the build's own file)")
    ap.add_argument("--gate-against", type=Path, help="an earlier output (engine A): fail any fight compared on fewer fields here "
                    "or worse against the game")
    a = ap.parse_args(argv)
    try:
        switches = resolve_switches(a.engine, a.switches)
    except ValueError as ex:
        print(ex, file=sys.stderr)
        return 2
    work = a.work or a.out.parent
    work.mkdir(parents=True, exist_ok=True)
    built = FZ.dll_for(Path(a.emudump), switches, work) if switches is not None else Path(a.emudump)  # once, here: the workers get this path
    try:
        return run(a, work, built, switches)
    finally:
        FZ.drop_build(built, Path(a.emudump))


def run(a: argparse.Namespace, work: Path, built: Path, switches: list[str] | None) -> int:
    emudump = str(built)
    jobs = []
    if a.mode == "corpus":
        c = C.Corpus(a.corpus)
        metas = c.load_index()["fights"][: a.limit or None]
        for ids in C.balance(metas, a.procs * 4):
            jobs.append((ids, [c.read(i) for i in ids], emudump, str(work), False, a.self_check, a.engine, switches))
    else:
        fights = []
        for f in a.files:
            for k, r in enumerate(json.loads(Path(f).read_text(encoding="utf-8"))["results"]):
                if r.get("ascension") is not None and any("net" in s for s in r.get("steps", [])):
                    fights.append((traced_fight_id(f, k), r))
        fights = fights[: a.limit or None]
        per = max(1, len(fights) // (a.procs * 2))
        for i in range(0, len(fights), per):
            chunk = fights[i:i + per]
            jobs.append(([x[0] for x in chunk], [x[1] for x in chunk], emudump, str(work), True, a.self_check, a.engine, switches))
    t0 = time.time()
    results = run_jobs(jobs, a.procs)
    results.sort(key=lambda r: r["id"])
    if a.self_check:
        tables = {"self": table(results, "fc")}  # no legacy flag: neither is a pass, struct_only a difference
    else:
        tables = {"fight_compare": table(results, "fc")}
        if a.mode == "traced":
            tables["netstate_diff"] = table(results, "nd")
    allowed: dict = {}
    for r in results:
        for k, v in (r.get("allowed") or {}).items():
            allowed[k] = allowed.get(k, 0) + v
    doc = {"mode": a.mode, "fights": len(results), "seconds": round(time.time() - t0), "allowed_sha": sc.rows_sha(),
           "engine": a.engine, "switches": switches, "tables": tables, "allowed_fired": allowed,
           "surplus": sorted({p for r in results for p in r.get("surplus", [])}), "sc_errors": sum("sc_error" in r for r in results), "results": results}
    a.out.write_text(json.dumps(doc), encoding="utf-8")
    print(json.dumps({k: v for k, v in doc.items() if k != "results"}, indent=1))
    if a.self_check:
        bad = [r for r in results if r.get("sc") is not None]
        for r in bad[:10]:
            print(f"self-check: {r['id']}: step {r['sc']}: {r.get('sc_paths') or r.get('sc_error')}")
        print(f"self-check: {len(bad)} of {len(results)} fights differ from their own game view")
        return 1 if bad else 0
    if a.gate_against:
        base = {r["id"]: r for r in json.loads(a.gate_against.read_text(encoding="utf-8"))["results"] if "census" in r}
        mine = {r["id"]: r for r in results if "census" in r}
        bad = sc.census_gate(base, mine)
        for fid, missing in list(bad.items())[:20]:
            print(f"census gate: {fid}: not compared on {', '.join(missing[:6])}")
        print(f"census gate: {len(bad)} of {len(base)} fights compared on fewer fields than {a.gate_against.name}")
        print(f"census gate: waived schema paths summed over fights: earlier {sc.waived_total(base)}, here {sc.waived_total(mine)}")
        cmp = compare_runs({r["id"]: r for r in json.loads(a.gate_against.read_text(encoding="utf-8"))["results"]}, mine)
        for fid, e in cmp["fights"].items():
            if e["sc"][2] != "same" or e.get("nd", ("", "", "same"))[2] != "same":
                print(f"against the game: {fid}: struct step {e['sc'][0]} -> {e['sc'][1]} ({e['sc'][2]})"
                      + (f", netstate step {e['nd'][0]} -> {e['nd'][1]} ({e['nd'][2]})" if "nd" in e else ""))
        doc["against_game"] = {"counts": cmp["counts"], "missing": cmp["missing"], "exposed_by_migrated_rows": exposed(mine)}
        a.out.write_text(json.dumps({**doc, "verdicts": cmp["fights"]}), encoding="utf-8")
        for k, name in (("sc", "struct_compare"), ("nd", "netstate_diff")):
            c = cmp["counts"][k]
            print(f"against the game, {name}: {c['better']} better, {c['same']} same, {c['worse']} worse of {sum(c.values())} fights")
        print(f"against the game: {exposed(mine)} fights flagged earlier by the rows of the migrated families than by the shared rows; "
              f"{len(cmp['missing'])} fights of the earlier output missing here")
        worse = sum(c["worse"] for c in cmp["counts"].values())
        return 1 if bad or worse or cmp["missing"] else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
