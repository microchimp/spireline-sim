"""How many of a mutation run's surviving mutants does the hook table kill?

    python -m tools.fidelity.hook_table mutants --run <mutation run dir> --work <scratch> [--json out.json]

The mutation run (tools/fidelity/mutate.py, ``schemata``) seeded bugs into the combat core at its own commit
and replayed the recorded fights; the survivors are the bugs those fights did not catch. Each survivor is
carried onto this checkout's engine by its line text (the line it changed, matched with the lines around it in
the run's commit; a line that is gone or changed is reported as not carried over), then judged twice:

  static  the mutant applied as text, the table rebuilt (``check``): killed when any (model, hook) row
          changes status or dispatch point;
  order   every carried mutant compiled into one instrumented engine behind ``MutCov.Is(id)`` (the
          schemata form), each scenario run once per mutant: killed when the set of order inversions, the
          gate keys or the harness errors differ from the same build with no mutant active, or the run hangs.

The kill rate is over the carried survivors; ``gate`` counts only the kills the checked-in gate itself would
report (a new or vanished open-list key, a new error, a new MISSING row).
"""

from __future__ import annotations

import collections
import json
import re
import subprocess
from pathlib import Path

from . import hook_order as ho
from . import hook_table as ht
from . import mutate as mu
from . import schemata as sm

MUTCOV = """// Written by tools/fidelity/hook_mutants.py into its scratch copy of the engine only: the active mutant.
namespace Sts2Emulator;

public static class MutCov
{
    public static int Active = -1;

    public static bool Is(int id) => Active == id;
}
"""
IDENT = re.compile(r"[A-Za-z_]\w*")


def survivors(path: Path) -> list[dict]:
    return [r for r in (json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip())
            if r.get("status") == "survived"]


def _git_show(root: Path, commit: str, rel: str) -> str | None:
    r = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{rel}"], stdout=subprocess.PIPE,
                       stderr=subprocess.DEVNULL)
    return r.stdout.decode("utf-8").replace("\r\n", "\n") if r.returncode == 0 else None


def carry(recs: list[dict], texts: dict[str, str], root: Path, commit: str) -> tuple[list, dict]:
    """Each survivor on this engine's text: [(record, Mutant)], and {id: why not} for the rest."""
    old_files: dict[str, str | None] = {}
    carried, lost = [], {}
    taken: set[tuple[str, int]] = set()
    for rec in recs:
        rel = rec["file"]
        cur = texts.get(rel)
        if cur is None:
            lost[rec["id"]] = "file gone"
            continue
        if rel not in old_files:
            old_files[rel] = _git_show(root, commit, f"{mu.ENGINE}/{rel}")
        old = (old_files[rel] or "").split("\n")
        want = rec["old"].split("\n")
        lines = cur.split("\n")
        span = len(want)
        cands = [i for i in range(len(lines) - span + 1) if lines[i:i + span] == want]
        if not cands:
            lost[rec["id"]] = "line changed since the run"
            continue
        at = rec["line"] - 1

        def score(i: int, at=at, lines=lines, old=old, span=span) -> tuple:
            same = sum(1 for k in range(-4, span + 4)
                       if 0 <= i + k < len(lines) and 0 <= at + k < len(old) and lines[i + k] == old[at + k])
            return same, -abs(i - at)

        best = max(cands, key=score)
        span_lines = {(rel, best + 1 + k) for k in range(span)}
        if span_lines & taken:
            lost[rec["id"]] = "shares a line with another survivor"
            continue
        taken |= span_lines
        m = mu.Mutant(id=rec["id"], file=rel, line=best + 1, op=rec["op"], old=rec["old"], new=rec["new"],
                      unit=rec.get("unit"), fights=rec.get("fights", 0), extra=rec.get("extra", {}))
        carried.append((rec, m))
    return carried, lost


def _signature(res: dict) -> dict[str, tuple]:
    return {f"{ht.short(r['model'])}.{r['hook']}": (r["status"], tuple(sorted(r["where"]))) for r in res["rows"]}


def static_kills(a, engine: ht.Engine, carried: list, log=print) -> dict[str, dict]:
    """The table rebuilt with each mutant applied as text. A mutant that changes no identifier (a flipped
    comparison, a literal) leaves every name where it was, so the table cannot change; it is not rebuilt."""
    args = (a.decompiled, ht.load_json(a.dispatch), ht.load_json(a.unmodelled), ht.read_open(a.open))
    base = ht.check(args[0], engine, *args[1:])
    sig0 = _signature(base)
    gate0 = set(base["problems"])
    out = {}
    for n, (rec, m) in enumerate(carried):
        text = engine.files[m.file]
        new = mu.apply(text, m)
        if collections.Counter(IDENT.findall(m.old)) == collections.Counter(IDENT.findall(m.new)):
            out[m.id] = {"killed": False, "gate": False, "why": "no name changed"}
            continue
        texts = dict(engine.files)
        texts[m.file] = new
        e2 = ht.Engine.load(engine.root, texts)
        res = ht.check(args[0], e2, *args[1:])
        sig = _signature(res)
        diff = sorted(k for k in sig0 if sig0[k] != sig.get(k))
        gate = set(res["problems"]) != gate0
        out[m.id] = {"killed": bool(diff), "gate": gate, "rows": diff[:6]}
        if n % 10 == 0:
            log(f"  static {n + 1}/{len(carried)}", flush=True)
    return out


def guarded_engine(engine: ht.Engine, carried: list, dropped: set[int]) -> tuple[ht.Engine, dict, dict]:
    """The engine with every carried mutant behind MutCov.Is(gid); (engine, {gid: mutant}, {gid: why not})."""
    by_file = collections.defaultdict(list)
    gids = {}
    for rec, m in carried:
        g = rec["gid"]
        if g in dropped:
            continue
        by_file[m.file].append((g, m))
        gids[g] = m
    texts = dict(engine.files)
    skipped = {}
    for rel, muts in by_file.items():
        texts[rel], sk = sm.guard(engine.files[rel], muts)
        skipped.update(sk)
    texts["Core/MutCov.cs"] = MUTCOV
    return ht.Engine.load(engine.root, texts), gids, skipped


def order_kills(a, engine: ht.Engine, carried: list, work: Path, log=print) -> tuple[dict, dict]:
    """Every carried mutant in one instrumented build, each run on every scenario."""
    dropped: set[int] = set()
    why_not: dict[int, str] = {}
    for attempt in range(12):
        eg, gids, skipped = guarded_engine(engine, carried, dropped)
        for g, w in skipped.items():
            why_not[g] = w
            dropped.add(g)
        res = ht.check(a.decompiled, eg, ht.load_json(a.dispatch), ht.load_json(a.unmodelled), ht.read_open(a.open))
        rows = [ht.Row(**r) for r in res["rows"]]
        try:
            rig = ht.prepare_order(eg, rows, work, log=log)
            break
        except ho.BuildFailed as e:
            bad = set()
            for f, line in ho.ERROR_LINE.findall(e.output):
                fp = Path(f).as_posix()
                for g, m in gids.items():
                    if fp.endswith(m.file) and int(line) in (m.line, m.line + 1) and g not in dropped:
                        bad.add(g)
            if not bad:
                raise
            log(f"  order build {attempt + 1}: {len(bad)} guarded mutants do not compile; dropped", flush=True)
            for g in bad:
                why_not[g] = "guarded form does not compile"
            dropped |= bad
    else:
        raise RuntimeError("guarded build did not converge")
    live = [g for g in gids if g not in dropped]
    results = ho.run_harness(rig.copy, rig.scenarios, work, mutants=[-1])
    base = ht.judge_order(a.decompiled, rig, results)[-1]
    verdicts: dict[int, dict] = {}

    def run(chunk: list[int], timeout: float) -> None:
        try:
            got = ht.judge_order(a.decompiled, rig, ho.run_harness(rig.copy, rig.scenarios, work, mutants=chunk,
                                                                  timeout=timeout))
        except subprocess.TimeoutExpired:
            if len(chunk) == 1:
                verdicts[chunk[0]] = {"killed": True, "gate": True, "why": "hang"}
                return
            for g in chunk:
                run([g], 30.0)
            return
        except RuntimeError as e:
            if len(chunk) == 1:
                verdicts[chunk[0]] = {"killed": True, "gate": True, "why": f"harness crashed: {str(e)[:120]}"}
                return
            for g in chunk:
                run([g], 30.0)
            return
        for g in chunk:
            v = got.get(g)
            if v is None:
                verdicts[g] = {"killed": True, "gate": True, "why": "no result"}
                continue
            keys = set(v["keys"]) != set(base["keys"])
            errs = v["errors"] != base["errors"]
            pairs = v["pair_digest"] != base["pair_digest"]
            verdicts[g] = {"killed": keys or errs or pairs, "gate": keys or bool(set(v["errors"]) - set(base["errors"])),
                           "why": ", ".join(w for w, b in (("gate keys", keys), ("errors", errs), ("pairs", pairs)) if b),
                           "new_keys": sorted(set(v["keys"]) - set(base["keys"]))[:4],
                           "errors": {k: e for k, e in v["errors"].items() if base["errors"].get(k) != e}}

    for i in range(0, len(live), 20):
        run(live[i:i + 20], 300.0)
        log(f"  order {min(i + 20, len(live))}/{len(live)}", flush=True)
    return verdicts, why_not


def cmd_mutants(a) -> int:
    run_dir = Path(a.run)
    results = Path(a.results) if a.results else run_dir / "schemataB" / "results.jsonl"
    prepared = json.loads((run_dir / "prepared.json").read_text(encoding="utf-8"))
    commit = prepared["commit"]
    recs = survivors(results)
    engine = ht.Engine.load(ht._engine_root(a))
    carried, lost = carry(recs, engine.files, ht.ROOT, commit)
    print(f"{len(recs)} survivors of {results}; {len(carried)} carried onto this engine, {len(lost)} not "
          f"({dict(collections.Counter(lost.values()))})", flush=True)
    st = static_kills(a, engine, carried)
    if a.no_order:
        od, nb = {}, {}
    else:
        od, nb = order_kills(a, engine, carried, Path(a.work))
    rows = []
    for rec, m in carried:
        g = rec["gid"]
        s = st.get(m.id, {})
        o = od.get(g, {})
        rows.append({"id": m.id, "gid": g, "op": m.op, "file": m.file, "line_now": m.line,
                     "static": s.get("killed", False), "order": o.get("killed", False),
                     "gate": s.get("gate", False) or o.get("gate", False),
                     "order_not_run": nb.get(g), "detail": {"static": s, "order": o}})
    n = len(rows)
    killed = [r for r in rows if r["static"] or r["order"]]
    gate = [r for r in rows if r["gate"]]
    by_op = collections.defaultdict(lambda: [0, 0])
    by_file = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        k = r["static"] or r["order"]
        by_op[r["op"]][0] += k
        by_op[r["op"]][1] += 1
        by_file[r["file"]][0] += k
        by_file[r["file"]][1] += 1
    summary = {"survivors": len(recs), "carried": n, "lost": len(lost), "killed": len(killed),
               "killed_static": sum(r["static"] for r in rows), "killed_order": sum(r["order"] for r in rows),
               "killed_by_gate": len(gate), "order_not_built": len(nb),
               "kill_rate": round(len(killed) / n, 3) if n else None,
               "gate_kill_rate": round(len(gate) / n, 3) if n else None,
               "by_op": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_op.items())},
               "by_file": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_file.items())}}
    print(json.dumps(summary, indent=1))
    for r in killed[:40]:
        d = r["detail"]
        print(f"  killed {r['id']} ({r['op']}) static={d['static'].get('rows') or ''} order={d['order'].get('why', '')} "
              f"{d['order'].get('new_keys') or ''}")
    if a.json:
        Path(a.json).write_text(json.dumps({"summary": summary, "rows": rows, "lost": lost}, indent=1), encoding="utf-8")
    return 0
