"""Add `rows_parity.npz` (the information-parity side blocks of every row) to the shards of a teacher store (D59 lane D).

`python rl/parity_reencode.py --store <store> --generation N [--procs 4] [--agent DIR] [--shard NAME]` adds, beside each shard's `rows.npz`, `rows_parity.npz`: one float32 array per
block of `rl/parity_blocks.py` (`relics`, `roles`, `events`, aligned with `rows` and in its order) and `known`, a bool per row. Like `run_teacher.py reencode` (which it borrows the
store's readers and the replay machinery from, and which it does not change) it builds each row's blocks from the screen's state view and options:

* a run the teacher played (`source` new) has its RunRecord, which holds every state view and option list, so no game is played;
* a stored exploring run (`source` stored) is replayed exactly (`hindsight.replay_run`), about 7 CPU-s a run.

Every row built again is checked against the stored one: the base vector of the option's afterstate (`features.encode_afterstate`) must equal the first 1,855 numbers of the stored
row. A run whose rows do not match, or that cannot be replayed, has its rows written as zeros with `known` False (and counted in the manifest); nothing is refused and no other
shard file is touched. The manifest gains `parity: true`, `parity_version`, `parity_blocks`, `parity_rows`, `parity_unknown_rows` and the time. It is idempotent (a shard that has its
file is left), resumes from `reencode.d/<seed>.parity.npz`, and takes its own lock (`parity.lock` in the generation folder), not the teacher job's: it only reads finished shards and the
run records and adds files. Processes run at BelowNormal; the store's other files are never rewritten.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import parity_blocks as pb  # noqa: E402

BASE_SIZE = 1855
RESULT_SUFFIX = ".parity.npz"
REENCODE_DIR = "reencode.d"
LOCK_NAME = "parity.lock"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def shard_has_parity(d: Path) -> bool:
    """Whether the shard has its `rows_parity.npz` and its manifest names this build's block layout."""
    if not (d / pb.SHARD_ROWS_PARITY).is_file():
        return False
    try:
        doc = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(doc.get("parity") and doc.get("parity_version") == pb.PARITY_VERSION and list(doc.get("parity_blocks") or []) == list(pb.BLOCK_NAMES))


def shard_folders(gen_dir: Path, names=None) -> list[Path]:
    out = [m.parent for m in sorted((Path(gen_dir) / "shards").glob("*/manifest.json")) if not m.parent.name.endswith(".tmp")]
    return [d for d in out if not names or d.name in set(names)]


def plan_shard(d: Path) -> dict[str, list[dict]]:
    """seed -> the screens of the shard with rows: each {ordinal, line, row_ix (the rows' places in rows.npz), opt (their option indices), base (their stored base vectors)}."""
    recs = [json.loads(x) for x in (d / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    with np.load(d / "rows.npz", allow_pickle=False) as z:
        rows, decision, option = z["rows"], z["decision"], z["option"]
    order = np.argsort(decision, kind="stable")
    sd = decision[order]
    plan: dict[str, list[dict]] = {}
    for line, r in enumerate(recs):
        if r.get("row0") is None:
            continue
        a, b = np.searchsorted(sd, line, side="left"), np.searchsorted(sd, line, side="right")
        ix = order[a:b]
        plan.setdefault(str(r["seed"]), []).append({"ordinal": int(r["ordinal"]), "line": line, "row_ix": ix.astype(np.int64), "opt": option[ix].astype(np.int64),
                                                    "base": np.ascontiguousarray(rows[ix, :BASE_SIZE])})
    return plan


def _screen_blocks(view: Any, decision: Any, screen: dict) -> tuple[dict[str, np.ndarray], bool]:
    """The parity blocks of the screen's stored rows (`screen["opt"]` indexes the decision's options), and whether the base vectors built again equal the stored ones."""
    from run_train import features  # noqa: PLC0415

    v = decision.view or view
    options = [decision.options[int(i)] for i in screen["opt"]]
    ok = True
    try:
        base = np.stack([features.encode_afterstate(v, o) for o in options]) if options else np.zeros((0, BASE_SIZE), dtype=np.float32)
        ok = base.shape == screen["base"].shape and bool(np.array_equal(base, screen["base"]))
    except (ValueError, IndexError):
        ok = False
    return pb.option_parity(v, options), ok


def _write_result(out: Path, seed: str, screens: list[dict], got: dict[int, tuple[dict[str, np.ndarray], bool]]) -> tuple[int, int]:
    """Write `<seed>.parity.npz`: the rows of every screen (zeros and not known where the screen was not built or does not match). Returns (rows, unknown rows)."""
    ix = np.concatenate([s["row_ix"] for s in screens]) if screens else np.zeros(0, dtype=np.int64)
    n = len(ix)
    blocks = {name: np.zeros((n, pb.WIDTHS[name]), dtype=np.float32) for name in pb.BLOCK_NAMES}
    known = np.zeros(n, dtype=bool)
    at = 0
    for s in screens:
        m = len(s["row_ix"])
        built = got.get(s["ordinal"])
        if built is not None and built[1]:
            for name in pb.BLOCK_NAMES:
                blocks[name][at:at + m] = built[0][name]
            known[at:at + m] = True
        at += m
    tmp = out / f"{seed}.tmp{RESULT_SUFFIX}"
    np.savez_compressed(tmp, row_ix=ix, known=known, **blocks)
    os.replace(tmp, out / f"{seed}{RESULT_SUFFIX}")
    return n, int((~known).sum())


def run_from_record(gen_dir: Path, seed: str, screens: list[dict]) -> dict:
    """A run the teacher played: its RunRecord holds every view and option list. No game is played."""
    from run_train import shards  # noqa: PLC0415

    t0 = time.perf_counter()
    path = Path(gen_dir) / "decisions.d" / f"{seed}.record.json"
    row: dict[str, Any] = {"seed": seed, "source": "new", "screens": len(screens)}
    got: dict[int, tuple] = {}
    if not path.is_file():
        row["why"] = "no run record"
    else:
        rec = json.loads(path.read_text(encoding="utf-8"))
        for s in screens:
            o = s["ordinal"]
            entry = rec["decisions"][o] if o < len(rec["decisions"]) else None
            if entry is None or int(entry["i"]) != o:
                row["why"] = f"the run record has no decision {o}"
                break
            got[o] = _screen_blocks(shards.state_view(rec, entry), shards.entry_decision(entry), s)
    out = Path(gen_dir) / REENCODE_DIR
    out.mkdir(parents=True, exist_ok=True)
    n, unknown = _write_result(out, seed, screens, got)
    row.update(rows=n, unknown=unknown, ok=unknown == 0, wall=round(time.perf_counter() - t0, 2))
    return row


def reencode_run(task: tuple) -> dict:
    """One stored exploring run, replayed (`run_teacher._J` holds the player: the pool's init). `task` = ("rt-parity", seed, ix, record, screens)."""
    import hindsight  # noqa: PLC0415
    import run_teacher as rt  # noqa: PLC0415

    t0, c0 = time.perf_counter(), time.process_time()
    _, seed, ix, record, screens = task
    left = {s["ordinal"]: s for s in screens}
    got: dict[int, tuple] = {}
    row: dict[str, Any] = {"_task": ["rt-parity", seed, int(ix)], "seed": seed, "source": "stored", "screens": len(screens), "ok": False}

    def hook(pt: Any) -> bool:
        s = left.get(pt.ordinal)
        if s is not None:
            got[pt.ordinal] = _screen_blocks(pt.view, pt.decision, s)
            del left[pt.ordinal]
        return not left

    try:
        res = hindsight.replay_run(seed, int(ix), rt._replayer(tuple(record["_explore"])), record, hook, config="a3-v1", own_schedule=True)  # noqa: SLF001
        row["replay"] = res
    except Exception as e:  # noqa: BLE001 -- the run stays unknown; the pool's other runs go on
        row["why"] = f"replay failed: {type(e).__name__}: {str(e)[:120]}"
    out = rt._J["gen_dir"] / REENCODE_DIR  # noqa: SLF001
    out.mkdir(parents=True, exist_ok=True)
    n, unknown = _write_result(out, seed, screens, got)
    row.update(rows=n, unknown=unknown, ok=unknown == 0, cpu=round(time.process_time() - c0, 2), wall=round(time.perf_counter() - t0, 2))
    return row


def assemble_shard(gen_dir: Path, d: Path, plan: dict[str, list[dict]]) -> dict:
    """Write the shard's `rows_parity.npz` from the runs' results and patch its manifest. A run with no result is unknown. Returns the counts."""
    with np.load(d / "rows.npz", allow_pickle=False) as z:
        n_rows = len(z["decision"])
    blocks = {name: np.zeros((n_rows, pb.WIDTHS[name]), dtype=np.float32) for name in pb.BLOCK_NAMES}
    known = np.zeros(n_rows, dtype=bool)
    missing = 0
    for seed, screens in plan.items():
        f = Path(gen_dir) / REENCODE_DIR / f"{seed}{RESULT_SUFFIX}"
        if not f.is_file():
            missing += 1
            continue
        with np.load(f, allow_pickle=False) as z:
            ix = z["row_ix"]
            known[ix] = z["known"]
            for name in pb.BLOCK_NAMES:
                blocks[name][ix] = z[name]
    tmp = d / "rows_parity.tmp.npz"
    np.savez_compressed(tmp, known=known, **blocks)
    os.replace(tmp, d / pb.SHARD_ROWS_PARITY)
    mf = d / "manifest.json"
    doc = json.loads(mf.read_text(encoding="utf-8"))
    doc.update(parity=True, parity_version=pb.PARITY_VERSION, parity_blocks=list(pb.BLOCK_NAMES), parity_rows=int(n_rows), parity_unknown_rows=int((~known).sum()),
               parity_reencoded_utc=utc_stamp())
    tmpm = d / "manifest.tmp.json"
    tmpm.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    for attempt in range(20):  # a reader may hold the manifest for a moment
        try:
            os.replace(tmpm, mf)
            break
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
    else:
        raise PermissionError(f"{mf} stayed locked")
    return {"shard": d.name, "rows": int(n_rows), "unknown_rows": int((~known).sum()), "runs": len(plan), "runs_without_result": missing}


def _lock(gen_dir: Path) -> bool:
    lock = Path(gen_dir) / LOCK_NAME
    if lock.is_file():
        try:
            pid = int(json.loads(lock.read_text(encoding="utf-8")).get("pid") or 0)
        except (OSError, ValueError):
            pid = 0
        if pid and pid != os.getpid():
            from tools.lab import winsys  # noqa: PLC0415

            if winsys.parent_alive(pid):
                return False
    lock.write_text(json.dumps({"pid": os.getpid(), "t": utc_stamp()}), encoding="utf-8")
    return True


def reencode_cmd(a: argparse.Namespace) -> int:
    import fight_gym  # noqa: PLC0415
    import run_teacher as rt  # noqa: PLC0415

    with contextlib.suppress(Exception):
        from sts2_gym import procutil  # noqa: PLC0415

        procutil.lower_priority()
    gen_dir = Path(a.store) / str(a.generation)
    todo = [d for d in shard_folders(gen_dir, a.shard) if not shard_has_parity(d)]
    if not todo:
        print("parity reencode: every shard has its rows_parity.npz", flush=True)
        return 0
    if not _lock(gen_dir):
        print("parity reencode: another parity reencode holds this generation", flush=True)
        return 2
    try:
        rows_by_seed = {r["seed"]: r for r in rt.done_rows(gen_dir)}
        plans = {d.name: plan_shard(d) for d in todo}
        stored_tasks: list[tuple] = []
        n_new = n_missing = 0
        stored_records: dict[str, Any] = {}
        need_stored = [s for p in plans.values() for s in p if rows_by_seed.get(s, {}).get("source") == "stored"]
        if need_stored:
            import hindsight  # noqa: PLC0415

            stored_records = {str(r["seed"]): r for r in hindsight.exploring_runs(Path(a.runs_root), int(a.generation), config=a.stored_config)}
        new_jobs: list[tuple[str, list[dict]]] = []
        for d in todo:
            for seed, screens in plans[d.name].items():
                if (Path(gen_dir) / REENCODE_DIR / f"{seed}{RESULT_SUFFIX}").is_file():
                    continue
                row = rows_by_seed.get(seed)
                src = (row or {}).get("source")
                if src == "new":
                    new_jobs.append((seed, screens))
                elif src == "stored" and seed in stored_records:
                    stored_tasks.append(("rt-parity", seed, int(row["ix"]), stored_records[seed], screens))
                else:
                    n_missing += 1
        if a.limit_runs:
            new_jobs, stored_tasks = new_jobs[: a.limit_runs], stored_tasks[: a.limit_runs]
        print(f"parity reencode: {len(todo)} shards, {len(new_jobs)} recorded runs, {len(stored_tasks)} stored runs to replay ({n_missing} with no row or stored record), {a.procs} processes", flush=True)
        t0 = time.time()
        for seed, screens in new_jobs:
            r = run_from_record(gen_dir, seed, screens)
            n_new += 1
            if not r["ok"]:
                print(f"parity reencode: run {seed}: {r.get('unknown')} of {r.get('rows')} rows unknown ({r.get('why', 'rows differ')})", flush=True)
        if stored_tasks:
            agent_dir = a.agent or (rt.read_json_file(Path(a.agent_root) / "current.json") or {}).get("dir")
            if not agent_dir:
                print("parity reencode: no player (give --agent: the player that made the shards)", flush=True)
                return 2
            import gen_loop  # noqa: PLC0415

            agent = gen_loop.load_agent(agent_dir)
            ident = rt.agent_identity(agent_dir, agent)
            for d in todo:
                have = (rt.read_json_file(d / "manifest.json") or {}).get("agent_sha256")
                if have != ident["agent_sha256"]:
                    print(f"parity reencode: shard {d.name} was made by another player than {ident['agent']} (agent_sha256 differs): give --agent", flush=True)
                    return 2
            lib = str(a.lib or _ROOT / "out")
            cfg = {"agent_dir": str(agent_dir), "replay_agent": str(a.replay_agent or agent_dir), "judge": None, "lib": lib, "gen_dir": str(gen_dir), "cal": None}
            fight_gym.MAX_PROCS = max(int(fight_gym.MAX_PROCS), int(a.procs))
            target = gen_dir / "reencode-parity.jsonl"
            fight_gym.run_tasks(stored_tasks, reencode_run, target, a.procs, hard_of=lambda t: rt.RUN_WALL_CAP_S, init=rt._init_job_worker, init_args=(cfg,),  # noqa: SLF001
                                capped_of=lambda t, reason, seconds, prog: {"_task": list(t[:3]), "seed": t[1], "ok": False, "why": reason, "wall": round(seconds, 1)})
            for r in rt.read_jsonl(target):
                if not r.get("ok"):
                    print(f"parity reencode: run {r.get('seed')}: {r.get('unknown', '?')} rows unknown ({r.get('why', 'rows differ')})", flush=True)
        if a.limit_runs:  # a smoke test: the runs' results are written, no shard is (a shard built from some of its runs would look finished)
            print(f"parity reencode: --limit-runs: {n_new} recorded and {len(stored_tasks)} stored runs built; no shard file written", flush=True)
            return 0
        done = [assemble_shard(gen_dir, d, plans[d.name]) for d in todo]
        res = {"shards_done": len(done), "rows": sum(x["rows"] for x in done), "unknown_rows": sum(x["unknown_rows"] for x in done), "runs": sum(x["runs"] for x in done),
               "runs_without_result": sum(x["runs_without_result"] for x in done), "seconds": round(time.time() - t0)}
        print("parity reencode: " + json.dumps(res), flush=True)
        return 0
    finally:
        with contextlib.suppress(OSError):
            (Path(gen_dir) / LOCK_NAME).unlink()


def main(argv: list[str] | None = None) -> int:
    import run_teacher as rt  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", default=rt.DEFAULT_STORE)
    ap.add_argument("--generation", type=int, required=True)
    ap.add_argument("--agent-root", default=rt.DEFAULT_AGENT_ROOT)
    ap.add_argument("--agent", default="", help="the player that made the shards (only for a store with stored runs to replay)")
    ap.add_argument("--replay-agent", default="")
    ap.add_argument("--lib", default="")
    ap.add_argument("--runs-root", default=rt.DEFAULT_RUNS_ROOT)
    ap.add_argument("--stored-config", default=rt.STORED_CONFIG)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--shard", action="append", default=[])
    ap.add_argument("--limit-runs", type=int, default=0)
    return reencode_cmd(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
