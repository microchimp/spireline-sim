"""S6: the fight solver's coverage curve on real decisions (docs/rebuild/FIGHT_SOLVER.md sections 11 and 16).

    python tools/solver/coverage.py collect --source NIGHT_ARM_DIR [--source ...] --runs N --seed S --out OUT
    python tools/solver/coverage.py sample  --out OUT --fights 40 --seed S
    python tools/solver/coverage.py dump    --out OUT [--procs 12]
    python tools/solver/coverage.py solve   --out OUT --cap 1000000 --wall-s 120 [--only-not-exact-at 1000000] [--procs 12]
    python tools/solver/coverage.py check   --out OUT --n 20 --budgets 10000,100000,1000000
    python tools/solver/coverage.py speed   --out OUT --n 60 --budget 100000          (one process, a quiet machine)
    python tools/solver/coverage.py speed-jit --out OUT                                  (the same decisions through EmuDump)
    python tools/solver/coverage.py curve   --out OUT [--second-name solves_1e7.jsonl]

Where the decisions come from. The real game's nightly block (envproto play) keeps no fight positions: its `runs.jsonl` and
`divergences.jsonl` hold seeds, outcomes and the diverged decisions' hand and energy only, so no real-game decision can be
re-anchored from them. Its emulator arm plays the same agent on the same seed block on the CPU prior path, which a hooked
replay reproduces exactly (`gen0.play_emu_run`: same agent seed, same run index), and the lockstep of the real arm follows
the emulator's decisions to within its divergence rate. So `collect` replays the arm's runs and lists every combat decision
(act, fight kind, floor, turn, the fight's outcome), `sample` draws whole fights stratified by act x kind x won with a fixed
seed, and `dump` replays the runs again and writes every decision of the sampled fights as an `sts2-emu-snapshot/1` document.

`solve` solves each decision once (expected mode, iterative deepening, all lossless prunings, a fresh table per decision)
under a node cap and a wall cap, through the native library in a spawn pool (one emulator per process, BelowNormal priority
through the Windows priority class, at most 12 workers), and keeps the deepening's horizons: for each, the cumulative nodes,
engine steps and seconds at which it ended and whether it was exact. Coverage at a node budget B is read from that one run:
exact within B when a horizon finished exact at cumulative nodes <= B and the size estimate did not skip an earlier horizon
at B (the deepening skips horizon h > 0 when its estimate is over log10(B) + 1). A time budget T is read the same way from
the cumulative seconds, with the estimate rule at the node cap. `check` re-solves a few decisions directly at each budget
and compares exactness with the reading. Results stay in OUT (the lab's results directory, never committed); they hold no
host name, user name or absolute path.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

NODE_BUDGETS = (10**3, 10**4, 10**5, 10**6, 10**7)
TIME_BUDGETS = (0.01, 0.1, 1.0, 10.0)
ESTIMATE_SLACK = 1.0  # SolveOptions.EstimateSlack
NODE_MONSTER, NODE_ELITE, NODE_BOSS = 1, 2, 6  # rl/run_train/vocab.py
STEP_CAP = 100_000  # engine runs per solve: bounds the live clones (an expansion holds every outcome while its children are solved)
MAX_HORIZON = 20
PHASES = ("step", "clone", "key_encode", "key_hash", "rational", "table", "chance", "bound", "remainder")


# ----------------------------------------------------------------------------------------------- pure parts (tested)


def read_jsonl(p: Path) -> list[dict]:
    if not Path(p).is_file():
        return []
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(p: Path, rows) -> None:
    Path(p).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def fight_kind(node_type: int) -> str:
    return {NODE_MONSTER: "hallway", NODE_ELITE: "elite", NODE_BOSS: "boss"}.get(int(node_type), "other")


def source_label(path: str | Path) -> str:
    """A source's label without host, user or drive: the last three parts of its path (nightly/<date>/<arm>)."""
    parts = Path(str(path).replace("\\", "/")).parts
    return "/".join(parts[-3:])


def fights_of(rows: list[dict]) -> list[dict]:
    """Decisions grouped into fights (run, floor): kind, act, turns, and won (the run went on past the fight, or won)."""
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["source"], r["seed"], r["floor"])].append(r)
    out = []
    for (src, seed, floor), ds in sorted(by.items()):
        ds.sort(key=lambda d: d["k"])
        run_end, run_floor = ds[0]["run_end"], ds[0]["run_floor"]
        last_fight = floor == max(x["floor"] for x in rows if x["source"] == src and x["seed"] == seed)
        if last_fight and run_end not in ("death", "win"):
            won = None  # a run cut by its wall cap or a crash: the fight's outcome is unknown
        else:
            won = not (last_fight and run_end == "death" and run_floor == floor)
        out.append({"fight": f"{src}|{seed}|{floor}", "source": src, "seed": seed, "ix": ds[0]["ix"], "floor": floor,
                    "act": ds[0]["act"], "kind": fight_kind(ds[0]["node_type"]), "won": won, "decisions": len(ds),
                    "ks": [d["k"] for d in ds], "turns": max(d["turn"] for d in ds)})
    return out


def stratum(f: dict) -> str:
    return f"act{f['act']}/{f['kind']}/{'won' if f['won'] else 'lost'}"


def sample_fights(fights: list[dict], n: int, seed: int) -> list[dict]:
    """n whole fights, stratified by act x kind x won: strata taken in turn (each shuffled with the seed), so every stratum
    present gets a fight before any gets a second. Fights of unknown outcome are left out. Reproducible from the seed."""
    rnd = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(fights, key=lambda f: f["fight"]):
        if f["won"] is not None:
            groups[stratum(f)].append(f)
    for k in sorted(groups):
        rnd.shuffle(groups[k])
    keys = sorted(groups)
    picked: list[dict] = []
    while len(picked) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
    return sorted(picked, key=lambda f: f["fight"])


def _estimate_ok(h: dict, budget: float) -> bool:
    e = h.get("estimate_log10")
    return h["horizon"] == 0 or e is None or e <= math.log10(budget) + ESTIMATE_SLACK


def exact_within_nodes(horizons: list[dict], budget: int) -> bool:
    """Read from one deepening run: whether a solve with this node budget ends exact (see the module's doc)."""
    for h in horizons:
        if not _estimate_ok(h, budget) or h["end"] != "done" or h["nodes"] > budget:
            return False
        if h["exact"]:
            return True
    return False


def exact_within_seconds(horizons: list[dict], seconds: float, cap_nodes: int) -> bool:
    """As exact_within_nodes for a time budget, with the size estimate's rule at the node cap."""
    for h in horizons:
        if not _estimate_ok(h, cap_nodes) or h["end"] != "done" or h["nodes"] > cap_nodes or h["seconds"] > seconds:
            return False
        if h["exact"]:
            return True
    return False


def deepest_within_nodes(horizons: list[dict], budget: int) -> int:
    """The deepest horizon a solve with this node budget completes (-1: none)."""
    best = -1
    for h in horizons:
        if not _estimate_ok(h, budget) or h["end"] != "done" or h["nodes"] > budget:
            break
        best = h["horizon"]
    return best


def needs_larger_cap(solve: dict, cap: int) -> bool:
    """Whether a solve at a smaller cap can end otherwise at `cap`: not refused, not exact, and its deepening stopped at the
    node or wall cap, or at a size estimate the larger cap would try (estimate <= log10(cap) + slack)."""
    hz = (solve.get("stats") or {}).get("horizons") or []
    if solve["status"] == -3 or not hz or any(h["exact"] for h in hz):
        return False
    last = hz[-1]
    if last["end"] == "budget":
        return True
    return last["end"] == "estimate" and _estimate_ok(last, cap)


def turns_left_bucket(t: int) -> str:
    return str(t) if t < 4 else "4+"


def _share(xs: list[bool]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def curve(decisions: list[dict], solves: list[dict], cap_nodes: int) -> dict:
    """The coverage tables: the share of decisions solved exactly at each node and time budget, overall and by turns left,
    fight kind, act and outcome; the share of fights whose every decision is exact within 1 s and 10 s; the turn from which
    each fight is exactly solvable (every later decision exact); refusals by reason."""
    by_id = {s["id"]: s for s in solves}
    rows = []
    for d in decisions:
        s = by_id.get(d["id"])
        if s is None:
            continue
        hz = (s.get("stats") or {}).get("horizons") or []
        refused = s["status"] == -3
        row = {"id": d["id"], "fight": d["fight"], "kind": d["kind"], "act": d["act"], "won": d["won"], "turn": d["turn"],
               "turns_left": d["turns_left"], "refused": refused, "reason": (s.get("reason") or "").split(":")[0] if refused else None,
               "censored": bool(s.get("censored"))}
        cut_at = hz[-1]["nodes"] if row["censored"] and hz else None
        for b in NODE_BUDGETS:
            row[f"n{b}"] = (not refused) and b <= cap_nodes and exact_within_nodes(hz, b)
            row[f"c{b}"] = cut_at is not None and cut_at < b and not row[f"n{b}"]
        for t in TIME_BUDGETS:
            row[f"t{t}"] = (not refused) and exact_within_seconds(hz, t, cap_nodes)
        rows.append(row)

    def table(sel) -> dict:
        out = {}
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[sel(r)].append(r)
        for g in sorted(groups):
            rs = groups[g]
            out[g] = {"n": len(rs), **{f"nodes_1e{int(math.log10(b))}": _share([r[f"n{b}"] for r in rs]) for b in NODE_BUDGETS if b <= cap_nodes},
                      **{f"censored_1e{int(math.log10(b))}": sum(r[f"c{b}"] for r in rs) for b in NODE_BUDGETS if b <= cap_nodes},
                      **{f"time_{t:g}s": _share([r[f"t{t}"] for r in rs]) for t in TIME_BUDGETS}}
        return out

    fights: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        fights[r["fight"]].append(r)
    full = {}
    first_exact = Counter()
    for t in (1.0, 10.0):
        full[f"{t:g}s"] = _share([all(r[f"t{t}"] for r in rs) for rs in fights.values()])
    for rs in fights.values():
        rs.sort(key=lambda r: (r["turn"], r["id"]))
        tl = None
        for i, r in enumerate(rs):
            if all(x["t10.0"] for x in rs[i:]):
                tl = r["turns_left"]
                break
        first_exact[turns_left_bucket(tl) if tl is not None else "never"] += 1
    return {
        "decisions": len(rows),
        "fights": len(fights),
        "cap_nodes": cap_nodes,
        "censored_by_wall": sum(r["censored"] for r in rows),
        "refused": dict(Counter(r["reason"] for r in rows if r["refused"])),
        "overall": table(lambda r: "all"),
        "by_turns_left": table(lambda r: turns_left_bucket(r["turns_left"])),
        "by_kind": table(lambda r: r["kind"]),
        "by_act": table(lambda r: f"act{r['act']}"),
        "by_outcome": table(lambda r: "won" if r["won"] else "lost"),
        "by_kind_turns_left": table(lambda r: f"{r['kind']}/{turns_left_bucket(r['turns_left'])}"),
        "fights_fully_exact": full,
        "fights_exact_from_turns_left_at_10s": dict(first_exact),
    }


def merged_horizons(first: dict, second: dict | None) -> dict:
    """A decision's solve: the one at the larger cap when it was re-solved (the deepening is deterministic in nodes)."""
    return second if second is not None else first


# ----------------------------------------------------------------------------------------------- the emulator parts


def _setup_paths() -> None:
    for p in (ROOT / "rl", ROOT / "src", ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _below_normal() -> None:
    _setup_paths()
    from tools.lab import winsys  # noqa: PLC0415

    winsys.set_below_normal()


def _arm(source: str) -> tuple[dict, list[dict], str, str]:
    """An emulator arm's summary, runs, agent dir and engine."""
    d = Path(source)
    summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    runs = read_jsonl(d / "runs.jsonl")
    res = json.loads((d.parent / d.name.replace("-emu", "") / "result.json").read_text(encoding="utf-8")) if (d.parent / d.name.replace("-emu", "") / "result.json").is_file() else {}
    engine = res.get("engine") or summ.get("engine") or "old"
    return summ, runs, summ["agent"]["dir"], engine


def _replay(task: tuple) -> dict:
    """Replays one run of an arm exactly; lists its combat decisions, and with `keep` writes the kept ones' documents."""
    source, label, agent_root, seed, ix, config, engine, keep, out = task
    _below_normal()
    _setup_paths()
    import gen0  # noqa: PLC0415

    gen0.select_engine(engine)
    import gen_loop  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    _, _, agent_dir, _ = _arm(source)
    agent = gen_loop.load_agent(Path(agent_root) / agent_dir)
    rows: list[dict] = []
    docs: list[dict] = []
    want = set(keep or [])

    def hook(k, root, s, pos):
        doc = native.get_state_json(root.handle)
        rows.append({"source": label, "seed": seed, "ix": ix, "k": k, "floor": int(s.info["floor"]), "node_type": int(s.info["node_type"]),
                     "act": int(pos.act) if pos is not None else None, "turn": int(doc.get("turn", 0)), "n_moves": pos.n_moves if pos else None})
        if k in want:
            docs.append({"id": f"{label}|{seed}|k{k:04d}", "doc": doc})
        return bool(want) and k >= max(want)

    t0 = time.perf_counter()
    r = gen0.play_emu_run(seed, ix, config, agent=agent, hook=hook)
    for x in rows:
        x["run_floor"], x["run_end"] = int(r["floor"]), r.get("end")
    if docs:
        with open(Path(out) / "docs" / f"{label.replace('/', '_')}_{seed}.jsonl", "w", encoding="utf-8") as fh:
            fh.write("".join(json.dumps(d) + "\n" for d in docs))
    return {"rows": rows, "floor": int(r["floor"]), "end": r.get("end"), "seconds": round(time.perf_counter() - t0, 1)}


def _pool(procs: int):
    return mp.get_context("spawn").Pool(processes=min(12, max(1, procs)), maxtasksperchild=8)


def cmd_collect(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = []
    meta = {"sources": [], "seed": a.seed, "runs_per_source": a.runs}
    rnd = random.Random(a.seed)
    for src in a.source:
        summ, runs, agent_dir, engine = _arm(src)
        label = source_label(src)
        runs = sorted((r for r in runs if not r.get("crash")), key=lambda r: r["ix"])
        pick = rnd.sample(runs, min(a.runs, len(runs)))
        meta["sources"].append({"label": label, "agent": agent_dir, "engine": engine, "config": summ.get("config"), "picked": len(pick)})
        for r in pick:
            tasks.append((src, label, a.agent_root, r["seed"], int(r["ix"]), summ.get("config") or "a3-v1", engine, None, str(out)))
    rows, replay = [], []
    with _pool(a.procs) as pool:
        for res, t in zip(pool.imap(_replay, tasks), tasks, strict=True):
            rows += res["rows"]
            want = next(r for r in read_jsonl(Path(t[0]) / "runs.jsonl") if r["seed"] == t[3])
            replay.append({"source": t[1], "seed": t[3], "floor": res["floor"], "end": res["end"], "arm_floor": want["floor"],
                           "arm_end": want.get("end"), "same": res["floor"] == want["floor"] and res["end"] == want.get("end"),
                           "seconds": res["seconds"]})
    write_jsonl(out / "decisions_all.jsonl", rows)
    write_jsonl(out / "replay_check.jsonl", replay)
    fights = fights_of(rows)
    write_jsonl(out / "fights_all.jsonl", fights)
    meta["replays_exact"] = sum(r["same"] for r in replay)
    meta["replays"] = len(replay)
    meta["strata"] = dict(Counter(stratum(f) for f in fights if f["won"] is not None))
    (out / "collect.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0


def cmd_sample(a) -> int:
    out = Path(a.out)
    fights = read_jsonl(out / "fights_all.jsonl")
    picked = sample_fights(fights, a.fights, a.seed)
    write_jsonl(out / "fights.jsonl", picked)
    rows = read_jsonl(out / "decisions_all.jsonl")
    keep = {f["fight"] for f in picked}
    by_fight = {f["fight"]: f for f in picked}
    dec = []
    for r in rows:
        fid = f"{r['source']}|{r['seed']}|{r['floor']}"
        if fid in keep:
            f = by_fight[fid]
            dec.append({"id": f"{r['source']}|{r['seed']}|k{r['k']:04d}", "fight": fid, "source": r["source"], "seed": r["seed"],
                        "ix": r["ix"], "k": r["k"], "floor": r["floor"], "act": f["act"], "kind": f["kind"], "won": f["won"],
                        "turn": r["turn"], "turns_left": f["turns"] - r["turn"]})
    write_jsonl(out / "decisions.jsonl", dec)
    print(json.dumps({"fights": len(picked), "decisions": len(dec), "strata": dict(Counter(stratum(f) for f in picked))}, indent=1))
    return 0


def cmd_dump(a) -> int:
    out = Path(a.out)
    (out / "docs").mkdir(exist_ok=True)
    meta = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    dec = read_jsonl(out / "decisions.jsonl")
    want: dict[tuple, list[int]] = defaultdict(list)
    ixs = {}
    for d in dec:
        want[(d["source"], d["seed"])].append(d["k"])
        ixs[(d["source"], d["seed"])] = d["ix"]
    srcs = {source_label(s): s for s in a.source}
    tasks = []
    for (label, seed), ks in sorted(want.items()):
        src = srcs[label]
        m = next(x for x in meta["sources"] if x["label"] == label)
        tasks.append((src, label, a.agent_root, seed, ixs[(label, seed)], m["config"] or "a3-v1", m["engine"], sorted(ks), str(out)))
    with _pool(a.procs) as pool:
        for _ in pool.imap_unordered(_replay, tasks):
            pass
    docs = []
    for f in sorted((out / "docs").glob("*.jsonl")):
        docs += read_jsonl(f)
    have = {d["id"] for d in docs}
    write_jsonl(out / "positions.jsonl", [d for d in docs if d["id"] in {x["id"] for x in dec}])
    print(json.dumps({"positions": len(have & {x["id"] for x in dec}), "decisions": len(dec)}))
    return 0


def _init_solver(engine: str, flags: int) -> None:
    _below_normal()
    _setup_paths()
    import gen0  # noqa: PLC0415

    gen0.select_engine(engine)
    from sts2_gym import native  # noqa: PLC0415

    native.solver_debug(flags)


def _solve(task: tuple) -> dict:
    pid, doc, cap, wall_s, step_cap = task
    from sts2_gym import native  # noqa: PLC0415

    try:
        h = native.load_combat_snapshot(doc, 0)
    except Exception as e:  # noqa: BLE001 - a document the engine cannot load is a refusal of its own, counted
        return {"id": pid, "cap": cap, "wall_cap_s": wall_s, "step_cap": step_cap, "status": -3,
                "reason": f"snapshot-load: {type(e).__name__}: {str(e)[:160]}", "exact": False, "horizon": -1, "wall_s": 0.0,
                "censored": False, "stats": {"horizons": []}}
    try:
        opts = native.SolveOptions(node_budget=int(cap), step_budget=int(step_cap), time_budget_ms=int(wall_s * 1000), max_horizon=MAX_HORIZON)
        solver = native.solver_create(h, opts)
        try:
            t0 = time.perf_counter()
            r = native.solve_expected(solver, h, opts)
            wall = time.perf_counter() - t0
            st = native.solver_stats(solver)
        finally:
            native.solver_destroy(solver)
    finally:
        native.destroy(h)
    hz = st.get("horizons") or []
    censored = bool(hz) and hz[-1]["end"] == "budget" and hz[-1]["nodes"] <= cap and not any(x["exact"] for x in hz)
    return {"id": pid, "cap": cap, "wall_cap_s": wall_s, "step_cap": step_cap, "status": r.status, "reason": r.reason, "exact": bool(r.exact),
            "horizon": r.horizon, "wall_s": round(wall, 4), "censored": censored, "stats": st}


def _engine_of(out: Path) -> str:
    meta = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    engines = {s["engine"] for s in meta["sources"]}
    if len(engines) != 1:
        raise SystemExit(f"one engine per solve run, got {sorted(engines)}")
    return engines.pop()


def cmd_solve(a) -> int:
    out = Path(a.out)
    pos = {p["id"]: p["doc"] for p in read_jsonl(out / "positions.jsonl")}
    target = out / a.out_name
    done = {r["id"] for r in read_jsonl(target)}
    ids = sorted(pos)
    if a.only_not_exact_at:
        prev = {r["id"]: r for r in read_jsonl(out / a.prev_name)}
        ids = [i for i in ids if i in prev and needs_larger_cap(prev[i], a.cap)]
    if a.n:
        ids = sorted(random.Random(a.seed).sample(ids, min(a.n, len(ids))))  # a seeded subsample (the cut, reported)
    tasks = [(i, pos[i], a.cap, a.wall_s, a.step_cap) for i in ids if i not in done]
    print(f"solving {len(tasks)} decisions (cap {a.cap} nodes, wall {a.wall_s} s, {a.procs} workers)", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(12, max(1, a.procs)), initializer=_init_solver, initargs=(_engine_of(out), a.debug_flags), maxtasksperchild=4) as pool, \
            open(target, "a", encoding="utf-8") as fh:
        for n, r in enumerate(pool.imap_unordered(_solve, tasks), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if n % 20 == 0:
                print(f"  {n}/{len(tasks)}", flush=True)
    return 0


def cmd_check(a) -> int:
    """Direct solves at each budget on the first n decisions, against the reading from the cap run."""
    out = Path(a.out)
    pos = {p["id"]: p["doc"] for p in read_jsonl(out / "positions.jsonl")}
    first = {r["id"]: r for r in read_jsonl(out / a.prev_name)}
    ids = sorted(i for i in first if first[i]["status"] != -3)
    rnd = random.Random(a.seed)
    ids = sorted(rnd.sample(ids, min(a.n, len(ids))))
    budgets = [int(x) for x in a.budgets.split(",")]
    _init_solver(_engine_of(out), 0)
    rows, agree = [], 0
    for i in ids:
        hz = first[i]["stats"].get("horizons") or []
        for b in budgets:
            r = _solve((i, pos[i], b, a.wall_s, a.step_cap))
            read = exact_within_nodes(hz, b)
            same = read == r["exact"]
            agree += same
            rows.append({"id": i, "budget": b, "direct_exact": r["exact"], "read_exact": read, "same": same,
                         "direct_horizon": r["horizon"], "read_horizon": deepest_within_nodes(hz, b), "direct_status": r["status"]})
    write_jsonl(out / "check.jsonl", rows)
    print(json.dumps({"pairs": len(rows), "agree": agree, "disagree": [r for r in rows if not r["same"]]}, indent=1))
    return 0


def _speed_rows(ids, pos, budget, flags, wall_s=60.0, step_cap=STEP_CAP) -> list[dict]:
    from sts2_gym import native  # noqa: PLC0415

    native.solver_debug(flags)
    rows = []
    for i in ids:
        rows.append(_solve((i, pos[i], budget, wall_s, step_cap)))
        st = rows[-1]["stats"]
        print(f"  {flags} {i[-24:]} nodes {st.get('nodes')} steps {st.get('engine_steps')} {rows[-1]['wall_s']} s", flush=True)
    native.solver_debug(0)
    return rows


def cmd_speed(a) -> int:
    """One process: the same decisions at a fixed node budget with the timers off, on, and with the merge audit."""
    out = Path(a.out)
    pos = {p["id"]: p["doc"] for p in read_jsonl(out / "positions.jsonl")}
    dec = {d["id"]: d for d in read_jsonl(out / "decisions.jsonl")}
    rnd = random.Random(a.seed)
    pool = sorted(pos)
    if (out / a.prev_name).is_file():  # the decisions the coverage run solved (not refused), when it has run
        pool = sorted(r["id"] for r in read_jsonl(out / a.prev_name) if r["status"] != -3)
    ids = sorted(rnd.sample(pool, min(a.n, len(pool))))
    _init_solver(_engine_of(out), 0)
    res = {}
    for name, flags in (("off", 0), ("timers", 1), ("off2", 0), ("timers2", 1)) if not a.audit_only else ():
        t0 = time.perf_counter()
        rows = _speed_rows(ids, pos, a.budget, flags, a.wall_s, a.step_cap)
        res[name] = {"wall_s": round(time.perf_counter() - t0, 3), "rows": rows}
    if a.audit:
        rows = _speed_rows(ids, pos, a.budget, 2, a.wall_s, a.step_cap)
        res["audit"] = {"rows": rows}
    for v in res.values():
        for r in v["rows"]:
            r["kind"], r["turns_left"] = dec[r["id"]]["kind"], dec[r["id"]]["turns_left"]
    (out / a.out_name).write_text(json.dumps({"budget": a.budget, "ids": ids, "runs": res}), encoding="utf-8")
    print(json.dumps(speed_report(res), indent=1))
    return 0


def speed_report(res: dict) -> dict:
    rep: dict = {}

    def rate(rows):
        n = sum(r["stats"]["nodes"] for r in rows)
        s = sum(r["stats"]["engine_steps"] for r in rows)
        w = sum(r["stats"]["wall_seconds"] for r in rows)
        return {"decisions": len(rows), "nodes": n, "steps": s, "seconds": round(w, 3), "nodes_s": round(n / w) if w else None,
                "steps_s": round(s / w) if w else None, "steps_per_node": round(s / n, 2) if n else None}

    for name in ("off", "off2", "timers", "timers2"):
        if name in res:
            rep[name] = rate([r for r in res[name]["rows"] if r["status"] != -3])
    if "off" in res:
        base = [r for r in res["off"]["rows"] if r["status"] != -3]
        for r in base:  # the JIT rows carry no decision fields
            r.setdefault("kind", "?")
            r.setdefault("turns_left", 0)
        rep["off_by_kind"] = {k: rate([r for r in base if r["kind"] == k]) for k in sorted({r["kind"] for r in base})}
        rep["off_by_turns_left"] = {k: rate([r for r in base if turns_left_bucket(r["turns_left"]) == k])
                                    for k in sorted({turns_left_bucket(r["turns_left"]) for r in base})}
        rows = base
        kinds = ("play", "end_turn", "screen", "potion")
        mv = {k: sum(r["stats"]["moves_by_kind"][k] for r in rows) for k in kinds}
        rn = {k: sum(r["stats"]["runs_by_kind"][k] for r in rows) for k in kinds}
        steps = sum(r["stats"]["engine_steps"] for r in rows)
        et_moves = mv["end_turn"]
        hits = {k: sum(r["stats"][k] for r in rows) for k in ("table_hits_exact", "table_hits_same_turns", "table_misses")}
        rep["runs_per_move"] = {k: round(rn[k] / mv[k], 2) if mv[k] else None for k in kinds}
        rep["moves"] = mv
        rep["runs"] = rn
        rep["chance_levels_per_run"] = round(sum(r["stats"]["chance_levels"] for r in rows) / steps, 3) if steps else None
        rep["chance_level_max"] = max((r["stats"]["chance_level_max"] for r in rows), default=None)
        rep["leaves_per_end_turn_move"] = round(sum(r["stats"]["end_turn_leaves"] for r in rows) / et_moves, 2) if et_moves else None
        rep["table"] = hits
        rep["wide_roll_leaves"] = sum(r["stats"]["wide_roll_leaves"] for r in rows)
    for name in ("timers", "timers2"):
        if name in res:
            rows = [r for r in res[name]["rows"] if r["status"] != -3 and "phase_seconds" in r["stats"]]
            tot = {p: sum(r["stats"]["phase_seconds"][p] for r in rows) for p in PHASES}
            wall = sum(r["stats"]["wall_seconds"] for r in rows)
            timed = sum(tot.values())
            rep[f"{name}_split"] = {p: round(v / wall, 4) if wall else None for p, v in tot.items()}
            rep[f"{name}_gap"] = round((timed - wall) / wall, 4) if wall else None
    if "off" in rep and "timers" in rep:
        off = rep["off"]["seconds"] + rep.get("off2", {}).get("seconds", 0)
        on = rep["timers"]["seconds"] + rep.get("timers2", {}).get("seconds", 0)
        rep["timer_overhead"] = round(on / off - 1, 4) if off else None
    if "audit" in res:
        rows = [r for r in res["audit"]["rows"] if r["status"] != -3 and "merge_audit" in r["stats"]]
        rep["merge_audit"] = audit_report(rows)
    return rep


def audit_report(rows: list[dict]) -> dict:
    """Distinct masked keys against distinct full keys per solve, and pooled."""
    full = sum(r["stats"]["merge_audit"]["distinct_full"] for r in rows)
    masked = sum(r["stats"]["merge_audit"]["distinct_masked"] for r in rows)
    keep = sum(r["stats"]["merge_audit"]["distinct_masked_keep_block"] for r in rows)
    per = [r["stats"]["merge_audit"]["distinct_masked"] / r["stats"]["merge_audit"]["distinct_full"] for r in rows
           if r["stats"]["merge_audit"]["distinct_full"]]
    per_keep = [r["stats"]["merge_audit"]["distinct_masked_keep_block"] / r["stats"]["merge_audit"]["distinct_full"] for r in rows
                if r["stats"]["merge_audit"]["distinct_full"]]
    return {"solves": len(rows), "distinct_full": full, "distinct_masked": masked, "distinct_masked_keep_block": keep,
            "ratio_pooled": round(masked / full, 4) if full else None, "ratio_keep_block_pooled": round(keep / full, 4) if full else None,
            "ratio_median": round(statistics.median(per), 4) if per else None,
            "ratio_keep_block_median": round(statistics.median(per_keep), 4) if per_keep else None}


def cmd_speed_jit(a) -> int:
    """The same decisions as `speed` through EmuDump under the JIT (`dotnet EmuDump.dll solve --snapshot`), timers off then on,
    each in one process: the JIT split beside the native one."""
    import subprocess  # noqa: PLC0415

    out = Path(a.out)
    sp = json.loads((out / "speed.json").read_text(encoding="utf-8"))
    pos = {p["id"]: p["doc"] for p in read_jsonl(out / "positions.jsonl")}
    docs = out / "speed_docs.jsonl"
    write_jsonl(docs, [{"id": i, "doc": pos[i]} for i in sp["ids"]])
    base = ["dotnet", a.emudump, "solve", "--snapshot", str(docs), "--engine", _engine_of(out), "--budget-nodes", str(sp["budget"]),
            "--budget-steps", str(a.step_cap), "--budget-ms", str(int(a.wall_s * 1000)), "--horizon", str(MAX_HORIZON)]
    res = {}
    for name, extra in (("off", []), ("timers", ["--phase-timers"]), ("off2", []), ("timers2", ["--phase-timers"])):
        t0 = time.perf_counter()
        p = subprocess.run(base + extra, capture_output=True, text=True, check=True, cwd=str(out))
        wall = time.perf_counter() - t0
        rows = []
        for line in p.stdout.splitlines():
            o = json.loads(line)
            st = o.get("stats") or {}
            rows.append({"id": o.get("id"), "status": {"exact": 0, "budget": 1}.get((o.get("expected") or {}).get("status"), -3), "stats": st})
        res[name] = {"wall_s": round(wall, 3), "rows": rows}
    (out / "speed_jit.json").write_text(json.dumps({"budget": sp["budget"], "runs": res}), encoding="utf-8")
    rep = {k: v for k, v in speed_report(res).items() if k in ("off", "off2", "timers", "timers2", "timers_split", "timers2_split", "timers_gap", "timers2_gap", "timer_overhead")}
    print(json.dumps(rep, indent=1))
    return 0


def cmd_curve(a) -> int:
    out = Path(a.out)
    dec = read_jsonl(out / "decisions.jsonl")
    first = {r["id"]: r for r in read_jsonl(out / a.prev_name)}
    second = {r["id"]: r for r in read_jsonl(out / a.second_name)} if a.second_name else {}
    solves = [merged_horizons(first[i], second.get(i)) for i in first]
    cap = max([r["cap"] for r in solves], default=0)
    doc = curve(dec, solves, cap)
    doc["re_solved_at_cap"] = len(second)
    doc["first_cap"] = max([r["cap"] for r in first.values()], default=0)
    agg = [r for r in solves if r["status"] != -3]
    n = sum(r["stats"]["nodes"] for r in agg)
    s = sum(r["stats"]["engine_steps"] for r in agg)
    doc["wide_roll_leaves"] = {"solves_with": sum(1 for r in agg if r["stats"]["wide_roll_leaves"]), "leaves": sum(r["stats"]["wide_roll_leaves"] for r in agg)}
    doc["steps_per_node"] = round(s / n, 2) if n else None
    (out / "curve.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps(doc, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "sample", "dump", "solve", "check", "speed", "speed-jit", "curve"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--seed", type=int, default=20261028)
        p.add_argument("--procs", type=int, default=12)
        p.add_argument("--source", action="append", default=[])
        p.add_argument("--agent-root", default="D:/spireline/referee/agent")
        p.add_argument("--runs", type=int, default=30)
        p.add_argument("--fights", type=int, default=40)
        p.add_argument("--cap", type=int, default=10**6)
        p.add_argument("--wall-s", type=float, default=120.0)
        p.add_argument("--only-not-exact-at", type=int, default=0)
        p.add_argument("--out-name", default="solves.jsonl")
        p.add_argument("--prev-name", default="solves.jsonl")
        p.add_argument("--second-name", default="")
        p.add_argument("--n", type=int, default=0)
        p.add_argument("--budgets", default="10000,100000,1000000")
        p.add_argument("--budget", type=int, default=100000)
        p.add_argument("--step-cap", type=int, default=STEP_CAP)
        p.add_argument("--debug-flags", type=int, default=0)
        p.add_argument("--audit", action="store_true")
        p.add_argument("--audit-only", action="store_true")
        p.add_argument("--emudump", default=str(ROOT / "tools" / "emudump" / "bin" / "Release" / "net9.0" / "EmuDump.dll"))
    a = ap.parse_args(argv)
    if a.cmd == "speed" and a.out_name == "solves.jsonl":
        a.out_name = "speed.json"
    return {"collect": cmd_collect, "sample": cmd_sample, "dump": cmd_dump, "solve": cmd_solve, "check": cmd_check,
            "speed": cmd_speed, "speed-jit": cmd_speed_jit, "curve": cmd_curve}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
