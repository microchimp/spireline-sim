"""S6: the fight solver's coverage curve on real decisions (docs/rebuild/FIGHT_SOLVER.md sections 11 and 16).

    python tools/solver/coverage.py collect --source NIGHT_ARM_DIR [--source ...] --runs N --seed S --out OUT
    python tools/solver/coverage.py sample  --out OUT --fights 40 --seed S
    python tools/solver/coverage.py dump    --out OUT [--procs 12]
    python tools/solver/coverage.py solve   --out OUT --cap 1000000 --wall-s 120 [--only-not-exact-at 1000000] [--procs 12]
    python tools/solver/coverage.py check   --out OUT --n 20 --budgets 10000,100000,1000000
    python tools/solver/coverage.py speed   --out OUT --n 60 --budget 100000          (one process, a quiet machine)
    python tools/solver/coverage.py speed-jit --out OUT                                  (the same decisions through EmuDump)
    python tools/solver/coverage.py curve   --out OUT [--second-name solves_1e7.jsonl]
    python tools/solver/coverage.py memory  --out OUT --ids-file IDS --step-cap N   (one solve per process: peak working set)
    python tools/solver/coverage.py macro-ab --out OUT --n 120                        (EmuDump, macro draw on against off)
    python tools/solver/coverage.py played  --out OUT --source ARM [--source ...]      (the agent's move as applied, per decision)
    python tools/solver/coverage.py solve   --out OUT --utility lethal --horizon 0 --wall-s 1 --out-name solves_lethal.jsonl
    python tools/solver/coverage.py regret  --out OUT                                  (L0 regret and the two root checks)

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
STEP_CAP = 10**9  # engine runs per solve: dropped (the wall bounds a solve). The baseline used 10**5 to bound memory while an expansion
# held every outcome's clone; the streamed expansion keeps one per level (85-118 MB per worker at 3 x 10**5 runs, 0.1-2.3 GB before)
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


# ----------------------------------------------------------------------------------------------- M1: regret and the root checks


def q_map(row: dict) -> dict:
    """A solve row's root move values {(action, target): value}, target -1 for none; {} for a row written before q was kept."""
    return {(int(a), int(t)): float(v) for a, t, v in row.get("q") or []}


def _enemy_same(a: dict, b: dict) -> bool:
    return {k: v for k, v in a.items() if k != "slot"} == {k: v for k, v in b.items() if k != "slot"}


def map_played(q: dict, action: int, target: int, doc: dict | None) -> tuple[tuple[int, int] | None, str]:
    """The solver's move key for the agent's move as applied, and how it was found.

    'direct': (action, target) is a key. 'untargeted': the agent named a target for a move without one, or none for a move
    with a single target. 'merged': the solver lists only the earliest of equal moves (MultisetHand: a hand card equal to an
    earlier one, a potion equal to an earlier slot's; InterchangeableEnemies: an enemy equal to an earlier one, slot aside),
    so the move takes the value of the move it was merged into; equality is read from the decision's document. 'unmapped':
    none of these (the decision is left out of regret and counted)."""
    a, t = int(action), int(target)
    if (a, t) in q:
        return (a, t), "direct"

    def aim(act: int, how: str):
        ts = sorted(x for (y, x) in q if y == act)
        if not ts:
            return None
        if (act, t) in q:
            return (act, t), how
        if ts == [-1]:
            return (act, -1), "untargeted" if how == "direct" else how
        if t == -1 and len(ts) == 1:
            return (act, ts[0]), "untargeted" if how == "direct" else how
        enemies = (doc or {}).get("enemies") or []
        if 0 <= t < len(enemies):
            for x in ts:
                if 0 <= x < t and _enemy_same(enemies[x], enemies[t]):
                    return (act, x), "merged"
        return None

    got = aim(a, "direct")
    if got is not None:
        return got
    hand = ((doc or {}).get("piles") or {}).get("hand") or []
    if (doc or {}).get("pending_selection") is None and 0 <= a < len(hand):
        for j in range(a):
            if hand[j] == hand[a] and (got := aim(j, "merged")) is not None:
                return got
    elif (doc or {}).get("pending_selection") is None and a > len(hand):
        slots = ((doc or {}).get("potions") or {}).get("slots") or []
        s = a - len(hand) - 1
        if 0 <= s < len(slots):
            for j in range(s):
                if slots[j] == slots[s] and (got := aim(len(hand) + 1 + j, "merged")) is not None:
                    return got
    return None, "unmapped"


def source_short(label: str) -> str:
    """nightly/2026-10-27/g7-emu -> g7."""
    return str(label).rstrip("/").split("/")[-1].split("-")[0]


def check_reading(row: dict | None, action: int, target: int, doc: dict | None) -> dict:
    """One root check (lethal or survival) at a decision: finished (the check's solve was exact), whether some move's
    chance is exactly 1 (the root value, a double; a move's chance is a float32 of its exact value), the played move's
    chance, and missed (some move's chance is 1 and the played move's below 1). Exact quantities only: an unfinished check
    has no reading."""
    if row is None or row.get("status") != 0 or not row.get("exact"):
        return {"done": False}
    q = q_map(row)
    key, how = map_played(q, action, target, doc)
    avail = row.get("value") == 1.0
    p = q.get(key) if key is not None else None
    return {"done": True, "avail": avail, "played": p, "how": how, "missed": bool(avail and p is not None and p < 1.0)}


def regret_row(d: dict, solve: dict | None, played: dict | None, lethal: dict | None, survival: dict | None, doc: dict | None) -> dict:
    """One decision's reading: L0 regret where the L0 solve is exact (full fight), and the two root checks."""
    row = {"id": d["id"], "fight": d["fight"], "source": source_short(d["source"]), "kind": d["kind"], "won": d["won"],
           "turns_left": d["turns_left"], "played": played is not None, "replay_same": bool(played and played.get("replay_same"))}
    if played is None or not row["replay_same"]:
        return row
    a, t = int(played["action"]), int(played["target"])
    max_hp = int(((doc or {}).get("player") or {}).get("max_hp") or 0)
    if solve is not None and solve.get("status") == 0 and solve.get("exact") and solve.get("q"):
        q = q_map(solve)
        key, how = map_played(q, a, t, doc)
        best, worst = max(q.values()), min(q.values())
        row.update({"l0_exact": True, "how": how, "q_best": best, "q_worst": worst,
                    "decisive": best - worst > 0.01 or (d["turns_left"] <= 1 and (d["won"] is False or d["kind"] == "boss"))})
        if key is not None:
            loss = max(0.0, best - q[key])
            row.update({"regret": loss, "regret_hp": loss * 2 * max_hp})
    else:
        row["l0_exact"] = False
    row["lethal"] = check_reading(lethal, a, t, doc)
    row["survival"] = check_reading(survival, a, t, doc)
    return row


def regret_report(rows: list[dict], n_fights: int) -> dict:
    """The regret tables: overall, by source, fight kind and turns left. Per fight figures divide by the fights given."""

    def part(rs: list[dict], fights: int) -> dict:
        ex = [r for r in rs if r.get("l0_exact")]
        val = [r for r in ex if "regret" in r]
        tot, tot_hp = sum(r["regret"] for r in val), sum(r["regret_hp"] for r in val)
        dec = [r for r in val if r["decisive"]]
        out = {"decisions": len(rs), "fights": fights, "l0_exact": len(ex), "l0_valued": len(val),
               "l0_unmapped": len(ex) - len(val),
               "mean_regret_l0": round(tot / len(val), 6) if val else None,
               "regret_l0_per_fight": round(tot / fights, 6) if fights else None,
               "regret_hp_per_fight": round(tot_hp / fights, 4) if fights else None,
               "mean_regret_hp": round(tot_hp / len(val), 4) if val else None,
               "loss_over_0.01": sum(r["regret"] > 0.01 for r in val),
               "share_loss_over_0.01": _share([r["regret"] > 0.01 for r in val]),
               "decisive": len(dec), "decisive_regret_l0": round(sum(r["regret"] for r in dec), 6),
               "decisive_loss_over_0.01": sum(r["regret"] > 0.01 for r in dec)}
        for c in ("lethal", "survival"):
            done = [r for r in rs if (r.get(c) or {}).get("done")]
            missed = [r for r in done if r[c]["missed"]]
            mf = {r["fight"] for r in missed}
            out[c] = {"checked": len(done), "available": sum(r[c]["avail"] for r in done),
                      "unmapped": sum(r[c]["played"] is None for r in done), "missed_decisions": len(missed),
                      "fights_with_miss": len(mf), "missed_per_100_fights": round(100 * len(mf) / fights, 2) if fights else None,
                      "missed_ids": sorted(r["id"] for r in missed)}
        return out

    def by(key) -> dict:
        g: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            g[key(r)].append(r)
        return {k: part(v, len({r["fight"] for r in v})) for k, v in sorted(g.items())}

    return {"overall": part(rows, n_fights), "by_source": by(lambda r: r["source"]), "by_kind": by(lambda r: r["kind"]),
            "by_turns_left": by(lambda r: turns_left_bucket(r["turns_left"])),
            "mapping": dict(Counter(r.get("how", "-") for r in rows if r.get("l0_exact")))}


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


def _init_solver(engine: str, flags: int, prune: int | None = None) -> None:
    global PRUNE
    _below_normal()
    _setup_paths()
    import gen0  # noqa: PLC0415

    gen0.select_engine(engine)
    from sts2_gym import native  # noqa: PLC0415

    native.solver_debug(flags)
    PRUNE = prune


def peak_working_set_mb() -> float | None:
    """This process's peak working set in MB (GetProcessMemoryInfo), None off Windows."""
    if sys.platform != "win32":
        return None
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    c = Counters()
    c.cb = ctypes.sizeof(Counters)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return None
    return round(c.PeakWorkingSetSize / 2**20, 1)


PRUNE = None  # the solve's prunings (native.PRUNE_ALL when None); set per worker by _init_solver


def _solve(task: tuple) -> dict:
    """One decision's expected solve. The row keeps the root: q as [[action, target, value], ...] (target -1 for none),
    the best move, the value and exact_mass. A sixth and seventh entry give a check utility (native.UTILITY_*) and its
    fixed horizon."""
    pid, doc, cap, wall_s, step_cap, *rest = task
    utility, horizon = (int(rest[0]), int(rest[1])) if rest else (0, MAX_HORIZON)
    from sts2_gym import native  # noqa: PLC0415

    try:
        h = native.load_combat_snapshot(doc, 0)
    except Exception as e:  # noqa: BLE001 - a document the engine cannot load is a refusal of its own, counted
        return {"id": pid, "cap": cap, "wall_cap_s": wall_s, "step_cap": step_cap, "status": -3,
                "reason": f"snapshot-load: {type(e).__name__}: {str(e)[:160]}", "exact": False, "horizon": -1, "wall_s": 0.0,
                "censored": False, "stats": {"horizons": []}}
    try:
        opts = native.SolveOptions(node_budget=int(cap), step_budget=int(step_cap), time_budget_ms=int(wall_s * 1000), max_horizon=horizon,
                                   prune=native.PRUNE_ALL if PRUNE is None else PRUNE, utility=utility)
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
    row = {"id": pid, "cap": cap, "wall_cap_s": wall_s, "step_cap": step_cap, "status": r.status, "reason": r.reason, "exact": bool(r.exact),
           "horizon": r.horizon, "wall_s": round(wall, 4), "censored": censored, "stats": st, "peak_ws_mb": peak_working_set_mb(),
           "value": r.value, "exact_mass": r.exact_mass, "best": [r.best_action, r.best_target],
           "q": [[a, t, v] for (a, t), v in sorted(r.q.items())]}
    if utility:
        row["utility"] = utility
    return row


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
    if a.ids_file:  # one decision id per line (M1: the exact set)
        keep = {x.strip() for x in Path(a.ids_file).read_text(encoding="utf-8").splitlines() if x.strip()}
        ids = [i for i in ids if i in keep]
    if a.n:
        ids = sorted(random.Random(a.seed).sample(ids, min(a.n, len(ids))))  # a seeded subsample (the cut, reported)
    utility = {"l0": 0, "lethal": 3, "survival": 4}[a.utility]
    extra = (utility, MAX_HORIZON if a.horizon < 0 else a.horizon) if utility else ()
    tasks = [(i, pos[i], a.cap, a.wall_s, a.step_cap, *extra) for i in ids if i not in done]
    print(f"solving {len(tasks)} decisions (cap {a.cap} nodes, wall {a.wall_s} s, {a.procs} workers, utility {a.utility})", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(12, max(1, a.procs)), initializer=_init_solver, initargs=(_engine_of(out), a.debug_flags, a.prune), maxtasksperchild=4) as pool, \
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
    if a.ids_from:  # another run's speed.json: its decisions and budget, for a before/after on the same subset
        prev = json.loads(Path(a.ids_from).read_text(encoding="utf-8"))
        ids, a.budget = prev["ids"], prev["budget"]
    _init_solver(_engine_of(out), 0, a.prune)
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


def _memory(task: tuple) -> dict:
    r = _solve(task)
    return {k: r[k] for k in ("id", "status", "reason", "exact", "horizon", "wall_s", "peak_ws_mb")} | {
        "nodes": r["stats"].get("nodes"), "steps": r["stats"].get("engine_steps"), "step_cap": task[4], "cap": task[2]}


def cmd_memory(a) -> int:
    """One solve per worker process (a fresh process each): the process's peak working set after the solve."""
    out = Path(a.out)
    pos = {p["id"]: p["doc"] for p in read_jsonl(out / "positions.jsonl")}
    ids = [x.strip() for x in Path(a.ids_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    tasks = [(i, pos[i], a.cap, a.wall_s, a.step_cap) for i in ids]
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(processes=min(6, max(1, a.procs)), initializer=_init_solver, initargs=(_engine_of(out), 0, a.prune), maxtasksperchild=1) as pool:
        for r in pool.imap(_memory, tasks):
            rows.append(r)
            print(f"  {r['id'][-24:]} status {r['status']} steps {r['steps']} nodes {r['nodes']} {r['wall_s']} s peak {r['peak_ws_mb']} MB", flush=True)
    write_jsonl(out / a.out_name, rows)
    return 0


def cmd_macro_ab(a) -> int:
    """The macro draw on against off, pruned against pruned, through EmuDump (exact rationals): every root move's value and the
    root value must be equal wherever both solves finished the same horizon."""
    import subprocess  # noqa: PLC0415

    out = Path(a.out)
    pos = {p["id"]: p["doc"] for p in read_jsonl(out / "positions.jsonl")}
    pool = sorted(r["id"] for r in read_jsonl(Path(a.prev_name)) if r["status"] != -3 and exact_within_nodes(r["stats"].get("horizons") or [], a.budget))
    ids = sorted(random.Random(a.seed).sample(pool, min(a.n, len(pool))))
    docs = out / "macro_ab_docs.jsonl"
    write_jsonl(docs, [{"id": i, "doc": pos[i]} for i in ids])
    res = {}
    for name, prune in (("on", "all"), ("off", "-MacroDraw")):
        p = subprocess.run(["dotnet", a.emudump, "solve", "--snapshot", str(docs), "--engine", _engine_of(out), "--budget-nodes", str(a.budget * 10),
                            "--budget-steps", str(a.step_cap), "--budget-ms", str(int(a.wall_s * 1000)), "--horizon", str(MAX_HORIZON), "--prune", prune],
                           capture_output=True, text=True, check=True, cwd=str(out))
        res[name] = {json.loads(x)["id"]: json.loads(x) for x in p.stdout.splitlines()}
    rows, diffs = [], []
    for i in ids:
        on, off = res["on"][i].get("expected") or {}, res["off"][i].get("expected") or {}
        same_h = on.get("horizon") == off.get("horizon") and on.get("status") == off.get("status") and on.get("status") in ("exact", "budget")
        d = []
        if same_h:
            if on.get("value") != off.get("value"):
                d.append("value")
            mv = {(m["action"], m["target"]): m["value"] for m in off.get("moves") or []}
            d += [f"move {m['action']}/{m['target']}" for m in on.get("moves") or [] if mv.get((m["action"], m["target"])) != m["value"]]
        st_on, st_off = res["on"][i].get("stats") or {}, res["off"][i].get("stats") or {}
        rows.append({"id": i, "compared": same_h, "status": on.get("status"), "horizon": on.get("horizon"), "diffs": d,
                     "steps_on": st_on.get("engine_steps"), "steps_off": st_off.get("engine_steps"), "macro_nodes": st_on.get("macro_nodes"),
                     "macro_root": st_on.get("macro_root")})
        diffs += [f"{i}: {x}" for x in d]
    write_jsonl(out / "macro_ab.jsonl", rows)
    rep = {"decisions": len(rows), "compared": sum(r["compared"] for r in rows), "differences": len(diffs), "first": diffs[:10],
           "steps_on": sum(r["steps_on"] or 0 for r in rows if r["compared"]), "steps_off": sum(r["steps_off"] or 0 for r in rows if r["compared"])}
    (out / "macro_ab.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0 if not diffs else 1


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


def _played(task: tuple) -> dict:
    """Replays one run exactly, as _replay does, and reads the agent's combat move as applied at each wanted decision by
    wrapping native.run_step (the call itself unchanged); hashes the wanted decisions' documents for the exactness check."""
    import hashlib  # noqa: PLC0415

    source, label, agent_root, seed, ix, config, engine, keep = task
    _below_normal()
    _setup_paths()
    import gen0  # noqa: PLC0415

    gen0.select_engine(engine)
    import gen_loop  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    _, _, agent_dir, _ = _arm(source)
    agent = gen_loop.load_agent(Path(agent_root) / agent_dir)
    want = set(keep)
    moves: dict[int, list[int]] = {}
    hashes: dict[int, str] = {}
    cur = {"k": None}
    run_step = native.run_step

    def step(h, a, t, *rest):
        if cur["k"] is not None:
            moves[cur["k"]] = [int(a), int(t)]
            cur["k"] = None
        return run_step(h, a, t, *rest)

    def hook(k, root, s, pos):
        if k in want:
            cur["k"] = k
            hashes[k] = hashlib.sha256(json.dumps(native.get_state_json(root.handle), sort_keys=True).encode()).hexdigest()
        return False

    native.run_step = step
    t0 = time.perf_counter()
    try:
        r = gen0.play_emu_run(seed, ix, config, agent=agent, hook=hook)
    finally:
        native.run_step = run_step
    return {"label": label, "seed": seed, "floor": int(r["floor"]), "end": r.get("end"), "moves": moves, "hashes": hashes,
            "seconds": round(time.perf_counter() - t0, 1)}


def cmd_played(a) -> int:
    """played.jsonl: the agent's move as applied ({id, action, target}) at every decision of the sampled fights, from an
    exact replay of each run; the replay must reproduce the arm (floor and end) and every decision's document."""
    import hashlib  # noqa: PLC0415

    out = Path(a.out)
    meta = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    dec = read_jsonl(out / "decisions.jsonl")
    docs = {p["id"]: hashlib.sha256(json.dumps(p["doc"], sort_keys=True).encode()).hexdigest() for p in read_jsonl(out / "positions.jsonl")}
    want: dict[tuple, list[int]] = defaultdict(list)
    ixs = {}
    for d in dec:
        want[(d["source"], d["seed"])].append(d["k"])
        ixs[(d["source"], d["seed"])] = d["ix"]
    srcs = {source_label(s): s for s in a.source}
    tasks = []
    for (label, seed), ks in sorted(want.items()):
        m = next(x for x in meta["sources"] if x["label"] == label)
        tasks.append((srcs[label], label, a.agent_root, seed, ixs[(label, seed)], m["config"] or "a3-v1", m["engine"], sorted(ks)))
    rows, checks = [], []
    with _pool(a.procs) as pool:
        for res in pool.imap_unordered(_played, tasks):
            src = srcs[res["label"]]
            arm = next(r for r in read_jsonl(Path(src) / "runs.jsonl") if r["seed"] == res["seed"])
            same = res["floor"] == arm["floor"] and res["end"] == arm.get("end")
            ks = sorted(k for (lb, sd), v in want.items() if lb == res["label"] and sd == res["seed"] for k in v)
            ids = {k: f"{res['label']}|{res['seed']}|k{k:04d}" for k in ks}
            doc_same = sum(res["hashes"].get(str(k), res["hashes"].get(k)) == docs.get(ids[k]) for k in ks)
            checks.append({"source": res["label"], "seed": res["seed"], "floor": res["floor"], "end": res["end"], "arm_floor": arm["floor"],
                           "arm_end": arm.get("end"), "same": same, "decisions": len(ks), "docs_same": doc_same,
                           "moves": len(res["moves"]), "seconds": res["seconds"]})
            for k in ks:
                mv = res["moves"].get(str(k), res["moves"].get(k))
                if mv is not None:
                    # replay_same: the decision's document equals the one solved (the move is a3-v1's at that position);
                    # run_same: the whole run reproduced the arm's floor and end.
                    rows.append({"id": ids[k], "action": mv[0], "target": mv[1], "replay_same": res["hashes"].get(str(k), res["hashes"].get(k)) == docs.get(ids[k]),
                                 "run_same": same})
    rows.sort(key=lambda r: r["id"])
    write_jsonl(out / "played.jsonl", rows)
    write_jsonl(out / "played_check.jsonl", sorted(checks, key=lambda r: (r["source"], r["seed"])))
    rep = {"runs": len(checks), "runs_same": sum(c["same"] for c in checks), "decisions": len(dec), "played": len(rows),
           "docs_same": sum(c["docs_same"] for c in checks), "replay_same_rows": sum(r["replay_same"] for r in rows),
           "rows_in_runs_not_same": sum(not r["run_same"] for r in rows)}
    print(json.dumps(rep, indent=1))
    return 0


def cmd_regret(a) -> int:
    """regret.json: a3-v1's L0 regret where the L0 solve is exact, and its missed lethals and missed forced survivals where
    each check finished (ExpectedSolve values only; the hindsight bound is never read)."""
    out = Path(a.out)
    dec = read_jsonl(out / "decisions.jsonl")
    docs = {p["id"]: p["doc"] for p in read_jsonl(out / "positions.jsonl")}
    solves = {r["id"]: r for r in read_jsonl(out / a.prev_name)}
    played = {r["id"]: r for r in read_jsonl(out / "played.jsonl")}
    lethal = {r["id"]: r for r in read_jsonl(out / "solves_lethal.jsonl")}
    survival = {r["id"]: r for r in read_jsonl(out / "solves_survival.jsonl")}
    rows = [regret_row(d, solves.get(d["id"]), played.get(d["id"]), lethal.get(d["id"]), survival.get(d["id"]), docs.get(d["id"])) for d in dec]
    write_jsonl(out / "regret_rows.jsonl", rows)
    rep = regret_report(rows, len({d["fight"] for d in dec}))
    rep["inputs"] = {"decisions": len(dec), "l0_solves": len(solves), "played": len(played), "replay_same": sum(r["replay_same"] for r in rows),
                     "lethal_solves": len(lethal), "survival_solves": len(survival)}
    (out / "regret.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("overall", "mapping", "inputs")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "sample", "dump", "solve", "check", "speed", "speed-jit", "curve", "memory", "macro-ab", "played", "regret"):
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
        p.add_argument("--prune", type=int, default=None, help="the prunings' bits (native.PRUNE_*); all when absent")
        p.add_argument("--ids-from", default="", help="speed: another run's speed.json, whose decisions and budget are taken")
        p.add_argument("--ids-file", default="", help="memory: one decision id per line")
        p.add_argument("--debug-flags", type=int, default=0)
        p.add_argument("--utility", choices=("l0", "lethal", "survival"), default="l0", help="solve: a check utility, solved at --horizon")
        p.add_argument("--horizon", type=int, default=-1, help="solve: a check utility's fixed horizon")
        p.add_argument("--audit", action="store_true")
        p.add_argument("--audit-only", action="store_true")
        p.add_argument("--emudump", default=str(ROOT / "tools" / "emudump" / "bin" / "Release" / "net9.0" / "EmuDump.dll"))
    a = ap.parse_args(argv)
    if a.cmd == "speed" and a.out_name == "solves.jsonl":
        a.out_name = "speed.json"
    return {"collect": cmd_collect, "sample": cmd_sample, "dump": cmd_dump, "solve": cmd_solve, "check": cmd_check,
            "speed": cmd_speed, "speed-jit": cmd_speed_jit, "curve": cmd_curve, "memory": cmd_memory, "macro-ab": cmd_macro_ab,
            "played": cmd_played, "regret": cmd_regret}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
