"""Round A3: the combat dev set (positions, long-search labels) and the search-settings tuning grid.

    python rl/a3_devset.py collect  --seeds-file referee/seeds/E0-2026-10-23.json --lo 300 --hi 500 --out DIR
    python rl/a3_devset.py manifest --out DIR [--n 200 --seed 20261023]
    python rl/a3_devset.py label    --seeds-file ... --out DIR [--procs 12]
    python rl/a3_devset.py report   --out DIR

Positions come from generation 0's emulator runs on E-block seeds (gen0-60 replayed exactly: same agent seed, same
run index), so a position is (E-block index, combat decision k). `collect` lists every combat decision with its
visible features; `manifest` draws a stratified sample from it (act x decision kind x stakes) with a fixed seed;
`label` replays each run to its positions, saves the position as an sts2-emu-snapshot/1 document, labels it with
long resampled searches, and runs the tuning grid there. All searches resample (the true hidden state is never
read); the replay itself is the agent's run. Outputs stay in DIR (outside the repo).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen0  # noqa: E402

LABEL = {"n": 4, "sims": 2000, "min_child_sims": 20, "c": 0.1, "confirm_pairs": 60}
GRID_SIMS = (30, 60, 120, 240)
GRID_M = (1, 2, 4)  # min_child_sims: the root sweep
GRID_C = (0.05, 0.1, 0.2)
GRID_REPS = 2


def stratum(r: dict) -> str:
    """Floor band (act 1 in three bands by floor, then act 2 and later as one) x decision kind x stakes."""
    f = int(r["floor"])
    band = "f01-06" if f <= 6 else "f07-12" if f <= 12 else "f13-17" if f <= 17 else "act2+"
    stakes = "high" if (r["hard"] or r["hp_frac"] < 0.5) else "low"
    return f"{band}/{r['kind']}/{stakes}"


def _pos_dict(k: int, pos, info: dict) -> dict:
    return {"k": k, "kind": pos.kind, "n_moves": pos.n_moves, "hp_frac": round(pos.hp_frac, 4), "hard": pos.hard,
            "act": pos.act, "floor": int(info["floor"]), "node_type": int(info["node_type"])}


def _collect_run(seed: str, ix: int) -> list[dict]:
    rows: list[dict] = []

    def hook(k, root, s, pos):
        rows.append({"ix": ix, "seed": seed, **_pos_dict(k, pos, s.info)})
        return False

    r = gen0.play_emu_run(seed, ix, "gen0-60", hook=hook)
    import gen_loop  # noqa: PLC0415

    gen_loop.replay_note(r.get("prior_path"), f"the run on seed {seed}")  # a log line, never a failure
    for x in rows:
        x["run_floor"] = r["floor"]
    return rows


def _key(st) -> tuple[int, int]:
    return (int(st.action), int(st.target))


def _same_move(st, label: dict) -> bool:
    return int(st.target) == label["target"] and (int(st.action) == label["action"] or int(st.action) in label["merged"])


def _label_run(seed: str, ix: int, ks: list[int], out: str) -> list[dict]:
    import live_search  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    want = set(ks)
    last = max(ks)
    rows: list[dict] = []

    def hook(k, root, s, pos):
        if k not in want:
            return False
        pid = f"e{ix:03d}-k{k:03d}"
        doc = native.get_state_json(root.handle)
        (Path(out) / "positions" / f"{pid}.json").write_text(json.dumps(doc), encoding="utf-8")
        rec = {"pid": pid, "ix": ix, "seed": seed, **_pos_dict(k, pos, s.info)}
        # The long searches: label = most visits summed over n independent resampled searches.
        visits: Counter = Counter()
        bests, labs = [], []
        for j in range(LABEL["n"]):
            t = time.perf_counter()
            res = live_search.search_live(root.handle, max_sims=LABEL["sims"], min_child_sims=LABEL["min_child_sims"],
                                          c=LABEL["c"], confirm_pairs=LABEL["confirm_pairs"], time_budget_s=900.0,
                                          seed=1_000_003 * (ix + 1) + 7919 * k + j, resample=True)
            labs.append((res, time.perf_counter() - t))
            for ch in res.children:
                visits[_key(ch)] += ch.visits
            bests.append(_key(res.best))
        top = max(visits, key=lambda x: (visits[x], -x[0], -x[1]))
        merged = next((list(ch.merged_actions) for res, _ in labs for ch in res.children if _key(ch) == top), [])
        label = {"action": top[0], "target": top[1], "merged": merged}
        kinds = {_key(ch): ch.kind for res, _ in labs for ch in res.children}
        # Regret of every move vs the label: paired confirmation gaps from the searches whose best was the label.
        gaps: dict = defaultdict(list)
        for res, _ in labs:
            if _key(res.best) != top:
                continue
            for ch in res.children:
                if ch.paired_diff_vs_best is not None:
                    gaps[_key(ch)].append(float(ch.paired_diff_vs_best))
        regret = {f"{a},{t}": round(sum(v) / len(v), 3) for (a, t), v in gaps.items()}
        rec["label"] = {**label, "kind": kinds.get(top), "seed_bests": [list(b) for b in bests],
                        "agree": sum(b == top for b in bests) / len(bests), "regret": regret,
                        "assumed": any(r.ai_state_assumed for r, _ in labs), "wall_s": round(sum(w for _, w in labs), 2)}
        # The tuning grid.
        grid = []
        for sims in GRID_SIMS:
            for m in GRID_M:
                for c in GRID_C:
                    for rep in range(GRID_REPS):
                        t = time.perf_counter()
                        res = live_search.search_live(root.handle, max_sims=sims, min_child_sims=m, c=c, confirm_pairs=0,
                                                      time_budget_s=120.0, resample=True,
                                                      seed=(ix * 1000 + k) * 100 + rep * 37 + m * 5 + GRID_SIMS.index(sims))
                        dt = time.perf_counter() - t
                        b = res.best
                        ok = _same_move(b, label)
                        grid.append([sims, m, c, rep, int(ok), regret.get(f"{b.action},{b.target}"), round(dt * 1000, 2),
                                     res.sims])
        rec["grid"] = grid
        rows.append(rec)
        return k >= last

    gen0.play_emu_run(seed, ix, "gen0-60", hook=hook)
    return rows


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 -- reported, the block goes on
        return [{"error": f"{type(e).__name__}: {e}"[:300], "args": [str(a) for a in args[:2]]}]


def _pool(tasks: list[tuple], fn, procs: int, out_file: Path) -> list[dict]:
    ctx = mp.get_context("spawn")
    rows: list[dict] = []
    with ctx.Pool(procs, initializer=gen0._init_worker, maxtasksperchild=10) as pool, open(out_file, "w", encoding="utf-8") as fh:  # noqa: SLF001
        for got in pool.imap_unordered(_star, [(fn, *t) for t in tasks]):
            for r in got:
                fh.write(json.dumps(r) + "\n")
                rows.append(r)
            fh.flush()
    return rows


def _star(args):
    return _safe(*args)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def make_manifest(cands: list[dict], n: int, seed: int) -> dict:
    """Stratified, seeded sample of decisions with 2+ distinct moves: equal shares per stratum, capped by what the
    stratum holds, the rest handed round-robin to strata with room. Pure (no native code)."""
    import random  # noqa: PLC0415

    pool = sorted((c for c in cands if c.get("n_moves", 0) >= 2 and "error" not in c), key=lambda c: (c["ix"], c["k"]))
    by: dict[str, list[dict]] = defaultdict(list)
    for c in pool:
        by[stratum(c)].append(c)
    names = sorted(by)
    take = dict.fromkeys(names, 0)
    left = min(n, len(pool))
    while left:
        open_ = [s for s in names if take[s] < len(by[s])]
        for s in open_:
            if left == 0:
                break
            take[s] += 1
            left -= 1
    rng = random.Random(seed)
    chosen = []
    for s in names:
        chosen += rng.sample(by[s], take[s])
    chosen.sort(key=lambda c: (c["ix"], c["k"]))
    return {"seed": seed, "n": len(chosen), "population": {s: len(by[s]) for s in names}, "sample": take,
            "positions": [{"pid": f"e{c['ix']:03d}-k{c['k']:03d}", "ix": c["ix"], "seed": c["seed"], "k": c["k"],
                           "stratum": stratum(c)} for c in chosen]}


# ------------------------------------------------------------------ the tuning report (pure)

DECISIVE_HP = 1.0  # a position is decisive if some move is at least this much worse than the label's (paired gap)


def _weights(rows: list[dict], manifest: dict) -> dict[str, float]:
    """Population weight per position: stratum population / stratum sample, normalized to mean 1."""
    pop, got = manifest["population"], Counter(stratum(r) for r in rows)
    w = {r["pid"]: pop[stratum(r)] / got[stratum(r)] for r in rows}
    z = sum(w.values()) / len(w)
    return {k: v / z for k, v in w.items()}


def grid_index(rows: list[dict]) -> dict:
    """{pid: {(sims, m, c): [(ok, regret, ms), ...]}}"""
    out: dict = {}
    for r in rows:
        g: dict = defaultdict(list)
        for sims, m, c, _rep, ok, reg, ms, _n in r["grid"]:
            g[(sims, m, c)].append((ok, reg, ms))
        out[r["pid"]] = g
    return out


def score(rows, w, idx, pick) -> dict:
    """Weighted agreement (all, decisive only), mean regret in HP and mean ms of the searches `pick(row)` names."""
    a = ad = rg = ms = wa = wd = wr = 0.0
    for r in rows:
        key = pick(r)
        wt = w[r["pid"]]
        dec = max(r["label"]["regret"].values(), default=0.0) >= DECISIVE_HP
        for ok, reg, t in idx[r["pid"]][key]:
            a += wt * ok
            wa += wt
            ms += wt * t
            if dec:
                ad += wt * ok
                wd += wt
            if reg is not None:
                rg += wt * reg
                wr += wt
    return {"agree": a / wa, "agree_decisive": ad / wd if wd else None, "regret_hp": rg / wr if wr else None,
            "ms": ms / wa}


def regret_by_pos(rows, idx, pick) -> dict[str, float]:
    out = {}
    for r in rows:
        regs = [reg for _ok, reg, _t in idx[r["pid"]][pick(r)] if reg is not None]
        out[r["pid"]] = sum(regs) / len(regs) if regs else 0.0
    return out


def agree_by_pos(rows, idx, pick) -> dict[str, float]:
    return {r["pid"]: sum(ok for ok, _r, _t in idx[r["pid"]][pick(r)]) / len(idx[r["pid"]][pick(r)]) for r in rows}


def paired_boot(a: dict, b: dict, w: dict, n: int = 2000, seed: int = 1) -> tuple[float, float, float]:
    """Weighted mean of a - b over positions, with a 95% bootstrap interval."""
    import random  # noqa: PLC0415

    keys = sorted(a)
    d = [(w[k], a[k] - b[k]) for k in keys]
    mean = sum(x * y for x, y in d) / sum(x for x, _ in d)
    rng = random.Random(seed)
    ms = []
    for _ in range(n):
        s = [d[rng.randrange(len(d))] for _ in d]
        ms.append(sum(x * y for x, y in s) / sum(x for x, _ in s))
    ms.sort()
    return mean, ms[int(0.025 * n)], ms[int(0.975 * n)]


def snap(n: float) -> int:
    return min(GRID_SIMS, key=lambda g: (abs(g - n), g))


def make_report(rows: list[dict], manifest: dict, cands: list[dict]) -> dict:
    import search_configs as sc  # noqa: PLC0415

    rows = [r for r in rows if "error" not in r]
    w = _weights(rows, manifest)
    idx = grid_index(rows)
    lab = [r["label"] for r in rows]
    doc: dict = {"n": len(rows), "label": {
        "seed_agreement": sum(x["agree"] for x in lab) / len(lab),
        "all_4_agree": sum(x["agree"] == 1.0 for x in lab) / len(lab),
        "decisive": sum(max(x["regret"].values(), default=0) >= DECISIVE_HP for x in lab) / len(lab),
        "assumed": sum(x["assumed"] for x in lab), "wall_s_mean": sum(x["wall_s"] for x in lab) / len(lab),
        "kind": Counter(x["kind"] for x in lab)}}
    doc["grid"] = [{"sims": s_, "m": m, "c": c, **score(rows, w, idx, lambda r, k=(s_, m, c): k)}
                   for s_ in GRID_SIMS for m in GRID_M for c in GRID_C]
    # Per stratum part (kind, stakes) at m=2, c=0.1 for every budget: where do more simulations pay?
    part = defaultdict(list)
    for r in rows:
        part[stratum(r).split("/", 1)[1]].append(r)
    doc["by_part"] = {p: {s_: score(rs, w, idx, lambda r, k=(s_, 2, 0.1): k) for s_ in GRID_SIMS} for p, rs in part.items()}
    # Schedules at matched cost: population mean sims over decisions with 2+ moves (one-move ones cost nothing).
    pop = [c for c in cands if c.get("n_moves", 0) >= 2 and "error" not in c]
    def pos_of(x):
        return sc.Position(kind=x["kind"], n_moves=x["n_moves"], hp_frac=x["hp_frac"], hard=bool(x["hard"]))
    scheds = []
    pop_pos = [pos_of(x) for x in pop]
    costs: dict = {}
    for base in (30, 60):
        for hi_mult in (1, 2):
            for many in (8, 99):
                for low_hp in (0.3, 0.5):
                    for tgt in (1, 2, 4):
                        for sel in (0.5, 1, 2, 4):
                            sched = sc.Schedule(high=base * hi_mult, low_hp=low_hp, many_moves=many,
                                                kind_mult={"target": tgt, "select": sel})
                            ck = (base, hi_mult, many, low_hp, tgt, sel)
                            costs[ck] = sum(snap(sc.SearchConfig("x", base, schedule=sched).sims_for(p_))
                                            for p_ in pop_pos) / len(pop_pos)
                            if costs[ck] > 63:
                                continue
                            for m in GRID_M:
                                for c in GRID_C:
                                    cfg = sc.SearchConfig("cand", base, m, c, schedule=sched)
                                    res = score(rows, w, idx, lambda r, cfg=cfg: (snap(cfg.sims_for(pos_of(r))), cfg.min_child_sims, cfg.c))
                                    scheds.append({"m": m, "c": c, "base": base, "high": base * hi_mult, "many_moves": many,
                                                   "low_hp": low_hp, "target_mult": tgt, "select_mult": sel,
                                                   "pop_mean_sims": round(costs[ck], 1), **res, "_cfg": cfg})
    scheds.sort(key=lambda x: x["regret_hp"])
    doc["n_schedules"] = len(scheds)
    flat_cmp = [sc.SearchConfig(f"flat60-m{m}-c{c}", 60, m, c) for m in GRID_M for c in GRID_C]
    ref = regret_by_pos(rows, idx, lambda r: (60, 2, 0.1))
    ref_ok = agree_by_pos(rows, idx, lambda r: (60, 2, 0.1))
    def cmp(cfg):
        pick = lambda r: (snap(cfg.sims_for(pos_of(r))), cfg.min_child_sims, cfg.c)  # noqa: E731
        return {"regret_vs_gen0_60": paired_boot(regret_by_pos(rows, idx, pick), ref, w),
                "agree_vs_gen0_60": paired_boot(agree_by_pos(rows, idx, pick), ref_ok, w)}
    for x in scheds[:10]:
        x.update(cmp(x.pop("_cfg")))
    for x in scheds[10:]:
        x.pop("_cfg")
    doc["schedules_top"] = scheds[:10]
    doc["flat60"] = {cfg.name: cmp(cfg) for cfg in flat_cmp}
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("collect", "manifest", "label", "report"))
    ap.add_argument("--seeds-file", default=str(gen0.ROOT / "referee/seeds/E0-2026-10-23.json"))
    ap.add_argument("--lo", type=int, default=300)
    ap.add_argument("--hi", type=int, default=500)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "collect":
        seeds = gen0.load_seeds(a.seeds_file)
        t0 = time.time()
        rows = _pool([(s, i) for i, s in enumerate(seeds) if a.lo <= i < a.hi], _collect_run, a.procs, out / "candidates.jsonl")
        print(len(rows), "decisions", round(time.time() - t0), "s")
    elif a.cmd == "manifest":
        m = make_manifest(read_jsonl(out / "candidates.jsonl"), a.n, a.seed)
        (out / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
        print(json.dumps({"n": m["n"], "sample": m["sample"]}))
    elif a.cmd == "report":
        doc = make_report(read_jsonl(out / "labels.jsonl"), json.loads((out / "manifest.json").read_text(encoding="utf-8")),
                          read_jsonl(out / "candidates.jsonl"))
        (out / "tuning.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(json.dumps({k: doc[k] for k in ("n", "label", "n_schedules")}, indent=1))
    else:
        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        (out / "positions").mkdir(exist_ok=True)
        runs: dict = defaultdict(list)
        for p in m["positions"]:
            runs[(p["seed"], p["ix"])].append(p["k"])
        tasks = sorted(((s, i, ks, str(out)) for (s, i), ks in runs.items()), key=lambda t: -len(t[2]))
        t0 = time.time()
        rows = _pool(tasks, _label_run, a.procs, out / "labels.jsonl")
        print(len(rows), "positions", round(time.time() - t0), "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
