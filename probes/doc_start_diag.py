"""The doc-start diagnosis (deliverables 1 and 2): why a boss fight started from its exported document plays worse than the same fight inside
the run. One process per door: replay the run to the door, then compare the two ways of standing in the same position.

  1. the position itself (observation, tokens, legal moves, resolved search kwargs, the encounter id) read off the in-run fork and off the loaded
     document; then the same search (same seeds) on both, and a played fight each way with every search call logged;
  2. the document round trip (export, load, export, diff: doc_start_roundtrip.py) and a fixed move sequence (end the turn until the fight ends)
     from the in-run fork and from the loaded document, oracle forks and resampled forks, enemy moves and HP turn by turn.

    python -I doc_start_diag.py --out DIR --doors SEED:IX,SEED:IX [--draws 2] [--searches 4] [--procs 4]

Reads no human record; the run seeds are the lab's own (generation 13, E14-2026-11-08).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doc_start_lib as L  # noqa: E402

rt = L.rt
LOG: list = []  # per-search records of the call being traced
_TRACE = {"on": False, "tag": "", "seed0": None}


def _sha(arr) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]


def tokens_of(handle: int, probe: bool):
    """The prior's input for the position on `handle`: the tokens `combat_tokens.encode` makes (types, ids, owners, features)."""
    import ctypes

    from combat_tokens import encode
    from sts2_gym import native

    obs = (ctypes.c_int * native.OBS_SIZE)()
    native.destroy(native.clone(handle, False, 0, obs))
    t = encode(handle, obs, native.encounter_id(handle), probe=probe)
    return t, obs


def tokens_diff(a, b) -> dict:
    import numpy as np

    out = {"len": [len(a), len(b)], "equal": False}
    if len(a) == len(b):
        out["types_equal"] = bool(np.array_equal(a.types, b.types))
        out["ids_equal"] = bool(np.array_equal(a.ids, b.ids))
        out["owners_equal"] = bool(np.array_equal(a.owners, b.owners))
        d = np.abs(a.feats - b.feats)
        out["feat_max_abs"] = float(d.max()) if d.size else 0.0
        rows, cols = np.nonzero(d > 1e-6)
        out["feat_diffs"] = [[int(r), int(c), int(a.types[r]), float(a.feats[r, c]), float(b.feats[r, c])] for r, c in list(zip(rows, cols))[:12]]
        out["equal"] = out["types_equal"] and out["ids_equal"] and out["owners_equal"] and out["feat_max_abs"] <= 1e-6
    return out


def moves_of(world) -> list:
    import live_search

    return [[m.action, m.target, m.kind] for m in live_search._world_moves(world)]  # noqa: SLF001


def kwargs_of(cfg, pos) -> dict:
    kw = cfg.search_kwargs(pos, env_prior=False)
    return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in kw.items()}


def pos_dict(pos) -> dict:
    return {"kind": pos.kind, "n_moves": pos.n_moves, "hp_frac": round(pos.hp_frac, 4), "hard": pos.hard, "act": pos.act, "fight": pos.fight}


def stat_of(res) -> dict:
    ch = [[c.action, c.target, c.kind, c.visits, round(c.value_mean, 3) if c.value_mean == c.value_mean else None, c.deaths] for c in res.children]
    return {"sims": res.sims, "exhausted": bool(res.budget_exhausted), "assumed": bool(res.ai_state_assumed), "elapsed": round(res.elapsed_s, 3),
            "best": [res.best.action, res.best.target, res.best.kind], "children": ch}


# ---------------------------------------------------------------------------------------------------------------- traced search calls


def install_trace():
    """Wrap `live_search.search_live` and `search_steps` so every search a fight plays leaves a record (kwargs, tokens, legal moves, outcome)."""
    import live_search
    from sts2_gym import native

    orig_live, orig_steps = live_search.search_live, live_search.search_steps

    def record(handle, kw, res, seed):
        t, _ = tokens_of(handle, probe=False)
        mask = native.valid_actions(handle, 32)
        LOG.append({"tag": _TRACE["tag"], "k": sum(1 for r in LOG if r["tag"] == _TRACE["tag"]), "seed": seed,
                    "kwargs": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else "<obj>") for k, v in kw.items()},
                    "tok_sha": [_sha(t.types), _sha(t.ids), _sha(t.owners), _sha(t.feats)], "n_tokens": len(t),
                    "legal": [i for i in range(32) if mask[i]], **stat_of(res)})

    def live(handle, **kw):
        if not _TRACE["on"]:
            return orig_live(handle, **kw)
        k = sum(1 for r in LOG if r["tag"] == _TRACE["tag"])
        if k == 0 and _TRACE["seed0"] is not None:
            kw["seed"] = _TRACE["seed0"]
        res = orig_live(handle, **kw)
        record(handle, kw, res, kw.get("seed"))
        return res

    def steps(handle, **kw):
        if not _TRACE["on"]:
            result = yield from orig_steps(handle, **kw)
            return result
        k = sum(1 for r in LOG if r["tag"] == _TRACE["tag"])
        if k == 0 and _TRACE["seed0"] is not None:
            kw["seed"] = _TRACE["seed0"]
        res = yield from orig_steps(handle, **kw)
        record(handle, kw, res, kw.get("seed"))
        return res

    live_search.search_live, live_search.search_steps = live, steps


def traced(tag, fn):
    _TRACE.update(on=True, tag=tag)
    try:
        return fn()
    finally:
        _TRACE["on"] = False


# ---------------------------------------------------------------------------------------------------------------- the fixed move sequence


def _buffs(bs):
    return sorted([b.get("id"), b.get("amount")] for b in bs) if bs and isinstance(bs[0], dict) else bs


def snap(handle: int) -> dict:
    """What the player sees of the position, from the exported document: HP, block, buffs, and each enemy's HP, block, intent and buffs."""
    from sts2_gym import native

    d = native.get_state_json(handle)
    p = d["player"]
    dr = d["piles"]["draw"]
    return {"turn": d["turn"], "hp": p["hp"], "block": p["block"], "buffs": _buffs(p["buffs"]),
            "enemies": [[e["def_id"], e["hp"], e["block"], e["intent"]["type"], e["intent"]["magnitude"], e["intent"]["hits"], _buffs(e["buffs"])] for e in d["enemies"]],
            "draw": len(dr.get("unordered", [])) + len(dr.get("known_top", [])) + len(dr.get("known_bottom", [])), "hand": len(d["piles"]["hand"])}


def end_turn_trace(world, max_turns: int = 12) -> list:
    """End the turn until the fight ends (or `max_turns`), reading the position after each: the fixed move sequence of deliverable 2."""
    import live_search
    from sts2_gym import native

    rows = [snap(world.handle)]
    for _ in range(max_turns):
        if world.done:
            break
        mask = native.valid_actions(world.handle, 32)
        legal = [i for i in range(32) if mask[i]]
        if not legal:
            break
        end_a = len(native.get_pile(world.handle, "hand"))  # end turn follows the hand
        if end_a not in legal:  # a selection screen: take the first legal answer
            end_a = legal[0]
        live_search._step(world, end_a, -1)  # noqa: SLF001
        rows.append(snap(world.handle) if not world.done else {"done": True, "won": bool(world.won()), "hp": world.hp()})
    return rows


def trace_summary(rows: list) -> list:
    """Per turn: the player's HP and block, each enemy's HP, block and intent, compactly."""
    out = []
    for r in rows:
        if r.get("done"):
            out.append({"done": True, "won": r["won"], "hp": r["hp"]})
        else:
            out.append({"hp": r["hp"], "blk": r["block"], "en": [[e[1], e[2], e[3], e[4], e[5]] for e in r["enemies"]]})
    return out


# ---------------------------------------------------------------------------------------------------------------- one door


def door_diag(s, info, seed, ix, player, a) -> dict:
    import combat_mcts
    import hindsight
    import live_search
    import numpy as np
    import search_configs
    from run_train import engine
    from sts2_gym import native

    cfg = player.agent.search_config("a3-v1")
    extra = player.agent.search_extra()
    fight = s.combat(fork_combat=True)
    root = fight.combat_fork(None)  # faithful (oracle) fork of the run's fight
    rec: dict = {"seed": seed, "ix": ix, "floor": int(info["floor"]), "hp": int(info["hp"]), "max_hp": int(info["max_hp"]), "cfg": cfg.name,
                 "extra_keys": sorted(extra)}
    doc = native.get_state_json(root.handle)
    rec["encounter_id_doc"] = doc["encounter_id"]
    base = combat_mcts.Combat(native.load_combat_snapshot(doc, 11))
    base.encounter_id = native.encounter_id(base.handle)
    doc_world = base.fork(None)  # the loaded document, faithful: what a search on the document's root looks at
    doc_world.encounter_id = base.encounter_id
    try:
        # ---- the position -------------------------------------------------------------------------------------------------------
        rec["encounter_id"] = {"run_fight": int(fight.encounter_id), "run_fork_handle": native.encounter_id(root.handle), "doc_handle": native.encounter_id(doc_world.handle)}
        rec["obs_equal"] = bool(list(root.obs) == list(doc_world.obs))
        rec["obs_diff"] = [[i, int(a_), int(b_)] for i, (a_, b_) in enumerate(zip(root.obs, doc_world.obs)) if a_ != b_][:30]
        ta, _ = tokens_of(root.handle, probe=False)
        tb, _ = tokens_of(doc_world.handle, probe=False)
        rec["tokens_noprobe"] = tokens_diff(ta, tb)
        ta, _ = tokens_of(root.handle, probe=True)
        tb, _ = tokens_of(doc_world.handle, probe=True)
        rec["tokens_probe"] = tokens_diff(ta, tb)
        rec["legal_run"], rec["legal_doc"] = root.legal(), doc_world.legal()
        rec["moves_run"], rec["moves_doc"] = moves_of(root), moves_of(doc_world)
        pos_run = search_configs.position_of(root, s.info)
        pos_doc = search_configs.position_of(doc_world, {"node_type": int(info["node_type"]), "floor": int(info["floor"]), "max_hp": doc_world.max_hp()})
        rec["pos_run"], rec["pos_doc"] = pos_dict(pos_run), pos_dict(pos_doc)
        rec["kwargs_run"], rec["kwargs_doc"] = kwargs_of(cfg, pos_run), kwargs_of(cfg, pos_doc)
        # ---- the same search on both, the same seeds ----------------------------------------------------------------------------
        rec["searches"] = []
        for sd in range(1, a.searches + 1):
            kw_run = {**cfg.search_kwargs(pos_run, env_prior=False), **extra}
            kw_doc = {**cfg.search_kwargs(pos_doc, env_prior=False), **extra}
            r1 = live_search.search_live(root.handle, seed=sd, **kw_run)
            r2 = live_search.search_live(doc_world.handle, seed=sd, **kw_doc)
            # and the doc side through the step driver the gym uses (`search_steps` driven by `_drive`): the same call for call?
            r3 = live_search._drive(live_search.search_steps(doc_world.handle, seed=sd, **kw_doc), extra.get("prior_fn"))  # noqa: SLF001
            rec["searches"].append({"seed": sd, "run": stat_of(r1), "doc": stat_of(r2), "doc_steps": stat_of(r3)})
        # ---- the fixed move sequence: end the turn until the fight ends -----------------------------------------------------------
        rec["endturn"] = {"oracle_run": trace_summary(end_turn_trace(root.fork(None))), "oracle_doc": trace_summary(end_turn_trace(doc_world.fork(None))),
                          "resampled_run": [], "resampled_doc": []}
        for n in range(a.sampled):
            w = root.fork(1000 + n)
            rec["endturn"]["resampled_run"].append(trace_summary(end_turn_trace(w)))
            w.close()
            w = doc_world.fork(1000 + n)
            rec["endturn"]["resampled_doc"].append(trace_summary(end_turn_trace(w)))
            w.close()
    finally:
        base.close()
        doc_world.close()
    # ---- played fights, every search logged ---------------------------------------------------------------------------------------
    import fight_gym

    fight_gym._W["agent"] = player.agent  # noqa: SLF001
    rec["games"] = []
    start = {"id": f"docstart:{seed}|f{info['floor']}", "node_type": int(info["node_type"]), "floor": int(info["floor"])}
    for j in range(a.draws):
        g = {"j": j}
        c = hindsight.clone_session(s, rt.resample_seed("docstart:" + seed, int(info["floor"]), j))
        try:
            rng = np.random.default_rng(rt.stream_key("docstart:" + seed, int(info["floor"]), j, 0))
            t0 = time.perf_counter()
            LOG.clear()
            _TRACE["seed0"] = 777 + j
            moves = traced(f"run{j}", lambda: player.play_fight(c, rng, time.perf_counter() + 1800))
            d = c.decision() if c.phase != engine.rc.PHASE_COMBAT else None
            g["run"] = {"won": bool(d is not None or c.won), "hp_run_block": int(c.info["hp"]), "moves": moves, "seconds": round(time.perf_counter() - t0, 1),
                        "searches": [r for r in LOG if r["tag"] == f"run{j}"]}
        finally:
            c.close()
        world = fight_gym._fresh_world(doc, start["id"], j, "dsd")  # noqa: SLF001
        try:
            t0 = time.perf_counter()
            _TRACE["seed0"] = 777 + j
            row = traced(f"doc{j}", lambda: fight_gym._play(world, start, j, cfg, extra, "dsd"))  # noqa: SLF001
            g["doc"] = {"won": bool(row["won"]), "end_hp": int(row["end_hp"]), "decisions": row["decisions"], "sims": row["sims"], "exhausted": row["budget_exhausted"],
                        "assumed": row["ai_assumed"], "seconds": round(time.perf_counter() - t0, 1), "searches": [r for r in LOG if r["tag"] == f"doc{j}"]}
        finally:
            world.close()
        rec["games"].append(g)
    root.close()
    return rec


def one(task):
    seed, ix, a = task
    L.lower()
    install_trace()
    player = L.build_player()
    out = Path(a.out)
    got = {}

    def on_door(s, info):
        got[int(info["floor"])] = door_diag(s, info, seed, ix, player, a)
        (out / f"{seed}_f{info['floor']}.json").write_text(json.dumps(got[int(info["floor"])]))
        return int(info["floor"]) >= (17 if a.act1_only else 33)

    L.walk_doors(seed, ix, player, on_door, acts=(0,) if a.act1_only else (0, 1))
    return seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--doors", required=True, help="SEED:IX,SEED:IX")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--searches", type=int, default=4)
    ap.add_argument("--sampled", type=int, default=12)
    ap.add_argument("--act1-only", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tasks = [(x.split(":")[0], int(x.split(":")[1]), a) for x in a.doors.split(",")]
    if a.procs <= 1 or len(tasks) == 1:
        for t in tasks:
            one(t)
        return
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(a.procs) as pool:
        for r in pool.imap_unordered(one, tasks):
            print("done", r, flush=True)


if __name__ == "__main__":
    main()
