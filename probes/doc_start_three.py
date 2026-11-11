"""Three ways to play a boss fight from the same door, to say where a document start's deficit comes from.

  A  the run clone, stepped on the run handle by the exam agent's fight player (`hindsight.SearchFighter`): the in-run way (`doc_vs_run.py`).
  B  a resampled combat fork of the run's fight (`combat_fork(seed)`), played by `fight_gym._play`: the gym's fight loop on a position that
     came out of the run, never exported.
  C  the exported document, loaded and played by `fight_gym._play`: the gym's way (`doc_vs_run.py`).

A against B: the run's own stepping against the combat engine's, and the HP the run block reads after the fight (it holds the rewards screen's
heals) against the fight's own. B against C: a position from the run against the same position through the document.

    python -I doc_start_three.py --out DIR --doors SEED:IX,... --draws 8 [--procs 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doc_start_lib as L  # noqa: E402

rt = L.rt


def measuring_fighter(agent, config, cfg):
    """The exam agent's fight player (`hindsight.SearchFighter`), unchanged in what it plays, that also reads the fight's own end HP: before each
    move is stepped on the run, the same move is stepped on the faithful combat copy the search already holds, which ends the fight at its own
    HP (the run block's HP holds the rewards screen's heals, Burning Blood's 6, as `RunCombat.combat_fork` says)."""
    import hindsight
    import live_search
    from run_train import engine
    from sts2_gym import native
    import search_configs

    class Measuring(hindsight.SearchFighter):
        fight_hp = None

        def __call__(self, s, rng, deadline=None):
            env, made = s.env, 0
            self.fight_hp = None
            while s.phase == engine.rc.PHASE_COMBAT:
                if deadline is not None and time.perf_counter() > deadline:
                    raise hindsight.Timeout("the continuation outlived its wall cap")
                root = s.combat(fork_combat=True).combat_fork(None)
                try:
                    pos = search_configs.position_of(root, s.info) if (self.cfg.schedule or self.cfg.needs_position) else None
                    kw = self.cfg.search_kwargs(pos, env_prior=False)
                    kw.update(self.extra)
                    res = live_search.search_live(root.handle, seed=int(rng.integers(1 << 30)), **kw)
                    live_search._step(root, res.best.action, res.best.target)  # noqa: SLF001
                    if root.done:
                        self.fight_hp = int(root.hp()) if root.won() else 0
                finally:
                    native.destroy(root.handle) if not hasattr(root, "close") else root.close()
                st = native.run_step(s.handle, res.best.action, res.best.target, env._run_obs_buf, env._run_rew_buf, env._run_terminal_buf, env._run_truncated_buf)
                if st != 0:
                    raise RuntimeError(f"combat action refused (status {st})")
                made += 1
            return made

    f = Measuring(agent, config, own_schedule=True)
    f.cfg = cfg
    return f


def _row(row):
    return {"won": bool(row.get("won")), "hp": int(row.get("end_hp") or 0) if row.get("won") else 0, "decisions": row.get("decisions"), "sims": row.get("sims"),
            "exhausted": row.get("budget_exhausted"), "assumed": row.get("ai_assumed"), "capped": row.get("capped")}


def door_games(s, info, seed, player, a, js):
    import fight_gym
    import hindsight
    import numpy as np
    from run_train import engine
    from sts2_gym import native

    import search_configs

    cfg = search_configs.get(a.config) if a.config else player.agent.search_config("a3-v1")
    extra = player.agent.search_extra()
    fight_gym._W["agent"] = player.agent  # noqa: SLF001
    fighter = measuring_fighter(player.agent, "a3-v1", cfg)  # the in-run player at the config under test; the replay goes on with the player's own
    start = {"id": f"three:{seed}|f{info['floor']}", "node_type": int(info["node_type"]), "floor": int(info["floor"])}
    fight = s.combat(fork_combat=True)
    root = fight.combat_fork(None)
    try:
        doc = native.get_state_json(root.handle)
    finally:
        root.close()
    out = {"seed": seed, "floor": int(info["floor"]), "hp": int(info["hp"]), "max_hp": int(info["max_hp"]), "games": []}
    for j in js:
        g = {"j": j}
        # A: the run clone, the run handle
        c = hindsight.clone_session(s, rt.resample_seed("three:" + seed, int(info["floor"]), j))
        try:
            rng = np.random.default_rng(rt.stream_key("three:" + seed, int(info["floor"]), j, 0))
            t0 = time.perf_counter()
            moves = fighter(c, rng, time.perf_counter() + 1800)
            d = c.decision() if c.phase != engine.rc.PHASE_COMBAT else None
            won = bool(d is not None or c.won)
            g["A"] = {"won": won, "hp": int(fighter.fight_hp or 0) if won else 0, "hp_run_block": int(c.info["hp"]) if won else 0, "hp_missing": bool(won and fighter.fight_hp is None), "decisions": moves,
                      "seconds": round(time.perf_counter() - t0, 1)}
        finally:
            c.close()
        # B: a resampled combat fork of the run's fight, played as the gym plays a world
        if "B" in a.ways:
            fight = s.combat(fork_combat=True)
            w = fight.combat_fork(rt.resample_seed("three-b:" + seed, int(info["floor"]), j))
            w.encounter_id = fight.encounter_id
            try:
                t0 = time.perf_counter()
                g["B"] = {**_row(fight_gym._play(w, start, j, cfg, extra, "thb")), "seconds": round(time.perf_counter() - t0, 1)}  # noqa: SLF001
            finally:
                w.close()
        # C: the document
        world = fight_gym._fresh_world(doc, start["id"], j, "thc")  # noqa: SLF001
        try:
            t0 = time.perf_counter()
            g["C"] = {**_row(fight_gym._play(world, start, j, cfg, extra, "thc")), "seconds": round(time.perf_counter() - t0, 1)}  # noqa: SLF001
        finally:
            world.close()
        out["games"].append(g)
    out["config"] = cfg.name
    return out


def one(task):
    seed, ix, js, a = task
    L.lower()
    player = L.build_player()
    res = {}

    def on_door(s, info):
        res[int(info["floor"])] = door_games(s, info, seed, player, a, js)
        name = f"three_{seed}_f{info['floor']}_{js[0]}-{js[-1]}.json"
        (Path(a.out) / name).write_text(json.dumps(res[int(info["floor"])]))
        return int(info["floor"]) >= (17 if a.act1_only else 33)

    L.walk_doors(seed, ix, player, on_door, acts=(0,) if a.act1_only else (0, 1))
    return seed, js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--doors", required=True)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2, help="draws per task (a replay costs about ten seconds)")
    ap.add_argument("--config", default="", help="a search config name for all three ways (default: the agent's own, a3-v1@h1e4b16)")
    ap.add_argument("--ways", default="ABC", help="which of A, B, C to play (A and C are the rerun of doc_vs_run.py)")
    ap.add_argument("--act1-only", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tasks = []
    for x in a.doors.split(","):
        seed, ix = x.split(":")
        for j0 in range(0, a.draws, a.chunk):
            js = list(range(j0, min(a.draws, j0 + a.chunk)))
            if not any((Path(a.out) / f"three_{seed}_f{f}_{js[0]}-{js[-1]}.json").exists() for f in (17, 33)):
                tasks.append((seed, int(ix), js, a))
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(a.procs) as pool:
        for r in pool.imap_unordered(one, tasks):
            print("done", r, flush=True)


if __name__ == "__main__":
    main()
