"""Do the two worlds play alike? The search is out of it: the same fixed rollout policy (`live_search._rollout`, the tempo-weighted playout the
search itself uses at its leaves) played to the end from many resampled forks of (a) the in-run fight and (b) the loaded document, same fork
seeds, and the outcome (win, end HP) compared. A difference here is in the worlds' dynamics, not in what the search or the prior sees.

    python -I doc_start_dyn.py --out DIR --doors SEED:IX,... [--n 400] [--procs 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doc_start_lib as L  # noqa: E402


def rollouts(world, n, seed0):
    import live_search
    import numpy as np

    out = []
    for i in range(n):
        w = world.fork(seed0 + i)
        try:
            rng = np.random.default_rng(seed0 + i)
            live_search._rollout(w, rng)  # noqa: SLF001
            o = live_search._outcome(w, live_search.POTION_VALUE if hasattr(live_search, "POTION_VALUE") else 10.0)  # noqa: SLF001
            out.append([int(bool(w.won())), int(w.hp()), round(float(o.value), 2)])
        finally:
            w.close()
    return out


def one(task):
    seed, ix, a = task
    L.lower()
    player = L.build_player()
    res = {}

    def on_door(s, info):
        import combat_mcts
        from sts2_gym import native

        fight = s.combat(fork_combat=True)
        root = fight.combat_fork(None)
        doc = native.get_state_json(root.handle)
        base = combat_mcts.Combat(native.load_combat_snapshot(doc, 11))
        base.encounter_id = native.encounter_id(base.handle)
        dw = base.fork(None)
        dw.encounter_id = base.encounter_id
        try:
            r1, r2 = rollouts(root, a.n, 5000), rollouts(dw, a.n, 5000)
        finally:
            root.close()
            dw.close()
            base.close()

        def sm(r):
            return {"win": sum(x[0] for x in r) / len(r), "hp_win": (sum(x[1] for x in r if x[0]) / max(1, sum(x[0] for x in r))), "value": sum(x[2] for x in r) / len(r)}

        res[int(info["floor"])] = {"seed": seed, "floor": int(info["floor"]), "run": sm(r1), "doc": sm(r2), "n": a.n}
        (Path(a.out) / f"dyn_{seed}_f{info['floor']}.json").write_text(json.dumps(res[int(info["floor"])]))
        return int(info["floor"]) >= 33 or a.act1_only

    L.walk_doors(seed, ix, player, on_door)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--doors", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--act1-only", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tasks = [(x.split(":")[0], int(x.split(":")[1]), a) for x in a.doors.split(",")]
    if a.procs <= 1 or len(tasks) == 1:
        for t in tasks:
            print(one(t))
        return
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(a.procs) as pool:
        for r in pool.imap_unordered(one, tasks):
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
