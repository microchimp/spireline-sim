"""Decision 0, many seeds: does the search answer alike from the in-run root and from the loaded document? The same position, the search at the
agent's own schedule run with seeds 1..N on a faithful combat fork of the run and on the loaded document; the best move's frequency and every
move's mean value and visit share are compared (two-sample z on the per-seed values). At one seed the two roots need not agree (their
resampled streams differ), but over many seeds they must, or the search sees something different at the two roots.

    python -I doc_start_root.py --doors SEED:IX,... [--n 24]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doc_start_lib as L  # noqa: E402


def z(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    return [round(ma, 2), round(mb, 2), round((ma - mb) / se, 2) if se > 0 else (0.0 if ma == mb else 99.0)]


def many(world, cfg, pos, extra, n):
    import live_search

    out = []
    for sd in range(1, n + 1):
        kw = {**cfg.search_kwargs(pos, env_prior=False), **extra}
        r = live_search.search_live(world.handle, seed=sd, **kw)
        tot = sum(c.visits for c in r.children) or 1
        out.append({"best": (r.best.action, r.best.target, r.best.kind),
                    "share": {f"{c.action},{c.target},{c.kind}": c.visits / tot for c in r.children},
                    "value": {f"{c.action},{c.target},{c.kind}": (c.value_mean if c.value_mean == c.value_mean else None) for c in r.children},
                    "assumed": bool(r.ai_state_assumed), "sims": r.sims})
    return out


def main():
    import combat_mcts
    import search_configs
    from sts2_gym import native

    ap = argparse.ArgumentParser()
    ap.add_argument("--doors", required=True)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    L.lower()
    player = L.build_player()
    cfg = player.agent.search_config("a3-v1")
    extra = player.agent.search_extra()
    for x in a.doors.split(","):
        seed, ix = x.split(":")

        def on_door(s, info, seed=seed):
            fight = s.combat(fork_combat=True)
            root = fight.combat_fork(None)
            doc = native.get_state_json(root.handle)
            base = combat_mcts.Combat(native.load_combat_snapshot(doc, 11))
            dw = base.fork(None)
            dw.encounter_id = base.encounter_id = native.encounter_id(dw.handle)
            try:
                pos_r = search_configs.position_of(root, s.info)
                pos_d = search_configs.position_of(dw, {"node_type": int(info["node_type"]), "floor": int(info["floor"]), "max_hp": dw.max_hp()})
                ra, rb = many(root, cfg, pos_r, extra, a.n), many(dw, cfg, pos_d, extra, a.n)
            finally:
                root.close()
                dw.close()
                base.close()
            ba, bb = Counter(r["best"] for r in ra), Counter(r["best"] for r in rb)
            moves = sorted(set(k for r in ra + rb for k in r["share"]))
            res = {"seed": seed, "floor": int(info["floor"]), "n": a.n, "best_run": {str(k): v for k, v in ba.items()}, "best_doc": {str(k): v for k, v in bb.items()},
                   "assumed": [sum(r["assumed"] for r in ra), sum(r["assumed"] for r in rb)],
                   "share_z": {m: z([r["share"].get(m, 0.0) for r in ra], [r["share"].get(m, 0.0) for r in rb]) for m in moves},
                   "value_z": {m: z([r["value"].get(m) for r in ra if r["value"].get(m) is not None], [r["value"].get(m) for r in rb if r["value"].get(m) is not None]) for m in moves}}
            print(json.dumps(res), flush=True)
            return True

        L.walk_doors(seed, int(ix), player, on_door, acts=(0,))


if __name__ == "__main__":
    main()
