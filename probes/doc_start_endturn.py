"""The fixed move sequence of deliverable 2 at scale: end the turn until the fight ends, N resampled forks of the in-run fight and N of the loaded
document, and the player's HP after each turn compared (two-sample z per turn; the turn the fight ends on). Oracle forks agree turn by turn
where the enemy's moves are fixed (doc_start_diag.py); where its moves are random (an oracle's stream is the run's, a document's is seeded) only
the distributions can agree, and this is the test that they do.

    python -I doc_start_endturn.py --doors SEED:IX,... [--n 300]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doc_start_lib as L  # noqa: E402


def run_many(world, n, max_turns=12):
    import live_search
    from sts2_gym import native

    rows = []
    for i in range(n):
        w = world.fork(5000 + i)
        try:
            hp, ends = [], 0
            for _ in range(max_turns):
                if w.done:
                    break
                mask = native.valid_actions(w.handle, 32)
                legal = [k for k in range(32) if mask[k]]
                end_a = len(native.get_pile(w.handle, "hand"))
                live_search._step(w, end_a if end_a in legal else legal[0], -1)  # noqa: SLF001
                ends += 1
                hp.append(0 if w.done else int(w.hp()))
            rows.append({"hp": hp, "ended_on": ends if w.done else None})
        finally:
            w.close()
    return rows


def second_turn(world, n):
    """After one real turn (the oracle copy ends its turn), N resampled forks end the next: does the enemy's hidden move behind the intent it
    shows get redrawn at this root, as it does at the door? The player's HP after that turn, one value per fork."""
    import live_search
    from sts2_gym import native

    def end_turn(w):
        mask = native.valid_actions(w.handle, 32)
        legal = [k for k in range(32) if mask[k]]
        end_a = len(native.get_pile(w.handle, "hand"))
        live_search._step(w, end_a if end_a in legal else legal[0], -1)  # noqa: SLF001

    w1 = world.fork(None)
    try:
        end_turn(w1)
        if w1.done:
            return None
        out = []
        for i in range(n):
            f = w1.fork(9000 + i)
            try:
                end_turn(f)
                out.append(0 if f.done else int(f.hp()))
            finally:
                f.close()
        return out
    finally:
        w1.close()


def zt(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    return [round(ma, 2), round(mb, 2), round((ma - mb) / se, 2) if se > 0 else (0.0 if ma == mb else 99.0)]


def main():
    import combat_mcts
    from sts2_gym import native

    ap = argparse.ArgumentParser()
    ap.add_argument("--doors", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    L.lower()
    player = L.build_player()
    for x in a.doors.split(","):
        seed, ix = x.split(":")

        def on_door(s, info, seed=seed):
            fight = s.combat(fork_combat=True)
            root = fight.combat_fork(None)
            doc = native.get_state_json(root.handle)
            base = combat_mcts.Combat(native.load_combat_snapshot(doc, 11))
            dw = base.fork(None)
            try:
                ra, rb = run_many(root, a.n), run_many(dw, a.n)
                sa, sb = second_turn(root, a.n), second_turn(dw, a.n)
            finally:
                root.close()
                dw.close()
                base.close()
            turns = max(len(r["hp"]) for r in ra + rb)
            per_turn = []
            for t in range(turns):
                xa = [r["hp"][t] if t < len(r["hp"]) else 0 for r in ra]
                xb = [r["hp"][t] if t < len(r["hp"]) else 0 for r in rb]
                per_turn.append(zt(xa, xb))
            ends = [zt([r["ended_on"] or 99 for r in ra], [r["ended_on"] or 99 for r in rb])]
            print(json.dumps({"seed": seed, "floor": int(info["floor"]), "n": a.n, "hp_after_turn_run_doc_z": per_turn, "ended_on_run_doc_z": ends,
                              "second_turn_hp_run_doc_z": zt(sa, sb) if sa and sb else None, "second_turn_distinct": [len(set(sa or [])), len(set(sb or []))]}), flush=True)
            return True

        L.walk_doors(seed, int(ix), player, on_door, acts=(0,))


if __name__ == "__main__":
    main()
