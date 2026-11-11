"""Do resampled forks of the in-run fight and of the loaded document draw the same hidden information? No search, no prior: N forks of each
(fork seeds 1..N), the player ends the turn at once (S1) or plays a uniformly random legal turn first (S2), and what the next turn shows is
compared: the hand drawn (each card's frequency), the enemies' next intents, HP and block. Two-sample z on means; chi-square on counts.

    python -I doc_start_fork.py --doors SEED:IX,... [--n 1000]
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


def play_turn(w, rng, random_turn: bool):
    import live_search
    from sts2_gym import native

    steps = 0
    while not w.done and steps < 40:
        moves = live_search._distinct(live_search._world_moves(w))  # noqa: SLF001
        if not moves:
            return
        ends = [m for m in moves if m.kind == "end_turn"]
        if not random_turn and ends:
            live_search._step(w, ends[0].action, ends[0].target)  # noqa: SLF001
            return
        m = moves[int(rng.integers(len(moves)))]
        live_search._step(w, m.action, m.target)  # noqa: SLF001
        steps += 1
        if m.kind == "end_turn":
            return


def sample(world, n, random_turn):
    import numpy as np
    from sts2_gym import native

    rows = []
    for i in range(n):
        w = world.fork(i + 1)
        try:
            rng = np.random.default_rng(i + 1)
            play_turn(w, rng, random_turn)
            if w.done:
                rows.append({"done": True, "won": bool(w.won())})
                continue
            d = native.get_state_json(w.handle)
            hand = Counter(c["def_id"] for c in d["piles"]["hand"])
            rows.append({"hp": d["player"]["hp"], "block": d["player"]["block"], "energy": d["player"]["energy"], "hand": dict(hand), "n_hand": len(d["piles"]["hand"]),
                         "en_hp": sum(e["hp"] for e in d["enemies"]), "intents": [[e["intent"]["type"], e["intent"]["magnitude"], e["intent"]["hits"]] for e in d["enemies"]],
                         "draw": len(d["piles"]["draw"].get("unordered", [])) + len(d["piles"]["draw"].get("known_top", [])), "disc": len(d["piles"]["discard"])})
        finally:
            w.close()
    return rows


def zmean(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    return (round(ma, 3), round(mb, 3), round((ma - mb) / se, 2) if se > 0 else (0.0 if ma == mb else 99.0))


def compare(ra, rb):
    ok_a, ok_b = [r for r in ra if "hp" in r], [r for r in rb if "hp" in r]
    out = {"n": [len(ra), len(rb)], "done": [sum(1 for r in ra if r.get("done")), sum(1 for r in rb if r.get("done"))]}
    for k in ("hp", "block", "energy", "n_hand", "en_hp", "draw", "disc"):
        out[k] = zmean([r[k] for r in ok_a], [r[k] for r in ok_b])
    cards = set(c for r in ok_a + ok_b for c in r["hand"])
    ca = {c: sum(r["hand"].get(c, 0) for r in ok_a) / max(1, len(ok_a)) for c in cards}
    cb = {c: sum(r["hand"].get(c, 0) for r in ok_b) / max(1, len(ok_b)) for c in cards}
    out["hand_freq_max_gap"] = round(max((abs(ca[c] - cb[c]) for c in cards), default=0.0), 3)
    ia = Counter(json.dumps(r["intents"]) for r in ok_a)
    ib = Counter(json.dumps(r["intents"]) for r in ok_b)
    keys = set(ia) | set(ib)
    out["intent_tv"] = round(0.5 * sum(abs(ia[k] / max(1, len(ok_a)) - ib[k] / max(1, len(ok_b))) for k in keys), 3)
    out["intents_run"] = {k: v for k, v in list(ia.most_common(4))}
    out["intents_doc"] = {k: v for k, v in list(ib.most_common(4))}
    return out


def main():
    import combat_mcts
    from sts2_gym import native

    ap = argparse.ArgumentParser()
    ap.add_argument("--doors", required=True)
    ap.add_argument("--n", type=int, default=1000)
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
                res = {"seed": seed, "floor": int(info["floor"]), "S1": compare(sample(root, a.n, False), sample(dw, a.n, False)),
                       "S2": compare(sample(root, a.n, True), sample(dw, a.n, True))}
            finally:
                root.close()
                dw.close()
                base.close()
            print(json.dumps(res), flush=True)
            return True

        L.walk_doors(seed, int(ix), player, on_door, acts=(0,))


if __name__ == "__main__":
    main()
