"""Do the run engine and the combat engine step the same fight alike? A run clone and a faithful combat fork of its fight are given the same
random legal move (by (action, target)) turn after turn; the observation (the run's opens with the combat's) is compared after every step.
A document is not involved: this is the A-against-B part of doc_start_three.py.

    python -I doc_start_step.py --doors SEED:IX,... [--trials 6] [--steps 60]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doc_start_lib as L  # noqa: E402

rt = L.rt


def trial(s, info, seed, t, steps):
    import hindsight
    import live_search
    import numpy as np
    from run_train import engine
    from sts2_gym import native

    c = hindsight.clone_session(s, rt.resample_seed("step:" + seed, int(info["floor"]), t))
    rng = np.random.default_rng([t, 99])
    out = {"t": t, "steps": 0, "first_diff": None}
    try:
        fight = c.combat(fork_combat=True)
        w = fight.combat_fork(None)  # a faithful copy of the clone's fight: same hidden state
        w.encounter_id = fight.encounter_id
        env = c.env
        try:
            n_obs = native.OBS_SIZE
            for k in range(steps):
                if w.done or c.phase != engine.rc.PHASE_COMBAT:
                    break
                moves = live_search._distinct(live_search._world_moves(w))  # noqa: SLF001
                if not moves:
                    break
                m = moves[int(rng.integers(len(moves)))]
                live_search._step(w, m.action, m.target)  # noqa: SLF001
                st = native.run_step(c.handle, m.action, m.target, env._run_obs_buf, env._run_rew_buf, env._run_terminal_buf, env._run_truncated_buf)
                out["steps"] = k + 1
                if st != 0:
                    out["first_diff"] = {"k": k, "what": f"run step refused {st}"}
                    break
                if w.done or c.phase != engine.rc.PHASE_COMBAT:
                    out["ended"] = {"k": k, "combat_done": bool(w.done), "run_phase": int(c.phase), "combat_hp": int(w.hp()), "run_hp": int(c.info["hp"])}
                    break
                ro = list(env._run_obs_buf[:n_obs])
                co = list(w.obs)
                if ro != co:
                    out["first_diff"] = {"k": k, "move": [m.action, m.target, m.kind], "idx": [[i, int(a), int(b)] for i, (a, b) in enumerate(zip(ro, co)) if a != b][:12]}
                    break
        finally:
            w.close()
    finally:
        c.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doors", required=True)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    L.lower()
    player = L.build_player()
    for x in a.doors.split(","):
        seed, ix = x.split(":")

        def on_door(s, info, seed=seed):
            res = [trial(s, info, seed, t, a.steps) for t in range(a.trials)]
            print(json.dumps({"seed": seed, "floor": int(info["floor"]), "trials": res}), flush=True)
            return True

        L.walk_doors(seed, int(ix), player, on_door, acts=(0,))


if __name__ == "__main__":
    main()
