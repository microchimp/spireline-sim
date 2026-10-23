"""Leak probes for DESIGN.md 7.6 (clairvoyance, L2, L3, L8), emulator only; used by tests/python/test_lockstep.py.

Nothing here runs in a game. It reads the emulator through resampled clones only, the way the lockstep search does.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def native_env() -> None:
    """Point at the fork's native build and allow it to be older than a fresh checkout's sources."""
    # this tree's own build first (out/), else the fork's main working copy's
    lib = next((d for d in (ROOT / "out", ROOT.parent / "sts2" / "out") if (d / "Sts2Emulator.dll").exists()), None)
    if "STS2_LIB_PATH" not in os.environ and lib is not None:
        os.environ["STS2_LIB_PATH"] = str(lib)
    os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")
    for p in (ROOT / "rl", ROOT / "src", ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def play_to(seed: str, stop, *, ascension: int = 10, sims: int = 5, limit: int = 60):
    """A RunSession played by the rules policy until `stop(session, decision)` is true (decision None in a fight)."""
    native_env()
    import numpy as np  # noqa: PLC0415
    from run_train import engine, selfplay  # noqa: PLC0415

    s = engine.RunSession(seed, ascension, engine.CombatBudget(sims, sims))
    pol, rng = selfplay.RulesPlayer(), np.random.default_rng(0)
    for _ in range(limit):
        d = s.decision()
        if stop(s, d):
            return s
        if d is None:
            raise RuntimeError(f"{seed}: stopped in phase {s.phase} before the position was reached")
        s.apply(d.options[pol.pick(s.view(), d, rng).index])
    raise RuntimeError(f"{seed}: position not reached in {limit} decisions")


def resampled_clone(session, seed: int) -> int:
    """A run handle forked with resample_hidden=True (the only clone a search may use)."""
    from sts2_gym import native  # noqa: PLC0415

    return native.run_clone(session.handle, True, seed, (ctypes.c_int * native.RUN_OBS_SIZE)())


def encounter_spread(session, n: int = 200) -> set:
    """L2: encounter ids seen when n resampled clones of a map state enter the same monster node."""
    from sts2_gym import native  # noqa: PLC0415

    d = session.decision()
    opt = next(o for o in d.options if o.kind == "map.node" and o.node[2] == 1)
    ids = set()
    env = session.env
    for k in range(n):
        h = resampled_clone(session, 1000 + k)
        native.run_step(h, opt.action, -1, env._run_obs_buf, env._run_rew_buf, env._run_terminal_buf, env._run_truncated_buf)
        ids.add(native.run_encounter_id(h))
        native.run_destroy(h)
    return ids


def reward_spread(session, n: int = 200) -> set:
    """L3: the card reward (three cards) after winning the fight the session stands in, over n resampled clones."""
    from sts2_gym import native  # noqa: PLC0415

    seen = set()
    real = session.env._run_handle
    try:
        for k in range(n):
            session.env._run_handle = real
            h = resampled_clone(session, 2000 + k)
            session.env._run_handle = h
            native.run_debug_resolve_combat(h, session.env._run_obs_buf)
            d = None
            for _ in range(8):
                d = session.decision()
                if d is None or d.type == "card_reward":
                    break
                session.apply(d.options[0])
            if d is not None and d.type == "card_reward":
                seen.add(tuple(sorted((o.card.card_id, o.card.upgraded) for o in d.options if o.card)))
            native.run_destroy(h)
    finally:
        session.env._run_handle = real
    return seen


def move_spread(session, n: int = 200) -> dict:
    """L8: per enemy, the distinct (current move index, intent type, magnitude) over n resampled forks of the fight."""
    from sts2_gym import native  # noqa: PLC0415

    root = session.combat(fork_combat=True).combat_fork(None)
    out: dict = {}
    try:
        for k in range(n):
            w = root.fork(3000 + k)
            for e in native.get_state_json(w.handle)["enemies"]:
                out.setdefault((e["def_id"], e["slot"]), set()).add(
                    (e["ai"]["move_index"], e["intent"]["type"], e["intent"]["magnitude"]))
            native.destroy(w.handle)
    finally:
        native.destroy(root.handle)
    return out


def root_search(session, *, resample: bool, sims: int = 40, seed: int = 7, root_seed: int | None = None):
    """search_live on the fight the session stands in. `root_seed` re-deals the hidden draw order of the position
    searched (a permutation of the unseen pile) before searching; the visible state is unchanged."""
    from sts2_gym import native  # noqa: PLC0415
    import live_search  # noqa: PLC0415

    root = session.combat(fork_combat=True).combat_fork(root_seed)
    try:
        return live_search.search_live(root.handle, max_sims=sims, min_child_sims=2, confirm_pairs=0, time_budget_s=600.0,
                                       seed=seed, resample=resample)
    finally:
        native.destroy(root.handle)
