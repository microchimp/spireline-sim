"""Shared pieces of the doc-start diagnosis (why a fight started from a document plays worse than the same fight inside the run).

Replays generation 13's runs (the E14-2026-11-08 seeds) exactly to their boss doors, as `doc_vs_run.py` does, and hands each door to a
callback with the live session. Nothing here edits the emulator or reads a human record.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

WT = Path(os.environ.get("DOC_START_TREE", str(Path(__file__).resolve().parents[1])))
for _p in (WT / "rl", WT / "src", WT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(WT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import run_teacher as rt  # noqa: E402

AGENT = rt.DEFAULT_AGENT


def lower() -> None:
    from sts2_gym import procutil

    procutil.lower_priority()


def build_player(agent_dir: str = AGENT):
    import hindsight

    return hindsight.build_player(agent_dir, config="a3-v1", own_schedule=True)


def walk_doors(seed: str, ix: int, player, on_door, acts=(0, 1)):
    """Replay run `seed` (stream index `ix`) as the measurement played it; at each boss door of the acts given call
    `on_door(session, info)`. The run goes on after the callback (a later door is reached too) unless it returns True."""
    import hindsight
    import numpy as np
    from run_train import engine

    rng = np.random.default_rng([rt.AGENT_SEED, int(ix)])
    explore = np.random.default_rng([rt.AGENT_SEED, int(ix), 1])
    s = hindsight.session_for(seed, player.agent, "a3-v1", own_schedule=True)
    seen: set[int] = set()
    try:
        while True:
            d = s.decision()
            if d is None and s.phase != engine.rc.PHASE_COMBAT:
                break
            if d is not None:
                s.apply(d.options[player.policy.pick(s.view(), d, explore).index])
                continue
            info = s.info
            if info["node_type"] == engine.vocab.NODE_BOSS and int(s.act_index) in acts and int(info["floor"]) not in seen:
                seen.add(int(info["floor"]))
                if on_door(s, info):  # True: stop here
                    break
            player.play_fight(s, rng, None)
    finally:
        s.close()
