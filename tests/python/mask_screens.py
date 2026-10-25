"""Shared by the mask-version tests (D32): a native run driven to its first card reward, holding a full belt."""

import ctypes
import unittest

import numpy as np


def potion_ids() -> dict:
    """Potion slug -> the emulator's potion id."""
    import lockstep  # noqa: PLC0415
    from run_train import vocab  # noqa: PLC0415

    return {vocab.game_slug(name): pid for name, pid in lockstep.EmuLock._id_map()["potions"].items()}


def to_card_reward(session, limit: int = 400):
    """Play the session with the rules player (fights won by the debug resolver) until a card reward is on screen."""
    from run_train import selfplay  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    pol, rng = selfplay.RulesPlayer(), np.random.default_rng(0)
    for _ in range(limit):
        d = session.decision()
        if d is None:
            if not native.run_debug_resolve_combat(session.handle, (ctypes.c_int * native.RUN_OBS_SIZE)()):
                break
            continue
        if d.type == "card_reward":
            return d
        session.apply(d.options[pol.pick(session.view(), d, rng).index])
    raise unittest.SkipTest("no card reward was reached")


def hold_belt(session, *slugs: str) -> None:
    """Put these potions on the session's belt (a three-slot belt, full when three are given)."""
    ids = potion_ids()
    session.env.debug_set_player(potions=[ids[s] for s in slugs], max_potion_slots=3)


def run_mask(handle: int) -> list:
    from sts2_gym import native  # noqa: PLC0415

    return [int(v) for v in native.run_action_mask(handle, native.RUN_MAX_ACTIONS)]
