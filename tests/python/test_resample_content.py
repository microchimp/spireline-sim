"""Leak test L3 (Spireline DESIGN.md 7.6) on the emulator alone: shop stock and event outcomes before
the node is entered are redrawn by a resampled clone, and a faithful clone keeps them."""

import ctypes
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))

import leakprobes  # noqa: E402

try:
    leakprobes.native_env()
    from sts2_gym import native  # noqa: E402

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    HAVE_NATIVE = False

NODE_SHOP, NODE_EVENT = 4, 7


def _map_before(node_type, limit=500):
    """A run played (fights won by the debug resolver: only the map position matters) to a map decision
    that offers a node of `node_type`; the session and the option."""
    import numpy as np  # noqa: PLC0415
    from run_train import engine, selfplay  # noqa: PLC0415

    for seed in ("L3A", "L3B", "L3C", "L3D", "L3E", "L3F"):
        s = engine.RunSession(seed, 10, engine.CombatBudget(5, 5))
        pol, rng = selfplay.RulesPlayer(), np.random.default_rng(0)
        for _ in range(limit):
            d = s.decision()
            if d is None:
                if not native.run_debug_resolve_combat(s.handle, (ctypes.c_int * native.RUN_OBS_SIZE)()):
                    break
                continue
            hit = [o for o in d.options if o.kind == "map.node" and o.node[2] == node_type]
            if d.type == "map" and hit:
                return s, hit[0]
            s.apply(d.options[pol.pick(s.view(), d, rng).index])
        s.close()
    raise unittest.SkipTest(f"no seed reached a map with node type {node_type}")


def _enter(s, option, resample, seed):
    env = s.env
    h = native.run_clone(s.handle, resample, seed, (ctypes.c_int * native.RUN_OBS_SIZE)())
    bufs = (env._run_obs_buf, env._run_rew_buf, env._run_terminal_buf, env._run_truncated_buf)
    native.run_step(h, option.action, -1, *bufs)
    return h, bufs


@unittest.skipUnless(HAVE_NATIVE, "needs the native emulator library")
class ContentRedrawTests(unittest.TestCase):
    def test_shop_stock_differs_across_resampled_clones_and_not_across_faithful_ones(self):
        s, option = _map_before(NODE_SHOP)
        try:
            stocks = {True: set(), False: set()}
            for resample in (True, False):
                for k in range(200):
                    h, _ = _enter(s, option, resample, 2000 + k)
                    try:
                        stocks[resample].add(
                            tuple(native.run_state_list(h, i, n) for i, n in ((8, 7), (9, 3), (10, 3), (4, 14)))
                        )
                    finally:
                        native.run_destroy(h)
            self.assertEqual(len(stocks[False]), 1, "a faithful clone must keep the shop it will meet")
            self.assertGreater(len(stocks[True]), 1, "every resampled clone meets the same shop")
            self.assertTrue(any(any(part) for part in next(iter(stocks[True]))), "the stock readout is empty")
        finally:
            s.close()

    def test_event_and_its_outcome_differ_across_resampled_clones_and_not_across_faithful_ones(self):
        s, option = _map_before(NODE_EVENT)
        try:
            seen = {True: set(), False: set()}
            for resample in (True, False):
                for k in range(200):
                    h, bufs = _enter(s, option, resample, 4000 + k)
                    try:
                        event = int(native.run_info(h)[9])
                        mask = native.run_action_mask(h, 128)
                        action = next(i for i, v in enumerate(mask) if v)
                        native.run_step(h, action, -1, *bufs)
                        outcome = (
                            tuple(native.run_info(h)),
                            native.run_state_list(h, 0, 256),
                            native.run_state_list(h, 1, 64),
                        )
                        seen[resample].add((event, outcome))
                    finally:
                        native.run_destroy(h)
            self.assertEqual(len(seen[False]), 1, "a faithful clone must keep the event it will meet")
            self.assertGreater(len(seen[True]), 1, "every resampled clone meets the same event and outcome")
            self.assertGreater(len({e for e, _ in seen[True]}), 1, "the event itself is not redrawn")
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
