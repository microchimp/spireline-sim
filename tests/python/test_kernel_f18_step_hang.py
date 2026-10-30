"""F18 (a kernel finding, 2026-10-30): one combat step that never returns, while the process's memory grows.

The M2 fight gym's x16 game on start 2JYFVUUNTF|f17, seed j = 1, hung in a playout of its search at decision 9: one
`native.step` never returned and the worker's private memory grew by about 0.12 GB a minute (20 GB after some hours).
The state before that step is tests/python/fixtures/f18_step_hang.json: card 353 played at enemy 0 with every card of the
draw pile an Attack, one Skill in the discard, and the player's powers 10, 24, 26 and 28. Loaded with any seed, with or
without a resampled clone, the step does not return. The test runs it in a child process with a timeout; it is expected to
fail until the engine is fixed, and then reports an unexpected success.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "python" / "fixtures" / "f18_step_hang.json"
TIMEOUT_S = 30.0


def _step(queue) -> None:
    try:
        for p in (ROOT / "rl", ROOT / "src", ROOT):
            sys.path.insert(0, str(p))
        case = json.loads(FIXTURE.read_text(encoding="utf-8"))
        import gen0  # noqa: PLC0415

        gen0.select_engine(case["engine"])
        import combat_mcts  # noqa: PLC0415
        from sts2_gym import native  # noqa: PLC0415

        world = combat_mcts.Combat(native.load_combat_snapshot(case["doc"], 1))
        obs, reward = world.obs, world.reward
        queue.put(("loaded", None))
        done = native.step(world.handle, case["action"], obs, reward, case["target"])
        queue.put(("stepped", bool(done)))
    except Exception as error:  # noqa: BLE001 -- the parent skips when the native library is not there
        queue.put(("error", f"{type(error).__name__}: {error}"))


class StepHangTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_the_step_returns(self):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        p = ctx.Process(target=_step, args=(queue,), daemon=True)
        p.start()
        p.join(TIMEOUT_S)
        hung = p.is_alive()
        if hung:
            p.kill()  # this test's own child
            p.join()
        said = []
        while not queue.empty():
            said.append(queue.get())
        errors = [msg for kind, msg in said if kind == "error"]
        if errors:
            self.skipTest(f"native library unavailable: {errors[0]}")
        self.assertIn("loaded", [kind for kind, _ in said])
        self.assertFalse(hung, f"the step did not return within {TIMEOUT_S:.0f} s")


if __name__ == "__main__":
    unittest.main()
