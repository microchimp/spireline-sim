"""D32/D34: agent.json carries mask_version (absent: 1, so generations 0-5 and every saved candidate; 2, 3 and 4 are read as
they are); a candidate written by train_generation has 4 (test_gen_loop pins that); the run an agent plays is built with it, and the version only changes
what the run's mask carries."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("rl", "referee", "tests/python"):
    sys.path.insert(0, str(ROOT / sub))

import leakprobes  # noqa: E402

try:
    leakprobes.native_env()
    import gen0  # noqa: E402
    import gen_loop  # noqa: E402
    import mask_screens  # noqa: E402
    from run_train import engine  # noqa: E402
    from sts2_gym import native  # noqa: E402

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    HAVE_NATIVE = False

BELT = ("BLOCK_POTION", "BLOOD_POTION", "FRUIT_JUICE")


def agent_dir(tmp: str, meta: dict | None) -> Path:
    d = Path(tmp) / "gen3"
    d.mkdir()
    if meta is not None:
        (d / "agent.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


@unittest.skipUnless(HAVE_NATIVE, "needs the native library (run API 31)")
class AgentMaskVersionTests(unittest.TestCase):
    def test_an_agent_without_the_key_is_version_one_and_the_key_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(gen_loop.load_agent(agent_dir(tmp, {"generation": 3, "macro": "rules"})).mask_version, 1)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(gen_loop.load_agent(agent_dir(tmp, None)).mask_version, 1)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(gen_loop.load_agent(agent_dir(tmp, {"generation": 4, "mask_version": 2})).mask_version, 2)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(gen_loop.load_agent(agent_dir(tmp, {"generation": 6, "mask_version": 3})).mask_version, 3)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(gen_loop.load_agent(agent_dir(tmp, {"generation": 8, "mask_version": 4})).mask_version, 4)
        self.assertEqual(gen_loop.Agent().mask_version, 1)
        for bad in (0, 5, "2", "3", "4", True):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                gen_loop.load_agent(agent_dir(tmp, {"generation": 4, "mask_version": bad}))

    def test_a_candidates_version_constant_is_four_and_the_accepted_versions_are_the_bindings(self):
        self.assertEqual(gen_loop.CANDIDATE_MASK_VERSION, 4)
        self.assertEqual(gen_loop.AGENT_MASK_VERSIONS, tuple(native.RUN_MASK_VERSIONS))

    def test_a_version_one_agents_run_is_built_as_it_always_was_and_its_card_reward_mask_is_todays(self):
        self.assertEqual(gen0.session_kwargs(gen_loop.Agent()), {})
        self.assertEqual(gen0.session_kwargs(gen_loop.Agent(), "SILENT"), {"character": "SILENT"})
        s = engine.RunSession("D32AGENT1", 0, engine.CombatBudget(4, 4), **gen0.session_kwargs(gen_loop.Agent()))
        try:
            self.assertEqual(s.mask_version, 1)
            mask_screens.to_card_reward(s)
            mask_screens.hold_belt(s, *BELT)
            mask = mask_screens.run_mask(s.handle)
            self.assertFalse(any(mask[native.RUN_USE_POTION_ACTION:]))
            self.assertEqual([i for i, v in enumerate(mask) if v], [0, 1, 2, 3])
        finally:
            s.close()

    def test_a_version_two_agents_run_has_the_discard_on_the_mask_and_plays_the_same_screen(self):
        agent = gen_loop.Agent(mask_version=2)
        self.assertEqual(gen0.session_kwargs(agent), {"mask_version": 2})
        s = engine.RunSession("D32AGENT1", 0, engine.CombatBudget(4, 4), **gen0.session_kwargs(agent))
        try:
            self.assertEqual(s.mask_version, 2)
            d = mask_screens.to_card_reward(s)
            mask_screens.hold_belt(s, *BELT)
            mask = mask_screens.run_mask(s.handle)
            self.assertEqual([i for i, v in enumerate(mask) if v], [0, 1, 2, 3, 248, 249, 250])
            self.assertEqual(engine._legal(s.handle), [0, 1, 2, 3])  # the rules and the lookahead still see today's screen
            self.assertEqual([o.action for o in s.decision().options], [o.action for o in d.options])
        finally:
            s.close()

    def test_a_version_three_agents_run_has_the_use_of_the_anytime_potions_on_the_mask_and_plays_the_same_screen(self):
        agent = gen_loop.Agent(mask_version=3)
        self.assertEqual(gen0.session_kwargs(agent), {"mask_version": 3})
        s = engine.RunSession("D34AGENT1", 0, engine.CombatBudget(4, 4), **gen0.session_kwargs(agent))
        try:
            self.assertEqual(s.mask_version, 3)
            d = mask_screens.to_card_reward(s)
            mask_screens.hold_belt(s, *BELT)
            mask = mask_screens.run_mask(s.handle)
            # Blood Potion (slot 1) and Fruit Juice (slot 2) can be used; the Block Potion is combat-only
            self.assertEqual([i for i, v in enumerate(mask) if v], [0, 1, 2, 3, 243, 244, 248, 249, 250])
            self.assertEqual(engine._legal(s.handle), [0, 1, 2, 3])  # the policies still hide 242..253
            self.assertFalse(s.env.action_masks()[native.RUN_USE_POTION_ACTION:].any())
            self.assertEqual([o.action for o in s.decision().options], [o.action for o in d.options])
        finally:
            s.close()

    def test_a_version_four_agents_run_has_version_threes_mask_and_the_policies_still_hide_the_belt(self):
        agent = gen_loop.Agent(mask_version=4)
        self.assertEqual(gen0.session_kwargs(agent), {"mask_version": 4})
        s = engine.RunSession("D34AGENT4", 0, engine.CombatBudget(4, 4), **gen0.session_kwargs(agent))
        try:
            self.assertEqual(s.mask_version, 4)
            d = mask_screens.to_card_reward(s)
            mask_screens.hold_belt(s, *BELT)
            mask = mask_screens.run_mask(s.handle)
            self.assertEqual([i for i, v in enumerate(mask) if v], [0, 1, 2, 3, 243, 244, 248, 249, 250])
            self.assertEqual(engine._legal(s.handle), [0, 1, 2, 3])
            self.assertFalse(s.env.action_masks()[native.RUN_USE_POTION_ACTION:].any())
            self.assertEqual([o.action for o in s.decision().options], [o.action for o in d.options])
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
