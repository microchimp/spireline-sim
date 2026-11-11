"""A fight started from its exported document sees what the same fight inside the run sees (the doc-start diagnosis, 2026-11-11).

Every gym practice start, the fight exam, the convergence lane's labels and the judge corpus start from documents; the in-run player fights from
a combat fork of the run. At the position itself the two must agree on everything the search reads: the observation, the prior's tokens, the legal
moves, the resolved search kwargs (simulations, stakes multiplier, potion hold, kill check) and, at the same seed, the search's own answer.
The run's fight and its loaded document are compared at a first fight of a fixture run; the fight kinds and the stakes schedule are exercised by
giving both sides the same node type.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import leakprobes  # noqa: E402

try:
    leakprobes.native_env()
    from sts2_gym import native  # noqa: E402

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    HAVE_NATIVE = False

NODE_MONSTER, NODE_ELITE, NODE_BOSS = 1, 2, 6  # run_train.vocab


@unittest.skipUnless(HAVE_NATIVE, "needs the native emulator library")
class DoorBothWaysTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import combat_mcts
        from run_train import engine

        cls.s = leakprobes.play_to("DS1", lambda s, d: d is None and s.phase == engine.rc.PHASE_COMBAT)
        fight = cls.s.combat(fork_combat=True)
        cls.root = fight.combat_fork(None)  # the in-run position: a faithful fork of the run's fight
        cls.root.encounter_id = fight.encounter_id
        cls.doc = native.get_state_json(cls.root.handle)
        base = combat_mcts.Combat(native.load_combat_snapshot(cls.doc, 11))
        cls.world = base.fork(None)  # the same position from its document
        base.close()
        cls.world.encounter_id = native.encounter_id(cls.world.handle)
        cls.start = {"id": "ds|f1", "node_type": int(cls.s.info["node_type"]), "floor": int(cls.s.info["floor"])}

    @classmethod
    def tearDownClass(cls):
        cls.root.close()
        cls.world.close()
        cls.s.close()

    def test_the_observation_and_the_legal_moves_agree(self):
        import live_search

        self.assertEqual(list(self.root.obs), list(self.world.obs))
        self.assertEqual(self.root.legal(), self.world.legal())
        self.assertEqual(live_search._world_moves(self.root), live_search._world_moves(self.world))  # noqa: SLF001

    def test_the_encounter_is_the_same_on_every_handle(self):
        self.assertEqual(self.root.encounter_id, self.world.encounter_id)
        self.assertEqual(native.encounter_id(self.root.handle), native.encounter_id(self.world.handle))
        self.assertEqual(self.doc["encounter_id"], native.encounter_id(self.world.handle))

    def test_the_prior_reads_the_same_tokens(self):
        import numpy as np
        from combat_tokens import encode

        for probe in (False, True):
            a = encode(self.root.handle, self.root.obs, native.encounter_id(self.root.handle), probe=probe)
            b = encode(self.world.handle, self.world.obs, native.encounter_id(self.world.handle), probe=probe)
            self.assertTrue(np.array_equal(a.types, b.types), probe)
            self.assertTrue(np.array_equal(a.ids, b.ids), probe)
            self.assertTrue(np.array_equal(a.owners, b.owners), probe)
            self.assertTrue(np.array_equal(a.feats, b.feats), probe)

    def test_the_round_trip_of_the_document_is_the_document(self):
        import json

        h = native.load_combat_snapshot(self.doc, 3)
        try:
            again = native.get_state_json(h)
        finally:
            native.destroy(h)

        def norm(d):
            d = json.loads(json.dumps(d))
            u = d["piles"]["draw"]["unordered"]
            d["piles"]["draw"]["unordered"] = sorted(u, key=lambda c: json.dumps(c, sort_keys=True))
            return d

        self.assertEqual(norm(self.doc), norm(again))

    def test_the_resolved_search_kwargs_agree_for_every_fight_kind(self):
        import fight_gym
        import search_configs

        for config in ("a3-v1", "a3-v1@h1e4b16", "a3-v1@h1e4b16+hold30", "a3-v1-k"):
            cfg = search_configs.get(config)
            for node in (NODE_MONSTER, NODE_ELITE, NODE_BOSS):
                run_info = dict(self.s.info, node_type=node)
                start = dict(self.start, node_type=node)
                pos_run = search_configs.position_of(self.root, run_info)
                pos_doc = search_configs.position_of(self.world, fight_gym.position_info(start, self.world))
                self.assertEqual(pos_run, pos_doc, (config, node))
                self.assertEqual(cfg.search_kwargs(pos_run, env_prior=False), cfg.search_kwargs(pos_doc, env_prior=False), (config, node))
        # the stakes schedule reads the fight kind off the document's start: a boss at x16 is 16 times the same decision in a hallway fight
        sched = search_configs.get("a3-v1@h1e4b16")
        sims = {node: sched.search_kwargs(search_configs.position_of(self.world, fight_gym.position_info(dict(self.start, node_type=node), self.world)),
                                          env_prior=False)["max_sims"] for node in (NODE_MONSTER, NODE_ELITE, NODE_BOSS)}
        self.assertEqual((sims[NODE_ELITE], sims[NODE_BOSS]), (4 * sims[NODE_MONSTER], 16 * sims[NODE_MONSTER]))

    def test_the_search_runs_the_same_budget_from_either_handle_and_the_step_driver_is_the_live_search(self):
        import fight_gym
        import live_search
        import search_configs

        cfg = search_configs.get("a3-v1@h1e4b16")
        start = dict(self.start, node_type=NODE_BOSS)
        kw_run = cfg.search_kwargs(search_configs.position_of(self.root, dict(self.s.info, node_type=NODE_BOSS)), env_prior=False)
        kw_doc = cfg.search_kwargs(search_configs.position_of(self.world, fight_gym.position_info(start, self.world)), env_prior=False)
        kw_run["max_sims"] = kw_doc["max_sims"] = 24  # the schedule is compared above; the search itself is kept short here
        for seed in (1, 2):
            a = live_search.search_live(self.root.handle, seed=seed, **kw_run)
            b = live_search.search_live(self.world.handle, seed=seed, **kw_doc)
            c = live_search._drive(live_search.search_steps(self.world.handle, seed=seed, **kw_doc), None)  # noqa: SLF001
            # the two roots need not answer alike at one seed (their resampled streams stand at other call counts), but they search the same
            # budget over the same root moves
            self.assertEqual((a.sims, a.budget_exhausted), (b.sims, b.budget_exhausted))
            self.assertEqual(sorted((x.action, x.target, x.kind) for x in a.children), sorted((x.action, x.target, x.kind) for x in b.children))
            # one handle, two drivers (the in-run `search_live`, the gym's `search_steps`): call for call the same search
            self.assertEqual((b.sims, b.budget_exhausted, b.best), (c.sims, c.budget_exhausted, c.best))
            self.assertEqual([(x.action, x.target, x.visits, x.deaths, x.value_sum) for x in b.children],
                             [(x.action, x.target, x.visits, x.deaths, x.value_sum) for x in c.children])


if __name__ == "__main__":
    unittest.main()
