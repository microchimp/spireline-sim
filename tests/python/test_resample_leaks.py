"""Leak tests of Spireline DESIGN.md 7.6 on the emulator alone (L2, L3, L8, clairvoyance): what a
`clone(resample_hidden=True)` hides. Runs under pytest or `python -m unittest`; skips without the
native library. Probes in referee/leakprobes.py."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))

import leakprobes  # noqa: E402

try:
    leakprobes.native_env()
    from sts2_gym import native  # noqa: E402,F401

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    HAVE_NATIVE = False


@unittest.skipUnless(HAVE_NATIVE, "needs the native emulator library")
class LeakTests(unittest.TestCase):
    """DESIGN.md 7.6 on the emulator: what a resampled clone hides."""

    @classmethod
    def setUpClass(cls):
        cls.fights = [leakprobes.play_to(seed, lambda s, d: d is None and s.phase == 0) for seed in ("LK1", "LK2", "LK3", "LK4")]

    @classmethod
    def tearDownClass(cls):
        for s in cls.fights:
            s.close()

    def test_L2_resampled_clones_entering_one_monster_node_differ_in_encounter(self):
        s = leakprobes.play_to("L2A", lambda s, d: d is not None and d.type == "map")
        try:
            ids = leakprobes.encounter_spread(s, 200)
        finally:
            s.close()
        self.assertGreater(len(ids), 1, f"every resampled clone enters encounter {ids}: the encounter is not hidden")

    def test_L3_resampled_clones_differ_in_reward_contents(self):
        cards = leakprobes.reward_spread(self.fights[0], 200)
        self.assertGreater(len(cards), 1, "every resampled clone offers the same card reward")

    def test_L8_resampled_forks_keep_every_shown_intent(self):
        """L8's firewall half: a fork re-rolls the current moves (IntentParticles) but never the intent on
        screen. Its other half, more than one move behind one intent, has no case in the emulator's tables:
        every move a monster can roll there shows an intent no other move it could roll at that point shows
        (and `move_index`, which the A1 probe read, counts the moves taken, which the player watched)."""
        for s in self.fights:
            for enemy, seen in leakprobes.move_spread(s, 50).items():
                self.assertEqual(len({shown for _, shown in seen}), 1, f"{enemy}: the intent on screen moved: {seen}")

    def test_clairvoyance_resampling_on_and_off_are_not_value_equivalent(self):
        diffs = []
        for s in self.fights:
            on = leakprobes.root_search(s, resample=True)
            off = leakprobes.root_search(s, resample=False)
            v_on = {(c.action, c.target): c.value_mean for c in on.children}
            v_off = {(c.action, c.target): c.value_mean for c in off.children}
            diffs.append(max(abs(v_on[k] - v_off[k]) for k in v_on.keys() & v_off.keys()))
        self.assertGreater(max(diffs), 0.5, f"values equal with and without resampling: {diffs}")

    def test_clairvoyance_choice_does_not_depend_on_the_hidden_draw_order(self):
        s = self.fights[1]
        dist = []
        for root_seed in (None, 11, 12, 13):
            picks = []
            for search_seed in (1, 2, 3, 4):
                r = leakprobes.root_search(s, resample=True, seed=search_seed, root_seed=root_seed)
                picks.append((r.best.action, r.best.target))
            dist.append(sorted(picks))
        modal = [max(set(d), key=d.count) for d in dist]
        share = [d.count(m) / len(d) for d, m in zip(dist, modal)]
        self.assertEqual(len(set(modal)), 1, f"modal choice moves with the hidden order: {dist}")
        self.assertGreaterEqual(min(share), 0.5, f"decision distribution moves with the hidden order: {dist}")


@unittest.skipUnless(HAVE_NATIVE, "needs the native emulator library")
class AiStateAssumedTests(unittest.TestCase):
    """DESIGN.md 1.1: a fork that keeps the true hidden enemy moves says so, and the search reports it."""

    def _opening(self):
        from sts2_gym.env import ENCOUNTER_IDS  # noqa: PLC0415

        import combat_mcts  # noqa: PLC0415

        return combat_mcts.Combat.opening(0, ENCOUNTER_IDS["cultists"], 10, 1)

    def _imported(self):
        """The same fight loaded from its document: no intent pass was recorded in it."""
        import combat_mcts  # noqa: PLC0415
        from sts2_gym import native  # noqa: PLC0415

        live = self._opening()
        try:
            return combat_mcts.Combat(native.load_combat_snapshot(native.get_state_json(live.handle), 1))
        finally:
            live.close()

    def test_a_successful_resample_leaves_the_flag_clear(self):
        root = self._opening()
        try:
            fork = root.fork(5)
            try:
                self.assertFalse(fork.ai_state_assumed)
            finally:
                fork.close()
        finally:
            root.close()

    def test_an_imported_fight_cannot_be_resampled_and_its_forks_stay_flagged(self):
        root = self._imported()
        try:
            self.assertFalse(root.fork(None).ai_state_assumed)
            fork = root.fork(5)
            again = fork.fork(6)
            plain = fork.fork(None)
            try:
                self.assertTrue(fork.ai_state_assumed)
                self.assertTrue(again.ai_state_assumed)
                self.assertTrue(plain.ai_state_assumed)
            finally:
                for w in (fork, again, plain):
                    w.close()
        finally:
            root.close()

    def test_the_search_result_records_it(self):
        import live_search  # noqa: PLC0415

        for make, expected in ((self._opening, False), (self._imported, True)):
            root = make()
            try:
                result = live_search.search_live(
                    root.handle, max_sims=20, min_child_sims=2, confirm_pairs=0, time_budget_s=600.0, seed=1
                )
            finally:
                root.close()
            self.assertEqual(result.ai_state_assumed, expected)


if __name__ == "__main__":
    unittest.main()
