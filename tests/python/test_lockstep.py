"""Lockstep adapter (round A1): divergence logging and re-anchor decision on a fake referee stream, and the
leak tests of DESIGN.md 7.6 (clairvoyance, L2, L3, L8) on the emulator alone. Runs under pytest or
`python -m unittest`; the emulator tests need the native library and skip without it."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))

import leakprobes  # noqa: E402  -- missing on main: every test here fails there
import lockstep  # noqa: E402

try:
    leakprobes.native_env()
    from sts2_gym import native  # noqa: E402,F401

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    HAVE_NATIVE = False


class FakeEmu:
    """An emulator side whose view follows a script: views[i] is its view at decision i."""

    def __init__(self, views, reanchor_ok=False, sync_reason=None):
        self.views, self.i, self.reanchor_ok, self.sync_reason = views, -1, reanchor_ok, sync_reason
        self.reanchored, self.applied = [], []

    def sync(self, kind, labels):
        self.i += 1
        return self.sync_reason

    def view(self):
        return self.views[min(self.i, len(self.views) - 1)]

    def choose(self, kind, labels, msg):
        return len(labels) - 1

    def apply(self, kind, labels, idx):
        self.applied.append(idx)

    def reanchor(self, plan, msg):
        self.reanchored.append(plan)
        return self.reanchor_ok


def obs(hp=80, floor=1, combat=False):
    o = {"act": 0, "floor": floor, "hp": hp, "max_hp": 80, "gold": 99, "deck": [["STRIKE_IRONCLAD", 0]],
         "relics": ["BURNING_BLOOD"], "potions": []}
    if combat:
        o["combat"] = {"hand": [["STRIKE_IRONCLAD", 0, 1, True]], "enemies": [{"hp": 15}]}
    return o


def msg(kind, o, n=2):
    return {"t": "decide", "kind": kind, "obs": o, "actions": [["x"]] * n, "h": "abc"}


def view_of(o):
    return lockstep.ref_view(o)


class DivergenceTests(unittest.TestCase):
    def run_stream(self, stream, emu):
        ls = lockstep.Lockstep(emu, lambda m: 0)
        return ls, [ls.decide(m) for m in stream]

    def test_identical_views_are_ranked_and_the_agent_answers(self):
        o = obs()
        ls, picks = self.run_stream([msg("map", o)], FakeEmu([view_of(o)]))
        self.assertEqual(picks, [1])  # the emulator side's choice, not the fallback's 0
        self.assertEqual((ls.stats["ranked"], ls.stats["diverged"], ls.log.records), (1, 0, []))

    def test_hash_difference_logs_an_agent_weighted_record_and_loses_lockstep(self):
        a, b = obs(hp=80), obs(hp=70)
        emu = FakeEmu([view_of(a), view_of(b)])
        ls, _ = self.run_stream([msg("rest", a), msg("card_reward", obs(hp=80, floor=2))], emu)
        rec = ls.log.records[0]
        self.assertEqual((rec["decision"], rec["type"], rec["floor"], rec["room"]), (1, "card_reward", 2, "non_combat"))
        self.assertNotEqual(rec["ref_hash"], rec["emu_hash"])
        self.assertEqual(rec["referee_state_hash"], "abc")
        self.assertTrue(rec["agent_weighted"])
        self.assertTrue(any(d[0] == ".hp" for d in rec["diff"]))
        self.assertIsNone(rec["reanchor_plan"])  # a card reward is neither a fight start nor a room entry
        self.assertIsNotNone(ls.lost)

    def test_later_decisions_after_the_loss_are_scored_not_ranked(self):
        a = obs()
        ls, picks = self.run_stream([msg("map", a), msg("rest", a), msg("rest", a)], FakeEmu([view_of(obs(hp=1))]))
        self.assertEqual((ls.stats["ranked"], ls.stats["not_ranked"]), (0, 3))
        self.assertEqual(picks, [0, 0, 0])  # the referee-only fallback answers
        self.assertEqual(len(ls.log.records), 1)  # no further comparisons once off the emulator

    def test_reanchor_at_room_entry_and_fight_start_continues_the_lockstep(self):
        a = obs(hp=80)
        emu = FakeEmu([view_of(obs(hp=70))], reanchor_ok=True)
        ls, _ = self.run_stream([msg("map", a)], emu)
        self.assertEqual(emu.reanchored, ["room_entry"])
        self.assertIsNone(ls.lost)
        self.assertEqual((ls.stats["reanchors"], ls.stats["ranked"]), (1, 1))
        self.assertTrue(ls.log.records[0]["reanchored"])
        emu = FakeEmu([view_of(obs(combat=True))], reanchor_ok=True)
        self.run_stream([msg("combat", obs(hp=60, combat=True))], emu)
        self.assertEqual(emu.reanchored, ["fight_start"])

    def test_no_reanchor_mid_fight(self):
        c = obs(combat=True)
        emu = FakeEmu([view_of(c), view_of(obs(hp=50, combat=True))], reanchor_ok=True)
        ls, _ = self.run_stream([msg("combat", c), msg("combat", c)], emu)
        self.assertEqual(emu.reanchored, [])
        self.assertIsNotNone(ls.lost)

    def test_failed_reanchor_goes_on_alone(self):
        a = obs(hp=80)
        ls, _ = self.run_stream([msg("map", a), msg("map", a)], FakeEmu([view_of(obs(hp=70))]))
        self.assertEqual((ls.stats["reanchor_failed"], ls.stats["not_ranked"]), (1, 2))

    def test_unmapped_kind_and_structure_mismatch_end_the_lockstep_with_a_reason(self):
        a = obs()
        ls, _ = self.run_stream([msg("shop", a)], FakeEmu([view_of(a)]))
        self.assertEqual(ls.stats["first_loss"]["reason"], "unmapped:shop")
        ls, _ = self.run_stream([msg("map", a)], FakeEmu([view_of(a)], sync_reason="screen:rest!=map"))
        self.assertEqual(ls.log.records[0]["reason"], "screen:rest!=map")

    def test_hash_ignores_pile_order_and_nothing_hidden_enters_it(self):
        a, b = obs(), obs()
        a["deck"] = [["A", 0], ["B", 1]]
        b["deck"] = [["B", 1], ["A", 0]]
        self.assertEqual(lockstep.view_hash(view_of(a)), lockstep.view_hash(view_of(b)))
        self.assertNotIn("rng", repr(lockstep.project(view_of(obs(combat=True)))))

    def test_reanchor_plan(self):
        self.assertEqual(lockstep.reanchor_plan("combat", "map"), "fight_start")
        self.assertIsNone(lockstep.reanchor_plan("combat", "combat"))
        self.assertEqual(lockstep.reanchor_plan("map", "rest"), "room_entry")
        self.assertIsNone(lockstep.reanchor_plan("rest", "map"))


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

    @unittest.expectedFailure  # KNOWN LEAK (A1 finding): the clone keeps the act's materialised encounter sequence
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

    @unittest.expectedFailure  # KNOWN LEAK (A1 finding): the clone does not re-roll an enemy's chosen current move
    def test_L8_resampled_forks_vary_some_current_move(self):
        spreads = leakprobes.move_spread(self.fights[0], 200)
        varied = {k: v for k, v in spreads.items() if len({m[0] for m in v}) > 1}
        self.assertTrue(varied, f"no enemy's current move varies across 200 resampled forks: {spreads}")

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


if __name__ == "__main__":
    unittest.main()
