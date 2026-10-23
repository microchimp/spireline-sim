"""Lockstep adapter (round A1): divergence logging and re-anchor decision on a fake referee stream, and the
leak tests of DESIGN.md 7.6 (clairvoyance, L2, L3, L8) on the emulator alone. Runs under pytest or
`python -m unittest`; the emulator tests need the native library and skip without it."""

import sys
import unittest
from types import SimpleNamespace
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
        self.reanchored, self.applied, self.anchored = [], [], None

    def sync(self, kind, labels, msg=None):
        if self.anchored is None:
            self.i += 1
        return self.sync_reason

    def view(self):
        if self.anchored is not None:  # after a successful re-anchor the fake follows the referee
            return self.anchored
        return self.views[min(self.i, len(self.views) - 1)]

    def choose(self, kind, labels, msg):
        return len(labels) - 1

    def apply(self, kind, labels, idx):
        self.applied.append(idx)

    def reanchor(self, plan, msg):
        self.reanchored.append(plan)
        ok = self.reanchor_ok.pop(0) if isinstance(self.reanchor_ok, list) else self.reanchor_ok
        if ok:
            self.anchored = lockstep.ref_view(msg["obs"])
        return ok


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
        # every later re-anchor point is tried again (round A1 amend), so both room entries count a failure
        self.assertEqual((ls.stats["reanchor_failed"], ls.stats["not_ranked"]), (2, 2))

    def test_unmapped_kind_and_structure_mismatch_end_the_lockstep_with_a_reason(self):
        a = obs()
        ls, _ = self.run_stream([msg("event", a)], FakeEmu([view_of(a)]))
        self.assertEqual(ls.stats["first_loss"]["reason"], "unmapped:event")  # shop is mapped since the A1 amend
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
class ResumeTests(unittest.TestCase):
    """Round A1 amend: a lost lockstep is re-anchored at the next fight start or room entry and goes on from there."""

    def test_lost_mid_fight_resumes_at_the_next_room_entry(self):
        c, a = obs(combat=True), obs()
        emu = FakeEmu([view_of(obs(hp=50, combat=True))], reanchor_ok=[False, True])
        ls = lockstep.Lockstep(emu, lambda m: 0)
        for m in (msg("combat", c), msg("combat", c), msg("reward", a), msg("map", a), msg("rest", a)):
            ls.decide(m)
        self.assertEqual(emu.reanchored, ["fight_start", "room_entry"])  # no attempt mid-fight or at the reward
        self.assertIsNone(ls.lost)
        s = ls.stats
        self.assertEqual((s["reanchors"], s["reanchor_failed"], s["resumed"], s["ranked"], s["not_ranked"]),
                         (1, 1, 1, 2, 3))

    def test_lost_at_a_reward_resumes_at_the_next_fight_start(self):
        a, c = obs(), obs(combat=True)
        emu = FakeEmu([view_of(obs(hp=10))], reanchor_ok=[True])
        ls = lockstep.Lockstep(emu, lambda m: 0)
        for m in (msg("reward", a), msg("card_reward", a), msg("combat", c)):
            ls.decide(m)
        self.assertEqual(emu.reanchored, ["fight_start"])
        self.assertEqual((ls.stats["resumed"], ls.stats["ranked"]), (1, 1))

    def test_an_unmapped_kind_suspends_and_the_next_room_entry_resumes(self):
        a = obs()
        emu = FakeEmu([view_of(a)], reanchor_ok=True)
        ls = lockstep.Lockstep(emu, lambda m: 0)
        for m in (msg("event", a), msg("event", a), msg("map", a)):
            ls.decide(m)
        self.assertTrue(ls.log.records[0]["gap"])  # a mapping gap, not an emulator divergence
        self.assertEqual((ls.stats["gaps"], ls.stats["diverged"], ls.stats["resumed"]), (1, 0, 1))
        self.assertIsNone(ls.lost)


def _opt(kind, action=0, **kw):
    base = {"card": None, "deck_index": None, "relic": None, "potion": None, "price": None, "node": None, "cards": ()}
    return SimpleNamespace(kind=kind, action=action, **{**base, **kw})


def _card(i, up=False):
    return SimpleNamespace(card_id=i, upgraded=up)


class FakeRun:
    """A RunSession whose screens follow a table: screens[name] = (type, options); moves[(name, kind)] = next name."""

    COMBAT, DONE, ITEMS, CARDS = 1, 2, 5, 6
    handle, env = 0, SimpleNamespace(_run_obs_buf=None, _run_rew_buf=None, _run_terminal_buf=None, _run_truncated_buf=None)

    def __init__(self, start, screens, moves):
        self.cur, self.screens, self.moves, self.taken = start, screens, moves, []

    @property
    def phase(self):
        return {"combat": self.COMBAT, "done": self.DONE, "cards": self.CARDS}.get(
            self.cur, self.ITEMS if self.screens.get(self.cur, ("",))[0] == "rewards" else 7)

    def decision(self):
        t, opts = self.screens[self.cur]
        return SimpleNamespace(type=t, options=opts, view=None)

    def apply(self, o):
        self.taken.append(o.kind)
        self.cur = self.moves[(self.cur, o.kind)]


class ItemsRun(FakeRun):
    """The engine's rewards screen: raw item steps (gold, potion, relic, cards); decision() opens cards first."""

    def __init__(self, **items):
        opts = (_opt("reward.proceed", 9),)
        screens = {"cards": ("card_reward", (_opt("card.take", 0, card=_card(8)), _opt("card.skip", 1))),
                   "items": ("rewards", opts), "map": ("map", (_opt("map.node", 0, node=(1, 2)), _opt("map.node", 1, node=(3, 2))))}
        super().__init__("items", screens, {("cards", "card.take"): "items", ("cards", "card.skip"): "items",
                                           ("items", "reward.proceed"): "map"})
        self.items = {"gold": 0, "potion": 0, "relic": 0, "card": 0, **items}

    def decision(self):
        if self.cur == "items" and self.items["card"]:
            self.items["card"], self.cur = 0, "cards"  # RunSession's auto step
        return super().decision()

    def _step(self, action):
        self.taken.append(f"step:{action}")
        present = [k for k in ("gold", "potion", "relic", "card") if self.items[k]]
        k = present[action]
        self.items[k] = 0
        if k == "card":
            self.cur = "cards"


def stub_emu(run, hand=None):
    """An EmuLock without the native library: its mapping logic over a FakeRun."""
    e = lockstep.EmuLock.__new__(lockstep.EmuLock)
    e.E = SimpleNamespace(rc=SimpleNamespace(PHASE_COMBAT=FakeRun.COMBAT, PHASE_COMPLETE=FakeRun.DONE,
                                             PHASE_RELIC_REWARD=FakeRun.ITEMS, PHASE_CARD_REWARD=FakeRun.CARDS))
    e.native = SimpleNamespace(run_action_mask=lambda h, n: [1, 1] + [0] * (n - 2),
                               run_state_list=lambda h, k, n: [getattr(run, "items", {}).get(x, 0) for x in ("gold", "potion", "relic", "card")])
    e.s, e.rq, e.dec, e._msg = run, [], None, None
    e.slug = {"cards": {5: "STRIKE_IRONCLAD", 6: "DEFEND_IRONCLAD", 7: "BASH", 8: "ANGER"}, "relics": {}, "potions": {}}
    e.ls = SimpleNamespace(_living_targets=lambda o: [0, 1])
    e._obs = lambda: None
    e._hand = lambda: hand or []
    return e


class MappingTests(unittest.TestCase):
    """The decisions the A1 smoke lost lockstep at, on a scripted emulator (no native library needed)."""

    def test_floor2_potion_take_is_mirrored_before_the_card_reward_opens(self):
        # 23 of the first smoke's 25 "state_diverged" losses (and 65 of the second's): the referee takes the potion
        # first; RunSession.decision() would open the card reward first, so the emulator lacked the potion at the
        # referee's next decision. The answer is now stepped on the items screen as it comes.
        run = ItemsRun(potion=1, card=1)
        e = stub_emu(run)
        tk = ["take", "skip"]
        self.assertIsNone(e.sync("reward", tk, {"ctx": {"type": "potion"}}))
        self.assertIsNone(e.apply("reward", tk, 0))
        self.assertEqual((run.items["potion"], run.cur), (0, "items"))  # taken now, card reward not yet open
        self.assertIsNone(e.sync("reward", tk, {"ctx": {"type": "card"}}))
        self.assertIsNone(e.apply("reward", tk, 0))
        self.assertEqual(run.cur, "cards")
        self.assertIsNone(e.sync("card_reward", [["ANGER", 0], "skip"], {}))
        self.assertIsNone(e.apply("card_reward", [["ANGER", 0], "skip"], 0))
        self.assertIsNone(e.sync("map", [[1, 2], [3, 2]], {}))
        self.assertEqual(run.taken, ["step:0", "step:0", "card.take", "reward.proceed"])

    def test_skipped_items_leave_the_screen_and_a_lone_path_enters_the_fight(self):
        # The 14 "phase:5" losses: the referee left the rewards screen and took a lone map node without asking.
        screens = {"items": ("rewards", (_opt("reward.relic", 1), _opt("reward.proceed", 9))),
                   "map": ("map", (_opt("map.node", 0, node=(1, 2)),))}
        run = FakeRun("items", screens, {("items", "reward.proceed"): "map", ("map", "map.node"): "combat"})
        e = stub_emu(run)
        self.assertIsNone(e.sync("reward", ["take", "skip"], {"ctx": {"type": "relic"}}))
        self.assertIsNone(e.apply("reward", ["take", "skip"], 1))
        self.assertIsNone(e.sync("combat", [["end"], ["play", 0, "BASH", 0]], {}))
        self.assertEqual(run.taken, ["reward.proceed", "map.node"])

    def test_a_play_is_matched_by_card_not_by_hand_position(self):
        e = stub_emu(FakeRun("combat", {}, {}), hand=[["DEFEND_IRONCLAD", 0], ["STRIKE_IRONCLAD", 0], ["BASH", 0]])
        ref = [["STRIKE_IRONCLAD", 0, 1, True], ["BASH", 0, 2, True], ["DEFEND_IRONCLAD", 0, 1, True]]
        m = {"obs": {"combat": {"hand": ref}}}
        self.assertEqual(e._action_of(["play", 0, "STRIKE_IRONCLAD", 1], m), (1, 1))
        self.assertEqual(e._action_of(["play", 2, "DEFEND_IRONCLAD", -1], m), (0, -1))
        self.assertEqual(e._action_of(["end"], m), (3, -1))
        self.assertIsNone(e._action_of(["play", 1, "ANGER", 0], m))

    def test_shop_bundle_and_treasure_are_mirrored(self):
        e = stub_emu(FakeRun("x", {}, {}))
        e.dec = SimpleNamespace(type="shop", options=(_opt("shop.card", 0, price=50), _opt("shop.card", 1, price=75),
                                                      _opt("shop.leave", 9)))
        self.assertEqual(e._pair("shop", [["card", 0, 75], ["card", 1, 50], ["leave"], ["remove", 0, 75]]),
                         [1, 0, 2, None])
        e.dec = SimpleNamespace(type="bundle", options=(_opt("bundle.pick", 0, cards=(_card(5), _card(6), _card(7))),
                                                        _opt("bundle.pick", 1, cards=(_card(8), _card(8), _card(5)))))
        labels = [["ANGER", "STRIKE_IRONCLAD", "ANGER"], ["BASH", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD"]]
        self.assertEqual(e._pair("bundle", labels), [1, 0])
        self.assertIsNone(e.apply("treasure", [["take", "R"], "skip"], 0))
        self.assertEqual(e.apply("treasure", [["take", "R"], "skip"], 1), "treasure_skip")
        for k in ("treasure", "shop", "bundle", "card_reward_combat"):
            self.assertIn(k, lockstep.MAPPED_KINDS)


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
