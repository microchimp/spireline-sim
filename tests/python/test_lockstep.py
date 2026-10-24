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

    def test_a_search_that_assumed_enemy_moves_marks_its_decision_record(self):
        class AssumingEmu(FakeEmu):
            ai_state_assumed = True

        a, b = obs(hp=80), obs(hp=70)
        ls, _ = self.run_stream([msg("rest", a), msg("card_reward", obs(hp=80, floor=2))],
                                AssumingEmu([view_of(a), view_of(b)]))
        ls.finish({})
        rec = ls.log.records[0]
        self.assertEqual(rec["decision"], 1)
        self.assertIs(rec["ai_state_assumed"], False)  # decision 1 was never searched: it diverged first
        self.assertEqual(ls.assumed, {0})

    def test_a_record_of_a_searched_decision_says_whether_the_ai_state_was_assumed(self):
        class AssumingEmu(FakeEmu):
            ai_state_assumed = True

            def choose(self, kind, labels, msg):
                raise RuntimeError("x")  # the search fails after it assumed: the loss record is of that decision

        a = obs()
        ls, _ = self.run_stream([msg("rest", a)], AssumingEmu([view_of(a)]))
        ls.finish({})
        self.assertEqual(ls.log.records[0]["ai_state_assumed"], True)

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
        ls, _ = self.run_stream([msg("minigame", a)], FakeEmu([view_of(a)]))
        self.assertEqual(ls.stats["first_loss"]["reason"], "unmapped:minigame")  # event, shop are mapped
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


class ProvEmu(FakeEmu):
    """A fake emulator side that reports how its last re-anchor was done, as EmuLock does."""

    def __init__(self, views, how=None, streams=None, **k):
        super().__init__(views, **k)
        self.how, self.streams_now, self.anchored_with, self.streams = how, streams, None, None

    def reanchor(self, plan, msg):
        self.anchored_with, self.streams = self.how, self.streams_now
        return super().reanchor(plan, msg)


class ProvenanceTests(unittest.TestCase):
    """Review 1 and 2: what a re-anchor was seeded from reaches the run's stats."""

    def drive(self, emu, stream):
        ls = lockstep.Lockstep(emu, lambda m: 0)
        for m in stream:
            ls.decide(m)
        return ls

    def test_the_streams_named_and_moved_go_into_the_run_stats(self):
        rep = {"named": 17, "moved": 16, "unmoved": 1, "unmoved_names": ["Nope"]}
        a = obs(hp=80)
        emu = ProvEmu([view_of(obs(hp=70))], how="streams", streams=rep, reanchor_ok=[False, True])  # counted when it fails too
        s = self.drive(emu, [msg("map", a), msg("map", a)]).stats
        self.assertEqual((s["streams_named"], s["streams_moved"]), (34, 32))
        self.assertEqual(s["streams_unmoved_names"], {"Nope": 2})
        zero = dict(rep, moved=0, unmoved=17, unmoved_names=["Nope"])
        s = self.drive(ProvEmu([view_of(obs(hp=70))], how=None, streams=zero, reanchor_ok=True), [msg("map", a)]).stats
        self.assertEqual((s["streams_named"], s["streams_moved"], s["streams_unmoved_names"]), (17, 0, {"Nope": 1}))

    def test_an_emulator_side_with_no_stream_report_adds_no_stream_counts(self):
        a = obs(hp=80)
        s = self.drive(FakeEmu([view_of(obs(hp=70))], reanchor_ok=True), [msg("map", a)]).stats
        self.assertNotIn("streams_named", s)


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
        for m in (msg("minigame", a), msg("minigame", a), msg("map", a)):
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
    e.fail, e.event_names = None, {7: "WELLSPRING", 8: "SLIPPERY_BRIDGE"}
    e.slug = {"cards": {5: "STRIKE_IRONCLAD", 6: "DEFEND_IRONCLAD", 7: "BASH", 8: "ANGER"}, "relics": {}, "potions": {}}
    e.ls = SimpleNamespace(_living_targets=lambda o: [0, 1], _SELECTION_KIND=0)
    e._obs = lambda: [0]
    e._hand = lambda: hand or []
    return e


class MappingTests(unittest.TestCase):
    """The decisions the A1 smoke lost lockstep at, on a scripted emulator (no native library needed)."""

    def test_floor2_potion_take_is_mirrored_before_the_card_reward_opens(self):
        # 23 of the first smoke's 25 "state_diverged" losses (and 65 of the second's): the referee takes the potion
        # first; RunSession.decision() would open the card reward first, so the emulator lacked the potion at the
        # referee's next decision. The answer is now stepped on the items screen as it comes.
        run = ItemsRun(gold=1, potion=1, card=1)
        e = stub_emu(run)
        tk = ["take", "skip"]
        self.assertIsNone(e.sync("reward", tk, {"ctx": {"type": "potion"}}))
        self.assertEqual(run.items["gold"], 0)  # the referee claimed the gold before asking (133 smoke-2 losses)
        self.assertIsNone(e.apply("reward", tk, 0))
        self.assertEqual((run.items["potion"], run.cur), (0, "items"))  # taken now, card reward not yet open
        self.assertIsNone(e.sync("reward", tk, {"ctx": {"type": "card"}}))
        self.assertIsNone(e.apply("reward", tk, 0))
        self.assertEqual(run.cur, "cards")
        self.assertIsNone(e.sync("card_reward", [["ANGER", 0], "skip"], {}))
        self.assertIsNone(e.apply("card_reward", [["ANGER", 0], "skip"], 0))
        self.assertIsNone(e.sync("map", [[1, 2], [3, 2]], {}))
        self.assertEqual(run.taken, ["step:0", "step:0", "step:0", "card.take", "reward.proceed"])

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


class LoneEndTurnTests(unittest.TestCase):
    """B1 reward-phase group (G1162c5de18): the referee asks only with two or more legal actions, so a lone End Turn is
    taken for it. A fight that turn ends (a power's end-of-turn damage, an enemy dying to its retaliation) reaches the
    rewards screen with no combat question in between; the emulator side has to take the same lone End Turn before the
    non-combat question is compared, or it still stands in the fight ("phase:0!=rewards")."""

    COMBAT_LEGAL = 3  # the End Turn slot of a three-card hand

    def _fight(self, legal):
        screens = {"items": ("rewards", (_opt("reward.proceed", 9),)), "map": ("map", (_opt("map.node", 0, node=(1, 2)),))}
        run = ItemsRun(gold=0)
        run.cur, run.screens, run.stepped = "combat", screens, []
        run.moves[("map", "map.node")] = "map"  # a lone path the lockstep may walk on its way to the screen asked about
        e = stub_emu(run)

        def step(handle, action, target, *bufs):
            run.stepped.append(action)
            run.cur = "items"  # the turn ends and the last enemy dies
            return 0

        e.native.run_action_mask = lambda h, n: [int(i in legal()) for i in range(n)] if run.cur == "combat" else [0] * n
        e.native.run_step = step
        return e, run

    def test_a_lone_end_turn_that_wins_the_fight_is_taken_before_the_rewards_question(self):
        e, run = self._fight(lambda: [self.COMBAT_LEGAL])
        self.assertIsNone(e.sync("reward", ["take", "skip"], {"ctx": {"type": "card"}}))
        self.assertEqual(run.stepped, [self.COMBAT_LEGAL])

    def test_every_non_combat_screen_finishes_a_fight_the_referee_has_left(self):
        for kind in ("reward", "card_reward", "treasure", "map", "rest"):
            e, run = self._fight(lambda: [self.COMBAT_LEGAL])
            e.sync(kind, [], {})
            self.assertEqual(run.stepped, [self.COMBAT_LEGAL], kind)

    def test_a_fight_with_a_choice_left_is_still_reported_as_a_difference(self):
        e, run = self._fight(lambda: [0, self.COMBAT_LEGAL])  # a card and End Turn: the referee would have asked
        self.assertEqual(e.sync("reward", ["take", "skip"], {"ctx": {"type": "card"}}), f"phase:{FakeRun.COMBAT}!=rewards")
        self.assertEqual(run.stepped, [])


def _ev(action, step=0, event=7):
    return _opt("event.option", action, event=(event, action, step))


class EventTests(unittest.TestCase):
    """Round A1 events: the emulator follows the referee through an event (34 of 50 first losses before)."""

    def _run(self):
        screens = {"p1": ("event", (_ev(0), _ev(1), _ev(2))),
                   "lone": ("event", (_ev(0, 1),)),
                   "p2": ("event", (_ev(0, 2), _ev(1, 2))),
                   "map": ("map", (_opt("map.node", 0, node=(1, 2)), _opt("map.node", 1, node=(3, 2))))}
        return FakeRun("p1", screens, {("p1", "event.option"): "lone", ("lone", "event.option"): "p2",
                                        ("p2", "event.option"): "map"})

    def test_event_is_a_mapped_kind_and_pairs_the_referee_option_by_position(self):
        self.assertIn("event", lockstep.MAPPED_KINDS)
        run = self._run()
        e = stub_emu(run)
        labels = ["WELLSPRING.pages.INITIAL.options.A", "WELLSPRING.pages.INITIAL.options.B", "RELIC_X"]
        self.assertIsNone(e.sync("event", labels, {"ctx": "WELLSPRING"}))
        self.assertEqual(e._pair("event", labels), [0, 1, 2])
        self.assertIsNone(e.apply("event", labels, 1))
        self.assertEqual(run.taken, ["event.option"])

    def test_a_page_with_one_option_is_passed_as_the_referee_passes_it(self):
        run = self._run()
        e = stub_emu(run)
        labels = ["a", "b", "c"]
        e.sync("event", labels, {"ctx": "WELLSPRING"})
        e.apply("event", labels, 0)
        self.assertIsNone(e.sync("event", ["x", "y"], {"ctx": "WELLSPRING"}))  # the lone page was stepped
        self.assertEqual(run.taken, ["event.option", "event.option"])
        self.assertEqual(run.cur, "lone" if False else "p2")

    def test_a_different_event_or_option_count_is_a_gap_not_a_guess(self):
        e = stub_emu(self._run())
        self.assertEqual(e.sync("event", ["a", "b", "c"], {"ctx": "SLIPPERY_BRIDGE"}), "event:SLIPPERY_BRIDGE!=WELLSPRING")
        e.dec = SimpleNamespace(type="event", options=(_ev(0), _ev(1), _ev(2)), view=None)
        self.assertEqual(e._pair("event", ["a", "b"]), [None, None])
        self.assertTrue(e.apply("event", ["a", "b"], 0).startswith("no_emulator_option:event"))

    def test_event_then_fight_then_rewards_resume_through_the_lockstep(self):
        a = obs()
        emu = FakeEmu([view_of(a)])
        ls = lockstep.Lockstep(emu, lambda m: 0)
        for m in (msg("event", a), msg("map", a)):
            ls.decide(m)
        self.assertEqual((ls.stats["gaps"], ls.stats["ranked"], ls.lost), (0, 2, None))


class MultiPickScreenTests(unittest.TestCase):
    """A non-combat card screen the driver asks pick by pick (ctx pick/of): the game applies the picks when the screen
    closes, the emulator after each one, so from pick 2 on the emulator deck is ahead by the earlier picks (G0a127979b6)."""

    OFFER = [["STRIKE_IRONCLAD", 0], ["DEFEND_IRONCLAD", 0]]

    @staticmethod
    def with_deck(deck, hp=80):
        o = obs(hp=hp)
        o["deck"] = deck
        return o

    def pick_msg(self, o, pick=2, of=2, kind="select_cards"):
        m = {**msg(kind, o), "actions": self.OFFER}
        if pick is not None:
            m["ctx"] = {"pick": pick, "of": of}
        return m

    def decide(self, ref_deck, emu_deck, pick=2, of=2, kind="select_cards", ref_hp=80, emu_hp=80):
        a = self.with_deck(ref_deck, hp=ref_hp)
        held = {**view_of(self.with_deck(emu_deck, hp=emu_hp))}
        ls = lockstep.Lockstep(FakeEmu([held]), lambda m: 0)
        ls.decide(self.pick_msg(a, pick, of, kind))
        return ls

    def test_the_picks_the_emulator_already_applied_are_not_a_deck_divergence(self):
        S, B = ["STRIKE_IRONCLAD", 0], ["BASH", 0]
        cases = {
            "add": ([S], [S, ["SHRUG_IT_OFF", 0]]),
            "remove": ([S, S], [S]),
            "upgrade": ([B], [["BASH", 1]]),
            "transform": ([S], [["STOKE", 0]]),
        }
        for name, (ref_deck, emu_deck) in cases.items():
            with self.subTest(name):
                ls = self.decide(ref_deck, emu_deck)
                self.assertEqual(ls.log.records, [])
                self.assertEqual((ls.stats["diverged"], ls.stats["ranked"]), (0, 1))
                self.assertIsNone(ls.lost)

    def test_pick_3_of_3_allows_two_earlier_picks(self):
        S = ["STRIKE_IRONCLAD", 0]
        ls = self.decide([S, S, S], [["STOKE", 0], ["OFFERING", 0], S], pick=3, of=3)  # two transforms
        self.assertEqual((ls.stats["diverged"], ls.stats["ranked"]), (0, 1))
        ls = self.decide([S, S, S], [["STOKE", 0], ["OFFERING", 0], ["NOT_YET", 0]], pick=3, of=3)  # three: more than two picks
        self.assertEqual(ls.stats["diverged"], 1)

    def test_the_first_pick_and_a_message_without_ctx_still_compare_the_deck(self):
        S = ["STRIKE_IRONCLAD", 0]
        for name, pick in (("pick 1", 1), ("no ctx", None)):
            with self.subTest(name):
                ls = self.decide([S], [S, ["SHRUG_IT_OFF", 0]], pick=pick)
                self.assertEqual((ls.stats["diverged"], len(ls.log.records)), (1, 1))
                rec = ls.log.records[0]
                self.assertEqual((rec["type"], [d[0] for d in rec["diff"]]), ("select_cards", [".deck"]))
                self.assertIsNotNone(ls.lost)

    def test_only_the_deck_is_skipped_at_pick_2(self):
        S = ["STRIKE_IRONCLAD", 0]
        ls = self.decide([S], [S, ["SHRUG_IT_OFF", 0]], ref_hp=63, emu_hp=70)
        self.assertEqual(ls.stats["diverged"], 1)
        self.assertEqual([d[0] for d in ls.log.records[0]["diff"]], [".hp"])

    def test_a_gross_deck_error_at_pick_2_still_shows(self):
        S = ["STRIKE_IRONCLAD", 0]
        ls = self.decide([S], [S, ["SHRUG_IT_OFF", 0], ["BASH", 0], ["ANGER", 0]])  # three cards more after one earlier pick
        self.assertEqual([d[0] for d in ls.log.records[0]["diff"]], [".deck"])

    def test_other_kinds_with_a_pick_context_are_not_skipped(self):
        S = ["STRIKE_IRONCLAD", 0]
        ls = self.decide([S], [S, ["SHRUG_IT_OFF", 0]], kind="card_reward")
        self.assertEqual(ls.stats["diverged"], 1)

    def test_the_combat_helper_is_unchanged_and_the_new_one_names_the_non_combat_case(self):
        a = self.with_deck([["STRIKE_IRONCLAD", 0]])
        pick2 = self.pick_msg(a)
        self.assertFalse(lockstep.mid_screen_pick("select_cards", pick2))
        self.assertTrue(lockstep.deck_pending_pick("select_cards", pick2))
        self.assertFalse(lockstep.deck_pending_pick("select_cards", self.pick_msg(a, pick=1)))
        self.assertFalse(lockstep.deck_pending_pick("select_cards", self.pick_msg(a, pick=None)))
        self.assertFalse(lockstep.deck_pending_pick("select_cards_combat", pick2))


class TreasureTests(unittest.TestCase):
    def test_the_chest_relic_the_emulator_already_holds_is_not_a_divergence(self):
        # 6 of the 12 divergences of the first events-round run: the emulator holds the chest's relic at the question.
        a = obs()
        held = {**view_of(a), "relics": ["BURNING_BLOOD", "LANTERN"]}
        m = {**msg("treasure", a), "actions": [["take", "LANTERN"], "skip"]}
        ls = lockstep.Lockstep(FakeEmu([held]), lambda m: 0)
        ls.decide(m)
        self.assertEqual((ls.stats["diverged"], ls.stats["ranked"]), (0, 1))
        other = {**msg("treasure", a), "actions": [["take", "ORICHALCUM"], "skip"]}  # a different relic is still a difference
        ls = lockstep.Lockstep(FakeEmu([held]), lambda m: 0)
        ls.decide(other)
        self.assertEqual(ls.stats["diverged"], 1)

    @staticmethod
    def question(a, relic):
        return {**msg("treasure", a), "actions": [["take", relic], "skip"]}

    def test_a_pickup_effect_the_emulator_already_applied_is_not_a_divergence(self):
        # G96df219a44 and its two siblings: the emulator took the chest relic, so what the pickup changes is already in its
        # view while the referee still shows the chest. The next decision compares those fields.
        a = obs(hp=63)
        b = obs()
        b["deck"] = [["STRIKE_IRONCLAD", 1]]
        cases = {
            "max hp": (a, {**view_of(a), "relics": ["BURNING_BLOOD", "STRAWBERRY"], "hp": 70, "max_hp": 87}, "STRAWBERRY"),
            "gold": (obs(), {**view_of(obs()), "relics": ["BURNING_BLOOD", "OLD_COIN"], "gold": 399}, "OLD_COIN"),
            "deck": (obs(), {**view_of(obs()), "relics": ["BURNING_BLOOD", "WHETSTONE"], "deck": view_of(b)["deck"]}, "WHETSTONE"),
            "potions": (obs(), {**view_of(obs()), "relics": ["BURNING_BLOOD", "POTION_BELT"], "potions": ["BLOCK_POTION"]}, "POTION_BELT"),
        }
        for name, (ref, held, relic) in cases.items():
            with self.subTest(name):
                ls = lockstep.Lockstep(FakeEmu([held]), lambda m: 0)
                ls.decide(self.question(ref, relic))
                self.assertEqual((ls.stats["diverged"], ls.stats["ranked"]), (0, 1))
                self.assertIsNone(ls.lost)

    def test_the_deferral_hides_nothing_but_the_pickup_fields(self):
        a = obs(hp=63)
        held = {**view_of(a), "relics": ["BURNING_BLOOD", "STRAWBERRY"], "hp": 70, "max_hp": 87}
        for name, change in (("floor", {"floor": 9}), ("act", {"act": 1})):  # a floor or act gap at the question still shows
            with self.subTest(name):
                ls = lockstep.Lockstep(FakeEmu([{**held, **change}]), lambda m: 0)
                ls.decide(self.question(obs(hp=63, floor=10), "STRAWBERRY"))
                self.assertEqual(ls.stats["diverged"], 1)
                self.assertIn("." + name, [d[0] for d in ls.log.records[0]["diff"]])
        with self.subTest("relic not held"):  # the emulator does not hold the offered relic: its hp is no pickup effect
            ls = lockstep.Lockstep(FakeEmu([{**held, "relics": ["BURNING_BLOOD"]}]), lambda m: 0)
            ls.decide(self.question(a, "STRAWBERRY"))
            self.assertEqual(ls.stats["diverged"], 1)

    def test_a_wrong_pickup_effect_shows_at_the_next_decision(self):
        a = obs(hp=63)
        held = {**view_of(a), "relics": ["BURNING_BLOOD", "STRAWBERRY"], "hp": 70, "max_hp": 87}
        ls = lockstep.Lockstep(FakeEmu([held, held]), lambda m: 0)
        ls.decide(self.question(a, "STRAWBERRY"))
        self.assertEqual(ls.stats["diverged"], 0)
        ls.decide(msg("map", a))  # the referee took the relic without the +7: the emulator's hp is wrong
        self.assertEqual(ls.stats["diverged"], 1)
        rec = ls.log.records[0]
        self.assertEqual((rec["decision"], rec["type"]), (1, "map"))
        self.assertTrue({".hp", ".max_hp"} <= {d[0] for d in rec["diff"]})


class FloorRun(FakeRun):
    """Every map node leads to a fight, whose resolution and rewards lead back to the next map; floor counts the rooms."""

    def __init__(self):
        opts = (_opt("map.node", 0, node=(1, 2)), _opt("map.node", 1, node=(3, 2)))
        super().__init__("map", {"map": ("map", opts)}, {})
        self.floor = 3

    def view(self):
        return SimpleNamespace(floor=self.floor)

    def apply(self, o):
        self.taken.append(o.kind)
        self.floor += 1  # a room entered and left


class ReanchorWalkTests(unittest.TestCase):
    def test_room_entry_walks_a_suspended_emulator_up_to_the_referees_floor(self):
        run = FloorRun()
        e = stub_emu(run)
        e.rules, e.rng = SimpleNamespace(pick=lambda v, d, r: SimpleNamespace(index=0)), None
        e.native = SimpleNamespace(run_debug_resolve_combat=lambda *a: True)
        e._set_player = lambda obs: True
        self.assertTrue(e.reanchor("room_entry", {"obs": {"floor": 6}}))
        self.assertEqual(run.floor, 6)  # was 3: three maps walked
        self.assertTrue(e.reanchor("fight_start", {"obs": {"floor": 9}}) is False and e.fail == "map_options:2")
        self.assertEqual(run.floor, 8)  # to the map before the fight's room (floor 9 - 1)


class ReanchorReasonTests(unittest.TestCase):
    def test_a_failed_reanchor_records_why(self):
        a = obs()
        emu = FakeEmu([view_of(obs(hp=10))], reanchor_ok=False)
        emu.fail = "in_fight"
        ls = lockstep.Lockstep(emu, lambda m: 0)
        ls.decide(msg("map", a))
        ls.decide(msg("map", a))
        self.assertEqual(ls.stats["reanchor_fail_reasons"], {"room_entry:in_fight": 2})


class CombatRun(FakeRun):
    """A fight the emulator stands in, on a given floor."""

    def __init__(self, floor):
        super().__init__("combat", {}, {})
        self.floor = floor

    def view(self):
        return SimpleNamespace(floor=self.floor)


class FightStartAnchorTests(unittest.TestCase):
    """A1b: a fight the emulator already stands in is put on the referee's record of it by the native export (it was
    reported as fight_start:in_fight, 33 of the 35 re-anchor failures of the events smoke)."""

    def emu(self, floor, answer):
        e = stub_emu(CombatRun(floor))
        calls = []
        e.native.run_reanchor_fight = lambda handle, record, buf: (calls.append(record), answer)[1]
        e.native.run_debug_resolve_combat = lambda *a: True
        return e, calls

    def test_the_fight_at_the_referees_floor_is_put_on_the_record(self):
        e, calls = self.emu(4, (3, "moves=same;patched=hand,player.energy"))
        record = {"hand": [["BASH", False]], "enemies": []}
        self.assertTrue(e.reanchor("fight_start", {"obs": {"floor": 4, "combat": record}}))
        self.assertEqual(calls, [record])
        self.assertIsNone(e.fail)

    def test_a_refused_record_names_the_reason(self):
        e, calls = self.emu(4, (-4, "the game has JAW_WORM where the emulator has none to pair with it"))
        self.assertFalse(e.reanchor("fight_start", {"obs": {"floor": 4, "combat": {"hand": []}}}))
        self.assertEqual(e.fail, "anchor:the game has JAW_WORM where the emulator has none to pair with it"[:60])  # kept short for the summary
        self.assertFalse(e.reanchor("fight_start", {"obs": {"floor": 4}}))
        self.assertEqual(e.fail, "no_record")
        self.assertEqual(len(calls), 1)

    def test_a_fight_at_an_earlier_floor_is_left_not_anchored(self):
        e, calls = self.emu(3, (0, "moves=same;patched="))
        e.rules, e.rng = SimpleNamespace(pick=lambda v, d, r: SimpleNamespace(index=0)), None
        self.assertFalse(e.reanchor("fight_start", {"obs": {"floor": 5, "combat": {"hand": []}}}))
        self.assertEqual(calls, [])  # it is another fight: walked past (here the fake never leaves it), not put on this record


class WalkRun(FakeRun):
    """A suspended emulator that never left its floor-3 fight while the referee went on: every map has one node, and
    that node leads into the next floor's fight."""

    def __init__(self, floor):
        super().__init__("combat", {"map": ("map", (_opt("map.node", 0, node=(1, 2)),))}, {})
        self.floor = floor

    def view(self):
        return SimpleNamespace(floor=self.floor)

    def apply(self, o):
        self.taken.append(o.kind)
        self.floor += 1
        self.cur = "combat"


class FightStartWalkThenAnchorTests(unittest.TestCase):
    """A1b review: an emulator in a fight at an earlier floor leaves it, walks the maps up to the referee's floor, enters
    the referee's fight, and only that fight is put on the referee's record."""

    def emu(self, floor):
        run = WalkRun(floor)
        e = stub_emu(run)
        e.rules, e.rng = SimpleNamespace(pick=lambda v, d, r: SimpleNamespace(index=0)), None
        e._set_player = lambda obs: True
        calls, anchored_at, resolved_at = [], [], []

        def resolve(handle, buf):
            resolved_at.append(run.floor)
            run.cur = "map"
            return True

        def anchor(handle, record, buf):
            calls.append(record)
            anchored_at.append(run.floor)
            return (3, "moves=same;patched=hand")

        e.native.run_debug_resolve_combat, e.native.run_reanchor_fight = resolve, anchor
        return e, run, calls, anchored_at, resolved_at

    def test_the_earlier_fights_are_resolved_and_only_the_referees_fight_is_anchored(self):
        e, run, calls, anchored_at, resolved_at = self.emu(3)
        record = {"hand": [["BASH", False]], "enemies": []}
        self.assertTrue(e.reanchor("fight_start", {"obs": {"floor": 5, "combat": record}}))
        self.assertIsNone(e.fail)
        self.assertEqual(calls, [record])  # one anchor, the referee's record, not one per fight passed
        self.assertEqual(anchored_at, [5])  # in the fight of floor 5
        self.assertEqual(resolved_at, [3, 4])  # the fights of floors 3 and 4 were walked past
        self.assertEqual((run.floor, run.cur), (5, "combat"))
        self.assertEqual(e.anchored_with, "observation")

    def test_the_lockstep_goes_on_from_the_referees_fight_without_a_second_walk(self):
        e, run, calls, anchored_at, resolved_at = self.emu(3)
        record = {"hand": [["BASH", False]], "enemies": []}
        msg = {"obs": {"floor": 5, "combat": record}}
        self.assertTrue(e.reanchor("fight_start", msg))
        self.assertTrue(e.reanchor("fight_start", msg))  # the emulator now stands in the fight at the referee's floor
        self.assertEqual(calls, [record, record])
        self.assertEqual(resolved_at, [3, 4])  # nothing further was resolved or walked
        self.assertEqual((run.floor, run.cur), (5, "combat"))


@unittest.skipUnless(HAVE_NATIVE, "needs the native emulator library")
class FightStartAnchorNativeTests(unittest.TestCase):
    """A real EmuLock in its own fight, put on a record whose hand and top card differ from it."""

    def test_an_emulock_re_anchors_the_fight_it_is_already_in(self):
        import ctypes  # noqa: PLC0415
        import json  # noqa: PLC0415
        import re  # noqa: PLC0415

        e = lockstep.EmuLock("QS2GYXRKWN", sims=2)
        self.addCleanup(e.close)
        for _ in range(60):
            if e.s.phase == e.E.rc.PHASE_COMBAT:
                break
            d = e.s.decision()
            e.s.apply(d.options[e.rules.pick(e.s.view(), d, e.rng).index])
        self.assertEqual(e.s.phase, e.E.rc.PHASE_COMBAT)
        native = e.native
        id_map = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))
        enemy = {v: re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", k).upper() for k, v in id_map["enemies"].items()}
        h = native.run_fork_combat(e.s.handle, False, 0, (ctypes.c_int * native.RUN_OBS_SIZE)())
        try:
            piles = {p: native.get_pile(h, p) for p in ("hand", "draw", "discard", "exhaust")}
            roster = native.get_enemies(h)
        finally:
            native.destroy(h)
        hand, draw = list(piles["hand"]), list(piles["draw"])
        hand[0], draw[0] = draw[0], hand[0]

        def pile(cards):
            return [[e.slug["cards"][c], bool(up)] for c, up in cards]

        view = e.s.view()
        record = {"in_combat": True, "round": 1,
                  "player": {"hp": view.hp, "max_hp": view.max_hp, "block": 0, "energy": 3, "powers": []},
                  "hand": pile(hand), "draw": sorted(pile(draw)), "discard": pile(piles["discard"]), "exhaust": [],
                  "enemies": [{"name": enemy[d], "hp": hp, "max_hp": hp, "block": 0, "powers": []} for d, _s, hp in roster]}
        self.assertNotEqual(e._hand(), [[c[0], int(c[1])] for c in pile(hand)])
        self.assertTrue(e.reanchor("fight_start", {"obs": {"floor": view.floor, "combat": record}}), e.fail)
        self.assertEqual(e._hand(), [[c[0], int(c[1])] for c in pile(hand)])
        self.assertIsNone(e.fail)


class AgentSeedTests(unittest.TestCase):
    def test_resample_seed_stream_follows_the_agent_seed_only(self):
        def stream(entropy):
            r = lockstep.agent_rng(entropy)
            return [int(r.integers(1 << 30)) for _ in range(6)]
        self.assertEqual(stream([1, 0]), stream([1, 0]))
        self.assertNotEqual(stream([1, 0]), stream([2, 0]))
        self.assertNotEqual(stream([1, 0]), stream([1, 1]))

    def test_the_referee_seed_never_reaches_the_rng_constructor(self):
        import inspect
        from unittest import mock
        sentinel = "SENTINEL-REFEREE-SEED"
        seen = []
        import numpy
        real = numpy.random.default_rng
        with mock.patch.object(numpy.random, "default_rng", lambda *a, **k: (seen.append((a, k)), real(*a, **k))[1]):
            lockstep.agent_rng([7, 3])
        self.assertNotIn(sentinel, repr(seen))
        # EmuLock builds its RNG from the agent seed alone: neither the sha256 of the game seed nor the seed itself
        src = inspect.getsource(lockstep.EmuLock.__init__)
        rng_line = next(ln for ln in src.splitlines() if "self.rng" in ln)
        self.assertIn("agent_rng(", rng_line)
        self.assertNotIn("seed.encode", rng_line)
        self.assertNotIn("self.seed", rng_line)
        self.assertEqual(list(inspect.signature(lockstep.agent_rng).parameters), ["entropy"])


def enemy_stub(slots, ref_hps=None):
    """An EmuLock in a fight whose observation buffer holds `slots` (hp, max hp) per enemy slot, six slots wide."""
    e = stub_emu(SimpleNamespace(
        phase=FakeRun.COMBAT, act_index=0,
        view=lambda: SimpleNamespace(floor=3, hp=70, max_hp=80, gold=9, deck=[], relics=[], potions=[])))
    buf = [0] * (4 + 6 * 8)
    for k, (hp, mx) in enumerate(slots):
        buf[4 + k * 8], buf[4 + k * 8 + 1] = hp, mx
    e.ls = SimpleNamespace(_ENEMIES=4, _ENEMY=8, _living_targets=lambda o: [])
    e.native = SimpleNamespace(MAX_ENEMIES=6)
    e._obs = lambda: buf
    return e


def ref_enemies(*hps):
    o = obs(combat=True)
    o["hp"], o["floor"], o["gold"], o["max_hp"], o["deck"], o["relics"] = 70, 3, 9, 80, [], []
    o["combat"]["hand"], o["combat"]["enemies"] = [], [{"hp": h} for h in hps]
    return o


class EnemyProjectionTests(unittest.TestCase):
    """Both sides list the enemies they still hold, dead ones with their 0 HP, by the same rule. The emulator used to
    list living enemies only while the referee listed all, so a corpse the game keeps was a diff whichever side held it."""

    def enemy_diff(self, slots, *hps):
        d = lockstep.structural_diff(lockstep.ref_view(ref_enemies(*hps)), enemy_stub(slots).view())
        return [x for x in d if x[0] == ".combat.enemy_hp"]

    def test_a_corpse_both_engines_hold_compares_equal(self):
        self.assertEqual(self.enemy_diff([(0, 40), (12, 40)], 0, 12), [])

    def test_a_corpse_only_the_referee_holds_is_a_diff(self):
        d = self.enemy_diff([(12, 40)], 0, 12)
        self.assertEqual([(x[1], x[2]) for x in d], [([0, 12], [12])])

    def test_a_corpse_only_the_emulator_holds_is_a_diff(self):
        d = self.enemy_diff([(0, 40), (12, 40)], 12)
        self.assertEqual([(x[1], x[2]) for x in d], [([12], [0, 12])])

    def test_empty_slots_are_not_enemies(self):
        self.assertEqual(enemy_stub([(12, 40), (0, 0)]).view()["combat"]["enemy_hp"], [12])


class RecordContextTests(unittest.TestCase):
    """A divergence record says what the step before it was, and where in the run it happened."""

    def combat_obs(self, rnd, hps):
        o = obs(combat=True)
        o["combat"]["round"] = rnd
        o["combat"]["enemies"] = [{"name": n, "hp": h} for n, h in zip(("cultist", "jaw_worm"), hps)]
        return o

    def run_fight(self, last_hp):
        m = {"t": "decide", "kind": "map", "obs": obs(), "actions": [[3, 1, "Monster"], [4, 1, "Elite"]], "h": "abc"}
        a, b = self.combat_obs(2, (15, 40)), self.combat_obs(3, (15, last_hp))
        play = {"t": "decide", "kind": "combat", "obs": a, "actions": [["play", 0, "STRIKE", 0], ["end"]], "h": "abc"}
        nxt = {"t": "decide", "kind": "combat", "obs": b, "actions": [["play", 0, "STRIKE", 0], ["end"]], "h": "abc"}
        emu = FakeEmu([view_of(m["obs"]), view_of(a), view_of(self.combat_obs(3, (15, 30)))])
        ls = lockstep.Lockstep(emu, lambda msg_: 0)
        for x in (m, play, nxt):
            ls.decide(x)
        return ls

    def test_a_record_carries_the_acted_step_the_turn_the_encounter_and_the_room_type(self):
        rec = self.run_fight(40).log.records[0]
        self.assertEqual(rec["acted"], {"type": "end_turn", "kind": "combat", "label": ["end"]})
        self.assertEqual((rec["turn"], rec["encounter"], rec["room_type"]), (3, "cultist+jaw_worm", "Elite"))

    def test_the_first_decision_has_no_step_before_it_and_a_non_combat_record_has_no_turn(self):
        o, b = obs(hp=80), obs(hp=70)
        ls = lockstep.Lockstep(FakeEmu([view_of(b), view_of(o)]), lambda m_: 0)
        ls.decide(msg("rest", o))
        rec = ls.log.records[0]
        self.assertEqual((rec["acted"], rec["turn"], rec["encounter"]), (None, None, None))
        ls.decide(msg("card_reward", o))  # the lockstep is lost, the step is still tracked
        self.assertEqual(ls.prev_act, {"type": "choice", "kind": "card_reward", "label": ["x"]})

    def test_a_play_and_a_potion_are_told_apart_from_an_end_turn(self):
        self.assertEqual(lockstep.act_type("combat", ["play", 0, "STRIKE", 0]), "play")
        self.assertEqual(lockstep.act_type("combat", ["potion", 0, "FIRE", -1]), "potion")
        self.assertEqual(lockstep.act_type("combat", ["end"]), "end_turn")
        self.assertEqual(lockstep.act_type("select_cards_combat", ["card", 1]), "choice")
        self.assertEqual(lockstep.act_type("map", [3, 1, "Monster"]), "choice")


class OutcomeTests(unittest.TestCase):
    class Env:
        def __init__(self, stream, end):
            self.stream, self.end, self.i = stream, end, 0

        def reset(self, seed, hash=False):
            return self.stream[0], False

        def step(self, a):
            self.i += 1
            if self.i < len(self.stream):
                return self.stream[self.i], False
            return self.end, True

    def _play(self, stream, emu, end):
        return lockstep.play_run(self.Env(stream, end), "SEEDX", emu_factory=lambda s: emu, agent_seed=[5, 0])

    def test_a_run_that_loses_lockstep_then_ends_has_outcome_and_span_in_its_records(self):
        a = obs()
        stream = [msg("rest", a), msg("rest", a), msg("rest", a), msg("rest", a)]
        emu = FakeEmu([view_of(obs(hp=1))])
        m, ls = self._play(stream, emu, {"t": "end", "end": "death", "floor": 7})
        recs = ls.log.records
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["outcome"], {"class": "death", "end": "death", "floor": 7})
        self.assertEqual(recs[0]["not_ranked_span"], {"first": 0, "last": 3, "count": 4})
        self.assertEqual(recs[0]["run"], "SEEDX")
        self.assertEqual(recs[0]["agent_seed"], [5, 0])

    def test_span_closes_at_resume_and_truncated_run_is_classed(self):
        a = obs()
        stream = [msg("rest", a), msg("rest", a), msg("map", a), msg("map", a)]
        emu = FakeEmu([view_of(obs(hp=1))], reanchor_ok=True)
        m, ls = self._play(stream, emu, {"t": "end", "end": "win", "floor": 50})
        self.assertEqual(ls.log.records[0]["not_ranked_span"], {"first": 0, "last": 1, "count": 2})
        self.assertEqual(ls.log.records[0]["outcome"]["class"], "win")
        self.assertEqual(lockstep.run_outcome({"floor": 3}, False)["class"], "truncated")

    def test_a_crash_keeps_the_records_on_the_exception(self):
        a = obs()

        class Boom(self.Env):
            def step(self, x):
                raise ValueError("x")
        with self.assertRaises(ValueError) as cx:
            lockstep.play_run(Boom([msg("rest", a)], None), "S", emu_factory=lambda s: FakeEmu([view_of(obs(hp=1))]),
                              agent_seed=[1, 1])
        rec = cx.exception.lockstep.log.records[0]
        self.assertEqual(rec["outcome"]["class"], "crash")
        self.assertEqual(rec["not_ranked_span"]["count"], 1)
