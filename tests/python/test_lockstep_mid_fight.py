"""Fix d of lane 21's memo (Act 3 week, lane 22): the copy is re-anchored inside a fight, at the very decision a difference is found
(or at the next fight decision after a loss there, such as a refused step).

Before, a difference inside a fight lost the copy until the next fight start or room entry, and the stand-in played the rest of the
fight: 78 of the 95 lost first-Act-3-boss arrivals of 2026-11-20 were lost inside the boss fight. A mid-fight re-anchor counts only
when afterwards the views agree, the energy agrees, the second (unranked) comparison is empty and the copy's legal actions are the
game's (one key a card, End Turn, one a potion); at most MID_FIGHT_LIMIT tries a fight. The hidden state it uses (the driver's
anchor_state command) reaches the emulator side's take_anchor_state only. Fakes first; the native-library tests at the end skip
without the library."""

import json
import os
import queue
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import leakprobes  # noqa: E402
import lockstep  # noqa: E402
from test_lockstep import FakeEmu, FakeRun, obs, stub_emu, view_of  # noqa: E402

try:
    leakprobes.native_env()
    from sts2_gym import native  # noqa: E402

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    HAVE_NATIVE = False

SENT = "SENTINEL_MIDFIGHT_7Q2"
HIDDEN = {"rng": {"Shuffle": {"seed": 1, "counter": 5}, SENT: {"seed": 0, "counter": 0}}, "draw": [[SENT, False]]}
LABELS = [["play", 0, "STRIKE_IRONCLAD", 0], ["end"]]
LEGAL = ["play:STRIKE_IRONCLAD", "end"]


def fight_obs(hp=80, energy=3, floor=5, act=0):
    o = obs(hp=hp, floor=floor, combat=True)
    o["act"] = act
    o["combat"]["player"] = {"hp": hp, "energy": energy, "block": 0, "powers": []}
    o["combat"]["round"] = 2
    return o


def fmsg(o, kind="combat", labels=LABELS):
    return {"t": "decide", "kind": kind, "obs": o, "actions": labels, "h": "abc"}


class MidEmu(FakeEmu):
    """A side that anchors inside a fight: the views follow a script until a re-anchor, then the game's; its probe and second
    comparison answer from settings (a condition of a counted re-anchor each)."""

    anchors_mid_fight = True
    wants_anchor_state = True

    def __init__(self, views, reanchor_ok=True, energy=3, legal=LEGAL, hidden_found=(), after=None):
        super().__init__(views, reanchor_ok=reanchor_ok)
        self.energy, self.legal, self.hidden_found, self.after = energy, list(legal), list(hidden_found), after
        self.told, self.seen = [], []

    def take_anchor_state(self, hidden):
        self.told.append(hidden)

    def sync(self, kind, labels, msg=None):
        self.seen.append(msg)
        return super().sync(kind, labels, msg)

    def choose(self, kind, labels, msg):
        self.seen.append(msg)
        return super().choose(kind, labels, msg)

    def reanchor(self, plan, msg):
        ok = super().reanchor(plan, msg)
        if ok and self.after is not None:
            self.anchored = self.after  # the views after the anchor, when they still differ
        return ok

    def probe(self):
        return {"emu_energy": self.energy, "emu_legal": self.legal, "emu_potions": [], "emu_selection": False}

    def hidden_diff(self, o):
        return self.hidden_found if self.anchored is not None else []


def run(emu, stream, **kw):
    ls = lockstep.Lockstep(emu, lambda m: 0, **kw)
    for m in stream:
        ls.decide(m)
    return ls


class MidFightRuleTests(unittest.TestCase):
    def test_a_difference_inside_a_fight_re_anchors_at_that_decision_and_the_agent_answers(self):
        a, b = fight_obs(), fight_obs(hp=60)
        emu = MidEmu([view_of(a), view_of(a)])
        ls = run(emu, [fmsg(a), fmsg(b)], ask_hidden=lambda: HIDDEN)
        self.assertEqual(emu.reanchored, ["mid_fight"])
        self.assertIsNone(ls.lost)
        self.assertEqual((ls.stats["ranked"], ls.stats["reanchors"]), (2, 1))
        rec = ls.log.records[0]
        self.assertEqual((rec["decision"], rec["reanchor_plan"], rec["reanchored"]), (1, "mid_fight", True))
        self.assertEqual({k: rec["reanchor"][k] for k in ("try", "ok", "why")}, {"try": 1, "ok": True, "why": None})
        self.assertEqual(ls.stats["reanchor_by_plan"], {"mid_fight:ok": 1})
        self.assertEqual(emu.applied[-1], 1)  # the agent's (the side's) answer, not the stand-in's 0

    def test_the_first_decision_of_a_fight_stays_a_fight_start(self):
        emu = MidEmu([view_of(fight_obs())])
        run(emu, [fmsg(fight_obs(hp=60))])
        self.assertEqual(emu.reanchored, ["fight_start"])

    def test_a_card_screen_waits_for_the_fight_decision_after_it(self):
        a, b = fight_obs(), fight_obs(hp=60)
        emu = MidEmu([view_of(a), view_of(a), view_of(a)])
        ls = run(emu, [fmsg(a), fmsg(b, kind="select_cards_combat", labels=[["STRIKE_IRONCLAD", 0], "done"]), fmsg(b)])
        self.assertEqual(emu.reanchored, ["mid_fight"])  # at decision 2, not at the screen
        self.assertEqual(ls.log.records[0]["reanchor_plan"], None)
        self.assertEqual(ls.log.records[0]["not_ranked_span"]["count"], 1)  # the screen itself, answered by the stand-in
        self.assertIsNone(ls.lost)
        self.assertEqual(ls.stats["resumed"], 1)

    def test_it_counts_only_when_energy_second_comparison_and_legal_actions_agree(self):
        a, b = fight_obs(), fight_obs(hp=60)
        cases = {
            "energy:3/2": dict(energy=2),
            "hidden:.enemies[0].powers": dict(hidden_found=[[".enemies[0].powers", [["WEAK_POWER", 1]], []]]),
            "legal:+play:STRIKE_IRONCLAD": dict(legal=["end"]),
            "legal:-potion:FIRE_POTION": dict(legal=LEGAL + ["potion:FIRE_POTION"]),
            "diff:.hp": dict(after=view_of(fight_obs(hp=61))),
        }
        for why, kw in cases.items():
            with self.subTest(why):
                emu = MidEmu([view_of(a), view_of(a)], **kw)
                ls = run(emu, [fmsg(a), fmsg(b)])
                rec = ls.log.records[0]
                self.assertEqual((rec["reanchored"], rec["reanchor"]["ok"], rec["reanchor"]["why"]), (False, False, why))
                self.assertEqual(ls.stats["reanchor_fail_reasons"], {f"mid_fight:{why}": 1})
                self.assertIsNotNone(ls.lost)

    def test_a_failed_re_anchor_leaves_the_row_as_a_loss_is_recorded_today(self):
        a, b = fight_obs(), fight_obs(hp=60)
        emu = MidEmu([view_of(a), view_of(a)], reanchor_ok=False)
        ls = run(emu, [fmsg(a), fmsg(b)])
        rec = ls.log.records[0]
        self.assertEqual((rec["reanchored"], ls.lost, ls.stats["first_loss"]), (False, "state_diverged", {"decision": 1, "reason": "state_diverged"}))
        self.assertEqual(emu.applied, [1])  # the stand-in's answer is not mirrored to a lost copy
        self.assertEqual((ls.stats["ranked"], ls.stats["not_ranked"]), (1, 1))

    def test_the_next_fight_decision_after_a_refused_step_re_anchors(self):
        class Refusing(MidEmu):
            def apply(self, kind, labels, idx):
                super().apply(kind, labels, idx)
                return "refused:(0, -1)" if len(self.applied) == 1 else None

        a = fight_obs()
        emu = Refusing([view_of(a), view_of(a), view_of(a)])
        ls = run(emu, [fmsg(a), fmsg(fight_obs(hp=70)), fmsg(fight_obs(hp=70))])
        self.assertEqual(emu.reanchored, ["mid_fight"])
        self.assertIsNone(ls.lost)
        loss = ls.log.records[0]
        self.assertTrue(loss["reason"].startswith("apply:refused"))
        self.assertEqual(loss["resumed_at"], {"decision": 1, "floor": 5, "plan": "mid_fight"})
        self.assertEqual([(t["decision"], t["ok"]) for t in loss["mid_fight_tries"]], [(1, True)])

    def test_at_most_eight_tries_a_fight_and_a_new_fight_starts_again(self):
        a, b = fight_obs(), fight_obs(hp=60)
        limit = lockstep.MID_FIGHT_LIMIT
        emu = MidEmu([view_of(a)] + [view_of(a)] * 12, reanchor_ok=[False] * limit + [True] * 3)
        ls = run(emu, [fmsg(a)] + [fmsg(b)] * 11)
        self.assertEqual(emu.reanchored, ["mid_fight"] * limit)
        self.assertEqual(ls.stats["mid_fight_capped"], 11 - limit)
        self.assertEqual(len(ls.log.records[0]["mid_fight_tries"]), limit - 1)  # the first try is the record's own
        emu.reanchored.clear()
        ls.decide({"t": "decide", "kind": "map", "obs": obs(floor=6), "actions": [[1, 1, "Monster"], [2, 1, "Monster"]], "h": "x"})
        ls.decide(fmsg(fight_obs(hp=60, floor=7)))
        ls.decide(fmsg(fight_obs(hp=59, floor=7)))
        self.assertEqual(emu.reanchored, ["room_entry", "fight_start", "mid_fight"])
        self.assertIsNone(ls.lost)

    def test_a_loss_from_before_the_fight_waits_for_the_next_boundary(self):
        # Lost at a rest site (floor 4) and the fight start on floor 5 not held: the fight's later decisions try nothing, as before.
        emu = MidEmu([view_of(obs(hp=50, floor=4))], reanchor_ok=False)
        ls = run(emu, [{"t": "decide", "kind": "rest", "obs": obs(floor=4), "actions": [["HEAL"], ["SMITH"]], "h": "x"},
                       fmsg(fight_obs()), fmsg(fight_obs()), fmsg(fight_obs(hp=70))])
        self.assertEqual(emu.reanchored, ["fight_start"])
        self.assertIsNotNone(ls.lost)

    def test_a_deck_or_relic_difference_is_not_tried_inside_the_fight(self):
        a = fight_obs()
        b = fight_obs(hp=60)
        b["deck"] = b["deck"] + [["BASH", 0]]  # a card the fight's record cannot give the copy's run deck
        emu = MidEmu([view_of(a), view_of(a)])
        ls = run(emu, [fmsg(a), fmsg(b)])
        self.assertEqual(emu.reanchored, [])
        self.assertIsNone(ls.log.records[0]["reanchor_plan"])
        self.assertIsNotNone(ls.lost)

    def test_a_boss_fight_counts_its_mid_fight_re_anchors(self):
        a, b = fight_obs(floor=48, act=2), fight_obs(hp=60, floor=48, act=2)
        emu = MidEmu([view_of(a), view_of(a)])
        ls = run(emu, [fmsg(a), fmsg(b), fmsg(b)])
        fight = ls.stats["boss_fights"][0]
        self.assertEqual((fight["reanchors"], fight["reanchor_failed"], fight["stand_in"]), (1, 0, 0))
        emu = MidEmu([view_of(a), view_of(a)], energy=0)
        ls = run(emu, [fmsg(a), fmsg(b), fmsg(b)])
        fight = ls.stats["boss_fights"][0]
        self.assertEqual((fight["reanchors"], fight["reanchor_failed"], fight["stand_in"]), (0, 2, 2))

    def test_a_side_without_mid_fight_anchoring_keeps_its_boss_entries_as_they_were(self):
        a = fight_obs(floor=48, act=2)
        ls = run(FakeEmu([view_of(a)]), [fmsg(a), fmsg(a)])
        self.assertEqual(set(ls.stats["boss_fights"][0]), {"act", "floor", "first", "decisions", "stand_in", "first_stand_in"})

    def test_the_action_keys_collapse_hand_positions_targets_and_potion_slots(self):
        labels = [["play", 0, "STRIKE_IRONCLAD", 0], ["play", 0, "STRIKE_IRONCLAD", 1], ["play", 2, "STRIKE_IRONCLAD", 0],
                  ["play", 1, "DEFEND_IRONCLAD", -1], ["end"], ["potion", 0, "FIRE_POTION", 0], ["potion", 2, "FIRE_POTION", 1]]
        self.assertEqual(lockstep.game_action_keys(labels), {"play:STRIKE_IRONCLAD", "play:DEFEND_IRONCLAD", "end", "potion:FIRE_POTION"})
        names = ["play:STRIKE_IRONCLAD", "play:STRIKE_IRONCLAD", "play:DEFEND_IRONCLAD", "end", "potion:FIRE_POTION", "decline"]
        self.assertEqual(lockstep.emu_action_keys(names), lockstep.game_action_keys(labels))


class MidFightFirewallTests(unittest.TestCase):
    def test_the_hidden_state_is_asked_only_at_a_try_and_reaches_the_sides_hook_and_nothing_else(self):
        a, b = fight_obs(), fight_obs(hp=60)
        asked, fallbacks = [], []
        emu = MidEmu([view_of(a), view_of(a), view_of(a)])

        def ask():
            asked.append(1)
            return HIDDEN

        ls = lockstep.Lockstep(emu, lambda m: (fallbacks.append(m), 0)[1], ask_hidden=ask)
        for m in (fmsg(a), fmsg(b), fmsg(b)):
            ls.decide(m)
        self.assertEqual(len(asked), 1)  # the one try; the decisions in step ask nothing
        self.assertEqual(emu.told, [None, None, HIDDEN, None])  # each decision's own None, the try's state
        ls.finish({"class": "death"})
        for m in emu.seen + fallbacks:
            self.assertNotIn(SENT, json.dumps(m))
        self.assertNotIn(SENT, json.dumps(ls.log.records))
        self.assertNotIn(SENT, json.dumps(ls.stats))
        self.assertEqual(ls.stats["mid_fight_hidden"], 1)

    def test_without_the_command_the_side_is_told_none_and_anchors_on_the_observation(self):
        a, b = fight_obs(), fight_obs(hp=60)
        for ask in (None, lambda: None):
            emu = MidEmu([view_of(a), view_of(a)])
            ls = run(emu, [fmsg(a), fmsg(b)], ask_hidden=ask)
            self.assertEqual(emu.told, [None, None, None])
            self.assertIsNone(ls.lost)
            self.assertNotIn("mid_fight_hidden", ls.stats)

    def test_play_run_hands_the_envs_command_over_only_when_the_reset_asked_for_the_hidden_state(self):
        class Env:
            def __init__(self):
                self.resets = []

            def reset(self, seed, **k):
                self.resets.append(k)
                return {"t": "end", "end": "death", "floor": 1}, True

            def anchor_state(self):
                return HIDDEN

        made = []

        def factory(s):
            made.append(MidEmu([view_of(obs())]))
            return made[-1]

        seen = []
        real = lockstep.Lockstep.__init__

        def spy(self, *a, **k):
            seen.append((k.get("ask_hidden"), k.get("mid_fight")))
            real(self, *a, **k)

        env = Env()
        with mock.patch.object(lockstep.Lockstep, "__init__", spy):
            lockstep.play_run(env, "S1", emu_factory=factory)
            lockstep.play_run(env, "S1", emu_factory=factory, anchor_state=False)
            with mock.patch.dict(os.environ, {"SPIRELINE_MID_FIGHT": "0"}):
                lockstep.play_run(env, "S1", emu_factory=factory)
        self.assertEqual(seen[0], (env.anchor_state, True))
        self.assertEqual(seen[1], (None, True))
        self.assertEqual(seen[2], (env.anchor_state, False))


class GameEnvCommandTests(unittest.TestCase):
    """GameEnv.anchor_state against a scripted driver: a new one answers the command, an old one answers bad_action and sends the
    decision again (both read and dropped, and the process is not asked again)."""

    def env(self, replies):
        import envproto  # noqa: PLC0415

        e = envproto.GameEnv.__new__(envproto.GameEnv)
        e._q, e.sent, e.last = queue.Queue(), [], {"t": "decide", "kind": "combat"}
        for r in replies:
            e._q.put(r)
        e._send = e.sent.append
        e.proc = SimpleNamespace(poll=lambda: None)
        return e

    def test_a_new_driver_answers_with_the_state_and_the_decision_stays_open(self):
        e = self.env([{"t": "anchor_state", "state": HIDDEN}])
        self.assertEqual(e.anchor_state(timeout=1), HIDDEN)
        self.assertEqual(e.sent, [{"cmd": "anchor_state"}])
        self.assertEqual(e.last, {"t": "decide", "kind": "combat"})  # the decision the caller holds is still the last
        self.assertTrue(e._q.empty())

    def test_an_old_drivers_refusal_and_its_decision_again_are_dropped_once_for_the_process(self):
        e = self.env([{"t": "bad_action", "n": 2}, {"t": "decide", "kind": "combat", "h": "again"}])
        self.assertIsNone(e.anchor_state(timeout=1))
        self.assertTrue(e._q.empty())
        self.assertIsNone(e.anchor_state(timeout=1))  # not asked again
        self.assertEqual(e.sent, [{"cmd": "anchor_state"}])
        self.assertEqual(e.last, {"t": "decide", "kind": "combat"})

    def test_anything_else_is_a_broken_channel(self):
        import envproto  # noqa: PLC0415

        with self.assertRaises(envproto.GameDied):
            self.env([{"t": "end", "end": "death"}]).anchor_state(timeout=1)
        with self.assertRaises(envproto.GameDied):
            self.env([{"t": "bad_action"}, {"t": "end"}]).anchor_state(timeout=1)

    def test_the_play_modes_wrapper_passes_its_timeout(self):
        import envproto  # noqa: PLC0415

        calls = []
        inner = SimpleNamespace(anchor_state=lambda timeout: (calls.append(timeout), HIDDEN)[1])
        self.assertEqual(envproto._Timed(inner, 42).anchor_state(), HIDDEN)
        self.assertEqual(calls, [42])
        self.assertIsNone(envproto._Timed(SimpleNamespace(), 42).anchor_state())


class EmuLockMidFightTests(unittest.TestCase):
    """EmuLock's side of plan mid_fight, the native calls stubbed."""

    def emu(self, phase=FakeRun.COMBAT, floor=5, selection=0):
        e = stub_emu(SimpleNamespace(phase=phase, handle=3, env=SimpleNamespace(_run_obs_buf=object()),
                                     view=lambda: SimpleNamespace(floor=floor)), hand=[["STRIKE_IRONCLAD", 0]])
        e._obs = lambda: [selection]
        e.calls = []
        e.native.run_reanchor_fight_hidden = lambda h, rec, buf: (e.calls.append(("hidden", rec)), (5, "moves=set;patched=hand"))[1]
        e.native.run_reanchor_fight = lambda h, rec, buf: (e.calls.append(("obs", rec)), (2, "moves=unknown:0;patched=hand"))[1]
        e._order_deck = lambda o: None
        e._relic_counters = lambda: []
        e.native.run_debug_set_player = lambda *a, **k: None
        return e

    def test_the_copys_own_fight_is_put_on_the_record_with_the_belt(self):
        e = self.emu()
        record = {"hand": [["STRIKE_IRONCLAD", False, 1, True]], "draw": [], "enemies": [], "round": 3}
        o = {"floor": 5, "gold": 9, "potions": [None, "FIRE_POTION"], "combat": record}
        e.take_anchor_state(HIDDEN)
        self.assertTrue(e.reanchor("mid_fight", {"obs": o}), e.fail)
        kind, sent = e.calls[0]
        self.assertEqual((kind, sent["potions"], sent["round"], sent["draw"]), ("hidden", [None, "FIRE_POTION"], 3, HIDDEN["draw"]))
        self.assertEqual((e.anchored_with, e.anchor_moves), ("hidden", "set"))
        self.assertIsNone(e._hidden)  # consumed
        e = self.emu()
        self.assertTrue(e.reanchor("mid_fight", {"obs": o}))  # no hidden state: the observation's record
        self.assertEqual((e.calls[0][0], e.anchored_with, e.anchor_moves), ("obs", "observation", "unknown:0"))

    def test_a_fight_start_record_is_sent_without_the_belt_as_before(self):
        e = self.emu()
        record = {"hand": [], "draw": [], "enemies": []}
        self.assertTrue(e.reanchor("fight_start", {"obs": {"floor": 5, "potions": ["FIRE_POTION"], "combat": record}}))
        self.assertNotIn("potions", e.calls[0][1])

    def test_a_copy_out_of_its_fight_elsewhere_or_on_a_card_screen_is_not_re_anchored(self):
        o = {"floor": 5, "combat": {"hand": [], "draw": [], "enemies": []}}
        for e, why in ((self.emu(phase=7), "not_in_fight"), (self.emu(floor=4), "floor"), (self.emu(selection=1), "selection_open")):
            with self.subTest(why):
                self.assertFalse(e.reanchor("mid_fight", {"obs": o}))
                self.assertEqual((e.fail, e.calls), (why, []))


def to_round_two(seed="QS2GYXRKWN"):
    """A real EmuLock in its first fight, at the start of round 2 (End Turn once)."""
    e = lockstep.EmuLock(seed, sims=2)
    for _ in range(60):
        if e.s.phase == e.E.rc.PHASE_COMBAT:
            break
        d = e.s.decision()
        e.s.apply(d.options[e.rules.pick(e.s.view(), d, e.rng).index])
    env = e.s.env
    e.native.run_step(e.s.handle, len(e._hand()), -1, env._run_obs_buf, env._run_rew_buf, env._run_terminal_buf,
                      env._run_truncated_buf)
    return e


@unittest.skipUnless(HAVE_NATIVE, "needs the native emulator library")
class MidFightAnchorNativeTests(unittest.TestCase):
    """A real EmuLock at round 2 of its own fight, and the game's record of that decision perturbed: the enemies lower, the player
    lower, the draw pile in an order of its own (the driver's hidden state). The lockstep re-anchors it mid-fight, and the four
    conditions are read off the real side (its view, probe and second comparison)."""

    def setUp(self):
        import re  # noqa: PLC0415

        self.e = e = to_round_two()
        self.addCleanup(e.close)
        self.assertEqual(e.s.phase, e.E.rc.PHASE_COMBAT)
        nat = e.native
        h = nat.run_fork_combat(e.s.handle, False, 0, (nat.ctypes.c_int * nat.RUN_OBS_SIZE)())
        try:
            piles = {p: nat.get_pile(h, p) for p in ("hand", "draw", "discard", "exhaust")}
            roster = nat.get_enemies(h)
        finally:
            nat.destroy(h)
        id_map = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))
        enemy = {v: re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", k).upper() for k, v in id_map["enemies"].items()}

        def pile(cards):
            return [[e.slug["cards"][c], bool(up)] for c, up in cards]

        view, rv = e.s.view(), e.view()
        self.hp = view.hp - 2
        self.enemy_hp = [max(1, hp - 3) for _d, _s, hp in roster]
        self.draw = pile(piles["draw"])[::-1]
        self.assertGreater(len(self.draw), 1)
        self.record = {"in_combat": True, "round": 2,
                       "player": {"hp": self.hp, "max_hp": view.max_hp, "block": 0, "energy": int(e._obs()[3]), "powers": []},
                       "hand": [[c, up, 1, True] for c, up in pile(piles["hand"])], "draw": sorted(pile(piles["draw"])),
                       "discard": pile(piles["discard"]), "exhaust": pile(piles["exhaust"]),
                       "enemies": [{"name": enemy[d], "hp": hp, "max_hp": hp + 3, "block": 0, "powers": []}
                                   for (d, _s, _h), hp in zip(roster, self.enemy_hp)]}
        belt = rv["potions"] + [None] * (view.max_potion_slots - len(rv["potions"]))
        self.obs = {"act": rv["act"], "floor": rv["floor"], "hp": self.hp, "max_hp": rv["max_hp"], "gold": rv["gold"],
                    "deck": rv["deck"], "relics": rv["relics"], "potions": belt, "combat": self.record}
        names = e.probe()["emu_legal"]
        self.labels = [["end"] if n == "end" else ["play", 0, n[5:], 0] if n.startswith("play:") else ["potion", 0, n[7:], -1]
                       for n in names]
        self.hidden = {"draw": self.draw, "rng": {"Shuffle": {"seed": 701, "counter": 9}, "CombatCardSelection": {"seed": 702, "counter": 3}}}

    def decide(self, labels):
        ls = lockstep.Lockstep(self.e, lockstep.referee_only_policy, ask_hidden=lambda: self.hidden)
        ls.prev_kind = "combat"  # a fight decision before this one: not the fight's first
        ls.decide({"t": "decide", "kind": "combat", "obs": self.obs, "actions": labels, "h": "x"})
        return ls

    def test_a_perturbed_round_two_record_is_re_anchored_and_holds(self):
        e = self.e
        self.assertNotEqual(e.view()["combat"]["enemy_hp"], self.enemy_hp)
        with mock.patch.object(lockstep.EmuLock, "choose", lambda *a: None):  # the stand-in answers: the state is read before any step
            ls = self.decide(self.labels)
        rec = ls.log.records[0]
        self.assertEqual(rec["reanchor_plan"], "mid_fight")
        self.assertTrue(rec["reanchored"], rec.get("reanchor"))
        self.assertEqual(rec["reanchor"]["with"], "hidden")
        self.assertIsNone(ls.lost)
        self.assertEqual(ls.stats["reanchor_by_plan"], {"mid_fight:ok": 1})

    def test_the_anchored_state_is_the_records(self):
        e = self.e
        with mock.patch.object(lockstep.EmuLock, "choose", lambda *a: None), \
                mock.patch.object(lockstep.EmuLock, "apply", lambda *a: None):  # the state as the anchor left it
            self.decide(self.labels)
        self.assertEqual(e.view()["combat"]["enemy_hp"], self.enemy_hp)
        self.assertEqual(e.view()["hp"], self.hp)
        self.assertEqual(e._hand(), [[c[0], int(c[1])] for c in self.record["hand"]])
        h = e.native.run_fork_combat(e.s.handle, False, 0, (e.native.ctypes.c_int * e.native.RUN_OBS_SIZE)())
        try:
            draw = [[e.slug["cards"][c], bool(up)] for c, up in e.native.get_pile(h, "draw")]
        finally:
            e.native.destroy(h)
        self.assertEqual(draw, self.draw)  # the game's order from the hidden state
        self.assertIsNone(e._hidden)

    def test_a_legal_list_the_copy_does_not_share_is_not_held(self):
        with mock.patch.object(lockstep.EmuLock, "choose", lambda *a: None):
            ls = self.decide(self.labels + [["potion", 2, "FIRE_POTION", 0]])
        rec = ls.log.records[0]
        self.assertFalse(rec["reanchored"])
        self.assertEqual(rec["reanchor"]["why"], "legal:+potion:FIRE_POTION")
        self.assertEqual(ls.lost, "state_diverged")


if __name__ == "__main__":
    unittest.main()
