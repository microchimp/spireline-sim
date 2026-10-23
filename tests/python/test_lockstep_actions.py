"""The referee's extra actions (none, card-reward alternatives, potions outside fights) and the emulator option each maps to."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "referee"))

import lockstep  # noqa: E402

SLUG = {"cards": {1: "STRIKE", 2: "DEFEND", 3: "ANGER"}, "relics": {}, "potions": {}}


def emu(options, deck=()):
    """An EmuLock with a canned emulator decision (no native library)."""
    e = lockstep.EmuLock.__new__(lockstep.EmuLock)
    applied = []
    e.slug = SLUG
    e.dec = NS(options=options, view=NS(deck=list(deck)))
    e.s = NS(apply=applied.append, view=lambda: NS(deck=list(deck)), _step=lambda a: applied.append(("step", a)))
    e._msg = {}
    e.rq = []
    e._flush_rewards = lambda: None
    e.applied = applied
    return e


def opt(kind, action=0, **kw):
    return NS(kind=kind, action=action, card=kw.get("card"), deck_index=kw.get("deck_index"), node=kw.get("node"),
              price=kw.get("price"), cards=kw.get("cards"), relic=kw.get("relic"), event=None)


def card(cid, up=0):
    return NS(card_id=cid, upgraded=bool(up))


POTION = ["use_potion", 0, "FRUIT_JUICE"]
DISCARD = ["discard_potion", 0, "FRUIT_JUICE"]


class ExtraKindTest(unittest.TestCase):
    def test_each_new_action_is_classified(self):
        self.assertEqual(lockstep.extra_kind("none"), "none")
        self.assertEqual(lockstep.extra_kind("alt:reroll"), "card_alt")
        self.assertEqual(lockstep.extra_kind(POTION), "potion_use")
        self.assertEqual(lockstep.extra_kind(DISCARD), "potion_discard")
        for ordinary in ("skip", "take", ["STRIKE", 0], ["play", 0, "STRIKE", -1], ["potion", 0, "X", -1], "HEAL", ["card", 0, 50]):
            self.assertIsNone(lockstep.extra_kind(ordinary))


class CardRewardTest(unittest.TestCase):
    def test_skip_maps_to_card_skip_and_other_alternatives_have_no_emulator_option(self):
        e = emu([opt("card.take", 0, card=card(1)), opt("card.take", 1, card=card(3)), opt("card.skip", 2)])
        labels = [["STRIKE", 0], ["ANGER", 0], "skip", "alt:reroll"]
        self.assertEqual(e._pair("card_reward", labels), [0, 1, 2, None])
        self.assertIsNone(e.apply("card_reward", labels, 2))
        self.assertEqual(e.applied[-1].kind, "card.skip")
        self.assertTrue(e.apply("card_reward", labels, 3).startswith("no_emulator_option:card_reward"))


class SelectCardsNoneTest(unittest.TestCase):
    def test_none_maps_to_offer_done_where_the_emulator_has_one(self):
        e = emu([opt("select.offer", 0, card=card(1)), opt("select.offer_done", 1)])
        labels = [["STRIKE", 0], "none"]
        self.assertEqual(e._pair("select_cards", labels), [0, 1])
        self.assertIsNone(e.apply("select_cards", labels, 1))
        self.assertEqual(e.applied[-1].kind, "select.offer_done")

    def test_none_on_a_deck_selection_has_no_emulator_option(self):
        e = emu([opt("select.remove", 0, deck_index=0)], deck=[card(1)])
        labels = [["STRIKE", 0], "none"]
        self.assertEqual(e._pair("select_cards", labels), [0, None])
        self.assertTrue(e.apply("select_cards", labels, 1).startswith("no_emulator_option:select_cards"))


class PotionOutsideFightTest(unittest.TestCase):
    def test_potion_actions_never_disturb_the_screens_own_pairing(self):
        cases = {
            "map": ([opt("map.node", 0, node=(1, 2)), opt("map.node", 1, node=(3, 2))], [[1, 2, "Monster"], [3, 2, "Shop"]]),
            "rest": ([opt("rest.heal", 0), opt("rest.smith", 1)], ["HEAL", "SMITH"]),
            "shop": ([opt("shop.card", 0, price=50), opt("shop.leave", 1)], [["card", 0, 50], ["leave"]]),
            "event": ([opt("event.option", 0), opt("event.option", 1)], ["OPTION_A", "OPTION_B"]),
        }
        for kind, (options, core) in cases.items():
            with self.subTest(kind):
                e = emu(options)
                labels = core + [POTION, DISCARD]
                self.assertEqual(e._pair(kind, labels), e._pair(kind, core) + [None, None])
                self.assertNotIn(None, e._pair(kind, core))
                for i in (len(core), len(core) + 1):
                    self.assertTrue(e.apply(kind, labels, i).startswith(f"no_emulator_option:{kind}"))
                self.assertEqual(e.applied, [])  # nothing was applied to the emulator

    def test_reward_screen_potion_action_is_not_taken_as_a_skip(self):
        e = emu([opt("reward.proceed", 0)])
        labels = ["take", "skip", POTION]
        self.assertTrue(e.apply("reward", labels, 2).startswith("no_emulator_option:reward"))
        self.assertEqual(e.rq, [])  # the reward queue the emulator mirrors is untouched



class FakeScreen:
    """The emulator's "up to N" fight screen (CombatEngine ResolveCardSelection, DiscardToHand): candidates, a decline one past
    them, and a reopen after each pick until `amount` are spent."""

    def __init__(self, cards, amount):
        self.cards, self.amount, self.taken, self.declined = list(cards), amount, [], False
        self.obs = [0] * 22
        self._publish()

    def _publish(self):
        self.obs[0] = 1 if self.cards and self.amount > 0 and not self.declined else 0
        self.obs[1] = len(self.cards) if self.obs[0] else 0
        for i, (cid, up) in enumerate(self.cards[:10]):
            self.obs[2 + 2 * i], self.obs[3 + 2 * i] = cid, up

    @property
    def is_open(self):
        return bool(self.obs[0])

    def mask(self, n):
        count = self.obs[1] if self.is_open else 0
        return ([True] * count + ([True] if self.is_open else []) + [False] * n)[:n]

    def step(self, a):
        if a == self.obs[1]:
            self.declined = True
        else:
            self.taken.append(self.cards.pop(a))
            self.amount -= 1
        self._publish()
        return 0


def screen_emu(cards, amount):
    """An EmuLock standing on a fake fight selection screen (no native library)."""
    screen = FakeScreen(cards, amount)
    e = emu([])
    e.ls = NS(_SELECTION_KIND=0, _SELECTION_COUNT=1, _SELECTION=2, _CARD=2, _MAX_CANDIDATES=10,
              _card_at=lambda obs, off: (obs[off], obs[off + 1]))
    e._obs = lambda: screen.obs
    e.native = NS(run_action_mask=lambda h, n: screen.mask(n), run_step=lambda h, a, t, *bufs: screen.step(a))
    e.s = NS(handle=0, phase=1, env=NS(_run_obs_buf=screen.obs, _run_rew_buf=None, _run_terminal_buf=None, _run_truncated_buf=None))
    e.E = NS(rc=NS(PHASE_COMBAT=1, PHASE_COMPLETE=9))
    e._forced = lambda want: None
    e._auto_combat = lambda: None
    return e, screen


PICK1 = [["STRIKE", 0], ["DEFEND", 0], ["ANGER", 0], "done"]  # a 0..2 screen over three cards, the first ask
PICK2 = [["STRIKE", 0], ["ANGER", 0], "done"]  # the second ask, after DEFEND
CTX1, CTX2 = {"ctx": {"pick": 1, "of": 2}}, {"ctx": {"pick": 2, "of": 2}}
THREE = [(1, 0), (2, 0), (3, 0)]


class FullSelectScreenTest(unittest.TestCase):
    """A fight card screen taken with full_select: each pick and the "done" meet the emulator's screen, and both leave it together."""

    def test_done_is_an_extra_action_that_has_an_emulator_option(self):
        self.assertEqual(lockstep.extra_kind("done"), "done")
        self.assertNotIn("done", lockstep.NO_EMULATOR_KINDS)

    def test_two_picks_close_both_screens_without_a_done(self):
        e, s = screen_emu(THREE, 2)
        self.assertIsNone(e.sync("select_cards_combat", PICK1, CTX1))
        self.assertIsNone(e.apply("select_cards_combat", PICK1, 1))  # DEFEND
        self.assertIsNone(e.sync("select_cards_combat", PICK2, CTX2))
        self.assertIsNone(e.apply("select_cards_combat", PICK2, 1))  # ANGER
        self.assertEqual(s.taken, [(2, 0), (3, 0)])
        self.assertIsNone(e.sync("combat", [["end"]], {}))  # the referee's next ask: the emulator's screen is closed too

    def test_zero_picks_then_done_declines_the_screen(self):
        e, s = screen_emu(THREE, 2)
        self.assertIsNone(e.sync("select_cards_combat", PICK1, CTX1))
        self.assertIsNone(e.apply("select_cards_combat", PICK1, 3))
        self.assertEqual((s.taken, s.is_open), ([], False))
        self.assertIsNone(e.sync("combat", [["end"]], {}))

    def test_one_pick_then_done(self):
        e, s = screen_emu(THREE, 2)
        self.assertIsNone(e.apply("select_cards_combat", PICK1, 1))
        self.assertIsNone(e.sync("select_cards_combat", PICK2, CTX2))
        self.assertIsNone(e.apply("select_cards_combat", PICK2, 2))
        self.assertEqual((s.taken, s.is_open), ([(2, 0)], False))
        self.assertIsNone(e.sync("combat", [["end"]], {}))

    def test_a_screen_the_emulator_keeps_open_is_a_difference(self):
        e, s = screen_emu(THREE, 3)  # the emulator would take a third; the referee's screen ended after two
        e.apply("select_cards_combat", PICK1, 1)
        e.apply("select_cards_combat", PICK2, 1)
        self.assertEqual(e.sync("combat", [["end"]], {}), "selection:left_open")

    def test_a_screen_the_emulator_closes_early_is_a_difference(self):
        e, s = screen_emu(THREE, 1)
        e.apply("select_cards_combat", PICK1, 1)
        self.assertEqual(e.sync("select_cards_combat", PICK2, CTX2), "selection:closed_early")

    def test_other_cards_left_on_the_screen_are_a_difference(self):
        e, s = screen_emu(THREE, 2)
        e.apply("select_cards_combat", PICK1, 1)
        other = [["STRIKE", 0], ["DEFEND", 0], "done"]
        self.assertEqual(e.sync("select_cards_combat", other, CTX2), "selection:candidates_differ")

    def test_a_done_the_emulator_cannot_take_is_refused(self):
        e, s = screen_emu(THREE, 2)
        s.obs[0] = 0  # no open screen: no decline
        self.assertTrue(e.apply("select_cards_combat", PICK1, 3).startswith("unmapped_label"))

    def test_views_are_compared_at_the_first_pick_only(self):
        class Fake:
            def sync(self, kind, labels, msg):
                return None

            def view(self):
                return {"act": 0, "floor": 2, "hp": 50, "max_hp": 80, "gold": 0, "deck": [], "relics": [], "potions": [],
                        "combat": {"hand": [["DEFEND", 0], ["STRIKE", 0]], "enemy_hp": [10]}}

        obs = {"act": 0, "floor": 2, "hp": 50, "max_hp": 80, "gold": 0, "deck": [], "relics": [], "potions": [],
               "combat": {"hand": [["DEFEND", 0]], "enemies": [{"hp": 10}]}}
        ls = lockstep.Lockstep(Fake(), lockstep.referee_only_policy)
        m1 = {"kind": "select_cards_combat", "actions": PICK1, "obs": obs, **CTX1}
        m2 = {"kind": "select_cards_combat", "actions": PICK2, "obs": obs, **CTX2}
        self.assertTrue(ls._equal("select_cards_combat", PICK1, m1)[1])  # the emulator's hand holds a card the referee's does not
        self.assertEqual(ls._equal("select_cards_combat", PICK2, m2)[1], [])  # inside the screen the referee has not moved it yet

    def test_the_fallback_takes_one_card_then_done(self):
        self.assertEqual(lockstep.referee_only_policy({"kind": "select_cards_combat", "actions": PICK1, **CTX1}), 0)
        self.assertEqual(lockstep.referee_only_policy({"kind": "select_cards_combat", "actions": PICK2, **CTX2}), 2)

if __name__ == "__main__":
    unittest.main()


class OfferedCountingTest(unittest.TestCase):
    """envproto.extras_of: how the smoke counts each new action kind offered on a canned decision message."""

    def test_each_new_kind_is_found_on_its_decision(self):
        import envproto

        msgs = [
            {"kind": "card_reward", "actions": [["STRIKE", 0], "skip", "alt:reroll"]},
            {"kind": "select_cards", "actions": [["STRIKE", 0], ["DEFEND", 0], "none"]},
            {"kind": "select_cards_combat", "actions": [["STRIKE", 0], "none"]},
            {"kind": "map", "actions": [[1, 2, "Monster"], [2, 2, "Shop"], ["use_potion", 0, "FRUIT_JUICE"], ["discard_potion", 0, "FRUIT_JUICE"]]},
            {"kind": "reward", "actions": ["take", "skip", ["discard_potion", 1, "X"]]},
            {"kind": "shop", "actions": [["card", 0, 50], ["leave"], ["use_potion", 0, "X"]]},
            {"kind": "event", "actions": ["A", "B", ["discard_potion", 0, "X"]]},
            {"kind": "rest", "actions": ["HEAL", "SMITH", ["use_potion", 0, "X"]]},
        ]
        got = sorted(k for m in msgs for _, k in envproto.extras_of(m))
        self.assertEqual(got, sorted([
            "card_skip@card_reward", "card_alt@card_reward", "none@select_cards", "none@select_cards_combat",
            "potion_use@map", "potion_discard@map", "potion_discard@reward", "potion_use@shop", "potion_discard@event",
            "potion_use@rest"]))
        # a reward screen's take/skip is not a card skip, and an ordinary combat message offers none of them
        self.assertEqual(envproto.extras_of({"kind": "reward", "actions": ["take", "skip"]}), [])
        self.assertEqual(envproto.extras_of({"kind": "combat", "actions": [["end"], ["play", 0, "STRIKE", 0], ["potion", 0, "X", -1]]}), [])


class FightSelectionNoneTest(unittest.TestCase):
    def test_none_in_a_fight_selection_is_not_an_emulator_action(self):
        e = emu([])
        self.assertIsNone(e._select_action("none"))  # was: ValueError from int("o") in the gate smoke
        self.assertTrue(e._apply_combat("select_cards_combat", [["STRIKE", 0], "none"], 1).startswith("unmapped_label"))
