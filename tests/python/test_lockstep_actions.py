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
