"""tools/kernel/listener_check.py: the game's L records against hook_table.game_order_key, on synthetic recordings.

Each case is made up (no game data): a faithful dispatch passes, a dispatch with one listener out of the rule's order
is caught with its first pair, and the plumbing around it (nesting, attribution, unplaced listeners, file shapes) is
seen to hold.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.kernel import listener_check as lc  # noqa: E402

HOOKS = {"AfterCardDrawn": "F11", "ModifyDamage": "F08", "AfterActEntered": "F15"}


def L(i, model, kind, owner, index):
    return ["L", i, model, kind, owner, index]


P0, P1, E0, E1 = ["Player", 0], ["Player", 1], ["Enemy", 0], ["Enemy", 1]


def combat_order(i):
    """A dispatch in the game's order: creatures in order (powers, relics, potions, orbs, cards each followed by
    its affliction and enchantment), the pet before the enemies, then the combat-level models."""
    return [
        L(i, "STRENGTH_POWER", "power", P0, 0), L(i, "FEEL_NO_PAIN_POWER", "power", P0, 1),
        L(i, "BURNING_BLOOD", "relic", P0, 0), L(i, "VAJRA", "relic", P0, 1),
        L(i, "FIRE_POTION", "potion", P0, 0), L(i, "BLOCK_POTION", "potion", P0, 2),
        L(i, "LIGHTNING_ORB", "orb", P0, 0),
        L(i, "STRIKE", "card", P0, ["hand", 0]), L(i, "SOME_AFFLICTION", "affliction", P0, ["hand", 0]),
        L(i, "SOME_ENCHANT", "enchantment", P0, ["hand", 0]),
        L(i, "DEFEND", "card", P0, ["hand", 1]), L(i, "BASH", "card", P0, ["draw", 0]),
        L(i, "STRIKE", "card", P0, ["discard", 0]), L(i, "ANGER", "card", P0, ["exhaust", 0]), L(i, "WHIRLWIND", "card", P0, ["play", 0]),
        L(i, "OSTY", "monster", P1, 0),
        L(i, "RAVENOUS_POWER", "power", E0, 0), L(i, "CORPSE_SLUG", "monster", E0, 0), L(i, "CORPSE_SLUG", "monster", E1, 0),
        L(i, "CCC_COMBO_MODEL", "badge", None, None), L(i, "MULTIPLAYER_SCALING_MODEL", "singleton", None, None),
    ]


def rec(events, seed="S1"):
    return {"seed": seed, "steps": [{"trace": events}]}


def enum(i, hook, listeners, level="combat"):
    return [["LB", i, hook, level]] + listeners + [["LE", i]]


class ListenerCheck(unittest.TestCase):
    def run_check(self, events):
        return lc.check([rec(events)], HOOKS)

    def test_faithful_dispatch_agrees(self):
        r = self.run_check(enum(0, "AfterCardDrawn", combat_order(0)))
        f = r["F11"]
        self.assertEqual((f["dispatches"], f["agree"], f["disagree"], f["unplaced"]), (1, 1, 0, 0))

    def test_relic_before_power_is_caught_with_its_pair(self):
        ls = combat_order(0)
        ls[0], ls[2] = ls[2], ls[0]  # BURNING_BLOOD, FEEL_NO_PAIN_POWER, STRENGTH_POWER, VAJRA
        f = self.run_check(enum(0, "ModifyDamage", ls))["F08"]
        self.assertEqual((f["agree"], f["disagree"]), (0, 1))
        self.assertEqual(f["first"]["a"][0], "BURNING_BLOOD")
        self.assertEqual(f["first"]["b"][0], "FEEL_NO_PAIN_POWER")
        self.assertEqual(f["first"]["hook"], "ModifyDamage")

    def test_each_kind_of_misorder_is_caught(self):
        cases = {
            "obtain order": lambda ls: (ls.__setitem__(slice(2, 4), [ls[3], ls[2]]), ls)[1],
            "potion slot": lambda ls: (ls.__setitem__(slice(4, 6), [ls[5], ls[4]]), ls)[1],
            "pile order": lambda ls: (ls.__setitem__(slice(7, 8), []), ls.insert(12, combat_order(0)[7]), ls)[2],
            "enemy before pet": lambda ls: (ls.__setitem__(slice(15, 17), [ls[16], ls[15]]), ls)[1],
            "enchantment before affliction": lambda ls: (ls.__setitem__(slice(8, 10), [ls[9], ls[8]]), ls)[1],
            "monster before its power": lambda ls: (ls.__setitem__(slice(16, 18), [ls[17], ls[16]]), ls)[1],
            "combat model before a creature": lambda ls: [ls[19]] + ls[:19] + ls[20:],
        }
        for name, bad in cases.items():
            with self.subTest(name):
                f = self.run_check(enum(0, "AfterCardDrawn", bad(combat_order(0))))["F11"]
                self.assertEqual(f["disagree"], 1, name)

    def test_nested_enumerations_are_separated_by_id(self):
        outer, inner = combat_order(0), combat_order(1)
        bad_inner = list(reversed(inner[:2])) + inner[2:]
        ev = [["LB", 0, "AfterCardDrawn", "combat"]] + outer[:5] + enum(1, "ModifyDamage", bad_inner) + outer[5:] + [["LE", 0]]
        r = self.run_check(ev)
        self.assertEqual((r["F11"]["agree"], r["F11"]["disagree"]), (1, 0))
        self.assertEqual((r["F08"]["agree"], r["F08"]["disagree"]), (0, 1))

    def test_unplaced_listeners_are_counted_and_left_out(self):
        ls = combat_order(0)
        ls.insert(3, L(0, "MOD_THING", "other", None, None))
        ls.insert(9, L(0, "FLOATING_CARD", "card", P0, ["none", -1]))
        ls.insert(0, L(0, "ODD", "relic", None, 0))  # a creature kind without an owner
        f = self.run_check(enum(0, "AfterCardDrawn", ls))["F11"]
        self.assertEqual((f["agree"], f["disagree"], f["unplaced"]), (1, 0, 3))
        self.assertEqual(f["unplaced_kinds"], {"other": 1, "card/none": 1, "relic": 1})

    def test_run_enumeration_deck_head(self):
        deck = [L(0, "STRIKE", "card", P0, ["deck", 0]), L(0, "SOME_ENCHANT", "enchantment", P0, ["deck", 0]),
                L(0, "DEFEND", "card", P0, ["deck", 1])]
        good = enum(0, "AfterActEntered", deck + combat_order(0), "run")
        f = self.run_check(good)["F15"]
        self.assertEqual((f["agree"], f["disagree"], f["unplaced"]), (1, 0, 3))
        late = enum(0, "AfterActEntered", combat_order(0)[:3] + deck + combat_order(0)[3:], "run")
        self.assertEqual(self.run_check(late)["F15"]["disagree"], 1)
        swapped = enum(0, "AfterActEntered", [deck[2], deck[0], deck[1]] + combat_order(0), "run")
        self.assertEqual(self.run_check(swapped)["F15"]["disagree"], 1)

    def test_attribution(self):
        ev = (enum(0, "?CardCostHelper.TryModifyEnergyCostWithHooks", combat_order(0)) + enum(1, "ModifyDamageInternal", combat_order(1))
              + enum(2, "?Other.Thing", combat_order(2)))
        r = self.run_check(ev)
        self.assertEqual(r["F07"]["dispatches"], 1)
        self.assertEqual(r["-"]["dispatches"], 2)
        # an LB without a hook (older records) takes the last H before it
        ev2 = [["H", "AfterCardDrawn"]] + [["LB", 0, None, "combat"]] + combat_order(0) + [["LE", 0]]
        self.assertEqual(self.run_check(ev2)["F11"]["dispatches"], 1)
        ev3 = [["H", "ModifyDamage"], ["LB", 0]] + combat_order(0) + [["LE", 0]]
        self.assertEqual(self.run_check(ev3)["F08"]["dispatches"], 1)

    def test_empty_and_unclosed_enumerations_do_not_disagree(self):
        r = self.run_check([["LB", 0, "AfterCardDrawn", "combat"], ["LE", 0], ["LB", 1, "AfterCardDrawn", "combat"]] + combat_order(1)[:3])
        self.assertEqual((r["F11"]["dispatches"], r["F11"]["agree"], r["F11"]["disagree"]), (2, 2, 0))

    def test_ids_restart_per_result(self):
        r = lc.check([rec(enum(0, "AfterCardDrawn", combat_order(0)), "A"), rec(enum(0, "AfterCardDrawn", combat_order(0)[::-1]), "B")], HOOKS)
        f = r["F11"]
        self.assertEqual((f["dispatches"], f["agree"], f["disagree"]), (2, 1, 1))
        self.assertEqual(f["first"]["result"], "B")

    def test_file_shapes_and_cli(self):
        good = rec(enum(0, "AfterCardDrawn", combat_order(0)))
        bad = rec(enum(0, "AfterCardDrawn", combat_order(0)[::-1]))
        with tempfile.TemporaryDirectory() as td:
            for name, text in {"a.json": json.dumps({"results": [good]}), "b.json": json.dumps([good]), "c.json": json.dumps(good),
                               "d.jsonl": json.dumps(good) + "\n" + json.dumps(good) + "\n"}.items():
                p = Path(td) / name
                p.write_text(text, encoding="utf-8")
                self.assertTrue(lc.load(p), name)
                self.assertEqual(lc.main([str(p)]), 0, name)
            p = Path(td) / "bad.json"
            p.write_text(json.dumps({"results": [bad]}), encoding="utf-8")
            self.assertEqual(lc.main([str(p)]), 1)

    def test_real_family_file_places_the_dispatchers(self):
        hooks, names = lc.load_families()
        self.assertEqual(len(names), 16)
        for dispatcher, fam in {"AfterCardDrawn": "F11", "ModifyDamage": "F08", "ModifyEnergyCostInCombat": "F07", "ShouldPlay": "F06",
                                "ModifyBlock": "F09", "ModifyHpLost": "F08", "AfterActEntered": "F16", "ModifyMaxEnergy": "F03"}.items():
            self.assertEqual(lc.family_of(dispatcher, hooks), fam, dispatcher)
        self.assertEqual(lc.family_of("NoSuchHook", hooks), "-")
        self.assertEqual(lc.family_of(None, hooks), "-")
        text = lc.table(lc.check([rec(enum(0, "AfterCardDrawn", combat_order(0)))], hooks), names)
        self.assertIn("dispatches", text)
        self.assertIn("F11", text)


if __name__ == "__main__":
    unittest.main()
