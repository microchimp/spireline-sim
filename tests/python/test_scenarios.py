"""referee/scenarios.py and tools/fidelity/setup_compare.py: setup fight specs and the item-by-item check.

A small export in the game's table shape stands in for the real one. Specs must keep the item order the
referee and EmuDump both rely on (map items, the fight, combat items), stay inside one character's
pools, and be flagged when no real run could hold them.
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT))

import scenarios  # noqa: E402
from tools.fidelity.setup_compare import compare  # noqa: E402


def card(i, kind="Attack", rarity="Common", up="1"):
    return {"id": i, "type": kind, "rarity": rarity, "max_upgrade_level": up, "multiplayer_constraint": "None"}


EXPORT = {
    "character_info": {"IRONCLAD": {"card_pool": "IroncladCardPool", "relic_pool": "IroncladRelicPool", "potion_pool": "IroncladPotionPool"},
                       "SILENT": {"card_pool": "SilentCardPool", "relic_pool": "SilentRelicPool", "potion_pool": "SilentPotionPool"}},
    "card_pools": {
        "IroncladCardPool": {"all": [card(f"IC{i}") for i in range(30)] + [card("INFLAME", "Power"), card("RAMPAGE", up="0")]},
        "ColorlessCardPool": {"all": [card("FLASH"), {**card("TANKY"), "multiplayer_constraint": "MultiplayerOnly"}]},
        "SilentCardPool": {"all": [card("NEUTRALIZE")]},
    },
    "relic_pools": {
        "IroncladRelicPool": {"all": [{"id": "RED_SKULL", "rarity": "Common"}, {"id": "BURNING_BLOOD", "rarity": "Starter"}]},
        "SharedRelicPool": {"all": [{"id": f"SR{i}", "rarity": "Common"} for i in range(25)]
                            + [{"id": "POTION_BELT", "rarity": "Common"}, {"id": "ORRERY", "rarity": "Shop"}]},
        "SilentRelicPool": {"all": [{"id": "SNECKO_SKULL", "rarity": "Common"}]},
    },
    "potion_pools": {
        "IroncladPotionPool": {"all": [{"id": "BLOOD_POTION"}]},
        "SharedPotionPool": {"all": [{"id": f"P{i}"} for i in range(6)]},
        "SilentPotionPool": {"all": [{"id": "POISON_POTION"}]},
    },
    "acts": {a: {"weak": [f"{a}_W"], "normal": [f"{a}_N"], "elite": [f"{a}_E"], "boss": [f"{a}_B"]} for a in scenarios.ACTS},
}
INFO = {"relics": {**{f"SR{i}": "None" for i in range(25)}, "RED_SKULL": "None", "POTION_BELT": "None", "ORRERY": "PreRolledCardReward"},
        "buff_keys": ["STRENGTH", "VULNERABLE", "WEAK", "POISON", "INFLAME", "THORNS"], "internal_keys": []}
ID_MAP = {"powers": {"StrengthPower": 1, "VulnerablePower": 2, "WeakPower": 3, "PoisonPower": 4, "InflamePower": 5, "ThornsPower": 6}}


class SpecTests(unittest.TestCase):
    def setUp(self):
        self.content = scenarios.Content(EXPORT, INFO, ID_MAP)

    def test_content_stays_in_the_characters_pools_and_what_the_mirror_builds(self):
        c = self.content
        self.assertNotIn("NEUTRALIZE", c.cards)
        self.assertNotIn("TANKY", c.cards)  # multiplayer only
        self.assertNotIn("ORRERY", c.relics)  # its pickup opens a screen the mirror cannot answer
        self.assertNotIn("BURNING_BLOOD", c.relics)
        self.assertIn("INFLAME_POWER", c.player_powers)  # the Ironclad's own power cards
        self.assertEqual(["POISON_POWER", "STRENGTH_POWER", "VULNERABLE_POWER", "WEAK_POWER"], c.enemy_powers)
        self.assertEqual(16, len(c.encounters))

    def order_ok(self, spec):
        cmds = [it["cmd"] for it in spec["setup"]]
        self.assertEqual(1, cmds.count("fight"))
        f = cmds.index("fight")
        self.assertTrue(all(x in ("relic", "potion", "card") for x in cmds[:f]))
        self.assertTrue(all(x == "power" for x in cmds[f + 1:]))
        self.assertTrue(spec["use_potions"])

    def test_coverage_specs(self):
        rng = random.Random(1)
        for n in range(50):
            s = scenarios.coverage_spec(self.content, {"items": {}}, rng, n)
            self.order_ok(s)
            self.assertIn(len(s["setup"]), range(2, 5))

    def test_dense_specs(self):
        rng = random.Random(2)
        ups = total = 0
        for n in range(40):
            s = scenarios.dense_spec(self.content, {"items": {}}, rng, n)
            self.order_ok(s)
            relics = [it["id"] for it in s["setup"] if it["cmd"] == "relic"]
            cards = [it for it in s["setup"] if it["cmd"] == "card"]
            potions = [it for it in s["setup"] if it["cmd"] == "potion"]
            self.assertTrue(10 <= len(relics) <= 20 and len(set(relics)) == len(relics))
            self.assertTrue(10 <= len(cards) <= 20)
            self.assertEqual(4 if "POTION_BELT" in relics else 2, len(potions))
            self.assertFalse(any(it["upgraded"] for it in cards if it["id"] == "RAMPAGE"))
            ups += sum(it["upgraded"] for it in cards)
            total += len(cards)
        self.assertTrue(0.25 < ups / total < 0.45)

    def test_flags(self):
        c = self.content
        plain = {"setup": [{"cmd": "relic", "id": "SR1"}, {"cmd": "fight", "id": "OVERGROWTH_W"}]}
        game = {"fight": {"act": "OVERGROWTH", "current_act": "OVERGROWTH", "kind": "weak"}}
        self.assertEqual([], scenarios.spec_flags(plain, c, game))
        self.assertEqual(["cross_act", "debug_room"], scenarios.spec_flags(
            plain, c, {"fight": {"act": "HIVE", "current_act": "OVERGROWTH", "kind": "boss"}}))
        silent = {"setup": [{"cmd": "fight", "id": "X"}, {"cmd": "power", "id": "POISON_POWER", "target": "enemy:0"}]}
        self.assertEqual(["cross_character"], scenarios.spec_flags(silent, c, game))
        twice = {"setup": [{"cmd": "relic", "id": "SR1"}, {"cmd": "relic", "id": "SR1"}]}
        self.assertEqual(["duplicate_relic"], scenarios.spec_flags(twice, c, game))

    def test_the_least_covered_items_come_first(self):
        cov = {"items": {f"relic:SR{i}": {"seen": 9, "matched": 9} for i in range(24)}}
        rng = random.Random(3)
        hits = sum("relic:SR24" in scenarios.pick(cov, [f"relic:SR{i}" for i in range(25)], 1, rng) for _ in range(400))
        self.assertGreater(hits, 100)  # 3.0 against 24 x 0.2: about 38%


def log(i, cmd, ok=True, up=5, rewards=2, **kw):
    return {"i": i, "cmd": cmd, "id": kw.get("id"), "ok": ok, "msg": "",
            "rng": {"UpFront": {"seed": 1, "counter": up}}, "player_rng": {"Rewards": {"seed": 2, "counter": rewards}}, **kw}


def elog(i, cmd, ok=True, up=5, rewards=2, **kw):
    return {"i": i, "cmd": cmd, "ok": ok, "msg": "", "rng": {"UpFront": up}, "player_rng": {"Rewards": rewards}, **kw}


class SetupCompareTests(unittest.TestCase):
    SETUP = [{"cmd": "relic", "id": "A"}, {"cmd": "relic", "id": "B"}, {"cmd": "fight", "id": "F"}, {"cmd": "power", "id": "P"}]

    def test_alike(self):
        g = {"setup": self.SETUP, "setup_log": [log(-1, "before"), log(0, "relic"), log(1, "relic"), log(3, "power")]}
        e = {"setup_log": [elog(-1, "before"), elog(0, "relic"), elog(1, "relic"), elog(3, "power", up=99)]}
        self.assertIsNone(compare(g, e)["first"])  # combat items are judged at step 0, not here

    def test_the_first_item_that_moves_a_stream(self):
        g = {"setup": self.SETUP, "setup_log": [log(-1, "before"), log(0, "relic", up=7), log(1, "relic", up=9)]}
        e = {"setup_log": [elog(-1, "before"), elog(0, "relic", up=7), elog(1, "relic", up=8)]}
        first = compare(g, e)["first"]
        self.assertEqual(1, first["i"])
        self.assertEqual({"rng.UpFront": (9, 8)}, first["diff"])

    def test_already_apart_before_the_setup(self):
        g = {"setup": self.SETUP, "setup_log": [log(-1, "before", rewards=3)]}
        e = {"setup_log": [elog(-1, "before", rewards=4)]}
        self.assertEqual(-1, compare(g, e)["first"]["i"])

    def test_refusals_and_mirror_gaps(self):
        g = {"setup": self.SETUP, "setup_log": [log(0, "relic"), log(1, "relic", ok=False)]}
        e = {"setup_log": [elog(0, "relic", unsupported=True, ok=False), elog(1, "relic")]}
        r = compare(g, e)
        self.assertEqual([0], r["unsupported"])
        self.assertEqual(1, r["first"]["i"])


if __name__ == "__main__":
    unittest.main()
