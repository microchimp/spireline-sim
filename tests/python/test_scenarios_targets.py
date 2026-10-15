"""referee/scenarios.py --mode targets: realistic specs aimed at the fix lab's weighted targets.

The small export of test_scenarios stands in for the game's tables and a hand-made profile for the Codex
aggregate counts. A spec must aim at a target the setup can reach, keep the item order the referee and
EmuDump rely on (act, deck, map items, the fight, combat items), stay in the character's pools and in the
act the profile says the target lives in, and hold each card, relic and potion as its own item.
"""

from __future__ import annotations

import collections
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scenarios  # noqa: E402
from test_scenarios import EXPORT, ID_MAP, INFO  # noqa: E402

PROFILE = {
    "runs": 1000,
    "meet": {"card": {"IC1": 300, "IC2": 200}, "relic": {"RED_SKULL": 100}, "encounter": {"HIVE_N": 50}},
    "acts": {
        "1": {"snapshots": 100, "rooms": {"monster": 90, "elite": 10}, "act_names": {"OVERGROWTH": 60, "UNDERDOCKS": 40},
              "encounters": {"monster": {"OVERGROWTH_N": 50, "UNDERDOCKS_N": 30, "OVERGROWTH_W": 10}, "elite": {"OVERGROWTH_E": 10}},
              "cards": {"IC1": 30, "IC2": 10, "IC3": 50}, "copies": {"IC1": 30, "IC2": 10, "IC3": 100}, "upgraded": {"IC3": 50},
              "relics": {"SR1": 20, "RED_SKULL": 5}, "potions": {"P1": 30},
              "deck_size": {"14": 100}, "relic_count": {"2": 100}, "potion_count": {"1": 100},
              "basics": {"STRIKE_IRONCLAD": {"4": 100}, "DEFEND_IRONCLAD": {"4": 100}, "BASH": {"1": 100}, "ASCENDERS_BANE": {"1": 100}},
              "pairs": {}},
        "2": {"snapshots": 50, "rooms": {"monster": 40, "boss": 10}, "act_names": {"HIVE": 50},
              "encounters": {"monster": {"HIVE_N": 30}, "boss": {"HIVE_B": 10}},
              "cards": {"IC2": 40, "IC4": 30}, "copies": {"IC2": 40, "IC4": 30}, "relics": {"SR2": 30}, "potions": {},
              "deck_size": {"20": 50}, "relic_count": {"3": 50}, "potion_count": {"0": 50},
              "basics": {"STRIKE_IRONCLAD": {"2": 50}, "DEFEND_IRONCLAD": {"3": 50}, "BASH": {"1": 50}, "ASCENDERS_BANE": {"1": 50}},
              "pairs": {"card:IC2|card:IC4": 30}},
    },
    "encounter_monsters": {"HIVE_N": ["HIVE_MONSTER"]},
}
TARGETS = {"profile": "cache/codex_profile.json", "targets": [
    {"kind": "card", "id": "IC2", "weight": 1.0, "reasons": []},
    {"kind": "relic", "id": "RED_SKULL", "weight": 0.5, "reasons": []},
    {"kind": "enemy", "id": "HIVE_MONSTER", "weight": 0.5, "reasons": []},
    {"kind": "relic", "id": "ORRERY", "weight": 9.0, "reasons": []},       # its pickup opens a screen the mirror cannot answer
    {"kind": "card", "id": "NEUTRALIZE", "weight": 9.0, "reasons": []},    # another character's
    {"kind": "event", "id": "NEOW", "weight": 9.0, "reasons": []},         # not a fight
]}


class AimTests(unittest.TestCase):
    def setUp(self):
        self.content = scenarios.Content(EXPORT, INFO, ID_MAP)
        self.aim = scenarios.Aim(TARGETS, PROFILE, self.content, {"KNOWNOVER1": "OVERGROWTH", "KNOWNUNDR1": "UNDERDOCKS"})

    def specs(self, n=120, seed=3):
        rng = random.Random(seed)
        return [scenarios.aimed_spec(self.aim, rng, i) for i in range(n)]

    def test_only_reachable_targets_are_drawn(self):
        self.assertEqual({"card:IC2", "relic:RED_SKULL", "enemy:HIVE_MONSTER"}, {f"{k}:{i}" for k, i, _ in self.aim.targets})
        self.assertEqual({"relic": 1, "card": 1, "event": 1}, dict(self.aim.skipped))
        got = collections.Counter(m["target"] for _, m in self.specs())
        self.assertGreater(got["card:IC2"], got["relic:RED_SKULL"])  # drawn by weight

    def test_item_order_and_pools(self):
        for spec, _ in self.specs():
            cmds = [it["cmd"] for it in spec["setup"]]
            self.assertEqual(1, cmds.count("fight"))
            f = cmds.index("fight")
            head = cmds[:f]
            if head[0] == "act":
                head = head[1:]
            self.assertEqual("deck_set", head[0])
            self.assertTrue(all(c in ("card", "relic", "potion") for c in head[1:]))
            self.assertTrue(all(c == "power" for c in cmds[f + 1:]))
            for it in spec["setup"]:
                if it["cmd"] == "card":
                    self.assertIn(it["id"], self.content.cards)
                    self.assertNotIn(it["id"], scenarios.BASICS)
                elif it["cmd"] == "relic":
                    self.assertIn(it["id"], self.content.relics)
                elif it["cmd"] == "potion":
                    self.assertIn(it["id"], self.content.potions)
            self.assertEqual([], scenarios.spec_flags(spec, self.content))  # no cross-character or duplicate item

    def test_the_target_is_in_the_spec_and_in_its_act(self):
        for spec, m in self.specs():
            setup = spec["setup"]
            enc = next(it["id"] for it in setup if it["cmd"] == "fight")
            act_name = self.aim.enc[enc][0]
            self.assertEqual(m["act"], scenarios.ACT_NUMBER[act_name])
            self.assertEqual(m["act"] > 1, setup[0] == {"cmd": "act", "index": m["act"]})
            kind, ident = m["target"].split(":")
            if kind == "card":
                self.assertIn(ident, [it["id"] for it in setup if it["cmd"] == "card"])
                self.assertIn(m["act"], (1, 2))  # IC2 is held in acts 1 and 2 only
            elif kind == "relic":
                self.assertIn(ident, [it["id"] for it in setup if it["cmd"] == "relic"])
                self.assertEqual(1, m["act"])
            elif kind == "enemy":
                self.assertEqual("HIVE_N", enc)  # the only encounter holding it

    def test_decks_follow_the_acts_shape(self):
        for spec, m in self.specs():
            deck = next(it["cards"] for it in spec["setup"] if it["cmd"] == "deck_set")
            basics = collections.Counter(c["id"] for c in deck)
            cards = [it for it in spec["setup"] if it["cmd"] == "card"]
            if m["act"] == 1:
                self.assertEqual({"STRIKE_IRONCLAD": 4, "DEFEND_IRONCLAD": 4, "BASH": 1, "ASCENDERS_BANE": 1}, dict(basics))
                self.assertIn(len(deck) + len(cards), (14, 15))  # the act's deck size; a second copy may add one
                self.assertTrue(all(it["id"] in ("IC1", "IC2", "IC3") for it in cards))  # held at act 1
            else:
                self.assertEqual(2, basics["STRIKE_IRONCLAD"])
                self.assertTrue(all(it["id"] in ("IC2", "IC4") for it in cards))

    def big_cache(self):
        cache = {f"KNOWNOVR{i:02d}": "OVERGROWTH" for i in range(80)} | {f"KNOWNUND{i:02d}": "UNDERDOCKS" for i in range(80)}
        return scenarios.Aim(TARGETS, PROFILE, self.content, cache), cache

    def test_act_one_seeds_come_from_the_cache_mostly(self):
        self.aim, cache = self.big_cache()
        specs = self.specs(200)
        known = sum(1 for spec, m in specs if m["act"] == 1 and spec["seed"] in cache)
        act1 = sum(1 for _, m in specs if m["act"] == 1)
        self.assertGreater(known, act1 // 2)
        for spec, m in specs:
            if spec["seed"] in cache:
                self.assertEqual(cache[spec["seed"]], self.aim.enc[m["encounter"]][0])
            self.assertEqual("cache" if spec["seed"] in cache else "fresh", m["seed_from"])

    def test_no_seed_twice_in_a_batch(self):
        # shrink's packs, traces and lab_groups key a batch's outputs by seed: a seed on two specs mixes their fights
        specs = self.specs(300)  # 2 cached seeds, many more act-1 specs
        seeds = [s["seed"] for s, _ in specs]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(2, sum(1 for _, m in specs if m["seed_from"] == "cache"))
        self.aim, cache = self.big_cache()
        seeds = [s["seed"] for s, _ in self.specs(300)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertGreater(len(set(seeds) & set(cache)), 60)
        clash = scenarios._seed(random.Random(1))  # a fresh seed that clashes with one already used is redrawn
        self.aim.used.add(clash)
        got = self.aim.seed(random.Random(1), 2, "HIVE")
        self.assertNotEqual(clash, got)
        self.assertIn(got, self.aim.used)

    def test_deterministic(self):
        a = [json.dumps(s, sort_keys=True) for s, _ in self.specs(30, 9)]
        self.setUp()  # one Aim is one batch (it holds the batch's used seeds)
        b = [json.dumps(s, sort_keys=True) for s, _ in self.specs(30, 9)]
        self.assertEqual(a, b)

    def test_act1_seed_cache(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "a.out.json"
            out.write_text(json.dumps({"results": [{"seed": "S1", "fight": {"current_act": "UNDERDOCKS"}},
                                                   {"seed": "S2", "fight": {"current_act": "HIVE"}}, None]}), encoding="utf-8")
            path = Path(td) / "cache" / "act1.json"
            self.assertEqual({"S1": "UNDERDOCKS"}, scenarios.act1_seed_cache(path, scan=[out]))
            got = scenarios.act1_seed_cache(path, [{"seed": "S3", "fight": {"current_act": "OVERGROWTH"}}])
            self.assertEqual({"S1": "UNDERDOCKS", "S3": "OVERGROWTH"}, got)


if __name__ == "__main__":
    unittest.main()
