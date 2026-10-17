"""referee/run_repro.py and its uses: a whole-run mismatch as a setup fight, shrunk per loaded item, grouped.

A run record in RunPlay's shape (decisions with a save after each) stands in for a runs.py batch. The fight's
spec must start from the save just before that fight, the shrinker must treat each card, relic and potion of
player_load as its own item (always keeping the load itself), and lab_groups must read them as ingredients.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT))

import run_repro  # noqa: E402
import shrink  # noqa: E402
from tools.fidelity import lab_groups  # noqa: E402


def snap(hp, act=0, potions=()):
    return {"current_act_index": act, "players": [{
        "current_hp": hp, "max_hp": 80, "gold": 99, "max_potion_slot_count": 2,
        "deck": [{"id": "CARD.STRIKE_IRONCLAD", "floor_added_to_deck": 1},
                 {"id": "CARD.BASH", "current_upgrade_level": 1, "floor_added_to_deck": 1,
                  "enchantment": {"id": "ENCHANTMENT.STEADY", "amount": 1}}],
        "relics": [{"id": "RELIC.BURNING_BLOOD", "floor_added_to_deck": 1}, {"id": "RELIC.VAJRA", "floor_added_to_deck": 3}],
        "potions": [{"id": "POTION." + p, "slot_index": i} for i, p in reversed(list(enumerate(potions)))]}]}


def run(act=0):
    return {"seed": "ABCDEFGHJK", "character": "IRONCLAD", "ascension": 10, "chooser_seed": 5, "decisions": [
        {"i": 0, "screen": "start", "snap": snap(80)},
        {"i": 1, "screen": "neow", "choice_index": 2, "snap": snap(80)},
        {"i": 2, "screen": "map", "snap": snap(61, act, ("FIRE_POTION", "BLOCK_POTION"))},
        {"i": 3, "screen": "combat", "floor": 4, "encounter": "FOGMOG_NORMAL", "fight": {"steps": []}, "snap": snap(40, act)}]}


def verdict(component="combat: in fight", decision=3):
    return {"seed": "ABCDEFGHJK", "first": {"decision": decision, "component": component, "floor": 4,
                                             "diffs": [["step 0 after null: .enemies", "a", "b"]]}}


class FightSpec(unittest.TestCase):
    def test_the_spec_starts_from_the_save_before_the_fight(self):
        spec, why = run_repro.fight_spec(run(), verdict())
        self.assertEqual(why, "")
        self.assertEqual(spec["seed"], "ABCDEFGHJK")
        self.assertEqual(spec["neow"], 2)
        load, fight = spec["setup"]
        self.assertEqual(fight, {"cmd": "fight", "id": "FOGMOG_NORMAL"})
        self.assertEqual(load["cmd"], "player_load")
        self.assertEqual((load["hp"], load["max_hp"], load["gold"], load["max_potion_slots"]), (61, 80, 99, 2))
        self.assertEqual(load["potions"], ["FIRE_POTION", "BLOCK_POTION"])  # slot order
        self.assertEqual(load["deck"][1]["enchantment"], {"id": "ENCHANTMENT.STEADY", "amount": 1})  # the save's own shape
        self.assertEqual((spec["run_repro"]["step"], spec["run_repro"]["field"]), (0, ".enemies"))
        self.assertIn("rng_counters", spec["run_repro"]["gaps"])

    def test_a_later_act_enters_its_act_first(self):
        spec, _ = run_repro.fight_spec(run(act=1), verdict())
        self.assertEqual(spec["setup"][0], {"cmd": "act", "index": 2})

    def test_a_divergence_outside_a_fight_is_left_out_with_why(self):
        specs, skipped = run_repro.specs_from({"results": [run()]}, {"runs": [verdict("rewards -> deck"), {"seed": "X", "first": None}]})
        self.assertEqual(specs, [])
        self.assertEqual(skipped, [{"seed": "ABCDEFGHJK", "why": "first divergence not in a fight (rewards -> deck)"}])

    def test_matched_tells_the_runs_divergence_from_another(self):
        spec, _ = run_repro.fight_spec(run(), verdict())
        self.assertEqual(run_repro.matched(spec, {"identical": False, "step": 0, "field": ".enemies"}), "same")
        self.assertEqual(run_repro.matched(spec, {"identical": False, "step": 4, "field": ".enemies"}), "field")
        self.assertEqual(run_repro.matched(spec, {"identical": False, "step": 0, "field": ".hand"}), "other")
        self.assertEqual(run_repro.matched(spec, {"identical": True}), "identical")


class ShrinkLoad(unittest.TestCase):
    def setUp(self):
        self.spec, _ = run_repro.fight_spec(run(), verdict())
        self.case = shrink.Case(self.spec, {"steps": [{"action": None}]}, 0, ".enemies")

    def test_each_loaded_card_relic_and_potion_is_an_item(self):
        self.assertEqual(len(self.case.items), 2 + 2 + 2)
        self.assertTrue(all(h[0] == "load" for h in self.case.items))

    def test_a_subset_keeps_the_load_with_only_its_kept_parts(self):
        keep = [("load", 0, "relics", 1), ("load", 0, "potions", 0)]
        s = self.case.subset_spec(keep)
        load = s["setup"][0]
        self.assertEqual((load["deck"], [r["id"] for r in load["relics"]], load["potions"]), ([], ["RELIC.VAJRA"], ["FIRE_POTION"]))
        self.assertEqual((load["hp"], load["gold"]), (61, 99))
        self.assertEqual(s["setup"][1]["cmd"], "fight")

    def test_the_result_names_the_kept_parts_as_plain_items(self):
        self.case.best = [("load", 0, "deck", 1), ("load", 0, "relics", 1)]
        items = self.case.result()["items"]
        self.assertEqual([(i["cmd"], i["id"], i.get("upgraded")) for i in items], [("card", "BASH", True), ("relic", "VAJRA", None)])


class GroupKeys(unittest.TestCase):
    def test_player_load_is_one_ingredient_per_card_relic_and_potion(self):
        spec, _ = run_repro.fight_spec(run(), verdict())
        keys = [k for it in spec["setup"] for k in lab_groups.item_keys(it)]
        self.assertEqual(keys, ["card:STRIKE_IRONCLAD", "card:BASH+", "relic:BURNING_BLOOD", "relic:VAJRA",
                                "potion:FIRE_POTION", "potion:BLOCK_POTION"])
        self.assertEqual(lab_groups.item_keys({"cmd": "relic", "id": "VAJRA"}), ["relic:VAJRA"])
        self.assertEqual(lab_groups.item_keys({"cmd": "fight", "id": "X"}), [])

    def test_a_run_symptom_names_its_repro_fight(self):
        import json  # noqa: PLC0415
        import shutil  # noqa: PLC0415
        import tempfile  # noqa: PLC0415
        lab = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, lab, True)
        d = lab / "runs" / "J" / "runs"
        d.mkdir(parents=True)
        (d / "w00.runs.json").write_text(json.dumps({"runs": [{**verdict(), "recorded": True}]}), encoding="utf-8")
        sym = lab_groups.run_symptoms(d / "w00.runs.json", lab, "t")
        self.assertIsNone(sym[0]["repro"])
        (d / "w00-repro.repro.json").write_text(json.dumps({"runs": [{"seed": "ABCDEFGHJK", "repro": "same"}]}), encoding="utf-8")
        sym = lab_groups.run_symptoms(d / "w00.runs.json", lab, "t")
        self.assertEqual(sym[0]["repro"], {"ref": "runs/J/runs/w00-repro#ABCDEFGHJK", "match": "same"})


if __name__ == "__main__":
    unittest.main()
