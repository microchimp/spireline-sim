"""tools/kernel/port_list.py: the handler port list of a family migration (docs/rebuild/MIGRATION_F08.md)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import hook_table as ht  # noqa: E402
from tools.kernel import port_list as pl  # noqa: E402

FAM_DOC = {"families": {}, "hooks": {
    "ModifyDamageMultiplicative": "F08", "AfterDeath": {"family": "F10"}, "AfterCombatEnd": "F10",
    "AfterCardPlayed": "F06"}}


def model(kind, cls, hooks, corpus=0, agent=0, lab=0, codex=0):
    return {"kind": kind, "class": cls, "combat_hooks": hooks, "corpus_fights": corpus, "agent_runs": agent,
            "lab_fights": lab, "codex_runs": codex}


MODELS = {
    "power:VULNERABLE_POWER": model("power", "VulnerablePower", ["ModifyDamageMultiplicative", "AfterSideTurnEnd"], 50, 0, 900),
    "power:RAVENOUS_POWER": model("power", "RavenousPower", ["AfterDeath"], 10),
    "power:UNREACHED_POWER": model("power", "UnreachedPower", ["AfterDeath"]),
    "power:PLAY_ONLY_POWER": model("power", "PlayOnlyPower", ["AfterCardPlayed"], 99),
    "relic:BURNING_BLOOD": model("relic", "BurningBlood", ["AfterCombatEnd"], 0, 0, 0, 5),
    "relic:NO_ROW": model("relic", "NoRow", ["AfterDeath"], 3),
    "badge:SOME_BADGE": model("badge", "SomeBadge", ["AfterDeath"], 1),
}
ROWS = [
    ht.Row("VulnerablePower", "power", "ModifyDamageMultiplicative", "registered", ["BuffSystem.IncomingDamage:377"]),
    ht.Row("VulnerablePower", "power", "AfterSideTurnEnd", "registered", ["BuffSystem.TickEndOfTurn:9"]),
    ht.Row("RavenousPower", "power", "AfterDeath", "open", [], "", ["CombatEngine.AnswerDeaths"]),
    ht.Row("UnreachedPower", "power", "AfterDeath", "registered", ["X.Y:1"]),
    ht.Row("PlayOnlyPower", "power", "AfterCardPlayed", "registered", ["X.Z:2"]),
    ht.Row("Relics.BurningBlood", "relic", "AfterCombatEnd", "not_modelled", [], "heals after the fight"),
    ht.Row("SomeBadge", "badge", "AfterDeath", "missing"),
]


class PortListTests(unittest.TestCase):
    def setUp(self):
        self.hooks = pl.family_hooks(FAM_DOC, ["F08", "F10"])
        self.doc = pl.build(ROWS, MODELS, self.hooks)
        self.items = {i["model"]: i for i in self.doc["items"]}

    def test_family_hooks_reads_both_entry_shapes(self):
        self.assertEqual(self.hooks, {"ModifyDamageMultiplicative": "F08", "AfterDeath": "F10", "AfterCombatEnd": "F10"})

    def test_only_reached_models_with_a_family_hook_are_listed(self):
        self.assertNotIn("power:UNREACHED_POWER", self.items)  # no source reaches it
        self.assertNotIn("power:PLAY_ONLY_POWER", self.items)  # its hook is another family's
        self.assertIn("relic:BURNING_BLOOD", self.items)       # reached by Codex only still counts

    def test_only_the_family_hooks_of_a_model_are_listed_with_their_old_location(self):
        v = self.items["power:VULNERABLE_POWER"]
        self.assertEqual([h["hook"] for h in v["hooks"]], ["ModifyDamageMultiplicative"])
        self.assertEqual(v["hooks"][0]["where"], ["BuffSystem.IncomingDamage:377"])
        self.assertEqual(v["families"], ["F08"])
        r = self.items["power:RAVENOUS_POWER"]["hooks"][0]
        self.assertEqual((r["status"], r["elsewhere"]), ("open", ["CombatEngine.AnswerDeaths"]))

    def test_a_namespaced_row_matches_its_class_and_keeps_its_reason(self):
        b = self.items["relic:BURNING_BLOOD"]["hooks"][0]
        self.assertEqual((b["status"], b["reason"]), ("not_modelled", "heals after the fight"))

    def test_a_reached_model_without_a_row_is_reported_not_dropped(self):
        self.assertEqual(self.doc["unmatched"], ["relic:NO_ROW"])
        self.assertEqual(self.items["relic:NO_ROW"]["hooks"][0]["status"], "unmatched")

    def test_batches_are_per_kind_by_reach_and_sized(self):
        extra = dict(MODELS)
        for n in range(5):
            extra[f"power:P{n}"] = model("power", f"P{n}", ["AfterDeath"], corpus=n)
        doc = pl.build(ROWS, extra, self.hooks)
        got = pl.batches(doc["items"], 3)
        self.assertNotIn("power:P0", {i["model"] for i in doc["items"]})  # corpus 0 and nothing else: unreached
        self.assertEqual([b["id"] for b in got], ["power-1", "power-2", "relic-1", "other-1"])
        self.assertEqual(got[0]["models"], ["power:VULNERABLE_POWER", "power:RAVENOUS_POWER", "power:P4"])
        self.assertEqual(got[2]["models"], ["relic:NO_ROW", "relic:BURNING_BLOOD"])  # corpus before Codex-only
        self.assertEqual(got[3]["models"], ["badge:SOME_BADGE"])
        self.assertEqual(sum(len(b["models"]) for b in got), len(doc["items"]))
        with self.assertRaises(ValueError):
            pl.batches(doc["items"], 0)

    def test_render_names_every_listed_model_once(self):
        doc = dict(self.doc, families=["F08", "F10"], summary=pl.summary(self.doc["items"]),
                   batches=pl.batches(self.doc["items"], 12))
        md = pl.render_md(doc)
        for key in self.items:
            self.assertEqual(md.count(f"| {key} |"), 1, key)
        self.assertIn("not modelled: heals after the fight", md)
        self.assertIn("Unmatched (manifest model with no decompile row): relic:NO_ROW", md)


if __name__ == "__main__":
    unittest.main()
