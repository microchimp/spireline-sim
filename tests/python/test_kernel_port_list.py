"""tools/kernel/port_list.py: the handler port list of a family migration (docs/rebuild/MIGRATION_F08.md)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
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

    def test_family_hooks_reads_both_entry_shapes_adds_pseudo_hooks_and_excludes(self):
        pseudo = {h: "F10" for h in pl.PSEUDO_HOOKS}
        self.assertEqual(self.hooks, {"ModifyDamageMultiplicative": "F08", "AfterDeath": "F10", "AfterCombatEnd": "F10",
                                      **pseudo})
        self.assertEqual(pl.family_hooks(FAM_DOC, ["F08"]), {"ModifyDamageMultiplicative": "F08"})
        b = pl.family_hooks(FAM_DOC, ["F08", "F10"], ("AfterCombatEnd",))
        self.assertNotIn("AfterCombatEnd", b)
        self.assertIn("AfterDeath", b)

    def test_a_pseudo_hook_row_lists_a_model_without_family_hooks(self):
        models = dict(MODELS, **{"monster:HOPPER": model("monster", "Hopper", ["AfterAttack"], 4)})
        rows = ROWS + [ht.Row("Hopper", "monster", "BeforeRemovedFromRoom", "open", [], "", ["EnemyAI.ExecuteIntent"])]
        doc = pl.build(rows, models, self.hooks)
        h = {i["model"]: i for i in doc["items"]}["monster:HOPPER"]["hooks"]
        self.assertEqual([(x["hook"], x["status"], x["pseudo"]) for x in h], [("BeforeRemovedFromRoom", "open", True)])
        self.assertIn("BeforeRemovedFromRoom (pseudo)", pl.render_md(dict(
            doc, families=["F10"], summary=pl.summary(doc["items"]), batches=pl.batches(doc["items"], 12))))

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


def write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def cs(head: str, *body: str) -> str:
    return head + "\n{\n" + "".join(f"\t{b}\n" for b in body) + "}\n"


class PseudoHookTests(unittest.TestCase):
    def test_pseudo_overrides_by_kind_inherited_and_event_subscriptions(self):
        with tempfile.TemporaryDirectory() as d:
            root, ns = Path(d), ht.MODELS_NS
            write(root, f"{ns}/AbstractModel.cs", cs("public abstract class AbstractModel",
                                                     "public virtual Task AfterDeath(int x)"))
            write(root, f"{ns}.Monsters/Hopper.cs", cs("public sealed class Hopper : MonsterModel",
                                                       "public override void BeforeRemovedFromRoom()", "{", "}"))
            write(root, f"{ns}.Monsters/Beetle.cs", cs("public sealed class Beetle : MonsterModel",
                                                       "public override void AfterAddedToRoom()", "{",
                                                       "\tbase.Creature.Died += AfterDeath;", "}"))
            write(root, f"{ns}.Powers/BaseShrink.cs", cs("public abstract class BaseShrink : PowerModel",
                                                         "public override Task AfterRemoved(Creature oldOwner)", "{", "}"))
            write(root, f"{ns}.Powers/ShrinkPower.cs", cs("public sealed class ShrinkPower : BaseShrink"))
            write(root, f"{ns}.Relics/Charm.cs", cs("public sealed class Charm : RelicModel",
                                                    "public override Task AfterRemoved()", "{", "}"))
            got = pl.pseudo_overrides(str(root))
        # inherited from an abstract base; the relic's AfterRemoved is another virtual (RelicModel), not the power's
        self.assertEqual(sorted(got), [("Beetle", "monster", "Died"), ("Hopper", "monster", "BeforeRemovedFromRoom"),
                                       ("ShrinkPower", "power", "AfterRemoved")])

    def test_a_value_hook_location_that_applies_the_buff_is_moved_to_elsewhere(self):
        texts = {"Core/EnemyAI.cs": cs("public static class EnemyAI", "private static void TriggerSuck(EnemyState e)",
                                       "{", "\tBuffSystem.Apply(e.Buffs, BuffId.Strength, 2);", "}"),
                 "Core/BuffSystem.cs": cs("public static class BuffSystem", "public static int IncomingDamage(int b)",
                                          "{", "\treturn b + Get(buffs, BuffId.Strength);", "}")}
        e = ht.Engine.load(Path("."), texts)
        sites = [[rel, t.index("BuffId.Strength")] for rel, t in sorted(texts.items())]
        where = [f"{ht.qual(e.method_at(r, o))}:{e.line_of(r, o)}" for r, o in sites]
        self.assertEqual(where, ["BuffSystem.IncomingDamage:5", "EnemyAI.TriggerSuck:5"])
        rows = [ht.Row("StrengthPower", "power", "ModifyDamageAdditive", "registered", list(where), "", [], sites),
                ht.Row("SuckPower", "power", "AfterDamageGiven", "registered", where[1:], "", [], sites[1:]),
                ht.Row("OnlyApplied", "power", "ModifyDamageCap", "registered", where[1:], "", [], sites[1:])]
        got = pl.drop_buff_writes(rows, e)
        self.assertEqual((got[0].status, got[0].where, got[0].elsewhere),
                         ("registered", ["BuffSystem.IncomingDamage:5"], ["EnemyAI.TriggerSuck"]))
        self.assertEqual(got[1].where, ["EnemyAI.TriggerSuck:5"])    # an After hook's body may write its own buff
        self.assertEqual((got[2].status, got[2].where), ("open", []))


class GateTests(unittest.TestCase):
    @staticmethod
    def fake_run(doc, code=0):
        """A stand-in for subprocess.run of ``dotnet EmuDump.dll kernel-handlers <out>``: writes doc to the out path."""
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            if code == 0 and doc is not None:
                Path(cmd[-1]).write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, code, "", "boom")
        run.calls = calls
        return run

    def test_handler_pairs_count_a_registered_handler_and_not_a_stub(self):
        doc = {"handlers": [{"kind": "Power", "model": "RavenousPower", "hook": "AfterDeath", "stub": False},
                            {"kind": "Monster", "model": "Hopper", "hook": "BeforeRemovedFromRoom", "stub": True},
                            {"kind": "Power", "model": "VulnerablePower", "hook": "ModifyDamageMultiplicative"}]}
        self.assertEqual(pl.handler_pairs(doc), {("RavenousPower", "AfterDeath"), ("VulnerablePower", "ModifyDamageMultiplicative")})

    def test_ported_is_the_dump_of_the_registry_and_never_the_source_text(self):
        with tempfile.TemporaryDirectory() as d:
            dll = Path(d) / "EmuDump.dll"
            dll.write_text("", encoding="utf-8")
            # a model file whose comment names a hook, and a stub: neither is a handler in the registry's list
            write(Path(d), "Core2/Models/Powers/Ravenous.cs", '// t.Model(ListenerKind.Power, "RavenousPower").On(HookId.AfterDeath, F);')
            run = self.fake_run({"handlers": [{"kind": "Monster", "model": "Hopper", "hook": "AfterDeath", "stub": True}]})
            self.assertEqual(pl.ported(dll, run), set())
            self.assertEqual(run.calls[0][:3], ["dotnet", str(dll), "kernel-handlers"])
            run = self.fake_run({"handlers": [{"kind": "Power", "model": "RavenousPower", "hook": "AfterDeath", "stub": False}]})
            self.assertEqual(pl.ported(dll, run), {("RavenousPower", "AfterDeath")})

    def test_an_unreadable_handler_list_is_an_error_not_an_empty_set(self):
        with tempfile.TemporaryDirectory() as d:
            dll = Path(d) / "EmuDump.dll"
            with self.assertRaises(pl.HandlerListError):
                pl.ported(dll, self.fake_run({"handlers": []}))            # not built
            dll.write_text("", encoding="utf-8")
            for run in (self.fake_run({"handlers": []}, code=1), self.fake_run(None), self.fake_run("not json"),
                        self.fake_run({"rows": []})):
                with self.assertRaises(pl.HandlerListError):
                    pl.ported(dll, run)

            def no_dotnet(cmd, **kw):
                raise FileNotFoundError("dotnet")
            with self.assertRaises(pl.HandlerListError):
                pl.ported(dll, no_dotnet)

    @unittest.skipUnless(pl.DEFAULT_DLL.is_file() and shutil.which("dotnet"), "EmuDump is not built")
    def test_the_built_emudump_lists_the_registry(self):
        got = pl.ported()
        self.assertTrue(all(isinstance(m, str) and isinstance(h, str) for m, h in got))

    def test_unported_pairs_count_every_pair_without_a_handler_but_not_modelled_ones(self):
        doc = pl.build(ROWS, MODELS, pl.family_hooks(FAM_DOC, ["F08", "F10"]))
        got = sorted(pl.unported_pairs(doc["items"]))
        statuses = {(m, h): s for m, h, s in got}
        self.assertEqual(statuses[("badge:SOME_BADGE", "AfterDeath")], "missing")
        self.assertEqual(statuses[("power:RAVENOUS_POWER", "AfterDeath")], "open")
        self.assertEqual(statuses[("relic:NO_ROW", "AfterDeath")], "unmatched")
        self.assertNotIn(("relic:BURNING_BLOOD", "AfterCombatEnd"), statuses)  # not modelled: no handler wanted
        self.assertEqual(len(pl.unported_pairs(doc["items"], {("RavenousPower", "AfterDeath")})), len(got) - 1)

    def test_a_registered_pair_without_a_core2_handler_is_unported_until_its_handler_exists(self):
        rows = [ht.Row("RavenousPower", "power", "AfterDeath", "registered", ["CombatEngine.AnswerDeaths:4402"], "", [])]
        models = {"power:RAVENOUS_POWER": model("power", "RavenousPower", ["AfterDeath"], 9)}
        doc = pl.build(rows, models, pl.family_hooks(FAM_DOC, ["F10"]))
        self.assertEqual(pl.unported_pairs(doc["items"]), [("power:RAVENOUS_POWER", "AfterDeath", "registered")])
        self.assertEqual(pl.unported_pairs(doc["items"], {("RavenousPower", "AfterDeath")}), [])
        self.assertEqual(pl.unported_pairs(doc["items"], {("RavenousPower", "AfterAttack")}),
                         [("power:RAVENOUS_POWER", "AfterDeath", "registered")])

    def test_gate_exit_code(self):
        doc = {"items": [], "unmatched": [], "families": ["F10"], "excluded": [], "summary": {}, "batches": [], "unported": []}
        orig = pl.make
        with tempfile.TemporaryDirectory() as d:
            md = str(Path(d) / "pl.md")
            try:
                pl.make = lambda *a, **k: dict(doc)
                self.assertEqual(pl.main(["--families", "F10", "--gate", "--md", md]), 0)
                pl.make = lambda *a, **k: dict(doc, unported=[["power:X", "AfterDeath", "registered"]])
                self.assertEqual(pl.main(["--families", "F10", "--md", md]), 0)
                self.assertEqual(pl.main(["--families", "F10", "--gate", "--md", md]), 1)

                def unreadable(*a, **k):
                    raise pl.HandlerListError("not built")
                pl.make = unreadable
                self.assertEqual(pl.main(["--families", "F10", "--gate", "--md", md]), 2)
            finally:
                pl.make = orig


if __name__ == "__main__":
    unittest.main()


class GateCountsUnreachedTests(unittest.TestCase):
    """Round R7g: ``--gate`` lists the ``--unreached`` pairs too, so a model no record reaches counts against the gate."""

    def _run(self, argv):
        seen = {}

        def fake_make(families, size, manifest, decompiled, exclude, dll, unreached=False):
            seen["unreached"] = unreached
            return {"families": families, "excluded": [], "items": [], "unmatched": [], "summary": {}, "batches": [],
                    "unported": [["power:X_POWER", "AfterDamageGiven", "registered"]] if unreached else []}

        old = pl.make
        pl.make = fake_make
        try:
            with open(Path(tempfile.gettempdir()) / "port_list_gate_test.md", "w", encoding="utf-8"):
                pass
            rc = pl.main(argv + ["--md", str(Path(tempfile.gettempdir()) / "port_list_gate_test.md")])
        finally:
            pl.make = old
        return rc, seen["unreached"]

    def test_gate_implies_unreached(self):
        self.assertEqual(self._run(["--families", "F08", "F10", "--gate"]), (1, True))

    def test_list_alone_keeps_the_reached_models(self):
        self.assertEqual(self._run(["--families", "F08", "F10"]), (0, False))

    def test_unreached_without_gate_lists_but_passes(self):
        self.assertEqual(self._run(["--families", "F08", "F10", "--unreached"]), (0, True))
