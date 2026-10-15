"""tools.fidelity.packet --groups: the fix kit heads the packet, its test is kept only when it fails as stated, grades.json.

The same synthetic world as test_packet and test_kit (a corpus of tiny fights, a decompile tree with no game code, a fake
EmuDump); the scratch build of the generated tests is replaced by a fake verdict, so nothing here needs .NET or the game.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import test_packet as tp  # noqa: E402
from test_kit import JOB, SETUP, group, lab_with  # noqa: E402

from tools.fidelity import kit as kt  # noqa: E402
from tools.fidelity import packet as pk  # noqa: E402


class KitPacketTest(unittest.TestCase):
    """packet.write_packets with a kit config: the kit heads the packet, its test is kept only when it fails as stated."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.w = tp.World(self.t, n_filler=3)
        dec = self.w.root / "decompiled"  # the shell's rule runs through a command one more model uses: a sibling to sweep
        (dec / "MegaCrit.Sts2.Core.Models.Powers/HardenedShellPower.cs").write_text(
            "public sealed class HardenedShellPower : AbstractModel\n{\n    public override decimal ModifyHpLostAfterOsty(decimal amount)\n    {\n"
            "        CreatureCmd.Damage(1);\n        return amount;\n    }\n}\n", encoding="utf-8")
        (dec / "MegaCrit.Sts2.Core.Models.Relics").mkdir(parents=True)
        (dec / "MegaCrit.Sts2.Core.Models.Relics/TheBoot.cs").write_text(
            "public sealed class TheBoot : AbstractModel\n{\n    public override Task AfterAttack()\n    {\n        CreatureCmd.Damage(5);\n    }\n}\n",
            encoding="utf-8")
        self.lab = lab_with(self.t, [group()])
        self.out = self.t / "packets"
        self.check = {"rows": [{"model": "HardenedShellPower", "kind": "power", "hook": "ModifyHpLostAfterOsty", "status": "registered",
                                "where": ["CardEffects.LandDamageOnEnemy:3"], "reason": ""}]}

    def tearDown(self):
        self.tmp.cleanup()

    def run_kit(self, outcome="failed", message="Assert.Fail(): KIT step 1 .enemies: game x | emulator y"):
        cx = self.w.context(self.t, emu=tp.FakeEmu(), do_sweep=True)
        cx.kit_cfg = {"groups": kt.load_groups(self.lab / "groups.json"), "lab": self.lab, "verify": True}
        seen = {}

        def fake_verify(sources, dll, work):
            seen.update(sources)
            return {c: {"outcome": outcome, "message": message} for c in sources}

        with mock.patch.object(pk.kt, "verify_tests", side_effect=fake_verify), mock.patch.object(pk.kt, "load_hook_check", return_value=self.check):
            res = pk.write_packets([tp.ITEM], cx, self.out, quiet=True)
        return res[0], seen

    def test_a_verified_failing_test_makes_grade_a(self):
        pk_, seen = self.run_kit()
        self.assertEqual(pk_["kit"]["grade"], "A")
        text = (self.out / "T-01.md").read_text(encoding="utf-8")
        self.assertLess(text.index("## Fix kit [grade A]"), text.index("## Completeness"))
        self.assertIn("first wrong at step 1 (after STRIKE_IRONCLAD), .enemies: game", text)
        self.assertIn("which ingredient causes it: power HARDENED_SHELL_POWER; removing any of relic VAJRA leaves it", text)
        self.assertIn("HardenedShellPower.ModifyHpLostAfterOsty at Models.Powers/HardenedShellPower.cs:", text)
        self.assertIn("sweep `ModifyHpLostAfterOsty`: 1 classes that override it", text)
        self.assertIn("sweep `CreatureCmd.Damage`: 2 game sites", text)
        self.assertIn("TheBoot.AfterAttack", text)
        (cls,) = seen
        f = self.out / "T-01" / "kit-test" / f"{cls}.cs"
        self.assertTrue(f.is_file())
        self.assertIn("KIT step 1 .enemies:", f.read_text(encoding="utf-8"))
        grades = json.loads((self.out / "grades.json").read_text(encoding="utf-8"))
        self.assertEqual(grades["T-01"]["grade"], "A")
        self.assertEqual(grades["T-01"]["main_sha"], pk.git("rev-parse", "HEAD"))
        self.assertNotIn("failing test", grades["T-01"]["missing"])

    def test_a_test_that_passes_on_main_is_not_kept(self):
        pk_, _ = self.run_kit(outcome="passed", message="")
        self.assertEqual(pk_["kit"]["grade"], "B")
        self.assertIn("failing test", pk_["kit"]["missing"])
        self.assertFalse((self.out / "T-01" / "kit-test").exists())
        self.assertIn("failing test: not kept, passed", (self.out / "T-01.md").read_text(encoding="utf-8"))

    def test_a_failure_for_another_reason_is_not_kept(self):
        pk_, _ = self.run_kit(message="Assert.Fail(): KIT step 0 .hand: game x | emulator y")
        self.assertEqual(pk_["kit"]["grade"], "B")

    def test_a_repro_the_emulator_now_matches_is_no_repro(self):
        self.lab = lab_with(self.t, [group()], {f"{JOB}/minimal/{tp.SEED_CITED}.out.json": [tp.fight(tp.SEED_CITED, 3, shell_after=14, setup=SETUP)]})
        pk_, seen = self.run_kit()
        self.assertEqual(seen, {})
        self.assertEqual(pk_["kit"]["grade"], "C")
        self.assertIn("MATCHES the game on this checkout", (self.out / "T-01.md").read_text(encoding="utf-8"))

    def test_without_groups_the_packet_is_unchanged(self):
        cx = self.w.context(self.t, emu=tp.FakeEmu())
        res = pk.write_packets([tp.ITEM], cx, self.out, quiet=True)
        self.assertNotIn("kit", res[0])
        self.assertNotIn("Fix kit", (self.out / "T-01.md").read_text(encoding="utf-8"))
        self.assertFalse((self.out / "grades.json").exists())

    def test_a_seed_ref_with_an_emulator_twin_and_no_minimal_setup_still_finds_the_game_record(self):
        rec = tp.fight(tp.SEED_CITED, 3, setup=SETUP)
        ex = dict(group()["examples"][0], ref=tp.SEED_CITED, minimal_setup=[])
        self.lab = lab_with(self.t, [group(examples=[ex])], {f"{JOB}/batch.out.json": [rec],
                                                              f"{JOB}/batch.emu.json": [tp.emu_doc_for(rec)],
                                                              f"{JOB}/batch.shrink.emu.json": [tp.emu_doc_for(rec)]})
        pk_, seen = self.run_kit()
        self.assertEqual(pk_["kit"]["grade"], "A")
        self.assertNotIn("errors", pk_["kit"])
        self.assertIn(f"record LAB/{JOB}/batch.out.json#0; on this checkout first wrong at step 1", (self.out / "T-01.md").read_text(encoding="utf-8"))

    def run_two(self, **patches):
        """Kits for T-01 and T-02 (one group names both), with kit functions patched to fail; every generated test fails as stated."""
        self.lab = lab_with(self.t, [group(backlog_ids=["T-01", "T-02"])])
        cx = self.w.context(self.t, emu=tp.FakeEmu(), do_sweep=True)
        cx.kit_cfg = {"groups": kt.load_groups(self.lab / "groups.json"), "lab": self.lab, "verify": True}

        def fake_verify(sources, dll, work):
            return {c: {"outcome": "failed", "message": "Assert.Fail(): KIT step 1 .enemies: game x | emulator y"} for c in sources}
        with mock.patch.object(pk.kt, "verify_tests", side_effect=fake_verify), \
                mock.patch.object(pk.kt, "load_hook_check", return_value=self.check), mock.patch.multiple(pk.kt, **patches):
            res = pk.write_packets([tp.ITEM, dict(tp.ITEM, id="T-02")], cx, self.out, quiet=True)
        return {r["item"]["id"]: r for r in res}, json.loads((self.out / "grades.json").read_text(encoding="utf-8"))

    def failing_once(self, name, exc):
        real, calls = getattr(kt, name), []

        def f(*a, **kw):
            calls.append(1)
            if len(calls) == 1:
                raise exc
            return real(*a, **kw)
        return f

    def test_a_test_that_cannot_be_generated_costs_that_test_only(self):
        res, grades = self.run_two(test_source=self.failing_once("test_source", KeyError("player")))
        self.assertEqual((res["T-01"]["kit"]["grade"], res["T-02"]["kit"]["grade"]), ("B", "A"))
        self.assertIn("KeyError: 'player'", grades["T-01"]["errors"][0])
        self.assertNotIn("errors", grades["T-02"])
        text = (self.out / "T-01.md").read_text(encoding="utf-8")
        self.assertIn("failing test: not generated, the test could not be generated from LAB/", text)
        self.assertIn("kit parts that failed", text)

    def test_a_group_that_fails_degrades_its_kit_and_the_run_goes_on(self):
        res, grades = self.run_two(sibling_targets=self.failing_once("sibling_targets", RuntimeError(f"bad file {self.t / 'x.cs'}")))
        self.assertEqual((res["T-01"]["kit"]["grade"], res["T-02"]["kit"]["grade"]), ("C", "A"))
        (err,) = grades["T-01"]["errors"]
        self.assertTrue(err.startswith("group g1a2b3c4d5e6: RuntimeError: bad file <x.cs>"), err)  # no machine folders
        self.assertIn("this group's kit part failed: RuntimeError", (self.out / "T-01.md").read_text(encoding="utf-8"))
        self.assertTrue((self.out / "T-02.md").is_file())

    def test_a_kit_that_cannot_be_built_is_grade_c_with_its_reason(self):
        res, grades = self.run_two(rule_pointers=self.failing_once("rule_pointers", ValueError("odd")))
        self.assertEqual((res["T-01"]["kit"]["grade"], res["T-02"]["kit"]["grade"]), ("C", "A"))
        self.assertEqual(grades["T-01"]["errors"], ["the item's kit: ValueError: odd"])
        self.assertNotIn("no lab group names this item", (self.out / "T-01.md").read_text(encoding="utf-8"))

    def test_a_record_the_comparison_cannot_read_is_no_repro(self):
        rec = tp.fight(tp.SEED_CITED, 3, setup=SETUP)
        del rec["steps"][1]["state"]["player"]["hp"]  # the game's shape at the top, a field the comparison needs missing
        self.lab = lab_with(self.t, [group()], {f"{JOB}/minimal/{tp.SEED_CITED}.out.json": [rec]})
        pk_, seen = self.run_kit()
        self.assertEqual((pk_["kit"]["grade"], seen), ("C", {}))
        self.assertIn("not comparable (KeyError: 'hp')", (self.out / "T-01.md").read_text(encoding="utf-8"))

    def test_an_item_no_group_matches_gets_grade_c(self):
        self.lab = lab_with(self.t, [group(backlog_ids=["OTHER"], entities=["VAJRA"], ingredient_effect=[])])
        pk_, _ = self.run_kit()
        self.assertEqual(pk_["kit"]["grade"], "C")
        self.assertIn("no lab group names this item", (self.out / "T-01.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
