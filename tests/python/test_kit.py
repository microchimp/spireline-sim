"""tools.fidelity.kit and packet --groups: fix kits built from the fix lab's groups, with no game and no .NET build.

The lab, its groups.json and the game records are synthetic (test_packet's World: a corpus of tiny fights, a decompile tree
in the decompile's layout with no game code, a fake EmuDump). The scratch build of the generated tests is replaced by a
fake dotnet that answers restore, build and test the way the real one writes its logs.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import test_packet as tp  # noqa: E402

from tools.fidelity import fight_compare as fc  # noqa: E402
from tools.fidelity import kit as kt  # noqa: E402
from tools.fidelity import packet as pk  # noqa: E402

SETUP = [{"cmd": "power", "id": "HARDENED_SHELL_POWER"}, {"cmd": "fight", "id": "SKULKING_COLONY_ELITE"}]
JOB = "runs/20261015T0100-fights"


def group(**kw) -> dict:
    g = {"id": "g1a2b3c4d5e6", "key": {"event": None, "field": ".enemies", "ingredients": ["power HARDENED_SHELL_POWER"]},
         "title": "Hardened Shell lowered by damage", "count": 3, "first_seen_utc": "2026-10-15T01:00Z",
         "examples": [{"source": "aimed batch", "ref": f"{JOB}/minimal/{tp.SEED_CITED}.out.json", "step": 1, "field": ".enemies",
                       "game": "shell 20", "emu": "shell 14", "first_event": None, "minimal_setup": SETUP}],
         "entities": ["HARDENED_SHELL"], "ingredient_effect": [{"item": "power HARDENED_SHELL_POWER", "removal_vanishes": True},
                                                              {"item": "relic VAJRA", "removal_vanishes": False}],
         "backlog_ids": ["T-01"], "relation": "confirms", "host_suspect": False, "host_reason": "", "impact_runs_est": 900,
         "sources": ["fights"]}
    g.update(kw)
    return g


def lab_with(tmp: Path, groups: list, records: dict | None = None) -> Path:
    lab = tmp / "lab"
    for name, recs in (records or {f"{JOB}/minimal/{tp.SEED_CITED}.out.json": [tp.fight(tp.SEED_CITED, 3, setup=SETUP)]}).items():
        p = lab / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"results": recs}), encoding="utf-8")
    (lab / "groups.json").write_text(json.dumps({"main_sha": "abc", "generated_utc": "2026-10-15T02:00Z", "groups": groups}), encoding="utf-8")
    return lab


class GroupsTest(unittest.TestCase):
    def test_a_group_naming_the_item_wins_over_entity_matches(self):
        doc = {"groups": [group(id="named", backlog_ids=["T-01"], impact_runs_est=5),
                          group(id="ents", backlog_ids=[], entities=["HARDENED_SHELL", "SKULKING_COLONY"])]}
        got = kt.matching_groups({"id": "T-01"}, {"HARDENED_SHELL", "SKULKING_COLONY"}, doc)
        self.assertEqual([(g["id"], how) for g, how in got], [("named", "backlog id")])
        got = kt.matching_groups({"id": "T-02"}, {"HARDENED_SHELL", "SKULKING_COLONY"}, doc)
        self.assertEqual([g["id"] for g, _ in got], ["ents"])

    def test_one_shared_entity_ties_only_a_group_of_one(self):
        doc = {"groups": [group(id="two", backlog_ids=[], entities=["HARDENED_SHELL", "VAJRA"], ingredient_effect=[]),
                          group(id="one", backlog_ids=[], entities=["HARDENED_SHELL"], ingredient_effect=[])]}
        self.assertEqual([g["id"] for g, _ in kt.matching_groups({"id": "X"}, {"HARDENED_SHELL"}, doc)], ["one"])

    def test_culprits_are_the_items_whose_removal_makes_it_vanish(self):
        self.assertEqual(kt.culprits(group()), ["HARDENED_SHELL_POWER"])
        self.assertEqual(kt.setup_items(["relic STONE_CRACKER", {"cmd": "fight", "id": "X"}]), [("relic", "STONE_CRACKER"), ("fight", "X")])

    def test_a_file_that_is_not_a_groups_file_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "g.json"
            p.write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                kt.load_groups(p)


class RecordTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        full = tp.fight(tp.SEED_CITED, 3, setup=[{"cmd": "relic", "id": "VAJRA"}, *SETUP])
        small = tp.fight(tp.SEED_CITED, 3, setup=SETUP)
        self.lab = lab_with(t, [], {f"{JOB}/batch.out.json": [tp.fight(tp.SEED_OTHER, 2), full],
                                    f"{JOB}/batch.shrink3.out.json": [small]})
        (self.lab / JOB / "minimal").mkdir()
        (self.lab / JOB / "minimal" / f"{tp.SEED_CITED}.json").write_text(json.dumps({"runs": [{"seed": tp.SEED_CITED, "setup": SETUP}]}), encoding="utf-8")
        self.ix = kt.RecordIndex(self.lab)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_seed_finds_the_record_with_the_minimal_setup(self):
        rec, where = self.ix.find(tp.SEED_CITED, SETUP)
        self.assertEqual(rec["setup"], SETUP)
        self.assertEqual(where, f"{JOB}/batch.shrink3.out.json#0")

    def test_a_path_with_a_seed_picks_that_fight(self):
        rec, where = self.ix.find(f"{JOB}/batch.out.json#{tp.SEED_CITED}", [])
        self.assertEqual(rec["seed"], tp.SEED_CITED)
        self.assertTrue(where.endswith("#1"))

    def test_a_pack_spec_ref_finds_its_game_record(self):
        rec, _ = self.ix.find(f"{JOB}/minimal/{tp.SEED_CITED}.json", [])
        self.assertEqual(rec["setup"], SETUP)

    def test_an_unknown_ref_says_so(self):
        rec, why = self.ix.find("ZZZZZZZZZZ", [])
        self.assertIsNone(rec)
        self.assertIn("no game record", why)

    def add_emulator_twins(self):
        """EmuDump's outputs beside the batch, as a real lab job holds them: the replays carry the seed but no setup."""
        twins = {"results": [tp.emu_doc_for(tp.fight(tp.SEED_OTHER, 2)), tp.emu_doc_for(tp.fight(tp.SEED_CITED, 3))]}
        for name in ("batch.emu.json", "batch.shrink.emu.json"):
            (self.lab / JOB / name).write_text(json.dumps(twins), encoding="utf-8")
        (self.lab / JOB / "batch.reanchor.json").write_text(json.dumps({"results": twins["results"], "buff_keys": []}), encoding="utf-8")
        return kt.RecordIndex(self.lab)

    def test_an_emulator_twin_of_the_seed_is_never_the_game_record(self):
        ix = self.add_emulator_twins()
        rec, where = ix.find(tp.SEED_CITED, [])  # no setup to filter on: the twins (no setup) used to sort first
        self.assertTrue(kt.is_game_record(rec))
        self.assertEqual(where, f"{JOB}/batch.out.json#1")  # the fight the batch played, not a shrink round's reduced setup
        self.assertTrue(all(not p.name.endswith((".emu.json", ".reanchor.json")) for ps in ix.by_seed.values() for p, _ in ps))
        rec, why = ix.find(f"{JOB}/batch.emu.json#{tp.SEED_CITED}", [])
        self.assertIsNone(rec)
        self.assertIn("no game record", why)

    def test_only_the_games_shape_is_a_game_record(self):
        rec = tp.fight(tp.SEED_CITED, 3, setup=SETUP)
        self.assertTrue(kt.is_game_record(rec))
        self.assertFalse(kt.is_game_record(tp.emu_doc_for(rec)))
        self.assertFalse(kt.is_game_record({"seed": "X", "steps": []}))
        broken = json.loads(json.dumps(rec))
        del broken["steps"][1]["state"]["enemies"]
        self.assertFalse(kt.is_game_record(broken))

    def test_a_seed_only_ref_finds_the_game_record(self):
        ix = self.add_emulator_twins()
        rec, where = ix.find(tp.SEED_OTHER, [])
        self.assertEqual((rec["seed"], where), (tp.SEED_OTHER, f"{JOB}/batch.out.json#0"))
        rec, where = ix.find(tp.SEED_CITED, SETUP)
        self.assertEqual((rec["setup"], where), (SETUP, f"{JOB}/batch.shrink3.out.json#0"))

    def test_a_batch_ref_without_its_extension_reads_the_batch_and_its_shrink_rounds(self):
        ix = self.add_emulator_twins()
        rec, where = ix.find(f"{JOB}/batch#{tp.SEED_CITED}", SETUP)
        self.assertEqual((rec["setup"], where), (SETUP, f"{JOB}/batch.shrink3.out.json#0"))
        rec, where = ix.find(f"{JOB}/batch#{tp.SEED_CITED}", [])
        self.assertEqual(where, f"{JOB}/batch.out.json#1")

    def test_a_tagged_ref_reads_its_base_and_never_another_fight_on_its_seed(self):
        base = Path(self.tmp.name) / "referee"
        other = tp.fight(tp.SEED_CITED, 2, setup=[{"cmd": "fight", "id": "OVICOPTER_NORMAL"}])
        (base / "jobs-setup").mkdir(parents=True)
        (base / "jobs-setup" / "dense60.out.json").write_text(json.dumps({"results": [other]}), encoding="utf-8")
        (base / "jobs-setup" / "dense60.emu.json").write_text(json.dumps({"results": [tp.emu_doc_for(other)]}), encoding="utf-8")
        ix = kt.RecordIndex(self.lab, {"referee/": base})
        rec, where = ix.find(f"referee/jobs-setup/dense60#{tp.SEED_CITED}", [])
        self.assertEqual((rec["setup"], where), (other["setup"], "referee/jobs-setup/dense60.out.json#0"))
        self.assertEqual(kt.where_text(where), "referee/jobs-setup/dense60.out.json#0")
        self.assertEqual(kt.where_text(f"{JOB}/batch.out.json#1"), f"LAB/{JOB}/batch.out.json#1")
        # without that base the ref's fight is unknown: the lab's fight on the same seed is another fight
        rec, _ = self.ix.find(f"referee/jobs-setup/dense60#{tp.SEED_CITED}", [])
        self.assertIsNone(rec)

    def test_a_ref_that_leaves_its_base_is_refused(self):
        outside = Path(self.tmp.name) / "outside.out.json"
        outside.write_text(json.dumps({"results": [tp.fight(tp.SEED_CITED, 3)]}), encoding="utf-8")
        for ref in ("../outside.out.json", f"{JOB}/../../../outside.out.json", outside.as_posix(), str(outside)):
            rec, _ = self.ix.find(ref, [])
            self.assertIsNone(rec, ref)

    def test_the_trimmed_record_carries_actions_and_no_states(self):
        rec = tp.fight(tp.SEED_CITED, 3, setup=SETUP)
        rec["setup_log"] = [{"i": 0, "cmd": "power", "id": "X", "ok": True, "msg": "", "rng": {"big": 1}}]
        rec["map"] = {"huge": True}
        t = kt.trim_record(rec, 1)
        self.assertEqual(len(t["steps"]), 2)
        self.assertTrue(all(set(s) == {"action"} for s in t["steps"]))
        self.assertNotIn("map", t)
        self.assertEqual(t["setup_log"], [{"i": 0, "cmd": "power", "id": "X", "ok": True, "msg": ""}])


class SourceTest(unittest.TestCase):
    def test_the_test_embeds_the_record_and_the_games_views(self):
        rec = tp.fight(tp.SEED_CITED, 3, setup=SETUP)
        rec["choices"] = [{"kind": "cards", "note": 'a """quoted""" text'}]
        src = kt.test_source("Kit_T_01", {"item": "T-01", "title": "t", "group": "g1", "where": "LAB/x", "sha": "abc"}, rec, 1, ".enemies")
        self.assertIn("public class Kit_T_01", src)
        self.assertIn("TheEmulatorReproducesTheGameUpToStep1", src)
        self.assertIn("fails with 'KIT step 1 .enemies:'", src)
        m = re.search(r'const string Record = ("{3,})(.*?)\1;', src)
        self.assertEqual(len(m.group(1)), 3)  # compact JSON escapes its quotes: no run longer than an empty string's
        self.assertEqual(kt._raw('a """" b'), '"""""a """" b"""""')  # a run inside: one quote more than it
        self.assertEqual(json.loads(m.group(2)), kt.trim_record(rec, 1))
        views = re.findall(r'^\s+("{3,})(\{.*\})\1,?$', src, re.MULTILINE)
        self.assertEqual([json.loads(v) for _, v in views], [fc.game_view(s["state"]) for s in rec["steps"][:2]])
        for k in fc.NOT_POWERS:
            self.assertIn(f'"{k}"', src)

    def test_class_names_are_identifiers(self):
        self.assertEqual(kt.cs_name("Kit", "C5-F07", "HM1ZS1TZRZ", "ab.cd"), "Kit_C5_F07_HM1ZS1TZRZ_ab_cd")


TRX = """<?xml version="1.0" encoding="utf-8"?>
<TestRun xmlns="http://microsoft.com/schemas/2006/teamtest/2010"><Results>
<UnitTestResult testName="Kit_A.TheEmulatorReproducesTheGameUpToStep1" outcome="Failed"><Output><ErrorInfo>
<Message>Assert.Fail(): KIT step 1 .enemies: game [1] | emulator [2]</Message></ErrorInfo></Output></UnitTestResult>
<UnitTestResult testName="Kit_C.TheEmulatorReproducesTheGameUpToStep0" outcome="Passed"/>
</Results></TestRun>"""


class VerifyTest(unittest.TestCase):
    def test_a_test_that_does_not_compile_is_dropped_and_the_rest_run(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "bin").mkdir()
            (t / "bin" / "EmuDump.dll").write_bytes(b"")
            calls = []

            def fake_run(cmd, cwd, timeout, log):
                verb = str(cmd[1])
                calls.append(verb)
                with open(log, "a", encoding="utf-8") as f:
                    if verb == "build" and calls.count("build") == 1:
                        f.write(r"C:\x\kit-tests\Kit_B.cs(12,5): error CS1002: ; expected [C:\x\KitTests.csproj]" + "\n")
                        return 1
                if verb == "test":
                    (Path(cmd[cmd.index("--results-directory") + 1]) / "kit-tests.trx").write_text(TRX, encoding="utf-8")
                    return 1
                return 0

            with mock.patch.object(kt, "_run", side_effect=fake_run), mock.patch.object(kt.shutil, "which", return_value="dotnet"):
                res = kt.verify_tests({"Kit_A": "class A {}", "Kit_B": "class B {", "Kit_C": "class C {}"}, t / "bin" / "EmuDump.dll", t / "w")
            self.assertEqual(calls, ["restore", "build", "build", "test"])
            self.assertEqual(res["Kit_B"]["outcome"], "compile_error")
            self.assertIn("CS1002", res["Kit_B"]["message"])
            self.assertTrue(kt.fails_as_stated(res["Kit_A"], 1, ".enemies"))
            self.assertFalse(kt.fails_as_stated(res["Kit_A"], 2, ".enemies"))
            self.assertEqual(res["Kit_C"]["outcome"], "passed")
            self.assertFalse((t / "w" / "kit-tests" / "Kit_B.cs").exists())
            proj = (t / "w" / "kit-tests" / "KitTests.csproj").read_text(encoding="utf-8")
            self.assertIn("<AssemblyName>EmuDump.Tests</AssemblyName>", proj)  # EmuDump's InternalsVisibleTo
            self.assertIn("EmuDump.dll", proj)

    def test_tool_output_loses_the_machines_folders(self):
        s = kt.scrub(r"error in C:\Users\someone\AppData\Local\Temp\kit-x\kit-tests\Kit_B.cs and /home/someone/w/x.log, D:/spireline/fork/a.dll")
        self.assertNotIn("someone", s)
        self.assertNotIn("spireline", s)
        self.assertIn("<Kit_B.cs>", s)
        self.assertEqual(kt.scrub("KIT step 1 .enemies: game [1] | emulator [2]"), "KIT step 1 .enemies: game [1] | emulator [2]")

    def test_no_build_means_no_verdict(self):
        res = kt.verify_tests({"Kit_A": "x"}, Path("nowhere/EmuDump.dll"), Path("w"))
        self.assertEqual(res["Kit_A"]["outcome"], "error")

    def test_the_grades(self):
        self.assertEqual(kt.grade({"minimal repro", "failing test", "rule pointer", "sibling list"})[0], "A")
        g, missing = kt.grade({"minimal repro", "rule pointer"})
        self.assertEqual(g, "B")
        self.assertIn("failing test", missing)
        self.assertEqual(kt.grade({"failing test", "rule pointer", "sibling list"})[0], "C")


SIBLING_DECOMPILE = {  # the decompile's layout, no game code: names only
    "MegaCrit.Sts2.Core.Commands/PlayerCmd.cs": "public static class PlayerCmd\n{\n    public static async Task GainEnergy(int n)\n    {\n    }\n}\n",
    "MegaCrit.Sts2.Core.Rooms/RoomType.cs": "public enum RoomType\n{\n    Monster,\n    Elite\n}\n",
    "MegaCrit.Sts2.Core.Models.Relics/BoomingConch.cs": (
        "public sealed class BoomingConch : AbstractModel\n{\n    public override decimal ModifyHandDraw(decimal n)\n    {\n"
        "        if (room.RoomType == RoomType.Elite) { card.SetCost(1); list.Where(x => x).ToList(); }\n"
        "        PlayerCmd.GainEnergy(1);\n        return n;\n    }\n}\n"),
    "MegaCrit.Sts2.Core.Models.Relics/SlingOfCourage.cs": (
        "public sealed class SlingOfCourage : AbstractModel\n{\n    public override Task AfterRoomEntered()\n    {\n"
        "        if (room.RoomType == RoomType.Elite) other.SetCost(0);\n        return Task.CompletedTask;\n    }\n}\n"),
}


class SiblingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "decompiled"
        for rel_, text in SIBLING_DECOMPILE.items():
            (root / rel_).parent.mkdir(parents=True, exist_ok=True)
            (root / rel_).write_text(text, encoding="utf-8")
        self.decomp = pk.Decompile(root)
        self.names = pk.Names(tp.ID_MAP)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_methods_rule_names_its_commands_enums_and_game_calls_not_plumbing(self):
        p = self.decomp.files["BoomingConch"][0]
        self.assertEqual(kt.body_targets(self.decomp, p, "ModifyHandDraw"), ["RoomType.Elite", "PlayerCmd.GainEnergy", "SetCost"])
        self.assertIn("RoomType.Elite", kt.method_body(self.decomp.text(p), "ModifyHandDraw"))

    def test_siblings_are_the_hook_then_the_names_other_models_share(self):
        got = kt.sibling_targets([("BOOMING_CONCH", "named by the group")], ".draw", None, self.names, self.decomp, {"ModifyHandDraw"})
        # PlayerCmd.GainEnergy has one use only (no sibling to list); RoomType.Elite and SetCost have two each
        self.assertEqual(got, ["ModifyHandDraw", "RoomType.Elite", "SetCost"])
        self.assertEqual(kt.site_count(self.decomp, "SetCost"), 2)
        # the card played at the wrong step is not a suspect: its calls are not swept
        self.assertEqual(kt.sibling_targets([("BOOMING_CONCH", "played at the wrong step")], ".draw", None, self.names, self.decomp, set()), [])


class PointerTest(unittest.TestCase):
    def test_the_culprits_overrides_are_named_with_their_lines(self):
        with tempfile.TemporaryDirectory() as t:
            w = tp.World(Path(t), n_filler=2)
            cx = w.context(Path(t))
            ps = kt.rule_pointers(["HARDENED_SHELL_POWER"], ".enemies", "ModifyHpLostAfterOsty", [], [], cx.names, cx.decomp,
                                  lambda p: pk.rel_decomp(p, cx))
            whats = [p["what"] for p in ps]
            self.assertIn("HardenedShellPower.ModifyHpLostAfterOsty", whats)
            self.assertIn("AbstractModel.ModifyHpLostAfterOsty", whats)
            first = ps[0]
            self.assertRegex(first["where"], r"^Models\.Powers/HardenedShellPower\.cs:\d+$")
            self.assertIn("first differing event", first["why"])

    def test_hook_rows_list_the_models_and_the_hooks_listeners(self):
        check = {"rows": [{"model": "HardenedShellPower", "kind": "power", "hook": "ModifyHpLostAfterOsty", "status": "registered",
                           "where": ["CardEffects.Land:10"], "reason": ""},
                          {"model": "OtherPower", "kind": "power", "hook": "ModifyHpLostAfterOsty", "status": "open", "where": [], "reason": "known gap"}]}
        rows = kt.hook_rows(check, {"HardenedShellPower"}, {"ModifyHpLostAfterOsty"})
        self.assertTrue(rows[0].startswith("  - HardenedShellPower.ModifyHpLostAfterOsty (power): registered at CardEffects.Land:10"))
        self.assertTrue(any("2 listeners in the game (open 1, registered 1)" in r for r in rows))
        self.assertTrue(any("OtherPower (power): open; known gap" in r for r in rows))


if __name__ == "__main__":
    unittest.main()
