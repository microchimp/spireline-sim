"""tools/fidelity/fuzz_core2.py: seeded generation, the per-step comparison on an injected difference (fake engine
outputs, no EmuDump), the triage file and the exit codes. One smoke test runs the real EmuDump when it is built."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import fuzz_core2 as F  # noqa: E402


def catalog(switches=()) -> F.Catalog:
    cards = [{"id": i, "upgradable": True, "families": f} for i, f in (
        ("STRIKE_IRONCLAD", []), ("DEFEND_IRONCLAD", []), ("BASH", []), ("ASCENDERS_BANE", []), ("ANGER", []), ("INFLAME", ["F08"]),
        ("RAMPAGE", ["F08", "F09"]), ("HEADBUTT", ["F10"]), ("WHIRLWIND", []))]
    relics = [{"id": i, "families": f} for i, f in (("BURNING_BLOOD", ["F10"]), ("VAJRA", []), ("ANCHOR", ["F01"]), ("POTION_BELT", []))]
    potions = [{"id": i, "families": f, "target": "Self"} for i, f in (("FIRE_POTION", []), ("BLOCK_POTION", ["F09"]))]
    encounters = [
        {"id": "NibbitsWeak", "act": "OVERGROWTH", "kind": "weak", "index": 1, "monsters": ["NIBBIT"], "families": []},
        {"id": "Mytes", "act": "HIVE", "kind": "normal", "index": 5, "monsters": ["MYTE"], "families": ["F02"]},
        {"id": "KaiserCrab", "act": "HIVE", "kind": "boss", "index": 0, "monsters": ["KAISER_CRAB"], "families": ["F08"]},
    ]
    return F.Catalog(cards, relics, potions, encounters, tuple(sorted(switches)))


def result(spec: dict, bend=None, error=None) -> dict:
    """What EmuDump fuzz writes for a spec, made up: a hash chain from the spec's seed; `bend(step, hash)` can change one."""
    n = 3 + spec["fuzz"]["seed"] % 4
    steps = []
    for i in range(n):
        h = hashlib.sha256(f"{spec['seed']}:{i}".encode()).hexdigest()[:16]
        steps.append({"action": None if i == 0 else {"kind": "play", "hand_index": i % 3, "target_slot": 0},
                      "state_hash": bend(i, h) if bend else h})
    out = {"steps": steps, "choices": [{"kind": "cards", "step": 1, "max": 1, "options": ["BASH"], "picked": [0]}], "room": [3, 1]}
    if error:
        out["error"] = error
    return out


def options(out: str, **kw) -> F.Options:
    return F.Options(dll="none", seed=kw.pop("seed", 7), fights=kw.pop("fights", 8), batch=kw.pop("batch", 4), out=out,
                     explained=kw.pop("explained", str(Path(out) / "none.json")), **kw)


class Generation(unittest.TestCase):
    def test_seeded_generation_is_deterministic(self):
        cat = catalog()
        a = [F.generate(cat, 5, i) for i in range(12)]
        again = [F.generate(cat, 5, i) for i in reversed(range(12))][::-1]  # the order of calls does not matter
        self.assertEqual(a, again)
        self.assertNotEqual(a[0], a[1])
        self.assertNotEqual(F.generate(cat, 5, 0), F.generate(cat, 6, 0))
        self.assertEqual(json.loads(json.dumps(a[3])), a[3])  # plain data
        self.assertEqual(len({s["seed"] for s in a}), 12)

    def test_a_record_has_the_setup_items_the_mirror_reads(self):
        for i in range(20):
            spec = F.generate(catalog(), 1, i)
            cmds = [it["cmd"] for it in spec["setup"]]
            self.assertEqual(cmds[-1], "fight")
            self.assertEqual(spec["setup"][-1]["id"], spec["fight"]["id"])
            self.assertEqual(sorted(cmds[cmds[0] == "act":-1]), sorted(["relics_set", "deck_set", "potions_set", "hp"]))
            self.assertNotIn("room", spec)  # the engine takes the first map option
            self.assertEqual(cmds[0] == "act", spec["fight"]["act"] in ("HIVE", "GLORY"))
            self.assertTrue(spec["fuzz"]["seed"] >= 0)

    def test_a_switch_set_draws_only_models_whose_families_are_on(self):
        cat = catalog(("F08",))
        allowed_cards = {c["id"] for c in cat.cards if set(c["families"]) <= {"F08"}}
        for i in range(60):
            spec = F.generate(cat, 2, i)
            deck = next(it for it in spec["setup"] if it["cmd"] == "deck_set")["cards"]
            self.assertLessEqual({c["id"] for c in deck}, allowed_cards)
            relics = next(it for it in spec["setup"] if it["cmd"] == "relics_set")["ids"]
            self.assertLessEqual(set(relics), {"VAJRA", "POTION_BELT"})
            self.assertLessEqual(set(next(it for it in spec["setup"] if it["cmd"] == "potions_set")["ids"]), {"FIRE_POTION"})
            self.assertIn(spec["fight"]["id"], ("NibbitsWeak", "KaiserCrab"))  # Mytes needs F02
        self.assertEqual(cat.coverage()["cards"], {"placeable": 7, "known": 9})

    def test_no_switch_draws_every_model(self):
        cat = catalog(())
        self.assertTrue(cat.unrestricted)
        seen = set()
        for i in range(200):
            spec = F.generate(cat, 3, i)
            seen |= {c["id"] for c in next(it for it in spec["setup"] if it["cmd"] == "deck_set")["cards"]}
            seen.add(spec["fight"]["id"])
        self.assertTrue({"INFLAME", "RAMPAGE", "HEADBUTT", "Mytes", "KaiserCrab"} <= seen)

    def test_a_switch_set_with_no_encounter_is_refused(self):
        cat = catalog(("F05",))
        cat.encounters = [e for e in cat.encounters if e["families"]]
        with self.assertRaises(ValueError):
            F.generate(cat, 1, 0)

    def test_parse_switches(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        (d / "m.json").write_text('{"migrated": ["F10", "F08"]}')
        self.assertEqual(F.parse_switches(None, d / "m.json"), ["F08", "F10"])
        self.assertEqual(F.parse_switches("none", d / "m.json"), [])
        self.assertEqual(F.parse_switches("f10, F08,F10"), ["F08", "F10"])
        self.assertEqual(F.parse_switches("all"), [f"F{n:02d}" for n in range(1, 17)])
        with self.assertRaises(ValueError):
            F.parse_switches("F08,F99")

    def test_a_scratch_build_carries_the_test_set(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        (d / "bin").mkdir()
        (d / "bin" / "EmuDump.dll").write_bytes(b"x")
        (d / "bin" / "migrated_families.json").write_text('{"migrated": []}')
        self.assertEqual(F.dll_for(d / "bin" / "EmuDump.dll", [], d / "w"), d / "bin" / "EmuDump.dll")  # the file's own set
        dll = F.dll_for(d / "bin" / "EmuDump.dll", ["F08", "F10"], d / "w")
        self.assertNotEqual(dll, d / "bin" / "EmuDump.dll")
        self.assertEqual(json.loads((dll.parent / "migrated_families.json").read_text()), {"migrated": ["F08", "F10"]})
        self.assertEqual(json.loads((d / "bin" / "migrated_families.json").read_text()), {"migrated": []})


class Comparison(unittest.TestCase):
    spec = {"seed": "ABC", "fuzz": {"seed": 4}}

    def test_identical_results_have_no_difference(self):
        self.assertIsNone(F.first_difference(result(self.spec), result(self.spec)))

    def test_an_injected_hash_difference_is_found_at_its_step(self):
        bent = result(self.spec, bend=lambda i, h: "0" * 16 if i == 2 else h)
        d = F.first_difference(result(self.spec), bent)
        self.assertEqual((d["step"], d["kind"]), (2, "state"))
        # the first of two differences is the one reported
        both = result(self.spec, bend=lambda i, h: "f" * 16 if i >= 1 else h)
        self.assertEqual(F.first_difference(result(self.spec), both)["step"], 1)

    def test_a_different_action_is_reported_before_its_hash(self):
        other = result(self.spec)
        other["steps"][2]["action"] = {"kind": "end_turn"}
        other["steps"][2]["state_hash"] = "1" * 16
        d = F.first_difference(result(self.spec), other)
        self.assertEqual((d["step"], d["kind"]), (2, "action"))

    def test_errors_and_lengths(self):
        a = result(self.spec)
        self.assertEqual(F.first_difference(a, result(self.spec, error="Boom: x\n at y"))["kind"], "error")
        # the same exception with other stack lines is no difference
        self.assertIsNone(F.first_difference(result(self.spec, error="Boom: x\n at y"), result(self.spec, error="Boom: x\n at z")))
        self.assertEqual(F.first_difference(result(self.spec, error="Boom: x"), result(self.spec, error="Boom: y"))["kind"], "error")
        short = copy.deepcopy(a)
        short["steps"].pop()
        d = F.first_difference(a, short)
        self.assertEqual((d["kind"], d["step"]), ("length", len(short["steps"])))

    def test_the_signature_names_the_differing_fields(self):
        self.assertEqual(F.differing_fields({"state_fields": {"a": "1", "b": "2", "c": "3"}}, {"state_fields": {"a": "1", "b": "9", "d": "4"}}),
                         ["b", "c", "d"])
        self.assertEqual(F.signature("state", ["b", "c"]), "state:b,c")
        self.assertEqual(F.signature("state", []), "state")
        self.assertEqual(F.signature("action", ["x"]), "action")

    def test_the_minimal_record_stops_at_the_step(self):
        spec = {"seed": "ABC", "fuzz": {"seed": 4}, "setup": [{"cmd": "fight", "id": "X"}]}
        res = result(spec)
        res["choices"] += [{"kind": "cards", "step": 0, "max": 1, "options": ["A"], "picked": [0]},
                           {"kind": "cards", "step": 3, "max": 1, "options": ["B"], "picked": [0]}]
        rec = F.replay_record(spec, res, 2)
        self.assertNotIn("fuzz", rec)
        self.assertEqual(rec["room"], [3, 1])
        self.assertEqual(len(rec["steps"]), 3)
        self.assertIsNone(rec["steps"][0]["action"])
        self.assertEqual([c["step"] for c in rec["choices"]], [1, 0])
        self.assertEqual(rec["room_choices_from"], 0)


class Triage(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.cat = catalog()

    def fakes(self, bend_index=3):
        """Core2 differs from old on the fight at `bend_index` (step 2, the hash), and says which field in the re-run."""
        def runner(engine, specs):
            return [result(s, bend=(lambda i, h: "0" * 16 if i == 2 else h) if engine == "core2" and s["tag"].endswith(f":{bend_index}") else None)
                    for s in specs]

        def rerunner(engine, recs):
            return [{"steps": [{"state_fields": {"ActiveCombat.Hand": "a" if engine == "old" else "b", "RunState.Gold": "1"}}
                               for _ in r["steps"]]} for r in recs]
        return runner, rerunner

    def explained(self, signature, **kw):
        p = self.dir / "explained.json"
        entry = {"signature": signature, "row": "missing X.Y", "note": "n"}
        entry.update(kw)
        p.write_text(json.dumps({"format": 1, "explained": [entry]}))
        return str(p)

    def fuzz(self, explained=None, bend_index=3):
        runner, rerunner = self.fakes(bend_index)
        opts = options(str(self.dir / "out"), explained=explained or str(self.dir / "none.json"))
        return F.run_fuzz(opts, self.cat, runner, rerunner)

    def test_no_difference_exits_0(self):
        s = self.fuzz(bend_index=99)
        self.assertEqual((s["fights"], s["differing_fights"], s["untriaged"], F.exit_code(s)), (8, 0, 0, 0))
        self.assertGreater(s["steps"], 0)

    def test_an_untriaged_difference_exits_1_and_saves_a_minimal_spec(self):
        s = self.fuzz()
        self.assertEqual((s["differing_fights"], s["untriaged"], s["explained"]), (1, 1, 0))
        self.assertEqual(F.exit_code(s), 1)
        self.assertEqual(s["top_fields"], [("ActiveCombat.Hand", 1)])
        self.assertEqual(s["first"][0]["signature"], "state:ActiveCombat.Hand")
        saved = json.loads((self.dir / "out" / "run-7-none" / "diff-3.json").read_text())
        rec = saved["results"][0]
        self.assertEqual(len(rec["steps"]), 3)  # cut at the differing step
        self.assertEqual(rec["fuzz_diff"]["step"], 2)
        self.assertNotIn("fuzz", rec)

    def test_a_triaged_difference_exits_0(self):
        s = self.fuzz(self.explained("state:ActiveCombat.Hand*", row="missing X.Y"))
        self.assertEqual((s["differing_fights"], s["untriaged"], s["explained"], F.exit_code(s)), (1, 0, 1, 0))
        s = self.fuzz(self.explained("state:ActiveCombat.Hand", doc="docs/rebuild/M.md: item 3"))
        self.assertEqual(F.exit_code(s), 0)

    def test_a_triage_entry_for_another_signature_does_not_explain_it(self):
        self.assertEqual(F.exit_code(self.fuzz(self.explained("state:RunState.*"))), 1)
        self.assertEqual(F.exit_code(self.fuzz(self.explained("action"))), 1)

    def test_a_failed_batch_exits_1(self):
        def runner(engine, specs):
            raise RuntimeError("no dotnet")
        s = F.run_fuzz(options(str(self.dir / "o2")), self.cat, runner, None)
        self.assertEqual((s["fights"], s["untriaged"], len(s["failed_batches"])), (0, 0, 2))  # two batches, none judged
        self.assertEqual(F.exit_code(s), 1)

    def test_the_triage_file_is_checked(self):
        rows = {"missing X.Y"}
        ok = [{"signature": "state:a*", "row": "missing X.Y", "note": "n"}, {"signature": "action", "doc": "d.md: 1", "note": "n"}]
        self.assertEqual(F.validate_explained(ok, rows), [])
        for broken, word in (({"signature": "a", "note": "n"}, "exactly one"),
                             ({"signature": "a", "row": "r", "doc": "d", "note": "n"}, "exactly one"),
                             ({"signature": "a", "row": "gone", "note": "n"}, "not in hook_open"),
                             ({"row": "missing X.Y", "note": "n"}, "no signature"),
                             ({"signature": "a", "row": "missing X.Y"}, "no note")):
            bad = F.validate_explained([broken], rows)
            self.assertTrue(bad and any(word in b for b in bad), (broken, bad))

    def test_the_checked_in_triage_file_is_valid(self):
        doc = json.loads(F.EXPLAINED.read_text(encoding="utf-8"))
        self.assertEqual(doc["format"], 1)
        self.assertEqual(F.validate_explained(doc["explained"], F.hook_open_rows()), [])


@unittest.skipUnless(F.DEFAULT_DLL.is_file() and shutil.which("dotnet"), "EmuDump is not built (dotnet build tools/emudump -c Release)")
class RealEmuDump(unittest.TestCase):
    def test_a_few_fights_agree_on_both_engines_and_replay_from_their_records(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        cat = F.load_catalog(F.DEFAULT_DLL, [], None, d)
        self.assertGreater(len(cat.pick("cards")), 50)
        specs = [F.generate(cat, 11, i, 30) for i in range(6)]
        old, new = (F.run_specs(F.DEFAULT_DLL, e, specs, d) for e in ("old", "core2"))
        self.assertTrue(all(len(r["steps"]) > 1 for r in old))
        self.assertEqual([F.first_difference(a, b) for a, b in zip(old, new)], [None] * 6)
        recs = [F.replay_record(s, r, len(r["steps"]) - 1) for s, r in zip(specs, old)]
        again = F.run_records(F.DEFAULT_DLL, "old", recs, d, fields=False)
        self.assertEqual([[s["state_hash"] for s in r["steps"]] for r in again], [[s["state_hash"] for s in r["steps"]] for r in old])

    def test_a_changed_move_in_a_record_is_named_by_its_fields(self):
        """The re-run path on real output: the same record with one move changed differs from step 1 on, and the fields say where."""
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        cat = F.load_catalog(F.DEFAULT_DLL, [], None, d)
        specs = [F.generate(cat, 12, i, 20) for i in range(8)]
        res = F.run_specs(F.DEFAULT_DLL, "old", specs, d)
        i = next(i for i, r in enumerate(res) if len(r["steps"]) > 3 and r["steps"][1]["action"]["kind"] == "play")
        rec = F.replay_record(specs[i], res[i], 3)
        alt = copy.deepcopy(rec)
        alt["steps"][1]["action"] = {"kind": "end_turn"}
        a, b = (F.run_records(F.DEFAULT_DLL, "old", [r], d)[0] for r in (rec, alt))
        self.assertEqual(F.differing_fields(a["steps"][0], b["steps"][0]), [])
        fields = F.differing_fields(a["steps"][1], b["steps"][1])
        self.assertTrue(fields and all(isinstance(f, str) for f in fields), fields)
        self.assertTrue(F.signature("state", fields).startswith("state:"))


if __name__ == "__main__":
    unittest.main()
