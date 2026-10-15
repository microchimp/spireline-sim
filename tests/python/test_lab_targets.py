"""tools/fidelity/lab_targets.py: the Codex profile (aggregate counts only) and the weighted target list.

A three-run history in the Codex shape stands in for the real dataset; the profile is built from it both in
process and through the isolated interpreter the script starts for itself.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "referee"))

from tools.fidelity import lab_targets as lt  # noqa: E402


def fight(enc: str, room: str = "monster", monsters=("TOADPOLE",), **stats) -> dict:
    st = {"current_hp": 60, "max_hp": 80, **stats}
    return {"rooms": [{"room_type": room, "model_id": f"ENCOUNTER.{enc}", "monster_ids": [f"MONSTER.{m}" for m in monsters]}],
            "player_stats": [st]}


def event(name: str, **stats) -> dict:
    return {"rooms": [{"room_type": "event", "model_id": f"EVENT.{name}"}], "player_stats": [{"current_hp": 60, "max_hp": 80, **stats}]}


def run(seed: str, acts: list, deck: list, relics: list, potions=(), win=False, **kw) -> dict:
    d = {"seed": seed, "ascension": 10, "game_mode": "standard", "modifiers": [], "build_id": "v0.107.1", "win": win,
         "acts": [f"ACT.{a}" for a in kw.get("act_names", ["OVERGROWTH", "HIVE", "GLORY"])[:len(acts)]],
         "players": [{"character": "CHARACTER.IRONCLAD", "deck": [{"id": f"CARD.{c}", "floor_added_to_deck": f} for c, f in deck],
                      "relics": [{"id": f"RELIC.{r}", "floor_added_to_deck": f} for r, f in relics],
                      "potions": [{"id": f"POTION.{p}"} for p in potions], "max_potion_slot_count": 2}],
         "map_point_history": acts, "run_hash": seed.lower()}
    return d


STARTER = [("STRIKE_IRONCLAD", 1)] * 5 + [("DEFEND_IRONCLAD", 1)] * 4 + [("BASH", 1), ("ASCENDERS_BANE", 1)]


def history() -> list[dict]:
    a = run("AAAAAAAAAA", [[event("NEOW"), fight("NIBBITS_WEAK", cards_gained=[{"id": "CARD.INFLAME"}]),
                            fight("NIBBITS_WEAK"), fight("TOADPOLES_ELITE", "elite")]],
            STARTER + [("INFLAME", 2)], [("BURNING_BLOOD", 1), ("VAJRA", 3)])
    b = run("BBBBBBBBBB", [[event("NEOW"), fight("NIBBITS_WEAK"), fight("NIBBITS_WEAK"), fight("NIBBITS_WEAK")],
                           [fight("BOWLBUGS_NORMAL", monsters=("BOWLBUG_ROCK",)), fight("BOWLBUGS_NORMAL", monsters=("BOWLBUG_ROCK",))]],
            STARTER + [("INFLAME", 2), ("ANGER", 2)], [("BURNING_BLOOD", 1), ("VAJRA", 2)])
    other = {**run("CCCCCCCCCC", [[fight("NIBBITS_WEAK")]], STARTER, [("BURNING_BLOOD", 1)]), "ascension": 3}
    return [a, b, other]


def write_codex(folder: Path) -> Path:
    p = folder / "runs.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        for d in history():
            f.write(json.dumps(d) + "\n")
    return p


class ProfileTests(unittest.TestCase):
    def test_aggregate_counts_only(self):
        with tempfile.TemporaryDirectory() as td:
            prof = lt.build_profile(write_codex(Path(td)))
        self.assertEqual(2, prof["runs"])  # the A3 run is not an Ironclad A10 run
        self.assertEqual(2, prof["meet"]["card"]["INFLAME"])
        self.assertEqual(1, prof["meet"]["card"]["ANGER"])
        self.assertEqual(2, prof["meet"]["relic"]["VAJRA"])
        self.assertEqual({"NIBBITS_WEAK": 2, "BOWLBUGS_NORMAL": 1, "TOADPOLES_ELITE": 1}, prof["meet"]["encounter"])
        self.assertEqual(["BOWLBUG_ROCK"], prof["encounter_monsters"]["BOWLBUGS_NORMAL"])
        a1 = prof["acts"]["1"]
        self.assertEqual(6, a1["fights"])
        self.assertEqual({"monster": 5, "elite": 1}, a1["rooms"])
        self.assertEqual(2, a1["snapshots"])  # one snapshot per run and act
        self.assertEqual(1, prof["acts"]["2"]["snapshots"])
        self.assertEqual({"OVERGROWTH": 2}, a1["act_names"])
        # no seed, no run hash, no per-run list anywhere in the profile
        text = json.dumps(prof)
        for d in history():
            self.assertNotIn(d["seed"], text)
            self.assertNotIn(d["run_hash"], text)

    def test_pairs_need_three_sightings(self):
        with tempfile.TemporaryDirectory() as td:
            prof = lt.build_profile(write_codex(Path(td)))
        self.assertEqual({}, prof["acts"]["1"]["pairs"])  # each pair seen once: under PAIR_MIN

    def test_isolated_rebuild_and_cache(self):
        with tempfile.TemporaryDirectory() as td:
            codex, lab = write_codex(Path(td)), Path(td) / "lab"
            logs = []
            prof = lt.ensure_profile(lab, codex, log=logs.append)
            self.assertEqual(2, prof["runs"])
            self.assertEqual(lt.source_stamp(codex, 0), prof["source"])
            self.assertTrue(any("isolated" in x for x in logs))
            logs.clear()
            again = lt.ensure_profile(lab, codex, log=logs.append)
            self.assertEqual(prof, again)
            self.assertEqual([], logs)  # cached: the dataset is not read again

    def test_profile_step_refuses_a_plain_interpreter(self):
        if sys.flags.isolated:
            self.skipTest("running isolated")
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(2, lt.write_profile(Path(td), write_codex(Path(td)), 0))


def profile_doc() -> dict:
    return {"runs": 100, "meet": {"card": {"INFLAME": 40, "ANGER": 10, "BASH": 100}, "relic": {"VAJRA": 20},
                                  "encounter": {"NIBBITS_WEAK": 50, "QUEEN_BOSS": 10}, "enemy": {"QUEEN": 10},
                                  "event": {"NEOW": 100}, "act": {"OVERGROWTH": 50, "GLORY": 10}},
            "acts": {"1": {"snapshots": 50, "rooms": {"monster": 80, "boss": 10}, "cards": {"INFLAME": 20, "BASH": 50},
                           "relics": {"VAJRA": 10}},
                     "3": {"snapshots": 10, "rooms": {"monster": 20, "boss": 5}, "cards": {"INFLAME": 5}}},
            "encounter_monsters": {"QUEEN_BOSS": ["QUEEN"]}, "encounter_rooms": {"QUEEN_BOSS": "boss", "NIBBITS_WEAK": "monster"}}


class TargetTests(unittest.TestCase):
    def targets(self, **kw):
        uni = {k: set() for k in lt.KINDS}
        uni.update({"card": {"INFLAME", "ANGER", "BASH", "NEVER_MET"}, "relic": {"VAJRA"}, "encounter": {"NIBBITS_WEAK", "QUEEN_BOSS"},
                    "enemy": {"QUEEN"}, "event": {"NEOW"}, "act": {"OVERGROWTH", "GLORY"}, "power": {"INFLAME_POWER"}})
        uni["_enc_act"] = {"NIBBITS_WEAK": ["OVERGROWTH"], "QUEEN_BOSS": ["GLORY"]}
        args = {"cov": {"card:BASH": 500, "card:INFLAME": 2, "encounter:NIBBITS_WEAK": 10, "encounter:QUEEN_BOSS": 4, "enemy:QUEEN": 4},
                "reach": {}, "cells": {(1, "monster"): 0.9, (3, "boss"): 0.2}, "backlog": {}, "walk": {},
                "psrc": {"INFLAME_POWER": {("card", "INFLAME")}}}
        args.update(kw)
        got = lt.compute_targets(lt.Profile(profile_doc()), uni, **args)
        return {f"{t['kind']}:{t['id']}": t for t in got}, got

    def test_never_met_ids_are_left_out_and_order_is_by_weight(self):
        by, got = self.targets()
        self.assertNotIn("card:NEVER_MET", by)
        self.assertEqual(sorted(got, key=lambda t: -t["weight"]), got)

    def test_gap_factors(self):
        by, _ = self.targets()
        self.assertAlmostEqual(0.10 * 3.0, by["card:ANGER"]["weight"])  # met by 10 of 100 runs, never covered
        self.assertIn("no recorded or setup fight reaches it yet", by["card:ANGER"]["reasons"])
        self.assertTrue(any("only 2" in r for r in by["card:INFLAME"]["reasons"]))
        boss = by["encounter:QUEEN_BOSS"]
        self.assertAlmostEqual(0.10 * (2.0 - 0.2), boss["weight"])  # act 3 boss cell, 20% identical
        self.assertIn("reference set act 3 boss: 20% identical", boss["reasons"])
        self.assertAlmostEqual(0.5 * 1.1, by["encounter:NIBBITS_WEAK"]["weight"])  # a 90% cell still counts a little
        self.assertNotIn("event:NEOW", [k for k, t in by.items() if "no recorded" in " ".join(t["reasons"])])

    def test_backlog_walk_and_reach_multiply(self):
        by, _ = self.targets(backlog={"card:ANGER": ["C5-F01", "C5-F02"]}, walk={"card:ANGER": 7}, reach={"card:ANGER": 2})
        t = by["card:ANGER"]
        self.assertAlmostEqual(0.10 * 3.0 * 1.5 * 2.0 * 1.5, t["weight"])
        self.assertTrue(any("C5-F01, C5-F02" in r for r in t["reasons"]))
        self.assertTrue(any("7 runs" in r for r in t["reasons"]))

    def test_power_meets_through_its_card(self):
        by, _ = self.targets()
        self.assertIn("power:INFLAME_POWER", by)
        self.assertTrue(any("INFLAME" in r for r in by["power:INFLAME_POWER"]["reasons"]))

    def test_refset_own_encounter_rate_wins(self):
        by, _ = self.targets(refset_enc={"QUEEN_BOSS": (3, 4)})
        self.assertAlmostEqual(0.10 * 1.25, by["encounter:QUEEN_BOSS"]["weight"])


class InputTests(unittest.TestCase):
    def test_stop_tokens_are_what_differs(self):
        stop = {"expected": "STRIKE_IRONCLAD, STRIKE_IRONCLAD, BASH", "emulator": "STRIKE_IRONCLAD, BASH, ANGER"}
        self.assertEqual(["ANGER", "STRIKE_IRONCLAD"], lt.stop_tokens(stop))

    def test_reach_ties_never_run_blocks_to_case_labels(self):
        with tempfile.TemporaryDirectory() as td:
            eng = Path(td)
            (eng / "Core").mkdir()
            (eng / "Core" / "X.cs").write_text("switch (id)\n{\n  case CardId.Inflame:\n    a();\n    b();\n  case Vajra when x:\n"
                                                "    c();\n}\n", encoding="utf-8")
            prep = {"candidates": [{"file": "Core/X.cs", "line": 4, "fights": 0}, {"file": "Core/X.cs", "line": 5, "fights": 0},
                                   {"file": "Core/X.cs", "line": 7, "fights": 3}]}
            known = lt.known_names({"cards": {"Inflame": 1}, "relics": {"Vajra": 2}})
            self.assertEqual({"card:INFLAME": 2}, lt.reach_entities(prep, eng, known))

    def test_refset_cells(self):
        score = {"realistic_table": {"act3 boss": {"kept": 33, "identical_pct_of_kept": 15.2}, "act1 monster": {"kept": 0}}}
        self.assertEqual({(3, "boss"): 0.152}, lt.refset_cells(score))

    def test_coverage_counts_setup_items_and_lab_batches(self):
        with tempfile.TemporaryDirectory() as td:
            lab = Path(td)
            cov = lab / "coverage-setup.json"
            cov.write_text(json.dumps({"items": {"card+:ANGER": {"seen": 2}, "power:VULNERABLE_POWER@enemy": {"seen": 1}}}), encoding="utf-8")
            job = lab / "runs" / "20261015T0000-fights"
            job.mkdir(parents=True)
            (job / "b.verdicts.json").write_text("[]", encoding="utf-8")
            (job / "b.json").write_text(json.dumps({"runs": [{"setup": [{"cmd": "relic", "id": "VAJRA"}, {"cmd": "fight", "id": "X"}]}]}),
                                        encoding="utf-8")
            got = lt.coverage_counts(lab, None, [cov], None)
        self.assertEqual(2, got["card:ANGER"])
        self.assertEqual(1, got["power:VULNERABLE_POWER"])
        self.assertEqual(1, got["relic:VAJRA"])
        self.assertEqual(1, got["encounter:X"])


if __name__ == "__main__":
    unittest.main()
