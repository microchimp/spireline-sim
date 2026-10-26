"""A11: the real side's boss fight and rest kinds, on values taken from generation 5's S3 real-game block (tests/python/fixtures/real_s3_*)."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "referee", ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import lockstep  # noqa: E402
import run_fields  # noqa: E402

FX = ROOT / "tests" / "python" / "fixtures"
RECORDS = json.loads((FX / "real_s3_boss_records.json").read_text(encoding="utf-8"))["records"]


def combat(act: int, floor: int, name: str, hp: int, max_hp: int) -> dict:
    return {"kind": "combat", "actions": [["end"]], "obs": {"act": act, "floor": floor, "combat": {
        "round": 1, "enemies": [{"name": name, "hp": hp, "max_hp": max_hp}]}}}


def played(msgs: list[dict], outcome: dict) -> dict:
    """A Lockstep with no emulator side (the fallback plays) fed a scripted stream; its row's fields."""
    ls = lockstep.Lockstep(None, lambda m: m.get("pick", 0), "seed")
    for m in msgs:
        ls.decide(m)
    ls.finish(outcome)
    return ls.fields.row()


class RealBossFightTests(unittest.TestCase):
    def test_the_fixture_is_real_boss_floor_records_with_a_stale_room_type(self):
        self.assertGreaterEqual(len(RECORDS), 10)
        self.assertEqual({r["encounter"] for r in RECORDS} - {"WATERFALL_GIANT", "LAGAVULIN_MATRIARCH", "CEREMONIAL_BEAST", "VANTOM",
                                                               "SOUL_FYSH", "KIN_FOLLOWER+KIN_FOLLOWER+KIN_PRIEST"}, set())
        self.assertNotIn("Boss", {r["room_type"] for r in RECORDS})  # the lockstep never saw a map label "Boss" on the real stream

    def test_the_boss_fight_is_recognised_on_the_boss_floor_whatever_the_last_map_label_was(self):
        checked = 0
        for rec in RECORDS:
            start, now = max(rec["ref_hp"][0], rec["emu_hp"][0]), rec["ref_hp"][0]
            name = rec["encounter"].split("+")[-1]
            stale_map = {"kind": "map", "actions": [[0, 14, rec["room_type"] or "Monster"]], "obs": {"act": 0, "floor": 15}}
            after = {"kind": "card_reward", "actions": ["skip"], "obs": {"act": 0, "floor": 17}}
            row = played([stale_map, combat(0, 17, name, start, start), combat(0, 17, name, now, start)],
                         {"class": "death", "end": "death", "floor": 17})
            want = round((start - now) / start, 4)
            self.assertEqual((row["boss_damage"], row["boss_damage_acts"]), (want, [[0, want]]), rec)
            won = played([stale_map, combat(0, 17, name, start, start), combat(0, 17, name, now, start), after], {"class": "truncated", "floor": 17})
            self.assertEqual(won["boss_damage"], 1.0, rec)  # the screen after the fight: it was won
            checked += 1
        self.assertEqual(checked, len(RECORDS))

    def test_only_the_boss_floor_of_the_act_is_a_boss_fight(self):
        self.assertEqual(played([combat(0, 16, "JAW_WORM", 40, 40), combat(0, 16, "JAW_WORM", 10, 40)], {"class": "death"})["boss_damage_acts"], [])
        act2 = played([combat(1, 33, "KAISER_CRAB", 200, 200), combat(1, 33, "KAISER_CRAB", 50, 200)], {"class": "death"})
        self.assertEqual((act2["boss_damage"], act2["boss_damage_acts"]), (0.0, [[1, 0.75]]))  # act 2's boss: act 1's field stays 0
        self.assertEqual(run_fields.ACT_BOSS_FLOORS, (17, 33, 48))

    def test_the_boss_floor_table_is_the_readiness_tables(self):
        from tools.fidelity import readiness  # noqa: PLC0415

        self.assertEqual(run_fields.ACT_BOSS_FLOORS, readiness.ACT_ENDS)


class RealRestTests(unittest.TestCase):
    def rests(self, *screens) -> list:
        msgs = [{"kind": "rest", "actions": list(labels), "pick": pick, "obs": {"act": 0, "floor": floor}} for floor, labels, pick in screens]
        return played(msgs, {"class": "death"})["rests"]

    def test_each_rest_label_is_recorded_as_the_emulators_kind(self):
        self.assertEqual(self.rests((8, ["HEAL", "SMITH"], 1), (12, ["HEAL", "SMITH"], 0), (13, ["LIFT", "HEAL"], 0), (14, ["DIG", "HEAL"], 0)),
                         [[8, "smith"], [12, "heal"], [13, "lift"], [14, "dig"]])
        for label, kind in lockstep.EmuLock.REST.items():  # the lockstep's table, label by label
            self.assertEqual(self.rests((3, [label], 0)), [[3, kind]], label)

    def test_the_games_proceed_and_leave_are_not_rest_choices_and_an_unknown_label_is_not_a_heal(self):
        self.assertEqual(self.rests((8, ["proceed"], 0), (9, ["leave"], 0)), [])  # the real log's rest screens end in "proceed"
        self.assertEqual(self.rests((8, ["TOKE", "HEAL"], 0)), [[8, "toke"]])


class RealRestLogTests(unittest.TestCase):
    def test_the_games_log_of_the_block_holds_heal_and_proceed_only_in_the_excerpt(self):
        labels = {ln.split('rest "')[1].rstrip('"') for ln in (FX / "real_s3_log_excerpt.txt").read_text(encoding="utf-8").splitlines() if ' rest "' in ln}
        self.assertEqual(labels, {"HEAL", "proceed"})
