"""rl/seed_exam_records.py and `run_seed_exam.py play --records` (D59, the Act 2 measure).

Pinned: a turn opens at the first decision after an end of turn and closes at its end of turn, the last turn of a won fight closes at 0 enemy HP, a lost
fight's last open turn closes at what was last seen, potions and cards are counted per turn; a decision is compacted to its kind, the option taken and
the ids the screen offered; the recorder keeps the lab's record exactly (the base class's methods run first) and adds the fights, decisions and decks to
the record's `measure` block, and a defect of its own never ends a run; `play_task` with the records folder set plays the run through the lab's recording
path, writes one derived file named by the seed's key (no seed inside), puts the lab's recorder back, and returns a row without the combat samples; without
the folder the run is the exam's, unchanged; `--records` reaches the workers through the environment.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_seed_exam as rse  # noqa: E402
import seed_exam_records as ser  # noqa: E402


class TurnTrackerTests(unittest.TestCase):
    def test_turns_open_at_the_first_decision_and_close_at_the_end_of_turn(self):
        t = ser.TurnTracker([[0, 50, 50], [1, 30, 30]])
        t.decision(80, 70, "card")
        t.decision(62, 70, "card")
        t.decision(62, 70, "end_turn")  # turn 1: enemies 80 -> 62, our HP 70 -> 70, two cards
        t.decision(60, 55, "potion")  # turn 2 begins with 55 HP: the enemy phase cost 15
        t.decision(41, 55, "card")
        self.assertEqual([[80, 62, 70, 70, 2, 0], [60, 0, 55, 55, 1, 1]], [x[:6] for x in t.finish(won=True)])
        self.assertEqual(1, t.potions_used)

    def test_a_lost_fight_closes_its_open_turn_at_what_was_last_seen(self):
        t = ser.TurnTracker([[0, 50, 50]])
        t.decision(50, 40, "card")
        t.decision(44, 12, "card")
        self.assertEqual([[50, 44, 40, 12, 2, 0]], t.finish(won=False))

    def test_a_fight_won_on_the_last_card_of_a_turn_that_had_ended_has_no_extra_turn(self):
        t = ser.TurnTracker([[0, 9, 9]])
        t.decision(9, 70, "end_turn")
        self.assertEqual(1, len(t.finish(won=True)))

    def test_the_first_observation_is_kept_for_the_enemy_roster(self):
        self.assertEqual([[0, 50, 50], [2, 30, 30]], ser.TurnTracker([[0, 50, 50], [2, 30, 30]]).enemies0)


def card(i: int, up: bool = False):
    return SimpleNamespace(card_id=i, upgraded=up)


def option(kind: str, **kw):
    base = dict(kind=kind, card=None, relic=None, potion=None, price=0)
    base.update(kw)
    return SimpleNamespace(**base)


def view(floor: int = 5, act: int = 0, hp: int = 60, deck=(), relics=(1, 2), potions=(7,), gold: int = 99):
    return SimpleNamespace(floor=floor, act_index=act, hp=hp, max_hp=80, gold=gold, deck=tuple(deck), relics=tuple(relics), potions=tuple(potions))


class CompactDecisionTests(unittest.TestCase):
    def test_a_card_pick_keeps_the_card_the_state_and_what_was_offered(self):
        d = SimpleNamespace(type="card_reward", options=(option("card.take", card=card(41, True)), option("card.take", card=card(42)), option("card.skip")))
        row = ser.compact_decision(view(deck=[card(1)] * 3), d, 0)
        self.assertEqual([5, 0, "card_reward", "card.take", 41, 1, None, None, 0, 60, 80, 99, 3, [["card.take", 41], ["card.take", 42], ["card.skip", -1]]], row)

    def test_a_relic_choice_keeps_the_relics_offered(self):
        d = SimpleNamespace(type="rewards", options=(option("reward.relic", relic=11), option("reward.relic", relic=12)))
        row = ser.compact_decision(view(), d, 1)
        self.assertEqual((12, [["reward.relic", 11], ["reward.relic", 12]]), (row[6], row[-1]))

    def test_a_map_choice_keeps_no_offers(self):
        d = SimpleNamespace(type="map", options=(option("map.node"), option("map.node")))
        self.assertIsNone(ser.compact_decision(view(), d, 0)[-1])


class FakeBase:
    """The lab recorder's four calls, with its attributes the loop reads (`fight`)."""

    def __init__(self):
        self.fight = None
        self.calls: list[str] = []

    def decision(self, v, d, pick):
        self.calls.append("decision")

    def open_fight(self, session, combat):
        self.calls.append("open")
        self.fight = {"floor": session.view().floor}

    def close_fight(self, session, won):
        self.calls.append("close")
        self.fight = None

    def sample(self, root, res, tokens):
        self.calls.append("sample")

    def finish(self, session, moves, won, **kw):
        self.calls.append("finish")
        return {"schema": 3, "outcome": {"won": won}}


class Rec(ser.MeasureMixin, FakeBase):
    def enemies_of(self, session):
        return [[0, 40, 40]]

    def enemy_hp_of(self, obs):
        return int(obs[1])


def session(floor=3, hp=50):
    s = SimpleNamespace(info={"hp": hp})
    s.view = lambda: view(floor=floor, hp=hp, deck=[card(2, True), card(1)], relics=(5,), potions=(7, 8))
    return s


def root(player_hp: int, enemy_hp: int):
    return SimpleNamespace(obs=[player_hp, enemy_hp])


def res(kind: str):
    return SimpleNamespace(best=SimpleNamespace(kind=kind))


class RecorderTests(unittest.TestCase):
    def play_fight(self, r: Rec, won: bool = True):
        r.open_fight(session(), SimpleNamespace(node_type=2, encounter_id=69))
        r.sample(root(50, 40), res("card"), None)
        r.sample(root(50, 30), res("end_turn"), None)
        r.sample(root(44, 30), res("potion"), None)
        r.sample(root(44, 30), res("card"), None)
        r.close_fight(session(hp=44), won=won)

    def test_the_lab_recorder_runs_first_and_the_record_gains_the_measure_block(self):
        r = Rec()
        self.play_fight(r)
        rec = r.finish(session(), 4, True)
        self.assertEqual(["open", "sample", "sample", "sample", "sample", "close", "finish"], r.calls)
        m = rec["measure"]
        self.assertEqual(1, len(m["fights"]))
        f = m["fights"][0]
        self.assertEqual((3, 2, 69, 50, True, 44, 1), (f["floor"], f["node"], f["enc"], f["hp0"], f["won"], f["hp1"], f["potions_used"]))
        self.assertEqual([[40, 30, 50, 50, 1, 0], [30, 0, 44, 44, 1, 1]], f["turns"])
        self.assertEqual([[0, 40, 40]], f["enemies"])
        self.assertEqual(([5], [7, 8]), (f["relics"], f["potions"]))
        self.assertEqual([[1, 0], [2, 1]], m["decks"][f["deck"]])  # the deck at the door, sorted
        self.assertIsNone(m["error"])

    def test_a_lost_fight_has_hp_after_zero(self):
        r = Rec()
        self.play_fight(r, won=False)
        self.assertEqual(0, r.finish(session(), 4, False)["measure"]["fights"][0]["hp1"])

    def test_equal_decks_share_one_entry(self):
        r = Rec()
        self.play_fight(r)
        self.play_fight(r)
        m = r.finish(session(), 8, True)["measure"]
        self.assertEqual((2, 1), (len(m["fights"]), len(m["decks"])))

    def test_decisions_are_compacted_in_order(self):
        r = Rec()
        d = SimpleNamespace(type="rest", options=(option("rest.heal"), option("rest.smith")))
        r.decision(view(floor=7), d, SimpleNamespace(index=1))
        rec = r.finish(session(), 0, False)
        self.assertEqual(["rest.smith"], [x[3] for x in rec["measure"]["decisions"]])

    def test_a_defect_of_the_accounting_never_ends_the_run(self):
        class Bad(Rec):
            def enemy_hp_of(self, obs):
                raise ValueError("boom")

        r = Bad()
        r.open_fight(session(), SimpleNamespace(node_type=1, encounter_id=1))
        r.sample(root(50, 40), res("card"), None)  # raises inside the accounting only
        r.close_fight(session(), won=True)
        rec = r.finish(session(), 1, True)
        self.assertTrue(rec["measure"]["error"].startswith("sample"))
        self.assertEqual(1, len(rec["measure"]["fights"]))


class RecordFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_the_file_is_named_by_the_key_and_round_trips(self):
        row = {"end": "death", "floor": 20, "nodes": "MRE", "rests": [[3, "smith"]], "seed": "SECRET-SEED"}
        p = ser.write_record(self.tmp / "r", "abcd1234abcd1234", 1, row, {"measure": {"fights": []}, "outcome": {"won": False}})
        self.assertEqual("abcd1234abcd1234-d1.json.gz", p.name)
        doc = ser.read_record(p)
        self.assertEqual(("abcd1234abcd1234", 1, "death", 20, "MRE"), (doc["key"], doc["draw"], doc["end"], doc["floor"], doc["nodes"]))
        self.assertNotIn("SECRET-SEED", json.dumps(doc))
        self.assertEqual([p], list((self.tmp / "r").iterdir()))  # no temp file left


class PlayTaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def fake_play(self, seen):
        def fake(seed, ix, config, ascension, agent=None, hook=None, record=False, **kw):
            import gen_loop  # noqa: PLC0415

            seen.update(record=record, recorder=gen_loop.RunRecorder)
            row = {"seed": seed, "ix": ix, "win": False, "floor": 22, "end": "death", "decisions": 9, "wall": 1.0, "crash": False, "config": config,
                   "generation": 14, "nodes": "MME", "rests": [], "cards": [], "boss_damage_acts": []}
            if record:
                row["_shard"] = {"record": {"measure": {"fights": [], "decisions": [], "decks": [], "error": None}, "outcome": {"won": False}}, "samples": {"big": 1}}
            return row

        return fake

    def test_with_the_folder_set_the_run_is_recorded_and_the_row_stays_small(self):
        import gen0  # noqa: PLC0415
        import gen_loop  # noqa: PLC0415

        lab = gen_loop.RunRecorder
        seen: dict = {}
        key = rse.seed_key("S1")
        with mock.patch.object(gen0, "_setup"), mock.patch.object(gen0, "play_emu_run", self.fake_play(seen)), \
                mock.patch.dict("os.environ", {rse.RECORDS_ENV: str(self.tmp / "rec")}):
            row = rse.play_task("S1", key, 1, 7, "a3-v1", 10, mock.Mock(), False)
        self.assertTrue(seen["record"])
        self.assertTrue(issubclass(seen["recorder"], ser.MeasureMixin) and issubclass(seen["recorder"], lab))
        self.assertIs(lab, gen_loop.RunRecorder)  # the lab's recorder is back
        self.assertNotIn("_shard", row)
        self.assertNotIn("seed", row)
        doc = ser.read_record(self.tmp / "rec" / f"{key}-d1.json.gz")
        self.assertEqual((key, 1, "MME"), (doc["key"], doc["draw"], doc["nodes"]))
        self.assertNotIn("S1", json.dumps(doc).replace(key, ""))

    def test_without_the_folder_the_run_is_the_exams_unchanged(self):
        import gen0  # noqa: PLC0415

        seen: dict = {}
        with mock.patch.object(gen0, "_setup"), mock.patch.object(gen0, "play_emu_run", self.fake_play(seen)), mock.patch.dict("os.environ", {}):
            import os  # noqa: PLC0415

            os.environ.pop(rse.RECORDS_ENV, None)
            row = rse.play_task("S1", rse.seed_key("S1"), 0, 7, "a3-v1", 10, mock.Mock(), False)
        self.assertFalse(seen["record"])
        self.assertEqual("death", row["end"])


class FlagTests(unittest.TestCase):
    def test_the_flag_is_on_play_and_reaches_the_workers_through_the_environment(self):
        a = rse.build_parser().parse_args(["play", "--out", "x", "--records", "somewhere"])
        self.assertEqual("somewhere", a.records)
        self.assertIsNone(rse.build_parser().parse_args(["play", "--out", "x"]).records)
        self.assertEqual("SPIRELINE_EXAM_RECORDS", rse.RECORDS_ENV)


if __name__ == "__main__":
    unittest.main()
