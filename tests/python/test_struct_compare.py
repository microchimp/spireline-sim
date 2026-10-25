"""tools/fidelity/struct_compare.py: structural diffs, the allowed rows, the field census and its gate, and a broken case
for each (a change that must be flagged is flagged)."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import struct_compare as sc  # noqa: E402

ROWS = sc.load_rows()


def snap(**kw) -> dict:
    s = {
        "in_combat": True, "round": 1,
        "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3, "powers": [["STRENGTH_POWER", 2], ["FRAIL_POWER", 1]]},
        "hand": [["STRIKE_IRONCLAD", False], ["BASH", False]], "draw": [["DEFEND_IRONCLAD", False]], "discard": [], "exhaust": [],
        "enemies": [{"name": "CORPSE_SLUG", "hp": 20, "max_hp": 28, "block": 0, "powers": [], "intents": [["Attack", 6, 1]]}],
        "rng": {"Shuffle": {"counter": 3}, "MonsterAi": {"counter": 1}},
    }
    s.update(kw)
    return s


def net(**kw) -> dict:
    n = {
        "creatures": [{"monster": None, "player": 1, "hp": 70, "max_hp": 80, "block": 0,
                       "powers": [["STRENGTH_POWER", 2], ["FRAIL_POWER", 1]]},
                      {"monster": "CORPSE_SLUG", "player": None, "hp": 20, "max_hp": 28, "block": 0, "powers": []}],
        "players": [{"id": 1, "energy": 3, "piles": {"Hand": [{"id": "STRIKE_IRONCLAD", "upgrades": 0}, {"id": "BASH", "upgrades": 0}],
                                                      "Draw": [{"id": "DEFEND_IRONCLAD"}], "Discard": [], "Exhaust": [], "Play": []},
                     "relics": [{"id": "BURNING_BLOOD"}]}],
        "rng": {"counters": {"Shuffle": 3, "MonsterAi": 1}},
    }
    n.update(kw)
    return n


def fight(rec_states: list, eng_states: list, rec_nets=None, eng_nets=None) -> tuple[dict, dict]:
    g = {"seed": "S", "steps": []}
    e = {"seed": "S", "steps": []}
    for k, st in enumerate(rec_states):
        step = {"action": None if k == 0 else {"kind": "end_turn"}, "state": st}
        if rec_nets is not None and rec_nets[k] is not None:
            step["net"] = [{"ctx": "c", "state": rec_nets[k]}]
        g["steps"].append(step)
    for k, st in enumerate(eng_states):
        e["steps"].append({"state": {}, "game_view": {"snapshot": st, "net": None if eng_nets is None else eng_nets[k],
                                                      "private": {}}})
    return g, e


def first(rec, eng, **kw) -> dict:
    return sc.compare_fight(*fight(rec, eng, **kw), ROWS)


def paths(r: dict) -> list:
    return [p for p, _, _ in r["at"]]


class Structural(unittest.TestCase):
    def test_identical(self):
        r = first([snap()], [snap()])
        self.assertIsNone(r["first"])
        for p in ("snap.player.hp", "snap.player.powers{*}", "snap.hand[*]", "snap.hand#len", "snap.enemies[*].intents[*]",
                  "snap.rng.Shuffle.counter"):
            self.assertIn(p, r["census"])

    def test_scalar_diff(self):
        r = first([snap(), snap()], [snap(), snap(player={**snap()["player"], "hp": 69})])
        self.assertEqual(r["first"], 1)
        self.assertEqual(r["at"], [("snap.player.hp", 70, 69)])

    def test_power_order_is_no_diff(self):
        p = snap()["player"]
        r = first([snap()], [snap(player={**p, "powers": list(reversed(p["powers"]))})])
        self.assertIsNone(r["first"])

    def test_power_missing_or_extra_is_a_diff(self):
        p = snap()["player"]
        r = first([snap()], [snap(player={**p, "powers": [["STRENGTH_POWER", 2]]})])
        self.assertEqual(paths(r), ["snap.player.powers{FRAIL_POWER}"])
        r = first([snap()], [snap(player={**p, "powers": p["powers"] + [["VULNERABLE_POWER", 1]]})])
        self.assertEqual(paths(r), ["snap.player.powers{VULNERABLE_POWER}"])

    def test_extra_card_is_len_and_entry(self):
        r = first([snap()], [snap(hand=snap()["hand"] + [["BASH", True]])])
        self.assertEqual(paths(r), ["snap.hand#len", "snap.hand[2]"])

    def test_record_only_field_is_a_diff_unless_unmodelled(self):
        rec = snap(rng={"Shuffle": {"counter": 3, "seed": 9}, "MonsterAi": {"counter": 1, "seed": 4}})
        r = first([rec], [snap()])
        self.assertIsNone(r["first"])
        self.assertIn("snap.rng.Shuffle.seed", r["unmodelled"])
        broken = snap(extra_field=5)
        r = first([broken], [snap()])
        self.assertEqual(paths(r), ["snap.extra_field"])

    def test_engine_only_field_is_unjudged(self):
        r = first([snap()], [snap(potions=[None, None])])
        self.assertIsNone(r["first"])
        self.assertIn("snap.potions#len", r["unjudged"])

    def test_length_and_error(self):
        r = first([snap(), snap()], [snap()])
        self.assertEqual((r["first"], paths(r)), (1, ["(length)"]))
        g, e = fight([snap(), snap()], [snap()])
        e["error"] = "boom"
        self.assertEqual(paths(sc.compare_fight(g, e, ROWS)), ["(error)"])

    def test_engine_without_game_view_raises(self):
        g, e = fight([snap()], [snap()])
        del e["steps"][0]["game_view"]
        with self.assertRaises(ValueError):
            sc.compare_fight(g, e, ROWS)


class AllowedRows(unittest.TestCase):
    def test_rows_need_reason_and_known_kind(self):
        for bad in ({"id": "x", "kind": "ignore", "patterns": ["a"], "legacy": "y"},
                    {"id": "x", "kind": "nope", "reason": "r", "legacy": "y"}):
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / "a.json"
                f.write_text(json.dumps({"rows": [bad]}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    sc.load_rows(f)

    def test_every_row_has_reason_and_legacy(self):
        for r in ROWS:
            self.assertTrue(r["reason"] and r["legacy"], r["id"])

    def test_power_fold(self):
        p = snap()["player"]
        rec = snap(player={**p, "powers": [["SETUP_STRIKE_POWER", 2], ["FLEX_POTION_POWER", 3]]})
        eng = snap(player={**p, "powers": [["TEMPORARY_STRENGTH_POWER", 5]]})
        r = first([rec], [eng])
        self.assertIsNone(r["first"])
        self.assertIn("power-folds", r["allowed"])
        broken = snap(player={**p, "powers": [["TEMPORARY_STRENGTH_POWER", 4]]})
        self.assertEqual(paths(first([rec], [broken])), ["snap.player.powers{TEMPORARY_STRENGTH_POWER}"])

    def test_intent_vocabulary(self):
        e0 = snap()["enemies"][0]
        rec = snap(enemies=[{**e0, "intents": [["DebuffStrong"], ["Buff"], ["Defend"]]}])
        eng = snap(enemies=[{**e0, "intents": [["Debuff"], ["Buff"]]}])
        r = first([rec], [eng])
        self.assertIsNone(r["first"])
        self.assertIn("intent-vocabulary", r["allowed"])
        rec = snap(enemies=[{**e0, "intents": [["Attack", 6, 2]]}])
        broken = snap(enemies=[{**e0, "intents": [["Attack", 7, 2]]}])
        self.assertEqual(paths(first([rec], [broken])), ["snap.enemies[0].intents[0]"])

    def test_dead_enemy_dropped(self):
        e0 = snap()["enemies"][0]
        rec = snap(enemies=[{**e0, "hp": 0}, e0])
        r = first([rec], [snap()])
        self.assertIsNone(r["first"])
        broken = snap(enemies=[{**e0, "hp": 1}, e0])
        self.assertIn("snap.enemies#len", paths(first([broken], [snap()])))

    def test_lost_fight_compares_hp_only(self):
        rec = snap(in_combat=False, player={**snap()["player"], "hp": 0}, hand=[])
        eng = snap(in_combat=False, player={**snap()["player"], "hp": 0, "block": 9})
        r = first([rec], [eng])
        self.assertIsNone(r["first"])
        broken = snap(in_combat=False, player={**snap()["player"], "hp": 3})
        self.assertEqual(paths(first([rec], [broken])), ["snap.player.hp"])

    def test_after_fight_energy_only_out_of_combat(self):
        p = {k: v for k, v in snap()["player"].items() if k != "energy"}
        r = first([snap(in_combat=False)], [snap(in_combat=False, player=p)])
        self.assertIsNone(r["first"])
        self.assertEqual(paths(first([snap()], [snap(player=p)])), ["snap.player.energy"])

    def test_move_id_unmodelled(self):
        e0 = snap()["enemies"][0]
        r = first([snap(enemies=[{**e0, "move": "GOOP_MOVE"}])], [snap()])
        self.assertIsNone(r["first"])
        self.assertIn("snap.enemies[*].move", r["unmodelled"])


class NetShape(unittest.TestCase):
    def run_net(self, rec_net, eng_net, rec_snap=None):
        rs = rec_snap or snap()
        return sc.compare_fight(*fight([rs], [rs], rec_nets=[rec_net], eng_nets=[eng_net]), ROWS)

    def test_identical_and_census(self):
        r = self.run_net(net(), net())
        self.assertIsNone(r["first"])
        self.assertIn("net.players[*].piles.Hand[*].id", r["census"])
        self.assertIn("net.players[*].piles.Hand[*].cost", r["census"])

    def test_card_defaults_compare_absent_cost(self):
        eng = net()
        eng["players"][0]["piles"]["Hand"][0]["cost"] = 0
        self.assertEqual(paths(self.run_net(net(), eng)), ["net.players[0].piles.Hand[0].cost"])

    def test_keywords_are_a_set(self):
        rec, eng = net(), net()
        rec["players"][0]["piles"]["Hand"][0]["keywords"] = ["Ethereal", "Exhaust"]
        eng["players"][0]["piles"]["Hand"][0]["keywords"] = ["Exhaust", "Ethereal"]
        self.assertIsNone(self.run_net(rec, eng)["first"])
        eng["players"][0]["piles"]["Hand"][0]["keywords"] = ["Exhaust"]
        self.assertEqual(paths(self.run_net(rec, eng)), ["net.players[0].piles.Hand[0].keywords"])

    def test_upgrades_both_positive(self):
        rec, eng = net(), net()
        rec["players"][0]["piles"]["Hand"][0]["upgrades"] = 2
        eng["players"][0]["piles"]["Hand"][0]["upgrades"] = 1
        upgraded = snap(hand=[["STRIKE_IRONCLAD", True], ["BASH", False]])  # the record's snapshot agrees with its net state
        self.assertIsNone(self.run_net(rec, eng, upgraded)["first"])
        eng["players"][0]["piles"]["Hand"][0]["upgrades"] = 0
        self.assertEqual(paths(self.run_net(rec, eng, upgraded)), ["net.players[0].piles.Hand[0].upgrades"])

    def test_props_floor_unmodelled(self):
        rec = net()
        rec["players"][0]["relics"][0].update(props={"Counter": 1}, floor=1)
        r = self.run_net(rec, net())
        self.assertIsNone(r["first"])
        self.assertIn("net.players[*].relics[*].props.Counter", r["unmodelled"])

    def test_settle_after_last_checksum(self):
        # The record's snapshot shows the hand moved after the step's last checkpoint (BASH played by the game later):
        # the net hand is judged by the snapshot comparison, which agrees.
        rec_snap = snap(hand=[["STRIKE_IRONCLAD", False]])
        eng = net()
        eng["players"][0]["piles"]["Hand"] = [{"id": "STRIKE_IRONCLAD", "upgrades": 0}]
        r = self.run_net(net(), eng, rec_snap)
        self.assertIsNone(r["first"])
        self.assertIn("after-last-checksum", r["allowed"])
        # Broken: the record did not move, so the same engine hand is a difference.
        r = self.run_net(net(), eng, snap())
        self.assertIn("net.players[0].piles.Hand#len", paths(r))

    def test_missing_engine_net_in_combat(self):
        self.assertEqual(paths(self.run_net(net(), None)), ["net"])

    def test_a_missing_engine_net_is_no_census_field(self):
        # R7h, the traced fight on fewer fields: an engine with no net view at a step compared nothing on net there, so
        # an engine that writes the view at that step is not compared on fewer fields than it.
        missing, written = self.run_net(net(), None), self.run_net(net(), net())
        self.assertNotIn("net", missing["census"])
        self.assertFalse(any(p.startswith("net") for p in missing["census"]))
        self.assertEqual(sc.census_gate({"f": missing}, {"f": written}), {})
        self.assertIn("f", sc.census_gate({"f": written}, {"f": missing}))

    def test_late_steps_stop_the_net_shape(self):
        g, e = fight([snap(), snap()], [snap(), snap()], rec_nets=[net(), net()], eng_nets=[net(), net()])
        g["steps"][1]["net"][0]["state"]["players"][0]["gold"] = 99  # a field only the record has, not unmodelled: a diff
        g["steps"][0]["checksums"] = [["x", "PlayCardAction"]]
        g["steps"][1].update(action={"kind": "play", "played": True}, checksums=[["x", "Other"]])
        self.assertIsNone(sc.compare_fight(g, e, ROWS)["first"])
        g["steps"][1]["checksums"] = [["x", "PlayCardAction"]]
        self.assertEqual(paths(sc.compare_fight(g, e, ROWS)), ["net.players[0].gold"])


class CensusGate(unittest.TestCase):
    def test_gate(self):
        seeded = snap(rng={"Shuffle": {"counter": 3, "seed": 9}, "MonsterAi": {"counter": 1, "seed": 4}})
        a = {"f1": sc.compare_fight(*fight([seeded], [seeded]), ROWS)}
        self.assertEqual(sc.census_gate(a, copy.deepcopy(a)), {})
        b = {"f1": sc.compare_fight(*fight([seeded], [snap()]), ROWS)}
        self.assertIsNone(b["f1"]["first"])  # B writes no seeds: unmodelled, no difference...
        self.assertEqual(sc.census_gate(a, b), {"f1": ["snap.rng.MonsterAi.seed", "snap.rng.Shuffle.seed"]})  # ...but fewer fields
        self.assertEqual(sc.census_gate(a, {}), {"f1": ["(no result)"]})
        self.assertEqual(sc.census_gate(b, a), {})

    def test_missing_entry_keeps_the_unmodelled_rows(self):
        # The record lists an enemy the engine lacks. Its move id (enemy-move-id, unmodelled) is not compared there either:
        # it leaves the census as it does for an entry both sides have. Its other fields and the roster's #len stay
        # differences (a record-only field is a difference unless a row lists it), so the step still differs.
        slug = {"name": "CORPSE_SLUG", "hp": 20, "max_hp": 28, "block": 0, "powers": [], "intents": [["Attack", 6, 1]]}
        rec = snap(enemies=[dict(slug, move="BITE"), dict(slug, name="TWIG_SLIME_S", move="TACKLE")])
        r = sc.compare_fight(*fight([rec], [snap(enemies=[slug])]), ROWS)
        self.assertEqual(r["first"], 0)
        self.assertIn("snap.enemies#len", paths(r))
        self.assertIn("snap.enemies[1].hp", paths(r))
        self.assertNotIn("snap.enemies[1].move", paths(r))
        self.assertNotIn("snap.enemies[*].move", r["census"])
        self.assertIn("snap.enemies[*].hp", r["census"])
        self.assertIn("snap.enemies[*].move", r["unmodelled"])
        self.assertEqual(r["allowed"].get("enemy-move-id"), 2)  # both enemies' move ids, the missing entry's included

    def test_missing_entry_on_the_engine_side_is_compared(self):
        # An entry only the engine has: its fields are the engine's and are compared (no unmodelled row applies to a
        # path the engine writes).
        slug = {"name": "CORPSE_SLUG", "hp": 20, "max_hp": 28, "block": 0, "powers": [], "intents": [["Attack", 6, 1]]}
        r = sc.compare_fight(*fight([snap(enemies=[slug])], [snap(enemies=[slug, dict(slug, move="BITE")])]), ROWS)
        self.assertIn("snap.enemies[1].move", paths(r))
        self.assertIn("snap.enemies[*].move", r["census"])


class Calibration(unittest.TestCase):
    def test_contingency_table(self):
        from tools.fidelity import struct_calibrate as cal  # noqa: PLC0415
        rs = [{"fc": 3, "sc": 3}, {"fc": 3, "sc": 1}, {"fc": 1, "sc": 3}, {"fc": 2, "sc": None}, {"fc": None, "sc": 0},
              {"fc": None, "sc": None}]
        self.assertEqual(cal.table(rs, "fc"), {"both_same_step": 1, "both_struct_earlier": 1, "both_struct_later": 1,
                                               "legacy_only": 1, "struct_only": 1, "neither": 1, "struct_crash": 0})


class PowerFoldCount(unittest.TestCase):
    def test_fold_counts_only_with_a_folded_subclass(self):
        p = snap()["player"]
        base = snap(player={**p, "powers": [["TEMPORARY_STRENGTH_POWER", 5]]})
        r = first([base], [base])
        self.assertIsNone(r["first"])
        self.assertNotIn("power-folds", r["allowed"])  # only the base power: nothing folded
        sub = snap(player={**p, "powers": [["SETUP_STRIKE_POWER", 5]]})
        self.assertEqual(first([sub], [base])["allowed"].get("power-folds"), 1)


class LateStepsRow(unittest.TestCase):
    def test_limit_applies_only_with_the_row_and_counts_the_skipped_steps(self):
        g, e = fight([snap(), snap()], [snap(), snap()], rec_nets=[net(), net()], eng_nets=[net(), net()])
        g["steps"][1]["net"][0]["state"]["players"][0]["gold"] = 99  # a field only the record has: a diff when judged
        g["steps"][0]["checksums"] = [["x", "PlayCardAction"]]
        g["steps"][1].update(action={"kind": "play", "played": True}, checksums=[["x", "Other"]])
        r = sc.compare_fight(g, e, ROWS)
        self.assertIsNone(r["first"])
        self.assertEqual(r["allowed"].get("late-steps"), 1)
        without = [x for x in ROWS if x["kind"] != "late_steps"]
        r = sc.compare_fight(g, e, without)
        self.assertEqual(paths(r), ["net.players[0].gold"])  # no row, no limit: the step is judged
        self.assertNotIn("late-steps", r["allowed"])


def run_net(rec_net, eng_net, rows=ROWS):
    rs = snap()
    return sc.compare_fight(*fight([rs], [rs], rec_nets=[rec_net], eng_nets=[eng_net]), rows)


class Waived(unittest.TestCase):
    def test_waived_cost_is_not_in_the_census(self):
        cost = "net.players[*].piles.Draw[*].cost"
        eng = net()
        eng["players"][0]["piles"]["Draw"][0]["cost"] = 3  # cost outside the hand: waived by cost-outside-hand
        for e in (net(), eng):  # equal or different: the row stands for the comparison, so it is waived and never compared
            r = run_net(net(), e)
            self.assertIsNone(r["first"])
            self.assertIn(cost, r["waived"])
            self.assertNotIn(cost, r["census"])
        self.assertIn("net.players[*].piles.Hand[*].cost", run_net(net(), net())["census"])  # cost in the hand is compared

    def test_gate_compares_census_only_and_reports_waived(self):
        cost = "net.players[*].piles.Draw[*].cost"
        a = {"f": run_net(net(), net(), [x for x in ROWS if x["kind"] != "ignore"])}  # an engine with no such row compares it
        b = {"f": run_net(net(), net())}
        self.assertIn(cost, a["f"]["census"])
        self.assertIn(cost, sc.census_gate(a, b)["f"])  # B waived it: not compared there, so the gate fails B
        self.assertEqual(sc.census_gate(b, a), {})
        self.assertEqual((sc.waived_total(a), sc.waived_total(b)), (0, len(b["f"]["waived"])))
        self.assertGreater(sc.waived_total(b), 0)


class CalibrateRound(unittest.TestCase):
    def test_a_crash_has_its_own_cell(self):
        from tools.fidelity import struct_calibrate as cal  # noqa: PLC0415
        rs = [{"fc": 3, "sc": -1}, {"fc": None, "sc": -1}, {"fc": None, "sc": 2}, {"fc": None, "sc": None}]
        self.assertEqual(cal.table(rs, "fc"), {"both_same_step": 0, "both_struct_earlier": 0, "both_struct_later": 0, "legacy_only": 0,
                                               "struct_only": 1, "neither": 1, "struct_crash": 2})

    def test_self_record_is_identical_and_not_limited_by_checksums(self):
        from tools.fidelity import struct_calibrate as cal  # noqa: PLC0415
        g, e = fight([snap(), snap()], [snap(), snap()], rec_nets=[net(), net()], eng_nets=[net(), net()])
        g["steps"][0]["checksums"] = [["x", "PlayCardAction"]]
        g["steps"][1].update(action={"kind": "play", "played": True}, checksums=[["x", "Other"]])  # a late step in the record
        rec = cal.self_record(g, e)
        self.assertEqual(len(rec["steps"]), 2)
        self.assertIsNone(sc.compare_fight(rec, e, ROWS)["first"])
        bad = copy.deepcopy(e)  # broken: the engine output differs from the view that stands as the record
        bad["steps"][1]["game_view"]["net"]["players"][0]["energy"] = 0
        self.assertEqual(paths(sc.compare_fight(rec, bad, ROWS)), ["net.players[0].energy"])  # the late step is judged too
        bad["steps"][1]["game_view"]["snapshot"]["player"]["hp"] = 1
        self.assertIn("snap.player.hp", paths(sc.compare_fight(rec, bad, ROWS)))


OLD_ONLY = ("power-folds", "intent-vocabulary", "cost-outside-hand", "affliction-amount", "upgrade-count", "enemy-move-id", "rng-seeds",
            "net-sequencing", "saved-props-floor", "relic-counters", "after-fight-turn-state", "fight-lost")
EVERY_ENGINE = ("dead-enemies", "after-last-checksum", "late-steps")


class EngineScope(unittest.TestCase):
    def test_a_row_needs_an_engines_list(self):
        for bad in (None, [], "old", [""]):
            row = {"id": "x", "kind": "ignore", "patterns": ["a"], "reason": "r", "legacy": "y"}
            if bad is not None:
                row["engines"] = bad
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / "a.json"
                f.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    sc.load_rows(f)

    def test_scopes(self):
        by = {r["id"]: r["engines"] for r in ROWS}
        self.assertEqual(by.pop("kernel-sequencing"), ["core2"])
        self.assertEqual(by.pop("holder-tallies"), ["core2"])
        self.assertEqual(sorted(by), sorted(OLD_ONLY + EVERY_ENGINE))
        for i in OLD_ONLY:
            self.assertEqual(by[i], ["old"], i)
        for i in EVERY_ENGINE:
            self.assertEqual(by[i], ["*"], i)
        self.assertEqual(sorted(r["id"] for r in sc.for_engine(ROWS, "core3")), sorted(EVERY_ENGINE))  # an engine with a state of its own: no old rows
        self.assertEqual(sorted(r["id"] for r in sc.for_engine(ROWS, "core2", [])), sorted(OLD_ONLY + EVERY_ENGINE + ("kernel-sequencing", "holder-tallies")))  # core2 with no family migrated runs over old state
        self.assertEqual(len(sc.for_engine(ROWS, "old")), len(ROWS) - 2)  # all but the two core2-only rows

    def test_scopes_under_the_checked_in_switch(self):
        # The default for core2 is the repository's migrated_families.json: an old row whose family is migrated no longer applies.
        migrated = set(sc.load_migrated())
        listed = {r["id"]: set(r.get("families", [])) for r in ROWS if "old" in r["engines"]}
        got = {r["id"] for r in sc.for_engine(ROWS, "core2")}
        self.assertEqual(got, {r["id"] for r in sc.for_engine(ROWS, "core2", sorted(migrated))})
        self.assertEqual({i for i in listed if i not in got}, {i for i, f in listed.items() if f & migrated})
        # F10 migrated (as checked in): the two old rows that could hide a combat-end rule are out; F08 alone leaves them in.
        out = {i for i in OLD_ONLY if i not in {r["id"] for r in sc.for_engine(ROWS, "core2", ["F10"])}}
        self.assertEqual(out, {"fight-lost", "after-fight-turn-state"})
        self.assertEqual({i for i in OLD_ONLY if i not in {r["id"] for r in sc.for_engine(ROWS, "core2", ["F08"])}}, set())
        if {"F08", "F10"} <= migrated:
            self.assertEqual({i for i in OLD_ONLY if i not in got}, {"fight-lost", "after-fight-turn-state"})

    def test_old_rows_do_not_apply_to_another_engine(self):
        p = snap()["player"]
        rec = snap(player={**p, "powers": [["SETUP_STRIKE_POWER", 5]]})
        eng = snap(player={**p, "powers": [["TEMPORARY_STRENGTH_POWER", 5]]})
        self.assertIsNone(first([rec], [eng])["first"])  # old folds both into the base power
        r = sc.compare_fight(*fight([rec], [eng]), ROWS, engine="core3")
        self.assertEqual(r["engine"], "core3")
        self.assertIn("snap.player.powers{SETUP_STRIKE_POWER}", paths(r))
        self.assertIn("snap.player.powers{TEMPORARY_STRENGTH_POWER}", paths(r))
        e0 = snap()["enemies"][0]
        moved = snap(enemies=[{**e0, "move": "GOOP_MOVE"}])  # old has no move id (unmodelled); another engine must write it
        self.assertIsNone(first([moved], [snap()])["first"])
        self.assertEqual(paths(sc.compare_fight(*fight([moved], [snap()]), ROWS, engine="core3")), ["snap.enemies[0].move"])
        dead = snap(enemies=[{**e0, "hp": 0}, e0])  # rows for every engine still apply
        self.assertIsNone(sc.compare_fight(*fight([dead], [snap()]), ROWS, engine="core3")["first"])

    def test_cli_engine_flag(self):
        from unittest import mock  # noqa: PLC0415
        import contextlib  # noqa: PLC0415
        import io  # noqa: PLC0415
        p = snap()["player"]
        g, e = fight([snap(player={**p, "powers": [["SETUP_STRIKE_POWER", 5]]})], [snap(player={**p, "powers": [["TEMPORARY_STRENGTH_POWER", 5]]})])
        with tempfile.TemporaryDirectory() as d:
            fg, fe = Path(d) / "g.json", Path(d) / "e.json"
            fg.write_text(json.dumps({"results": [g]}), encoding="utf-8")
            fe.write_text(json.dumps({"results": [e]}), encoding="utf-8")

            def run(*extra):
                with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(sys, "argv", ["struct_compare", str(fg), str(fe), *extra]):
                    return sc.main()
            self.assertEqual(run(), 0)
            self.assertEqual(run("--engine", "core3"), 1)

    def test_intent_unmapped_kinds_differ(self):
        e0 = snap()["enemies"][0]
        summon = snap(enemies=[{**e0, "intents": [["Summon"]]}])
        escape = snap(enemies=[{**e0, "intents": [["Escape"]]}])
        self.assertEqual(paths(first([summon], [escape])), ["snap.enemies[0].intents[0]"])  # two different unmapped kinds
        self.assertIsNone(first([summon], [summon])["first"])
        mapped = snap(enemies=[{**e0, "intents": [["DebuffStrong"]]}])
        self.assertIsNone(first([mapped], [snap(enemies=[{**e0, "intents": [["Debuff"]]}])])["first"])  # a mapped kind still folds


class IntentWildcard(unittest.TestCase):
    def test_unknown_stands_for_any_unmapped_kind_only(self):
        e0 = snap()["enemies"][0]

        def it(*intents):
            return snap(enemies=[{**e0, "intents": [list(i) for i in intents]}])
        self.assertIsNone(first([it(["Sleep"])], [it(["Unknown"])])["first"])  # the old engine cannot say Sleep
        self.assertIsNone(first([it(["Escape"])], [it(["Unknown"])])["first"])
        self.assertEqual(paths(first([it(["Summon"])], [it(["Escape"])])), ["snap.enemies[0].intents[0]"])
        for mapped in (["Attack", 6, 1], ["Buff"]):  # a mapped kind is never hidden behind Unknown
            self.assertEqual(paths(first([it(mapped)], [it(["Unknown"])])), ["snap.enemies[0].intents[0]"])
        self.assertEqual(paths(sc.compare_fight(*fight([it(["Sleep"])], [it(["Unknown"])]), ROWS, engine="core3")),
                         ["snap.enemies[0].intents[0]"])  # the wildcard is the old engine's row


class CalibrateIds(unittest.TestCase):
    def test_traced_fight_ids_are_unique_across_files(self):
        from tools.fidelity import struct_calibrate as cal  # noqa: PLC0415
        a = cal.traced_fight_id("runs/x/fights/p1.packs/traces/traced.out.json", 3)
        b = cal.traced_fight_id("runs/x/fights/p2.packs/traces/traced.out.json", 3)
        self.assertNotEqual(a, b)  # the same file name and index in two runs were one id: the census gate lost fights
        self.assertEqual(a, "p1.packs/traces/traced.out.json#3")

    def test_judge_records_the_waived_paths_and_the_self_check(self):
        from tools.fidelity import struct_calibrate as cal  # noqa: PLC0415
        g, e = fight([snap()], [snap()], rec_nets=[net()], eng_nets=[net()])
        out = cal._judge("f", g, e, None, ROWS, True, True)
        self.assertIsNone(out["sc"])
        self.assertIn("net.players[*].piles.Draw[*].cost", out["waived"])
        self.assertNotIn("net.players[*].piles.Draw[*].cost", out["census"])


if __name__ == "__main__":
    unittest.main()
