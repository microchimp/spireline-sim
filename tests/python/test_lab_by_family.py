"""tools/lab/by_family.py: the divergence-by-hook-family table, its rule table, and its hook in tools/lab/rank.py write()."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.lab import by_family as bf, rank  # noqa: E402
from tests.python.test_lab_rank import BIGGER, HP, RUN_HP, SAME_SIZE, card, hand_diff, rec, write_records  # noqa: E402

# sha256 of ranked.json and top20.md that tools/lab/rank.py (97b03d1e, before by_family) wrote for fixture_divs(), over the text with
# line endings normalised to LF (write_text writes CRLF on Windows; the first golden was the CRLF hash, which failed anywhere else).
GOLDEN = {"ranked.json": "a129adeb40c1550dadffc2ab16b3e6e40f831f80b5ed166e3a6d91df607ccb0b",
          "top20.md": "04606dfb7e7ada85e1e09ff0b34b521ace575b6ad0bdf03678f82c8ea1868ab9"}

UP = [[".combat.hand", [card("ASCENDERS_BANE")], [json.dumps(["ASCENDERS_BANE", 1])]]]
CORPSE = [[".combat.enemy_hp", [0, 66], [66]]]  # the referee lists a 0-hp enemy the emulator does not
EMU_CORPSE = [[".combat.enemy_hp", [66], [0, 66]]]  # the emulator lists a 0-hp enemy the referee does not (the shape of the S2 block)
LIVING_EXTRA = [[".combat.enemy_hp", [20, 30], [20]]]
GAP_HAND = hand_diff(["BASH"], ["BASH", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD"])  # the emulator holds two cards more
PLAY = {"type": "play", "kind": "combat", "label": ["play", 0, "HEADBUTT", 1]}
END = {"type": "end_turn", "kind": "combat", "label": ["end"]}
POTION = {"type": "potion", "kind": "combat", "label": ["potion", 1, "FORTIFIER", -1]}
CHOICE = {"type": "choice", "kind": "select_cards", "label": ["HEADBUTT", 0]}


def fixture_divs():
    return [rec(1, 40, HP), rec(2, 5, SAME_SIZE), rec(3, 5, SAME_SIZE, span=9), rec(5, 3, [[".combat.powers", 1, 2]]),
            rec(6, 4, BIGGER), rec(7, 8, [], type="reward", reason="phase:0!=rewards"), rec(8, 6, [[".gold", 100, 103]], type="reward"),
            rec(9, 9, SAME_SIZE + RUN_HP)]


def group(gid, weight, tag, type, diff, reason=None, divergences=1, aff=3, confirmed=0, rk=1):
    kinds = sorted({rank._kind(p) for p, _, _ in diff}) or ["reason:" + str(reason)]
    return {"id": gid, "rank": rk, "tag": tag, "class": "x", "weight": weight, "decisions_affected": aff, "divergences": divergences,
            "first_floor": 3, "outcome_changing_confirmed": confirmed, "outcome_changing_proxy": 0,
            "display": {"type": type, "kinds": kinds, "entities": []}, "samples": [{"run": "S1", "decision": 1, "floor": 3, "type": type, "diff": diff}]}


def synthetic_doc():
    gs = [group("Ga", 50.0, "architectural", "combat", SAME_SIZE, aff=20, confirmed=1),
          group("Gb", 30.0, "architectural", "combat", BIGGER, aff=10),
          group("Gc", 10.0, "architectural", "combat", HP, aff=4, confirmed=1),
          group("Gd", 8.0, "single_site", "reward", [], reason="phase:0!=rewards", aff=5),
          group("Ge", 2.0, "single_site", "reward", [[".gold", 100, 103]], aff=1),
          group("Gf", 4.0, "architectural", "combat", SAME_SIZE + RUN_HP, aff=2)]
    for i, g in enumerate(sorted(gs, key=lambda g: -g["weight"]), 1):
        g["rank"] = i
    return {"summary": {"total_weight": 104.0, "arch_share": 0.9038}, "groups": sorted(gs, key=lambda g: -g["weight"])}


class RuleTableTest(unittest.TestCase):
    def check(self, r, rule, family, evidence=None):
        a = bf.assign(r)
        self.assertEqual((rule, family), (a["rule"], a["family"]), r.get("diff"))
        if evidence:
            self.assertEqual(evidence, a["evidence"])
        return a

    def test_each_rule_places_its_shape(self):
        self.check(rec(1, 3, [], type="reward", reason="phase:0!=rewards"), "R01", "F10", "reason")
        self.check(rec(1, 3, [], type="rest", reason="screen:map!=rest"), "R02", "F16", "reason")
        self.check(rec(1, 3, RUN_HP, type="reward"), "R03", "F17", "diff")
        self.check(rec(1, 3, CORPSE), "R04b", "F10", "shape")
        self.check(rec(1, 3, [[".combat.enemy_hp", [999999999], [32768]]]), "R05", bf.UNASSIGNED)
        self.check(rec(1, 3, HP), "R06", "F08", "diff")
        self.check(rec(1, 3, RUN_HP), "R06", "F08")
        self.check(rec(1, 3, HP + RUN_HP), "R06", "F08")
        self.check(rec(1, 3, [[".combat.block", 5, 8]]), "R07", "F09")
        self.check(rec(1, 3, hand_diff(["BASH", "STRIKE_IRONCLAD"], ["STRIKE_IRONCLAD", "BASH"])), "R08", "F11", "shape")
        self.check(rec(1, 3, UP), "R09", bf.UNASSIGNED, "none")
        a = self.check(rec(1, 3, SAME_SIZE), "R10", "F11", "shape")
        self.assertEqual("RNG order", a["coarse"])
        a = self.check(rec(1, 3, BIGGER), "R11", "F11")
        self.assertEqual("exhaust and discard triggers", a["coarse"])
        self.check(rec(1, 3, hand_diff(["BASH"], ["BASH", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD"])), "R12", bf.UNASSIGNED)
        self.check(rec(1, 3, hand_diff(["BASH", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD"], ["BASH"])), "R12", bf.UNASSIGNED)
        self.check(rec(1, 3, SAME_SIZE + RUN_HP), "R13", bf.UNASSIGNED)
        self.check(rec(1, 3, [[".gold", 100, 103]], type="reward"), "R14", "F15", "type")
        self.check(rec(1, 3, [[".potions", ["A"], ["B"]]], type="card_reward"), "R15", "F15")
        self.check(rec(1, 3, [[".deck", [card("BASH")], [card("BASH"), card("ANGER")]]]), "R16", "F15")
        self.check(rec(1, 3, [[".deck", [card("BASH")], [card("ANGER")]]], type="select_cards"), "R17", "F16", "type")
        self.check(rec(1, 3, [[".x", 1, 2]], type="weird"), "R18", bf.UNASSIGNED, "none")

    def test_combat_selection_screen_uses_the_combat_rules(self):
        self.check(rec(1, 3, SAME_SIZE, type="select_cards_combat"), "R10", "F11")
        self.check(rec(1, 3, HP, type="card_reward_combat"), "R06", "F08")
        self.check(rec(1, 3, [[".potions", ["A"], ["B"]]], type="card_reward_combat"), "R15", "F15")
        self.assertIs(rank.COMBAT_TYPES, bf.COMBAT_TYPES)  # one list for the tag and the family rules

    def test_a_corpse_is_placed_by_the_side_that_holds_it_and_the_step_before(self):
        a = self.check(rec(1, 3, EMU_CORPSE, type="select_cards_combat", acted=PLAY), "R04a", "F10", "shape")
        self.assertEqual(("HEADBUTT", "death and victory"), (a["ingredient"], a["coarse"]))
        self.check(rec(1, 3, EMU_CORPSE, acted=PLAY), "R04a", "F10", "shape")  # at combat too
        self.check(rec(1, 3, CORPSE, acted=PLAY), "R04b", "F10", "shape")  # the referee side: a removal refusal, whatever the step
        self.check(rec(1, 3, CORPSE, acted=END), "R04b", "F10", "shape")
        # not placed: the emulator's corpse after an end turn or with no acted; a living entry more on one side
        self.check(rec(1, 3, EMU_CORPSE, acted=END), "R04", bf.UNASSIGNED, "none")
        self.check(rec(1, 3, EMU_CORPSE), "R04", bf.UNASSIGNED, "none")
        self.check(rec(1, 3, LIVING_EXTRA, acted=PLAY), "R04", bf.UNASSIGNED, "none")
        self.assertEqual(("emu", "ref", None, None), (bf.corpse_side([66], [0, 66]), bf.corpse_side([0, 66], [66]),
                                                      bf.corpse_side([20, 30], [20]), bf.corpse_side([66], [66])))

    def test_phase_and_screen_reasons(self):
        self.check(rec(1, 3, [], type="reward", reason="phase:0!=rewards"), "R01", "F10", "reason")
        self.check(rec(1, 3, [], type="rest", reason="screen:None!=rest"), "R01", "F10", "reason")  # was R02, run flow
        self.check(rec(1, 3, [], type="combat", reason="phase:6"), "R01b", "F10", "reason")  # the emulator's player is dead
        self.check(rec(1, 3, [], type="rest", reason="screen:map!=rest"), "R02", "F16", "reason")
        self.check(rec(1, 3, [], type="event", reason="event:A!=B"), "R02", "F16", "reason")

    def test_hand_gaps_are_split_by_the_step_before(self):
        for gap, ids in ((GAP_HAND, ("R12a", "R12b", "R12c", "R12")), (SAME_SIZE + RUN_HP, ("R13a", "R13b", "R13c", "R13"))):
            a = self.check(rec(1, 3, gap, acted=END), ids[0], "F02-F05", "shape")
            self.assertEqual(("start- and end-of-turn hooks", None), (a["coarse"], a["ingredient"]))
            a = self.check(rec(1, 3, gap, acted=PLAY), ids[1], "F06", "shape")
            self.assertEqual(("card-play hooks", "HEADBUTT"), (a["coarse"], a["ingredient"]))  # the card is the ingredient
            a = self.check(rec(1, 3, gap, acted=POTION), ids[2], "F14", "shape")
            self.assertEqual("FORTIFIER", a["ingredient"])
            self.check(rec(1, 3, gap, acted=CHOICE), ids[3], bf.UNASSIGNED, "none")  # a step the split does not name
            self.check(rec(1, 3, gap), ids[3], bf.UNASSIGNED, "none")  # a record with no acted
        self.check(rec(1, 3, SAME_SIZE, acted=END), "R10", "F11", "shape")  # the equal-size and off-by-one rules do not read acted
        self.check(rec(1, 3, BIGGER, acted=PLAY), "R11", "F11", "shape")

    def test_every_rule_has_a_matcher_a_known_family_and_a_reason(self):
        ids = [r[0] for r in bf.RULES]
        self.assertEqual(sorted(set(ids)), sorted(ids))
        for rid, matcher, family, coarse, evidence, reason in bf.RULES:
            self.assertTrue(family in bf.FAMILIES or family == bf.UNASSIGNED, rid)
            self.assertTrue(coarse in bf.COARSE or (family == bf.UNASSIGNED and not coarse), rid)
            self.assertIn(evidence, bf.EVIDENCE)
            self.assertTrue(reason and matcher, rid)
        self.assertEqual(["R01b", "R04a", "R04b", "R12a", "R12b", "R12c", "R13a", "R13b", "R13c"], list(bf.SPLIT_RULES))


class TableTest(unittest.TestCase):
    def table(self):
        return bf.table(bf.parts_from_ranked(synthetic_doc()), synthetic_doc()["groups"])

    def test_family_table_and_cumulative_curve(self):
        t = self.table()
        fams = [(r["family"], r["weight"]) for r in t["families"]]
        self.assertEqual([("F11", 80.0), ("F08", 10.0), ("F10", 8.0), ("F15", 2.0)], fams)
        self.assertEqual([0.7692, 0.8654, 0.9423, 0.9615], [r["cum_share"] for r in t["families"]])
        f11, f08, f10, f15 = t["families"]
        self.assertEqual((1.0, 80.0, 0.8511), (f11["arch_share_within"], f11["arch_weight"], f11["arch_share_of_arch"]))
        self.assertEqual((0.0, 0.0), (f10["arch_share_within"], f10["arch_weight"]))
        self.assertEqual((2, 30, 1), (f11["groups"], f11["decisions_affected"], f11["outcome_changing_confirmed"]))
        self.assertEqual(1, f08["outcome_changing_confirmed"])
        self.assertEqual("RNG order", f11["coarse"])
        self.assertEqual(104.0, t["summary"]["total_weight"])
        self.assertEqual(94.0, t["summary"]["architectural_weight"])

    def test_top_five_share_of_the_architectural_weight(self):
        s = self.table()["summary"]
        self.assertEqual(["F11", "F08"], s["top5_arch"]["families"])  # F10 and F15 hold no architectural weight
        self.assertEqual(0.9574, s["top5_arch"]["share_of_arch"])
        self.assertEqual(1.0, s["top5_arch_upper_bound"])
        self.assertTrue(s["verdict"].startswith("Tier 1"))
        self.assertEqual(0.9574, s["readings"]["B: the draw is F03"]["share_of_arch"])
        self.assertEqual(["F03", "F08"], s["readings"]["B: the draw is F03"]["families"])

    def test_verdict_thresholds(self):
        self.assertTrue(bf.verdict(0.70).startswith("Tier 1"))
        self.assertTrue(bf.verdict(0.6999).startswith("between"))
        self.assertTrue(bf.verdict(0.50).startswith("between"))
        self.assertTrue(bf.verdict(0.4999).startswith("Tier 2"))

    def test_unassigned_is_its_own_bucket_not_a_family(self):
        t = self.table()
        u = t["unassigned"]
        self.assertNotIn(bf.UNASSIGNED, [r["family"] for r in t["families"]])
        self.assertEqual((4.0, 0.0385, 1), (u["weight"], u["share"], u["groups"]))
        self.assertEqual({"R13": 4.0}, u["by_rule"])
        self.assertEqual(round(1 - 4 / 104, 4), t["summary"]["assigned_share"])
        self.assertEqual(round(4 / 94, 4), u["arch_share_of_arch"])
        self.assertEqual(104.0, round(sum(r["weight"] for r in t["families"]) + u["weight"], 3))

    def test_single_site_groups_are_listed_apart_and_ranked(self):
        t = self.table()
        self.assertEqual([("Gd", "F10", "R01"), ("Ge", "F15", "R14")], [(r["id"], r["family"], r["rule"]) for r in t["single_site"]])

    def test_the_verdict_carries_its_bounds_and_is_undecidable_when_they_straddle(self):
        self.assertTrue(bf.bounded_verdict(0.80, 1.0).startswith("Tier 1"))
        self.assertTrue(bf.bounded_verdict(0.30, 0.45).startswith("Tier 2"))
        self.assertTrue(bf.bounded_verdict(0.55, 0.69).startswith("between"))
        v = bf.bounded_verdict(0.366, 1.0)  # the review's reading: R04 unassigned
        self.assertTrue(v.startswith("undecidable"), v)
        self.assertTrue("36.6%" in v and "100.0%" in v and "50%" in v and "70%" in v, v)
        v = bf.bounded_verdict(0.60, 1.0)
        self.assertTrue(v.startswith("undecidable") and "70%" in v and "50%" not in v, v)
        v = bf.bounded_verdict(0.40, 0.60)
        self.assertTrue(v.startswith("undecidable") and "50%" in v and "70%" not in v, v)
        self.assertIn("lower bound 80.0%", bf.bounded_verdict(0.80, 1.0))

    def test_the_table_reports_both_bounds_and_says_undecidable_when_they_straddle(self):
        s = self.table()["summary"]  # lower 95.7%, upper 100%: one side of both thresholds
        self.assertEqual((0.9574, 1.0, True), (s["top5_arch_lower_bound"], s["top5_arch_upper_bound"], s["verdict_decided"]))
        gs = [group("Ga", 30.0, "architectural", "combat", SAME_SIZE), group("Gb", 70.0, "architectural", "combat", SAME_SIZE + RUN_HP)]
        t = bf.build(None, {"groups": gs})  # Gb has no acted: R13, unassigned
        s = t["summary"]
        self.assertEqual((0.3, 1.0, False), (s["top5_arch_lower_bound"], s["top5_arch_upper_bound"], s["verdict_decided"]))
        self.assertTrue(s["verdict"].startswith("undecidable"))
        self.assertIn("undecidable", bf.md(t))
        self.assertIn("between 30.0%", bf.md(t))

    def test_a_reading_can_flip_the_verdict(self):
        gs = [group("Ga", 50.0, "architectural", "combat", SAME_SIZE), group("Gb", 40.0, "architectural", "combat", CORPSE),
              group("Gc", 10.0, "architectural", "combat", HP)]
        s = bf.table(bf.parts_from_ranked({"groups": gs}), gs)["summary"]
        self.assertEqual((1.0, ["F11", "F10", "F08"]), (s["top5_arch"]["share_of_arch"], s["top5_arch"]["families"]))  # R04b is F10
        self.assertTrue(s["verdict"].startswith("Tier 1"))
        c = s["readings"]["C: a listed corpse is the comparer's"]  # without the corpse nothing is known about 40% of the weight
        self.assertEqual((0.6, 1.0, ["F11", "F08"]), (c["share_of_arch"], c["upper_share_of_arch"], c["families"]))
        self.assertTrue(c["verdict"].startswith("undecidable"))

    def test_empty_input(self):
        t = bf.build(None, rank.rank(Path(tempfile.gettempdir()) / "no-such-score-real-dir"))
        self.assertEqual(0.0, t["summary"]["total_weight"])
        self.assertEqual([], t["families"])
        self.assertIn("unassigned", bf.md(t))


class RecordsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def folder(self, divs):
        write_records(self.tmp / "in", divs)
        return self.tmp / "in"

    def test_a_group_that_mixes_is_split_by_divergence_and_weights_reconcile(self):
        f = self.folder([rec(1, 5, LIVING_EXTRA), rec(2, 5, [[".combat.enemy_hp", [20], [21]]], span=3), rec(3, 5, SAME_SIZE)])
        doc = rank.rank(f)
        self.assertEqual(2, len(doc["groups"]))  # the two enemy hp divergences share a group key
        t = bf.build(f, doc)
        mixed = [g for g in t["groups"] if g["mixed"]]
        self.assertEqual(1, len(mixed))
        self.assertEqual(["F08", bf.UNASSIGNED], mixed[0]["families"])
        self.assertEqual("divergences", t["summary"]["basis"])
        self.assertEqual(0.0, t["summary"]["weight_check_max_abs_error"])
        self.assertEqual(round(doc["summary"]["total_weight"], 3), t["summary"]["total_weight"])
        by = {r["family"]: r for r in t["families"]}
        self.assertEqual(4, by["F08"]["decisions_affected"])

    def test_records_and_samples_agree_on_a_single_shape_group(self):
        f = self.folder([rec(1, 5, SAME_SIZE), rec(2, 6, HP), rec(3, 7, [], type="reward", reason="phase:0!=rewards")])
        doc = rank.rank(f)
        a, b = bf.build(f, doc), bf.build(None, doc)
        self.assertEqual("group samples", b["summary"]["basis"])
        self.assertEqual([(r["family"], r["weight"]) for r in a["families"]], [(r["family"], r["weight"]) for r in b["families"]])

    def test_the_emulator_arm_reaches_the_weights_and_the_report_says_which_rule_was_used(self):
        f = self.folder([rec(1, 50, RUN_HP, type="map"), rec(2, 50, HP)])
        arm = self.tmp / "emu"
        arm.mkdir()
        (arm / "runs.jsonl").write_text(json.dumps({"seed": "S1", "win": True, "floor": 51, "end": "win"}) + "\n", encoding="utf-8")
        doc = rank.rank(f, arm)
        t = bf.build(f, doc)
        self.assertEqual(3.0, doc["summary"]["total_weight"])  # both confirmed: 1.5 each
        self.assertEqual(0.0, t["summary"]["weight_check_max_abs_error"])  # by_family joins the same arm
        self.assertEqual(2, sum(r["outcome_changing_confirmed"] for r in t["families"]))
        self.assertIn("paired emulator arm", t["summary"]["outcome_rule"])
        self.assertIn("Outcome weighting: confirmed", bf.md(t))
        plain = bf.build(f, rank.rank(f))
        self.assertTrue(plain["summary"]["outcome_rule"].startswith("proxy"))
        self.assertIn("Outcome weighting: proxy", bf.md(plain))
        self.assertEqual(0, bf.main([str(f), "--out", str(self.tmp / "cli"), "--emu-dir", str(arm)]))
        self.assertIn("Outcome weighting: confirmed", (self.tmp / "cli" / "by_family.md").read_text(encoding="utf-8"))

    def test_the_corpse_after_a_play_is_the_games_rule_and_reading_c_is_the_bound(self):
        text = {r[0]: r[5] for r in bf.RULES}["R04a"]
        self.assertIn("M0 game trace", text)
        self.assertIn("primary reading", text)
        self.assertIn("bound", bf.READINGS["C: a listed corpse is the comparer's"][1])

    def test_acted_turn_and_encounter_are_read_from_the_records(self):
        f = self.folder([rec(1, 5, EMU_CORPSE, type="select_cards_combat", acted=PLAY, turn=2, encounter="SLIMES"),
                         rec(2, 5, GAP_HAND, acted=END, turn=3, encounter="SLIMES"), rec(3, 5, GAP_HAND, acted=PLAY, turn=5, encounter="NIBBITS"),
                         rec(4, 5, GAP_HAND, acted=CHOICE, turn=6)])
        t = bf.build(f, rank.rank(f))
        rows = {r["rule"]: r for r in t["placed_by_split"]}
        self.assertEqual({"R04a", "R12a", "R12b"}, set(rows))
        self.assertEqual(({"HEADBUTT": 1.9}, {"SLIMES": 1.9}, 2), (rows["R04a"]["ingredients"], rows["R04a"]["encounters"], rows["R04a"]["median_turn"]))
        self.assertEqual(({"select_cards_combat": 1.9}, {"play": 1.9}), (rows["R04a"]["by_type"], rows["R04a"]["by_acted"]))
        self.assertEqual(({"HEADBUTT": 1.9}, {"NIBBITS": 1.9}, 5), (rows["R12b"]["ingredients"], rows["R12b"]["encounters"], rows["R12b"]["median_turn"]))
        self.assertEqual(5.7, t["summary"]["split_placed_weight"])
        self.assertEqual({"shape": 5.7}, t["summary"]["split_placed_by_evidence"])
        by = {r["family"]: r for r in t["families"]}
        self.assertEqual({"HEADBUTT": 1.9}, by["F06"]["ingredients"])
        self.assertEqual({"R12": 1.9}, t["unassigned"]["by_rule"])
        self.assertEqual([("R12", "choice", "combat", 1.9)], [(r["rule"], r["acted"], r["type"], r["weight"]) for r in t["unassigned"]["by_shape"]])
        text = bf.md(t)
        self.assertIn("## Placed by the acted-type splits", text)
        self.assertIn("HEADBUTT 1.9", text)


class RankHookTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        write_records(self.tmp / "in", fixture_divs())

    def test_rank_outputs_are_unchanged_and_by_family_is_written_beside_them(self):
        doc = rank.write(self.tmp / "in", self.tmp / "out")
        out = self.tmp / "out"
        for name, digest in GOLDEN.items():
            self.assertEqual(digest, hashlib.sha256((out / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), name)
        md = (out / "by_family.md").read_text(encoding="utf-8")
        self.assertTrue(md.startswith("# Divergence by hook family"))
        js = json.loads((out / "by_family.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["summary"]["total_weight"], js["summary"]["total_weight"])
        self.assertEqual(doc["summary"]["arch_share"], js["summary"]["arch_share"])
        self.assertFalse((out / "by_family.error.txt").exists())

    def test_a_by_family_failure_never_costs_the_ranking(self):
        real = bf.write

        def boom(*a, **k):
            raise ValueError("x")
        bf.write = boom
        try:
            rank.write(self.tmp / "in", self.tmp / "out")
        finally:
            bf.write = real
        self.assertTrue((self.tmp / "out" / "ranked.json").exists() and (self.tmp / "out" / "top20.md").exists())
        self.assertEqual("ValueError", (self.tmp / "out" / "by_family.error.txt").read_text())

    def test_cli(self):
        self.assertEqual(0, bf.main([str(self.tmp / "in"), "--out", str(self.tmp / "cli")]))
        self.assertTrue((self.tmp / "cli" / "by_family.md").exists())


if __name__ == "__main__":
    unittest.main()
