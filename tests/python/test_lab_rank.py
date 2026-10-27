"""tools/lab/rank.py: the agent-weighted ranking of lockstep divergence groups, and its hook in the score-real job."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.lab import mix, rank  # noqa: E402
from tests.python.test_lab_jobs import Base  # noqa: E402


def card(c):
    return json.dumps([c, 0])


def hand_diff(a, b):
    return [[".combat.hand", [card(x) for x in a], [card(x) for x in b]]]


def rec(decision, floor, diff, type="combat", run="S1", span=0, **kw):
    r = {"decision": decision, "floor": floor, "room": type, "type": type, "gap": False, "diff": diff, "n_diff": len(diff),
         "agent_weighted": True, "reanchored": False, "run": run, "seed": run, "ai_state_assumed": False,
         "outcome": {"class": "death", "end": "death", "floor": 9}, "not_ranked_span": {"first": None, "last": None, "count": span}}
    r.update(kw)
    return r


def write_records(folder: Path, divs, decisions=1000):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "divergences.jsonl").write_text("\n".join(json.dumps(d) for d in divs), encoding="utf-8")
    (folder / "runs.jsonl").write_text(json.dumps({"ev": "end", "seed": "S1", "end": "death", "floor": 9, "asked": decisions}),
                                       encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps({"decisions": decisions}), encoding="utf-8")


HP = [[".combat.enemy_hp", [17, 10], [23, 10]]]
RUN_HP = [[".hp", 59, 61]]
SAME_SIZE = hand_diff(["BASH", "STRIKE_IRONCLAD"], ["BASH", "DEFEND_IRONCLAD"])
BIGGER = hand_diff(["BASH"], ["BASH", "DEFEND_IRONCLAD"])
# Cut from a real nightly record (an Ascender's Bane in an opening hand: the game keeps it at 0, the emulator had upgraded it).
# The same five cards on both sides; only the upgrade bit of the curse differs.
BANE_BIT = [[".combat.hand",
             ['["ASCENDERS_BANE", 0]', '["HEMOKINESIS", 1]', '["INFERNO", 1]', '["STAMPEDE", 1]', '["UPPERCUT", 1]'],
             ['["ASCENDERS_BANE", 1]', '["HEMOKINESIS", 1]', '["INFERNO", 1]', '["STAMPEDE", 1]', '["UPPERCUT", 1]']]]


class RankTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def doc(self, divs, decisions=1000):
        write_records(self.tmp / "in", divs, decisions)
        return rank.rank(self.tmp / "in")

    def test_same_cards_with_a_different_attribute_is_a_single_site(self):
        # An upgrade bit (or cost) that differs on the same cards is no shuffle, draw or roll: not rng_stream.
        self.assertEqual("single_site", rank.classify(rec(61, 14, BANE_BIT)))
        g = self.doc([rec(61, 14, BANE_BIT), rec(70, 9, BANE_BIT)])["groups"]
        self.assertEqual([("single_site", "single_site")], [(x["tag"], x["class"]) for x in g])

    def test_other_pile_differences_keep_their_class(self):
        # different cards at equal size: still a stream; the same ids in another order (a shuffle) too; a size change: queue
        self.assertEqual("rng_stream", rank.classify(rec(1, 5, SAME_SIZE)))
        order = [[".combat.draw", [card("BASH"), card("STRIKE_IRONCLAD")], [card("STRIKE_IRONCLAD"), card("BASH")]]]
        self.assertEqual("rng_stream", rank.classify(rec(1, 5, order)))
        self.assertEqual("action_queue", rank.classify(rec(1, 5, BIGGER)))
        # an attribute-only pile diff beside a real one does not hide the real one
        self.assertEqual("rng_stream", rank.classify(rec(1, 5, BANE_BIT + SAME_SIZE)))

    def test_order_weights_and_tags(self):
        divs = [rec(1, 40, HP), rec(2, 5, SAME_SIZE), rec(3, 5, SAME_SIZE, span=9), rec(5, 3, [[".combat.powers", 1, 2]])]
        g = self.doc(divs)["groups"]
        by = {x["display"]["kinds"][0]: x for x in g}
        pile = by[".combat.hand"]
        self.assertEqual(1, pile["rank"])
        self.assertEqual(["card:DEFEND_IRONCLAD", "card:STRIKE_IRONCLAD"], pile["display"]["entities"])
        self.assertAlmostEqual(1.9 + 10 * 1.9, pile["weight"], places=3)
        self.assertEqual(12 - 1, pile["decisions_affected"])
        self.assertEqual(5, pile["first_floor"])
        self.assertEqual(("architectural", "rng_stream"), (pile["tag"], pile["class"]))
        hp = by[".combat.enemy_hp"]
        self.assertEqual(("architectural", "damage_block_pipeline"), (hp["tag"], hp["class"]))
        self.assertAlmostEqual(1.2, hp["weight"], places=3)
        self.assertEqual("listener_order", by[".combat.powers"]["class"])
        self.assertEqual([1, 2, 3], [x["rank"] for x in g])
        self.assertEqual(sorted((x["weight"] for x in g), reverse=True), [x["weight"] for x in g])

    def test_pile_size_difference_is_the_action_queue(self):
        self.assertEqual("action_queue", self.doc([rec(1, 5, BIGGER)])["groups"][0]["class"])

    def test_a_selection_or_card_reward_in_a_fight_is_a_combat_decision(self):
        # enemy hp alone at the selection screen a card opened is the damage pipeline, as at `combat` (it was single_site)
        for typ in ("combat", "select_cards_combat", "card_reward_combat"):
            g = self.doc([rec(1, 5, HP, type=typ)])["groups"][0]
            self.assertEqual(("architectural", "damage_block_pipeline"), (g["tag"], g["class"]), typ)
        self.assertEqual(("single_site", "single_site"), tuple(self.doc([rec(1, 5, HP, type="select_cards")])["groups"][0][k] for k in ("tag", "class")))
        self.assertEqual("listener_order", self.doc([rec(1, 5, [[".combat.powers", 1, 2]], type="select_cards_combat")])["groups"][0]["class"])
        self.assertEqual(("combat", "select_cards_combat", "card_reward_combat"), rank.COMBAT_TYPES)

    def test_the_relic_holding_at_a_combat_decision_is_not_a_listener_order(self):
        # `.relics` is carried state (which relics the run owns), like `.deck`: an order of triggers lives in the combat block
        for typ in ("combat", "select_cards_combat"):
            for diff in ([[".relics", ["A"], ["A", "B"]]], [[".deck", [card("BASH")], [card("BASH"), card("ANGER")]]]):
                g = self.doc([rec(1, 5, diff, type=typ)])["groups"][0]
                self.assertEqual(("single_site", "single_site"), (g["tag"], g["class"]), (typ, diff[0][0]))

    def test_run_layer_state_is_single_site(self):
        g = self.doc([rec(1, 5, [[".relics", ["A"], ["A", "B"]]], type="treasure")])["groups"][0]
        self.assertEqual(("single_site", "single_site"), (g["tag"], g["class"]))

    def test_outcome_changing_proxy_and_confirmed(self):
        d = self.doc([rec(1, 50, RUN_HP, type="map"), rec(2, 50, HP)])
        by = {g["display"]["type"]: g for g in d["groups"]}
        self.assertEqual((0, 1), (by["map"]["outcome_changing_confirmed"], by["map"]["outcome_changing_proxy"]))  # carried state: proxy
        self.assertAlmostEqual(1.0, by["map"]["weight"])  # the proxy does not weigh
        self.assertEqual((0, 0), (by["combat"]["outcome_changing_confirmed"], by["combat"]["outcome_changing_proxy"]))
        self.assertEqual((0, 1), (d["summary"]["outcome_confirmed"], d["summary"]["outcome_proxy"]))
        # a repaired (re-anchored) divergence is not outcome-changing by proxy
        self.assertEqual(0, self.doc([rec(1, 50, RUN_HP, type="map", reanchored=True)])["groups"][0]["outcome_changing_proxy"])
        # an emulator outcome, when recorded, decides
        same = self.doc([rec(1, 50, HP, emu_outcome={"class": "death", "floor": 9})])["groups"][0]
        diff = self.doc([rec(1, 50, HP, emu_outcome={"class": "win", "floor": 51})])["groups"][0]
        self.assertEqual((0, 1), (same["outcome_changing_confirmed"], diff["outcome_changing_confirmed"]))
        self.assertEqual(1, self.doc([rec(1, 50, HP, emu_outcome={"class": "win"})])["summary"]["outcome_confirmed"])

    def test_old_folders_without_span_outcome_or_assumed_flag(self):
        old = rec(1, 5, SAME_SIZE)
        for k in ("outcome", "not_ranked_span", "ai_state_assumed", "run"):
            old.pop(k)
        d = self.doc([old])
        self.assertEqual(1, d["groups"][0]["decisions_affected"])

    def test_ai_state_assumed_is_excluded_and_counted(self):
        d = self.doc([rec(1, 5, HP, ai_state_assumed=True), rec(2, 5, HP, ai_state_assumed=True), rec(3, 30, SAME_SIZE)])
        self.assertEqual(2, d["summary"]["ai_state_assumed"])
        self.assertEqual(1, d["summary"]["groups"])
        self.assertEqual(1, d["summary"]["state_divergences"])
        self.assertAlmostEqual(0.001, d["summary"]["state_divergence_rate"])
        self.assertEqual([], [g for g in d["groups"] if g["display"]["kinds"] == [".combat.enemy_hp"]])

    def test_gaps_are_counted_apart(self):
        d = self.doc([rec(1, 5, [], type="event", gap=True, reason="unmapped:event"), rec(2, 5, HP)])
        self.assertEqual((1, 1), (d["summary"]["mapping_gaps"], d["summary"]["groups"]))
        self.assertEqual({"unmapped:event": 1}, d["summary"]["gap_reasons"])

    def test_arch_share_and_rate(self):
        # architectural: hp group at floor 50 = 1.0; single site: relics at floor 50 = 1.0 (a proxy does not weigh)
        d = self.doc([rec(1, 50, HP), rec(2, 50, [[".relics", ["A"], ["B"]]], type="treasure")], decisions=200)
        s = d["summary"]
        self.assertAlmostEqual(0.5, s["arch_share"], places=3)
        self.assertAlmostEqual(0.01, s["state_divergence_rate"])

    def emu_arm(self, *rows):
        d = self.tmp / "emu"
        d.mkdir(exist_ok=True)
        (d / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return d

    def test_the_paired_emulator_arm_is_joined_by_seed_into_the_outcome(self):
        write_records(self.tmp / "in", [rec(1, 50, RUN_HP, type="map"), rec(2, 50, HP)])  # the referee run S1 died on floor 9
        won = self.emu_arm({"seed": "S1", "win": True, "floor": 51, "end": "win"}, {"seed": "S9", "win": False, "floor": 3, "end": "death"})
        d = rank.rank(self.tmp / "in", won)
        by = {g["display"]["type"]: g for g in d["groups"]}
        self.assertEqual((1, 0, 1.5), (by["map"]["outcome_changing_confirmed"], by["map"]["outcome_changing_proxy"], by["map"]["weight"]))
        self.assertEqual((1, 1.5), (by["combat"]["outcome_changing_confirmed"], by["combat"]["weight"]))
        s = d["summary"]
        self.assertEqual((2, 1), (s["outcome_confirmed"], s["emu_runs_joined"]))
        self.assertIn("paired emulator arm", s["outcome_rule"])
        self.assertIn("Outcome weighting: confirmed", rank.top_md(d))
        # the same end on both engines: nothing is confirmed, and the proxy no longer applies (the arm decides)
        same = rank.rank(self.tmp / "in", self.emu_arm({"seed": "S1", "win": False, "floor": 9, "end": "death"}))
        self.assertEqual((0, 0), (same["summary"]["outcome_confirmed"], same["summary"]["outcome_proxy"]))
        # no arm: the proxy rule as before, and no outcome keys in the summary (the golden bytes do not change)
        none = rank.rank(self.tmp / "in")
        self.assertEqual((0, 1), (none["summary"]["outcome_confirmed"], none["summary"]["outcome_proxy"]))
        self.assertNotIn("outcome_rule", none["summary"])
        self.assertNotIn("Outcome weighting", rank.top_md(none))
        # a seed the arm lacks falls back to the proxy
        other = rank.rank(self.tmp / "in", self.emu_arm({"seed": "S9", "end": "death", "floor": 3}))
        self.assertEqual((0, 1, 0), (other["summary"]["outcome_confirmed"], other["summary"]["outcome_proxy"], other["summary"]["emu_runs_joined"]))

    def test_the_cli_takes_the_emulator_arm_and_the_folder_names_its_own(self):
        write_records(self.tmp / "in", [rec(1, 50, HP)])
        arm = self.emu_arm({"seed": "S1", "win": True, "floor": 51, "end": "win"})
        self.assertIsNone(rank.paired_emu_dir(self.tmp / "in"))
        self.assertEqual(0, rank.main([str(self.tmp / "in"), "--out", str(self.tmp / "out"), "--emu-dir", str(arm)]))
        self.assertEqual(1, json.loads((self.tmp / "out" / "ranked.json").read_text())["summary"]["outcome_confirmed"])
        self.assertIn("Outcome weighting: confirmed", (self.tmp / "out" / "top20.md").read_text())
        for name in ("summary.json", "compare.json"):
            (self.tmp / "in" / "summary.json").write_text(json.dumps({"decisions": 1000}), encoding="utf-8")
            (self.tmp / "in" / name).write_text(json.dumps({"emu_arm": {"dir": arm.as_posix()}}), encoding="utf-8")
            self.assertEqual(arm, rank.paired_emu_dir(self.tmp / "in"), name)
        (self.tmp / "in" / "compare.json").write_text(json.dumps({"emu_arm": {"dir": "nowhere"}}), encoding="utf-8")
        (self.tmp / "in" / "summary.json").write_text(json.dumps({"decisions": 1000}), encoding="utf-8")
        self.assertIsNone(rank.paired_emu_dir(self.tmp / "in"))

    def test_cli_writes_the_files_and_a_missing_folder_is_empty(self):
        write_records(self.tmp / "in", [rec(1, 5, SAME_SIZE)])
        self.assertEqual(0, rank.main([str(self.tmp / "in"), "--out", str(self.tmp / "out")]))
        doc = json.loads((self.tmp / "out" / "ranked.json").read_text())
        self.assertEqual(1, doc["summary"]["groups"])
        self.assertIn("| 1 |", (self.tmp / "out" / "top20.md").read_text())
        self.assertEqual(0, rank.rank(self.tmp / "nothing")["summary"]["groups"])

    def test_confirmed_and_proxy_are_kept_apart_and_only_confirmed_weighs(self):
        d = self.doc([rec(1, 50, RUN_HP, type="map"), rec(2, 50, HP, emu_outcome={"class": "win"})])
        by = {g["display"]["type"]: g for g in d["groups"]}
        self.assertEqual((0, 1, 1.0), (by["map"]["outcome_changing_confirmed"], by["map"]["outcome_changing_proxy"], by["map"]["weight"]))
        self.assertEqual((1, 0, 1.5), (by["combat"]["outcome_changing_confirmed"], by["combat"]["outcome_changing_proxy"], by["combat"]["weight"]))
        md = rank.top_md(d)
        self.assertIn("outcome-changing confirmed", md)
        self.assertIn("outcome-changing (proxy)", md)

    def test_group_key_is_the_labs(self):
        from tools.fidelity import lab_groups as lg
        r = rec(1, 5, SAME_SIZE)
        sym = {"source": "walk", "phase": "combat", "field": ".combat.hand", "first_event": None,
               "ingredients": ["card:DEFEND_IRONCLAD", "card:STRIKE_IRONCLAD"], "implicated": []}
        lab = lg.group_key({"keys": [lg.base_key(sym)], "mechanism": []})
        self.assertEqual(lab, rank.group_key(r))
        self.assertEqual(lg.stable_id(lab), self.doc([r])["groups"][0]["id"])


class JobHookTest(Base):
    def test_a_score_real_job_writes_the_ranking_and_the_yield_columns(self):
        recs = self.tmp / "records"
        write_records(recs, [rec(1, 5, SAME_SIZE), rec(2, 30, HP)], decisions=500)
        code = "import sys, shutil; shutil.copytree(sys.argv[1], sys.argv[2], dirs_exist_ok=True)"
        self.config(**{"score-real": {"cmd": ["{python}", "-c", code, str(recs), "{job}"]}})
        self.marker("score/pending")
        r = self.runner()
        run = r.new_run("score-real", "abc")
        self.assertIsNone(r.execute(run))
        self.assertEqual(2, len(json.loads((run.dir / "ranked.json").read_text())["groups"]))
        self.assertIn("| rank |", (run.dir / "top20.md").read_text())
        row = [x for x in mix.read_yield(self.lab_dir) if x["strategy"] == "score-real"][0]
        self.assertEqual(("2", "2"), (row["divergences"], row["groups"]))
        self.assertNotEqual("NA", row["arch_share"])
        self.assertTrue((self.lab_dir / "score" / "last-ok").exists())

    def test_the_hook_joins_the_emulator_arm_the_summary_names(self):
        recs, arm = self.tmp / "records", self.tmp / "emu-arm"
        write_records(recs, [rec(1, 5, SAME_SIZE), rec(2, 30, HP)], decisions=500)
        arm.mkdir()
        (arm / "runs.jsonl").write_text(json.dumps({"seed": "S1", "win": True, "floor": 51, "end": "win"}) + "\n", encoding="utf-8")
        (recs / "summary.json").write_text(json.dumps({"decisions": 500, "emu_arm": {"dir": arm.as_posix()}}), encoding="utf-8")
        code = "import sys, shutil; shutil.copytree(sys.argv[1], sys.argv[2], dirs_exist_ok=True)"
        self.config(**{"score-real": {"cmd": ["{python}", "-c", code, str(recs), "{job}"]}})
        self.marker("score/pending")
        r = self.runner()
        run = r.new_run("score-real", "abc")
        self.assertIsNone(r.execute(run))
        s = json.loads((run.dir / "ranked.json").read_text())["summary"]
        self.assertEqual((2, 1), (s["outcome_confirmed"], s["emu_runs_joined"]))


class StallsTest(unittest.TestCase):
    """L3: the run rows' longest identical (screen, action) streak and the timeout end make a stall visible the morning after."""

    ROWS = [
        {"ev": "end", "seed": "A", "end": "death", "floor": 12, "streak": {"n": 50, "screen": "event", "action": 0}},
        {"ev": "end", "seed": "B", "end": "death", "floor": 4, "streak": {"n": 412, "screen": "event", "action": 0}},
        {"ev": "end", "seed": "C", "end": "timeout", "floor": 7},
        {"ev": "crash", "seed": "D", "error": "TimeoutError", "streak": {"n": 90, "screen": "map", "action": 1}},
        {"ev": "net", "w": 0, "conns": []},
        {"ev": "end", "seed": "E", "end": "win", "floor": 51, "streak": {"n": 3, "screen": "map", "action": 0}},
        {"ev": "end", "seed": "F", "end": "timeout", "floor": 4, "streak": {"n": 900, "screen": "event", "action": 0}},
    ]

    def test_streaks_over_fifty_and_timeouts_are_listed_with_the_key(self):
        got = rank.stalls(self.ROWS)
        self.assertEqual(["B", "C", "D", "F"], [s["seed"] for s in got])
        self.assertEqual({"seed": "B", "ev": "end", "end": "death", "floor": 4, "streak": {"n": 412, "screen": "event", "action": 0},
                          "arm": "real", "why": ["streak"]}, got[0])
        self.assertEqual((["timeout"], None), (got[1]["why"], got[1]["streak"]))  # a row from before the field has no streak
        self.assertEqual((["streak"], None), (got[2]["why"], got[2]["floor"]))
        self.assertEqual(["timeout", "streak"], got[3]["why"])

    def test_no_stall_is_an_empty_list_and_the_threshold_is_a_parameter(self):
        self.assertEqual([], rank.stalls([self.ROWS[0], self.ROWS[5]]))
        self.assertEqual(["A", "B", "C", "D", "F"], [s["seed"] for s in rank.stalls(self.ROWS, min_streak=49)])

    EMU = [{"seed": "E1", "win": False, "floor": 4, "end": "timeout"}, {"seed": "E2", "win": False, "floor": 12, "end": "death"},
           {"seed": "E3", "win": False, "floor": 0, "crash": True, "end": "?"}]  # gen0.run_block rows: no `ev` key

    def test_the_emulator_arms_timeouts_are_listed_with_their_arm(self):
        self.assertEqual([{"seed": "E1", "ev": None, "end": "timeout", "floor": 4, "streak": None, "arm": "emu", "why": ["timeout"]}],
                         rank.stalls(self.EMU, arm="emu"))
        self.assertEqual([], rank.stalls(self.EMU))  # a row without ev is not a real-arm row
        self.assertEqual([], rank.stalls(self.ROWS, arm="emu"))  # and a real-arm row is not an emulator row
        self.assertEqual({"real"}, {s["arm"] for s in rank.stalls(self.ROWS)})

    def test_the_summary_adds_the_paired_emulator_arms_stalls(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(d, ignore_errors=True))
        write_records(d / "in", [])
        (d / "in" / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in self.ROWS), encoding="utf-8")
        (d / "emu").mkdir()
        (d / "emu" / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in self.EMU), encoding="utf-8")
        got = rank.rank(d / "in", d / "emu")["summary"]["stalls"]
        self.assertEqual([("real", "B"), ("real", "C"), ("real", "D"), ("real", "F"), ("emu", "E1")], [(s["arm"], s["seed"]) for s in got])
        self.assertIn("emu E1 floor 4", rank.top_md(rank.rank(d / "in", d / "emu")))
        self.assertEqual(["B", "C", "D", "F"], [s["seed"] for s in rank.rank(d / "in")["summary"]["stalls"]])  # no arm given: real only

    def test_the_ranking_summary_lists_the_stalls(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(d, ignore_errors=True))
        write_records(d / "in", [])
        (d / "in" / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in self.ROWS), encoding="utf-8")
        doc = rank.rank(d / "in")
        self.assertEqual(["B", "C", "D", "F"], [s["seed"] for s in doc["summary"]["stalls"]])
        self.assertIn("Stalled runs", rank.top_md(doc))


if __name__ == "__main__":
    unittest.main()
