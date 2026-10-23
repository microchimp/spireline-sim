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


class RankTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def doc(self, divs, decisions=1000):
        write_records(self.tmp / "in", divs, decisions)
        return rank.rank(self.tmp / "in")

    def test_order_weights_and_tags(self):
        divs = [rec(1, 40, HP), rec(2, 5, SAME_SIZE), rec(3, 5, SAME_SIZE, span=9), rec(5, 3, [[".combat.powers", 1, 2]])]
        g = self.doc(divs)["groups"]
        by = {x["key"]["kinds"][0]: x for x in g}
        pile = by[".combat.hand"]
        self.assertEqual(1, pile["rank"])
        self.assertEqual(["card:DEFEND_IRONCLAD", "card:STRIKE_IRONCLAD"], pile["key"]["entities"])
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

    def test_run_layer_state_is_single_site(self):
        g = self.doc([rec(1, 5, [[".relics", ["A"], ["A", "B"]]], type="treasure")])["groups"][0]
        self.assertEqual(("single_site", "single_site"), (g["tag"], g["class"]))

    def test_outcome_changing_proxy_and_confirmed(self):
        d = self.doc([rec(1, 50, RUN_HP, type="map"), rec(2, 50, HP)])
        by = {g["key"]["type"]: g for g in d["groups"]}
        self.assertEqual(1.0, by["map"]["outcome_changing_share"])  # carried state, no re-anchor: proxy
        self.assertAlmostEqual(1.5, by["map"]["weight"])
        self.assertEqual(0.0, by["combat"]["outcome_changing_share"])
        self.assertEqual((0, 1), (d["summary"]["outcome_confirmed"], d["summary"]["outcome_proxy"]))
        # a repaired (re-anchored) divergence is not outcome-changing by proxy
        self.assertEqual(0.0, self.doc([rec(1, 50, RUN_HP, type="map", reanchored=True)])["groups"][0]["outcome_changing_share"])
        # an emulator outcome, when recorded, decides
        same = self.doc([rec(1, 50, HP, emu_outcome={"class": "death", "floor": 9})])["groups"][0]
        diff = self.doc([rec(1, 50, HP, emu_outcome={"class": "win", "floor": 51})])["groups"][0]
        self.assertEqual((0.0, 1.0), (same["outcome_changing_share"], diff["outcome_changing_share"]))
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
        self.assertEqual([], [g for g in d["groups"] if g["key"]["kinds"] == [".combat.enemy_hp"]])

    def test_gaps_are_counted_apart(self):
        d = self.doc([rec(1, 5, [], type="event", gap=True, reason="unmapped:event"), rec(2, 5, HP)])
        self.assertEqual((1, 1), (d["summary"]["mapping_gaps"], d["summary"]["groups"]))
        self.assertEqual({"unmapped:event": 1}, d["summary"]["gap_reasons"])

    def test_arch_share_and_rate(self):
        # architectural: hp group at floor 50 = 1.0; single site: relics at floor 50 = 1.0 * 1.5 (proxy outcome)
        d = self.doc([rec(1, 50, HP), rec(2, 50, [[".relics", ["A"], ["B"]]], type="treasure")], decisions=200)
        s = d["summary"]
        self.assertAlmostEqual(1.0 / 2.5, s["arch_share"], places=3)
        self.assertAlmostEqual(0.01, s["state_divergence_rate"])

    def test_cli_writes_the_files_and_a_missing_folder_is_empty(self):
        write_records(self.tmp / "in", [rec(1, 5, SAME_SIZE)])
        self.assertEqual(0, rank.main([str(self.tmp / "in"), "--out", str(self.tmp / "out")]))
        doc = json.loads((self.tmp / "out" / "ranked.json").read_text())
        self.assertEqual(1, doc["summary"]["groups"])
        self.assertIn("| 1 |", (self.tmp / "out" / "top20.md").read_text())
        self.assertEqual(0, rank.rank(self.tmp / "nothing")["summary"]["groups"])


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


if __name__ == "__main__":
    unittest.main()
