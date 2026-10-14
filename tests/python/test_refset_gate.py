"""refset_score: its judge FAILS on a deliberately broken replay, and its gate (--against, compare) fails on
anything worse per cell, fight, class, coverage kind or codex run, even under equal counts, and on a
comparison that cannot be made (fixtures: broken_cases.py).
"""

from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from broken_cases import NM, PERTURB, fight_pair  # noqa: E402



class RefsetScoreBrokenCases(unittest.TestCase):
    def judge(self, g, e):
        from tools.fidelity import refset_score as rss
        r = {"seed": e.get("seed"), "steps": e.get("steps", [])}
        return rss.judge("realistic", g, e, r, NM, None)

    def test_a_faithful_replay_is_identical_and_each_broken_one_is_not(self):
        g, e = fight_pair(4)
        self.assertTrue(self.judge(g, e)["identical"])
        bad = copy.deepcopy(e)
        PERTURB["player.energy"](bad["steps"][2]["state"])
        v = self.judge(g, bad)
        self.assertEqual((v["identical"], v["step"]), (False, 2))
        short = copy.deepcopy(e)
        del short["steps"][3:]
        v = self.judge(g, short)
        self.assertEqual((v["identical"], v.get("step")), (False, 3))
        self.assertFalse(self.judge(g, dict(e, seed="OTHERSEED0"))["identical"])
        self.assertFalse(self.judge(g, {"seed": g["seed"], "error": "boom", "steps": []})["identical"])


def score(**over) -> dict:
    """A stored refset score, cut down to what the gate reads."""
    cell = {"specs": 3, "kept": 3, "identical": 2, "verified": 10}
    s = {"subset": "all", "record": {"stamp": "r1"}, "not_quotable": [],
         "headline": {"realistic": dict(cell)}, "realistic_table": {"act1 monster": dict(cell)},
         "verdicts": [{"id": "A", "identical": True, "verified": 5}, {"id": "B", "identical": True, "verified": 5},
                      {"id": "C", "identical": False, "verified": 0, "step": 1, "field": "player.hp", "card": "BASH"}],
         "regression": {"emulator": {"failing": ["thorns"]}},
         "coverage": {"counts": {"card": {"clean": 4}}},
         "codex": {"runs": 3, "complete": 2, "per_run": {"0": ["complete", 50, ""], "1": ["complete", 50, ""],
                                                         "2": ["mismatch", 10, "EMULATOR: card_reward"]}},
         "runs": {"runs": 2, "runs_matching": 1}}
    s.update(over)
    return copy.deepcopy(s)


class RefsetGateBrokenCases(unittest.TestCase):
    """refset_score's gate (--against / compare): the same score passes; each thing worse fails it, per fight and
    per run even when every count is unchanged."""

    def gate(self, old, new, explained=None):
        from tools.fidelity import refset_score as rss
        return rss.gate(old, new, explained)

    def assertWorse(self, new, needle, old=None):
        res = self.gate(old or score(), new)
        self.assertFalse(res["ok"], needle)
        self.assertTrue(any(needle in w for w in res["worse"] + res["unsound"]), (needle, res))

    def test_the_same_score_passes_and_a_better_one_too(self):
        self.assertTrue(self.gate(score(), score())["ok"])
        better = score()
        better["verdicts"][2].update(identical=True, verified=4)
        better["headline"]["realistic"].update(identical=3, verified=14)
        better["realistic_table"]["act1 monster"].update(identical=3, verified=14)
        res = self.gate(score(), better)
        self.assertTrue(res["ok"])
        self.assertTrue(res["better"])

    def test_two_fights_swapping_with_every_count_equal_fail(self):
        new = score()
        new["verdicts"][0].update(identical=False, verified=0, step=2)
        new["verdicts"][2].update(identical=True, verified=5)
        self.assertWorse(new, "fight A: identical before")

    def test_a_cell_with_fewer_identical_or_verified_or_another_set_fails(self):
        for k, v, needle in (("identical", 1, "identical 2 -> 1"), ("verified", 9, "verified 10 -> 9"), ("kept", 2, "specs/kept")):
            new = score()
            new["realistic_table"]["act1 monster"][k] = v
            self.assertWorse(new, needle)
        new = score(realistic_table={})
        self.assertWorse(new, "cell act1 monster: missing")

    def test_a_fight_missing_or_less_verified_fails(self):
        new = score()
        del new["verdicts"][1]
        self.assertWorse(new, "fight B: missing")
        new = score()
        new["verdicts"][2]["verified"] = 0
        old = score()
        old["verdicts"][2]["verified"] = 3
        self.assertWorse(new, "fight C: steps verified 3 -> 0", old)

    def test_a_class_coverage_or_whole_runs_getting_worse_fails(self):
        self.assertWorse(score(regression={"emulator": {"failing": ["thorns", "rupture"]}}), "class emulator rupture")
        self.assertWorse(score(coverage={"counts": {"card": {"clean": 3}}}), "coverage card")
        self.assertWorse(score(runs={"runs": 2, "runs_matching": 0}), "whole runs")

    def test_codex_runs_swapping_with_equal_totals_fail(self):
        new = score()
        new["codex"]["per_run"].update({"0": ["mismatch", 30, "EMULATOR: shop"], "2": ["complete", 50, ""]})
        self.assertWorse(new, "codex run 0: complete before")
        new = score()
        new["codex"]["per_run"]["2"] = ["mismatch", 4, "EMULATOR: map"]
        self.assertWorse(new, "codex run 2: stops at floor 4")
        new = score()
        del new["codex"]["per_run"]["1"]
        self.assertWorse(new, "codex run 1: not walked")
        self.assertWorse(score(codex={"skipped": "not run (--codex)"}), "codex: walked before")

    def test_a_comparison_that_cannot_be_made_fails(self):
        self.assertWorse(score(record={"stamp": "r2"}), "different records")
        self.assertWorse(score(subset="pilot"), "subsets differ")
        self.assertWorse(score(not_quotable=["dirty emulator checkout"]), "NOT QUOTABLE")
        old_walk = score()
        old_walk["codex"]["per_run"] = None
        self.assertWorse(score(), "without per-run results", old_walk)

    def test_a_comparer_change_is_named_by_file_and_another_commit_alone_is_not_one(self):
        from tools.fidelity import refset_score as rss
        files = {"tools/fidelity/fight_compare.py": "1" * 40, "tools/fidelity/refset_score.py": "2" * 40}
        same = self.gate(score(comparer={"commit": "a" * 40, "files": files}),
                         score(comparer={"commit": "b" * 40, "files": dict(files)}))
        self.assertTrue(same["ok"])
        self.assertEqual(same["notes"], [], "two checkouts with the same comparer files")
        changed = dict(files, **{"tools/fidelity/fight_compare.py": "3" * 40})
        res = self.gate(score(comparer={"commit": "a" * 40, "files": files}),
                        score(comparer={"commit": "b" * 40, "files": changed}))
        self.assertTrue(res["ok"])
        self.assertIn("the comparer changed (tools/fidelity/fight_compare.py)", res["notes"][0])
        # A score from before scores kept their files: its commit's files, from git.
        def blobs(c, paths, root=None):
            return dict(files) if c == "a" * 40 else None
        with mock.patch.object(rss, "blobs_at", side_effect=blobs):
            self.assertEqual(self.gate(score(comparer={"commit": "a" * 40}),
                                       score(comparer={"commit": "b" * 40, "files": files}))["notes"], [])
            self.assertIn("cannot tell", self.gate(score(comparer={"commit": "c" * 40}),
                                                   score(comparer={"commit": "b" * 40, "files": files}))["notes"][0])
        self.assertIn("modified checkout", self.gate(score(comparer={"commit": "a" * 40, "dirty": True}),
                                                     score(comparer={"commit": "b" * 40, "files": files}))["notes"][0])

    def test_the_comparer_files_are_this_checkouts_judge(self):
        from tools.fidelity import refset_score as rss
        got = rss.comparer_files()
        self.assertIn("tools/fidelity/refset_score.py", got)
        self.assertIn("tools/fidelity/fight_compare.py", got)
        self.assertTrue(all(len(v) == 40 for v in got.values()), got)
        head = rss.blobs_at("HEAD", ["tools/fidelity/fight_compare.py", "no/such/file.py"])
        self.assertEqual(len(head["tools/fidelity/fight_compare.py"]), 40)
        self.assertIsNone(head["no/such/file.py"])
        self.assertIsNone(rss.blobs_at("0" * 40, ["tools/fidelity/fight_compare.py"]))

    def test_an_explained_change_passes_and_an_unused_explanation_is_named(self):
        new = score()
        new["verdicts"][0].update(identical=False, verified=0)
        res = self.gate(score(), new, {"fight A": "two wrongs cancelled before (ledger L12)", "fight Z": "stale"})
        self.assertTrue(res["ok"], res)
        self.assertEqual(len(res["explained"]), 1)
        self.assertEqual(res["unused_explanations"], ["fight Z"])

    def test_the_command_line_exits_1_on_worse_and_0_on_the_same(self):
        from tools.fidelity import refset_score as rss
        with tempfile.TemporaryDirectory() as d:
            a, b, ex = Path(d) / "a.json", Path(d) / "b.json", Path(d) / "explained.txt"
            a.write_text(json.dumps(score()), encoding="utf-8")
            worse = score()
            worse["verdicts"][1]["identical"] = False
            b.write_text(json.dumps(worse), encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(rss.compare_main([str(a), str(a)]), 0)
                self.assertEqual(rss.compare_main([str(a), str(b)]), 1)
                ex.write_text("# accepted\nfight B  the old match was luck\n", encoding="utf-8")
                self.assertEqual(rss.compare_main([str(a), str(b), "--explained", str(ex)]), 0)
            self.assertIn("FAIL", out.getvalue())


if __name__ == "__main__":
    unittest.main()
