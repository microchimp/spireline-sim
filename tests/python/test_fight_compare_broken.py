"""fight_compare FAILS on a deliberately broken replay: one perturbation per compared field, a replay that
stops early, an emulator error, a missing result or one of another seed (fixtures: broken_cases.py).
"""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from broken_cases import (  # noqa: E402
    NM, NOT_COMPARED, PERTURB, SKITTISH, SKITTISH_BROKEN, SKITTISH_NOT_COMPARED, fight_pair, gardener_pair, run_main, write_json)
from tools.fidelity import corpus as cp  # noqa: E402
from tools.fidelity import fight_compare as fc  # noqa: E402


class FightCompareBrokenCases(unittest.TestCase):
    def test_the_faithful_replay_is_identical(self):
        g, e = fight_pair()
        self.assertIsNone(fc.first_difference(g, e, NM))
        self.assertIsNone(fc.report(g, e, NM)[1])

    def test_every_compared_field_when_wrong_is_a_difference(self):
        missed = []
        for name, wreck in PERTURB.items():
            g, e = fight_pair()
            wreck(e["steps"][1]["state"])
            fd = fc.first_difference(g, e, NM)
            if fd is None or fd[0] != 1:
                missed.append(name)
        self.assertEqual(missed, [], "fields the fight comparison does not see")

    def test_what_it_skips_on_purpose_is_skipped(self):
        for name, change in NOT_COMPARED.items():
            g, e = fight_pair()
            change(e["steps"][1]["state"])
            self.assertIsNone(fc.first_difference(g, e, NM), name)

    def test_a_replay_that_stops_early_differs_where_it_stopped(self):
        g, e = fight_pair(4)
        del e["steps"][2:]
        self.assertEqual(fc.first_difference(g, e, NM)[0], 2)
        e["error"] = "System.Exception: boom"
        self.assertEqual(fc.first_difference(g, e, NM)[1][0][0], "(error)")
        self.assertEqual(fc.report(g, e, NM)[1], 2)

    def test_every_shared_judge_sees_the_short_replay(self):
        from tools.fidelity import coverage
        g, e = fight_pair(4)
        del e["steps"][3:]
        self.assertEqual(coverage.first_divergence(g, e, NM), 3)
        self.assertEqual(cp.judge_fight(g, e, NM)["fd"], 3)

    def test_the_command_line_fails_on_a_difference_a_missing_result_or_another_seed(self):
        with tempfile.TemporaryDirectory() as d:
            g, e = fight_pair()
            ref = write_json(Path(d) / "g.json", {"results": [g]})
            good = write_json(Path(d) / "e.json", {"results": [e]})
            self.assertEqual(run_main(fc.main, [ref, good])[0], 0)
            bad = copy.deepcopy(e)
            PERTURB["enemy hp"](bad["steps"][2]["state"])
            self.assertEqual(run_main(fc.main, [ref, write_json(Path(d) / "b.json", {"results": [bad]})])[0], 1)
            self.assertEqual(run_main(fc.main, [ref, write_json(Path(d) / "n.json", {"results": []})])[0], 1)
            other = dict(e, seed="OTHERSEED0")
            self.assertEqual(run_main(fc.main, [ref, write_json(Path(d) / "o.json", {"results": [other]})])[0], 1)
            self.assertEqual(run_main(fc.main, [write_json(Path(d) / "z.json", {"results": []}), good])[0], 1)

    def test_the_command_line_fails_on_an_emulator_error_after_the_last_step(self):
        # Every step matched, then the replay threw (or the game fight has no steps at all): not reproduced.
        with tempfile.TemporaryDirectory() as d:
            g, e = fight_pair()
            ref = write_json(Path(d) / "g.json", {"results": [g]})
            thrown = dict(copy.deepcopy(e), error="System.NullReferenceException: x")
            code, text = run_main(fc.main, [ref, write_json(Path(d) / "e.json", {"results": [thrown]})])
            self.assertEqual(code, 1, text)
            empty_g, empty_e = dict(g, steps=[]), {"seed": g["seed"], "steps": [], "error": "System.NullReferenceException: y"}
            self.assertEqual(run_main(fc.main, [write_json(Path(d) / "g0.json", {"results": [empty_g]}),
                                                write_json(Path(d) / "e0.json", {"results": [empty_e]})])[0], 1)
            self.assertIsNone(fc.first_difference(g, thrown, NM), "the shared judges keep the error apart (corpus ferr)")


class SkittishBrokenCases(unittest.TestCase):
    """H01: the emulator's SkittishSpent latch is no game power; Skittish's amount and its block are compared."""

    def test_the_faithful_gardener_fight_is_identical_with_the_latch_up(self):
        g, e = gardener_pair()
        up = [s["state"]["snapshot"]["enemies"][0]["buffs"] for s in e["steps"]]
        self.assertTrue(any(b["name"] == "SkittishSpent" for buffs in up for b in buffs), "the fixture holds the latch")
        self.assertIsNone(fc.first_difference(g, e, NM))
        self.assertEqual(fc.emu_view(e["steps"][1]["state"], NM)["enemies"][0]["powers"], [["SKITTISH", SKITTISH]])

    def test_every_wrong_skittish_is_a_difference_at_its_step(self):
        for name, (step, wreck) in SKITTISH_BROKEN.items():
            g, e = gardener_pair()
            wreck(e["steps"][step]["state"])
            fd = fc.first_difference(g, e, NM)
            self.assertIsNotNone(fd, name)
            self.assertEqual(fd[0], step, name)
            self.assertTrue(all(p == ".enemies" for p, _, _ in fd[1]), name)

    def test_the_latch_alone_is_not_compared(self):
        for name, (step, change) in SKITTISH_NOT_COMPARED.items():
            g, e = gardener_pair()
            change(e["steps"][step]["state"])
            self.assertIsNone(fc.first_difference(g, e, NM), name)

    def test_the_shared_judges_agree(self):
        from tools.fidelity import coverage
        for name, (step, wreck) in {None: (None, None), **SKITTISH_BROKEN}.items():
            g, e = gardener_pair()
            if wreck:
                wreck(e["steps"][step]["state"])
            self.assertEqual(coverage.first_divergence(g, e, NM), step, name)
            self.assertEqual(cp.judge_fight(g, e, NM)["fd"], step, name)


if __name__ == "__main__":
    unittest.main()
