"""reanchor_compare FAILS on a deliberately broken re-anchored replay: a diverging or throwing step, an
unusable result, a game fight without a result or with another seed's, and exits 3 on a step not judged
(fixtures: broken_cases.py).
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from broken_cases import (  # noqa: E402
    NM, PERTURB, SKITTISH_BROKEN, SKITTISH_NOT_COMPARED, fight_pair, gardener_pair, run_main, write_json)
from tools.fidelity import reanchor_compare as rc  # noqa: E402



class ReanchorCompareBrokenCases(unittest.TestCase):
    def files(self, d: str, games: list, emus: list) -> list:
        return [write_json(Path(d) / "g.json", {"results": games}), write_json(Path(d) / "r.json", {"results": emus})]

    def test_all_matching_passes_and_each_broken_case_fails(self):
        with tempfile.TemporaryDirectory() as d:
            g, e = fight_pair()
            self.assertEqual(run_main(rc.main, self.files(d, [g], [e]))[0], 0)
            bad = copy.deepcopy(e)
            PERTURB["player.block"](bad["steps"][1]["state"])
            self.assertEqual(run_main(rc.main, self.files(d, [g], [bad]))[0], 1, "a diverging step")
            thrown = copy.deepcopy(e)
            thrown["steps"][2]["error"] = "NullReferenceException: x"
            self.assertEqual(run_main(rc.main, self.files(d, [g], [thrown]))[0], 1, "a step that threw")
            self.assertEqual(run_main(rc.main, self.files(d, [g], [{"seed": g["seed"], "error": "no game result", "steps": []}]))[0], 1,
                             "an unusable result")
            g2, e2 = fight_pair(seed="SECONDSEED")
            self.assertEqual(run_main(rc.main, self.files(d, [g, g2], [e]))[0], 1, "a game fight without a result")
            self.assertEqual(run_main(rc.main, self.files(d, [g, g2], [e2, e]))[0], 0, "paired by seed when out of order")
            self.assertEqual(run_main(rc.main, self.files(d, [g], [dict(e, seed="SECONDSEED")]))[0], 1,
                             "a result of another seed")
            short = copy.deepcopy(e)
            short["steps"][2]["anchor_error"] = "no emulator id for X"
            self.assertEqual(run_main(rc.main, self.files(d, [g], [short]))[0], 3, "a step not re-anchored is no pass")

    def test_json_summary_is_written(self):
        with tempfile.TemporaryDirectory() as d:
            g, e = fight_pair()
            out = Path(d) / "s.json"
            self.assertEqual(run_main(rc.main, [*self.files(d, [g], [e]), "--json", str(out)])[0], 0)
            self.assertIn("summary", json.loads(out.read_text(encoding="utf-8")))


class SkittishReanchorCases(unittest.TestCase):
    """H01: a gardener's SkittishSpent latch is no power of the game's, so the re-anchored verdicts judge the
    Skittish block, not the latch (the fixtures of broken_cases.py)."""

    def verdicts(self, g: dict, e: dict) -> list:
        return [s["verdict"] for s in rc.analyse(g, e, NM)["steps"]]

    def test_the_faithful_fight_matches_at_every_step(self):
        g, e = gardener_pair()
        a = rc.analyse(g, e, NM)
        self.assertIsNone(a["first_divergence"])
        self.assertEqual(self.verdicts(g, e), ["match"] * len(g["steps"]))

    def test_each_wrong_skittish_diverges_at_its_step_and_only_there(self):
        for name, (step, wreck) in SKITTISH_BROKEN.items():
            g, e = gardener_pair()
            wreck(e["steps"][step]["state"])
            a = rc.analyse(g, e, NM)
            self.assertEqual(a["first_divergence"], step, name)
            want = ["match"] * len(g["steps"])
            want[step] = "diverge"
            self.assertEqual(self.verdicts(g, e), want, name)

    def test_the_latch_alone_is_not_judged(self):
        for name, (step, change) in SKITTISH_NOT_COMPARED.items():
            g, e = gardener_pair()
            change(e["steps"][step]["state"])
            self.assertIsNone(rc.analyse(g, e, NM)["first_divergence"], name)

    def test_the_command_line_fails_on_a_wrong_skittish_only(self):
        with tempfile.TemporaryDirectory() as d:
            g, e = gardener_pair()
            ref = write_json(Path(d) / "g.json", {"results": [g]})
            self.assertEqual(run_main(rc.main, [ref, write_json(Path(d) / "r.json", {"results": [e]})])[0], 0)
            name, (step, wreck) = next(iter(SKITTISH_BROKEN.items()))
            wreck(e["steps"][step]["state"])
            self.assertEqual(run_main(rc.main, [ref, write_json(Path(d) / "b.json", {"results": [e]})])[0], 1, name)


if __name__ == "__main__":
    unittest.main()
