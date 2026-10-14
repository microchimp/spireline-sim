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

from broken_cases import PERTURB, fight_pair, run_main, write_json  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
