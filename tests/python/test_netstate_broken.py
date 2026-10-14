"""netstate_diff FAILS on a deliberately broken case: --check without the tap or with a late step; game
against emulator with a divergence, a missing result or trace, or a replay that stops early or errs;
game against game with another checksum (fixtures: broken_cases.py, test_netstate_diff.py).
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

from broken_cases import run_main, write_json  # noqa: E402
from tools.fidelity import netstate_diff as nd  # noqa: E402


def tapped_fight(late: bool = False) -> dict:
    from test_netstate_diff import END, PHASES, PLAY, PLAYED, fight, net
    rows = [(None, [("After player turn start", net())], net(game=False)),
            (END, [(c, net()) for c in PHASES], net(game=False)),
            (PLAY, [] if late else [(PLAYED, net(ehp=21, hand=()))], net(ehp=21, hand=(), game=False))]
    return fight(rows)


class NetstateBrokenCases(unittest.TestCase):
    def test_check_passes_a_clean_tapped_fight_and_fails_the_rest(self):
        g, _ = tapped_fight()
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run_main(nd.main, ["--check", write_json(Path(d) / "a.json", {"results": [g]})])[0], 0)
            late, _ = tapped_fight(late=True)
            self.assertEqual(run_main(nd.main, ["--check", write_json(Path(d) / "b.json", {"results": [late]})])[0], 1)
            bare = copy.deepcopy(g)
            for s in bare["steps"]:
                s.pop("checksums")
            code, text = run_main(nd.main, ["--check", write_json(Path(d) / "c.json", {"results": [bare]})])
            self.assertEqual(code, 1, "a fight recorded without the tap is never OK")
            self.assertIn("UNCHECKED", text)
            self.assertEqual(run_main(nd.main, ["--check", write_json(Path(d) / "e.json", {"results": []})])[0], 1)

    def test_a_step_without_checksums_in_a_tapped_fight_is_late(self):
        g, _ = tapped_fight()
        del g["steps"][2]["checksums"]
        self.assertEqual(nd.late_steps(g), [2])

    def test_game_against_emulator_fails_on_a_divergence_or_a_missing_result(self):
        g, e = tapped_fight()
        with tempfile.TemporaryDirectory() as d:
            ref = write_json(Path(d) / "g.json", {"results": [g]})
            self.assertEqual(run_main(nd.main, [ref, write_json(Path(d) / "e.json", {"results": [e]})])[0], 0)
            bad = copy.deepcopy(e)
            bad["steps"][2]["net"][0]["state"]["creatures"][1]["hp"] = 20
            self.assertEqual(run_main(nd.main, [ref, write_json(Path(d) / "b.json", {"results": [bad]})])[0], 1)
            self.assertEqual(run_main(nd.main, [ref, write_json(Path(d) / "n.json", {"results": []})])[0], 1)
            notrace = copy.deepcopy(e)
            del notrace["steps"][1]["net"]
            self.assertEqual(run_main(nd.main, [ref, write_json(Path(d) / "t.json", {"results": [notrace]})])[0], 1)

    def test_game_against_emulator_fails_on_a_replay_that_stops_early_or_errs(self):
        # Identical as far as both go was "N steps identical on every modelled field", exit 0.
        g, e = tapped_fight()
        with tempfile.TemporaryDirectory() as d:
            ref = write_json(Path(d) / "g.json", {"results": [g]})
            for keep in (1, 2):
                short = copy.deepcopy(e)
                del short["steps"][keep:]
                code, text = run_main(nd.main, [ref, write_json(Path(d) / f"s{keep}.json", {"results": [short]})])
                self.assertEqual(code, 1, text)
                self.assertIn(f"stops after {keep} of the game's 3 steps", text)
            thrown = dict(copy.deepcopy(e), error="System.Exception: boom")
            code, text = run_main(nd.main, [ref, write_json(Path(d) / "x.json", {"results": [thrown]})])
            self.assertEqual(code, 1, text)
            self.assertIn("reports an error", text)

    def test_game_against_game_fails_on_another_checksum(self):
        g, _ = tapped_fight()
        with tempfile.TemporaryDirectory() as d:
            ref = write_json(Path(d) / "g.json", {"results": [g]})
            self.assertEqual(run_main(nd.main, [ref, write_json(Path(d) / "s.json", {"results": [copy.deepcopy(g)]})])[0], 0)
            other = copy.deepcopy(g)
            other["steps"][2]["checksums"][0][2] = 12345
            self.assertEqual(run_main(nd.main, [ref, write_json(Path(d) / "o.json", {"results": [other]})])[0], 1)
            shorter = copy.deepcopy(g)
            shorter["steps"][2]["checksums"] = []
            self.assertEqual(run_main(nd.main, [ref, write_json(Path(d) / "h.json", {"results": [shorter]})])[0], 1)


if __name__ == "__main__":
    unittest.main()
