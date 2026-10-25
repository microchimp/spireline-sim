"""`envproto.py play --recycle N` and `gen0.py score --recycle N`: a game process is restarted after N runs (D8). No game, no emulator."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "rl", ROOT / "referee"):
    sys.path.insert(0, str(p))

import envproto  # noqa: E402
import gen0  # noqa: E402


class FakeLockstep:
    stats = {"decisions": 3, "ranked": 1, "diverged": 0, "reanchors": 0, "reanchor_failed": 0, "first_loss": None}
    lost = 0
    log = types.SimpleNamespace(records=[])
    agent_seed = [0, 0]
    assumed: list = []


class PlayWorkerRecycleTest(unittest.TestCase):
    def play(self, recycle: int, seeds: list[str]) -> tuple[list[str], list[dict]]:
        boots: list[str] = []
        closes: list[int] = []

        class FakeEnv:
            pid = 1

            def __init__(self, game, scratch, install=False):
                boots.append(Path(scratch).name)

            def close(self):
                closes.append(1)

        fake = types.SimpleNamespace(play_run=lambda env, seed, **kw: ({"end": "death", "floor": 3, "asked": 1}, FakeLockstep()))
        q: queue.Queue = queue.Queue()
        with tempfile.TemporaryDirectory() as d, unittest.mock.patch.object(envproto, "GameEnv", FakeEnv), \
                unittest.mock.patch.dict(sys.modules, {"lockstep": fake}), \
                unittest.mock.patch.object(envproto, "below_normal_self"), \
                unittest.mock.patch.object(envproto.rr, "connections", return_value=set()), \
                unittest.mock.patch.object(envproto, "_fields_row", return_value={}):
            envproto.play_worker(0, "g", d, seeds, 60, q, 10.0, 10, 0, None, 0.0, "IRONCLAD", recycle)
        events = []
        while not q.empty():
            events.append(q.get())
        self.assertEqual(len(boots), len(closes))  # every game process that started was closed
        return boots, events

    def test_a_game_is_restarted_after_recycle_runs(self):
        boots, events = self.play(2, list("ABCDE"))
        self.assertEqual(["p00r0", "p00r1", "p00r2"], boots)
        self.assertEqual(5, sum(e["ev"] == "end" for e in events))

    def test_no_recycle_keeps_one_game_for_the_whole_share(self):
        boots, _ = self.play(0, list("ABCDE"))
        self.assertEqual(["p00r0"], boots)


class ScoreCommandTest(unittest.TestCase):
    def test_score_passes_recycle_to_the_play_block_and_the_parser_takes_it(self):
        a = gen0.build_parser().parse_args(["score", "--game", "g", "--seeds-file", "f", "--out", "o", "--recycle", "60", "--only", "X"])
        self.assertEqual(60, a.recycle)
        seen: list[list[str]] = []
        with tempfile.TemporaryDirectory() as d:
            a.out = d
            with unittest.mock.patch("subprocess.run", side_effect=lambda cmd, **kw: seen.append(cmd) or types.SimpleNamespace(returncode=0)):
                gen0.score_real(a)
        cmd = seen[0]
        self.assertEqual("60", cmd[cmd.index("--recycle") + 1])
        self.assertIn("--engine", cmd)

    def test_the_default_is_no_recycle_flag(self):
        a = gen0.build_parser().parse_args(["score", "--game", "g", "--seeds-file", "f", "--out", "o", "--only", "X"])
        seen: list[list[str]] = []
        with tempfile.TemporaryDirectory() as d:
            a.out = d
            with unittest.mock.patch("subprocess.run", side_effect=lambda cmd, **kw: seen.append(cmd) or types.SimpleNamespace(returncode=0)):
                gen0.score_real(a)
            json.loads((Path(d) / "result.json").read_text(encoding="utf-8"))
        self.assertNotIn("--recycle", seen[0])


if __name__ == "__main__":
    unittest.main()
