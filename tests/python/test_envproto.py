"""referee/envproto.py: the channel protocol against a fake game (no game needed), and the bench summary."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
import envproto  # noqa: E402

FAKE = r'''
import json, sys
P = "\x1eENV "
def send(o): sys.stdout.write("log noise\n" + P + json.dumps(o) + "\n"); sys.stdout.flush()
send({"t": "ready", "pid": 1})
for line in sys.stdin:
    c = json.loads(line)
    if c.get("cmd") == "quit": break
    if c.get("cmd") == "reset":
        left = 3
        send({"t": "decide", "kind": "combat", "obs": {}, "actions": [["end"], ["play", 0, "BASH", 0]]})
        continue
    if c["a"] not in (0, 1): send({"t": "bad_action"}); continue
    left -= 1
    send({"t": "end", "end": "death"} if left == 0 else {"t": "decide", "kind": "map", "obs": {}, "actions": [1, 2, 3]})
'''


class EnvProtoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        dll = self.tmp / "game" / "data_sts2_windows_x86_64" / "sts2.dll"
        dll.parent.mkdir(parents=True)
        dll.write_bytes(b"fake")
        os.environ["SPIRELINE_REFEREE_KNOWN_SHA256"] = hashlib.sha256(b"fake").hexdigest()
        (self.tmp / "fake.py").write_text(FAKE, encoding="utf-8")

    def test_reset_step_end(self):
        env = envproto.GameEnv(self.tmp / "game", self.tmp / "scratch", install=False,
                               command=[sys.executable, str(self.tmp / "fake.py")])
        m, done = env.reset("S1")
        self.assertEqual((m["kind"], done), ("combat", False))
        m, done = env.step(1)
        self.assertEqual(m["kind"], "map")
        m, done = env.step(0)
        m, done = env.step(0)
        self.assertTrue(done)
        self.assertEqual(m["end"], "death")
        m, done = env.reset("S2")
        with self.assertRaises(ValueError):
            env.step(7)
        env.close()
        self.assertIn(b"log noise", (self.tmp / "scratch" / "game-stdout.log").read_bytes())

    def test_summary(self):
        ev = [{"ev": "ready", "w": 0, "pid": 1, "boot_s": 3, "t": 0},
              {"ev": "tick", "w": 0, "dec": 200, "t": 40},
              {"ev": "end", "w": 0, "end": "death", "floor": 3, "asked": 250, "mem_mb": 900, "dec": 50, "t": 50, "s": 9},
              {"ev": "end", "w": 0, "end": "event_no_options:X", "floor": 5, "asked": 10, "mem_mb": 950, "dec": 10, "t": 60, "s": 9}]
        s = envproto.summarize(ev, 1)
        self.assertEqual(s["decisions"], 260)
        self.assertEqual(s["ends"], {"death": 1, "event_no_options": 1})
        self.assertEqual(s["mem_growth_mb_mean"], 50)



class FullSelectSpecTest(unittest.TestCase):
    def test_reset_spec_carries_the_flag_only_when_on(self):
        self.assertNotIn("full_select", envproto.reset_spec("S1"))  # old records and specs are unchanged
        self.assertIs(envproto.reset_spec("S1", full_select=True)["full_select"], True)

    def test_record_meta_stores_the_flag(self):
        with tempfile.TemporaryDirectory() as d:
            envproto.write_record_meta(Path(d), "S1", "IRONCLAD", False, True)
            self.assertTrue(envproto.read_record_meta(Path(d), "S1")["full_select"])
            self.assertFalse(envproto.read_record_meta(Path(d), "S2")["full_select"])  # no sidecar: off


class PlayInstallsItsDriverTest(unittest.TestCase):
    def test_play_installs_this_trees_driver_before_any_game_boots(self):
        from types import SimpleNamespace
        from unittest import mock

        calls = []

        class Stop(Exception):
            pass

        def no_spawn(_kind):
            calls.append("spawn")
            raise Stop

        a = SimpleNamespace(out=tempfile.mkdtemp(), game="G", character="IRONCLAD", runs=1, seeds_file=None, agent_seed=1,
                            stop_after_h=0, procs=1, sims=1, step_timeout=1, ascension=10)
        with mock.patch.object(envproto.rr, "check_build", lambda g: calls.append("check")), \
                mock.patch.object(envproto.rr, "install", lambda g, d: calls.append(("install", Path(d).parts[-4:]))), \
                mock.patch.object(envproto.mp, "get_context", no_spawn):
            with self.assertRaises(Stop):
                envproto.play(a)
        self.assertEqual(calls, ["check", ("install", ("driver", "bin", "Release", "net9.0")), "spawn"])


if __name__ == "__main__":
    unittest.main()
