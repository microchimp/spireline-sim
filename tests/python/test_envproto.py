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



class ProvenanceSummaryTest(unittest.TestCase):
    """Review 1.6: the play summary says what the block's re-anchors were seeded from, so a block that ran on observation
    anchoring does not read like one that ran on the game's hidden state."""

    def runs(self):
        a = {"stats": {"anchored_with": {"hidden": 3, "streams": 1}, "streams_named": 34, "streams_moved": 30,
                       "streams_unmoved_names": {"Nope": 2}}}
        b = {"stats": {"anchored_with": {"observation": 4, "none": 1}, "streams_named": 17, "streams_moved": 0,
                       "streams_unmoved_names": {"Nope": 1, "Odd": 1}}}
        return [a, b, {"stats": {}}]  # a run that never re-anchored carries none of the keys

    def test_the_counts_are_summed_over_the_runs(self):
        p = envproto._provenance(self.runs())
        self.assertEqual(p["anchored_with"], {"hidden": 3, "observation": 4, "streams": 1, "none": 1})
        self.assertEqual((p["streams_named"], p["streams_moved"]), (51, 30))
        self.assertEqual(p["streams_unmoved_names"], {"Nope": 3, "Odd": 1})

    def test_the_walks_are_summed_over_the_runs(self):
        w = {"walks": 2, "floors": 5, "picks": 7, "fights": 3}
        p = envproto._provenance([{"stats": {"walk": w}}, {"stats": {"walk": dict(w, walks=1)}}, {"stats": {}}])
        self.assertEqual(p["walk"], {"walks": 3, "floors": 10, "picks": 14, "fights": 6})
        self.assertEqual(envproto._provenance([])["walk"], {"walks": 0, "floors": 0, "picks": 0, "fights": 0})

    def test_a_block_on_observation_anchoring_shows_zero_hidden(self):
        p = envproto._provenance([{"stats": {"anchored_with": {"observation": 5}}}])
        self.assertEqual(p["anchored_with"], {"hidden": 0, "observation": 5, "streams": 0, "none": 0})
        self.assertEqual(envproto._provenance([])["anchored_with"], {"hidden": 0, "observation": 0, "streams": 0, "none": 0})


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


class PlayOnlyTest(unittest.TestCase):
    """B7: `--only` plays chosen seeds of the block with the run index each has in the FULL file (the agent's RNG is [agent seed, index])."""

    SEEDS = ["S0", "S1", "S2", "S3", "S4", "S5"]

    def test_select_runs_keeps_the_place_in_the_full_file(self):
        self.assertEqual(envproto.select_runs(self.SEEDS, 4, None), (["S0", "S1", "S2", "S3"], [0, 1, 2, 3]))
        self.assertEqual(envproto.select_runs(self.SEEDS, 2, "S5,S3"), (["S3", "S5"], [3, 5]))  # --runs does not cut a chosen seed
        with self.assertRaises(ValueError):
            envproto.select_runs(self.SEEDS, 6, "S3,NOPE")

    def test_the_worker_is_handed_the_full_file_index(self):
        from types import SimpleNamespace
        from unittest import mock

        import json

        seen = []

        class Stop(Exception):
            pass

        class FakeProc:
            def __init__(self, target=None, args=()):
                seen.append((target, args))

            def start(self):
                raise Stop

        fake_ctx = SimpleNamespace(Queue=lambda: None, Process=FakeProc)
        f = Path(tempfile.mkdtemp()) / "block.json"
        f.write_text(json.dumps({"name": "x", "seeds": self.SEEDS}), encoding="utf-8")
        a = SimpleNamespace(out=tempfile.mkdtemp(), game="G", character="IRONCLAD", runs=3, seeds_file=str(f), agent_seed=1,
                            stop_after_h=0, procs=2, sims=1, step_timeout=1, ascension=10, only="S5,S2,S3")
        with mock.patch.object(envproto.rr, "check_build", lambda g: None), mock.patch.object(envproto.rr, "install", lambda g, d: None),                 mock.patch.object(envproto.mp, "get_context", lambda _k: fake_ctx):
            with self.assertRaises(Stop):
                envproto.play(a)
        # chosen seeds S2, S3, S5 sit at 2, 3, 5 of the file; two workers take every other one
        shares = [(args[3], args[9]) for _t, args in seen]
        self.assertEqual(shares, [(["S2", "S5"], [2, 5]), (["S3"], [3])])


if __name__ == "__main__":
    unittest.main()
