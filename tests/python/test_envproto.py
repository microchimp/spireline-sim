"""referee/envproto.py: the channel protocol against a fake game (no game needed), and the bench summary."""

from __future__ import annotations

import hashlib
import json
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


class PlayResumeTest(unittest.TestCase):
    """N3: `play --resume` keeps the finished rows of an earlier session and plays only the missing seeds, each with the run id and
    agent seed it has in a fresh block. The game workers are replaced by a function that puts the events a real worker puts."""

    SEEDS = ["S0", "S1", "S2", "S3", "S4", "S5"]
    AS = 77
    AGENT = {"generation": 3, "dir": "gen3/c", "prior": True, "prior_sha256": "ab", "macro": "rules"}

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        self.out = self.tmp / "block"
        self.out.mkdir()
        self.seeds_file = self.tmp / "block.json"
        self.seeds_file.write_text(json.dumps({"name": "x", "seeds": self.SEEDS}), encoding="utf-8")
        self.calls: list[tuple] = []
        self.during: dict = {}

    def stamp(self, **over):
        s = {"agent": self.AGENT, "sims": "a3-v1", "ascension": 10, "character": "IRONCLAD", "engine": "core2", "agent_seed": self.AS}
        s.update(over)
        return s

    def row(self, seed, ix=None, agent_seed=None, **over):
        ix = self.SEEDS.index(seed) if ix is None else ix
        e = {"ev": "end", "w": 0, "seed": seed, "character": "IRONCLAD", "end": "death", "floor": 10 + ix, "asked": 5, "s": 2.0,
             "stats": {"decisions": 100, "ranked": 40, "diverged": 1, "reanchors": 0, "reanchor_failed": 0, "first_loss": None},
             "lost": None, "divergences": [{"type": "state", "decision": 3, "floor": 1, "diff": [["hp", 1, 2]], "reason": f"old-{seed}"}],
             "agent_seed": [self.AS if agent_seed is None else agent_seed, ix], "assumed": 0}
        e.update(over)
        return e

    def write(self, rows, identity=True, tail="", wall=None, divergences=None):
        (self.out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + tail, encoding="utf-8")
        if identity:
            (self.out / "resume.json").write_text(json.dumps(self.stamp()), encoding="utf-8")
        if wall is not None:
            (self.out / "wall.json").write_text(json.dumps({"wall_s": wall}), encoding="utf-8")
        if divergences is not None:
            (self.out / "divergences.jsonl").write_text("\n".join(json.dumps(d) for d in divergences), encoding="utf-8")

    def play(self, resume=True, procs=2, **over):
        import contextlib
        import io
        import queue as queue_mod
        from types import SimpleNamespace
        from unittest import mock

        test = self

        def fake_worker(wid, game, out, seeds, sims, q, step_timeout, ascension, agent_seed, run_ids, stop_at, character, recycle):
            if not test.calls and (Path(out) / "divergences.jsonl").is_file():  # what the files hold while the first worker plays
                runs = [json.loads(x) for x in (Path(out) / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
                divs = (Path(out) / "divergences.jsonl").read_text(encoding="utf-8").splitlines()
                test.during = {"runs": [r["seed"] for r in runs], "divs": [json.loads(x)["seed"] for x in divs if x.strip()]}
            __import__("time").sleep(0.03)  # the clock of Windows ticks every 15 ms: a block has a wall
            test.calls.append((wid, list(seeds), list(run_ids), agent_seed, stop_at))
            for s, ix in zip(seeds, run_ids):
                q.put(test.row(s, ix, agent_seed=agent_seed, w=wid, divergences=[{"type": "state", "decision": 9, "floor": 2,
                                                                                    "diff": [["hp", 3, 4]], "reason": f"new-{s}"}]))
            q.put({"ev": "done", "w": wid})

        class FakeProc:
            def __init__(self, target=None, args=()):
                self.target, self.args = target, args

            def start(self):
                self.target(*self.args)

            def is_alive(self):
                return False

            def join(self, timeout=None):
                pass

        ctx = SimpleNamespace(Queue=queue_mod.Queue, Process=FakeProc)
        a = SimpleNamespace(out=str(self.out), game="G", character="IRONCLAD", runs=6, seeds_file=str(self.seeds_file), agent_seed=self.AS,
                            stop_after_h=0, procs=procs, sims="a3-v1", step_timeout=1, ascension=10, engine="core2", only=None, recycle=0)
        if resume:
            a.resume, a.agent_identity = True, json.dumps(self.AGENT)
        for k, v in over.items():
            setattr(a, k, v)
        err = io.StringIO()
        with mock.patch.object(envproto.rr, "check_build", lambda g: None), mock.patch.object(envproto.rr, "install", lambda g, d: None),                 mock.patch.object(envproto.mp, "get_context", lambda _k: ctx), mock.patch.object(envproto, "play_worker", fake_worker),                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            envproto.play(a)
        self.err = err.getvalue()
        return json.loads((self.out / "summary.json").read_text(encoding="utf-8"))

    def final_rows(self):
        return [json.loads(x) for x in (self.out / "runs.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]

    def test_a_fresh_block_without_the_flag_writes_the_files_it_always_wrote(self):
        summary = self.play(resume=False)
        self.assertEqual(["divergences.jsonl", "runs.jsonl", "summary.json"], sorted(p.name for p in self.out.iterdir()))
        events = [self.row(s, i, w=w, divergences=[{"type": "state", "decision": 9, "floor": 2, "diff": [["hp", 3, 4]], "reason": f"new-{s}"}])
                  for w in (0, 1) for i, s in enumerate(self.SEEDS) if i % 2 == w]
        self.assertEqual("\n".join(json.dumps(e) for e in events), (self.out / "runs.jsonl").read_text(encoding="utf-8"))  # no trailing newline
        self.assertEqual("\n".join(json.dumps(dict(e["divergences"][0], seed=e["seed"])) for e in events),
                         (self.out / "divergences.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(["agent_seed", "character", "engine", "sims", "runs", "crashes", "wins", "ends", "mean_floor", "decisions",
                          "decisions_per_s", "wall_s", "ranked_share", "diverged_decisions", "divergence_rate", "reanchors", "reanchor_failed",
                          "first_loss_by_reason", "first_divergences_by_type", "gaps", "resumed", "losses_by_reason", "reanchor_fail_reasons",
                          "reanchor_by_plan", "anchored_with", "streams_named", "streams_moved", "streams_unmoved_names", "walk",
                          "divergences_by_type", "run_s_total", "decisions_per_s_per_proc", "ranked_decisions_per_s_per_proc",
                          "network_connections", "errors"], list(summary))
        self.assertEqual([(0, ["S0", "S2", "S4"], [0, 2, 4]), (1, ["S1", "S3", "S5"], [1, 3, 5])], [c[:3] for c in self.calls])

    def test_the_finished_rows_are_kept_and_only_the_missing_seeds_play_with_their_fresh_block_ids(self):
        kept = [self.row("S1"), self.row("S4")]
        self.write(kept, tail='{"ev": "end", "seed": "S5", "wo', divergences=[
            {"seed": "S0", "reason": "stale"}, {"seed": "S1", "reason": "old-S1"}, {"seed": "OTHER", "reason": "stale"}])
        summary = self.play()
        # S0 S2 S3 S5 are missing; two workers take every other one, each seed with its place in the full list and the same agent seed
        self.assertEqual([(0, ["S0", "S3"], [0, 3]), (1, ["S2", "S5"], [2, 5])], [c[:3] for c in self.calls])
        self.assertEqual({self.AS}, {c[3] for c in self.calls})
        self.assertEqual({"runs": ["S1", "S4"], "divs": ["S1", "S4"]}, self.during)  # the half line and the stale rows are gone at once
        rows = self.final_rows()
        self.assertEqual(["S1", "S4", "S0", "S3", "S2", "S5"], [r["seed"] for r in rows])
        self.assertEqual(kept, rows[:2])
        self.assertEqual([[self.AS, i] for i in (0, 3, 2, 5)], [r["agent_seed"] for r in rows[2:]])
        divs = [json.loads(x) for x in (self.out / "divergences.jsonl").read_text(encoding="utf-8").split("\n")]
        self.assertEqual(["old-S1", "old-S4", "new-S0", "new-S3", "new-S2", "new-S5"], [d["reason"] for d in divs])
        self.assertEqual((6, 2), (summary["runs"], summary["resumed_rows"]))
        self.assertEqual(600, summary["decisions"])  # all six rows count
        self.assertEqual(json.dumps(self.stamp()), (self.out / "resume.json").read_text(encoding="utf-8"))

    def test_another_identity_starts_empty_with_a_stderr_line(self):
        self.write([self.row("S1")], wall=5000.0)
        (self.out / "resume.json").write_text(json.dumps(self.stamp(sims="a2-v1")), encoding="utf-8")
        summary = self.play()
        self.assertIn("another agent or setting", self.err)
        self.assertEqual([c[1] for c in self.calls], [["S0", "S2", "S4"], ["S1", "S3", "S5"]])
        self.assertEqual((0, 0.0), (summary["resumed_rows"], summary["wall_s_before"]))  # the other block's seconds are not this one's
        self.assertEqual(self.stamp(), json.loads((self.out / "resume.json").read_text(encoding="utf-8")))

    def test_another_agent_seed_or_another_agent_is_another_identity(self):
        self.write([self.row("S1")])
        self.assertEqual(0, self.play(agent_seed=self.AS + 1)["resumed_rows"])
        self.write([self.row("S1")])
        self.AGENT = dict(self.AGENT, prior_sha256="cd")
        self.assertEqual(0, self.play()["resumed_rows"])

    def test_a_file_with_rows_and_no_identity_is_kept_only_when_every_seed_is_in_the_list(self):
        self.write([self.row("S1"), self.row("S2")], identity=False)
        self.assertEqual(2, self.play()["resumed_rows"])
        (self.out / "resume.json").unlink()
        self.write([self.row("S1"), self.row("ELSEWHERE", 1)], identity=False)
        self.calls.clear()
        self.assertEqual(0, self.play()["resumed_rows"])
        self.assertIn("another agent or setting", self.err)

    def test_a_boot_failure_or_a_row_of_another_run_id_is_played_again_and_the_first_row_of_a_seed_wins(self):
        rows = [self.row("S0", ev="crash", where="boot", agent_seed=None), self.row("S1", agent_seed=1), self.row("S2", ev="crash", where="agent"),
                self.row("S3"), self.row("S3", floor=1), self.row("S4", ix=2)]
        rows[0].pop("agent_seed")
        self.write(rows)
        summary = self.play()
        self.assertEqual(2, summary["resumed_rows"])  # S2's agent crash and S3 (the first of its two rows); S1 and S4 carry the wrong id
        self.assertEqual(["S2", "S3"], [r["seed"] for r in self.final_rows()[:2]])
        self.assertEqual(13, self.final_rows()[1]["floor"])  # the first of S3's two rows
        self.assertEqual(sorted(self.SEEDS), sorted(r["seed"] for r in self.final_rows()))
        self.assertEqual(4, sum(len(c[1]) for c in self.calls))
        self.assertEqual(1, summary["crashes"])

    def test_the_wall_is_the_sum_of_the_sessions_and_the_time_limit_counts_the_earlier_ones(self):
        self.write([self.row("S1")], wall=1000.0)
        t = __import__("time").time()
        summary = self.play(stop_after_h=1.0)
        self.assertEqual(1000.0, summary["wall_s_before"])
        self.assertAlmostEqual(summary["wall_s"], summary["wall_s_before"] + summary["wall_s_session"], delta=0.11)
        self.assertGreaterEqual(summary["wall_s"], 1000.0)
        self.assertEqual(summary["wall_s"], json.loads((self.out / "wall.json").read_text(encoding="utf-8"))["wall_s"])
        left = self.calls[0][4] - t  # an hour minus the 1000 s already run
        self.assertTrue(2590 < left < 2610, left)

    def test_a_block_with_every_seed_done_starts_no_worker(self):
        self.write([self.row(s) for s in self.SEEDS])
        summary = self.play()
        self.assertEqual(([], 6, 6), (self.calls, summary["runs"], summary["resumed_rows"]))
        self.assertEqual(6, len(self.final_rows()))

    def test_the_stopped_sessions_game_copies_are_cleared_only_with_the_flag(self):
        stale = self.out / "scratch" / "p00r0"
        stale.mkdir(parents=True)
        (stale / "game-stdout.log").write_text("x", encoding="utf-8")
        self.play(resume=False)
        self.assertTrue(stale.exists())
        self.play()
        self.assertFalse(stale.exists())

    def test_the_parser_takes_resume_and_the_agent_identity(self):
        a = envproto.parser().parse_args(["play", "--game", "G", "--out", "O", "--resume", "--agent-identity", "{}"])
        self.assertEqual((True, "{}"), (a.resume, a.agent_identity))
        self.assertFalse(envproto.parser().parse_args(["play", "--game", "G", "--out", "O"]).resume)


class RunRowTest(unittest.TestCase):
    """L3: the end and crash rows of a play worker say when the run ended, how long its game process took to boot, how often the worker
    has restarted its game, and the longest run of identical consecutive (screen kind, action) pairs (a stalled run-layer loop)."""

    def play(self, scripts, boots=(1.5, 2.5), recycle=0):
        """Run play_worker over the seeds of `scripts` ({seed: ([(screen kind, action)...], exception or None)}) with a fake game
        process and a fake lockstep; the end and crash rows it queues."""
        import queue as queue_mod
        import time
        import types
        from unittest import mock

        booted = []

        class FakeEnv:
            def __init__(self, game, scratch, install=True):
                n = len(booted)
                booted.append(self)
                if n >= len(boots) or boots[n] is None:
                    raise envproto.GameDied("no ready")
                self.boot_s, self.pid, self.last, self.screens = boots[n], 100 + n, None, iter(())

            def reset(self, seed, timeout=0, **k):
                self.last = {"kind": next(self.screens)}
                return self.last, False

            def step(self, a, timeout=0):
                self.last = {"kind": next(self.screens)}
                return self.last, False

            def close(self):
                return 0

        def play_run(timed, seed, ascension, sims, agent_seed, character):
            pairs, boom = scripts[seed]
            env = timed.env
            env.screens = iter([k for k, _a in pairs] + ["end"])
            timed.reset(seed)
            for _k, act in pairs:
                timed.step(act)
            if boom:
                raise boom
            ls = types.SimpleNamespace(stats={"decisions": len(pairs)}, lost=None, log=types.SimpleNamespace(records=[]),
                                       agent_seed=[1, 0], assumed=[], fields=None)
            return {"end": "death", "floor": 3, "asked": len(pairs)}, ls

        q = queue_mod.Queue()
        lock = types.ModuleType("lockstep")
        lock.play_run = play_run
        self.t0 = time.time()
        with mock.patch.dict(sys.modules, {"lockstep": lock}), mock.patch.object(envproto, "GameEnv", FakeEnv), \
                mock.patch.object(envproto, "below_normal_self", lambda: None), mock.patch.object(envproto.rr, "connections", lambda pid: []):
            envproto.play_worker(0, "G", tempfile.mkdtemp(), list(scripts), 1, q, 1, 10, 7, None, 0.0, "IRONCLAD", recycle)
        self.t1 = time.time()
        rows = []
        while not q.empty():
            rows.append(q.get())
        return [r for r in rows if r["ev"] in ("end", "crash")]

    def test_an_end_row_carries_t_end_boot_s_restarts_and_the_longest_streak(self):
        pairs = [("map", 0)] + [("event", 0)] * 60 + [("event", 1), ("combat", 2), ("combat", 2)]
        (row,) = self.play({"S1": (pairs, None)})
        self.assertEqual("end", row["ev"])
        self.assertTrue(self.t0 - 1 <= row["t_end"] <= self.t1 + 1)
        self.assertEqual((1.5, 0), (row["boot_s"], row["restarts"]))
        self.assertEqual({"n": 60, "screen": "event", "action": 0}, row["streak"])

    def test_the_same_action_on_another_screen_is_not_a_streak(self):
        pairs = [("event", 0), ("map", 0), ("event", 0), ("map", 0)] * 20
        (row,) = self.play({"S1": (pairs, None)})
        self.assertEqual(1, row["streak"]["n"])

    def test_a_crash_row_carries_them_too_and_the_next_run_has_the_restart_and_its_own_boot(self):
        crash = ([("event", 3)] * 7 + [("map", 1)], TimeoutError("no message in 180 s"))
        rows = self.play({"S1": crash, "S2": ([("map", 0), ("map", 0)], None)})
        self.assertEqual(["crash", "end"], [r["ev"] for r in rows])
        c, e = rows
        self.assertEqual((1.5, 0, {"n": 7, "screen": "event", "action": 3}), (c["boot_s"], c["restarts"], c["streak"]))
        self.assertTrue(c["t_end"] > 0)
        self.assertEqual((2.5, 1, {"n": 2, "screen": "map", "action": 0}), (e["boot_s"], e["restarts"], e["streak"]))

    def test_a_boot_failure_row_has_the_time_and_the_restart_count(self):
        rows = self.play({"S1": ([], None)}, boots=(None,))
        self.assertEqual(("crash", "boot", 0), (rows[0]["ev"], rows[0]["where"], rows[0]["restarts"]))
        self.assertTrue(rows[0]["t_end"] > 0)


class SampleMachineFlagTest(unittest.TestCase):
    """L3: `play --sample-machine SECONDS` runs the machine sampler (machine.jsonl in the out folder) for the block; off by default."""

    def play(self, start_error=None, **over):
        from unittest import mock

        import machine_sampler

        made = []

        class FakeSampler:
            def __init__(self, path, interval, probes, runs_done=lambda: 0, **k):
                self.args, self.runs_done, self.started, self.stopped = (Path(path), interval), runs_done, False, False
                made.append(self)

            def start(self):
                if start_error:
                    raise start_error
                self.started = True

            def stop(self):
                self.stopped = True

        helper = PlayResumeTest("test_the_parser_takes_resume_and_the_agent_identity")
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        with mock.patch.object(machine_sampler, "MachineSampler", FakeSampler), mock.patch.object(machine_sampler, "WinProbes", lambda *a, **k: object()):
            summary = helper.play(resume=False, **over)
        return made, helper, summary

    def test_the_parser_takes_the_flag_and_it_is_off_by_default(self):
        base = ["play", "--game", "G", "--out", "O"]
        self.assertEqual(0, envproto.parser().parse_args(base).sample_machine)
        self.assertEqual(30.0, envproto.parser().parse_args([*base, "--sample-machine", "30"]).sample_machine)

    def test_without_the_flag_no_sampler_is_made(self):
        made, _h, _s = self.play()
        self.assertEqual([], made)

    def test_with_the_flag_one_sampler_writes_machine_jsonl_in_the_out_folder_and_stops_with_the_block(self):
        made, h, summary = self.play(sample_machine=30.0)
        (s,) = made
        self.assertEqual((h.out.resolve() / "machine.jsonl", 30.0), s.args)
        self.assertEqual((True, True), (s.started, s.stopped))
        self.assertEqual(6, s.runs_done())  # the runs finished so far: the fake workers played all six
        self.assertEqual(6, summary["runs"])

    def test_a_sampler_that_cannot_start_does_not_stop_the_block(self):
        made, h, summary = self.play(start_error=OSError("no counters"), sample_machine=30.0)
        self.assertEqual((1, 6), (len(made), summary["runs"]))
        self.assertIn("machine sampler not started", h.err)

    def test_a_fresh_block_clears_an_old_machine_jsonl_and_a_resumed_one_keeps_it(self):
        from unittest import mock

        import machine_sampler

        class Quiet:
            def __init__(self, *a, **k):
                pass

            start = stop = lambda self: None  # noqa: E731

        for resume, kept in ((False, False), (True, True)):
            h = PlayResumeTest("test_the_parser_takes_resume_and_the_agent_identity")
            h.setUp()
            (h.out / "machine.jsonl").write_text('{"t": 1}\n', encoding="utf-8")
            if resume:
                h.write([h.row("S0")])
            with mock.patch.object(machine_sampler, "MachineSampler", Quiet), mock.patch.object(machine_sampler, "WinProbes", lambda *a, **k: object()):
                h.play(resume=resume, sample_machine=30.0)
            self.assertEqual(kept, (h.out / "machine.jsonl").exists(), resume)
            __import__("shutil").rmtree(h.tmp, True)


if __name__ == "__main__":
    unittest.main()
