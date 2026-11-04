"""referee/envproto.py: the channel protocol against a fake game (no game needed), and the bench summary."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
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



class FailedStartTest(unittest.TestCase):
    """A start that fails leaves no game process behind (the night of 2026-11-03 left one per timed-out start, 13 to 17 game processes for
    12 workers) and carries what the process had used on the exception (`boot_probe`)."""

    SILENT = "import time\nwhile True:\n    time.sleep(0.1)\n"  # never says ready
    EXITS = "import sys\nsys.exit(3)\n"  # gone before ready

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        dll = self.tmp / "game" / "data_sts2_windows_x86_64" / "sts2.dll"
        dll.parent.mkdir(parents=True)
        dll.write_bytes(b"fake")
        from unittest import mock

        patcher = mock.patch.dict(os.environ, {"SPIRELINE_REFEREE_KNOWN_SHA256": hashlib.sha256(b"fake").hexdigest()})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.procs = []

    def start(self, script, timeout):
        from unittest import mock

        (self.tmp / "fake.py").write_text(script, encoding="utf-8")
        real = envproto.rr.popen_no_wer

        def record(*a, **k):
            p = real(*a, **k)
            self.procs.append(p)
            return p

        with mock.patch.object(envproto.rr, "popen_no_wer", record), mock.patch.object(envproto, "BOOT_SAMPLE_S", 0.4):
            try:
                envproto.GameEnv(self.tmp / "game", self.tmp / "scratch", install=False, boot_timeout=timeout,
                                 command=[sys.executable, str(self.tmp / "fake.py")])
            except Exception as e:  # noqa: BLE001
                return e
        self.fail("the start should have failed")

    def test_a_start_that_times_out_is_killed_and_reports_what_it_had_used(self):
        e = self.start(self.SILENT, 1.5)
        self.assertIsInstance(e, TimeoutError)
        self.assertEqual("no message in 1.5 s", str(e))  # the message of the whole wait, as before
        (proc,) = self.procs
        self.assertIsNotNone(proc.poll())  # the process is gone, not left to finish starting
        probe = e.boot_probe
        self.assertIsNone(probe["exit"])  # it had not exited when it was given up
        self.assertGreaterEqual(len(probe["trace"]), 2)  # a reading every slice of the wait
        self.assertEqual(0, probe["log_b"])
        self.assertIn("scratch_files", probe)
        if sys.platform == "win32":
            self.assertGreaterEqual(probe["cpu_s"], 0.0)
            self.assertIn("read_ops", probe)
            self.assertTrue(all("cpu_s" in t for t in probe["trace"]))

    def test_a_process_that_exits_before_ready_is_a_game_died_with_its_exit_code(self):
        e = self.start(self.EXITS, 30)
        self.assertIsInstance(e, envproto.GameDied)
        self.assertIn("game exited", str(e))
        (proc,) = self.procs
        self.assertEqual(3, proc.poll())
        self.assertEqual(3, e.boot_probe["exit"])

    def test_a_silent_start_is_retried_on_a_fresh_process_and_the_failed_one_is_gone(self):
        """boot_game over real processes: the first start never says ready (the stall), the second does. Nothing of the first is left."""
        from unittest import mock

        flaky = ("import json, os, sys, time\nM = sys.argv[1]\nif not os.path.exists(M):\n    open(M, 'w').close()\n"
                 "    while True:\n        time.sleep(0.1)\n"
                 "sys.stdout.write('\\x1eENV ' + json.dumps({'t': 'ready'}) + '\\n')\nsys.stdout.flush()\n"
                 "for line in sys.stdin:\n    if json.loads(line).get('cmd') == 'quit':\n        break\n")
        (self.tmp / "flaky.py").write_text(flaky, encoding="utf-8")
        real = envproto.rr.popen_no_wer

        def record(*a, **k):
            proc = real(*a, **k)
            self.procs.append(proc)
            return proc

        def make(k):
            return envproto.GameEnv(self.tmp / "game", self.tmp / f"scratch{k}", install=False, boot_timeout=1.5,
                                    command=[sys.executable, str(self.tmp / "flaky.py"), str(self.tmp / "marker")])

        lines: list[dict] = []
        with mock.patch.object(envproto.rr, "popen_no_wer", record), mock.patch.object(envproto, "BOOT_SAMPLE_S", 0.4):
            env, tries, err = envproto.boot_game(make, 60.0, pause_s=0.0, say=lines.append)
        try:
            self.assertEqual((tries, err), (2, None))
            self.assertEqual(2, len(self.procs))
            self.assertIsNotNone(self.procs[0].poll())  # the silent one was killed, not left running
            self.assertIsNone(self.procs[1].poll())  # the second is the game
            self.assertEqual([(x["try"], x["ok"], x["final"]) for x in lines], [(1, False, False), (2, True, True)])
            self.assertIn("no message in 1.5 s", lines[0]["error"])
            self.assertTrue(lines[0]["trace"])
            self.assertLess(env.boot_s, 5.0)
        finally:
            if env is not None:
                env.close()

    def test_process_probe_reads_this_process_and_nothing_for_a_pid_that_does_not_exist(self):
        if sys.platform != "win32":
            self.assertEqual({}, envproto.process_probe(os.getpid()))
            return
        mine = envproto.process_probe(os.getpid())
        self.assertGreater(mine["cpu_s"], 0.0)
        self.assertGreater(mine["read_ops"], 0)
        self.assertEqual({}, envproto.process_probe(0x7FFFFFF0))


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


    def test_the_workers_are_handed_the_boot_budget_and_the_parser_takes_it(self):
        from types import SimpleNamespace
        from unittest import mock

        seen = []

        class Stop(Exception):
            pass

        class FakeProc:
            def __init__(self, target=None, args=()):
                seen.append(args)

            def start(self):
                raise Stop

        fake_ctx = SimpleNamespace(Queue=lambda: None, Process=FakeProc)
        f = Path(tempfile.mkdtemp()) / "block.json"
        f.write_text(json.dumps({"name": "x", "seeds": self.SEEDS}), encoding="utf-8")
        for over, want in (({}, envproto.BOOT_BUDGET_S), ({"boot_budget_s": 0.0}, 0.0), ({"boot_budget_s": 90.0}, 90.0)):
            seen.clear()
            a = SimpleNamespace(out=tempfile.mkdtemp(), game="G", character="IRONCLAD", runs=3, seeds_file=str(f), agent_seed=1,
                                stop_after_h=0, procs=1, sims=1, step_timeout=1, ascension=10, only=None, **over)
            with mock.patch.object(envproto.rr, "check_build", lambda g: None), mock.patch.object(envproto.rr, "install", lambda g, d: None), \
                    mock.patch.object(envproto.mp, "get_context", lambda _k: fake_ctx):
                with self.assertRaises(Stop):
                    envproto.play(a)
            self.assertEqual(want, seen[0][15])  # the last argument of play_worker
        base = ["play", "--game", "G", "--out", "O"]
        self.assertEqual(1200.0, envproto.parser().parse_args(base).boot_budget_s)
        self.assertEqual(0.0, envproto.parser().parse_args([*base, "--boot-budget-s", "0"]).boot_budget_s)


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
        self.extra: dict[int, list[dict]] = {}  # worker -> events its fake puts before its rows

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

        def fake_worker(wid, game, out, seeds, sims, q, step_timeout, ascension, agent_seed, run_ids, stop_at, character, recycle, run_deadline_s=0.0, wrapup_deadline_s=0.0, boot_budget_s=0.0):
            if not test.calls and (Path(out) / "divergences.jsonl").is_file():  # what the files hold while the first worker plays
                runs = [json.loads(x) for x in (Path(out) / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
                divs = (Path(out) / "divergences.jsonl").read_text(encoding="utf-8").splitlines()
                test.during = {"runs": [r["seed"] for r in runs], "divs": [json.loads(x)["seed"] for x in divs if x.strip()]}
            __import__("time").sleep(0.03)  # the clock of Windows ticks every 15 ms: a block has a wall
            test.calls.append((wid, list(seeds), list(run_ids), agent_seed, stop_at))
            for ev in test.extra.get(wid, []):
                q.put(dict(ev, w=wid))
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
        self.assertEqual(["agent_seed", "character", "engine", "sims", "runs", "crashes", "wrapup_terminated", "deadline_respawns", "died_respawns", "unplayed_after_cap", "boot_retries", "boot_gave_up", "wins", "ends", "mean_floor", "decisions",
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

    def test_boot_lines_go_to_boot_jsonl_and_the_summary_not_to_the_rows(self):
        """A worker's `boot` events (a start that failed, a retried one that worked) are no rows of the block: boot.jsonl keeps them whole,
        the summary counts the retries and the seeds given up, and runs.jsonl holds the rows alone."""
        fail = {"ev": "boot", "seed": "S0", "try": 1, "ok": False, "final": False, "s": 181.2, "error": "TimeoutError('no message in 180 s')",
                "cpu_s": 0.1, "read_mb": 2.0, "trace": [{"s": 30.0, "cpu_s": 0.1, "read_mb": 2.0}]}
        self.extra = {0: [fail, {**fail, "try": 2},
                          {"ev": "boot", "seed": "S0", "try": 3, "ok": True, "final": True, "s": 30.0, "boot_s": 29.5, "waited_s": 400.0}],
                      1: [{"ev": "boot", "seed": "S1", "try": 1, "ok": False, "final": True, "s": 180.5, "error": "GameDied('x')"}]}
        summary = self.play(resume=False)
        lines = [json.loads(x) for x in (self.out / "boot.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(b["w"], b["try"], b["ok"]) for b in lines], [(0, 1, False), (0, 2, False), (0, 3, True), (1, 1, False)])
        self.assertEqual((summary["boot_retries"], summary["boot_gave_up"]), (2, 1))
        self.assertEqual((summary["boot"]["failed_tries"], summary["boot"]["failed_s"], summary["boot"]["slowest_s"]), (3, 542.9, 181.2))
        self.assertEqual(len(summary["boot"]["lines"]), 4)
        self.assertEqual(sorted(r["seed"] for r in self.final_rows()), sorted(self.SEEDS))  # the rows are the block's alone
        self.assertTrue(all(r["ev"] == "end" for r in self.final_rows()))

    def test_a_block_with_no_boot_trouble_writes_no_boot_file_and_counts_zero(self):
        summary = self.play(resume=False)
        self.assertFalse((self.out / "boot.jsonl").exists())
        self.assertEqual((summary["boot_retries"], summary["boot_gave_up"]), (0, 0))
        self.assertNotIn("boot", summary)

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

    def play(self, scripts, boots=(1.5, 2.5), recycle=0, deadline=0.0, wrapup=0.0, stale=(), boot_budget=0.0):
        """Run play_worker over the seeds of `scripts` ({seed: ([(screen kind, action)...], exception or None)}) with a fake game
        process and a fake lockstep; the end and crash rows it queues."""
        import queue as queue_mod
        import time
        import types
        from unittest import mock

        test = self
        booted = []
        self.scratches = []  # the scratch folder of every start, failed or not

        class FakeEnv:
            def __init__(self, game, scratch, install=True):
                n = len(booted)
                booted.append(self)
                test.scratches.append(Path(scratch).name)
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
            return {"end": "death", "floor": 3, "asked": len(pairs), **({"stale": True} if seed in stale else {})}, ls

        q = queue_mod.Queue()
        lock = types.ModuleType("lockstep")
        lock.play_run = play_run
        self.fh = []  # what the worker asked of faulthandler, in order

        class FakeFaulthandler:
            @staticmethod
            def dump_traceback_later(timeout, repeat=False, file=None, exit=False):
                self.fh.append(("arm", timeout, repeat, file))

            @staticmethod
            def cancel_dump_traceback_later():
                self.fh.append(("cancel",))

        self.out_dir = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.out_dir, True)
        self.t0 = time.time()
        with mock.patch.dict(sys.modules, {"lockstep": lock}), mock.patch.object(envproto, "GameEnv", FakeEnv), \
                mock.patch.object(envproto, "faulthandler", FakeFaulthandler), \
                mock.patch.object(envproto, "below_normal_self", lambda: None), mock.patch.object(envproto.rr, "connections", lambda pid: []),                 mock.patch.object(envproto, "BOOT_PAUSE_S", 0.0):
            envproto.play_worker(0, "G", str(self.out_dir), list(scripts), 1, q, 1, 10, 7, None, 0.0, "IRONCLAD", recycle, deadline, wrapup,
                                 boot_budget)
        self.t1 = time.time()
        rows = []
        while not q.empty():
            rows.append(q.get())
        self.all_events = rows
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

    def test_each_run_begins_with_a_start_event_the_main_loops_run_clock_reads(self):
        """N5: the play loop's --run-deadline-s counts from the worker's start event of the run."""
        self.play({"S1": ([("map", 0)], None), "S2": ([("map", 0)], RuntimeError("boom"))})
        seq = [(r["ev"], r["seed"]) for r in self.all_events if r["ev"] in ("start", "end", "crash")]
        self.assertEqual([("start", "S1"), ("end", "S1"), ("start", "S2"), ("crash", "S2")], seq)
        for r in self.all_events:
            if r["ev"] == "start":
                self.assertEqual(0, r["w"])
                self.assertTrue(self.t0 - 1 <= r["t"] <= self.t1 + 1)

    def test_a_run_arms_a_stack_dump_into_the_workers_stack_file_and_cancels_it_when_it_ends(self):
        """N5: faulthandler writes all threads' stacks to out/scratch/worker-<wid>.stack two minutes before the deadline."""
        self.play({"S1": ([("map", 0)], None)}, deadline=1800.0)
        self.assertEqual(["arm", "cancel"], [c[0] for c in self.fh])  # once each
        _kind, timeout, repeat, f = self.fh[0]
        self.assertEqual((1680.0, False), (timeout, repeat))
        self.assertEqual(self.out_dir / "scratch" / "worker-0.stack", Path(f.name))
        self.assertEqual("a", f.mode)

    def test_every_run_ends_with_a_cancel_the_normal_end_and_the_crash_branches_alike(self):
        scripts = {"S1": ([("map", 0)], None), "S2": ([("map", 0)], RuntimeError("boom")), "S3": ([("map", 0)], envproto.GameDied("gone")),
                   "S4": ([("map", 0)], None)}
        self.play(scripts, boots=(1.5, 2.5, 3.5, 4.5), deadline=1800.0)
        self.assertEqual(["arm", "cancel"] * 4, [c[0] for c in self.fh])
        self.assertEqual({self.fh[0][3]}, {c[3] for c in self.fh if c[0] == "arm"})  # one file for the worker, opened once

    def test_the_dump_comes_at_least_60_s_into_the_run(self):
        self.play({"S1": ([("map", 0)], None)}, deadline=100.0)
        self.assertEqual(60, self.fh[0][1])

    def test_with_the_deadline_off_nothing_is_armed_and_no_stack_file_is_made(self):
        self.play({"S1": ([("map", 0)], None)}, deadline=0.0)
        self.assertEqual([], self.fh)
        self.assertFalse((self.out_dir / "scratch" / "worker-0.stack").exists())

    def test_a_boot_failure_row_has_the_time_and_the_restart_count(self):
        rows = self.play({"S1": ([], None)}, boots=(None,))
        self.assertEqual(("crash", "boot", 0), (rows[0]["ev"], rows[0]["where"], rows[0]["restarts"]))
        self.assertTrue(rows[0]["t_end"] > 0)

    def test_a_start_that_fails_is_retried_on_a_fresh_folder_and_the_seed_is_not_charged(self):
        """The night of 2026-11-03: seven starts timed out in a stall and each took its seed's row. A failed start is now tried again."""
        (row,) = self.play({"S1": ([("map", 0)], None)}, boots=(None, None, 2.5), boot_budget=100.0, deadline=1800.0)
        self.assertEqual(("end", 2.5, 2), (row["ev"], row["boot_s"], row["restarts"]))  # the run was played by the third start
        self.assertEqual(["p00r0", "p00r1", "p00r2"], self.scratches)  # each try its own folder
        boots = [r for r in self.all_events if r["ev"] == "boot"]
        self.assertEqual([(b["try"], b["ok"], b["final"], b["restarts"]) for b in boots], [(1, False, False, 0), (2, False, False, 1), (3, True, True, 2)])
        self.assertTrue(all(b["seed"] == "S1" and b["w"] == 0 and b["t"] > 0 for b in boots))
        self.assertIn("GameDied", boots[0]["error"])
        self.assertEqual(["start", "boot", "boot", "boot", "start", "end"], [r["ev"] for r in self.all_events if r["ev"] not in ("wrapup", "net", "done")])
        self.assertEqual(["arm", "arm", "cancel"], [c[0] for c in self.fh])  # the stack dump is armed again once the start worked

    def test_a_first_try_that_works_adds_no_event_so_the_normal_path_is_as_it_was(self):
        self.play({"S1": ([("map", 0)], None)}, boots=(1.5,), boot_budget=100.0)
        self.assertEqual([], [r for r in self.all_events if r["ev"] == "boot"])

    def test_a_slow_first_start_that_worked_is_reported(self):
        lines: list[dict] = []
        ticks = iter([0.0, 0.0, 45.0, 45.0])  # the first try's start twice (the budget's and the try's), its end, the wait in all
        env, tries, err = envproto.boot_game(lambda k: type("E", (), {"boot_s": 44.0})(), 100.0, clock=lambda: next(ticks), say=lines.append)
        self.assertEqual((tries, err), (1, None))
        self.assertEqual([(x["try"], x["ok"], x["s"], x["boot_s"]) for x in lines], [(1, True, 45.0, 44.0)])

    def test_every_try_failing_gives_the_seed_a_boot_row_and_the_next_seed_one_try_not_the_budget(self):
        from unittest import mock

        scripts = {"S1": ([], None), "S2": ([], None), "S3": ([("map", 0)], None)}
        # S1: 3 tries, all fail (the cap); S2: one try, fails; S3: one try, the game starts
        with mock.patch.object(envproto, "BOOT_TRIES_MAX", 3):
            rows = self.play(scripts, boots=(None, None, None, None, 2.0), boot_budget=100.0)
        self.assertEqual(["crash", "crash", "end"], [r["ev"] for r in rows])
        c1, c2, e3 = rows
        self.assertEqual(("S1", "boot", 3, 2), (c1["seed"], c1["where"], c1["boot_tries"], c1["restarts"]))
        self.assertEqual(("S2", "boot", 1, 3), (c2["seed"], c2["where"], c2["boot_tries"], c2["restarts"]))
        self.assertEqual(["p00r0", "p00r1", "p00r2", "p00r3", "p00r4"], self.scratches)
        self.assertEqual((4, 2.0), (e3["restarts"], e3["boot_s"]))
        boots = [(b["seed"], b["try"], b["ok"], b["final"]) for b in self.all_events if b["ev"] == "boot"]
        self.assertEqual(boots, [("S1", 1, False, False), ("S1", 2, False, False), ("S1", 3, False, True), ("S2", 1, False, True)])

    def test_the_budget_ends_the_tries_before_the_cap(self):
        ticks = [0.0]

        def now():
            return ticks[0]

        def sleep(s):
            ticks[0] += s

        def make(k):
            ticks[0] += 181.0  # each try waits out its 180 s
            raise TimeoutError("no message in 180 s")

        lines: list[dict] = []
        env, tries, err = envproto.boot_game(make, 400.0, pause_s=5.0, clock=now, sleep=sleep, say=lines.append)
        # 181 s and a pause, 181 s and a pause (372 s in, under 400), a third try; 553 s in, no fourth
        self.assertEqual((env, tries), (None, 3))
        self.assertIsInstance(err, TimeoutError)
        self.assertEqual([(x["try"], x["final"]) for x in lines], [(1, False), (2, False), (3, True)])
        self.assertEqual(ticks[0], 181.0 * 3 + 5.0 * 2)
        ticks[0] = 0.0  # a budget of 0 is the one try
        env, tries, err = envproto.boot_game(make, 0.0, clock=now, sleep=sleep, say=None)
        self.assertEqual((env, tries), (None, 1))

    def test_a_system_exit_is_not_a_start_that_failed(self):
        def make(k):
            raise SystemExit(2)  # an unknown sts2.dll: the worker ends, as it always did

        with self.assertRaises(SystemExit):
            envproto.boot_game(make, 100.0, sleep=lambda s: None)

    def test_the_worker_says_wrapup_after_its_last_result_and_before_the_final_close_and_done(self):
        """N5B: the main loop's wrap-up clock (--wrapup-deadline-s) starts at the worker's wrapup event."""
        self.play({"S1": ([("map", 0)], None), "S2": ([("map", 0)], RuntimeError("boom"))}, boots=(1.5, 2.5, 3.5))
        self.assertEqual(["start", "end", "start", "crash", "wrapup", "done"],
                         [r["ev"] for r in self.all_events if r["ev"] != "net"])
        self.assertLess([r["ev"] for r in self.all_events].index("crash"), [r["ev"] for r in self.all_events].index("wrapup"))
        (wrap,) = [r for r in self.all_events if r["ev"] == "wrapup"]
        self.assertEqual(0, wrap["w"])
        self.assertTrue(self.t0 - 1 <= wrap["t"] <= self.t1 + 1)

    def test_a_worker_with_no_seed_to_play_still_says_wrapup_before_done(self):
        self.play({})
        self.assertEqual(["wrapup", "done"], [r["ev"] for r in self.all_events])

    def test_a_stale_run_says_wrapup_before_its_cleanup_the_connection_listing_and_the_close(self):
        """The stale path lists connections and closes the game after the run's row, with no run in flight: its own clock."""
        self.play({"S1": ([("map", 0)], None), "S2": ([("map", 0)], None)}, boots=(1.5, 2.5), stale=("S1",))
        self.assertEqual(["start", "end", "wrapup", "net", "start", "end", "wrapup", "net", "done"], [r["ev"] for r in self.all_events])

    def test_the_wrapup_arms_a_stack_dump_into_the_same_file_a_minute_before_the_limit_and_cancels_it_on_done(self):
        self.play({"S1": ([("map", 0)], None)}, deadline=1800.0, wrapup=300.0)
        self.assertEqual(["arm", "cancel", "arm", "cancel"], [c[0] for c in self.fh])  # the run's, then the wrap-up's
        self.assertEqual((1680.0, 240.0), (self.fh[0][1], self.fh[2][1]))
        self.assertEqual((False, False), (self.fh[0][2], self.fh[2][2]))
        self.assertEqual(self.fh[0][3], self.fh[2][3])  # one file for the worker
        self.assertEqual(self.out_dir / "scratch" / "worker-0.stack", Path(self.fh[2][3].name))

    def test_the_wrapup_dump_is_at_least_60_s_in_and_works_with_the_run_deadline_off(self):
        self.play({"S1": ([("map", 0)], None)}, deadline=0.0, wrapup=100.0)
        self.assertEqual([("arm", 60), ("cancel",)], [(c[0], *c[1:2]) for c in self.fh])  # no run dump, one wrap-up dump
        self.assertTrue((self.out_dir / "scratch" / "worker-0.stack").exists())

    def test_a_stale_runs_cleanup_dump_is_cancelled_before_the_next_run(self):
        self.play({"S1": ([("map", 0)], None), "S2": ([("map", 0)], None)}, boots=(1.5, 2.5), stale=("S1",), deadline=0.0, wrapup=300.0)
        self.assertEqual(["arm", "cancel", "arm", "cancel"], [c[0] for c in self.fh])  # S1's cleanup, then the final wrap-up

    def test_with_the_wrapup_limit_off_the_event_is_still_sent_but_nothing_is_armed(self):
        self.play({"S1": ([("map", 0)], None)}, deadline=0.0, wrapup=0.0)
        self.assertEqual([], self.fh)
        self.assertIn("wrapup", [r["ev"] for r in self.all_events])


class SampleMachineFlagTest(unittest.TestCase):
    """L3: `play --sample-machine SECONDS` runs the machine sampler (machine.jsonl in the out folder) for the block; off by default."""

    def play(self, start_error=None, **over):
        from unittest import mock

        import machine_sampler

        made = []

        class FakeSampler:
            def __init__(self, path, interval, probes=None, runs_done=lambda: 0, **k):
                self.probes, self.kw = probes, k
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
        self.assertIsNone(s.probes)  # the readings are built by the sampler's own thread, through this factory
        self.assertTrue(callable(s.kw["make_probes"]))
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


class RemainingAfterTest(unittest.TestCase):
    """The seeds (and run ids) a replacement worker plays after a worker was closed for its run deadline: its share after the hung seed."""

    def test_the_seeds_and_run_ids_after_the_hung_one_in_the_workers_order(self):
        self.assertEqual((["C", "E"], [3, 4]), envproto.remaining_after(["A", "B", "C", "E"], [0, 2, 3, 4], "B"))
        self.assertEqual((["B", "C"], [1, 2]), envproto.remaining_after(["A", "B", "C"], [0, 1, 2], "A"))

    def test_the_last_seed_and_a_missing_seed_leave_nothing(self):
        self.assertEqual(([], []), envproto.remaining_after(["A", "B"], [0, 1], "B"))
        self.assertEqual(([], []), envproto.remaining_after(["A", "B"], [0, 1], "Z"))
        self.assertEqual(([], []), envproto.remaining_after([], [], "A"))

    def test_it_keeps_the_run_ids_of_a_share_that_is_not_a_prefix_of_the_block(self):  # a resumed block's share: ids are the full list's places
        self.assertEqual((["S9", "S12"], [30, 33]), envproto.remaining_after(["S3", "S6", "S9", "S12"], [6, 18, 30, 33], "S6"))


class RunDeadlineTest(unittest.TestCase):
    """N5: the play loop gives up a worker whose current run has no result after --run-deadline-s: the worker is terminated through its
    own Process object, its seed gets a crash row (where: deadline) with the agent seed the worker would have written, and the block
    closes with every other worker's rows. The workers are threads that put the events a real worker puts (start, end, done); a hung
    worker puts a start event that is older than the deadline and then waits until it is terminated."""

    SEEDS = ["S0", "S1", "S2", "S3", "S4", "S5"]
    AS = 77
    AGENT = {"generation": 3, "dir": "gen3/c", "prior": True, "prior_sha256": "ab", "macro": "rules"}

    def setUp(self):
        import collections
        import threading

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        self.out = self.tmp / "block"
        self.out.mkdir()
        self.seeds_file = self.tmp / "block.json"
        self.seeds_file.write_text(json.dumps({"name": "x", "seeds": self.SEEDS}), encoding="utf-8")
        self.handed: list[tuple] = []
        self.terminated: list[int] = []
        self.release = collections.defaultdict(threading.Event)
        self.live_at_join = ""
        self.live_procs: list = []
        self.peak_live = 0
        self.deadline_seen: list[float] = []
        self.wrapup_seen: list[float] = []
        self.exit_codes: dict[int, int] = {}  # worker -> the exit code its (fake) process reports once it has ended; absent: 0
        self.sync_start = False  # True: a worker runs to its end inside start(), so the main loop begins with every process gone and every row queued
        from unittest import mock
        patch = mock.patch.object(envproto, "POLL_S", 0.05, create=True)  # the main loop's queue wait: short, so a death is seen at once
        patch.start()
        self.addCleanup(patch.stop)

    def row(self, seed, ix, w):
        return {"ev": "end", "w": w, "seed": seed, "character": "IRONCLAD", "end": "death", "floor": 10 + ix, "asked": 5, "s": 2.0,
                "stats": {"decisions": 100, "ranked": 40, "diverged": 0, "reanchors": 0, "reanchor_failed": 0, "first_loss": None},
                "lost": None, "divergences": [], "agent_seed": [self.AS, ix], "assumed": 0}

    def normal(self, wid, seeds, run_ids, q, start_age=0.0, pause=0.0, done=True):
        """Play the seeds as a worker does: a start event, then the end row. start_age: how long ago the first run began."""
        for k, (s, ix) in enumerate(zip(seeds, run_ids)):
            q.put({"ev": "start", "w": wid, "seed": s, "t": time.time() - (start_age if k == 0 else 0.0)})
            time.sleep(pause)
            q.put(self.row(s, ix, wid))
        if done:
            q.put({"ev": "wrapup", "w": wid, "t": time.time()})
            q.put({"ev": "done", "w": wid})

    def hang(self, after=0, age=5000.0, stack_text=None):
        """Worker 0 plays `after` seeds, then begins one `age` seconds ago and never ends it (until terminated); the others play.
        `stack_text`: what its faulthandler has written to out/scratch/worker-0.stack by then."""
        def behave(wid, seeds, run_ids, q):
            if wid != 0:
                return self.normal(wid, seeds, run_ids, q)
            if stack_text is not None:
                (self.out / "scratch").mkdir(exist_ok=True)
                (self.out / "scratch" / "worker-0.stack").write_text(stack_text, encoding="utf-8")
            self.normal(wid, seeds[:after], run_ids[:after], q, done=False)
            q.put({"ev": "start", "w": wid, "seed": seeds[after], "t": time.time() - age})
            self.release[wid].wait(10)  # a block without the deadline fails here by its rows, not by hanging the suite
        return behave

    def stuck_wrapup(self, age=1000.0):
        """Worker 0 plays its seeds to the end, says wrapup `age` seconds ago and never says done (until terminated)."""
        def behave(wid, seeds, run_ids, q):
            if wid != 0:
                return self.normal(wid, seeds, run_ids, q)
            self.normal(wid, seeds, run_ids, q, done=False)
            q.put({"ev": "wrapup", "w": wid, "t": time.time() - age})
            self.release[wid].wait(10)
        return behave

    def play(self, behave=None, procs=2, resume=False, deadline=1800.0, wrapup=300.0, cap=3, stop_after_h=0):
        import contextlib
        import io
        import queue as queue_mod
        import threading
        from types import SimpleNamespace
        from unittest import mock

        test = self
        behave = behave or (lambda wid, seeds, run_ids, q: test.normal(wid, seeds, run_ids, q))

        def fake_worker(wid, game, out, seeds, sims, q, step_timeout, ascension, agent_seed, run_ids, stop_at, character, recycle, run_deadline_s=0.0, wrapup_deadline_s=0.0, boot_budget_s=0.0):
            test.handed.append((wid, list(seeds)))
            test.deadline_seen.append(run_deadline_s)
            test.wrapup_seen.append(wrapup_deadline_s)
            behave(wid, list(seeds), list(run_ids), q)

        class FakeProc:
            def __init__(self, target=None, args=()):
                self.target, self.args, self.thread = target, args, None

            def start(self):
                self.thread = threading.Thread(target=self.target, args=self.args, daemon=True)
                self.thread.start()
                if test.sync_start:
                    self.thread.join()
                test.live_procs.append(self)
                test.peak_live = max(test.peak_live, sum(x.is_alive() for x in test.live_procs))  # the most workers alive at once

            def is_alive(self):
                return self.thread.is_alive()

            @property
            def exitcode(self):  # as multiprocessing's: None while it runs
                return None if self.thread.is_alive() else test.exit_codes.get(self.args[0], 0)

            def terminate(self):
                test.terminated.append(self.args[0])
                test.release[self.args[0]].set()

            def join(self, timeout=None):
                self.thread.join(timeout)
                test.live_at_join = (test.out / "runs.jsonl").read_text(encoding="utf-8")  # the live file, as the block closes

        ctx = SimpleNamespace(Queue=queue_mod.Queue, Process=FakeProc)
        a = SimpleNamespace(out=str(self.out), game="G", character="IRONCLAD", runs=6, seeds_file=str(self.seeds_file), agent_seed=self.AS,
                            stop_after_h=stop_after_h, procs=procs, sims="a3-v1", step_timeout=1, ascension=10, engine="core2", only=None, recycle=0,
                            run_deadline_s=deadline, wrapup_deadline_s=wrapup, deadline_respawn_cap=cap)
        if resume:
            a.resume, a.agent_identity = True, json.dumps(self.AGENT)
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(envproto.rr, "check_build", lambda g: None), mock.patch.object(envproto.rr, "install", lambda g, d: None), \
                mock.patch.object(envproto.mp, "get_context", lambda _k: ctx), mock.patch.object(envproto, "play_worker", fake_worker), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            started = time.time()
            envproto.play(a)
        self.elapsed = time.time() - started
        self.stdout, self.stderr = stdout.getvalue(), stderr.getvalue()
        return json.loads((self.out / "summary.json").read_text(encoding="utf-8"))

    def final_rows(self):
        return [json.loads(x) for x in (self.out / "runs.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]

    def test_a_worker_past_the_deadline_is_terminated_its_seed_gets_a_crash_row_and_the_block_closes(self):
        summary = self.play(self.hang(after=1), cap=0)  # worker 0: S0 ends, then S2 is begun 5000 s ago and never ends; worker 1: S1 S3 S5
        self.assertEqual([0], self.terminated)  # through its own Process object, once
        rows = {r["seed"]: r for r in self.final_rows()}
        self.assertEqual({"S0", "S1", "S2", "S3", "S5"}, set(rows))  # S4 was never played: the worker was stopped
        self.assertEqual(["end"] * 4, [rows[s]["ev"] for s in ("S0", "S1", "S3", "S5")])  # the other rows are intact
        c = rows["S2"]
        self.assertEqual(("crash", "deadline", 0), (c["ev"], c["where"], c["w"]))
        self.assertEqual([self.AS, 2], c["agent_seed"])  # as the worker writes it: [agent seed, the seed's index in the block]
        n = int(c["error"].removeprefix("worker deadline: no result after ").removesuffix(" s"))
        self.assertTrue(5000 <= n <= 5030, c["error"])
        self.assertEqual((4, 1), (summary["runs"], summary["crashes"]))
        self.assertEqual((0, 1), (summary["deadline_respawns"], summary["unplayed_after_cap"]))  # cap 0: S4 stays unplayed and the block says so
        self.assertEqual("deadline", summary["errors"][0]["where"])
        self.assertNotIn("stack", c)  # the worker left no stack file
        self.assertIn('"where": "deadline"', self.live_at_join)  # also in the live runs.jsonl, before the block's rewrite
        self.assertEqual(1, sum(line.startswith("worker 0: worker deadline") for line in self.stdout.splitlines()))  # one line printed

    def test_a_replacement_worker_plays_the_rest_of_the_closed_workers_share_with_the_same_run_ids(self):
        summary = self.play(self.hang(after=1))  # worker 0: S0 ends, S2 hangs, S4 is behind it; worker 1: S1 S3 S5
        self.assertEqual([0], self.terminated)  # the hung worker only, through its own Process object
        handed = {wid: seeds for wid, seeds in self.handed}
        self.assertEqual({0: ["S0", "S2", "S4"], 1: ["S1", "S3", "S5"], 2: ["S4"]}, handed)  # a new worker id (2), the seeds after the hung one
        rows = {r["seed"]: r for r in self.final_rows()}
        self.assertEqual(["S0", "S1", "S2", "S3", "S4", "S5"], sorted(rows))  # the seed list is played once: no seed twice, none missing
        self.assertEqual(6, len(self.final_rows()))
        self.assertEqual(("crash", "deadline", 0), (rows["S2"]["ev"], rows["S2"]["where"], rows["S2"]["w"]))  # the hung seed is not played again
        self.assertEqual(("end", 2, [self.AS, 4]), (rows["S4"]["ev"], rows["S4"]["w"], rows["S4"]["agent_seed"]))  # the run id S4 has in the block
        self.assertEqual((5, 1, 1, 0), (summary["runs"], summary["crashes"], summary["deadline_respawns"], summary["unplayed_after_cap"]))
        self.assertLessEqual(self.peak_live, 2)  # never more than --procs workers at once: the closed one is gone before the new one starts
        self.assertEqual(1, self.stdout.count("replacement worker 2"))
        self.assertEqual([self.deadline_seen[0]] * 3, self.deadline_seen)  # the replacement is handed the same deadline settings
        self.assertEqual([self.wrapup_seen[0]] * 3, self.wrapup_seen)

    def test_a_replacement_that_hangs_too_is_replaced_again_up_to_the_cap_then_the_rest_stays_unplayed(self):
        def behave(wid, seeds, run_ids, q):  # one slot: each worker (0, then its replacements 1, 2, 3) hangs on the first seed it is handed
            q.put({"ev": "start", "w": wid, "seed": seeds[0], "t": time.time() - 5000.0})
            self.release[wid].wait(10)

        summary = self.play(behave, procs=1)
        self.assertEqual([0, 1, 2, 3], self.terminated)  # each one through its own Process object
        self.assertEqual([(0, ["S0", "S1", "S2", "S3", "S4", "S5"]), (1, ["S1", "S2", "S3", "S4", "S5"]), (2, ["S2", "S3", "S4", "S5"]),
                          (3, ["S3", "S4", "S5"])], self.handed)  # three replacements and no fourth
        rows = sorted(self.final_rows(), key=lambda r: r["seed"])
        self.assertEqual(["S0", "S1", "S2", "S3"], [r["seed"] for r in rows])
        self.assertEqual({"deadline"}, {r["where"] for r in rows})
        self.assertEqual([[self.AS, i] for i in range(4)], [r["agent_seed"] for r in rows])  # a replacement's rows carry the run ids of the block
        self.assertEqual((0, 4, 3, 2), (summary["runs"], summary["crashes"], summary["deadline_respawns"], summary["unplayed_after_cap"]))
        self.assertLessEqual(self.peak_live, 1)

    def test_the_cap_counts_per_original_worker_slot(self):
        def behave(wid, seeds, run_ids, q):  # slot 0 (worker 0) hangs once, then its replacement plays; slot 1 (worker 1) plays
            if wid == 0:
                self.hang(after=0)(wid, seeds, run_ids, q)
            else:
                self.normal(wid, seeds, run_ids, q)

        summary = self.play(behave, cap=1)
        self.assertEqual((5, 1, 1, 0), (summary["runs"], summary["crashes"], summary["deadline_respawns"], summary["unplayed_after_cap"]))
        self.assertEqual(["S0", "S1", "S2", "S3", "S4", "S5"], sorted(r["seed"] for r in self.final_rows()))

    def test_a_worker_that_hung_on_its_last_seed_has_no_replacement_and_a_wrapup_deadline_never_has_one(self):
        summary = self.play(self.hang(after=2))  # worker 0: S0 S2 end, S4 (its last) hangs
        self.assertEqual([0], self.terminated)
        self.assertEqual([0, 1], sorted(wid for wid, _s in self.handed))  # nobody to replace it with: S4 is the end of its share
        self.assertEqual((0, 0), (summary["deadline_respawns"], summary["unplayed_after_cap"]))
        self.handed.clear()
        self.terminated.clear()
        self.release.clear()  # the first half's terminate left worker 0's event set: this worker would end at once, a gone process and not a stuck one
        summary = self.play(self.stuck_wrapup())  # every run of worker 0 has its row: the wrap-up deadline stops it and starts nothing
        self.assertEqual([0], self.terminated)
        self.assertEqual([0, 1], sorted(wid for wid, _s in self.handed))
        self.assertEqual((6, 1, 0, 0), (summary["runs"], summary["wrapup_terminated"], summary["deadline_respawns"], summary["unplayed_after_cap"]))

    def test_a_resume_after_a_block_that_left_seeds_unplayed_plays_them_and_not_the_deadline_seed(self):
        self.play(self.hang(after=0), resume=True, cap=0)  # worker 0 hangs on S0 and has no replacement: S2 and S4 have no row
        self.handed.clear()
        self.terminated.clear()
        self.play(self.hang(after=0), procs=1, resume=True)  # worker 0 gets S2 S4 and hangs on S2: S4 goes to a replacement, S0 stays a crash row
        self.assertEqual({0: ["S2", "S4"], 1: ["S4"]}, dict(self.handed))
        rows = {r["seed"]: r for r in self.final_rows()}
        self.assertEqual({"S0": "crash", "S1": "end", "S2": "crash", "S3": "end", "S4": "end", "S5": "end"}, {s: r["ev"] for s, r in rows.items()})
        self.assertEqual([self.AS, 4], rows["S4"]["agent_seed"])

    def test_the_deadline_row_carries_the_last_stack_dump_of_the_workers_file_without_any_directory(self):
        lib = os.path.join(str(self.tmp), "lib")
        dump = lambda *frames: "".join(f'  File "{os.path.join(lib, f)}", line {n} in {fn}\n' for f, n, fn in frames)  # noqa: E731
        text = ("Timeout (0:28:00)!\nThread 0x0000dead (most recent call first):\n" + dump(("old.py", 1, "old")) + "\n"
                + "Timeout (0:28:00)!\nThread 0x00001111 (most recent call first):\n" + dump(("envproto.py", 117, "_reader"))
                + '  File "D:/elsewhere/deep/threading.py", line 9 in run\n\n'
                + "Current thread 0x00002222 [MainThread] (most recent call first):\n"
                + dump(("live_search.py", 511, "_step"), ("lockstep.py", 1342, "play_run"), ("<string>", 3, "<module>")))
        self.play(self.hang(after=1, stack_text=text))
        (c,) = [r for r in self.final_rows() if r.get("where") == "deadline"]
        self.assertEqual(["Thread 0x00001111", "envproto.py:117 in _reader", "threading.py:9 in run",
                          "Thread 0x00002222", "live_search.py:511 in _step", "lockstep.py:1342 in play_run", "<string>:3 in <module>"],
                         c["stack"])
        self.assertFalse(any("/" in x or "\\" in x for x in c["stack"]))
        self.assertNotIn("old", " ".join(c["stack"]))  # only the last dump

    def test_a_dump_without_the_timeout_marker_is_read_whole_and_a_long_one_is_cut_to_80_lines_with_every_thread_kept(self):
        one = 'Thread 0x1 (most recent call first):\n  File "/a/b.py", line 4 in f\n'
        self.assertEqual(["Thread 0x1", "b.py:4 in f"], envproto.reduce_stack(one))
        many = "Timeout (0:1:00)!\n" + "".join(
            f"Thread 0x{t:x} (most recent call first):\n" + "".join(f'  File "C:\\x\\m.py", line {i} in g{i}\n' for i in range(30))
            for t in range(10))
        out = envproto.reduce_stack(many)
        self.assertEqual(80, len(out))
        self.assertEqual(10, sum(x.startswith("Thread ") for x in out))
        self.assertEqual("m.py:0 in g0", out[1])  # the innermost frames stay
        self.assertEqual(80, len(envproto.reduce_stack("".join(f'  File "m.py", line {i} in g\n' for i in range(200)))))

    def test_a_worker_within_the_deadline_is_left_alone(self):
        def slow(wid, seeds, run_ids, q):
            self.normal(wid, seeds, run_ids, q, start_age=10.0, pause=0.4 if wid == 0 else 0.0)

        summary = self.play(slow, deadline=100.0)
        self.assertEqual([100.0, 100.0], self.deadline_seen)  # each worker is handed the deadline (it arms its stack dump from it)
        self.assertEqual([], self.terminated)
        self.assertEqual((6, 0), (summary["runs"], summary["crashes"]))
        self.assertEqual({"end"}, {r["ev"] for r in self.final_rows()})  # no start event is a row

    def test_deadline_zero_disables_it(self):
        def old_but_slow(wid, seeds, run_ids, q):
            self.normal(wid, seeds, run_ids, q, start_age=5000.0, pause=0.4 if wid == 0 else 0.0)

        summary = self.play(old_but_slow, deadline=0)
        self.assertEqual([], self.terminated)
        self.assertEqual((6, 0), (summary["runs"], summary["crashes"]))
        self.assertEqual({"end"}, {r["ev"] for r in self.final_rows()})  # no start event is a row

    def test_a_resume_after_a_deadline_row_does_not_replay_that_seed(self):
        self.play(self.hang(after=0), resume=True, cap=0)  # S0 (worker 0's first) hits the deadline; S2 and S4 were not reached (cap 0: no replacement)
        self.assertEqual({"S0": "crash", "S1": "end", "S3": "end", "S5": "end"}, {r["seed"]: r["ev"] for r in self.final_rows()})
        self.terminated.clear()
        self.handed.clear()
        summary = self.play(resume=True)
        self.assertEqual(["S2", "S4"], sorted(x for _w, xs in self.handed for x in xs))  # the unreached seeds play, the deadline seed does not
        rows = self.final_rows()
        self.assertEqual(["S0", "S1", "S2", "S3", "S4", "S5"], sorted(r["seed"] for r in rows))
        (s0,) = [r for r in rows if r["seed"] == "S0"]
        self.assertEqual("deadline", s0["where"])
        self.assertEqual([self.AS, 4], [r for r in rows if r["seed"] == "S4"][0]["agent_seed"])
        self.assertEqual((4, 6, 1), (summary["resumed_rows"], summary["runs"] + summary["crashes"], summary["crashes"]))

    def test_a_worker_with_no_run_in_flight_is_never_overdue_and_a_closed_workers_late_rows_are_ignored(self):
        w = envproto.RunDeadline(1800.0, self.AS, {0: {"S0": 0, "S2": 2}})
        self.assertTrue(w.seen({"ev": "start", "w": 0, "seed": "S0", "t": 100.0}, 100.0))
        self.assertTrue(w.seen({"ev": "end", "w": 0, "seed": "S0"}, 150.0))  # past stop_at, between runs: nothing in flight
        self.assertEqual([], w.overdue(10 ** 7))
        self.assertTrue(w.seen({"ev": "start", "w": 0, "seed": "S2", "t": 200.0}, 200.0))
        self.assertEqual([], w.overdue(2000.0))  # 1800 s exactly is not past it
        self.assertEqual([0], w.overdue(2001.0))
        row = w.close(0, 2001.0)
        self.assertEqual(("S2", "deadline", [self.AS, 2]), (row["seed"], row["where"], row["agent_seed"]))
        self.assertFalse(w.seen({"ev": "end", "w": 0, "seed": "S2"}, 2002.0))
        self.assertEqual([], w.overdue(10 ** 7))

    def test_a_worker_stuck_in_its_wrapup_is_terminated_and_counted_with_no_crash_row_and_the_block_closes(self):
        """N5B: the worker's last result is put, then it hangs (the connection listing, the game close): after --wrapup-deadline-s the
        main loop stops it as it stops a hung run, writes no row (every run has one), and the block closes without waiting."""
        summary = self.play(self.stuck_wrapup(age=1000.0))
        self.assertEqual([0], self.terminated)  # through its own Process object, once
        self.assertEqual([300.0, 300.0], self.wrapup_seen)  # each worker is handed the limit (it arms its stack dump from it)
        self.assertEqual(1, summary["wrapup_terminated"])
        self.assertEqual((6, 0), (summary["runs"], summary["crashes"]))  # all six runs have their end row, no crash row
        self.assertEqual({"end"}, {r["ev"] for r in self.final_rows()})
        self.assertEqual({"end"}, {json.loads(x)["ev"] for x in self.live_at_join.split("\n") if x.strip()})  # nor in the live file
        self.assertLess(self.elapsed, 4.0)  # `done` reached the worker count: no wait for the queue to run dry
        lines = [x for x in self.stderr.split("\n") if "wrap-up" in x]
        self.assertEqual(1, len(lines))
        self.assertRegex(lines[0], r"^worker 0 wrap-up deadline after 10\d\d s: terminated$")

    def test_a_wrapup_that_is_over_in_time_is_left_alone(self):
        def slow(wid, seeds, run_ids, q):
            self.normal(wid, seeds, run_ids, q, done=False)
            q.put({"ev": "wrapup", "w": wid, "t": time.time() - 100.0})
            time.sleep(0.4 if wid == 0 else 0.0)
            q.put({"ev": "done", "w": wid})

        summary = self.play(slow, wrapup=300.0)
        self.assertEqual([], self.terminated)
        self.assertEqual((6, 0, 0), (summary["runs"], summary["crashes"], summary["wrapup_terminated"]))
        self.assertEqual("", "".join(x for x in self.stderr.split("\n") if "wrap-up" in x))

    def test_wrapup_deadline_zero_arms_nothing(self):
        def old_but_slow(wid, seeds, run_ids, q):
            self.normal(wid, seeds, run_ids, q, done=False)
            q.put({"ev": "wrapup", "w": wid, "t": time.time() - 5000.0})
            time.sleep(0.4 if wid == 0 else 0.0)
            q.put({"ev": "done", "w": wid})

        summary = self.play(old_but_slow, wrapup=0)
        self.assertEqual([0.0, 0.0], self.wrapup_seen)
        self.assertEqual([], self.terminated)
        self.assertEqual((6, 0, 0), (summary["runs"], summary["crashes"], summary["wrapup_terminated"]))
        self.assertNotIn("wrap-up", self.stderr)

    def test_a_stale_runs_wrapup_clock_ends_with_the_workers_next_start(self):
        w = envproto.RunDeadline(1800.0, self.AS, {0: {"S0": 0, "S2": 2}}, wrapup_s=300.0)
        self.assertTrue(w.seen({"ev": "start", "w": 0, "seed": "S0", "t": 100.0}, 100.0))
        self.assertTrue(w.seen({"ev": "end", "w": 0, "seed": "S0"}, 150.0))
        self.assertEqual([], w.overdue_wrapup(10 ** 7))  # no wrapup said: nothing to clock
        self.assertTrue(w.seen({"ev": "wrapup", "w": 0, "t": 160.0}, 160.0))
        self.assertEqual([], w.overdue(10 ** 7))  # the wrap-up is not a run in flight: the run clock stays out of it
        self.assertEqual([], w.overdue_wrapup(460.0))  # 300 s exactly is not past it
        self.assertEqual([0], w.overdue_wrapup(461.0))
        self.assertTrue(w.seen({"ev": "start", "w": 0, "seed": "S2", "t": 470.0}, 470.0))  # it went on to another run
        self.assertEqual([], w.overdue_wrapup(10 ** 7))
        self.assertEqual([0], w.overdue(2271.0))  # and that run is clocked as ever

    def test_the_wrapup_clock_ends_with_done_and_a_closed_worker_is_ignored_and_gets_no_row(self):
        w = envproto.RunDeadline(1800.0, self.AS, {0: {}, 1: {}}, wrapup_s=300.0)
        w.seen({"ev": "wrapup", "w": 0, "t": 100.0}, 100.0)
        w.seen({"ev": "wrapup", "w": 1, "t": 100.0}, 100.0)
        w.seen({"ev": "done", "w": 0}, 200.0)
        self.assertEqual([1], w.overdue_wrapup(10 ** 7))
        self.assertEqual(901, w.close_wrapup(1, 1001.0))  # the seconds in the wrap-up; a number, no crash row
        self.assertFalse(w.seen({"ev": "done", "w": 1}, 1002.0))
        self.assertEqual([], w.overdue_wrapup(10 ** 7))
        off = envproto.RunDeadline(1800.0, self.AS, {0: {}}, wrapup_s=0)
        off.seen({"ev": "wrapup", "w": 0, "t": 1.0}, 1.0)
        self.assertEqual([], off.overdue_wrapup(10 ** 7))

    def test_the_parser_takes_the_wrapup_deadline_with_the_default_of_300(self):
        base = ["play", "--game", "G", "--out", "O"]
        self.assertEqual(300.0, envproto.parser().parse_args(base).wrapup_deadline_s)
        self.assertEqual(0.0, envproto.parser().parse_args([*base, "--wrapup-deadline-s", "0"]).wrapup_deadline_s)

    def test_the_parser_takes_the_deadline_with_the_default_of_1800(self):
        base = ["play", "--game", "G", "--out", "O"]
        self.assertEqual(1800.0, envproto.parser().parse_args(base).run_deadline_s)
        self.assertEqual(0.0, envproto.parser().parse_args([*base, "--run-deadline-s", "0"]).run_deadline_s)

    def test_the_parser_takes_the_deadline_respawn_cap_with_the_default_of_3(self):
        base = ["play", "--game", "G", "--out", "O"]
        self.assertEqual(3, envproto.parser().parse_args(base).deadline_respawn_cap)
        self.assertEqual(0, envproto.parser().parse_args([*base, "--deadline-respawn-cap", "0"]).deadline_respawn_cap)

    # ---- a worker whose process dies (the emulator's fail-fast) posts nothing: the pool notices it and hands the rest on

    DIED_CODE = 3221226505  # 0xc0000409, the fail-fast exit code of the 2026-10-31 night

    def dying(self, who=(0,), after=0, code=DIED_CODE):
        """Behave so that the workers in `who` (None: every worker) play `after` seeds, begin the next one, and their process exits with `code` and no
        `done`; the others play. The thread's end is the process's end: the main loop sees it through is_alive and exitcode."""
        def behave(wid, seeds, run_ids, q):
            if who is not None and wid not in who:
                return self.normal(wid, seeds, run_ids, q)
            self.normal(wid, seeds[:after], run_ids[:after], q, done=False)
            q.put({"ev": "start", "w": wid, "seed": seeds[after], "t": time.time()})
            self.exit_codes[wid] = code
        return behave

    def hang_on_first(self, wid, seeds, q):
        """Play as a worker that begins its first seed 5000 s ago and never ends it (until terminated)."""
        q.put({"ev": "start", "w": wid, "seed": seeds[0], "t": time.time() - 5000.0})
        self.release[wid].wait(10)

    def test_a_worker_whose_process_exits_without_done_gets_a_died_row_and_a_replacement_plays_the_rest(self):
        for sync in (False, True):  # sync: the process is gone and every row is queued before the main loop looks (the drain comes first)
            with self.subTest(sync_start=sync):
                self.handed.clear()
                self.terminated.clear()
                self.live_procs.clear()
                self.deadline_seen.clear()
                self.sync_start = sync
                summary = self.play(self.dying(after=1))  # worker 0: S0 ends, S2 begins and the process exits 0xc0000409; worker 1: S1 S3 S5
                self.assertEqual([], self.terminated)  # a dead process is not terminated
                self.assertEqual({0: ["S0", "S2", "S4"], 1: ["S1", "S3", "S5"], 2: ["S4"]}, dict(self.handed))  # a new worker id, the seeds after S2
                rows = {r["seed"]: r for r in self.final_rows()}
                self.assertEqual(["S0", "S1", "S2", "S3", "S4", "S5"], sorted(rows))
                self.assertEqual(6, len(self.final_rows()))  # S0's row (put before the death) is kept once; no seed twice
                self.assertEqual(("end", 0), (rows["S0"]["ev"], rows["S0"]["w"]))
                c = rows["S2"]
                self.assertEqual(("crash", "died", 0), (c["ev"], c["where"], c["w"]))
                self.assertEqual([self.AS, 2], c["agent_seed"])  # as the deadline row and the worker's rows write it
                self.assertIn("worker process exited 3221226505", c["error"])
                self.assertIn("0xc0000409", c["error"])
                self.assertEqual((3221226505, []), (c["exitcode"], c["divergences"]))
                self.assertIn("t_end", c)
                self.assertIn("s", c)
                self.assertEqual(("end", 2, [self.AS, 4]), (rows["S4"]["ev"], rows["S4"]["w"], rows["S4"]["agent_seed"]))  # the run id S4 has in the block
                self.assertEqual((5, 1, 0, 1, 0), (summary["runs"], summary["crashes"], summary["deadline_respawns"], summary["died_respawns"],
                                                   summary["unplayed_after_cap"]))
                self.assertEqual("died", summary["errors"][0]["where"])
                self.assertLessEqual(self.peak_live, 2)
                self.assertIn('"where": "died"', self.live_at_join)  # also in the live runs.jsonl
                self.assertEqual(1, self.stdout.count("replacement worker 2"))
                self.assertEqual([self.deadline_seen[0]] * 3, self.deadline_seen)  # the replacement is handed the same deadline settings
                self.assertEqual(1, sum(line.startswith("worker 0: worker process exited") for line in self.stdout.splitlines()))
                self.assertLess(self.elapsed, 4.0)  # seen at once, not at a deadline

    def test_deaths_past_the_cap_leave_the_rest_unplayed_and_the_summary_counts_it(self):
        summary = self.play(self.dying(who=None), procs=1)  # one slot: every worker dies on the first seed it is handed
        self.assertEqual([(0, ["S0", "S1", "S2", "S3", "S4", "S5"]), (1, ["S1", "S2", "S3", "S4", "S5"]), (2, ["S2", "S3", "S4", "S5"]),
                          (3, ["S3", "S4", "S5"])], self.handed)  # three replacements and no fourth
        rows = sorted(self.final_rows(), key=lambda r: r["seed"])
        self.assertEqual(["S0", "S1", "S2", "S3"], [r["seed"] for r in rows])
        self.assertEqual({"died"}, {r["where"] for r in rows})
        self.assertEqual([[self.AS, i] for i in range(4)], [r["agent_seed"] for r in rows])
        self.assertEqual((0, 4, 0, 3, 2), (summary["runs"], summary["crashes"], summary["deadline_respawns"], summary["died_respawns"],
                                          summary["unplayed_after_cap"]))
        self.assertIn("2 remaining seeds stay unplayed", self.stdout)

    def test_a_cap_of_zero_means_no_replacement_for_a_death_either(self):
        summary = self.play(self.dying(), cap=0)  # worker 0 dies on S0; S2 and S4 stay unplayed
        self.assertEqual([0, 1], sorted(wid for wid, _s in self.handed))
        self.assertEqual((0, 2), (summary["died_respawns"], summary["unplayed_after_cap"]))
        self.assertEqual(["S0", "S1", "S3", "S5"], sorted(r["seed"] for r in self.final_rows()))

    def test_a_worker_that_posts_done_and_exits_is_never_read_as_dead(self):
        for sync in (False, True):  # sync: the process is gone, and its rows and its done are still in the queue when the loop looks
            with self.subTest(sync_start=sync):
                self.handed.clear()
                self.sync_start = sync
                self.exit_codes.update({0: self.DIED_CODE, 1: 1})  # even a bad exit code after the done (a fault at interpreter shutdown) is no death
                summary = self.play()
                self.assertEqual([0, 1], sorted(wid for wid, _s in self.handed))
                self.assertEqual({"end"}, {r["ev"] for r in self.final_rows()})
                self.assertEqual(6, len(self.final_rows()))
                self.assertEqual((6, 0, 0, 0, 0), (summary["runs"], summary["crashes"], summary["deadline_respawns"], summary["died_respawns"],
                                                   summary["unplayed_after_cap"]))

    def test_a_process_that_exits_after_its_last_row_or_in_its_wrapup_is_left_alone_and_the_block_closes(self):
        def gone(stage):
            def behave(wid, seeds, run_ids, q):
                if wid != 0:
                    return self.normal(wid, seeds, run_ids, q)
                self.normal(wid, seeds, run_ids, q, done=False)  # every run of worker 0 has its row
                if stage == "wrapup":
                    q.put({"ev": "wrapup", "w": wid, "t": time.time()})
                self.exit_codes[wid] = self.DIED_CODE
            return behave

        for stage in ("last row", "wrapup"):
            for sync in (False, True):
                with self.subTest(stage=stage, sync_start=sync):
                    self.handed.clear()
                    self.terminated.clear()
                    self.sync_start = sync
                    summary = self.play(gone(stage))
                    self.assertEqual([0, 1], sorted(wid for wid, _s in self.handed))  # nothing to replace
                    self.assertEqual([], self.terminated)
                    self.assertEqual((6, 0, 0, 0, 0, 0), (summary["runs"], summary["crashes"], summary["deadline_respawns"],
                                                          summary["died_respawns"], summary["unplayed_after_cap"], summary["wrapup_terminated"]))
                    self.assertLess(self.elapsed, 4.0)  # counted as finished: the block does not wait for any deadline

    def test_a_worker_that_dies_between_runs_gets_no_row_but_its_remaining_seeds_go_to_a_replacement(self):
        def between(played):
            def behave(wid, seeds, run_ids, q):
                if wid != 0:
                    return self.normal(wid, seeds, run_ids, q)
                self.normal(wid, seeds[:played], run_ids[:played], q, done=False)  # then nothing in flight: no start event
                self.exit_codes[wid] = 7
            return behave

        for played, rest in ((1, ["S2", "S4"]), (0, ["S0", "S2", "S4"])):  # after S0 ended; before it began anything
            with self.subTest(played=played):
                self.handed.clear()
                summary = self.play(between(played))
                self.assertEqual({0: ["S0", "S2", "S4"], 1: ["S1", "S3", "S5"], 2: rest}, dict(self.handed))
                rows = self.final_rows()
                self.assertEqual(["S0", "S1", "S2", "S3", "S4", "S5"], sorted(r["seed"] for r in rows))
                self.assertEqual({"end"}, {r["ev"] for r in rows})  # no crash row: no run was in flight
                self.assertEqual((6, 0, 1, 0), (summary["runs"], summary["crashes"], summary["died_respawns"], summary["unplayed_after_cap"]))

    def test_a_worker_that_dies_on_its_last_seed_has_a_row_and_no_replacement(self):
        summary = self.play(self.dying(after=2))  # worker 0: S0 S2 end, S4 (its last) begins and the process is gone
        self.assertEqual([0, 1], sorted(wid for wid, _s in self.handed))
        rows = {r["seed"]: r for r in self.final_rows()}
        self.assertEqual(("crash", "died"), (rows["S4"]["ev"], rows["S4"]["where"]))
        self.assertEqual([self.AS, 4], rows["S4"]["agent_seed"])
        self.assertEqual((5, 1, 0, 0), (summary["runs"], summary["crashes"], summary["died_respawns"], summary["unplayed_after_cap"]))

    def test_a_death_after_the_time_limit_has_its_row_and_no_replacement_and_nothing_counted_unplayed(self):
        summary = self.play(self.dying(after=1), stop_after_h=1e-9)  # the limit is long past by the time the loop looks
        self.assertEqual([0, 1], sorted(wid for wid, _s in self.handed))
        self.assertEqual("died", {r["seed"]: r for r in self.final_rows()}["S2"]["where"])
        self.assertEqual((0, 0), (summary["died_respawns"], summary["unplayed_after_cap"]))

    def test_deaths_and_deadline_closes_of_one_slot_share_the_one_cap(self):
        def die_then_hang(wid, seeds, run_ids, q):  # slot 0: worker 0 dies on S0, its replacement (worker 2: S2 S4) hangs on S2; slot 1 plays
            if wid == 0:
                self.dying()(wid, seeds, run_ids, q)
            elif wid == 2:
                self.hang_on_first(wid, seeds, q)
            else:
                self.normal(wid, seeds, run_ids, q)

        def hang_then_die(wid, seeds, run_ids, q):  # slot 0: worker 0 hangs on S0, its replacement (worker 2) dies on S2; slot 1 plays
            if wid == 0:
                self.hang_on_first(wid, seeds, q)
            elif wid == 2:
                self.dying(who=None)(wid, seeds, run_ids, q)
            else:
                self.normal(wid, seeds, run_ids, q)

        for name, behave, first, second in (("die, hang", die_then_hang, "died", "deadline"), ("hang, die", hang_then_die, "deadline", "died")):
            with self.subTest(name):
                self.handed.clear()
                summary = self.play(behave, cap=1)  # one replacement per slot in all: the second close hands nothing on
                self.assertEqual([0, 1, 2], sorted(wid for wid, _s in self.handed))
                rows = {r["seed"]: r for r in self.final_rows()}
                self.assertEqual(["S0", "S1", "S2", "S3", "S5"], sorted(rows))  # S4 is unplayed
                self.assertEqual((first, second), (rows["S0"]["where"], rows["S2"]["where"]))
                self.assertEqual((3, 2, 1, 1), (summary["runs"], summary["crashes"], summary["deadline_respawns"] + summary["died_respawns"],
                                               summary["unplayed_after_cap"]))

    def test_both_kinds_of_close_in_one_slot_are_counted_each_under_its_own_name_up_to_the_cap(self):
        def behave(wid, seeds, run_ids, q):  # worker 0 dies on S0; worker 2 (S2 S4) hangs on S2; worker 3 (S4) plays
            if wid == 0:
                self.dying()(wid, seeds, run_ids, q)
            elif wid == 2:
                self.hang_on_first(wid, seeds, q)
            else:
                self.normal(wid, seeds, run_ids, q)

        summary = self.play(behave, cap=2)
        self.assertEqual([0, 1, 2, 3], sorted(wid for wid, _s in self.handed))
        rows = {r["seed"]: r for r in self.final_rows()}
        self.assertEqual(["S0", "S1", "S2", "S3", "S4", "S5"], sorted(rows))
        self.assertEqual(("died", "deadline", "end"), (rows["S0"]["where"], rows["S2"]["where"], rows["S4"]["ev"]))
        self.assertEqual((4, 2, 1, 1, 0), (summary["runs"], summary["crashes"], summary["deadline_respawns"], summary["died_respawns"],
                                          summary["unplayed_after_cap"]))

    def test_a_resume_after_a_died_row_does_not_replay_that_seed(self):
        self.play(self.dying(), resume=True, cap=0)  # S0 (worker 0's first) dies; S2 and S4 are not reached (cap 0: no replacement)
        self.assertEqual({"S0": "crash", "S1": "end", "S3": "end", "S5": "end"}, {r["seed"]: r["ev"] for r in self.final_rows()})
        self.handed.clear()
        summary = self.play(resume=True)
        self.assertEqual(["S2", "S4"], sorted(x for _w, xs in self.handed for x in xs))  # the unreached seeds play, the died seed does not
        rows = self.final_rows()
        self.assertEqual(["S0", "S1", "S2", "S3", "S4", "S5"], sorted(r["seed"] for r in rows))
        (s0,) = [r for r in rows if r["seed"] == "S0"]
        self.assertEqual("died", s0["where"])
        self.assertEqual([self.AS, 4], [r for r in rows if r["seed"] == "S4"][0]["agent_seed"])
        self.assertEqual((4, 6, 1), (summary["resumed_rows"], summary["runs"] + summary["crashes"], summary["crashes"]))

    def test_close_dead_gives_the_row_of_the_run_in_flight_and_names_where_the_rest_begins(self):
        w = envproto.RunDeadline(1800.0, self.AS, {0: {"S0": 0, "S2": 2, "S4": 4}})
        w.seen({"ev": "start", "w": 0, "seed": "S0", "t": 100.0}, 100.0)
        w.seen({"ev": "end", "w": 0, "seed": "S0"}, 110.0)
        w.seen({"ev": "start", "w": 0, "seed": "S2", "t": 120.0}, 120.0)
        row, after = w.close_dead(0, 150.0, 3221226505)
        self.assertEqual(("crash", 0, "S2", "died", [self.AS, 2], 30.0), (row["ev"], row["w"], row["seed"], row["where"], row["agent_seed"], row["s"]))
        self.assertEqual("S2", after)
        self.assertFalse(w.seen({"ev": "end", "w": 0, "seed": "S2"}, 160.0))  # closed: late rows are not taken
        v = envproto.RunDeadline(1800.0, self.AS, {1: {"S1": 1, "S3": 3}})
        v.seen({"ev": "start", "w": 1, "seed": "S1", "t": 100.0}, 100.0)
        v.seen({"ev": "end", "w": 1, "seed": "S1"}, 110.0)
        self.assertEqual((None, "S1"), v.close_dead(1, 150.0, 1))  # nothing in flight: no row; the rest begins after the last row
        u = envproto.RunDeadline(1800.0, self.AS, {2: {"S2": 2}})
        self.assertEqual((None, None), u.close_dead(2, 150.0, 1))  # nothing seen: the whole share is left
        d = envproto.RunDeadline(1800.0, self.AS, {3: {"S3": 3}})
        d.seen({"ev": "done", "w": 3}, 100.0)
        self.assertIn(3, d.finished)

    def test_the_help_of_the_respawn_cap_says_a_died_worker_is_replaced_too(self):
        (act,) = [x for x in envproto.parser()._actions if x.dest == "deadline_respawn_cap"]
        self.assertIn("died", act.help)
        self.assertIn("deadline", act.help)
        self.assertEqual(3, act.default)
        self.assertEqual(0, envproto.parser().parse_args(["play", "--game", "G", "--out", "O", "--deadline-respawn-cap", "0"]).deadline_respawn_cap)


if __name__ == "__main__":
    unittest.main()
