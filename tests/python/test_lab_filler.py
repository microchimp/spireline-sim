"""D48: practice as the filler, cores not turns (tools/lab/cores.py, runner.Filler, guards.TreeCpu); D52: the filler runs from a frozen
copy of the lab tree (tools/lab/frozen.py), so a presort that moves the tree does not stop it.

Trivial `python -c` commands stand in for the self-play block and the jobs; no game, no emulator.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.lab import cores, frozen  # noqa: E402
from tools.lab import guards as gd  # noqa: E402
from tools.lab import plans as pl  # noqa: E402
from tools.lab import runner as rn  # noqa: E402
from tools.lab import winsys  # noqa: E402

# The live selfplay command's shape (LAB/jobs.json before D48): with the flags that make gen0.py emu write LAB/train/ready-<gen>.
LIVE = ["{python}", "rl/gen0.py", "emu", "--stream", "{jobname}", "--config", "a3-v1", "--procs", "16", "--block-min", "20", "--lib",
        "{tree}/out", "--agent-root", "D:/spireline/referee/agent", "--record", "--review-guard", "--lab", "D:/spireline/referee/lab",
        "--shards-root", "D:/spireline/referee/lab/runs", "--budget-runs", "2000", "--out", "{job}", "--prior-server", "auto",
        "--gpu-python", "D:/spireline/venv-gpu/Scripts/python.exe", "--server-procs", "20"]
MARKER_FLAGS = ("--lab", "--shards-root", "--budget-runs", "--review-guard", "--gen-dir", "--explore-tau", "--explore-eps")

# A stand-in block: asks for Normal priority for itself and a child (the job's limit must keep both at BelowNormal), records its
# argv and the two pids in its job folder, then sleeps until stopped (or for SECS).
BLOCK = ("import ctypes, json, os, subprocess, sys, time\n"
         "k = ctypes.windll.kernel32 if os.name == 'nt' else None\n"
         "k and k.SetPriorityClass(k.GetCurrentProcess(), 0x20)\n"
         "c = subprocess.Popen([sys.executable, '-c', 'import ctypes, os, time; k = ctypes.windll.kernel32 if os.name == \"nt\" else None; "
         "k and k.SetPriorityClass(k.GetCurrentProcess(), 0x20); time.sleep(120)'])\n"
         "out = sys.argv[sys.argv.index('--out') + 1]\n"
         "json.dump(dict(argv=sys.argv[1:], pids=[os.getpid(), c.pid], cwd=os.getcwd()), open(os.path.join(out, 'argv.json'), 'w'))\n"
         "time.sleep(SECS)\n"
         "c.kill()\n")


def block_cmd(secs: float = 120.0) -> list[str]:
    return ["{python}", "-c", BLOCK.replace("SECS", str(secs)), "emu", "--agent-root", "A", "--lab", "L", "--shards-root", "S",
            "--budget-runs", "5", "--review-guard", "--procs", "16", "--prior-server", "auto", "--server-procs", "20", "--out", "{job}"]


class CoresRulesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_the_filler_command_has_no_marker_flags_and_its_cores(self):
        got = cores.filler_cmd(LIVE + ["--gen-dir", "G", "--explore-tau", "0.1", "--explore-eps=0.02"], 6)
        for f in MARKER_FLAGS:
            self.assertFalse(any(x == f or x.startswith(f + "=") for x in got), f)
        self.assertEqual(["--procs", "6"], got[got.index("--procs"):got.index("--procs") + 2])
        self.assertEqual("6", got[got.index("--server-procs") + 1])
        self.assertEqual(1, got.count("--procs"))
        self.assertIn("--record", got)
        self.assertIn("--agent-root", got)
        self.assertEqual("{job}", got[got.index("--out") + 1])  # the shards go to its own folder under LAB/runs
        plain = cores.filler_cmd([x for x in LIVE if x != "--record"], 4)
        self.assertIn("--record", plain)
        self.assertNotIn("--server-procs", cores.filler_cmd(["py", "gen0.py", "emu", "--agent-root", "A"], 4))
        explore = cores.filler_cmd(LIVE, 4, ["--explore-tau", "0.1", "--explore-eps", "0.02"])
        self.assertEqual(["--explore-tau", "0.1", "--explore-eps", "0.02"], explore[-4:])
        with self.assertRaises(ValueError):
            cores.filler_cmd(["py", "gen0.py", "emu", "--gen-dir", "G"], 4)  # no current agent
        with self.assertRaises(ValueError):
            cores.filler_cmd(["py", "gen0.py", "train", "--agent-root", "A"], 4)

    def test_gen0_never_reaches_its_marker_writer_with_the_filler_command(self):
        sys.path.insert(0, str(ROOT / "rl"))
        try:
            import gen0  # noqa: PLC0415
        finally:
            sys.path.remove(str(ROOT / "rl"))
        argv = cores.filler_cmd(LIVE, 8)
        a = gen0.build_parser().parse_args(argv[argv.index("emu"):])
        self.assertTrue(a.record)
        self.assertFalse(bool(a.record and a.lab and a.shards_root))  # gen0.py emu calls gen_loop.mark_ready only when this holds
        self.assertIsNone(a.gen_dir)
        self.assertEqual("D:/spireline/referee/agent", a.agent_root)
        self.assertEqual((8, 8), (a.procs, a.server_procs))
        before = gen0.build_parser().parse_args(LIVE[LIVE.index("emu"):])
        self.assertTrue(before.record and before.lab and before.shards_root)  # the live command would have written one

    def test_the_cores_table(self):
        doc = {"cores": {"total": 16, "fights": 6}, "score-real": {"procs": 12, "cmd": ["x", "--procs", "12"]},
               "train": {"cmd": ["x", "--procs", "16"]}, "bench-env": {"cores": 16, "procs": 12}}
        self.assertEqual(16, cores.job_cores("nightly-real", {}, doc))
        self.assertEqual(12, cores.job_cores("score-real", {}, doc))
        self.assertEqual(16, cores.job_cores("train", {}, doc))
        self.assertEqual(16, cores.job_cores("bench-env", {"procs": 12}, doc))  # its own figure wins over procs
        self.assertEqual(6, cores.job_cores("fights", {}, doc))
        self.assertEqual(4, cores.job_cores("presort", {}, doc))  # its codex shards (D52): the filler runs beside it
        self.assertEqual(16, cores.job_cores("presort", {}, {"cores": {"total": 16, "presort": 16}}))  # the D48 table
        self.assertEqual(4, cores.job_cores("probes", {}, doc))
        self.assertEqual(16, cores.job_cores("score-real", {}, {"cores": {"total": 16}, "score-real": {"cores": 40}}))
        self.assertEqual(4, cores.filler_size(16, 12, 4))
        self.assertEqual(0, cores.filler_size(16, 13, 4))
        self.assertEqual(16, cores.filler_size(16, 0, 4))
        self.assertEqual(os.cpu_count() or 16, cores.total_cores({}))

    def touch(self, rel: str, age_s: float = 0.0) -> Path:
        p = self.tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
        t = time.time() - age_s
        os.utime(p, (t, t))
        return p

    def test_the_night_holds_the_filler(self):
        doc = {"nightly-real": {"enabled": True, "every_h": 20}}
        lab = self.tmp
        self.touch("nightly/last-ok", age_s=20 * 3600 - 2 * 3600)  # due in 2 h
        self.assertIsNone(cores.night_hold(lab, doc))
        self.touch("nightly/last-ok", age_s=20 * 3600 - 20 * 60)  # due in 20 min
        self.assertIn("no start within 30 min", cores.night_hold(lab, doc))
        self.assertIsNone(cores.night_hold(lab, doc, starting=False))  # a running block goes on to the due time
        self.assertIsNotNone(cores.night_hold(lab, {**doc, "selfplay": {"night_margin_min": 10}}, now=time.time() + 15 * 60))
        self.touch("nightly/last-ok", age_s=20 * 3600 - 60)  # due in a minute: a running block is stopped before the night
        self.assertIn("due at", cores.night_hold(lab, doc, starting=False))
        self.touch("nightly/last-ok", age_s=20 * 3600 + 60)  # due a minute ago
        self.assertIn("due at", cores.night_hold(lab, doc, starting=False))
        self.touch("nightly/last-ok", age_s=3600)
        self.assertIsNone(cores.night_hold(lab, doc))
        self.touch("nightly/active")
        self.assertIn("nightly/active", cores.night_hold(lab, doc))
        self.assertIn("nightly/active", cores.night_hold(lab, doc, starting=False))
        (lab / "nightly" / "active").unlink()
        self.touch("nightly/last-ok", age_s=21 * 3600)  # due, but the last attempt was skipped 50 min ago: retry in 10 min
        self.touch("nightly/last-skip", age_s=50 * 60)
        self.assertIn("no start", cores.night_hold(lab, doc))
        self.assertIsNone(cores.night_hold(lab, doc, starting=False))
        self.assertIsNone(cores.night_hold(lab, {"nightly-real": {"enabled": False}}))

    def test_quiet_lock_lines_the_filler_yields_to(self):
        q = self.tmp / "QUIET.lock"
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 3 * 3600))
        self.assertIsNone(cores.quiet_blocker(q, {}))
        q.write_text(f"gym-loop {now} pid=1\nlab-score {now} pid=2\n")
        self.assertIsNone(cores.quiet_blocker(q, {}))
        q.write_text(f"gym-loop {now} pid=1\nlab-nightly {now} pid=2\n")
        self.assertEqual("lab-nightly", cores.quiet_blocker(q, {}))
        q.write_text(f"reset-x {old} pid=3\n")
        self.assertIsNone(cores.quiet_blocker(q, {}))  # stale
        q.write_text("w3-blocks 2026-11-02T10:00Z pid=3\n")  # the loop's minute form of the time
        t = cores._line_time("2026-11-02T10:00:00Z")
        self.assertEqual("w3-blocks", cores.quiet_blocker(q, {}, now=t + 600))
        self.assertIsNone(cores.quiet_blocker(q, {}, now=t + 3 * 3600))
        self.assertIsNone(cores.quiet_blocker(q, {"selfplay": {"quiet_ignore": ["w3-blocks"]}}, now=t + 600))

    def test_the_loops_status(self):
        s = self.tmp / "status"
        self.assertEqual((None, False), cores.loop_state(s))
        s.write_text(json.dumps({"pid": 42, "phase": "generate"}))
        self.assertEqual((42, True), cores.loop_state(s))
        self.assertEqual((None, False), cores.loop_state(s, alive=lambda pid: False))
        s.write_text(json.dumps({"pid": 42, "phase": "waiting", "why": "nightly-active"}))
        self.assertEqual((42, False), cores.loop_state(s))


class LoadRuleTest(unittest.TestCase):
    """The pause rule counts only load that is neither the runner's (the filler is in its Job Object) nor the loop's."""

    def machine(self, other: float, filler: float, loop: float):
        clock, state = [0.0], {"idle": 0, "kernel": 0, "ours": 0, "loop": 0}

        def tick(dt):
            total = dt * 10_000_000 * 16
            state["idle"] += int(total * (1 - other - filler - loop))
            state["kernel"] += int(total)
            state["ours"] += int(total * filler)
            state["loop"] += int(total * loop)
            clock[0] += dt

        tree = gd.TreeCpu(lambda: (7, None), times=lambda pid: {(7, 0): state["loop"]})
        return clock, state, tick, tree

    def run_monitor(self, ours, clock, tick, state):
        mon = gd.LoadMonitor(ours, window=120.0, interval=5.0, clock=lambda: clock[0],
                             times=lambda: (state["idle"], state["kernel"], 0))
        mon.sample()
        for _ in range(30):
            tick(5.0)
            mon.sample()
        return mon

    def test_filler_and_loop_load_do_not_pause_a_job(self):
        clock, state, tick, tree = self.machine(other=0.3, filler=0.3, loop=0.35)
        mon = self.run_monitor(lambda: state["ours"] + tree(), clock, tick, state)
        self.assertFalse(mon.sustained(0.5))
        self.assertAlmostEqual(0.3, mon.mean()[0], places=2)

    def test_without_the_loop_view_the_same_machine_would_pause(self):
        clock, state, tick, _ = self.machine(other=0.3, filler=0.3, loop=0.35)
        mon = self.run_monitor(lambda: state["ours"], clock, tick, state)
        self.assertTrue(mon.sustained(0.5))

    def test_real_other_load_still_pauses(self):
        clock, state, tick, tree = self.machine(other=0.6, filler=0.3, loop=0.05)
        mon = self.run_monitor(lambda: state["ours"] + tree(), clock, tick, state)
        self.assertTrue(mon.sustained(0.5))

    def test_tree_cpu_counts_from_its_first_look_and_refuses_a_reused_pid(self):
        procs = {(7, 0): 1000}
        who = [(7, None)]
        t = gd.TreeCpu(lambda: who[0], times=lambda pid: dict(procs) if pid == 7 else {})
        self.assertEqual(0, t())  # what the loop used before the runner looked is not counted
        procs[(7, 0)] = 1500
        procs[(9, 5)] = 300  # a worker the loop started since
        self.assertEqual(800, t())
        del procs[(9, 5)]  # it ended
        procs[(7, 0)] = 1600
        self.assertEqual(900, t())
        reused = {(7, gd.FILETIME_EPOCH + int(2_000_000_000 * 1e7)): 5000}
        t2 = gd.TreeCpu(lambda: (7, 1_000_000_000.0), times=lambda pid: dict(reused))
        t2()
        reused[next(iter(reused))] = 9000
        self.assertEqual(0, t2())  # created after the loop's status was written: not the loop

    def test_the_runner_leaves_the_loop_out_of_other_load(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        a = rn.parse_args(["--lab", str(tmp / "lab"), "--tree", str(tmp / "tree"), "--loop-status", str(tmp / "status")])
        r = rn.Runner(a)
        r.loop_cpu = lambda: 12345  # what the monitor's "ours" adds to the runner's own Job Object
        self.assertEqual(12345, r.load.ours())


@unittest.skipUnless(winsys.IS_WIN, "Job Objects and priority classes are Windows")
class FillerBase(unittest.TestCase):
    FILLER_TREE = "off"  # --filler-tree: off runs the block from the lab tree (D48), auto from a frozen copy (D52)

    def setUp(self):
        self.sha = "abc"  # the presorted build the runner's state names
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        saved = dict(pl.PLANS)
        self.addCleanup(lambda: (pl.PLANS.clear(), pl.PLANS.update(saved)))
        tick = rn.FILLER_TICK_S
        self.addCleanup(lambda: setattr(rn, "FILLER_TICK_S", tick))
        rn.FILLER_TICK_S = 0.0
        self.lab = self.tmp / "lab"
        (self.lab / "runs").mkdir(parents=True)
        self.jobs({})
        last_ok = self.lab / "nightly" / "last-ok"
        last_ok.parent.mkdir(parents=True)
        last_ok.write_text("x\n")  # the night ran just now: due in 20 h

    def jobs(self, extra: dict, cmd: list[str] | None = None) -> None:
        doc = {"cores": {"total": 16, "fights": 8}, "nightly-real": {"enabled": True, "every_h": 20},
               "selfplay": {"enabled": True, "filler": True, "floor": 4, "cmd": cmd or block_cmd(), "cwd": "{lab}", "needs": [], "timeout_min": 5}}
        for k, v in extra.items():
            doc[k] = {**doc.get(k, {}), **v} if isinstance(v, dict) else v
        (self.lab / "jobs.json").write_text(json.dumps(doc), encoding="utf-8")

    def runner(self, *extra: str) -> rn.Runner:
        a = rn.parse_args(["--lab", str(self.lab), "--tree", str(self.tmp / "tree"), "--game", str(self.tmp / "game" / "g"),
                           "--backlog-dir", str(self.tmp / "bl"), "--poll", "0.05", "--wait-poll", "0.05", "--grace", "0",
                           "--no-guard", "user,load,disk", "--loop-status", str(self.tmp / "no-loop"), "--filler", "on",
                           "--filler-tree", self.FILLER_TREE, *extra])
        r = rn.Runner(a)
        r.state["tree_sha"] = r.state["last_presorted_sha"] = self.sha
        self.addCleanup(lambda: r.stop_filler("test end", retry=False))
        return r

    def py(self, name: str, code: str, **kw) -> pl.Cmd:
        return pl.Cmd(name, ["{py}", "-c", code], cwd="{lab}", **kw)

    def started(self, r: rn.Runner, seconds: float = 30.0) -> dict:
        r.service_filler()
        self.assertIsNotNone(r.filler, self.log(r))
        p = r.filler.run.dir / "argv.json"
        end = time.time() + seconds
        while time.time() < end:
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except ValueError:
                    pass
            time.sleep(0.05)
        self.fail("the stand-in block did not start")

    def log(self, r: rn.Runner) -> str:
        p = r.lab.root / "runner.log"
        return p.read_text(encoding="utf-8") if p.exists() else ""


@unittest.skipUnless(winsys.IS_WIN, "Job Objects and priority classes are Windows")
class FillerRunnerTest(FillerBase):
    """D48's rules with the block running from the lab tree itself (--filler-tree off): a presort stops it."""

    def test_the_filler_runs_without_marker_flags_at_below_normal_and_takes_no_lock(self):
        r = self.runner()
        got = self.started(r)
        for f in MARKER_FLAGS:
            self.assertNotIn(f, got["argv"])
        self.assertEqual("16", got["argv"][got["argv"].index("--procs") + 1])  # the whole machine: no job, no loop
        self.assertEqual("16", got["argv"][got["argv"].index("--server-procs") + 1])
        self.assertIn("--record", got["argv"])
        for pid in got["pids"]:  # the block and its child both asked for Normal
            self.assertEqual(winsys.BELOW_NORMAL_PRIORITY_CLASS, winsys.priority_class_of(pid))
        self.assertFalse((self.tmp / "QUIET.lock").exists())
        self.assertFalse((self.tmp / "bl" / "main.lock").exists())
        self.assertFalse((self.lab / "train").exists())
        run = r.filler.run
        r.stop_filler("test", retry=False)
        self.assertIsNone(r.filler)
        end = time.time() + 20
        while any(winsys.parent_alive(pid) for pid in got["pids"]) and time.time() < end:
            time.sleep(0.05)
        self.assertFalse(any(winsys.parent_alive(pid) for pid in got["pids"]))  # gone with the Job Object
        self.assertEqual("0", (run.dir / "done.rc").read_text().strip())
        self.assertIn("filler block stopped", json.loads((run.dir / "job.json").read_text(encoding="utf-8"))["note"])

    def test_a_block_that_ends_is_recorded_and_the_next_one_starts(self):
        self.jobs({}, cmd=block_cmd(0.2))
        r = self.runner()
        self.started(r)
        first = r.filler.run
        end = time.time() + 30
        while r.filler is not None and r.filler.run is first and time.time() < end:
            r.service_filler()
            time.sleep(0.05)
        self.assertEqual("0", (first.dir / "done.rc").read_text().strip())
        self.assertIn("filler: " + first.name + " finished rc=0", self.log(r))
        r.service_filler()
        self.assertIsNotNone(r.filler)
        self.assertIsNot(first, r.filler.run)

    def test_a_job_starts_beside_the_filler_when_its_cores_are_free(self):
        r = self.runner()
        r.main_cores = 8  # the filler starts beside an 8-core job ...
        self.started(r)
        self.assertEqual(8, r.filler.cores)
        r.main_cores = 0
        block = r.filler.running.proc
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [self.py("one", "import time; time.sleep(0.5)", core=True)], mode="serial")]
        run = r.new_run("fights", "abc")  # ... and the next 8-core job runs beside it
        self.assertIsNone(r.execute(run))
        self.assertEqual("0", (run.dir / "done.rc").read_text().strip())
        self.assertIsNotNone(r.filler)
        self.assertIs(block, r.filler.running.proc)
        self.assertIsNone(block.poll())
        self.assertNotIn("needs its cores", self.log(r))

    def test_the_filler_shrinks_when_a_job_needs_its_cores(self):
        self.jobs({"cores": {"total": 16, "fights": 12}})
        r = self.runner()
        self.started(r)
        self.assertEqual(16, r.filler.cores)
        block = r.filler.running.proc
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [self.py("one", "import time; time.sleep(1.5)", core=True)], mode="serial")]
        run = r.new_run("fights", "abc")
        self.assertIsNone(r.execute(run))
        self.assertIsNotNone(block.poll())  # stopped before the job ran
        self.assertIn("a job needs its cores (16 held, 4 left)", self.log(r))
        self.assertIsNotNone(r.filler)  # started again beside the job at what was left
        self.assertEqual(4, r.filler.cores)
        self.assertIn("at 4 cores beside fights", self.log(r))

    def test_no_filler_under_the_floor_or_beside_a_whole_machine_job(self):
        self.jobs({"cores": {"total": 16, "fights": 13}})
        r = self.runner()
        r.main_cores = 13
        r.service_filler()
        self.assertIsNone(r.filler)
        self.assertIn("filler: not started (under 4 cores free)", self.log(r))
        s = self.tmp / "loop-status"
        s.write_text(json.dumps({"pid": os.getpid(), "phase": "generate"}))
        r.a.loop_status = str(s)
        r.main_cores = 8
        self.assertEqual(16 - 8 - 6, 16 - r.busy_cores(json.loads((self.lab / "jobs.json").read_text())))
        r.service_filler()
        self.assertIsNone(r.filler)  # 2 left beside the stepping loop

    def test_the_night_stops_the_filler_and_holds_it(self):
        r = self.runner()
        self.started(r)
        block = r.filler.running.proc
        (self.lab / "nightly" / "active").write_text("1\n")
        r.service_filler()
        self.assertIsNone(r.filler)
        self.assertIsNotNone(block.poll())
        self.assertIn("LAB/nightly/active", self.log(r))
        r.filler_next = 0.0
        r.service_filler()
        self.assertIsNone(r.filler)
        self.assertIn("filler: not started (a nightly block is running", self.log(r))
        (self.lab / "nightly" / "active").unlink()
        t = time.time() - 20 * 3600 + 20 * 60  # the night is due in 20 minutes
        os.utime(self.lab / "nightly" / "last-ok", (t, t))
        r.service_filler()
        self.assertIsNone(r.filler)
        self.assertIn("no start within 30 min", self.log(r))

    def test_the_nightly_job_stops_the_filler_before_it_starts(self):
        r = self.runner()
        self.started(r)
        block = r.filler.running.proc
        seen = []
        pl.PLANS["nightly-real"] = lambda c: [pl.Step("n", [self.py("n", "pass", core=True)], mode="serial")]
        orig = r._execute
        r._execute = lambda run: (seen.append((r.filler, block.poll())), orig(run))[1]
        r.execute(r.new_run("nightly-real", "abc"))
        self.assertIsNone(seen[0][0])
        self.assertIsNotNone(seen[0][1])
        self.assertIn("the night starts", self.log(r))

    def test_a_presort_stops_the_filler(self):
        r = self.runner()
        self.started(r)
        pl.PLANS["presort"] = lambda c: [pl.Step("p", [self.py("p", "import time; time.sleep(0.3)", core=True)], mode="serial")]
        r.execute(r.new_run("presort", "abc"))
        self.assertIn("a presort moves the tree", self.log(r))

    def test_a_guard_or_a_quiet_line_stops_the_filler(self):
        r = self.runner()
        self.started(r)
        (self.lab / "PAUSE").write_text("now\n")
        r.poll()
        self.assertIsNone(r.filler)
        (self.lab / "PAUSE").unlink()
        r.poll()
        r.filler_next = 0.0
        q = self.tmp / "QUIET.lock"
        q.write_text(f"reset-x {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} pid=1\n")
        r.service_filler()
        self.assertIsNone(r.filler)
        self.assertIn("QUIET.lock is held by reset-x", self.log(r))
        q.write_text(f"gym-loop {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} pid=1\n")
        self.started(r)  # the loop's own line does not hold it
        q.write_text(f"lab-nightly {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} pid=1\n")
        r.service_filler()
        self.assertIsNone(r.filler)

    def test_the_ladder_never_picks_selfplay_while_it_is_the_filler(self):
        r = self.runner()
        self.assertTrue(r.filler_owns_selfplay())
        self.assertEqual("", r.pick_other("abc")[0])
        old = self.runner("--filler", "off")
        self.assertEqual("selfplay", old.pick_other("abc")[0])  # the old way: the bottom of the ladder

    def test_a_filler_block_left_by_a_stopped_runner_is_closed_not_resumed(self):
        r = self.runner()
        self.started(r)
        left = r.filler.run
        r.filler.sj.terminate()
        r.filler = None  # as if the runner had died
        r2 = self.runner()
        job, run, _ = r2.choose("abc")
        self.assertIsNot(run, left)
        self.assertEqual("3", (left.dir / "done.rc").read_text().strip())

    def test_the_filler_is_off_without_always_on_or_with_the_switch_off(self):
        a = rn.parse_args(["--lab", str(self.lab), "--tree", str(self.tmp / "tree")])
        self.assertFalse(rn.Runner(a).filler_mode)
        self.assertTrue(rn.Runner(rn.parse_args(["--lab", str(self.lab), "--tree", str(self.tmp / "t"), "--always-on"])).filler_mode)
        self.assertFalse(rn.Runner(rn.parse_args(["--lab", str(self.lab), "--tree", str(self.tmp / "t"), "--always-on",
                                                  "--once", "fights"])).filler_mode)
        self.jobs({"selfplay": {"filler": False}})
        r = self.runner()
        r.service_filler()
        self.assertIsNone(r.filler)

    def test_the_jobs_account_leaves_the_filler_out(self):
        class Fake:
            def __init__(self, cpu):
                self.cpu = cpu

            def cpu_time(self):
                return int(self.cpu * 1e7)

        r = self.runner()
        r.root_job = Fake(100.0)
        r.report_cpu_done, r.filler_cpu_done = 10.0, 30.0
        self.assertAlmostEqual(60.0, r.cpu_secs())


# ---------------------------------------------------------------- D52: the frozen copy

def put(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def git(tree: Path, *args: str) -> str:
    r = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "-C", str(tree), *args],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def commit_tree(tree: Path, tag: str) -> str:
    """A stand-in lab tree at `tag`, committed in a git repo at `tree` (its sha comes back): tracked rl/, tools/, data/, src/ and a
    test fixture; the published out/ and the build folders are untracked, as in the real tree."""
    if not (tree / ".git").exists():
        tree.mkdir(parents=True, exist_ok=True)
        git(tree, "init", "-q")
        put(tree / ".gitignore", "out/\nbin/\nobj/\n__pycache__/\n")
    for rel in ("rl/gen0.py", "rl/gen_loop.py", "tools/lab/x.py", "src/Sts2Emulator/Core.cs"):
        put(tree / rel, f"# {tag}\n")
    put(tree / "data" / "id_map.json", "{}\n")
    put(tree / "tests" / "fixtures" / "big.json", "{}\n")
    for rel in ("out/Sts2Emulator.dll", "src/Sts2Emulator/bin/Debug/a.dll", "src/Sts2Emulator/obj/b.o", "rl/__pycache__/gen0.pyc"):
        put(tree / rel, tag)
    git(tree, "add", "-A")
    git(tree, "commit", "-q", "-m", tag)
    return git(tree, "rev-parse", "HEAD")


class FrozenCopyTest(unittest.TestCase):
    """tools/lab/frozen.py: one whole copy of the lab tree per presorted build."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.tree, self.root = self.tmp / "tree", self.tmp / "filler-tree"
        self.sha = commit_tree(self.tree, "v1")

    def test_the_copy_holds_the_sources_and_the_build_and_not_the_rest(self):
        d = frozen.make(self.tree, self.root, self.sha, self.sha)
        self.assertEqual(self.root / self.sha[:12], d)
        for rel in ("rl/gen0.py", "rl/gen_loop.py", "out/Sts2Emulator.dll", "tools/lab/x.py", "data/id_map.json", "src/Sts2Emulator/Core.cs"):
            self.assertTrue((d / rel).is_file(), rel)
        for rel in (".git", "tests", "src/Sts2Emulator/bin", "src/Sts2Emulator/obj", "rl/__pycache__"):
            self.assertFalse((d / rel).exists(), rel)
        self.assertEqual(self.sha, (d / frozen.MARK).read_text(encoding="utf-8").strip())
        self.assertEqual(d, frozen.ready(self.root, self.sha))
        self.assertEqual([self.sha[:12]], [x.name for x in self.root.iterdir()])  # no .part left behind
        self.assertEqual("# v1\n", (self.tree / "rl" / "gen0.py").read_text(encoding="utf-8"))  # the tree is only read

    def test_a_copy_is_not_changed_when_the_tree_moves(self):
        d = frozen.make(self.tree, self.root, self.sha, self.sha)
        (self.tree / "rl" / "gen0.py").write_text("# moved\n", encoding="utf-8")
        (self.tree / "out" / "Sts2Emulator.dll").unlink()
        self.assertEqual("# v1\n", (d / "rl" / "gen0.py").read_text(encoding="utf-8"))
        self.assertEqual("v1", (d / "out" / "Sts2Emulator.dll").read_text(encoding="utf-8"))
        self.assertEqual(d, frozen.ready(self.root, self.sha))

    def test_a_tree_that_is_not_at_the_build_or_has_no_published_build_is_refused(self):
        with self.assertRaises(frozen.FreezeError) as e:
            frozen.make(self.tree, self.root, self.sha, "f" * 40)
        self.assertIn("not at the presorted build", str(e.exception))
        with self.assertRaises(frozen.FreezeError):
            frozen.make(self.tree, self.root, self.sha, "")
        (self.tree / "out" / "Sts2Emulator.dll").unlink()
        with self.assertRaises(frozen.FreezeError) as e:
            frozen.make(self.tree, self.root, self.sha, self.sha)
        self.assertIn("Sts2Emulator.dll", str(e.exception))
        self.assertFalse(self.root.exists() and any(self.root.iterdir()))

    def test_only_a_whole_copy_of_that_sha_is_ready(self):
        d = self.root / self.sha[:12]
        put(d / "rl" / "gen0.py", "x")
        self.assertIsNone(frozen.ready(self.root, self.sha))  # no marker: a copy that never finished
        put(d / frozen.MARK, "0" * 40)
        self.assertIsNone(frozen.ready(self.root, self.sha))  # another sha
        put(d / frozen.MARK, self.sha)
        self.assertIsNone(frozen.ready(self.root, self.sha))  # a file a block starts from is missing
        got = frozen.make(self.tree, self.root, self.sha, self.sha)  # the half-made folder is replaced, not trusted
        self.assertEqual(got, frozen.ready(self.root, self.sha))
        self.assertTrue((got / "out" / "Sts2Emulator.dll").is_file())

    def test_a_copy_that_fails_leaves_no_folder(self):
        import unittest.mock as mock  # noqa: PLC0415

        with mock.patch.object(frozen.shutil, "copytree", side_effect=OSError("disk full")), self.assertRaises(OSError):
            frozen.make(self.tree, self.root, self.sha, self.sha)
        self.assertEqual([], list(self.root.iterdir()))
        self.assertIsNone(frozen.ready(self.root, self.sha))

    def test_clean_keeps_the_copy_in_use_and_drops_the_rest(self):
        keep, other = self.root / "aaaaaaaaaaaa", self.root / "bbbbbbbbbbbb"
        fresh, old = self.root / "cccccccccccc.part", self.root / "dddddddddddd.part"
        for d in (keep, other, fresh, old):
            put(d / "f", "x")
        t = time.time() - 2 * 3600
        os.utime(old, (t, t))
        gone = frozen.clean(self.root, [keep])
        self.assertEqual(["bbbbbbbbbbbb", "dddddddddddd.part"], sorted(gone))
        self.assertTrue(keep.exists() and fresh.exists())
        self.assertEqual([], frozen.clean(self.tmp / "nowhere"))


class FrozenFillerTest(FillerBase):
    """D52: the filler runs from a frozen copy of the lab tree, so a presort that moves the tree does not stop it and cannot reach it."""

    FILLER_TREE = "auto"

    def setUp(self):
        super().setUp()
        self.tree = self.tmp / "tree"
        self.sha = commit_tree(self.tree, "v1")
        self.jobs({"selfplay": {"cwd": "{tree}", "needs": ["{tree}/rl/gen0.py", "{tree}/out/Sts2Emulator.dll"]}})

    def copy_of(self, sha: str | None = None) -> Path:
        return self.tmp / "filler-tree" / (sha or self.sha)[:12]

    def test_the_block_runs_from_the_frozen_copy_not_from_the_tree(self):
        r = self.runner()
        got = self.started(r)
        self.assertEqual(self.copy_of().resolve(), Path(got["cwd"]).resolve())
        self.assertNotEqual(self.tree.resolve(), Path(got["cwd"]).resolve())
        self.assertIn(f"filler: froze the lab tree at {self.sha[:7]}", self.log(r))
        self.assertIn(f"(frozen copy of {self.sha[:7]})", self.log(r))
        self.assertEqual(self.sha, r.filler.sha)
        self.assertEqual(self.copy_of(), r.filler.ctx.tree)

    def test_a_presort_that_moves_the_tree_neither_stops_the_block_nor_reaches_it(self):
        r = self.runner()
        self.started(r)
        before = r.filler.run.name
        moves = ("import pathlib, time; t = pathlib.Path(%r); (t / 'rl' / 'gen0.py').write_text('moved'); "
                 "(t / 'out' / 'Sts2Emulator.dll').unlink(); time.sleep(0.8); (t / 'out' / 'Sts2Emulator.dll').write_text('rebuilt')" % str(self.tree))
        pl.PLANS["presort"] = lambda c: [pl.Step("p", [self.py("p", moves, core=True)], mode="serial")]
        self.assertIsNone(r.execute(r.new_run("presort", self.sha)))
        self.assertEqual("moved", (self.tree / "rl" / "gen0.py").read_text(encoding="utf-8"))  # the presort did move the tree
        self.assertNotIn("a presort moves the tree", self.log(r))
        self.assertIsNotNone(r.filler)  # the block shrank to the 12 cores the presort leaves (4 for its codex shards) and plays on
        self.assertEqual(12, r.filler.cores)
        self.assertIn("a job needs its cores (16 held, 12 left)", self.log(r))
        self.assertIn("at 12 cores beside presort (frozen copy of", self.log(r))
        self.assertEqual(self.copy_of(), r.filler.ctx.tree)
        self.assertEqual("# v1\n", (self.copy_of() / "rl" / "gen0.py").read_text(encoding="utf-8"))
        self.assertEqual("v1", (self.copy_of() / "out" / "Sts2Emulator.dll").read_text(encoding="utf-8"))
        self.assertNotEqual(before, r.filler.run.name)

    def test_the_first_start_waits_for_a_presort_that_is_moving_the_tree(self):
        r = self.runner()
        r.running_job = "presort"
        r.service_filler()
        self.assertIsNone(r.filler)
        self.assertIn("filler: not started (no frozen copy of", self.log(r))
        self.assertFalse(self.copy_of().exists())  # nothing is copied from a tree in motion
        r.running_job = None
        got = self.started(r)
        self.assertEqual(self.copy_of().resolve(), Path(got["cwd"]).resolve())

    def test_a_presort_left_unfinished_by_a_restart_holds_the_first_copy(self):
        d = self.lab / "runs" / "20261104T0000-presort"
        put(d / "job.json", "{}")
        r = self.runner()
        r.service_filler()
        self.assertIsNone(r.filler)
        self.assertIn("a presort is moving the tree", self.log(r))
        self.assertFalse(self.copy_of().exists())
        put(d / "done.rc", "0")
        self.started(r)

    def test_a_tree_that_is_not_at_the_presorted_build_is_not_copied(self):
        r = self.runner()
        r.state["tree_sha"] = r.state["last_presorted_sha"] = "f" * 40  # the tree is at self.sha: a presort has moved it since
        r.service_filler()
        self.assertIsNone(r.filler)
        self.assertIn("not at the presorted build", self.log(r))
        self.assertFalse(self.copy_of("f" * 40).exists())

    def test_a_newer_presorted_build_replaces_the_block_and_the_old_copy(self):
        r = self.runner()
        self.started(r)
        old_copy, block = self.copy_of(), r.filler.running.proc
        sha2 = commit_tree(self.tree, "v2")  # the presort moved the tree to the new build and finished
        r.state["tree_sha"] = r.state["last_presorted_sha"] = sha2
        r.service_filler(start=False)  # between jobs: the old block stops (at once, no wait for the next)
        self.assertIsNone(r.filler)
        self.assertIsNotNone(block.poll())
        self.assertIn(f"the presort of {sha2[:7]} finished; this block plays {self.sha[:7]}", self.log(r))
        self.assertLessEqual(r.filler_next, r.clock())
        got = self.started(r)
        self.assertEqual(self.copy_of(sha2).resolve(), Path(got["cwd"]).resolve())
        self.assertFalse(old_copy.exists())
        self.assertIn("removed the frozen copies of " + self.sha[:12], self.log(r))
        self.assertEqual("# v2\n", (self.copy_of(sha2) / "rl" / "gen0.py").read_text(encoding="utf-8"))

    def test_a_failed_presort_leaves_the_block_on_its_copy_and_starts_nothing_new(self):
        r = self.runner()
        self.started(r)
        block = r.filler.running.proc
        r.state["tree_sha"] = None  # finish_job's mark for a presort that failed: the tree is not a build
        r.service_filler(start=False)
        self.assertIsNotNone(r.filler)
        self.assertIsNone(block.poll())
        r.stop_filler("test", retry=False)
        r.service_filler()
        self.assertIsNone(r.filler)
        self.assertIn("the tree is not a presorted build", self.log(r))

    def test_the_night_still_stops_it_and_the_copy_stays_for_the_morning(self):
        r = self.runner()
        self.started(r)
        pl.PLANS["nightly-real"] = lambda c: [pl.Step("n", [self.py("n", "pass", core=True)], mode="serial")]
        r.execute(r.new_run("nightly-real", self.sha))
        self.assertIn("the night starts", self.log(r))
        self.assertTrue(frozen.ready(self.tmp / "filler-tree", self.sha))

    def test_off_runs_from_the_lab_tree_as_before(self):
        r = self.runner("--filler-tree", "off")
        got = self.started(r)
        self.assertEqual(self.tree.resolve(), Path(got["cwd"]).resolve())
        self.assertFalse((self.tmp / "filler-tree").exists())
        self.assertEqual("", r.filler.sha)


if __name__ == "__main__":
    unittest.main()
