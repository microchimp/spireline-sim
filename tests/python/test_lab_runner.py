"""tools/lab: the fix lab's runner, its guards, its disk pruning and its job plans, with no game and no PC state.

The guards are driven with fake clocks and probes; the runner runs trivial `python -c` commands; the Job Object
test starts a real process tree and kills its parent from outside (Windows only).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.lab import guards as gd  # noqa: E402
from tools.lab import plans as pl  # noqa: E402
from tools.lab import prune as pr  # noqa: E402
from tools.lab import runner as rn  # noqa: E402
from tools.lab import winsys  # noqa: E402

WIN = os.name == "nt"


def fake_clock():
    t = [0.0]

    def clock() -> float:
        return t[0]

    def advance(dt: float) -> None:
        t[0] += dt
    return clock, advance


class GuardsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.lab = self.tmp / "lab"
        self.lab.mkdir()
        self.lock = self.tmp / "main.lock"

    def guards(self, **kw):
        kw.setdefault("idle_fn", lambda: 1e6)
        kw.setdefault("free_fn", lambda p: 10**13)
        return gd.Guards(self.lab, self.lock, **kw)

    def test_ok_when_nothing_is_wrong(self):
        self.assertTrue(self.guards().poll().ok)

    def test_pause_file_gives_grace_and_now_stops_at_once(self):
        g = self.guards(grace=77)
        (self.lab / "PAUSE").write_text("", encoding="utf-8")
        v = g.poll()
        self.assertEqual((v.kind, v.grace, v.leave), ("pause", 77, False))
        (self.lab / "PAUSE").write_text("now\n", encoding="utf-8")
        self.assertEqual(g.poll().grace, 0)
        (self.lab / "PAUSE").unlink()
        self.assertTrue(g.poll().ok)

    def test_main_lock_pauses(self):
        g = self.guards()
        self.lock.write_text("", encoding="utf-8")
        self.assertEqual(g.poll().kind, "lock")

    def test_user_input_leaves_at_once_and_unknown_idle_does_not(self):
        idle = [5.0]
        g = self.guards(idle_fn=lambda: idle[0])
        v = g.poll()
        self.assertEqual((v.kind, v.leave, v.grace), ("user", True, 0))
        idle[0] = 61.0
        self.assertTrue(g.poll().ok)
        idle[0] = None
        self.assertTrue(g.poll().ok)

    def test_halt_file_leaves(self):
        (self.lab / "HALT").write_text("network seen\n", encoding="utf-8")
        v = self.guards().poll()
        self.assertEqual((v.kind, v.leave, v.detail), ("halt", True, "network seen"))

    def test_disabled_guard_is_off(self):
        (self.lab / "PAUSE").write_text("", encoding="utf-8")
        self.assertTrue(self.guards(disabled={"pause"}).poll().ok)

    @unittest.skipUnless(WIN, "parent liveness is a Windows check")
    def test_exit_with_parent_when_the_parent_is_gone(self):
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        self.assertEqual(self.guards(parent_pid=p.pid).poll().kind, "parent")
        self.assertTrue(self.guards(parent_pid=os.getpid()).poll().ok)
        self.assertTrue(self.guards().poll().ok)

    def test_trial_deadline(self):
        clock, advance = fake_clock()
        g = self.guards(deadline=100.0, clock=clock, grace=500)
        self.assertTrue(g.poll().ok)
        advance(100)
        v = g.poll()
        self.assertEqual((v.kind, v.leave, v.grace), ("trial", True, 60.0))  # grace is capped at a minute

    def test_disk_prunes_then_passes_or_blocks(self):
        calls = []
        free = [10 * gd.GB]

        def prune(need):
            calls.append(need)
            free[0] += need
            return need

        g = self.guards(free_fn=lambda p: free[0], free_gb=50, prune=prune)
        self.assertTrue(g.poll().ok)  # pruned back over the minimum
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(calls[0], 40 * gd.GB)
        clock, advance = fake_clock()
        g2 = self.guards(free_fn=lambda p: 1 * gd.GB, free_gb=50, prune=lambda need: 0, clock=clock)
        v = g2.poll()
        self.assertEqual(v.kind, "disk")
        self.assertFalse(v.leave)

    def test_lab_size_limit(self):
        (self.lab / "big.bin").write_bytes(b"x" * 4096)
        g = self.guards(lab_gb=1e-6, prune=lambda need: 0)  # 1e-6 GB is about 1 KB
        self.assertEqual(g.poll().kind, "disk")


class LoadMonitorTest(unittest.TestCase):
    def monitor(self, other: float, ours: float, window=120.0):
        """A machine of 4 CPUs where `other` and `ours` (shares of the whole machine) are busy."""
        clock, advance = fake_clock()
        state = {"idle": 0, "kernel": 0, "user": 0, "ours": 0}

        def tick(dt):
            total = dt * 10_000_000 * 4
            busy = total * (other + ours)
            state["idle"] += int(total - busy)
            state["kernel"] += int(total)  # kernel includes idle
            state["user"] += 0
            state["ours"] += int(total * ours)
            advance(dt)

        mon = gd.LoadMonitor(lambda: state["ours"], window=window, interval=5.0, clock=clock,
                             times=lambda: (state["idle"], state["kernel"], state["user"]))
        mon.sample()
        return mon, tick

    def run_for(self, mon, tick, seconds):
        for _ in range(int(seconds // 5)):
            tick(5.0)
            mon.sample()

    def test_other_load_over_threshold_for_the_window_is_sustained(self):
        mon, tick = self.monitor(other=0.8, ours=0.1)
        self.run_for(mon, tick, 60)
        self.assertFalse(mon.sustained(0.5))  # the window is not yet covered
        self.run_for(mon, tick, 70)
        self.assertTrue(mon.sustained(0.5))
        self.assertAlmostEqual(mon.mean()[0], 0.8, places=2)

    def test_our_own_children_do_not_count(self):
        mon, tick = self.monitor(other=0.1, ours=0.8)
        self.run_for(mon, tick, 150)
        self.assertFalse(mon.sustained(0.5))
        self.assertAlmostEqual(mon.mean()[0], 0.1, places=2)

    def test_samples_closer_than_the_interval_are_dropped(self):
        mon, tick = self.monitor(other=0.5, ours=0.0)
        tick(1.0)
        self.assertIsNone(mon.sample())


def make_fights_job(lab: Path, name: str, n=4, matched=(0, 2), days_old=0):
    """A finished fights job: raw outputs for n fights, verdicts (matched ones identical), scratch, a minimal pack."""
    job = lab / "runs" / name
    f = job / "fights"
    (f / "x.packs" / "minimal").mkdir(parents=True)
    (f / "x.packs" / "minimal" / "SEED1.json").write_text('{"runs": []}', encoding="utf-8")
    (job / "logs").mkdir()
    res = [{"seed": f"S{i}", "steps": [{"state": "x" * 2000}] * 5} for i in range(n)]
    for suffix in (".out.json", ".emu.json", ".reanchor.json"):
        (f / ("x" + suffix)).write_text(json.dumps({"results": res}), encoding="utf-8")
    (f / "x.verdicts.json").write_text(json.dumps([{"seed": f"S{i}", "identical": i in matched, "mirror_gap": []} for i in range(n)]), encoding="utf-8")
    (f / "x.summary.txt").write_text("4 fights\n", encoding="utf-8")
    (f / "x.shrink.emu.json").write_text("y" * 5000, encoding="utf-8")
    (job / "serve").mkdir()
    (job / "serve" / "scratch.bin").write_bytes(b"z" * 5000)
    (job / "job.json").write_text("{}", encoding="utf-8")
    (job / "done.rc").write_text("0\n", encoding="utf-8")
    if days_old:
        old = time.time() - days_old * 86400
        os.utime(job / "done.rc", (old, old))
    return job


class PruneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.lab = self.tmp / "lab"

    def test_protected_names(self):
        for p in ("groups.json", "summary.md", "targets.json", "state.json", "status.txt", "job.json", "done.rc", "a.verdicts.json",
                  "x.packs/minimal/S.json", "kits/C3-F01.md", "packets/grades.json"):
            self.assertTrue(pr.protected(Path(p)), p)
        for p in ("x.out.json", "serve/a", "x.emu.json"):
            self.assertFalse(pr.protected(Path(p)), p)

    def test_scratch_goes_first_and_enough_stops_it(self):
        job = make_fights_job(self.lab, "20260101T0000-fights")
        freed = pr.prune(self.lab, need=1000)
        self.assertGreaterEqual(freed, 5000)
        self.assertFalse((job / "serve").exists())
        self.assertFalse((job / "fights" / "x.shrink.emu.json").exists())
        self.assertEqual(len(json.loads((job / "fights" / "x.out.json").read_text(encoding="utf-8"))["results"][1]["steps"]), 5)  # raw kept

    def test_matched_fights_are_stubbed_in_place_and_stay_aligned(self):
        job = make_fights_job(self.lab, "20260101T0000-fights")
        pr.prune(self.lab, need=10**9)  # as much as it may
        for suffix in (".out.json", ".emu.json", ".reanchor.json"):
            res = json.loads((job / "fights" / ("x" + suffix)).read_text(encoding="utf-8"))["results"]
            self.assertEqual([bool(r.get("pruned")) for r in res], [True, False, True, False])
            self.assertEqual(res[0], {"seed": "S0", "pruned": True, "steps": []})
            self.assertTrue(res[1]["steps"])  # a diverging fight keeps everything
        self.assertTrue((job / "pruned.json").exists())

    def test_never_removes_minimal_packs_groups_verdicts_or_state(self):
        job = make_fights_job(self.lab, "20260101T0000-fights", days_old=30)
        (self.lab / "groups.json").write_text("{}", encoding="utf-8")  # newer than done.rc: tier C is allowed
        (self.lab / "summary.md").write_text("s", encoding="utf-8")
        (self.lab / "state.json").write_text("{}", encoding="utf-8")
        pr.prune(self.lab, need=10**12)
        self.assertTrue((job / "fights" / "x.packs" / "minimal" / "SEED1.json").exists())
        self.assertTrue((job / "fights" / "x.verdicts.json").exists())
        self.assertTrue((job / "fights" / "x.summary.txt").exists())
        for n in ("job.json", "done.rc"):
            self.assertTrue((job / n).exists())
        for n in ("groups.json", "summary.md", "state.json"):
            self.assertTrue((self.lab / n).exists())
        self.assertFalse((job / "fights" / "x.out.json").exists())  # tier C took the raw files of an old, processed job

    def test_a_running_job_is_not_touched(self):
        job = make_fights_job(self.lab, "20260101T0000-fights")
        pr.prune(self.lab, need=10**9, busy={job.name})
        self.assertTrue((job / "serve").exists())

    def test_tier_c_waits_for_age_and_for_groups(self):
        job = make_fights_job(self.lab, "20260101T0000-fights", days_old=1)
        (self.lab / "groups.json").write_text("{}", encoding="utf-8")
        pr.prune(self.lab, need=10**12)
        self.assertTrue((job / "fights" / "x.out.json").exists())  # too young
        job2 = make_fights_job(self.lab, "20250101T0000-fights", days_old=30)
        (self.lab / "groups.json").unlink()
        pr.prune(self.lab, need=10**12)
        self.assertTrue((job2 / "fights" / "x.out.json").exists())  # nothing has read it yet


class PlansTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.tree = self.tmp / "tree"
        (self.tree / "referee").mkdir(parents=True)
        (self.tree / "tools" / "fidelity").mkdir(parents=True)

    def ctx(self, **params):
        return pl.Ctx(lab=self.tmp / "lab", tree=self.tree, game=self.tmp / "g" / "game", fork=self.tmp / "fork", py="py", sha="a" * 40,
                      job_dir=self.tmp / "lab" / "runs" / "20260101T0000-fights", job_name="20260101T0000-fights",
                      params={"batches": 2, "batch": 5, "start": 7, "servers": 3, "shrink": 2, "procs_refset": 4, "procs_corpus": 8, **params}, backlog_dir=self.tmp / "bl",
                      backlog=None, codex=self.tmp / "codex.gz", decompiled=self.tmp / "dec", records=self.tmp / "rec",
                      game_src=self.tmp / "gsrc", nuget=self.tmp / "nuget")

    def test_expand_keeps_unknown_braces(self):
        self.assertEqual(self.ctx().expand("{sha7}-{nope}"), "aaaaaaa-{nope}")

    def test_fights_fall_back_to_coverage_with_the_jobs_shim_on_an_old_tree(self):
        (self.tree / "referee" / "scenarios.py").write_text('choices=("dense", "coverage")', encoding="utf-8")
        mode, has_jobs = pl.fights_mode(self.tree)
        self.assertEqual((mode, has_jobs), ("coverage", False))
        steps = pl.plan_fights(self.ctx())
        batches = [s for s in steps if s.name.startswith("batch-")]
        self.assertEqual(len(batches), 2)
        argv = batches[1].cmds[0].argv
        self.assertEqual(argv[:2], ["{py}", "-c"])
        self.assertIn("coverage", argv)
        self.assertEqual(argv[argv.index("--start") + 1], "8")  # start + batch index: a new stream per batch
        self.assertEqual(argv[argv.index("--fights") + 1], "5")
        self.assertNotIn("--jobs", argv)

    def test_fights_use_targets_and_jobs_when_the_tree_has_them(self):
        (self.tree / "referee" / "scenarios.py").write_text('choices=("dense", "coverage", "targets") "--jobs"', encoding="utf-8")
        (self.tree / "tools" / "fidelity" / "lab_targets.py").write_text("", encoding="utf-8")
        mode, has_jobs = pl.fights_mode(self.tree)
        self.assertEqual((mode, has_jobs), ("targets", True))
        steps = pl.plan_fights(self.ctx())
        self.assertIn("targets", [s.name for s in steps])
        argv = [s for s in steps if s.name == "batch-00"][0].cmds[0].argv
        self.assertEqual(argv[:2], ["{py}", "referee/scenarios.py"])
        self.assertEqual(argv[argv.index("--targets") + 1], "{lab}/targets.json")
        self.assertEqual(argv[argv.index("--jobs") + 1], "{job}/fights")

    def test_a_command_with_a_missing_tool_is_skipped_with_a_reason(self):
        c = self.ctx()
        steps = pl.plan_presort(c)
        by = {cmd.name: cmd for s in steps for cmd in s.cmds}
        self.assertIn("missing on this tree", by["refset"].skip_reason(c))
        self.assertIn("lab_groups.py", by["groups"].skip_reason(c))
        c.skip = ("corpus",)
        self.assertIn("--skip corpus", by["corpus"].skip_reason(c))

    def test_skip_matches_a_step_name_too(self):
        c = self.ctx()
        c.skip = ("checks",)
        steps = pl.plan_presort(c)
        step = [s for s in steps if s.name == "checks"][0]
        self.assertTrue(all("--skip checks" in (cmd.skip_reason(c, "checks") or "") for cmd in step.cmds))

    def test_runnable_names_the_missing_input(self):
        (self.tree / "referee" / "scenarios.py").write_text("", encoding="utf-8")
        c = self.ctx()
        why = pl.runnable(pl.plan_fights(c), c)
        self.assertIn("missing on this tree", why)
        for rel in ("tools/emudump/bin/Release/net9.0/EmuDump.dll", "referee/driver/bin/Release/net9.0/SpirelineReferee.dll"):
            (self.tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.tree / rel).write_text("", encoding="utf-8")
        self.assertIsNone(pl.runnable(pl.plan_fights(c), c))

    def test_jobs_with_no_input_are_unavailable(self):
        with self.assertRaises(pl.Unavailable):
            pl.plan_probes(self.ctx(probe_ids=[]))
        with self.assertRaises(pl.Unavailable):
            pl.plan_mutation(self.ctx(mutation_source=None))

    def test_probe_ids_come_from_kits_that_miss_a_rule(self):
        bl = self.tmp / "bl"
        (bl / "packets").mkdir(parents=True)
        (bl / "packets" / "grades.json").write_text(json.dumps({
            "A": {"grade": "B", "missing": ["rule probe"], "probes": ["P01", "P02"]},
            "B": {"grade": "A", "missing": [], "probes": ["P09"]},
            "C": {"grade": "C", "missing": ["rule"], "probes": ["P02", "P03"]}}), encoding="utf-8")
        self.assertEqual(pl.probe_ids(bl), ["P01", "P02", "P03"])

    def test_the_safety_check_halts_on_a_connection_or_a_touched_profile(self):
        c = self.ctx()
        f = c.job_dir / "fights"
        f.mkdir(parents=True)
        check = pl._halt_on_unsafe("b")
        (f / "b.summary.txt").write_text("6 fights: 6 identical, network [], real profile touched False\n", encoding="utf-8")
        self.assertIsNone(check(c))
        (f / "b.summary.txt").write_text("6 fights, network ['1.2.3.4:443'], real profile touched False\n", encoding="utf-8")
        self.assertTrue(check(c).startswith("HALT:"))
        (f / "b.summary.txt").write_text("6 fights, network [], real profile touched True\n", encoding="utf-8")
        self.assertTrue(check(c).startswith("HALT:"))
        (f / "b.summary.txt").write_text("6 fights, network None, real profile touched None\n", encoding="utf-8")
        self.assertIsNone(check(c))  # unknown is a failed batch, not a halt


class RunnerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        saved = dict(pl.PLANS)
        self.addCleanup(lambda: (pl.PLANS.clear(), pl.PLANS.update(saved)))

    def runner(self, *extra: str) -> rn.Runner:
        a = rn.parse_args(["--lab", str(self.tmp / "lab"), "--tree", str(self.tmp / "tree"), "--game", str(self.tmp / "game" / "g"),
                           "--backlog-dir", str(self.tmp / "bl"), "--poll", "0.05", "--wait-poll", "0.05", "--grace", "0",
                           "--no-guard", "user,load,disk", *extra])
        r = rn.Runner(a)
        r.lab.runs.mkdir(parents=True, exist_ok=True)
        return r

    def py(self, name: str, code: str, **kw) -> pl.Cmd:
        return pl.Cmd(name, ["{py}", "-c", code], cwd="{lab}", **kw)

    def test_a_job_runs_to_a_done_rc_written_last(self):
        pl.PLANS["fights"] = lambda c: [
            pl.Step("a", [self.py("one", "print('hello')")], mode="serial", required=True),
            pl.Step("b", [self.py("two", "import sys; sys.exit(1)", ok_rcs=(0, 1)), self.py("three", "print(3)")]),
            pl.Step("c", [pl.Cmd("gone", ["{py}", "x"], needs=["{tree}/nothing"])]),
        ]
        r = self.runner()
        run = r.new_run("fights", "abc")
        self.assertIsNone(r.execute(run))
        d = json.loads((run.dir / "job.json").read_text(encoding="utf-8"))
        self.assertEqual([s["status"] for s in d["steps"]], ["done", "done", "skipped"])
        self.assertEqual((run.dir / "done.rc").read_text().strip(), "0")
        self.assertEqual(d["rc"], 0)
        self.assertEqual(rn.read_rc(run.dir / "logs" / "two.rc"), 1)
        self.assertIn("hello", (run.dir / "logs" / "one.log").read_text())
        self.assertIn("# {py} -c", (run.dir / "logs" / "one.log").read_text())  # the symbolic command, not an absolute path
        self.assertEqual(r.state["counters"]["jobs_done"], {"fights": 1})
        self.assertTrue((r.lab.root / "status.txt").read_text().startswith("20"))
        self.assertEqual(len((r.lab.root / "status.txt").read_text().splitlines()), 1)

    def test_a_failed_required_step_stops_the_job_and_marks_the_presort(self):
        pl.PLANS["presort"] = lambda c: [
            pl.Step("tree", [self.py("t", "import sys; sys.exit(3)")], mode="serial", required=True),
            pl.Step("later", [self.py("l", "open('later.txt','w')")]),
        ]
        r = self.runner()
        run = r.new_run("presort", "deadbeef" * 5)
        r.execute(run)
        self.assertEqual((run.dir / "done.rc").read_text().strip(), "1")
        self.assertFalse((r.lab.root / "later.txt").exists())
        self.assertEqual(run.rec("later")["status"], "skipped")
        self.assertIsNone(r.state["last_presorted_sha"])
        self.assertEqual(r.state["bad_shas"]["deadbeef" * 5]["fails"], 1)
        self.assertIsNone(r.state["tree_sha"])
        job, _, why = r.choose("deadbeef" * 5)
        self.assertEqual(job, "")  # not retried at once
        self.assertIn("retry after", why)

    def test_counts_and_backoff_after_a_job_whose_core_commands_all_fail(self):
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [self.py("good", "pass", core=True, counts={"fights": 5})], mode="serial")]
        r = self.runner()
        r.execute(r.new_run("fights", "abc"))
        self.assertEqual(r.state["counters"]["fights"], 5)
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [self.py("bad", "import sys; sys.exit(2)", core=True, counts={"fights": 5})])]
        first = r.new_run("fights", "abc")
        r.execute(first)
        self.assertEqual((first.dir / "done.rc").read_text().strip(), "1")
        self.assertEqual(r.state["counters"]["fights"], 5)  # a failed batch counts nothing
        self.assertNotIn("fights", r.state["backoff"])
        r.execute(r.new_run("fights", "abc"))
        self.assertIn("fights", r.state["backoff"])  # the second failure in a row backs the job off
        self.assertIn("backing off", r.unavailable("fights", "abc"))
        r.state["backoff"]["fights"] = 0
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [self.py("good", "pass", core=True)])]
        r.execute(r.new_run("fights", "abc"))
        self.assertEqual(r.state["fail_streak"]["fights"], 0)

    def test_resume_keeps_finished_steps_and_commands(self):
        pl.PLANS["fights"] = lambda c: [
            pl.Step("a", [self.py("one", "open('ran-a','w')")], mode="serial"),
            pl.Step("b", [self.py("two", "open('ran-two','w')"), self.py("three", "open('ran-three','w')")]),
        ]
        r = self.runner()
        run = r.new_run("fights", "abc")
        run.rec("a")["status"] = "done"
        (run.dir / "logs" / "two.rc").write_text("0\n")  # one command of step b finished before the stop
        run.save()
        again = rn.Run.load(run.dir)
        self.assertIsNone(r.execute(again))
        self.assertFalse((r.lab.root / "ran-a").exists())
        self.assertFalse((r.lab.root / "ran-two").exists())
        self.assertTrue((r.lab.root / "ran-three").exists())
        self.assertEqual((run.dir / "done.rc").read_text().strip(), "0")

    def test_choose_resumes_a_job_on_the_same_main_and_closes_one_on_another(self):
        r = self.runner()
        run = r.new_run("fights", "abc")
        r.state["last_presorted_sha"] = r.state["tree_sha"] = "abc"
        job, got, why = r.choose("abc")
        self.assertEqual((job, got.name, why), ("fights", run.name, f"resume {run.name}"))
        job, got, why = r.choose("def")
        self.assertEqual((job, got), ("presort", None))
        self.assertEqual((run.dir / "done.rc").read_text().strip(), "3")

    def test_rotation_skips_a_job_that_cannot_run_and_moves_on(self):
        def unavailable(c):
            raise pl.Unavailable("nothing to do")
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [self.py("x", "pass", core=True)])]
        pl.PLANS["probes"] = unavailable
        pl.PLANS["mutation"] = unavailable
        pl.PLANS["invariants"] = unavailable
        r = self.runner()
        r.state["last_presorted_sha"] = r.state["tree_sha"] = "abc"
        seen = []
        for _ in range(3):
            job, _, _ = r.choose("abc")
            seen.append(job)
        self.assertEqual(seen, ["fights", "fights", "fights"])

    def test_once_needs_the_tree_presorted(self):
        r = self.runner("--once", "fights")
        job, _, why = r.choose("abc")
        self.assertEqual(job, "")
        self.assertIn("presort", why)
        self.assertEqual(self.runner("--once", "presort").choose("abc")[0], "presort")

    def test_pause_now_stops_the_command_and_it_restarts_from_scratch_when_released(self):
        marker = self.tmp / "runs.txt"
        code = f"import time; open({str(marker)!r}, 'a').write('x'); time.sleep(2.5); open({str(marker)!r}, 'a').write('done')"
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [self.py("slow", code, cleanup=[])], mode="serial")]
        r = self.runner()
        run = r.new_run("fights", "abc")
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("v", r.execute(run)))
        t.start()
        deadline = time.time() + 20
        while not marker.exists() and time.time() < deadline:
            time.sleep(0.05)
        (r.lab.root / "PAUSE").write_text("now\n", encoding="utf-8")
        while r.state["counters"]["interrupts"].get("pause", 0) < 1 and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(r.state["counters"]["interrupts"].get("pause"), 1)
        self.assertFalse((run.dir / "done.rc").exists())
        self.assertIsNone(rn.read_rc(run.dir / "logs" / "slow.rc"))  # a killed command leaves no rc
        (r.lab.root / "PAUSE").unlink()
        t.join(30)
        self.assertFalse(t.is_alive())
        self.assertEqual(marker.read_text(), "x" + "x" + "done")  # started, killed, started again, finished
        self.assertEqual((run.dir / "done.rc").read_text().strip(), "0")

    def test_a_pause_with_grace_lets_the_step_finish(self):
        marker = self.tmp / "g.txt"
        code = f"import time; open({str(marker)!r}, 'a').write('x'); time.sleep(1.0); open({str(marker)!r}, 'a').write('done')"
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [self.py("slow", code)], mode="serial"), pl.Step("b", [self.py("after", "open('b','w')")])]
        r = self.runner("--grace", "30")
        run = r.new_run("fights", "abc")
        t = threading.Thread(target=lambda: r.execute(run))
        t.start()
        deadline = time.time() + 20
        while not marker.exists() and time.time() < deadline:
            time.sleep(0.05)
        (r.lab.root / "PAUSE").write_text("", encoding="utf-8")
        time.sleep(2.5)  # the running step ends inside its grace; the next step must not start
        self.assertEqual(marker.read_text(), "xdone")
        self.assertFalse((r.lab.root / "b").exists())
        self.assertFalse((run.dir / "done.rc").exists())
        (r.lab.root / "PAUSE").unlink()
        t.join(30)
        self.assertTrue((r.lab.root / "b").exists())

    def test_the_unsafe_batch_halts_the_lab(self):
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [self.py("g", "print('x')", post=lambda c: "HALT: the game opened a connection")], mode="serial")]
        r = self.runner()
        run = r.new_run("fights", "abc")
        with self.assertRaises(rn.Halted):
            r.execute(run)
        self.assertIn("connection", (r.lab.root / "HALT").read_text())
        self.assertFalse((run.dir / "done.rc").exists())  # the job stays open
        self.assertEqual(r.guards.poll().kind, "halt")

    def test_user_input_stops_the_runner_and_leaves_state_behind(self):
        r = self.runner()
        r.guards.disabled.discard("user")
        r.guards.idle_fn = lambda: 3.0
        r.setup = lambda: 0  # not the priority, job object and signal handlers of the test process
        self.assertEqual(r.run(), 0)
        self.assertIn("stopped(user)", (r.lab.root / "status.txt").read_text())
        self.assertEqual(list((r.lab.root / "runs").iterdir()), [])
        self.assertTrue((r.lab.root / "state.json").exists())

    def test_setup_refuses_the_main_checkout_and_the_reviewed_tree(self):
        a = rn.parse_args(["--lab", str(self.tmp / "lab"), "--tree", str(ROOT), "--fork", str(ROOT)])
        self.assertEqual(rn.Runner(a).setup(), 2)
        a = rn.parse_args(["--lab", "D:/spireline/src/lab", "--tree", str(self.tmp / "tree"), "--fork", str(ROOT)])
        self.assertEqual(rn.Runner(a).setup(), 2)
        a = rn.parse_args(["--lab", str(ROOT / "lab-inside"), "--tree", str(self.tmp / "tree"), "--fork", str(ROOT)])
        self.assertEqual(rn.Runner(a).setup(), 2)

    def test_dry_run_prints_the_plan_and_changes_nothing(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = rn.main(["--dry-run", "--lab", str(self.tmp / "lab-dry"), "--tree", str(self.tmp / "tree"), "--game", str(self.tmp / "g"),
                          "--backlog-dir", str(self.tmp / "bl")])
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("guard verdict now:", text)
        self.assertIn("next job:", text)
        self.assertFalse((self.tmp / "lab-dry").exists())
        self.assertFalse((self.tmp / "tree").exists())

    def test_atomic_write_and_state_roundtrip(self):
        r = self.runner()
        r.state["rotation"] = 3
        r.state["counters"]["fights"] = 12
        r.lab.save_state(r.state)
        again = rn.Lab(r.lab.root).load_state()
        self.assertEqual((again["rotation"], again["counters"]["fights"], again["counters"]["interrupts"]), (3, 12, {}))
        self.assertEqual([p.name for p in r.lab.root.iterdir() if p.name.endswith(".tmp")], [])


@unittest.skipUnless(WIN, "Job Objects are Windows only")
class JobObjectTest(unittest.TestCase):
    def test_killing_the_owner_from_outside_kills_its_whole_tree(self):
        code = textwrap.dedent(f"""
            import os, subprocess, sys, time
            sys.path.insert(0, {str(ROOT)!r})
            from tools.lab import winsys
            j = winsys.JobObject(); j.assign_self()
            g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
            print(os.getpid(), g.pid, flush=True)
            time.sleep(120)
        """)
        p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        pids = []
        try:
            owner, grandchild = (int(x) for x in p.stdout.readline().split())
            pids = [owner, grandchild]
            self.assertTrue(winsys.parent_alive(grandchild))  # parent_alive(pid) is "pid is running"
            subprocess.run(["taskkill", "/F", "/PID", str(owner)], capture_output=True, check=False)  # no cleanup runs: a hard kill
            deadline = time.time() + 15
            while winsys.parent_alive(grandchild) and time.time() < deadline:
                time.sleep(0.1)
            self.assertFalse(winsys.parent_alive(grandchild), "the grandchild outlived the job's owner")
        finally:
            for pid in pids:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False)
            p.kill()

    def test_step_jobs_nest_inside_the_runner_job_and_count_cpu(self):
        code = textwrap.dedent(f"""
            import subprocess, sys, time
            sys.path.insert(0, {str(ROOT)!r})
            from tools.lab import winsys
            root = winsys.JobObject(); root.assign_self()
            step = winsys.JobObject()
            p = winsys.suspended_popen([sys.executable, "-c", "sum(range(3*10**7))"], step)
            p.wait()
            print(step.cpu_time() > 0, root.cpu_time() >= step.cpu_time(), winsys.set_below_normal(), winsys.priority_class() == 0x4000)
        """)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(out.stdout.split(), ["True"] * 4, out.stdout + out.stderr)

    def test_system_times_and_idle_are_readable(self):
        idle, kernel, user = winsys.system_times()
        self.assertTrue(kernel >= idle >= 0 and user >= 0)
        self.assertIsNotNone(winsys.user_idle_seconds())


@unittest.skipUnless(WIN and shutil.which("powershell"), "needs Windows PowerShell")
class RegisterTaskTest(unittest.TestCase):
    def tasks(self) -> str:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-ScheduledTask -TaskName 'Spireline fix lab' -ErrorAction SilentlyContinue | Measure-Object).Count"],
                           capture_output=True, text=True, timeout=60, check=False)
        return r.stdout.strip()

    def test_whatif_prints_the_definition_and_registers_nothing(self):
        before = self.tasks()
        script = ROOT / "tools" / "lab" / "register_task.ps1"
        for extra in ([], ["-Unregister"]):
            r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-WhatIf", *extra],
                               capture_output=True, text=True, timeout=120, check=False)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("Nothing changed", r.stdout)
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-WhatIf"],
                           capture_output=True, text=True, timeout=120, check=False)
        for want in ("Spireline fix lab", "Interactive, Limited", "on idle", "RunOnlyIfIdle=True", "IdleDuration=PT10M", "StopOnIdleEnd=True",
                     "RestartOnIdle=True", "Priority=7", "WakeToRun=False", "DisallowStartIfOnBatteries=True", "ExecutionTimeLimit=PT12H",
                     "MultipleInstances=IgnoreNew", "runner.py"):
            self.assertIn(want, r.stdout)
        self.assertEqual(self.tasks(), before)


if __name__ == "__main__":
    unittest.main()
