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
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.lab import guards as gd  # noqa: E402
from tools.lab import plans as pl  # noqa: E402
from tools.lab import prune as pr  # noqa: E402
from tools.lab import runner as rn  # noqa: E402
from tools.lab import walk_reuse as wr  # noqa: E402
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

    def test_the_pause_tag_is_the_first_word_after_an_optional_now(self):
        pause = self.lab / "PAUSE"
        self.assertEqual(gd.pause_tag(self.lab), "")  # no file
        for text, tag in (("", ""), ("now\n", ""), ("w4-macro 2026-10-25T07:01:00Z\n", "w4-macro"), ("NOW w4-macro\n", "w4-macro"),
                          ("now\r\nsecond line\n", ""), ("a/b\\c:d" + "x" * 80, "abc:d" + "x" * 35)):
            pause.write_text(text, encoding="utf-8")
            self.assertEqual(gd.pause_tag(self.lab), tag, text)
        pause.write_bytes("now w4-macro\n".encode("utf-16"))  # PowerShell 5.1 writes UTF-16 with a BOM
        self.assertEqual(gd.pause_tag(self.lab), "w4-macro")

    def test_main_lock_pauses(self):
        g = self.guards()
        self.lock.write_text("", encoding="utf-8")
        self.assertEqual(g.poll().kind, "lock")

    def test_main_lock_is_ignored_when_the_caller_says_the_running_job_is_not_a_presort(self):
        g = self.guards()
        self.lock.write_text("", encoding="utf-8")
        v = g.poll(lock_stops=False)
        self.assertTrue(v.ok)
        self.assertTrue(g.readings["lock_ignored"])
        self.assertEqual(g.poll().kind, "lock")  # the default is the old behaviour
        self.assertFalse(g.readings["lock_ignored"])
        (self.lab / "PAUSE").write_text("", encoding="utf-8")
        self.assertEqual(g.poll(lock_stops=False).kind, "pause")  # other guards are unchanged
        (self.lab / "PAUSE").unlink()
        self.lock.unlink()
        g.poll(lock_stops=False)
        self.assertFalse(g.readings["lock_ignored"])

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

    def test_control_files_are_read_whoever_wrote_them(self):
        # PowerShell 5.1's `echo now > PAUSE` writes UTF-16 LE with a BOM, 7 writes UTF-8 with a BOM, an editor may leave NULs
        g = self.guards(grace=77)
        for raw in ("now\r\n".encode("utf-16"), b"\xef\xbb\xbfnow\r\n", b"n\x00o\x00w\x00\n\x00", b"now\x00\x00\x00", b"  Now  \n"):
            (self.lab / "PAUSE").write_bytes(raw)
            v = g.poll()
            self.assertEqual((v.kind, v.grace), ("pause", 0.0), raw)
        (self.lab / "PAUSE").write_bytes("later\r\n".encode("utf-16"))
        self.assertEqual(77, g.poll().grace)
        (self.lab / "PAUSE").write_bytes(b"\xff\xff\xff\x00")  # garbage is a pause with grace, not a crash
        v = g.poll()
        self.assertEqual(("pause", 77), (v.kind, v.grace))
        (self.lab / "PAUSE").write_bytes(b"")
        self.assertEqual(77, g.poll().grace)
        (self.lab / "PAUSE").unlink()
        (self.lab / "HALT").write_bytes("network seen\r\n".encode("utf-16"))
        self.assertEqual("network seen", g.poll().detail)

    def test_stay_pauses_on_input_and_halt_instead_of_leaving(self):
        idle = [5.0]
        g = self.guards(idle_fn=lambda: idle[0], stay=True)
        v = g.poll()
        self.assertEqual((v.kind, v.leave, v.grace), ("user", False, 0))
        idle[0] = 500.0
        self.assertTrue(g.poll().ok)
        (self.lab / "HALT").write_text("network seen\n", encoding="utf-8")
        v = g.poll()
        self.assertEqual((v.kind, v.leave), ("halt", False))
        (self.lab / "HALT").unlink()
        self.assertTrue(g.poll().ok)
        clock, advance = fake_clock()  # the trial's end still ends the runner
        g = self.guards(deadline=10.0, clock=clock, stay=True)
        advance(10)
        self.assertTrue(g.poll().leave)


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

    def test_shrink_rounds_and_session_copies_are_scratch_and_the_records_of_the_lab_stay(self):
        job = make_fights_job(self.lab, "20260101T0000-fights")
        runs = self.lab / "runs" / "20260101T0100-runs" / "runs"
        runs.mkdir(parents=True)
        for rel in ("fights/x.shrink0.json", "fights/x.shrink12.out.json", "fights/x.shrink.json"):
            (job / rel).write_bytes(b"s" * 300)
        for n in ("r.p0.json", "r.p3.out.json", "r.out.json", "r.runs.json", "r.summary.txt"):
            (runs / n).write_bytes(b"r" * 300)
        (runs.parent / "done.rc").write_text("0\n", encoding="utf-8")
        for n in ("yield.tsv", "mix.json", "coverage-random.json"):
            (self.lab / n).write_text("keep", encoding="utf-8")
        pr.prune(self.lab, need=10**9)
        self.assertFalse((job / "fights" / "x.shrink0.json").exists())
        self.assertFalse((job / "fights" / "x.shrink12.out.json").exists())
        self.assertTrue((job / "fights" / "x.shrink.json").exists())  # the shrink record itself is what lab_groups reads
        self.assertFalse((runs / "r.p0.json").exists())
        self.assertFalse((runs / "r.p3.out.json").exists())
        for n in ("r.runs.json", "r.summary.txt"):
            self.assertTrue((runs / n).exists(), n)
        for n in ("yield.tsv", "mix.json", "coverage-random.json"):
            self.assertTrue((self.lab / n).exists(), n)
        for n in ("yield.tsv", "mix.json", "x.verdicts.json", "r.runs.json", "x.shrink.json"):
            self.assertTrue(pr.protected(Path(n)), n)

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

    def test_a_write_that_fails_leaves_the_file_as_it_was_and_the_rest_goes_on(self):
        job = make_fights_job(self.lab, "20260101T0000-fights")
        before = (job / "fights" / "x.out.json").read_bytes()

        def locked(src, dst):
            raise PermissionError("locked")

        with unittest.mock.patch.object(pr.os, "replace", locked):
            freed = pr.prune(self.lab, need=10**9)  # tier B cannot replace the raw files
        self.assertGreater(freed, 0)  # the scratch (tier A) still went
        self.assertEqual(before, (job / "fights" / "x.out.json").read_bytes())
        self.assertEqual([], list((job / "fights").glob("*.tmp")))  # no half-written copy is left behind
        self.assertFalse((job / "pruned.json").exists())

    def test_a_failing_note_or_log_cut_does_not_stop_the_pruning(self):
        job = make_fights_job(self.lab, "20260101T0000-fights")
        log = job / "logs" / "big.log"
        log.write_bytes(b"L" * 4000)
        real_write_text, real_write_bytes = Path.write_text, Path.write_bytes

        def no_note(p, *a, **kw):
            if p.name == "pruned.json":
                raise OSError("disk full")
            return real_write_text(p, *a, **kw)

        def no_cut(p, *a, **kw):
            if p.suffix == ".log":
                raise OSError("disk full")
            return real_write_bytes(p, *a, **kw)

        with unittest.mock.patch.object(pr, "LOG_CUT", 100), unittest.mock.patch.object(Path, "write_text", no_note), \
                unittest.mock.patch.object(Path, "write_bytes", no_cut):
            freed = pr.prune(self.lab, need=10**9)
        self.assertGreater(freed, 0)
        self.assertEqual(4000, log.stat().st_size)  # not cut, not damaged
        res = json.loads((job / "fights" / "x.out.json").read_text(encoding="utf-8"))["results"]
        self.assertTrue(res[0].get("pruned"))  # tier B went on without its note

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

    def test_a_fights_job_ends_with_the_corpus_import_after_groups(self):
        c = self.ctx()
        steps = pl.plan_fights(c)
        self.assertEqual([s.name for s in steps][-3:], ["groups", "kits", "corpus-import"])
        cmd = steps[-1].cmds[0]
        self.assertEqual(cmd.argv[1:4], ["tools/lab/corpus_import.py", "--job", "{job}"])
        self.assertFalse(cmd.core)  # an import that fails or is missing does not fail the fights
        self.assertIn("missing on this tree", cmd.skip_reason(c))
        (self.tree / "tools" / "lab").mkdir(parents=True, exist_ok=True)
        (self.tree / "tools" / "lab" / "corpus_import.py").write_text("", encoding="utf-8")
        self.assertIsNone(cmd.skip_reason(c))

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

    def test_presort_outputs_use_the_roles_lab_groups_reads(self):
        from tools.fidelity import lab_groups as lg
        by = {cmd.name: cmd for s in pl.plan_presort(self.ctx()) for cmd in s.cmds}
        self.assertEqual(by["refset"].outputs, {"refset_score": "refset/score.json"})
        self.assertEqual(by["corpus"].outputs, {"corpus": "corpus/results.json"})
        for name in ("refset", "corpus", "codex-s0"):
            for role in by[name].outputs:
                if not role.endswith("_err"):
                    self.assertIn(role, lg.OUTPUT_KINDS, f"{name}: role {role} is not read by lab_groups")

    def test_presort_copies_the_refset_score_and_corpus_results_into_the_job(self):
        c = self.ctx()
        score, corpus = self.tmp / "refset" / "all-lab-aaaaaaa.json", self.tmp / "corpus"
        score.parent.mkdir(parents=True)
        score.write_text('{"set": "x"}', encoding="utf-8")
        (c.job_dir / "logs").mkdir(parents=True)
        (c.job_dir / "logs" / "refset.log").write_text(f"table\n  stored {score}\n", encoding="utf-8")
        self.assertIsNone(pl._keep_refset(c))
        self.assertEqual((c.job_dir / "refset" / "score.json").read_text(encoding="utf-8"), '{"set": "x"}')
        res = corpus / "runs" / "lab-aaaaaaa" / "results.json"
        res.parent.mkdir(parents=True)
        res.write_text('{"format": 2}', encoding="utf-8")
        with unittest.mock.patch.dict(os.environ, {"SPIRELINE_CORPUS": str(corpus)}):
            self.assertIsNone(pl._keep_corpus(c))
        self.assertEqual((c.job_dir / "corpus" / "results.json").read_text(encoding="utf-8"), '{"format": 2}')
        (c.job_dir / "logs" / "refset.log").write_text("no score line\n", encoding="utf-8")
        self.assertIn("no refset score", pl._keep_refset(c))  # a note, never a halt

    def test_corpus_root_comes_from_the_trees_corpus_tool(self):
        (self.tree / "tools" / "fidelity").mkdir(parents=True, exist_ok=True)
        (self.tree / "tools" / "fidelity" / "corpus.py").write_text(
            'DEFAULT_CORPUS = Path(os.environ.get("SPIRELINE_CORPUS", r"X:\\c\\corpus"))\n', encoding="utf-8")
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPIRELINE_CORPUS", None)
            self.assertEqual(pl.corpus_root(self.tree), Path("X:\\c\\corpus"))

    def test_kits_go_to_the_backlog_dirs_packets(self):
        c = self.ctx()
        c.backlog = self.tmp / "bl" / "backlog-c9.json"
        kits = [cmd for s in pl.report_steps(c) for cmd in s.cmds if cmd.name == "kits"][0]
        self.assertEqual(kits.argv[kits.argv.index("--out") + 1], "{backlog_dir}/packets")

    def test_groups_name_their_own_job_when_lab_groups_can_read_it(self):
        c = self.ctx()
        groups = lambda: [cmd for s in pl.report_steps(c) for cmd in s.cmds if cmd.name == "groups"][0].argv  # noqa: E731
        (self.tree / "tools" / "fidelity").mkdir(parents=True, exist_ok=True)
        (self.tree / "tools" / "fidelity" / "lab_groups.py").write_text('ap.add_argument("--batch")', encoding="utf-8")
        self.assertNotIn("--job", groups())
        (self.tree / "tools" / "fidelity" / "lab_groups.py").write_text('ap.add_argument("--job")', encoding="utf-8")
        argv = groups()
        self.assertEqual(argv[argv.index("--job") + 1], "{job}")

    def test_probe_ids_come_from_kits_that_miss_a_probe_answer(self):
        bl = self.tmp / "bl"
        (bl / "packets").mkdir(parents=True)
        (bl / "packets" / "grades.json").write_text(json.dumps({
            "A": {"grade": "B", "missing": ["probe answer"], "probes": ["P01", "P02"]},
            "B": {"grade": "A", "missing": [], "probes": ["P09"]},
            "C": {"grade": "C", "missing": ["rule pointer", "probe answer"], "probes": ["P02", "P03"]},
            "D": {"grade": "C", "missing": ["probe answer"]}}), encoding="utf-8")
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

    def test_every_job_type_plans_without_a_crash_on_an_empty_lab(self):
        r = self.runner()
        r.state["last_presorted_sha"] = r.state["tree_sha"] = "abc"
        for job in rn.JOB_TYPES:
            self.assertIsInstance(r.default_params(job), dict)
            r.unavailable(job, "abc")  # a reason or None, never an exception
        for _ in range(len(rn.ROTATION) + 1):
            r.choose("abc")

    def test_mutation_prepares_from_the_biggest_raw_batch_of_the_newest_fights_job(self):
        r = self.runner()
        make_fights_job(r.lab.root, "20260101T0000-fights")
        job = make_fights_job(r.lab.root, "20260102T0000-fights")
        for suffix in (".out.json", ".emu.json", ".reanchor.json"):
            shutil.copy(job / "fights" / ("x" + suffix), job / "fights" / ("y" + suffix))
        (job / "fights" / "y.verdicts.json").write_text("[]", encoding="utf-8")
        (job / "fights" / "y.out.json").write_text("z" * 99999, encoding="utf-8")
        self.assertEqual(r.mutation_source(), "runs/20260102T0000-fights/fights/y")
        (job / "pruned.json").write_text("[]", encoding="utf-8")  # a pruned job's raw files are stubs: not a source
        self.assertEqual(r.mutation_source(), "runs/20260101T0000-fights/fights/x")

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

    # ---- the presort gap: one presort per batch of main moves, the other jobs on the presorted tree meanwhile

    def presorted(self, *extra: str, ago_min: float = 5.0, others: bool = True) -> rn.Runner:
        """A runner whose lab was presorted at aaa, ago_min minutes ago; presort and fights are trivial, the rest cannot run."""
        def unavailable(c):
            raise pl.Unavailable("nothing to do")
        ok = lambda c: [pl.Step("a", [self.py("x", "pass", core=True)])]  # noqa: E731
        pl.PLANS["presort"] = ok
        pl.PLANS["fights"] = ok if others else unavailable
        for j in ("probes", "mutation", "invariants"):
            pl.PLANS[j] = unavailable
        r = self.runner(*extra)
        r.state["last_presorted_sha"] = r.state["tree_sha"] = "aaa"
        r.state["last_presort_end"] = time.time() - ago_min * 60
        return r

    def made_jobs(self, r: rn.Runner) -> list[tuple[str, str]]:
        """(job type, sha) of every job the runner makes from now on, in order."""
        made: list[tuple[str, str]] = []
        orig = r.new_run
        r.new_run = lambda job, sha: (made.append((job, sha)), orig(job, sha))[1]
        return made

    def test_two_main_moves_inside_the_gap_give_one_presort_after_the_gap(self):
        r = self.presorted()  # the default gap is 60 minutes
        made = self.made_jobs(r)
        head = ["bbb"]
        r.main_sha = lambda: head[0]
        self.assertIsNone(r.one_pass())  # main moved to bbb: the rotation's job runs on the presorted tree
        head[0] = "ccc"
        self.assertIsNone(r.one_pass())  # and moved again
        self.assertEqual(made, [("fights", "aaa"), ("fights", "aaa")])
        self.assertEqual((r.state["last_presorted_sha"], r.state["tree_sha"]), ("aaa", "aaa"))
        r.state["last_presort_end"] = time.time() - 61 * 60  # the gap has passed
        self.assertIsNone(r.one_pass())
        self.assertEqual(made[2], ("presort", "ccc"))  # one presort for both moves
        self.assertEqual((r.state["last_presorted_sha"], r.state["tree_sha"]), ("ccc", "ccc"))
        self.assertIsNone(r.one_pass())
        self.assertEqual(made[3], ("fights", "ccc"))
        self.assertEqual([j for j, _ in made].count("presort"), 1)
        self.assertIn("presort of ccc deferred", r.lab.log_path.read_text(encoding="utf-8"))

    def test_inside_the_gap_the_next_pick_is_the_rotations_on_the_presorted_tree(self):
        r = self.presorted()
        job, run, why = r.choose("bbb")
        self.assertEqual((job, run), ("fights", None))
        self.assertTrue(why.startswith("rotation"), why)
        self.assertEqual(r.defer_sha, "aaa")
        self.assertIn("presort of bbb deferred", why)
        self.assertEqual(r.state["rotation"], 1)  # the rotation moved on: this was its slot
        r.main_sha = lambda: "bbb"
        r.one_pass()
        run = rn.Run.load(next(r.lab.runs.iterdir()))
        self.assertEqual((run.job, run.data["main_sha"]), ("fights", "aaa"))  # the job record names the tree it ran on

    def test_a_job_on_the_presorted_tree_resumes_inside_the_gap_and_closes_after_it(self):
        r = self.presorted()
        run = r.new_run("fights", "aaa")
        job, got, _ = r.choose("bbb")
        self.assertEqual((job, got.name), ("fights", run.name))
        r.state["last_presort_end"] = time.time() - 61 * 60
        job, got, _ = r.choose("bbb")
        self.assertEqual((job, got), ("presort", None))
        self.assertEqual((run.dir / "done.rc").read_text().strip(), "3")

    def test_once_presort_ignores_the_gap(self):
        self.assertEqual(self.presorted().choose("bbb")[0], "fights")  # the same lab without --once: deferred
        job, _, why = self.presorted("--once", "presort").choose("bbb")
        self.assertEqual((job, why), ("presort", "--once presort"))
        job, _, why = self.presorted("--once", "fights").choose("bbb")  # any other --once still needs the tree at main
        self.assertEqual(job, "")
        self.assertIn("presorted", why)

    def test_a_presort_is_not_deferred_when_no_other_job_can_run(self):
        r = self.presorted(others=False)
        job, run, why = r.choose("bbb")
        self.assertEqual((job, run), ("presort", None))
        self.assertIn("no other job can run on the tree at aaa", why)
        self.assertIsNone(r.defer_sha)

    def test_a_presort_is_not_deferred_without_a_good_presorted_tree_or_with_the_gap_off(self):
        r = self.presorted()
        r.state["tree_sha"] = None  # a failed presort: nothing may run on the tree
        self.assertEqual(r.choose("bbb")[0], "presort")
        r = self.presorted()
        r.state["last_presort_end"] = None
        self.assertEqual(r.choose("bbb")[0], "presort")  # no presort on record and no folder: long ago
        self.assertEqual(self.presorted("--presort-gap-min", "0").choose("bbb")[0], "presort")  # the gap off: every move
        r = self.presorted()
        r.state["last_presort_end"] = None
        d = r.lab.runs / "20260101T0000-presort"
        d.mkdir()
        (d / "done.rc").write_text("0", encoding="utf-8")  # an older runner left no note: the newest presort folder's end is read
        os.utime(d / "done.rc", (1.0, 1.0))
        self.assertEqual(r.choose("bbb")[0], "presort")  # that was long ago
        os.utime(d / "done.rc")
        self.assertEqual(r.choose("bbb")[0], "fights")  # that was just now

    def test_the_paused_line_carries_the_pause_files_tag(self):
        r = self.runner()
        (r.lab.root / "PAUSE").write_text("w4-macro 2026-10-25T07:01:00Z\n", encoding="utf-8")
        t = threading.Thread(target=r.gate)
        t.start()
        deadline = time.time() + 20
        read = lambda: r.lab.log_path.read_text(encoding="utf-8") if r.lab.log_path.exists() else ""  # noqa: E731
        while "paused:" not in read() and time.time() < deadline:
            time.sleep(0.05)
        (r.lab.root / "PAUSE").unlink()
        t.join(20)
        self.assertFalse(t.is_alive())
        self.assertIn("paused: pause (LAB/PAUSE) (w4-macro)\n", read())

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
        while marker.read_text() != "xdone" and time.time() < deadline:  # the running step ends inside its grace
            time.sleep(0.05)
        time.sleep(1.0)  # ... and the next step must not start
        self.assertEqual(marker.read_text(), "xdone")
        self.assertFalse((r.lab.root / "b").exists())
        self.assertFalse((run.dir / "done.rc").exists())
        (r.lab.root / "PAUSE").unlink()
        t.join(30)
        self.assertTrue((r.lab.root / "b").exists())

    # ---- N4: main.lock stops only a running presort

    def lock_file(self, r: rn.Runner) -> Path:
        d = Path(r.a.backlog_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d / "main.lock"

    def read_log(self, r: rn.Runner) -> str:
        return r.lab.log_path.read_text(encoding="utf-8") if r.lab.log_path.exists() else ""

    def wait_for(self, cond, secs=20.0) -> bool:
        end = time.time() + secs
        while not cond() and time.time() < end:
            time.sleep(0.05)
        return bool(cond())

    LOCK_LINE = "main.lock held: {} continues (only a presort stops for it)\n"

    def test_a_running_nightly_step_survives_a_main_lock_episode_and_the_next_step_starts_under_it(self):
        marker = self.tmp / "n.txt"
        code = f"import time; open({str(marker)!r}, 'a').write('x'); time.sleep(1.5); open({str(marker)!r}, 'a').write('done')"
        pl.PLANS["nightly-real"] = lambda c: [pl.Step("a", [self.py("slow", code, cleanup=[])], mode="serial"),
                                              pl.Step("b", [self.py("after", "open('b','w')")])]
        r = self.runner()  # --grace 0: a stopping guard would end the step at once
        run = r.new_run("nightly-real", "abc")
        lock = self.lock_file(r)
        t = threading.Thread(target=lambda: r.execute(run))
        t.start()
        self.assertTrue(self.wait_for(marker.exists))
        lock.write_text("", encoding="utf-8")
        self.assertTrue(self.wait_for(lambda: (r.lab.root / "b").exists()))  # step b started with the lock still up
        self.assertTrue(lock.exists())
        self.assertEqual(marker.read_text(), "xdone")  # one start, no kill, no restart
        lock.unlink()
        t.join(30)
        self.assertFalse(t.is_alive())
        self.assertEqual((run.dir / "done.rc").read_text().strip(), "0")
        self.assertNotIn("lock", r.state["counters"]["interrupts"])
        self.assertEqual(run.rec("a").get("interrupted", 0), 0)
        log = self.read_log(r)
        self.assertEqual(log.count("main.lock held"), 1)  # once for the episode, however many polls saw it
        self.assertIn(self.LOCK_LINE.format(run.name), log)
        self.assertNotIn("stopped: lock", log)
        self.assertTrue(r.lock_stops)  # between jobs the lock holds every pick again

    def test_a_running_presort_step_stops_on_main_lock_as_before(self):
        marker = self.tmp / "p.txt"
        code = f"import time; open({str(marker)!r}, 'a').write('x'); time.sleep(2.0); open({str(marker)!r}, 'a').write('done')"
        pl.PLANS["presort"] = lambda c: [pl.Step("tree", [self.py("slow", code, cleanup=[])], mode="serial", required=True)]
        r = self.runner()
        run = r.new_run("presort", "abc")
        lock = self.lock_file(r)
        t = threading.Thread(target=lambda: r.execute(run))
        t.start()
        self.assertTrue(self.wait_for(marker.exists))
        lock.write_text("", encoding="utf-8")
        self.assertTrue(self.wait_for(lambda: r.state["counters"]["interrupts"].get("lock", 0) >= 1))
        self.assertEqual(r.state["counters"]["interrupts"].get("lock"), 1)
        self.assertFalse((run.dir / "done.rc").exists())
        self.assertIn("step tree stopped: lock (backlog main.lock)", self.read_log(r))
        self.assertNotIn("main.lock held", self.read_log(r))
        lock.unlink()
        t.join(30)
        self.assertFalse(t.is_alive())
        self.assertEqual(marker.read_text(), "xxdone")  # started, stopped, started again, finished
        self.assertEqual((run.dir / "done.rc").read_text().strip(), "0")

    def test_the_status_line_shows_the_lock_while_a_non_presort_step_runs(self):
        r = self.runner()
        lock = self.lock_file(r)
        r.current = ("job-1", "step-a")
        r.lock_stops = False
        r.status("running", "step-a")
        self.assertNotIn("main.lock", r.lab.status_path.read_text(encoding="utf-8"))
        lock.write_text("", encoding="utf-8")
        self.assertTrue(r.poll().ok)  # the lock does not stop it ...
        line = r.lab.status_path.read_text(encoding="utf-8")
        self.assertIn("running", line)  # ... the line still says it runs, and shows the lock
        self.assertIn("main.lock=held(job continues)", line)
        r.status("running", "step-a")  # a later heartbeat keeps it
        self.assertIn("main.lock=held(job continues)", r.lab.status_path.read_text(encoding="utf-8"))
        lock.unlink()
        self.assertTrue(r.poll().ok)
        r.status("running", "step-a")
        self.assertNotIn("main.lock", r.lab.status_path.read_text(encoding="utf-8"))

    def test_the_continues_line_comes_once_per_lock_episode_and_the_other_guards_still_stop(self):
        r = self.runner()
        lock = self.lock_file(r)
        r.current = ("job-1", "step-a")
        r.lock_stops = False
        lock.write_text("", encoding="utf-8")
        for _ in range(4):
            self.assertTrue(r.poll().ok)
        self.assertEqual(self.read_log(r).count("main.lock held"), 1)
        (r.lab.root / "PAUSE").write_text("", encoding="utf-8")  # PAUSE is not the lock's business: it still stops
        self.assertEqual(r.poll().kind, "pause")
        (r.lab.root / "PAUSE").unlink()
        (r.lab.root / "HALT").write_text("unsafe\n", encoding="utf-8")
        self.assertEqual(r.poll().kind, "halt")
        (r.lab.root / "HALT").unlink()
        lock.unlink()
        self.assertTrue(r.poll().ok)
        lock.write_text("", encoding="utf-8")  # a second episode
        self.assertTrue(r.poll().ok)
        self.assertTrue(r.poll().ok)
        log = self.read_log(r)
        self.assertEqual(log.count("main.lock held"), 2)
        self.assertEqual(log.count(self.LOCK_LINE.format("job-1")), 2)

    def test_no_job_of_any_kind_starts_under_the_lock(self):
        r = self.runner()
        lock = self.lock_file(r)
        lock.write_text("", encoding="utf-8")
        for job in ("presort", "fights", "nightly-real"):
            pl.PLANS[job] = lambda c: []
            run = r.new_run(job, "abc")
            r.execute(run)  # the job ran its (empty) plan; afterwards the pick gate is back to holding on the lock
            self.assertTrue(r.lock_stops)
            self.assertEqual(r.poll().kind, "lock", job)
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("v", r.gate()))  # the gate at the top of every pass
        t.start()
        self.assertTrue(self.wait_for(lambda: "paused: lock" in self.read_log(r)))
        self.assertTrue(t.is_alive())
        lock.unlink()
        t.join(20)
        self.assertFalse(t.is_alive())
        self.assertIsNone(out["v"])

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
        r.setup = lambda: None  # not the priority, job object and signal handlers of the test process
        self.assertEqual(r.run(), 0)
        self.assertIn("stopped(user)", (r.lab.root / "status.txt").read_text())
        self.assertEqual(list((r.lab.root / "runs").iterdir()), [])
        self.assertTrue((r.lab.root / "state.json").exists())

    def test_a_crash_in_a_pass_is_retried_and_three_in_a_row_end_the_runner(self):
        r = self.runner("--idle-poll", "0.1")
        r.setup = lambda: None
        r.main_sha = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        self.assertEqual(r.run(), 1)
        self.assertIn("CRASH 3/3", (r.lab.root / "runner.log").read_text())
        self.assertIn("crashed", (r.lab.root / "status.txt").read_text())

    @unittest.skipUnless(WIN, "the lab mutex is a Windows object")
    def test_a_second_runner_on_a_lab_in_use_does_nothing(self):
        lab = self.tmp / "lab"
        code = textwrap.dedent(f"""
            import sys, time
            sys.path.insert(0, {str(ROOT)!r})
            from tools.lab import runner as rn
            a = rn.parse_args(["--lab", {str(lab)!r}, "--tree", {str(self.tmp / "tree")!r}, "--game", {str(self.tmp / "g")!r}, "--backlog-dir", {str(self.tmp / "bl")!r}])
            r = rn.Runner(a)
            lab_ok = r.setup()
            print("first", lab_ok, flush=True)
            time.sleep(60)
        """)
        first = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(first.stdout.readline().strip(), "first None")  # it holds the lab
            r = self.runner()
            self.assertEqual(r.run(), 0)  # exits at once ...
            self.assertEqual(list((r.lab.root / "runs").iterdir()), [])  # ... and starts nothing
            self.assertIn("another runner already holds", (r.lab.root / "runner.log").read_text())
        finally:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(first.pid)], capture_output=True, check=False)

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

    def test_dry_run_reports_a_full_disk_but_never_prunes(self):
        job = make_fights_job(self.tmp / "lab", "20260101T0000-fights")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rn.main(["--dry-run", "--lab", str(self.tmp / "lab"), "--tree", str(self.tmp / "tree"), "--game", str(self.tmp / "g"),
                     "--backlog-dir", str(self.tmp / "bl"), "--lab-gb", "0.000001",
                     "--no-guard", "user,load"])  # someone at the PC (or a busy CPU) would otherwise answer before the disk guard
        self.assertIn("guard verdict now: disk", buf.getvalue())
        self.assertTrue((job / "serve").exists())
        self.assertTrue((job / "fights" / "x.shrink.emu.json").exists())

    def test_atomic_write_and_state_roundtrip(self):
        r = self.runner()
        r.state["rotation"] = 3
        r.state["counters"]["fights"] = 12
        r.lab.save_state(r.state)
        again = rn.Lab(r.lab.root).load_state()
        self.assertEqual((again["rotation"], again["counters"]["fights"], again["counters"]["interrupts"]), (3, 12, {}))
        self.assertEqual([p.name for p in r.lab.root.iterdir() if p.name.endswith(".tmp")], [])


    def test_weight_the_running_code_cannot_spend_is_logged_on_every_pick_and_changed_sources_are_flagged(self):
        r = self.runner()
        (r.lab.root / "mix.json").write_text(json.dumps({"aimed-bugs": 100, "ghost-a": 15, "ghost-b": 5}), encoding="utf-8")
        r.unavailable = lambda job, sha, strategy=None: None
        read = lambda: (r.lab.root / "runner.log").read_text(encoding="utf-8")  # noqa: E731
        self.assertEqual(r.pick_fights("abc"), ("fights", "aimed-bugs"))
        r.pick_fights("abc")
        self.assertEqual(read().count("20 of 120 weight (17%) is idle"), 2)  # every pick, not only the first
        self.assertIn("restart", read())
        self.assertNotIn("code on disk changed", read())
        r.code_digest0 = {**r.code_digest0, "mix.py": 0}  # a lab source on disk is not what this process loaded
        r.pick_fights("abc")
        r.pick_fights("abc")
        self.assertEqual(read().count("code on disk changed"), 1)  # said once per change
        self.assertIn("(mix.py)", read())


class WalkReuseTest(unittest.TestCase):
    """The presort reuses an earlier presort's codex walk when the EmuDump sources and the data cannot differ."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.fork = self.tmp / "fork"
        self.tree = self.tmp / "tree"
        self.runs = self.tmp / "lab" / "runs"
        self.runs.mkdir(parents=True)
        for rel in (*wr.WALK_TREES, "tools/lab", "docs"):
            (self.fork / rel).mkdir(parents=True)
            (self.fork / rel / "f.txt").write_text("1", encoding="utf-8")
        self.git("init", "-q")
        self.c1 = self.commit("start")
        self.codex = self.tmp / "codex.gz"
        self.codex.write_bytes(b"x")
        self.dec = self.tmp / "dec"
        self.dec.mkdir()
        (self.dec / "a.cs").write_text("x", encoding="utf-8")
        self.old = time.time() - 3600
        for f in (self.codex, self.dec, self.dec / "a.cs"):
            os.utime(f, (self.old - 100, self.old - 100))
        self.sig = pl.walk_sig()
        wr.trees.cache_clear()

    def git(self, *a: str) -> str:
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        return subprocess.run(["git", "-C", str(self.fork), *a], capture_output=True, text=True, check=True, env=env).stdout.strip()

    def commit(self, msg: str, **files: str) -> str:
        for rel, text in files.items():
            (self.fork / rel).write_text(text, encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", msg, "--allow-empty")
        return self.git("rev-parse", "HEAD")

    def job(self, name: str, sha: str, *, shards=4, rc=0, mtime=None, reuse=False, argv=None, params=None) -> Path:
        """A finished presort folder: the walk's four shard commands (or its reuse command) and the shard files."""
        d = self.runs / name
        (d / "codex").mkdir(parents=True)
        mtime = self.old if mtime is None else mtime
        for i in range(shards):
            (d / "codex" / f"s{i}.jsonl").write_text('{"checks": 1}\n', encoding="utf-8")
            os.utime(d / "codex" / f"s{i}.jsonl", (mtime, mtime))
        if reuse:
            cmds = {"codex-reuse": {"status": "done", "rc": rc}}
        else:
            argvs = argv or [c.symbolic() for c in pl.walk_cmds()]
            cmds = {f"codex-s{i}": {"status": "done" if rc == 0 else "failed", "rc": rc, "argv": argvs[i]} for i in range(4)}
        doc = {"main_sha": sha, "params": params or {}, "steps": [{"name": "checks", "cmds": cmds}]}
        (d / "job.json").write_text(json.dumps(doc), encoding="utf-8")
        return d

    def find(self, sha: str) -> str | None:
        wr.trees.cache_clear()
        return wr.find_source(self.runs, self.fork, sha, self.sig, wr.data_floor(self.codex, self.dec))

    def test_walk_trees_are_what_refset_score_builds_emudump_from(self):
        from tools.fidelity import refset_score as rs
        self.assertEqual(wr.WALK_TREES, tuple(sub for sub, _ in rs.EMU_SOURCES))

    def test_a_main_move_that_leaves_the_emudump_sources_alone_reuses_the_walk(self):
        self.job("20261001T0100-presort", self.c1)
        c2 = self.commit("lab code", **{"tools/lab/f.txt": "2", "docs/f.txt": "2"})
        self.assertEqual(self.find(c2), "20261001T0100-presort")
        self.assertEqual(self.find(self.c1), "20261001T0100-presort")

    def test_a_change_under_any_walk_tree_runs_the_walk(self):
        self.job("20261001T0100-presort", self.c1)
        for rel in wr.WALK_TREES:
            c = self.commit("engine", **{f"{rel}/f.txt": "changed " + rel})
            self.assertIsNone(self.find(c), rel)

    def test_the_newest_matching_walk_is_taken_and_a_revert_matches_the_older_one(self):
        self.job("20261001T0100-presort", self.c1)
        c2 = self.commit("engine", **{"src/Sts2Emulator/f.txt": "2"})
        self.job("20261001T0200-presort", c2)
        c3 = self.commit("lab", **{"docs/f.txt": "3"})
        self.assertEqual(self.find(c3), "20261001T0200-presort")
        c4 = self.commit("engine back", **{"src/Sts2Emulator/f.txt": "1"})
        self.assertEqual(self.find(c4), "20261001T0100-presort")  # the trees equal the first walk's again

    def test_an_unfinished_short_or_failed_walk_is_not_reused(self):
        c2 = self.commit("lab", **{"docs/f.txt": "2"})
        self.job("20261001T0100-presort", self.c1, shards=3)
        self.job("20261001T0200-presort", self.c1, rc=1)
        self.assertIsNone(self.find(c2))
        shutil.rmtree(self.runs / "20261001T0100-presort")
        d = self.job("20261001T0300-presort", self.c1)
        (d / "codex" / "s2.jsonl").write_text("", encoding="utf-8")  # an empty shard
        self.assertIsNone(self.find(c2))

    def test_a_walk_older_than_its_data_or_run_with_other_commands_is_not_reused(self):
        c2 = self.commit("lab", **{"docs/f.txt": "2"})
        self.job("20261001T0100-presort", self.c1, mtime=self.old - 500)  # the codex file is newer than the walk
        self.assertIsNone(self.find(c2))
        shutil.rmtree(self.runs / "20261001T0100-presort")
        self.job("20261001T0200-presort", self.c1, argv=["dotnet other"] * 4)
        self.assertIsNone(self.find(c2))
        self.assertIsNone(wr.find_source(self.runs, self.fork, c2, self.sig, float("inf")))  # data that cannot be read
        self.assertIsNone(wr.find_source(self.runs, self.fork, "0" * 40, self.sig, 0.0))  # a sha the fork does not have

    def test_a_job_that_reused_a_walk_is_itself_a_source_and_pruning_the_first_changes_nothing(self):
        self.job("20261001T0100-presort", self.c1)
        c2 = self.commit("lab", **{"docs/f.txt": "2"})
        self.job("20261001T0200-presort", c2, reuse=True, params={"walk_sig": self.sig})
        c3 = self.commit("lab 2", **{"docs/f.txt": "3"})
        self.assertEqual(self.find(c3), "20261001T0200-presort")
        shutil.rmtree(self.runs / "20261001T0100-presort")
        self.assertEqual(self.find(c3), "20261001T0200-presort")
        self.job("20261001T0300-presort", c3, reuse=True)  # no walk_sig recorded: its commands are unknown
        self.assertEqual(self.find(c3), "20261001T0200-presort")

    def test_copy_shards_keeps_times_and_never_leaves_a_half_file(self):
        src = self.job("20261001T0100-presort", self.c1)
        (src / "codex" / "s1.err").write_text("e", encoding="utf-8")
        dst = self.tmp / "out" / "codex"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(wr.copy_shards(src / "codex", dst), 0)
        self.assertEqual(sorted(p.name for p in dst.iterdir()), ["s0.jsonl", "s1.err", "s1.jsonl", "s2.jsonl", "s3.jsonl"])
        self.assertEqual(int((dst / "s2.jsonl").stat().st_mtime), int(self.old))
        (src / "codex" / "s3.jsonl").unlink()
        dst2 = self.tmp / "out2" / "codex"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(wr.copy_shards(src / "codex", dst2), 2)
        self.assertFalse(any(p.name.endswith(".part") for p in dst2.iterdir()))

    def test_the_plan_copies_the_walk_when_the_job_names_a_source_and_walks_otherwise(self):
        c = PlansTest.ctx(self)
        codex_cmds = lambda cx: [x.name for s in pl.plan_presort(cx) for x in s.cmds if x.name.startswith("codex")]  # noqa: E731
        self.assertEqual(codex_cmds(c), ["codex-s0", "codex-s1", "codex-s2", "codex-s3"])
        c.params["walk_from"] = "20261001T0100-presort"
        self.assertEqual(codex_cmds(c), ["codex-reuse"])
        reuse = [x for s in pl.plan_presort(c) for x in s.cmds if x.name == "codex-reuse"][0]
        self.assertTrue(reuse.core)
        self.assertEqual(reuse.argv[:2], ["{py}", "{labtools}/walk_reuse.py"])
        self.assertIn("20261001T0100-presort/codex", reuse.argv[2])
        from tools.fidelity import lab_groups as lg
        for role in reuse.outputs:
            if not role.endswith("_err"):
                self.assertIn(role, lg.OUTPUT_KINDS)
        self.assertEqual(reuse.outputs, pl.walk_cmds()[0].outputs)  # the roles the real walk's first shard names
        c.skip = ("codex",)
        self.assertIn("--skip codex", reuse.skip_reason(c))
        c.params["walk_from"] = "..\\..\\elsewhere"  # a name that is not a presort folder is ignored
        self.assertEqual(codex_cmds(c), ["codex-s0", "codex-s1", "codex-s2", "codex-s3"])

    def test_a_new_presort_records_the_signature_and_the_source_and_the_job_copies_the_walk(self):
        self.job("20261001T0100-presort", self.c1)
        c2 = self.commit("lab", **{"docs/f.txt": "2"})
        saved = dict(pl.PLANS)
        self.addCleanup(lambda: (pl.PLANS.clear(), pl.PLANS.update(saved)))

        def runner(lab: str, fork: Path) -> rn.Runner:
            a = rn.parse_args(["--lab", str(self.tmp / lab), "--tree", str(self.tmp / "tree"), "--game", str(self.tmp / "game" / "g"),
                               "--fork", str(fork), "--backlog-dir", str(self.tmp / "bl"), "--codex", str(self.codex),
                               "--decompiled", str(self.dec), "--poll", "0.05", "--no-guard", "user,load,disk"])
            r = rn.Runner(a)
            r.lab.runs.mkdir(parents=True, exist_ok=True)
            return r
        r = runner("lab", self.fork)
        run = r.new_run("presort", c2)
        self.assertEqual(run.data["params"]["walk_from"], "20261001T0100-presort")
        self.assertEqual(run.data["params"]["walk_sig"], self.sig)
        self.assertIn("20261001T0100-presort stands", r.lab.log_path.read_text(encoding="utf-8"))
        # the walk step of the real plan, run by the real runner: the shards arrive in the job folder under their roles
        pl.PLANS["presort"] = lambda c: [pl.Step("checks", [x for s in pl.plan_presort(c) for x in s.cmds if x.name.startswith("codex")])]
        self.assertIsNone(r.execute(run))
        self.assertEqual(sorted(p.name for p in (run.dir / "codex").glob("s*.jsonl")), [f"s{i}.jsonl" for i in range(4)])
        d = json.loads((run.dir / "job.json").read_text(encoding="utf-8"))
        self.assertEqual(d["steps"][0]["cmds"]["codex-reuse"]["status"], "done")
        self.assertEqual(d["outputs"]["codex_walk"], ["codex/s0.jsonl"])
        self.assertEqual((run.dir / "done.rc").read_text().strip(), "0")
        # no source: the params carry the signature only, and a missing fork never stops a job being made
        r2 = runner("lab2", self.tmp / "no-fork")
        params = r2.new_run("presort", c2).data["params"]
        self.assertEqual(params["walk_sig"], self.sig)
        self.assertNotIn("walk_from", params)


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



    def test_logon_trigger_is_always_on_and_registers_nothing(self):
        before = self.tasks()
        script = ROOT / "tools" / "lab" / "register_task.ps1"
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-Trigger", "Logon", "-WhatIf"],
                           capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for want in ("Spireline fix lab", "Interactive, Limited", "no password stored", "at logon of", "(always on)", "--always-on", "RunOnlyIfIdle=False",
                     "StopOnIdleEnd=False", "Priority=7", "ExecutionTimeLimit=PT0S", "MultipleInstances=IgnoreNew", "RestartCount=3",
                     "RestartInterval=PT5M", "Nothing changed"):
            self.assertIn(want, r.stdout)
        self.assertNotIn("on idle", r.stdout)
        self.assertNotIn("--exit-with-parent", r.stdout)  # nothing would start it again if it took its starter's exit for its own
        self.assertEqual(self.tasks(), before)
        idle = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-WhatIf"],
                              capture_output=True, text=True, timeout=120, check=False)
        self.assertNotIn("--always-on", idle.stdout)  # the idle mode stays the default and is unchanged
        self.assertIn("--exit-with-parent", idle.stdout)
        self.assertNotIn("--lab-gb", idle.stdout)
        self.assertIn("--lab-gb 100", r.stdout)  # a week of unattended data gets room; D: has it
        sized = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-Trigger", "Logon", "-LabGb", "40",
                                "-WhatIf"], capture_output=True, text=True, timeout=120, check=False)
        self.assertIn("--lab-gb 40", sized.stdout)
        bad = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-Trigger", "Weekly", "-WhatIf"],
                             capture_output=True, text=True, timeout=120, check=False)
        self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()


class HiddenConsoleTest(unittest.TestCase):
    """Job commands get a hidden console of their own, so the console programs they start share it instead of each
    opening a visible window (CREATE_NO_WINDOW leaves them no console to share)."""

    @unittest.skipUnless(os.name == "nt", "Windows only")
    def test_job_commands_get_a_hidden_console_of_their_own(self):
        kw = rn.hidden_console_kw()
        self.assertEqual(rn.CREATE_NEW_CONSOLE, kw["creationflags"])
        self.assertFalse(kw["creationflags"] & rn.CREATE_NO_WINDOW)
        si = kw["startupinfo"]
        self.assertTrue(si.dwFlags & subprocess.STARTF_USESHOWWINDOW)
        self.assertEqual(0, si.wShowWindow)


class StaleLockAndLogTest(unittest.TestCase):
    """N1: a `lab-*` QUIET.lock line whose holder is gone is cleared before the next pick; a restarted nightly appends to its log."""

    setUp, runner, py = RunnerCase.setUp, RunnerCase.runner, RunnerCase.py

    def lock(self, line: str) -> Path:
        f = self.tmp / "QUIET.lock"  # next to the lab folder (tmp/lab)
        f.write_text(line + "\n")
        return f

    def test_a_stale_lab_line_with_a_dead_pid_is_cleared_with_a_log_line(self):
        r = self.runner()
        r.lab.root.mkdir(parents=True, exist_ok=True)
        f = self.lock(f"lab-nightly {rn.utc(time.time() - 4000)} pid=99999999")
        with unittest.mock.patch.object(winsys, "parent_alive", return_value=False) as alive:
            self.assertTrue(r.clear_stale_quiet_lock())
        alive.assert_called_once_with(99999999)
        self.assertFalse(f.exists())
        self.assertIn("QUIET.lock: cleared the stale line 'lab-nightly", r.lab.log_path.read_text(encoding="utf-8"))
        self.assertIn("process 99999999 is gone", r.lab.log_path.read_text(encoding="utf-8"))

    def test_a_line_with_no_pid_that_is_old_is_cleared_and_a_young_one_is_not(self):
        r = self.runner()
        r.lab.root.mkdir(parents=True, exist_ok=True)
        f = self.lock(f"lab-nightly {rn.utc(time.time() - 1500)}")  # under three refreshes: live
        self.assertFalse(r.clear_stale_quiet_lock())
        self.assertTrue(f.exists())
        f = self.lock(f"lab-nightly {rn.utc(time.time() - 3000)}")  # over 3 x 15 minutes
        self.assertTrue(r.clear_stale_quiet_lock())
        self.assertFalse(f.exists())

    def test_a_live_pid_another_sessions_line_and_a_missing_file_are_left_alone(self):
        r = self.runner()
        r.lab.root.mkdir(parents=True, exist_ok=True)
        old = rn.utc(time.time() - 7200)
        f = self.lock(f"lab-nightly {old} pid={os.getpid()}")
        with unittest.mock.patch.object(winsys, "parent_alive", return_value=True):
            self.assertFalse(r.clear_stale_quiet_lock())
        self.assertTrue(f.exists())
        f = self.lock(f"w3-blocks {old}")
        self.assertFalse(r.clear_stale_quiet_lock())
        self.assertTrue(f.exists())
        f.unlink()
        self.assertFalse(r.clear_stale_quiet_lock())

    def test_the_pick_clears_the_stale_line_first(self):
        r = self.runner("--once", "presort")
        f = self.lock(f"lab-nightly {rn.utc(time.time() - 4000)} pid=99999999")
        with unittest.mock.patch.object(winsys, "parent_alive", return_value=False), \
                unittest.mock.patch.object(r, "choose", return_value=("", None, "test: nothing")):
            r.one_pass()
        self.assertFalse(f.exists())

    def test_a_restarted_command_appends_to_its_log_after_a_marker_and_others_start_afresh(self):
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [self.py("keep", "print('attempt', open('n.txt').read())", append_log=True),
                                                      self.py("fresh", "print('attempt', open('n.txt').read())")], mode="serial")]
        r = self.runner()
        run = r.new_run("fights", "abc")
        for attempt in ("one", "two"):
            (r.lab.root / "n.txt").write_text(attempt)
            again = rn.Run.load(run.dir)
            again.rec("a")["status"] = "pending"
            again.data["done"] = False
            again.save()
            for name in ("keep", "fresh"):
                (run.dir / "logs" / f"{name}.rc").unlink(missing_ok=True)
            (run.dir / "done.rc").unlink(missing_ok=True)
            self.assertIsNone(r.execute(again))
        keep = (run.dir / "logs" / "keep.log").read_text(encoding="utf-8")
        fresh = (run.dir / "logs" / "fresh.log").read_text(encoding="utf-8")
        self.assertIn("attempt one", keep)
        self.assertIn("attempt two", keep)
        self.assertEqual(1, keep.count("---- restart"))
        self.assertLess(keep.index("attempt one"), keep.index("---- restart"))
        self.assertLess(keep.index("---- restart"), keep.index("attempt two"))
        self.assertNotIn("attempt one", fresh)
        self.assertNotIn("---- restart", fresh)
