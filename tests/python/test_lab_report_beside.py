"""tools/lab/runner.py Report: the groups and kits of a finished job run beside the next job (reset order D44 step D).

Trivial `python -c` commands stand in for the groups and kits; no game, no emulator.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.lab import plans as pl  # noqa: E402
from tools.lab import runner as rn  # noqa: E402


class ReportBesideTest(unittest.TestCase):
    """runner.Report: one chain at a time, groups then kits, beside whatever job runs next."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        saved = dict(pl.PLANS)
        self.addCleanup(lambda: (pl.PLANS.clear(), pl.PLANS.update(saved)))
        self.orig_cmds = pl.report_cmds
        self.addCleanup(lambda: setattr(pl, "report_cmds", self.orig_cmds))
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [self.py("one", "pass", core=True)], mode="serial")]
        pl.PLANS["presort"] = lambda c: [pl.Step("a", [self.py("p", "pass", core=True)], mode="serial")]
        pl.PLANS["nightly-real"] = lambda c: [pl.Step("a", [self.py("n", "pass", core=True)], mode="serial")]

    def runner(self, *extra: str) -> rn.Runner:
        a = rn.parse_args(["--lab", str(self.tmp / "lab"), "--tree", str(self.tmp / "tree"), "--game", str(self.tmp / "game" / "g"),
                           "--backlog-dir", str(self.tmp / "bl"), "--poll", "0.05", "--wait-poll", "0.05", "--grace", "0",
                           "--no-guard", "user,load,disk", *extra])
        r = rn.Runner(a)
        r.lab.runs.mkdir(parents=True, exist_ok=True)
        return r

    def py(self, name: str, code: str, **kw) -> pl.Cmd:
        return pl.Cmd(name, ["{py}", "-c", code], cwd="{lab}", **kw)

    def chain(self, groups: str = "open('groups-ran', 'w')", kits: str = "open('kits-ran', 'w')") -> None:
        pl.report_cmds = lambda c: [self.py("groups", groups), self.py("kits", kits)]

    def wait_report(self, r: rn.Runner, seconds: float = 30.0) -> None:
        end = time.time() + seconds
        while (r.report is not None or r.report_due) and time.time() < end:
            r.service_report()
            time.sleep(0.05)
        self.assertIsNone(r.report, "the report did not end")

    def yield_rows(self, r: rn.Runner) -> list[list[str]]:
        p = r.lab.root / "yield.tsv"
        return [ln.split("\t") for ln in p.read_text(encoding="utf-8").splitlines()[1:]] if p.exists() else []

    def log(self, r: rn.Runner) -> str:
        return (r.lab.root / "runner.log").read_text(encoding="utf-8")

    def steps(self, run: rn.Run, name: str) -> dict:
        return next(s for s in json.loads((run.dir / "job.json").read_text(encoding="utf-8"))["steps"] if s["name"] == name)

    def test_the_report_is_on_with_always_on_only(self):
        self.assertFalse(self.runner().report_beside)
        self.assertTrue(self.runner("--always-on").report_beside)
        self.assertFalse(self.runner("--always-on", "--once", "fights").report_beside)
        self.assertFalse(self.runner("--always-on", "--trial", "5").report_beside)
        self.assertFalse(self.runner("--always-on", "--report-beside", "off").report_beside)
        self.assertTrue(self.runner("--report-beside", "on").report_beside)

    def test_the_plan_drops_the_report_steps_only_when_they_run_beside(self):
        r = self.runner("--report-beside", "on")
        ctx = r.ctx(None, "fights", "abc", r.default_params("fights"))
        self.assertTrue(ctx.report_beside)
        self.assertEqual([], pl.report_steps(ctx))
        ctx.report_beside = False
        self.assertEqual(["groups", "kits"], [s.name for s in pl.report_steps(ctx)])

    def test_a_finished_job_leaves_its_row_for_the_end_of_the_report(self):
        self.chain(kits="import time; time.sleep(0.3); open('kits-ran', 'w')")
        r = self.runner("--report-beside", "on")
        run = r.new_run("fights", "abc")
        self.assertIsNone(r.execute(run))
        self.assertEqual((run.dir / "done.rc").read_text().strip(), "0")  # the job is finished: the next one may start
        self.assertIsNotNone(r.report)
        self.assertEqual([], self.yield_rows(r))  # its row waits for the groups
        self.wait_report(r)
        self.assertTrue((r.lab.root / "groups-ran").exists())
        self.assertTrue((r.lab.root / "kits-ran").exists())
        rows = self.yield_rows(r)
        self.assertEqual([run.name], [row[1] for row in rows])
        self.assertEqual("done", self.steps(run, "kits")["status"])
        self.assertIn("report beside", self.log(r))

    def test_the_next_job_runs_while_the_report_does(self):
        self.chain(groups="import time; time.sleep(1.5); open('groups-ran', 'w')")
        r = self.runner("--report-beside", "on")
        r.execute(r.new_run("fights", "abc"))
        self.assertIsNotNone(r.report)
        second = r.new_run("fights", "abc")
        r.execute(second)  # a whole job, while the first one's groups sleeps
        self.assertEqual((second.dir / "done.rc").read_text().strip(), "0")
        self.assertFalse((r.lab.root / "groups-ran").exists())  # the report is still on its first command
        self.assertEqual([], self.yield_rows(r))
        self.wait_report(r)
        self.assertEqual(2, len(self.yield_rows(r)))  # the second job's report followed the first

    def test_a_failed_groups_stops_the_chain_and_the_row_has_no_group_counts(self):
        self.chain(groups="import sys; sys.exit(1)")
        r = self.runner("--report-beside", "on")
        r.execute(r.new_run("fights", "abc"))
        self.wait_report(r)
        self.assertFalse((r.lab.root / "kits-ran").exists())
        row = self.yield_rows(r)[0]
        self.assertEqual(["NA", "NA", "NA"], row[10:13])

    def test_a_guard_stops_the_report_and_the_job_waits_for_the_next_one(self):
        self.chain(groups="import time; time.sleep(60)")
        r = self.runner("--report-beside", "on")
        run = r.new_run("fights", "abc")
        r.execute(run)
        self.assertIsNotNone(r.report)
        proc = r.report.running.proc
        r.kill_report("the owner is at the keyboard")
        self.assertIsNone(r.report)
        self.assertIsNotNone(proc.poll())
        self.assertEqual([run.name], [x.name for x, _ in r.report_due])
        self.assertEqual([], self.yield_rows(r))
        self.assertEqual("pending", self.steps(run, "groups")["status"])
        r.guards_ok = False
        r.service_report()
        self.assertIsNone(r.report)  # not while a guard says stop
        r.guards_ok = True
        self.chain()
        r.service_report()
        self.assertIsNotNone(r.report)  # then it starts over
        self.wait_report(r)
        self.assertEqual(1, len(self.yield_rows(r)))

    def test_a_main_lock_does_not_stop_the_report(self):
        self.chain(groups="import time; time.sleep(0.5)")
        r = self.runner("--report-beside", "on")
        r.execute(r.new_run("fights", "abc"))
        (self.tmp / "bl").mkdir(exist_ok=True)
        (self.tmp / "bl" / "main.lock").write_text("pid=1\n")
        r.poll()
        self.assertIsNotNone(r.report)
        self.wait_report(r)
        self.assertEqual(1, len(self.yield_rows(r)))

    def test_a_nightly_window_keeps_the_report_from_starting(self):
        self.chain()
        r = self.runner("--report-beside", "on")
        (r.lab.root / "nightly").mkdir(parents=True)
        (r.lab.root / "nightly" / "active").write_text("1\n")
        r.report_due.append((r.new_run("fights", "abc"), 0))
        r.service_report()
        self.assertIsNone(r.report)
        (r.lab.root / "nightly" / "active").unlink()
        r.service_report()
        self.assertIsNotNone(r.report)
        self.wait_report(r)

    def test_a_nightly_job_stops_a_running_report(self):
        self.chain(groups="import time; time.sleep(60)")
        r = self.runner("--report-beside", "on")
        r.execute(r.new_run("fights", "abc"))
        proc = r.report.running.proc
        r.execute(rn.Run.create(r.lab, "nightly-real", "abc", {}))
        self.assertIsNotNone(proc.poll())
        self.assertIn("a nightly window opens", self.log(r))

    def test_a_presort_waits_for_the_report_to_end_before_it_moves_the_tree(self):
        self.chain(groups="import time; time.sleep(0.8); open('groups-ran', 'w')")
        r = self.runner("--report-beside", "on")
        r.execute(r.new_run("fights", "abc"))
        self.assertIsNotNone(r.report)
        r.execute(r.new_run("presort", "abc"))
        self.assertTrue((r.lab.root / "groups-ran").exists())  # groups ended before the presort's steps began
        self.assertIn("waiting for it to end", self.log(r))

    def test_a_fault_in_the_report_turns_it_off_and_keeps_the_rows(self):
        def boom(c):
            raise RuntimeError("no plan")
        pl.report_cmds = boom
        r = self.runner("--report-beside", "on")
        self.assertIsNone(r.execute(r.new_run("fights", "abc")))
        self.assertFalse(r.report_beside)
        self.assertEqual(1, len(self.yield_rows(r)))  # the job's row, without group counts
        self.assertIn("run as steps of each job", self.log(r))

    def test_a_failed_job_writes_its_row_at_once(self):
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [self.py("bad", "import sys; sys.exit(2)", core=True)], mode="serial", required=True)]
        self.chain()
        r = self.runner("--report-beside", "on")
        r.execute(r.new_run("fights", "abc"))
        self.assertEqual(1, len(self.yield_rows(r)))
        self.assertIsNone(r.report)

    def test_the_runner_stopping_writes_the_rows_that_wait(self):
        self.chain()
        r = self.runner("--report-beside", "on")
        (r.lab.root / "nightly").mkdir(parents=True)
        (r.lab.root / "nightly" / "active").write_text("1\n")  # a window is open: the report cannot start
        r.execute(r.new_run("fights", "abc"))
        self.assertIsNone(r.report)
        self.assertEqual(1, len(r.report_due))
        r.leave(rn.guards.Verdict("signal", "test", leave=True))
        self.assertEqual(1, len(self.yield_rows(r)))
        self.assertEqual([], r.report_due)


if __name__ == "__main__":
    unittest.main()
