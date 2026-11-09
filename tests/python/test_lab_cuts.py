"""The lab's cuts to wasted hours (D55 addendum 18, 2026-11-09; tools/lab/runner.py, alerts.py, guards.py, cores.py):

  * a job kind that fails twice in a row stops being picked and raises LAB/alerts/<kind>.txt; a job killed at its time limit is listed there
    even the first time; the daily readers print the alerts at the top
  * the groups and kits beside a job (the report) start only on an idle machine: not beside a judgement, a score or a nightly job, and not
    under the cores they need; a queued report never holds a pick
  * no practice block starts when a presort would be chosen within the hour
  * a running train job holds a `lab-train` line in QUIET.lock (refreshed, gone however the job ends) and the load guard does not pause it

Trivial `python -c` commands stand in for the jobs; a fake clock where time matters; no game, no emulator.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.lab import alerts as al  # noqa: E402
from tools.lab import cores  # noqa: E402
from tools.lab import plans as pl  # noqa: E402
from tools.lab import runner as rn  # noqa: E402
from tools.lab import winsys  # noqa: E402
from test_lab_filler import FillerBase  # noqa: E402
from test_lab_jobs import Base as JobsBase  # noqa: E402

FAIL_CODE = "print('line1'); print('line2'); print('line3'); print('line4'); print('line5'); import sys; sys.exit(2)"


def fake_clock():
    t = [1000.0]
    return (lambda: t[0]), (lambda dt: t.__setitem__(0, t[0] + dt))


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        saved = dict(pl.PLANS)
        self.addCleanup(lambda: (pl.PLANS.clear(), pl.PLANS.update(saved)))
        self.lab = self.tmp / "lab"
        self.lab.mkdir()
        (self.lab / "jobs.json").write_text(json.dumps({"cores": {"total": 16, "fights": 8, "train": 16, "presort": 4, "report": 2, "loop": 6}}))

    def runner(self, *extra: str, clock=None) -> rn.Runner:
        a = rn.parse_args(["--rotation", "all", "--reports", "all", "--lab", str(self.lab), "--tree", str(self.tmp / "tree"), "--game", str(self.tmp / "game" / "g"),
                           "--backlog-dir", str(self.tmp / "bl"), "--poll", "0.05", "--wait-poll", "0.05", "--grace", "0",
                           "--no-guard", "user,load,disk", "--loop-status", str(self.tmp / "no-loop"), "--quiet-lock", str(self.tmp / "QUIET.lock"),
                           *extra])
        r = rn.Runner(a, clock=clock) if clock else rn.Runner(a)
        r.lab.runs.mkdir(parents=True, exist_ok=True)
        return r

    def py(self, name: str, code: str, **kw) -> pl.Cmd:
        return pl.Cmd(name, ["{py}", "-c", code], cwd="{lab}", **kw)

    def log(self, r: rn.Runner) -> str:
        p = r.lab.root / "runner.log"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def failing(self, kind: str = "fights", code: str = FAIL_CODE) -> None:
        pl.PLANS[kind] = lambda c: [pl.Step("a", [self.py("bad", code, core=True)], mode="serial")]

    def good(self, kind: str = "fights") -> None:
        pl.PLANS[kind] = lambda c: [pl.Step("a", [self.py("ok", "print('fine')", core=True)], mode="serial")]


class AlertTests(Case):
    def test_two_failures_in_a_row_raise_the_alert_and_stop_the_picks(self):
        self.failing()
        r = self.runner()
        first = r.new_run("fights", "abc")
        r.execute(first)
        self.assertFalse(al.alert_path(r.lab.root, "fights").exists())  # one failure is one failure
        self.assertIsNone(r.unavailable("fights", "abc"))
        second = r.new_run("fights", "abc")
        r.execute(second)
        path = al.alert_path(r.lab.root, "fights")
        self.assertEqual(path, r.lab.root / "alerts" / "fights.txt")
        text = path.read_text(encoding="utf-8")
        self.assertIn("ALERT fights: 2 failures in a row", text)
        for run in (first, second):
            self.assertIn(f"job runs/{run.name}: rc=1 duration=", text)  # the two job folders, their rc and duration
        self.assertEqual(text.count("bad log, last 3 line(s):"), 2)
        for tail in ("line3", "line4", "line5"):
            self.assertEqual(text.count(f"    {tail}"), 2, tail)  # the last three lines of each log ...
        self.assertNotIn("line2", text)  # ... and no more
        self.assertIn("Remove this file", text)
        why = r.unavailable("fights", "abc")
        self.assertIn("stopped after repeated failures", why)  # the picks stop
        self.assertIn("LAB/alerts/fights.txt", why)
        self.assertGreater(r.state["backoff"]["fights"], time.time() + 365 * 24 * 3600)  # the backoff has no end
        for _ in range(3):
            r.unavailable("fights", "abc")
        self.assertEqual(self.log(r).count("ALERT fights:"), 1)  # said once, however often the pick looks
        self.assertIn("no longer picked until LAB/alerts/fights.txt is removed by hand", self.log(r))

    def test_a_fights_strategy_is_stopped_alone(self):
        self.failing()
        r = self.runner()
        for _ in range(2):
            r.next_strategy = "shrink"
            r.execute(r.new_run("fights", "abc"))
        self.assertTrue(al.alert_path(r.lab.root, "shrink").exists())
        self.assertFalse(al.alert_path(r.lab.root, "fights").exists())
        self.assertIn("stopped after repeated failures", r.unavailable("fights", "abc", "shrink"))
        self.assertIsNone(r.unavailable("fights", "abc", "dense"))  # the other strategies go on

    def test_a_success_in_between_resets_the_count(self):
        r = self.runner()
        self.failing()
        r.execute(r.new_run("fights", "abc"))
        self.good()
        r.execute(r.new_run("fights", "abc"))
        self.assertEqual(r.state["fail_streak"]["fights"], 0)
        self.failing()
        r.execute(r.new_run("fights", "abc"))
        self.assertFalse(al.alert_path(r.lab.root, "fights").exists())  # fail, ok, fail: no two in a row
        self.assertIsNone(r.unavailable("fights", "abc"))
        r.execute(r.new_run("fights", "abc"))
        self.assertTrue(al.alert_path(r.lab.root, "fights").exists())  # the next failure is the second in a row

    def test_removing_the_file_by_hand_lets_the_kind_run_again_with_the_count_started_over(self):
        self.failing()
        r = self.runner()
        r.execute(r.new_run("fights", "abc"))
        r.execute(r.new_run("fights", "abc"))
        self.assertIn("stopped", r.unavailable("fights", "abc"))
        al.alert_path(r.lab.root, "fights").unlink()
        self.assertIsNone(r.unavailable("fights", "abc"))
        self.assertIn("alert fights.txt removed: fights is picked again", self.log(r))
        self.assertNotIn("fights", r.state["backoff"])
        r.execute(r.new_run("fights", "abc"))
        self.assertFalse(al.alert_path(r.lab.root, "fights").exists())  # one failure again, not two
        self.assertIsNone(r.unavailable("fights", "abc"))

    def test_a_file_the_state_does_not_know_holds_the_kind_all_the_same(self):
        r = self.runner()
        al.alert_path(r.lab.root, "score-real").parent.mkdir(parents=True)
        al.alert_path(r.lab.root, "score-real").write_text("by hand\n")
        self.assertIn("stopped after repeated failures", r.unavailable("score-real", "abc"))

    def test_a_presort_raises_the_alert_without_holding_its_own_retry_rule(self):
        pl.PLANS["presort"] = lambda c: [pl.Step("a", [self.py("bad", FAIL_CODE, core=True)], mode="serial", required=True)]
        r = self.runner()
        for _ in range(2):
            r.execute(r.new_run("presort", "deadbeef" * 5))
        text = al.alert_path(r.lab.root, "presort").read_text(encoding="utf-8")
        self.assertIn("ALERT presort: 2 failures in a row", text)
        self.assertIn("still picks presort by its own retry rule", text)
        self.assertNotIn("presort", r.state.get("alerted", {}))
        self.assertNotIn("presort", r.state["backoff"])

    def test_a_job_killed_at_its_time_limit_is_listed_the_first_time(self):
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [self.py("slow", "print('working', flush=True); import time; time.sleep(30)", core=True, timeout=1.0)],
                                               mode="serial")]
        r = self.runner()
        run = r.new_run("fights", "abc")
        r.execute(run)
        listed = (r.lab.root / "alerts" / "timeouts.txt").read_text(encoding="utf-8")
        self.assertIn(f"fights runs/{run.name} slow: killed at its 1s limit (ran ", listed)
        self.assertIn("working", listed)  # the log's last lines are beside it
        self.assertEqual(self.log(r).count("ALERT timeout: fights/slow"), 1)
        self.assertFalse(al.alert_path(r.lab.root, "fights").exists())  # one failure: listed, not stopped
        self.assertIsNone(r.unavailable("fights", "abc"))

    def test_the_daily_readers_print_the_alerts_at_the_top(self):
        self.failing()
        r = self.runner()
        summary = r.lab.root / "summary.md"
        summary.write_text("# Fix lab groups\n\nmain abc\n", encoding="utf-8")
        r.status("idle")
        self.assertTrue((r.lab.root / "status.txt").read_text().startswith("20"))  # no alert: the line as it was
        r.execute(r.new_run("fights", "abc"))
        r.execute(r.new_run("fights", "abc"))
        r.status("idle")
        status = (r.lab.root / "status.txt").read_text(encoding="utf-8")
        self.assertTrue(status.startswith("ALERT fights (LAB/alerts) | 20"), status)
        text = summary.read_text(encoding="utf-8")  # raise_alert refreshed the banner at once
        self.assertTrue(text.startswith(al.MARK_OPEN + "\n# ALERTS"), text)
        self.assertIn("- fights.txt: ALERT fights: 2 failures in a row", text)
        self.assertTrue(text.endswith("# Fix lab groups\n\nmain abc\n"))
        self.assertEqual(text.count(al.MARK_OPEN), 1)
        r.refresh_banners()
        self.assertEqual(summary.read_text(encoding="utf-8"), text)  # idempotent
        al.alert_path(r.lab.root, "fights").unlink()
        r.unavailable("fights", "abc")  # the lifting refreshes the banner too
        self.assertEqual(summary.read_text(encoding="utf-8"), "# Fix lab groups\n\nmain abc\n")

    def test_a_groups_run_writes_the_banner_into_its_summary(self):
        lab = self.tmp / "lab"
        al.write_timeout(lab, "fights", "20261109T0100-fights", "slow", 600, 601)
        got = al.with_banner("# Fix lab groups\n", lab)
        self.assertTrue(got.startswith(al.MARK_OPEN))
        self.assertEqual(al.strip_banner(got), "# Fix lab groups\n")
        self.assertEqual(al.with_banner(got, lab), got)  # a banner is replaced, never doubled

    def test_user_paths_never_reach_an_alert(self):
        self.failing(code="print('see C:/' + 'Users' + '/someone/AppData/x.log'); import sys; sys.exit(2)")  # (built from parts: the landing check greps diffs for it)
        r = self.runner()
        r.execute(r.new_run("fights", "abc"))
        r.execute(r.new_run("fights", "abc"))
        text = al.alert_path(r.lab.root, "fights").read_text(encoding="utf-8")
        self.assertNotIn("C:/" + "Users", text)  # (the command's own header line is the source text; the output line is what is scrubbed)
        self.assertIn("see <home>/AppData/x.log", text)


class ReportHoldTests(Case):
    def setUp(self):
        super().setUp()
        saved = pl.report_cmds
        self.addCleanup(lambda: setattr(pl, "report_cmds", saved))
        pl.report_cmds = lambda c: [self.py("groups", "open('groups-ran', 'w')"), self.py("kits", "open('kits-ran', 'w')")]

    def test_the_report_does_not_start_beside_a_judgement_a_score_or_a_nightly_job(self):
        r = self.runner("--report-beside", "on")
        self.good()
        run = r.new_run("fights", "abc")
        for job in ("train", "score-real", "nightly-real"):
            r.running_job = job  # a heavy job is the one running; a finished fights job's report is due
            r.report_due = [(run, 0)]
            for _ in range(4):
                r.service_report()
            self.assertIsNone(r.report, job)
            self.assertEqual(len(r.report_due), 1, job)  # queued, not dropped
            if job != "nightly-real":  # (a nightly window is its own rule, as before)
                held = f"report beside: held (a {job} job is running); 1 job(s) wait, tried again at the next pick"
                self.assertEqual(self.log(r).count(held), 1, job)  # said once, not at every pick
            r.running_job = None
            r.service_report()
            self.assertIsNotNone(r.report, job)  # and tried again once the machine is idle
            r.kill_report("test end")

    def test_the_report_waits_for_the_cores_it_needs(self):
        r = self.runner("--report-beside", "on")
        self.good()
        pl.report_cmds = lambda c: [self.py("groups", "pass")]
        run = r.new_run("fights", "abc")
        r.report_due = [(run, 0)]
        r.main_cores = 15  # 16 cores, a job holding 15: one is free and the report needs two
        self.assertEqual(r.report_hold(), "1 cores free, the report needs 2")
        r.service_report()
        self.assertIsNone(r.report)
        self.assertEqual(self.log(r).count("1 cores free, the report needs 2"), 1)
        r.service_report()
        self.assertEqual(self.log(r).count("1 cores free, the report needs 2"), 1)
        r.main_cores = 14
        self.assertIsNone(r.report_hold())
        r.service_report()
        self.assertIsNotNone(r.report)
        r.kill_report("test end")

    def test_the_loops_cores_count_while_it_steps(self):
        r = self.runner("--report-beside", "on")
        status = self.tmp / "loop-status"
        status.write_text(json.dumps({"pid": os.getpid(), "phase": "generate"}))
        r.a.loop_status = str(status)
        r.report_due = [(object(), 0)]  # (only its length is read)
        r.main_cores = 9  # 16 - 9 - the loop's 6 = 1
        self.assertEqual(r.report_hold(), "1 cores free, the report needs 2")
        r.main_cores = 8
        self.assertIsNone(r.report_hold())
        status.write_text(json.dumps({"pid": os.getpid(), "phase": "waiting"}))  # the loop is between steps: its cores are free
        r.main_cores = 14
        self.assertIsNone(r.report_hold())

    def test_a_queued_report_never_holds_a_pick(self):
        self.good()
        pl.PLANS["presort"] = lambda c: [pl.Step("a", [self.py("p", "pass", core=True)], mode="serial")]
        r = self.runner("--report-beside", "on")
        pl.report_cmds = lambda c: [self.py("groups", "pass")]
        run = r.new_run("fights", "abc")
        r.report_due = [(run, 0)]
        r.running_job = "train"  # the report is held ...
        self.assertIsNotNone(r.report_hold())
        r.running_job = None
        job, _, why = r.choose("newmain")  # ... and the pick does not look at it: a presort is chosen because main moved
        self.assertEqual(job, "presort", why)
        pre = r.new_run("presort", "newmain")
        self.assertIsNone(r.execute(pre))  # and a presort runs with the report queued
        self.assertEqual((pre.dir / "done.rc").read_text().strip(), "0")

    def test_a_report_running_when_a_judgement_starts_gives_way(self):
        self.good()
        pl.report_cmds = lambda c: [self.py("groups", "import time; time.sleep(60)")]
        pl.PLANS["train"] = lambda c: [pl.Step("a", [self.py("t", "pass", core=True)], mode="serial")]
        r = self.runner("--report-beside", "on")
        r.execute(r.new_run("fights", "abc"))
        self.assertIsNotNone(r.report)
        proc = r.report.running.proc
        self.assertIsNone(r.execute(r.new_run("train", "abc")))
        self.assertIsNone(r.report)
        self.assertIsNotNone(proc.poll())
        self.assertEqual(len(r.report_due), 1)  # its job waits for the next report
        self.assertIn("report beside: stopped (a train job starts)", self.log(r))


@unittest.skipUnless(winsys.IS_WIN, "Job Objects and priority classes are Windows")
class FillerPresortTests(FillerBase):
    """No practice block starts when a presort would be chosen within the hour."""

    def held(self, r: rn.Runner) -> None:
        r.service_filler()
        self.assertIsNone(r.filler)

    def test_a_presort_due_within_the_hour_holds_the_start_and_the_log_says_so_once(self):
        r = self.runner()
        r.main_sha = lambda: "feed" * 10  # main has moved past the presorted tree ...
        r.state["last_presort_end"] = time.time() - 2 * 3600  # ... and the last presort ended two hours ago
        for _ in range(4):
            self.held(r)
        self.assertEqual(self.log(r).count("filler: not started (a presort of feedfee is due within 60 min (main moved past the presorted tree))"), 1)

    def test_a_presort_that_will_be_due_by_then_holds_it_too(self):
        r = self.runner()
        r.main_sha = lambda: "feed" * 10
        r.state["last_presort_end"] = time.time() - 20 * 60  # the gap (60 min) ends in 40 min: due within the hour
        self.held(r)
        self.assertIn("is due within 60 min", self.log(r))

    def test_a_presort_further_off_does_not_hold_it(self):
        r = self.runner("--presort-gap-min", "180")
        r.main_sha = lambda: "feed" * 10
        r.state["last_presort_end"] = time.time() - 20 * 60  # due in 160 min
        self.started(r)
        self.assertIsNotNone(r.filler)

    def test_main_not_moved_means_no_presort_and_the_filler_starts(self):
        r = self.runner()
        r.state["last_presort_end"] = time.time() - 5 * 3600
        self.started(r)  # (the base runner's main is the presorted tree)
        self.assertNotIn("presort of", self.log(r))

    def test_a_build_that_failed_waits_for_its_retry_before_it_counts_as_due(self):
        r = self.runner()
        r.main_sha = lambda: "feed" * 10
        r.state["last_presort_end"] = time.time() - 5 * 3600
        r.state["bad_shas"]["feed" * 10] = {"fails": 2, "retry_after": time.time() + 5 * 3600}
        self.started(r)  # the presort will not be chosen for five hours: the machine is the filler's meanwhile

    def test_the_filler_still_yields_to_the_lab_train_line(self):
        q = self.tmp / "QUIET.lock"
        r = self.runner("--quiet-lock", str(q))
        q.write_text(f"lab-train {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} pid={os.getpid()}\n")
        self.held(r)
        self.assertIn("QUIET.lock is held by lab-train", self.log(r))
        self.assertNotIn("lab-train", cores.DEFAULT_QUIET_IGNORE)  # the practice and the report give way to the judgement, like the loop's others
        q.write_text("gym-loop 2026-11-09T00:00:00Z pid=1\n")  # the loop's own line never held it
        self.assertIsNone(cores.quiet_blocker(q, {}))


class TrainLineTests(JobsBase):
    """A running train job holds `lab-train` in QUIET.lock; the load guard does not pause it; everything else still yields to load."""

    def setUp(self):
        super().setUp()
        self.quiet = self.tmp / "QUIET.lock"
        self.seen = self.tmp / "seen.txt"

    def train_runner(self, *extra: str, clock=None, code: str | None = None, guards_on: tuple[str, ...] = ()) -> rn.Runner:
        code = code or f"import shutil; shutil.copy({str(self.quiet)!r}, {str(self.seen)!r})"
        self.config(train={"cmd": ["{python}", "-c", code]})
        self.marker("train/ready-1")
        r = self.runner(guards_on, "--quiet-lock", str(self.quiet), "--loop-status", str(self.tmp / "no-loop"), *extra)
        if clock:
            r.clock = clock
        return r

    def now(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def test_the_line_appears_with_the_job_and_is_gone_after_it(self):
        self.quiet.write_text(f"gym-loop {self.now()} pid=1\n")
        r = self.train_runner()
        run = r.new_run("train", "abc")
        self.assertIsNone(r.execute(run))
        seen = self.seen.read_text(encoding="utf-8").splitlines()  # what the job saw while it ran
        mine = [x for x in seen if x.startswith("lab-train ")]
        self.assertEqual(len(mine), 1, seen)
        self.assertTrue(mine[0].endswith(f"pid={os.getpid()}"), mine)  # the runner's Windows pid
        self.assertTrue(any(x.startswith("gym-loop ") for x in seen))  # others' lines are kept
        left = self.quiet.read_text(encoding="utf-8").splitlines()
        self.assertEqual([x.split()[0] for x in left], ["gym-loop"])  # gone with the job
        self.assertIn("QUIET.lock: lab-train line written", (self.lab_dir / "runner.log").read_text(encoding="utf-8"))
        self.assertIn("QUIET.lock: lab-train line removed", (self.lab_dir / "runner.log").read_text(encoding="utf-8"))

    def test_other_jobs_hold_no_line(self):
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [pl.Cmd("f", ["{py}", "-c", f"import shutil; shutil.copy({str(self.quiet)!r}, {str(self.seen)!r})"],
                                                              cwd="{lab}", core=True)], mode="serial")]
        self.quiet.write_text(f"gym-loop {self.now()} pid=1\n")
        r = self.train_runner()
        r.execute(r.new_run("fights", "abc"))
        self.assertNotIn("lab-train", self.seen.read_text(encoding="utf-8"))

    def test_the_line_is_refreshed_while_the_job_runs(self):
        clock, advance = fake_clock()
        r = self.train_runner(clock=clock)
        r.take_train_line()
        beats = []
        with mock.patch.object(r.train_lock, "announce", side_effect=lambda: beats.append(clock())):
            r.service_train_line()
            advance(rn.TRAIN_BEAT_S - 1)
            r.service_train_line()
            self.assertEqual(beats, [])  # not yet
            advance(2)
            r.service_train_line()
            self.assertEqual(len(beats), 1)
            r.service_train_line()
            self.assertEqual(len(beats), 1)  # once per period
            advance(rn.TRAIN_BEAT_S + 1)
            r.service_train_line()
            self.assertEqual(len(beats), 2)
        self.assertLessEqual(rn.TRAIN_BEAT_S, 30 * 60)  # the brief: at least every 30 minutes (the loop reads a line over 2 h old as dead)
        r.drop_train_line()

    def test_the_line_is_rewritten_with_a_fresh_time(self):
        r = self.train_runner()
        r.take_train_line()
        first = self.quiet.read_text(encoding="utf-8").splitlines()[0]
        old = "2026-11-09T00:00:00Z"
        self.quiet.write_text(first.replace(first.split()[1], old) + "\n", encoding="utf-8")  # the line has aged
        r._train_beat = -1e9
        r.service_train_line()
        again = self.quiet.read_text(encoding="utf-8").splitlines()[0]
        self.assertNotEqual(again.split()[1], old)
        self.assertEqual(again.split()[0], "lab-train")
        r.drop_train_line()

    def test_the_line_is_gone_after_a_simulated_kill(self):
        r = self.train_runner(code="import time; time.sleep(60)")
        run = r.new_run("train", "abc")
        threading.Timer(1.0, lambda: setattr(r, "stop_signal", "signal test")).start()  # the runner is told to stop while the job runs
        self.assertTrue(self.wait_for(lambda: True))
        v = r.execute(run)
        self.assertIsNotNone(v)  # the job was stopped, not finished
        self.assertFalse((run.dir / "done.rc").exists())
        self.assertFalse(self.quiet.exists() and "lab-train" in self.quiet.read_text(encoding="utf-8"), "the line outlived the job")

    def test_the_line_is_gone_after_a_crash_in_the_job_and_after_the_runner_leaves(self):
        r = self.train_runner()
        with mock.patch.object(rn.Runner, "_execute", side_effect=RuntimeError("boom")), self.assertRaises(RuntimeError):
            r.execute(r.new_run("train", "abc"))
        self.assertFalse(self.quiet.exists() and "lab-train" in self.quiet.read_text(encoding="utf-8"))
        r.take_train_line()
        r.leave(rn.guards.Verdict("signal", "test", leave=True))
        self.assertFalse(self.quiet.exists() and "lab-train" in self.quiet.read_text(encoding="utf-8"))

    def wait_for(self, cond, seconds: float = 1.0) -> bool:
        end = time.time() + seconds
        while time.time() < end:
            if cond():
                return True
            time.sleep(0.01)
        return False

    def busy(self, r: rn.Runner) -> None:
        class Busy:
            def sample(self):
                return None

            def mean(self):
                return (0.9, 120.0)

            def sustained(self, th):
                return True
        r.guards.load = Busy()

    def test_the_load_guard_does_not_pause_the_judgement_that_holds_the_line(self):
        r = self.train_runner(guards_on=("load",))
        self.busy(r)
        self.assertEqual(r.poll().kind, "load")  # with no line held (and for any other job) the guard stops as before
        r.running_job = "fights"
        self.assertEqual(r.poll().kind, "load")
        r.take_train_line()
        r.running_job = "train"
        self.assertTrue(r.poll().ok)  # the judgement goes on
        self.assertFalse(r.guards_ok)  # ... while the filler and the report still give way to the load
        self.assertEqual(self.log_text().count("load guard: other processes 90% of the CPU over 120s; the train job holds the lab-train line"), 1)
        r.poll()
        self.assertEqual(self.log_text().count("the train job holds the lab-train line and goes on"), 1)  # said once
        r.drop_train_line()
        self.assertEqual(r.poll().kind, "load")  # the line gone, the guard applies again

    def log_text(self) -> str:
        return (self.lab_dir / "runner.log").read_text(encoding="utf-8")

    def test_a_train_job_runs_to_its_end_under_load_and_counts_no_interrupt(self):
        r = self.train_runner(guards_on=("load",))
        self.busy(r)
        run = r.new_run("train", "abc")
        self.assertIsNone(r.execute(run))
        self.assertEqual((run.dir / "done.rc").read_text().strip(), "0")
        self.assertNotIn("load", r.state["counters"]["interrupts"])

    def test_the_load_still_stops_the_report_and_the_filler_beside_a_judgement(self):
        saved = pl.report_cmds
        self.addCleanup(lambda: setattr(pl, "report_cmds", saved))
        pl.report_cmds = lambda c: [pl.Cmd("groups", ["{py}", "-c", "import time; time.sleep(60)"], cwd="{lab}")]
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [pl.Cmd("f", ["{py}", "-c", "pass"], cwd="{lab}", core=True)], mode="serial")]
        r = self.train_runner("--report-beside", "on", guards_on=("load",))
        r.execute(r.new_run("fights", "abc"))  # a report is running (no load yet)
        self.assertIsNotNone(r.report)
        self.busy(r)
        r.take_train_line()
        r.running_job = "train"
        self.assertTrue(r.poll().ok)
        self.assertIsNone(r.report)  # the report gave way
        self.assertIn("report beside: stopped", self.log_text())
        r.drop_train_line()

    def test_a_due_presort_is_not_held_by_the_line(self):
        r = self.train_runner()
        self.quiet.write_text(f"lab-train {self.now()} pid={os.getpid()}\n")
        r.state["tree_sha"] = r.state["last_presorted_sha"] = "abc"
        pl.PLANS["presort"] = lambda c: [pl.Step("a", [pl.Cmd("p", ["{py}", "-c", "pass"], cwd="{lab}", core=True)], mode="serial")]
        job, _, why = r.choose("newmain")
        self.assertEqual(job, "presort", why)  # the runner's own picks never read QUIET.lock


if __name__ == "__main__":
    unittest.main()
