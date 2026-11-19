"""The lab keeps its own books (2026-11-19, lane 6a of the Act 3 week; the 2026-11-18 review of the machine, change 1):

  * tools/lab/labrun.py: heavy work started by hand runs at BelowNormal in a Job Object and leaves one line in LAB/hand-jobs.jsonl; `--report`
    prints core-hours by tag, purpose and consumer for the runner's jobs and the hand jobs together
  * a train marker may carry the keep rule its candidate was built for; the train job judges under the newer of that and LAB/jobs.json's, logs both,
    and the verdict (promotion.json, the ledger's guard cell) says which it used and why
  * the three silent failures raise alerts: more than two failures of a kind in a day, two time-limit kills in a week, a dead holder's QUIET.lock line

Trivial `python -c` commands stand in for the work; the arms are stubbed; no game, no emulator.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, ROOT / "tests" / "python"):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import gen0  # noqa: E402
import gen_loop  # noqa: E402
import test_funnel as tf  # noqa: E402
import test_gen12_plumbing as tg12  # noqa: E402
from test_lab_cuts import FAIL_CODE, Case  # noqa: E402
from tools.lab import alerts as al  # noqa: E402
from tools.lab import jobs  # noqa: E402
from tools.lab import labrun as lr  # noqa: E402
from tools.lab import plans as pl  # noqa: E402
from tools.lab import winsys  # noqa: E402

PY = sys.executable
UTC = "%Y-%m-%dT%H:%M:%SZ"


def utc(t: float) -> str:
    return time.strftime(UTC, time.gmtime(t))


def spin(seconds: float) -> str:
    """A busy loop that burns `seconds` of CPU time (not wall time: a loaded machine may run it slowly and the CPU figure is what is asserted)."""
    return f"import time; t=time.process_time()\nwhile time.process_time()-t<{seconds}: pass"


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.lab = self.tmp / "LAB"
        self.lab.mkdir()
        self.qpath = self.tmp / "QUIET.lock"


class LabrunTests(Tmp):
    def go(self, argv, **kw):
        kw.setdefault("tag", "t1")
        kw.setdefault("purpose", "check the wrapper")
        kw.setdefault("consumer", "the test")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            rc, rec = lr.run_command(argv, lab=self.lab, quiet_lock_path=self.qpath, **kw)
        self.err = err.getvalue()
        return rc, rec

    def lines(self) -> list[dict]:
        p = self.lab / lr.HAND_JOBS
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []

    def test_one_line_with_the_command_s_time_cpu_rc_tag_purpose_and_consumer(self):
        rc, rec = self.go([PY, "-I", "-c", spin(0.4)])
        self.assertEqual(0, rc)
        self.assertEqual([rec], self.lines())
        for key in ("tag", "purpose", "consumer", "start", "end", "wall_s", "cpu_s", "rc", "quiet", "cores_used", "core_hours", "cmd"):
            self.assertIn(key, rec, key)
        self.assertEqual(("t1", "check the wrapper", "the test", 0, False), (rec["tag"], rec["purpose"], rec["consumer"], rec["rc"], rec["quiet"]))
        self.assertGreaterEqual(rec["wall_s"], 0.35)
        self.assertGreaterEqual(rec["cpu_s"], 0.3)  # a busy loop: one core for its wall time
        self.assertAlmostEqual(rec["core_hours"], rec["cpu_s"] / 3600, places=3)
        self.assertRegex(rec["start"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertIn("t1 rc=0", self.err)

    def test_the_cpu_is_the_whole_trees_not_the_first_process(self):
        code = ("import subprocess, sys\n"
                f"subprocess.run([sys.executable, '-I', '-c', {spin(0.5)!r}])\n"
                f"subprocess.run([sys.executable, '-I', '-c', {spin(0.5)!r}])\n")
        rc, rec = self.go([PY, "-I", "-c", code])
        self.assertEqual(0, rc)
        self.assertGreaterEqual(rec["cpu_s"], 0.8)  # two children, half a second each, and the parent did nothing

    @unittest.skipUnless(winsys.IS_WIN, "a Windows Job Object")
    def test_the_command_runs_at_below_normal_and_so_do_its_children(self):
        out = self.tmp / "prio.txt"
        script = self.tmp / "prio.py"
        child = f"import sys; sys.path.insert(0, {str(ROOT)!r}); from tools.lab import winsys; print(winsys.priority_class())"
        script.write_text("import subprocess, sys\n"
                          f"sys.path.insert(0, {str(ROOT)!r})\n"
                          "from tools.lab import winsys\n"
                          f"child = {child!r}\n"
                          "kid = subprocess.run([sys.executable, '-I', '-c', child], capture_output=True, text=True).stdout.strip()\n"
                          f"open({str(out)!r}, 'w').write(str(winsys.priority_class()) + ' ' + kid)\n", encoding="utf-8")
        rc, _ = self.go([PY, str(script)])
        self.assertEqual(0, rc)
        parent, kid = out.read_text().split()
        self.assertEqual((str(winsys.BELOW_NORMAL_PRIORITY_CLASS), str(winsys.BELOW_NORMAL_PRIORITY_CLASS)), (parent, kid))

    def test_the_commands_exit_code_is_labruns_and_is_recorded(self):
        rc, rec = self.go([PY, "-I", "-c", "import sys; sys.exit(3)"])
        self.assertEqual((3, 3), (rc, rec["rc"]))

    def test_a_command_that_cannot_start_is_127_and_still_written_down(self):
        rc, rec = self.go([str(self.tmp / "no-such-program.exe")])
        self.assertEqual((lr.NOT_STARTED, lr.NOT_STARTED), (rc, rec["rc"]))
        self.assertIn("could not start", rec["note"])
        self.assertEqual(1, len(self.lines()))

    def test_a_user_path_in_the_command_is_made_anonymous(self):
        rc, rec = self.go([PY, "-I", "-c", "pass", "C:/" + "Users" + "/someone/data.json"])
        self.assertNotIn("someone", rec["cmd"])
        self.assertIn("<home>", rec["cmd"])

    def test_the_tag_is_one_word(self):
        rc, rec = self.go([PY, "-I", "-c", "pass"], tag="two words")
        self.assertEqual((lr.REFUSED, None), (rc, rec))
        self.assertEqual([], self.lines())

    def test_a_declared_core_count_is_recorded_and_exceeding_it_is_said(self):
        code = ("import subprocess, sys\n"
                "ps = [subprocess.Popen([sys.executable, '-I', '-c', " + repr(spin(0.6)) + "]) for _ in range(3)]\n"
                "[p.wait() for p in ps]\n")
        rc, rec = self.go([PY, "-I", "-c", code], cores_declared=1)
        self.assertEqual(1, rec["cores_declared"])
        if rec["cores_used"] > 1.25:  # a loaded machine may not run three at once
            self.assertIn("declared", self.err)

    # ---- --quiet

    def test_quiet_holds_its_own_line_while_the_command_runs_and_removes_it_after(self):
        seen = self.tmp / "seen.txt"
        self.qpath.write_text(f"gym-loop {utc(time.time())} pid=1234\n")  # the always-on loop's line neither blocks nor is touched
        code = f"open({str(seen)!r}, 'w').write(open({str(self.qpath)!r}).read())"
        rc, rec = self.go([PY, "-I", "-c", code], tag="q1", quiet=True)
        self.assertEqual(0, rc)
        during = seen.read_text().splitlines()
        mine = [x for x in during if x.startswith("q1 ")]
        self.assertEqual(1, len(mine), during)
        tag, when, pid = mine[0].split()
        self.assertRegex(when, r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertRegex(pid, r"^pid=\d+$")
        self.assertTrue(winsys.parent_alive(int(pid[4:])) if winsys.IS_WIN else True)  # a Windows process id that was alive: the wrapper's
        self.assertTrue(any(x.startswith("gym-loop ") for x in during))
        after = self.qpath.read_text().splitlines()
        self.assertEqual(1, len(after))
        self.assertTrue(after[0].startswith("gym-loop "))  # ours gone, the loop's kept
        self.assertTrue(rec["quiet"])

    def test_quiet_leaves_no_file_behind_when_it_held_the_only_line(self):
        rc, _ = self.go([PY, "-I", "-c", "pass"], tag="q1", quiet=True)
        self.assertFalse(self.qpath.exists())

    def test_the_line_goes_even_when_the_command_fails(self):
        rc, _ = self.go([PY, "-I", "-c", "import sys; sys.exit(9)"], tag="q1", quiet=True)
        self.assertEqual(9, rc)
        self.assertFalse(self.qpath.exists())

    def test_quiet_is_refused_at_once_while_another_holders_line_stands(self):
        self.qpath.write_text(f"w3-blocks {utc(time.time() - 600)} pid=1234\n")
        rc, rec = self.go([PY, "-I", "-c", "open(" + repr(str(self.tmp / "ran.txt")) + ", 'w')"], tag="q1", quiet=True)
        self.assertEqual((lr.REFUSED, None), (rc, rec))
        self.assertFalse((self.tmp / "ran.txt").exists())  # nothing started
        self.assertEqual([], self.lines())
        self.assertIn("w3-blocks", self.err)
        self.assertTrue(self.qpath.read_text().startswith("w3-blocks"))

    def test_a_line_over_two_hours_old_does_not_block(self):
        self.qpath.write_text(f"w3-blocks {utc(time.time() - 3 * 3600)} pid=1234\n")
        rc, rec = self.go([PY, "-I", "-c", "pass"], tag="q1", quiet=True)
        self.assertEqual(0, rc)
        self.assertFalse(self.qpath.exists())  # the stale line is dropped when the file is rewritten, ours removed

    def test_a_plain_run_does_not_look_at_the_lock(self):
        self.qpath.write_text(f"w3-blocks {utc(time.time())} pid=1234\n")
        rc, rec = self.go([PY, "-I", "-c", "pass"], quiet=False)
        self.assertEqual(0, rc)
        self.assertTrue(self.qpath.read_text().startswith("w3-blocks"))  # untouched

    # ---- the night

    def test_nothing_starts_while_a_nightly_block_runs(self):
        (self.lab / "nightly").mkdir()
        (self.lab / "nightly" / "active").write_text("x\n")
        rc, rec = self.go([PY, "-I", "-c", "pass"])
        self.assertEqual((lr.REFUSED, None), (rc, rec))
        self.assertIn("nightly", self.err)
        self.assertEqual([], self.lines())
        rc, rec = self.go([PY, "-I", "-c", "pass"], allow_night=True)
        self.assertEqual(0, rc)

    def test_nothing_starts_while_the_nightly_holds_the_quiet_lock(self):
        self.qpath.write_text(f"lab-nightly {utc(time.time())} pid=1234\n")
        rc, rec = self.go([PY, "-I", "-c", "pass"])
        self.assertEqual(lr.REFUSED, rc)

    # ---- the command line

    def test_main_runs_the_command_after_the_double_dash_and_returns_its_code(self):
        argv = ["--lab", str(self.lab), "--quiet-lock", str(self.qpath), "--tag", "m1", "--purpose", "p", "--consumer", "c", "--cores", "1",
                "--", PY, "-I", "-c", "import sys; sys.exit(4)"]
        with contextlib.redirect_stderr(io.StringIO()):
            rc = lr.main(argv)
        self.assertEqual(4, rc)
        self.assertEqual(["m1"], [x["tag"] for x in self.lines()])

    def test_main_needs_tag_purpose_consumer_and_a_command(self):
        for argv in (["--tag", "a", "--purpose", "p", "--", PY], ["--tag", "a", "--consumer", "c", "--", PY], ["--purpose", "p", "--consumer", "c", "--", PY],
                     ["--tag", "a", "--purpose", "p", "--consumer", "c"]):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                lr.main(["--lab", str(self.lab), *argv])
            self.assertEqual(2, cm.exception.code, argv)
        self.assertEqual([], self.lines())


class ReportTests(Tmp):
    RUNNER = "\n".join([
        "2026-11-01T10:00:00Z job 1: ladder: selfplay (priority 6)",
        "2026-11-01T11:00:00Z selfplay/selfplay: rc=0 in 3600.0s",  # before cpu= existed: estimated
        "2026-11-04T11:00:00Z selfplay/selfplay: rc=0 in 3600.0s cpu=43200.0s reads=1 read_mb=1.0 write_mb=1.0 out=?",  # 12 cores
        "2026-11-04T12:00:00Z selfplay/selfplay: rc=0 in 3600.0s cpu=43200.0s reads=1 read_mb=1.0 write_mb=1.0 out=?",
        "2026-11-05T01:00:00Z score-real/score-real: rc=1 in 7200.0s cpu=28800.0s reads=1 read_mb=1.0 write_mb=1.0 out=?",  # a failed four cores
        "2026-11-06T01:00:00Z teacher/teacher: rc=0 in 3600.0s cpu=36000.0s reads=1 read_mb=1.0 write_mb=1.0 out=?",
        "2026-11-06T02:00:00Z batch-03/analyze-03: rc=0 in 100.0s",
        "2026-11-06T03:00:00Z build/build-emudump: rc=0 in 60.0s cpu=30.0s",
        "garbage line",
    ]) + "\n"

    def setUp(self):
        super().setUp()
        (self.lab / "runner.log").write_text(self.RUNNER, encoding="utf-8")
        hand = [{"tag": "gate-a", "purpose": "land lane 6a", "consumer": "the checker", "end": "2026-11-06T04:00:00Z", "wall_s": 1800.0,
                 "cpu_s": 7200.0, "core_hours": 2.0, "rc": 0},
                {"tag": "gate-a", "purpose": "land lane 6a", "consumer": "the checker", "end": "2026-11-06T05:00:00Z", "wall_s": 600.0,
                 "cpu_s": 3600.0, "core_hours": 1.0, "rc": 1},
                {"tag": "probe", "purpose": "a speed probe", "consumer": "lane 4", "end": "2026-11-07T05:00:00Z", "wall_s": 3600.0, "rc": 0}]  # no cpu figure
        (self.lab / lr.HAND_JOBS).write_text("\n".join(json.dumps(h) for h in hand) + "\nnot json\n", encoding="utf-8")

    def rows(self, rep, table):
        return {(r["name"], r["source"]): r for r in rep[table]}

    def test_core_hours_by_tag_purpose_and_consumer_for_both_logs(self):
        rep = lr.build_report(self.lab, now=lr.parse_when("2026-11-08"))
        tags = self.rows(rep, "by_tag")
        self.assertAlmostEqual(36.0, tags[("selfplay", "lab")]["core_h"])  # two measured blocks of 12 core-hours and one estimated at the same 12
        self.assertEqual(3, tags[("selfplay", "lab")]["runs"])
        self.assertAlmostEqual(8.0, tags[("score-real", "lab")]["core_h"])
        self.assertEqual((1, 8.0), (tags[("score-real", "lab")]["failed"], tags[("score-real", "lab")]["failed_core_h"]))
        self.assertAlmostEqual(10.0, tags[("teacher", "lab")]["core_h"])
        self.assertAlmostEqual(3.0, tags[("gate-a", "hand")]["core_h"])
        self.assertEqual(1, tags[("gate-a", "hand")]["failed"])
        self.assertAlmostEqual(1.0, tags[("probe", "hand")]["core_h"])  # no CPU figure: one core for the wall hour
        self.assertIn(("batch", "lab"), tags)  # `batch-03` is the batch step
        purposes = self.rows(rep, "by_purpose")
        self.assertIn(("self-play practice (lab)", "lab"), purposes)
        self.assertIn(("land lane 6a", "hand"), purposes)
        self.assertIn(("fidelity and bug hunting", "lab"), purposes)
        self.assertIn(("builds (presort)", "lab"), purposes)
        consumers = self.rows(rep, "by_consumer")
        self.assertEqual({"the lab", "the checker", "lane 4"}, {n for n, _ in consumers})
        self.assertAlmostEqual(rep["hand_core_h"], 4.0)
        self.assertAlmostEqual(rep["logged_core_h"], sum(r["core_h"] for r in rep["by_tag"]), delta=0.1)

    def test_a_line_without_cpu_is_estimated_at_the_median_cores_measured_later_for_the_same_step(self):
        rep = lr.build_report(self.lab, since=lr.parse_when("2026-11-01"), until=lr.parse_when("2026-11-02"), now=lr.parse_when("2026-11-08"))
        row = self.rows(rep, "by_tag")[("selfplay", "lab")]
        self.assertAlmostEqual(12.0, row["core_h"])  # one wall hour at the 12 cores the two measured blocks used
        self.assertAlmostEqual(12.0, row["est_core_h"])
        self.assertAlmostEqual(12.0, rep["est_core_h"])

    def test_since_and_until_cut_by_the_end_of_each_step(self):
        rep = lr.build_report(self.lab, since=lr.parse_when("2026-11-06"), until=lr.parse_when("2026-11-07"), now=lr.parse_when("2026-11-08"))
        self.assertEqual({"teacher", "batch", "build", "gate-a"}, {r["name"] for r in rep["by_tag"]})

    def test_the_share_of_the_pcs_thread_hours_the_logs_account_for(self):
        rep = lr.build_report(self.lab, since=lr.parse_when("2026-11-04"), until=lr.parse_when("2026-11-05"), now=lr.parse_when("2026-11-08"))
        self.assertEqual(24.0, rep["window_h"])
        self.assertEqual(16 * 24, rep["thread_hours"])
        self.assertAlmostEqual(24.0 / (16 * 24), rep["share_logged"], delta=0.001)

    def test_the_text_and_the_json_say_the_same(self):
        text = lr.render(lr.build_report(self.lab, now=lr.parse_when("2026-11-08")))
        self.assertIn("By tag", text)
        self.assertIn("By purpose", text)
        self.assertIn("By consumer", text)
        self.assertIn("gate-a", text)
        self.assertIn("Not in either log", text)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(0, lr.main(["--report", "--lab", str(self.lab), "--since", "2026-11-01", "--json"]))
        self.assertEqual(3.0, {r["name"]: r for r in json.loads(out.getvalue())["by_tag"]}["gate-a"]["core_h"])

    def test_a_bad_date_is_refused(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            lr.main(["--report", "--lab", str(self.lab), "--since", "last week"])

    def test_no_logs_is_an_empty_report_not_an_error(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        rep = lr.build_report(empty, now=time.time())
        self.assertEqual(([], 0.0), (rep["by_tag"], rep["logged_core_h"]))
        self.assertIn("Lab books", lr.render(rep))

    def test_the_rolled_over_log_is_read_too(self):
        (self.lab / "runner.log.1").write_text("2026-10-30T10:00:00Z teacher/teacher: rc=0 in 3600.0s cpu=7200.0s\n", encoding="utf-8")
        rep = lr.build_report(self.lab, now=lr.parse_when("2026-11-08"))
        self.assertAlmostEqual(12.0, self.rows(rep, "by_tag")[("teacher", "lab")]["core_h"])  # the 2 of the older file and the 10 of the current one

    def test_the_live_log_formats_parse(self):
        line = "2026-11-18T17:06:04Z teacher/teacher: rc=0 in 3654.5s cpu=43269.2s reads=108849 read_mb=15669.0 write_mb=293.0 out=?"
        (self.lab / "runner.log").write_text(line + "\n", encoding="utf-8")
        got = lr.read_runner(self.lab)
        self.assertEqual([("teacher", "teacher", 0, 3654.5, 43269.2)], [(s["step"], s["cmd"], s["rc"], s["wall"], s["cpu"]) for s in got])


class MarkerTests(Tmp):
    def test_the_first_line_stays_the_time_and_a_later_line_carries_the_rule(self):
        text = jobs.marker_text("2026-11-19T01:00:00Z", "v3")
        self.assertEqual("2026-11-19T01:00:00Z\nkeep_rule: v3\n", text)
        p = self.tmp / "ready-5-g7"
        p.write_text(text)
        got = jobs.read_marker(p)
        self.assertEqual(("2026-11-19T01:00:00Z", "v3"), (got["utc"], got["keep_rule"]))
        self.assertEqual("2026-11-19T01:00:00Z\n", jobs.marker_text("2026-11-19T01:00:00Z"))  # no rule: the old marker, byte for byte

    def test_a_marker_may_be_json_or_have_the_key_with_an_equals_sign_or_in_capitals(self):
        for body in ('{"utc": "2026-11-19T01:00:00Z", "keep_rule": "v3"}', "2026-11-19T01:00:00Z\nkeep-rule = v3\n", "2026-11-19T01:00:00Z\nKeep_Rule: v3"):
            p = self.tmp / "m"
            p.write_text(body)
            self.assertEqual("v3", jobs.read_marker(p)["keep_rule"], body)

    def test_old_and_odd_markers_say_nothing(self):
        for body in ("", "x", "2026-11-18T12:42:50Z\n", "\ufeff2026-11-18T12:42:50Z\r\n", "{broken"):
            p = self.tmp / "m"
            p.write_text(body, encoding="utf-8")
            self.assertIsNone(jobs.read_marker(p)["keep_rule"], repr(body))
        self.assertEqual({"utc": None, "keep_rule": None, "fields": {}}, jobs.read_marker(self.tmp / "none"))

    def test_marker_text_refuses_a_rule_that_does_not_exist(self):
        with self.assertRaises(ValueError):
            jobs.marker_text("2026-11-19T01:00:00Z", "v9")

    def jobs_json(self, rule: str | None, form: str = "pair") -> None:
        cmd = ["py", "rl/gen0.py", "train", "--promote"]
        if rule:
            cmd += ["--keep-rule", rule] if form == "pair" else [f"--keep-rule={rule}"]
        (self.lab / "jobs.json").write_text(json.dumps({"train": {"cmd": cmd}}))

    def marker(self, rule: str | None) -> Path:
        p = self.tmp / "ready-5-g7"
        p.write_text(jobs.marker_text("2026-11-19T01:00:00Z", rule))
        return p

    def test_the_lab_s_rule_is_the_one_jobs_json_names_now(self):
        self.jobs_json("v2")
        self.assertEqual("v2", jobs.lab_keep_rule(self.lab))
        self.jobs_json("v3", "equals")
        self.assertEqual("v3", jobs.lab_keep_rule(self.lab))
        self.jobs_json(None)
        self.assertIsNone(jobs.lab_keep_rule(self.lab))
        self.assertIsNone(jobs.lab_keep_rule(self.tmp / "no-lab"))

    def test_a_marker_newer_than_the_lab_judges_and_the_verdict_says_so(self):
        # the generation 18 race: the worker's marker (built for v3) beat the switch of jobs.json to v3, which still said v2
        self.jobs_json("v2")
        c = jobs.choose_keep_rule("v2", self.lab, self.marker("v3"))
        self.assertEqual(("v3", "marker", "v3", "v2"), (c["rule"], c["source"], c["marker_rule"], c["lab_rule"]))
        self.assertIn("marker v3, newer than lab v2", c["why"])
        self.assertTrue(any("judged under v3" in n for n in c["notes"]), c["notes"])

    def test_the_lab_newer_than_the_marker_judges(self):
        self.jobs_json("v3")
        c = jobs.choose_keep_rule("v2", self.lab, self.marker("v2"))  # the job was made with v2 and jobs.json moved on: the lab's rule is jobs.json's
        self.assertEqual(("v3", "lab"), (c["rule"], c["source"]))
        self.assertIn("lab v3 (LAB/jobs.json), newer than marker v2", c["why"])
        self.assertTrue(any("job was made with --keep-rule v2" in n for n in c["notes"]), c["notes"])

    def test_the_same_rule_on_both_sides(self):
        self.jobs_json("v3")
        c = jobs.choose_keep_rule("v3", self.lab, self.marker("v3"))
        self.assertEqual(("v3", "both"), (c["rule"], c["source"]))

    def test_a_marker_without_the_field_gets_the_lab_s_rule_and_a_log_line(self):
        self.jobs_json("v3")
        c = jobs.choose_keep_rule(None, self.lab, self.marker(None))
        self.assertEqual(("v3", "lab", None), (c["rule"], c["source"], c["marker_rule"]))
        self.assertTrue(any("carries no keep_rule" in n for n in c["notes"]), c["notes"])

    def test_without_jobs_json_the_jobs_own_flag_then_the_variable_then_v1(self):
        self.assertEqual(("v2", "the job's --keep-rule"), (lambda c: (c["rule"], c["lab_source"]))(jobs.choose_keep_rule("v2", self.lab, self.marker(None))))
        self.assertEqual("v3", jobs.choose_keep_rule(None, self.lab, self.marker(None), env="v3")["rule"])
        self.assertEqual(("v1", "the default"), (lambda c: (c["rule"], c["lab_source"]))(jobs.choose_keep_rule(None, self.lab, self.marker(None))))
        self.assertEqual("v3", jobs.choose_keep_rule(None, None, self.marker("v3"))["rule"])  # no lab folder at all

    def test_a_marker_that_names_a_rule_that_does_not_exist_is_refused_not_guessed(self):
        p = self.tmp / "ready-5-g7"
        p.write_text("2026-11-19T01:00:00Z\nkeep_rule: v9\n")
        with self.assertRaises(ValueError) as cm:
            jobs.choose_keep_rule("v2", self.lab, p)
        self.assertIn("v9", str(cm.exception))

    def test_a_jobs_json_rule_that_is_not_a_rule_falls_back_to_the_jobs_own_setting_and_says_so(self):
        self.jobs_json("v7")
        c = jobs.choose_keep_rule("v2", self.lab, self.marker(None))
        self.assertEqual("v2", c["rule"])
        self.assertTrue(any("'v7'" in n for n in c["notes"]), c["notes"])

    def test_the_newer_of_two_rules(self):
        self.assertEqual("v3", jobs.newer_rule("v2", "v3"))
        self.assertEqual("v3", jobs.newer_rule("v3", None, "v1"))
        self.assertEqual("v1", jobs.newer_rule("v1", None))
        self.assertIsNone(jobs.newer_rule(None, None))

    def test_the_rule_names_are_the_judges(self):
        self.assertEqual(gen_loop.KEEP_RULES, jobs.KEEP_RULES)

    def test_the_train_job_s_record_shows_both_rules(self):
        (self.lab / "train").mkdir()
        (self.lab / "train" / "ready-17-g18-cheap-v3e").write_text(jobs.marker_text("2026-11-19T01:00:00Z", "v3"))
        self.jobs_json("v2")
        cfg = jobs.config_from({"train": {"cmd": ["py"], "enabled": True}})["train"]
        p = jobs.params_for("train", self.lab, cfg)
        self.assertEqual(("ready-17-g18-cheap-v3e", "v3", "v2"), (p["marker"], p["marker_keep_rule"], p["lab_keep_rule"]))
        (self.lab / "train" / "ready-17-g18-cheap-v3e").write_text("2026-11-19T01:00:00Z\n")
        self.assertIsNone(jobs.params_for("train", self.lab, cfg)["marker_keep_rule"])


class TrainJudgesUnderTheNewerRuleTests(unittest.TestCase):
    """`gen0.py train --marker-dir` with the promotion run replaced: which rule it hands the judge, and what it says about it."""

    setUp, tearDown, build = tg12.PlacementTests.setUp, tg12.PlacementTests.tearDown, tg12.PlacementTests.build

    def jobs_json(self, rule: str | None) -> None:
        cmd = ["py", "rl/gen0.py", "train"] + (["--keep-rule", rule] if rule else [])
        (self.root / "jobs.json").write_text(json.dumps({"train": {"cmd": cmd}}))

    def train(self, *extra: str):
        seeds = self.root / "seeds"
        seeds.mkdir(exist_ok=True)
        args = ["train", "--marker-dir", str(self.marker_dir), "--shards-root", str(self.root), "--agent-root", str(self.agents), "--lab", str(self.root),
                "--promote", "--emu-python", "python", "--seeds-file", str(seeds / "E{k}-{date}.json"), "--review-guard", *extra]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(gen_loop, "train_generation") as tg, mock.patch.object(gen_loop, "run_promote_subprocess", return_value=0) as rp, \
                mock.patch.object(gen_loop, "COMMITTED_SEEDS", seeds), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = gen0.main(args)
        return rc, rp, out.getvalue(), err.getvalue()

    def test_build_writes_the_rule_into_its_marker_and_a_build_without_one_writes_the_old_marker(self):
        self.assertEqual(0, self.build("--keep-rule", "v3"))
        marker = self.marker_dir / "ready-5-g7"
        got = jobs.read_marker(marker)
        self.assertEqual("v3", got["keep_rule"])
        self.assertRegex(got["utc"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        marker.unlink()
        self.assertEqual(0, self.build("--attempt", "2"))
        text = (self.marker_dir / "ready-5-g7-a2").read_text()
        self.assertEqual(1, len(text.splitlines()))  # the time and nothing else

    def test_the_marker_s_newer_rule_is_the_one_the_judge_gets_with_the_reason(self):
        self.assertEqual(0, self.build("--keep-rule", "v3"))
        self.jobs_json("v2")  # the lab has not been switched yet
        rc, rp, out, err = self.train("--keep-rule", "v2")
        self.assertEqual(0, rc, err)
        self.assertEqual("v3", rp.call_args.kwargs["keep_rule"])
        self.assertIn("marker v3, newer than lab v2", rp.call_args.kwargs["keep_rule_why"])
        self.assertIn("keep rule: judged under v3", out)

    def test_a_switch_made_after_the_job_was_made_is_the_lab_s_rule(self):
        self.assertEqual(0, self.build("--keep-rule", "v2"))
        self.jobs_json("v3")  # the orchestrator switched jobs.json after the job's command was fixed at --keep-rule v2
        rc, rp, out, err = self.train("--keep-rule", "v2")
        self.assertEqual(0, rc, err)
        self.assertEqual("v3", rp.call_args.kwargs["keep_rule"])
        self.assertIn("lab v3 (LAB/jobs.json), newer than marker v2", rp.call_args.kwargs["keep_rule_why"])

    def test_a_marker_without_the_field_gets_the_lab_s_rule_and_a_log_line(self):
        self.assertEqual(0, self.build())
        self.jobs_json("v3")
        rc, rp, out, err = self.train()
        self.assertEqual(0, rc, err)
        self.assertEqual("v3", rp.call_args.kwargs["keep_rule"])
        self.assertIn("the marker carries no keep_rule, so the lab's rule judges it", out)
        self.assertIn("marker names no rule", rp.call_args.kwargs["keep_rule_why"])

    def test_when_nothing_names_a_rule_nothing_is_passed_on_and_the_child_reads_the_variable_or_v1(self):
        self.assertEqual(0, self.build())  # no rule in the marker, no jobs.json, no flag
        with mock.patch.dict(os.environ):
            os.environ.pop(gen_loop.KEEP_RULE_FLAG, None)
            rc, rp, out, err = self.train()
        self.assertEqual(0, rc, err)
        self.assertIsNone(rp.call_args.kwargs["keep_rule"])
        self.assertIn("the default", rp.call_args.kwargs["keep_rule_why"])
        self.assertIn("keep rule: judged under v1", out)

    def test_a_marker_that_names_a_rule_that_does_not_exist_stops_the_round_and_stays(self):
        self.assertEqual(0, self.build())
        marker = self.marker_dir / "ready-5-g7"
        marker.write_text("2026-11-19T01:00:00Z\nkeep_rule: v9\n")
        rc, rp, out, err = self.train()
        self.assertEqual(2, rc)
        self.assertIn("v9", err)
        rp.assert_not_called()
        self.assertTrue(marker.exists())  # the marker stays: nothing was judged

    def test_the_extension_follows_the_rule_chosen(self):
        self.assertEqual(0, self.build("--keep-rule", "v3"))
        self.jobs_json("v2")
        cand = self.agents / "gen7" / "candidate"
        with mock.patch.object(gen_loop, "extension_of", return_value=None) as ext:
            rc, rp, out, err = self.train("--keep-rule", "v2")
        self.assertEqual(0, rc, err)
        ext.assert_called_once()  # the v3 branch ran: the chosen rule is v3 though the job's flag said v2
        self.assertEqual(cand, rp.call_args[0][1])

    def test_the_promote_command_hands_the_reason_to_the_verdict(self):
        ran = []
        with mock.patch.object(gen_loop.subprocess, "run", lambda cmd, check=False: ran.append(cmd) or mock.Mock(returncode=0)):
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None, keep_rule="v3", keep_rule_why="marker v3, newer than lab v2")
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None, keep_rule="v3")
        self.assertEqual("v3", ran[0][ran[0].index("--keep-rule") + 1])
        self.assertEqual("marker v3, newer than lab v2", ran[0][ran[0].index("--keep-rule-why") + 1])
        self.assertNotIn("--keep-rule-why", ran[1])
        a = gen0.build_parser().parse_args(["promote", "--cand", "c", "--agent-root", "a", "--keep-rule", "v3", "--keep-rule-why", "x"])
        self.assertEqual("x", a.keep_rule_why)

    def test_a_manual_train_without_a_marker_directory_is_as_it_was(self):
        # --gen N with an explicit rule: no marker, so no choice (jobs.json is not read) and the flag goes through untouched
        self.jobs_json("v3")
        seeds = self.root / "seeds"
        seeds.mkdir(exist_ok=True)
        with mock.patch.object(gen_loop, "train_generation", return_value=self.agents / "gen5" / "candidate"), \
                mock.patch.object(gen_loop, "run_promote_subprocess", return_value=0) as rp, mock.patch.object(gen_loop, "COMMITTED_SEEDS", seeds), \
                mock.patch.object(gen_loop, "candidate_identity", return_value=("s", "h")), mock.patch.object(gen_loop, "resolve_e_block", return_value=seeds / "E.json"), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            gen0.main(["train", "--gen", "5", "--shards-root", str(self.root), "--agent-root", str(self.agents), "--lab", str(self.root), "--promote",
                       "--emu-python", "python", "--seeds-file", str(seeds / "E{k}-{date}.json"), "--keep-rule", "v1"])
        self.assertEqual("v1", rp.call_args.kwargs["keep_rule"])
        self.assertIsNone(rp.call_args.kwargs["keep_rule_why"])


class PromoteRecordsTheChoiceTests(unittest.TestCase):
    """`gen_loop.promote(keep_rule_why=...)` on the synthetic arms: promotion.json and the ledger row say which rule judged it and why."""

    setUp, tearDown = tf.PromoteV2Tests.setUp, tf.PromoteV2Tests.tearDown

    def run_promote(self, cand: Path, **kw):
        arms = tf.StubArms(self.cand_rows, self.base_rows)
        with mock.patch.object(gen_loop, "run_arm", arms):
            return gen_loop.promote(cand, None, self.seed_file, agent_root=self.agents, boot=500, sequential=True, keep_rule="v2", **kw)

    def test_the_reason_is_in_promotion_json_and_the_guard_cell(self):
        cand = tf.stub_candidate(self.root, "c1", "sha-a", generation=4)
        doc = self.run_promote(cand, keep_rule_why="marker v2, newer than lab v1")
        self.assertEqual({"rule": "v2", "why": "marker v2, newer than lab v1"}, doc["keep_rule_choice"])
        self.assertEqual(doc["keep_rule_choice"], json.loads((cand / "promotion.json").read_text())["keep_rule_choice"])
        row = gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]
        self.assertEqual("v2", row["rule"])  # the ledger's rule column: the rule that judged
        self.assertIn(" | keep rule v2: marker v2, newer than lab v1", row["guard"])
        self.assertNotIn("\t", row["guard"])

    def test_without_a_reason_the_record_is_as_it_was(self):
        cand = tf.stub_candidate(self.root, "c1", "sha-a", generation=4)
        doc = self.run_promote(cand)
        self.assertNotIn("keep_rule_choice", doc)
        self.assertNotIn("keep rule", gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]["guard"])


# ---------------------------------------------------------------- the silent failures


class SilentAlertUnitTests(Tmp):
    NOW = 1_800_000_000.0

    def test_three_failures_in_a_day_are_more_than_twice(self):
        st: dict = {}
        for i in range(2):
            al.record_failure(st, "train", f"job{i}", 1, 60.0, self.NOW + i)
        self.assertEqual([], al.raise_repeats(self.lab, st, self.NOW + 5))
        al.record_failure(st, "train", "job2", 1, 60.0, self.NOW + 10)
        made = al.raise_repeats(self.lab, st, self.NOW + 11)
        self.assertEqual([self.lab / "alerts" / "train-repeat.txt"], made)
        text = made[0].read_text(encoding="utf-8")
        self.assertIn("ALERT train-repeat: train failed 3 times in 24 hours", text)
        for j in ("job0", "job1", "job2"):
            self.assertIn(f"runs/{j}: rc=1", text)
        self.assertEqual([], al.raise_repeats(self.lab, st, self.NOW + 12))  # one file, however often it is looked for

    def test_failures_older_than_a_day_do_not_count(self):
        st: dict = {}
        al.record_failure(st, "train", "old1", 1, 1.0, self.NOW)
        al.record_failure(st, "train", "old2", 1, 1.0, self.NOW + 60)
        al.record_failure(st, "train", "new", 1, 1.0, self.NOW + 25 * 3600)
        self.assertEqual([], al.raise_repeats(self.lab, st, self.NOW + 25 * 3600))
        self.assertEqual(1, len(st["fail_times"]["train"]))  # the two old ones were forgotten when the new one was noted

    def test_kinds_are_counted_apart(self):
        st: dict = {}
        for kind in ("train", "fights", "selfplay"):
            al.record_failure(st, kind, "j", 1, 1.0, self.NOW)
        self.assertEqual([], al.raise_repeats(self.lab, st, self.NOW))

    def test_removing_the_alert_starts_the_count_again(self):
        st: dict = {}
        for i in range(3):
            al.record_failure(st, "train", f"j{i}", 1, 1.0, self.NOW + i)
        (path,) = al.raise_repeats(self.lab, st, self.NOW + 5)
        path.unlink()
        self.assertEqual([], al.raise_repeats(self.lab, st, self.NOW + 6))  # the count is cleared, nothing is raised again at once
        self.assertEqual({}, st.get("fail_times", {}))
        for i in range(3):
            al.record_failure(st, "train", f"k{i}", 1, 1.0, self.NOW + 10 + i)
        self.assertEqual(1, len(al.raise_repeats(self.lab, st, self.NOW + 20)))

    def test_a_file_that_stands_without_the_state_knowing_is_not_written_over(self):
        st: dict = {}
        path = al.alert_path(self.lab, "train-repeat")
        path.parent.mkdir(parents=True)
        path.write_text("by hand\n")
        for i in range(3):
            al.record_failure(st, "train", f"j{i}", 1, 1.0, self.NOW + i)
        self.assertEqual([], al.raise_repeats(self.lab, st, self.NOW + 5))
        self.assertEqual("by hand\n", path.read_text())

    def test_two_time_limit_kills_in_a_week(self):
        st: dict = {}
        al.record_timeout(st, "score-real", "job1", "score-real", 14400.0, 14400.0, self.NOW)
        self.assertEqual([], al.raise_repeats(self.lab, st, self.NOW))
        al.record_timeout(st, "score-real", "job2", "score-real", 14400.0, 14401.0, self.NOW + 3 * 86400)
        (path,) = al.raise_repeats(self.lab, st, self.NOW + 3 * 86400)
        self.assertEqual("score-real-timeout.txt", path.name)
        text = path.read_text(encoding="utf-8")
        self.assertIn("ALERT score-real-timeout: score-real was killed at its time limit 2 times in 7 days", text)
        self.assertIn("runs/job1 score-real: killed at its 14400s limit", text)
        self.assertIn("runs/job2 score-real", text)

    def test_kills_more_than_a_week_apart_are_two_single_kills(self):
        st: dict = {}
        al.record_timeout(st, "train", "j1", "train", 60.0, 61.0, self.NOW)
        al.record_timeout(st, "train", "j2", "train", 60.0, 61.0, self.NOW + 8 * 86400)
        self.assertEqual([], al.raise_repeats(self.lab, st, self.NOW + 8 * 86400))

    def line(self, tag: str, age_h: float, pid: str | None = "4242") -> str:
        return f"{tag} {utc(self.NOW - age_h * 3600)}" + (f" pid={pid}" if pid else "")

    def test_a_line_over_an_hour_old_whose_process_is_gone_is_listed(self):
        self.qpath.write_text("\n".join([self.line("w3-blocks", 2.0), self.line("fresh", 0.5), self.line("alive", 3.0, "77"), self.line("nopid", 5.0, None),
                                         "garbage"]) + "\n")
        got = al.dead_quiet_lines(self.qpath, self.NOW, alive=lambda pid: pid == 77)
        self.assertEqual(["w3-blocks"], [d["tag"] for d in got])
        self.assertEqual((4242, round(got[0]["age_h"], 1)), (got[0]["pid"], 2.0))

    def test_the_dead_holders_line_raises_one_alert_once(self):
        st: dict = {}
        self.qpath.write_text(self.line("w3-blocks", 2.0) + "\n")
        (path,) = al.raise_dead_locks(self.lab, self.qpath, st, self.NOW, alive=lambda pid: False)
        self.assertEqual("quiet-lock-w3-blocks.txt", path.name)
        text = path.read_text(encoding="utf-8")
        self.assertIn("ALERT quiet-lock-w3-blocks: the QUIET.lock line 'w3-blocks ", text)
        self.assertIn("process 4242 is gone", text)
        self.assertIn("Remove that one line from QUIET.lock by hand", text)
        self.assertEqual([], al.raise_dead_locks(self.lab, self.qpath, st, self.NOW + 600, alive=lambda pid: False))
        path.unlink()  # removed by hand while the line stands: not raised again every look
        self.assertEqual([], al.raise_dead_locks(self.lab, self.qpath, st, self.NOW + 1200, alive=lambda pid: False))
        self.qpath.write_text(self.line("w3-blocks", 1.5, "5151") + "\n")  # another dead line of that tag: its own alert
        self.assertEqual(1, len(al.raise_dead_locks(self.lab, self.qpath, st, self.NOW + 1300, alive=lambda pid: False)))

    def test_a_live_holder_or_a_young_line_raises_nothing(self):
        st: dict = {}
        self.qpath.write_text(self.line("lab-nightly", 3.0, "77") + "\n" + self.line("young", 0.9) + "\n")
        self.assertEqual([], al.raise_dead_locks(self.lab, self.qpath, st, self.NOW, alive=lambda pid: pid == 77))

    def test_no_lock_file_is_nothing(self):
        self.assertEqual([], al.raise_dead_locks(self.lab, self.tmp / "none", {}, self.NOW, alive=lambda pid: False))

    @unittest.skipUnless(winsys.IS_WIN, "asks Windows whether a process lives")
    def test_the_default_asks_whether_the_process_lives(self):
        self.qpath.write_text(f"mine {utc(time.time() - 7200)} pid={os.getpid()}\ngone {utc(time.time() - 7200)} pid=99999999\n")
        self.assertEqual(["gone"], [d["tag"] for d in al.dead_quiet_lines(self.qpath)])

    def test_the_alerts_show_in_the_banner_like_any_other(self):
        st: dict = {}
        for i in range(3):
            al.record_failure(st, "train", f"j{i}", 1, 1.0, self.NOW + i)
        al.raise_repeats(self.lab, st, self.NOW + 5)
        self.assertIn("ALERT train-repeat", al.status_prefix(self.lab))


class SilentAlertRunnerTests(Case):
    """The runner wiring: a failure or a kill is noted at the job's end and the alert is raised at once; the QUIET.lock look runs on its tick."""

    def test_fail_ok_fail_ok_fail_never_two_in_a_row_but_three_in_a_day(self):
        r = self.runner()
        for i in range(5):
            (self.good if i % 2 else self.failing)()
            r.execute(r.new_run("fights", "abc"))
            if i < 4:
                self.assertFalse(al.alert_path(r.lab.root, "fights-repeat").exists(), i)
        self.assertFalse(al.alert_path(r.lab.root, "fights").exists())  # the two-in-a-row alert never fired
        self.assertIsNone(r.unavailable("fights", "abc"))  # the kind is still picked: the new alert holds nothing
        path = al.alert_path(r.lab.root, "fights-repeat")
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("ALERT fights-repeat: fights failed 3 times in 24 hours", text)
        self.assertIn("rc=1", text)
        self.assertIn("line5", text)  # the failed command's log tail, as the other alert shows it
        self.assertEqual(1, self.log(r).count("ALERT fights-repeat: raised"))
        self.assertIn("alerted_extra", r.state)

    def test_the_counts_survive_a_restart_of_the_runner(self):
        r = self.runner()
        self.failing()
        for i in range(2):
            self.failing()
            r.execute(r.new_run("fights", "abc"))
            self.good()
            r.execute(r.new_run("fights", "abc"))
        self.assertEqual(2, len(r.state["fail_times"]["fights"]))
        r.lab.save_state(r.state)
        r2 = self.runner()
        self.failing()
        r2.execute(r2.new_run("fights", "abc"))
        self.assertTrue(al.alert_path(r2.lab.root, "fights-repeat").exists())  # the third, counted with the two the first process saw

    def test_two_kills_at_the_time_limit_raise_the_timeout_alert(self):
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [self.py("slow", "import time; time.sleep(30)", core=True, timeout=1.0)], mode="serial")]
        r = self.runner()
        first = r.new_run("fights", "abc")
        r.execute(first)
        self.assertFalse(al.alert_path(r.lab.root, "fights-timeout").exists())
        second = r.new_run("fights", "abc")
        r.execute(second)
        text = al.alert_path(r.lab.root, "fights-timeout").read_text(encoding="utf-8")
        self.assertIn("killed at its time limit 2 times in 7 days", text)
        self.assertIn(f"runs/{first.name} slow", text)
        self.assertIn(f"runs/{second.name} slow", text)
        self.assertIn("ALERT fights-timeout: raised", self.log(r))

    def test_a_dead_holders_line_is_raised_on_the_tick_and_not_before_the_next(self):
        r = self.runner()
        (self.tmp / "QUIET.lock").write_text(f"w3-blocks {utc(time.time() - 7200)} pid=99999999\n")
        with mock.patch.object(winsys, "parent_alive", return_value=False):
            r.service_alerts(force=True)
            self.assertTrue(al.alert_path(r.lab.root, "quiet-lock-w3-blocks").exists())
            self.assertIn("ALERT quiet-lock-w3-blocks: raised", self.log(r))
            al.alert_path(r.lab.root, "quiet-lock-w3-blocks").unlink()
            (self.tmp / "QUIET.lock").write_text(f"other {utc(time.time() - 7200)} pid=99999998\n")
            r.service_alerts()  # inside the ten minutes: not looked at
            self.assertFalse(al.alert_path(r.lab.root, "quiet-lock-other").exists())
            r._alert_check -= 700
            r.service_alerts()
            self.assertTrue(al.alert_path(r.lab.root, "quiet-lock-other").exists())

    def test_the_filler_service_call_makes_the_look(self):
        r = self.runner()
        (self.tmp / "QUIET.lock").write_text(f"w3-blocks {utc(time.time() - 7200)} pid=99999999\n")
        with mock.patch.object(winsys, "parent_alive", return_value=False):
            r.service_filler(start=False)
        self.assertTrue(al.alert_path(r.lab.root, "quiet-lock-w3-blocks").exists())

    def test_a_fault_in_the_look_is_a_log_line(self):
        r = self.runner()
        with mock.patch.object(al, "raise_repeats", side_effect=RuntimeError("boom")):
            r.service_alerts(force=True)
        self.assertIn("alerts: the silent-failure check failed (RuntimeError)", self.log(r))

    def test_a_dry_run_looks_at_nothing(self):
        r = self.runner("--dry-run")
        (self.tmp / "QUIET.lock").write_text(f"w3-blocks {utc(time.time() - 7200)} pid=99999999\n")
        with mock.patch.object(winsys, "parent_alive", return_value=False):
            r.service_alerts(force=True)
        self.assertFalse(al.alert_path(r.lab.root, "quiet-lock-w3-blocks").exists())


if __name__ == "__main__":
    _ = FAIL_CODE
    unittest.main()
