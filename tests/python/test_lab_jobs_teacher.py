"""tools/lab/jobs.py: the `teacher` kind (D56, D57 part 3; rl/run_teacher.py `job`, docs/lab/TEACHER_JOB.md).

The filler's rules as one function of the lab folder (`teacher_hold`): the night, a live QUIET.lock line (judgement first), a waiting train
or score marker, a presort due within the hour, HALT and PAUSE, and the daily core-hour budget (the ledger). Scratch lab folders and a
stand-in clock only: no game, no emulator, no git.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.lab import cores  # noqa: E402
from tools.lab import jobs as jb  # noqa: E402
from tools.lab import plans as pl  # noqa: E402
from tools.lab import runner as rn  # noqa: E402

NOW = time.time()


def iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.lab = self.tmp / "ref" / "LAB"
        self.lab.mkdir(parents=True)
        self.quiet = self.tmp / "ref" / "QUIET.lock"
        self.store = self.tmp / "store"
        git = mock.patch.object(jb, "_git_main", return_value=None)
        git.start()
        self.addCleanup(git.stop)

    def config(self, **kinds):
        (self.lab / "jobs.json").write_text(json.dumps(kinds), encoding="utf-8")

    def cfg(self, **over):
        c = dict(jb.config_from(json.loads((self.lab / "jobs.json").read_text(encoding="utf-8")) if (self.lab / "jobs.json").exists() else {})["teacher"])
        c.update(enabled=True, store=str(self.store), **over)
        return c

    def lines(self, *lines: str):
        self.quiet.write_text("".join(x + "\n" for x in lines), encoding="utf-8")

    def hold(self, starting=True, now=None, **kw):
        cfg = kw.pop("cfg", None) or self.cfg()
        kw.setdefault("main_sha", None)
        kw.setdefault("ledger_rows", [])
        return jb.teacher_hold(self.lab, cfg, starting=starting, now=NOW if now is None else now, quiet_path=self.quiet, **kw)


class KindTests(Base):
    def test_it_is_a_kind_with_the_lowest_priority_and_off_until_jobs_json_says_so(self):
        self.assertIn("teacher", jb.NEW_KINDS)
        self.assertIn("teacher", pl.PLANS)
        self.assertIn("teacher", rn.JOB_TYPES)
        self.assertGreater(jb.PRIORITY["teacher"], max(v for k, v in jb.PRIORITY.items() if k != "teacher"))
        d = jb.DEFAULTS["teacher"]
        self.assertFalse(d["enabled"])
        self.assertEqual(24.0, d["core_hours_per_day"])
        self.assertEqual(4, cores.job_cores("teacher", None, None))
        self.assertIn("is off", jb.unready("teacher", self.lab, jb.config_from({})["teacher"]))
        p = jb.params_for("teacher", self.lab, jb.config_from({})["teacher"])
        self.assertEqual("teacher", p["strategy"])
        self.assertIn("job", p["cmd"])

    def test_a_shipped_default_command_runs_the_job_command_with_the_runners_tokens(self):
        cmd = jb.DEFAULTS["teacher"]["cmd"]
        self.assertEqual(["{python}", "rl/run_teacher.py", "job"], cmd[:3])
        self.assertIn("{lab}", cmd)
        self.assertEqual("{tree}", jb.DEFAULTS["teacher"]["cwd"])

    def test_the_other_kinds_are_unchanged_by_it(self):
        for k, want in (("selfplay", False), ("train", True), ("score-real", True), ("nightly-real", False)):
            self.assertEqual(want, jb.DEFAULTS[k]["enabled"], k)
        self.assertEqual(("selfplay", "train", "score-real"), jb.LOOP_KINDS)  # the teacher is no part of the old loop's share

    def test_a_ready_lab_lets_it_start(self):
        self.assertIsNone(self.hold())
        self.config(teacher={"enabled": True, "store": str(self.store)})
        self.assertIsNone(jb.unready("teacher", self.lab, jb.load_config(self.lab)["teacher"]))


class QuietAndJudgementFirstTests(Base):
    def test_it_yields_to_every_live_line_but_the_loops_and_the_daytime_scores(self):
        for tag in ("lab-train", "w3-blocks", "judge-corpus", "lab-nightly", "gate", "anything-else"):
            self.lines(f"{tag} {iso(NOW - 600)} pid=7")
            self.assertEqual(f"QUIET.lock is held by {tag}", self.hold(), tag)
            self.assertEqual(f"QUIET.lock is held by {tag}", self.hold(starting=False), tag)  # a running block stops too
        self.lines(f"gym-loop {iso(NOW - 600)} pid=7", f"lab-score {iso(NOW - 600)} pid=8")
        self.assertIsNone(self.hold())
        self.lines(f"lab-train {iso(NOW - 3 * 3600)} pid=7")  # older than two hours: the holder is gone
        self.assertIsNone(self.hold())

    def test_its_own_ignore_list_replaces_the_default(self):
        self.lines(f"judge-corpus {iso(NOW - 60)} pid=7", f"gym-loop {iso(NOW - 60)} pid=8")
        self.assertIsNone(self.hold(cfg=self.cfg(quiet_ignore=["gym-loop", "judge-corpus"])))
        self.assertEqual("QUIET.lock is held by gym-loop", self.hold(cfg=self.cfg(quiet_ignore=["judge-corpus"])))

    def test_it_holds_no_line_of_its_own(self):
        text = (ROOT / "tools" / "lab" / "jobs.py").read_text(encoding="utf-8")
        self.assertNotIn("lab-teacher", text)  # the existing filler kinds hold none, so neither does this one

    def test_a_waiting_train_marker_or_score_marker_comes_first_but_does_not_stop_a_running_block(self):
        (self.lab / "train").mkdir()
        (self.lab / "train" / "ready-13-g14-stakes").write_text("1\n")
        self.assertEqual("judgement first: ready-13-g14-stakes is waiting", self.hold())
        self.assertIsNone(self.hold(starting=False))
        self.config(train={"enabled": False})  # a switched-off kind is not waiting for the machine
        self.assertIsNone(self.hold())
        self.config()
        (self.lab / "train" / "ready-13-g14-stakes").unlink()
        (self.lab / "score").mkdir()
        (self.lab / "score" / "pending").write_text("1\n")
        self.config(**{"score-real": {"enabled": True}})
        self.assertEqual("judgement first: LAB/score/pending is waiting", self.hold())

    def test_halt_and_pause(self):
        (self.lab / "HALT").write_text("x")
        self.assertEqual("LAB/HALT is up", self.hold())
        self.assertEqual("LAB/HALT is up", self.hold(starting=False))
        (self.lab / "HALT").unlink()
        (self.lab / "PAUSE").write_text("x")
        self.assertEqual("LAB/PAUSE is up", self.hold())


class NightTests(Base):
    def night(self, due_in_s: float):
        """A nightly-real kind that is due `due_in_s` seconds from now (last-ok + every_h)."""
        self.config(**{"nightly-real": {"enabled": True, "every_h": 20.0}})
        nd = self.lab / "nightly"
        nd.mkdir(exist_ok=True)
        ok = nd / "last-ok"
        ok.write_text("x")
        t = NOW + due_in_s - 20 * 3600
        import os

        os.utime(ok, (t, t))

    def test_none_while_the_night_runs(self):
        (self.lab / "nightly").mkdir()
        (self.lab / "nightly" / "active").write_text("1\n")
        self.assertEqual("a nightly block is running (LAB/nightly/active)", self.hold())
        self.assertEqual("a nightly block is running (LAB/nightly/active)", self.hold(starting=False))

    def test_no_start_within_the_margin_of_the_nights_due_time_and_a_running_block_stops_two_minutes_before(self):
        self.night(due_in_s=20 * 60)  # due in 20 minutes: inside the 30-minute margin
        self.assertIn("the night is due at", self.hold())
        self.assertIsNone(self.hold(starting=False))  # a block running now ends within its own block_min, stopped only two minutes before
        self.night(due_in_s=90)
        self.assertIn("the night is due at", self.hold(starting=False))
        self.night(due_in_s=3 * 3600)
        self.assertIsNone(self.hold())

    def test_a_pending_promotion_makes_the_night_due_now(self):
        self.night(due_in_s=10 * 3600)
        (self.lab / "nightly" / "pending").write_text("1\n")
        self.assertIn("the night is due at", self.hold())


class PresortTests(Base):
    def test_a_presort_due_within_the_hour_keeps_a_block_from_starting(self):
        st = {"last_presorted_sha": "aaaaaaa", "tree_sha": "aaaaaaa", "last_presort_end": NOW - 300}
        self.assertIsNone(jb.presort_due_soon(st, "aaaaaaa", NOW, 3600))  # main is the presorted tree
        self.assertIn("a presort of bbbbbbb is due within 60 min", jb.presort_due_soon(st, "bbbbbbb", NOW, 3600))
        self.assertIsNone(jb.presort_due_soon(st, "bbbbbbb", NOW, 3600, gap_min=180))  # the gap since the last presort puts it later
        self.assertIsNone(jb.presort_due_soon(st, None, NOW, 3600))  # main unknown: no rule
        bad = {**st, "bad_shas": {"bbbbbbb": {"fails": 2, "retry_after": NOW + 5 * 3600}}}
        self.assertIsNone(jb.presort_due_soon(bad, "bbbbbbb", NOW, 3600))  # a failed build waits out its retry
        self.assertIsNone(jb.presort_due_soon({"last_presort_end": NOW - 86400}, "", NOW, 3600))  # no sha known

    def test_the_hold_reads_the_runners_state_file_and_forks_main(self):
        (self.lab / "state.json").write_text(json.dumps({"last_presorted_sha": "aaaaaaa", "tree_sha": "aaaaaaa", "last_presort_end": NOW - 300}))
        self.assertEqual("a presort of bbbbbbb is due within 60 min (main moved past the presorted tree)", self.hold(main_sha="bbbbbbb"))
        self.assertIsNone(self.hold(main_sha="bbbbbbb", starting=False))  # a running block carries on: it ends within its block time
        self.assertIsNone(self.hold(main_sha="aaaaaaa"))
        with mock.patch.object(jb, "_git_main", return_value="ccccccc") as g:
            self.assertIn("ccccccc"[:7], self.hold(main_sha=""))
            g.assert_called_once()

    def test_a_state_without_the_last_presort_end_reads_the_newest_presort_folder(self):
        d = self.lab / "runs" / "20261111T0100-presort"
        d.mkdir(parents=True)
        (d / "done.rc").write_text("0\n")
        self.assertIsNotNone(jb._last_presort_end(self.lab))
        self.assertIsNone(jb._last_presort_end(self.tmp / "nowhere"))


class BudgetTests(Base):
    DAY = time.strftime("%Y-%m-%d", time.gmtime(NOW))

    def end(self, cpu, seed="s", job="j1", day=None, **extra):
        t = NOW if day is None else day
        return {"t": iso(t), "ev": "end", "job": job, "seed": seed, "cpu": cpu, **extra}

    def start(self, seed="s", job="j1", est=1000.0, age_s=60.0):
        return {"t": iso(NOW - age_s), "ev": "start", "job": job, "seed": seed, "est": est}

    def test_the_days_spend_is_the_finished_cpu_plus_the_runs_in_flight(self):
        rows = [self.end(36000.0, "a"), self.end(18000.0, "b"), self.start("c", est=3600.0), self.start("a", est=999.0)]  # a started and ended
        self.assertAlmostEqual(36000 + 18000 + 3600, jb.teacher_spent_s(rows, self.DAY, NOW))
        cfg = {"core_hours_per_day": 24.0}
        self.assertAlmostEqual(24.0 - 16.0, jb.teacher_budget_left_h(cfg, rows, NOW))

    def test_a_reservation_without_an_estimate_uses_the_mean_of_the_finished_runs(self):
        rows = [self.end(1000.0, "a"), self.end(3000.0, "b"), {"t": iso(NOW - 10), "ev": "start", "job": "j1", "seed": "c"}]
        self.assertAlmostEqual(4000 + 2000, jb.teacher_spent_s(rows, self.DAY, NOW))
        self.assertAlmostEqual(jb.TEACHER_EST_CPU_S, jb.teacher_spent_s([{"t": iso(NOW - 10), "ev": "start", "job": "j", "seed": "c"}], self.DAY, NOW))

    def test_a_job_that_ended_a_retraction_and_an_old_start_are_not_in_flight(self):
        rows = [self.start("a", job="dead", est=5000.0), {"t": iso(NOW), "ev": "job_end", "job": "dead"},
                self.start("b", est=5000.0, age_s=3 * 3600),  # a killed job's leftover: older than the reserve
                self.start("c", est=5000.0), self.end(0.0, "c", retracted=True)]
        self.assertEqual(0.0, jb.teacher_spent_s(rows, self.DAY, NOW))

    def test_another_utc_day_does_not_count(self):
        rows = [self.end(80000.0, "a", day=NOW - 2 * 86400)]
        self.assertEqual(0.0, jb.teacher_spent_s(rows, self.DAY, NOW))
        self.assertEqual(24.0, jb.teacher_budget_left_h({"core_hours_per_day": 24.0}, rows, NOW))

    def test_the_budget_stops_the_job_at_its_start_not_in_the_middle(self):
        full = [self.end(24 * 3600.0, "a")]
        self.assertIn("the day's budget is spent (24 core-hours", self.hold(ledger_rows=full))
        self.assertIsNone(self.hold(ledger_rows=full, starting=False))  # a running block is not cut; its workers start no run the budget lacks room for
        self.assertIsNone(self.hold(ledger_rows=[self.end(23 * 3600.0, "a")]))
        self.assertIn("budget is spent", self.hold(ledger_rows=[self.end(20 * 3600.0, "a"), self.start("b", est=5 * 3600.0)]))  # in flight counts
        self.assertIsNone(self.hold(cfg=self.cfg(core_hours_per_day=48.0), ledger_rows=full))

    def test_unready_is_the_hold_with_the_switch(self):
        self.config(teacher={"enabled": True, "store": str(self.store)})
        cfg = jb.load_config(self.lab)["teacher"]
        self.assertIsNone(jb.unready("teacher", self.lab, cfg))
        jb.append_ledger(jb.teacher_ledger(self.store), {"ev": "end", "job": "j", "seed": "s", "cpu": 25 * 3600.0})
        self.assertIn("budget is spent", jb.unready("teacher", self.lab, cfg))

    def test_the_ledger_is_one_json_line_an_event_and_a_cut_line_is_dropped(self):
        led = jb.teacher_ledger(self.store)
        jb.append_ledger(led, {"ev": "start", "job": "j", "seed": "s", "est": 10})
        jb.append_ledger(led, {"ev": "end", "job": "j", "seed": "s", "cpu": 7.5})
        with open(led, "a", encoding="utf-8") as fh:
            fh.write('{"ev": "end", "job"')  # cut by a kill
        rows = jb.read_ledger(led)
        self.assertEqual(["start", "end"], [r["ev"] for r in rows])
        self.assertRegex(rows[0]["t"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual([], jb.read_ledger(self.store / "nothing.jsonl"))


class FragmentTests(unittest.TestCase):
    def test_the_documented_jobs_json_fragment_parses_and_is_disabled(self):
        doc = (ROOT / "docs" / "lab" / "TEACHER_JOB.md").read_text(encoding="utf-8")
        m = re.search(r"```json\n(\{.*?\n\})\n```", doc, re.S)
        self.assertIsNotNone(m, "docs/lab/TEACHER_JOB.md carries the jobs.json fragment in a json block")
        frag = json.loads(m.group(1))
        self.assertIn("teacher", frag)
        self.assertFalse(frag["teacher"]["enabled"])
        cfg = jb.config_from(frag)["teacher"]
        self.assertEqual(["{python}", "rl/run_teacher.py", "job"], cfg["cmd"][:3])
        self.assertEqual(jb.DEFAULTS["teacher"]["cmd"], cfg["cmd"])  # the fragment is the shipped default, written out
        self.assertEqual(jb.DEFAULTS["teacher"]["needs"], cfg["needs"])
        self.assertEqual(24.0, cfg["core_hours_per_day"])
        self.assertNotIn("selfplay", frag)  # it changes nothing else of the live file


if __name__ == "__main__":
    unittest.main()
