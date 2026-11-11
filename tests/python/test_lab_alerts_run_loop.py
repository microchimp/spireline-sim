"""tools/lab/alerts.py: the run loop's two alert files (D58): `run-student.txt` (every kind failed the human bar twice running) and `run-plateau.txt`
(ten days without a run-loop promotion). Pure file work in a scratch folder."""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.lab import alerts as al  # noqa: E402

NOW = 1_800_000_000.0
ROUNDS = [
    {"round": 3, "tag": "student-r3", "utc": "2026-12-01T03:00:00Z", "candidate": None, "judged": None,
     "verdict": {"card": {"pass": False, "leg1": True, "leg2": False, "why": "leg 2: won +0.0100 (at most +0.002), all +0.0300 (at most +0.02)"},
                 "route": {"pass": True, "leg1": True, "leg2": True, "why": ""}}},
    {"round": 4, "tag": "student-r4", "utc": "2026-12-04T03:00:00Z", "candidate": "gen20/candidate-student-r4", "judged": "rejected",
     "verdict": {"card": {"pass": False, "leg1": False, "leg2": None, "why": "C:/" + "Users" + "/someone/x/leg.json: bad"}}},
]


class RunLoopAlertTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_run_student_names_the_rounds_and_the_kinds_that_failed_and_how_to_clear_it(self):
        path = al.write_run_student(self.tmp, ROUNDS, now=NOW)
        self.assertEqual(self.tmp / "alerts" / "run-student.txt", path)
        text = path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("ALERT run-student: every kind failed the human bar in 2 student rounds running, raised 2027-01-15T"))
        self.assertIn("remove this file by hand", text)
        self.assertIn("LAB/student/frozen.json", text)
        self.assertIn("round 3 (student-r3, 2026-12-01): no candidate; judged no", text)
        self.assertIn("round 4 (student-r4, 2026-12-04): gen20/candidate-student-r4; judged rejected", text)
        self.assertIn("  card: leg 1 True, leg 2 False: leg 2: won +0.0100 (at most +0.002), all +0.0300 (at most +0.02)", text)
        self.assertNotIn("  route:", text)  # a kind that passed is not a failure
        self.assertNotIn("Users", text)  # no user path in a results file

    def test_run_plateau_gives_the_start_of_the_days_and_the_last_verdicts(self):
        path = al.write_run_plateau(self.tmp, NOW - 11 * 86400, 10.0, ROUNDS, now=NOW)
        text = path.read_text(encoding="utf-8")
        self.assertIn("ALERT run-plateau: no run-loop promotion in 10 days (since 2027-01-04), raised 2027-01-15T", text)
        self.assertIn("The loop keeps running", text)
        self.assertIn("round 4 (student-r4", text)
        self.assertNotIn("Users", text)
        empty = al.write_run_plateau(self.tmp, NOW - 11 * 86400, 10.0, [], now=NOW).read_text(encoding="utf-8")
        self.assertIn("(no student round yet)", empty)

    def test_both_are_listed_like_any_alert_and_clear_when_the_file_is_removed(self):
        al.write_run_student(self.tmp, ROUNDS, now=NOW)
        al.write_run_plateau(self.tmp, NOW, 10.0, [], now=NOW)
        self.assertEqual({"run-student", "run-plateau"}, {a["kind"] for a in al.listing(self.tmp)})
        banner = al.banner(self.tmp)
        self.assertIn("run-student.txt: ALERT run-student", banner)
        self.assertIn("run-plateau.txt: ALERT run-plateau", banner)
        al.alert_path(self.tmp, "run-student").unlink()
        al.alert_path(self.tmp, "run-plateau").unlink()
        self.assertEqual("", al.status_prefix(self.tmp))

    def test_the_existing_failure_alert_is_unchanged(self):
        path = al.write_failure(self.tmp, "train", [{"job": "j1", "rc": 1, "secs": 5}, {"job": "j2", "rc": 1, "secs": 6}], now=time.time())
        self.assertIn("ALERT train: 2 failures in a row", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
