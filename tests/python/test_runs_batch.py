"""referee/runs.py: a whole-run batch survives a crashed game session, and run_compare2 --json says where each run diverged.

The game, EmuDump and run_referee.py are replaced by stand-ins that print the reports and write the result files the real
ones do; what is checked is what runs.py keeps, what it logs and which exit code it gives.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runs  # noqa: E402
from tools.fidelity import run_compare2 as rc2  # noqa: E402

CLEAN = {"exit": 0, "seconds": 5, "network_connections": [], "real_profile_touched": False, "scratch": "s"}
CRASH = {**CLEAN, "exit": 3221225477}  # 0xC0000005
NET = {**CLEAN, "network_connections": ["1.2.3.4:443"]}


def report_text(r: dict) -> str:
    return "noise before\n" + json.dumps(r, indent=2) + "\n"


class FakeProc:
    def __init__(self, report: dict | None):
        self.report = report

    def communicate(self):
        return (report_text(self.report) if self.report is not None else "the game died without a report\n"), None


class RunsBatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def go(self, reports: list, written: dict, procs=2, n=6, name="r0", extra=()):
        """Run main() with fake sessions: session p returns reports[p] and has written the results in written[p]."""
        launched = []

        def popen(cmd, **kw):
            job = json.loads(Path(cmd[cmd.index("--job") + 1]).read_text(encoding="utf-8"))
            p = len(launched)
            if p in written:
                Path(job["out"]).write_text(json.dumps({"results": [{"seed": r["seed"], "decisions": [], "end": "win"} for r in job["runs"][: written[p]]]}),
                                            encoding="utf-8")
            launched.append(job)
            return FakeProc(reports[p])

        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            if "run_compare2" in " ".join(map(str, cmd)):
                return mock.Mock(returncode=1, stdout="compared\n", stderr="")
            kw["stdout"].write("{}")  # EmuDump
            return mock.Mock(returncode=0)

        argv = ["runs.py", "--name", name, "--runs", str(n), "--procs", str(procs), "--jobs", str(self.tmp), "--game", "g", *extra]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(runs.subprocess, "Popen", popen), \
                mock.patch.object(runs.subprocess, "run", run), mock.patch.object(runs.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = runs.main()
        return rc, launched, calls

    def merged(self, name="r0"):
        return json.loads((self.tmp / f"{name}.out.json").read_text(encoding="utf-8"))["results"]

    def test_clean_sessions_are_compared_and_the_comparisons_code_is_returned(self):
        rc, launched, calls = self.go([CLEAN, CLEAN], {0: 3, 1: 3})
        self.assertEqual(1, rc)  # run_compare2 found divergences
        self.assertEqual(2, len(launched))
        self.assertEqual(6, len(self.merged()))
        self.assertTrue(all("decisions" in r for r in self.merged()))
        self.assertTrue(any("dll" in str(c).lower() or "EmuDump" in str(c) for c in calls))
        self.assertTrue(any("--json" in c for c in calls))
        text = (self.tmp / "r0.summary.txt").read_text(encoding="utf-8")
        self.assertIn("compared", text)

    def test_a_crashed_session_is_logged_and_the_rest_is_compared(self):
        rc, launched, calls = self.go([CLEAN, CRASH], {0: 3, 1: 1})  # session 1 wrote one of its three runs before it died
        self.assertEqual(runs.EXIT_CRASHED, rc)
        merged = self.merged()
        self.assertEqual(6, len(merged))
        # session 0 played specs 0, 2, 4 and session 1 specs 1, 3, 5; session 1 wrote only spec 1 before it died
        self.assertEqual([True, True, True, False, True, False], ["decisions" in r for r in merged])
        self.assertEqual("no result (session ended)", merged[3]["error"])
        text = (self.tmp / "r0.summary.txt").read_text(encoding="utf-8")
        self.assertIn("session 1 FAILED, the batch goes on", text)
        self.assertIn("1 of its 3 runs were kept", text)
        self.assertNotIn("session 0 FAILED", text)
        self.assertTrue(any("run_compare2" in " ".join(map(str, c)) for c in calls))  # the replay and the comparison ran

    def test_every_session_dead_without_a_run_is_exit_two_and_nothing_is_replayed(self):
        rc, _, calls = self.go([CRASH, CRASH], {})
        self.assertEqual(runs.EXIT_NOTHING, rc)
        self.assertEqual([], calls)
        self.assertTrue(all("error" in r for r in self.merged()))
        self.assertIn("no run was recorded", (self.tmp / "r0.summary.txt").read_text(encoding="utf-8"))

    def test_a_failed_safety_check_stops_before_the_comparison_but_keeps_the_runs(self):
        rc, _, calls = self.go([CLEAN, NET], {0: 3, 1: 3})
        self.assertEqual(runs.EXIT_SAFETY, rc)
        self.assertEqual([], calls)
        self.assertEqual(6, len(self.merged()))  # the batch is not lost
        text = (self.tmp / "r0.summary.txt").read_text(encoding="utf-8")
        self.assertIn("SAFETY CHECK FAILED", text)
        self.assertIn("1.2.3.4:443", text)

    def test_a_session_with_no_report_at_all_is_a_safety_failure(self):
        rc, _, calls = self.go([CLEAN, None], {0: 3, 1: 2})
        self.assertEqual(runs.EXIT_SAFETY, rc)
        self.assertEqual([], calls)
        self.assertIn("network_connections missing", (self.tmp / "r0.summary.txt").read_text(encoding="utf-8"))

    def test_merge_puts_each_sessions_runs_back_in_spec_order(self):
        specs = [{"seed": f"S{i}"} for i in range(5)]
        got = runs.merge_results([[{"seed": "S0", "decisions": []}, {"seed": "S2", "decisions": []}], [{"seed": "S1", "decisions": []}]], 5, 2, specs)
        self.assertEqual(["S0", "S1", "S2", "S3", "S4"], [r["seed"] for r in got])
        self.assertEqual([False, False, False, True, True], ["error" in r for r in got])
        self.assertEqual(3, len(runs.merge_results([[], []], 3, 2, specs)))

    def test_unreadable_session_files_are_empty_not_errors(self):
        bad = self.tmp / "bad.json"
        bad.write_text('{"results": [{"seed": "S0", "deci', encoding="utf-8")  # cut off by a crash
        self.assertEqual([], runs._read_results(bad))
        self.assertEqual([], runs._read_results(self.tmp / "missing.json"))

    def test_jobs_folder_is_created_and_used_for_every_output(self):
        sub = self.tmp / "deeper" / "runs"
        argv = ["runs.py", "--name", "x", "--runs", "2", "--procs", "1", "--jobs", str(sub), "--game", "g"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(runs.subprocess, "Popen", lambda cmd, **kw: FakeProc(CRASH)), \
                mock.patch.object(runs.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            runs.main()
        self.assertEqual(["x.out.json", "x.p0.json", "x.summary.txt"], sorted(p.name for p in sub.iterdir()))


class RunCompareJsonTest(unittest.TestCase):
    def test_one_record_per_run(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        refs = {"results": [{"seed": "A", "decisions": [], "end": "win", "floor": 51}, {"seed": "B", "error": "no result (session ended)"}]}
        emus = {"results": [{"seed": "A", "decisions": []}, {"seed": "B", "error": "no run recorded"}]}
        (tmp / "r.json").write_text(json.dumps(refs), encoding="utf-8")
        (tmp / "e.json").write_text(json.dumps(emus), encoding="utf-8")
        got: list = []
        with contextlib.redirect_stdout(io.StringIO()):
            rc = rc2.run(str(tmp / "r.json"), str(tmp / "e.json"), False, None, got)
        self.assertEqual(1, rc)  # a run that was not recorded is not clean
        self.assertEqual([("A", True, None), ("B", False, None)], [(r["seed"], r["recorded"], r.get("first")) for r in got])
        self.assertEqual(51, got[0]["floor"])
        self.assertIn("no result", got[1]["error"])
        with contextlib.redirect_stdout(io.StringIO()):  # without collect, nothing changes
            self.assertEqual(1, rc2.run(str(tmp / "r.json"), str(tmp / "e.json"), False, None))

    def test_a_divergence_is_recorded_with_its_component_and_fields(self):
        from test_run_compare2_broken import run_pair  # noqa: PLC0415
        from broken_cases import run_main, write_json  # noqa: PLC0415
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        ref, emu = run_pair()
        wrecked = json.loads(json.dumps(emu))
        wrecked["decisions"][0]["snap"] = dict(wrecked["decisions"][0]["snap"], gold=1)
        for name, e in (("faithful", emu), ("wrecked", wrecked)):
            out = tmp / f"{name}.json"
            code, _ = run_main(rc2.main, ["run", write_json(tmp / "g.json", {"results": [ref]}), write_json(tmp / "e.json", {"results": [e]}),
                                         "--id-map", str(ROOT / "data" / "id_map.json"), "--json", str(out)])
            rec = json.loads(out.read_text(encoding="utf-8"))["runs"][0]
            self.assertEqual(("RUNBROKEN1", True, 3), (rec["seed"], rec["recorded"], rec["decisions"]))
            if name == "faithful":
                self.assertEqual((0, None), (code, rec["first"]))
            else:
                self.assertEqual(1, code)
                self.assertEqual({"decision", "component", "floor", "choice", "diffs"}, set(rec["first"]))
                self.assertEqual(0, rec["first"]["decision"])
                self.assertTrue(rec["first"]["diffs"] and all(len(x) == 3 for x in rec["first"]["diffs"]))


if __name__ == "__main__":
    unittest.main()
