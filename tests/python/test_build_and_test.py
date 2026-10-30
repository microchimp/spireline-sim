"""tools/build_and_test.sh FAILS when any step fails: each step broken in turn against a fake toolchain
(tests/python/fakes), a new Python failure, a suite that did not run; a skipped corpus check is no pass.
The Release EmuDump is built once, before the test steps, and a failed build fails the gate.
The C# suite, the Python suite and the corpus check run at once at BelowNormal (the EmuDump suite after the C# one, which it
would otherwise race on the library's Debug build); one failing step never stops the others, each keeps its log and exit code,
and results.json keeps its fields. PARALLEL=0 runs them one after another with the same verdicts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))




def find_bash() -> str | None:
    """Git's bash (never WSL's launcher in System32, which would run the script somewhere else)."""
    for p in (os.environ.get("SPIRELINE_BASH"), r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        if p and Path(p).exists():
            return p
    w = shutil.which("bash")
    return w if w and "system32" not in w.lower() else None


BASH = find_bash()


@unittest.skipUnless(BASH, "needs Git Bash")
class BuildAndTestBrokenCases(unittest.TestCase):
    """tools/build_and_test.sh in a scratch root against fake dotnet and Python (tests/python/fakes)."""

    FAILS = ["FAIL: test_a (m.T.test_a)", "ERROR: setUpClass (m.U)"]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "root"
        (self.root / "tools").mkdir(parents=True)
        shutil.copy(ROOT / "tools" / "build_and_test.sh", self.root / "tools" / "build_and_test.sh")
        self.bin = Path(self.tmp.name) / "bin"
        self.bin.mkdir()
        shutil.copy(HERE / "fakes" / "dotnet", self.bin / "dotnet")
        self.fakepy = (HERE / "fakes" / "python").as_posix()
        self.corpus = Path(self.tmp.name) / "corpus"
        self.corpus.mkdir()
        (self.corpus / "index.json").write_text("{}", encoding="utf-8")
        self.baseline = Path(self.tmp.name) / "baseline.txt"
        self.baseline.write_text("".join(sorted(f.split(": ", 1)[1] + "\n" for f in self.FAILS)), encoding="utf-8")
        self.pylog = Path(self.tmp.name) / "py.log"
        self.pylog_text("\n".join(self.FAILS) + "\n\nRan 9 tests in 0.1s\n\nFAILED (failures=1, errors=1)\n")

    def tearDown(self):
        self.tmp.cleanup()

    def pylog_text(self, text: str) -> None:
        self.pylog.write_bytes(text.replace("\n", "\r\n").encode())  # Windows Python writes CRLF

    def gate(self, **env) -> tuple[int, dict]:
        e = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ.get("PATH", ""), SPIRELINE_CORPUS=str(self.corpus),
                 BASELINE_FAILURES=str(self.baseline), FAKE_PY_LOG_FILE=str(self.pylog), FAKE_PY_EXIT="1")
        e.update(env)
        log = Path(self.tmp.name) / "log"
        shutil.rmtree(log, ignore_errors=True)
        p = subprocess.run([BASH, (self.root / "tools" / "build_and_test.sh").as_posix(), log.as_posix(), self.fakepy],
                           env=e, capture_output=True, text=True, timeout=120)
        res = json.loads((log / "results.json").read_text(encoding="utf-8"))
        return p.returncode, res

    def test_every_step_passing_exits_0(self):
        code, res = self.gate()
        self.assertEqual((code, res["gate"], res["gate_failures"]), (0, "pass", []), res)
        self.assertEqual(res["python_new_failures_vs_baseline"], "0")

    def test_each_broken_step_fails(self):
        cases = {
            "restore": dict(FAKE_RESTORE="1"),
            "publish": dict(FAKE_PUBLISH="1"),
            "csharp failures": dict(FAKE_TEST="1", FAKE_CS_SUMMARY="Failed!  - Failed:     2, Passed:     5, Skipped:     0, Total:     7"),
            "csharp summary missing": dict(FAKE_CS_SUMMARY=""),
            "csharp ran nothing": dict(FAKE_CS_SUMMARY="Passed!  - Failed:     0, Passed:     0, Skipped:     0, Total:     0"),
            "emudump.Tests failures": dict(FAKE_EMU_TEST="1", FAKE_EMU_SUMMARY="Failed!  - Failed:     1, Passed:     2, Skipped:     0, Total:     3"),
            "emudump.Tests summary missing": dict(FAKE_EMU_SUMMARY=""),
            "emudump.Tests exit without a failure line": dict(FAKE_EMU_TEST="1"),
            "corpus regression": dict(FAKE_CORPUS="1", FAKE_CORPUS_LINE="FAIL: 3 regressed fights"),
            "emudump build": dict(FAKE_BUILD="1"),
            "baseline file missing": dict(BASELINE_FAILURES=str(Path(self.tmp.name) / "nope.txt")),
            "python failure without a baseline": dict(BASELINE_FAILURES=""),
        }
        passed = {}
        for name, env in cases.items():
            code, res = self.gate(**env)
            if code != 1 or res["gate"] != "fail":
                passed[name] = (code, res["gate"])
        self.assertEqual(passed, {}, "broken steps build_and_test.sh let through")

    def test_the_release_emudump_is_built_once_before_the_python_suite_and_a_failed_build_fails(self):
        calls = Path(self.tmp.name) / "calls.log"
        code, res = self.gate(FAKE_CALL_LOG=str(calls))
        self.assertEqual((code, res["emudump_build_exit"]), (0, 0), res)
        lines = calls.read_text(encoding="utf-8").splitlines()
        builds = [i for i, ln in enumerate(lines) if ln.startswith("dotnet build ")]
        suites = [i for i, ln in enumerate(lines) if ln.startswith("python tools/py_tests.py fast")]
        corpus = [i for i, ln in enumerate(lines) if "corpus.py" in ln]
        self.assertEqual((len(builds), len(suites), len(corpus)), (1, 1, 1), lines)
        self.assertIn("tools/emudump/EmuDump.csproj -c Release", lines[builds[0]])
        self.assertLess(builds[0], suites[0], "the Python suite ran before the EmuDump build")
        self.assertLess(builds[0], corpus[0])
        tests = [i for i, ln in enumerate(lines) if ln.startswith("dotnet test ")]
        self.assertEqual(len(tests), 2, lines)
        self.assertLess(builds[0], min(tests), "no test step may start before the EmuDump build and its restore are done")
        restores = [i for i, ln in enumerate(lines) if ln.startswith("dotnet restore ")]
        self.assertLess(max(restores), min(tests + suites + corpus), "no step restores while another builds")
        code, res = self.gate(FAKE_BUILD="1")
        self.assertEqual((code, res["gate"], res["emudump_build_exit"]), (1, "fail", 1), res)
        self.assertEqual(res["gate_failures"], ["emudump-build", "corpus"], res)
        self.assertIn("EmuDump build failed", (Path(self.tmp.name) / "log" / "5-corpus-check.log").read_text(encoding="utf-8"))
        code, res = self.gate(FAKE_BUILD="1", CORPUS_CHECK="0")
        self.assertEqual((code, res["gate_failures"]), (1, ["emudump-build"]), res)

    def test_a_new_python_failure_or_a_suite_that_did_not_run_fails(self):
        self.pylog_text("\n".join([*self.FAILS, "FAIL: test_new (m.T.test_new)"]) + "\n\nRan 9 tests\n\nFAILED (failures=2)\n")
        code, res = self.gate()
        self.assertEqual((code, res["python_new_failures_vs_baseline"]), (1, "1"))
        self.pylog_text("Traceback (most recent call last):\nImportError: boom\n")
        self.assertEqual(self.gate()[0], 1)

    def test_a_skipped_corpus_check_is_incomplete_not_a_pass(self):
        code, res = self.gate(CORPUS_CHECK="0")
        self.assertEqual((code, res["gate"]), (3, "incomplete"))

    # ---- the parallel form

    def intervals(self, log: Path) -> dict[str, tuple[float, float]]:
        out = {}
        for line in log.read_text(encoding="utf-8").splitlines():
            name, s, e = line.split()
            out[name] = (float(s), float(e))
        return out

    def test_the_test_steps_run_at_once_and_the_two_dotnet_suites_one_after_the_other(self):
        log = Path(self.tmp.name) / "steps.log"
        code, res = self.gate(FAKE_STEP_LOG=str(log), FAKE_SLEEP="1.0", FAKE_PY_EXIT="0", BASELINE_FAILURES="")
        self.assertEqual((code, res["gate"]), (0, "pass"), res)
        iv = self.intervals(log)
        self.assertEqual(sorted(iv), ["corpus", "csharp", "emudump-tests", "python"])
        first = ("csharp", "python", "corpus")
        self.assertLess(max(iv[n][0] for n in first), min(iv[n][1] for n in first), "the C#, Python and corpus steps did not overlap")
        self.assertGreaterEqual(iv["emudump-tests"][0], iv["csharp"][1] - 0.05, "the EmuDump suite started inside the C# suite")
        timing = (Path(self.tmp.name) / "log" / "timing.log").read_text(encoding="utf-8").split("\n")
        self.assertEqual({ln.split()[0] for ln in timing if ln},
                         {"restore", "publish", "emudump-build", "csharp", "emudump-tests", "python", "corpus", "parallel-phase"})

    def test_serial_mode_runs_one_after_another_with_the_same_verdicts(self):
        log = Path(self.tmp.name) / "steps.log"
        code, res = self.gate(FAKE_STEP_LOG=str(log), FAKE_SLEEP="0.3", PARALLEL="0")
        self.assertEqual(code, 0, res)
        iv = self.intervals(log)
        order = sorted(iv, key=lambda n: iv[n][0])
        self.assertEqual(order, ["csharp", "emudump-tests", "python", "corpus"])
        for a, b in zip(order, order[1:]):
            self.assertGreaterEqual(iv[b][0], iv[a][1] - 0.05, f"{b} started inside {a}")
        code2, res2 = self.gate()
        res.pop("minutes"), res2.pop("minutes")
        self.assertEqual(res, res2)

    def test_a_failing_step_does_not_stop_the_others_and_every_step_keeps_its_log_and_exit_code(self):
        log = Path(self.tmp.name) / "steps.log"
        code, res = self.gate(FAKE_STEP_LOG=str(log), FAKE_SLEEP="0.5", FAKE_TEST="1",
                              FAKE_CS_SUMMARY="Failed!  - Failed:     2, Passed:     5, Skipped:     0, Total:     7")
        self.assertEqual((code, res["gate"], res["gate_failures"]), (1, "fail", ["csharp"]), res)
        self.assertEqual(sorted(self.intervals(log)), ["corpus", "csharp", "emudump-tests", "python"])
        d = Path(self.tmp.name) / "log"
        for name in ("3-dotnet-test", "3b-emudump-test", "4-python-test", "5-corpus-check"):
            self.assertTrue((d / f"{name}.log").is_file(), name)
            self.assertTrue((d / f"{name}.rc").is_file(), name)
        self.assertEqual((d / "3-dotnet-test.rc").read_text().strip(), "1")
        self.assertEqual((d / "3b-emudump-test.rc").read_text().strip(), "0")
        self.assertEqual((d / "5-corpus-check.rc").read_text().strip(), "0")
        self.assertEqual((res["dotnet_test_exit"], res["emudump_test_exit"], res["corpus_check_exit"]), (1, 0, "0"))

    def test_two_steps_failing_at_once_are_both_named(self):
        code, res = self.gate(FAKE_TEST="1", FAKE_CS_SUMMARY="Failed!  - Failed:     2, Passed:     5, Skipped:     0, Total:     7",
                              FAKE_CORPUS="1", FAKE_CORPUS_LINE="FAIL: 3 regressed fights")
        self.assertEqual((code, res["gate_failures"]), (1, ["csharp", "corpus"]), res)

    def test_results_json_keeps_its_fields(self):
        code, res = self.gate()
        self.assertEqual(code, 0, res)
        self.assertEqual(list(res), ["commit", "tree", "dotnet_sdk", "restore_exit", "publish_exit", "native_lib_sha256", "dotnet_test_exit",
                                     "tests_csharp", "emudump_test_exit", "tests_emudump", "emudump_build_exit", "python_test_exit",
                                     "tests_python", "python_new_failures_vs_baseline", "corpus_check_exit", "corpus_check", "gate",
                                     "gate_failures", "minutes"])
        self.assertEqual(res["restore_exit"], [0, 0, 0])

    def test_the_python_tier_is_fast_unless_asked(self):
        calls = Path(self.tmp.name) / "calls.log"
        self.gate(FAKE_CALL_LOG=str(calls))
        self.assertIn("python tools/py_tests.py fast", calls.read_text(encoding="utf-8"))
        self.assertTrue((Path(self.tmp.name) / "log" / "4-python-test.log").read_text(encoding="utf-8").startswith("tier: fast"))
        calls.unlink()
        self.gate(FAKE_CALL_LOG=str(calls), PY_TIER="all")
        self.assertIn("python tools/py_tests.py all", calls.read_text(encoding="utf-8"))

    def test_a_publish_that_failed_skips_the_python_suite_and_fails(self):
        code, res = self.gate(FAKE_PUBLISH="1")
        self.assertEqual((code, res["gate"], res["python_test_exit"]), (1, "fail", "skipped"), res)
        self.assertIn("publish", res["gate_failures"])

    @unittest.skipUnless(os.name == "nt", "Windows priority classes")
    def test_the_parallel_steps_run_below_the_priority_the_serial_ones_keep(self):
        """nice -n 10 under Git Bash is the Windows BelowNormal class: one step down from the caller's, so from Normal it is BelowNormal."""
        prio = Path(self.tmp.name) / "prio.log"
        env = dict(FAKE_PRIO_LOG=str(prio), FAKE_PRIO_PY=sys.executable, FAKE_PRIO_SCRIPT=str(HERE / "fakes" / "prio_class.py"))
        rank = {"0x40": 0, "0x4000": 1, "0x20": 2, "0x8000": 3, "0x80": 4}  # idle < below normal < normal < above normal < high
        self.gate(PARALLEL="0", **env)
        serial = dict(line.split() for line in prio.read_text(encoding="utf-8").splitlines())
        self.assertEqual(sorted(serial), ["corpus", "csharp", "emudump-tests", "python"])
        self.assertEqual(len(set(serial.values())), 1, serial)  # nothing in serial mode changes the caller's priority
        prio.unlink()
        self.gate(**env)
        par = dict(line.split() for line in prio.read_text(encoding="utf-8").splitlines())
        self.assertEqual(sorted(par), ["corpus", "csharp", "emudump-tests", "python"])
        base = next(iter(serial.values()))
        for step, cls in par.items():
            self.assertLessEqual(rank[cls], rank[base], (step, par, serial))
            if base == "0x20":
                self.assertEqual(cls, "0x4000", step)
        self.assertEqual(len(set(par.values())), 1, par)


if __name__ == "__main__":
    unittest.main()
