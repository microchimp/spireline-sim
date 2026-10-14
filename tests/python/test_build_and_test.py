"""tools/build_and_test.sh FAILS when any step fails: each step broken in turn against a fake toolchain
(tests/python/fakes), a new Python failure, a suite that did not run; a skipped corpus check is no pass.
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
            "corpus build": dict(FAKE_BUILD="1"),
            "baseline file missing": dict(BASELINE_FAILURES=str(Path(self.tmp.name) / "nope.txt")),
            "python failure without a baseline": dict(BASELINE_FAILURES=""),
        }
        passed = {}
        for name, env in cases.items():
            code, res = self.gate(**env)
            if code != 1 or res["gate"] != "fail":
                passed[name] = (code, res["gate"])
        self.assertEqual(passed, {}, "broken steps build_and_test.sh let through")

    def test_a_new_python_failure_or_a_suite_that_did_not_run_fails(self):
        self.pylog_text("\n".join([*self.FAILS, "FAIL: test_new (m.T.test_new)"]) + "\n\nRan 9 tests\n\nFAILED (failures=2)\n")
        code, res = self.gate()
        self.assertEqual((code, res["python_new_failures_vs_baseline"]), (1, "1"))
        self.pylog_text("Traceback (most recent call last):\nImportError: boom\n")
        self.assertEqual(self.gate()[0], 1)

    def test_a_skipped_corpus_check_is_incomplete_not_a_pass(self):
        code, res = self.gate(CORPUS_CHECK="0")
        self.assertEqual((code, res["gate"]), (3, "incomplete"))


if __name__ == "__main__":
    unittest.main()
