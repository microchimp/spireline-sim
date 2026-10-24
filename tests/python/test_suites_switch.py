"""tools.fidelity.suites_switch: the old suites' off/on table (the trx parse, the verdicts, the filter, the outputs).

dotnet is not run: a made-up trx pair stands in for the two runs.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.fidelity import suites_switch as S  # noqa: E402

_TRX = """<?xml version="1.0" encoding="utf-8"?>
<TestRun xmlns="http://microsoft.com/schemas/VisualStudio/TeamTest/2010">
  <Results>
{results}
  </Results>
  <TestDefinitions>
{defs}
  </TestDefinitions>
</TestRun>
"""


def _trx(path: Path, outcomes: dict) -> Path:
    results, defs = [], []
    for i, (name, (suite, outcome, msg)) in enumerate(outcomes.items()):
        err = (f"<Output><ErrorInfo><Message>{msg}</Message></ErrorInfo></Output>" if msg else "")
        results.append(f'<UnitTestResult testId="id{i}" testName="{name}" outcome="{outcome}">{err}</UnitTestResult>')
        defs.append(f'<UnitTest id="id{i}" name="{name}"><TestMethod className="Sts2Emulator.Tests.{suite}" '
                    f'name="{name}"/></UnitTest>')
    path.write_text(_TRX.format(results="\n".join(results), defs="\n".join(defs)), encoding="utf-8")
    return path


class SuitesSwitchTests(unittest.TestCase):
    def test_verdicts(self):
        self.assertEqual(S.verdict("Passed", "Passed"), "pass both")
        self.assertEqual(S.verdict("Passed", "Failed"), "fail on only")
        self.assertEqual(S.verdict("Failed", "Passed"), "fail off only")
        self.assertEqual(S.verdict("Failed", "Failed"), "fail both")
        self.assertEqual(S.verdict("NotExecuted", "NotExecuted"), "skipped")
        self.assertEqual(S.verdict("Passed", None), "missing on")

    def test_filter_names_each_class_dotted(self):
        self.assertEqual(S.test_filter(["A", "B"]), "FullyQualifiedName~.A.|FullyQualifiedName~.B.")
        self.assertEqual(len(S.SUITES), len(set(S.SUITES)))

    def test_parse_and_compare_scrub_paths(self):
        with tempfile.TemporaryDirectory() as d:
            off = S.parse_trx(_trx(Path(d) / "off.trx", {
                "X.T.a": ("T", "Passed", ""), "X.T.b": ("T", "Passed", ""), "X.U.c(n: 1)": ("U", "Failed", "boom")}))
            on = S.parse_trx(_trx(Path(d) / "on.trx", {
                "X.T.a": ("T", "Passed", ""), "X.T.b": ("T", "Failed", "Assert.True() Failure at C:\\w\\x\\f.cs\nmore"),
                "X.U.c(n: 1)": ("U", "Failed", "boom")}))
        rows = S.compare(off, on)
        by = {r["test"]: r for r in rows}
        self.assertEqual(by["X.T.a"]["verdict"], "pass both")
        self.assertEqual(by["X.T.b"]["verdict"], "fail on only")
        self.assertEqual(by["X.T.b"]["suite"], "T")
        self.assertEqual(by["X.T.b"]["message"], "Assert.True() Failure at <path>")
        self.assertEqual(by["X.U.c(n: 1)"]["verdict"], "fail both")
        self.assertEqual(S.counts(rows), {"fail both": 1, "fail on only": 1, "pass both": 1})
        md = S.markdown("F08,F10", ["T", "U"], rows)
        self.assertIn("| T | 2 | 1 | 1 | 0 | 0 |", md)
        self.assertIn("| X.T.b | fail on only |", md)

    def test_main_runs_off_then_on_and_writes_the_table(self):
        seen = []

        def fake_run(cmd, cwd, env, stdout, stderr):
            on = env.get(S.VARIABLE)
            seen.append((cmd[1], on))
            if cmd[1] == "test":
                d = Path(cmd[cmd.index("--results-directory") + 1])
                name = cmd[cmd.index("--logger") + 1].split("=")[1]
                _trx(d / name, {"X.T.a": ("T", "Failed" if on else "Passed", "x")})
            return mock.Mock(returncode=0)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(S.subprocess, "run", fake_run):
            self.assertEqual(S.main(["--out", d, "--suites", "T"]), 0)
            doc = json.loads((Path(d) / "suites_switch.json").read_text(encoding="utf-8"))
        self.assertEqual(seen, [("build", None), ("test", None), ("test", "F08,F10")])
        self.assertEqual(doc["counts"], {"fail on only": 1})
        self.assertEqual(doc["suites"], ["T"])


if __name__ == "__main__":
    unittest.main()
