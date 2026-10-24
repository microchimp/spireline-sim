"""tools.fidelity.suites_switch: the old suites' off/on table (the trx parse, the verdicts, coverage, crashed runs, the
filter, the outputs).

dotnet is not run: made-up trx files and a coverage file stand in for the two runs.
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
  <ResultSummary outcome="Failed"><RunInfos>{info}</RunInfos></ResultSummary>
  <Results>
{results}
  </Results>
  <TestDefinitions>
{defs}
  </TestDefinitions>
</TestRun>
"""

_CRASH = "The active test run was aborted. Reason: Test host process crashed"


def _trx(path: Path, outcomes: dict, info: str = "") -> Path:
    results, defs = [], []
    for i, (name, (suite, outcome, msg)) in enumerate(outcomes.items()):
        err = (f"<Output><ErrorInfo><Message>{msg}</Message></ErrorInfo></Output>" if msg else "")
        results.append(f'<UnitTestResult testId="id{i}" testName="{name}" outcome="{outcome}">{err}</UnitTestResult>')
        defs.append(f'<UnitTest id="id{i}" name="{name}"><TestMethod className="Sts2Emulator.Tests.{suite}" '
                    f'name="{name}"/></UnitTest>')
    run_info = f'<RunInfo computerName="x" outcome="Error"><Text>{info}</Text></RunInfo>' if info else ""
    path.write_text(_TRX.format(results="\n".join(results), defs="\n".join(defs), info=run_info), encoding="utf-8")
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
            off_trx = _trx(Path(d) / "off.trx", {
                "X.T.a": ("T", "Passed", ""), "X.T.b": ("T", "Passed", ""), "X.U.c(n: 1)": ("U", "Failed", "boom")})
            off = S.parse_trx(off_trx)
            on = S.parse_trx(_trx(Path(d) / "on.trx", {
                "X.T.a": ("T", "Passed", ""), "X.T.b": ("T", "Failed", "Assert.True() Failure at C:\\w\\x\\f.cs\nmore"),
                "X.U.c(n: 1)": ("U", "Failed", "boom")}))
            self.assertFalse(S.aborted(off_trx))
        rows = S.compare(off, on)
        by = {r["test"]: r for r in rows}
        self.assertEqual(by["X.T.a"]["verdict"], "pass both")  # no coverage file given: no coverage verdict
        self.assertEqual(by["X.T.b"]["verdict"], "fail on only")
        self.assertEqual(by["X.T.b"]["suite"], "T")
        self.assertEqual(by["X.T.b"]["message"], "Assert.True() Failure at <path>")
        self.assertEqual(by["X.U.c(n: 1)"]["verdict"], "fail both")
        self.assertEqual(S.counts(rows), {"fail both": 1, "fail on only": 1, "pass both": 1})
        md = S.markdown("F08,F10", ["T", "U"], rows)
        self.assertIn("| T | 2 | 1 | 0 | 1 | 0 | 0 |", md)
        self.assertIn("| X.T.b | fail on only |", md)

    def test_a_pass_with_no_bound_step_is_not_covered(self):
        off = {n: {"suite": "T", "outcome": "Passed", "message": ""} for n in ("N.T.a", "N.T.b(x: 1)", "N.T.b(x: 2)")}
        on = {n: dict(v) for n, v in off.items()}
        rows = S.compare(off, on, {"N.T.b": 3})
        by = {r["test"]: r for r in rows}
        self.assertEqual((by["N.T.a"]["verdict"], by["N.T.a"]["bound_steps"]), ("not covered", 0))
        # A theory's cases share their method's count.
        self.assertEqual((by["N.T.b(x: 2)"]["verdict"], by["N.T.b(x: 2)"]["bound_steps"]), ("pass both", 3))
        self.assertEqual(S.uncovered(rows), {"T": (1, 3)})
        self.assertIn("Not covered (no kernel-bound step with the switch on): T (1 of 3)", S.markdown("F10", ["T"], rows))
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "c.txt").write_text("N.T.b\nN.T.b\n\nN.T.c\n", encoding="utf-8")
            self.assertEqual(S.read_coverage(Path(d) / "c.txt"), {"N.T.b": 2, "N.T.c": 1})
            self.assertEqual(S.read_coverage(Path(d) / "none.txt"), {})

    def _main(self, on_writes):
        """main() on a fake dotnet: the off run passes X.T.a and X.T.b; the on run writes what `on_writes` does."""
        seen = []

        def fake_run(cmd, cwd, env, stdout, stderr):
            seen.append((cmd[1], env.get(S.VARIABLE), bool(env.get(S.COVERAGE))))
            if cmd[1] == "test":
                d = Path(cmd[cmd.index("--results-directory") + 1])
                name = cmd[cmd.index("--logger") + 1].split("=")[1]
                if env.get(S.VARIABLE):
                    on_writes(d / name, Path(env[S.COVERAGE]))
                else:
                    _trx(d / name, {"X.T.a": ("T", "Passed", ""), "X.T.b": ("T", "Passed", "")})
            return mock.Mock(returncode=0)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(S.subprocess, "run", fake_run):
            code = S.main(["--out", d, "--suites", "T"])
            doc = json.loads((Path(d) / "suites_switch.json").read_text(encoding="utf-8"))
            md = (Path(d) / "suites_switch.md").read_text(encoding="utf-8")
        return code, doc, md, seen

    def test_main_runs_off_then_on_and_writes_the_table(self):
        def on(trx, cov):
            _trx(trx, {"X.T.a": ("T", "Failed", "x"), "X.T.b": ("T", "Passed", "")})
            cov.write_text("X.T.b\nX.T.b\n", encoding="utf-8")

        code, doc, md, seen = self._main(on)
        self.assertEqual(code, 0)
        self.assertEqual(seen, [("build", None, False), ("test", None, False), ("test", "F08,F10", True)])
        self.assertEqual(doc["counts"], {"fail on only": 1, "pass both": 1})
        self.assertEqual({r["test"]: r["bound_steps"] for r in doc["rows"]}, {"X.T.a": 0, "X.T.b": 2})
        self.assertEqual((doc["suites"], doc["crashed"], doc["uncovered"]), (["T"], [], {}))

    def test_a_crashed_on_run_is_reported_and_fails_the_runner(self):
        # The test host died after one test: the trx is partial and its run info names the crash.
        code, doc, md, _ = self._main(lambda trx, cov: _trx(trx, {"X.T.a": ("T", "Passed", "")}, info=_CRASH))
        self.assertEqual(code, 1)
        self.assertEqual(doc["crashed"], ["on"])
        self.assertEqual({r["test"]: r["verdict"] for r in doc["rows"]}, {"X.T.a": "not covered", "X.T.b": "missing on"})
        self.assertIn("Crashed or incomplete run: on", md)
        # It died before the logger wrote anything: every test is missing from the on run.
        code, doc, _, _ = self._main(lambda trx, cov: None)
        self.assertEqual((code, doc["crashed"]), (1, ["on"]))
        self.assertEqual({r["verdict"] for r in doc["rows"]}, {"missing on"})
        # No crash text, but a test lost: still not a clean table.
        code, doc, _, _ = self._main(lambda trx, cov: _trx(trx, {"X.T.a": ("T", "Passed", "")}))
        self.assertEqual((code, doc["crashed"]), (1, ["on lost tests"]))


if __name__ == "__main__":
    unittest.main()
