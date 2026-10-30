"""tools/py_tests.py: the Python suite in a fast and a slow tier, from one list (tests/python/slow_tier.txt).

Each case runs the entry point as a process over a scratch tests folder of tiny modules, so the exit codes and the lines
the gates read (FAIL:/ERROR: ids, Ran N tests, OK/FAILED) are the real ones.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools import py_tests as pt  # noqa: E402

GOOD = "import unittest\nclass T(unittest.TestCase):\n    def test_ok(self):\n        pass\n    def test_ok2(self):\n        pass\n"
BAD = textwrap.dedent("""
    import unittest
    class T(unittest.TestCase):
        def test_ok(self):
            pass
        def test_bad(self):
            \"\"\"A docstring: the runner prints it on the second line.\"\"\"
            self.assertEqual(1, 2)
""")


class Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "tests"
        self.dir.mkdir()
        self.slow = Path(self.tmp.name) / "slow.txt"

    def module(self, name: str, text: str = GOOD) -> None:
        (self.dir / f"{name}.py").write_text(text, encoding="utf-8")

    def run_tier(self, tier: str, *extra: str, slow: str | None = None) -> subprocess.CompletedProcess:
        if slow is not None:
            self.slow.write_text(slow, encoding="utf-8")
        return subprocess.run([sys.executable, str(ROOT / "tools" / "py_tests.py"), tier, "--tests-dir", str(self.dir), "--slow-list", str(self.slow), *extra],
                              capture_output=True, text=True, timeout=120)

    def modules(self, tier: str, slow: str | None = None) -> list[str]:
        p = self.run_tier(tier, "--list", slow=slow)
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout.split()


class TiersTest(Scratch):
    def setUp(self):
        super().setUp()
        for m in ("test_a", "test_b", "test_c"):
            self.module(m)
        (self.dir / "helper.py").write_text("X = 1\n", encoding="utf-8")  # not a test module

    def test_the_tiers_split_the_modules_by_the_slow_list_and_all_is_both(self):
        text = "# the slow tier\ntest_b  # 85 s\n\n"
        self.assertEqual(self.modules("slow", text), ["test_b"])
        self.assertEqual(self.modules("fast", text), ["test_a", "test_c"])
        self.assertEqual(sorted(self.modules("all", text)), ["test_a", "test_b", "test_c"])

    def test_moving_a_module_is_editing_one_line(self):
        self.assertEqual(self.modules("fast", "test_a\n"), ["test_b", "test_c"])
        self.assertEqual(self.modules("fast", "test_c\n"), ["test_a", "test_b"])
        self.assertEqual(self.modules("fast", ""), ["test_a", "test_b", "test_c"])

    def test_a_tier_that_ran_prints_the_lines_the_gates_read(self):
        p = self.run_tier("fast", slow="test_b\n")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertRegex(p.stderr, r"(?m)^Ran 4 tests in ")
        self.assertRegex(p.stderr, r"(?m)^OK$")
        self.assertIn("tier fast: 2 modules, 4 tests", p.stderr)

    def test_a_slow_list_that_names_a_missing_module_or_an_empty_tier_exits_2(self):
        p = self.run_tier("slow", slow="test_b\ntest_nope\n")
        self.assertEqual(p.returncode, 2)
        self.assertIn("test_nope", p.stderr)
        self.assertEqual(self.run_tier("slow", slow="").returncode, 2)  # nothing listed: no slow tier to run
        self.assertEqual(self.run_tier("fast", slow="test_a\ntest_b\ntest_c\n").returncode, 2)

    def test_the_timing_file_names_every_module_that_ran_slowest_first(self):
        out = Path(self.tmp.name) / "t.json"
        self.module("test_c", "import time, unittest\nclass T(unittest.TestCase):\n    def test_s(self):\n        time.sleep(0.3)\n")
        p = self.run_tier("all", "--timing", str(out), slow="")
        self.assertEqual(p.returncode, 0, p.stderr)
        doc = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(doc["tier"], "all")
        self.assertEqual(sorted(doc["seconds"]), ["test_a", "test_b", "test_c"])
        self.assertEqual(next(iter(doc["seconds"])), "test_c")
        self.assertGreaterEqual(doc["seconds"]["test_c"], 0.3)

    def test_durations_are_passed_to_unittest(self):
        p = self.run_tier("fast", "--durations", "2", slow="test_b\n")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("Slowest test durations", p.stderr)


class FailuresTest(Scratch):
    def setUp(self):
        super().setUp()
        self.module("test_a")
        self.module("test_bad", BAD)

    def test_a_failure_exits_1_with_the_id_the_gates_grep(self):
        p = self.run_tier("fast", slow="")
        self.assertEqual(p.returncode, 1)
        self.assertRegex(p.stderr, r"(?m)^FAIL: test_bad \(test_bad\.T\.test_bad\)$")
        self.assertRegex(p.stderr, r"(?m)^FAILED \(failures=1\)$")

    def test_a_failure_in_the_baseline_passes_and_a_new_one_does_not(self):
        base = Path(self.tmp.name) / "baseline.txt"
        base.write_bytes(b"test_bad (test_bad.T.test_bad)\r\n")  # the gates' file: ids without the prefix, CRLF on Windows
        p = self.run_tier("fast", "--baseline", str(base), slow="")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("failures: 1, beyond the baseline: 0", p.stderr)
        base.write_text("test_other (test_other.T.test_other)\n", encoding="utf-8")
        p = self.run_tier("fast", "--baseline", str(base), slow="")
        self.assertEqual(p.returncode, 1)
        self.assertIn("beyond the baseline: 1", p.stderr)
        p = self.run_tier("fast", "--baseline", str(Path(self.tmp.name) / "nope.txt"), slow="")
        self.assertEqual(p.returncode, 1)  # a baseline that cannot be read is no pass

    def test_a_module_that_does_not_import_is_an_error_in_the_fast_tier(self):
        self.module("test_broken", "import no_such_module_anywhere\n")
        self.assertIn("test_broken", self.modules("fast", ""))
        p = self.run_tier("fast", slow="")
        self.assertEqual(p.returncode, 1)
        self.assertRegex(p.stderr, r"(?m)^ERROR: test_broken ")
        self.assertNotIn("test_broken", self.modules("fast", "test_broken\n"))  # listed slow: it moves with its line

    def test_only_the_failing_tier_fails(self):
        self.assertEqual(self.run_tier("fast", slow="test_bad\n").returncode, 0)
        self.assertEqual(self.run_tier("slow", slow="test_bad\n").returncode, 1)


class RealListTest(unittest.TestCase):
    """tests/python/slow_tier.txt itself."""

    def test_every_listed_module_is_a_test_file_listed_once(self):
        names = pt.read_slow_list(pt.SLOW_LIST)
        self.assertEqual(len(names), len(set(names)), "a module is listed twice")
        for n in names:
            self.assertRegex(n, r"^test_\w+$")
            self.assertTrue((HERE / f"{n}.py").is_file(), f"slow_tier.txt names {n}, which is not under tests/python")

    def test_the_slow_list_does_not_hold_this_file(self):
        self.assertNotIn("test_py_tests", pt.read_slow_list(pt.SLOW_LIST), "the tier machinery's own test runs in every landing's fast tier")

    def test_every_line_carries_its_measured_seconds(self):
        for line in pt.SLOW_LIST.read_text(encoding="utf-8").splitlines():
            if line.split("#", 1)[0].strip():
                self.assertRegex(line, r"#\s*\d+ s", f"no measured seconds on: {line}")

    def test_the_reader_skips_notes_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.txt"
            p.write_text("# head\n\ntest_x  # 70 s\n  test_y\n#test_z\n", encoding="utf-8")
            self.assertEqual(pt.read_slow_list(p), ["test_x", "test_y"])
            self.assertEqual(pt.read_slow_list(Path(d) / "absent.txt"), [])


class SelectTest(unittest.TestCase):
    def test_select_keeps_discovery_order(self):
        mods = ["a", "b", "c", "d"]
        self.assertEqual(pt.select(mods, ["d", "b"], "slow"), ["b", "d"])
        self.assertEqual(pt.select(mods, ["d", "b"], "fast"), ["a", "c"])
        self.assertEqual(pt.select(mods, ["d", "b"], "all"), mods)

    def test_the_failure_ids_match_the_printed_form(self):
        class T(unittest.TestCase):
            def test_f(self):
                self.fail("x")

            def test_e(self):
                raise RuntimeError("x")
        r = unittest.TestResult()
        unittest.defaultTestLoader.loadTestsFromTestCase(T).run(r)
        ids = pt.failure_ids(r)
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(re.match(r"^(FAIL|ERROR): test_[fe] \(", i) for i in ids))


if __name__ == "__main__":
    unittest.main()
