#!/usr/bin/env python3
"""The Python suite in two tiers, one entry point: every landing runs the fast tier, a nightly lab job runs the slow one.

    python tools/py_tests.py fast|slow|all [--durations N] [--timing FILE] [--baseline FILE] [--require-aot] [--list]
                             [--tests-dir DIR] [--slow-list FILE]

The slow tier is the modules named in tests/python/slow_tier.txt (one module name per line, text after # is a note:
the seconds it was measured at); the fast tier is every other module under tests/python; `all` is both, which is what
`python -m unittest discover -s tests/python` ran before. The modules are found by unittest's own discovery from the
repository root, so a test's id (`test_x (test_module.Class.test_x)`) and the OK/FAILED lines are the ones discover prints and
the gates' failure comparison reads them unchanged. A module that fails to import is reported in the fast tier (and in the
slow one when it is listed there). Moving a test between tiers is editing one line of slow_tier.txt.

  --durations N   unittest's slowest-N list (Python 3.12).
  --timing FILE   write {module: seconds} (setUpClass and teardown included) for the modules that ran, slowest first: the
                  numbers slow_tier.txt is kept from. Run `all` with it on a quiet machine.
  --baseline FILE the file of known failures (one `FAIL: ...` / `ERROR: ...` id per line, as the gates keep it): exit 0 when
                  every failure is in it, 1 on a failure that is not. Without it any failure exits 1.
  --require-aot   set STS2_REQUIRE_AOT=1 (the native tests fail instead of skipping when the library is not published).
  --list          print the modules of the tier and exit.

Exit status: 0 passed, 1 failed (beyond the baseline), 2 the slow list names a module that does not exist or the tier is empty.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests" / "python"
SLOW_LIST = TESTS / "slow_tier.txt"
TIERS = ("fast", "slow", "all")


def read_slow_list(path: Path) -> list[str]:
    """The module names in a slow-tier file: one per line, `#` starts a note, blank lines are skipped."""
    out: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        name = line.split("#", 1)[0].strip()
        if name:
            out.append(name)
    return out


def flatten(suite: unittest.TestSuite):
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            yield from flatten(t)
        else:
            yield t


def module_of(test: unittest.TestCase) -> str:
    """The test module a test belongs to; for a module that failed to import, its name (what unittest keeps as the method name)."""
    if type(test).__name__ == "_FailedTest":
        return str(getattr(test, "_testMethodName", "?"))
    return type(test).__module__


def discover(tests_dir: Path) -> dict[str, list[unittest.TestCase]]:
    """Every test under tests_dir by module, in unittest's discovery order."""
    top = str(tests_dir)
    suite = unittest.TestLoader().discover(top, pattern="test*.py", top_level_dir=top)
    by: dict[str, list[unittest.TestCase]] = {}
    for t in flatten(suite):
        by.setdefault(module_of(t), []).append(t)
    return by


def select(modules: list[str], slow: list[str], tier: str) -> list[str]:
    """The modules of a tier, in discovery order: the slow ones, the rest, or all."""
    if tier == "all":
        return list(modules)
    return [m for m in modules if (m in slow) == (tier == "slow")]


class TimedSuite(unittest.TestSuite):
    """The tests of one module, timed as a whole (fixtures included) into `sink`."""

    def __init__(self, name: str, tests, sink: dict[str, float]):
        super().__init__(tests)
        self.name, self.sink = name, sink

    def run(self, result, debug=False):  # noqa: FBT002
        t0 = time.perf_counter()
        try:
            return super().run(result, debug)
        finally:
            self.sink[self.name] = self.sink.get(self.name, 0.0) + time.perf_counter() - t0


def failure_ids(result: unittest.TestResult) -> list[str]:
    """`FAIL: <test>` / `ERROR: <test>` for every failure and error, as the runner prints them and the gates compare them."""
    out = [f"FAIL: {t}" for t, _ in result.failures] + [f"ERROR: {t}" for t, _ in result.errors]
    return sorted(set(out))


def read_baseline(path: Path) -> set[str]:
    return {ln.strip() for ln in path.read_text(encoding="utf-8").replace("\r", "").splitlines() if ln.strip()}


def baseline_key(line: str) -> str:
    """The failure id without its FAIL:/ERROR: prefix (the gates' baseline file keeps the ids that way)."""
    return re.sub(r"^(FAIL|ERROR): ", "", line)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tier", choices=TIERS)
    ap.add_argument("--durations", type=int, default=None, metavar="N")
    ap.add_argument("--timing", type=Path, default=None, metavar="FILE")
    ap.add_argument("--baseline", type=Path, default=None, metavar="FILE")
    ap.add_argument("--require-aot", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tests-dir", type=Path, default=TESTS)
    ap.add_argument("--slow-list", type=Path, default=None)
    a = ap.parse_args(argv)

    tests_dir = a.tests_dir.resolve()
    slow = read_slow_list(a.slow_list if a.slow_list else tests_dir / "slow_tier.txt")
    sys.dont_write_bytecode = True
    os.chdir(ROOT)  # the tests are written for the repository root as the working directory ...
    sys.path.insert(0, str(ROOT))  # ... and, as under `python -m unittest`, importing rl/, tools/ and scripts/ from it
    if a.require_aot:
        os.environ["STS2_REQUIRE_AOT"] = "1"
    by = discover(tests_dir)
    missing = [m for m in slow if m not in by]
    if missing:
        print(f"slow-tier list names modules that are not under {tests_dir.name}: {', '.join(missing)}", file=sys.stderr)
        return 2
    chosen = select(list(by), slow, a.tier)
    if a.list:
        print("\n".join(chosen))
        return 0
    if not chosen:
        print(f"the {a.tier} tier holds no tests", file=sys.stderr)
        return 2
    timings: dict[str, float] = {}
    suite = unittest.TestSuite(TimedSuite(m, by[m], timings) for m in chosen)
    print(f"tier {a.tier}: {len(chosen)} modules, {sum(len(by[m]) for m in chosen)} tests", file=sys.stderr, flush=True)
    kw = {"durations": a.durations} if a.durations else {}
    runner = unittest.TextTestRunner(stream=sys.stderr, verbosity=1, **kw)
    result = runner.run(suite)
    if a.timing:
        a.timing.parent.mkdir(parents=True, exist_ok=True)
        a.timing.write_text(json.dumps({"tier": a.tier, "seconds": {k: round(v, 2) for k, v in sorted(timings.items(), key=lambda kv: -kv[1])}},
                                       indent=1) + "\n", encoding="utf-8")
    fails = failure_ids(result)
    if a.baseline:
        try:
            known = {baseline_key(b) for b in read_baseline(a.baseline)}
        except OSError as e:
            print(f"baseline unreadable: {type(e).__name__}", file=sys.stderr)
            return 1
        new = [f for f in fails if baseline_key(f) not in known]
        print(f"failures: {len(fails)}, beyond the baseline: {len(new)}", file=sys.stderr)
        return 1 if new else 0
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
