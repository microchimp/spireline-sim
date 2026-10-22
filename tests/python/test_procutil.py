"""src/sts2_gym/procutil.py: priority, parent liveness and child listing, on the platform running."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sts2_gym import procutil  # noqa: E402

SRC = str(ROOT / "src")


def run_py(code: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {SRC!r})\n" + code],
        capture_output=True,
        text=True,
        timeout=60,
        **kw,
    )


class PriorityTest(unittest.TestCase):
    def test_lower_priority_lowers_this_process_and_is_idempotent(self):
        probe = (
            "import os\n"
            "from sts2_gym import procutil as p\n"
            "p.lower_priority(10); p.lower_priority(10)\n"
            "print(p.priority_class() if p.IS_WINDOWS else os.nice(0))\n"
        )
        done = run_py(probe)
        self.assertEqual(done.returncode, 0, done.stderr)
        value = int(done.stdout.strip())
        if procutil.IS_WINDOWS:
            self.assertEqual(value, procutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            self.assertGreaterEqual(value, 10)

    def test_a_child_started_with_the_popen_kwargs_runs_lowered(self):
        probe = (
            "import os\n"
            "from sts2_gym import procutil as p\n"
            "print(p.priority_class() if p.IS_WINDOWS else os.nice(0))\n"
        )
        done = run_py(probe, **procutil.popen_priority_kwargs(10))
        self.assertEqual(done.returncode, 0, done.stderr)
        value = int(done.stdout.strip())
        if procutil.IS_WINDOWS:
            self.assertEqual(value, procutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            self.assertGreaterEqual(value, 10)


class ParentTest(unittest.TestCase):
    def test_parent_alive_for_this_process_and_a_dead_one(self):
        self.assertTrue(procutil.parent_alive(os.getpid()) if procutil.IS_WINDOWS else True)
        quick = subprocess.Popen([sys.executable, "-c", "pass"])
        quick.wait()
        self.assertFalse(procutil.parent_alive(quick.pid))

    def test_a_child_sees_its_parent_go(self):
        # parent P starts child C and prints C's pid, then exits; C reports when P is gone.
        child_code = (
            "import sys, time\n"
            f"sys.path.insert(0, {SRC!r})\n"
            "from sts2_gym import procutil as p\n"
            "parent = int(sys.argv[1])\n"
            "assert p.parent_alive(parent)\n"
            "end = time.time() + 30\n"
            "while p.parent_alive(parent) and time.time() < end: time.sleep(0.05)\n"
            "print('gone' if not p.parent_alive(parent) else 'stuck')\n"
        )
        parent_code = (
            "import os, subprocess, sys\n"
            f"c = subprocess.Popen([sys.executable, '-c', {child_code!r}, str(os.getpid())],"
            " stdout=sys.stderr)\n"
            "import time; time.sleep(1.5)\n"  # child has seen the parent alive
            "print(c.pid)\n"
        )
        # The child writes to a file through the parent's stderr redirect.
        out = ROOT / "out" / f"procutil_test_{os.getpid()}.txt"
        try:
            with open(out, "w") as fh:
                done = subprocess.run(
                    [sys.executable, "-c", parent_code],
                    stdout=subprocess.PIPE,
                    stderr=fh,
                    text=True,
                    timeout=60,
                )
            self.assertEqual(done.returncode, 0)
            child = int(done.stdout.strip())
            deadline = time.monotonic() + 30
            while procutil.parent_alive(child) and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertEqual(out.read_text().strip(), "gone")
        finally:
            out.unlink(missing_ok=True)


class ChildrenTest(unittest.TestCase):
    def test_child_processes_lists_a_direct_child(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            deadline = time.monotonic() + 10
            while child.pid not in [p for p, _ in procutil.child_processes(os.getpid())]:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.1)
        finally:
            child.kill()
            child.wait()


if __name__ == "__main__":
    unittest.main()
