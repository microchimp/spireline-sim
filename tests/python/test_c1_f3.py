"""F3: typeperf is launched by its full path under %SystemRoot%\System32."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "referee"))

import perfcounters  # noqa: E402


class TypeperfPath(unittest.TestCase):
    def test_path_is_under_system32_of_systemroot(self):
        with mock.patch.dict(os.environ, {"SystemRoot": r"C:\FakeRoot"}):
            self.assertEqual(perfcounters.typeperf_path(), os.path.join(r"C:\FakeRoot", "System32", "typeperf.exe"))

    def test_the_sampler_starts_that_path_not_a_bare_name(self):
        started = []

        class P:
            stdout = None

        def popen(args, **k):
            started.append(args[0])
            raise OSError("no child in a unit test")

        with mock.patch.object(perfcounters.subprocess, "Popen", popen):
            perfcounters.PerfSampler(set()).start()
        self.assertEqual(started, [perfcounters.typeperf_path()])
        self.assertNotEqual(started[0], "typeperf")


if __name__ == "__main__":
    unittest.main()
