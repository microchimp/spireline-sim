"""F2: the card-reward alternatives are cross-checked against the screen's own list and the check is counted."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "referee"))

import envproto  # noqa: E402


class AltCheckCounts(unittest.TestCase):
    def test_summary_counts_alt_checks_and_mismatches(self):
        t = time.time()
        end = {"ev": "end", "w": 0, "seed": "s", "end": "death", "floor": 3, "asked": 30, "ms": 1, "mem_mb": 900, "dec": 30, "t": t + 100,
               "s": 5.0, "stale": None, "slowest_step": [0.1, "map"], "fight_errors": None, "probe": None}
        ev = [{"ev": "ready", "w": 0, "pid": 1, "boot_s": 5.0, "t": t}, dict(end, alt_checked=4, alt_mismatch=0),
              dict(end, seed="s2", alt_checked=3, alt_mismatch=1), dict(end, seed="s3")]
        self.assertEqual(envproto.summarize(ev, 1)["alt_checks"], {"checked": 7, "mismatch": 1})


if __name__ == "__main__":
    unittest.main()
