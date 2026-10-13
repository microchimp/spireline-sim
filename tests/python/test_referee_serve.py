"""The referee dispatcher's inbox housekeeping (referee/serve.py)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))

import serve  # noqa: E402


class ReleaseClaimLocksTest(unittest.TestCase):
    def test_only_the_dead_servers_locks_go(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inbox = Path(d)
            (inbox / "000001.json").write_text("{}", encoding="utf-8")
            (inbox / "000001.json.lock").write_text("4242", encoding="ascii")  # dead server
            (inbox / "000002.json").write_text("{}", encoding="utf-8")
            (inbox / "000002.json.lock").write_text("777", encoding="ascii")  # a live server mid-claim
            (inbox / "000003.json.taken.4242").write_text("{}", encoding="utf-8")

            self.assertEqual(serve.release_claim_locks(inbox, 4242), 1)

            left = sorted(p.name for p in inbox.iterdir())
            self.assertEqual(left, ["000001.json", "000002.json", "000002.json.lock", "000003.json.taken.4242"])


if __name__ == "__main__":
    unittest.main()
