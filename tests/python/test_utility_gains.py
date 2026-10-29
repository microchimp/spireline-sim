"""The scanned utility-gain registry (docs/rebuild/FIGHT_SOLVER.md section 19): every utility-raising site the scan finds
is classed, the files on disk are current, and a class removed makes ``--check`` fail."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN = ROOT / "tools" / "solver" / "scan_utility_gains.py"
CLASSES = ROOT / "tools" / "solver" / "utility_gain_classes.json"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCAN), *args], capture_output=True, text=True, cwd=ROOT, timeout=600)


class UtilityGainRegistryTests(unittest.TestCase):
    def test_every_site_is_classed_and_the_files_are_current(self) -> None:
        r = run("--check")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_a_removed_class_fails_the_check(self) -> None:
        doc = json.loads(CLASSES.read_text(encoding="utf-8"))
        gains = [k for k, v in doc["sites"].items() if v["class"] == "gain"]
        self.assertTrue(gains)
        del doc["sites"][gains[0]]
        with tempfile.TemporaryDirectory() as d:
            alt = Path(d) / "classes.json"
            alt.write_text(json.dumps(doc), encoding="utf-8")
            r = run("--check", "--classes", str(alt))
        self.assertEqual(r.returncode, 1)
        self.assertIn(f"unclassed: {gains[0]}", r.stderr)

    def test_the_registry_holds_the_known_gains(self) -> None:
        reg = json.loads((ROOT / "tools" / "solver" / "utility_gains.json").read_text(encoding="utf-8"))
        holders = set(reg["holders"])
        for h in ("card:Feed", "potion:BloodPotion", "potion:FairyInABottle", "relic:LizardTail", "buff:Regen", "potion:RegenPotion", "card:InfernalBlade"):
            self.assertIn(h, holders)
        self.assertNotIn("relic:BurningBlood", holders)
        self.assertEqual({e["applied"] for e in reg["end"]}, {"after-terminal"})


if __name__ == "__main__":
    unittest.main()
