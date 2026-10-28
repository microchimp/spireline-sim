"""docs/reference: the README maps every A7 field of the AI to a corpus table, and the committed reference holds aggregates only (D13)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT):
    sys.path.insert(0, str(_p))

import run_fields  # noqa: E402

from tools.reference import codex_curve as cc  # noqa: E402

DOCS = ROOT / "docs" / "reference"
TABLES = ("survival", "boss_kill", "floor", "deck", "cards", "rests", "boss_hp", "end_causes")
BS = chr(92)


class ReferenceReadme(unittest.TestCase):
    def setUp(self):
        self.text = (DOCS / "README.md").read_text(encoding="utf-8")
        self.json_text = (DOCS / "codex-a10-ironclad.json").read_text(encoding="utf-8")
        self.doc = json.loads(self.json_text)

    def test_every_a7_field_has_a_row(self):
        for key in run_fields.ROW_KEYS:
            self.assertIn(f"`{key}`", self.text, f"the README has no row for the A7 field {key}")

    def test_every_reference_table_is_named(self):
        for t in TABLES:
            self.assertIn(t, self.doc["reference"], t)
            self.assertIn(t, self.text, f"the README does not name the table {t}")

    def test_rule_and_regeneration_are_stated(self):
        for needle in ("D13", "tools/reference/codex_curve.py", "the corpus lacks", "FORBIDDEN_KEYS"):
            self.assertIn(needle, self.text)

    def test_committed_reference_is_aggregates_only(self):
        cc.check_clean(self.doc)
        self.assertEqual(self.doc["header"]["min_cell"], 20)
        self.assertTrue(self.doc["header"]["corpus"].startswith("v"))
        self.assertGreaterEqual(self.doc["reference"]["runs"], 20)
        blob = self.json_text + (DOCS / "codex-a10-ironclad.md").read_text(encoding="utf-8")
        for needle in ("/Users/", "C:/", "D:/", "Users" + BS, "C:" + BS, "D:" + BS):
            self.assertNotIn(needle, blob)


if __name__ == "__main__":
    unittest.main()
