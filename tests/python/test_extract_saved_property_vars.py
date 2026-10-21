"""extract_data reads a Damage/Block var built from a [SavedProperty] (The Scythe, Genetic Algorithm),
and the generated card table carries those numbers."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import extract_data as ed  # noqa: E402

GENERATED = (ROOT / "src" / "Sts2Emulator" / "Generated" / "Cards.g.cs").read_text(encoding="utf-8")


def table_value(entry: str, field: str) -> int:
    m = re.search(r'Entry: "' + entry + r'".*?' + field + r": (-?\d+)", GENERATED)
    assert m, entry
    return int(m.group(1))


class SavedPropertyVars(unittest.TestCase):
    def test_extractor_reads_field_initial(self):
        text = "private int _currentDamage = 13;\n new DamageVar(CurrentDamage, ValueProp.Move)"
        self.assertEqual(ed.saved_property_var(text, "Damage"), 13)
        self.assertEqual(ed.saved_property_var(text, "Block"), 0)
        self.assertEqual(ed.saved_property_var("new DamageVar(ValueProp.Move)", "Damage"), 0)

    def test_generated_table(self):
        self.assertEqual(table_value("THE_SCYTHE", "BaseDamage"), 13)
        self.assertEqual(table_value("GENETIC_ALGORITHM", "BaseBlock"), 1)


if __name__ == "__main__":
    unittest.main()
