"""referee/driver: a Harmony lookup of a Driver method by name alone must not name an overloaded method.

AccessTools.Method(typeof(Driver), "Snapshot") threw AmbiguousMatchException once Export.cs added Snapshot(CardModel)
beside Snapshot(RunState, Player): every nodeless server died at boot (6451e7f1), and the lab, which runs the full host,
did not notice for days. The driver is not built here, so this reads the sources.
"""

from __future__ import annotations

import re
import unittest
from collections import Counter
from pathlib import Path

DRIVER = Path(__file__).resolve().parents[2] / "referee" / "driver"

# AccessTools.Method(typeof(Driver), "Name") / nameof(Driver.Name), with nothing after the name: a lookup by name alone.
BY_NAME = re.compile(r'AccessTools\.(?:Declared)?Method\(\s*typeof\(Driver\)\s*,\s*(?:"(\w+)"|nameof\(Driver\.(\w+)\))\s*\)')
# A method declaration in the partial class Driver: modifiers, a return type, the name and its parameter list.
DECLARATION = re.compile(r"^\s*(?:(?:public|internal|private|protected|static|async|unsafe|extern|new)\s+)+[\w<>\[\],.?\s]+?\s(\w+)\s*(?:<[^>(]*>)?\(", re.M)


def lookups_by_name(text: str) -> list[str]:
    return [a or b for a, b in BY_NAME.findall(text)]


def overloads(text: str) -> Counter:
    return Counter(DECLARATION.findall(text))


def driver_text() -> str:
    parts = []
    for p in sorted(DRIVER.glob("*.cs")):
        t = p.read_text(encoding="utf-8")
        if re.search(r"\bpartial\s+class\s+Driver\b", t):
            parts.append(t)
    return "\n".join(parts)


class FinderTest(unittest.TestCase):
    def test_finds_a_name_only_lookup_and_the_overloads(self):
        src = '''
        h.Patch(AccessTools.Method(typeof(Driver), "Snapshot"), postfix: x);
        h.Patch(AccessTools.Method(typeof(Driver), nameof(Driver.Step)), postfix: x);
        h.Patch(AccessTools.Method(typeof(Driver), "Pinned", new[] { typeof(int) }), postfix: x);
        h.Patch(AccessTools.Method(typeof(Other), "Snapshot"), postfix: x);
        '''
        self.assertEqual(lookups_by_name(src), ["Snapshot", "Step"])
        decl = '''
        public static partial class Driver {
            internal static JsonObject Snapshot(RunState state, Player player) { }
            internal static JsonObject Snapshot(CardModel card)
            { }
            static async Task<bool> SettleStep(RunState s) { }
        }'''
        self.assertEqual(overloads(decl)["Snapshot"], 2)
        self.assertEqual(overloads(decl)["SettleStep"], 1)


class DriverLookupsTest(unittest.TestCase):
    def test_a_lookup_by_name_alone_names_one_method(self):
        text = driver_text()
        self.assertIn("Snapshot", overloads(text), "the declaration pattern no longer finds the driver's methods")
        count = overloads(text)
        ambiguous = {n: count[n] for n in lookups_by_name("\n".join(p.read_text(encoding="utf-8") for p in DRIVER.glob("*.cs")))
                     if count[n] != 1}
        self.assertEqual(ambiguous, {}, "pass the parameter types: AccessTools.Method(typeof(Driver), name, new[] { ... })")

    def test_the_nodeless_snapshot_patch_names_its_parameter_types(self):
        text = (DRIVER / "NodelessPatches.cs").read_text(encoding="utf-8")
        self.assertRegex(text, r'AccessTools\.Method\(typeof\(Driver\), "Snapshot", new\[\] \{ typeof\(RunState\), typeof\(Player\) \}\)')


if __name__ == "__main__":
    unittest.main()
