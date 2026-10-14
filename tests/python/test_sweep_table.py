"""tools.fidelity.sweep_table: every game caller of a command, through the game's wrappers, with the emulator
sites that look like the same behaviour, and a --check that fails until every caller has a status.

The decompile here is a synthetic tree in the decompile's layout (namespace folders, one class a file),
written for the test: no game code.
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import sweep_table as sw  # noqa: E402

FILES = {
    "decompiled/MegaCrit.Sts2.Core.Commands/FakeCmd.cs": """namespace MegaCrit.Sts2.Core.Commands;
public static class FakeCmd
{
    public static async Task Apply(int amount)
    {
        await Task.CompletedTask;
    }

    public static async Task Apply(int amount, bool loud)
    {
        await Apply(amount);
    }

    public static async Task ApplyTwice(int amount)
    {
        await FakeCmd.Apply(amount);
        await Apply(amount);
    }

    public static async Task ApplyAll()
    {
        await ApplyTwice(1);
    }
}
""",
    "decompiled/MegaCrit.Sts2.Core.Models.Relics/GlassBell.cs": """namespace MegaCrit.Sts2.Core.Models.Relics;
public sealed class GlassBell
{
    public override async Task AfterCombatEnd()
    {
        await FakeCmd.ApplyAll();
    }

    private void Apply(int amount)
    {
    }

    public override async Task AfterTurnEnd()
    {
        Apply(4);
    }
}
""",
    "decompiled/MegaCrit.Sts2.Core.Models.Cards/SharpBlade.cs": """namespace MegaCrit.Sts2.Core.Models.Cards;
public sealed class SharpBlade
{
    public override async Task OnPlay()
    {
        // FakeCmd.Apply(1) in a comment is no call
        await FakeCmd.Apply(3);
    }
}
""",
    "decompiled/MegaCrit.Sts2.Core.Models.Relics/MossyStone.cs": """namespace MegaCrit.Sts2.Core.Models.Relics;
public sealed class MossyStone
{
    public override async Task AfterTurnStart()
    {
        await FakeCmd.ApplyTwice(2);
    }
}
""",
    "decompiled/MegaCrit.Sts2.Core.Models.Powers/QuietPower.cs": """namespace MegaCrit.Sts2.Core.Models.Powers;
public sealed class QuietPower
{
    public override async Task AfterDamage()
    {
        await FakeCmd.Apply(1);
    }
}
""",
    "emu/Core/Effects.cs": """static class Effects
{
    static void SharpBlade(State s) => Fake.Apply(s, 3);
    static void MossyStoneTurn(State s) { }
}
""",
    "tests/RelicTests.cs": """public class RelicTests
{
    [Fact] public void Mossy_stone_applies_twice() { }  // MossyStone
}
""",
}


class SweepTableTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        for rel, text in FILES.items():
            p = self.d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        self.map = self.d / "map.json"
        self.map.write_text('{"FakeCmd.Apply": {"emulator": ["Effects.Fake"], "terms": ["Fake.Apply"]}}', encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def main(self, *extra) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out):
            code = sw.main(["FakeCmd.Apply", "--decompiled", str(self.d / "decompiled"), "--emulator", str(self.d / "emu"),
                            "--tests", str(self.d / "tests"), "--map", str(self.map), *extra])
        return code, out.getvalue()

    def rows(self, text: str) -> dict:
        cols = None
        out = {}
        for line in text.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and cells[0] == "#":
                cols = cells
            elif cols and len(cells) == len(cols) and cells[0].isdigit():
                r = dict(zip(cols, cells))
                out[r["game caller"]] = r
        return out

    def test_every_caller_through_the_wrappers_and_nothing_else(self):
        code, text = self.main()
        self.assertEqual(code, 0)
        rows = self.rows(text)
        # The card and the power call it; MossyStone through FakeCmd.ApplyTwice (depth 2), GlassBell through
        # FakeCmd.ApplyAll, which calls ApplyTwice (depth 3). The wrappers' own calls are listed too, also
        # the ones the decompile writes without the class inside FakeCmd itself; the definition, the
        # overload calling it, the comment and GlassBell's own Apply are not.
        self.assertEqual(set(rows), {"SharpBlade.OnPlay", "QuietPower.AfterDamage", "FakeCmd.ApplyTwice", "MossyStone.AfterTurnStart",
                                     "FakeCmd.ApplyAll", "GlassBell.AfterCombatEnd"})
        self.assertEqual(rows["MossyStone.AfterTurnStart"]["via"], "FakeCmd.ApplyTwice")
        self.assertEqual(rows["FakeCmd.ApplyAll"]["via"], "FakeCmd.ApplyTwice")
        self.assertEqual(rows["GlassBell.AfterCombatEnd"]["via"], "FakeCmd.ApplyAll")
        self.assertEqual(rows["SharpBlade.OnPlay"]["decompile"], "SharpBlade.cs:7")
        self.assertIn("FakeCmd.cs:17", text)  # ApplyTwice's second call, unqualified
        self.assertNotIn("SharpBlade.cs:6", text)
        self.assertNotIn("FakeCmd.cs:11", text)  # the overload
        self.assertNotIn("GlassBell.cs:15", text)  # another class's own Apply

    def test_depth_caps_the_wrapper_levels(self):
        rows = self.rows(self.main("--depth", "2")[1])
        self.assertIn("FakeCmd.ApplyAll", rows)
        self.assertNotIn("GlassBell.AfterCombatEnd", rows)

    def test_match_says_how_good_the_emulator_hit_is(self):
        rows = self.rows(self.main()[1])
        self.assertEqual(rows["SharpBlade.OnPlay"]["match"], "strong")  # names the card and uses Fake.Apply
        self.assertEqual(rows["MossyStone.AfterTurnStart"]["match"], "weak")
        self.assertEqual(rows["QuietPower.AfterDamage"]["match"], "none")
        self.assertEqual(rows["QuietPower.AfterDamage"]["emulator sites"], "**none found**")
        self.assertIn("RelicTests.cs", rows["MossyStone.AfterTurnStart"]["tests"])

    def test_check_fails_until_every_caller_has_a_valid_status(self):
        code, out = self.main("--check")
        self.assertEqual(code, 1)
        self.assertIn("7 game callers, 7 problems", out)  # FakeCmd.ApplyTwice calls it twice: two rows
        table = self.d / "sweep.md"
        self.main("--out", str(table))
        text = table.read_text(encoding="utf-8")
        filled = []
        for line in text.splitlines():
            if line.startswith("| ") and line.split("|")[1].strip().isdigit():
                cells = line.split("|")
                caller = cells[2].strip()
                status, note = {"SharpBlade.OnPlay": ("converted", ""), "QuietPower.AfterDamage": ("unreachable", ""),
                                "FakeCmd.ApplyTwice": ("tested", ""), "MossyStone.AfterTurnStart": ("done", ""),
                                "FakeCmd.ApplyAll": ("tested", ""), "GlassBell.AfterCombatEnd": ("converted", "")}[caller]
                cells[-3], cells[-2] = f" {status} ", f" {note} "
                line = "|".join(cells)
            filled.append(line)
        table.write_text("\n".join(filled) + "\n", encoding="utf-8")
        code, out = self.main("--keep", str(table), "--check")
        self.assertEqual(code, 1)
        self.assertIn("unreachable needs a reason", out)
        self.assertIn("status 'done' is not one of", out)
        text = table.read_text(encoding="utf-8").replace(" unreachable |  |", " unreachable | the power is not in the game's pools |")
        text = text.replace(" done ", " converted ")
        table.write_text(text, encoding="utf-8")
        code, out = self.main("--keep", str(table), "--check")
        self.assertEqual(code, 0, out)
        # A kept row whose call site is gone (the decompile moved on) fails the check.
        table.write_text(text + "| 99 | GoneRelic.OnPlay |  | GoneRelic.cs:3 | relic | - | none | - | converted |  |\n",
                         encoding="utf-8")
        code, out = self.main("--keep", str(table), "--check")
        self.assertEqual(code, 1)
        self.assertIn("GoneRelic.OnPlay (GoneRelic.cs:3): in the kept table, no such call site now", out)

    def test_a_command_with_no_caller_fails(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = sw.main(["NoSuchCmd.Run", "--decompiled", str(self.d / "decompiled"), "--emulator", str(self.d / "emu"),
                            "--tests", str(self.d / "tests"), "--map", str(self.map)])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
