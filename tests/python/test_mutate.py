"""tools.fidelity.mutate: mutants of the engine source, the coverage probes, sampling and the kill curve.

The source handling is plain text with comments and strings masked, so these pin what it may and may not
touch: a comparison but not a generic or a lambda arrow, a literal only on a damage/block/... line, a probe
only where C# allows a statement, and line numbers that never move (mutants found on the clean source must
apply to the probed copy).
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import mutate as mu  # noqa: E402

SRC = """namespace X;

public static class Engine
{
    // a < b in a comment, and 3 damage
    static readonly Dictionary<int, List<int>> Table = new()
    {
        [1] = [2],
    };

    public static int Hit(State s, int damage)
    {
        var label = "a < b and 7 damage";
        if (damage > 0 && s.Block >= 2)
        {
            s.Hp -= damage + 1;
        }

        switch (s.Kind)
        {
            case 1:
            case 2:
                Apply(s, 5);
                Draw(s, 1);
                break;
            default:
                s.Hp = Math.Max(0, s.Hp);
                break;
        }

        var pick = s.Items.Where(x => x.Damage < 3).ToList();
        s.ShuffleRng.NextInt(4);
        return damage;
    }
}
"""


class MaskTest(unittest.TestCase):
    def test_comments_and_strings_are_blank_and_lengths_kept(self):
        m = mu.mask(SRC)
        self.assertEqual(len(m), len(SRC))
        self.assertEqual(m.count("\n"), SRC.count("\n"))
        self.assertNotIn("a < b", m)
        self.assertIn("damage > 0", m)

    def test_verbatim_and_interpolated_strings(self):
        m = mu.mask('var a = $"x {y < 2} z"; var b = @"p ""q"" < r"; var c = \'<\';')
        self.assertNotIn("<", m)
        self.assertIn("var b =", m)


class InstrumentTest(unittest.TestCase):
    def setUp(self):
        self.p = mu.instrument(SRC)
        self.lines = SRC.split("\n")
        self.out = self.p.text.split("\n")

    def line_of(self, text: str) -> int:
        return next(i for i, ln in enumerate(self.lines) if text in ln)

    def test_line_numbers_do_not_move(self):
        self.assertEqual(len(self.out), len(self.lines))

    def test_probes_on_method_if_and_case_sections_only(self):
        probed = {self.lines[ln - 1].strip() for ln in self.p.probes.values()}
        self.assertEqual(len(self.p.probes), 4)  # method body, if block, the case 1/2 section, default
        self.assertIn("case 2:", probed)
        self.assertIn("default:", probed)
        self.assertNotIn("case 1:", probed)  # a probe between two labels would not compile
        # Not the class body, the initializer or the switch body itself.
        for text in ("public static class Engine", "Table = new()", "switch (s.Kind)"):
            self.assertNotIn("MutCov", self.out[self.line_of(text) + 1])

    def test_units(self):
        unit = self.p.unit
        method = unit[self.line_of("var label")]
        self.assertIsNotNone(method)
        self.assertNotEqual(unit[self.line_of("s.Hp -= damage")], method)
        self.assertEqual(unit[self.line_of("ShuffleRng")], method)  # back in the method after the if and switch
        self.assertNotEqual(unit[self.line_of("Apply(s, 5)")], unit[self.line_of("Math.Max")])
        self.assertIsNone(unit[self.line_of("[1] = [2]")])  # a field initializer: no probe holds it

    def test_skip_lines(self):
        first = min(self.p.probes.values())
        p = mu.instrument(SRC, 10, frozenset({first}))
        self.assertNotIn(first, p.probes.values())
        self.assertEqual(min(p.probes), 10)


class SitesTest(unittest.TestCase):
    def setUp(self):
        self.ms = mu.sites("Core/Engine.cs", SRC)

    def ops(self, text: str) -> dict:
        ln = next(i for i, x in enumerate(SRC.split("\n")) if text in x) + 1
        out = {}
        for m in self.ms:
            if m.line == ln:
                out.setdefault(m.op, []).append(m.new.strip())
        return out

    def test_comparisons(self):
        o = self.ops("if (damage > 0")
        self.assertIn("if (damage >= 0 && s.Block >= 2)", o["cmp-boundary"])
        self.assertIn("if (damage > 0 && s.Block < 2)", o["cmp-negate"])
        self.assertIn("var pick = s.Items.Where(x => x.Damage >= 3).ToList();", self.ops("var pick")["cmp-negate"])
        # Generics, lambda arrows, comments and strings are not comparisons.
        self.assertNotIn("cmp-negate", self.ops("static readonly Dictionary"))
        self.assertNotIn("cmp-negate", self.ops("var label"))

    def test_literals_only_on_amount_lines(self):
        self.assertIn("s.Hp -= damage + 2;", self.ops("s.Hp -= damage")["literal+1"])
        self.assertNotIn("literal-1", self.ops("Apply(s, 5)"))  # no amount word on the line
        self.assertNotIn("literal+1", self.ops("case 2:"))
        self.assertNotIn("literal+1", self.ops("var label"))

    def test_calls_rng_and_clamps(self):
        self.assertEqual(self.ops("Apply(s, 5)")["delete-call"], ["{ }"])
        swap = [m for m in self.ms if m.op == "swap-calls"]
        self.assertEqual(len(swap), 1)
        self.assertEqual([x.strip() for x in swap[0].new.split("\n")], ["Draw(s, 1);", "Apply(s, 5);"])
        self.assertEqual(self.ops("ShuffleRng")["rng-stream"], ["s.CardSelectionRng.NextInt(4);"])
        self.assertEqual(self.ops("Math.Max")["drop-clamp"], ["s.Hp = (s.Hp);"])  # the literal bound goes

    def test_apply_round_trip(self):
        for m in self.ms:
            text = mu.apply(SRC, m)
            self.assertEqual(text.count("\n"), SRC.count("\n"), m.id)
        with self.assertRaises(AssertionError):
            mu.apply(SRC.replace("Apply(s, 5)", "Apply(s, 6)"), next(m for m in self.ms if m.op == "delete-call"
                                                                      and "Apply" in m.old))


class SampleAndCurveTest(unittest.TestCase):
    def test_sample_skips_unrun_and_favours_hot_code(self):
        cands = [mu.Mutant(f"m{k}", "f", k, "op", "", "", unit=k, fights=f) for k, f in enumerate([0] * 50 + [1] * 50 + [400] * 50)]
        got = mu.sample(cands, 40, random.Random(3))
        self.assertEqual(len({m.id for m in got}), 40)
        self.assertTrue(all(m.fights > 0 for m in got))
        self.assertGreater(sum(m.fights == 400 for m in got), sum(m.fights == 1 for m in got))

    def test_kill_curve(self):
        recs = [{"status": "killed", "first_kill_pos": 3}, {"status": "killed", "first_kill_pos": 120},
                {"status": "survived", "first_kill_pos": None}, {"status": "build_failed"}]
        c = mu.kill_curve(recs, 200, ns=(50, 100))
        self.assertEqual(c, {50: 1 / 3, 100: 1 / 3, 200: 2 / 3})


if __name__ == "__main__":
    unittest.main()


class SchemataTest(unittest.TestCase):
    """Every mutant behind MutCov.Is(id): same line count, and the original still there for every other id."""

    def test_guarded_forms(self):
        from tools.fidelity import schemata as sc  # noqa: PLC0415

        lines = SRC.split("\n")
        muts, seen = [], set()
        for m in mu.sites("Core/Engine.cs", SRC):
            if m.line not in seen and m.line + 1 not in seen:
                seen.add(m.line)
                muts.append((len(muts), m))
        text, skipped = sc.guard(SRC, muts)
        self.assertEqual(skipped, {})
        out = text.split("\n")
        self.assertEqual(len(out), len(lines))
        is_ = sc.IS.format
        got = {}
        for _, m in muts:
            got.setdefault(m.op, out[m.line - 1].strip())
        self.assertEqual(got["cmp-boundary"], f"if (({is_(0)} ? (damage >= 0) : (damage > 0)) && s.Block >= 2)")
        self.assertIn(f"({is_(1)} ? 2 : 1)", got["literal+1"])
        self.assertEqual(got["delete-call"], f"{{ if (!{is_(2)}) Apply(s, 5); }}")
        self.assertEqual(got["rng-stream"], f"({is_(6)} ? s.CardSelectionRng : s.ShuffleRng).NextInt(4);")

    def test_operands(self):
        from tools.fidelity import schemata as sc  # noqa: PLC0415

        def spans(line):
            m = mu.mask(line)
            c = mu.CMP.search(m)
            eq = c.group(1) in ("==", "!=")
            left, right = sc.operand_left(m, c.start(), eq, ";"), sc.operand_right(m, c.end(), eq, "")
            return None if left is None or right is None else line[left:right]

        self.assertEqual(spans("    if (a.B?.C < 3 && x)"), "a.B?.C < 3")
        self.assertEqual(spans("    return k-- == 0;"), "k-- == 0")
        self.assertEqual(spans("    var t = (int)x + 1 >= y ? 1 : 2;"), "(int)x + 1 >= y")
        self.assertEqual(spans("    .Where(c => c.Cost < state.Energy)"), "c.Cost < state.Energy")
        self.assertEqual(spans("    bool b = x < y == z;"), "x < y")
        self.assertIsNone(spans("    if (x is > 3)"))  # a relational pattern has no left operand
        self.assertIsNone(spans("    var q = Foo<int>(x) < 3;"))  # a generic: give up rather than guess

    def test_memo_anchors_match_current_runengine(self):
        """A move in RunEngine.cs must fail here, not in the lab's mutation job."""
        from tools.fidelity import schemata as sc  # noqa: PLC0415

        text = (mu.ROOT / mu.ENGINE / "Core/Run/RunEngine.cs").read_bytes().decode("utf-8").replace("\r\n", "\n")
        out = sc.apply_memo_patches(text)
        self.assertIn("private set; } = new();", out)
        self.assertIn("MutCov.Recall(", out)
        self.assertIn("MutCov.Remember(", out)


class ScratchCopyTest(unittest.TestCase):
    """make_copy builds the tree a mutant is compiled in: EmuDump compiles the invariant sources in and links two data files."""

    def test_the_copy_holds_everything_emudump_needs_and_no_build_output(self):
        import shutil
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        src, dst = tmp / "src", tmp / "dst"
        files = {"src/Sts2Emulator/Core/A.cs": "a", "tools/emudump/Program.cs": "p",
                 "src/Sts2Emulator.Tests/Support/Invariants/Metamorphic.cs": "m", "data/id_map.json": "{}",
                 "tools/fidelity/power_aliases.json": "{}", "src/Sts2Emulator/obj/junk.cs": "x", "tools/emudump/bin/junk.dll": "x",
                 "src/Sts2Emulator.Tests/Other.cs": "not copied"}
        for rel, text in files.items():
            (src / rel).parent.mkdir(parents=True, exist_ok=True)
            (src / rel).write_text(text, encoding="utf-8")
        mu.make_copy(src, dst)
        got = sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file())
        self.assertEqual(["data/id_map.json", "src/Sts2Emulator.Tests/Support/Invariants/Metamorphic.cs", "src/Sts2Emulator/Core/A.cs",
                          "tools/emudump/Program.cs", "tools/fidelity/power_aliases.json"], got)
        (dst / "src/Sts2Emulator/Core/A.cs").write_text("edited", encoding="utf-8")  # a mutant's edit is put back by the next copy
        mu.make_copy(src, dst)
        self.assertEqual("a", (dst / "src/Sts2Emulator/Core/A.cs").read_text(encoding="utf-8"))
        (src / "tools/fidelity/power_aliases.json").unlink()  # a tree without an optional file still copies
        mu.make_copy(src, tmp / "dst2")
