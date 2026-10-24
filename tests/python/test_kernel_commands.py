"""tools/kernel/commands.py (`construct commands`): the call-sequence extractor, the declaration check both ways and the
HP/block write scan, each seen to catch a broken case. The cases use a few lines of made-up C# in the shapes the tool
reads (no game code). One test runs `construct commands` on the real decompile when it is there.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import hook_table as ht  # noqa: E402
from tools.kernel import commands as cm, construct  # noqa: E402

GAME = textwrap.dedent("""
    public static class Cmd
    {
    	public static bool IsReady
    	{
    		get
    		{
    			if (Hook.ShouldWait(x))
    			{
    				return false;
    			}
    			return true;
    		}
    	}

    	public static async Task Run(Thing thing, bool force)
    	{
    		await Hook.Before(thing, "Hook.NotACall(");
    		// Hook.Commented(x);
    		foreach (Thing t in thing.All)
    		{
    			var y = (t == null) ? t : Hook.Pick(t);
    			if (force || Hook.ShouldGo(t))
    			{
    				await Helper(t);
    			}
    			else if (IsReady)
    			{
    				await Hook.After(t);
    			}
    			else
    			{
    				Other(t);
    			}
    		}
    	}
    }
    """)

CALLS = ((r"\bawait Helper\s*\(", "Helper"), (r"\bOther\s*\(", "Other"),
         (r"\bIsReady\b", cm.Inline(r"public static bool IsReady\b")))

CORE2_OK = textwrap.dedent("""
    public static class Cmd2
    {
        // construct: game Cmd.Run
        //   -                      Hook.Before
        //   foreach/cond           Hook.Pick
        //   foreach/if-cond        Hook.ShouldGo
        //   foreach/if             Helper
        //   foreach/elif-cond/if-cond Hook.ShouldWait
        //   foreach/elif           Hook.After
        //   foreach/else           Other ! not modelled: the test's reason
        // end construct
        public static void Run(KernelState k)
        {
            Hooks.Before(k, new());
            var y = Hooks.Pick(k, new());
            if (Hooks.ShouldGo(k, new()))
                Helper(k);
            else if (!Hooks.ShouldWait(k, new()))
                Settle(k, Hooks.After(k, a), st => Hooks.AfterResume(k, a, st));
        }
    }
    """)

CORE2_CALLS = ((r"\bHelper\(k\b", "Helper"),)


def _cmd(game_file="Game/Cmd.cs", core2_file="Core2/Commands/Cmd2.cs"):
    return cm.Command(cm.Side(game_file, r"public static async Task Run\(Thing thing", CALLS),
                      cm.Side(core2_file, r"public static void Run\(KernelState k", CORE2_CALLS))


class ExtractorTests(unittest.TestCase):
    def test_sequence_labels_inlining_strings_and_comments(self):
        got = cm.sequence(GAME, r"public static async Task Run\(Thing thing", CALLS)
        self.assertEqual(got, [
            ("-", "Hook.Before"),
            ("foreach/cond", "Hook.Pick"),
            ("foreach/if-cond", "Hook.ShouldGo"),
            ("foreach/if", "Helper"),
            ("foreach/elif-cond/if-cond", "Hook.ShouldWait"),
            ("foreach/elif", "Hook.After"),
            ("foreach/else", "Other"),
        ])

    def test_core2_names_skip_the_resume_continuations(self):
        side = _cmd().core2
        self.assertEqual(cm.core2_names(CORE2_OK, side), ["Before", "Pick", "ShouldGo", "Helper", "ShouldWait", "After"])

    def test_a_member_matched_twice_or_not_at_all_fails(self):
        with self.assertRaises(cm.CommandsError):
            cm.member_body(GAME + GAME, r"public static async Task Run\(")
        with self.assertRaises(cm.CommandsError):
            cm.member_body(GAME, r"Missing\(")


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "dec" / "Game").mkdir(parents=True)
        (self.root / "eng" / "Core2" / "Commands").mkdir(parents=True)
        (self.root / "dec" / "Game" / "Cmd.cs").write_text(GAME, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, core2: str) -> list[str]:
        (self.root / "eng" / "Core2" / "Commands" / "Cmd2.cs").write_text(core2, encoding="utf-8")
        return cm.check_command("Cmd.Run", _cmd(), self.root / "dec", self.root / "eng")

    def test_the_intact_command_matches(self):
        self.assertEqual(self.check(CORE2_OK), [])

    def test_broken_cases_are_caught(self):
        cases = {
            "a call missing from the declaration": CORE2_OK.replace("//   -                      Hook.Before\n", ""),
            "a label that differs": CORE2_OK.replace("foreach/cond           Hook.Pick", "foreach/if             Hook.Pick"),
            "the order swapped": CORE2_OK.replace("Hook.ShouldWait", "Hook.TMP").replace("Hook.After", "Hook.ShouldWait")
            .replace("Hook.TMP", "Hook.After"),
            "a call missing from Core2": CORE2_OK.replace("var y = Hooks.Pick(k, new());", ""),
            "a Core2 call not declared": CORE2_OK.replace("Hooks.Before(k, new());", "Hooks.Before(k, new()); Hooks.Extra(k);"),
            "no declaration": CORE2_OK.replace("// construct: game Cmd.Run", "// construct: game Cmd.Other"),
        }
        for what, src in cases.items():
            self.assertNotEqual(src, CORE2_OK, what)
            with self.subTest(what):
                self.assertNotEqual(self.check(src), [], what)

    def test_a_not_modelled_call_on_the_core2_side_is_caught(self):
        cmd = _cmd()
        side = cm.Side(cmd.core2.file, cmd.core2.member, CORE2_CALLS + ((r"\bOther\(k\b", "Other"),))
        src = CORE2_OK.replace("Helper(k);", "Helper(k);\n        Other(k);")
        (self.root / "eng" / "Core2" / "Commands" / "Cmd2.cs").write_text(src, encoding="utf-8")
        got = cm.check_command("Cmd.Run", cm.Command(cmd.game, side), self.root / "dec", self.root / "eng")
        self.assertTrue(any("Other only in Core2's calls" in p for p in got), got)

    def test_a_bad_or_unterminated_declaration_fails(self):
        with self.assertRaises(cm.CommandsError):
            cm.declarations("// construct: game X\n//   -   Hook.A\n")
        with self.assertRaises(cm.CommandsError):
            cm.declarations("// construct: game X\n// oops\n// end construct\n")


OLD = textwrap.dedent("""
    internal static class Old
    {
        internal static void Hit(CombatState state, EnemyState enemy)
        {
            enemy.Hp -= 3;
            if (enemy.Hp == 0 && state.PlayerHp != 1)
            {
                state.PlayerBlock = 0;
            }
            if (Kernel.If(Switch.F08, state) is { } k)
            {
                CreatureCmd.Damage(k, targets, 3m);
            }
            else
            {
                enemy.Hp -= 3;
                if (x)
                {
                    state.PlayerHp = 0;
                }
            }
            if (Kernel.If(Switch.F10, state) is { } k2)
            {
                Kill(k2);
            }
            else if (y)
            {
                enemy.MaxHp = 1;
            }
            else
            {
                ++enemy.Hp;
            }
            state.OstyHp += 1;
        }

        internal static EnemyState CreateEnemy(int id) { var e = new EnemyState(); e.Hp = 5; return e; }
    }
    """)


class WriteScanTests(unittest.TestCase):
    def test_writes_outside_gated_else_branches_are_listed(self):
        got = [(w.text, w.method, w.field, w.family) for w in cm.scan_writes(OLD, "Core/Old.cs")]
        self.assertEqual(got, [("enemy.Hp -= 3;", "Hit", "Hp", "F08"), ("state.PlayerBlock = 0;", "Hit", "PlayerBlock", "F09"),
                               ("state.OstyHp += 1;", "Hit", "OstyHp", "F08")])

    def test_the_setup_allow_list(self):
        self.assertEqual(cm.scan_writes(OLD, "Core/CombatFactory.cs"), [])
        self.assertEqual(cm.scan_writes("void F() { s.PlayerHp = 1; }\n", "Core/Run/RunEngine.cs"), [])

    def test_an_ungated_if_does_not_hide_its_else(self):
        src = OLD.replace("Kernel.If(Switch.F08, state)", "Other.If(Switch.F08, state)")
        got = [w.text for w in cm.scan_writes(src, "Core/Old.cs")]
        self.assertIn("state.PlayerHp = 0;", got)
        self.assertEqual(got.count("enemy.Hp -= 3;"), 2)

    def test_core2_writes_outside_commands_and_oldstate_fail(self):
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d)
            (eng / "Core2" / "Commands").mkdir(parents=True)
            (eng / "Core2" / "Commands" / "Cmd.cs").write_text("s.PlayerHp = 1;\n", encoding="utf-8")
            (eng / "Core2" / "OldState.cs").write_text("e.Hp -= 1;\n", encoding="utf-8")
            (eng / "Core2" / "Kernel.cs").write_text("// s.PlayerHp = 1;\nx.Block += 2;\n", encoding="utf-8")
            got = [(w.file, w.line, w.field) for w in cm.core2_writes(eng)]
            self.assertEqual(got, [("Core2/Kernel.cs", 2, "Block")])


@unittest.skipUnless(os.path.isdir(os.path.join(ht.DEFAULT_DECOMPILED, "MegaCrit.Sts2.Core.Commands")), "no local decompile")
class RealCommandsTests(unittest.TestCase):
    def test_construct_commands_holds_on_this_tree(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = construct.main(["commands"])
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("6 commands, 0 differences", out.getvalue())

    def test_a_reordered_core2_command_fails_on_the_real_decompile(self):
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d)
            src = (cm.ENGINE / "Core2" / "Commands" / "CombatEnd.cs").read_text(encoding="utf-8")
            (eng / "Core2" / "Commands").mkdir(parents=True)
            broken = src.replace("//   if                 ProcessPendingLoss\n", "")
            (eng / "Core2" / "Commands" / "CombatEnd.cs").write_text(broken, encoding="utf-8")
            got = cm.check_command("CombatManager.CheckWinCondition", cm.COMMANDS["CombatManager.CheckWinCondition"],
                                   Path(ht.DEFAULT_DECOMPILED), eng)
            self.assertTrue(any("ProcessPendingLoss" in p for p in got), got)


if __name__ == "__main__":
    unittest.main()
