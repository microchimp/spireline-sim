"""tools.fidelity.debugbench: tasks harvested from history, their signal, the score of a candidate, the report.

A scratch repository with the fork's layout stands in for the fork: an engine (src/Sts2Emulator/Calc.cs) and a
test project whose tests a fake toolchain "runs" (a test passes when the file it names holds the text it names;
`// uses:` text missing from the engine fails the build, as a test calling a method the engine lacks would).
The fake corpus gate replays three fights whose outcome follows the engine's text. The history holds two real
fixes (one with a later test that depends on it: collateral), a feature, a fix whose revert conflicts on main,
a fix whose test passes without it and a fix whose test cannot build without it.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import corpus as cp  # noqa: E402
from tools.fidelity import debugbench as db  # noqa: E402

PROJ = "src/Sts2Emulator.Tests/Sts2Emulator.Tests.csproj"
CALC = "src/Sts2Emulator/Calc.cs"


def calc(add="a - b", frail="amount * 3 / 4", zero="return 0;", extra="", head="") -> str:
    return f"""namespace Sts2Emulator;
{head}
public static class Calc
{{
    public static int Add(int a, int b)
    {{
        return {add};
    }}

    public static int Block(int amount, bool frail)
    {{
        if (amount == 0)
        {{
            {zero}
        }}
        return frail ? {frail} : amount;
    }}

    public static int BlockPreview(int amount) => {frail};
{extra}}}
"""


def test_file(cls: str, tests: dict, header: str = "") -> str:
    body = "".join(f"""
    [Fact]
    public void {name}()
    {{
        // needs: {need}
        Assert.True(true);
    }}
""" for name, need in tests.items())
    return f"""using Xunit;

namespace Sts2Emulator.Tests;

{header}public class {cls}
{{{body}}}
"""


class FakeToolchain:
    """dotnet's place: a build fails on a `// uses:` the engine lacks; a test passes when its `// needs:` holds."""

    def __init__(self):
        self.builds = 0

    @staticmethod
    def _engine(tree: Path) -> str:
        return "".join(p.read_text(encoding="utf-8") for p in sorted((tree / "src" / "Sts2Emulator").rglob("*.cs")))

    def build(self, tree: Path, proj: str, log: Path) -> bool:
        self.builds += 1
        if not (tree / proj).exists():
            return False
        eng = self._engine(tree)
        for p in (tree / proj).parent.rglob("*.cs"):
            for ln in p.read_text(encoding="utf-8").splitlines():
                if "// uses:" in ln and ln.split("// uses:", 1)[1].strip() not in eng:
                    return False
        return True

    def tests(self, tree: Path, proj: str, filters, work: Path, log: Path):
        out = {}
        for p in (tree / proj).parent.rglob("*.cs"):
            text = p.read_text(encoding="utf-8")
            lines = text.splitlines()
            for fqn, (s, e) in db.test_fqns(text).items():
                if filters and not any(f in fqn for f in filters):
                    continue
                ok = True
                for ln in lines[s - 1:e]:
                    if "// needs:" in ln:
                        path, _, want = ln.split("// needs:", 1)[1].strip().partition(" contains ")
                        f = tree / path
                        ok &= f.exists() and want in f.read_text(encoding="utf-8")
                out[fqn] = "pass" if ok else "fail"
        return out

    def emudump(self, tree: Path, log: Path):
        dll = tree / "EmuDump.dll"
        dll.write_text("fake", encoding="utf-8")
        return dll


class FakeGates:
    """The corpus gate's place: f1 matches only with Add summing, f2 only with Frail rounding down; f3 always."""

    def __init__(self, corpus: Path):
        self.corpus = corpus
        self.runs = 0

    def corpus_run(self, tree: Path, dll: Path, label: str, keep: Path) -> dict:
        self.runs += 1
        text = (tree / CALC).read_text(encoding="utf-8")
        fights = {"f1": {"n": 4, "fd": None, "rv": "MMM"} if "a + b" in text else
                  {"n": 4, "fd": 2, "field": "player.hp", "card": "STRIKE", "rv": "MDM", "dp": {"2": ["player.hp"]}},
                  "f2": {"n": 3, "fd": None, "rv": "MM"} if "(int)(amount * 0.75)" in text else
                  {"n": 3, "fd": 1, "field": "player.block", "card": "DEFEND", "rv": "DM", "dp": {"1": ["player.block"]}},
                  "f3": {"n": 3, "fd": None, "rv": "MM"}}
        doc = {"format": cp.RESULTS_FORMAT, "label": label, "subset": False, "problems": [], "skipped": {},
               "emulator": {"commit": "x", "dirty": False, "judge_sha256": "j", "stale_build": False},
               "totals": cp.totals(fights), "fights": fights}
        keep.mkdir(parents=True, exist_ok=True)
        (keep / "corpus-results.json").write_text(json.dumps(doc), encoding="utf-8")
        return doc

    def invariants(self, tree: Path, dll: Path, work: Path) -> dict:
        text = (tree / CALC).read_text(encoding="utf-8")
        return {"emulator": {} if "a + b" in text else {"hp_range": 3}, "live": {}, "replay_errors": 0, "notes": {}}

    def refset(self, tree, label, against):
        return {"exit": 0, "tail": ""}


def sh(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def commit(repo: Path, files: dict, msg: str) -> str:
    for rel, text in files.items():
        p = repo / rel
        if text is None:
            p.unlink()
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    sh(repo, "add", "-A")
    sh(repo, "commit", "-q", "-m", msg)
    return sh(repo, "rev-parse", "HEAD")


def history(d: Path) -> tuple[Path, dict]:
    repo = d / "fork"
    repo.mkdir()
    sh(repo, "init", "-q", "-b", "main")
    sh(repo, "config", "user.email", "t@example.invalid")
    sh(repo, "config", "user.name", "t")
    sh(repo, "config", "core.autocrlf", "false")
    c = {}
    base_tests = {"BlockOfOneIsOne": f"{CALC} contains public static int Block"}
    c["c0"] = commit(repo, {CALC: calc(), PROJ: "<Project/>\n", "referee/FINDINGS.md": "# Findings\n",
                            "tools/fidelity/judge.py": "JUDGE = 1\n",
                            "src/Sts2Emulator.Tests/CalcTests.cs": test_file("CalcTests", base_tests)}, "initial")
    c["fix1"] = commit(repo, {CALC: calc(add="a + b"), "src/Sts2Emulator.Tests/AddTests.cs": test_file(
        "AddTests", {"AddSums": f"{CALC} contains a + b"}, "// Add used to subtract: the bug this file pins.\n")},
        "Add sums its operands\n\nThe emulator subtracted.\n\nCo-Authored-By: someone <x@example.invalid>")
    c["feat"] = commit(repo, {CALC: calc(add="a + b", extra="    public static int Mul(int a, int b) => a * b;\n"),
                              "src/Sts2Emulator.Tests/MulTests.cs": test_file("MulTests", {"MulWorks": f"{CALC} contains a * b"})},
                       "feat(calc): Mul")
    tests2 = {**base_tests, "FrailRoundsDown": f"{CALC} contains (int)(amount * 0.75)"}
    mul = "    public static int Mul(int a, int b) => a * b;\n"
    c["fix2"] = commit(repo, {CALC: calc(add="a + b", frail="(int)(amount * 0.75)", extra=mul),
                              "src/Sts2Emulator.Tests/CalcTests.cs": test_file("CalcTests", tests2)},
                       "Frail block rounds down at both sites")
    c["later"] = commit(repo, {"src/Sts2Emulator.Tests/OtherTests.cs": test_file(
        "OtherTests", {"AddIsCommutative": f"{CALC} contains a + b", "Unrelated": f"{CALC} contains Calc"})},
        "Tests that lean on Add")
    c["findings"] = commit(repo, {"referee/FINDINGS.md": f"# Findings\n- 2026-01-01 Add subtracted: fix {c['fix1'][:8]}\n"
                                                         "- 2026-01-02 something else: open\n"}, "Findings")
    tests3 = {**tests2, "ZeroBlockIsMarked": f"{CALC} contains return 0; // zero"}
    c["conflict"] = commit(repo, {CALC: calc(add="a + b", frail="(int)(amount * 0.75)", zero="return 0; // zero", extra=mul),
                                  "src/Sts2Emulator.Tests/CalcTests.cs": test_file("CalcTests", tests3)},
                           "Zero block is marked")
    c["over"] = commit(repo, {CALC: calc(add="a + b", frail="(int)(amount * 0.75)", zero="return 0; // zero block", extra=mul)},
                       "Zero block comment")
    tests4 = {**tests3, "CalcIsThere": f"{CALC} contains class Calc"}
    neg = mul + "    public static int Neg(int a) => -a;\n"
    c["noop"] = commit(repo, {CALC: calc(add="a + b", frail="(int)(amount * 0.75)", zero="return 0; // zero block",
                                         extra=mul, head="// Calc is static.\n"),
                              "src/Sts2Emulator.Tests/CalcTests.cs": test_file("CalcTests", tests4)},
                       "Calc's tests also check it exists")
    c["api"] = commit(repo, {CALC: calc(add="a + b", frail="(int)(amount * 0.75)", zero="return 0; // zero block",
                                        extra=neg, head="// Calc is static.\n"),
                             "src/Sts2Emulator.Tests/NegTests.cs": test_file("NegTests", {"NegNegates": f"{CALC} contains => -a"})
                             .replace("// needs:", "// uses: Neg(\n        // needs:")},
                      "Neg negates")
    return repo, c


class Bench(unittest.TestCase):
    """One harvest for the class; each test reads what it made."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.repo, cls.c = history(d)
        cls.bench = d / "bench"
        corpus = d / "corpus"
        corpus.mkdir()
        (corpus / "index.json").write_text(json.dumps({"format": cp.INDEX_FORMAT, "fights": [
            {"id": k, "seed": k.upper(), "steps": n, "needs": []} for k, n in (("f1", 4), ("f2", 3), ("f3", 3))],
            "scanned": {}, "excluded": {}}), encoding="utf-8")
        cls.tc, cls.gates = FakeToolchain(), FakeGates(corpus)
        cls.ctx = db.Ctx(cls.repo, cls.bench, "main", toolchain=cls.tc, gates=cls.gates)
        out = io.StringIO()
        with redirect_stdout(out):
            db.harvest(cls.ctx, "main", None, 0, 300, 2, False)
        cls.harvest_out = out.getvalue()
        cls.tasks = {t["fix"]: t for t in db.all_tasks(cls.bench)}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def task(self, key: str) -> tuple[dict, Path]:
        t = self.tasks[self.c[key]]
        return t, self.bench / "tasks" / t["split"] / t["id"]

    def test_harvest_takes_the_two_fixes_and_says_why_it_skips_the_rest(self):
        self.assertEqual(set(self.tasks), {self.c["fix1"], self.c["fix2"]}, self.harvest_out)
        state = self.bench / "harvest" / f"{self.ctx.main[:12]}.jsonl"
        why = {json.loads(ln)["fix"]: json.loads(ln).get("skip", "") for ln in state.read_text(encoding="utf-8").splitlines()}
        self.assertIn("not a fix (feat)", why[self.c["feat"]])
        self.assertIn("does not apply cleanly", why[self.c["conflict"]])
        self.assertIn("no test of the fix fails without it", why[self.c["noop"]])
        self.assertIn("do not build without its engine change", why[self.c["api"]])
        idx = json.loads((self.bench / "tasks" / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(len(idx["tasks"]), 2)

    def test_a_task_hides_its_tests_and_the_collateral_ones_and_names_its_truth(self):
        t, tdir = self.task("fix1")
        self.assertEqual(t["id"], db.task_id(self.c["fix1"]))
        self.assertEqual(t["split"], db.split_of(t["id"]))
        self.assertEqual(t["hidden"], ["Sts2Emulator.Tests.AddTests.AddSums"])
        self.assertEqual(t["collateral"], ["Sts2Emulator.Tests.OtherTests.AddIsCommutative"])
        self.assertEqual(t["removed_files"], ["src/Sts2Emulator.Tests/AddTests.cs"])  # the fix added it; nothing is left
        self.assertEqual(t["truth"]["sites"], [f"{CALC}::Calc.Add"])
        self.assertEqual(t["message"], "Add sums its operands\n\nThe emulator subtracted.")  # trailers dropped
        base = t["base_tree"]
        self.assertIn("a - b", db.blob(self.repo, base, CALC).decode())
        self.assertIsNone(db.blob(self.repo, base, "src/Sts2Emulator.Tests/AddTests.cs"))
        other = db.blob(self.repo, base, "src/Sts2Emulator.Tests/OtherTests.cs").decode()
        self.assertNotIn("AddIsCommutative", other)
        self.assertIn("Unrelated", other)
        findings = db.blob(self.repo, base, "referee/FINDINGS.md").decode()
        self.assertNotIn(self.c["fix1"][:7], findings)
        self.assertIn("something else", findings)
        self.assertTrue((tdir / "hidden" / "src/Sts2Emulator.Tests/OtherTests.cs").exists())
        # Everything but the engine change and the hidden tests is today's main.
        changed = {f for _, f in db.changed(self.repo, base, "main")}
        self.assertEqual(changed, {CALC, "src/Sts2Emulator.Tests/AddTests.cs", "src/Sts2Emulator.Tests/OtherTests.cs",
                                   "referee/FINDINGS.md"})

    def test_a_fix_of_two_sites_has_both_in_its_truth(self):
        t, _ = self.task("fix2")
        self.assertEqual(t["hidden"], ["Sts2Emulator.Tests.CalcTests.FrailRoundsDown"])
        self.assertEqual(t["collateral"], [])
        self.assertEqual(t["truth"]["sites"], [f"{CALC}::Calc.Block", f"{CALC}::Calc.BlockPreview"])
        self.assertNotIn("FrailRoundsDown", db.blob(self.repo, t["base_tree"], "src/Sts2Emulator.Tests/CalcTests.cs").decode())

    def test_signal_sees_the_bug_through_the_corpus_and_the_invariants(self):
        t, tdir = self.task("fix1")
        with redirect_stdout(io.StringIO()):
            db.run_signals(self.ctx, [t], 1, True, None)
        s = json.loads((tdir / "signal.json").read_text(encoding="utf-8"))
        self.assertTrue(s["detected"])
        g = s["gates"]["corpus"]
        self.assertEqual((g["regressed"], g["regressed_ids"], g["fields"]), (1, ["f1"], {"player.hp": 1}))
        self.assertTrue(s["gates"]["invariants"]["detected"])
        self.assertIn("REGRESSIONS: 1 fights", (tdir / "corpus-check.txt").read_text(encoding="utf-8"))
        self.assertNotIn(t["fix"][:7], json.dumps(s))  # the signal names no commit of the truth

    def candidate(self, key: str, name: str, files: dict) -> Path:
        t, _ = self.task(key)
        out = self.bench.parent / f"prep-{name}"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(db.prepare(self.ctx, t["id"], out, heldout=True), 0)
        self.assertEqual(db.tree_of(out, "base"), t["base_tree"])
        self.assertEqual(sh(out, "rev-list", "--count", "base"), "1")  # no history to read the fix from
        sh(out, "config", "user.email", "t@example.invalid")
        sh(out, "config", "user.name", "t")
        sh(out, "checkout", "-q", "-b", "fix")
        text = (out / CALC).read_text(encoding="utf-8")
        for a, b in files.items():
            text = text.replace(a, b)
        (out / CALC).write_text(text, encoding="utf-8", newline="\n")
        (out / "tools/fidelity/judge.py").write_text("JUDGE = 'gamed'\n", encoding="utf-8")
        sh(out, "commit", "-qam", "candidate")
        return out

    def score(self, key: str, repo: Path, run: str, report: dict) -> dict:
        t, tdir = self.task(key)
        with redirect_stdout(io.StringIO()):
            db.run_signals(self.ctx, [t], 1, False, None)
            return db.score_one(self.ctx, t, tdir, repo, "fix", run, report, False, self.bench / "wt" / f"score-{run}")

    def test_a_right_fix_scores_correct_and_restores_main(self):
        repo = self.candidate("fix1", "right", {"a - b": "a + b"})
        r = self.score("fix1", repo, "r-right", {"suspects": [f"{CALC}::Calc.Add"], "hours": 0.5, "tokens": 1000})
        self.assertTrue(r["correct"], r)
        self.assertEqual(r["hidden"], {"Sts2Emulator.Tests.AddTests.AddSums": "pass"})
        self.assertEqual(r["collateral"], {"Sts2Emulator.Tests.OtherTests.AddIsCommutative": "pass"})
        self.assertEqual((r["corpus"]["regressed"], r["corpus"]["restored"], r["corpus"]["base_regressed"]), (0, 1, 1))
        self.assertEqual(r["missed_sites"], [])
        self.assertEqual(r["unit_regressions"], [])
        self.assertTrue(r["localisation"]["top1_file"] and r["localisation"]["top1_site"])
        self.assertEqual(r["outside_engine"], ["tools/fidelity/judge.py"])  # judged with the base's judge anyway

    def test_a_fix_of_one_of_two_sites_passes_its_test_but_misses_a_sibling(self):
        repo = self.candidate("fix2", "half", {"=> amount * 3 / 4": "=> (int)(amount * 0.75)"})
        r = self.score("fix2", repo, "r-half", {"suspects": ["Calc.cs::BlockPreview", f"{CALC}::Calc.Block"]})
        self.assertTrue(r["hidden_pass"])
        self.assertEqual(r["missed_sites"], [f"{CALC}::Calc.Block"])
        self.assertTrue(r["localisation"]["top1_file"])

    def test_a_wrong_fix_fails_and_the_report_counts_it(self):
        repo = self.candidate("fix1", "wrong", {"return frail": "return !frail"})
        r = self.score("fix1", repo, "r-wrong", {"suspects": ["src/Sts2Emulator/Other.cs"], "verdict": "reject", "round": 1})
        self.assertFalse(r["correct"])
        self.assertFalse(r["hidden_pass"])
        self.assertEqual(r["missed_sites"], [f"{CALC}::Calc.Add"])
        self.assertFalse(r["localisation"]["top3_file"])
        # A run of one right and one wrong fix.
        shutil.copy(self.bench / "runs" / "r-wrong" / f"{r['task']}.json", self.bench / "runs" / "r-wrong" / "db-0000000000.json")
        right = json.loads((self.bench / "runs" / "r-wrong" / f"{r['task']}.json").read_text(encoding="utf-8"))
        right.update(task=r["task"], correct=True, hidden_pass=True, verdict="accept", hours=2, tokens=500,
                     missed_sites=[], localisation={**right["localisation"], "top1_file": True, "top3_file": True})
        (self.bench / "runs" / "r-wrong" / "db-0000000000.json").write_text(json.dumps(right), encoding="utf-8")
        text = "\n".join(db.run_report(self.bench, "r-wrong", "all"))
        self.assertIn("fix correctness: 50% (1/2)", text)
        self.assertIn("first-round acceptance: 50% (1/2)", text)
        self.assertIn("localisation, file: top-1 50% (1/2)", text)
        self.assertIn("cost per correct fix: 2.00 h, 500 tokens", text)

    def test_a_held_out_task_is_not_prepared_without_the_flag(self):
        t, tdir = self.task("fix1")
        held = self.bench / "tasks" / "heldout" / "db-heldout00"
        shutil.copytree(tdir, held)
        doc = json.loads((held / "task.json").read_text(encoding="utf-8"))
        doc.update(id="db-heldout00", split="heldout")
        (held / "task.json").write_text(json.dumps(doc), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(db.prepare(self.ctx, "db-heldout00", self.bench.parent / "prep-held", heldout=False), 2)
        shutil.rmtree(held)

    def test_the_base_is_rebuilt_from_its_diff_when_its_objects_are_gone(self):
        t, tdir = self.task("fix2")
        clone = self.bench.parent / "clone"  # main's history only: the base was never on a branch
        subprocess.run(["git", "clone", "-q", "--no-local", str(self.repo), str(clone)], check=True, capture_output=True)
        self.assertFalse(db.has_object(clone, t["base_tree"], "tree"))
        ctx = db.Ctx(clone, self.bench, "main", toolchain=self.tc, gates=self.gates)
        self.assertEqual(db.ensure_base(ctx, t, tdir), t["base_commit"])  # the same commit, from base.diff on main
        with self.assertRaises(db.BenchError):
            db.ensure_base(ctx, {**t, "base_commit": "1" * 40}, tdir)  # it never makes another commit than the task's


class Outline(unittest.TestCase):
    SRC = """using X;
namespace Game.Core;

// a { brace in a comment
public sealed class Engine : Base
{
    private static readonly int[] Table = { 1, 2, 3 };
    public int Hp { get; set; } = 5;
    public string Name => $"{{literal}} {(Hp > 0 ? "alive {" : "dead")}";

    [Theory]
    [InlineData("}")]
    public void Play(string card)
    {
        var s = @"verbatim "" } ";
        char c = '{';
        switch (card)
        {
            case "Bash":
                Hit();
                break;
            case "Strike":
            {
                Hit();
                break;
            }
        }
        After();
    }

    public (int a, int b) Pair() => (1, 2);

    public Engine(int x) : base(x)
    {
    }

    private class Inner
    {
        void Deep() { }
    }
}
"""

    def test_members_types_and_their_lines(self):
        o = db.cs_outline(self.SRC)
        self.assertEqual(o["namespace"], "Game.Core")
        names = {q: (s, e) for q, s, e, _ in o["members"]}
        self.assertEqual(set(names), {"Engine.Table", "Engine.Hp", "Engine.Name", "Engine.Play", "Engine.Pair",
                                      "Engine.Engine", "Engine.Inner.Deep"})
        self.assertEqual(names["Engine.Play"], (11, 29))
        self.assertEqual({q for q, _, _ in o["types"]}, {"Engine", "Engine.Inner"})
        self.assertEqual(db.test_fqns(self.SRC), {"Game.Core.Engine.Play": (11, 29)})
        self.assertEqual(db.class_filters(self.SRC), ["Game.Core.Engine"])

    def test_a_line_in_a_switch_section_is_that_case(self):
        o = db.cs_outline(self.SRC)
        self.assertEqual(db.site_of(o, 20), 'Engine.Play[case "Bash"]')
        self.assertEqual(db.site_of(o, 24), 'Engine.Play[case "Strike"]')
        self.assertEqual(db.site_of(o, 28), "Engine.Play")  # after the switch
        self.assertEqual(db.site_of(o, 7), "Engine.Table")
        self.assertEqual(db.site_of(o, 3), "(file)")
        self.assertEqual(db.member_of('x.cs::Engine.Play[case "Bash"]'), "x.cs::Engine.Play")

    def test_strip_takes_the_method_its_attributes_and_the_comment_above(self):
        src = test_file("T", {"A": "x", "B": "y"}).replace("    [Fact]\n    public void B", "    // B pins y\n    [Fact]\n    public void B")
        spans = db.test_fqns(src)
        out = db.strip_members(src, [spans["Sts2Emulator.Tests.T.B"]])
        self.assertIn("public void A", out)
        self.assertNotIn("public void B", out)
        self.assertNotIn("B pins y", out)
        self.assertEqual(db.cs_mask(out).count("{"), db.cs_mask(out).count("}"))

    def test_trx_names_and_reports(self):
        self.assertEqual(db.test_key('N.Outer+Inner.M(card: "x(y)")'), "N.Outer.Inner.M")
        md = "# Report\n\n## Suspects\n1. src/Sts2Emulator/A.cs::A.Run first\n2. `B.cs` maybe\n\n## Other\nC.cs\n"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.md"
            p.write_text(md, encoding="utf-8")
            self.assertEqual(db.parse_report(p)["suspects"], ["src/Sts2Emulator/A.cs::A.Run", "B.cs"])
        truth = {"files": ["src/Sts2Emulator/B.cs"], "sites": ["src/Sts2Emulator/B.cs::B.Go[case 1]"]}
        loc = db.localise(["src/Sts2Emulator/A.cs::A.Run", "B.cs::Go"], truth)
        self.assertEqual((loc["top1_file"], loc["top3_file"], loc["top1_site"], loc["top3_site"]), (False, True, False, True))

    def test_invariants_judge_counts_only_what_got_worse(self):
        main = {"emulator": {"a": 2}, "live": {}, "replay_errors": 0, "notes": {"n": 5}}
        self.assertFalse(db.judge_invariants(main, {"emulator": {"a": 2}, "live": {}, "replay_errors": 0, "notes": {"n": 4}})["detected"])
        r = db.judge_invariants(main, {"emulator": {"a": 3}, "live": {"b": 1}, "replay_errors": 0, "notes": {}})
        self.assertEqual(r["worse"], {"emulator:a": [2, 3], "live:b": [0, 1]})

    def test_the_split_is_fixed_by_the_id(self):
        ids = [db.task_id(f"{i:040x}") for i in range(600)]
        held = sum(db.split_of(t) == "heldout" for t in ids)
        self.assertTrue(150 < held < 250, held)
        self.assertEqual([db.split_of(t) for t in ids], [db.split_of(t) for t in ids])


if __name__ == "__main__":
    unittest.main()
