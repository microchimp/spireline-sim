"""tools.fidelity.debugbench: tasks harvested from history, their signal, the score of a candidate, the report.

A scratch repository with the fork's layout stands in for the fork: an engine (src/Sts2Emulator/Calc.cs) and a
test project whose tests a fake toolchain "runs" (a test passes when the file it names holds the text it names;
`// uses:` text missing from the engine fails the build, as a test calling a method the engine lacks would).
The fake corpus gate replays four fights whose outcome follows the engine's text (f4 diverges on main too: a
candidate that fixes it beats main). The history holds two real fixes (one with a later test that depends on it:
collateral, and later docs and comments that name its test), a feature, a fix whose revert conflicts on main, a
fix whose test passes without it and a fix whose test cannot build without it.

The pilot's seven fairness findings each have a test here: the truth out of the fixer's reach
(test_truth_*), no hint in the base, its commit or the brief (test_the_base_and_brief_*), silent tasks scored
apart (test_silent_tasks_*), the mechanism's siblings (test_a_fix_of_one_of_two_sites_*, Mechanism),
a fix better than main (test_a_fix_better_than_main_*), the harness's own time and tokens (test_the_harness_*),
and the two added detection signals (test_the_added_gates_*).
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
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
    """The corpus gate's place: f1 matches only with Add summing, f2 only with Frail rounding down; f3 always; f4
    only with a BETTER marker no commit has (main diverges on it)."""

    def __init__(self, corpus: Path):
        self.corpus = corpus
        self.runs = 0
        self.refset_runs = 0

    def corpus_run(self, tree: Path, dll: Path, label: str, keep: Path) -> dict:
        self.runs += 1
        text = (tree / CALC).read_text(encoding="utf-8")
        fights = {"f1": {"n": 4, "fd": None, "rv": "MMM"} if "a + b" in text else
                  {"n": 4, "fd": 2, "field": "player.hp", "card": "STRIKE", "rv": "MDM", "dp": {"2": ["player.hp"]}},
                  "f2": {"n": 3, "fd": None, "rv": "MM"} if "(int)(amount * 0.75)" in text else
                  {"n": 3, "fd": 1, "field": "player.block", "card": "DEFEND", "rv": "DM", "dp": {"1": ["player.block"]}},
                  "f3": {"n": 3, "fd": None, "rv": "MM"},
                  "f4": {"n": 3, "fd": None, "rv": "MM"} if "BETTER" in text else
                  {"n": 3, "fd": 1, "field": "player.energy", "card": "BASH", "rv": "DM", "dp": {"1": ["player.energy"]}}}
        doc = {"format": cp.RESULTS_FORMAT, "label": label, "subset": False, "problems": [], "skipped": {},
               "emulator": {"commit": "x", "dirty": False, "judge_sha256": "j", "stale_build": False},
               "totals": cp.totals(fights), "fights": fights}
        keep.mkdir(parents=True, exist_ok=True)
        (keep / "corpus-results.json").write_text(json.dumps(doc), encoding="utf-8")
        return doc

    def invariants(self, tree: Path, dll: Path, work: Path) -> dict:
        text = (tree / CALC).read_text(encoding="utf-8")
        return {"emulator": {} if "a + b" in text else {"hp_range": 3}, "live": {}, "replay_errors": 0, "notes": {}}

    def refset(self, tree, label, view, against, codex, work):
        """refset_score's place: fight R-2 follows Frail's rounding; codex run 7 follows Add (run-level)."""
        self.refset_runs += 1
        text = (tree / CALC).read_text(encoding="utf-8")
        if against is None:
            p = view / db.REFSET_VERSION / "scores" / f"all-{label}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"label": label, "codex": {"runs": 3} if codex else {}}), encoding="utf-8")
            return {"score": str(p)}
        worse = []
        if "(int)(amount * 0.75)" not in text:
            worse.append("fight R-0002: identical before, now differs at step 1 (.player.block after DEFEND)")
        if codex and "a + b" not in text:
            worse.append("codex run 7: complete before, now mismatch at floor 3 (EMULATOR: card_reward)")
        return {"worse": worse, "unsound": [], "better": [], "ok": not worse, "minutes": 0.1,
                "codex_runs": 3 if codex else None}


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


def init_repo(repo: Path) -> None:
    repo.mkdir()
    sh(repo, "init", "-q", "-b", "main")
    sh(repo, "config", "user.email", "t@example.invalid")
    sh(repo, "config", "user.name", "t")
    sh(repo, "config", "core.autocrlf", "false")


def history(d: Path) -> tuple[Path, dict]:
    repo = d / "fork"
    init_repo(repo)
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
        "OtherTests", {"AddIsCommutative": f"{CALC} contains a + b", "Unrelated": f"{CALC} contains Calc"},
        "// AddSums is the sum's own test; these lean on it.\n")},
        "Tests that lean on Add")
    c["findings"] = commit(repo, {"referee/FINDINGS.md": f"# Findings\n- 2026-01-01 Add subtracted: fix {c['fix1'][:8]}\n"
                                                         "- 2026-01-02 something else: open\n",
                                  "docs/notes.md": "# Notes\n\n- AddSums pins the sum\n  - it failed while Add subtracted\n"
                                                   "- an unrelated note\n"}, "Findings")
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


def no_hooks(ctx, tree):
    return None  # the hook table needs the game's decompile; Mechanism tests it with rows of their own


def bench_files(root: Path) -> str:
    """Every file under root (but git's own) as one text."""
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in root.rglob("*")
                     if p.is_file() and ".git" not in p.relative_to(root).parts)


class Bench(unittest.TestCase):
    """One harvest for the class; each test reads what it made."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.repo, cls.c = history(d)
        cls.bench, cls.truth_store = d / "bench", d / "truth"
        corpus = d / "corpus"
        corpus.mkdir()
        (corpus / "index.json").write_text(json.dumps({"format": cp.INDEX_FORMAT, "fights": [
            {"id": k, "seed": k.upper(), "steps": n, "needs": []} for k, n in (("f1", 4), ("f2", 3), ("f3", 3), ("f4", 3))],
            "scanned": {}, "excluded": {}}), encoding="utf-8")
        cls.tc, cls.gates = FakeToolchain(), FakeGates(corpus)
        refset = d / "refset" / db.REFSET_VERSION
        refset.mkdir(parents=True)
        for name in ("refs-all.json", "manifest.json"):
            (refset / name).write_text("{}", encoding="utf-8")
        cls._refset_dir, db.REFSET_DIR = db.REFSET_DIR, d / "refset"
        cls.ctx = db.Ctx(cls.repo, cls.bench, "main", toolchain=cls.tc, gates=cls.gates, truth=cls.truth_store,
                         hooks=no_hooks)
        out = io.StringIO()
        with redirect_stdout(out):
            db.harvest(cls.ctx, "main", None, 0, 300, 2, False)
        cls.harvest_out = out.getvalue()
        cls.tasks = {db.load_truth(cls.ctx, t["id"])[0]["fix"]: t for t in db.all_tasks(cls.bench)}

    @classmethod
    def tearDownClass(cls):
        db.REFSET_DIR = cls._refset_dir
        cls.tmp.cleanup()

    def task(self, key: str) -> tuple[dict, Path]:
        t = self.tasks[self.c[key]]
        return t, self.bench / "tasks" / t["split"] / t["id"]

    def truth(self, key: str) -> tuple[dict, Path]:
        return db.load_truth(self.ctx, self.tasks[self.c[key]]["id"])

    def test_harvest_takes_the_two_fixes_and_says_why_it_skips_the_rest(self):
        self.assertEqual(set(self.tasks), {self.c["fix1"], self.c["fix2"]}, self.harvest_out)
        state = self.truth_store / "harvest" / f"{self.ctx.main[:12]}.jsonl"
        why = {json.loads(ln)["fix"]: json.loads(ln).get("skip", "") for ln in state.read_text(encoding="utf-8").splitlines()}
        self.assertIn("not a fix (feat)", why[self.c["feat"]])
        self.assertIn("does not apply cleanly", why[self.c["conflict"]])
        self.assertIn("no test of the fix fails without it", why[self.c["noop"]])
        self.assertIn("do not build without its engine change", why[self.c["api"]])
        idx = json.loads((self.bench / "tasks" / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(len(idx["tasks"]), 2)

    def test_a_task_hides_its_tests_and_the_collateral_ones_and_names_its_truth(self):
        t, _ = self.task("fix1")
        tr, trdir = self.truth("fix1")
        self.assertEqual(t["id"], db.task_id(self.c["fix1"]))
        self.assertEqual(t["split"], db.split_of(t["id"]))
        self.assertEqual(tr["hidden"], ["Sts2Emulator.Tests.AddTests.AddSums"])
        self.assertEqual(tr["collateral"], ["Sts2Emulator.Tests.OtherTests.AddIsCommutative"])
        self.assertEqual(tr["removed_files"], ["src/Sts2Emulator.Tests/AddTests.cs"])  # the fix added it; nothing is left
        self.assertEqual(tr["truth"]["sites"], [f"{CALC}::Calc.Add"])
        self.assertEqual(tr["message"], "Add sums its operands\n\nThe emulator subtracted.")  # trailers dropped
        base = t["base_tree"]
        self.assertIn("a - b", db.blob(self.repo, base, CALC).decode())
        self.assertIsNone(db.blob(self.repo, base, "src/Sts2Emulator.Tests/AddTests.cs"))
        other = db.blob(self.repo, base, "src/Sts2Emulator.Tests/OtherTests.cs").decode()
        self.assertNotIn("AddIsCommutative", other)
        self.assertIn("Unrelated", other)
        findings = db.blob(self.repo, base, "referee/FINDINGS.md").decode()
        self.assertNotIn(self.c["fix1"][:7], findings)
        self.assertIn("something else", findings)
        self.assertTrue((trdir / "hidden" / "src/Sts2Emulator.Tests/OtherTests.cs").exists())
        # Everything but the engine change, the hidden tests and the lines naming the fix is today's main.
        changed = {f for _, f in db.changed(self.repo, base, "main")}
        self.assertEqual(changed, {CALC, "src/Sts2Emulator.Tests/AddTests.cs", "src/Sts2Emulator.Tests/OtherTests.cs",
                                   "referee/FINDINGS.md", "docs/notes.md"})

    def test_truth_stays_out_of_the_bench_and_out_of_the_fixers_reach(self):
        t, tdir = self.task("fix1")
        tr, trdir = self.truth("fix1")
        self.assertEqual(set(json.loads((tdir / "task.json").read_text(encoding="utf-8"))),
                         {"id", "split", "main", "base_tree", "base_commit", "made", "format", "hidden_count",
                          "collateral_count"})
        for name in ("gold.diff", "base.diff", "fix.diff", "hidden"):
            self.assertFalse((tdir / name).exists(), name)
            self.assertTrue((trdir / name).exists(), name)
        with redirect_stdout(io.StringIO()):
            db.run_signals(self.ctx, [t], 1, ("corpus", "invariants"))
        public = bench_files(self.bench / "tasks")
        for secret in (self.c["fix1"][:7], "AddSums", "AddIsCommutative", "Add sums its operands", "Calc.Add"):
            self.assertNotIn(secret, public)
        # A fixer's repo is never put where the answer is: in the bench, the truth store or the fork.
        with redirect_stdout(io.StringIO()):
            for bad in (self.bench / "pilot" / "x", self.truth_store / "x", self.repo / "x", self.bench.parent):
                self.assertEqual(db.prepare(self.ctx, t["id"], bad, heldout=True), 2, bad)
        # The truth store has no default: a fixer running the tool from its own repo finds nothing.
        bare = db.Ctx(self.repo, self.bench, "main", toolchain=self.tc, gates=self.gates)
        with self.assertRaises(db.BenchError):
            db.load_truth(bare, t["id"])
        with self.assertRaises(db.BenchError):
            db.Ctx(self.repo, self.bench, "main", truth=self.bench / "truth").need_truth()  # never under the bench
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(db.main(["--fork", str(self.repo), "--bench", str(self.bench), "list", "--truth"]), 2)
        self.assertIn("truth store", out.getvalue())

    def test_the_base_and_brief_name_nothing_of_the_fix(self):
        t, tdir = self.task("fix1")
        tr, _ = self.truth("fix1")
        base = t["base_tree"]
        notes = db.blob(self.repo, base, "docs/notes.md").decode()
        self.assertEqual(notes, "# Notes\n\n- an unrelated note\n")  # the item that names the test, with its sub-item
        self.assertNotIn("AddSums", db.blob(self.repo, base, "src/Sts2Emulator.Tests/OtherTests.cs").decode())
        self.assertIn("AddSums", tr["hints"]["names"])
        self.assertEqual(sh(self.repo, "log", "-1", "--format=%B", t["base_commit"]), "base")
        with redirect_stdout(io.StringIO()):
            db.run_signals(self.ctx, [t], 1, ("corpus", "invariants"))
        out = self.bench.parent / "prep-neutral"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(db.prepare(self.ctx, t["id"], out, heldout=True), 0)
        brief = (out.parent / "prep-neutral.brief.md").read_text(encoding="utf-8")
        self.assertIn("corpus check against the reference run: 1 fights worse", brief)
        for secret in (t["id"], self.ctx.main[:10], "debugbench", " main", "AddSums", str(self.bench)):
            self.assertNotIn(secret, brief)
        self.assertEqual(sh(out, "branch", "--format=%(refname:short)"), "base")
        self.assertEqual(sh(out, "log", "--format=%B"), "base")

    def test_signal_sees_the_bug_through_the_corpus_and_the_invariants(self):
        t, tdir = self.task("fix1")
        with redirect_stdout(io.StringIO()):
            db.run_signals(self.ctx, [t], 1, ("corpus", "invariants"))
        s = json.loads((tdir / "signal.json").read_text(encoding="utf-8"))
        self.assertTrue(s["detected"])
        g = s["gates"]["corpus"]
        self.assertEqual((g["regressed"], g["regressed_ids"], g["fields"]), (1, ["f1"], {"player.hp": 1}))
        self.assertTrue(s["gates"]["invariants"]["detected"])
        text = (tdir / "corpus-check.txt").read_text(encoding="utf-8")
        self.assertIn("REGRESSIONS: 1 fights", text)
        self.assertNotIn(self.ctx.main[:10], text)  # the baseline is "reference", not main's commit

    def test_the_added_gates_judge_against_mains_score_and_keep_the_first_runs(self):
        t1, d1 = self.task("fix1")
        t2, d2 = self.task("fix2")
        with redirect_stdout(io.StringIO()):
            db.run_signals(self.ctx, [t1, t2], 1, ("corpus", "invariants"))
            first = json.loads((d1 / "signal.json").read_text(encoding="utf-8"))
            db.run_signals(self.ctx, [t1, t2], 1, ("refset", "codex"))
        s1, s2 = (json.loads((d / "signal.json").read_text(encoding="utf-8")) for d in (d1, d2))
        self.assertEqual(list(s1["gates"]), ["corpus", "invariants", "refset", "codex"])
        self.assertEqual(s1["gates"]["corpus"], first["gates"]["corpus"])  # a gate not run now keeps its result
        self.assertTrue(s1["gates"]["codex"]["detected"])  # Add is run-level: the walk sees it
        self.assertEqual(s1["gates"]["codex"]["stops"], {"EMULATOR: card_reward": 1})
        self.assertFalse(s1["gates"]["refset"]["detected"])
        self.assertTrue(s2["gates"]["refset"]["detected"])
        self.assertFalse(s2["gates"]["codex"]["detected"])
        self.assertTrue(any(ln.startswith("codex whole-run replay") for ln in s1["lines"]))
        # Main's score is made once, with the walk, in the bench's own copy of the reference set's folder.
        scores = list((self.ctx.main_dir / "refset" / db.REFSET_VERSION / "scores").glob("all-debugbench-main-*"))
        self.assertEqual(len(scores), 1, scores)
        self.assertTrue(scores[0].name.endswith("-codex.json"))
        text = "\n".join(db.detection_lines([s1, s2]))
        self.assertIn("refset: 50% (1/2)", text)
        self.assertIn("codex: 50% (1/2)", text)
        self.assertIn("corpus, invariants) miss 0", text)
        # A task the first gates miss and the walk catches is counted as caught.
        silent = {"gates": {"corpus": {"detected": False}, "invariants": {"detected": False},
                            "refset": {"detected": False}, "codex": {"detected": True}}}
        text = "\n".join(db.detection_lines([s1, s2, silent]))
        self.assertIn("corpus, invariants) miss 1", text)
        self.assertIn("any of refset, codex: 1; still silent 0", text)

    def test_an_unsound_refset_comparison_is_no_detection(self):
        g = db.judge_refset({"worse": ["codex run 3: an error now at floor 2 (error: x)"], "better": [],
                             "unsound": ["unsound: a codex walk without per-run results (scored before ...)"]},
                            {"refset", "codex"})
        self.assertIsNone(g["codex"]["detected"])
        self.assertFalse(g["refset"]["detected"])  # the unsound line is the walk's alone
        self.assertIsNone(db.judge_refset({"worse": [], "unsound": ["unsound: subsets differ (all vs pilot)"]},
                                          {"refset"})["refset"]["detected"])
        self.assertIsNone(db.judge_refset({"error": "refset_score failed (exit 2)"}, {"codex"})["codex"]["detected"])

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
            db.run_signals(self.ctx, [t], 1, ("corpus",))
            return db.score_one(self.ctx, t, tdir, repo, "fix", run, report, False, self.bench / "wt" / f"score-{run}")

    def test_a_right_fix_scores_correct_and_restores_main(self):
        repo = self.candidate("fix1", "right", {"a - b": "a + b"})
        r = self.score("fix1", repo, "r-right", {"suspects": [f"{CALC}::Calc.Add"], "hours": 0.5, "tokens": 1000})
        self.assertTrue(r["correct"], r)
        self.assertEqual(r["outcome"], "fixed")
        self.assertFalse(r["beats_main"])
        self.assertEqual(r["hidden"], {"Sts2Emulator.Tests.AddTests.AddSums": "pass"})
        self.assertEqual(r["collateral"], {"Sts2Emulator.Tests.OtherTests.AddIsCommutative": "pass"})
        self.assertEqual((r["corpus"]["regressed"], r["corpus"]["restored"], r["corpus"]["base_regressed"]), (0, 1, 1))
        self.assertEqual(r["differs_from_gold"]["missed_sites"], [])
        self.assertEqual((r["siblings"]["credit"], r["siblings"]["units"]), (1.0, 1))
        self.assertEqual(r["unit_regressions"], [])
        self.assertTrue(r["localisation"]["top1_file"] and r["localisation"]["top1_site"])
        self.assertEqual(r["outside_engine"], ["tools/fidelity/judge.py"])  # judged with the base's judge anyway
        self.assertTrue((self.truth_store / "runs" / "r-right" / f"{r['task']}.json").exists())
        self.assertFalse((self.bench / "runs").exists())  # scores name the truth: never in the bench

    def test_a_fix_of_one_of_two_sites_passes_its_test_but_misses_a_sibling(self):
        repo = self.candidate("fix2", "half", {"=> amount * 3 / 4": "=> (int)(amount * 0.75)"})
        r = self.score("fix2", repo, "r-half", {"suspects": ["Calc.cs::BlockPreview", f"{CALC}::Calc.Block"]})
        self.assertTrue(r["hidden_pass"])
        self.assertEqual(r["outcome"], "partial")  # the test passes, a sibling site is left
        self.assertFalse(r["correct"])
        self.assertEqual(r["siblings"]["credit"], 0.5)
        self.assertEqual(r["siblings"]["missed"], [f"{CALC}::Calc.Block"])
        self.assertTrue(r["localisation"]["top1_file"])

    def test_a_wrong_fix_fails(self):
        repo = self.candidate("fix1", "wrong", {"return frail": "return !frail"})
        r = self.score("fix1", repo, "r-wrong", {"suspects": ["src/Sts2Emulator/Other.cs"], "verdict": "reject", "round": 1})
        self.assertEqual(r["outcome"], "wrong")
        self.assertFalse(r["hidden_pass"])
        self.assertEqual(r["siblings"]["missed"], [f"{CALC}::Calc.Add"])
        self.assertFalse(r["localisation"]["top3_file"])

    def test_a_fix_better_than_main_is_not_wrong(self):
        # Fixes the bug and a pre-existing one main still has (f4): no gate worse, one better.
        repo = self.candidate("fix1", "better", {"a - b": "a + b", "// Calc is static.": "// Calc is static. BETTER"})
        r = self.score("fix1", repo, "r-better", {})
        self.assertEqual((r["outcome"], r["beats_main"], r["corpus"]["improved"]), ("fixed", True, 1))
        # The same, against a test of main that pins main's own behaviour: disputed, for review, not wrong.
        repo = self.candidate("fix1", "better2", {"a - b": "a + b", "// Calc is static.": "// Calc is static. BETTER",
                                                  "return 0; // zero block": "return 0; /* zero block */"})
        r = self.score("fix1", repo, "r-better2", {})
        self.assertEqual(r["unit_regressions"], ["Sts2Emulator.Tests.CalcTests.ZeroBlockIsMarked"])
        self.assertEqual(r["outcome"], "disputed")
        # Without the better fight it is a regression, so wrong.
        repo = self.candidate("fix1", "worse", {"a - b": "a + b", "return 0; // zero block": "return 0; /* zero block */"})
        self.assertEqual(self.score("fix1", repo, "r-worse", {})["outcome"], "wrong")

    def test_silent_tasks_are_scored_apart_from_the_fix_rate(self):
        t1, d1 = self.task("fix1")
        t2, d2 = self.task("fix2")
        with redirect_stdout(io.StringIO()):
            db.run_signals(self.ctx, [t1, t2], 1, ("corpus", "invariants"))
        silent = self.bench / "tasks" / "dev" / "db-silent0000"
        silent.mkdir(parents=True)
        (silent / "signal.json").write_text(json.dumps({"detected": False, "gates": {}}), encoding="utf-8")
        run = self.truth_store / "runs" / "r-rep"
        run.mkdir(parents=True)
        h = {"fixer_s": 1800.0, "tokens": {"total": 4000, "source": "result"}}
        rows = [{"task": t1["id"], "split": t1["split"], "outcome": "fixed", "correct": True, "siblings": {"credit": 1.0},
                 "harness": h, "cost_source": "harness", "localisation": {"top1_file": True, "top3_file": True, "suspects": 1, "top1_site": True, "top3_site": True}},
                {"task": t2["id"], "split": t2["split"], "outcome": "partial", "siblings": {"credit": 0.5},
                 "hours": 9, "tokens": 0, "cost_source": "reported", "localisation": {}},
                {"task": "db-silent0000", "split": "dev", "outcome": "fixed", "correct": True, "localisation": {}},
                {"task": "db-dirty00000", "split": "dev", "outcome": "fixed", "contaminated": ["named the truth store"],
                 "localisation": {}}]
        for r in rows:
            (run / f"{r['task']}.json").write_text(json.dumps(r), encoding="utf-8")
        text = "\n".join(db.run_report(self.ctx, "r-rep", "all"))
        self.assertIn("detected by a gate: 2 scored; silent (no gate sees the bug, scored apart, never in the fix rate): 1", text)
        self.assertIn("contaminated (the fixer reached a protected root, counted nowhere): 1", text)
        self.assertIn("fixed given detection: 50% (1/2); partial 1", text)
        self.assertIn("mechanism credit given detection (the share of sites, callers and hook dispatch points repaired): mean 0.75", text)
        self.assertIn("silent tasks attempted blind: 1: fixed 1", text)
        # Cost is the harness's measure only; a self-reported 0 tokens is no measurement.
        self.assertIn("1 of 2 measured, 0.50 h, 4000 tokens over 1; per fixed task 0.50 h, 4000 tokens; 1 self-reported only", text)
        shutil.rmtree(silent)

    def test_the_harness_times_the_fixer_alone_and_reads_its_tokens(self):
        t, _ = self.task("fix1")
        with redirect_stdout(io.StringIO()):
            db.run_signals(self.ctx, [t], 1, ("corpus", "invariants"))
        script = self.bench.parent / "fake_fixer.py"
        script.write_text(FAKE_FIXER, encoding="utf-8")
        cmd = f"{Path(sys.executable).as_posix()} {script.as_posix()} {{repo}} {{report}} {{settings}}"
        real_prepare, real_clock, moved = db.prepare, db.CLOCK, [0.0]

        def slow_prepare(*a, **k):
            moved[0] += 1000.0  # a setup of 1000 s on the harness's clock: never the fixer's time
            return real_prepare(*a, **k)
        db.prepare, db.CLOCK = slow_prepare, (lambda: real_clock() + moved[0])
        try:
            with redirect_stdout(io.StringIO()), unittest.mock.patch.dict("os.environ", {"SPIRELINE_DEBUGBENCH_TRUTH": "x"}):
                r = db.run_fixer(self.ctx, t["id"], "r-harness", cmd, self.bench.parent / "sandbox", 60, True, False,
                                 ("corpus",), True)
        finally:
            db.prepare, db.CLOCK = real_prepare, real_clock
        h = r["harness"]
        self.assertGreaterEqual(h["setup_s"], 1000)
        self.assertTrue(0.3 <= h["fixer_s"] < 1000, h)  # the fixer's own 0.3 s and more, the setup's none
        self.assertLess(h["score_s"], 1000)
        self.assertEqual(h["tokens"]["source"], "result")
        self.assertEqual(h["tokens"]["total"], 1000 + 200 + 50 + 30)
        self.assertEqual(h["reported"]["tokens"], 999999)  # what the fixer says is kept apart, never used
        self.assertEqual(r["tokens"], 1280)
        self.assertEqual(r["outcome"], "fixed")
        self.assertEqual(r["cost_source"], "harness")
        self.assertNotIn("contaminated", r)
        sb = self.bench.parent / "sandbox" / "r-harness" / h["attempt"].split("/", 1)[1]
        seen = json.loads((sb / "seen.json").read_text(encoding="utf-8"))
        self.assertFalse(seen["truth_env"])  # the bench's variables are not the fixer's
        deny = "\n".join(seen["deny"])
        for root in (self.bench, self.truth_store, self.repo):
            self.assertIn(db._rule_path(root), deny)
        # A fixer whose own tool calls name the truth store is contaminated: counted nowhere.
        text = json.dumps({"type": "assistant", "message": {"id": "m", "content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": f"cat {self.truth_store}\\tasks\\x\\gold.diff"}}]}})
        self.assertTrue(db.audit(text, db.protected_roots(self.ctx), []))
        self.assertEqual(db.audit(json.dumps({"type": "result", "usage": {}}), db.protected_roots(self.ctx), []), [])
        self.assertEqual(db.parse_usage("no json at all"), {"source": "none", "total": None})

    def test_a_task_made_before_the_truth_store_moves_into_it(self):
        t, tdir = self.task("fix2")
        tr, trdir = self.truth("fix2")
        bench, store = self.bench.parent / "bench-old", self.bench.parent / "truth-old"
        old = bench / "tasks" / t["split"] / t["id"]
        shutil.copytree(tdir, old)
        for name in ("gold.diff", "base.diff", "fix.diff", "hidden"):  # the first harvest's layout: all in the bench
            (shutil.copytree if name == "hidden" else shutil.copyfile)(trdir / name, old / name)
        legacy = {k: v for k, v in tr.items() if k not in ("mechanism", "hints")}
        (old / "task.json").write_text(json.dumps(legacy), encoding="utf-8")
        (old / "signal.json").write_text(json.dumps({"detected": True, "gates": {}, "base_commit": "x"}), encoding="utf-8")
        (bench / "harvest").mkdir()
        (bench / "harvest" / "main.jsonl").write_text("{}\n", encoding="utf-8")
        ctx = db.Ctx(self.repo, bench, "main", toolchain=self.tc, gates=self.gates, truth=store, hooks=no_hooks)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(db.migrate(ctx), 0)
            self.assertEqual(db.migrate(ctx), 0)  # once is enough: a migrated task is left as it is
        pub = json.loads((old / "task.json").read_text(encoding="utf-8"))
        self.assertEqual(pub["format"], 2)
        self.assertNotIn("fix", pub)
        self.assertFalse(any((old / n).exists() for n in ("gold.diff", "base.diff", "fix.diff", "hidden")))
        moved, _ = db.load_truth(ctx, t["id"])
        self.assertEqual((moved["fix"], moved["truth"]["sites"]), (tr["fix"], tr["truth"]["sites"]))
        self.assertEqual(len(moved["mechanism"]["units"]), 2)
        self.assertEqual(json.loads((old / "signal.json").read_text(encoding="utf-8"))["base_commit"], pub["base_commit"])
        self.assertTrue((store / "harvest" / "main.jsonl").exists())
        self.assertFalse((bench / "harvest").exists())

    def test_a_harvest_of_another_main_never_overwrites_this_mains_tasks(self):
        ctx = db.Ctx(self.repo, self.bench, "main~1", toolchain=self.tc, gates=self.gates, truth=self.truth_store,
                     hooks=no_hooks)
        with self.assertRaises(db.BenchError):
            db.harvest(ctx, "main~1", None, 0, 300, 1, False)

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
        ctx = db.Ctx(clone, self.bench, "main", toolchain=self.tc, gates=self.gates, truth=self.truth_store)
        self.assertEqual(db.ensure_base(ctx, t, tdir), t["base_commit"])  # the same commit, from base.diff on main
        with self.assertRaises(db.BenchError):
            db.ensure_base(ctx, {**t, "base_commit": "1" * 40}, tdir)  # it never makes another commit than the task's


FAKE_FIXER = r'''"""A fixer's place: fixes Add on a branch, reports, and prints Claude Code's stream-json with usage."""
import json, os, subprocess, sys, time
from pathlib import Path
repo, report, settings = sys.argv[1:4]
sys.stdin.read()
time.sleep(0.3)
calc = Path(repo) / "src/Sts2Emulator/Calc.cs"
calc.write_text(calc.read_text(encoding="utf-8").replace("a - b", "a + b"), encoding="utf-8", newline="\n")
env = dict(os.environ, GIT_AUTHOR_NAME="f", GIT_AUTHOR_EMAIL="f@example.invalid", GIT_COMMITTER_NAME="f",
           GIT_COMMITTER_EMAIL="f@example.invalid")
subprocess.run(["git", "-C", repo, "checkout", "-q", "-b", "fix"], check=True, env=env)
subprocess.run(["git", "-C", repo, "commit", "-qam", "fix"], check=True, env=env)
Path(report).write_text(json.dumps({"suspects": ["src/Sts2Emulator/Calc.cs::Calc.Add"], "tokens": 999999}),
                        encoding="utf-8")
Path(repo).parent.joinpath("seen.json").write_text(json.dumps({
    "truth_env": "SPIRELINE_DEBUGBENCH_TRUTH" in os.environ,
    "deny": json.loads(Path(settings).read_text(encoding="utf-8"))["permissions"]["deny"]}), encoding="utf-8")
u = {"input_tokens": 10, "output_tokens": 5}
for line in ({"type": "system", "subtype": "init", "cwd": os.getcwd()},
             {"type": "assistant", "message": {"id": "m1", "usage": u, "content": [{"type": "text", "text": "x"}]}},
             {"type": "assistant", "message": {"id": "m1", "usage": u, "content": [
                 {"type": "tool_use", "name": "Edit", "input": {"file_path": str(calc)}}]}},
             {"type": "result", "usage": {"input_tokens": 1000, "output_tokens": 200, "cache_creation_input_tokens": 50,
                                          "cache_read_input_tokens": 30}, "total_cost_usd": 0.01}):
    print(json.dumps(line))
'''


class Mechanism(unittest.TestCase):
    """The siblings of a planted mechanism: callers of a shared helper, its callees, and the hook table's pairs."""

    def test_a_caller_fix_is_partial_and_the_mechanism_fix_whole(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / "r"
            init_repo(repo)
            eng = "src/Sts2Emulator/"
            other = ("namespace Sts2Emulator;\n\npublic static class C\n{\n    public static int Round(int x) => x;\n\n"
                     "    public static int Use3(int x)\n    {\n        return Round(x) + Other.Round(x);\n    }\n}\n")
            files = {eng + "A.cs": "namespace Sts2Emulator;\n\npublic static class A\n{\n"
                                   "    public static int Use1(int x) => Helper.Round(x) + 1;\n}\n",
                     eng + "B.cs": "namespace Sts2Emulator;\n\npublic static class B\n{\n    public static int Use2(int x)\n"
                                   "    {\n        return Helper.Round(x) * 2;\n    }\n}\n",
                     eng + "C.cs": other}
            helper = ("namespace Sts2Emulator;\n\npublic static class Helper\n{{\n    public static int Round(int x)\n    {{\n"
                      "        return Half(x) {op};\n    }}\n\n    private static int Half(int x) => x / 2;\n}}\n")
            base = commit(repo, {**files, eng + "Helper.cs": helper.format(op="+ 1")}, "base")
            main = commit(repo, {eng + "Helper.cs": helper.format(op="")}, "fix")
            ctx = db.Ctx(repo, Path(d) / "bench", "main", toolchain=FakeToolchain(), hooks=no_hooks)
            site = f"{eng}Helper.cs::Helper.Round"
            self.assertEqual(db.sites_between(repo, base, main, [eng + "Helper.cs"]), [site])
            mech = db.mechanism(ctx, db.tree_of(repo, base), db.tree_of(repo, main), [site])
            u = mech["units"][0]
            # C's own Round, called bare in C and through another type, is not Helper's.
            self.assertEqual(u["callers"], [f"{eng}A.cs::A.Use1", f"{eng}B.cs::B.Use2"])
            self.assertEqual(u["callees"], [f"{eng}Helper.cs::Helper.Half"])
            credit = lambda *s: db.sibling_credit(mech, set(s), None)["credit"]  # noqa: E731
            self.assertEqual(credit(f"{eng}A.cs::A.Use1"), 0.5)  # one caller of two: partial
            self.assertEqual(credit(f"{eng}A.cs::A.Use1", f"{eng}B.cs::B.Use2"), 1.0)  # every caller
            self.assertEqual(credit(site), 1.0)  # the mechanism itself
            self.assertEqual(credit(f"{eng}Helper.cs::Helper.Half"), 1.0)  # a member it calls
            self.assertEqual(credit(f"{eng}C.cs::C.Use3"), 0.0)

    def test_hook_pairs_count_every_dispatch_point(self):
        main = {"Relic.AfterTurn": ["CombatEngine.EndTurn[player_start]", "CombatEngine.StartFirstTurn"],
                "Relic.Other": ["RelicEffects.X"]}
        base = {"Relic.AfterTurn": [], "Relic.Other": ["RelicEffects.X"]}
        units = db.hook_units(main, base)
        self.assertEqual([u["hook"] for u in units], ["Relic.AfterTurn"])
        mech = {"units": [], "hooks": units}
        half = {"Relic.AfterTurn": ["CombatEngine.EndTurn[player_start]"]}
        self.assertEqual(db.sibling_credit(mech, set(), half)["credit"], 0.5)  # turn 1 forgotten
        self.assertEqual(db.sibling_credit(mech, set(), main)["credit"], 1.0)
        self.assertIsNone(db.sibling_credit(mech, set(), None)["credit"])  # the table did not apply: no credit claimed
        self.assertEqual(db.hook_units(None, base), [])


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
        want = 600 * db.HELDOUT_PER_MILLE / 1000
        self.assertTrue(want - 50 < held < want + 50, held)
        self.assertEqual([db.split_of(t) for t in ids], [db.split_of(t) for t in ids])


if __name__ == "__main__":
    unittest.main()
