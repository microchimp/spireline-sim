"""tools.fidelity.nightly: tonight's main against the previous main, and the bisect of a fight that got worse.

The night is judged by the corpus gate itself (corpus.check), so it fails as the gate fails: a fight
worse, a fight the previous run had and tonight's did not compare, EmuDump failures; with nothing to
compare with it is not a pass either. The bisect is git's own (`git bisect run`), driven here over a
scratch repository whose test fails from one known commit on.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import corpus as cp  # noqa: E402
from tools.fidelity import nightly as nt  # noqa: E402

LAST = {"f1": {"n": 4, "fd": None, "rv": "MMM"}, "f2": {"n": 4, "fd": 2, "rv": "MDM", "dp": {"2": [".enemies"]}},
        "f3": {"n": 3, "fd": None, "rv": "MM"}}


def sh(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def scratch_repo(d: Path, n: int) -> tuple[Path, list]:
    repo = d / "repo"
    repo.mkdir()
    sh(repo, "init", "-q")
    sh(repo, "config", "user.email", "t@example.invalid")
    sh(repo, "config", "user.name", "t")
    commits = []
    for i in range(n):
        (repo / "value.txt").write_text(f"bad {i}\n" if i >= 6 else f"good {i}\n", encoding="utf-8")
        sh(repo, "add", "value.txt")
        sh(repo, "commit", "-q", "-m", f"commit {i}")
        commits.append(sh(repo, "rev-parse", "HEAD"))
    return repo, commits


class NightTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.repo, self.commits = scratch_repo(d, 3)
        self.corpus, self.state = d / "corpus", d / "state"
        self.corpus.mkdir()
        self.state.mkdir()
        # The corpus still holds every fight (a fight missing from a run is then a fight not compared).
        (self.corpus / "index.json").write_text(json.dumps({"format": cp.INDEX_FORMAT, "fights": [
            {"id": k, "seed": k, "steps": v["n"], "needs": []} for k, v in LAST.items()], "scanned": {}, "excluded": {}}),
            encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def run_doc(self, fights: dict, commit: str, label: str, **over) -> Path:
        doc = {"format": cp.RESULTS_FORMAT, "label": label, "subset": False, "problems": [], "skipped": {},
               "emulator": {"commit": commit, "dirty": False, "judge_sha256": "j", "stale_build": False},
               "totals": cp.totals(fights), "fights": fights}
        doc.update(over)
        p = self.corpus / "runs" / label / "results.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc), encoding="utf-8")
        return p

    def night(self, fights: dict, **over) -> tuple[int, dict]:
        out = io.StringIO()
        with redirect_stdout(out):
            return nt.night(self.repo, self.state, self.corpus, self.commits[2],
                            self.run_doc(fights, self.commits[2], "tonight", **over), bisect_max=0)

    def last_night(self, fights: dict) -> None:
        p = self.run_doc(fights, self.commits[1], "last")
        (self.state / "last.json").write_text(json.dumps({"commit": self.commits[1], "run": str(p)}), encoding="utf-8")

    def test_the_same_night_passes_and_becomes_the_next_comparison(self):
        self.last_night(LAST)
        code, rep = self.night(dict(LAST))
        self.assertEqual(code, 0, rep.get("gate_text"))
        self.assertIn("last night's run", rep["previous"])
        last = json.loads((self.state / "last.json").read_text(encoding="utf-8"))
        self.assertEqual(last["commit"], self.commits[2])
        md = next(self.state.glob("*.md")).read_text(encoding="utf-8")
        self.assertIn("1. identical fights: 2 -> 2 of 3", md)

    def test_a_fight_that_got_worse_fails_the_night(self):
        self.last_night(LAST)
        code, rep = self.night(dict(LAST, f1={"n": 4, "fd": 3, "rv": "MMD"}))
        self.assertEqual(code, 1)
        self.assertEqual([r["id"] for r in rep["compare"]["regressions"]], ["f1"])
        self.assertTrue(nt.worse(LAST["f1"], {"n": 4, "fd": 3, "rv": "MMD"}))
        self.assertFalse(nt.worse(LAST["f2"], {"n": 4, "fd": None, "rv": "MMM"}))

    def test_a_fight_tonight_did_not_compare_fails_the_night(self):
        self.last_night(LAST)
        code, rep = self.night({k: v for k, v in LAST.items() if k != "f3"})
        self.assertEqual(code, 1)
        self.assertIn("NOT COMPARED", rep["gate_text"])

    def test_emudump_failures_fail_the_night_and_are_not_the_next_comparison(self):
        self.last_night(LAST)
        code, rep = self.night(dict(LAST), problems=["EmuDump fight exit 3"])
        self.assertEqual(code, 1)
        self.assertIn("UNSOUND", rep["gate_text"])
        self.assertEqual(json.loads((self.state / "last.json").read_text(encoding="utf-8"))["commit"], self.commits[1])

    def test_a_first_night_compares_with_the_nearest_ancestor_baseline(self):
        (self.corpus / "baselines").mkdir()
        base = {"format": cp.RESULTS_FORMAT, "commit": self.commits[0], "run": "old", "judge_sha256": "j",
                "totals": cp.totals(LAST), "fights": LAST}
        (self.corpus / "baselines" / f"{self.commits[0]}.json").write_text(json.dumps(base), encoding="utf-8")
        code, rep = self.night(dict(LAST, f3={"n": 3, "fd": 1, "rv": "DD"}))
        self.assertEqual(code, 1)
        self.assertEqual(rep["previous"], f"corpus baseline {self.commits[0][:10]}")
        self.assertEqual(rep["commits"], 2)

    def test_nothing_to_compare_with_is_not_a_pass(self):
        code, rep = self.night(dict(LAST))
        self.assertEqual(code, 2)
        self.assertIn("this run is the first", rep["previous"])


class BisectTest(unittest.TestCase):
    def test_git_bisect_run_finds_the_first_commit_the_test_fails_at(self):
        with tempfile.TemporaryDirectory() as d:
            repo, commits = scratch_repo(Path(d), 9)
            test = [sys.executable, "-c", "import sys; sys.exit(1 if open('value.txt').read().startswith('bad') else 0)"]
            res = nt.bisect_run(repo, commits[1], commits[8], test)
            self.assertEqual(res["first_bad"], commits[6])
            self.assertEqual(res["subject"], "commit 6")
            self.assertEqual(sh(repo, "rev-parse", "HEAD"), commits[8], "the bisect is reset afterwards")

    def test_a_regression_that_does_not_reproduce_blames_no_commit(self):
        # git bisect takes the bad end on trust: with a test that passes everywhere (the judge changed since
        # the night, a flake) it named the bad end, never tested, as the first bad commit.
        with tempfile.TemporaryDirectory() as d:
            repo, commits = scratch_repo(Path(d), 9)
            res = nt.bisect_run(repo, commits[1], commits[8], [sys.executable, "-c", "import sys; sys.exit(0)"])
            self.assertIsNone(res["first_bad"])
            self.assertIn("not reproduced", res["error"])
            self.assertEqual(sh(repo, "rev-parse", "HEAD"), commits[8])
            self.assertEqual(sh(repo, "symbolic-ref", "--short", "HEAD"), sh(repo, "branch", "--show-current"))

    def test_a_good_end_that_is_already_bad_or_an_untestable_end_blames_no_commit(self):
        with tempfile.TemporaryDirectory() as d:
            repo, commits = scratch_repo(Path(d), 9)
            res = nt.bisect_run(repo, commits[1], commits[8], [sys.executable, "-c", "import sys; sys.exit(1)"])
            self.assertIsNone(res["first_bad"])
            self.assertIn("already worse", res["error"])
            res = nt.bisect_run(repo, commits[1], commits[8], [sys.executable, "-c", f"import sys; sys.exit({nt.SKIP})"])
            self.assertIsNone(res["first_bad"])
            self.assertIn("cannot be tested", res["error"])
            self.assertFalse((repo / ".git" / "BISECT_LOG").exists(), "no bisect left running")


if __name__ == "__main__":
    unittest.main()
