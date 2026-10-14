"""corpus.py check FAILS on a deliberately broken run: a regression, a baseline fight not compared, a
check that compared nothing, a stale build, EmuDump failures; and it reads recordings in nested folders
(fixtures: broken_cases.py).
"""

from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from broken_cases import fight_pair  # noqa: E402
from tools.fidelity import corpus as cp  # noqa: E402



class CorpusCheckBrokenCases(unittest.TestCase):
    BASE = {"f1": {"n": 3, "fd": None, "rv": "MM"}, "f2": {"n": 3, "fd": 1, "rv": "DM", "dp": {"1": [".enemies"]}}}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = cp.Corpus(Path(self.tmp.name) / "corpus")
        bdir = self.corpus.path / "baselines"
        bdir.mkdir(parents=True)
        self.bp = bdir / "abc.json"
        self.bp.write_text(json.dumps({"format": cp.RESULTS_FORMAT, "commit": "abc", "run": "r0", "judge_sha256": "j",
                                       "totals": cp.totals(self.BASE), "fights": self.BASE}), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, fights, **doc) -> tuple[int, str]:
        run_doc = {"format": cp.RESULTS_FORMAT, "label": "t", "fights": fights, "totals": cp.totals(fights),
                   "subset": False, "emulator": {"judge_sha256": "j"}, **doc}
        out = io.StringIO()
        with redirect_stdout(out):
            code = cp.check(self.corpus, run_doc, self.bp)
        return code, out.getvalue()

    def test_the_same_results_pass(self):
        self.assertEqual(self.check(copy.deepcopy(self.BASE))[0], 0)

    def test_a_regression_fails(self):
        self.assertEqual(self.check(dict(self.BASE, f1={"n": 3, "fd": 2, "rv": "MD"}))[0], 1)

    def test_a_baseline_fight_not_compared_fails(self):
        self.assertEqual(self.check({"f1": self.BASE["f1"]})[0], 1)

    def test_a_check_that_compared_nothing_fails(self):
        # Every fight is "new": the run and the baseline share none, which is no check at all.
        code, text = self.check({"x1": {"n": 3, "fd": None, "rv": "MM"}}, subset=True)
        self.assertEqual(code, 1)
        self.assertIn("UNSOUND: no baseline fight was compared", text)

    def test_a_stale_build_or_emudump_failures_fail(self):
        self.assertEqual(self.check(copy.deepcopy(self.BASE), emulator={"judge_sha256": "j", "stale_build": True})[0], 1)
        self.assertEqual(self.check(copy.deepcopy(self.BASE), problems=["EmuDump fight exit 1, 0 of 2 results"])[0], 1)

    def test_recordings_in_nested_folders_are_read(self):
        # The corpus once skipped 219 files in subfolders and passed with their fights missing.
        jobs = Path(self.tmp.name) / "jobs"
        for rel in ("jobs/a.out.json", "jobs/wave/b.out.json", "jobs/wave/deep/c.out.json", "jobs-setup/x/d.out.json"):
            p = Path(self.tmp.name) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            g, _ = fight_pair(seed=p.stem.split(".")[0].upper() * 10)
            g.update(character="IRONCLAD", neow_pick=0, deck_add=[], room=[0, 1])
            p.write_text(json.dumps({"results": [g]}), encoding="utf-8")
        self.assertEqual(len(cp.recordings(jobs)), 4)


if __name__ == "__main__":
    unittest.main()
