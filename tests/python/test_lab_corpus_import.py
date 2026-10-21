"""tools/lab/corpus_import.py and prune tier 0: the lab fights into the corpus, and their raw recordings off the disk."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "python"))

from test_corpus import recording  # noqa: E402
from tools.fidelity import corpus as cp  # noqa: E402
from tools.lab import corpus_import as ci  # noqa: E402
from tools.lab import prune as pr  # noqa: E402

MATCH = {"identical": True, "mirror_gap": []}
DIVERGE = {"identical": False, "step": 2, "field": ".enemies", "mirror_gap": []}


def seeds(n, start=0):
    return [f"{chr(65 + (start + i) % 26)}{chr(65 + (start + i) // 26)}AAAAAAAA" for i in range(n)]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.lab = self.tmp / "lab"
        self.corpus = cp.Corpus(self.tmp / "corpus")

    def job(self, name, results, verdicts, base="b00"):
        d = self.lab / "runs" / name
        (d / "fights").mkdir(parents=True)
        (d / "fights" / f"{base}.out.json").write_text(json.dumps({"results": results}), encoding="utf-8")
        (d / "fights" / f"{base}.verdicts.json").write_text(json.dumps(verdicts), encoding="utf-8")
        (d / "done.rc").write_text("0\n", encoding="utf-8")
        return d

    def run_import(self, job, **kw):
        return ci.import_job(job, self.corpus, rand_cards=set(), **kw)


class ImportTest(Base):
    def test_matches_and_divergences_come_in_tagged_and_the_rest_is_left_out(self):
        s = seeds(5)
        err = dict(recording(seed=s[3]), error="idle_timeout")
        job = self.job("20260101T0000-fights",
                       [recording(seed=s[0]), recording(seed=s[1]), recording(seed=s[2]), err, recording(seed=s[4])],
                       [MATCH, DIVERGE, dict(MATCH, mirror_gap=["x"]), MATCH, {"error": "boom"}])
        c = self.run_import(job)
        self.assertEqual((c["match"], c["diverge"], c["left_out"]), (1, 1, 3))
        by = {m["seed"]: m for m in self.corpus.load_index()["fights"]}
        self.assertEqual(set(by), {s[0], s[1]})
        self.assertEqual(by[s[0]]["tags"], [])
        self.assertEqual(by[s[1]]["tags"], [ci.DIVERGED_TAG])
        self.assertTrue(by[s[0]]["sources"][0].startswith("lab/20260101T0000-fights/fights/"))
        self.assertNotIn(str(self.tmp), json.dumps(by))

    def test_no_diverge_keeps_matches_only(self):
        s = seeds(2)
        job = self.job("j", [recording(seed=s[0]), recording(seed=s[1])], [MATCH, DIVERGE])
        self.assertEqual(self.run_import(job, diverge=False)["diverge"], 0)
        self.assertEqual(len(self.corpus.load_index()["fights"]), 1)

    def test_one_fight_per_setup_across_jobs_and_runs_and_matches_win(self):
        a = recording(seed=seeds(1)[0])
        other_steps = recording(seed=a["seed"], hps=(27, 20, 10))  # same setup, another game side
        j1 = self.job("j1", [other_steps, a], [DIVERGE, MATCH])
        c = self.run_import(j1)
        self.assertEqual((c["match"], c["diverge"], c["duplicate"]), (1, 0, 1))
        j2 = self.job("j2", [recording(seed=a["seed"])], [MATCH])
        self.assertEqual(self.run_import(j2)["duplicate"], 1)
        self.assertEqual(len(self.corpus.load_index()["fights"]), 1)

    def test_size_budget(self):
        s = seeds(6)
        job = self.job("j", [recording(seed=x) for x in s], [MATCH] * 6)
        one = ci.gz_size(recording(seed=s[0]))
        c = self.run_import(job, budget=int(one * 2.5))
        self.assertEqual((c["match"], c["over_budget"]), (2, 4))
        self.assertLessEqual(self.corpus.size(), int(one * 2.5))
        c = self.run_import(job, budget=int(one * 2.5))  # full: nothing more
        self.assertEqual(c["match"], 0)

    def test_dry_run_writes_nothing_and_counts_across_jobs(self):
        s = seeds(3)
        j1 = self.job("j1", [recording(seed=s[0]), recording(seed=s[1])], [MATCH, MATCH])
        j2 = self.job("j2", [recording(seed=s[1]), recording(seed=s[2])], [MATCH, MATCH])
        st: dict = {}
        c1, c2 = self.run_import(j1, dry=True, state=st), self.run_import(j2, dry=True, state=st)
        self.assertEqual((c1["match"], c2["match"], c2["duplicate"]), (2, 1, 1))
        self.assertFalse(self.corpus.index_path.exists())
        self.assertFalse((j1 / ci.MARKER).exists())

    def test_no_marker_leaves_the_job_unmarked(self):
        job = self.job("j", [recording(seed=seeds(1)[0])], [MATCH])
        self.assertEqual(self.run_import(job, mark=False)["match"], 1)
        self.assertFalse((job / ci.MARKER).exists())
        self.assertEqual(len(self.corpus.load_index()["fights"]), 1)

    def test_marker_lists_the_setups_it_covers(self):
        s = seeds(2)
        job = self.job("j", [recording(seed=s[0]), recording(seed=s[1])], [MATCH, DIVERGE])
        self.run_import(job)
        doc = json.loads((job / ci.MARKER).read_text(encoding="utf-8"))
        (e,) = doc["files"].values()
        self.assertTrue(e["covered"])
        self.assertEqual(set(e["specs"]), self.corpus.specs())

    def test_shrink_replays_are_not_recordings(self):
        s = seeds(2)
        job = self.job("j", [recording(seed=s[0])], [MATCH])
        (job / "fights" / "b00.shrink0.out.json").write_text(json.dumps({"results": [recording(seed=s[1])]}), encoding="utf-8")
        (job / "fights" / "b00.shrink0.verdicts.json").write_text(json.dumps([MATCH]), encoding="utf-8")
        self.assertEqual(self.run_import(job)["match"], 1)

    def test_stored_fights_keep_their_ids(self):
        s = seeds(2)
        job = self.job("j", [recording(seed=s[0]), recording(seed=s[1])], [MATCH, DIVERGE])
        self.run_import(job)
        ids = {m["id"] for m in self.corpus.load_index()["fights"]}
        self.assertEqual(len(ids), 2)
        for fid in ids:
            self.assertEqual(cp.fight_id(self.corpus.read(fid)), fid)
        self.corpus.add_files([], rand_cards=set())  # the index loads and saves as before


class PruneTier0Test(Base):
    def prepare(self, covered_all=True):
        s = seeds(2)
        results = [recording(seed=s[0]), recording(seed=s[1])]
        job = self.job("20260101T0000-fights", results, [MATCH, DIVERGE if covered_all else {"error": "x"}])
        (job / "fights" / "b00.emu.json").write_text("x" * 3000, encoding="utf-8")
        self.run_import(job)
        return job

    def groups_after(self, job):
        g = self.lab / "groups.json"
        g.write_text("[]", encoding="utf-8")
        t = (job / "fights" / "b00.out.json").stat().st_mtime + 10
        os.utime(g, (t, t))

    def test_imported_recording_goes_before_scratch(self):
        job = self.prepare()
        (job / "serve").mkdir()
        (job / "serve" / "s.bin").write_bytes(b"z" * 100)
        self.groups_after(job)
        size = (job / "fights" / "b00.out.json").stat().st_size
        freed = pr.prune(self.lab, need=size)
        self.assertGreaterEqual(freed, size)
        self.assertFalse((job / "fights" / "b00.out.json").exists())
        self.assertTrue((job / "serve").exists())  # enough was freed: tier A never ran
        self.assertTrue((job / "fights" / "b00.verdicts.json").exists())
        self.assertTrue((job / ci.MARKER).exists())
        self.assertTrue(any(n["tier"] == "0" for n in json.loads((job / "pruned.json").read_text(encoding="utf-8"))))

    def test_nothing_goes_unless_covered_unchanged_read_and_stored(self):
        job = self.prepare(covered_all=False)  # one result had an error: not fully in the corpus
        self.groups_after(job)
        self.assertEqual(pr.tier_0(job, (self.lab / "groups.json").stat().st_mtime), 0)
        job2 = self.job("20260102T0000-fights", [recording(seed=seeds(1, 9)[0])], [MATCH])
        self.run_import(job2)
        out = job2 / "fights" / "b00.out.json"
        gt = out.stat().st_mtime + 10
        self.assertEqual(pr.tier_0(job2, None), 0)  # groups.json has not read it
        self.assertEqual(pr.tier_0(job2, out.stat().st_mtime - 1), 0)  # it read an older file
        out.write_text(out.read_text(encoding="utf-8") + " ", encoding="utf-8")  # changed since the import
        self.assertEqual(pr.tier_0(job2, gt + 100), 0)
        self.assertTrue(out.exists())

    def test_nothing_goes_when_the_corpus_lost_a_setup(self):
        job = self.prepare()
        idx = self.corpus.load_index()
        idx["fights"] = idx["fights"][:1]
        self.corpus.save_index(idx)
        self.assertEqual(pr.tier_0(job, time.time() + 100), 0)
        self.assertTrue((job / "fights" / "b00.out.json").exists())

    def test_nothing_goes_when_the_corpus_is_gone(self):
        job = self.prepare()
        shutil.rmtree(self.corpus.path)
        self.assertEqual(pr.tier_0(job, time.time() + 100), 0)
        self.assertTrue((job / "fights" / "b00.out.json").exists())


if __name__ == "__main__":
    unittest.main()
