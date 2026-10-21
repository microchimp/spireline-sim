"""tools/lab/hypotheses.py: the setup file parser, the specs, the verdict classifier, the result writer, the plan."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.lab import hypotheses as hyp  # noqa: E402
from tools.lab import mix  # noqa: E402

GOOD = {"id": "t-1", "claim": "c", "spec": {"setup": [{"cmd": "relic", "id": "VAJRA"}, {"cmd": "fight", "id": "NIBBITS_WEAK"}]},
        "seeds": 3, "compare": {"fields": [".hand"], "from_step": 1}}


def write(p: Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")


class ParseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_good_setup_is_normalised_and_takes_its_source_from_the_folder(self):
        s, why = hyp.parse_setup(GOOD, "audit-x")
        self.assertIsNone(why)
        self.assertEqual(("audit-x", ".hand", 1, True), (s["source"], s["fields"][0], s["from_step"], s["testable"]))

    def test_bad_setups_name_the_problem(self):
        for doc, word in (([], "object"), ({**GOOD, "id": "a b"}, "bad id"), ({**GOOD, "claim": ""}, "claim"),
                          ({**GOOD, "spec": {"setup": []}}, "setup list"), ({**GOOD, "compare": {"fields": ["hand"]}}, "fields"),
                          ({**GOOD, "compare": {}}, "fields"), ({**GOOD, "seeds": 0}, "seeds"), ({**GOOD, "seeds": True}, "seeds"),
                          ({**GOOD, "compare": {"fields": [".hand"], "from_step": -1}}, "from_step")):
            s, why = hyp.parse_setup(doc, "x")
            self.assertIsNone(s)
            self.assertIn(word, why)

    def test_not_testable_needs_no_spec(self):
        s, why = hyp.parse_setup({"id": "n", "claim": "c", "testable": False, "why": "needs a run"}, "x")
        self.assertEqual((None, False, "needs a run"), (why, s["testable"], s["why"]))

    def test_load_reads_every_source_folder_and_keeps_going_past_a_bad_file(self):
        write(self.tmp / "a" / "one.json", GOOD)
        write(self.tmp / "b" / "two.json", [{**GOOD, "id": "t-2"}, {**GOOD, "id": "t-1"}, {"id": "bad"}])
        write(self.tmp / "b" / "broken.json", "{nope")
        write(self.tmp / "loose.json", GOOD)  # not under a source folder: ignored
        setups, problems = hyp.load_setups(self.tmp)
        self.assertEqual(["t-1", "t-2"], sorted(s["id"] for s in setups))
        self.assertEqual(3, len(problems))
        self.assertTrue(any("duplicate id t-1" in p for p in problems))

    def test_specs_have_a_tag_derived_seeds_and_the_defaults_the_setup_overrides(self):
        s, _ = hyp.parse_setup({**GOOD, "spec": {**GOOD["spec"], "character": "REGENT"}}, "x")
        specs = hyp.to_specs([s])
        self.assertEqual(3, len(specs))
        self.assertEqual({"hyp:t-1"}, {x["tag"] for x in specs})
        self.assertEqual(3, len({x["seed"] for x in specs}))
        self.assertEqual({"REGENT"}, {x["character"] for x in specs})
        self.assertEqual({10}, {x["ascension"] for x in specs})
        self.assertEqual([x["seed"] for x in specs], [x["seed"] for x in hyp.to_specs([s])])  # stable
        listed, _ = hyp.parse_setup({**GOOD, "seeds": ["AAAAAAAAAA"]}, "x")
        self.assertEqual(["AAAAAAAAAA"], [x["seed"] for x in hyp.to_specs([listed])])
        nt, _ = hyp.parse_setup({"id": "n", "claim": "c", "testable": False}, "x")
        self.assertEqual([], hyp.to_specs([nt]))

    def test_the_shipped_setups_all_parse(self):
        root = Path("D:/spireline/referee/hypotheses")
        if not root.is_dir():
            self.skipTest("no hypotheses folder on this machine")
        setups, problems = hyp.load_setups(root)
        self.assertEqual([], problems)
        self.assertTrue(setups)


class ResultTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.s, _ = hyp.parse_setup(GOOD, "x")

    def v(self, **kw):
        return {"seed": "S", "tag": "hyp:t-1", **kw}

    def test_classify(self):
        c = lambda *vs: hyp.classify(self.s, list(vs))[0]  # noqa: E731
        self.assertEqual("confirmed", c(self.v(identical=True), self.v(identical=False, field=".hand", step=2)))
        self.assertEqual("not testable", c(self.v(identical=False, field=".hand", step=0)))  # before from_step
        self.assertEqual("not testable", c(self.v(identical=False, field=".enemies", step=2)))
        self.assertEqual("refuted", c(self.v(identical=True), self.v(identical=True)))
        self.assertEqual("not testable", c(self.v(identical=True, mirror_gap=1)))
        self.assertEqual("not testable", c(self.v(error="boom"), self.v(identical=True)))
        self.assertEqual("not testable", c())

    def test_judge_all_writes_one_entry_per_setup_with_fights_and_the_not_testable_ones(self):
        nt, _ = hyp.parse_setup({"id": "n", "claim": "c", "testable": False, "why": "w"}, "x")
        other, _ = hyp.parse_setup({**GOOD, "id": "t-2"}, "x")
        out = hyp.judge_all([self.s, nt, other], [self.v(identical=True), {"tag": "probe:z"}], "runs/j/fights/a.verdicts.json", "U", {})
        self.assertEqual({"t-1", "n"}, set(out))
        self.assertEqual(("refuted", 1, 1, "runs/j/fights/a.verdicts.json"),
                         (out["t-1"]["status"], out["t-1"]["fights"], out["t-1"]["attempts"], out["t-1"]["evidence"]))
        self.assertEqual(("not testable", "w"), (out["n"]["status"], out["n"]["reason"]))
        again = hyp.judge_all([self.s], [self.v(identical=True)], "e", "U", {"t-1": {"attempts": 1}})
        self.assertEqual(2, again["t-1"]["attempts"])

    def test_write_merges_and_never_overturns_a_decision(self):
        def e(st):
            return {"status": st, "reason": "r", "claim": "c", "source": "x", "utc": "U", "attempts": 1}
        hyp.write_results(self.tmp, {"a": e("confirmed"), "b": e("not testable")})
        cur = hyp.write_results(self.tmp, {"a": e("refuted"), "b": e("refuted"), "c": e("not testable")})
        self.assertEqual(("confirmed", "refuted", "not testable"), (cur["a"]["status"], cur["b"]["status"], cur["c"]["status"]))
        self.assertEqual(cur, hyp.read_results(self.tmp))
        self.assertEqual(["a", "b", "c"], list(json.loads((self.tmp / "hypotheses.json").read_text())))
        self.assertEqual({}, hyp.read_results(self.tmp / "none"))

    def test_pending_skips_decided_and_exhausted_setups(self):
        a, b, c, d = (hyp.parse_setup({**GOOD, "id": i}, "x")[0] for i in "abcd")
        res = {"a": {"status": "refuted"}, "b": {"status": "not testable", "attempts": hyp.MAX_ATTEMPTS},
               "c": {"status": "not testable", "attempts": 1}}
        self.assertEqual(["c", "d"], [s["id"] for s in hyp.pending([a, b, c, d], res)])

    def test_collect_reads_the_job_verdicts_and_updates_the_lab_file(self):
        setups, lab, job = self.tmp / "set", self.tmp / "lab", self.tmp / "lab" / "runs" / "J1"
        write(setups / "x" / "a.json", GOOD)
        write(job / "fights" / "n.verdicts.json", [self.v(identical=False, field=".hand", step=3)])
        self.assertEqual(1, hyp.collect(job, setups, lab, "U"))
        r = hyp.read_results(lab)["t-1"]
        self.assertEqual(("confirmed", "runs/J1/fights/n.verdicts.json"), (r["status"], r["evidence"]))

    def test_collect_judges_each_verdicts_file_once(self):
        setups, lab, job = self.tmp / "set", self.tmp / "lab", self.tmp / "lab" / "runs" / "J1"
        write(setups / "x" / "a.json", GOOD)
        write(job / "fights" / "b00.verdicts.json", [self.v(error="boom")])
        hyp.collect(job, setups, lab, "U")
        self.assertEqual(0, hyp.collect(job, setups, lab, "U"))
        self.assertEqual(1, hyp.read_results(lab)["t-1"]["attempts"])
        write(job / "fights" / "b01.verdicts.json", [self.v(identical=True)])
        hyp.collect(job, setups, lab, "U")
        r = hyp.read_results(lab)["t-1"]
        self.assertEqual(("refuted", 2, "runs/J1/fights/b01.verdicts.json"), (r["status"], r["attempts"], r["evidence"]))
        write(job / "fights" / "b02.verdicts.json", [])  # an empty batch (nothing testable) writes nothing
        self.assertEqual(0, hyp.collect(job, setups, lab, "U"))

    def test_doc_of_round_trips_through_the_parser(self):
        again, why = hyp.parse_setup(hyp.doc_of(self.s), "x")
        self.assertIsNone(why)
        self.assertEqual(self.s, again)


class MixTest(unittest.TestCase):
    def test_the_strategy_has_a_modest_weight(self):
        self.assertIn("hypotheses", mix.STRATEGIES)
        self.assertEqual(5, mix.DEFAULT_MIX["hypotheses"])


if __name__ == "__main__":
    unittest.main()
