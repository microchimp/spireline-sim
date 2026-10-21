import json
import tempfile
import unittest
from pathlib import Path

from tools.lab import mix, retest


def group(gid, seen, spec=True, fields=(".hand",)):
    return {"id": gid, "first_seen_utc": seen, "retest": {"state": "pending", "fields": list(fields),
                                                         "spec": {"seed": gid, "setup": [{"cmd": "fight", "id": "X"}]} if spec else None}}


class RetestTests(unittest.TestCase):
    def test_pending_is_oldest_first_and_skips_decided_and_exhausted(self):
        groups = [group("G3", "2026-10-03"), group("G1", "2026-10-01"), group("G2", "2026-10-02"), group("G4", "2026-10-04"),
                  group("G5", "2026-10-05", spec=False), {"id": "G6", "first_seen_utc": "2026-10-01"}]
        results = {"G1": {"verdict": "identical", "attempts": 1}, "G2": {"verdict": "unclear", "attempts": 3}, "G3": {"verdict": "unclear", "attempts": 1}}
        self.assertEqual(["G3", "G4"], [x["id"] for x in retest.pending(groups, results)])
        self.assertEqual(["G3"], [x["id"] for x in retest.pending(groups, results, limit=1)])

    def test_specs_are_tagged_with_the_group(self):
        self.assertEqual("retest:G1", retest.to_specs(retest.pending([group("G1", "a")], {}))[0]["tag"])

    def test_classify(self):
        self.assertEqual("identical", retest.classify([{"identical": True}])[0])
        self.assertEqual("unclear", retest.classify([{"identical": True, "mirror_gap": True}])[0])
        self.assertEqual("unclear", retest.classify([{"error": "x"}])[0])
        self.assertEqual("unclear", retest.classify([])[0])
        self.assertEqual("diverges", retest.classify([{"field": ".hand", "step": 3}], [".hand"])[0])
        self.assertEqual("diverges", retest.classify([{"field": ".hand[2].cost", "step": 3}], [".hand"])[0])
        # a first difference in a field that is not the group's is no evidence of the group's symptom
        self.assertEqual("unclear", retest.classify([{"field": ".hand", "step": 3}], [".draw"])[0])
        self.assertEqual("unclear", retest.classify([{"field": ".handsize", "step": 3}], [".hand"])[0])
        self.assertEqual("unclear", retest.classify([{"field": ".hand", "step": 3}])[0])

    def test_collect_judges_each_verdicts_file_once_and_keeps_decided(self):
        with tempfile.TemporaryDirectory() as td:
            lab, job = Path(td) / "lab", Path(td) / "lab" / "runs" / "J1"
            (job / "retest-in").mkdir(parents=True)
            (job / "fights").mkdir()
            items = retest.pending([group("G1", "a"), group("G2", "b"), group("G3", "c", fields=[".draw"])], {})
            (job / "retest-in" / "pending.json").write_text(json.dumps(items), encoding="utf-8")
            (job / "fights" / "b.verdicts.json").write_text(json.dumps(
                [{"tag": "retest:G1", "identical": True}, {"tag": "retest:G2", "field": ".hand", "step": 2},
                 {"tag": "retest:G3", "field": ".hand", "step": 2}]), encoding="utf-8")
            self.assertEqual(3, retest.collect(job, lab, "u1", "sha"))
            self.assertEqual(0, retest.collect(job, lab, "u2", "sha"))  # once
            r = retest.read_results(lab)
            self.assertEqual(("identical", "diverges", 1), (r["G1"]["verdict"], r["G2"]["verdict"], r["G2"]["attempts"]))
            self.assertEqual("unclear", r["G3"]["verdict"])  # G3's symptom is in .draw: a .hand difference does not decide it
            retest.write_results(lab, {"G1": {"verdict": "unclear", "attempts": 2}})
            self.assertEqual("identical", retest.read_results(lab)["G1"]["verdict"])

    def test_it_is_a_strategy_with_a_modest_weight(self):
        self.assertIn("retest", mix.STRATEGIES)
        self.assertLessEqual(mix.DEFAULT_MIX["retest"], 5)


if __name__ == "__main__":
    unittest.main()
