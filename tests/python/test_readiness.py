"""tools/fidelity/readiness.py on a fixture referee tree."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import readiness as rd  # noqa: E402


def w(p: Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")


def fixture(root: Path) -> None:
    per = {"1": ["complete", 30, ""], "2": ["mismatch", 5, "EMULATOR: deck"], "3": ["mismatch", 20, "EMULATOR: deck"],
           "4": ["mismatch", 40, "WALKER: flow"]}
    cx = {"runs": 4, "complete": 1, "judged": 4, "judged_matching_pct": 25.0, "components_at_zero": ["rest"],
          "stops_by_class": {"EMULATOR": 2}, "stops": {"EMULATOR: deck": 2, "WALKER: flow": 1}, "per_run": per}
    w(root / "refset/v2/scores/all-old.json",
      {"scored": "2026-10-01 10:00", "label": "old", "codex": {"runs": 4, "complete": 4, "stops": {}}})
    w(root / "refset/v2/scores/all-new.json", {"scored": "2026-10-20 10:00", "label": "new", "codex": cx})
    w(root / "refset/v2/scores/all-skip.json", {"scored": "2026-10-21 10:00", "codex": {"skipped": "x"}})
    w(root / "corpus/baselines/a.json",
      {"made": "2026-10-19T01:00:00", "totals": {"fights": 10, "identical": 7, "reanchored": 20, "reanchor_matched": 15}})
    w(root / "lab/groups.json", {"generated_utc": "2026-10-20T00:00:00Z", "groups": [
        {"id": "Ga", "title": "a", "impact_runs_est": 5, "host_suspect": True, "backlog_ids": ["C1"]},
        {"id": "Gb", "title": "b", "impact_runs_est": 9, "backlog_ids": []},
        {"id": "Gc", "title": "c", "impact_runs_est": 1}]})
    w(root / "audit/data-checks/t1.txt", "x\nTOTAL: 46 difference(s)\n")
    w(root / "hypotheses/a/a.json", [{"id": "h1", "claim": "c"}, {"id": "h2", "claim": "c", "testable": False}, {"id": "h3"}])
    w(root / "lab/hypotheses.json", {"h1": {"status": "confirmed"}, "h3": {"status": "refuted"}})


class T(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = Path(self.d.name)
        fixture(self.root)

    def tearDown(self):
        self.d.cleanup()

    def test_headline_and_codex(self):
        r = rd.build(self.root)
        self.assertEqual(r["headline_pct"], 25.0)
        c = r["codex"]
        self.assertEqual(c["label"], "new")
        self.assertEqual(c["first_divergence_floor_median"], 20)
        self.assertEqual(c["stopped_by_act"], {"1": 1, "2": 1, "3": 1})
        self.assertEqual(c["deepest_act_stopped"], 3)

    def test_subscores(self):
        r = rd.build(self.root)
        self.assertEqual(r["corpus"]["identical_pct"], 70.0)
        self.assertEqual(r["data"]["mismatches"], 46)
        self.assertEqual(r["lab"]["host_suspect"], 1)
        self.assertEqual(r["lab"]["open_groups"], 2)
        h = r["hypotheses"]
        self.assertEqual((h["confirmed"], h["refuted"], h["open"], h["not_testable"]), (1, 1, 1, 1))

    def test_blockers_sorted(self):
        r = rd.build(self.root, top=2)
        self.assertEqual([g["id"] for g in r["blockers"]], ["Gb", "Ga"])
        self.assertEqual(r["codex_stop_causes"][0], {"cause": "EMULATOR: deck", "runs": 2})

    def test_missing_inputs(self):
        with tempfile.TemporaryDirectory() as e:
            r = rd.build(Path(e))
            self.assertIsNone(r["headline_pct"])
            self.assertEqual(len(r["notes"]), 5)
            self.assertIn("n/a", rd.render(r))

    def test_cli_writes_json(self):
        out = self.root / "o" / "r.json"
        self.assertEqual(rd.main(["--referee", str(self.root), "--json", str(out)]), 0)
        self.assertEqual(json.loads(out.read_text())["headline_pct"], 25.0)


if __name__ == "__main__":
    unittest.main()
