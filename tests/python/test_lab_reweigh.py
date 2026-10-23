"""tools/lab/reweigh.py: the weekly re-weighting of the strategy mix (docs/PLAN.md section 7, rule D1).

Lab folders in a temporary directory only: no game, no emulator, no network, no live lab file.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.lab import reweigh as rw  # noqa: E402

NOW = "2026-10-24T00:00:00Z"
HEAD = "\t".join(["utc", "job", "strategy", "main", "rc", "wall_min", "cpu_min", "fights", "mismatches", "unbuildable",
                  "groups", "fix_ready", "screen_groups"])
JOBS = {"_note": "test", "selfplay": {"enabled": True, "cmd": ["{python}", "rl/gen0.py"], "share": 0.5, "timeout_min": 45},
        "train": {"enabled": False}}


def at(days_before: float) -> str:
    """A UTC time that many days before NOW (negative: after it)."""
    t = dt.datetime(2026, 10, 24, tzinfo=dt.timezone.utc) - dt.timedelta(days=days_before)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def row(when: str, strategy: str, cpu: float = 20.0, fix_ready: object = 0) -> str:
    """A yield.tsv row with the 13 columns of the live file."""
    return "\t".join([when, f"{when}-{strategy}", strategy, "abc1234", "0", "50.0", f"{cpu:.1f}", "300", "10", "0", "20",
                      str(fix_ready), "0"])


def yield_text(*rows: str) -> str:
    return "\n".join([HEAD, *rows]) + "\n"


class ReweighCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.lab = self.tmp / "lab"
        self.lab.mkdir()

    def put(self, name: str, text: str) -> None:
        (self.lab / name).write_text(text, encoding="utf-8")

    def put_json(self, name: str, doc: object) -> None:
        self.put(name, json.dumps(doc, indent=1) + "\n")

    def text(self, name: str) -> str:
        return (self.lab / name).read_text(encoding="utf-8")

    def plan(self) -> dict:
        return rw.plan(self.lab, rw.parse_utc(NOW))

    def by_name(self, p: dict) -> dict:
        return {r["strategy"]: r for r in p["table"]}

    def cli(self, *extra: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = rw.main(["--lab", str(self.lab), "--now", NOW, *extra])
        return rc, out.getvalue(), err.getvalue()


class VerdictTest(ReweighCase):
    def test_a_fights_strategy_with_no_fix_ready_halves_and_one_with_fix_ready_keeps(self):
        self.put_json("mix.json", {"aimed-bugs": 30, "random": 5})
        self.put_json("jobs.json", JOBS)
        self.put("yield.tsv", yield_text(*[row(at(d), "aimed-bugs", 50, 0) for d in (1, 3, 5)],
                                         *[row(at(d), "random", 20, 4) for d in (2, 4, 6)]))
        t = self.by_name(self.plan())
        self.assertEqual(15, t["aimed-bugs"]["new"])
        self.assertTrue(t["aimed-bugs"]["verdict"].startswith("halved"), t["aimed-bugs"]["verdict"])
        self.assertEqual(5, t["random"]["new"])
        self.assertTrue(t["random"]["verdict"].startswith("kept"), t["random"]["verdict"])

    def test_under_the_evidence_floor_a_strategy_is_not_judged(self):
        # 2 jobs and 40 CPU min: less than 3 jobs and less than 60 CPU min, so it keeps its weight even at 0 fix_ready
        self.put_json("mix.json", {"aimed-gaps": 25, "dense": 10})
        self.put_json("jobs.json", JOBS)
        self.put("yield.tsv", yield_text(row(at(1), "aimed-gaps", 20, 0), row(at(2), "aimed-gaps", 20, 0),
                                         row(at(1), "dense", 35, 0), row(at(2), "dense", 35, 0)))
        t = self.by_name(self.plan())
        self.assertEqual(25, t["aimed-gaps"]["new"])
        self.assertTrue(t["aimed-gaps"]["verdict"].startswith("not judged"), t["aimed-gaps"]["verdict"])
        # the same 2 jobs with 70 CPU min pass the other floor: judged, and halved
        self.assertEqual(5, t["dense"]["new"])
        self.assertTrue(t["dense"]["verdict"].startswith("halved"), t["dense"]["verdict"])

    def test_a_strategy_with_only_na_fix_ready_is_not_judged(self):
        self.put_json("mix.json", {"under-tested": 10})
        self.put_json("jobs.json", JOBS)
        self.put("yield.tsv", yield_text(*[row(at(d), "under-tested", 30, "NA") for d in (1, 2, 3)]))
        t = self.by_name(self.plan())
        self.assertEqual(10, t["under-tested"]["new"])
        self.assertIn("no fix_ready figure", t["under-tested"]["verdict"])


class ShareTest(ReweighCase):
    def test_the_share_formula_on_a_worked_example(self):
        # fights weights 30 + 10 + 10 = 50 with selfplay at 0.5; aimed-bugs halves to 15, so F' = 35
        self.put_json("mix.json", {"aimed-bugs": 30, "random": 10, "dense": 10})
        self.put_json("jobs.json", JOBS)
        self.put("yield.tsv", yield_text(*[row(at(d), "aimed-bugs", 20, 0) for d in (1, 2, 3)],
                                         *[row(at(d), "random", 20, 2) for d in (1, 2, 3)],
                                         *[row(at(d), "dense", 20, 1) for d in (1, 2, 3)]))
        p = self.plan()
        self.assertAlmostEqual(0.65, p["share_new"])  # 1 - 0.5 * 35 / 50
        self.assertEqual(50, p["F"])
        self.assertEqual(35, p["F2"])
        # an unchanged strategy keeps its absolute CPU share: random 0.5 * 10 / 50 before, 0.35 * 10 / 35 after
        before = (1 - p["share_old"]) * 10 / p["F"]
        after = (1 - p["share_new"]) * 10 / p["F2"]
        self.assertAlmostEqual(before, after)
        self.assertAlmostEqual(0.10, after)

    def test_the_share_is_capped_at_0_9_and_unchanged_when_no_weight_moves(self):
        self.assertEqual((0.9, True), rw.new_share(0.95, 50, 49))
        self.assertEqual((0.5, False), rw.new_share(0.5, 50, 50))
        share, capped = rw.new_share(0.5, 50, 35)
        self.assertAlmostEqual(0.65, share)
        self.assertFalse(capped)


class ApplyTest(ReweighCase):
    def test_apply_writes_backups_and_keeps_the_notes(self):
        original_mix = json.dumps({"_note": "keep me", "aimed-bugs": 30, "random": 5}, indent=1) + "\n"
        original_jobs = json.dumps(JOBS, indent=1) + "\n"
        self.put("mix.json", original_mix)
        self.put("jobs.json", original_jobs)
        self.put("yield.tsv", yield_text(*[row(at(d), "aimed-bugs", 50, 0) for d in (1, 2, 3)],
                                         *[row(at(d), "random", 20, 4) for d in (1, 2, 3)]))
        rc, out, err = self.cli("--apply")
        self.assertEqual(0, rc, err)
        # the backups hold the originals, under the dated names
        self.assertEqual(original_mix, self.text("mix.json.2026-10-24.bak"))
        self.assertEqual(original_jobs, self.text("jobs.json.2026-10-24.bak"))
        mix_doc = json.loads(self.text("mix.json"))
        self.assertEqual("keep me", mix_doc["_note"])
        self.assertEqual(15, mix_doc["aimed-bugs"])
        self.assertEqual(5, mix_doc["random"])  # random has fix_ready 12: kept
        self.assertTrue(mix_doc["_reweigh"].startswith("2026-10-24"))
        self.assertIn("aimed-bugs: halved", mix_doc["_reweigh"])
        jobs_doc = json.loads(self.text("jobs.json"))
        self.assertEqual(0.7143, jobs_doc["selfplay"]["share"])  # 1 - 0.5 * 20 / 35
        self.assertEqual(["{python}", "rl/gen0.py"], jobs_doc["selfplay"]["cmd"])  # the rest of selfplay kept
        self.assertEqual({"enabled": False}, jobs_doc["train"])
        # a second run the same day keeps the first backup and makes a numbered one
        rc, _, err = self.cli("--apply")
        self.assertEqual(0, rc, err)
        self.assertEqual(original_mix, self.text("mix.json.2026-10-24.bak"))
        self.assertTrue((self.lab / "mix.json.2026-10-24.2.bak").exists())

    def test_the_dry_run_writes_nothing_and_prints_the_table(self):
        original_mix = json.dumps({"aimed-bugs": 30, "random": 5}) + "\n"
        self.put("mix.json", original_mix)
        self.put_json("jobs.json", JOBS)
        self.put("yield.tsv", yield_text(*[row(at(d), "aimed-bugs", 50, 0) for d in (1, 2, 3)]))
        rc, out, err = self.cli()
        self.assertEqual(0, rc, err)
        self.assertEqual(original_mix, self.text("mix.json"))
        self.assertFalse(any(p.name.endswith(".bak") for p in self.lab.iterdir()))
        self.assertIn("aimed-bugs", out)
        self.assertIn("halved", out)
        self.assertIn("selfplay share: 0.5000 -> 0.7143 (fights weight 35 -> 20)", out)  # random is not judged: kept at 5
        self.assertNotIn(str(self.tmp), out)  # no absolute paths in the output

    def test_the_agent_jobs_are_never_halved_and_an_unknown_key_is_left_alone(self):
        self.put_json("mix.json", {"selfplay": 5, "aimed-bugs": 30})
        self.put_json("jobs.json", JOBS)
        self.put("yield.tsv", yield_text(*[row(at(d), "selfplay", 60, 0) for d in (1, 2, 3, 4)],
                                         *[row(at(d), "aimed-bugs", 50, 3) for d in (1, 2, 3)]))
        p = self.plan()
        self.assertNotIn("selfplay", self.by_name(p))
        self.assertTrue(any("unknown key 'selfplay'" in n for n in p["notes"]), p["notes"])
        self.assertEqual(30, self.by_name(p)["aimed-bugs"]["new"])
        self.assertEqual(0.5, p["share_new"])  # nothing moved, so the selfplay share stays

    def test_a_missing_yield_file_changes_nothing(self):
        self.put_json("mix.json", {"aimed-bugs": 30, "random": 5})
        self.put_json("jobs.json", JOBS)
        p = self.plan()
        self.assertEqual({"aimed-bugs": 30, "random": 5}, {r["strategy"]: r["new"] for r in p["table"]})
        self.assertEqual(0.5, p["share_new"])
        self.assertTrue(any("yield.tsv is missing" in n for n in p["notes"]))


class WindowTest(ReweighCase):
    def test_rows_outside_the_window_are_ignored(self):
        # the window is (now - 14 days, now]: 15, 30 and 14 days back and a time after NOW are out; NOW itself is in
        self.put_json("mix.json", {"aimed-bugs": 30})
        self.put_json("jobs.json", JOBS)
        self.put("yield.tsv", yield_text(
            row(at(15), "aimed-bugs", 500, 9), row(at(30), "aimed-bugs", 500, 9), row(at(14), "aimed-bugs", 500, 9),
            row(at(-1), "aimed-bugs", 500, 9),
            row(at(1), "aimed-bugs", 10, 0), row(at(2), "aimed-bugs", 10, 0), row(at(0), "aimed-bugs", 10, 0)))
        p = self.plan()
        t = self.by_name(p)["aimed-bugs"]
        self.assertEqual(3, t["jobs"])
        self.assertEqual(30.0, t["cpu"])
        self.assertEqual(0, t["fix_ready"])
        self.assertEqual(4, p["outside"])
        self.assertEqual(15, t["new"])

    def test_a_malformed_row_is_skipped_with_a_note(self):
        self.put_json("mix.json", {"aimed-bugs": 30})
        self.put_json("jobs.json", JOBS)
        short = "\t".join(row(at(1), "aimed-bugs", 10, 0).split("\t")[:-1])  # one cell short
        self.put("yield.tsv", yield_text(
            row(at(1), "aimed-bugs", 40, 0), short,  # line 3: one cell short
            row(at(2), "aimed-bugs", 40, 0).replace("\t40.0\t", "\tx\t"),  # line 4: cpu_min is not a number
            row("yesterday", "aimed-bugs", 40, 0),  # line 5: utc is not a time
            row(at(3), "aimed-bugs", 40, "two"),  # line 6: fix_ready is not a count
            row(at(4), "aimed-bugs", 40, 0)))
        p = self.plan()
        t = self.by_name(p)["aimed-bugs"]
        self.assertEqual(2, t["jobs"])  # the two valid rows, 80 CPU min: judged
        self.assertEqual(15, t["new"])
        skipped = [n for n in p["notes"] if "skipped" in n]
        self.assertEqual(4, len(skipped), skipped)
        self.assertTrue(any("line 3 skipped" in n for n in skipped), skipped)
        self.assertTrue(any("line 4 skipped" in n for n in skipped), skipped)


class RefusalTest(ReweighCase):
    def test_apply_refuses_without_a_selfplay_share_and_writes_nothing(self):
        original = json.dumps({"aimed-bugs": 30}) + "\n"
        self.put("mix.json", original)
        self.put_json("jobs.json", {"selfplay": {"enabled": True}})
        self.put("yield.tsv", yield_text(*[row(at(d), "aimed-bugs", 50, 0) for d in (1, 2, 3)]))
        rc, _, err = self.cli("--apply")
        self.assertEqual(1, rc)
        self.assertIn("nothing written", err)
        self.assertEqual(original, self.text("mix.json"))
        self.assertFalse(any(p.name.endswith(".bak") for p in self.lab.iterdir()))

    def test_a_malformed_mix_file_stops_the_run_untouched(self):
        self.put("mix.json", "{not json")
        self.put_json("jobs.json", JOBS)
        rc, _, err = self.cli("--apply")
        self.assertEqual(1, rc)
        self.assertEqual("{not json", self.text("mix.json"))
        self.assertFalse((self.lab / "jobs.json.2026-10-24.bak").exists())


if __name__ == "__main__":
    unittest.main()
