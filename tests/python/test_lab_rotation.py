"""The fidelity-era job kinds and report steps are configuration switches, off unless LAB/jobs.json names them (D57 addendum 1, 2026-11-09):

  "rotation": ["fights", "probes", "mutation", "invariants", "runs"]   which of them the runner may pick; absent or empty: none
  "reports": {"checks": {"enabled": true, "every_h": 168}, "corpus-import": true, "groups": true, "kits": true}   which report steps run; absent: none

Tests on a fake config and trivial `python -c` commands; no game, no emulator.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.lab import jobs  # noqa: E402
from tools.lab import plans as pl  # noqa: E402
from tools.lab import runner as rn  # noqa: E402


class ParseTests(unittest.TestCase):
    def test_absent_empty_and_malformed_rotation_entries_are_none(self):
        for doc in (None, {}, {"rotation": []}, {"rotation": None}, {"rotation": "fights"}, {"rotation": {}}, {"rotation": {"kinds": []}}, {"cores": {}}):
            self.assertEqual(jobs.rotation_kinds(doc), ((), []), doc)

    def test_a_list_or_a_dict_names_the_kinds_in_the_rotations_own_order(self):
        self.assertEqual(jobs.rotation_kinds({"rotation": ["runs", "fights"]}), (("fights", "runs"), []))
        self.assertEqual(jobs.rotation_kinds({"rotation": {"kinds": ["probes"]}}), (("probes",), []))
        self.assertEqual(jobs.rotation_kinds({"rotation": ["fights", "probes", "mutation", "invariants", "runs"]})[0], jobs.FIDELITY_KINDS)

    def test_a_name_that_is_not_a_kind_is_reported_and_enables_nothing(self):
        kinds, unknown = jobs.rotation_kinds({"rotation": ["fights", "train", "presort", "fight"]})
        self.assertEqual((kinds, unknown), (("fights",), ["train", "presort", "fight"]))

    def test_reports_are_off_unless_enabled_and_take_a_bool_or_a_dict(self):
        off = jobs.reports_config({})
        self.assertEqual({k: v["enabled"] for k, v in off.items()}, {"checks": False, "corpus-import": False, "groups": False, "kits": False})
        got = jobs.reports_config({"reports": {"checks": {"enabled": True, "every_h": 168}, "groups": True, "kits": {"enabled": False}, "other": True,
                                               "corpus-import": "yes"}})
        self.assertEqual(got["checks"], {"enabled": True, "every_h": 168.0})
        self.assertEqual((got["groups"]["enabled"], got["kits"]["enabled"], got["corpus-import"]["enabled"]), (True, False, False))
        self.assertNotIn("other", got)
        self.assertIsNone(jobs.reports_config({"reports": {"checks": {"enabled": True, "every_h": -3}}})["checks"]["every_h"])
        self.assertEqual({k: v["enabled"] for k, v in jobs.reports_config({"reports": []}).items()}, {k: False for k in jobs.REPORT_STEPS})


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        saved = dict(pl.PLANS)
        self.addCleanup(lambda: (pl.PLANS.clear(), pl.PLANS.update(saved)))
        self.lab = self.tmp / "lab"
        self.lab.mkdir()
        for kind in ("fights", "probes", "mutation", "invariants", "runs", "presort"):
            pl.PLANS[kind] = lambda c: [pl.Step("a", [pl.Cmd("x", ["{py}", "-c", "pass"], cwd="{lab}", core=True)], mode="serial")]

    def config(self, **doc) -> None:
        (self.lab / "jobs.json").write_text(json.dumps(doc), encoding="utf-8")

    def runner(self, *extra: str) -> rn.Runner:
        a = rn.parse_args(["--lab", str(self.lab), "--tree", str(self.tmp / "tree"), "--game", str(self.tmp / "game" / "g"),
                           "--backlog-dir", str(self.tmp / "bl"), "--poll", "0.05", "--wait-poll", "0.05", "--grace", "0",
                           "--no-guard", "user,load,disk", "--loop-status", str(self.tmp / "no-loop"), *extra])
        r = rn.Runner(a)
        r.lab.runs.mkdir(parents=True, exist_ok=True)
        r.state["last_presorted_sha"] = r.state["tree_sha"] = "abc"
        return r

    def log(self, r: rn.Runner) -> str:
        p = r.lab.root / "runner.log"
        return p.read_text(encoding="utf-8") if p.exists() else ""


class RotationTests(Case):
    def picks(self, r: rn.Runner, n: int = 14) -> list[str]:
        out = []
        for _ in range(n):
            job, _ = r.pick_other("abc")
            out.append(job)
        return out

    def test_with_no_entry_nothing_of_the_rotation_is_picked(self):
        r = self.runner()
        self.assertEqual(r.pick_other("abc"), ("", "no job is runnable"))
        for job in jobs.FIDELITY_KINDS:
            self.assertIn("not in LAB/jobs.json rotation", r.unavailable(job, "abc") or "", job)
        self.config(rotation=[])
        self.assertEqual(self.runner().pick_other("abc")[0], "")

    def test_only_the_named_kinds_are_picked(self):
        self.config(rotation=["probes", "invariants"])
        r = self.runner()
        got = set(self.picks(r))
        self.assertEqual(got, {"probes", "invariants"})
        self.assertIsNone(r.unavailable("probes", "abc"))
        self.assertIn("not in LAB/jobs.json rotation", r.unavailable("fights", "abc", "dense"))
        self.assertIn("not in LAB/jobs.json rotation", r.unavailable("mutation", "abc"))

    def test_the_rotation_tuple_is_filtered_by_the_entry_not_reordered(self):
        self.config(rotation=["fights", "probes", "mutation", "invariants", "runs"])
        r = self.runner()
        r.state["rotation"] = 0
        first = [r.pick_other("abc")[0] for _ in range(len(rn.ROTATION))]
        self.assertEqual(first.count("fights"), 4)  # ROTATION's four fights slots (the mix may hand one to whole-runs: a "runs" job)
        self.config(rotation=["probes"])
        r.state["rotation"] = 0
        self.assertEqual([r.pick_other("abc")[0] for _ in range(3)], ["probes"] * 3)

    def test_the_fights_slot_goes_on_when_only_whole_runs_is_named(self):
        self.config(rotation=["runs"])
        r = self.runner()
        self.assertIsNone(r.unavailable("runs", "abc", "whole-runs"))
        self.assertIn("not in LAB/jobs.json rotation", r.unavailable("fights", "abc", "dense"))
        job, why = r.pick_other("abc")
        self.assertEqual(job, "runs", why)  # the mix is left with whole-runs

    def test_a_kind_that_is_off_leaves_no_line_at_every_pick_and_the_settings_are_logged_once(self):
        self.config(rotation=["probes"], reports={"groups": True})
        r = self.runner()
        for _ in range(5):
            r.pick_other("abc")
        text = self.log(r)
        self.assertEqual(text.count("LAB/jobs.json rotation: probes; reports: groups"), 1)
        self.assertNotIn("fights: not runnable", text)
        self.assertNotIn("mutation: not runnable", text)
        self.config(rotation=["probes", "mutation"], reports={})
        r.pick_other("abc")
        self.assertIn("rotation: probes, mutation; reports: none", self.log(r))
        self.config(rotation=["probes", "train"])
        r.pick_other("abc")
        self.assertIn("not kinds: train", self.log(r))

    def test_the_command_line_overrides_and_a_once_run_names_its_own_kind(self):
        r = self.runner("--rotation", "all")
        self.assertEqual(set(self.picks(r)) - {"runs"}, {"fights", "probes", "mutation", "invariants"})
        r = self.runner("--once", "mutation")  # asked for by hand: the entry is not read
        self.assertIsNone(r.unavailable("mutation", "abc"))
        self.assertEqual(r.choose("abc")[0], "mutation")

    def test_the_config_is_read_at_every_pick(self):
        r = self.runner()
        self.assertEqual(r.pick_other("abc")[0], "")
        self.config(rotation=["invariants"])
        self.assertEqual(r.pick_other("abc")[0], "invariants")
        self.config(rotation=[])
        self.assertEqual(r.pick_other("abc")[0], "")

    def test_an_unfinished_job_of_a_kind_that_was_switched_off_is_closed_not_resumed(self):
        self.config(rotation=["probes"])
        r = self.runner()
        old = r.new_run("fights", "abc")
        job, run, why = r.choose("abc")
        self.assertEqual((job, run), ("probes", None), why)
        self.assertEqual((old.dir / "done.rc").read_text().strip(), "3")
        self.assertIn("is not in LAB/jobs.json rotation", json.loads((old.dir / "job.json").read_text())["note"])
        kept = r.new_run("probes", "abc")
        job, run, _ = r.choose("abc")
        self.assertEqual((job, run.name if run else None), ("probes", kept.name))  # a kind that is on is resumed as ever

    def test_the_agent_era_kinds_and_the_presort_are_not_in_the_entry(self):
        self.config(rotation=[])
        r = self.runner()
        self.assertEqual(r.choose("zzz")[0], "presort")  # main moved: the presort is chosen whatever the entry says
        self.assertIsNone(r.rotation_hold("presort"))
        self.assertIsNone(r.rotation_hold("train"))


class ReportsTests(Case):
    def test_every_report_step_is_off_without_an_entry_and_the_command_line_turns_them_on(self):
        self.assertEqual(self.runner().reports_off(), ("checks", "corpus-import", "groups", "kits"))
        self.assertEqual(self.runner("--reports", "all").reports_off(), ())
        self.config(reports={"groups": True, "kits": {"enabled": True}})
        self.assertEqual(self.runner().reports_off(), ("checks", "corpus-import"))

    def test_a_step_that_is_off_has_its_commands_skipped_with_the_reason(self):
        self.config(reports={"groups": True})
        r = self.runner()
        ctx = r.ctx(None, "presort", "abc", {"procs_refset": 4, "procs_corpus": 8})
        steps = {s.name: s for s in pl.plan_presort(ctx)}
        for cmd in steps["checks"].cmds:
            self.assertEqual(cmd.skip_reason(ctx, "checks"), "off in LAB/jobs.json reports.checks", cmd.name)  # codex-s0..3 and the reference set
        self.assertNotIn("reports.", steps["corpus"].cmds[0].skip_reason(ctx, "corpus") or "")  # the corpus check is not one of the four
        groups = next(c for c in pl.report_cmds(ctx) if c.name == "groups")
        kits = next(c for c in pl.report_cmds(ctx) if c.name == "kits")
        self.assertNotIn("reports.", groups.skip_reason(ctx, "groups") or "")
        self.assertEqual(kits.skip_reason(ctx, "kits"), "off in LAB/jobs.json reports.kits")
        imp = pl.import_step().cmds[0]
        self.assertEqual(imp.skip_reason(ctx, "corpus-import"), "off in LAB/jobs.json reports.corpus-import")

    def test_an_off_checks_step_is_skipped_by_the_runner_and_the_job_still_finishes(self):
        pl.PLANS["presort"] = lambda c: [pl.Step("tree", [pl.Cmd("t", ["{py}", "-c", "pass"], cwd="{lab}", core=True)], mode="serial", required=True),
                                         pl.Step("checks", [pl.Cmd("codex-s0", ["{py}", "-c", "open('ran-codex', 'w')"], cwd="{lab}", core=True)])]
        self.config(reports={})
        r = self.runner()
        run = r.new_run("presort", "newsha")
        self.assertIsNone(r.execute(run))
        self.assertEqual((run.dir / "done.rc").read_text().strip(), "0")
        self.assertFalse((r.lab.root / "ran-codex").exists())
        self.assertEqual(run.rec("checks")["status"], "skipped")
        self.assertNotIn("last_checks", r.state)
        self.config(reports={"checks": True})
        run = r.new_run("presort", "newsha2")
        self.assertIsNone(r.execute(run))
        self.assertTrue((r.lab.root / "ran-codex").exists())
        self.assertAlmostEqual(r.state["last_checks"], time.time(), delta=60)

    def test_checks_run_at_most_once_in_every_h(self):
        self.config(reports={"checks": {"enabled": True, "every_h": 168}})
        r = self.runner()
        self.assertNotIn("checks", r.reports_off())  # never run: due
        r.state["last_checks"] = time.time() - 3 * 24 * 3600
        self.assertIn("checks", r.reports_off())  # three days ago: not yet weekly
        r.state["last_checks"] = time.time() - 8 * 24 * 3600
        self.assertNotIn("checks", r.reports_off())  # eight days ago: due
        self.config(reports={"checks": True})
        r.state["last_checks"] = time.time() - 60
        self.assertNotIn("checks", r.reports_off())  # without every_h, every presort
        self.config(reports={"checks": {"enabled": False, "every_h": 168}})
        r.state["last_checks"] = 0
        self.assertIn("checks", r.reports_off())  # off is off

    def test_groups_and_kits_both_off_queue_no_report_and_the_yield_row_is_written_at_once(self):
        r = self.runner("--report-beside", "on")
        run = r.new_run("fights", "abc")
        self.assertIsNone(r.execute(run))
        self.assertEqual(r.report_due, [])
        self.assertIsNone(r.report)
        rows = (r.lab.root / "yield.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 2)  # header and this job's row
        self.assertIn(run.name, rows[1])

    def test_groups_or_kits_on_queue_the_report_as_before(self):
        pl.report_cmds, real = (lambda c: [pl.Cmd("groups", ["{py}", "-c", "pass"], cwd="{lab}"), pl.Cmd("kits", ["{py}", "-c", "pass"], cwd="{lab}")]), pl.report_cmds
        self.addCleanup(lambda: setattr(pl, "report_cmds", real))
        self.config(reports={"kits": True})
        r = self.runner("--report-beside", "on")
        run = r.new_run("fights", "abc")
        self.assertIsNone(r.execute(run))
        self.assertTrue(r.report is not None or r.report_due, "the report was not queued")
        end = time.time() + 20
        while (r.report is not None or r.report_due) and time.time() < end:
            r.service_report()
            time.sleep(0.05)
        self.assertIn("groups/groups: skipped (off in LAB/jobs.json reports.groups)", self.log(r))
        self.assertNotIn("kits/kits: skipped", self.log(r))


if __name__ == "__main__":
    unittest.main()
