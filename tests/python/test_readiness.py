"""tools/fidelity/readiness.py on a fixture referee tree."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import readiness as rd  # noqa: E402

SHA_A = "a" * 40
SHA_B = "b" * 40


def w(p: Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")


def hyp(hid: str, **kw) -> dict:
    return {"id": hid, "claim": "c", "spec": {"setup": [{"cmd": "fight", "id": "X"}]}, "compare": {"fields": [".hand"]}, **kw}


def fixture(root: Path) -> None:
    per = {"1": ["complete", 30, ""], "2": ["mismatch", 5, "EMULATOR: deck"], "3": ["mismatch", 20, "EMULATOR: deck"],
           "4": ["mismatch", 40, "WALKER: flow"], "5": "junk"}
    cx = {"runs": 4, "complete": 1, "judged": 4, "judged_matching_pct": 25.0, "components_at_zero": ["rest"],
          "runs_in_set": 5, "gap": 1,
          "stops_by_class": {"EMULATOR": 2}, "stops": {"EMULATOR: deck": 2, "WALKER: flow": 1}, "per_run": per}
    w(root / "refset/v2/scores/all-old.json",
      {"scored": "2026-10-01 10:00", "label": "old", "codex": {"runs": 4, "complete": 4, "stops": {}}})
    w(root / "refset/v2/scores/all-new.json",
      {"scored": "2026-10-20 10:00", "label": "new", "emulator": {"commit": SHA_A, "dirty": False}, "codex": cx})
    w(root / "refset/v2/scores/all-new2.json",
      {"scored": "2026-10-20 09:00", "label": "earlier-same-day", "codex": {"runs": 4, "complete": 0, "stops": {}}})
    w(root / "refset/v2/scores/all-skip.json", {"scored": "2026-10-21 10:00", "codex": {"skipped": "x"}})
    w(root / "corpus/baselines/a.json", {"made": "2026-10-19T01:00:00", "commit": SHA_A,
                                         "totals": {"fights": 10, "identical": 7, "reanchored": 20, "reanchor_matched": 15}})
    w(root / "lab/groups.json", {"generated_utc": "2026-10-20T00:00:00Z", "main_sha": SHA_B, "inputs": {"profile_runs": 50}, "groups": [
        {"id": "Ga", "title": "a", "impact_runs_est": 5000, "host_suspect": True, "backlog_ids": ["C1"]},
        {"id": "Gb", "title": "b", "impact_runs_est": 900, "count": 3, "backlog_ids": ["C2"], "relation": "merges"},
        {"id": "Gc", "title": "c", "impact_runs_est": 1},
        {"id": "Gd", "title": "d", "impact_runs_est": 900, "count": 7},
        {"id": "Ge", "title": "e", "impact_runs_est": 4000, "stale": True},
        {"id": "Gf", "title": "f", "impact_runs_est": 0},
        {"id": "Gg", "title": "g", "impact_runs_est": 1000}]})
    w(root / "audit/data-checks/2026-10-18-t0.txt", "TOTAL: 99 difference(s)\n")
    w(root / "audit/data-checks/2026-10-20-t1.txt",
      "cards: 0 difference(s)\nneow: 1 difference(s)\n  [neow] x\npowers: 45 difference(s)\nTOTAL: 46 difference(s)\n")
    w(root / "hypotheses/a/a.json", [hyp("h1"), {"id": "h2", "claim": "c", "testable": False, "why": "needs a screen"}, hyp("h3"), hyp("h4")])
    w(root / "hypotheses/a/bad.json", [{"id": "bad id", "claim": "c"}])
    w(root / "lab/hypotheses.json", {"h1": {"status": "confirmed"}, "h3": {"status": "refuted"}})


class BenchPreservedTest(unittest.TestCase):
    def test_regenerating_the_json_keeps_a_bench_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "referee"
            fixture(root)
            out = Path(tmp) / "readiness.json"
            with contextlib.redirect_stdout(io.StringIO()):
                rd.main(["--referee", str(root), "--json", str(out)])
                self.assertNotIn("bench", json.loads(out.read_text()))
                doc = json.loads(out.read_text())
                doc["bench"] = {"emu": {"date": "2026-10-22"}}
                out.write_text(json.dumps(doc, indent=1))
                rd.main(["--referee", str(root), "--json", str(out)])
            self.assertEqual(json.loads(out.read_text())["bench"], {"emu": {"date": "2026-10-22"}})


class NightlyPreservedTest(unittest.TestCase):
    def test_regenerating_the_json_keeps_the_nightly_key_and_prints_its_numbers(self):
        nightly = {"date": "2026-10-25", "agents": {"g3": {"runs": 20, "win_rate": 0.0, "mean_floor": 12.5, "boss_kills": 1,
                                                          "divergence_rate": 0.003, "crashes": 1}}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "referee"
            fixture(root)
            out = Path(tmp) / "readiness.json"
            with contextlib.redirect_stdout(io.StringIO()):
                rd.main(["--referee", str(root), "--json", str(out)])
            doc = json.loads(out.read_text())
            self.assertNotIn("nightly", doc)
            out.write_text(json.dumps({**doc, "nightly": nightly}, indent=1))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rd.main(["--referee", str(root), "--json", str(out)])
            self.assertEqual(nightly, json.loads(out.read_text())["nightly"])
            r = json.loads(out.read_text())
            self.assertIn("g3: 20 runs", "\n".join(rd.render(r).splitlines()))


class T(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = Path(self.d.name)
        fixture(self.root)
        self._min = rd.MIN_AT_RISK
        rd.MIN_AT_RISK = 1

    def tearDown(self):
        rd.MIN_AT_RISK = self._min
        self.d.cleanup()

    def test_headline_and_codex(self):
        r = rd.build(self.root)
        self.assertEqual(r["headline_pct"], 25.0)
        c = r["codex"]
        self.assertEqual(c["label"], "new")
        self.assertEqual(c["commit"], SHA_A)
        self.assertEqual((c["runs_in_set"], c["not_walked"], c["identical_pct_of_set"]), (5, 1, 20.0))
        self.assertEqual(c["first_divergence_floor_median"], 20)
        self.assertEqual(c["stopped_by_act"], {"1": 1, "2": 1, "3": 1})
        self.assertEqual(c["complete_run_floors_median"], 30)
        self.assertEqual(c["complete_runs_by_act"], {"1": 0, "2": 1, "3": 0})
        self.assertEqual(c["stopped"], 3)

    def test_history_is_newest_score_of_each_day(self):
        h = rd.build(self.root)["codex"]["history"]
        self.assertEqual([(x["date"], x["identical"]) for x in h], [("2026-10-01", 4), ("2026-10-20", 1)])

    def test_act_floors_are_rooms_plus_two(self):
        # Overgrowth/Underdocks 15 rooms, Hive 14, Glory 13, each plus the ancient floor and the boss floor
        self.assertEqual(rd.ACT_FLOORS, (17, 16, 15))
        self.assertEqual(rd.ACT_ENDS, (17, 33, 48))
        self.assertEqual([rd.act_of(f) for f in (1, 17, 18, 33, 34, 48, 49)], [1, 1, 2, 2, 3, 3, 3])

    def test_stopped_run_diverges_at_the_floor_it_stopped_on(self):
        # floors_walked of a stopped run is the floor it stopped on (the walker counts a floor before walking it): a stop at
        # floor 1 means floor 1 did not match, so the share identical through floor 1 is below 100%
        runs = [(False, 1)] + [(True, 10)] * 3
        self.assertEqual(rd.survival_curve(runs)[1], (75.0, 4))
        self.assertEqual(rd.survival_curve(runs)[10], (75.0, 3))
        self.assertEqual(rd.survival_curve([(False, 0), (True, 3)])[0], (50.0, 2))
        # every run stopped on floor 5 (as ancient-check stops do on floor 1): nothing is identical through that floor
        self.assertEqual(rd.survival_curve([(False, 5)] * 4)[5], (0.0, 4))
        self.assertEqual(rd.survival_curve([(False, 5)] * 4)[4], (100.0, 4))

    def test_survival_curve(self):
        # stops on floors 5, 20 and 40 diverge there; the complete 30-floor run leaves after floor 30
        runs = [(True, 30), (False, 5), (False, 20), (False, 40)]
        self.assertEqual(rd.survival(runs, 1), {17: [75.0, 3], 33: [50.0, 1], 40: [0.0, 1]})
        self.assertEqual(rd.survival(runs, 2), {17: [75.0, 3], 30: [50.0, 2]})
        self.assertEqual(rd.survival(runs, 100), {})
        self.assertEqual(rd.survival([], 1), {})
        self.assertEqual(rd.survival_curve(runs)[4], (100.0, 4))
        self.assertEqual(rd.survival_curve(runs)[5], (75.0, 4))
        self.assertEqual(rd.survival_curve(runs)[20], (50.0, 3))
        self.assertEqual(rd.survival_curve(runs)[21], (50.0, 2))

    def test_survival_does_not_count_runs_that_ended_against_the_emulator(self):
        # 100 runs that end identically after 10 floors, nothing stops: every floor stays at 100%
        self.assertEqual(rd.survival([(True, 10)] * 100, 1), {10: [100.0, 100]})

    def test_full_run_estimate_in_build(self):
        e = rd.build(self.root)["codex"]["full_run_estimate"]
        self.assertEqual([(x["floor"], x["identical_pct"], x["runs_followed"]) for x in e],
                         [(17, 75.0, 3), (33, 50.0, 1), (40, 0.0, 1)])

    def test_full_run_rows_are_labelled_with_the_act_end(self):
        runs = {str(i): ["complete", 48, ""] for i in range(120)} | {str(200 + i): ["mismatch", 33, "EMULATOR: x"] for i in range(60)}
        p = self.root / "refset/v2/scores/all-new.json"
        d = json.loads(p.read_text())
        d["codex"].update(per_run=runs, runs=180, complete=120)
        w(p, d)
        text = rd.render(rd.build(self.root))
        self.assertIn("full run to floor 17 (end of act 1)", text)
        self.assertIn("full run to floor 33 (end of act 2)", text)
        self.assertIn("full run to floor 48 (end of act 3)", text)
        self.assertNotIn("floor 34", text)
        sa = rd.build(self.root)["codex"]["stopped_by_act"]
        self.assertEqual(sa, {"1": 0, "2": 60, "3": 0})

    def test_trend_shows_shares_not_only_counts(self):
        text = rd.render(rd.build(self.root))
        self.assertIn("100.0% = 4/4 (10-01) -> 25.0% = 1/4 (10-20)", text)

    def test_subscores(self):
        r = rd.build(self.root)
        self.assertEqual(r["corpus"]["identical_pct"], 70.0)
        self.assertEqual(r["corpus"]["steps_verified_pct"], 75.0)
        self.assertEqual((r["data"]["mismatches"], r["data"]["date"]), (46, "2026-10-20"))
        self.assertEqual(r["data"]["by_check"], {"neow": 1, "powers": 45})

    def test_lab_open_groups_leave_out_host_suspect_and_stale(self):
        b = rd.build(self.root)["lab"]
        self.assertEqual((b["groups"], b["open_groups"], b["host_suspect"], b["stale"]), (7, 5, 1, 1))
        self.assertEqual(b["open_by_impact_runs"], {">=1000": 1, "100-999": 2, "1-99": 1, "0": 1})
        self.assertEqual(b["open_with_backlog_id"], 1)
        self.assertEqual(b["open_runs_impact_max"], 1000)

    def test_hypotheses(self):
        h = rd.build(self.root)["hypotheses"]
        self.assertEqual((h["confirmed"], h["refuted"], h["not_testable"], h["open"], h["total"]), (1, 1, 1, 1, 4))
        self.assertEqual(h["bad_files"], 1)
        self.assertEqual(h["source"], "lab/hypotheses.json")

    def test_hypotheses_without_verdicts_are_open(self):
        (self.root / "lab/hypotheses.json").unlink()
        r = rd.build(self.root)
        self.assertEqual((r["hypotheses"]["confirmed"], r["hypotheses"]["open"], r["hypotheses"]["not_testable"]), (0, 3, 1))
        self.assertIn("no verdicts file", r["hypotheses"]["source"])

    def test_idle_mix_note_follows_the_lab_modules_reading_of_mix_json(self):
        (self.root / "lab/hypotheses.json").unlink()
        mix = self.root / "lab/mix.json"

        def idle() -> bool:
            return "hypotheses_idle" in rd.build(self.root)["notes"]

        # no mix.json: the lab runs its default mix, where hypotheses weighs 5; nothing is created by reading
        self.assertFalse(idle())
        self.assertEqual(rd.build(self.root)["hypotheses"]["lab_mix_weight"], 5.0)
        self.assertFalse(mix.exists())
        # a mix that omits hypotheses (or weighs it 0) leaves the setups unrun
        w(mix, {"aimed-bugs": 5})
        self.assertTrue(idle())
        w(mix, {"aimed-bugs": 3, "hypotheses": 0})
        self.assertTrue(idle())
        w(mix, {"_note": "x", "hypotheses": 5})
        self.assertFalse(idle())
        # a mix the lab cannot use falls back to its default, which runs hypotheses: no idle note, a note on the mix instead
        for bad in ("{not json", "[]", {"hypotheses": 0}, {"_note": "only notes"}, {"hypotheses": "5", "nope": 1}):
            w(mix, bad)
            r = rd.build(self.root)
            self.assertFalse(idle(), bad)
            self.assertEqual(r["hypotheses"]["lab_mix_weight"], 5.0, bad)
            self.assertIn("default mix", r["notes"]["lab_mix"], bad)
        # a verdict means a setup was run
        w(mix, {"aimed-bugs": 5})
        w(self.root / "lab/hypotheses.json", {"h1": {"status": "confirmed"}})
        self.assertFalse(idle())

    def test_blockers(self):
        r = rd.build(self.root, top=2)
        self.assertEqual(r["blockers"], [{"cause": "EMULATOR: deck", "class": "EMULATOR", "runs": 2, "pct_of_runs": 50.0},
                                         {"cause": "WALKER: flow", "class": "WALKER", "runs": 1, "pct_of_runs": 25.0}])
        # by exposure, then by divergences seen; the host-suspect Ga and the stale Ge are not blockers
        self.assertEqual([g["id"] for g in r["lab_blockers"]], ["Gg", "Gd"])
        self.assertEqual(r["lab_blockers"][1]["divergences"], 7)
        self.assertEqual([g["id"] for g in rd.build(self.root, top=9)["lab_blockers"]], ["Gg", "Gd", "Gb", "Gc", "Gf"])

    def test_notes_flag_mixed_commits_and_dirty_trees(self):
        r = rd.build(self.root)
        self.assertIn("commits_differ", r["notes"])
        self.assertIn("aaaaaaaa", r["notes"]["commits_differ"])
        p = self.root / "refset/v2/scores/all-new.json"
        d = json.loads(p.read_text())
        d["emulator"]["dirty"] = True
        d["not_quotable"] = ["no mirror"]
        w(p, d)
        n = rd.build(self.root)["notes"]
        self.assertIn("codex_dirty", n)
        self.assertEqual(n["codex_not_quotable"], "no mirror")

    def test_missing_inputs(self):
        with tempfile.TemporaryDirectory() as e:
            r = rd.build(Path(e))
            self.assertIsNone(r["headline_pct"])
            self.assertEqual(sorted(r["notes"]), ["codex", "corpus", "data", "hypotheses", "lab"])
            self.assertEqual((r["blockers"], r["lab_blockers"]), ([], []))
            self.assertIn("n/a", rd.render(r))

    def test_unreadable_files_are_skipped(self):
        w(self.root / "refset/v2/scores/z.json", "{not json")
        w(self.root / "lab/groups.json", "[]")
        r = rd.build(self.root)
        self.assertEqual(r["codex"]["label"], "new")
        self.assertIsNone(r["lab"])
        self.assertIn("lab", r["notes"])

    def test_render_names_each_source_and_date(self):
        text = rd.render(rd.build(self.root))
        self.assertTrue(text.startswith("READINESS (Ironclad A10): 25.0% of Codex runs replayed identically to the end (1/4)"))
        for src in ("refset/v2/scores/all-new.json", "corpus/baselines/a.json", "lab/groups.json",
                    "audit/data-checks/2026-10-20-t1.txt", "lab/hypotheses.json"):
            self.assertIn(f"[{src}, ", text)
        self.assertIn("Top blockers by run impact", text)
        self.assertIn("1. EMULATOR: deck  2 runs (50.0%)", text)
        self.assertIn("1 of 7 groups", text)

    def test_output_holds_no_absolute_path(self):
        r = rd.build(self.root)
        blob = json.dumps(r) + rd.render(r)
        self.assertNotIn(str(self.root), blob)
        self.assertNotIn(self.root.as_posix(), blob)

    def test_cli_writes_json(self):
        out = self.root / "o" / "r.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rd.main(["--referee", str(self.root), "--json", str(out)]), 0)
        doc = json.loads(out.read_text())
        self.assertEqual(doc["headline_pct"], 25.0)
        self.assertEqual(set(doc), {"codex", "corpus", "data", "hypotheses", "lab", "headline_pct", "blockers", "lab_blockers", "notes"})


class UtilisationTest(unittest.TestCase):
    """lab.utilisation: CPU minutes over the wall minutes the owner left, from yield.tsv, runner.log and state.json."""

    NOW = __import__("datetime").datetime(2026, 10, 24, tzinfo=__import__("datetime").timezone.utc).timestamp()
    LOG = "\n".join([
        "2026-10-10T00:00:00Z paused: user (input 0s ago)",         # before the window: ignored
        "2026-10-10T05:00:00Z resuming (was held by: user)",
        "2026-10-20T10:00:00Z paused: user (input 0s ago)",         # 60 min of owner use
        "2026-10-20T11:00:00Z resuming (was held by: user)",
        "2026-10-21T00:00:00Z paused: pause (LAB/PAUSE)",           # 120 min held by another session
        "2026-10-21T02:00:00Z resuming (was held by: pause)",
        "2026-10-22T00:00:00Z paused: load (other processes 90% of the CPU over 120s)",  # 30 min: reported, not subtracted
        "2026-10-22T00:30:00Z resuming (was held by: load)",
        "2026-10-23T22:00:00Z paused: pause (LAB/PAUSE)",           # still open at the end: 120 min to NOW
        "2026-10-23T22:00:05Z runner started (trial=None)",         # a restart closes the span ...
        "2026-10-23T22:00:05Z paused: pause (LAB/PAUSE)",           # ... and the new process reports the same pause: no gap, no double count
    ]) + "\n"
    YIELD = "\t".join(["utc", "job", "strategy", "main", "rc", "wall_min", "cpu_min"]) + "\n" + "".join("\t".join(r) + "\n" for r in [
        ("2026-10-20T09:00:00Z", "j1", "aimed-bugs", "abc", "0", "100.0", "4000.0"),
        ("2026-10-22T09:00:00Z", "j2", "aimed-bugs", "abc", "0", "50.0", "2000.0"),
        ("2026-10-23T09:00:00Z", "j3", "random", "abc", "0", "80.0", "1824.0"),
        ("2026-10-23T10:00:00Z", "j4", "invariants", "abc", "3", "0.0", "NA"),
        ("2026-10-10T09:00:00Z", "old", "random", "abc", "0", "999.0", "99999.0"),  # outside the 7 days
    ])

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        lab = self.tmp / "lab"
        w(lab / "groups.json", {"groups": [], "main_sha": SHA_A})
        w(lab / "yield.tsv", self.YIELD)
        w(lab / "runner.log", self.LOG)
        w(lab / "state.json", {"counters": {"active_seconds": 60000, "interrupts": {"user": 2}}, "strategy_cpu": {"aimed-bugs": 360000, "random": 109440}})
        w(lab / "mix.json", {"aimed-bugs": 10, "random": 5, "hypotheses": 5})

    def test_cpu_minutes_over_the_wall_minutes_the_owner_left(self):
        u = rd.build(self.tmp, now=self.NOW, cores=4)["lab"]["utilisation"]
        self.assertEqual(u["cpu_min"], 7824.0)  # the old row is outside the window; the NA row counts 0
        self.assertEqual(u["wall_min"], 10080.0)
        self.assertEqual(u["owner_min"], 300.0)  # 60 user + 120 pause + 120 pause open to the end; load is not owner time
        self.assertEqual(u["held_min"], {"load": 30.0, "pause": 240.0, "user": 60.0})
        self.assertEqual(u["available_wall_min"], 9780.0)
        self.assertEqual(u["pct"], 20.0)  # 7824 / (9780 * 4 cores)
        self.assertEqual((u["target_pct"], u["meets_target"], u["cores"]), (90.0, False, 4))
        self.assertEqual(u["by_strategy"]["aimed-bugs"], {"jobs": 2, "cpu_min": 6000.0, "wall_min": 150.0, "share_pct": 76.7, "pct_of_available": 15.34})
        self.assertEqual(list(u["by_strategy"]), ["aimed-bugs", "random", "invariants"])
        self.assertEqual(u["idle_with_weight"], ["hypotheses"])  # weight in mix.json, no CPU in the window
        self.assertEqual(u["all_time"], {"cpu_min": 7824.0, "active_min": 1000.0, "interrupts": {"user": 2}})

    def test_days_narrows_the_window_to_the_last_two_days(self):
        u = rd.build(self.tmp, now=self.NOW, cores=4, days=2)["lab"]["utilisation"]
        self.assertEqual((u["window_days"], u["wall_min"], u["from_utc"]), (2, 2880.0, "2026-10-22T00:00Z"))
        self.assertEqual(u["cpu_min"], 3824.0)  # j2 and j3; the 7-day jobs before the window are out
        self.assertEqual(u["owner_min"], 120.0)  # only the open pause counts; the earlier user and pause spans are outside
        self.assertEqual(u["held_min"], {"load": 30.0, "pause": 120.0, "user": 0.0})
        self.assertEqual(u["pct"], 34.6)  # 3824 / (2760 * 4 cores)
        self.assertEqual(rd.UTIL_DAYS, 7)  # the default window is unchanged

    def test_main_days_option_reaches_the_window(self):
        out = self.tmp / "out.json"
        with contextlib.redirect_stdout(io.StringIO()):
            rd.main(["--referee", str(self.tmp), "--cores", "4", "--days", "3", "--json", str(out)])
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["lab"]["utilisation"]["window_days"], 3)

    def test_no_yield_gives_a_null_and_a_note(self):
        (self.tmp / "lab" / "yield.tsv").unlink()
        r = rd.build(self.tmp, now=self.NOW, cores=4)
        self.assertIsNone(r["lab"]["utilisation"])
        self.assertIn("yield.tsv", r["notes"]["lab_utilisation"])

    def test_render_has_the_utilisation_row(self):
        text = rd.render(rd.build(self.tmp, now=self.NOW, cores=4))
        self.assertIn("lab utilisation", text)
        self.assertIn("20.0% of 4 cores", text)


if __name__ == "__main__":
    unittest.main()
