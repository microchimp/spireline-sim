"""D65 addendum 2 (2026-11-17, owner: "Lift the cap"): the two-attempts-per-generation cap is recorded, not gating.

The attempt number and the old cap are still computed and written (the decision's `checks.attempt` keeps the value the cap gave, the
verdict's attempt block says whether the old cap would have refused), and the decision says `attempt_advisory`; no path refuses a candidate,
a training round, a marker or a block for being the third attempt of a generation or later. Every other check decides as before.

Constructed data only, the arms stubbed: no game, no emulator.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, ROOT / "tests" / "python"):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import gen0  # noqa: E402
import gen_loop  # noqa: E402
import keep_v2_fixtures as fx  # noqa: E402
import test_funnel as tf  # noqa: E402
import test_gen12_plumbing as tg12  # noqa: E402
import test_promotion_guard as tpg  # noqa: E402
import test_vrun_swap as tvs  # noqa: E402
from tools.lab import keep_v2 as k2  # noqa: E402
from tools.lab import promotion_guard as pg  # noqa: E402


def judged(agents: Path, cand_gen: int, *shas: str) -> None:
    """Candidates of `cand_gen` already judged and rejected (rows of the ledger), each on a block of its own."""
    for sha in shas:
        gen_loop.append_promotion(agents / "promotions.tsv", {"cand_gen": cand_gen, "decision": "rejected", "seed_block": f"E-{sha}", "cand_sha": sha})


class RecordTests(unittest.TestCase):
    """`promotion_guard.retry_cap`, `judge` and the helpers every aggregate of checks goes through."""

    def test_the_cap_is_still_computed_and_recorded_and_refuses_nothing(self):
        for prior, within in ((0, True), (1, True), (2, False), (5, False)):
            r = pg.retry_cap(prior)
            self.assertEqual((r["attempt"], r["cap"]), (prior + 1, 2))
            self.assertEqual((r["within_cap"], r["would_refuse"]), (within, not within))  # what the old cap said
            self.assertEqual((r["allowed"], r["blocks"], r["advisory"]), (True, False, True))  # what it does now: nothing
        self.assertEqual(pg.ATTEMPT_CAP, 2)

    def test_the_helpers_leave_the_advisory_check_out_of_the_verdict_and_nothing_else(self):
        self.assertEqual(pg.ADVISORY_CHECKS, ("attempt",))
        self.assertEqual(gen_loop.ADVISORY_CHECKS, pg.ADVISORY_CHECKS)  # the loop's own copy: the flag-off path imports nothing
        for mod in (pg, gen_loop):
            self.assertTrue(mod.checks_hold({"floor": True, "boss_kill": True, "attempt": False}))
            self.assertFalse(mod.checks_hold({"floor": True, "boss_kill": False, "attempt": True}))
            self.assertFalse(mod.checks_hold({"floor": True, "sequential": False, "attempt": False}))
            self.assertEqual(mod.failed_checks({"floor": False, "attempt": False, "drift": True, "wins": False}), ["floor", "wins"])
            self.assertEqual(mod.failed_checks({"floor": True, "attempt": False}), [])

    def test_a_clean_block_is_not_blocked_by_the_attempt_and_the_verdict_names_it_advisory(self):
        seeds = tpg.seeds(400)
        cand = {s: i < 30 for i, s in enumerate(seeds)}  # the candidate alone kills 30; the incumbent alone 20
        base = {s: 30 <= i < 50 for i, s in enumerate(seeds)}
        third = pg.judge(cand, base, prior_attempts=2, boot=500)
        self.assertEqual((third["blocks"], third["blocking"]), (False, []))
        self.assertEqual((third["attempt"]["attempt"], third["attempt"]["would_refuse"]), (3, True))
        second = pg.judge(cand, base, prior_attempts=1, boot=500)
        self.assertEqual((second["attempt"]["within_cap"], second["attempt"]["would_refuse"]), (True, False))
        low = {s: i < 8 for i, s in enumerate(seeds)}  # 8 candidate-only against 20 incumbent-only kills: the kill guard fires at 0.05
        high = {s: 8 <= i < 28 for i, s in enumerate(seeds)}
        worse = pg.judge(low, high, prior_attempts=2, boot=500)  # blocked for the kill guard, not for the attempt
        self.assertEqual(worse["blocking"], ["kill"])

    def test_the_summary_line_says_the_old_cap_would_have_refused_only_when_it_would(self):
        seeds = tpg.seeds(400)
        cand = {s: i < 30 for i, s in enumerate(seeds)}
        base = {s: 30 <= i < 50 for i, s in enumerate(seeds)}
        self.assertTrue(pg.summary(pg.judge(cand, base, prior_attempts=1, boot=500)).endswith("attempt 2/2"))
        line = pg.summary(pg.judge(cand, base, prior_attempts=2, boot=500))
        self.assertIn("attempt 3/2 (over the old cap: recorded, gates nothing)", line)
        self.assertNotIn("\t", line)


class KeepV2Tests(unittest.TestCase):
    def test_a_third_candidate_that_passes_every_other_check_is_promoted(self):
        _, cand, base = fx.arms()
        third = k2.judge(cand, base, frontier=1, prior_attempts=2, boot=1000)
        self.assertTrue(third["route"], third["routes"])
        self.assertTrue(third["promote"])
        self.assertEqual((third["blocks"], third["blocking"]), (False, []))
        self.assertFalse(third["checks"]["attempt"])  # the check keeps the value the cap gave ...
        self.assertTrue(third["attempt_advisory"])  # ... and says it decides nothing
        self.assertEqual((third["attempt"]["attempt"], third["attempt"]["cap"], third["attempt"]["would_refuse"]), (3, 2, True))
        self.assertIn("attempt 3/2 (over the old cap: recorded, gates nothing)", k2.summary(third))
        second = k2.judge(cand, base, frontier=1, prior_attempts=1, boot=1000)
        self.assertTrue(second["checks"]["attempt"])
        self.assertTrue(second["attempt_advisory"])
        self.assertIn("attempt 2/2", k2.summary(second))
        self.assertNotIn("old cap", k2.summary(second))

    def test_the_other_checks_still_block_a_third_candidate(self):
        _, cand, base = fx.arms()
        lineage = [(1100, 2000), (1130, 2000), (1120, 2000)]
        blocked = k2.judge(cand, base, frontier=1, lineage=lineage, previous_flagged=True, prior_attempts=2, boot=1000)
        self.assertFalse(blocked["promote"])
        self.assertEqual(blocked["blocking"], ["drift"])  # drift, not the attempt
        self.assertFalse(blocked["checks"]["attempt"])


class PromoteThirdCandidateV1Tests(unittest.TestCase):
    """`promote` under the review guard (v1), the arms stubbed, 400 seeds: the third candidate of generation 5 is played and judged."""

    setUp, tearDown = tpg.PromoteBase.setUp, tpg.PromoteBase.tearDown
    candidate, arms, block, run_promote = tpg.PromoteBase.candidate, tpg.PromoteBase.arms, tpg.PromoteBase.block, tpg.PromoteBase.run_promote

    def test_a_third_candidate_is_played_and_promoted_when_its_figures_pass_and_the_record_says_the_old_cap_would_have_refused(self):
        judged(self.agents, 5, "s1", "s2")
        cand_rows, base_rows = self.arms(30, 20)
        doc = self.run_promote(self.candidate("gen5c", "sha-a"), cand_rows, base_rows, review_guard=True)
        self.assertEqual((doc["decision_text"], doc["decision"]["rule"]), ("promoted", "D24+" + pg.RULE))
        d = doc["decision"]
        self.assertTrue(d["promote"])
        self.assertTrue(d["attempt_advisory"])
        self.assertEqual(d["checks"], {"floor": True, "boss_kill": True, "boss_damage": True, "drift": True, "attempt": False})  # the old value
        a = doc["review_guard"]["attempt"]
        self.assertEqual((a["attempt"], a["cap"], a["would_refuse"], a["within_cap"]), (3, 2, True, False))
        self.assertEqual((doc["review_guard"]["blocks"], doc["review_guard"]["blocking"]), (False, []))
        row = gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]
        self.assertEqual((row["decision"], row["attempt"]), ("promoted", "3"))
        self.assertIn("attempt 3/2 (over the old cap: recorded, gates nothing)", row["guard"])
        self.assertEqual(json.loads((self.agents / "current.json").read_text())["generation"], 5)

    def test_a_third_candidate_that_fails_a_figure_is_still_refused_for_the_figure(self):
        judged(self.agents, 5, "s1", "s2")
        cand_rows, base_rows = self.arms(8, 20)  # the kill guard fires at 0.05
        doc = self.run_promote(self.candidate("gen5c", "sha-a"), cand_rows, base_rows, review_guard=True)
        self.assertEqual(doc["decision_text"], "rejected")
        self.assertEqual(doc["review_guard"]["blocking"], ["kill"])
        self.assertEqual([k for k, v in doc["decision"]["checks"].items() if not v], ["boss_kill", "attempt"])
        self.assertEqual(gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]["attempt"], "3")
        self.assertFalse((self.agents / "current.json").exists())

    def test_a_fourth_candidate_is_played_too(self):
        judged(self.agents, 5, "s1", "s2", "s3")
        cand_rows, base_rows = self.arms(30, 20)
        doc = self.run_promote(self.candidate("gen5c", "sha-a"), cand_rows, base_rows, review_guard=True)
        self.assertEqual(doc["decision_text"], "promoted")
        self.assertEqual(doc["review_guard"]["attempt"]["attempt"], 4)

    def test_within_the_cap_the_record_says_so(self):
        judged(self.agents, 5, "s1")
        cand_rows, base_rows = self.arms(30, 20)
        doc = self.run_promote(self.candidate("gen5c", "sha-a"), cand_rows, base_rows, review_guard=True)
        self.assertEqual((doc["decision"]["checks"]["attempt"], doc["decision"]["attempt_advisory"]), (True, True))
        self.assertEqual(doc["review_guard"]["attempt"]["would_refuse"], False)
        self.assertNotIn("old cap", gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]["guard"])

    def test_the_flag_off_decision_is_as_it_was(self):
        judged(self.agents, 5, "s1", "s2")
        cand_rows, base_rows = self.arms(30, 20)
        doc = self.run_promote(self.candidate("gen5c", "sha-a"), cand_rows, base_rows, review_guard=False)
        self.assertEqual((doc["decision"]["rule"], set(doc["decision"]["checks"])), ("D24", {"floor", "boss_kill"}))
        self.assertNotIn("attempt_advisory", doc["decision"])
        self.assertEqual(gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]["attempt"], "3")


class PromoteThirdCandidateV2Tests(unittest.TestCase):
    """The keep test v2 through the funnel (stub arms, the synthetic route A gain): the third candidate of generation 4 promotes early."""

    setUp, tearDown, run_promote = tf.PromoteV2Tests.setUp, tf.PromoteV2Tests.tearDown, tf.PromoteV2Tests.run_promote

    def test_a_third_candidate_is_promoted_early_and_the_record_shows_the_attempt_and_the_old_cap(self):
        judged(self.agents, 4, "sha-x", "sha-y")
        doc, arms = self.run_promote(self.cand_rows, self.base_rows, tf.stub_candidate(self.root, "c1", "sha-a", generation=4))
        self.assertTrue(arms.calls)  # played
        self.assertEqual(doc["decision_text"], "promoted")
        self.assertEqual(doc["sequential"]["stop"]["action"], "promote")
        d = doc["decision"]
        self.assertTrue(d["promote"])
        self.assertTrue(d["attempt_advisory"])
        self.assertFalse(d["checks"]["attempt"])
        self.assertTrue(d["checks"]["sequential"])
        self.assertEqual([k for k, v in d["checks"].items() if not v], ["attempt"])
        k = doc["keep_v2"]
        self.assertEqual((k["attempt"]["attempt"], k["attempt"]["cap"], k["attempt"]["would_refuse"]), (3, 2, True))
        new = gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]
        self.assertEqual((new["decision"], new["attempt"], new["rule"]), ("promoted", "3", "v2"))
        self.assertIn("attempt 3/2 (over the old cap: recorded, gates nothing)", new["guard"])
        cur = json.loads((self.agents / "current.json").read_text())
        self.assertEqual((cur["generation"], Path(cur["dir"]).name), (4, "c1"))

    def test_the_decision_the_keep_verdict_carries_into_compare_arms_is_advisory_too(self):
        judged(self.agents, 4, "sha-x", "sha-y")
        rows = gen_loop.read_promotions(self.agents / "promotions.tsv")
        info = gen_loop.incumbent_frontier(self.agents, rows, "a3-v1", block="Etest-2")
        keep = gen_loop.keep_v2_verdict(self.cand_rows, self.base_rows, rows, cand_gen=4, cand_sha="sha-a", block="Etest-2", frontier_info=info, boot=500)
        decision = gen_loop.decide_v2(keep)
        self.assertEqual((decision["promote"], decision["attempt_advisory"], decision["checks"]["attempt"]), (True, True, False))


class RoundTests(unittest.TestCase):
    """The lab's rounds: a third self-play budget, a third training round and a third built candidate all go ahead."""

    setUp, tearDown, judged, train, markers = (tpg.CappedRoundTests.setUp, tpg.CappedRoundTests.tearDown, tpg.CappedRoundTests.judged,
                                               tpg.CappedRoundTests.train, tpg.CappedRoundTests.markers)

    def test_mark_ready_issues_a_third_budget_with_the_guard_on(self):
        lab, runs, tsv = self.root / "lab", self.root / "runs", self.agents / "promotions.tsv"
        (lab / "train").mkdir(parents=True)
        for name in ("done-1", "done-1-r1"):
            (lab / "train" / name).write_text("x")
        self.judged(2, "a", "b")
        with mock.patch.object(gen_loop, "runs_in_generation", return_value=15):  # three budgets of 5 spent
            for flag in (True, None):
                with mock.patch.dict(os.environ, {pg.FLAG: "1"}):
                    got = gen_loop.mark_ready(lab, 1, runs, budget=5, tsv=tsv, review_guard=flag)
                self.assertEqual(got, lab / "train" / "ready-1-r2", flag)
                got.unlink()
        # a budget still waits for the generation's earlier rejection and for its runs, as before
        (lab / "train" / "ready-1-r2").write_text("x")
        with mock.patch.object(gen_loop, "runs_in_generation", return_value=15):
            self.assertIsNone(gen_loop.mark_ready(lab, 1, runs, budget=5, tsv=tsv, review_guard=True))  # r2 issued, r3 needs its rejection
        with mock.patch.object(gen_loop, "runs_in_generation", return_value=3):
            (lab / "train" / "ready-1-r2").rename(lab / "train" / "done-1-r2")
            self.judged(2, "c")
            self.assertIsNone(gen_loop.mark_ready(lab, 1, runs, budget=5, tsv=tsv, review_guard=True))  # not enough runs yet

    def test_train_goes_on_with_two_candidates_judged_and_says_where_the_old_cap_would_have_stopped_it(self):
        self.judged(10, "a5801c02", "e9d89ea6")
        for argv in (("--review-guard",), ()):
            if not argv:
                os.environ[pg.FLAG] = "1"
            rc, calls, out = self.train(*argv, marker="ready-9-r2")
            self.assertEqual((rc, len(calls)), (0, 1), argv)
            self.assertEqual(self.markers(), ["ready-9-r2"])  # the lab renames it after the job, as for any round
            self.assertIn('"candidate"', out)
            (self.train_dir / "ready-9-r2").unlink()
        self.assertEqual(gen_loop.attempts_spent(self.agents / "promotions.tsv", 10, True), 2)  # the figure the old cap acted on is still read

    def test_train_under_v2_goes_on_too(self):
        self.judged(10, "a", "b")
        with mock.patch.dict(os.environ):
            os.environ.pop(pg.FLAG, None)
            rc, calls, _ = self.train("--keep-rule", "v2")
        self.assertEqual((rc, len(calls)), (0, 1))


class BuildThirdAttemptTests(unittest.TestCase):
    """`gen0.py build --attempt 3`: the third candidate of a generation is built beside the others and judged on the promotion path."""

    setUp, tearDown, build = tg12.PlacementTests.setUp, tg12.PlacementTests.tearDown, tg12.PlacementTests.build

    def test_the_third_attempt_is_built_and_its_marker_is_taken_by_the_train_job(self):
        tsv = self.agents / "promotions.tsv"
        for sha in ("older-1", "older-2"):  # generation 7 has two judged candidates
            gen_loop.append_promotion(tsv, {"cand_gen": 7, "seed_block": f"E7-{sha}", "cand_sha": sha, "decision": "rejected"})
        self.assertEqual(0, self.build("--attempt", "3"))
        third = self.agents / "gen7" / "candidate-a3"
        self.assertTrue(third.is_dir())
        self.assertEqual(["ready-5-g7-a3"], [p.name for p in self.marker_dir.iterdir()])
        self.assertEqual(third, gen_loop.built_candidate(self.agents, 7, 0, 3))
        self.assertEqual(3, json.loads((third / "agent.json").read_text())["built"]["attempt"])
        seeds = self.root / "seeds"
        seeds.mkdir()
        args = ["train", "--marker-dir", str(self.marker_dir), "--shards-root", str(self.root), "--agent-root", str(self.agents),
                "--promote", "--emu-python", "python", "--seeds-file", str(seeds / "E{k}-{date}.json"), "--review-guard"]
        out = io.StringIO()
        with mock.patch.object(gen_loop, "train_generation") as tg, mock.patch.object(gen_loop, "run_promote_subprocess", return_value=0) as rp, \
                mock.patch.object(gen_loop, "COMMITTED_SEEDS", seeds), contextlib.redirect_stdout(out):
            self.assertEqual(0, gen0.main(args))
        tg.assert_not_called()  # built, not trained
        self.assertEqual(third, rp.call_args[0][1])  # and sent to the judge
        self.assertNotIn("nothing trained", out.getvalue())

    def test_a_built_vrun_candidate_of_the_third_attempt_carries_its_number(self):
        cand = gen_loop.build_candidate(self.prior, self.agents / "current.json", self.agents, 7, why="third", export=False, attempt=3)
        self.assertEqual("candidate-a3", cand.name)
        self.assertEqual(3, json.loads((cand / "agent.json").read_text())["built"]["attempt"])


class BuildVrunThirdAttemptTests(unittest.TestCase):
    """`gen0.py build-vrun` (round 4's way of building): the third candidate of a generation under the incumbent's recipe builds and goes to the judge."""

    setUp, build, record = tvs.VrunSwapTests.setUp, tvs.VrunSwapTests.build, tvs.VrunSwapTests.record

    def test_the_command_builds_the_third_candidate_and_the_train_job_sends_it_to_the_judge(self):
        recipe = gen_loop.candidate_recipe(gen_loop.load_agent(self.inc))
        tsv = self.agents / "promotions.tsv"
        for sha in ("older-1", "older-2"):  # two candidates of generation 7 under the incumbent's recipe were judged and refused
            gen_loop.append_promotion(tsv, {"cand_gen": 7, "seed_block": f"E7-{sha}", "cand_sha": sha, "run_head_sha": sha, "macro": recipe,
                                            "decision": "rejected", "attempt": 1})
        argv = ["build-vrun", "--base", str(self.inc), "--vrun-fit", str(self.fit), "--agent-root", str(self.agents), "--candidate-generation", "7",
                "--tag", "cheap-v3", "--why", "test", "--no-export", "--marker-dir", str(self.lab / "train")]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(0, gen0.main(argv))
        cand = self.agents / "gen7" / "candidate-cheap-v3"
        self.assertTrue((cand / "run_value.npz").is_file())
        self.assertTrue((self.lab / "train" / "ready-6-g7-cheap-v3").is_file())
        sha, head = gen_loop.candidate_identity(gen_loop.load_agent(cand))
        rows = gen_loop.read_promotions(tsv)
        self.assertEqual(gen_loop.attempts_before(rows, 7, sha, head, recipe) + 1, 3)  # it is attempt 3 of its recipe ...
        self.assertEqual(gen_loop.attempts_spent(tsv, 7, True, recipe), 2)  # ... where the old cap stood
        seeds = Path(self.tmp.name) / "seeds"
        seeds.mkdir()
        args = ["train", "--marker-dir", str(self.lab / "train"), "--shards-root", str(self.tmp.name), "--agent-root", str(self.agents),
                "--promote", "--emu-python", "python", "--seeds-file", str(seeds / "E{k}-{date}.json"), "--review-guard"]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(gen_loop, "train_generation") as tg, mock.patch.object(gen_loop, "run_promote_subprocess", return_value=0) as rp,                 mock.patch.object(gen_loop, "COMMITTED_SEEDS", seeds), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(0, gen0.main(args))
        tg.assert_not_called()
        self.assertEqual(cand, rp.call_args[0][1])  # judged
        self.assertNotIn("nothing trained", out.getvalue() + err.getvalue())
        self.assertIn("recorded, not gating", err.getvalue())


class CountTests(unittest.TestCase):
    def test_the_attempt_is_still_counted_by_distinct_candidates_of_a_generation(self):
        rows = [{"cand_gen": "2", "cand_sha": "a", "decision": "rejected"}, {"cand_gen": "2", "cand_sha": "b", "decision": "rejected"}]
        self.assertEqual(gen_loop.attempts_before(rows, 2, "c") + 1, 3)
        self.assertTrue(pg.retry_cap(gen_loop.attempts_before(rows, 2, "c"))["would_refuse"])
        self.assertFalse(pg.retry_cap(gen_loop.attempts_before(rows, 3, "c"))["would_refuse"])


if __name__ == "__main__":
    unittest.main()
