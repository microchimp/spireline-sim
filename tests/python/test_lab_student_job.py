"""tools/lab/student_job.py: the lab's student job command (docs/lab/RUN_LOOP.md).

Lane 3's `run_fit` and `build_student_candidate` are stood in for (no GPU, no fit): the tests pin what the command adds to `run_student.py round`: a
frozen kind is left out of the loader and the candidate, the result the runner reads, the exit codes (0 built, 3 no kind passed) and the run-level bar
file beside it.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import run_student as rs  # noqa: E402
import test_run_student as trs  # noqa: E402 -- its miniature stores: a planted teacher recording the log-probabilities of a prior
from tools.lab import jobs  # noqa: E402
from tools.lab import student_job as sj  # noqa: E402


def verdict(passing=("card",), failing=()):
    out = {k: {"pass": False, "leg1": False, "leg2": None, "lambda": None, "why": "leg 1: no held-out decisions of this kind"} for k in rs.KINDS}
    for k in passing:
        out[k] = {"pass": True, "leg1": True, "leg2": True if k in rs.HUMAN_KINDS else None, "lambda": [0.1, 0.1, 0.1], "why": ""}
    for k in failing:
        out[k] = {"pass": False, "leg1": True, "leg2": False, "lambda": [0.1, 0.1, 0.1], "why": "leg 2: won +0.0100 (at most +0.002)"}
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.lab = self.tmp / "LAB"
        self.agents = self.tmp / "agent"
        self.agents.mkdir()
        self.job = self.tmp / "job"
        self.job.mkdir()
        self.fit_args: tuple = ()
        self.teacher_prior_path: object = "unset"
        self.build: dict = {}
        self.loader_calls: list[dict] = []
        self.legs: dict = {"kinds_passing": {"card": [0.1, 0.1, 0.1]}, "verdict": verdict()}
        # lane 3's loader is a recorder; the command wraps it for the fit and puts it back
        self.recorder = mock.Mock(side_effect=lambda root, **kw: self.loader_calls.append(kw) or "TS")
        for target, kw in (("_priority", {}), ("load_shards", {"new": self.recorder}), ("run_fit", {"side_effect": self.fake_fit}),
                           ("build_student_candidate", {"side_effect": self.fake_build})):
            p = mock.patch.object(rs, target, **kw)
            p.start()
            self.addCleanup(p.stop)

    def fake_fit(self, shards_root, prior, out, *, human_dir=None, cfg=None, device=None, exam_ids=None, log=print, teacher_prior_path=None):
        """What run_fit does with the loader, and the files it leaves."""
        rs.load_shards(shards_root, prior=None)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / rs.FIT_FILE).write_text(json.dumps({"data": {"decisions": 12345}}), encoding="utf-8")
        self.fit_args = (str(shards_root), str(prior), str(out), None if human_dir is None else str(human_dir))
        self.teacher_prior_path = teacher_prior_path
        return self.legs

    def fake_build(self, out, base, agent_root, target, *, why, tag="student", kinds=None, **kw):
        self.build = {"base": str(base), "target": target, "tag": tag, "kinds": None if kinds is None else tuple(kinds)}
        cand = Path(agent_root) / f"gen{target}" / f"candidate-{tag}"
        cand.mkdir(parents=True)
        return cand

    def argv(self, **over):
        a = {"--lab": self.lab, "--shards-root": self.tmp / "store", "--prior": self.tmp / "prior.npz", "--human-shards": self.tmp / "human",
             "--out": self.job / "student", "--base": self.agents / "current.json", "--agent-root": self.agents, "--target": 14, "--tag": "student",
             "--result": self.job / "result.json"}
        a.update(over)
        out: list[str] = []
        for k, v in a.items():
            out += [k, str(v)]
        return out

    def result(self):
        return json.loads((self.job / "result.json").read_text(encoding="utf-8"))


class CommandTests(Base):
    def test_a_built_candidate_is_reported_for_the_runner_and_the_exit_is_0(self):
        self.assertEqual(0, sj.main(self.argv()))
        res = self.result()
        self.assertEqual((True, "gen14/candidate-student", 12345, ["card"]), (res["built"], res["candidate"], res["decisions"], sorted(res["kinds_passing"])))
        self.assertTrue(res["verdict"]["card"]["pass"])
        self.assertEqual(14, res["generation"])
        self.assertEqual({"base": str(self.agents / "current.json"), "target": 14, "tag": "student", "kinds": tuple(rs.KINDS)}, self.build)
        self.assertEqual((str(self.tmp / "store"), str(self.tmp / "prior.npz"), str(self.job / "student"), str(self.tmp / "human")), self.fit_args)
        self.assertEqual([], res["frozen"])

    def test_the_prior_the_teacher_decided_with_reaches_the_fit_and_without_the_flag_it_is_none(self):
        sj.main(self.argv())
        self.assertIsNone(self.teacher_prior_path)  # round 0: the start prior is the teacher's own
        sj.main(self.argv(**{"--teacher-prior": self.tmp / "human_prior.npz", "--out": self.job / "student2", "--tag": "student-r2"}))
        self.assertEqual(str(self.tmp / "human_prior.npz"), str(self.teacher_prior_path))
        self.assertEqual(str(self.tmp / "prior.npz"), self.fit_args[1])  # the fit still starts from `--prior`

    def test_the_verdict_says_which_human_bars_were_measured(self):
        cell = {"n": 100, "runs": 30, "diff": 0.0, "lo95": -0.01, "hi95": 0.01}
        self.legs = {"kinds_passing": {"card": [0.1] * 3},
                     "verdict": verdict(passing=("card",), failing=("rest", "route")),
                     "leg2": {"measured": True, "kinds": {"card": {"won": cell, "all": cell}, "rest": {"won": None, "all": cell}}}}
        sj.main(self.argv())
        v = self.result()["verdict"]
        self.assertEqual((True, False, False), (v["card"]["leg2_measured"], v["rest"]["leg2_measured"], v["route"]["leg2_measured"]))
        self.assertNotIn("leg2_measured", v["reward"])  # a kind with no human record has no bar to measure
        self.assertNotIn("leg2_measured", v["select"])

    def test_the_marker_is_left_to_the_runner(self):
        sj.main(self.argv())
        self.assertFalse((self.lab / "train").exists())

    def test_no_kind_passing_is_exit_3_with_the_verdicts_and_no_candidate(self):
        self.legs = {"kinds_passing": {}, "verdict": verdict(passing=(), failing=("card", "rest"))}
        self.assertEqual(3, sj.main(self.argv()))
        res = self.result()
        self.assertEqual((False, None, {}), (res["built"], res["candidate"], res["kinds_passing"]))
        self.assertFalse(res["verdict"]["card"]["leg2"])
        self.assertIn("no kind passed both legs", res["why"])
        self.assertEqual({}, self.build)
        self.assertTrue((self.job / "run-bar.md").exists())

    def test_a_frozen_kind_is_left_out_of_the_candidate_and_a_stale_pass_for_it_is_not_handed_over(self):
        jobs.write_frozen(self.lab, {"card": {"failures": 2}, "rest": {"failures": 2}})
        self.legs = {"kinds_passing": {"card": [0.1] * 3, "route": [0.2] * 3}, "verdict": verdict(passing=("card", "route"))}
        self.assertEqual(0, sj.main(self.argv()))
        res = self.result()
        self.assertEqual((["card", "rest"], ["route"]), (res["frozen"], sorted(res["kinds_passing"])))
        self.assertNotIn("card", res["fitted_kinds"])
        self.assertNotIn("rest", res["fitted_kinds"])
        self.assertNotIn("card", self.build["kinds"])
        self.assertIn("route", self.build["kinds"])

    def test_the_loader_is_asked_for_the_unfrozen_kinds_only_and_put_back_after(self):
        jobs.write_frozen(self.lab, {"shop": {}})
        sj.main(self.argv())
        self.assertEqual(1, len(self.loader_calls))
        self.assertEqual(tuple(k for k in rs.KINDS if k != "shop"), tuple(self.loader_calls[0]["kinds"]))
        self.assertIsNone(self.loader_calls[0]["prior"])
        self.assertIs(self.recorder, rs.load_shards)  # the module is as lane 3 left it
        self.loader_calls.clear()
        jobs.write_frozen(self.lab, {})
        sj.main(self.argv(**{"--out": self.job / "student2", "--tag": "student-r2"}))
        self.assertEqual(tuple(rs.KINDS), tuple(self.loader_calls[0]["kinds"]))  # nothing frozen: every kind

    def test_the_loader_is_put_back_when_the_fit_fails(self):
        with mock.patch.object(rs, "run_fit", side_effect=RuntimeError("cuda")):
            with self.assertRaises(RuntimeError):
                sj.main(self.argv())
        self.assertIs(self.recorder, rs.load_shards)

    def test_every_kind_frozen_is_exit_3_and_nothing_is_fitted(self):
        jobs.write_frozen(self.lab, {k: {} for k in rs.KINDS})
        with mock.patch.object(rs, "run_fit") as fit:
            self.assertEqual(3, sj.main(self.argv()))
            fit.assert_not_called()
        self.assertEqual("every kind is frozen", self.result()["why"])

    def test_the_run_bar_file_is_written_beside_the_result(self):
        sj.main(self.argv())
        self.assertIn("run-level human bar", (self.job / "run-bar.md").read_text(encoding="utf-8"))

    def test_the_run_bar_never_raises(self):
        with mock.patch.dict(sys.modules, {"run_human_bar": None}):
            self.assertIn("not available", sj.run_bar_text(str(self.tmp)))


class Loaded(Exception):
    """Raised by the stand-in for the fit once the loader has passed: the shards were accepted for the prior they were given."""


class StoreTests(unittest.TestCase):
    """The real loader, the real `run_fit` up to the fit: a store the teacher recorded under prior A, a start prior B (the incumbent's student file)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        weights_a, weights_b = trs.random_weights(seed=3), trs.random_weights(seed=99)
        self.human = trs.write_weights(self.tmp / "human_prior.npz", weights_a)  # what the teacher decided with (rl/run_teacher.py: pi_h)
        self.student = trs.student_file(self.tmp / "run_student.npz", weights_b, kinds=("card", "route", "shop", "reward"))  # the incumbent's student
        self.store = self.tmp / "store"
        trs.synth_shards(self.store, trs.Synth(weights_a), runs=12)
        for target, kw in (("_priority", {}), ("fit", {"side_effect": Loaded})):
            p = mock.patch.object(rs, target, **kw)
            p.start()
            self.addCleanup(p.stop)

    def argv(self, prior: Path, *extra: str) -> list[str]:
        return ["--lab", str(self.tmp / "LAB"), "--shards-root", str(self.store), "--prior", str(prior), "--out", str(self.tmp / "out"),
                "--base", str(self.tmp / "current.json"), "--agent-root", str(self.tmp / "agent"), "--target", "15", "--result", str(self.tmp / "result.json"), *extra]

    def test_the_students_file_as_the_start_prior_alone_is_refused_because_the_teacher_recorded_the_human_priors_log_probabilities(self):
        with self.assertRaisesRegex(rs.ShardError, "differ from this prior"):
            sj.main(self.argv(self.student))

    def test_with_the_teachers_prior_named_the_same_store_loads_for_the_students_start(self):
        with self.assertRaises(Loaded):  # the loader accepted the store: the fit is the next thing, and it is stood in for
            sj.main(self.argv(self.student, "--teacher-prior", str(self.human)))

    def test_the_human_prior_as_the_start_prior_needs_no_flag(self):
        with self.assertRaises(Loaded):
            sj.main(self.argv(self.human))

    def test_a_wrong_teacher_prior_is_still_refused(self):
        other = trs.write_weights(self.tmp / "other.npz", trs.random_weights(seed=5))
        with self.assertRaisesRegex(rs.ShardError, "differ from the teacher prior"):
            sj.main(self.argv(self.student, "--teacher-prior", str(other)))


if __name__ == "__main__":
    unittest.main()
