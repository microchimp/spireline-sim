"""A candidate that is the incumbent with its V_run fit replaced and nothing else (D60 item 10, lane J: the cheap-run student), and the
fit's ranking check without the human leg (a lane whose rules keep it off the human exam fifth runs the shard leg only, and says so)."""

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
import screens  # noqa: E402
from test_gen_loop import student_world  # noqa: E402
from test_vrun_matched import OLD, fake_train_run_value, shard_block  # noqa: E402

HUMAN_MISSING = {"name": "human_agreement", "pass": False, "why": "the human leg did not run"}


def ranking_doc(*, heal=True, rest=True, human=None, status="ok") -> dict:
    checks = [{"name": "rest_rate", "candidate": 0.8, "incumbent": 0.8, "diff": 0.0, "limit": -0.05, "pass": rest},
              {"name": "heal_rate", "candidate": 0.5, "incumbent": 0.5, "diff": 0.0, "limit": -0.1, "pass": heal}]
    checks.append(dict(HUMAN_MISSING) if human is None else human)
    return {"status": status, "pass": all(c["pass"] for c in checks), "checks": checks, "set": {"digest": "d" * 8}}


class VrunSwapTests(unittest.TestCase):
    """`gen_loop.build_vrun_candidate` and `gen0.py build-vrun`: the base's folder, its V_run fit replaced by a fit folder's."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.agents, _base5, self.prior, student, _human = student_world(root)
        # the incumbent: generation 6, a student recipe with the base's blend, stakes schedule and hold (the champion's shape)
        self.inc = gen_loop.build_candidate(self.prior, self.agents / "current.json", self.agents, 6, why="incumbent", export=False, tag="inc",
                                            student=student, student_lams={"route": (0.3715, 2.0, 2.0), "card": (0.5, 0.5, 0.5)})
        (self.inc / "run_student_legs.json").write_text(json.dumps({"student_sha256": gen_loop.file_sha256(self.inc / "run_student.npz")}))
        (self.inc / "promotion.json").write_text(json.dumps({"promoted": True}))
        (self.inc / "combat_net.onnx").write_bytes(b"the incumbent's combat onnx")
        (self.inc / "run_value.onnx").write_bytes(b"the incumbent's V onnx")
        (self.agents / "current.json").write_text(json.dumps({"generation": 6, "dir": str(self.inc)}))
        self.fit = root / "fit"
        self.fit.mkdir()
        (self.fit / "run_value.npz").write_bytes(b"the plain-run fit")
        (self.fit / "run_value.pt").write_bytes(b"the plain-run fit, torch")
        self.record(kept=True)
        self.lab = root / "LAB"

    def record(self, *, kept: bool, d33: bool = True, ranking: dict | None = None) -> None:
        sha = gen_loop.load_agent(self.inc).describe()["prior_sha256"]
        if ranking is None:
            ranking = ranking_doc(human={"name": "human_agreement_card_reward", "pass": True})
        (self.fit / gen_loop.VRUN_FIT_JSON).write_text(json.dumps({
            "recipe": gen_loop.MATCHED_RECIPE, "prior_sha256": sha, "beta": 0.3, "kept": kept,
            "keep": {"all_computable_pass": d33, "ranking": ranking},
            "window": {"generations": [6], "runs_by_generation": {"6": 20000}}, "sha256": gen_loop.file_sha256(self.fit / "run_value.npz")}))

    def build(self, **kw):
        return gen_loop.build_vrun_candidate(self.agents / "current.json", self.fit, self.agents, 7, why="test", tag="cheap-v", export=False, **kw)

    def test_the_candidate_is_the_incumbent_with_only_its_v_run_fit_replaced(self):
        cand = self.build()
        self.assertEqual(cand, self.agents / "gen7" / "candidate-cheap-v")
        self.assertEqual(cand, gen_loop.built_candidate(self.agents, 7, tag="cheap-v"))  # the train job judges it as built
        a, b = gen_loop.load_agent(cand), gen_loop.load_agent(self.inc)
        self.assertEqual((7, b.macro, b.recipe()), (a.generation, a.macro, a.recipe()))
        self.assertEqual(gen_loop.candidate_recipe(b), gen_loop.candidate_recipe(a))
        self.assertEqual((b.lam, b.lam_acts, b.stakes, b.hold, b.student_lams), (a.lam, a.lam_acts, a.stakes, a.hold, a.student_lams))
        self.assertEqual(b"the plain-run fit", (cand / "run_value.npz").read_bytes())
        self.assertEqual(b"the plain-run fit, torch", (cand / "run_value.pt").read_bytes())
        self.assertFalse((cand / "run_value.onnx").exists())  # the base's V export is the old fit's: never carried
        self.assertFalse((cand / "promotion.json").exists())
        for name in ("combat_policy.pt", "combat_net.pt", "combat_net.onnx", "human_prior.npz", "run_student.npz", "run_student_legs.json"):
            self.assertEqual((self.inc / name).read_bytes(), (cand / name).read_bytes(), name)
        self.assertEqual(["value"], screens.differing(cand, self.inc))  # the run-level judgement alone: no screen reads it
        self.assertIsNone(screens.kind_of(cand, self.inc))
        meta = json.loads((cand / "agent.json").read_text())
        base_meta = json.loads((self.inc / "agent.json").read_text())
        for key in ("macro", "recipe", "human_prior", "student", "search_schedule", "potion_hold", "mask_version"):
            self.assertEqual(base_meta.get(key), meta.get(key), key)
        self.assertEqual((7, 6), (meta["generation"], meta["built_on"]))
        self.assertEqual(("fit", gen_loop.file_sha256(cand / "run_value.npz"), True, 0.3),
                         (meta["vrun_fit"]["source"], meta["vrun_fit"]["sha256"], meta["vrun_fit"]["matched"], meta["vrun_fit"]["beta"]))
        self.assertEqual("run", meta["vrun_fit"]["human_leg"])
        self.assertIn("V_run fit alone", meta["note"])
        self.assertEqual("vrun", meta[gen_loop.BUILT_KEY]["swap"])
        self.assertEqual(gen_loop.file_sha256(self.inc / "run_value.npz"), meta[gen_loop.BUILT_KEY]["base"]["run_value_sha256"])
        self.assertEqual("fit", json.loads((cand / "train.json").read_text())[gen_loop.BUILT_KEY]["vrun_fit"]["source"])
        _, head_a = gen_loop.candidate_identity(a)
        _, head_b = gen_loop.candidate_identity(b)
        self.assertNotEqual(head_a, head_b)  # a new run head: a new attempt in the ledger, under the incumbent's recipe

    def test_a_fit_that_was_not_kept_is_refused_before_anything_is_written(self):
        self.record(kept=False, d33=False)
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse((self.agents / "gen7").exists())

    def test_a_fit_whose_only_missing_check_is_the_human_leg_needs_the_flag_and_says_so(self):
        self.record(kept=False, ranking=ranking_doc())
        with self.assertRaisesRegex(ValueError, "keep test"):
            self.build()
        self.assertFalse((self.agents / "gen7").exists())
        meta = json.loads((self.build(without_human_leg=True) / "agent.json").read_text())
        self.assertEqual("not run", meta["vrun_fit"]["human_leg"][:7])
        self.assertIn("human leg", meta["note"])

    def test_the_flag_never_passes_a_fit_that_failed_anything_else(self):
        for kw in ({"ranking": ranking_doc(heal=False)}, {"ranking": ranking_doc(rest=False)}, {"d33": False, "ranking": ranking_doc()},
                   {"ranking": ranking_doc(status="not_run")},
                   {"ranking": ranking_doc(human={"name": "human_agreement_rest", "pass": False, "diff": -0.05, "limit": -0.01})}):
            self.record(kept=False, **kw)
            with self.assertRaises(ValueError, msg=str(kw)):
                self.build(without_human_leg=True)
        self.assertFalse((self.agents / "gen7").exists())

    def test_an_existing_folder_or_a_lower_generation_is_refused(self):
        self.build()
        with self.assertRaises(FileExistsError):
            self.build()
        with self.assertRaises(ValueError):
            gen_loop.build_vrun_candidate(self.inc, self.fit, self.agents, 6, why="t", tag="again", export=False)

    def test_a_base_whose_chooser_reads_no_v_is_refused(self):
        rules = self.agents / "gen3" / "candidate"
        rules.mkdir(parents=True)
        for name in ("combat_policy.pt", "combat_net.pt", "model.json"):
            (rules / name).write_bytes((self.inc / name).read_bytes())
        (rules / "agent.json").write_text(json.dumps({"generation": 3, "macro": "rules", "mask_version": 4}))
        with self.assertRaisesRegex(ValueError, "reads no V"):
            gen_loop.build_vrun_candidate(rules, self.fit, self.agents, 7, why="t", tag="cheap-v", export=False)

    def test_the_command_builds_it_and_writes_the_marker_once(self):
        self.record(kept=False, ranking=ranking_doc())
        argv = ["build-vrun", "--base", str(self.inc), "--vrun-fit", str(self.fit), "--agent-root", str(self.agents), "--candidate-generation", "7",
                "--tag", "cheap-v", "--why", "test", "--no-export", "--marker-dir", str(self.lab / "train")]
        with mock.patch("sys.stdout"), self.assertRaises(ValueError):
            gen0.main(argv)
        self.assertFalse((self.lab / "train" / "ready-6-g7-cheap-v").exists())
        with mock.patch("sys.stdout"):
            self.assertEqual(0, gen0.main([*argv, "--without-human-leg"]))
        marker = self.lab / "train" / "ready-6-g7-cheap-v"
        self.assertTrue(marker.is_file())
        self.assertEqual((7, "cheap-v"), (gen_loop.marker_target(marker), gen_loop.marker_tag(marker)))
        with mock.patch("sys.stderr"), mock.patch("sys.stdout"):
            self.assertEqual(2, gen0.main([*argv, "--without-human-leg"]))  # the marker stands: not built again


class RankingWithoutHumanLegTests(unittest.TestCase):
    """`fit_run_value_matched(ranking_runs_root=, human_leg=False)` and `gen0.py vrun-fit --ranking-runs-root --no-human-leg`."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.runs = root / "store" / "jobs"
        shard_block(self.runs, "j15a", 15, OLD, ["A000000001", "A000000002"])
        self.pool = root / "lab-runs"
        self.pool.mkdir()
        fake_train_run_value.calls, fake_train_run_value.passes = [], True
        self.incumbent = root / "inc" / "run_value.npz"
        self.incumbent.parent.mkdir()
        self.incumbent.write_bytes(b"incumbent head")
        self.no_seeds = root / "no-seeds"

    def fit(self, **kw):
        seen: list = []

        def check(candidate_npz, incumbent_npz, runs_root, out_dir, **k):
            seen.append({"runs_root": Path(runs_root), **k})
            if "human" in k and k["human"] is None:
                return ranking_doc()
            return ranking_doc(human={"name": "human_agreement_rest", "pass": True})

        with mock.patch.object(gen_loop, "train_run_value", side_effect=fake_train_run_value), \
             mock.patch("run_train.ranking.run_check", side_effect=check):
            doc = gen_loop.fit_run_value_matched(self.runs, OLD, Path(self.tmp.name) / "fit", beta=0.3, seed_dir=self.no_seeds,
                                                 incumbent=self.incumbent, **kw)
        return doc, seen

    def test_the_ranking_reads_the_pool_it_is_given_and_skips_the_human_leg_when_asked(self):
        doc, seen = self.fit(ranking_runs_root=self.pool, human_leg=False)
        self.assertEqual(self.pool, seen[0]["runs_root"])
        self.assertIsNone(seen[0]["human"])
        self.assertEqual(["A000000001", "A000000002"], sorted(seen[0]["exclude_seeds"]))
        self.assertFalse(doc["kept"])  # a missing check never keeps the fit ...
        self.assertFalse(doc["human_leg"])
        self.assertEqual(0.3, doc["beta"])
        out = Path(self.tmp.name) / "fit"
        with self.assertRaises(ValueError):
            gen_loop.read_vrun_fit(out)
        self.assertEqual(doc, gen_loop.read_vrun_fit(out, without_human_leg=True))  # ... the flag reads it, the human leg alone missing

    def test_by_default_the_check_is_as_before(self):
        doc, seen = self.fit()
        self.assertEqual(self.runs, seen[0]["runs_root"])
        self.assertNotIn("human", seen[0])
        self.assertTrue(doc["kept"])
        self.assertTrue(doc["human_leg"])

    def test_the_command_passes_the_pool_and_the_flag(self):
        got = {}

        def fake(shards_root, sha, out, **kw):
            got.update(kw, shards_root=shards_root)
            return {"window": {"runs_by_generation": {}}, "beta": kw["beta"], "kept": False, "keep": {}, "run_value": {}}

        with mock.patch.object(gen_loop, "fit_run_value_matched", side_effect=fake), mock.patch("sys.stdout"):
            rc = gen0.main(["vrun-fit", "--shards-root", str(self.runs), "--prior-sha", OLD, "--incumbent", str(self.incumbent), "--beta", "0.3",
                            "--out", str(Path(self.tmp.name) / "fit"), "--ranking-runs-root", str(self.pool), "--no-human-leg"])
        self.assertEqual(3, rc)
        self.assertEqual((self.pool, False, 0.3), (got["ranking_runs_root"], got["human_leg"], got["beta"]))


if __name__ == "__main__":
    unittest.main()
