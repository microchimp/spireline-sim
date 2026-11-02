"""The fighter-matched V_run recipe (reset order step C): the window by combat prior, the pure Monte Carlo fit, the built
candidate that carries the fit, and run-level exploration that only recorded self-play may switch on."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, ROOT / "tests" / "python"):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import gen0  # noqa: E402
import gen_loop  # noqa: E402
from test_gen_loop import fake_record, fake_run_arrays, trained_candidate  # noqa: E402

OLD, NEW = "a" * 64, "b" * 64


def shard_block(root: Path, name: str, generation: int, prior_sha: str | None, seeds: list[str]) -> Path:
    rng = np.random.default_rng(len(seeds))
    agent = {"generation": generation, "prior": prior_sha is not None, "prior_sha256": prior_sha, "macro": "vrun"}
    sink = gen_loop.ShardSink(root / name, generation, config="a3-v1", base_sims=30, ascension=10, agent=agent, part_runs=2)
    for s in seeds:
        sink.add(fake_record(s, generation), fake_run_arrays(rng, 2, 3))
    sink.close()
    return root / name


def fake_train_run_value(paths, out, **kw):
    """Stands in for the torch fit: writes the two files and returns a keep record that passes unless told otherwise."""
    Path(out, "run_value.npz").write_bytes(b"fit " + str(len(paths)).encode())
    Path(out, "run_value.pt").write_bytes(b"torch fit")
    fake_train_run_value.calls.append({"paths": [Path(p).parent.parent.name for p in paths], **kw})
    return {"wall_min": 0.1, "value": {"runs": 4, "keep": {"all_computable_pass": fake_train_run_value.passes}}}


class FighterWindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self.tmp.name) / "runs"
        shard_block(self.runs, "j9a", 9, OLD, ["A000000001", "A000000002"])
        shard_block(self.runs, "j9b", 9, OLD, ["A000000003", "A000000004"])
        shard_block(self.runs, "j11", 11, NEW, ["B000000001", "B000000002"])
        shard_block(self.runs, "jrules", 3, None, ["C000000001", "C000000002"])
        fake_train_run_value.calls, fake_train_run_value.passes = [], True
        self.no_seeds = Path(self.tmp.name) / "no-seeds"

    def tearDown(self):
        self.tmp.cleanup()

    def fit(self, sha=OLD, out="fit", **kw):
        kw.setdefault("seed_dir", self.no_seeds)
        with mock.patch.object(gen_loop, "train_run_value", side_effect=fake_train_run_value):
            return gen_loop.fit_run_value_matched(self.runs, sha, Path(self.tmp.name) / out, **kw)

    def test_the_window_is_the_blocks_one_prior_played(self):
        self.assertEqual(["j9a", "j9b"], sorted(d.parent.name for d, _ in gen_loop.fighter_manifests(self.runs, OLD)))
        self.assertEqual(["j11"], [d.parent.name for d, _ in gen_loop.fighter_manifests(self.runs, NEW)])
        self.assertEqual([], gen_loop.fighter_manifests(self.runs, NEW, gens=[9]))
        self.assertEqual([], gen_loop.fighter_manifests(self.runs, None))  # a block without a prior sha is never matched

    def test_the_fit_reads_only_that_window_with_pure_monte_carlo_and_records_it(self):
        doc = self.fit()
        call = fake_train_run_value.calls[0]
        self.assertEqual({"j9a", "j9b"}, set(call["paths"]))
        self.assertEqual((9, 1, 0.0, 4), (call["generation"], call["value_generations"], call["beta"], call["threads"]))
        self.assertTrue(doc["kept"])
        self.assertEqual({"9": 4}, doc["window"]["runs_by_generation"])
        out = Path(self.tmp.name) / "fit"
        saved = json.loads((out / gen_loop.VRUN_FIT_JSON).read_text())
        self.assertEqual((OLD, 0.0, gen_loop.file_sha256(out / "run_value.npz")), (saved["prior_sha256"], saved["beta"], saved["sha256"]))
        self.assertEqual(saved, gen_loop.read_vrun_fit(out))

    def test_a_window_holding_an_e_block_seed_or_nothing_is_refused(self):
        seeds = Path(self.tmp.name) / "seeds"
        seeds.mkdir()
        (seeds / "E12-test.json").write_text(json.dumps({"seeds": ["A000000003"]}))
        with self.assertRaises(ValueError):
            self.fit(seed_dirs=[seeds])
        with self.assertRaises(ValueError):
            self.fit(sha="c" * 64, out="fit2")
        self.assertEqual([], fake_train_run_value.calls)

    def test_a_fit_that_failed_the_keep_test_or_was_changed_is_never_read_as_kept(self):
        out = Path(self.tmp.name) / "fit"
        fake_train_run_value.passes = False
        self.assertFalse(self.fit()["kept"])
        with self.assertRaises(ValueError):
            gen_loop.read_vrun_fit(out)
        fake_train_run_value.passes = True
        self.fit()
        (out / "run_value.npz").write_bytes(b"another file")
        with self.assertRaises(ValueError):
            gen_loop.read_vrun_fit(out)

    def test_the_fit_passes_beta_and_the_buffer_to_the_trainer(self):
        import torch

        from run_train import train as rt

        seen = {}

        def fit(shards, prev, pt, **kw):
            seen.update(kw)
            torch.save({"meta": {"value": {"runs": 1}, "config": {}}}, pt)

        with mock.patch.object(rt, "fit", side_effect=fit):
            gen_loop.train_run_value([], Path(self.tmp.name), generation=9, base_sims=30, ascension=10, threads=2, device="cpu",
                                     value_generations=1, beta=0.0)
            self.assertEqual((1, 0.0), (seen["config"].value_generations, seen["config"].beta))
            gen_loop.train_run_value([], Path(self.tmp.name), generation=9, base_sims=30, ascension=10, threads=2, device="cpu")
            self.assertEqual((gen_loop.GEN_WINDOW, rt.TrainConfig().beta), (seen["config"].value_generations, seen["config"].beta))


class BuiltWithFitTests(unittest.TestCase):
    """`build_candidate(vrun_fit=)` and `gen0.py build --vrun-fit`: the prior of one export with a fighter-matched fit."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        _, _, trained = trained_candidate()
        self.agents = root / "agent"
        self.base = self.agents / "gen11" / "candidate"
        self.base.mkdir(parents=True)
        for name in ("combat_policy.pt", "combat_net.pt", "model.json"):
            (self.base / name).write_bytes((trained / name).read_bytes())
        (self.base / "run_value.npz").write_bytes(b"the base fit")
        (self.base / "run_value.pt").write_bytes(b"the base fit, torch")
        (self.base / "agent.json").write_text(json.dumps({"generation": 11, "macro": "vrun", "mask_version": 4}))
        (self.agents / "current.json").write_text(json.dumps({"generation": 11, "dir": str(self.base)}))
        self.sha = gen_loop.load_agent(self.base).describe()["prior_sha256"]
        self.fit = root / "fit"
        self.fit.mkdir()
        (self.fit / "run_value.npz").write_bytes(b"the matched fit")
        (self.fit / "run_value.pt").write_bytes(b"the matched fit, torch")
        self.record(kept=True, sha=self.sha)

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, kept: bool, sha: str) -> None:
        (self.fit / gen_loop.VRUN_FIT_JSON).write_text(json.dumps({
            "recipe": gen_loop.MATCHED_RECIPE, "prior_sha256": sha, "beta": 0.0, "kept": kept,
            "window": {"generations": [11], "runs_by_generation": {"11": 8000}},
            "sha256": gen_loop.file_sha256(self.fit / "run_value.npz")}))

    def build(self, **kw):
        return gen_loop.build_candidate(self.base, self.agents / "current.json", self.agents, 12, why="test", export=False, **kw)

    def test_the_candidate_carries_the_fit_and_says_what_its_e_block_judges(self):
        cand = self.build(vrun_fit=self.fit)
        agent = gen_loop.load_agent(cand)
        self.assertEqual((12, "vrun", self.sha), (agent.generation, agent.macro, agent.describe()["prior_sha256"]))
        self.assertEqual(b"the matched fit", Path(agent.macro_npz).read_bytes())
        self.assertEqual(b"the matched fit, torch", (cand / "run_value.pt").read_bytes())
        meta = json.loads((cand / "agent.json").read_text())
        self.assertEqual(("fit", True, 0.0), (meta["vrun_fit"]["source"], meta["vrun_fit"]["matched"], meta["vrun_fit"]["beta"]))
        self.assertIn("the prior and the fit together", meta["note"])
        self.assertEqual("fit", json.loads((cand / "train.json").read_text())["built"]["vrun_fit"]["source"])

    def test_a_fit_another_prior_played_is_carried_but_marked(self):
        self.record(kept=True, sha="d" * 64)
        meta = json.loads((self.build(vrun_fit=self.fit) / "agent.json").read_text())
        self.assertFalse(meta["vrun_fit"]["matched"])

    def test_a_fit_that_was_not_kept_is_refused_before_anything_is_written(self):
        self.record(kept=False, sha=self.sha)
        with self.assertRaises(ValueError):
            self.build(vrun_fit=self.fit)
        self.assertFalse((self.agents / "gen12").exists())

    def test_without_a_fit_the_build_is_as_before(self):
        meta = json.loads((self.build() / "agent.json").read_text())
        self.assertEqual("base", meta["vrun_fit"]["source"])
        self.assertIn("the prior swap alone", meta["note"])

    def test_the_build_command_takes_the_fit(self):
        with mock.patch("sys.stdout"):
            rc = gen0.main(["build", "--prior", str(self.base), "--agent-root", str(self.agents), "--candidate-generation", "12",
                            "--why", "test", "--no-export", "--vrun-fit", str(self.fit)])
        self.assertEqual(0, rc)
        self.assertEqual("fit", json.loads((self.agents / "gen12" / "candidate" / "agent.json").read_text())["vrun_fit"]["source"])


class ExplorationTests(unittest.TestCase):
    """DESIGN.md 5.5's sampling, for recorded self-play only."""

    def agent(self, **kw):
        return gen_loop.Agent("x/gen11/candidate", 11, None, "vrun", "x.npz", **kw)

    def test_the_exploring_agent_samples_and_says_so_and_the_argmax_agent_is_unchanged(self):
        base = self.agent()
        self.assertNotIn("tau", base.describe())
        self.assertNotIn("eps", base.describe())
        got = gen_loop.exploring_agent(base, 0.25, 0.02)
        self.assertEqual((0.25, 0.02), (got.tau, got.eps))
        self.assertEqual((0.25, 0.02), (got.describe()["tau"], got.describe()["eps"]))
        self.assertEqual({k: v for k, v in got.describe().items() if k not in ("tau", "eps")}, base.describe())

    def test_only_a_vrun_macro_explores(self):
        with self.assertRaises(ValueError):
            gen_loop.exploring_agent(gen_loop.Agent(), 0.25, 0.0)
        with self.assertRaises(ValueError):
            gen_loop.exploring_agent(None, 0.25, 0.0)
        with self.assertRaises(ValueError):
            gen_loop.exploring_agent(self.agent(), 0.25, 1.5)

    def test_emu_explores_only_when_it_records(self):
        with mock.patch.object(gen_loop, "load_agent", return_value=self.agent()), \
                mock.patch.object(gen0, "stream_seeds", return_value=["S1"]), mock.patch.object(gen0, "_setup"), \
                mock.patch.object(gen0, "run_block", return_value={}) as rb:
            with self.assertRaises(SystemExit):
                gen0.main(["emu", "--stream", "x", "--out", "o", "--gen-dir", "g", "--explore-tau", "0.25", "--prior-server", "off"])
            rb.assert_not_called()
            with mock.patch("sys.stdout"):
                gen0.main(["emu", "--stream", "x", "--out", "o", "--gen-dir", "g", "--record", "--explore-tau", "0.25",
                           "--explore-eps", "0.02", "--prior-server", "off"])
            self.assertEqual((0.25, 0.02), (rb.call_args[1]["agent"].tau, rb.call_args[1]["agent"].eps))
            with mock.patch("sys.stdout"):
                gen0.main(["emu", "--stream", "x", "--out", "o", "--gen-dir", "g", "--record", "--prior-server", "off"])
            self.assertEqual((0.0, 0.0), (rb.call_args[1]["agent"].tau, rb.call_args[1]["agent"].eps))

    def test_an_exploring_lookahead_draws_from_the_runs_stream(self):
        from run_train import policy

        pol = policy.LookaheadPolicy.__new__(policy.LookaheadPolicy)
        pol.tau, pol.eps = 0.25, 0.02
        pi = pol.distribution(np.array([10.0, 9.9, 8.0]))
        self.assertTrue(pi[1] > 0.2 and pi[2] > 0.0)
        with self.assertRaises(ValueError):
            policy._draw(pi, None)  # an exploring pick without the run's stream is an error, never a silent argmax
        a = [policy._draw(pi, np.random.default_rng([7, i, 1])) for i in range(50)]
        b = [policy._draw(pi, np.random.default_rng([7, i, 1])) for i in range(50)]
        self.assertEqual(a, b)
        self.assertGreater(len(set(a)), 1)


if __name__ == "__main__":
    unittest.main()
