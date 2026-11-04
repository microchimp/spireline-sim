"""docs/rl/PRIOR_SWEEP.md: the recipe sweep (rl/prior_sweep.py) on synthetic shards: the evaluation set against the training sets, the grid
and its order, the temporary-folder fit and the stop file, the measures on synthetic predictions, the main-effects arithmetic, the data
arm, the confirmation, the queue writer, and no absolute path in anything written. CPU, no lab, no emulator."""

import argparse
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "rl", ROOT / "src", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))

from tests.python.test_gen_loop import fake_record, fake_run_arrays, gen_loop  # noqa: E402  (puts rl/ on the path)

import prior_sweep as ps  # noqa: E402
import train_combat  # noqa: E402

from tools.gpu_lane import quiet_runner as qr  # noqa: E402

PATH_PATTERN = re.compile(r"(?<![\w])[A-Za-z]:[\\/]|(?<![\w:])/(?:home|Users|tmp|var|opt|mnt|d)/", re.IGNORECASE)


def assert_no_paths(test: unittest.TestCase, text: str, *roots: Path) -> None:
    test.assertIsNone(PATH_PATTERN.search(text), text[:200])
    for root in roots:
        test.assertNotIn(str(root), text)
        test.assertNotIn(root.as_posix(), text)


def build_lab(root: Path, runs_per_job: int = 30, generations: tuple[int, ...] = (0, 1)) -> tuple[Path, Path, dict[str, list[str]]]:
    """A synthetic lab with one closed block per generation; returns (lab, agent root, seeds by generation)."""
    rng = np.random.default_rng(5)
    lab, agent = root / "lab", root / "agent"
    agent.mkdir(parents=True)
    (agent / "current.json").write_text(json.dumps({"generation": max(generations)}), encoding="utf-8")
    seeds: dict[str, list[str]] = {}
    for g in generations:
        job = lab / "runs" / f"job{g}"
        sink = gen_loop.ShardSink(job, g, config="a3-v1", base_sims=30, ascension=10, agent={}, part_runs=7)
        for i in range(runs_per_job):
            seed = f"S{g}{i:08d}"
            sink.add(fake_record(seed, g), fake_run_arrays(rng, 3, 4))
            seeds.setdefault(str(g), []).append(seed)
        sink.close()
        (job / "done.rc").write_text("0\n", encoding="utf-8")
    return lab, agent, seeds


def eval_seed_flags(seeds: list[str]) -> list[bool]:
    return list(ps.runs_in_range(np.array(seeds)))


class GridTests(unittest.TestCase):
    def test_the_grid_has_24_cells_named_target_epochs_dmodel_layers_with_the_labs_first(self):
        names = ps.grid_names()
        self.assertEqual(len(names), 24)
        self.assertEqual(len(set(names)), 24)
        self.assertEqual(names[0], "visits-16-128-2")
        self.assertEqual(names[1:4], ["visits-16-128-4", "visits-16-256-2", "visits-16-256-4"])
        self.assertEqual(names[-1], "cq3-32-256-4")
        self.assertEqual({n.split("-")[0] for n in names}, {"visits", "cq1", "cq3"})

    def test_arms_resolve_to_the_recipe_their_names_say(self):
        a = ps.resolve_arm("cq3-32-256-4")
        self.assertEqual((a.policy_target, a.q_temperature, a.epochs, a.d_model, a.layers, a.data_name), ("completed_q", 3.0, 32, 256, 4, "train-6400"))
        self.assertEqual(ps.resolve_arm("cq1-16-128-2").q_temperature, 1.0)
        v = ps.resolve_arm("visits-16-128-2")
        self.assertEqual((v.policy_target, v.q_temperature), ("visits", None))
        b = ps.resolve_arm("lab-bs8")
        self.assertEqual((b.target, b.epochs, b.d_model, b.layers, b.build_seed, b.data_name), ("visits", 16, 128, 2, 8, "train-6400-bs8"))
        d = ps.resolve_arm("data12800")
        self.assertEqual((d.epochs, d.d_model, d.layers, d.size, d.data_name), (16, 128, 2, 12800, "train-12800"))
        for bad in ("visits-8-128-2", "cq2-16-128-2", "visits-16-64-2", "lab", "nope"):
            with self.assertRaises(ValueError):
                ps.resolve_arm(bad)

    def test_the_sweep_order_is_the_labs_cell_its_seeds_the_single_factor_cells_the_rest_then_the_data_arm(self):
        order = ps.all_order((0, 1))
        self.assertEqual(order[:6], [("visits-16-128-2", 0), ("visits-16-128-2", 1), ("visits-16-128-2", 2), ("visits-16-128-2", 3),
                                     ("lab-bs8", 0), ("lab-bs9", 0)])
        singles = ["cq1-16-128-2", "cq3-16-128-2", "visits-32-128-2", "visits-16-256-2", "visits-16-128-4"]
        self.assertEqual(ps.single_factor_cells(), singles)
        self.assertEqual(order[6:16], [(n, s) for n in singles for s in (0, 1)])
        rest = [n for n in ps.grid_names() if n != ps.LAB_ARM and n not in singles]
        self.assertEqual(order[16:-2], [(n, s) for n in rest for s in (0, 1)])
        self.assertEqual(order[-2:], [("data12800", 0), ("data12800", 1)])
        self.assertEqual(len(order), 54)  # 48 + 4 + 2
        self.assertEqual(len(set(order)), 54)
        self.assertEqual(ps.all_order((0, 1, 2, 3))[:4], [("visits-16-128-2", s) for s in range(4)])  # no seed twice
        self.assertEqual(len(set(ps.all_order((0, 1, 2, 3)))), len(ps.all_order((0, 1, 2, 3))))


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.lab, cls.agent, cls.seeds = build_lab(cls.root)
        cls.out = cls.root / "prior"
        cls.manifest = ps.build(cls.lab, cls.agent, cls.out)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_evaluation_runs_are_the_ones_whose_seed_hash_is_in_the_first_tenth_and_no_training_set_has_a_row_of_theirs(self):
        all_seeds = [s for g in sorted(self.seeds) for s in self.seeds[g]]
        flags = eval_seed_flags(all_seeds)
        n_eval, n_train = sum(flags), len(flags) - sum(flags)
        self.assertGreaterEqual(n_eval, 1)
        self.assertGreaterEqual(n_train, 10)
        ev = ps.load_npz(self.out / "eval.npz")
        self.assertEqual(sorted(ev), sorted([*ps.TOKEN_KEYS, "lengths", "legal", "visits", "qvals", "runs", "fights"]))
        self.assertEqual(np.unique(ev["runs"]).size, n_eval)
        self.assertEqual(self.manifest["eval"]["runs"], n_eval)
        self.assertEqual(self.manifest["eval"]["samples"], 12 * n_eval)
        self.assertEqual(self.manifest["eval"]["fights"], 3 * n_eval)
        self.assertEqual(ev["lengths"].size, 12 * n_eval)
        self.assertEqual(int(ev["lengths"].sum()), ev["types"].size)
        for name, info in self.manifest["sets"].items():
            with self.subTest(set=name):
                train = ps.load_npz(self.out / f"{name}.npz")
                self.assertTrue(set(np.unique(ev["runs"]).tolist()).isdisjoint(np.unique(train["fights"]).tolist()))
                self.assertNotIn("_fight_ix", train)  # the saved sets carry the lab's keys only
                self.assertEqual(info["runs"], n_train)  # the caps are far above this window's fights: every other run is in
                self.assertEqual(info["samples"], 12 * n_train)
                self.assertEqual(info["build_seed"], int(name.split("-bs")[1]) if "-bs" in name else ps.BUILD_SEED)

    def test_disjoint_also_when_the_cap_samples_the_training_fights(self):
        with tempfile.TemporaryDirectory() as scratch:
            mans = gen_loop.window_manifests(self.lab / "runs", 1)
            eval_mans, train_mans = ps.split_window(mans, Path(scratch))
            evaluation = gen_loop.build_training_set(eval_mans, max_fights=10**6, seed=0)
            for cap, seed in ((5, 7), (12, 8)):
                sampled = gen_loop.build_training_set(train_mans, max_fights=cap, seed=seed)
                self.assertEqual(int(sampled["_fights_kept"][0]), cap)
                self.assertTrue(set(np.unique(evaluation["fights"]).tolist()).isdisjoint(np.unique(sampled["fights"]).tolist()))
                fights = sampled["fights"].astype(np.int64) * ps.FIGHT_KEY + sampled["_fight_ix"]
                self.assertEqual(np.unique(fights).size, cap)
                self.assertTrue(set(fights.tolist()).isdisjoint((evaluation["fights"].astype(np.int64) * ps.FIGHT_KEY + evaluation["_fight_ix"]).tolist()))
            full = gen_loop.build_training_set(train_mans, max_fights=10**6, seed=0)
            self.assertEqual(int(full["lengths"].size) + int(evaluation["lengths"].size), 12 * 60)

    def test_the_sets_the_manifest_and_the_cache_key(self):
        self.assertEqual(sorted(self.manifest["sets"]), ["train-12800", "train-6400", "train-6400-bs8", "train-6400-bs9"])
        for name, info in self.manifest["sets"].items():
            self.assertEqual(info["sha256"], ps.sha256_file(self.out / f"{name}.npz"))
            self.assertTrue(set(ps.load_npz(self.out / f"{name}.npz")) >= {"types", "visits", "legal", "qvals", "fights", "lengths"})
        self.assertEqual(self.manifest["generation"], 1)
        self.assertEqual(self.manifest["build_seed"], 7)
        self.assertEqual(self.manifest["eval_range"], [0.0, 0.1])
        self.assertEqual([b["manifest"] for b in self.manifest["blocks"]], ["runs/job0/shards/shards.json", "runs/job1/shards/shards.json"])
        self.assertEqual(len(self.manifest["key"]), 64)
        again = ps.build(self.lab, self.agent, self.out)  # the same window: nothing to build, the same manifest
        self.assertEqual(again["key"], self.manifest["key"])
        self.assertEqual(again["build"], self.manifest["build"])
        self.assertEqual(ps.build_seeds_of(self.out), [8, 9])

    def test_a_changed_window_is_refused_and_an_open_block_is_not_used(self):
        lab, agent, _ = build_lab(self.root / "other", runs_per_job=10)
        (lab / "runs" / "job1" / "done.rc").unlink()  # still open (just written) and unfinished
        out = self.root / "other-prior"
        manifest = ps.build(lab, agent, out)
        self.assertEqual([b["manifest"] for b in manifest["blocks"]], ["runs/job0/shards/shards.json"])
        self.assertEqual(len(manifest["skipped_blocks"]), 1)
        (lab / "runs" / "job1" / "done.rc").write_text("0\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            ps.build(lab, agent, out)

    def test_nothing_written_holds_an_absolute_path(self):
        for path in [self.out / "manifest.json"]:
            assert_no_paths(self, path.read_text(encoding="utf-8"), self.root)
        self.assertFalse(list(self.out.glob("split-*")))  # the scratch split is removed


def tiny_fit_hook(arm_folder: Path, epochs: int = 2, stop: Path | None = None, seen: list | None = None):
    """gen_loop.train_combat_prior as the sweep calls it, cut down to a tiny net and `epochs` epochs; `seen` records what the hook saw."""
    real = gen_loop.train_combat_prior

    def fit(data, out, **kw):
        kw.update(epochs=epochs, d_model=16, layers=1)
        hook = kw["epoch_end"]

        def wrapped(epoch, stats):
            if seen is not None:
                seen.append((epoch, sorted(p.name for p in arm_folder.iterdir()), (Path(out) / "combat_policy.pt").is_file()))
            if stop is not None and epoch == 1:
                stop.write_text("x", encoding="utf-8")
            hook(epoch, stats)

        kw["epoch_end"] = wrapped
        return real(data, out, **kw)

    return fit


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        lab, agent, _ = build_lab(cls.root)
        cls.out = cls.root / "prior"
        ps.build(lab, agent, cls.out)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        for folder in self.out.iterdir():
            if folder.is_dir():
                import shutil

                shutil.rmtree(folder)

    def namespace(self, **kw) -> argparse.Namespace:
        base = dict(out=str(self.out), arms="visits-16-128-2", confirm="", seeds="0", device="cpu", stop_file=None, budget_min=None,
                    devset=None, no_devset=True)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_a_fit_lands_in_a_temporary_folder_that_is_renamed_on_success(self):
        seen: list = []
        arm_folder = self.out / "visits-16-128-2"
        with mock.patch.object(gen_loop, "train_combat_prior", side_effect=tiny_fit_hook(arm_folder, 2, None, seen)), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ps.cmd_run(self.namespace()), 0)
        self.assertEqual([s[0] for s in seen], [1, 2])
        self.assertTrue(all(names == ["tmp-s0"] and has_policy for _, names, has_policy in seen[:1]))  # mid-fit: only the temporary folder
        self.assertEqual(sorted(p.name for p in arm_folder.iterdir()), ["s0"])
        fit = arm_folder / "s0"
        self.assertEqual(sorted(p.name for p in fit.iterdir()), ["combat_policy.pt", "measures.json", "model.json", "work"])
        self.assertEqual(sorted(p.name for p in (fit / "work").iterdir()), ["train_combat.log"])  # the fit's copy of its data is gone
        m = json.loads((fit / "measures.json").read_text(encoding="utf-8"))
        self.assertEqual((m["arm"], m["seed"], m["format"]), ("visits-16-128-2", 0, ps.FIT_FORMAT))
        self.assertEqual({k for k in m["eval"]}, {"rows", "exp_q_regret", "top1_q_regret", "top1_most_visited", "entropy"})
        self.assertGreater(m["eval"]["rows"], 0)
        self.assertGreaterEqual(m["eval"]["exp_q_regret"], 0.0)
        self.assertIsNone(m["devset"])
        self.assertEqual(len(m["curves"]), 2)
        self.assertIn("policy_val", m["validation"])
        recipe = m["recipe"]
        self.assertEqual((recipe["target"], recipe["epochs"], recipe["d_model"], recipe["layers"], recipe["threads"]), ("visits", 16, 128, 2, 2))
        self.assertEqual((recipe["build_seed"], recipe["torch_seed"], recipe["split_seed"], recipe["max_fights"]), (7, 0, 0, 6400))
        self.assertEqual(sorted(recipe["seed_rules"]), ["build_seed", "torch_seed"])
        assert_no_paths(self, (fit / "measures.json").read_text(encoding="utf-8"), self.root)
        model_json = json.loads((fit / "model.json").read_text(encoding="utf-8"))
        self.assertEqual(model_json["iteration"], 2)  # the build's generation + 1
        with mock.patch.object(gen_loop, "train_combat_prior") as again, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ps.cmd_run(self.namespace()), 0)  # a fit with measures.json is skipped
            again.assert_not_called()

    def test_the_stop_file_between_epochs_drops_the_fit_and_leaves_nothing_behind(self):
        stop = self.root / "quiet.stop"
        stop.unlink(missing_ok=True)
        seen: list = []
        arm_folder = self.out / "cq3-16-128-2"
        self.out.joinpath("cq3-16-128-2").mkdir()
        with mock.patch.object(gen_loop, "train_combat_prior", side_effect=tiny_fit_hook(arm_folder, 3, stop, seen)), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            code = ps.cmd_run(self.namespace(arms="cq3-16-128-2", stop_file=str(stop)))
        self.assertEqual(code, ps.STOP_EXIT)
        self.assertEqual(code, 3)
        self.assertEqual([s[0] for s in seen], [1])  # stopped at the end of epoch 1 of 3; the checkpoint of that epoch had been saved
        self.assertTrue(seen[0][2])
        self.assertEqual(list(arm_folder.iterdir()), [])  # the temporary folder, checkpoint and all, is gone; no s0 either
        self.assertIn("stop file seen", printed.getvalue())
        stop.unlink()
        with mock.patch.object(gen_loop, "train_combat_prior") as fit, contextlib.redirect_stdout(io.StringIO()):
            stop.write_text("x", encoding="utf-8")
            self.assertEqual(ps.cmd_run(self.namespace(arms="cq3-16-128-2", stop_file=str(stop))), 3)  # present at a fit boundary
            fit.assert_not_called()

    def test_a_leftover_temporary_folder_is_replaced_and_an_error_removes_its_folder(self):
        arm_folder = self.out / "visits-16-128-2"
        (arm_folder / "tmp-s0").mkdir(parents=True)
        (arm_folder / "tmp-s0" / "junk.pt").write_text("x", encoding="utf-8")
        with mock.patch.object(gen_loop, "train_combat_prior", side_effect=RuntimeError("boom")), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                ps.cmd_run(self.namespace())
        self.assertEqual(list(arm_folder.iterdir()), [])

    def test_the_budget_stops_between_fits_and_confirm_fits_go_to_their_own_folder(self):
        with mock.patch.object(gen_loop, "train_combat_prior") as fit, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ps.cmd_run(self.namespace(arms="all", seeds="0,1", budget_min=0.0)), 0)
            fit.assert_not_called()
        arm_folder = self.out / "visits-16-128-2"
        with mock.patch.object(gen_loop, "train_combat_prior", side_effect=tiny_fit_hook(arm_folder, 1)), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ps.cmd_run(self.namespace(arms="", confirm="visits-16-128-2", seeds="10,11")), 0)
        self.assertEqual(sorted(p.name for p in arm_folder.iterdir()), ["confirm-s10", "confirm-s11"])
        fits, confirms = ps.load_fits(self.out)
        self.assertEqual(fits, {})
        self.assertEqual(sorted(confirms["visits-16-128-2"]), [10, 11])
        self.assertTrue(confirms["visits-16-128-2"][10]["confirm"])

    def test_measure_3_needs_the_emulator_library_and_a_missing_one_stops_the_run_before_any_fit(self):
        empty = self.root / "no-lib"
        empty.mkdir(exist_ok=True)
        with mock.patch.dict("os.environ", {"STS2_LIB_PATH": str(empty)}):
            self.assertIn("STS2_LIB_PATH", ps.native_problem())
            with mock.patch.object(gen_loop, "train_combat_prior") as fit, contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(ps.cmd_run(self.namespace(no_devset=False, devset=str(self.root / "devset"))), 2)
            fit.assert_not_called()
            self.assertIn("STS2_LIB_PATH", err.getvalue())
            (empty / "Sts2Emulator.dll").write_bytes(b"")
            self.assertIsNone(ps.native_problem())

    def test_the_search_refuses_while_a_quiet_lock_exists(self):
        lock = self.root / "QUIET.lock"
        lock.write_text("w4-x 2026-10-28T00:00:00Z\n", encoding="utf-8")
        with mock.patch("sys.stderr"), mock.patch("sts2_gym.procutil.lower_priority"):  # main lowers the process it runs in
            code = ps.main(["search", "--out", str(self.out), "--arms", "visits-16-128-2", "--quiet-lock", str(lock)])
        self.assertEqual(code, 2)
        lock.unlink()


class MeasureTests(unittest.TestCase):
    def test_completed_q_and_the_regrets_on_a_hand_checked_decision(self):
        nan = np.nan
        visits = np.array([[5, 3, 0, 0], [0, 0, 0, 0], [2, 2, 0, 9]], np.float32)
        legal = np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0]], np.int8)
        q = np.array([[4, 10, nan, nan], [nan] * 4, [-3, 1, nan, 99]], np.float32)
        cq, keep = ps.completed_q(visits, legal, q)
        root = (5 * 4 + 3 * 10) / 8
        np.testing.assert_allclose(cq[0], [4, 10, root, -np.inf])
        self.assertEqual(keep.tolist(), [True, False, True])  # the middle row has nothing visited: no signal
        probs = np.array([[0.2, 0.5, 0.3, 0.0], [0.5, 0.5, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]])
        m = ps.regret_measures(probs, legal, visits, q)
        want_exp = [10 - (0.2 * 4 + 0.5 * 10 + 0.3 * root), 1 - (0.9 * -3 + 0.1 * 1)]
        self.assertEqual(m["rows"], 2)
        self.assertAlmostEqual(m["exp_q_regret"], np.mean(want_exp), places=9)
        self.assertAlmostEqual(m["top1_q_regret"], np.mean([0.0, 4.0]), places=9)  # row 0 picks the best; row 2 picks -3 where 1 was there
        self.assertAlmostEqual(m["top1_most_visited"], 0.5)  # row 0: argmax 1 against most visited 0; row 2: argmax 0 and most visited 0 (a tie goes to the first)
        entropy = lambda p: -sum(x * np.log(x) for x in p if x > 0)  # noqa: E731
        self.assertAlmostEqual(m["entropy"], np.mean([entropy([0.2, 0.5, 0.3]), entropy([0.9, 0.1])]), places=9)

    def test_it_agrees_with_the_trainers_completed_q_target(self):
        rng = np.random.default_rng(3)
        legal = (rng.random((200, 12)) < 0.5).astype(np.int8)
        visits = (rng.integers(0, 6, (200, 12)) * legal).astype(np.float32)
        q = np.where(visits > 0, rng.normal(0, 8, (200, 12)), np.nan).astype(np.float32)
        cq, keep = ps.completed_q(visits, legal, q)
        for t in (1.0, 3.0):
            want = train_combat.completed_q_target(visits, legal, q, t)
            z = np.where(keep[:, None], cq / t, 0.0)
            z = np.where(np.isfinite(z), z, -np.inf)
            got = np.exp(z - np.where(np.isfinite(z).any(1, keepdims=True), z.max(1, keepdims=True), 0))
            got = np.where(keep[:, None], got / np.maximum(got.sum(1, keepdims=True), 1e-300), 0.0)
            np.testing.assert_allclose(got, want, atol=1e-6)
        self.assertEqual(keep.tolist(), (train_combat.completed_q_target(visits, legal, q, 1.0).sum(1) > 0).tolist())

    def test_sharpness_lowers_expected_regret_but_not_top1_regret_which_is_why_measure_1_is_read_within_a_target_level(self):
        rng = np.random.default_rng(4)
        legal = np.ones((300, 6), np.int8)
        visits = rng.integers(1, 9, (300, 6)).astype(np.float32)
        q = rng.normal(0, 10, (300, 6)).astype(np.float32)
        cq, _ = ps.completed_q(visits, legal, q)

        def softmax(t):
            z = cq / t
            e = np.exp(z - z.max(1, keepdims=True))
            return e / e.sum(1, keepdims=True)

        exp = [ps.regret_measures(softmax(t), legal, visits, q)["exp_q_regret"] for t in (0.25, 1.0, 3.0, 10.0, 100.0)]
        self.assertTrue(all(a < b for a, b in zip(exp, exp[1:], strict=False)), exp)  # the sharper the policy, the lower the expected regret
        top = {ps.regret_measures(softmax(t), legal, visits, q)["top1_q_regret"] for t in (0.25, 1.0, 3.0, 10.0, 100.0)}
        self.assertEqual(top, {0.0})  # every temperature has the same argmax: measure 2 does not reward sharpness
        visit_policy = visits / visits.sum(1, keepdims=True)  # a spread visit distribution
        m = ps.regret_measures(visit_policy, legal, visits, q)
        self.assertGreater(m["exp_q_regret"], exp[0])  # the sharpness bias: a visit-trained net loses to a sharp completed-Q one on measure 1
        self.assertGreater(m["top1_q_regret"], 0.0)

    def test_all_rows_unvisited_is_an_error(self):
        with self.assertRaises(ValueError):
            ps.regret_measures(np.full((1, 3), 1 / 3), np.ones((1, 3), np.int8), np.zeros((1, 3), np.float32), np.full((1, 3), np.nan, np.float32))


def cells_with(effects: dict[str, float], base: float = 10.0, delta: float = 0.05, lab_extra: bool = False):
    """The 8 cells of (epochs, d_model, layers) with y = base + sum(effect * code) plus the pair-wise terms in `effects`; two seeds at +/-delta."""
    cells = []
    for e, d, n in [(e, d, n) for e in (16, 32) for d in (128, 256) for n in (2, 4)]:
        code = {"epochs": 0.5 if e == 32 else -0.5, "d_model": 0.5 if d == 256 else -0.5, "layers": 0.5 if n == 4 else -0.5}
        y = base + sum(effects.get(k, 0.0) * v for k, v in code.items()) + effects.get("epochs x layers", 0.0) * code["epochs"] * code["layers"]
        values = [y - delta, y + delta] + ([y - delta, y + delta] if lab_extra and (e, d, n) == (16, 128, 2) else [])
        cells.append(({"epochs": e, "d_model": d, "layers": n}, values))
    return cells


class EffectsTests(unittest.TestCase):
    def test_a_known_effect_and_residual_are_recovered_with_their_standard_errors(self):
        model = ps.effect_model(cells_with({"epochs": 2.0, "layers": -1.0, "epochs x layers": 0.8}), ["epochs", "d_model", "layers"])
        self.assertTrue(model["estimable"] and model["interactions"])
        self.assertEqual((model["cells"], model["residual_df"]), (8, 8))
        self.assertAlmostEqual(model["residual_variance"], 2 * 0.05**2, places=12)  # two seeds at +/-delta
        by = {e["name"]: e for e in model["effects"]}
        self.assertAlmostEqual(by["epochs:32"]["effect"], 2.0, places=9)
        self.assertAlmostEqual(by["layers:4"]["effect"], -1.0, places=9)
        self.assertAlmostEqual(by["d_model:256"]["effect"], 0.0, places=9)
        self.assertAlmostEqual(by["epochs:32 x layers:4"]["effect"], 0.8, places=9)
        self.assertAlmostEqual(by["epochs:32 x d_model:256"]["effect"], 0.0, places=9)
        sigma = np.sqrt(2 * 0.05**2)
        self.assertAlmostEqual(by["epochs:32"]["se"], sigma / 2, places=9)  # sqrt(var / (n * sum(code^2))) = sigma / sqrt(2 * 8 * 0.25)
        self.assertAlmostEqual(by["epochs:32 x layers:4"]["se"], sigma, places=9)  # product codes are +/-0.25: sum of squares 0.5, times n = 1
        self.assertEqual({n for n, e in by.items() if e["exceeds"]}, {"epochs:32", "layers:4", "epochs:32 x layers:4"})

    def test_an_effect_below_twice_its_error_is_not_flagged(self):
        sigma = np.sqrt(2 * 0.05**2)
        model = ps.effect_model(cells_with({"d_model": 1.5 * sigma / 2 * 1.0, "epochs": 2.5 * sigma / 2}), ["epochs", "d_model", "layers"])
        by = {e["name"]: e for e in model["effects"]}
        self.assertFalse(by["d_model:256"]["exceeds"])  # 1.5 SE
        self.assertTrue(by["epochs:32"]["exceeds"])  # 2.5 SE

    def test_extra_seeds_of_one_cell_join_the_pooled_variance_and_the_weights(self):
        model = ps.effect_model(cells_with({"epochs": 2.0}, lab_extra=True), ["epochs", "d_model", "layers"])
        self.assertEqual(model["residual_df"], 8 + 2)
        self.assertAlmostEqual(model["residual_variance"], (7 * 2 + 4) * 0.05**2 / 10, places=12)  # a pair has ss 2 delta^2, the lab's four have 4 delta^2
        by = {e["name"]: e for e in model["effects"]}
        self.assertAlmostEqual(by["epochs:32"]["effect"], 2.0, places=9)  # the cell means still give the effect exactly
        v, df = ps.pooled_variance([[1.0, 3.0], [2.0, 2.0, 5.0], [7.0]])
        self.assertEqual(df, 1 + 2 + 0)
        self.assertAlmostEqual(v, (2.0 + (1 + 1 + 4)) / 3)  # sums of squares about each mean: 2, then (2-3)^2*2 + (5-3)^2 = 6

    def test_a_three_level_target_gives_each_level_minus_the_first_and_interactions_need_spare_cells(self):
        cells = []
        for t in ("visits", "cq1", "cq3"):
            for e in (16, 32):
                for d in (128, 256):
                    for n in (2, 4):
                        y = 5.0 + {"visits": 0.0, "cq1": -1.0, "cq3": -3.0}[t] + (0.5 if e == 32 else 0.0)
                        cells.append(({"target": t, "epochs": e, "d_model": d, "layers": n}, [y - 0.1, y + 0.1]))
        model = ps.effect_model(cells, ["target", "epochs", "d_model", "layers"])
        self.assertTrue(model["interactions"])
        by = {e["name"]: e for e in model["effects"]}
        self.assertAlmostEqual(by["target:cq1"]["effect"], -1.0, places=9)
        self.assertAlmostEqual(by["target:cq3"]["effect"], -3.0, places=9)
        self.assertAlmostEqual(by["epochs:32"]["effect"], 0.5, places=9)
        self.assertTrue(by["target:cq3"]["exceeds"] and by["epochs:32"]["exceeds"] and not by["layers:4"]["exceeds"])
        three = [({"target": "visits", "epochs": 16, "d_model": 128, "layers": 2}, [1.0, 1.2]),
                 ({"target": "cq1", "epochs": 16, "d_model": 256, "layers": 2}, [1.0, 1.2]),
                 ({"target": "cq1", "epochs": 16, "d_model": 128, "layers": 4}, [1.0, 1.2])]
        self.assertFalse(ps.effect_model(three, ["target", "epochs", "d_model", "layers"])["estimable"])  # four parameters from three cells
        main_only = ps.effect_model(cells_with({"epochs": 1.0})[:7], ["epochs", "d_model", "layers"])  # seven cells: 7 parameters leave no spare cell
        self.assertFalse(main_only["interactions"])
        self.assertTrue(main_only["estimable"])

    def test_welch_interval_and_t_table(self):
        w = ps.welch_interval([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        self.assertAlmostEqual(w["diff"], -3.0)
        self.assertAlmostEqual(w["df"], 4.0)
        self.assertAlmostEqual(w["hi"], -3.0 + 2.776 * np.sqrt(2 / 3), places=6)
        self.assertTrue(w["excludes_zero"])
        self.assertFalse(ps.welch_interval([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])["excludes_zero"])
        self.assertEqual((ps.t_crit(4), ps.t_crit(4.9), ps.t_crit(11), ps.t_crit(500)), (2.776, 2.776, 2.228, 1.98))


def fake_fit(arm: str, seed: int, exp: float, top1: float, *, dev: tuple[float, float] | None = None, positions: dict | None = None,
             confirm: bool = False) -> dict:
    devset = None
    if dev is not None:
        devset = {"positions": 3, "exp_regret": dev[0], "top1_regret": dev[1], "agree": 0.5, "per_position": positions or {}}
    return {"format": ps.FIT_FORMAT, "arm": arm, "seed": seed, "confirm": confirm, "recipe": {}, "fit_min": 3.0, "gpu_peak_mb": 100.0,
            "eval": {"rows": 10, "exp_q_regret": exp, "top1_q_regret": top1, "top1_most_visited": 0.5, "entropy": 1.0},
            "devset": devset, "validation": {"policy_val": 0.9, "value_val": 3.0}}


def write_fits(out: Path, fits: list[dict]) -> None:
    for m in fits:
        ps.save_json(ps.fit_dir(out, m["arm"], m["seed"], m["confirm"]) / "measures.json", m)


def synthetic_sweep(out: Path, *, epochs_effect: float = 0.0, data_gain: float = 0.0, noise: float = 0.02) -> None:
    fits = []
    for arm in ps.grid_names():
        c = ps.split_arm(arm)
        level = {"visits": 0.0, "cq1": -0.2, "cq3": -0.5}[c["target"]]
        base_exp = 4.0 + {"visits": 0.0, "cq1": -1.0, "cq3": -2.0}[c["target"]]  # sharper targets read lower on measure 1 by construction
        for seed in range(2):
            sign = -1.0 if seed == 0 else 1.0
            gain = epochs_effect if c["epochs"] == 32 else 0.0
            fits.append(fake_fit(arm, seed, base_exp - gain + sign * noise, 3.0 + level - gain + sign * noise))
    for seed in (2, 3):
        fits.append(fake_fit(ps.LAB_ARM, seed, 4.0 + (-1 if seed == 2 else 1) * 0.03, 3.0 + (-1 if seed == 2 else 1) * 0.03))
    for n, bump in ((8, 0.05), (9, -0.04)):
        fits.append(fake_fit(f"lab-bs{n}", 0, 4.0 + bump, 3.0 + bump))
    for seed in range(2):
        fits.append(fake_fit(ps.DATA_ARM, seed, 4.0 - data_gain + (-1 if seed == 0 else 1) * noise, 3.0 - data_gain))
    write_fits(out, fits)


class ReportTests(unittest.TestCase):
    def test_the_chosen_cell_follows_the_main_effects_and_the_lab_cell_when_nothing_moves(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            synthetic_sweep(out, epochs_effect=0.0)
            fits, _ = ps.load_fits(out)
            effects = ps.main_effects(fits)
            chosen = ps.choose_cell(effects)
            self.assertEqual(chosen["cell"], "cq3-16-128-2")  # only the target moves the prior; the factors that do nothing stay at the lab's level
            self.assertEqual(sorted(chosen["moves"]), ["target:cq1", "target:cq3"])
            self.assertEqual(ps.best_target_cell(fits), "cq3-16-128-2")
            ps.save_json(out.joinpath("x").with_suffix(".json"), effects)  # NaN-free JSON
            ps.save_json(out / "pad.json", {"nan": float("nan")})
            self.assertIsNone(json.loads((out / "pad.json").read_text(encoding="utf-8"))["nan"])

    def test_an_epochs_effect_moves_the_chosen_cell_and_measure_1_within_a_target_level_has_its_own_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            synthetic_sweep(out, epochs_effect=0.5)
            fits, _ = ps.load_fits(out)
            effects = ps.main_effects(fits)
            self.assertEqual(ps.choose_cell(effects)["cell"], "cq3-32-128-2")
            for target in ps.TARGETS:
                within = effects["exp_q_regret"][target]
                self.assertTrue(within["estimable"])
                self.assertEqual(within["cells"], 8)
                e = ps.effect_of(within, "epochs:32")
                self.assertAlmostEqual(e["effect"], -0.5, places=9)
                self.assertTrue(e["exceeds"])
            self.assertAlmostEqual(ps.effect_of(effects["top1_q_regret"], "epochs:32")["effect"], -0.5, places=9)
            self.assertEqual(effects["top1_q_regret"]["residual_df"], 24 + 2)  # 24 cells x 1 df, plus the lab's cell's two extra seeds

    def test_a_within_target_conflict_keeps_the_labs_level(self):
        effects = {"top1_q_regret": {"estimable": True, "effects": [{"name": "epochs:32", "effect": -0.4, "se": 0.1, "exceeds": True}]},
                   "exp_q_regret": {"visits": {"estimable": True, "effects": [{"name": "epochs:32", "effect": 0.3, "se": 0.05, "exceeds": True}]}}}
        chosen = ps.choose_cell(effects)
        self.assertEqual(chosen["cell"], ps.LAB_ARM)
        self.assertEqual(len(chosen["notes"]), 1)

    def test_seed_spreads_and_the_data_arm_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            synthetic_sweep(out, data_gain=0.5)
            fits, _ = ps.load_fits(out)
            spread = ps.spreads(fits)["exp_q_regret"]
            self.assertEqual(spread["torch_seeds"]["n"], 4)  # the lab's cell: seeds 0..3
            self.assertEqual(spread["build_seeds"]["n"], 3)  # torch seed 0 at build seeds 7, 8, 9
            self.assertAlmostEqual(spread["build_seeds"]["min"], 3.96)  # build seed 9: 4.0 - 0.04
            self.assertAlmostEqual(spread["build_seeds"]["max"], 4.05)
            reading = ps.data_reading(fits)["exp_q_regret"]
            self.assertAlmostEqual(reading["vs_visits-32-128-2"]["diff"], -0.5, places=9)  # data at equal steps: 0.5 HP better than 32 epochs
            self.assertTrue(reading["vs_visits-32-128-2"]["exceeds"])
            flat = tempfile.TemporaryDirectory()
            synthetic_sweep(Path(flat.name), data_gain=0.0)
            none = ps.data_reading(ps.load_fits(Path(flat.name))[0])["exp_q_regret"]["vs_visits-32-128-2"]
            self.assertFalse(none["exceeds"])
            flat.cleanup()

    def test_the_confirmation_reads_the_six_fresh_fits_only(self):
        positions = lambda shift: {f"p{i}": [1.0, 2.0 + shift + 0.01 * i, 3.0 + shift] for i in range(5)}  # noqa: E731
        chosen = "cq3-32-128-2"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertEqual(ps.confirmation({}, ps.LAB_ARM)["status"], "the chosen cell is the lab's cell; nothing to confirm")
            fits = [fake_fit(ps.LAB_ARM, s, 4.0 + d, 3.0 + d, dev=(2.0, 3.0), positions=positions(0.0), confirm=True) for s, d in zip((10, 11, 12), (-0.02, 0, 0.02), strict=True)]
            self.assertEqual(ps.confirmation({ps.LAB_ARM: {m["seed"]: m for m in fits}}, chosen)["status"], "waiting for confirm fits")
            new = [fake_fit(chosen, s, 2.0 + d, 2.5 + d, dev=(2.0, 3.0), positions=positions(-0.1), confirm=True) for s, d in zip((10, 11, 12), (-0.02, 0, 0.02), strict=True)]
            confirms = {ps.LAB_ARM: {m["seed"]: m for m in fits}, chosen: {m["seed"]: m for m in new}}
            r = ps.confirmation(confirms, chosen)
            self.assertEqual(r["status"], "pass")
            self.assertFalse(r["same_target"])
            self.assertIsNone(r["checks"]["exp_q_regret_better_within_target"])  # not comparable across targets: not required
            self.assertTrue(r["checks"]["top1_q_regret_better"] and r["checks"]["dev_not_worse"])
            worse = [fake_fit(chosen, s, 2.0, 2.5, dev=(2.0, 3.0), positions=positions(+0.5), confirm=True) for s in (10, 11, 12)]
            r = ps.confirmation({ps.LAB_ARM: confirms[ps.LAB_ARM], chosen: {m["seed"]: m for m in worse}}, chosen)
            self.assertEqual(r["status"], "fail")  # better top-1 regret but the net alone is worse on the dev set beyond the paired interval
            self.assertFalse(r["checks"]["dev_not_worse"])
            same = [fake_fit(ps.LAB_ARM, s + 5, 4.0, 3.0, confirm=True) for s in (10, 11, 12)]
            noisy = [fake_fit(chosen, s, 4.0 + d, 3.0 + d, confirm=True) for s, d in zip((10, 11, 12), (-0.5, 0.0, 0.5), strict=True)]
            r = ps.confirmation({ps.LAB_ARM: {m["seed"]: m for m in same}, chosen: {m["seed"]: m for m in noisy}}, chosen)
            self.assertEqual(r["status"], "fail")  # an interval that contains 0 does not pass
            write_fits(out, fits + new)
            _, from_disk = ps.load_fits(out)
            self.assertEqual(sorted(from_disk), sorted([ps.LAB_ARM, chosen]))

    def test_the_within_target_condition_applies_when_the_chosen_cell_keeps_the_labs_target(self):
        chosen = "visits-32-128-2"
        lab = {s: fake_fit(ps.LAB_ARM, s, 4.0 + d, 3.0 + d, confirm=True) for s, d in zip((10, 11, 12), (-0.02, 0, 0.02), strict=True)}
        new = {s: fake_fit(chosen, s, 4.0 + d, 2.0 + d, confirm=True) for s, d in zip((10, 11, 12), (-0.02, 0, 0.02), strict=True)}
        r = ps.confirmation({ps.LAB_ARM: lab, chosen: new}, chosen)
        self.assertTrue(r["same_target"])
        self.assertFalse(r["checks"]["exp_q_regret_better_within_target"])  # same expected regret: not better
        self.assertEqual(r["status"], "fail")

    def test_the_report_is_written_as_markdown_and_json_with_numbers_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            synthetic_sweep(out, epochs_effect=0.5, data_gain=0.3)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ps.cmd_table(argparse.Namespace(out=str(out))), 0)
            text = (out / "report.md").read_text(encoding="utf-8")
            for needle in ("# Combat prior recipe sweep", "## Cells", "### Measure 2", "### Measure 1 (expected regret) within target cq3", "## Chosen cell: cq3-32-128-2",
                           "## Seed spreads", "## Data arm", "## Confirmation", "| visits-16-128-2 | 4 |", "| data12800 | 2 |", "| lab-bs8 | 1 |"):
                self.assertIn(needle, text)
            assert_no_paths(self, text, out)
            doc = json.loads((out / "table.json").read_text(encoding="utf-8"))
            assert_no_paths(self, json.dumps(doc), out)
            self.assertEqual(doc["chosen"]["cell"], "cq3-32-128-2")
            self.assertEqual(doc["fits"], 24 * 2 + 2 + 2 + 2)


class QueueTests(unittest.TestCase):
    def test_the_sweep_is_queued_in_order_with_tool_and_cpu_procs_and_the_search_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "prior"
            jobs = ps.queue_jobs(out)
            self.assertEqual(len(jobs), 55)  # 54 fits and the search
            self.assertEqual([j["name"] for j in jobs[:6]], ["prior-visits-16-128-2-s0", "prior-visits-16-128-2-s1", "prior-visits-16-128-2-s2",
                                                           "prior-visits-16-128-2-s3", "prior-lab-bs8-s0", "prior-lab-bs9-s0"])
            self.assertEqual(jobs[-3]["name"], "prior-data12800-s0")
            self.assertEqual(jobs[-1]["name"], "prior-search")
            self.assertEqual(jobs[0]["args"], ["run", "--out", str(out), "--arms", "visits-16-128-2", "--seeds", "0"])
            self.assertEqual(jobs[-1]["args"][:2], ["search", "--out"])
            self.assertEqual(jobs[-1]["args"][3:], ["--arms", "auto", "--procs", "8", "--reps", "24"])
            for job in jobs:
                self.assertEqual(job["tool"], "prior_sweep")
                self.assertEqual((job["state"], job["attempts"]), ("queued", 0))
                self.assertEqual(job["cpu_procs"], 8 if job["name"] == "prior-search" else 2)
            self.assertEqual(len({j["name"] for j in jobs}), 55)
            write_fits(out, [fake_fit(ps.LAB_ARM, 0, 4.0, 3.0)])  # a fit that exists is not queued again
            self.assertEqual(len(ps.queue_jobs(out)), 54)
            self.assertNotIn("prior-visits-16-128-2-s0", [j["name"] for j in ps.queue_jobs(out)])

    def test_the_queue_command_appends_without_touching_what_is_there_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, queue = Path(tmp) / "prior", Path(tmp) / "lane" / "queue.json"
            queue.parent.mkdir()
            existing = {"name": "vrun-lab-s0", "args": ["--arms", "lab"], "state": "done", "attempts": 0}
            qr.save_queue(queue, [existing])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ps.main(["queue", "--out", str(out), "--queue", str(queue), "--repo", str(ROOT)]), 0)
                first = qr.load_queue(queue)
                self.assertEqual(ps.main(["queue", "--out", str(out), "--queue", str(queue), "--repo", str(ROOT)]), 0)
            self.assertEqual(first[0], existing)
            self.assertEqual(len(first), 56)
            self.assertEqual(qr.load_queue(queue), first)
            self.assertEqual(first[-1]["cpu_procs"], 8)
            self.assertEqual(qr.next_job(first), 1)  # the done job is passed; the lab's cell is next
            self.assertEqual(qr.next_job(first, wide=False), 1)

    def test_the_runner_builds_the_prior_sweep_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "rl").mkdir(parents=True)
            args = argparse.Namespace(lab=tmp, referee_dir=tmp, repo=str(repo), gpu_python="py", queue=str(Path(tmp) / "q.json"),
                                      log=str(Path(tmp) / "l.log"), cache="C", idle_s=1, busy_s=1)
            runner = qr.QuietRunner(args)
            job = ps.queue_jobs(Path(tmp) / "prior")[0]
            cmd = runner.command(job)
            self.assertEqual(cmd[:3], ["py", str(repo / "rl" / "prior_sweep.py"), "run"])
            self.assertEqual(cmd[-4:-1], ["--device", "cuda", "--stop-file"])
            self.assertNotIn("--cache", cmd)  # the sweep cache is vrun_sweep's


if __name__ == "__main__":
    unittest.main()
