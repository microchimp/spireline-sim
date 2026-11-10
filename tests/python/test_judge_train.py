"""The judge trained on the corpus (rl/judge_train.py; D58 C2, D57 item 1).

Pinned: the reader turns a synthetic store into the trainer's columns (the teacher's and the human's outcome with their masks, the search's
root value, death share and tier, the policy target from the visits, the split by run, the cells and sources) and refuses a corpus the manifest
does not call usable; the weights (pool, cell, none, balanced; the clip and the renormalisation); the loss (the outcome term is `judge.judge_loss`
with unit weights, a row without a label adds nothing, the value term is the search's own `judge_client.estimate`, the policy term is the cross
entropy against the visits); the batch the data hands the network; a short training run on a tiny network (the loss falls, the checkpoint is a
judge `judge.load` reads, the Platt scaling is fitted); and the held-out scoring per cell and source with two judges side by side.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402

import judge as jd  # noqa: E402
import judge_client as jcl  # noqa: E402
import judge_corpus as jc  # noqa: E402
import judge_train as jt  # noqa: E402
from combat_tokens import VOCAB  # noqa: E402


def runs_outside_the_exam(n: int, prefix: str = "run") -> list[str]:
    import human_starts as hs

    return [r for r in (f"{prefix}{i}" for i in range(20 * n)) if hs.exam_bucket(r) != 0][:n]


def toks(hand: int, rng: np.random.Generator, potions: int = 1) -> dict:
    """A position the network can read: the global token, `hand` hand cards, end turn, potions, an enemy and its intent (action i is token 1 + i)."""
    types = [1] + [2] * hand + [10] + [8] * potions + [6, 15, 14]
    n = len(types)
    ids = rng.integers(0, 5, n)  # below the smallest id space (6 intents)
    ids[0] = rng.integers(0, 10)
    return {"types": np.array(types, np.uint8), "ids": ids.astype(np.int32), "owners": rng.integers(0, 3, n).astype(np.uint8),
            "feats": rng.random((n, 18)).astype(np.float32)}


def child(a: int, n: int, v: float | None, kind: str = "card", pd: float | None = 0.1) -> dict:
    return {"a": a, "t": -1, "kind": kind, "card": 7, "n": n, "v": v, "se": 1.0, "pd": pd}


def mkrow(i: int, run: str, src: str, kind: str, act: int, tier: str, won: bool, root: float | None, hand: int = 3, **extra) -> dict:
    ch = [child(0, 30, root, pd=0.2), child(hand, 10, root, kind="end_turn", pd=0.4), child(1, 20, root)]
    human = src == "human"
    out = {"pid": f"{run}|f{act}#{i}", "src": src, "sub": "human-start" if human else "night", "start": f"{src}:{run}|f{act}{kind[0]}",
           "run": run, "split": jc.split_of(run), "kind": kind, "act": act, "enemy": "x", "floor": 5, "k": i, "turn": 1 + i, "stage": "early",
           "stage_n": 3, "decisions": 9, "door": i == 0, "hp": 40, "max_hp": 80, "door_max_hp": 80, "hp_frac": 0.5, "potions": 1, "relics": [3],
           "deck": {"1": 3}, "deck_n": 3, "arch": "mixed", "w": 2.0 if human else 1.0, "w_cell": 1.0,
           "label": {"tier": tier, "rel": 2, "sims": 480, "root_value": root, "best_value": root, "p_death": 0.2 if (root is not None and human) else None,
                     "children": ch},
           "teacher": {"won": won, "end_hp": 20 if won else 0, "max_hp": 80, "potions_held": 1 if won else 0, "L0": 0.6 if won else 0.0, "decisions": 9,
                       "turns": 3},
           "human": ({"survived": won, "end_hp": 30 if won else 0, "hp_lost": 50, "turns": 7, "potions_used": 1, "potions_held_end": 1,
                      "l0": 40.0, "l0_unit": 0.7 if won else 0.0} if human else None)}
    out.update(extra)
    return out


class Store:
    """A small synthetic corpus: human boss and elite rows (settled, soft, played), self-play boss and hallway rows."""

    def __init__(self, d: str, usable: bool = True, n_runs: int = 30):
        self.root = Path(d)
        rng = np.random.default_rng(0)
        self.runs = runs_outside_the_exam(n_runs)
        hb, he, sb, sh = [], [], [], []
        for ri, r in enumerate(self.runs):
            for i in range(4):
                won = (ri + i) % 3 != 0
                hb.append(mkrow(i, r, "human", "boss", 2, ["settled", "soft", "played", "settled"][i], won, 20.0 if won else -60.0, hand=3 + i % 2))
                he.append(mkrow(i, r, "human", "elite", 1, "settled", won, 25.0 if won else -80.0))
                sb.append(mkrow(i, f"sp{ri}", "selfplay", "boss", 1, "played", won, 30.0 if won else (-100.0 if i % 2 else None), human=None))
                sh.append(mkrow(i, f"sp{ri}", "selfplay", "hallway", 2, "played", True, 40.0, human=None))
        for stem, rows in (("human-boss-act2-0000", hb), ("human-elite-act1-0000", he), ("selfplay-boss-act1-0000", sb), ("selfplay-hallway-act2-0000", sh)):
            jc.write_shard(self.root / "shards" / stem, rows, [toks(3 + (r["k"] % 2), rng) for r in rows])
        (self.root / "manifest.json").write_text(json.dumps({"schema": "judge-corpus-v1", "usable_for_training": usable, "complete": False}), encoding="utf-8")


def is_human(a: dict) -> np.ndarray:
    return np.array([str(a["tags"][t]).startswith("human") for t in a["tag"]])


class ReaderTests(unittest.TestCase):
    def test_the_reader_makes_the_trainers_columns(self):
        with tempfile.TemporaryDirectory() as d:
            Store(d)
            a, rep = jt.read_corpus(d)
        n = rep["rows"]
        self.assertEqual(n, 30 * 16)
        self.assertEqual(int(a["lens"].sum()), len(a["types"]))
        self.assertEqual(a["feats"].shape, (len(a["types"]), 18))
        self.assertEqual(a["pol"].shape, (n, 32))
        self.assertEqual(rep["human_rows"], 30 * 8)
        self.assertEqual(sorted(a["tags"]), ["human:human-start", "selfplay:night"])
        self.assertEqual(set(a["split"].tolist()), {0, 1, 2})
        run_of_row = a["start_run"][a["group"]]
        held = {r for r, g in zip(run_of_row, a["split"], strict=True) if g == 2}
        self.assertTrue(all(jc.split_of(r) == "heldout" for r in held))
        self.assertTrue(all(jc.split_of(r) == "train" for r, g in zip(run_of_row, a["split"], strict=True) if g in (0, 1)))
        per_run: dict = {}
        for r, g in zip(run_of_row, a["split"], strict=True):
            per_run.setdefault(r, set()).add(int(g))
        self.assertTrue(all(len(s) == 1 for s in per_run.values()))  # one run's rows are all in one split

    def test_labels_masks_and_the_search_columns(self):
        with tempfile.TemporaryDirectory() as d:
            Store(d)
            a, _ = jt.read_corpus(d)
        self.assertTrue(np.array_equal(a["has_human"], is_human(a)))
        self.assertTrue(a["has_teacher"].all())
        i = int(np.flatnonzero(a["has_human"] & (a["y_human"][:, 0] > 0.5))[0])
        np.testing.assert_allclose(a["y_human"][i], [1.0, 30 / 80, 1.0, 0.7], rtol=1e-6)
        np.testing.assert_allclose(a["y_teacher"][i], [1.0, 20 / 80, 1.0, 0.6], rtol=1e-6)
        self.assertEqual(set(np.unique(a["tier"]).tolist()), {0, 1, 2})  # played, soft, settled
        self.assertTrue(np.isnan(a["root_value"][~a["has_human"]]).any())  # a self-play row without a root value stays NaN
        self.assertTrue(np.isnan(a["p_death"][~a["has_human"]]).all())  # self-play rows have no death share
        self.assertTrue(np.allclose(a["p_death"][a["has_human"]], 0.2))
        self.assertEqual(sorted(str(c) for c in a["cells"]), sorted(jt.CELLS))

    def test_the_policy_target_is_the_visit_distribution_over_the_action_slots(self):
        t = jt.policy_target([child(0, 30, 1.0), child(3, 10, 1.0, kind="end_turn"), child(1, 20, 1.0), child(0, 30, 1.0)])
        self.assertEqual(t.shape, (32,))
        self.assertAlmostEqual(float(t.astype(np.float64).sum()), 1.0, places=2)
        np.testing.assert_allclose([float(t[0]), float(t[1]), float(t[3])], [60 / 90, 20 / 90, 10 / 90], atol=1e-3)  # a target's moves share a slot
        self.assertIsNone(jt.policy_target([]))
        self.assertIsNone(jt.policy_target([child(0, 0, 1.0)]))
        self.assertIsNone(jt.policy_target([child(99, 5, 1.0)]))  # no such slot

    def test_a_corpus_the_manifest_does_not_call_usable_is_refused_unless_asked(self):
        with tempfile.TemporaryDirectory() as d:
            Store(d, usable=False)
            with self.assertRaisesRegex(RuntimeError, "usable_for_training"):
                jt.read_corpus(d)
            _, rep = jt.read_corpus(d, require_usable=False, max_per_cell=5)
            self.assertEqual(rep["rows"], 20)  # 5 of each of the four (source, kind, act) files

    def test_a_human_row_of_the_exam_fifth_is_still_refused(self):
        import human_starts as hs

        run = next(f"x{i}" for i in range(10000) if hs.exam_bucket(f"x{i}") == 0)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            r = mkrow(0, run, "human", "boss", 2, "settled", True, 10.0)
            jc.write_shard(root / "shards" / "human-boss-act2-0000", [r], [toks(3, np.random.default_rng(0))])
            (root / "manifest.json").write_text(json.dumps({"usable_for_training": True}), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "exam fifth"):
                jt.read_corpus(root)


class WeightTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        Store(self.d.name)
        self.a, _ = jt.read_corpus(self.d.name)

    def tearDown(self):
        self.d.cleanup()

    def test_modes_and_the_mean_over_the_training_rows_is_one(self):
        train = self.a["split"] == 0
        for mode in ("pool", "cell", "none", "balanced"):
            w = jt.weights_of(self.a, mode, None)
            self.assertAlmostEqual(float(w[train].mean()), 1.0, places=4, msg=mode)
        w = jt.weights_of(self.a, "pool", None)
        human = is_human(self.a)
        self.assertAlmostEqual(float(w[human].mean() / w[~human].mean()), 2.0, places=3)  # the corpus's w: 2 for a human row, 1 for self-play
        with self.assertRaises(ValueError):
            jt.weights_of(self.a, "bogus")

    def test_balanced_gives_every_cell_the_same_total(self):
        w = jt.weights_of(self.a, "balanced", None)
        totals = [float(w[self.a["cell"] == c].sum()) for c in np.unique(self.a["cell"])]
        self.assertTrue(max(totals) / min(totals) < 1.0001, totals)

    def test_the_clip_cuts_a_large_weight_and_the_human_multiplier_scales_the_human_rows(self):
        a = dict(self.a)
        a["w"] = self.a["w"].copy()
        a["w"][0] = 500.0
        unclipped = jt.weights_of(a, "pool", None)
        w = jt.weights_of(a, "pool", 10.0)
        self.assertLess(float(w.max()), float(unclipped.max()) / 5)  # the 500 is cut to ten times the mean (then renormalised)
        base = jt.weights_of(self.a, "pool", None)
        boosted = jt.weights_of(self.a, "pool", None, human_mult=3.0)
        human = is_human(self.a)
        self.assertGreater(float(boosted[human].sum() / boosted[~human].sum()), 2.5 * float(base[human].sum() / base[~human].sum()))


class LossTests(unittest.TestCase):
    def test_the_outcome_term_is_judge_loss_with_unit_weights(self):
        g = torch.Generator().manual_seed(1)
        out = torch.randn(64, 3, generator=g)
        y = torch.cat([(torch.rand(64, 1, generator=g) > 0.4).float(), torch.rand(64, 1, generator=g), torch.rand(64, 1, generator=g) * 2], 1)
        mine = jt.outcome_loss(out, y, torch.ones(64))
        self.assertAlmostEqual(float(mine), float(jd.judge_loss(out, y)), places=5)

    def test_a_row_without_the_label_adds_nothing_and_a_weight_scales_a_row(self):
        g = torch.Generator().manual_seed(2)
        out = torch.randn(10, 3, generator=g)
        y = torch.cat([(torch.rand(10, 1, generator=g) > 0.4).float(), torch.rand(10, 1, generator=g), torch.rand(10, 1, generator=g)], 1)
        w = torch.ones(10)
        w[4] = 0.0
        keep = torch.arange(10) != 4
        self.assertAlmostEqual(float(jt.outcome_loss(out, y, w)), float(jt.outcome_loss(out[keep], y[keep], torch.ones(9))), places=5)
        self.assertEqual(float(jt.outcome_loss(out, y, torch.zeros(10))), 0.0)
        w2 = torch.ones(10)
        w2[2] = 3.0  # row 2 counts as three copies
        rep_out, rep_y = torch.cat([out, out[2:3], out[2:3]]), torch.cat([y, y[2:3], y[2:3]])
        self.assertAlmostEqual(float(jt.outcome_loss(out, y, w2)), float(jt.outcome_loss(rep_out, rep_y, torch.ones(12))), places=5)

    def test_the_value_is_the_searchs_leaf_value(self):
        out = torch.tensor([[0.3, -0.2, 0.5], [-2.0, 1.0, 0.0], [4.0, 0.0, 1.0]])
        mx = torch.tensor([80.0, 60.0, 50.0])
        mine = jt.value_of(out, mx).numpy()
        for i in range(3):
            want = jcl.estimate(out[i].numpy(), int(mx[i])).value
            self.assertAlmostEqual(float(mine[i]), want, places=4)
            self.assertAlmostEqual(float(jt.value_np(out[i : i + 1].numpy().astype(np.float64), np.array([float(mx[i])]))[0]), want, places=4)

    def test_the_policy_loss_is_the_cross_entropy_against_the_visits(self):
        logits = torch.tensor([[1.0, 2.0, float("-inf"), 0.0]])
        target = torch.tensor([[0.25, 0.75, 0.0, 0.0]])
        lp = torch.log_softmax(logits, -1)
        expect = -(0.25 * lp[0, 0] + 0.75 * lp[0, 1])
        self.assertAlmostEqual(float(jt.policy_loss(logits, target, torch.ones(1))), float(expect), places=5)
        self.assertEqual(float(jt.policy_loss(logits, target, torch.zeros(1))), 0.0)

    def test_each_term_reads_only_its_rows_and_a_zero_weight_leaves_a_term_out(self):
        n = 6
        t = {"w": torch.ones(n), "y_teacher": torch.zeros(n, 4), "m_teacher": torch.ones(n, dtype=torch.bool), "y_human": torch.zeros(n, 4),
             "m_human": torch.zeros(n, dtype=torch.bool), "root_value": torch.zeros(n), "m_value": torch.zeros(n, dtype=torch.bool), "p_death": torch.zeros(n),
             "m_pd": torch.zeros(n, dtype=torch.bool), "tier": torch.zeros(n, dtype=torch.long), "max_hp": torch.full((n,), 80.0), "pol": torch.zeros(n, 32),
             "m_pol": torch.zeros(n, dtype=torch.bool)}
        out = torch.randn(n, 3)
        _, parts = jt.corpus_loss(out, torch.randn(n, 32), t, jt.LossCfg())
        self.assertEqual(parts["human"], 0.0)
        self.assertEqual(parts["value"], 0.0)
        self.assertEqual(parts["policy"], 0.0)
        self.assertGreater(parts["teacher"], 0.0)
        _, off = jt.corpus_loss(out, None, t, jt.LossCfg(w_human=0.0, w_value=0.0, w_search_win=0.0, w_policy=0.0))
        self.assertEqual(set(off), {"teacher"})


class TrainingTests(unittest.TestCase):
    @staticmethod
    def args(**kw):
        ns = jt.argparse.Namespace(init=None, init_judge=None, d_model=24, layers=1, dropout=0.0, seed=0, lr=3e-3, trunk_lr=1.0, wd=0.0, epochs=6, batch=32,
                                   pause=0.0, w_teacher=1.0, w_human=0.5, w_value=1.0, w_search_win=0.5, w_policy=0.1, tier_w=[0.25, 0.5, 1.0])
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def test_the_batch_has_the_tokens_the_network_reads(self):
        with tempfile.TemporaryDirectory() as d:
            Store(d)
            a, _ = jt.read_corpus(d)
        data = jt.CorpusData(a, torch.device("cpu"), jt.weights_of(a, "pool"))
        idx = np.array([0, 5, 9])
        x, _ = data.batch(idx)
        self.assertEqual(x["types"].shape[0], 3)
        self.assertEqual(int(x["types"][0, 0]), 1)  # the global token first
        self.assertEqual(x["types"].dtype, torch.int64)
        n0 = int(a["lens"][0])
        width = x["types"].shape[1]
        if width > n0:
            self.assertTrue(bool((x["types"][0, n0:] == 0).all()))  # padded with the pad type
            self.assertTrue(bool(x["mask"][0, n0:].all()))
        self.assertFalse(bool(x["mask"][0, :n0].any()))
        t = data.targets(idx)
        self.assertEqual(t["pol"].shape, (3, 32))
        self.assertEqual(t["max_hp"].shape, (3,))

    def test_a_short_run_learns_saves_a_judge_and_scores_it_beside_another(self):
        with tempfile.TemporaryDirectory() as d:
            Store(d)
            a, _ = jt.read_corpus(d)
            data = jt.CorpusData(a, torch.device("cpu"), jt.weights_of(a, "pool"))
            model, info = jt.train(self.args(), data, torch.device("cpu"), log=lambda m: None)
            self.assertLess(info["log"][-1]["train_loss"], info["log"][0]["train_loss"])
            self.assertTrue(all(np.isfinite(h["val_loss"]) for h in info["log"]))
            self.assertIn("scale", info["win_calibration"])
            self.assertAlmostEqual(float(model.win_cal[0]), info["win_calibration"]["scale"], places=3)
            # the checkpoint is a judge the loader reads, and gives the rows the trainer's model gives
            path = Path(d) / "j" / "judge.pt"
            path.parent.mkdir()
            torch.save({"kind": jd.KIND, "version": "judge-v2", "model": info["state"], **info["dims"]}, path)
            loaded, blob = jd.load(path)
            self.assertEqual(blob["version"], "judge-v2")
            idx = np.flatnonzero(data.split == 2)[:20]
            np.testing.assert_allclose(jt.predict(loaded, data, idx), jt.predict(model, data, idx), atol=1e-5)
            # a second, untrained judge scored on the same held-out rows
            torch.manual_seed(3)
            other = jd.JudgeNet(VOCAB, info["dims"]["n_buckets"], d_model=24, layers=1, heads=info["dims"]["heads"])
            p2 = Path(d) / "k" / "judge.pt"
            p2.parent.mkdir()
            torch.save({"kind": jd.KIND, "version": "judge-x", "model": other.state_dict(), **info["dims"]}, p2)
            report = jt.evaluate_paths([str(path), str(p2)], data, 2, log=lambda m: None, min_n=10)
        self.assertEqual(set(report["judges"]), {str(path), str(p2)})
        sc = report["judges"][str(path)]["scores"]
        self.assertIn("all", sc)
        self.assertIn("act2-boss", sc)
        self.assertIn("teacher", sc["all"])
        self.assertIn("human", sc["all"])
        self.assertIn("search_settled", sc["act2-boss"])
        self.assertIn("act2-boss | human:human-start", sc)
        for k in ("pearson", "spearman", "win_ece", "win_brier"):
            self.assertIn(k, sc["all"]["teacher"])
        md = jt.md_tables(report, {str(path): "v2", str(p2): "rand"})
        self.assertIn("v2 pearson", md)
        self.assertIn("rand pearson", md)

    def test_a_frozen_trunk_keeps_its_weights(self):
        with tempfile.TemporaryDirectory() as d:
            Store(d)
            a, _ = jt.read_corpus(d)
            data = jt.CorpusData(a, torch.device("cpu"), jt.weights_of(a, "pool"))
            ns = self.args(trunk_lr=0.0, epochs=3)
            torch.manual_seed(0)
            m0, _ = jt.make_model(ns, torch.device("cpu"))
            before = {k: v.clone() for k, v in m0.state_dict().items() if k.startswith(("head.", "policy.", "encoder."))}
            torch.manual_seed(0)
            model, info = jt.train(ns, data, torch.device("cpu"), log=lambda m: None)
            self.assertTrue(info["log"])
            after = model.state_dict()
            self.assertTrue(before)
            for k, v in before.items():
                self.assertTrue(torch.equal(v, after[k].cpu()), k)


class SearchCalibrationTests(unittest.TestCase):
    def test_pav_pools_adjacent_violators_and_keeps_the_weighted_mean(self):
        np.testing.assert_allclose(jt.pav(np.array([1.0, 3.0, 2.0, 4.0]), np.ones(4)), [1.0, 2.5, 2.5, 4.0])
        np.testing.assert_allclose(jt.pav(np.array([3.0, 1.0]), np.array([1.0, 3.0])), [1.5, 1.5])
        up = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(jt.pav(up, np.ones(3)), up)

    def test_the_monotone_map_is_the_conditional_mean_and_never_falls(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-100, 50, 20000)
        y = 0.8 * x + 25 + rng.normal(0, 10, len(x))  # the outcomes sit 25 above a search that reads low
        kx, ky = jt.monotone_map(x, y, bins=50)
        self.assertTrue(bool(np.all(np.diff(ky) >= 0)))
        np.testing.assert_allclose(np.interp([-50.0, 0.0, 30.0], kx, ky), [0.8 * v + 25 for v in (-50.0, 0.0, 30.0)], atol=2.0)

    def test_calibrate_search_moves_the_level_to_the_outcomes_fits_on_training_rows_only_and_leaves_missing_values(self):
        with tempfile.TemporaryDirectory() as d:
            Store(d, n_runs=40)
            a, _ = jt.read_corpus(d)
        rng = np.random.default_rng(1)
        a = dict(a)
        n = len(a["lens"])
        value = jt.outcome_value(a)
        a["root_value"] = (value - 30.0 + rng.normal(0, 3, n)).astype(np.float32)  # a search that reads 30 HP low
        a["root_value"][::7] = np.nan
        cal = jt.calibrate_search(a, bins=20, min_rows=20)
        ok = ~np.isnan(a["root_value"])
        self.assertTrue(np.array_equal(np.isnan(cal["root_cal"]), ~ok))  # a row without a value stays without
        before = float(np.mean(a["root_value"][ok] - value[ok]))
        after = float(np.mean(cal["root_cal"][ok] - value[ok]))
        self.assertLess(abs(after), abs(before) / 5)
        self.assertGreater(len(cal["report"]), 0)
        # held-out outcomes do not move the map
        b = dict(a)
        b["y_teacher"] = a["y_teacher"].copy()
        held = a["split"] == 2
        b["y_teacher"][held, 0] = 1.0 - b["y_teacher"][held, 0]
        cal_b = jt.calibrate_search(b, bins=20, min_rows=20)
        np.testing.assert_allclose(cal["root_cal"], cal_b["root_cal"])
        # too few rows for any map: the values are left as they are
        keep = jt.calibrate_search(a, bins=20, min_rows=10**9)
        np.testing.assert_allclose(keep["root_cal"], a["root_value"].astype(np.float32), equal_nan=True)

    def test_the_death_share_is_put_on_the_wins_scale_too(self):
        with tempfile.TemporaryDirectory() as d:
            Store(d, n_runs=40)
            a, _ = jt.read_corpus(d)
        a = dict(a)
        won = a["y_teacher"][:, 0] > 0.5
        a["p_death"] = np.where(won, 0.5, 0.9).astype(np.float32)  # a search that reads far too many deaths
        cal = jt.calibrate_search(a, bins=10, min_rows=20)
        self.assertLess(float(np.mean(cal["pd_cal"][won])), 0.2)
        self.assertGreater(float(np.mean(cal["pd_cal"][~won])), 0.8)
        d_ = jt.CorpusData(a, torch.device("cpu"), jt.weights_of(a, "none"), cal)
        raw = jt.CorpusData(a, torch.device("cpu"), jt.weights_of(a, "none"))
        idx = np.arange(5)
        self.assertFalse(torch.allclose(d_.targets(idx)["p_death"], raw.targets(idx)["p_death"]))
        self.assertTrue(np.allclose(d_.a["p_death"], a["p_death"]))  # the evaluation still reads the raw number


class InitTests(unittest.TestCase):
    def test_a_trunk_starts_from_the_agents_network_and_a_judge_from_a_judge(self):
        import tempfile as tf

        from combat_net import CombatNet

        torch.manual_seed(0)
        agent = CombatNet(VOCAB, 19, d_model=24, layers=1, heads=2)
        with tf.TemporaryDirectory() as d:
            ap = Path(d) / "combat_net.pt"
            torch.save({"model": agent.state_dict(), "n_buckets": 19, "d_model": 24, "layers": 1, "heads": 2}, ap)
            model, dims = jt.make_model(TrainingTests.args(init=str(ap)), torch.device("cpu"))
            self.assertEqual(dims, {"n_buckets": 19, "d_model": 24, "layers": 1, "heads": 2})
            for k, v in agent.state_dict().items():
                self.assertTrue(torch.equal(model.state_dict()[k], v), k)  # the agent's trunk, head and policy are the judge's start
            self.assertTrue(bool((model.src_emb.weight == 0).all()))
            # a judge checkpoint to continue from
            jp = Path(d) / "judge.pt"
            torch.save({"kind": jd.KIND, "model": model.state_dict(), **dims}, jp)
            again, dims2 = jt.make_model(TrainingTests.args(init_judge=str(jp)), torch.device("cpu"))
            self.assertEqual(dims2, dims)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(again.state_dict()[k], v), k)


class SourceInputTests(unittest.TestCase):
    """The judge head's source input: the teacher (what the search asks) is the baseline, the human reading is an offset trained to the human's outcome."""

    def test_the_default_reading_is_the_teachers_and_an_old_checkpoint_loads_unchanged(self):
        torch.manual_seed(0)
        shape = {"n_buckets": 19, "d_model": 24, "layers": 1, "heads": 2}
        model = jd.JudgeNet(VOCAB, **shape).eval()
        self.assertTrue(bool((model.src_emb.weight == 0).all()))
        rng = np.random.default_rng(0)
        x = jd.batch_tensors([type("T", (), toks(3, rng))() for _ in range(4)])
        base = model.judge_out(**x)
        zeros = model.judge_out(**x, src=torch.zeros(4, dtype=torch.long))
        self.assertTrue(torch.equal(base, zeros))
        # a checkpoint without the source input (v0 and v1) loads and reads the same
        state = {k: v for k, v in model.state_dict().items() if not k.startswith("src_emb")}
        old = jd.JudgeNet.from_blob({"model": state, **shape}).eval()
        self.assertTrue(torch.allclose(old.judge_out(**x), base, atol=1e-6))
        with self.assertRaises(ValueError):
            jd.JudgeNet.from_blob({"model": {k: v for k, v in state.items() if not k.startswith("judge.")}, **shape})

    def test_the_human_term_moves_only_the_human_reading(self):
        with tempfile.TemporaryDirectory() as d:
            Store(d)
            a, _ = jt.read_corpus(d)
            data = jt.CorpusData(a, torch.device("cpu"), jt.weights_of(a, "pool"))
            model, info = jt.train(TrainingTests.args(epochs=3), data, torch.device("cpu"), log=lambda m: None)
            w = model.src_emb.weight.detach()
            self.assertTrue(bool((w[0] == 0).all()))  # the teacher row never moves
            self.assertGreater(float(w[1].abs().sum()), 0.0)  # the human row learned something
            idx = np.flatnonzero(data.split == 2)[:16]
            t, h = jt.predict(model, data, idx), jt.predict(model, data, idx, human=True)
            self.assertFalse(np.allclose(t, h))
            # without the human term the human row stays where it started
            model2, _ = jt.train(TrainingTests.args(epochs=3, w_human=0.0), data, torch.device("cpu"), log=lambda m: None)
            self.assertTrue(bool((model2.src_emb.weight == 0).all()))


if __name__ == "__main__":
    unittest.main()
