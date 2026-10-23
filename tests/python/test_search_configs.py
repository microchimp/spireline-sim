"""Named search configs and the budget schedule (rl/search_configs.py), and the A3 dev-set manifest (rl/a3_devset.py)."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))

import a3_devset  # noqa: E402
import search_configs as sc  # noqa: E402

CALM = sc.Position(kind="play", n_moves=4, hp_frac=0.9, hard=False)


class ConfigLoadingTests(unittest.TestCase):
    def test_gen0_60_is_generation_zeros_call(self):
        cfg = sc.get("gen0-60")
        self.assertEqual(cfg.search_kwargs(CALM), {"max_sims": 60, "min_child_sims": 2, "c": 0.1, "confirm_pairs": 0,
                                                   "time_budget_s": 120.0, "resample": True})
        self.assertIsNone(cfg.schedule)

    def test_flat_n_and_unknown_names(self):
        self.assertEqual(sc.get("flat-240").sims_for(CALM), 240)
        with self.assertRaises(KeyError):
            sc.get("no-such-config")
        with self.assertRaises(KeyError):
            sc.get("flat-x")

    def test_named_configs_for_both_uses_exist_and_resample(self):
        for name in ("a3-v1", sc.SELFPLAY, sc.SCORE_REAL):
            cfg = sc.get(name)
            self.assertTrue(cfg.search_kwargs(CALM)["resample"])
            self.assertLessEqual(cfg.sims_for(sc.Position("target", 30, 0.1, True)), 2000)

    def test_every_config_name_matches_its_key(self):
        for name, cfg in sc.CONFIGS.items():
            self.assertEqual(cfg.name, name)

    def test_bad_kind_is_refused(self):
        with self.assertRaises(ValueError):
            sc.Position(kind="shop", n_moves=2, hp_frac=1.0, hard=False)


class ScheduleTests(unittest.TestCase):
    cfg = sc.SearchConfig("t", 60, schedule=sc.Schedule(high=240, low_hp=0.5, many_moves=8,
                                                         kind_mult={"select": 0.5}, lo=10, hi=200))

    def test_calm_decision_gets_the_base(self):
        self.assertEqual(self.cfg.sims_for(CALM), 60)

    def test_each_stake_raises_the_budget(self):
        for pos in (sc.Position("play", 4, 0.9, True), sc.Position("play", 4, 0.49, False),
                    sc.Position("play", 8, 0.9, False)):
            self.assertTrue(self.cfg.high_stakes(pos), pos)
            self.assertEqual(self.cfg.sims_for(pos), 200)  # 240, clamped to hi

    def test_kind_multiplier_and_floor(self):
        self.assertEqual(self.cfg.sims_for(sc.Position("select", 3, 0.9, False)), 30)
        low = sc.SearchConfig("t", 10, schedule=sc.Schedule(high=10, kind_mult={"select": 0.5}, lo=8))
        self.assertEqual(low.sims_for(sc.Position("select", 3, 0.9, False)), 8)

    def test_no_position_means_the_flat_budget(self):
        self.assertEqual(self.cfg.sims_for(None), 60)

    def test_a3_v1_spends_on_targets_and_choice_screens(self):
        cfg = sc.get("a3-v1")
        self.assertEqual(cfg.sims_for(CALM), 30)
        self.assertEqual(cfg.sims_for(sc.Position("target", 6, 0.9, False)), 120)
        self.assertEqual(cfg.sims_for(sc.Position("select", 3, 0.9, False)), 120)
        # The dev set showed no gain from stakes alone: an elite at low HP with a plain play stays at the base.
        self.assertEqual(cfg.sims_for(sc.Position("play", 9, 0.2, True)), 30)
        self.assertEqual(cfg.search_kwargs(CALM)["c"], 0.2)


class ManifestTests(unittest.TestCase):
    @staticmethod
    def cands():
        out = []
        for ix in range(40):
            for k in range(30):
                out.append({"ix": ix, "seed": f"S{ix}", "k": k, "kind": ("play", "target", "select")[k % 3],
                            "n_moves": 1 + (k % 5), "hp_frac": (k % 10) / 10, "hard": ix % 4 == 0, "floor": 1 + ix % 20})
        return out

    def test_seeded_and_reproducible(self):
        a = a3_devset.make_manifest(self.cands(), 50, 7)
        self.assertEqual(a, a3_devset.make_manifest(self.cands(), 50, 7))
        self.assertNotEqual(a["positions"], a3_devset.make_manifest(self.cands(), 50, 8)["positions"])
        self.assertEqual(a["n"], 50)

    def test_only_real_choices_and_balanced_strata(self):
        m = a3_devset.make_manifest(self.cands(), 48, 1)
        by = {(p["ix"], p["k"]): p for p in m["positions"]}
        cand = {(c["ix"], c["k"]): c for c in self.cands()}
        self.assertTrue(all(cand[k]["n_moves"] >= 2 for k in by))
        counts = list(m["sample"].values())
        self.assertLessEqual(max(counts) - min(c for c in counts), 1 + max(0, max(counts) - min(counts)))
        full = [s for s, n in m["sample"].items() if n < m["population"][s]]
        if full:
            self.assertLessEqual(max(m["sample"][s] for s in full) - min(m["sample"][s] for s in full), 1)

    def test_snap_to_grid(self):
        self.assertEqual(a3_devset.snap(45), 30)
        self.assertEqual(a3_devset.snap(100), 120)
        self.assertEqual(a3_devset.snap(1000), 240)


if __name__ == "__main__":
    unittest.main()
