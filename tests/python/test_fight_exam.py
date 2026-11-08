"""The fight exam (rl/fight_exam.py; D55, item 3.1 of the 2026-11-04 train-and-test memo).

The pinned facts: the exam is stratified by act, kind and boss, up to the per-cell target, from clean starts (held-out split or never
played by the generation) of runs the promotion board does not stand on, and a short cell is reported short; its exclusion list names
the runs and the start ids behind it; the halves are a fixed hash of the start id; the draws' seeds are in their own namespace and do
not depend on the player; the calibration flags give always-won and never-won starts weight zero; the paired differences, the
McNemar counts, the bootstrap and the baseline adjustment come out as the arithmetic says on synthetic outcomes; the report text
names the cells, the bosses, the flags and the halves' sizes; the lock protocol reads the labs' files as the loop does; and, when
the native library is there, the observer changes no move of the gym's game and its measures are consistent with the fight.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fight_exam as fe  # noqa: E402
import fight_gym as fg  # noqa: E402
import gym_loop as gl  # noqa: E402

F18 = HERE / "fixtures" / "f18_step_hang.json"


def lite(i: int, act: int, kind: str, enc: int = 0, seed: str | None = None, split: str = "train") -> dict:
    cat = f"act{act}-{kind}"
    return {"id": f"n/g:R{i}|f{i % 30:02d}", "cat": cat, "kind": kind, "act": act, "floor": i % 30, "seed": seed or f"R{i}", "ix": i, "night": "n/g",
            "k": 0, "node_type": 1, "hp": 50, "max_hp": 80, "deck_size": 12, "potions": 1, "relics": 3, "encounter_id": enc, "turn": 0,
            "kind_same": True, "replay_same": False, "split": split, "doc": {"x": i}}


def synthetic_pool() -> list[dict]:
    """Act 1: hallways 40, elites 12, bosses 70 / 30 / 5 of encounters 74 / 77 / 79; Act 2: hallways 25, elites 6, bosses 3 / 30 of 75 / 76."""
    pool, i = [], 0
    for act, kind, enc, n in ((1, "hallway", 11, 40), (1, "elite", 62, 12), (1, "boss", 74, 70), (1, "boss", 77, 30), (1, "boss", 79, 5),
                              (2, "hallway", 21, 25), (2, "elite", 63, 6), (2, "boss", 75, 3), (2, "boss", 76, 30)):
        for _ in range(n):
            pool.append(lite(i, act, kind, enc))
            i += 1
    return pool


class SelectionTests(unittest.TestCase):
    def test_cells_are_act_kind_and_for_a_boss_the_boss(self):
        self.assertEqual(fe.cell_of(lite(1, 1, "hallway", 11)), "act1-hallway")
        self.assertEqual(fe.cell_of(lite(1, 2, "elite", 63)), "act2-elite")
        self.assertEqual(fe.cell_of(lite(1, 1, "boss", 74)), "act1-boss-74")
        self.assertEqual(fe.kind_cell(lite(1, 1, "boss", 74)), "act1-boss")
        self.assertIn("ceremonial-beast", fe.cell_label("act1-boss-74"))
        self.assertEqual(fe.cell_label("act1-elite"), "act1-elite")

    def test_the_pick_fills_each_cell_to_the_target_and_reports_the_short_ones(self):
        pool = synthetic_pool()
        chosen, report = fe.pick_exam(pool, exclude_runs=set(), practised={}, per_cell=20)
        per = {c: sum(1 for s in chosen if s["cell"] == c) for c in report}
        self.assertEqual(per, {"act1-hallway": 20, "act1-elite": 12, "act1-boss-74": 20, "act1-boss-77": 20, "act1-boss-79": 5,
                               "act2-hallway": 20, "act2-elite": 6, "act2-boss-75": 3, "act2-boss-76": 20})
        short = sorted(c for c, v in report.items() if v["short"])
        self.assertEqual(short, ["act1-boss-79", "act1-elite", "act2-boss-75", "act2-elite"])
        self.assertEqual(report["act1-boss-74"], {"pool": 70, "clean": 70, "chosen": 20, "practised_added": 0, "short": False})
        by_kind = {k: sum(1 for s in chosen if s["kind_cell"] == k) for k in ("act1-boss", "act2-boss")}
        self.assertEqual(by_kind, {"act1-boss": 45, "act2-boss": 23})
        for s in chosen:
            self.assertEqual(s["half"], fe.half_of(s["id"]))
            self.assertTrue(s["clean"])

    def test_the_pick_is_fixed_by_its_seed_and_not_by_the_pools_order(self):
        pool = synthetic_pool()
        a, _ = fe.pick_exam(pool, exclude_runs=set(), practised={}, per_cell=10, seed=7)
        b, _ = fe.pick_exam(list(reversed(pool)), exclude_runs=set(), practised={}, per_cell=10, seed=7)
        c, _ = fe.pick_exam(pool, exclude_runs=set(), practised={}, per_cell=10, seed=8)
        self.assertEqual(sorted(s["id"] for s in a), sorted(s["id"] for s in b))
        self.assertNotEqual(sorted(s["id"] for s in a), sorted(s["id"] for s in c))

    def test_runs_the_board_stands_on_are_never_picked(self):
        pool = synthetic_pool()
        board = {s["seed"] for s in pool[:30]}
        chosen, report = fe.pick_exam(pool, exclude_runs=board, practised={}, per_cell=100)
        self.assertTrue(all(s["seed"] not in board for s in chosen))
        self.assertEqual(report["act1-hallway"]["pool"], 10)  # 40 hallways, 30 of them in excluded runs

    def test_only_clean_starts_are_taken_unless_asked_and_then_the_least_played_first(self):
        pool = synthetic_pool()
        boss = [s for s in pool if s["encounter_id"] == 74]
        practised = {s["id"]: 3 for s in boss[:60]}  # 60 of the 70 were played; 10 never
        practised[boss[0]["id"]] = 9
        boss[5]["split"] = "val"  # played, but its run is held out of training: clean
        chosen, report = fe.pick_exam(pool, exclude_runs=set(), practised=practised, per_cell=30)
        got = [s for s in chosen if s["cell"] == "act1-boss-74"]
        self.assertEqual(len(got), 11)  # the 10 never played and the held-out one
        self.assertTrue(report["act1-boss-74"]["short"])
        self.assertEqual(report["act1-boss-74"]["clean"], 11)
        more, rep2 = fe.pick_exam(pool, exclude_runs=set(), practised=practised, per_cell=30, allow_practised=True)
        got2 = [s for s in more if s["cell"] == "act1-boss-74"]
        self.assertEqual(len(got2), 30)
        self.assertEqual(sum(1 for s in got2 if not s["clean"]), 19)
        self.assertEqual(rep2["act1-boss-74"]["practised_added"], 19)
        self.assertNotIn(boss[0]["id"], [s["id"] for s in got2])  # the one played nine times is the last to be added
        self.assertTrue(all(s["practised"] == 3 for s in got2 if not s["clean"]))

    def test_the_exclusion_lists_name_the_runs_and_the_starts(self):
        chosen, _ = fe.pick_exam(synthetic_pool(), exclude_runs=set(), practised={}, per_cell=5)
        ex = fe.exclusion(chosen)
        self.assertEqual(ex["starts"], len(chosen))
        self.assertEqual(ex["start_ids"], sorted(s["id"] for s in chosen))
        self.assertEqual(ex["run_seeds"], sorted({s["seed"] for s in chosen}))
        self.assertEqual(ex["by_stratum"], {"ai": len(chosen)})


class HalfAndDrawTests(unittest.TestCase):
    def test_the_half_is_a_fixed_hash_of_the_start_id_and_the_halves_are_even(self):
        ids = [f"n/g:S{i}|f{i % 30:02d}" for i in range(4000)]
        halves = [fe.half_of(x) for x in ids]
        self.assertEqual(halves, [fe.half_of(x) for x in ids])
        a = halves.count("A")
        self.assertTrue(1900 < a < 2100, a)
        # pinned: the rule is sha256("fight-exam-half:" + id), first 32 bits below 2^31 is A; a change of rule moves the halves
        import hashlib  # noqa: PLC0415

        for x in ids[:50]:
            h = int(hashlib.sha256(("fight-exam-half:" + x).encode()).hexdigest()[:8], 16)
            self.assertEqual(fe.half_of(x), "A" if h < 2**31 else "B")
        self.assertEqual({fe.half_of(x) for x in ("n/g:S0|f00", "n/g:S1|f01")} <= {"A", "B"}, True)

    def test_the_draws_seeds_have_their_own_namespace_and_never_depend_on_the_player(self):
        self.assertEqual(fe.NS, "x")
        self.assertNotIn(fe.NS, (gl.PROMOTION_RULE["ns"], gl.NS_GEN, gl.NS_WEIGHTS))
        for j in range(fe.DRAWS):
            self.assertEqual(fe.draw_seed("S|f01", j), fg.game_seed("S|f01", j, "xload"))
            self.assertNotEqual(fe.draw_seed("S|f01", j), fg.game_seed("S|f01", j, "gload"))
            self.assertNotEqual(fe.draw_seed("S|f01", j), fe.draw_seed("S|f01", j + 1))
        starts = {s["id"]: {**s, "cell": fe.cell_of(s), "half": fe.half_of(s["id"])} for s in synthetic_pool()[:20]}
        t1 = fe.exam_tasks(starts, "v2", "v2", "a3-v1", Path("e"), set())
        t2 = fe.exam_tasks(starts, "c0150", "c0150", "a3-v1", Path("e"), set())
        key = lambda t: (t[1], t[2], t[4]["ns"])  # noqa: E731
        self.assertEqual(sorted(map(key, t1)), sorted(map(key, t2)))  # the same games to play, whoever plays them
        self.assertEqual(len(t1), 20 * fe.DRAWS)
        self.assertEqual({t[4]["ns"] for t in t1}, {"x"})
        self.assertEqual(len(fe.exam_tasks(starts, "v2", "v2", "a3-v1", Path("e"), {(t[1], t[2]) for t in t1[:7]})), len(t1) - 7)
        half = fe.exam_tasks(starts, "v2", "v2", "a3-v1", Path("e"), set(), half="A")
        self.assertTrue(all(starts[t[1]]["half"] == "A" for t in half))
        order = [starts[t[1]]["kind"] for t in t1]  # long fights first
        self.assertEqual(order, sorted(order, key={"boss": 0, "elite": 1, "hallway": 2}.get))


# ---------------------------------------------------------------------------------------------------------- the readings


def games_for(starts: dict, p_of, rnd: random.Random, draws: int = 4, tweak=None) -> dict:
    """Synthetic games: start s wins each draw with probability p_of(s); the other measures follow the win."""
    out = {}
    for sid, s in starts.items():
        for j in range(draws):
            won = rnd.random() < p_of(s)
            out[(sid, j)] = {"start": sid, "j": j, "won": won, "capped": False, "L0": (0.6 + 0.3 * rnd.random()) if won else 0.0,
                             "hp_lost": float(rnd.randint(0, 20) if won else 50), "potions_used": float(rnd.randint(0, 1)),
                             "turns": float(rnd.randint(3, 9)), "dealt": float(rnd.randint(20, 90)), "energy": float(rnd.randint(8, 30)),
                             "wasted": float(rnd.randint(0, 12))}
            if tweak:
                tweak(out[(sid, j)], s, j)
    return out


def synth_starts(n_per_cell: int = 12) -> dict:
    pool = []
    i = 0
    for act, kind, enc in ((1, "hallway", 11), (1, "elite", 62), (1, "boss", 74), (1, "boss", 77), (2, "hallway", 21), (2, "boss", 75)):
        for _ in range(n_per_cell):
            pool.append(lite(i, act, kind, enc))
            i += 1
    return {s["id"]: {**{k: v for k, v in s.items() if k != "doc"}, "cell": fe.cell_of(s), "kind_cell": fe.kind_cell(s), "half": fe.half_of(s["id"]),
                      "stratum": "ai"} for s in pool}


class CalibrationAndPairingTests(unittest.TestCase):
    def setUp(self):
        self.starts = synth_starts()
        self.rnd = random.Random(5)

    def test_always_and_never_won_starts_get_weight_zero_and_the_rest_one(self):
        ids = sorted(self.starts)
        wins = {ids[0]: [1, 1, 1, 1], ids[1]: [0, 0, 0, 0], ids[2]: [1, 0, 0, 0], ids[3]: [1, 1, 1, 0]}
        games = {}
        for sid in ids:
            for j in range(4):
                if sid in wins:
                    games[(sid, j)] = {"start": sid, "j": j, "won": bool(wins[sid][j]), "capped": False}
        games[(ids[4], 0)] = {"start": ids[4], "j": 0, "won": True, "capped": False}  # only one draw: not flagged
        cal = fe.calibration(games, self.starts)
        self.assertEqual(cal["always"], [ids[0]])
        self.assertEqual(cal["never"], [ids[1]])
        self.assertEqual(sorted(cal["mixed"]), sorted([ids[2], ids[3]]))
        self.assertIn(ids[4], cal["unplayed"])
        w = fe.weights_of(cal, self.starts)
        self.assertEqual((w[ids[0]], w[ids[1]], w[ids[2]], w[ids[3]], w[ids[4]], w[ids[10]]), (0.0, 0.0, 1.0, 1.0, 0.0, 0.0))
        capped = {(ids[5], j): {"start": ids[5], "j": j, "won": True, "capped": True} for j in range(4)}
        self.assertEqual(fe.calibration(capped, self.starts)["never"], [ids[5]])  # a capped game is lost

    def test_the_paired_difference_is_the_mean_over_weight_one_starts_by_cell_on_shared_draws(self):
        champ = games_for(self.starts, lambda s: 0.5, self.rnd)
        cand = {k: dict(v) for k, v in champ.items()}
        ids = sorted(self.starts)
        for sid in ids[:6]:  # the candidate wins one more draw than the champion on six starts, if it can
            for j in range(4):
                if not champ[(sid, j)]["won"]:
                    cand[(sid, j)]["won"] = True
                    break
        cal = fe.calibration(champ, self.starts)
        w = fe.weights_of(cal, self.starts)
        groups = fe.paired_starts(champ, cand, self.starts, w)
        flat = [it for v in groups.values() for it in v]
        self.assertEqual({it["start"] for it in flat}, {s for s in ids if w[s] > 0})
        want = sum(it["d_won"] for it in flat) / len(flat)
        self.assertAlmostEqual(fe.mean_stat("d_won")(groups), want)
        for it in flat:
            self.assertAlmostEqual(it["d_won"], it["cand_won"] - it["champ_won"])
        part = fe.paired_starts(champ, {k: v for k, v in cand.items() if k[1] < 2}, self.starts, w)  # a candidate with two draws
        self.assertTrue(all(it["draws"] == 2 for v in part.values() for it in v))

    def test_mcnemar_counts_match_the_loops_exact_test(self):
        ids = sorted(self.starts)[:4]
        champ, cand = {}, {}
        outcomes = [(1, 0)] * 5 + [(0, 1)] * 2 + [(1, 1)] * 3 + [(0, 0)] * 2
        k = 0
        for sid in ids:
            for j in range(4):
                if k < len(outcomes):
                    champ[(sid, j)] = {"start": sid, "j": j, "won": bool(outcomes[k][0]), "capped": False}
                    cand[(sid, j)] = {"start": sid, "j": j, "won": bool(outcomes[k][1]), "capped": False}
                    k += 1
        starts = {s: self.starts[s] for s in ids}
        mc = fe.mcnemar_counts(champ, cand, starts, {s: 1.0 for s in ids})
        self.assertEqual((mc["cand_only"], mc["champ_only"], mc["both"], mc["neither"], mc["pairs"]), (2, 5, 3, 2, 12))
        self.assertAlmostEqual(mc["p"], gl.mcnemar_one_sided(2, 5))
        zero = fe.mcnemar_counts(champ, cand, starts, {s: 0.0 for s in ids})
        self.assertEqual(zero["pairs"], 0)
        self.assertEqual(zero["p"], 1.0)

    def test_the_cross_fit_uses_flags_from_other_draws_and_a_selection_the_all_draws_flags_make_shows_only_in_those(self):
        # Two identical players: the difference is zero in expectation. Flags from all four of the champion's draws select the
        # starts it happened to play in the middle of its range, and the candidate's draws on them are unselected: positive bias.
        starts = {f"s{i}": {"cell": "act1-boss-74", "kind_cell": "act1-boss", "half": "A", "id": f"s{i}", "kind": "boss"} for i in range(5000)}
        rnd = random.Random(11)
        pt = {sid: rnd.choice((0.8, 0.85, 0.9, 0.95, 0.5)) for sid in starts}
        champ = games_for(starts, lambda s: pt[s["id"]], rnd)
        cand = games_for(starts, lambda s: pt[s["id"]], rnd)
        cal = fe.calibration(champ, starts)
        all4 = fe.mean_stat("d_won")(fe.paired_starts(champ, cand, starts, fe.weights_of(cal, starts)))
        cf = fe.crossfit_difference(champ, cand, starts, boot=50)
        self.assertGreater(all4, 0.03)  # selected on the champion's own draws: the candidate looks better than an equal player
        self.assertLess(abs(cf["estimate"]), 0.02)  # flags from the other draws: no selection
        self.assertLess(abs(cf["estimate"]), all4 / 2)


class StatisticsTests(unittest.TestCase):
    def test_ols_recovers_a_plane(self):
        rnd = random.Random(1)
        rows = [[rnd.random(), rnd.random()] for _ in range(200)]
        y = [0.3 + 2.0 * r[0] - 1.5 * r[1] for r in rows]
        b = fe.ols(rows, y)
        for got, want in zip(b, (0.3, 2.0, -1.5)):
            self.assertAlmostEqual(got, want, places=5)

    def test_the_bootstrap_is_fixed_by_its_seed_and_covers_the_mean(self):
        rnd = random.Random(2)
        groups = {"a": [{"d": rnd.gauss(0.1, 1.0)} for _ in range(300)], "b": [{"d": rnd.gauss(0.1, 1.0)} for _ in range(300)]}
        one = fe.bootstrap(groups, fe.mean_stat("d"), boot=400, seed=3)
        two = fe.bootstrap(groups, fe.mean_stat("d"), boot=400, seed=3)
        self.assertEqual(one, two)
        self.assertEqual(one["n"], 600)
        self.assertLess(one["lo"], one["estimate"])
        self.assertGreater(one["hi"], one["estimate"])
        self.assertAlmostEqual(one["se"], 1.0 / 600**0.5, delta=0.01)
        self.assertLess(one["lo"], 0.1 + 0.05)
        self.assertGreater(one["hi"], 0.1 - 0.05)

    def test_the_bootstrap_resamples_inside_the_cells(self):
        groups = {"a": [{"d": 1.0}] * 10, "b": [{"d": 0.0}] * 30}
        b = fe.bootstrap(groups, fe.mean_stat("d"), boot=200)
        self.assertAlmostEqual(b["estimate"], 0.25)
        self.assertAlmostEqual(b["lo"], 0.25)  # the cells' sizes are kept: no resample moves the mean
        self.assertAlmostEqual(b["hi"], 0.25)

    def _items(self, rnd, n, shift=0.0):
        items = []
        for i in range(n):
            x = min(0.97, max(0.03, rnd.random() * 0.9 + 0.05 + shift))
            d = 0.8 * x * (1 - x) * 2 + rnd.gauss(0, 0.02)
            items.append({"start": f"s{i}", "x": x, "d_won": d})
        return items

    def test_the_adjustment_moves_the_mean_to_the_reference_and_narrows_the_interval_when_the_baseline_explains_the_differences(self):
        rnd = random.Random(9)
        everyone = self._items(rnd, 2000)
        ref = [sum(f) / len(everyone) for f in zip(*[fe.features(it["x"]) for it in everyone])]
        truth = sum(it["d_won"] for it in everyone) / len(everyone)
        sample = self._items(rnd, 300, shift=0.3)  # a half whose baselines run high: its raw mean is off
        raw = fe.mean_stat("d_won")({"c": sample})
        adj = fe.adjusted_stat("d_won", ref)({"c": sample})
        self.assertGreater(abs(raw - truth), 0.03)
        self.assertLess(abs(adj - truth), 0.012)
        b_raw = fe.bootstrap({"c": sample}, fe.mean_stat("d_won"), boot=300)
        b_adj = fe.bootstrap({"c": sample}, fe.adjusted_stat("d_won", ref), boot=300)
        self.assertLess(b_adj["hi"] - b_adj["lo"], 0.5 * (b_raw["hi"] - b_raw["lo"]))

    def test_with_the_sample_equal_to_the_reference_the_adjusted_estimate_is_the_raw_one(self):
        rnd = random.Random(10)
        items = self._items(rnd, 400)
        ref = [sum(f) / len(items) for f in zip(*[fe.features(it["x"]) for it in items])]
        self.assertAlmostEqual(fe.adjusted_stat("d_won", ref)({"c": items}), fe.mean_stat("d_won")({"c": items}), places=9)

    def test_items_without_a_baseline_are_left_out_of_the_adjustment(self):
        items = [{"start": f"s{i}", "x": None, "d_won": 0.1} for i in range(10)]
        self.assertIsNone(fe.adjusted_stat("d_won", [0.5, 0.25])({"c": items}))

    def test_the_baseline_is_a_mean_over_a_starts_draws(self):
        rows = [{"start": "a", "j": 0, "p_win": 0.2, "l0": 0.4}, {"start": "a", "j": 1, "p_win": 0.4, "l0": 0.6}, {"start": "b", "j": 0, "p_win": 0.9, "l0": 0.9}]
        got = fe.baseline_by_start(rows)
        self.assertAlmostEqual(got["a"]["p_win"], 0.3)
        self.assertEqual(got["b"]["draws"], 1)
        starts = {"a": {}, "b": {}, "c": {}}
        ref = fe.reference_means(starts, {"a": 1.0, "b": 1.0, "c": 1.0}, got)  # c has no baseline: left out
        self.assertAlmostEqual(ref[0], 0.6)
        self.assertAlmostEqual(ref[1], (0.3 * 0.7 + 0.9 * 0.1) / 2)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.starts = synth_starts()
        rnd = random.Random(21)
        # champion: wins boss 74 never, boss 77 half the time, the rest mostly; flags follow
        p = {"act1-boss-74": 0.0, "act1-boss-77": 0.5, "act1-hallway": 1.0, "act1-elite": 0.7, "act2-hallway": 0.9, "act2-boss-75": 0.1}
        self.games = games_for(self.starts, lambda s: p[s["cell"]], rnd)
        self.cal = fe.calibration(self.games, self.starts)

    def test_the_report_names_the_cells_the_bosses_the_flags_and_the_halves(self):
        text = fe.report_text(self.games, self.starts, "v2", self.cal)
        n = len(self.starts)
        halves = {h: sum(1 for s in self.starts.values() if s["half"] == h) for h in "AB"}
        self.assertIn(f"{n} of {n} starts played, {n * 4} games", text)
        self.assertIn(f"half A {halves['A']} starts", text)
        self.assertIn(f"half B {halves['B']} /", text)
        for cell in ("act1-hallway", "act1-elite", "act1-boss", "act2-hallway", "act2-boss", "act1-boss-74 (ceremonial-beast)",
                     "act1-boss-77 (lagavulin-matriarch)", "act2-boss-75 (kaiser-crab)"):
            self.assertIn(cell, text)
        self.assertIn("calibration flags", text)
        self.assertIn("hp lost", text)
        self.assertIn("wasted blk", text)
        # every act1-hallway start is won on all draws, every act1-boss-74 start on none
        flag_lines = {ln.split()[0]: ln.split()[1:] for ln in text.splitlines() if ln.startswith(("act1-hallway ", "act1-boss ", "all "))}
        self.assertEqual(flag_lines["act1-hallway"][:3], ["12", "0", "0"])
        self.assertEqual(int(flag_lines["act1-boss"][1]), 12)  # boss 74's twelve starts are never won
        self.assertEqual(int(flag_lines["all"][0]), len(self.cal["always"]))
        self.assertEqual(int(flag_lines["all"][1]), len(self.cal["never"]))

    def test_the_precision_lines_give_each_halfs_rate_and_the_start_level_agreement(self):
        starts = {f"s{i}": {"cell": "act1-boss-74", "kind_cell": "act1-boss", "half": "AB"[i % 2], "id": f"s{i}", "kind": "boss"} for i in range(400)}
        rnd = random.Random(4)
        pt = {sid: rnd.choice((0.05, 0.95)) for sid in starts}  # starts differ a great deal: draws agree
        games = games_for(starts, lambda s: pt[s["id"]], rnd)
        p = fe.precision(games, starts, boot=60)
        self.assertGreater(p["split_r"], 0.8)
        self.assertEqual(p["split_starts"], 400)
        self.assertEqual(p["A"]["starts"], 200)
        self.assertAlmostEqual(p["A"]["unpaired_gap"], 1.96 * 2**0.5 * p["A"]["se"])
        flat = {sid: 0.5 for sid in starts}  # starts alike: draws are luck
        low = fe.precision(games_for(starts, lambda s: flat[s["id"]], rnd), starts, boot=20)
        self.assertLess(abs(low["split_r"]), 0.3)
        text = " ".join(fe.precision_lines(games, starts))
        self.assertIn("half A: win rate", text)
        self.assertIn("start-level agreement of draws 0-1 with draws 2-3", text)
        self.assertIn("precision of the exam on this player", fe.report_text(games, starts, "v2"))

    def test_the_weight_one_table_leaves_out_the_flagged_starts(self):
        w = fe.weights_of(self.cal, self.starts)
        rows = fe.cell_rows(self.games, self.starts, w)
        self.assertNotIn("act1-hallway", rows)
        self.assertNotIn("act1-boss-74", rows)
        self.assertEqual(rows["act1-boss-77"]["starts"], sum(1 for sid in self.cal["mixed"] if self.starts[sid]["cell"] == "act1-boss-77"))
        every = fe.cell_rows(self.games, self.starts)
        self.assertEqual(every["act1-boss"]["starts"], 24)
        self.assertEqual(every["act1-boss"]["games"], 96)
        self.assertEqual(every["act1-boss-74"]["won"], 0.0)

    def test_the_report_adds_the_baseline_agreement_when_there_is_one(self):
        base = {sid: {"p_win": 0.5, "l0": 0.5, "draws": 4} for sid in self.starts}
        self.assertIn("the judge's opening P(win)", fe.report_text(self.games, self.starts, "v2", self.cal, base))
        self.assertNotIn("the judge's opening P(win)", fe.report_text(self.games, self.starts, "v2", self.cal, None))

    def test_the_comparison_prints_the_raw_adjusted_and_cross_fitted_differences_and_decides_nothing(self):
        rnd = random.Random(33)
        p_c = {"act1-boss-77": 0.5, "act1-elite": 0.5, "act2-boss-75": 0.3}
        champ = games_for({s: v for s, v in self.starts.items() if v["cell"] in p_c}, lambda s: p_c[s["cell"]], rnd)
        sub = {s: v for s, v in self.starts.items() if v["cell"] in p_c}
        cand = games_for(sub, lambda s: p_c[s["cell"]] + 0.15, rnd)
        cal = fe.calibration(champ, sub)
        base = {sid: {"p_win": p_c[sub[sid]["cell"]], "l0": 0.5, "draws": 4} for sid in sub}
        text = fe.compare_text(champ, cand, sub, cal, base, "v2", "c0150", half="all", boot=60)
        for needle in ("McNemar on the games", "win rate difference, raw", "win rate difference, adjusted", "cross-fitted", "HP lost a fight",
                       "potions spent", "enemy HP removed per energy", "block wasted a fight", "act1-boss-77 (lagavulin-matriarch)", "points"):
            self.assertIn(needle, text)
        self.assertNotIn("promote", text.lower().replace("promotion rule", ""))
        text_a = fe.compare_text(champ, cand, sub, cal, None, "v2", "c0150", half="A", boot=20)
        self.assertIn("no baseline", text_a)
        self.assertIn("on half A", text_a)


# ---------------------------------------------------------------------------------------------------------- the observer


class ObserverTests(unittest.TestCase):
    def doc(self, hp, block, energy, turn, enemies):
        return {"player": {"hp": hp, "block": block, "energy": energy}, "turn": turn, "enemies": enemies}

    def test_the_digest_reads_hp_block_energy_and_the_announced_attack(self):
        d = fe.digest(self.doc(40, 6, 2, 3, [{"hp": 30, "intent": {"type": 0, "magnitude": 5, "hits": 2, "carries_damage": False}},
                                              {"hp": 12, "intent": {"type": 3, "magnitude": 9, "hits": 1, "carries_damage": False}},
                                              {"hp": 8, "intent": {"type": 4, "magnitude": 7, "hits": 1, "carries_damage": True}},
                                              {"hp": 0, "intent": {"type": 0, "magnitude": 99, "hits": 1, "carries_damage": False}}]))
        self.assertEqual(d, {"hp": 40, "block": 6, "energy": 2, "turn": 3, "ehp": 50, "incoming": 5 * 2 + 7 + 99})
        self.assertEqual(fe.digest({"player": {}, "enemies": []})["incoming"], 0)

    def test_dealt_energy_and_wasted_block_follow_the_steps(self):
        class Best:
            def __init__(self, kind):
                self.kind = kind

        looks = iter([
            {"hp": 50, "block": 0, "energy": 3, "turn": 1, "ehp": 40, "incoming": 10},  # play a card: attack, 1 energy, 8 dealt
            {"hp": 50, "block": 0, "energy": 2, "turn": 1, "ehp": 32, "incoming": 10},  # a potion: no energy
            {"hp": 50, "block": 0, "energy": 2, "turn": 1, "ehp": 32, "incoming": 10},  # a card: 2 energy, block 12 gained
            {"hp": 50, "block": 12, "energy": 0, "turn": 1, "ehp": 32, "incoming": 10},  # end turn: 12 block, 10 announced: 2 wasted
            {"hp": 50, "block": 0, "energy": 3, "turn": 2, "ehp": 30, "incoming": 6},  # next turn: 2 removed during the enemy phase
            {"hp": 44, "block": 5, "energy": 3, "turn": 2, "ehp": 30, "incoming": 6},  # end turn again: block 5 under 6 announced
        ])
        obs = fe.Observer()
        with mock.patch.object(fe.Observer, "look", staticmethod(lambda world: next(looks))):
            for kind in ("card", "potion", "card", "end_turn", "card", "end_turn"):
                obs(None, 0, Best(kind))
            looks2 = iter([{"hp": 44, "block": 0, "energy": 3, "turn": 3, "ehp": 4, "incoming": 0}])
            with mock.patch.object(fe.Observer, "look", staticmethod(lambda world: next(looks2))):
                obs.finish(None, False)
        s = obs.summary()
        self.assertEqual(s["energy"], 1.0 + 0.0 + 2.0)  # first card 1, the potion none, the card after it 2 (the third look to the fourth)
        self.assertEqual(s["wasted"], 2.0)
        self.assertEqual(s["end_turns"], 2)
        self.assertEqual(s["block_end"], 17.0)
        self.assertEqual(s["dealt"], 8.0 + 2.0 + 26.0)  # 40 -> 32, 32 -> 30 over the enemy phase, 30 -> 4 in the last step

    def test_an_unkillable_phase_and_an_unreadable_position_do_not_count_or_end_the_game(self):
        d = fe.digest(self.doc(40, 0, 3, 1, [{"hp": 1_000_000_000, "intent": {"type": 0, "magnitude": 1, "hits": 1}},
                                             {"hp": 25, "intent": {"type": 1, "magnitude": 0, "hits": 1}}]))
        self.assertEqual(d["ehp"], 25)

        class Best:
            kind = "card"

        reads = iter([{"hp": 50, "block": 0, "energy": 3, "turn": 1, "ehp": 40, "incoming": 0}, RuntimeError("stopped_step"),
                      {"hp": 50, "block": 0, "energy": 0, "turn": 1, "ehp": 31, "incoming": 0}])

        def look(world):
            r = next(reads)
            if isinstance(r, Exception):
                raise r
            return r

        obs = fe.Observer()
        with mock.patch.object(fe.Observer, "look", staticmethod(look)):
            for _ in range(3):
                obs(None, 0, Best())
        s = obs.summary()
        self.assertEqual((s["looks"], s["look_misses"]), (2, 1))
        self.assertEqual((s["dealt"], s["energy"]), (9.0, 3.0))  # the change across the unreadable decision is carried to the next reading

    def test_a_won_fight_removes_what_the_enemies_had_left(self):
        class Best:
            kind = "card"

        obs = fe.Observer()
        with mock.patch.object(fe.Observer, "look", staticmethod(lambda world: {"hp": 9, "block": 0, "energy": 3, "turn": 1, "ehp": 14, "incoming": 0})):
            obs(None, 0, Best())
        obs.finish(None, True)
        self.assertEqual(obs.summary()["dealt"], 14.0)


# ---------------------------------------------------------------------------------------------------------- the folder and the lock


class ExamFolderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.loop = t / "loop"
        (self.loop / "starts").mkdir(parents=True)
        self.pool = [{**s, "doc": {"x": s["id"], "hp": 3}} for s in synthetic_pool()]
        gl.write_start_set(self.loop / "starts", self.pool)
        gl.write_json(self.loop / "state.json", {"next_j": {s["id"]: 2 for s in self.pool[:30]}})
        self.out = t / "exam"
        self.board = {self.pool[100]["seed"], self.pool[101]["seed"]}
        self.p = mock.patch.object(gl, "board_run_seeds", lambda: self.board)
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def test_select_writes_the_exam_its_exclusion_list_and_the_meta_and_keeps_the_documents(self):
        meta = fe.build_exam(self.loop, self.out, per_cell=20)
        starts = fe.load_exam(self.out)
        self.assertEqual(meta["starts"], len(starts))
        self.assertEqual(meta["halves"], {"A": sum(1 for s in starts.values() if s["half"] == "A"), "B": sum(1 for s in starts.values() if s["half"] == "B")})
        self.assertEqual(meta["by_kind_cell"]["act1-boss"], 45)  # 20, 20 and the 5 there are; the board's two runs are not among them
        self.assertEqual(meta["by_cell"]["act1-boss-74"]["pool"], 68)
        self.assertIn("act1-boss-79", meta["short_cells"])
        ex = json.loads((self.out / "exclude_runs.json").read_text())
        self.assertEqual(ex["start_ids"], sorted(starts))
        self.assertTrue(set(ex["run_seeds"]).isdisjoint(self.board))
        self.assertEqual(set(ex["run_seeds"]), {s["seed"] for s in starts.values()})
        # the documents are in the exam's own copy, by the gym's byte index
        sid = sorted(starts)[3]
        self.assertEqual(gl.load_start(str(self.out / "starts.jsonl"), sid)["doc"]["x"], sid)
        self.assertEqual({s["id"] for s in starts.values()} & {s["id"] for s in self.pool[:30]}, {s["id"] for s in starts.values() if s["split"] == "val"})
        self.assertTrue(all(s["clean"] for s in starts.values()))

    def test_an_exam_is_not_replaced_without_rebuild_and_a_rebuild_keeps_the_old_files(self):
        fe.build_exam(self.loop, self.out, per_cell=10)
        with self.assertRaises(FileExistsError):
            fe.build_exam(self.loop, self.out, per_cell=10)
        fe.build_exam(self.loop, self.out, per_cell=5, rebuild=True)
        self.assertTrue(any(p.name.startswith("starts.jsonl.") and p.name.endswith(".bak") for p in self.out.iterdir()))
        self.assertTrue(any(p.name.startswith("exclude_runs.json.") for p in self.out.iterdir()))
        self.assertLess(len(fe.load_exam(self.out)), 60)

    def test_the_hook_takes_a_second_start_file_as_a_stratum_and_adds_its_runs_to_the_exclusion(self):
        extra = self.out.parent / "human.jsonl"
        recs = [{"id": f"human:H{i}|f10", "kind": "boss", "act": 2, "seed": f"H{i}", "encounter_id": 76, "doc": {"hp": 1}, "stratum": "human",
                 "hp": 30, "max_hp": 80, "node_type": 6, "floor": 33, "potions": 1} for i in range(8)]
        recs += [{"id": f"human:E{i}|f09", "kind": "elite", "act": 1, "seed": f"E{i}", "encounter_id": 62, "doc": {"hp": 1}, "hp": 30,
                  "max_hp": 80, "node_type": 2, "floor": 9, "potions": 0} for i in range(3)]
        extra.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        meta = fe.build_exam(self.loop, self.out, per_cell=10, extra_path=extra)
        starts = fe.load_exam(self.out)
        human = [s for s in starts.values() if s["stratum"] == "human"]
        self.assertEqual(len(human), 8)
        self.assertTrue(all(s["cell"] == "act2-boss-76" and s["half"] == fe.half_of(s["id"]) for s in human))
        self.assertEqual(meta["by_stratum"], {"ai": len(starts) - 11, "human": 8, "extra": 3})
        ex = json.loads((self.out / "exclude_runs.json").read_text())
        self.assertIn("H3", ex["run_seeds"])
        self.assertIn("human:H3|f10", ex["start_ids"])
        self.assertEqual(meta["by_cell"]["act2-boss-76"].get("extra"), 8)

    def test_a_bad_extra_record_is_refused(self):
        with self.assertRaises(ValueError):
            fe.pick_extra([{"id": "a", "kind": "boss", "act": 1, "seed": "s", "doc": {}}])  # a boss without its encounter id
        with self.assertRaises(ValueError):
            fe.pick_extra([{"id": "a", "kind": "elite", "act": 1, "seed": "s"}])  # no document
        ok = {"id": "a", "kind": "elite", "act": 1, "seed": "s", "doc": {}, "encounter_id": 3}
        with self.assertRaises(ValueError):
            fe.pick_extra([ok, dict(ok)])
        self.assertEqual(len(fe.pick_extra([ok])), 1)


# ---------------------------------------------------------------------------------------------------------- the human stratum


HUMAN_CELLS = ((1, "hallway", 11), (1, "elite", 62), (1, "boss", 74), (1, "boss", 77), (1, "boss", 79), (1, "boss", 82), (1, "boss", 83),
               (1, "boss", 84), (2, "hallway", 21), (2, "elite", 63), (2, "boss", 75), (2, "boss", 76), (2, "boss", 81))


def _run_ids(n: int, prefix: str = "H") -> list[str]:
    """Run ids outside the corpus's exam fifth (the stratum asserts that)."""
    import human_starts as hs  # noqa: PLC0415

    out, i = [], 0
    while len(out) < n:
        rid = f"{prefix}{i:05d}"
        if hs.exam_bucket(rid) != hs.EXAM_BUCKET:
            out.append(rid)
        i += 1
    return out


def human_pool(per_cell: int = 60, survive: float = 0.7, seed: int = 4) -> tuple[list[dict], dict]:
    """Human start records (lite plus a document) for every cell of the stratum and the sidecar's outcomes for them."""
    rnd = random.Random(seed)
    runs = iter(_run_ids(per_cell * len(HUMAN_CELLS)))
    recs, outcomes = [], {}
    for act, kind, enc in HUMAN_CELLS:
        for _ in range(per_cell):
            rid = next(runs)
            sid = f"human:{rid}|f{10 + act}"
            recs.append({"id": sid, "cat": f"act{act}-{kind}", "kind": kind, "act": act, "floor": 10 + act, "seed": rid, "run": rid, "ix": -1,
                         "night": "human", "source": "human", "k": 0, "node_type": 1, "hp": 50, "max_hp": 80, "deck_size": 12, "potions": 1,
                         "relics": 2, "encounter_id": enc, "turn": 0, "kind_same": True, "replay_same": None, "split": "train", "doc": {"x": sid}})
            ok = rnd.random() < survive
            lost = rnd.randint(5, 40) if ok else 50
            outcomes[sid] = {"human_survived": ok, "human_end_hp": 50 - lost if ok else 0, "human_hp_lost": lost, "human_damage_taken": lost,
                             "human_post_heal": 6 if ok else 0, "human_turns": 5, "human_potions_used": 0, "human_potions_held_end": 1,
                             "human_l0": float(50 - lost + 10) if ok else -100.0, "human_l0_unit": 0.8 if ok else 0.0}
    return recs, outcomes


def write_human_files(starts_dir: Path, recs: list[dict], outcomes: dict) -> None:
    fe.write_stem_set(starts_dir, "human-starts", recs)
    (starts_dir / "human-outcomes.json").write_text(json.dumps({"rule": "human-outcomes-v1", "written": "2026-11-08T00:00:00Z", "outcomes": outcomes}),
                                                    encoding="utf-8")


class HumanQuotaTests(unittest.TestCase):
    def test_equal_quotas_then_the_remainder_in_a_fixed_hash_order(self):
        avail = {f"c{i:02d}": 1000 for i in range(13)}
        q = fe.human_quota(avail, 50, 400, 7)
        self.assertEqual(sum(q.values()), 400)
        self.assertEqual(sorted(set(q.values())), [30, 31])
        self.assertEqual(sum(1 for v in q.values() if v == 31), 10)
        self.assertEqual(q, fe.human_quota(avail, 50, 400, 7))
        self.assertNotEqual(q, fe.human_quota(avail, 50, 400, 8))  # a different seed names other cells for the remainder

    def test_a_small_cell_takes_what_it_has_and_the_others_share_the_rest(self):
        avail = {"a": 4, "b": 1000, "c": 1000, "d": 1000}
        q = fe.human_quota(avail, 50, 100, 1)
        self.assertEqual(q["a"], 4)
        self.assertEqual(sum(q.values()), 100)
        self.assertTrue(all(32 <= q[c] <= 33 for c in "bcd"), q)

    def test_the_per_cell_cap_and_a_total_nobody_can_fill(self):
        self.assertEqual(set(fe.human_quota({c: 500 for c in "abcd"}, 50, 10_000, 1).values()), {50})
        self.assertEqual(sum(fe.human_quota({"a": 3, "b": 2}, 50, 400, 1).values()), 5)
        self.assertEqual(fe.human_quota({}, 50, 400, 1), {})


class HumanPickTests(unittest.TestCase):
    def setUp(self):
        self.recs, self.outcomes = human_pool(60)
        self.lite = [{k: v for k, v in r.items() if k != "doc"} for r in self.recs]

    def test_the_pick_is_clean_one_start_a_run_fixed_by_its_seed_and_carries_the_humans_result(self):
        played = {self.recs[0]["run"], self.recs[1]["run"], self.recs[200]["run"]}
        chosen, report, left_out = fe.pick_human(self.lite, self.outcomes, played=played, per_cell=50, total=400)
        self.assertEqual(len(chosen), 400)
        self.assertEqual(len({s["run"] for s in chosen}), 400)
        self.assertTrue(all(s["run"] not in played for s in chosen))
        self.assertEqual(left_out["its run was played by the loop"], 3)
        self.assertEqual({s["stratum"] for s in chosen}, {"human"})
        self.assertTrue(all(s["clean"] and s["practised"] == 0 and s["half"] == fe.half_of(s["id"]) for s in chosen))
        s = chosen[0]
        for k, v in self.outcomes[s["id"]].items():
            self.assertEqual(s[k], v)
        self.assertEqual(s["cell"], fe.cell_of(s))
        self.assertEqual(sorted(report), sorted(f"act{a}-{k}-{e}" if k == "boss" else f"act{a}-{k}" for a, k, e in HUMAN_CELLS))
        per = Counter(s["cell"] for s in chosen)
        self.assertEqual(set(per.values()), {30, 31})
        self.assertTrue(all(not v["short"] for v in report.values()))
        again, _, _ = fe.pick_human(list(reversed(self.lite)), self.outcomes, played=played, per_cell=50, total=400)
        self.assertEqual(sorted(s["id"] for s in again), sorted(s["id"] for s in chosen))  # the pool's order does not matter
        other, _, _ = fe.pick_human(self.lite, self.outcomes, played=played, per_cell=50, total=400, seed=1)
        self.assertNotEqual(sorted(s["id"] for s in other), sorted(s["id"] for s in chosen))

    def test_a_start_without_an_outcome_or_of_a_played_run_is_never_taken_and_a_short_cell_is_reported(self):
        outcomes = {k: v for k, v in self.outcomes.items() if k != self.recs[300]["id"]}  # one start with no outcome (not a boss-74 one)
        boss74 = sorted(r["run"] for r in self.recs if r["kind"] == "boss" and r["encounter_id"] == 74)
        played = set(boss74[5:])  # five runs of the boss cell stay clean
        chosen, report, left_out = fe.pick_human(self.lite, outcomes, played=played, per_cell=50, total=400)
        self.assertEqual(report["act1-boss-74"]["chosen"], 5)
        self.assertTrue(report["act1-boss-74"]["short"])
        self.assertEqual(sum(1 for s in chosen if s["cell"] == "act1-boss-74"), 5)
        self.assertEqual(left_out["no outcome in the sidecar"], 1)
        self.assertEqual(left_out["its run was played by the loop"], len(boss74) - 5)
        self.assertTrue(all(not v["short"] for c, v in report.items() if c != "act1-boss-74"))
        self.assertTrue(all(s["id"] in outcomes for s in chosen))

    def test_the_loops_evidence_is_the_start_counts_and_the_plays_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "starts").mkdir()
            gl.write_json(root / "state.json", {"next_j": {"human:RA|f05": 2, "human:RB|f05": 0, "2026-11-01/g9:SEED1|f17": 1,
                                                          "preloss:human:RC|f09|j0|k4": 1}})
            (root / "starts" / "plays.jsonl").write_text(json.dumps({"start": "human:RD|f33", "j": 0, "root": 1.0, "played": 2.0}) + "\n"
                                                         + json.dumps({"shard": "s1.npz", "done": True}) + "\n{cut", encoding="utf-8")
            self.assertEqual(fe.played_runs(root), {"RA", "SEED1", "RC", "RD"})  # RB was drawn into a task list and never played: next_j 0
            self.assertEqual(fe.played_runs(root / "nowhere"), set())


class HumanFolderTests(unittest.TestCase):
    """`add-human` on a loop folder with an AI exam: the second file, the exclusion list and the proof that the AI exam did not move."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.loop = t / "loop"
        (self.loop / "starts").mkdir(parents=True)
        self.pool = [{**s, "doc": {"x": s["id"], "hp": 3}} for s in synthetic_pool()]
        gl.write_start_set(self.loop / "starts", self.pool)
        gl.write_json(self.loop / "state.json", {"next_j": {s["id"]: 2 for s in self.pool[:30]}})
        self.out = t / "exam"
        with mock.patch.object(gl, "board_run_seeds", lambda: set()):
            fe.build_exam(self.loop, self.out, per_cell=15)
        self.recs, self.outcomes = human_pool(60)
        write_human_files(self.loop / "starts", self.recs, self.outcomes)
        self.exam_files = [self.out / n for n in ("starts.jsonl", "starts.index.json", "starts.lite.json", "meta.json")]

    def tearDown(self):
        self.tmp.cleanup()

    def sums(self):
        return {f.name: gl.sha256_file(f) for f in self.exam_files}

    def test_the_stratum_is_a_second_file_and_the_ai_exam_is_byte_for_byte_what_it_was(self):
        before, ai_before = self.sums(), fe.load_exam(self.out)
        ex_before = json.loads((self.out / "exclude_runs.json").read_text())
        meta = fe.build_human(self.loop, self.out, exam_ids=[])
        self.assertEqual(self.sums(), before)  # starts, index, lite list and meta of the AI exam: untouched
        self.assertEqual(fe.load_exam(self.out), ai_before)
        self.assertEqual(meta["ai_files_sha256"], {k: v for k, v in before.items()})
        human = fe.load_human_exam(self.out)
        self.assertEqual((meta["starts"], len(human), meta["runs"]), (400, 400, 400))
        self.assertTrue(set(human).isdisjoint(ai_before))
        self.assertEqual({s["stratum"] for s in human.values()}, {"human"})
        self.assertEqual(meta["halves"], dict(Counter(s["half"] for s in human.values())))
        self.assertEqual(set(meta["by_kind_cell"]), {"act1-hallway", "act1-elite", "act1-boss", "act2-hallway", "act2-elite", "act2-boss"})
        doc = gl.load_start(str(self.out / "starts-human.jsonl"), sorted(human)[3])  # the document is the exam's own copy, by the byte index
        self.assertEqual(doc["doc"]["x"], sorted(human)[3])
        self.assertEqual(doc["stratum"], "human")
        self.assertIn("human_survived", doc)
        ex = json.loads((self.out / "exclude_runs.json").read_text())
        self.assertTrue(set(ex_before["run_seeds"]) <= set(ex["run_seeds"]))  # the old lists stay whole
        self.assertTrue(set(ex_before["start_ids"]) <= set(ex["start_ids"]))
        self.assertTrue({s["run"] for s in human.values()} <= set(ex["run_seeds"]))
        self.assertTrue(set(human) <= set(ex["start_ids"]))
        self.assertEqual(ex["by_stratum"]["human"], 400)
        self.assertEqual(ex["by_stratum"]["ai"], ex_before["by_stratum"]["ai"])
        self.assertEqual(json.loads((self.out / "exclude_runs.json.pre-human.bak").read_text()), ex_before)
        self.assertEqual(json.loads((self.out / "human-meta.json").read_text())["rule"], fe.HUMAN_RULE)
        # the sampler's own reader (<loop root>/exam/exclude_runs.json) drops the stratum's runs and starts
        import practice_mix as pm  # noqa: PLC0415

        excl = pm.read_exclusion(self.out.parent)
        self.assertTrue(all(pm.excluded({"id": s["id"], "seed": s["seed"], "run": s["run"]}, excl) for s in human.values()))

    def test_the_readings_of_the_ai_exam_do_not_change(self):
        starts = fe.load_exam(self.out)
        rnd = random.Random(8)
        p = {c: 0.3 + 0.05 * i for i, c in enumerate(sorted({s["cell"] for s in starts.values()}))}
        games = games_for(starts, lambda s: p[s["cell"]], rnd)
        with open(self.out / "games-v2.jsonl", "w", encoding="utf-8") as fh:
            for r in games.values():
                fh.write(json.dumps({**r, "config": "a3-v1", "prior_path": "server", "engine": "abc"}) + "\n")
        before = self.report("v2")
        fe.build_human(self.loop, self.out, exam_ids=[])
        self.assertEqual(self.report("v2"), before)  # the same bytes with the stratum added and no game of it played
        human = fe.load_human_exam(self.out)
        hg = games_for(human, lambda s: 0.9 if s["human_survived"] else 0.4, random.Random(9))
        with open(self.out / "games-v2.jsonl", "a", encoding="utf-8") as fh:
            for r in hg.values():
                fh.write(json.dumps({**r, "config": "a3-v1", "prior_path": "server", "engine": "abc", "stratum": "human"}) + "\n")
        after = self.report("v2")
        self.assertTrue(after.startswith(before.rstrip("\n")), "the exam's own report must come first, unchanged")
        tail = after[len(before.rstrip("\n")):]
        self.assertIn("fight exam, human stratum, player v2: 400 of 400 starts played, 1600 games", tail)
        self.assertNotIn("1600 games", before)
        self.assertIn(f"{len(starts)} of {len(starts)} starts played, {len(games)} games", after)  # the header counts the AI exam's games only

    def report(self, player: str) -> str:
        import argparse  # noqa: PLC0415
        import contextlib  # noqa: PLC0415
        import io  # noqa: PLC0415

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = fe.cmd_report(argparse.Namespace(exam_dir=str(self.out), player=player, write_calibration=False, boot=60))
        self.assertEqual(rc, 0)
        return buf.getvalue()

    def test_the_comparison_prints_the_exams_own_lines_first_and_the_human_lines_after(self):
        import argparse  # noqa: PLC0415
        import contextlib  # noqa: PLC0415
        import io  # noqa: PLC0415

        starts = fe.load_exam(self.out)
        fe.build_human(self.loop, self.out, exam_ids=[])
        human = fe.load_human_exam(self.out)
        rnd = random.Random(2)
        for label, bump in (("v2", 0.0), ("c1", 0.1)):
            ga = games_for(starts, lambda s: min(0.95, 0.5 + bump), rnd)
            gb = games_for(human, lambda s: min(0.95, (0.85 if s["human_survived"] else 0.4) + bump), rnd)
            with open(self.out / f"games-{label}.jsonl", "w", encoding="utf-8") as fh:
                for r in list(ga.values()) + list(gb.values()):
                    fh.write(json.dumps({**r, "config": "a3-v1", "prior_path": "server", "engine": "abc"}) + "\n")

        def run(fn, **kw):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = fn(argparse.Namespace(exam_dir=str(self.out), **kw))
            return rc, buf.getvalue()

        run(fe.cmd_report, player="v2", write_calibration=True, boot=40)
        rc, text = run(fe.cmd_compare, candidate="c1", champion="v2", no_play=True, boot=40)
        self.assertEqual(rc, 0)
        i, j = text.index("fight exam, c1 against v2 on half A"), text.index("human stratum, half A")
        self.assertLess(i, j)
        tail = text[j:]
        self.assertIn("c1 minus v2 on the human stratum", tail)
        self.assertIn("fight exam, human stratum, player c1", tail)
        self.assertIn("fight exam, human stratum, player v2", tail)
        self.assertIn("win rate difference", tail)
        self.assertNotIn("human", text[:j])  # the exam's own comparison says nothing of the stratum

    def test_the_stratum_is_not_replaced_without_rebuild_and_a_rebuild_keeps_the_old_files(self):
        fe.build_human(self.loop, self.out, exam_ids=[])
        with self.assertRaises(FileExistsError):
            fe.build_human(self.loop, self.out, exam_ids=[])
        fe.build_human(self.loop, self.out, per_cell=5, total=20, rebuild=True, exam_ids=[])
        self.assertTrue(any(p.name.startswith("starts-human.jsonl.") and p.name.endswith(".bak") for p in self.out.iterdir()))
        self.assertTrue(any(p.name.startswith("human-meta.json.") for p in self.out.iterdir()))
        self.assertEqual(len(fe.load_human_exam(self.out)), 20)

    def test_no_exam_no_outcomes_and_an_exam_fifth_run_stop_it(self):
        with self.assertRaises(FileNotFoundError):
            fe.build_human(self.loop, self.out.parent / "none", exam_ids=[])
        (self.loop / "starts" / "human-outcomes.json").write_text(json.dumps({"outcomes": {}}))
        with self.assertRaises(FileNotFoundError):
            fe.build_human(self.loop, self.out, exam_ids=[])
        write_human_files(self.loop / "starts", self.recs, self.outcomes)
        with self.assertRaises(AssertionError):
            fe.build_human(self.loop, self.out, exam_ids=[self.recs[7]["run"]])  # a run the corpus keeps for its exam: never a start
        self.assertFalse((self.out / "starts-human.jsonl").exists())
        self.assertFalse((self.out / "exclude_runs.json.pre-human.bak").exists())

    def test_the_games_to_play_include_the_stratum_from_its_own_file_and_the_stratum_option_restricts(self):
        fe.build_human(self.loop, self.out, per_cell=3, total=39, exam_ids=[])
        starts, human = fe.load_exam(self.out), fe.load_human_exam(self.out)
        t = fe.exam_tasks(starts, "v2", "v2", "a3-v1", self.out, set(), human=human)
        self.assertEqual(len(t), (len(starts) + len(human)) * fe.DRAWS)
        for arm, sid, j, cat, info in t:
            self.assertEqual(Path(info["starts"]).name, "starts-human.jsonl" if sid in human else "starts.jsonl")
            self.assertEqual(info["ns"], "x")
        only = fe.exam_tasks({}, "v2", "v2", "a3-v1", self.out, set(), human=human, half="A")
        self.assertTrue(all(t[1] in human and human[t[1]]["half"] == "A" for t in only))
        self.assertEqual(fe.exam_tasks(starts, "v2", "v2", "a3-v1", self.out, set()), [x for x in t if x[1] in starts])
        order = [(starts.get(x[1]) or human[x[1]])["kind"] for x in t]
        self.assertEqual(order, sorted(order, key={"boss": 0, "elite": 1, "hallway": 2}.get))  # long fights first across both files
        import argparse  # noqa: PLC0415

        a = argparse.Namespace(exam_dir=str(self.out.parent / "empty"), stratum="human", config="a3-v1")
        (self.out.parent / "empty").mkdir()
        gl.write_start_set(self.out.parent / "empty", [])
        self.assertEqual(fe.run_player(a, "v2", Path("p"), "all", self.out.parent / "g.jsonl"), 2)  # no human stratum there: nothing to play


class HumanReadingTests(unittest.TestCase):
    def setUp(self):
        recs, outcomes = human_pool(40, survive=0.5, seed=3)
        chosen, _, _ = fe.pick_human([{k: v for k, v in r.items() if k != "doc"} for r in recs], outcomes, played=set(), per_cell=25, total=400)
        self.human = {s["id"]: s for s in chosen}

    def test_the_player_is_read_against_the_human_on_the_same_starts_by_kind_and_boss(self):
        rnd = random.Random(1)
        games = games_for(self.human, lambda s: 0.9 if s["human_survived"] else 0.5, rnd,
                          tweak=lambda r, s, j: r.update(hp_lost=float(s["human_hp_lost"] - 4) if r["won"] else 50.0))
        items = fe.human_items(games, self.human)
        rows = {r["label"]: r for r in fe.human_rows(items, boot=100)}
        allr = rows["all"]
        n = len(self.human)
        self.assertEqual(allr["starts"], n)
        self.assertEqual(allr["games"], n * fe.DRAWS)
        sm = fe.start_means(games, self.human)
        self.assertAlmostEqual(allr["p_won"], sum(m["won"] for m in sm.values()) / n)
        self.assertAlmostEqual(allr["h_won"], sum(1 for s in self.human.values() if s["human_survived"]) / n)
        self.assertAlmostEqual(allr["d_won"]["estimate"], allr["p_won"] - allr["h_won"])
        self.assertAlmostEqual(allr["d_hp_lost"]["estimate"], allr["p_hp_lost"] - allr["h_hp_lost"])
        self.assertLessEqual(allr["d_won"]["lo"], allr["d_won"]["estimate"])
        self.assertGreaterEqual(allr["d_won"]["hi"], allr["d_won"]["estimate"])
        self.assertAlmostEqual(allr["h_hp_lost"], sum(s["human_hp_lost"] for s in self.human.values()) / n)
        self.assertEqual({"act1-hallway", "act1-elite", "act1-boss", "act2-hallway", "act2-elite", "act2-boss", "all"} <= set(rows), True)
        self.assertEqual(rows["act1-boss"]["starts"], 6 * 25)
        self.assertEqual(rows["act1-boss-74"]["starts"], 25)
        text = fe.human_report_text(games, self.human, "v2", boot=60)
        self.assertIn("fight exam, human stratum, player v2: ", text)
        self.assertIn("act1-boss-74 (ceremonial-beast)", text)
        self.assertIn("act2-boss-76 (knowledge-demon)", text)
        self.assertIn("a death costs the whole HP going in on both sides", text)
        self.assertNotIn("act1-hallway-", text)  # only a boss is split further

    def test_a_per_boss_line_under_the_minimum_is_not_printed_and_the_counts_are_the_starts_played(self):
        small = {sid: s for sid, s in self.human.items() if s["cell"] != "act1-boss-74" or sid in sorted(x for x in self.human if self.human[x]["cell"] == "act1-boss-74")[:10]}
        games = games_for(small, lambda s: 0.6, random.Random(2))
        rows = [r["label"] for r in fe.human_rows(fe.human_items(games, small), boot=20)]
        self.assertNotIn("act1-boss-74", rows)
        self.assertIn("act1-boss-77", rows)
        half = {k: v for k, v in games.items() if k[1] < 2}
        text = fe.human_report_text(half, small, "v2", boot=20)
        self.assertIn(f"{len(small)} of {len(small)} starts played, {len(small) * 2} games", text)
        self.assertEqual(fe.human_report_text({}, small, "v2"), f"fight exam, human stratum, player v2: 0 of {len(small)} starts played, 0 games "
                         f"(half A 0 starts, half B 0); human survived on 0 of the played starts")

    def test_the_comparison_pairs_the_candidate_with_the_champion_on_the_same_starts(self):
        rnd = random.Random(3)
        champ = games_for(self.human, lambda s: 0.5, rnd)
        cand = games_for(self.human, lambda s: 0.8, rnd)
        text = fe.human_compare_text(champ, cand, self.human, "v2", "c1", half="A", boot=60)
        self.assertIn("human stratum, half A:", text)
        self.assertIn("c1 minus v2 on the human stratum", text)
        self.assertRegex(text, r"win rate difference +: \+\d+\.\d+ points \[")
        self.assertIn("act1-boss", text)
        none = fe.human_compare_text({}, cand, self.human, "v2", "c1", half="A", boot=20)
        self.assertIn("no start of the human stratum has games of both c1 and v2 yet", none)
        self.assertEqual(fe.human_compare_text(champ, cand, {}, "v2", "c1"), "human stratum: none in the exam")

    def test_a_tag_named_with_allow_quiet_does_not_hold_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = Path(tmp) / "QUIET.lock"
            q.write_text(f"lab-score {gl.stamp()} pid=5\n")
            with mock.patch.object(fg, "QUIET", q), mock.patch.object(gl, "in_night_window", lambda now=None: False), \
                    mock.patch.object(fg, "ACTIVE", Path(tmp) / "none"), mock.patch.object(gl, "PAUSE", Path(tmp) / "pause"), \
                    mock.patch.object(gl, "HALT", Path(tmp) / "halt"), mock.patch.object(fg, "MAIN_LOCK", Path(tmp) / "main.lock"):
                self.assertEqual(fe.blocked_reason(starting=True), "quiet:lab-score")
                with mock.patch.object(fe, "ALLOW_QUIET", ["lab-score"]):
                    self.assertIsNone(fe.blocked_reason(starting=True))
                self.assertEqual(fe.blocked_reason(starting=True), "quiet:lab-score")  # the default is unchanged


def det_games(starts: dict, win, hp, draws: int = 4, **extra) -> dict:
    """Deterministic games: `win(start, j)` and `hp(start, j)` (HP lost), rows as the exam writes them."""
    out = {}
    for sid, s in starts.items():
        for j in range(draws):
            w = bool(win(s, j))
            out[(sid, j)] = {"start": sid, "j": j, "won": w, "capped": False, "L0": 0.7 if w else 0.0,
                             "hp_lost": float(hp(s, j) if w else 50), "config": "a3-v1", "stratum": "human", **extra}
    return out


def human_gym_folder(root: Path, per_cell: int = 3, games: bool = True) -> tuple[Path, Path, str]:
    """A gym-loop folder for the lab's human-bar line: player v2's file is the bytes `net two`, the exam holds a human stratum of 13 x
    `per_cell` starts and, with `games`, v2's games on it (engine e1; it wins the starts the human survived, 12 HP lost). Returns
    (loop folder, exam folder, the sha256 of v2's network)."""
    import hashlib  # noqa: PLC0415

    loop = Path(root) / "gym"
    exam = loop / "exam"
    exam.mkdir(parents=True)
    (loop / "players" / "v2").mkdir(parents=True)
    (loop / "players" / "v2" / "combat_policy.pt").write_bytes(b"net two")
    recs, outcomes = human_pool(per_cell + 10, survive=0.6, seed=2)
    chosen, _, _ = fe.pick_human([{k: v for k, v in r.items() if k != "doc"} for r in recs], outcomes, played=set(), per_cell=per_cell,
                                 total=13 * per_cell)
    fe.write_stem_set(exam, "starts-human", [{**s, "doc": {"x": s["id"]}} for s in chosen])
    if games:
        rows = det_games({s["id"]: s for s in chosen}, lambda s, j: s["human_survived"], lambda s, j: 12, engine="e1")
        (exam / "games-v2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows.values()), encoding="utf-8")
    return loop, exam, hashlib.sha256(b"net two").hexdigest()


class HumanBarReadingTests(unittest.TestCase):
    """The human line of one judgement (D55 addendum 11): candidate, incumbent and human on the same starts, report-only."""

    def setUp(self):
        recs, outcomes = human_pool(40, survive=0.5, seed=5)
        chosen, _, _ = fe.pick_human([{k: v for k, v in r.items() if k != "doc"} for r in recs], outcomes, played=set(), per_cell=25, total=400)
        self.human = {s["id"]: s for s in chosen}
        self.n = len(self.human)
        self.order = sorted(self.human)
        self.rank = {sid: i for i, sid in enumerate(self.order)}

    def games(self, every: int, hp_lost: int, draws: int = 4, **extra) -> dict:
        """Wins on all draws of the starts whose rank is not a multiple of `every`, on none of the others."""
        return det_games(self.human, lambda s, j: self.rank[s["id"]] % every != 0, lambda s, j: hp_lost, draws, **extra)

    def test_the_reading_is_the_arithmetic_of_the_same_starts_and_the_line_has_the_agreed_shape(self):
        cand, inc = self.games(3, 20), self.games(5, 14)
        r = fe.human_bar_reading(cand, inc, self.human, boot=200)
        n = self.n
        wins = lambda k: sum(1 for i in range(n) if i % k != 0)  # noqa: E731
        self.assertEqual((r["starts"], r["stratum_starts"], r["cand_games"], r["inc_games"], r["complete"]), (n, n, 4 * n, 4 * n, True))
        self.assertAlmostEqual(r["cand_surv"], 100 * wins(3) / n, delta=0.01)
        self.assertAlmostEqual(r["inc_surv"], 100 * wins(5) / n, delta=0.01)
        self.assertAlmostEqual(r["human_surv"], 100 * sum(1 for s in self.human.values() if s["human_survived"]) / n, delta=0.01)
        self.assertAlmostEqual(r["diff_surv"], r["cand_surv"] - r["inc_surv"], delta=0.02)
        self.assertLessEqual(r["diff_surv_lo"], r["diff_surv"])
        self.assertGreaterEqual(r["diff_surv_hi"], r["diff_surv"])
        self.assertEqual(r["diff_starts"], n)
        # a death costs the whole HP going in (50): HP lost is the mean over the starts of 50 (a loss) or the HP lost (a win)
        self.assertAlmostEqual(r["cand_hp_lost"], (50 * (n - wins(3)) + 20 * wins(3)) / n, delta=0.011)
        self.assertAlmostEqual(r["inc_hp_lost"], (50 * (n - wins(5)) + 14 * wins(5)) / n, delta=0.011)
        self.assertAlmostEqual(r["human_hp_lost"], sum(s["human_hp_lost"] for s in self.human.values()) / n, delta=0.011)
        self.assertAlmostEqual(r["diff_hp_lost"], r["cand_hp_lost"] - r["inc_hp_lost"], delta=0.02)
        line = fe.human_bar_line(r, "v2")
        self.assertRegex(line, rf"^human bar: cand \d+\.\d / inc \d+\.\d / human \d+\.\d in 100 on {n} starts; cand-inc [+-]\d+\.\d "
                               r"\[[+-]\d+\.\d, [+-]\d+\.\d\]; HP lost \d+\.\d / \d+\.\d / \d+\.\d$")
        self.assertNotIn("engines differ", line)  # no engine field on these rows: nothing to differ
        self.assertEqual(fe.human_bar_reading(cand, inc, self.human, boot=200), r)  # fixed by its bootstrap seed

    def test_the_reading_is_on_the_starts_and_draws_both_played_and_never_leaves_the_stratum(self):
        cand = self.games(3, 20)
        inc = {k: v for k, v in self.games(5, 14).items() if self.rank[k[0]] < 100 and k[1] < 2}  # 100 starts, two draws
        alien = {("other|f1", 0): {"start": "other|f1", "j": 0, "won": True, "capped": False, "hp_lost": 0.0}}
        r = fe.human_bar_reading({**cand, **alien}, {**inc, **alien}, self.human, boot=50)
        self.assertEqual((r["starts"], r["diff_starts"], r["cand_games"], r["inc_games"], r["complete"]), (100, 100, 4 * self.n, 200, True))
        sub = {s: self.human[s] for s in self.order if self.rank[s] < 100}
        again = fe.human_bar_reading({k: v for k, v in cand.items() if k[0] in sub and k[1] < 2}, inc, sub, boot=50)
        for key in ("cand_surv", "inc_surv", "human_surv", "cand_hp_lost", "inc_hp_lost", "human_hp_lost", "diff_surv"):
            self.assertEqual(r[key], again[key], key)  # the same starts and draws, the same numbers
        self.assertIn(f"on 100 of {self.n} starts", fe.human_bar_line(r))

    def test_an_incumbent_with_no_game_leaves_its_side_out_and_says_so(self):
        cand = self.games(3, 20)
        for inc in (None, {}, {("alien|f1", 0): {"start": "alien|f1", "j": 0, "won": True, "capped": False}}):
            r = fe.human_bar_reading(cand, inc, self.human, boot=50)
            self.assertEqual((r["starts"], r["inc_games"], r["same_engine"]), (self.n, 0, None))
            self.assertNotIn("inc_surv", r)
            self.assertNotIn("diff_surv", r)
            line = fe.human_bar_line(r, "v2")
            self.assertIn("(no human-stratum games for the incumbent v2)", line)
            self.assertNotIn("cand-inc", line)
        # the candidate alone is the stratum reading of `human_items`
        items = fe.human_items(cand, self.human)
        self.assertAlmostEqual(r["cand_surv"], 100 * fe.mean(it["p_won"] for v in items.values() for it in v), delta=0.01)
        # games that never meet (the incumbent's draws are not the candidate's) fall back the same way
        meet = {k: v for k, v in self.games(5, 14).items() if k[1] == 3}
        r = fe.human_bar_reading({k: v for k, v in cand.items() if k[1] < 3}, meet, self.human, boot=50)
        self.assertEqual((r["inc_games"], r["starts"], r["cand_games"]), (0, self.n, 3 * self.n))
        none = fe.human_bar_reading({}, cand, self.human)
        self.assertEqual(none["starts"], 0)
        self.assertEqual(fe.human_bar_line(none), "human bar: no human-stratum games of the candidate")

    def test_engines_that_differ_are_named_and_a_partial_play_is_marked(self):
        cand, inc = self.games(3, 20, engine="e1"), self.games(5, 14, engine="e0")
        r = fe.human_bar_reading(cand, inc, self.human, boot=30)
        self.assertEqual((r["cand_engine"], r["inc_engine"], r["same_engine"]), (["e1"], ["e0"], False))
        self.assertTrue(fe.human_bar_line(r).endswith("; engines differ (cand e1, inc e0)"))
        same = fe.human_bar_reading(cand, {k: {**v, "engine": "e1"} for k, v in inc.items()}, self.human, boot=30)
        self.assertTrue(same["same_engine"])
        half = {k: v for k, v in cand.items() if k[1] < 2}
        self.assertFalse(fe.human_bar_reading(half, inc, self.human, boot=30)["complete"])
        self.assertTrue(fe.human_bar_reading(half, inc, self.human, boot=30, draws=2)["complete"])

    def test_games_of_another_config_are_not_the_strata_reading(self):
        rows = list(self.games(3, 20).values()) + [{**r, "config": "judge-v0"} for r in self.games(2, 5).values()]
        only = fe.human_games(rows, self.human, "a3-v1")
        self.assertEqual(len(only), 4 * self.n)
        self.assertTrue(all(r["config"] == "a3-v1" for r in only.values()))
        self.assertEqual(len(fe.human_games(rows, {}, "a3-v1")), 0)

    # -- the lab's judgement line
    def lab_folder(self, tmp: Path, hp: int = 18) -> tuple[Path, Path, str]:
        """A loop folder with players v1 and v2 (v2's bytes are the net), an exam with the human stratum and v2's games on it."""
        import hashlib  # noqa: PLC0415

        loop = tmp / "loop"
        exam = loop / "exam"
        exam.mkdir(parents=True)
        for name, blob in (("v1", b"net one"), ("v2", b"net two")):
            (loop / "players" / name).mkdir(parents=True)
            (loop / "players" / name / "combat_policy.pt").write_bytes(blob)
        fe.write_stem_set(exam, "starts-human", [{**s, "doc": {"x": s["id"]}} for s in self.human.values()])
        rows = list(self.games(3, hp, engine="e1").values())
        (exam / "games-v2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return loop, exam, hashlib.sha256(b"net two").hexdigest()

    def test_the_keep_tests_line_names_the_network_and_its_gym_player_and_reads_the_exams_games(self):
        import hashlib  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            loop, exam, sha = self.lab_folder(Path(tmp))
            note = fe.lab_human_bar(sha, exam, loop)
            self.assertTrue(note["line"].startswith(f"human bar: fight network {sha[:12]} (gym player v2): survival "), note["line"])
            n = self.n
            want = 100 * sum(1 for i in range(n) if i % 3 != 0) / n
            self.assertIn(f"survival {want:.1f} / human ", note["line"])
            self.assertIn("HP lost ", note["line"])
            self.assertIn(f"on {n} of {n} starts, {4 * n} games, engine e1, games file written ", note["line"])
            self.assertRegex(note["line"], r"games file written \d{4}-\d{2}-\d{2}$")
            self.assertEqual((note["gym_player"], note["reading_of"], note["network"]), ("v2", "v2", sha[:12]))
            self.assertEqual(note["date"], fe.games_date(exam / "games-v2.jsonl"))
            # a network that is a gym player's but has no human-stratum games
            one = fe.lab_human_bar(hashlib.sha256(b"net one").hexdigest(), exam, loop)
            self.assertEqual(one["line"], f"human bar: no human-stratum games for fight network {one['network']} (gym player v1)")
            # a network that is no gym player's: the incumbent's reading, labelled as the incumbent's
            gl.write_json(loop / "state.json", {"incumbent": "v2"})
            odd = fe.lab_human_bar("ab" * 32, exam, loop)
            self.assertIn("(no gym player holds these bytes); the gym incumbent v2's reading, not this network's: survival ", odd["line"])
            self.assertEqual((odd["gym_player"], odd["reading_of"]), (None, "v2"))
            (exam / "games-v2.jsonl").unlink()
            self.assertEqual(fe.lab_human_bar("ab" * 32, exam, loop)["line"],
                             f"human bar: no human-stratum games for fight network {'ab' * 6} (no gym player holds these bytes); "
                             "the gym incumbent v2 has none either")
            self.assertEqual(fe.lab_human_bar(sha, exam, loop)["line"], f"human bar: no human-stratum games for fight network {sha[:12]} (gym player v2)")
            self.assertEqual(fe.lab_human_bar(None, exam, loop)["line"], "human bar: the judged player has no fight network (no combat prior), nothing to read")
            self.assertIn("(the exam has no human stratum)", fe.lab_human_bar(sha, Path(tmp), loop)["line"])

    def test_a_checkpoint_with_games_is_a_gym_player_too_and_versions_sort_by_number(self):
        import hashlib  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            loop, exam, sha = self.lab_folder(Path(tmp))
            (loop / "players" / "v10").mkdir()
            (loop / "players" / "v10" / "combat_policy.pt").write_bytes(b"net ten")
            (loop / "checkpoints" / "c0007").mkdir(parents=True)
            (loop / "checkpoints" / "c0007" / "combat_policy.pt").write_bytes(b"net seven")
            (loop / "checkpoints" / "c0008").mkdir()
            (loop / "checkpoints" / "c0008" / "combat_policy.pt").write_bytes(b"net eight")
            (exam / "games-c0007.jsonl").write_text("", encoding="utf-8")  # only a checkpoint with games is looked at
            self.assertEqual([n for n, _ in fe.gym_player_names(exam, loop)], ["v10", "v2", "v1", "c0007"])
            self.assertEqual(fe.gym_player_of(hashlib.sha256(b"net seven").hexdigest(), exam, loop), "c0007")
            self.assertIsNone(fe.gym_player_of(hashlib.sha256(b"net eight").hexdigest(), exam, loop))

    def test_play_takes_a_folder_as_the_player(self):
        import argparse  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            ck = Path(tmp) / "checkpoints" / "c0190"
            ck.mkdir(parents=True)
            seen = []
            with mock.patch.object(fe, "run_player", lambda a, label, d, half, games: seen.append((label, d, games.name)) or 0):
                a = argparse.Namespace(exam_dir=str(Path(tmp) / "exam"), player="", player_dir=str(ck), label="", half="all")
                self.assertEqual(fe.cmd_play(a), 0)
                self.assertEqual(seen, [("c0190", ck, "games-c0190.jsonl")])
                self.assertEqual(fe.cmd_play(argparse.Namespace(exam_dir=tmp, player="", player_dir="", label="", half="all")), 2)
                fe.cmd_play(argparse.Namespace(exam_dir=tmp, player=str(ck), player_dir="", label="x", half="A"))  # --player takes a folder as before
                self.assertEqual(seen[-1][0], "x")
            args = []
            for flag in ("--player-dir", "--player"):
                with mock.patch.object(fe, "cmd_play", lambda a: args.append((a.player, a.player_dir)) or 0):
                    fe.main(["play", flag, "p"])
            self.assertEqual(args, [("", "p"), ("p", "")])


class CommandTests(unittest.TestCase):
    """`report` and `compare` end to end on an exam built from a synthetic loop folder and synthetic games files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.loop = t / "loop"
        (self.loop / "starts").mkdir(parents=True)
        gl.write_start_set(self.loop / "starts", [{**s, "doc": {"x": 1}} for s in synthetic_pool()])
        self.out = t / "exam"
        with mock.patch.object(gl, "board_run_seeds", lambda: set()):
            fe.build_exam(self.loop, self.out, per_cell=15)
        self.starts = fe.load_exam(self.out)
        rnd = random.Random(3)
        p = {"act1-hallway": 0.97, "act1-elite": 0.7, "act1-boss-74": 0.5, "act1-boss-77": 0.4, "act1-boss-79": 0.5, "act2-hallway": 0.9,
             "act2-elite": 0.6, "act2-boss-75": 0.2, "act2-boss-76": 0.15}
        champ = games_for(self.starts, lambda s: p[s["cell"]], rnd)
        cand = games_for(self.starts, lambda s: min(0.99, p[s["cell"]] + 0.1), rnd)
        for label, games in (("v2", champ), ("c1", cand)):
            with open(self.out / f"games-{label}.jsonl", "w", encoding="utf-8") as fh:
                for r in games.values():
                    fh.write(json.dumps({**r, "config": "a3-v1", "prior_path": "server", "engine": "abc"}) + "\n")
        with open(self.out / "baseline.jsonl", "w", encoding="utf-8") as fh:
            for sid, s in self.starts.items():
                for j in range(4):
                    fh.write(json.dumps({"start": sid, "j": j, "p_win": p[s["cell"]], "l0": 0.5}) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def run_cmd(self, fn, **kw):
        import argparse  # noqa: PLC0415
        import contextlib  # noqa: PLC0415
        import io  # noqa: PLC0415

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = fn(argparse.Namespace(exam_dir=str(self.out), **kw))
        return rc, buf.getvalue()

    def test_report_prints_the_cells_and_stores_the_calibration_flags(self):
        rc, text = self.run_cmd(fe.cmd_report, player="v2", write_calibration=True)
        self.assertEqual(rc, 0)
        self.assertIn("fight exam, player v2", text)
        self.assertIn("act2-boss-75 (kaiser-crab)", text)
        self.assertIn("cells short of the target 15", text)
        cal = json.loads((self.out / "calibration-v2.json").read_text())
        self.assertEqual(sorted(cal["always"] + cal["never"] + cal["mixed"]), sorted(self.starts))
        self.assertEqual(cal["player"], "v2")
        rc, text = self.run_cmd(fe.cmd_report, player="nobody", write_calibration=False)
        self.assertEqual(rc, 1)

    def test_compare_reads_half_a_against_the_stored_flags_and_warns_when_the_players_differ_in_setup(self):
        self.run_cmd(fe.cmd_report, player="v2", write_calibration=True)
        rc, text = self.run_cmd(fe.cmd_compare, candidate="c1", champion="v2", no_play=True, boot=40)
        self.assertEqual(rc, 0)
        self.assertIn("c1 against v2 on half A", text)
        self.assertIn("win rate difference, adjusted", text)
        self.assertNotIn("WARNING", text)
        rows = [json.loads(x) for x in (self.out / "games-c1.jsonl").read_text().splitlines()]
        with open(self.out / "games-c1.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({**r, "prior_path": "cpu"}) + "\n")
        rc, text = self.run_cmd(fe.cmd_compare, candidate="c1", champion="v2", no_play=True, boot=20)
        self.assertIn("WARNING: the two players' games differ in prior_path", text)
        rc, text = self.run_cmd(fe.cmd_compare, candidate="c9", champion="v2", no_play=True, boot=20)
        self.assertEqual(rc, 1)


class LockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        (t / "LAB" / "nightly").mkdir(parents=True)
        (t / "backlog").mkdir()
        self.t = t
        self.patches = [mock.patch.object(fg, "QUIET", t / "QUIET.lock"), mock.patch.object(fg, "MAIN_LOCK", t / "backlog" / "main.lock"),
                        mock.patch.object(fg, "ACTIVE", t / "LAB" / "nightly" / "active"), mock.patch.object(gl, "PAUSE", t / "LAB" / "PAUSE"),
                        mock.patch.object(gl, "HALT", t / "LAB" / "HALT"), mock.patch.object(gl, "in_night_window", lambda now=None: False)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_the_run_reads_the_labs_locks_as_the_loop_does(self):
        b = lambda starting=True, **kw: fe.blocked_reason(starting=starting, **kw)  # noqa: E731
        self.assertIsNone(b())
        now = gl.stamp()
        fg.QUIET.write_text(f"gym-loop {now} pid=2\ngym-loop-train {now} pid=3\nw3-blocks {now} pid=1\nfight-exam {now} pid=9\n"
                            "old-gate 2026-01-01T00:00:00Z pid=4\n")
        self.assertIsNone(b())  # the loop's own lines, w3-blocks, our own and a line over two hours old do not block
        fg.QUIET.write_text(fg.QUIET.read_text() + f"bench-config {now} pid=5\n")
        self.assertEqual(b(), "quiet:bench-config")
        self.assertEqual(b(starting=False), "quiet:bench-config")
        fg.QUIET.unlink()
        (self.t / "backlog" / "main.lock").write_text("x\n")
        self.assertEqual(b(), "main.lock")
        self.assertIsNone(b(starting=False))  # a running step carries on through a landing
        (self.t / "backlog" / "main.lock").unlink()
        for f, why in ((self.t / "LAB" / "PAUSE", "pause"), (self.t / "LAB" / "nightly" / "active", "nightly-active"), (self.t / "LAB" / "HALT", "halt")):
            f.write_text("x\n")
            self.assertEqual(b(), why)
            f.unlink()
        stop = self.t / "STOP"
        stop.write_text("x\n")
        self.assertEqual(b(stop_file=stop), "stop")
        stop.unlink()
        self.assertEqual(b(until="2000-01-01T00:00:00Z"), "until")
        self.assertIsNone(b(until="2999-01-01T00:00:00Z"))
        with mock.patch.object(gl, "in_night_window", lambda now=None: True):
            self.assertEqual(b(), "night-window")

    def test_the_line_carries_our_pid_and_keeps_the_others(self):
        fg.QUIET.write_text("gym-loop 2026-10-30T10:00:00Z pid=1\n")
        fe.hold_line(True)
        lines = fg.QUIET.read_text().splitlines()
        self.assertEqual(lines[0], "gym-loop 2026-10-30T10:00:00Z pid=1")
        self.assertTrue(lines[1].startswith("fight-exam ") and lines[1].endswith(f"pid={os.getpid()}"))
        fe.hold_line(True)
        self.assertEqual(len(fg.QUIET.read_text().splitlines()), 2)
        fe.hold_line(False)
        self.assertEqual(fg.QUIET.read_text().splitlines(), ["gym-loop 2026-10-30T10:00:00Z pid=1"])
        fg.QUIET.unlink()
        fe.hold_line(True)
        fe.hold_line(False)
        self.assertFalse(fg.QUIET.exists())

    def test_the_lock_is_refused_while_another_holds_the_machine_and_waits_for_the_loops_line_to_go(self):
        now = gl.stamp()
        fg.QUIET.write_text(f"bench-config {now} pid=5\n")
        with self.assertRaises(fe.Blocked):
            with fe.ExamLock():
                self.fail("entered")
        self.assertEqual(fg.QUIET.read_text(), f"bench-config {now} pid=5\n")  # ours was never written
        fg.QUIET.write_text(f"gym-loop {now} pid=2\n")
        sleeps = []

        def sleep(s):  # the loop drops its line within its stop poll
            sleeps.append(s)
            if len(sleeps) == 2:
                fg.QUIET.write_text("".join(x + "\n" for x in fg.QUIET.read_text().splitlines() if not x.startswith("gym-loop")))

        with mock.patch.object(fe.time, "sleep", sleep):
            with fe.ExamLock() as lock:
                lines = fg.QUIET.read_text().splitlines()
                self.assertEqual([x.split()[0] for x in lines], ["fight-exam"])
                self.assertIsNone(lock.stop_reason())
                fg.QUIET.write_text(fg.QUIET.read_text() + f"bench-config {now} pid=5\n")
                self.assertEqual(lock.stop_reason(), "quiet:bench-config")  # another's line stops a running play at once
        self.assertEqual([x.split()[0] for x in fg.QUIET.read_text().splitlines()], ["bench-config"])

    def test_the_lock_gives_up_when_the_loops_line_stays(self):
        fg.QUIET.write_text(f"gym-loop {gl.stamp()} pid=2\n")
        with mock.patch.object(fe.time, "sleep", lambda s: None), self.assertRaises(fe.Blocked):
            with fe.ExamLock(wait_s=0.0):
                self.fail("entered")
        self.assertEqual([x.split()[0] for x in fg.QUIET.read_text().splitlines()], ["gym-loop"])


# ---------------------------------------------------------------------------------------------------------- real games


class ExamGameTests(unittest.TestCase):
    """Short arena fights at a3-v1 with the fixture prior in the loop's lockstep processes: the observer changes no move, and its
    measures agree with the fight. Skipped where the native library is not there."""

    @classmethod
    def setUpClass(cls):
        try:
            from prior_fixture import export_prior  # noqa: PLC0415 -- torch and the native library
        except (FileNotFoundError, ImportError) as error:
            raise unittest.SkipTest(f"native library unavailable: {error}") from error
        from test_gym_lockstep import _make_starts  # noqa: PLC0415

        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        export_prior(root / "agents" / "fixture", prior_min_visits=0)
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        p = ctx.Process(target=_make_starts, args=(queue, json.loads(F18.read_text(encoding="utf-8"))["doc"]), daemon=True)
        p.start()
        kind, got = queue.get(timeout=120)
        p.join(30)
        if kind != "ok":
            raise unittest.SkipTest(f"native library unavailable: {got}")
        cls.exam = root / "exam"
        cls.exam.mkdir()
        starts = [{**s, "seed": f"RUN{i}", "kind": "hallway", "cell": "act1-hallway", "kind_cell": "act1-hallway", "half": fe.half_of(s["id"]),
                   "stratum": "ai", "split": "train"} for i, s in enumerate(got) if not s["id"].startswith("F18")]
        gl.write_start_set(cls.exam, starts)
        cls.starts = fe.load_exam(cls.exam)
        cls.agent = root / "agents" / "fixture"
        cls.init = ("core2", str(cls.agent.parent), cls.agent.name, str(root))
        import gym_lockstep  # noqa: PLC0415

        tasks = fe.exam_tasks(cls.starts, "fixture", "fixture", "a3-v1", cls.exam, set(), draws=2)
        cls.rows = cls.exam / "rows.jsonl"
        stopped, _ = gym_lockstep.run_lockstep(tasks, cls.rows, 1, 2, cls.init, {"prior": "cpu", "game_steps": "fight_exam:exam_game_steps",
                                                                                  "paths": [str(ROOT / "rl")]})
        assert stopped is None
        info = {"starts": str(cls.exam / "starts.jsonl"), "incoming": "", "player": "fixture", "config": "a3-v1", "record": False, "ns": fe.NS}
        plain = [(a, s, j, c, info) for a, s, j, c, _ in tasks]
        cls.plain = cls.exam / "plain.jsonl"
        stopped, _ = gym_lockstep.run_lockstep(plain, cls.plain, 1, 2, cls.init, {"prior": "cpu", "game_steps": "gym_loop:gen_game_steps",
                                                                                   "paths": [str(ROOT / "rl")]})
        assert stopped is None

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_observer_changes_no_move_and_the_game_is_the_gyms(self):
        a, b = fe.index_games(fg.read_jsonl(self.rows)), fe.index_games(fg.read_jsonl(self.plain))
        self.assertEqual(len(a), 6)
        self.assertEqual(sorted(a), sorted(b))
        for key in a:
            for f in ("won", "end_hp", "turns", "decisions", "sims", "moves_sha", "first_move", "L0", "potions_used"):
                self.assertEqual(a[key][f], b[key][f], (key, f))

    def test_the_measures_are_consistent_with_the_fight(self):
        for r in fg.read_jsonl(self.rows):
            self.assertEqual(r["exam"], fe.EXAM_RULE)
            self.assertEqual(r["cell"], "act1-hallway")
            self.assertEqual(r["hp_lost"], r["start_hp"] - (r["end_hp"] if r["won"] else 0))
            self.assertGreaterEqual(r["energy"], 0.0)
            self.assertLessEqual(r["energy"], 3.0 * r["turns"] + 12)
            self.assertGreaterEqual(r["wasted"], 0.0)
            self.assertLessEqual(r["wasted"], r["block_end"])
            if r["won"]:
                self.assertGreater(r["dealt"], 0.0)
            self.assertEqual(r["j"] in range(2), True)
            self.assertEqual(len(r["engine"]), 12)


if __name__ == "__main__":
    unittest.main()
