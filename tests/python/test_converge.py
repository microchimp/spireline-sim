"""The converged-labels grader (rl/converge.py; D55 addendum 15).

Pinned: the sample (which fights, how many turns each, bosses first, a second turn only where a cell has too few fights); the turn pick
(distinct, eligible, fixed seed); the ladder (the 1x rung of a kind is the teacher's own budget, each rung doubles the simulations,
every rung has its own fixed seed); the stopping rule on fixed numbers (the move must equal the half rung's and the gap be at least two
standard errors; a capped rung says nothing); the convergence budget and the CPU spent to reach it; the tables' arithmetic on a small
fixed set of turns, including the 10,000-turn cost; the claim files the processes share the work through; and, on a real fight, that a
turn found again by replaying the moves is the same position and that a search of it is a function of its seed.
"""

from __future__ import annotations

import json
import datetime as dt
import math
import os
import random
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import converge as cv  # noqa: E402
import search_configs  # noqa: E402

# ------------------------------------------------------------------------------------------------------ builders


def lite(i: int, kind_cell: str, cell: str, kind: str | None = None) -> dict:
    act = int(kind_cell[3])
    return {"id": f"human:{kind_cell}-{i:03d}|f{10 * act}", "kind_cell": kind_cell, "cell": cell, "kind": kind or kind_cell.split("-")[1],
            "act": act, "enemy": cell.upper(), "stratum": "human"}


def child(a: int, n: int, v: float | None, se: float | None, t: int = -1, kind: str = "card", merged: tuple = ()) -> dict:
    return {"a": a, "t": t, "kind": kind, "card": 7, "merged": list(merged), "n": n, "v": v, "se": se, "pd": 0.0}


def rung(rel, children: list[dict], cpu: float = 1.0, capped: bool = False, load: float = 0.5) -> dict:
    children = sorted(children, key=lambda c: (-c["n"], c["a"], c["t"]))
    return {"rel": rel, "budget": None, "seed": 1, "max_sims": 100, "sims": 100, "failures": 0, "capped": capped, "cpu_s": cpu, "load": load,
            "pos_kind": "play", "n_moves": len(children), "best": children[0], "children": children}


def two(first: int, second: int, gap: float, se: float = 1.0, n: tuple = (70, 30)) -> list[dict]:
    """Two root moves: `first` the more visited with value 10, `second` `gap` lower, both with standard error `se`."""
    return [child(first, n[0], 10.0, se), child(second, n[1], 10.0 - gap, se)]


def turn(name: str, kind: str, act: int, rungs: list[dict], game_move: int = 1, eligible: int = 10) -> dict:
    game = rung("game", two(game_move, 9, 5.0), cpu=None, load=None)
    return {"turn": name, "start": name, "n": 0, "kind": kind, "act": act, "cell": "c", "k": 3, "decisions": 12, "eligible": eligible,
            "pos_kind": "play", "n_moves": 2, "replay_ok": True, "game": {"cpu_s": 40.0, "won": True}, "rungs": [game, *rungs]}


# A boss turn that converges at 1x: the same move at 0.5x and 1x and a clear gap at 1x (6 against 2 * sqrt(2) = 2.83).
def turn_at_1x(name="t1", act=1, cpu=2.0) -> dict:
    return turn(name, "boss", act, [rung(0.5, two(1, 2, 3.0), cpu), rung(1, two(1, 2, 6.0), cpu), rung(2, two(1, 2, 6.0), cpu),
                                    rung(4, two(1, 2, 6.0), cpu), rung(8, two(1, 2, 6.0), cpu), rung(16, two(1, 2, 6.0), cpu)])


# A boss turn whose move changes at 1x, is unsure at 2x (gap 1 under 2.83) and converges at 4x on the other move.
def turn_at_4x(name="t4", act=2, cpu=3.0) -> dict:
    return turn(name, "boss", act, [rung(0.5, two(1, 2, 3.0), cpu), rung(1, two(2, 1, 6.0), cpu), rung(2, two(2, 1, 1.0), cpu),
                                    rung(4, two(2, 1, 6.0), cpu), rung(8, two(2, 1, 6.0), cpu), rung(16, two(2, 1, 6.0), cpu)], game_move=1)


# A boss turn the rule never holds for: the gap stays inside two standard errors on every rung.
def turn_never(name="tn", act=2, cpu=5.0) -> dict:
    return turn(name, "boss", act, [rung(r, two(1, 2, 1.0), cpu) for r in cv.RELS], game_move=2)


# ------------------------------------------------------------------------------------------------------ the sample


class SampleTests(unittest.TestCase):
    def setUp(self):
        self.lite = ([lite(i, "act1-boss", f"act1-boss-{70 + i % 6}") for i in range(185)]
                     + [lite(i, "act2-boss", f"act2-boss-{80 + i % 3}") for i in range(91)]
                     + [lite(i, "act1-elite", "act1-elite") for i in range(31)] + [lite(i, "act2-elite", "act2-elite") for i in range(31)]
                     + [lite(i, "act1-hallway", "act1-hallway") for i in range(31)] + [lite(i, "act2-hallway", "act2-hallway") for i in range(31)])
        self.fights = cv.select_fights(self.lite)

    def tally(self):
        t: dict[str, list[int]] = {}
        for f in self.fights:
            t.setdefault(f["kind_cell"], [0, 0])
            t[f["kind_cell"]][0] += 1
            t[f["kind_cell"]][1] += f["turns"]
        return t

    def test_the_quotas_in_turns_and_the_fights_that_give_them(self):
        self.assertEqual(self.tally(), {"act1-boss": [100, 100], "act2-boss": [91, 100], "act1-elite": [31, 50], "act2-elite": [31, 50],
                                        "act1-hallway": [31, 50], "act2-hallway": [31, 50]})
        self.assertEqual(sum(f["turns"] for f in self.fights), 400)

    def test_a_fight_is_chosen_once_and_a_second_turn_only_where_the_cell_is_short(self):
        ids = [f["start"] for f in self.fights]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(f["turns"] == 1 for f in self.fights if f["kind_cell"] == "act1-boss"))
        self.assertEqual(sum(f["turns"] == 2 for f in self.fights if f["kind_cell"] == "act2-boss"), 9)
        self.assertEqual(sum(f["turns"] == 2 for f in self.fights if f["kind_cell"] == "act1-elite"), 19)

    def test_a_boss_cell_is_stratified_over_its_bosses(self):
        got = {}
        for f in self.fights:
            if f["kind_cell"] == "act1-boss":
                got[f["cell"]] = got.get(f["cell"], 0) + 1
        self.assertEqual(sorted(got.values()), [16, 16, 17, 17, 17, 17])

    def test_a_cell_with_too_few_of_one_boss_is_filled_from_the_others(self):
        lt = [lite(i, "act1-boss", "act1-boss-1") for i in range(3)] + [lite(i, "act1-boss", "act1-boss-2") for i in range(40)]
        f = cv.select_fights(lt, {"act1-boss": 20})
        self.assertEqual(len(f), 20)
        self.assertEqual(sum(x["cell"] == "act1-boss-1" for x in f), 3)
        self.assertTrue(all(x["turns"] == 1 for x in f))

    def test_bosses_come_first_and_the_order_is_fixed(self):
        kinds = [f["kind"] for f in self.fights]
        self.assertEqual(kinds, sorted(kinds, key=cv.KINDS.index))
        self.assertEqual(self.fights, cv.select_fights(list(reversed(self.lite))))
        self.assertEqual({f["act"] for f in self.fights[:50]}, {1, 2})  # the acts are mixed within a kind

    def test_pick_turns_are_distinct_eligible_and_fixed(self):
        elig = [2, 5, 6, 9, 14, 20]
        a = cv.pick_turns("human:abc|f17", elig, 2)
        self.assertEqual(a, cv.pick_turns("human:abc|f17", elig, 2))
        self.assertEqual(len(set(a)), 2)
        self.assertTrue(set(a) <= set(elig))
        self.assertEqual(cv.pick_turns("x", [4], 3), [4])
        self.assertEqual(cv.pick_turns("x", [], 1), [])

    def test_pick_turns_are_uniform_over_the_eligible_decisions(self):
        elig = list(range(10))
        counts = [0] * 10
        for i in range(5000):
            counts[cv.pick_turns(f"fight-{i}", elig, 1)[0]] += 1
        self.assertTrue(all(400 < c < 600 for c in counts), counts)


# ------------------------------------------------------------------------------------------------------ the ladder


class LadderTests(unittest.TestCase):
    POSITIONS = [dict(kind=k, n_moves=n, hp_frac=h) for k in ("play", "target", "select") for n in (2, 12) for h in (0.3, 0.9)]

    def sims(self, cfg, kind, p):
        return cfg.sims_for(search_configs.Position(hard=kind != "hallway", fight={"hallway": "hall"}.get(kind, kind), **p))

    def test_the_1x_rung_is_the_teachers_own_budget_for_every_kind_of_decision(self):
        teacher = search_configs.get(cv.TEACHER_CONFIG)
        for kind in cv.KINDS:
            for p in self.POSITIONS:
                self.assertEqual(self.sims(cv.config_at(kind, 1), kind, p), self.sims(teacher, kind, p), (kind, p))

    def test_each_rung_doubles_the_simulations(self):
        for kind in cv.KINDS:
            for p in self.POSITIONS:
                sims = [self.sims(cv.config_at(kind, r), kind, p) for r in cv.RELS]
                for lo, hi in zip(sims, sims[1:], strict=False):
                    self.assertEqual(hi, 2 * lo, (kind, p, sims))

    def test_the_ladders_in_the_recipes_units(self):
        self.assertEqual([b for _, b in cv.ladder("boss")], [8, 16, 32, 64, 128, 256])
        self.assertEqual([b for _, b in cv.ladder("elite")], [2, 4, 8, 16, 32, 64])
        self.assertEqual([b for _, b in cv.ladder("hallway")], [0.5, 1, 2, 4, 8, 16])
        self.assertEqual(self.sims(cv.config_at("boss", 16), "boss", dict(kind="play", n_moves=5, hp_frac=0.9)), 30 * 256)
        self.assertEqual(self.sims(cv.config_at("hallway", 0.5), "hallway", dict(kind="select", n_moves=5, hp_frac=0.9)), 60)

    def test_seeds_are_fixed_and_differ_by_rung_decision_and_fight(self):
        a = cv.search_seed("human:r|f33", 0, 12, 4)
        self.assertEqual(a, cv.search_seed("human:r|f33", 0, 12, 4))
        seeds = {cv.search_seed("human:r|f33", 0, 12, r) for r in cv.RELS} | {cv.search_seed("human:r|f33", 0, 13, 4), cv.search_seed("human:q|f33", 0, 12, 4)}
        self.assertEqual(len(seeds), len(cv.RELS) + 2)
        self.assertTrue(all(0 <= s < 1 << 30 for s in seeds))


# ------------------------------------------------------------------------------------------------------ the rule


class RuleTests(unittest.TestCase):
    def test_same_move_by_action_target_kind_and_merged_copies(self):
        a = child(3, 5, 1.0, 1.0, t=0)
        self.assertTrue(cv.same_move(a, child(3, 9, 2.0, 1.0, t=0)))
        self.assertFalse(cv.same_move(a, child(3, 9, 2.0, 1.0, t=1)))
        self.assertFalse(cv.same_move(a, child(4, 9, 2.0, 1.0, t=0)))
        self.assertFalse(cv.same_move(a, child(3, 9, 2.0, 1.0, t=0, kind="potion")))
        self.assertTrue(cv.same_move(child(3, 5, 1.0, 1.0, t=0, merged=(4,)), child(4, 9, 2.0, 1.0, t=0)))
        self.assertTrue(cv.same_move(child(4, 9, 2.0, 1.0, t=0), child(3, 5, 1.0, 1.0, t=0, merged=(4,))))

    def test_the_gap_and_its_error_are_the_top_two_by_visits(self):
        r = rung(1, [child(1, 10, 9.0, 3.0), child(2, 60, 12.0, 4.0), child(3, 30, 7.0, 0.5)])
        g = cv.gap_stats(r, "own")
        self.assertEqual((g["n1"], g["n2"]), (60, 30))
        self.assertAlmostEqual(g["gap"], 5.0)
        self.assertAlmostEqual(g["se"], math.hypot(4.0, 0.5))
        self.assertAlmostEqual(g["z"], 5.0 / math.hypot(4.0, 0.5))

    def test_the_pooled_error_pools_the_two_spreads_over_their_degrees_of_freedom(self):
        # variances se^2 * n: 4^2 * 60 = 960 and 0.5^2 * 30 = 7.5; pooled over 59 and 29 degrees of freedom
        r = rung(1, [child(1, 10, 9.0, 3.0), child(2, 60, 12.0, 4.0), child(3, 30, 7.0, 0.5)])
        g = cv.gap_stats(r)
        var = (59 * 960.0 + 29 * 7.5) / 88
        self.assertAlmostEqual(g["se"], math.sqrt(var * (1 / 60 + 1 / 30)))
        self.assertEqual(cv.gap_stats(r, "pooled"), g)
        with self.assertRaises(ValueError):
            cv.gap_stats(r, "bogus")

    def test_a_runner_up_with_one_visit_has_no_own_error_but_a_pooled_one(self):
        r = rung(1, [child(1, 101, 10.0, 2.0), child(2, 1, -50.0, None)])
        self.assertIsNone(cv.gap_stats(r, "own")["se"])
        g = cv.gap_stats(r)  # the best's spread: 2^2 * 101 = 404, over (1/101 + 1/1)
        self.assertAlmostEqual(g["se"], math.sqrt(404.0 * (1 / 101 + 1)))
        self.assertAlmostEqual(g["gap"], 60.0)
        self.assertIsNone(cv.gap_stats(rung(1, [child(1, 1, 10.0, None), child(2, 1, -50.0, None)]))["se"])

    def test_a_gap_that_cannot_be_estimated_is_none(self):
        self.assertIsNone(cv.gap_stats(rung(1, [child(1, 5, 1.0, 1.0)])))
        self.assertIsNone(cv.gap_stats(rung(1, [child(1, 5, 1.0, 1.0), child(2, 0, None, None)]))["gap"])
        self.assertIsNone(cv.gap_stats(rung(1, [child(1, 5, 1.0, 1.0), child(2, 1, 0.0, None)]), "own")["se"])

    def test_the_rule_needs_the_same_move_and_a_gap_of_two_standard_errors(self):
        gap = 2 * math.sqrt(2)  # two standard errors of two moves with an error of 1 each
        for g, want in ((gap + 0.01, True), (gap - 0.01, False), (-gap - 1, False)):
            t = turn("x", "boss", 1, [rung(0.5, two(1, 2, 5.0)), rung(1, two(1, 2, g))])
            self.assertEqual(cv.rule_holds(t, 1, se_mode="own"), want, g)
        moved = turn("x", "boss", 1, [rung(0.5, two(1, 2, 5.0)), rung(1, two(2, 1, 20.0))])
        self.assertFalse(cv.rule_holds(moved, 1))
        self.assertFalse(cv.rule_holds(moved, 1, se_mode="own"))

    def test_a_gap_of_exactly_two_standard_errors_holds(self):
        # errors 3 and 4 add to 5; values 20 and 10 differ by exactly 10
        pair = [child(1, 70, 20.0, 3.0), child(2, 30, 10.0, 4.0)]
        t = turn("x", "boss", 1, [rung(0.5, two(1, 2, 5.0)), rung(1, pair)])
        self.assertTrue(cv.rule_holds(t, 1, se_mode="own"))
        self.assertFalse(cv.rule_holds(t, 1, gap_se=2.01, se_mode="own"))

    def test_overturn_against_the_1x_rung_counts_a_converged_move_the_1x_rung_did_not_choose(self):
        d = turn("d", "boss", 1, [rung(0.5, two(1, 2, 3.0)), rung(1, two(1, 2, 1.0)), rung(2, two(2, 1, 1.0)), rung(4, two(2, 1, 6.0)),
                                  rung(8, two(2, 1, 6.0)), rung(16, two(2, 1, 6.0))], game_move=1)
        g = cv.group_summary([d])
        self.assertEqual((g["overturn_1x"]["k"], g["overturn_game"]["k"], g["overturn_game_converged"]["k"]), (1, 1, 1))
        self.assertEqual((g["agree_1_4"]["k"], g["agree_4_16"]["k"]), (0, 1))

    def test_a_capped_or_missing_rung_says_nothing(self):
        t = turn("x", "boss", 1, [rung(0.5, two(1, 2, 5.0)), rung(1, two(1, 2, 20.0), capped=True)])
        self.assertIsNone(cv.rule_holds(t, 1))
        t = turn("x", "boss", 1, [rung(0.5, two(1, 2, 5.0), capped=True), rung(1, two(1, 2, 20.0))])
        self.assertIsNone(cv.rule_holds(t, 1))
        self.assertIsNone(cv.rule_holds(t, 4))

    def test_a_move_with_one_visit_does_not_converge_on_its_own_error_only(self):
        t = turn("x", "boss", 1, [rung(0.5, two(1, 2, 5.0)), rung(1, [child(1, 70, 10.0, 1.0), child(2, 1, -50.0, None)])])
        self.assertFalse(cv.rule_holds(t, 1, se_mode="own"))
        self.assertTrue(cv.rule_holds(t, 1))  # pooled: a gap of 60 against an error of 8.4

    def test_convergence_budget_move_and_cpu(self):
        c = cv.convergence(turn_at_1x())
        self.assertEqual((c["rel"], c["last"], c["cpu_to"], c["move"]["a"]), (1, 1, 4.0, 1))
        self.assertEqual(c["holds"][1], True)
        c = cv.convergence(turn_at_4x())
        self.assertEqual((c["rel"], c["last"], c["cpu_to"], c["move"]["a"]), (4, 4, 12.0, 2))
        self.assertEqual((c["holds"][1], c["holds"][2], c["holds"][4]), (False, False, True))
        c = cv.convergence(turn_never())
        self.assertEqual((c["rel"], c["last"], c["cpu_to"], c["move"]["a"]), (None, 16, 30.0, 1))

    def test_wilson_and_quantile(self):
        lo, hi = cv.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertIsNone(cv.wilson(0, 0))
        self.assertEqual(cv.wilson(0, 10)[0], 0.0)
        xs = [1.0, 2.0, 3.0, 4.0, 10.0]
        self.assertEqual((cv.quantile(xs, 0.5), cv.quantile(xs, 0.0)), (3.0, 1.0))
        self.assertAlmostEqual(cv.quantile(xs, 0.9), 7.6)
        self.assertIsNone(cv.quantile([], 0.5))


# ------------------------------------------------------------------------------------------------------ the tables


class SummaryTests(unittest.TestCase):
    def setUp(self):
        # act 1 boss: one turn converged at 1x; act 2 boss: one converged at 4x, one never (teacher's game move 2 on the last)
        self.turns = [turn_at_1x("a", act=1), turn_at_4x("b", act=2), turn_never("c", act=2)]
        self.s = cv.summarize(self.turns, {"act1-boss": 3000, "act2-boss": 1000})

    def test_counts_and_shares(self):
        g1, g2 = self.s["groups"]["act1-boss"], self.s["groups"]["act2-boss"]
        self.assertEqual((g1["turns"], g2["turns"], self.s["kinds"]["boss"]["turns"]), (1, 2, 3))
        self.assertEqual((g1["rule_at"][1]["k"], g1["rule_at"][1]["n"]), (1, 1))
        self.assertEqual((g2["rule_at"][1]["k"], g2["rule_at"][4]["k"], g2["rule_at"][16]["k"]), (0, 1, 1))  # the never-turn holds nowhere
        self.assertEqual([g2["converged_by"][r]["k"] for r in cv.REPORT_RELS], [0, 1, 1])
        self.assertEqual(g2["never"]["k"], 1)
        self.assertEqual((g1["agree_1_4"]["k"], g2["agree_1_4"]["k"], g2["agree_4_16"]["k"]), (1, 2, 2))
        self.assertEqual(g2["agree_1_16"]["k"], 2)

    def test_cpu_to_convergence(self):
        g2 = self.s["groups"]["act2-boss"]
        self.assertEqual(self.s["groups"]["act1-boss"]["cpu_to_median"], 4.0)
        self.assertEqual((g2["cpu_to_median"], g2["cpu_to_mean"]), (21.0, 21.0))  # 12 and 30
        self.assertEqual(g2["cpu_to_conv_median"], 12.0)
        self.assertEqual(g2["own_cpu_conv_median"], 3.0)

    def test_overturn_is_the_converged_move_against_the_teachers_game_move(self):
        g1, g2 = self.s["groups"]["act1-boss"], self.s["groups"]["act2-boss"]
        self.assertEqual(g1["overturn_game"]["k"], 0)  # move 1 played, converged on 1
        self.assertEqual((g2["overturn_game"]["k"], g2["overturn_game"]["n"]), (2, 2))  # b: played 1, converged on 2; c: played 2, top rung says 1
        self.assertEqual((g2["overturn_game_converged"]["k"], g2["overturn_game_converged"]["n"]), (1, 1))  # only b converged
        self.assertEqual((g2["overturn_1x"]["k"], g2["overturn_1x"]["n"]), (0, 2))  # b's 1x move is the converged one, and c's 1x move is the top rung's
        self.assertEqual(g2["loss_n"], 2)  # b loses 6 HP (move 1 at 4x: 4 against 10), c loses 1
        self.assertEqual((g2["loss_median"], g2["loss_mean"]), (3.5, 3.5))
        self.assertEqual((g2["overturn_material"]["k"], g2["overturn_material"]["n"]), (1, 2))
        self.assertEqual(g2["loss_unmeasured"], 0)

    def test_the_load_and_cap_figures(self):
        g1 = self.s["groups"]["act1-boss"]
        self.assertEqual((g1["load_median"], g1["load_min"], g1["load_max"], g1["capped_rungs"]), (0.5, 0.5, 0.5, 0))
        capped = turn_at_1x("d")
        capped["rungs"][3]["capped"] = True
        self.assertEqual(cv.group_summary([capped])["capped_rungs"], 1)

    def test_corpus_cost_is_the_weighted_mean_per_turn_times_ten_thousand(self):
        # act 1 converges for 4 CPU-s a turn, act 2 for 21; at 3:1 the mean is 8.25 s a turn
        c = self.s["cost"]["bosses alone"]["to convergence"]
        self.assertAlmostEqual(c["per_turn_s"], 0.75 * 4.0 + 0.25 * 21.0)
        self.assertAlmostEqual(c["core_hours"], c["per_turn_s"] * 10000 / 3600)
        fixed = self.s["cost"]["bosses alone"]["fixed 1x"]
        self.assertAlmostEqual(fixed["per_turn_s"], 0.75 * 2.0 + 0.25 * 4.0)

    def test_a_mix_leaves_out_a_group_with_no_turns(self):
        self.assertAlmostEqual(cv.corpus_cost({"a": 10.0, "b": None}, {"a": 1.0, "b": 5.0, "c": 5.0})["per_turn_s"], 10.0)
        self.assertIsNone(cv.corpus_cost({"a": None}, {"a": 1.0})["core_hours"])

    def test_a_mix_over_every_kind_is_priced_only_when_every_kind_has_turns(self):
        self.assertEqual(list(cv.summarize(self.turns)["cost"]), ["bosses alone"])  # the default pools name six groups, two have turns
        kinds = [turn("e1", "elite", 1, turn_at_1x()["rungs"][1:]), turn("e2", "elite", 2, turn_at_1x()["rungs"][1:]),
                 turn("h1", "hallway", 1, turn_at_1x()["rungs"][1:]), turn("h2", "hallway", 2, turn_at_1x()["rungs"][1:])]
        full = cv.summarize(self.turns + kinds)["cost"]
        self.assertEqual(list(full), ["own mix (fights)", "own mix (turns)", "bosses alone", "elites alone", "hallways alone"])
        self.assertEqual(set(full["elites alone"]["weights"]), {"act1-elite", "act2-elite"})
        w = full["own mix (fights)"]["weights"]
        self.assertEqual((w["act1-elite"], w["act2-hallway"]), (cv.HUMAN_META_POOLS["act1-elite"], cv.HUMAN_META_POOLS["act2-hallway"]))
        self.assertEqual(set(full["bosses alone"]["weights"]), {"act1-boss", "act2-boss"})

    def test_the_tables_print(self):
        text = cv.tables(self.s)
        self.assertIn("act1-boss", text)
        self.assertIn("all boss", text)
        self.assertIn("10,000-turn corpus", text)
        json.dumps(self.s)

    def test_busy_fraction(self):
        self.assertEqual(cv.busy_fraction((100, 1000, 500), (150, 2000, 1000)), round(1 - 50 / 1500, 3))
        self.assertIsNone(cv.busy_fraction(None, (1, 2, 3)))
        self.assertIsNone(cv.busy_fraction((1, 2, 3), (1, 2, 3)))


# ------------------------------------------------------------------------------------------------------ the files


class FileTests(unittest.TestCase):
    def test_a_fight_is_claimed_once_and_a_dead_claim_is_taken_over(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self.assertTrue(cv.claim(out, "human:a|f17"))
            self.assertFalse(cv.claim(out, "human:a|f17"))
            self.assertTrue(cv.claim(out, "human:b|f17"))
            path = out / "claims" / (cv.fight_file(out, "human:a|f17").stem + ".claim")
            path.write_text("2147483000")  # no such process
            self.assertTrue(cv.claim(out, "human:a|f17"))
            self.assertEqual(path.read_text().strip(), str(os.getpid()))

    def test_turns_are_written_whole_and_read_back(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            cv.write_atomic(cv.fight_file(out, "human:a|f17"), [{"x": 1}, {"x": 2}])
            cv.write_atomic(cv.fight_file(out, "human:b|f17"), [{"x": 3}])
            self.assertEqual(sorted(t["x"] for t in cv.read_turns(out)), [1, 2, 3])
            self.assertEqual(list((out / "turns").glob("*.tmp*")), [])


# ------------------------------------------------------------------------------------------------------ the engine calibration


def cal_row(act: int, game_alike: bool, old: float, new: float, rungs: list[tuple]) -> dict:
    """rungs: (rel, old cpu, new cpu, alike)."""
    return {"turn": f"t{act}", "act": act, "kind": "boss", "k": 3, "game_alike": game_alike, "game_old_cpu": old, "game_new_cpu": new,
            "rungs": [{"rel": r, "old_cpu": o, "new_cpu": n, "old_load": 0.95, "new_load": 0.5, "alike": a} for r, o, n, a in rungs]}


class CalibrationTests(unittest.TestCase):
    def test_search_alike_needs_the_same_simulations_moves_visits_and_values(self):
        a = rung(1, two(1, 2, 5.0))
        self.assertTrue(cv.search_alike(a, rung(1, two(1, 2, 5.0), cpu=9.0)))
        self.assertFalse(cv.search_alike(a, rung(1, two(1, 2, 5.1))))
        self.assertFalse(cv.search_alike(a, rung(1, two(1, 2, 5.0, n=(71, 29)))))
        self.assertFalse(cv.search_alike(a, {**rung(1, two(1, 2, 5.0)), "sims": 99}))

    def test_the_summary_is_the_ratio_of_the_summed_cpu_and_the_count_alike(self):
        rows = [cal_row(1, True, 100.0, 40.0, [(1, 4.0, 1.5, True), (2, 8.0, 3.0, True), (4, 16.0, 6.0, True)]),
                cal_row(2, False, 200.0, 70.0, [(1, 6.0, 2.5, True), (2, 12.0, 5.0, False), (4, 24.0, 10.0, True)])]
        s = cv.calibration_summary(rows)
        self.assertEqual((s["turns"], s["games_alike"], s["rungs"], s["rungs_alike"]), (2, 1, 6, 5))
        self.assertAlmostEqual(s["game_cpu_ratio"], 110.0 / 300.0)
        self.assertAlmostEqual(s["rung_cpu_ratio"], 28.0 / 70.0)
        self.assertAlmostEqual(s["rung_cpu_ratio_by_rel"]["1"], 4.0 / 10.0)
        self.assertEqual((s["load_old_median"], s["load_new_median"]), (0.95, 0.5))
        self.assertIsNone(cv.calibration_summary([])["rung_cpu_ratio"])

    def test_the_scale_comes_from_the_calibration_file_and_prices_only_that_kind(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self.assertIsNone(cv.cpu_scale_of(out))
            cv.write_atomic(out / "calibration.json", {"summary": cv.calibration_summary([cal_row(1, True, 10.0, 4.0, [(1, 4.0, 1.0, True)])]), "rows": []})
            self.assertEqual(cv.cpu_scale_of(out), {"boss": 0.25})
        boss = [turn_at_1x("a", act=1), turn_at_1x("b", act=2)]
        elite = [turn("e1", "elite", 1, turn_at_1x()["rungs"][1:]), turn("e2", "elite", 2, turn_at_1x()["rungs"][1:])]
        pools = {"act1-boss": 1, "act2-boss": 1, "act1-elite": 1, "act2-elite": 1}
        raw = cv.summarize(boss + elite, pools)
        scaled = cv.summarize(boss + elite, pools, cpu_scale={"boss": 0.5})
        self.assertAlmostEqual(scaled["cost"]["bosses alone"]["fixed 1x"]["per_turn_s"], 0.5 * raw["cost"]["bosses alone"]["fixed 1x"]["per_turn_s"])
        self.assertAlmostEqual(scaled["cost"]["elites alone"]["to convergence"]["per_turn_s"], raw["cost"]["elites alone"]["to convergence"]["per_turn_s"])
        self.assertAlmostEqual(scaled["cost_as_measured"]["bosses alone"]["fixed 1x"]["per_turn_s"], raw["cost"]["bosses alone"]["fixed 1x"]["per_turn_s"])
        mix = scaled["cost"]["own mix (fights)"]["to convergence"]["per_turn_s"]  # boss 4.0 * 0.5 and elite 4.0, equal weights
        self.assertAlmostEqual(mix, (2 * 2.0 + 2 * 4.0) / 4)
        text = cv.tables(scaled)
        self.assertIn("CPU-s of boss x 0.500", text)
        self.assertIn("hallway x1 / x4 / x16, elite x4 / x16 / x64, boss x16 / x64 / x256", text)


# ------------------------------------------------------------------------------------------------------ the quiet lock


class QuietTests(unittest.TestCase):
    """The measurement pauses at fight and rung boundaries while another job holds the machine (QUIET.lock, a night block)."""

    NOW = dt.datetime(2026, 11, 9, 3, 0, 0, tzinfo=dt.UTC)

    def referee(self, lines: list[str], night: bool = False) -> tempfile.TemporaryDirectory:
        d = tempfile.TemporaryDirectory()
        root = Path(d.name)
        if lines:
            (root / "QUIET.lock").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
        if night:
            (root / "LAB" / "nightly").mkdir(parents=True)
            (root / "LAB" / "nightly" / "active").write_text("x")
        return d

    def test_a_live_foreign_line_holds_the_job(self):
        with self.referee(["lab-train 2026-11-09T02:30:00Z pid=1"]) as d:
            self.assertEqual(cv.quiet_reason(self.NOW, Path(d)), "quiet:lab-train")

    def test_the_lines_the_lab_runs_beside_do_not_hold_it(self):
        with self.referee(["gym-loop 2026-11-09T02:59:00Z pid=1", "lab-score 2026-11-09T02:59:00Z pid=2", "w3-blocks 2026-11-09T02:59:00Z pid=3"]) as d:
            self.assertIsNone(cv.quiet_reason(self.NOW, Path(d)))

    def test_a_line_over_two_hours_old_is_stale_and_a_line_with_no_time_is_live(self):
        with self.referee(["lab-train 2026-11-09T00:59:00Z pid=1"]) as d:
            self.assertIsNone(cv.quiet_reason(self.NOW, Path(d)))
        with self.referee(["lab-train 2026-11-09T01:01:00Z pid=1"]) as d:
            self.assertEqual(cv.quiet_reason(self.NOW, Path(d)), "quiet:lab-train")
        with self.referee(["odd-job"]) as d:
            self.assertEqual(cv.quiet_reason(self.NOW, Path(d)), "quiet:odd-job")

    def test_a_night_block_holds_it_and_no_file_does_not(self):
        with self.referee([], night=True) as d:
            self.assertEqual(cv.quiet_reason(self.NOW, Path(d)), "nightly-active")
        with self.referee([]) as d:
            self.assertIsNone(cv.quiet_reason(self.NOW, Path(d)))

    def test_pause_point_waits_for_the_line_to_go_and_logs_the_wait(self):
        with self.referee([]) as d, tempfile.TemporaryDirectory() as out:
            lock = Path(d) / "QUIET.lock"
            stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            lock.write_text(f"lab-train {stamp} pid=1" + chr(10), encoding="utf-8")
            said: list[str] = []
            timer = threading.Timer(0.5, lambda: lock.unlink())
            with mock.patch.object(cv, "REFEREE", Path(d)):
                timer.start()
                waited = cv.pause_point("act2-boss rung 4x", said.append, poll_s=0.1, out=Path(out))
                self.assertGreaterEqual(waited, 0.4)
                self.assertEqual(cv.pause_point("again", said.append, poll_s=0.1, out=Path(out)), 0.0)
            self.assertTrue(said[0].startswith("pause (quiet:lab-train)") and said[1].startswith("resume after"))
            log = (Path(out) / "pauses.log").read_text(encoding="utf-8").split()
            self.assertEqual(log[3:], ["quiet:lab-train", "act2-boss", "rung", "4x"])

    def test_run_fight_pauses_before_the_game_and_before_every_rung(self):
        labels: list[str] = []
        fight = {"start": "human:a|f17", "kind": "hallway", "kind_cell": "act1-hallway", "act": 1, "cell": "act1-hallway", "turns": 1}
        dec = [{"k": 0, "n_moves": 3, "pos_kind": "play", "move": [1, -1], "obs": "h0",
                "search": rung("game", two(1, 2, 5.0), cpu=None, load=None)}]
        world = mock.Mock()
        with (mock.patch("gym_loop.load_start", return_value={"id": "human:a|f17", "node_type": 1, "floor": 3}),
              mock.patch.object(cv, "play_teacher", return_value=({"cpu_s": 1.0}, dec)),
              mock.patch.object(cv, "replay_to", return_value=world), mock.patch.object(cv, "obs_hash", return_value="h0"),
              mock.patch.object(cv, "search_rung", side_effect=lambda w, s, kind, rel, k, extra: rung(rel, two(1, 2, 5.0)))):
            turns = cv.run_fight(fight, {}, "unused", log=lambda m: None, pause=labels.append)
        self.assertEqual(labels, ["act1-hallway game"] + [f"act1-hallway rung {r:g}x" for r in cv.RELS])
        self.assertEqual([r["rel"] for r in turns[0]["rungs"]], ["game", *cv.RELS])

    def test_the_tables_carry_the_own_error_table_when_given_one(self):
        turns = [turn_at_1x("a", act=1), turn_at_4x("b", act=2)]
        s, strict = cv.summarize(turns, {"act1-boss": 3, "act2-boss": 1}), cv.summarize(turns, {"act1-boss": 3, "act2-boss": 1}, se_mode="own")
        self.assertIn("own error only", cv.tables(s, strict))
        self.assertNotIn("own error only", cv.tables(s))


# ------------------------------------------------------------------------------------------------------ a real fight


class RealFightTests(unittest.TestCase):
    """A real opening on the emulator, played forward at random: replaying the moves reaches the same position, and a rung's search of it
    is a function of its seed (the record is the same twice, apart from the clock)."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("STS2_ENGINE", "core2")
        import combat_mcts  # noqa: PLC0415

        from sts2_gym import native  # noqa: PLC0415
        from sts2_gym.env import ENCOUNTER_IDS  # noqa: PLC0415

        cls.combat_mcts, cls.native, cls.enc = combat_mcts, native, ENCOUNTER_IDS["toadpoles"]
        cls.start = {"id": "human:test|f3", "node_type": 1, "floor": 3}

    def opening(self):
        w = self.combat_mcts.Combat.opening(11, self.enc, 10, 3)
        w.encounter_id = self.enc
        return w

    def play_random(self, world, n):
        import live_search  # noqa: PLC0415

        rnd, moves = random.Random(5), []
        for _ in range(n):
            mask = self.native.valid_actions(world.handle, live_search.MAX_ACTIONS)
            legal = [a for a in range(live_search.MAX_ACTIONS) if mask[a]]
            if world.done or not legal:
                break
            action = rnd.choice(legal)
            live_search._step(world, action, -1)  # noqa: SLF001
            moves.append([action, -1])
        return moves

    def test_replaying_the_moves_reaches_the_same_observation(self):
        a = self.opening()
        moves = self.play_random(a, 4)
        hashes = cv.obs_hash(a)
        b = self.opening()
        cv.step_moves(b, moves)
        self.assertEqual(cv.obs_hash(b), hashes)
        c = self.opening()
        cv.step_moves(c, moves[:-1])
        self.assertNotEqual(cv.obs_hash(c), hashes)
        for w in (a, b, c):
            w.close()

    def test_a_rung_is_a_function_of_its_seed_and_its_record_is_json(self):
        cfg = search_configs.flat(24, min_child_sims=1)
        w = self.opening()
        r1 = cv.search_rung(w, self.start, "hallway", 1, 0, {}, config=cfg)
        r2 = cv.search_rung(w, self.start, "hallway", 1, 0, {}, config=cfg)
        r3 = cv.search_rung(w, self.start, "hallway", 2, 0, {}, config=cfg)
        w.close()
        for r in (r1, r2, r3):
            self.assertEqual(r["max_sims"], 24)
            self.assertFalse(r["capped"])
            self.assertEqual(r["sims"], 24)
            self.assertEqual(sum(c["n"] for c in r["children"]), 24)
            self.assertGreaterEqual(r["n_moves"], 2)
            self.assertGreater(r["cpu_s"], 0.0)
            json.dumps(r)
        strip = lambda r: {k: v for k, v in r.items() if k not in ("cpu_s", "load")}  # noqa: E731
        self.assertEqual(strip(r1), strip(r2))
        self.assertNotEqual(r1["seed"], r3["seed"])

    def test_a_search_past_the_cpu_cap_is_marked_capped(self):
        w = self.opening()
        r = cv.search_rung(w, self.start, "hallway", 1, 0, {}, cap_s=0.0, config=search_configs.flat(400, min_child_sims=1))
        w.close()
        self.assertTrue(r["capped"])
        self.assertLess(r["sims"], 400)
        json.dumps(r)


if __name__ == "__main__":
    unittest.main()
