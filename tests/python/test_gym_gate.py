"""The fight gym's gate rule v3 and its plateau alarm (rl/gym_gate.py, rl/gym_loop.py; D57 item 2 and D57 addendum 2, 2026-11-11).

The pinned facts: pair differences, the start-clustered standard error and the pooled paired difference are the hand calculation; stage 1
is a screen that stops only a candidate clearly worse (a level one goes on); stage 2 promotes alone by v2's own test or by the pooled trend
of the last five stage-2 candidates of the incumbent (at least three, the candidate not worse); level candidates never promote and a +0.6
point trend does within five; a promotion restarts the pool; a promotion inside six hours of the last is a reject that says so; the plateau
alarm trips at 25 gated candidates or 5 days, lists the last ten verdicts, and goes at the next promotion. The games are synthetic: the
board's 200 starts and seeds 0..9, the incumbent's wins drawn once, a candidate's wins the incumbent's with a few pairs turned over.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fight_gym as fg  # noqa: E402
import gym_gate as gg  # noqa: E402
import gym_loop as gl  # noqa: E402

STARTS = [f"S{i:03d}" for i in range(200)]
KEYS = [(s, j) for s in STARTS for j in range(10)]
RULE = gl.PROMOTION_RULE


def incumbent_wins(seed: int = 1) -> dict:
    rnd = random.Random(seed)
    return {k: 1 if rnd.random() < 0.45 else 0 for k in KEYS}


def candidate_wins(inc: dict, up: int, down: int, seed: int) -> dict:
    """The incumbent's games with `up` of its losses turned into wins and `down` of its wins into losses (a net gain of up - down pairs)."""
    rnd = random.Random(seed)
    losses = [k for k in KEYS if not inc[k]]
    wins = [k for k in KEYS if inc[k]]
    flip_up, flip_down = set(rnd.sample(losses, up)), set(rnd.sample(wins, down))
    return {k: 1 if k in flip_up else 0 if k in flip_down else inc[k] for k in KEYS}


def rows(wins: dict, who: str, js=None) -> list[dict]:
    return [{"start": s, "j": j, "won": bool(w), "capped": False, "L0": 0.5 if w else 0.0, "arm": who, "player": who}
            for (s, j), w in wins.items() if js is None or j in js]


def write_rows(p: Path, rs: list[dict]) -> None:
    p.write_text("".join(json.dumps(r) + "\n" for r in rs), encoding="utf-8")


class StatisticsTests(unittest.TestCase):
    def test_pair_differences_count_a_capped_game_as_lost(self):
        cand = [{"start": "A", "j": 0, "won": True, "capped": True}, {"start": "A", "j": 1, "won": True}, {"start": "B", "j": 0, "won": False},
                {"start": "C", "j": 0, "won": True}]
        inc = [{"start": "A", "j": 0, "won": True}, {"start": "A", "j": 1, "won": False}, {"start": "B", "j": 0, "won": True}]
        self.assertEqual(gg.pair_diffs(cand, inc), {("A", 0): -1, ("A", 1): 1, ("B", 0): -1})  # C is nobody's pair
        self.assertEqual(gg.pair_diffs(cand, inc, lambda j: j == 1), {("A", 1): 1})

    def test_the_clustered_standard_error_is_the_hand_calculation(self):
        d = {("A", 0): 1, ("A", 1): 1, ("B", 0): -1, ("B", 1): 0, ("C", 0): 0, ("C", 1): 0}
        m, se, n, c = gg.cluster_mean_se(d)
        self.assertEqual((n, c), (6, 3))
        self.assertAlmostEqual(m, 1 / 6)
        means = [1.0, -0.5, 0.0]  # equal clusters: the standard deviation of the start means over sqrt(starts)
        sd = math.sqrt(sum((x - 1 / 6) ** 2 for x in means) / 2)
        self.assertAlmostEqual(se, sd / math.sqrt(3))
        self.assertEqual(gg.cluster_mean_se({}), (None, None, 0, 0))
        self.assertIsNone(gg.cluster_mean_se({("A", 0): 1, ("A", 1): 0})[1])  # one start: no spread to speak of

    def test_unequal_clusters_use_the_ratio_form(self):
        d = {("A", 0): 1, ("B", 0): 0, ("B", 1): 0, ("B", 2): 0}
        m, se, _n, _c = gg.cluster_mean_se(d)
        self.assertAlmostEqual(m, 0.25)
        want = math.sqrt(2 / 1 * ((1 - 0.25) ** 2 + (0 - 0.75) ** 2)) / 4
        self.assertAlmostEqual(se, want)

    def test_the_normal_tail_and_the_pool(self):
        self.assertAlmostEqual(gg.upper_tail(0.0), 0.5)
        self.assertAlmostEqual(gg.upper_tail(1.2815515655446004), 0.10, places=6)
        self.assertAlmostEqual(gg.upper_tail(gg.Z95), 0.05, places=6)
        a = {(s, j): v for (s, j), v in zip(KEYS, ([1, 0, 0, -1, 0] * 400))}
        one = gg.pooled([a])
        two = gg.pooled([a, dict(a)])
        self.assertEqual((one["k"], two["k"], one["pairs"]), (1, 2, 2000))
        self.assertAlmostEqual(one["diff"], two["diff"])
        self.assertAlmostEqual(one["se"], two["se"])  # the same candidate twice is no new evidence
        neg = {k: -v for k, v in a.items()}
        flat = gg.pooled([a, neg])
        self.assertAlmostEqual(flat["diff"], 0.0)
        self.assertEqual(gg.pooled([])["p"], 1.0)
        # members on different pairs: only the pairs all of them have
        part = {k: v for k, v in list(a.items())[:1000]}
        self.assertEqual(gg.pooled([a, part])["pairs"], 1000)


class StageOneTests(unittest.TestCase):
    """The screen: 400 pairs of stage 1's seed pair (j 4 and 5, pair 2), one-sided 95% bounds."""

    def setUp(self):
        self.inc = incumbent_wins()
        self.js = (4, 5)
        self.pair = 2

    def stage1(self, wins: dict, rule: dict | None = None) -> dict:
        return gl.decide(rows(wins, "c", self.js), rows(self.inc, "v", self.js), 1, rule or RULE, pair=self.pair)

    def test_a_level_candidate_goes_on(self):
        v = self.stage1(dict(self.inc))
        self.assertEqual((v["verdict"], v["pairs"], v["win_diff"]), ("continue", 400, 0.0))

    def test_a_candidate_a_little_worse_goes_on_and_a_clearly_worse_one_stops(self):
        slightly = candidate_wins(self.inc, 8, 20, 3)  # about -3 points on 400 pairs: the 95% upper bound is still above -2
        # (the stage-1 pairs only: the flips are drawn over all 2,000, so count what landed there)
        v = self.stage1(slightly)
        self.assertEqual(v["verdict"], "continue", (100 * v["win_diff"], 100 * v["diff_bound"]))
        clearly = {k: (0 if (k[1] in self.js and self.inc[k] and i % 3 == 0) else self.inc[k]) for i, k in enumerate(KEYS)}
        v = self.stage1(clearly)
        self.assertEqual(v["verdict"], "reject")
        self.assertLess(100 * v["diff_bound"], -2.0)
        self.assertIn("clearly worse", v["why"])

    def test_the_screen_is_about_the_upper_bound_and_the_literal_lower_reading_would_stop_level_candidates(self):
        # a level candidate with the noise a real one has: 24 pairs it wins and the incumbent loses, 24 the other way, among the 400
        mine = [k for k in KEYS if k[1] in self.js]
        up = [k for k in mine if not self.inc[k]][:24]
        down = [k for k in mine if self.inc[k]][:24]
        level = {**self.inc, **{k: 1 for k in up}, **{k: 0 for k in down}}
        lower = {**RULE, "screen": {**RULE["screen"], "bound": "lower"}}
        v = self.stage1(level)
        self.assertEqual((v["verdict"], v["win_diff"], v["cand_only"], v["inc_only"]), ("continue", 0.0, 24, 24))
        self.assertGreater(100 * v["diff_bound"], 2.0)  # the upper bound is well above the line
        self.assertEqual(self.stage1(level, lower)["verdict"], "reject")  # the lower bound, 0 - 1.645 SE, is under -2 when SE is 1.7 points

    def test_too_few_pairs_is_incomplete(self):
        half = rows(dict(self.inc), "c", self.js)[:200]
        v = gl.decide(half, rows(self.inc, "v", self.js), 1, RULE, pair=self.pair)
        self.assertEqual(v["verdict"], "incomplete")


class Sequence:
    """A run of stage-2 verdicts through `decide` and the ledger, as the daemon makes them: the incumbent v1's games once, each candidate's
    games in its own file, the pool read back from `gate/verdicts.jsonl`."""

    def __init__(self, tmp: str, rule: dict | None = None, inc: str = "v1") -> None:
        self.loop = gl.Loop(Path(tmp) / "loop")
        self.rule, self.inc_name = rule or RULE, inc
        self.inc = incumbent_wins()
        write_rows(self.loop.gates / f"inc-{inc}-cfg-eng.jsonl", rows(self.inc, inc))
        self.n = 0
        self.verdicts: list[dict] = []

    def add(self, up: int, down: int, seed: int | None = None, name: str | None = None, wins: dict | None = None, stamp: str | None = None) -> dict:
        self.n += 1
        cand = name or f"c{self.n:04d}"
        w = wins if wins is not None else candidate_wins(self.inc, up, down, seed if seed is not None else 100 + self.n)
        cand_rows = rows(w, cand)
        write_rows(self.loop.gates / f"{cand}-cfg-eng.jsonl", cand_rows)
        pool = gl.pool_members(self.loop, self.inc_name, "cfg", self.rule, cand)
        v = gl.decide(cand_rows, rows(self.inc, self.inc_name), 2, self.rule, pool=pool)
        row = {"t": stamp or f"2026-11-1{self.n % 10}T00:00:00Z", "candidate": cand, "incumbent": self.inc_name, "config": "cfg", "engine": "eng",
               "rule": self.rule["name"], **v}
        with open(self.loop.gates / "verdicts.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        self.verdicts.append(v)
        return v


class StageTwoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_level_candidates_never_promote(self):
        seq = Sequence(self.tmp.name)
        for i in range(8):
            v = seq.add(0, 0, name=f"id{i}")  # the incumbent's own games again
            self.assertEqual((v["verdict"], v["win_diff"]), ("reject", 0.0), i)
        for i in range(10):
            v = seq.add(60, 60)  # 120 pairs turned over each way: no net difference, a different set every time
            self.assertEqual(v["verdict"], "reject", i)
            self.assertAlmostEqual(v["win_diff"], 0.0)
            self.assertLessEqual(v["pool_k"], 5)
        self.assertEqual({v["pool_k"] for v in seq.verdicts[-5:]}, {5})  # the pool is the last five of this incumbent

    def test_a_candidate_not_better_than_level_is_no_trend_even_with_a_good_look_now_and_then(self):
        seq = Sequence(self.tmp.name)
        for up, down in [(65, 60), (60, 65), (62, 60), (60, 62), (66, 60), (60, 64), (61, 60)]:
            self.assertEqual(seq.add(up, down)["verdict"], "reject")

    def test_a_trend_of_plus_point_six_promotes_by_the_pool_within_five_candidates_and_not_before_three(self):
        seq = Sequence(self.tmp.name)
        got = [seq.add(72, 60) for _ in range(5)]  # +0.6 points each: McNemar alone is p 0.15, the pool of three is p 0.04
        self.assertEqual([v["pool_k"] for v in got], [1, 2, 3, 4, 5])
        self.assertEqual([v["verdict"] for v in got][:2], ["reject", "reject"])
        self.assertIn("pool of 1 under the 3 candidates a trend needs", got[0]["why"])
        self.assertIn("pool of 2 under", got[1]["why"])
        first = next(i for i, v in enumerate(got) if v["verdict"] == "promote")
        self.assertEqual(first, 2)
        v = got[first]
        self.assertEqual(v["path"], "pool")
        self.assertGreater(v["diff_lb"], -0.01)  # not worse
        self.assertLess(v["pool_p"], 0.10)
        self.assertGreater(v["p"], 0.05)  # the candidate alone would not have
        self.assertGreater(v["pool_diff"], 0)

    def test_a_trend_of_a_point_promotes_within_five_candidates(self):
        seq = Sequence(self.tmp.name)
        got = [seq.add(80, 60) for _ in range(5)]
        self.assertTrue(any(v["verdict"] == "promote" for v in got), [(v["verdict"], v["pool_p"]) for v in got])
        self.assertEqual(next(v for v in got if v["verdict"] == "promote")["stage"], 2)

    def test_a_clearly_better_candidate_promotes_alone_by_v2s_test(self):
        seq = Sequence(self.tmp.name)
        v = seq.add(110, 70)  # +2.0 points, McNemar p 0.007, first in its pool
        self.assertEqual((v["verdict"], v["path"], v["pool_k"]), ("promote", "single", 1))
        self.assertLess(v["p"], 0.05)

    def test_the_old_rules_own_margin_is_kept_not_the_briefs_plus_one_and_a_half_floor(self):
        seq = Sequence(self.tmp.name)
        v = seq.add(100, 70)  # +1.5 points here; the margin of v2's promotions was +1.3 to +1.6 at p 0.013 to 0.025
        self.assertEqual((v["verdict"], v["path"]), ("promote", "single"))
        floor = {**RULE, "single": {**RULE["single"], "min_points": 3.0}}
        seq2 = Sequence(self.tmp.name + "-b", floor)
        self.assertEqual(seq2.add(100, 70)["verdict"], "reject")  # the constant can ask for a floor on top

    def test_a_candidate_with_less_l0_is_not_promoted_alone(self):
        seq = Sequence(self.tmp.name)
        w = candidate_wins(seq.inc, 110, 70, 7)
        cand = rows(w, "c")
        for r in cand:  # wins as often, but leaves less HP where it wins
            r["L0"] = 0.1 if r["won"] else 0.0
        v = gl.decide(cand, rows(seq.inc, "v1"), 2, RULE, pool=[])
        self.assertLess(v["l0_diff"], 0)
        self.assertEqual(v["verdict"], "reject")

    def test_a_worse_candidate_never_promotes_by_the_pool_whatever_the_pool_says(self):
        seq = Sequence(self.tmp.name)
        seq.add(100, 40, name="good1")  # +3 points each (they stand in the ledger as verdicts; the rule is on the last one)
        seq.add(100, 40, name="good2")
        v = seq.add(45, 75)  # -1.5 points: pooled over three it is still +1.5
        self.assertGreater(v["pool_diff"], 0.01)
        self.assertLess(v["pool_p"], 0.10)
        self.assertEqual(v["verdict"], "reject")
        self.assertIn("not worse fails", v["why"])

    def test_the_pool_is_this_incumbents_last_five_of_this_rule_version_and_never_the_candidate_itself(self):
        seq = Sequence(self.tmp.name)
        loop = seq.loop

        def row(cand: str, **kw) -> None:
            base = {"t": "2026-11-10T00:00:00Z", "candidate": cand, "incumbent": "v1", "config": "cfg", "engine": "eng", "rule": RULE["name"],
                    "stage": 2, "verdict": "reject"}
            with open(loop.gates / "verdicts.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({**base, **kw}) + "\n")

        for name in ("m1", "m2", "m3", "m4", "m5", "m6"):  # six members with games
            write_rows(loop.gates / f"{name}-cfg-eng.jsonl", rows(candidate_wins(seq.inc, 70, 60, hash(name) % 1000), name))
            row(name)
        row("old", rule="gym-loop-promote-v2")  # a v2 row: selected on its stage-1 floor, not in the pool
        write_rows(loop.gates / "old-cfg-eng.jsonl", rows(dict(seq.inc), "old"))
        row("s1", stage=1)  # a stage-1 verdict
        row("other", incumbent="v0")  # another incumbent
        write_rows(loop.gates / "other-cfg-eng.jsonl", rows(dict(seq.inc), "other"))
        row("gone")  # no games on disk
        row("m6")  # the candidate gated twice (a stopped gate resumed): once
        got = gl.pool_members(loop, "v1", "cfg", RULE, "me")
        self.assertEqual([n for n, _ in got], ["m3", "m4", "m5", "m6"])  # the newest four, oldest first
        self.assertEqual([n for n, _ in gl.pool_members(loop, "v1", "cfg", RULE, "m6")], ["m2", "m3", "m4", "m5"])  # m6 is the one being gated
        self.assertEqual(gl.pool_members(loop, "v2", "cfg", RULE, "me"), [])  # a promotion restarts the pool
        self.assertEqual(gl.pool_members(loop, "v1", "other-cfg", RULE, "me"), [])  # another config or judge: another games file
        self.assertTrue(all(len(d) == 2000 for _, d in got))

    def test_a_promotion_restarts_the_pool(self):
        seq = Sequence(self.tmp.name)
        got = [seq.add(72, 60) for _ in range(3)]
        self.assertEqual(got[-1]["verdict"], "promote")
        nxt = Sequence(self.tmp.name + "-v2", inc="v2")  # the next incumbent's gate: its own incumbent games, an empty ledger of its own
        v = nxt.add(72, 60)
        self.assertEqual((v["pool_k"], v["verdict"]), (1, "reject"))

    def test_the_pool_may_span_engines(self):
        seq = Sequence(self.tmp.name)
        seq.add(72, 60)
        seq.add(72, 60)
        # the third candidate on another engine: its own incumbent games, and the ledger row says so
        write_rows(seq.loop.gates / "inc-v1-cfg-eng2.jsonl", rows(seq.inc, "v1"))
        w = candidate_wins(seq.inc, 72, 60, 555)
        write_rows(seq.loop.gates / "c0003-cfg-eng2.jsonl", rows(w, "c0003"))
        pool = gl.pool_members(seq.loop, "v1", "cfg", RULE, "c0003")
        v = gl.decide(rows(w, "c0003"), rows(seq.inc, "v1"), 2, RULE, pool=pool)
        self.assertEqual((v["pool_k"], v["verdict"]), (3, "promote"))
        self.assertEqual(v["pool_prior"], ["c0001", "c0002"])

    def test_a_v2_rule_still_decides_as_it_did(self):
        v = gl.decide(rows(candidate_wins(incumbent_wins(), 110, 70, 5), "c"), rows(incumbent_wins(), "v"), 2, gl.PROMOTION_RULE_V2)
        self.assertEqual(v["verdict"], "promote")
        self.assertNotIn("pool_k", v)
        lvl = gl.decide(rows(incumbent_wins(), "c"), rows(incumbent_wins(), "v"), 1, gl.PROMOTION_RULE_V2, pair=2)
        self.assertEqual(lvl["verdict"], "reject")  # +0.0 is under v2's +1.0 floor; v3's screen lets it on


class GuardTests(unittest.TestCase):
    def test_one_promotion_in_six_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = gl.Loop(Path(tmp) / "loop")
            now = dt.datetime(2026, 11, 12, 12, 0, 0, tzinfo=dt.timezone.utc)
            promote = {"verdict": "promote", "stage": 2, "path": "pool", "cand_wins": 5}
            self.assertEqual(gl.promotion_guard(loop, promote, now=now), promote)  # no promotion yet: nothing to guard
            loop.state["promotions"].append({"version": "v3", "checkpoint": "c0185", "over": "v2", "t": "2026-11-12T09:30:00Z"})
            g = gl.promotion_guard(loop, promote, now=now)
            self.assertEqual((g["verdict"], g["path"], g["guarded_path"]), ("reject", None, "pool"))
            self.assertIn("v3 was promoted 2.5 h ago", g["guard"])
            self.assertIn("guard:", g["why"])
            self.assertEqual(g["cand_wins"], 5)  # the games' figures are kept
            late = gl.promotion_guard(loop, promote, now=now + dt.timedelta(hours=3.6))  # 6.1 h
            self.assertEqual(late["verdict"], "promote")
            rej = {"verdict": "reject", "stage": 2}
            self.assertEqual(gl.promotion_guard(loop, rej, now=now), rej)
            loop.state["promotions"][-1]["t"] = "x"  # an unreadable time: no guard
            self.assertEqual(gl.promotion_guard(loop, promote, now=now), promote)
            self.assertEqual(gl.promotion_guard(loop, promote, {**RULE, "guard_h": 0}, now=now), promote)


def ledger_rows(points: list[float], t0: dt.datetime, step_h: float = 1.0, stage: int = 1) -> list[dict]:
    return [{"t": (t0 + dt.timedelta(hours=step_h * i)).strftime("%Y-%m-%dT%H:%M:%SZ"), "candidate": f"c{i:04d}", "stage": stage,
             "verdict": "reject", "win_diff": p / 100.0, "pairs": 400} for i, p in enumerate(points)]


class PlateauTests(unittest.TestCase):
    T0 = dt.datetime(2026, 11, 9, 12, 0, 0, tzinfo=dt.timezone.utc)

    def promos(self, t: str = "2026-11-09T12:00:00Z") -> list[dict]:
        return [{"version": "v3", "checkpoint": "c0185", "over": "v2", "t": t}]

    def test_twenty_five_gated_candidates_trip_it_and_twenty_four_do_not(self):
        led = ledger_rows([0.1] * 24, self.T0 + dt.timedelta(minutes=5), step_h=0.1)
        now = self.T0 + dt.timedelta(days=2)
        st = gg.plateau_state(led, self.promos(), now)
        self.assertEqual((st["candidates"], st["tripped"]), (24, False))
        led2 = led + ledger_rows([0.2], self.T0 + dt.timedelta(hours=5), stage=2)
        led2[-1]["candidate"] = "c9999"
        st = gg.plateau_state(led2, self.promos(), now)
        self.assertEqual((st["candidates"], st["tripped"]), (25, True))
        self.assertAlmostEqual(st["days"], 2.0)

    def test_five_days_trip_it_with_few_candidates(self):
        led = ledger_rows([0.1] * 3, self.T0 + dt.timedelta(hours=1))
        self.assertFalse(gg.plateau_state(led, self.promos(), self.T0 + dt.timedelta(days=4, hours=23))["tripped"])
        st = gg.plateau_state(led, self.promos(), self.T0 + dt.timedelta(days=5))
        self.assertEqual((st["tripped"], st["candidates"]), (True, 3))

    def test_only_candidates_after_the_last_promotion_count_and_skipped_ones_never_do(self):
        led = ledger_rows([0.1] * 40, self.T0 - dt.timedelta(days=3), step_h=1.0)  # before the promotion: 40 rows, an old stall
        led += [{"t": "2026-11-10T00:00:00Z", "candidate": "c1", "stage": 1, "verdict": "skipped"},
                {"t": "2026-11-10T01:00:00Z", "candidate": "c2", "stage": 2, "verdict": "reject", "win_diff": 0.001, "pairs": 2000},
                {"t": "2026-11-10T02:00:00Z", "candidate": "c3", "stage": 1, "verdict": "incomplete"}]
        st = gg.plateau_state(led, self.promos(), self.T0 + dt.timedelta(days=1))
        self.assertEqual(st["candidates"], 1)
        self.assertFalse(st["tripped"])

    def test_with_no_promotion_it_counts_from_the_first_verdict(self):
        led = ledger_rows([0.0] * 30, self.T0, step_h=0.5)
        st = gg.plateau_state(led, [], self.T0 + dt.timedelta(days=1))
        self.assertEqual((st["since"], st["candidates"], st["tripped"]), ("2026-11-09T12:00:00Z", 29, True))

    def test_the_file_holds_days_candidates_and_the_last_ten_verdicts_and_the_log_speaks_once(self):
        led = ledger_rows([0.5, -0.5, 1.1, 0.0, 0.25, 0.3, 0.1, -1.0, 0.7, 0.2, 0.4, 0.45, 0.9] * 2, self.T0 + dt.timedelta(minutes=30), step_h=0.2)
        logs: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            alerts = Path(tmp) / "alerts"
            now = self.T0 + dt.timedelta(days=1, hours=2)  # 2026-11-10 14:00Z
            self.assertEqual(gg.plateau_update(led, self.promos(), "v3", alerts, logs.append, now), "raised")
            f = alerts / "gym-plateau.txt"
            text = f.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("ALERT gym-plateau: no promotion for 1.1 days and 26 gated candidates"), text.splitlines()[0])
            self.assertIn("incumbent v3", text)
            body = [x for x in text.splitlines() if x.startswith("  c")]
            self.assertEqual(len(body), 10)  # the last ten
            self.assertEqual(body[-1], "  c0025 stage 1 reject +0.90 points (400 pairs)")
            self.assertEqual(body[0], "  c0016 stage 1 reject +0.00 points (400 pairs)")
            self.assertEqual(len(logs), 1)
            self.assertIn("wrote alerts/gym-plateau.txt", logs[0])
            # standing: rewritten with the newer figures, no second log line
            led.append({"t": "2026-11-10T14:30:00Z", "candidate": "c0100", "stage": 2, "verdict": "reject", "win_diff": 0.003, "pairs": 2000})
            self.assertEqual(gg.plateau_update(led, self.promos(), "v3", alerts, logs.append, now + dt.timedelta(hours=1)), "standing")
            self.assertIn("c0100 stage 2 reject +0.30 points (2000 pairs)", f.read_text(encoding="utf-8"))
            self.assertEqual(len(logs), 1)
            # a promotion takes it down
            promos = self.promos() + [{"version": "v4", "checkpoint": "c0100", "over": "v3", "t": "2026-11-10T15:30:00Z"}]
            self.assertEqual(gg.plateau_update(led, promos, "v4", alerts, logs.append, now + dt.timedelta(hours=2)), "cleared")
            self.assertFalse(f.exists())
            self.assertIn("plateau: cleared", logs[-1])
            self.assertEqual(gg.plateau_update(led, promos, "v4", alerts, logs.append, now + dt.timedelta(hours=3)), "quiet")
            self.assertEqual(len(logs), 2)

    def test_the_labs_status_prints_it_as_an_alert(self):
        from tools.lab import alerts as lab_alerts  # noqa: PLC0415

        led = ledger_rows([0.0] * 26, self.T0 + dt.timedelta(minutes=5), step_h=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            gg.plateau_update(led, self.promos(), "v3", Path(tmp) / "alerts", lambda m: None, self.T0 + dt.timedelta(hours=4))
            self.assertEqual(lab_alerts.status_prefix(tmp), "ALERT gym-plateau (LAB/alerts)")
            self.assertIn("gym-plateau.txt: ALERT gym-plateau: no promotion", lab_alerts.banner(tmp))

    def test_the_daemons_check_is_off_unless_asked_writes_after_five_days_and_a_promotion_removes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = gl.Loop(Path(tmp) / "loop")
            loop.state["promotions"] = self.promos("2026-11-01T00:00:00Z")
            loop.state["incumbent"] = "v3"
            with open(loop.gates / "verdicts.jsonl", "w", encoding="utf-8") as fh:
                for r in ledger_rows([0.1] * 3, dt.datetime(2026, 11, 2, tzinfo=dt.timezone.utc)):
                    fh.write(json.dumps(r) + "\n")
            alerts = Path(tmp) / "alerts"
            d = gl.Daemon.__new__(gl.Daemon)
            d.loop = loop
            d.a = argparse.Namespace()  # a daemon built without the flag writes nothing (these tests never touch the real LAB/alerts)
            with mock.patch.object(gl, "utc", lambda: dt.datetime(2026, 11, 9, tzinfo=dt.timezone.utc)):
                d.plateau(force=True)
            self.assertFalse(alerts.exists())
            d.a = argparse.Namespace(plateau_alert="on", alerts_dir=str(alerts))
            with mock.patch.object(gl, "utc", lambda: dt.datetime(2026, 11, 4, tzinfo=dt.timezone.utc)):
                d.plateau(force=True)
                self.assertFalse((alerts / "gym-plateau.txt").exists())  # three days
            with mock.patch.object(gl, "utc", lambda: dt.datetime(2026, 11, 6, 1, tzinfo=dt.timezone.utc)):
                d.plateau()  # inside the ten-minute spacing since the last look
                self.assertFalse((alerts / "gym-plateau.txt").exists())
                d.plateau(force=True)
                self.assertTrue((alerts / "gym-plateau.txt").exists())  # five days and an hour
            loop.state["promotions"].append({"version": "v4", "checkpoint": "c9", "over": "v3", "t": "2026-11-06T02:00:00Z"})
            with mock.patch.object(gl, "utc", lambda: dt.datetime(2026, 11, 6, 3, tzinfo=dt.timezone.utc)):
                d.plateau(force=True)
            self.assertFalse((alerts / "gym-plateau.txt").exists())
            self.assertIn("plateau: no promotion for 5.0 days", (loop.root / "loop.log").read_text(encoding="utf-8"))
            self.assertIn("plateau: cleared", (loop.root / "loop.log").read_text(encoding="utf-8"))
            d.a = argparse.Namespace(plateau_alert="off", alerts_dir=str(alerts))
            self.assertIsNone(d.plateau(force=True))

    def test_the_cli_flags(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(gl, "Daemon") as daemon:
            daemon.return_value.run.return_value = 0
            gl.main(["run", "--root", tmp])
            gl.main(["run", "--root", tmp, "--plateau-alert", "off", "--alerts-dir", tmp])
            a, b = (c.args[0] for c in daemon.call_args_list)
            self.assertEqual((a.plateau_alert, a.human_bar, Path(a.alerts_dir).name), ("on", "promotion", "alerts"))
            self.assertEqual((b.plateau_alert, b.alerts_dir), ("off", tmp))


class GateEndToEndTests(unittest.TestCase):
    """`Daemon.gate` and `run_gate` on faked play: the verdicts a run of candidates gets, as the ledger and the log record them."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.loop = gl.Loop(t / "loop")
        self.agent = t / "agent"
        self.agent.mkdir()
        (self.agent / "model.json").write_text("{}", encoding="utf-8")
        self.loop.state["versions"]["v1"] = {"dir": str(self.agent), "n": 1}
        self.loop.state["incumbent"] = "v1"
        self.loop.state["gate_pair"] = 2
        self.wins: dict[str, dict] = {"v1": incumbent_wins()}
        self.board = [{"id": s, "cat": "act1-boss"} for s in STARTS]
        self.played: list[tuple[str, int]] = []

    def tearDown(self):
        self.tmp.cleanup()

    def promoted_as(self, cand: str) -> None:
        """A promotion makes the candidate the incumbent: the fake player's games under the new version's name are the candidate's."""
        self.wins[self.loop.state["incumbent"]] = self.wins[cand]

    def fake_play(self, loop, tasks, target, player_dir, procs, k, prior, what, **kw):
        with open(target, "a", encoding="utf-8") as fh:
            for _kind, sid, j, _cat, info in tasks:
                self.played.append((info["label"], j))
                w = self.wins[info["label"]][(sid, j)]
                fh.write(json.dumps({"start": sid, "j": j, "won": bool(w), "capped": False, "L0": 0.5 if w else 0.0, "arm": info["label"]}) + "\n")
        return None

    def daemon(self) -> gl.Daemon:
        a = argparse.Namespace(player_config="cfg", procs=1, gate_k=4, prior="cpu", judge="", prefilter="off", board="off", human_bar="off")
        d = gl.Daemon.__new__(gl.Daemon)
        d.a, d.loop = a, self.loop
        d.status = lambda phase, **k: None
        return d

    def gate(self, d: gl.Daemon, cand: str, up: int, down: int, seed: int) -> dict:
        self.wins[cand] = candidate_wins(self.wins[self.loop.state["incumbent"]], up, down, seed)
        before = self.loop.state["incumbent"]
        with mock.patch.object(gl, "play_tasks", self.fake_play), mock.patch.object(gl, "engine_tag", lambda: "eng"), \
                mock.patch.object(gl, "board_starts", lambda: self.board):
            d.gate(cand, self.agent)
        if self.loop.state["incumbent"] != before:
            self.promoted_as(cand)
        rows_ = [json.loads(x) for x in (self.loop.gates / "verdicts.jsonl").read_text(encoding="utf-8").splitlines()]
        return rows_[-1]

    def test_a_run_of_candidates_a_level_one_a_screened_one_a_trend_and_the_guard_after_it(self):
        d = self.daemon()
        # a clearly worse candidate is stopped at stage 1 after 400 games
        bad = {k: (0 if (k[1] in (4, 5) and self.wins["v1"][k] and i % 3 == 0) else self.wins["v1"][k]) for i, k in enumerate(KEYS)}
        self.wins["c0001"] = bad
        with mock.patch.object(gl, "play_tasks", self.fake_play), mock.patch.object(gl, "engine_tag", lambda: "eng"), \
                mock.patch.object(gl, "board_starts", lambda: self.board):
            d.gate("c0001", self.agent)
        led = json.loads((self.loop.gates / "verdicts.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual((led["candidate"], led["stage"], led["verdict"], led["pairs"], led["rule"]), ("c0001", 1, "reject", 400, "gym-loop-promote-v3"))
        self.assertLess(led["diff_bound"], -0.02)
        self.assertEqual(sorted({j for who, j in self.played if who == "c0001"}), [4, 5])  # stage 1 played the pair, nothing more
        # a level candidate (the incumbent's games again) goes to stage 2 and is rejected there, with the pool's numbers on the ledger
        led = self.gate(d, "c0002", 0, 0, 1)
        self.assertEqual((led["stage"], led["verdict"], led["pairs"], led["pool_k"]), (2, "reject", 2000, 1))
        self.assertEqual(self.loop.state["gated"]["c0002"]["pool_k"], 1)
        # the trend: +0.6 points a candidate on fresh seeds; the pool is c0002 (level) and the new ones
        led = self.gate(d, "c0003", 72, 60, 3)
        self.assertEqual((led["verdict"], led["pool_k"]), ("reject", 2))  # a pool of two does not speak
        self.assertIn("pool of 2 under the 3", led["why"])
        led = self.gate(d, "c0004", 72, 60, 4)  # level, +0.6, +0.6: pooled +0.4, p 0.07, and this one not worse
        self.assertEqual((led["verdict"], led["path"], led["pool_k"]), ("promote", "pool", 3))
        self.assertGreater(led["p"], 0.05)  # v2's test alone would not have
        self.assertEqual(self.loop.state["incumbent"], "v2")
        self.assertEqual(self.loop.state["gated"]["c0004"]["path"], "pool")
        # the new incumbent starts a new pool: a level candidate against v2 has a pool of one
        led = self.gate(d, "c0005", 0, 0, 5)
        self.assertEqual((led["incumbent"], led["pool_k"], led["verdict"]), ("v2", 1, "reject"))
        log = (self.loop.root / "loop.log").read_text(encoding="utf-8")
        self.assertRegex(log, r"gate c0002: stage 2 reject: cand \d+ inc \d+ of 2000 \(\+0\.0 points, SE [\d.]+\) p [\d.]+; lower bound [+-][\d.]+ points; "
                               r"pool of 1: \+0\.00 points \(SE [\d.]+\), p 1\.000; reject: pool of 1 under the 3 candidates a trend needs")
        self.assertIn("promote by path pool", log)
        self.assertIn("promoted c0004 as v2 over v1", log)
        self.assertIn("95% upper bound", log)  # stage 1's line

    def test_a_clearly_better_candidate_right_after_a_promotion_is_held_by_the_six_hour_guard(self):
        d = self.daemon()
        self.loop.state["promotions"].append({"version": "v1", "checkpoint": "c0000", "over": "v0", "t": gl.stamp()})
        led = self.gate(d, "c0001", 120, 60, 9)  # +3 points: passes v2's test alone
        self.assertEqual((led["stage"], led["verdict"], led["path"]), (2, "reject", None))
        self.assertIn("v1 was promoted 0.0 h ago", led["guard"])
        self.assertEqual(led["guarded_path"], "single")
        self.assertEqual(self.loop.state["incumbent"], "v1")  # nothing was promoted
        self.assertIn("guard: v1 was promoted", (self.loop.root / "loop.log").read_text(encoding="utf-8"))
        self.loop.state["promotions"][-1]["t"] = "2026-01-01T00:00:00Z"  # long ago
        led = self.gate(d, "c0002", 120, 60, 10)
        self.assertEqual((led["verdict"], led["path"]), ("promote", "single"))
        self.assertEqual(self.loop.state["incumbent"], "v2")

    def test_a_stage_one_verdict_leaves_the_pool_alone_and_a_stopped_gate_writes_nothing(self):
        d = self.daemon()

        def stop_play(loop, tasks, target, player_dir, procs, k, prior, what, **kw):
            return "night-window"

        with mock.patch.object(gl, "play_tasks", stop_play), mock.patch.object(gl, "engine_tag", lambda: "eng"), \
                mock.patch.object(gl, "board_starts", lambda: self.board):
            d.gate("c0001", self.agent)
        self.assertFalse((self.loop.gates / "verdicts.jsonl").exists())
        self.assertEqual(self.loop.state["gated"], {})


class RuleConstantsTests(unittest.TestCase):
    def test_the_rule_is_v3_and_says_what_it_decides(self):
        self.assertEqual(RULE["name"], "gym-loop-promote-v3")
        self.assertEqual(RULE["screen"], {"bound": "upper", "z": gg.Z95, "points": -2.0})
        self.assertEqual(RULE["pool"], {"k": 5, "min": 3, "alpha": 0.10, "not_worse_points": -1.0, "z_lower": gg.Z95})
        self.assertEqual((RULE["single"]["alpha"], RULE["guard_h"], RULE["seeds"], RULE["starts"]), (0.05, 6.0, 10, 200))
        self.assertEqual(gg.PLATEAU, {"days": 5.0, "candidates": 25, "last": 10, "file": "gym-plateau.txt"})
        self.assertEqual(gl.HUMAN_BAR["prior"], "cpu")
        self.assertEqual(gl.HUMAN_BAR["config"], "a3-v1")

    def test_the_games_the_rule_replays_are_the_gates_and_the_rows_are_read_with_the_gates_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "verdicts.jsonl"
            p.write_text('{"a": 1}\n{"b"', encoding="utf-8")  # a line cut by a stop
            self.assertEqual(gg.read_ledger(p), [{"a": 1}])
            self.assertEqual(gg.read_ledger(Path(tmp) / "none.jsonl"), [])
        self.assertEqual(fg.read_jsonl(Path("no-such-file")), [])


if __name__ == "__main__":
    unittest.main()
