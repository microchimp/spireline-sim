"""The arrival exam (S1 of the judge design, judge build 1, D68 addendum 6 outcome 3): rl/act3_starts.py (the ruler of the champion's first-Act-3-boss
arrivals rebuilt from its judged arms, and the driver that plays it) and rl/gym_gate.py (`arrival_exam`, the reading).

The pinned facts, on a tiny synthetic ruler: test k plays draws 6k to 6k + 5; the statistic is the per-start mean of the survival difference (candidate
minus champion, a capped game lost) averaged unweighted over the starts, its SE taken over the starts, and nothing counts but the pairs both players
played on the test's draws; the 0.005 lower bound and the 95% upper bound are printed with what each would mean and no rule reads them; the ruler keeps
a replay only when it matches its row, takes the second boss only for a run that entered it, leaves out the blocks the champion was selected on, and
its hash is the starts' own (the order they were written in and the annotations on them do not change it; an edited file does not load); the driver
plays the champion first, plays only the games with no row, and refuses two players that hold the same network. Nothing plays a game: the replay step's
pool (`fight_gym.run_tasks`) and the lockstep (`fight_exam.play_exam_tasks`) are replaced.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
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

import act3_starts as a3  # noqa: E402
import fight_exam as fe  # noqa: E402
import fight_gym as fg  # noqa: E402
import gym_gate as gg  # noqa: E402
import gym_loop as gl  # noqa: E402

PM = "±"


def game(start, j, won, capped=False, eng="E2"):
    return {"start": start, "j": j, "won": won, "capped": capped, "engine": eng, "config": "a3-v1", "start_hp": 60, "end_hp": 30 if won else 0}


def games(spec: dict[str, list[int]], offset: int = 0) -> dict:
    """{(start, j): row} for `spec` {start: [won per draw]} (draw numbers from `offset`)."""
    return gg.indexed([game(s, offset + j, bool(w)) for s, ws in spec.items() for j, w in enumerate(ws)])


def sd_mean(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


class DrawIndexingTests(unittest.TestCase):
    def test_test_k_plays_draws_6k_to_6k_plus_5(self):
        self.assertEqual(gg.arrival_js(0), (0, 1, 2, 3, 4, 5))
        self.assertEqual(gg.arrival_js(1), (6, 7, 8, 9, 10, 11))
        self.assertEqual(gg.arrival_js(3), (18, 19, 20, 21, 22, 23))
        self.assertEqual(gg.arrival_js(2, draws=4), (8, 9, 10, 11))
        self.assertEqual(gg.ARRIVAL_DRAWS, 6)
        with self.assertRaises(ValueError):
            gg.arrival_js(-1)


class StatisticTests(unittest.TestCase):
    CAND = {"a": [1, 1, 1, 0, 0, 0], "b": [0] * 6, "c": [1, 1, 0, 0, 0, 0], "d": [1] * 6}
    INC = {"a": [0] * 6, "b": [1, 0, 0, 0, 0, 0], "c": [1, 1, 0, 0, 0, 0], "d": [1, 1, 1, 1, 0, 0]}

    def test_the_statistic_is_the_per_start_mean_averaged_unweighted_with_the_se_over_starts(self):
        r = gg.arrival_exam(games(self.CAND), games(self.INC))
        means = [3 / 6, -1 / 6, 0.0, 2 / 6]  # per start: candidate minus champion over the six draws
        m, se = sd_mean(means)
        self.assertEqual((r["starts"], r["pairs"], r["complete"]), (4, 24, True))
        self.assertAlmostEqual(r["diff"], m)
        self.assertAlmostEqual(r["se"], se)
        self.assertAlmostEqual(r["pm2se"], 2 * se)
        self.assertAlmostEqual(r["points"], 100 * m)  # passes per 100 arrivals
        self.assertAlmostEqual(r["se_points"], 100 * se)
        self.assertAlmostEqual(r["pm2se_points"], 200 * se)
        # both rates are per-start means averaged unweighted
        self.assertAlmostEqual(r["cand_rate"], (3 / 6 + 0 + 2 / 6 + 1) / 4)
        self.assertAlmostEqual(r["inc_rate"], (0 + 1 / 6 + 2 / 6 + 4 / 6) / 4)
        self.assertTrue(r["report_only"])
        self.assertEqual((r["test"], r["draws"], r["draws_per_start"]), (0, [0, 1, 2, 3, 4, 5], 6))

    def test_the_se_is_taken_over_starts_not_over_games(self):
        # the same start six times over adds no information: two starts whose six draws agree have an SE from the two start means, not from twelve games
        cand = {"a": [1] * 6, "b": [0] * 6}
        inc = {"a": [0] * 6, "b": [0] * 6}
        r = gg.arrival_exam(games(cand), games(inc))
        self.assertAlmostEqual(r["diff"], 0.5)
        self.assertAlmostEqual(r["se"], sd_mean([1.0, 0.0])[1])  # 0.5, not the 0.144 a pair-level SE over twelve games would give
        self.assertEqual(r["pairs"], 12)

    def test_a_start_with_fewer_draws_counts_once_like_any_other_and_the_test_is_incomplete(self):
        cand = {"a": [1] * 6, "b": [1, 0, 0]}
        inc = {"a": [0] * 6, "b": [0, 0, 0]}
        r = gg.arrival_exam(games(cand), games(inc))
        self.assertAlmostEqual(r["diff"], (1.0 + 1 / 3) / 2)  # unweighted over starts: b's three draws weigh the same as a's six
        self.assertFalse(r["complete"])
        # an id that was asked for and never played also makes it incomplete
        self.assertFalse(gg.arrival_exam(games({"a": [1] * 6}), games({"a": [0] * 6}), ids={"a", "z"})["complete"])
        self.assertTrue(gg.arrival_exam(games({"a": [1] * 6}), games({"a": [0] * 6}), ids={"a"})["complete"])

    def test_only_the_pairs_both_played_on_the_tests_draws_count(self):
        cand = {**games({"a": [1] * 6, "b": [1] * 6}), **games({"a": [0] * 6}, offset=6), **games({"c": [1] * 6})}
        inc = {**games({"a": [0] * 6, "b": [0] * 6}), **games({"a": [1] * 6}, offset=6)}  # c: the champion never played it
        r0 = gg.arrival_exam(cand, inc, test=0)
        self.assertEqual((r0["starts"], r0["pairs"], r0["points"]), (2, 12, 100.0))
        r1 = gg.arrival_exam(cand, inc, test=1)  # draws 6..11: only start a, where the candidate lost all six and the champion won them
        self.assertEqual((r1["starts"], r1["pairs"], r1["draws"]), (1, 6, [6, 7, 8, 9, 10, 11]))
        self.assertAlmostEqual(r1["diff"], -1.0)
        self.assertIsNone(r1["se"])  # one start: no spread to read
        self.assertIsNone(r1["lb995"])
        self.assertEqual(r1["reads"], {"hold": False, "reversal": False})
        # a start list restricts the reading
        self.assertEqual(gg.arrival_exam(cand, inc, ids={"b"})["starts"], 1)
        self.assertEqual(gg.arrival_exam({}, {})["starts"], 0)

    def test_a_capped_game_is_a_loss(self):
        cand = gg.indexed([game("a", 0, True, capped=True), game("a", 1, True)])
        inc = gg.indexed([game("a", 0, False), game("a", 1, False)])
        r = gg.arrival_exam(cand, inc, draws=2)
        self.assertAlmostEqual(r["diff"], 0.5)  # the capped win is a loss: one pair in two

    def test_by_cell_and_the_rulers_hash_are_carried(self):
        r = gg.arrival_exam(games(self.CAND), games(self.INC), cells={"a": "x", "b": "x", "c": "y", "d": "y"}, ruler_hash="abc123")
        self.assertEqual(r["ruler_hash"], "abc123")
        self.assertEqual(sorted(r["by_cell"]), ["x", "y"])
        self.assertAlmostEqual(r["by_cell"]["x"]["diff"], (3 / 6 - 1 / 6) / 2)
        self.assertEqual(r["by_cell"]["y"]["starts"], 2)


class BoundTests(unittest.TestCase):
    """The two bounds the design names, printed and never acted on."""

    def flat(self, n_up: int, n: int, down: bool = False):
        """`n` starts where `n_up` start means differ by +1 (or -1 when `down`) over all six draws and the rest are level."""
        step = [0] * 6
        up = {f"s{i}": ([1] * 6 if not down else [0] * 6) for i in range(n_up)}
        level = {f"s{i}": step for i in range(n_up, n)}
        cand = {**up, **level}
        inc = {**{f"s{i}": ([0] * 6 if not down else [1] * 6) for i in range(n_up)}, **level}
        return gg.arrival_exam(games(cand), games(inc))

    def test_the_lower_bound_at_0_005_and_the_upper_bound_at_95_are_the_normal_bounds(self):
        r = self.flat(20, 200)  # +10 passes per 100, SE 2.1: the 0.005 lower bound is above 0
        m, se = r["diff"], r["se"]
        self.assertAlmostEqual(r["lb995"], m - 2.5758293035489004 * se)
        self.assertAlmostEqual(r["ub95"], m + 1.6448536269514722 * se)
        self.assertGreater(r["lb995"], 0)
        self.assertEqual(r["reads"], {"hold": True, "reversal": False})
        self.assertEqual(r["hold_points"], 1.0)

    def test_a_gain_of_a_point_with_a_wide_bound_does_not_read_as_a_hold(self):
        r = self.flat(1, 100)  # exactly +1.0 point (one start of a hundred), SE 1.0: the 0.005 bound is far below 0
        self.assertAlmostEqual(r["points"], 1.0)
        self.assertLess(r["lb995"], 0)
        self.assertFalse(r["reads"]["hold"])
        self.assertFalse(r["reads"]["reversal"])

    def test_a_gain_under_the_minimum_does_not_read_as_a_hold_even_with_a_tight_bound(self):
        cand = {f"s{i}": [1, 0, 0, 0, 0, 0] for i in range(50)}
        inc = {f"s{i}": [0] * 6 for i in range(50)}
        r = gg.arrival_exam(games(cand), games(inc))  # every start is +1/6: no spread, so the bound is the gain itself
        self.assertAlmostEqual(r["points"], 100 / 6)
        self.assertAlmostEqual(r["se"], 0.0, places=12)
        self.assertAlmostEqual(r["lb995"], r["diff"])
        self.assertTrue(r["reads"]["hold"])
        tight = gg.arrival_exam(games(cand), games(inc), hold_points=20.0)  # the same games, a bar of +20 points: the bound clears 0 and the gain does not
        self.assertGreater(tight["lb995"], 0)
        self.assertFalse(tight["reads"]["hold"])
        self.assertTrue(self.flat(40, 400)["reads"]["hold"])  # +10 points over 400 starts, the bound above 0

    def test_a_95_percent_upper_bound_under_0_reads_as_a_reversal(self):
        r = self.flat(50, 100, down=True)  # -50 points, SE 5
        self.assertLess(r["ub95"], 0)
        self.assertEqual(r["reads"], {"hold": False, "reversal": True})
        r = self.flat(1, 100, down=True)  # -1 point, SE 1: the upper bound +0.64 is above 0: no reversal
        self.assertGreater(r["ub95"], 0)
        self.assertFalse(r["reads"]["reversal"])


class ReportTests(unittest.TestCase):
    def result(self, hold: bool = False):
        first = gg.arrival_exam(games({f"s{i}": [1] * 6 if i < 3 else [0] * 6 for i in range(30)}, offset=6), games({f"s{i}": [0] * 6 for i in range(30)}, offset=6), test=1,
                                ruler_hash="deadbeef", cells={f"s{i}": "act3-boss1-80" for i in range(30)})
        return {"rule": a3.ARRIVAL_RULE, "config": "a3-v1", "prior": "cpu", "engine": "E2", "ruler": "gen20-candidate-chain", "ruler_hash": "deadbeef",
                "candidate": {"dir": "candidate-fight-v5", "policy_sha256": "c" * 64}, "champion": {"dir": "candidate-chain", "policy_sha256": "d" * 64},
                "test": 1, "draws": list(gg.arrival_js(1)), "stopped": None, "first_boss": first,
                "second_boss_record": gg.arrival_exam({}, {})}

    def test_the_report_says_the_reading_its_margin_the_starts_draws_hash_and_both_bounds(self):
        text = gg.arrival_report(self.result())
        for part in ("a report: no rule reads it", "hash deadbeef", "draws 6 to 11 (6 a start, paired)", "30 starts, 180 games", "per 100 arrivals",
                     f"{PM}", "(2 SE)", "0.005 lower bound", "95% upper bound", "a reversal when it is below 0: no", "act3-boss1-80", "CPU prior"):
            self.assertIn(part, text)
        self.assertNotIn("INCOMPLETE", text)
        r = self.result()
        r["first_boss"]["complete"] = False
        r["stopped"] = "night-window"
        text = gg.arrival_report(r)
        self.assertIn("INCOMPLETE", text)
        self.assertIn("stopped: night-window", text)
        self.assertNotIn("second Act 3 boss", text)  # no second-boss starts: no line for them


def nodes_row(seed: str, kind: str, ix: int = 0) -> dict:
    """A judged run: "past" went past the first Act 3 boss (died on floor 49), "died" died at it (floor 48), "early" died on floor 30, "won" won."""
    base = "M" * 15 + "B" + "M" * 15 + "B" + "M" * 14 + "B"  # floors 2 to 48, the bosses on 17, 33 and 48
    nodes, floor, win = {"past": (base + "B", 49, False), "died": (base, 48, False), "early": ("M" * 15 + "B" + "M" * 13, 30, False),
                         "won": (base + "B", 49, True)}[kind]
    return {"seed": seed, "ix": ix, "floor": floor, "win": win, "end": "win" if win else "death", "crash": False, "nodes": nodes}


def replay_row(row: dict, label: str, *, same: bool = True, cap: str | None = None, missing: bool = False, kind_same: bool = True) -> dict:
    """What `fight_gym._replay_task` writes for a task: the starts of the wanted boss floors, with the replay's own checks."""
    fl = a3.arrival_floors(row)
    if cap:
        return {"_task": ["replay", row["seed"], row["ix"]], "capped": True, "cap_reason": cap, "seed": row["seed"], "ix": row["ix"], "same": False,
                "starts": [], "wanted": len(fl), "captured": 0, "path_same": False, "cards_same": False, "rests_same": False, "floor": None, "end": cap}
    starts = [{"id": f"{label}:{row['seed']}|f{f:02d}", "cat": "act3-boss", "kind": "boss", "act": 3, "floor": f, "seed": row["seed"], "ix": row["ix"],
               "night": label, "k": 4, "node_type": 6, "hp": 60 + no, "max_hp": 80, "deck_size": 21, "potions": 1, "relics": 7,
               "encounter_id": 78 if no == 1 else 80, "turn": 0, "doc": {"seed": row["seed"], "floor": f, "hp": 60 + no}, "kind_same": kind_same,
               "replay_same": same} for no, f in fl][: len(fl) - 1 if missing else len(fl)]
    return {"_task": ["replay", row["seed"], row["ix"]], "seed": row["seed"], "ix": row["ix"], "wanted": len(fl), "captured": len(starts), "floor": row["floor"],
            "end": "hooked", "path_same": same, "cards_same": True, "rests_same": True, "same": same and not missing and kind_same, "seconds": 40.0,
            "starts": [{**s, "replay_same": same} for s in starts]}


class RulerPartsTests(unittest.TestCase):
    def test_the_arrivals_of_a_run_are_read_from_its_node_string(self):
        self.assertEqual(a3.arrival_floors(nodes_row("a", "died")), [(1, 48)])
        self.assertEqual(a3.arrival_floors(nodes_row("a", "past")), [(1, 48), (2, 49)])
        self.assertEqual(a3.arrival_floors(nodes_row("a", "won")), [(1, 48), (2, 49)])
        self.assertEqual(a3.arrival_floors(nodes_row("a", "early")), [])
        self.assertEqual(a3.arrival_floors({**nodes_row("a", "died"), "crash": True}), [])
        self.assertEqual(a3.arrival_floors({**nodes_row("a", "died"), "nodes": None}), [])  # no complete node string: not rebuilt
        long_act2 = nodes_row("a", "died")
        long_act2["nodes"] = "M" * 15 + "B" + "M" * 17 + "B" + "M" * 14 + "B"  # an 18-room Act 2: the Act 3 boss is on floor 50
        long_act2["floor"] = len(long_act2["nodes"]) + 1
        self.assertEqual(a3.arrival_floors(long_act2), [(1, 50)])

    def test_tasks_want_the_second_boss_only_for_a_run_that_entered_it_and_take_a_seed_once(self):
        rows = [nodes_row("d1", "died", 1), nodes_row("p1", "past", 2), nodes_row("e1", "early", 3), nodes_row("w1", "won", 4)]
        taken: set = set()
        tasks = a3.arrival_tasks(rows, "B1", "a3-v1", taken)
        self.assertEqual([t[1] for t in tasks], ["d1", "p1", "w1"])
        self.assertEqual({t[1]: [w["floor"] for w in t[4]] for t in tasks}, {"d1": [48], "p1": [48, 49], "w1": [48, 49]})
        t = tasks[1]
        self.assertEqual((t[0], t[2], t[3]), ("replay", 2, "a3-v1"))
        self.assertEqual(t[4][0], {"id": "B1:p1|f48", "cat": "act3-boss", "kind": "boss", "act": 3, "floor": 48, "seed": "p1", "ix": 2, "night": "B1"})
        self.assertIs(t[5], rows[1])  # the replay checks the run against its own row
        self.assertEqual(a3.arrival_tasks(rows, "B2", "a3-v1", taken), [])  # another block that holds the same seeds leaves them to the first

    def test_a_champions_arms_leave_out_the_blocks_it_was_selected_on(self):
        with tempfile.TemporaryDirectory() as d:
            champ = Path(d) / "gen7" / "cand"
            for block in ("E7-2026-11-01", "E8-2026-11-02", "E8-2026-11-02-r1"):
                arm = champ / "arms" / f"{block}-a3-v1"
                arm.mkdir(parents=True)
                (arm / "summary.json").write_text(json.dumps({"config": "a3-v1", "agent": {"dir": "gen7/cand"}}), encoding="utf-8")
                (arm / "runs.jsonl").write_text("", encoding="utf-8")
            (champ / "arms" / "stray").mkdir()  # a folder with no summary or runs is not an arm
            self.assertEqual(a3.selected_blocks(champ), set())
            self.assertEqual(len(a3.champion_arms(champ)), 3)
            (champ / "promotion.json").write_text(json.dumps({"seed_block": "E7-2026-11-01", "pooled": {"blocks": [{"block": "E8-2026-11-02-r1"}]}}), encoding="utf-8")
            self.assertEqual(a3.selected_blocks(champ), {"E7-2026-11-01", "E8-2026-11-02-r1"})
            self.assertEqual([a3.arm_label(p) for p in a3.champion_arms(champ)], ["E8-2026-11-02"])
            self.assertEqual([a3.arm_label(p) for p in a3.champion_arms(champ, exclude=["E8-2026-11-02"])], [])

    def test_only_a_replay_that_matches_its_row_gives_starts(self):
        rows = {r["seed"]: r for r in [nodes_row("p1", "past", 1), nodes_row("d1", "died", 2), nodes_row("d2", "died", 3), nodes_row("d3", "died", 4),
                                      nodes_row("d4", "died", 5), nodes_row("w1", "won", 6)]}
        reps = [replay_row(rows["p1"], "B"), replay_row(rows["d1"], "B", same=False), replay_row(rows["d2"], "B", cap="deadline (killed)"),
                replay_row(rows["d3"], "B", missing=True), replay_row(rows["d4"], "B"), replay_row(rows["w1"], "B")]
        recs, counts = a3.arrival_records(reps, "B", rows, "old")
        self.assertEqual(counts["replays"], 6)
        self.assertEqual((counts["matched"], counts["mismatched"], counts["capped"]), (3, 3, 1))
        self.assertEqual(counts["why"]["path"], 1)
        self.assertEqual(counts["why"]["capped"], 1)
        self.assertEqual(counts["why"]["start not reached"], 1)
        self.assertEqual((counts["first"], counts["second"]), (3, 2))
        ids = sorted(r["id"] for r in recs)
        self.assertEqual(ids, ["B:d4|f48", "B:p1|f48", "B:p1|f49", "B:w1|f48", "B:w1|f49"])
        by = {r["id"]: r for r in recs}
        self.assertEqual((by["B:p1|f48"]["boss_no"], by["B:p1|f49"]["boss_no"]), (1, 2))
        self.assertEqual(by["B:p1|f48"]["cell"], "act3-boss1-78")
        self.assertEqual(by["B:p1|f49"]["cell"], "act3-boss2-80")
        for r in recs:
            self.assertEqual((r["cat"], r["kind"], r["stratum"], r["source"], r["block"], r["engine"], r["run"]), ("act3-boss", "boss", "own", "arrival", "B", "old", r["seed"]))
            self.assertIn(r["split"], ("train", "val"))
        # the champion's own result at the fight: past the first boss for p1 (it went on to the second); lost at the first for d4; won the second for w1
        self.assertTrue(by["B:p1|f48"]["arm_run_passed"])
        self.assertFalse(by["B:d4|f48"]["arm_run_passed"])
        self.assertFalse(by["B:p1|f49"]["arm_run_passed"])
        self.assertTrue(by["B:w1|f49"]["arm_run_passed"])

    def test_the_hash_is_the_starts_own(self):
        recs = [{"id": f"B:{s}|f48", "doc": {"seed": s, "hp": i}, "k": 1} for i, s in enumerate("abc")]
        h = a3.ruler_hash(recs)
        self.assertEqual(len(h), 64)
        self.assertEqual(a3.ruler_hash(list(reversed(recs))), h)  # the order they were written in
        self.assertEqual(a3.ruler_hash([{**r, "k": 99, "note": "x"} for r in recs]), h)  # annotations on a start
        self.assertNotEqual(a3.ruler_hash([{**recs[0], "doc": {"seed": "a", "hp": 99}}, *recs[1:]]), h)  # another position
        self.assertNotEqual(a3.ruler_hash(recs[:2]), h)  # another set
        self.assertNotEqual(a3.ruler_hash([{**recs[0], "id": "B:z|f48"}, *recs[1:]]), h)

    def test_a_written_ruler_loads_only_as_written_and_lists_its_runs_for_exclusion(self):
        rows = {r["seed"]: r for r in [nodes_row("p1", "past", 1), nodes_row("d1", "died", 2)]}
        recs, counts = a3.arrival_records([replay_row(rows["p1"], "B"), replay_row(rows["d1"], "B")], "B", rows, "old")
        arm = {"label": "B", "records": recs, "counts": counts, "rows": 10, "arrivals": 2}
        with tempfile.TemporaryDirectory() as d:
            meta = a3.build_ruler(d, "gen7-cand", "gen7/cand", [arm], "old", "a3-v1")
            self.assertEqual((meta["starts"], meta["first"], meta["second"], meta["arrivals"], meta["matched"]), (3, 2, 1, 2, 2))
            self.assertEqual(meta["ruler_hash"], a3.ruler_hash(recs))
            self.assertEqual(meta["by_cell"], {"act3-boss1-78": 2, "act3-boss2-80": 1})
            loaded, lite = a3.load_ruler(d)
            self.assertEqual(loaded["ruler_hash"], meta["ruler_hash"])
            self.assertEqual(sorted(lite), sorted(r["id"] for r in recs))
            self.assertNotIn("doc", next(iter(lite.values())))
            ex = json.loads((Path(d) / "exclude_runs.json").read_text(encoding="utf-8"))
            self.assertEqual(ex["run_seeds"], ["d1", "p1"])  # every run behind a start, for the practice's exclusion list
            self.assertEqual((ex["starts"], len(ex["start_ids"])), (3, 3))
            # the starts' file as the gym reads it: through its byte index
            self.assertEqual(gl.load_start(str(Path(d) / f"{a3.ARRIVAL_STEM}.jsonl"), "B:p1|f49")["doc"], {"seed": "p1", "floor": 49, "hp": 62})
            text = (Path(d) / f"{a3.ARRIVAL_STEM}.jsonl").read_text(encoding="utf-8")
            (Path(d) / f"{a3.ARRIVAL_STEM}.jsonl").write_text(text.replace('"floor":48,"hp":61}', '"floor":48,"hp":99}', 1), encoding="utf-8")
            with self.assertRaises(ValueError):
                a3.load_ruler(d)  # an edited ruler is not the ruler the hash names
        with self.assertRaises(ValueError):
            a3.build_ruler(tempfile.mkdtemp(), "x", "x", [arm, arm], "old", "a3-v1")  # two arms made the same starts


class BuildDriverTests(unittest.TestCase):
    """`arrivals` end to end: arms on disk, the replay pool replaced by one that writes the rows a matching (or not) replay would."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.agents = self.root / "agent"
        self.champ = self.agents / "gen7" / "cand"
        self.champ.mkdir(parents=True)
        (self.champ / "promotion.json").write_text(json.dumps({"seed_block": "E7-2026-11-01"}), encoding="utf-8")
        self.rows = {"E7-2026-11-01": [nodes_row("sel1", "died", 1)],
                     "E8-2026-11-02": [nodes_row(f"a{i}", k, i) for i, k in enumerate(["died", "past", "early", "died", "won"])],
                     "E8-2026-11-02-r1": [nodes_row(f"b{i}", k, i) for i, k in enumerate(["past", "died", "died"])]}
        for block, rows in self.rows.items():
            arm = self.champ / "arms" / f"{block}-a3-v1"
            arm.mkdir(parents=True)
            (arm / "summary.json").write_text(json.dumps({"config": "a3-v1", "agent": {"dir": "gen7/cand"}}), encoding="utf-8")
            (arm / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.bad = {"a3"}  # the seeds whose replay does not match its row
        self.calls: list[tuple] = []
        self.blocked = None
        for p in (mock.patch.object(fg, "run_tasks", self.fake_pool), mock.patch.object(fe, "blocked_reason", lambda **kw: self.blocked),
                  mock.patch.object(fe, "prepare_process", lambda: None)):
            p.start()
            self.addCleanup(p.stop)

    def fake_pool(self, tasks, fn, target, procs, *, init=None, init_args=(), **kw):
        self.calls.append((len(tasks), Path(target).name, init_args[0], init_args[2], procs, fg.MAX_PROCS))
        with open(target, "a", encoding="utf-8", newline="\n") as fh:
            for t in tasks:
                label = Path(target).stem.removeprefix("replays-")
                fh.write(json.dumps(replay_row(t[5], label, same=t[1] not in self.bad)) + "\n")
        return None

    def run_main(self, *extra):
        out = self.root / "ruler"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = a3.main(["arrivals", "--champion", str(self.champ), "--out", str(out), "--agent-root", str(self.agents), "--procs", "3", *extra])
        return rc, buf.getvalue(), out

    def test_the_ruler_is_built_from_the_blocks_it_was_not_selected_on(self):
        rc, text, out = self.run_main()
        self.assertEqual(rc, 0, text)
        meta, lite = a3.load_ruler(out)
        # arrivals: block E8: a0 died, a1 past, a3 died, a4 won (a2 never arrived) = 4; block E8-r1: b0 past, b1 died, b2 died = 3; the selection block is left out
        self.assertEqual(meta["arrivals"], 7)
        self.assertEqual([x["block"] for x in meta["arms"]], ["E8-2026-11-02", "E8-2026-11-02-r1"])
        self.assertEqual([(x["rows"], x["arrivals"], x["replays"], x["matched"]) for x in meta["arms"]], [(5, 4, 4, 3), (3, 3, 3, 3)])
        self.assertEqual(meta["matched"], 6)  # a3's replay did not match its row: it is not in the ruler
        self.assertNotIn("E8-2026-11-02:a3|f48", lite)
        self.assertEqual((meta["first"], meta["second"]), (6, 3))
        self.assertEqual({s["boss_no"] for s in lite.values()}, {1, 2})
        self.assertFalse([s for s in lite.values() if s["seed"].startswith("sel")])
        self.assertEqual(meta["engine"], "old")  # the arms name no engine: the judge's arms are the old engine's
        self.assertEqual(self.calls[0][2:4], ("old", "gen7/cand"))  # replayed on the engine they were played on, with the champion's own agent
        self.assertGreaterEqual(self.calls[0][5], 3)  # the pool is allowed the processes asked for
        ex = json.loads((out / "exclude_runs.json").read_text(encoding="utf-8"))
        self.assertEqual(ex["run_seeds"], sorted({"a0", "a1", "a4", "b0", "b1", "b2"}))
        self.assertIn("match their row", meta["note"])
        # a second run replays nothing
        n = len(self.calls)
        rc2, _t, _o = self.run_main()
        self.assertEqual((rc2, len(self.calls)), (0, n))

    def test_a_check_replays_a_few_rows_and_writes_no_ruler(self):
        rc, text, out = self.run_main("--limit", "3")
        self.assertEqual(rc, 0, text)
        self.assertFalse((out / "meta.json").exists())
        v = json.loads((out / "verify.json").read_text(encoding="utf-8"))
        self.assertEqual((v["mode"], v["limit"], v["engine"]), ("check", 3, "old"))
        self.assertEqual(sum(b["tried"] for b in v["blocks"].values()), 3)
        self.assertEqual(sorted(v["blocks"]), ["E8-2026-11-02", "E8-2026-11-02-r1"])
        for b in v["blocks"].values():
            self.assertEqual(b["replays"], b["tried"])
            self.assertEqual(b["matched"] + b["mismatched"], b["replays"])

    def test_arms_another_agent_played_are_refused(self):
        (self.champ / "arms" / "E8-2026-11-02-a3-v1" / "summary.json").write_text(json.dumps({"config": "a3-v1", "agent": {"dir": "gen9/other"}}), encoding="utf-8")
        rc, text, out = self.run_main()
        self.assertEqual(rc, 2)
        self.assertIn("were not played by gen7/cand", text)
        self.assertFalse(self.calls)

    def test_a_stop_keeps_the_rows_and_writes_no_ruler(self):
        self.blocked = "night-window"
        rc, text, out = self.run_main()
        self.assertEqual(rc, 3)
        self.assertIn("not started: night-window", text)
        self.assertFalse((out / "meta.json").exists())


def player(root: Path, name: str, weights: bytes) -> Path:
    d = root / name
    d.mkdir()
    (d / "combat_policy.pt").write_bytes(weights)
    (d / "model.json").write_text("{}", encoding="utf-8")
    return d


class ExamDriverTests(unittest.TestCase):
    """`arrival-exam` end to end on a written ruler, with the lockstep replaced by a function that writes the rows a player's games would."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rows = {r["seed"]: r for r in [nodes_row(f"p{i}", "past", i) for i in range(4)] + [nodes_row(f"d{i}", "died", 10 + i) for i in range(2)]}
        reps = [replay_row(r, "B") for r in rows.values()]
        recs, counts = a3.arrival_records(reps, "B", rows, "old")
        self.recs = recs
        self.ruler = self.root / "ruler"
        self.meta = a3.build_ruler(self.ruler, "gen7-cand", "gen7/cand", [{"label": "B", "records": recs, "counts": counts, "rows": 6, "arrivals": 6}], "old", "a3-v1")
        self.cand = player(self.root, "candidate", b"candidate-net")
        self.champ = player(self.root, "champion", b"champion-net")
        self.asked: list[dict] = []
        self.stop_after: str | None = None
        self.wins = {"champion": lambda sid, j: sid.endswith("|f48") and sid.startswith("B:d"), "candidate": lambda sid, j: sid.startswith("B:p0|") or sid.startswith("B:d")}
        for p in (mock.patch.object(fe, "play_exam_tasks", self.fake_lockstep), mock.patch.object(fe, "blocked_reason", lambda **kw: None),
                  mock.patch.object(fe, "prepare_process", lambda: None), mock.patch.object(gl, "engine_tag", lambda: "E2")):
            p.start()
            self.addCleanup(p.stop)

    def fake_lockstep(self, exam_dir, lock, tasks, target, player_dir, what, *, procs, k, prior="server", judge=None):
        side = "champion" if Path(player_dir).name == "champion" else "candidate"
        self.asked.append({"side": side, "what": what, "n": len(tasks), "js": sorted({t[2] for t in tasks}), "starts": len({t[1] for t in tasks}),
                           "ns": {t[4]["ns"] for t in tasks}, "config": {t[4]["config"] for t in tasks}, "procs": procs, "k": k, "prior": prior})
        with open(target, "a", encoding="utf-8", newline="\n") as fh:
            for _a, sid, j, _cat, _info in tasks:
                fh.write(json.dumps(game(sid, j, self.wins[side](sid, j))) + "\n")
        return self.stop_after if len(self.asked) == 1 and self.stop_after else None

    def run_main(self, *extra, champion=None):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = a3.main(["arrival-exam", "--ruler", str(self.ruler), "--candidate", str(self.cand), "--champion", str(champion or self.champ),
                          "--procs", "5", "--report-dir", str(self.root / "report"), *extra])
        return rc, buf.getvalue()

    def test_the_champion_plays_first_every_draw_of_the_test_and_the_reading_is_written(self):
        rc, text = self.run_main("--test", "1")
        self.assertEqual(rc, 0, text)
        self.assertEqual([x["side"] for x in self.asked], ["champion", "candidate"])  # the champion's games are the reference
        for x in self.asked:
            self.assertEqual((x["n"], x["js"], x["starts"]), (6 * 10, [6, 7, 8, 9, 10, 11], 10))  # 10 starts (6 first-boss, 4 second-boss), draws 6..11
            self.assertEqual((x["ns"], x["config"], x["prior"], x["procs"], x["k"]), ({"ar"}, {"a3-v1"}, "cpu", 5, 32))
        self.assertTrue(all(x["what"].startswith("arr-") for x in self.asked))
        res = json.loads((self.root / "report" / "results.json").read_text(encoding="utf-8"))
        r = res["first_boss"]
        self.assertEqual((r["starts"], r["pairs"], r["complete"], r["draws"]), (6, 36, True, [6, 7, 8, 9, 10, 11]))
        # first-boss starts: p0..p3 and d0, d1 at floor 48; the champion wins the d starts' first boss and nothing else, the candidate wins p0's and the d starts'
        means = {"B:p0|f48": 1.0, "B:p1|f48": 0.0, "B:p2|f48": 0.0, "B:p3|f48": 0.0, "B:d0|f48": 0.0, "B:d1|f48": 0.0}
        m, se = sd_mean(list(means.values()))
        self.assertAlmostEqual(r["diff"], m)
        self.assertAlmostEqual(r["se"], se)
        self.assertEqual(res["ruler_hash"], self.meta["ruler_hash"])
        self.assertEqual(res["candidate"]["policy_sha256"], a3.policy_sha(self.cand))
        self.assertNotEqual(res["candidate"]["policy_sha256"], res["champion"]["policy_sha256"])
        # second-boss starts (p0..p3 at floor 49) are read on their own, for the record: the candidate wins p0's
        s = res["second_boss_record"]
        self.assertEqual(s["starts"], 4)
        self.assertAlmostEqual(s["diff"], 0.25)
        self.assertEqual(r["by_cell"]["act3-boss1-78"]["starts"], 6)
        report = (self.root / "report" / "report.txt").read_text(encoding="utf-8")
        self.assertIn(self.meta["ruler_hash"], report)
        self.assertIn("second Act 3 boss (for the record)", report)
        self.assertIn(PM, report)
        self.assertIn("draws 6 to 11", report)
        self.assertEqual(text.strip(), report.strip())

    def test_a_second_run_plays_nothing_and_a_new_test_plays_its_own_draws(self):
        self.assertEqual(self.run_main("--test", "0")[0], 0)
        n = len(self.asked)
        self.assertEqual(self.run_main("--test", "0")[0], 0)
        self.assertEqual(len(self.asked), n)  # every game has its row
        self.assertEqual(self.run_main("--test", "2")[0], 0)
        self.assertEqual([x["js"] for x in self.asked[n:]], [list(range(12, 18))] * 2)  # test 2: draws 12 to 17, for both players

    def test_a_stop_leaves_an_incomplete_reading_and_a_resume_finishes_it(self):
        self.stop_after = "night-window"
        rc, text = self.run_main()
        self.assertEqual(rc, 3)
        self.assertEqual(len(self.asked), 1)  # the champion's games, then the stop: the candidate has none
        self.assertIn("INCOMPLETE", text)
        self.assertIn("stopped: night-window", text)
        self.stop_after = None
        rc, text = self.run_main()
        self.assertEqual(rc, 0, text)
        self.assertEqual([x["side"] for x in self.asked], ["champion", "candidate"])  # the champion's games were kept: only the candidate's were played
        self.assertNotIn("INCOMPLETE", text)

    def test_the_same_network_twice_is_refused_and_an_edited_ruler_does_not_load(self):
        twin = player(self.root, "twin", b"candidate-net")
        rc, text = self.run_main(champion=twin)
        self.assertEqual(rc, 2)
        self.assertIn("same fight network", text)
        self.assertFalse(self.asked)
        path = self.ruler / f"{a3.ARRIVAL_STEM}.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace('"floor":48,"hp":61}', '"floor":48,"hp":99}', 1), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_main()

    def test_a_stop_before_the_start_plays_nothing(self):
        with mock.patch.object(fe, "blocked_reason", lambda **kw: "pause"):
            rc, text = self.run_main()
        self.assertEqual(rc, 3)
        self.assertIn("not started: pause", text)
        self.assertFalse(self.asked)


if __name__ == "__main__":
    unittest.main()
