"""The practice mix (rl/practice_mix.py, rl/gym_loop.py `sample_tasks`; memo 2026-11-04 section 3.2 and 5).

The pinned facts: a kind's draws split between human runs and the AI side by the configured share; the kind weights are untouched;
a start's regret score is the mean over its latest plays of how far the outcome fell short of the teacher's root value at the
opening; the probability of a start is the rank weight of that score mixed with a staleness weight (both terms, so starts not drawn
lately come up and starts that surprised the teacher come up more); the exam's exclusion file keeps starts and runs out whatever
its shape and is no exclusion when absent; the segment log says what was drawn by kind and source and the regret quartiles;
a lost fight leaves the position three decisions before its end as a start and the segment merges it into the set.
No game is played here except in the last class, which plays short arena fights on the copy.
"""

from __future__ import annotations

import argparse
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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fight_gym as fg  # noqa: E402
import gym_loop as gl  # noqa: E402
import practice_mix as pm  # noqa: E402


def _starts(prefix: str, cat: str, n: int, **extra) -> list[dict]:
    return [{"id": f"{prefix}:R{i:03d}|f10", "cat": cat, "seed": f"R{i:03d}", **extra} for i in range(n)]


def _pool(items: list[dict], source: str) -> list[dict]:
    return [{"id": s["id"], "source": source, "file": f"{source}.jsonl", "score0": s.get("score0")} for s in items]


def _cfg(**over) -> dict:
    cfg = {k: dict(v) for k, v in pm.DEFAULTS.items()}
    for k, v in over.items():
        cfg[k].update(v)
    return cfg


class ShareTests(unittest.TestCase):
    def test_the_defaults_are_the_briefs_shares_per_act_and_kind(self):
        hs = pm.DEFAULTS["human_share"]
        for k in ("act2-hallway", "act2-elite", "act2-boss", "act1-boss"):
            self.assertEqual(hs[k], 0.5, k)
        for k in ("act1-hallway", "act1-elite"):
            self.assertEqual(hs[k], 0.25, k)
        self.assertEqual(pm.DEFAULTS["preloss"]["back"], 3)

    def _pools(self):
        pools = {}
        for cat in ("act1-hallway", "act2-boss"):
            pools[(cat, "ai")] = _pool(_starts("n", cat, 120), "night")
            pools[(cat, "human")] = _pool(_starts("human", cat, 120), "human")
        return pools

    def test_a_kinds_draws_split_by_its_share_and_the_kind_weights_stand(self):
        pools = self._pools()
        weights = {"act1-hallway": 3.0, "act2-boss": 1.0}
        s = pm.Sampler(pools, weights, _cfg(), {}, {}, 5)
        rnd = random.Random(1)
        n = 8000
        got = Counter((d["cat"], d["source"]) for d in (s.draw(rnd) for _ in range(n)))
        kind = Counter()
        for (c, _), v in got.items():
            kind[c] += v
        self.assertAlmostEqual(kind["act1-hallway"] / n, 0.75, delta=0.03)  # the weights, unchanged
        self.assertAlmostEqual(got[("act2-boss", "human")] / kind["act2-boss"], 0.5, delta=0.05)
        self.assertAlmostEqual(got[("act1-hallway", "human")] / kind["act1-hallway"], 0.25, delta=0.04)
        self.assertEqual({k[1] for k in got}, {"night", "human"})

    def test_the_share_is_configurable_and_a_kind_with_one_side_draws_from_it(self):
        pools = self._pools()
        del pools[("act2-boss", "human")]
        s = pm.Sampler(pools, {"act2-boss": 1.0}, _cfg(), {}, {}, 5)
        self.assertEqual({s.draw(random.Random(i))["source"] for i in range(50)}, {"night"})
        pools = self._pools()
        del pools[("act2-boss", "ai")]
        s = pm.Sampler(pools, {"act2-boss": 1.0}, _cfg(), {}, {}, 5)
        self.assertEqual({s.draw(random.Random(i))["source"] for i in range(50)}, {"human"})
        s = pm.Sampler(self._pools(), {"act2-boss": 1.0}, _cfg(human_share={"act2-boss": 1.0}), {}, {}, 5)
        self.assertEqual({s.draw(random.Random(i))["source"] for i in range(50)}, {"human"})
        s = pm.Sampler(self._pools(), {"act2-boss": 1.0}, _cfg(human_share={"act2-boss": 0.0}), {}, {}, 5)
        self.assertEqual({s.draw(random.Random(i))["source"] for i in range(50)}, {"night"})

    def test_a_kind_without_a_weight_or_starts_is_never_drawn_and_no_starts_at_all_is_an_error(self):
        pools = self._pools()
        s = pm.Sampler(pools, {"act1-hallway": 1.0, "act2-boss": 0.0, "act1-boss": 5.0}, _cfg(), {}, {}, 5)
        self.assertEqual({s.draw(random.Random(i))["cat"] for i in range(80)}, {"act1-hallway"})
        with self.assertRaises(RuntimeError):
            pm.Sampler({}, {"act1-boss": 1.0}, _cfg(), {}, {}, 5)

    def test_a_config_file_changes_the_shares_and_leaves_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(pm.config(Path(tmp)), pm.DEFAULTS)
            (Path(tmp) / "mix.json").write_text(json.dumps({"human_share": {"act2-boss": 0.9}, "regret": {"temperature": 2.0}}))
            cfg = pm.config(Path(tmp))
            self.assertEqual(cfg["human_share"]["act2-boss"], 0.9)
            self.assertEqual(cfg["human_share"]["act1-boss"], 0.5)
            self.assertEqual(cfg["regret"]["temperature"], 2.0)
            self.assertEqual(cfg["regret"]["staleness"], 0.1)


class RegretTests(unittest.TestCase):
    def test_the_root_value_is_the_visit_weighted_mean_and_a_forced_move_has_none(self):
        self.assertAlmostEqual(pm.root_value([3, 1, 0, 0], [10.0, 50.0, 999.0, float("nan")]), 20.0)
        self.assertIsNone(pm.root_value([0, 0], [float("nan"), 5.0]))
        self.assertAlmostEqual(pm.root_value([2, 0], [float("nan"), 5.0]) or 0.0, 0.0)  # a visited move with no value counts for nothing

    def test_a_death_is_minus_100_and_a_win_is_end_hp_plus_ten_a_potion(self):
        self.assertEqual(pm.played_value({"won": False, "capped": False}), -100.0)
        self.assertEqual(pm.played_value({"won": True, "end_hp": 30, "potions_held": 2}), 50.0)
        self.assertEqual(pm.played_value({"won": True, "end_hp": 30, "start_potions": 2, "potions_used": 1}), 40.0)
        self.assertIsNone(pm.played_value({"won": True, "end_hp": 30, "capped": True}))
        self.assertIsNone(pm.played_value({"won": False, "value_ok": False}))

    def test_a_score_is_the_mean_shortfall_over_the_latest_plays(self):
        plays = [{"start": "A", "root": 50.0, "played": -100.0},  # fell short by 150
                 {"start": "A", "root": 50.0, "played": 80.0},  # beat the root: no shortfall
                 {"start": "A", "root": 50.0, "played": 20.0},  # 30
                 {"start": "B", "root": -20.0, "played": 40.0},
                 {"start": "C", "root": None, "played": 10.0},
                 {"start": "C", "root": 5.0, "played": None}]
        sc = pm.scores(plays, n_last=4)
        self.assertAlmostEqual(sc["A"], (150 + 0 + 30) / 3)
        self.assertEqual(sc["B"], 0.0)
        self.assertNotIn("C", sc)  # nothing to compare
        self.assertAlmostEqual(pm.scores(plays, n_last=2)["A"], (0 + 30) / 2)  # the latest two only
        self.assertAlmostEqual(pm.scores(plays, n_last=4, mode="absolute")["A"], (150 + 30 + 30) / 3)

    def test_ranks_order_by_score_and_ties_share_their_mean_rank(self):
        w = pm.rank_weights([5.0, 9.0, 1.0], temperature=1.0)
        self.assertEqual(w, [1 / 2, 1.0, 1 / 3])
        w = pm.rank_weights([3.0, 3.0, 3.0, 3.0], temperature=1.0)
        self.assertEqual(len(set(w)), 1)  # a pool of equal scores is uniform
        w = pm.rank_weights([2.0, 1.0], temperature=2.0)
        self.assertAlmostEqual(w[0], 1.0)
        self.assertAlmostEqual(w[1], 2 ** -0.5)  # a higher temperature flattens it

    def test_the_probability_follows_the_score_and_a_start_not_drawn_lately_gains(self):
        items = _pool(_starts("n", "act2-boss", 4), "night")
        ids = [s["id"] for s in items]
        score = {ids[0]: 100.0, ids[1]: 10.0, ids[2]: 10.0}  # ids[3] unplayed
        seg = 50
        last = {i: seg for i in ids}  # all drawn this segment: staleness equal
        p = pm.pool_weights(items, score, last, seg, {"temperature": 1.0, "staleness": 0.1})
        self.assertAlmostEqual(sum(p), 1.0)
        self.assertGreater(p[0], p[1])
        self.assertGreater(p[0], max(p[1:]))
        # the unplayed start takes the mean score of the played (40), so it ranks between the hard start and the easy ones
        self.assertGreater(p[3], p[1])
        self.assertLess(p[3], p[0])
        last = {ids[0]: seg, ids[1]: seg, ids[2]: seg - 40, ids[3]: seg}  # ids[2] not drawn for 40 segments
        q = pm.pool_weights(items, score, last, seg, {"temperature": 1.0, "staleness": 0.1})
        self.assertGreater(q[2], p[2])  # staleness lifts it
        self.assertGreater(q[2], q[1])  # over its equal-scored neighbour
        self.assertAlmostEqual(sum(q), 1.0)

    def test_either_term_alone_is_not_the_rule_both_are_mixed(self):
        items = _pool(_starts("n", "act2-boss", 5), "night")
        ids = [s["id"] for s in items]
        score = {ids[0]: 90.0, ids[1]: 5.0, ids[2]: 5.0, ids[3]: 5.0, ids[4]: 5.0}
        last = {ids[0]: 10, ids[1]: 10, ids[2]: 10, ids[3]: 0, ids[4]: 10}
        seg = 10
        only_rank = pm.pool_weights(items, score, last, seg, {"temperature": 1.0, "staleness": 0.0})
        only_stale = pm.pool_weights(items, score, last, seg, {"temperature": 1.0, "staleness": 1.0})
        mixed = pm.pool_weights(items, score, last, seg, {"temperature": 1.0, "staleness": 0.1})
        self.assertAlmostEqual(only_stale[3], 1.0, places=6)  # only the start never drawn
        self.assertAlmostEqual(only_rank[1], only_rank[3])  # the ranks of equal scores tie
        self.assertGreater(mixed[3], mixed[1])  # in the mixture the stale one comes up ...
        self.assertGreater(mixed[0], mixed[3])  # ... and the high-regret one still leads
        self.assertAlmostEqual(mixed[3], 0.9 * only_rank[3] + 0.1 * only_stale[3])

    def test_a_pre_loss_start_takes_the_score_of_its_loss_until_it_is_played(self):
        items = _pool(_starts("n", "act2-boss", 3), "night")
        items[2]["score0"] = 160.0
        ids = [s["id"] for s in items]
        p = pm.pool_weights(items, {ids[0]: 10.0, ids[1]: 20.0}, {i: 5 for i in ids}, 5, {"temperature": 1.0, "staleness": 0.0})
        self.assertEqual(max(range(3), key=lambda i: p[i]), 2)
        p = pm.pool_weights(items, {ids[0]: 10.0, ids[1]: 20.0, ids[2]: 0.0}, {i: 5 for i in ids}, 5,
                            {"temperature": 1.0, "staleness": 0.0})
        self.assertEqual(min(range(3), key=lambda i: p[i]), 2)  # once played, the plays decide

    def test_the_draws_follow_the_probabilities(self):
        items = _pool(_starts("n", "act2-boss", 30), "night")
        score = {items[0]["id"]: 200.0}
        for s in items[1:]:
            score[s["id"]] = 1.0
        smp = pm.Sampler({("act2-boss", "ai"): items}, {"act2-boss": 1.0}, _cfg(human_share={"act2-boss": 0.0}), score,
                         {s["id"]: 100 for s in items}, 100)
        rnd = random.Random(3)
        got = Counter(smp.draw(rnd)["id"] for _ in range(6000))
        self.assertEqual(got.most_common(1)[0][0], items[0]["id"])
        self.assertGreater(got[items[0]["id"]], 5 * got[items[5]["id"]])
        self.assertGreater(len(got), 25)  # the others still come up


class ExclusionTests(unittest.TestCase):
    def _root(self, doc=None, raw: str | None = None) -> Path:
        d = Path(tempfile.mkdtemp())
        if doc is not None or raw is not None:
            (d / "exam").mkdir()
            (d / "exam" / "exclude_runs.json").write_text(raw if raw is not None else json.dumps(doc), encoding="utf-8")
        return d

    def test_no_file_and_a_bad_file_exclude_nothing(self):
        self.assertEqual(pm.read_exclusion(self._root()), frozenset())
        self.assertEqual(pm.read_exclusion(self._root(raw="{not json")), frozenset())
        self.assertEqual(pm.read_exclusion(self._root({"count": 338})), frozenset())

    def test_every_shape_the_exam_may_write_is_read(self):
        for doc in (["A", "B"], {"runs": ["A", "B"], "n": 2}, {"runs": {"A": {"x": 1}, "B": {}}}, {"A": True, "B": True},
                    [{"run": "A"}, {"seed": "B"}], {"runs": ["A"], "starts": ["B"]}):
            self.assertEqual(pm.read_exclusion(self._root(doc)), frozenset({"A", "B"}), doc)

    def test_a_start_is_out_by_its_id_its_run_its_seed_or_its_parent(self):
        ex = frozenset({"RUNX", "2026-11-03/g11:RUNY|f17", "HUMANHASH"})
        self.assertTrue(pm.excluded({"id": "2026-11-02/g9:RUNX|f17", "seed": "RUNX"}, ex))  # by the run the id names
        self.assertTrue(pm.excluded({"id": "2026-11-03/g11:RUNY|f17", "seed": "RUNY"}, ex))  # by the start id
        self.assertTrue(pm.excluded({"id": "human:HUMANHASH|f20", "seed": "HUMANHASH", "run": "HUMANHASH"}, ex))
        self.assertTrue(pm.excluded({"id": "preloss:2026-11-02/g9:RUNX|f17|j2|k20", "seed": "OTHER", "parent": "2026-11-02/g9:RUNX|f17"}, ex))
        self.assertTrue(pm.excluded({"id": "n:Z|f02", "seed": "RUNX"}, ex))  # by seed
        self.assertFalse(pm.excluded({"id": "2026-11-02/g9:RUNZ|f17", "seed": "RUNZ"}, ex))
        self.assertFalse(pm.excluded({"id": "x:RUNX|f02", "seed": "RUNX"}, frozenset()))

    def test_run_ids_come_out_of_every_kind_of_id(self):
        self.assertEqual(pm.run_of("2026-10-26/g5:ZX2QP2Y25J|f30"), "ZX2QP2Y25J")
        self.assertEqual(pm.run_of("human:005eb04211428611|f17"), "005eb04211428611")
        self.assertEqual(pm.run_of("preloss:2026-10-26/g5:ZX2QP2Y25J|f30|j2|k25"), "ZX2QP2Y25J")

    def test_pools_leave_out_the_excluded_and_say_how_many(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "starts").mkdir()
            night = _starts("2026-11-01/g9", "act2-boss", 5, ix=0)
            human = [{"id": f"human:H{i}|f33", "cat": "act2-boss", "seed": f"H{i}", "run": f"H{i}", "source": "human"} for i in range(4)]
            pre = [{"id": "preloss:2026-11-01/g9:R001|f10|j0|k9", "cat": "act2-boss", "seed": "R001", "run": "R001", "source": "preloss",
                    "parent": "2026-11-01/g9:R001|f10", "score0": 120.0}]
            (root / "starts" / "starts.lite.json").write_text(json.dumps(night))
            (root / "starts" / "human-starts.lite.json").write_text(json.dumps(human))
            (root / "starts" / "preloss-starts.lite.json").write_text(json.dumps(pre))
            pools, dropped = pm.load_pools(root, frozenset())
            self.assertEqual((len(pools[("act2-boss", "ai")]), len(pools[("act2-boss", "human")]), sum(dropped.values())), (6, 4, 0))
            self.assertEqual({s["source"] for s in pools[("act2-boss", "ai")]}, {"night", "preloss"})
            pools, dropped = pm.load_pools(root, frozenset({"R001", "H2"}))
            ids = {s["id"] for v in pools.values() for s in v}
            self.assertNotIn("human:H2|f33", ids)
            self.assertNotIn("2026-11-01/g9:R001|f10", ids)
            self.assertNotIn("preloss:2026-11-01/g9:R001|f10|j0|k9", ids)  # the pre-loss start of an excluded run goes with it
            self.assertEqual(dict(dropped), {"night": 1, "human": 1, "preloss": 1})
            self.assertEqual(len(ids), 7)


class LogTests(unittest.TestCase):
    def test_the_quartiles(self):
        self.assertIsNone(pm.quartiles([]))
        self.assertEqual(pm.quartiles([1.0]), [1.0, 1.0, 1.0])
        self.assertEqual(pm.quartiles([0, 10, 20, 30, 40]), [10.0, 20.0, 30.0])
        self.assertEqual(pm.quartiles([40, 0, 20, 10, 30, 50]), [12.5, 25.0, 37.5])

    def test_the_summary_and_the_line_name_the_mix_by_kind_and_source_and_the_regret_quartiles(self):
        drawn = ([{"cat": "act2-boss", "source": "human", "score": 40.0}] * 3 + [{"cat": "act2-boss", "source": "night", "score": None}] * 2
                 + [{"cat": "act1-boss", "source": "night", "score": 10.0}, {"cat": "act1-boss", "source": "preloss", "score": 160.0}])
        m = pm.mix_summary(drawn)
        self.assertEqual(m["by_kind"], {"act1-boss": {"night": 1, "preloss": 1}, "act2-boss": {"human": 3, "night": 2}})
        self.assertEqual((m["drawn"], m["scored"]), (7, 5))
        self.assertEqual(m["regret_quartiles"], [40.0, 40.0, 40.0])
        line = pm.mix_line(m)
        self.assertIn("act1-boss night 1 preloss 1", line)
        self.assertIn("act2-boss night 2 human 3", line)
        self.assertIn("regret q1/med/q3 40/40/40 (5 of 7 drawn starts scored)", line)
        self.assertIn("regret q1/med/q3 none", pm.mix_line(pm.mix_summary([{"cat": "act1-boss", "source": "night", "score": None}])))


def _fake_shard(path: Path, fights: list[dict]) -> None:
    """A buffer shard with the arrays `shard_plays` reads: one row per decision, the first row's visits and values set."""
    fight, visits, qvals, k, ok, metas = [], [], [], [], [], []
    for i, f in enumerate(fights):
        for d in range(f.get("rows", 3)):
            fight.append(i)
            k.append(d)
            v = np.zeros(32, np.float32)
            q = np.full(32, np.nan, np.float32)
            if d == 0 and f.get("root") is not None:
                v[:2], q[:2] = [3, 1], [f["root"], f["root"]]
            visits.append(v)
            qvals.append(q)
            ok.append(0 if f.get("capped") else 1)
        metas.append({"start": f["start"], "j": f.get("j", 0), "won": f["won"], "end_hp": f.get("end_hp", 0), "capped": bool(f.get("capped")),
                      "potions_held": f.get("held", 0), "cat": "act2-boss", "split": "train"})
    np.savez_compressed(path, fight=np.array(fight, np.int32), k=np.array(k, np.int16), visits=np.stack(visits), qvals=np.stack(qvals),
                        value_ok=np.array(ok, np.int8), fights_meta=np.array(json.dumps(metas)))


class PlaysTests(unittest.TestCase):
    def test_a_shard_gives_each_fights_root_value_and_outcome_and_the_log_reads_each_shard_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "starts").mkdir()
            sh = root / "buffer" / "shards"
            sh.mkdir(parents=True)
            _fake_shard(sh / "s000001-1.npz", [{"start": "A", "root": 60.0, "won": False}, {"start": "B", "root": 10.0, "won": True, "end_hp": 30, "held": 1},
                                               {"start": "C", "root": 5.0, "won": True, "end_hp": 9, "capped": True},
                                               {"start": "D", "root": None, "won": False}])
            plays = pm.sync_plays(root)
            by = {p["start"]: p for p in plays}
            self.assertEqual((by["A"]["root"], by["A"]["played"]), (60.0, -100.0))
            self.assertEqual((by["B"]["root"], by["B"]["played"]), (10.0, 40.0))
            self.assertIsNone(by["C"]["played"])  # a capped fight has no outcome
            self.assertIsNone(by["D"]["root"])  # nothing was searched
            self.assertEqual(len(plays), 4)
            self.assertEqual(len(pm.sync_plays(root)), 4)  # the second call adds nothing
            _fake_shard(sh / "s000002-2.npz", [{"start": "A", "j": 1, "root": 55.0, "won": True, "end_hp": 20}])
            plays = pm.sync_plays(root)
            self.assertEqual(len(plays), 5)
            sc = pm.scores(plays)
            self.assertAlmostEqual(sc["A"], (160 + 35) / 2)
            self.assertEqual(sc["B"], 0.0)
            (sh / "s000001-1.npz").unlink()  # retention removes a shard: its plays stay in the log
            self.assertEqual(len(pm.sync_plays(root)), 5)


class SampleTasksTests(unittest.TestCase):
    """`gym_loop.sample_tasks` on a loop folder built by hand: no game is played."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "loop"
        self.loop = gl.Loop(self.root)
        sd = self.loop.starts
        night = (_starts("2026-11-01/g9", "act2-boss", 60) + _starts("2026-11-01/g9", "act1-hallway", 60))
        self.human = [{"id": f"human:H{i:03d}|f33", "cat": "act2-boss", "seed": f"H{i:03d}", "run": f"H{i:03d}", "source": "human"}
                      for i in range(60)]
        (sd / "starts.lite.json").write_text(json.dumps(night))
        (sd / "human-starts.lite.json").write_text(json.dumps(self.human))
        (sd / "weights.json").write_text(json.dumps({"weights": {"act2-boss": 1.0, "act1-hallway": 1.0}}))
        self.loop.state["segment"] = 7

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_tasks_come_from_both_sources_with_their_own_files_and_the_log_gets_the_mix(self):
        tasks = gl.sample_tasks(self.loop, "c0001", 1500.0, "x16", 0.25, random.Random(5))
        self.assertGreater(len(tasks), 10)
        files = {t[4]["starts"] for t in tasks}
        self.assertEqual({Path(f).name for f in files}, {"starts.jsonl", "human-starts.jsonl"})
        for arm, sid, j, cat, info in tasks:
            self.assertEqual(arm, "x16")
            self.assertEqual(sid.startswith("human:"), Path(info["starts"]).name == "human-starts.jsonl")
            self.assertTrue(info["preloss"]["dir"].endswith("preloss-new"))
            self.assertEqual(info["player"], "c0001")
        by = self.loop.mix["by_kind"]
        self.assertEqual(sum(sum(v.values()) for v in by.values()), len(tasks))
        self.assertGreater(by["act2-boss"].get("human", 0), 0)
        self.assertNotIn("human", by["act1-hallway"])  # no human starts of that kind: all from the nights
        line = pm.mix_line(self.loop.mix)
        self.assertRegex(line, r"act2-boss night \d+ human \d+")
        self.assertIn("regret q1/med/q3", line)

    def test_the_games_numbers_and_the_drawn_segment_are_recorded(self):
        tasks = gl.sample_tasks(self.loop, "c0001", 900.0, "x16", 0.25, random.Random(6))
        nj = self.loop.state["next_j"]
        self.assertEqual(sum(nj.values()), len(tasks))
        for _, sid, j, _, _ in tasks:
            self.assertLess(j, nj[sid])
            self.assertEqual(self.loop.state["last_seg"][sid], 7)
        again = gl.sample_tasks(self.loop, "c0001", 900.0, "x16", 0.25, random.Random(6))
        seen = {t[1] for t in tasks}
        self.assertTrue(all(t[2] >= 1 for t in again if t[1] in seen))  # a start drawn before plays its next game

    def test_the_exams_file_keeps_starts_and_runs_out_and_a_missing_file_is_fine(self):
        gl.sample_tasks(self.loop, "c0001", 900.0, "x16", 0.25, random.Random(1))  # no exam folder: fine
        (self.root / "exam").mkdir()
        runs = [f"H{i:03d}" for i in range(0, 60, 2)] + [f"R{i:03d}" for i in range(0, 60, 3)]
        (self.root / "exam" / "exclude_runs.json").write_text(json.dumps({"runs": runs}))
        tasks = gl.sample_tasks(self.loop, "c0001", 3000.0, "x16", 0.25, random.Random(2))
        for _, sid, _, _, _ in tasks:
            self.assertNotIn(pm.run_of(sid), runs, sid)
        self.assertEqual(self.loop.mix["left_out"]["human"], 30)
        self.assertEqual(self.loop.mix["excluded_ids"], len(runs))

    def test_a_start_with_regret_is_drawn_more_and_the_play_log_feeds_it(self):
        sh = self.loop.shards
        hard = self.human[7]["id"]
        _fake_shard(sh / "s000001-1.npz", [{"start": hard, "root": 90.0, "won": False}] +
                    [{"start": h["id"], "root": 20.0, "won": True, "end_hp": 60} for h in self.human[8:30]])
        got = Counter()
        for seed in range(30):
            for t in gl.sample_tasks(self.loop, "c0001", 1200.0, "x16", 0.25, random.Random(seed)):
                got[t[1]] += 1
        self.assertEqual(got.most_common(1)[0][0], hard)
        easy = [got[h["id"]] for h in self.human[8:30]]
        self.assertGreater(got[hard], 3 * (sum(easy) / len(easy)))
        self.assertEqual(self.loop.mix["scored_starts"], 23)

    def test_no_preloss_when_it_is_off(self):
        tasks = gl.sample_tasks(self.loop, "c0001", 900.0, "x16", 0.25, random.Random(1), preloss=False)
        self.assertTrue(all("preloss" not in t[4] for t in tasks))

    def test_the_segment_puts_the_mix_and_the_regret_quartiles_in_the_log_and_in_segments_jsonl(self):
        _fake_shard(self.loop.shards / "s000001-1.npz", [{"start": self.human[3]["id"], "root": 80.0, "won": False}])
        self.loop.save()
        a = argparse.Namespace(root=str(self.root), procs=1, segment_min=12.0, teacher="x16", exact_wall_s=0.25, k=2, prior="cpu",
                               buffer_rows=10**6, buffer_days=21.0, buffer_gb=20.0, no_preloss=False)
        d = gl.Daemon(a)
        with mock.patch.object(gl, "play_tasks", lambda *args, **kw: None), mock.patch.object(gl.Daemon, "status", lambda *args, **kw: None):
            d.segment()
        log = (self.root / "loop.log").read_text(encoding="utf-8")
        line = next(x for x in log.splitlines() if "segment 8:" in x)
        self.assertRegex(line, r"fights by .* under x16; mix act1-hallway night \d+; act2-boss night \d+ human \d+; regret q1/med/q3 ")
        row = json.loads((self.root / "games" / "segments.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(row["segment"], 8)
        self.assertEqual(set(row["mix"]), {"act1-hallway", "act2-boss"})
        self.assertEqual(sum(sum(v.values()) for v in row["mix"].values()), row["tasks"])
        self.assertIn("regret_quartiles", row)
        self.assertEqual(row["preloss"], {})


class PrelossTests(unittest.TestCase):
    """A lost fight leaves the position `back` decisions before its end; the segment merges it into the set."""

    class Rec:
        def __init__(self, n: int, back: int = 3, bad: int | None = None):
            import collections

            self.rows = list(range(n))
            self.states = collections.deque(maxlen=back + 1)
            for k in range(n):
                doc = "refused: ExportRefused" if k == bad else {"schema": "sts2-emu-snapshot/1", "turn": 2, "encounter_id": 83, "k": k,
                                                                  "player": {"hp": 30 - k, "max_hp": 85}, "relics": [{"id": 1}] * 4,
                                                                  "potions": {"max_slots": 2, "slots": [5, 0, 0]},
                                                                  "piles": {"hand": [{}] * 3, "discard": [{}] * 2, "exhaust": [],
                                                                            "draw": {"known_top": [], "known_bottom": [], "unordered": [{}] * 6}}}
                self.states.append((k, doc, 40.0 + k))

    PARENT = {"id": "2026-11-02/g9:CYVX|f17", "cat": "act1-boss", "kind": "boss", "act": 1, "floor": 17, "seed": "CYVX", "ix": 4,
              "node_type": 6, "split": "val"}

    def test_the_position_three_decisions_before_the_end_is_the_start_with_the_losses_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"dir": tmp, "back": 3, "min_decisions": 5}
            word = gl.write_preloss(cfg, self.PARENT, self.PARENT["id"], 2, {"won": False, "capped": False}, self.Rec(12), "c0144")
            self.assertEqual(word, "ok")
            files = list(Path(tmp).glob("*.json"))
            self.assertEqual(len(files), 1)
            rec = json.loads(files[0].read_text())
            self.assertEqual(rec["k"], 8)  # decisions 0..11; the last is 11; three before it is 8
            self.assertEqual(rec["doc"]["k"], 8)
            self.assertEqual(rec["id"], "preloss:2026-11-02/g9:CYVX|f17|j2|k8")
            self.assertEqual((rec["cat"], rec["kind"], rec["floor"], rec["node_type"], rec["split"], rec["seed"]),
                             ("act1-boss", "boss", 17, 6, "val", "CYVX"))
            self.assertEqual((rec["source"], rec["parent"], rec["hp"], rec["max_hp"], rec["deck_size"], rec["potions"], rec["relics"]),
                             ("preloss", self.PARENT["id"], 22, 85, 11, 1, 4))
            self.assertEqual(rec["score0"], 48.0 + 100.0)  # the teacher's root value there (40 + 8) against the death
            self.assertTrue(pm.excluded(rec, frozenset({"CYVX"})))

    def test_nothing_is_written_for_a_win_a_capped_fight_a_short_one_a_refused_export_or_a_chained_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"dir": tmp, "back": 3, "min_decisions": 5}
            P = self.PARENT
            self.assertIsNone(gl.write_preloss(cfg, P, P["id"], 0, {"won": True, "capped": False}, self.Rec(12), "c"))
            self.assertIsNone(gl.write_preloss(cfg, P, P["id"], 0, {"won": False, "capped": True}, self.Rec(12), "c"))
            self.assertEqual(gl.write_preloss(cfg, P, P["id"], 0, {"won": False, "capped": False}, self.Rec(4), "c"), "short")
            self.assertEqual(gl.write_preloss(cfg, P, P["id"], 0, {"won": False, "capped": False}, self.Rec(12, bad=8), "c"), "refused")
            self.assertEqual(gl.write_preloss(cfg, {**P, "source": "preloss"}, P["id"], 0, {"won": False, "capped": False}, self.Rec(12), "c"),
                             "chained")
            self.assertEqual(list(Path(tmp).glob("*.json")), [])

    def test_the_segment_merges_them_into_a_set_the_loader_reads_and_keeps_the_newest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "starts" / "preloss-new").mkdir(parents=True)
            cfg = {"dir": str(root / "starts" / "preloss-new"), "back": 3, "min_decisions": 5}
            for j in range(6):
                gl.write_preloss(cfg, self.PARENT, self.PARENT["id"], j, {"won": False, "capped": False}, self.Rec(12), "c0144")
            r = pm.merge_preloss(root, max_keep=100)
            self.assertEqual((r["merged"], r["kept"], r["dropped"]), (6, 6, 0))
            self.assertEqual(list((root / "starts" / "preloss-new").glob("*.json")), [])
            lite = json.loads((root / "starts" / "preloss-starts.lite.json").read_text())
            self.assertEqual(len(lite), 6)
            self.assertTrue(all("doc" not in x for x in lite))
            gl._START_CACHE.clear()
            got = gl.load_start(str(root / "starts" / "preloss-starts.jsonl"), "preloss:2026-11-02/g9:CYVX|f17|j3|k8")
            self.assertEqual(got["doc"]["k"], 8)
            for j in range(6, 12):  # a set a quarter over its cap is cut to the newest
                gl.write_preloss(cfg, self.PARENT, self.PARENT["id"], j, {"won": False, "capped": False}, self.Rec(12), "c0144")
            r = pm.merge_preloss(root, max_keep=8)
            self.assertEqual((r["merged"], r["kept"], r["dropped"]), (6, 8, 4))
            gl._START_CACHE.clear()
            ids = {x["id"] for x in json.loads((root / "starts" / "preloss-starts.lite.json").read_text())}
            self.assertEqual(ids, {f"preloss:2026-11-02/g9:CYVX|f17|j{j}|k8" for j in range(4, 12)})
            for sid in ids:
                self.assertEqual(gl.load_start(str(root / "starts" / "preloss-starts.jsonl"), sid)["id"], sid)

    # "Newest" is a total order carried in the records (their sequence numbers), not read off the file system: a segment's games
    # write dozens of starts inside one clock tick (about 16 ms on Windows), and file times and names then tie.

    def _inbox(self, root: Path) -> Path:
        d = root / "starts" / "preloss-new"
        d.mkdir(parents=True)
        return d

    def _hand_record(self, parent_id: str, j: int, **over) -> dict:
        parent = {**self.PARENT, "id": parent_id}
        rec = pm.preloss_start(parent, parent_id, j, 8, self.Rec(12).states[0][1], 48.0, "c0144", "2026-11-05T00:00:00Z")
        rec.update(over)
        return rec

    def _lite_ids(self, root: Path) -> list[str]:
        return [x["id"] for x in json.loads((root / "starts" / "preloss-starts.lite.json").read_text())]

    def test_many_starts_written_inside_one_tick_are_kept_newest_exactly(self):
        pid = self.PARENT["id"]
        self.addCleanup(setattr, pm, "_SEQ_LAST", pm._SEQ_LAST)  # the frozen clock must not leave a future number behind
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pm.time, "time_ns", return_value=1_800_000_000_000_000_000):
            root = Path(tmp)
            inbox = self._inbox(root)
            cfg = {"dir": str(inbox), "back": 3, "min_decisions": 5}
            lost = {"won": False, "capped": False}
            for j in range(20):
                gl.write_preloss(cfg, self.PARENT, pid, j, lost, self.Rec(12), "c0144")
            for p in inbox.glob("*.json"):  # one tick: every file the same time too
                os.utime(p, ns=(1_800_000_000_000_000_000, 1_800_000_000_000_000_000))
            self.assertEqual(pm.merge_preloss(root, max_keep=100)["merged"], 20)
            # a second parent's games interleaved with the first's, still inside the same tick
            other = "2026-11-02/g9:ZZZZ|f17"
            for j in range(20, 40):
                gl.write_preloss(cfg, {**self.PARENT, "id": other}, other, j, lost, self.Rec(12), "c0144")
                gl.write_preloss(cfg, self.PARENT, pid, j, lost, self.Rec(12), "c0144")
            for p in inbox.glob("*.json"):
                os.utime(p, ns=(1_800_000_000_000_000_000, 1_800_000_000_000_000_000))
            self.assertEqual(len({p.stat().st_mtime_ns for p in inbox.glob("*.json")}), 1)
            r = pm.merge_preloss(root, max_keep=8)
            self.assertEqual((r["merged"], r["kept"], r["dropped"]), (40, 8, 52))
            # the last eight written: j36..39 of both parents, in write order
            want = [f"preloss:{s}|j{j}|k8" for j in range(36, 40) for s in (other, pid)]
            self.assertEqual(self._lite_ids(root), want)
            gl._START_CACHE.clear()
            for sid in want:
                self.assertEqual(gl.load_start(str(root / "starts" / "preloss-starts.jsonl"), sid)["id"], sid)

    def test_sequence_numbers_rise_inside_a_frozen_clock_and_when_the_clock_steps_back(self):
        self.addCleanup(setattr, pm, "_SEQ_LAST", pm._SEQ_LAST)
        with mock.patch.object(pm.time, "time_ns", return_value=5_000_000_000_000_000_000):
            a, b, c = pm.next_seq(), pm.next_seq(), pm.next_seq()
        self.assertTrue(a < b < c)
        with mock.patch.object(pm.time, "time_ns", return_value=1):
            d = pm.next_seq()
        self.assertGreater(d, c)

    def test_equal_sequence_numbers_from_two_writers_tie_by_parent_then_game_number_whatever_the_file_names(self):
        recs = [self._hand_record(pid, j, seq=777) for pid in ("run:B|f17", "run:A|f17") for j in (10, 2, 3)]
        want = [f"preloss:{pid}|j{j}|k8" for pid in ("run:A|f17", "run:B|f17") for j in (2, 3, 10)]  # j2 before j10: a number
        for names in (lambda i: f"a{i:02d}", lambda i: f"z{len(recs) - i:02d}"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                inbox = self._inbox(root)
                for i, rec in enumerate(recs):
                    (inbox / f"{names(i)}.json").write_text(json.dumps(rec), encoding="utf-8")
                pm.merge_preloss(root, max_keep=100)
                self.assertEqual(self._lite_ids(root), want)

    def test_two_writers_that_tie_on_sequence_parent_and_game_come_out_in_writer_order_whatever_the_file_names(self):
        pid = "run:A|f17"
        # The same game of the same start, played by writers 11 and 5 in one clock tick. The old key tied on (seq, parent, j) and
        # fell to the id text, where "k10" sorts before "k2" whichever writer wrote it; a repeated id fell to the directory order.
        a = self._hand_record(pid, 3, seq=777, writer=11)
        b = self._hand_record(pid, 3, seq=777, writer=5)
        a["id"], b["id"] = f"preloss:{pid}|j3|k10", f"preloss:{pid}|j3|k2"
        twin_hi = self._hand_record(pid, 4, seq=777, writer=11, made="by-11")  # one id, two records
        twin_lo = self._hand_record(pid, 4, seq=777, writer=5, made="by-5")
        for names in (lambda i: f"a{i:02d}", lambda i: f"z{9 - i:02d}"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                inbox = self._inbox(root)
                for i, rec in enumerate((a, b, twin_hi, twin_lo)):
                    (inbox / f"{names(i)}.json").write_text(json.dumps(rec), encoding="utf-8")
                self.assertEqual(pm.merge_preloss(root, max_keep=100)["merged"], 3)
                self.assertEqual(self._lite_ids(root), [b["id"], a["id"], f"preloss:{pid}|j4|k8"])  # writer 5 first
                lite = json.loads((root / "starts" / "preloss-starts.lite.json").read_text())
                self.assertEqual(lite[-1]["made"], "by-5")  # the lower writer's record is the one kept
        # a record from this process carries its pid, and inside one process the rank still follows the sequence alone
        mine = self._hand_record(pid, 1)
        self.assertEqual(mine["writer"], os.getpid())
        later = self._hand_record(pid, 0)
        self.assertLess(pm.preloss_rank(mine), pm.preloss_rank(later))

    def test_a_start_written_before_sequence_numbers_existed_is_ranked_by_its_files_time_then_game_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inbox = self._inbox(root)
            pid = "run:A|f17"
            # (j, file time): j5 is the oldest file; j10 and j2 share a time; j7 is the newest. Names run against the times.
            for name, j, t in (("a", 7, 4_000), ("b", 2, 3_000), ("c", 10, 3_000), ("d", 5, 1_000)):
                rec = self._hand_record(pid, j)
                del rec["seq"], rec["j"]
                p = inbox / f"{name}.json"
                p.write_text(json.dumps(rec), encoding="utf-8")
                os.utime(p, ns=(t, t))
            pm.merge_preloss(root, max_keep=100)
            self.assertEqual(self._lite_ids(root), [f"preloss:{pid}|j{j}|k8" for j in (5, 2, 10, 7)])


# ---------------------------------------------------------------------------------------------------------- real games


def _lost_fight_starts(queue) -> None:
    """Child: two arena fights that a basic deck cannot win (the Kaiser Crab at 3 HP), as gym starts, made on core2."""
    try:
        for p in (ROOT / "rl", ROOT / "src", ROOT):
            sys.path.insert(0, str(p))
        import gen0  # noqa: PLC0415

        gen0.select_engine("core2")
        import ctypes  # noqa: PLC0415

        from sts2_gym import native  # noqa: PLC0415
        from sts2_gym.env import ENCOUNTER_IDS  # noqa: PLC0415

        deck = native.starting_deck(10)
        out = []
        for i, seed in enumerate((3, 5)):
            h = native.create(seed)
            obs = (ctypes.c_int * native.OBS_SIZE)()
            native.reset_arena(h, deck, obs, encounter_id=ENCOUNTER_IDS["kaiser-crab"], player_hp=3, player_max_hp=80, ascension=10)
            doc = native.get_state_json(h)
            native.destroy(h)
            out.append({"id": f"L{i}|f05", "cat": "act1-boss", "kind": "boss", "act": 1, "floor": 5, "node_type": 6, "hp": 3, "max_hp": 80,
                        "deck_size": len(deck), "potions": 0, "seed": f"RUN{i}", "split": "train", "doc": doc})
        queue.put(("ok", out))
    except Exception as error:  # noqa: BLE001 -- the test skips when the native library is not there
        queue.put(("error", f"{type(error).__name__}: {error}"))


class PrelossGameTests(unittest.TestCase):
    """Short arena fights at a3-v1 with the fixture prior: recording the positions changes no move, and a lost fight leaves a start
    the copy loads and plays."""

    @classmethod
    def setUpClass(cls):
        from prior_fixture import export_prior  # noqa: PLC0415 -- torch

        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        export_prior(root / "agents" / "fixture", prior_min_visits=0)
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        p = ctx.Process(target=_lost_fight_starts, args=(queue,), daemon=True)
        p.start()
        kind, got = queue.get(timeout=120)
        p.join(30)
        if kind != "ok":
            raise unittest.SkipTest(f"native library unavailable: {got}")
        cls.loop = gl.Loop(root / "loop")
        gl.write_start_set(cls.loop.starts, got)
        agent = root / "agents" / "fixture"
        cls.init = ("core2", str(agent.parent), agent.name, str(cls.loop.root))
        pre = {"dir": str(cls.loop.starts / "preloss-new"), "back": 2, "min_decisions": 3}
        base = {"starts": str(cls.loop.starts / "starts.jsonl"), "incoming": str(cls.loop.incoming), "player": "fixture",
                "config": "a3-v1", "record": True, "ns": "g", "exact_wall_s": 0.0}
        cls.starts = got
        import gym_lockstep  # noqa: PLC0415

        cls.with_pre, cls.without = cls.loop.games / "with.jsonl", cls.loop.games / "without.jsonl"
        for target, info in ((cls.with_pre, {**base, "preloss": pre}), (cls.without, base)):
            tasks = [("a3-v1", s["id"], j, "act1-boss", info) for s in got for j in range(2)]
            stopped, _ = gym_lockstep.run_lockstep(tasks, target, 1, 2, cls.init, {"prior": "cpu", "game_steps": "gym_loop:gen_game_steps"})
            assert stopped is None

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _by(self, path):
        return {(r["start"], int(r["j"])): r for r in fg.read_jsonl(path)}

    def test_keeping_the_positions_changes_no_move(self):
        a, b = self._by(self.with_pre), self._by(self.without)
        self.assertEqual(sorted(a), sorted(b))
        for key in a:
            for f in ("won", "end_hp", "turns", "decisions", "sims", "moves_sha", "first_move", "L0"):
                self.assertEqual(a[key][f], b[key][f], (key, f))

    def test_a_lost_fight_leaves_a_start_the_copy_loads_and_plays_on(self):
        lost = [r for r in self._by(self.with_pre).values() if not r["won"] and not r["capped"]]
        self.assertTrue(lost, "the 3-HP fights should be lost")
        made = [r for r in lost if r.get("preloss") == "ok"]
        self.assertTrue(made or all(r.get("preloss") in ("short", "refused") for r in lost), [r.get("preloss") for r in lost])
        if not made:
            self.skipTest("every lost fight was too short to leave a position")
        merged = pm.merge_preloss(self.loop.root, 100)
        self.assertEqual(merged["merged"], len(made))
        lite = json.loads((self.loop.starts / "preloss-starts.lite.json").read_text())
        gl._START_CACHE.clear()
        rec = gl.load_start(str(self.loop.starts / "preloss-starts.jsonl"), lite[0]["id"])
        self.assertEqual(rec["source"], "preloss")
        self.assertEqual(rec["split"], "train")
        info = {"starts": str(self.loop.starts / "preloss-starts.jsonl"), "incoming": str(self.loop.incoming), "player": "fixture",
                "config": "a3-v1", "record": True, "ns": "g", "exact_wall_s": 0.0, "preloss": {"dir": str(self.loop.starts / "preloss-new"),
                                                                                               "back": 2, "min_decisions": 3}}
        import gym_lockstep  # noqa: PLC0415

        out = self.loop.games / "again.jsonl"
        stopped, _ = gym_lockstep.run_lockstep([("a3-v1", rec["id"], 0, "act1-boss", info)], out, 1, 1, self.init,
                                               {"prior": "cpu", "game_steps": "gym_loop:gen_game_steps"})
        self.assertIsNone(stopped)
        row = fg.read_jsonl(out)[0]
        self.assertGreaterEqual(row["decisions"], 1)
        self.assertLessEqual(row["decisions"], 8)  # a position a few decisions from the end


if __name__ == "__main__":
    unittest.main()
