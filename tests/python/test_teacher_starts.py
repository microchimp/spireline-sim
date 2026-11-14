"""The run teacher's fights as a fourth source of the practice mix (rl/teacher_starts.py, rl/practice_mix.py; D60 item 5, lane L).

The pinned facts: from a store of three teacher runs the builder makes one start for each Act 1 elite, Act 1 boss, Act 2 elite and Act 2
boss the run reached and none for a hallway, an event fight, an Act 3 fight or a fight past floor 33, with the id, kind, act, floor, run,
split and source (`teacher`) the gym's loader reads; a run whose seed is behind the board's Act-1 bosses or in the exam's exclusion
file is never replayed and never drawn; a run is replayed once (a second build replays only new runs; the set is stale once enough new
runs have finished); the replay takes the stored teacher pick on a screen that has a record and the player's own elsewhere, draws the
player's pick (and so its exploration stream) at every screen, checks every fight start against the run's own record and stops at the
first difference, keeping the starts made before it; the sampler draws the teacher side by its share of a kind (default none: the
draws of today are unchanged to the byte), and the share comes out of the draw before the human share splits the rest. No game is
played here: the replay's game is a fake session; the real replay of ten stored runs is the lane's measurement (FIGHT-LOOP-L.md).
"""

from __future__ import annotations

import argparse
import json
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

import fight_gym as fg  # noqa: E402
import gym_loop as gl  # noqa: E402
import practice_mix as pm  # noqa: E402
import teacher_starts as ts  # noqa: E402

AGENT = "gen15/candidate-x"
SHA = "ab" * 32


def _fight(floor: int, node_type: int, act_index: int, hp: int = 70, max_hp: int = 80, deck: int = 12) -> dict:
    return {"floor": floor, "node_type": node_type, "encounter": 3, "won": True,
            "state": {"act_index": act_index, "floor": floor, "hp": hp, "max_hp": max_hp, "deck": [[1, 0, 0, 0]] * deck}}


# the fights of the three runs: (floor, node type: 1 hallway, 2 elite, 6 boss, 7 an event's fight, act index)
RUNS = {
    "S1AAAAAAAA": [(3, 1, 0), (6, 2, 0), (17, 6, 0)],  # died at the Act 1 boss: an elite and a boss
    "S2BBBBBBBB": [(4, 1, 0), (9, 2, 0), (12, 7, 0), (17, 6, 0), (19, 1, 1), (24, 2, 1), (33, 6, 1), (35, 2, 2)],  # all four kinds, an event fight, an Act 3 fight
    "S3CCCCCCCC": [(2, 1, 0), (5, 2, 0)],  # died at the first elite
}
EXPECTED = {
    "teacher:S1AAAAAAAA|f06": "act1-elite", "teacher:S1AAAAAAAA|f17": "act1-boss",
    "teacher:S2BBBBBBBB|f09": "act1-elite", "teacher:S2BBBBBBBB|f17": "act1-boss", "teacher:S2BBBBBBBB|f24": "act2-elite",
    "teacher:S2BBBBBBBB|f33": "act2-boss", "teacher:S3CCCCCCCC|f05": "act1-elite",
}


def make_store(root: Path, runs: dict | None = None, generation: int = 15, job: str = "JOB1", shard: bool = True) -> Path:
    store = Path(root) / "run-teacher"
    gen = store / str(generation)
    (gen / "decisions.d").mkdir(parents=True, exist_ok=True)
    if shard:
        (gen / "shards" / job).mkdir(parents=True, exist_ok=True)
        (gen / "shards" / job / "manifest.json").write_text(json.dumps(
            {"job": job, "agent": AGENT, "agent_sha256": SHA, "engine": "e1", "generation": generation}), encoding="utf-8")
    rows = []
    for i, (seed, fights) in enumerate((runs or RUNS).items()):
        rows.append({"_task": ["rt-new", seed, 1000000 + i], "arm": "teacher", "source": "new", "job": job, "seed": seed,
                     "ix": 1000000 + i, "end": "death", "floor": fights[-1][0]})
        (gen / "decisions.d" / f"{seed}.record.json").write_text(json.dumps(
            {"seed": seed, "fights": [_fight(f, nt, a) for f, nt, a in fights]}), encoding="utf-8")
        screens = [{"seed": seed, "ordinal": 0, "type": "neow", "floor": 1, "hp": 64, "max_hp": 80, "n_options": 3, "player": 2, "teacher": 1},
                   {"seed": seed, "ordinal": 2, "type": "card_reward", "floor": 2, "hp": 60, "max_hp": 80, "n_options": 4, "player": 2, "teacher": 2},
                   {"seed": seed, "ordinal": 4, "type": "rest", "floor": 8, "hp": 40, "max_hp": 80, "n_options": 7, "player": 1, "refused": "x"}]
        (gen / "decisions.d" / f"{seed}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in screens), encoding="utf-8")
    (gen / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return store


def fake_replay(task: tuple) -> dict:
    """What the child returns when every fight replays: one start per wanted fight, a document each."""
    _, seed, ix, info = task
    starts = [{**w, "k": 0, "node_type": {"elite": fg.NODE_ELITE, "boss": fg.NODE_BOSS}[w["kind"]], "hp": 70, "max_hp": 80, "deck_size": 12,
               "potions": 1, "relics": 3, "encounter_id": "ENC", "turn": 0, "doc": {"encounter_id": "ENC", "turn": 0},
               "kind_same": True, "replay_same": True, "source": "teacher"} for w in info["wanted"]]
    return {"_task": [task[0], seed, ix], "seed": seed, "ix": ix, "wanted": len(info["wanted"]), "captured": len(starts), "end": "stopped",
            "same": True, "starts": starts}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = make_store(self.tmp.name)
        self.loop = gl.Loop(Path(self.tmp.name) / "loop")

    def test_a_run_wants_its_elite_and_boss_fights_of_acts_one_and_two_only(self):
        runs = ts.finished_runs(self.store, (15,))
        self.assertEqual([r["seed"] for r in runs], list(RUNS))
        self.assertEqual({(r["agent"], r["agent_sha256"], r["job"]) for r in runs}, {(AGENT, SHA, "JOB1")})
        by = {r["seed"]: [(w["floor"], w["cat"]) for w in ts.wanted_of(ts.record_fights(self.store / "15", r["seed"]), r["seed"], 15)]
              for r in runs}
        self.assertEqual(by["S1AAAAAAAA"], [(6, "act1-elite"), (17, "act1-boss")])
        self.assertEqual(by["S2BBBBBBBB"], [(9, "act1-elite"), (17, "act1-boss"), (24, "act2-elite"), (33, "act2-boss")])
        self.assertEqual(by["S3CCCCCCCC"], [(5, "act1-elite")])

    def test_the_census_counts_the_starts_by_kind_without_playing(self):
        c = ts.census(self.store, (15,))
        self.assertEqual(c["runs"], 3)
        self.assertEqual(c["starts"], {"act1-elite": 3, "act1-boss": 2, "act2-elite": 1, "act2-boss": 1})
        self.assertEqual(c["runs_with"], {"act1-elite": 3, "act1-boss": 2, "act2-elite": 1, "act2-boss": 1})

    def test_a_run_of_a_job_without_a_shard_waits_and_an_unfinished_run_is_not_one(self):
        make_store(self.tmp.name + "/b", shard=False)
        self.assertEqual(ts.finished_runs(Path(self.tmp.name) / "b" / "run-teacher", (15,)), [])
        gen = self.store / "15"
        with open(gen / "runs.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"_task": ["rt-new", "S9", 1], "arm": "teacher", "job": "JOB1", "seed": "S9", "ix": 1, "capped": True,
                                 "cap_reason": "block-end"}) + "\n")
            fh.write(json.dumps({"_task": ["rt-new", "S8", 2], "arm": "teacher", "job": "JOB1", "seed": "S8", "ix": 2, "end": "timeout"}) + "\n")
        self.assertEqual([r["seed"] for r in ts.finished_runs(self.store, (15,))], list(RUNS))

    def test_the_stored_screens_are_the_records_by_ordinal_and_a_refused_one_keeps_the_players_pick(self):
        s = ts.stored_screens(self.store / "15", "S1AAAAAAAA")
        self.assertEqual(sorted(s), [0, 2, 4])
        self.assertEqual((s[0]["teacher"], s[2]["teacher"], s[4]["teacher"]), (1, 2, 1))
        self.assertEqual(s[0]["n_options"], 3)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = make_store(self.tmp.name)
        self.loop = gl.Loop(Path(self.tmp.name) / "loop")

    def _build(self, replay=fake_replay, **kw):
        with mock.patch.object(gl, "board_run_seeds", lambda: set()):
            return ts.build(self.loop, self.store, (15,), procs=1, replay=replay, **kw)

    def test_the_set_holds_one_start_per_wanted_fight_with_the_gyms_fields(self):
        meta = self._build()
        lite = json.loads((self.loop.starts / "teacher-starts.lite.json").read_text(encoding="utf-8"))
        self.assertEqual({s["id"]: s["cat"] for s in lite}, EXPECTED)
        self.assertEqual(meta["starts"], 7)
        self.assertEqual(meta["by_cat"], {"act1-elite": 3, "act1-boss": 2, "act2-elite": 1, "act2-boss": 1})
        for s in lite:
            self.assertEqual(s["source"], "teacher")
            self.assertEqual(pm.run_of(s["id"]), s["seed"])
            self.assertEqual(s["split"], "val" if gl.is_val_seed(s["seed"]) else "train")
            self.assertTrue(s["kind_same"] and s["replay_same"])
            self.assertNotIn("doc", s)
            self.assertEqual(s["night"], "teacher-g15")
        # the loader the daemon's games use reads it, through the byte index, and the document is there
        got = gl.load_start(str(self.loop.starts / "teacher-starts.jsonl"), "teacher:S2BBBBBBBB|f33")
        self.assertEqual((got["cat"], got["floor"], got["doc"]["encounter_id"]), ("act2-boss", 33, "ENC"))

    def test_the_replays_task_carries_what_the_child_needs_and_nothing_of_the_store(self):
        seen = []

        def spy(task):
            seen.append(task)
            return fake_replay(task)

        self._build(spy)
        t = next(t for t in seen if t[1] == "S2BBBBBBBB")
        self.assertEqual(t[0], "teacher-replay")
        info = t[3]
        self.assertEqual((info["agent"], info["agent_sha256"], info["generation"], info["job"]), (AGENT, SHA, 15, "JOB1"))
        self.assertEqual([w["floor"] for w in info["wanted"]], [9, 17, 24, 33])
        self.assertEqual(sorted(info["screens"]), [0, 2, 4])
        self.assertEqual({f["floor"] for f in info["fights"]}, {4, 9, 12, 17, 19, 24, 33, 35})  # every fight of the record checks the replay
        self.assertEqual(info["fights"][0], {"floor": 4, "node_type": 1, "act": 1, "hp": 70, "max_hp": 80, "deck": 12})

    def test_a_run_is_replayed_once_and_a_second_build_plays_only_the_new_ones(self):
        self._build()
        calls = []
        make_store(self.tmp.name + "/more", runs={"S4DDDDDDDD": [(5, 2, 0)]}, job="JOB2")  # a fourth run, in another job
        more = Path(self.tmp.name) / "more" / "run-teacher" / "15"
        with open(self.store / "15" / "runs.jsonl", "a", encoding="utf-8") as fh:
            fh.write((more / "runs.jsonl").read_text(encoding="utf-8"))
        for name in ("S4DDDDDDDD.record.json", "S4DDDDDDDD.jsonl"):
            (self.store / "15" / "decisions.d" / name).write_bytes((more / "decisions.d" / name).read_bytes())
        (self.store / "15" / "shards" / "JOB2").mkdir()
        (self.store / "15" / "shards" / "JOB2" / "manifest.json").write_text(json.dumps(
            {"job": "JOB2", "agent": AGENT, "agent_sha256": SHA, "engine": "e1", "generation": 15}), encoding="utf-8")

        def spy(task):
            calls.append(task[1])
            return fake_replay(task)

        meta = self._build(spy)
        self.assertEqual(calls, ["S4DDDDDDDD"])
        self.assertEqual(meta["starts"], 8)
        self.assertEqual(len(json.loads((self.loop.starts / "teacher-starts.lite.json").read_text(encoding="utf-8"))), 8)

    def test_the_boards_runs_and_the_exams_runs_are_never_replayed_and_never_in_the_set(self):
        (self.loop.root / "exam").mkdir()
        (self.loop.root / "exam" / "exclude_runs.json").write_text(json.dumps({"runs": ["S3CCCCCCCC"]}), encoding="utf-8")
        calls = []

        def spy(task):
            calls.append(task[1])
            return fake_replay(task)

        with mock.patch.object(gl, "board_run_seeds", lambda: {"S1AAAAAAAA"}):
            meta = ts.build(self.loop, self.store, (15,), procs=1, replay=spy)
        self.assertEqual(calls, ["S2BBBBBBBB"])
        self.assertEqual(meta["excluded"], 2)
        ids = {s["id"] for s in json.loads((self.loop.starts / "teacher-starts.lite.json").read_text(encoding="utf-8"))}
        self.assertEqual({pm.run_of(i) for i in ids}, {"S2BBBBBBBB"})
        # and a replay file that holds an excluded run (an earlier build, before the exam named it) is left out of the set
        with mock.patch.object(gl, "board_run_seeds", lambda: set()):
            self.loop = gl.Loop(self.loop.root)
            ts.build(self.loop, self.store, (15,), procs=1, replay=fake_replay)
        (self.loop.root / "exam" / "exclude_runs.json").write_text(json.dumps({"runs": ["S3CCCCCCCC", "S2BBBBBBBB"]}), encoding="utf-8")
        with mock.patch.object(gl, "board_run_seeds", lambda: set()):
            ts.build(self.loop, self.store, (15,), procs=1, replay=fake_replay)
        ids = {s["id"] for s in json.loads((self.loop.starts / "teacher-starts.lite.json").read_text(encoding="utf-8"))}
        self.assertEqual({pm.run_of(i) for i in ids}, {"S1AAAAAAAA"})

    def test_a_replay_that_failed_or_found_nothing_is_done_all_the_same(self):
        def broken(task):
            if task[1] == "S2BBBBBBBB":
                return {"_task": [task[0], task[1], task[2]], "seed": task[1], "capped": True, "cap_reason": "worker died"}
            if task[1] == "S3CCCCCCCC":
                return {**fake_replay(task), "starts": [], "captured": 0, "diverged": "fight 5: hp 70 against 66", "end": "diverged"}
            return fake_replay(task)

        meta = self._build(broken)
        self.assertEqual(meta["starts"], 2)
        self.assertEqual(meta["done"], {"S1AAAAAAAA": "ok", "S2BBBBBBBB": "capped: worker died", "S3CCCCCCCC": "diverged: fight 5: hp 70 against 66"})
        self.assertIsNone(ts.stale(self.loop, self.store, (15,), 1))

    def test_the_set_is_stale_when_there_is_none_and_when_enough_new_runs_have_finished(self):
        self.assertRegex(ts.stale(self.loop, self.store, (15,), 25), r"no teacher start set \(3 runs\)")
        self._build()
        self.assertIsNone(ts.stale(self.loop, self.store, (15,), 25))
        def finish(i: int) -> None:
            seed = f"N{i:09d}"
            with open(self.store / "15" / "runs.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"_task": ["rt-new", seed, i], "arm": "teacher", "job": "JOB1", "seed": seed, "ix": i, "end": "death"}) + "\n")
            (self.store / "15" / "decisions.d" / f"{seed}.record.json").write_text(json.dumps({"seed": seed, "fights": []}), encoding="utf-8")

        for i in range(24):
            finish(i)
        self.assertIsNone(ts.stale(self.loop, self.store, (15,), 25))
        finish(24)
        self.assertEqual(ts.stale(self.loop, self.store, (15,), 25), "25 new teacher runs")
        self.assertIsNone(ts.stale(self.loop, Path(self.tmp.name) / "nowhere", (15,), 1))  # no store: nothing to follow


class FakeSession:
    """A run as a script of events: ('decision', type, floor, hp, max_hp, n_options), ('fight', floor, hp, max_hp, deck, node_type), then
    the end. `play_fight` goes past the fight; `apply` records the option taken."""

    PHASE_COMBAT, PHASE_RUN = "combat", "run"

    def __init__(self, script):
        self.script, self.i, self.taken, self.won = list(script), 0, [], False
        self.act_index = 0

    @property
    def phase(self):
        return self.PHASE_COMBAT if self.i < len(self.script) and self.script[self.i][0] == "fight" else self.PHASE_RUN

    @property
    def info(self):
        ev = self.script[min(self.i, len(self.script) - 1)]
        if ev[0] == "fight":
            return {"floor": ev[1], "hp": ev[2], "max_hp": ev[3], "deck_size": ev[4], "node_type": ev[5], "relic_count": 3}
        return {"floor": ev[2], "hp": ev[3], "max_hp": ev[4], "deck_size": 12, "node_type": 0, "relic_count": 3}

    def decision(self):
        if self.i < len(self.script) and self.script[self.i][0] == "decision":
            ev = self.script[self.i]
            return mock.Mock(type=ev[1], options=[object() for _ in range(ev[5])], view=None)
        return None

    def view(self):
        ev = self.script[self.i]
        return mock.Mock(floor=ev[2], hp=ev[3], max_hp=ev[4], act_index=0)

    def apply(self, option):
        self.taken.append(self.script[self.i][1:3] + (option,))
        self.i += 1

    def close(self):
        pass


class FakePlayer:
    def __init__(self, session, picks):
        self.session, self.picks, self.asked = session, list(picks), []
        self.policy = mock.Mock()
        self.policy.pick.side_effect = self._pick

    def _pick(self, view, d, explore):
        explore.random()  # a draw from the exploration stream at every screen, as the run's own policy does
        self.asked.append(len(self.asked))
        return mock.Mock(index=self.picks[len(self.asked) - 1])

    def play_fight(self, s, rng, deadline):
        rng.random()
        s.i += 1
        return 3


class ReplayLoopTests(unittest.TestCase):
    INFO = {"wanted": [{"id": "teacher:S|f06", "cat": "act1-elite", "kind": "elite", "act": 1, "floor": 6, "seed": "S", "ix": 1, "night": "teacher-g15"},
                       {"id": "teacher:S|f17", "cat": "act1-boss", "kind": "boss", "act": 1, "floor": 17, "seed": "S", "ix": 1, "night": "teacher-g15"}],
            "fights": [{"floor": 3, "node_type": 1, "act": 1, "hp": 70, "max_hp": 80, "deck": 12},
                       {"floor": 6, "node_type": 2, "act": 1, "hp": 66, "max_hp": 80, "deck": 13},
                       {"floor": 17, "node_type": 6, "act": 1, "hp": 50, "max_hp": 80, "deck": 15}],
            "screens": {0: {"type": "neow", "floor": 1, "hp": 64, "max_hp": 80, "n_options": 3, "player": 2, "teacher": 1},
                        2: {"type": "card_reward", "floor": 2, "hp": 60, "max_hp": 80, "n_options": 4, "player": 2, "teacher": 3}},
            "generation": 15, "job": "J"}
    SCRIPT = [("decision", "neow", 1, 64, 80, 3), ("decision", "map", 1, 64, 80, 3), ("decision", "card_reward", 2, 60, 80, 4),
              ("fight", 3, 70, 80, 12, 1), ("decision", "map", 3, 62, 80, 2), ("fight", 6, 66, 80, 13, 2), ("decision", "rest", 7, 50, 80, 7),
              ("fight", 17, 50, 80, 15, 6), ("decision", "map", 18, 10, 80, 1), ("fight", 19, 10, 80, 15, 1)]

    def _run(self, script=None, picks=(2, 0, 2, 0, 0), info=None):
        import numpy as np

        s = FakeSession(script or self.SCRIPT)
        p = FakePlayer(s, picks)
        rng, explore = np.random.default_rng(1), np.random.default_rng(2)
        docs = []

        def capture(sess):
            docs.append(sess.info["floor"])
            return {"encounter_id": f"E{sess.info['floor']}", "turn": 0, "potions": {"max_slots": 2, "slots": [1, None]}}

        res = ts.replay_run(s, p, rng, explore, info or self.INFO, combat_phase=FakeSession.PHASE_COMBAT, capture=capture)
        return res, s, p, docs, rng, explore

    def test_a_stored_teacher_pick_stands_where_there_is_a_record_and_the_players_elsewhere(self):
        res, s, p, docs, _, _ = self._run()
        self.assertEqual(len(p.asked), 5)  # the player is asked at every screen: its streams run as in the run
        self.assertEqual(res["end"], "stopped")  # the last wanted fight is captured: the replay stops there
        self.assertEqual([st["id"] for st in res["starts"]], ["teacher:S|f06", "teacher:S|f17"])
        self.assertEqual(docs, [6, 17])
        self.assertEqual(res["checked"], 2 + 3)  # the two screens that have a record and the three fights before the stop
        self.assertIsNone(res["diverged"])

    def test_the_picks_taken_are_the_stored_ones_on_recorded_screens(self):
        taken_idx = []

        class Spy(FakeSession):
            def apply(self, option):
                taken_idx.append(option)
                super().apply(option)

        import numpy as np

        s = Spy(self.SCRIPT)
        p = FakePlayer(s, (2, 0, 2, 0, 0))
        built = []

        def decision():
            d = FakeSession.decision(s)
            if d is not None:
                d.options = list(range(len(d.options)))  # options are their own indices
            return d

        s.decision = decision
        ts.replay_run(s, p, np.random.default_rng(1), np.random.default_rng(2), self.INFO, combat_phase=FakeSession.PHASE_COMBAT,
                      capture=lambda sess: built.append(1) or {"turn": 0})
        # screens 0 and 2 have records (teacher picks 1 and 3); screen 1 (a map) has none: the player's pick (0); then 3 and 4 the player's
        self.assertEqual(taken_idx, [1, 0, 3, 0, 0])

    def test_the_start_has_the_documents_fields_and_the_info_of_the_fight_start(self):
        res, *_ = self._run()
        st = res["starts"][0]
        self.assertEqual((st["id"], st["cat"], st["kind"], st["act"], st["floor"], st["seed"], st["source"]),
                         ("teacher:S|f06", "act1-elite", "elite", 1, 6, "S", "teacher"))
        self.assertEqual((st["node_type"], st["hp"], st["max_hp"], st["deck_size"], st["relics"], st["potions"], st["encounter_id"], st["turn"]),
                         (2, 66, 80, 13, 3, 1, "E6", 0))
        self.assertTrue(st["kind_same"] and st["replay_same"])
        self.assertEqual(st["generation"], 15)

    def test_a_screen_that_differs_from_its_record_ends_the_replay_and_the_starts_before_it_stand(self):
        info = {**self.INFO, "screens": {**self.INFO["screens"], 2: {**self.INFO["screens"][2], "hp": 59}}}
        res, *_ = self._run(info=info)
        self.assertRegex(res["diverged"], r"^screen 2 \(card_reward\): hp 60 against 59")
        self.assertEqual(res["starts"], [])
        self.assertEqual(res["end"], "diverged")
        # the player's own pick differing from the stored one means the player is not the run's player
        info = {**self.INFO, "screens": {**self.INFO["screens"], 0: {**self.INFO["screens"][0], "player": 1}}}
        res, *_ = self._run(info=info)
        self.assertRegex(res["diverged"], r"^screen 0 \(neow\): the player's pick 2 against the run's 1")

    def test_a_fight_that_starts_with_other_hp_than_the_run_had_ends_it_after_the_starts_it_made(self):
        info = {**self.INFO, "fights": [self.INFO["fights"][0], self.INFO["fights"][1], {**self.INFO["fights"][2], "hp": 49}]}
        res, *_ = self._run(info=info)
        self.assertEqual([s["floor"] for s in res["starts"]], [6])  # the elite was made before the boss's fight differed
        self.assertRegex(res["diverged"], r"^fight on floor 17: hp 50 against 49")
        self.assertFalse(res["same"])
        self.assertEqual(res["captured"], 1)

    def test_a_fight_the_record_does_not_have_ends_the_replay(self):
        info = {**self.INFO, "fights": self.INFO["fights"][:1] + self.INFO["fights"][2:]}
        res, *_ = self._run(info=info)
        self.assertRegex(res["diverged"], r"^fight on floor 6: the run's record has none")

    def test_the_replay_stops_after_the_last_wanted_fight_and_plays_no_further(self):
        res, s, *_ = self._run()
        self.assertEqual(s.i, 7)  # the boss fight (event 7) is where it stopped, not played
        res2, s2, *_ = self._run(info={**self.INFO, "wanted": self.INFO["wanted"][:1]})
        self.assertEqual(s2.i, 5)  # stopped at the elite

    def test_a_run_that_ends_before_a_wanted_fight_says_so(self):
        res, *_ = self._run(script=self.SCRIPT[:7])
        self.assertEqual(res["end"], "death")
        self.assertEqual([s["floor"] for s in res["starts"]], [6])
        self.assertFalse(res["same"])  # one wanted fight was never reached


class SamplerTests(unittest.TestCase):
    def _pools(self, with_teacher: bool):
        def pool(prefix, cat, n, source):
            return [{"id": f"{prefix}:R{cat}{i:03d}|f10", "source": source, "file": f"{source}.jsonl", "score0": None} for i in range(n)]

        pools = {}
        for cat in ("act1-boss", "act2-boss", "act2-elite", "act1-hallway"):
            pools[(cat, "ai")] = pool("n", cat, 60, "night")
            pools[(cat, "human")] = pool("human", cat, 60, "human")
        if with_teacher:
            for cat in ("act1-boss", "act2-boss", "act2-elite"):
                pools[(cat, "teacher")] = pool("teacher", cat, 60, "teacher")
        return pools

    W = {"act1-boss": 1.0, "act2-boss": 1.0, "act2-elite": 1.0, "act1-hallway": 1.0}

    def _cfg(self, **over):
        cfg = {k: dict(v) for k, v in pm.DEFAULTS.items()}
        for k, v in over.items():
            cfg[k].update(v)
        return cfg

    def test_the_defaults_give_the_teacher_no_share_and_the_draws_of_today_are_unchanged_to_the_byte(self):
        self.assertEqual(set(pm.DEFAULTS["teacher_share"]), set(pm.KINDS))
        self.assertEqual(set(pm.DEFAULTS["teacher_share"].values()), {0.0})
        self.assertIn("teacher", pm.SOURCES)
        a = pm.Sampler(self._pools(False), self.W, self._cfg(), {}, {}, 5)
        b = pm.Sampler(self._pools(True), self.W, self._cfg(), {}, {}, 5)  # teacher starts in the pools, a share of none
        ra, rb = random.Random(9), random.Random(9)
        da = [a.draw(ra) for _ in range(400)]
        db = [b.draw(rb) for _ in range(400)]
        self.assertEqual([d["id"] for d in da], [d["id"] for d in db])
        self.assertEqual(ra.random(), rb.random())  # not a single extra draw from the generator

    def test_the_teacher_takes_its_share_of_a_kind_first_and_the_human_share_splits_the_rest(self):
        share = {"act2-boss": 0.5, "act2-elite": 0.5, "act1-boss": 0.25}
        s = pm.Sampler(self._pools(True), self.W, self._cfg(teacher_share=share), {}, {}, 5)
        rnd = random.Random(3)
        n = 24000
        got = Counter((d["cat"], d["source"]) for d in (s.draw(rnd) for _ in range(n)))
        kind = Counter()
        for (c, _), v in got.items():
            kind[c] += v
        for cat, t, h, a in (("act2-boss", 0.5, 0.25, 0.25), ("act2-elite", 0.5, 0.25, 0.25), ("act1-boss", 0.25, 0.375, 0.375)):
            self.assertAlmostEqual(got[(cat, "teacher")] / kind[cat], t, delta=0.03, msg=cat)
            self.assertAlmostEqual(got[(cat, "human")] / kind[cat], h, delta=0.03, msg=cat)
            self.assertAlmostEqual(got[(cat, "night")] / kind[cat], a, delta=0.03, msg=cat)
        self.assertEqual(got[("act1-hallway", "teacher")], 0)
        self.assertNotIn(("act1-hallway", "teacher"), got)

    def test_a_share_with_no_teacher_starts_draws_the_other_sides_and_a_kind_with_teacher_starts_only_draws_them(self):
        pools = self._pools(True)
        del pools[("act2-boss", "teacher")]
        s = pm.Sampler(pools, {"act2-boss": 1.0}, self._cfg(teacher_share={"act2-boss": 1.0}), {}, {}, 5)
        self.assertEqual({d["source"] for d in (s.draw(random.Random(i)) for i in range(60))}, {"night", "human"})
        pools = {("act2-boss", "teacher"): self._pools(True)[("act2-boss", "teacher")]}
        s = pm.Sampler(pools, {"act2-boss": 1.0}, self._cfg(), {}, {}, 5)
        self.assertEqual({d["source"] for d in (s.draw(random.Random(i)) for i in range(30))}, {"teacher"})

    def test_the_mix_line_names_the_teacher_source(self):
        drawn = [{"cat": "act2-boss", "source": s, "score": None} for s in ("teacher", "teacher", "human", "night")]
        self.assertRegex(pm.mix_line(pm.mix_summary(drawn)), r"act2-boss night 1 human 1 teacher 2")

    def test_a_config_file_sets_the_shares_and_leaves_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(pm.config(Path(tmp)), pm.DEFAULTS)
            (Path(tmp) / "mix.json").write_text(json.dumps({"regret": {"mode": "human-shortfall"}, "teacher_share": {"act2-boss": 0.5}}))
            cfg = pm.config(Path(tmp))
            self.assertEqual((cfg["teacher_share"]["act2-boss"], cfg["teacher_share"]["act2-elite"]), (0.5, 0.0))
            self.assertEqual(cfg["human_share"]["act2-boss"], 0.5)


class SampleTasksTests(unittest.TestCase):
    """`gym_loop.sample_tasks` with a teacher set beside the nights' and the human set: the tasks name the teacher file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "loop"
        self.loop = gl.Loop(self.root)
        sd = self.loop.starts
        night = [{"id": f"2026-11-01/g9:R{i:03d}|f33", "cat": "act2-boss", "seed": f"R{i:03d}"} for i in range(40)]
        teach = [{"id": f"teacher:T{i:03d}|f33", "cat": "act2-boss", "seed": f"T{i:03d}", "source": "teacher"} for i in range(40)]
        (sd / "starts.lite.json").write_text(json.dumps(night))
        (sd / "teacher-starts.lite.json").write_text(json.dumps(teach))
        (sd / "weights.json").write_text(json.dumps({"weights": {"act2-boss": 1.0}}))
        (sd / "mix.json").write_text(json.dumps({"teacher_share": {"act2-boss": 0.5}}))

    def test_half_the_act_two_boss_draws_come_from_the_teacher_file(self):
        tasks = gl.sample_tasks(self.loop, "c0001", 20000.0, "x16", 0.25, random.Random(4))
        self.assertGreater(len(tasks), 100)
        files = Counter(Path(t[4]["starts"]).name for t in tasks)
        self.assertEqual(set(files), {"starts.jsonl", "teacher-starts.jsonl"})
        self.assertAlmostEqual(files["teacher-starts.jsonl"] / len(tasks), 0.5, delta=0.1)
        for _, sid, _, _, info in tasks:
            self.assertEqual(sid.startswith("teacher:"), Path(info["starts"]).name == "teacher-starts.jsonl")
        self.assertRegex(pm.mix_line(self.loop.mix), r"act2-boss night \d+ teacher \d+")

    def test_the_exams_file_keeps_a_teacher_run_out_too(self):
        (self.root / "exam").mkdir()
        (self.root / "exam" / "exclude_runs.json").write_text(json.dumps({"runs": [f"T{i:03d}" for i in range(0, 40, 2)]}))
        tasks = gl.sample_tasks(self.loop, "c0001", 20000.0, "x16", 0.25, random.Random(5))
        for _, sid, _, _, _ in tasks:
            self.assertNotIn(pm.run_of(sid), [f"T{i:03d}" for i in range(0, 40, 2)])
        self.assertEqual(self.loop.mix["left_out"]["teacher"], 20)


class DaemonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = make_store(self.tmp.name)

    def _daemon(self, **kw):
        a = argparse.Namespace(root=str(Path(self.tmp.name) / "loop"), teacher_store=str(self.store), teacher_gens="15", teacher_min_new=25,
                               procs=1, prior="cpu", **kw)
        return gl.Daemon(a)

    def test_the_flag_is_off_by_default_and_the_step_then_does_nothing(self):
        with mock.patch.object(gl, "Daemon") as daemon:
            daemon.return_value.run.return_value = 0
            gl.main(["run", "--root", str(Path(self.tmp.name) / "loop")])
            a = daemon.call_args.args[0]
        self.assertEqual((a.teacher_store, a.teacher_gens, a.teacher_min_new), ("", "14,15", 25))
        d = self._daemon()
        d.a.teacher_store = ""
        with mock.patch.object(ts, "build") as build:
            self.assertFalse(d.teacher_step())
        build.assert_not_called()

    def test_a_stale_set_is_built_between_steps_and_a_current_one_is_left_alone(self):
        d = self._daemon()
        with mock.patch.object(ts, "build", return_value={"starts": 7}) as build, mock.patch.object(gl.Daemon, "status", lambda *a, **k: None):
            self.assertTrue(d.teacher_step())
        build.assert_called_once()
        self.assertEqual(build.call_args.args[1], self.store)
        self.assertEqual(tuple(build.call_args.args[2]), (15,))
        self.assertIn("teacher starts: building (no teacher start set (3 runs))", (Path(d.loop.root) / "loop.log").read_text(encoding="utf-8"))
        with mock.patch.object(gl, "board_run_seeds", lambda: set()):
            ts.build(d.loop, self.store, (15,), procs=1, replay=fake_replay)
        with mock.patch.object(ts, "build") as build:
            self.assertFalse(d.teacher_step())
        build.assert_not_called()

    def test_a_failing_build_is_logged_and_held_off_not_repeated_every_step(self):
        d = self._daemon()
        with mock.patch.object(ts, "build", side_effect=RuntimeError("no engine")) as build, mock.patch.object(gl.Daemon, "status", lambda *a, **k: None):
            self.assertFalse(d.teacher_step())
            self.assertFalse(d.teacher_step())
        self.assertEqual(build.call_count, 1)
        self.assertIn("teacher starts: build failed (RuntimeError: no engine)", (Path(d.loop.root) / "loop.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
