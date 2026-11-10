"""The judge corpus's builder (rl/judge_corpus_build.py): exclusions, the plan on a small set of starts, a fight's assembly (the door and three
turns, the ladder for bosses and elites, the teacher's own search for a hallway, the replay check), the compaction into shards (weights from the
plan, the human's result joined, stale shards removed), the self-play rows from a buffer shard, and the manifest.

The game itself is mocked here (the converge tests pin the replay and the rungs on a real fight); what is pinned is the corpus's own logic.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter, namedtuple
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import judge_corpus as jc  # noqa: E402
import judge_corpus_build as jb  # noqa: E402

Info = namedtuple("Info", "card_type rarity exhaust base_block")
CARDS = {472: Info("Attack", "Basic", False, 0), 131: Info("Skill", "Basic", False, 5), 10001: Info("Curse", "Curse", False, 0)}


def child(a: int, n: int, v: float | None, se: float | None = 1.0, t: int = -1, kind: str = "card", pd: float | None = 0.0) -> dict:
    return {"a": a, "t": t, "kind": kind, "card": 7, "merged": [], "n": n, "v": v, "se": se, "pd": pd}


def rec(rel, children: list[dict], cpu: float = 1.0, capped: bool = False, duel: dict | None = None) -> dict:
    children = sorted(children, key=lambda c: (-c["n"], c["a"], c["t"]))
    out = {"rel": rel, "budget": 16, "seed": 1, "sims": sum(c["n"] for c in children), "capped": capped, "cpu_s": cpu, "children": children,
           "best": children[0], "n_moves": len(children)}
    if duel is not None:
        out["duel"] = duel
        out["duel_cpu_s"] = 0.2
    return out


def runs_outside_the_exam(n: int) -> list[str]:
    import human_starts as hs

    return [r for r in (f"run{i}" for i in range(10 * n)) if hs.exam_bucket(r) != 0][:n]


class Toks:
    def __init__(self, n: int = 6):
        self.types = np.array([jc.T_GLOBAL, jc.T_HAND, jc.T_HAND, jc.T_DRAW, jc.T_RELIC, jc.T_POTION][:n], np.int64)
        self.ids = np.array([1, 1, 2, 1, 1, 0][:n], np.int64)
        self.owners = np.zeros(n, np.int64)
        self.feats = np.zeros((n, 18), np.float32)
        self.feats[0, jc.F_HP] = 0.5
        self.feats[5, jc.F_MAGNITUDE] = 1.0


class ExclusionTests(unittest.TestCase):
    def test_a_start_ids_run_is_its_part_before_the_bar_without_a_prefix(self):
        for sid, run in (("4XP3DMDKWN|f17", "4XP3DMDKWN"), ("2026-10-30/g9:8EABJ94H8D|f33", "8EABJ94H8D"), ("human:ab12|f8", "ab12"),
                         ("preloss:human:ab12|f33|j0|k24", "ab12")):
            self.assertEqual(jb.bare_run_of(sid), run)

    def test_the_exam_file_and_a_boards_start_lists_are_read(self):
        with tempfile.TemporaryDirectory() as d:
            exam = Path(d) / "exclude_runs.json"
            exam.write_text(json.dumps({"run_seeds": ["r1", "r2"], "start_ids": ["human:r1|f8"]}), encoding="utf-8")
            board = Path(d) / "meta-main.json"
            board.write_text(json.dumps({"act1_boss_starts": ["4XP|f17"], "act2_boss_starts": ["2026-10-30/g9:8EA|f33"], "timing_starts": {"a": ["X|f3"], "b": 40},
                                         "player": "v2"}), encoding="utf-8")
            runs, ids = jb.excluded_from(exam)
            self.assertEqual((runs, ids), ({"r1", "r2"}, {"human:r1|f8"}))
            runs, ids = jb.excluded_from(board)
            self.assertEqual(runs, {"4XP", "8EA", "X"})
            self.assertEqual(ids, {"4XP|f17", "2026-10-30/g9:8EA|f33", "X|f3"})

    def test_every_lab_board_and_the_exam_are_gathered(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "loop" / "exam").mkdir(parents=True)
            (root / "loop" / "exam" / "exclude_runs.json").write_text(json.dumps({"run_seeds": ["r1"], "start_ids": []}), encoding="utf-8")
            b = root / "LAB" / "results" / "fight-gym" / "2026-11-06-board-a"
            b.mkdir(parents=True)
            (b / "meta-main.json").write_text(json.dumps({"act1_boss_starts": ["Z|f17"]}), encoding="utf-8")
            m2 = root / "LAB" / "results" / "fight-gym" / "2026-10-29-m2"
            m2.mkdir(parents=True)
            (m2 / "starts.jsonl").write_text(json.dumps({"id": "M|f17"}) + "\n", encoding="utf-8")
            with mock.patch.object(jc, "REFEREE", root / "LAB" / ".."):
                (root / "lab").mkdir(exist_ok=True)
                # the lab folder is `lab` under the referee: put the boards there
                (root / "lab" / "results" / "fight-gym").mkdir(parents=True, exist_ok=True)
                (root / "lab" / "results" / "fight-gym" / "2026-11-06-board-a").mkdir(exist_ok=True)
                (root / "lab" / "results" / "fight-gym" / "2026-11-06-board-a" / "meta-main.json").write_text(json.dumps({"act1_boss_starts": ["Z|f17"]}), encoding="utf-8")
                runs, ids, names = jb.exclusions(root / "loop")
        self.assertIn("r1", runs)
        self.assertTrue(any("exclude_runs.json" in n for n in names))


class FeatureAndPlanTests(unittest.TestCase):
    def doc(self, deck: list[int], relics: list[int], potions: list[int]) -> dict:
        card = lambda i: {"def_id": i}  # noqa: E731
        return {"piles": {"hand": [card(i) for i in deck[:2]], "draw": {"known_top": [], "middle": [card(i) for i in deck[2:]], "known_bottom": []}, "discard": [],
                          "exhaust": []},
                "relics": [{"id": r, "counter": 0} for r in relics], "potions": {"max_slots": 2, "slots": potions + [0] * (6 - len(potions))}}

    def test_a_starts_strata_from_its_document(self):
        rec_ = {"doc": self.doc([472, 472, 131, 10001], [36, 4, 36], [24]), "encounter_id": 68, "hp": 40, "max_hp": 80}
        f = jb.start_features(rec_, CARDS)
        self.assertEqual(f["relics"], [4, 36])
        self.assertEqual(f["deck"], {472: 2, 131: 1, 10001: 1})
        self.assertEqual((f["potions"], f["hp"], f["max_hp"], f["arch"]), (1, 40, 80, "starter"))
        self.assertEqual(f["enemy"], "byrdonis")

    def test_the_plan_leaves_out_the_exam_and_keeps_the_rest_in_order(self):
        import human_starts as hs

        runs = runs_outside_the_exam(40)
        with tempfile.TemporaryDirectory() as d:
            loop, store = Path(d) / "loop", Path(d) / "store"
            (loop / "starts").mkdir(parents=True)
            (loop / "exam").mkdir()
            lite, lines, index, off = [], [], {}, 0
            for i, r in enumerate(runs):
                kind, act, cat = [("boss", 1, "act1-boss"), ("elite", 2, "act2-elite")][i % 2]
                sid = f"human:{r}|f{17 if kind == 'boss' else 20}"
                rec_ = {"id": sid, "cat": cat, "kind": kind, "act": act, "floor": 17, "run": r, "enemy": "X", "encounter_id": 68, "hp": 50, "max_hp": 80,
                        "doc": self.doc([472, 131], [36] + ([4] if i % 5 == 0 else []), [])}
                lite.append({k: v for k, v in rec_.items() if k != "doc"})
                line = json.dumps(rec_) + "\n"
                index[sid] = off
                off += len(line.encode())
                lines.append(line)
            (loop / "starts" / "human-starts.jsonl").write_text("".join(lines), encoding="utf-8", newline="\n")
            (loop / "starts" / "human-starts.index.json").write_text(json.dumps(index), encoding="utf-8")
            (loop / "starts" / "human-starts.lite.json").write_text(json.dumps(lite), encoding="utf-8")
            (loop / "exam" / "exclude_runs.json").write_text(json.dumps({"run_seeds": [runs[0]], "start_ids": []}), encoding="utf-8")
            args = mock.Mock(store=str(store), loop=str(loop), support=100)
            with (mock.patch.object(hs, "read_exam_ids", return_value=frozenset({"nobody"})), mock.patch.object(jc, "REFEREE", Path(d) / "none"),
                  mock.patch.object(jc, "QUOTA", {"act1-boss": 5, "act2-elite": 100, "act2-boss": 0, "act1-elite": 0, "act2-hallway": 0, "act1-hallway": 0})):
                self.assertEqual(jb.cmd_plan(args), 0)
            plan = json.loads((store / "work" / "plan.json").read_text(encoding="utf-8"))
            pool = jb.read_jsonl(store / "work" / "pool.jsonl")
            self.assertEqual(len(pool), 39)  # the excluded run is out
            self.assertNotIn(f"human:{runs[0]}|f17", {r["id"] for r in pool})
            self.assertEqual(plan["cells"], {"act1-boss": 19, "act2-elite": 20})
            self.assertEqual(Counter(jb.read_json(store / "work" / "plan.json")["order"][:5]).total(), 5)
            by = {r["id"]: r for r in pool}
            cells = Counter(by[s]["cell"] for s in plan["order"])
            self.assertEqual(cells, {"act1-boss": 5, "act2-elite": 20})
            # a start with the rarer relic (4) is boosted over the others in its pool
            boosts = {r["id"]: r["boost"] for r in pool}
            self.assertTrue(all(b >= 1.0 for b in boosts.values()))
            # relic 4 is in three of the nineteen boss starts; five of them are to be built (rate 5/19), support 100
            self.assertAlmostEqual(max(boosts.values()), 100 / (3 * 5 / 19), places=2)

    def test_inclusion_per_cell_follows_the_built_count(self):
        pool = [{"id": f"a{i}", "cell": "act1-boss", "boost": 1.0} for i in range(10)] + [{"id": f"b{i}", "cell": "act2-boss", "boost": 1.0} for i in range(4)]
        pi = jb.cell_inclusion(pool, {"a0", "a1", "a2", "a3", "a4", "b0", "b1", "b2", "b3"})
        self.assertAlmostEqual(pi["a0"], 0.5)
        self.assertEqual(pi["b0"], 1.0)

    def test_the_humans_result_is_the_sidecars_in_short_names(self):
        e = {"human_survived": True, "human_end_hp": 37, "human_hp_lost": 25, "human_damage_taken": 25, "human_post_heal": 6, "human_turns": 4,
             "human_potions_used": 1, "human_potions_held_end": 0, "human_l0": 37.0, "human_l0_unit": 0.7312}
        self.assertEqual(jb.human_outcome(e), {"survived": True, "end_hp": 37, "hp_lost": 25, "turns": 4, "potions_used": 1, "potions_held_end": 0, "l0": 37.0, "l0_unit": 0.7312})
        self.assertIsNone(jb.human_outcome(None))

    def test_attempts_are_counted_and_pending_fights_are_the_plans_without_a_result(self):
        with tempfile.TemporaryDirectory() as d:
            store = Path(d)
            self.assertEqual([jb.note_attempt(store, "x"), jb.note_attempt(store, "x"), jb.note_attempt(store, "y")], [1, 2, 1])
            plan = {"order": ["a", "b", "c"]}
            jc.write_atomic(jb.fight_path(store, "b"), {"start": "b"})
            self.assertEqual(jb.pending(store, plan), ["a", "c"])
            self.assertEqual(jb.fight_path(store, "b"), jb.fight_path(store, "b"))


class BuildFightTests(unittest.TestCase):
    """`build_fight` with the game mocked: which turns are kept, which rungs are searched, what a position row carries."""

    def run_build(self, kind: str, eligible: list[int], searches: list[int], settled: bool = True, replay_ok: bool = True, capped_game: bool = False):
        import converge as cv

        n = 12
        dec = [{"k": k, "n_moves": 3 if k in eligible else 1, "pos_kind": "play", "move": [k % 4, -1], "obs": f"h{k}",
                "search": {**rec("game", [child(1, 400, 12.0), child(2, 80, 11.0)]), "best": {**child(1, 400, 12.0), "kind": "end_turn" if k % 5 == 4 else "card"}}}
               for k in range(n)]
        game = {"won": True, "end_hp": 22, "max_hp": 80, "turns": 3, "decisions": n, "sims": 999, "capped": capped_game, "moves_sha": "x", "potions_held": 1, "L0": 0.6,
                "cpu_s": 5.0, "wall_s": 6.0}
        world = mock.Mock()
        world.hp.return_value, world.max_hp.return_value = 40, 80

        def search_rung(w, s, kind_, rel, k, extra, pairs=0):
            searches.append((k, rel, pairs))
            gap = 6.0 if settled else 1.0
            return rec(rel, [child(1, 900, 13.0), child(2, 60, 5.0)], cpu=2.0, duel={"n": pairs, "mean": gap, "se": 1.0, "p_death_diff": 0.0})

        item = {"id": "human:run1|f33", "run": "run1", "kind": kind, "act": 2, "cell": f"act2-{kind}", "enemy": "kaiser-crab-boss"}
        ctx = {"extra": {}, "starts_path": "unused", "inv_card": {1: 472, 2: 131}, "inv_relic": {1: 36}, "cards": CARDS, "pairs": {2: 128, 4: 256}}
        start = {"id": item["id"], "node_type": 6, "floor": 33, "max_hp": 80}
        labels: list[str] = []
        with (mock.patch("gym_loop.load_start", return_value=start), mock.patch.object(cv, "play_teacher", return_value=(game, dec)),
              mock.patch.object(cv, "replay_to", return_value=world), mock.patch.object(cv, "obs_hash", side_effect=lambda w: "bad" if not replay_ok else w.last),
              mock.patch.object(cv, "search_rung", side_effect=search_rung), mock.patch("prior_client.encode_position", side_effect=lambda h, probe: Toks())):
            # replay_to returns a world whose last hash the mocked obs_hash reads: set it from the k asked for
            def replay(s, moves, k):
                world.last = f"h{k}"
                return world

            with mock.patch.object(cv, "replay_to", side_effect=replay):
                res = jb.build_fight(item, ctx, log=lambda m: None, pause=labels.append)
        return res, labels

    def test_a_boss_fight_keeps_the_door_and_three_turns_and_settles_each_at_the_first_rung(self):
        searches: list = []
        res, labels = self.run_build("boss", list(range(2, 11)), searches)
        ks = [p["k"] for p in res["positions"]]
        self.assertEqual(ks[0], 0)
        self.assertEqual([p["stage"] for p in res["positions"]], ["door", "early", "middle", "late"])
        self.assertEqual(len(ks), 4)
        self.assertTrue(all(2 <= k <= 10 for k in ks[1:]))
        self.assertEqual([s[1] for s in searches], [2, 2, 2])  # one rung a turn: settled at x32, no deeper rung
        self.assertEqual({s[2] for s in searches}, {128})
        door = res["positions"][0]
        self.assertEqual(door["label"]["tier"], "played")
        self.assertTrue(door["door"])
        for p in res["positions"][1:]:
            self.assertEqual(p["label"]["tier"], "settled")
            self.assertEqual(p["label"]["rel"], 2)
            self.assertEqual(p["cpu_s"], 2.2)
        p = res["positions"][1]
        self.assertEqual((p["src"], p["kind"], p["act"], p["enemy"], p["floor"], p["run"]), ("human", "boss", 2, "kaiser-crab-boss", 33, "run1"))
        self.assertEqual((p["hp"], p["max_hp"], p["door_max_hp"], p["potions"], p["relics"]), (40, 80, 80, 1, [36]))
        self.assertEqual(p["deck"], {"472": 2, "131": 1})  # Toks: two cards in hand and one in the draw pile
        self.assertEqual(p["teacher"], {"won": True, "end_hp": 22, "max_hp": 80, "potions_held": 1, "L0": 0.6, "decisions": 12, "turns": 3})
        self.assertEqual(p["replay"]["ns"], "x")
        self.assertEqual(len(jc.moves_of(p["replay"]["moves"])), p["k"])
        self.assertEqual(p["played"], [p["k"] % 4, -1])
        self.assertEqual(jc.tokens_of_b64(p["tok"])["ids"].tolist(), [1, 1, 2, 1, 1, 0])
        self.assertEqual(res["decisions"], 12)
        self.assertEqual(labels[0], "act2-boss game")
        self.assertIn("act2-boss rung 2x", labels)
        json.dumps(res)

    def test_the_door_is_searched_on_the_ladder_when_it_has_a_choice(self):
        searches: list = []
        res, _ = self.run_build("boss", [0, *range(2, 11)], searches)
        door = next(p for p in res["positions"] if p["k"] == 0)
        self.assertTrue(door["door"])
        self.assertEqual((door["label"]["tier"], door["label"]["rel"]), ("settled", 2))
        self.assertIn(0, [s[0] for s in searches])

    def test_a_turn_the_rule_never_holds_for_goes_to_the_cap_and_is_soft(self):
        searches: list = []
        res, _ = self.run_build("elite", list(range(2, 11)), searches, settled=False)
        self.assertEqual(sorted({s[1] for s in searches}), [2, 4])
        self.assertEqual({s[2] for s in searches if s[1] == 4}, {256})
        for p in res["positions"][1:]:
            self.assertEqual((p["label"]["tier"], p["label"]["rel"]), ("soft", 4))
            self.assertEqual([r["rel"] for r in p["label"]["rungs"]], [1, 2, 4])

    def test_a_hallway_turn_keeps_the_teachers_own_search(self):
        searches: list = []
        res, _ = self.run_build("hallway", list(range(2, 11)), searches)
        self.assertEqual(searches, [])
        self.assertEqual({p["label"]["tier"] for p in res["positions"]}, {"played"})
        self.assertEqual(res["positions"][1]["label"]["rel"], 1)

    def test_a_fight_with_no_choice_keeps_only_the_door(self):
        res, _ = self.run_build("boss", [], [])
        self.assertEqual([p["stage"] for p in res["positions"]], ["door"])

    def test_a_replay_that_does_not_reach_the_same_position_keeps_nothing_of_it(self):
        res, _ = self.run_build("boss", list(range(2, 11)), [], replay_ok=False)
        self.assertEqual(res["positions"], [])
        self.assertEqual(len(res["problems"]), 4)

    def test_a_game_cut_by_a_cap_has_no_teacher_outcome(self):
        res, _ = self.run_build("hallway", [3, 4, 5, 6], [], capped_game=True)
        self.assertIsNone(res["positions"][0]["teacher"])
        self.assertIsNone(jb.teacher_result({"capped": True}))


class SupervisorTests(unittest.TestCase):
    """`run` starts the work processes, starts a dead one again, and ends when the plan has no fight left or a STOP file stands."""

    class Fake:
        codes: list[list[int | None]] = []
        made: list[list[str]] = []

        def __init__(self, cmd, **kw):
            type(self).made.append(cmd)
            self.rc_seq = list(type(self).codes.pop(0)) if type(self).codes else [0]

        def poll(self):
            return self.rc_seq.pop(0) if len(self.rc_seq) > 1 else self.rc_seq[0]

    def args(self, store: Path, procs: int = 2):
        return mock.Mock(store=str(store), loop="L", player="v3", procs=procs, cells="")

    def test_a_dead_work_process_is_started_again_and_the_supervisor_ends_with_the_plan(self):
        with tempfile.TemporaryDirectory() as d:
            store = Path(d)
            jc.write_atomic(jb.work_dir(store) / "plan.json", {"order": ["a", "b"]})
            self.Fake.codes, self.Fake.made = [[1], [0]], []  # w0 dies at once with 1, w1 ends well; w0's second life ends well
            with (mock.patch.object(jb.subprocess, "Popen", self.Fake), mock.patch.object(jb.time, "sleep"), mock.patch.object(jb, "pending", return_value=[]),
                  mock.patch.object(jb.subprocess, "run") as run):
                self.assertEqual(jb.cmd_run(self.args(store)), 0)
            self.assertEqual(len(self.Fake.made), 3)
            self.assertEqual(run.call_args_list[-1].args[0][3], "finalize")  # the last step: compaction, coverage and the manifest's complete flag
            self.assertTrue(all("--shard" in c and "work" in c for c in self.Fake.made))
            self.assertEqual([c[c.index("--shard") + 1] for c in self.Fake.made], ["w0", "w1", "w0"])

    def test_fights_left_behind_by_a_finished_pass_get_one_more_pass_up_to_three(self):
        with tempfile.TemporaryDirectory() as d:
            store = Path(d)
            jc.write_atomic(jb.work_dir(store) / "plan.json", {"order": ["a"]})
            self.Fake.codes, self.Fake.made = [], []
            with (mock.patch.object(jb.subprocess, "Popen", self.Fake), mock.patch.object(jb.time, "sleep"), mock.patch.object(jb, "pending", return_value=["a"]),
                  mock.patch.object(jb.subprocess, "run")):
                jb.cmd_run(self.args(store, procs=1))
            self.assertEqual(len(self.Fake.made), 4)  # the first pass and three more

    def test_a_stop_file_ends_it_without_new_work(self):
        with tempfile.TemporaryDirectory() as d:
            store = Path(d)
            jc.write_atomic(jb.work_dir(store) / "plan.json", {"order": ["a"]})
            self.Fake.codes, self.Fake.made = [], []

            def sleep(_s):
                (jb.work_dir(store) / jb.STOP_NAME).write_text("x")

            with (mock.patch.object(jb.subprocess, "Popen", self.Fake), mock.patch.object(jb.time, "sleep", side_effect=sleep), mock.patch.object(jb, "pending", return_value=["a"]),
                  mock.patch.object(jb.subprocess, "run")):
                jb.cmd_run(self.args(store, procs=1))
            self.assertEqual(len(self.Fake.made), 1)


class WorkTests(unittest.TestCase):
    def store(self, d: str, n: int = 6) -> Path:
        store = Path(d)
        ids = [f"human:r{i}|f33" for i in range(n)]
        jc.write_atomic(jb.work_dir(store) / "plan.json", {"order": ids})
        (jb.work_dir(store) / "pool.jsonl").write_text("".join(json.dumps({"id": s, "run": f"r{i}", "cell": "act2-boss", "kind": "boss", "act": 2, "enemy": "x"}) + "\n"
                                                                for i, s in enumerate(ids)), encoding="utf-8")
        return store

    def args(self, store: Path, **kw):
        return mock.Mock(store=str(store), loop="L", player="v3", shard="t", cells="", limit=kw.get("limit", 0))

    def test_each_fight_is_claimed_built_and_written_once(self):
        import converge as cv

        with tempfile.TemporaryDirectory() as d:
            store = self.store(d)
            built: list[str] = []

            def build(item, ctx, say, pause):
                built.append(item["id"])
                return {"start": item["id"], "cell": item["cell"], "positions": [], "cpu_s": 1.0}

            with mock.patch.object(jb, "make_ctx", return_value={}), mock.patch.object(jb, "build_fight", side_effect=build), \
                    mock.patch.object(cv, "pause_point", return_value=0.0):
                self.assertEqual(jb.cmd_work(self.args(store, limit=4)), 0)
                self.assertEqual(len(built), 4)
                self.assertEqual(jb.cmd_work(self.args(store)), 0)  # the rest; the four with a result are not built again
            self.assertEqual(len(built), 6)
            self.assertEqual(len(set(built)), 6)
            self.assertEqual(len(jb.read_fights(store)), 6)

    def test_three_failed_fights_in_a_row_stop_the_process_before_the_plan_is_spent(self):
        import converge as cv

        with tempfile.TemporaryDirectory() as d:
            store = self.store(d, n=8)
            with mock.patch.object(jb, "make_ctx", return_value={}), mock.patch.object(jb, "build_fight", side_effect=RuntimeError("boom")), \
                    mock.patch.object(cv, "pause_point", return_value=0.0):
                self.assertEqual(jb.cmd_work(self.args(store)), 3)
            fights = jb.read_fights(store)
            self.assertEqual(len(fights), 3)
            self.assertTrue(all("boom" in f["error"] for f in fights))

    def test_a_stop_file_ends_the_process_between_fights(self):
        import converge as cv

        with tempfile.TemporaryDirectory() as d:
            store = self.store(d)
            (jb.work_dir(store) / jb.STOP_NAME).write_text("x")
            with mock.patch.object(jb, "make_ctx", return_value={}), mock.patch.object(jb, "build_fight") as bf, mock.patch.object(cv, "pause_point", return_value=0.0):
                self.assertEqual(jb.cmd_work(self.args(store)), 0)
            bf.assert_not_called()

    def test_a_fight_whose_workers_died_three_times_is_recorded_as_failed_not_built(self):
        import converge as cv

        with tempfile.TemporaryDirectory() as d:
            store = self.store(d, n=1)
            for _ in range(3):
                jb.note_attempt(store, "human:r0|f33")
            with mock.patch.object(jb, "make_ctx", return_value={}), mock.patch.object(jb, "build_fight") as bf, mock.patch.object(cv, "pause_point", return_value=0.0):
                jb.cmd_work(self.args(store))
            bf.assert_not_called()
            self.assertIn("died", jb.read_fights(store)[0]["error"])


class CompactTests(unittest.TestCase):
    def fight(self, store: Path, run: str, n_pos: int = 3, kind: str = "boss", act: int = 2) -> str:
        sid = f"human:{run}|f33"
        positions = []
        for i in range(n_pos):
            r = {"pid": f"{sid}#{i}", "src": "human", "sub": "human-start", "start": sid, "run": run, "split": jc.split_of(run), "kind": kind, "act": act,
                 "enemy": "x", "floor": 33, "k": i * 3, "turn": i + 1, "stage": ["early", "middle", "late"][i % 3], "stage_n": 3, "decisions": 12, "door": i == 0,
                 "hp": 40, "max_hp": 80, "door_max_hp": 80, "hp_frac": 0.5, "potions": 0, "relics": [1], "deck": {"472": 4}, "deck_n": 4, "arch": "starter",
                 "label": {"tier": "settled", "rel": 2, "sims": 9, "root_value": 3.0, "best_value": 3.0, "p_death": 0.0, "children": []},
                 "teacher": {"won": True, "end_hp": 10, "max_hp": 80, "potions_held": 0, "L0": 0.5, "decisions": 12, "turns": 3}, "tok": jc.b64_tokens(
                     {"types": np.full(4 + i, 2, np.uint8), "ids": np.arange(4 + i, dtype=np.int32), "owners": np.zeros(4 + i, np.uint8),
                      "feats": np.zeros((4 + i, 18), np.float32)})}
            positions.append(r)
        jc.write_atomic(jb.fight_path(store, sid), {"start": sid, "cell": f"act{act}-{kind}", "kind": kind, "act": act, "run": run, "enemy": "x", "game": {},
                                                     "decisions": 12, "eligible": 9, "cpu_s": 30.0, "done": "t", "problems": [], "positions": positions})
        return sid

    def setup(self, d: str, n_runs: int = 6) -> tuple[Path, Path, list[str]]:
        store, loop = Path(d) / "store", Path(d) / "loop"
        (loop / "starts").mkdir(parents=True)
        (loop / "exam").mkdir()
        runs = runs_outside_the_exam(n_runs)
        pool = [{"id": f"human:{r}|f33", "run": r, "cell": "act2-boss", "kind": "boss", "act": 2, "boost": 1.0 if i < 4 else 4.0} for i, r in enumerate(runs)]
        jb.work_dir(store).mkdir(parents=True)
        (jb.work_dir(store) / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool), encoding="utf-8")
        jc.write_atomic(jb.work_dir(store) / "plan.json", {"order": [p["id"] for p in pool], "quota": {"act2-boss": 4}, "support": 100, "cells": {"act2-boss": 6}})
        out = {f"human:{r}|f33": {"human_survived": i % 2 == 0, "human_end_hp": 30 if i % 2 == 0 else 0, "human_hp_lost": 40, "human_turns": 5, "human_potions_used": 1,
                                  "human_potions_held_end": 0, "human_l0": 30.0, "human_l0_unit": 0.6} for i, r in enumerate(runs)}
        (loop / "starts" / "human-outcomes.json").write_text(json.dumps({"rule": "x", "outcomes": out}), encoding="utf-8")
        (loop / "exam" / "exclude_runs.json").write_text(json.dumps({"run_seeds": [], "start_ids": []}), encoding="utf-8")
        return store, loop, runs

    def compact(self, store: Path, loop: Path) -> None:
        import human_starts as hs

        with (mock.patch.object(hs, "read_exam_ids", return_value=frozenset({"nobody"})), mock.patch.object(jc, "REFEREE", store.parent / "none"),
              mock.patch.object(jb, "jc_exam_ids_path", return_value=store.parent / "none.txt")):
            jb.cmd_compact(mock.Mock(store=str(store), loop=str(loop)))

    def test_fights_become_shards_with_weights_outcomes_and_a_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            store, loop, runs = self.setup(d)
            sids = [self.fight(store, r) for r in runs[:4]]
            self.compact(store, loop)
            c = jc.Corpus(store)
            rows = [r for r, _, _ in c.iter_rows()]
            self.assertEqual(len(rows), 12)
            self.assertTrue(all("tok" not in r for r in rows))
            # four of the six starts were built: the two boosted starts (4.0) were not, so the built four share pi = 1 under n = 4 of boosts 1,1,1,1,4,4?  lambda*b with sum 4
            pis = {r["start"]: r["pi"] for r in rows}
            self.assertAlmostEqual(sum(jc.inclusion([1.0] * 4 + [4.0] * 2, 4)), 4.0, places=5)
            self.assertEqual(set(pis), set(sids))
            self.assertAlmostEqual(rows[0]["w"], 1 / rows[0]["pi"], places=3)
            self.assertAlmostEqual(float(np.mean([r["w_cell"] for r in rows])), 1.0, places=2)
            h = [r for r in rows if r["start"] == sids[0]][0]["human"]
            self.assertEqual((h["survived"], h["end_hp"], h["l0_unit"]), (True, 30, 0.6))
            self.assertEqual([r["human"]["survived"] for r in rows if r["start"] == sids[1]][0], False)
            first = next(iter(c.iter_positions(src="human")))
            self.assertEqual(first.tokens["ids"].tolist(), [0, 1, 2, 3])
            man = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(man["schema"], jc.SCHEMA)
            self.assertEqual(man["counts"]["positions"], 12)
            self.assertEqual(man["counts"]["human_fights"], 4)
            self.assertEqual(man["counts"]["by_source_kind_act"], {"human-boss-act2": 12})
            self.assertEqual(man["cpu_hours"]["fights_built"], 4)
            self.assertAlmostEqual(man["cpu_hours"]["human_fights"], 4 * 30.0 / 3600, places=2)
            self.assertEqual(set(man["shards"]), {"human-boss-act2-0000"})
            self.assertIn("attribution", man)
            self.assertNotIn("C:/Users", json.dumps(man))

    def test_a_second_compaction_after_more_fights_replaces_the_shards_and_a_failed_fight_is_left_out(self):
        with tempfile.TemporaryDirectory() as d:
            store, loop, runs = self.setup(d)
            for r in runs[:2]:
                self.fight(store, r)
            jc.write_atomic(jb.fight_path(store, f"human:{runs[5]}|f33"), {"start": f"human:{runs[5]}|f33", "cell": "act2-boss", "error": "boom", "positions": []})
            self.compact(store, loop)
            self.assertEqual(len(list(jc.Corpus(store).iter_rows())), 6)
            stale = store / "shards" / "human-boss-act1-0000.jsonl"
            stale.write_text("{}\n", encoding="utf-8")
            stale.with_suffix(".npz").write_bytes(b"x")
            for r in runs[2:4]:
                self.fight(store, r, n_pos=2)
            self.compact(store, loop)
            self.assertEqual(len(list(jc.Corpus(store).iter_rows())), 10)
            self.assertFalse(stale.exists())
            self.assertEqual(json.loads((store / "manifest.json").read_text(encoding="utf-8"))["cpu_hours"]["fights_failed"], 1)

    def test_the_compaction_refuses_a_run_of_the_exam_fifth(self):
        import human_starts as hs

        with tempfile.TemporaryDirectory() as d:
            store, loop, runs = self.setup(d)
            bad = next(f"x{i}" for i in range(1000) if hs.exam_bucket(f"x{i}") == 0)
            self.fight(store, bad)
            with self.assertRaises(AssertionError), mock.patch.object(hs, "read_exam_ids", return_value=frozenset({"nobody"})):
                jb.cmd_compact(mock.Mock(store=str(store), loop=str(loop)))


class ReadinessTests(unittest.TestCase):
    """The manifest's flags for the judge-training lane."""

    def rows(self, boss1: int, boss2: int, elite1: int, elite2: int, tier: str = "settled") -> list[dict]:
        out = []
        for kind, act, n in (("boss", 1, boss1), ("boss", 2, boss2), ("elite", 1, elite1), ("elite", 2, elite2)):
            out += [{"src": "human", "kind": kind, "act": act, "label": {"tier": tier}} for _ in range(n)]
        out += [{"src": "selfplay", "kind": "boss", "act": 1, "label": {"tier": "played"}}] * 5
        return out

    def test_usable_once_each_kind_and_act_has_enough_labelled_turns_and_it_stays(self):
        few = jb.readiness(self.rows(400, 100, 600, 600), {}, {"order": list(range(10))}, 40, 0)
        self.assertFalse(few["usable_for_training"])  # act 2 bosses under 300
        self.assertNotIn("usable_counts", few)
        ok = jb.readiness(self.rows(700, 400, 600, 600, "soft"), {}, {"order": list(range(10))}, 90, 1)
        self.assertTrue(ok["usable_for_training"])
        self.assertEqual(ok["usable_counts"]["human_boss_labelled_by_act"], {"act1": 700, "act2": 400})
        self.assertEqual(ok["usable_counts"]["selfplay_positions"], 5)
        self.assertEqual(ok["usable_counts"]["human_fights_built"], 90)
        self.assertEqual((ok["human_build"]["planned"], ok["human_build"]["built"], ok["human_build"]["failed"]), (10, 90, 1))
        later = jb.readiness(self.rows(5000, 5000, 5000, 5000), {"usable_for_training": True, "usable_counts": ok["usable_counts"]}, {}, 500, 0)
        self.assertEqual(later["usable_counts"], ok["usable_counts"])  # the counts at the moment it became usable
        dip = jb.readiness(self.rows(1, 1, 1, 1), {"usable_for_training": True, "usable_counts": ok["usable_counts"]}, {}, 5, 0)
        self.assertTrue(dip["usable_for_training"])  # sticky

    def test_played_turns_do_not_count_as_labelled_and_complete_is_kept_only_for_the_same_fights(self):
        r = jb.readiness(self.rows(700, 700, 700, 700, "played"), {}, {}, 10, 0)
        self.assertFalse(r["usable_for_training"])
        done = {"complete": True, "human_build": {"built": 10}}
        self.assertTrue(jb.readiness(self.rows(1, 1, 1, 1), done, {}, 10, 0)["complete"])
        self.assertFalse(jb.readiness(self.rows(1, 1, 1, 1), done, {}, 11, 0)["complete"])  # more fights since
        self.assertFalse(jb.readiness(self.rows(1, 1, 1, 1), {}, {}, 10, 0)["complete"])

    def test_the_manifest_carries_the_flags_and_what_holds_the_work(self):
        with tempfile.TemporaryDirectory() as d:
            store, loop, _runs = CompactTests().setup(d)
            with mock.patch.object(jb, "git_commit", return_value="abc"):
                CompactTests().compact(store, loop)
            man = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(man["usable_for_training"])
            self.assertFalse(man["complete"])
            self.assertEqual(set(man["human_build"]), {"planned", "built", "failed", "paused_by", "labelled_boss_by_act", "labelled_elite_by_act"})

    def test_finalize_marks_complete_only_when_every_fight_of_the_plan_has_a_result(self):
        with tempfile.TemporaryDirectory() as d:
            c = CompactTests()
            store, loop, runs = c.setup(d)
            for r in runs:
                c.fight(store, r)
            args = mock.Mock(store=str(store), loop=str(loop))
            import human_starts as hs

            with (mock.patch.object(hs, "read_exam_ids", return_value=frozenset({"nobody"})), mock.patch.object(jc, "REFEREE", Path(d) / "none"),
                  mock.patch.object(jb, "jc_exam_ids_path", return_value=Path(d) / "none.txt"), mock.patch.object(jb, "cmd_coverage", return_value=0)):
                self.assertEqual(jb.cmd_finalize(args), 0)
                man = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
                self.assertTrue(man["complete"])
                self.assertEqual(man["human_build"]["pending"], 0)
                jb.fight_path(store, f"human:{runs[0]}|f33").unlink()
                self.assertEqual(jb.cmd_finalize(args), 1)
                self.assertFalse(json.loads((store / "manifest.json").read_text(encoding="utf-8"))["complete"])


class SelfPlayTests(unittest.TestCase):
    def shard(self):
        """Two fights in one buffer shard: fight 0 (4 decisions, act 2 boss, a night start), fight 1 (capped), fight 2 (an excluded run)."""
        fights = [0] * 4 + [1] * 2 + [2] * 2
        n = len(fights)
        types, ids, lengths = [], [], []
        for r in range(n):
            t = [jc.T_GLOBAL, jc.T_HAND, jc.T_HAND, 10, jc.T_RELIC, jc.T_POTION, jc.T_POTION]  # action 0 and 1: hand cards, action 2: end turn
            types += t
            ids += [5, 1, 2, 0, 1, 0, 0]
            lengths.append(len(t))
        feats = np.zeros((len(types), 18), np.float32)
        for r in range(n):
            feats[r * 7, jc.F_HP] = 0.9 - 0.1 * r
            feats[r * 7 + 5, jc.F_MAGNITUDE] = 1.0
        visits = np.zeros((n, 32), np.float32)
        q = np.zeros((n, 32), np.float32)
        visits[:, 0], visits[:, 1], visits[:, 2] = 30, 10, 5
        q[:, 0], q[:, 1], q[:, 2] = 12.0, 8.0, -3.0
        played = np.array([0, 2, 0, 2, 0, 0, 0, 0], np.int16)
        z = {"types": np.array(types, np.int16), "ids": np.array(ids, np.int32), "owners": np.zeros(len(types), np.int16), "feats": feats,
             "lengths": np.array(lengths, np.int32), "visits": visits, "qvals": q, "legal": np.ones((n, 32), np.int8), "sims": np.full(n, 480, np.int32),
             "k": np.array([0, 1, 2, 3, 0, 1, 0, 1], np.int16), "q_played": np.full(n, 12.0, np.float32), "played": played, "fight": np.array(fights, np.int32)}
        metas = [{"start": "night/g9:AAA|f33", "j": 0, "cat": "act2-boss", "player": "c0250", "config": "x16", "seed": "AAA", "won": True, "end_hp": 20, "max_hp": 80,
                  "L0": 0.6, "capped": False, "potions_used": 1, "turns": 2, "decisions": 4, "encounter": 63, "value_ok": True, "potions_held": 1, "source": "night"},
                 {"start": "human:BBB|f8", "j": 1, "cat": "act1-elite", "player": "c0250", "config": "x16", "seed": "BBB", "won": False, "end_hp": 0, "max_hp": 70,
                  "L0": 0.0, "capped": True, "value_ok": True},
                 {"start": "human:CCC|f8", "j": 0, "cat": "act1-elite", "player": "c0250", "config": "x16", "seed": "CCC", "won": True, "end_hp": 9, "max_hp": 70,
                  "L0": 0.57, "capped": False, "value_ok": True, "potions_held": 0, "turns": 1, "decisions": 2, "encounter": 68}]
        return z, metas

    def rows(self, skip_runs=frozenset()):
        z, metas = self.shard()
        ctx = {"inv_card": {1: 472, 2: 131}, "inv_relic": {1: 36}, "cards": CARDS}
        skipped: Counter = Counter()
        out = list(jb.selfplay_rows(z, metas, set(skip_runs), set(), ctx, {63: "entomancer", 68: "byrdonis"}, skipped))
        return out, skipped

    def test_a_usable_fights_decisions_become_rows_with_the_search_as_played_and_the_outcome(self):
        out, skipped = self.rows({"CCC"})
        self.assertEqual(len(out), 4)
        self.assertEqual(skipped, {"capped or no value label": 1, "excluded run or start": 1})
        r0, t0 = out[0]
        self.assertEqual((r0["src"], r0["sub"], r0["kind"], r0["act"], r0["floor"], r0["run"], r0["player"]), ("selfplay", "night", "boss", 2, 33, "AAA", "c0250"))
        self.assertEqual(r0["pid"], "night/g9:AAA|f33~0~c0250#0")
        self.assertEqual(r0["split"], jc.split_of("AAA"))
        self.assertEqual([r["stage"] for r, _ in out], ["early", "early", "middle", "late"])  # thirds of the fight's four decisions
        self.assertEqual(r0["label"]["tier"], "played")
        self.assertEqual([c["a"] for c in r0["label"]["children"]], [0, 1, 2])
        self.assertEqual([c["kind"] for c in r0["label"]["children"]], ["card", "card", "end_turn"])
        self.assertEqual([c["card"] for c in r0["label"]["children"]], [472, 131, None])
        self.assertAlmostEqual(r0["label"]["root_value"], (30 * 12.0 + 10 * 8.0 + 5 * -3.0) / 45)
        self.assertEqual(r0["label"]["move"]["a"], 0)
        self.assertEqual(r0["teacher"], {"won": True, "end_hp": 20, "max_hp": 80, "potions_held": 1, "L0": 0.6, "decisions": 4, "turns": 2})
        self.assertEqual((r0["potions"], r0["relics"], r0["deck"]), (1, [36], {"131": 1, "472": 1}))
        self.assertAlmostEqual(r0["hp_frac"], 0.9)
        self.assertEqual(r0["hp"], 72)
        self.assertEqual(r0["enemy"], "entomancer")
        self.assertEqual((r0["w"], r0["human"]), (1.0, None))
        # the turn count goes up after an end-turn move was played
        self.assertEqual([r["turn"] for r, _ in out], [1, 1, 2, 2])
        self.assertEqual(t0["types"].tolist(), [1, 2, 2, 10, 14, 8, 8])
        json.dumps(r0)

    def test_capped_fights_and_fights_without_a_value_label_are_not_rows(self):
        self.assertIsNone(jb.buffer_labels({"capped": True, "value_ok": True}))
        self.assertIsNone(jb.buffer_labels({"capped": False, "value_ok": False}))
        lab = jb.buffer_labels({"won": True, "end_hp": 9, "max_hp": 70, "L0": 0.57, "capped": False, "value_ok": True, "start_potions": 2, "potions_used": 1, "decisions": 5, "turns": 2})
        self.assertEqual((lab["potions_held"], lab["won"]), (1, True))
        self.assertEqual(jb.buffer_labels({"won": False, "end_hp": 0, "max_hp": 70, "capped": False, "value_ok": True, "potions_held": 2})["potions_held"], 0)

    def test_start_sources(self):
        self.assertEqual([jb.start_source(s) for s in ("preloss:human:a|f3|j0|k4", "human:a|f3", "night/g9:A|f3")], ["preloss", "human", "night"])

    def test_the_command_writes_shards_a_ledger_and_takes_only_new_buffer_shards_next_time(self):
        z, metas = self.shard()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            loop, store = root / "loop", root / "store"
            (loop / "buffer" / "shards").mkdir(parents=True)
            (loop / "exam").mkdir()
            (loop / "exam" / "exclude_runs.json").write_text(json.dumps({"run_seeds": ["CCC"], "start_ids": []}), encoding="utf-8")
            np.savez_compressed(loop / "buffer" / "shards" / "s000001.npz", fights_meta=np.array(json.dumps(metas)), **z)
            (loop / "buffer" / "index.jsonl").write_text(json.dumps({"shard": "s000001.npz", "segment": 1, "t": "T1"}) + "\n", encoding="utf-8")
            jc.write_atomic(jb.work_dir(store) / "plan.json", {})
            with mock.patch.object(jc, "REFEREE", root / "none"):
                self.assertEqual(jb.cmd_selfplay(mock.Mock(store=str(store), loop=str(loop))), 0)
                c = jc.Corpus(store)
                self.assertEqual(len(list(c.iter_rows(src="selfplay"))), 4)
                self.assertEqual([p.stem for p in c.shards(src="selfplay")], ["selfplay-boss-act2-0000"])
                tokens_ = next(iter(c.iter_positions(src="selfplay")))
                self.assertEqual(tokens_.tokens["types"].tolist(), [1, 2, 2, 10, 14, 8, 8])
                # a second buffer shard is added; the first is in the ledger and is not read again (its file may be gone)
                (loop / "buffer" / "shards" / "s000001.npz").unlink()
                np.savez_compressed(loop / "buffer" / "shards" / "s000002.npz", fights_meta=np.array(json.dumps(metas)), **z)
                with (loop / "buffer" / "index.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"shard": "s000002.npz", "segment": 2, "t": "T2"}) + "\n")
                jb.cmd_selfplay(mock.Mock(store=str(store), loop=str(loop)))
                self.assertEqual(len(list(jc.Corpus(store).iter_rows(src="selfplay"))), 8)
                self.assertEqual(sorted(p.stem for p in jc.Corpus(store).shards(src="selfplay")), ["selfplay-boss-act2-0000", "selfplay-boss-act2-0001"])
            ledger = json.loads((jb.work_dir(store) / "selfplay-ledger.json").read_text(encoding="utf-8"))
            self.assertEqual(set(ledger["shards"]), {"s000001.npz", "s000002.npz"})


if __name__ == "__main__":
    unittest.main()
