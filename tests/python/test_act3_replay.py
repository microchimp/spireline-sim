"""The Act 3 boss replay (rl/act3_replay.py; lane 4 of the Act 3 week).

The pinned facts: the 1x arm is the champion's own search config and the 4x and 16x arms multiply only its boss multiplier (the
hallway and elite budgets are untouched); a position comes from the Act 3 boss fights of a run's decision record, first and second,
and from nothing else; the opening's arguments carry the deck (upgraded cards negative), enchantments, relics, potions and HP; the
sample is a fixed hash order that skips capped and repeated rows; the arithmetic of the report (Wilson interval, paired difference
with a bootstrap over positions, the sign test, capped games never won) comes out as written on tiny tables; the tasks come
dearest-first or cheapest-first or by position, nested by the positions' order, and a game with a row is never asked again; the
size plan fits its budget; and, where the native library is there, a rebuilt boss opening loads and has legal moves.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "rl", ROOT / "src", ROOT, ROOT / "referee"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import act3_replay as ar  # noqa: E402
import search_configs as sc  # noqa: E402

STAKES = "a3-v1@h1e4b16"


def fight(floor: int, node_type: int, enc: int, act_index: int, won: bool = True, hp: int = 60) -> dict:
    deck = [[30, 1, 0, 0], [131, 0, 0, 0], [131, 0, 0, 0], [472, 1, 15, 3], [472, 0, 0, 0], [10001, 0, 0, 0]]
    return {"floor": floor, "node_type": node_type, "encounter": enc, "won": won, "hp_before": hp, "hp_after": hp - 10,
            "state": {"act_index": act_index, "floor": floor, "hp": hp, "max_hp": 80, "gold": 120, "deck": deck, "relics": [36, 89],
                      "potions": [59], "max_potion_slots": 2, "gaps": {"counters": [[89, 7]]}}}


def record(seed: str, boss_fights: int) -> dict:
    fights = [fight(2, 1, 3, 0), fight(17, 6, 83, 0), fight(33, 6, 76, 1)]
    for i in range(boss_fights):
        fights.append(fight(48 + i, 6, (73, 78, 80)[i], 2, won=i == 0))
    return {"seed": seed, "fights": fights}


class ArmTests(unittest.TestCase):
    def test_one_x_is_the_champions_own_config(self):
        self.assertEqual(ar.arm_config(STAKES, None, 1), sc.for_agent("a3-v1", STAKES, None))

    def test_dearer_arms_multiply_the_boss_multiplier_only(self):
        c4, c16 = ar.arm_config(STAKES, None, 4), ar.arm_config(STAKES, None, 16)
        self.assertEqual(c4.stakes, (1, 4, 64))
        self.assertEqual(c16.stakes, (1, 4, 256))
        self.assertEqual(c16.name, "a3-v1@h1e4b256")
        boss = sc.Position(kind="play", n_moves=5, hp_frac=0.9, hard=True, act=3, fight="boss")
        target = sc.Position(kind="target", n_moves=5, hp_frac=0.9, hard=True, act=3, fight="boss")
        hall = sc.Position(kind="play", n_moves=5, hp_frac=0.9, hard=False, act=3, fight="hall")
        elite = sc.Position(kind="play", n_moves=5, hp_frac=0.9, hard=True, act=3, fight="elite")
        one = ar.arm_config(STAKES, None, 1)
        self.assertEqual([one.sims_for(boss), c4.sims_for(boss), c16.sims_for(boss)], [480, 1920, 7680])
        self.assertEqual([one.sims_for(target), c4.sims_for(target), c16.sims_for(target)], [1920, 7680, 30720])
        for pos in (hall, elite):  # the other fight kinds are not touched
            self.assertEqual(one.sims_for(pos), c4.sims_for(pos))
            self.assertEqual(one.sims_for(pos), c16.sims_for(pos))

    def test_the_safety_clock_grows_with_the_budget(self):
        pos = sc.Position(kind="play", n_moves=5, hp_frac=0.9, hard=True, act=3, fight="boss")
        t = [ar.arm_config(STAKES, None, m).search_kwargs(pos, env_prior=False)["time_budget_s"] for m in (1, 4, 16)]
        self.assertEqual(t[1], 4 * t[0])
        self.assertEqual(t[2], 16 * t[0])

    def test_an_agent_without_a_schedule_still_gets_arms(self):
        c = ar.arm_config(None, None, 4)
        self.assertEqual(c.stakes, (1, 1, 4))


class PositionTests(unittest.TestCase):
    def test_first_and_second_boss_of_act_three_and_nothing_else(self):
        rec = record("AAA", 2)
        a, b = ar.position_spec(rec, 1, "18"), ar.position_spec(rec, 2, "18")
        self.assertEqual((a["floor"], a["encounter_id"], a["ordinal"], a["run_won"]), (48, 73, 1, True))
        self.assertEqual((b["floor"], b["encounter_id"], b["ordinal"], b["run_won"]), (49, 78, 2, False))
        self.assertEqual(a["id"], "18:AAA|b1")
        self.assertIsNone(ar.position_spec(record("BBB", 1), 2, "18"))  # the run never met a second boss
        self.assertIsNone(ar.position_spec(record("CCC", 0), 1, "18"))  # the Act 1 and Act 2 bosses are not Act 3's

    def test_a_record_without_the_state_is_refused(self):
        rec = record("DDD", 1)
        rec["fights"][-1]["state"] = {}
        self.assertIsNone(ar.position_spec(rec, 1, "18"))

    def test_the_spec_keeps_the_run_state(self):
        s = ar.position_spec(record("EEE", 1), 1, "17")
        self.assertEqual((s["hp"], s["max_hp"], s["gold"], s["relics"], s["potions"], s["counters"]), (60, 80, 120, [36, 89], [59], [[89, 7]]))
        self.assertEqual(len(s["deck"]), 6)

    def test_the_arena_arguments(self):
        a = ar.reset_args(ar.position_spec(record("FFF", 1), 1, "17"))
        self.assertEqual(a["deck_ids"], [-30, 131, 131, -472, 472, 10001])  # upgraded cards negative
        self.assertEqual(a["enchantments"], [(0, 0), (0, 0), (0, 0), (15, 3), (0, 0), (0, 0)])  # parallel to the deck
        self.assertEqual((a["encounter_id"], a["player_hp"], a["player_max_hp"], a["ascension"], a["total_floor"]), (73, 60, 80, 10, 48))
        self.assertEqual(a["potion_ids"], [59])
        self.assertGreaterEqual(a["max_potion_slots"], 1)
        spec = ar.position_spec(record("FFG", 1), 1, "17")
        spec["potions"], spec["potion_slots"] = [1, 2, 3], 2
        self.assertEqual(ar.reset_args(spec)["max_potion_slots"], 3)  # never narrower than the potions held

    def test_counters_go_to_the_matching_relic(self):
        doc = {"relics": [{"id": 36, "counter": 0}, {"id": 89, "counter": 0}]}
        self.assertEqual(ar.apply_counters(doc, [[89, 7], [5, 1]]), 1)
        self.assertEqual(doc["relics"], [{"id": 36, "counter": 0}, {"id": 89, "counter": 7}])


class SampleTests(unittest.TestCase):
    def make_store(self, d: Path, rows: list[dict]) -> Path:
        store = d / "17"
        store.mkdir()
        (store / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return store

    def test_candidates_by_ordinal_in_a_fixed_hash_order(self):
        a3 = lambda n: [{"act": 2, "floor": 48 + i} for i in range(n)]  # noqa: E731
        rows = [{"seed": "R1", "bosses": a3(1)}, {"seed": "R2", "bosses": a3(2)}, {"seed": "R3", "bosses": []},
                {"seed": "R4", "bosses": a3(1), "capped": True}, {"seed": "R1", "bosses": a3(1)}, {"seed": "R5", "bosses": a3(2)}]
        with tempfile.TemporaryDirectory() as t:
            store = self.make_store(Path(t), rows)
            c = ar.candidates([store])
            again = ar.candidates([store])
        self.assertEqual(c, again)
        self.assertEqual(sorted(x[3] for x in c[1]), ["R1", "R2", "R5"])  # a capped row, a run without an Act 3 boss and a repeat are out
        self.assertEqual(sorted(x[3] for x in c[2]), ["R2", "R5"])
        self.assertEqual([x[0] for x in c[1]], sorted(x[0] for x in c[1]))

    def test_the_salt_moves_the_order(self):
        keys = {ar.order_key(s, 1, "17", "R1") for s in (1, 2, 3)}
        self.assertEqual(len(keys), 3)
        self.assertEqual(ar.order_key(1, 1, "17", "R1"), ar.order_key(1, 1, "17", "R1"))

    def test_boss_names_are_the_tables(self):
        self.assertEqual(ar.boss_name(73), "AEONGLASS_BOSS")
        self.assertEqual(ar.boss_name(999), "enc999")


def game(arm: str, pos: str, won: bool, j: int = 0, **kw) -> dict:
    return {"_task": [arm, pos, j], "arm": arm, "pos": pos, "j": j, "won": won, "seconds": kw.pop("seconds", 10.0), "sims": 100,
            "end_hp": 20 if won else 0, "turns": 5, "decisions": 20, "boss": kw.pop("boss", "A"), "ordinal": kw.pop("ordinal", 1),
            "run_won": kw.pop("run_won", False), **kw}


class ArithmeticTests(unittest.TestCase):
    def test_wilson(self):
        lo, hi = ar.wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2775, places=3)
        lo, hi = ar.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertEqual(ar.wilson(0, 0), (0.0, 1.0))

    def test_sign_test(self):
        self.assertEqual(ar.sign_test_p(0, 0), 1.0)
        self.assertAlmostEqual(ar.sign_test_p(8, 2), 0.109375, places=6)
        self.assertAlmostEqual(ar.sign_test_p(5, 5), 1.0, places=6)

    def test_a_capped_game_is_never_a_kill(self):
        self.assertTrue(ar.won_of(game("1x", "p", True)))
        self.assertFalse(ar.won_of({**game("1x", "p", True), "capped": True}))
        s = ar.arm_summary([game("1x", "a", True), {**game("1x", "b", True), "capped": True}, game("1x", "c", False)])
        self.assertEqual((s["n"], s["kills"], s["capped"]), (3, 1, 1))

    def test_a_position_the_search_calls_lost_at_the_start(self):
        rows = [game("1x", "a", True, root_p_death=0.97), game("1x", "b", False, root_p_death=1.0), game("1x", "c", True, root_p_death=0.4),
                game("1x", "d", False)]
        s = ar.arm_summary(rows)
        self.assertEqual((s["hopeless"], s["hopeless_kills"], s["kills"]), (2, 1, 2))
        self.assertIn("kills among them 1", ar.report_text({"positions": 4, "arms": {"1x": s}, "paired": [], "splits": {}}))

    def test_paired_difference_and_discordant_counts(self):
        rows = []
        # ten positions: one at 1x only killed; four killed only at 4x; both kill two; neither kills three
        spec = [(True, False)] + [(False, True)] * 4 + [(True, True)] * 2 + [(False, False)] * 3
        for i, (x, y) in enumerate(spec):
            rows += [game("1x", f"p{i}", x), game("4x", f"p{i}", y)]
        rows.append(game("1x", "lonely", True))  # no 4x row: not in the pair
        p = ar.paired(ar.index_rows(rows), "1x", "4x", boot=300)
        self.assertEqual((p["n"], p["up"], p["down"], p["kills_a"], p["kills_b"]), (10, 4, 1, 3, 6))
        self.assertAlmostEqual(p["diff"], 0.3)
        self.assertLessEqual(p["lo"], 0.3)
        self.assertGreaterEqual(p["hi"], 0.3)
        self.assertAlmostEqual(p["p_sign"], ar.sign_test_p(4, 1))
        self.assertEqual(p["same_play"], 0)  # no moves hashes in the rows: nothing is called the same play
        a, b = game("1x", "q", True, moves_sha="x"), game("4x", "q", True, moves_sha="x")
        self.assertEqual(ar.paired(ar.index_rows([a, b, game("1x", "r", True, moves_sha="y"), game("4x", "r", True, moves_sha="z")]),
                                   "1x", "4x", boot=50)["same_play"], 1)

    def test_a_later_row_of_the_same_game_replaces_the_earlier(self):
        idx = ar.index_rows([game("1x", "p", False), game("1x", "p", True)])
        self.assertTrue(ar.won_of(idx["1x"][("p", 0)]))

    def test_estimate_and_size_plan(self):
        self.assertAlmostEqual(ar.estimate({1: 36.0, 4: 90.0}, {1: 100, 4: 40}, 12)["core_hours"], 2.0)
        plan = ar.size_plan({1: 60.0, 4: 150.0, 16: 500.0}, 300, 90.0, 12, 8.0)
        n = plan["n"]
        self.assertLessEqual(plan["core_hours"], 90.0 + 1e-6)
        self.assertLessEqual(plan["wall_hours"], 8.0 + 1e-6)
        self.assertTrue(n[16] <= n[4] <= n[1] <= 300)
        cheap = ar.size_plan({1: 1.0, 4: 2.0, 16: 8.0}, 50, 90.0, 12, 8.0)
        self.assertEqual(cheap["n"], {1: 50, 4: 50, 16: 50})  # all fit: every budget plays every position


class TaskTests(unittest.TestCase):
    def positions(self, n: int = 6) -> dict:
        return {f"p{i}": {"id": f"p{i}", "order": i, "ordinal": 1} for i in range(n)}

    def test_counts_are_nested_by_order(self):
        t = ar.tasks_for(self.positions(), {1: 6, 4: 4, 16: 2}, 1, set(), "cheapest")
        by = {arm: sorted(pid for a, pid, _ in t if a == arm) for arm in ("1x", "4x", "16x")}
        self.assertEqual(by["16x"], ["p0", "p1"])
        self.assertEqual(by["4x"], ["p0", "p1", "p2", "p3"])
        self.assertEqual(len(by["1x"]), 6)

    def test_the_three_orders(self):
        pos, counts = self.positions(3), {1: 3, 4: 3, 16: 3}
        dear = ar.tasks_for(pos, counts, 1, set(), "dearest")
        cheap = ar.tasks_for(pos, counts, 1, set(), "cheapest")
        byp = ar.tasks_for(pos, counts, 1, set(), "position")
        self.assertEqual([a for a, _, _ in dear[:3]], ["16x"] * 3)
        self.assertEqual([a for a, _, _ in cheap[:3]], ["1x"] * 3)
        self.assertEqual(byp[:3], [("16x", "p0", 0), ("4x", "p0", 0), ("1x", "p0", 0)])
        with self.assertRaises(ValueError):
            ar.tasks_for(pos, counts, 1, set(), "random")

    def test_a_game_with_a_row_is_not_asked_again_and_draws_multiply(self):
        done = ar.done_keys([game("1x", "p0", True), game("16x", "p1", False, j=1)])
        t = ar.tasks_for(self.positions(2), {1: 2, 16: 2}, 2, done, "cheapest")
        self.assertNotIn(("1x", "p0", 0), t)
        self.assertNotIn(("16x", "p1", 1), t)
        self.assertEqual(len(t), 2 * 2 * 2 - 2)

    def test_counts_text(self):
        self.assertEqual(ar.parse_counts("1:60,4:60,16:20", [1, 4, 16], 40), {1: 40, 4: 40, 16: 20})
        self.assertEqual(ar.parse_counts("", [1, 4], 7), {1: 7, 4: 7})
        self.assertEqual(ar.parse_counts("1:5", [1, 4], 7), {1: 5, 4: 0})

    def test_the_capped_row_is_a_lost_game_with_its_task(self):
        r = ar.capped_row(("16x", "p3", 0), "deadline (killed)", 99.0, {"k": 12, "sims": 5})
        self.assertEqual((r["_task"], r["capped"], r["won"], r["decisions"]), (["16x", "p3", 0], True, False, 12))
        self.assertEqual(ar.done_keys([r]), {("16x", "p3", 0)})
        self.assertGreater(ar.hard_deadline(("16x", "p3", 0)), ar.hard_deadline(("1x", "p3", 0)))


class ReportTests(unittest.TestCase):
    def test_the_table_reads_the_files(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            pos = [{"id": f"p{i}", "order": i, "ordinal": 1 + (i % 2), "boss": "A" if i < 3 else "B", "run_won": i == 0} for i in range(6)]
            (out / ar.FILES["positions"]).write_text("".join(json.dumps(p) + "\n" for p in pos), encoding="utf-8")
            rows = []
            for i, p in enumerate(pos):
                meta = {"boss": p["boss"], "ordinal": p["ordinal"], "run_won": p["run_won"]}
                rows += [game("1x", p["id"], i == 0, seconds=10, **meta), game("4x", p["id"], i < 2, seconds=25, **meta),
                         game("16x", p["id"], i < 3, seconds=80, **meta)]
            (out / ar.FILES["games"]).write_text("".join(json.dumps(r) + "\n" for r in rows) + "{cut line\n", encoding="utf-8")
            d = ar.report_data(out)
            text = ar.report_text(d)
        self.assertEqual(d["positions"], 6)
        self.assertEqual([d["arms"][a]["kills"] for a in ("1x", "4x", "16x")], [1, 2, 3])
        self.assertEqual(d["arms"]["16x"]["wall_mean"], 80.0)
        self.assertEqual(len(d["paired"]), 3)
        self.assertEqual(d["splits"]["boss"]["A"]["arms"]["16x"]["kills"], 3)
        self.assertEqual(d["splits"]["run_result"]["the run beat it"]["arms"]["1x"]["n"], 1)
        self.assertIn("4x minus 1x", text)
        self.assertIn("By boss", text)


try:
    import leakprobes

    leakprobes.native_env()
    from sts2_gym import native  # noqa: E402

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    HAVE_NATIVE = False


@unittest.skipUnless(HAVE_NATIVE, "needs the native emulator library")
class OpeningTests(unittest.TestCase):
    def test_a_rebuilt_boss_opening_loads_and_has_moves(self):
        import combat_mcts
        import fight_gym as fg

        spec = ar.position_spec(record("NAT", 1), 1, "17")
        spec["deck"] = [[30, 0, 0, 0]] + [[131, 0, 0, 0]] * 5 + [[472, 0, 0, 0]] * 5 + [[10001, 0, 0, 0]]
        spec["relics"], spec["potions"], spec["counters"] = [36], [], []
        spec["encounter_id"], spec["hp"], spec["max_hp"] = 80, 60, 80
        doc, why = ar.build_doc(spec, native)
        self.assertEqual(why, "")
        self.assertEqual(doc["encounter_id"], 80)
        self.assertEqual((doc["player"]["hp"], doc["player"]["max_hp"]), (60, 80))
        self.assertTrue(doc["enemies"])
        world = fg._fresh_world(doc, spec["id"], 0, ar.NS)  # noqa: SLF001
        try:
            self.assertFalse(world.done)
            self.assertTrue(world.legal())
            self.assertEqual(native.encounter_id(world.handle), 80)
        finally:
            world.close()


if __name__ == "__main__":
    unittest.main()
