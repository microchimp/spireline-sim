"""rl/run_fields.py: the per-run record fields (D32 addendum item 2) in the emulator arms' rows, the real block's end events,
the tables, and the D24 damage guard that reads them. Constructed data only: no game, no native library."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "referee", ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import gen0  # noqa: E402
import lockstep  # noqa: E402
import run_fields  # noqa: E402

from tools.lab import promotion_guard as pg  # noqa: E402


def boss_msg(act: int, hps: list[int], kind: str = "combat") -> dict:
    return {"kind": kind, "actions": [["end"]], "obs": {"act": act, "floor": 17, "combat": {
        "round": 1, "enemies": [{"name": f"E{i}", "hp": hp, "max_hp": 100} for i, hp in enumerate(hps)]}}}


def boss(row: dict) -> dict:
    return {k: row[k] for k in run_fields.BOSS_KEYS}


def lockstep_run(msgs: list[dict], outcome: dict) -> dict:
    """A Lockstep with no emulator side (the fallback plays) fed a scripted stream; its record fields."""
    ls = lockstep.Lockstep(None, lambda m: 0, "seed")
    for m in msgs:
        ls.decide(m)
    ls.finish(outcome)
    return boss(ls.fields.row())


MAP_BOSS = {"kind": "map", "actions": [[3, 15, "Boss"]], "obs": {"act": 0, "floor": 16}}
MAP_MONSTER = {"kind": "map", "actions": [[3, 5, "Monster"]], "obs": {"act": 0, "floor": 4}}
AFTER_FIGHT = {"kind": "card_reward", "actions": ["skip"], "obs": {"act": 0, "floor": 17}}


class BossDamageTests(unittest.TestCase):
    def test_a_fake_boss_fight_gives_the_fraction_of_its_starting_hp(self):
        f = run_fields.RunFields()
        f.boss_enemies(0, [(0, 100, 100)])
        f.boss_enemies(0, [(0, 70, 100)])
        f.boss_enemies(0, [(0, 40, 100)])
        f.boss_enemies(0, [(0, 55, 100)])  # healed back up: the lowest HP it reached counts
        self.assertEqual(boss(f.row()), {"boss_damage": 0.6, "boss_damage_acts": [[0, 0.6]]})  # the run stopped in the fight

    def test_a_won_fight_is_full_damage_and_an_unreached_boss_is_zero_not_absent(self):
        f = run_fields.RunFields()
        f.boss_enemies(0, [(0, 100, 100)])
        f.boss_enemies(0, [(0, 20, 100)])
        f.boss_over(won=True)
        self.assertEqual(f.row()["boss_damage"], 1.0)
        none = boss(run_fields.RunFields().row())
        self.assertEqual(none, {"boss_damage": 0.0, "boss_damage_acts": []})

    def test_later_bosses_are_listed_and_a_summoned_minion_changes_nothing(self):
        f = run_fields.RunFields()
        f.boss_enemies(0, [("a", 50, 50), ("b", 50, 50)])
        f.boss_enemies(0, [("a", 0, 50), ("b", 25, 50), ("minion", 99, 99)])  # one of the pair dead, half of the other
        f.boss_over(won=False)
        f.boss_enemies(1, [(0, 200, 200)])
        f.boss_over(won=True)
        self.assertEqual(boss(f.row()), {"boss_damage": 0.75, "boss_damage_acts": [[0, 0.75], [1, 1.0]]})

    def test_the_readers_of_the_two_sides(self):
        obs = [0] * 40
        for slot, (hp, mx) in enumerate([(30, 40), (0, 25), (0, 0)]):  # offset 5, slot size 4: hp, max_hp
            obs[5 + slot * 4], obs[6 + slot * 4] = hp, mx
        self.assertEqual(run_fields.obs_enemies(obs, 5, 4, 3), [(0, 30, 40), (1, 0, 25)])
        c = {"enemies": [{"name": "X", "hp": 5, "max_hp": 9}, {"name": "X", "hp": 0, "max_hp": 9}, "junk", {"name": "Y", "hp": None}]}
        self.assertEqual(run_fields.real_enemies(c), [(("X", 0), 5, 9), (("X", 1), 0, 9)])

    def test_the_real_side_records_the_boss_fight_of_a_lockstep(self):
        lost = lockstep_run([MAP_BOSS, boss_msg(0, [100]), boss_msg(0, [60])], {"class": "death", "end": "death", "floor": 17})
        self.assertEqual(lost, {"boss_damage": 0.4, "boss_damage_acts": [[0, 0.4]]})
        won = lockstep_run([MAP_BOSS, boss_msg(0, [100]), boss_msg(0, [30], "card_reward_combat"), AFTER_FIGHT],
                           {"class": "truncated", "end": None, "floor": 17})
        self.assertEqual(won["boss_damage"], 1.0)  # the next screen outside the fight: it was won
        final = lockstep_run([MAP_BOSS, boss_msg(2, [100]), boss_msg(2, [10])], {"class": "win", "end": "win", "floor": 51})
        self.assertEqual(final, {"boss_damage": 0.0, "boss_damage_acts": [[2, 1.0]]})  # the Act 3 boss: act 1's field stays 0
        normal = lockstep_run([MAP_MONSTER, boss_msg(0, [100]), boss_msg(0, [1])], {"class": "death", "end": "death", "floor": 4})
        self.assertEqual(normal, {"boss_damage": 0.0, "boss_damage_acts": []})  # a fight that is not the boss's is not recorded

    def test_the_rows_of_both_arms_and_the_real_block_carry_the_fields(self):
        with mock.patch.object(gen0, "play_emu_run", side_effect=RuntimeError("boom")):
            crash = gen0._run_safe("S", 0, 8, 10)
        self.assertEqual((crash["crash"], crash["boss_damage"]), (True, 0.0))  # a crash row keeps the guard available
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "runs.jsonl"
            end = {"ev": "end", "seed": "a", "end": "death", "floor": 17, "s": 3.0, "stats": {"decisions": 9},
                   **lockstep_run([MAP_BOSS, boss_msg(0, [100]), boss_msg(0, [50])], {"class": "death"}), "cards": [[2, "BASH"]]}
            old = {"ev": "end", "seed": "b", "end": "death", "floor": 7, "s": 3.0, "stats": {"decisions": 9}}
            p.write_text("\n".join(json.dumps(e) for e in (end, old)), encoding="utf-8")
            rows = gen0.real_runs(p)
        self.assertEqual((rows[0]["boss_damage"], rows[0]["boss_damage_acts"]), (0.5, [[0, 0.5]]))
        self.assertNotIn("boss_damage", rows[1])  # an older row: unknown, not 0
        self.assertEqual(rows[0]["cards"], [[2, "BASH"]])
        self.assertNotIn("cards", rows[1])

    def test_the_guard_fires_on_rows_that_carry_the_field_and_stays_inert_without(self):
        seeds = [f"S{i}" for i in range(300)]

        def arm(damage, with_field=True):
            rows = [{"seed": s, "floor": 17, "win": False, "crash": False, **({"boss_damage": damage(i)} if with_field else {})}
                    for i, s in enumerate(seeds)]
            return rows

        base = arm(lambda i: 0.5 + 0.3 * ((i % 5) - 2) / 2)
        worse = arm(lambda i: 0.2 + 0.3 * ((i % 5) - 2) / 2)
        _kc, _kb, dc, db = pg.from_rows(worse, base)
        self.assertIsNotNone(dc)
        self.assertTrue(pg.damage_guard(dc, db, boot=500)["fires"])
        _kc, _kb, dc, db = pg.from_rows(base, base)
        self.assertFalse(pg.damage_guard(dc, db, boot=500)["fires"])
        old = pg.from_rows(arm(lambda i: 0.0, with_field=False), arm(lambda i: 0.0, with_field=False))
        self.assertEqual(old[2:], (None, None))
        self.assertEqual(pg.damage_guard(*old[2:])["available"], False)  # inert: not fired, not zero


def reward(floor: int, labels: list, pick: int) -> dict:
    return {"kind": "card_reward", "actions": labels, "pick": pick, "obs": {"act": 0, "floor": floor}}


class CardsPerFloorTests(unittest.TestCase):
    def test_the_emulator_side_records_the_macro_decisions_by_floor(self):
        from types import SimpleNamespace as NS

        f = run_fields.RunFields()
        slugs = {7: "BASH", 9: "ANGER"}
        f.emu_macro(1, NS(kind="map.node", node=(0, 1, 1)), slugs)
        f.emu_macro(2, NS(kind="card.take", card=NS(card_id=7, upgraded=False)), slugs)
        f.emu_macro(3, NS(kind="map.node", node=(0, 2, 3)), slugs)
        f.emu_macro(3, NS(kind="rest.smith"), slugs)
        f.emu_macro(3, NS(kind="rest.leave"), slugs)  # leaving is not a choice recorded
        f.emu_macro(4, NS(kind="card.skip"), slugs)
        f.emu_macro(5, NS(kind="card.take", card=NS(card_id=9, upgraded=True)), slugs)
        f.emu_macro(5, NS(kind="card.reroll"), slugs)
        r = f.row()
        self.assertEqual(r["cards"], [[2, "BASH"], [4, "skip"], [5, "ANGER+"]])
        self.assertEqual((r["nodes"], r["rests"]), ("MR", [[3, "smith"]]))
        self.assertEqual(run_fields.RunFields().row()["cards"], [])  # a run that decided none: an empty list, not absent

    def test_the_real_side_records_the_same_from_the_lockstep(self):
        ls = lockstep.Lockstep(None, lambda m: m.get("pick", 0), "seed")
        for m in (MAP_MONSTER, reward(5, [["BASH", 0], ["ANGER", 0], "skip"], 1), reward(6, [["BASH", 0], "skip"], 1),
                  reward(7, [["ANGER", 1], "alt:relic"], 1), {"kind": "rest", "actions": ["HEAL", "SMITH"], "pick": 1, "obs": {"floor": 8}},
                  {"kind": "map", "actions": [[1, 9, "RestSite"], [2, 9, "Unknown"]], "pick": 1, "obs": {"floor": 8}}):
            ls.decide(m)
        ls.finish({"class": "death"})
        r = ls.fields.row()
        self.assertEqual(r["cards"], [[5, "ANGER"], [6, "skip"]])  # the alternative at floor 7 is neither
        self.assertEqual((r["nodes"], r["rests"]), ("M?", [[8, "smith"]]))


def row(seed: str, **kw) -> dict:
    return {"seed": seed, "win": False, "floor": 17, "decisions": 10, "wall": 1.0, "crash": False, **kw}


class TableTests(unittest.TestCase):
    ROWS = [row("a", boss_damage=0.5, cards=[[2, "BASH"], [3, "skip"], [4, "ANGER"]]), row("b", boss_damage=1.0, cards=[[2, "skip"]]),
            row("c")]  # an older row: no fields

    def test_the_aggregates_count_only_the_rows_that_carry_the_fields(self):
        a = run_fields.aggregate(self.ROWS)
        self.assertEqual(a["boss_damage"], {"n": 2, "mean": 0.75})
        c = a["cards"]
        self.assertEqual((c["runs"], c["taken"], c["skipped"], c["skip_rate"], c["taken_per_run"]), (2, 2, 2, 0.5, 1.0))
        self.assertEqual((c["taken_by_floor"], c["skipped_by_floor"]), ({"2": 1, "4": 1}, {"2": 1, "3": 1}))
        self.assertEqual(run_fields.aggregate([row("c"), row("d")]), {})  # none carry them: unknown, no zeros
        self.assertEqual(run_fields.aggregate([row("e", boss_damage=0.0)]), {"boss_damage": {"n": 1, "mean": 0.0}})

    def test_summary_compare_and_promotion_tables_carry_them_and_old_rows_load(self):
        import gen_loop

        self.assertEqual(gen0.engine_block(self.ROWS)["run_fields"]["boss_damage"]["mean"], 0.75)
        self.assertEqual(gen_loop.arm_summary(self.ROWS)["run_fields"]["cards"]["skip_rate"], 0.5)  # promotion.json's cand/base summary
        old = [row("c"), row("d")]
        self.assertNotIn("run_fields", gen0.engine_block(old))
        self.assertNotIn("run_fields", gen_loop.arm_summary(old))
        cmp = gen0.compare(self.ROWS, [row("a", boss_damage=0.25), row("c")])
        self.assertEqual(cmp["emu"]["run_fields"]["boss_damage"], {"n": 1, "mean": 0.5})  # the shared seeds only
        self.assertEqual(cmp["real"]["run_fields"]["boss_damage"], {"n": 1, "mean": 0.25})

    def test_readiness_agent_section_shows_the_latest_block_and_n_a_for_old_rows(self):
        from tools.fidelity import readiness as rd

        def promo(root: Path, gen: int, written: str, fields: bool) -> None:
            arm = {"summary": {"runs": 2000, **({"run_fields": gen0.engine_block(self.ROWS)["run_fields"]} if fields else {})}}
            p = root / "agent" / f"gen{gen}" / "candidate" / "promotion.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"written": written, "seed_block": f"E{gen}", "decision": {"promote": False, "checks": {}}, "cand": arm, "base": arm}), encoding="utf-8")

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(rd.agent_score(root), (None, None))  # no agent folder: no section, no note
            promo(root, 1, "2026-10-20T00:00:00Z", fields=True)
            promo(root, 2, "2026-10-24T00:00:00Z", fields=False)  # the newest block predates the fields
            sec, note = rd.agent_score(root)
            self.assertEqual((sec["block"], sec["arms"]["cand"]["boss_damage_mean"], sec["arms"]["cand"]["skip_rate"]), ("E2", None, None))
            self.assertIn("predate", note["note"])
            promo(root, 3, "2026-10-25T00:00:00Z", fields=True)
            sec, note = rd.agent_score(root)
            self.assertIsNone(note)
            self.assertEqual(sec["decision"], "rejected")  # promotion.json holds the decide dict, shown as promoted or rejected
            self.assertEqual((sec["block"], sec["arms"]["cand"]["boss_damage_mean"], sec["arms"]["base"]["skip_rate"]), ("E3", 0.75, 0.5))
            self.assertEqual(sec["arms"]["cand"]["cards_taken_by_floor"], {"2": 1, "4": 1})
            self.assertEqual(rd.build(root)["agent"]["block"], "E3")
            self.assertIn("agent latest block", rd.render(rd.build(root)))


if __name__ == "__main__":
    unittest.main()


class FieldHookFenceTests(unittest.TestCase):
    """A fault in a record-field hook never ends the scored run: it is recorded once and the hooks go quiet."""

    def test_a_raising_hook_is_recorded_once_and_the_run_goes_on(self):
        import lockstep

        class Raising:
            in_boss_fight = False
            calls = 0

            def real_macro(self, *a, **k):
                self.calls += 1
                raise TypeError("int() argument must be a string")

        ls = lockstep.Lockstep.__new__(lockstep.Lockstep)
        ls.fields, ls.fields_error = Raising(), None
        ls._fields("real_macro", "card_reward", ["take", "STRIKE"], None)
        ls._fields("real_macro", "card_reward", ["take", "STRIKE"], None)
        self.assertEqual(ls.fields_error, "real_macro: TypeError: int() argument must be a string")
        self.assertEqual(ls.fields.calls, 1)

    def test_a_callable_argument_is_evaluated_inside_the_fence(self):
        import lockstep

        class Quiet:
            seen = None

            def boss_enemies(self, act, enemies):
                self.seen = (act, enemies)

        ls = lockstep.Lockstep.__new__(lockstep.Lockstep)
        ls.fields, ls.fields_error = Quiet(), None
        ls._fields("boss_enemies", 0, lambda: [("a", 10, 10)])
        self.assertEqual(ls.fields.seen, (0, [("a", 10, 10)]))
        ls._fields("boss_enemies", 0, lambda: 1 / 0)
        self.assertTrue(ls.fields_error.startswith("boss_enemies: ZeroDivisionError"))
