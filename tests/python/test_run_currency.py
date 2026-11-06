"""The run currency (training design section 3, brief 3a; rl/run_currency.py): a fight's end state priced by the frozen run head.

Pinned: the currency's arithmetic on a smooth stub head (a death is the fight's floor and -100, a full-HP win holding every starting potion is
the start's max HP, T rises with HP and with each potion held, a potion costs the head's marginal and not 10, the end-of-combat heal, the
judge leaf's potion count, the clip, the memo, the fall back to HP when the head gives the fight no scale); the contexts a start record
stands in (its document, its run view, the unknown boss and region); the mean map table that fills a view without a map; the head's batches
(mixed regions and bosses agree with one-by-one); the audit and each way it stops; the config grammar (`a3-v1-r`, `-r` on any base, refusing
to search without the fight's currency); the search's terminal function (a won playout, a death, a judge leaf, parity with the old units
when the function says HP plus 10 a potion); and the games that use it (the fight gym's row, a whole run's row with the boss door's hand).

No game files are read; the emulator's library plays the real fights.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import combat_mcts  # noqa: E402
import judge_client  # noqa: E402
import live_search  # noqa: E402
import run_currency as rc  # noqa: E402
import search_configs as sc  # noqa: E402
from run_train import FEATURE_VERSION, features, models, vocab  # noqa: E402
from run_train.features import CardRef, MapNode, RunView  # noqa: E402

from sts2_gym import native  # noqa: E402
from test_live_search import BLOCK_POTION, DEFEND, NIBBITS_WEAK, STRIKE, Fight  # noqa: E402

# The views here are built from the same `run_train` modules the currency imports (test_run_train_features builds its own from `rl.run_train`,
# a second copy of the classes, and two copies never compare equal).
BASH = 30
BURNING_BLOOD = vocab.relic_id_by_name("BurningBlood")
FIRE_POTION = next(p for p, name in vocab.names.potions().items() if name == "FirePotion")
SMALL_MAP = (
    MapNode(3, 0, vocab.NODE_ANCIENT, ((2, 1), (4, 1))),
    MapNode(2, 1, vocab.NODE_MONSTER, ((2, 2), (3, 2))),
    MapNode(4, 1, vocab.NODE_UNKNOWN, ((3, 2),)),
    MapNode(2, 2, vocab.NODE_ELITE, ((3, 3),)),
    MapNode(3, 2, vocab.NODE_REST, ((3, 3),)),
    MapNode(3, 3, vocab.NODE_SHOP, ((3, 4),)),
    MapNode(3, 4, vocab.NODE_BOSS, ()),
)


def make_view(**overrides):
    fields = {"ascension": 8, "act_index": 0, "act_region": vocab.REGION_OVERGROWTH, "floor": 3, "hp": 50, "max_hp": 80, "gold": 120,
              "deck": (CardRef(STRIKE, False), CardRef(DEFEND, False), CardRef(BASH, False)), "relics": (BURNING_BLOOD,),
              "potions": (BLOCK_POTION,), "max_potion_slots": 2, "map_nodes": SMALL_MAP, "position": (3, 0), "boss": 83}
    fields.update(overrides)
    return RunView(**fields)

POOLS = vocab.encounter_pools()
OVERGROWTH, UNDERDOCKS, HIVE = vocab.REGION_OVERGROWTH, vocab.REGION_UNDERDOCKS, vocab.REGION_HIVE
OVERGROWTH_ONLY = next(e for e in POOLS[OVERGROWTH]["weak"] if not any(e in ids for ids in POOLS[UNDERDOCKS].values()))


class StubHead:
    """A smooth, monotone head: the final floor is the view's floor + BASE + HP_RATE x hp + a potion's worth (by kind) for each held."""

    sha256 = "stub0000"

    def __init__(self, base=20.0, hp_rate=0.05, potion=0.8, kinds=None, gold_rate=0.0):
        self.base, self.hp_rate, self.potion, self.kinds, self.gold_rate = base, hp_rate, potion, kinds or {}, gold_rate
        self.calls = self.rows = 0

    def value(self, view):
        return (view.floor + self.base + self.hp_rate * view.hp + self.gold_rate * view.gold
                + sum(self.kinds.get(p, self.potion) for p in view.potions))

    def floors(self, ctxs, table=None):
        self.calls += 1
        self.rows += len(ctxs)
        return np.array([self.value(c.view) for c in ctxs])


def view(**kw):
    kw.setdefault("floor", 3)
    kw.setdefault("hp", 50)
    kw.setdefault("max_hp", 80)
    kw.setdefault("relics", ())
    kw.setdefault("potions", (BLOCK_POTION, FIRE_POTION))
    kw.setdefault("max_potion_slots", 3)
    return make_view(**kw)


def currency(head=None, **kw):
    return rc.FightCurrency(head or StubHead(), [rc.Ctx(view(**kw), rc.NODE_MONSTER)])


class CurrencyMathTests(unittest.TestCase):
    def test_a_full_hp_win_holding_every_potion_is_the_starts_max_hp_and_u_is_one(self):
        c = currency()
        self.assertEqual(c.u(80, 80, c.potions0), 1.0)
        self.assertEqual(c(80, 80, c.potions0), 80.0)
        self.assertEqual(c.max_hp0, 80)

    def test_the_spread_is_todays_death_to_max_hp(self):
        c = currency()
        self.assertEqual(rc.DEATH_VALUE, live_search.DEATH_VALUE)
        self.assertEqual(c(80, 80, c.potions0) - rc.DEATH_VALUE, c.max_hp0 - rc.DEATH_VALUE)
        # a win the head values no more than the fight's own floor is a death's value, and no less
        zero = currency(StubHead(base=0.0, hp_rate=0.0, potion=1.0))
        self.assertEqual(zero.u(10, 80, ()), 0.0)
        self.assertEqual(zero(10, 80, ()), rc.DEATH_VALUE)

    def test_the_lost_floor_is_the_fights_own_floor(self):
        c = currency(floor=17)
        self.assertEqual(c.t_lost, 17.0)
        self.assertEqual(c.floor, 17.0)
        self.assertAlmostEqual(c.span, c.t_full - 17.0)
        self.assertEqual(rc.FightCurrency(StubHead(), [rc.Ctx(view(floor=17))], floor=16).t_lost, 16.0)

    def test_t_rises_with_hp_and_with_each_potion_held(self):
        c = currency()
        hps = [c.u(hp, 80, c.potions0) for hp in range(1, 81, 7)]
        self.assertEqual(hps, sorted(hps))
        self.assertLess(hps[0], hps[-1])
        both = c.potions0
        self.assertLess(c.u(40, 80, ()), c.u(40, 80, both[:1]))
        self.assertLess(c.u(40, 80, ()), c.u(40, 80, both[1:]))
        self.assertLess(c.u(40, 80, both[:1]), c.u(40, 80, both))

    def test_a_potion_costs_the_heads_marginal_not_ten(self):
        head = StubHead(base=2.0, hp_rate=0.05, potion=0.2)
        c = currency(head)
        per_potion = c(50, 80, c.potions0) - c(50, 80, c.potions0[:1])
        expected = 0.2 / c.span * (80 - rc.DEATH_VALUE)
        self.assertAlmostEqual(per_potion, expected)
        self.assertNotAlmostEqual(per_potion, rc.POTION_VALUE, delta=0.5)
        # and one HP is the head's, too: a tenth of a floor per two HP here
        per_hp = c(51, 80, ()) - c(50, 80, ())
        self.assertAlmostEqual(per_hp, 0.05 / c.span * (80 - rc.DEATH_VALUE))

    def test_burning_blood_heals_after_a_won_fight_and_the_cap_holds(self):
        with_bb = currency(relics=(BURNING_BLOOD,))
        without = currency()
        self.assertEqual((with_bb.end_heal, without.end_heal), (6, 0))
        self.assertEqual(with_bb.u(74, 80, ()), with_bb.u(80, 80, ()))  # 74 + 6 is the cap
        self.assertEqual(with_bb.u(50, 80, ()), without.u(56, 80, ()))  # the heal is added to the HP the head is asked about
        self.assertLess(without.u(74, 80, ()), without.u(80, 80, ()))
        self.assertEqual(rc.end_heal_of([vocab.relic_id_by_name("BlackBlood")]), 12)
        self.assertEqual(rc.end_heal_of([vocab.relic_id_by_name("BlackBlood"), BURNING_BLOOD]), 12)
        self.assertEqual(rc.end_heal_of([]), 0)

    def test_a_judge_leafs_potion_count_averages_the_subsets_and_interpolates(self):
        head = StubHead(kinds={BLOCK_POTION: 0.2, FIRE_POTION: 1.0})
        c = currency(head)
        t = lambda *ps: c.u(50, 80, ps)  # noqa: E731
        self.assertAlmostEqual(c.u(50, 80, 0.0), t())
        self.assertAlmostEqual(c.u(50, 80, 1.0), (t(BLOCK_POTION) + t(FIRE_POTION)) / 2)
        self.assertAlmostEqual(c.u(50, 80, 2.0), t(BLOCK_POTION, FIRE_POTION))
        self.assertAlmostEqual(c.u(50, 80, 0.5), (t() + c.u(50, 80, 1.0)) / 2)
        self.assertAlmostEqual(c.u(50, 80, 7.0), t(BLOCK_POTION, FIRE_POTION))  # more than it started with is all of them
        self.assertAlmostEqual(c(50, 80, 1.5), c(50, 80, 1.0) / 2 + c(50, 80, 2.0) / 2)

    def test_the_memo_answers_a_repeated_state_without_the_head_and_prefill_is_one_call(self):
        head = StubHead()
        c = currency(head)
        calls = head.calls
        c.u(40, 80, c.potions0)
        self.assertEqual(head.calls, calls + 1)
        for _ in range(5):
            c.u(40, 80, c.potions0[::-1])  # the order of the potions is not the key
        self.assertEqual(head.calls, calls + 1)
        c.prefill([(hp, 80, c.potions0) for hp in range(1, 61)])
        self.assertEqual(head.calls, calls + 2)
        c.u(33, 80, c.potions0)
        self.assertEqual(head.calls, calls + 2)
        self.assertEqual(c.summary()["evals"], 1 + 1 + 59)  # the full state, the first state, and the 59 states of the prefill not yet known

    def test_a_win_is_clipped_to_the_scale_and_the_clips_are_counted(self):
        c = currency(StubHead(base=0.5, hp_rate=0.05, potion=0.2))
        self.assertEqual(c.u(100000, 100000, ()), rc.U_MAX)
        self.assertEqual(c.stats["clipped_high"], 1)
        self.assertEqual(c.u(-100000, 80, ()), 0.0)
        self.assertEqual(c.stats["clipped_low"], 1)

    def test_a_fight_with_no_scale_is_priced_as_today(self):
        flat = currency(StubHead(base=0.0, hp_rate=0.0, potion=0.0))
        self.assertTrue(flat.fallback)
        self.assertEqual(flat(40, 80, (BLOCK_POTION, FIRE_POTION)), 40 + 2 * rc.POTION_VALUE)
        self.assertEqual(flat(40, 80, 1.5), 40 + 1.5 * rc.POTION_VALUE)
        self.assertTrue(flat.summary()["fallback"])
        self.assertFalse(currency().summary()["fallback"])

    def test_nan_from_the_head_is_a_fall_back_not_a_search_of_nans(self):
        class NanHead(StubHead):
            def floors(self, ctxs, table=None):
                return np.full(len(ctxs), np.nan)

        self.assertTrue(rc.FightCurrency(NanHead(), [rc.Ctx(view())]).fallback)

    def test_several_contexts_are_averaged(self):
        class BossHead(StubHead):
            def value(self, v):
                return super().value(v) + (1.0 if v.boss == 84 else 3.0)

        head = BossHead()
        a, b = view(boss=84), view(boss=74)
        both = rc.FightCurrency(head, [rc.Ctx(a), rc.Ctx(b)])
        one_a, one_b = rc.FightCurrency(head, [rc.Ctx(a)]), rc.FightCurrency(head, [rc.Ctx(b)])
        self.assertAlmostEqual(both.t_full, (one_a.t_full + one_b.t_full) / 2)
        self.assertAlmostEqual(both.t_full - one_a.t_full, 1.0)
        calls = head.calls
        both.u(40, 80, ())
        self.assertEqual(head.calls, calls + 1)  # one state is one head call over all its contexts


class ViewTests(unittest.TestCase):
    def test_a_view_round_trips_through_json(self):
        v = view(potions=(BLOCK_POTION, FIRE_POTION))
        self.assertEqual(rc.view_from_json(json.loads(json.dumps(rc.view_to_json(v)))), v)
        self.assertTrue(rc.view_to_json(v)["map"])

    def test_a_view_without_a_map_gets_the_mean_block_and_the_rows_to_the_boss(self):
        table = rc.MapTable({(d, t): np.full(features.C_SIZE, d + 0.01 * t, dtype=np.float32) for d in range(0, 16) for t in (1, 2, 6)})
        bare = dataclasses.replace(view(floor=5), map_nodes=(), position=(0, 0))
        x = rc.encode_ctx(rc.Ctx(bare, rc.NODE_ELITE), table)
        d = 17 - 5
        np.testing.assert_allclose(x[features.C_OFFSET : features.C_OFFSET + features.C_SIZE], d + 0.02, rtol=1e-5)
        self.assertAlmostEqual(float(x[features.A_OFFSET + 15]), d / features.MAP_ROWS)
        plain = features.encode(bare)
        keep = np.ones(len(x), bool)
        keep[features.C_OFFSET : features.C_OFFSET + features.C_SIZE] = False
        keep[features.A_OFFSET + 15] = False
        np.testing.assert_array_equal(x[keep], plain[keep])
        exact = rc.Ctx(view(floor=5), rc.NODE_ELITE)
        np.testing.assert_array_equal(rc.encode_ctx(exact, table), features.encode(exact.view))  # a view with a map is features.encode

    def test_the_rows_to_the_boss_count_floors_in_every_act(self):
        self.assertEqual(rc.rows_to_boss(view(floor=2, act_index=0)), 15)
        self.assertEqual(rc.rows_to_boss(view(floor=17, act_index=0)), 0)
        self.assertEqual(rc.rows_to_boss(view(floor=30, act_index=1)), 3)
        self.assertEqual(rc.rows_to_boss(view(floor=40, act_index=2)), 8)
        self.assertEqual(rc.rows_to_boss(view(floor=60, act_index=2)), 0)

    def test_the_map_table_takes_the_nearest_distance_of_the_type(self):
        table = rc.MapTable({(2, 1): np.ones(features.C_SIZE) * 2, (9, 1): np.ones(features.C_SIZE) * 9, (0, 6): np.zeros(features.C_SIZE)})
        self.assertEqual(float(table.block(3, 1)[0]), 2.0)
        self.assertEqual(float(table.block(12, 1)[0]), 9.0)
        self.assertEqual(float(table.block(4, 6)[0]), 0.0)  # the boss has one distance
        self.assertEqual(float(table.block(3, 2)[0]), 2.0)  # a type the maps lack reads a monster's
        again = rc.MapTable.from_json(json.loads(json.dumps(table.to_json())))
        self.assertEqual(set(again.blocks), set(table.blocks))
        with self.assertRaises(ValueError):
            rc.MapTable({})

    def test_the_shipped_table_has_every_fight_type_near_every_boss_distance(self):
        table = rc.default_table()
        self.assertIn((0, rc.NODE_BOSS), table.blocks)
        # a hallway fight stands 2 to 15 rows below the boss (the row above the boss is a rest site, and one more row is not a fight on
        # most maps); an elite at most ten
        self.assertTrue({d for d, t in table.blocks if t == rc.NODE_MONSTER} >= set(range(9, 16)) | set(range(2, 7)))
        self.assertTrue(any(t == rc.NODE_ELITE for _, t in table.blocks))
        for block in table.blocks.values():
            self.assertEqual(block.shape, (features.C_SIZE,))
            self.assertTrue(np.isfinite(block).all())


def fake_doc(*, hp=60, max_hp=80, gold=99, deck=None, relics=(36,), potions=(BLOCK_POTION,), slots=2, encounter=OVERGROWTH_ONLY):
    deck = deck or [STRIKE] * 5 + [DEFEND] * 4 + [BASH]
    cards = [{"def_id": c, "upgraded": i == 0, "enchantment": 0, "enchant_amount": 0} for i, c in enumerate(deck)]
    return {"ascension": 10, "encounter_id": encounter, "turn": 0, "player": {"hp": hp, "max_hp": max_hp, "gold": gold},
            "relics": [{"id": r, "counter": 0} for r in relics],
            "potions": {"max_slots": slots, "slots": list(potions) + [0] * (6 - len(potions))},
            "piles": {"hand": cards[:3], "discard": [], "exhaust": [], "draw": {"top": cards[3:5], "middle": cards[5:8], "bottom": cards[8:]}}}


def fake_start(floor=5, act=1, node_type=rc.NODE_MONSTER, encounter=OVERGROWTH_ONLY, **doc):
    kind = {1: "hallway", 2: "elite", 6: "boss"}[node_type]
    return {"id": f"T|f{floor:02d}", "cat": f"act{act}-{kind}", "kind": kind, "act": act, "floor": floor, "node_type": node_type,
            "encounter_id": encounter, "doc": fake_doc(encounter=encounter, **doc)}


class MapTableBuilderTests(unittest.TestCase):
    def test_the_builder_averages_block_c_over_real_act_one_maps(self):
        table = rc.build_map_table(["RCMAP1", "RCMAP2", "RCMAP3"])
        self.assertIn((0, rc.NODE_BOSS), table.blocks)
        self.assertTrue(any(t == rc.NODE_MONSTER for _, t in table.blocks))
        for block in table.blocks.values():
            self.assertEqual(block.shape, (features.C_SIZE,))
            self.assertTrue(np.isfinite(block).all())
        # one path from the boss node to the boss: no node types ahead and a single path (log 2 / 5)
        self.assertAlmostEqual(float(table.block(0, rc.NODE_BOSS)[24]), math.log(2) / 5, places=5)
        self.assertEqual(table.block(0, rc.NODE_BOSS)[3], 1.0)  # no monster ahead of the boss


class ContextTests(unittest.TestCase):
    def test_a_start_without_a_run_view_is_built_from_its_document(self):
        ctxs = rc.contexts_of_start(fake_start(potions=(BLOCK_POTION, FIRE_POTION), slots=3, hp=41, gold=123))
        v = ctxs[0].view
        self.assertEqual((v.hp, v.max_hp, v.gold, v.floor, v.act_index, v.ascension), (41, 80, 123, 5, 0, 10))
        self.assertEqual(v.potions, tuple(sorted((BLOCK_POTION, FIRE_POTION))))
        self.assertEqual(len(v.deck), 10)  # every pile, the draw pile's known top, middle and bottom included
        self.assertEqual(v.relics, (36,))
        self.assertEqual(v.max_potion_slots, 3)
        self.assertFalse(ctxs[0].exact)

    def test_an_unknown_boss_is_every_boss_of_the_region_and_a_known_one_is_one(self):
        hall = rc.contexts_of_start(fake_start())
        self.assertEqual({c.view.act_region for c in hall}, {OVERGROWTH})
        self.assertEqual({c.view.boss for c in hall}, set(POOLS[OVERGROWTH]["boss"]))
        boss_id = POOLS[UNDERDOCKS]["boss"][0]
        boss = rc.contexts_of_start(fake_start(floor=17, node_type=rc.NODE_BOSS, encounter=boss_id))
        self.assertEqual([(c.view.act_region, c.view.boss) for c in boss], [(UNDERDOCKS, boss_id)])

    def test_the_region_of_act_one_is_both_when_the_encounter_does_not_say(self):
        self.assertEqual(rc.regions_of(1, None), [OVERGROWTH, UNDERDOCKS])
        self.assertEqual(rc.regions_of(1, OVERGROWTH_ONLY), [OVERGROWTH])
        self.assertEqual(rc.regions_of(2, 12345), [HIVE])
        self.assertEqual(rc.regions_of(3, None), [vocab.REGION_GLORY])

    def test_a_run_view_in_the_record_is_the_one_exact_context(self):
        v = view(floor=9)
        start = fake_start(floor=9)
        start["run_view"] = rc.view_to_json(v)
        [ctx] = rc.contexts_of_start(start)
        self.assertEqual(ctx.view, v)
        self.assertTrue(ctx.exact)

    def test_act_two_starts_are_in_the_hive(self):
        ctxs = rc.contexts_of_start(fake_start(floor=25, act=2, encounter=POOLS[HIVE]["elite"][0], node_type=rc.NODE_ELITE))
        self.assertEqual({c.view.act_region for c in ctxs}, {HIVE})
        self.assertEqual({c.view.act_index for c in ctxs}, {1})

    def test_the_currency_of_a_start_is_kept_for_the_process(self):
        start = fake_start()
        a = rc.currency_for_start(start, StubHead())
        self.assertIs(a, rc.currency_for_start(start, StubHead()))
        rc._CURRENCIES.clear()


def random_nets():
    from test_run_train_models import fight_pair, value_pair

    return models.NumpyNets(fight_pair()[1], value_pair()[1], {"feature_version": FEATURE_VERSION, "format": models.CHECKPOINT_FORMAT})


class HeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.head = rc.Head(random_nets())

    def test_a_batch_of_mixed_regions_and_bosses_agrees_with_one_by_one(self):
        ctxs = [rc.Ctx(view()), rc.Ctx(view(hp=20)), rc.Ctx(view(act_region=HIVE, act_index=1, floor=20, boss=POOLS[HIVE]["boss"][0])),
                rc.Ctx(view(boss=POOLS[OVERGROWTH]["boss"][1])), rc.Ctx(view(hp=70, potions=()))]
        together = self.head.floors(ctxs)
        alone = np.array([self.head.floors([c])[0] for c in ctxs])
        np.testing.assert_allclose(together, alone, rtol=0, atol=1e-5)
        self.assertEqual(len(set(np.round(together, 4))), len(ctxs))

    def test_the_value_is_the_expected_floor_of_the_nets_own_distribution(self):
        v = view()
        x = features.encode(v)[None, : self.head.nets.base_size]
        block = self.head.nets.fight.summary(x, features.fight_queries(v))
        probs = self.head.nets.value.floor_probs(np.concatenate([x, block], axis=1))
        self.assertAlmostEqual(float(self.head.floors([rc.Ctx(v)])[0]), float(models.expected_floors(probs, rc.W_WON)[0]), places=5)
        self.assertEqual(self.head.floors([]).shape, (0,))

    def test_a_currency_over_the_real_nets_is_finite_and_scaled(self):
        c = rc.FightCurrency(self.head, [rc.Ctx(view())])
        self.assertTrue(math.isfinite(c.t_full))
        if not c.fallback:
            self.assertEqual(c(80, 80, c.potions0), 80.0)

    def test_the_frozen_head_is_pinned_by_its_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "head.npz"
            from test_run_train_models import fight_pair, value_pair

            models.save(path.with_suffix(".pt"), fight_pair()[0], value_pair()[0], {"generation": 1})
            (Path(tmp) / "agent" / "gen11" / "candidate").mkdir(parents=True)
            (Path(tmp) / "agent" / "gen11" / "candidate" / "run_value.npz").write_bytes(path.read_bytes())
            with mock.patch.dict(os.environ, {rc.REFEREE_ENV: tmp}):
                os.environ.pop(rc.HEAD_ENV, None)
                with self.assertRaisesRegex(ValueError, "not the frozen run head"):
                    rc.default_head()
                os.environ[rc.HEAD_ENV] = str(path)  # a head named on purpose is taken as it is
                self.assertEqual(rc.default_head().sha256, rc.file_sha256(path))


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.starts = [fake_start(floor=2 + i % 14, hp=30 + 3 * i, potions=(BLOCK_POTION,) if i % 2 else (FIRE_POTION, BLOCK_POTION), slots=2)
                      for i in range(12)]
        cls.starts += [fake_start(floor=17, act=1, node_type=rc.NODE_BOSS, encounter=POOLS[OVERGROWTH]["boss"][0], hp=55 + i)
                       for i in range(6)]
        cls.table = rc.MapTable({(d, t): np.zeros(features.C_SIZE, np.float32) for d in range(16) for t in (1, 2, 6)})

    def run_audit(self, head):
        return rc.audit(head, self.starts, per_cell=12, table=self.table, kinds=[BLOCK_POTION, FIRE_POTION])

    def test_a_sane_head_passes_and_the_table_is_filled(self):
        rep = self.run_audit(StubHead(base=2.0, hp_rate=0.05, potion=0.3))
        self.assertTrue(rep["passed"], rep["stop"])
        hall = rep["cells"]["act1-hallway"]
        self.assertAlmostEqual(hall["potion_hp_equivalent"], 0.3 / 0.05, places=1)
        self.assertEqual(hall["potion_hp_equivalent_today"], 10.0)
        self.assertAlmostEqual(hall["floors_per_10hp"]["60"], 0.5, places=3)
        self.assertEqual(rep["cells"]["act1-boss"]["states"], 6)
        self.assertTrue(all(m["increasing"] for m in rep["monotone"].values()))
        self.assertIn("act1-mid", rep["monotone"])
        self.assertIn("PASSED", rc.audit_markdown(rep))

    def test_a_potion_worth_almost_nothing_stops_the_audit(self):
        rep = self.run_audit(StubHead(base=2.0, hp_rate=0.05, potion=0.05))
        self.assertFalse(rep["passed"])
        self.assertTrue(any("worth 1.0 HP" in s for s in rep["stop"]), rep["stop"])

    def test_a_potion_worth_a_hundred_hp_stops_the_audit(self):
        rep = self.run_audit(StubHead(base=2.0, hp_rate=0.05, potion=5.0))
        self.assertFalse(rep["passed"])
        self.assertTrue(any("worth 100.0 HP" in s for s in rep["stop"]), rep["stop"])

    def test_a_head_that_falls_with_hp_stops_the_audit_in_every_cell(self):
        rep = self.run_audit(StubHead(base=40.0, hp_rate=-0.05, potion=0.3))
        self.assertFalse(rep["passed"])
        text = " ".join(rep["stop"])
        self.assertIn("act1-hallway: V does not rise with HP", text)
        self.assertIn("not increasing in HP", text)
        self.assertFalse(rep["monotone"]["act1-hallway"]["increasing"])

    def test_a_band_that_dips_is_named_and_the_others_are_not(self):
        class Dip(StubHead):
            def value(self, v):  # falls with HP in the late band only
                late = rc.band_of(v) == "late"
                return v.floor + 20.0 + (-0.05 if late else 0.05) * v.hp + 0.3 * len(v.potions)

        rep = self.run_audit(Dip())
        self.assertIn("act1-late", " ".join(rep["stop"]))
        self.assertTrue(rep["monotone"]["act1-early"]["increasing"])

    def test_the_weighted_median_follows_the_kinds_held(self):
        self.assertEqual(rc._weighted_median([(1.0, 1), (50.0, 1), (7.0, 5)]), 7.0)
        self.assertEqual(rc._weighted_median([(3.0, 1)]), 3.0)


class ConfigTests(unittest.TestCase):
    def test_the_currency_configs_are_the_base_with_r(self):
        for name, base in (("a3-v1-r", "a3-v1"), ("judge-v0-r", "judge-v0"), ("a3-v1-k-r", "a3-v1-k")):
            cfg, plain = sc.get(name), sc.get(base)
            self.assertTrue(cfg.run_currency, name)
            self.assertFalse(plain.run_currency, base)
            self.assertEqual(dataclasses.replace(cfg, name=base, run_currency=False), plain)

    def test_any_known_base_takes_the_suffix_and_an_unknown_one_is_refused(self):
        self.assertTrue(sc.get("gen0-60-r").run_currency)
        self.assertEqual(sc.get("gen0-60-r").sims, 60)
        self.assertTrue(sc.get("flat-30-r").run_currency)
        with self.assertRaises(KeyError):
            sc.get("nothing-r")

    def test_the_stakes_and_the_hold_go_after_the_r(self):
        cfg = sc.get("a3-v1-r@h1e4b16+hold30")
        self.assertEqual((cfg.run_currency, cfg.stakes, cfg.potion_hold), (True, (1, 4, 16), 0.3))
        self.assertEqual(cfg.name, "a3-v1-r@h1e4b16+hold30")
        self.assertEqual(sc.for_agent("a3-v1-r", "a3-v1-r@h1e4b16", None).stakes, (1, 4, 16))
        with self.assertRaises(ValueError):
            sc.with_currency(sc.get("a3-v1@h1e4b16"))
        with self.assertRaises(ValueError):
            sc.with_currency(sc.get("a3-v1-r"))

    def test_a_currency_config_will_not_search_without_the_fights_currency(self):
        cfg = sc.get("a3-v1-r")
        with self.assertRaisesRegex(ValueError, "run's terms"):
            cfg.search_kwargs(None, env_prior=False)
        f = lambda hp, mx, p: 0.0  # noqa: E731
        self.assertIs(cfg.search_kwargs(None, env_prior=False, terminal=f)["terminal"], f)

    def test_a_plain_config_ignores_a_terminal_and_its_kwargs_are_what_they_were(self):
        plain = sc.get("a3-v1").search_kwargs(None, env_prior=False)
        self.assertNotIn("terminal", plain)
        self.assertEqual(sc.get("a3-v1").search_kwargs(None, env_prior=False, terminal=lambda *a: 0.0), plain)
        self.assertEqual({k: v for k, v in sc.get("a3-v1-r").search_kwargs(None, env_prior=False, terminal=1).items() if k != "terminal"}, plain)

    def test_a_currency_config_is_a_3_v1_in_every_other_setting(self):
        a, r = sc.get("a3-v1"), sc.get("a3-v1-r")
        pos = sc.Position("target", 6, 0.9, True, fight="boss")
        self.assertEqual((a.sims_for(pos), a.needs_position, a.wall_factor()), (r.sims_for(pos), r.needs_position, r.wall_factor()))


class SearchTerminalTests(unittest.TestCase):
    def fight(self, **kw):
        kw.setdefault("potions", (BLOCK_POTION,))
        return Fight(0, NIBBITS_WEAK, [STRIKE] * 5 + [DEFEND] * 4 + [BASH], **kw)

    def test_the_terminal_prices_a_capped_playout_from_hp_max_hp_and_the_potions_held(self):
        fight = self.fight()
        seen = []
        try:
            world = live_search.Combat(fight.handle).fork(3)
            try:
                out = live_search._outcome(world, 10.0, lambda hp, mx, p: seen.append((hp, mx, p)) or 123.5)
                plain = live_search._outcome(world, 10.0)
            finally:
                world.close()
        finally:
            fight.close()
        self.assertEqual(seen, [(64, 80, (BLOCK_POTION,))])
        self.assertEqual((out.value, out.hp, out.died, out.truncated), (123.5, 64, False, True))
        self.assertEqual(plain.value, 64 + 10.0)  # without it: what it always was

    def test_a_death_is_minus_100_whatever_the_terminal_says(self):
        fight = self.fight(hp=1)
        try:
            world = live_search.Combat(fight.handle).fork(3)
            try:
                for _ in range(40):
                    if world.done:
                        break
                    end = next(m for m in live_search._world_moves(world) if m.kind == "end_turn")
                    live_search._step(world, end.action, end.target)
                self.assertTrue(world.done and not world.won())
                out = live_search._outcome(world, 10.0, lambda hp, mx, p: 999.0)
            finally:
                world.close()
        finally:
            fight.close()
        self.assertEqual((out.value, out.died), (live_search.DEATH_VALUE, True))

    def test_a_terminal_that_says_hp_plus_ten_a_potion_is_the_old_search_exactly(self):
        def run(**kw):
            fight = self.fight()
            try:
                return live_search.search_live(fight.handle, time_budget_s=120.0, min_child_sims=2, max_sims=60, confirm_pairs=2, seed=11, **kw)
            finally:
                fight.close()

        old = run()
        new = run(terminal=lambda hp, mx, p: hp + 10.0 * (len(p) if isinstance(p, tuple) else p))
        key = lambda r: [(c.action, c.target, c.visits, round(c.value_mean, 9), c.p_death) for c in r.children]  # noqa: E731
        self.assertEqual(key(old), key(new))
        self.assertEqual((old.best.action, old.best.target), (new.best.action, new.best.target))

    def test_the_terminal_sets_the_value_of_every_won_playout(self):
        fight = self.fight()
        try:
            result = live_search.search_live(fight.handle, time_budget_s=120.0, min_child_sims=2, max_sims=40, confirm_pairs=1, seed=5,
                                             terminal=lambda hp, mx, p: 123.0)
        finally:
            fight.close()
        sure = [c for c in result.children if c.p_death == 0.0]
        self.assertTrue(sure)
        for c in sure:
            self.assertAlmostEqual(c.value_mean, 123.0)

    def test_the_steps_form_takes_it_too_and_the_parallel_searcher_does_not(self):
        fight = self.fight()
        try:
            steps = live_search.search_steps(fight.handle, time_budget_s=120.0, min_child_sims=1, max_sims=10, confirm_pairs=0, seed=2,
                                             terminal=lambda hp, mx, p: 77.0)
            result = live_search._drive(steps, None)
        finally:
            fight.close()
        sure = [c for c in result.children if c.p_death == 0.0]
        self.assertTrue(sure)
        for c in sure:
            self.assertAlmostEqual(c.value_mean, 77.0)
        self.assertNotIn("terminal", live_search._SEARCH_SETTINGS)

    def test_the_judges_leaf_composes_through_the_terminal(self):
        row = [0.0, 0.0, 1.5]  # P(win) 1/2, end HP half of max, 1.5 potions
        old = judge_client.estimate(row, 80)
        self.assertAlmostEqual(old.value, 0.5 * (40 + 15) + 0.5 * (-100))
        seen = []
        new = judge_client.estimate(row, 80, terminal=lambda hp, mx, p: seen.append((hp, mx, p)) or 60.0)
        self.assertEqual(seen, [(40.0, 80, 1.5)])
        self.assertAlmostEqual(new.value, 0.5 * 60.0 + 0.5 * (-100))
        self.assertEqual((new.p_death, new.hp_if_win, new.potions_if_win), (old.p_death, old.hp_if_win, old.potions_if_win))
        self.assertEqual(judge_client.estimate(row, 80, 10.0, None), old)

    def test_the_old_label_generators_terminal_value_reads_the_same_module(self):
        class FakeSim:
            def __init__(self, won):
                self._won = won
                self.obs = [0] * native.OBS_SIZE
                from combat_tokens import POTION_OFFSET

                self.obs[POTION_OFFSET] = BLOCK_POTION

            def won(self):
                return self._won

            def hp(self):
                return 30

            def max_hp(self):
                return 60

        got = []
        cur = SimpleNamespace(u=lambda hp, mx, p: got.append((hp, mx, p)) or 0.4375)
        self.assertEqual(combat_mcts.terminal_value(FakeSim(True)), 0.5)
        self.assertEqual(combat_mcts.terminal_value(FakeSim(True), cur), 0.4375)
        self.assertEqual(got, [(30, 60, (BLOCK_POTION,))])
        self.assertEqual(combat_mcts.terminal_value(FakeSim(False), cur), 0.0)
        self.assertEqual(combat_mcts.held_potions(FakeSim(True)), (BLOCK_POTION,))


def head_file(tmp: str) -> Path:
    """A random head saved as the nets' checkpoint pair; its npz is the file a currency config prices with."""
    from test_run_train_models import fight_pair, value_pair

    pt = Path(tmp) / "head.pt"
    models.save(pt, fight_pair()[0], value_pair()[0], {"generation": 1})
    return models.npz_for(pt)


class GamesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.head_path = head_file(cls._tmp.name)
        cls._env = os.environ.get(rc.HEAD_ENV)
        os.environ[rc.HEAD_ENV] = str(cls.head_path)

    @classmethod
    def tearDownClass(cls):
        if cls._env is None:
            os.environ.pop(rc.HEAD_ENV, None)
        else:
            os.environ[rc.HEAD_ENV] = cls._env
        cls._tmp.cleanup()

    def tearDown(self):
        rc._CURRENCIES.clear()

    def test_the_gym_plays_a_start_in_the_run_currency_and_the_row_says_so(self):
        import fight_gym as fg

        fight = Fight(0, NIBBITS_WEAK, [STRIKE] * 5 + [DEFEND] * 4 + [BASH], potions=(BLOCK_POTION,), relics=(BURNING_BLOOD,))
        try:
            doc = native.get_state_json(fight.handle)
        finally:
            fight.close()
        start = {"id": "UT|f05", "cat": "act1-hallway", "kind": "hallway", "act": 1, "floor": 5, "node_type": 1, "doc": doc, "hp": 64,
                 "max_hp": 80, "encounter_id": doc.get("encounter_id")}
        cfg = dataclasses.replace(sc.get("a3-v1-r"), sims=8, schedule=None, min_child_sims=1)
        world = fg._fresh_world(doc, start["id"], 0, "g")
        try:
            row = fg._play(world, start, 0, cfg, {}, "g")
        finally:
            world.close()
        self.assertTrue(0.0 <= row["u_end"] <= rc.U_MAX, row["u_end"])
        if row["won"]:
            self.assertGreater(row["u_end"], 0.0)
        self.assertGreater(row["currency"]["priced"], 0)
        self.assertEqual(row["currency"]["t_lost"], 5.0)
        # the same start and draw under the plain config carries neither field and plays in HP
        world = fg._fresh_world(doc, start["id"], 0, "g")
        try:
            plain = fg._play(world, start, 0, dataclasses.replace(cfg, run_currency=False, name="x"), {}, "g")
        finally:
            world.close()
        self.assertNotIn("currency", plain)
        self.assertNotIn("u_end", plain)

    def test_the_arm_is_known_to_the_gym(self):
        import fight_gym as fg

        cfg = fg.arm_settings("a3-v1-r")["cfg"]
        self.assertTrue(cfg.run_currency)
        self.assertIn("a3-v1-r", fg.ARMS)
        for cat in ("act1-boss", "act2"):
            self.assertGreater(fg.MEAN_S[(cat, "a3-v1-r")], fg.MEAN_S[(cat, "a3-v1")])

    def test_a_replays_start_record_carries_the_run_view_and_the_lite_list_leaves_it_out(self):
        import gym_loop as gl

        self.assertEqual(set(gl.LITE_DROPS), {"doc", "run_view"})
        with tempfile.TemporaryDirectory() as tmp:
            rec = {"id": "N:S|f05", "seed": "S", "doc": {"x": 1}, "run_view": rc.view_to_json(view()), "cat": "act1-hallway", "kind": "hallway",
                   "act": 1, "floor": 5, "node_type": 1, "split": "train", "kind_same": True}
            gl.write_start_set(Path(tmp), [rec])
            lite = json.loads((Path(tmp) / "starts.lite.json").read_text(encoding="utf-8"))
            full = json.loads((Path(tmp) / "starts.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(set(lite[0]), set(rec) - {"doc", "run_view"})
        self.assertEqual(full["run_view"], rec["run_view"])

    def test_a_whole_run_records_the_boss_doors_hand_and_prices_its_fights(self):
        import gen0
        import gen_loop

        plain = gen0.play_emu_run("RCURR1", 0, "flat-3", agent=gen_loop.Agent())
        self.assertIsInstance(plain["boss_door"], list)
        self.assertNotIn("currency", plain)
        priced = gen0.play_emu_run("RCURR1", 0, "flat-3-r", agent=gen_loop.Agent())
        self.assertEqual(priced["config"], "flat-3-r")
        self.assertGreaterEqual(priced["currency"]["fights"], 1)
        self.assertGreater(priced["currency"]["priced"], 0)
        self.assertEqual(priced["currency"]["head"], rc.file_sha256(self.head_path)[:8])
        for door in priced["boss_door"]:
            self.assertEqual(len(door), 5)


if __name__ == "__main__":
    unittest.main()
