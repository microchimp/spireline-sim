"""The Act 2 chance model A2 and the deck forecast H2 (rl/act2_chance.py; D59, referee/backlog/act2-build/OBJECTIVE.md sections 1, 3, 4).

Pinned: the target rule (humans: a run that ended before floor 18 is dropped, 1 iff it ended on floor 34 or later or won; ours: the same
from the run's outcome), the exam fifth and fifth 4 never enter a fit (`check_no_exam`, the human rows' split, our held-out seeds); H2 is
FightNet's summary against the Hive's pools and three bosses, equal to `nets.fight.summary` on the Hive's queries; A2's source term is read
at ours (0) and moves human rows only; a relation planted in synthetic rows is recovered; the numpy forward pass equals the torch one;
`Act2Scorer` reads the written file (`.pt` or `.npz`) and its probabilities are the net's at source 0; the slope of Look 0 (a) is recovered
on a planted relation and its bootstrap interval is over runs. Synthetic rows throughout: nothing here reads the corpus or the lab.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import act2_chance as a2  # noqa: E402
import human_prior as hp  # noqa: E402
import test_features_map_grid as tfm  # noqa: E402
from run_train import features, models, shards, vocab  # noqa: E402
from run_train.features import CardRef, Option  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover
    torch = None


def fight_params(seed: int = 3, scale: float = 0.08) -> dict[str, np.ndarray]:
    """A FightNet's weights (the numpy net's keys and shapes) at random: what H2 is computed from in these tests."""
    rng = np.random.default_rng(seed)
    shapes = {"embed.deck.weight": (64, 1154), "embed.relics.weight": (32, 296), "embed.potions.weight": (8, 63),
              "hidden1.weight": (256, models.FIGHT_DENSE + 104), "hidden1.bias": (256,), "hidden2.weight": (128, 256), "hidden2.bias": (128,),
              "out.weight": (22, 128), "out.bias": (22,)}
    return {k: (rng.standard_normal(s) * scale).astype(np.float32) for k, s in shapes.items()}


def a2_params(seed: int = 5, scale: float = 0.05, h2_width: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shapes = {"embed.deck.weight": (64, 1154), "embed.relics.weight": (32, 296), "embed.potions.weight": (8, 63),
              "hidden1.weight": (256, 350 + 104 + h2_width), "hidden1.bias": (256,), "hidden2.weight": (256, 256), "hidden2.bias": (256,),
              "score.weight": (1, 256), "score.bias": (1,), "src_w": (256,)}
    return {k: (rng.standard_normal(s) * scale).astype(np.float32) for k, s in shapes.items()}


def real_bases(n: int, seed: int = 0, region: int = vocab.REGION_OVERGROWTH, act_index: int = 0) -> np.ndarray:
    """`n` base vectors (1,855 wide) of real encoded views: varied hp, gold and decks."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        deck = tuple(CardRef(int(c), bool(rng.integers(0, 2))) for c in rng.choice([472, 131, 30, 10001], size=int(rng.integers(3, 9))))
        view = tfm.make_view(hp=int(rng.integers(5, 80)), gold=int(rng.integers(0, 400)), deck=deck, act_region=region, act_index=act_index,
                             boss=75 if region == vocab.REGION_HIVE else 83, floor=int(rng.integers(2, 16)))
        out.append(features.encode(view))
    return np.stack(out).astype(np.float32)


class TargetRuleTests(unittest.TestCase):
    def test_human_target_drops_before_floor_18_and_is_one_from_floor_34_or_won(self):
        self.assertIsNone(a2.target_of(17, False))  # died on the Act 1 boss: not a run that killed it
        self.assertIsNone(a2.target_of(5, False))
        self.assertEqual(0, a2.target_of(18, False))
        self.assertEqual(0, a2.target_of(33, False))  # died on the Act 2 boss
        self.assertEqual(1, a2.target_of(34, False))
        self.assertEqual(1, a2.target_of(49, False))
        self.assertEqual(1, a2.target_of(20, True))  # won is one whatever floor the record ends on
        self.assertEqual((18, 34), (a2.ACT1_KILLED_FINAL, a2.ACT2_KILLED_FINAL))

    def test_our_target_reads_the_runs_outcome_the_same_way(self):
        self.assertIsNone(a2.ours_target({"final_floor": 17, "won": False, "acts_cleared": 0}))
        self.assertEqual(0, a2.ours_target({"final_floor": 25, "won": False, "acts_cleared": 1}))
        self.assertEqual(0, a2.ours_target({"final_floor": 33, "won": False, "acts_cleared": 1}))
        self.assertEqual(1, a2.ours_target({"final_floor": 34, "won": False, "acts_cleared": 2}))
        self.assertEqual(1, a2.ours_target({"final_floor": 52, "won": True, "acts_cleared": 3}))
        self.assertIsNone(a2.ours_target({"final_floor": 30, "won": False, "acts_cleared": 1, "truncated": True}))  # a run the clock cut is no outcome


class ExamTests(unittest.TestCase):
    def test_check_no_exam_refuses_the_exam_fifth_and_unknown_fifths_and_listed_ids(self):
        a2.check_no_exam([1, 2, 3])
        a2.check_no_exam([4], run_ids=["a", "b"], exam_ids={"c"})
        with self.assertRaises(ValueError):
            a2.check_no_exam([0, 1])  # the exam bucket
        with self.assertRaises(ValueError):
            a2.check_no_exam([1, 5])
        with self.assertRaises(ValueError):
            a2.check_no_exam([1], run_ids=["a", "b"], exam_ids={"b"})

    def test_the_module_never_opens_the_corpus_folder_or_the_exam_ids_file(self):
        text = Path(a2.__file__).read_text(encoding="utf-8")  # the shards were built without the exam fifth (fifths 1 to 4 only); the check reads the shard's own fifth column
        self.assertNotIn("spire-codex", text)
        self.assertNotIn("EXAM_IDS", text)
        self.assertNotIn("read_exam_ids", text)

    def test_training_and_report_fifths_do_not_overlap_and_exclude_the_exam_bucket(self):
        self.assertEqual((1, 2, 3), a2.TRAIN_FIFTHS)
        self.assertEqual(4, a2.REPORT_FIFTH)
        self.assertNotIn(a2.REPORT_FIFTH, a2.TRAIN_FIFTHS)
        self.assertNotIn(hp.EXAM_BUCKET, a2.TRAIN_FIFTHS + (a2.REPORT_FIFTH,))

    def test_human_rows_split_by_fifth_and_refuse_an_exam_row(self):
        ds = human_dataset(fifths=[1, 2, 3, 4], n_runs=80)
        rows = a2.human_rows(ds)
        parts = {int(p) for p in rows["part"]}
        self.assertEqual({a2.PART["train"], a2.PART["val"], a2.PART["report"]}, parts)
        fifth_of_row = {}
        for r, f in zip(rows["fifth"], rows["part"]):
            fifth_of_row.setdefault(int(r), set()).add(int(f))
        self.assertEqual({a2.PART["report"]}, fifth_of_row[4])  # fifth 4 is the report set and nothing else
        for fifth in (1, 2, 3):
            self.assertNotIn(a2.PART["report"], fifth_of_row[fifth])
        bad = human_dataset(fifths=[0, 1], n_runs=6)
        with self.assertRaises(ValueError):
            a2.human_rows(bad)

    def test_human_rows_keep_act_1_runs_past_the_boss_and_their_chosen_row_only(self):
        ds = human_dataset(fifths=[1, 2, 3, 4], n_runs=30)
        rows = a2.human_rows(ds)
        self.assertTrue(len(rows["y"]) > 0)
        self.assertEqual({0, 1}, set(rows["y"].tolist()) | {0, 1})
        dec = ds.dec
        kept = (dec[:, hp.C["act"]] == 0) & (dec[:, hp.C["final"]] >= 18) & (dec[:, hp.C["chosen"]] >= 0)
        kinds = np.isin(dec[:, hp.C["kind"]], [hp.KIND[k] for k in a2.KINDS])
        self.assertEqual(int((kept & kinds).sum()), len(rows["y"]))
        # the target of each row follows the human rule on its decision
        want = np.where((dec[kept & kinds, hp.C["final"]] >= 34) | (dec[kept & kinds, hp.C["won"]] == 1), 1, 0)
        np.testing.assert_array_equal(want, rows["y"])
        self.assertEqual(rows["dense"].shape, (len(rows["y"]), 350))
        self.assertEqual(rows["counts"].shape, (len(rows["y"]), 1513))

    def test_a_row_with_an_absurd_number_is_left_out_and_counted_and_h2_comes_with_the_rows(self):
        ds = human_dataset(fifths=[1, 2, 3, 4], n_runs=40)
        clean = a2.human_rows(ds)
        ds.dense[:, hp.S1_A.start + 4] = np.where(np.arange(len(ds.dense)) % 97 == 0, 1.0e6, ds.dense[:, hp.S1_A.start + 4])  # gold / 100 of a record with 1e8 gold
        stats: dict = {}
        rows = a2.human_rows(ds, fight=models.NumpyFightNet(fight_params()), stats=stats)
        self.assertGreater(stats["outlier_rows"], 0)
        self.assertEqual(len(clean["y"]) - stats["outlier_rows"], len(rows["y"]))
        self.assertTrue(np.all(np.isfinite(rows["dense"].astype(np.float32))))
        self.assertEqual((len(rows["y"]), a2.H2_SIZE), rows["h2"].shape)
        self.assertTrue(np.all(np.isfinite(rows["h2"])))
        fight = models.NumpyFightNet(fight_params())
        np.testing.assert_allclose(rows["h2"], a2.hive_block_parts(fight, rows["dense"], rows["counts"]), rtol=2e-3, atol=2e-3)  # the float16 store reads back to the same H2
        for run in np.unique(rows["run"]):
            self.assertEqual(1, int(rows["door"][rows["run"] == run].sum()))  # a door row is still one a run

    def test_one_door_row_per_run_the_last_act_1_decision(self):
        ds = human_dataset(fifths=[1, 2], n_runs=12)
        rows = a2.human_rows(ds)
        for run in np.unique(rows["run"]):
            m = rows["run"] == run
            self.assertEqual(1, int(rows["door"][m].sum()))
            last = np.flatnonzero(m)[np.argmax(rows["floor"][m] * 1000 + np.arange(m.sum()))]
            self.assertTrue(bool(rows["door"][last]))


def human_dataset(fifths: list[int], n_runs: int, seed: int = 1) -> hp.Dataset:
    """A tiny human Dataset: `n_runs` runs spread over `fifths`, each with a few Act 1 decisions (card, rest, route, shop, event) and one Act 2 one."""
    rng = np.random.default_rng(seed)
    dec, dense, counts, run_ids = [], [], [], []
    start = 0
    for ri in range(n_runs):
        fifth = fifths[ri % len(fifths)]
        final = int(rng.choice([8, 17, 25, 33, 40, 49]))
        won = int(final == 49 and ri % 2 == 0)
        run_ids.append(f"synthetic-{ri}")
        specs = [("card", 0, 3), ("route", 0, 4), ("rest", 0, 9), ("shop", 0, 7), ("event", 0, 12), ("card", 1, 20), ("rest", 0, 15)]
        for kind, act, floor in specs:
            n = int(rng.integers(2, 4))
            row = np.zeros(len(hp.COLS), dtype=np.int32)
            row[hp.C["kind"]] = hp.KIND[kind]
            row[hp.C["fifth"]], row[hp.C["act"]], row[hp.C["floor"]], row[hp.C["run"]] = fifth, act, floor, ri
            row[hp.C["won"]], row[hp.C["final"]] = won, final
            row[hp.C["start"]], row[hp.C["n"]] = start, n
            row[hp.C["chosen"]] = int(rng.integers(0, n))
            dec.append(row)
            d = rng.normal(0, 0.3, (n, hp.N_DENSE_V1)).astype(np.float32)
            dense.append(d)
            c = np.zeros((n, 1513), dtype=np.uint8)
            c[:, :6] = rng.integers(0, 3, (n, 6))
            counts.append(c)
            start += n
    ds = hp.Dataset(np.stack(dec), np.concatenate(dense), np.concatenate(counts), np.full(start, -1, np.int8), np.zeros(start, dtype=np.int32), run_ids)
    ds._index(None)  # noqa: SLF001
    return ds


class HiveBlockTests(unittest.TestCase):
    def test_act_2_is_always_the_hive_and_its_pools_and_bosses_are_the_vocabularys(self):
        self.assertEqual(vocab.REGION_HIVE, a2.HIVE_REGION)
        self.assertEqual(1, vocab.act_index_of_region()[vocab.REGION_HIVE])  # D58 addendum 16: the second act is [[Overgrowth, Underdocks], [Hive], [Glory]]
        self.assertNotIn(vocab.REGION_HIVE, (vocab.REGION_OVERGROWTH, vocab.REGION_UNDERDOCKS))
        queries = a2.hive_queries()
        pools = vocab.encounter_pools()[vocab.REGION_HIVE]
        for pool in ("weak", "normal", "elite", "boss"):
            self.assertEqual(list(pools[pool]), [q.encounter_id for q in queries if q.pool == pool])
        self.assertEqual(3, len([q for q in queries if q.pool == "boss"]))
        self.assertEqual({bid for bid, (_, region) in ((b, vocab.BOSSES[s]) for s, b in ((s, v[0]) for s, v in vocab.BOSSES.items())) if region == vocab.REGION_HIVE},
                         {q.encounter_id for q in queries if q.pool == "boss"})
        self.assertTrue(all(q.weak == (q.pool == "weak") for q in queries))

    def test_h2_equals_the_fight_nets_summary_on_the_hives_queries(self):
        fight = models.NumpyFightNet(fight_params())
        bases = real_bases(6, seed=2)
        h2 = a2.hive_block(fight, bases)
        self.assertEqual((6, features.FIGHT_BLOCK_SIZE), h2.shape)
        np.testing.assert_array_equal(h2, fight.summary(bases, a2.hive_queries()))

    def test_h2_is_the_current_regions_h_computed_with_the_hives_pools_and_the_mean_of_its_three_bosses(self):
        fight = models.NumpyFightNet(fight_params(seed=8))
        bases = real_bases(5, seed=4)
        h2 = a2.hive_block(fight, bases)
        per_boss = []
        for boss in (75, 76, 81):
            view = tfm.make_view(act_region=vocab.REGION_HIVE, act_index=1, boss=boss, floor=20)
            per_boss.append(fight.summary(bases, features.fight_queries(view)))
        for b in per_boss[1:]:
            np.testing.assert_allclose(per_boss[0][:, :6], b[:, :6], rtol=0, atol=0)  # weak, normal and elite slots: the pools do not depend on the boss
        np.testing.assert_allclose(np.mean([b[:, 6:] for b in per_boss], axis=0), h2[:, 6:], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(per_boss[0][:, :6], h2[:, :6], rtol=1e-6, atol=1e-7)

    def test_h2_reads_the_1863_wide_row_and_the_stored_parts_the_same_as_the_base(self):
        fight = models.NumpyFightNet(fight_params(seed=9))
        bases = real_bases(5, seed=6)
        rows = np.concatenate([bases, np.full((5, 8), 0.7, np.float32)], axis=1)
        h2 = a2.hive_block(fight, bases)
        np.testing.assert_array_equal(h2, a2.hive_block(fight, rows))
        dense, counts = models.value_inputs(rows)
        np.testing.assert_allclose(h2, a2.hive_block_parts(fight, dense, counts, chunk=2), rtol=1e-6, atol=1e-7)
        with self.assertRaises(ValueError):
            a2.hive_block(fight, bases[:, :1800])

    def test_a_deck_with_more_cards_changes_h2(self):
        fight = models.NumpyFightNet(fight_params())
        h2 = a2.hive_block(fight, real_bases(8, seed=3))
        self.assertGreater(float(np.ptp(h2[:, 1])), 0.0)  # the net reads the deck, so H2 is not one number for every row


class SplitTests(unittest.TestCase):
    def test_our_seeds_split_deterministically_into_train_val_and_report(self):
        seeds = [f"SEED{i:05d}" for i in range(4000)]
        parts = [a2.part_of_seed(s) for s in seeds]
        self.assertEqual(parts, [a2.part_of_seed(s) for s in seeds])
        share = {p: parts.count(p) / len(parts) for p in set(parts)}
        self.assertEqual({a2.PART["train"], a2.PART["val"], a2.PART["report"]}, set(share))
        self.assertAlmostEqual(a2.OURS_REPORT_FRACTION, share[a2.PART["report"]], delta=0.03)
        self.assertAlmostEqual((1 - a2.OURS_REPORT_FRACTION) * a2.VAL_FRACTION, share[a2.PART["val"]], delta=0.03)

    def test_select_returns_one_part_only_and_report_seeds_never_reach_training(self):
        nets = fake_nets()
        records = [fake_record(f"S{i:04d}", final=25 + (i % 12)) for i in range(150)]
        rows = a2.ours_rows(records, nets, with_h2=False)
        train, report = a2.select(rows, "train"), a2.select(rows, "report")
        self.assertTrue(len(train["y"]) > 0 and len(report["y"]) > 0)
        self.assertEqual(set(), {str(s) for s in train["seed"]} & {str(s) for s in report["seed"]})
        self.assertEqual({a2.PART["train"]}, set(train["part"].tolist()))
        for key in ("dense", "counts", "y", "run", "seed"):
            self.assertEqual(len(train["y"]), len(train[key]))

    def test_a_report_seed_never_has_a_training_row(self):
        records = [fake_record(f"S{i:04d}", final=25 + (i % 12), n_decisions=3) for i in range(120)]
        nets = fake_nets()
        rows = a2.ours_rows(records, nets, with_h2=False)
        for seed, part in zip(rows["seed"], rows["part"]):
            self.assertEqual(a2.part_of_seed(str(seed)), int(part))


def fake_nets() -> types.SimpleNamespace:
    return types.SimpleNamespace(base_size=features.BASE_SIZE, fight=models.NumpyFightNet(fight_params()),
                                 value=types.SimpleNamespace(floor_probs=lambda rows: np.zeros((len(rows), 53))))


def fake_record(seed: str, final: int, n_decisions: int = 3, won: bool = False, truncated: bool = False) -> dict:
    """A run record as the self-play shards hold it: a card reward, a rest, a shop and an Act 2 decision, the outcome as given."""
    view = tfm.make_view()
    acts = 3 if won else 2 if final >= 34 else 1 if final >= 18 else 0
    rec = {"seed": seed, "generation": 13, "outcome": {"final_floor": final, "won": won, "acts_cleared": acts, "truncated": truncated},
           "maps": [shards.map_to_json(view)], "decisions": []}

    def entry(i, typ, options, chosen, act_index=0, floor=3):
        v = tfm.make_view(act_index=act_index, floor=floor, hp=60 - i)
        return {"i": i, "type": typ, "floor": floor, "state": shards.view_to_json(v, 0), "options": [shards.option_to_json(o) for o in options], "chosen": chosen}

    cards = [Option("card.take", 0, card=CardRef(472, False)), Option("card.skip", 1)]
    rest = [Option("rest.heal", 0), Option("rest.smith", 1, deck_index=0)]
    shop = [Option("shop.leave", 0), Option("shop.card", 1, card=CardRef(131, False), price=50)]
    rec["decisions"] = [entry(0, "card_reward", cards, 0, floor=3), entry(1, "rest", rest, 1, floor=9), entry(2, "shop", shop, 0, floor=12)][:n_decisions]
    rec["decisions"].append(entry(9, "card_reward", cards, 1, act_index=1, floor=20))  # Act 2: never a row of A2
    rec["decisions"].append({**entry(10, "card_reward", cards, 0, floor=4), "chosen": None})  # nothing chosen: no row
    rec["decisions"].append(entry(11, "event", [Option("event.option", 0), Option("event.option", 1)], 0, floor=5))  # a kind A2 does not read
    return rec


class OurRowsTests(unittest.TestCase):
    def test_a_run_that_died_before_the_act_1_boss_has_no_rows_and_targets_follow_the_outcome(self):
        nets = fake_nets()
        recs = [fake_record("A", final=17), fake_record("B", final=25), fake_record("C", final=34), fake_record("D", final=52, won=True),
                fake_record("E", final=30, truncated=True)]
        rows = a2.ours_rows(recs, nets, with_h2=False)
        by_seed = {}
        for s, y in zip(rows["seed"], rows["y"]):
            by_seed.setdefault(str(s), set()).add(int(y))
        self.assertEqual({"B": {0}, "C": {1}, "D": {1}}, by_seed)
        self.assertEqual({0}, set(rows["src"].tolist()))  # ours is source 0, the one A2 is read at

    def test_only_act_1_chosen_rows_of_the_kinds_the_human_rows_have(self):
        nets = fake_nets()
        rows = a2.ours_rows([fake_record("B", final=25)], nets, with_h2=False)
        self.assertEqual(3, len(rows["y"]))  # the card, the rest and the shop; not Act 2, not the unchosen one, not the event
        self.assertEqual([3, 9, 12], rows["floor"].tolist())
        self.assertEqual([0, 0, 1], rows["door"].astype(int).tolist())  # the last Act 1 decision is the door's
        self.assertEqual(rows["dense"].shape, (3, 350))
        self.assertEqual(rows["counts"].shape, (3, 1513))

    def test_a_row_is_the_chosen_options_afterstate_row_with_the_current_regions_h(self):
        from human_prior_infer import option_rows  # noqa: PLC0415

        nets = fake_nets()
        rec = fake_record("B", final=25)
        rows = a2.ours_rows([rec], nets, with_h2=False)
        entry = rec["decisions"][0]
        dec = shards.entry_decision(entry)
        view = shards.state_view(rec, entry)
        want = option_rows(nets, view, [dec.options[entry["chosen"]]])[0]
        dense, counts = models.value_inputs(want[None, :])
        np.testing.assert_allclose(rows["dense"][0].astype(np.float32), dense[0], atol=2e-3)
        np.testing.assert_array_equal(rows["counts"][0].astype(np.float32), counts[0])

    def test_seeds_known_to_other_blocks_are_left_out(self):
        nets = fake_nets()
        recs = [fake_record("B", final=25), fake_record("C", final=34)]
        rows = a2.ours_rows(recs, nets, with_h2=False, skip_seeds={"C"})
        self.assertEqual({"B"}, {str(s) for s in rows["seed"]})


@unittest.skipIf(torch is None, "torch is needed for the fit")
class NetTests(unittest.TestCase):
    def planted(self, n: int, seed: int, ours_share: float = 0.5, src_shift: float = 1.5):
        """Rows whose Act 2 chance is a known function of four dense numbers and one H2 number, with human rows more likely to win by `src_shift`."""
        rng = np.random.default_rng(seed)
        dense = rng.normal(0, 0.1, (n, 350)).astype(np.float32)  # most columns are near-constant, as the real rows' are
        dense[:, [3, 10, 25]] = rng.normal(0, 1.0, (n, 3))
        counts = np.zeros((n, 1513), np.float32)
        counts[:, :6] = rng.integers(0, 4, (n, 6))
        h2 = rng.uniform(0, 1, (n, 8)).astype(np.float32)
        src = (rng.uniform(size=n) >= ours_share).astype(np.int8)
        logit = 1.2 * dense[:, 3] - 0.9 * dense[:, 10] + 0.6 * dense[:, 25] + 1.0 * (h2[:, 1] - 0.5) * 2 - 0.4 + src_shift * src
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
        run = np.arange(n)
        return {"dense": dense.astype(np.float16), "counts": counts.astype(np.uint8), "h2": h2, "src": src, "y": y, "run": run}, logit

    def test_a_planted_relation_is_recovered_and_the_source_term_is_read_at_ours(self):
        train, _ = self.planted(7000, 1)
        val, _ = self.planted(1500, 2)
        test, test_logit = self.planted(3000, 3)
        params, hist = a2.fit_arrays(train, val, init=None, epochs=6, batch=64, lr=1e-3, weight_decay=1e-4, seed=0, device="cpu")
        self.assertIn("best_epoch", hist)
        self.assertTrue(len(hist["evals"]) >= 2)
        ours = test["src"] == 0
        logit_ours = a2.np_logits(params, test["dense"].astype(np.float32), test["counts"].astype(np.float32), test["h2"], src=np.zeros(len(ours)))
        logit_human = a2.np_logits(params, test["dense"].astype(np.float32), test["counts"].astype(np.float32), test["h2"], src=np.ones(len(ours)))
        planted_ours = test_logit[ours]
        corr = float(np.corrcoef(planted_ours, logit_ours[ours])[0, 1])
        self.assertGreater(corr, 0.9)
        # read at ours (the plain network), ours rows are calibrated: mean prediction equals the observed share; a human reading is higher
        p_ours = 1 / (1 + np.exp(-logit_ours[ours]))
        self.assertAlmostEqual(float(test["y"][ours].mean()), float(p_ours.mean()), delta=0.05)
        self.assertGreater(float((1 / (1 + np.exp(-logit_human[ours]))).mean()), float(p_ours.mean()) + 0.05)
        self.assertGreater(a2.auc(test["y"][ours], logit_ours[ours]), 0.8)

    def test_numpy_inference_equals_torch(self):
        params = a2_params(seed=11, scale=0.04)
        rng = np.random.default_rng(1)
        dense = rng.normal(0, 1, (40, 350)).astype(np.float32)
        counts = np.zeros((40, 1513), np.float32)
        counts[:, :8] = rng.integers(0, 4, (40, 8))
        counts[:, 1154:1160] = rng.integers(0, 2, (40, 6))
        h2 = rng.uniform(0, 1, (40, 8)).astype(np.float32)
        for src in (np.zeros(40), np.ones(40), np.tile([0.0, 1.0], 20)):
            want = a2.torch_logits(params, dense, counts, h2, src)
            got = a2.np_logits(params, dense, counts, h2, src=src)
            np.testing.assert_allclose(want, got, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(a2.np_logits(params, dense, counts, h2), a2.np_logits(params, dense, counts, h2, src=np.zeros(40)), atol=0)

    def test_the_source_column_is_zero_at_start_and_the_init_is_the_value_nets_trunk(self):
        rng = np.random.default_rng(2)
        value = {"embed.deck.weight": rng.normal(size=(64, 1154)).astype(np.float32), "embed.relics.weight": rng.normal(size=(32, 296)).astype(np.float32),
                 "embed.potions.weight": rng.normal(size=(8, 63)).astype(np.float32), "hidden1.weight": rng.normal(size=(256, 454)).astype(np.float32),
                 "hidden1.bias": rng.normal(size=256).astype(np.float32), "hidden2.weight": rng.normal(size=(256, 256)).astype(np.float32),
                 "hidden2.bias": rng.normal(size=256).astype(np.float32)}
        p = a2.init_params(value, seed=0)
        self.assertEqual((256, 350 + 104 + 8), p["hidden1.weight"].shape)
        np.testing.assert_array_equal(value["hidden1.weight"], p["hidden1.weight"][:, :454])
        self.assertEqual(0.0, float(np.abs(p["hidden1.weight"][:, 454:]).max()))  # H2's columns start at zero: the first read is V's trunk alone
        np.testing.assert_array_equal(np.zeros(256, np.float32), p["src_w"])
        np.testing.assert_array_equal(value["hidden2.weight"], p["hidden2.weight"])
        self.assertEqual((1, 256), p["score.weight"].shape)

    def test_an_unfitted_source_term_changes_nothing(self):
        params = a2_params(seed=4)
        params["src_w"][:] = 0.0
        rng = np.random.default_rng(0)
        dense = rng.normal(0, 1, (10, 350)).astype(np.float32)
        counts = np.zeros((10, 1513), np.float32)
        h2 = rng.uniform(0, 1, (10, 8)).astype(np.float32)
        np.testing.assert_array_equal(a2.np_logits(params, dense, counts, h2, src=np.zeros(10)), a2.np_logits(params, dense, counts, h2, src=np.ones(10)))

    def test_average_of_fits_is_the_mean_and_one_fit_is_itself(self):
        p, q = a2_params(seed=1), a2_params(seed=2)
        m = a2.average_params([p, q])
        for k in p:
            np.testing.assert_allclose(m[k], (p[k] + q[k]) / 2, rtol=1e-6, atol=1e-7)
        for k, v in a2.average_params([p]).items():
            np.testing.assert_array_equal(p[k], v)

    def test_the_numpy_log_loss_is_the_one_the_fit_selects_by(self):
        train, _ = self.planted(1500, 1)
        val, _ = self.planted(800, 2)
        params, hist = a2.fit_arrays(train, val, init=None, epochs=2, batch=64, lr=1e-3, weight_decay=1e-4, seed=0, device="cpu", evals_per_epoch=1)
        self.assertAlmostEqual(hist["best_val_loss"], a2.weighted_log_loss(params, val), delta=2e-3)  # the fit reads float16 and float32 inputs alike

    def test_run_weights_make_every_run_count_once(self):
        run = np.array([0, 0, 0, 0, 1, 2, 2])
        src = np.array([0, 0, 0, 0, 0, 1, 1])
        w = a2.run_weights(run, src)
        np.testing.assert_allclose(w[:4].sum(), w[4], rtol=1e-6)  # a run of four rows weighs the same as a run of one, inside a source
        self.assertAlmostEqual(1.0, float(w[src == 0].mean()), places=6)  # mean 1 inside each source
        self.assertAlmostEqual(1.0, float(w[src == 1].mean()), places=6)


@unittest.skipIf(torch is None, "torch is needed for the files")
class ScorerTests(unittest.TestCase):
    def write(self, d: Path, params=None, fight=None):
        params = params or a2_params(seed=7)
        fight = fight or fight_params(seed=4)
        return a2.write_model(d / "a2", params, fight, meta={"note": "synthetic"}), params, fight

    def rows(self, n=12, seed=5):
        bases = real_bases(n, seed=seed)
        return np.concatenate([bases, np.random.default_rng(seed).uniform(0, 1, (n, 8)).astype(np.float32)], axis=1)

    def test_both_files_score_the_same_and_a_probability_is_the_net_at_source_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            files, params, fight_p = self.write(Path(tmp))
            rows = self.rows()
            s_pt, s_npz = a2.Act2Scorer(files["pt"]), a2.Act2Scorer(files["npz"])
            p_pt, p_npz = s_pt.score(rows), s_npz.score(rows)
            np.testing.assert_allclose(p_pt, p_npz, rtol=1e-6, atol=1e-7)
            self.assertTrue(np.all((p_npz > 0) & (p_npz < 1)))
            dense, counts = models.value_inputs(rows)
            h2 = a2.hive_block(models.NumpyFightNet(fight_p), rows)
            want = 1 / (1 + np.exp(-a2.np_logits(params, dense, counts, h2, src=np.zeros(len(rows)))))
            np.testing.assert_allclose(want, p_npz, rtol=1e-5, atol=1e-6)
            self.assertEqual(a2.file_sha256(files["pt"]), s_pt.sha256)
            self.assertEqual(a2.file_sha256(files["npz"]), s_npz.sha256)
            self.assertEqual(64, len(s_pt.sha256))

    def test_the_default_reading_is_ours_and_the_human_reading_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            params = a2_params(seed=7)
            params["src_w"] = (np.random.default_rng(1).normal(size=256) * 0.5).astype(np.float32)
            files, _, _ = self.write(Path(tmp), params=params)
            scorer = a2.Act2Scorer(files["npz"])
            rows = self.rows()
            np.testing.assert_array_equal(scorer.score(rows), scorer.score(rows, source=0))
            self.assertGreater(float(np.abs(scorer.score(rows, source=1) - scorer.score(rows)).max()), 1e-4)
            np.testing.assert_allclose(scorer.logit(rows), np.log(scorer.score(rows) / (1 - scorer.score(rows))), rtol=1e-4, atol=1e-4)

    def test_rows_of_another_width_are_refused_and_a_precomputed_h2_is_used_as_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            files, _, fight_p = self.write(Path(tmp))
            scorer = a2.Act2Scorer(files["npz"])
            rows = self.rows()
            with self.assertRaises(ValueError):
                scorer.score(rows[:, :1855])
            h2 = a2.hive_block(models.NumpyFightNet(fight_p), rows)
            np.testing.assert_allclose(scorer.score(rows), scorer.score(rows, h2=h2), rtol=1e-6, atol=1e-7)
            self.assertEqual(0, len(scorer.score(rows[:0])))

    def test_a_file_that_is_not_an_a2_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "x.npz"
            np.savez(bad, a=np.zeros(2))
            with self.assertRaises(ValueError):
                a2.Act2Scorer(bad)

    def test_the_meta_carries_the_format_and_no_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            files, _, _ = self.write(Path(tmp))
            meta = a2.Act2Scorer(files["npz"]).meta
            self.assertEqual(a2.FORMAT, meta["format"])
            self.assertEqual(a2.H2_SIZE, meta["h2_size"])
            self.assertNotIn("Users", json.dumps(meta))


class MetricsTests(unittest.TestCase):
    def test_auc_known_values(self):
        self.assertAlmostEqual(1.0, a2.auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])))
        self.assertAlmostEqual(0.0, a2.auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])))
        self.assertAlmostEqual(0.5, a2.auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])))
        self.assertTrue(np.isnan(a2.auc(np.array([1, 1]), np.array([0.1, 0.2]))))

    def test_logistic_slope_recovers_a_planted_slope(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1.5, 20000)
        for slope in (0.0, 0.5, 1.0):
            y = (rng.uniform(size=len(x)) < 1 / (1 + np.exp(-(slope * x - 0.3)))).astype(float)
            fit = a2.logistic_slope(x, y)
            self.assertAlmostEqual(slope, fit["slope"], delta=0.06)
            self.assertAlmostEqual(-0.3, fit["intercept"], delta=0.08)

    def test_bootstrap_over_runs_brackets_the_planted_slope_and_the_stop_rule_reads_the_lower_end(self):
        rng = np.random.default_rng(1)
        n = 3000
        x = rng.normal(0, 1.2, n)
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.9 * x - 1.0)))).astype(float)
        run = np.arange(n)
        out = a2.slope_interval(x, y, run, reps=300, seed=0)
        self.assertLess(out["lower"], 0.9)
        self.assertGreater(out["upper"], 0.9)
        self.assertGreater(out["lower"], 0.5)
        self.assertEqual("go", a2.look0_verdict(out["lower"]))
        noise = a2.slope_interval(rng.normal(size=n), y, run, reps=300, seed=0)
        self.assertEqual("stop", a2.look0_verdict(noise["lower"]))
        self.assertEqual("stop", a2.look0_verdict(0.5))  # the rule is a strict lower bound above 0.5
        self.assertEqual(0.5, a2.STOP_SLOPE)

    def test_bootstrap_resamples_whole_runs_not_rows(self):
        # two rows per run with the same logit and outcome: drawing whole runs gives the single-row answer exactly; drawing rows would narrow the interval
        rng = np.random.default_rng(2)
        n = 400
        x1 = rng.normal(0, 1.2, n)
        y1 = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.9 * x1)))).astype(float)
        by_run = a2.slope_interval(np.repeat(x1, 2), np.repeat(y1, 2), np.repeat(np.arange(n), 2), reps=300, seed=0)
        single = a2.slope_interval(x1, y1, np.arange(n), reps=300, seed=0)
        self.assertAlmostEqual(by_run["upper"], single["upper"], delta=1e-3)
        self.assertAlmostEqual(by_run["lower"], single["lower"], delta=1e-3)

    def test_calibration_bins_hide_cells_under_20(self):
        rng = np.random.default_rng(3)
        p = np.concatenate([rng.uniform(0.0, 0.1, 500), rng.uniform(0.9, 1.0, 12)])
        y = (rng.uniform(size=len(p)) < p).astype(float)
        cal = a2.calibration(p, y, bins=10, min_cell=20)
        counts = [c["n"] for c in cal["bins"] if c["n"] is not None]
        self.assertTrue(all(c >= 20 for c in counts))
        self.assertTrue(any(c["n"] is None for c in cal["bins"] if c["lo"] >= 0.9))  # twelve rows in the top bin: not shown
        self.assertIn("ece", cal)
        self.assertIn("brier", cal)

    def test_evaluate_reports_by_source_and_at_the_door_without_any_id(self):
        rng = np.random.default_rng(4)
        n = 600
        logit = rng.normal(0, 1, n)
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
        run = np.repeat(np.arange(n // 3), 3)
        door = np.tile([False, False, True], n // 3)
        ev = a2.evaluate(logit, y, run, door, reps=100, seed=0)
        for key in ("rows", "runs", "door_runs", "auc_rows", "auc_door", "calibration_door", "calibration_rows", "slope_door", "slope_rows"):
            self.assertIn(key, ev)
        self.assertEqual(n // 3, ev["door_runs"])
        self.assertEqual(n, ev["rows"])
        self.assertNotIn("Users", json.dumps(ev, default=str))

    def test_the_markdown_names_the_stop_rule_and_the_verdict(self):
        report = {"written": "2026-11-13T00:00:00Z", "look0a": {"slope": 0.8, "lower": 0.6, "upper": 1.0, "verdict": "go", "door_runs": 100, "kills": 20, "stop_rule": 0.5},
                  "sha256": {"pt": "a" * 64, "npz": "b" * 64}, "counts": {}, "by_source": {}}
        md = a2.render_md(report)
        self.assertIn("0.5", md)
        self.assertIn("go", md)
        self.assertIn("a" * 64, md)


if __name__ == "__main__":
    unittest.main()
