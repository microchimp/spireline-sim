"""The fights inside plain runs as fight-trainer rows (D65 lane R: rl/fight_rows.py, gen_loop.FightTap, run_teacher --combat-rows,
gym_loop_train --extra-rows-root).

The pinned facts: with `--combat-rows none` (the default) nothing a plain job writes changes; with `elite-boss` the run worker keeps the
searched decisions of elite and boss fights as rows (the dataset layout of `combat-g<N>-NNNN.npz`) plus, per row, the fight's kind, its
grade (the stakes multiplier the search ran at) and the simulations the config asked for; the run's RunRecord is the same either way;
the trainer reads such a part beside its own shards, takes the held-out split by run seed, masks the policy loss on rows searched shallower
than the gym's own grade (x16) and keeps the value loss on them, with a weight field per row; the newest rows win the window whatever
their source. Synthetic arrays and a tiny network on the CPU; no game, no card.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fight_rows as fr  # noqa: E402
import gen_loop  # noqa: E402
import gym_loop as gl  # noqa: E402
import gym_loop_train as tr  # noqa: E402
import run_teacher as rt  # noqa: E402
import search_configs  # noqa: E402

from test_gen_loop import _Tok  # noqa: E402
from test_gym_loop_train import fake_shard, make_base, namespace  # noqa: E402

CFG = search_configs.get("a3-v1@h1e4b16")  # the champion's own schedule: hallway x1, elite x4, boss x16


def pos(fight: str, kind: str = "play") -> search_configs.Position:
    return search_configs.Position(kind=kind, n_moves=4, hp_frac=0.9, hard=fight != "hall", fight=fight)


def fake_root():
    return SimpleNamespace(handle=7, obs=None, encounter_id=3)


def fake_res(n_children: int = 3, sims: int = 480):
    kids = [SimpleNamespace(action=a, merged_actions=[], visits=10 * (a + 1), value_mean=0.1 * a) for a in range(n_children)]
    return SimpleNamespace(children=kids, sims=sims, best=kids[-1])


def fake_tap(kinds=("elite", "boss"), cfg=CFG):
    """A tap on a recorder stand-in (the samples and the open fight's facts), with the tokens and the legal mask faked (no native)."""
    rng = np.random.default_rng(3)
    rec = SimpleNamespace(samples=gen_loop.CombatSamples(), fight={"encounter": 5, "deck": 12, "relics": 2, "max_hp": 80, "decisions": 0})
    tap = gen_loop.FightTap(rec, kinds, cfg, encode_fn=lambda root, probe: _Tok(9, rng),
                            legal_fn=lambda root: np.array([1, 1, 1, 1] + [0] * (fr.MAX_ACTIONS - 4), dtype=np.int8))
    return rec, tap


def play_fight(rec, tap, fight: str, decisions: int, end_hp: int, *, forced: int = 0) -> None:
    """What `hindsight.SearchFighter` does around each search of one fight, then the recorder closing the fight."""
    p = pos(fight)
    for i in range(decisions):
        root = fake_root()
        if tap.wants(p):
            toks = tap.encode(root)
            tap.record(root, fake_res(1 if i < forced else 3, CFG.sims_for(p)), toks, p)
    rec.samples.end_fight(end_hp, 80)


class RuleTests(unittest.TestCase):
    def test_a_row_searched_at_the_gyms_grade_is_a_policy_and_value_row_and_a_shallower_one_is_a_value_row(self):
        wp, wv = fr.row_weights(np.array([16, 4, 1, 32]))
        self.assertEqual(wp.tolist(), [1.0, 0.0, 0.0, 1.0])
        self.assertEqual(wv.tolist(), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(wp.dtype, np.float32)

    def test_the_policy_threshold_is_a_parameter_so_a_later_rule_changes_no_format(self):
        wp, _ = fr.row_weights(np.array([16, 4, 1]), min_policy_grade=4)
        self.assertEqual(wp.tolist(), [1.0, 1.0, 0.0])
        wp, _ = fr.row_weights(np.array([16, 4, 1]), min_policy_grade=32)
        self.assertEqual(wp.tolist(), [0.0, 0.0, 0.0])

    def test_the_row_kinds_are_the_fight_kinds_of_the_search_config_in_its_order(self):
        import macro_recipe

        self.assertEqual(len(fr.KIND_NAMES), len(macro_recipe.FIGHT_KINDS))
        self.assertEqual(fr.KIND_NAMES, ("hallway", "elite", "boss"))
        self.assertEqual(fr.KINDS_OF["none"], ())
        self.assertEqual(fr.KINDS_OF["elite-boss"], ("elite", "boss"))
        self.assertEqual(fr.KINDS_OF["all"], ("hall", "elite", "boss"))
        self.assertEqual(fr.GYM_GRADE, 16)


class SamplesTests(unittest.TestCase):
    def test_rows_with_a_grade_carry_kind_grade_and_sims_asked_and_rows_without_keep_the_old_layout(self):
        plain = gen_loop.CombatSamples()
        v = np.zeros(32, np.float32)
        v[:2] = (3, 4)
        legal = np.zeros(32, np.int8)
        legal[:2] = 1
        plain.add(_Tok(9, np.random.default_rng(0)), v, v.copy(), legal, 1, 10, 1, 30)
        plain.end_fight(20, 80)
        old = plain.arrays()
        self.assertEqual(set(old), {"types", "ids", "owners", "feats", "lengths", "labels", "encounters", "deck_sizes", "relic_counts",
                                    "visits", "legal", "returns", "qvals", "sims", "fight_ix", "max_hp"})
        graded = gen_loop.CombatSamples()
        graded.add(_Tok(9, np.random.default_rng(0)), v, v.copy(), legal, 1, 10, 1, 30, kind=2, grade=16, grade_sims=480)
        graded.end_fight(20, 80)
        new = graded.arrays()
        self.assertEqual(set(new) - set(old), {"fight_kind", "grade", "grade_sims"})
        self.assertEqual((new["fight_kind"].tolist(), new["grade"].tolist(), new["grade_sims"].tolist()), ([2], [16], [480]))
        for k in old:  # the shared columns are the same either way
            np.testing.assert_array_equal(old[k], new[k])

    def test_a_part_keeps_the_extra_columns_of_its_runs_and_a_mixed_part_is_refused(self):
        rec, tap = fake_tap()
        play_fight(rec, tap, "elite", 2, 33)
        graded = rec.samples.arrays()
        with tempfile.TemporaryDirectory() as d:
            n = gen_loop.write_part(Path(d) / "combat-g1-0000.npz", [("AAAAAAAAAA", graded), ("BBBBBBBBBB", None), ("CCCCCCCCCC", graded)], 1)
            self.assertEqual(n, 4)
            blob = gen_loop.read_part(Path(d) / "combat-g1-0000.npz")
            self.assertEqual(blob["grade"].tolist(), [4, 4, 4, 4])
            self.assertEqual(blob["fights"].tolist(), [0, 0, 2, 2])  # the run's index in the part: a split by it is a split by run
            self.assertEqual([str(s) for s in blob["seeds"]], ["AAAAAAAAAA", "BBBBBBBBBB", "CCCCCCCCCC"])
            plain = {k: v for k, v in graded.items() if k not in gen_loop.ROW_EXTRA_KEYS}
            with self.assertRaises(ValueError):
                gen_loop.write_part(Path(d) / "mixed.npz", [("A", graded), ("B", plain)], 1)
            gen_loop.write_part(Path(d) / "old.npz", [("A", plain), ("B", None)], 1)
            self.assertFalse(set(gen_loop.read_part(Path(d) / "old.npz")) & set(gen_loop.ROW_EXTRA_KEYS))  # no extra column unless asked


class TapTests(unittest.TestCase):
    def test_a_synthetic_run_with_a_hallway_an_elite_and_a_boss_fight_keeps_the_elite_and_boss_rows(self):
        rec, tap = fake_tap()
        play_fight(rec, tap, "hall", 2, 70)
        play_fight(rec, tap, "elite", 3, 61)
        play_fight(rec, tap, "boss", 4, 0, forced=1)  # the boss kills the player; its first decision was a forced move
        a = rec.samples.arrays()
        self.assertEqual(a["lengths"].size, 6)  # 3 elite rows, 3 boss rows (a forced move teaches nothing); the hallway is not asked for
        self.assertEqual([fr.KIND_NAMES[k] for k in a["fight_kind"]], ["elite"] * 3 + ["boss"] * 3)
        self.assertEqual(a["grade"].tolist(), [4, 4, 4, 16, 16, 16])
        self.assertEqual(a["grade_sims"].tolist(), [120] * 3 + [480] * 3)  # a3-v1's 30 at a plain play, times the kind's multiplier
        self.assertEqual(a["labels"].tolist(), [61] * 3 + [0] * 3)  # the fight's final HP, and 0 for the fight the player died in
        self.assertEqual(a["fight_ix"].tolist(), [1] * 3 + [2] * 3)  # the fight's index in the run (the hallway fight was 0)
        self.assertEqual(a["sims"].tolist(), [120] * 3 + [480] * 3)
        self.assertEqual(a["visits"].shape, (6, fr.MAX_ACTIONS))
        self.assertEqual(a["legal"].sum(axis=1).tolist(), [4] * 6)
        self.assertEqual(int(a["max_hp"][0]), 80)

    def test_all_adds_the_hallway_fights_at_grade_one(self):
        rec, tap = fake_tap(("hall", "elite", "boss"))
        play_fight(rec, tap, "hall", 2, 70)
        play_fight(rec, tap, "boss", 1, 50)
        a = rec.samples.arrays()
        self.assertEqual([fr.KIND_NAMES[k] for k in a["fight_kind"]], ["hallway", "hallway", "boss"])
        self.assertEqual(a["grade"].tolist(), [1, 1, 16])
        self.assertEqual(a["grade_sims"].tolist(), [30, 30, 480])

    def test_a_choice_screen_in_a_fight_is_asked_at_four_times_the_play_budget_and_the_row_says_so(self):
        rec, tap = fake_tap()
        p = pos("boss", "select")
        tap.record(fake_root(), fake_res(3, 1920), tap.encode(fake_root()), p)
        a = rec.samples.arrays()
        self.assertEqual((a["grade"].tolist(), a["grade_sims"].tolist()), ([16], [1920]))  # the grade is the fight's multiplier, the sims what was asked

    def test_the_tap_reads_and_never_counts_a_decision_so_the_run_record_is_the_same(self):
        rec, tap = fake_tap()
        play_fight(rec, tap, "boss", 3, 40)
        self.assertEqual(rec.fight["decisions"], 0)  # the plain job's RunRecord has always held 0 here; recording changes no field of it
        self.assertEqual(tap.stats()["rows"], {"boss": 3})

    def test_the_tap_counts_its_own_seconds_and_the_rows_by_kind(self):
        rec, tap = fake_tap()
        play_fight(rec, tap, "elite", 2, 30)
        play_fight(rec, tap, "boss", 1, 30)
        s = tap.stats()
        self.assertEqual(s["rows"], {"elite": 2, "boss": 1})
        self.assertGreaterEqual(s["seconds"], 0.0)

    def test_a_config_without_a_stakes_schedule_grades_every_row_at_one(self):
        rec, tap = fake_tap(cfg=search_configs.get("a3-v1"))
        play_fight(rec, tap, "boss", 1, 30)
        self.assertEqual(rec.samples.arrays()["grade"].tolist(), [1])


class PartRowsTests(unittest.TestCase):
    @staticmethod
    def plain_part(path: Path, runs=("AAAAAAAAAA", "BBBBBBBBBB", "CCCCCCCCCC")) -> None:
        rng = np.random.default_rng(1)
        arrays = []
        for i, _ in enumerate(runs):
            rec, tap = fake_tap()
            play_fight(rec, tap, "elite", 2, 20 + i)
            play_fight(rec, tap, "boss", 3, 0 if i == 1 else 50)
            arrays.append(rec.samples.arrays())
        gen_loop.write_part(path, list(zip(runs, arrays, strict=True)), 17)

    def test_a_part_becomes_buffer_rows_with_the_loops_layout_and_weights_by_grade(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "combat-g17-0000.npz"
            self.plain_part(path)
            rows = fr.part_to_buffer(gen_loop.read_part(path), is_val=lambda s: s == "BBBBBBBBBB")
        n = 3 * 5
        self.assertEqual(rows["lengths"].size, n)
        for key in ("visits", "legal", "label", "value_ok", "val", "exact", "exact_value", "q_played", "solver_best", "played", "w_policy", "w_value",
                    "src", "grade", "types", "ids", "owners", "feats", "lengths"):
            self.assertIn(key, rows)
        self.assertEqual(rows["types"].dtype, np.int16)
        self.assertEqual(rows["ids"].dtype, np.int32)
        self.assertEqual(rows["visits"].shape, (n, 32))
        self.assertEqual(rows["w_policy"].tolist(), ([0.0] * 2 + [1.0] * 3) * 3)  # elite rows (x4): value only; boss rows (x16): policy too
        self.assertEqual(rows["w_value"].tolist(), [1.0] * n)
        self.assertEqual(set(rows["src"].tolist()), {1})
        self.assertEqual(int(rows["exact"].sum()), 0)  # no solver grade on these rows
        self.assertEqual(rows["solver_best"].tolist(), [-1] * n)
        self.assertEqual(rows["label"].tolist(), [20] * 2 + [50] * 3 + [21] * 2 + [0] * 3 + [22] * 2 + [50] * 3)
        self.assertEqual(rows["value_ok"].tolist(), [1] * n)

    def test_the_held_out_split_is_by_run_seed_so_a_runs_rows_stay_together(self):
        seeds = tuple(f"S{i:09d}" for i in range(40))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "combat-g17-0000.npz"
            self.plain_part(path, seeds)
            blob = gen_loop.read_part(path)
            rows = fr.part_to_buffer(blob, is_val=gl.is_val_seed)
        want = np.repeat([1 if gl.is_val_seed(s) else 0 for s in seeds], 5)
        self.assertEqual(rows["val"].tolist(), want.tolist())
        self.assertTrue(0 < want.sum() < want.size, "the fixed tenth of 40 run seeds holds some and not all")

    def test_a_part_of_runs_without_a_grade_or_rows_is_not_a_fight_row_part(self):
        with tempfile.TemporaryDirectory() as d:
            gen_loop.write_part(Path(d) / "a.npz", [("A", None)], 17)
            self.assertIsNone(fr.part_to_buffer(gen_loop.read_part(Path(d) / "a.npz"), is_val=lambda s: False))
            rec, tap = fake_tap()
            play_fight(rec, tap, "boss", 2, 30)
            plain = {k: v for k, v in rec.samples.arrays().items() if k not in gen_loop.ROW_EXTRA_KEYS}
            gen_loop.write_part(Path(d) / "b.npz", [("A", plain)], 17)  # a self-play part: its search ran at an unknown grade
            self.assertIsNone(fr.part_to_buffer(gen_loop.read_part(Path(d) / "b.npz"), is_val=lambda s: False))

    def test_part_paths_find_the_stores_parts_in_a_generation_folder_or_a_store(self):
        with tempfile.TemporaryDirectory() as d:
            store = Path(d)
            for gen, job, name in ((16, "J1", "combat-g16-0000.npz"), (17, "J2", "combat-g17-0000.npz"), (17, "J2", "combat-g17-0001.tmp.npz"),
                                    (17, "J3-p001", "combat-g17-0000.npz")):
                p = store / str(gen) / "jobs" / job / "shards"
                p.mkdir(parents=True, exist_ok=True)
                (p / name).write_bytes(b"x")
            (store / "17" / "jobs" / "J2" / "shards" / "r0000.jsonl.gz").write_bytes(b"x")
            one = fr.part_paths(store / "17")
            self.assertEqual(sorted(p.parent.parent.name for p in one), ["J2", "J3-p001"])  # the .tmp file and the run shards are not parts
            every = fr.part_paths(store)
            self.assertEqual(len(every), 3)


def write_plain_store(root: Path, parts: list[tuple[str, tuple[str, ...]]], t0: float = 1_700_000_000.0) -> list[Path]:
    """A plain store's generation folder with one part per (job name, its run seeds), each stamped a minute after the last."""
    out = []
    for i, (job, seeds) in enumerate(parts):
        d = root / "jobs" / job / "shards"
        d.mkdir(parents=True, exist_ok=True)
        PartRowsTests.plain_part(d / "combat-g17-0000.npz", seeds)
        os.utime(d / "combat-g17-0000.npz", (t0 + 60 * i, t0 + 60 * i))
        out.append(d / "combat-g17-0000.npz")
    return out


def stamp_gym(path: Path, t: float) -> None:
    os.utime(path, (t, t))


class BufferTests(unittest.TestCase):
    def test_a_plain_store_part_joins_the_gyms_shards_in_one_buffer(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shards = tmp / "buffer" / "shards"
            shards.mkdir(parents=True)
            fake_shard(shards / "s000001-1000.npz", 40, 1)
            store = tmp / "store17"
            write_plain_store(store, [("J1", ("AAAAAAAAAA", "BBBBBBBBBB"))])
            alone = tr.Buffer(shards, 10_000)
            self.assertEqual(alone.refresh(), 40)
            self.assertEqual(set(alone.arrays()["src"].tolist()), {0})
            self.assertEqual(set(alone.arrays()["w_policy"].tolist()), {1.0})  # the gym's rows weigh as they always did
            buf = tr.Buffer(shards, 10_000, extra_roots=[store])
            self.assertEqual(buf.refresh(), 40 + 10)
            flat = buf.arrays()
            self.assertEqual(flat["lengths"].size, 50)
            self.assertEqual(sorted(set(flat["src"].tolist())), [0, 1])
            self.assertEqual(int((flat["src"] == 1).sum()), 10)
            self.assertEqual(flat["offsets"][-1] + flat["lengths"][-1], flat["types"].size)  # the tokens line up across both sources
            self.assertEqual(buf.refresh(), 0)  # nothing new

    def test_the_newest_rows_win_the_window_whatever_their_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shards = tmp / "buffer" / "shards"
            shards.mkdir(parents=True)
            fake_shard(shards / "s000001-1000.npz", 30, 1)
            fake_shard(shards / "s000002-1001.npz", 30, 2)
            stamp_gym(shards / "s000001-1000.npz", 1_700_000_000.0)
            stamp_gym(shards / "s000002-1001.npz", 1_700_000_500.0)
            store = tmp / "store17"
            write_plain_store(store, [("J1", ("AAAAAAAAAA",)), ("J2", ("BBBBBBBBBB",))], t0=1_700_000_100.0)  # between the two gym shards, then later
            buf = tr.Buffer(shards, 30, extra_roots=[store])
            buf.refresh()
            order = [name for name, _ in buf.parts]
            self.assertEqual(buf.rows, sum(int(p["lengths"].size) for _, p in buf.parts))
            self.assertEqual(order, ["s000002-1001.npz"])  # 70 rows arrived, the window is 30: the newest part stays, and it is a gym shard here
            wide = tr.Buffer(shards, 10_000, extra_roots=[store])
            wide.refresh()
            self.assertEqual([n.startswith("x") for n, _ in wide.parts], [False, True, True, False])  # arrival order across the two sources

    def test_extra_parts_are_loaded_newest_first_and_the_ones_the_window_would_drop_are_not_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shards = tmp / "buffer" / "shards"
            shards.mkdir(parents=True)
            store = tmp / "store17"
            seeds = [(f"J{i}", (f"R{i:09d}",)) for i in range(6)]
            write_plain_store(store, seeds)
            buf = tr.Buffer(shards, 12, extra_roots=[store])
            reads = []
            real = np.load

            def counting(path, *a, **k):
                reads.append(str(path))
                return real(path, *a, **k)

            with mock.patch.object(np, "load", counting):
                buf.refresh()
            self.assertLessEqual(len([r for r in reads if "combat-g" in r]), 3)  # 5 rows a part: three parts cover the 12-row window
            names = [n for n, _ in buf.parts]
            self.assertIn("J5", names[-1])  # the newest is kept
            self.assertEqual(buf.refresh(), 0)  # the aged-out parts are not read on the next look either

    def test_a_part_being_written_and_a_part_of_no_rows_are_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shards = tmp / "buffer" / "shards"
            shards.mkdir(parents=True)
            store = tmp / "store17"
            d = store / "jobs" / "J1" / "shards"
            d.mkdir(parents=True)
            (d / "combat-g17-0000.tmp.npz").write_bytes(b"half")
            gen_loop.write_part(d / "combat-g17-0001.npz", [("A" * 10, None)], 17)  # the sink's part of runs with no searched decision
            (d / "combat-g17-0002.npz").write_bytes(b"not an npz")
            buf = tr.Buffer(shards, 1000, extra_roots=[store])
            self.assertEqual(buf.refresh(), 0)
            self.assertEqual(buf.rows, 0)

    def test_the_extra_window_caps_the_rows_taken_from_the_plain_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shards = tmp / "buffer" / "shards"
            shards.mkdir(parents=True)
            fake_shard(shards / "s000001-1000.npz", 30, 1)
            stamp_gym(shards / "s000001-1000.npz", 1_600_000_000.0)  # older than every plain part: the plain rows would push it out
            store = tmp / "store17"
            write_plain_store(store, [(f"J{i}", (f"R{i:09d}",)) for i in range(6)])
            buf = tr.Buffer(shards, 40, extra_roots=[store], extra_window=10)
            buf.refresh()
            flat = buf.arrays()
            self.assertEqual(int((flat["src"] == 1).sum()), 10)  # two plain parts of 5 rows
            self.assertEqual(int((flat["src"] == 0).sum()), 30)  # the gym's rows kept their place

    def test_the_held_out_rows_of_a_run_are_all_held_out_and_the_trainer_never_trains_on_them(self):
        seeds = tuple(f"S{i:09d}" for i in range(40))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shards = tmp / "buffer" / "shards"
            shards.mkdir(parents=True)
            store = tmp / "store17"
            write_plain_store(store, [("J1", seeds)])
            buf = tr.Buffer(shards, 10_000, extra_roots=[store])
            buf.refresh()
            flat = buf.arrays()
            val_runs = {i for i, s in enumerate(seeds) if gl.is_val_seed(s)}
            self.assertTrue(val_runs)
            held = set(flat["val_idx"].tolist())
            for i in range(len(seeds)):
                rows = set(range(5 * i, 5 * i + 5))
                self.assertTrue(rows <= held if i in val_runs else not (rows & held), i)
            self.assertFalse(set(flat["train_idx"].tolist()) & held)


class LossTests(unittest.TestCase):
    def test_the_policy_loss_is_masked_on_a_shallow_row_and_the_value_loss_is_kept(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shards = tmp / "buffer" / "shards"
            shards.mkdir(parents=True)
            store = tmp / "store17"
            write_plain_store(store, [("J1", ("AAAAAAAAAA", "BBBBBBBBBB"))])
            buf = tr.Buffer(shards, 10_000, extra_roots=[store])
            buf.refresh()
            flat = buf.arrays()
            make_base(tmp / "base")
            model, shape = tr.load_net(tmp / "base" / "combat_policy.pt")
            elite = np.flatnonzero((flat["w_policy"] == 0))
            boss = np.flatnonzero((flat["w_policy"] == 1))
            self.assertEqual((elite.size, boss.size), (4, 6))
            inputs, tg = tr.batch_tensors(flat, elite, "cpu", shape["width"], shape["n_buckets"])
            self.assertEqual(tg["has_policy"].tolist(), [0.0] * 4)  # the policy target of an x4 row is not learned from
            self.assertEqual(tg["value_ok"].tolist(), [1.0] * 4)  # its outcome is
            vl, pl, _ = tr.losses(model, inputs, tg)
            self.assertEqual(float(pl), 0.0)
            self.assertGreater(float(vl), 0.0)
            inputs, tg = tr.batch_tensors(flat, boss, "cpu", shape["width"], shape["n_buckets"])
            self.assertEqual(tg["has_policy"].tolist(), [1.0] * 6)
            _, pl, _ = tr.losses(model, inputs, tg)
            self.assertGreater(float(pl), 0.0)

    def test_a_policy_row_still_counts_when_the_flat_has_no_weight_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            shards = Path(tmp) / "shards"
            shards.mkdir()
            fake_shard(shards / "s000001-1000.npz", 20, 1)
            buf = tr.Buffer(shards, 1000)
            buf.refresh()
            flat = {k: v for k, v in buf.arrays().items() if k not in ("w_policy", "w_value", "src", "grade")}
            make_base(Path(tmp) / "base")
            _, shape = tr.load_net(Path(tmp) / "base" / "combat_policy.pt")
            _, tg = tr.batch_tensors(flat, np.arange(5), "cpu", shape["width"], shape["n_buckets"])
            self.assertEqual(tg["has_policy"].tolist(), [1.0] * 5)  # a flat built before the field behaves as it did

    def test_the_weights_scale_a_rows_share_of_the_loss(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shards = tmp / "shards"
            shards.mkdir()
            fake_shard(shards / "s000001-1000.npz", 20, 1)
            buf = tr.Buffer(shards, 1000)
            buf.refresh()
            flat = buf.arrays()
            make_base(tmp / "base")
            model, shape = tr.load_net(tmp / "base" / "combat_policy.pt")
            rows = np.arange(8)
            inputs, tg = tr.batch_tensors(flat, rows, "cpu", shape["width"], shape["n_buckets"])
            full = [float(x) for x in tr.losses(model, inputs, tg)[:2]]
            flat["w_value"] = np.where(np.arange(flat["lengths"].size) < 4, 0.0, 1.0).astype(np.float32)
            _, tg2 = tr.batch_tensors(flat, rows, "cpu", shape["width"], shape["n_buckets"])
            self.assertEqual(tg2["value_ok"][:4].tolist(), [0.0] * 4)
            half = float(tr.losses(model, inputs, tg2)[0])
            self.assertNotEqual(round(half, 6), round(full[0], 6))  # the value loss is now the other four rows' mean


class TrainerTests(unittest.TestCase):
    def test_the_trainer_takes_extra_rows_roots_and_a_checkpoint_says_which_rows_it_saw(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"GYM_LOOP_NO_GPU": "1"}):
            tmp = Path(tmp)
            root = tmp / "loop"
            (root / "buffer" / "shards").mkdir(parents=True)
            fake_shard(root / "buffer" / "shards" / "s000001-1000.npz", 200, 1)
            make_base(tmp / "base")
            store = tmp / "store17"
            write_plain_store(store, [("J1", tuple(f"S{i:09d}" for i in range(40)))])
            t = tr.Trainer(namespace(root, tmp / "base", extra_rows_root=[str(store)], max_epochs=1.0))
            t.refresh()
            self.assertEqual(sorted(set(t.buffer.arrays()["src"].tolist())), [0, 1])
            self.assertGreater(t.st["budget"], 0)
            n = t.train_chunk("cpu")
            self.assertGreater(n, 0)
            name = t.checkpoint()
            info = json.loads((root / "checkpoints" / name / "train.json").read_text(encoding="utf-8"))
            self.assertIn("plain", info["split"])  # another held-out set than the gate's: the pre-filter scores the gym's own rows itself
            self.assertNotEqual(info["split"], gl.val_split_id())
            self.assertEqual(info["extra_rows"]["rows"], int((t.buffer.arrays()["src"] == 1).sum()))
            self.assertEqual(info["extra_rows"]["policy_rows"], 120)  # 40 runs x 3 boss rows
            self.assertEqual(info["extra_rows"]["roots"], 1)

    def test_without_the_flag_the_trainer_reads_what_it_did_and_its_split_is_the_gates(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"GYM_LOOP_NO_GPU": "1"}):
            tmp = Path(tmp)
            root = tmp / "loop"
            (root / "buffer" / "shards").mkdir(parents=True)
            fake_shard(root / "buffer" / "shards" / "s000001-1000.npz", 200, 1)
            make_base(tmp / "base")
            t = tr.Trainer(namespace(root, tmp / "base", max_epochs=1.0))
            t.refresh()
            t.train_chunk("cpu")
            name = t.checkpoint()
            info = json.loads((root / "checkpoints" / name / "train.json").read_text(encoding="utf-8"))
            self.assertEqual(info["split"], gl.val_split_id())
            self.assertNotIn("extra_rows", info)

    def test_the_command_line_takes_the_flag_more_than_once(self):
        a = tr.trainer_parser().parse_args(["--root", "R", "--base", "B", "--extra-rows-root", "S1", "--extra-rows-root", "S2",
                                            "--extra-window", "50000", "--policy-min-grade", "4"])
        self.assertEqual(a.extra_rows_root, ["S1", "S2"])
        self.assertEqual((a.extra_window, a.policy_min_grade), (50000, 4))
        d = tr.trainer_parser().parse_args(["--root", "R", "--base", "B"])
        self.assertEqual((d.extra_rows_root, d.extra_window, d.policy_min_grade), ([], 0, 16))


class TeacherJobTests(unittest.TestCase):
    def test_the_flag_defaults_to_none_and_takes_three_values(self):
        for cmd in ("job", "label"):
            self.assertEqual(rt.build_parser().parse_args([cmd]).combat_rows, "none")
        for v in ("none", "elite-boss", "all"):
            self.assertEqual(rt.build_parser().parse_args(["job", "--combat-rows", v]).combat_rows, v)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            rt.build_parser().parse_args(["job", "--combat-rows", "boss"])

    def test_a_runs_files_name_a_combat_file_beside_the_record(self):
        f = rt.RunFiles("G", "SEED")
        self.assertEqual(f.combat.name, "SEED.combat.npz")
        self.assertEqual(f.combat.parent, f.record.parent)

    @staticmethod
    def ctx() -> dict:
        return {"generation": 17, "identity": {"agent": "gen17/x", "agent_sha256": "0" * 64}, "judge": None, "lib": None, "tree": "abc", "k": 2, "k_max": 4,
                "max_options": 4, "cal_doc": {}}

    def finished_run(self, gen_dir: Path, seed: str, *, rows: bool) -> None:
        """What a worker leaves for a run that ended: its row in runs.jsonl, its RunRecord and (when asked) its combat file."""
        rec, tap = fake_tap()
        play_fight(rec, tap, "elite", 2, 40)
        play_fight(rec, tap, "boss", 3, 0)
        files = rt.RunFiles(gen_dir, seed)
        files.recs.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            rt.write_combat_rows(files, rec.samples.arrays())
        record = {"schema": 1, "seed": seed, "generation": 17, "policy": {}, "outcome": {"won": False}, "decisions": [], "fights": []}
        rt.write_json(files.record, record)
        with open(gen_dir / "runs.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"_task": ["rt-new", seed, 1], "seed": seed, "source": "new", "end": "death", "teacher_screens": 0}) + "\n")

    def consolidate(self, gen_dir: Path, name: str):
        agent = SimpleNamespace(generation=17, search_config=lambda n: SimpleNamespace(sims=30, c=0.2, name="a3-v1@h1e4b16"),
                                describe=lambda: {"generation": 17})
        return rt.consolidate(gen_dir, name, self.ctx(), agent)

    def test_consolidate_puts_a_runs_rows_in_the_combat_part_and_removes_its_combat_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen_dir = Path(tmp) / "17"
            gen_dir.mkdir()
            for seed in ("RTROWS0001", "RTROWS0002"):
                self.finished_run(gen_dir, seed, rows=True)
            self.consolidate(gen_dir, "J1")
            parts = fr.part_paths(gen_dir)
            self.assertEqual(len(parts), 1)
            blob = gen_loop.read_part(parts[0])
            self.assertEqual(blob["grade"].tolist(), [4, 4, 16, 16, 16] * 2)
            self.assertEqual([str(s) for s in blob["seeds"]], ["RTROWS0001", "RTROWS0002"])
            self.assertEqual(blob["fights"].tolist(), [0] * 5 + [1] * 5)
            man = json.loads((parts[0].parent / "shards.json").read_text(encoding="utf-8"))
            self.assertEqual((man["runs"], man["samples"]), (2, 10))
            self.assertEqual(man["combat_rows"], "kept")  # the manifest says the part holds fight rows
            self.assertFalse(rt.RunFiles(gen_dir, "RTROWS0001").combat.exists())  # the per-run file is in the part now
            self.assertTrue(rt.RunFiles(gen_dir, "RTROWS0001").record.exists())

    def test_a_job_that_recorded_no_rows_writes_the_part_and_the_manifest_it_always_did(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen_dir = Path(tmp) / "17"
            gen_dir.mkdir()
            self.finished_run(gen_dir, "RTROWS0003", rows=False)
            self.consolidate(gen_dir, "J1")
            (part,) = fr.part_paths(gen_dir) if fr.part_paths(gen_dir) else gen_dir.glob("jobs/J1/shards/combat-g17-0000.npz")
            blob = gen_loop.read_part(part)
            self.assertEqual(sorted(blob), ["generation", "n_runs", "prior_path", "seeds"])  # exactly the keys a plain part has today
            man = json.loads((part.parent / "shards.json").read_text(encoding="utf-8"))
            self.assertNotIn("combat_rows", man)
            self.assertEqual(man["samples"], 0)



NATIVE = (ROOT / "out" / "Sts2Emulator.dll").exists() or bool(os.environ.get("STS2_LIB_PATH") and (Path(os.environ["STS2_LIB_PATH"]) / "Sts2Emulator.dll").exists())


@unittest.skipUnless(NATIVE, "the native library of this tree is not built")
class NativeJobRowsTests(unittest.TestCase):
    """`new_run` on the real engine with a stand-in fighter that calls the tap as `hindsight.SearchFighter` does (a fake boss row per combat
    step): the worker's plumbing, the files it leaves and the RunRecord it writes with the flag on and off. The search itself is the
    measurement's, not this test's."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")
        from run_train import engine, selfplay

        import hindsight

        cls.engine, cls.hindsight = engine, hindsight

        class Policy:
            def __init__(self):
                self.rules = selfplay.RulesPlayer()

            def pick(self, view, d, rng):
                return self.rules.pick(view, d, np.random.default_rng(0))

            def describe(self):
                return {"kind": "rules"}

        class Fighter:
            cfg = CFG
            probe = False

            def __call__(self, s, rng, deadline=None, tap=None):
                from sts2_gym import native

                made = 0
                while s.phase == engine.rc.PHASE_COMBAT:
                    if tap is not None and tap.wants(pos("boss")):
                        tap.record(fake_root(), fake_res(), _Tok(9, np.random.default_rng(made)), pos("boss"))
                    native.run_debug_resolve_combat(s.handle, s.env._run_obs_buf)
                    made += 1
                    if made > 3:
                        break
                return made

        agent = SimpleNamespace(generation=0, prior_dir=None, search_config=lambda name: SimpleNamespace(sims=4, c=0.1, name="stand-in"))
        cls.player = hindsight.Player(policy=Policy(), play_fight=Fighter(), head=lambda v, d: None, judge=lambda handle, max_hp: {"p_death": 0.5},
                                      agent=agent)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        saved = dict(rt._J)
        self.addCleanup(lambda: (rt._J.clear(), rt._J.update(saved)))
        rt._J.clear()
        rt._J.update(deadline=time.time() + 3600, stop_file="", ledger=str(self.tmp / "ledger.jsonl"), job="J", budget_h=1000.0, gen_dir=self.tmp, k=2,
                     k_max=4, max_options=4, cal_map={}, player=self.player, kinds=(), est_new_s=10.0, stop_floor=0)
        engine, hindsight = self.engine, self.hindsight
        for patch in (mock.patch.object(hindsight, "session_for", lambda seed, agent, config="a3-v1", ascension=10, own_schedule=False:
                                        engine.RunSession(seed, 10, engine.CombatBudget(4, 4))),
                      mock.patch("sts2_gym.native.valid_actions", lambda handle, n: [1, 1, 1, 1] + [0] * (n - 4))):
            patch.start()
            self.addCleanup(patch.stop)

    @staticmethod
    def untimed(path: Path) -> dict:
        rec = json.loads(path.read_text(encoding="utf-8"))
        rec["stats"].pop("seconds", None)
        for f in rec["fights"]:
            f.pop("seconds", None)
        return rec

    def test_the_flag_off_leaves_no_combat_file_and_on_leaves_one_and_the_run_record_is_the_same(self):
        rt._J["combat_rows"] = "none"
        off = rt.new_run(("rt-new", "RTNATIVE01", 5))
        files = rt.RunFiles(self.tmp, "RTNATIVE01")
        self.assertFalse(files.combat.exists())
        self.assertNotIn("fight_rows", off)
        if not files.record.is_file():
            self.skipTest("the stand-in run did not end")
        rec_off = self.untimed(files.record)
        shutil.rmtree(self.tmp / "decisions.d")
        rt._J["combat_rows"] = "elite-boss"
        on = rt.new_run(("rt-new", "RTNATIVE01", 5))
        self.assertEqual((off["end"], off["floor"], off["decisions"]), (on["end"], on["floor"], on["decisions"]))  # recording changes nothing the run does
        self.assertEqual(rec_off, self.untimed(files.record))  # nor the RunRecord
        self.assertTrue(files.combat.is_file())
        self.assertGreater(on["fight_rows"]["boss"], 0)
        self.assertGreaterEqual(on["rows_s"], 0.0)
        part = rt.read_combat_rows(files)
        self.assertEqual(set(part["grade"].tolist()), {16})
        self.assertEqual(int(part["lengths"].size), on["fight_rows"]["boss"])


if __name__ == "__main__":
    unittest.main()
