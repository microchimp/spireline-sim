"""The boss chain (Act 3 week, lane 3b; rl/boss_chain.py, rl/imitation_chooser.py, rl/macro_recipe.py, rl/gen_loop.py, rl/gen0.py).

Pinned: the outcome rule (`passed`); which decisions give rows (Act 2 up to floor 32 for runs past the Act 1 boss, at most 12 a run and the first always; Act 3 up to floor
47; nothing at floor 48 or later, nothing unscorable or unchosen) and what each row's two targets and masks say; the parts (only generation 18 reports, one seed always one part);
the run weights of a masked head; the numpy and torch forward passes agree and H3's columns start at zero; a planted relation is recovered; `chain_terms` for every act and
floor; the scorer reads the offset at generation 18 and clips; a chain fitted with another FightNet is refused; the chooser with no chain (or kappa 0, or an Act 1 screen, or a
floor past 47) picks, records and replays exactly what the unmodified chooser did on forty stored screens, and with a chain adds to the value of each option on the rest,
card, student and plain screens; the agent block and the recipe's `+chain` part; the candidate builder; the emulator arm's identity; the report, the verdict and the
flip check's counts. Synthetic rows and hand-built screens throughout, the real judgement and prior weights where the files are present.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gzip
import io
import json
import math
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import act2_chance as a2  # noqa: E402
import boss_chain as bc  # noqa: E402
import gen0  # noqa: E402
import gen_loop  # noqa: E402
import human_prior_infer as hpi  # noqa: E402
import imitation_chooser as ic  # noqa: E402
import macro_recipe as mr  # noqa: E402
import screens  # noqa: E402
from boss_chain_screens import act23_screens, card_screen, rest_screen, shop_screen, view_of  # noqa: E402
from run_train import FEATURE_VERSION  # noqa: E402
from run_train import features as F  # noqa: E402
from run_train import models, shards, vocab  # noqa: E402
from run_train.features import CardRef, Option  # noqa: E402
from run_train.policy import LookaheadPolicy  # noqa: E402

from tests.python.test_gen_loop import student_world  # noqa: E402
from tests.python.test_imitation_chooser import REAL_JUDGE, REAL_WEIGHTS  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover
    torch = None

REAL_OK = REAL_WEIGHTS.is_file() and REAL_JUDGE.is_file() and torch is not None

# What the unmodified chooser (the fork's main before the chain) chose on the forty screens of `act23_screens` (Act 1 at floors 5 and 12, Act 2 from its first decision to the
# boss, Act 3 from its first to floor 47, and floor 48), and the value it gave each option (4 places): [index, values, options], per policy. Made by running those screens
# through the unmodified module; with no chain the policy must still give exactly these.
GOLDEN = {
    "student": [
        [0, [16.2654, 15.1232, 14.0654], 3],
        [2, [13.5775, 13.5455, 13.7676], 3],
        [0, [13.7676, 13.6131, 13.6705], 3],
        [1, [13.1581, 13.7464], 2],
        [0, [14.0493, 11.4169, 10.9027], 3],
        [2, [10.5817, 10.5627, 10.7182], 3],
        [0, [10.7182, 10.6262, 10.6731], 3],
        [1, [10.2698, 10.4993], 2],
        [0, [25.1762, 24.949, 24.8981], 3],
        [1, [24.8456, 24.8468, 24.8964], 3],
        [1, [24.8964, 24.8897, 24.9102], 3],
        [1, [24.7256, 24.7787], 2],
        [0, [24.7045, 24.3987, 24.3593], 3],
        [1, [24.3222, 24.3223, 24.3531], 3],
        [2, [24.3531, 24.361, 24.3788], 3],
        [1, [24.1896, 24.2393], 2],
        [0, [24.9061, 24.8241, 24.7565], 3],
        [1, [24.727, 24.725, 24.7684], 3],
        [2, [24.7684, 24.7669, 24.786], 3],
        [1, [24.6124, 24.6583], 2],
        [0, [24.7016, 24.4285, 24.3858], 3],
        [1, [24.3596, 24.3598, 24.3904], 3],
        [2, [24.3904, 24.3982, 24.4151], 3],
        [1, [24.247, 24.2877], 2],
        [1, [24.5823, 25.8481, 24.8998], 3],
        [2, [23.8827, 23.906, 24.5823], 3],
        [0, [24.5823, 23.8436, 23.8086], 3],
        [1, [23.4418, 23.8837], 2],
        [0, [22.9749, 22.82, 22.0296], 3],
        [2, [21.291, 21.27, 21.7825], 3],
        [0, [21.7825, 21.1926, 21.1733], 3],
        [1, [20.7051, 21.1458], 2],
        [0, [24.9688, 24.7161, 23.9921], 3],
        [2, [23.3411, 23.3468, 23.7318], 3],
        [0, [23.7318, 23.2987, 23.2736], 3],
        [1, [22.7948, 23.1864], 2],
        [0, [24.6096, 24.4188, 23.6799], 3],
        [2, [23.0539, 23.058, 23.434], 3],
        [0, [23.434, 23.0088, 22.9854], 3],
        [1, [22.5237, 22.9307], 2],
    ],
    "plain": [
        [0, [16.2654, 15.1232, 14.0654], 3],
        [2, [13.5775, 13.5455, 13.7676], 3],
        [0, [13.7676, 13.6131, 13.6705], 3],
        [1, [13.1581, 13.7464], 2],
        [0, [14.0493, 11.4169, 10.9027], 3],
        [2, [10.5817, 10.5627, 10.7182], 3],
        [0, [10.7182, 10.6262, 10.6731], 3],
        [1, [10.2698, 10.4993], 2],
        [0, [25.1762, 24.949, 24.8981], 3],
        [1, [24.8456, 24.8468, 24.8964], 3],
        [2, [24.8964, 24.8897, 24.9102], 3],
        [1, [24.7256, 24.7787], 2],
        [0, [24.7045, 24.3987, 24.3593], 3],
        [1, [24.3222, 24.3223, 24.3531], 3],
        [2, [24.3531, 24.361, 24.3788], 3],
        [1, [24.1896, 24.2393], 2],
        [0, [24.9061, 24.8241, 24.7565], 3],
        [1, [24.727, 24.725, 24.7684], 3],
        [2, [24.7684, 24.7669, 24.786], 3],
        [1, [24.6124, 24.6583], 2],
        [0, [24.7016, 24.4285, 24.3858], 3],
        [1, [24.3596, 24.3598, 24.3904], 3],
        [2, [24.3904, 24.3982, 24.4151], 3],
        [1, [24.247, 24.2877], 2],
        [1, [24.5823, 25.8481, 24.8998], 3],
        [2, [23.8827, 23.906, 24.5823], 3],
        [0, [24.5823, 23.8436, 23.8086], 3],
        [1, [23.4418, 23.8837], 2],
        [0, [22.9749, 22.82, 22.0296], 3],
        [2, [21.291, 21.27, 21.7825], 3],
        [0, [21.7825, 21.1926, 21.1733], 3],
        [1, [20.7051, 21.1458], 2],
        [0, [24.9688, 24.7161, 23.9921], 3],
        [2, [23.3411, 23.3468, 23.7318], 3],
        [0, [23.7318, 23.2987, 23.2736], 3],
        [1, [22.7948, 23.1864], 2],
        [0, [24.6096, 24.4188, 23.6799], 3],
        [2, [23.0539, 23.058, 23.434], 3],
        [0, [23.434, 23.0088, 22.9854], 3],
        [1, [22.5237, 22.9307], 2],
    ],
}
STUDENT_LAMS = {"route": (0.3715, 2.0, 2.0), "shop": (0.3038, 0.0, 0.0), "card": (0.5, 0.5, 0.5)}
BASE = "vrun+pi_h:lambda=0.5:pi_h=aaaaaaaa"
STAKES = "a3-v1@h1e4b16"
SHA = "0123456789abcdef" * 4


# ------------------------------------------------------------------------------------------------------------------ fixtures


def fight_params(seed: int = 3, scale: float = 0.08) -> dict[str, np.ndarray]:
    """A FightNet's weights (the numpy net's keys and shapes) at random: what H and H3 are computed from in these tests."""
    rng = np.random.default_rng(seed)
    shapes = {"embed.deck.weight": (64, 1154), "embed.relics.weight": (32, 296), "embed.potions.weight": (8, 63),
              "hidden1.weight": (256, models.FIGHT_DENSE + 104), "hidden1.bias": (256,), "hidden2.weight": (128, 256), "hidden2.bias": (128,),
              "out.weight": (22, 128), "out.bias": (22,)}
    return {k: (rng.standard_normal(s) * scale).astype(np.float32) for k, s in shapes.items()}


def fake_value(seed: int = 2) -> dict[str, np.ndarray]:
    """A run brain's value arrays (the trunk's keys and shapes) at random."""
    rng = np.random.default_rng(seed)
    return {"embed.deck.weight": rng.normal(size=(64, 1154)).astype(np.float32), "embed.relics.weight": rng.normal(size=(32, 296)).astype(np.float32),
            "embed.potions.weight": rng.normal(size=(8, 63)).astype(np.float32), "hidden1.weight": rng.normal(size=(256, 454)).astype(np.float32),
            "hidden1.bias": rng.normal(size=256).astype(np.float32), "hidden2.weight": rng.normal(size=(256, 256)).astype(np.float32),
            "hidden2.bias": rng.normal(size=256).astype(np.float32)}


def fake_nets() -> types.SimpleNamespace:
    return types.SimpleNamespace(base_size=F.BASE_SIZE, fight=models.NumpyFightNet(fight_params()),
                                 value=types.SimpleNamespace(floor_probs=lambda rows: np.full((len(rows), 53), 1 / 53)))


def chain_params(seed: int = 5, scale: float = 0.05) -> dict[str, np.ndarray]:
    """The chain's weights, every layer random (the head and the offsets too)."""
    rng = np.random.default_rng(seed)
    p = bc.init_params(None, seed)
    for k in ("embed.deck.weight", "embed.relics.weight", "embed.potions.weight", "hidden1.weight", "hidden2.weight", "head.weight"):
        p[k] = (rng.standard_normal(p[k].shape) * scale).astype(np.float32)
    p["head.bias"] = rng.normal(size=2).astype(np.float32)
    p["gen_off"] = rng.normal(size=p["gen_off"].shape).astype(np.float32)
    return p


def fake_record(seed: str, final: int, *, gen: int = 18, won: bool = False, n2: int = 3, n3: int = 3, truncated: bool = False, low_hp_rest: bool = False) -> dict:
    """A run record as the self-play shards hold it: Act 1 decisions (card, rest, shop), `n2` Act 2 ones when the run got past the Act 1 boss and `n3` Act 3 ones when it got
    past the Act 2 boss, a floor 33 Act 2 one and a floor 48 Act 3 one after the bosses, one with nothing chosen and one whose chosen option does not score."""
    view0 = view_of(F, vocab, 0, 3, 60)
    cards = [Option("card.take", 0, card=CardRef(472, False)), Option("card.skip", 1)]
    rest = [Option("rest.heal", 0), Option("rest.smith", 1, deck_index=0)]
    shop = [Option("shop.leave", 0), Option("shop.card", 1, card=CardRef(131, False), price=50)]
    acts = 3 if won else 2 if final >= 34 else 1 if final >= 18 else 0
    rec = {"seed": seed, "generation": gen, "ascension": 10, "feature_version": FEATURE_VERSION, "combat": {"budget": "30"},
           "outcome": {"final_floor": final, "won": won, "acts_cleared": acts, "truncated": truncated}, "maps": [shards.map_to_json(view0)], "decisions": []}

    def entry(typ, options, chosen, act, floor, hp=60):
        v = view_of(F, vocab, act, floor, hp)
        return {"i": len(rec["decisions"]), "type": typ, "floor": floor, "state": shards.view_to_json(v, 0), "options": [shards.option_to_json(o) for o in options], "chosen": chosen}

    cycle = [("card_reward", cards), ("rest", rest), ("shop", shop)]
    for k, floor in enumerate((3, 9, 12)):
        typ, opts = cycle[k]
        rec["decisions"].append(entry(typ, opts, 0, 0, floor))
    if final >= 18:
        for k in range(n2):
            typ, opts = cycle[k % 3]
            rec["decisions"].append(entry(typ, opts, k % 2, 1, 17 + (k % 16), hp=40 + (k % 30)))
        if final >= 34:
            rec["decisions"].append(entry("card_reward", cards, 0, 1, 33))  # after the Act 2 boss: no row
    if final >= 34:
        for k in range(n3):
            typ, opts = cycle[k % 3]
            hp = 30 if (low_hp_rest and typ == "rest") else 55
            rec["decisions"].append(entry(typ, opts, 0, 2, (33, 38, 44, 47)[k % 4], hp=hp))
        if final >= 48:
            rec["decisions"].append(entry("rest", rest, 0, 2, 48))  # after the first Act 3 boss: no row
    rec["decisions"].append({**entry("card_reward", cards, 0, 0, 4), "chosen": None})  # nothing chosen: no row
    unscored = [Option("card.take", 0, card=CardRef(472, False), scorable=False), Option("card.skip", 1)]
    rec["decisions"].append(entry("card_reward", unscored, 0, 0, 5))  # the chosen option does not score: no row
    return rec


def make_chain_dir(root: Path, judge: Path, *, scale: float = 0.05, other_fight: bool = False, verdict: str = "pass") -> Path:
    """A `boss_chain.py fit` OUT: boss_chain.npz and .pt, fit.json and report.json, fitted (in name) with `judge`'s FightNet (or another one)."""
    root.mkdir(parents=True, exist_ok=True)
    fight = a2.fight_arrays(judge)
    if other_fight:
        fight = {**fight, "out.bias": fight["out.bias"] + 1.0}
    bc.write_model(root / "boss_chain", chain_params(seed=7, scale=scale), fight, {"q4": {"runs_past_b3a": 100, "wins": 4, "q4": 5 / 102}, "counts": {}})
    (root / "fit.json").write_text(json.dumps({"q4": {"runs_past_b3a": 100, "wins": 4, "q4": 5 / 102}, "fit": {"kept_seed": 1}}), encoding="utf-8")
    (root / "report.json").write_text(json.dumps({"verdict": {"verdict": verdict, "band": "go", "door_auc": 0.74}}), encoding="utf-8")
    return root


# ------------------------------------------------------------------------------------------------------------------ outcomes, parts, constants


class OutcomeTests(unittest.TestCase):
    def test_passed_reads_the_final_floor(self):
        self.assertEqual((False, False, False, False), bc.passed(17))  # died on the Act 1 boss
        self.assertEqual((True, False, False, False), bc.passed(18))
        self.assertEqual((True, False, False, False), bc.passed(33))  # died on the Act 2 boss
        self.assertEqual((True, True, False, False), bc.passed(34))
        self.assertEqual((True, True, False, False), bc.passed(48))  # died on the first Act 3 boss
        self.assertEqual((True, True, True, False), bc.passed(49))
        self.assertEqual((True, True, True, True), bc.passed(53))  # won
        self.assertEqual((True, True, True, True), bc.passed(30, won=True))

    def test_the_constants_agree_with_the_rest_of_the_code(self):
        self.assertEqual(vocab.REGION_GLORY, bc.GLORY_REGION)
        self.assertEqual(mr.ACT_LAST_FLOORS[0] + 1, bc.PASS_B1)
        self.assertEqual(mr.ACT_LAST_FLOORS[1] + 1, bc.PASS_B2)
        self.assertEqual(bc.LAST_PRE_B2, mr.ACT_LAST_FLOORS[1] - 1)
        self.assertEqual(bc.LAST_PRE_B3A, ic.LAST_CHAIN_FLOOR)
        self.assertEqual(models.FLOOR_CLASSES, bc.WON)
        self.assertEqual((0.0, 10.0, 10.0), bc.KAPPA)
        self.assertEqual(350 + 104 + 8, bc.INPUT_WIDTH)

    def test_the_module_never_opens_the_corpus_and_writes_no_seed_array(self):
        text = Path(bc.__file__).read_text(encoding="utf-8")
        self.assertNotIn("spire-codex", text)
        self.assertNotIn("EXAM", text)

    def test_the_chosen_scorable_rule_is_the_trainers(self):
        from run_train import train

        for chosen, opts in ((0, [{"kind": "a"}, {"kind": "b"}]), (1, [{"kind": "a"}, {"kind": "b", "scorable": False}]), (0, [{"kind": "a", "scorable": False}])):
            entry = {"options": opts, "chosen": chosen}
            self.assertEqual(train._chosen_scorable(entry), bc._chosen_scorable(entry))  # noqa: SLF001


class PartTests(unittest.TestCase):
    def test_only_generation_18_reports_and_a_seed_always_has_one_part(self):
        seeds = [f"SEED{i:05d}" for i in range(4000)]
        for g in (15, 16, 17):
            self.assertNotIn(bc.PART["report"], {bc.part_of(g, s) for s in seeds})
        parts = [bc.part_of(18, s) for s in seeds]
        self.assertEqual(parts, [bc.part_of(18, s) for s in seeds])
        share = {p: parts.count(p) / len(parts) for p in set(parts)}
        self.assertEqual({bc.PART["train"], bc.PART["val"], bc.PART["report"]}, set(share))
        self.assertAlmostEqual(bc.REPORT_FRACTION, share[bc.PART["report"]], delta=0.03)
        self.assertAlmostEqual((1 - bc.REPORT_FRACTION) * bc.VAL_FRACTION, share[bc.PART["val"]], delta=0.03)
        other = [bc.part_of(17, s) for s in seeds]
        self.assertAlmostEqual(bc.VAL_FRACTION, other.count(bc.PART["val"]) / len(other), delta=0.03)

    def test_a_report_seed_never_has_a_training_row(self):
        recs = [fake_record(f"S{i:04d}", final=40 + (i % 10)) for i in range(120)]
        rows = bc.chain_rows(recs, fake_nets())
        train, report = bc.select(rows, "train"), bc.select(rows, "report")
        self.assertTrue(len(train["y2"]) > 0 and len(report["y2"]) > 0)
        self.assertEqual({bc.PART["train"]}, set(train["part"].tolist()))
        self.assertEqual({bc.PART["report"]}, set(report["part"].tolist()))
        self.assertEqual(set(), set(train["run"].tolist()) & set(report["run"].tolist()))  # a run is in one part


# ------------------------------------------------------------------------------------------------------------------ rows


class RowTests(unittest.TestCase):
    def rows_of(self, records, **kw):
        return bc.chain_rows(records, fake_nets(), **kw)

    def test_a_run_that_died_on_the_act_1_boss_has_no_rows_and_act_1_never_has_any(self):
        stats: dict = {}
        rows = self.rows_of([fake_record("A", final=17)], stats=stats)
        self.assertEqual(0, len(rows["y2"]))
        self.assertEqual({"records": 1}, stats)

    def test_act_2_rows_belong_to_runs_past_the_act_1_boss_with_the_targets_of_the_outcome(self):
        rows = self.rows_of([fake_record("B", final=25, n2=3)])
        self.assertEqual({1}, set(rows["act"].tolist()))
        self.assertEqual(3, len(rows["y2"]))
        self.assertEqual([1, 1, 1], rows["m2"].tolist())
        self.assertEqual([0, 0, 0], rows["y2"].tolist())  # died on the way to the Act 2 boss
        self.assertEqual([0, 0, 0], rows["m3"].tolist())  # q3 is conditional on getting past the Act 2 boss: no row of it here
        self.assertTrue(int(rows["floor"].max()) <= bc.LAST_PRE_B2)
        died_on_boss = self.rows_of([fake_record("B2", final=33, n2=2)])
        self.assertEqual([0, 0], died_on_boss["y2"].tolist())
        past = self.rows_of([fake_record("C", final=40, n2=2, n3=2)])
        act2 = past["act"] == 1
        self.assertEqual([1, 1], past["y2"][act2].tolist())
        self.assertEqual([1, 1], past["m3"][~act2].tolist())  # the Act 3 rows are q3's too
        self.assertEqual([1, 1], past["m3"][act2].tolist())  # m3 = past the Act 2 boss
        self.assertEqual([0, 0], past["y3"][act2].tolist())  # died on the first Act 3 boss's way

    def test_act_3_rows_are_floors_up_to_47_with_q3_only_and_floor_48_and_after_are_left_out(self):
        rows = self.rows_of([fake_record("D", final=52, won=True, n2=2, n3=4)])
        act3 = rows["act"] == 2
        self.assertEqual(4, int(act3.sum()))
        self.assertEqual([0] * 4, rows["m2"][act3].tolist())
        self.assertEqual([1] * 4, rows["m3"][act3].tolist())
        self.assertEqual([1] * 4, rows["y3"][act3].tolist())
        self.assertEqual(sorted((33, 38, 44, 47)), sorted(rows["floor"][act3].tolist()))
        self.assertTrue(int(rows["floor"].max()) <= 47)
        self.assertEqual([1] * len(rows["y3"]), rows["won"].tolist())
        for final, y3 in ((47, 0), (48, 0), (49, 1), (53, 1)):
            r = self.rows_of([fake_record("E", final=final, n3=2)])
            self.assertEqual({y3}, set(r["y3"].tolist()), final)

    def test_nothing_unchosen_or_unscorable_and_no_act_1_decision_gives_a_row(self):
        rows = self.rows_of([fake_record("F", final=40, n2=1, n3=1)])
        self.assertEqual(2, len(rows["y2"]))  # one Act 2 row and one Act 3 row: not the Act 1 ones, the post-boss ones, the unchosen one or the unscorable one

    def test_at_most_twelve_act_2_rows_chosen_by_a_hash_with_the_first_always_in(self):
        rec = fake_record("G", final=30, n2=40)
        a, b = self.rows_of([rec]), self.rows_of([rec])
        self.assertEqual(12, int((a["act"] == 1).sum()))
        np.testing.assert_array_equal(a["floor"], b["floor"])
        np.testing.assert_array_equal(a["dense"], b["dense"])
        first_floor = min(17 + (k % 16) for k in range(40))
        self.assertEqual(first_floor, int(a["floor"][0]))
        self.assertTrue(bool(a["entry2"][0]))
        other = self.rows_of([fake_record("H", final=30, n2=40)])
        self.assertEqual(12, int((other["act"] == 1).sum()))
        self.assertEqual(sorted(bc._cap_pick("G", list(range(40)), 12)), bc._cap_pick("G", list(range(40)), 12))  # noqa: SLF001
        self.assertNotEqual(bc._cap_pick("G", list(range(40)), 12), bc._cap_pick("H", list(range(40)), 12))  # noqa: SLF001
        self.assertEqual(list(range(5)), bc._cap_pick("G", list(range(5)), 12))  # noqa: SLF001 -- under the cap: all

    def test_the_flags_mark_the_first_act_2_row_the_first_and_last_act_3_rows_and_the_runs_that_reached_the_boss(self):
        rows = self.rows_of([fake_record("I", final=48, n2=3, n3=4), fake_record("J", final=40, n2=3, n3=3), fake_record("K", final=25, n2=3)])
        for run in np.unique(rows["run"]):
            m = rows["run"] == run
            self.assertEqual(1, int(rows["entry2"][m].sum()))
            self.assertTrue(bool(rows["entry2"][np.flatnonzero(m & (rows["act"] == 1))[0]]))
        three = rows["act"] == 2
        for run in np.unique(rows["run"][three]):
            m = (rows["run"] == run) & three
            self.assertEqual(1, int(rows["entry3"][m].sum()))
            self.assertEqual(1, int(rows["door"][m].sum()))
            self.assertTrue(bool(rows["entry3"][np.flatnonzero(m)[0]]))
            self.assertTrue(bool(rows["door"][np.flatnonzero(m)[-1]]))
        self.assertEqual(int(((rows["run"] == 0)).sum()), int(rows["reached48"][rows["run"] == 0].sum()))  # the run that died on the boss reached it
        self.assertEqual(0, int(rows["reached48"][rows["run"] == 1].sum()))
        self.assertEqual(0, int(rows["reached48"][rows["run"] == 2].sum()))

    def test_a_row_is_the_chosen_options_afterstate_row_with_h3_and_the_champions_forecasts(self):
        from human_prior_infer import option_rows

        nets = fake_nets()
        rec = fake_record("L", final=40, n2=1, n3=1)
        rows = bc.chain_rows([rec], nets)
        entry = next(e for e in rec["decisions"] if int(e["state"]["act_index"]) == 1 and e["state"]["floor"] <= 32 and e["chosen"] is not None)
        dec = shards.entry_decision(entry)
        want = option_rows(nets, shards.state_view(rec, entry), [dec.options[entry["chosen"]]])[0]
        dense, counts = models.value_inputs(want[None, :])
        np.testing.assert_allclose(rows["dense"][0].astype(np.float32), dense[0], atol=2e-3)
        np.testing.assert_array_equal(rows["counts"][0].astype(np.float32), counts[0])
        np.testing.assert_allclose(rows["h3"][0], bc.region_block(nets.fight, want[None, :], vocab.REGION_GLORY)[0], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(rows["pv"][0], [20 / 53, 5 / 53], atol=1e-6)  # the stand-in brain is uniform over 53 classes: floors 34 and up are 20 of them, 49 and up 5
        self.assertEqual(18, int(rows["gen"][0]))
        for k, v in rows.items():
            self.assertIn(v.dtype.kind, "fiub", k)  # no seed, no string, in any array

    def test_records_of_other_eras_other_generations_truncated_and_other_blocks_are_left_out(self):
        stats: dict = {}
        recs = [fake_record("M1", final=40), fake_record("M2", final=40, truncated=True), fake_record("M3", final=40, gen=12), fake_record("M4", final=40),
                {**fake_record("M5", final=40), "ascension": 5}, {**fake_record("M6", final=40), "combat": {"budget": "120"}}, fake_record("M7", final=40, gen=17)]
        rows = self.rows_of(recs, skip_seeds={"M4"}, stats=stats)
        self.assertEqual({0, 1}, set(np.unique(rows["run"]).tolist()))  # M1 and M7
        self.assertEqual({"records": 7, "runs": 2, "skipped_other_block": 1, "dropped_truncated": 1, "dropped_generation": 1, "dropped_era": 2}, stats)
        self.assertEqual({17, 18}, set(rows["gen"].tolist()))

    def test_a_copy_of_a_run_is_counted_once_across_shards_and_runs_are_numbered_on(self):
        nets = fake_nets()
        recs = [fake_record(f"N{i}", final=40 + i) for i in range(4)]
        a = bc.chain_rows(recs[:3], nets, with_keys=True)
        b = bc.chain_rows(recs[1:], nets, with_keys=True)  # N1 and N2 again
        joined = bc.concat_rows([a, b])
        self.assertEqual(4, len(np.unique(joined["run"])))
        self.assertEqual(list(range(4)), np.unique(joined["run"]).tolist())
        self.assertNotIn("run_key", joined)
        plain = bc.concat_rows([bc.chain_rows(recs[:3], nets), bc.chain_rows(recs[1:], nets)])  # without keys nothing is dropped
        self.assertEqual(6, len(np.unique(plain["run"])))

    def test_counts_and_q4(self):
        recs = [fake_record("O1", final=49, n3=2), fake_record("O2", final=53, won=True, n3=2), fake_record("O3", final=48, n3=2), fake_record("O4", final=25),
                fake_record("O5", final=45, gen=16, n3=2)]
        rows = self.rows_of(recs)
        c = bc.counts_block(rows)
        self.assertEqual({"runs": 3, "positives": 2}, {k: c["18"]["q3"][k] for k in ("runs", "positives")})
        self.assertEqual({"runs": 4, "positives": 3}, {k: c["18"]["q2"][k] for k in ("runs", "positives")})
        self.assertEqual({"runs": 1, "positives": 0}, {k: c["16"]["q3"][k] for k in ("runs", "positives")})
        q4 = bc.q4_of(rows)
        self.assertEqual((2, 1), (q4["runs_past_b3a"], q4["wins"]))
        self.assertAlmostEqual((1 + 1) / (2 + 2), q4["q4"])


class MaskedWeightTests(unittest.TestCase):
    def test_each_run_weighs_one_inside_a_head_and_the_mean_is_one_over_the_masked_rows(self):
        run = np.array([0, 0, 0, 1, 2, 2, 3])
        mask = np.array([1, 1, 0, 1, 1, 1, 0])
        w = bc.run_weights_masked(run, mask)
        self.assertEqual(0.0, float(w[2]))
        self.assertEqual(0.0, float(w[6]))
        self.assertAlmostEqual(1.0, float(w[mask > 0].mean()), places=6)
        per_run = {r: float(w[(run == r) & (mask > 0)].sum()) for r in (0, 1, 2)}
        self.assertAlmostEqual(per_run[0], per_run[1], places=6)
        self.assertAlmostEqual(per_run[1], per_run[2], places=6)
        self.assertEqual(0, int((bc.run_weights_masked(run, np.zeros(7)) > 0).sum()))


# ------------------------------------------------------------------------------------------------------------------ H3


class H3Tests(unittest.TestCase):
    def test_h3_is_the_fight_nets_summary_on_act_3s_pools_and_all_its_bosses(self):
        fight = models.NumpyFightNet(fight_params())
        bases = np.stack([F.encode(view_of(F, vocab, 1, 20 + k, 40 + k, deck_extra=k)) for k in range(5)]).astype(np.float32)
        h3 = bc.region_block(fight, bases, vocab.REGION_GLORY)
        self.assertEqual((5, F.FIGHT_BLOCK_SIZE), h3.shape)
        np.testing.assert_array_equal(h3, fight.summary(bases, bc.region_queries(vocab.REGION_GLORY)))
        pools = vocab.encounter_pools()[vocab.REGION_GLORY]
        for pool in ("weak", "normal", "elite", "boss"):
            self.assertEqual(list(pools[pool]), [q.encounter_id for q in bc.region_queries(vocab.REGION_GLORY) if q.pool == pool])
        self.assertEqual(3, len([q for q in bc.region_queries(vocab.REGION_GLORY) if q.pool == "boss"]))
        self.assertEqual([q.encounter_id for q in a2.hive_queries()], [q.encounter_id for q in bc.region_queries(vocab.REGION_HIVE)])
        rows = np.concatenate([bases, np.full((5, 8), 0.7, np.float32)], axis=1)
        np.testing.assert_array_equal(h3, bc.region_block(fight, rows, vocab.REGION_GLORY))  # the 1,863-wide row's H is not read
        with self.assertRaises(ValueError):
            bc.region_block(fight, bases[:, :1800], vocab.REGION_GLORY)

    def test_the_pools_of_act_3_do_not_depend_on_the_boss_so_h_and_h3_share_their_first_six_numbers(self):
        fight = models.NumpyFightNet(fight_params(seed=8))
        bases = np.stack([F.encode(view_of(F, vocab, 2, 40, 50, deck_extra=k)) for k in range(3)]).astype(np.float32)
        own = fight.summary(bases, F.fight_queries(view_of(F, vocab, 2, 40, 50)))
        np.testing.assert_allclose(own[:, :6], bc.region_block(fight, bases, vocab.REGION_GLORY)[:, :6], rtol=1e-6, atol=1e-7)


# ------------------------------------------------------------------------------------------------------------------ the net


class NetTests(unittest.TestCase):
    def inputs(self, n: int = 40, seed: int = 1):
        rng = np.random.default_rng(seed)
        dense = rng.normal(0, 1, (n, 350)).astype(np.float32)
        counts = np.zeros((n, 1513), np.float32)
        counts[:, :8] = rng.integers(0, 4, (n, 8))
        counts[:, 1154:1160] = rng.integers(0, 2, (n, 6))
        h3 = rng.uniform(0, 1, (n, 8)).astype(np.float32)
        gen = rng.choice(bc.STORES, n).astype(np.int8)
        return dense, counts, h3, gen

    def test_h3_columns_start_at_zero_the_trunk_is_the_run_brains_and_the_offsets_are_zero(self):
        value = fake_value()
        p = bc.init_params(value, seed=0)
        self.assertEqual((256, 350 + 104 + 8), p["hidden1.weight"].shape)
        np.testing.assert_array_equal(value["hidden1.weight"], p["hidden1.weight"][:, :454])
        self.assertEqual(0.0, float(np.abs(p["hidden1.weight"][:, 454:]).max()))
        np.testing.assert_array_equal(value["hidden2.weight"], p["hidden2.weight"])
        np.testing.assert_array_equal(np.zeros((2, 4), np.float32), p["gen_off"])
        self.assertEqual((2, 256), p["head.weight"].shape)
        self.assertLess(float(np.abs(p["head.weight"]).max()), 0.1 / 16 + 1e-6)
        dense, counts, h3, gen = self.inputs()
        a = bc.np_logits(p, dense, counts, h3, gen)
        b = bc.np_logits(p, dense, counts, np.zeros_like(h3), gen)
        np.testing.assert_array_equal(a, b)  # H3 changes no logit at initialisation
        with self.assertRaises(ValueError):
            bc.init_params({**value, "hidden1.weight": value["hidden1.weight"][:, :400]})

    @unittest.skipIf(torch is None, "torch is needed for the fit")
    def test_numpy_inference_equals_torch(self):
        params = chain_params(seed=11, scale=0.04)
        dense, counts, h3, gen = self.inputs()
        np.testing.assert_allclose(bc.torch_logits(params, dense, counts, h3, gen), bc.np_logits(params, dense, counts, h3, gen), rtol=1e-4, atol=1e-5)
        with self.assertRaises(ValueError):
            bc.np_logits(params, dense, counts, h3, np.full(len(dense), 12, np.int8))  # a generation with no offset column

    def test_the_offset_column_of_the_rows_generation_is_added_to_both_heads(self):
        params = chain_params(seed=2)
        dense, counts, h3, _ = self.inputs(6)
        base = bc.np_logits(params, dense, counts, h3, np.full(6, 15, np.int8))
        for k, g in enumerate(bc.STORES):
            got = bc.np_logits(params, dense, counts, h3, np.full(6, g, np.int8))
            np.testing.assert_allclose(got - base, np.tile(params["gen_off"][:, k] - params["gen_off"][:, 0], (6, 1)), atol=1e-5)

    @unittest.skipIf(torch is None, "torch is needed for the fit")
    def test_a_planted_relation_is_recovered_by_both_heads(self):
        def planted(n: int, seed: int):
            rng = np.random.default_rng(seed)
            dense = rng.normal(0, 0.1, (n, 350)).astype(np.float32)
            dense[:, [3, 10, 25]] = rng.normal(0, 1.0, (n, 3))
            counts = np.zeros((n, 1513), np.float32)
            counts[:, :6] = rng.integers(0, 4, (n, 6))
            h3 = rng.uniform(0, 1, (n, 8)).astype(np.float32)
            act2 = rng.uniform(size=n) < 0.5
            sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
            y2 = (rng.uniform(size=n) < sig(2.0 * dense[:, 25] + 1.5 * (h3[:, 1] - 0.5) * 2)).astype(np.int8)
            logit3 = 2.5 * dense[:, 3] + 1.5 * dense[:, 10] - 0.5
            y3 = np.where(act2 & (y2 == 0), 0, (rng.uniform(size=n) < sig(logit3)).astype(np.int8)).astype(np.int8)
            m3 = (~act2) | (y2 == 1)
            return {"dense": dense.astype(np.float16), "counts": counts.astype(np.uint8), "h3": h3, "gen": np.full(n, 18, np.int8), "run": np.arange(n),
                    "y2": np.where(act2, y2, 1).astype(np.int8), "m2": act2.astype(np.int8), "y3": y3, "m3": m3.astype(np.int8)}, act2

        train, _ = planted(8000, 1)
        val, _ = planted(1500, 2)
        test, act2 = planted(3000, 3)
        params, hist = bc.fit_arrays(train, val, init=None, epochs=6, batch=64, lr=1e-3, weight_decay=1e-4, seed=0, device="cpu", dropout=0.1)
        self.assertIn("best_epoch", hist)
        self.assertTrue(len(hist["evals"]) >= 2)
        z = bc.np_logits(params, test["dense"].astype(np.float32), test["counts"].astype(np.float32), test["h3"], test["gen"])
        m3 = test["m3"] > 0
        self.assertGreater(bc.auc(test["y3"][m3], z[m3, 1]), 0.8)
        m2 = test["m2"] > 0
        self.assertGreater(bc.auc(test["y2"][m2], z[m2, 0]), 0.8)
        # the validation loss the fit kept is the numpy one's
        loss = bc.weighted_loss(params, val)
        self.assertAlmostEqual(hist["best_val_loss"], loss["total"], delta=2e-3)
        # report cells on rows laid out as the report set is: every row its own run, an entry and a door
        rep = {**test, "floor": np.zeros(len(test["y2"]), np.int16), "act": np.where(act2, 1, 2).astype(np.int8), "entry2": act2, "entry3": ~act2, "door": ~act2,
               "reached48": np.ones(len(test["y2"]), bool), "won": np.zeros(len(test["y2"]), np.int8), "pv": np.full((len(test["y2"]), 2), 0.1, np.float32),
               "part": np.full(len(test["y2"]), bc.PART["report"], np.int8)}
        cells = bc.report_cells(rep, params, champion=True)
        self.assertGreater(cells["q3_door"]["chain"]["auc"], 0.8)
        self.assertEqual(0.5, cells["q3_door"]["champion"]["auc"])  # a constant forecast is no ranking
        self.assertEqual(int((~act2).sum()), cells["q3_door"]["runs"])


class VerdictAndCellTests(unittest.TestCase):
    def test_the_pre_registered_verdict(self):
        good = {"auc": 0.74, "slope": 1.0, "mean_p": 0.15, "rate": 0.16}
        self.assertEqual(("pass", "go"), tuple(bc.verdict_of(good)[k] for k in ("verdict", "band")))
        self.assertEqual("no-pass", bc.verdict_of({**good, "slope": 1.5})["verdict"])
        self.assertEqual("no-pass", bc.verdict_of({**good, "slope": 0.5})["verdict"])
        self.assertEqual("no-pass", bc.verdict_of({**good, "mean_p": 0.20})["verdict"])  # more than 3 points off
        self.assertEqual(("no-pass", "between"), tuple(bc.verdict_of({**good, "auc": 0.68})[k] for k in ("verdict", "band")))
        self.assertEqual("stop", bc.verdict_of({**good, "auc": 0.60})["band"])
        self.assertEqual("stop", bc.verdict_of({"rows": None})["band"])

    def test_cell_metrics_hold_the_deciles_and_suppress_small_cells(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.02, 0.4, 500)
        y = (rng.uniform(size=500) < p).astype(int)
        m = bc.cell_metrics(p, y)
        self.assertEqual(500, m["rows"])
        self.assertEqual(10, len(m["deciles"]))
        self.assertAlmostEqual(float(p.mean()), m["mean_p"])
        self.assertGreater(m["auc"], 0.55)
        self.assertGreater(m["slope"], 0.5)
        self.assertEqual({"rows": None}, bc.cell_metrics(p[:10], y[:10]))
        self.assertTrue(all(d["rows"] == 10 for d in bc.cell_metrics(p[:100], y[:100], min_cell=10)["deciles"]))
        self.assertTrue(all(d["rows"] is None and d["rate"] is None for d in bc.cell_metrics(p[:100], y[:100], min_cell=20)["deciles"]))  # a decile under the cell size is not shown


# ------------------------------------------------------------------------------------------------------------------ the scorer and chain_terms


@unittest.skipIf(torch is None, "torch is needed to write the chain file")
class ScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.fight = fight_params()
        cls.params = chain_params(seed=4, scale=0.08)
        cls.files = bc.write_model(cls.root / "boss_chain", cls.params, cls.fight, {"q4": {"q4": 0.04}})
        rng = np.random.default_rng(3)
        cls.rows = np.stack([F.encode(view_of(F, vocab, 1 + (k % 2), 20 + k, 40 + k, deck_extra=k)) for k in range(6)]).astype(np.float32)
        cls.rows = np.concatenate([cls.rows, rng.uniform(0, 1, (6, 8)).astype(np.float32)], axis=1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_scorer_reads_the_offset_at_generation_18_and_computes_h3_from_its_own_fight_net(self):
        s = bc.ChainScorer(self.files["npz"])
        dense, counts = models.value_inputs(self.rows)
        h3 = bc.region_block(models.NumpyFightNet(self.fight), self.rows, vocab.REGION_GLORY)
        want = bc.np_logits(self.params, dense, counts, h3, np.full(6, 18, np.int8))
        np.testing.assert_allclose(want, s.logits(self.rows), atol=1e-5)
        other = bc.np_logits(self.params, dense, counts, h3, np.full(6, 15, np.int8))
        self.assertGreater(float(np.abs(want - other).max()), 1e-3)  # the offsets differ by generation: the player reads 18's
        np.testing.assert_allclose(want, s.logits(self.rows, h3), atol=1e-6)
        np.testing.assert_allclose(np.log(np.clip(1 / (1 + np.exp(-want)), bc.CLIP, 1.0)), s.log_q(self.rows), atol=1e-5)
        self.assertEqual(bc.file_sha256(self.files["npz"]), s.sha256)
        self.assertEqual(bc.fight_arrays_sha256(self.fight), s.fight_sha256)
        self.assertEqual(18, s.meta["read_gen"])

    def test_the_log_chance_is_clipped_below_at_the_clip_and_never_above_zero(self):
        p = {**self.params, "head.bias": np.array([-40.0, 40.0], np.float32)}
        files = bc.write_model(self.root / "extreme", p, self.fight, {})
        lq = bc.ChainScorer(files["npz"]).log_q(self.rows)
        np.testing.assert_allclose(lq[:, 0], math.log(bc.CLIP), atol=1e-9)  # a chance of 4e-18 reads as the clip
        np.testing.assert_allclose(lq[:, 1], 0.0, atol=1e-9)  # a chance of 1 has no log

    def test_a_file_that_is_not_a_chain_or_whose_fight_arrays_are_not_its_metas_is_refused(self):
        bad = self.root / "bad.npz"
        np.savez(bad, nothing=np.zeros(1))
        with self.assertRaises(ValueError):
            bc.ChainScorer(bad)
        with np.load(self.files["npz"], allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["fight/out.bias"] = arrays["fight/out.bias"] + 1.0
        tampered = self.root / "tampered.npz"
        np.savez(tampered, **arrays)
        with self.assertRaises(ValueError):
            bc.ChainScorer(tampered)

    def test_fight_sha_of_a_run_value_file_is_the_one_write_model_records(self):
        rv = self.root / "run_value.npz"
        np.savez(rv, meta=np.array("{}"), **{f"fight/{k}": v for k, v in self.fight.items()}, **{"value/hidden1.weight": np.zeros((2, 2), np.float32)})
        self.assertEqual(bc.ChainScorer(self.files["npz"]).fight_sha256, bc.fight_sha_of(rv))


class ChainTermsTests(unittest.TestCase):
    lq = np.log(np.array([[0.5, 0.2], [0.4, 0.1], [0.9, 0.5]]))

    def test_act_2_up_to_the_boss_adds_both_logs_and_after_it_the_third_logs_only(self):
        k = (0.0, 10.0, 7.0)
        np.testing.assert_allclose(10.0 * (self.lq[:, 0] + self.lq[:, 1]), bc.chain_terms(self.lq, 1, 17, k))
        np.testing.assert_allclose(10.0 * (self.lq[:, 0] + self.lq[:, 1]), bc.chain_terms(self.lq, 1, 32, k))
        np.testing.assert_allclose(10.0 * self.lq[:, 1], bc.chain_terms(self.lq, 1, 33, k))  # Act 2's floor 33: after the boss
        np.testing.assert_allclose(7.0 * self.lq[:, 1], bc.chain_terms(self.lq, 2, 33, k))
        np.testing.assert_allclose(7.0 * self.lq[:, 1], bc.chain_terms(self.lq, 2, 47, k))

    def test_nothing_in_act_1_at_kappa_0_past_floor_47_or_outside_acts_1_to_2(self):
        for args in ((0, 5, (5.0, 10.0, 10.0)), (1, 20, (10.0, 0.0, 10.0)), (2, 40, (10.0, 10.0, 0.0)), (2, 48, bc.KAPPA), (2, 52, bc.KAPPA), (3, 40, (1, 1, 1))):
            self.assertIsNone(bc.chain_terms(self.lq, *args), args)
        self.assertIsNotNone(bc.chain_terms(self.lq, 2, 47, bc.KAPPA))


# ------------------------------------------------------------------------------------------------------------------ the chooser


@unittest.skipUnless(REAL_OK, "the fitted prior, the run judgement and torch are not on this machine")
class PolicySetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nets = models.load_numpy(REAL_JUDGE)
        cls.prior = hpi.PriorScorer.load(REAL_WEIGHTS)
        cls.tmp = tempfile.TemporaryDirectory()
        with np.load(REAL_WEIGHTS, allow_pickle=False) as z:
            params = {k: z[k] for k in z.files if k != "meta"}
        cls.student = Path(cls.tmp.name) / "same.npz"
        hpi.write_student(cls.student, params, kinds=("card", "route", "shop", "reward"), prior_sha256=cls.prior.sha256)
        cls.chain_dir = make_chain_dir(Path(cls.tmp.name) / "chain", REAL_JUDGE)
        cls.screens = act23_screens(F, vocab)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def policy(self, which: str, chain=True, kappa=bc.KAPPA, scorer=None, **kw):
        extra = {"student": {"student": self.student, "student_lams": STUDENT_LAMS}, "plain": {}}[which]
        if chain:
            kw.update(chain=scorer if scorer is not None else self.chain_dir / "boss_chain.npz", chain_kappa=kappa)
        return ic.BlendedPolicy(REAL_JUDGE, self.prior, 0.5, W=53.0, **extra, **kw)

    def scorer(self, fn):
        """The chain file's scorer with its log-chances replaced by `fn(rows) -> (n, 2)`."""
        s = bc.ChainScorer(self.chain_dir / "boss_chain.npz")
        s.log_q = lambda rows, h3=None: np.asarray(fn(rows), dtype=np.float64)  # type: ignore[method-assign]
        return s

    @staticmethod
    def favour(j: int, col=(0, 1)):
        """A chain that gives option j a chance of 0.9 and every other a chance of 0.05 (in the columns given)."""
        def fn(rows):
            out = np.log(np.full((len(rows), 2), 0.5))
            for c in col:
                out[:, c] = np.log(0.05)
                out[j, c] = np.log(0.9)
            return out
        return fn

    def picks(self, pol, screens_):
        got = []
        for v, d in screens_:
            pk = pol.pick(v, d, None)
            got.append((int(pk.index), tuple(round(float(x), 4) for x in pk.values), tuple(round(float(x), 6) for x in pk.p_floor), dict(pol.last) if pol.last else None))
        return got


class NoChainTests(PolicySetup):
    def test_with_no_chain_the_choices_and_values_on_forty_stored_screens_are_what_they_were(self):
        self.assertEqual(40, len(self.screens))
        for which in ("student", "plain"):
            for kw in ({}, {"chain": None, "chain_kappa": None}):
                pol = ic.BlendedPolicy(REAL_JUDGE, self.prior, 0.5, W=53.0, **({"student": self.student, "student_lams": STUDENT_LAMS} if which == "student" else {}), **kw)
                got = [[int(pol.pick(v, d, None).index), [round(float(x), 4) for x in pol.pick(v, d, None).values], len(d.options)] for v, d in self.screens]
                self.assertEqual(GOLDEN[which], got, f"{which} {kw}")

    def test_a_chain_with_every_kappa_zero_changes_nothing_either(self):
        for which in ("student", "plain"):
            self.assertEqual(self.picks(self.policy(which, chain=False), self.screens), self.picks(self.policy(which, kappa=(0.0, 0.0, 0.0)), self.screens), which)

    def test_act_1_screens_and_floors_past_47_are_what_they_were_whatever_the_chain_says(self):
        for which in ("student", "plain"):
            plain = self.policy(which, chain=False)
            for j in range(3):
                pol = self.policy(which, scorer=self.scorer(self.favour(j)))
                for k, (v, d) in enumerate(self.screens):
                    if v.act_index == 0 or v.floor > 47:
                        a, b = plain.pick(v, d, None), pol.pick(v, d, None)
                        self.assertEqual((a.index, a.values, a.p_floor), (b.index, b.values, b.p_floor), (which, j, k))
                        self.assertEqual(plain.last, pol.last)

    def test_an_equal_chance_for_every_option_changes_no_pick_in_acts_2_and_3(self):
        for which in ("student", "plain"):
            pol = self.policy(which, scorer=self.scorer(lambda rows: np.log(np.full((len(rows), 2), 0.3))))
            for k, (v, d) in enumerate(self.screens):
                self.assertEqual(GOLDEN[which][k][0], pol.pick(v, d, None).index, (which, k))


class ChainEffectTests(PolicySetup):
    def pick_with(self, which, j, v, d, kappa=bc.KAPPA, col=(0, 1)):
        return self.policy(which, scorer=self.scorer(self.favour(j, col)), kappa=kappa).pick(v, d, None)

    def test_a_strongly_favoured_option_is_the_pick_on_every_screen_the_chain_applies_to(self):
        applied = {"rest": 0, "card_reward": 0, "shop": 0, "map": 0}
        for which in ("student", "plain"):
            for k, (v, d) in enumerate(self.screens):
                if v.act_index not in (1, 2) or v.floor > 47:
                    continue
                for j in range(len(d.options)):
                    pk = self.pick_with(which, j, v, d)
                    self.assertEqual(j, pk.index, (which, k, j, d.type))
                applied[d.type] += 1
        self.assertTrue(all(applied.values()), applied)  # the rest (blend_rest), card (blend_pick), shop and route (vrun and the student) screens were all met

    def test_the_recorded_values_and_floor_distributions_are_the_run_brains(self):
        for which in ("student", "plain"):
            plain = self.policy(which, chain=False)
            for k, (v, d) in enumerate(self.screens):
                if v.act_index not in (1, 2) or v.floor > 47:
                    continue
                a, b = plain.pick(v, d, None), self.pick_with(which, len(d.options) - 1, v, d)
                self.assertEqual(a.values, b.values, (which, k))
        # the recorded floor distribution is the chosen option's own row's, not the chain's
        v, d = next((v, d) for v, d in self.screens if v.act_index == 2 and v.floor == 40 and d.type == "rest")
        pk = self.pick_with("plain", 1, v, d)
        rows = hpi.option_rows(self.nets, v, list(d.options))
        np.testing.assert_allclose(self.nets.value.floor_probs(rows)[1], pk.p_floor, atol=1e-9)

    def test_the_values_the_choice_reads_are_the_judgement_plus_kappa_times_the_log_chances(self):
        v, d = next((v, d) for v, d in self.screens if v.act_index == 1 and v.floor == 25 and d.type == "map")
        rng = np.random.default_rng(1)
        lq = np.log(rng.uniform(0.05, 0.9, (len(d.options), 2)))
        for kappa in ((0.0, 10.0, 10.0), (0.0, 3.0, 3.0)):
            pol = self.policy("plain", scorer=self.scorer(lambda rows: lq), kappa=kappa)
            values = pol.pick(v, d, None).values
            want = np.asarray(values) + kappa[1] * (lq[:, 0] + lq[:, 1])
            self.assertEqual(int(np.argmax(want)), pol.pick(v, d, None).index)
        v3, d3 = next((v, d) for v, d in self.screens if v.act_index == 2 and v.floor == 40 and d.type == "map")
        lq3 = np.log(rng.uniform(0.05, 0.9, (len(d3.options), 2)))
        pol = self.policy("plain", scorer=self.scorer(lambda rows: lq3))
        values = pol.pick(v3, d3, None).values
        self.assertEqual(int(np.argmax(np.asarray(values) + 10.0 * lq3[:, 1])), pol.pick(v3, d3, None).index)  # Act 3: the second column alone

    def test_act_2_before_the_boss_reads_both_columns_and_floor_33_and_act_3_read_the_second_only(self):
        for k, (v, d) in enumerate(self.screens):
            if v.act_index not in (1, 2) or v.floor > 47:
                continue
            pol = self.policy("plain", scorer=self.scorer(self.favour(len(d.options) - 1, col=(0,))))  # only q2 favours the last option
            got = pol.pick(v, d, None).index
            if v.act_index == 1 and v.floor <= 32:
                self.assertEqual(len(d.options) - 1, got, k)
            else:
                self.assertEqual(GOLDEN["plain"][k][0], got, k)

    def test_a_kappa_of_zero_for_an_act_leaves_that_act_alone(self):
        for k, (v, d) in enumerate(self.screens):
            if v.act_index not in (1, 2) or v.floor > 47:
                continue
            kappa = (0.0, 0.0, 10.0) if v.act_index == 1 else (0.0, 10.0, 0.0)
            self.assertEqual(GOLDEN["plain"][k][0], self.pick_with("plain", len(d.options) - 1, v, d, kappa=kappa).index, k)

    def test_the_vrun_path_with_rows_scores_the_same_numbers_as_without_them(self):
        pol = self.policy("plain", chain=False)
        for v, d in self.screens:
            if d.type not in ("shop", "map") or len(d.options) < 2:
                continue
            rows = hpi.option_rows(self.nets, v, list(d.options))
            a = pol._vrun_with_bonus(v, d, None, [0.0] * len(d.options))  # noqa: SLF001
            b = pol._vrun_with_bonus(v, d, None, [0.0] * len(d.options), rows)  # noqa: SLF001
            c = LookaheadPolicy.pick(pol, v, d, None)
            self.assertEqual((a.index, a.values, a.p_floor), (b.index, b.values, b.p_floor))
            self.assertEqual((a.index, a.values, a.p_floor), (c.index, c.values, c.p_floor))

    def test_exploring_and_sampling_a_chained_screen_replay_and_stay_among_the_options(self):
        v, d = next((v, d) for v, d in self.screens if v.act_index == 2 and v.floor == 40 and d.type == "card_reward")
        pol = self.policy("plain", scorer=self.scorer(self.favour(1)), temperature=0.5, eps=0.0)
        rng = np.random.default_rng(0)
        picks = {pol.pick(v, d, rng).index for _ in range(40)}
        self.assertTrue(picks <= set(range(len(d.options))))
        sampled = self.policy("plain", scorer=self.scorer(self.favour(1)), pick=mr.PICK_SAMPLE)
        sampled.begin_run("SEED")
        first = [sampled.pick(v, d, None).index for _ in range(6)]
        sampled.begin_run("SEED")
        self.assertEqual(first, [sampled.pick(v, d, None).index for _ in range(6)])

    def test_describe_names_the_chain_only_where_there_is_one_and_a_bad_chain_or_kappa_is_refused(self):
        self.assertNotIn("boss_chain", self.policy("plain", chain=False).describe())
        pol = self.policy("plain")
        sha = bc.file_sha256(self.chain_dir / "boss_chain.npz")
        self.assertEqual(f"{sha[:8]}@0/10/10", pol.describe()["boss_chain"])
        path = self.chain_dir / "boss_chain.npz"
        with self.assertRaises(ValueError):
            ic.BlendedPolicy(REAL_JUDGE, self.prior, 0.5, W=53.0, chain=path)  # a chain needs its kappa
        with self.assertRaises(ValueError):
            ic.BlendedPolicy(REAL_JUDGE, self.prior, 0.5, W=53.0, chain_kappa=(0, 10, 10))  # a kappa needs its chain
        for bad in ((0, 10), (0, 10, 10, 10), (-1, 10, 10), (0, 10, 51), (0, math.nan, 10), (0, math.inf, 10), "0/10/10"):
            with self.assertRaises(ValueError, msg=str(bad)):
                ic.BlendedPolicy(REAL_JUDGE, self.prior, 0.5, W=53.0, chain=path, chain_kappa=bad)

    def test_the_chain_applies_to_a_screen_of_two_or_more_scored_options_in_acts_2_and_3_up_to_floor_47(self):
        pol = self.policy("plain")
        v = view_of(F, vocab, 2, 40, 50)
        d = card_screen(F)
        self.assertTrue(pol.chain_applies(v, d))
        self.assertFalse(pol.chain_applies(v, F.Decision("card_reward", d.options[:1])))  # one option is not a decision
        self.assertFalse(pol.chain_applies(v, F.Decision("card_reward", (d.options[0], dataclasses.replace(d.options[1], scorable=False)))))
        self.assertFalse(pol.chain_applies(dataclasses.replace(v, floor=48), d))
        self.assertFalse(pol.chain_applies(dataclasses.replace(v, act_index=0, floor=10), d))
        self.assertTrue(pol.chain_applies(dataclasses.replace(v, act_index=1, floor=20), d))
        self.assertFalse(self.policy("plain", chain=False).chain_applies(v, d))
        self.assertIsNone(self.policy("plain", chain=False).chain_extra(v, d))
        extra = pol.chain_extra(v, rest_screen(F))
        self.assertEqual(3, len(extra))
        self.assertTrue(all(isinstance(x, float) and x <= 0.0 for x in extra))  # the log of a chance, times kappa


# ------------------------------------------------------------------------------------------------------------------ recipe


class RecipeTests(unittest.TestCase):
    def test_the_part_is_last_in_one_canonical_form(self):
        self.assertEqual("+chain:0123abcd@0/10/10", mr.chain_part("0123abcd", (0, 10, 10)))
        self.assertEqual("+chain:0123abcd@0/2.5/10", mr.chain_part("0123abcd", (0.0, 2.5, 10.0)))
        self.assertEqual("0123abcd@0/10/10", mr.chain_tag("0123abcd" + "0" * 56, (0, 10, 10)))
        chain = ("0123abcd", (0.0, 10.0, 10.0))
        self.assertEqual(BASE + "+chain:0123abcd@0/10/10", mr.full_recipe(BASE, chain=chain))
        self.assertEqual(BASE + f"+stakes:{STAKES}+hold:30+elite:1@60+chain:0123abcd@0/10/10", mr.full_recipe(BASE, STAKES, 30, (1.0, 60), chain))
        self.assertEqual(BASE + f"+stakes:{STAKES}+hold:30", mr.full_recipe(BASE, STAKES, 30))  # without it the recipe is what it was
        self.assertEqual(BASE, mr.full_recipe(BASE))
        self.assertEqual("vrun+chain:0123abcd@0/10/10", mr.full_recipe("vrun", chain=chain))

    def test_it_reads_back_and_the_split_is_the_same_with_and_without_it(self):
        chain = ("0123abcd", (0.0, 10.0, 10.0))
        for stakes, hold, elite in ((None, None, None), (STAKES, None, None), (STAKES, 30, None), (STAKES, 30, (2.0, 75)), (None, 30, (1.0, 60))):
            plain, full = mr.full_recipe(BASE, stakes, hold, elite), mr.full_recipe(BASE, stakes, hold, elite, chain)
            self.assertEqual(mr.split_recipe(plain), mr.split_recipe(full), full)
            self.assertEqual((BASE, stakes, hold), mr.split_recipe(full), full)
            self.assertEqual(elite, mr.parse_elite(full), full)
            self.assertEqual(chain, mr.parse_chain(full), full)
            self.assertIsNone(mr.parse_chain(plain))
        self.assertIsNone(mr.parse_chain(""))
        self.assertIsNone(mr.parse_chain(None))
        self.assertEqual(("", None, None), mr.split_recipe(""))

    def test_a_bad_sha_or_kappa_is_refused(self):
        for sha in ("", "xyz", "ABCDEF01"):
            with self.assertRaises(ValueError, msg=sha):
                mr.chain_part(sha, (0, 10, 10))
        for kappa in ((0, 10), (0, 10, 10, 1), (-1, 10, 10), (0, 51, 0), (0, math.nan, 0), (0, math.inf, 0), "0/10/10", None, (0, "x", 1)):
            with self.assertRaises(ValueError, msg=str(kappa)):
                mr.chain_part("0123abcd", kappa)
        with self.assertRaises(ValueError):
            mr.parse_chain("vrun+chain:0123abcd@0/inf/10")

    def test_a_recipe_with_the_part_is_a_recipe_of_its_own_in_the_ledger(self):
        plain = mr.full_recipe(BASE, STAKES)
        a, b = mr.full_recipe(BASE, STAKES, None, None, ("0123abcd", (0, 10, 10))), mr.full_recipe(BASE, STAKES, None, None, ("0123abcd", (0, 5, 5)))
        self.assertEqual(3, len({plain, a, b}))


# ------------------------------------------------------------------------------------------------------------------ the agent and the candidate


@unittest.skipUnless(REAL_OK, "the run judgement and torch are not on this machine")
class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.agents, _base, self.prior, self.student, self.human = student_world(self.root)
        self.inc = gen_loop.build_candidate(self.prior, self.agents / "current.json", self.agents, 6, why="incumbent", export=False, tag="inc",
                                            student=self.student, student_lams={"route": (0.3715, 2.0, 2.0), "shop": (0.2858, 0.2311, 0.2311)})
        shutil.copyfile(REAL_JUDGE, self.inc / "run_value.npz")  # a real run brain: the chain's FightNet is checked against its
        (self.agents / "current.json").write_text(json.dumps({"generation": 6, "dir": str(self.inc)}))
        self.chain_dir = make_chain_dir(self.root / "chain", REAL_JUDGE)

    def build(self, **kw):
        kw.setdefault("tag", "chain")
        kw.setdefault("kappa", (0.0, 10.0, 10.0))
        return gen_loop.build_chain_candidate(self.inc, self.chain_dir, self.agents, 7, why="test", **kw)

    def test_an_agent_without_the_block_is_what_it_was(self):
        a = gen_loop.load_agent(self.inc)
        self.assertIsNone(a.chain)
        self.assertNotIn("boss_chain", a.describe())
        self.assertNotIn("+chain", a.recipe())
        self.assertEqual(mr.full_recipe(a.macro, a.stakes, a.hold), a.recipe())
        self.assertIn("+stakes:a3-v1@h1e4b16+hold:30", a.recipe())  # the pinned recipe of the fixture's agent

    def test_the_candidate_is_the_base_with_the_chain_and_nothing_else(self):
        cand = self.build()
        base, got = gen_loop.load_agent(self.inc), gen_loop.load_agent(cand)
        sha = bc.file_sha256(cand / "boss_chain.npz")
        self.assertEqual(f"{sha[:8]}@0/10/10", got.describe()["boss_chain"])
        self.assertEqual(base.recipe() + f"+chain:{sha[:8]}@0/10/10", got.recipe())
        self.assertTrue(got.recipe().endswith("+chain:" + sha[:8] + "@0/10/10"))
        self.assertEqual((0.0, 10.0, 10.0), got.chain_kappa)
        self.assertEqual(sha, got.chain_sha256)
        self.assertEqual(base.macro, got.macro)
        self.assertEqual((base.student_lams, base.stakes, base.hold, base.lam, base.lam_acts), (got.student_lams, got.stakes, got.hold, got.lam, got.lam_acts))
        self.assertEqual(base.describe()["prior_sha256"], got.describe()["prior_sha256"])
        meta = json.loads((cand / "agent.json").read_text())
        self.assertEqual((7, 6), (meta["generation"], meta["built_on"]))
        self.assertEqual(got.recipe(), meta["recipe"])
        block = meta["boss_chain"]
        self.assertEqual(("boss_chain.npz", sha, [0.0, 10.0, 10.0]), (block["file"], block["sha256"], block["kappa"]))
        self.assertEqual("pass", block["report_verdict"])
        self.assertAlmostEqual(5 / 102, block["q4"]["q4"])
        self.assertEqual("chain", meta["built"]["swap"])
        self.assertNotIn("lineage", meta)
        self.assertEqual(["settings"], screens.differing(cand, self.inc))
        self.assertIsNone(screens.kind_of(cand, self.inc))

    def test_every_file_of_the_base_is_copied_byte_for_byte_but_the_four_written_anew(self):
        cand = self.build()
        for f in sorted(self.inc.iterdir()):
            if not f.is_file():
                continue
            if f.name in gen_loop.VRUN_SWAP_NOT_COPIED:
                self.assertTrue((cand / f.name).is_file(), f.name)
                continue
            self.assertEqual(f.read_bytes(), (cand / f.name).read_bytes(), f.name)
        self.assertEqual((self.chain_dir / "boss_chain.npz").read_bytes(), (cand / "boss_chain.npz").read_bytes())
        self.assertEqual((self.chain_dir / "fit.json").read_bytes(), (cand / "boss_chain.json").read_bytes())
        self.assertEqual(7, json.loads((cand / "model.json").read_text())["iteration"])  # written anew for the target generation
        self.assertEqual(json.loads((self.inc / "model.json").read_text())["policy_sha256"], json.loads((cand / "model.json").read_text())["policy_sha256"])

    def test_a_second_build_refuses_to_write_over_the_first_and_a_failed_one_leaves_no_folder(self):
        cand = self.build()
        before = (cand / "agent.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(before, (cand / "agent.json").read_bytes())
        for kw in ({"kappa": (0, 10)}, {"kappa": (0, 99, 1)}, {"kappa": (0, math.nan, 1)}):
            with self.assertRaises(ValueError, msg=str(kw)):
                self.build(tag="other", **kw)
        self.assertFalse((self.agents / "gen7" / "candidate-other").exists())
        with self.assertRaises(ValueError):  # a tag the markers cannot name
            self.build(tag="X")

    def test_a_chain_fitted_with_another_fight_net_is_refused_at_the_build_and_at_the_load(self):
        wrong = make_chain_dir(self.root / "wrong", REAL_JUDGE, other_fight=True)
        with self.assertRaisesRegex(ValueError, "FightNet"):
            gen_loop.build_chain_candidate(self.inc, wrong, self.agents, 7, why="t", kappa=(0, 10, 10), tag="wrong")
        self.assertFalse((self.agents / "gen7" / "candidate-wrong").exists())
        cand = self.build()
        shutil.copyfile(wrong / "boss_chain.npz", cand / "boss_chain.npz")  # a swapped file: its sha is not the block's
        with self.assertRaises(ValueError):
            gen_loop.load_agent(cand)
        meta = json.loads((cand / "agent.json").read_text())
        meta["boss_chain"]["sha256"] = bc.file_sha256(cand / "boss_chain.npz")
        (cand / "agent.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "FightNet"):  # the sha is the block's now, the FightNet is not the run brain's
            gen_loop.load_agent(cand)

    def test_a_malformed_block_a_missing_file_or_a_recipe_that_disagrees_is_refused_at_load(self):
        cand = self.build()
        meta = json.loads((cand / "agent.json").read_text())
        for name, change in (("kappa_high", {"kappa": [0, 10, 51]}), ("kappa_len", {"kappa": [0, 10]}), ("kappa_neg", {"kappa": [0, -1, 10]}), ("nokappa", {"kappa": None}),
                             ("sha", {"sha256": "0" * 64}), ("nofile", {"file": "missing.npz"}), ("kappa_recipe", {"kappa": [0, 5, 5]})):  # the last: the recipe names 10
            block = {**meta["boss_chain"], **change}
            if change.get("kappa") is None and "kappa" in change:
                block.pop("kappa")
            (cand / "agent.json").write_text(json.dumps({**meta, "boss_chain": block}))
            with self.assertRaises((ValueError, FileNotFoundError), msg=name):
                gen_loop.load_agent(cand)
        (cand / "agent.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "boss_chain"}))
        with self.assertRaises(ValueError):  # the recipe still names the chain, the fields do not
            gen_loop.load_agent(cand)
        (cand / "agent.json").write_text(json.dumps(meta))
        self.assertEqual(meta["recipe"], gen_loop.load_agent(cand).recipe())

    def test_a_plain_vrun_macro_cannot_carry_a_chain_and_a_base_with_one_is_not_built_on(self):
        root = self.root / "plain"
        root.mkdir()
        agents, base, prior, student, human = student_world(root, blended=False)
        shutil.copyfile(REAL_JUDGE, base / "run_value.npz")
        shutil.copyfile(self.chain_dir / "boss_chain.npz", base / "boss_chain.npz")
        meta = json.loads((base / "agent.json").read_text())
        sha = bc.file_sha256(base / "boss_chain.npz")
        meta.update(boss_chain={"file": "boss_chain.npz", "sha256": sha, "kappa": [0, 10, 10]}, recipe=f"vrun+chain:{sha[:8]}@0/10/10")
        (base / "agent.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "blended"):
            gen_loop.load_agent(base)
        with self.assertRaisesRegex(ValueError, "blended"):  # nor does the builder take a plain macro for a base
            plain_root = self.root / "plain2"
            plain_root.mkdir()
            agents2, base2, *_ = student_world(plain_root, blended=False)
            gen_loop.build_chain_candidate(base2, self.chain_dir, agents2, 7, why="t", kappa=(0, 10, 10))
        cand = self.build()
        with self.assertRaisesRegex(ValueError, "already carries"):
            gen_loop.build_chain_candidate(cand, self.chain_dir, self.agents, 8, why="again", kappa=(0, 5, 5), tag="again")

    def test_the_policy_the_agent_plays_carries_the_chain(self):
        sentinel = {}

        class Spy(ic.BlendedPolicy):
            def __init__(self, *a, **k):
                sentinel.update(k)
                super().__init__(*a, **k)

        agent = gen_loop.Agent(macro=mr.macro_name(0.5, hpi.PriorScorer.load(REAL_WEIGHTS).sha256), macro_npz=str(REAL_JUDGE), human_prior=str(REAL_WEIGHTS), lam=0.5,
                               chain=str(self.chain_dir / "boss_chain.npz"), chain_sha256=bc.file_sha256(self.chain_dir / "boss_chain.npz"), chain_kappa=(0.0, 10.0, 10.0))
        with mock.patch.object(ic, "BlendedPolicy", Spy):
            policy = agent.macro_policy()
        self.assertEqual((0.0, 10.0, 10.0), sentinel.get("chain_kappa"))
        self.assertEqual(str(self.chain_dir / "boss_chain.npz"), sentinel.get("chain"))
        self.assertEqual(agent.describe()["boss_chain"], policy.describe()["boss_chain"])
        sentinel.clear()
        with mock.patch.object(ic, "BlendedPolicy", Spy):
            dataclasses.replace(agent, chain=None, chain_sha256=None, chain_kappa=None).macro_policy()
        self.assertNotIn("chain", sentinel)
        self.assertNotIn("chain_kappa", sentinel)

    def test_the_candidate_is_a_recipe_of_its_own_with_the_same_identity_pair(self):
        base, got = gen_loop.load_agent(self.inc), gen_loop.load_agent(self.build())
        self.assertNotEqual(gen_loop.candidate_recipe(base), gen_loop.candidate_recipe(got))
        self.assertEqual(gen_loop.candidate_identity(base), gen_loop.candidate_identity(got))

    def test_the_emulator_arm_of_another_chain_does_not_match(self):
        a, b = gen_loop.load_agent(self.inc), gen_loop.load_agent(self.build())
        with tempfile.TemporaryDirectory() as d:
            arm = Path(d)
            (arm / "runs.jsonl").write_text("{}\n")
            (arm / "summary.json").write_text(json.dumps({"agent": a.describe()}))
            self.assertTrue(gen0.emu_arm_match(arm, a.describe())[0])
            self.assertFalse(gen0.emu_arm_match(arm, b.describe())[0])
            (arm / "summary.json").write_text(json.dumps({"agent": b.describe()}))
            self.assertTrue(gen0.emu_arm_match(arm, b.describe())[0])
            self.assertFalse(gen0.emu_arm_match(arm, a.describe())[0])

    def test_build_chain_prints_the_candidate_and_the_marker_and_writes_no_marker(self):
        out = io.StringIO()
        marker_dir = self.root / "LAB" / "train"
        args = argparse.Namespace(base=str(self.inc), chain_dir=str(self.chain_dir), agent_root=str(self.agents), target=7, kappa="0/10/10", tag="chain", why="test")
        with contextlib.redirect_stdout(out):
            self.assertEqual(0, gen0.build_chain_cmd(args))
        doc = json.loads(out.getvalue().strip().splitlines()[-1])
        self.assertEqual("ready-6-g7-chain", doc["marker"])
        self.assertTrue(doc["candidate"].endswith("gen7/candidate-chain"))
        self.assertEqual("pass", doc["boss_chain"]["report_verdict"])
        self.assertFalse(marker_dir.exists())
        self.assertEqual([], [p for p in self.root.rglob("ready-*")])


# ------------------------------------------------------------------------------------------------------------------ build task and flips


@unittest.skipUnless(REAL_JUDGE.is_file(), "the run judgement is not on this machine")
class StoreTests(unittest.TestCase):
    def test_a_store_of_complete_jobs_gives_its_shards_and_a_task_gives_the_rows_of_one(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            recs = [fake_record(f"P{i}", final=40 + (i % 5), n2=2, n3=2) for i in range(6)]
            job = root / "18" / "jobs" / "job1" / "shards"
            job.mkdir(parents=True)
            (job / "r0000.jsonl.gz").write_bytes(gzip.compress("\n".join(json.dumps(r) for r in recs).encode()))
            (job / "shards.json").write_text(json.dumps({"generation": 18, "run_shards": ["r0000.jsonl.gz"]}))
            half = root / "18" / "jobs" / "job2" / "shards"  # a job with no shards.json is not complete: not read
            half.mkdir(parents=True)
            (half / "r0000.jsonl.gz").write_bytes(b"not read")
            tasks = bc.store_shards([18, 17], root)
            self.assertEqual([(18, job / "r0000.jsonl.gz")], tasks)
            rows, stats = bc.chain_task(str(tasks[0][1]), str(REAL_JUDGE), [])
            self.assertEqual(6, stats["runs"])
            self.assertEqual(6, len(rows["run_key"]))
            self.assertEqual(4 * 6, len(rows["y2"]))  # two Act 2 and two Act 3 rows each


class FlipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)
        (self.store / "decisions.d").mkdir()
        seeds = []
        i = 0
        while len(seeds) < 25:  # report-part seeds only
            s = f"F{i:05d}"
            i += 1
            if bc.part_of(18, s) == bc.PART["report"]:
                seeds.append(s)
        with open(self.store / "runs.jsonl", "w", encoding="utf-8") as fh:
            for s in seeds + ["notreport0"]:
                fh.write(json.dumps({"seed": s}) + "\n")
        for s in seeds:
            (self.store / "decisions.d" / f"{s}.record.json").write_text(json.dumps(fake_record(s, final=49, n2=3, n3=3, low_hp_rest=True)), encoding="utf-8")
        self.seeds = seeds

    @staticmethod
    def agent(picker):
        policy = types.SimpleNamespace(begin_run=lambda seed: None, pick=lambda view, decision, rng: types.SimpleNamespace(index=picker(view, decision)))
        return types.SimpleNamespace(macro_policy=lambda: policy)

    def run_flips(self, champion, candidate, runs=400):
        agents = {"champ": self.agent(champion), "cand": self.agent(candidate)}
        with mock.patch.object(gen_loop, "load_agent", lambda d: agents[Path(d).name]):
            return bc.flips("champ", "cand", self.store, runs)

    def test_the_counts_by_act_and_the_side_figures(self):
        # the champion takes the first option (the recorded pick of every Act 2 and 3 decision is its own index k%2 in Act 2: the replay is not exact there); the candidate
        # takes the last option in Acts 2 and 3 and the first in Act 1
        doc = self.run_flips(lambda v, d: 0, lambda v, d: 0 if v.act_index == 0 else len(d.options) - 1)
        self.assertEqual(25, doc["runs_replayed"])
        self.assertEqual(0, doc["by_act"]["act1"]["changed"])
        self.assertEqual(doc["by_act"]["act2"]["eligible"], doc["by_act"]["act2"]["changed"])
        self.assertEqual(1.0, doc["by_act"]["act3"]["changed_share"])
        self.assertEqual(25 * 3, doc["by_act"]["act3"]["eligible"])  # the floor 48 rest, the unchosen and the unscorable ones are not decisions it chooses by
        rest = doc["side"]["rest_low_hp"]
        self.assertEqual(25, rest["screens"])  # one low-HP Act 3 rest site a run (n3 = 3 cycles card, rest, shop)
        self.assertEqual(1.0, rest["champion_heal_share"])  # the champion heals; the candidate takes the last option, a smith
        self.assertEqual(0.0, rest["candidate_heal_share"])
        shop = doc["side"]["shop_act3"]
        self.assertEqual((0, 25), (shop["purchases_champion"], shop["purchases_candidate"]))  # the champion leaves, the candidate buys a card (no potion in these shops)
        self.assertEqual(0.0, shop["candidate_potion_share"])
        self.assertIsNone(shop["champion_potion_share"])  # no purchases: no share
        vetoes = bc.flip_vetoes({**doc, "expect": None})
        self.assertTrue(any("act3 changed share" in v for v in vetoes), vetoes)
        self.assertTrue(any("heal share" in v for v in vetoes), vetoes)

    def test_an_identical_candidate_changes_nothing_and_vetoes_nothing_when_the_champion_replays_the_record(self):
        picker = lambda v, d: 0  # noqa: E731
        doc = self.run_flips(picker, picker)
        for act in ("act1", "act2", "act3"):
            self.assertEqual(0, doc["by_act"][act]["changed"])
        self.assertEqual(0, doc["by_act"]["act3"]["changed"])
        # the champion's agreement with the recorded choices is below 98 in 100 here (Act 2 records pick k%2), and that is the replay veto
        self.assertTrue(any("replay agreement" in v for v in bc.flip_vetoes(doc)))

    def test_the_vetoes_of_the_brief(self):
        base = {"by_act": {"act1": {"changed": 0}, "act2": {"changed_share": 0.1}, "act3": {"changed_share": 0.2}}, "all": {"champion_agreement": 0.99},
                "side": {"rest_low_hp": {"champion_heal_share": 0.6, "candidate_heal_share": 0.55}}}
        self.assertEqual([], bc.flip_vetoes(base))
        self.assertTrue(bc.flip_vetoes({**base, "by_act": {**base["by_act"], "act1": {"changed": 2}}}))
        self.assertTrue(bc.flip_vetoes({**base, "all": {"champion_agreement": 0.97}}))
        self.assertTrue(bc.flip_vetoes({**base, "by_act": {**base["by_act"], "act2": {"changed_share": 0.51}}}))
        self.assertTrue(bc.flip_vetoes({**base, "by_act": {**base["by_act"], "act3": {"changed_share": 0.51}}}))
        self.assertTrue(bc.flip_vetoes({**base, "side": {"rest_low_hp": {"champion_heal_share": 0.6, "candidate_heal_share": 0.45}}}))
        self.assertEqual([], bc.flip_vetoes({**base, "side": {"rest_low_hp": {"champion_heal_share": None, "candidate_heal_share": None}}}))


if __name__ == "__main__":
    unittest.main()
