"""The judge corpus's store and its pure parts (rl/judge_corpus.py; D58 C1, D57 addendum 7).

Pinned: the split by run and the stages (thirds of a fight's decisions); the stage turns (one a stage, a hash coin, fixed); the deck type
tag; the strata read off tokens; the sampling (boosts for rare features, the order, the inclusion probabilities that add up to the sample
size, the work order in blocks); the ladder's screen and cap (`settle`: settled at the first rung where the rule holds, only then no deeper
rung, soft at the cap, a capped rung says nothing); the label's root value and rung lines; the shards (rows and tokens round trip), the
loader (by source, kind, act, split and tier; the npz layout the judge reads; the weights; the labels) and its refusals (a human row of
the exam fifth, a split that is not the run's); the coverage table's suppression of small cells; and, with the emulator, that the token
constants are the real encoder's.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter, namedtuple
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import judge_corpus as jc  # noqa: E402

Info = namedtuple("Info", "card_type rarity exhaust base_block")


def child(a: int, n: int, v: float | None, se: float | None = 1.0, t: int = -1, kind: str = "card", pd: float | None = 0.0) -> dict:
    return {"a": a, "t": t, "kind": kind, "card": 7, "merged": [], "n": n, "v": v, "se": se, "pd": pd}


def rec(rel, children: list[dict], cpu: float = 1.0, capped: bool = False, duel: dict | None = None) -> dict:
    children = sorted(children, key=lambda c: (-c["n"], c["a"], c["t"]))
    out = {"rel": rel, "budget": 16, "seed": 1, "sims": sum(c["n"] for c in children), "capped": capped, "cpu_s": cpu, "children": children,
           "best": children[0], "n_moves": len(children)}
    if duel is not None:
        out["duel"] = duel
        out["duel_cpu_s"] = 0.1
    return out


def duel(mean: float, se: float, n: int = 128) -> dict:
    return {"n": n, "mean": mean, "se": se, "p_death_diff": 0.0}


def tokens(n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {"types": rng.integers(1, 17, n).astype(np.uint8), "ids": rng.integers(0, 500, n).astype(np.int32), "owners": rng.integers(0, 4, n).astype(np.uint8),
            "feats": rng.random((n, 18)).astype(np.float32)}


def row(i: int, run: str, src: str = "human", kind: str = "boss", act: int = 2, stage: str = "early", tier: str = "settled", **extra) -> dict:
    out = {"pid": f"{run}|f33#{i}", "src": src, "sub": "human-start", "start": f"human:{run}|f33", "run": run, "split": jc.split_of(run), "kind": kind, "act": act,
           "enemy": "kaiser-crab-boss", "floor": 33, "k": i, "turn": 1 + i, "stage": stage, "stage_n": 5, "decisions": 20, "door": i == 0, "hp": 40, "max_hp": 80,
           "door_max_hp": 80, "hp_frac": 0.5, "potions": 1, "relics": [3, 9], "deck": {"472": 4, "131": 4}, "deck_n": 8, "arch": "mixed", "w": 2.0, "w_cell": 1.0,
           "label": {"tier": tier, "rel": 2, "sims": 960, "root_value": 12.5, "best_value": 13.0, "p_death": 0.1, "children": [child(1, 50, 13.0)]},
           "teacher": {"won": True, "end_hp": 20, "max_hp": 80, "potions_held": 1, "L0": 0.6, "decisions": 20, "turns": 6},
           "human": {"survived": True, "end_hp": 30, "hp_lost": 50, "turns": 7, "potions_used": 1, "potions_held_end": 1, "l0": 40.0, "l0_unit": 0.7}}
    out.update(extra)
    return out


def run_in_bucket(bucket: int, prefix: str = "r") -> str:
    import human_starts as hs

    return next(f"{prefix}{i}" for i in range(10000) if hs.exam_bucket(f"{prefix}{i}") == bucket)


class SplitAndStageTests(unittest.TestCase):
    def test_a_fifth_of_runs_is_held_out_by_a_fixed_hash(self):
        runs = [f"run{i}" for i in range(2000)]
        held = sum(jc.split_of(r) == "heldout" for r in runs)
        self.assertTrue(300 < held < 500, held)
        self.assertEqual([jc.split_of(r) for r in runs[:50]], [jc.split_of(r) for r in runs[:50]])
        self.assertEqual(jc.split_of("abc"), "heldout" if int(__import__("hashlib").sha256(b"abc").hexdigest()[:8], 16) % 5 == 0 else "train")

    def test_stages_are_thirds_of_the_decisions(self):
        self.assertEqual([jc.stage_of(i, 9) for i in range(9)], ["early"] * 3 + ["middle"] * 3 + ["late"] * 3)
        self.assertEqual(jc.stage_of(0, 1), "early")
        self.assertEqual([jc.stage_of(i, 2) for i in range(2)], ["early", "middle"])

    def test_one_turn_a_stage_each_in_its_third_and_fixed(self):
        elig = list(range(3, 63, 2))  # 30 decisions
        picks = jc.pick_stage_turns("human:a|f17", elig)
        self.assertEqual([p["stage"] for p in picks], ["early", "middle", "late"])
        self.assertEqual([p["stage_n"] for p in picks], [10, 10, 10])
        self.assertTrue(all(p["k"] in elig for p in picks))
        self.assertEqual(picks, jc.pick_stage_turns("human:a|f17", elig))
        self.assertEqual(sorted({p["k"] for p in picks}), [p["k"] for p in picks])
        self.assertTrue(elig[:10].count(picks[0]["k"]) == 1 and elig[10:20].count(picks[1]["k"]) == 1 and elig[20:].count(picks[2]["k"]) == 1)
        self.assertNotEqual([p["k"] for p in picks], [p["k"] for p in jc.pick_stage_turns("human:b|f17", elig)])

    def test_a_short_fight_gives_fewer_turns_and_two_a_stage_are_distinct(self):
        self.assertEqual([p["stage"] for p in jc.pick_stage_turns("x", [4])], ["early"])
        self.assertEqual(jc.pick_stage_turns("x", []), [])
        two = jc.pick_stage_turns("human:a|f17", list(range(60)), per_stage=2)
        self.assertEqual(len(two), 6)
        self.assertEqual(len({p["k"] for p in two}), 6)

    def test_the_stage_picks_are_roughly_uniform_over_each_third(self):
        counts: Counter = Counter()
        for i in range(600):
            counts[jc.pick_stage_turns(f"s{i}", list(range(12)))[0]["k"]] += 1
        self.assertEqual(set(counts), {0, 1, 2, 3})
        self.assertTrue(all(100 < v < 200 for v in counts.values()), counts)

    def test_bands_and_potion_states(self):
        self.assertEqual([jc.health_band(x) for x in (0.1, 0.33, 0.34, 0.66, 0.67, 1.0)], ["critical", "critical", "hurt", "hurt", "healthy", "healthy"])
        self.assertEqual([jc.potion_state(n) for n in (0, 1, 2, 3)], ["none", "one", "two+", "two+"])


class DeckTagTests(unittest.TestCase):
    cards = {1: Info("Attack", "Basic", False, 0), 2: Info("Skill", "Basic", False, 5), 3: Info("Curse", "Curse", False, 0), 10: Info("Power", "Uncommon", False, 0),
             11: Info("Power", "Rare", False, 0), 12: Info("Power", "Rare", False, 0), 20: Info("Skill", "Common", True, 0), 21: Info("Skill", "Uncommon", True, 0),
             22: Info("Attack", "Common", True, 0), 30: Info("Skill", "Common", False, 8), 31: Info("Skill", "Uncommon", False, 9), 32: Info("Skill", "Common", False, 7),
             40: Info("Attack", "Common", False, 0), 41: Info("Attack", "Uncommon", False, 0), 42: Info("Attack", "Common", False, 0), 43: Info("Skill", "Common", False, 0)}

    def test_the_first_rule_that_holds_decides(self):
        tag = lambda d: jc.deck_tag(d, self.cards)  # noqa: E731
        self.assertEqual(tag({1: 5, 2: 4, 3: 1, 40: 2, 41: 1}), "starter")  # three core cards
        self.assertEqual(tag({1: 5, 10: 1, 11: 1, 12: 1, 40: 3, 41: 2}), "power")  # three powers, a quarter of the core
        self.assertEqual(tag({1: 5, 20: 2, 21: 1, 40: 3, 41: 2, 42: 2}), "exhaust")  # three exhaust cards, 30%
        self.assertEqual(tag({1: 5, 30: 2, 31: 2, 32: 1, 40: 2, 43: 1}), "block")
        self.assertEqual(tag({1: 5, 40: 3, 41: 3, 42: 2, 43: 2}), "attack")  # 8 of 10
        self.assertEqual(tag({1: 5, 40: 2, 41: 1, 43: 3, 30: 1}), "mixed")

    def test_curses_basics_and_unknown_cards_are_not_the_core(self):
        self.assertEqual(jc.deck_tag({1: 9, 2: 9, 3: 5, 999: 4}, self.cards), "starter")

    def test_token_features_read_strata_off_the_tokens(self):
        T = jc
        types = np.array([T.T_GLOBAL, T.T_HAND, T.T_HAND, T.T_DRAW, T.T_DISCARD, T.T_EXHAUST, 11, T.T_POTION, T.T_POTION, T.T_RELIC, T.T_RELIC, 6])
        ids = np.array([5, 1, 2, 1, 1, 3, 1, 0, 0, 2, 1, 0])  # vocabulary indices
        feats = np.zeros((12, 18), np.float32)
        feats[0, T.F_HP] = 0.4
        feats[7, T.F_MAGNITUDE] = 1.0  # one potion held, one empty slot
        f = jc.token_features(types, ids, feats, {1: 472, 2: 131, 3: 10001}, {1: 36, 2: 4}, self.cards | {472: self.cards[1], 131: self.cards[2], 10001: self.cards[3]})
        self.assertAlmostEqual(f["hp_frac"], 0.4)
        self.assertEqual(f["potions"], 1)
        self.assertEqual(f["relics"], [4, 36])
        self.assertEqual(f["deck"], {"131": 1, "472": 3, "10001": 1})  # the choice screen's candidate (type 11) is not a second copy
        self.assertEqual(f["deck_n"], 5)
        self.assertEqual(f["arch"], "starter")


class SamplingTests(unittest.TestCase):
    def test_boost_is_the_rarest_features_support_over_its_count(self):
        feats = [{"relic:1", "card:7"}] * 50 + [{"relic:1", "card:8"}] * 10 + [{"relic:2"}] * 1000
        b = jc.boost_weights(feats, support=100)
        self.assertEqual(b[0], 2.0)  # card 7 in 50 starts: 100 / 50
        self.assertEqual(b[50], 10.0)  # card 8 in 10
        self.assertEqual(b[60], 1.0)  # relic 2 in 1000: no boost
        self.assertEqual(jc.boost_weights([{"x"}] * 5, support=100, cap=7.0), [7.0] * 5)

    def test_a_boosted_start_comes_earlier_in_the_order(self):
        ids = [f"human:r{i}|f17" for i in range(4000)]
        plain = sorted(ids, key=lambda s: jc.order_key(s, 1.0))
        keys = {s: jc.order_key(s, 20.0 if s.endswith("7|f17") else 1.0) for s in ids}
        boosted = sorted(ids, key=lambda s: keys[s])
        rank = lambda order, s: order.index(s)  # noqa: E731
        sevens = [s for s in ids if s.endswith("7|f17")]
        self.assertLess(np.mean([rank(boosted, s) for s in sevens]), np.mean([rank(plain, s) for s in sevens]) / 3)
        self.assertEqual(jc.order_key("a", 2.0) * 2, jc.order_key("a", 1.0))

    def test_inclusion_probabilities_add_up_to_the_sample_and_follow_the_boosts(self):
        boosts = [1.0] * 90 + [5.0] * 10
        for n in (10, 30, 60):
            p = jc.inclusion(boosts, n)
            self.assertAlmostEqual(sum(p), n, places=5)
            self.assertTrue(all(0 <= x <= 1 for x in p))
            self.assertAlmostEqual(p[95] / p[0], 5.0) if p[95] < 1 else self.assertEqual(p[95], 1.0)
        self.assertEqual(jc.inclusion([1.0, 2.0], 5), [1.0, 1.0])
        self.assertEqual(jc.inclusion([1.0, 2.0], 0), [0.0, 0.0])
        self.assertTrue(all(x <= 1 for x in jc.inclusion([1.0, 1000.0, 1.0, 1.0], 3)))

    def test_the_weights_bring_the_oversampled_strata_back(self):
        """Sampling by the plan's order and weighting by 1 / inclusion estimates the pool's mix: a rare group taken more often is weighted less."""
        rare = [f"human:q{i}|f17" for i in range(200)]
        common = [f"human:c{i}|f17" for i in range(1800)]
        boost = {**{s: 5.0 for s in rare}, **dict.fromkeys(common, 1.0)}
        order = sorted(boost, key=lambda s: jc.order_key(s, boost[s]))
        n = 600
        taken = set(order[:n])
        pi = dict(zip(boost, jc.inclusion([boost[s] for s in boost], n), strict=True))
        share_taken = sum(s in taken for s in rare) / n
        share_weighted = sum(1 / pi[s] for s in rare if s in taken) / sum(1 / pi[s] for s in taken)
        self.assertGreater(share_taken, 0.25)  # rare starts are over-represented in the sample (200 of 2000 is 10%)
        self.assertAlmostEqual(share_weighted, 0.1, delta=0.03)  # and the weights give the pool's share back

    def test_the_work_order_takes_blocks_of_each_cell_in_turn_up_to_its_quota(self):
        cells = {"a": [f"a{i}" for i in range(7)], "b": [f"b{i}" for i in range(3)], "c": [f"c{i}" for i in range(10)]}
        order = jc.plan_order(cells, {"a": 5, "b": 9, "c": 0}, block=2, cell_order=("a", "b", "c"))
        self.assertEqual(order, ["a0", "a1", "b0", "b1", "a2", "a3", "b2", "a4"])
        self.assertEqual(jc.plan_order({}, {"a": 5}, block=2, cell_order=("a",)), [])


class LabelTests(unittest.TestCase):
    def test_root_value_is_the_visit_weighted_mean(self):
        ch = [child(1, 60, 10.0), child(2, 40, 20.0), child(3, 5, None)]
        self.assertAlmostEqual(jc.root_value(ch), 14.0)
        self.assertIsNone(jc.root_value([child(1, 0, 1.0)]))

    def test_a_label_carries_the_move_the_distribution_and_the_rungs(self):
        game = rec("game", [child(1, 400, 12.0), child(2, 80, 11.0, pd=0.2)])
        r2 = rec(2, [child(1, 800, 13.0, pd=0.1), child(2, 160, 9.0, pd=0.5)], cpu=3.0, duel=duel(4.0, 1.0))
        lab = jc.label_of(r2, "settled", 2, [{**game, "rel": 1, "from_game": True}, r2])
        self.assertEqual(lab["tier"], "settled")
        self.assertEqual(lab["move"], {"a": 1, "t": -1, "kind": "card", "card": 7})
        self.assertAlmostEqual(lab["root_value"], (800 * 13.0 + 160 * 9.0) / 960)
        self.assertEqual(lab["visits"], 960)
        self.assertEqual([c["a"] for c in lab["children"]], [1, 2])
        self.assertEqual(lab["duel"]["mean"], 4.0)
        self.assertAlmostEqual(lab["p_death"], (800 * 0.1 + 160 * 0.5) / 960)
        self.assertEqual([r["rel"] for r in lab["rungs"]], [1, 2])
        self.assertTrue(lab["rungs"][0]["from_game"])
        json.dumps(lab)

    def test_the_ladder_stops_at_the_first_rung_where_the_rule_holds(self):
        game = rec("game", [child(1, 400, 12.0), child(2, 80, 11.0)])
        called: list[int] = []

        def search(rel):
            called.append(rel)
            return rec(rel, [child(1, 900, 13.0), child(2, 60, 5.0)], cpu=2.0, duel=duel(6.0, 1.0))

        res = jc.settle(game, search)
        self.assertEqual((res["tier"], res["rel"], called), ("settled", 2, [2]))
        self.assertEqual(res["cpu_s"], 2.1)  # the rung and its duel; the game's own search was paid for by the game

    def test_a_move_that_changes_or_an_unclear_gap_goes_on_to_the_cap(self):
        game = rec("game", [child(1, 400, 12.0), child(2, 80, 11.0)])

        def search(rel):
            if rel == 2:  # the move changes
                return rec(2, [child(2, 900, 13.0), child(1, 60, 5.0)], duel=duel(6.0, 1.0))
            return rec(4, [child(2, 900, 13.0), child(1, 60, 5.0)], duel=duel(5.0, 1.0))

        res = jc.settle(game, search)
        self.assertEqual((res["tier"], res["rel"], res["rec"]["rel"]), ("settled", 4, 4))

        def unclear(rel):
            return rec(rel, [child(1, 900, 13.0), child(2, 60, 5.0)], duel=duel(1.0, 1.0))

        soft = jc.settle(game, unclear)
        self.assertEqual((soft["tier"], soft["rel"], soft["rec"]["rel"]), ("soft", 4, 4))
        self.assertEqual(len(soft["rungs"]), 3)

    def test_a_capped_rung_says_nothing_and_the_label_is_the_last_uncapped_rung(self):
        game = rec("game", [child(1, 400, 12.0), child(2, 80, 11.0)])

        def search(rel):
            return rec(rel, [child(1, 900, 13.0), child(2, 60, 5.0)], capped=(rel == 4), duel=duel(6.0 if rel == 4 else 1.0, 1.0))

        res = jc.settle(game, search)
        self.assertEqual((res["tier"], res["rel"]), ("soft", 2))
        self.assertEqual(res["rec"]["rel"], 2)

    def test_a_rung_with_no_duel_is_never_settled_in_paired_mode(self):
        game = rec("game", [child(1, 400, 12.0), child(2, 80, 11.0)])
        res = jc.settle(game, lambda rel: rec(rel, [child(1, 900, 13.0), child(2, 60, 5.0)]))
        self.assertEqual(res["tier"], "soft")

    def test_moves_round_trip(self):
        self.assertEqual(jc.moves_of(jc.moves_string([[3, -1], [0, 2]])), [[3, -1], [0, 2]])
        self.assertEqual(jc.moves_of(""), [])
        self.assertEqual(jc.moves_string([]), "")


def runs_outside_the_exam(n: int) -> list[str]:
    import human_starts as hs

    return [r for r in (f"run{i}" for i in range(10 * n)) if hs.exam_bucket(r) != 0][:n]


class StoreTests(unittest.TestCase):
    RUNS = runs_outside_the_exam(12)

    def make(self, d: str) -> Path:
        root = Path(d)
        runs = self.RUNS
        rows = [row(i, r, tier="settled" if i % 2 else "soft") for r in runs for i in range(3)]
        toks = [tokens(5 + i, seed=hash((r, i)) % 1000) for r in runs for i in range(3)]
        jc.write_shard(root / "shards" / "human-boss-act2-0000", rows, toks)
        sp = [row(i, f"sp{j}", src="selfplay", kind="elite", act=1, stage="late", tier="played", sub="night", human=None) for j in range(4) for i in range(2)]
        for r in sp:
            r["start"] = f"night:{r['run']}|f8"
        jc.write_shard(root / "shards" / "selfplay-elite-act1-0000", sp, [tokens(4, seed=i) for i in range(len(sp))])
        return root

    def test_tokens_round_trip_through_a_shard(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.make(d)
            c = jc.Corpus(root)
            got = list(c.iter_positions(src="human"))
            self.assertEqual(len(got), 36)
            first = got[0]
            want = tokens(5, seed=hash((self.RUNS[0], 0)) % 1000)
            for k in ("types", "ids", "owners", "feats"):
                np.testing.assert_array_equal(first.tokens[k], want[k])
            self.assertEqual(first.row["pid"], f"{self.RUNS[0]}|f33#0")
            self.assertEqual(len(got[7].tokens["types"]), 5 + got[7].row["k"])

    def test_the_loader_filters_by_source_kind_act_split_tier_and_stage(self):
        with tempfile.TemporaryDirectory() as d:
            c = jc.Corpus(self.make(d))
            self.assertEqual(len(list(c.iter_rows())), 44)
            self.assertEqual(len(list(c.iter_rows(src="selfplay"))), 8)
            self.assertEqual(len(list(c.iter_rows(kind="elite"))), 8)
            self.assertEqual(len(list(c.iter_rows(act=1))), 8)
            self.assertEqual(len(list(c.iter_rows(src="human", tiers=["settled"]))), 12)
            self.assertEqual(len(list(c.iter_rows(src="human", stages=["late"]))), 0)
            train = list(c.iter_rows(src="human", split="train"))
            held = list(c.iter_rows(src="human", split="heldout"))
            self.assertEqual(len(train) + len(held), 36)
            self.assertTrue({r["run"] for r, _, _ in train}.isdisjoint({r["run"] for r, _, _ in held}))
            self.assertTrue(all(jc.split_of(r["run"]) == "train" for r, _, _ in train))
            with self.assertRaises(ValueError):
                list(c.iter_rows(split="valid"))

    def test_a_human_row_of_the_exam_fifth_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            bad = run_in_bucket(0)
            jc.write_shard(Path(d) / "shards" / "human-boss-act2-0000", [row(0, bad)], [tokens(4)])
            with self.assertRaisesRegex(AssertionError, "exam fifth"):
                list(jc.Corpus(d).iter_rows())
            ok = run_in_bucket(1)
            jc.write_shard(Path(d) / "shards" / "human-boss-act2-0000", [row(0, ok)], [tokens(4)])
            self.assertEqual(len(list(jc.Corpus(d).iter_rows())), 1)

    def test_a_row_whose_split_is_not_its_runs_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            r = row(0, run_in_bucket(2))
            r["split"] = "heldout" if r["split"] == "train" else "train"
            jc.write_shard(Path(d) / "shards" / "human-boss-act2-0000", [r], [tokens(4)])
            with self.assertRaisesRegex(AssertionError, "split"):
                list(jc.Corpus(d).iter_rows())

    def test_a_shard_needs_a_token_set_a_row(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            jc.write_shard(Path(d) / "x", [row(0, "a")], [])

    def test_the_arrays_are_the_layout_the_judge_reads(self):
        with tempfile.TemporaryDirectory() as d:
            z = jc.Corpus(self.make(d)).arrays(src="human", split="all")
            n = 36
            self.assertEqual(len(z["lens"]), n)
            self.assertEqual(int(z["lens"].sum()), len(z["types"]))
            self.assertEqual(z["feats"].shape, (len(z["types"]), 18))
            self.assertEqual(z["labels"].shape, (n, 4))
            self.assertEqual(z["meta"].shape, (n, 8))
            self.assertEqual(len(z["group"]), n)
            self.assertEqual(len(z["start_ids"]), 12)
            self.assertEqual(len(z["start_heldout"]), 12)
            self.assertEqual(list(z["cats"]), ["act2-boss"])
            self.assertEqual(z["group"].max(), 11)
            self.assertTrue(np.allclose(z["weight"], 2.0))
            # a human row's label: won, end HP over the max HP, potions held, the unit L0
            np.testing.assert_allclose(z["labels"][0], [1.0, 30 / 80, 1.0, 0.7], rtol=1e-6)
            self.assertAlmostEqual(float(z["meta"][0, 4]), 12.5, places=4)
            t = jc.Corpus(self.make(tempfile.mkdtemp())).arrays(src="human", label="teacher", weight="none")
            np.testing.assert_allclose(t["labels"][0], [1.0, 20 / 80, 1.0, 0.6], rtol=1e-6)
            self.assertTrue(np.allclose(t["weight"], 1.0))

    def test_the_default_label_is_the_teachers_for_self_play_and_the_humans_for_human_rows(self):
        h, s = row(0, "a"), row(0, "b", src="selfplay", human=None)
        self.assertEqual(jc.label_row(h)[1], 30 / 80)
        self.assertEqual(jc.label_row(s)[1], 20 / 80)
        self.assertIsNone(jc.label_row(s, "human"))
        lost = row(0, "c", teacher={"won": False, "end_hp": 0, "max_hp": 80, "L0": 0.0})
        self.assertEqual(jc.label_row(lost, "teacher"), [0.0, 0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            jc.Corpus("x").arrays(label="bogus")

    def test_weight_modes(self):
        r = row(0, "a", w=4.0, w_cell=0.5)
        self.assertEqual([jc.row_weight(r, m) for m in ("pool", "cell", "none")], [4.0, 0.5, 1.0])
        with self.assertRaises(ValueError):
            jc.row_weight(r, "bogus")
        with self.assertRaises(ValueError):
            jc.row_weight(r, "balanced")
        self.assertEqual(jc.Position(r, {}, "s").weight, 4.0)

    def test_balanced_weights_give_every_kind_and_act_cell_the_same_total(self):
        with tempfile.TemporaryDirectory() as d:
            z = jc.Corpus(self.make(d)).arrays(weight="balanced")  # 36 human act-2 boss rows and 8 self-play act-1 elite rows
            cat = lambda i: z["cats"][z["start_cat"][z["group"][i]]]  # noqa: E731
            by = {c: float(sum(z["weight"][i] for i in range(len(z["group"])) if cat(i) == c)) for c in ("act2-boss", "act1-elite")}
            self.assertAlmostEqual(by["act2-boss"], by["act1-elite"], places=3)
            self.assertAlmostEqual(float(z["weight"].mean()), 1.0, places=3)

    def test_the_shard_listing_reads_source_kind_and_act_from_the_name(self):
        with tempfile.TemporaryDirectory() as d:
            c = jc.Corpus(self.make(d))
            self.assertEqual([p.stem for p in c.shards()], ["human-boss-act2-0000", "selfplay-elite-act1-0000"])
            self.assertEqual([p.stem for p in c.shards(src="human")], ["human-boss-act2-0000"])
            self.assertEqual(c.shards(kind="hallway"), [])
            self.assertEqual(c.shards(act=2)[0].stem, "human-boss-act2-0000")


class CoverageTests(unittest.TestCase):
    def test_a_cell_under_the_minimum_is_not_shown_and_the_rest_is_counted(self):
        rows = [row(i, f"run{i % 30}", kind="boss", act=2, arch="attack") for i in range(120)]
        rows += [row(i, f"run{i}", kind="elite", act=1, arch="block", relics=[3, 77]) for i in range(12)]  # twelve runs: under 20
        md = jc.coverage_md(rows, {"cell": 100, "relic": 50}, {"relic": {3: "Burning Blood"}})
        self.assertIn("| act2-boss | 120 | 30 | yes |", md)
        self.assertIn("| act1-elite | <20 | <20 | no |", md)
        self.assertIn("| attack | 120 | 30 |", md)
        self.assertIn("| block | <20 | <20 |", md)
        self.assertIn("| Burning Blood | 132 | 30 |", md)  # every row carries relic 3; 30 runs among the bosses and the elites' twelve
        self.assertNotIn("| 77 |", md)  # a relic in under 20 runs is not listed
        self.assertIn("Source: human", md)
        self.assertNotIn("run0", md)  # no run id is printed

    def test_each_source_has_its_own_tables(self):
        rows = [row(i, f"run{i % 25}") for i in range(50)] + [row(i, f"sp{i % 25}", src="selfplay", tier="played") for i in range(50)]
        md = jc.coverage_md(rows, {})
        self.assertIn("Source: human", md)
        self.assertIn("Source: selfplay", md)
        self.assertIn("| played | 50 | 25 |", md)


class RealEncoderTests(unittest.TestCase):
    """The token layout the loader reads is the encoder's (needs the emulator library)."""

    def test_the_constants_and_the_vocabulary_are_the_encoders(self):
        try:
            import combat_tokens as ct
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"no emulator library: {e}")
        self.assertEqual((jc.T_GLOBAL, jc.T_HAND, jc.T_DRAW, jc.T_DISCARD, jc.T_EXHAUST, jc.T_POTION, jc.T_HAND_LOCKED, jc.T_RELIC),
                         (ct.TYPE_GLOBAL, ct.TYPE_HAND, ct.TYPE_DRAW, ct.TYPE_DISCARD, ct.TYPE_EXHAUST, ct.TYPE_POTION, ct.TYPE_HAND_LOCKED, ct.TYPE_RELIC))
        self.assertEqual((jc.F_HP, jc.F_MAGNITUDE, 18), (ct.F_HP, ct.F_MAGNITUDE, ct.N_FEATURES))
        inv_card, inv_relic = jc.vocab_inverse()
        self.assertEqual(inv_card, {i: c for c, i in ct.VOCAB.card.items()})
        self.assertEqual(inv_relic, {i: r for r, i in ct.VOCAB.relic.items()})
        self.assertEqual(ct.TYPE_END_TURN, 10)


if __name__ == "__main__":
    unittest.main()
