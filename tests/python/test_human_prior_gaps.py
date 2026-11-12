"""The human prior's state gaps (rl/human_prior.py v3, rl/human_prior_infer.py; D58 addenda 14 and 16).

Pinned: what a human record gives of block G (the "?" rooms' state, by the shared rule `features.unknown_room_odds` over the "?" rooms entered this act before the
decision, with what the roll could not pick from the room before, the rooms after and Juzu Bracelet; the merchant's removals bought; never the relic counters),
a decision's value at each floor of a hand-made run, where an ambiguous position makes the rest of the act's "?" state unknown, a v3 shard is the v1 shard plus
`gaps` and `gaps_ix` (every other array byte for byte), the row's block G is its afterstate's (and equals what `option_gaps` gives at play), a net with the columns
starts as the plain one, the prior round-trips from the net through `export_net` to the numpy scorers and learns, and the refusals.
"""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _path in (HERE, ROOT / "rl", ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import human_prior as hp  # noqa: E402
import human_prior_infer as hpi  # noqa: E402
import human_starts as hs  # noqa: E402
import test_human_prior as thp  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover
    torch = None

from rl.run_train import features  # noqa: E402

UNKNOWN = 7  # the map's "?" node type


class GapsFixture(unittest.TestCase):
    """The hand-made run of `test_human_prior.HumanPriorV1Tests` with some of its rooms turned into "?" rooms."""

    @classmethod
    def setUpClass(cls):
        thp.HumanPriorV1Tests.setUpClass()
        cls.h = thp.HumanPriorV1Tests("test_map_identity_check")
        cls.t = thp.HumanPriorV1Tests.t
        cls.train_id = thp.HumanPriorV1Tests.train_id

    def record(self, unknown: dict[int, str] | None = None, removal_at_shop: bool = True, relics: tuple[str, ...] = ()):
        """`synthetic_run` with the points `unknown` (index -> the room type it turned out to be) turned into "?" rooms, a removal bought at its shop, and `relics`
        held from the start."""
        r = copy.deepcopy(thp.synthetic_run(self.train_id))
        pts = r["map_point_history"][0]
        for i, room in (unknown or {}).items():
            pts[i]["map_point_type"] = "unknown"
            pts[i]["rooms"] = [{"room_type": room, "model_id": "EVENT.X" if room == "event" else "ENCOUNTER.X"}]
        if removal_at_shop:
            pts[4]["player_stats"][0]["cards_removed"] = [{"id": "CARD.STRIKE_IRONCLAD"}]
        for name in relics:
            r["players"][0]["relics"].append({"id": f"RELIC.{name}", "floor_added_to_deck": 1})
        return r

    def truth(self, unknown_nodes: tuple[int, ...] = (), cols=None):
        """The truth record of the same act, the nodes at rows `unknown_nodes` (column 3) typed "?"."""
        nodes = [list(n) for n in self.h.NODES]
        for n in nodes:
            if n[0] == 3 and n[1] in unknown_nodes:
                n[2] = UNKNOWN
        doc = self.h.truth()
        doc["acts"][0] = dict(doc["acts"][0], nodes=nodes, **({"cols": cols, "unique": False} if cols else {}))
        return doc

    def build(self, record, truth, gaps=True):
        stats, seen, price = Counter(), Counter(), Counter()
        specs = hp.run_decisions(record, self.t, stats, seen, price, truth=truth, gaps=gaps)
        return specs, stats

    @staticmethod
    def by_floor(specs, kind):
        return {s["floor"]: s for s in specs if s["kind"] == kind}


class DerivationTests(GapsFixture):
    def test_the_rooms_of_a_decision_are_the_shared_rule_over_the_visits_before_it(self):
        specs, stats = self.build(self.record({1: "monster", 2: "event"}), self.truth((1, 2)))
        rooms = {(s["kind"], s["floor"]): s["gaps"]["rooms"] for s in specs}
        # the route decisions use the state after the floor before: before floor 1 nothing was visited, before floor 4 both "?" were
        self.assertEqual(rooms[("route", 1)], [0, 100, 20, 30])
        self.assertEqual(rooms[("route", 3)], [2, 200, 60, 90])
        # the others use the state with the room they stand in: floor 2 is the first "?" (a fight: its odds reset, the others climbed), floor 3 the second (an event)
        self.assertEqual(rooms[("card", 2)], [1, 100, 40, 60])
        self.assertEqual(rooms[("card", 3)], [2, 200, 60, 90])
        self.assertEqual(rooms[("shop", 5)], [2, 200, 60, 90])
        self.assertEqual(rooms[("rest", 6)], [2, 200, 60, 90])
        self.assertEqual(stats["gaps|visits"], 2)
        self.assertEqual(stats["gaps|visits_unplaced"], 0)
        # every number is `features.unknown_room_odds` over the visits, computed here by hand from the same rule
        F = features
        self.assertEqual(rooms[("card", 3)], list(F.unknown_room_odds([F.UnknownVisit(1), F.UnknownVisit(7)])))
        # a run with no "?" room has the start state throughout
        plain, _ = self.build(self.record(), self.truth())
        self.assertTrue(all(s["gaps"]["rooms"] == [0, 100, 20, 30] for s in plain))

    def test_the_removals_bought_before_a_decision(self):
        specs, _ = self.build(self.record(), self.truth())
        removals = {(s["kind"], s["floor"]): s["gaps"]["removals"] for s in specs}
        self.assertEqual(removals[("card", 2)], [0, 0])
        self.assertEqual(removals[("shop", 5)], [0, 0])  # the quote the visit opened with: the removal bought in it is not yet
        self.assertEqual(removals[("rest", 6)], [1, 0])
        self.assertEqual(removals[("rest", 7)], [1, 0])
        # a run with no removal at its shop
        none, _ = self.build(self.record(removal_at_shop=False), self.truth())
        self.assertTrue(all(s["gaps"]["removals"] == [0, 0] for s in none))
        # the price the block writes: 100 + 50 each at Ascension 10
        view = hp.spec_view(self.by_floor(specs, "rest")[6])
        x = features.gaps_block(view)
        self.assertAlmostEqual(float(x[features.GAPS_REMOVALS_AT]), 1.5, places=6)

    def test_the_counters_are_never_known_on_a_human_row_and_the_other_parts_are(self):
        specs, _ = self.build(self.record({1: "monster"}), self.truth((1,)))
        for sp in specs:
            for x in hp.spec_gaps(sp):
                known = x[features.GAPS_KNOWN_AT : features.GAPS_KNOWN_AT + 3].tolist()
                self.assertEqual(known, [0.0, 1.0, 1.0], (sp["kind"], sp["floor"]))
                self.assertFalse(x[: features.GAPS_N_COUNTERS].any())

    def test_a_question_mark_right_after_a_shop_cannot_roll_a_shop(self):
        # point 5 (the first rest, right after the shop) is a "?" that turned out to be a fight: the shop was not in the roll, so its odds stay
        specs, _ = self.build(self.record({1: "monster", 2: "event", 5: "monster"}), self.truth((1, 2, 5)))
        self.assertEqual(self.by_floor(specs, "rest")[6]["gaps"]["rooms"], [3, 100, 80, 90])
        # a second visit in a row after the shop would have raised the shop odds had it been in the set
        loose, _ = self.build(self.record({1: "monster", 2: "event"}), self.truth((1, 2)))
        self.assertEqual(self.by_floor(loose, "rest")[6]["gaps"]["rooms"], [2, 200, 60, 90])

    def test_juzu_bracelet_takes_the_fights_out_of_the_roll(self):
        without, _ = self.build(self.record({1: "event"}), self.truth((1,)))
        with_juzu, _ = self.build(self.record({1: "event"}, relics=("JUZU_BRACELET",)), self.truth((1,)))
        self.assertEqual(self.by_floor(without, "card")[2]["gaps"]["rooms"], [1, 200, 40, 60])
        self.assertEqual(self.by_floor(with_juzu, "card")[2]["gaps"]["rooms"], [1, 100, 40, 60])  # the fight odds do not climb

    def test_an_ambiguous_position_leaves_the_rest_of_the_acts_question_marks_unknown(self):
        # the human's node at point 1 cannot be told: the "?" there cannot be placed, so the state after it is unknown for the rest of the act
        truth = self.truth((1, 2), cols=[3, -1, 3, 3, 3, 3, 3])
        specs, stats = self.build(self.record({1: "monster", 2: "event"}), truth)
        later = [s for s in specs if s["floor"] >= 3]
        self.assertTrue(later)
        self.assertTrue(all(s["gaps"]["rooms"] is None for s in later), [(s["kind"], s["floor"], s["gaps"]) for s in later])
        self.assertEqual(stats["gaps|visits_unplaced"], 1)
        # ... and the block says so: known bit 0, zeros there; the removals are still known
        x = hp.spec_gaps(later[0])[0]
        self.assertEqual(x[features.GAPS_KNOWN_AT : features.GAPS_KNOWN_AT + 3].tolist(), [0.0, 0.0, 1.0])
        self.assertFalse(x[features.GAPS_ROOMS_AT : features.GAPS_REMOVALS_AT].any())

    def test_nothing_changes_without_the_flag_and_the_rows_are_the_same_with_it(self):
        record, truth = self.record({1: "monster", 2: "event"}), self.truth((1, 2))
        plain, _ = self.build(record, truth, gaps=False)
        gapped, _ = self.build(record, truth, gaps=True)
        self.assertTrue(all("gaps" not in s for s in plain))
        self.assertEqual(len(plain), len(gapped))
        for a, b in zip(plain, gapped):
            self.assertEqual({k: v for k, v in b.items() if k != "gaps"}, a)
            for ra, rb in zip(hp.spec_rows(a), hp.spec_rows(b)):
                self.assertEqual(ra.tobytes(), rb.tobytes())  # the judgement's rows are byte for byte as they were

    def test_a_run_left_out_whole_has_no_gaps_to_derive(self):
        specs, stats = self.build(self.record({1: "monster"}), {**self.truth((1,)), "compass": True})
        self.assertEqual(specs, [])
        self.assertEqual(stats["gaps|visits"], 0)


class ShardTests(GapsFixture):
    def specs(self):
        specs = self.build(self.record({1: "monster", 2: "event"}), self.truth((1, 2)))[0]
        for sp in specs:
            sp["run_ix"] = 0
        return specs

    def write(self, tmp: str, specs, name: str, **kw) -> Path:
        path = Path(tmp) / name
        hp.write_shard(path, specs, [self.train_id], thp.fake_judge(), {}, None, **kw)
        return path

    def test_a_v3_shard_is_the_v1_shard_plus_the_gaps(self):
        specs = self.specs()
        with tempfile.TemporaryDirectory() as tmp:
            v1 = self.write(tmp, specs, "v1.npz")
            v3 = self.write(tmp, specs, "v3.npz", state_gaps=True)
            with np.load(v1) as a, np.load(v3) as b:
                self.assertEqual(sorted(set(b.files) - set(a.files)), ["gaps", "gaps_ix"])
                for key in a.files:
                    self.assertEqual(a[key].tobytes(), b[key].tobytes(), key)  # every v1 array is unchanged
                rows = len(a["dense"])
                self.assertEqual(b["gaps_ix"].shape, (rows,))
                self.assertEqual(b["gaps"].dtype, np.float32)
                self.assertEqual(b["gaps"].shape[1], features.GAPS_SIZE)
                self.assertLess(len(b["gaps"]), rows)  # rows that share a block share an entry
                self.assertEqual(len({g.tobytes() for g in b["gaps"]}), len(b["gaps"]))
            # with the map grid too: both tables beside the rows
            both = self.write(tmp, specs, "both.npz", map_grid=True, state_gaps=True)
            with np.load(both) as z:
                self.assertTrue({"grid", "grid_ix", "gaps", "gaps_ix"} <= set(z.files))

    def test_a_row_carries_the_block_of_its_afterstate_and_play_builds_the_same_block(self):
        specs = self.specs()
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp, specs, "shard-00000.npz", state_gaps=True)
            ds = hp.Dataset.load(Path(tmp), [hs.exam_bucket(self.train_id)], None)
        self.assertTrue(ds.has_gaps)
        start = 0
        for sp in specs:
            view = hp.spec_view(sp)
            options = [hp._option(o, view, j) for j, o in enumerate(sp["options"])]
            want = np.stack(hp.spec_gaps(sp))
            np.testing.assert_array_equal(ds.gaps_rows(np.arange(start, start + len(options))), want)
            np.testing.assert_array_equal(hpi.option_gaps(view, options), want)  # the play side and the build side agree
            start += len(options)
        # a v1 dataset has none
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp, specs, "shard-00000.npz")
            plain = hp.Dataset.load(Path(tmp), [hs.exam_bucket(self.train_id)], None)
        self.assertFalse(plain.has_gaps)
        with self.assertRaisesRegex(ValueError, "state gaps"):
            plain.gaps_rows(np.arange(2))
        with self.assertRaisesRegex(ValueError, "state gaps"):
            hp.row_scores_np(plain, None, lambda d, c, gaps=None: np.zeros(len(d)), gaps=True)

    def test_a_folder_with_and_without_the_gaps_is_refused_and_a_spec_needs_its_gaps(self):
        specs = self.specs()
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp, specs, "shard-00000.npz", state_gaps=True)
            self.write(tmp, specs, "shard-00001.npz")
            with self.assertRaises(SystemExit):
                hp.Dataset.load(Path(tmp), [hs.exam_bucket(self.train_id)], None)
        bare = [{k: v for k, v in sp.items() if k != "gaps"} for sp in specs]
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "block G"):
            self.write(tmp, bare, "shard-00000.npz", state_gaps=True)
        v0 = [{k: v for k, v in sp.items() if k not in ("map_nodes",)} for sp in specs]
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            self.write(tmp, v0, "shard-00000.npz", state_gaps=True)

    def test_build_refusals_and_the_default_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):  # the derivation reads the true map's route
                hp.main(["build", "--state-gaps", "--out-dir", tmp])
            for name in ("decisions-v1", "decisions-v2"):  # an earlier version's shards are never overwritten
                with self.assertRaises(SystemExit):
                    hp.main(["build", "--state-gaps", "--truth", tmp, "--decisions", name, "--out-dir", tmp])
        import argparse

        self.assertEqual("decisions-v3", hp.default_decisions(argparse.Namespace(state_gaps=True, map_grid=True, truth="x")))
        self.assertEqual("decisions-v2", hp.default_decisions(argparse.Namespace(state_gaps=False, map_grid=True, truth="x")))
        self.assertEqual("decisions-v1", hp.default_decisions(argparse.Namespace(state_gaps=False, map_grid=False, truth="x")))
        self.assertEqual("decisions", hp.default_decisions(argparse.Namespace(state_gaps=False, map_grid=False, truth="")))

    def test_the_counts_report_the_gaps(self):
        record, truth = self.record({1: "monster", 2: "event"}), self.truth((1, 2))
        specs, stats = self.build(record, truth)
        for sp in specs:
            sp["run_ix"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, specs, "shard-00000.npz", state_gaps=True)
            summary = hp.summarize(Path(tmp), [{"stats": dict(stats), "seen": {}, "price": {}, "runs": 1, "path": str(path)}], Counter())
        counts = summary["state_gaps"]
        self.assertEqual(counts["visits"], 2)
        self.assertEqual(counts["decisions|route"], sum(s["kind"] == "route" for s in specs))
        self.assertEqual(counts["decisions|card|rooms_known"], counts["decisions|card"])  # every one of them placed here
        self.assertEqual(summary["state_gaps_check"]["visit 1"], f"<{hp.MIN_CELL}")  # a cell over so few visits is hidden
        # the check's cells: the first visit of an act was read at the start odds (a fight 0.10, a treasure room 0.02, a shop 0.03)
        big = Counter({"gaps|check|v1|n": 100, "gaps|check|v1|monster|observed": 12, "gaps|check|v1|monster|expected_per_mille": 10000,
                       "gaps|check|v1|shop|observed": 4, "gaps|check|v1|shop|expected_per_mille": 3000})
        cell = hp.gaps_check_table(big)["visit 1"]
        self.assertEqual((cell["n"], cell["monster"], cell["shop"]), (100, {"observed": 0.12, "expected": 0.1}, {"observed": 0.04, "expected": 0.03}))
        stats_one = self.build(record, truth)[1]
        self.assertEqual(stats_one["gaps|check|v1|n"], 1)  # the first "?" of the act: read at the start odds
        self.assertEqual(stats_one["gaps|check|v1|monster|observed"], 1)  # it was a fight
        self.assertEqual(stats_one["gaps|check|v1|monster|expected_per_mille"], 100)
        self.assertEqual(stats_one["gaps|check|v2|monster|observed"], 0)  # the second was an event, read at the odds the first one left
        self.assertEqual(stats_one["gaps|check|v2|treasure|expected_per_mille"], 40)
        # the same run with a node that cannot be placed: its later decisions are counted as not known
        specs2, stats2 = self.build(record, self.truth((1, 2), cols=[3, -1, 3, 3, 3, 3, 3]))
        self.assertEqual(stats2["gaps|visits_unplaced"], 1)
        self.assertLess(stats2["gaps|decisions|shop|rooms_known"], stats2["gaps|decisions|shop"])


@unittest.skipIf(torch is None, "torch is not installed")
class NetTests(GapsFixture):
    def v3_dataset(self, tmp: str, map_grid: bool = False):
        specs = ShardTests.specs(self)
        for sp in specs:
            sp["flags"] &= ~hp.F_REPLAY_MISMATCH  # the synthetic run's deck does not replay to its final deck: keep its decisions in a fit
        hp.write_shard(Path(tmp) / "shard-00000.npz", specs, [self.train_id], thp.fake_judge(), {}, None, map_grid=map_grid, state_gaps=True)
        return hp.Dataset.load(Path(tmp), [hs.exam_bucket(self.train_id)], None)

    def judge_file(self, tmp: str, plain) -> Path:
        state = {f"value/{k}": v.detach().numpy() for k, v in plain.state_dict().items() if not k.startswith("score.")}
        state["value/floor_head.weight"] = np.zeros((2, 256), np.float32)
        judge = Path(tmp) / "judge.npz"
        np.savez(judge, **state)
        return judge

    def test_a_net_with_the_columns_starts_as_the_plain_net(self):
        torch.manual_seed(3)
        plain = hp.make_net()
        g = features.GAPS_SIZE
        with tempfile.TemporaryDirectory() as tmp:
            judge = self.judge_file(tmp, plain)
            rng = np.random.default_rng(2)
            dense = torch.from_numpy(rng.random((9, 350)).astype(np.float32))
            counts = torch.from_numpy((rng.random((9, 1513)) < 0.01).astype(np.float32))
            block = torch.from_numpy(rng.random((9, g)).astype(np.float32))
            grid = torch.from_numpy((rng.random((9, features.MAP_GRID_SIZE)) < 0.3).astype(np.int8))
            exported = Path(tmp) / "plain.npz"
            hp.export_net(plain, exported, {})
            plain.eval()
            for kw in ({"state_gaps": True}, {"state_gaps": True, "map_grid": True}):
                net, net2 = hp.make_net(**kw), hp.make_net(**kw)
                hp.init_from_judge(net, judge)
                hp.init_from_prior(net2, exported)
                wide = 350 + 104 + (64 if kw.get("map_grid") else 0) + g
                self.assertEqual(net.hidden1.weight.shape, (256, wide))
                self.assertEqual(float(net.hidden1.weight[:, -g:].detach().abs().sum()), 0.0)  # the new columns are zeros
                net.score.load_state_dict(plain.score.state_dict())  # (the judge file has no score head)
                net2.score.load_state_dict(plain.score.state_dict())
                net.eval()
                net2.eval()
                extra = {"grid": grid} if kw.get("map_grid") else {}
                with torch.no_grad():
                    for n in (net, net2):
                        np.testing.assert_allclose(n(dense, counts, gaps=block, **extra).numpy(), plain(dense, counts).numpy(), rtol=1e-5, atol=1e-5)
                with self.assertRaises(ValueError):
                    net(dense, counts, **extra)  # a net with the columns never scores without block G
            # a start that already has the tower: only block G's columns are added
            tower_net = hp.make_net(map_grid=True)
            tower_path = Path(tmp) / "tower.npz"
            hp.export_net(tower_net, tower_path, {})
            both = hp.make_net(map_grid=True, state_gaps=True)
            hp.init_from_prior(both, tower_path)
            self.assertEqual(float(both.hidden1.weight[:, -g:].detach().abs().sum()), 0.0)
            np.testing.assert_array_equal(both.hidden1.weight[:, :-g].detach().numpy(), tower_net.hidden1.weight.detach().numpy())  # the tower's columns carried over

    def test_the_prior_round_trips_to_the_numpy_scorers_and_learns(self):
        torch.manual_seed(7)
        g = features.GAPS_SIZE
        with tempfile.TemporaryDirectory() as tmp:
            ds = self.v3_dataset(tmp)
            net = hp.make_net(state_gaps=True)
            with torch.no_grad():  # columns that speak
                net.hidden1.weight[:, -g:] = torch.randn(256, g) * 0.05
            gt = hp.Gpu(ds, None, torch.device("cpu"), 1.0)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
            net.train()
            before = net.hidden1.weight[:, -g:].detach().clone()
            for _ in range(2):
                loss = gt.loss(net, "card", torch.from_numpy(ds.kind_idx["card"]))
                self.assertTrue(torch.isfinite(loss).item())
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.assertFalse(torch.equal(before, net.hidden1.weight[:, -g:].detach()))  # the gradient reaches the columns
            path = Path(tmp) / "weights.npz"
            hp.export_net(net, path, {"kinds": ["card", "ancient", "rest", "shop", "route"]})
            prior = hp.NumpyPrior(path)
            self.assertTrue(prior.state_gaps and not prior.map_grid)
            net.eval()
            torch_scores = gt.scores(net)
            np.testing.assert_allclose(torch_scores, hp.row_scores_np(ds, None, prior), rtol=1e-4, atol=1e-4)
            scorer = hpi.PriorScorer.load(path)
            self.assertTrue(scorer.state_gaps)
            self.assertTrue(scorer.meta["state_gaps"])
            self.assertEqual(scorer.meta["gaps_version"], features.GAPS_VERSION)
            x = hp.assemble_np_v1(ds.dense)
            counts = ds.counts.astype(np.float32)
            rows = np.arange(len(x))
            got = scorer.logits(x, counts, None, ds.gaps_rows(rows))
            np.testing.assert_allclose(got, torch_scores, rtol=1e-4, atol=1e-4)
            with self.assertRaisesRegex(ValueError, "state gaps"):
                scorer.logits(x, counts)  # such a prior never scores without block G
            self.assertFalse(np.allclose(scorer.logits(x, counts, None, ds.gaps_rows(rows)[::-1].copy()), got))  # the block matters
            # a plain prior is handed block G and ignores it
            plain_path = Path(tmp) / "plain.npz"
            hp.export_net(hp.make_net(), plain_path, {})
            plain = hpi.PriorScorer.load(plain_path)
            self.assertFalse(plain.state_gaps)
            np.testing.assert_array_equal(plain.logits(x, counts), plain.logits(x, counts, None, ds.gaps_rows(rows)))

    def test_both_flags_together_and_the_width_checks(self):
        torch.manual_seed(5)
        with tempfile.TemporaryDirectory() as tmp:
            ds = self.v3_dataset(tmp, map_grid=True)
            net = hp.make_net(map_grid=True, state_gaps=True)
            g = features.GAPS_SIZE
            with torch.no_grad():
                net.hidden1.weight[:, -g:] = torch.randn(256, g) * 0.05
                net.hidden1.weight[:, -g - 64 : -g] = torch.randn(256, 64) * 0.05
            path = Path(tmp) / "weights.npz"
            hp.export_net(net, path, {})
            scorer = hpi.PriorScorer.load(path)
            self.assertTrue(scorer.map_grid and scorer.state_gaps)
            gt = hp.Gpu(ds, None, torch.device("cpu"), 1.0)
            net.eval()
            x = hp.assemble_np_v1(ds.dense)
            rows = np.arange(len(x))
            np.testing.assert_allclose(scorer.logits(x, ds.counts.astype(np.float32), ds.grid_rows(rows), ds.gaps_rows(rows)), gt.scores(net), rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(hp.row_scores_np(ds, None, hp.NumpyPrior(path)), gt.scores(net), rtol=1e-4, atol=1e-4)
            # the meta and the weights must agree
            with np.load(path) as z:
                arrays = {k: z[k] for k in z.files if k != "meta"}
            with self.assertRaisesRegex(ValueError, "first layer"):
                hpi.PriorScorer(arrays, {"map_grid": True})  # the columns are there and the meta does not say so
            with self.assertRaisesRegex(ValueError, "block G version"):
                hpi.PriorScorer(arrays, {"map_grid": True, "state_gaps": True, "gaps_version": 9})

    def test_fit_one_with_the_columns_runs_a_whole_variant(self):
        import types

        torch.manual_seed(2)
        with tempfile.TemporaryDirectory() as tmp:
            ds = self.v3_dataset(tmp)
            dev = torch.device("cpu")
            gt, gv = hp.Gpu(ds, None, dev, 1.0), hp.Gpu(ds, None, dev, 1.0)
            a = types.SimpleNamespace(seed=1, epochs=2, patience=2, batch=8, with_shops=False)
            res = hp.fit_one(gt, gv, "scratch", 1.0, a, ["card", "ancient", "rest", "shop", "route"], Path("unused"), state_gaps=True)
            self.assertTrue(res["state_gaps"])
            self.assertEqual(res["epochs_run"], 2)
            self.assertTrue(np.isfinite(res["best"]["score"]))
            hp.export_net(res["net"], Path(tmp) / "w.npz", {})
            import json

            with np.load(Path(tmp) / "w.npz") as z:
                meta = json.loads(str(z["meta"]))
            self.assertTrue(meta["state_gaps"])

    def test_a_dataset_without_the_gaps_cannot_train_a_net_that_reads_them(self):
        specs = ShardTests.specs(self)
        with tempfile.TemporaryDirectory() as tmp:
            hp.write_shard(Path(tmp) / "shard-00000.npz", specs, [self.train_id], thp.fake_judge(), {}, None)
            ds = hp.Dataset.load(Path(tmp), [hs.exam_bucket(self.train_id)], None)
            gt = hp.Gpu(ds, None, torch.device("cpu"), 1.0)
            net = hp.make_net(state_gaps=True)
            with self.assertRaisesRegex(ValueError, "state gaps"):
                gt.loss(net, "card", torch.from_numpy(ds.kind_idx["card"]))


if __name__ == "__main__":
    unittest.main()
