"""The human deck prior (rl/human_prior.py; milestones 2 and 3 of the 2026-11-05 strategy memo, sections 3.1 to 3.3 and 3.6).

The pinned facts: the exam fifth is never read (a run of the fifth is dropped before any worker sees it, a worker handed one stops,
and the written shard's run ids pass `assert_no_exam`); each decision kind has the option set the memo says (a card reward: the
offers and a skip, a screen of six split in two, five or seven offers dropped; an ancient: its three relics; a rest site: heal and
one smith option per distinct upgradable card, with the smith's target as the chosen row; a shop: one leave-or-buy pair per offer,
the bought card taken from the cards gained); a decision goes through build-row (shard) and a fit step (loss, backward, export) and the
numpy prior reproduces the torch net; the built counts follow counts.json's definitions (screens seen per kind and act); shop prices
follow the copy's rules. A synthetic run record stands in for the corpus; the real corpus is not read here.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import human_prior as hp  # noqa: E402
import human_starts as hs  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover
    torch = None

REAL_BUILT = Path("D:/spireline/referee/human-prior/decisions/counts.json")
REAL_REFERENCE = hp.REF_COUNTS


def _find_ids(want_exam: bool, n: int, prefix: str = "h") -> list[str]:
    out, i = [], 0
    while len(out) < n:
        rid = f"{prefix}{i:014x}"
        if (hs.exam_bucket(rid) == hs.EXAM_BUCKET) == want_exam:
            out.append(rid)
        i += 1
    return out


def _stats(hp_, max_hp=80, gold=99, **extra):
    return {"current_hp": hp_, "max_hp": max_hp, "current_gold": gold, "damage_taken": 0, "hp_healed": 0, "gold_spent": 0, "player_id": 1, **extra}


def _point(kind, stats, rooms=None):
    return {"map_point_type": kind, "player_stats": [stats], "rooms": rooms or [{"room_type": kind}]}


def _offers(*picked_ids, all_ids):
    return [{"card": {"id": f"CARD.{c}"}, "was_picked": c in picked_ids} for c in all_ids]


def synthetic_run(rid: str) -> dict:
    """A short Ironclad A10 run: an ancient, a card reward (take), a merged screen of six (a pick in each half), a screen of five
    (dropped), a shop (one card bought, one relic bought, 50 + 175 gold), a rest that smiths Bash, a rest that heals, then the end."""
    ancient = _stats(64, ancient_choice=[{"TextKey": k, "title": {"table": "relics"}, "was_chosen": k == "NEOWS_TALISMAN"}
                                         for k in ("FISHING_ROD", "NEOWS_TALISMAN", "LARGE_CAPSULE")],
                     relic_choices=[{"choice": "RELIC.NEOWS_TALISMAN", "was_picked": True}])
    fight = lambda **kw: _point("monster", _stats(60, gold=110, **kw), [{"model_id": "ENCOUNTER.SEAPUNK_WEAK", "monster_ids": ["MONSTER.SEAPUNK"], "room_type": "monster"}])  # noqa: E731
    six = _offers("BODY_SLAM", "IRON_WAVE", all_ids=["ANGER", "BODY_SLAM", "HEADBUTT", "POMMEL_STRIKE", "IRON_WAVE", "SHRUG_IT_OFF"])
    shop = _stats(60, gold=7 + 0, gold_spent=225, cards_gained=[{"id": "CARD.HAVOC"}],
                  card_choices=[{"card": {"id": f"CARD.{c}"}, "was_picked": False} for c in ("ANGER", "HEADBUTT")],
                  relic_choices=[{"choice": "RELIC.ANCHOR", "was_picked": True}, {"choice": "RELIC.BAG_OF_PREPARATION", "was_picked": False}],
                  bought_relics=["RELIC.ANCHOR"], potion_choices=[{"choice": "POTION.FIRE_POTION", "was_picked": False}])
    act1 = [
        _point("ancient", ancient, [{"model_id": "EVENT.NEOW", "room_type": "event"}]),
        fight(card_choices=_offers("ANGER", all_ids=["ANGER", "BODY_SLAM", "HEADBUTT"]), cards_gained=[{"id": "CARD.ANGER"}]),
        fight(card_choices=six, cards_gained=[{"id": "CARD.BODY_SLAM"}, {"id": "CARD.IRON_WAVE"}]),
        fight(card_choices=_offers(all_ids=["ANGER", "HEADBUTT", "HAVOC", "INFLAME", "RAMPAGE"])),
        _point("shop", shop),
        _point("rest_site", _stats(60, rest_site_choices=["SMITH"], upgraded_cards=["CARD.BASH"])),
        _point("rest_site", _stats(80, rest_site_choices=["HEAL"], hp_healed=20)),
    ]
    return {"acts": ["ACT.OVERGROWTH", "ACT.HIVE", "ACT.GLORY"], "ascension": 10, "game_mode": "standard", "modifiers": [], "was_abandoned": False,
            "win": False, "run_hash": rid, "seed": "SYNTH", "start_time": 1,
            "players": [{"character": "CHARACTER.IRONCLAD", "deck": [],
                         "relics": [{"id": "RELIC.BURNING_BLOOD", "floor_added_to_deck": 1}, {"id": "RELIC.NEOWS_TALISMAN", "floor_added_to_deck": 1},
                                    {"id": "RELIC.ANCHOR", "floor_added_to_deck": 5}]}],
            "map_point_history": [act1]}


def fake_judge():
    """Stands in for the run judgement: a fight block of zeros (the rows' shape is what these tests pin)."""
    return types.SimpleNamespace(base_size=1855, fight=types.SimpleNamespace(summary=lambda bases, queries: np.zeros((len(bases), 8), np.float32)))


def fake_donors():
    rng = np.random.default_rng(1)
    keys = np.array([[1, r] for r in range(17)], dtype=np.int32)
    return {"keys": keys, "n": np.full(len(keys), 8, dtype=np.int32), "c": rng.random((len(keys), 8, 25)).astype(np.float32),
            "a15": rng.random((len(keys), 8)).astype(np.float32)}


class HumanPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hp.stack()
        cls.t = hp.Tables()
        cls.train_id = _find_ids(False, 1)[0]
        cls.exam_id = _find_ids(True, 1, "x")[0]
        cls.record = synthetic_run(cls.train_id)

    def decisions(self):
        stats, seen, price = Counter(), Counter(), Counter()
        specs = hp.run_decisions(self.record, self.t, stats, seen, price)
        return specs, stats, seen, price

    # ----------------------------------------------------------------------------------------------- the exam fifth

    def test_exam_fifth_is_never_read(self):
        exam_run = synthetic_run(self.exam_id)
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "runs.jsonl.gz"
            with gzip.open(corpus, "wt", encoding="utf-8") as fh:
                for r in (exam_run, self.record):
                    fh.write('{"x":1}\n')  # lines the screen skips
                    fh.write(json.dumps(r, separators=(",", ":")).replace('"ascension":10', '"ascension":10,"pad":0').replace('"pad":0,', "", 1) + "\n")
            ids = frozenset()
            stats: Counter = Counter()
            lines = list(hp.iter_run_lines(corpus, ids, stats))
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["run_hash"], self.train_id)
            self.assertEqual(stats["exam_fifth_left_out"], 1)
            # a run listed in the ids file is out even when the hash puts it in a training fifth
            stats = Counter()
            self.assertEqual(list(hp.iter_run_lines(corpus, frozenset({self.train_id}), stats)), [])
            # a worker handed an exam line stops
            hp._WORK.update(t=self.t, judge=fake_judge(), donors=None, dindex={}, exam=frozenset())
            try:
                with self.assertRaises(AssertionError):
                    hp.build_chunk({"lines": [json.dumps(exam_run)], "out": str(Path(tmp) / "shard.npz")})
                ok = hp.build_chunk({"lines": [json.dumps(self.record)], "out": str(Path(tmp) / "shard-00000.npz")})
            finally:
                hp._WORK.clear()
            with np.load(ok["path"], allow_pickle=False) as z:
                self.assertEqual(hs.assert_no_exam([str(x) for x in z["run_ids"]], frozenset()), 1)
            with self.assertRaises(AssertionError):
                hs.assert_no_exam([self.exam_id], frozenset())

    # ----------------------------------------------------------------------------------------------- the option sets

    def test_option_sets(self):
        specs, stats, seen, _ = self.decisions()
        by = Counter(s["kind"] for s in specs)
        cards = [s for s in specs if s["kind"] == "card"]
        self.assertEqual(len(cards), 3)  # the take, and the two halves of the merged screen; the screen of five is dropped
        self.assertEqual([len(s["options"]) for s in cards], [4, 4, 4])
        self.assertTrue(all(s["options"][-1]["k"] == "card.skip" for s in cards))
        self.assertEqual([s["chosen"] for s in cards], [0, 1, 1])  # Anger; Body Slam (second of the first half); Iron Wave (second of the second)
        self.assertEqual(stats["drop|act1|card|card screen of 5 offers"], 1)
        self.assertEqual(sum(1 for s in cards if s["flags"] & hp.F_SPLIT), 2)
        anc = [s for s in specs if s["kind"] == "ancient"]
        self.assertEqual(len(anc), 1)
        self.assertEqual((len(anc[0]["options"]), anc[0]["chosen"]), (3, 1))
        self.assertTrue(all(o["k"] == "neow.relic" for o in anc[0]["options"]))
        rests = [s for s in specs if s["kind"] == "rest"]
        self.assertEqual(len(rests), 2)
        smith, heal = rests
        self.assertEqual(smith["options"][0]["k"], "rest.heal")
        distinct = {tuple(o["card"]) for o in smith["options"][1:]}
        self.assertEqual(len(distinct), len(smith["options"]) - 1)  # one smith option per distinct upgradable card
        self.assertEqual((smith["aux"], smith["options"][smith["chosen"]]["card"][0]), (hp.ROLE_SMITH, self.t.cards["BASH"]))
        self.assertEqual((heal["aux"], heal["chosen"]), (hp.ROLE_HEAL, 0))
        shop = [s for s in specs if s["kind"] == "shop"]
        self.assertTrue(all([o["k"] for o in s["options"]][0] == "shop.leave" and len(s["options"]) == 2 for s in shop))
        bought = sorted((s["offer"], s["options"][1]["id"]) for s in shop if s["chosen"] == 1)
        self.assertEqual(bought, sorted([(hp.OFFER_CARD, self.t.cards["HAVOC"]), (hp.OFFER_RELIC, self.t.relics["ANCHOR"])]))
        self.assertEqual(len(shop), 2 + 1 + 1 + 1 + 1)  # shelf: Anger and Cleave, Havoc (bought); two relics; one potion
        self.assertEqual(by["ancient"] + by["rest"], 3)

    def test_rows_have_the_judges_shape(self):
        specs, *_ = self.decisions()
        F = hp.stack().F
        for sp in specs:
            rows = hp.spec_rows(sp)
            self.assertEqual(len(rows), len(sp["options"]))
            for r in rows:
                self.assertEqual(r.shape, (F.BASE_SIZE,))
        card = next(s for s in specs if s["kind"] == "card")
        bases = np.stack(hp.spec_rows(card)).astype(np.float32)
        counts = hp.pack_counts(bases)
        self.assertEqual(counts.shape[1], F.COUNTS_END - F.E_OFFSET)
        dense = hp.pack_dense(bases, np.zeros((len(bases), 8), np.float32))
        self.assertEqual(dense.shape[1], hp.N_DENSE)
        # a taken card shows in the deck counts: the afterstate of option 0 holds one more card than the skip's
        self.assertEqual(int(counts[0].sum()) - int(counts[-1].sum()), 1)

    # --------------------------------------------------------------------------------------- build row -> fit step

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_round_trip_build_row_and_fit_step(self):
        specs, *_ = self.decisions()
        donors = fake_donors()
        with tempfile.TemporaryDirectory() as tmp:
            for sp in specs:
                sp["run_ix"] = 0
            path = Path(tmp) / "shard-00000.npz"
            n_rows = hp.write_shard(path, specs, [self.train_id], fake_judge(), hp.donor_index(donors), donors)
            self.assertEqual(n_rows, sum(len(s["options"]) for s in specs))
            ds = hp.Dataset.load(Path(tmp), [hs.exam_bucket(self.train_id)], donors)
            self.assertEqual(len(ds.dec), len(specs))
            self.assertEqual(ds.coverage["own"] + ds.coverage["borrowed"], len(set((s["region"], s["row"]) for s in specs)))
            net = hp.make_net()
            dev = torch.device("cpu")
            gt = hp.Gpu(ds, donors, dev, 2.0)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
            before = net.score.weight.detach().clone()
            net.train()
            total = 0.0
            for kind in ("card", "ancient", "rest", "shop"):
                idx = torch.from_numpy(ds.kind_idx[kind])
                loss = gt.loss(net, kind, idx)
                self.assertTrue(torch.isfinite(loss).item(), kind)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item())
            self.assertGreater(total, 0.0)
            self.assertFalse(torch.equal(before, net.score.weight))
            # the exported numpy prior reproduces the torch net on the same rows and donors
            hp.export_net(net, Path(tmp) / "weights.npz", {"kinds": ["card", "ancient", "rest", "shop"]})
            prior = hp.NumpyPrior(Path(tmp) / "weights.npz")
            torch_scores = gt.scores(net)
            numpy_scores = hp.row_scores_np(ds, donors, prior)
            np.testing.assert_allclose(torch_scores, numpy_scores, rtol=1e-4, atol=1e-4)
            m = hp.metrics_from_scores(ds, numpy_scores, ~hp._bool_unavailable(ds, True, False))
            self.assertIn("card", m)
            self.assertEqual(m["card"]["n"], 3)
            self.assertIn("rest_smith_target", m)
            self.assertTrue(0.0 <= m["card"]["top1"] <= 1.0)

    # ------------------------------------------------------------------------------------------------- counts contract

    def test_counts_follow_the_reference_definitions(self):
        specs, stats, seen, _ = self.decisions()
        # counts.json's definitions, applied to the record: card screens off the shop, rest points with a choice, ancient screens, shop visits
        want: Counter = Counter()
        for pt in self.record["map_point_history"][0]:
            ps = pt["player_stats"][0]
            if ps.get("card_choices") and pt["map_point_type"] != "shop":
                want["act1|card"] += 1
            if ps.get("rest_site_choices"):
                want["act1|rest"] += 1
            if ps.get("ancient_choice"):
                want["act1|ancient"] += 1
            if pt["map_point_type"] == "shop":
                want["act1|shop"] += 1
        self.assertEqual(dict(seen), dict(want))
        reference = {"decision_kinds_training_fifths": {f"act{a}": {"card_reward_screens": want["act1|card"] if a == 1 else 0,
                                                                     "rest_points_with_choice": want["act1|rest"] if a == 1 else 0,
                                                                     "ancient_screens": want["act1|ancient"] if a == 1 else 0,
                                                                     "points:shop": want["act1|shop"] if a == 1 else 0} for a in (1, 2, 3)}}
        report = hp.contract_report({"seen": dict(seen)}, reference)
        self.assertTrue(all(v["equal"] for v in report.values()), report)
        reference["decision_kinds_training_fifths"]["act1"]["card_reward_screens"] += 1
        self.assertFalse(hp.contract_report({"seen": dict(seen)}, reference)["card|act1"]["equal"])

    @unittest.skipUnless(REAL_BUILT.is_file() and REAL_REFERENCE.is_file(), "the real build or the reference counts are not on this machine")
    def test_real_build_matches_counts_json(self):
        built = json.loads(REAL_BUILT.read_text(encoding="utf-8"))
        if "contract" not in built:
            self.skipTest("the build on disk is a partial one")
        for key, row in built["contract"].items():
            self.assertTrue(row["equal"], (key, row))

    # --------------------------------------------------------------------------------------------------------- prices

    def test_shop_prices_follow_the_copys_rules(self):
        t = self.t
        V = hp.stack().V
        rare = next(i for i in range(1, 600) if i in V.cards() and V.card(i).rarity == "Rare" and i not in t.colorless)
        mid, lo, hi = hp.shop_price(hp.OFFER_CARD, rare, [], t)
        self.assertEqual((mid, lo, hi), (150, round(150 * 0.95), round(150 * 1.05)))
        colourless_uncommon = next(i for i in t.colorless if V.card(i).rarity == "Uncommon")
        self.assertEqual(hp.shop_price(hp.OFFER_CARD, colourless_uncommon, [], t)[0], round(75 * 1.15))
        relic = next(i for i, r in t.relic_rarity.items() if r == "Common")
        self.assertEqual(hp.shop_price(hp.OFFER_RELIC, relic, [], t)[0], 175)
        self.assertEqual(hp.shop_price(hp.OFFER_RELIC, relic, [t.membership], t)[0], 175 // 2)
        self.assertEqual(hp.shop_price(hp.OFFER_RELIC, relic, [t.courier], t)[0], int(175 * 0.8))
        self.assertEqual((hp.removal_price(0, [], t), hp.removal_price(2, [], t)), (100, 200))  # Inflation at Ascension 10
        chk = hp.price_check([(hp.OFFER_CARD, rare, False)], 100, 250, [], t)
        self.assertTrue(chk["within"] and chk["strict_within"])
        half = hp.price_check([(hp.OFFER_CARD, rare, False)], None, 75, [], t)  # the sale slot halves one coloured card
        self.assertTrue(half["within"] and not half["strict_within"])
        self.assertFalse(hp.price_check([(hp.OFFER_CARD, rare, False)], None, 400, [], t)["within"])

    def test_shop_visit_price_check_is_counted(self):
        _, _, _, price = self.decisions()
        self.assertEqual(price["visits"], 1)
        self.assertEqual(price["visits_with_spend"], 1)
        summary = hp.price_summary(price)
        self.assertEqual(summary["visits"], 1)
        self.assertIn("within10_spend", summary)


class HumanPriorV1Tests(unittest.TestCase):
    """v1: the true map (identity check, position of the human on it), the route kind, the replay's shop prices and their flags, the won input."""

    @classmethod
    def setUpClass(cls):
        hp.stack()
        cls.t = hp.Tables()
        cls.train_id = _find_ids(False, 1, "v")[0]
        cls.record = synthetic_run(cls.train_id)

    # a small act: start (3,0), three rooms in row 1, the human's path along column 3, boss at row 7
    NODES = [[3, 0, 8], [2, 1, 1], [3, 1, 1], [4, 1, 3], [2, 2, 4], [3, 2, 1], [3, 3, 1], [4, 3, 1], [3, 4, 4], [3, 5, 3], [3, 6, 3], [3, 7, 6]]
    EDGES = [[3, 0, 2, 1], [3, 0, 3, 1], [3, 0, 4, 1], [3, 1, 3, 2], [2, 1, 2, 2], [2, 2, 3, 3], [4, 1, 3, 2], [3, 2, 3, 3], [3, 2, 4, 3], [3, 3, 3, 4], [4, 3, 3, 4],
             [3, 4, 3, 5], [3, 5, 3, 6], [3, 6, 3, 7]]

    def visit(self, **kw):
        t = self.t
        v = {"floor": 5, "clean": True, "clean2": True, "shelf_ok": True, "removal": 100, "purchase": True, "spent": 225, "predicted": 225, "found_all": True,
             "exact": True, "moving": False,
             "cards": [[t.cards["ANGER"], 52, 0], [t.cards["HEADBUTT"], 77, 0], [t.cards["HAVOC"], 48, 0]],
             "relics": [[t.relics["ANCHOR"], 177], [t.relics["BAG_OF_PREPARATION"], 210]], "potions": [[t.potions["FIRE_POTION"], 52]]}
        v.update(kw)
        return v

    def truth(self, **kw):
        act = {"act": 1, "points": 7, "status": "ok", "reading": "plain", "jumps": 0, "unique": True, "cols": [3] * 7, "nodes": self.NODES, "edges": self.EDGES,
               "start": [3, 0]}
        doc = {"id": self.train_id, "compass": False, "boots": None, "spoils_act1": False, "acts": [act], "shops": [self.visit()]}
        doc.update(kw)
        return doc

    def build(self, truth):
        stats, seen, price = Counter(), Counter(), Counter()
        specs = hp.run_decisions(self.record, self.t, stats, seen, price, truth=truth)
        return specs, stats, seen, price

    # ------------------------------------------------------------------------------------------ the identity check

    def test_map_identity_check(self):
        import human_prior_replay as hpr  # noqa: PLC0415

        nodes = {(3, 0): 8, (2, 1): 1, (4, 1): 1, (2, 2): 1, (4, 2): 3, (3, 3): 6}
        edges = [((3, 0), (2, 1)), ((3, 0), (4, 1)), ((2, 1), (2, 2)), ((4, 1), (4, 2)), ((2, 2), (3, 3)), ((4, 2), (3, 3))]
        graph = {"nodes": nodes, "edges": edges, "current": (3, 0)}
        left = hpr.match_route(graph, [1, 1, 6])  # the route is a path, and the only one
        self.assertEqual((left["ok"], left["unique"], left["cols"]), (True, True, [3, 2, 2, 3]))
        right = hpr.match_route(graph, [1, 3, 6])
        self.assertEqual((right["ok"], right["cols"]), (True, [3, 4, 4, 3]))
        bad = hpr.match_route(graph, [1, 5, 6])  # no room of that type: not a path of this map, and how far it got
        self.assertEqual((bad["ok"], bad["best"]), (False, 1))
        prefix = hpr.match_route(graph, [1])  # a route that ended early fits both rooms: the record cannot say which
        self.assertEqual((prefix["ok"], prefix["unique"], prefix["cols"]), (True, False, [3, -1]))
        # Winged Boots: a step to any room of the next row, after the floor they were picked up on, at the fewest jumps
        nodes2 = dict(nodes)
        nodes2[(4, 1)] = 2
        graph2 = {"nodes": nodes2, "edges": edges, "current": (3, 0)}
        self.assertFalse(hpr.match_route(graph2, [1, 3])["ok"])
        boots = hpr.match_route(graph2, [1, 3], boots_floor=1)
        self.assertEqual((boots["ok"], boots["jumps"], boots["cols"]), (True, 1, [3, 2, 4]))
        self.assertFalse(hpr.match_route(graph2, [1, 3], boots_floor=3)["ok"])  # picked up too late for the jump (the step is floor 3)
        # the other side of the check: the same route against another seed's map
        other = {"nodes": {(3, 0): 8, (1, 1): 3, (3, 3): 6}, "edges": [((3, 0), (1, 1)), ((1, 1), (3, 3))], "current": (3, 0)}
        self.assertFalse(hpr.match_route(other, [1, 1, 6])["ok"])

    def test_runs_whose_map_cannot_stand_are_left_out(self):
        ok = self.truth()
        self.assertIsNone(hp.run_exclusion(ok))
        self.assertIsNone(hp.run_exclusion(None))
        self.assertEqual(hp.run_exclusion(self.truth(compass=True)), "golden_compass")
        bad = self.truth()
        bad["acts"][0] = dict(bad["acts"][0], status="no_path")
        self.assertEqual(hp.run_exclusion(bad), "map_mismatch")
        self.assertEqual(hp.run_exclusion(self.truth(maps_crash="boom")), "map_crash")
        specs, stats, seen, _ = self.build(self.truth(compass=True))
        self.assertEqual(specs, [])
        self.assertEqual(stats["runs_left_out|golden_compass"], 1)
        self.assertEqual(stats["runs_true_map"], 0)
        self.assertEqual(seen["act1|card"], 3)  # the screens are still counted (counts.json's definitions)
        self.assertGreater(stats["drop|act1|card|run left out: golden_compass"], 0)

    # ----------------------------------------------------------------------------------------------- route rows

    def test_route_rows_option_set(self):
        specs, stats, _, _ = self.build(self.truth())
        routes = [s for s in specs if s["kind"] == "route"]
        self.assertEqual(len(routes), 2)  # six steps, two with more than one room to go to
        first, second = routes
        self.assertEqual([o["node"] for o in first["options"]], [[2, 1, 1], [3, 1, 1], [4, 1, 3]])  # the rooms the start leads to, by column
        self.assertEqual(first["chosen"], 1)  # the human went to (3, 1)
        self.assertEqual([o["node"] for o in second["options"]], [[3, 3, 1], [4, 3, 1]])
        self.assertEqual(second["chosen"], 0)
        self.assertEqual((first["floor"], first["position"]), (1, (3, 0)))  # the state is the one after floor 1, standing on the start
        self.assertEqual((second["floor"], second["position"]), (3, (3, 2)))
        self.assertEqual(stats["route|act1|steps"], 6)
        self.assertEqual((stats["route|act1|choices"], stats["route|act1|single_room"], stats["route|act1|type_choices"]), (2, 4, 1))
        # every other decision carries the map and the node the human stood on
        others = [s for s in specs if s["kind"] != "route"]
        self.assertTrue(others and all("map_nodes" in s and s["position"][0] == 3 for s in others))
        self.assertEqual({s["floor"]: s["position"] for s in others if s["kind"] in ("card", "shop")}, {2: (3, 1), 3: (3, 2), 5: (3, 4)})  # floor 4's screen of five is dropped
        # the afterstates: the step is pending (the room's type), one floor on, the position moved; the distance to the boss is the same for all
        F = hp.stack().F
        rows = np.stack(hp.spec_rows(first))
        b = F.B_OFFSET
        room = [int(np.argmax(r[b:b + 9])) for r in rows]
        self.assertEqual(room[0], room[1])  # two monsters
        self.assertNotEqual(room[0], room[2])  # and a rest
        self.assertEqual(len({round(float(r[15]), 6) for r in rows}), 1)
        self.assertAlmostEqual(float(rows[0][15]), (7 - 1) / 16, places=6)
        self.assertGreater(float(rows[0][5]), float(np.stack(hp.spec_rows({**first, "options": [{"k": "card.skip"}]}))[0][5]))  # floor / 17 moved on by one
        # the map block differs between the rooms of the step (a shop lies ahead of one, not of the other)
        self.assertFalse(np.array_equal(rows[0][F.C_OFFSET:F.C_OFFSET + F.C_SIZE], rows[1][F.C_OFFSET:F.C_OFFSET + F.C_SIZE]))

    def test_route_rows_are_dropped_not_guessed(self):
        # a step whose node the record cannot tell (several paths of the same room types) is no decision, nor is what stands on it
        amb = self.truth()
        amb["acts"][0] = dict(amb["acts"][0], cols=[3, 3, -1, 3, 3, 3, 3], unique=False)
        specs, stats, _, _ = self.build(amb)
        self.assertEqual([s["floor"] for s in specs if s["kind"] == "route"], [1])
        self.assertEqual(stats["drop|act1|route|route: position ambiguous (several paths of the same room types)"], 1)
        self.assertEqual(stats["drop|act1|card|map position ambiguous (several paths of the same room types)"], 1)
        self.assertNotIn(3, {s["floor"] for s in specs if s["kind"] == "card"})
        # after Winged Boots the next room may be any of the row: not modelled, left out
        boots = self.build(self.truth(boots=1))
        self.assertEqual([s for s in boots[0] if s["kind"] == "route"], [])
        self.assertEqual(boots[1]["drop|act1|route|route: after Winged Boots were picked up (free steps are not modelled)"], 2)  # both steps with a choice
        # a step that is not along an edge of the map
        edges = [e for e in self.EDGES if e[:4] != [3, 0, 3, 1]]
        off = self.truth()
        off["acts"][0] = dict(off["acts"][0], edges=edges)
        self.assertEqual(self.build(off)[1]["drop|act1|route|route step is not along an edge of the map"], 1)

    # ------------------------------------------------------------------------------------------------ shop prices

    def test_price_flags_follow_the_replay(self):
        def shop_rows(visit):
            truth = self.truth(shops=[visit] if visit else [])
            return [s for s in self.build(truth)[0] if s["kind"] == "shop"]

        t = self.t
        strict = shop_rows(self.visit())
        self.assertTrue(strict and all(not (s["flags"] & (hp.F_PRICE_UNVERIFIED | hp.F_SHELF_ONLY)) for s in strict))
        quoted = {(s["offer"], s["options"][1]["id"]): s["options"][1]["price"] for s in strict}
        self.assertEqual(quoted[(hp.OFFER_CARD, t.cards["HAVOC"])], 48)  # the copy's price, not the rules' band
        self.assertEqual(quoted[(hp.OFFER_RELIC, t.relics["ANCHOR"])], 177)
        self.assertEqual(quoted[(hp.OFFER_POTION, t.potions["FIRE_POTION"])], 52)
        self.assertTrue(all(s["price"] == s["options"][1]["price"] for s in strict))
        # the replay had left the record's path, but the shelf and the prices match anyway: usable apart, not in the verified set
        loose = shop_rows(self.visit(clean=False))
        self.assertTrue(all(s["flags"] & hp.F_PRICE_UNVERIFIED and s["flags"] & hp.F_SHELF_ONLY for s in loose))
        self.assertEqual({(s["offer"], s["options"][1]["id"]): s["options"][1]["price"] for s in loose}, quoted)
        # anything short of that keeps the flag and the rules' price
        for reason, v in (("shelf", self.visit(shelf_ok=False)), ("price", self.visit(exact=False)), ("moving", self.visit(moving=True)), ("no replay", None)):
            rows = shop_rows(v)
            self.assertTrue(rows and all(s["flags"] & hp.F_PRICE_UNVERIFIED and not s["flags"] & hp.F_SHELF_ONLY for s in rows), reason)
            self.assertEqual(rows[0]["options"][1]["price"], hp.shop_price(rows[0]["offer"], rows[0]["options"][1]["id"], rows[0]["relics"], t)[0], reason)
        # a visit without a purchase has nothing to add up: the shelf alone vouches for it
        self.assertTrue(all(not s["flags"] & hp.F_PRICE_UNVERIFIED for s in shop_rows(self.visit(purchase=False, exact=None))))
        # the grades, as the build counts them
        stats = self.build(self.truth(shops=[self.visit(shelf_ok=False)]))[1]
        self.assertEqual((stats["shop_visit|shelf_differs"], stats["shop_visit|strict"]), (1, 0))
        self.assertEqual(self.build(self.truth())[1]["shop_visit|strict"], 1)
        # a v1 fit leaves out the unverified rows (the shelf-only ones too, unless asked), a v0 fit all of them
        ds = types.SimpleNamespace(dec=np.zeros((4, len(hp.COLS)), np.int32))
        ds.dec[:, hp.C["kind"]] = hp.KIND["shop"]
        ds.dec[:, hp.C["flags"]] = [0, hp.F_PRICE_UNVERIFIED, hp.F_PRICE_UNVERIFIED | hp.F_SHELF_ONLY, hp.F_UNAVAILABLE]
        self.assertEqual(hp._bool_unavailable(ds, True, False).tolist(), [False, True, True, True])
        self.assertEqual(hp._bool_unavailable(ds, True, False, shelf_only_ok=True).tolist(), [False, True, False, True])
        self.assertEqual(hp._bool_unavailable(ds, True, False, verified_only=False).tolist(), [False, False, False, True])
        self.assertEqual(hp._bool_unavailable(ds, False, False).tolist(), [True] * 4)

    # ------------------------------------------------------------------------------------ the shard, the fit, the scorer

    def v1_shard(self, tmp, specs):
        for sp in specs:
            sp["run_ix"] = 0
        path = Path(tmp) / "shard-00000.npz"
        n_rows = hp.write_shard(path, specs, [self.train_id], fake_judge(), {}, None)
        self.assertEqual(n_rows, sum(len(s["options"]) for s in specs))
        return hp.Dataset.load(Path(tmp), [hs.exam_bucket(self.train_id)], None)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_v1_round_trip_with_true_maps(self):
        specs = self.build(self.truth())[0]
        F = hp.stack().F
        with tempfile.TemporaryDirectory() as tmp:
            ds = self.v1_shard(tmp, specs)
            self.assertTrue(ds.has_map)
            self.assertEqual(ds.dense.shape[1], hp.N_DENSE_V1)
            self.assertEqual(len(ds.kind_idx["route"]), 2)
            # the stored map block is the true one of the node the human stood on, and the distance to the boss comes with it
            card = int(ds.kind_idx["card"][0])
            row = int(ds.dec[card, hp.C["start"]])
            node_map = hp.act_map(self.truth()["acts"][0])["nodes"]
            pos = specs[[s["kind"] for s in specs].index("card")]["position"]
            np.testing.assert_allclose(ds.dense[row, hp.S1_C], F._map_ahead(node_map, pos))
            self.assertAlmostEqual(float(ds.dense[row, 15]), (7 - pos[1]) / 16, places=6)
            # routes carry the room's type and column per option
            r0 = int(ds.kind_idx["route"][0])
            s0, n0 = int(ds.dec[r0, hp.C["start"]]), int(ds.dec[r0, hp.C["n"]])
            self.assertEqual(ds.role[s0:s0 + n0].tolist(), [1, 1, 3])
            self.assertEqual(ds.ident[s0:s0 + n0].tolist(), [2, 3, 4])
            self.assertEqual(int(ds.dec[r0, hp.C["aux"]]), 3)  # the boss node's column
            # fit steps on every kind, the numpy prior equals the torch net, and the metrics know the route kind
            dev = torch.device("cpu")
            net = hp.make_net()
            gt = hp.Gpu(ds, None, dev, 1.0)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
            net.train()
            for kind in ("card", "ancient", "rest", "shop", "route"):
                loss = gt.loss(net, kind, torch.from_numpy(ds.kind_idx[kind]))
                self.assertTrue(torch.isfinite(loss).item(), kind)
                opt.zero_grad()
                loss.backward()
                opt.step()
            hp.export_net(net, Path(tmp) / "weights.npz", {"kinds": ["card", "ancient", "rest", "shop", "route"]})
            prior = hp.NumpyPrior(Path(tmp) / "weights.npz")
            np.testing.assert_allclose(gt.scores(net), hp.row_scores_np(ds, None, prior), rtol=1e-4, atol=1e-4)
            m = hp.metrics_from_scores(ds, hp.row_scores_np(ds, None, prior), ~hp._bool_unavailable(ds, True, False))
            self.assertEqual(m["route"]["n"], 2)
            self.assertEqual(m["route_type_choices"]["n"], 1)  # only the first step offers rooms of different types
            self.assertAlmostEqual(m["route_coin_top1"], (1 / 3 + 1 / 2) / 2)
            # the baselines: a coin, and the room nearest the boss's column (ties share the credit)
            old = hp.MIN_CELL
            hp.MIN_CELL = 1
            try:
                base = hp.route_baselines(ds, ~hp._bool_unavailable(ds, True, False), ds)
            finally:
                hp.MIN_CELL = old
            self.assertAlmostEqual(base["uniform"]["all"]["top1"], (1 / 3 + 1 / 2) / 2)
            # first step: columns 2, 3, 4 against the boss's 3: the human took (3, 1), the nearest; second: 3 and 4, the human took (3, 3), the nearest
            self.assertAlmostEqual(base["nearest_to_boss_column"]["all"]["top1"], 1.0)
            self.assertEqual(base["decisions"], 2)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_win_input(self):
        specs = self.build(self.truth())[0]
        for sp in specs:
            sp["won"] = sp["kind"] == "card"  # rows of 'winning' and 'losing' runs
        with tempfile.TemporaryDirectory() as tmp:
            ds = self.v1_shard(tmp, specs)
            dev = torch.device("cpu")
            net = hp.make_net(win=True)
            self.assertTrue(net.uses_win)
            torch.manual_seed(3)
            net.won_w.data = torch.randn_like(net.won_w) * 0.5  # a trained-looking input weight (it starts at zero)
            gt = hp.Gpu(ds, None, dev, 1.0)
            self.assertEqual(float(net.won_w.abs().sum() > 0), 1.0)
            # the input is the decision's run outcome unless a value is given; the loss sees it and trains the weight
            as_recorded, at_play, at_zero = gt.scores(net), gt.scores(net, win=1.0), gt.scores(net, win=0.0)
            self.assertFalse(np.allclose(at_play, at_zero))
            card_rows = np.concatenate([np.arange(ds.dec[i, hp.C["start"]], ds.dec[i, hp.C["start"]] + ds.dec[i, hp.C["n"]]) for i in ds.kind_idx["card"]])
            np.testing.assert_allclose(as_recorded[card_rows], at_play[card_rows], rtol=1e-5, atol=1e-5)  # won decisions: input 1
            other = np.setdiff1d(np.arange(len(as_recorded)), card_rows)
            np.testing.assert_allclose(as_recorded[other], at_zero[other], rtol=1e-5, atol=1e-5)
            net.train()
            loss = gt.loss(net, "card", torch.from_numpy(ds.kind_idx["card"]))
            loss.backward()
            self.assertGreater(float(net.won_w.grad.abs().sum()), 0.0)
            # the export keeps the v0 format: the same arrays and the same forward, with the input weight folded into the first bias at the play setting
            path = Path(tmp) / "weights.npz"
            hp.export_net(net, path, {"kinds": ["card"]})
            prior = hp.NumpyPrior(path)
            self.assertEqual([e["name"] for e in prior.extra_inputs], ["won"])
            self.assertEqual(prior.meta["extra_inputs"][0]["value_at_play"], 1.0)
            self.assertIn("extra.won.weight", prior.p)
            self.assertNotIn("won_w", prior.p)
            np.testing.assert_allclose(hp.row_scores_np(ds, None, prior), at_play, rtol=1e-4, atol=1e-4)  # no extra given: the play setting
            np.testing.assert_allclose(hp.row_scores_np(ds, None, prior, extra=np.zeros(len(ds.dense), np.float32)), at_zero, rtol=1e-4, atol=1e-4)
            rec = np.repeat(ds.dec[:, hp.C["won"]], ds.dec[:, hp.C["n"]]).astype(np.float32)
            np.testing.assert_allclose(hp.row_scores_np(ds, None, prior, extra=rec), as_recorded, rtol=1e-4, atol=1e-4)
            # a loader that knows nothing of extra inputs (the v0 formula over the stored arrays) scores at play
            p = prior.p

            def v0_forward(dense, counts):
                e = np.concatenate([counts[:, :1154] @ p["embed.deck.weight"].T, counts[:, 1154:1450] @ p["embed.relics.weight"].T,
                                    counts[:, 1450:] @ p["embed.potions.weight"].T], axis=1)
                h = np.maximum(np.concatenate([dense, e], axis=1) @ p["hidden1.weight"].T + p["hidden1.bias"], 0.0)
                h = np.maximum(h @ p["hidden2.weight"].T + p["hidden2.bias"], 0.0)
                return (h @ p["score.weight"].T + p["score.bias"])[:, 0]

            np.testing.assert_allclose(hp.row_scores_np(ds, None, v0_forward), at_play, rtol=1e-4, atol=1e-4)
            # a plain prior takes no input and the won column is not in its file
            plain = hp.make_net()
            self.assertFalse(plain.uses_win)
            hp.export_net(plain, Path(tmp) / "plain.npz", {})
            self.assertEqual(hp.NumpyPrior(Path(tmp) / "plain.npz").extra_inputs, [])

    def test_kind_and_flag_ids_do_not_collide(self):
        self.assertEqual(len(set(hp.KIND.values())), len(hp.KIND))
        self.assertEqual({k: hp.KIND[k] for k in ("card", "rest", "ancient", "shop")}, {"card": 0, "rest": 1, "ancient": 2, "shop": 3})  # v0's ids
        bits = [hp.F_SPLIT, hp.F_PRICE_UNVERIFIED, hp.F_UNAVAILABLE, hp.F_REPLAY_MISMATCH, hp.F_BOSS_GUESSED, hp.F_GATE_UNKNOWN, hp.F_SHELF_ONLY]
        self.assertEqual(len(set(bits)), len(bits))
        self.assertTrue(all(b & (b - 1) == 0 for b in bits))

    def test_truth_chunk_must_match_the_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "truth-00000.json.gz"
            with gzip.open(good, "wt", encoding="utf-8") as fh:
                json.dump({"run_ids": [self.train_id], "runs": [self.truth()]}, fh)
            self.assertEqual(list(hp.load_truth_chunk(good)), [self.train_id])
            bad = Path(tmp) / "truth-00001.json.gz"
            with gzip.open(bad, "wt", encoding="utf-8") as fh:
                json.dump({"run_ids": [self.train_id, "x"], "runs": [self.truth()]}, fh)
            with self.assertRaises(ValueError):
                hp.load_truth_chunk(bad)
            hp._WORK.update(t=self.t, judge=fake_judge(), donors=None, dindex={}, exam=frozenset())
            try:
                other = synthetic_run(_find_ids(False, 2, "w")[1])
                with self.assertRaises(AssertionError):  # a run the truth chunk has no record of
                    hp.build_chunk({"lines": [json.dumps(other)], "out": str(Path(tmp) / "shard.npz"), "truth": str(good)})
                ok = hp.build_chunk({"lines": [json.dumps(self.record)], "out": str(Path(tmp) / "shard-00000.npz"), "truth": str(good)})
            finally:
                hp._WORK.clear()
            self.assertEqual(ok["runs"], 1)
            self.assertEqual(ok["stats"]["runs_true_map"], 1)


if __name__ == "__main__":
    unittest.main()
