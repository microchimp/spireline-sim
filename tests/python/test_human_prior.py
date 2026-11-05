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


if __name__ == "__main__":
    unittest.main()
