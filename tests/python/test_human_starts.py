"""Fight starts from human runs (rl/human_starts.py; D38, memo 2026-11-04 section 3.2).

The pinned facts: the exam fifth is never a start (a run listed in the ids file or put in the fifth by the hash rule is left out of
the read, and a written record of such a run stops the build); a human run is rebuilt floor by floor into the state the human
walked in with (deck with its removals, transformations, upgrades and enchantments, the relics gained before the fight, the potions
held, HP and max HP from the floor before) and what cannot be rebuilt is counted by reason; the copy deals the opening with that
state and the human's monsters; the records are in the gym's schema with the byte index `load_start` reads; the coverage report
shows no cell under 20 runs. A synthetic run record stands in for the corpus; the real corpus is not read here.
"""

from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gym_loop as gl  # noqa: E402
import human_starts as hs  # noqa: E402

REAL_EXAM_IDS = Path("D:/spireline/data/spire-codex/human-exam-fifth-ids.txt")


def _find_ids(want_exam: bool, n: int, prefix: str = "r") -> list[str]:
    out, i = [], 0
    while len(out) < n:
        rid = f"{prefix}{i:014x}"
        if (hs.exam_bucket(rid) == hs.EXAM_BUCKET) == want_exam:
            out.append(rid)
        i += 1
    return out


def _stats(hp, max_hp=80, gold=99, **extra):
    return {"current_hp": hp, "max_hp": max_hp, "current_gold": gold, "damage_taken": 0, "hp_healed": 0, "player_id": 1, **extra}


def _point(kind, stats, rooms=None):
    return {"map_point_type": kind, "player_stats": [stats], "rooms": rooms or [{"room_type": kind}]}


def _fight(kind, model, monsters, stats):
    return _point({"monster": "monster", "elite": "elite", "boss": "boss"}[kind], stats,
                  [{"model_id": model, "monster_ids": monsters, "room_type": kind}])


def synthetic_run(rid: str, *, act2: bool = False, drop_final: bool = False) -> dict:
    """An Ironclad A10 run: Neow (one Strike upgraded, a relic), a weak slimes fight, a shop removing a Strike, a rest upgrading
    Bash, an elite, the boss; optionally an Act 2 with an elite."""
    a1 = [
        _point("ancient", _stats(64, 80, 99, upgraded_cards=["CARD.STRIKE_IRONCLAD"],
                                 relic_choices=[{"choice": "RELIC.NEOWS_TALISMAN", "was_picked": True}]),
               [{"model_id": "EVENT.NEOW", "room_type": "event"}]),
        _fight("monster", "ENCOUNTER.SLIMES_WEAK", ["MONSTER.LEAF_SLIME_S", "MONSTER.LEAF_SLIME_M", "MONSTER.TWIG_SLIME_S"],
               _stats(55, 80, 112, cards_gained=[{"id": "CARD.POMMEL_STRIKE"}],
                      potion_choices=[{"choice": "POTION.FIRE_POTION", "was_picked": True}])),
        _point("shop", _stats(55, 80, 12, cards_removed=[{"id": "CARD.STRIKE_IRONCLAD"}])),
        _point("rest_site", _stats(70, 80, 12, rest_site_choices=["SMITH"], upgraded_cards=["CARD.BASH"])),
        _fight("elite", "ENCOUNTER.SKULKING_COLONY_ELITE", ["MONSTER.SKULKING_COLONY"],
               _stats(30, 80, 60, cards_gained=[{"id": "CARD.BATTLE_TRANCE"}], potion_used=["POTION.FIRE_POTION"],
                      relic_choices=[{"choice": "RELIC.ODDLY_SMOOTH_STONE", "was_picked": True}])),
        _fight("boss", "ENCOUNTER.VANTOM_BOSS", ["MONSTER.VANTOM"], _stats(40, 80, 150, cards_gained=[{"id": "CARD.OFFERING"}])),
    ]
    acts = [a1]
    deck = [("ASCENDERS_BANE", 0), ("STRIKE_IRONCLAD", 1)] + [("STRIKE_IRONCLAD", 0)] * 3 + [("DEFEND_IRONCLAD", 0)] * 4 + [("BASH", 1)] \
        + [("POMMEL_STRIKE", 0), ("BATTLE_TRANCE", 0), ("OFFERING", 0)]
    relics = [("BURNING_BLOOD", 1), ("NEOWS_TALISMAN", 1), ("ODDLY_SMOOTH_STONE", 5)]
    if act2:
        a2 = [
            _point("ancient", _stats(80, 80, 150), [{"model_id": "EVENT.OROBAS", "room_type": "event"}]),
            _fight("elite", "ENCOUNTER.ENTOMANCER_ELITE", ["MONSTER.ENTOMANCER", "MONSTER.PAELS_LEGION"], _stats(50, 80, 200)),
        ]
        acts.append(a2)
    if drop_final:
        deck = deck[:-1]  # the final record disagrees with the replay
    return {"ascension": 10, "game_mode": "standard", "was_abandoned": False, "modifiers": [], "run_hash": rid, "seed": "S" + rid[:6],
            "acts": ["ACT.OVERGROWTH", "ACT.HIVE", "ACT.GLORY"], "win": False, "map_point_history": acts,
            "players": [{"character": "CHARACTER.IRONCLAD", "id": 1,
                         "deck": [{"id": f"CARD.{n}", "floor_added_to_deck": 1, **({"current_upgrade_level": lv} if lv else {})}
                                  for n, lv in deck],
                         "relics": [{"id": f"RELIC.{n}", "floor_added_to_deck": f} for n, f in relics], "potions": []}]}


class ExamTests(unittest.TestCase):
    def test_a_run_is_exam_when_listed_or_when_the_hash_puts_it_in_the_fifth(self):
        listed, hashed, other = _find_ids(False, 1, "l")[0], _find_ids(True, 1, "h")[0], _find_ids(False, 1, "o")[0]
        ids = frozenset({listed})
        self.assertTrue(hs.is_exam(listed, ids))  # in the ids file (a run the hash rule would not name)
        self.assertTrue(hs.is_exam(hashed, ids))  # not in the file, in the fifth by the rule: out all the same
        self.assertFalse(hs.is_exam(other, ids))
        with self.assertRaises(AssertionError):
            hs.assert_no_exam([other, listed], ids)
        with self.assertRaises(AssertionError):
            hs.assert_no_exam([hashed], ids)
        self.assertEqual(hs.assert_no_exam([other, other], ids), 2)

    def test_the_hash_rule_is_a_fifth_of_runs_and_does_not_move(self):
        n = sum(hs.exam_bucket(f"x{i}") == hs.EXAM_BUCKET for i in range(5000))
        self.assertTrue(900 < n < 1100, n)
        self.assertEqual(hs.exam_bucket("0016540b2159554f"), hs.exam_bucket("0016540b2159554f"))

    def test_the_real_ids_file_sits_in_the_fifth_the_rule_names(self):
        if not REAL_EXAM_IDS.is_file():
            self.skipTest("the exam ids are not on this machine")
        ids = hs.read_exam_ids(REAL_EXAM_IDS)
        self.assertGreater(len(ids), 2000)
        self.assertTrue(all(hs.exam_bucket(i) == hs.EXAM_BUCKET for i in ids))  # the file is a part of what the rule names

    def test_an_empty_ids_file_stops_the_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ids.txt"
            p.write_text("", encoding="utf-8")
            with self.assertRaises(SystemExit):
                hs.read_exam_ids(p)

    def test_the_read_leaves_out_the_fifth_the_wrong_character_and_ascension_and_modded_acts(self):
        listed, hashed, ok1, ok2 = _find_ids(False, 1, "l")[0], _find_ids(True, 1, "h")[0], *_find_ids(False, 2, "k")
        runs = [synthetic_run(ok1), synthetic_run(listed), synthetic_run(hashed), synthetic_run(ok2)]
        runs[3]["acts"] = ["ACT.ACTSFROMTHEPAST-EXORDIUM_ACT", "ACT.HIVE", "ACT.GLORY"]
        silent = synthetic_run(_find_ids(False, 1, "s")[0])
        silent["players"][0]["character"] = "CHARACTER.SILENT"
        a9 = synthetic_run(_find_ids(False, 1, "n")[0])
        a9["ascension"] = 9
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "corpus.jsonl.gz"
            with gzip.open(p, "wt", encoding="utf-8") as fh:
                for r in runs + [silent, a9]:
                    fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            stats = Counter()
            got = [hs.run_id(r) for r in hs.read_runs(p, frozenset({listed}), stats)]
        self.assertEqual(got, [ok1])
        self.assertEqual((stats["lines"], stats["ironclad_a10"], stats["exam_fifth_left_out"], stats["modded_acts"]), (6, 4, 2, 1))


class RebuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = hs.Tables.load()

    def _specs(self, run, fail=None, seen=None):
        return hs.rebuild_run(run, self.tables, fail if fail is not None else Counter(), seen)

    def test_each_fight_is_rebuilt_with_the_deck_relics_potions_and_hp_the_human_walked_in_with(self):
        rid = _find_ids(False, 1, "k")[0]
        specs = {s["floor"]: s for s in self._specs(synthetic_run(rid))}
        self.assertEqual(sorted(specs), [2, 5, 6])
        t = self.tables
        cid = t.cards
        first = specs[2]  # the weak slimes fight, walking in from Neow's floor
        self.assertEqual((first["hp"], first["max_hp"], first["gold"], first["kind"], first["act"], first["cat"]),
                         (64, 80, 99, "hallway", 1, "act1-hallway"))
        self.assertEqual(sorted(first["deck"]), sorted([cid["ASCENDERS_BANE"], -cid["STRIKE_IRONCLAD"]] + [cid["STRIKE_IRONCLAD"]] * 4
                                                       + [cid["DEFEND_IRONCLAD"]] * 4 + [cid["BASH"]]))  # Neow's upgrade, nothing more
        self.assertEqual(sorted(first["relics"]), sorted([t.relics["BURNING_BLOOD"], t.relics["NEOWS_TALISMAN"]]))
        self.assertEqual(first["potions"], [])
        self.assertTrue(first["weak"])
        self.assertEqual(first["monsters"], ["leafslimem", "leafslimes", "twigslimes"])
        self.assertEqual(first["id"], f"human:{rid}|f02")
        elite = specs[5]
        self.assertEqual((elite["hp"], elite["max_hp"], elite["gold"]), (70, 80, 12))  # the rest floor's HP, after it
        want = [cid["ASCENDERS_BANE"], -cid["STRIKE_IRONCLAD"]] + [cid["STRIKE_IRONCLAD"]] * 3 + [cid["DEFEND_IRONCLAD"]] * 4 \
            + [-cid["BASH"], cid["POMMEL_STRIKE"]]  # the shop took a Strike, the rest upgraded Bash, the first fight gave Pommel Strike
        self.assertEqual(sorted(elite["deck"]), sorted(want))  # not the Battle Trance the elite itself gave
        self.assertEqual(sorted(elite["relics"]), sorted([t.relics["BURNING_BLOOD"], t.relics["NEOWS_TALISMAN"]]))  # not the stone from the elite
        self.assertEqual(elite["potions"], [t.potions["FIRE_POTION"]])
        self.assertEqual((elite["kind"], elite["node_type"], elite["enemy"]), ("elite", 2, "SKULKING_COLONY_ELITE"))
        boss = specs[6]
        self.assertEqual((boss["hp"], boss["kind"], boss["node_type"], boss["cat"]), (30, "boss", 6, "act1-boss"))
        self.assertIn(cid["BATTLE_TRANCE"], boss["deck"])
        self.assertIn(t.relics["ODDLY_SMOOTH_STONE"], boss["relics"])
        self.assertEqual(boss["potions"], [])  # the fire potion was used in the elite fight
        self.assertEqual(len(boss["enchantments"]), len(boss["deck"]))

    def test_act_two_fights_are_rebuilt_and_act_three_is_counted_not_built(self):
        rid = _find_ids(False, 1, "k")[0]
        run = synthetic_run(rid, act2=True)
        run["map_point_history"].append([_fight("boss", "ENCOUNTER.THE_INSATIABLE_BOSS", ["MONSTER.THE_INSATIABLE"], _stats(10))])
        run["map_point_history"][2][0]["player_stats"][0]["cards_gained"] = []
        fail, seen = Counter(), Counter()
        specs = self._specs(run, fail, seen)
        self.assertEqual(Counter(s["cat"] for s in specs), Counter({"act1-hallway": 1, "act1-elite": 1, "act1-boss": 1, "act2-elite": 1}))
        e = next(s for s in specs if s["cat"] == "act2-elite")
        self.assertEqual((e["floor"], e["hp"], e["act"], e["enemy"]), (8, 80, 2, "ENTOMANCER_ELITE"))  # floor 7 is the Act 2 ancient
        self.assertEqual(e["monsters"], ["entomancer", "paelslegion"])
        self.assertEqual(fail["act 3 or later: not built"], 1)
        self.assertEqual(seen["act3-boss"], 1)

    def test_what_cannot_be_rebuilt_is_counted_by_reason(self):
        rid = _find_ids(False, 1, "k")[0]
        fail = Counter()
        self.assertEqual(self._specs(synthetic_run(rid, drop_final=True), fail), [])  # the replay disagrees with the final deck
        self.assertEqual(fail["deck replay disagrees with the run's final deck: run dropped"], 3)
        run = synthetic_run(rid)
        run["map_point_history"][0][1]["rooms"] = [{"model_id": "EVENT.X", "room_type": "event"},
                                                    {"model_id": "ENCOUNTER.SLIMES_WEAK", "monster_ids": ["MONSTER.LEAF_SLIME_S"],
                                                     "room_type": "monster"}]
        fail = Counter()
        specs = self._specs(run, fail)
        self.assertEqual([s["floor"] for s in specs], [5, 6])
        self.assertEqual(fail["an event page before the fight: HP going in not known"], 1)
        run = synthetic_run(rid)
        run["map_point_history"][0][4]["rooms"][0]["model_id"] = "ENCOUNTER.A_MODDED_FIGHT_ELITE"
        fail = Counter()
        self.assertEqual([s["floor"] for s in self._specs(run, fail)], [2, 6])
        self.assertEqual(fail["encounter unknown to the copy"], 1)
        run = synthetic_run(rid)
        run["map_point_history"][0][1]["player_stats"][0]["cards_gained"] = [{"id": "CARD.POMMEL_STRIKE"}, {"id": "CARD.A_MODDED_CARD"}]
        run["players"][0]["deck"].append({"id": "CARD.A_MODDED_CARD", "floor_added_to_deck": 2})
        fail = Counter()
        self.assertEqual([s["floor"] for s in self._specs(run, fail)], [2])  # the card is in every later deck
        self.assertEqual(fail["card unknown to the copy"], 2)

    def test_removals_transformations_enchantments_downgrades_and_finished_quests_change_the_deck_in_order(self):
        deck = [("STRIKE_IRONCLAD", 0, None, 0)] * 2 + [("DEFEND_IRONCLAD", 1, None, 0), ("SPOILS_MAP", 0, None, 0)]
        hs.apply_point(deck, {"cards_transformed": [{"original_card": {"id": "CARD.STRIKE_IRONCLAD"},
                                                     "final_card": {"id": "CARD.HEMOKINESIS", "current_upgrade_level": 1}}],
                              "cards_gained": [{"id": "CARD.ANGER"}, {"id": "CARD.ANGER"}], "bought_colorless": ["CARD.SPLASH"],
                              "completed_quests": ["CARD.SPOILS_MAP"]})
        self.assertEqual(Counter(deck), Counter([("STRIKE_IRONCLAD", 0, None, 0), ("HEMOKINESIS", 1, None, 0), ("ANGER", 0, None, 0),
                                                 ("ANGER", 0, None, 0), ("SPLASH", 0, None, 0), ("DEFEND_IRONCLAD", 1, None, 0)]))
        hs.apply_point(deck, {"cards_enchanted": [{"card": {"id": "CARD.ANGER", "enchantment": {"id": "ENCHANTMENT.SHARP", "amount": 2}}}],
                              "cards_removed": [{"id": "CARD.STRIKE_IRONCLAD"}], "downgraded_cards": ["CARD.DEFEND_IRONCLAD"],
                              "upgraded_cards": ["CARD.ANGER"]})
        self.assertEqual(Counter(deck), Counter([("HEMOKINESIS", 1, None, 0), ("ANGER", 1, "SHARP", 2), ("ANGER", 0, None, 0),
                                                 ("SPLASH", 0, None, 0), ("DEFEND_IRONCLAD", 0, None, 0)]))
        # a bought colorless card the gain list also names counts once
        deck = []
        hs.apply_point(deck, {"cards_gained": [{"id": "CARD.SPLASH"}], "bought_colorless": ["CARD.SPLASH"]})
        self.assertEqual(deck, [("SPLASH", 0, None, 0)])

    def test_an_enchanted_card_goes_to_the_copy_with_the_enchantment_and_its_amount(self):
        rid = _find_ids(False, 1, "k")[0]
        run = synthetic_run(rid)
        run["map_point_history"][0][3]["player_stats"][0]["cards_enchanted"] = [
            {"card": {"id": "CARD.DEFEND_IRONCLAD", "enchantment": {"id": "ENCHANTMENT.GOOPY", "amount": 1}}}]
        run["players"][0]["deck"][5] = {"id": "CARD.DEFEND_IRONCLAD", "floor_added_to_deck": 1,
                                        "enchantment": {"id": "ENCHANTMENT.GOOPY", "amount": 4}}
        specs = {s["floor"]: s for s in self._specs(run)}
        self.assertEqual([e for e in specs[5]["enchantments"] if e[0]], [(self.tables.enchants["GOOPY"], 1)])
        self.assertEqual([e for e in specs[2]["enchantments"] if e[0]], [])  # the card was plain until the shop
        self.assertTrue(specs[5]["approx"]["enchanted"])
        self.assertFalse(specs[2]["approx"]["enchanted"])

    def test_a_relic_lost_later_is_held_before_it_and_the_starter_until_black_blood(self):
        rid = _find_ids(False, 1, "k")[0]
        run = synthetic_run(rid)
        run["map_point_history"][0][4]["player_stats"][0]["relics_removed"] = ["RELIC.NEOWS_TALISMAN"]
        run["players"][0]["relics"] = [{"id": "RELIC.BURNING_BLOOD", "floor_added_to_deck": 1}]
        run["map_point_history"][0][0]["player_stats"][0]["relic_choices"] = [{"choice": "RELIC.NEOWS_TALISMAN", "was_picked": True}]
        got = hs.relics_at(run, 4, {"NEOWS_TALISMAN": 1}, {"NEOWS_TALISMAN": 5})
        self.assertEqual(sorted(got), ["BURNING_BLOOD", "NEOWS_TALISMAN"])
        self.assertEqual(hs.relics_at(run, 6, {"NEOWS_TALISMAN": 1}, {"NEOWS_TALISMAN": 5}), ["BURNING_BLOOD"])
        run["players"][0]["relics"] = [{"id": "RELIC.BLACK_BLOOD", "floor_added_to_deck": 17}]
        self.assertEqual(hs.relics_at(run, 20, {}, {}), ["BLACK_BLOOD"])
        self.assertEqual(hs.relics_at(run, 5, {}, {}), ["BURNING_BLOOD"])

    def test_the_chosen_set_keeps_every_elite_and_boss_and_caps_the_hallway_fights_half_to_each_act(self):
        specs = []
        for i in range(40):
            specs += [{"id": f"h{i}a1", "kind": "hallway", "act": 1}, {"id": f"h{i}a2", "kind": "hallway", "act": 2},
                      {"id": f"e{i}", "kind": "elite", "act": 1}, {"id": f"b{i}", "kind": "boss", "act": 2}]
        got = hs.pick_specs(specs, 10)
        self.assertEqual(Counter(s["kind"] for s in got), Counter({"hallway": 10, "elite": 40, "boss": 40}))
        self.assertEqual(Counter(s["act"] for s in got if s["kind"] == "hallway"), Counter({1: 5, 2: 5}))
        self.assertEqual([s["id"] for s in got], sorted(s["id"] for s in got))
        self.assertEqual(got, hs.pick_specs(list(reversed(specs)), 10))  # the same corpus gives the same set
        few = [s for s in specs if s["kind"] != "hallway" or s["act"] == 2]
        self.assertEqual(Counter(s["act"] for s in hs.pick_specs(few, 30) if s["kind"] == "hallway"), Counter({2: 30}))

    def test_the_roster_check_accepts_what_the_fight_summons_later(self):
        self.assertTrue(hs.sub_multiset(["ovicopter"], ["ovicopter", "toughegg"]))
        self.assertFalse(hs.sub_multiset(["leafslimem", "leafslimes"], ["leafslimes", "twigslimes"]))
        self.assertFalse(hs.sub_multiset(["seapunk", "seapunk"], ["seapunk"]))
        self.assertEqual(hs._norm("DecimillipedeSegmentFront"), hs._norm("DECIMILLIPEDE_SEGMENT_BACK"))  # noqa: SLF001


class ReportTests(unittest.TestCase):
    def test_no_cell_under_twenty_runs_is_shown_and_no_run_is_named(self):
        recs = []
        for i in range(25):
            recs.append({"act": 2, "kind": "boss", "enemy": "KAISER_CRAB_BOSS", "run": f"run{i:02d}", "approx": {"enchanted": False, "potions_unknown": False}})
        for i in range(7):
            recs.append({"act": 2, "kind": "boss", "enemy": "THE_RARE_BOSS", "run": f"run{i:02d}", "approx": {"enchanted": True, "potions_unknown": False}})
        rep = hs.coverage(recs, Counter({"lines": 100, "ironclad_a10": 40, "runs_usable": 30}), Counter({"roster differs": 3}),
                          Counter({"act2-boss": 40}), 35, 33)
        text = hs.coverage_text(rep)
        self.assertIn("act2 boss KAISER_CRAB_BOSS: 25", text)
        self.assertNotIn("THE_RARE_BOSS", text)
        self.assertIn("1 further cells are under 20 runs", text)
        self.assertIn("roster differs: 3", text)
        self.assertNotIn("run00", text)
        small = hs.coverage(recs[:5], Counter(), Counter(), Counter(), 5, 5)
        self.assertIn("< 20", hs.coverage_text(small))


# ---------------------------------------------------------------------------------------------------------- the copy


def _deal_in_child(queue, specs) -> None:
    try:
        for p in (ROOT / "rl", ROOT / "src", ROOT):
            sys.path.insert(0, str(p))
        import human_starts as h  # noqa: PLC0415

        h._init_worker()  # noqa: SLF001
        out = []
        for s in specs:
            doc, why = h.build_doc(s)
            out.append((s["id"], doc, why))
        queue.put(("ok", out))
    except Exception as error:  # noqa: BLE001 -- the tests skip when the native library is not there
        queue.put(("error", f"{type(error).__name__}: {error}"))


class OpeningTests(unittest.TestCase):
    """The copy deals the openings: real games' state, no game played."""

    @classmethod
    def setUpClass(cls):
        cls.tables = hs.Tables.load()
        rid = _find_ids(False, 1, "k")[0]
        run = synthetic_run(rid, act2=True)
        run["map_point_history"][0][3]["player_stats"][0]["cards_enchanted"] = [
            {"card": {"id": "CARD.DEFEND_IRONCLAD", "enchantment": {"id": "ENCHANTMENT.SHARP", "amount": 2}}}]
        run["players"][0]["deck"][5] = {"id": "CARD.DEFEND_IRONCLAD", "floor_added_to_deck": 1,
                                        "enchantment": {"id": "ENCHANTMENT.SHARP", "amount": 2}}
        cls.rid = rid
        cls.specs = {s["floor"]: s for s in hs.rebuild_run(run, cls.tables, Counter())}
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        p = ctx.Process(target=_deal_in_child, args=(queue, [cls.specs[f] for f in sorted(cls.specs)]), daemon=True)
        p.start()
        kind, got = queue.get(timeout=180)
        p.join(30)
        if kind != "ok":
            raise unittest.SkipTest(f"native library unavailable: {got}")
        cls.docs = {s: (d, why) for s, d, why in got}

    def test_the_doc_carries_the_state_the_human_walked_in_with(self):
        spec = self.specs[5]
        doc, why = self.docs[spec["id"]]
        self.assertEqual(why, "")
        self.assertEqual((doc["player"]["hp"], doc["player"]["max_hp"], doc["turn"], doc["ascension"]), (70, 80, 0, 10))
        self.assertEqual(doc["encounter_id"], spec["encounter_id"])
        self.assertEqual(sorted(r["id"] for r in doc["relics"]), sorted(spec["relics"]))
        self.assertEqual([x for x in doc["potions"]["slots"] if x], spec["potions"])
        cards = Counter((c["def_id"], c["upgraded"], c["enchantment"], c["enchant_amount"]) for c in hs.doc_cards(doc))
        want = Counter((abs(i), i < 0, e[0], e[1]) for i, e in zip(spec["deck"], spec["enchantments"]))
        self.assertEqual(cards, want)
        self.assertEqual(sum(1 for c in hs.doc_cards(doc) if c["enchantment"]), 1)  # the Sharp Defend
        self.assertEqual([e["max_hp"] > 0 for e in doc["enemies"]], [True])

    def test_the_weak_slimes_are_the_weak_roster_and_the_hand_is_dealt(self):
        spec = self.specs[2]
        doc, why = self.docs[spec["id"]]
        self.assertEqual(why, "")
        self.assertEqual(doc["player"]["hp"], 64)
        self.assertEqual(len(doc["piles"]["hand"]), 5)
        self.assertEqual(doc["potions"]["max_slots"], 2)
        self.assertEqual(len(doc["enemies"]), 3)

    def test_the_act_two_elite_gets_its_roster_checked_against_the_humans(self):
        spec = self.specs[8]
        doc, why = self.docs[spec["id"]]
        self.assertEqual(why, "")
        self.assertEqual(doc["player"]["hp"], 80)

    def test_a_roster_the_copy_cannot_deal_is_refused_by_reason(self):
        child_specs = [{**self.specs[5], "id": "human:never|f05", "monsters": ["somethingelse"]}]
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        p = ctx.Process(target=_deal_in_child, args=(queue, child_specs), daemon=True)
        p.start()
        kind, got = queue.get(timeout=180)
        p.join(30)
        self.assertEqual(kind, "ok")
        self.assertEqual((got[0][1], got[0][2]), (None, "roster differs"))

    def test_a_record_is_in_the_gyms_schema_and_the_loader_reads_it_back_through_the_index(self):
        spec = self.specs[5]
        doc, _ = self.docs[spec["id"]]
        rec = hs.start_record(spec, doc)
        for k in ("id", "cat", "kind", "act", "floor", "seed", "ix", "night", "k", "node_type", "hp", "max_hp", "deck_size", "potions", "relics",
                  "encounter_id", "turn", "doc", "split", "kind_same"):
            self.assertIn(k, rec)
        self.assertEqual((rec["source"], rec["run"], rec["seed"], rec["cat"], rec["act"], rec["floor"], rec["enemy"]),
                         ("human", self.rid, self.rid, "act1-elite", 1, 5, "SKULKING_COLONY_ELITE"))
        self.assertEqual((rec["hp"], rec["max_hp"], rec["deck_size"], rec["potions"], rec["relics"], rec["node_type"]), (70, 80, 11, 1, 2, 2))
        self.assertEqual(rec["split"], "val" if gl.is_val_seed(self.rid) else "train")
        with tempfile.TemporaryDirectory() as tmp:
            w = hs.SetWriter(Path(tmp))
            lite = w.add(json.dumps(rec))
            self.assertNotIn("doc", lite)
            with self.assertRaises(ValueError):
                w.add(json.dumps(rec))
            self.assertEqual(w.close(), 1)
            gl._START_CACHE.clear()  # noqa: SLF001
            got = gl.load_start(str(Path(tmp) / "human-starts.jsonl"), rec["id"])
            self.assertEqual(got["doc"]["player"]["hp"], 70)
            self.assertEqual(json.loads((Path(tmp) / "human-starts.lite.json").read_text())[0]["id"], rec["id"])
            self.assertTrue((Path(tmp) / "human-starts.index.json").is_file())

    def test_the_build_writes_the_set_the_report_and_keeps_the_exam_out(self):
        ok = _find_ids(False, 3, "k")
        exam_listed, exam_hash = _find_ids(False, 1, "l")[0], _find_ids(True, 1, "h")[0]
        runs = [synthetic_run(r, act2=True) for r in ok] + [synthetic_run(exam_listed), synthetic_run(exam_hash)]
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            with gzip.open(t / "corpus.jsonl.gz", "wt", encoding="utf-8") as fh:
                for r in runs:
                    fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            (t / "ids.txt").write_text(exam_listed + "\n", encoding="utf-8")
            rep = hs.build(t / "corpus.jsonl.gz", t / "out", 1, 5000, exam_path=t / "ids.txt", doc_path=t / "report.md")
            lite = json.loads((t / "out" / "human-starts.lite.json").read_text())
            index = json.loads((t / "out" / "human-starts.index.json").read_text())
            self.assertEqual(len(lite), 12)  # four fights in each of three runs
            self.assertEqual(sorted(index), sorted(r["id"] for r in lite))
            self.assertEqual({r["run"] for r in lite}, set(ok))
            self.assertEqual(rep["exam_fifth_left_out"], 2)
            self.assertEqual((rep["runs_usable"], rep["records"]), (3, 12))
            text = (t / "report.md").read_text(encoding="utf-8")
            self.assertIn("left out as the exam fifth (ids file or hash rule): 2", text)
            self.assertNotIn(exam_listed, text)
            self.assertNotIn(exam_hash, text)
            self.assertIn("further cells are under 20 runs", text)  # three runs: every cell is suppressed
            self.assertEqual(rep["cells"], {})
            hs.assert_no_exam((r["run"] for r in lite), frozenset({exam_listed}))


if __name__ == "__main__":
    unittest.main()
