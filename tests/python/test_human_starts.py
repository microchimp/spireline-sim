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


def synthetic_run(rid: str, *, act2: bool = False, drop_final: bool = False, boss_at_17: bool = True) -> dict:
    """An Ironclad A10 run: Neow (one Strike upgraded, a relic), a weak slimes fight, a shop removing a Strike, a rest upgrading
    Bash, an elite, eleven quiet rests, the Act 1 boss on floor 17; optionally an Act 2 with an elite on floor 19."""
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
        *[_point("rest_site", _stats(30, 80, 60, rest_site_choices=["NOTHING"])) for _ in range(11 if boss_at_17 else 0)],
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
        self.assertEqual(sorted(specs), [2, 5, 17])
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
        boss = specs[17]
        self.assertEqual((boss["hp"], boss["kind"], boss["node_type"], boss["cat"]), (30, "boss", 6, "act1-boss"))
        self.assertIn(cid["BATTLE_TRANCE"], boss["deck"])
        self.assertIn(t.relics["ODDLY_SMOOTH_STONE"], boss["relics"])
        self.assertEqual(boss["potions"], [])  # the fire potion was used in the elite fight
        self.assertEqual(len(boss["enchantments"]), len(boss["deck"]))

    def test_act_two_fights_are_rebuilt_and_act_three_is_counted_not_built(self):
        rid = _find_ids(False, 1, "k")[0]
        run = synthetic_run(rid, act2=True)
        run["map_point_history"].append([_fight("boss", "ENCOUNTER.THE_INSATIABLE_BOSS", ["MONSTER.THE_INSATIABLE"], _stats(10))])
        fail, seen = Counter(), Counter()
        specs = self._specs(run, fail, seen)
        self.assertEqual(Counter(s["cat"] for s in specs), Counter({"act1-hallway": 1, "act1-elite": 1, "act1-boss": 1, "act2-elite": 1}))
        e = next(s for s in specs if s["cat"] == "act2-elite")
        self.assertEqual((e["floor"], e["hp"], e["act"], e["enemy"]), (19, 80, 2, "ENTOMANCER_ELITE"))  # floor 18 is the Act 2 ancient
        self.assertEqual(e["monsters"], ["entomancer", "paelslegion"])
        self.assertEqual(fail["act 3 or later: not built"], 1)
        self.assertEqual(seen["act3-boss"], 1)

    def test_what_cannot_be_rebuilt_is_counted_by_reason(self):
        rid = _find_ids(False, 1, "k")[0]
        fail = Counter()
        self.assertEqual(self._specs(synthetic_run(rid, drop_final=True), fail), [])  # the replay disagrees with the final deck
        self.assertEqual(fail["deck replay disagrees with the run's final deck: run dropped"], 3)
        fail = Counter()  # a boss on floor 6: the route is not the copy's (17 and 33)
        self.assertEqual([s["floor"] for s in self._specs(synthetic_run(rid, boss_at_17=False), fail)], [2, 5])
        self.assertEqual(fail["boss at an unusual floor: the route is not the copy's"], 1)
        run = synthetic_run(rid)
        run["map_point_history"][0][15]["player_stats"][0]["current_hp"] = 0  # the floor before the boss records no HP at all
        run["map_point_history"][0][15]["player_stats"][0]["max_hp"] = 0
        fail = Counter()
        self.assertEqual([s["floor"] for s in self._specs(run, fail)], [2, 5])
        self.assertEqual(fail["HP before the fight not credible"], 1)
        run = synthetic_run(rid)
        run["map_point_history"][0][1]["rooms"] = [{"model_id": "EVENT.X", "room_type": "event"},
                                                    {"model_id": "ENCOUNTER.SLIMES_WEAK", "monster_ids": ["MONSTER.LEAF_SLIME_S"],
                                                     "room_type": "monster"}]
        fail = Counter()
        specs = self._specs(run, fail)
        self.assertEqual([s["floor"] for s in specs], [5, 17])
        self.assertEqual(fail["an event page before the fight: HP going in not known"], 1)
        run = synthetic_run(rid)
        run["map_point_history"][0][4]["rooms"][0]["model_id"] = "ENCOUNTER.A_MODDED_FIGHT_ELITE"
        fail = Counter()
        self.assertEqual([s["floor"] for s in self._specs(run, fail)], [2, 17])
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


class OutcomeTests(unittest.TestCase):
    """The human's own result of a fight floor (D55 addendum 10): the gym's end HP (before the run engine's post-fight heals), HP lost,
    survival, potions, turns and the value in the search's currency."""

    @staticmethod
    def stats(cur, dmg, heal, max_hp=80, **extra):
        return {"current_hp": cur, "max_hp": max_hp, "damage_taken": dmg, "hp_healed": heal, **extra}

    def out(self, hp_in, ps, survived=True, held=("BURNING_BLOOD",), potions_in=0, room=None):
        return hs.fight_outcome(hp_in=hp_in, max_hp_in=80, survived=survived, ps=ps, room=room if room is not None else {"turns_taken": 5},
                                held=held, potions_in=potions_in)

    def test_the_end_hp_is_the_hp_after_the_floor_less_the_post_fight_heal(self):
        o = self.out(50, self.stats(36, 20, 6))  # 50 - 20 + 6 = 36: Burning Blood's six come off
        self.assertEqual((o["human_end_hp"], o["human_hp_lost"], o["human_damage_taken"], o["human_post_heal"]), (30, 20, 20, 6))
        self.assertEqual((o["human_survived"], o["human_turns"], o["human_potions_used"]), (True, 5, 0))
        o = self.out(50, self.stats(44, 20, 14))  # a heal inside the fight (8) stays: the gym's end HP has it
        self.assertEqual((o["human_end_hp"], o["human_hp_lost"], o["human_damage_taken"], o["human_post_heal"]), (38, 12, 20, 6))

    def test_a_heal_cut_short_by_the_hp_room_is_all_post_fight(self):
        o = self.out(78, self.stats(80, 3, 5))  # 78 - 3 = 75, Burning Blood could only heal five
        self.assertEqual((o["human_end_hp"], o["human_hp_lost"], o["human_post_heal"]), (75, 3, 5))

    def test_the_relics_that_heal_at_the_end_are_black_blood_chosen_cheese_and_meat_on_the_bone(self):
        self.assertEqual(hs.post_fight_heal(["BLACK_BLOOD"], 30), (12, False))
        self.assertEqual(hs.post_fight_heal(["BURNING_BLOOD", "CHOSEN_CHEESE"], 7), (7, False))
        self.assertEqual(hs.post_fight_heal(["BURNING_BLOOD"], 3), (3, False))
        self.assertEqual(hs.post_fight_heal(["POTION_BELT"], 9), (0, False))  # a relic that does not heal at the end owns none of it
        self.assertEqual(hs.post_fight_heal(["BURNING_BLOOD", "MEAT_ON_THE_BONE"], 18), (18, True))  # it triggered: the floor healed six and twelve
        self.assertEqual(hs.post_fight_heal(["BURNING_BLOOD", "MEAT_ON_THE_BONE"], 6), (6, True))  # HP stayed above half: it did not
        o = self.out(60, self.stats(44, 34, 18), held=("BLACK_BLOOD",))
        self.assertEqual((o["human_end_hp"], o["human_post_heal"]), (32, 12))

    def test_a_death_loses_the_whole_hp_going_in_and_scores_the_death_value(self):
        o = self.out(37, self.stats(0, 37, 0), survived=False, potions_in=2)
        self.assertEqual((o["human_survived"], o["human_end_hp"], o["human_hp_lost"], o["human_post_heal"]), (False, 0, 37, 0))
        self.assertEqual((o["human_l0"], o["human_l0_unit"]), (hs.DEATH_VALUE, 0.0))
        self.assertEqual(hs.DEATH_VALUE, -100.0)
        self.assertNotIn("human_flags", o)
        self.assertEqual(self.out(37, self.stats(5, 37, 0), survived=False)["human_flags"], ["death_hp_not_zero"])

    def test_the_value_is_the_search_currency_and_the_unit_l0_is_the_gyms(self):
        o = self.out(70, self.stats(30, 52, 12, potion_used=["POTION.FIRE_POTION"]), potions_in=2)
        self.assertEqual((o["human_end_hp"], o["human_potions_used"], o["human_potions_held_end"]), (24, 1, 1))
        self.assertEqual(o["human_l0"], 24 + hs.POTION_VALUE * 1)  # end HP plus ten a held potion, as practice_mix.played_value
        import fight_gym as fg  # noqa: PLC0415

        for won, hp, pots in ((True, 24, 1), (True, 80, 3), (False, 0, 0), (True, 1, 0)):
            self.assertEqual(hs.l0_unit(won, hp, pots, 80), fg.l0(won, hp, pots, 80))
        self.assertAlmostEqual(o["human_l0_unit"], fg.l0(True, 24, 1, 80), places=4)
        self.assertEqual(hs.POTION_VALUE, fg.POTION_VALUE)
        import practice_mix as pm  # noqa: PLC0415

        self.assertEqual(pm.played_value({"won": True, "end_hp": 24, "potions_held": 1}), o["human_l0"])

    def test_a_floor_that_lacks_a_figure_has_no_outcome_and_says_why(self):
        self.assertEqual(self.out(50, {"max_hp": 80, "damage_taken": 3, "hp_healed": 0}), "the floor lacks current_hp")
        self.assertEqual(self.out(50, {"current_hp": 40, "damage_taken": None, "hp_healed": 0}), "the floor lacks damage_taken")
        self.assertEqual(self.out(50, {"current_hp": 40, "damage_taken": 3}), "the floor lacks hp_healed")
        self.assertEqual(hs.fight_outcome(hp_in=None, max_hp_in=None, survived=True, ps=self.stats(1, 1, 0), room={}, held=[], potions_in=0),
                         "HP going in not known")
        o = self.out(50, self.stats(40, 16, 6), room={})  # no turn count: the outcome is kept, the turns are not known
        self.assertIsNone(o["human_turns"])

    def test_an_end_hp_the_figures_cannot_hold_is_cut_and_flagged(self):
        o = self.out(5, self.stats(3, 8, 6))  # 5 - 8 + 6 = 3, less six: below one for a run that went on (a revive's figures)
        self.assertEqual((o["human_end_hp"], o["human_flags"]), (1, ["end_hp_raised_to_1"]))
        o = self.out(80, self.stats(80, 0, 6, max_hp=80), held=("MEAT_ON_THE_BONE",))
        self.assertEqual(o["human_flags"], ["heal_guess"])


class OutcomeRunTests(unittest.TestCase):
    """The outcomes of a whole run follow the same floors `rebuild_run` builds starts from."""

    @classmethod
    def setUpClass(cls):
        cls.tables = hs.Tables.load()

    @staticmethod
    def run_with_results(rid, act2=True):
        run = synthetic_run(rid, act2=act2)
        a1 = run["map_point_history"][0]

        def put(point, cur, dmg, heal, turns, **extra):
            ps = point["player_stats"][0]
            ps.update(current_hp=cur, damage_taken=dmg, hp_healed=heal, **extra)
            point["rooms"][-1]["turns_taken"] = turns

        put(a1[1], 55, 15, 6, 4)  # 64 in: slimes
        put(a1[4], 30, 52, 12, 9, potion_used=["POTION.FIRE_POTION"])  # 70 in: the elite, a potion used, six healed inside the fight
        put(a1[16], 26, 10, 6, 7)  # 30 in: the Act 1 boss
        if act2:
            put(run["map_point_history"][1][1], 0, 80, 0, 6)  # 80 in: the Act 2 elite kills
        return run

    def test_the_outcomes_are_found_for_the_same_floors_the_starts_are_and_equal_the_specs(self):
        rid = _find_ids(False, 1, "k")[0]
        run = self.run_with_results(rid)
        outs, why, checks = hs.outcomes_of_run(run)
        specs = {s["id"]: s for s in hs.rebuild_run(run, self.tables, Counter())}
        self.assertEqual(sorted(outs), sorted(specs))
        self.assertEqual(why, {})
        for sid, s in specs.items():
            self.assertEqual(s["outcome"], outs[sid])
            self.assertEqual(checks[sid]["hp_in"], s["hp"])
        a, b, c, d = (outs[f"human:{rid}|f{f:02d}"] for f in (2, 5, 17, 19))
        self.assertEqual((a["human_survived"], a["human_end_hp"], a["human_hp_lost"], a["human_turns"]), (True, 49, 15, 4))
        self.assertEqual((b["human_end_hp"], b["human_hp_lost"], b["human_potions_used"], b["human_potions_held_end"]), (24, 46, 1, 0))
        self.assertEqual(b["human_l0"], 24.0)
        self.assertEqual((c["human_survived"], c["human_end_hp"], c["human_hp_lost"]), (True, 20, 10))  # the boss died: the run went on to Act 2
        self.assertEqual((d["human_survived"], d["human_end_hp"], d["human_hp_lost"], d["human_l0"]), (False, 0, 80, -100.0))  # last floor of the run

    def test_the_run_without_an_act_two_ends_at_its_boss(self):
        rid = _find_ids(False, 1, "k")[0]
        outs, _, _ = hs.outcomes_of_run(self.run_with_results(rid, act2=False))
        self.assertTrue(outs[f"human:{rid}|f05"]["human_survived"])
        self.assertFalse(outs[f"human:{rid}|f17"]["human_survived"])  # the boss is the last floor: the run ended there

    def test_a_floor_without_a_figure_is_counted_by_reason_and_an_event_page_leaves_no_start(self):
        rid = _find_ids(False, 1, "k")[0]
        run = self.run_with_results(rid)
        del run["map_point_history"][0][4]["player_stats"][0]["damage_taken"]
        run["map_point_history"][0][1]["rooms"] = [{"model_id": "EVENT.X", "room_type": "event"},
                                                    {"model_id": "ENCOUNTER.SLIMES_WEAK", "monster_ids": ["MONSTER.LEAF_SLIME_S"],
                                                     "room_type": "monster"}]
        outs, why, _ = hs.outcomes_of_run(run)
        self.assertEqual(sorted(outs), [f"human:{rid}|f17", f"human:{rid}|f19"])
        self.assertEqual(why, {f"human:{rid}|f05": "the floor lacks damage_taken"})

    def test_the_start_record_carries_the_fields_for_a_future_build(self):
        rid = _find_ids(False, 1, "k")[0]
        spec = next(s for s in hs.rebuild_run(self.run_with_results(rid), self.tables, Counter()) if s["floor"] == 5)
        doc = {"potions": {"slots": [3, 0], "max_slots": 2}, "piles": {"hand": [{"x": 1}]}, "turn": 0, "encounter_id": spec["encounter_id"]}
        rec = hs.start_record(spec, doc)
        for k in hs.OUTCOME_KEYS:
            self.assertIn(k, rec)
        self.assertEqual((rec["human_end_hp"], rec["human_hp_lost"], rec["human_survived"]), (24, 46, True))
        self.assertNotIn("human_outcome_missing", rec)
        spec["outcome"], spec["outcome_missing"] = None, "the floor lacks hp_healed"
        rec = hs.start_record(spec, doc)
        self.assertEqual(rec["human_outcome_missing"], "the floor lacks hp_healed")
        self.assertNotIn("human_end_hp", rec)


class OutcomeSidecarTests(unittest.TestCase):
    """The pass over the corpus writes `human-outcomes.json` beside the start set, atomically, and the doc section."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        ok = _find_ids(False, 4, "k")
        self.exam_listed = _find_ids(False, 1, "l")[0]
        self.runs = [OutcomeRunTests.run_with_results(r) for r in ok]
        with gzip.open(self.t / "corpus.jsonl.gz", "wt", encoding="utf-8") as fh:
            for r in self.runs + [OutcomeRunTests.run_with_results(self.exam_listed)]:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        (self.t / "ids.txt").write_text(self.exam_listed + "\n", encoding="utf-8")
        self.out = self.t / "starts"
        self.out.mkdir()
        lite = []
        for run in self.runs[:3]:  # the fourth run's starts are not in the set: its outcomes are not kept
            rid = hs.run_id(run)
            for f, act, kind, hp in ((2, 1, "hallway", 64), (5, 1, "elite", 70), (17, 1, "boss", 30), (19, 2, "elite", 80)):
                lite.append({"id": f"human:{rid}|f{f:02d}", "run": rid, "act": act, "kind": kind, "hp": hp})
        lite.append({"id": f"human:{self.runs[0]['run_hash']}|f09", "run": self.runs[0]["run_hash"], "act": 1, "kind": "hallway", "hp": 70})
        (self.out / "human-starts.lite.json").write_text(json.dumps(lite), encoding="utf-8")
        self.lite = lite

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *extra):
        import contextlib  # noqa: PLC0415
        import io  # noqa: PLC0415

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = hs.main(["outcomes", "--corpus", str(self.t / "corpus.jsonl.gz"), "--exam-ids", str(self.t / "ids.txt"),
                          "--out-dir", str(self.out), "--doc", str(self.t / "doc.md"), *extra])
        return rc, buf.getvalue()

    def test_the_sidecar_is_keyed_by_start_id_and_written_whole(self):
        rc, text = self.run_main()
        self.assertEqual(rc, 0)
        doc = json.loads((self.out / "human-outcomes.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["rule"], hs.OUTCOME_RULE)
        self.assertEqual(doc["n"], 12)
        self.assertEqual(len(doc["outcomes"]), 12)
        self.assertEqual(sorted(doc["outcomes"]), sorted(r["id"] for r in self.lite if r["id"].endswith(("f02", "f05", "f17", "f19"))))
        rid = self.lite[0]["run"]
        self.assertEqual(doc["outcomes"][f"human:{rid}|f19"]["human_l0"], -100.0)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["human-outcomes.json", "human-starts.lite.json"])  # no temp file left
        self.assertIn("start not found again in the corpus pass: 1", text)  # the f09 start the corpus has no fight on
        self.assertNotIn(self.exam_listed, text)

    def test_the_doc_section_is_replaced_not_repeated_and_a_build_keeps_it(self):
        self.run_main()
        self.run_main()
        text = (self.t / "doc.md").read_text(encoding="utf-8")
        self.assertEqual(text.count("## Human outcomes"), 1)
        self.assertIn("| act1-boss | 3 | < 20 |", text)  # three runs: the cell is hidden
        self.assertNotIn(self.runs[0]["run_hash"], text)
        (self.t / "doc.md").write_text("# Human fight starts: coverage\n\nold coverage\n\n" + text[text.index("## Human outcomes"):], encoding="utf-8")
        hs.write_doc_section(self.t / "doc.md", "## Human outcomes\n\nnew\n")
        again = (self.t / "doc.md").read_text(encoding="utf-8")
        self.assertTrue(again.startswith("# Human fight starts: coverage\n\nold coverage\n\n## Human outcomes\n\nnew"))
        self.assertEqual(again.count("## Human outcomes"), 1)

    def test_a_limited_pass_does_not_touch_the_sidecar_and_the_exam_fifth_is_never_a_start(self):
        self.run_main("--limit-runs", "2")
        self.assertFalse((self.out / "human-outcomes.json").exists())
        self.run_main("--limit-runs", "2", "--force-partial")
        self.assertTrue((self.out / "human-outcomes.json").exists())
        bad = json.loads((self.out / "human-starts.lite.json").read_text())
        bad.append({"id": f"human:{self.exam_listed}|f05", "run": self.exam_listed, "act": 1, "kind": "elite", "hp": 70})
        (self.out / "human-starts.lite.json").write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(AssertionError):
            self.run_main()

    def test_a_start_whose_hp_going_in_differs_from_the_records_is_not_given_an_outcome(self):
        lite = json.loads((self.out / "human-starts.lite.json").read_text())
        lite[1]["hp"] = 71  # the start says 71 going into the elite; the floor before it says 70
        (self.out / "human-starts.lite.json").write_text(json.dumps(lite), encoding="utf-8")
        _, text = self.run_main()
        doc = json.loads((self.out / "human-outcomes.json").read_text(encoding="utf-8"))
        self.assertNotIn(lite[1]["id"], doc["outcomes"])
        self.assertEqual(len(doc["outcomes"]), 11)
        self.assertIn("hp_going_in_differs_from_the_start: 1", text)


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
        spec = self.specs[19]
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
