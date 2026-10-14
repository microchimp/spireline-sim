"""referee/refset.py and tools/fidelity/refset_score.py: the reference set's specs, references and numbers.

Synthetic run histories (the game's run-history shape) stand in for the codex data, and a small export in
the game's table shape for the real one. What must hold: a real run's state before a floor is rebuilt from the
history (cards added or upgraded later are undone, removed cards come back, saved counters rebuilt and checked
against the run's end, or flagged), the setup goes through the game's save-load path in the fight's own act,
coverage plans reach every item, a fight whose two recordings differ, whose game side failed or whose setup the
game could not build is never a reference, the game's errors land on the fight that logged them, a driver build
without the fixes cannot record, and the numbers state exclusions and flags next to every rate.
"""

from __future__ import annotations

import collections
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT))

import refset  # noqa: E402
from tools.fidelity import refset_score  # noqa: E402


def card(cid, floor=1, up=0, ench=None, amount=1, props=None):
    c = {"id": "CARD." + cid, "floor_added_to_deck": floor}
    if up:
        c["current_upgrade_level"] = up
    if ench:
        c["enchantment"] = {"id": "ENCHANTMENT." + ench, "amount": amount}
    if props:
        c["props"] = props
    return c


def point(kind="monster", enc="NIBBITS_WEAK", hp=60, max_hp=80, turns=3, gold=99, **stats):
    room = ({"model_id": "ENCOUNTER." + enc, "room_type": kind, "turns_taken": turns} if kind in ("monster", "elite", "boss")
            else {"model_id": "EVENT.X", "room_type": kind, "turns_taken": 0})
    return {"map_point_type": kind, "rooms": [room],
            "player_stats": [dict({"current_hp": hp, "max_hp": max_hp, "current_gold": gold}, **stats)]}


def ints(**kv):
    return {"ints": [{"name": k, "value": v} for k, v in kv.items()]}


def history():
    """Neow (floor 1, Happy Flower picked), a fight at 2 (Anger picked, 3 turns), an upgrade of Bash at 3, a fight at 4
    (4 turns; Vajra), a removal of a Strike and an enchanted Defend at 5, a fight at 6 (Pommel Strike added, a potion
    used, 2 turns). Happy Flower's end value: (3 + 4 + 2) % 3 = 0, saved as absent."""
    pts = [
        point("ancient", ancient_choice=[{"TextKey": "A", "was_chosen": False}, {"TextKey": "NEOW_RELIC", "was_chosen": True}],
              relic_choices=[{"choice": "RELIC.NEOW_RELIC", "was_picked": True}]),
        point(potion_choices=[{"choice": "POTION.FIRE_POTION", "was_picked": True}], gold=120,
              card_choices=[{"card": card("ANGER", 2), "was_picked": True}]),
        point("rest_site", upgraded_cards=["CARD.BASH"], gold=120),
        point("elite", "BYRDONIS_ELITE", hp=40, turns=4, gold=150, relic_choices=[{"choice": "RELIC.VAJRA", "was_picked": True}]),
        point("event", cards_removed=[card("STRIKE_IRONCLAD")], gold=150,
              cards_enchanted=[{"card": card("DEFEND_IRONCLAD", ench="SHARP"), "enchantment": "ENCHANTMENT.SHARP"}]),
        point(hp=30, turns=2, potion_used=["POTION.FIRE_POTION"], card_choices=[{"card": card("POMMEL_STRIKE", 6), "was_picked": True}]),
    ]
    deck = ([card("STRIKE_IRONCLAD")] * 4 + [card("DEFEND_IRONCLAD", ench="SHARP")] + [card("DEFEND_IRONCLAD")] * 3
            + [card("BASH", up=1), card("ANGER", 2), card("POMMEL_STRIKE", 6)])
    relics = [{"id": "RELIC.BURNING_BLOOD", "floor_added_to_deck": 1}, {"id": "RELIC.NEOW_RELIC", "floor_added_to_deck": 1},
              {"id": "RELIC.HAPPY_FLOWER", "floor_added_to_deck": 1},
              {"id": "RELIC.VAJRA", "floor_added_to_deck": 4}]
    return {"seed": "ABCDE12345", "run_hash": "r1", "acts": ["ACT.OVERGROWTH"], "map_point_history": [pts], "win": True,
            "players": [{"deck": deck, "relics": relics, "potions": [], "max_potion_slot_count": 2}]}


class Rebuild(unittest.TestCase):
    def pts(self, d):
        return [p for a in d["map_point_history"] for p in a]

    def test_deck_before_a_floor_undoes_what_came_later(self) -> None:
        d = history()
        deck, exact = refset.deck_before(d, self.pts(d), 4)
        got = collections.Counter((c[0], c[1], c[2]) for c in deck)
        # Floor 4: Anger is in, Bash is upgraded (floor 3), the removed Strike is back, the Defend not yet enchanted,
        # Pommel Strike (floor 6) is not there.
        self.assertEqual(got, collections.Counter({("STRIKE_IRONCLAD", 0, ""): 5, ("DEFEND_IRONCLAD", 0, ""): 4,
                                                   ("BASH", 1, ""): 1, ("ANGER", 0, ""): 1}))
        self.assertTrue(exact)

    def test_deck_before_the_upgrade_has_a_plain_bash(self) -> None:
        d = history()
        deck, _ = refset.deck_before(d, self.pts(d), 2)
        self.assertIn(["BASH", 0, "", 0, 1], [c[:5] for c in deck])
        self.assertNotIn("ANGER", [c[0] for c in deck])

    def test_deck_after_the_event_keeps_its_changes(self) -> None:
        d = history()
        deck, _ = refset.deck_before(d, self.pts(d), 6)
        got = collections.Counter((c[0], c[2]) for c in deck)
        self.assertEqual(got[("STRIKE_IRONCLAD", "")], 4)
        self.assertEqual(got[("DEFEND_IRONCLAD", "SHARP")], 1)

    def test_relics_before_a_floor(self) -> None:
        d = history()
        self.assertEqual(refset.relics_before(d, self.pts(d), 4), ["BURNING_BLOOD", "NEOW_RELIC", "HAPPY_FLOWER"])
        self.assertEqual(refset.relics_before(d, self.pts(d), 6), ["BURNING_BLOOD", "NEOW_RELIC", "HAPPY_FLOWER", "VAJRA"])

    def test_a_removed_relic_comes_back_before_its_removal(self) -> None:
        d = history()
        self.pts(d)[4]["player_stats"][0]["relics_removed"] = ["RELIC.NEOW_RELIC"]
        d["players"][0]["relics"] = [r for r in d["players"][0]["relics"] if "NEOW" not in r["id"]]
        self.assertEqual(refset.relics_before(d, self.pts(d), 4), ["BURNING_BLOOD", "HAPPY_FLOWER", "NEOW_RELIC"])
        self.assertEqual(refset.relics_before(d, self.pts(d), 6), ["BURNING_BLOOD", "HAPPY_FLOWER", "VAJRA"])
        # The removed relic has no end record: its saved state (if any) is unknown.
        full = refset.relics_before_full(d, self.pts(d), 4)
        self.assertEqual([r[3] for r in full if r[1] == "NEOW_RELIC"], [False])

    def test_potions_walk_forward_and_check_the_final_belt(self) -> None:
        d = history()
        belt, exact = refset.potions_before(self.pts(d), 4, d["players"][0]["potions"])
        self.assertEqual(belt, ["FIRE_POTION"])
        self.assertTrue(exact)
        _, exact = refset.potions_before(self.pts(d), 4, [{"id": "POTION.BLOCK_POTION"}])
        self.assertFalse(exact)


class SavedState(unittest.TestCase):
    """state_before: the [SavedProperty] values before a floor, by STATE_RULES, checked against the run's end."""

    def ev(self, turns=(3, 4, 2)):
        d = history()
        pts = [p for a in d["map_point_history"] for p in a]
        for p, t in zip((pts[1], pts[3], pts[5]), turns):
            p["rooms"][0]["turns_taken"] = t
        return refset.floor_events(pts)

    def test_a_turn_counter_is_rebuilt_and_checked_against_the_end(self) -> None:
        flags = []
        # Happy Flower from floor 1: before floor 6, 3 + 4 turns seen = 7 % 3 = 1; the end (9 % 3 = 0) is absent (fresh).
        got = refset.state_before("HAPPY_FLOWER", {}, True, 1, 6, self.ev(), 0, flags)
        self.assertEqual(got, {"ints": [{"name": "TurnsSeen", "value": 1}]})
        self.assertEqual(flags, [])
        # An end value the rule cannot reach: the rebuilt value is kept but the spec is flagged.
        flags = []
        refset.state_before("HAPPY_FLOWER", ints(TurnsSeen=2), True, 1, 6, self.ev(), 0, flags)
        self.assertEqual(flags, ["state_unverified:HAPPY_FLOWER.TurnsSeen"])

    def test_a_fixed_pick_is_the_end_value(self) -> None:
        flags = []
        props = {"cards": [{"name": "AncientCard", "value": {"id": "CARD.BREAK"}},
                           {"name": "StarterCard", "value": {"id": "CARD.BASH", "floor_added_to_deck": 1}}]}
        self.assertEqual(refset.state_before("ARCHAIC_TOOTH", props, True, 3, 6, self.ev(), 0, flags), props)
        mad = {"ints": [{"name": "TinkerTimeType", "value": 2}, {"name": "TinkerTimeRider", "value": 5}]}
        self.assertEqual(refset.state_before("MAD_SCIENCE", mad, True, 2, 6, self.ev(), 0, flags), mad)
        self.assertEqual(flags, [])

    def test_a_counter_the_record_cannot_place_is_flagged_not_freshened_silently(self) -> None:
        flags = []
        got = refset.state_before("PEN_NIB", ints(AttacksPlayed=37), True, 2, 6, self.ev(), 0, flags)
        self.assertEqual(flags, ["state_unknown:PEN_NIB.AttacksPlayed"])
        self.assertEqual(got, ints(AttacksPlayed=0))
        flags = []
        refset.state_before("PEN_NIB", ints(AttacksPlayed=0), True, 2, 6, self.ev(), 0, flags)
        self.assertEqual(flags, [])  # fresh at the end: fresh before the floor too
        flags = []
        refset.state_before("PEN_NIB", {}, False, 2, 6, self.ev(), 0, flags)  # left the run before its end
        self.assertEqual(flags, ["state_unknown:PEN_NIB"])

    def test_combat_counters(self) -> None:
        flags = []
        # Bone Tea from floor 1 (1 combat left): spent by the fight at floor 2, so 0 before floor 4, and 0 at the end.
        self.assertEqual(refset.state_before("BONE_TEA", ints(CombatsLeft=0), True, 1, 4, self.ev(), 0, flags),
                         ints(CombatsLeft=0))
        # Sword of Stone from floor 1: the floor-4 elite won, so 1 before floor 6.
        self.assertEqual(refset.state_before("SWORD_OF_STONE", ints(ElitesDefeated=1), True, 1, 6, self.ev(), 0, flags),
                         ints(ElitesDefeated=1))
        self.assertEqual(flags, [])
        # The fight a lost run ended on was not won: an elite there does not count.
        flags = []
        refset.state_before("SWORD_OF_STONE", ints(ElitesDefeated=0), True, 1, 4, self.ev(), 4, flags)
        self.assertEqual(flags, [])

    def test_reward_and_room_counters(self) -> None:
        ev = self.ev()
        ev[1]["card_reward"] = ev[3]["card_reward"] = True
        ev[2]["treasure"] = True
        ev[3]["act1_boss"] = True
        flags = []
        # Silver Crucible from floor 1: card rewards on floors 2, 4 and 6 (3 by the end, up to 3: 2 before floor 6) and one
        # treasure room (floor 3).
        got = refset.state_before("SILVER_CRUCIBLE", ints(TimesUsed=3, TreasureRoomsEntered=1), True, 1, 6, ev, 0, flags)
        self.assertEqual(got, ints(TimesUsed=2, TreasureRoomsEntered=1))
        # Silken Tress spends itself on the first card reward (floor 2): used before floor 4, fresh before floor 2.
        tress = {"bools": [{"name": "IsUsed", "value": True}]}
        self.assertEqual(refset.state_before("SILKEN_TRESS", tress, True, 1, 4, ev, 0, flags), tress)
        self.assertEqual(refset.state_before("SILKEN_TRESS", tress, True, 1, 2, ev, 0, flags), {"bools": [{"name": "IsUsed", "value": False}]})
        # Lava Rock's act 1 boss (floor 4): triggered before floor 6, not before floor 4, and not when the run died there.
        rock = {"bools": [{"name": "HasTriggered", "value": True}]}
        self.assertEqual(refset.state_before("LAVA_ROCK", rock, True, 1, 6, ev, 0, flags), rock)
        self.assertEqual(flags, [])
        refset.state_before("LAVA_ROCK", rock, True, 1, 6, ev, 4, flags)
        self.assertEqual(flags, ["state_unverified:LAVA_ROCK.HasTriggered"])

    def test_an_unknown_saved_property_flags_the_spec(self) -> None:
        flags = []
        refset.state_before("SOME_RELIC", ints(Mystery=3), True, 1, 6, self.ev(), 0, flags)
        self.assertEqual(flags, ["state_unknown:SOME_RELIC.Mystery"])


class FakeTables(refset.Tables):
    def __init__(self):  # noqa: D107 (no export: just the fields the spec makers read)
        self.follow = {r: "None" for r in ("BURNING_BLOOD", "NEOW_RELIC", "VAJRA", "ANCHOR", "POTION_BELT", "ORRERY", "HAPPY_FLOWER",
                                           "ARCHAIC_TOOTH", "TOUCH_OF_OROBAS", "BLACK_BLOOD")}
        self.follow["ORRERY"] = "PreRolledCardReward"
        self.cards_emu = {"STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH", "ANGER", "POMMEL_STRIKE", "ASCENDERS_BANE", "INFLAME", "BREAK"}
        self.potions_emu = {"FIRE_POTION", "BLOCK_POTION"}
        self.enchant_emu = {"SHARP"}
        self.encounters = {"NIBBITS_WEAK": ("OVERGROWTH", "weak", 1), "BYRDONIS_ELITE": ("OVERGROWTH", "elite", 1),
                           "KAISER_CRAB_BOSS": ("HIVE", "boss", 0)}
        self.card_info = {c: {"id": c, "rarity": "Common", "max_upgrade_level": "1"} for c in self.cards_emu}
        self.cards = sorted(self.cards_emu - {"ASCENDERS_BANE", "BREAK"})
        self.upgradable = set(self.cards)
        self.other_cards, self.other_relics, self.other_potions = set(), set(), set()
        self.pool_relics = {"VAJRA": "Common", "ANCHOR": "Common", "POTION_BELT": "Shop", "BURNING_BLOOD": "Starter", "ORRERY": "Shop"}
        self.event_relics = set()
        self.pool_potions = {"FIRE_POTION", "BLOCK_POTION"}


class Specs(unittest.TestCase):
    def test_realistic_specs_load_the_floor_state_through_the_save_load_path(self) -> None:
        stats = collections.Counter()
        got = list(refset.realistic_candidates(history(), FakeTables(), stats))
        floors = {c["floor"]: c for _, _, _, c in got}
        self.assertEqual(sorted(floors), [2, 4, 6])
        c = floors[6]
        self.assertEqual((c["room"], c["encounter"], c["hp"], c["neow"], c["neow_relic"]), ("monster", "NIBBITS_WEAK", 60, 1, "NEOW_RELIC"))  # HP and gold as floor 5 left them
        entry = refset.realistic_entry(c, FakeTables())
        items = entry["spec"]["setup"]
        self.assertEqual([it["cmd"] for it in items], ["player_load", "fight"])  # act 1: no act item
        load = items[0]
        self.assertEqual((load["hp"], load["max_hp"], load["gold"], load["max_potion_slots"]), (60, 80, 150, 2))
        # The save's own shape, with the rebuilt counter (Happy Flower: 3 + 4 turns before floor 6).
        flower = next(r for r in load["relics"] if r["id"] == "RELIC.HAPPY_FLOWER")
        self.assertEqual(flower["props"], {"ints": [{"name": "TurnsSeen", "value": 1}]})
        self.assertIn({"id": "CARD.BASH", "current_upgrade_level": 1, "floor_added_to_deck": 1}, load["deck"])
        self.assertIn({"id": "CARD.DEFEND_IRONCLAD", "enchantment": {"id": "ENCHANTMENT.SHARP", "amount": 1}, "floor_added_to_deck": 1},
                      load["deck"])
        self.assertEqual(entry["spec"]["seed"], "ABCDE12345")
        self.assertEqual(entry["spec"]["neow"], 1)
        self.assertEqual((entry["chooser"], entry["spec"]["chooser"], entry["spec"]["relic_flashes"]), ("heuristic", "heuristic", True))
        self.assertEqual(entry["flags"], [])

    def test_an_act_2_fight_is_met_in_act_2(self) -> None:
        d = history()
        d["acts"] = ["ACT.OVERGROWTH", "ACT.HIVE"]
        pts = d["map_point_history"][0]
        d["map_point_history"] = [pts[:3], pts[3:]]
        pts[5]["rooms"][0]["model_id"] = "ENCOUNTER.KAISER_CRAB_BOSS"
        pts[5]["rooms"][0]["room_type"] = "boss"
        stats = collections.Counter()
        c = next(c for _, _, _, c in refset.realistic_candidates(d, FakeTables(), stats) if c["floor"] == 6)
        entry = refset.realistic_entry(c, FakeTables())
        self.assertEqual([it["cmd"] for it in entry["spec"]["setup"]], ["act", "player_load", "fight"])
        self.assertEqual(entry["spec"]["setup"][0]["index"], 2)
        self.assertEqual(entry["expect_act"], "HIVE")

    def test_realism_flags_name_what_the_record_cannot_give(self) -> None:
        d = history()
        d["players"][0]["relics"].append({"id": "RELIC.PEN_NIB", "floor_added_to_deck": 2, "props": ints(AttacksPlayed=12)})
        d["map_point_history"][0][1]["player_stats"][0]["relic_choices"] = [{"choice": "RELIC.PEN_NIB", "was_picked": True}]
        t = FakeTables()
        t.follow["PEN_NIB"] = "None"
        c = next(c for _, _, _, c in refset.realistic_candidates(d, t, collections.Counter()) if c["floor"] == 6)
        self.assertEqual(refset.realistic_entry(c, t)["flags"], ["state_unknown:PEN_NIB.AttacksPlayed"])

    def test_a_relic_the_emulator_cannot_mirror_drops_the_floor(self) -> None:
        d = history()
        d["players"][0]["relics"].append({"id": "RELIC.ORRERY", "floor_added_to_deck": 3})
        stats = collections.Counter()
        floors = [c["floor"] for _, _, _, c in refset.realistic_candidates(d, FakeTables(), stats)]
        self.assertEqual(floors, [2])
        self.assertEqual(stats["cand_skip_relic_unmodelled"], 2)

    def test_coverage_plans_every_item_its_quota(self) -> None:
        t = FakeTables()
        specs = refset.coverage_specs(t, {"relics": set(), "potions": set()})
        tg = refset.coverage_targets(t, {})
        enc = collections.Counter(s["targets"]["encounter"] for s in specs)
        self.assertTrue(all(enc[e] >= refset.PLAN["encounter"] for e in tg["encounters"]))
        cards = collections.Counter(c for s in specs for c in s["targets"]["cards"])
        self.assertTrue(all(cards[c + ("+" if u else "")] >= refset.PLAN["card"] for c, u in tg["variants"]))
        relics = collections.Counter(r for s in specs for r in s["targets"]["relics"])
        self.assertTrue(all(relics[r] >= refset.PLAN["relic"] for r in tg["relics"]))
        self.assertNotIn("ORRERY", tg["relics"])  # its pickup is not mirrored
        for s in specs:
            items = s["spec"]["setup"]
            self.assertEqual(items[-1]["cmd"], "fight")
            self.assertEqual(items[-2]["cmd"], "player_load")
            ids = [r["id"] for r in items[-2]["relics"]]
            self.assertEqual(len(set(ids)), len(ids))  # no duplicate relic
            self.assertEqual(s["chooser"], "random")
            act = refset.ACT_NUMBER[t.encounters[s["targets"]["encounter"]][0]]
            self.assertEqual(items[0]["cmd"] == "act", act > 1)

    def test_coverage_holds_an_event_relic_as_its_pickup_left_it(self) -> None:
        fixed = {"ARCHAIC_TOOTH": {"cards": [{"name": "AncientCard", "value": {"id": "CARD.BREAK"}},
                                             {"name": "StarterCard", "value": {"id": "CARD.BASH"}}]},
                 "TOUCH_OF_OROBAS": {"model_ids": [{"name": "StarterRelic", "value": "RELIC.BURNING_BLOOD"},
                                                   {"name": "UpgradedRelic", "value": "RELIC.BLACK_BLOOD"}]},
                 "DUSTY_TOME": {"model_ids": [{"name": "AncientCard", "value": "CARD.CORRUPTION"}]}}
        deck, relics = refset.coverage_holdings(["BURNING_BLOOD", "ARCHAIC_TOOTH", "TOUCH_OF_OROBAS", "DUSTY_TOME"],
                                                [["STRIKE_IRONCLAD", 0, "", 0], ["BASH", 1, "", 0]], fixed)
        self.assertEqual([c["id"] for c in deck], ["CARD.STRIKE_IRONCLAD", "CARD.BREAK", "CARD.CORRUPTION"])
        self.assertEqual(deck[1].get("current_upgrade_level"), 1)  # the transform keeps the upgrade
        self.assertEqual(deck[2].get("current_upgrade_level"), 1)  # Dusty Tome's card comes upgraded
        self.assertEqual([r["id"] for r in relics], ["RELIC.BLACK_BLOOD", "RELIC.ARCHAIC_TOOTH", "RELIC.TOUCH_OF_OROBAS", "RELIC.DUSTY_TOME"])
        self.assertEqual(relics[1]["props"], fixed["ARCHAIC_TOOTH"])

    def test_regression_classes_carry_their_trigger(self) -> None:
        self.assertEqual(refset.trigger_of(["STRIKE_IRONCLAD", "RUPTURE"], [], [], []), {"plays": ["RUPTURE"]})
        self.assertEqual(refset.trigger_of(["DEFEND_IRONCLAD"], [], [], [("THORNS_POWER", 3, "player")]),
                         {"plays": ["DEFEND_IRONCLAD"], "powers_hit": ["THORNS_POWER"]})
        self.assertEqual(refset.trigger_of([], ["PHIAL_HOLSTER"], [], []), {"relic_flashes": ["PHIAL_HOLSTER"]})
        self.assertEqual(refset.trigger_of([], [], [], []), {"replay": True})
        self.assertEqual(refset.trigger_of([], ["PHIAL_HOLSTER"], [], [], pickup=True), {"pickup": ["PHIAL_HOLSTER"]})
        v = [{"relics": ["BURNING_BLOOD", "PHIAL_HOLSTER"], "fires": []}]
        self.assertTrue(refset_score.trigger_reached({"pickup": ["PHIAL_HOLSTER"]}, v))
        self.assertFalse(refset_score.trigger_reached({"relic_flashes": ["PHIAL_HOLSTER"]}, v))
        self.assertEqual(refset.class_kind("driver: turn-start auto-play"), "driver")

    def test_the_thorns_class_is_reached_only_when_an_attacker_dies_against_it(self) -> None:
        name = "hook order: the player's Thorns before an enemy hit lands (open)"
        self.assertEqual(refset.class_trigger(name), {"plays": ["DEFEND_IRONCLAD"], "powers_hit": ["THORNS_POWER"],
                                                      "attacker_killed": ["THORNS_POWER"]})
        # Its minimal setup's Thorns can kill a 48 HP Nibbit inside its attacks (3 never did).
        self.assertTrue(any(c[0] == name and c[6] == [("THORNS_POWER", 15, "player")] for c in refset.REGRESSION_CLASSES))

        def st(nibbits, thorns=True):
            return {"player": {"hp": 50, "powers": [["THORNS_POWER", 15]] if thorns else []},
                    "enemies": [{"name": "NIBBIT", "hp": hp, "intents": [["Attack", 7, 1]]} for hp in nibbits]}
        attacked = {"steps": [{"action": None, "state": st([40, 30])}, {"action": {"kind": "end_turn"}, "state": st([25, 15])}]}
        killed = {"steps": [{"action": None, "state": st([40, 10])}, {"action": {"kind": "end_turn"}, "state": st([25, 0])}]}
        ev_a, ev_k = refset_score.evidence(attacked, {1}), refset_score.evidence(killed, {1})
        self.assertEqual((ev_a["powers_hit"], ev_a["attacker_killed"]), (["THORNS_POWER"], []))
        self.assertEqual(ev_k["attacker_killed"], ["THORNS_POWER"])
        trig = refset.class_trigger(name)
        play = {"plays": [["DEFEND_IRONCLAD", True]]}
        self.assertFalse(refset_score.trigger_reached(trig, [ev_a | play]))
        self.assertTrue(refset_score.trigger_reached(trig, [ev_k | play]))


def result(seed="S", steps=3, err=None, setup_ok=True, msg=""):
    r = {"seed": seed, "steps": [{"action": None if i == 0 else {"kind": "end_turn"}, "state": {"in_combat": True, "round": i}}
                                 for i in range(steps)], "fight": {"id": "NIBBITS_WEAK", "current_act": "OVERGROWTH"},
         "setup_log": [{"i": -1, "cmd": "before", "ok": True}, {"i": 0, "cmd": "player_load", "ok": setup_ok, "msg": msg}],
         "phase_ms": {"x": 1}}
    if err:
        r["error"] = err
    return r


class Recording(unittest.TestCase):
    entry = {"id": "X", "spec": {"seed": "S", "setup": [{"cmd": "player_load"}, {"cmd": "fight", "id": "NIBBITS_WEAK"}]}}  # noqa: RUF012

    def test_a_clean_recording_is_kept(self) -> None:
        self.assertIsNone(refset.exclusion(self.entry, result()))

    def test_game_errors_and_failed_setups_are_excluded(self) -> None:
        self.assertIn("game error", refset.exclusion(self.entry, result(err="idle_timeout after step 3")))
        self.assertTrue(refset.exclusion(self.entry, result(setup_ok=False)).startswith("setup:"))
        self.assertIn("no result", refset.exclusion(self.entry, None))
        r = result()
        del r["fight"]
        self.assertIn("not entered", refset.exclusion(self.entry, r))

    def test_a_deck_the_setup_could_not_build_is_a_setup_failure(self) -> None:
        # Even when an older driver logged it ok: a residual is not the asked deck.
        why = refset.exclusion(self.entry, result(msg="residual: 2 extra, missing DEFEND_IRONCLAD,DEFEND_IRONCLAD | deck: X"))
        self.assertTrue(why.startswith("setup:"), why)
        self.assertEqual(refset.cause_of(why), "setup")

    def test_a_fight_met_in_the_wrong_act_is_a_setup_failure(self) -> None:
        e = dict(self.entry, expect_act="HIVE")
        self.assertIn("met in OVERGROWTH, not HIVE", refset.exclusion(e, result()))

    def test_timing_does_not_make_a_fight_unstable_but_a_state_does(self) -> None:
        a, b = result(), result()
        b["phase_ms"] = {"x": 999}
        self.assertEqual(refset.comparable(a), refset.comparable(b))
        b["steps"][2]["state"]["round"] = 7
        self.assertNotEqual(refset.comparable(a), refset.comparable(b))
        self.assertTrue(refset.first_difference(a, b).startswith("step 2"))

    def test_causes(self) -> None:
        self.assertEqual(refset.cause_of("the game logged an error: X at Y"), "game_error")
        self.assertEqual(refset.cause_of("game error: idle_timeout after step 3: {}"), "driver_wait_limit")
        self.assertEqual(refset.cause_of("unstable: the two recordings differ at step 4"), "unstable")
        self.assertEqual(refset.cause_of("game log: the game log cannot be cut into this server's fights"), "log_uncut")
        # A fight that killed two servers (crash, run-start exception, the watchdog) is not a logged game error.
        self.assertEqual(refset.cause_of("game error: stalled: the game failed this fight twice (no progress, crash or "
                                         "run-start exception)"), "stalled")
        self.assertEqual(refset.cause_of("game error: System.NullReferenceException"), "game_error")

    def test_the_summary_counts_exclusions_and_flags_per_cell(self) -> None:
        idx = [{"id": "a", "part": "realistic", "strata": {"act": 3, "room": "elite"}, "flags": []},
               {"id": "b", "part": "realistic", "strata": {"act": 3, "room": "elite"}, "flags": ["state_unknown:PEN_NIB.AttacksPlayed"]},
               {"id": "c", "part": "realistic", "strata": {"act": 3, "room": "elite"}, "flags": [], "excluded": "setup: item 0 failed"},
               {"id": "d", "part": "coverage", "strata": {"act": 1, "room": "monster"}, "flags": []}]
        s = refset.summarize(idx)
        self.assertEqual(s["realistic_cells"]["act3 elite"], {"specs": 3, "kept": 2, "excluded": {"setup": 1}, "flagged": 1,
                                                              "flags": {"state_unknown": 1}})
        self.assertEqual(s["parts"]["coverage"]["kept"], 1)
        self.assertEqual(s["flags_by_item"], {"state_unknown:PEN_NIB.AttacksPlayed": 1})


    def test_the_manifest_counts_its_specs_flags_per_cause_and_cell(self) -> None:
        es = [{"id": "a", "part": "realistic", "strata": {"act": 3, "room": "elite"},
               "flags": ["state_unknown:PEN_NIB.AttacksPlayed", "state_unknown:JOSS_PAPER.CardsExhausted"]},
              {"id": "b", "part": "realistic", "strata": {"act": 3, "room": "elite"}, "flags": ["state_unverified:GUILTY.CombatsSeen"]},
              {"id": "c", "part": "realistic", "strata": {"act": 1, "room": "boss"}, "flags": []},
              {"id": "d", "part": "coverage", "strata": {"act": 1, "room": "monster"}, "flags": []}]
        f = refset.manifest_flags(es)
        self.assertEqual((f["realistic"]["specs"], f["realistic"]["flagged"]), (3, 2))
        # A spec counts once per cause, however many of its items carry that cause.
        self.assertEqual(f["realistic"]["by_cause"], {"state_unknown": 1, "state_unverified": 1})
        self.assertEqual(f["realistic"]["cells"]["act3 elite"]["flagged"], 2)
        self.assertEqual(f["realistic"]["cells"]["act1 boss"], {"specs": 1, "flagged": 0, "by_cause": {}})
        self.assertEqual(f["realistic"]["by_flag"]["state_unknown:PEN_NIB.AttacksPlayed"], 1)
        self.assertEqual(f["coverage"], {"specs": 1, "flagged": 0, "by_cause": {}, "by_flag": {}})

    def test_a_trial_never_replaces_the_subsets_record(self) -> None:
        self.assertEqual(refset.record_pointer("pilot", "", 0), "refs-pilot.json")
        self.assertEqual(refset.record_pointer("all", "", 0), "refs-all.json")
        self.assertIsNone(refset.record_pointer("all", "R1M-X-F03,C0001-Y", 0))
        self.assertIsNone(refset.record_pointer("pilot", "", 5))


class GameLog(unittest.TestCase):
    def run_dir(self, d: str, ids: list, log: list) -> Path:
        run = Path(d)
        (run / "out").mkdir()
        (run / "scratch" / "s00").mkdir(parents=True)
        (run / "out" / "s00.jsonl").write_text("\n".join(json.dumps({"_id": f"{i:06d}"} if i is not None else {}) for i in ids) + "\n",
                                               encoding="utf-8")
        (run / "scratch" / "s00" / "game-stdout.log").write_text("\n".join(log), encoding="utf-8")
        return run

    LOG = ["[referee] server ready", "[referee] neow options [] choosing 0",  # noqa: RUF012
           "[ERROR] System.ObjectDisposedException: Cannot access a disposed object.", "Object name: 'Godot.Node2D'.",
           "   at Godot.GodotObject.GetPtr(GodotObject instance)", "   at MegaCrit.Sts2.Core.Nodes.Vfx.NSomeVfx.Play()",
           "[referee] neow options [] choosing 1",
           "[ERROR] System.ArgumentOutOfRangeException: Specified argument was out of the range of valid values.",
           "   at MegaCrit.Sts2.Core.Models.Cards.MadScience.OnPlay(PlayerChoiceContext choiceContext, CardPlay cardPlay)"]

    def test_errors_land_on_the_fight_that_logged_them(self) -> None:
        # serve.py numbers a pass's specs from 0 ("_id" 000000 is the pass's first spec): this server played the
        # pass's specs 0 and 2, and the Mad Science error is spec 2's own, not spec 1's.
        with tempfile.TemporaryDirectory() as d:
            errs = refset.log_errors(self.run_dir(d, [0, 2], self.LOG))
        self.assertEqual(sorted(errs), [0, 2])
        self.assertTrue(errs[0][0][2])   # spec 0: a visual fault only
        self.assertFalse(errs[2][0][2])  # spec 2: the game's own code threw
        self.assertIn("MadScience", errs[2][0][1])
        self.assertNotIn(1, errs)

    def test_the_top_bar_namespace_is_visual_too(self) -> None:
        log = ["[referee] neow options [] choosing 0", "[ERROR] System.ObjectDisposedException: Cannot access a disposed object.",
               "   at MegaCrit.sts2.Core.Nodes.TopBar.NTopBarGold.UpdateGoldAnim()"]
        with tempfile.TemporaryDirectory() as d:
            errs = refset.log_errors(self.run_dir(d, [0], log))
        self.assertIs(errs[0][0][2], True)

    def test_a_log_that_cannot_be_cut_is_not_a_game_error(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            errs = refset.log_errors(self.run_dir(d, [0, 1, 2], self.LOG))
        self.assertEqual(errs[1][0][2], "uncut")
        e = {"id": "X", "spec": {"seed": "S", "setup": []}}
        j = refset.judge_pair([e], {"X": (result(), errs[1])}, {"X": (result(), None)})["X"]
        self.assertEqual(refset.cause_of(j["excluded"]), "log_uncut")
        self.assertTrue(j["retry"])

    def test_a_fight_whose_game_code_threw_is_excluded_and_not_retried(self) -> None:
        e = {"id": "X", "spec": {"seed": "S", "setup": []}}
        bad = [["System.ArgumentOutOfRangeException", "MegaCrit.Sts2.Core.Models.Cards.MadScience.OnPlay", False]]
        j = refset.judge_pair([e], {"X": (result(err="idle_timeout after step 3"), bad)}, {"X": (result(), None)})["X"]
        self.assertIn("logged an error", j["excluded"])
        self.assertFalse(j.get("retry"))
        j = refset.judge_pair([e], {"X": (result(err="idle_timeout entering the room"), None)}, {"X": (result(), None)})["X"]
        self.assertTrue(j["retry"])
        vis = [["ObjectDisposedException", "MegaCrit.Sts2.Core.Nodes.Vfx.NSomeVfx.Play", True]]
        j = refset.judge_pair([e], {"X": (result(), vis)}, {"X": (result(), None)})["X"]
        self.assertIn("result", j)
        self.assertEqual(j["visual_errors"], ["MegaCrit.Sts2.Core.Nodes.Vfx.NSomeVfx.Play"])


def fake_dll(path: Path, commit: str, dirty: str, members=()) -> Path:
    """Bytes shaped like the DLL's metadata: the stamp attributes' blobs and a strings heap of member names."""
    def ser(s: str) -> bytes:
        b = s.encode()
        return bytes([len(b)]) + b if len(b) < 0x80 else bytes([0x80 | (len(b) >> 8), len(b) & 0xFF]) + b
    blob = b"MZ\x00" + b"\x01\x00" + ser("spireline.commit") + ser(commit) + b"\x00\x00" + b"\x01\x00" + ser("spireline.dirty")
    blob += (ser(dirty) if dirty else b"\x00") + b"\x00\x00\x00" + b"\x00".join(m.encode() for m in members) + b"\x00"
    path.write_bytes(blob)
    return path


class Driver(unittest.TestCase):
    ALL = [m for ms in refset.CAP_MEMBERS.values() for m in ms]  # noqa: RUF012

    def test_the_build_stamp_is_read_off_the_dll(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            dll = fake_dll(Path(d) / "x.dll", "a" * 40, "", self.ALL)
            self.assertEqual(refset.dll_stamp(dll), {"commit": "a" * 40, "dirty": ""})
            self.assertEqual(refset.static_caps(dll), list(refset.CAP_MEMBERS))
            dll = fake_dll(Path(d) / "y.dll", "b" * 40, " M referee/driver/Combat.cs" * 6, ["InPlayPhase"])
            self.assertTrue(refset.dll_stamp(dll)["dirty"].startswith(" M referee/driver/Combat.cs"))
            self.assertEqual(refset.static_caps(dll), ["play_phase_gate"])

    def test_a_server_line_without_the_fixes_refuses_the_record(self) -> None:
        good = {"commit": "c" * 40, "dirty": False, "capabilities": list(refset.REQUIRED_CAPS), "settle": "flags"}
        self.assertEqual(refset.driver_runtime_problems([good, good], "c" * 40), [])
        self.assertTrue(refset.driver_runtime_problems([], "c" * 40))
        no_gate = dict(good, capabilities=[c for c in refset.REQUIRED_CAPS if c != "play_phase_gate"])
        self.assertIn("lacks play_phase_gate", " ".join(refset.driver_runtime_problems([good, no_gate], "c" * 40)))
        self.assertTrue(refset.driver_runtime_problems([dict(good, dirty=True)], "c" * 40))
        self.assertTrue(refset.driver_runtime_problems([good], "d" * 40))
        self.assertTrue(refset.driver_runtime_problems([dict(good, settle="frames")], "c" * 40))

    def test_the_job_start_line_is_read_from_every_server(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for s in ("s00", "s01"):
                (Path(d) / "scratch" / s).mkdir(parents=True)
                (Path(d) / "scratch" / s / "game-stdout.log").write_text(
                    "[referee] x\n[referee] driver " + json.dumps({"commit": s}) + "\n", encoding="utf-8")
            self.assertEqual([x["commit"] for x in refset.driver_lines(Path(d))], ["s00", "s01"])


class Score(unittest.TestCase):
    def test_identical_needs_every_step_and_no_extra_one(self) -> None:
        nm = {}
        g = {"steps": [{"action": None, "state": {"in_combat": True, "hand": []}},
                       {"action": {"kind": "end_turn"}, "state": {"in_combat": True, "hand": []}}]}
        e = {"steps": [{"state": {"in_combat": True, "hand": []}}, {"state": {"in_combat": True, "hand": []}}]}
        orig = refset_score.fc.emu_view
        refset_score.fc.emu_view = lambda st, _nm: st
        orig_g = refset_score.fc.game_view
        refset_score.fc.game_view = lambda st: st
        try:
            self.assertIsNone(refset_score.first_divergence(g, e, nm))
            short = {"steps": e["steps"][:1]}
            self.assertEqual(refset_score.first_divergence(g, short, nm)[0], 1)
            longer = {"steps": e["steps"] + [e["steps"][0]]}
            self.assertEqual(refset_score.first_divergence(g, longer, nm)[0], 2)
        finally:
            refset_score.fc.emu_view, refset_score.fc.game_view = orig, orig_g

    def test_a_result_paired_with_another_seed_is_an_error(self) -> None:
        v = refset_score.judge("realistic", {"seed": "A", "steps": [], "refset": {"id": "X"}}, {"seed": "B", "steps": []}, {}, {}, None)
        self.assertFalse(v["identical"])
        self.assertIn("seed B", v["error"])

    def test_coverage_items_need_their_quota_and_no_divergence(self) -> None:
        v = [{"plays": [["ANGER", True]] * 5 + [["BASH+", True]] * 2, "uses": [["FIRE_POTION", True]] * 3,
              "primary": [[4, "INFLAME", [".player.powers"]]], "encounter": "NIBBITS_WEAK", "identical": True,
              "relics": ["VAJRA", "ANCHOR"], "fires": [["ANCHOR", True]] * 3}]
        v.append({"plays": [["INFLAME", True]] * 6, "uses": [], "primary": [], "encounter": "NIBBITS_WEAK", "identical": True,
                  "relics": ["VAJRA"], "fires": []})
        rep = refset_score.coverage_report(v, {"lists": {"cards": ["ANGER", "BASH+", "INFLAME"], "encounters": ["NIBBITS_WEAK"],
                                                         "relics": ["VAJRA", "ANCHOR"], "potions": ["FIRE_POTION"]}})
        self.assertEqual(rep["not_reached"]["card"], ["BASH+"])
        self.assertEqual(rep["not_clean"]["card"], ["INFLAME"])  # reached, but a play of it diverged
        self.assertEqual(rep["not_reached"]["encounter"], ["NIBBITS_WEAK"])  # 2 fights of 3
        self.assertEqual(rep["counts"]["potion"]["clean"], 1)
        # Held in two identical fights but never fired: not covered. Fired 3 times, verified: covered.
        self.assertEqual(rep["not_reached"]["relic"], ["VAJRA"])
        self.assertEqual(rep["counts"]["relic"]["clean"], 1)
        self.assertEqual(rep["counts"]["relic"]["held_never_fired"], 1)

    def test_a_regression_class_passes_only_with_its_trigger_reached(self) -> None:
        entries = {"g1": {"class_kind": "emulator", "trigger": {"plays": ["RUPTURE"]}},
                   "g2": {"class_kind": "emulator", "trigger": {"powers_hit": ["THORNS_POWER"]}},
                   "g3": {"class_kind": "driver", "trigger": {"replay": True}}}
        vs = [{"id": "g1", "part": "regression", "class": "rupture", "identical": True, "plays": [["RUPTURE", True]]},
              {"id": "g2", "part": "regression", "class": "thorns", "identical": True, "plays": [], "powers_hit": []},
              {"id": "g3", "part": "regression", "class": "driver: x", "identical": True}]
        rep = refset_score.regression_report(vs, entries)
        self.assertEqual(rep["emulator"]["passing"], 1)
        self.assertEqual(rep["emulator"]["identical_but_trigger_not_reached"], ["thorns"])
        self.assertEqual((rep["driver"]["classes"], rep["driver"]["passing"]), (1, 1))

    def test_an_emudump_build_older_than_its_sources_is_not_the_commits(self) -> None:
        import os  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as d:
            emu = Path(d)
            dll = emu / "tools/emudump/bin/Release/net9.0/EmuDump.dll"
            for f in ("src/Sts2Emulator/Core/Card.cs", "src/Sts2Emulator/obj/x.cs", "tools/emudump/Program.cs",
                      "src/Sts2Emulator.Tests/Support/Invariants/Inv.cs", "src/Sts2Emulator.Tests/Combats/T.cs"):
                (emu / f).parent.mkdir(parents=True, exist_ok=True)
                (emu / f).write_text("x", encoding="utf-8")
                os.utime(emu / f, (1000, 1000))
            dll.parent.mkdir(parents=True, exist_ok=True)
            dll.write_bytes(b"MZ")
            os.utime(dll, (2000, 2000))
            self.assertEqual(refset_score.stale_sources(emu, dll), [])
            for f in ("src/Sts2Emulator/Core/Card.cs", "src/Sts2Emulator/obj/x.cs", "src/Sts2Emulator.Tests/Combats/T.cs",
                      "src/Sts2Emulator.Tests/Support/Invariants/Inv.cs"):
                os.utime(emu / f, (3000, 3000))
            # Build output and tests EmuDump does not compile are not its sources.
            self.assertEqual(refset_score.stale_sources(emu, dll),
                             ["src/Sts2Emulator.Tests/Support/Invariants/Inv.cs", "src/Sts2Emulator/Core/Card.cs"])

    def test_a_codex_walk_is_reused_only_for_the_same_emulator_sources(self) -> None:
        import subprocess  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as d:
            emu = Path(d)

            def git(*a):
                return subprocess.run(["git", "-C", str(emu), *a], capture_output=True, text=True, check=True).stdout.strip()
            git("init", "-q")
            git("config", "user.email", "t@t")
            git("config", "user.name", "t")
            for sub, _ in refset_score.EMU_SOURCES:
                (emu / sub).mkdir(parents=True)
                (emu / sub / "a.cs").write_text("1", encoding="utf-8")
            git("add", "-A")
            git("commit", "-q", "-m", "one")
            c1 = git("rev-parse", "HEAD")
            (emu / "referee").mkdir()
            (emu / "referee" / "Driver.cs").write_text("driver only", encoding="utf-8")
            git("add", "-A")
            git("commit", "-q", "-m", "driver")
            c2 = git("rev-parse", "HEAD")
            (emu / "src/Sts2Emulator/a.cs").write_text("2", encoding="utf-8")
            git("add", "-A")
            git("commit", "-q", "-m", "engine")
            c3 = git("rev-parse", "HEAD")
            prev = emu / "prev.json"
            prev.write_text(json.dumps({"codex": {"runs": 10, "complete": 5, "commit": {"commit": c1, "dirty": False}}}), encoding="utf-8")
            self.assertEqual(refset_score.reuse_codex(prev, emu, c2)["reused_from"], {"score": "prev.json", "commit": c1})
            self.assertIn("skipped", refset_score.reuse_codex(prev, emu, c3))
            prev.write_text(json.dumps({"codex": {"runs": 10, "commit": {"commit": c1, "dirty": True}}}), encoding="utf-8")
            self.assertIn("skipped", refset_score.reuse_codex(prev, emu, c2))

    def test_the_headline_states_exclusions_flags_both_rates_and_the_builds(self) -> None:
        rows = [{"id": i, "part": "realistic", "flags": []} for i in "abcd"]
        rows[3]["excluded"] = "setup: item 0 player_load failed: deck 9 cards, asked 10"
        vs = [{"id": "a", "identical": True, "transitions": 4, "verified": 4},
              {"id": "b", "identical": False, "transitions": 4, "verified": 2, "flags": ["state_unknown:PEN_NIB.AttacksPlayed"]},
              {"id": "c", "identical": True, "transitions": 2, "verified": 2, "chooser": "heuristic"}]
        b = refset_score.block(rows, vs)
        self.assertEqual((b["specs"], b["kept"], b["excluded"], b["flagged"], b["identical"]), (4, 3, 1, 1, 2))
        self.assertEqual(b["excluded_by_cause"], {"setup": 1})
        self.assertEqual((b["identical_pct_of_kept"], b["identical_pct_of_specs"]), (66.7, 50.0))
        s = {"set": "Ironclad A10 reference set v2", "subset": "pilot", "commit_label": "abc123def456", "not_quotable": [],
             "record": {"driver_build": {"commit": "f" * 40, "capabilities": ["settle_flags", "play_phase_gate"], "settle": "flags"}},
             "headline": {"realistic": b}, "coverage": {"counts": {}},
             "regression": {"emulator": {"classes": 2, "passing": 1, "identical_but_trigger_not_reached": ["x"], "failing": []}},
             "codex": {}, "runs": {}, "open_bug_classes": []}
        line = refset_score.summary_line(s)
        for want in ("emulator abc123def456", "driver ffffffffffff", "settle_flags,play_phase_gate", "4 specs, 3 kept, 1 excluded (setup 1), 1 flagged",
                     "realistic [chooser heuristic 1, random 2]",
                     "2/3 = 66.7% of kept", "50.0% counting exclusions as failures"):
            self.assertIn(want, line)
        self.assertFalse(line.startswith("NOT QUOTABLE"))
        s["not_quotable"] = ["dirty emulator checkout"]
        self.assertTrue(refset_score.summary_line(s).startswith("NOT QUOTABLE"))


if __name__ == "__main__":
    unittest.main()
