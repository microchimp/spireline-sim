"""referee/refset.py and tools/fidelity/refset_score.py: the reference set's specs, references and numbers.

Synthetic run histories (the game's run-history shape) stand in for the codex data, and a small export in
the game's table shape for the real one. What must hold: a real run's state before a floor is rebuilt from the
history (cards added or upgraded later are undone, removed cards come back), coverage plans reach every item,
a fight whose two recordings differ or whose game side failed is never a reference, and a fight counts as
identical only when every recorded step matches.
"""

from __future__ import annotations

import collections
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT))

import refset  # noqa: E402
from tools.fidelity import refset_score  # noqa: E402


def card(cid, floor=1, up=0, ench=None, amount=1):
    c = {"id": "CARD." + cid, "floor_added_to_deck": floor}
    if up:
        c["current_upgrade_level"] = up
    if ench:
        c["enchantment"] = {"id": "ENCHANTMENT." + ench, "amount": amount}
    return c


def point(kind="monster", enc="NIBBITS_WEAK", hp=60, max_hp=80, **stats):
    room = {"model_id": "ENCOUNTER." + enc, "room_type": kind} if kind in ("monster", "elite", "boss") else {"model_id": "EVENT.X", "room_type": kind}
    return {"map_point_type": kind, "rooms": [room], "player_stats": [dict({"current_hp": hp, "max_hp": max_hp}, **stats)]}


def history():
    """Neow (floor 1), a fight at 2 (Anger picked), an upgrade of Bash at 3, a fight at 4, a removal of a Strike
    and an enchanted Defend at 5, a fight at 6 (Pommel Strike added, a potion used)."""
    pts = [
        point("ancient", ancient_choice=[{"TextKey": "A", "was_chosen": False}, {"TextKey": "NEOW_RELIC", "was_chosen": True}],
              relic_choices=[{"choice": "RELIC.NEOW_RELIC", "was_picked": True}]),
        point(potion_choices=[{"choice": "POTION.FIRE_POTION", "was_picked": True}],
              card_choices=[{"card": card("ANGER", 2), "was_picked": True}]),
        point("rest_site", upgraded_cards=["CARD.BASH"]),
        point("elite", "BYRDONIS_ELITE", hp=40, relic_choices=[{"choice": "RELIC.VAJRA", "was_picked": True}]),
        point("event", cards_removed=[card("STRIKE_IRONCLAD")],
              cards_enchanted=[{"card": card("DEFEND_IRONCLAD", ench="SHARP"), "enchantment": "ENCHANTMENT.SHARP"}]),
        point(hp=30, potion_used=["POTION.FIRE_POTION"], card_choices=[{"card": card("POMMEL_STRIKE", 6), "was_picked": True}]),
    ]
    deck = ([card("STRIKE_IRONCLAD")] * 4 + [card("DEFEND_IRONCLAD", ench="SHARP")] + [card("DEFEND_IRONCLAD")] * 3
            + [card("BASH", up=1), card("ANGER", 2), card("POMMEL_STRIKE", 6)])
    relics = [{"id": "RELIC.BURNING_BLOOD", "floor_added_to_deck": 1}, {"id": "RELIC.NEOW_RELIC", "floor_added_to_deck": 1},
              {"id": "RELIC.VAJRA", "floor_added_to_deck": 4}]
    return {"seed": "ABCDE12345", "run_hash": "r1", "acts": ["ACT.OVERGROWTH"], "map_point_history": [pts],
            "players": [{"deck": deck, "relics": relics, "potions": []}]}


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
        self.assertIn(["BASH", 0, "", 0, 1], deck)
        self.assertNotIn("ANGER", [c[0] for c in deck])

    def test_deck_after_the_event_keeps_its_changes(self) -> None:
        d = history()
        deck, _ = refset.deck_before(d, self.pts(d), 6)
        got = collections.Counter((c[0], c[2]) for c in deck)
        self.assertEqual(got[("STRIKE_IRONCLAD", "")], 4)
        self.assertEqual(got[("DEFEND_IRONCLAD", "SHARP")], 1)

    def test_relics_before_a_floor(self) -> None:
        d = history()
        self.assertEqual(refset.relics_before(d, self.pts(d), 4), ["BURNING_BLOOD", "NEOW_RELIC"])
        self.assertEqual(refset.relics_before(d, self.pts(d), 6), ["BURNING_BLOOD", "NEOW_RELIC", "VAJRA"])

    def test_a_removed_relic_comes_back_before_its_removal(self) -> None:
        d = history()
        self.pts(d)[4]["player_stats"][0]["relics_removed"] = ["RELIC.NEOW_RELIC"]
        d["players"][0]["relics"] = [r for r in d["players"][0]["relics"] if "NEOW" not in r["id"]]
        self.assertEqual(refset.relics_before(d, self.pts(d), 4), ["BURNING_BLOOD", "NEOW_RELIC"])
        self.assertEqual(refset.relics_before(d, self.pts(d), 6), ["BURNING_BLOOD", "VAJRA"])

    def test_potions_walk_forward_and_check_the_final_belt(self) -> None:
        d = history()
        belt, exact = refset.potions_before(self.pts(d), 4, d["players"][0]["potions"])
        self.assertEqual(belt, ["FIRE_POTION"])
        self.assertTrue(exact)
        _, exact = refset.potions_before(self.pts(d), 4, [{"id": "POTION.BLOCK_POTION"}])
        self.assertFalse(exact)


class FakeTables(refset.Tables):
    def __init__(self):  # noqa: D107 (no export: just the fields the spec makers read)
        self.follow = {r: "None" for r in ("BURNING_BLOOD", "NEOW_RELIC", "VAJRA", "ANCHOR", "POTION_BELT", "ORRERY")}
        self.follow["ORRERY"] = "PreRolledCardReward"
        self.cards_emu = {"STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH", "ANGER", "POMMEL_STRIKE", "ASCENDERS_BANE", "INFLAME"}
        self.potions_emu = {"FIRE_POTION", "BLOCK_POTION"}
        self.enchant_emu = {"SHARP"}
        self.encounters = {"NIBBITS_WEAK": ("OVERGROWTH", "weak", 1), "BYRDONIS_ELITE": ("OVERGROWTH", "elite", 1),
                           "KAISER_CRAB_BOSS": ("HIVE", "boss", 0)}
        self.card_info = {c: {"id": c, "rarity": "Common", "max_upgrade_level": "1"} for c in self.cards_emu}
        self.cards = sorted(self.cards_emu - {"ASCENDERS_BANE"})
        self.upgradable = set(self.cards)
        self.other_cards, self.other_relics, self.other_potions = set(), set(), set()
        self.pool_relics = {"VAJRA": "Common", "ANCHOR": "Common", "POTION_BELT": "Shop", "BURNING_BLOOD": "Starter", "ORRERY": "Shop"}
        self.event_relics = set()
        self.pool_potions = {"FIRE_POTION", "BLOCK_POTION"}


class Specs(unittest.TestCase):
    def test_realistic_candidates_set_up_the_floor_state(self) -> None:
        stats = collections.Counter()
        got = list(refset.realistic_candidates(history(), FakeTables(), stats))
        floors = {c["floor"]: c for _, _, _, c in got}
        self.assertEqual(sorted(floors), [2, 4, 6])
        c = floors[4]
        self.assertEqual((c["room"], c["encounter"], c["hp"], c["neow"], c["neow_relic"]), ("elite", "BYRDONIS_ELITE", 60, 1, "NEOW_RELIC"))
        entry = refset.realistic_entry(c, FakeTables())
        cmds = [it["cmd"] for it in entry["spec"]["setup"]]
        self.assertEqual(cmds, ["deck_set", "relics_set", "deck_set", "hp", "potions_set", "fight"])
        self.assertEqual(entry["spec"]["seed"], "ABCDE12345")
        self.assertEqual(entry["spec"]["neow"], 1)

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
            self.assertEqual(len(set(items[1]["ids"])), len(items[1]["ids"]))  # no duplicate relic


def result(seed="S", steps=3, err=None, setup_ok=True):
    r = {"seed": seed, "steps": [{"action": None if i == 0 else {"kind": "end_turn"}, "state": {"in_combat": True, "round": i}}
                                 for i in range(steps)], "fight": {"id": "NIBBITS_WEAK"},
         "setup_log": [{"i": -1, "cmd": "before", "ok": True}, {"i": 0, "cmd": "relics_set", "ok": setup_ok, "msg": ""}],
         "phase_ms": {"x": 1}}
    if err:
        r["error"] = err
    return r


class Recording(unittest.TestCase):
    entry = {"id": "X", "spec": {"seed": "S", "setup": [{"cmd": "relics_set"}, {"cmd": "fight", "id": "NIBBITS_WEAK"}]}}  # noqa: RUF012

    def test_a_clean_recording_is_kept(self) -> None:
        self.assertIsNone(refset.exclusion(self.entry, result()))

    def test_game_errors_and_refused_setup_are_excluded(self) -> None:
        self.assertIn("game error", refset.exclusion(self.entry, result(err="idle_timeout after step 3")))
        self.assertIn("refused", refset.exclusion(self.entry, result(setup_ok=False)))
        self.assertIn("no result", refset.exclusion(self.entry, None))
        r = result()
        del r["fight"]
        self.assertIn("not entered", refset.exclusion(self.entry, r))

    def test_timing_does_not_make_a_fight_unstable_but_a_state_does(self) -> None:
        a, b = result(), result()
        b["phase_ms"] = {"x": 999}
        self.assertEqual(refset.comparable(a), refset.comparable(b))
        b["steps"][2]["state"]["round"] = 7
        self.assertNotEqual(refset.comparable(a), refset.comparable(b))
        self.assertTrue(refset.first_difference(a, b).startswith("step 2"))


class GameLog(unittest.TestCase):
    def test_errors_are_cut_per_fight_and_visual_ones_kept(self) -> None:
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "out").mkdir()
            (run / "scratch" / "s00").mkdir(parents=True)
            (run / "out" / "s00.jsonl").write_text("\n".join(json.dumps({"_id": f"{i:06d}"}) for i in (1, 3)) + "\n", encoding="utf-8")
            (run / "scratch" / "s00" / "game-stdout.log").write_text("\n".join([
                "[referee] server ready", "[referee] neow options [] choosing 0",
                "[ERROR] System.ObjectDisposedException: Cannot access a disposed object.", "Object name: 'Godot.Node2D'.",
                "   at Godot.GodotObject.GetPtr(GodotObject instance)", "   at MegaCrit.Sts2.Core.Nodes.Vfx.NSomeVfx.Play()",
                "[referee] neow options [] choosing 1",
                "[ERROR] System.ArgumentOutOfRangeException: Specified argument was out of the range of valid values.",
                "   at MegaCrit.Sts2.Core.Models.Cards.MadScience.OnPlay(PlayerChoiceContext choiceContext, CardPlay cardPlay)"]),
                encoding="utf-8")
            errs = refset.log_errors(run)
        self.assertTrue(errs[0][0][2])   # first fight: a visual fault only
        self.assertFalse(errs[2][0][2])  # third spec of the pass: the game's own code threw
        self.assertIn("MadScience", errs[2][0][1])

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

    def test_coverage_items_need_their_quota_and_no_divergence(self) -> None:
        v = [{"plays": [["ANGER", True]] * 5 + [["BASH+", True]] * 2, "uses": [["FIRE_POTION", True]] * 3,
              "primary": [[4, "INFLAME", [".player.powers"]]], "encounter": "NIBBITS_WEAK", "identical": True,
              "relics": ["VAJRA"]}]
        v.append({"plays": [["INFLAME", True]] * 6, "uses": [], "primary": [], "encounter": "NIBBITS_WEAK", "identical": True, "relics": []})
        rep = refset_score.coverage_report(v, {"lists": {"cards": ["ANGER", "BASH+", "INFLAME"], "encounters": ["NIBBITS_WEAK"],
                                                         "relics": ["VAJRA"], "potions": ["FIRE_POTION"]}})
        self.assertEqual(rep["not_reached"]["card"], ["BASH+"])
        self.assertEqual(rep["not_clean"]["card"], ["INFLAME"])  # reached, but a play of it diverged
        self.assertEqual(rep["not_reached"]["encounter"], ["NIBBITS_WEAK"])  # 2 fights of 3
        self.assertEqual(rep["counts"]["potion"]["clean"], 1)


if __name__ == "__main__":
    unittest.main()
