"""referee/probe.py and referee/probes.py: the rule probes and their judge.

The judge must never call a probe AGREE without a value from both sides, must say "not reached" when
the game never got to the measured step or the scenario's precondition failed, and must name the rule
each side follows from the hypotheses' predictions. The catalog must stay inside what the referee's
setup commands and the script runner accept.
"""

# ruff: noqa: ARG005  (every hypothesis and measure takes the same (pre, ctx) / (pre, post, ctx) arguments)
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT))

import probe  # noqa: E402
import probes  # noqa: E402

NM = {"cards": {1: "STRIKE_IRONCLAD", 2: "DEFEND_IRONCLAD", 3: "SECOND_WIND"}, "enemies": {7: "NIBBIT"}}


def game_state(hp=64, block=0, energy=3, powers=(), hand=("STRIKE_IRONCLAD",), enemy_hp=48, enemy_powers=(),
               intents=(("Attack", 13, 1),), in_combat=True, potions=None):
    s = {"in_combat": in_combat, "round": 1,
         "player": {"hp": hp, "max_hp": 80, "block": block, "energy": energy, "powers": [list(p) for p in powers]},
         "hand": [[c, False] for c in hand], "draw": [["DEFEND_IRONCLAD", False]], "discard": [], "exhaust": [],
         "enemies": [{"name": "NIBBIT", "hp": enemy_hp, "max_hp": 48, "block": 0, "powers": [list(p) for p in enemy_powers],
                      "move": "M", "intents": [list(i) for i in intents]}] if in_combat else [],
         "rng": {"Shuffle": {"seed": 1, "counter": 0}}}
    if potions is not None:
        s["potions"] = potions
    return s


def emu_state(hp=64, block=0, energy=3, buffs=(), hand=(1,), enemy_hp=48, in_combat=True):
    if not in_combat:
        return {"in_combat": False, "run_phase": "Over"}
    return {"in_combat": True, "run_phase": "Combat",
            "snapshot": {"player": {"hp": hp, "max_hp": 80, "block": block, "energy": energy,
                                    "buffs": [{"id": 0, "name": n, "amount": a, "data": 0} for n, a in buffs]},
                         "piles": {"hand": [{"def_id": c, "upgraded": False} for c in hand], "discard": [], "exhaust": []},
                         "enemies": [{"def_id": 7, "hp": enemy_hp, "max_hp": 48, "block": 0, "buffs": [],
                                      "intent": {"type": 0, "magnitude": 13, "hits": 1}}]},
            "draw_order": [[2, False]], "rng": {"Shuffle": 0}}


def toy_probe(**kw) -> probes.Probe:
    """Strike with Strength: does Strength add to the hit?"""
    args = dict(
        id="T01", ledger=["C1-F99"], question="Does Strength add to a Strike?",
        setup=[probes.fight("NIBBITS_WEAK"), probes.power("STRENGTH_POWER", 3)],
        script=[probes.play("STRIKE_IRONCLAD", 0)],
        measure=probes.dealt(0),
        hyps=[probes.Hyp("adds", "Strength adds", lambda pre, ctx: 6 + pre.power("STRENGTH")),
              probes.Hyp("ignored", "Strength ignored", lambda pre, ctx: 6)],
        need=lambda pre, ctx: None if "STRIKE_IRONCLAD" in pre.hand else "no Strike")
    args.update(kw)
    return probes.Probe(**args)


def fight(pre, post):
    return {"seed": "S", "steps": [{"action": None, "state": pre},
                                   {"action": {"kind": "play", "card": "STRIKE_IRONCLAD", "hand_index": 0}, "state": post}]}


def anchored(post_emu):
    return {"steps": [{"state": emu_state()}, {"state": post_emu}]}


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.ctx = probe.Ctx({"STRIKE_IRONCLAD": "Attack", "DEFEND_IRONCLAD": "Skill", "SECOND_WIND": "Skill"})
        self.pre = game_state(powers=[("STRENGTH_POWER", 3)])

    def judge(self, g, r, e=None, p=None):
        return probe.judge_one(p or toy_probe(), g, r, e, NM, self.ctx)

    def test_agree_names_the_same_rule_on_both_sides(self):
        g = fight(self.pre, game_state(powers=[("STRENGTH_POWER", 3)], hand=(), enemy_hp=39, energy=2))
        v = self.judge(g, anchored(emu_state(buffs=[("Strength", 3)], hand=(), enemy_hp=39, energy=2)))
        self.assertEqual(v["status"], "AGREE")
        self.assertEqual((v["game"], v["emulator"]), (9, 9))
        self.assertEqual(v["game_rule"], ["adds"])
        self.assertEqual(v["emulator_rule"], ["adds"])
        self.assertEqual(v["answer"], "Strength adds")

    def test_differ_names_each_sides_rule(self):
        g = fight(self.pre, game_state(hand=(), enemy_hp=39))
        v = self.judge(g, anchored(emu_state(hand=(), enemy_hp=42)))
        self.assertEqual(v["status"], "DIFFER")
        self.assertEqual(v["game_rule"], ["adds"])
        self.assertEqual(v["emulator_rule"], ["ignored"])

    def test_no_emulator_state_is_never_agree(self):
        # Deliberately broken case: the emulator's fight ended (EmuDump dumps no state), or re-anchoring
        # stopped before the step. Both used to look like "nothing differs".
        g = fight(self.pre, game_state(hand=(), enemy_hp=39))
        for r in (anchored(emu_state(in_combat=False)), {"steps": [{"state": emu_state()}]},
                  {"steps": [{"state": emu_state()}, {"anchor_error": "cannot rebuild"}]}, None):
            v = self.judge(g, r)
            self.assertEqual(v["status"], "emulator: no value", r)
            self.assertIsNone(v["emulator"])
            self.assertEqual(v["emulator_rule"], [])

    def test_a_fight_that_ends_is_judged_from_the_plain_replay(self):
        p = toy_probe(measure=lambda pre, post, ctx: post.outcome, ends=True,
                      hyps=[probes.Hyp("won", "won", lambda pre, ctx: "won"), probes.Hyp("lost", "lost", lambda pre, ctx: "lost")])
        g = fight(self.pre, game_state(hp=0, in_combat=False))
        plain = {"steps": [{"state": emu_state(buffs=[("Strength", 3)])}, {"in_combat": False, "state": {"in_combat": False, "run_phase": "Complete"}}]}
        v = self.judge(g, anchored(emu_state(in_combat=False)), plain, p=p)
        self.assertEqual((v["game"], v["emulator"], v["emulator_source"], v["status"]), ("lost", "lost", "plain replay", "AGREE"))
        self.assertEqual(v["reanchor_note"], "no measurement from the emulator's state")  # the dump said "Over"
        # Without the plain replay nothing says how the emulator's fight ended: no verdict.
        v = self.judge(g, anchored(emu_state(in_combat=False)), None, p=p)
        self.assertEqual(v["status"], "emulator: no value")

    def test_not_reached_when_the_game_stopped_early(self):
        g = {"seed": "S", "steps": [{"action": None, "state": self.pre}],
             "script_illegal": {"step": 1, "why": "STRIKE_IRONCLAD not playable from hand"}}
        v = self.judge(g, anchored(emu_state()))
        self.assertEqual(v["status"], "not reached")
        self.assertIn("not playable", v["why"])
        self.assertNotIn("game", v)

    def test_not_reached_when_the_precondition_fails(self):
        g = fight(game_state(hand=("DEFEND_IRONCLAD",)), game_state(hand=()))
        v = self.judge(g, anchored(emu_state(hand=())))
        self.assertEqual(v["status"], "not reached")
        self.assertEqual(v["why"], "no Strike")

    def test_game_error_and_probe_error(self):
        self.assertEqual(self.judge({"error": "idle_timeout", "steps": []}, None)["status"], "game error")
        broken = toy_probe(measure=lambda pre, post, ctx: 1 / 0)
        g = fight(self.pre, game_state(hand=(), enemy_hp=39))
        v = self.judge(g, anchored(emu_state(hand=(), enemy_hp=39)), p=broken)
        self.assertEqual(v["status"], "probe error")
        self.assertIn("ZeroDivisionError", v["why"])

    def test_outcome_no_hypothesis_predicts(self):
        g = fight(self.pre, game_state(hand=(), enemy_hp=30))
        v = self.judge(g, anchored(emu_state(hand=(), enemy_hp=30)))
        self.assertEqual(v["status"], "AGREE")
        self.assertEqual(v["game_rule"], ["none of the hypotheses"])
        self.assertIsNone(v["answer"])

    def test_plain_replay_used_only_while_it_matched(self):
        g = fight(self.pre, game_state(hand=(), enemy_hp=39))
        same = {"steps": [{"state": emu_state(buffs=[("Strength", 3)])}, {"state": emu_state(hand=(), enemy_hp=39)}]}
        other = {"steps": [{"state": emu_state(hp=50)}, {"state": emu_state(hand=(), enemy_hp=39)}]}
        self.assertEqual(self.judge(g, anchored(emu_state(hand=(), enemy_hp=39)), same)["plain"], 9)
        self.assertTrue(self.judge(g, anchored(emu_state(hand=(), enemy_hp=39)), other)["plain"].startswith("n/a"))

    def test_measured_step_from_a_function(self):
        p = toy_probe(at=lambda steps: None)
        v = self.judge(fight(self.pre, game_state()), anchored(emu_state()), p=p)
        self.assertEqual(v["status"], "not reached")

    def test_judge_finds_each_probe_by_tag(self):
        a, b = toy_probe(id="T01"), toy_probe(id="T02")
        g1 = {**fight(self.pre, game_state(hand=(), enemy_hp=39)), "tag": "probe:T02"}
        g2 = {"tag": "probe:T01", "error": "crash", "steps": []}
        out = probe.judge([a, b], [g1, g2], [anchored(emu_state(hand=(), enemy_hp=39)), None], [None, None], NM, self.ctx)
        self.assertEqual([v["id"] for v in out], ["T01", "T02"])
        self.assertEqual(out[0]["status"], "game error")
        self.assertEqual(out[1]["status"], "AGREE")


class ViewTests(unittest.TestCase):
    def test_game_view_keeps_the_player_after_the_fight(self):
        v = probe.V.game(game_state(hp=20, in_combat=False))
        self.assertFalse(v.in_combat)
        self.assertEqual(v.hp, 20)
        self.assertEqual(v.enemies, [])

    def test_powers_by_name_and_temporary_strength_folded(self):
        v = probe.V.game(game_state(powers=[("STRENGTH_POWER", 5), ("FLEX_POTION_POWER", 5)]))
        self.assertEqual(v.power("STRENGTH"), 5)
        self.assertEqual(v.power("TEMPORARY_STRENGTH"), 5)
        self.assertEqual(v.power("DEXTERITY"), 0)

    def test_emulator_view_after_the_fight_has_only_its_outcome(self):
        self.assertIsNone(probe.V.emu(None, NM))
        over = probe.V.emu(emu_state(in_combat=False), NM)  # the re-anchored dump: "Over" says neither
        self.assertFalse(over.in_combat)
        self.assertIsNone(over.outcome)
        self.assertIsNone(over.hp)
        with self.assertRaises(ValueError):
            over.power("STRENGTH")
        self.assertEqual(probe.V.emu({"in_combat": False, "run_phase": "Complete"}, NM).outcome, "lost")
        self.assertEqual(probe.V.emu({"in_combat": False, "run_phase": "CardReward"}, NM).outcome, "won")
        self.assertEqual(probe.V.game(game_state(hp=0, in_combat=False)).outcome, "lost")
        self.assertEqual(probe.V.game(game_state(hp=9, in_combat=False)).outcome, "won")
        self.assertEqual(probe.V.game(game_state()).outcome, "going")
        e = probe.V.emu(emu_state(buffs=[("Strength", 2)], hand=(1, 3)), NM)
        self.assertEqual(e.hand, ["STRIKE_IRONCLAD", "SECOND_WIND"])
        self.assertEqual(e.power("STRENGTH"), 2)
        self.assertEqual(e.draw, ["DEFEND_IRONCLAD"])

    def test_ctx_helpers(self):
        ctx = probe.Ctx({"STRIKE_IRONCLAD": "Attack", "DEFEND_IRONCLAD": "Skill", "SECOND_WIND": "Skill"})
        self.assertEqual(ctx.non_attacks(["SECOND_WIND", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD"], "SECOND_WIND"), 2)
        pre = probe.V.game(game_state(enemy_powers=[("STRENGTH_POWER", 2)]))
        self.assertEqual(ctx.hit_after_block(pre, 5), 10)
        self.assertEqual(ctx.hit_after_block(pre, 20), 0)
        self.assertIsNone(ctx.hit_after_block(probe.V.game(game_state(intents=(("Buff",),))), 5))


class CatalogTests(unittest.TestCase):
    MAP_ONLY = {"relic", "potion", "deck_set"}
    COMBAT_ONLY = {"power", "energy", "upgrade"}
    KINDS = {"relic", "potion", "card", "fight", "power", "energy", "upgrade", "deck_set"}

    def test_at_least_ten_probes_with_unique_ids_and_ledger_items(self):
        ids = [p.id for p in probes.PROBES]
        self.assertGreaterEqual(len(ids), 10)
        self.assertEqual(len(ids), len(set(ids)))
        for p in probes.PROBES:
            self.assertTrue(p.ledger and all(re.fullmatch(r"C\d-[A-Z]\d{2}", x) for x in p.ledger), p.id)
            self.assertTrue(p.question.endswith("?"), p.id)
            self.assertTrue(p.game_class, p.id)
            names = [h.name for h in p.hyps]
            self.assertEqual(len(names), len(set(names)), p.id)
            self.assertTrue(p.hyps or p.evidence, f"{p.id}: no hypothesis and no trace evidence")

    def test_setup_order_is_what_the_referee_runs(self):
        for p in probes.PROBES:
            cmds = [it["cmd"] for it in p.setup]
            self.assertEqual(cmds.count("fight"), 1, p.id)
            f = cmds.index("fight")
            self.assertTrue(set(cmds) <= self.KINDS, p.id)
            self.assertFalse(self.MAP_ONLY & set(cmds[f + 1:]), f"{p.id}: relic/potion after the fight")
            self.assertFalse(self.COMBAT_ONLY & set(cmds[:f]), f"{p.id}: combat item before the fight")
            for it in p.setup[f + 1:]:
                if it["cmd"] == "card":
                    self.assertIn(it.get("pile"), ("Hand", "Draw", "Discard"), p.id)
            hand = sum(1 for it in p.setup if it["cmd"] == "card" and it.get("pile") == "Hand")
            self.assertLessEqual(5 + hand, 10, f"{p.id}: more cards than a hand holds")

    def test_script_and_spec(self):
        for p in probes.PROBES:
            self.assertTrue(p.script, p.id)
            for act in p.script:
                self.assertIn(act["kind"], ("play", "end_turn", "potion"), p.id)
                if act["kind"] == "play":
                    self.assertTrue(act["card"].isupper(), p.id)
            if not callable(p.at):
                self.assertTrue(1 <= p.step([]) <= len(p.script), p.id)
            s = p.spec()
            self.assertEqual((s["kind"], s["character"], s["ascension"]), ("combat", "IRONCLAD", 10))
            self.assertTrue(s["trace"] and s["use_potions"])
            self.assertEqual(s["script"], p.script)
            self.assertEqual(s["tag"], "probe:" + p.id)

    def test_pen_nib_hypotheses_tell_the_orders_apart(self):
        p = probes.BY_ID["P01"]
        pre = probe.V.game(game_state(powers=[("STRENGTH_POWER", 3)], hand=("POMMEL_STRIKE",)))
        preds = {h.name: h.predict(pre, None) for h in p.hyps}
        self.assertEqual(preds, {"add-then-double": 24, "double-then-add": 21, "not-doubled": 12})

    KITS = {"C5-F01": ["P21"], "C5-F03": ["P22", "P23", "P24"], "C5-F07": ["P25"], "C5-F09": ["P26"], "C5-H03": ["P27", "P28"]}

    def test_each_kit_without_a_probe_has_catalog_probes_naming_it(self):
        for kit, ids in self.KITS.items():
            self.assertEqual([p.id for p in probes.PROBES if kit in p.ledger], ids, kit)
            for i in ids:
                self.assertTrue(probes.BY_ID[i].need, i)  # a scenario the game did not reach is "not reached", never a verdict

    def test_deck_set_probes_list_the_deck_in_the_refset_shape(self):
        for p in probes.PROBES:
            for it in p.setup:
                if it["cmd"] == "deck_set":
                    self.assertTrue(it["cards"], p.id)
                    for c in it["cards"]:
                        self.assertEqual(set(c), {"id", "upgraded", "enchantment", "amount"}, p.id)
                        self.assertTrue(c["id"].isupper() and c["enchantment"] == c["enchantment"].upper(), p.id)
                        self.assertEqual(c["amount"] > 0, c["enchantment"] != "", p.id)

    def test_new_probe_hypotheses_tell_the_rules_apart(self):
        def preds(pid, **kw):
            pre = probe.V.game(game_state(**kw))
            return {h.name: h.predict(pre, None) for h in probes.BY_ID[pid].hyps}
        self.assertEqual(preds("P21"), {"per-repetition": 6, "once": 3})
        self.assertEqual(preds("P22", powers=[("STRENGTH_POWER", 2)]), {"own-damage-var": 11, "enchant-amount": 9})
        self.assertEqual(preds("P23"), {"sharp-applies": 23, "sharp-skipped": 20})
        self.assertEqual(preds("P24", hand=("SECOND_WIND",) + ("DEFEND_IRONCLAD",) * 4), {"nimble-each": 28, "nimble-skipped": 20})
        self.assertEqual(preds("P25"), {"elite": (3, 7), "not-elite": (2, 5)})
        for pid in ("P26", "P27", "P28"):
            vals = [str(v) for v in preds(pid).values()]
            self.assertEqual(len(vals), len(set(vals)), pid)

    def test_bellows_measure_reads_the_upgrade_flags_of_the_hand(self):
        p = probes.BY_ID["P26"]
        pre = probe.V.game(game_state(hand=("ASCENDERS_BANE", "INJURY", "DEFEND_IRONCLAD")))
        self.assertIsNone(p.need(pre, None))
        raw = game_state(hand=("ASCENDERS_BANE", "INJURY"))
        raw["hand"] = [["ASCENDERS_BANE", True], ["INJURY", False]]
        self.assertEqual(p.measure(pre, probe.V.game(raw), None), [("ASCENDERS_BANE", True), ("INJURY", False)])
        self.assertEqual(probe.V.game(raw).hand_up, [True, False])
        self.assertEqual(probe.V.emu(emu_state(in_combat=False), NM).hand_up, [])

    def test_strike_places_counts_every_pile(self):
        raw = game_state(hand=("DEFEND_IRONCLAD",))
        raw["discard"] = [["STRIKE_IRONCLAD", False]]
        post = probe.V.game(raw)
        self.assertEqual(probes._strike_places(None, post, None), [1, 0, 0, 0])
        self.assertIsNone(probes._strike_places(None, None, None))

    def test_shuffle_step_finder(self):
        st = lambda n: {"rng": {"Shuffle": {"counter": n}}}  # noqa: E731
        self.assertEqual(probes.first_shuffle_step([{"state": st(0)}, {"state": st(0)}, {"state": st(3)}]), 2)
        self.assertIsNone(probes.first_shuffle_step([{"state": st(0)}, {"state": st(0)}]))


if __name__ == "__main__":
    unittest.main()
