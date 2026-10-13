"""tools.fidelity.trace_diff: the first segment of a step whose changes differ, and the first draw one side lacks.

Synthetic traces in the shapes referee/driver/Trace.cs and tools/emudump/Trace.cs write.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import fight_compare as fc  # noqa: E402
from tools.fidelity import trace_diff as td  # noqa: E402

ID_MAP = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))
NM = fc.names(ID_MAP)
CID, EID = ID_MAP["cards"], ID_MAP["enemies"]
STRIKE, SLUG = "STRIKE_IRONCLAD", "CORPSE_SLUG"


def game_state(hp):
    return {"in_combat": True, "round": 1, "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 2, "powers": []},
            "hand": [], "draw": [], "discard": [[STRIKE, False]], "exhaust": [],
            "enemies": [{"name": SLUG, "hp": hp, "max_hp": 27, "block": 0, "powers": []}],
            "rng": {"Shuffle": {"seed": 1, "counter": 1}}}


def emu_state(hp):
    return {"in_combat": True, "draw_order": [], "rng": {"Shuffle": 1},
            "snapshot": {"player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 2, "buffs": []},
                         "piles": {"hand": [], "discard": [{"def_id": CID["StrikeIronclad"], "upgraded": False}], "exhaust": []},
                         "enemies": [{"def_id": EID["CorpseSlug"], "hp": hp, "block": 0, "buffs": []}]}}


def mini(hp, energy, hand, discard):
    return {"player": [70, 0, energy], "powers": [], "enemies": [[SLUG, hp, 0, []]], "hand": hand, "n": [0, discard, 0]}


PLAY = {"kind": "play", "hand_index": 0, "card": STRIKE, "target": 0, "target_slot": 0}


def fights(emu_hp, emu_draws):
    start = dict(game_state(27), hand=[[STRIKE, False]], discard=[], player=dict(game_state(27)["player"], energy=3))
    g = {"seed": "T", "steps": [
        {"action": None, "state": start, "hidden": {"cards": [["hand", STRIKE, 1, None, 0, None]], "relics": [], "powers": []}},
        {"action": PLAY, "state": game_state(21), "trace": [
            ["E", "EnergySpent", 1],
            ["M", "play", STRIKE, mini(27, 2, [], 0)],
            ["R", "Shuffle", 1, "A.B"], ["R", "Shuffle", 1, "A.B"],  # one draw, logged by two nested Next* calls
            ["E", "DamageReceived", "CORPSE_SLUG#0", "player", 6, 0, STRIKE],
            ["M", "played", STRIKE, mini(21, 2, [], 0)]]}]}
    e = {"seed": "T", "steps": [
        {"action": None, "state": emu_state(27)},
        {"action": PLAY, "state": emu_state(emu_hp), "hidden_pre": {"cards": [["hand", STRIKE, 1, None, 0]], "relics": [], "powers": []},
         "trace": [["M", "start", None, mini(27, 3, [STRIKE], 0)], ["M", "play", STRIKE, mini(27, 2, [], 0)]]
         + [["R", "Shuffle", c, "CardEffects.X"] for c in emu_draws]
         + [["M", "played", STRIKE, mini(emu_hp, 2, [], 0)]]}]}
    return g, e


class TraceDiffTest(unittest.TestCase):
    def test_names_the_segment_and_the_differing_change(self):
        g, e = fights(18, [1])
        d = td.step_diff(g, e, 1, NM)
        self.assertEqual(d["segment"]["from"], f"play {STRIKE}")
        self.assertEqual(d["segment"]["to"], f"played {STRIKE}")
        self.assertIn("game CORPSE_SLUG#0 hp -6; emulator CORPSE_SLUG#0 hp -9", d["first_difference"])
        self.assertIn("log DamageReceived CORPSE_SLUG#0 player 6 0 " + STRIKE, d["segment"]["game_log"])
        self.assertIsNone(d["draw_difference"])  # the nested call's second log is the same draw

    def test_a_draw_only_the_emulator_makes(self):
        g, e = fights(21, [1, 2])
        d = td.step_diff(g, e, 1, NM)
        self.assertNotIn("first_difference", d)
        dd = d["draw_difference"]
        self.assertEqual((dd["stream"], dd["after_same"], dd["game_total"], dd["emu_total"]), ("Shuffle", 1, 1, 2))
        self.assertIn("FIRST DRAW DIFFERENCE on Shuffle", td.text(d))

    def test_hidden_cost_difference_at_step_start(self):
        g, e = fights(21, [1])
        e["steps"][1]["hidden_pre"]["cards"] = [["hand", STRIKE, 0, None, 0]]
        d = td.step_diff(g, e, 1, NM)
        self.assertEqual(d["hidden"], [f"hand {STRIKE}: game cost 1 -, emulator cost 0 -"])

    def test_an_end_of_turn_recorded_before_the_turn_start_finished(self):
        end = {"action": {"kind": "end_turn"}, "state": game_state(21)}
        early = dict(end, trace=[["H", "BeforeSideTurnStart", "Player"], ["H", "AfterPlayerTurnStart"]])
        self.assertIn("had not finished", td.unsettled(early, {"trace": []}))
        mid = dict(end, trace=[["H", "AfterSideTurnStart", "Player"], ["H", "AfterAutoPrePlayPhaseEntered"], ["R", "Shuffle", 4, "x"]])
        self.assertIn("middle of its auto pre-play phase", td.unsettled(mid, {"trace": []}))
        emu_auto = {"trace": [["M", "turn_start", None, {}], ["M", "play", STRIKE, {}]]}
        begun = dict(end, trace=[["H", "AfterSideTurnStart", "Player"], ["H", "AfterAutoPrePlayPhaseEntered"]])
        self.assertIn("auto-plays " + STRIKE, td.unsettled(begun, emu_auto))
        self.assertIsNone(td.unsettled(begun, {"trace": []}))

    def test_compress_runs(self):
        self.assertEqual(td.compress(["hook A", "hook A", "draw S#1 by f", "draw S#2 by f", "draw S#3 by g"]),
                         ["hook A x2", "draw S#1..2 by f", "draw S#3 by g"])


if __name__ == "__main__":
    unittest.main()
