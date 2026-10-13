"""tools.fidelity.netstate_diff: per-checksum comparison of the game's full combat state.

Synthetic fights in the shapes ChecksumTap ("trace": true) and `EmuDump fight --netstate` write. A
divergence is reported at the first step whose final state differs on a field both sides have,
blamed on the game checkpoint (sub-action) where the game's value left the emulator's; fields only
the game has are listed, not compared; a step whose action left no checksum ends the comparison
as a driver artifact; and two game runs are compared checksum by checksum.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from itertools import starmap

from tools.fidelity import netstate_diff as nd  # noqa: E402

PLAY = {"kind": "play", "hand_index": 0, "card": "STRIKE_IRONCLAD", "target": 0, "target_slot": 0, "played": True}
END = {"kind": "end_turn"}
PHASES = ["After player turn phase one end", "after player turn phase two end", "After enemy turn start",
          "After enemy turn end", "After player turn start"]


def net(php=70, block=0, ehp=27, hand=("STRIKE_IRONCLAD",), ai=0, game=True, powers=()):
    """One NetFullCombatState-shaped state; game=True adds the fields only the game writes."""
    card = {"id": None, "upgrades": 0}
    st = {
        "creatures": [
            {"monster": None, "player": 1, "hp": php, "max_hp": 80, "block": block,
             "powers": [list(p) for p in powers]},
            {"monster": "CORPSE_SLUG", "player": None, "hp": ehp, "max_hp": 27, "block": 0,
             "powers": [["RAVENOUS_POWER", 5]]},
        ],
        "players": [{"id": 1, "character": "IRONCLAD", "turn": 1, "energy": 3, "stars": 0, "max_potions": 2,
                     "gold": 99, "potions": [], "relics": [{"id": "BURNING_BLOOD"}], "orbs": [],
                     "piles": {p: [] for p in nd.PILES}, "rng": {"counters": {"Transformations": 0}},
                     "grab_bag": {"Common": ["ANCHOR"]}}],
        "rng": {"counters": {"Shuffle": 3, "MonsterAi": ai}},
    }
    st["players"][0]["piles"]["Hand"] = [dict(card, id=c) for c in hand]
    if game:
        st.update(choice_ids=[], reward_ids=[0], last_action=None, last_hook=None)
        st["players"][0]["phase"] = "Play"
        st["players"][0]["relics"][0]["floor"] = 1
    else:
        st["players"][0]["relics"][0]["counter"] = 0
    return st


def snap(st):
    """The driver's idle snapshot (Combat.cs) of a game state, so the overlay finds nothing new."""
    p = next(c for c in st["creatures"] if c["player"] is not None)
    piles = st["players"][0]["piles"]
    return {"in_combat": True, "round": 1,
            "player": {"hp": p["hp"], "max_hp": p["max_hp"], "block": p["block"], "energy": st["players"][0]["energy"],
                       "powers": p["powers"]},
            "hand": [[c["id"], False] for c in piles["Hand"]], "draw": [], "discard": [], "exhaust": [],
            "enemies": [{"name": c["monster"], "hp": c["hp"], "max_hp": c["max_hp"], "block": c["block"],
                         "powers": c["powers"]} for c in st["creatures"] if c["monster"]],
            "rng": {k: {"seed": 1, "counter": v} for k, v in st["rng"]["counters"].items()}}


def step(action, points, emu=None):
    """points: [(ctx, game state)]; the step's sums are made up from the context names."""
    g = {"action": action, "state": snap(points[-1][1]) if points else {"in_combat": True},
         "checksums": [["action" if c.startswith("finished") else c, "PlayCardAction" if c.startswith("finished") else None, i]
                       for i, (c, _) in enumerate(points)],
         "net": [{"ctx": c, "id": i, "state": s} for i, (c, s) in enumerate(points)]}
    e = {"action": action, "net": [{"ctx": "step", "state": emu}]}
    return g, e


def fight(rows):
    gs, es = zip(*list(starmap(step, rows)))
    return {"seed": "T", "steps": list(gs)}, {"seed": "T", "steps": list(es)}


PLAYED = "finished action execution PlayCardAction"


class NetStateDiffTest(unittest.TestCase):
    def test_identical_on_every_modelled_field(self):
        g, e = fight([(None, [("After player turn start", net())], net(game=False)),
                      (PLAY, [(PLAYED, net(ehp=21, hand=()))], net(ehp=21, hand=(), game=False))])
        r = nd.diff_fight(g, e, verbose=False)
        self.assertNotIn("step", r)
        self.assertIn("2 steps identical", r["verdict"])
        # Fields only the game writes are listed, never compared.
        self.assertIn("relics.floor", r["not_modelled"])
        self.assertIn("phase", r["not_modelled"])
        self.assertNotIn("relics.counter", r["not_modelled"])

    def test_power_names_follow_fight_compare(self):
        # TEMPORARY_STRENGTH sums the game's subclasses; the emulator's BuffId may already end in Power.
        g, e = fight([(None, [("After player turn start", net(powers=[["SETUP_STRIKE_POWER", 2], ["FRAIL_POWER", 1]]))],
                       net(game=False, powers=[["TEMPORARY_STRENGTH_POWER", 2], ["FRAIL_POWER", 1]]))])
        self.assertNotIn("step", nd.diff_fight(g, e, verbose=False))

    def test_end_turn_divergence_is_blamed_on_the_phase_where_the_game_moved_on(self):
        start = net()
        phases = [(c, net(ai=1 if c == "After player turn start" else 0)) for c in PHASES]
        g, e = fight([(None, [("After player turn start", start)], net(game=False)),
                      (END, phases, net(ai=0, game=False))])
        r = nd.diff_fight(g, e, verbose=True)
        self.assertEqual(r["step"], 1)
        self.assertEqual(r["subaction"], "After player turn start")
        path, gv, ev, why = r["fields"][0]
        self.assertEqual((path, gv, ev), ("rng.MonsterAi", 1, 0))
        self.assertIn("emulator stops", why)
        self.assertEqual([v for _, v in r["timeline"]["rng.MonsterAi"]], [0, 0, 0, 0, 0, 1])

    def test_a_value_the_game_never_had_is_blamed_on_its_last_change(self):
        phases = [(c, net(php=70 if i < 3 else 64)) for i, c in enumerate(PHASES)]
        g, e = fight([(None, [("After player turn start", net())], net(game=False)),
                      (END, phases, net(php=61, game=False))])
        r = nd.diff_fight(g, e, verbose=False)
        self.assertEqual(r["subaction"], "After enemy turn end")

    def test_a_field_the_game_left_alone_is_not_localisable(self):
        g, e = fight([(None, [("After player turn start", net())], net(game=False)),
                      (PLAY, [(PLAYED, net(ehp=21, hand=()))], net(ehp=21, hand=(), block=5, game=False))])
        r = nd.diff_fight(g, e, verbose=False)
        self.assertEqual(r["subaction"], "not localisable from the game side")
        self.assertEqual(r["fields"][0][:3], ("player.block", 0, 5))

    def test_work_after_the_last_checksum_comes_from_the_driver_snapshot(self):
        # The auto-play phase runs after "After player turn start": the snapshot shows its block.
        g, e = fight([(None, [("After player turn start", net())], net(game=False))])
        g["steps"][0]["state"]["player"]["block"] = 4
        r = nd.diff_fight(g, e, verbose=False)
        self.assertEqual(r["moved"], [(0, ["player.block"])])
        self.assertEqual(r["fields"][0][:3], ("player.block", 4, 0))

    def test_a_play_without_its_checksum_is_a_driver_artifact(self):
        g, e = fight([(None, [("After player turn start", net())], net(game=False)),
                      (END, [(c, net()) for c in PHASES], net(game=False)),
                      (PLAY, [], net(ehp=21, hand=(), game=False)),          # nothing finished in the game
                      (PLAY, [(PLAYED, net(ehp=21, hand=()))], net(ehp=15, hand=(), game=False))])
        self.assertEqual(nd.late_steps(g), [2])
        r = nd.diff_fight(g, e, verbose=False)
        self.assertNotIn("step", r)
        # The end of turn before it may still have been running its turn-start work.
        self.assertIn("steps from 1 on are not comparable", r["verdict"])

    def test_refused_play_and_finished_fight_are_not_late(self):
        g, _ = fight([(None, [("After player turn start", net())], net(game=False)),
                      (dict(PLAY, played=False), [], net(game=False)),
                      (END, [(PHASES[0], net())], net(game=False))])
        g["steps"][2]["state"] = {"in_combat": False}
        self.assertEqual(nd.late_steps(g), [])

    def test_two_game_runs_compare_by_checksum(self):
        a, _ = fight([(None, [("After player turn start", net())], net(game=False)),
                      (PLAY, [(PLAYED, net(ehp=21, hand=()))], net(game=False))])
        b = copy.deepcopy(a)
        self.assertEqual(nd.diff_games(a, b)["equal_prefix"], 2)
        b["steps"][1]["checksums"][0][2] = 99
        b["steps"][1]["net"][0]["state"]["creatures"][1]["hp"] = 20
        r = nd.diff_games(a, b)
        self.assertEqual((r["index"], r["steps"]), (1, (1, 1)))
        self.assertEqual(r["fields"], [("enemies[0]", ["CORPSE_SLUG", 21, 27, 0], ["CORPSE_SLUG", 20, 27, 0])])

    def test_same_checksum_in_another_step_is_a_step_boundary_shift(self):
        a, _ = fight([(None, [("After player turn start", net())], net(game=False)),
                      (PLAY, [(PLAYED, net(ehp=21, hand=()))], net(game=False)),
                      (PLAY, [], net(game=False))])
        b = copy.deepcopy(a)
        b["steps"][2]["checksums"], b["steps"][1]["checksums"] = b["steps"][1]["checksums"], []
        r = nd.diff_games(a, b)
        self.assertEqual(r["first_step_shift"], {"checksum": 1, "steps": (1, 2)})


if __name__ == "__main__":
    unittest.main()
