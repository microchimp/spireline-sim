"""tools.fidelity.reanchor_compare: per-step verdicts of a re-anchored fight.

Synthetic fights in the shapes the referee and `EmuDump reanchor` write. A step that matches verifies
its card whatever came before it; a divergence is blamed on its card only when the step before it
matched; an end of turn whose enemies differ ends the fight's re-anchoring (or, in "plays" mode, only
its later ends of turn); and powers the emulator models in pieces never cause a difference.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import coverage  # noqa: E402
from tools.fidelity import fight_compare as fc
from tools.fidelity import reanchor_compare as rc

ID_MAP = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))
NM = fc.names(ID_MAP)
CID, EID = ID_MAP["cards"], ID_MAP["enemies"]
STRIKE, SLUG = "STRIKE_IRONCLAD", "CORPSE_SLUG"
KEYS = ({"STRENGTH", "WEAK", "RAVENOUS"}, {"CRIMSON_MANTLE_BLOCK"})


def game_state(hp=27, php=70, powers=(), player_powers=()):
    return {"in_combat": True, "round": 1,
            "player": {"hp": php, "max_hp": 80, "block": 0, "energy": 3, "powers": [list(p) for p in player_powers]},
            "hand": [[STRIKE, False]], "draw": [], "discard": [], "exhaust": [],
            "enemies": [{"name": SLUG, "hp": hp, "max_hp": 27, "block": 0, "powers": [list(p) for p in powers]}],
            "rng": {"Shuffle": {"seed": 1, "counter": 1}}}


def emu_state(hp=27, php=70, buffs=(), player_buffs=()):
    card = {"def_id": CID["StrikeIronclad"], "upgraded": False}
    return {"in_combat": True, "draw_order": [], "rng": {"Shuffle": 1},
            "snapshot": {"player": {"hp": php, "max_hp": 80, "block": 0, "energy": 3,
                                    "buffs": [{"name": n, "amount": a} for n, a in player_buffs]},
                         "piles": {"hand": [card], "discard": [], "exhaust": []},
                         "enemies": [{"def_id": EID["CorpseSlug"], "hp": hp, "block": 0,
                                      "buffs": [{"name": n, "amount": a} for n, a in buffs]}]}}


PLAY = {"kind": "play", "hand_index": 0, "card": STRIKE, "target": 0, "target_slot": 0}
END = {"kind": "end_turn"}


def fight(rows):
    """rows: (action, game state kwargs, emulator state kwargs); the first row is the start."""
    g = {"seed": "T", "steps": [{"action": a, "state": game_state(**gk)} for a, gk, _ in rows]}
    e = {"seed": "T", "steps": [{"action": a, "state": emu_state(**ek)} for a, _, ek in rows]}
    return g, e


def verdicts(rows, **kw):
    g, e = fight(rows)
    return rc.analyse(g, e, NM, **kw)


class ReanchorCompareTest(unittest.TestCase):
    def test_every_step_verifies_its_card_even_after_a_divergence(self):
        a = verdicts([(None, {}, {}),
                      (PLAY, dict(hp=21), dict(hp=21)),
                      (PLAY, dict(hp=15), dict(hp=14)),   # diverges: primary
                      (PLAY, dict(hp=9), dict(hp=9)),     # the next step was anchored again and matches
                      (PLAY, dict(hp=3), dict(hp=2))])    # diverges: primary again (the step before matched)
        self.assertEqual([s["verdict"] for s in a["steps"]], ["match", "match", "diverge", "match", "diverge"])
        self.assertEqual([s["primary"] for s in a["steps"]], [False, False, True, False, True])
        self.assertEqual(a["first_divergence"], 2)
        sm = rc.summarise([a])["totals"]
        self.assertEqual((sm["verified"], sm["verified_before"], sm["diverging"]), (2, 1, 2))

    def test_second_divergence_in_a_row_is_knock_on(self):
        a = verdicts([(None, {}, {}), (PLAY, dict(hp=21), dict(hp=20)), (PLAY, dict(hp=15), dict(hp=13))])
        self.assertEqual([s["primary"] for s in a["steps"] if s["j"]], [True, False])

    def test_enemy_difference_at_end_of_turn_stops_the_fight(self):
        rows = [(None, {}, {}), (END, dict(hp=27, php=60), dict(hp=27, php=64)), (PLAY, dict(hp=21), dict(hp=21))]
        a = verdicts(rows)
        self.assertEqual(len(a["steps"]), 2)  # the end of turn is judged, the play after it is not tried
        self.assertIn("enemy move/intent", a["stop"]["why"])
        self.assertEqual(rc.summarise([a])["totals"]["not_reanchored"], 1)

    def test_plays_mode_goes_on_and_leaves_later_ends_of_turn_unjudged(self):
        rows = [(None, {}, {}), (END, dict(php=60), dict(php=64)), (PLAY, dict(hp=21), dict(hp=21)),
                (END, dict(hp=21, php=50), dict(hp=21, php=44))]
        a = verdicts(rows, after_desync="plays")
        self.assertEqual([s["verdict"] for s in a["steps"]], ["match", "diverge", "match", "unjudged"])
        self.assertIsNone(a["stop"])
        sm = rc.summarise([a])["totals"]
        self.assertEqual((sm["reanchored"], sm["verified"], sm["not_reanchored"]), (2, 1, 1))

    def test_player_side_power_at_end_of_turn_is_not_an_enemy_move(self):
        game = dict(php=60, player_powers=[("BEACON_OF_HOPE_POWER", 1)])
        a = verdicts([(None, {}, {}), (END, game, dict(php=60)), (PLAY, dict(hp=21), dict(hp=21))])
        self.assertIsNone(a["stop"])
        self.assertEqual(a["steps"][1]["verdict"], "diverge")

    def test_enemy_applied_debuff_at_end_of_turn_stops(self):
        a = verdicts([(None, {}, {}), (END, dict(player_powers=[("WEAK_POWER", 1)]), {})])
        self.assertIsNotNone(a["stop"])

    def test_powers_the_emulator_models_in_pieces_never_differ(self):
        game = dict(hp=21, player_powers=[("CRIMSON_MANTLE_POWER", 8)])
        emu = dict(hp=21, player_buffs=[("CrimsonMantleBlock", 8)])
        # fight_compare now maps CrimsonMantleBlock to CRIMSON_MANTLE (EMU_RENAME), so even the
        # unmasked comparison agrees; the masked one must too.
        strict = verdicts([(None, {}, {}), (PLAY, game, emu)])
        self.assertEqual(strict["steps"][1]["verdict"], "match")
        g, e = fight([(None, {}, {}), (PLAY, game, emu)])
        masked = rc.analyse(g, e, NM, keys=KEYS)
        self.assertEqual(masked["steps"][1]["verdict"], "match")
        self.assertIsNone(masked["first_divergence"])  # the plain replay maps it too (EMU_RENAME)

    def test_unrebuildable_state_ends_the_fight_before_the_step(self):
        g, e = fight([(None, {}, {}), (PLAY, dict(hp=21), dict(hp=21))])
        e["steps"][1] = {"action": PLAY, "anchor_error": "no emulator id for BOGUS"}
        a = rc.analyse(g, e, NM)
        self.assertIn("BOGUS", a["stop"]["why"])
        self.assertEqual(a["steps"][-1]["j"], 0)

    def test_coverage_credits_matches_and_primary_divergences_only(self):
        a = verdicts([(None, {}, {}), (PLAY, dict(hp=21), dict(hp=21)), (PLAY, dict(hp=15), dict(hp=14)),
                      (PLAY, dict(hp=9), dict(hp=7))])
        cov = {"cards": {}}
        out = coverage.update_reanchored(cov, [a], "abc")
        self.assertEqual(cov["cards"][STRIKE]["verified"], 1)
        self.assertEqual(cov["cards"][STRIKE]["suspect_count"], 1)  # the third step is a knock-on
        self.assertEqual(out["new_suspects"], 1)


class PerFightHelpersTest(unittest.TestCase):
    """fight_compare.report and reanchor_compare.analyses: what batch.py computes chunk by chunk must be
    what fight_compare's main prints and what learn() folds in for the whole batch."""

    def fights(self):
        same = fight([(None, {}, {}), (PLAY, dict(hp=21), dict(hp=21))])
        differ = fight([(None, {}, {}), (PLAY, dict(hp=21), dict(hp=21)), (PLAY, dict(hp=15), dict(hp=14))])
        broken = ({"seed": "E", "steps": [], "error": "idle_timeout"}, {"seed": "E", "steps": [], "error": "no game result to replay"})
        return [same, differ, broken]

    def test_report_is_main_output_and_first_divergence(self):
        import contextlib
        import io
        import tempfile

        pairs = self.fights()
        with tempfile.TemporaryDirectory() as d:
            gp, ep = Path(d) / "g.json", Path(d) / "e.json"
            gp.write_text(json.dumps({"results": [g for g, _ in pairs]}), encoding="utf-8")
            ep.write_text(json.dumps({"results": [e for _, e in pairs]}), encoding="utf-8")
            buf = io.StringIO()
            argv = sys.argv
            sys.argv = ["fight_compare", str(gp), str(ep), "--id-map", str(ROOT / "data" / "id_map.json")]
            try:
                with contextlib.redirect_stdout(buf):
                    fc.main()
            finally:
                sys.argv = argv
        lines = [ln for g, e in pairs for ln in fc.report(g, e, NM)[0]]
        self.assertEqual(buf.getvalue().splitlines(), lines)
        self.assertEqual(lines[-1], "E: 0 steps identical (game 0, emulator 0)")
        firsts = [fc.report(g, e, NM)[1] for g, e in pairs]
        self.assertEqual(firsts, [coverage.first_divergence(g, e, NM) for g, e in pairs])
        self.assertEqual(firsts, [None, 2, None])

    def test_analyses_is_what_learn_folds_in(self):
        pairs = self.fights()
        games = [g for g, _ in pairs]
        doc = {"results": [e for _, e in pairs], "buff_keys": sorted(KEYS[0]), "internal_keys": sorted(KEYS[1])}
        got = rc.analyses(games, doc, NM)
        self.assertEqual(got, [rc.analyse(g, r, NM, "stop", rc.keys_of(doc)) for g, r in rc.pair(games, doc["results"])])
        a, b = {}, {}
        rc.learn(a, games, doc, NM, "c1")
        coverage.update_reanchored(b, got, "c1")
        self.assertEqual(a, b)
        self.assertTrue(a["cards"])
def with_moves(rows, moves, game_intents, emu_intents):
    """A fight whose records carry the enemies' moves: the game's intents per step, EmuDump's "moves"
    verdict per step and the emulator's announced intent (type, magnitude, hits)."""
    g, e = fight(rows)
    for st, its in zip(g["steps"], game_intents):
        st["state"]["enemies"][0]["intents"] = its
    for st, it in zip(e["steps"], emu_intents):
        st["state"]["snapshot"]["enemies"][0]["intent"] = {"type": it[0], "magnitude": it[1], "hits": it[2],
                                                           "carries_damage": False, "secondary": None}
    for st, m in zip(e["steps"][1:], moves):
        st["moves"] = m[0]
        if m[0] == "unknown":
            st["moves_unknown"] = [0]
    return g, e


GLOMP, GOOP = [["Attack", 9, 1]], [["Debuff"]]


class RecordedMovesTest(unittest.TestCase):
    def test_end_of_turn_with_the_game_moves_is_judged_and_the_fight_goes_on(self):
        # The emulator hits harder at the end of turn but announced the game's move before it: a
        # divergence of the end of turn, not a reason to stop re-anchoring.
        rows = [(None, {}, {}), (END, dict(php=60), dict(php=55)), (PLAY, dict(hp=21), dict(hp=21)),
                (END, dict(hp=21, php=50), dict(hp=21, php=50))]
        g, e = with_moves(rows, [("same",), ("same",), ("same",)], [GLOMP, GOOP, GOOP, GLOMP],
                          [(0, 9, 1), (3, 2, 1), (3, 2, 1), (0, 9, 1)])
        a = rc.analyse(g, e, NM)
        self.assertIsNone(a["stop"])
        self.assertEqual([s["verdict"] for s in a["steps"]], ["match", "diverge", "match", "match"])

    def test_wrong_move_shows_at_the_step_that_chose_it(self):
        rows = [(None, {}, {}), (END, dict(php=61), dict(php=61))]
        g, e = with_moves(rows, [("same",)], [GLOMP, GOOP], [(0, 9, 1), (0, 3, 2)])  # WHIP_SLAP instead of GOOP
        a = rc.analyse(g, e, NM)
        self.assertEqual(a["steps"][1]["verdict"], "diverge")
        self.assertEqual(a["steps"][1]["paths"], [".enemies"])

    def test_unknown_moves_leave_the_end_of_turn_unjudged_and_plays_judged_without_them(self):
        rows = [(None, {}, {}), (PLAY, dict(hp=21), dict(hp=21)), (END, dict(hp=21, php=60), dict(hp=21, php=58))]
        g, e = with_moves(rows, [("unknown",), ("unknown",)], [GLOMP, GLOMP, GOOP], [(0, 9, 1), (2, 0, 1), (2, 0, 1)])
        a = rc.analyse(g, e, NM)
        self.assertEqual([s["verdict"] for s in a["steps"]], ["match", "match", "unjudged"])
        self.assertIsNone(a["stop"])


if __name__ == "__main__":
    unittest.main()
