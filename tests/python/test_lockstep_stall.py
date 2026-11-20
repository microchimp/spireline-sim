"""The stand-in must not repeat an answer the game ignored, and a run the game will not move ends as `stalled`.

The 2026-11-19 real block: 14 runs at the Test Subject ended `fight_unfinished`. The driver offered an enemy-target potion with no
target while no creature was alive, the game cancelled it (the potion stayed, nothing changed), the fixed stand-in answered the same
decision the same way 376 to 394 times and the driver's 400-action cap ended the run. `lockstep.StallGuard` is the guard: an answer
the game has ignored at a state is not given again (End Turn instead, else the next action), and after `STALL_LIMIT` ignored answers
the run ends with `end: "stalled"` carrying the answer. No game and no emulator here: a scripted game stands in for the driver."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))

import lockstep  # noqa: E402

END = ["end"]
POTION = ["potion", 0, "VULNERABLE_POTION", -1]
CARD = ["play", 0, "DEFEND_IRONCLAD", -1]


def decision(actions, h="s0", floor=48, kind="combat", ctx=None):
    m = {"t": "decide", "kind": kind, "obs": {"floor": floor, "hp": 60, "combat": {"hand": [], "enemies": []}}, "actions": actions, "h": h}
    if ctx is not None:
        m["ctx"] = ctx
    return m


class StallGuardTests(unittest.TestCase):
    def test_the_first_answer_at_a_state_is_the_one_asked_for(self):
        g = lockstep.StallGuard(3)
        self.assertEqual(g.answer(decision([END, POTION]), 1), 1)

    def test_an_ignored_answer_is_not_given_again_end_turn_instead(self):
        g = lockstep.StallGuard(3)
        m = decision([END, POTION, ["potion", 1, "WEAK_POTION", -1]])
        self.assertEqual(g.answer(m, 1), 1)  # the potion; the game ignores it: the same state comes back
        self.assertEqual(g.answer(m, 1), 0)  # not the potion again: End Turn

    def test_without_an_end_action_the_next_untried_action_is_taken(self):
        g = lockstep.StallGuard(3)
        m = decision([["play", 0, "A", -1], ["play", 1, "B", -1], ["play", 2, "C", -1]], kind="select_cards")
        self.assertEqual(g.answer(m, 0), 0)
        self.assertEqual(g.answer(m, 0), 1)
        self.assertEqual(g.answer(m, 0), 2)

    def test_after_the_limit_the_guard_raises_with_the_first_ignored_answer(self):
        g = lockstep.StallGuard(2)
        m = decision([END, POTION, CARD])
        g.answer(m, 1)
        g.answer(m, 1)
        with self.assertRaises(lockstep.Stalled) as cx:
            g.answer(m, 1)
        self.assertEqual(cx.exception.record(), {"floor": 48, "kind": "combat", "action": POTION, "answers": 2})

    def test_it_raises_when_every_action_has_been_tried(self):
        g = lockstep.StallGuard(5)
        m = decision([END, POTION])
        g.answer(m, 1)
        g.answer(m, 1)  # End Turn: also ignored
        with self.assertRaises(lockstep.Stalled):
            g.answer(m, 1)

    def test_a_state_that_moved_starts_afresh(self):
        g = lockstep.StallGuard(2)
        self.assertEqual(g.answer(decision([END, POTION], h="a"), 1), 1)
        self.assertEqual(g.answer(decision([END, POTION], h="b"), 1), 1)  # the game moved: the same answer is fine
        self.assertEqual(g.answer(decision([END, POTION], h="b"), 1), 0)

    def test_the_picks_of_one_card_screen_are_not_a_repeat(self):
        # The driver holds a screen's picks until the last: the game's state (its hash) stands still between them, but the
        # screen's actions and its pick number differ.
        g = lockstep.StallGuard(2)
        a = decision([["A", 0], ["B", 0], ["C", 0]], kind="select_cards_combat", ctx={"pick": 1, "of": 2})
        b = decision([["B", 0], ["C", 0], "done"], kind="select_cards_combat", ctx={"pick": 2, "of": 2})
        self.assertEqual(g.answer(a, 0), 0)
        self.assertEqual(g.answer(b, 0), 0)

    def test_without_a_state_hash_the_observation_tells_the_state(self):
        g = lockstep.StallGuard(3)
        m = decision([END, POTION])
        del m["h"]
        g.answer(m, 1)
        self.assertEqual(g.answer(m, 1), 0)
        m2 = decision([END, POTION])
        del m2["h"]
        m2["obs"]["hp"] = 55  # a different state
        self.assertEqual(g.answer(m2, 1), 1)


class LockstepGuardTests(unittest.TestCase):
    def stream(self, n, actions):
        return [decision(actions) for _ in range(n)]

    def test_the_stand_in_ends_the_turn_when_the_game_ignored_its_potion(self):
        ls = lockstep.Lockstep(None, lockstep.referee_only_policy, "S", stall_limit=3)
        m = decision([END, POTION])
        self.assertEqual(ls.decide(m), 1)  # the first non-end action: the potion
        self.assertEqual(ls.decide(m), 0)  # ignored: End Turn
        self.assertEqual(ls.stats["ignored_answers"], 1)

    def test_a_game_that_ignores_everything_stops_the_stand_in(self):
        ls = lockstep.Lockstep(None, lockstep.referee_only_policy, "S", stall_limit=3)
        m = decision([END, POTION])
        ls.decide(m)
        ls.decide(m)
        with self.assertRaises(lockstep.Stalled):
            ls.decide(m)

    def test_without_a_limit_the_stand_in_repeats_as_it_always_did(self):
        ls = lockstep.Lockstep(None, lockstep.referee_only_policy, "S")
        m = decision([END, POTION])
        self.assertEqual([ls.decide(m) for _ in range(5)], [1] * 5)


class ScriptedGame:
    """A game that cancels every potion it is handed: the state (its hash) stands still, the decision comes back. Any other
    answer moves it one state on. `refuse_all` ignores every answer. It ends the run (death) after `moves` accepted answers."""

    def __init__(self, moves=3, refuse_all=False):
        self.moves, self.refuse_all, self.state, self.asked, self.last = moves, refuse_all, 0, 0, None

    def _msg(self):
        self.last = decision([END, POTION], h=f"state{self.state}")
        return self.last

    def reset(self, seed, hash=False, **kw):
        return self._msg(), False

    def step(self, a):
        self.asked += 1
        refused = self.refuse_all or self.last["actions"][a][0] == "potion"
        if not refused:
            self.state += 1
            if self.state >= self.moves:
                return {"t": "end", "end": "death", "floor": 48}, True
        return self._msg(), False


class PlayRunTests(unittest.TestCase):
    def play(self, game, **kw):
        return lockstep.play_run(game, "SEEDX", emu_factory=lambda s: None, agent_seed=[5, 0], **kw)

    def test_a_potion_the_game_cancels_costs_one_answer_not_four_hundred(self):
        game = ScriptedGame(moves=3)
        m, ls = self.play(game)
        self.assertEqual((m["end"], m["floor"]), ("death", 48))  # the run went on and ended on its own
        # State 0: the potion (ignored), End Turn; state 1: the potion (ignored), End Turn; and so on: two answers a state.
        self.assertEqual(game.asked, 6)
        self.assertEqual(ls.stats["ignored_answers"], 3)

    def test_a_game_that_ignores_everything_ends_the_run_as_stalled(self):
        game = ScriptedGame(refuse_all=True)
        m, ls = self.play(game)
        self.assertEqual((m["end"], m["floor"], m["t"]), ("stalled", 48, "end"))
        self.assertEqual(m["stalled"], {"floor": 48, "kind": "combat", "action": POTION, "answers": 2})
        self.assertTrue(m["stale"])  # the game stands at an unanswered decision: the worker starts a fresh one
        self.assertEqual(game.asked, 2)  # not 400
        self.assertEqual(ls.stats["stalled"], m["stalled"])

    def test_the_stalled_run_is_a_truncated_outcome_on_its_records(self):
        game = ScriptedGame(refuse_all=True)
        m, ls = self.play(game)
        self.assertEqual(lockstep.run_outcome(m, True), {"class": "truncated", "end": "stalled", "floor": 48})

    def test_the_limit_can_be_set_or_turned_off(self):
        game = ScriptedGame(refuse_all=True)
        m, _ = self.play(game, stall_limit=1)
        self.assertEqual((m["end"], game.asked), ("stalled", 1))
        game = ScriptedGame(refuse_all=True)
        m, _ = self.play(game, stall_limit=0, max_decisions=50)  # no guard: the old behaviour, the same answer to the end
        self.assertEqual((m["t"], game.asked), ("decide", 50))


if __name__ == "__main__":
    unittest.main()
