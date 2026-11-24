"""Fix c of lane 21's memo (Act 3 week, lane 22): the copy steps its End Turn exactly as often as the game took one without asking.

The driver asks only with two or more legal actions and lists each lone fight action it took unasked ("unasked", driver/Unasked.cs).
Before, the lockstep inferred the game's unasked End Turns from the copy's own legal list: when the game's player had nothing left to
play and the copy still had a card, the copy stood a turn behind (group A2: 10 of the 95 lost first-Act-3-boss arrivals of
2026-11-20, 251 of the block's 1,459 loss records). Fakes in the style of test_lockstep.LoneEndTurnTests; no native library."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import lockstep  # noqa: E402
from test_lockstep import FakeEmu, FakeRun, ItemsRun, _opt, msg, obs, stub_emu, view_of  # noqa: E402

HAND = [["STRIKE_IRONCLAD", 0], ["DEFEND_IRONCLAD", 0], ["BASH", 0]]
END = 3  # the copy's End Turn: one past a three-card hand
LABELS = [["play", 0, "STRIKE_IRONCLAD", 0], ["end"]]
GAME_END = ["combat", ["end"]]


def fight(legal, win_after=None):
    """An EmuLock stub in a fight. `legal[k]` is the copy's legal list after k steps (the last one stays); after `win_after` steps
    the fight is over (the rewards screen)."""
    screens = {"items": ("rewards", (_opt("reward.proceed", 9),)), "map": ("map", (_opt("map.node", 0, node=(1, 2)),))}
    run = ItemsRun(gold=0)
    run.cur, run.screens, run.stepped = "combat", screens, []
    run.moves[("map", "map.node")] = "map"
    e = stub_emu(run, hand=HAND)

    def mask(h, n):
        now = legal[min(len(run.stepped), len(legal) - 1)] if run.cur == "combat" else []
        return [int(i in now) for i in range(n)]

    def step(handle, action, target, *bufs):
        run.stepped.append(action)
        if win_after is not None and len(run.stepped) >= win_after:
            run.cur = "items"
        return 0

    e.native.run_action_mask, e.native.run_step = mask, step
    return e, run


class UnaskedEndTurnTests(unittest.TestCase):
    def test_the_games_unasked_end_turn_is_stepped_though_the_copy_could_still_play(self):
        e, run = fight([[0, END]])
        e.take_unasked([GAME_END])
        self.assertIsNone(e.sync("combat", LABELS, {"kind": "combat"}))
        self.assertEqual(run.stepped, [END])  # once: the copy's next turn is the game's, and it plays it
        self.assertEqual(e.unasked_counts, {"game_end_turns": 1, "stepped": 1, "stepped_with_choice": 1, "held_lone_end": 0})

    def test_an_old_driver_without_the_list_leaves_the_copy_as_it_was(self):
        e, run = fight([[0, END]])
        e.take_unasked(None)
        self.assertIsNone(e.sync("combat", LABELS, {"kind": "combat"}))
        self.assertEqual(run.stepped, [])  # a choice left: nothing is taken, as before
        self.assertIsNone(e.unasked_counts)
        e, run = fight([[END]], win_after=1)
        e.take_unasked(None)
        e.sync("reward", ["take", "skip"], {"ctx": {"type": "card"}})
        self.assertEqual(run.stepped, [END])  # the copy's lone End Turn, as before

    def test_a_lone_end_turn_the_game_did_not_take_is_not_taken(self):
        e, run = fight([[END]], win_after=1)
        e.take_unasked([])
        self.assertIsNone(e.sync("combat", LABELS, {"kind": "combat"}))
        self.assertIsNone(e.sync("combat", LABELS, {"kind": "combat"}))  # synced again (after a re-anchor): counted once
        self.assertEqual(run.stepped, [])
        self.assertEqual(e.unasked_counts["held_lone_end"], 1)

    def test_end_turns_are_stepped_exactly_as_often_as_the_game_took_them(self):
        e, run = fight([[0, END]])
        e.take_unasked([GAME_END, ["select_cards_combat", ["STRIKE_IRONCLAD", 0]], GAME_END])
        e.sync("combat", LABELS, {"kind": "combat"})
        self.assertEqual(run.stepped, [END, END])
        self.assertEqual(e.unasked_counts["stepped"], 2)

    def test_the_copys_other_lone_actions_are_taken_as_before(self):
        # A lone action that is not End Turn (one card on a fight card screen, one card to play): taken, with or without a game End Turn.
        e, run = fight([[1], [0, END]])
        e.take_unasked([GAME_END])
        e.sync("combat", LABELS, {"kind": "combat"})
        self.assertEqual(run.stepped, [1, END])
        e, run = fight([[1], [0, END]])
        e.take_unasked([])
        e.sync("combat", LABELS, {"kind": "combat"})
        self.assertEqual(run.stepped, [1])

    def test_a_fight_the_game_left_with_an_unasked_end_turn_is_left_by_the_copy_too(self):
        e, run = fight([[0, END]], win_after=1)  # the copy could still play a card; the game's End Turn won the fight
        e.take_unasked([GAME_END])
        self.assertIsNone(e.sync("reward", ["take", "skip"], {"ctx": {"type": "card"}}))
        self.assertEqual(run.stepped, [END])
        e, run = fight([[END]], win_after=1)  # the game's fight ended on the asked step; the copy's did not: a difference, as it is
        e.take_unasked([])
        self.assertEqual(e.sync("reward", ["take", "skip"], {"ctx": {"type": "card"}}), f"phase:{FakeRun.COMBAT}!=rewards")
        self.assertEqual(run.stepped, [])

    def test_no_end_turn_is_stepped_on_an_open_card_screen(self):
        e, run = fight([[0, 1]])
        e._obs = lambda: [1]  # the selection kind slot (stub: index 0) set: the actions are the screen's cards
        e.take_unasked([GAME_END])
        e.sync("combat", LABELS, {"kind": "combat"})
        self.assertEqual(run.stepped, [])

    def test_a_re_anchor_leaves_no_end_turn_to_take(self):
        # The game's record is of this decision, after its unasked steps: a re-anchor puts the copy there.
        e, _ = fight([[0, END]])
        e.take_unasked([GAME_END, GAME_END])
        self.assertFalse(e.reanchor("no_such_plan", {"obs": {}}))
        self.assertEqual(e._unasked_ends, 0)
        e.take_unasked(None)
        e.reanchor("no_such_plan", {"obs": {}})
        self.assertIsNone(e._unasked_ends)  # an old driver stays an old driver


class LockstepHandsTheListOverTests(unittest.TestCase):
    class TakingEmu(FakeEmu):
        def __init__(self, views):
            super().__init__(views)
            self.told = []
            self.unasked_counts = {"game_end_turns": 1, "stepped": 1, "stepped_with_choice": 1, "held_lone_end": 0}

        def take_unasked(self, unasked):
            self.told.append(unasked)

    def test_once_per_decision_none_without_the_key_and_into_the_record_and_the_stats(self):
        c, d = obs(combat=True), obs(hp=50, combat=True)
        emu = self.TakingEmu([view_of(c), view_of(c)])
        ls = lockstep.Lockstep(emu, lambda m: 0)
        ls.decide({**msg("combat", c), "unasked": []})
        ls.decide({**msg("combat", d), "unasked": [GAME_END]})  # the copy kept 80 HP: a record
        ls.decide(msg("combat", d))
        self.assertEqual(emu.told, [[], [GAME_END], None])
        self.assertEqual(ls.log.records[0]["ref_unasked"], [GAME_END])
        ls.finish({"class": "death"})
        self.assertEqual(ls.stats["unasked"]["stepped_with_choice"], 1)

    def test_a_side_without_the_hook_is_never_handed_it_and_the_stats_have_no_entry(self):
        c = obs(combat=True)
        ls = lockstep.Lockstep(FakeEmu([view_of(c)]), lambda m: 0)
        ls.decide({**msg("combat", c), "unasked": [GAME_END]})
        ls.finish({"class": "death"})
        self.assertNotIn("unasked", ls.stats)


class DriverListsTheLoneFightActionsTests(unittest.TestCase):
    def test_env_decide_lists_a_lone_fight_action_and_sends_the_list_with_the_next_question_and_the_end(self):
        src = (ROOT / "referee" / "driver" / "Env.cs").read_text(encoding="utf-8")
        decide = src[src.index("internal static int Decide("):src.index("internal static List<int> DecideMany(")]
        lone = decide[decide.index("if (n <= 1)"):decide.index('var msg = new JsonObject { ["t"] = "decide"')]
        self.assertIn("Unasked.Took(kind", lone)  # taken without asking: listed
        self.assertIn('msg["unasked"] = Unasked.Drain();', decide)  # every asked decision carries the list, empty or not
        serve = src[src.index("private static async Task<int> ServeEnv("):]
        self.assertIn('msg["unasked"] = Env.Unasked.Drain();', serve)
        run_play = (ROOT / "referee" / "driver" / "RunPlay.cs").read_text(encoding="utf-8")
        self.assertIn("Env.Unasked.Clear();", run_play)  # a new run starts with an empty list


if __name__ == "__main__":
    unittest.main()
