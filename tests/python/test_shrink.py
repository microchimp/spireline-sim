"""referee/shrink.py: delta debugging of a diverging setup fight, against a fake oracle.

The oracle stands in for "play the subset in the game in script mode, replay it in the emulator, and
ask whether it still first diverges at the same step and field". A case is the minimal set of culprits
the oracle needs; the shrinker must find exactly it, within the per-case cap of 48 game fights, with
every case of a batch sharing one game round at a time.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT))

import shrink  # noqa: E402


def spec(n_items: int, seed: str = "SEED000001") -> dict:
    setup = [{"cmd": "relic", "id": f"R{i}"} for i in range(n_items)]
    setup.insert(n_items // 2, {"cmd": "fight", "id": "SLIMES_WEAK"})
    return {"kind": "combat", "seed": seed, "setup": setup, "use_potions": True}


def game(steps: int = 6) -> dict:
    return {"steps": [{"action": None}] + [{"action": {"kind": "end_turn", "n": i}} for i in range(steps)]}


def fake_round(culprits_by_seed: dict, log: list, flaky: set = frozenset(), unresolved=lambda ids: False):
    """run_round: a subset fails when it still holds every culprit of its case."""
    def run_round(name: str, specs: list) -> list:
        log.append((name, len(specs)))
        out = []
        for s in specs:
            ids = {it["id"] for it in s["setup"]}
            assert "SLIMES_WEAK" in ids, "the fight item must always stay"
            assert s["max_actions"] == len(s["script"]) == s["_expect"]["step"]
            if s["seed"] in flaky:
                out.append(False)
            elif unresolved(ids):
                out.append(None)  # an illegal scripted action: counts as not failing
            else:
                out.append(culprits_by_seed[s["seed"]] <= ids)
        return out
    return run_round


class DdminTests(unittest.TestCase):
    def drive(self, items, fails):
        gen, tests = shrink.ddmin(items), 0
        try:
            subsets = next(gen)
            while True:
                tests += len(subsets)
                subsets = gen.send([fails(x) for x in subsets])
        except StopIteration as stop:
            return stop.value, tests

    def test_one_culprit_is_a_binary_search(self):
        items = list(range(40))
        got, tests = self.drive(items, lambda xs: 17 in xs)
        self.assertEqual([17], got)
        self.assertLessEqual(tests, 16)

    def test_two_culprits_in_different_halves(self):
        got, tests = self.drive(list(range(40)), lambda xs: {7, 23} <= set(xs))
        self.assertEqual([7, 23], got)
        self.assertLessEqual(tests, 48)

    def test_three_culprits(self):
        got, _ = self.drive(list(range(30)), lambda xs: {0, 14, 29} <= set(xs))
        self.assertEqual([0, 14, 29], got)

    def test_no_item_needed(self):
        got, _ = self.drive(list(range(5)), lambda xs: True)
        self.assertEqual([], got)


class ShrinkAllTests(unittest.TestCase):
    def test_cases_share_rounds_and_find_their_culprits(self):
        a, b = spec(40, "SEEDAAAAAA"), spec(24, "SEEDBBBBBB")
        cases = [shrink.Case(a, game(), 4, ".enemies"), shrink.Case(b, game(), 2, ".player.hp")]
        log: list = []
        minimal, rep = shrink.shrink_all(cases, fake_round({"SEEDAAAAAA": {"R3", "R30"}, "SEEDBBBBBB": {"R11"}}, log), "t")
        self.assertEqual({"R3", "R30"}, {it["id"] for it in minimal["SEEDAAAAAA"]["items"]})
        self.assertEqual({"R11"}, {it["id"] for it in minimal["SEEDBBBBBB"]["items"]})
        self.assertEqual(len(log), rep["rounds"])
        self.assertEqual(2, log[0][1], "the first round reproduces both cases together")
        for c in cases:
            self.assertEqual("minimal", c.status)
            self.assertLessEqual(c.spent, 48)
        # The minimal spec runs as is: script mode, the fight item kept, the culprits only.
        m = minimal["SEEDAAAAAA"]["spec"]
        self.assertEqual(["R3", "SLIMES_WEAK", "R30"], [it["id"] for it in m["setup"]])
        self.assertEqual(4, len(m["script"]))
        self.assertNotIn("_expect", m)

    def test_a_fight_that_does_not_diverge_again_is_flaky(self):
        cases = [shrink.Case(spec(10, "SEEDFLAKY1"), game(), 3, ".hand")]  # noqa: the oracle answers False
        log: list = []
        minimal, rep = shrink.shrink_all(cases, fake_round({"SEEDFLAKY1": {"R1"}}, log, flaky={"SEEDFLAKY1"}), "t")
        self.assertEqual({}, minimal)
        self.assertEqual("flaky", cases[0].status)
        self.assertEqual(1, rep["fights"])

    def test_unresolved_subsets_count_as_passing(self):
        # R5 makes every scripted action illegal when present, so no subset holding it can be judged:
        # the culprit R2 alone is still found, R5 is dropped.
        cases = [shrink.Case(spec(12, "SEEDUNRES1"), game(), 3, ".hand")]
        minimal, _ = shrink.shrink_all(cases, fake_round({"SEEDUNRES1": {"R2"}}, [], unresolved=lambda ids: "R5" in ids), "t")
        # The whole setup holds R5, so the reproduction itself cannot be judged.
        self.assertEqual("unresolved", cases[0].status)
        self.assertEqual({}, minimal)
        # A subset that holds R5 without R0 cannot be judged (the full setup holds both): such subsets
        # are simply not failing ones, and the culprit is still found.
        cases = [shrink.Case(spec(12, "SEEDUNRES2"), game(), 3, ".hand")]
        minimal, _ = shrink.shrink_all(cases, fake_round({"SEEDUNRES2": {"R2"}}, [],
                                                         unresolved=lambda ids: "R5" in ids and "R0" not in ids), "t")
        self.assertEqual({"R2"}, {it["id"] for it in minimal["SEEDUNRES2"]["items"]})

    def test_the_cap_keeps_the_smallest_failing_subset(self):
        cases = [shrink.Case(spec(40, "SEEDCAPPED"), game(), 1, ".hand", cap=10)]
        minimal, _ = shrink.shrink_all(cases, fake_round({"SEEDCAPPED": {"R1", "R9", "R20", "R33"}}, []), "t")
        self.assertEqual("capped", cases[0].status)
        self.assertLessEqual(cases[0].spent, 10)
        got = {it["id"] for it in minimal["SEEDCAPPED"]["items"]}
        self.assertTrue({"R1", "R9", "R20", "R33"} <= got)
        self.assertLess(len(got), 40)

    def test_budget_stops_every_case(self):
        cases = [shrink.Case(spec(40, f"SEEDBUDGE{i}"), game(), 1, ".hand") for i in range(3)]
        _, rep = shrink.shrink_all(cases, fake_round({c.seed: {"R0", "R39"} for c in cases}, []), "t", budget=20)
        self.assertLessEqual(rep["fights"], 20)

    def test_deck_add_entries_are_items_too(self):
        s = spec(4, "SEEDDECKAD")
        s["deck_add"] = ["ANGER", "CLASH"]
        c = shrink.Case(s, game(), 2, ".hand")
        self.assertIn(("deck_add", 1), c.items)
        sub = c.subset_spec([("deck_add", 1)])
        self.assertEqual(["CLASH"], sub["deck_add"])
        self.assertEqual(["SLIMES_WEAK"], [it["id"] for it in sub["setup"]])


if __name__ == "__main__":
    unittest.main()
