"""run_compare2 run FAILS on a deliberately broken replay and passes the faithful one: a snapshot field, a
stopped replay, a fight step, a fight the replay never played, an offer or snapshot it did not export,
an event's options it did not export or got wrong, a fight's outcome (also without --id-map), another
run's replay, a run with no replay (fixtures: broken_cases.py, test_run_snapshot.py).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from broken_cases import PERTURB, fight_pair, run_main, write_json  # noqa: E402
from tools.fidelity import run_compare2 as rc2  # noqa: E402


def run_pair(seed: str = "RUNBROKEN1") -> tuple[dict, dict]:
    """A recorded run of three decisions (a map move into a fight, the fight, an event) and its faithful replay.

    The replay exports what each screen offered as RunReplay.Apply does: the map's as "offer", the event's as
    its action mask ("enabled", "skip_action", "option_count"), never as an offer.
    """
    from test_run_snapshot import EMU, SAVE
    g, e = fight_pair(2)
    ref = {"seed": seed, "end": "death", "floor": 3, "decisions": [
        {"screen": "map", "choice": [3, 1], "offer": [[3, 1, "Monster"]], "snap": SAVE, "floor": 1},
        {"screen": "combat", "result": "win", "encounter": "ENCOUNTER.CORPSE_SLUGS_WEAK", "fight": {"steps": g["steps"]},
         "snap": SAVE, "floor": 2},
        {"screen": "event", "event": "EVENT.WELLSPRING", "floor": 3, "snap": SAVE,
         "offer": [["WELLSPRING.pages.INITIAL.options.BOTTLE", False], ["WELLSPRING.pages.INITIAL.options.BATHE", False],
                   ["WELLSPRING.pages.INITIAL.options.LEAVE", True]],
         "choice": "WELLSPRING.pages.INITIAL.options.BATHE", "choice_index": 1}]}
    emu = {"seed": seed, "decisions": [
        {"offer": [[3, 1, "Monster"]], "snap": EMU},
        {"fight": {"steps": e["steps"]}, "fight_end": "RelicReward", "outcome_snap": EMU, "snap": EMU},
        {"enabled": [0, 1], "skip_action": 3, "option_count": 3, "snap": EMU}]}
    return ref, emu


class RunCompareBrokenCases(unittest.TestCase):
    def compare(self, refs: list, emus: list) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as d:
            return run_main(rc2.main, ["run", write_json(Path(d) / "g.json", {"results": refs}),
                                       write_json(Path(d) / "e.json", {"results": emus}),
                                       "--id-map", str(ROOT / "data" / "id_map.json")])

    def test_the_faithful_replay_passes(self):
        ref, emu = run_pair()
        code, text = self.compare([ref], [emu])
        self.assertEqual(code, 0, text)

    def test_each_broken_replay_fails(self):
        cases = {
            "a snapshot field": lambda e: e["decisions"][0].__setitem__("snap", dict(e["decisions"][0]["snap"], gold=1)),
            "the replay stopped": lambda e: e["decisions"].pop(),
            "a fight step": lambda e: PERTURB["enemy hp"](e["decisions"][1]["fight"]["steps"][1]["state"]),
            "a fight the replay never played": lambda e: e["decisions"][1].pop("fight"),
            "an offer the replay did not export": lambda e: e["decisions"][0].pop("offer"),
            "a snapshot the replay did not export": lambda e: e["decisions"][0].pop("snap"),
            "the fight's outcome": lambda e: e["decisions"][1].__setitem__("outcome_snap", dict(e["decisions"][1]["outcome_snap"], hp=1)),
            "another run's replay": lambda e: e.__setitem__("seed", "RUNBROKEN2"),
            "an event's options the replay did not export": lambda e: e["decisions"][2].pop("enabled"),
            "an event option the game has locked enabled": lambda e: e["decisions"][2].__setitem__("enabled", [0, 1, 2]),
            "an event's option count": lambda e: e["decisions"][2].__setitem__("option_count", 2),
        }
        passed = []
        for name, wreck in cases.items():
            ref, emu = run_pair()
            wreck(emu)
            if self.compare([ref], [emu])[0] != 1:
                passed.append(name)
        self.assertEqual(passed, [], "broken replays the run comparison passed")

    def test_an_event_is_compared_by_its_options_not_by_an_offer(self):
        # The replay never writes an offer for an event: taking that for a missing offer flagged every event.
        ref, emu = run_pair()
        self.assertNotIn("offer", emu["decisions"][2])
        self.assertEqual(rc2.compare_decision(ref["decisions"][2], emu["decisions"][2], None), [])
        broken = dict(emu["decisions"][2], enabled=[0])
        self.assertEqual(rc2.compare_decision(ref["decisions"][2], broken, None)[0][0], "event options enabled")
        failed = {"error": "event option X not offered"}
        self.assertEqual([p for p, _, _ in rc2.compare_decision(ref["decisions"][2], failed, None)], ["replay"])

    def test_without_an_id_map_the_steps_are_not_compared_but_the_outcome_is(self):
        def compare(refs, emus):
            with tempfile.TemporaryDirectory() as d:
                return run_main(rc2.main, ["run", write_json(Path(d) / "g.json", {"results": refs}),
                                           write_json(Path(d) / "e.json", {"results": emus}), "--id-map", ""])
        ref, emu = run_pair()
        code, text = compare([ref], [emu])
        self.assertEqual(code, 0, text)
        self.assertIn("not compared here: fight steps (no --id-map)", text)
        self.assertIn("1 played, 0 identical at every step", text)
        PERTURB["enemy hp"](emu["decisions"][1]["fight"]["steps"][1]["state"])
        self.assertEqual(compare([ref], [emu])[0], 0, "a step difference is not seen without the map")
        ref, emu = run_pair()
        emu["decisions"][1]["outcome_snap"] = dict(emu["decisions"][1]["outcome_snap"], hp=1)
        code, text = compare([ref], [emu])
        self.assertEqual(code, 1, text)
        self.assertIn("combat outcome -> hp", text)

    def test_a_run_without_a_replay_fails(self):
        ref, emu = run_pair()
        ref2, _ = run_pair("RUNBROKEN2")
        self.assertEqual(self.compare([ref, ref2], [emu])[0], 1)

    def test_fight_steps_not_compared_are_never_identical(self):
        ref, emu = run_pair()
        self.assertEqual(rc2.fight_diff(ref["decisions"][1], emu["decisions"][1], None), rc2.NOT_COMPARED_FIGHT)


if __name__ == "__main__":
    unittest.main()
