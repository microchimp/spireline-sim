"""referee/lockstep_trace.py: a lockstep divergence record to a traced real fight (no game, no emulator needed).

The decision mapping on a synthetic run record, the spec builder's output shape, the logging lockstep and the command line.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT))

import lockstep  # noqa: E402
import lockstep_trace as lt  # noqa: E402


def state(rnd, hp):
    return {"round": rnd, "enemies": [{"name": "SOUL_FYSH", "hp": h} for h in hp]}


def play(card="STRIKE_IRONCLAD"):
    return {"kind": "play", "hand_index": 0, "card": card, "target": 0, "target_slot": 0, "played": True}


# Fight on floor 17: asks n=10 (start), 11, 12 (after an unasked End Turn the round changes), 13.
# steps: 0 start, 1 play (ask 10 answered), 2 play (ask 11), 3 end_turn (unasked: a lone End Turn), 4 play (ask 12), 5 play (ask 13)
STEPS = [{"action": None, "state": state(1, [200])},
         {"action": play(), "state": state(1, [194])},
         {"action": play("BASH"), "state": state(1, [185])},
         {"action": {"kind": "end_turn"}, "state": state(2, [185])},
         {"action": play(), "state": state(2, [179])},
         {"action": play(), "state": state(2, [175])}]


def run_record():
    snap = {"current_act_index": 1, "players": [{"current_hp": 50, "max_hp": 80, "gold": 9, "max_potion_slot_count": 2,
                                                 "deck": [{"id": "CARD.STRIKE_IRONCLAD"}], "relics": [{"id": "RELIC.BURNING_BLOOD"}], "potions": []}]}
    return {"seed": "FTV3HZZPFE", "character": "IRONCLAD", "ascension": 10, "chooser_seed": 4, "decisions": [
        {"i": 0, "screen": "start", "snap": snap},
        {"i": 1, "screen": "map", "floor": 16, "snap": snap},
        {"i": 2, "screen": "combat", "floor": 17, "encounter": "SOUL_FYSH", "fight": {"steps": STEPS}, "snap": snap}]}


def trail():
    t = [{"n": 5, "kind": "map", "floor": 16, "round": None, "hp": []}]
    for n, rnd, hp in ((10, 1, [200]), (11, 1, [194]), (12, 2, [185]), (13, 2, [179]), (14, 2, [175])):
        t.append({"n": n, "kind": "combat", "floor": 17, "round": rnd, "hp": hp})
    return t


RECORD = {"decision": 14, "seed": "FTV3HZZPFE", "floor": 17, "turn": 2, "agent_seed": [20261023, 235],
          "diff": [[".combat.enemy_hp", [175], [177]]]}


class MapDecision(unittest.TestCase):
    def test_unasked_end_turn_is_a_step(self):
        m = lt.map_decision(trail(), run_record(), 14)
        # four combat asks (10..13) came before ask 14, plus the lone End Turn nobody asked: the ask sees step 5
        self.assertEqual((m["fight"], m["step"], m["asked_before"], m["verified"]), (2, 5, 4, True))

    def test_before_the_unasked_step(self):
        m = lt.map_decision(trail(), run_record(), 11)
        self.assertEqual((m["step"], m["asked_before"], m["verified"]), (1, 1, True))

    def test_unverified_when_no_step_has_the_asks_state(self):
        t = trail()
        t[-1]["hp"] = [999]
        m = lt.map_decision(t, run_record(), 14)
        self.assertEqual((m["step"], m["verified"]), (4, False))

    def test_not_a_fight_ask(self):
        with self.assertRaises(lt.UsageError):
            lt.map_decision(trail(), run_record(), 5)
        with self.assertRaises(lt.UsageError):
            lt.map_decision(trail(), run_record(), 99)

    def test_reproduced(self):
        self.assertTrue(lt.reproduced(trail(), RECORD)[0])
        t = trail()
        t[-1]["hp"] = [177]
        ok, why = lt.reproduced(t, RECORD)
        self.assertFalse(ok)
        self.assertIn("enemy hp", why)
        self.assertFalse(lt.reproduced(trail()[:2], RECORD)[0])


class BuildSpec(unittest.TestCase):
    def setUp(self):
        self.m = lt.map_decision(trail(), run_record(), 14)

    def test_natural_shape(self):
        s = lt.build_spec(run_record(), self.m, RECORD, "natural")
        self.assertEqual((s["kind"], s["trace"], s["seed"], s["max_actions"]), ("combat", True, "FTV3HZZPFE", 20))
        self.assertNotIn("script", s)
        self.assertEqual([x["cmd"] for x in s["setup"]], ["act", "player_load", "fight"])
        self.assertEqual(s["setup"][-1]["id"], "SOUL_FYSH")
        self.assertEqual((s["run_repro"]["decision"], s["run_repro"]["step"], s["run_repro"]["field"]), (2, 5, ".combat.enemy_hp"))
        self.assertTrue(s["tag"].endswith(":lockstep:14"))

    def test_script_is_the_recorded_actions_up_to_the_step(self):
        s = lt.build_spec(run_record(), self.m, RECORD, "script")
        self.assertEqual(s["script"], [x["action"] for x in STEPS[1:6]])
        self.assertEqual(s["max_actions"], 5)
        json.dumps(s)

    def test_no_save_before_the_fight(self):
        r = run_record()
        for d in r["decisions"][:2]:
            d.pop("snap")
        with self.assertRaises(lt.UsageError):
            lt.build_spec(r, self.m, RECORD, "natural")


class Msg(dict):
    pass


def ask(kind="combat", rnd=1, hp=(100,), acts=(["play", 0, "STRIKE_IRONCLAD", 0], ["end"])):
    return {"kind": kind, "actions": list(acts), "obs": {"floor": 3, "combat": {"round": rnd, "enemies": [{"hp": h} for h in hp]}}}


class TraceLockstepTests(unittest.TestCase):
    def test_logs_every_ask_and_stops_the_emulator_after_the_target(self):
        ls = lt.TraceLockstep(None, lockstep.referee_only_policy, "SEED", 1, [1, 0], stop_after=1)
        answers = [ls.decide(ask(hp=(100 - 5 * i,))) for i in range(4)]
        self.assertEqual(answers, [0, 0, 0, 0])
        self.assertEqual([t["n"] for t in ls.trail], [0, 1, 2, 3])
        self.assertEqual(ls.trail[2]["hp"], [90])
        self.assertEqual(ls.trail[3]["chosen"], ["play", 0, "STRIKE_IRONCLAD", 0])
        self.assertEqual(ls.n, 4)
        self.assertEqual(ls.stats["decisions"], 2)  # asks after the target do not go through Lockstep.decide

    def test_without_stop_every_ask_is_a_lockstep_decision(self):
        ls = lt.TraceLockstep(None, lockstep.referee_only_policy, "SEED", 1, [1, 0])
        for _ in range(3):
            ls.decide(ask())
        self.assertEqual(ls.stats["decisions"], 3)


class CommandLine(unittest.TestCase):
    def block(self, tmp):
        p = Path(tmp)
        (p / "divergences.jsonl").write_text(json.dumps(RECORD) + "\n" + json.dumps({**RECORD, "decision": 20}) + "\n", encoding="utf-8")
        return p

    def test_find_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(lt.find_record(self.block(tmp), "FTV3HZZPFE", 20)["decision"], 20)
            with self.assertRaises(lt.UsageError):
                lt.find_record(self.block(tmp), "FTV3HZZPFE", 21)
            with self.assertRaises(lt.UsageError):
                lt.find_record(Path(tmp) / "nope", "FTV3HZZPFE", 14)

    def test_required_arguments_and_choices(self):
        p = lt.parser()
        with self.assertRaises(SystemExit):
            p.parse_args(["--seed", "X"])
        with self.assertRaises(SystemExit):
            p.parse_args(["--block", "b", "--seed", "X", "--decision", "1", "--game", "g", "--mode", "bogus"])
        a = p.parse_args(["--block", "b", "--seed", "X", "--decision", "7", "--game", "g"])
        self.assertEqual((a.decision, a.phase, a.mode, a.config, a.gen_dir), (7, "all", "auto", "a3-v1", None))

    def test_missing_record_and_missing_locks_exit_2_before_any_game(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = ["--block", str(self.block(tmp)), "--seed", "FTV3HZZPFE", "--game", str(Path(tmp) / "nogame"), "--work", str(Path(tmp) / "w"),
                    "--referee-dir", str(Path(tmp) / "ref")]
            self.assertEqual(lt.main(base + ["--decision", "999"]), 2)
            self.assertEqual(lt.main(base + ["--decision", "14"]), 2)  # no QUIET.lock and no lab/PAUSE in the (empty) referee folder
            ref = Path(tmp) / "ref"
            (ref / "lab").mkdir(parents=True)
            (ref / "QUIET.lock").write_text("x", encoding="utf-8")
            self.assertFalse(lt.locks_ok(lt.parser().parse_args(base + ["--decision", "14"])))
            (ref / "lab" / "PAUSE").write_text("x", encoding="utf-8")
            self.assertTrue(lt.locks_ok(lt.parser().parse_args(base + ["--decision", "14"])))

    def test_trace_phase_without_a_replay_has_nothing_to_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ["--block", str(self.block(tmp)), "--seed", "FTV3HZZPFE", "--decision", "14", "--game", "g", "--work", str(Path(tmp) / "w"),
                    "--no-lock-check", "--phase", "trace"]
            self.assertEqual(lt.main(args), 2)


if __name__ == "__main__":
    unittest.main()
