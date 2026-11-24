"""Who played each boss fight (`stats["boss_fights"]`) and where a loss was recovered (`resumed_at`): the record fields that let a
reader tell the agent's boss fights from the stand-in's (Act 3 week, lane 21). Fake referee streams."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import lockstep  # noqa: E402
from test_lockstep import FakeEmu, msg, obs, view_of  # noqa: E402


def boss_obs(floor=48, hp=80, act=2):
    o = obs(hp=hp, floor=floor, combat=True)
    o["act"] = act
    return o


class BossFightTallyTests(unittest.TestCase):
    def test_a_boss_fight_the_agent_answers_throughout_has_no_stand_in_decision(self):
        b = boss_obs()
        emu = FakeEmu([view_of(obs(floor=47)), view_of(b), view_of(b), view_of(b), view_of(obs(floor=48))])
        ls = lockstep.Lockstep(emu, lambda m: 0)
        for m in (msg("map", obs(floor=47)), msg("combat", b), msg("combat", b), msg("combat", b), msg("reward", obs(floor=48))):
            ls.decide(m)
        self.assertEqual(ls.stats["boss_fights"], [{"act": 2, "floor": 48, "first": 1, "decisions": 3, "stand_in": 0, "first_stand_in": None}])

    def test_a_loss_inside_the_boss_fight_counts_the_stand_ins_decisions_from_it(self):
        b = boss_obs()
        emu = FakeEmu([view_of(obs(floor=47)), view_of(b), view_of(boss_obs(hp=50))])
        ls = lockstep.Lockstep(emu, lambda m: 0)
        for m in (msg("map", obs(floor=47)), msg("combat", b), msg("combat", b), msg("combat", b)):
            ls.decide(m)
        self.assertEqual(ls.stats["boss_fights"], [{"act": 2, "floor": 48, "first": 1, "decisions": 3, "stand_in": 2, "first_stand_in": 2}])

    def test_the_two_act_3_bosses_are_two_fights_and_a_hallway_fight_is_none(self):
        emu = FakeEmu([view_of(obs(floor=47, combat=True))])
        ls = lockstep.Lockstep(emu, lambda m: 0)
        hall = obs(floor=47, combat=True)
        hall["act"] = 2
        for m in (msg("combat", hall), msg("combat", boss_obs(48)), msg("combat", boss_obs(49)), msg("combat", boss_obs(49))):
            ls.decide(m)
        self.assertEqual([(f["floor"], f["decisions"]) for f in ls.stats["boss_fights"]], [(48, 1), (49, 2)])

    def test_a_run_with_no_boss_fight_carries_no_key(self):
        emu = FakeEmu([view_of(obs())])
        ls = lockstep.Lockstep(emu, lambda m: 0)
        ls.decide(msg("map", obs()))
        self.assertNotIn("boss_fights", ls.stats)


class ResumedAtTests(unittest.TestCase):
    def test_a_recovered_loss_says_where_and_by_which_re_anchor(self):
        c = obs(combat=True, floor=5)
        emu = FakeEmu([view_of(obs(hp=50, combat=True, floor=5))], reanchor_ok=[False, True])
        ls = lockstep.Lockstep(emu, lambda m: 0)
        for m in (msg("combat", c), msg("combat", c), msg("reward", obs(floor=5)), msg("map", obs(floor=6))):
            ls.decide(m)
        rec = ls.log.records[0]
        self.assertEqual(rec["resumed_at"], {"decision": 3, "floor": 6, "plan": "room_entry"})
        self.assertEqual(rec["not_ranked_span"], {"first": 0, "last": 2, "count": 3})

    def test_a_loss_never_recovered_has_none(self):
        c = obs(combat=True)
        emu = FakeEmu([view_of(obs(hp=50, combat=True))])
        ls = lockstep.Lockstep(emu, lambda m: 0)
        for m in (msg("combat", c), msg("combat", c)):
            ls.decide(m)
        self.assertNotIn("resumed_at", ls.log.records[0])

if __name__ == "__main__":
    unittest.main()
