"""The Act 1 elite bonus on the route screen (D60 item 9, lane I; rl/imitation_chooser.py, rl/macro_recipe.py, rl/gen_loop.py).

Pinned: the recipe part `+elite:<bonus>@<hp per cent>` (one canonical form, after the stakes schedule and the hold, a recipe of its own in the ledger); the
elite count of a route option (the most elite rooms on any path through the room, the room itself counted); the bonus applied only to a route screen in
Act 1 and only while HP is at or above the fraction of max HP, never to another screen; and, with the option absent, the policy's choices on twenty stored
route screens exactly what they were before the option existed (three policies: the student at a lambda above 0, at 0, and none). Agent.json's `route_elite`
read back by the loader (refused when malformed or beside a plain macro), the candidate builder (the base's files and recipe kept, the option added), and the
emulator arm's identity naming it.

No game: hand-built act maps, the real judgement and prior weights where the files are present.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import gen0  # noqa: E402
import gen_loop  # noqa: E402
import human_prior as hp  # noqa: E402
import human_prior_infer as hpi  # noqa: E402
import imitation_chooser as ic  # noqa: E402
import macro_recipe as mr  # noqa: E402
from route_elite_screens import ELITE, MONSTER, REST, act_map, route_screens  # noqa: E402
from run_train import features as F  # noqa: E402
from run_train import models  # noqa: E402

from tests.python.test_gen_loop import student_world  # noqa: E402
from tests.python.test_human_prior import _find_ids, synthetic_run  # noqa: E402
from tests.python.test_imitation_chooser import REAL_JUDGE, REAL_WEIGHTS  # noqa: E402

# What the policy chose on the twenty screens of `route_screens`, written before the option existed: [index, V of each option (4 places), options], per policy.
GOLDEN = {
    "student": [
        [0, [17.6253, 16.7088, 17.1751], 3],
        [2, [16.7082, 16.7082, 16.7082], 3],
        [1, [9.529, 12.1552], 2],
        [0, [11.504, 11.3407, 10.9323], 3],
        [1, [15.1488, 15.3439], 2],
        [0, [10.9555, 10.9555], 2],
        [0, [10.7394, 10.2124], 2],
        [0, [12.5493, 11.1401, 9.8016], 3],
        [1, [11.0268, 17.1445], 2],
        [1, [12.8906, 17.0943], 2],
        [0, [14.1449, 13.4463, 11.0329], 3],
        [1, [16.2108, 16.8542], 2],
        [1, [18.4105, 18.628], 2],
        [0, [11.1918, 10.7951], 2],
        [1, [11.8668, 13.1195], 2],
        [0, [12.8355, 12.8355, 12.8355], 3],
        [0, [17.4747, 17.4613], 2],
        [1, [15.268, 16.6676], 2],
        [2, [10.3962, 10.1937, 11.5757], 3],
        [1, [17.2776, 17.6451], 2],
    ],
    "student0": [
        [0, [17.6253, 16.7088, 17.1751], 3],
        [0, [16.7082, 16.7082, 16.7082], 3],
        [1, [9.529, 12.1552], 2],
        [0, [11.504, 11.3407, 10.9323], 3],
        [1, [15.1488, 15.3439], 2],
        [0, [10.9555, 10.9555], 2],
        [0, [10.7394, 10.2124], 2],
        [0, [12.5493, 11.1401, 9.8016], 3],
        [1, [11.0268, 17.1445], 2],
        [1, [12.8906, 17.0943], 2],
        [0, [14.1449, 13.4463, 11.0329], 3],
        [1, [16.2108, 16.8542], 2],
        [1, [18.4105, 18.628], 2],
        [0, [11.1918, 10.7951], 2],
        [1, [11.8668, 13.1195], 2],
        [0, [12.8355, 12.8355, 12.8355], 3],
        [0, [17.4747, 17.4613], 2],
        [1, [15.268, 16.6676], 2],
        [2, [10.3962, 10.1937, 11.5757], 3],
        [1, [17.2776, 17.6451], 2],
    ],
    "plain": [
        [0, [17.6253, 16.7088, 17.1751], 3],
        [0, [16.7082, 16.7082, 16.7082], 3],
        [1, [9.529, 12.1552], 2],
        [0, [11.504, 11.3407, 10.9323], 3],
        [1, [15.1488, 15.3439], 2],
        [0, [10.9555, 10.9555], 2],
        [0, [10.7394, 10.2124], 2],
        [0, [12.5493, 11.1401, 9.8016], 3],
        [1, [11.0268, 17.1445], 2],
        [1, [12.8906, 17.0943], 2],
        [0, [14.1449, 13.4463, 11.0329], 3],
        [1, [16.2108, 16.8542], 2],
        [1, [18.4105, 18.628], 2],
        [0, [11.1918, 10.7951], 2],
        [1, [11.8668, 13.1195], 2],
        [0, [12.8355, 12.8355, 12.8355], 3],
        [0, [17.4747, 17.4613], 2],
        [1, [15.268, 16.6676], 2],
        [2, [10.3962, 10.1937, 11.5757], 3],
        [1, [17.2776, 17.6451], 2],
    ],
}

BASE = "vrun+pi_h:lambda=0.5:pi_h=aaaaaaaa"
STAKES = "a3-v1@h1e4b16"


class RecipeTests(unittest.TestCase):
    def test_the_part_is_one_canonical_form_after_the_stakes_and_the_hold(self):
        self.assertEqual("+elite:1@60", mr.elite_part(1.0, 60))
        self.assertEqual("+elite:0.5@60", mr.elite_part(0.5, 60))
        self.assertEqual("+elite:2.5@70", mr.elite_part(2.5, 70))
        self.assertEqual(BASE + "+elite:1@60", mr.full_recipe(BASE, None, None, (1.0, 60)))
        self.assertEqual(BASE + f"+stakes:{STAKES}+elite:1@60", mr.full_recipe(BASE, STAKES, None, (1.0, 60)))
        self.assertEqual(BASE + f"+stakes:{STAKES}+hold:30+elite:1@60", mr.full_recipe(BASE, STAKES, 30, (1.0, 60)))
        self.assertEqual(BASE + f"+stakes:{STAKES}+hold:30", mr.full_recipe(BASE, STAKES, 30))  # without it the recipe is what it was
        self.assertEqual(BASE, mr.full_recipe(BASE))

    def test_the_recipe_reads_back_and_the_old_split_is_unchanged(self):
        for stakes, hold, elite in ((None, None, (1.0, 60)), (STAKES, None, (0.5, 60)), (STAKES, 30, (2.0, 75)), (STAKES, 30, None), (None, None, None)):
            full = mr.full_recipe(BASE, stakes, hold, elite)
            self.assertEqual((BASE, stakes, hold), mr.split_recipe(full), full)
            self.assertEqual(elite, mr.parse_elite(full), full)
        self.assertIsNone(mr.parse_elite(""))
        self.assertIsNone(mr.parse_elite("vrun"))

    def test_a_bad_bonus_or_fraction_is_refused(self):
        for bonus in (0, -1, math.inf, math.nan, 51, True):
            with self.assertRaises(ValueError, msg=str(bonus)):
                mr.elite_part(bonus, 60)
        for pct in (0, 101, 59.5, -3, True):
            with self.assertRaises(ValueError, msg=str(pct)):
                mr.elite_part(1.0, pct)

    def test_a_recipe_with_the_part_is_a_recipe_of_its_own_in_the_ledger(self):
        plain = mr.full_recipe(BASE, STAKES)
        a, b = mr.full_recipe(BASE, STAKES, None, (1.0, 60)), mr.full_recipe(BASE, STAKES, None, (2.0, 60))
        self.assertEqual(3, len({plain, a, b}))


def tiny_map():
    """Rows 0 to 4 and the boss: (3,0) ancient -> (2,1) M, (4,1) E; (2,1) -> (2,2) M; (4,1) -> (4,2) E, (5,2) R; (2,2) -> (2,3) E; (4,2) -> (4,3) E; (5,2) -> (5,3) M;
    row 3 -> boss (3,4)."""
    n = F.MapNode
    return (n(3, 0, 8, ((2, 1), (4, 1))), n(2, 1, MONSTER, ((2, 2),)), n(4, 1, ELITE, ((4, 2), (5, 2))), n(2, 2, MONSTER, ((2, 3),)),
            n(4, 2, ELITE, ((4, 3),)), n(5, 2, REST, ((5, 3),)), n(2, 3, ELITE, ((3, 4),)), n(4, 3, ELITE, ((3, 4),)), n(5, 3, MONSTER, ((3, 4),)),
            n(3, 4, 6, ()))


class EliteCountTests(unittest.TestCase):
    def test_a_room_counts_the_most_elites_on_any_path_through_it_itself_included(self):
        m = tiny_map()
        got = {(x.col, x.row): ic.elites_ahead(m, (x.col, x.row)) for x in m}
        self.assertEqual({(3, 0): 3, (2, 1): 1, (4, 1): 3, (2, 2): 1, (4, 2): 2, (5, 2): 0, (2, 3): 1, (4, 3): 1, (5, 3): 0, (3, 4): 0}, got)

    def test_a_room_off_the_map_counts_nothing(self):
        self.assertEqual(0, ic.elites_ahead(tiny_map(), (6, 6)))

    def test_the_seeded_maps_have_elites_to_steer_by(self):
        counts = [ic.elites_ahead(act_map(s, F), (3, 0)) for s in range(1, 9)]
        self.assertTrue(all(c >= 1 for c in counts), counts)


class ScreenSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hp.stack()
        cls.t = hp.Tables()
        cls.nets = models.load_numpy(REAL_JUDGE)
        cls.prior = hpi.PriorScorer.load(REAL_WEIGHTS)
        cls.tmp = tempfile.TemporaryDirectory()
        with np.load(REAL_WEIGHTS, allow_pickle=False) as z:
            params = {k: z[k] for k in z.files if k != "meta"}
        cls.same = Path(cls.tmp.name) / "same.npz"
        hpi.write_student(cls.same, params, kinds=("card", "route", "shop", "reward"), prior_sha256=cls.prior.sha256)
        specs = hp.run_decisions(synthetic_run(_find_ids(False, 1)[0]), cls.t, Counter(), Counter(), Counter())
        cls.view = hp.spec_view(next(s for s in specs if s["kind"] == "card"))
        cls.screens = route_screens(cls.view, F)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def policy(self, which: str, **kw):
        extra = {"student": {"student": self.same, "student_lams": {"route": (0.3715, 2.0, 2.0)}},
                 "student0": {"student": self.same, "student_lams": {"route": (0.0, 2.0, 2.0)}}, "plain": {}}[which]
        return ic.BlendedPolicy(REAL_JUDGE, self.prior, 0.5, W=53.0, **extra, **kw)


class OptionAbsentTests(ScreenSetup):
    def test_there_are_twenty_screens_with_a_choice(self):
        self.assertEqual(20, len(self.screens))
        self.assertTrue(all(len(d.options) >= 2 for _, d in self.screens))

    def test_with_the_option_absent_the_choices_on_twenty_stored_screens_are_what_they_were(self):
        for which in ("student", "student0", "plain"):
            for kw in ({}, {"route_elite": None}):
                pol = self.policy(which, **kw)
                got = []
                for v, d in self.screens:
                    pk = pol.pick(v, d, None)
                    got.append([int(pk.index), [round(float(x), 4) for x in pk.values], len(d.options)])
                self.assertEqual(GOLDEN[which], got, f"{which} {kw}")

    def test_the_golden_choices_are_not_all_the_same_option(self):
        for which in ("student", "student0", "plain"):
            self.assertGreater(len({r[0] for r in GOLDEN[which]}), 1, which)


class BonusRuleTests(ScreenSetup):
    def with_hp(self, view, hp):
        return dataclasses.replace(view, hp=hp, max_hp=100)

    def test_the_bonus_is_the_value_times_the_elites_on_each_options_best_path(self):
        pol = self.policy("student", route_elite=(1.5, 60))
        v = self.with_hp(self.screens[0][0], 90)
        d = self.screens[0][1]
        got = pol.route_bonus(v, d)
        want = [1.5 * ic.elites_ahead(v.map_nodes, o.node[:2]) for o in d.options]
        self.assertEqual(want, got)

    def test_it_applies_at_the_threshold_and_not_below_it(self):
        pol = self.policy("student", route_elite=(1.0, 60))
        v, d = self.screens[0]
        self.assertIsNotNone(pol.route_bonus(self.with_hp(v, 60), d))
        self.assertIsNone(pol.route_bonus(self.with_hp(v, 59), d))
        self.assertIsNotNone(pol.route_bonus(self.with_hp(v, 100), d))

    def test_it_applies_in_act_1_only(self):
        pol = self.policy("student", route_elite=(1.0, 60))
        v, d = self.screens[0]
        v = self.with_hp(v, 100)
        self.assertIsNotNone(pol.route_bonus(dataclasses.replace(v, floor=17), d))  # the last Act 1 floor
        for floor in (18, 25, 40):
            self.assertIsNone(pol.route_bonus(dataclasses.replace(v, floor=floor), d), floor)

    def test_it_applies_to_a_route_screen_and_to_no_other(self):
        pol = self.policy("student", route_elite=(1.0, 60))
        v, d = self.screens[0]
        v = self.with_hp(v, 100)
        for dtype in ("card_reward", "rest", "shop", "event", "neow", "rewards"):
            self.assertIsNone(pol.route_bonus(v, F.Decision(dtype, d.options)), dtype)
        self.assertIsNone(pol.route_bonus(v, F.Decision("map", d.options[:1])))  # one room to go to is not a decision
        self.assertIsNone(self.policy("student").route_bonus(v, d))  # the option absent

    def test_the_other_screens_are_decided_as_without_the_option(self):
        v, d = self.screens[0]
        v = self.with_hp(v, 100)
        for which in ("student", "plain"):
            a = self.policy(which).pick(v, F.Decision("shop", d.options), None)
            b = self.policy(which, route_elite=(5.0, 60)).pick(v, F.Decision("shop", d.options), None)
            self.assertEqual((a.index, a.values), (b.index, b.values), which)

    def test_a_large_bonus_takes_the_room_with_the_most_elites_on_its_path_on_every_path_of_deciding(self):
        for which in ("student", "student0", "plain"):
            pol = self.policy(which, route_elite=(50.0, 1))
            moved = 0
            for v, d in self.screens:
                counts = [ic.elites_ahead(v.map_nodes, o.node[:2]) for o in d.options]
                pk = pol.pick(v, d, None)
                if which == "plain" or which == "student0":  # the judgement alone, no guard: the most elites, ties by V
                    top = [i for i, c in enumerate(counts) if c == max(counts)]
                    self.assertIn(pk.index, top, which)
                    if len(top) == 1:
                        moved += GOLDEN[which][self.screens.index((v, d))][0] != pk.index
                else:  # the student's blend: the guard may rule an option out, the rest follow the count
                    rows = hpi.option_rows(self.nets, v, d.options)
                    logp = hpi.log_softmax(hpi.load_student(self.same).row_logits(rows))
                    allowed = [i for i in range(len(d.options)) if math.exp(logp[i]) >= ic.guard_prob_for(len(d.options), True)] or list(range(len(d.options)))
                    best = max(counts[i] for i in allowed)
                    if not pol.last["guard"]:
                        self.assertEqual(best, counts[pk.index])
                    moved += GOLDEN[which][self.screens.index((v, d))][0] != pk.index
            self.assertGreater(moved, 0, which)  # the bonus moved a pick on some screen: the check is not vacuous

    def test_the_recorded_values_are_the_judgements_and_not_the_judgement_plus_the_bonus(self):
        for which in ("student", "student0", "plain"):
            a, b = self.policy(which), self.policy(which, route_elite=(3.0, 1))
            for v, d in self.screens[:6]:
                self.assertEqual(a.pick(v, d, None).values, b.pick(v, d, None).values, which)

    def test_below_the_threshold_the_choices_are_the_golden_ones(self):
        for which in ("student", "student0", "plain"):
            pol = self.policy(which, route_elite=(50.0, 100))
            for k, (v, d) in enumerate(self.screens):
                v = dataclasses.replace(v, hp=79, max_hp=80)  # under the whole of max HP
                a = self.policy(which).pick(v, d, None)
                b = pol.pick(v, d, None)
                self.assertEqual((a.index, a.values), (b.index, b.values), (which, k))

    def test_a_policy_with_the_option_describes_it_and_without_it_describes_as_it_did(self):
        self.assertNotIn("route_elite", self.policy("student").describe())
        self.assertEqual([1.0, 60], self.policy("student", route_elite=(1.0, 60)).describe()["route_elite"])

    def test_a_bad_option_is_refused(self):
        for bad in ((0.0, 60), (1.0, 0), (1.0, 101), (-1.0, 60), (1.0,), (math.nan, 60)):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.policy("student", route_elite=bad)

    def test_exploring_and_sampling_a_route_screen_with_the_bonus_stay_in_the_guarded_set_and_replay(self):
        v, d = self.screens[3]
        pol = self.policy("student", temperature=0.5, eps=0.0, route_elite=(2.0, 1))
        rng = np.random.default_rng(0)
        picks = {pol.pick(v, d, rng).index for _ in range(40)}
        self.assertTrue(picks <= set(range(len(d.options))))
        sampled = self.policy("student", pick=mr.PICK_SAMPLE, route_elite=(2.0, 1))
        sampled.begin_run("SEED")
        first = [sampled.pick(v, d, None).index for _ in range(6)]
        sampled.begin_run("SEED")
        self.assertEqual(first, [sampled.pick(v, d, None).index for _ in range(6)])


class AgentIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.agents, _base, self.prior, self.student, self.human = student_world(self.root)
        self.inc = gen_loop.build_candidate(self.prior, self.agents / "current.json", self.agents, 6, why="incumbent", export=False, tag="inc",
                                            student=self.student, student_lams={"route": (0.3715, 2.0, 2.0), "shop": (0.2858, 0.2311, 0.2311)})
        (self.inc / "run_student_legs.json").write_text(json.dumps({"legs": "of the incumbent"}), encoding="utf-8")
        (self.agents / "current.json").write_text(json.dumps({"generation": 6, "dir": str(self.inc)}))

    def build(self, **kw):
        kw.setdefault("tag", "elite-probe")
        return gen_loop.build_route_elite_candidate(self.inc, self.agents, 7, why="test", export=False, bonus=1.0, hp_frac=0.6, **kw)

    def test_an_agent_without_the_field_is_what_it_was(self):
        a = gen_loop.load_agent(self.inc)
        self.assertIsNone(a.route_elite)
        self.assertNotIn("route_elite", a.describe())
        self.assertNotIn("+elite", a.recipe())

    def test_the_candidate_is_the_base_with_the_option_and_nothing_else(self):
        cand = self.build()
        base, got = gen_loop.load_agent(self.inc), gen_loop.load_agent(cand)
        self.assertEqual((1.0, 60), got.route_elite)
        self.assertEqual(base.macro, got.macro)  # the same chooser, number for number
        self.assertEqual(base.recipe() + "+elite:1@60", got.recipe())
        self.assertEqual(base.student_lams, got.student_lams)
        self.assertEqual((base.stakes, base.hold, base.lam, base.lam_acts), (got.stakes, got.hold, got.lam, got.lam_acts))
        for name in ("combat_policy.pt", "combat_net.pt", "run_value.npz", "human_prior.npz", "run_student.npz", "run_student_legs.json"):
            self.assertEqual((self.inc / name).read_bytes(), (cand / name).read_bytes(), name)
        meta = json.loads((cand / "agent.json").read_text())
        self.assertEqual({"bonus": 1.0, "hp_frac": 0.6, "act": 1}, {k: meta["route_elite"][k] for k in ("bonus", "hp_frac", "act")})
        self.assertEqual(got.recipe(), meta["recipe"])
        self.assertEqual(base.describe()["prior_sha256"], got.describe()["prior_sha256"])
        self.assertEqual([1.0, 60], got.describe()["route_elite"])
        self.assertEqual((7, 6), (meta["generation"], meta["built_on"]))
        self.assertEqual(json.loads((self.inc / "agent.json").read_text())["student"]["sha256"], meta["student"]["sha256"])

    def test_the_policy_the_agent_plays_carries_the_option(self):
        a = gen_loop.load_agent(self.build())
        self.assertEqual((1.0, 60), a.route_elite)
        with_option = dataclasses.replace(a)
        self.assertEqual((1.0, 60), with_option.route_elite)

    def test_the_candidate_is_a_recipe_of_its_own_with_the_same_identity_pair(self):
        base, got = gen_loop.load_agent(self.inc), gen_loop.load_agent(self.build())
        self.assertNotEqual(gen_loop.candidate_recipe(base), gen_loop.candidate_recipe(got))
        self.assertEqual(gen_loop.candidate_identity(base), gen_loop.candidate_identity(got))

    def test_a_malformed_option_or_recipe_is_refused_at_load(self):
        cand = self.build()
        meta = json.loads((cand / "agent.json").read_text())
        for name, change in (("bonus", {"route_elite": {"bonus": 0, "hp_frac": 0.6, "act": 1}}),
                             ("frac", {"route_elite": {"bonus": 1.0, "hp_frac": 0.605, "act": 1}}),
                             ("frac0", {"route_elite": {"bonus": 1.0, "hp_frac": 0.0, "act": 1}}),
                             ("act", {"route_elite": {"bonus": 1.0, "hp_frac": 0.6, "act": 2}}),
                             ("nobonus", {"route_elite": {"hp_frac": 0.6, "act": 1}}),
                             ("recipe", {"route_elite": {"bonus": 2.0, "hp_frac": 0.6, "act": 1}})):  # the recipe names a bonus of 1
            (cand / "agent.json").write_text(json.dumps({**meta, **change}))
            with self.assertRaises(ValueError, msg=name):
                gen_loop.load_agent(cand)
        (cand / "agent.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "route_elite"}))
        with self.assertRaises(ValueError):  # the recipe still names the option, the fields do not
            gen_loop.load_agent(cand)

    def test_a_plain_vrun_macro_cannot_carry_it(self):
        root = self.root / "plain"
        root.mkdir()
        agents, base, prior, student, human = student_world(root, blended=False)
        meta = json.loads((base / "agent.json").read_text())
        meta.update(route_elite={"bonus": 1.0, "hp_frac": 0.6, "act": 1}, recipe="vrun+elite:1@60")
        (base / "agent.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "blended"):
            gen_loop.load_agent(base)

    def test_what_cannot_be_built_is_refused_and_leaves_no_folder(self):
        for kw in ({"bonus": 0.0}, {"hp_frac": 0.0}, {"hp_frac": 1.5}, {"bonus": math.inf}):
            with self.assertRaises(ValueError, msg=str(kw)):
                gen_loop.build_route_elite_candidate(self.inc, self.agents, 7, why="t", export=False, **{"bonus": 1.0, "hp_frac": 0.6, **kw})
        self.assertFalse((self.agents / "gen7").exists())
        with self.assertRaises(ValueError):  # a base that plays no student has nothing to keep
            root = self.root / "plain"
            root.mkdir()
            agents, base, *_ = student_world(root)
            gen_loop.build_route_elite_candidate(base, agents, 7, why="t", export=False, bonus=1.0, hp_frac=0.6)
        self.assertFalse((self.root / "plain" / "agent" / "gen7").exists())

    def test_a_candidate_built_on_one_with_the_option_keeps_it_unless_told_otherwise(self):
        cand = self.build()
        again = gen_loop.build_route_elite_candidate(cand, self.agents, 8, why="again", export=False, tag="elite-again", bonus=2.0, hp_frac=0.7)
        self.assertEqual((2.0, 70), gen_loop.load_agent(again).route_elite)

    def test_a_student_candidate_built_on_one_with_the_option_keeps_the_option(self):
        cand = self.build()
        kept = gen_loop.build_candidate(self.prior, cand, self.agents, 8, why="next student", export=False, tag="next",
                                        student=self.student, student_lams={"route": (0.5, 0.5, 0.5)})
        got = gen_loop.load_agent(kept)
        self.assertEqual((1.0, 60), got.route_elite)
        self.assertTrue(got.recipe().endswith("+elite:1@60"), got.recipe())
        with self.assertRaises(ValueError):  # a plain build on a base with the option would drop it silently: refused
            gen_loop.build_candidate(self.prior, cand, self.agents, 9, why="plain", export=False, tag="plainer")

    def test_the_emulator_arm_of_another_option_does_not_match(self):
        a, b = gen_loop.load_agent(self.inc), gen_loop.load_agent(self.build())
        with tempfile.TemporaryDirectory() as d:
            arm = Path(d)
            (arm / "runs.jsonl").write_text("{}\n")
            (arm / "summary.json").write_text(json.dumps({"agent": a.describe()}))
            self.assertTrue(gen0.emu_arm_match(arm, a.describe())[0])
            self.assertFalse(gen0.emu_arm_match(arm, b.describe())[0])
            (arm / "summary.json").write_text(json.dumps({"agent": b.describe()}))
            self.assertTrue(gen0.emu_arm_match(arm, b.describe())[0])
            self.assertFalse(gen0.emu_arm_match(arm, a.describe())[0])


class BuiltPolicyTests(ScreenSetup):
    def test_the_agents_policy_is_built_with_the_option(self):
        sentinel = {}

        class Spy(ic.BlendedPolicy):
            def __init__(self, *a, **k):
                sentinel.update(k)
                super().__init__(*a, **k)

        agent = gen_loop.Agent(macro=mr.macro_name(0.5, self.prior.sha256), macro_npz=str(REAL_JUDGE), human_prior=str(REAL_WEIGHTS), lam=0.5, route_elite=(1.5, 70))
        with mock.patch.object(ic, "BlendedPolicy", Spy):
            agent.macro_policy()
        self.assertEqual((1.5, 70), sentinel.get("route_elite"))
        sentinel.clear()
        with mock.patch.object(ic, "BlendedPolicy", Spy):
            dataclasses.replace(agent, route_elite=None).macro_policy()
        self.assertNotIn("route_elite", sentinel)


if __name__ == "__main__":
    unittest.main()
