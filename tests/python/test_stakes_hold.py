"""The stakes schedule, the hallway potion hold and the sampled chooser (2026-11-06, docs/rl/STAKES_AND_HOLD.md).

Pinned: a search config's stakes multiplier reaches the search per fight kind (hallway x1, elite x4, boss x16 of a3-v1; at a boss it is
the play-or-deck measurement's x16 arm exactly), the config names and the recipe strings that carry it, an agent's own schedule and hold
read from its agent.json (and refused when malformed), the ledger counting attempts per recipe (a schedule and a hold are recipes of
their own), the lab's paths (the emulator arm, the lockstep's real-game player, the arm cache, the build command) honouring the
schedule; the hold rule on synthetic roots (a potion dropped where the fight is safe, kept where it is not) and on real fights; and the
chooser's `sample` and `skipcal` pick modes (reproducible draws seeded by the run seed and the floor; the skip bias search).

No game: synthetic positions, stub arms, the emulator's library for the real-fight checks.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import fight_gym  # noqa: E402
import gen0  # noqa: E402
import gen_loop  # noqa: E402
import imitation_chooser as ic  # noqa: E402
import live_search  # noqa: E402
import macro_recipe as mr  # noqa: E402
import run_fields  # noqa: E402
import search_configs as sc  # noqa: E402
import skip_calibrate as skc  # noqa: E402

from tests.python.test_funnel import StubArms, make_block, stub_candidate  # noqa: E402
from tests.python.test_gen12_plumbing import PRIOR_SHA, legacy_tsv  # noqa: E402
from tests.python.test_imitation_chooser import blended_stub, lp, random_weights  # noqa: E402
from tests.python.test_live_search import (  # noqa: E402
    BLOCK_POTION,
    NIBBITS_WEAK,
    STRIKE,
    Fight,
)

A3 = sc.get("a3-v1")
STAKES = "a3-v1@h1e4b16"
SCHEDULE = {"base": "a3-v1", "mult": {"hall": 1, "elite": 4, "boss": 16}}


def pos(kind: str = "play", fight: str = "hall", hp_frac: float = 0.9, n_moves: int = 4) -> sc.Position:
    return sc.Position(kind=kind, n_moves=n_moves, hp_frac=hp_frac, hard=fight != "hall", fight=fight)


class RecipeGrammarTests(unittest.TestCase):
    def test_a_stakes_tag_names_the_base_and_the_three_multipliers(self):
        self.assertEqual(STAKES, mr.stakes_tag("a3-v1", {"hall": 1, "elite": 4, "boss": 16}))
        self.assertEqual(STAKES, mr.stakes_tag("a3-v1", (1, 4, 16)))
        self.assertEqual(("a3-v1", {"hall": 1, "elite": 4, "boss": 16}), mr.parse_stakes_tag(STAKES))
        self.assertEqual(("a3-v1-k", {"hall": 2, "elite": 3, "boss": 5}), mr.parse_stakes_tag("a3-v1-k@h2e3b5"))
        for bad in ("a3-v1", "a3-v1@h1e4", "@h1e4b16", "a3-v1@h0e4b16", "a3-v1@h1e4b99"):
            with self.assertRaises(ValueError, msg=bad):
                mr.parse_stakes_tag(bad)
        with self.assertRaises(ValueError):
            mr.stakes_tag("a3-v1", {"hall": 1, "elite": 4.5, "boss": 16})
        with self.assertRaises(ValueError):
            mr.stakes_tag("a3-v1", {"hall": 1, "elite": 4, "boss": 16, "event": 2})
        with self.assertRaises(ValueError):
            mr.stakes_tag("a3-v1", (1, 4))

    def test_a_config_name_carries_the_schedule_and_the_hold(self):
        self.assertEqual("a3-v1", mr.config_name("a3-v1"))
        self.assertEqual(STAKES, mr.config_name("a3-v1", (1, 4, 16)))
        self.assertEqual("a3-v1+hold30", mr.config_name("a3-v1", None, 30))
        self.assertEqual("a3-v1@h1e4b16+hold30", mr.config_name("a3-v1", (1, 4, 16), 30))
        for name, parsed in (("a3-v1", ("a3-v1", None, None)), ("flat-60", ("flat-60", None, None)),
                             ("a3-v1+hold30", ("a3-v1", None, 30)), ("a3-v1@h1e4b16+hold30", ("a3-v1", {"hall": 1, "elite": 4, "boss": 16}, 30))):
            self.assertEqual(parsed, mr.parse_config_name(name))
        for bad in (0, 100, 12.5, True):
            with self.assertRaises(ValueError, msg=str(bad)):
                mr.hold_pct(bad)

    def test_the_recipe_adds_the_schedule_and_the_hold_to_the_macro(self):
        self.assertEqual("", mr.full_recipe("vrun"))  # an agent with neither is the recipe it always was
        self.assertEqual("", mr.full_recipe("rules"))
        blended = mr.macro_name(0.5, "ab" * 32)
        self.assertEqual(blended, mr.full_recipe(blended))
        self.assertEqual("vrun+stakes:a3-v1@h1e4b16", mr.full_recipe("vrun", STAKES))
        self.assertEqual("vrun+stakes:a3-v1@h1e4b16+hold:30", mr.full_recipe("vrun", STAKES, 30))
        self.assertEqual("vrun+hold:30", mr.full_recipe("vrun", None, 30))
        self.assertEqual(blended + "+stakes:a3-v1@h1e4b16+hold:30", mr.full_recipe(blended, STAKES, 30))
        for macro in ("vrun", blended):
            for stakes in (None, STAKES):
                for hold in (None, 30):
                    if stakes or hold:
                        self.assertEqual((macro, stakes, hold), mr.split_recipe(mr.full_recipe(macro, stakes, hold)))
        self.assertEqual(("", None, None), mr.split_recipe(""))
        # three recipes, three strings: the schedule and the hold each make a recipe of their own
        self.assertEqual(3, len({"", mr.full_recipe("vrun", STAKES), mr.full_recipe("vrun", STAKES, 30), mr.full_recipe("vrun", None, 30)}) - 1)

    def test_the_pick_mode_is_part_of_the_macro_string(self):
        sha = "2ebfe6d7f25e" + "0" * 52
        base = mr.macro_name(0.5, sha)
        self.assertEqual("vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7", base)  # the argmax names nothing: every earlier recipe is unchanged
        self.assertEqual(base + ":pick=sample", mr.macro_name(0.5, sha, "sample"))
        self.assertEqual(base + ":pick=skipcal0.35", mr.macro_name(0.5, sha, "skipcal", 0.35))
        self.assertEqual(base + ":pick=skipcal-0.2", mr.macro_name(0.5, sha, "skipcal", -0.2))
        self.assertEqual(base + ":pick=skipcal0", mr.macro_name(0.5, sha, "skipcal", 0.0))
        for name, want in ((base, ("argmax", 0.0)), (base + ":pick=sample", ("sample", 0.0)), (base + ":pick=skipcal0.35", ("skipcal", 0.35)),
                           (base + ":pick=skipcal-0.2", ("skipcal", -0.2))):
            self.assertEqual(want, mr.parse_pick(name))
            self.assertEqual((0.5, "2ebfe6d7"), mr.parse_macro(name))  # lambda and the prior are read through the pick part
            self.assertTrue(mr.is_blended(name))
        self.assertEqual(4, len({mr.macro_name(0.5, sha, "argmax"), mr.macro_name(0.5, sha, "sample"), mr.macro_name(0.5, sha, "skipcal", 0.3),
                                 mr.macro_name(0.5, sha, "skipcal", 0.4)}))
        with self.assertRaises(ValueError):
            mr.macro_name(0.5, sha, "sample", 0.3)  # a bias belongs to skipcal
        with self.assertRaises(ValueError):
            mr.macro_name(0.5, sha, "greedy")
        with self.assertRaises(ValueError):
            mr.parse_pick(base + ":pick=greedy")


class StakesScheduleTests(unittest.TestCase):
    """The multiplier reaches the search per fight kind."""

    cfg = sc.get(STAKES)

    def test_the_name_reads_back_as_the_base_with_a_schedule(self):
        self.assertEqual(STAKES, self.cfg.name)
        self.assertEqual((1, 4, 16), self.cfg.stakes)
        self.assertIsNone(self.cfg.potion_hold)
        self.assertEqual((A3.sims, A3.c, A3.min_child_sims, A3.schedule), (self.cfg.sims, self.cfg.c, self.cfg.min_child_sims, self.cfg.schedule))
        self.assertIsNone(A3.stakes)  # the base config itself is as it was
        self.assertEqual(self.cfg, sc.get(STAKES))
        hash(self.cfg)  # frozen and hashable, like every config
        with self.assertRaises(KeyError):
            sc.get("no-such-config@h1e4b16")
        with self.assertRaises(ValueError):
            sc.get("a3-v1@h1e4")
        with self.assertRaises(ValueError):
            sc.with_options(self.cfg, (1, 4, 16))  # no stacking a schedule on a schedule

    def test_each_fight_kind_gets_its_own_multiple_of_the_base(self):
        for kind, base in (("play", 30), ("target", 120), ("select", 120)):
            self.assertEqual(base * 1, self.cfg.sims_for(pos(kind, "hall")), kind)
            self.assertEqual(base * 4, self.cfg.sims_for(pos(kind, "elite")), kind)
            self.assertEqual(base * 16, self.cfg.sims_for(pos(kind, "boss")), kind)
        # the hallway is a3-v1 exactly, whatever the position (HP, move count)
        for p in (pos("play", "hall", 0.2, 60), pos("target", "hall", 0.9, 3)):
            self.assertEqual(A3.sims_for(p), self.cfg.sims_for(p))
        self.assertEqual(A3.sims, self.cfg.sims_for(None))  # no position: the base count

    def test_a_boss_fight_is_the_play_or_deck_x16_arm_and_an_elite_its_x4(self):
        for mult, fight in ((16, "boss"), (4, "elite")):
            arm = fight_gym.arm_settings(f"x{mult}")["cfg"]
            for kind in ("play", "target", "select"):
                for hp_frac in (0.9, 0.3):
                    p = pos(kind, fight, hp_frac, 5)
                    self.assertEqual(arm.sims_for(p), self.cfg.sims_for(p), (fight, kind, hp_frac))
                    self.assertEqual(arm.search_kwargs(p, env_prior=False), self.cfg.search_kwargs(p, env_prior=False), (fight, kind))

    def test_the_safety_clock_and_the_clamp_scale_with_the_multiplier(self):
        self.assertEqual(A3.time_budget_s, self.cfg.search_kwargs(pos(fight="hall"), env_prior=False)["time_budget_s"])
        self.assertEqual(4 * A3.time_budget_s, self.cfg.search_kwargs(pos(fight="elite"), env_prior=False)["time_budget_s"])
        self.assertEqual(16 * A3.time_budget_s, self.cfg.search_kwargs(pos(fight="boss"), env_prior=False)["time_budget_s"])
        tight = dataclasses.replace(sc.flat(10), stakes=(1, 1, 8))  # a flat config has no schedule: the stakes still multiply
        self.assertEqual(80, tight.sims_for(pos(fight="boss")))
        clamped = sc.SearchConfig("t", 60, schedule=sc.Schedule(high=240, hi=200, lo=10), stakes=(1, 1, 4))
        self.assertEqual(200, clamped.sims_for(pos(fight="hall", hp_frac=0.1)))  # the clamp's top is the schedule's
        self.assertEqual(800, clamped.sims_for(pos(fight="boss", hp_frac=0.1)))  # ... times the multiplier at a boss

    def test_a_config_without_the_options_passes_the_search_what_it_always_did(self):
        kw = A3.search_kwargs(pos(fight="boss"), env_prior=False)
        self.assertEqual({"max_sims", "min_child_sims", "c", "confirm_pairs", "time_budget_s", "resample"}, set(kw))
        self.assertNotIn("potion_hold", self.cfg.search_kwargs(pos(fight="hall"), env_prior=False))
        self.assertTrue(A3.needs_position)  # a3-v1 has a schedule
        self.assertTrue(self.cfg.needs_position)
        self.assertFalse(sc.flat(60).needs_position)  # a flat config reads no position
        self.assertTrue(sc.get("flat-60+hold30").needs_position)

    def test_the_wall_factor_is_the_largest_multiplier_over_four(self):
        self.assertEqual(4.0, self.cfg.wall_factor())
        self.assertEqual(1.0, A3.wall_factor())
        self.assertEqual(1.0, sc.get("a3-v1@h1e2b3").wall_factor())
        self.assertEqual(1.0, sc.get("a3-v1+hold30").wall_factor())

    def test_a_position_says_which_fight_it_is(self):
        self.assertEqual("hall", sc.Position("play", 3, 0.5, False).fight_kind)
        self.assertEqual("elite", sc.Position("play", 3, 0.5, True).fight_kind)  # a hard fight whose kind is not said reads as an elite
        self.assertEqual("boss", sc.Position("play", 3, 0.5, True, fight="boss").fight_kind)
        with self.assertRaises(ValueError):
            sc.Position("play", 3, 0.5, True, fight="event")

    def test_position_of_reads_the_fight_kind_from_the_node_type(self):
        moves = [live_search._Move(("card", (1, 0, 0, 0), -1), 0, -1, "card", 1)]
        root = SimpleNamespace(obs=[0] * native_size(), hp=lambda: 40)
        with mock.patch.object(live_search, "_world_moves", return_value=moves):
            for node, fight, hard in ((1, "hall", False), (2, "elite", True), (6, "boss", True), (0, "hall", False)):
                p = sc.position_of(root, {"node_type": node, "max_hp": 80, "floor": 5})
                self.assertEqual((fight, hard), (p.fight, p.hard), node)


def native_size() -> int:
    from sts2_gym import native

    return native.OBS_SIZE


# ------------------------------------------------------------------------------------------------------------------ the agent


def vrun_stub(root: Path, name: str, **meta) -> Path:
    """A candidate folder the agent loader accepts, playing the vrun macro, with agent.json keys `meta` added."""
    c = stub_candidate(root, name, PRIOR_SHA, 12)
    (c / "run_value.npz").write_bytes(b"gen9 head")
    (c / "agent.json").write_text(json.dumps({"generation": 12, "macro": "vrun", **meta}))
    return c


class AgentSearchIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_agent_without_the_fields_is_what_it_was(self):
        a = gen_loop.load_agent(vrun_stub(self.root, "plain"))
        self.assertEqual((None, None, ""), (a.stakes, a.hold, a.recipe()))
        self.assertEqual({"generation", "dir", "prior", "prior_sha256", "macro"}, set(a.describe()))
        self.assertEqual("a3-v1", a.config_name("a3-v1"))
        self.assertEqual("flat-60", a.config_name(60))
        self.assertEqual(A3, a.search_config("a3-v1"))
        self.assertEqual(sc.flat(60), a.search_config(60))
        self.assertEqual(1.0, a.wall_factor())
        self.assertEqual("", gen_loop.candidate_recipe(a))

    def test_the_schedule_and_the_hold_are_identity_and_recipe(self):
        c = vrun_stub(self.root, "stakes", search_schedule=SCHEDULE, potion_hold={"hp_frac": 0.3, "fights": "hall"},
                      recipe="vrun+stakes:a3-v1@h1e4b16+hold:30")
        a = gen_loop.load_agent(c)
        self.assertEqual((STAKES, 30), (a.stakes, a.hold))
        self.assertEqual("vrun+stakes:a3-v1@h1e4b16+hold:30", gen_loop.candidate_recipe(a))
        d = a.describe()
        self.assertEqual((STAKES, 30, "vrun"), (d["stakes"], d["hold"], d["macro"]))
        # the schedule's base replaces the config the lab passes; the hold rides along
        for lab_config in ("a3-v1", "a3-v1-k", "gen0-60", 60):
            cfg = a.search_config(lab_config)
            self.assertEqual("a3-v1@h1e4b16+hold30", cfg.name, lab_config)
            self.assertEqual(((1, 4, 16), 0.3), (cfg.stakes, cfg.potion_hold))
        self.assertEqual("a3-v1@h1e4b16+hold30", a.config_name("a3-v1"))
        self.assertEqual(4.0, a.wall_factor())
        # the same agent with only the schedule, or only the hold, is another recipe
        s_only = gen_loop.load_agent(vrun_stub(self.root, "s", search_schedule=SCHEDULE))
        h_only = gen_loop.load_agent(vrun_stub(self.root, "h", potion_hold={"hp_frac": 0.3}))
        self.assertEqual(["vrun+stakes:a3-v1@h1e4b16", "vrun+hold:30"], [s_only.recipe(), h_only.recipe()])
        self.assertEqual("a3-v1-k+hold30", h_only.search_config("a3-v1-k").name)  # a hold-only agent keeps the lab's config
        self.assertEqual(1.0, h_only.wall_factor())
        self.assertNotEqual(s_only.describe(), h_only.describe())

    def test_a_malformed_schedule_hold_or_recipe_is_refused_at_load(self):
        for name, meta in (
            ("base", {"search_schedule": {"base": "nope", "mult": {"hall": 1, "elite": 4, "boss": 16}}}),
            ("mult", {"search_schedule": {"base": "a3-v1", "mult": {"hall": 1, "elite": 4.5, "boss": 16}}}),
            ("kinds", {"search_schedule": {"base": "a3-v1", "mult": {"hall": 1, "elite": 4, "event": 16}}}),
            ("nomult", {"search_schedule": {"base": "a3-v1"}}),
            ("tag", {"search_schedule": {**SCHEDULE, "tag": "a3-v1@h1e1b1"}}),
            ("frac", {"potion_hold": {"hp_frac": 0.305}}),
            ("fights", {"potion_hold": {"hp_frac": 0.3, "fights": "boss"}}),
            ("zero", {"potion_hold": {"hp_frac": 0.0}}),
            ("recipe", {"search_schedule": SCHEDULE, "recipe": "vrun+stakes:a3-v1@h1e2b4"}),
        ):
            with self.assertRaises(ValueError, msg=name):
                gen_loop.load_agent(vrun_stub(self.root, "bad-" + name, **meta))

    def test_the_stakes_agent_plays_its_schedule_in_the_lockstep_players_config_too(self):
        sys.path.insert(0, str(ROOT / "referee"))
        import lockstep

        c = vrun_stub(self.root, "lock", search_schedule=SCHEDULE, potion_hold={"hp_frac": 0.3})
        plain = vrun_stub(self.root, "lockplain")
        with mock.patch.dict(os.environ, {gen_loop.ENV_GEN_DIR: str(c)}):
            self.assertEqual("a3-v1@h1e4b16+hold30", lockstep.search_config("a3-v1").name)
        with mock.patch.dict(os.environ, {gen_loop.ENV_GEN_DIR: str(plain)}):
            self.assertEqual(A3, lockstep.search_config("a3-v1"))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(gen_loop.ENV_GEN_DIR, None)
            self.assertEqual(A3, lockstep.search_config("a3-v1"))  # generation 0: no agent, no schedule

    def test_the_emulator_arm_and_the_arm_cache_name_the_schedule(self):
        c = gen_loop.load_agent(vrun_stub(self.root, "arm", search_schedule=SCHEDULE))
        plain = gen_loop.load_agent(vrun_stub(self.root, "armplain"))
        arm = self.root / "arm-dir"
        arm.mkdir()
        (arm / "runs.jsonl").write_text("")
        (arm / "summary.json").write_text(json.dumps({"agent": c.describe(), "config": "a3-v1"}))
        self.assertTrue(gen0.emu_arm_match(arm, c.describe())[0])
        ok, why = gen0.emu_arm_match(arm, plain.describe())
        self.assertFalse(ok)
        self.assertIn("stakes", why)  # an arm played at another schedule is not this agent's
        (arm / "summary.json").write_text(json.dumps({"agent": plain.describe()}))
        self.assertFalse(gen0.emu_arm_match(arm, c.describe())[0])
        held = gen_loop.load_agent(vrun_stub(self.root, "armhold", search_schedule=SCHEDULE, potion_hold={"hp_frac": 0.3}))
        (arm / "summary.json").write_text(json.dumps({"agent": c.describe()}))
        self.assertFalse(gen0.emu_arm_match(arm, held.describe())[0])  # the hold is identity too

    def test_the_real_arm_deadline_constant_is_the_referees(self):
        text = (ROOT / "referee" / "envproto.py").read_text(encoding="utf-8")
        m = re.search(r"^RUN_DEADLINE_S = ([0-9.]+)", text, re.M)
        self.assertIsNotNone(m)
        self.assertEqual(float(m[1]), gen0.REAL_RUN_DEADLINE_S)

    def test_a_stakes_run_gets_a_wall_limit_that_scales_with_its_schedule(self):
        """run_block's per-run timeout is multiplied by the agent's wall factor (the plain agent's is untouched)."""
        seen = {}

        class Boom(Exception):
            pass

        def fake_pool(*a, **kw):
            seen["timeout"] = kw.get("run_timeout_s")
            raise Boom

        for name, meta, want in (("plain", {}, gen0.RUN_TIMEOUT_S), ("sched", {"search_schedule": SCHEDULE}, 4 * gen0.RUN_TIMEOUT_S)):
            agent = gen_loop.load_agent(vrun_stub(self.root, "wall-" + name, **meta))
            with mock.patch.object(gen0.spawn_pool, "SupervisedPool", side_effect=fake_pool), self.assertRaises(Boom):
                gen0.run_block(["AAAAAAAAAA"], "a3-v1", 1, self.root / ("out-" + name), agent=agent)
            self.assertEqual(want, seen["timeout"], name)


# ------------------------------------------------------------------------------------------------------------------ the ledger


class LedgerPerRecipeTests(unittest.TestCase):
    """A schedule and a hold are recipes of their own: attempts, blocks and the cap are counted per recipe."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.agents = self.root / "agent"
        self.agents.mkdir()
        self.tsv = legacy_tsv(self.agents)  # generation 12's closed self-play line: one row of the plain recipe
        (self.root / "seeds").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def promote(self, cand: Path, name: str, d_mean: float = -4.0):
        seeds, cand_rows, base_rows = make_block(d_mean, prefix=name[-1])
        seed_file = self.root / "seeds" / f"{name}.json"
        seed_file.write_text(json.dumps({"seeds": seeds}))
        with mock.patch.object(gen_loop, "run_arm", StubArms(cand_rows, base_rows)):
            return gen_loop.promote(cand, None, seed_file, agent_root=self.agents, boot=500, review_guard=True, sequential=True)

    def test_each_recipe_counts_its_own_attempts(self):
        stakes_recipe = mr.full_recipe("vrun", STAKES)
        hold_recipe = mr.full_recipe("vrun", STAKES, 30)
        add = gen_loop.append_promotion
        add(self.tsv, {"cand_gen": 12, "seed_block": "E12-s1", "cand_sha": PRIOR_SHA, "run_head_sha": "h1", "macro": stakes_recipe,
                       "decision": "rejected", "attempt": 1})
        rows = gen_loop.read_promotions(self.tsv)
        # the same fighter and head under the plain recipe: the closed line's attempt; under the schedule: its own
        self.assertEqual(1, gen_loop.attempts_before(rows, 12, PRIOR_SHA, "h2", stakes_recipe))
        self.assertEqual(0, gen_loop.attempts_before(rows, 12, PRIOR_SHA, "h1", stakes_recipe))  # the same pair again is not a new attempt
        self.assertEqual(0, gen_loop.attempts_before(rows, 12, PRIOR_SHA, "h2", hold_recipe))  # the hold has not been tried
        self.assertEqual(1, gen_loop.attempts_before(rows, 12, PRIOR_SHA, "h2", ""))  # the plain line keeps its single legacy row
        # a block judged under the schedule is not another recipe's to judge
        self.assertTrue(gen_loop.block_used_by_another(self.tsv, "E12-s1", PRIOR_SHA, "h1", hold_recipe))
        self.assertTrue(gen_loop.block_used_by_another(self.tsv, "E12-s1", PRIOR_SHA, "h1", ""))
        self.assertFalse(gen_loop.block_used_by_another(self.tsv, "E12-s1", PRIOR_SHA, "h1", stakes_recipe))
        # the cap of two is the recipe's own
        add(self.tsv, {"cand_gen": 12, "seed_block": "E12-s2", "cand_sha": PRIOR_SHA, "run_head_sha": "h2", "macro": stakes_recipe,
                       "decision": "rejected", "attempt": 2})
        self.assertEqual(2, gen_loop.attempts_spent(self.tsv, 12, True, stakes_recipe))
        self.assertIsNone(gen_loop.attempts_spent(self.tsv, 12, True, hold_recipe))

    def test_the_promotion_row_carries_the_recipe_of_the_stakes_candidate(self):
        c = vrun_stub(self.root, "candidate-stakes", search_schedule=SCHEDULE)
        agent = gen_loop.load_agent(c)
        doc = self.promote(c, "E12-2026-11-0A")
        self.assertEqual("vrun+stakes:a3-v1@h1e4b16", doc["macro"])
        row = gen_loop.read_promotions(self.tsv)[-1]
        self.assertEqual(("vrun+stakes:a3-v1@h1e4b16", "1"), (row["macro"], row["attempt"]))  # attempt 1 of its own line
        self.assertEqual(agent.describe()["stakes"], doc["cand"]["agent"]["stakes"])
        self.assertNotIn("stakes", doc["base"]["agent"])  # the base (generation 11, the plain a3-v1) is the plain agent

    def test_the_hold_candidate_is_a_recipe_of_its_own_and_gets_its_own_block(self):
        stakes = vrun_stub(self.root, "candidate-stakes", search_schedule=SCHEDULE)
        hold = vrun_stub(self.root, "candidate-stakes-hold", search_schedule=SCHEDULE, potion_hold={"hp_frac": 0.3})
        self.promote(stakes, "E12-2026-11-0A")
        with self.assertRaisesRegex(ValueError, "already judged another candidate"):
            self.promote(hold, "E12-2026-11-0A")
        self.promote(hold, "E12-2026-11-0B")
        rows = gen_loop.read_promotions(self.tsv)
        self.assertEqual(["", "vrun+stakes:a3-v1@h1e4b16", "vrun+stakes:a3-v1@h1e4b16+hold:30"], [r.get("macro", "") for r in rows])
        self.assertEqual("1", rows[-1]["attempt"])

    def test_the_train_job_counts_the_built_candidates_schedule_recipe(self):
        """`gen0.py train` reads the built candidate's recipe (schedule and hold included) for the review guard's cap check."""
        import shutil

        cand = self.agents / "gen12" / "candidate-stakes-hold"
        cand.parent.mkdir(parents=True)
        shutil.copytree(vrun_stub(self.root, "src", search_schedule=SCHEDULE, potion_hold={"hp_frac": 0.3}), cand)
        meta = json.loads((cand / "agent.json").read_text())
        meta["built"] = {"why": "test"}
        (cand / "agent.json").write_text(json.dumps(meta))
        markers = self.root / "train"
        markers.mkdir()
        (markers / "ready-11-g12-stakes-hold").write_text("x")
        args = ["train", "--marker-dir", str(markers), "--shards-root", str(self.root), "--agent-root", str(self.agents), "--promote",
                "--emu-python", "python", "--seeds-file", str(self.root / "seeds" / "E{k}-{date}.json"), "--review-guard"]
        recipe = "vrun+stakes:a3-v1@h1e4b16+hold:30"
        with mock.patch.object(gen_loop, "train_generation") as tg, mock.patch.object(gen_loop, "run_promote_subprocess", return_value=0) as rp, \
                mock.patch.object(gen_loop, "COMMITTED_SEEDS", self.root / "seeds"):
            self.assertEqual(0, gen0.main(args))
        tg.assert_not_called()
        self.assertEqual(cand, rp.call_args[0][1])  # judged: the plain line's one attempt is not this recipe's
        for i in (1, 2):
            gen_loop.append_promotion(self.tsv, {"cand_gen": 12, "seed_block": f"E12-x{i}", "cand_sha": PRIOR_SHA, "run_head_sha": f"head{i}",
                                                 "macro": recipe, "decision": "rejected", "attempt": i})
        with mock.patch.object(gen_loop, "train_generation"), mock.patch.object(gen_loop, "run_promote_subprocess") as rp:
            self.assertEqual(0, gen0.main(args))
        rp.assert_not_called()  # the recipe at its own cap of two
        self.assertTrue((markers / "done-11-g12-stakes-hold").exists())


# ------------------------------------------------------------------------------------------------------------------ the build


class BuildStakesTests(unittest.TestCase):
    """`gen0.py build --stakes --potion-hold --pick`: the candidate folders the lab judges."""

    def setUp(self):
        from tests.python.test_gen_loop import trained_candidate

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _, _, trained = trained_candidate()
        self.agents = self.root / "agent"
        self.base = self.agents / "gen5" / "candidate"
        self.base.mkdir(parents=True)
        for name in ("combat_policy.pt", "combat_net.pt", "model.json"):
            (self.base / name).write_bytes((trained / name).read_bytes())
        (self.base / "run_value.npz").write_bytes(b"the base fit")
        (self.base / "run_value.pt").write_bytes(b"the base fit, torch")
        (self.base / "agent.json").write_text(json.dumps({"generation": 5, "macro": "vrun", "mask_version": 4}))
        (self.agents / "current.json").write_text(json.dumps({"generation": 5, "dir": str(self.base)}))
        self.prior = self.root / "players" / "v2"
        self.prior.mkdir(parents=True)
        for name in ("combat_policy.pt", "combat_net.pt", "model.json"):
            (self.prior / name).write_bytes((trained / name).read_bytes())
        self.weights = self.root / "pi_h-v1-win" / "weights.npz"
        self.weights.parent.mkdir()
        np.savez(self.weights, meta=np.array(json.dumps({"variant": "t"})), **random_weights())
        self.markers = self.root / "train"

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, *extra: str) -> int:
        return gen0.main(["build", "--prior", str(self.prior), "--agent-root", str(self.agents), "--candidate-generation", "7", "--why", "test",
                          "--no-export", "--marker-dir", str(self.markers), *extra])

    def test_the_stakes_and_hold_candidates_load_with_their_recipes_and_markers(self):
        self.assertEqual(0, self.build("--tag", "stakes", "--stakes", STAKES))
        self.assertEqual(0, self.build("--tag", "stakes-hold", "--stakes", STAKES, "--potion-hold", "30"))
        a, b = (gen_loop.load_agent(self.agents / "gen7" / n) for n in ("candidate-stakes", "candidate-stakes-hold"))
        self.assertEqual(("vrun+stakes:a3-v1@h1e4b16", "vrun+stakes:a3-v1@h1e4b16+hold:30"), (a.recipe(), b.recipe()))
        self.assertEqual("vrun", a.macro)  # the macro stays the plain vrun: only the search's identity is new
        meta = json.loads((self.agents / "gen7" / "candidate-stakes-hold" / "agent.json").read_text())
        self.assertEqual({"base": "a3-v1", "mult": {"hall": 1, "elite": 4, "boss": 16}, "tag": STAKES}, meta["search_schedule"])
        self.assertEqual({"hp_frac": 0.3, "fights": "hall"}, meta["potion_hold"])
        self.assertEqual("vrun+stakes:a3-v1@h1e4b16+hold:30", meta["recipe"])
        self.assertIn("fight search alone", meta["note"])
        self.assertEqual(["ready-5-g7-stakes", "ready-5-g7-stakes-hold"], sorted(p.name for p in self.markers.iterdir()))
        # the fight player and the run head are the base's, and the candidate is the same identity pair as the plain one: only the recipe differs
        self.assertEqual(gen_loop.candidate_identity(a), gen_loop.candidate_identity(b))
        self.assertEqual(gen_loop.file_sha256(self.base / "run_value.npz"), gen_loop.run_head_sha(a))

    def test_a_malformed_schedule_or_hold_is_refused_before_anything_is_written(self):
        for extra in (("--stakes", "a3-v1@h1e4"), ("--stakes", "nope@h1e4b16"), ("--potion-hold", "0"), ("--potion-hold", "100")):
            with self.assertRaises(ValueError, msg=str(extra)):
                self.build("--tag", "bad", *extra)
        self.assertFalse((self.agents / "gen7").exists())

    def test_the_pick_modes_build_into_the_macro_string_and_agent_json(self):
        sha = gen_loop.file_sha256(self.weights)
        self.assertEqual(0, self.build("--tag", "winprior-sample", "--stakes", STAKES, "--potion-hold", "30", "--human-prior", str(self.weights),
                                       "--lambda", "0.5", "--pick", "sample"))
        a = gen_loop.load_agent(self.agents / "gen7" / "candidate-winprior-sample")
        self.assertEqual("vrun+pi_h:lambda=0.5:pi_h=" + sha[:8] + ":pick=sample", a.macro)
        self.assertEqual(a.macro + "+stakes:a3-v1@h1e4b16+hold:30", a.recipe())
        meta = json.loads((self.agents / "gen7" / "candidate-winprior-sample" / "agent.json").read_text())
        self.assertEqual(("sample", 1.0), (meta["human_prior"]["pick"], meta["human_prior"]["tau"]))
        self.assertEqual(0, self.build("--tag", "winprior-skipcal", "--human-prior", str(self.weights), "--lambda", "0.5", "--pick", "skipcal",
                                       "--skip-bias", "0.35"))
        b = gen_loop.load_agent(self.agents / "gen7" / "candidate-winprior-skipcal")
        self.assertEqual("vrun+pi_h:lambda=0.5:pi_h=" + sha[:8] + ":pick=skipcal0.35", b.macro)
        self.assertEqual(("skipcal", 0.35), mr.parse_pick(b.macro))
        self.assertEqual(0.35, json.loads((self.agents / "gen7" / "candidate-winprior-skipcal" / "agent.json").read_text())["human_prior"]["skip_bias"])
        # a pick mode needs the chooser; agent.json and the macro must agree
        with self.assertRaises(ValueError):
            self.build("--tag", "lonely", "--pick", "sample")
        p = self.agents / "gen7" / "candidate-winprior-sample" / "agent.json"
        meta["human_prior"]["pick"] = "skipcal"
        p.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "pick"):
            gen_loop.load_agent(p.parent)

    def test_a_base_with_a_schedule_is_not_built_on_silently(self):
        (self.base / "agent.json").write_text(json.dumps({"generation": 5, "macro": "vrun", "search_schedule": SCHEDULE}))
        with self.assertRaisesRegex(ValueError, "plain generation"):
            self.build("--tag", "again", "--stakes", STAKES)


# ------------------------------------------------------------------------------------------------------------------ the hold rule


class HoldRuleTests(unittest.TestCase):
    """`live_search.hold_keeps_potions` and the root's potion moves."""

    def test_the_rule_on_synthetic_numbers(self):
        keep = live_search.hold_keeps_potions
        # safe: 60 of 80 HP, 8 coming: 52 left, threshold 24 -> the potion is dropped
        k, rec = keep(False, 60, 80, 0, 8, 0.30)
        self.assertFalse(k)
        self.assertEqual((52, 24.0, 8, 0, False), (rec["after"], rec["threshold"], rec["incoming"], rec["block"], rec["killed"]))
        # in danger: 30 of 80, 20 coming: 10 left < 24 -> kept
        self.assertTrue(keep(False, 30, 80, 0, 20, 0.30)[0])
        # block counts: the same 20 coming with 12 block is 8 through: 22 left, still under 24 -> kept; with 16 block 26 left -> dropped
        self.assertTrue(keep(False, 30, 80, 12, 20, 0.30)[0])
        self.assertFalse(keep(False, 30, 80, 16, 20, 0.30)[0])
        # exactly at the threshold is not under it
        self.assertFalse(keep(False, 44, 80, 0, 20, 0.30)[0])  # 24 left
        self.assertTrue(keep(False, 43, 80, 0, 20, 0.30)[0])  # 23 left
        # a kill this turn is no danger, however low the HP
        self.assertFalse(keep(True, 5, 80, 0, 60, 0.30)[0])
        # a check that could not run keeps the potions
        self.assertTrue(keep(None, 60, 80, 0, 8, 0.30)[0])
        # no attack announced: nothing to fall under (block never makes it negative)
        self.assertFalse(keep(False, 20, 80, 30, 0, 0.20)[0])
        self.assertTrue(keep(False, 15, 80, 0, 0, 0.20)[0])  # but 15 of 80 is under 16 already: after = hp
        # the threshold is the hold's own
        self.assertTrue(keep(False, 60, 80, 0, 8, 0.70)[0])  # 52 < 56
        self.assertFalse(keep(False, 60, 80, 0, 8, 0.50)[0])

    @staticmethod
    def stub_search(kinds: list[str], *, hp=60, max_hp=80, block=0, incoming=8, kill_check=None):
        """A `_Search` with just what `hold_potions` reads: root moves of the given kinds, a root that forks a view of these numbers."""
        s = object.__new__(live_search._Search)
        s.children = [SimpleNamespace(node=SimpleNamespace(kind=k), key=(k, i)) for i, k in enumerate(kinds)]
        s.kill_check = kill_check
        s.hold = None
        view = SimpleNamespace(obs=[hp, max_hp, block], close=lambda: None)
        s.root = SimpleNamespace(fork=lambda _seed: view)
        s._incoming = incoming
        return s

    def hold(self, search, frac=0.30, killed=False):
        report = SimpleNamespace(kill=object() if killed else None)
        with mock.patch.object(live_search, "incoming_damage", lambda _obs: search._incoming), \
                mock.patch.object(live_search.lethal, "find_lethal", return_value=report) as fl:
            rec = search.hold_potions(frac)
        return rec, fl

    def test_a_synthetic_root_in_a_safe_fight_loses_its_potion_moves(self):
        s = self.stub_search(["card", "card", "potion", "end_turn"])
        rec, fl = self.hold(s)
        self.assertEqual(["card", "card", "end_turn"], [c.node.kind for c in s.children])
        self.assertFalse(rec["kept"])
        self.assertEqual((1, 52), (rec["potions"], rec["after"]))
        self.assertIs(rec, s.hold)
        fl.assert_called_once()  # no kill check was run before: it runs now

    def test_a_synthetic_root_in_danger_keeps_its_potion_moves(self):
        s = self.stub_search(["card", "potion", "potion", "end_turn"], hp=20, incoming=30)
        rec, _ = self.hold(s)
        self.assertEqual(["card", "potion", "potion", "end_turn"], [c.node.kind for c in s.children])
        self.assertTrue(rec["kept"])
        self.assertEqual(-10, rec["after"])

    def test_a_kill_makes_the_potion_unneeded_even_at_low_hp(self):
        s = self.stub_search(["card", "potion", "end_turn"], hp=20, incoming=30)
        rec, _ = self.hold(s, killed=True)
        self.assertEqual(["card", "end_turn"], [c.node.kind for c in s.children])
        self.assertTrue(rec["killed"])

    def test_a_kill_check_that_already_looked_is_not_run_again(self):
        s = self.stub_search(["card", "potion", "end_turn"], hp=20, incoming=30, kill_check={"fired": False, "reason": "none"})
        rec, fl = self.hold(s)
        fl.assert_not_called()
        self.assertTrue(rec["kept"])  # no kill, 20 - 30 under 24
        e = self.stub_search(["card", "potion", "end_turn"], hp=70, incoming=5, kill_check={"fired": False, "reason": "error"})
        rec, fl = self.hold(e)
        self.assertTrue(rec["kept"])  # a check that failed keeps the potions
        self.assertEqual(3, len(e.children))

    def test_a_failing_kill_check_keeps_the_potions(self):
        s = self.stub_search(["card", "potion", "end_turn"], hp=70, incoming=5)
        with mock.patch.object(live_search, "incoming_damage", lambda _obs: 5), \
                mock.patch.object(live_search.lethal, "find_lethal", side_effect=RuntimeError("handle pool full")):
            rec = s.hold_potions(0.30)
        self.assertTrue(rec["kept"])
        self.assertEqual(3, len(s.children))

    def test_nothing_to_hold_is_not_judged(self):
        for kinds in (["card", "end_turn"], ["potion", "potion"], ["potion"]):
            s = self.stub_search(kinds)
            rec, fl = self.hold(s)
            self.assertIsNone(rec)
            self.assertIsNone(s.hold)
            self.assertEqual(len(kinds), len(s.children))
            fl.assert_not_called()

    def test_the_search_entry_points_take_the_hold(self):
        import inspect

        for fn in (live_search.search_live, live_search.search_steps):
            p = inspect.signature(fn).parameters["potion_hold"]
            self.assertIsNone(p.default, fn.__name__)
        self.assertIn("potion_hold", live_search._SEARCH_SETTINGS)
        self.assertIsNone(live_search.SearchResult.__dataclass_fields__["hold"].default)

    def test_only_a_hallway_fight_passes_the_hold_to_the_search(self):
        cfg = sc.get("a3-v1+hold30")
        self.assertEqual(0.3, cfg.search_kwargs(pos(fight="hall"), env_prior=False)["potion_hold"])
        for fight in ("elite", "boss"):
            self.assertNotIn("potion_hold", cfg.search_kwargs(pos(fight=fight), env_prior=False), fight)
        self.assertNotIn("potion_hold", cfg.search_kwargs(None, env_prior=False))
        both = sc.get("a3-v1@h1e4b16+hold30")
        self.assertEqual(0.3, both.search_kwargs(pos(fight="hall"), env_prior=False)["potion_hold"])
        self.assertEqual(30, both.search_kwargs(pos("play", "hall"), env_prior=False)["max_sims"])  # a hallway fight stays at a3-v1's budget
        self.assertEqual(120, both.search_kwargs(pos("target", "hall"), env_prior=False)["max_sims"])

    def test_the_real_incoming_damage_reads_the_announced_attacks(self):
        from sts2_gym import native

        obs = [0] * native.OBS_SIZE
        base = native.OBS_ENEMY_OFFSET
        obs[0], obs[1], obs[2] = 60, 80, 3
        obs[base:base + 5] = [40, 44, 0, 0, 12]  # a living attacker announcing 12
        obs[base + native.OBS_ENEMY_SLOT_SIZE:base + native.OBS_ENEMY_SLOT_SIZE + 5] = [0, 44, 0, 0, 9]  # a dead one
        self.assertEqual(12, live_search.incoming_damage(obs))
        keep, rec = live_search.hold_keeps_potions(False, obs[0], obs[1], obs[2], live_search.incoming_damage(obs), 0.30)
        self.assertFalse(keep)
        self.assertEqual(51, rec["after"])  # 60 - (12 - 3)


class HoldOnRealFightsTests(unittest.TestCase):
    """The hold on the emulator: a safe fight drops the potion move, a fight that is about to be lost keeps it."""

    def search(self, fight: Fight, **kw):
        return live_search.search_live(fight.handle, time_budget_s=60.0, min_child_sims=20, max_sims=120, confirm_pairs=0, **kw)

    def test_a_safe_opening_hand_does_not_offer_the_potion(self):
        fight = Fight(0, NIBBITS_WEAK, [STRIKE] * 5, potions=(BLOCK_POTION,), hp=64, max_hp=80)
        try:
            plain = self.search(fight)
            held = self.search(fight, potion_hold=0.30)
        finally:
            fight.close()
        self.assertIn("potion", [c.kind for c in plain.children])
        self.assertNotIn("potion", [c.kind for c in held.children])
        self.assertIsNone(plain.hold)
        self.assertFalse(held.hold["kept"])
        self.assertEqual(1, held.hold["potions"])

    def test_a_fight_about_to_be_lost_keeps_the_potion(self):
        fight = Fight(0, NIBBITS_WEAK, [STRIKE] * 5, potions=(BLOCK_POTION, BLOCK_POTION), hp=10)
        try:
            for _ in range(3):
                fight.play(STRIKE, target=0)
            self.assertGreaterEqual(fight.enemy(0, 4), 10)
            held = self.search(fight, potion_hold=0.30)
        finally:
            fight.close()
        self.assertTrue(held.hold["kept"])
        self.assertIn("potion", [c.kind for c in held.children])
        self.assertEqual("potion", held.best.kind)  # the search still takes the potion that saves the run

    def test_the_hold_off_is_the_search_it_always_was(self):
        fight = Fight(0, NIBBITS_WEAK, [STRIKE] * 5, potions=(BLOCK_POTION,), hp=64, max_hp=80)
        try:
            a = self.search(fight, potion_hold=None)
            b = self.search(fight)
        finally:
            fight.close()
        self.assertEqual((a.best.action, a.best.kind, a.sims), (b.best.action, b.best.kind, b.sims))


# ------------------------------------------------------------------------------------------------------------------ the sampled chooser


def bare_policy(lam: float = 0.5, pick: str = "sample", skip_bias: float = 0.0, tau: float = 0.0, eps: float = 0.0) -> ic.BlendedPolicy:
    """A `BlendedPolicy` with only what the pick arithmetic reads (no checkpoint, no prior file)."""
    p = object.__new__(ic.BlendedPolicy)
    p.lam, p.pick_mode, p.skip_bias, p.tau, p.eps = lam, pick, skip_bias, tau, eps
    p._run_key, p._screens = None, {}
    return p


class SampledPickTests(unittest.TestCase):
    def test_the_distribution_is_softmax_of_v_plus_lambda_log_pi_at_tau_1(self):
        p = bare_policy(0.5)
        values, logp = np.array([2.0, 1.0, 0.0]), np.array(lp(0.5, 0.3, 0.2))
        choice = ic.blend_pick(values, logp, 0.5)
        pi = p.distribution(values, logp, choice, tau=1.0)
        score = values + 0.5 * logp
        want = np.exp(score - score.max())
        np.testing.assert_allclose(want / want.sum(), pi, atol=1e-12)
        self.assertAlmostEqual(1.0, float(pi.sum()))
        # tau 0 is the blend's own pick; a warmer tau spreads the mass
        np.testing.assert_allclose([1.0, 0.0, 0.0], p.distribution(values, logp, choice, tau=0.0))
        self.assertLess(float(p.distribution(values, logp, choice, tau=1.0).max()), float(p.distribution(values, logp, choice, tau=0.2).max()))
        # the guard's set is kept: an option the prior gives under 2 in 100 has no mass (unless the guard picked it)
        logp2 = np.array(lp(0.6, 0.39, 0.01))
        pi2 = p.distribution(values, logp2, ic.blend_pick(values, logp2, 0.5), tau=1.0)
        self.assertEqual(0.0, float(pi2[2]))
        # at lambda inf the prior alone is sampled
        pinf = bare_policy(math.inf).distribution(values, logp, ic.blend_pick(values, logp, math.inf), tau=1.0)
        np.testing.assert_allclose(np.exp(logp), pinf, atol=1e-12)

    def test_the_draws_are_reproducible_and_depend_on_the_seed_the_floor_and_the_screen_only(self):
        values, logp = np.array([2.0, 1.9, 1.8, 1.7]), np.array(lp(0.3, 0.3, 0.2, 0.2))
        choice = ic.blend_pick(values, logp, 0.5)

        def run(seed, floors):
            p = bare_policy()
            p.begin_run(seed)
            pi = p.distribution(values, logp, choice, tau=1.0)
            return [ic._draw(pi, p._sample_stream(f, "card")) for f in floors]

        floors = list(range(2, 17))
        self.assertEqual(run("ABCDEFGHJK", floors), run("ABCDEFGHJK", floors))
        self.assertNotEqual(run("ABCDEFGHJK", floors), run("ABCDEFGHJL", floors))  # another run, other draws
        self.assertGreater(len(set(run("ABCDEFGHJK", floors))), 1)  # it does sample
        # a floor's draw does not depend on which floors were drawn before it
        self.assertEqual(run("ABCDEFGHJK", [9]), [run("ABCDEFGHJK", floors)[floors.index(9)]])
        # the same floor and screen twice in a run is two draws (a second reward screen), each reproducible
        p = bare_policy()
        p.begin_run("ABCDEFGHJK")
        first, second = p._sample_stream(5, "card").random(), p._sample_stream(5, "card").random()
        q = bare_policy()
        q.begin_run("ABCDEFGHJK")
        self.assertEqual((first, second), (q._sample_stream(5, "card").random(), q._sample_stream(5, "card").random()))
        self.assertNotEqual(first, second)
        # the screen kinds have streams of their own
        r = bare_policy()
        r.begin_run("ABCDEFGHJK")
        self.assertNotEqual(r._sample_stream(5, "card").random(), bare_policy_stream("ABCDEFGHJK", 5, "rest"))
        # a new begin_run starts the counts again
        p.begin_run("ABCDEFGHJK")
        self.assertEqual(first, p._sample_stream(5, "card").random())

    def test_a_sample_pick_needs_the_runs_seed(self):
        with self.assertRaisesRegex(ValueError, "begin_run"):
            bare_policy()._sample_stream(3, "card")

    def test_the_draw_frequencies_follow_the_distribution(self):
        values, logp = np.array([1.0, 0.0]), np.array(lp(0.5, 0.5))
        p = bare_policy()
        pi = p.distribution(values, logp, ic.blend_pick(values, logp, 0.5), tau=1.0)
        counts = np.zeros(2)
        for i in range(4000):
            p.begin_run(f"S{i}")
            counts[ic._draw(pi, p._sample_stream(7, "card"))] += 1
        self.assertAlmostEqual(float(pi[0]), counts[0] / 4000, delta=0.03)

    def test_the_skip_bias_moves_the_blend_pick_and_nothing_else(self):
        values, logp = [1.0, 0.8, 0.0], lp(0.5, 0.3, 0.2)  # two cards and the skip (index 2)
        self.assertEqual(0, ic.blend_pick(values, logp, 0.5).index)
        gap = ic.skip_gap(values, logp, 0.5, 2)
        self.assertAlmostEqual(1.0 + 0.5 * math.log(0.5) - (0.0 + 0.5 * math.log(0.2)), gap)
        bonus = lambda b: [0.0, 0.0, b]  # noqa: E731
        self.assertEqual(0, ic.blend_pick(values, logp, 0.5, bonus(gap - 0.01)).index)
        self.assertEqual(2, ic.blend_pick(values, logp, 0.5, bonus(gap + 0.01)).index)
        self.assertEqual(0, ic.blend_pick(values, logp, 0.5, None).index)
        # the same at lambda inf, where the bias is in log-probability
        self.assertEqual(0, ic.blend_pick(values, logp, math.inf, bonus(0.5)).index)
        self.assertEqual(2, ic.blend_pick(values, logp, math.inf, bonus(1.0)).index)  # log(.5/.2) = 0.916
        # the prior_top and v_top the run field reports are the bare ones
        c = ic.blend_pick(values, logp, 0.5, bonus(5.0))
        self.assertEqual((2, 0, 0), (c.index, c.prior_top, c.v_top))
        with self.assertRaises(ValueError):
            ic.blend_pick(values, logp, 0.5, [0.0])

    def test_the_skipcal_bias_is_in_the_exploring_distribution_too(self):
        values, logp = np.array([1.0, 0.8, 0.0]), np.array(lp(0.5, 0.3, 0.2))
        p = bare_policy(0.5, "skipcal", 3.0, tau=1.0)
        choice = ic.blend_pick(values, logp, 0.5, [0.0, 0.0, 3.0])
        pi = p.distribution(values, logp, choice, bonus=[0.0, 0.0, 3.0])
        self.assertEqual(2, int(np.argmax(pi)))

    def test_the_run_row_carries_the_skip_gap_only_when_the_policy_says_so(self):
        f = run_fields.RunFields()
        f.decide(5, {"screen": "card", "prior": 1, "v": 0, "guard": 0})
        f.decide(6, {"screen": "card", "prior": 0, "v": 1, "guard": 0, "gap": 0.42})
        f.decide(7, {"screen": "rest", "prior": 1, "v": 1, "guard": 0})
        self.assertEqual([[5, "card", 1, 0, 0], [6, "card", 0, 1, 0, 0.42], [7, "rest", 1, 1, 0]], f.row()["decided"])


def bare_policy_stream(seed: str, floor: int, screen: str) -> float:
    p = bare_policy()
    p.begin_run(seed)
    return float(p._sample_stream(floor, screen).random())


@unittest.skipUnless(Path("D:/spireline/referee/human-prior/pi_h-v1-win/weights.npz").is_file()
                     and Path("D:/spireline/referee/agent/gen11/candidate/run_value.npz").is_file(), "the fitted prior and the judgement are not here")
class SampledPickOnRealWeightsTests(unittest.TestCase):
    """The policy on a decision built from a record's state, with the real judgement and the win-conditioned prior."""

    WEIGHTS = Path("D:/spireline/referee/human-prior/pi_h-v1-win/weights.npz")
    JUDGE = Path("D:/spireline/referee/agent/gen11/candidate/run_value.npz")

    @classmethod
    def setUpClass(cls):
        import human_prior as hp
        import human_prior_infer as hpi
        from tests.python.test_human_prior import _find_ids, synthetic_run

        hp.stack()
        cls.hp, cls.hpi = hp, hpi
        cls.prior = hpi.PriorScorer.load(cls.WEIGHTS)
        from collections import Counter

        t = hp.Tables()
        specs = hp.run_decisions(synthetic_run(_find_ids(False, 1)[0]), t, Counter(), Counter(), Counter())
        sp = next(s for s in specs if s["kind"] == "card")
        cls.view = hp.spec_view(sp)
        from run_train import features as F

        cls.decision = F.Decision("card_reward", tuple(hp._option(o, cls.view, j) for j, o in enumerate(sp["options"])))

    def policy(self, pick="argmax", bias=0.0, lam=0.5, **kw):
        return ic.BlendedPolicy(self.JUDGE, self.prior, lam, W=53.0, pick=pick, skip_bias=bias, **kw)

    def test_the_sample_mode_draws_reproducibly_and_the_argmax_mode_is_untouched(self):
        argmax = self.policy()
        sampled = self.policy("sample")
        picks = []
        for seed in ("RUNAAAAAAA", "RUNAAAAAAB", "RUNAAAAAAC") * 2:
            sampled.begin_run(seed)
            picks.append(sampled.pick(self.view, self.decision, None).index)
        self.assertEqual(picks[:3], picks[3:])  # the same seeds, the same draws
        wide = set()
        for i in range(80):
            sampled.begin_run(f"WIDE{i:06d}")
            wide.add(sampled.pick(self.view, self.decision, None).index)
        self.assertGreater(len(wide), 1)  # a draw, not an argmax
        self.assertEqual(argmax.pick(self.view, self.decision, None).index, argmax.pick(self.view, self.decision, None).index)
        self.assertNotIn("pick", argmax.describe())
        self.assertEqual("sample", sampled.describe()["pick"])
        with self.assertRaisesRegex(ValueError, "begin_run"):
            self.policy("sample").pick(self.view, self.decision, None)

    def test_a_large_skip_bias_takes_the_skip_and_the_row_names_the_gap(self):
        n = len(self.decision.options)
        skip = next((i for i, o in enumerate(self.decision.options) if o.kind == "card.skip"), None)
        self.assertIsNotNone(skip, "the test's reward has no skip option")
        none = self.policy("skipcal", 0.0)
        high = self.policy("skipcal", 25.0)
        self.assertEqual(skip, high.pick(self.view, self.decision, None).index)
        none.pick(self.view, self.decision, None)
        gap = none.last["gap"]
        self.assertIsInstance(gap, float)
        self.assertLess(gap, 25.0)
        self.assertEqual(self.policy().pick(self.view, self.decision, None).index, none.pick(self.view, self.decision, None).index)  # bias 0 is the argmax
        self.assertNotIn("gap", self.policy().last or {})
        self.assertLessEqual(n, 4)


# ------------------------------------------------------------------------------------------------------------------ the skip calibration


class SkipCalibrationTests(unittest.TestCase):
    @staticmethod
    def rows(skips: int, takes: int, gaps: list[float] | None = None, floor0: int = 2) -> list[dict]:
        """One run row per card screen: `skips` skips and `takes` cards in Act 1, plus an Act 2 screen that must not count."""
        out = []
        for i in range(skips + takes):
            out.append({"cards": [[floor0 + i % 14, "skip" if i < skips else "STRIKE"], [20, "skip"]],
                        "decided": ([[floor0 + i % 14, "card", 1, 1, 0, gaps[i]]] if gaps else []) + [[20, "card", 1, 1, 0, 0.0]]})
        return out

    def test_the_act_1_skip_rate_reads_the_cards_field(self):
        rate, n = skc.skip_rate(self.rows(15, 85))
        self.assertEqual((0.15, 100), (rate, n))
        self.assertEqual((None, 0), skc.skip_rate([{"cards": [[20, "skip"]]}, {}]))
        self.assertEqual((1.0, 1), skc.skip_rate([{"cards": [[17, "skip"], [18, "skip"]]}]))  # floor 17 is Act 1, floor 18 is not

    def test_the_gaps_are_the_act_1_card_screens_with_a_gap(self):
        rows = self.rows(0, 4, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual([1.0, 2.0, 3.0, 4.0], skc.act1_gaps(rows))
        self.assertEqual([], skc.act1_gaps(self.rows(1, 1)))  # an argmax run has no gaps

    def test_the_open_loop_rate_is_the_share_of_gaps_at_or_under_the_bias(self):
        gaps = [float(g) for g in range(100)]  # 0 .. 99
        self.assertEqual(0.0, skc.rate_at(gaps, -1.0))
        self.assertEqual(0.5, skc.rate_at(gaps, 49.0))
        self.assertEqual(1.0, skc.rate_at(gaps, 99.0))
        with self.assertRaises(ValueError):
            skc.rate_at([], 0.0)

    def test_the_bias_for_a_target_rate_lands_on_it(self):
        rng = np.random.default_rng(5)
        gaps = list((rng.gamma(2.0, 1.5, 400) - 0.5).round(3))
        for target in (0.05, 0.154, 0.3, 0.6):
            b = skc.bias_for(gaps, target)
            self.assertAlmostEqual(target, skc.rate_at(gaps, b), delta=0.005, msg=str(target))
        self.assertLess(skc.bias_for(gaps, 0.05), skc.bias_for(gaps, 0.154))  # monotone
        self.assertEqual(round(skc.bias_for(gaps, 0.154), 3), skc.bias_for(gaps, 0.154))  # the three decimals the recipe keeps
        self.assertLessEqual(skc.bias_for([1.0, 2.0], 0.0), 0.95)  # a zero target: just under the smallest gap
        self.assertGreaterEqual(skc.bias_for([1.0, 2.0], 1.0), 2.05)
        with self.assertRaises(ValueError):
            skc.bias_for([], 0.15)

    def test_a_calibration_that_lands_in_the_second_round(self):
        # a world whose skip rate at bias b is the share of 400 fixed gaps at or under b, shifted a little by the bias (the decks move)
        rng = np.random.default_rng(1)
        base = rng.gamma(2.0, 1.5, 400) - 0.5
        played = []

        def play(bias: float) -> list[dict]:
            played.append(bias)
            gaps = list(base + 0.15 * bias)  # more skipping, other decks: the gaps shift with the bias
            skips = int(sum(g <= bias for g in gaps))
            return self.rows(skips, 400 - skips, gaps, 2)

        res = skc.calibrate(play, target=0.154, tol=0.03, max_rounds=5)
        self.assertTrue(res["landed"])
        self.assertLessEqual(abs(res["skip_rate"] - 0.154), 0.03)
        self.assertEqual(0.0, played[0])  # round 0 is the argmax
        self.assertGreater(len(played), 1)
        self.assertTrue(res["rounds"][-1]["landed"])
        self.assertTrue(all("skip_rate" in r for r in res["rounds"]))

    def test_a_round_that_is_already_within_the_tolerance_ends_the_calibration(self):
        res = skc.calibrate(lambda b: self.rows(60, 340, [float(i) for i in range(400)]), target=0.154, tol=0.03)
        self.assertTrue(res["landed"])
        self.assertEqual((0.0, 1), (res["bias"], len(res["rounds"])))

    def test_rows_without_gaps_stop_the_calibration_with_a_reason(self):
        res = skc.calibrate(lambda b: self.rows(5, 395), target=0.154)
        self.assertFalse(res["landed"])
        self.assertIn("no skip gaps", res["rounds"][0]["stop"])

    def test_a_calibration_that_never_lands_reports_the_closest_round(self):
        def play(bias: float) -> list[dict]:
            return self.rows(4, 396, [float(i % 40) for i in range(400)])  # the world ignores the bias: 1 in 100, whatever it is

        res = skc.calibrate(play, target=0.154, tol=0.03, max_rounds=3)
        self.assertFalse(res["landed"])
        self.assertIsNotNone(res["bias"])
        self.assertLessEqual(len(res["rounds"]), 3)

    def test_too_thin_a_round_is_not_a_landing(self):
        res = skc.calibrate(lambda b: self.rows(15, 85, [float(i) for i in range(100)]), target=0.15, tol=0.05, max_rounds=1, min_screens=150)
        self.assertFalse(res["landed"])

    def test_the_seed_agent_puts_the_bias_in_the_macro_and_drops_the_search_options(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            c = blended_stub(Path(tmp.name), "cand", 0.5, b"prior bytes")
            meta = json.loads((c / "agent.json").read_text())
            meta.update(search_schedule=SCHEDULE, potion_hold={"hp_frac": 0.3})
            (c / "agent.json").write_text(json.dumps(meta))
            sha8 = gen_loop.file_sha256(c / "human_prior.npz")[:8]
            a = skc.seed_agent(c, 0.35, keep_search=False)
            self.assertEqual(f"vrun+pi_h:lambda=0.5:pi_h={sha8}:pick=skipcal0.35", a.macro)
            self.assertEqual((None, None), (a.stakes, a.hold))
            b = skc.seed_agent(c, -0.1, keep_search=True)
            self.assertEqual((STAKES, 30), (b.stakes, b.hold))
            self.assertEqual(("skipcal", -0.1), mr.parse_pick(b.macro))
            plain = vrun_stub(Path(tmp.name), "plain")
            with self.assertRaisesRegex(ValueError, "not a blended"):
                skc.seed_agent(plain, 0.1, keep_search=False)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
