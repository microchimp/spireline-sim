"""Lambda by act (2026-11-08, D55 addendum 12; rl/macro_recipe.py, rl/imitation_chooser.py, rl/gen_loop.py, rl/gen0.py `build --lambda-act`).

Pinned: the recipe's `:act<N>=<lambda>` parts and their one canonical form; the plain recipe unchanged byte for byte (every string a
candidate carried before, and the agent.json a plain build writes); the act of a floor (the probe's switch: floor 18, the Act 2 ancient);
the blended policy deciding an Act 1 screen exactly as `vrun` does and an Act 2 screen exactly as the plain blend does; the loader's
agreement between the recipe and agent.json; the builder and its command line.

No game and no emulator: synthetic options, stub candidate folders, the stored judgement and prior (as test_imitation_chooser).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import gen0  # noqa: E402
import gen_loop  # noqa: E402
import human_prior as hp  # noqa: E402
import human_prior_infer as hpi  # noqa: E402
import imitation_chooser as ic  # noqa: E402
import macro_recipe as mr  # noqa: E402
import run_fields  # noqa: E402
from run_train import features as F  # noqa: E402
from run_train import models  # noqa: E402

from tests.python.test_funnel import stub_candidate  # noqa: E402
from tests.python.test_gen12_plumbing import PRIOR_SHA  # noqa: E402
from tests.python.test_human_prior import _find_ids, synthetic_run  # noqa: E402
from tests.python.test_imitation_chooser import REAL_JUDGE, REAL_WEIGHTS, random_weights  # noqa: E402

SHA = "2ebfe6d7f25e6b12c7b95d7ec1835577f04fc2a29ca63d3fa82177582d50b41e"


class RecipeTests(unittest.TestCase):
    def test_the_plain_recipes_are_byte_for_byte_what_they_were(self):
        for args, want in (
            ((math.inf,), "vrun+pi_h:lambda=inf:pi_h=2ebfe6d7"),
            ((0.5,), "vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7"),
            ((0.25,), "vrun+pi_h:lambda=0.25:pi_h=2ebfe6d7"),
            ((0.5, mr.PICK_SAMPLE), "vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7:pick=sample"),
            ((0.5, mr.PICK_SKIPCAL, 0.35), "vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7:pick=skipcal0.35"),
        ):
            self.assertEqual(want, mr.macro_name(args[0], SHA, *args[1:]))
            self.assertEqual(want, mr.macro_name(args[0], SHA, *args[1:], acts=None))
            self.assertEqual(want, mr.macro_name(args[0], SHA, *args[1:], acts={}))
            self.assertEqual({}, mr.parse_act_lambdas(want))
            self.assertEqual(want, mr.recipe_of(want))
        # generation 13's whole recipe, as the ledger writes it
        self.assertEqual("vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7+stakes:a3-v1@h1e4b16",
                         mr.full_recipe(mr.macro_name(0.5, SHA), "a3-v1@h1e4b16", None))

    def test_an_act_part_names_the_acts_own_lambda(self):
        a1 = mr.macro_name(0.5, SHA, acts={1: 0.0})
        self.assertEqual("vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7:act1=0", a1)
        self.assertEqual("vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7:act1=0:act3=inf", mr.macro_name(0.5, SHA, acts={3: math.inf, 1: 0}))
        self.assertEqual("vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7:act1=0:pick=skipcal0.3", mr.macro_name(0.5, SHA, mr.PICK_SKIPCAL, 0.3, {1: 0}))
        self.assertEqual((0.5, "2ebfe6d7"), mr.parse_macro(a1))  # the base lambda and the prior are read through the act parts
        self.assertEqual({1: 0.0}, mr.parse_act_lambdas(a1))
        self.assertEqual({1: 0.0, 3: math.inf}, mr.parse_act_lambdas(mr.macro_name(0.5, SHA, acts={3: math.inf, 1: 0})))
        self.assertEqual((mr.PICK_SKIPCAL, 0.3), mr.parse_pick("vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7:act1=0:pick=skipcal0.3"))
        self.assertTrue(mr.is_blended(a1))
        self.assertEqual(a1, mr.recipe_of(a1))
        self.assertEqual(a1 + "+stakes:a3-v1@h1e4b16", mr.full_recipe(a1, "a3-v1@h1e4b16", None))
        self.assertEqual((a1, "a3-v1@h1e4b16", None), mr.split_recipe(a1 + "+stakes:a3-v1@h1e4b16"))

    def test_an_act_equal_to_the_base_is_not_written_and_a_recipe_has_one_form(self):
        self.assertEqual(mr.macro_name(0.5, SHA), mr.macro_name(0.5, SHA, acts={2: 0.5, 3: 0.5}))  # the same chooser, the same string
        self.assertEqual("vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7:act1=0", mr.macro_name(0.5, SHA, acts={1: 0, 2: 0.5}))
        for bad in ("vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7:act2=0.5",  # names the base twice
                    "vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7:act2=0:act1=0",  # out of order
                    "vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7:act1=0:act1=0.25",  # twice
                    "vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7:act4=0",  # no such act
                    "vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7:act1=x",
                    "vrun+pi_h:lambda=0.5:act1=0:pi_h=2ebfe6d7"):
            with self.assertRaises(ValueError, msg=bad):
                mr.parse_macro(bad)
            with self.assertRaises(ValueError, msg=bad):
                mr.parse_act_lambdas(bad)
        for bad_acts in ({0: 0.0}, {4: 0.0}, {1: -1.0}, {True: 0.0}):
            with self.assertRaises(ValueError, msg=str(bad_acts)):
                mr.macro_name(0.5, SHA, acts=bad_acts)

    def test_another_per_act_lambda_is_another_recipe(self):
        names = {mr.macro_name(0.5, SHA), mr.macro_name(0.5, SHA, acts={1: 0}), mr.macro_name(0.5, SHA, acts={1: 0.25}),
                 mr.macro_name(0.5, SHA, acts={2: 0}), mr.macro_name(0.5, SHA, acts={1: 0, 3: 0})}
        self.assertEqual(5, len(names))

    def test_the_act_of_a_floor_is_the_probes_switch(self):
        self.assertEqual([0, 0, 0, 1, 1, 2, 2], [mr.act_of_floor(f) for f in (0, 1, 17, 18, 33, 34, 48)])
        self.assertEqual(mr.ACT_LAST_FLOORS, run_fields.ACT_BOSS_FLOORS[:2])  # the same boss floors the run rows use
        acts = {1: 0.0}
        self.assertEqual([0.0, 0.0, 0.5, 0.5], [mr.lambda_for_floor(0.5, acts, f) for f in (0, 17, 18, 40)])
        self.assertEqual([0.5, 0.5], [mr.lambda_for_floor(0.5, None, f) for f in (0, 40)])

    def test_the_two_modules_agree_on_the_names(self):
        self.assertIs(ic.lambda_for_floor, mr.lambda_for_floor)


class PolicyTests(unittest.TestCase):
    """The policy on a decision built from a record's state, with the real judgement and prior (as test_imitation_chooser)."""

    @classmethod
    def setUpClass(cls):
        hp.stack()
        cls.t = hp.Tables()
        cls.nets = models.load_numpy(REAL_JUDGE)
        cls.prior = hpi.PriorScorer.load(REAL_WEIGHTS)

    def view_and_decision(self):
        from collections import Counter

        specs = hp.run_decisions(synthetic_run(_find_ids(False, 1)[0]), self.t, Counter(), Counter(), Counter())
        sp = next(s for s in specs if s["kind"] == "card")
        view = hp.spec_view(sp)
        options = tuple(hp._option(o, view, j) for j, o in enumerate(sp["options"]))
        return view, F.Decision("card_reward", options)

    def test_act_1_is_vrun_and_act_2_is_the_plain_blend(self):
        from run_train.policy import LookaheadPolicy

        view, d = self.view_and_decision()
        plain = LookaheadPolicy(REAL_JUDGE, 0.0, eps=0.0, W=53.0)
        full = ic.BlendedPolicy(REAL_JUDGE, self.prior, math.inf, W=53.0)  # the prior alone: it differs from V where it matters
        gated = ic.BlendedPolicy(REAL_JUDGE, self.prior, math.inf, W=53.0, act_lam={1: 0.0})
        act1 = dataclasses.replace(view, floor=5)
        act2 = dataclasses.replace(view, floor=18)
        self.assertEqual(plain.pick(act1, d, None).index, gated.pick(act1, d, None).index)
        self.assertIsNone(gated.last)  # an Act 1 screen reads no prior and says nothing, as lambda 0 never does
        pick = gated.pick(act2, d, None)
        self.assertEqual(full.pick(act2, d, None).index, pick.index)
        self.assertEqual("card", gated.last["screen"])
        self.assertEqual(full.last, gated.last)
        # the floor decides: the Act 1 boss floor is still Act 1, the next floor is Act 2's ancient
        for floor, act_one in ((0, True), (17, True), (18, False), (33, False), (34, False)):
            v = dataclasses.replace(view, floor=floor)
            gated.pick(v, d, None)
            self.assertEqual(act_one, gated.last is None, floor)

    def test_exploring_draws_are_the_plain_policys_in_each_act(self):
        from run_train.policy import LookaheadPolicy

        view, d = self.view_and_decision()
        kw = dict(temperature=0.5, eps=0.05, W=53.0)
        plain = LookaheadPolicy(REAL_JUDGE, 0.5, eps=0.05, W=53.0)
        blend = ic.BlendedPolicy(REAL_JUDGE, self.prior, 0.5, **kw)
        gated = ic.BlendedPolicy(REAL_JUDGE, self.prior, 0.5, act_lam={1: 0.0}, **kw)
        act1, act2 = dataclasses.replace(view, floor=5), dataclasses.replace(view, floor=20)
        for seed in range(25):
            self.assertEqual(plain.pick(act1, d, np.random.default_rng(seed)).index, gated.pick(act1, d, np.random.default_rng(seed)).index)
            self.assertEqual(blend.pick(act2, d, np.random.default_rng(seed)).index, gated.pick(act2, d, np.random.default_rng(seed)).index)
        # the probabilities the draw came from are the act's too
        self.assertEqual(plain.pick(act1, d, np.random.default_rng(3)).p_chosen, gated.pick(act1, d, np.random.default_rng(3)).p_chosen)
        self.assertEqual(blend.pick(act2, d, np.random.default_rng(3)).p_chosen, gated.pick(act2, d, np.random.default_rng(3)).p_chosen)

    def test_describe_names_the_acts_only_when_there_are_some(self):
        self.assertNotIn("lambda_acts", ic.BlendedPolicy(REAL_JUDGE, self.prior, 0.5, W=53.0).describe())
        self.assertEqual({"1": "0"}, ic.BlendedPolicy(REAL_JUDGE, self.prior, 0.5, W=53.0, act_lam={1: 0}).describe()["lambda_acts"])
        with self.assertRaises(ValueError):
            ic.BlendedPolicy(REAL_JUDGE, self.prior, 0.5, W=53.0, act_lam={4: 0.0})
        with self.assertRaises(ValueError):
            ic.BlendedPolicy(REAL_JUDGE, self.prior, 0.5, W=53.0, act_lam={1: -0.5})


def acts_stub(root: Path, name: str, lam: float, acts: dict | None, named: dict | None = None) -> Path:
    c = stub_candidate(root, name, PRIOR_SHA, 12)
    (c / "human_prior.npz").write_bytes(b"prior bytes")
    sha = gen_loop.file_sha256(c / "human_prior.npz")
    macro = mr.macro_name(lam, sha, acts=acts)
    meta = {"generation": 12, "macro": macro}
    if named is not None:
        meta["human_prior"] = {"file": "human_prior.npz", "sha256": sha, "lambda": mr.format_lambda(lam), **named}
    (c / "agent.json").write_text(json.dumps(meta))
    (c / "run_value.npz").write_bytes(b"head")
    return c


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_plain_blended_candidate_loads_as_before(self):
        a = gen_loop.load_agent(acts_stub(self.root, "plain", 0.5, None))
        self.assertEqual(((), 0.5), (a.lam_acts, a.lam))
        self.assertEqual(a.macro, a.recipe())
        self.assertEqual({k: v for k, v in a.describe().items()}, a.describe())  # nothing about acts in the description
        self.assertNotIn("lam_acts", a.describe())

    def test_the_recipe_carries_the_acts_into_the_agent_and_its_identity(self):
        c = acts_stub(self.root, "g", 0.5, {1: 0.0}, named={"lambda_acts": {"1": "0"}})
        a = gen_loop.load_agent(c)
        self.assertEqual(((1, 0.0),), a.lam_acts)
        self.assertTrue(a.macro.endswith(":act1=0"))
        self.assertEqual(a.macro, a.describe()["macro"])
        self.assertEqual(a.macro, gen_loop.candidate_recipe(a))
        self.assertNotEqual(gen_loop.candidate_recipe(gen_loop.load_agent(acts_stub(self.root, "p", 0.5, None))), gen_loop.candidate_recipe(a))
        with mock.patch("imitation_chooser.BlendedPolicy") as policy:
            a.macro_policy()
        self.assertEqual({1: 0.0}, policy.call_args.kwargs["act_lam"])
        with mock.patch("imitation_chooser.BlendedPolicy") as policy:
            gen_loop.load_agent(acts_stub(self.root, "q", 0.5, None)).macro_policy()
        self.assertIsNone(policy.call_args.kwargs["act_lam"])  # no acts: the policy is built as it was

    def test_agent_json_must_agree_with_the_recipe(self):
        with self.assertRaisesRegex(ValueError, "lambda_acts"):
            gen_loop.load_agent(acts_stub(self.root, "a", 0.5, {1: 0.0}, named={}))  # the recipe names act 1, agent.json does not
        with self.assertRaisesRegex(ValueError, "lambda_acts"):
            gen_loop.load_agent(acts_stub(self.root, "b", 0.5, None, named={"lambda_acts": {"1": "0"}}))  # agent.json names one the recipe lacks
        with self.assertRaisesRegex(ValueError, "lambda_acts"):
            gen_loop.load_agent(acts_stub(self.root, "c", 0.5, {1: 0.0}, named={"lambda_acts": {"1": "0.25"}}))
        gen_loop.load_agent(acts_stub(self.root, "d", 0.5, None, named={}))  # an older agent.json without the key loads

    def test_the_skip_calibration_keeps_the_acts(self):
        import skip_calibrate

        c = acts_stub(self.root, "s", 0.5, {1: 0.0})
        seeded = skip_calibrate.seed_agent(c, 0.3, keep_search=True)
        self.assertEqual("vrun+pi_h:lambda=0.5:pi_h=" + gen_loop.file_sha256(c / "human_prior.npz")[:8] + ":act1=0:pick=skipcal0.3", seeded.macro)
        self.assertEqual(((1, 0.0),), seeded.lam_acts)


class BuildTests(unittest.TestCase):
    """`gen0.py build --human-prior --lambda --lambda-act`: the candidate folder, its recipe, agent.json, its marker."""

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
        self.weights = self.root / "pi_h-v0" / "weights.npz"
        self.weights.parent.mkdir()
        np.savez(self.weights, meta=np.array(json.dumps({"variant": "t"})), **random_weights())
        self.markers = self.root / "train"
        self.sha = gen_loop.file_sha256(self.weights)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, *extra: str) -> int:
        return gen0.main(["build", "--prior", str(self.prior), "--agent-root", str(self.agents), "--candidate-generation", "7", "--why", "test",
                          "--no-export", "--marker-dir", str(self.markers), "--human-prior", str(self.weights), *extra])

    def meta(self, tag: str) -> dict:
        return json.loads((self.agents / "gen7" / f"candidate-{tag}" / "agent.json").read_text())

    def test_act_1_at_zero_builds_the_probes_recipe(self):
        self.assertEqual(0, self.build("--tag", "act2", "--lambda", "0.5", "--lambda-act", "1=0"))
        meta = self.meta("act2")
        want = f"vrun+pi_h:lambda=0.5:pi_h={self.sha[:8]}:act1=0"
        self.assertEqual(want, meta["macro"])
        self.assertEqual(want, meta["recipe"])
        self.assertEqual({"1": "0"}, meta["human_prior"]["lambda_acts"])
        self.assertEqual("0.5", meta["human_prior"]["lambda"])
        a = gen_loop.load_agent(self.agents / "gen7" / "candidate-act2")
        self.assertEqual((0.5, ((1, 0.0),), want), (a.lam, a.lam_acts, a.macro))
        self.assertEqual(["ready-5-g7-act2"], sorted(p.name for p in self.markers.iterdir()))

    def test_the_plain_build_writes_the_agent_json_it_always_wrote(self):
        self.assertEqual(0, self.build("--tag", "plain", "--lambda", "0.5"))
        self.assertEqual(0, self.build("--tag", "same", "--lambda", "0.5", "--lambda-act", "1=0.5", "--lambda-act", "3=0.5"))  # no-op acts
        plain, same = self.meta("plain"), self.meta("same")
        self.assertEqual(f"vrun+pi_h:lambda=0.5:pi_h={self.sha[:8]}", plain["macro"])
        self.assertEqual(set(plain["human_prior"]), {"file", "sha256", "lambda", "source", "guard"})  # no lambda_acts key
        for key in ("macro", "recipe", "human_prior"):
            self.assertEqual(plain[key], same[key], key)

    def test_the_option_needs_the_prior_and_a_clean_form(self):
        for bad in (["--lambda-act", "1"], ["--lambda-act", "x=0"], ["--lambda-act", "1=0", "--lambda-act", "1=0.25"], ["--lambda-act", "4=0"]):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.build("--tag", "bad", "--lambda", "0.5", *bad)
        with self.assertRaisesRegex(ValueError, "blended"):
            gen_loop.build_candidate(self.prior, self.base, self.agents, 7, why="t", export=False, tag="nop", lam_acts={1: 0.0})
        self.assertFalse((self.agents / "gen7").exists() and any((self.agents / "gen7").iterdir()))  # nothing was written for any of them

    def test_the_stakes_schedule_and_acts_sit_in_one_recipe(self):
        self.assertEqual(0, self.build("--tag", "full", "--lambda", "0.5", "--lambda-act", "1=0", "--stakes", "a3-v1@h1e4b16"))
        meta = self.meta("full")
        self.assertEqual(f"vrun+pi_h:lambda=0.5:pi_h={self.sha[:8]}:act1=0+stakes:a3-v1@h1e4b16", meta["recipe"])
        a = gen_loop.load_agent(self.agents / "gen7" / "candidate-full")
        self.assertEqual(("a3-v1@h1e4b16", ((1, 0.0),)), (a.stakes, a.lam_acts))
        self.assertEqual(meta["recipe"], gen_loop.candidate_recipe(a))


if __name__ == "__main__":
    unittest.main()
