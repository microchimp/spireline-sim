"""The round 1 follow-up (D55 addendum 12, docs/design/2026-11-08-labels-vs-blocks.md section 6; rl/round1*.py, rl/hindsight.py, docs/rl/ROUND.md):
the config defaults, the human-weight selection, the report-only guards, labels two a run per act with at least half in Act 2 and the ancient
as a kind, the continuations on the x8 schedule, and chain.json's sha256 at every handoff. Pure data and mocks: nothing here plays the game."""

import dataclasses
import json
import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import hindsight  # noqa: E402
import round1  # noqa: E402
import round1_data as rd  # noqa: E402
import round1_judgement as rj  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover
    torch = None

from test_round1 import rec  # noqa: E402


class DefaultsTests(unittest.TestCase):
    """The weights call of D55 addendum 12 as the config's defaults."""

    def test_the_prior_refit_is_the_human_prior_with_the_anchor(self):
        P = rd.load_config()["prior"]
        self.assertEqual({"human": 1.0, "agent": 0.0, "label": 0.0}, P["weights"])
        self.assertEqual(1.0, P["kl_beta"])
        self.assertEqual((0.5, "fixed", {}), (P["lambda"], P["lambda_rule"], P["lambda_acts"]))  # no label-chosen lambda, no act gate yet

    def test_the_judgement_weight_grid_holds_the_memos_three_and_is_chosen_on_clears_act2(self):
        J = rd.load_config()["judgement"]
        self.assertTrue({0.25, 1.0, 4.0} <= set(J["human_weights"]))
        self.assertEqual(("clears_act2", 0.0), (J["weight_rule"], J["act1_tolerance"]))
        self.assertEqual(("report", -0.1, 5.0), (J["guards"], J["label_bound_floors"], J["skip_guard_points"]))

    def test_the_label_mix_is_two_a_run_per_act_half_in_act_2_with_ancients_and_x8_continuations(self):
        L = rd.load_config()["label"]
        self.assertEqual((2, 2, 8, True), (L["k"], L["per_act"], L["cont_boss_mult"], L["own_schedule"]))
        shares = rd.act_shares(L["quota"])
        self.assertEqual(1200, shares["total"])
        self.assertGreaterEqual(shares["a2"], 0.5)
        for cell in ("ancient/a1", "ancient/a2", "card/a2", "rest/a2"):
            self.assertGreater(L["quota"][cell], 0, cell)
        self.assertGreater(L["quota"]["card/a2"], L["quota"]["shop/a2"])  # the deck screens first
        self.assertEqual(1.0, L["rates"]["ancient"])  # a kind with no rate is never labelled

    def test_the_kinds_the_config_names_are_the_kinds_hindsight_labels(self):
        self.assertEqual(hindsight.LABEL_KINDS, rd.LABEL_KINDS)
        self.assertEqual("ancient", hindsight.KIND_OF_TYPE["neow"])
        # the kind codes the streams' entropy uses are what they were for the four old kinds
        self.assertEqual({"card": 1, "rest": 2, "route": 3, "shop": 4, "ancient": 5}, hindsight.KIND_CODE)

    def test_a_quota_under_half_in_act_2_is_refused(self):
        with self.assertRaisesRegex(ValueError, "Act 2"):
            rd.load_config(label={"quota": {"card/a1": 300, "card/a2": 100}})
        rd.load_config(label={"quota": {"card/a1": 100, "card/a2": 100}})  # exactly half
        rd.load_config(label={"quota": {"card/a1": 300, "card/a2": 100}, "act2_min_share": 0.25})  # the minimum is a parameter

    def test_other_bad_values_are_refused(self):
        for over in ({"label": {"quota": {"mystery/a1": 5, "card/a2": 5}}}, {"label": {"quota": {"card/a3": 5}}}, {"label": {"per_act": -1}},
                     {"label": {"own_schedule": False}}, {"prior": {"lambda_acts": {"4": 0.0}}}, {"prior": {"lambda_acts": {"1": -1.0}}},
                     {"judgement": {"guards": "ignore"}}, {"judgement": {"weight_rule": "best"}}):
            with self.assertRaises(ValueError, msg=str(over)):
                rd.load_config(**over)
        c = rd.load_config(prior={"lambda_acts": {"1": 0.0}})  # replaced, not merged into the empty default
        self.assertEqual({"1": 0.0}, c["prior"]["lambda_acts"])


class SelectionTests(unittest.TestCase):
    INC = {"clears_act1": 0.50, "clears_act2": 0.60, "mean": 0.55}

    def grid(self, rows):
        return [{"human_weight": w, "clears_act1": a1, "clears_act2": a2, "mean": (a1 + a2) / 2} for w, a1, a2 in rows]

    def test_the_lowest_clears_act2_among_the_weights_that_keep_clears_act1(self):
        g = self.grid([(0.25, 0.49, 0.58), (1.0, 0.50, 0.55), (4.0, 0.60, 0.40)])  # 4 is best on Act 2 and worse on Act 1: out
        sel = rj.select_human_weight(g, self.INC)
        self.assertEqual(1.0, sel["chosen"])
        self.assertEqual([0.25, 1.0, 4.0], sel["grid"])
        self.assertEqual([True, True, False], [r["eligible"] for r in sel["table"]])
        self.assertIsNone(sel["fallback"])
        self.assertEqual(0.60, sel["incumbent"]["clears_act2"])
        self.assertIn("clears-Act-2", sel["reason"])

    def test_the_old_rule_takes_the_lowest_mean(self):
        g = self.grid([(0.25, 0.49, 0.58), (1.0, 0.50, 0.55), (4.0, 0.60, 0.40)])
        self.assertEqual(4.0, rj.select_human_weight(g, self.INC, "mean")["chosen"])

    def test_the_tolerance_widens_the_eligible_set(self):
        g = self.grid([(0.25, 0.49, 0.58), (1.0, 0.50, 0.55), (4.0, 0.52, 0.40)])
        self.assertEqual(1.0, rj.select_human_weight(g, self.INC, "clears_act2", 0.0)["chosen"])
        self.assertEqual(4.0, rj.select_human_weight(g, self.INC, "clears_act2", 0.03)["chosen"])

    def test_when_no_weight_keeps_clears_act1_the_lowest_clears_act1_is_kept_and_it_says_so(self):
        g = self.grid([(0.25, 0.55, 0.50), (1.0, 0.53, 0.52), (4.0, 0.60, 0.40)])
        sel = rj.select_human_weight(g, self.INC)
        self.assertEqual(1.0, sel["chosen"])
        self.assertIn("no weight keeps", sel["fallback"])

    def test_a_figure_that_could_not_be_computed_ranks_last(self):
        nan = float("nan")
        g = [{"human_weight": 0.25, "clears_act1": 0.49, "clears_act2": nan, "mean": 0.49}, {"human_weight": 1.0, "clears_act1": 0.49,
                                                                                            "clears_act2": 0.70, "mean": 0.6}]
        self.assertEqual(1.0, rj.select_human_weight(g, self.INC)["chosen"])
        both = [{"human_weight": w, "clears_act1": 0.49, "clears_act2": nan, "mean": 0.49} for w in (0.25, 1.0)]
        sel = rj.select_human_weight(both, self.INC)
        self.assertEqual((0.25, True), (sel["chosen"], sel["fallback"] is not None))
        with self.assertRaises(ValueError):
            rj.select_human_weight([], self.INC)

    def test_a_missing_incumbent_figure_leaves_every_computable_weight_eligible(self):
        g = self.grid([(0.25, 0.9, 0.6), (1.0, 0.9, 0.5)])
        self.assertEqual(1.0, rj.select_human_weight(g, {"clears_act1": float("nan")})["chosen"])


class GuardTests(unittest.TestCase):
    J = rd.DEFAULTS["judgement"]

    def keep_doc(self, rank_pass=True, skip_cand=0.10, skip_inc=0.20):
        return {"all_computable_pass": True,
                "ranking": {"status": "ok", "pass": rank_pass, "checks": [{"name": "heal_rate", "pass": rank_pass}],
                            "shard": {"skip_rate": {"n": 3000, "candidate": skip_cand, "incumbent": skip_inc, "diff": skip_cand - skip_inc}}}}

    def leg(self, lo95, mean=0.0):
        return {"all": {"n": 240, "new_judgement_new_prior_minus_incumbent_blend": {"mean": mean, "se": 0.1, "lo95": lo95, "hi95": 0.3}}}

    def test_the_numbers_are_reported_in_both_modes(self):
        g = rj.guards_report(self.keep_doc(), self.leg(-0.2), self.J)
        self.assertEqual(("report", False), (g["mode"], g["decides_kept"]))
        self.assertFalse(g["label_bound"]["ok"])  # lo95 -0.2 is under the -0.1 floor a decision
        self.assertAlmostEqual(g["skip_guard"]["fall_points"], 10.0)
        self.assertFalse(g["skip_guard"]["ok"])  # a 10-point fall in card skips, the label leg does not favour the candidate
        self.assertFalse(g["pass"])
        self.assertTrue(g["ranking"]["pass"])
        self.assertEqual(["heal_rate"], [c["name"] for c in g["ranking"]["checks"]])
        gate = rj.guards_report(self.keep_doc(), self.leg(-0.2), {**self.J, "guards": "gate"})
        self.assertTrue(gate["decides_kept"])

    def test_the_skip_guard_is_waived_when_the_label_leg_favours_the_candidate(self):
        g = rj.guards_report(self.keep_doc(), self.leg(0.05, 0.2), self.J)
        self.assertTrue(g["skip_guard"]["ok"])
        self.assertTrue(g["label_bound"]["ok"])
        self.assertTrue(g["pass"])
        small = rj.guards_report(self.keep_doc(skip_cand=0.17), self.leg(-0.2), self.J)  # a 3-point fall: inside the 5
        self.assertTrue(small["skip_guard"]["ok"])

    def test_a_figure_that_is_not_there_does_not_fail_a_guard(self):
        g = rj.guards_report({"all_computable_pass": True}, {}, self.J)
        self.assertEqual((None, None, None), (g["ranking"]["pass"], g["label_bound"]["ok"], g["skip_guard"]["ok"]))
        self.assertTrue(g["pass"])
        self.assertFalse(g["ranking"]["ran"])

    def test_a_failed_ranking_check_is_reported(self):
        g = rj.guards_report(self.keep_doc(rank_pass=False), self.leg(0.05, 0.2), self.J)
        self.assertEqual((False, False), (g["ranking"]["pass"], g["pass"]))


class PerActPlanTests(unittest.TestCase):
    RATES = {"card": 1.0, "rest": 1.0, "route": 1.0, "shop": 1.0, "ancient": 1.0}

    def plan(self, **kw):
        return hindsight.Plan(per_act=2, rates=self.RATES, **kw)

    def test_two_labels_a_run_in_each_act(self):
        plan, none = self.plan(), Counter()
        self.assertIsNone(plan.wants("R", 1, "card", 0, 5, none, none))
        self.assertEqual("run_cap", plan.wants("R", 1, "card", 0, 5, Counter({"card/a1": 1}), none))  # one a kind and act
        self.assertEqual("run_cap", plan.wants("R", 1, "rest", 0, 5, Counter({"card/a1": 1, "shop/a1": 1}), none))  # two in the act
        # the second act has its own two, whatever Act 1 took
        two = Counter({"card/a1": 1, "shop/a1": 1})
        self.assertIsNone(plan.wants("R", 1, "card", 1, 20, two, none))
        self.assertIsNone(plan.wants("R", 1, "rest", 1, 20, Counter({**two, "card/a2": 1}), none))
        self.assertEqual("run_cap", plan.wants("R", 1, "route", 1, 20, Counter({**two, "card/a2": 1, "rest/a2": 1}), none))

    def test_the_run_is_full_when_every_labelled_act_is(self):
        plan = self.plan()
        self.assertFalse(plan.run_full(Counter({"card/a1": 1, "shop/a1": 1})))  # Act 2 is still to come
        self.assertTrue(plan.run_full(Counter({"card/a1": 1, "shop/a1": 1, "card/a2": 1, "rest/a2": 1})))
        self.assertTrue(plan.run_full(Counter({"card/a1": 1, "shop/a1": 1}), act=0))
        self.assertFalse(plan.run_full(Counter(), act=0))

    def test_without_per_act_the_plan_is_per_run_as_it_was(self):
        old = hindsight.Plan(per_run=2, rates=self.RATES)
        self.assertEqual(0, old.per_act)
        self.assertEqual("run_cap", old.wants("R", 1, "rest", 1, 20, Counter({"card/a1": 1, "shop/a1": 1}), Counter()))  # two a run in all
        self.assertTrue(old.run_full(Counter({"card/a1": 1, "shop/a1": 1})))
        self.assertEqual(0, dataclasses.asdict(old)["per_act"])

    def test_the_ancient_is_a_kind_with_its_own_cell_and_no_rate_by_default(self):
        plan = hindsight.Plan(per_act=2, rates=self.RATES, quota={"ancient/a2": 2})
        self.assertIsNone(plan.wants("R", 0, "ancient", 1, 18, Counter(), Counter()))
        self.assertEqual("quota", plan.wants("R", 0, "ancient", 1, 18, Counter(), Counter({"ancient/a2": 2})))
        self.assertEqual("post_boss", plan.wants("R", 0, "ancient", 1, 33, Counter(), Counter()))
        self.assertEqual("act", plan.wants("R", 0, "ancient", 2, 34, Counter(), Counter()))  # Act 3 is not labelled
        self.assertEqual("coin", hindsight.Plan().wants("R", 0, "ancient", 0, 1, Counter(), Counter()))  # an old caller's rates name no ancient

    def test_the_worker_rebuilds_the_plan_with_per_act(self):
        plan = self.plan(quota={"card/a2": 5})
        with tempfile.TemporaryDirectory() as d:
            try:
                hindsight._W.clear()
                with mock.patch("sts2_gym.procutil.lower_priority"):
                    hindsight._init_worker(d, "agent", "a3-v1", "", False, dataclasses.asdict(plan), "", True, "a3-v1@h1e4b8")
                self.assertEqual((2, "a3-v1@h1e4b8"), (hindsight._W["plan"].per_act, hindsight._W["cont_stakes"]))
                hindsight._W.clear()
                with mock.patch("sts2_gym.procutil.lower_priority"):
                    hindsight._init_worker(d, "agent", "a3-v1", "", False, dataclasses.asdict(plan), "", True)
                self.assertIsNone(hindsight._W["cont_stakes"])  # a caller that names none: the continuations search the agent's own schedule
            finally:
                hindsight._W.clear()


class ContinuationScheduleTests(unittest.TestCase):
    def test_the_boss_multiplier_is_lowered_and_nothing_else_changes(self):
        self.assertEqual("a3-v1@h1e4b8", hindsight.continuation_stakes("a3-v1@h1e4b16", 8))
        self.assertEqual("a3-v1@h2e4b8", hindsight.continuation_stakes("a3-v1@h2e4b16", 8))  # the other fight kinds keep their multipliers
        self.assertIsNone(hindsight.continuation_stakes("a3-v1@h1e4b16", 0))  # none asked
        self.assertIsNone(hindsight.continuation_stakes("a3-v1@h1e4b8", 8))  # already there
        self.assertIsNone(hindsight.continuation_stakes("a3-v1@h1e4b4", 8))  # never raised
        self.assertIsNone(hindsight.continuation_stakes(None, 8))  # a player with no stakes schedule has no boss multiplier to lower

    def test_the_continuation_player_searches_x8_and_the_replay_player_keeps_the_agents_own(self):
        import gen_loop

        agent = gen_loop.Agent(path="x", macro="vrun", stakes="a3-v1@h1e4b16")
        with mock.patch.object(gen_loop, "load_agent", return_value=agent), mock.patch.object(gen_loop.Agent, "macro_policy", return_value=object()):
            own = hindsight.build_player("x", config="a3-v1", own_schedule=True)
            cont = hindsight.build_player("x", config="a3-v1", own_schedule=True, stakes="a3-v1@h1e4b8")
            plain = hindsight.build_player("x", config="a3-v1", own_schedule=False)
            with self.assertRaises(ValueError):
                hindsight.build_player("x", config="a3-v1", own_schedule=False, stakes="a3-v1@h1e4b8")
        self.assertEqual("a3-v1@h1e4b16", own.agent.config_name("a3-v1"))
        self.assertEqual("a3-v1@h1e4b8", cont.agent.config_name("a3-v1"))
        self.assertEqual(("a3-v1@h1e4b16", "a3-v1@h1e4b8"), (own.play_fight.cfg.name, cont.play_fight.cfg.name))
        self.assertEqual("a3-v1", plain.play_fight.cfg.name)  # without own_schedule the lab's config, as before

    def test_only_the_continuations_get_the_schedule_of_their_own(self):
        seen = []
        try:
            hindsight._W.clear()
            hindsight._W.update(agent_dir="x", config="a3-v1", own_schedule=True, judge=None, keep_doors=False, cont_stakes="a3-v1@h1e4b8", players={})
            with mock.patch.object(hindsight, "build_player", side_effect=lambda *a, **k: seen.append(k) or object()):
                hindsight._player((0.1, 0.02))  # the replay of the exploring run
                hindsight._player(None)  # the continuations
        finally:
            hindsight._W.clear()
        self.assertEqual([None, "a3-v1@h1e4b8"], [k["stakes"] for k in seen])

    def test_the_parser_takes_the_new_options_and_a_schedule_needs_own_schedule(self):
        a = hindsight.build_parser().parse_args(["pilot", "--out", "o", "--per-act", "2", "--cont-boss-mult", "8", "--own-schedule"])
        self.assertEqual((2, 8), (a.per_act, a.cont_boss_mult))
        d = hindsight.build_parser().parse_args(["pilot", "--out", "o"])
        self.assertEqual((0, 0), (d.per_act, d.cont_boss_mult))  # the old command line is the old plan
        with tempfile.TemporaryDirectory() as t, mock.patch.object(hindsight, "blocked", return_value=None):
            bad = hindsight.build_parser().parse_args(["pilot", "--out", str(Path(t) / "o"), "--runs-root", t, "--cont-boss-mult", "8"])
            self.assertEqual(2, hindsight.pilot_cmd(bad))

    def test_the_manifest_names_both_schedules(self):
        import gen_loop

        agent = gen_loop.Agent(path="x", macro="vrun", stakes="a3-v1@h1e4b16")
        with tempfile.TemporaryDirectory() as t, mock.patch.object(gen_loop, "load_agent", return_value=agent):
            plan = hindsight.Plan(per_act=2)
            doc = hindsight.manifest_of(Path(t), Path(t), plan, config="a3-v1", judge=None, lib=None, runs=[], tree=None, own_schedule=True,
                                        cont_stakes="a3-v1@h1e4b8")
            old = hindsight.manifest_of(Path(t), Path(t), plan, config="a3-v1", judge=None, lib=None, runs=[], tree=None)
        self.assertEqual({"schedule": "a3-v1@h1e4b8", "play_schedule": "a3-v1@h1e4b16"},
                         {k: doc["policy"]["continuation"][k] for k in ("schedule", "play_schedule")})
        self.assertEqual(2, doc["plan"]["per_act"])
        self.assertNotIn("schedule", old["policy"]["continuation"])  # an old caller's manifest is as it was


class LabelFileTests(unittest.TestCase):
    def test_schema_3_records_both_schedules(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "labels.d").mkdir()
            with open(out / "labels.d" / "S1.jsonl", "w", encoding="utf-8") as fh:
                for i, (kind, act, cs) in enumerate((("card", 0, "a3-v1@h1e4b8"), ("ancient", 1, "a3-v1@h1e4b8"), ("rest", 1, None))):
                    r = rec({0: 1.0, 1: -1.0}, seed=f"S{i}")
                    r.update(id=f"S{i}|d001|{kind}|f03", kind=kind, act=act)
                    if cs:
                        r["cont_schedule"] = cs
                    fh.write(json.dumps(r) + "\n")
            round1.consolidate_labels(out, rd.load_config()["label"], "a3-v1@h1e4b16")
            rows = [json.loads(x) for x in (out / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
            summ = json.loads((out / "label-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(3, rd.LABEL_SCHEMA)
        self.assertEqual({"a3-v1@h1e4b16"}, {r["schedule"] for r in rows})
        self.assertEqual(["a3-v1@h1e4b8", "a3-v1@h1e4b8", "a3-v1@h1e4b16"], [r["cont_schedule"] for r in rows])  # an old record: the play schedule
        self.assertEqual({"a1": 1, "a2": 2}, {k: summ["by_act"][k] for k in ("a1", "a2")})
        self.assertAlmostEqual(2 / 3, summ["by_act"]["act2_share"])
        self.assertEqual({"play": "a3-v1@h1e4b16", "continuations": ["a3-v1@h1e4b16", "a3-v1@h1e4b8"]}, summ["schedules"])

    def test_label_mix_counts_by_act_and_cell(self):
        recs = [{"kind": "card", "act": 0}, {"kind": "card", "act": 1}, {"kind": "ancient", "act": 1}, {"kind": "card", "act": 1}]
        self.assertEqual({"a1": 1, "a2": 3, "cells": {"ancient/a2": 1, "card/a1": 1, "card/a2": 2}}, round1.label_mix(recs))

    def test_the_label_step_hands_the_plan_its_options(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = round1.Ctx(rd.load_config(), Path(d))
            agent = mock.Mock(path=str(ctx.agent_root / "gen13" / "candidate-stakes"), generation=13, macro="m", stakes="a3-v1@h1e4b16")
            agent.describe.return_value = {"prior_sha256": "p"}
            ctx.p("play").mkdir(parents=True)
            ctx.p("play/play.json").write_text("{}", encoding="utf-8")
            seen = {}

            def pilot(ns):
                seen.update(vars(ns))
                (Path(ns.out)).mkdir(parents=True, exist_ok=True)
                (Path(ns.out) / "manifest.json").write_text("{}", encoding="utf-8")
                return 0

            with mock.patch.object(round1, "play_runs", return_value=(agent, "a3-v1@h1e4b16", [])), mock.patch("hindsight.pilot_cmd", pilot):
                round1.step_label(ctx)
        L = ctx.cfg["label"]
        self.assertEqual((2, 8, True), (seen["per_act"], seen["cont_boss_mult"], seen["own_schedule"]))
        self.assertEqual(L["rates"], json.loads(seen["rates"]))
        self.assertIn("ancient/a2=100", seen["quota"])


@unittest.skipIf(torch is None, "torch")
class PriorSourceTests(unittest.TestCase):
    def setUp(self):
        import human_prior

        human_prior.stack()
        import round1_prior as rp

        self.rp = rp

    def test_a_source_with_weight_zero_is_neither_trained_on_nor_steers_the_stop(self):
        class Sized(list):
            pass

        sets = {"agent": Sized([1]), "label": Sized([1, 2])}
        self.assertEqual([], self.rp.active_sources({"human": 1.0, "agent": 0.0, "label": 0.0}, sets))
        self.assertEqual(["label"], self.rp.active_sources({"human": 1.0, "agent": 0.0, "label": 0.5}, sets))
        self.assertEqual(["agent", "label"], self.rp.active_sources({"human": 1.0, "agent": 0.5, "label": 1.0}, sets))
        self.assertEqual([], self.rp.active_sources({"agent": 1.0}, {"agent": Sized()}))  # an empty set is not trained on
        extra = {"agent": {"n": 9, "nll": 2.0}, "label": {"n": 9, "nll": 3.0}}
        self.assertEqual(1.0, self.rp.stop_score(1.0, extra, {"agent": 0.0, "label": 0.0}))
        self.assertEqual(3.0, self.rp.stop_score(1.0, extra, {"agent": 1.0, "label": 0.0}))
        self.assertEqual(6.0, self.rp.stop_score(1.0, extra, {"agent": 1.0, "label": 1.0}))
        self.assertEqual(1.0, self.rp.stop_score(1.0, {"agent": {"n": 0}}, {"agent": 1.0}))  # no figure: nothing to add

    def test_the_ancient_is_a_screen_the_sets_and_the_leg_know(self):
        items = [rd.Item(rows=np.zeros((3, 4), np.float32), screen="ancient", act=1, seed=f"s{i}", chosen=0, played=[0, 1, 2],
                         A={0: -1.0, 1: 2.0, 2: -1.0}, k=2, kinds=["neow.relic"] * 3, groups=[0, 0, 0]) for i in range(30)]
        vals = [[0.0, 1.0, 0.0]] * 30
        leg = rd.leg_picks(items, {"judgement": (vals, [[0.0] * 3] * 30, 0.0)})
        self.assertEqual(30, leg["ancient"]["n"])
        self.assertEqual(30, leg["act2"]["n"])
        self.assertNotIn("card", leg)

    def test_a_per_act_lambda_gives_each_act_its_own_blend_in_the_label_leg(self):
        items = []
        for i in range(40):
            act = i % 2
            items.append(rd.Item(rows=np.zeros((3, 4), np.float32), screen="card", act=act, seed=f"s{i}", chosen=0, played=[0, 1, 2],
                                 A={0: -1.0, 1: 2.0, 2: -1.0}, k=2, kinds=["card.take"] * 3, groups=[0, 1, 2]))
        vals = [[0.0, 1.0, 0.0]] * 40  # V takes option 1 (A +2)
        prior = [[0.0, -9.0, 0.0]] * 40  # the prior wants option 0 (A -1) at lambda inf
        act1_plain = lambda it: 0.0 if it.act == 0 else math.inf  # noqa: E731
        leg = rd.leg_picks(items, {"v": (vals, prior, 0.0), "prior": (vals, prior, math.inf), "gated": (vals, prior, act1_plain)})
        self.assertAlmostEqual(leg["all"]["v"][0], 2.0)
        self.assertAlmostEqual(leg["all"]["prior"][0], -1.0)
        self.assertAlmostEqual(leg["all"]["gated"][0], 0.5)  # half the decisions (act 1) as V, half (act 2) as the prior
        self.assertAlmostEqual(leg["act1"]["gated"][0], 2.0)
        self.assertAlmostEqual(leg["act2"]["gated"][0], -1.0)


class ChainTests(unittest.TestCase):
    def fill(self, ctx, step, text="x"):
        for rel in round1.IO[step][1]:
            ctx.p(rel).parent.mkdir(parents=True, exist_ok=True)
            ctx.p(rel).write_text(text + rel, encoding="utf-8")

    def test_every_step_names_every_handoff_file_it_reads_and_leaves(self):
        for step, (needs, leaves) in round1.IO.items():
            self.assertTrue(set(needs) <= set(round1.PRODUCER), step)
            self.assertTrue(all(round1.PRODUCER[r] != step for r in needs), step)
        # what the candidate step reads: the prior and its refit report (the lambda), the judgement and its fit record
        self.assertTrue({"prior/weights.npz", "prior/refit.json", "judgement/vrun_fit.json", "judgement/run_value.npz"}
                        <= set(round1.IO["candidate"][0]))
        self.assertIn("candidate/candidate.json", round1.IO["block"][0])  # the block reads the candidate's folder too

    def test_a_step_records_sha256_and_size_of_every_input_and_output(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = round1.Ctx(rd.load_config(), Path(d))
            self.fill(ctx, "play")
            self.fill(ctx, "label")
            ctx.done("play")
            entry = ctx.done("label")
            self.assertEqual(round1.IO["label"][0], list(entry["inputs"]))
            self.assertEqual(round1.IO["label"][1], list(entry["outputs"]))
            for rel, got in {**entry["inputs"], **entry["outputs"]}.items():
                self.assertEqual(round1.sha256(ctx.p(rel)), got["sha256"], rel)
                self.assertEqual(ctx.p(rel).stat().st_size, got["bytes"], rel)
            self.assertEqual("play", entry["inputs"]["play/play.json"]["from"])
            self.assertEqual(entry["outputs"]["labels/labels.jsonl"]["sha256"], entry["sha256"])
            self.assertEqual(64, len(entry["config_sha256"]))
            self.assertEqual(entry["inputs"]["play/play.json"]["sha256"], ctx.chain()["steps"]["play"]["outputs"]["play/play.json"]["sha256"])

    def test_every_step_of_the_chain_carries_a_sha256(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = round1.Ctx(rd.load_config(), Path(d))
            for step in round1.STEPS:
                self.fill(ctx, step)
                ctx.done(step)
            for step in round1.STEPS:
                e = ctx.chain()["steps"][step]
                self.assertEqual(64, len(e["sha256"]), step)
                self.assertTrue(all(len(v["sha256"]) == 64 for v in {**e["inputs"], **e["outputs"]}.values()), step)
                self.assertEqual(set(round1.IO[step][0]), set(e["inputs"]), step)
                self.assertEqual(set(round1.IO[step][1]), set(e["outputs"]), step)

    def test_a_step_refuses_an_input_changed_since_its_producer_recorded_it(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = round1.Ctx(rd.load_config(), Path(d))
            self.fill(ctx, "play")
            self.fill(ctx, "label")
            ctx.done("play")
            ctx.need("label")  # untouched: fine
            ctx.p("play/play.json").write_text("edited by hand", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "changed since"):
                ctx.need("label")
            ctx.done("play")  # the producer records the new file: the handoff is whole again
            ctx.need("label")

    def test_a_file_placed_by_hand_with_no_record_is_not_refused(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = round1.Ctx(rd.load_config(), Path(d))
            self.fill(ctx, "play")
            ctx.need("label")
            self.assertEqual({}, ctx.done("play")["inputs"])

    def candidate_folder(self, root: Path) -> Path:
        cand = root / "agent" / "gen14" / "candidate-t"
        cand.mkdir(parents=True)
        (cand / "agent.json").write_text("{}", encoding="utf-8")
        (cand / "human_prior.npz").write_bytes(b"prior")
        return cand

    def test_the_judged_steps_check_the_candidate_folder_against_the_candidate_step(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = round1.Ctx(rd.load_config(), Path(d), dry=True)
            cand = self.candidate_folder(Path(d))
            files = round1.folder_files(cand)
            self.assertEqual({"agent.json", "human_prior.npz"}, set(files))
            ctx.p("candidate").mkdir(parents=True)
            ctx.p("candidate/candidate.json").write_text(json.dumps({"dir": str(cand), "files": files}), encoding="utf-8")
            entry = round1.step_judged(ctx, "smoke")
            self.assertEqual(files, {k.split("/", 1)[1]: v for k, v in entry["artifacts"].items()})
            (cand / "promotion.json").write_text("{}", encoding="utf-8")  # what a judgement writes there is not part of the identity
            (cand / "arms").mkdir()
            round1.step_judged(ctx, "block")
            self.assertTrue(set(round1.IO["block"][0]) <= set(ctx.chain()["steps"]["block"]["inputs"]))
            (cand / "human_prior.npz").write_bytes(b"another prior")
            with self.assertRaisesRegex(SystemExit, "human_prior.npz"):
                round1.step_judged(ctx, "smoke")


if __name__ == "__main__":
    unittest.main()
