"""The imitation chooser (rl/imitation_chooser.py, rl/human_prior_infer.py, rl/macro_recipe.py; D55 milestone 4 of the 2026-11-05 strategy memo).

Pinned: the numpy scorer of the human prior reproduces the torch net (and the fit's own input assembly, row for row); the blend's argmax
on synthetic options at lambda 0, 0.5 and inf; the guard; the rest site's two choices; the screens the blend decides and the ones it
leaves to the judgement; the recipe string; and the ledger's attempt count per recipe (generation 12's closed self-play line does not
count against the imitation line, a lambda or a prior of its own starts at attempt 1, a recipe has its own cap), with the marker forms
that name a tagged candidate.

No game and no emulator: stub arms, synthetic options, the stored shards of the prior (skipped where they are not).
"""

from __future__ import annotations

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
from run_train import features as F  # noqa: E402
from run_train import models  # noqa: E402

from tests.python.test_funnel import StubArms, make_block, stub_candidate  # noqa: E402
from tests.python.test_gen12_plumbing import (  # noqa: E402
    OLD_HEADER,
    PRIOR_SHA,
    legacy_tsv,
)
from tests.python.test_human_prior import (  # noqa: E402
    _find_ids,
    fake_judge,
    synthetic_run,
)
from tools.lab import jobs  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

REAL_WEIGHTS = Path("D:/spireline/referee/human-prior/pi_h-v0/weights.npz")
REAL_SHARDS = Path("D:/spireline/referee/human-prior/decisions")
REAL_DONORS = Path("D:/spireline/referee/human-prior/donors.npz")
REAL_JUDGE = Path("D:/spireline/referee/agent/gen11/candidate/run_value.npz")


def random_weights(seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shapes = {"embed.deck.weight": (64, 1154), "embed.relics.weight": (32, 296), "embed.potions.weight": (8, 63), "hidden1.weight": (256, 454),
              "hidden1.bias": (256,), "hidden2.weight": (256, 256), "hidden2.bias": (256,), "score.weight": (1, 256), "score.bias": (1,)}
    return {k: (rng.standard_normal(s) * 0.05).astype(np.float32) for k, s in shapes.items()}


def lp(*p: float) -> list[float]:
    """Log-probabilities of the probabilities `p` (given as they are)."""
    return [math.log(x) for x in p]


class ScorerTests(unittest.TestCase):
    def test_the_scorer_reproduces_the_torch_net_on_random_weights(self):
        if torch is None:
            self.skipTest("torch is not installed")
        params = random_weights()
        scorer = hpi.PriorScorer(params)
        rng = np.random.default_rng(5)
        dense = rng.standard_normal((40, 350)).astype(np.float32)
        counts = rng.integers(0, 3, (40, 1513)).astype(np.float32)
        net = hp.make_net()
        net.eval()
        net.load_state_dict({k: torch.from_numpy(v) for k, v in params.items()})
        with torch.no_grad():
            want = net(torch.from_numpy(dense), torch.from_numpy(counts)).numpy()
        np.testing.assert_allclose(scorer.logits(dense, counts), want, rtol=1e-4, atol=1e-4)

    def test_a_weights_file_loads_and_names_itself_by_its_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.npz"
            np.savez(path, meta=np.array(json.dumps({"variant": "x"})), **random_weights())
            s = hpi.PriorScorer.load(path)
            self.assertEqual(hpi.file_sha256(path), s.sha256)
            self.assertEqual({"variant": "x"}, s.meta)
            bad = dict(random_weights())
            del bad["score.bias"]
            with self.assertRaises(ValueError):
                hpi.PriorScorer(bad)

    @unittest.skipUnless(REAL_WEIGHTS.is_file() and REAL_SHARDS.is_dir() and REAL_DONORS.is_file() and torch is not None,
                         "the fitted prior and its stored decisions are not on this machine")
    def test_200_stored_decisions_score_alike_in_numpy_and_torch(self):
        """The exported weights on 200 decisions of the validation fifth (their stored rows, each with its fixed donor map): the numpy
        scorer and the torch net agree to 1e-4 in logits and in each decision's log-probabilities."""
        donors = hp.load_donors(REAL_DONORS)
        decs, denses, counts, roles, idents, rids = [], [], [], [], [], []
        off = got = 0
        for p in sorted(REAL_SHARDS.glob("shard-*.npz")):
            with np.load(p, allow_pickle=False) as z:
                dec = z["dec"]
                keep = dec[:, hp.C["fifth"]] == hp.VAL_FIFTH  # the validation fifth; the exam fifth is in no shard
                if not keep.any():
                    continue
                d = dec[keep].copy()
                sel = np.concatenate([np.arange(s, s + n) for s, n in zip(d[:, hp.C["start"]], d[:, hp.C["n"]])])
                d[:, hp.C["start"]] = np.concatenate([[0], np.cumsum(d[:, hp.C["n"]])[:-1]]) + off
                d[:, hp.C["run"]] += len(rids)
                rids.extend(str(x) for x in z["run_ids"])
                decs.append(d)
                denses.append(z["dense"][sel])
                counts.append(z["counts"][sel])
                roles.append(z["role"][sel])
                idents.append(z["ident"][sel])
                off += len(sel)
                got += len(d)
            if got >= 200:
                break
        ds = hp.Dataset(np.concatenate(decs), np.concatenate(denses), np.concatenate(counts), np.concatenate(roles), np.concatenate(idents), rids)
        ds._index(donors)
        n_dec = 200
        per = ds.dec[:n_dec, hp.C["n"]]
        rows = int(per.sum())
        a15, c = hp.row_donors(ds, donors, np.arange(n_dec), per)
        dense = hp.assemble_np(ds.dense[:rows], a15, c)
        cnt = ds.counts[:rows].astype(np.float32)
        scorer = hpi.PriorScorer.load(REAL_WEIGHTS)
        net = hp.make_net()
        net.eval()
        with np.load(REAL_WEIGHTS, allow_pickle=False) as z:
            net.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files if k != "meta"})
        with torch.no_grad():
            want = net(torch.from_numpy(dense), torch.from_numpy(cnt)).numpy()
        got_logits = scorer.logits(dense, cnt)
        self.assertLess(float(np.abs(got_logits - want).max()), 1e-4)
        start = 0
        for n in per:
            a = hpi.log_softmax(got_logits[start:start + n])
            b = torch.log_softmax(torch.from_numpy(want[start:start + n]).double(), 0).numpy()
            np.testing.assert_allclose(a, b, atol=1e-4)
            start += n

    def test_the_live_path_feeds_the_net_what_the_fit_fed_it(self):
        """A decision scored at play (`option_rows`: the afterstate rows with H, split by `models.value_inputs`) gives the inputs the
        fit assembled from its stored rows (`pack_dense`, `assemble_np`, the donor's map block and distance), when the map is the same."""
        hp.stack()
        train_id = _find_ids(False, 1)[0]
        t = hp.Tables()
        from collections import Counter

        specs = hp.run_decisions(synthetic_run(train_id), t, Counter(), Counter(), Counter())
        judge = fake_judge()
        scorer = hpi.PriorScorer(random_weights(11))
        checked = 0
        for sp in specs:
            view = hp.spec_view(sp)
            options = [hp._option(o, view, j) for j, o in enumerate(sp["options"])]
            live_rows = hpi.option_rows(judge, view, options)
            self.assertEqual((len(options), F.BASE_SIZE + F.FIGHT_BLOCK_SIZE), live_rows.shape)
            bases = np.stack(hp.spec_rows(sp)).astype(np.float32)
            h = np.zeros((len(bases), 8), np.float32)
            dense = hp.assemble_np(hp.pack_dense(bases, h), bases[:, 15], bases[:, F.C_OFFSET:F.C_OFFSET + F.C_SIZE])
            counts = hp.pack_counts(bases).astype(np.float32)
            live_dense, live_counts = models.value_inputs(live_rows)
            np.testing.assert_allclose(live_dense, dense, atol=1e-6)
            np.testing.assert_array_equal(live_counts, counts)
            np.testing.assert_allclose(scorer.row_logits(live_rows), scorer.logits(dense, counts), atol=1e-5)
            checked += 1
        self.assertGreater(checked, 5)

    def test_log_softmax_is_a_softmax(self):
        x = np.array([1.0, 2.0, -3.0])
        np.testing.assert_allclose(np.exp(hpi.log_softmax(x)).sum(), 1.0)
        np.testing.assert_allclose(hpi.log_softmax(x + 100.0), hpi.log_softmax(x))


class BlendTests(unittest.TestCase):
    def test_lambda_0_is_the_judgement_alone(self):
        for values, logp in (([10.0, 11.0, 9.0], lp(0.7, 0.2, 0.1)), ([5.0, 5.5], lp(0.99, 0.01)), ([3.0], lp(1.0))):
            c = ic.blend_pick(values, logp, 0.0)
            self.assertEqual(int(np.argmax(values)), c.index)
            self.assertFalse(c.guard)

    def test_lambda_inf_is_the_prior_alone_and_ties_go_to_the_judgement(self):
        c = ic.blend_pick([10.0, 14.0, 9.0], lp(0.5, 0.3, 0.2), math.inf)
        self.assertEqual(0, c.index)  # the prior's favourite, whatever V says
        self.assertEqual((0, 1), (c.prior_top, c.v_top))
        tied = ic.blend_pick([10.0, 14.0, 9.0], lp(0.4, 0.4, 0.2), math.inf)
        self.assertEqual(1, tied.index)  # two options the prior rates alike: V decides

    def test_lambda_half_weighs_the_floors_against_the_log_odds(self):
        # A: p .6, V 10.0; B: p .3, V 11.0: B wins by 1 floor against 0.5 ln 2 = 0.35
        self.assertEqual(1, ic.blend_pick([10.0, 11.0], lp(0.6, 0.3), 0.5).index)
        # with B only 0.3 floors better the prior's 2-to-1 preference holds
        self.assertEqual(0, ic.blend_pick([10.0, 10.3], lp(0.6, 0.3), 0.5).index)
        # the memo's figure: at lambda 0.5 a 3-to-1 preference is 0.55 floors
        self.assertEqual(0, ic.blend_pick([10.0, 10.5], lp(0.75, 0.25), 0.5).index)
        self.assertEqual(1, ic.blend_pick([10.0, 10.6], lp(0.75, 0.25), 0.5).index)

    def test_a_larger_lambda_moves_the_pick_from_the_judgement_to_the_prior(self):
        values, logp = [10.0, 11.5], lp(0.8, 0.1)
        picks = [ic.blend_pick(values, logp, lam).index for lam in (0.0, 0.25, 0.5, 1.0, math.inf)]
        self.assertEqual([1, 1, 1, 0, 0], picks)

    def test_the_guard_takes_a_near_zero_option_only_when_the_judgement_prefers_it_by_over_a_floor(self):
        # the prior gives B 1 in 100; at lambda 0.1 the blend alone would take it on a 0.6 floor edge (0.6 - 0.46 > 0)
        values, logp = [10.0, 10.6], lp(0.99, 0.01)
        c = ic.blend_pick(values, logp, 0.1)
        self.assertEqual((0, False), (c.index, c.guard))  # 0.6 floors is not over a floor: the prior's 99 in 100 stands
        c = ic.blend_pick([10.0, 11.5], logp, 0.1)
        self.assertEqual((1, True), (c.index, c.guard))  # 1.5 floors: the judgement overrules
        c = ic.blend_pick([10.0, 11.5], logp, math.inf)
        self.assertEqual((1, True), (c.index, c.guard))  # also at lambda = inf
        c = ic.blend_pick([10.0, 10.9], logp, math.inf)
        self.assertEqual((0, False), (c.index, c.guard))
        # exactly one floor is not "more than a floor"
        self.assertEqual(0, ic.blend_pick([10.0, 11.0], logp, math.inf).index)
        # lambda 0 is vrun: the guard is the prior's, so it is off
        self.assertEqual((1, False), (ic.blend_pick([10.0, 10.6], logp, 0.0).index, ic.blend_pick([10.0, 10.6], logp, 0.0).guard))

    def test_the_guard_picks_the_best_of_several_overruling_options(self):
        c = ic.blend_pick([10.0, 12.0, 13.0, 9.0], lp(0.95, 0.012, 0.008, 0.03), math.inf)
        self.assertEqual((2, True), (c.index, c.guard))

    def test_no_option_over_the_guard_floor_leaves_every_option_allowed(self):
        # 60 options of 1 in 60 each: none reaches 2 in 100; the prior's tie goes to V and nothing is overruled
        n = 60
        c = ic.blend_pick(list(range(n)), [math.log(1 / n)] * n, math.inf)
        self.assertEqual((n - 1, False), (c.index, c.guard))

    def test_ties_in_the_blend_go_to_the_judgement_then_the_first(self):
        self.assertEqual(1, ic.blend_pick([9.0, 10.0], [math.log(0.5)] * 2, 0.5).index)
        self.assertEqual(0, ic.blend_pick([10.0, 10.0], [math.log(0.5)] * 2, 0.5).index)

    def test_bad_input_is_refused(self):
        with self.assertRaises(ValueError):
            ic.blend_pick([1.0, 2.0], [0.0], 0.5)
        with self.assertRaises(ValueError):
            ic.blend_pick([], [], 0.5)

    # ---------------------------------------------------------------------------------------------------- the rest site

    def test_the_rest_site_takes_the_kind_by_the_marginal_then_the_smith_target(self):
        # heal 0.45 of the joint; two smith targets 0.30 and 0.25: the kind smith has 0.55, so smith, and the better target
        logits = [math.log(0.45), math.log(0.30), math.log(0.25)]
        groups = [0, 1, 1]
        c = ic.blend_rest([10.0, 10.0, 10.0], logits, groups, math.inf)
        self.assertEqual(1, c.index)
        self.assertEqual(1, c.prior_top)
        # the flat softmax would have healed: the fit's "rest kind" and "smith target" are the two choices
        self.assertEqual(0, int(np.argmax(logits)))

    def test_the_rest_site_blend_lets_the_judgement_move_the_kind(self):
        logits = [math.log(0.7), math.log(0.2), math.log(0.1)]
        groups = [0, 1, 1]
        self.assertEqual(0, ic.blend_rest([10.0, 10.5, 10.0], logits, groups, math.inf).index)  # the prior: heal
        self.assertEqual(1, ic.blend_rest([10.0, 12.0, 10.0], logits, groups, 0.5).index)  # smith is 2 floors better, over 0.5 ln(0.3/0.7)
        self.assertEqual(1, ic.blend_rest([10.0, 12.0, 10.0], logits, groups, 0.0).index)  # vrun
        self.assertEqual(2, ic.blend_rest([10.0, 10.0, 12.0], logits, groups, 0.0).index)  # vrun takes the best target by V

    def test_the_smith_target_is_the_priors_softmax_inside_the_kind(self):
        logits = [-5.0, 1.0, 2.0, 0.0]
        groups = [0, 1, 1, 1]
        c = ic.blend_rest([10.0, 10.0, 10.0, 10.0], logits, groups, math.inf)
        self.assertEqual(2, c.index)
        c = ic.blend_rest([10.0, 10.0, 10.0, 10.9], logits, groups, 1.0)  # V likes the third target by 0.9: the prior's 2 over 0 is 2 nats
        self.assertEqual(2, c.index)
        c = ic.blend_rest([10.0, 10.0, 10.0, 13.0], logits, groups, 1.0)
        self.assertEqual(3, c.index)

    def test_the_relic_given_rest_options_are_decided_by_the_judgement_within_the_kind(self):
        logits = [0.0, -4.0, 3.0, 3.0]  # heal, smith, and two relic-given others the fit does not tell apart
        groups = [0, 1, 2, 2]
        self.assertEqual(3, ic.blend_rest([9.0, 9.0, 9.0, 9.5], logits, groups, math.inf).index)
        self.assertEqual(2, ic.blend_rest([9.0, 9.0, 9.5, 9.0], logits, groups, math.inf).index)

    def test_the_rest_guard_acts_on_the_kinds(self):
        # the prior gives smith 1 in 100 in total; the judgement likes it by 1.5 floors
        logits = [math.log(0.99), math.log(0.01)]
        c = ic.blend_rest([10.0, 11.5], logits, [0, 1], math.inf)
        self.assertEqual((1, True), (c.index, c.guard))
        c = ic.blend_rest([10.0, 10.5], logits, [0, 1], math.inf)
        self.assertEqual((0, False), (c.index, c.guard))


def decision(dtype: str, *kinds: str, scorable: bool = True) -> F.Decision:
    return F.Decision(dtype, tuple(F.Option(k, i, scorable=scorable) for i, k in enumerate(kinds)))


class ScreenTests(unittest.TestCase):
    def test_the_deck_screens_are_the_ones_the_prior_was_fitted_on(self):
        self.assertEqual("card", ic.screen_of(decision("card_reward", "card.take", "card.take", "card.take", "card.skip")))
        self.assertEqual("card", ic.screen_of(decision("card_reward", "card.take", "card.take", "card.take", "card.take", "card.skip")))
        self.assertEqual("ancient", ic.screen_of(decision("neow", "neow.relic", "neow.relic", "neow.relic")))
        self.assertEqual("rest", ic.screen_of(decision("rest", "rest.heal", "rest.smith", "rest.smith", "rest.lift")))

    def test_every_other_screen_stays_with_the_judgement(self):
        for d in (decision("map", "map.node", "map.node"), decision("shop", "shop.card", "shop.relic", "shop.leave"),
                  decision("event", "event.option", "event.option"), decision("rewards", "reward.potion", "reward.proceed"),
                  decision("card_select", "select.upgrade", "select.upgrade"), decision("bundle", "bundle.pick", "bundle.pick"),
                  decision("card_reward", "card.take"), decision("card_reward", "card.take", "card.skip", "card.reroll"),
                  decision("card_reward", "card.take", "card.take", "card.sacrifice"),
                  decision("rest", "rest.heal", "rest.smith", "rest.leave"), decision("rest", "rest.heal", "rest.hatch"),
                  decision("neow", "neow.relic")):
            self.assertIsNone(ic.screen_of(d), [o.kind for o in d.options])
        self.assertIsNone(ic.screen_of(decision("card_reward", "card.take", "card.skip", scorable=False)))

    def test_rest_groups(self):
        self.assertEqual([0, 1, 2, 2], [ic.rest_group(F.Option(k, 0)) for k in ("rest.heal", "rest.smith", "rest.lift", "rest.dig")])


@unittest.skipUnless(REAL_WEIGHTS.is_file() and REAL_JUDGE.is_file(), "the fitted prior and the run judgement are not on this machine")
class PolicyTests(unittest.TestCase):
    """The policy on a decision built from a record's state, with the real judgement and prior."""

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

    def test_lambda_inf_takes_the_priors_argmax_lambda_0_is_vrun_and_the_run_field_says_who_decided(self):
        from run_train.policy import LookaheadPolicy

        view, d = self.view_and_decision()
        rows = hpi.option_rows(self.nets, view, d.options)
        top = int(np.argmax(self.prior.row_logits(rows)))
        pol = ic.BlendedPolicy(REAL_JUDGE, self.prior, math.inf, W=53.0)
        pick = pol.pick(view, d, None)
        self.assertEqual(top, pick.index)
        self.assertEqual({"screen": "card", "prior": 1, "v": int(top == int(np.argmax(pick.values))), "guard": 0}, pol.last)
        plain = LookaheadPolicy(REAL_JUDGE, 0.0, eps=0.0, W=53.0)
        zero = ic.BlendedPolicy(REAL_JUDGE, self.prior, 0.0, W=53.0)
        self.assertEqual(plain.pick(view, d, None).index, zero.pick(view, d, None).index)
        self.assertIsNone(zero.last)  # lambda 0 never consults the prior
        # a screen the prior was not fitted on is the judgement's, and says nothing
        shop = F.Decision("shop", d.options[:2])
        self.assertEqual(plain.pick(view, shop, None).index, pol.pick(view, shop, None).index)
        self.assertIsNone(pol.last)

    def test_the_describe_names_the_blend(self):
        pol = ic.BlendedPolicy(REAL_JUDGE, self.prior, 0.5, W=53.0)
        self.assertEqual(("lookahead+pi_h", "0.5", self.prior.sha256[:8]), tuple(pol.describe()[k] for k in ("kind", "lambda", "pi_h")))

    def test_exploring_samples_around_the_blend_and_stays_in_the_guarded_set(self):
        view, d = self.view_and_decision()
        pol = ic.BlendedPolicy(REAL_JUDGE, self.prior, 0.5, temperature=0.5, eps=0.0, W=53.0)
        rng = np.random.default_rng(0)
        picks = {pol.pick(view, d, rng).index for _ in range(60)}
        self.assertGreater(len(picks), 1)
        rows = hpi.option_rows(self.nets, view, d.options)
        p = np.exp(hpi.log_softmax(self.prior.row_logits(rows)))
        self.assertTrue(all(p[i] >= ic.GUARD_PROB for i in picks))


class RecipeTests(unittest.TestCase):
    SHA = "2ebfe6d7f25e6b12c7b95d7ec1835577f04fc2a29ca63d3fa82177582d50b41e"

    def test_the_macro_string_names_lambda_and_the_prior(self):
        self.assertEqual("vrun+pi_h:lambda=inf:pi_h=2ebfe6d7", mr.macro_name(math.inf, self.SHA))
        self.assertEqual("vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7", mr.macro_name(0.5, self.SHA))
        self.assertEqual("vrun+pi_h:lambda=0.25:pi_h=2ebfe6d7", mr.macro_name(0.25, self.SHA))
        self.assertEqual("vrun+pi_h:lambda=1:pi_h=2ebfe6d7", mr.macro_name(1.0, self.SHA))
        for lam in (math.inf, 0.0, 0.25, 0.5, 1.0, 3.0):
            self.assertEqual((lam, "2ebfe6d7"), mr.parse_macro(mr.macro_name(lam, self.SHA)))
        with self.assertRaises(ValueError):
            mr.macro_name(-1.0, self.SHA)
        for bad in ("vrun", "vrun+pi_h", "vrun+pi_h:lambda=x:pi_h=2ebfe6d7", "vrun+pi_h:lambda=inf:pi_h=2EBFE6D7"):
            with self.assertRaises(ValueError):
                mr.parse_macro(bad)

    def test_another_lambda_or_another_prior_is_another_recipe(self):
        a = mr.macro_name(math.inf, self.SHA)
        self.assertNotEqual(a, mr.macro_name(0.5, self.SHA))
        self.assertNotEqual(a, mr.macro_name(math.inf, "3bfba869" + self.SHA[8:]))

    def test_the_ledger_reads_the_plain_macros_as_one_recipe(self):
        self.assertEqual("", mr.recipe_of("vrun"))
        self.assertEqual("", mr.recipe_of("rules"))
        self.assertEqual("", mr.recipe_of(None))
        self.assertEqual("vrun+pi_h:lambda=inf:pi_h=2ebfe6d7", mr.recipe_of("vrun+pi_h:lambda=inf:pi_h=2ebfe6d7"))
        self.assertTrue(mr.is_blended("vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7"))
        self.assertFalse(mr.is_blended("vrun"))

    def test_the_two_modules_agree_on_the_prefix(self):
        self.assertEqual(mr.PREFIX, ic.PREFIX)


def blended_stub(root: Path, name: str, lam: float, prior_bytes: bytes, head: bytes = b"gen9 head", generation: int = 12) -> Path:
    """A candidate folder the agent loader accepts: generation 11's fight player (never loaded), a run head, and the prior's file."""
    c = stub_candidate(root, name, PRIOR_SHA, generation)
    (c / "human_prior.npz").write_bytes(prior_bytes)
    macro = mr.macro_name(lam, gen_loop.file_sha256(c / "human_prior.npz"))
    (c / "agent.json").write_text(json.dumps({"generation": generation, "macro": macro}))
    (c / "run_value.npz").write_bytes(head)
    return c


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_blended_candidate_loads_with_its_recipe_head_and_lambda(self):
        c = blended_stub(self.root, "imit", math.inf, b"prior bytes")
        a = gen_loop.load_agent(c)
        self.assertEqual(("vrun+pi_h:lambda=inf:pi_h=" + gen_loop.file_sha256(c / "human_prior.npz")[:8], math.inf), (a.macro, a.lam))
        self.assertEqual(str(c / "human_prior.npz"), a.human_prior)
        self.assertEqual(a.macro, a.describe()["macro"])
        self.assertEqual(gen_loop.file_sha256(c / "run_value.npz"), gen_loop.run_head_sha(a))  # the head is the judgement's, as for vrun
        self.assertEqual(a.macro, gen_loop.candidate_recipe(a))
        self.assertEqual(0.5, gen_loop.load_agent(blended_stub(self.root, "l05", 0.5, b"prior bytes")).lam)

    def test_the_folder_must_hold_the_prior_the_recipe_names(self):
        c = blended_stub(self.root, "imit", math.inf, b"prior bytes")
        (c / "human_prior.npz").write_bytes(b"another prior")
        with self.assertRaisesRegex(ValueError, "is not the prior"):
            gen_loop.load_agent(c)
        (c / "human_prior.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            gen_loop.load_agent(c)

    def test_agent_json_records_the_prior_and_the_loader_holds_it_to_the_recipe(self):
        c = blended_stub(self.root, "imit", math.inf, b"prior bytes")
        meta = json.loads((c / "agent.json").read_text())
        sha = gen_loop.file_sha256(c / "human_prior.npz")
        meta["human_prior"] = {"file": "human_prior.npz", "sha256": sha, "lambda": "inf"}
        (c / "agent.json").write_text(json.dumps(meta))
        self.assertEqual(math.inf, gen_loop.load_agent(c).lam)
        meta["human_prior"]["sha256"] = "0" * 64
        (c / "agent.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "is not the prior"):
            gen_loop.load_agent(c)
        meta["human_prior"]["sha256"] = sha
        meta["human_prior"]["lambda"] = "0.5"
        (c / "agent.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "disagree"):
            gen_loop.load_agent(c)

    def test_the_plain_macros_have_no_recipe(self):
        vrun = stub_candidate(self.root, "v", PRIOR_SHA, 12)
        (vrun / "agent.json").write_text(json.dumps({"generation": 12, "macro": "vrun"}))
        (vrun / "run_value.npz").write_bytes(b"x")
        a = gen_loop.load_agent(vrun)
        self.assertEqual(("", None, 0.0), (gen_loop.candidate_recipe(a), a.human_prior, a.lam))

    def test_an_exploring_agent_may_play_the_blend(self):
        a = gen_loop.load_agent(blended_stub(self.root, "imit", 0.5, b"p"))
        e = gen_loop.exploring_agent(a, 0.1, 0.02)
        self.assertEqual((0.1, 0.02, a.macro), (e.tau, e.eps, e.macro))

    def test_a_summary_names_the_recipe_so_another_recipes_arm_does_not_match(self):
        a = gen_loop.load_agent(blended_stub(self.root, "imit", math.inf, b"p"))
        b = gen_loop.load_agent(blended_stub(self.root, "l05", 0.5, b"p"))
        arm = self.root / "arm"
        arm.mkdir()
        (arm / "runs.jsonl").write_text("{}\n")
        (arm / "summary.json").write_text(json.dumps({"agent": a.describe()}))
        self.assertTrue(gen0.emu_arm_match(arm, a.describe())[0])
        ok, why = gen0.emu_arm_match(arm, b.describe())
        self.assertFalse(ok)
        self.assertIn("macro", why)


class AttemptCountTests(unittest.TestCase):
    """Attempts are counted per recipe: generation 12's two self-play refusals do not count against the imitation line."""

    PRIOR_BYTES = b"the fitted prior pi_h-v0"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.agents = self.root / "agent"
        self.agents.mkdir()
        self.tsv = legacy_tsv(self.agents)  # attempt 1 of generation 12 (vrun, no head named)
        # attempt 2 of generation 12: the same fighter, another head, a row of the new shape (macro column empty)
        gen_loop.append_promotion(self.tsv, {"cand_gen": 12, "seed_block": "E12-2026-11-05-r1", "cand_sha": PRIOR_SHA, "run_head_sha": "d12eef80",
                                             "decision": "rejected", "attempt": 2})
        (self.root / "seeds").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def promote(self, cand: Path, name: str, d_mean: float = -4.0):
        seeds, cand_rows, base_rows = make_block(d_mean, prefix=name[-1])
        seed_file = self.root / "seeds" / f"{name}.json"
        seed_file.write_text(json.dumps({"seeds": seeds}))
        with mock.patch.object(gen_loop, "run_arm", StubArms(cand_rows, base_rows)):
            return gen_loop.promote(cand, None, seed_file, agent_root=self.agents, boot=500, review_guard=True, sequential=True)

    def test_the_closed_line_holds_two_attempts_and_the_imitation_line_starts_at_one(self):
        rows = gen_loop.read_promotions(self.tsv)
        self.assertEqual(2, gen_loop.attempts_spent(self.tsv, 12, True))  # the self-play line is at its cap
        imit = "vrun+pi_h:lambda=inf:pi_h=2ebfe6d7"
        self.assertEqual(2, gen_loop.attempts_before(rows, 12, PRIOR_SHA, "f0a199fb"))  # counted as before the recipe column existed
        self.assertEqual(0, gen_loop.attempts_before(rows, 12, PRIOR_SHA, "f0a199fb", imit))
        self.assertIsNone(gen_loop.attempts_spent(self.tsv, 12, True, imit))
        self.assertEqual(2, gen_loop.attempts_spent(self.tsv, 12, True, ""))

    def test_the_first_imitation_candidate_is_judged_as_attempt_1_on_a_block_of_its_own(self):
        imit = blended_stub(self.root, "candidate-imit", math.inf, self.PRIOR_BYTES)
        macro = gen_loop.load_agent(imit).macro
        doc = self.promote(imit, "E12-2026-11-0A")
        self.assertEqual(macro, doc["macro"])
        rows = gen_loop.read_promotions(self.tsv)
        self.assertEqual(["", "", macro], [r.get("macro", "") for r in rows])
        self.assertEqual("1", rows[-1]["attempt"])  # attempt 1, though two other candidates of generation 12 were judged before
        self.assertTrue(rows[-1]["guard"].endswith("attempt 1/2") or "attempt 1/2" in rows[-1]["guard"])
        self.assertEqual(self.tsv.read_text().splitlines()[0].split("\t"), list(gen_loop.TSV_COLUMNS))
        self.assertEqual(OLD_HEADER.split("\t"), list(gen_loop.TSV_COLUMNS)[:len(OLD_HEADER.split("\t"))])  # the old columns keep their places
        self.assertEqual("macro", gen_loop.TSV_COLUMNS[-10])  # the keep test v2's nine columns follow it (gen_loop.TSV_COLUMNS[-9:])

    def test_the_lambda_half_candidate_is_a_recipe_of_its_own_and_gets_its_own_block(self):
        imit = blended_stub(self.root, "candidate-imit", math.inf, self.PRIOR_BYTES)
        l05 = blended_stub(self.root, "candidate-imit-l05", 0.5, self.PRIOR_BYTES)
        self.promote(imit, "E12-2026-11-0A")
        # the same fighter, the same head, the same prior file: only lambda differs, and the block of the first is not the second's
        self.assertEqual(gen_loop.candidate_identity(gen_loop.load_agent(imit)), gen_loop.candidate_identity(gen_loop.load_agent(l05)))
        with self.assertRaisesRegex(ValueError, "already judged another candidate"):
            self.promote(l05, "E12-2026-11-0A")
        doc = self.promote(l05, "E12-2026-11-0B")
        self.assertEqual("1", gen_loop.read_promotions(self.tsv)[-1]["attempt"])  # its own line, its own first attempt
        self.assertEqual(gen_loop.load_agent(l05).macro, doc["macro"])
        # the same pair under the same recipe judged again is not a new attempt
        rows = gen_loop.read_promotions(self.tsv)
        sha, head = gen_loop.candidate_identity(gen_loop.load_agent(imit))
        self.assertEqual(0, gen_loop.attempts_before(rows, 12, sha, head, gen_loop.load_agent(imit).macro))
        self.assertFalse(gen_loop.block_used_by_another(self.tsv, "E12-2026-11-0A", sha, head, gen_loop.load_agent(imit).macro))
        self.assertTrue(gen_loop.block_used_by_another(self.tsv, "E12-2026-11-0A", sha, head, gen_loop.load_agent(l05).macro))
        self.assertTrue(gen_loop.block_used_by_another(self.tsv, "E12-2026-11-0A", sha, head, ""))

    def test_a_recipe_has_its_own_cap_of_two_and_the_closed_line_still_does(self):
        imit = blended_stub(self.root, "candidate-imit", math.inf, self.PRIOR_BYTES)
        second = blended_stub(self.root, "candidate-imit-b", math.inf, self.PRIOR_BYTES, head=b"another head")
        third = blended_stub(self.root, "candidate-imit-c", math.inf, self.PRIOR_BYTES, head=b"a third head")
        self.promote(imit, "E12-2026-11-0A")
        self.promote(second, "E12-2026-11-0B")
        self.assertEqual(["1", "2"], [r["attempt"] for r in gen_loop.read_promotions(self.tsv)[-2:]])
        macro = gen_loop.load_agent(imit).macro
        self.assertEqual(2, gen_loop.attempts_spent(self.tsv, 12, True, macro))
        seeds, cand_rows, base_rows = make_block(-4.0, prefix="Z")
        fresh = self.root / "seeds" / "E12-2026-11-0Z.json"
        fresh.write_text(json.dumps({"seeds": seeds}))
        arms = StubArms(cand_rows, base_rows)
        with mock.patch.object(gen_loop, "run_arm", arms), self.assertRaisesRegex(ValueError, "already had 2 promotion attempts"):
            gen_loop.promote(third, None, fresh, agent_root=self.agents, boot=500, review_guard=True, sequential=True)
        self.assertEqual([], arms.calls)
        # the plain line is still at its cap, as it was
        plain = stub_candidate(self.root, "plain", PRIOR_SHA, 12)
        (plain / "agent.json").write_text(json.dumps({"generation": 12, "macro": "vrun"}))
        (plain / "run_value.npz").write_bytes(b"a plain third head")
        with mock.patch.object(gen_loop, "run_arm", arms), self.assertRaisesRegex(ValueError, "already had 2 promotion attempts"):
            gen_loop.promote(plain, None, fresh, agent_root=self.agents, boot=500, review_guard=True, sequential=True)

    def test_resolve_e_block_gives_each_recipe_a_block_that_no_other_recipe_used(self):
        seeds = self.root / "seeds"
        tpl = str(seeds / "E{k}-{date}.json")
        imit = gen_loop.load_agent(blended_stub(self.root, "candidate-imit", math.inf, self.PRIOR_BYTES))
        sha, head = gen_loop.candidate_identity(imit)
        kw = {"tsv": self.tsv, "also_dirs": [seeds], "n": 30}
        (seeds / "E12-2026-11-0A.json").write_text(json.dumps({"seeds": ["A"]}))
        gen_loop.append_promotion(self.tsv, {"cand_gen": 12, "seed_block": "E12-2026-11-0A", "cand_sha": sha, "run_head_sha": head,
                                             "macro": imit.macro, "decision": "rejected", "attempt": 1})
        again = gen_loop.resolve_e_block(tpl, 12, cand_sha=sha, head_sha=head, recipe=imit.macro, **kw)
        self.assertEqual("E12-2026-11-0A.json", again.name)  # the same recipe may be re-read on its own block
        other = gen_loop.resolve_e_block(tpl, 12, cand_sha=sha, head_sha=head, recipe="vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7", **kw)
        self.assertNotEqual("E12-2026-11-0A.json", other.name)

    def test_the_train_job_counts_the_tagged_candidates_recipe(self):
        """`gen0.py train` with a tagged marker: the cap check reads the built candidate's recipe, so the two refusals of the closed
        line do not stop the round, and a recipe at its own cap does."""
        agents = self.agents
        imit_dir = agents / "gen12" / "candidate-imit"
        imit_dir.parent.mkdir(parents=True)
        src = blended_stub(self.root, "src-imit", math.inf, self.PRIOR_BYTES)
        import shutil

        shutil.copytree(src, imit_dir)
        meta = json.loads((imit_dir / "agent.json").read_text())
        meta["built"] = {"why": "test"}
        (imit_dir / "agent.json").write_text(json.dumps(meta))
        markers = self.root / "train"
        markers.mkdir()
        (markers / "ready-11-g12-imit").write_text("x")
        args = ["train", "--marker-dir", str(markers), "--shards-root", str(self.root), "--agent-root", str(agents),
                "--promote", "--emu-python", "python", "--seeds-file", str(self.root / "seeds" / "E{k}-{date}.json"), "--review-guard"]
        with mock.patch.object(gen_loop, "train_generation") as tg, mock.patch.object(gen_loop, "run_promote_subprocess", return_value=0) as rp, \
                mock.patch.object(gen_loop, "COMMITTED_SEEDS", self.root / "seeds"):
            self.assertEqual(0, gen0.main(args))
        tg.assert_not_called()
        self.assertEqual(imit_dir, rp.call_args[0][1])  # judged, not refused for the closed line's two attempts
        recipe = meta["macro"]
        for i in (1, 2):  # the recipe's own two attempts spent
            gen_loop.append_promotion(self.tsv, {"cand_gen": 12, "seed_block": f"E12-x{i}", "cand_sha": PRIOR_SHA, "run_head_sha": f"head{i}",
                                                 "macro": recipe, "decision": "rejected", "attempt": i})
        with mock.patch.object(gen_loop, "train_generation") as tg, mock.patch.object(gen_loop, "run_promote_subprocess") as rp:
            self.assertEqual(0, gen0.main(args))
        rp.assert_not_called()
        self.assertTrue((markers / "done-11-g12-imit").exists())


class MarkerAndPlacementTests(unittest.TestCase):
    def test_the_tagged_marker_forms_parse_and_sort_after_the_attempts_of_their_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "train").mkdir()
            for n in ("ready-12", "ready-11-g12-imit-l05", "ready-11-g12-a2", "ready-11-g12-imit", "ready-11-g12", "ready-11-r1", "ready-11",
                      "done-11-g12-imit", "ready-11-g12-Imit", "ready-11-g12-imit-", "ready-11-g13"):
                (d / "train" / n).write_text("x")
            want = ["ready-11", "ready-11-r1", "ready-11-g12", "ready-11-g12-imit", "ready-11-g12-imit-l05", "ready-11-g12-a2", "ready-11-g13", "ready-12"]
            self.assertEqual(want, [p.name for _, p in gen_loop.ready_markers(d / "train")])
            self.assertEqual(want, [p.name for p in jobs.train_markers(d)][:len(want)])  # the runner takes the same order
        m = Path("ready-11-g12-imit-l05")
        self.assertEqual(((11, 0), 12, 1, "imit-l05"), (gen_loop.marker_parts(m), gen_loop.marker_target(m), gen_loop.marker_attempt(m), gen_loop.marker_tag(m)))
        self.assertEqual("", gen_loop.marker_tag(Path("ready-11-g12-a2")))
        self.assertEqual(2, gen_loop.marker_attempt(Path("ready-11-g12-a2")))
        self.assertIsNone(gen_loop.marker_parts(Path("done-11-g12-imit")))
        self.assertIsNone(gen_loop._MARKER.fullmatch("ready-11-g12-imit"))  # mark_ready's budget count does not see it

    def test_the_labs_runner_reads_the_tagged_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab = Path(tmp)
            (lab / "train").mkdir()
            (lab / "train" / "ready-11-g12-imit-l05").write_text("x")
            (lab / "train" / "ready-11-g12-imit").write_text("x")
            self.assertEqual(["ready-11-g12-imit", "ready-11-g12-imit-l05"], [p.name for p in jobs.train_markers(lab)])
            p = jobs.params_for("train", lab, {"cmd": ["x"]})
            self.assertEqual(("ready-11-g12-imit", "11", 12, "imit"), (p["marker"], p["generation"], p["candidate_generation"], p["tag"]))
            self.assertNotIn("attempt", p)
            self.assertEqual(lab / "train" / "done-11-g12-imit", jobs.consume_marker(lab / "train" / "ready-11-g12-imit"))
            p = jobs.params_for("train", lab, {"cmd": ["x"]})
            self.assertEqual(("ready-11-g12-imit-l05", "imit-l05"), (p["marker"], p["tag"]))
            (lab / "train" / "ready-11-g12-imit-l05").unlink()
            (lab / "train" / "ready-11-g12-a2").write_text("x")
            p = jobs.params_for("train", lab, {"cmd": ["x"]})
            self.assertEqual((2, False), (p["attempt"], "tag" in p))

    def test_candidate_dirname_and_built_candidate_take_a_tag(self):
        self.assertEqual("candidate-imit-l05", gen_loop.candidate_dirname(0, 1, "imit-l05"))
        self.assertEqual("candidate", gen_loop.candidate_dirname(0, 1, ""))
        self.assertEqual("candidate-a2", gen_loop.candidate_dirname(0, 2))
        for bad in ("Imit", "-x", "x-", "9x", "a b", "a", "a2", "b2", "ab"):
            with self.assertRaises(ValueError):
                gen_loop.candidate_dirname(0, 1, bad)
        with self.assertRaises(ValueError):
            gen_loop.candidate_dirname(0, 2, "imit")  # a tag is not an attempt


class BuildTests(unittest.TestCase):
    """`gen0.py build --human-prior --lambda --tag`: the candidate folder, its recipe, its marker."""

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

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, *extra: str) -> int:
        return gen0.main(["build", "--prior", str(self.prior), "--agent-root", str(self.agents), "--candidate-generation", "7", "--why", "test",
                          "--no-export", "--marker-dir", str(self.markers), *extra])

    def test_two_recipes_of_one_generation_sit_side_by_side_with_their_own_markers(self):
        self.assertEqual(0, self.build("--tag", "imit", "--human-prior", str(self.weights), "--lambda", "inf"))
        self.assertEqual(0, self.build("--tag", "imit-l05", "--human-prior", str(self.weights), "--lambda", "0.5"))
        sha = gen_loop.file_sha256(self.weights)
        a, b = (gen_loop.load_agent(self.agents / "gen7" / n) for n in ("candidate-imit", "candidate-imit-l05"))
        self.assertEqual(("vrun+pi_h:lambda=inf:pi_h=" + sha[:8], math.inf), (a.macro, a.lam))
        self.assertEqual(("vrun+pi_h:lambda=0.5:pi_h=" + sha[:8], 0.5), (b.macro, b.lam))
        self.assertEqual(["ready-5-g7-imit", "ready-5-g7-imit-l05"], sorted(p.name for p in self.markers.iterdir()))
        self.assertEqual(self.agents / "gen7" / "candidate-imit", gen_loop.built_candidate(self.agents, 7, 0, 1, "imit"))
        self.assertIsNone(gen_loop.built_candidate(self.agents, 7))  # the plain candidate folder is not there
        meta = json.loads((self.agents / "gen7" / "candidate-imit" / "agent.json").read_text())
        self.assertEqual(("human_prior.npz", sha, "inf"), (meta["human_prior"]["file"], meta["human_prior"]["sha256"], meta["human_prior"]["lambda"]))
        self.assertEqual("imit", meta["built"]["tag"])
        self.assertEqual(gen_loop.file_sha256(self.weights), gen_loop.file_sha256(self.agents / "gen7" / "candidate-imit" / "human_prior.npz"))
        # the fight player and the run head are the base's, untouched
        self.assertEqual(gen_loop.file_sha256(self.base / "run_value.npz"), gen_loop.run_head_sha(a))
        self.assertEqual(2, self.build("--tag", "imit", "--human-prior", str(self.weights), "--lambda", "inf"))  # the marker stands: not built again

    def test_the_prior_and_lambda_come_together_and_need_a_vrun_base(self):
        with self.assertRaises(ValueError):
            self.build("--human-prior", str(self.weights))
        with self.assertRaises(ValueError):
            self.build("--lambda", "0.5")
        with self.assertRaises(ValueError):
            gen0.parse_lambda("-1")
        (self.base / "agent.json").write_text(json.dumps({"generation": 5, "macro": "rules"}))
        with self.assertRaisesRegex(ValueError, "vrun macro"):
            self.build("--human-prior", str(self.weights), "--lambda", "0.5")

    def test_the_train_job_takes_the_tagged_marker_and_judges_candidate_imit_as_it_is(self):
        self.assertEqual(0, self.build("--tag", "imit", "--human-prior", str(self.weights), "--lambda", "inf"))
        (self.root / "seeds").mkdir()
        args = ["train", "--marker-dir", str(self.markers), "--shards-root", str(self.root), "--agent-root", str(self.agents), "--promote",
                "--emu-python", "python", "--seeds-file", str(self.root / "seeds" / "E{k}-{date}.json"), "--review-guard"]
        with mock.patch.object(gen_loop, "train_generation") as tg, mock.patch.object(gen_loop, "run_promote_subprocess", return_value=0) as rp, \
                mock.patch.object(gen_loop, "COMMITTED_SEEDS", self.root / "seeds"):
            self.assertEqual(0, gen0.main(args))
        tg.assert_not_called()
        self.assertEqual(self.agents / "gen7" / "candidate-imit", rp.call_args[0][1])
        self.assertTrue(rp.call_args[0][3].stem.startswith("E7-"))

    def test_a_tag_marker_without_a_built_candidate_trains_nothing(self):
        self.markers.mkdir()
        (self.markers / "ready-5-g7-imit").write_text("x")
        with mock.patch.object(gen_loop, "train_generation") as tg:
            rc = gen0.main(["train", "--marker-dir", str(self.markers), "--shards-root", str(self.root), "--agent-root", str(self.agents),
                            "--review-guard"])
        self.assertEqual(2, rc)
        tg.assert_not_called()
        self.assertTrue((self.markers / "ready-5-g7-imit").exists())


class RunFieldsTests(unittest.TestCase):
    def test_a_blended_run_row_says_which_component_decided_and_other_rows_are_as_they_were(self):
        import run_fields

        f = run_fields.RunFields()
        self.assertNotIn("decided", f.row())
        f.decide(3, {"screen": "card", "prior": 1, "v": 0, "guard": 0})
        f.decide(11, {"screen": "rest", "prior": 0, "v": 1, "guard": 1})
        row = f.row()
        self.assertEqual([[3, "card", 1, 0, 0], [11, "rest", 0, 1, 1]], row["decided"])
        self.assertEqual({"boss_damage", "boss_damage_acts", "cards", "nodes", "rests", "decided"}, set(row))
        self.assertEqual(set(run_fields.ROW_KEYS), set(row) - {"decided"})


if __name__ == "__main__":
    unittest.main()
