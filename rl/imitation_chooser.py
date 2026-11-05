"""The blended run chooser `vrun+pi_h` (milestone 4 of the 2026-11-05 strategy memo, section 3.3).

The `vrun` macro scores each option's afterstate with the run judgement V (expected final floors) and takes the best. The human prior
`pi_h` (`human_prior_infer`) is a softmax over a decision's options fitted to what humans took. The blended chooser takes

    argmax_o [ V(o) + lambda * log pi_h(o) ]

on the deck decisions the prior was fitted on: a card reward (the cards and the skip), an ancient's relic, a rest site's kind and the
smith's target. `lambda = inf` is the prior alone (ties by V), `lambda = 0` is `vrun`. Everything else is `vrun` unchanged: routes,
events, shops, potions, relic rewards, deck selections, and a deck decision that holds an option the prior has no row for (a card
reward with a reroll or sacrifice, a rest site's leave or hatch). Fights are not the macro's.

Guard (lambda > 0): an option the prior gives under `GUARD_PROB` is taken only when V prefers it to the blend's pick by more than
`GUARD_FLOORS` floors (the first attempt's "never rest" would be a prior-given zero the judgement could not argue with).

The rest site is two choices, as the fit has it: the kind (heal, smith, or the relic-given others) by the prior's marginal over the
kinds, then within smith the target by the prior's softmax over the smith options; within "others" V decides (the fit has no second
softmax there). Each stage is the same blend; a kind's V is its best option's.

The candidate's `macro` names the recipe (`macro_name`): `vrun+pi_h:lambda=<inf|number>:pi_h=<8 hex of the weights' sha256>`. A
different lambda or different weights is a different recipe, for the ledger and the attempt count (`gen_loop.candidate_recipe`).

`BlendedPolicy` is a `run_train.policy.LookaheadPolicy` (same values, same exploration arguments) that decides the deck screens by the
blend; `last` tells the driver which screen it decided and whether the prior's top option, V's top option, or the guard decided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from human_prior_infer import PriorScorer, log_softmax, option_rows
from human_prior_infer import load as load_prior
from macro_recipe import (  # noqa: F401 -- the recipe's names live there
    PREFIX,
    format_lambda,
    is_blended,
    macro_name,
    parse_macro,
)
from run_train import models
from run_train.policy import LookaheadPolicy
from run_train.selfplay import Pick

if TYPE_CHECKING:
    from run_train.features import Decision, Option, RunView

GUARD_PROB = 0.02  # an option the prior gives under this is taken only on the judgement's say-so
GUARD_FLOORS = 1.0  # ... and only when V prefers it by more than this many floors
TIE = 1e-9

CARD_KINDS = frozenset({"card.take", "card.skip"})
REST_KINDS = frozenset({"rest.heal", "rest.smith", "rest.dig", "rest.lift", "rest.cook", "rest.kindle", "rest.clone"})
SCREENS = ("card", "ancient", "rest")


# ------------------------------------------------------------------------------------------------------------------ the blend


@dataclass(frozen=True)
class Choice:
    """The blend's pick among a decision's options, with what each component would have taken."""

    index: int
    prior_top: int  # the option the prior alone would take (its highest probability; ties by V)
    v_top: int  # the option V alone would take
    guard: bool  # the guard overrode the blend: V's choice of a prior-given-near-zero option


def _argmax_by(primary, secondary, among):
    """The index in `among` with the highest `primary`, ties (within TIE) by the highest `secondary`, then the first."""
    top = max(primary[i] for i in among)
    tied = [i for i in among if primary[i] >= top - TIE]
    return max(tied, key=lambda i: (secondary[i], -i))


def _blend(values, logp, lam: float, guard: bool) -> tuple[int, bool]:
    """The index the blend takes and whether the guard decided it."""
    n = len(values)
    everyone = list(range(n))
    if lam == 0:
        return _argmax_by(values, [0.0] * n, everyone), False
    allowed = [i for i in everyone if math.exp(logp[i]) >= GUARD_PROB] if guard else everyone
    allowed = allowed or everyone
    if math.isinf(lam):
        best = _argmax_by(logp, values, allowed)
    else:
        score = [v + lam * x for v, x in zip(values, logp, strict=True)]
        best = _argmax_by(score, values, allowed)
    over = [i for i in everyone if i not in allowed and values[i] > values[best] + GUARD_FLOORS]
    if over:
        return _argmax_by(values, [0.0] * n, over), True
    return best, False


def blend_pick(values, logp, lam: float) -> Choice:
    """argmax_o [ values[o] + lam * logp[o] ] over one decision's options, with the guard. `lam = inf` takes the highest `logp`
    (ties by value); `lam = 0` takes the highest value and the guard is off (it is `vrun`)."""
    values = [float(v) for v in values]
    logp = [float(x) for x in logp]
    n = len(values)
    if n == 0 or n != len(logp):
        raise ValueError("a decision needs one value and one log-probability per option")
    index, fired = _blend(values, logp, lam, True)
    return Choice(index, _argmax_by(logp, values, range(n)), _argmax_by(values, [0.0] * n, range(n)), fired)


def _lse(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    top = float(x.max())
    return top + float(np.log(np.exp(x - top).sum()))


def _rest_stages(values, logits, groups, lam: float, guard: bool) -> tuple[int, bool]:
    n = len(values)
    z = _lse(logits)
    ids = sorted(set(groups))
    members = {g: [i for i in range(n) if groups[i] == g] for g in ids}
    lse_g = {g: _lse([logits[i] for i in members[g]]) for g in ids}
    kind, fired = _blend([max(values[i] for i in members[g]) for g in ids], [lse_g[g] - z for g in ids], lam, guard)
    g = ids[kind]
    inside = members[g]
    if g == 1 and len(inside) > 1:  # the smith's target: the prior's softmax over the smith options
        j, again = _blend([values[i] for i in inside], [logits[i] - lse_g[g] for i in inside], lam, guard)
        return inside[j], fired or again
    return max(inside, key=lambda i: (values[i], -i)), fired  # heal is one option; among the others V decides


def blend_rest(values, logits, groups, lam: float) -> Choice:
    """The rest site's two choices: the kind (a group of options: 0 heal, 1 smith, 2 the relic-given others) by the prior's marginal
    over the kinds, then within the smith group the target by the prior's softmax over its options. A group's value is its best
    option's. The heal is one option and among the others V decides (the fit has no softmax there)."""
    values = [float(v) for v in values]
    logits = [float(x) for x in logits]
    groups = list(groups)
    if not values or not (len(values) == len(logits) == len(groups)):
        raise ValueError("a rest site needs one value, logit and group per option")
    pick, fired = _rest_stages(values, logits, groups, lam, True)
    prior_top, _ = _rest_stages(values, logits, groups, math.inf, False)
    return Choice(pick, prior_top, _argmax_by(values, [0.0] * len(values), range(len(values))), fired)


# ------------------------------------------------------------------------------------------------------------------ the policy


def screen_of(decision: Decision) -> str | None:
    """Which deck screen the prior decides: "card", "ancient" or "rest"; None for every other screen, and for one that holds an
    option the prior has no row for or one the judgement does not score."""
    options = decision.options
    if len(options) < 2 or any(not o.scorable for o in options):
        return None
    kinds = {o.kind for o in options}
    if decision.type == "card_reward" and kinds <= CARD_KINDS:
        return "card"
    if decision.type == "neow" and kinds == {"neow.relic"}:
        return "ancient"
    if decision.type == "rest" and kinds <= REST_KINDS:
        return "rest"
    return None


def rest_group(option: Option) -> int:
    return 0 if option.kind == "rest.heal" else 1 if option.kind == "rest.smith" else 2


class BlendedPolicy(LookaheadPolicy):
    """`vrun` with the deck screens decided by `blend_pick` / `blend_rest`. With `lam = 0` it is `LookaheadPolicy`."""

    def __init__(self, checkpoint: str | Path, prior: str | Path | PriorScorer, lam: float, temperature: float = 0.0, eps: float = 0.0,
                 W: float | None = None):  # noqa: N803
        super().__init__(checkpoint, temperature, eps, W)
        if not (lam >= 0):
            raise ValueError(f"lambda is 0 or more, not {lam}")
        self.lam = float(lam)
        self.prior = prior if isinstance(prior, PriorScorer) else load_prior(prior)
        self.last: dict[str, Any] | None = None

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "kind": "lookahead+pi_h", "lambda": format_lambda(self.lam), "pi_h": self.prior.sha256[:8]}

    def pick(self, view: RunView, decision: Decision, rng: np.random.Generator | None) -> Pick:
        self.last = None
        screen = screen_of(decision) if self.lam > 0 else None
        if screen is None:
            return super().pick(view, decision, rng)
        view = decision.view or view
        options = list(decision.options)
        rows = option_rows(self.nets, view, options)
        probs = self.nets.value.floor_probs(rows)
        values = models.expected_floors(probs, self.W)
        logits = self.prior.row_logits(rows)
        if screen == "rest":
            choice = blend_rest(values, logits, [rest_group(o) for o in options], self.lam)
        else:
            choice = blend_pick(values, log_softmax(logits), self.lam)
        k, p = choice.index, 1.0
        if self.tau > 0 or self.eps > 0:  # exploring self-play: sample around the blend (the rest site on its flat score)
            pi = self.distribution(values, log_softmax(logits), choice)
            k = _draw(pi, rng)
            p = float(pi[k])
        self.last = {"screen": screen, "prior": int(k == choice.prior_top), "v": int(k == choice.v_top), "guard": int(choice.guard)}
        return Pick(k, values=tuple(float(v) for v in values), p_chosen=p, p_floor=tuple(float(x) for x in probs[k]))

    def distribution(self, values, logp=None, choice: Choice | None = None) -> np.ndarray:  # type: ignore[override]
        """The sampling distribution over a deck screen's options: (1 - eps) softmax(score / tau) + eps / n, where score is
        V + lam * log pi_h (log pi_h alone at lam = inf) and an option under the guard's probability has no mass unless the guard
        picked it. tau = 0 is the blend's own pick. Called with the values alone (no prior, a plain screen) it is `vrun`'s."""
        if logp is None or choice is None:
            return super().distribution(values)
        values = np.asarray(values, dtype=np.float64)
        logp = np.asarray(logp, dtype=np.float64)
        n = len(values)
        if self.tau > 0:
            score = logp if math.isinf(self.lam) else values + self.lam * logp
            allowed = np.exp(logp) >= GUARD_PROB
            if not allowed.any():
                allowed[:] = True
            if choice.guard:
                allowed = np.zeros(n, dtype=bool)
                allowed[choice.index] = True
            masked = np.where(allowed, score, -np.inf)
            greedy = models.softmax((masked - masked.max()) / self.tau)
        else:
            greedy = np.zeros(n)
            greedy[choice.index] = 1.0
        return (1.0 - self.eps) * greedy + self.eps / n


def _draw(pi: np.ndarray, rng: np.random.Generator | None) -> int:
    if len(pi) == 1 or pi.max() >= 1.0:
        return int(np.argmax(pi))
    if rng is None:
        raise ValueError("an exploring pick needs the run's exploration stream")
    return min(int(np.searchsorted(np.cumsum(pi), rng.random(), side="right")), len(pi) - 1)
