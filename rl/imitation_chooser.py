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

Lambda by act (2026-11-08, D55 addendum 12; the recipe's `:act<N>=<lambda>` parts, `macro_recipe`): a deck screen is decided with the
lambda of its floor's act, the base `lambda=` for an act the recipe does not name. `lambda=0.5:...:act1=0` is `vrun` through Act 1 (an
Act 1 screen is the plain `LookaheadPolicy`'s, exactly: no prior is read, the guard is off) and the 0.5 blend from the Act 2 ancient on,
the arm of the labels-question probe (`docs/design/2026-11-08-labels-vs-blocks.md` 3.2). Without parts the policy is what it was.

How the deck screen's option is taken (the recipe's `:pick=` part, `macro_recipe.PICKS`; 2026-11-06, docs/rl/STAKES_AND_HOLD.md):

* `argmax` (no part): the blend's argmax, as above. Measured on the first sanity run it skips Act 1 card rewards 5.4 times in 100 against
  the humans' 15.4, because a prior's top option is the card whenever the card is likelier than the skip.
* `sample`: the option is drawn from softmax((V + lambda log pi_h) / tau) at tau 1 over the options the guard leaves (the rest site on its
  flat joint score, as exploring self-play samples it), from a stream seeded by the run's seed, the floor and the screen's count on that
  floor (`begin_run`), so a block replays exactly and the draw never depends on what any other decision consumed.
* `skipcal<bias>`: the argmax of the blend with `bias` floors added to the skip option's score; the bias is the one
  `rl/skip_calibrate.py` found so that the agent's Act 1 skip rate lands near the humans' 15.4. A card screen's run row (`decided`)
  then carries its skip gap: the bias at which the skip would have been taken.

The run student (2026-11-11, `rl/run_student.py`, recipe `vrun+pi_s:...`, `macro_recipe`): a policy built with a student file and per-kind
lambdas decides the kinds the student has taken over (card, ancient, rest, route, shop, event, reward, select: `STUDENT_KINDS`) by
`V + lambda_act log pi_s` with that kind's lambda for the floor's act (`macro_recipe.student_lambda_for_floor`; 0 is `vrun` for that kind and
act), through the same blend, guard and sampling as the human prior's, over every option of the screen when the judgement scores them all.
The guard's probability is `min(GUARD_PROB, STUDENT_GUARD_SHARE / n)` for a screen of n options, so a shop of 29 offers does not rule out an
option for sitting under a fixed 2 in 100. Every kind the student has not taken over is decided as before (the human prior's blend on the deck
screens, `vrun` elsewhere): a policy without a student is the policy it was, number for number.

The boss chain (2026-11-19, the Act 3 week, `rl/boss_chain.py`; the recipe's `+chain:<8 hex>@<k1>/<k2>/<k3>` part, agent.json `boss_chain`): on a screen in Act 2 or Act 3 up to
floor 47 whose options all score, each option's value gains `kappa_act * log(chance of getting through)` from the chain's two forecasts (past the Act 2 boss; past the
first Act 3 boss given that): `log q2 + log q3` up to the Act 2 boss's floor, `log q3` after it. The blend, the guard, the draw and the skip gap read that value; the
recorded values and floor distributions stay the run brain's. An Act 1 screen (kappa 0 there) and a policy without a chain are what they were, choice for choice.

The Act 1 elite bonus (2026-11-14, D60 item 9, the elite test; the recipe's `+elite:<bonus>@<per cent>` part, agent.json `route_elite`): on an Act 1 route
screen, while HP is at or above the per cent of max HP, each option's score gains `bonus` floors for each elite room on the best path through the room it leads
to (`elites_ahead`, the room itself counted), whoever decides the screen (the student's blend, or V alone when the student's lambda is 0 or there is no student).
The recorded values stay the judgement's; the guard is unchanged. Without the option a policy is what it was, choice for choice.

`BlendedPolicy` is a `run_train.policy.LookaheadPolicy` (same values, same exploration arguments) that decides the deck screens by the
blend; `last` tells the driver which screen it decided and whether the prior's top option, V's top option, or the guard decided.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from human_prior_infer import PriorScorer, StudentScorer, log_softmax, option_gaps, option_grids, option_rows, option_sides
from human_prior_infer import load as load_prior
from human_prior_infer import load_student
from macro_recipe import (  # noqa: F401 -- the recipe's names live there
    PICK_ARGMAX,
    PICK_SAMPLE,
    PICK_SKIPCAL,
    PICKS,
    PREFIX,
    STUDENT_KINDS,
    act_of_floor,
    chain_tag,
    elite_bonus,
    elite_pct,
    format_lambda,
    is_blended,
    macro_name,
    lambda_for_floor,
    parse_macro,
    parse_pick,
    student_lambda_for_floor,
)
from macro_recipe import chain_kappa as validated_chain_kappa
from run_train import models, vocab
from run_train.policy import LookaheadPolicy, floor_probs
from run_train.selfplay import Pick

if TYPE_CHECKING:
    from run_train.features import Decision, Option, RunView

GUARD_PROB = 0.02  # an option the prior gives under this is taken only on the judgement's say-so
GUARD_FLOORS = 1.0  # ... and only when V prefers it by more than this many floors
TIE = 1e-9
LAST_CHAIN_FLOOR = 47  # the boss chain adds to screens up to this floor (`boss_chain.LAST_PRE_B3A`: a test holds them equal): the first Act 3 boss stands on floor 48

CARD_KINDS = frozenset({"card.take", "card.skip"})
REST_KINDS = frozenset({"rest.heal", "rest.smith", "rest.dig", "rest.lift", "rest.cook", "rest.kindle", "rest.clone"})
SCREENS = ("card", "ancient", "rest")
STUDENT_GUARD_SHARE = 0.25  # the student's guard: an option under min(GUARD_PROB, this / n options) of the student's probability needs V's say-so
# a decision's type (`run_train.features.Decision.type`) -> the kind the run teacher and the student call it
KIND_OF_TYPE = {"map": "route", "card_reward": "card", "rest": "rest", "shop": "shop", "neow": "ancient", "event": "event", "rewards": "reward",
                "card_select": "select"}


# ------------------------------------------------------------------------------------------------------------------ the elite bonus


@lru_cache(maxsize=4096)
def _elite_table(nodes) -> dict[tuple[int, int], int]:
    """{(col, row): the most elite rooms on any path from the room to the boss, the room itself counted} for an act map (a tuple of `features.MapNode`)."""
    best: dict[tuple[int, int], int] = {}
    for n in sorted(nodes, key=lambda x: -x.row):
        ahead = max((best[c] for c in n.children if c in best), default=0)
        best[(n.col, n.row)] = ahead + int(n.node_type == vocab.NODE_ELITE)
    return best


def elites_ahead(nodes, coord) -> int:
    """The most elite rooms on any path from the room at `coord` (col, row) to the boss, the room itself counted; 0 for a coordinate not on the map."""
    return _elite_table(tuple(nodes)).get((int(coord[0]), int(coord[1])), 0)


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


def guard_prob_for(n: int, student: bool = False) -> float:
    """The guard's probability for a screen of `n` options: `GUARD_PROB` for the human prior's, `min(GUARD_PROB, STUDENT_GUARD_SHARE / n)` for the
    student's."""
    return min(GUARD_PROB, STUDENT_GUARD_SHARE / max(1, n)) if student else GUARD_PROB


def _blend(values, logp, lam: float, guard: bool, bonus=None, guard_prob: float = GUARD_PROB) -> tuple[int, bool]:
    """The index the blend takes and whether the guard decided it. `bonus` (one number per option, or None) is added to each option's
    score (the prior's log-probability at lambda = inf): the skip bias of the `skipcal` pick mode."""
    n = len(values)
    everyone = list(range(n))
    if lam == 0:
        return _argmax_by(values, [0.0] * n, everyone), False
    allowed = [i for i in everyone if math.exp(logp[i]) >= guard_prob] if guard else everyone
    allowed = allowed or everyone
    extra = [0.0] * n if bonus is None else [float(b) for b in bonus]
    if math.isinf(lam):
        best = _argmax_by([x + b for x, b in zip(logp, extra, strict=True)], values, allowed)
    else:
        score = [v + lam * x + b for v, x, b in zip(values, logp, extra, strict=True)]
        best = _argmax_by(score, values, allowed)
    over = [i for i in everyone if i not in allowed and values[i] > values[best] + GUARD_FLOORS]
    if over:
        return _argmax_by(values, [0.0] * n, over), True
    return best, False


def blend_pick(values, logp, lam: float, bonus=None, guard_prob: float = GUARD_PROB) -> Choice:
    """argmax_o [ values[o] + lam * logp[o] + bonus[o] ] over one decision's options, with the guard. `lam = inf` takes the highest
    `logp` (plus bonus; ties by value); `lam = 0` takes the highest value and the guard is off (it is `vrun`). `bonus` is None except in
    the `skipcal` pick mode, where it is the skip bias on the skip option."""
    values = [float(v) for v in values]
    logp = [float(x) for x in logp]
    n = len(values)
    if n == 0 or n != len(logp):
        raise ValueError("a decision needs one value and one log-probability per option")
    if bonus is not None and len(bonus) != n:
        raise ValueError("a bonus needs one number per option")
    index, fired = _blend(values, logp, lam, True, bonus, guard_prob)
    return Choice(index, _argmax_by(logp, values, range(n)), _argmax_by(values, [0.0] * n, range(n)), fired)


def skip_gap(values, logp, lam: float, skip: int) -> float:
    """The skip bias at which the blend would take the skip option `skip` over the best other option: the best other score minus the
    skip's (V + lam log pi_h, or log pi_h at lam = inf), the guard left out. Negative: the blend already takes the skip. The skip is
    taken at bias b exactly when b >= gap (ties go to the skip, which the argmax's tie rule decides the other way by 1e-9: not a case
    that matters). Used by `rl/skip_calibrate.py` to read what skip rate any bias would give on the screens an agent has met."""
    score = [float(x) if math.isinf(lam) else float(v) + lam * float(x) for v, x in zip(values, logp, strict=True)]
    other = max(s for i, s in enumerate(score) if i != skip)
    return other - score[skip]


def _lse(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    top = float(x.max())
    return top + float(np.log(np.exp(x - top).sum()))


def _rest_stages(values, logits, groups, lam: float, guard: bool, guard_prob: float = GUARD_PROB) -> tuple[int, bool]:
    n = len(values)
    z = _lse(logits)
    ids = sorted(set(groups))
    members = {g: [i for i in range(n) if groups[i] == g] for g in ids}
    lse_g = {g: _lse([logits[i] for i in members[g]]) for g in ids}
    kind, fired = _blend([max(values[i] for i in members[g]) for g in ids], [lse_g[g] - z for g in ids], lam, guard, None, guard_prob)
    g = ids[kind]
    inside = members[g]
    if g == 1 and len(inside) > 1:  # the smith's target: the prior's softmax over the smith options
        j, again = _blend([values[i] for i in inside], [logits[i] - lse_g[g] for i in inside], lam, guard, None, guard_prob)
        return inside[j], fired or again
    return max(inside, key=lambda i: (values[i], -i)), fired  # heal is one option; among the others V decides


def blend_rest(values, logits, groups, lam: float, guard_prob: float = GUARD_PROB) -> Choice:
    """The rest site's two choices: the kind (a group of options: 0 heal, 1 smith, 2 the relic-given others) by the prior's marginal
    over the kinds, then within the smith group the target by the prior's softmax over its options. A group's value is its best
    option's. The heal is one option and among the others V decides (the fit has no softmax there)."""
    values = [float(v) for v in values]
    logits = [float(x) for x in logits]
    groups = list(groups)
    if not values or not (len(values) == len(logits) == len(groups)):
        raise ValueError("a rest site needs one value, logit and group per option")
    pick, fired = _rest_stages(values, logits, groups, lam, True, guard_prob)
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


def student_kind_of(decision: Decision) -> str | None:
    """The kind the student would decide this screen as, or None: the decision's type must be one of `KIND_OF_TYPE`, it needs two or more
    options and the judgement must score them all (the human prior's deck screens also hold to `screen_of`'s option sets)."""
    kind = KIND_OF_TYPE.get(decision.type)
    options = decision.options
    if kind is None or len(options) < 2 or any(not o.scorable for o in options):
        return None
    if kind in SCREENS:
        return kind if screen_of(decision) == kind else None
    return kind


def rest_group(option: Option) -> int:
    return 0 if option.kind == "rest.heal" else 1 if option.kind == "rest.smith" else 2


class BlendedPolicy(LookaheadPolicy):
    """`vrun` with the deck screens decided by `blend_pick` / `blend_rest`. With `lam = 0` it is `LookaheadPolicy`."""

    def __init__(self, checkpoint: str | Path, prior: str | Path | PriorScorer, lam: float, temperature: float = 0.0, eps: float = 0.0,
                 W: float | None = None, pick: str = PICK_ARGMAX, skip_bias: float = 0.0,  # noqa: N803
                 act_lam: Mapping[int, float] | None = None, student: str | Path | StudentScorer | None = None,
                 student_lams: Mapping[str, Sequence[float]] | None = None, route_elite: tuple[float, int] | None = None,
                 chain: str | Path | Any | None = None, chain_kappa: Sequence[float] | None = None):
        super().__init__(checkpoint, temperature, eps, W)
        if not (lam >= 0):
            raise ValueError(f"lambda is 0 or more, not {lam}")
        self.act_lam = {int(a): float(v) for a, v in (act_lam or {}).items()}  # {act number from 1: lambda}, the recipe's act<N> parts
        for a, v in self.act_lam.items():
            if a not in (1, 2, 3) or not v >= 0:
                raise ValueError(f"a per-act lambda is {{act 1 to 3: 0 or more}}, not {{{a}: {v}}}")
        if pick not in PICKS:
            raise ValueError(f"unknown pick mode {pick!r}; one of {', '.join(PICKS)}")
        if pick != PICK_SKIPCAL and skip_bias:
            raise ValueError("a skip bias belongs to the skipcal pick mode")
        self.route_elite: tuple[float, int] | None = None  # (floors per elite room, HP per cent at or above which it applies): Act 1 route screens only
        if route_elite is not None:
            if len(route_elite) != 2:
                raise ValueError(f"the elite bonus is (floors per elite room, HP per cent), not {route_elite!r}")
            self.route_elite = (elite_bonus(route_elite[0]), elite_pct(route_elite[1]))
        self.chain: Any = None  # a boss_chain.ChainScorer (the chain's file) and its kappa per act (Acts 2 and 3 only act): None is every policy before it
        self.chain_kappa: tuple[float, float, float] | None = None
        if (chain is None) != (chain_kappa is None):
            raise ValueError("a boss chain needs its kappa, and a kappa needs a boss chain")
        if chain is not None:
            import boss_chain  # noqa: PLC0415 -- numpy only; a policy without a chain never loads it

            self.chain_kappa = validated_chain_kappa(chain_kappa)
            self.chain = chain if isinstance(chain, boss_chain.ChainScorer) else boss_chain.ChainScorer(chain)
        self.lam = float(lam)
        self.pick_mode = pick
        self.skip_bias = float(skip_bias)
        self.prior = prior if isinstance(prior, PriorScorer) else load_prior(prior)
        if (student is None) != (not student_lams):
            raise ValueError("a student needs its per-kind lambdas, and lambdas need a student")
        self.student: StudentScorer | None = None
        self.student_lams: dict[str, tuple[float, ...]] = {}
        if student is not None:
            self.student = student if isinstance(student, StudentScorer) else load_student(student)
            for kind, lams in (student_lams or {}).items():
                lams = tuple(float(x) for x in lams)
                if kind not in STUDENT_KINDS or len(lams) != 3 or not all(x >= 0 for x in lams):
                    raise ValueError(f"a student's lambda is three numbers (Acts 1 to 3), 0 or more, for one of {', '.join(STUDENT_KINDS)}: {kind!r} {lams!r}")
                if not self.student.has(kind):
                    raise ValueError(f"the student was not fitted for {kind} (its file names {', '.join(self.student.kinds)})")
                self.student_lams[kind] = lams
        self.last: dict[str, Any] | None = None
        self._run_key: bytes | None = None  # the run's seed hashed: what a sampled pick's stream is built from (`begin_run`)
        self._screens: dict[tuple[int, str], int] = {}  # (floor, screen) -> screens of that kind decided on that floor so far

    def describe(self) -> dict[str, Any]:
        extra = {} if self.pick_mode == PICK_ARGMAX else {"pick": self.pick_mode, **({"skip_bias": self.skip_bias} if self.skip_bias else {})}
        if self.act_lam:
            extra["lambda_acts"] = {str(a): format_lambda(v) for a, v in sorted(self.act_lam.items())}
        if self.student is not None:
            extra["pi_s"] = self.student.sha256[:8]
            extra["student_kinds"] = {k: [format_lambda(x) for x in v] for k, v in self.student_lams.items()}
        if self.route_elite is not None:
            extra["route_elite"] = [self.route_elite[0], self.route_elite[1]]
        if self.chain is not None:
            extra["boss_chain"] = chain_tag(self.chain.sha256, self.chain_kappa)
        return {**super().describe(), "kind": "lookahead+pi_h", "lambda": format_lambda(self.lam), "pi_h": self.prior.sha256[:8], **extra}

    def lam_at(self, floor: int) -> float:
        """The lambda a deck screen on `floor` is decided with: its act's (`act_lam`), else the base `lam`."""
        return lambda_for_floor(self.lam, self.act_lam, floor)

    def begin_run(self, seed: str | int) -> None:
        """Name the run this policy now plays (its game seed): a `sample` pick's draws are seeded by it, the floor and the screen's
        count on the floor, never by the order of anything else, so the same seed always draws the same options and the block replays."""
        self._run_key = hashlib.sha256(f"spireline-imitation-sample|{seed}".encode()).digest()
        self._screens = {}

    def _sample_stream(self, floor: int, screen: str) -> np.random.Generator:
        if self._run_key is None:
            raise ValueError("a sample pick needs begin_run(seed): the draw is seeded by the run's seed and the floor")
        nth = self._screens.get((floor, screen), 0)
        self._screens[(floor, screen)] = nth + 1
        digest = hashlib.sha256(self._run_key + f"|{floor}|{screen}|{nth}".encode()).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def student_at(self, view: RunView, decision: Decision) -> tuple[str, float] | None:
        """(kind, lambda) when the student decides this screen: it has taken the kind over and the judgement scores every option; the
        lambda is the kind's for the floor's act. None when the human prior's path or `vrun` decides it."""
        if self.student is None:
            return None
        kind = student_kind_of(decision)
        if kind is None or kind not in self.student_lams:
            return None
        return kind, student_lambda_for_floor(self.student_lams[kind], (decision.view or view).floor)

    def route_bonus(self, view: RunView, decision: Decision) -> list[float] | None:
        """The elite bonus of each option of an Act 1 route screen (the bonus times `elites_ahead` of the room the option leads to), or None where it does not
        apply: no `route_elite`, not a route screen the policy decides (two or more options, all scored), not Act 1, or HP under the threshold."""
        if self.route_elite is None or decision.type != "map" or student_kind_of(decision) != "route":
            return None
        v = decision.view or view
        bonus, pct = self.route_elite
        if act_of_floor(v.floor) != 0 or v.max_hp <= 0 or v.hp * 100 < pct * v.max_hp:
            return None
        return [bonus * elites_ahead(v.map_nodes, o.node[:2]) if o.node is not None else 0.0 for o in decision.options]

    def chain_applies(self, view: RunView, decision: Decision) -> bool:
        """Whether the boss chain adds to this screen: a chain is set, the act's kappa is above 0 (Acts 2 and 3 only), the floor is 47 or lower and the screen has
        two or more options that all score."""
        if self.chain is None:
            return False
        v = decision.view or view
        act, options = int(v.act_index), decision.options
        if act not in (1, 2) or self.chain_kappa[act] <= 0 or int(v.floor) > LAST_CHAIN_FLOOR or len(options) < 2:  # type: ignore[index]
            return False
        return all(o.scorable for o in options)

    def chain_extra(self, view: RunView, decision: Decision, rows: np.ndarray | None = None) -> list[float] | None:
        """The boss chain's addition to each option's value (`boss_chain.chain_terms` of the chain's log-chances of the options' rows), or None where the chain
        adds nothing (`chain_applies`). `rows` are the options' judgement rows when the caller holds them."""
        if not self.chain_applies(view, decision):
            return None
        import boss_chain  # noqa: PLC0415

        v = decision.view or view
        rows = option_rows(self.nets, v, list(decision.options)) if rows is None else rows
        terms = boss_chain.chain_terms(self.chain.log_q(rows), int(v.act_index), int(v.floor), self.chain_kappa)
        return None if terms is None else [float(x) for x in terms]

    @staticmethod
    def _joined(*extras: list[float] | None) -> list[float] | None:
        """The sum of the additions that apply (None when none does)."""
        live = [e for e in extras if e is not None]
        if not live:
            return None
        return [float(sum(x)) for x in zip(*live, strict=True)]

    def _vrun_with_bonus(self, view: RunView, decision: Decision, rng: np.random.Generator | None, extra: list[float], rows: np.ndarray | None = None) -> Pick:
        """`LookaheadPolicy.pick` on a screen whose options all score, with `extra` (one number per option) added to each option's value in the choice; the
        recorded values and floor distributions are the judgement's. `rows` are the options' judgement rows (`option_rows`) when the caller holds them: the same
        numbers `floor_probs` scores."""
        view = decision.view or view
        options = list(decision.options)
        probs = floor_probs(self.nets, view, options) if rows is None else self.nets.value.floor_probs(rows)  # as `LookaheadPolicy.pick` scores them
        values = models.expected_floors(probs, self.W)
        pi = self.distribution(np.asarray(values, dtype=np.float64) + np.asarray(extra, dtype=np.float64))
        k = _draw(pi, rng)
        return Pick(k, values=tuple(float(v) for v in values), p_chosen=float(pi[k]), p_floor=tuple(float(x) for x in probs[k]))

    def pick(self, view: RunView, decision: Decision, rng: np.random.Generator | None) -> Pick:
        self.last = None
        extra = self.route_bonus(view, decision)  # None unless the elite bonus applies to this screen
        chain: list[float] | None = None
        rows: np.ndarray | None = None
        if self.chain_applies(view, decision):  # the chain's addition (Acts 2 and 3); the rows are scored once for the chain and the choice
            v = decision.view or view
            rows = option_rows(self.nets, v, list(decision.options))
            chain = self.chain_extra(view, decision, rows)
        taken = self.student_at(view, decision)
        if taken is not None:
            kind, lam_s = taken
            if lam_s <= 0:  # the student's lambda is 0 here: the judgement alone, as `vrun` decides it
                if extra is not None or chain is not None:
                    return self._vrun_with_bonus(view, decision, rng, self._joined(extra, chain), rows)
                return LookaheadPolicy.pick(self, view, decision, rng)
            return self._decide(view, decision, rng, kind, lam_s, self.student.scorer_for(kind), student=True, extra=extra, chain=chain, rows=rows)  # a merged student's net for this kind
        lam = self.lam_at(view.floor)
        screen = screen_of(decision) if lam > 0 else None
        if screen is None:
            if extra is not None or chain is not None:
                return self._vrun_with_bonus(view, decision, rng, self._joined(extra, chain), rows)
            return super().pick(view, decision, rng)
        return self._decide(view, decision, rng, screen, lam, self.prior, chain=chain, rows=rows)

    def _decide(self, view: RunView, decision: Decision, rng: np.random.Generator | None, screen: str, lam: float, scorer: PriorScorer,
                student: bool = False, extra: list[float] | None = None, chain: list[float] | None = None, rows: np.ndarray | None = None) -> Pick:
        """One screen decided by the blend of V and `scorer`'s log-probabilities (the human prior's, or the student's for a kind it took over). `chain` (one number
        per option, or None) is the boss chain's addition to each option's value: the blend, the guard, the draw and the skip gap read V + chain, the record
        keeps V. `rows` are the options' judgement rows when `pick` scored them already."""
        view = decision.view or view
        options = list(decision.options)
        tower = bool(getattr(scorer, "map_grid", False))  # a scorer with the map tower reads each option's block M (the rows are the same bytes)
        gapped = bool(getattr(scorer, "state_gaps", False))  # one with the state gaps reads each option's block G (the view's counters, "?" odds and removals)
        if rows is not None:
            grid = option_grids(view, options) if tower else None
        elif tower:
            rows, grid = option_rows(self.nets, view, options, with_grid=True)
        else:
            rows, grid = option_rows(self.nets, view, options), None
        probs = self.nets.value.floor_probs(rows)
        values = models.expected_floors(probs, self.W)
        u = values if chain is None else np.asarray(values, dtype=np.float64) + np.asarray(chain, dtype=np.float64)  # what the choice reads: V plus the chain
        sided = tuple(getattr(scorer, "side_blocks", ()) or ())  # a student fitted with the loader's side blocks reads them too (the held ones from the rows)
        kw = {"sides": option_sides(view, options, sided, rows)} if sided else {}
        if gapped:
            logits = scorer.row_logits(rows, grid, option_gaps(view, options), **kw)
        else:
            logits = scorer.row_logits(rows, grid, **kw) if (tower or sided) else scorer.row_logits(rows)
        logp = log_softmax(logits)
        gp = guard_prob_for(len(options), student)
        skip = next((i for i, o in enumerate(options) if o.kind == "card.skip"), None) if screen == "card" else None
        bonus = None if extra is None else [float(x) for x in extra]  # the elite bonus of a route screen (`route_bonus`): added to each score, never to the record
        if self.pick_mode == PICK_SKIPCAL and skip is not None and self.skip_bias:
            bonus = [self.skip_bias if i == skip else 0.0 for i in range(len(options))]
        if screen == "rest":
            choice = blend_rest(u, logits, [rest_group(o) for o in options], lam, gp)
        else:
            choice = blend_pick(u, logp, lam, bonus, gp)
        k, p = choice.index, 1.0
        exploring = self.tau > 0 or self.eps > 0
        if exploring or self.pick_mode == PICK_SAMPLE:  # sample around the blend (the rest site on its flat score)
            pi = self.distribution(u, logp, choice, tau=None if exploring else 1.0, bonus=bonus, lam=lam, guard_prob=gp)
            k = _draw(pi, rng if exploring else self._sample_stream(int(view.floor), ("s:" + screen) if student else screen))
            p = float(pi[k])
        self.last = {"screen": ("s:" + screen) if student else screen, "prior": int(k == choice.prior_top), "v": int(k == choice.v_top),
                     "guard": int(choice.guard)}
        if self.pick_mode == PICK_SKIPCAL and skip is not None:  # the skip's gap, for the calibration (run_fields.decide appends it)
            self.last["gap"] = round(skip_gap(u, logp, lam, skip), 4)
        return Pick(k, values=tuple(float(v) for v in values), p_chosen=p, p_floor=tuple(float(x) for x in probs[k]))

    def distribution(self, values, logp=None, choice: Choice | None = None, tau: float | None = None,
                     bonus=None, lam: float | None = None, guard_prob: float = GUARD_PROB) -> np.ndarray:  # type: ignore[override]
        """The sampling distribution over a deck screen's options: (1 - eps) softmax(score / tau) + eps / n, where score is
        V + lam * log pi_h (log pi_h alone at lam = inf) and an option under the guard's probability has no mass unless the guard
        picked it. tau = 0 is the blend's own pick. `tau` None is the policy's own (the exploration temperature); the `sample` pick
        mode passes 1. `lam` is the screen's lambda (the policy's own for None; `pick` passes its act's). Called with the values alone
        (no prior, a plain screen) it is `vrun`'s."""
        if logp is None or choice is None:
            return super().distribution(values)
        values = np.asarray(values, dtype=np.float64)
        logp = np.asarray(logp, dtype=np.float64)
        n = len(values)
        tau = self.tau if tau is None else tau
        lam = self.lam if lam is None else lam
        if tau > 0:
            score = logp if math.isinf(lam) else values + lam * logp
            if bonus is not None:
                score = score + np.asarray(bonus, dtype=np.float64)
            allowed = np.exp(logp) >= guard_prob
            if not allowed.any():
                allowed[:] = True
            if choice.guard:
                allowed = np.zeros(n, dtype=bool)
                allowed[choice.index] = True
            masked = np.where(allowed, score, -np.inf)
            greedy = models.softmax((masked - masked.max()) / tau)
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
