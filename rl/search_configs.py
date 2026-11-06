"""Named combat-search settings for the PLAY agent (PLAN.md round A3), selectable from rl/gen0.py.

A config fixes how `live_search.search_live` is called at a combat decision: the simulation budget, the root
sweep (`min_child_sims`: every root move gets this many common-random-number playouts before UCT; this is the
search's breadth knob, the counterpart of a Gumbel root's sampled-action count m, which the live search does not
have), the UCT exploration constant `c`, and the confirmation rounds. Every search is resampled (the firewall).

A config may carry a budget schedule: simulations per decision from what the player can see at it (decision kind,
number of distinct legal moves, HP fraction, elite or boss fight). `Position` holds those features; `sims_for`
is pure so it can be tested without native code. `position_of` reads them from a live fight.

A config may also carry the two options of `docs/rl/STAKES_AND_HOLD.md` (2026-11-06), both off in every config that existed before:

* `stakes`: a multiplier of the simulations per fight kind, (hallway, elite, boss), applied after the budget schedule and to its clamp
  and to the safety clock (`time_budget_s`): the hallway x1, elite x4, boss x16 schedule is `a3-v1@h1e4b16`, which at a boss fight is
  exactly what the play-or-deck measurement's `x16` arm played (`fight_gym.arm_settings`) and in a hallway fight is a3-v1;
* `potion_hold`: in a hallway fight the root's potion moves are dropped unless the kill check fails and the player would fall under
  that fraction of max HP after the enemies' intended damage (`live_search.search_live(potion_hold=...)`).

A config may price a fight in the run's terms (`run_currency`, training design section 3 and brief 3a; `rl/run_currency.py`): a won fight is
worth what the frozen run head says of the post-fight state (HP, max HP and the potions still held) instead of its end HP plus 10 a potion.
Its name is the base's with `-r` (`a3-v1-r`, `judge-v0-r`, `a3-v1-k-r`; `with_currency`), and the stakes and hold options go after it
(`a3-v1-r@h1e4b16`). The search needs the fight's currency from its caller (`search_kwargs(pos, terminal=...)`: the caller knows the run's
view or the start record); a currency config asked for kwargs without one raises rather than quietly play in HP.

A config with either has a name `<base>[@h<H>e<E>b<B>][+hold<N>]` (`macro_recipe.config_name`) that `get` reads, so every consumer of a
config name (the lab's self-play, the promotion arms, the lockstep's real-game player) can play it; an agent's own schedule comes from
its agent.json (`gen_loop.Agent.search_config`, `for_agent` here).

    python rl/gen0.py emu --config a3-v1 ...
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
from dataclasses import dataclass, field

import macro_recipe

KINDS = ("play", "target", "select")
NODE_ELITE, NODE_BOSS = 2, 6  # run_train.vocab, copied so this module imports no native code


@dataclass(frozen=True)
class Position:
    """What a combat decision looks like to the player, for the budget schedule."""

    kind: str  # "select" (a choice screen in the fight), "target" (some move has 2+ targets), else "play"
    n_moves: int  # distinct legal root moves (a card with k targets counts k)
    hp_frac: float  # player HP / max HP when the decision is asked
    hard: bool  # an elite or boss fight
    act: int = 0
    fight: str = ""  # "hall", "elite" or "boss" (`position_of` sets it); "": not said, `fight_kind` reads `hard`

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown decision kind {self.kind!r}")
        if self.fight and self.fight not in macro_recipe.FIGHT_KINDS:
            raise ValueError(f"unknown fight kind {self.fight!r}")

    @property
    def fight_kind(self) -> str:
        """The fight's kind: what `position_of` read, else "elite" for a hard fight (the kind is not known) and "hall" for the rest."""
        return self.fight or ("elite" if self.hard else "hall")


@dataclass(frozen=True)
class Schedule:
    """sims = base * kind multiplier, raised to `high` at a high-stakes decision, clamped to [lo, hi].

    High stakes: an elite or boss fight, HP below `low_hp`, or at least `many_moves` distinct moves.
    """

    high: int
    low_hp: float = 0.5
    many_moves: int = 99
    kind_mult: dict[str, float] = field(default_factory=dict)
    lo: int = 1
    hi: int = 2000

    def __hash__(self) -> int:  # the dict field is fixed at definition; hash by its items
        return hash((self.high, self.low_hp, self.many_moves, tuple(sorted(self.kind_mult.items())), self.lo, self.hi))


@dataclass(frozen=True)
class SearchConfig:
    name: str
    sims: int  # base simulations per decision (sweep + UCT; the confirmation comes on top)
    min_child_sims: int = 2
    c: float = 0.1
    confirm_pairs: int = 0
    time_budget_s: float = 120.0  # a safety cap; the budget is the simulation count
    schedule: Schedule | None = None
    leaf: str = "rollout"  # how a simulation's leaf is valued: a rollout to the fight's end, or the fight judge ("judge")
    # Before the search, look for a kill with this turn's cards (rl/lethal.py) and play its first move if one is verified; with no
    # kill found the search runs as it always did. Off in every config that existed before the option.
    lethal_check: bool = False
    # A multiplier of the simulations (and of the safety clock) per fight kind, (hall, elite, boss); None: the config's own budget.
    stakes: tuple[int, int, int] | None = None
    # The hallway potion hold's threshold, a fraction of max HP; None: no hold (see the module docstring).
    potion_hold: float | None = None
    # Price a won fight in the run head's terms (rl/run_currency.py) instead of HP and 10 a potion; the caller passes the fight's currency.
    run_currency: bool = False

    @property
    def needs_position(self) -> bool:
        """Whether a decision's `Position` changes what the search does (a schedule, a stakes multiplier or a hold): the callers read
        the fight only for such a config."""
        return bool(self.schedule) or self.stakes is not None or self.potion_hold is not None

    def stakes_mult(self, pos: Position | None) -> int:
        """The multiplier of the fight `pos` is in: 1 without a stakes schedule or a position."""
        if self.stakes is None or pos is None:
            return 1
        return self.stakes[macro_recipe.FIGHT_KINDS.index(pos.fight_kind)]

    def wall_factor(self) -> float:
        """How much longer than the base config's a run of this config may take: the largest multiplier over four (a boss at x16
        takes about 16 times as long as a3-v1's, but a boss is one fight of a run), never under 1. The dispatchers' per-run wall
        limits are multiplied by it."""
        return max(1.0, max(self.stakes) / 4) if self.stakes is not None else 1.0

    def high_stakes(self, pos: Position) -> bool:
        s = self.schedule
        return bool(s) and (pos.hard or pos.hp_frac < s.low_hp or pos.n_moves >= s.many_moves)

    def sims_for(self, pos: Position | None) -> int:
        """Simulations for this decision. Without a schedule (or a position) the flat `sims`. A stakes multiplier scales the base
        count before the kind multiplier and the clamp's top, so a boss fight at x16 is a3-v1's 30 simulations times 16 and its
        choice screens 120 times 16, as the `x16` arm of `fight_gym.arm_settings` plays them."""
        s = self.schedule
        m = self.stakes_mult(pos)
        if s is None or pos is None:
            return self.sims * m
        n = (s.high if self.high_stakes(pos) else self.sims) * m
        n = round(n * s.kind_mult.get(pos.kind, 1.0))
        return max(s.lo, min(s.hi * m, n))

    def search_kwargs(self, pos: Position | None = None, env_prior: bool = True, terminal=None) -> dict:
        """Keyword arguments for `live_search.search_live` (add `seed`). With `env_prior`, SPIRELINE_GEN_DIR names a
        generation whose combat prior is added (score-real's lockstep player builds its search from a config alone);
        the emulator driver passes False, because there the agent argument alone decides the prior.

        `terminal` is the fight's run currency (`run_currency.FightCurrency`); a `run_currency` config passes it on as `terminal` and
        refuses to go on without one, and any other config ignores it."""
        kw = {"max_sims": self.sims_for(pos), "min_child_sims": self.min_child_sims, "c": self.c,
              "confirm_pairs": self.confirm_pairs, "time_budget_s": self.time_budget_s * self.stakes_mult(pos), "resample": True}
        if self.run_currency:
            if terminal is None:
                raise ValueError(f"search config {self.name!r} prices fights in the run's terms: its caller must pass the fight's currency "
                                 f"(rl/run_currency.py), or play a config without -r")
            kw["terminal"] = terminal
        if self.potion_hold is not None and pos is not None and pos.fight_kind == "hall":  # a hallway fight only; every other call is unchanged
            kw["potion_hold"] = self.potion_hold
        if self.leaf != "rollout":  # a rollout config's arguments stay what they always were
            kw["leaf"] = self.leaf
        if self.lethal_check:  # likewise: a config without the option passes nothing new
            kw["lethal_check"] = True
        if env_prior and os.environ.get("SPIRELINE_GEN_DIR"):  # score-real's lockstep player: the promoted generation's combat prior
            import gen_loop  # noqa: PLC0415

            kw.update(gen_loop.active_agent_extra())
        return kw


def flat(sims: int, name: str | None = None, **kw) -> SearchConfig:
    return SearchConfig(name or f"gen0-{sims}", sims, **kw)


# a3-v1, from round A3's 200-position dev set (resampled 4 x 2,000-simulation labels): more simulations paid off at
# target and choice-screen decisions, not at plain card plays, and not by stakes (low HP, elite/boss, many moves);
# c = 0.2 was never worse than 0.1 and 0.05 was worse; the sweep size m made no measurable difference. So: 30 on a
# plain play, 120 where a target or a choice-screen pick is asked. Mean cost over gen 0's decisions: 53.5 sims (60 flat).
A3_V1 = SearchConfig("a3-v1", 30, min_child_sims=1, c=0.2,
                     schedule=Schedule(high=30, many_moves=99, kind_mult={"target": 4, "select": 4}, lo=1, hi=2000))

# judge-v0 (D41): a3-v1's search with the fight judge (rl/judge.py) at each simulation's leaf in place of a play-out, at
# a3-v1's wall time per decision. A judged simulation costs a fraction of a played-out one, so the same time buys
# JUDGE_SIMS_X times the simulations (measured on the lockstep gym, docs/rl/JUDGE.md); the schedule's shape is a3-v1's.
JUDGE_SIMS_X = 3
JUDGE_V0 = SearchConfig("judge-v0", 30 * JUDGE_SIMS_X, min_child_sims=1, c=0.2, leaf="judge",
                        schedule=Schedule(high=30 * JUDGE_SIMS_X, many_moves=99, kind_mult={"target": 4, "select": 4}, lo=1,
                                          hi=2000 * JUDGE_SIMS_X))

# a3-v1-k (2026-11-04): a3-v1 with the kill check at the root (rl/lethal.py). Not the night's or self-play's config until a board
# measurement says so.
A3_V1_K = dataclasses.replace(A3_V1, name="a3-v1-k", lethal_check=True)

CURRENCY_SUFFIX = "-r"


def with_currency(cfg: SearchConfig) -> SearchConfig:
    """`cfg` priced in the run's terms (named `<cfg>-r`). Only a plain base config takes the suffix: the stakes schedule and the hold go after it."""
    if cfg.run_currency:
        raise ValueError(f"search config {cfg.name!r} already prices fights in the run's terms")
    if cfg.stakes is not None or cfg.potion_hold is not None:
        raise ValueError(f"search config {cfg.name!r} has a stakes schedule or a hold: put -r on its base ({cfg.name.split('@')[0].split('+')[0]}-r)")
    return dataclasses.replace(cfg, name=cfg.name + CURRENCY_SUFFIX, run_currency=True)


A3_V1_R = with_currency(A3_V1)  # the brief's a3-v1-r: a3-v1 with the run currency
JUDGE_V0_R = with_currency(JUDGE_V0)
A3_V1_K_R = with_currency(A3_V1_K)

CONFIGS: dict[str, SearchConfig] = {
    "gen0-60": flat(60),
    "gen0-120": flat(120),
    "a3-v1": A3_V1,
    "a3-v1-k": A3_V1_K,
    "judge-v0": JUDGE_V0,
    "a3-v1-r": A3_V1_R,
    "judge-v0-r": JUDGE_V0_R,
    "a3-v1-k-r": A3_V1_K_R,
}
SELFPLAY = "a3-v1"  # the config self-play runs
SCORE_REAL = "a3-v1"  # the config score-real plays (the same agent, so the promotion test grades what self-play made)


def get(name: str) -> SearchConfig:
    """The named config; "flat-N" is N flat simulations with gen 0's other settings; `<base>@h<H>e<E>b<B>` and `<base>+hold<N>` are
    a config with a stakes schedule and a potion hold (`with_options`)."""
    if name in CONFIGS:
        return CONFIGS[name]
    if name.startswith("flat-") and name[5:].isdigit():
        return flat(int(name[5:]), name)
    if name.endswith(CURRENCY_SUFFIX) and "@" not in name and "+" not in name and len(name) > len(CURRENCY_SUFFIX):
        with contextlib.suppress(KeyError):  # `<base>-r` for any base this module knows (gen0-60-r, flat-30-r)
            return with_currency(get(name[: -len(CURRENCY_SUFFIX)]))
    if "@" in name or "+" in name:
        base, mult, hold = macro_recipe.parse_config_name(name)
        return with_options(get(base), mult, hold)
    raise KeyError(f"unknown search config {name!r}; known: {', '.join(sorted(CONFIGS))}, flat-N, <config>@h<H>e<E>b<B>[+hold<N>]")


def with_options(cfg: SearchConfig, mult=None, hold: int | None = None) -> SearchConfig:
    """`cfg` with a stakes schedule (`mult`: {"hall", "elite", "boss"} or a (hall, elite, boss) sequence) and/or the potion hold at
    `hold` per cent of max HP; the name says both (`a3-v1@h1e4b16+hold30`). Neither: `cfg` itself. A config that already has the
    options is not given them again."""
    if cfg.stakes is not None or cfg.potion_hold is not None:
        raise ValueError(f"search config {cfg.name!r} already has a stakes schedule or a potion hold")
    stakes = None
    if mult is not None:
        m = macro_recipe.stakes_mult(mult)
        stakes = tuple(m[k] for k in macro_recipe.FIGHT_KINDS)
    if stakes is None and hold is None:
        return cfg
    return dataclasses.replace(cfg, name=macro_recipe.config_name(cfg.name, mult, hold), stakes=stakes,
                               potion_hold=(macro_recipe.hold_pct(hold) / 100.0 if hold is not None else None))


def for_agent(sims, stakes: str | None = None, hold: int | None = None) -> SearchConfig:
    """The config an agent plays: `sims` (a config name or a flat budget) as the lab passes it, unless the agent has a stakes schedule
    (agent.json `search_schedule`, as its tag `<base>@h<H>e<E>b<B>`), whose base config is the agent's own and replaces `sims`; and
    the agent's potion hold (`potion_hold`, per cent of max HP) on either."""
    if stakes is not None:
        base, mult = macro_recipe.parse_stakes_tag(stakes)
        return with_options(get(base), mult, hold)
    cfg = get(sims) if isinstance(sims, str) else flat(int(sims))
    if hold is not None and cfg.potion_hold == macro_recipe.hold_pct(hold) / 100.0:
        return cfg  # the lab already asked for this very hold by name
    return with_options(cfg, None, hold)


def position_of(root, info: dict) -> Position:
    """Read the schedule's features from a combat handle's world (`root` has .obs and .handle) and the run info."""
    import live_search  # noqa: PLC0415 -- native

    moves = live_search._distinct(live_search._world_moves(root))  # noqa: SLF001
    if root.obs[live_search._SELECTION_KIND]:  # noqa: SLF001
        kind = "select"
    else:
        per_action: dict[int, int] = {}
        for m in moves:
            per_action[m.action] = per_action.get(m.action, 0) + 1
        kind = "target" if any(v > 1 for v in per_action.values()) else "play"
    max_hp = max(1, int(info.get("max_hp", 1)))
    hp = int(root.hp()) if hasattr(root, "hp") else int(info.get("hp", 0))
    node = int(info.get("node_type", 0))
    return Position(kind=kind, n_moves=len(moves), hp_frac=hp / max_hp, hard=node in (NODE_ELITE, NODE_BOSS),
                    act=1 + max(0, int(info.get("floor", 1)) - 1) // 17,  # the info's "act" is a region id
                    fight="boss" if node == NODE_BOSS else "elite" if node == NODE_ELITE else "hall")
