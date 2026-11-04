"""Named combat-search settings for the PLAY agent (PLAN.md round A3), selectable from rl/gen0.py.

A config fixes how `live_search.search_live` is called at a combat decision: the simulation budget, the root
sweep (`min_child_sims`: every root move gets this many common-random-number playouts before UCT; this is the
search's breadth knob, the counterpart of a Gumbel root's sampled-action count m, which the live search does not
have), the UCT exploration constant `c`, and the confirmation rounds. Every search is resampled (the firewall).

A config may carry a budget schedule: simulations per decision from what the player can see at it (decision kind,
number of distinct legal moves, HP fraction, elite or boss fight). `Position` holds those features; `sims_for`
is pure so it can be tested without native code. `position_of` reads them from a live fight.

    python rl/gen0.py emu --config a3-v1 ...
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field

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

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown decision kind {self.kind!r}")


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

    def high_stakes(self, pos: Position) -> bool:
        s = self.schedule
        return bool(s) and (pos.hard or pos.hp_frac < s.low_hp or pos.n_moves >= s.many_moves)

    def sims_for(self, pos: Position | None) -> int:
        """Simulations for this decision. Without a schedule (or a position) the flat `sims`."""
        s = self.schedule
        if s is None or pos is None:
            return self.sims
        n = s.high if self.high_stakes(pos) else self.sims
        n = round(n * s.kind_mult.get(pos.kind, 1.0))
        return max(s.lo, min(s.hi, n))

    def search_kwargs(self, pos: Position | None = None, env_prior: bool = True) -> dict:
        """Keyword arguments for `live_search.search_live` (add `seed`). With `env_prior`, SPIRELINE_GEN_DIR names a
        generation whose combat prior is added (score-real's lockstep player builds its search from a config alone);
        the emulator driver passes False, because there the agent argument alone decides the prior."""
        kw = {"max_sims": self.sims_for(pos), "min_child_sims": self.min_child_sims, "c": self.c,
              "confirm_pairs": self.confirm_pairs, "time_budget_s": self.time_budget_s, "resample": True}
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

CONFIGS: dict[str, SearchConfig] = {
    "gen0-60": flat(60),
    "gen0-120": flat(120),
    "a3-v1": A3_V1,
    "a3-v1-k": A3_V1_K,
    "judge-v0": JUDGE_V0,
}
SELFPLAY = "a3-v1"  # the config self-play runs
SCORE_REAL = "a3-v1"  # the config score-real plays (the same agent, so the promotion test grades what self-play made)


def get(name: str) -> SearchConfig:
    """The named config; "flat-N" is N flat simulations with gen 0's other settings."""
    if name in CONFIGS:
        return CONFIGS[name]
    if name.startswith("flat-") and name[5:].isdigit():
        return flat(int(name[5:]), name)
    raise KeyError(f"unknown search config {name!r}; known: {', '.join(sorted(CONFIGS))}, flat-N")


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
    return Position(kind=kind, n_moves=len(moves), hp_frac=hp / max_hp,
                    hard=int(info.get("node_type", 0)) in (NODE_ELITE, NODE_BOSS),
                    act=1 + max(0, int(info.get("floor", 1)) - 1) // 17)  # the info's "act" is a region id
