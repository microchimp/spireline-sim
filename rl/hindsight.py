"""Hindsight labels for the run layer: the value of each option on a decision, measured by playing it forward (brief 2a).

For a decision d at state s with options o_1..o_n, the copy forks s (`clone_session`, a resampled run clone), applies one
option, and plays K continuations to the end of the current act, the boss included. A continuation's return G is the floor
the run died on if it died before the act ended; the frozen run head's expected final floor at the first decision of the next
act if it got there; the won-run floor if it won. The label Q(d, o) is the mean of G over the K continuations. The unit is
floors, the run head's own unit, so the label is both: floors reached inside the horizon and the head's value beyond it.

How the player in the replay is fixed (training design 2.3, written into the manifest of every label file): the exam agent of
the label run (its combat prior with the a3-v1 search), the frozen run head of the same generation for the continuations' run
choices, argmax, no exploration; every random stream in a namespace of its own, derived from (run id, floor, decision kind, k)
only. Continuation k of every option therefore uses the same resample seed and the same search seeds, so the options meet the
same map, encounters and reward rolls for as long as their decks allow (common random numbers).

Where the decisions come from: the AI's own exploring runs (a lab self-play job with `--explore-tau`). A run is a function of
its seed, its index and its agent, so it is replayed on the copy (`replay_run`, the same loop as `gen0.play_emu_run`) to the
decision and checked against its record (the cards taken, the map path and the rest choices so far); a replay that has left
its record is refused by reason. The human corpus is not read here: its labels are brief 2b (and D38 keeps the exam fifth of it
out of every label), and no record of a human run enters a label in this file.

    python rl/hindsight.py pilot  --out DIR --k 8 --procs 6 [--quota-per-cell N] [--per-run 2] [--lib DIR] [--until HH:MM]
    python rl/hindsight.py report --out DIR

The pure parts (seeds, the label, the statistics, the plan, the report) import no native code. Lab code: nothing here changes
the engine, the search, self-play, the loop or the night.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LABEL_VERSION = 1
NAMESPACE = "hindsight-v1"  # the seeds' own namespace: no game seed, agent seed or search seed of any other job comes from it
AGENT_SEED = 20261023  # gen0.AGENT_SEED: the base agent seed the exploring runs were played under (an agent seed, never a game seed)
ASCENSION = 10
DEFAULT_K = 8
MAX_ACT_INDEX = 1  # the labelled acts, 0-based: Acts 1 and 2 (Act 3's copy fidelity is barely verified, D38)
CONT_WALL_CAP_S = 600.0  # a continuation still going after this long is stopped and counted as a timeout
CONT_DECISION_CAP = 4000  # engine decisions (run choices and combat moves) in one continuation
REPLAY_WALL_CAP_S = 1800.0  # gen0.RUN_WALL_CAP_S
MAX_OPTIONS = 6  # a screen with more options is shortlisted: the unchanged options and the head's best
ACT_BOSS_FLOORS = (17, 33, 48)  # run_fields.ACT_BOSS_FLOORS: a decision on or past the act's boss floor has nothing left to play
MAX_NEXT_ACT_DECISIONS = 6  # decisions into the next act to find one the head can value
KIND_OF_TYPE = {"card_reward": "card", "rest": "rest", "map": "route", "shop": "shop"}
KINDS = ("card", "rest", "route", "shop")
KIND_CODE = {k: i + 1 for i, k in enumerate(KINDS)}
DEFAULT_RATES = {"card": 0.30, "rest": 0.50, "route": 0.07, "shop": 0.60}  # the chance a decision on the replay is taken, per kind
TAG = "hindsight"  # this job's QUIET.lock line
UNCHANGED_KINDS = frozenset({"card.skip", "select.offer_done", "reward.proceed", "shop.leave", "rest.leave"})
REF = Path(os.environ.get("HINDSIGHT_REFEREE", "D:/spireline/referee"))
QUIET = REF / "QUIET.lock"
MAIN_LOCK = REF / "backlog" / "main.lock"
ACTIVE = REF / "LAB" / "nightly" / "active"


# ---------------------------------------------------------------------------------------------------------- seeds


def _digest(*parts: Any) -> int:
    """A 64-bit number from the parts: stable across processes and runs (never `hash()`)."""
    text = "|".join(str(p) for p in (NAMESPACE, *parts))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def resample_seed(run_id: str, floor: int, kind: str, k: int) -> int:
    """The seed of continuation k's resampled run clone: the same for every option of a decision, a positive 31-bit number."""
    return int(_digest("resample", run_id, int(floor), kind, int(k)) % (2**31 - 1))


def stream_key(run_id: str, floor: int, kind: str, k: int, stream: int = 0) -> list[int]:
    """The entropy of continuation k's random streams (`numpy.random.default_rng`): stream 0 the searches' seeds, 1 the run
    policy's exploration stream (never drawn from at tau = eps = 0, but given so a rules fallback has one)."""
    return [_digest("stream", run_id, int(floor), kind, int(k), int(stream)), KIND_CODE.get(kind, 0), int(floor), int(k), int(stream)]


def uniform(*parts: Any) -> float:
    """A number in [0, 1) from the parts: the plan's coin for one decision of one run."""
    return _digest("coin", *parts) / 2**64


def decision_id(run_id: str, ordinal: int, kind: str, floor: int) -> str:
    return f"{run_id}|d{int(ordinal):03d}|{kind}|f{int(floor):02d}"


# ---------------------------------------------------------------------------------------------------------- the label


def label_return(end: str, floor: int, head_value: float | None, won_floor: float) -> float | None:
    """G of one continuation (training design 2.1): the death floor if the run died before the act ended; the frozen head's
    expected final floor at the next act's first decision if it got there; the won-run floor if it won. None for any other end
    (a timeout, a step cap, no value from the head): such a continuation has no return and the decision is refused."""
    if end == "death":
        return float(floor)
    if end == "next_act":
        return None if head_value is None else float(head_value)
    if end == "win":
        return float(won_floor)
    return None


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def centred(values: Sequence[float]) -> list[float]:
    """A(d, o) = Q(d, o) minus the mean of Q over the decision's options (training design 2.5): the luck of the rebuilt state
    moves every option together and cancels."""
    m = mean(values)
    return [v - m for v in values]


def sample_var(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def option_values(g: Sequence[Sequence[float]]) -> list[float]:
    """Q(d, o) for each option: the mean over the K continuations. `g[o][k]` is G of option o, continuation k."""
    return [mean(row) for row in g]


def paired_variances(g: Sequence[Sequence[float]]) -> list[tuple[float, int]]:
    """For every pair of options of one decision, the variance over k of the per-draw difference G[o][k] - G[o'][k], with its
    degrees of freedom. The draws are paired (same k, same seeds), so this is the noise of an option difference."""
    out = []
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            n = min(len(g[i]), len(g[j]))
            if n >= 2:
                out.append((sample_var([g[i][k] - g[j][k] for k in range(n)]), n - 1))
    return out


def single_variances(g: Sequence[Sequence[float]]) -> list[tuple[float, int]]:
    """For every option, the variance over k of its own return, with its degrees of freedom."""
    return [(sample_var(row), len(row) - 1) for row in g if len(row) >= 2]


def advantage_variances(g: Sequence[Sequence[float]]) -> list[tuple[float, int]]:
    """For every option, the variance over k of its draw minus the mean of the options' draws on the same k (the noise of one
    centred option value), with its degrees of freedom."""
    n = min((len(r) for r in g), default=0)
    if n < 2 or len(g) < 2:
        return []
    mk = [mean([g[o][k] for o in range(len(g))]) for k in range(n)]
    return [(sample_var([g[o][k] - mk[k] for k in range(n)]), n - 1) for o in range(len(g))]


def pooled(vs: Iterable[tuple[float, int]]) -> float | None:
    """The degrees-of-freedom-weighted mean of variances, as a variance; None when there are none."""
    num = den = 0.0
    for v, df in vs:
        if df > 0 and not math.isnan(v):
            num += v * df
            den += df
    return num / den if den else None


def k_needed(sigma: float, n_decisions: int = 1500, se: float = 0.05, tau2: float = 0.0) -> float | None:
    """The continuations per option that put the standard error of a mean advantage over `n_decisions` decisions at `se`
    floors: SE = sqrt((tau2 + sigma^2 / K) / n), so K = sigma^2 / (n se^2 - tau2). `sigma` is the paired SD of an option
    difference; `tau2` the true spread of the advantages between decisions (zero is the design's arithmetic). None when
    n se^2 <= tau2: no K reaches it, only more decisions."""
    room = n_decisions * se * se - tau2
    return None if room <= 0 else sigma * sigma / room


# ---------------------------------------------------------------------------------------------------------- the run clone


def clone_session(s: Any, seed: int) -> Any:
    """A `RunSession` standing in a resampled fork of `s`'s run (`Sts2RunEnv.clone(resample_hidden=True, seed=seed)`), with the
    session's own bookkeeping copied: everything the run has not yet paid out is redrawn from `seed`, what has happened is
    untouched. The base session is not changed. Close the clone when done."""
    c = copy.copy(s)
    c.stats = copy.deepcopy(s.stats)
    c.visit = copy.deepcopy(s.visit)
    c.env = s.env.clone(resample_hidden=True, seed=int(seed))
    return c


# ---------------------------------------------------------------------------------------------------------- the player


@dataclass
class Player:
    """What plays a continuation or a replay: the run policy, the fight player, the head's value of a state, the judge at a
    boss door. Tests replace any of them; `build_player` makes the real ones."""

    policy: Any  # pick(view, decision, rng) -> Pick
    play_fight: Callable[[Any, Any, float | None], int]  # (session, rng, deadline) -> combat decisions made; plays the open fight to its end
    head: Callable[[Any, Any], float | None]  # (view, decision) -> the frozen head's expected final floor of the state, or None
    won_floor: float = 53.0
    judge: Callable[[int, int], dict | None] | None = None  # (combat handle, max_hp) -> the judge's estimate of the fight
    keep_doors: bool = False
    wall_cap_s: float = CONT_WALL_CAP_S
    decision_cap: int = CONT_DECISION_CAP
    agent: Any = None  # the gen_loop.Agent behind a real player (the session's mask version)


def head_value(policy: Any, view: Any, decision: Any) -> float | None:
    """The frozen run head's expected final floor of a state, read at a decision: the value of the option its argmax would take
    (the best scorable option's afterstate). None when no option is scorable."""
    from run_train import models  # noqa: PLC0415
    from run_train import policy as run_policy  # noqa: PLC0415

    view = decision.view or view
    scorable = [o for o in decision.options if o.scorable]
    if not scorable:
        return None
    probs = run_policy.floor_probs(policy.nets, view, scorable)
    return float(models.expected_floors(probs, policy.W).max())


def option_head_values(policy: Any, view: Any, decision: Any) -> list[float | None]:
    """The head's value of each option's afterstate (None for an unscorable option)."""
    from run_train import models  # noqa: PLC0415
    from run_train import policy as run_policy  # noqa: PLC0415

    view = decision.view or view
    idx = [i for i, o in enumerate(decision.options) if o.scorable]
    out: list[float | None] = [None] * len(decision.options)
    if idx:
        probs = run_policy.floor_probs(policy.nets, view, [decision.options[i] for i in idx])
        for i, v in zip(idx, models.expected_floors(probs, policy.W).tolist(), strict=True):
            out[i] = float(v)
    return out


class Timeout(RuntimeError):
    """A continuation outlived its wall cap."""


class SearchFighter:
    """The exam agent's fight play, as `gen0.play_emu_run` plays a fight: each combat decision is `live_search.search_live` on a
    combat fork of the position under the agent's search config and prior, and the best move is stepped on the run handle."""

    def __init__(self, agent: Any, config: str = "a3-v1"):
        import search_configs  # noqa: PLC0415

        self.cfg = search_configs.get(config)
        self.extra = agent.search_extra()

    def __call__(self, s: Any, rng: Any, deadline: float | None = None) -> int:
        import live_search  # noqa: PLC0415
        import search_configs  # noqa: PLC0415
        from run_train import engine  # noqa: PLC0415
        from sts2_gym import native  # noqa: PLC0415

        env, made = s.env, 0
        while s.phase == engine.rc.PHASE_COMBAT:
            if deadline is not None and time.perf_counter() > deadline:
                raise Timeout("the continuation outlived its wall cap")
            root = s.combat(fork_combat=True).combat_fork(None)
            try:
                pos = search_configs.position_of(root, s.info) if self.cfg.schedule else None
                kw = self.cfg.search_kwargs(pos, env_prior=False)  # the agent argument alone decides the prior
                kw.update(self.extra)
                res = live_search.search_live(root.handle, seed=int(rng.integers(1 << 30)), **kw)
            finally:
                native.destroy(root.handle) if not hasattr(root, "close") else root.close()
            st = native.run_step(s.handle, res.best.action, res.best.target, env._run_obs_buf, env._run_rew_buf,
                                 env._run_terminal_buf, env._run_truncated_buf)
            if st != 0:
                raise RuntimeError(f"combat action refused (status {st})")
            made += 1
        return made


def build_player(agent_dir: str | Path, *, config: str = "a3-v1", judge_path: str | Path | None = None, keep_doors: bool = False,
                 explore: tuple[float, float] | None = None) -> Player:
    """The real player of a generation directory: its combat prior with the config's search, its run head. With `explore`
    (tau, eps) the run choices are sampled as the exploring self-play job did (a replay); without, argmax (a continuation)."""
    import gen_loop  # noqa: PLC0415

    agent = gen_loop.load_agent(agent_dir)
    if agent.macro != "vrun":
        raise ValueError(f"{agent_dir}: hindsight labels need a vrun agent (the head values the next act)")
    agent = gen_loop.exploring_agent(agent, *explore) if explore else dataclasses.replace(agent, tau=0.0, eps=0.0)
    policy = agent.macro_policy()
    judge = None
    if judge_path:
        import judge_client  # noqa: PLC0415

        cpu = judge_client.CpuJudge(str(judge_path))

        def judge(handle: int, max_hp: int) -> dict | None:
            e = judge_client.estimate(cpu.rows([handle])[0], max_hp)
            return {"value": round(e.value, 3), "p_death": round(e.p_death, 4), "hp_if_win": round(e.hp_if_win, 2),
                    "potions_if_win": round(e.potions_if_win, 3)}

    from run_train import vocab  # noqa: PLC0415

    return Player(policy=policy, play_fight=SearchFighter(agent, config), head=lambda view, d: head_value(policy, view, d),
                  won_floor=float(vocab.WON_FLOOR), judge=judge, keep_doors=keep_doors, agent=agent)


def session_for(seed: str, agent: Any, config: str = "a3-v1", ascension: int = ASCENSION) -> Any:
    """The run session of an exploring run: the budget and mask version `gen0.play_emu_run` builds it with."""
    import gen0  # noqa: PLC0415
    import search_configs  # noqa: PLC0415
    from run_train import engine  # noqa: PLC0415

    cfg = search_configs.get(config)
    return engine.RunSession(seed, ascension, engine.CombatBudget(cfg.sims, cfg.sims), **gen0.session_kwargs(agent))


# ---------------------------------------------------------------------------------------------------------- one continuation


def door_of(s: Any, player: Player) -> dict:
    """The state at the boss door and the judge's estimate of the fight there, recorded beside the continuation so that a
    multi-fidelity label (the judge at the door for every continuation, the boss played for some) needs no replay."""
    from sts2_gym import native  # noqa: PLC0415

    info = s.info
    fight = s.combat(fork_combat=True)
    root = fight.combat_fork(None)
    try:
        door: dict[str, Any] = {"floor": int(info["floor"]), "hp": int(info["hp"]), "max_hp": int(info["max_hp"]),
                                "gold": int(info["gold"]), "deck": int(info["deck_size"]), "relics": int(info["relic_count"]),
                                "encounter": int(fight.encounter_id)}
        if player.judge is not None:
            door["judge"] = player.judge(root.handle, int(info["max_hp"]))
        if player.keep_doors:
            door["doc"] = native.get_state_json(root.handle)
    finally:
        native.destroy(root.handle) if not hasattr(root, "close") else root.close()
    return door


def run_to_act_end(s: Any, player: Player, start_act: int, rng: Any, explore: Any) -> dict:
    """Play `s` on to the end of the act it started in, boss included, under the fixed player.

    Returns {end, floor, head, door, fights, moves, decisions}: `end` is "death" (the run died; `floor` is where), "next_act"
    (the first decision of the next act whose state the head can value; `head` is that value), "win" (the last act's boss died),
    "timeout", "step_cap" or "no_value" (the next act began and the head could value none of its first decisions).
    """
    from run_train import engine  # noqa: PLC0415

    deadline = time.perf_counter() + player.wall_cap_s
    out: dict[str, Any] = {"end": "?", "floor": 0, "head": None, "door": None, "fights": 0, "moves": 0, "decisions": 0}
    past = 0  # decisions seen in the next act without a value
    try:
        while True:
            if time.perf_counter() > deadline:
                raise Timeout("the continuation outlived its wall cap")
            if out["decisions"] + out["moves"] >= player.decision_cap:
                out["end"] = "step_cap"
                break
            d = s.decision()
            if d is None and s.phase != engine.rc.PHASE_COMBAT:
                out["end"] = "win" if s.won else "death"
                break
            if d is not None:
                view = s.view()
                if s.act_index > start_act:
                    value = player.head(view, d)
                    if value is not None:
                        out["end"], out["head"] = "next_act", value
                        break
                    past += 1
                    if past > MAX_NEXT_ACT_DECISIONS:
                        out["end"] = "no_value"
                        break
                pick = player.policy.pick(view, d, explore)
                s.apply(d.options[pick.index])
                out["decisions"] += 1
                continue
            if out["door"] is None and s.info["node_type"] == engine.vocab.NODE_BOSS:
                out["door"] = door_of(s, player)
            out["fights"] += 1
            out["moves"] += player.play_fight(s, rng, deadline)
    except Timeout:
        out["end"] = "timeout"
    out["floor"] = int(s.info["floor"])
    return out


def continuation(base: Any, option: Any, player: Player, *, run_id: str, floor: int, kind: str, k: int) -> dict:
    """Continuation k of one option: fork `base` with the decision's resample seed for k, apply the option, play on to the end
    of the act. The seeds depend on (run id, floor, kind, k) only, never on the option: every option meets the same draws."""
    import numpy as np  # noqa: PLC0415

    t0, c0 = time.perf_counter(), time.process_time()
    start_act = base.act_index
    rng = np.random.default_rng(stream_key(run_id, floor, kind, k, 0))
    explore = np.random.default_rng(stream_key(run_id, floor, kind, k, 1))
    c = clone_session(base, resample_seed(run_id, floor, kind, k))
    try:
        c.apply(option)
        out = run_to_act_end(c, player, start_act, rng, explore)
    finally:
        c.close()
    g = label_return(out["end"], out["floor"], out["head"], player.won_floor)
    return {"k": k, "G": g, "end": out["end"], "floor": out["floor"], "head": out["head"], "fights": out["fights"],
            "moves": out["moves"], "seconds": round(time.perf_counter() - t0, 2), "cpu": round(time.process_time() - c0, 2),
            "door": out["door"]}


def shortlist(options: Sequence[Any], values: Sequence[float | None], limit: int = MAX_OPTIONS,
              keep: Iterable[int] = ()) -> list[int]:
    """The options to play forward: all of them when there are `limit` or fewer, else the unchanged options (skip, leave), the
    options in `keep` (the run's own pick) and the head's best of the rest, in screen order. Options of kind "unknown" are never
    played (nothing to apply)."""
    allowed = [i for i, o in enumerate(options) if o.kind != "unknown"]
    if len(allowed) <= limit:
        return allowed
    chosen = {i for i in allowed if options[i].kind in UNCHANGED_KINDS} | {i for i in keep if i in allowed}
    rest = sorted((i for i in allowed if i not in chosen), key=lambda i: -(values[i] if values[i] is not None else -1e9))
    for i in rest:
        if len(chosen) >= limit:
            break
        chosen.add(i)
    return sorted(chosen)


def label_decision(base: Any, decision: Any, view: Any, pick: Any, player: Player, *, run_id: str, ordinal: int, kind: str, k: int,
                   max_options: int = MAX_OPTIONS) -> dict:
    """Play every option of the decision K times and return its label record (without the run's own fields)."""
    from run_train import shards  # noqa: PLC0415

    heads = option_head_values(player.policy, view, decision) if hasattr(player.policy, "nets") else [None] * len(decision.options)
    chosen = shortlist(decision.options, heads, max_options, keep=[pick.index])
    conts: list[dict] = []
    for kk in range(k):  # draw-major: the same draw is played for every option before the next draw
        for oi in chosen:
            row = continuation(base, decision.options[oi], player, run_id=run_id, floor=view.floor, kind=kind, k=kk)
            row["o"] = oi
            conts.append(row)
    bad = [c for c in conts if c["G"] is None]
    rec: dict[str, Any] = {
        "version": LABEL_VERSION,
        "id": decision_id(run_id, ordinal, kind, view.floor),
        "ordinal": ordinal, "kind": kind, "type": decision.type, "act": int(view.act_index), "floor": int(view.floor),
        "hp": int(view.hp), "max_hp": int(view.max_hp), "gold": int(view.gold), "deck": len(view.deck), "relics": len(view.relics),
        "potions": len(view.potions), "k": k,
        "options": [dict(shards.option_to_json(decision.options[i]), idx=i) for i in range(len(decision.options))],
        "played": chosen, "taken": int(pick.index), "head_values": heads,
        "continuations": conts,
    }
    if bad:
        rec["refused"] = "no_return:" + ",".join(sorted({c["end"] for c in bad}))
        return rec
    g = decision_g(rec)
    assert g is not None
    q = option_values(g)
    rec["Q"] = dict(zip(map(str, chosen), (round(x, 4) for x in q), strict=True))
    rec["A"] = dict(zip(map(str, chosen), (round(x, 4) for x in centred(q)), strict=True))
    return rec


# ---------------------------------------------------------------------------------------------------------- the replay


@dataclass
class Point:
    """A run decision reached by the replay, handed to the hook before it is applied."""

    session: Any
    decision: Any
    view: Any
    pick: Any  # what the exploring policy took here (already drawn; the hook must not draw from the run's streams)
    ordinal: int
    kind: str | None  # one of KINDS, or None for a screen the labels do not cover
    fields: Any


def prefix_check(fields: Any, record: dict) -> str | None:
    """None when the replay so far (cards, rests, map path) is a prefix of the run's record; else the field that differs."""
    for name, mine in (("cards", fields.cards), ("rests", fields.rests)):
        want = record.get(name)
        if isinstance(want, list) and [list(x) for x in mine] != [list(x) for x in want[: len(mine)]]:
            return name
    nodes, want_nodes = "".join(fields.nodes), record.get("nodes")
    if isinstance(want_nodes, str) and want_nodes[: len(nodes)] != nodes:
        return "nodes"
    return None


def replay_run(seed: str, ix: int, replayer: Player, record: dict, hook: Callable[[Point], bool | None], *,
               config: str = "a3-v1", agent_seed: int = AGENT_SEED, ascension: int = ASCENSION) -> dict:
    """Replay a recorded exploring run on the copy and call `hook` at each of its run decisions, before the decision is applied.

    The loop is `gen0.play_emu_run`'s, with the same streams (`default_rng([agent_seed, ix])` for the searches' seeds,
    `[agent_seed, ix, 1]` for the exploration), so the run is the recorded one; the hook only reads and forks. `record` is the
    run's row (`runs.jsonl`): at every decision the replay's cards, rests and map path so far are checked against it, and a
    replay that has left its record stops there ("diverged:<field>"). The hook returns True to stop the replay.
    Returns {end, decisions, floor, checked, seconds}.
    """
    import numpy as np  # noqa: PLC0415

    import run_fields  # noqa: PLC0415
    from run_train import engine  # noqa: PLC0415

    t0 = time.perf_counter()
    rng = np.random.default_rng([agent_seed, ix])
    explore = np.random.default_rng([agent_seed, ix, 1])
    s = session_for(seed, replayer.agent, config, ascension)
    fields, slugs = run_fields.RunFields(), run_fields.card_slugs()
    res: dict[str, Any] = {"end": "?", "decisions": 0, "floor": 0, "checked": 0}
    ordinal = 0
    try:
        while True:
            if time.perf_counter() - t0 > REPLAY_WALL_CAP_S:
                res["end"] = "timeout"
                break
            d = s.decision()
            if d is None and s.phase != engine.rc.PHASE_COMBAT:
                res["end"] = "win" if s.won else "death"
                break
            if d is not None:
                view = s.view()
                pick = replayer.policy.pick(view, d, explore)
                diff = prefix_check(fields, record)
                if diff is not None:
                    res["end"] = "diverged:" + diff
                    break
                res["checked"] += 1
                kind = KIND_OF_TYPE.get(d.type)
                if hook(Point(s, d, view, pick, ordinal, kind, fields)):
                    res["end"] = "hooked"
                    break
                ordinal += 1
                fields.emu_macro(view.floor, d.options[pick.index], slugs)
                s.apply(d.options[pick.index])
                res["decisions"] += 1
                continue
            replayer.play_fight(s, rng, None)
        res["floor"] = int(s.info["floor"])
        if res["end"] in ("win", "death") and not (
            res["floor"] == int(record.get("floor", res["floor"])) and prefix_check(fields, record) is None
        ):
            res["end"] = "diverged:final"
    finally:
        s.close()
    res["seconds"] = round(time.perf_counter() - t0, 2)
    return res


# ---------------------------------------------------------------------------------------------------------- the plan and the sources


@dataclass(frozen=True)
class Plan:
    """Which decisions of a replay are labelled."""

    k: int = DEFAULT_K
    quota: dict[str, int] = field(default_factory=dict)  # "card/a1" -> decisions wanted; empty: no limit
    per_run: int = 2  # decisions labelled from one run (the replay is paid once), at most one per kind and act
    rates: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RATES))
    max_options: int = MAX_OPTIONS
    acts: tuple[int, ...] = tuple(range(MAX_ACT_INDEX + 1))

    def cell(self, kind: str, act: int) -> str:
        return f"{kind}/a{act + 1}"

    def wants(self, run_id: str, ordinal: int, kind: str | None, act: int, floor: int, taken: Counter,
              have: Callable[[], Counter] | Counter) -> str | None:
        """None to label this decision, else why not (the reason it is passed over). `have` is the label count of each cell so far
        (a function, read only when the cheaper tests pass)."""
        if kind is None:
            return "kind"
        if act not in self.acts:
            return "act"
        if act < len(ACT_BOSS_FLOORS) and floor >= ACT_BOSS_FLOORS[act]:
            return "post_boss"
        cell = self.cell(kind, act)
        if sum(taken.values()) >= self.per_run or taken[cell]:
            return "run_cap"
        if uniform(run_id, ordinal) >= self.rates.get(kind, 0.0):
            return "coin"
        if self.quota:
            counts = have() if callable(have) else have
            if counts[cell] >= self.quota.get(cell, 0):
                return "quota"
        return None

    def open_cells(self, have: Counter) -> list[str]:
        if not self.quota:
            return [self.cell(k, a) for k in KINDS for a in self.acts]
        return [c for c, n in self.quota.items() if have[c] < n]


def exploring_runs(runs_root: Path, generation: int, prior_sha: str | None = None) -> list[dict]:
    """The recorded exploring runs of one generation's fight player: the rows of every lab self-play job whose command names
    `--explore-tau`, each with its job and its exploration (tau, eps). A job played by another prior is left out when `prior_sha`
    is given."""
    out = []
    for job in sorted(Path(runs_root).glob("*-selfplay")):
        try:
            doc = json.loads((job / "job.json").read_text(encoding="utf-8"))
            summ = json.loads((job / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cmd = doc.get("params", {}).get("cmd", [])
        if "--explore-tau" not in cmd or doc.get("rc") != 0:
            continue
        agent = summ.get("agent", {})
        if agent.get("generation") != generation or (prior_sha and agent.get("prior_sha256") != prior_sha):
            continue
        explore = (float(agent.get("tau", 0.0)), float(agent.get("eps", 0.0)))
        for line in (job / "runs.jsonl").read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(ValueError):
                r = json.loads(line)
                if r.get("end") in ("win", "death") and not r.get("crash") and not r.get("lost") and r.get("config") == "a3-v1":
                    out.append({**r, "_job": job.name, "_explore": explore})
    return out


def labels_dir(out: Path) -> Path:
    return Path(out) / "labels.d"


def read_labels_file(path: Path) -> list[dict]:
    recs = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            with contextlib.suppress(ValueError):
                r = json.loads(line)  # a line cut by a stop does not parse and is dropped: its decision is played again
                if isinstance(r, dict) and r.get("id"):
                    recs.append(r)
    return recs


def read_labels(out: Path) -> list[dict]:
    """Every label record under `out/labels.d` (one file per run)."""
    d = labels_dir(out)
    return [r for f in (sorted(d.glob("*.jsonl")) if d.is_dir() else []) for r in read_labels_file(f)]


def count_cells(recs: Iterable[dict]) -> Counter:
    return Counter(f"{r['kind']}/a{r['act'] + 1}" for r in recs if not r.get("refused"))


# ---------------------------------------------------------------------------------------------------------- the worker task

_W: dict[str, Any] = {}


def _init_worker(out: str, agent_dir: str, config: str, judge: str, keep_doors: bool, plan: dict, lib: str) -> None:
    if lib:
        os.environ["STS2_LIB_PATH"] = lib
    os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")
    from sts2_gym import procutil  # noqa: PLC0415

    procutil.lower_priority()  # BelowNormal (Windows priority class)
    _W.update(out=Path(out), agent_dir=agent_dir, config=config, judge=judge or None, keep_doors=keep_doors,
              plan=Plan(k=plan["k"], quota=plan["quota"], per_run=plan["per_run"], rates=plan["rates"],
                        max_options=plan["max_options"], acts=tuple(plan["acts"])), players={})


def _player(explore: tuple[float, float] | None) -> Player:
    cache = _W["players"]
    if explore not in cache:
        cache[explore] = build_player(_W["agent_dir"], config=_W["config"], judge_path=_W["judge"] if explore is None else None,
                                      keep_doors=_W["keep_doors"] and explore is None, explore=explore)
    return cache[explore]


def run_task(task: tuple) -> dict:
    """Replay one recorded run and label the decisions the plan takes from it. `task` is ("hs", seed, ix, record)."""
    _, seed, ix, record = task
    plan: Plan = _W["plan"]
    out: Path = _W["out"]
    replayer = _player(tuple(record["_explore"]))
    cont = _player(None)
    sink_path = labels_dir(out) / f"{seed}.jsonl"
    sink_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = {r["id"] for r in read_labels_file(sink_path)}
    taken: Counter = Counter()
    passed: Counter = Counter()
    refused: Counter = Counter()
    state = {"labelled": 0, "seconds": 0.0}

    cache: dict[str, Any] = {"at": -1e9, "counts": Counter()}

    def have() -> Counter:
        """The labels of each cell so far, read from the files at most every five seconds."""
        if time.monotonic() - cache["at"] > 5.0:
            cache["at"], cache["counts"] = time.monotonic(), count_cells(read_labels(out))
        return cache["counts"]

    def hook(pt: Point) -> bool:
        act = int(pt.view.act_index)
        if act > max(plan.acts) or (plan.quota and not plan.open_cells(have())):
            return True
        why = plan.wants(seed, pt.ordinal, pt.kind, act, int(pt.view.floor), taken, have)
        if why is None and sum(1 for o in pt.decision.options if o.kind != "unknown") < 2:
            why = "one_option"
        if why is not None:
            passed[why] += 1
            return sum(taken.values()) >= plan.per_run
        kind = pt.kind
        assert kind is not None
        taken[plan.cell(kind, act)] += 1
        if decision_id(seed, pt.ordinal, kind, int(pt.view.floor)) in done_ids:
            return sum(taken.values()) >= plan.per_run
        t1 = time.perf_counter()
        try:
            rec = label_decision(pt.session, pt.decision, pt.view, pt.pick, cont, run_id=seed, ordinal=pt.ordinal, kind=kind,
                                 k=plan.k, max_options=plan.max_options)
        except Exception as error:  # noqa: BLE001 -- refused by reason, never lost
            refused["error:" + type(error).__name__] += 1
            return sum(taken.values()) >= plan.per_run
        state["seconds"] += time.perf_counter() - t1
        rec["run"] = {"seed": seed, "ix": ix, "job": record["_job"], "final_floor": record.get("floor")}
        if rec.get("refused"):
            refused[rec["refused"]] += 1
        else:
            state["labelled"] += 1
        with open(sink_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return sum(taken.values()) >= plan.per_run

    res = replay_run(seed, ix, replayer, record, hook, config=_W["config"])
    return {"_task": ["hs", seed, ix], "seed": seed, "ix": ix, "job": record["_job"], "labelled": state["labelled"],
            "refused": dict(refused), "passed": dict(passed), "replay": res, "label_seconds": round(state["seconds"], 1)}


def capped_row(task: tuple, reason: str, seconds: float, prog: dict) -> dict:
    return {"_task": ["hs", task[1], task[2]], "seed": task[1], "ix": task[2], "capped": True, "cap_reason": reason,
            "seconds": round(seconds, 1), "labelled": 0}


# ---------------------------------------------------------------------------------------------------------- the machine guards


def utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _quiet_lines() -> list[str]:
    try:
        return [x for x in QUIET.read_text(encoding="utf-8").splitlines() if x.strip()]
    except FileNotFoundError:
        return []


def _age_h(line: str) -> float:
    try:
        stamp = line.split()[1].rstrip("Z")
        t = dt.datetime.fromisoformat(stamp if len(stamp) > 16 else stamp + ":00").replace(tzinfo=dt.timezone.utc)
        return (utc() - t).total_seconds() / 3600
    except (IndexError, ValueError):
        return 0.0


def blocked() -> str | None:
    """Why the job must not compute now, or None: the night is running, main.lock stands, or another live QUIET line does (any tag
    but our own and the always-on loop's `gym-loop`, which pauses its step while ours stands)."""
    if ACTIVE.exists():
        return "nightly-active"
    if MAIN_LOCK.exists():
        return "main.lock"
    for x in _quiet_lines():
        tag = x.split()[0]
        if tag not in (TAG, "gym-loop") and _age_h(x) < 2.0:
            return "quiet:" + tag
    return None


def hold(on: bool) -> None:
    """Take (or drop) this job's own QUIET.lock line (`<tag> <UTC> pid=<Windows PID>`); other lines are kept as they are."""
    keep = [x for x in _quiet_lines() if x.split()[0] != TAG]
    if on:
        keep.append(f"{TAG} {utc().strftime('%Y-%m-%dT%H:%M:%SZ')} pid={os.getpid()}")
    if keep:
        QUIET.write_text("".join(x + "\n" for x in keep), encoding="utf-8", newline="\n")
    else:
        QUIET.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------------------- the report


def decision_g(rec: dict) -> list[list[float]] | None:
    """G[option][k] of a label record (the played options in order), or None for a refused record."""
    if rec.get("refused"):
        return None
    cell = {(c["o"], c["k"]): c["G"] for c in rec["continuations"]}
    return [[cell[(o, kk)] for kk in range(rec["k"])] for o in rec["played"]]


def die_at_boss(rec: dict) -> bool:
    """Every continuation of every played option died on the act's boss floor (the boss killed it)."""
    b = ACT_BOSS_FLOORS[min(int(rec["act"]), len(ACT_BOSS_FLOORS) - 1)]
    return all(c["end"] == "death" and c["floor"] == b for c in rec["continuations"])


def summarize(recs: Sequence[dict], n_decisions: int = 1500, se: float = 0.05) -> dict:
    """The pilot's aggregates for a set of label records (one cell, or all). Floors throughout."""
    good = [r for r in recs if not r.get("refused")]
    out: dict[str, Any] = {"decisions": len(good), "refused": len(recs) - len(good)}
    if not good:
        return out
    gs = [g for g in (decision_g(r) for r in good) if g is not None]
    v_pair = pooled(x for g in gs for x in paired_variances(g))
    v_single = pooled(x for g in gs for x in single_variances(g))
    v_adv = pooled(x for g in gs for x in advantage_variances(g))
    qs = [option_values(g) for g in gs]
    ranges = sorted(max(q) - min(q) for q in qs)
    conts = [c for r in good for c in r["continuations"]]
    ends = Counter(c["end"] for c in conts)
    kk = good[0]["k"]
    # the true spread of the centred values between decisions: the variance of the observed ones less their noise
    obs_a = [a for q in qs for a in centred(q)]
    tau2 = max(0.0, sample_var(obs_a) - (v_adv or 0.0) / kk) if len(obs_a) > 2 else 0.0
    sigma = math.sqrt(v_pair) if v_pair is not None else None
    secs = sorted(c["seconds"] for c in conts)
    cpu = [c["cpu"] for c in conts if c.get("cpu") is not None]
    out.update({
        "options_mean": round(mean([len(g) for g in gs]), 2),
        "k": kk,
        "paired_sd": None if sigma is None else round(sigma, 3),
        "single_sd": None if v_single is None else round(math.sqrt(v_single), 3),
        "option_corr": None if not (v_pair is not None and v_single) else round(1 - v_pair / (2 * v_single), 3),
        "advantage_noise_sd": None if v_adv is None else round(math.sqrt(v_adv), 3),
        "q_range_mean": round(mean(ranges), 3), "q_range_median": round(ranges[len(ranges) // 2], 3),
        "q_range_p90": round(ranges[min(len(ranges) - 1, int(0.9 * len(ranges)))], 3),
        "flat_share": round(sum(1 for x in ranges if x < 1e-9) / len(ranges), 3),
        "all_die_at_boss_share": round(sum(1 for r in good if die_at_boss(r)) / len(good), 3),
        "tau_adv": round(math.sqrt(tau2), 3),
        "cont_seconds_mean": round(mean(secs), 2), "cont_seconds_median": round(secs[len(secs) // 2], 2),
        "cont_cpu_seconds_mean": round(mean(cpu), 2) if cpu else None,
        "cont_fights_mean": round(mean([c["fights"] for c in conts]), 2),
        "cont_moves_mean": round(mean([c["moves"] for c in conts]), 1),
        "ends": {e: round(n / len(conts), 3) for e, n in sorted(ends.items())},
        "continuations": len(conts),
        "decision_seconds_mean": round(mean([sum(c["seconds"] for c in r["continuations"]) for r in good]), 1),
    })
    out.update(head_agreement(good))
    if sigma is not None:
        kn = k_needed(sigma, n_decisions, se)
        kt = k_needed(sigma, n_decisions, se, tau2)
        out["k_for_se"] = {"n_decisions": n_decisions, "se": se, "k": None if kn is None else round(kn, 2),
                           "k_with_tau": None if kt is None else round(kt, 2)}
    return out


def head_agreement(recs: Sequence[dict]) -> dict:
    """How the labels sit against the head's own ranking of the played options: the share of decisions whose label-best option is
    the head's best, and the mean Spearman correlation between the head's option values and the labels (decisions with at least
    three options). A label that always agrees with the head adds nothing the head does not have."""
    agree = n = 0
    rhos: list[float] = []
    for r in recs:
        heads = [r["head_values"][o] for o in r["played"]]
        if any(h is None for h in heads) or r.get("Q") is None:
            continue
        q = [r["Q"][str(o)] for o in r["played"]]
        n += 1
        top_q = [i for i, v in enumerate(q) if v == max(q)]
        agree += int(heads.index(max(heads)) in top_q)
        if len(q) >= 3:
            rho = spearman(heads, q)
            if rho is not None:
                rhos.append(rho)
    return {"head_top1_agree": None if not n else round(agree / n, 3),
            "head_label_spearman": None if not rhos else round(mean(rhos), 3), "head_compared": n}


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for m in range(i, j + 1):
            ranks[order[m]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Rank correlation (ties share their mean rank); None when either side is constant."""
    rx, ry = _ranks(x), _ranks(y)
    mx, my = mean(rx), mean(ry)
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True)) / (sx * sy)


def report(out: Path) -> dict:
    """The pilot's report: aggregates only, by kind and act, over every label record under `out`."""
    recs = read_labels(out)
    cells: dict[str, list[dict]] = defaultdict(list)
    by_kind: dict[str, list[dict]] = defaultdict(list)
    by_act: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        cells[f"{r['kind']}/a{r['act'] + 1}"].append(r)
        by_kind[r["kind"]].append(r)
        by_act[f"a{r['act'] + 1}"].append(r)
    doc: dict[str, Any] = {"all": summarize(recs), "by_kind": {k: summarize(v) for k, v in sorted(by_kind.items())},
                           "by_act": {k: summarize(v) for k, v in sorted(by_act.items())},
                           "cells": {k: summarize(v) for k, v in sorted(cells.items())}}
    tasks = []
    tf = Path(out) / "tasks.jsonl"
    if tf.is_file():
        for line in tf.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(ValueError):
                tasks.append(json.loads(line))
    ends = Counter(str((t.get("replay") or {}).get("end", "?")).split(":")[0] for t in tasks)
    doc["tasks"] = {"rows": len(tasks), "capped": sum(1 for t in tasks if t.get("capped")), "replay_ends": dict(ends),
                    "replay_diverged": sum(1 for t in tasks if str((t.get("replay") or {}).get("end", "")).startswith("diverged")),
                    "replay_seconds_mean": round(mean([t["replay"]["seconds"] for t in tasks if t.get("replay")]), 1) if tasks else None,
                    "passed": dict(sum((Counter(t.get("passed") or {}) for t in tasks), Counter())),
                    "refused": dict(sum((Counter(t.get("refused") or {}) for t in tasks), Counter()))}
    return doc


# ---------------------------------------------------------------------------------------------------------- the manifest and the CLI


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_of(agent_dir: Path, agent_root: Path, plan: Plan, *, config: str, judge: Path | None, lib: Path | None, runs: list[dict],
                tree: str | None) -> dict:
    """The manifest of a label file: the policy shas, the seeds' namespace, K, the engine build and the runs the decisions come from."""
    import gen_loop  # noqa: PLC0415

    agent = gen_loop.load_agent(agent_dir)
    mj = Path(agent_dir) / "model.json"
    meta = json.loads(mj.read_text(encoding="utf-8")) if mj.is_file() else {}
    try:
        rel = Path(agent_dir).resolve().relative_to(Path(agent_root).resolve()).as_posix()
    except ValueError:
        rel = Path(agent_dir).name
    dll = Path(lib) / "Sts2Emulator.dll" if lib else None
    return {
        "version": LABEL_VERSION, "created_utc": utc().strftime("%Y-%m-%dT%H:%M:%SZ"), "kind": "hindsight-labels",
        "unit": "floors (the run head's own unit); G = death floor | the frozen head's expected final floor at the next act | won-run floor",
        "policy": {
            "exam_agent": {"dir": rel, "generation": agent.generation, "prior_sha256": meta.get("policy_sha256"), "config": config,
                           "mask_version": agent.mask_version},
            "run_head": {"file": "run_value.npz", "sha256": file_sha256(agent.macro_npz) if agent.macro_npz else None, "W": agent.W},
            "continuation": {"tau": 0.0, "eps": 0.0, "note": "argmax, no exploration"},
            "judge": None if judge is None else {"file": Path(judge).name, "sha256": file_sha256(judge)},
        },
        "seeds": {"namespace": NAMESPACE, "agent_seed_of_the_replay": AGENT_SEED,
                  "continuation": "resample seed and search streams from (run id, floor, decision kind, k) only; the same for every option"},
        "engine": {"lib_sha256": file_sha256(dll) if dll is not None and dll.is_file() else None},
        "plan": {"k": plan.k, "quota": plan.quota, "per_run": plan.per_run, "rates": plan.rates, "max_options": plan.max_options,
                 "acts": list(plan.acts), "horizon": "the end of the decision's act, its boss included"},
        "source": {"kind": "the AI's own exploring runs (lab self-play, replayed on the copy and checked against their records)",
                   "runs_available": len(runs), "jobs": sorted({r["_job"] for r in runs}),
                   "human_corpus": "not read (D38: the exam fifth is never a label source; brief 2b is the human part)"},
        "tree": tree,
    }


def _git_head() -> str | None:
    import subprocess  # noqa: PLC0415

    with contextlib.suppress(Exception):
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip() or None
    return None


def parse_quota(text: str) -> dict[str, int]:
    out = {}
    for part in filter(None, (text or "").split(",")):
        cell, _, n = part.partition("=")
        out[cell.strip()] = int(n)
    return out


def until_today(hhmm: str) -> dt.datetime | None:
    """Today's UTC time `HH:MM` as a datetime (None for ""): the pilot stops by then."""
    if not hhmm:
        return None
    h, m = (int(x) for x in hhmm.split(":"))
    return utc().replace(hour=h, minute=m, second=0, microsecond=0)


_BEAT = [0.0]


def _beat(every_s: float = 900.0) -> bool:
    now = time.time()
    if now - _BEAT[0] > every_s:
        _BEAT[0] = now
        return True
    return False


def pilot_cmd(a: argparse.Namespace) -> int:
    why = blocked()
    if why:
        print(f"hindsight: not starting ({why})", file=sys.stderr)
        return 2
    until = until_today(a.until)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    agent_root = Path(a.agent_root)
    agent_dir = agent_root / a.agent_dir
    runs = exploring_runs(Path(a.runs_root), a.generation)
    quota = parse_quota(a.quota) or ({f"{k}/a{act + 1}": a.quota_per_cell for k in KINDS for act in range(MAX_ACT_INDEX + 1)}
                                     if a.quota_per_cell else {})
    plan = Plan(k=a.k, quota=quota, per_run=a.per_run, max_options=a.max_options)
    order = sorted(runs, key=lambda r: _digest("order", a.order_seed, r["seed"], r["ix"]))
    if a.limit_runs:
        order = order[: a.limit_runs]
    judge = Path(a.judge) if a.judge else None
    lib = Path(a.lib) if a.lib else None
    mf = out / "manifest.json"
    if not mf.is_file():
        mf.write_text(json.dumps(manifest_of(agent_dir, agent_root, plan, config=a.config, judge=judge, lib=lib, runs=runs,
                                             tree=_git_head()), indent=1), encoding="utf-8")
    import fight_gym  # noqa: PLC0415 -- its supervised pool (deadlines, a stop poll); its own guards are not used

    tf = out / "tasks.jsonl"
    done = {tuple(json.loads(x)["_task"]) for x in tf.read_text(encoding="utf-8").splitlines() if x.strip()} if tf.is_file() else set()
    tasks = [("hs", r["seed"], r["ix"], r) for r in order if ("hs", r["seed"], r["ix"]) not in done]

    def stop() -> str | None:
        b = blocked()
        if b:
            return b
        if until is not None and utc() >= until:
            return "deadline"
        if quota and not plan.open_cells(count_cells(read_labels(out))):
            return "quota-filled"
        return None

    print(f"hindsight: {len(tasks)} runs to replay, K {plan.k}, quota {quota or 'none'}, {a.procs} processes", flush=True)
    if not a.no_hold:
        hold(True)
    try:
        reason = fight_gym.run_tasks(tasks, run_task, tf, a.procs, hard_of=lambda t: a.task_cap_s, capped_of=capped_row, init=_init_worker,
                                     init_args=(str(out), str(agent_dir), a.config, str(judge) if judge else "", bool(a.doors),
                                                dataclasses.asdict(plan), str(lib) if lib else ""),
                                     stop=stop, refresh=lambda: hold(True) if (_beat() and not a.no_hold) else None)
    finally:
        if not a.no_hold:
            hold(False)
    print(f"hindsight: stopped ({reason or 'all runs replayed'})", flush=True)
    (out / "report.json").write_text(json.dumps(report(out), indent=1), encoding="utf-8")
    return 0


def report_cmd(a: argparse.Namespace) -> int:
    doc = report(Path(a.out))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "report.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps(doc, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("pilot", help="label decisions of the exploring runs (resumable; stops for the night and for any QUIET line)")
    pl.add_argument("--out", required=True)
    pl.add_argument("--agent-root", default=str(REF / "agent"))
    pl.add_argument("--agent-dir", default="gen11/candidate")
    pl.add_argument("--generation", type=int, default=11)
    pl.add_argument("--runs-root", default=str(REF / "LAB" / "runs"))
    pl.add_argument("--config", default="a3-v1")
    pl.add_argument("--k", type=int, default=DEFAULT_K)
    pl.add_argument("--procs", type=int, default=6)
    pl.add_argument("--per-run", type=int, default=2)
    pl.add_argument("--max-options", type=int, default=MAX_OPTIONS)
    pl.add_argument("--quota", default="", help="cells wanted, e.g. card/a1=50,rest/a2=50")
    pl.add_argument("--quota-per-cell", type=int, default=0, help="every kind x act cell (Acts 1 and 2) wants this many decisions")
    pl.add_argument("--limit-runs", type=int, default=0)
    pl.add_argument("--order-seed", type=int, default=1)
    pl.add_argument("--task-cap-s", type=float, default=3600.0)
    pl.add_argument("--until", default="", help="UTC HH:MM today: stop there (workers are killed; finished decisions are kept)")
    pl.add_argument("--judge", default="", help="the fight judge's checkpoint: its estimate of the boss fight is recorded at each boss door")
    pl.add_argument("--doors", action="store_true", help="keep the combat snapshot at each boss door (large)")
    pl.add_argument("--lib", default="", help="the native build the exploring runs were played with (STS2_LIB_PATH)")
    pl.add_argument("--no-hold", action="store_true", help="do not write the QUIET.lock line (a caller that holds one for the whole session)")
    pl.set_defaults(fn=pilot_cmd)
    rp = sub.add_parser("report", help="the aggregates of the label files under --out")
    rp.add_argument("--out", required=True)
    rp.set_defaults(fn=report_cmd)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    # through the module name, so the pool's workers unpickle `hindsight.run_task`, not `__main__.run_task`
    import hindsight as _self  # noqa: PLC0415

    raise SystemExit(_self.main())
