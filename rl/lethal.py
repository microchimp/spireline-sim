"""A kill check at the root of the fight search (2026-11-04).

The search spends its whole simulation budget at every decision, even when this turn's cards kill every enemy. `find_lethal`
looks for that first: it enumerates the playable sequences of this turn on a copy of the position and, when one ends the fight
(every enemy dead, the player alive), names it. The caller plays that sequence's first move and skips the search. When no kill
is found, within the limits below, the caller searches as it always did: nothing about play changes.

What counts as a kill, and why a wrong one cannot happen silently:

- The enumeration runs on a *resampled* copy of the position (`root.fork(seed)`): the hand, the energy and the enemies are what
  the player sees, the draw pile is reshuffled and the enemies' hidden rolls are drawn again. The check never reads the true
  draw order, so it keeps the search's firewall.
- A sequence is a kill only if stepping the copy through it ends the fight with the player alive (`done` and `won()`).
  End Turn is never part of a sequence: a kill is made with this turn's cards, not by what the enemies do next.
- A kill the enumeration finds is replayed at once, move by move (each move found again by what it plays, not by its index), in
  `verify_worlds` other resampled copies, and counts only if it ends the fight in every one. A kill that depends on a draw or a
  roll the player cannot see fails there and is dropped; the enumeration goes on. With no verified kill the caller searches.
- Among kills, the one with no potion used comes first, then the least HP lost, then the fewest plays.

Limits (a stop at any of them without a verified kill means "not found", never "no kill exists"): `max_sequences` finished
sequences, `max_steps` card plays tried, `max_seconds` of wall time, `max_open` copies held at once (the emulator's handle pool
is shared with the fights a process plays). The sequence and step limits make the answer a function of the position alone; the
time limit is the safety net.

Two orders of the same cards that leave the same visible position (`rules.state_key`, when the rules give one) are explored once:
the cards of a hand come in many orders and most orders end alike, so this is most of the saving. A path pruned that way is
never a missed *verified* kill, only a missed look: every kill reported has been replayed.

The module reaches the position only through `rules` and the world's methods, so it imports no native code and its tests run on
synthetic positions. `rl/live_search.py` supplies the emulator's rules (`_LIVE_RULES`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

MAX_SEQUENCES = 200  # finished sequences (a kill, a death, or a position with nothing left to play but End Turn)
MAX_STEPS = 240  # plays tried (about 45 ms at the emulator's 0.18 ms a play, so the count binds before the clock does and the answer is the position's alone)
MAX_SECONDS = 0.050
MAX_DEPTH = 24  # plays in one sequence
MAX_OPEN = 48  # copies held open at once (one level's children for each level of the sequence)
VERIFY_WORLDS = 3  # other resampled copies a kill must hold up in
VERIFY_CANDIDATES = 6  # kills replayed for verification in one check (the rest are not looked at once one is verified or this is spent)
CHECK_SEED = 0x1E7A1  # the enumeration copy's resample seed; the verification copies take the next ones


class Rules(Protocol):
    """What the enumeration needs to know about a position; the moves are objects with .key, .action, .target and .kind."""

    def moves(self, world: Any) -> list[Any]:
        """The distinct moves in the world, End Turn included (kind "end_turn": the enumeration leaves it out)."""

    def play(self, world: Any, move: Any) -> None:
        """Step the world by the move; the world's `done` is set."""

    def enemy_hp(self, world: Any) -> float:
        """The living enemies' total HP, to try the moves that hurt most first."""

    def match(self, world: Any, move: Any) -> Any | None:
        """The same move in another world of the same position (found by what it plays), or None."""

    # Optional: `state_key(world)`, a hashable of everything the player can see in the world. Two worlds with the same key have the
    # same moves ahead of them, so the enumeration explores the second one no further.


@dataclass(frozen=True)
class Limits:
    max_sequences: int = MAX_SEQUENCES
    max_steps: int = MAX_STEPS
    max_seconds: float = MAX_SECONDS
    max_depth: int = MAX_DEPTH
    max_open: int = MAX_OPEN
    verify_worlds: int = VERIFY_WORLDS
    verify_candidates: int = VERIFY_CANDIDATES
    seed: int = CHECK_SEED


@dataclass
class Kill:
    """A verified kill: the moves in order, the HP the player lost on the enumeration copy, the potions used."""

    path: list[Any]
    hp_lost: float
    potions: int
    verified: int  # the other copies it held up in


@dataclass
class Report:
    kill: Kill | None
    reason: str  # "kill"; "none": every sequence tried, no kill; "cap", "steps", "time", "open": a limit stopped it; "unverified": kills found, none held up
    sequences: int = 0
    steps: int = 0
    kills_seen: int = 0  # kills the enumeration found, verified or not
    rejected: int = 0  # of those, how many failed the verification
    seconds: float = 0.0
    pruned: int = 0  # positions not explored again: the same visible position reached by another order of the cards

    def record(self) -> dict:
        """The decision record: JSON-able, small."""
        out: dict = {"fired": self.kill is not None, "reason": self.reason, "sequences": self.sequences, "steps": self.steps,
                     "kills_seen": self.kills_seen, "rejected": self.rejected, "pruned": self.pruned, "seconds": round(self.seconds, 5)}
        if self.kill is not None:
            out.update(plays=len(self.kill.path), hp_lost=self.kill.hp_lost, potions=self.kill.potions, verified=self.kill.verified,
                       line=[[m.action, m.target, m.kind] for m in self.kill.path])
        return out


@dataclass
class _Found:
    path: list[Any]
    hp_lost: float
    potions: int

    def order(self) -> tuple:
        return (self.potions > 0, self.hp_lost, len(self.path))


@dataclass
class _Run:
    root: Any
    rules: Rules
    limits: Limits
    clock: Callable[[], float]
    deadline: float
    hp0: float = 0.0
    sequences: int = 0
    steps: int = 0
    kills_seen: int = 0
    rejected: int = 0
    replays: int = 0
    open_now: int = 0  # copies held open right now
    pruned: int = 0  # positions not explored again (the same visible position reached by another order)
    seen: set = field(default_factory=set)
    best: _Found | None = None  # the best verified kill so far
    stopped: str | None = None

    def over(self) -> bool:
        if self.stopped is not None:
            return True
        if self.sequences >= self.limits.max_sequences:
            self.stopped = "cap"
        elif self.steps >= self.limits.max_steps:
            self.stopped = "steps"
        elif self.clock() >= self.deadline:
            self.stopped = "time"
        elif self.open_now >= self.limits.max_open:
            self.stopped = "open"
        elif self.best is not None and self.best.potions == 0 and self.best.hp_lost <= 0:
            self.stopped = "kill"  # nothing better can be found: no potion, no HP lost
        return self.stopped is not None

    def consider(self, found: _Found) -> None:
        """A kill the enumeration found: replayed in other resampled copies unless a verified one is at least as good already."""
        self.kills_seen += 1
        if self.best is not None and found.order() >= self.best.order():
            return
        if self.replays >= self.limits.verify_candidates:
            if self.best is None:
                self.stopped = "unverified"  # the replays are spent and none held up: the caller searches
            return
        self.replays += 1
        if all(_holds(self.root, self.rules, found, self.limits.seed + 1 + i) for i in range(self.limits.verify_worlds)):
            self.best = found
        else:
            self.rejected += 1


def _is_end_turn(move: Any) -> bool:
    return move.kind == "end_turn"


def _children(run: _Run, world: Any, moves: list[Any], held: list[tuple[Any, Any]]) -> None:
    """Play each move on its own copy of the world and put (move, copy) in `held`, best first: a kill, then the one that leaves the
    enemies lowest (ties keep the position's order). The copies stay open for the descent; the caller closes whatever is in `held`,
    however this ends."""
    scored: list[tuple[tuple, Any, Any]] = []
    for index, move in enumerate(moves):
        if run.over():
            break
        child = world.fork(None)
        run.open_now += 1
        held.append((move, child))  # held at once: a failure below still finds it there to close
        run.rules.play(child, move)
        run.steps += 1
        won = bool(child.done and child.won())
        scored.append(((not won, 0.0 if won else float(run.rules.enemy_hp(child)), index), move, child))
    scored.sort(key=lambda item: item[0])
    held[:] = [(move, child) for _, move, child in scored]


def _close(run: _Run, children: list[tuple[Any, Any]]) -> None:
    for _, child in children:
        child.close()
        run.open_now -= 1


def _descend(run: _Run, world: Any, path: list[Any], potions: int) -> None:
    moves = [m for m in run.rules.moves(world) if not _is_end_turn(m)]
    if run.best is not None and run.best.potions == 0:  # a kill with no potion beats any kill with one: those paths are not worth the steps
        moves = [m for m in moves if m.kind != "potion"]
    if not moves or len(path) >= run.limits.max_depth:
        run.sequences += 1  # a finished sequence: all that is left to do is End Turn
        return
    state_key = getattr(run.rules, "state_key", None)
    children: list[tuple[Any, Any]] = []
    try:
        _children(run, world, moves, children)
        for move, child in list(children):
            if run.over():
                return
            used = potions + (move.kind == "potion")
            if child.done:
                run.sequences += 1
                if child.won():
                    run.consider(_Found([*path, move], run.hp0 - float(child.hp()), used))
                continue
            if state_key is not None:
                key = state_key(child)
                if key in run.seen:
                    run.pruned += 1
                    continue
                run.seen.add(key)
            _descend(run, child, [*path, move], used)
    finally:
        _close(run, children)


def _holds(root: Any, rules: Rules, found: _Found, seed: int) -> bool:
    """Whether the sequence ends the fight in the resampled copy of `seed`, each move found again there by what it plays."""
    world = root.fork(seed)
    try:
        for move in found.path:
            if world.done:
                break
            here = rules.match(world, move)
            if here is None:
                return False
            rules.play(world, here)
        return bool(world.done and world.won())
    finally:
        world.close()


def find_lethal(root: Any, rules: Rules, limits: Limits | None = None, *, clock: Callable[[], float] = time.perf_counter) -> Report:
    """Look for a verified kill among this turn's sequences from `root` (a combat that is forked, never stepped or closed)."""
    limits = limits or Limits()
    started = clock()
    run = _Run(root, rules, limits, clock, started + limits.max_seconds)
    world = root.fork(limits.seed)
    try:
        run.hp0 = float(world.hp())
        _descend(run, world, [], 0)
    finally:
        world.close()
    report = Report(None, run.stopped or "none", run.sequences, run.steps, run.kills_seen, run.rejected, pruned=run.pruned)
    if run.best is not None:
        report.kill = Kill(run.best.path, run.best.hp_lost, run.best.potions, limits.verify_worlds)
        report.reason = "kill"
    elif run.rejected and report.reason in ("none", "cap", "steps", "time", "open"):
        report.reason = "unverified"  # kills were found, none held up in the other copies
    report.seconds = clock() - started
    return report
