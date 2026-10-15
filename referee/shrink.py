#!/usr/bin/env python3
"""Shrink a diverging setup fight to a minimal setup: delta debugging over its setup items.

A fight whose first divergence is at step d, field f (tools/fidelity/reanchor_compare.group) is played
again in the game in script mode: the recorded actions up to step d, matched by card id
(referee/driver/Setup.cs "script"), with a subset of its setup items (the "fight" item always stays,
deck_add entries count as items too). Each result is replayed in the emulator; the subset still fails
when the first divergence is again at step d in field f. A scripted action that is not legal any more,
a fight that ends sooner or an error leaves the subset unresolved, which counts as not failing.

1. Reproduce: the whole setup in script mode. A fight that does not diverge again is flaky; skipped.
2. ddmin, divide and conquer: halves tested together, keeping one that still fails; when neither
   does alone, each half is reduced with the other kept (about 2k log2(n) fights for k culprits).
3. A last pass tries dropping each remaining item once.

Each round sends the open subsets of every case to the game at once (one serve.py call). A case stops
at --cap fights (48), the whole run at the budget or at `deadline` (a time.time() after which no round starts);
a stopped case keeps the smallest subset that failed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Generator, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

Items = list  # item handles, e.g. ("setup", 3) or ("deck_add", 0)


def _split(c: list, n: int) -> list[list]:
    size, rest = divmod(len(c), n)
    out, at = [], 0
    for i in range(n):
        k = size + (1 if i < rest else 0)
        out.append(c[at:at + k])
        at += k
    return [x for x in out if x]


def _minus(c: list, drop: Iterable) -> list:
    d = set(drop)
    return [x for x in c if x not in d]


def ddmin(items: list) -> Generator[list[list], list, list]:
    """Delta debugging as a generator: yields the subsets to test this round, receives one outcome per
    subset (True = still fails), returns a minimal failing subset in the input's order. The input itself
    is assumed to fail. Halves are tested together; when neither half fails alone, the first half is
    reduced with the whole second half kept, then the second with what the first needs (the divide and
    conquer form of ddmin), so k culprits among n items cost about 2k log2(n) fights."""
    order = {x: i for i, x in enumerate(items)}
    m = yield from _reduce([], list(items))
    # Last pass: drop each item once (with one item left, this asks whether it is needed at all).
    tests = [_minus(m, [x]) for x in m]
    if tests:
        res = yield tests
        drop = [x for x, ok in zip(m, res) if ok]
        if len(drop) > 1:
            together = _minus(m, drop)
            m = together if (yield [together])[0] else _minus(m, drop[:1])
        elif drop:
            m = _minus(m, drop)
    return sorted(m, key=order.__getitem__)


def _reduce(background: list, c: list) -> Generator[list[list], list, list]:
    """A minimal part M of c such that background + M still fails, given that background + c fails."""
    if len(c) <= 1:
        return c  # the last pass checks it is needed
    c1, c2 = _split(c, 2)
    r1, r2 = yield [background + c1, background + c2]
    if r1:
        return (yield from _reduce(background, c1))
    if r2:
        return (yield from _reduce(background, c2))
    m1 = yield from _reduce(background + c2, c1)
    m2 = yield from _reduce(background + m1, c2)
    return m1 + m2


class Case:
    """One diverging fight being shrunk."""

    def __init__(self, spec: dict, game: dict, step: int, field: str, cap: int = 48):
        self.spec, self.step, self.field, self.cap = spec, step, field, cap
        self.seed = spec["seed"]
        self.script = [s["action"] for s in game["steps"][1:step + 1]]
        self.items: Items = [("setup", i) for i, it in enumerate(spec.get("setup", [])) if it.get("cmd") != "fight"]
        self.items += [("deck_add", j) for j in range(len(spec.get("deck_add", [])))]
        self.spent = 0
        self.status = "reproducing"
        self.best: Items = list(self.items)
        self._gen = ddmin(self.items)
        self._pending: list[Items] = [list(self.items)]  # reproduce first

    @property
    def pending(self) -> list[Items]:
        return self._pending if self.status in ("reproducing", "shrinking") else []

    def subset_spec(self, keep: Items) -> dict:
        k = set(keep)
        s = {key: v for key, v in self.spec.items() if key not in ("setup", "deck_add", "trace_setup")}
        s["setup"] = [it for i, it in enumerate(self.spec.get("setup", [])) if it.get("cmd") == "fight" or ("setup", i) in k]
        if self.spec.get("deck_add"):
            s["deck_add"] = [x for j, x in enumerate(self.spec["deck_add"]) if ("deck_add", j) in k]
        s["script"] = self.script
        s["max_actions"] = len(self.script)
        s["_expect"] = {"step": self.step, "field": self.field}
        return s

    def feed(self, outcomes: list) -> None:
        self.spent += len(outcomes)
        fails = [o is True for o in outcomes]
        if self.status == "reproducing":
            if not fails[0]:
                self.status = "flaky" if outcomes[0] is False else "unresolved"
                return
            self.status = "shrinking"
            try:
                self._pending = next(self._gen)
            except StopIteration as stop:  # nothing to shrink
                self.best = list(stop.value)
                self.status = "minimal"
            return
        for subset, ok in zip(self._pending, fails):
            if ok and len(subset) < len(self.best):
                self.best = list(subset)
        try:
            self._pending = self._gen.send(fails)
        except StopIteration as stop:
            self.best = list(stop.value)
            self.status = "minimal"

    def result(self) -> dict:
        items = [self.spec["setup"][i] if kind == "setup" else {"deck_add": self.spec["deck_add"][i]} for kind, i in self.best]
        return {"seed": self.seed, "status": self.status, "fights": self.spent, "step": self.step, "field": self.field,
                "items": items, "of": len(self.items), "spec": {k: v for k, v in self.subset_spec(self.best).items() if k != "_expect"}}


def shrink_all(cases: list[Case], run_round: Callable[[str, list], list], name: str, budget: int = 400,
               deadline: float | None = None) -> tuple[dict, dict]:
    """Shrinks every case, one game round for all of them at a time. run_round(round name, specs) gives one
    outcome per spec: True (still fails), False or None (unresolved)."""
    spent, rnd, late = 0, 0, False
    while True:
        batch = []
        late = late or (deadline is not None and time.time() >= deadline)
        for c in cases:
            todo = c.pending
            if not todo:
                continue
            if late or c.spent + len(todo) > c.cap or spent + len(todo) > budget:
                c.status = "capped" if c.status == "shrinking" else "unresolved"
                continue
            batch.append((c, todo))
            spent += len(todo)
        if not batch:
            break
        specs = [c.subset_spec(t) for c, todo in batch for t in todo]
        outcomes = run_round(f"{name}.shrink{rnd}", specs)
        at = 0
        for c, todo in batch:
            c.feed(outcomes[at:at + len(todo)])
            at += len(todo)
        rnd += 1
    minimal = {c.seed: c.result() for c in cases if c.status in ("minimal", "capped")}
    lines = [f"{c.seed}: {c.status}, {len(c.best)} of {len(c.items)} items in {c.spent} game fights (step {c.step}, "
             f"{c.field}): {[(it.get('cmd'), it.get('id')) for it in c.result()['items']]}" for c in cases]
    done = [c for c in cases if c.status == "minimal"]
    summary = (f"{len(cases)} cases, {len(done)} minimal, {sum(c.status == 'capped' for c in cases)} capped, "
               f"{sum(c.status == 'flaky' for c in cases)} flaky; {spent} game fights in {rnd} rounds"
               + (f", {sum(c.spent for c in done) / len(done):.1f} per minimal case" if done else "")
               + (", stopped at the time limit" if late and any(c.status in ("capped", "unresolved") for c in cases) else ""))
    return minimal, {"summary": summary, "lines": lines, "fights": spent, "rounds": rnd}


def outcome(game: dict, emu: dict, step: int, field: str, nm: dict) -> bool | None:
    """Whether a scripted re-run still first diverges at `step` in `field`; None when it could not be judged."""
    from scenarios import first_divergence  # noqa: PLC0415
    if game.get("error") or emu.get("error") or game.get("script_illegal") or len(game.get("steps", [])) < step + 1:
        return None
    fd = first_divergence(game, emu, nm)
    return fd is not None and fd[0] == step and fd[1] == field


def cases_from(specs: list, games: list, verdicts: list, k: int, cap: int = 48) -> list[Case]:
    """Up to k diverging fights to shrink: no error, no mirror gap, plausible ones first, earliest step first."""
    pick = [(s, g, v) for s, g, v in zip(specs, games, verdicts)
            if not v.get("error") and not v.get("identical") and not v.get("mirror_gap") and "step" in v]
    pick.sort(key=lambda t: (bool(t[2]["flags"]), t[2]["step"]))
    return [Case(s, g, v["step"], v["field"], cap) for s, g, v in pick[:k]]


def oracle(play: Callable[[str, list], list], replay: Callable[[list], list], nm: dict) -> Callable[[str, list], list]:
    """run_round for shrink_all: play the specs in the game, replay them in the emulator, judge each."""
    def run_round(name: str, specs: list) -> list:
        games = play(name, specs)
        emus = replay(games)
        return [outcome(g, e, s["_expect"]["step"], s["_expect"]["field"], nm) for s, g, e in zip(specs, games, emus)]
    return run_round


if __name__ == "__main__":
    print(__doc__)
