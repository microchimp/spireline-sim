"""Card coverage for referee batches: which cards are verified, which are suspect, which are untested.

A fight verifies every card played before its first divergence; the card whose play diverged
becomes suspect. Suspects are quarantined (left out of new decks) until the emulator changes,
so a batch spends its fights on cards it has not learned about yet instead of re-finding a
known bug. State lives in a local JSON file (not committed: it holds game card ids only).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .fight_compare import diff, emu_view, game_view

VERIFIED_ENOUGH = 5  # identical plays after which a card is drawn rarely


def first_divergence(game: dict, emu: dict, nm: dict) -> int | None:
    """Index of the first step whose states differ, or None if every compared step matches."""
    for i, (gs, es) in enumerate(zip(game.get("steps", []), emu.get("steps", []))):
        if diff(game_view(gs["state"]), emu_view(es["state"], nm)):
            return i
    return None


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"cards": {}}


def save(path: Path, cov: dict) -> None:
    path.write_text(json.dumps(cov, indent=1, sort_keys=True), encoding="utf-8")


def update(cov: dict, games: list, emus: list, nm: dict, commit: str) -> dict:
    """Fold one batch's results in; returns counts of what changed."""
    cards = cov.setdefault("cards", {})
    new_suspects = 0
    for g, e in zip(games, emus):
        if g.get("error") or e.get("error") or not g.get("steps"):
            continue
        idx = first_divergence(g, e, nm)
        upto = len(g["steps"]) if idx is None else idx
        for st in g["steps"][1:upto]:
            card = (st.get("action") or {}).get("card")
            if card:
                c = cards.setdefault(card, {})
                c["verified"] = c.get("verified", 0) + 1
        if idx is not None and idx > 0:
            card = (g["steps"][idx].get("action") or {}).get("card")
            if card:
                c = cards.setdefault(card, {})
                if c.get("suspect_commit") != commit:
                    new_suspects += 1
                c["suspect_commit"] = commit
                c["suspect_seed"] = g.get("seed")
                c["suspect_count"] = c.get("suspect_count", 0) + 1 if c.get("suspect_commit") == commit else 1
    return {"new_suspects": new_suspects}


def weight(card: str, cov: dict, commit: str) -> float:
    c = cov.get("cards", {}).get(card, {})
    if c.get("suspect_commit") == commit:
        return 0.0  # quarantined until the emulator changes
    if c.get("suspect_commit"):
        return 2.0  # suspect under an older emulator: recheck soon
    v = c.get("verified", 0)
    return 3.0 if v == 0 else (1.0 if v < VERIFIED_ENOUGH else 0.15)


def pick(cov: dict, candidates: list, k: int, rng: random.Random, commit: str) -> list:
    """k distinct cards, weighted towards untested and away from quarantined ones."""
    pool = [(c, weight(c, cov, commit)) for c in candidates]
    pool = [(c, w) for c, w in pool if w > 0]
    chosen = []
    for _ in range(min(k, len(pool))):
        total = sum(w for _, w in pool)
        x = rng.random() * total
        for j, (c, w) in enumerate(pool):
            x -= w
            if x <= 0:
                chosen.append(c)
                pool.pop(j)
                break
    return chosen


def summary(cov: dict, candidates: list, commit: str) -> str:
    cards = cov.get("cards", {})
    untested = sum(1 for c in candidates if not cards.get(c, {}).get("verified") and not cards.get(c, {}).get("suspect_commit"))
    verified = sum(1 for c in candidates if cards.get(c, {}).get("verified", 0) >= VERIFIED_ENOUGH)
    suspects = sorted(c for c in candidates if cards.get(c, {}).get("suspect_commit") == commit)
    return (f"coverage: {len(candidates)} cards, {verified} verified >= {VERIFIED_ENOUGH} plays, {untested} untested, "
            f"{len(suspects)} quarantined: {', '.join(suspects[:12])}{' ...' if len(suspects) > 12 else ''}")
