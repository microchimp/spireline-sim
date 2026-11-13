"""Block P2 of the information-parity side blocks: what role each card plays in a deck (D59 lane D, `docs/rl/HUMAN_PRIOR.md` "v4").

A human reads a deck as roles: how much area damage it has, whether it scales (Strength, Dexterity), how much it draws, whether it has block it can repeat, what it
exhausts. The run encoder holds a deck as the count of each card id (block E) and 25 aggregates (block D); the roles are left to be learned per card. This module
gives every card id (and its upgraded form) a **role vector** and the deck's role counts, for the deck and for each option's card:

    multi_target       hits every enemy (the card table's target) and deals damage
    scaling_damage     a following Strike hits harder, this turn or at the next turn's start, because this card was played (measured by playing it:
                       `rl/card_tags.py`'s rule, and the same fight carried one turn on for the powers that act at a turn's start)
    scaling_block      a following Defend blocks more (measured, the same two ways)
    draw               cards drawn beyond the one that left the hand (measured)
    energy             energy gained beyond the cost (measured)
    repeatable_block   at least 7 block per energy and does not exhaust (printed, upgrade-aware): a block a deck can make every turn
    exhaust            exhausts when played (printed, upgrade-aware)
    power              is a Power
    junk               a Curse or Status card

`rl/parity_data/card_probe.json` holds the measured part, written by `python rl/parity_cards.py probe` (every card of the engine's table played once on the engine,
scaling and the energy test with a four-energy opening for the cards a three-energy turn cannot test). A side block: `BASE_SIZE` and the feature version are untouched.
"""

from __future__ import annotations

import json
import re
import sys
from functools import cache
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ROLES = ("multi_target", "scaling_damage", "scaling_block", "draw", "energy", "repeatable_block", "exhaust", "power", "junk")
N_ROLES = len(ROLES)
PROBE_FILE = Path(__file__).resolve().parent / "parity_data" / "card_probe.json"
CARDS_G = _ROOT / "src" / "Sts2Emulator" / "Generated" / "Cards.g.cs"
# the deck's role counts are divided by this (then clipped to 1); the size by DECK_SIZE_SCALE
COUNT_SCALE = 6.0
DECK_SIZE_SCALE = 40.0
REPEATABLE_BLOCK_PER_ENERGY = 7
EXTRA_ENERGY_RELIC = 283  # VeryHotCocoa: four more energy on the first turn, so a card that costs three can be tested with its follow-up


@cache
def card_targets() -> dict[int, str]:
    """card id -> its target in the engine's card table (`Self`, `AnyEnemy`, `AllEnemies`, ...)."""
    text = CARDS_G.read_text(encoding="utf-8")
    out: dict[int, str] = {}
    for m in re.finditer(r"new CardDef\(Id: (\d+),[^\n]*?Target: CardTarget\.(\w+)", text):
        out[int(m.group(1))] = m.group(2)
    return out


@cache
def probe_table() -> dict[int, dict]:
    """card id -> the measured tags of `rl/parity_data/card_probe.json` (front_damage, front_block, scaling_damage, scaling_block, draw, energy, playable)."""
    raw = json.loads(PROBE_FILE.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def card_roles(card_id: int, upgraded: bool = False) -> np.ndarray:
    """The role vector of one card, float32 (N_ROLES,), 0/1 per role. A card the probe table lacks has no measured roles (the printed ones still apply)."""
    from run_train import vocab  # noqa: PLC0415

    return _roles_cached(int(card_id), bool(upgraded)).copy()


@cache
def _roles_cached(card_id: int, upgraded: bool) -> np.ndarray:
    from run_train import vocab  # noqa: PLC0415

    v = np.zeros(N_ROLES, dtype=np.float32)
    try:
        facts = vocab.card(card_id)
    except KeyError:
        return v
    probe = probe_table().get(card_id, {})
    damage = facts.damage_for(upgraded)
    block = facts.block_for(upgraded)
    cost = max(0, facts.cost_for(upgraded))
    exhaust = facts.is_exhaust(upgraded)
    v[ROLES.index("multi_target")] = float(card_targets().get(card_id) == "AllEnemies" and damage > 0)
    v[ROLES.index("scaling_damage")] = float(bool(probe.get("scaling_damage")) or bool(probe.get("scaling_damage_next_turn")))
    v[ROLES.index("scaling_block")] = float(bool(probe.get("scaling_block")) or bool(probe.get("scaling_block_next_turn")))
    v[ROLES.index("draw")] = float(bool(probe.get("draw")))
    v[ROLES.index("energy")] = float(bool(probe.get("energy")))
    v[ROLES.index("repeatable_block")] = float(block > 0 and not exhaust and facts.card_type != "Power" and block / max(1, cost) >= REPEATABLE_BLOCK_PER_ENERGY)
    v[ROLES.index("exhaust")] = float(exhaust)
    v[ROLES.index("power")] = float(facts.card_type == "Power")
    v[ROLES.index("junk")] = float(facts.card_type in ("Curse", "Status"))
    v.setflags(write=False)
    return v


def deck_roles(deck) -> np.ndarray:  # noqa: ANN001
    """The role counts of a deck of `CardRef`s: N_ROLES counts over COUNT_SCALE (clipped to 1), then the deck's size over DECK_SIZE_SCALE. float32 (N_ROLES + 1,)."""
    total = np.zeros(N_ROLES, dtype=np.float64)
    n = 0
    for card in deck:
        total += _roles_cached(int(card.card_id), bool(card.upgraded))
        n += 1
    out = np.zeros(N_ROLES + 1, dtype=np.float32)
    out[:N_ROLES] = np.minimum(1.0, total / COUNT_SCALE)
    out[N_ROLES] = min(1.0, n / DECK_SIZE_SCALE)
    return out


def cards_roles(cards) -> np.ndarray:  # noqa: ANN001
    """The summed role vector of a few cards (an option's card or cards), over COUNT_SCALE's third, clipped to 1: float32 (N_ROLES,)."""
    total = np.zeros(N_ROLES, dtype=np.float64)
    for card in cards:
        total += _roles_cached(int(card.card_id), bool(card.upgraded))
    return np.minimum(1.0, total / 2.0).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------- the probe (writes card_probe.json)


def probe_all() -> dict[int, dict]:
    """Play every card of the engine's table once and record what `rl/card_tags.py` measures; a card whose scaling the three-energy turn cannot test is measured again
    with a four-energy opening. Returns card id -> row (also what `main` writes)."""
    import card_tags as ct  # noqa: PLC0415
    from run_train import vocab  # noqa: PLC0415

    rows = {cid: ct.probe(cid) for cid in vocab.card_ids()}
    saved_fresh, saved_energy = ct.fresh, ct.BASE_ENERGY
    try:
        ct.fresh = _fresh
        ct.BASE_ENERGY = saved_energy + 4
        for cid, row in rows.items():
            if row.get("playable") and (row.get("scaling_damage") is None or row.get("scaling_block") is None):
                again = ct.probe(cid)
                for tag in ("scaling_damage", "scaling_block"):
                    if row.get(tag) is None:
                        row[tag] = again.get(tag)
                row["probed_with_extra_energy"] = True
            if row.get("playable"):
                for tag, follow in (("scaling_damage_next_turn", ct.STRIKE), ("scaling_block_next_turn", ct.DEFEND)):
                    row[tag] = next_turn_effect(ct, cid, follow)
    finally:
        ct.fresh, ct.BASE_ENERGY = saved_fresh, saved_energy
    for cid, row in rows.items():
        row["target"] = card_targets().get(cid, "?")
    return rows


def _fresh():  # noqa: ANN202
    """The probe's fight with the extra-energy relic on its first turn (the card-tag probe's one enemy, 132 HP, no block)."""
    import card_tags as ct  # noqa: PLC0415
    from combat_mcts import Combat  # noqa: PLC0415

    return Combat.opening(ct.PROBE_SEED, ct.PROBE_ENCOUNTER, 8, 0, relics=[EXTRA_ENERGY_RELIC])


def next_turn_effect(ct, card_id: int, follow: int) -> int | None:  # noqa: ANN001
    """How much more a `follow` card (a Strike or a Defend) does on the NEXT turn because `card_id` was played on this one: the same fight twice, the card played or not,
    both ending the turn, then the follow-up played at the second turn's start. Catches the powers that act at a turn's start (Demon Form, Barricade). None when the
    trial cannot be run (the card cannot be played, or the follow-up cannot)."""
    from sts2_gym import native  # noqa: PLC0415

    results = []
    for play_first in (True, False):
        combat = _fresh()
        try:
            if play_first:
                native.debug_add_card_to_hand(combat.handle, card_id, combat.obs, False)
                if ct.play_measured(combat, card_id) is None:
                    return None
            hand = len(native.get_pile(combat.handle, "hand"))
            if combat.step(hand):  # end the turn (End Turn's index is the hand's size)
                return None
            native.debug_add_card_to_hand(combat.handle, follow, combat.obs, False)
            outcome = ct.play_measured(combat, follow)
            if outcome is None:
                return None
            results.append(outcome["damage"] if follow == ct.STRIKE else outcome["block"])
        finally:
            combat.close()
    return max(0, results[0] - results[1])


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "show"])
    a = ap.parse_args()
    if a.cmd == "probe":
        rows = probe_all()
        PROBE_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROBE_FILE.write_text(json.dumps({str(k): v for k, v in sorted(rows.items())}, indent=0, sort_keys=True), encoding="utf-8")
        untested = sum(1 for r in rows.values() if r.get("playable") and (r.get("scaling_damage") is None or r.get("scaling_block") is None))
        print(f"wrote {PROBE_FILE.name}: {len(rows)} cards, {sum(1 for r in rows.values() if r.get('playable'))} playable on turn one, {untested} with scaling still untested")
    else:
        for cid, row in sorted(probe_table().items()):
            print(cid, row.get("name"), {t: row.get(t) for t in ("front_damage", "front_block", "scaling_damage", "scaling_block", "draw", "energy", "playable")})


if __name__ == "__main__":
    main()
