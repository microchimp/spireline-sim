"""The information-parity side blocks of an option's row, and the file that stores them (D59 lane D, `docs/rl/HUMAN_PRIOR.md` "v4").

The brief: at every decision the player's inputs hold what the game shows a human there, plus what humans derive. The run row (`BASE_SIZE` 1855 and the fight block H) holds
the deck as card ids, the relics as ids and an event option as a slot; block M (the act map) and block G (relic counters, "?" odds, removals) are side blocks the prior and
the student can take. This module builds the next side blocks, per option and from its afterstate, without a change to the base vector or the feature version:

    relics   P1: the effect tags of the relics held after the option (a sum), then the tags of the option's own relic         (`rl/parity_relics.py`)
    roles    P2: the deck's role counts after the option and its size, then the roles of the option's card or cards            (`rl/parity_cards.py`)
    events   P3: what the option pays, for an event option: known, the one-step changes of HP, max HP, gold, deck, relics, curses, potions and where it leads
                                                                                                                                (`rl/parity_events.py`)
    potions  P4: the effect tags of the potions held after the option (a sum), then the tags of the option's own potion                  (`rl/parity_potions.py`)

`option_parity(view, options)` returns one float32 array per block, `(len(options), width)`, row i for option i (the same afterstates `human_prior_infer.option_rows` encodes).
A shard stores them as `rows_parity.npz` beside `rows.npz` (one array per block, aligned with `rows`), its manifest saying `parity: true`, `parity_version` and `parity_blocks`.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import parity_cards as pc  # noqa: E402
import parity_events as pe  # noqa: E402
import parity_potions as pp  # noqa: E402
import parity_relics as pr  # noqa: E402

PARITY_VERSION = 1
SHARD_ROWS_PARITY = "rows_parity.npz"
BLOCK_NAMES = ("relics", "roles", "events", "potions")
WIDTHS = {"relics": 2 * pr.RELIC_TAG_WIDTH, "roles": (pc.N_ROLES + 1) + pc.N_ROLES, "events": pe.EVENT_WIDTH, "potions": 2 * pp.POTION_TAG_WIDTH}


def afterstate(view, option):  # noqa: ANN001, ANN201
    """The state after `option` as the run encoder computes it (`features.apply_option`); for the options it has no afterstate for, a random transform (the card leaves the deck; its
    replacement is an expectation the roles do not model) or an option the rules policy decides, the best stand-in: the view with the card gone, or the view itself."""
    from run_train import features  # noqa: PLC0415

    try:
        return features.apply_option(view, option)
    except ValueError:
        if option.kind == "select.transform" and option.deck_index is not None and 0 <= option.deck_index < len(view.deck):
            return replace(view, deck=view.deck[: option.deck_index] + view.deck[option.deck_index + 1:])
        return view


def option_cards(view, option):  # noqa: ANN001, ANN201
    """The cards an option takes, gives or touches: its card, a bundle's cards, or the deck card it removes, upgrades, transforms or duplicates."""
    if option.card is not None:
        return (option.card,)
    if option.cards:
        return tuple(option.cards)
    if option.deck_index is not None and 0 <= option.deck_index < len(view.deck):
        return (view.deck[option.deck_index],)
    return ()


def option_parity(view, options) -> dict[str, np.ndarray]:  # noqa: ANN001
    """The parity blocks of each option's afterstate: {block name: float32 (n, width)}."""
    n = len(options)
    out = {name: np.zeros((n, WIDTHS[name]), dtype=np.float32) for name in BLOCK_NAMES}
    for i, option in enumerate(options):
        after = afterstate(view, option)
        w = pr.RELIC_TAG_WIDTH
        out["relics"][i, :w] = pr.held_tags(after.relics)
        if option.relic is not None:
            out["relics"][i, w:] = pr.relic_tags(option.relic)
        out["roles"][i, : pc.N_ROLES + 1] = pc.deck_roles(after.deck)
        out["roles"][i, pc.N_ROLES + 1:] = pc.cards_roles(option_cards(view, option))
        if option.kind == "event.option" and option.event is not None:
            out["events"][i] = pe.event_payoff(option.event)
        out["potions"][i, : pp.POTION_TAG_WIDTH] = pp.held_tags(after.potions)
        if option.potion is not None:
            out["potions"][i, pp.POTION_TAG_WIDTH:] = pp.potion_tags(option.potion)
    return out


# ---------------------------------------------------------------------------------------------------------- the held parts from a base vector


def _count_maps():  # noqa: ANN202
    """The three matrices that take a base vector's counts to the held parts: cards (E: 2 x 577 columns, base and upgraded) to role counts, relics (F) to relic tags, potions (G) to potion tags."""
    from run_train import features, vocab  # noqa: PLC0415

    cards = np.zeros((features.E_SIZE, pc.N_ROLES), dtype=np.float32)
    for rank, card_id in enumerate(vocab.card_ids()):
        for up in (0, 1):
            cards[2 * rank + up] = pc.card_roles(card_id, bool(up))
    relics = np.stack([pr.relic_tags(rid) for rid in vocab.relic_ids()]).astype(np.float32)
    potions = np.stack([pp.potion_tags(pid) for pid in vocab.potion_ids()]).astype(np.float32)
    return cards, relics, potions


_MAPS: tuple | None = None


def held_from_base(base: np.ndarray) -> dict[str, np.ndarray]:
    """The held halves of P1, P2 and P4 of rows' afterstates from their base vectors alone (float32 (n, 1855+), the first 1,855 columns read): `relics` (n, RELIC_TAG_WIDTH), `roles` (n,
    N_ROLES + 1: the deck's role counts and its size) and `potions` (n, POTION_TAG_WIDTH). They are linear in the counts of block E (the deck), F (relics) and G (potions), so a row stored as counts
    (the human shards) needs no view. Equal to `option_parity`'s held halves, except for a random transform, whose base vector holds the expectation over the pool (a fraction of each outcome
    card) where `option_parity` takes the card out of the deck."""
    global _MAPS  # noqa: PLW0603
    from run_train import features  # noqa: PLC0415

    if _MAPS is None:
        _MAPS = _count_maps()
    cards, relics, potions = _MAPS
    x = np.asarray(base, dtype=np.float32)
    e = x[:, features.E_OFFSET: features.E_OFFSET + features.E_SIZE]
    f = x[:, features.F_OFFSET: features.F_OFFSET + features.F_SIZE]
    g = x[:, features.G_OFFSET: features.G_OFFSET + features.G_SIZE]
    roles = np.zeros((len(x), pc.N_ROLES + 1), dtype=np.float32)
    roles[:, : pc.N_ROLES] = np.minimum(1.0, (e @ cards) / pc.COUNT_SCALE)
    roles[:, pc.N_ROLES] = np.minimum(1.0, e.sum(axis=1) / pc.DECK_SIZE_SCALE)
    return {"relics": (f @ relics).astype(np.float32), "roles": roles, "potions": (g @ potions).astype(np.float32)}
