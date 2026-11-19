"""Hand-built Act 2 and Act 3 screens for `test_boss_chain` (a fixed set to compare policies on, choice for choice; not a game and not a record).

Every screen is a (view, decision) pair on a seeded 17-row map: a rest site, a card reward, a shop, a route choice and a 'rest' with a heal and two smith targets, at
a few (act, floor, HP) places from the first Act 2 decision to the last Act 3 one before the first Act 3 boss, and one past it (floor 48). The module reads the
engine's names through the `F` (run_train.features) and `vocab` arguments, so a script can build the same screens with another tree's modules.
"""

from __future__ import annotations

import dataclasses

from route_elite_screens import act_map

# (act index from 0, state floor, hp): the decision places. Floor 48 is past the chain's last floor (47).
PLACES = ((0, 5, 60), (0, 12, 40), (1, 17, 60), (1, 25, 35), (1, 32, 70), (1, 33, 50), (2, 33, 80), (2, 40, 30), (2, 47, 55), (2, 48, 55))
REGIONS = ("OVERGROWTH", "HIVE", "GLORY")
BOSSES = (83, 75, 73)


def view_of(F, vocab, act_index: int, floor: int, hp: int, deck_extra: int = 0):  # noqa: N803
    cards = [F.CardRef(472, False), F.CardRef(131, False), F.CardRef(30, False)] + [F.CardRef(10001, bool(i % 2)) for i in range(deck_extra)]
    region = getattr(vocab, "REGION_" + REGIONS[act_index])
    return F.RunView(ascension=10, act_index=act_index, act_region=region, floor=floor, hp=hp, max_hp=80, gold=140, deck=tuple(cards),
                     relics=(vocab.relic_id_by_name("BurningBlood"),), potions=(), max_potion_slots=3, map_nodes=act_map(1, F), position=(3, 0),
                     boss=BOSSES[act_index])


def rest_screen(F):  # noqa: N803
    return F.Decision("rest", (F.Option("rest.heal", 0), F.Option("rest.smith", 1, deck_index=0), F.Option("rest.smith", 2, deck_index=1)))


def card_screen(F):  # noqa: N803
    return F.Decision("card_reward", (F.Option("card.take", 0, card=F.CardRef(472, False)), F.Option("card.take", 1, card=F.CardRef(131, False)), F.Option("card.skip", 2)))


def shop_screen(F):  # noqa: N803
    return F.Decision("shop", (F.Option("shop.leave", 0), F.Option("shop.card", 1, card=F.CardRef(131, False), price=50),
                               F.Option("shop.card", 2, card=F.CardRef(472, False), price=70)))


def route_screen(F):  # noqa: N803
    nodes = act_map(1, F)
    by = {(n.col, n.row): n for n in nodes}
    here = by[(3, 0)]
    return F.Decision("map", tuple(F.Option(kind="map.node", action=i, node=(c, r, by[(c, r)].node_type)) for i, (c, r) in enumerate(here.children)))


def act23_screens(F, vocab):  # noqa: N803
    """[(view, decision)] of every place in `PLACES` with a rest site, a card reward, a shop and a route choice each, the deck a little different at each place."""
    out = []
    for k, (act, floor, hp) in enumerate(PLACES):
        view = view_of(F, vocab, act, floor, hp, deck_extra=k)
        for make in (rest_screen, card_screen, shop_screen, route_screen):
            decision = make(F)
            if len(decision.options) >= 2:
                out.append((view, decision))
    return out


def with_floor(view, **kw):
    return dataclasses.replace(view, **kw)
