"""Twenty deterministic Act 1 route screens for `test_route_elite` (hand-built act maps, seeded): a map of 16 rows with elites on it, a position with a
choice, a state to stand in (HP, floor), the options (the rooms the node leads to). Not a game and not a record: a fixed set of screens to compare two
policies on, choice for choice."""

from __future__ import annotations

import dataclasses

import numpy as np

COLS = 7
MONSTER, ELITE, REST, SHOP, TREASURE, BOSS, UNKNOWN, ANCIENT = 1, 2, 3, 4, 5, 6, 7, 8


def act_map(seed: int, F):
    """(nodes, {coord: node}) of a seeded act map: an ancient at (3, 0), rows 1 to 15 of three to five rooms, the boss at (3, 16); every room has a
    child within a column of its own (the nearest room of the next row when there is none)."""
    rng = np.random.default_rng(seed)
    rows: dict[int, list[int]] = {0: [3], 16: [3]}
    for r in range(1, 16):
        rows[r] = sorted(int(c) for c in rng.choice(COLS, size=int(rng.integers(3, 6)), replace=False))

    def room(r: int) -> int:
        if r == 0:
            return ANCIENT
        if r == 16:
            return BOSS
        if r == 1:
            return MONSTER
        if r == 9:
            return TREASURE
        if r == 15:
            return REST
        pool = [MONSTER] * 4 + [UNKNOWN] * 2 + [REST, SHOP] + ([ELITE] * 3 if r >= 5 else [])
        return pool[int(rng.integers(len(pool)))]

    nodes = []
    for r in range(0, 17):
        for c in rows[r]:
            kids = []
            if r < 16:
                kids = [(k, r + 1) for k in rows[r + 1] if abs(k - c) <= 1] or [(min(rows[r + 1], key=lambda k: (abs(k - c), k)), r + 1)]
            nodes.append(F.MapNode(c, r, room(r), tuple(kids)))
    return tuple(nodes)


def route_screens(view, F, n: int = 20):
    """[(view, decision)] of `n` route screens built on `view` (its deck and relics): each on its own seeded map, standing on a room of rows 1 to 13 that
    leads to two or more rooms, at some HP of 80 and the floor that row has. The same `n` every time."""
    out = []
    seed = 0
    while len(out) < n:
        seed += 1
        rng = np.random.default_rng(1000 + seed)
        nodes = act_map(seed, F)
        choices = [x for x in nodes if 1 <= x.row <= 13 and len(x.children) >= 2]
        if not choices:
            continue
        here = choices[int(rng.integers(len(choices)))]
        by = {(x.col, x.row): x for x in nodes}
        hp = int(rng.integers(20, 81))
        v = dataclasses.replace(view, act_index=0, floor=here.row + 1, hp=hp, max_hp=80, map_nodes=nodes, position=(here.col, here.row))
        options = tuple(F.Option(kind="map.node", action=i, node=(c, r, by[(c, r)].node_type)) for i, (c, r) in enumerate(here.children))
        out.append((v, F.Decision("map", options)))
    return out
