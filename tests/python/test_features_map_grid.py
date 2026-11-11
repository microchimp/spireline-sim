"""Block M of the feature contract: the whole act map as a grid (`features.map_grid`, D58 addendum 13).

The expected numbers are written by hand from the encoding (per cell: eight node-type one-hots, three edge bits, reachable, here), on a
small map built for it: a merge (two parents of one node), a dead column, a dead end (a node no path leaves), the start's fan and a node
too far from its parent to draw. The real-map checks read the act maps the engine draws.
"""

import hashlib
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "rl", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rl.run_train import features, vocab  # noqa: E402
from rl.run_train.engine import RunSession  # noqa: E402
from rl.run_train.features import CardRef, MapNode, RunView  # noqa: E402

M, E, R, S, T, B, U, A = (
    vocab.NODE_MONSTER, vocab.NODE_ELITE, vocab.NODE_REST, vocab.NODE_SHOP, vocab.NODE_TREASURE, vocab.NODE_BOSS, vocab.NODE_UNKNOWN,
    vocab.NODE_ANCIENT,
)
# the channel of each type: its vocab number less one (monster 0 ... ancient 7); then the edges (8, 9, 10: column -1, 0, +1), reachable 11, here 12
CH = {t: t - 1 for t in (M, E, R, S, T, B, U, A)}
EDGE_L, EDGE_S, EDGE_R, REACH, HERE = 8, 9, 10, 11, 12

# (3,0) ancient -> (2,1) monster, (4,1) unknown
# (2,1) monster -> (2,2) elite, (3,2) rest
# (4,1) unknown -> (3,2) rest, (5,2) treasure (a dead end: no node follows it)
# (2,2) elite -> (3,3) shop;  (3,2) rest -> (3,3) shop;  (3,3) shop -> (3,4) boss
# columns 0, 1 and 6 hold no node at all
HAND_MAP = (
    MapNode(3, 0, A, ((2, 1), (4, 1))),
    MapNode(2, 1, M, ((2, 2), (3, 2))),
    MapNode(4, 1, U, ((3, 2), (5, 2))),
    MapNode(2, 2, E, ((3, 3),)),
    MapNode(3, 2, R, ((3, 3),)),
    MapNode(5, 2, T, ()),
    MapNode(3, 3, S, ((3, 4),)),
    MapNode(3, 4, B, ()),
)


def make_view(**overrides) -> RunView:
    fields = {"ascension": 10, "act_index": 0, "act_region": vocab.REGION_OVERGROWTH, "floor": 3, "hp": 50, "max_hp": 80, "gold": 120,
              "deck": (CardRef(472, False), CardRef(131, False), CardRef(30, False)), "relics": (vocab.relic_id_by_name("BurningBlood"),),
              "potions": (), "max_potion_slots": 3, "map_nodes": HAND_MAP, "position": (3, 0), "boss": 83}
    fields.update(overrides)
    return RunView(**fields)


def cell_of(grid: np.ndarray, col: int, row: int) -> np.ndarray:
    return grid.reshape(features.MAP_GRID_SHAPE)[row, col]


def expected_cell(node_type: int | None = None, edges: tuple[int, ...] = (), reach: bool = False, here: bool = False) -> np.ndarray:
    out = np.zeros(13, dtype=np.int8)
    if node_type is not None:
        out[CH[node_type]] = 1
    for dcol in edges:
        out[EDGE_S + dcol] = 1
    out[REACH], out[HERE] = int(reach), int(here)
    return out


class ShapeTests(unittest.TestCase):
    def test_the_layout_constants(self):
        self.assertEqual(features.MAP_ROWS, 16)
        self.assertEqual(features.MAP_GRID_ROWS, 17)  # rows 0 to 16: Act 1's boss row is 16
        self.assertEqual(features.MAP_COLS, 7)
        self.assertEqual(features.MAP_GRID_SHAPE, (17, 7, 13))
        self.assertEqual(features.MAP_GRID_SIZE, 17 * 7 * 13)
        grid = features.map_grid(make_view())
        self.assertEqual(grid.shape, (features.MAP_GRID_SIZE,))
        self.assertEqual(grid.dtype, np.int8)
        self.assertTrue(set(np.unique(grid)) <= {0, 1})

    def test_the_base_vector_is_not_widened_and_not_changed(self):
        self.assertEqual(features.BASE_SIZE, 1855)
        view = make_view()
        before = features.encode(view)
        features.map_grid(view)
        after = features.encode(view)
        self.assertEqual(before.tobytes(), after.tobytes())
        self.assertEqual(before.shape, (1855,))
        # the judgement's rows are pinned byte for byte: a digest taken on main before block M existed
        self.assertEqual(hashlib.sha256(before.tobytes()).hexdigest()[:16], PINNED_ROW_DIGEST)


class HandMapTests(unittest.TestCase):
    def test_every_cell_at_the_start(self):
        g = features.map_grid(make_view(position=(3, 0)))
        want = {
            (3, 0): expected_cell(A, (-1, 1), reach=True, here=True),
            (2, 1): expected_cell(M, (0, 1), reach=True),
            (4, 1): expected_cell(U, (-1, 1), reach=True),
            (2, 2): expected_cell(E, (1,), reach=True),
            (3, 2): expected_cell(R, (0,), reach=True),
            (5, 2): expected_cell(T, (), reach=False),  # a dead end: reachable from the position, but no path from it to the boss
            (3, 3): expected_cell(S, (0,), reach=True),
            (3, 4): expected_cell(B, (), reach=True),
        }
        for r in range(features.MAP_GRID_ROWS):
            for c in range(features.MAP_COLS):
                with self.subTest(col=c, row=r):
                    np.testing.assert_array_equal(cell_of(g, c, r), want.get((c, r), expected_cell()))
        # an empty column and everything below the boss are all zero
        self.assertEqual(int(g.reshape(features.MAP_GRID_SHAPE)[:, 0].sum() + g.reshape(features.MAP_GRID_SHAPE)[:, 1].sum()), 0)
        self.assertEqual(int(g.reshape(features.MAP_GRID_SHAPE)[5:].sum()), 0)

    def test_the_position_moves_here_and_reachable(self):
        g = features.map_grid(make_view(position=(2, 1)))
        shaped = g.reshape(features.MAP_GRID_SHAPE)
        here = np.argwhere(shaped[:, :, HERE] == 1).tolist()
        self.assertEqual(here, [[1, 2]])  # row 1, column 2
        reach = sorted((int(c), int(r)) for r, c in np.argwhere(shaped[:, :, REACH] == 1))
        self.assertEqual(reach, [(2, 1), (2, 2), (3, 2), (3, 3), (3, 4)])  # not the start, not its sibling, not the dead end
        # the node types and the edges do not depend on the position
        np.testing.assert_array_equal(g.reshape(features.MAP_GRID_SHAPE)[:, :, :EDGE_R + 1],
                                      features.map_grid(make_view(position=(3, 0))).reshape(features.MAP_GRID_SHAPE)[:, :, :EDGE_R + 1])

    def test_a_merge_is_reachable_from_either_parent(self):
        for position in ((2, 1), (4, 1)):
            with self.subTest(position=position):
                shaped = features.map_grid(make_view(position=position)).reshape(features.MAP_GRID_SHAPE)
                self.assertEqual(int(shaped[2, 3, REACH]), 1)  # the rest site both lead to
                self.assertEqual(int(shaped[3, 3, REACH]), 1)
        # both of the merge node's parents have an edge into it, at columns +1 and -1
        shaped = features.map_grid(make_view()).reshape(features.MAP_GRID_SHAPE)
        self.assertEqual(int(shaped[1, 2, EDGE_R]), 1)
        self.assertEqual(int(shaped[1, 4, EDGE_L]), 1)

    def test_the_boss_is_the_last_reachable_cell(self):
        shaped = features.map_grid(make_view(position=(3, 4))).reshape(features.MAP_GRID_SHAPE)
        self.assertEqual(int(shaped[:, :, REACH].sum()), 1)
        self.assertEqual(int(shaped[4, 3, REACH]), 1)
        self.assertEqual(int(shaped[4, 3, HERE]), 1)

    def test_a_position_off_the_map_has_nothing_reachable_and_no_here_cell(self):
        shaped = features.map_grid(make_view(position=(0, 0), map_nodes=())).reshape(features.MAP_GRID_SHAPE)
        self.assertEqual(int(shaped.sum()), 0)
        shaped = features.map_grid(make_view(position=(0, 7))).reshape(features.MAP_GRID_SHAPE)
        self.assertEqual(int(shaped[:, :, REACH].sum() + shaped[:, :, HERE].sum()), 0)

    def test_the_start_and_boss_fans_are_not_drawn_past_one_column(self):
        # the start's children in columns 0 and 6 and a node whose child is three columns over: only the +-1 edges are bits
        nodes = (
            MapNode(3, 0, A, ((0, 1), (3, 1), (6, 1))),
            MapNode(0, 1, M, ((0, 2),)),
            MapNode(3, 1, M, ((2, 2), (4, 2))),
            MapNode(6, 1, M, ((6, 2),)),
            MapNode(0, 2, R, ((3, 3),)),
            MapNode(2, 2, R, ((3, 3),)),
            MapNode(4, 2, R, ((3, 3),)),
            MapNode(6, 2, R, ((3, 3),)),
            MapNode(3, 3, B, ()),
        )
        shaped = features.map_grid(make_view(map_nodes=nodes, position=(3, 0))).reshape(features.MAP_GRID_SHAPE)
        np.testing.assert_array_equal(shaped[0, 3, EDGE_L:EDGE_R + 1], [0, 1, 0])  # the child in the same column only
        np.testing.assert_array_equal(shaped[2, 0, EDGE_L:EDGE_R + 1], [0, 0, 0])  # three columns to the boss: not a bit
        np.testing.assert_array_equal(shaped[2, 2, EDGE_L:EDGE_R + 1], [0, 0, 1])
        np.testing.assert_array_equal(shaped[2, 4, EDGE_L:EDGE_R + 1], [1, 0, 0])
        # but the reachable bit still says the whole fan is on a path
        self.assertEqual(int(shaped[1, 0, REACH] + shaped[1, 6, REACH] + shaped[2, 0, REACH] + shaped[2, 6, REACH]), 4)

    def test_a_node_outside_the_grid_or_of_an_unknown_type_is_refused(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            features.map_grid(make_view(map_nodes=HAND_MAP + (MapNode(7, 1, M, ()),)))
        with self.assertRaisesRegex(ValueError, "outside"):
            features.map_grid(make_view(map_nodes=HAND_MAP + (MapNode(1, features.MAP_GRID_ROWS, M, ()),)))
        with self.assertRaisesRegex(ValueError, "type"):
            features.map_grid(make_view(map_nodes=HAND_MAP + (MapNode(1, 1, 0, ()),)))

    def test_the_cache_hands_out_copies(self):
        a = features.map_grid(make_view())
        a[:] = 7
        b = features.map_grid(make_view())
        self.assertTrue(set(np.unique(b)) <= {0, 1})
        c = features.map_grid(replace(make_view(), hp=1))  # a view that differs only off the map: the same grid
        np.testing.assert_array_equal(b, c)


class RealMapTests(unittest.TestCase):
    """The act maps the engine draws: Act 1's boss is on the last row of the grid, a later act's earlier, every edge but the two fans is drawn."""

    def test_the_maps_of_every_act_fit_the_grid_and_draw_every_edge_but_the_fans(self):
        seen_rows = set()
        for seed in ("MAPA0001", "MAPB0002", "MAPC0003"):
            session = RunSession(seed, 10, 1)
            try:
                for act in range(vocab.ACTS):
                    if act:
                        session.env.debug_enter_next_act()
                    view = features.view_from_engine(session.env, 10)
                    with self.subTest(seed=seed, act=act):
                        grid = features.map_grid(view)
                        shaped = grid.reshape(features.MAP_GRID_SHAPE)
                        by_coord = {(n.col, n.row): n for n in view.map_nodes}
                        bosses = [n for n in view.map_nodes if n.node_type == vocab.NODE_BOSS]
                        seen_rows.add(max(n.row for n in bosses))
                        self.assertLessEqual(max(n.row for n in bosses), features.MAP_ROWS)
                        self.assertEqual(int(shaped[:, :, :EDGE_L].sum()), len(view.map_nodes))  # one type bit a node
                        self.assertEqual(int(shaped[view.position[1], view.position[0], HERE]), 1)
                        self.assertEqual(int(shaped[:, :, HERE].sum()), 1)
                        self.assertEqual(int(shaped[view.position[1], view.position[0], REACH]), 1)
                        for node in view.map_nodes:
                            for col, row in node.children:
                                self.assertEqual(row, node.row + 1)
                                near = abs(col - node.col) <= 1
                                fan = node.node_type == vocab.NODE_ANCIENT or by_coord[(col, row)].node_type == vocab.NODE_BOSS
                                self.assertTrue(near or fan, f"edge {(node.col, node.row)} -> {(col, row)} is neither within a column nor a fan")
                                self.assertEqual(int(shaped[node.row, node.col, EDGE_S + col - node.col]) if near else 1, 1)
                        # from the start every node on a path to the boss is reachable: all of them in a drawn map
                        self.assertEqual(int(shaped[:, :, REACH].sum()), len(view.map_nodes))
            finally:
                session.close()
        self.assertIn(features.MAP_ROWS, seen_rows)  # Act 1's boss row is the grid's last


# `encode(make_view())` of this file's view, as main built it before block M (the first 16 hex digits of the sha256 of its float32 bytes)
PINNED_ROW_DIGEST = "beda711f28f75a39"


if __name__ == "__main__":
    unittest.main()
