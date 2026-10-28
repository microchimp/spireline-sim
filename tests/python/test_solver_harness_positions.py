"""The equal-value harness's position list is the one tools/solver/harness.py writes (piece S3)."""

from __future__ import annotations

import json
import unittest
from collections import Counter

from tools.solver import harness


def rows() -> list[dict]:
    return [json.loads(line) for line in harness.OUT.read_text(encoding="utf-8").splitlines()]


class HarnessPositionsTests(unittest.TestCase):
    def test_positions_file_is_current(self):
        assert harness.OUT.read_text(encoding="utf-8") == harness.render()

    def test_positions_cover_both_engines_sources_and_horizons(self):
        rs = rows()
        assert len(rs) >= 400
        assert {r["engine"] for r in rs} == {"old", "core2"}
        assert {r["source"] for r in rs} == {"corpus", "selfplay"}
        assert {r["horizon"] for r in rs} == {0, 1}
        assert len({r["id"] for r in rs}) == len(rs)
        assert any(r["after"] > 0 for r in rs) and any(r["mix"] for r in rs)

    def test_horizon_is_drawn_apart_from_the_other_fields(self):
        h1 = [r for r in rows() if r["horizon"] == 1]
        assert len(h1) >= 60
        for field in ("engine", "source"):
            counts = Counter(r[field] for r in h1)
            assert len(counts) == 2 and min(counts.values()) >= 12, (field, counts)
        assert {2, 5, 9, 14} <= {r["enemy_hp"] for r in h1}
        assert len(Counter((r["engine"], r["source"]) for r in h1)) == 4

    def test_screen_and_order_dependence_recipes(self):
        rs = rows()
        plays = Counter(r["play"] for r in rs if r["play"])
        assert set(plays) == set(harness.SCREEN_PLAYS) and min(plays.values()) >= 4
        assert {"SeekerStrike", "Cleanse", "Wish"} <= set(plays)
        fights = Counter(r["encounter"] for r in rs if r["encounter"])
        assert fights["HunterKiller"] >= 2 and fights["Toadpoles"] >= 2
        # The order-dependence recipes play until a named point; a macro recipe's encounter is only its fight.
        assert all(r["until"] for r in rs if r["encounter"] and not r["macro"])


if __name__ == "__main__":
    unittest.main()
