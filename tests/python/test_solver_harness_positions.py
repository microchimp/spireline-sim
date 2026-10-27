"""The equal-value harness's position list is the one tools/solver/harness.py writes (piece S3)."""

from __future__ import annotations

import json

from tools.solver import harness


def test_positions_file_is_current():
    assert harness.OUT.read_text(encoding="utf-8") == harness.render()


def test_positions_cover_both_engines_sources_and_horizons():
    rows = [json.loads(line) for line in harness.OUT.read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 200
    assert {r["engine"] for r in rows} == {"old", "core2"}
    assert {r["source"] for r in rows} == {"corpus", "selfplay"}
    assert {r["horizon"] for r in rows} == {0, 1}
    assert len({r["id"] for r in rows}) == len(rows)
    assert any(r["after"] > 0 for r in rows) and any(r["mix"] for r in rows)
