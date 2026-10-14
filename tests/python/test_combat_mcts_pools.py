"""rl/combat_mcts.py's act-1 encounter pools are a copy of Core/Run/RunConstants.cs.

The copy went stale once: Underdocks normal kept the weak Corpse Slugs (9) and Seapunk (12)
after the run engine started dealing their normal twins (88, 22) from that list, so the
"underdocks-normal" pool and every script built on it fought the weak rosters. This holds
each copied pool equal to the engine's, in order.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))

import combat_mcts  # noqa: E402

RUN_CONSTANTS = ROOT / "src" / "Sts2Emulator" / "Core" / "Run" / "RunConstants.cs"

COPIES = {
    "OvergrowthWeakEncounters": combat_mcts.OVERGROWTH_WEAK,
    "OvergrowthNormalEncounters": combat_mcts.OVERGROWTH_NORMAL,
    "OvergrowthEliteEncounters": combat_mcts.OVERGROWTH_ELITE,
    "UnderdocksWeakEncounters": combat_mcts.UNDERDOCKS_WEAK,
    "UnderdocksNormalEncounters": combat_mcts.UNDERDOCKS_NORMAL,
    "UnderdocksEliteEncounters": combat_mcts.UNDERDOCKS_ELITE,
}


def engine_pool(name: str) -> tuple[int, ...]:
    text = RUN_CONSTANTS.read_text(encoding="utf-8")
    m = re.search(rf"ReadOnlySpan<int>\s+{name}\s*=>\s*\[([^\]]*)\]", text)
    if m is None:
        raise AssertionError(f"{name} not found in RunConstants.cs")
    # Literal ids only: a named constant in the list would need resolving first.
    return tuple(int(x) for x in m.group(1).split(",") if x.strip())


class CombatMctsPoolTests(unittest.TestCase):
    def test_every_copied_pool_equals_the_engines(self) -> None:
        for name, copy in COPIES.items():
            with self.subTest(pool=name):
                self.assertEqual(copy, engine_pool(name))

    def test_act_one_lists_each_copy_under_its_name(self) -> None:
        self.assertEqual(
            set(combat_mcts.ACT_ONE.values()), set(COPIES.values())
        )

    def test_underdocks_normal_deals_the_normal_slugs_and_seapunk(self) -> None:
        self.assertIn(88, combat_mcts.UNDERDOCKS_NORMAL)
        self.assertIn(22, combat_mcts.UNDERDOCKS_NORMAL)
        self.assertNotIn(9, combat_mcts.UNDERDOCKS_NORMAL)
        self.assertNotIn(12, combat_mcts.UNDERDOCKS_NORMAL)

    def test_every_id_any_act_deals_has_a_gym_label(self) -> None:
        # Sts2CombatEnv and Sts2RunEnv name a fight through ENCOUNTER_LABELS; an id the
        # engine deals and the table lacks reads "none" (normal Corpse Slugs, 88, did).
        text = RUN_CONSTANTS.read_text(encoding="utf-8")
        consts = {
            m.group(1): int(m.group(2))
            for m in re.finditer(r"const int (\w+) = (\d+);", text)
        }
        pools = re.findall(r"ReadOnlySpan<int>\s+(\w+Encounters)\s*=>\s*\[([^\]]*)\]", text)
        self.assertGreater(len(pools), 6)
        for name, body in pools:
            for token in (t.strip() for t in body.split(",") if t.strip()):
                eid = int(token) if token.isdigit() else consts[token]
                with self.subTest(pool=name, id=eid):
                    self.assertIn(eid, combat_mcts.ENCOUNTER_LABELS)


if __name__ == "__main__":
    unittest.main()
