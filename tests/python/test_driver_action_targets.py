"""referee/driver: no action that aims at an enemy is offered while no creature is alive (the Test Subject between its forms).

The rule itself (ActionTargets.For) is tested in referee/driver.Tests (ActionTargetsTests, xunit); the driver needs the game and is not
built here, so this reads the sources and keeps the fight loop of RunPlay.cs on that rule: the potion list once offered every
enemy-target potion at target -1 when no enemy was alive (the game cancelled it and kept the potion, the stand-in repeated it until the
400-action cap: 14 runs of the 2026-11-19 real block ended `fight_unfinished`)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REFEREE = Path(__file__).resolve().parents[2] / "referee"


def code(path: Path) -> str:
    """The file's text without // comments."""
    return "\n".join(re.sub(r"//.*$", "", line) for line in path.read_text(encoding="utf-8").splitlines())


class FightLoopTargetsTest(unittest.TestCase):
    def fight_loop(self) -> str:
        text = code(REFEREE / "driver" / "RunPlay.cs")
        start = text.index("private static async Task<string?> FightLoop(")
        return text[start:text.index("CombatManager.Instance.IsInProgress ? \"max_actions\" : null", start)]

    def test_cards_and_potions_take_their_targets_from_the_rule(self):
        body = self.fight_loop()
        self.assertEqual(3, len(re.findall(r"ActionTargets\.For\(", body)))  # the cards, the usable-potion filter, the potion actions
        self.assertIn("options.Add((h, t))", body)
        self.assertIn("potionActs.Add((pm, ps, t))", body)

    def test_a_potion_with_no_legal_target_is_not_usable(self):
        body = self.fight_loop()
        usable = body[body.index("var usable = "):body.index("var potionActs")]
        self.assertRegex(usable, r"ActionTargets\.For\(x\.p\.TargetType == TargetType\.AnyEnemy, enemies\.Count\)\.Count > 0")

    def test_no_target_minus_one_fallback_for_an_enemy_potion_remains(self):
        body = self.fight_loop()
        self.assertNotRegex(body, r"potionActs\.Add\(\(pm, ps, -1\)\)")

    def test_the_rule_is_compiled_into_the_driver_tests(self):
        csproj = (REFEREE / "driver.Tests" / "DriverPickLoop.Tests.csproj").read_text(encoding="utf-8")
        self.assertIn("ActionTargets.cs", csproj)
        self.assertTrue((REFEREE / "driver.Tests" / "ActionTargetsTests.cs").is_file())


if __name__ == "__main__":
    unittest.main()
