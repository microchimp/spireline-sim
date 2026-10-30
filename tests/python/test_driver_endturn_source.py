"""The driver's End Turn request carries the player's own turn counter, as the game's End Turn button does (W5, night 2026-10-30).

The game drops an EndPlayerTurnAction whose number differs from `player.PlayerCombatState.TurnNumber`; the combat's `RoundNumber` runs
behind that counter for the rest of a fight after an extra turn (the Pael's Eye relic), so a request built from it is dropped for ever
and the fight loop ends the run as `fight_unfinished`. The driver is a game mod and cannot be built or run in a unit test, so this reads
the source: every place that builds the action must pass the player's turn counter. No game, no emulator."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

DRIVER = Path(__file__).resolve().parents[2] / "referee" / "driver"
BUILD = re.compile(r"new EndPlayerTurnAction\(\s*(?P<args>[^;]*?)\)\s*\)\s*;")


class DriverEndTurnSourceTest(unittest.TestCase):
    def sites(self) -> list[tuple[str, str]]:
        found = []
        for f in sorted(DRIVER.glob("*.cs")):
            for m in BUILD.finditer(f.read_text(encoding="utf-8")):
                found.append((f.name, " ".join(m.group("args").split())))
        return found

    def test_the_two_request_sites_are_found(self):
        self.assertEqual(["Combat.cs", "RunPlay.cs"], sorted(f for f, _a in self.sites()))

    def test_every_end_turn_request_passes_the_players_turn_counter_not_the_round_number(self):
        for name, args in self.sites():
            with self.subTest(file=name):
                self.assertEqual("player, player.PlayerCombatState.TurnNumber", args)
                self.assertNotIn("RoundNumber", args)


if __name__ == "__main__":
    unittest.main()
