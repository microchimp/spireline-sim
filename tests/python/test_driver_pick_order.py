"""referee/driver: card picks go back to the game in the order the agent made them, at every DecideMany site.

The game's own card screens hand a selection back in click order and nothing in CardSelectCmd re-sorts it; the emulator
takes each pick as it is made. The driver once sorted a multi-pick answer into grid order outside full select, so the
game added the two cards of a Room Full of Cheese Gorge to the deck in grid order against the emulator's pick order, and
the opening shuffle (position based) transposed the pair in every later draw pile (F11 mechanism A). The loop itself is
tested in referee/driver.Tests (PickLoopTests, xunit); the driver is not built here, so this reads the sources and keeps
the sort from coming back and the two sites on the one loop.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

DRIVER = Path(__file__).resolve().parents[2] / "referee" / "driver"


def code(name: str) -> str:
    """The file's text without // comment lines and trailing // comments (a comment may name the old sort)."""
    text = (DRIVER / name).read_text(encoding="utf-8")
    return "\n".join(re.sub(r"//.*$", "", line) for line in text.splitlines())


class PickOrderSourceTest(unittest.TestCase):
    def test_decide_many_uses_the_pick_loop_and_does_not_sort(self):
        env = code("Env.cs")
        start = env.index("internal static List<int> DecideMany")
        body = env[start:env.index("internal static List<(string act, int slot, PotionModel potion)> PotionActs", start)]
        self.assertIn("PickLoop.Run(", body)
        self.assertNotRegex(body, r"\.Sort\(|OrderBy\(")

    def test_the_pick_loop_does_not_sort(self):
        self.assertNotRegex(code("PickLoop.cs"), r"\.Sort\(|OrderBy\(|Order\(")

    def test_both_sites_ask_through_decide_many(self):
        self.assertIn('Env.DecideMany("select_cards_combat"', code("Env.cs"))
        self.assertIn('Env.DecideMany("select_cards"', code("RunPlay.cs"))
        # The answer goes to the game as picked: no re-ordering between DecideMany and the cards handed back.
        run_play = code("RunPlay.cs")
        self.assertIn("idx.Select(i => list[i]).ToList()", run_play)
        self.assertNotRegex(run_play, r"idx\.(Order|Sort)|Sort\(\)")


if __name__ == "__main__":
    unittest.main()
