"""referee/driver: an empty treasure chest is played, not a run end.

TreasureRoomRelicSynchronizer.BeginRelicPicking (run at room entry by TreasureRoom.EnterInternal) generates a relic only
while Hook.ShouldGenerateTreasure holds; SilverCrucible.ShouldGenerateTreasure is its one override (the relic's first
treasure room), and with nothing generated the synchronizer ends the voting at once, so CurrentRelics is null before the
chest is clicked and NTreasureRoomRelicCollection shows the empty chest. The driver once ended the run there
("treasure_without_relic_picking": 6% of the generation-7 night's runs, every one a Silver Crucible taken at Neow, at the
Act-1 chest). The driver is not built here, so this reads PlayTreasure's source: the empty chest is logged as a treasure
entry with an empty offer and the choice "empty", its chest task is awaited with the bounded wait, and the run walks on;
only a faulted or timed-out chest keeps a stop string. The live proof is a replay of a Silver Crucible run past floor 10.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

DRIVER = Path(__file__).resolve().parents[2] / "referee" / "driver"


def code(name: str) -> str:
    """The file's text without // comment lines and trailing // comments (a comment may name the old stop)."""
    text = (DRIVER / name).read_text(encoding="utf-8")
    return "\n".join(re.sub(r"//.*$", "", line) for line in text.splitlines())


def play_treasure() -> str:
    src = code("RunPlay.cs")
    start = src.index("private static async Task<string?> PlayTreasure")
    return src[start:src.index("private static async Task<string?> PlayShop", start)]


class EmptyChestSourceTest(unittest.TestCase):
    def test_a_null_relic_list_is_the_empty_chest_and_is_played(self):
        body = play_treasure()
        self.assertNotIn('"treasure_without_relic_picking"', body)
        self.assertIn('e["choice"] = "empty"', body)
        self.assertIn('e["offer"] = new JsonArray()', body)
        # The chest task (closed by CompleteWithNoRelics) is awaited with the bounded wait before the run walks on.
        self.assertRegex(body, r'WaitFor\(chest, "empty chest"\)')
        # The empty chest ends with the run continuing, as a played room does.
        empty = body[: body.index('var r = log.Begin("treasure")')]
        self.assertIn("return null;", empty)

    def test_only_a_faulted_or_timed_out_chest_keeps_a_stop(self):
        empty = play_treasure()
        empty = empty[: empty.index('var r = log.Begin("treasure")')]
        stops = re.findall(r'return "([a-z_]+)";', empty)
        self.assertEqual(sorted(stops), ["treasure_chest_error", "treasure_chest_timeout"])
        self.assertIn("chest.IsFaulted", empty)

    def test_no_driver_file_ends_a_run_on_a_null_relic_list(self):
        for name in sorted(p.name for p in DRIVER.glob("*.cs")):
            self.assertNotIn("treasure_without_relic_picking", code(name), name)


if __name__ == "__main__":
    unittest.main()
