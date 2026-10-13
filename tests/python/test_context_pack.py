"""tools.fidelity.context_pack: divergences group by cause and each group gets a short pack.

Synthetic fights in the shapes the referee and EmuDump write (game ids UPPER_SNAKE, emulator ids
through data/id_map.json). Two fights break the same way after the same card and must fall into
one group; a third breaks differently. Paths only appear in the pack, never source code.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import context_pack as cp  # noqa: E402

ID_MAP = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))
CID, EID = ID_MAP["cards"], ID_MAP["enemies"]


def game_state(hp, powers):
    return {"in_combat": True, "round": 1,
            "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "hand": [["STRIKE_IRONCLAD", False]], "draw": [], "discard": [], "exhaust": [],
            "enemies": [{"name": "CORPSE_SLUG", "hp": hp, "max_hp": 27, "block": 0, "powers": powers}],
            "rng": {"Shuffle": {"seed": 1, "counter": 1}}}


def emu_state(hp, buffs):
    card = {"def_id": CID["StrikeIronclad"], "upgraded": False}
    return {"in_combat": True, "draw_order": [], "rng": {"Shuffle": 1},
            "snapshot": {"player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3, "buffs": []},
                         "piles": {"hand": [card], "discard": [], "exhaust": []},
                         "enemies": [{"def_id": EID["CorpseSlug"], "hp": hp, "block": 0, "buffs": buffs}]}}


def fight(seed, game_hp, emu_hp, game_powers=(), emu_buffs=()):
    play = {"kind": "play", "hand_index": 0, "card": "STRIKE_IRONCLAD", "target": 0, "target_slot": 0}
    g = {"seed": seed, "steps": [{"action": None, "state": game_state(27, [])},
                                  {"action": play, "state": game_state(game_hp, list(game_powers))}]}
    e = {"seed": seed, "steps": [{"action": None, "state": emu_state(27, [])},
                                  {"action": play, "state": emu_state(emu_hp, list(emu_buffs))}]}
    return g, e


class ContextPackTest(unittest.TestCase):
    def test_grouping_and_pack(self):
        fights = [fight("AAA", 21, 20), fight("BBB", 21, 19), fight("CCC", 21, 21, [["RAVENOUS_POWER", 5]]),
                  fight("DDD", 21, 21)]
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "g.json").write_text(json.dumps({"results": [f[0] for f in fights]}))
            (t / "e.json").write_text(json.dumps({"results": [f[1] for f in fights]}))
            text = cp.run(t / "g.json", t / "e.json", t / "packs")
            self.assertIn("4 fights: 1 identical, 3 diverging in 2 groups", text)
            self.assertIn("[2] enemies.hp after STRIKE_IRONCLAD vs CORPSE_SLUG", text)
            packs = sorted((t / "packs").glob("g0*.md"))
            self.assertEqual(len(packs), 2)
            body = packs[0].read_text(encoding="utf-8")
            self.assertLessEqual(len(body.splitlines()), cp.MAX_LINES)
            self.assertIn("AAA, BBB", body)
            self.assertIn("hp: game 21  emu 20", body)
            self.assertIn("monster CORPSE_SLUG", body)
            self.assertIn("card STRIKE_IRONCLAD", body)
            self.assertNotIn("{", body)  # no pasted source

    def test_trace_diff_of_a_seed_in_the_group_is_folded_in(self):
        fights = [fight("AAA", 21, 20), fight("BBB", 21, 19)]
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "g.json").write_text(json.dumps({"results": [f[0] for f in fights]}))
            (t / "e.json").write_text(json.dumps({"results": [f[1] for f in fights]}))
            (t / "packs" / "traces").mkdir(parents=True)
            (t / "packs" / "traces" / "BBB.trace.txt").write_text("BBB step 1: play X\n  FIRST DIFFERENCE here\n", encoding="utf-8")
            cp.run(t / "g.json", t / "e.json", t / "packs")
            body = sorted((t / "packs").glob("g0*.md"))[0].read_text(encoding="utf-8")
            self.assertIn("## Trace diff (seed BBB", body)
            self.assertIn("FIRST DIFFERENCE here", body)


if __name__ == "__main__":
    unittest.main()
