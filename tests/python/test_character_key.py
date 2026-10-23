"""The reset spec's `character` key through envproto (play, bench, record) and rl/gen0.py (score, emu)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT))

import envproto  # noqa: E402
import gen0  # noqa: E402


class SpecCarriesTheCharacter(unittest.TestCase):
    def test_default_is_ironclad_and_the_key_is_sent(self):
        self.assertEqual(envproto.reset_spec("S1")["character"], "IRONCLAD")
        self.assertEqual(envproto.reset_spec("S1", "SILENT")["character"], "SILENT")
        sent = []
        env = envproto.GameEnv.__new__(envproto.GameEnv)
        env._send = sent.append
        env._next = lambda timeout: ({"t": "decide"}, False)
        env.reset("S1")
        env.reset("S2", character="DEFECT", spec={"buff_max_hp": 5})
        self.assertEqual(sent[0]["spec"]["character"], "IRONCLAD")
        self.assertEqual(sent[1]["spec"]["character"], "DEFECT")
        self.assertEqual(sent[1]["spec"]["buff_max_hp"], 5)
        json.dumps(sent)  # the command is plain JSON

    def test_every_envproto_mode_takes_character(self):
        for mode in ("play", "bench", "record", "smoke", "replay"):
            a = envproto.parser().parse_args([mode, "--game", "g", "--out", "o", "--character", "NECROBINDER"])
            self.assertEqual(a.character, "NECROBINDER")
            self.assertEqual(envproto.parser().parse_args([mode, "--game", "g", "--out", "o"]).character, "IRONCLAD")

    def test_gen0_score_and_emu_take_character(self):
        p = gen0.build_parser()
        a = p.parse_args(["score", "--game", "g", "--out", "o", "--seeds-file", "s", "--character", "REGENT"])
        self.assertEqual(a.character, "REGENT")
        self.assertEqual(p.parse_args(["score", "--game", "g", "--out", "o", "--seeds-file", "s"]).character, "IRONCLAD")
        e = p.parse_args(["emu", "--seeds-file", "s", "--out", "o", "--character", "SILENT"])
        self.assertEqual(e.character, "SILENT")


if __name__ == "__main__":
    unittest.main()
