"""Sts2Run_ReanchorFight through its ctypes binding (src/sts2_gym/native.py run_reanchor_fight): the fight a run stands
in is put on the referee's record of it, as an agent sees that record (the draw pile as a sorted list, no streams).
The record here is read off the run's own fight, with its hand and its top draw card swapped, so what the native
call must do is observable by reading the piles back. Needs the native library and skips without it."""

import ctypes
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))

import leakprobes  # noqa: E402

try:
    leakprobes.native_env()
    from run_train import engine  # noqa: E402
    from sts2_gym import native  # noqa: E402

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    HAVE_NATIVE = False


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).upper()


@unittest.skipUnless(HAVE_NATIVE, "needs the native library")
class NativeReanchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        id_map = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))
        cls.card = {v: snake(k) for k, v in id_map["cards"].items()}
        cls.enemy = {v: snake(k) for k, v in id_map["enemies"].items()}

    def in_fight(self):
        s = leakprobes.play_to("QS2GYXRKWN", lambda s, d: s.phase == engine.rc.PHASE_COMBAT)
        self.addCleanup(s.close)
        return s

    def piles(self, s):
        """The fight's piles in true order, read off a faithful combat handle (a test reads the oracle's view)."""
        obs = (ctypes.c_int * native.RUN_OBS_SIZE)()
        h = native.run_fork_combat(s.handle, False, 0, obs)
        try:
            out = {p: native.get_pile(h, p) for p in ("hand", "draw", "discard", "exhaust")}
            out["enemies"] = native.get_enemies(h)
        finally:
            native.destroy(h)
        return out

    def record(self, s, piles):
        view = s.view()

        def pile(cards):
            return [[self.card[c], bool(up)] for c, up in cards]

        return {"in_combat": True, "round": 1,
                "player": {"hp": view.hp, "max_hp": view.max_hp, "block": 0, "energy": 3, "powers": []},
                "hand": pile(piles["hand"]), "draw": sorted(pile(piles["draw"])), "discard": pile(piles["discard"]),
                "exhaust": pile(piles["exhaust"]),
                "enemies": [{"name": self.enemy[d], "hp": hp, "max_hp": hp, "block": 0, "powers": []}
                            for d, _slot, hp in piles["enemies"]]}

    def test_the_run_api_version_is_the_one_with_the_export(self):
        self.assertGreaterEqual(native._REQUIRED_RUN_NATIVE_API_VERSION, 24)
        self.assertTrue(hasattr(native, "run_reanchor_fight"))

    def test_a_fight_is_put_on_a_record_that_swaps_the_hand_and_the_top_card(self):
        s = self.in_fight()
        before = self.piles(s)
        self.assertTrue(before["draw"], "the fight has a draw pile to take a card from")
        # swap the hand's first card with the draw pile's top card (the draw pile is given only as a sorted list)
        hand, draw = list(before["hand"]), list(before["draw"])
        hand[0], draw[0] = draw[0], hand[0]
        want = dict(before, hand=hand, draw=draw)
        status, message = native.run_reanchor_fight(s.handle, self.record(s, want), s.env._run_obs_buf)
        self.assertGreaterEqual(status, 0, message)
        self.assertIn("patched=", message)
        after = self.piles(s)
        self.assertEqual(after["hand"], hand)
        self.assertEqual(sorted(after["draw"]), sorted(draw))
        self.assertEqual(after["enemies"], before["enemies"])
        # the same record again changes nothing
        status, message = native.run_reanchor_fight(s.handle, self.record(s, want), s.env._run_obs_buf)
        self.assertEqual(status, 0, message)

    def test_a_record_with_a_card_the_emulator_lacks_is_refused_and_leaves_the_fight(self):
        s = self.in_fight()
        before = self.piles(s)
        rec = self.record(s, before)
        rec["hand"].append(["NO_SUCH_CARD", False])
        status, message = native.run_reanchor_fight(s.handle, rec, s.env._run_obs_buf)
        self.assertEqual(status, -4)
        self.assertTrue(message.startswith("no emulator id for"), message)
        self.assertEqual(self.piles(s), before)

    def test_no_fight_and_no_json_have_their_own_codes(self):
        s = self.in_fight()
        buf = s.env._run_obs_buf
        self.assertEqual(native.run_reanchor_fight(s.handle, "not json", buf)[0], -2)
        self.assertEqual(native.run_reanchor_fight(255, {}, buf)[0], -1)


if __name__ == "__main__":
    unittest.main()
