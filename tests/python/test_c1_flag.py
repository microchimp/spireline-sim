"""F1: the reset-spec flag extra_actions (default off; play and bench turn it on; record stores it, replay uses the stored value)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "referee"))

import envproto  # noqa: E402


class ExtraActionsFlag(unittest.TestCase):
    def test_spec_default_is_off_and_the_key_is_always_sent(self):
        self.assertIs(envproto.reset_spec("S")["extra_actions"], False)
        self.assertIs(envproto.reset_spec("S", extra_actions=True)["extra_actions"], True)
        sent = []
        env = envproto.GameEnv.__new__(envproto.GameEnv)
        env._send = sent.append
        env._next = lambda timeout: ({"t": "decide"}, False)
        env.reset("S1")
        env.reset("S2", extra_actions=True)
        self.assertEqual([c["spec"]["extra_actions"] for c in sent], [False, True])

    def test_play_turns_it_on_and_record_smoke_replay_leave_it_off(self):
        sent = []

        class E:
            def reset(self, *a, **k):
                sent.append(k)
                return {}, False

        envproto._Timed(E(), 5).reset("S")
        self.assertIs(sent[0]["extra_actions"], True)
        p = envproto.parser()
        for mode in ("record", "replay", "smoke"):
            self.assertIs(p.parse_args([mode, "--game", "g", "--out", "o"]).extra_actions, False)
        self.assertIs(p.parse_args(["record", "--game", "g", "--out", "o", "--extra-actions"]).extra_actions, True)

    def test_record_stores_the_flag_and_replay_reads_it(self):
        with tempfile.TemporaryDirectory() as d:
            envproto.write_record_meta(Path(d), "FID1", "SILENT", True)
            envproto.write_record_meta(Path(d), "FID2", "IRONCLAD", False)
            self.assertEqual(envproto.read_record_meta(Path(d), "FID1"), {"character": "SILENT", "extra_actions": True, "full_select": False})
            self.assertEqual(envproto.read_record_meta(Path(d), "FID2")["extra_actions"], False)
            self.assertEqual(envproto.read_record_meta(Path(d), "OLD"), {"character": "IRONCLAD", "extra_actions": False, "full_select": False})  # no sidecar: off
            self.assertEqual(list(Path(d).glob("rec-*.json")), [])  # the sidecar is not mistaken for a record


if __name__ == "__main__":
    unittest.main()
