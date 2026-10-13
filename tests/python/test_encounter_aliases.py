"""The encounter name table keeps a weak encounter and its normal twin apart.

`scripts/generate_encounter_tags.py` maps each of the game's encounter models to the
emulator's `ActOneEncounter` entry. Corpse Slugs and Seapunk used to be folded into one
entry per pair, so a weak fight built with the normal roster (or the reverse) compared
equal by name and only the roster could tell -- and every comparison that goes through
this table (`replay_real_run`, the Codex walker's copy of it) was blind to it.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

tags = importlib.import_module("generate_encounter_tags")

# Every encounter an act deals as both a weak and a normal fight, by game model name
# (the Weak and Normal classes of MegaCrit.Sts2.Core.Models.Encounters). Tunneler's
# normal model is dealt by no act, but it is a pair all the same.
PAIRS = [
    ("NibbitsWeak", "NibbitsNormal"),
    ("SlimesWeak", "SlimesNormal"),
    ("CorpseSlugsWeak", "CorpseSlugsNormal"),
    ("SeapunkWeak", "SeapunkNormal"),
    ("BowlbugsWeak", "BowlbugsNormal"),
    ("ExoskeletonsWeak", "ExoskeletonsNormal"),
    ("ScrollsOfBitingWeak", "ScrollsOfBitingNormal"),
    ("TunnelerWeak", "TunnelerNormal"),
]


class WeakAndNormalNameTests(unittest.TestCase):
    def test_a_weak_encounter_and_its_normal_twin_have_two_names(self) -> None:
        for weak, normal in PAIRS:
            with self.subTest(pair=(weak, normal)):
                self.assertNotEqual(
                    tags.emulator_name(weak), tags.emulator_name(normal)
                )

    def test_and_two_ids(self) -> None:
        ids = tags.encounter_ids()
        for weak, normal in PAIRS:
            with self.subTest(pair=(weak, normal)):
                self.assertNotEqual(
                    ids[tags.emulator_name(weak)],
                    ids[tags.emulator_name(normal)],
                )

    def test_the_normal_slugs_and_seapunk_are_named_for_their_models(self) -> None:
        self.assertEqual(tags.emulator_name("CorpseSlugsNormal"), "CorpseSlugsNormal")
        self.assertEqual(tags.emulator_name("SeapunkNormal"), "SeapunkNormal")
        self.assertEqual(tags.emulator_name("CorpseSlugsWeak"), "CorpseSlugs")
        self.assertEqual(tags.emulator_name("SeapunkWeak"), "Seapunk")


if __name__ == "__main__":
    unittest.main()
