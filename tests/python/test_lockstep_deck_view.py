"""The copy's deck past the observation's 64 cards (Act 3 week, lane 21): `EmuLock._deck_view` reads the whole deck from the run when
the observation's deck block is full. Stubs, and the native library (skipped without it)."""

import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))

import leakprobes  # noqa: E402
import lockstep  # noqa: E402

try:
    leakprobes.native_env()
    from sts2_gym import native  # noqa: E402,F401

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    HAVE_NATIVE = False


class StubNative:
    RUN_OBS_LAYOUT = {"max_deck": 4}

    def __init__(self, deck, in_fight=None):
        self.deck, self.in_fight = deck, len(deck) if in_fight is None else in_fight

    def run_info(self, handle):
        return [0, 0, 0, self.in_fight]

    def run_state_list(self, handle, list_id, capacity):
        assert list_id == 0
        return tuple(self.deck[:capacity])


def stub_lock(deck, in_fight=None):
    slug = {1: "STRIKE_IRONCLAD", 2: "DEFEND_IRONCLAD", 3: "BASH"}
    e = SimpleNamespace(native=StubNative(deck, in_fight), s=SimpleNamespace(handle=7), slug={"cards": slug})
    e._card = lambda c: [slug[c.card_id], int(c.upgraded)]
    return e


def ref(card_id, up=False):
    return SimpleNamespace(card_id=card_id, upgraded=up)


class DeckPastTheObservationTests(unittest.TestCase):
    def test_under_the_limit_the_observations_deck_stands(self):
        e = stub_lock([1, 2, -3])
        self.assertEqual(lockstep.EmuLock._deck_view(e, [ref(1), ref(2), ref(3, True)]),
                         [["STRIKE_IRONCLAD", 0], ["DEFEND_IRONCLAD", 0], ["BASH", 1]])

    def test_at_the_limit_the_whole_deck_is_read_from_the_run(self):
        e = stub_lock([1, 1, 2, 2, -3, 3])
        got = lockstep.EmuLock._deck_view(e, [ref(1), ref(1), ref(2), ref(2)])
        self.assertEqual(Counter(map(tuple, got)), Counter([("STRIKE_IRONCLAD", 0)] * 2 + [("DEFEND_IRONCLAD", 0)] * 2
                                                           + [("BASH", 1), ("BASH", 0)]))

    def test_a_card_out_of_the_deck_in_a_fight_keeps_the_observations_cards(self):
        e = stub_lock([1, 1, 2, 2, 3, 3], in_fight=5)  # a stolen card: the run's deck and the fight's deck differ
        self.assertEqual(len(lockstep.EmuLock._deck_view(e, [ref(1), ref(1), ref(2), ref(2)])), 4)

    def test_a_library_without_the_list_keeps_the_observations_cards(self):
        e = stub_lock([1, 1, 2, 2, 3])
        e.native.run_state_list = None  # not callable: a stub of an older build
        self.assertEqual(len(lockstep.EmuLock._deck_view(e, [ref(1), ref(1), ref(2), ref(2)])), 4)


@unittest.skipUnless(HAVE_NATIVE, "needs the native emulator library")
class DeckPastTheObservationNativeTests(unittest.TestCase):
    def test_a_deck_of_70_cards_is_compared_whole(self):
        e = lockstep.EmuLock("QS2GYXRKWN", sims=2)
        self.addCleanup(e.close)
        cap = e.native.RUN_OBS_LAYOUT["max_deck"]
        strike = e.ids["cards"]["STRIKE_IRONCLAD"]
        bash = e.ids["cards"]["BASH"]
        cards = [(strike, 0, 0, 0)] * (cap - 4) + [(bash, 1, 0, 0)] * 10
        e.native.run_debug_set_player(e.s.handle, e._obs(), cards=cards)
        e.s.view()  # the observation the view reads is the one the write refreshed
        deck = e.view()["deck"]
        self.assertEqual(len(deck), cap + 6)
        self.assertEqual(Counter(map(tuple, deck)), Counter({("STRIKE_IRONCLAD", 0): cap - 4, ("BASH", 1): 10}))
        game = {"act": 0, "floor": e.view()["floor"], "hp": e.view()["hp"], "max_hp": e.view()["max_hp"], "gold": e.view()["gold"],
                "deck": [["STRIKE_IRONCLAD", 0]] * (cap - 4) + [["BASH", 1]] * 10, "relics": [], "potions": []}
        diff = [d for d in lockstep.structural_diff(lockstep.ref_view(game), e.view()) if d[0] == ".deck"]
        self.assertEqual(diff, [])


if __name__ == "__main__":
    unittest.main()
