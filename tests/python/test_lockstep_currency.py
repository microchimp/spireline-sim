"""The real-game player on a run-currency config (`a3-v1-r`, rl/run_currency.py): `EmuLock._choose_combat` builds the fight's currency from
the emulator's run view at the fight's first combat decision, keeps it for the fight and passes it to the search as `terminal`; on a config
without `-r` the search is asked for exactly the keywords it always was and the currency is never touched.

No game is played: the emulator side is a canned stand-in, the search records what it is asked.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, ROOT / "referee"):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import lockstep  # noqa: E402
import run_currency  # noqa: E402
import search_configs as sc  # noqa: E402


class Asked:
    """A `search_live` stand-in: records the keywords of each call and answers with the first label's move."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search_live(self, handle, **kw):
        self.calls.append(dict(kw))
        return NS(best=NS(action=0, target=-1), ai_state_assumed=False)


def emu(cfg_name: str, floor: int = 5, encounter: int = 7, hp: int = 60):
    """An EmuLock with a canned run view and a root whose handle is a number (no native library)."""
    e = lockstep.EmuLock.__new__(lockstep.EmuLock)
    e.asked = Asked()
    e.cfg = sc.get(cfg_name)
    e.ls = e.asked
    e.rng = np.random.default_rng(3)
    e.native = NS(destroy=lambda handle: None)
    e.position_of = lambda root, info: None
    state = {"floor": floor, "enc": encounter, "hp": hp}
    e.state = state
    root_of = lambda: NS(handle=1, encounter_id=state["enc"])  # noqa: E731
    e.s = NS(combat=lambda fork_combat: NS(combat_fork=lambda _: root_of()), info=state, act_index=0,
             view=lambda: NS(hp=state["hp"], floor=state["floor"]))
    e._key = lambda action, target: ("end",)
    e._action_of = lambda label, msg=None: (0, -1)
    return e


class RunCurrencyConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        os.environ.pop("SPIRELINE_GEN_DIR", None)  # the prior is not under test
        self.built: list = []

        def for_view(view, head=None, **kw):
            cur = NS(view=view, head=head)
            self.built.append(cur)
            return cur

        self.head = object()
        p1 = mock.patch.object(run_currency.FightCurrency, "for_view", staticmethod(for_view))
        p2 = mock.patch.object(run_currency, "default_head", lambda: self.head)
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        self.addCleanup(self.env.stop)

    def test_the_search_is_given_the_fights_currency_built_from_the_view_at_its_first_decision(self):
        e = emu("a3-v1-r", hp=60)
        self.assertEqual(e._choose_combat("combat", [["end"]], {}), 0)
        self.assertEqual(len(self.built), 1)
        self.assertEqual(self.built[0].view.hp, 60)
        self.assertIs(self.built[0].head, self.head)
        self.assertIs(e.asked.calls[0]["terminal"], self.built[0])
        # the fight goes on: HP falls, the currency stays the one built as the fight opened (and so does its memo)
        e.state["hp"] = 35
        e._choose_combat("combat", [["end"]], {})
        self.assertEqual(len(self.built), 1)
        self.assertIs(e.asked.calls[1]["terminal"], self.built[0])
        self.assertEqual(e.currency_fights, 1)

    def test_a_new_fight_gets_a_new_currency(self):
        e = emu("a3-v1-r", floor=5, encounter=7, hp=60)
        e._choose_combat("combat", [["end"]], {})
        e.state.update(floor=6, enc=9, hp=48)
        e._choose_combat("combat", [["end"]], {})
        self.assertEqual(len(self.built), 2)
        self.assertEqual(self.built[1].view.hp, 48)
        self.assertIs(e.asked.calls[1]["terminal"], self.built[1])
        self.assertEqual(e.currency_fights, 2)

    def test_a_card_screen_of_the_fight_shares_the_fights_currency(self):
        e = emu("a3-v1-r")
        e._choose_combat("combat", [["end"]], {})
        e._select_action = lambda label: 0
        e._choose_combat("select", [["STRIKE", 0]], {})
        self.assertEqual(len(self.built), 1)
        self.assertIs(e.asked.calls[1]["terminal"], self.built[0])

    def test_the_other_search_keywords_are_the_configs_own(self):
        e = emu("a3-v1-r")
        e._choose_combat("combat", [["end"]], {})
        want = e.cfg.search_kwargs(None, terminal=self.built[0])
        self.assertEqual(e.asked.calls[0], {**want, "seed": e.asked.calls[0]["seed"]})
        self.assertTrue(e.cfg.run_currency)

    def test_the_lab_asks_for_the_config_by_name_and_gets_the_currency_config(self):
        self.assertTrue(lockstep.search_config("a3-v1-r").run_currency)
        self.assertFalse(lockstep.search_config("a3-v1").run_currency)

    def test_the_old_player_would_have_raised_on_the_config(self):
        with self.assertRaises(ValueError):
            sc.get("a3-v1-r").search_kwargs(None)


class PlainConfigUnchangedTest(unittest.TestCase):
    def setUp(self) -> None:
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("SPIRELINE_GEN_DIR", None)

    def test_a_config_without_r_asks_for_the_keywords_it_always_did_and_never_touches_the_currency(self):
        for name in ("a3-v1", "judge-v0", "a3-v1-k"):
            with mock.patch.object(run_currency.FightCurrency, "for_view", side_effect=AssertionError("the currency was built")), \
                    mock.patch.object(run_currency, "default_head", side_effect=AssertionError("the head was read")):
                e = emu(name)
                for _ in range(3):
                    self.assertEqual(e._choose_combat("combat", [["end"]], {}), 0)
            kw = e.asked.calls[0]
            self.assertNotIn("terminal", kw)
            self.assertEqual({k: v for k, v in kw.items() if k != "seed"}, e.cfg.search_kwargs(None))
            self.assertEqual(e.currency_fights, 0)
            self.assertIsNone(e._currency)

    def test_the_same_seeds_give_the_same_search_requests_with_and_without_the_new_code_path(self):
        a, b = emu("a3-v1"), emu("a3-v1")
        for e in (a, b):
            for _ in range(4):
                e._choose_combat("combat", [["end"]], {})
        self.assertEqual(a.asked.calls, b.asked.calls)
        self.assertEqual([c["seed"] for c in a.asked.calls], [c["seed"] for c in b.asked.calls])


if __name__ == "__main__":
    unittest.main()
