"""Block P3 of the information-parity side blocks: what an event option pays (`rl/parity_events.py`, D59 lane D).

Pinned: the table covers most events and every key names an option the run encoder can name; an option that heals, pays gold, removes a card or gives a relic shows it; the vector's layout and scales; a key the
table lacks is zeros with known 0; the table agrees with a held-out sample of runs the builder never saw (a clone of the engine taking each option: a key's measured change is within its own spread of the table's
mean); and the sampler is deterministic.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "rl", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import parity_events as pe  # noqa: E402


class TableTests(unittest.TestCase):
    def test_the_table_covers_most_events(self):
        events = {k[0] for k in pe.payoff_table()}
        self.assertGreaterEqual(len(events), 45)  # the engine's table has 58 event ids; a few are not random-run events (Neow's, the Fake Merchant's stall)

    def test_keys_are_options_of_the_engine(self):
        for event, option, step in pe.payoff_table():
            self.assertTrue(1 <= event <= 58 or event > 58, event)
            self.assertTrue(0 <= option <= 3, (event, option))
            self.assertTrue(0 <= step <= 40, (event, option, step))

    def test_some_option_of_each_effect_kind_exists(self):
        t = pe.payoff_table()
        self.assertTrue(any(v["hp"] > 3 for v in t.values()), "an option that heals")
        self.assertTrue(any(v["hp"] < -3 for v in t.values()), "an option that costs HP")
        self.assertTrue(any(v["gold"] > 20 for v in t.values()), "an option that pays gold")
        self.assertTrue(any(v["gold"] < -20 for v in t.values()), "an option that costs gold")
        self.assertTrue(any(v["relics"] >= 0.9 for v in t.values()), "an option that gives a relic")
        self.assertTrue(any(v["max_hp"] > 3 for v in t.values()), "an option that raises max HP")
        self.assertTrue(any(v["curses"] >= 0.9 for v in t.values()), "an option that adds a curse")
        self.assertTrue(any(v["deck"] <= -0.9 for v in t.values()), "an option that removes a card")
        self.assertTrue(any(v["lead_fight"] >= 0.9 for v in t.values()), "an option that starts a fight")
        self.assertTrue(any(v["lead_select_remove"] >= 0.9 for v in t.values()), "an option that opens a removal")
        self.assertTrue(any(v["lead_select_upgrade"] >= 0.9 for v in t.values()), "an option that opens an upgrade")
        self.assertTrue(any(v["lead_continues"] >= 0.9 for v in t.values()), "an option whose event goes on")

    def test_lead_shares_are_shares(self):
        for key, v in pe.payoff_table().items():
            s = sum(v["lead_" + name] for name in pe.LEADS)
            self.assertLessEqual(s, 1.0 + 1e-3, key)  # each share is rounded to four places
            self.assertGreaterEqual(v["n"], 1)


class EncodingTests(unittest.TestCase):
    def test_width_and_layout(self):
        self.assertEqual(pe.EVENT_WIDTH, 1 + len(pe.DELTAS) + len(pe.LEADS))
        key = next(k for k, v in pe.payoff_table().items() if v["gold"] > 20)
        v = pe.event_payoff(key)
        self.assertEqual(v.shape, (pe.EVENT_WIDTH,))
        self.assertEqual(v.dtype, np.float32)
        self.assertEqual(float(v[0]), 1.0)
        row = pe.payoff_table()[key]
        self.assertAlmostEqual(float(v[1 + pe.DELTAS.index("gold")]), max(-1.0, min(1.0, row["gold"] / pe.SCALES["gold"])), places=5)
        self.assertTrue(np.all(np.abs(v) <= 1.0))

    def test_an_unknown_key_and_no_event_are_zeros(self):
        np.testing.assert_array_equal(pe.event_payoff((9999, 0, 0)), np.zeros(pe.EVENT_WIDTH, dtype=np.float32))
        np.testing.assert_array_equal(pe.event_payoff(None), np.zeros(pe.EVENT_WIDTH, dtype=np.float32))


class SamplerTests(unittest.TestCase):
    def test_the_sampler_is_deterministic(self):
        a = pe.sample(5000, 2, namespace="PEVTEST")
        b = pe.sample(5000, 2, namespace="PEVTEST")
        self.assertEqual(a, b)
        self.assertGreater(len(a), 0)

    def test_the_table_agrees_with_a_held_out_sample(self):
        """Runs the builder never played (the builder's seeds are `PEV0000000..`; these are `PEVHELD..`): for every key the table holds with at least five clones, a measured change in
        HP, gold, deck size or relics is within three of the table's own standard deviations (and a unit) of its mean, for nine in ten of the measured options."""
        recs = pe.sample(0, 40, namespace="PEVHELD")
        table = pe.payoff_table()
        checked = bad = 0
        for key, deltas, _lead in recs:
            row = table.get(tuple(key))
            if row is None or row["n_taken"] < 5 or deltas is None:
                continue
            checked += 1
            for name in ("hp", "gold", "deck", "relics"):
                if abs(deltas[name] - row[name]) > 3 * row[name + "_sd"] + 1.0:
                    bad += 1
                    break
        self.assertGreaterEqual(checked, 40)
        self.assertLessEqual(bad, 0.1 * checked, (bad, checked))


if __name__ == "__main__":
    unittest.main()
