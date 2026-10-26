"""tools/run_viewer.py: a run record (the driver's run.json) rendered as one self-contained HTML page (no game, no emulator needed).

The fixture is a real generation-5 record cut down to the keys the viewer reads (fixtures/run_viewer_record.json): every screen
the viewer knows, a boss fight and the fight the run was lost in.
"""

from __future__ import annotations

import getpass
import json
import platform
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import run_viewer as rv  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "run_viewer_record.json"


def record() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class RenderSections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = record()
        cls.page = rv.render(cls.rec, title="Fixture run", generation=5, note="A replay note.")

    def test_every_section_is_present(self):
        want = ["Fixture run", "Generation: <b>5</b>", "A replay note.", "Seed: <b>", "Floor reached: <b>", "How it ended: <b>",
                "The run at a glance", "<svg", "Numbers per floor", "Floor by floor", "Act 1", "Act 2",
                "Neow&#x27;s offer", "<b>Map</b> - paths offered", "chose the", "Turn by turn", "Player HP", "Played",
                "<b>Rewards</b>", "took <b>", "<b>Rest site</b>", "<b>Shop</b>", "bought", "left with", "<b>Event:", "<b>Treasure chest</b>",
                "<b>Crystal sphere</b>", "<b>Next act</b>", 'class="tag Boss"', 'class="tag Elite"' if False else 'class="tag Monster"',
                "result <b", "potions used", "The end", "The run ended:", "Final deck", "Relics:"]
        for w in want:
            self.assertIn(w, self.page, w)

    def test_boss_and_lost_fights_open_normal_fights_folded(self):
        self.assertRegex(self.page, r"<details open><summary>Turn by turn \(\d+ turns")
        self.assertRegex(self.page, r"<details><summary>Turn by turn \(\d+ turns")

    def test_death_is_named(self):
        last = self.rec["decisions"][-1]
        self.assertEqual(last["result"], "death")
        self.assertIn(f"{self.rec['end']} in the fight against {rv.enc_name(last['encounter'])}", self.page)
        self.assertIn('class="floor end"', self.page)

    def test_run_strip_has_one_point_per_floor(self):
        secs = rv.sections(self.rec["decisions"])
        self.assertEqual(self.page.count("<circle"), len(secs))
        self.assertEqual(self.page.count('<section class="floor'), len(secs))

    def test_names_are_readable(self):
        self.assertEqual(rv.pretty("RELIC.BURNING_BLOOD"), "Burning Blood")
        self.assertEqual(rv.card_name(["STRIKE_IRONCLAD", 1]), "Strike+")
        self.assertEqual(rv.card_name(["BASH", 0]), "Bash")
        self.assertEqual(rv.option_label(["TABLET_OF_TRUTH.pages.INITIAL.options.DECIPHER_1", False]), "Decipher 1")
        self.assertEqual(rv.enc_name("TOADPOLES_WEAK"), "Toadpoles")
        self.assertNotIn("STRIKE_IRONCLAD", self.page)

    def test_fight_turns_group_by_end_turn(self):
        fight = next(d for d in self.rec["decisions"] if d["screen"] == "combat")
        turns = rv.fight_turns(fight["fight"]["steps"])
        ends = sum(1 for s in fight["fight"]["steps"] if (s.get("action") or {}).get("kind") == "end_turn")
        self.assertEqual(len(turns), ends + 1)
        self.assertTrue(all(t["hp_end"] is not None for t in turns))
        plays = sum(1 for s in fight["fight"]["steps"] if (s.get("action") or {}).get("kind") in ("play", "potion"))
        self.assertEqual(sum(len(t["plays"]) for t in turns), plays)


class SelfContained(unittest.TestCase):
    def test_no_path_host_user_or_external_asset(self):
        page = rv.render(record(), title="Fixture run")
        self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", page), "a drive path")
        self.assertIsNone(re.search(r"/(Users|home|mnt|tmp)/", page), "a user or temp path")
        self.assertNotIn("\\\\", page)
        for bad in ("http://", "https://", "<script", "<link", "<img", "@import", "url("):
            self.assertNotIn(bad, page, bad)
        for who in {platform.node(), getpass.getuser()}:
            if len(who) >= 4:
                self.assertNotIn(who.lower(), page.lower(), "host or user name")

    def test_markup_in_names_is_escaped(self):
        rec = record()
        rec["decisions"][7]["event"] = "<script>alert(1)</script>"
        page = rv.render(rec)
        self.assertNotIn("<script>alert", page)
        self.assertIn("&lt;", page)

    def test_partial_records_still_render(self):
        for rec in ({}, {"decisions": []}, {"decisions": [{"i": 0, "screen": "combat"}, {"i": 1, "screen": "weird", "choice": {"a": 1}}]},
                    {"decisions": [{"i": 0, "screen": "map", "floor": 1, "offer": [[1, 1, "Monster"]], "choice": [1, 1]}]}):
            page = rv.render(rec)
            self.assertIn("<html", page)
            self.assertIn("The end", page)


class Annotations(unittest.TestCase):
    def test_none_means_no_notes(self):
        self.assertNotIn('<div class="note">', rv.render(record(), annotations={}))

    def test_notes_render_under_their_decision(self):
        rec = record()
        ids = [d["i"] for d in rec["decisions"]]
        a, b, c = ids[3], ids[5], ids[-1]
        ann = {str(a): "Solver: this line wins in 2 turns.", b: {"title": "Regret 0.4", "text": "Taking the card is better."},
               str(c): ["first line", "second line"], "9999": "no such decision"}
        page = rv.render(rec, annotations=ann)
        self.assertEqual(page.count('<div class="note">'), 3)
        for text in ("Solver: this line wins in 2 turns.", "<b>Regret 0.4</b>Taking the card is better.", "first line\nsecond line"):
            self.assertIn(text, page)
        self.assertNotIn("no such decision", page)
        i = page.index(f'id="d{a}"')
        self.assertLess(i, page.index("Solver: this line wins"))
        self.assertLess(page.index("Solver: this line wins"), page.index(f'id="d{ids[4]}"'))
        self.assertIn("3 annotation(s)", page)


class CommandLine(unittest.TestCase):
    def test_writes_the_page_with_annotations(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ann = td / "ann.json"
            ann.write_text(json.dumps({"3": "a note"}), encoding="utf-8")
            out = td / "page.html"
            rc = rv.main(["--record", str(FIXTURE), "--out", str(out), "--title", "T", "--generation", "5", "--note", "N",
                          "--annotations", str(ann)])
            self.assertEqual(rc, 0)
            page = out.read_text(encoding="utf-8")
            self.assertIn("<title>T</title>", page)
            self.assertIn("a note", page)
            self.assertNotIn(str(td), page)


if __name__ == "__main__":
    unittest.main()
