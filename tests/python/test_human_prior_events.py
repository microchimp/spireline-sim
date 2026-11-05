"""Event rows of the human deck prior (rl/human_prior.py `build --events`; rl/event_keys.py; docs/rl/HUMAN_PRIOR.md).

The pinned facts: the default build has no event rows; with the switch, each option page of an event visit whose recorded key the option
table places is one decision over the page's shown options, as `event.option` afterstates (event id, index, step); the chosen option is the
index the table gives the key (This or That's ORNATE is the second option); a series is one decision per page with the step counted; the
conveyor's grab and Leave are on pages the record names differently and still pair up; an option the state locks is not offered; a key
the table cannot place (a rolled page, a mod's event) or a page with one option is dropped and counted by reason; the rows have the run
judgement's shape and a shard round-trips. A synthetic run record stands in for the corpus; the real corpus is not read here.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import human_prior as hp  # noqa: E402
import human_starts as hs  # noqa: E402
from test_human_prior import _find_ids, _point, _stats, fake_donors, fake_judge, synthetic_run  # noqa: E402


def _ekey(event, page, option):
    return {"title": {"key": f"{event}.pages.{page}.options.{option}.title", "table": "events"}}


def _event_point(event, choices, gold=99, hp_=60, **extra):
    stats = _stats(hp_, gold=gold, event_choices=[_ekey(event, p, o) for p, o in choices], **extra)
    return _point("unknown", stats, [{"model_id": f"EVENT.{event}", "room_type": "event"}])


def event_run(rid: str, gold: int = 99) -> dict:
    """A run with an ancient and then events: one choice (This or That), a series (Abyssal Baths: Immerse, Linger, Exit), a gated page
    (Zen Weaver), a one-option page and rolled pages (Tinker Time), a conveyor dish (the grab and the Leave sit on differently named
    pages), another mod's event, and Fake Merchant (no option choice)."""
    base = synthetic_run(rid)
    ancient = base["map_point_history"][0][0]
    act1 = [
        ancient,
        _event_point("THIS_OR_THAT", [("INITIAL", "ORNATE")], gold=gold),
        _event_point("ABYSSAL_BATHS", [("INITIAL", "IMMERSE"), ("ALL", "LINGER"), ("ALL", "EXIT_BATHS")], gold=gold),
        _event_point("ZEN_WEAVER", [("INITIAL", "BREATHING_TECHNIQUES")], gold=gold),
        _event_point("TINKER_TIME", [("INITIAL", "CHOOSE_CARD_TYPE"), ("CHOOSE_CARD_TYPE", "SKILL"), ("CHOOSE_RIDER", "WISDOM")], gold=gold),
        _event_point("ENDLESS_CONVEYOR", [("ALL", "CAVIAR"), ("GRAB_SOMETHING_OFF_THE_BELT", "LEAVE")], gold=gold),
        _event_point("INTEGRATEDSTRATEGYEVENTS-GLIMPSE_EVENT", [("INITIAL", "X")], gold=gold),
        _event_point("FAKE_MERCHANT", [], gold=gold),
        _event_point("BRAIN_LEECH", [("INITIAL", "RIP")], gold=gold, hp_=55, damage_taken=5, cards_gained=[{"id": "CARD.ANGER"}],
                     card_choices=[{"card": {"id": f"CARD.{c}"}, "was_picked": c == "ANGER"} for c in ("ANGER", "BODY_SLAM", "HEADBUTT")]),
        _point("rest_site", _stats(80, rest_site_choices=["HEAL"], hp_healed=20)),
    ]
    base["map_point_history"] = [act1]
    return base


class HumanPriorEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hp.stack()
        cls.t = hp.Tables()
        cls.rid = _find_ids(False, 1, "e")[0]
        cls.slug = hp.stack().V.event_id_by_slug()

    def build(self, gold=99, events=True):
        stats, seen, price = Counter(), Counter(), Counter()
        specs = hp.run_decisions(event_run(self.rid, gold), self.t, stats, seen, price, events)
        return [s for s in specs if s["kind"] == "event"], specs, stats, seen

    def test_the_default_build_has_no_event_rows(self):
        ev, _, stats, seen = self.build(events=False)
        self.assertEqual(ev, [])
        self.assertFalse(any("event" in k for k in seen))
        self.assertFalse(any("|event|" in k for k in stats))

    def test_a_single_choice_page_is_a_decision_over_its_shown_options(self):
        ev, *_ = self.build()
        this = ev[0]
        eid = self.slug["THIS_OR_THAT"]
        self.assertEqual([o["event"] for o in this["options"]], [[eid, 0, 0], [eid, 1, 0]])
        self.assertEqual(this["chosen"], 1)  # ORNATE is the second option ("That")
        self.assertEqual((this["kind"], this["aux"], this["aux2"]), ("event", eid, 0))
        self.assertEqual(this["gold"], 99)  # the gold walking in: the floor before's

    def test_a_series_is_one_decision_per_page_with_the_step_counted(self):
        ev, *_ = self.build()
        bath = [s for s in ev if s["aux"] == self.slug["ABYSSAL_BATHS"]]
        self.assertEqual([s["aux2"] for s in bath], [0, 1, 2])
        self.assertEqual([[o["event"][1] for o in s["options"]] for s in bath], [[0, 1]] * 3)
        self.assertEqual([s["chosen"] for s in bath], [0, 0, 1])  # Immerse, Linger, Exit Baths
        self.assertEqual([[o["event"][2] for o in s["options"]] for s in bath], [[0, 0], [1, 1], [2, 2]])

    def test_the_conveyor_names_its_pages_by_the_option_not_the_screen(self):
        ev, *_ = self.build()
        belt = [s for s in ev if s["aux"] == self.slug["ENDLESS_CONVEYOR"]]
        self.assertEqual([(s["chosen"], len(s["options"])) for s in belt], [(0, 2), (1, 2)])  # the grab on page ALL; Leave on its own

    def test_an_option_the_state_locks_is_not_offered(self):
        zen = self.slug["ZEN_WEAVER"]
        ev, _, stats, _ = self.build(gold=99)
        self.assertEqual([s for s in ev if s["aux"] == zen], [])  # at 99 gold only Breathing Techniques is on offer: not a decision
        self.assertEqual(stats["drop|act1|event|page with one option on offer"], 1)
        ev, *_ = self.build(gold=300)
        page = next(s for s in ev if s["aux"] == zen)
        self.assertEqual([o["event"][1] for o in page["options"]], [0, 1, 2])
        self.assertEqual(page["chosen"], 0)

    def test_what_the_table_cannot_place_is_dropped_and_counted_by_reason(self):
        ev, _, stats, seen = self.build()
        self.assertEqual([s for s in ev if s["aux"] == self.slug["TINKER_TIME"]], [])
        self.assertEqual(stats["drop|act1|event|page with one option"], 1)  # Tinker Time's first page has one option
        self.assertEqual(stats["drop|act1|event|page with rolled options"], 2)  # the type and the rider
        self.assertEqual(stats["drop|act1|event|event of another mod, not in the table"], 1)
        self.assertEqual(seen["act1|event"], 7)  # visits with a recorded option (Fake Merchant has none)
        self.assertEqual(stats["event_decisions|act1"], len(ev))
        self.assertEqual(len(ev), 1 + 3 + 2 + 1)  # This or That, Abyssal Baths x3, the conveyor x2, Brain Leech

    def test_the_other_kinds_are_the_same_with_the_switch_on_and_off(self):
        # Brain Leech's point holds a card reward too: it is built on the state after the event, event rows on the state walking in
        _, off_specs, *_ = self.build(events=False)
        ev, on_specs, *_ = self.build(events=True)
        leech = next(s for s in ev if s["aux"] == self.slug["BRAIN_LEECH"])
        others_on = [s for s in on_specs if s["kind"] != "event"]
        self.assertEqual(json.dumps(off_specs, sort_keys=True, default=list), json.dumps(others_on, sort_keys=True, default=list))
        self.assertTrue(any(s["kind"] == "card" and s["floor"] == leech["floor"] for s in others_on))
        card = next(s for s in others_on if s["kind"] == "card" and s["floor"] == leech["floor"])
        self.assertEqual((leech["hp"], leech["gold"]), (60, 99))  # the floor before's
        self.assertEqual((card["hp"], card["gold"]), (55, 99))  # the point's own, as without the switch

    def test_rows_are_afterstates_of_distinct_event_slots(self):
        ev, *_ = self.build()
        F = hp.stack().F
        for sp in ev:
            rows = hp.spec_rows(sp)
            self.assertEqual(len(rows), len(sp["options"]))
            self.assertTrue(all(r.shape == (F.BASE_SIZE,) for r in rows))
            self.assertFalse(np.array_equal(rows[0], rows[1]))  # the option's slot moves the afterstate

    def test_an_event_shard_round_trips_and_other_kinds_are_unchanged(self):
        ev, specs, *_ = self.build()
        donors = fake_donors()
        with tempfile.TemporaryDirectory() as tmp:
            for sp in specs:
                sp["run_ix"] = 0
            path = Path(tmp) / "shard-00000.npz"
            n_rows = hp.write_shard(path, specs, [self.rid], fake_judge(), hp.donor_index(donors), donors)
            self.assertEqual(n_rows, sum(len(s["options"]) for s in specs))
            ds = hp.Dataset.load(Path(tmp), [hs.exam_bucket(self.rid)], donors)
            self.assertEqual(len(ds.kind_idx["event"]), len(ev))
            self.assertEqual({hp.KIND_NAME[int(k)] for k in ds.dec[:, hp.C["kind"]]}, {"ancient", "rest", "card", "event"})

    def test_the_build_job_carries_the_events_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            hp._WORK.update(t=self.t, judge=fake_judge(), donors=None, dindex={}, exam=frozenset())
            try:
                line = json.dumps(event_run(self.rid))
                off = hp.build_chunk({"lines": [line], "out": str(Path(tmp) / "shard-00000.npz")})
                on = hp.build_chunk({"lines": [line], "out": str(Path(tmp) / "shard-00001.npz"), "events": True})
            finally:
                hp._WORK.clear()
        self.assertEqual(off["seen"].get("act1|event", 0), 0)
        self.assertEqual(on["seen"]["act1|event"], 7)
        self.assertGreater(on["stats"]["decisions"], off["stats"]["decisions"])


if __name__ == "__main__":
    unittest.main()
