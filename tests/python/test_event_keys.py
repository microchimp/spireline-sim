"""Event option keys of the human run record to the copy's option index (rl/event_keys.py, data/event_option_titles.json).

The pinned facts: every title the live captures show resolves to the index it has in the capture (the game is the arbiter); the
record's option names resolve to the same indices; titles that carry a number or a dish match by pattern; a name written out beats a
pattern that also fits it; a rolled page resolves only when the caller says what was offered; and anything the table does not know
(an unknown event, an unknown name, a name of another event) is None, never a guess. The coverage count classifies a visit by its
weakest key. Nothing here reads the corpus.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "rl", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_event_option_titles as gen  # noqa: E402
import event_keys as ek  # noqa: E402

TABLE = ek.EventTable.load()
CAPTURES = gen.load_captures()
# Captured titles that are rolled per visit, so no fixed index belongs to them (the dolls, the Tinker Time types).
ROLLED_TITLES = {
    "DOLL_ROOM": {"Daughter of the Wind", "Mr. Struggles", "Bing Bong"},
    "TINKER_TIME": {"Gadget", "Weapon", "Contraption"},
}


def _captured_options():
    """(event, index, title) of every unlocked, non-proceed option on every captured page."""
    for event, cap in CAPTURES.items():
        pages = [cap["page1"] or []] + [info["options"] for info in cap["after"].values()]
        for page in pages:
            for opt in page:
                if opt["is_locked"] or opt["is_proceed"] or opt["title"] in ("Locked", "Proceed"):
                    continue
                yield event, opt["index"], opt["title"]


class CaptureTitles(unittest.TestCase):
    def test_every_capture_title_resolves_to_its_own_index(self):
        seen = 0
        for event, index, title in _captured_options():
            if title in ROLLED_TITLES.get(event, ()):
                continue
            with self.subTest(event=event, title=title):
                self.assertEqual(TABLE.resolve(event, title), index)
            seen += 1
        self.assertGreater(seen, 100)

    def test_rolled_titles_have_no_fixed_index(self):
        for event, titles in ROLLED_TITLES.items():
            for title in titles:
                self.assertIsNone(TABLE.resolve(event, title), (event, title))

    def test_fifty_two_captured_events_and_the_table_covers_them(self):
        self.assertGreaterEqual(len(CAPTURES), 52)
        self.assertEqual(set(CAPTURES) - set(TABLE.by_name), set())

    def test_a_stale_options_capture_is_flagged_and_the_opt_captures_win(self):
        # JungleMazeAdventure-options.json holds another event's page; the -opt captures name Solo Quest 0 and Join Forces 1
        self.assertEqual(TABLE.resolve("JUNGLE_MAZE_ADVENTURE", "Solo Quest"), 0)
        self.assertEqual(TABLE.resolve("JUNGLE_MAZE_ADVENTURE", "Join Forces"), 1)
        self.assertIn("another event", TABLE.by_name["JUNGLE_MAZE_ADVENTURE"].get("note", ""))
        self.assertIsNone(TABLE.resolve("JUNGLE_MAZE_ADVENTURE", "Observe the Chef"))


class RecordKeys(unittest.TestCase):
    def test_full_key_bare_name_page_option_and_event_id_forms_agree(self):
        full = "ABYSSAL_BATHS.pages.INITIAL.options.IMMERSE.title"
        for event in ("ABYSSAL_BATHS", "AbyssalBaths", 20, "EVENT.ABYSSAL_BATHS"):
            for title in (full, "INITIAL.IMMERSE", "IMMERSE", "Immerse", "immerse"):
                with self.subTest(event=event, title=title):
                    self.assertEqual(TABLE.resolve(event, title), 0)
        self.assertEqual(TABLE.resolve("ABYSSAL_BATHS", "ABSTAIN"), 1)
        self.assertEqual(ek.resolve("ABYSSAL_BATHS", full), 0)  # the module-level function reads the shipped table

    def test_the_record_names_that_match_no_title_by_words(self):
        # (event, name, index): named by the engine's order, a capture's lock text or the record's outcomes
        for event, name, index in [
            ("THIS_OR_THAT", "PLAIN", 0), ("THIS_OR_THAT", "ORNATE", 1), ("LOST_WISP", "CLAIM", 0), ("SAPPHIRE_SEED", "EAT", 0),
            ("STONE_OF_ALL_TIME", "LIFT", 0), ("TEA_MASTER", "EMBER_TEA", 1), ("RANWID_THE_ELDER", "POTION", 0),
            ("RANWID_THE_ELDER", "RELIC", 2), ("ZEN_WEAVER", "EMOTIONAL_AWARENESS", 1), ("ZEN_WEAVER", "ARACHNID_ACUPUNCTURE", 2),
            ("WELCOME_TO_WONGOS", "BARGAIN_BIN", 0), ("WELCOME_TO_WONGOS", "MYSTERY_BOX", 2),
            ("SELF_HELP_BOOK", "READ_ENTIRE_BOOK", 2), ("LUMINOUS_CHOIR", "OFFER_TRIBUTE", 1), ("RELIC_TRADER", "BOTTOM", 2),
            ("COLOSSAL_FLOWER", "REACH_DEEPER_1", 1), ("GRAVE_OF_THE_FORGOTTEN", "CONFRONT", 0), ("ROUND_TEA_PARTY", "PICK_FIGHT", 1),
        ]:
            with self.subTest(event=event, name=name):
                self.assertEqual(TABLE.resolve(event, name), index)
                row = TABLE.describe(event, name)
                self.assertEqual(row["mapping"], "inferred")  # not claimed as a title match

    def test_title_match_rows_say_where_the_title_is(self):
        for name, event in TABLE.by_name.items():
            for page in event["pages"]:
                for row in page["options"]:
                    if row["mapping"] == "exact" and row["key"] and not row.get("key_pattern"):
                        flat = "_".join(gen.words(row["key"]))
                        title = "_".join(gen.words(row["title"] or ""))
                        with self.subTest(event=name, key=row["key"]):
                            self.assertTrue(row["title"] and (set(gen.words(row["key"])) - gen.STOP_WORDS
                                                              <= set(gen.words(row["title"])) or flat in title), row)

    def test_multi_page_events_use_the_page_the_record_names(self):
        self.assertEqual(TABLE.resolve("ABYSSAL_BATHS", "ALL.LINGER"), 0)
        self.assertEqual(TABLE.resolve("ABYSSAL_BATHS", "ALL.EXIT_BATHS"), 1)
        self.assertEqual(TABLE.resolve("TABLET_OF_TRUTH", "INITIAL.DECIPHER_1"), 0)
        self.assertEqual(TABLE.resolve("TABLET_OF_TRUTH", "DECIPHER_3.DECIPHER"), 0)
        self.assertEqual(TABLE.resolve("TABLET_OF_TRUTH", "DECIPHER.GIVE_UP"), 1)
        self.assertEqual(TABLE.resolve("TRIAL", "NOBLE.INNOCENT"), 1)
        self.assertEqual(TABLE.resolve("TRIAL", "REJECT.ACCEPT"), 0)
        self.assertEqual(TABLE.resolve("TRIAL", "REJECT.DOUBLE_DOWN"), 1)
        self.assertEqual(TABLE.resolve("DENSE_VEGETATION", "REST.FIGHT"), 0)
        self.assertEqual(TABLE.resolve("PUNCH_OFF", "I_CAN_TAKE_THEM.FIGHT"), 0)
        self.assertEqual(TABLE.resolve("COLOSSAL_FLOWER", "REACH_DEEPER_2.POLLINOUS_CORE"), 1)
        self.assertEqual(TABLE.resolve("COLOSSAL_FLOWER", "REACH_DEEPER_2.EXTRACT_INSTEAD"), 0)

    def test_the_overcome_option_keeps_its_first_page_name_on_later_pages(self):
        for key in ("INITIAL.OVERCOME", "OVERCOME", "HOLD_ON_3.OVERCOME"):
            self.assertEqual(TABLE.resolve("SLIPPERY_BRIDGE", key), 0)


class Patterns(unittest.TestCase):
    def test_a_title_with_a_number_matches_whatever_the_number(self):
        for gold in (50, 100, 137, 999):
            self.assertEqual(TABLE.resolve("RANWID_THE_ELDER", f"Give {gold} Gold"), 1)
        for n in (1, 2, 3):
            self.assertEqual(TABLE.resolve("BATTLEWORN_DUMMY", f"Setting {n}"), n - 1)
        self.assertIsNone(TABLE.resolve("BATTLEWORN_DUMMY", "Setting"))

    def test_a_counted_name_matches_whatever_the_count(self):
        for n in range(0, 7):
            self.assertEqual(TABLE.resolve("SLIPPERY_BRIDGE", f"HOLD_ON_{n}.HOLD_ON_{n + 1}"), 1, n)
        self.assertEqual(TABLE.resolve("SLIPPERY_BRIDGE", "HOLD_ON_6.HOLD_ON_LOOP"), 1)
        self.assertEqual(TABLE.resolve("SLIPPERY_BRIDGE", "HOLD_ON_LOOP.HOLD_ON_LOOP"), 1)
        for n in (1, 2, 3, 4):
            self.assertEqual(TABLE.resolve("TABLET_OF_TRUTH", f"DECIPHER_{n}.DECIPHER"), 0)

    def test_a_dish_on_the_belt_matches_by_pattern_and_by_name(self):
        for dish in ("Fried Eel", "Jelly Liver", "Caviar", "Golden Fysh"):
            self.assertEqual(TABLE.resolve("ENDLESS_CONVEYOR", f"Grab {dish} off the Belt"), 0)
        for dish in ("FRIED_EEL", "JELLY_LIVER", "CAVIAR", "CLAM_ROLL", "GOLDEN_FYSH", "SEAPUNK_SALAD", "SPICY_SNAPPY", "SUSPICIOUS_CONDIMENT"):
            self.assertEqual(TABLE.resolve("ENDLESS_CONVEYOR", f"ALL.{dish}"), 0)
        self.assertEqual(TABLE.resolve("ENDLESS_CONVEYOR", "Leave"), 1)
        self.assertEqual(TABLE.resolve("WELCOME_TO_WONGOS", "Leave"), 3)  # Wongos' door is index 3: the proceed action's number is also 3

    def test_a_name_written_out_beats_a_pattern_that_also_fits_it(self):
        # SETTING_1..3 are rows of their own; the counted-name pattern on the first row must not make SETTING_2 ambiguous
        self.assertEqual(TABLE.resolve("BATTLEWORN_DUMMY", "SETTING_2"), 1)
        self.assertEqual(TABLE.resolve("BATTLEWORN_DUMMY", "BATTLEWORN_DUMMY.pages.INITIAL.options.SETTING_3.title"), 2)


class Unresolved(unittest.TestCase):
    def test_unknown_event_name_or_key_is_none(self):
        self.assertIsNone(TABLE.resolve("NOT_AN_EVENT", "IMMERSE"))
        self.assertIsNone(TABLE.resolve(9999, "IMMERSE"))
        self.assertIsNone(TABLE.resolve("ABYSSAL_BATHS", "NOT_AN_OPTION"))
        self.assertIsNone(TABLE.resolve("ABYSSAL_BATHS", "Not a title"))
        self.assertIsNone(TABLE.resolve("ABYSSAL_BATHS", ""))
        self.assertIsNone(TABLE.resolve("INTEGRATEDSTRATEGYEVENTS-GLIMPSE_EVENT", "INITIAL.X"))

    def test_a_key_of_another_event_is_none(self):
        self.assertIsNone(TABLE.resolve("AMALGAMATOR", "ABYSSAL_BATHS.pages.INITIAL.options.IMMERSE.title"))
        self.assertIsNone(TABLE.resolve("AMALGAMATOR", "IMMERSE"))

    def test_names_the_record_has_and_the_table_does_not_resolve(self):
        self.assertIsNone(TABLE.resolve("REFLECTIONS", "STARE"))
        self.assertIsNone(TABLE.resolve("COLORFUL_PHILOSOPHERS", "IRONCLAD"))

    def test_a_rolled_page_needs_the_offered_list(self):
        self.assertIsNone(TABLE.resolve("TINKER_TIME", "CHOOSE_CARD_TYPE.ATTACK"))
        self.assertEqual(TABLE.resolve("TINKER_TIME", "CHOOSE_CARD_TYPE.ATTACK", {"offered": ["SKILL", "ATTACK"]}), 1)
        self.assertEqual(TABLE.resolve("TINKER_TIME", "CHOOSE_CARD_TYPE.SKILL", {"offered": ["SKILL", "ATTACK"]}), 0)
        self.assertIsNone(TABLE.resolve("TINKER_TIME", "CHOOSE_CARD_TYPE.POWER", {"offered": ["SKILL", "ATTACK"]}))
        self.assertIsNone(TABLE.resolve("DOLL_ROOM", "BING_BONG.title"))
        self.assertEqual(TABLE.resolve("DOLL_ROOM", "BING_BONG.title", {"offered": ["MR_STRUGGLES", "BING_BONG", "DAUGHTER_OF_THE_WIND"]}), 1)
        self.assertIsNone(TABLE.resolve("COLORFUL_PHILOSOPHERS", "SILENT"))
        self.assertEqual(TABLE.resolve("COLORFUL_PHILOSOPHERS", "SILENT", {"offered": ["DEFECT", "REGENT", "SILENT"]}), 2)
        self.assertIsNone(TABLE.resolve("THE_FUTURE_OF_POTIONS", "POTION"))

    def test_a_fixed_option_ignores_the_offered_list(self):
        self.assertEqual(TABLE.resolve("ABYSSAL_BATHS", "IMMERSE", {"offered": ["ABSTAIN", "IMMERSE"]}), 0)

    def test_every_unresolved_row_has_a_reason_and_no_index(self):
        for name, event in TABLE.by_name.items():
            for page in event["pages"]:
                for row in page["options"]:
                    if row["mapping"] == "unresolved" and row["key"]:
                        with self.subTest(event=name, key=row["key"]):
                            self.assertTrue(row.get("evidence"))
                            self.assertTrue(row.get("dynamic") or row["index"] is None)


class TableShape(unittest.TestCase):
    def test_engine_ids_are_the_engines(self):
        ids = gen.engine_ids()
        for name, event in TABLE.by_name.items():
            self.assertEqual(event["engine_id"], ids[name], name)

    def test_no_event_the_engine_models_is_missing(self):
        modelled = set(gen.engine_ids()) - {"THE_ARCHITECT"}  # the Architect is the run's last room, not an event the pool deals
        self.assertEqual(modelled - set(TABLE.by_name), set())

    def test_indices_are_in_the_range_the_engine_gives_an_event(self):
        text = (ROOT / "src" / "Sts2Emulator" / "Generated" / "EventOptions.g.cs").read_text(encoding="utf-8")
        counts = {int(i): int(n) for i, n in re.findall(r"(\d+) => (\d+),", text)}
        for name, event in TABLE.by_name.items():
            limit = max(counts.get(event["engine_id"], 0), 4)
            for page in event["pages"]:
                for row in page["options"]:
                    if row.get("index") is not None:
                        self.assertLess(row["index"], limit, (name, row["key"]))

    def test_the_first_page_is_the_capture_where_there_is_one(self):
        for event, cap in CAPTURES.items():
            first = next(p for p in TABLE.by_name[event]["pages"] if p["page"] == "INITIAL")
            have = {r["index"] for r in first["options"] if r.get("index") is not None}
            for opt in cap["page1"] or []:
                if not opt["is_proceed"] and opt["title"] != "Proceed":
                    self.assertIn(opt["index"], have, (event, opt))

    def test_counts_are_aggregates_with_small_cells_suppressed(self):
        for event in TABLE.by_name.values():
            for page in event["pages"]:
                for row in page["options"]:
                    n = row.get("n")
                    if n is not None:
                        self.assertTrue(isinstance(n, str) or n >= ek.MIN_CELL, row)
            v = event.get("visits")
            self.assertTrue(v is None or isinstance(v, str) or v >= ek.MIN_CELL)

    def test_the_file_names_its_sources_and_holds_no_user_paths(self):
        text = (ROOT / "data" / "event_option_titles.json").read_text(encoding="utf-8")
        doc = json.loads(text)
        self.assertEqual(doc["record"], "Spire Codex public run export for v0.107.1")
        self.assertNotRegex(text, r"[A-Za-z]:[\\/]")
        self.assertNotIn("decompil", text.lower())


class Coverage(unittest.TestCase):
    def test_a_visit_is_as_weak_as_its_weakest_key(self):
        seqs = [
            ("ABYSSAL_BATHS", ["INITIAL.IMMERSE", "ALL.LINGER", "ALL.EXIT_BATHS"], 30),  # every key a captured title
            ("THIS_OR_THAT", ["INITIAL.PLAIN"], 25),  # inferred from outcomes
            ("COLOSSAL_FLOWER", ["INITIAL.REACH_DEEPER_1", "REACH_DEEPER_1.REACH_DEEPER_2"], 21),  # engine only
            ("TINKER_TIME", ["INITIAL.CHOOSE_CARD_TYPE", "CHOOSE_CARD_TYPE.SKILL"], 22),  # a rolled page
            ("REFLECTIONS", ["INITIAL.STARE"], 3),  # unknown name
            ("FAKE_MERCHANT", [], 40),  # no option choice at all
            ("SOME_MOD-EVENT", ["INITIAL.X"], 2),  # another mod's event
        ]
        cov = ek.coverage(TABLE, seqs)
        t = cov["totals"]
        self.assertEqual((t["visits_capture"], t["visits_inferred"], t["visits_engine"], t["visits_dynamic"], t["visits_unresolved"]),
                         (30, 25, 21, 22, 3))
        self.assertEqual((t["visits_no_choice"], t["visits_other"]), (40, 2))
        self.assertEqual(t["decisions_capture"], 90 + 0)  # three keys of the first visit; nothing else is a captured title
        self.assertEqual(cov["event_class"]["ABYSSAL_BATHS"], "capture")
        self.assertEqual(cov["event_class"]["TINKER_TIME"], "dynamic")
        self.assertEqual(cov["event_class"]["COLOSSAL_FLOWER"], "engine")
        self.assertEqual(cov["event_class"]["THIS_OR_THAT"], "inferred")

    def test_an_event_class_ignores_keys_chosen_fewer_than_twenty_times(self):
        seqs = [("REFLECTIONS", ["INITIAL.TOUCH_A_MIRROR"], 400), ("REFLECTIONS", ["INITIAL.STARE"], 1)]
        self.assertEqual(ek.coverage(TABLE, seqs)["event_class"]["REFLECTIONS"], "capture")


if __name__ == "__main__":
    unittest.main()
