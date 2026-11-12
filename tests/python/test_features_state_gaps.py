"""Block G of the feature contract: the state gaps (`features.gaps_block`, D58 addenda 14 and 16).

Pinned: the counter table is read off the engine's own constants; the base vector (BASE_SIZE 1855, FEATURE_VERSION 2) and every row of it are byte for byte as
they were (a digest taken before block G existed); the block's layout and scales are written by hand from the encoding; the "?" rooms' rule
(`unknown_room_odds`) on hand-made visit lists and against the engine on walked runs (at every "?" visit, over several runs and acts, the derived state equals state
list 26); the removal quote against the merchant's own price; the afterstates; the run records' codec; the lists the engine exports (25, 26, 27).
"""

import hashlib
import re
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "rl", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import human_prior_infer as hpi  # noqa: E402
from rl.run_train import features, rules, shards, vocab  # noqa: E402
from rl.run_train.engine import CombatBudget, RunSession  # noqa: E402
from rl.run_train.features import CardRef, MapNode, Option, RunView, StateGaps, UnknownVisit  # noqa: E402
from sts2_gym import native  # noqa: E402
from sts2_gym import run_constants as rc  # noqa: E402

CORE = ROOT / "src" / "Sts2Emulator" / "Core"
M, E, R, S, T, B, U, A = (
    vocab.NODE_MONSTER, vocab.NODE_ELITE, vocab.NODE_REST, vocab.NODE_SHOP, vocab.NODE_TREASURE, vocab.NODE_BOSS, vocab.NODE_UNKNOWN, vocab.NODE_ANCIENT,
)
MAP = (
    MapNode(3, 0, A, ((2, 1), (4, 1))),
    MapNode(2, 1, M, ((2, 2),)),
    MapNode(4, 1, U, ((3, 2),)),
    MapNode(2, 2, E, ((3, 3),)),
    MapNode(3, 2, R, ((3, 3),)),
    MapNode(3, 3, S, ((3, 4),)),
    MapNode(3, 4, B, ()),
)
# `encode(make_view())` of this file's view, as main built it before block G (the first 16 hex digits of the sha256 of its float32 bytes; the map-grid
# test pins the same view's digest)
PINNED_ROW_DIGEST = "beda711f28f75a39"


def make_view(**overrides) -> RunView:
    fields = {"ascension": 10, "act_index": 0, "act_region": vocab.REGION_OVERGROWTH, "floor": 3, "hp": 50, "max_hp": 80, "gold": 120,
              "deck": (CardRef(472, False), CardRef(131, False), CardRef(30, False)), "relics": (vocab.relic_id_by_name("BurningBlood"),),
              "potions": (), "max_potion_slots": 3, "map_nodes": MAP, "position": (3, 0), "boss": 83}
    fields.update(overrides)
    return RunView(**fields)


def pin_view() -> RunView:
    """The map-grid test's view (the same map and holdings), whose base row the digest above pins."""
    nodes = (
        MapNode(3, 0, A, ((2, 1), (4, 1))),
        MapNode(2, 1, M, ((2, 2), (3, 2))),
        MapNode(4, 1, U, ((3, 2), (5, 2))),
        MapNode(2, 2, E, ((3, 3),)),
        MapNode(3, 2, R, ((3, 3),)),
        MapNode(5, 2, T, ()),
        MapNode(3, 3, S, ((3, 4),)),
        MapNode(3, 4, B, ()),
    )
    return RunView(ascension=10, act_index=0, act_region=vocab.REGION_OVERGROWTH, floor=3, hp=50, max_hp=80, gold=120,
                   deck=(CardRef(472, False), CardRef(131, False), CardRef(30, False)), relics=(vocab.relic_id_by_name("BurningBlood"),),
                   potions=(), max_potion_slots=3, map_nodes=nodes, position=(3, 0), boss=83)


def cs_const(path: str, name: str) -> int:
    text = (CORE / path).read_text(encoding="utf-8")
    m = re.search(rf"const int {name} = (\d+);", text)
    assert m, f"{path} declares no {name}"
    return int(m.group(1))


class TableTests(unittest.TestCase):
    def test_every_counter_relic_resolves_and_the_table_has_no_repeat(self):
        ids = vocab.counter_relic_ids()
        self.assertEqual(len(ids), len(vocab.COUNTER_RELICS))
        self.assertEqual(len(set(ids)), len(ids))
        self.assertTrue(all(scale >= 1 for _, scale, _ in vocab.COUNTER_RELICS))

    def test_the_scales_are_the_engines_constants(self):
        scales = {name: scale for name, scale, _ in vocab.COUNTER_RELICS}
        effects = "Effects/RelicEffects.cs"
        self.assertEqual(scales["SwordOfStone"], cs_const(effects, "SwordOfStoneElites"))
        self.assertEqual(scales["BoneTea"], cs_const(effects, "BoneTeaCombats"))
        self.assertEqual(scales["EmberTea"], cs_const(effects, "EmberTeaCombats"))
        self.assertEqual(scales["TeaOfDiscourtesy"], cs_const(effects, "TeaOfDiscourtesyCombats"))
        self.assertEqual(scales["PollinousCore"], cs_const(effects, "PollinousCoreTurns"))
        self.assertEqual(scales["FakeHappyFlower"], cs_const(effects, "FakeHappyFlowerTurns"))
        self.assertEqual(scales["WongosMysteryTicket"], cs_const(effects, "WongosTicketCombats") + 1)  # past 5 is the spent flag
        self.assertEqual(scales["IronClub"], cs_const(effects, "IronClubCards"))
        self.assertEqual(scales["ToyBox"], cs_const("Run/RunConstants.cs", "ToyBoxCombats") * cs_const("Run/RunConstants.cs", "ToyBoxRelics"))
        self.assertEqual(scales["PaelsWing"], cs_const("Run/RunConstants.cs", "PaelsWingSacrifices"))
        self.assertEqual(scales["FishingRod"], cs_const("Run/RunConstants.cs", "FishingRodCombats"))
        self.assertEqual(scales["WingedBoots"], cs_const("Run/RunConstants.cs", "WingedBootsTravels"))
        # the wrapped tallies: the period the engine takes the count modulo
        text = (CORE / effects).read_text(encoding="utf-8")
        for name, period in (("Nunchaku", 10), ("TuningFork", 10), ("HappyFlower", 3), ("Pendulum", 3)):
            self.assertRegex(text, rf"CountTowards\(state, {name}, period: {period}\)")
            self.assertEqual(scales[name], period)
        self.assertRegex(text, r"var next = \(state\.Relics\[index\]\.Counter \+ 1\) % 10|int next = \(state\.Relics\[index\]\.Counter \+ 1\) % 10")
        self.assertEqual(scales["PenNib"], 10)
        self.assertRegex(text, r"Counter = total % 5")
        self.assertEqual(scales["JossPaper"], 5)
        # a count that grows past its period but is read modulo it is the one the table says wraps
        wraps = {name for name, _, w in vocab.COUNTER_RELICS if w}
        self.assertEqual(wraps, {"PaelsWing", "FishingRod", "LastingCandy"})
        # a held relic's starting counter fits inside its scale
        for relic_id, start in vocab.counter_on_pickup().items():
            scale = next(s for r, s, _ in vocab.counter_relic_scales() if r == relic_id)
            self.assertLessEqual(start, scale)

    def test_the_layout_and_the_other_sides_constants(self):
        self.assertEqual(features.GAPS_SIZE, len(vocab.COUNTER_RELICS) + 4 + 2 + 3)
        self.assertEqual(features.GAPS_SIZE, 36)
        self.assertEqual(hpi.GAPS_SIZE, features.GAPS_SIZE)
        import run_student  # noqa: PLC0415
        import run_teacher  # noqa: PLC0415

        self.assertEqual(run_teacher.GAPS_WIDTH, features.GAPS_SIZE)
        self.assertEqual(run_student.GAPS_WIDTH, features.GAPS_SIZE)
        self.assertEqual(run_teacher.GAPS_VERSION, features.GAPS_VERSION)
        self.assertEqual(run_student.GAPS_VERSION, features.GAPS_VERSION)
        self.assertEqual(features.INFLATION_ASCENSION, cs_const("Ascension.cs", "Inflation"))

    def test_the_base_vector_is_not_widened_and_not_changed(self):
        self.assertEqual(features.BASE_SIZE, 1855)
        self.assertEqual(features.FEATURE_VERSION, 2)
        view = pin_view()
        before = features.encode(view)
        full = replace(view, gaps=StateGaps(((vocab.relic_id_by_name("Girya"), 2),), (3, 300, 40, 90), (2, 1)))
        features.gaps_block(full)
        after = features.encode(full)
        self.assertEqual(before.tobytes(), after.tobytes())  # block G is never read by encode
        self.assertEqual(before.shape, (1855,))
        self.assertEqual(hashlib.sha256(before.tobytes()).hexdigest()[:16], PINNED_ROW_DIGEST)
        for option in (Option("card.take", 0, card=CardRef(30, False)), Option("shop.remove", 1, deck_index=0, price=100)):
            self.assertEqual(features.encode_afterstate(view, option).tobytes(), features.encode_afterstate(full, option).tobytes())


class BlockTests(unittest.TestCase):
    def setUp(self):
        self.girya = vocab.relic_id_by_name("Girya")
        self.pen_nib = vocab.relic_id_by_name("PenNib")
        self.paels = vocab.relic_id_by_name("PaelsWing")
        self.toy = vocab.relic_id_by_name("ToyBox")
        self.slot = {rid: i for i, rid in enumerate(vocab.counter_relic_ids())}

    def test_a_view_with_nothing_known_is_all_zero_with_every_known_bit_off(self):
        x = features.gaps_block(make_view())
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(x.shape, (features.GAPS_SIZE,))
        self.assertFalse(x.any())
        x = features.gaps_block(make_view(gaps=StateGaps()))
        self.assertFalse(x.any())

    def test_each_part_is_written_where_the_layout_says_and_its_known_bit_goes_on(self):
        counters = StateGaps(counters=((self.girya, 2), (self.pen_nib, 7), (self.paels, 5), (self.toy, 30)))
        x = features.gaps_block(make_view(gaps=counters))
        self.assertAlmostEqual(float(x[self.slot[self.girya]]), 2 / 3, places=6)  # a count to a limit
        self.assertAlmostEqual(float(x[self.slot[self.pen_nib]]), 0.7, places=6)  # a tally that wraps inside its period
        self.assertAlmostEqual(float(x[self.slot[self.paels]]), (5 % 2) / 2, places=6)  # a count read modulo its period
        self.assertAlmostEqual(float(x[self.slot[self.toy]]), 1.0, places=6)  # capped at its scale
        self.assertEqual(x[features.GAPS_KNOWN_AT : features.GAPS_KNOWN_AT + 3].tolist(), [1.0, 0.0, 0.0])
        self.assertFalse(x[features.GAPS_ROOMS_AT : features.GAPS_KNOWN_AT].any())  # the other parts are untouched
        rooms = StateGaps(rooms=(3, 300, 40, 90))
        x = features.gaps_block(make_view(gaps=rooms))
        np.testing.assert_allclose(x[features.GAPS_ROOMS_AT : features.GAPS_ROOMS_AT + 4], [0.3, 0.3, 0.04, 0.09], rtol=1e-6)
        self.assertEqual(x[features.GAPS_KNOWN_AT : features.GAPS_KNOWN_AT + 3].tolist(), [0.0, 1.0, 0.0])
        self.assertFalse(x[: features.GAPS_ROOMS_AT].any())
        removals = StateGaps(removals=(2, 1))
        x = features.gaps_block(make_view(gaps=removals))
        np.testing.assert_allclose(x[features.GAPS_REMOVALS_AT : features.GAPS_REMOVALS_AT + 2], [2.0, 1 / 3], rtol=1e-6)  # 100 + 50 * 2 at A10
        self.assertEqual(x[features.GAPS_KNOWN_AT : features.GAPS_KNOWN_AT + 3].tolist(), [0.0, 0.0, 1.0])
        # a relic not in the table is ignored, a held one at counter 0 is a known zero (the bit says so), a relic not held is absent
        x = features.gaps_block(make_view(gaps=StateGaps(counters=((vocab.relic_id_by_name("BurningBlood"), 4), (self.girya, 0)))))
        self.assertFalse(x[: features.GAPS_ROOMS_AT].any())
        self.assertEqual(float(x[features.GAPS_KNOWN_AT]), 1.0)

    def test_the_removal_price_is_the_merchants_rule(self):
        self.assertEqual([features.removal_price(n, 10) for n in range(4)], [100, 150, 200, 250])  # Inflation: 100 + 50 each
        self.assertEqual([features.removal_price(n, 5) for n in range(4)], [75, 100, 125, 150])  # below it: 75 + 25 each
        self.assertEqual(features.removal_price(0, features.INFLATION_ASCENSION), 100)
        self.assertEqual(features.removal_price(0, features.INFLATION_ASCENSION - 1), 75)
        text = (CORE / "Run" / "RunRewardGenerator.cs").read_text(encoding="utf-8")
        self.assertRegex(text, r"\(inflation \? 100 : 75\) \+ \(inflation \? 50 : 25\) \* state\.ShopRemovalsUsed")

    def test_the_afterstate_changes_only_what_the_option_changes(self):
        crucible = vocab.relic_id_by_name("SilverCrucible")
        gaps = StateGaps(counters=((self.girya, 1),), rooms=(2, 200, 60, 90), removals=(1, 0))
        view = make_view(gaps=gaps)
        here = features.gaps_block(view)
        # a removal at the merchant: the next quote is one step higher
        after = features.afterstate_gaps(view, Option("shop.remove", 1, deck_index=0, price=150))
        self.assertAlmostEqual(float(after[features.GAPS_REMOVALS_AT]), 2.0, places=6)
        np.testing.assert_array_equal(np.delete(after, features.GAPS_REMOVALS_AT), np.delete(here, features.GAPS_REMOVALS_AT))
        # a relic taken arrives with its starting counter (and one without a counter changes nothing)
        got = features.afterstate_gaps(view, Option("reward.relic", 0, relic=crucible))
        self.assertAlmostEqual(float(got[self.slot[crucible]]), 1.0, places=6)  # 3 of 3
        np.testing.assert_array_equal(features.afterstate_gaps(view, Option("reward.relic", 0, relic=vocab.relic_id_by_name("Anchor"))), here)
        # the others leave it: a card, a map step into a "?" (the room is rolled when it is entered), an event option
        for option in (Option("card.take", 0, card=CardRef(30, False)), Option("map.node", 0, node=(4, 1, U)), Option("rest.heal", 0),
                       Option("event.option", 0, event=(5, 0, 0)), Option("shop.card", 0, card=CardRef(30, False), price=50)):
            np.testing.assert_array_equal(features.afterstate_gaps(view, option), here, option.kind)
        # a view that knows nothing writes the zero block for every option, and a part that is unknown stays so
        np.testing.assert_array_equal(features.afterstate_gaps(make_view(), Option("shop.remove", 1, deck_index=0, price=100)), np.zeros(features.GAPS_SIZE))
        unknown_removals = features.afterstate_gaps(make_view(gaps=StateGaps(rooms=(0, 100, 20, 30))), Option("shop.remove", 1, deck_index=0, price=100))
        self.assertEqual(float(unknown_removals[features.GAPS_KNOWN_AT + 2]), 0.0)

    def test_the_run_record_keeps_the_gaps_and_an_old_record_has_none(self):
        view = make_view(gaps=StateGaps(((self.girya, 2),), (1, 100, 40, 60), (1, 0)))
        state = shards.view_to_json(view, 0)
        self.assertIn("gaps", state)
        maps = [shards.map_to_json(view)]
        back = shards.view_from_json(state, maps)
        self.assertEqual(back.gaps, view.gaps)
        self.assertEqual(back, view)
        import jsonschema  # noqa: PLC0415

        validator = jsonschema.Draft202012Validator(shards._STATE)  # noqa: SLF001 -- the record's schema for a state
        validator.validate(state)  # the run record's schema allows the part it now holds
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate({**state, "gaps": {"rooms": [1, 2, 3]}})  # ... and holds it to its shape
        partial = make_view(gaps=StateGaps(rooms=(0, 100, 20, 30)))
        self.assertEqual(shards.view_from_json(shards.view_to_json(partial, 0), [shards.map_to_json(partial)]).gaps, partial.gaps)
        plain = make_view()
        state = shards.view_to_json(plain, 0)
        self.assertNotIn("gaps", state)  # a record written before block G, and a view without it, read the same
        self.assertIsNone(shards.view_from_json(state, [shards.map_to_json(plain)]).gaps)


class RoomOddsTests(unittest.TestCase):
    def test_the_start_and_a_visit_of_each_kind(self):
        self.assertEqual(features.unknown_room_odds([]), (0, 100, 20, 30))
        self.assertEqual(features.unknown_room_odds([UnknownVisit(M)]), (1, 100, 40, 60))  # a fight resets its odds; the others climb by theirs
        self.assertEqual(features.unknown_room_odds([UnknownVisit(U)]), (1, 200, 40, 60))  # an event resets none
        self.assertEqual(features.unknown_room_odds([UnknownVisit(T)]), (1, 200, 20, 60))
        self.assertEqual(features.unknown_room_odds([UnknownVisit(S)]), (1, 200, 40, 30))
        self.assertEqual(features.unknown_room_odds([UnknownVisit(E)]), (1, 200, 40, 60))  # an elite is never a chance: nothing resets
        self.assertEqual(features.unknown_room_odds([UnknownVisit(M), UnknownVisit(U)]), (2, 200, 60, 90))
        self.assertEqual(features.unknown_room_odds([UnknownVisit(U)] * 3), (3, 400, 80, 120))

    def test_what_the_roll_could_not_pick_does_not_climb(self):
        # no shop in the set (a shop before it, or only shops after it): the shop odds stay where they were, a fight resets
        self.assertEqual(features.unknown_room_odds([UnknownVisit(M, shop_ok=False)]), (1, 100, 40, 30))
        self.assertEqual(features.unknown_room_odds([UnknownVisit(U, shop_ok=False), UnknownVisit(U, shop_ok=False)]), (2, 300, 60, 30))
        # Juzu Bracelet: no fight in the set, so the fight odds do not move either way
        self.assertEqual(features.unknown_room_odds([UnknownVisit(U, monster_ok=False)]), (1, 100, 40, 60))
        # Golden Compass: the room counts as visited and moves nothing
        self.assertEqual(features.unknown_room_odds([UnknownVisit(U, forced_event=True), UnknownVisit(U, forced_event=True)]), (2, 100, 20, 30))
        self.assertEqual(features.unknown_room_odds([UnknownVisit(M), UnknownVisit(U, forced_event=True)]), (2, 100, 40, 60))

    def test_the_context_of_a_visit(self):
        self.assertEqual(features.unknown_visit_allows(M, [M, R]), (True, True))
        self.assertEqual(features.unknown_visit_allows(S, [M, R]), (True, False))  # the room before was a shop
        self.assertEqual(features.unknown_visit_allows(M, [S, S]), (True, False))  # every room after it is a shop
        self.assertEqual(features.unknown_visit_allows(M, [S, M]), (True, True))
        self.assertEqual(features.unknown_visit_allows(M, []), (True, True))  # a "?" that leads nowhere is not blacklisted
        self.assertEqual(features.unknown_visit_allows(M, [M], juzu_bracelet=True), (False, True))


class EngineTests(unittest.TestCase):
    """The lists the engine exports and the rule against the engine, on runs walked by the rules policy (fights resolved without play)."""

    SEEDS = ("GAPS0001", "GAPS0002", "GAPS0003", "GAPS0004", "GAPS0005", "GAPS0006", "GAPS0007", "GAPS0008", "GAPS0009", "GAPS0010")

    @staticmethod
    def drive(seed: str, on_decision, *, gold: int | None = None, max_decisions: int = 900) -> None:
        """Walk the run with the rules policy. `on_decision(session, decision)` sees every decision before it is taken and answers None (the rules' choice),
        an Option (taken instead), "done" (it took the step itself) or "stop". A fight is won without being played."""
        session = RunSession(seed, 10, CombatBudget(1, 1))
        policy = rules.RulesPolicy()
        rng = np.random.default_rng(0)
        try:
            if gold is not None:
                session.env.debug_set_player(gold=gold)
            for _ in range(max_decisions):
                if session.phase == rc.PHASE_COMBAT:
                    session.env.debug_resolve_combat()
                    continue
                decision = session.decision()
                if decision is None:
                    if session.phase == rc.PHASE_COMBAT:
                        continue
                    return
                out = on_decision(session, decision)
                if out == "stop":
                    return
                if out == "done":
                    continue
                session.apply(out if isinstance(out, Option) else decision.options[policy.choose(session.view(), decision, rng)])
        finally:
            session.close()

    def test_the_lists_at_the_start_of_a_run(self):
        session = RunSession(self.SEEDS[0], 10, CombatBudget(1, 1))
        try:
            h = session.handle
            self.assertEqual(native.run_state_list(h, 26, 4), (0, 100, 20, 30))
            self.assertEqual(native.run_state_list(h, 27, 2), (0, 0))
            gaps = session.view().gaps
            self.assertEqual(gaps, StateGaps((), (0, 100, 20, 30), (0, 0)))
            x = features.gaps_block(session.view())
            self.assertEqual(x[features.GAPS_KNOWN_AT : features.GAPS_KNOWN_AT + 3].tolist(), [1.0, 1.0, 1.0])
            np.testing.assert_allclose(x[features.GAPS_ROOMS_AT : features.GAPS_REMOVALS_AT], [0.0, 0.1, 0.02, 0.03], rtol=1e-6)
            self.assertAlmostEqual(float(x[features.GAPS_REMOVALS_AT]), 1.0, places=6)  # 100 at A10
        finally:
            session.close()

    def test_a_relics_counter_is_read_from_list_25(self):
        session = RunSession(self.SEEDS[1], 10, CombatBudget(1, 1))
        try:
            girya, pen, maw = (vocab.relic_id_by_name(n) for n in ("Girya", "PenNib", "MawBank"))
            burning = vocab.relic_id_by_name("BurningBlood")
            session.env.debug_set_player(relics=[burning, girya, pen, maw], relic_counters=[(girya, 2), (pen, 7), (maw, 1)])
            gaps = session.view().gaps
            self.assertEqual(gaps.counters, tuple(sorted(((girya, 2), (pen, 7), (maw, 1)))))
            x = features.gaps_block(session.view())
            slot = {rid: i for i, rid in enumerate(vocab.counter_relic_ids())}
            self.assertAlmostEqual(float(x[slot[girya]]), 2 / 3, places=6)
            self.assertAlmostEqual(float(x[slot[pen]]), 0.7, places=6)
            self.assertAlmostEqual(float(x[slot[maw]]), 1.0, places=6)
        finally:
            session.close()

    def test_a_library_without_the_new_lists_leaves_their_parts_unknown(self):
        from unittest import mock

        real = native.run_state_list

        def old_library(handle, list_id, capacity=256, *, hidden=False):
            if list_id in (26, 27):
                raise RuntimeError(f"Sts2Run_GetStateList failed with status -3 for list {list_id}.")
            return real(handle, list_id, capacity, hidden=hidden)

        session = RunSession(self.SEEDS[2], 10, CombatBudget(1, 1))
        try:
            with mock.patch.object(native, "run_state_list", old_library):
                gaps = features.gaps_from_engine(session.handle)
            self.assertEqual(gaps, StateGaps((), None, None))
            x = features.gaps_block(replace(session.view(), gaps=gaps))
            self.assertEqual(x[features.GAPS_KNOWN_AT : features.GAPS_KNOWN_AT + 3].tolist(), [1.0, 0.0, 0.0])
        finally:
            session.close()

    def test_the_derived_odds_equal_the_engines_at_every_question_mark(self):
        checked = {"visits": 0, "runs": 0, "acts": set()}
        for seed in self.SEEDS:
            visits: list[UnknownVisit] = []
            state = {"act": None, "n": 0}

            def on_decision(session, decision, seed=seed, visits=visits, state=state):
                act = session.info["act"]
                if act != state["act"]:  # a new act: the odds start again
                    state["act"] = act
                    visits.clear()
                    checked["acts"].add(act)
                if decision.type != "map":
                    return None
                view = session.view()
                if vocab.relic_id_by_name("GoldenCompass") in view.relics:
                    return "stop"  # a compass act's "?" are events by another rule (the human build leaves those runs out)
                wanted = [o for o in decision.options if o.node is not None and o.node[2] == U]
                if not wanted:
                    return None
                option = wanted[0]
                by_coord = {(n.col, n.row): n for n in view.map_nodes}
                kids = [by_coord[c].node_type for c in by_coord[(option.node[0], option.node[1])].children if c in by_coord]
                previous = session.info["node_type"]
                juzu = vocab.relic_id_by_name("JuzuBracelet") in view.relics
                session.apply(option)
                visits.append(UnknownVisit(session.info["node_type"], *features.unknown_visit_allows(previous, kids, juzu)))
                got = native.run_state_list(session.handle, 26, 4)
                self.assertEqual(got, features.unknown_room_odds(visits), f"{seed}: after {len(visits)} visits {[tuple(v) for v in visits]}")
                self.assertEqual(session.view().gaps.rooms, got)
                state["n"] += 1
                return "done"

            self.drive(seed, on_decision)
            checked["visits"] += state["n"]
            checked["runs"] += int(state["n"] > 0)
        self.assertGreaterEqual(checked["visits"], 40, "too few question marks visited to say anything")
        self.assertGreaterEqual(checked["runs"], 8)
        self.assertGreaterEqual(len(checked["acts"]), 2)  # more than the first act: the reset at an act's start is covered

    def test_the_merchants_quote_is_the_rule_and_a_removal_moves_it(self):
        found: dict = {}
        for seed in self.SEEDS[:6]:

            def on_decision(session, decision):
                if decision.type == "map":
                    shops = [o for o in decision.options if o.node is not None and o.node[2] == S]
                    return shops[0] if shops else None
                if decision.type != "shop":
                    return None
                view = session.view()
                if vocab.relic_id_by_name("MembershipCard") in view.relics:
                    return "stop"
                removal = [o for o in decision.options if o.kind == "shop.remove"]
                if not removal:
                    return None
                found.update(quote=removal[0].price, before=view, predicted=features.afterstate_gaps(view, removal[0]))
                session.apply(removal[0])
                found["after"] = session.view()
                return "stop"

            self.drive(seed, on_decision, gold=900)
            if found:
                break
        self.assertTrue(found, "no shop with a removal on offer was reached")
        self.assertEqual(found["before"].gaps.removals, (0, 0))
        self.assertEqual(found["quote"], features.removal_price(0, 10))  # 100 at A10 with nothing bought
        self.assertEqual(found["after"].gaps.removals, (1, 0))  # the engine counted the removal
        self.assertEqual(features.gaps_block(found["after"]).tobytes(), found["predicted"].tobytes())  # the afterstate predicted exactly that
        self.assertEqual(features.removal_price(1, 10), 150)


if __name__ == "__main__":
    unittest.main()
