"""`rl/run_train/engine.py`: the run session over the native run engine (DESIGN.md 1, 7).

Tests that touch the run engine pass their ascension explicitly (erratum RF1). Most play
at 8, the level the committed traces and live fixtures were captured at.
"""

import ctypes
import re
import sys
import unittest
from collections import Counter
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))

import combat_mcts  # noqa: E402  # ty: ignore[unresolved-import] -- rl/ is on sys.path

from rl.run_train import engine, features, rules, seeding, vocab  # noqa: E402
from rl.run_train.engine import CombatBudget, RunCombat, RunSession  # noqa: E402
from rl.run_train.features import Decision  # noqa: E402
from sts2_gym import native  # noqa: E402
from sts2_gym import run_constants as rc  # noqa: E402

RUN_CONSTANTS = ROOT / "src" / "Sts2Emulator" / "Core" / "Run" / "RunConstants.cs"
RUN_PHASE = ROOT / "src" / "Sts2Emulator" / "Core" / "Run" / "RunPhase.cs"
A8 = 8
STRIKE = 472


def _cs_constants() -> dict[str, int]:
    text = RUN_CONSTANTS.read_text(encoding="utf-8")
    return {
        name: int(value)
        for name, value in re.findall(r"public const int (\w+)\s*=\s*(-?\d+);", text)
    }


def _cs_phases() -> dict[str, int]:
    body = re.search(
        r"enum RunPhase\s*\{(?P<body>.*?)\}",
        RUN_PHASE.read_text(),
        re.DOTALL,
    )
    assert body is not None
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", body.group("body"), flags=re.DOTALL)
    members = [m.split("=")[0].strip() for m in text.split(",") if m.strip()]
    return {member: index for index, member in enumerate(members)}


def _hand_size(combat) -> int:
    size = 0
    while (
        size < 10 and combat.obs[combat_mcts.HAND_OFFSET + size * combat_mcts.CARD_SLOT]
    ):
        size += 1
    return size


def play_combat(session: RunSession, sims: int, rng, move: list[int]) -> None:
    """Search the fight the session stands in to its end (DESIGN.md 1.4 item 2)."""
    combat = session.combat()
    while not combat.done:
        legal = combat.legal()
        tree = combat_mcts.run_search(combat, legal, sims, rng, 0.1, move[0], crn=True)
        combat.step(combat_mcts.best_action(tree, legal, rng))
        move[0] += 1


def deck_card(view, option) -> features.CardRef:
    """Return the deck card an option names, which the caller expects it to name."""
    assert option.deck_index is not None, option
    return view.deck[option.deck_index]


def next_decision(session: RunSession) -> Decision:
    """Return the session's next decision, which the caller expects to exist."""
    decision = session.decision()
    assert decision is not None, session.info
    return decision


def drive_until(
    seed: str,
    predicate,
    *,
    sims: int = 5,
    max_steps: int = 3000,
    prefer_node: int | None = None,
) -> RunSession | None:
    """Advance a rules-policy session until `predicate(session, decision)` holds.

    `prefer_node` steers the map toward that node type and gives the run 999 HP
    (`Sts2Run_DebugSetHp`, the probe's `--boost-hp`) so it survives to get there; it only
    serves to REACH a screen, never to judge play. Returns the session (the caller closes
    it), or None when the run ended first.
    """
    session = RunSession(seed, ascension=A8, budget=CombatBudget(sims, sims))
    if prefer_node is not None:
        native.run_debug_set_hp(
            session.handle,
            999,
            999,
            session.env._run_obs_buf,
        )
    policy = rules.RulesPolicy()
    rng, explore, move_base = seeding.run_streams(seed)
    move = [move_base]
    for _ in range(max_steps):
        decision = session.decision()
        if decision is None:
            if session.phase == rc.PHASE_COMBAT:
                play_combat(session, sims, rng, move)
                continue
            session.close()
            return None
        if predicate(session, decision):
            return session
        index = policy.choose(session.view(), decision, explore)
        if decision.type == "map" and prefer_node is not None:
            preferred = [
                i
                for i, o in enumerate(decision.options)
                if o.node is not None and o.node[2] == prefer_node
            ]
            index = preferred[0] if preferred else index
        session.apply(decision.options[index])
    session.close()
    return None


class ConstantsTests(unittest.TestCase):
    def test_the_mirrored_constants_match_run_constants(self):
        cs = _cs_constants()
        mirrored = {
            "RewardSkipAction": engine.REWARD_SKIP_ACTION,
            "RewardSacrificeAction": engine.REWARD_SACRIFICE_ACTION,
            "RewardFourthCardAction": engine.REWARD_FOURTH_CARD_ACTION,
            "RewardRerollAction": engine.REWARD_REROLL_ACTION,
            "RestHealAction": engine.REST_HEAL_ACTION,
            "RestUpgradeAction": engine.REST_UPGRADE_ACTION,
            "RestCloneAction": engine.REST_CLONE_ACTION,
            "RestLiftAction": engine.REST_LIFT_ACTION,
            "RestDigAction": engine.REST_DIG_ACTION,
            "RestHatchAction": engine.REST_HATCH_ACTION,
            "RestCookAction": engine.REST_COOK_ACTION,
            "RestKindleAction": engine.REST_KINDLE_ACTION,
            "ShopRemoveAction": engine.SHOP_REMOVE_ACTION,
            "ShopSkipAction": engine.SHOP_SKIP_ACTION,
            "EventSkipAction": engine.EVENT_SKIP_ACTION,
            "BundleConfirmAction": engine.BUNDLE_CONFIRM_ACTION,
            "FakeMerchantThrowAction": engine.FAKE_MERCHANT_THROW_ACTION,
            "FakeMerchantLeaveAction": engine.FAKE_MERCHANT_LEAVE_ACTION,
            "EventResultPending": engine.EVENT_RESULT_PENDING,
            "EventWelcomeToWongos": engine.EVENT_WELCOME_TO_WONGOS,
            "CardRewardOptionCount": engine.CARD_REWARD_OPTION_COUNT,
            "TreasureTakeRelicAction": engine.TREASURE_TAKE_RELIC_ACTION,
            "NodeNormal": vocab.NODE_MONSTER,
            "NodeElite": vocab.NODE_ELITE,
            "NodeRest": vocab.NODE_REST,
            "NodeShop": vocab.NODE_SHOP,
            "NodeRelic": vocab.NODE_TREASURE,
            "NodeBoss": vocab.NODE_BOSS,
            "NodeEvent": vocab.NODE_UNKNOWN,
            "NodeAncient": vocab.NODE_ANCIENT,
            "ActOvergrowth": vocab.REGION_OVERGROWTH,
            "ActUnderdocks": vocab.REGION_UNDERDOCKS,
            "ActHive": vocab.REGION_HIVE,
            "ActGlory": vocab.REGION_GLORY,
            "FakeMerchantRelicCost": engine.FAKE_MERCHANT_RELIC_COST,
            "MapBossRow": vocab.MAP_BOSS_ROW,
            "PotionSlotCapacity": engine.POTION_SLOT_CAPACITY,
            # `UsePotionAction = 2 * CrystalSphereCells`: an expression, which the regex
            # above does not read, so it is rebuilt from the board's side.
            "UsePotionAction": 2 * cs["CrystalSphereSize"] ** 2,
        }
        for name, value in mirrored.items():
            with self.subTest(name=name):
                if name == "UsePotionAction":
                    self.assertEqual(engine.USE_POTION_ACTION, value)
                else:
                    self.assertEqual(value, cs[name])

    def test_the_mirrored_phases_match_run_phase(self):
        phases = _cs_phases()
        self.assertEqual(engine.PHASE_FAKE_MERCHANT, phases["FakeMerchant"])
        self.assertEqual(rc.PHASE_BUNDLE_SELECT, phases["BundleSelect"])
        self.assertEqual(rc.PHASE_COMPLETE, phases["Complete"])

    def test_the_deck_selection_ordinals_match_run_state(self):
        selections = vocab.names.deck_selections()
        expected = {
            "None": engine.SEL_NONE,
            "Enchant": engine.SEL_ENCHANT,
            "TransformTo": engine.SEL_TRANSFORM_TO,
            "TransformToCarrying": engine.SEL_TRANSFORM_TO_CARRYING,
            "Upgrade": engine.SEL_UPGRADE,
            "TransformToRandom": engine.SEL_TRANSFORM_TO_RANDOM,
            "TransformToRandomUpgraded": engine.SEL_TRANSFORM_TO_RANDOM_UPGRADED,
            "Remove": engine.SEL_REMOVE,
            "RemoveUpgradable": engine.SEL_REMOVE_UPGRADABLE,
            "RemoveTaggedBasic": engine.SEL_REMOVE_TAGGED_BASIC,
            "Duplicate": engine.SEL_DUPLICATE,
        }
        for name, ordinal in expected.items():
            self.assertEqual(selections[ordinal], name)


class PotionUseActionTests(unittest.TestCase):
    """The belt's use-a-potion moves (actions 242..247) belong to no screen's decision.

    The emulator offers them wherever the belt is open and a slot holds one of the four
    `AnyTime` potions (`RunPotionUse`, afa23768). The screen readers index per-slot lists by
    action, so a map screen read with one of them in the mask failed with an IndexError, in
    every run that picked up a Blood Potion, Fruit Juice, Entropic Brew or Foul Potion
    (DESIGN.md 1.5.8).
    """

    BLOOD_POTION = 6

    def test_a_held_anytime_potion_leaves_the_map_decision_as_it_was(self):
        session = drive_until(
            "POTION01",
            lambda _s, d: d.type == "map",
            sims=5,
        )
        self.assertIsNotNone(session, "the run ended before it reached a map screen")
        try:
            before = [(o.kind, o.action, o.node) for o in next_decision(session).options]
            native.run_debug_set_player(
                session.handle,
                session.env._run_obs_buf,
                potions=[self.BLOOD_POTION],
            )
            raw = [
                i
                for i, legal in enumerate(
                    native.run_action_mask(session.handle, native.RUN_MAX_ACTIONS),
                )
                if legal
            ]
            # The setup does what the test is about: the emulator offers the move.
            self.assertIn(engine.USE_POTION_ACTION, raw)
            self.assertNotIn(engine.USE_POTION_ACTION, engine._legal(session.handle))
            after = [(o.kind, o.action, o.node) for o in next_decision(session).options]
            self.assertEqual(after, before)
        finally:
            session.close()


class CombatBudgetTests(unittest.TestCase):
    def test_parse_and_format(self):
        self.assertEqual(CombatBudget.parse("120/500"), CombatBudget(120, 500))
        self.assertEqual(CombatBudget.parse("120"), CombatBudget(120, 120))
        self.assertEqual(str(CombatBudget(120, 500)), "120/500")
        self.assertEqual(str(CombatBudget(250, 250)), "250")
        # An int still means a flat budget (DESIGN.md 2.7).
        self.assertEqual(CombatBudget.of(60), CombatBudget(60, 60))
        self.assertEqual(CombatBudget.of(CombatBudget(1, 2)), CombatBudget(1, 2))

    def test_hard_fights_are_elites_and_bosses(self):
        budget = CombatBudget(120, 500)
        self.assertEqual(budget.sims_for(vocab.NODE_ELITE), 500)
        self.assertEqual(budget.sims_for(vocab.NODE_BOSS), 500)
        self.assertEqual(budget.sims_for(vocab.NODE_MONSTER), 120)
        # A fight an event starts counts as ordinary.
        self.assertEqual(budget.sims_for(vocab.NODE_UNKNOWN), 120)

    def test_budgets_that_overflow_the_clone_seed_are_refused(self):
        with self.assertRaises(ValueError):
            CombatBudget(120, seeding.MAX_SIMS + 1)
        with self.assertRaises(ValueError):
            CombatBudget(0, 500)


class AscensionTests(unittest.TestCase):
    """The run engine takes the level (run API 20); the session passes it straight on."""

    def test_the_level_is_passed_straight_through(self):
        created = []

        class FakeEnv:
            def __init__(self, seed, **kwargs: int):
                created.append(kwargs["ascension"])

        with mock.patch.object(engine, "Sts2RunEnv", FakeEnv):
            engine.new_run_env("RF1TEST0", 3)
        self.assertEqual(created, [3])

    def test_a_session_plays_the_level_it_was_given(self):
        # Ascender's Bane from A5, a two-slot belt from A4, 80% HP from A2 (Ascension.cs).
        for level, bane, slots in ((0, False, 3), (9, True, 2)):
            with self.subTest(level=level):
                session = RunSession("RF1TEST0", ascension=level)
                try:
                    view = session.view()
                    self.assertEqual(view.ascension, level)
                    self.assertEqual(view.hp, native.starting_hp(level))
                    self.assertEqual(view.max_potion_slots, slots)
                    self.assertEqual(features.CardRef(10001, False) in view.deck, bane)
                finally:
                    session.close()

    def test_view_from_engine_refuses_the_wrong_level(self):
        session = RunSession("RF1TEST0", ascension=A8)
        try:
            with self.assertRaises(ValueError):
                features.view_from_engine(session.env, 0)
        finally:
            session.close()


class ViewFromEngineTests(unittest.TestCase):
    def test_the_opening_view(self):
        session = RunSession("VIEW0001", ascension=A8)
        try:
            view = session.view()
            self.assertEqual(
                (view.hp, view.max_hp, view.gold, view.floor),
                (64, 80, 99, 1),
            )
            self.assertEqual(len(view.deck), 11)
            self.assertIn(
                features.CardRef(10001, False),
                view.deck,
            )  # Ascender's Bane (A5)
            self.assertEqual(view.relics, (vocab.relic_id_by_name("BurningBlood"),))
            self.assertEqual((view.potions, view.max_potion_slots), ((), 2))
            self.assertEqual((view.act_index, view.ascension), (0, A8))
            self.assertIn(
                view.act_region,
                (vocab.REGION_OVERGROWTH, vocab.REGION_UNDERDOCKS),
            )
            self.assertIn(view.boss, vocab.encounter_pools()[view.act_region]["boss"])
            nodes = {(n.col, n.row): n for n in view.map_nodes}
            self.assertEqual(view.position, (3, 0))
            self.assertEqual(nodes[3, 0].node_type, vocab.NODE_ANCIENT)
            bosses = [n for n in view.map_nodes if n.node_type == vocab.NODE_BOSS]
            self.assertEqual(len(bosses), 1)
            self.assertEqual(bosses[0].row, 16)
            self.assertEqual(view.pending, features.Pending())
            np.testing.assert_array_equal(
                features.from_run_engine(session.env, A8),
                features.encode(view),
            )
        finally:
            session.close()


class SessionTests(unittest.TestCase):
    def test_the_run_opens_on_neows_three_relics(self):
        session = RunSession("NEOW0001", ascension=A8)
        try:
            decision = next_decision(session)
            self.assertEqual(decision.type, "neow")
            self.assertEqual([o.kind for o in decision.options], ["neow.relic"] * 3)
            self.assertEqual([o.action for o in decision.options], [0, 1, 2])
            self.assertEqual(
                tuple(o.relic for o in decision.options),
                native.run_state_list(session.handle, 3, 3),
            )
            self.assertEqual(session.info["phase"], rc.PHASE_ANCIENT)
        finally:
            session.close()

    def test_neows_proceed_page_is_auto(self):
        # After the choice, Neow's page offers only Proceed (NeowAwaitingProceed): the
        # session steps it, so no later decision is Neow's.
        for seed in ("NEOW0001", "NEOW0002", "NEOW0003"):
            session = RunSession(seed, ascension=A8)
            try:
                session.apply(next_decision(session).options[0])
                for _ in range(20):
                    decision = next_decision(session)
                    if decision.type == "map":
                        break
                    self.assertNotEqual(decision.type, "neow")
                    session.apply(decision.options[0])
                self.assertEqual(decision.type, "map")
            finally:
                session.close()

    def test_a_map_decision_names_the_nodes(self):
        session = drive_until("MAPS0001", lambda _s, d: d.type == "map")
        assert session is not None
        try:
            decision = next_decision(session)
            view = session.view()
            nodes = {(n.col, n.row): n for n in view.map_nodes}
            here = nodes[view.position]
            for opt in decision.options:
                assert opt.node is not None
                col, row, node_type = opt.node
                self.assertIn((col, row), here.children)
                self.assertEqual(nodes[col, row].node_type, node_type)
        finally:
            session.close()

    def test_decisions_are_none_in_combat_and_combat_hands_out_the_fight(self):
        session = RunSession("FIGHT001", ascension=A8, budget=CombatBudget(7, 11))
        try:
            policy = rules.RulesPolicy()
            rng = np.random.default_rng(0)
            while session.phase != rc.PHASE_COMBAT:
                decision = next_decision(session)
                session.apply(
                    decision.options[policy.choose(session.view(), decision, rng)],
                )
            self.assertIsNone(session.decision())
            combat = session.combat()
            self.assertFalse(combat.done)
            self.assertEqual(
                combat.sims,
                7 if combat.node_type == vocab.NODE_MONSTER else 11,
            )
        finally:
            session.close()


class UnknownDecisionTests(unittest.TestCase):
    def test_an_unknown_decision_is_counted_under_its_reason(self):
        # Each unknown screen names why (a purpose ordinal, `sel_none`, the Eternal
        # transform stopgap, an unhandled phase), so status.json can bucket the rates.
        session = RunSession("UNKNOWN1", ascension=A8)
        real = engine._read_screen
        calls = []

        def unknown_first(*args: Any) -> engine._Screen:
            calls.append(args)
            if len(calls) == 1:
                return engine._unknown_screen([0], "purpose_42")
            return real(*args)

        try:
            with mock.patch.object(engine, "_read_screen", unknown_first):
                decision = next_decision(session)
                self.assertEqual(decision.type, "unknown")
                self.assertEqual([o.scorable for o in decision.options], [False])
                session.apply(decision.options[0])
            self.assertEqual(
                dict(session.stats.unknown_decisions),
                {f"{rc.PHASE_ANCIENT}/purpose_42": 1},
            )
        finally:
            session.close()


class ActTwoScreenTests(unittest.TestCase):
    """Crystal Sphere and the Fake Merchant are known decisions, not unknown ones."""

    @staticmethod
    def context(legal: list[int]) -> Any:
        return mock.Mock(legal=legal, handle=0)

    def test_the_crystal_sphere_offers_one_unscorable_cell_per_legal_divination(self):
        screen = engine._crystal_sphere(self.context([3, 40, 124]))
        assert screen.decision is not None
        self.assertEqual(screen.decision.type, "sphere")
        self.assertEqual(
            [(o.kind, o.action, o.scorable) for o in screen.decision.options],
            [("sphere.cell", 3, False), ("sphere.cell", 40, False), ("sphere.cell", 124, False)],
        )
        self.assertIsNone(screen.reason)

    def test_the_fake_merchant_sells_scorable_relics_read_off_a_clone(self):
        bought = {0: 11, 2: 12, 4: None}
        with mock.patch.object(engine, "_bought_relic", lambda _h, a: bought[a]):
            screen = engine._fake_merchant(self.context([0, 2, 4, 6, 7]))
        assert screen.decision is not None
        self.assertEqual(screen.decision.type, "shop")
        self.assertEqual(
            [(o.kind, o.action, o.relic, o.price, o.scorable) for o in screen.decision.options],
            [
                ("shop.relic", 0, 11, 50, True),
                ("shop.relic", 2, 12, 50, True),
                ("unknown", 4, None, 0, False),
                ("merchant.throw", 6, None, 0, False),
                ("shop.leave", 7, None, 0, True),
            ],
        )


class FlattenedOptionTests(unittest.TestCase):
    def assert_enumeration_leaves_the_run_unchanged(
        self,
        session: RunSession,
    ) -> Decision:
        handle = session.handle
        before = (
            native.run_phase(handle),
            np.array(native.run_action_mask(handle, native.RUN_MAX_ACTIONS)),
            np.ctypeslib.as_array(session.env._run_obs_buf).copy(),
            native.run_state_list(handle, 0, 256),
        )
        decision = engine.decision_of(session.env, A8)
        after = (
            native.run_phase(handle),
            np.array(native.run_action_mask(handle, native.RUN_MAX_ACTIONS)),
            np.ctypeslib.as_array(session.env._run_obs_buf).copy(),
            native.run_state_list(handle, 0, 256),
        )
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        np.testing.assert_array_equal(before[2], after[2])
        self.assertEqual(before[3], after[3])
        assert decision is not None
        return decision

    def test_the_smith_follow_up_is_read_on_a_faithful_clone(self):
        for seed in ("REST0001", "REST0002", "REST0003", "REST0004", "REST0005"):
            session = drive_until(
                seed,
                lambda _s, d: any(o.kind == "rest.smith" for o in d.options),
            )
            if session is None:
                continue
            try:
                decision = self.assert_enumeration_leaves_the_run_unchanged(session)
                smiths = [o for o in decision.options if o.kind == "rest.smith"]
                view = session.view()
                upgradable = {
                    c for c in view.deck if vocab.card_upgradable(c.card_id, c.upgraded)
                }
                self.assertEqual({deck_card(view, o) for o in smiths}, upgradable)
                self.assertEqual(
                    len(smiths),
                    len(upgradable),
                )  # identical cards collapse
                for o in smiths:
                    self.assertEqual(
                        o.deck_index,
                        view.deck.index(deck_card(view, o)),
                    )
                return
            finally:
                session.close()
        self.fail("no seed reached a rest site")

    def test_a_card_the_follow_up_screen_does_not_offer_is_a_clear_error(self):
        session = drive_until(
            "REST0001",
            lambda _s, d: any(o.kind == "rest.smith" for o in d.options),
            prefer_node=vocab.NODE_REST,
        )
        assert session is not None
        try:
            view = session.view()
            bane = view.deck.index(features.CardRef(10001, False))  # never upgradable
            with self.assertRaisesRegex(RuntimeError, r"10001.*phase 9"):
                session.apply(features.Option("rest.smith", 1, deck_index=bane))
        finally:
            session.close()

    def test_the_removal_follow_up_is_read_on_a_faithful_clone(self):
        for seed in (
            "SHOP0001",
            "SHOP0002",
            "SHOP0003",
            "SHOP0004",
            "SHOP0005",
            "SHOP0006",
        ):
            session = drive_until(
                seed,
                lambda _s, d: any(o.kind == "shop.remove" for o in d.options),
                prefer_node=vocab.NODE_SHOP,
            )
            if session is None:
                continue
            try:
                decision = self.assert_enumeration_leaves_the_run_unchanged(session)
                removals = [o for o in decision.options if o.kind == "shop.remove"]
                view = session.view()
                removable = {c for c in view.deck if vocab.card_removable(c.card_id)}
                self.assertEqual({deck_card(view, o) for o in removals}, removable)
                self.assertNotIn(
                    features.CardRef(10001, False),
                    removable,
                )  # Eternal (E4)
                return
            finally:
                session.close()
        self.fail("no seed reached a shop with an affordable removal")


class RunCombatTests(unittest.TestCase):
    @staticmethod
    def reach_first_fight(seed: str) -> RunSession:
        session = RunSession(seed, ascension=A8)
        policy = rules.RulesPolicy()
        rng = np.random.default_rng(0)
        while session.phase != rc.PHASE_COMBAT:
            decision = next_decision(session)
            session.apply(
                decision.options[policy.choose(session.view(), decision, rng)],
            )
        return session

    def step_and_check(self, session: RunSession, combat, action: int) -> None:
        combat.step(action)
        left = native.run_phase(session.handle) != rc.PHASE_COMBAT
        self.assertEqual(combat.done, left)

    def test_done_flips_when_a_won_fight_leaves_combat(self):
        session = self.reach_first_fight("RCWIN001")
        try:
            handle = session.handle
            native.run_debug_set_hp(handle, 999, 999, session.env._run_obs_buf)
            combat = session.combat()
            rng = np.random.default_rng(1)
            for move in range(combat_mcts.STEP_CAP):
                if combat.done:
                    break
                legal = combat.legal()
                tree = combat_mcts.run_search(combat, legal, 20, rng, 0.1, move)
                self.step_and_check(
                    session,
                    combat,
                    combat_mcts.best_action(tree, legal, rng),
                )
            self.assertTrue(combat.done)
            self.assertTrue(combat.won())
            self.assertEqual(native.run_phase(handle), rc.PHASE_RELIC_REWARD)
            self.assertEqual(combat.hp(), session.view().hp)
        finally:
            session.close()

    def test_done_flips_when_a_lost_fight_ends_the_run(self):
        session = self.reach_first_fight("RCLOSE01")
        try:
            handle = session.handle
            native.run_debug_set_hp(handle, 1, 80, session.env._run_obs_buf)
            combat = session.combat()
            for _ in range(combat_mcts.STEP_CAP):
                if combat.done:
                    break
                # End the turn: its action is the hand size (`CombatEngine.ValidActions`).
                end_turn = _hand_size(combat)
                self.assertIn(end_turn, combat.legal())
                self.step_and_check(session, combat, end_turn)
            self.assertTrue(combat.done)
            self.assertFalse(combat.won())
            self.assertEqual(native.run_phase(handle), rc.PHASE_COMPLETE)
            self.assertTrue(session.died)
            self.assertIsNone(session.decision())
        finally:
            session.close()

    def test_forks_are_independent_and_the_root_is_never_destroyed(self):
        session = self.reach_first_fight("RCFORK01")
        try:
            combat = session.combat()
            child = combat.fork(12345)
            before = combat.hp(), combat.legal()
            child.step(child.legal()[-1])
            self.assertEqual((combat.hp(), combat.legal()), before)
            child.close()
            combat.close()  # the session's own handle: a no-op
            self.assertEqual(native.run_phase(session.handle), rc.PHASE_COMBAT)
        finally:
            session.close()


class CombatForkTests(unittest.TestCase):
    """`RunCombat(fork_combat=True)`: forks are combat handles (`Sts2Run_ForkCombat`)."""

    def test_a_combat_fork_plays_the_run_clones_world(self):
        session = RunCombatTests.reach_first_fight("RCFORK02")
        try:
            combat = session.combat(fork_combat=True)
            for seed in (7, 99_999):
                run_world = RunCombat(session.handle, owns_handle=False).fork(seed)
                fight = combat.fork(seed)
                try:
                    self.assertIsInstance(fight, combat_mcts.Combat)
                    self.assertEqual(fight.encounter_id, combat.encounter_id)
                    for _ in range(combat_mcts.STEP_CAP):
                        legal = fight.legal()
                        self.assertEqual(legal, run_world.legal())
                        self.assertEqual(
                            list(fight.obs),
                            list(run_world.obs[: native.OBS_SIZE]),
                        )
                        if fight.step(legal[0]) | run_world.step(legal[0]):
                            break
                    self.assertTrue(fight.done and run_world.done)
                    self.assertEqual(fight.won(), run_world.won())
                finally:
                    fight.close()
                    run_world.close()
            self.assertEqual(native.run_phase(session.handle), rc.PHASE_COMBAT)
        finally:
            session.close()

    def test_a_faithful_combat_fork_is_the_fight_itself(self):
        session = RunCombatTests.reach_first_fight("RCFORK03")
        try:
            combat = session.combat(fork_combat=True)
            view = combat.combat_fork(None)
            try:
                self.assertEqual(list(view.obs), list(combat.obs[: native.OBS_SIZE]))
                self.assertEqual(view.legal(), combat.legal())
            finally:
                view.close()
        finally:
            session.close()

    def test_a_run_outside_a_fight_has_none_to_fork(self):
        session = RunSession("RCFORK04", ascension=A8)
        try:
            self.assertNotEqual(session.phase, rc.PHASE_COMBAT)
            obs = (ctypes.c_int * native.OBS_SIZE)()
            with self.assertRaisesRegex(RuntimeError, "not in a fight"):
                native.run_fork_combat(session.handle, True, 1, obs)
        finally:
            session.close()


class HopperReturnedCardTests(unittest.TestCase):
    """The Thieving Hopper's returned card is claimed with the gold, before the card offer.

    The fixture is decisions 58-62 of a recorded run (floor 19): the deck is 20 before the
    fight, 19 after the kill (the steal took the card out of the deck), 19 with the screen
    gold, `special_card`, card; then 20 with the returned card last, then 21 with the pick
    after it. The driver takes gold and the special row unasked; the run layer read only four
    reward counts, so the row stayed on the screen and the card was lost.
    """

    FIXTURE = ROOT / "tests" / "python" / "fixtures" / "hopper_returned_card.json"

    class _Screen6:
        """List 6 and the mask layout of one rewards screen: gold, card items, special rows."""

        def __init__(self, offer, deck):
            self.gold = int(any(r["type"] == "gold" for r in offer))
            self.cards = sum(r["type"] == "card" for r in offer)
            self.rows = [tuple(r["card"]) for r in offer if r["type"] == "special_card"]
            self.deck = list(deck)
            self.steps = []
            self.card_opened = False

        def state_list(self, _handle, list_id, capacity=256):
            assert list_id == 6
            counts = (self.gold, 0, 0, self.cards, len(self.rows))
            return counts[:capacity]

        def step(self, action):
            self.steps.append(action)
            layout = ["gold"] * self.gold + ["card"] * self.cards + ["special"] * len(self.rows)
            kind = layout[action]
            if kind == "gold":
                self.gold = 0
            elif kind == "special":
                card_id, upgraded = self.rows.pop(0)
                self.deck.append([f"CARD.{card_id}", upgraded])
            else:
                self.card_opened = True

    def _fixture(self):
        import json

        data = json.loads(self.FIXTURE.read_text(encoding="utf-8"))
        return {d["i"]: d for d in data["decisions"]}

    def test_the_returned_card_is_claimed_before_the_card_offer_opens(self):
        decisions = self._fixture()
        screen = self._Screen6(decisions[60]["offer"], decisions[59]["deck"])
        self.assertEqual(len(decisions[58]["deck"]), 20)
        self.assertEqual(len(screen.deck), 19)
        ctx = mock.Mock(handle=1)
        ctx.view = mock.Mock(potions=(), max_potion_slots=3)

        with mock.patch.object(engine.native, "run_state_list", screen.state_list):
            for _ in range(5):
                if screen.card_opened:
                    break
                step = engine._rewards(ctx)
                self.assertIsNone(step.decision)
                screen.step(step.auto)

        # Gold, then the special row, then the card item: the driver's order.
        self.assertEqual(screen.steps, [0, 1, 0])
        self.assertTrue(screen.card_opened)
        # The deck the card offer opens on is the game's: 20, the returned card last.
        self.assertEqual(screen.deck, decisions[61]["deck"])
        self.assertEqual(screen.deck[-1], ["CARD.FIGHT_ME", 0])
        self.assertEqual(screen.deck, decisions[62]["deck"][:-1])


class WongosLeaveVisibilityTests(unittest.TestCase):
    """D34 (owner, 2026-10-27): Welcome to Wongos' Leave (action 3) is shown to the policies from mask version 4.

    The reader dropped action 3 on every event page because it is the emulator-only leave everywhere but here: Wongos is the
    one event with a real fourth option. Versions 1-3 read the page as they always did; version 4 lists Leave after the
    purchases the purse allows. No other page changes.
    """

    @staticmethod
    def _page(event_id, legal, mask_version, event_step=0):
        import types

        env = types.SimpleNamespace(_mask_version=mask_version)
        ctx = engine._Context(env, 0, list(legal), None, event_step, False)
        with mock.patch.object(engine.native, "run_info", return_value=[0] * 9 + [event_id]):
            return engine._event(ctx)

    def test_versions_one_to_three_show_three_options_and_version_four_shows_four(self):
        for version in (1, 2, 3):
            with self.subTest(version=version):
                screen = self._page(engine.EVENT_WELCOME_TO_WONGOS, [0, 1, 2, 3], version)
                self.assertEqual([o.action for o in screen.decision.options], [0, 1, 2])
        screen = self._page(engine.EVENT_WELCOME_TO_WONGOS, [0, 1, 2, 3], 4)
        self.assertEqual([o.action for o in screen.decision.options], [0, 1, 2, 3])
        leave = screen.decision.options[3]
        self.assertEqual((leave.kind, leave.event), ("event.option", (engine.EVENT_WELCOME_TO_WONGOS, 3, 0)))

    def test_a_thin_purse_loses_only_the_purchases_it_cannot_make(self):
        self.assertEqual([o.action for o in self._page(41, [0, 3], 3).decision.options], [0])
        self.assertEqual([o.action for o in self._page(41, [0, 3], 4).decision.options], [0, 3])

    def test_a_page_with_only_leave_is_not_a_decision_in_any_version(self):
        for version in (1, 2, 3, 4):
            with self.subTest(version=version):
                screen = self._page(41, [3], version)
                self.assertIsNone(screen.decision)
                self.assertEqual((screen.auto, screen.auto_stat), (3, "forced_skip/event"))

    def test_no_other_event_page_shows_action_three_in_any_version(self):
        for event_id in range(0, 80):
            if event_id == engine.EVENT_WELCOME_TO_WONGOS:
                continue
            for version in (1, 2, 3, 4):
                with self.subTest(event_id=event_id, version=version):
                    screen = self._page(event_id, [0, 1, 3], version)
                    self.assertEqual([o.action for o in screen.decision.options], [0, 1])


class AutoScreenTests(unittest.TestCase):
    def test_the_treasure_room_takes_the_chest_relic(self):
        # Erratum RF1: action 0 takes the relic while it is in the chest, then 3 leaves.
        self.assertEqual(engine.treasure_action([0, 3]), 0)
        self.assertEqual(engine.treasure_action([3]), 3)

    def test_walking_through_a_treasure_room_adds_exactly_the_chest_relic(self):
        # The run engine grants the chest relic (run API 20, `StepTreasure`), and the
        # session's auto step takes it: the room adds one relic, and only one.
        def treasure_on_offer(_session, decision) -> bool:
            return decision.type == "map" and any(
                o.node is not None and o.node[2] == vocab.NODE_TREASURE
                for o in decision.options
            )

        for seed in ("TREAS001", "TREAS002", "TREAS003", "TREAS004", "TREAS005"):
            session = drive_until(
                seed,
                treasure_on_offer,
                prefer_node=vocab.NODE_TREASURE,
            )
            if session is None:
                continue
            try:
                decision = next_decision(session)
                chest = next(
                    o
                    for o in decision.options
                    if o.node is not None and o.node[2] == vocab.NODE_TREASURE
                )
                before = session.view().relics
                session.apply(chest)
                self.assertEqual(session.phase, rc.PHASE_TREASURE)
                self.assertEqual(next_decision(session).type, "map")
                after = session.view().relics
                self.assertEqual(len(after), len(before) + 1)
                self.assertEqual(session.info["relic_count"], len(after))
                gained = Counter(after) - Counter(before)
                self.assertEqual(sum(gained.values()), 1)
                return
            finally:
                session.close()
        self.fail("no seed reached a treasure room")


class OfferGridUpgradedFlagTests(unittest.TestCase):
    """An offer grid (Room Full of Cheese's eight, Brain Leech's five) keeps what its cards rolled.

    The run state keeps `PendingOfferUpgraded` (list 23) next to the grid (list 17); the reader built every
    `select.offer` card with `upgraded` False, so the referee's `[TWIN_STRIKE, 1]` could never equal the emulator's
    `[TWIN_STRIKE, 0]` (the 96th select record, seed NRBMFG74JL decision 84, floor 11).
    """

    TWIN_STRIKE = 7
    LISTS = {17: (3, TWIN_STRIKE, 9), 23: (0, 1, 0)}

    def _screen(self, lists):
        def run_state_list(handle, list_id, capacity=256, **kw):
            return lists.get(list_id, ())

        ctx = mock.Mock(handle=1, legal=[0, 1, 2, 3])
        with mock.patch.object(engine.native, "run_state_list", run_state_list):
            return engine._card_select(ctx).decision

    def test_a_card_the_offer_rolled_upgraded_is_read_upgraded(self):
        offers = {o.action: o.card for o in self._screen(self.LISTS).options if o.kind == "select.offer"}
        self.assertEqual(offers[1], features.CardRef(self.TWIN_STRIKE, True))
        self.assertEqual(offers[0], features.CardRef(3, False))
        self.assertEqual(offers[2], features.CardRef(9, False))

    def test_a_grid_with_no_flags_listed_reads_every_card_plain(self):
        decision = self._screen({17: self.LISTS[17]})
        cards = [o.card.upgraded for o in decision.options if o.kind == "select.offer"]
        self.assertEqual(cards, [False, False, False])


class ShopUpgradedFlagTests(unittest.TestCase):
    """A shop card keeps what it rolled (list 22, `ShopCardUpgraded`), as the offer grid's cards do (list 23).

    The reader built every `shop.card` with `upgraded` False, so the planner could not tell an upgraded card on the shelf
    from a plain one (the "+" the screen shows, and the card that lands in the deck).
    """

    ANGER, TWIN_STRIKE = 8, 7
    LISTS = {8: (ANGER, TWIN_STRIKE, 3, 0, 0, 0, 0), 4: (50, 60, 70) + (0,) * 11, 22: (0, 1, 0, 0, 0, 0, 0)}

    def _screen(self, lists, legal=(0, 1, 2)):
        def run_state_list(handle, list_id, capacity=256, **kw):
            return lists.get(list_id, ())

        ctx = mock.Mock(handle=1, legal=list(legal))
        with mock.patch.object(engine.native, "run_state_list", run_state_list):
            return engine._shop(ctx).decision

    def test_a_card_the_shop_rolled_upgraded_is_read_upgraded(self):
        cards = {o.action: (o.card, o.price) for o in self._screen(self.LISTS).options if o.kind == "shop.card"}
        self.assertEqual(cards[1], (features.CardRef(self.TWIN_STRIKE, True), 60))
        self.assertEqual(cards[0], (features.CardRef(self.ANGER, False), 50))
        self.assertEqual(cards[2], (features.CardRef(3, False), 70))

    def test_a_shop_with_no_flags_listed_reads_every_card_plain(self):
        decision = self._screen({k: v for k, v in self.LISTS.items() if k != 22})
        cards = [o.card.upgraded for o in decision.options if o.kind == "shop.card"]
        self.assertEqual(cards, [False, False, False])


if __name__ == "__main__":
    unittest.main()
