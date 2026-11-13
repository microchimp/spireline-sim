"""The parity side blocks of an option's row (`rl/parity_blocks.py`, D59 lane D).

Pinned: `BASE_SIZE` 1855 and the feature version are untouched, and the base row of a view is byte for byte as before (a digest taken on main before the blocks); the widths; an option's afterstate
(a relic taken is held and is the option's relic, a card taken is in the deck's roles and is the option's card, a removal takes the card out of the deck's counts but names it, a random transform,
an unchanged option); an event option carries its payoff and any other option zeros there; the blocks are the same numbers whichever way they are asked for.
"""

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "rl", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import parity_blocks as pb  # noqa: E402
import parity_cards as pc  # noqa: E402
import parity_events as pe  # noqa: E402
import parity_potions as pp  # noqa: E402
import parity_relics as pr  # noqa: E402
from run_train import features, vocab  # noqa: E402
from run_train.features import CardRef, MapNode, Option, RunView  # noqa: E402

M, E, R, S, T, B, U, A = (
    vocab.NODE_MONSTER, vocab.NODE_ELITE, vocab.NODE_REST, vocab.NODE_SHOP, vocab.NODE_TREASURE, vocab.NODE_BOSS, vocab.NODE_UNKNOWN, vocab.NODE_ANCIENT,
)
MAP = (MapNode(3, 0, A, ((2, 1), (4, 1))), MapNode(2, 1, M, ((2, 2),)), MapNode(4, 1, U, ((3, 2),)), MapNode(2, 2, E, ((3, 3),)), MapNode(3, 2, R, ((3, 3),)),
       MapNode(3, 3, S, ((3, 4),)), MapNode(3, 4, B, ()))
PINNED_ROW_DIGEST = "beda711f28f75a39"  # the same view and digest as test_features_state_gaps: `encode(make_view())` as main built it before block G


def cid(name):
    return vocab.card_id_by_name(name)


def rid(name):
    return vocab.relic_id_by_name(name)


def make_view(**over):
    f = {"ascension": 10, "act_index": 0, "act_region": vocab.REGION_OVERGROWTH, "floor": 3, "hp": 50, "max_hp": 80, "gold": 120,
         "deck": (CardRef(472, False), CardRef(131, False), CardRef(30, False)), "relics": (rid("BurningBlood"),), "potions": (), "max_potion_slots": 3, "map_nodes": MAP,
         "position": (3, 0), "boss": 83}
    f.update(over)
    return RunView(**f)


def pin_view():
    """The map-grid and state-gaps tests' view, whose base row the digest pins."""
    nodes = (MapNode(3, 0, A, ((2, 1), (4, 1))), MapNode(2, 1, M, ((2, 2), (3, 2))), MapNode(4, 1, U, ((3, 2), (5, 2))), MapNode(2, 2, E, ((3, 3),)), MapNode(3, 2, R, ((3, 3),)),
             MapNode(5, 2, T, ()), MapNode(3, 3, S, ((3, 4),)), MapNode(3, 4, B, ()))
    return make_view(map_nodes=nodes)


class UntouchedTests(unittest.TestCase):
    def test_the_base_vector_and_version_are_unchanged(self):
        self.assertEqual(features.BASE_SIZE, 1855)
        self.assertEqual(features.FEATURE_VERSION, 2)
        row = features.encode(pin_view())
        self.assertEqual(hashlib.sha256(row.tobytes()).hexdigest()[:16], PINNED_ROW_DIGEST)


class WidthTests(unittest.TestCase):
    def test_widths(self):
        out = pb.option_parity(make_view(), [Option("card.skip", 0)])
        self.assertEqual(set(out), set(pb.BLOCK_NAMES))
        for name, arr in out.items():
            self.assertEqual(arr.shape, (1, pb.WIDTHS[name]))
            self.assertEqual(arr.dtype, np.float32)
        self.assertEqual(pb.WIDTHS["relics"], 2 * pr.RELIC_TAG_WIDTH)
        self.assertEqual(pb.WIDTHS["roles"], pc.N_ROLES + 1 + pc.N_ROLES)
        self.assertEqual(pb.WIDTHS["events"], pe.EVENT_WIDTH)
        self.assertEqual(pb.WIDTHS["potions"], 2 * pp.POTION_TAG_WIDTH)

    def test_no_options_is_an_empty_array(self):
        out = pb.option_parity(make_view(), [])
        for name, arr in out.items():
            self.assertEqual(arr.shape, (0, pb.WIDTHS[name]))


class OptionTests(unittest.TestCase):
    def test_a_relic_taken_is_held_and_is_the_options_relic(self):
        v = make_view()
        w = pr.RELIC_TAG_WIDTH
        out = pb.option_parity(v, [Option("neow.relic", 0, relic=rid("Lantern")), Option("neow.relic", 1, relic=rid("Vajra"))])
        np.testing.assert_allclose(out["relics"][0, :w], pr.held_tags((rid("BurningBlood"), rid("Lantern"))), atol=1e-7)
        np.testing.assert_allclose(out["relics"][0, w:], pr.relic_tags(rid("Lantern")), atol=1e-7)
        np.testing.assert_allclose(out["relics"][1, w:], pr.relic_tags(rid("Vajra")), atol=1e-7)
        self.assertGreater(out["relics"][0, :w][pr.EFFECTS.index("energy")], 0)  # Lantern
        self.assertGreater(out["relics"][1, :w][pr.EFFECTS.index("strength")], 0)  # Vajra

    def test_an_unchanged_option_holds_the_views_relics_and_names_none(self):
        v = make_view()
        out = pb.option_parity(v, [Option("card.skip", 0)])
        np.testing.assert_allclose(out["relics"][0, : pr.RELIC_TAG_WIDTH], pr.held_tags(v.relics), atol=1e-7)
        np.testing.assert_array_equal(out["relics"][0, pr.RELIC_TAG_WIDTH:], 0)
        np.testing.assert_allclose(out["roles"][0, : pc.N_ROLES + 1], pc.deck_roles(v.deck), atol=1e-7)
        np.testing.assert_array_equal(out["roles"][0, pc.N_ROLES + 1:], 0)

    def test_a_card_taken_is_in_the_decks_roles_and_is_the_options_card(self):
        v = make_view()
        card = CardRef(cid("Whirlwind"), False)
        out = pb.option_parity(v, [Option("card.take", 0, card=card), Option("card.skip", 1)])
        base = pc.deck_roles(v.deck)
        k = pc.ROLES.index("multi_target")
        self.assertAlmostEqual(float(out["roles"][0, k]) - float(base[k]), 1 / pc.COUNT_SCALE, places=6)
        self.assertAlmostEqual(float(out["roles"][0, pc.N_ROLES]) - float(base[pc.N_ROLES]), 1 / pc.DECK_SIZE_SCALE, places=6)
        self.assertEqual(float(out["roles"][0, pc.N_ROLES + 1 + k]), 0.5)
        self.assertEqual(float(out["roles"][1, k]), float(base[k]))

    def test_a_removal_leaves_the_deck_and_names_the_card(self):
        card = CardRef(cid("DemonForm"), False)
        v = make_view(deck=(CardRef(472, False), CardRef(131, False), card))
        index = v.deck.index(card)
        out = pb.option_parity(v, [Option("select.remove", 0, deck_index=index)])
        k = pc.ROLES.index("scaling_damage")
        self.assertEqual(float(out["roles"][0, k]), 0.0)  # the deck lost it
        self.assertEqual(float(out["roles"][0, pc.N_ROLES + 1 + k]), 0.5)  # and the option names it

    def test_an_upgrade_changes_the_deck_roles_through_the_printed_numbers(self):
        v = make_view(deck=(CardRef(472, False), CardRef(131, False)))
        index = v.deck.index(CardRef(131, False))
        out = pb.option_parity(v, [Option("rest.smith", 0, deck_index=index)])
        k = pc.ROLES.index("repeatable_block")
        self.assertGreater(float(out["roles"][0, k]), 0.0)  # an upgraded Defend blocks 8 for one energy

    def test_a_random_transform_does_not_fail(self):
        v = make_view()
        out = pb.option_parity(v, [Option("select.transform", 0, deck_index=0, arg=0)])
        self.assertEqual(out["roles"].shape, (1, pb.WIDTHS["roles"]))
        self.assertAlmostEqual(float(out["roles"][0, pc.N_ROLES]), 2 / pc.DECK_SIZE_SCALE, places=6)

    def test_an_event_option_carries_its_payoff_and_other_options_zeros(self):
        key = next(k for k, v in pe.payoff_table().items() if v["gold"] > 20)
        v = make_view()
        out = pb.option_parity(v, [Option("event.option", 0, event=key), Option("card.skip", 1)])
        np.testing.assert_array_equal(out["events"][0], pe.event_payoff(key))
        np.testing.assert_array_equal(out["events"][1], np.zeros(pe.EVENT_WIDTH, dtype=np.float32))
        self.assertEqual(float(out["events"][0, 0]), 1.0)

    def test_a_potion_taken_is_held_and_is_the_options_potion(self):
        v = make_view()
        w = pp.POTION_TAG_WIDTH
        block = {v: k for k, v in __import__("sts2_gym").names.potions().items()}["BlockPotion"]
        out = pb.option_parity(v, [Option("reward.potion", 0, potion=block), Option("reward.proceed", 1)])
        np.testing.assert_allclose(out["potions"][0, :w], pp.potion_tags(block), atol=1e-7)
        np.testing.assert_allclose(out["potions"][0, w:], pp.potion_tags(block), atol=1e-7)
        np.testing.assert_array_equal(out["potions"][1], np.zeros(2 * w, dtype=np.float32))
        held = make_view(potions=(block,))
        out2 = pb.option_parity(held, [Option("reward.proceed", 0)])
        np.testing.assert_allclose(out2["potions"][0, :w], pp.potion_tags(block), atol=1e-7)

    def test_the_same_numbers_whichever_way_they_are_asked_for(self):
        v = make_view()
        opts = [Option("card.take", 0, card=CardRef(cid("Inflame"), False)), Option("card.skip", 1)]
        a = pb.option_parity(v, opts)
        b = [pb.option_parity(v, [o]) for o in opts]
        for name in pb.BLOCK_NAMES:
            np.testing.assert_array_equal(a[name], np.concatenate([x[name] for x in b]))


class FromBaseTests(unittest.TestCase):
    """The held parts of P1, P2 and P4 are linear in the base vector's counts (block E: copies of each card, base and upgraded; F: relics; G: potions), so a row that has only its base vector (a
    human shard, whose rows are stored as counts) gets them with three matrix products and no view: `held_from_base` equals `option_parity` on every option of real decisions."""

    @staticmethod
    def decisions(n_runs=4):
        import random

        from run_train import vocab as v
        from run_train.engine import RunSession
        from sts2_gym import native
        from sts2_gym import run_constants as rc

        out = []
        for i in range(n_runs):
            rng = random.Random(i)
            s = RunSession(f"PARB{i:06d}", 10)
            try:
                for _ in range(1500):
                    if s.phase == rc.PHASE_COMPLETE:
                        break
                    if s.phase == rc.PHASE_COMBAT:
                        native.run_debug_resolve_combat(s.handle, s.env._run_obs_buf)
                        continue
                    d = s.decision()
                    if d is None:
                        continue
                    out.append((s.view(), d))
                    opts = list(d.options)
                    if d.type == "map":
                        unk = [o for o in opts if o.node and o.node[2] == v.NODE_UNKNOWN]
                        pick = rng.choice(unk or opts)
                    else:
                        pick = rng.choice(opts)
                    s.apply(pick)
            finally:
                s.close()
        return out

    def test_held_from_base_equals_the_option_parity_of_real_decisions(self):
        checked = 0
        kinds = set()
        for view, d in self.decisions():
            v = d.view or view
            opts = [o for o in d.options if o.scorable and o.kind != "select.transform"]
            if not opts:
                continue
            base = np.stack([features.encode_afterstate(v, o) for o in opts])
            got = pb.held_from_base(base)
            want = pb.option_parity(v, opts)
            np.testing.assert_allclose(got["relics"], want["relics"][:, : pr.RELIC_TAG_WIDTH], atol=1e-5)
            np.testing.assert_allclose(got["roles"], want["roles"][:, : pc.N_ROLES + 1], atol=1e-5)
            np.testing.assert_allclose(got["potions"], want["potions"][:, : pp.POTION_TAG_WIDTH], atol=1e-5)
            checked += len(opts)
            kinds.update(o.kind for o in opts)
        self.assertGreater(checked, 300)
        self.assertGreaterEqual(len(kinds), 8, kinds)

    def test_a_random_transform_is_the_expectation_the_base_vector_holds(self):
        v = make_view()
        o = Option("select.transform", 0, deck_index=0, arg=0)
        got = pb.held_from_base(features.encode_afterstate(v, o)[None, :])
        self.assertEqual(got["roles"].shape, (1, pc.N_ROLES + 1))
        self.assertAlmostEqual(float(got["roles"][0, pc.N_ROLES]), 3 / pc.DECK_SIZE_SCALE, places=5)  # the deck keeps its size: one card out, the expectation of the pool in


if __name__ == "__main__":
    unittest.main()
