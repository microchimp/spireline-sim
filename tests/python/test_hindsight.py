"""Hindsight labels (rl/hindsight.py, brief 2a): the value of each option on a decision, played forward on resampled run clones.

The pure parts (seeds, the label, the statistics, the plan, the report, the machine guard) need no native library. The rest runs
on the real run engine, with fights resolved by the debug resolver or played at four simulations and a stub run head, so that
what is under test is the mechanism: common random numbers across the options of one draw, G as the death floor or as the head's
value at the next act, the clone leaving the base alone, and the replay being the run `gen0.play_emu_run` plays.
"""

import json
import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hindsight  # noqa: E402
import leakprobes  # noqa: E402

try:
    leakprobes.native_env()
    from sts2_gym import native  # noqa: E402

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    HAVE_NATIVE = False

WON = 53.0


class SeedTests(unittest.TestCase):
    def test_seeds_are_a_function_of_run_floor_kind_and_k_only(self):
        a = hindsight.resample_seed("RUN1", 14, "card", 3)
        self.assertEqual(a, hindsight.resample_seed("RUN1", 14, "card", 3))
        for other in ((("RUN2", 14, "card", 3)), ("RUN1", 15, "card", 3), ("RUN1", 14, "rest", 3), ("RUN1", 14, "card", 4)):
            self.assertNotEqual(a, hindsight.resample_seed(*other), other)
        self.assertTrue(0 <= a < 2**31 - 1)
        self.assertEqual(hindsight.stream_key("RUN1", 14, "card", 3), hindsight.stream_key("RUN1", 14, "card", 3))
        self.assertNotEqual(hindsight.stream_key("RUN1", 14, "card", 3, 0), hindsight.stream_key("RUN1", 14, "card", 3, 1))

    def test_seeds_live_in_their_own_namespace(self):
        import gen0

        self.assertNotIn(hindsight.resample_seed("RUN1", 14, "card", 0), (gen0.AGENT_SEED, 0))
        self.assertNotEqual(hindsight._digest("resample", "RUN1", 14, "card", 0), hindsight.NAMESPACE)
        self.assertEqual(hindsight.AGENT_SEED, gen0.AGENT_SEED)

    def test_the_coin_is_stable_and_spread(self):
        coins = [hindsight.uniform("RUN", i) for i in range(2000)]
        self.assertEqual(coins[5], hindsight.uniform("RUN", 5))
        self.assertTrue(all(0.0 <= c < 1.0 for c in coins))
        self.assertAlmostEqual(sum(coins) / len(coins), 0.5, delta=0.03)


class LabelTests(unittest.TestCase):
    def test_return_is_the_death_floor_the_head_value_or_the_won_floor(self):
        self.assertEqual(hindsight.label_return("death", 12, None, WON), 12.0)
        self.assertEqual(hindsight.label_return("death", 17, 30.0, WON), 17.0)  # a death never reads the head
        self.assertEqual(hindsight.label_return("next_act", 17, 31.5, WON), 31.5)
        self.assertEqual(hindsight.label_return("win", 51, None, WON), WON)

    def test_a_continuation_without_a_return_has_none(self):
        for end in ("timeout", "step_cap", "no_value", "?"):
            self.assertIsNone(hindsight.label_return(end, 9, 20.0, WON))
        self.assertIsNone(hindsight.label_return("next_act", 17, None, WON))

    def test_option_values_and_centring(self):
        g = [[10.0, 20.0], [30.0, 40.0], [20.0, 30.0]]
        q = hindsight.option_values(g)
        self.assertEqual(q, [15.0, 35.0, 25.0])
        self.assertEqual(hindsight.centred(q), [-10.0, 10.0, 0.0])
        self.assertAlmostEqual(sum(hindsight.centred([3.0, 9.5, 4.25, 1.0])), 0.0)

    def test_a_shift_that_moves_every_option_together_leaves_the_centred_values_alone(self):
        a = hindsight.centred(hindsight.option_values([[10.0, 14.0], [20.0, 22.0]]))
        b = hindsight.centred(hindsight.option_values([[25.0, 29.0], [35.0, 37.0]]))
        self.assertEqual(a, b)


class StatisticTests(unittest.TestCase):
    def test_paired_variance_is_zero_when_the_options_move_together(self):
        g = [[1.0, 5.0, 9.0], [2.0, 6.0, 10.0]]  # the same draws, one floor apart
        (v, df), = hindsight.paired_variances(g)
        self.assertEqual((v, df), (0.0, 2))
        self.assertGreater(hindsight.pooled(hindsight.single_variances(g)), 0.0)

    def test_paired_variance_matches_the_difference_series(self):
        g = [[0.0, 4.0, 2.0, 6.0], [1.0, 1.0, 5.0, 3.0]]  # differences 1, -3, 3, -3
        (v, df), = hindsight.paired_variances(g)
        self.assertAlmostEqual(v, hindsight.sample_var([1.0, -3.0, 3.0, -3.0]))
        self.assertEqual(df, 3)

    def test_pooling_weights_by_degrees_of_freedom(self):
        self.assertAlmostEqual(hindsight.pooled([(4.0, 1), (1.0, 3)]), (4.0 * 1 + 1.0 * 3) / 4)
        self.assertIsNone(hindsight.pooled([]))
        self.assertIsNone(hindsight.pooled([(float("nan"), 3), (2.0, 0)]))

    def test_advantage_variance_is_zero_for_options_that_differ_by_constants(self):
        g = [[1.0, 5.0], [3.0, 7.0], [2.0, 6.0]]
        self.assertTrue(all(v == 0.0 for v, _ in hindsight.advantage_variances(g)))

    def test_k_for_the_design_example(self):
        # the design's example: a paired SD of 4 floors, 1,500 decisions, 0.05 floors
        self.assertAlmostEqual(hindsight.k_needed(4.0, 1500, 0.05), 16 / 3.75)
        self.assertAlmostEqual(math.sqrt(16 / 4 / 1500), 0.0516, places=3)  # K 4: the design's "about 0.05"
        self.assertIsNone(hindsight.k_needed(4.0, 1500, 0.05, tau2=3.75))  # the decisions' own spread alone exceeds it
        self.assertGreater(hindsight.k_needed(4.0, 1500, 0.05, tau2=1.0), hindsight.k_needed(4.0, 1500, 0.05))


def fake_record(g, kind="card", act=0, floor=5, deaths=(), seconds=3.0, rid="R", heads=None):
    """A label record for the matrix `g[o][k]`; `deaths` are (o, k) pairs that died at the act's boss floor."""
    boss = hindsight.ACT_BOSS_FLOORS[act]
    conts = []
    for o, row in enumerate(g):
        for k, val in enumerate(row):
            died = (o, k) in deaths
            conts.append({"o": o, "k": k, "G": val, "end": "death" if died else "next_act", "floor": boss, "head": None if died else val,
                          "fights": 3, "moves": 40, "seconds": seconds, "cpu": seconds / 2, "door": None})
    q = hindsight.option_values(g)
    return {"id": f"{rid}|d000|{kind}|f{floor:02d}", "kind": kind, "act": act, "floor": floor, "k": len(g[0]),
            "played": list(range(len(g))), "continuations": conts, "head_values": heads or [None] * len(g),
            "Q": {str(o): v for o, v in enumerate(q)}}


class ReportTests(unittest.TestCase):
    def test_all_die_at_the_boss_and_flat_decisions(self):
        boss = float(hindsight.ACT_BOSS_FLOORS[0])
        dead = fake_record([[boss] * 4, [boss] * 4], deaths={(o, k) for o in range(2) for k in range(4)})
        live = fake_record([[30.0, 28.0, 31.0, 29.0], [26.0, 25.0, 27.0, 28.0]], rid="S")
        s = hindsight.summarize([dead, live])
        self.assertEqual(s["decisions"], 2)
        self.assertEqual(s["all_die_at_boss_share"], 0.5)
        self.assertEqual(s["flat_share"], 0.5)
        self.assertEqual(s["k"], 4)
        self.assertEqual(s["continuations"], 16)
        self.assertEqual(s["cont_seconds_mean"], 3.0)
        self.assertAlmostEqual(s["ends"]["death"], 0.5)

    def test_pairing_shows_as_option_correlation(self):
        base = [20.0, 30.0, 25.0, 35.0, 22.0, 28.0, 31.0, 24.0]
        rec = fake_record([base, [x + 1.0 for x in base]])
        s = hindsight.summarize([rec])
        self.assertEqual(s["paired_sd"], 0.0)
        self.assertEqual(s["option_corr"], 1.0)
        self.assertEqual(s["q_range_mean"], 1.0)

    def test_refused_records_are_counted_not_summarised(self):
        ok = fake_record([[1.0, 2.0], [3.0, 4.0]])
        bad = dict(fake_record([[1.0, 2.0], [3.0, 4.0]], rid="X"), refused="no_return:timeout")
        s = hindsight.summarize([ok, bad])
        self.assertEqual((s["decisions"], s["refused"]), (1, 1))
        self.assertEqual(hindsight.summarize([bad]), {"decisions": 0, "refused": 1})

    def test_report_reads_the_label_files_by_kind_and_act(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            sink = hindsight.labels_dir(out)
            sink.mkdir(parents=True)
            recs = [fake_record([[10.0, 12.0], [14.0, 16.0]], "card", 0, rid="A"), fake_record([[20.0, 22.0], [24.0, 26.0]], "rest", 1,
                                                                                                floor=25, rid="B")]
            (sink / "A.jsonl").write_text(json.dumps(recs[0]) + "\n" + '{"id": "cut off', encoding="utf-8")  # a line cut by a stop
            (sink / "B.jsonl").write_text(json.dumps(recs[1]) + "\n", encoding="utf-8")
            doc = hindsight.report(out)
            self.assertEqual(doc["all"]["decisions"], 2)
            self.assertEqual(sorted(doc["cells"]), ["card/a1", "rest/a2"])
            self.assertEqual(sorted(doc["by_act"]), ["a1", "a2"])
            self.assertEqual(hindsight.count_cells(hindsight.read_labels(out)), Counter({"card/a1": 1, "rest/a2": 1}))


class HeadAgreementTests(unittest.TestCase):
    def test_spearman(self):
        self.assertAlmostEqual(hindsight.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(hindsight.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(hindsight.spearman([1, 2, 3], [5, 5, 5]))
        self.assertAlmostEqual(hindsight.spearman([1, 2, 2, 3], [1, 2, 2, 3]), 1.0)  # ties share their mean rank

    def test_the_labels_against_the_heads_own_ranking(self):
        agree = fake_record([[10.0, 12.0], [20.0, 22.0], [15.0, 17.0]], heads=[1.0, 3.0, 2.0])
        differ = fake_record([[10.0, 12.0], [20.0, 22.0], [15.0, 17.0]], heads=[3.0, 1.0, 2.0], rid="S")
        out = hindsight.head_agreement([agree, differ])
        self.assertEqual(out["head_compared"], 2)
        self.assertEqual(out["head_top1_agree"], 0.5)
        self.assertAlmostEqual(out["head_label_spearman"], 0.0)  # the ranks agree on one decision (+1) and reverse on the other (-1)
        self.assertEqual(hindsight.head_agreement([fake_record([[1.0, 2.0], [3.0, 4.0]])]), {"head_top1_agree": None,
                                                                                         "head_label_spearman": None, "head_compared": 0})


class PlanTests(unittest.TestCase):
    def test_wants_post_boss_acts_kinds_and_caps(self):
        plan = hindsight.Plan(per_run=2, rates={"card": 1.0, "rest": 1.0, "route": 1.0, "shop": 1.0})
        none = Counter()
        self.assertIsNone(plan.wants("R", 1, "card", 0, 5, none, none))
        self.assertEqual(plan.wants("R", 1, None, 0, 5, none, none), "kind")
        self.assertEqual(plan.wants("R", 1, "card", 2, 40, none, none), "act")  # Act 3 is not labelled
        self.assertEqual(plan.wants("R", 1, "card", 0, 17, none, none), "post_boss")  # the boss reward: nothing left to play
        self.assertEqual(plan.wants("R", 1, "card", 1, 33, none, none), "post_boss")
        self.assertIsNone(plan.wants("R", 1, "rest", 0, 16, none, none))  # the rest before the boss is a decision
        self.assertEqual(plan.wants("R", 1, "card", 0, 5, Counter({"card/a1": 1}), none), "run_cap")  # one a kind and act
        self.assertEqual(plan.wants("R", 1, "card", 0, 5, Counter({"rest/a1": 1, "route/a1": 1}), none), "run_cap")  # two a run

    def test_coin_and_quota(self):
        plan = hindsight.Plan(quota={"card/a1": 2}, rates={"card": 0.0})
        self.assertEqual(plan.wants("R", 1, "card", 0, 5, Counter(), Counter()), "coin")
        plan = hindsight.Plan(quota={"card/a1": 2}, rates={"card": 1.0})
        self.assertIsNone(plan.wants("R", 1, "card", 0, 5, Counter(), Counter({"card/a1": 1})))
        self.assertEqual(plan.wants("R", 1, "card", 0, 5, Counter(), lambda: Counter({"card/a1": 2})), "quota")
        self.assertEqual(plan.open_cells(Counter({"card/a1": 2})), [])
        self.assertEqual(plan.open_cells(Counter()), ["card/a1"])

    def test_the_coin_depends_on_the_run_and_ordinal_only(self):
        plan = hindsight.Plan()
        got = [plan.wants("R", i, "route", 0, 5, Counter(), Counter()) for i in range(400)]
        self.assertEqual(got, [plan.wants("R", i, "route", 0, 5, Counter(), Counter()) for i in range(400)])
        share = sum(1 for g in got if g is None) / 400
        self.assertAlmostEqual(share, hindsight.DEFAULT_RATES["route"], delta=0.05)

    def test_quota_text(self):
        self.assertEqual(hindsight.parse_quota("card/a1=50, rest/a2=7"), {"card/a1": 50, "rest/a2": 7})
        self.assertEqual(hindsight.parse_quota(""), {})


class Opt:
    def __init__(self, kind, scorable=True):
        self.kind, self.scorable = kind, scorable


class ShortlistTests(unittest.TestCase):
    def test_few_options_are_all_played_except_unknown(self):
        opts = [Opt("card.take"), Opt("card.take"), Opt("unknown"), Opt("card.skip")]
        self.assertEqual(hindsight.shortlist(opts, [1.0, 2.0, None, 0.0], limit=6), [0, 1, 3])

    def test_many_options_keep_the_unchanged_the_runs_pick_and_the_heads_best(self):
        opts = [Opt("shop.card")] * 9 + [Opt("shop.leave")]
        values = [5.0, 9.0, 1.0, 2.0, 8.0, 3.0, 4.0, 7.0, 6.0, None]
        got = hindsight.shortlist(opts, values, limit=5, keep=[2])
        self.assertEqual(got, [1, 2, 4, 7, 9])  # leave, the run's pick (2), the best three of the rest
        self.assertEqual(len(hindsight.shortlist(opts, values, limit=3, keep=[])), 3)


class FieldsStub:
    def __init__(self, cards, rests, nodes):
        self.cards, self.rests, self.nodes = cards, rests, nodes


class PrefixTests(unittest.TestCase):
    RECORD = {"cards": [[2, "BASH"], [3, "skip"]], "rests": [[8, "smith"]], "nodes": "MMR?"}

    def test_a_prefix_of_the_record_passes(self):
        self.assertIsNone(hindsight.prefix_check(FieldsStub([[2, "BASH"]], [], list("MM")), self.RECORD))
        self.assertIsNone(hindsight.prefix_check(FieldsStub([], [], []), self.RECORD))

    def test_a_replay_that_has_left_its_record_names_the_field(self):
        self.assertEqual(hindsight.prefix_check(FieldsStub([[2, "ANGER"]], [], []), self.RECORD), "cards")
        self.assertEqual(hindsight.prefix_check(FieldsStub([], [[8, "heal"]], []), self.RECORD), "rests")
        self.assertEqual(hindsight.prefix_check(FieldsStub([], [], list("MR")), self.RECORD), "nodes")
        self.assertEqual(hindsight.prefix_check(FieldsStub([[2, "BASH"], [3, "skip"], [4, "BASH"]], [], []), self.RECORD), "cards")


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.quiet, self.main_lock, self.active = d / "QUIET.lock", d / "main.lock", d / "active"
        self.patches = [mock.patch.object(hindsight, "QUIET", self.quiet), mock.patch.object(hindsight, "MAIN_LOCK", self.main_lock),
                        mock.patch.object(hindsight, "ACTIVE", self.active)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def line(self, tag, age_h=0.0):
        t = hindsight.utc() - hindsight.dt.timedelta(hours=age_h)
        return f"{tag} {t.strftime('%Y-%m-%dT%H:%M:%SZ')} pid=1\n"

    def test_free_machine(self):
        self.assertIsNone(hindsight.blocked())

    def test_another_tag_the_night_and_main_lock_block_but_not_our_own_or_the_loops(self):
        self.quiet.write_text(self.line("hindsight") + self.line("gym-loop"), encoding="utf-8")
        self.assertIsNone(hindsight.blocked())
        self.quiet.write_text(self.line("practice-gate"), encoding="utf-8")
        self.assertEqual(hindsight.blocked(), "quiet:practice-gate")
        self.quiet.write_text(self.line("practice-gate", age_h=3.0), encoding="utf-8")  # stale
        self.assertIsNone(hindsight.blocked())
        self.quiet.unlink()
        self.main_lock.write_text("1", encoding="utf-8")
        self.assertEqual(hindsight.blocked(), "main.lock")
        self.main_lock.unlink()
        self.active.write_text("", encoding="utf-8")
        self.assertEqual(hindsight.blocked(), "nightly-active")

    def test_hold_keeps_other_lines_and_removes_only_ours(self):
        self.quiet.write_text(self.line("gym-loop"), encoding="utf-8")
        hindsight.hold(True)
        tags = [x.split()[0] for x in self.quiet.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(tags, ["gym-loop", "hindsight"])
        self.assertIn("pid=", self.quiet.read_text(encoding="utf-8"))
        hindsight.hold(False)
        self.assertEqual([x.split()[0] for x in self.quiet.read_text(encoding="utf-8").splitlines()], ["gym-loop"])
        self.quiet.unlink()
        hindsight.hold(True)
        hindsight.hold(False)
        self.assertFalse(self.quiet.exists())


# ------------------------------------------------------------------------------------------------ on the real run engine


def resolve_fight(s, rng, deadline=None):
    """A fight player that wins the open fight without playing it (the debug resolver)."""
    native.run_debug_resolve_combat(s.handle, s.env._run_obs_buf)
    return 0


class MapFirst:
    """The run policy of the fixtures: the first node on a map, the rules everywhere else (deterministic, draws nothing)."""

    def __init__(self):
        from run_train import selfplay

        self.rules = selfplay.RulesPlayer()

    def pick(self, view, decision, rng):
        from run_train import selfplay

        return selfplay.Pick(0) if decision.type == "map" else self.rules.pick(view, decision, rng)


def fixture_player(head=None, fight=resolve_fight, wall_cap_s=60.0):
    return hindsight.Player(policy=MapFirst(), play_fight=fight, head=head or (lambda view, d: None), won_floor=WON, wall_cap_s=wall_cap_s)


def play_to_card_reward(seed="HSFIX1", player=None):
    """A session of the fixture run, advanced (fights resolved, rules choices) to its first card reward; the decision."""
    import numpy as np
    from run_train import engine

    player = player or fixture_player()
    s = engine.RunSession(seed, 10, engine.CombatBudget(4, 4))
    rng = np.random.default_rng(0)
    for _ in range(300):
        d = s.decision()
        if d is None:
            player.play_fight(s, rng, None)
            continue
        if d.type == "card_reward":
            return s, d
        s.apply(d.options[player.policy.pick(s.view(), d, rng).index])
    raise AssertionError("no card reward reached")


@unittest.skipUnless(HAVE_NATIVE, "needs the native emulator library")
class CloneTests(unittest.TestCase):
    def test_the_clone_leaves_the_base_alone(self):
        s, d = play_to_card_reward()
        try:
            before = (s.info, s.view(), s.steps, dict(s.visit.__dict__))
            c = hindsight.clone_session(s, hindsight.resample_seed("HSFIX1", s.floor, "card", 0))
            c.apply(d.options[-1])  # skip
            c.decision()
            c.close()
            self.assertEqual((s.info, s.view(), s.steps, dict(s.visit.__dict__)), before)
            self.assertIsNotNone(s.decision())  # the base still stands at its decision
        finally:
            s.close()

    def test_options_of_one_draw_meet_the_same_first_encounter_and_map_and_other_draws_do_not(self):
        s, d = play_to_card_reward()
        policy = fixture_player().policy
        import numpy as np

        rng = np.random.default_rng(0)
        met = {}
        try:
            floor = s.floor
            for k in range(1, 7):
                for oi in (0, len(d.options) - 1):  # take the first card / skip
                    c = hindsight.clone_session(s, hindsight.resample_seed("HSFIX1", floor, "card", k))
                    try:
                        c.apply(d.options[oi])
                        while (e := c.decision()) is not None:
                            c.apply(e.options[policy.pick(c.view(), e, rng).index])
                        fight = c.combat()
                        encounter = fight.encounter_id
                        c.env.debug_enter_next_act()  # the next act's contents are redrawn from the clone's seed
                        met[(k, oi)] = (encounter, c.view().map_nodes)
                    finally:
                        c.close()
            for k in range(1, 7):
                self.assertEqual(met[(k, 0)], met[(k, len(d.options) - 1)], f"draw {k}: the options met different futures")
            self.assertGreater(len({met[(k, 0)][0] for k in range(1, 7)}), 1, "the draws all met one encounter")
            self.assertEqual(len({met[(k, 0)][1] for k in range(1, 7)}), 6, "the draws all met one map")
        finally:
            s.close()


@unittest.skipUnless(HAVE_NATIVE, "needs the native emulator library")
class ReturnTests(unittest.TestCase):
    def test_a_forced_death_returns_the_death_floor(self):
        s, d = play_to_card_reward()
        try:
            floor = s.floor
            s.env.debug_set_player(hp=1)  # the clone inherits one hit point
            seen = []
            player = fixture_player(head=lambda v, dd: seen.append(v) or 99.0, fight=lambda *a: _end_turns(*a))
            row = hindsight.continuation(s, d.options[-1], player, run_id="HSFIX1", floor=floor, kind="card", k=0)
            self.assertEqual(row["end"], "death")
            self.assertEqual(row["G"], float(row["floor"]))
            self.assertGreaterEqual(row["floor"], floor)
            self.assertEqual(seen, [], "a death must not read the head")
        finally:
            s.close()

    def test_the_head_value_at_the_next_acts_first_decision_otherwise(self):
        s, d = play_to_card_reward()
        try:
            calls = []

            def head(view, decision):
                calls.append((view.act_index, view.floor))
                return 31.5 if view.act_index == 1 else None

            player = fixture_player(head=head)
            row = hindsight.continuation(s, d.options[0], player, run_id="HSFIX1", floor=s.floor, kind="card", k=0)
            self.assertEqual(row["end"], "next_act")
            self.assertEqual(row["G"], 31.5)
            self.assertEqual(row["head"], 31.5)
            self.assertTrue(calls and all(act == 1 for act, _ in calls), calls)  # asked only once the next act began
            self.assertGreaterEqual(calls[0][1], hindsight.ACT_BOSS_FLOORS[0])  # the next act opens on the boss floor's number
            self.assertEqual(row["floor"], calls[0][1])
        finally:
            s.close()

    def test_a_head_with_no_value_in_the_next_act_is_no_return(self):
        s, d = play_to_card_reward()
        try:
            player = fixture_player(head=lambda v, dd: None)
            row = hindsight.continuation(s, d.options[0], player, run_id="HSFIX1", floor=s.floor, kind="card", k=0)
            self.assertEqual(row["end"], "no_value")
            self.assertIsNone(row["G"])
        finally:
            s.close()

    def test_the_wall_cap_ends_a_continuation_as_a_timeout(self):
        s, d = play_to_card_reward()
        try:
            player = fixture_player(wall_cap_s=-1.0)
            row = hindsight.continuation(s, d.options[0], player, run_id="HSFIX1", floor=s.floor, kind="card", k=0)
            self.assertEqual((row["end"], row["G"]), ("timeout", None))
        finally:
            s.close()

    def test_the_door_is_recorded_at_the_boss(self):
        s, d = play_to_card_reward()
        try:
            player = fixture_player(head=lambda v, dd: 30.0 if v.act_index == 1 else None)
            player.judge = lambda handle, max_hp: {"value": -12.0, "p_death": 0.25}
            row = hindsight.continuation(s, d.options[0], player, run_id="HSFIX1", floor=s.floor, kind="card", k=0)
            door = row["door"]
            self.assertEqual(door["floor"], hindsight.ACT_BOSS_FLOORS[0])
            self.assertEqual(door["judge"], {"value": -12.0, "p_death": 0.25})
            self.assertGreater(door["max_hp"], 0)
        finally:
            s.close()

    def test_every_option_of_a_draw_plays_the_same_seeds_so_the_label_is_reproducible(self):
        s, d = play_to_card_reward()
        try:
            head = lambda v, dd: 20.0 + v.floor  # noqa: E731 -- a stub that depends on the state it is asked at
            a = hindsight.label_decision(s, d, s.view(), _Pick(len(d.options) - 1), fixture_player(head=head), run_id="HSFIX1",
                                         ordinal=3, kind="card", k=3)
            b = hindsight.label_decision(s, d, s.view(), _Pick(len(d.options) - 1), fixture_player(head=head), run_id="HSFIX1",
                                         ordinal=3, kind="card", k=3)
            strip = lambda r: [{k: v for k, v in c.items() if k not in ("seconds", "cpu")} for c in r["continuations"]]  # noqa: E731
            self.assertEqual(strip(a), strip(b))
            self.assertEqual(a["Q"], b["Q"])
            self.assertEqual(sorted(a["played"]), a["played"])
            self.assertAlmostEqual(sum(a["A"].values()), 0.0, places=3)
            self.assertEqual(a["id"], "HSFIX1|d003|card|f02")
            self.assertEqual({(c["o"], c["k"]) for c in a["continuations"]}, {(o, k) for o in a["played"] for k in range(3)})
        finally:
            s.close()


class _Pick:
    def __init__(self, index):
        self.index = index


def _end_turns(s, rng, deadline=None):
    from run_train import engine

    n = 0
    while s.phase == engine.rc.PHASE_COMBAT and n < 400:
        legal = s.combat().legal()
        s.combat().step(max(legal))
        n += 1
    return n


@unittest.skipUnless(HAVE_NATIVE, "needs the native emulator library")
class ReplayTests(unittest.TestCase):
    CONFIG = "flat-4"

    def play(self, seed, ix):
        import gen0
        import gen_loop

        return gen0.play_emu_run(seed, ix, self.CONFIG, agent=gen_loop.Agent())

    def player(self):
        import gen_loop
        from run_train import selfplay

        agent = gen_loop.Agent()
        return hindsight.Player(policy=selfplay.RulesPlayer(), play_fight=hindsight.SearchFighter(agent, self.CONFIG),
                                head=lambda v, d: None, won_floor=WON, agent=agent)

    def test_the_replay_is_the_run_gen0_plays_and_the_hook_sees_every_decision(self):
        row = self.play("HSRPL1", 4)
        seen = []
        res = hindsight.replay_run("HSRPL1", 4, self.player(), row, lambda pt: seen.append((pt.ordinal, pt.kind, pt.view.floor)) and False,
                                   config=self.CONFIG)
        self.assertEqual(res["end"], row["end"], res)
        self.assertEqual(res["floor"], row["floor"])
        self.assertEqual(res["decisions"], res["checked"])
        self.assertEqual([o for o, _, _ in seen], list(range(len(seen))))
        self.assertTrue({k for _, k, _ in seen} >= {"card", "route"}, seen)

    def test_a_replay_that_has_left_its_record_stops_by_reason(self):
        row = self.play("HSRPL1", 4)
        bad = dict(row, cards=[[row["cards"][0][0], "NOT_A_CARD"], *row["cards"][1:]])
        res = hindsight.replay_run("HSRPL1", 4, self.player(), bad, lambda pt: False, config=self.CONFIG)
        self.assertEqual(res["end"], "diverged:cards")
        worse = dict(row, nodes="Z" + row["nodes"][1:])
        self.assertEqual(hindsight.replay_run("HSRPL1", 4, self.player(), worse, lambda pt: False, config=self.CONFIG)["end"], "diverged:nodes")

    def test_a_hook_that_stops_ends_the_replay_at_that_decision(self):
        row = self.play("HSRPL1", 4)
        res = hindsight.replay_run("HSRPL1", 4, self.player(), row, lambda pt: pt.ordinal == 5, config=self.CONFIG)
        self.assertEqual(res["end"], "hooked")
        self.assertEqual(res["decisions"], 5)


class HeadTests(unittest.TestCase):
    def test_head_value_is_the_best_afterstate_value(self):
        import numpy as np
        from test_run_train_features import make_view
        from test_run_train_policy import REWARD, save_checkpoint

        from run_train import policy as run_policy

        with tempfile.TemporaryDirectory() as d:
            npz = save_checkpoint(Path(d), seed=3)
            pol = run_policy.LookaheadPolicy(npz, 0.0, eps=0.0, W=26.0)
            view = make_view()
            heads = hindsight.option_head_values(pol, view, REWARD)
            self.assertEqual(len(heads), len(REWARD.options))
            self.assertAlmostEqual(hindsight.head_value(pol, view, REWARD), max(heads))
            pick = pol.pick(view, REWARD, None)
            self.assertEqual(int(np.argmax(heads)), pick.index)


if __name__ == "__main__":
    unittest.main()
