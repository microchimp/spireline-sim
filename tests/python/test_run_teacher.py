"""The run-level teacher's measurement (rl/run_teacher.py; docs/rl/RUN_SELF_PLAY.md).

The pure parts need no native library: the door leaf, the teacher's pick (the player's blend with the played values in place of V,
the guard, the rest site's two stages), the split halves, the noise and spread, the door-leaf rescoring of stored labels, the report,
the seeds and the machine guard. The native parts (the continuations to the door and the run loop) are checked on the real engine
when its library is built: a run with the player deciding is the run `gen0.play_emu_run` plays, and a teacher's draw is the same
clone for every option.
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(p))

import run_teacher as rt  # noqa: E402


class LeafTests(unittest.TestCase):
    def test_death_win_and_door(self):
        self.assertEqual(rt.door_return("death", 12, None), 12.0)
        self.assertEqual(rt.door_return("win", 50, None), rt.WON_FLOOR)
        beyond = {0: 30.0, 1: 42.0}
        self.assertAlmostEqual(rt.door_return("door", 17, 0.25, beyond), 0.25 * 17 + 0.75 * 30.0)
        self.assertAlmostEqual(rt.door_return("door", 33, 1.0, beyond), 33.0)
        self.assertAlmostEqual(rt.door_return("door", 33, 0.0, beyond), 42.0)

    def test_no_return_for_the_rest(self):
        for end in ("timeout", "step_cap", "?", "next_act"):
            self.assertIsNone(rt.door_return(end, 10, 0.5))
        self.assertIsNone(rt.door_return("door", 17, None))
        self.assertIsNone(rt.door_return("door", 48, 0.5))  # Act 3's door: no value for surviving it
        self.assertIsNone(rt.door_return("door", 17, 1.5))

    def test_a_lower_death_risk_at_the_door_is_worth_more(self):
        self.assertGreater(rt.door_return("door", 17, 0.3), rt.door_return("door", 17, 0.6))
        self.assertGreater(rt.door_return("door", 17, 0.99), rt.door_return("death", 16, None))

    def test_act_of_floor(self):
        self.assertEqual([rt.act_of_floor(f) for f in (1, 17, 18, 33, 34, 48)], [0, 0, 1, 1, 2, 2])
        self.assertEqual(rt.LAST_FLOOR, 33)


class PickTests(unittest.TestCase):
    def test_plain_screen_takes_the_highest_value_ties_to_the_player(self):
        self.assertEqual(rt.teacher_index({0: 20.0, 2: 21.0, 3: 19.0}, 4, 0), 2)
        self.assertEqual(rt.teacher_index({0: 20.0, 2: 20.0}, 4, 2), 2)
        self.assertEqual(rt.teacher_index({0: 20.0, 2: 20.0}, 4, 3), 0)
        self.assertEqual(rt.teacher_index({}, 4, 1), 1)

    def test_blended_screen_adds_the_prior(self):
        logp = [math.log(0.7), math.log(0.2), math.log(0.1)]
        # 0.5 * (log 0.7 - log 0.2) = 0.63 floors: option 1 needs more than that over option 0
        self.assertEqual(rt.teacher_index({0: 20.0, 1: 20.5, 2: 20.0}, 3, 0, screen="card", lam=0.5, logp=logp), 0)
        self.assertEqual(rt.teacher_index({0: 20.0, 1: 20.7, 2: 20.0}, 3, 0, screen="card", lam=0.5, logp=logp), 1)
        # lambda 0 is the plain argmax
        self.assertEqual(rt.teacher_index({0: 20.0, 1: 20.5}, 3, 0, screen="card", lam=0.0, logp=logp), 1)

    def test_unplayed_options_are_never_taken(self):
        logp = [math.log(0.05), math.log(0.9), math.log(0.05)]
        # option 1 is the prior's favourite but was not played
        self.assertEqual(rt.teacher_index({0: 20.0, 2: 21.0}, 3, 0, screen="card", lam=0.5, logp=logp), 2)

    def test_the_guard_holds_against_a_prior_given_zero(self):
        logp = [math.log(0.99), math.log(0.005), math.log(0.005)]
        # option 1 is under the guard's 0.02: taken only when its value beats the blend's pick by more than a floor
        self.assertEqual(rt.teacher_index({0: 20.0, 1: 20.9}, 3, 0, screen="card", lam=0.5, logp=logp), 0)
        self.assertEqual(rt.teacher_index({0: 20.0, 1: 21.2}, 3, 0, screen="card", lam=0.5, logp=logp), 1)

    def test_rest_site_two_stages(self):
        # heal (0), two smith targets (1, 2): the kinds' marginal first, then the target
        logits = [math.log(0.45), math.log(0.30), math.log(0.25)]
        groups = [0, 1, 1]
        logp = [math.log(0.45), math.log(0.30), math.log(0.25)]
        self.assertEqual(rt.teacher_index({0: 20.0, 1: 20.0, 2: 20.0}, 3, 0, screen="rest", lam=0.5, logp=logp, logits=logits, groups=groups), 1)
        self.assertEqual(rt.teacher_index({0: 22.0, 1: 20.0, 2: 20.0}, 3, 1, screen="rest", lam=0.5, logp=logp, logits=logits, groups=groups), 0)

    def test_prior_alone_keeps_the_players_pick(self):
        self.assertEqual(rt.teacher_index({0: 20.0, 1: 30.0}, 2, 0, screen="card", lam=math.inf, logp=[0.0, -5.0]), 0)


class GuardTests(unittest.TestCase):
    def test_the_guard_takes_the_teacher_only_past_z_standard_errors(self):
        self.assertEqual(rt.guarded(2, 0, 3.0, 1.0), 2)
        self.assertEqual(rt.guarded(2, 0, 1.9, 1.0), 0)
        self.assertEqual(rt.guarded(0, 0, 9.0, 0.1), 0)
        self.assertEqual(rt.guarded(2, 0, 5.0, math.inf), 0)

    def test_agreeing_draws_do_not_make_a_tiny_error(self):
        # four identical differences: the own variance is 0, the prior's keeps the error near sd / sqrt(K)
        se = rt.guard_se([3.0, 3.0, 3.0, 3.0], "card")
        self.assertAlmostEqual(se, math.sqrt((4 * 3.41**2) / 7 / 4), places=6)
        self.assertGreater(se, 1.0)
        # a noisy decision's own variance counts
        self.assertGreater(rt.guard_se([10.0, -10.0, 10.0, -10.0], "card"), se)
        # an unknown kind takes the default prior
        self.assertAlmostEqual(rt.guard_se([0.0], None), rt.PAIRED_SD_DEFAULT, places=6)


class StatTests(unittest.TestCase):
    def test_split_halves(self):
        a, b = rt.split_halves({0: [1.0, 3.0, 5.0, 7.0], 1: [2.0, 2.0, 2.0, 2.0]})
        self.assertEqual(a, {0: 2.0, 1: 2.0})
        self.assertEqual(b, {0: 6.0, 1: 2.0})
        self.assertIsNone(rt.split_halves({0: [1.0], 1: [2.0]}))

    def test_paired_se(self):
        self.assertAlmostEqual(rt.paired_se([3.0, 5.0], [1.0, 1.0]), 1.0)
        self.assertIsNone(rt.paired_se([1.0], [2.0]))

    def test_noise_and_spread_on_pure_noise_and_on_pure_signal(self):
        import random

        rnd = random.Random(7)
        noise = [[[rnd.gauss(0, 2) for _ in range(4)] for _ in range(3)] for _ in range(800)]
        r = rt.noise_and_spread(noise)
        self.assertAlmostEqual(r["noise_sd"], 2 * math.sqrt(2 / 3), delta=0.15)  # a centred draw over three options
        self.assertLess(r["spread_sd"], 0.25)
        signal = [[[float(o)] * 4 for o in range(3)] for _ in range(50)]
        r = rt.noise_and_spread(signal)
        self.assertEqual(r["noise_sd"], 0.0)
        self.assertAlmostEqual(r["spread_sd"], math.sqrt(2 / 3), delta=0.01)  # the sd of the centred values -1, 0, 1
        self.assertEqual(r["w"], 1.0)
        self.assertEqual(rt.noise_and_spread([])["decisions"], 0)


class LambdaTests(unittest.TestCase):
    def test_lambda_recovers_the_floors_a_log_probability_is_worth(self):
        import random

        rnd = random.Random(3)
        decisions = []
        for _ in range(400):
            lp = [math.log(x) for x in (0.6, 0.3, 0.1)]
            v = [rnd.gauss(20, 1) for _ in range(3)]
            q = [vi + 0.8 * li + rnd.gauss(0, 1.0) for vi, li in zip(v, lp)]
            decisions.append(list(zip(q, v, lp)))
        r = rt.lambda_fit(decisions)
        self.assertEqual(r["decisions"], 400)
        self.assertAlmostEqual(r["lambda"], 0.8, delta=3 * r["se"])
        self.assertLess(r["se"], 0.1)

    def test_no_prior_signal_reads_zero_and_degenerate_input_reads_none(self):
        decisions = [[(20.0 + d % 3, 20.0 + d % 3, math.log(0.5)), (19.0, 19.0, math.log(0.5))] for d in range(10)]
        self.assertIsNone(rt.lambda_fit(decisions)["lambda"])  # no spread in log p
        self.assertIsNone(rt.lambda_fit([])["lambda"])
        flat = [[(20.0, 20.0, math.log(0.7)), (19.0, 19.0, math.log(0.3))] for _ in range(10)]
        self.assertEqual(rt.lambda_fit(flat)["lambda"], 0.0)


def _label(i: int, played_g: dict[int, list[tuple[float, str, int, float | None]]], kind: str = "card", act: int = 0) -> dict:
    conts = []
    for o, draws in played_g.items():
        for k, (g, end, floor, p) in enumerate(draws):
            conts.append({"o": o, "k": k, "G": g, "end": end, "floor": floor,
                          "door": None if p is None else {"floor": 17, "judge": {"p_death": p}}})
    return {"id": f"R{i}", "kind": kind, "act": act, "k": len(next(iter(played_g.values()))), "played": sorted(played_g), "taken": 0,
            "continuations": conts}


class RelabelTests(unittest.TestCase):
    def test_the_door_leaf_replaces_the_played_boss(self):
        # option 0 always dies at the boss though the judge said 0.4; option 1 always survives though it said 0.6
        recs = []
        for i in range(30):
            recs.append(_label(i, {0: [(17.0, "death", 17, 0.4)] * 4, 1: [(30.0, "next_act", 18, 0.6)] * 4}))
        res = rt.relabel(recs, beyond={0: 30.0})
        self.assertEqual(res["labels"], 30)
        self.assertGreater(res["played"]["all"]["spread_sd"], res["door"]["all"]["spread_sd"])
        self.assertEqual(res["door_judge"]["n"], 240)
        self.assertAlmostEqual(res["door_judge"]["boss_death_rate"], 0.5)
        self.assertEqual(res["played"]["all"]["split_half_agree"], 1.0)

    def test_a_death_before_the_door_keeps_its_floor(self):
        recs = [_label(i, {0: [(9.0, "death", 9, None)] * 2, 1: [(17.0, "death", 17, 1.0)] * 2}) for i in range(25)]
        res = rt.relabel(recs, beyond={0: 30.0})
        self.assertAlmostEqual(res["door"]["all"]["spread_sd"], 4.0, delta=0.05)  # (17 - 9) / 2 either side of the mean

    def test_beyond_defaults_to_the_survivors_head(self):
        recs = [_label(i, {0: [(26.0, "next_act", 18, 0.2)] * 2, 1: [(17.0, "death", 17, 0.9)] * 2}) for i in range(25)]
        self.assertEqual(rt.relabel(recs)["beyond"], {"0": 26.0})


class ReportTests(unittest.TestCase):
    def test_report_on_written_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            runs = []
            for s, (pf, tf) in enumerate(((20, 22), (17, 17), (30, 33))):
                for arm, fl, cpu in (("player", pf, 30.0), ("teacher", tf, 900.0)):
                    runs.append({"_task": ["rt", arm, f"S{s}", s], "arm": arm, "seed": f"S{s}", "ix": s, "floor": fl, "cpu": cpu,
                                 "wall": cpu, "teacher_cpu": cpu - 30.0, "teacher_screens": 3, "screens": {"card/a1": 3},
                                 "bosses": [{"act": 0, "floor": 17, "hp": 40, "max_hp": 80, "won": fl > 17}]})
            runs.append({"_task": ["rt", "teacher", "S9", 9], "arm": "teacher", "seed": "S9", "ix": 9, "capped": True})
            (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs), encoding="utf-8")
            dd = out / "decisions.d"
            dd.mkdir()
            for s in range(3):
                recs = [{"seed": f"S{s}", "kind": "card", "act": 0, "played": [0, 1], "player": 0, "teacher": 1 if s == 0 else 0,
                         "teacher_raw": 1, "g": {"0": [20.0, 21.0], "1": [26.0, 27.0]}, "gain": 6.0, "gain_se": 0.5,
                         "half_picks": [1, 1], "half_played": [1, 0], "cpu": 100.0}]
                (dd / f"S{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
            res = rt.report(out)
            self.assertEqual(res["runs"], {"player": 3, "teacher": 3})
            self.assertEqual(res["teacher"]["cpu_mean"], 900.0)
            self.assertEqual(res["paired"]["floor"]["n"], 3)
            self.assertAlmostEqual(res["paired"]["floor"]["mean"], (2 + 0 + 3) / 3, places=3)
            cell = res["screens"]["card/a1"]
            self.assertEqual(cell["screens"], 3)
            self.assertAlmostEqual(cell["changed"], 1 / 3, places=3)
            self.assertAlmostEqual(cell["changed_raw"], 1.0, places=3)
            self.assertEqual(cell["split_half_agree"], 1.0)
            self.assertEqual(cell["split_half_agree_played"], 0.0)
            self.assertEqual(cell["chance"], 0.5)
            self.assertIn("act1", res["screens"])


class SeedsAndGuardTests(unittest.TestCase):
    def test_resample_seeds_do_not_depend_on_the_option(self):
        a = rt.resample_seed("RUN", 7, 2)
        self.assertEqual(a, rt.resample_seed("RUN", 7, 2))
        self.assertNotEqual(a, rt.resample_seed("RUN", 7, 3))
        self.assertNotEqual(a, rt.resample_seed("RUN", 8, 2))
        self.assertTrue(0 <= a < 2**31 - 1)
        self.assertNotEqual(rt.stream_key("RUN", 7, 2, 0), rt.stream_key("RUN", 7, 2, 1))

    def test_seeds_from_a_lab_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "runs.jsonl"
            f.write_text("".join(json.dumps({"seed": s, "ix": i}) + "\n" for i, s in ((2, "C"), (0, "A"), (1, "B"))) + "garbage\n", encoding="utf-8")
            self.assertEqual(rt.seeds_from(f, 2), [("A", 0), ("B", 1)])

    def test_the_guard_reads_other_lines_but_not_the_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp)
            (ref / "backlog").mkdir()
            (ref / "LAB" / "nightly").mkdir(parents=True)
            now = rt.utc().strftime("%Y-%m-%dT%H:%M:%SZ")
            with mock.patch.multiple(rt, QUIET=ref / "QUIET.lock", MAIN_LOCK=ref / "backlog" / "main.lock", ACTIVE=ref / "LAB" / "nightly" / "active"):
                self.assertIsNone(rt.blocked())
                (ref / "QUIET.lock").write_text(f"gym-loop {now} pid=1\n", encoding="utf-8")
                self.assertIsNone(rt.blocked())
                rt.hold(True)
                self.assertIn(rt.TAG, (ref / "QUIET.lock").read_text(encoding="utf-8"))
                self.assertIsNone(rt.blocked())
                (ref / "QUIET.lock").write_text((ref / "QUIET.lock").read_text(encoding="utf-8") + f"judge-corpus {now} pid=2\n", encoding="utf-8")
                self.assertEqual(rt.blocked(), "quiet:judge-corpus")
                rt.hold(False)
                lines = (ref / "QUIET.lock").read_text(encoding="utf-8").splitlines()
                self.assertEqual([x.split()[0] for x in lines], ["gym-loop", "judge-corpus"])
                (ref / "backlog" / "main.lock").write_text("land 1 pid=3\n", encoding="utf-8")
                self.assertEqual(rt.blocked(), "main.lock")
                self.assertEqual(rt.must_stop(), "quiet:judge-corpus")  # another line stops a running job even during a landing
                (ref / "QUIET.lock").write_text(f"gym-loop {now} pid=1\n", encoding="utf-8")
                self.assertEqual(rt.blocked(), "main.lock")  # a landing keeps a job from starting
                self.assertIsNone(rt.must_stop())  # but lets a running one finish
                (ref / "LAB" / "nightly" / "active").write_text("", encoding="utf-8")
                self.assertEqual(rt.blocked(), "nightly-active")
                self.assertEqual(rt.must_stop(), "nightly-active")
                self.assertEqual(rt.night_only(), "nightly-active")
                (ref / "LAB" / "nightly" / "active").unlink()
                (ref / "QUIET.lock").write_text(f"judge-corpus {now} pid=2\n", encoding="utf-8")
                self.assertIsNone(rt.night_only())  # a shared job stops for nobody's line

    def test_eligible(self):
        d = SimpleNamespace(options=[SimpleNamespace(kind="card.take"), SimpleNamespace(kind="card.skip")])
        one = SimpleNamespace(options=[SimpleNamespace(kind="card.take"), SimpleNamespace(kind="unknown")])
        self.assertTrue(rt.eligible("card", SimpleNamespace(floor=5), d, rt.TEACHER_KINDS))
        self.assertFalse(rt.eligible("card", SimpleNamespace(floor=33), d, rt.TEACHER_KINDS))
        self.assertFalse(rt.eligible("card", SimpleNamespace(floor=5), one, rt.TEACHER_KINDS))
        self.assertFalse(rt.eligible(None, SimpleNamespace(floor=5), d, rt.TEACHER_KINDS))
        self.assertFalse(rt.eligible("shop", SimpleNamespace(floor=5), d, ("card",)))

    def test_shortlist_keeps_the_pick_and_the_unchanged(self):
        opts = [SimpleNamespace(kind=k) for k in ("shop.buy_card", "shop.buy_card", "shop.buy_relic", "shop.leave", "unknown", "shop.buy_potion")]
        scores = [1.0, 5.0, 3.0, 0.0, None, 4.0]
        self.assertEqual(rt.shortlist(opts, scores, 3, keep=0), [0, 1, 3])
        self.assertEqual(rt.shortlist(opts, scores, 4, keep=0), [0, 1, 3, 5])
        self.assertEqual(rt.shortlist(opts[:3], scores[:3], 4, keep=0), [0, 1, 2])


class CommandTests(unittest.TestCase):
    def test_parser(self):
        self.assertTrue(rt.build_parser().parse_args(["play", "--out", "X", "--share"]).share)
        a = rt.build_parser().parse_args(["play", "--out", "X"])
        self.assertFalse(a.share)
        self.assertEqual((a.k, a.max_options, a.procs, a.runs, a.arm), (rt.DEFAULT_K, rt.DEFAULT_MAX_OPTIONS, 4, 50, "player,teacher"))
        self.assertEqual(rt.build_parser().parse_args(["relabel", "--labels", "L"]).cmd, "relabel")

    def test_no_quoted_job_word_of_the_bound(self):
        text = (ROOT / "rl" / "run_teacher.py").read_text(encoding="utf-8")
        self.assertNotIn('"' + "hindsight" + '"', text)
        self.assertNotIn("'" + "hindsight" + "'", text)


NATIVE = (ROOT / "out" / "Sts2Emulator.dll").exists()


@unittest.skipUnless(NATIVE, "the native library of this tree is not built")
class NativeTests(unittest.TestCase):
    """On the real engine with a stand-in player (the rules run policy, fights resolved at four simulations): the mechanism, not
    the agent."""

    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
        os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")
        import numpy as np
        from run_train import engine, selfplay

        import hindsight

        cls.np, cls.engine, cls.hindsight = np, engine, hindsight

        class Policy:
            def __init__(self):
                self.rules = selfplay.RulesPlayer()

            def pick(self, view, d, rng):
                return self.rules.pick(view, d, np.random.default_rng(0))

        def play_fight(s, rng, deadline):
            from sts2_gym import native

            made = 0
            while s.phase == engine.rc.PHASE_COMBAT:
                native.run_debug_resolve_combat(s.handle, s.env._run_obs_buf)
                made += 1
                if made > 3:
                    break
            return made

        cls.player = hindsight.Player(policy=Policy(), play_fight=play_fight, head=lambda v, d: None,
                                      judge=lambda handle, max_hp: {"p_death": 0.5})

    def _at_card_reward(self):
        s = self.engine.RunSession("RTEACHER1", 10, self.engine.CombatBudget(4, 4))
        pol = self.player.policy
        for _ in range(400):
            d = s.decision()
            if d is not None and d.type == "card_reward" and len(d.options) >= 2:
                return s, d
            if d is None:
                self.player.play_fight(s, None, None)
                continue
            s.apply(d.options[pol.pick(s.view(), d, None).index])
        s.close()
        self.skipTest("no card reward reached")

    def test_a_draw_leaves_the_base_alone_and_reaches_an_end(self):
        s, d = self._at_card_reward()
        try:
            before = (s.info["floor"], s.info["hp"], s.info["gold"], s.info["deck_size"])
            row = rt.continuation(s, d.options[0], self.player, seed="RTEACHER1", ordinal=3, k=0)
            self.assertEqual(before, (s.info["floor"], s.info["hp"], s.info["gold"], s.info["deck_size"]))
            self.assertIn(row["end"], ("door", "death", "win", "timeout", "step_cap"))
            if row["end"] == "door":
                self.assertEqual(row["p_death"], 0.5)
                self.assertAlmostEqual(row["G"], rt.door_return("door", row["floor"], 0.5), places=3)
            again = rt.continuation(s, d.options[0], self.player, seed="RTEACHER1", ordinal=3, k=0)
            self.assertEqual((row["end"], row["floor"], row["G"]), (again["end"], again["floor"], again["G"]))
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
