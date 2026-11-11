"""The run-level teacher's measurement (rl/run_teacher.py; docs/rl/RUN_SELF_PLAY.md).

The pure parts need no native library: the door leaf, the teacher's pick (the player's blend with the played values in place of V,
the guard, the rest site's two stages), the split halves, the noise and spread, the door-leaf rescoring of stored labels, the report,
the seeds and the machine guard. The native parts (the continuations to the door and the run loop) are checked on the real engine
when its library is built: a run with the player deciding is the run `gen0.play_emu_run` plays, and a teacher's draw is the same
clone for every option.
"""

import json
import math
import shutil
import sys
import tempfile
import time
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


class CalibrationTests(unittest.TestCase):
    """The door's recalibration of the judge's P(death) on the player's own door outcomes, on synthetic doors."""

    @staticmethod
    def doors(n: int, a: float, b: float, act: int = 0, seed: int = 5):
        """(act, the judge's P(death), died) of n doors: the judge says p, the truth is sigmoid(a + b logit p)."""
        import random

        rnd = random.Random(seed)
        out = []
        for _ in range(n):
            p = min(max(rnd.betavariate(2, 2), 0.02), 0.98)
            truth = rt._sigmoid(a + b * rt._logit(p))
            out.append((act, p, int(rnd.random() < truth)))
        return out

    def test_a_judge_that_is_too_sure_is_pulled_back_and_the_fit_recovers_the_planted_map(self):
        pairs = self.doors(4000, a=-0.4, b=0.5)
        cal = rt.fit_door_calibration(pairs)
        self.assertEqual([0], list(cal))
        self.assertAlmostEqual(cal[0]["a"], -0.4, delta=0.15)
        self.assertAlmostEqual(cal[0]["b"], 0.5, delta=0.1)
        self.assertEqual(4000, cal[0]["n"])
        self.assertLess(cal[0]["brier_after"], cal[0]["brier_before"])
        self.assertLess(abs(cal[0]["p_mean_after"] - cal[0]["death_rate"]), 0.02)  # calibrated in the large
        m = rt.calibration_map(cal)
        self.assertLess(rt.recalibrate(0.95, 0, m), 0.95)  # a sure death is less sure
        self.assertGreater(rt.recalibrate(0.05, 0, m), 0.05)

    def test_a_judge_that_is_right_keeps_its_numbers(self):
        cal = rt.fit_door_calibration(self.doors(4000, a=0.0, b=1.0))
        self.assertAlmostEqual(cal[0]["a"], 0.0, delta=0.12)
        self.assertAlmostEqual(cal[0]["b"], 1.0, delta=0.12)
        m = rt.calibration_map(cal)
        self.assertAlmostEqual(rt.recalibrate(0.3, 0, m), 0.3, delta=0.03)

    def test_acts_are_fitted_apart_and_an_act_with_few_doors_is_left_alone(self):
        pairs = self.doors(500, a=-1.0, b=1.0, act=0) + self.doors(500, a=1.0, b=1.0, act=1, seed=9) + self.doors(rt.CAL_MIN_DOORS - 1, a=2.0, b=1.0, act=2)
        cal = rt.fit_door_calibration(pairs)
        self.assertEqual([0, 1], sorted(cal))
        self.assertLess(cal[0]["a"], 0.0)  # Act 1's bosses kill less than the judge says
        self.assertGreater(cal[1]["a"], 0.0)  # Act 2's more
        m = rt.calibration_map(cal)
        self.assertEqual(0.4, rt.recalibrate(0.4, 2, m))  # no map for Act 3: the judge's own number
        self.assertEqual(0.4, rt.recalibrate(0.4, 0, None))

    def test_the_ridge_holds_a_thin_sample_near_the_judge(self):
        thin = rt.fit_door_calibration(self.doors(rt.CAL_MIN_DOORS, a=-2.0, b=0.3, seed=2))
        big = rt.fit_door_calibration(self.doors(5000, a=-2.0, b=0.3, seed=2))
        self.assertGreater(thin[0]["b"], big[0]["b"])  # nearer the identity slope

    def test_degenerate_input_does_not_blow_up(self):
        self.assertEqual({}, rt.fit_door_calibration([]))
        all_dead = [(0, 0.5, 1)] * 200
        cal = rt.fit_door_calibration(all_dead)
        self.assertTrue(0.5 < rt.recalibrate(0.5, 0, rt.calibration_map(cal)) < 1.0)
        self.assertEqual({}, rt.fit_door_calibration([(0, 1.5, 1)] * 100 + [(0, 0.5, 2)] * 100))  # outside [0, 1], not a 0/1 outcome: dropped

    def test_the_leaf_uses_the_recalibrated_death_risk_and_the_record_keeps_the_judges(self):
        beyond = {0: 30.0, 1: 42.0}
        cal = {0: (0.0, 0.5)}  # halves the log-odds
        raw = rt.door_return("door", 17, 0.9, beyond)
        mapped = rt.door_return("door", 17, 0.9, beyond, cal=cal)
        self.assertGreater(mapped, raw)  # less sure of dying: worth more
        p = rt.recalibrate(0.9, 0, cal)
        self.assertAlmostEqual(mapped, p * 17 + (1 - p) * 30.0)
        self.assertEqual(rt.door_return("door", 33, 0.9, beyond, cal=cal), rt.door_return("door", 33, 0.9, beyond))  # no map for Act 2
        self.assertEqual(12.0, rt.door_return("death", 12, None, cal=cal))  # a death on the way is its floor

    def test_the_players_own_doors_are_read_from_the_stores_run_rows(self):
        runs = [{"bosses": [{"act": 0, "floor": 17, "p_death": 0.3, "won": True}, {"act": 1, "floor": 33, "p_death": 0.8, "won": False}]},
                {"bosses": [{"act": 0, "floor": 17, "won": True}]},  # no judge there: not a pair
                {"capped": True, "bosses": [{"act": 0, "floor": 17, "p_death": 0.5, "won": False}]},
                {}]
        self.assertEqual([(0, 0.3, 0), (1, 0.8, 1)], rt.door_pairs(runs))

    def test_stored_labels_give_the_same_pairs_for_a_first_calibration(self):
        label = {"id": "x", "act": 0, "continuations": [
            {"o": 0, "k": 0, "end": "death", "floor": 17, "door": {"floor": 17, "judge": {"p_death": 0.7}}},
            {"o": 0, "k": 1, "end": "next_act", "floor": 20, "door": {"floor": 17, "judge": {"p_death": 0.2}}},
            {"o": 1, "k": 0, "end": "death", "floor": 9, "door": None}]}
        self.assertEqual([(0, 0.7, 1), (0, 0.2, 0)], rt.label_door_pairs([label, {"refused": "x", "act": 0, "continuations": [{}]}]))


def fake_decision(kinds=("card.take", "card.take", "card.skip"), typ="card_reward"):
    return SimpleNamespace(type=typ, view=None, options=[SimpleNamespace(kind=k, scorable=False, i=i) for i, k in enumerate(kinds)])


VIEW = SimpleNamespace(act_index=0, floor=5, hp=60, max_hp=80)


class AdaptiveKTests(unittest.TestCase):
    def test_the_top_two_and_their_gap(self):
        a, b, gap, mean_diff, diffs = rt.top_two({0: [20.0, 22.0], 1: [24.0, 26.0], 2: [10.0, 10.0]})
        self.assertEqual((a, b), (1, 0))
        self.assertAlmostEqual(gap, 4.0)
        self.assertEqual(diffs, [4.0, 4.0])
        # the prior's share ranks the options as the pick does
        a, b, gap, *_ = rt.top_two({0: [20.0, 20.0], 1: [20.5, 20.5]}, {0: 1.0, 1: 0.0})
        self.assertEqual((a, b), (0, 1))
        self.assertAlmostEqual(gap, 0.5)
        self.assertIsNone(rt.top_two({0: [1.0, 2.0]}))

    def test_the_stopping_arithmetic_two_standard_errors_of_the_pooled_error(self):
        # two draws, kind card: the error is the guard's, sqrt(((K-1) own + NU0 prior^2) / (K-1+NU0) / K)
        se = rt.guard_se([5.0, 5.0], "card")
        self.assertAlmostEqual(se, math.sqrt((4 * 3.41**2) / 5 / 2), places=6)
        edge = 2.0 * se
        far = {0: [20.0, 20.0], 1: [20.0 + edge + 0.05] * 2}
        near = {0: [20.0, 20.0], 1: [20.0 + edge - 0.05] * 2}
        self.assertFalse(rt.wants_more_draws(far, "card"))
        self.assertTrue(rt.wants_more_draws(near, "card"))
        self.assertTrue(rt.wants_more_draws({0: [20.0, 20.0], 1: [20.0, 20.0]}, "card"))  # a tie is within the error
        self.assertFalse(rt.wants_more_draws({0: [20.0, 20.0]}, "card"))  # one option valued: nothing to separate
        # a kind with a wider prior needs a wider gap
        self.assertTrue(rt.wants_more_draws({0: [20.0, 20.0], 1: [20.0 + edge + 0.05] * 2}, "route"))

    def test_noisy_draws_widen_the_error_and_keep_looking(self):
        quiet = {0: [20.0, 20.0], 1: [26.0, 26.0]}
        noisy = {0: [10.0, 30.0], 1: [26.0, 26.0]}
        self.assertFalse(rt.wants_more_draws(quiet, "card"))
        self.assertTrue(rt.wants_more_draws(noisy, "card"))  # the same mean gap of 6 inside a large own variance

    def run_decide(self, table, *, k=2, k_max=4, player=0):
        calls = []

        def cont(base, option, pl, *, seed, ordinal, k, cal=None):
            calls.append((option.i, seed, ordinal, k))
            return {"k": k, "G": table[option.i][k], "end": "door", "floor": 17, "p_death": 0.3, "door_hp": 50, "fights": 1, "cpu": 0.0, "seconds": 0.0}

        rd = {"lam": 0.0, "screen": None, "logp": None, "logits": None, "groups": None, "scores": [None] * 3, "v": [None] * 3,
              "rows": __import__("numpy").arange(3 * 4, dtype="float32").reshape(3, 4)}
        rows_out: list = []
        with mock.patch.object(rt, "continuation", cont), mock.patch.object(rt, "screen_inputs", return_value=rd):
            rec = rt.decide(None, fake_decision(), VIEW, SimpleNamespace(index=player), SimpleNamespace(policy=None), seed="RUN", ordinal=7, k=k,
                            max_options=4, k_max=k_max, rows_out=rows_out)
        return rec, calls, rows_out

    def test_a_clear_screen_stops_at_the_first_two_draws(self):
        table = {0: [20.0, 20.0, 20.0, 20.0], 1: [35.0, 35.0, 35.0, 35.0], 2: [10.0, 10.0, 10.0, 10.0]}
        rec, calls, rows_out = self.run_decide(table)
        self.assertEqual(2, rec["k"])
        self.assertEqual(6, len(calls))  # two draws of three options
        self.assertEqual({0, 1}, {c[3] for c in calls})
        self.assertEqual(1, rec["teacher_raw"])
        self.assertEqual(1, rec["teacher"])  # a gain of 15 floors over the player's pick is over two errors (2 x 2.16): taken
        self.assertEqual([[20.0, 20.0], [35.0, 35.0], [10.0, 10.0]], [rec["g"][str(o)] for o in (0, 1, 2)])

    def test_a_close_screen_takes_two_more_draws_and_the_draws_are_the_fixed_k_draws(self):
        table = {0: [20.0, 21.0, 19.0, 22.0], 1: [21.0, 20.0, 23.0, 21.0], 2: [5.0, 5.0, 5.0, 5.0]}
        rec, calls, _ = self.run_decide(table)
        self.assertEqual(4, rec["k"])
        self.assertEqual(12, len(calls))
        fixed, fixed_calls, _ = self.run_decide(table, k=4, k_max=None)
        self.assertEqual(4, fixed["k"])
        self.assertEqual(rec["g"], fixed["g"])  # the adaptive draws are the first draws of a fixed K 4
        self.assertEqual(sorted(calls), sorted(fixed_calls))

    def test_the_seeds_never_depend_on_the_option(self):
        table = {0: [20.0, 21.0, 19.0, 22.0], 1: [21.0, 20.0, 23.0, 21.0], 2: [20.5, 20.0, 21.0, 21.0]}
        _, calls, _ = self.run_decide(table)
        by_draw: dict[int, set] = {}
        for option, seed, ordinal, kk in calls:
            by_draw.setdefault(kk, set()).add((seed, ordinal))
        self.assertEqual({0, 1, 2, 3}, set(by_draw))
        for kk, keys in by_draw.items():
            self.assertEqual({("RUN", 7)}, keys, kk)  # one (run, decision) per draw, whatever the option
        # and the resample seed and streams of a draw are functions of (run, decision, draw) only
        self.assertEqual(rt.resample_seed("RUN", 7, 2), rt.resample_seed("RUN", 7, 2))
        import inspect

        self.assertNotIn("option", inspect.signature(rt.resample_seed).parameters)
        self.assertNotIn("option", inspect.signature(rt.stream_key).parameters)

    def test_fixed_k_is_the_measurements_k_and_the_rows_come_back(self):
        table = {0: [20.0, 20.0, 20.0, 20.0], 1: [20.0, 20.0, 20.0, 20.0], 2: [20.0, 20.0, 20.0, 20.0]}
        rec, calls, rows_out = self.run_decide(table, k=2, k_max=None)
        self.assertEqual(2, rec["k"])  # no adaptive look without a k_max
        self.assertEqual(1, len(rows_out))
        self.assertEqual((3, 4), rows_out[0].shape)
        rec, calls, _ = self.run_decide(table, k=2, k_max=2)
        self.assertEqual(2, rec["k"])
        self.assertEqual(6, len(calls))

    def test_a_draw_with_no_return_ends_the_looking_and_the_decision_falls_back(self):
        table = {0: [20.0, 20.0, 20.0, 20.0], 1: [None, None, None, None], 2: [20.0, 20.0, 20.0, 20.0]}
        rec, calls, _ = self.run_decide(table)
        self.assertEqual("no_return", rec["refused"])
        self.assertEqual(2, rec["k"])
        self.assertEqual(rec["player"], rec["teacher"])


class ScreenRowsTests(unittest.TestCase):
    def test_the_rows_of_a_screen_are_option_rows_as_built_at_play(self):
        import numpy as np

        want = np.random.default_rng(1).normal(size=(3, rt.ROWS_WIDTH)).astype("float64")
        policy = SimpleNamespace(nets=object(), lam_at=lambda floor: 0.0)
        with mock.patch("human_prior_infer.option_rows", return_value=want) as orows:
            rd = rt.screen_inputs(policy, VIEW, fake_decision(), rows=True)
            orows.assert_called_once()
        self.assertEqual(np.float32, rd["rows"].dtype)
        np.testing.assert_array_equal(rd["rows"], want.astype("float32"))
        self.assertIsNone(rd["logp"])  # the player does not blend the prior at lambda 0
        with mock.patch("human_prior_infer.option_rows", return_value=want) as orows:
            self.assertIsNone(rt.screen_inputs(policy, VIEW, fake_decision())["rows"])  # not asked for: not built
            orows.assert_not_called()


def synthetic_record(seed: str, ordinal: int, n: int = 3, kind: str = "card", k: int = 2, **over) -> dict:
    played = list(range(min(n, 3)))
    g = {str(o): [20.0 + o, 21.0 + o][:k] for o in played}
    rec = {"v": rt.VERSION, "seed": seed, "ordinal": ordinal, "type": "card_reward", "kind": kind, "act": 0, "floor": 3 + ordinal, "hp": 70, "max_hp": 80,
           "n_options": n, "played": played, "player": 0, "k": k, "screen": None, "lam": 0.0, "v_values": [None] * n, "logp": None, "g": g,
           "ends": {"door": 2 * len(played)}, "p_death": {str(o): [0.3] * k for o in played}, "door_hp": {str(o): [50] * k for o in played},
           "options": ["card.take"] * (n - 1) + ["card.skip"], "teacher": 0, "teacher_raw": 1, "q": {str(o): 20.5 + o for o in played},
           "gain": 1.0, "gain_se": 2.0, "gain_se_own": 1.9, "half_picks": [1, 1], "half_played": [0, 0], "cpu": 12.5, "seconds": 13.0}
    rec.update(over)
    return rec


def rows_for(n: int, base: float):
    import numpy as np

    return (np.arange(n * rt.ROWS_WIDTH, dtype="float32").reshape(n, rt.ROWS_WIDTH) / 1e4 + base).astype("float32")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def make_run(self, seed, n_screens=3, widths=(3, 4, 2)):
        rf = rt.RunFiles(self.tmp, seed)
        rf.load()
        rows = []
        for i in range(n_screens):
            n = widths[i % len(widths)]
            r = rows_for(n, i + sum(map(ord, seed)) % 7)
            rows.append(r)
            rf.append(synthetic_record(seed, i * 2, n=n), r)
        return rf, rows

    def test_records_and_rows_round_trip_and_row0_counts_rows(self):
        rf, rows = self.make_run("AAA")
        again = rt.RunFiles(self.tmp, "AAA")
        got = again.load()
        self.assertEqual([0, 2, 4], sorted(got))
        self.assertEqual([0, 3, 7], [got[o]["row0"] for o in sorted(got)])
        self.assertEqual(9, again.n_rows)

    def test_a_cut_row_file_or_a_cut_record_line_loses_only_the_screens_it_cuts(self):
        rf, rows = self.make_run("BBB")
        size = rf.rows.stat().st_size
        with open(rf.rows, "r+b") as fh:
            fh.truncate(size - 10)  # the last screen's rows are short
        again = rt.RunFiles(self.tmp, "BBB")
        self.assertEqual([0, 2], sorted(again.load()))
        self.assertEqual(7 * rt.ROWS_WIDTH * 4, rf.rows.stat().st_size)  # cut back to the consistent prefix
        with open(again.recs, "a", encoding="utf-8") as fh:
            fh.write('{"seed": "BBB", "ordinal": 9, "n_opt')  # a line cut by a kill
        self.assertEqual([0, 2], sorted(rt.RunFiles(self.tmp, "BBB").load()))
        # a screen that is appended after the cut takes the cut screen's place
        again = rt.RunFiles(self.tmp, "BBB")
        again.load()
        again.append(synthetic_record("BBB", 4, n=2), rows_for(2, 9.0))
        self.assertEqual([0, 2, 4], sorted(rt.RunFiles(self.tmp, "BBB").load()))

    def test_a_replay_that_left_the_stored_screens_drops_them_from_there(self):
        rf, _ = self.make_run("CCC")
        rf = rt.RunFiles(self.tmp, "CCC")
        rf.load()
        rf.drop_from(2)
        self.assertEqual([0], sorted(rt.RunFiles(self.tmp, "CCC").load()))
        self.assertEqual(3 * rt.ROWS_WIDTH * 4, rf.rows.stat().st_size)

    def test_rows_must_match_the_options(self):
        rf = rt.RunFiles(self.tmp, "DDD")
        rf.load()
        with self.assertRaises(ValueError):
            rf.append(synthetic_record("DDD", 0, n=3), rows_for(2, 0.0))
        r = rf.append(synthetic_record("DDD", 0, n=3), None)  # no rows could be built: the record is kept with row0 None
        self.assertIsNone(r["row0"])

    def manifest(self, **over):
        base = rt.shard_manifest(job="J1", generation=13, identity={"agent": "gen13/x", "agent_sha256": "ab" * 32, "generation": 13, "recipe": "r",
                                                                  "agent_files": {}}, judge=None, lib=None, tree="abc", k=2, k_max=4, max_options=4,
                                 cal={}, sources={"new": 2})
        base.update(over)
        return base

    def test_the_shard_has_the_rows_of_every_decision_in_option_order_and_row0_points_at_them(self):
        import numpy as np

        _, rows_a = self.make_run("AAA")
        _, rows_b = self.make_run("BBB", widths=(2, 5, 3))
        doc = rt.build_shard(self.tmp, "J1", ["AAA", "BBB"], self.manifest())
        d = self.tmp / "shards" / "J1"
        self.assertEqual({"decisions.jsonl", "rows.npz", "manifest.json"}, {p.name for p in d.iterdir()})
        recs = [json.loads(x) for x in (d / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
        z = np.load(d / "rows.npz", allow_pickle=False)
        self.assertEqual((np.float32, np.int32, np.int16), (z["rows"].dtype, z["decision"].dtype, z["option"].dtype))
        self.assertEqual(doc["counts"]["decisions"], len(recs))
        self.assertEqual(doc["counts"]["rows"], len(z["rows"]))
        want = rows_a + rows_b
        for i, r in enumerate(recs):  # a decision's rows are the rows built at play, option by option
            n = int(r["n_options"])
            self.assertEqual([i] * n, z["decision"][r["row0"]:r["row0"] + n].tolist())
            self.assertEqual(list(range(n)), z["option"][r["row0"]:r["row0"] + n].tolist())
            np.testing.assert_array_equal(z["rows"][r["row0"]:r["row0"] + n], want[i])
        self.assertEqual(["AAA", "BBB"], [r["seed"] for r in recs if r["ordinal"] == 0])
        self.assertEqual(["AAA", "BBB"], doc["seeds"])

    def test_the_manifest_names_who_played_and_holds_no_absolute_path(self):
        self.make_run("AAA")
        doc = rt.build_shard(self.tmp, "J1", ["AAA"], self.manifest(judge="v0/judge.pt", judge_sha256="cd" * 32, engine_sha256="ef" * 32))
        for key in ("agent", "agent_sha256", "judge", "judge_sha256", "engine", "engine_sha256", "k", "m", "beyond", "created_utc", "runs", "k_max", "recal",
                    "guard", "leaf_b", "counts", "version"):
            self.assertIn(key, doc)
        self.assertEqual(rt.BEYOND, doc["beyond"])
        self.assertEqual(rt.Z, doc["guard"]["z"])
        self.assertEqual(rt.PAIRED_SD_PRIOR, doc["guard"]["paired_sd_prior"])
        text = (self.tmp / "shards" / "J1" / "manifest.json").read_text(encoding="utf-8")
        self.assertNotRegex(text, r"[A-Za-z]:[\\/]")
        self.assertEqual("v0/judge.pt", rt.clean_path("D:/spireline/referee/agent/judge/v0/judge.pt"))

    def test_a_shard_is_rebuilt_whole_and_a_finished_run_is_in_one_shard_only(self):
        self.make_run("AAA")
        self.make_run("BBB")
        rt.build_shard(self.tmp, "J1", ["AAA"], self.manifest())
        self.assertEqual({"AAA"}, rt.sharded_seeds(self.tmp))
        rt.build_shard(self.tmp, "J2", ["BBB"], self.manifest(job="J2"))
        self.assertEqual({"AAA", "BBB"}, rt.sharded_seeds(self.tmp))
        self.assertEqual({"AAA"}, rt.sharded_seeds(self.tmp, but="J2"))  # the shard being rebuilt does not count itself
        rt.build_shard(self.tmp, "J2", ["BBB"], self.manifest(job="J2"))  # again: replaced, not added to
        self.assertEqual(3, json.loads((self.tmp / "shards" / "J2" / "manifest.json").read_text(encoding="utf-8"))["counts"]["decisions"])
        self.assertFalse(list((self.tmp / "shards").glob("*.tmp")))

    def test_an_empty_shard_is_still_a_shard(self):
        doc = rt.build_shard(self.tmp, "J0", [], self.manifest())
        self.assertEqual(0, doc["counts"]["decisions"])
        import numpy as np

        z = np.load(self.tmp / "shards" / "J0" / "rows.npz")
        self.assertEqual((0, rt.ROWS_WIDTH), z["rows"].shape)

    def test_the_run_student_loader_reads_it(self):
        try:
            import run_student as rs
        except ImportError as e:  # pragma: no cover
            self.skipTest(f"run_student not importable here ({e})")
        import numpy as np

        self.make_run("AAA")
        self.make_run("BBB", widths=(2, 5, 3))
        rt.build_shard(self.tmp, "J1", ["AAA", "BBB"], self.manifest())
        sh = rs.read_shard(self.tmp / "shards" / "J1")
        self.assertEqual(6, len(sh["records"]))
        self.assertEqual(0, sh["tail_cut"])
        ts = rs.load_shards(self.tmp)
        self.assertEqual(6, len(ts))
        self.assertEqual(2, ts.stats["runs"])
        self.assertEqual({"card": 6}, ts.stats["kept"])
        self.assertEqual(2, int(ts.k[0]))
        self.assertEqual("J1", ts.manifests[0]["shard"])
        np.testing.assert_array_equal(ts.q[:3], np.array([20.5, 21.5, 22.5], dtype="float32"))  # the teacher's values of the first screen's options
        self.assertTrue(np.isnan(ts.q[int(ts.start[1]) + 3]))  # the second screen's fourth option was off the shortlist: it has none


class SourceTests(unittest.TestCase):
    def test_new_seeds_are_ten_lab_alphabet_characters_in_a_namespace_of_their_own(self):
        a, b = rt.new_seed(13, 0), rt.new_seed(13, 1)
        self.assertEqual(10, len(a))
        self.assertTrue(set(a) <= set(rt.SEED_ALPHABET))
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, rt.new_seed(14, 0))
        self.assertEqual(a, rt.new_seed(13, 0))
        self.assertEqual(rt.NEW_IX0 + 5, rt.new_ix(5))

    def test_new_tasks_skip_the_finished_and_replay_the_half_done_first(self):
        done = {rt.new_seed(13, 0), rt.new_seed(13, 2)}
        t = rt.new_tasks(13, done, 3)
        self.assertEqual([rt.new_seed(13, 1), rt.new_seed(13, 3), rt.new_seed(13, 4)], [x[1] for x in t])
        self.assertEqual({"rt-new"}, {x[0] for x in t})
        self.assertEqual([rt.new_ix(1), rt.new_ix(3), rt.new_ix(4)], [x[2] for x in t])

    def test_stored_tasks_take_the_dated_runs_in_a_fixed_shuffle_without_the_finished(self):
        rows = [{"seed": s, "ix": i, "_job": j, "_explore": (0.1, 0.02)} for i, (s, j) in enumerate(
            (("A", "20261105T1200-selfplay"), ("B", "20261107T0000-selfplay"), ("C", "20261108T2359-selfplay"), ("D", "20261109T0001-selfplay"),
             ("E", "20261104T2359-selfplay"), ("F", "20261106T0000-selfplay")))]
        got = rt.stored_tasks(rows, {"B"}, 10)
        self.assertEqual({"A", "C", "F"}, {t[1] for t in got})  # round 0's dates, B finished, D and E outside
        self.assertEqual({"rt-label"}, {t[0] for t in got})
        self.assertEqual(got, rt.stored_tasks(list(reversed(rows)), {"B"}, 10))  # the order does not depend on the input's
        self.assertEqual(1, len(rt.stored_tasks(rows, set(), 1)))
        self.assertIs(got[0][3], next(r for r in rows if r["seed"] == got[0][1]))

    def test_auto_is_round_0_first_then_new_play(self):
        self.assertEqual("stored", rt.choose_source("auto", 0, 6889))
        self.assertEqual("stored", rt.choose_source("auto", rt.STORED_TARGET - 1, 5))
        self.assertEqual("new", rt.choose_source("auto", rt.STORED_TARGET, 5))
        self.assertEqual("new", rt.choose_source("auto", 10, 0))  # no stored run left to replay
        self.assertEqual("new", rt.choose_source("new", 0, 100))
        self.assertEqual("stored", rt.choose_source("stored", rt.STORED_TARGET * 2, 0))

    def test_the_kinds_a_job_covers(self):
        self.assertEqual(rt.TEACHER_KINDS, rt.kinds_of("all"))
        self.assertEqual((), rt.kinds_of("none"))
        self.assertEqual(("card", "rest"), rt.kinds_of("card, rest"))
        with self.assertRaises(SystemExit):
            rt.kinds_of("card,potion")

    def test_declined_runs_leave_no_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "runs.jsonl"
            rows = [{"_task": ["rt-new", "A", 1], "seed": "A", "cpu": 1.0}, {"_task": ["rt-new", "B", 2], "seed": "B", "capped": True, "cap_reason": "budget"},
                    {"_task": ["rt-new", "C", 3], "seed": "C", "capped": True, "cap_reason": "block-end"},
                    {"_task": ["rt-new", "D", 4], "seed": "D", "capped": True, "cap_reason": "deadline (killed)"}]
            f.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            rt.drop_skips(f)
            self.assertEqual(["A", "D"], [r["seed"] for r in rt.read_jsonl(f)])  # a killed run's cap row stays as the record of what happened
            self.assertEqual(["A"], [r["seed"] for r in rt.done_rows(Path(tmp))])


class WorkerGateTests(unittest.TestCase):
    """`_gate` (a worker may start a run) and `_screen` (a stored screen is reused when the replay is where it was)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        saved = dict(rt._J)
        self.addCleanup(lambda: (rt._J.clear(), rt._J.update(saved)))
        rt._J.clear()
        rt._J.update(deadline=time.time() + 3600, stop_file=str(self.tmp / "STOP"), ledger=str(self.tmp / "ledger.jsonl"), job="J", budget_h=1.0,
                     gen_dir=self.tmp, k=2, k_max=4, max_options=4, cal_map={}, player=SimpleNamespace(policy=None))

    def ledger(self):
        from tools.lab import jobs

        return jobs.read_ledger(self.tmp / "ledger.jsonl")

    def test_a_worker_starts_a_run_and_reserves_its_estimate(self):
        self.assertIsNone(rt._gate(1800.0, "A"))
        self.assertEqual([("start", "A", 1800.0)], [(r["ev"], r["seed"], r["est"]) for r in self.ledger()])

    def test_the_budget_has_no_room_for_a_run_it_cannot_pay_for_and_the_earlier_reservation_wins(self):
        self.assertIsNone(rt._gate(2400.0, "A"))  # 40 minutes of the hour
        self.assertEqual("budget", rt._gate(2400.0, "B"))  # does not fit: retracted, not left standing
        ev = [(r["ev"], r["seed"]) for r in self.ledger()]
        self.assertEqual([("start", "A"), ("start", "B"), ("end", "B")], ev)
        self.assertTrue(self.ledger()[-1]["retracted"])
        self.assertIsNone(rt._gate(1000.0, "C"))  # a smaller one still fits beside A
        from tools.lab import jobs

        rows = self.ledger()
        self.assertAlmostEqual((2400 + 1000) / 3600, jobs.teacher_spent_s(rows, time.strftime("%Y-%m-%d", time.gmtime()), time.time()) / 3600, places=3)

    def test_the_block_s_end_and_the_stop_file_decline_a_run_before_it_reserves(self):
        rt._J["deadline"] = time.time() - 1
        self.assertEqual("block-end", rt._gate(10.0, "A"))
        rt._J["deadline"] = time.time() + 60
        (self.tmp / "STOP").write_text("x")
        self.assertEqual("stop-file", rt._gate(10.0, "A"))
        self.assertEqual([], self.ledger())

    def test_a_skipped_task_leaves_a_capped_row_with_its_reason(self):
        row = rt._skip_row(("rt-label", "S", 4, {"big": "record"}), "budget")
        self.assertEqual({"_task": ["rt-label", "S", 4], "arm": "label", "seed": "S", "ix": 4, "capped": True, "cap_reason": "budget", "wall": 0.0}, row)
        self.assertEqual(["rt-new", "S", 9], rt._skip_row(("rt-new", "S", 9), "block-end")["_task"])

    def screen_setup(self):
        rf = rt.RunFiles(self.tmp, "S1")
        rf.load()
        rf.append(synthetic_record("S1", 4, n=3, floor=5, player=1), rows_for(3, 1.0))
        rf.append(synthetic_record("S1", 6, n=3, floor=7, player=1), rows_for(3, 2.0))
        return rf

    def test_a_stored_screen_is_reused_when_the_replay_stands_where_it_was(self):
        rf = self.screen_setup()
        d, view, pick = fake_decision(), SimpleNamespace(floor=5, act_index=0), SimpleNamespace(index=1)
        counters = {"reused": 0, "reused_cpu": 0.0}
        with mock.patch.object(rt, "decide", side_effect=AssertionError("must not play it again")):
            got = rt._screen(rf, 4, d, view, pick, None, "S1", counters)
        self.assertEqual(4, got["ordinal"])
        self.assertEqual((1, 12.5), (counters["reused"], counters["reused_cpu"]))

    def test_a_screen_that_differs_is_played_again_and_what_followed_is_forgotten(self):
        rf = self.screen_setup()
        d, view, pick = fake_decision(), SimpleNamespace(floor=5, act_index=0), SimpleNamespace(index=2)  # the player now picks another option
        counters = {"reused": 0, "reused_cpu": 0.0}
        fresh = synthetic_record("S1", 4, n=3, floor=5, player=2)
        rows = rows_for(3, 5.0)

        def fake_decide(*a, rows_out=None, **kw):
            rows_out.append(rows)
            return dict(fresh)

        with mock.patch.object(rt, "decide", side_effect=fake_decide):
            got = rt._screen(rf, 4, d, view, pick, None, "S1", counters)
        self.assertEqual(0, got["row0"])  # the old screens were cut from this ordinal on, so it starts at row 0 again
        again = rt.RunFiles(self.tmp, "S1")
        self.assertEqual([4], sorted(again.load()))
        self.assertEqual(0, counters["reused"])

    def test_reusing_needs_the_same_run_type_floor_options_and_player_pick(self):
        have = synthetic_record("S1", 4, n=3, floor=5, player=1)
        d = fake_decision()
        view = SimpleNamespace(floor=5)
        pick = SimpleNamespace(index=1)
        self.assertTrue(rt._reusable(have, "S1", d, view, pick))
        self.assertFalse(rt._reusable(None, "S1", d, view, pick))
        self.assertFalse(rt._reusable(have, "S2", d, view, pick))
        self.assertFalse(rt._reusable(have, "S1", d, SimpleNamespace(floor=6), pick))
        self.assertFalse(rt._reusable(have, "S1", d, view, SimpleNamespace(index=0)))
        self.assertFalse(rt._reusable(have, "S1", fake_decision(kinds=("a", "b")), view, pick))
        self.assertFalse(rt._reusable(have, "S1", fake_decision(typ="shop"), view, pick))


class LabelHookTests(unittest.TestCase):
    """`label_run` on a stand-in replay: which screens of a stored run are labelled, and that the run is left as it is."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        saved = dict(rt._J)
        self.addCleanup(lambda: (rt._J.clear(), rt._J.update(saved)))
        rt._J.clear()
        rt._J.update(deadline=time.time() + 3600, stop_file="", ledger=str(self.tmp / "ledger.jsonl"), job="J", budget_h=100.0, gen_dir=self.tmp, k=2,
                     k_max=4, max_options=4, cal_map={}, label_rate=1.0, per_run=3, kinds=rt.TEACHER_KINDS, replayers={},
                     player=SimpleNamespace(policy=SimpleNamespace(pick=lambda view, d, rng: SimpleNamespace(index=0))), est_label_s=100.0)

    def run_label(self, floors, rate=1.0, per_run=3):
        rt._J.update(label_rate=rate, per_run=per_run)
        seen = []

        def fake_replay(seed, ix, replayer, record, hook, **kw):
            res = {"end": "?", "decisions": 0, "floor": 0, "checked": 0}
            for ordinal, floor in enumerate(floors):
                pt = SimpleNamespace(view=SimpleNamespace(floor=floor, act_index=0), decision=fake_decision(), ordinal=ordinal, session=None)
                if hook(pt):
                    res["end"] = "hooked"
                    break
                res["decisions"] += 1
                seen.append(ordinal)
            res["floor"] = floors[-1]
            return res

        def fake_decide(s, d, view, pick, player, *, seed, ordinal, rows_out=None, **kw):
            rows_out.append(rows_for(3, ordinal))
            return synthetic_record(seed, ordinal, floor=int(view.floor), teacher=1 if ordinal == 2 else 0)

        task = ("rt-label", "SEEDX", 7, {"_job": "20261106T0000-selfplay", "_explore": (0.1, 0.02), "floor": 30})
        with mock.patch.object(rt, "_replayer", return_value=object()), mock.patch.object(rt, "decide", side_effect=fake_decide), \
                mock.patch("hindsight.replay_run", side_effect=fake_replay):
            return rt.label_run(task), seen

    def test_the_replay_stops_after_the_runs_quota_and_the_run_is_not_changed(self):
        row, seen = self.run_label([2, 3, 4, 5, 6, 7])
        self.assertEqual(3, row["teacher_screens"])
        self.assertEqual("hooked", row["replay"]["end"])
        self.assertEqual(1, row["changed"])  # the teacher's pick differed at one of the three: recorded, never applied
        self.assertEqual(2, len(seen))  # the third labelled screen ended the replay at once
        self.assertEqual({"card/a1": 3}, row["screens"])
        self.assertEqual("label", row["arm"])
        self.assertEqual(["rt-label", "SEEDX", 7], row["_task"])
        self.assertEqual([0, 1, 2], sorted(rt.RunFiles(self.tmp, "SEEDX").load()))

    def test_a_coin_picks_the_screens_and_nothing_past_the_act_2_boss_is_labelled(self):
        row, _ = self.run_label([2, 3, 4, 5, 6, 7, 8, 9], rate=0.0)
        self.assertEqual(0, row["teacher_screens"])
        row, _ = self.run_label([30, 31, 33, 34], rate=1.0, per_run=9)
        self.assertEqual(2, row["teacher_screens"])  # floor 33 is the boss floor: the replay stops there
        self.assertEqual("hooked", row["replay"]["end"])

    def test_the_coin_is_a_fixed_hash_of_the_seed_and_the_ordinal(self):
        self.assertEqual(rt.coin("A", 3), rt.coin("A", 3))
        vals = [rt.coin("A", i) for i in range(2000)]
        self.assertTrue(all(0.0 <= v < 1.0 for v in vals))
        self.assertAlmostEqual(sum(1 for v in vals if v < rt.LABEL_RATE) / 2000, rt.LABEL_RATE, delta=0.04)
        self.assertNotEqual([rt.coin("A", i) for i in range(5)], [rt.coin("B", i) for i in range(5)])


class NoHumanRecordTests(unittest.TestCase):
    """No human record is read anywhere in the teacher or its job: no corpus path, no human-starts file, no exam list."""

    def test_the_sources_name_no_human_corpus_file_or_exam_list(self):
        for rel in ("rl/run_teacher.py", "tools/lab/jobs.py"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            for word in ("spire-codex", "spire_codex", "runs-v0.107", "human-starts", "human_starts", "exam_bucket", "exam_ids", "EXAM_IDS", "map_point_history"):
                self.assertNotIn(word, text, f"{rel} names {word}")
            self.assertNotIn("import human_prior\n", text)
            self.assertNotRegex(text, r"^\s*(import|from) human_(prior|starts|corpus)\b", rel)
        self.assertIn("No human record is read", (ROOT / "rl" / "run_teacher.py").read_text(encoding="utf-8"))

    def test_the_manifest_says_so(self):
        self.assertEqual("not read", rt.shard_manifest(job="j", generation=1, identity={}, judge=None, lib=None, tree=None, k=2, k_max=4, max_options=4,
                                                      cal=None, sources={})["human_corpus"])


class JobCommandTests(unittest.TestCase):
    def test_the_job_and_label_parsers(self):
        a = rt.build_parser().parse_args(["job"])
        self.assertEqual((a.k, a.k_max, a.procs, a.source, a.block_min, a.share, a.loop), (rt.JOB_K, rt.JOB_K_MAX, 4, "auto", 60.0, False, False))
        self.assertEqual((rt.JOB_K, rt.JOB_K_MAX), (2, 4))
        a = rt.build_parser().parse_args(["label", "--per-run", "7", "--stored-from", "20261106"])
        self.assertEqual((a.cmd, a.per_run, a.stored_from), ("label", 7, "20261106"))

    def test_a_held_job_writes_a_result_and_plays_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lab = tmp / "LAB"
            lab.mkdir()
            (lab / "PAUSE").write_text("x")
            agents = tmp / "agent"
            agents.mkdir()
            (agents / "current.json").write_text(json.dumps({"generation": 13, "dir": "D:/nowhere/gen13/x"}))
            result = tmp / "result.json"
            a = rt.build_parser().parse_args(["job", "--lab", str(lab), "--store", str(tmp / "store"), "--agent-root", str(agents), "--result", str(result)])
            with mock.patch.object(rt, "job_block", side_effect=AssertionError("held: no block")):
                self.assertEqual(0, rt.job_cmd(a))
            res = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual("LAB/PAUSE is up", res["held"])
            self.assertEqual((0, 0, 13), (res["runs"], res["decisions"], res["generation"]))

    def test_one_job_at_a_time_on_a_store(self):
        from tools.lab import winsys

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            self.assertIsNone(rt.take_store_lock(store))
            self.assertIsNone(rt.take_store_lock(store))  # our own lock
            (store / "job.lock").write_text(json.dumps({"pid": 4242, "t": "2026-11-11T00:00:00Z"}), encoding="utf-8")
            with mock.patch.object(winsys, "parent_alive", return_value=True):
                self.assertEqual("another teacher job holds the store (pid 4242)", rt.take_store_lock(store))
            rt.drop_store_lock(store)  # not ours: left alone
            self.assertTrue((store / "job.lock").exists())
            with mock.patch.object(winsys, "parent_alive", return_value=False):
                self.assertIsNone(rt.take_store_lock(store))  # its job is gone: taken over
            rt.drop_store_lock(store)
            self.assertFalse((store / "job.lock").exists())

    def test_a_block_that_lost_every_run_it_tried_fails_and_one_that_played_does_not(self):
        from tools.lab import jobs

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "LAB").mkdir()
            (tmp / "agent").mkdir()
            (tmp / "agent" / "current.json").write_text(json.dumps({"generation": 13, "dir": "D:/nowhere/gen13/x"}), encoding="utf-8")
            a = rt.build_parser().parse_args(["job", "--lab", str(tmp / "LAB"), "--store", str(tmp / "store"), "--agent-root", str(tmp / "agent")])
            with mock.patch.object(jobs, "_git_main", return_value=None):
                with mock.patch.object(rt, "job_block", return_value={"runs": 0, "capped": 5, "decisions": 0}):
                    self.assertEqual(1, rt.job_cmd(a))
                self.assertFalse((tmp / "store" / "job.lock").exists())  # the lock is dropped whatever the block did
                with mock.patch.object(rt, "job_block", return_value={"runs": 2, "capped": 1, "decisions": 9}):
                    self.assertEqual(0, rt.job_cmd(a))
                with mock.patch.object(rt, "job_block", return_value={"runs": 0, "capped": 0, "decisions": 0, "held": "x"}):
                    self.assertEqual(0, rt.job_cmd(a))  # nothing to do is not a failure

    def test_no_player_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = rt.build_parser().parse_args(["job", "--lab", tmp, "--store", tmp, "--agent-root", tmp])
            self.assertEqual(2, rt.job_cmd(a))

    def test_the_report_reads_the_label_arm_beside_the_teacher_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            runs = [{"_task": ["rt-new", "N1", 1], "arm": "teacher", "source": "new", "seed": "N1", "ix": 1, "floor": 20, "cpu": 900.0, "wall": 900.0,
                     "teacher_cpu": 880.0, "teacher_screens": 2, "screens": {"card/a1": 2}, "bosses": []},
                    {"_task": ["rt-label", "L1", 2], "arm": "label", "source": "stored", "seed": "L1", "ix": 2, "floor": 12, "cpu": 100.0, "wall": 100.0,
                     "teacher_cpu": 80.0, "teacher_screens": 1, "screens": {"card/a1": 1}}]
            (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs), encoding="utf-8")
            dd = out / "decisions.d"
            dd.mkdir()
            for s in ("N1", "L1"):
                (dd / f"{s}.jsonl").write_text(json.dumps(synthetic_record(s, 0)) + "\n", encoding="utf-8")
                (dd / f"{s}.rows.f32").write_bytes(b"")  # not read as decisions
            res = rt.report(out)
            self.assertEqual({"teacher": 1, "label": 1}, res["runs"])
            self.assertEqual(2, res["screens"]["all"]["screens"])
            self.assertEqual(100.0, res["label"]["cpu_mean"])
            self.assertEqual(rt.stored_screens_done(out), 1)



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


@unittest.skipUnless(NATIVE, "the native library of this tree is not built")
class NativeJobTests(unittest.TestCase):
    """`new_run` on the real engine with a stand-in player (the rules run policy, fights resolved by the engine's own resolver, a judge that
    says 0.5) and a stand-in `decide`: the run loop, the run record, the doors, the replay of a killed run. The mechanism, not the agent."""

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

            def describe(self):
                return {"kind": "rules"}

        def play_fight(s, rng, deadline):
            from sts2_gym import native

            made = 0
            while s.phase == engine.rc.PHASE_COMBAT:
                native.run_debug_resolve_combat(s.handle, s.env._run_obs_buf)
                made += 1
                if made > 3:
                    break
            return made

        agent = SimpleNamespace(generation=0, prior_dir=None, search_config=lambda name: SimpleNamespace(sims=4, c=0.1, name="stand-in"))
        cls.player = hindsight.Player(policy=Policy(), play_fight=play_fight, head=lambda v, d: None, judge=lambda handle, max_hp: {"p_death": 0.5},
                                      agent=agent)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        saved = dict(rt._J)
        self.addCleanup(lambda: (rt._J.clear(), rt._J.update(saved)))
        rt._J.clear()
        rt._J.update(deadline=time.time() + 3600, stop_file="", ledger=str(self.tmp / "ledger.jsonl"), job="J", budget_h=1000.0, gen_dir=self.tmp, k=2,
                     k_max=4, max_options=4, cal_map={}, player=self.player, kinds=rt.TEACHER_KINDS, est_new_s=10.0, stop_floor=0)
        engine, hindsight = self.engine, self.hindsight
        patch = mock.patch.object(hindsight, "session_for", lambda seed, agent, config="a3-v1", ascension=10, own_schedule=False:
                                  engine.RunSession(seed, 10, engine.CombatBudget(4, 4)))
        patch.start()
        self.addCleanup(patch.stop)
        self.decided: list[int] = []

    def fake_decide(self, flip_at=None):
        """A stand-in teacher: options 0 and 1 'valued', the player's own pick taken (or another at ordinal `flip_at`); the rows are zeros."""
        n_calls = self.decided

        def decide(s, d, view, pick, player, *, seed, ordinal, rows_out=None, **kw):
            n_calls.append(ordinal)
            rows_out.append(self.np.full((len(d.options), rt.ROWS_WIDTH), float(ordinal), dtype="float32"))
            teacher = (int(pick.index) + 1) % len(d.options) if ordinal == flip_at else int(pick.index)
            return synthetic_record(seed, ordinal, n=len(d.options), floor=int(view.floor), player=int(pick.index), teacher=teacher, type=d.type,
                                    kind=rt.KIND_OF_TYPE.get(d.type))

        return decide

    def play(self, seed, *, kinds, decide=None):
        rt._J["kinds"] = kinds
        with mock.patch.object(rt, "decide", decide or self.fake_decide()):
            return rt.new_run(("rt-new", seed, 5))

    @staticmethod
    def same(a: dict, b: dict) -> tuple:
        return tuple(a[k] for k in ("end", "floor", "decisions", "screens", "bosses"))

    def test_a_teacher_that_agrees_with_the_player_everywhere_is_the_players_run(self):
        plain = self.play("RTJOB0001", kinds=())
        self.assertEqual([], self.decided)
        self.assertEqual(0, plain["teacher_screens"])
        agree = self.play("RTJOB0001", kinds=rt.TEACHER_KINDS)
        self.assertGreater(agree["teacher_screens"], 0)
        self.assertEqual(0, agree["changed"])
        self.assertEqual(self.same(plain, agree), self.same(agree, plain))
        self.assertEqual(plain["floor"], agree["floor"])
        self.assertEqual(plain["decisions"], agree["decisions"])
        self.assertEqual(plain["bosses"], agree["bosses"])

    def test_the_run_record_and_the_doors_are_kept_and_the_budget_ledger_is_written(self):
        row = self.play("RTJOB0002", kinds=())
        files = rt.RunFiles(self.tmp, "RTJOB0002")
        self.assertEqual(row["end"] in ("win", "death"), files.record.is_file())
        if files.record.is_file():
            rec = json.loads(files.record.read_text(encoding="utf-8"))
            self.assertEqual("RTJOB0002", rec["seed"])
            self.assertEqual(row["floor"] if row["end"] == "death" else rt.WON_FLOOR, rec["outcome"]["final_floor"])
            self.assertTrue(rec["decisions"])
        for b in row["bosses"]:
            self.assertEqual(0.5, b["p_death"])  # the judge's number at the door, beside the fight's outcome
            self.assertIn("won", b)
        from tools.lab import jobs

        ev = [(r["ev"], r["seed"]) for r in jobs.read_ledger(self.tmp / "ledger.jsonl")]
        self.assertEqual([("start", "RTJOB0002"), ("end", "RTJOB0002")], ev)
        self.assertEqual(row["cpu"], jobs.read_ledger(self.tmp / "ledger.jsonl")[-1]["cpu"])

    def test_a_teacher_pick_is_what_is_applied_and_what_the_run_record_keeps(self):
        agree = self.play("RTJOB0003", kinds=rt.TEACHER_KINDS)
        flip = self.play("RTJOB0004", kinds=rt.TEACHER_KINDS, decide=self.fake_decide(flip_at=0))
        self.assertEqual(1, flip["changed"])
        files = rt.RunFiles(self.tmp, "RTJOB0004")
        files.load()
        first = files.by_ordinal[0]
        self.assertNotEqual(first["player"], first["teacher"])
        if files.record.is_file():
            rec = json.loads(files.record.read_text(encoding="utf-8"))
            self.assertEqual(first["teacher"], rec["decisions"][0]["chosen"])  # the run record has the option that was taken
        self.assertGreater(agree["teacher_screens"], 0)

    def test_a_run_that_is_played_again_reuses_its_screens_and_is_the_same_run(self):
        first = self.play("RTJOB0005", kinds=rt.TEACHER_KINDS)
        n_first = len(self.decided)
        self.assertEqual(first["teacher_screens"], n_first)
        self.decided.clear()
        again = self.play("RTJOB0005", kinds=rt.TEACHER_KINDS, decide=mock.Mock(side_effect=AssertionError("a stored screen is played again")))
        self.assertEqual([], self.decided)
        self.assertEqual(n_first, again["reused_screens"])
        self.assertEqual(self.same(first, again), self.same(again, first))
        self.assertEqual((first["floor"], first["decisions"], first["end"]), (again["floor"], again["decisions"], again["end"]))
        self.assertEqual(first["teacher_screens"], again["teacher_screens"])

    def test_a_run_stopped_in_the_middle_is_finished_by_its_replay(self):
        full = self.play("RTJOB0006", kinds=rt.TEACHER_KINDS)
        files = rt.RunFiles(self.tmp, "RTJOB0006")
        files.load()
        keep = sorted(files.by_ordinal)[: max(1, len(files.by_ordinal) // 2)]
        files.drop_from(keep[-1] + 1)  # a kill that left the first half of the screens
        self.decided.clear()
        done = self.play("RTJOB0006", kinds=rt.TEACHER_KINDS)
        self.assertEqual(len(keep), done["reused_screens"])
        self.assertEqual(full["teacher_screens"] - len(keep), len(self.decided))  # only the screens it did not have are played
        self.assertEqual(self.same(full, done), self.same(done, full))

    def test_the_run_records_go_to_the_labs_shards_and_the_decisions_to_the_store(self):
        import gen_loop

        rows = []
        for seed in ("RTJOB0010", "RTJOB0011"):
            rows.append(self.play(seed, kinds=rt.TEACHER_KINDS))
        (self.tmp / "runs.jsonl").write_text("".join(json.dumps(r) + chr(10) for r in rows), encoding="utf-8")
        agent = SimpleNamespace(generation=0, search_config=lambda name: SimpleNamespace(sims=4, c=0.1, name="stand-in"), describe=lambda: {"generation": 0})
        ctx = {"generation": 0, "identity": {"agent": "gen0/x", "agent_sha256": "0" * 64}, "judge": None, "lib": None, "tree": "abc", "k": 2, "k_max": 4,
               "max_options": 4, "cal_doc": {}}
        doc = rt.consolidate(self.tmp, "JOB1", ctx, agent)
        self.assertEqual(sum(r["teacher_screens"] for r in rows), doc["counts"]["decisions"])
        self.assertEqual(["RTJOB0010", "RTJOB0011"], doc["seeds"])
        recorded = [r["seed"] for r in rows if r["end"] in ("win", "death")]
        found = gen_loop.find_manifests(self.tmp / "jobs")
        self.assertTrue(recorded, "the stand-in runs end in a win or a death")
        self.assertEqual(1, len(found))
        self.assertEqual("selfplay-shards", found[0][1]["kind"])  # the run-value refit reads these as it reads the lab's own
        self.assertEqual(set(recorded), gen_loop.shard_run_seeds(found[0][0], 0))
        again = rt.consolidate(self.tmp, "JOB2", ctx, agent)
        self.assertEqual(0, again["counts"]["decisions"])  # a finished run is in one shard only
        self.assertEqual([], again["seeds"])
        self.assertEqual(doc["counts"]["decisions"], rt.store_manifest(self.tmp, ctx)["totals"]["decisions"])

    def test_the_stop_floor_ends_a_smoke_run_without_a_record(self):
        rt._J["stop_floor"] = 4
        row = self.play("RTJOB0007", kinds=())
        self.assertIn(row["end"], ("stopped", "death", "win"))
        if row["end"] == "stopped":
            self.assertFalse(rt.RunFiles(self.tmp, "RTJOB0007").record.is_file())

    def test_a_run_the_budget_has_no_room_for_is_not_played(self):
        rt._J["budget_h"] = 0.0
        row = self.play("RTJOB0008", kinds=rt.TEACHER_KINDS)
        self.assertTrue(row["capped"])
        self.assertEqual("budget", row["cap_reason"])
        self.assertEqual([], self.decided)
        self.assertFalse(rt.RunFiles(self.tmp, "RTJOB0008").recs.exists())


if __name__ == "__main__":
    unittest.main()
