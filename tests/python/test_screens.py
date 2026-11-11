"""The cheap screens before the judgement's arms (D55 addendum 18; rl/screens.py, gen_loop.promote): a candidate that differs from the incumbent
in the fight network only plays the fight exam's human stratum and is rejected (screen:fight) when its survival is below the incumbent's by more
than two standard errors; one that differs in the human prior only is rejected (screen:prior) when its held-out human-choice log-loss is worse;
anything else plays the arms as before. No game is played: the exam's games are written by a stand-in, the prior scorer is injected."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import fight_exam as fe  # noqa: E402
import gen_loop  # noqa: E402
import keep_v2_fixtures as fx  # noqa: E402
import screens  # noqa: E402
from test_fight_exam import det_games, human_gym_folder, human_pool  # noqa: E402
from test_funnel import StubArms, make_block, stub_candidate  # noqa: E402


def folder(root: Path, name: str, *, fight: bytes = b"f1", prior: bytes | None = b"p1", value: bytes = b"v1", fight_net: bytes = b"n1",
           macro: str = "vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7", lam: str = "0.5", generation: int = 4) -> Path:
    """A candidate folder with the parts the screens compare."""
    d = Path(root) / name
    d.mkdir(parents=True)
    (d / "combat_policy.pt").write_bytes(fight)
    (d / "combat_net.pt").write_bytes(fight + b"-net")
    (d / "run_value.npz").write_bytes(value)
    (d / "fight_net.onnx").write_bytes(fight_net)
    if prior is not None:
        (d / "human_prior.npz").write_bytes(prior)
    doc = {"generation": generation, "macro": macro, "recipe": macro + "+stakes:a3-v1@h1e4b16", "mask_version": 4,
           "search_schedule": {"base": "a3-v1", "mult": {"hall": 1, "elite": 4, "boss": 16}, "tag": "a3-v1@h1e4b16"},
           "human_prior": {"file": "human_prior.npz", "sha256": "x" * 64, "lambda": lam, "source": "w.npz"}, "written": f"2026-11-0{generation}T00:00:00Z",
           "lineage": {"decision": f"built for {name}"}}
    (d / "agent.json").write_text(json.dumps(doc), encoding="utf-8")
    (d / "model.json").write_text(json.dumps({"iteration": generation, "policy_sha256": "ab" * 32}), encoding="utf-8")
    return d


class KindTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))
        self.base = folder(self.tmp, "base", generation=3)

    def test_identical_folders_and_bookkeeping_alone_get_no_screen(self):
        same = folder(self.tmp, "same", generation=5)  # another generation, date and lineage: bookkeeping
        self.assertEqual(screens.differing(same, self.base), [])
        self.assertIsNone(screens.kind_of(same, self.base))

    def test_only_the_fight_network_differing_is_the_fight_screen(self):
        c = folder(self.tmp, "c", fight=b"f2")
        self.assertEqual(screens.differing(c, self.base), ["fight"])
        self.assertEqual(screens.kind_of(c, self.base), "fight")

    def test_only_the_human_prior_differing_is_the_prior_screen_though_its_tag_is_in_the_macro(self):
        c = folder(self.tmp, "c", prior=b"p2", macro="vrun+pi_h:lambda=0.5:pi_h=9c1d4e5f")  # the macro and recipe name the prior by 8 hex digits
        self.assertEqual(screens.differing(c, self.base), ["prior"])
        self.assertEqual(screens.kind_of(c, self.base), "prior")

    def test_two_parts_differing_or_any_other_part_gets_no_screen(self):
        self.assertIsNone(screens.kind_of(folder(self.tmp, "a", fight=b"f2", prior=b"p2"), self.base))  # fight and prior
        self.assertIsNone(screens.kind_of(folder(self.tmp, "b", value=b"v2"), self.base))  # the run value
        self.assertIsNone(screens.kind_of(folder(self.tmp, "c", fight_net=b"n2"), self.base))  # the run judgement's FightNet: no fight in it
        self.assertIsNone(screens.kind_of(folder(self.tmp, "d", lam="0.0"), self.base))  # the prior's lambda is a setting
        self.assertIsNone(screens.kind_of(folder(self.tmp, "e", macro="vrun"), self.base))  # another macro
        self.assertIsNone(screens.kind_of(folder(self.tmp, "f", fight=b"f2", lam="0.0"), self.base))  # a fight network and a setting
        self.assertEqual(screens.differing(folder(self.tmp, "g", fight=b"f2", value=b"v2"), self.base), ["fight", "value"])

    def test_a_candidate_with_no_prior_against_one_with_a_prior_is_a_prior_difference(self):
        c = folder(self.tmp, "c", prior=None)
        self.assertEqual(screens.differing(c, self.base), ["prior"])

    def test_a_folder_that_is_not_there_never_raises(self):
        self.assertIsNotNone(screens.differing(self.tmp / "nope", self.base))


class FightScreenTests(unittest.TestCase):
    """`screens.run_fight` on a gym folder with a human stratum and the incumbent's games (no game is played: `play` writes the candidate's)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))
        self.loop, self.exam, self.sha = human_gym_folder(self.tmp, per_cell=3)
        self.human = fe.load_human_exam(self.exam)
        self.rank = {sid: i for i, sid in enumerate(sorted(self.human))}

    def player(self, win, engine: str = "e1", rc: int = 0, draws: int = 4):
        """A `play` stand-in: writes the candidate's games (`win(start, j)`) to its file, returns rc."""
        def play(label: str) -> int:
            rows = det_games(self.human, win, lambda s, j: 12, draws, engine=engine)
            (self.exam / f"games-{label}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows.values()), encoding="utf-8")
            return rc
        return play

    def screen(self, play, **kw):
        return screens.run_fight(self.tmp / "cand", self.sha, "lab-g4-cand-ab12cd34", play, exam_dir=self.exam, loop_root=self.loop, boot=300, **kw)

    def test_a_candidate_as_good_as_the_incumbent_passes(self):
        r = self.screen(self.player(lambda s, j: s["human_survived"]))
        self.assertFalse(r["rejected"])
        self.assertEqual((r["starts"], r["cand_surv"], r["inc_surv"], r["diff"]), (len(self.human), r["inc_surv"], r["inc_surv"], 0.0))
        self.assertIn("screen:fight passed", r["line"])
        self.assertEqual(r["incumbent"], "v2")

    def test_a_candidate_far_below_the_incumbent_is_rejected_by_more_than_two_standard_errors(self):
        r = self.screen(self.player(lambda s, j: False))
        self.assertTrue(r["rejected"])
        self.assertLess(r["diff"], -2 * r["se"])
        self.assertLess(r["z"], -2.0)
        self.assertIn("screen:fight REJECTED", r["line"])
        self.assertEqual(r["rule"], "screen:fight")

    def test_a_candidate_a_little_below_passes(self):
        lose_one = next(iter(sorted(self.human)))
        r = self.screen(self.player(lambda s, j: s["human_survived"] and s["id"] != lose_one))
        if self.human[lose_one]["human_survived"]:
            self.assertLess(r["diff"], 0)
            self.assertGreaterEqual(r["z"], -2.0)
        self.assertFalse(r["rejected"])

    def test_a_better_candidate_passes(self):
        r = self.screen(self.player(lambda s, j: True))
        self.assertFalse(r["rejected"])
        self.assertGreaterEqual(r["diff"], 0)

    def test_the_bound_is_two_standard_errors_exactly(self):
        # draws are alike per start, so the figures are the starts': win on a share of the survivors and sit the difference on the line
        r = screens.fight_reading(det_games(self.human, lambda s, j: s["human_survived"], lambda s, j: 12, engine="e1"),
                                  det_games(self.human, lambda s, j: s["human_survived"], lambda s, j: 12, engine="e1"), self.human, boot=100)
        self.assertEqual((r["diff"], r["rejected"]), (0.0, False))
        worse = det_games(self.human, lambda s, j: s["human_survived"] and self.rank[s["id"]] % 4 != 0, lambda s, j: 12, engine="e1")
        base = det_games(self.human, lambda s, j: s["human_survived"], lambda s, j: 12, engine="e1")
        for bound in (0.5, 1.0, 100.0):
            q = screens.fight_reading(worse, base, self.human, boot=300, se_bound=bound)
            self.assertEqual(q["rejected"], q["diff"] < -bound * q["se"], bound)

    def test_engines_that_differ_are_not_compared(self):
        r = self.screen(self.player(lambda s, j: False, engine="e2"))  # the incumbent's games carry engine e1
        self.assertFalse(r["rejected"])
        self.assertIn("engines differ (candidate e2, incumbent e1)", r["skipped"])

    def test_a_run_that_did_not_finish_its_games_is_skipped_not_rejected(self):
        r = self.screen(self.player(lambda s, j: False, rc=3, draws=1))
        self.assertIn("not all played (exit 3, ", r["skipped"])
        self.assertNotIn("rejected", r)

    def test_no_games_for_the_incumbents_network_or_no_human_stratum_is_skipped(self):
        r = screens.run_fight(self.tmp / "cand", "ff" * 32, "lab-x", lambda label: 0, exam_dir=self.exam, loop_root=self.loop, boot=100)
        self.assertIn("no human-stratum games for the incumbent's fight network (no gym player holds these bytes)", r["skipped"])
        loop2, exam2, sha2 = human_gym_folder(self.tmp / "two", games=False)
        r = screens.run_fight(self.tmp / "cand", sha2, "lab-x", lambda label: 0, exam_dir=exam2, loop_root=loop2, boot=100)
        self.assertIn("no human-stratum games for the incumbent's fight network (gym player v2)", r["skipped"])
        (self.tmp / "empty").mkdir()
        r = screens.run_fight(self.tmp / "cand", self.sha, "lab-x", lambda label: 0, exam_dir=self.tmp / "empty", loop_root=self.loop, boot=100)
        self.assertEqual(r["skipped"], "the fight exam has no human stratum")


class PriorScreenTests(unittest.TestCase):
    """screen:prior is the selection rule of the strong sets (D55 addendum 15 item 3) on the held-out leg: better on the won runs by more than the
    0.002 tie margin, worse on all runs by no more than 0.02 nats; rejected when the primary or the guard fails."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))
        self.base = folder(self.tmp, "base", generation=3)
        self.cand = folder(self.tmp, "cand", prior=b"p2", generation=4)
        self.work = self.tmp / "work"
        (self.work / "prior-a10" / "decisions").mkdir(parents=True)
        (self.work / "prior-a10" / "donors.npz").write_bytes(b"x")

    def run_with(self, won: float, allr: float, base_a: float = 0.87, base_b: float = 0.95):
        """The screen with the candidate `won` nats from the incumbent on the won runs and `allr` on all runs (negative: better)."""
        def legs(paths, work):
            self.assertEqual([Path(x).parent.name for x in paths], ["cand", "base"])
            return [{"a": base_a + won, "b": base_b + allr, "runs_won": 384, "runs_all": 3601}, {"a": base_a, "b": base_b, "runs_won": 384, "runs_all": 3601}]
        return screens.run_prior(self.cand, self.base, work=self.work, legs=legs)

    def test_the_best_players_case_passes_better_on_the_won_runs_and_worse_on_all_within_the_limit(self):
        r = self.run_with(-0.008, 0.017)  # candidate-best-players on the half splits: won -0.008 to -0.013, all +0.017 to +0.018
        self.assertFalse(r["rejected"])
        self.assertEqual((r["primary_ok"], r["guard_ok"]), (True, True))
        self.assertEqual(r["ledger_rule"], "screen:prior won=-0.008 all=+0.017")
        self.assertIn("screen:prior passed", r["line"])
        self.assertEqual((r["runs_won"], r["runs_all"]), (384, 3601))
        self.assertFalse(self.run_with(-0.013, 0.018)["rejected"])

    def test_the_primary_must_beat_the_incumbent_by_more_than_the_tie_margin(self):
        for won in (0.0, 0.004, -0.001, -0.002):  # equal, worse, better by less than the margin, by exactly the margin
            r = self.run_with(won, 0.0)
            self.assertTrue(r["rejected"], won)
            self.assertFalse(r["primary_ok"], won)
            self.assertIn("the won-run log-loss is not better by more than 0.002", r["line"])
        self.assertFalse(self.run_with(-0.0021, 0.0)["rejected"])

    def test_the_guard_is_worse_by_no_more_than_two_hundredths_on_all_runs(self):
        r = self.run_with(-0.03, 0.0201)
        self.assertTrue(r["rejected"])
        self.assertEqual((r["primary_ok"], r["guard_ok"]), (True, False))
        self.assertIn("the all-run log-loss is worse by more than 0.02", r["line"])
        self.assertEqual(r["ledger_rule"], "screen:prior won=-0.030 all=+0.020")
        self.assertFalse(self.run_with(-0.03, 0.02)["rejected"])  # exactly the limit is allowed

    def test_both_failing_says_both(self):
        r = self.run_with(0.01, 0.05)
        self.assertTrue(r["rejected"])
        self.assertIn("not better by more than", r["line"])
        self.assertIn("worse by more than", r["line"])

    def test_the_rule_is_the_strong_sets_own(self):
        import strong_sets as ss  # noqa: PLC0415
        r = self.run_with(-0.01, 0.0)
        self.assertEqual((r["tie"], r["limit"]), (ss.TIE_NATS, ss.LIMIT_B_NATS))
        self.assertEqual((ss.TIE_NATS, ss.LIMIT_B_NATS), (0.002, 0.02))

    def test_no_held_out_set_or_a_missing_prior_is_skipped(self):
        r = screens.run_prior(self.cand, self.base, work=self.tmp / "none")
        self.assertIn("the held-out set (prior-a10/decisions and donors.npz) is not under none", r["skipped"])
        (self.cand / "human_prior.npz").unlink()
        r = screens.run_prior(self.cand, self.base, work=self.work)
        self.assertEqual(r["skipped"], "one of the two has no human_prior.npz")

    def test_the_exam_fifth_is_never_an_input(self):
        import human_prior as hp  # noqa: PLC0415
        self.assertNotEqual(hp.VAL_FIFTH, hp.EXAM_BUCKET)  # the leg is fifth 4; heldout_legs asserts the exam ids absent before it scores

    def test_the_lab_s_candidate_best_players_passes_on_the_real_held_out_set(self):
        import strong_sets as ss  # noqa: PLC0415
        agent = Path(os.environ.get("SPIRELINE_REFEREE", "D:/spireline/referee")) / "agent"
        cand, base = agent / "gen14" / "candidate-best-players", agent / "gen13" / "candidate-stakes"
        if not (cand / "human_prior.npz").is_file() or not (base / "human_prior.npz").is_file() or not (ss.WORK / "prior-a10" / "decisions").is_dir():
            self.skipTest("the lab's candidates and the held-out set are not here")
        self.assertEqual(screens.kind_of(cand, base), "prior")  # it differs from generation 13's candidate in the human prior only
        r = screens.run_prior(cand, base)
        self.assertFalse(r["rejected"], r["line"])
        self.assertLess(r["won"], -0.002)
        self.assertGreater(r["all"], 0.0)
        self.assertLessEqual(r["all"], 0.02)


class PromoteScreenTests(unittest.TestCase):
    """`promote` asks the screens before it plays an arm."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.agents = self.root / "agent"
        self.agents.mkdir()
        (self.root / "seeds").mkdir()
        self.seed_file = self.root / "seeds" / "Etest-8.json"
        seeds, self.cand_rows, self.base_rows = make_block(0.5, n=400, prefix="S")
        self.seed_file.write_text(json.dumps({"seeds": seeds}))
        for name in ("EXAM_DIR", "LOOP_ROOT"):
            patcher = mock.patch.object(fe, name, self.root / "no-gym")
            patcher.start()
            self.addCleanup(patcher.stop)
        self.base = stub_candidate(self.root, "base", "sha-b", generation=3)
        self.cand = stub_candidate(self.root, "cand", "sha-a", generation=4)
        (self.cand / "combat_policy.pt").write_bytes(b"a different fight network")  # the only part that differs

    def tearDown(self):
        self.tmp.cleanup()

    def promote(self, **kw):
        arms = StubArms(self.cand_rows, self.base_rows)
        out = io.StringIO()
        with mock.patch.object(gen_loop, "run_arm", arms), contextlib.redirect_stdout(out):
            doc = gen_loop.promote(self.cand, self.base, self.seed_file, agent_root=self.agents, lab=self.root / "lab", boot=300, review_guard=True,
                                   futility=False, **kw)
        return doc, arms, out.getvalue()

    def reading(self, kind: str, rejected: bool, **extra):
        return {"rule": f"screen:{kind}", "kind": kind, "rejected": rejected, "line": f"screen:{kind} {'REJECTED' if rejected else 'passed'}: figures", **extra}

    def test_a_fight_only_candidate_below_the_incumbent_is_rejected_before_any_arm(self):
        with mock.patch.object(screens, "run_fight", return_value=self.reading("fight", True, cand_surv=40.0, inc_surv=60.0, diff=-20.0, se=3.0)) as rf:
            doc, arms, printed = self.promote()
        self.assertEqual(arms.calls, [])  # not one run was played
        self.assertEqual((doc["decision_text"], doc["n_seeds"]), ("rejected", 0))
        self.assertEqual((doc["decision"]["rule"], doc["decision"]["promote"], doc["decision"]["checks"]), ("screen:fight", False, {"screen": False}))
        self.assertEqual(doc["screen"]["diff"], -20.0)
        self.assertIn("screen:fight REJECTED", printed)
        self.assertEqual(json.loads((self.cand / "promotion.json").read_text())["screen"]["rule"], "screen:fight")
        self.assertFalse((self.agents / "current.json").exists())
        self.assertFalse((self.root / "lab" / "score" / "pending").exists())
        rows = gen_loop.read_promotions(self.agents / "promotions.tsv")
        self.assertEqual([(r["rule"], r["seeds"], r["decision"], r["attempt"], r["cand_sha"]) for r in rows], [("screen:fight", "0", "rejected", "1", "sha-a")])
        self.assertIn("screen:fight REJECTED", rows[0]["guard"])
        self.assertEqual(len(rows[0]), len(gen_loop.TSV_COLUMNS))
        self.assertEqual(rf.call_args[0][0], self.cand)

    def test_the_rejection_spends_the_attempt_as_any_rejection_does(self):
        with mock.patch.object(screens, "run_fight", return_value=self.reading("fight", True, diff=-20.0, se=3.0)):
            self.promote()
        tsv = self.agents / "promotions.tsv"
        self.assertEqual(gen_loop.attempts_before(gen_loop.read_promotions(tsv), 4, "sha-other"), 1)

    def test_a_prior_only_candidate_that_fails_the_screen_is_rejected_with_both_numbers_in_the_rule_column(self):
        (self.cand / "combat_policy.pt").write_bytes(b"x")
        (self.cand / "human_prior.npz").write_bytes(b"new prior")
        (self.base / "human_prior.npz").write_bytes(b"old prior")
        canned = self.reading("prior", True, won=-0.001, all=0.002, ledger_rule="screen:prior won=-0.001 all=+0.002")
        with mock.patch.object(screens, "run_prior", return_value=canned):
            doc, arms, _ = self.promote()
        self.assertEqual((doc["decision_text"], arms.calls, doc["decision"]["rule"]), ("rejected", [], "screen:prior"))
        self.assertEqual(gen_loop.read_promotions(self.agents / "promotions.tsv")[0]["rule"], "screen:prior won=-0.001 all=+0.002")

    def test_a_passed_screen_plays_the_arms_and_keeps_its_figures(self):
        with mock.patch.object(screens, "run_fight", return_value=self.reading("fight", False, diff=1.0, se=3.0)):
            doc, arms, printed = self.promote(sequential=True, smoke=200, batch=100)
        self.assertTrue(arms.calls)  # the arms were played
        self.assertEqual(doc["screen"]["line"], "screen:fight passed: figures")
        row = gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]
        self.assertIn("screen:fight passed: figures", row["guard"])
        self.assertNotEqual(row["rule"], "screen:fight")
        self.assertIn("screen:fight passed", printed)

    def test_a_screen_that_could_not_be_read_or_failed_plays_the_arms(self):
        with mock.patch.object(screens, "run_fight", return_value={"rule": "screen:fight", "kind": "fight", "skipped": "engines differ"}):
            doc, arms, printed = self.promote()
        self.assertTrue(arms.calls)
        self.assertEqual(doc["screen"]["skipped"], "engines differ")
        self.assertIn("screen:fight: not applied (engines differ)", printed)
        shutil_rm = __import__("shutil").rmtree
        shutil_rm(self.agents, True)
        self.agents.mkdir()
        with mock.patch.object(screens, "run_fight", side_effect=OSError("disk gone")):
            doc, arms, _ = self.promote()
        self.assertTrue(arms.calls)
        self.assertEqual(doc["screen"]["error"], "OSError")

    def test_the_switch_and_the_other_parts_differing_mean_no_screen(self):
        with mock.patch.object(screens, "run_fight") as rf:
            doc, arms, _ = self.promote(screen=False)
            self.assertEqual(rf.call_count, 0)
            self.assertTrue(arms.calls)
            self.assertNotIn("screen", doc)
        shutil_rm = __import__("shutil").rmtree
        shutil_rm(self.agents, True)
        self.agents.mkdir()
        (self.cand / "run_value.npz").write_bytes(b"another run value")  # two parts differ now: no screen
        with mock.patch.object(screens, "run_fight") as rf:
            doc, arms, _ = self.promote()
            self.assertEqual(rf.call_count, 0)
            self.assertNotIn("screen", doc)

    def test_the_flags(self):
        import gen0  # noqa: PLC0415
        p = gen0.build_parser()
        self.assertFalse(p.parse_args(["promote", "--cand", "c", "--agent-root", "a"]).no_screen)
        self.assertTrue(p.parse_args(["promote", "--cand", "c", "--agent-root", "a", "--no-screen"]).no_screen)
        self.assertTrue(p.parse_args(["train", "--marker-dir", "m", "--shards-root", "s", "--agent-root", "a", "--no-screen"]).no_screen)
        ran = []
        with mock.patch.object(gen_loop.subprocess, "run", lambda cmd, check=False: ran.append(cmd) or mock.Mock(returncode=0)):
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None)
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None, screen=False)
        self.assertNotIn("--no-screen", ran[0])
        self.assertIn("--no-screen", ran[1])

    def test_the_play_command_is_the_fight_exams_on_the_human_stratum_beside_the_lab_train_line(self):
        ran = []
        with mock.patch.object(gen_loop.subprocess, "run", lambda cmd, check=False: ran.append(cmd) or mock.Mock(returncode=0)):
            rc = gen_loop.fight_exam_play(self.cand, "lab-g4-cand-ab12cd34", 12)
        cmd = ran[0]
        self.assertEqual(rc, 0)
        self.assertEqual(Path(cmd[1]).name, "fight_exam.py")
        self.assertEqual(cmd[2], "play")
        self.assertEqual(cmd[cmd.index("--stratum") + 1], "human")
        self.assertEqual(cmd[cmd.index("--player-dir") + 1], str(self.cand))
        self.assertEqual(cmd[cmd.index("--label") + 1], "lab-g4-cand-ab12cd34")
        self.assertEqual(cmd[cmd.index("--allow-quiet") + 1], "lab-train")  # the judgement's own line does not hold its screen
        self.assertEqual(cmd[cmd.index("--procs") + 1], "12")



# ---------------------------------------------------------------------------------------------------------- screen:student

STUDENT_MACRO = "vrun+pi_s:lambda=0.5:pi_h=2ebfe6d7:pi_s=aaaaaaaa:route=0.2/0.2/0.2"


def sha_of(b: bytes) -> str:
    import hashlib

    return hashlib.sha256(b).hexdigest()


def legs_doc(student: bytes, baseline: bytes, *, kinds=("route",), gain_lo=0.02, won=0.0, allr=0.005, human=("route",), n=500, lam=0.2) -> dict:
    """The stored legs of a student (what `run_student.run_legs` writes), for the kinds given."""
    l1 = {k: {"n": n, "runs": 80, "gain": {"mean": gain_lo + 0.02, "lo95": gain_lo, "hi95": gain_lo + 0.04, "runs": 80}} for k in kinds}
    l2 = {k: {"won": {"n": 900, "runs": 200, "diff": won, "lo95": won - 0.01, "hi95": won + 0.01},
              "all": {"n": 9000, "runs": 2000, "diff": allr, "lo95": allr - 0.01, "hi95": allr + 0.01}} for k in human}
    return {"version": 1, "student_sha256": sha_of(student), "baseline": {"sha256": sha_of(baseline), "kind": "human prior"},
            "leg1": {"kinds": l1}, "leg2": {"measured": True, "kinds": l2, "no_human_fit": [k for k in kinds if k not in human] + ["reward", "select"]},
            "lambda": {"chosen": {k: [lam, lam, lam] for k in kinds}}}


def student_folder(root: Path, name: str, *, student: bytes = b"s1", kinds: dict | None = None, legs: dict | None = None, prior: bytes | None = b"p1",
                   macro: str = STUDENT_MACRO, **kw) -> Path:
    d = folder(root, name, prior=prior, macro=macro, **kw)
    (d / "run_student.npz").write_bytes(student)
    doc = json.loads((d / "agent.json").read_text(encoding="utf-8"))
    doc["student"] = {"file": "run_student.npz", "kinds": kinds if kinds is not None else {"route": ["0.2", "0.2", "0.2"]}}
    (d / "agent.json").write_text(json.dumps(doc), encoding="utf-8")
    if legs is not None:
        (d / "run_student_legs.json").write_text(json.dumps(legs), encoding="utf-8")
    return d


class StudentKindTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))
        self.base = folder(self.tmp, "base", generation=3)

    def test_a_candidate_that_differs_in_the_student_and_the_recipe_naming_it_is_the_student_screen(self):
        c = student_folder(self.tmp, "c")
        self.assertEqual(screens.differing(c, self.base), ["student", "settings"])
        self.assertEqual(screens.kind_of(c, self.base), "student")

    def test_two_students_with_the_same_recipe_differ_in_the_student_alone(self):
        a = student_folder(self.tmp, "a", student=b"s1")
        b = student_folder(self.tmp, "b", student=b"s2", macro=STUDENT_MACRO.replace("aaaaaaaa", "bbbbbbbb"))  # the tag is masked, as the prior's is
        self.assertEqual(screens.differing(b, a), ["student"])
        self.assertEqual(screens.kind_of(b, a), "student")

    def test_a_student_candidate_is_never_read_by_the_prior_or_the_fight_screen(self):
        # the student's file in place of the human prior would fail screen:prior by design (it is fitted to the teacher): no screen at all
        swapped = student_folder(self.tmp, "p", prior=b"the weights of a student in the priors place")
        self.assertEqual(screens.differing(swapped, self.base), ["prior", "student", "settings"])
        self.assertIsNone(screens.kind_of(swapped, self.base))
        self.assertIsNone(screens.kind_of(student_folder(self.tmp, "f", fight=b"f2"), self.base))
        self.assertIsNone(screens.kind_of(student_folder(self.tmp, "v", value=b"v2"), self.base))
        # the same student as an incumbent that has it: nothing differs, nothing to screen
        same = student_folder(self.tmp, "same")
        self.assertIsNone(screens.kind_of(same, student_folder(self.tmp, "twin")))
        # a candidate with no student never gets the student screen, whatever the incumbent has
        self.assertIsNone(screens.kind_of(folder(self.tmp, "plain", generation=5), student_folder(self.tmp, "inc")))
        self.assertEqual(screens.kind_of(folder(self.tmp, "plain2", prior=b"p9"), self.base), "prior")  # and a plain prior difference is as it was

    def test_the_other_screens_are_as_they_were(self):
        self.assertEqual(screens.kind_of(folder(self.tmp, "c", fight=b"f2"), self.base), "fight")


class StudentScreenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))
        self.base = folder(self.tmp, "base", generation=3)  # its human prior is b"p1"

    def cand(self, name="c", **kw):
        legs = kw.pop("legs", None)
        doc = kw.pop("doc", {})
        student = kw.get("student", b"s1")
        return student_folder(self.tmp, name, legs=legs if legs is not None else legs_doc(student, b"p1", **doc), **kw)

    def test_a_student_that_passes_both_legs_is_passed_and_its_figures_are_in_the_line_and_the_ledger_rule(self):
        r = screens.run_student(self.cand(), self.base)
        self.assertFalse(r["rejected"])
        self.assertEqual((r["gain"], r["won"], r["all"]), (0.04, 0.0, 0.005))
        self.assertEqual(r["ledger_rule"], "screen:student gain=+0.040 won=+0.000 all=+0.005")
        self.assertIn("screen:student passed", r["line"])
        self.assertIn("route", r["line"])

    def test_a_gain_whose_lower_bound_is_not_above_zero_is_rejected(self):
        r = screens.run_student(self.cand(doc={"gain_lo": -0.001}), self.base)
        self.assertTrue(r["rejected"])
        self.assertEqual(r["failed"], ["route"])
        self.assertIn("leg 1", r["line"])
        self.assertIn("REJECTED", r["line"])

    def test_the_human_bar_is_the_second_leg_with_its_two_margins(self):
        self.assertFalse(screens.run_student(self.cand("a", doc={"won": 0.0015, "allr": 0.019}), self.base)["rejected"])
        r = screens.run_student(self.cand("b", doc={"won": 0.0021}), self.base)
        self.assertTrue(r["rejected"])
        self.assertIn("leg 2", r["line"])
        self.assertTrue(screens.run_student(self.cand("c", doc={"allr": 0.0201}), self.base)["rejected"])

    def test_a_kind_with_no_human_fit_is_read_on_leg_1_alone(self):
        c = self.cand(kinds={"reward": ["0.3", "0.3", "0.3"]}, doc={"kinds": ("reward",), "human": (), "won": 9.0, "allr": 9.0})
        r = screens.run_student(c, self.base)
        self.assertFalse(r["rejected"])
        self.assertEqual((r["won"], r["all"]), (None, None))
        self.assertIn("n/a", r["line"])

    def test_one_failing_kind_rejects_the_candidate_and_the_line_names_it(self):
        legs = legs_doc(b"s1", b"p1", kinds=("route", "reward"), human=("route",))
        legs["leg1"]["kinds"]["reward"]["gain"]["lo95"] = -0.5
        c = self.cand(kinds={"route": ["0.2"] * 3, "reward": ["0.3"] * 3}, legs=legs)
        r = screens.run_student(c, self.base)
        self.assertTrue(r["rejected"])
        self.assertEqual(r["failed"], ["reward"])
        self.assertIn("reward: leg 1", r["line"])

    def test_a_kind_the_legs_do_not_cover_is_a_rejection_not_a_pass(self):
        c = self.cand(kinds={"route": ["0.2"] * 3, "shop": ["0.1"] * 3})
        r = screens.run_student(c, self.base)
        self.assertTrue(r["rejected"])
        self.assertEqual(r["failed"], ["shop"])

    def test_a_screen_that_cannot_be_read_says_why(self):
        c = student_folder(self.tmp, "nolegs")
        self.assertIn("no readable run_student_legs.json", screens.run_student(c, self.base)["skipped"])
        other = student_folder(self.tmp, "other", student=b"s9", legs=legs_doc(b"s1", b"p1"))
        self.assertIn("not for this student's weights", screens.run_student(other, self.base)["skipped"])
        wrong = student_folder(self.tmp, "wrong", legs=legs_doc(b"s1", b"another prior"))
        self.assertIn("another baseline", screens.run_student(wrong, self.base)["skipped"])
        nokinds = student_folder(self.tmp, "nokinds", kinds={}, legs=legs_doc(b"s1", b"p1"))
        self.assertIn("agent.json names no student kinds", screens.run_student(nokinds, self.base)["skipped"])
        self.assertIn("agent.json cannot be read", screens.run_student(self.tmp / "nope", self.base)["skipped"])

    def test_an_incumbent_with_a_student_is_the_baseline_the_legs_must_name(self):
        inc = student_folder(self.tmp, "inc", student=b"s0", macro=STUDENT_MACRO.replace("aaaaaaaa", "cccccccc"))
        against_student = student_folder(self.tmp, "n1", student=b"s1", legs=legs_doc(b"s1", b"s0"))
        self.assertFalse(screens.run_student(against_student, inc)["rejected"])
        against_prior = student_folder(self.tmp, "n2", student=b"s1", legs=legs_doc(b"s1", b"p1"))  # measured against the prior: not this incumbent
        self.assertIn("another baseline", screens.run_student(against_prior, inc)["skipped"])

    def test_the_rule_names_are_distinct(self):
        self.assertEqual(screens.SCREEN_STUDENT, "screen:student")
        self.assertNotIn(screens.SCREEN_STUDENT, (screens.SCREEN_FIGHT, screens.SCREEN_PRIOR))

    def test_the_run_screen_of_the_loop_reads_the_student_and_names_its_rule_when_it_fails(self):
        c = self.cand()
        with mock.patch.object(screens, "run_student", side_effect=RuntimeError("boom")):
            got = gen_loop.run_screen(c, self.base, None, None, say=lambda m: None)
        self.assertEqual((got["rule"], got["kind"], got["error"]), ("screen:student", "student", "RuntimeError"))
        got = gen_loop.run_screen(c, self.base, None, None, say=lambda m: None)
        self.assertEqual("screen:student", got["rule"])
        self.assertFalse(got["rejected"])


class StudentPromoteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.agents = self.root / "agent"
        self.agents.mkdir()
        (self.root / "seeds").mkdir()
        self.seed_file = self.root / "seeds" / "Etest-8.json"
        seeds, self.cand_rows, self.base_rows = make_block(0.5, n=400, prefix="S")
        self.seed_file.write_text(json.dumps({"seeds": seeds}))
        for name in ("EXAM_DIR", "LOOP_ROOT"):
            patcher = mock.patch.object(fe, name, self.root / "no-gym")
            patcher.start()
            self.addCleanup(patcher.stop)
        self.base = stub_candidate(self.root, "base", "sha-b", generation=3)
        self.cand = stub_candidate(self.root, "cand", "sha-a", generation=4)
        (self.cand / "run_student.npz").write_bytes(b"a student")  # the only part that differs

    def tearDown(self):
        self.tmp.cleanup()

    def promote(self):
        arms = StubArms(self.cand_rows, self.base_rows)
        with mock.patch.object(gen_loop, "run_arm", arms), contextlib.redirect_stdout(io.StringIO()):
            doc = gen_loop.promote(self.cand, self.base, self.seed_file, agent_root=self.agents, lab=self.root / "lab", boot=300, review_guard=True,
                                   futility=False)
        return doc, arms

    def test_a_student_that_fails_its_screen_is_rejected_before_any_arm_with_its_rule_in_the_ledger(self):
        canned = {"rule": "screen:student", "kind": "student", "rejected": True, "gain": -0.01, "won": 0.01, "all": 0.03,
                  "line": "screen:student REJECTED: figures", "ledger_rule": "screen:student gain=-0.010 won=+0.010 all=+0.030"}
        with mock.patch.object(screens, "run_student", return_value=canned), mock.patch.object(screens, "run_prior") as rp, \
                mock.patch.object(screens, "run_fight") as rf:
            doc, arms = self.promote()
        self.assertEqual((arms.calls, doc["decision"]["rule"], doc["decision_text"]), ([], "screen:student", "rejected"))
        rp.assert_not_called()  # the prior's screen never reads a student candidate
        rf.assert_not_called()
        row = gen_loop.read_promotions(self.agents / "promotions.tsv")[0]
        self.assertEqual(row["rule"], "screen:student gain=-0.010 won=+0.010 all=+0.030")
        self.assertEqual((row["seeds"], row["decision"]), ("0", "rejected"))

    def test_a_student_that_passes_plays_the_arms(self):
        canned = {"rule": "screen:student", "kind": "student", "rejected": False, "line": "screen:student passed: figures", "ledger_rule": "x"}
        with mock.patch.object(screens, "run_student", return_value=canned):
            doc, arms = self.promote()
        self.assertTrue(arms.calls)
        self.assertEqual(doc["screen"]["line"], "screen:student passed: figures")


if __name__ == "__main__":
    unittest.main()
