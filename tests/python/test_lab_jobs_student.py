"""tools/lab/jobs.py and runner.py: the `student` kind and the run loop's stops (D56, D57 part 3, D58 C4; docs/lab/RUN_LOOP.md).

Scratch lab and agent folders, stand-in commands (`python -c`), a stand-in clock: no game, no emulator, no GPU, no git. The chain is driven through the
runner's real step machinery: the teacher job leaves shards, the student job is due at 10,000 new labelled screens, builds a candidate and its marker,
the train job judges it and a promotion rewrites current.json, which the next teacher block reads.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.lab import alerts as al  # noqa: E402
from tools.lab import cores  # noqa: E402
from tools.lab import jobs as jb  # noqa: E402
from tools.lab import plans as pl  # noqa: E402
from tools.lab import runner as rn  # noqa: E402

NOW = time.time()
DAY = 86400.0
ALL_GUARDS = ("halt", "user", "pause", "lock", "load", "disk", "parent")
# stand-in `check`: counts the lines of every decisions.jsonl under the root and prints the JSON `run_student.py check` prints; counts its own calls
CHECK = ["{python}", "-c",
         "import sys, json, pathlib; r = pathlib.Path(sys.argv[1]); open(sys.argv[2], 'a').write('x'); "
         "print(json.dumps({'decisions': sum(len(f.read_text().splitlines()) for f in r.rglob('decisions.jsonl'))}))", "{root}", "CALLS"]


def iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


class Base(unittest.TestCase):
    def setUp(self):
        global NOW
        NOW = time.time()
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ref = self.tmp / "ref"
        self.lab = self.ref / "LAB"
        self.lab.mkdir(parents=True)
        self.store = self.tmp / "store"
        self.agents = self.tmp / "agent"
        self.calls = self.tmp / "check-calls.txt"
        git = mock.patch.object(jb, "_git_main", return_value=None)
        git.start()
        self.addCleanup(git.stop)
        jb._count_cache.clear()
        self.addCleanup(jb._count_cache.clear)
        self.promote(13)

    # ---- the agent folder: current.json and a generation with a human prior (or a student)

    def promote(self, gen: int, *, student: bool = False) -> Path:
        d = self.agents / f"gen{gen}" / "candidate-x"
        d.mkdir(parents=True, exist_ok=True)
        meta: dict = {"generation": gen, "macro": "vrun+pi_h:lambda=0.5:pi_h=aaaaaaaa", "human_prior": {"file": "human_prior.npz", "lambda": "0.5"}}
        (d / "human_prior.npz").write_bytes(b"prior")
        if student:
            meta["student"] = {"file": "run_student.npz"}
            (d / "run_student.npz").write_bytes(b"student")
        (d / "agent.json").write_text(json.dumps(meta), encoding="utf-8")
        (self.agents / "current.json").write_text(json.dumps({"generation": gen, "dir": str(d).replace("/", "\\"), "promoted": iso(NOW)}), encoding="utf-8")
        return d

    # ---- the store: shards under <store>/<generation>/shards/<job> (the daily job) and <store>/gen13-r0/<generation>/shards/<job> (round 0)

    def shard(self, n: int, job: str = "j1", round0: bool = False, gen: str = "13") -> Path:
        d = (self.store / "gen13-r0" / gen if round0 else self.store / gen) / "shards" / job
        d.mkdir(parents=True, exist_ok=True)
        (d / "decisions.jsonl").write_text("".join(f'{{"i": {i}}}\n' for i in range(n)), encoding="utf-8")
        (d / "manifest.json").write_text(json.dumps({"counts": {"decisions": n}}), encoding="utf-8")
        return d

    def config(self, **student):
        cfg = {"enabled": True, "shards_root": str(self.store), "agent_root": str(self.agents),
               "check_cmd": [*CHECK[:-1], str(self.calls)]}
        cfg.update(student)
        doc = {"student": cfg, "teacher": {"enabled": False, "store": str(self.store)}, "train": {"enabled": True}}
        (self.lab / "jobs.json").write_text(json.dumps(doc), encoding="utf-8")
        return jb.load_config(self.lab)["student"]

    def n_checks(self) -> int:
        return len(self.calls.read_text()) if self.calls.exists() else 0

    def state(self, **over):
        st = jb.read_student_state(self.lab)
        st.update(over)
        jb.write_student_state(self.lab, st)
        return st

    def hold(self, cfg=None, **kw):
        cfg = cfg or self.config()
        kw.setdefault("now", NOW)
        kw.setdefault("quiet_path", self.ref / "QUIET.lock")
        return jb.student_hold(self.lab, cfg, **kw)


class KindTests(Base):
    def test_it_is_a_kind_off_until_jobs_json_says_so_between_train_and_the_fights(self):
        self.assertIn("student", jb.NEW_KINDS)
        self.assertIn("student", pl.PLANS)
        self.assertIn("student", rn.JOB_TYPES)
        self.assertGreater(jb.PRIORITY["student"], jb.PRIORITY["train"])
        self.assertLess(jb.PRIORITY["student"], jb.PRIORITY["fights-aimed"])
        d = jb.DEFAULTS["student"]
        self.assertFalse(d["enabled"])
        self.assertEqual(10000, d["new_screens"])
        self.assertEqual(4, cores.job_cores("student", None, None))
        self.assertIn("is off", jb.unready("student", self.lab, jb.config_from({})["student"]))
        self.assertEqual(("selfplay", "train", "score-real"), jb.LOOP_KINDS)  # the old loop's share is unchanged

    def test_the_kind_lists_are_lane_3s(self):
        import macro_recipe  # noqa: PLC0415
        import run_student  # noqa: PLC0415

        self.assertEqual(macro_recipe.STUDENT_KINDS, jb.STUDENT_KINDS)
        self.assertEqual(run_student.HUMAN_KINDS, jb.HUMAN_BAR_KINDS)

    def test_the_default_command_runs_the_job_wrapper_in_the_gpu_python_with_the_incumbents_tokens(self):
        cmd = jb.DEFAULTS["student"]["cmd"]
        self.assertTrue(cmd[0].endswith("venv-gpu/Scripts/python.exe"))
        self.assertEqual("tools/lab/student_job.py", cmd[1])
        for token in ("{student_prior}", "{student_target}", "{student_tag}", "{lab}"):
            self.assertIn(token, cmd)
        self.assertIn("{job}/result.json", cmd)
        self.assertEqual([0, 3], jb.DEFAULTS["student"]["ok_rcs"])
        self.assertIn("{tree}/tools/lab/student_job.py", jb.DEFAULTS["student"]["needs"])


class TriggerTests(Base):
    def test_nothing_is_due_until_ten_thousand_new_screens_exist(self):
        self.shard(4000)
        self.shard(5999, "j2", round0=True)
        why = self.hold()
        self.assertIn("9999 new screens recorded since the last fit; the student wants 10000", why)
        self.assertEqual(0, self.n_checks())  # the cheap count (the manifests) came first: `check` was not asked
        self.shard(1, "j3")
        self.assertIsNone(self.hold())
        self.assertEqual(1, self.n_checks())  # then `check` counted what the loader keeps

    def test_the_count_is_check_over_the_whole_store_both_layouts_and_a_stale_manifest_is_not_believed(self):
        self.shard(6000)
        self.shard(6000, "j2", round0=True)
        d = self.shard(6000, "j3")
        (d / "decisions.jsonl").write_text("".join('{"i": 0}\n' for _ in range(100)), encoding="utf-8")  # the manifest says 6,000, the shard has 100
        rec, _ = jb.student_recorded_screens(self.store)
        self.assertEqual(18000, rec)  # the cheap count reads the manifests of both layouts
        self.assertIsNone(self.hold())  # ... and `check` says what is really there, 12,100: due
        (self.store / "13" / "shards" / "j1" / "decisions.jsonl").write_text("".join('{"i": 0}\n' for _ in range(1000)), encoding="utf-8")
        jb._count_cache.clear()
        self.assertIn("7100 new labelled screens since the last fit (check)", self.hold())  # manifests still say 18,000: only `check` knows

    def test_check_below_the_threshold_says_so_and_is_not_asked_again_until_a_shard_changes(self):
        d = self.shard(10000)
        d.joinpath("decisions.jsonl").write_text("".join('{"i": 0}\n' for _ in range(7000)), encoding="utf-8")  # 3,000 recorded screens the loader drops
        why = self.hold()
        self.assertIn("7000 new labelled screens since the last fit (check); the student wants 10000", why)
        self.assertEqual(1, self.n_checks())
        self.hold()
        self.hold()
        self.assertEqual(1, self.n_checks())  # cached by the shards' sizes and times
        time.sleep(0.02)
        d.joinpath("decisions.jsonl").write_text("".join('{"i": 0}\n' for _ in range(10050)), encoding="utf-8")
        self.assertIsNone(self.hold())
        self.assertEqual(2, self.n_checks())

    def test_a_fit_resets_the_count_and_the_next_fit_wants_ten_thousand_more(self):
        self.shard(12000)
        self.assertIsNone(self.hold())
        self.state(last_fit={"t": NOW, "screens": 12000, "recorded": 12000, "round": 1})
        self.assertIn("0 new screens recorded", self.hold())
        self.shard(9000, "j2")
        self.assertIn("9000 new screens recorded", self.hold())
        self.shard(1000, "j3")
        self.assertIsNone(self.hold())

    def test_the_weekly_fit_needs_the_week_and_a_thousand_new_screens(self):
        self.state(last_fit={"t": NOW - 6 * DAY, "screens": 20000, "recorded": 20000, "round": 1})
        self.shard(20000 + 5000)
        self.assertIn("wants 10000", self.hold())  # five thousand new, the week not yet gone
        self.state(last_fit={"t": NOW - 8 * DAY, "screens": 20000, "recorded": 20000, "round": 1})
        self.assertIsNone(self.hold())  # the week is gone and 5,000 are new
        (self.store / "13" / "shards" / "j1" / "decisions.jsonl").write_text("".join('{"i": 0}\n' for _ in range(20999)), encoding="utf-8")
        (self.store / "13" / "shards" / "j1" / "manifest.json").write_text(json.dumps({"counts": {"decisions": 20999}}), encoding="utf-8")
        self.assertIn("wants 1000", self.hold())  # 999 new: the weekly floor is a thousand
        self.assertIn("the weekly fit is due", self.hold())

    def test_the_weekly_clock_of_a_loop_that_never_fitted_starts_with_the_loop(self):
        self.shard(3000)
        self.state(loop_start=NOW - DAY)
        self.assertIn("wants 10000", self.hold())
        self.state(loop_start=NOW - 8 * DAY)
        self.assertIsNone(self.hold())

    def test_a_check_that_fails_holds_the_job_with_its_reason(self):
        self.shard(12000)
        cfg = self.config(check_cmd=["{python}", "-c", "import sys; sys.stderr.write('ShardError: a bad line'); sys.exit(2)"])
        why = self.hold(cfg)
        self.assertIn("cannot count the labelled screens: the check failed (rc 2): ShardError: a bad line", why)
        cfg = self.config(check_cmd=["{python}", "-c", "print('no json')"])
        jb._count_cache.clear()
        self.assertIn("the check printed no count", self.hold(cfg))

    def test_the_default_check_is_run_student_check_over_the_root(self):
        cmd = jb.student_check_cmd({}, "R")
        self.assertEqual(["check", "--shards-root", "R"], cmd[2:])
        self.assertTrue(cmd[1].replace("\\", "/").endswith("rl/run_student.py"))

    def test_the_store_walk_is_two_globs_not_a_recursive_walk(self):
        self.shard(10)
        (self.store / "13" / "decisions.d").mkdir()
        (self.store / "13" / "decisions.d" / "decisions.jsonl").write_text("{}\n", encoding="utf-8")  # a per-run folder: not a shard
        self.assertEqual(1, len(jb.student_shard_dirs(self.store)))


class HoldTests(Base):
    def ready(self):
        self.shard(12000)
        self.assertIsNone(self.hold())

    def test_a_ready_lab_lets_it_start_and_unready_is_the_hold(self):
        self.ready()
        cfg = self.config()
        self.assertIsNone(jb.unready("student", self.lab, cfg))

    def test_halt_pause_and_the_night(self):
        self.ready()
        (self.lab / "HALT").write_text("x")
        self.assertEqual("LAB/HALT is up", self.hold())
        (self.lab / "HALT").unlink()
        (self.lab / "PAUSE").write_text("x")
        self.assertEqual("LAB/PAUSE is up", self.hold())
        (self.lab / "PAUSE").unlink()
        (self.lab / "nightly").mkdir()
        (self.lab / "nightly" / "active").write_text("x")
        self.assertIn("a nightly block is running", self.hold())

    def test_a_live_quiet_line_holds_it_but_the_loops_and_the_daytime_scores_do_not(self):
        self.ready()
        q = self.ref / "QUIET.lock"
        for tag in ("lab-train", "judge-corpus", "anything"):
            q.write_text(f"{tag} {iso(NOW - 60)} pid=7\n")
            self.assertEqual(f"QUIET.lock is held by {tag}", self.hold(), tag)
        q.write_text(f"gym-loop {iso(NOW - 60)} pid=7\nlab-score {iso(NOW - 60)} pid=8\n")
        self.assertIsNone(self.hold())

    def test_judgement_first_a_waiting_train_marker_holds_it(self):
        self.ready()
        (self.lab / "train").mkdir()
        (self.lab / "train" / "ready-13-g14-student").write_text("x")
        self.assertEqual("judgement first: ready-13-g14-student is waiting", self.hold())

    def test_the_run_student_alert_holds_it_until_the_file_is_removed(self):
        self.ready()
        al.write_run_student(self.lab, [], now=NOW)
        self.assertIn("stopped: every kind failed the human bar twice running (LAB/alerts/run-student.txt)", self.hold())
        al.alert_path(self.lab, "run-student").unlink()
        self.assertIsNone(self.hold())

    def test_an_incumbent_it_cannot_start_from_is_said(self):
        self.shard(12000)
        (self.agents / "current.json").unlink()
        self.assertEqual("agent/current.json is not readable", self.hold())
        d = self.promote(13)
        (d / "agent.json").write_text(json.dumps({"generation": 13, "macro": "vrun"}), encoding="utf-8")
        self.assertIn("plays no human prior or student", self.hold())
        (d / "agent.json").write_text(json.dumps({"generation": 13, "human_prior": {"file": "nope.npz"}}), encoding="utf-8")
        self.assertIn("nope.npz is missing in candidate-x", self.hold())


class ParamsTests(Base):
    def test_the_job_is_made_for_the_next_generation_from_the_incumbents_prior(self):
        self.shard(12000)
        cfg = self.config()
        p = jb.params_for("student", self.lab, cfg)
        self.assertEqual(("student", "13", 14, "student", "ready-13-g14-student"), (p["strategy"], p["generation"], p["candidate_generation"], p["tag"], p["marker"]))
        self.assertEqual([0, 3], p["ok_rcs"])
        self.assertEqual((12000, 12000, "human prior"), (p["screens"], p["recorded"], p["prior_kind"]))
        cmd = p["cmd"]
        self.assertEqual("14", cmd[cmd.index("--target") + 1])
        self.assertEqual("student", cmd[cmd.index("--tag") + 1])
        self.assertTrue(cmd[cmd.index("--prior") + 1].endswith("candidate-x/human_prior.npz"))
        self.assertFalse(any("{student_" in t for t in cmd))
        self.assertIn("{job}/student", cmd)  # the runner's own tokens are left for it

    def test_an_incumbent_that_has_a_student_is_the_prior_the_next_fit_starts_from(self):
        self.promote(14, student=True)
        self.shard(12000)
        p = jb.params_for("student", self.lab, self.config())
        self.assertEqual(("14", 15, "student"), (p["generation"], p["candidate_generation"], p["prior_kind"]))
        self.assertTrue(p["cmd"][p["cmd"].index("--prior") + 1].endswith("run_student.npz"))

    def test_a_tag_is_never_used_twice_at_a_target(self):
        self.shard(12000)
        cfg = self.config()
        self.assertEqual("student", jb.params_for("student", self.lab, cfg)["tag"])
        (self.agents / "gen14" / "candidate-student").mkdir(parents=True)
        self.assertEqual("student-r2", jb.params_for("student", self.lab, cfg)["tag"])
        (self.lab / "train").mkdir()
        (self.lab / "train" / "done-13-g14-student-r2").write_text("x")
        self.assertEqual("student-r3", jb.params_for("student", self.lab, cfg)["tag"])
        self.assertEqual("ready-13-g14-student-r3", jb.params_for("student", self.lab, cfg)["marker"])
        # ... and the train job reads the marker as a tagged target
        (self.lab / "train" / "ready-13-g14-student-r3").write_text("x")
        t = jb.params_for("train", self.lab, jb.load_config(self.lab)["train"])
        self.assertEqual(("13", 14, "student-r3"), (t["generation"], t["candidate_generation"], t["tag"]))

    def test_a_marker_the_student_writes_sorts_with_the_other_target_markers(self):
        names = ["ready-13-g14-student", "ready-13-g14-student-r2", "ready-13-g14-imit"]
        for n in names:
            self.assertEqual((13, 1, 14), jb.gen_key(n)[:3])


def legs(card=True, rest=True, route=None, **kw):
    """The verdict block of a round: leg 2 True/False per kind with a human fit, None for kinds without."""
    out = {}
    for k in jb.STUDENT_KINDS:
        l2 = kw.get(k, {"card": card, "rest": rest, "route": route}.get(k, None))
        out[k] = {"pass": l2 is not False, "leg1": True, "leg2": l2, "lambda": [0.1, 0.1, 0.1], "why": "" if l2 is not False else "leg 2: won +0.0100 (at most +0.002)"}
    return out


class FinishTests(Base):
    def job(self, verdict, built=True, decisions=12000, tag="student", target=14, name="job1", **extra):
        d = self.lab / "runs" / name
        d.mkdir(parents=True, exist_ok=True)
        cand = f"gen{target}/candidate-{tag}"
        if built:
            (self.agents / cand).mkdir(parents=True, exist_ok=True)
            (self.agents / cand / "agent.json").write_text("{}", encoding="utf-8")
        res = {"built": built, "candidate": cand if built else None, "decisions": decisions, "verdict": verdict,
               "kinds_passing": {k: [0.1] for k, v in verdict.items() if v["pass"]} if built else {}}
        res.update(extra)
        (d / "result.json").write_text(json.dumps(res), encoding="utf-8")
        params = {"generation": "13", "candidate_generation": target, "tag": tag, "marker": f"ready-13-g{target}-{tag}", "screens": decisions,
                  "recorded": decisions + 500, "prior_kind": "human prior"}
        self.config()
        return d, params

    def test_a_built_candidate_gets_its_marker_and_the_round_is_recorded(self):
        d, params = self.job(legs())
        self.assertIsNone(jb.finish("student", self.lab, d, params))
        marker = self.lab / "train" / "ready-13-g14-student"
        self.assertTrue(marker.exists())
        st = jb.read_student_state(self.lab)
        self.assertEqual((1, 12000, 12500, "student"), (st["round_count"], st["last_fit"]["screens"], st["last_fit"]["recorded"], st["last_fit"]["tag"]))
        r = st["rounds"][0]
        self.assertEqual(("ready-13-g14-student", "gen14/candidate-student", None), (r["marker"], r["candidate"], r["judged"]))
        text = (self.lab / "student" / "verdicts.md").read_text(encoding="utf-8")
        self.assertIn("candidate gen14/candidate-student, marker ready-13-g14-student", text)
        self.assertIn("| card | pass | pass | yes |", text)
        self.assertIn("| shop | pass | no human fit | yes |", text)

    def test_a_round_that_built_nothing_writes_no_marker_but_still_counts_as_the_fit(self):
        d, params = self.job(legs(card=False, rest=False), built=False)
        self.assertIsNone(jb.finish("student", self.lab, d, params))
        self.assertFalse((self.lab / "train").exists())
        st = jb.read_student_state(self.lab)
        self.assertEqual((12000, False), (st["last_fit"]["screens"], st["last_fit"]["built"]))  # the next fit wants 10,000 more, not the same screens again
        self.assertIn("no candidate (no kind passed", (self.lab / "student" / "verdicts.md").read_text(encoding="utf-8"))

    def test_a_marker_is_not_written_for_a_candidate_that_is_not_there(self):
        d, params = self.job(legs())
        shutil.rmtree(self.agents / "gen14")
        jb.finish("student", self.lab, d, params)
        self.assertFalse((self.lab / "train" / "ready-13-g14-student").exists())

    def test_the_run_level_bar_the_job_wrote_is_beside_the_verdict(self):
        d, params = self.job(legs())
        (d / "run-bar.md").write_text("run-level human bar: THE TABLE\n", encoding="utf-8")
        jb.finish("student", self.lab, d, params)
        self.assertIn("THE TABLE", (self.lab / "student" / "verdicts.md").read_text(encoding="utf-8"))
        d2, p2 = self.job(legs(), name="job2", tag="student-r2")
        jb.finish("student", self.lab, d2, p2)
        self.assertIn("no recorded runs of the copy yet", (self.lab / "student" / "verdicts.md").read_text(encoding="utf-8"))

    def test_bookkeeping_never_fails_the_job(self):
        d, params = self.job(legs())
        (d / "result.json").write_text("not json", encoding="utf-8")
        self.assertIsNone(jb.finish("student", self.lab, d, params))  # an empty result is a round with no verdict
        with mock.patch.object(jb, "read_student_state", side_effect=RuntimeError("boom")):
            self.assertEqual("student: bookkeeping failed (RuntimeError)", jb.finish("student", self.lab, d, params))

    # ---- the stops

    def rounds(self, *verdicts):
        for i, v in enumerate(verdicts):
            d, params = self.job(v, name=f"job{i}", tag="student" if i == 0 else f"student-r{i + 1}", built=any(x["pass"] for x in v.values()))
            jb.student_finish(self.lab, d, params, now=NOW + i)

    def test_a_kind_that_fails_the_human_bar_twice_running_is_frozen_and_the_file_says_where(self):
        self.rounds(legs(card=False))
        st = jb.read_student_state(self.lab)
        self.assertEqual(1, st["fails"]["card"])
        self.assertEqual({}, jb.read_frozen(self.lab))
        self.assertFalse((self.lab / "student" / "frozen.json").exists())
        self.rounds(legs(card=False))
        self.assertEqual(["card"], sorted(jb.read_frozen(self.lab)))
        doc = json.loads((self.lab / "student" / "frozen.json").read_text(encoding="utf-8"))
        self.assertEqual(2, doc["kinds"]["card"]["failures"])
        self.assertIn("Delete its entry", doc["_note"])
        self.assertIn("card: frozen after 2 rounds failing the human bar", (self.lab / "student" / "verdicts.md").read_text(encoding="utf-8"))

    def test_a_pass_in_between_starts_the_count_again_and_other_kinds_are_not_frozen(self):
        self.rounds(legs(card=False), legs(card=True), legs(card=False))
        self.assertEqual({}, jb.read_frozen(self.lab))
        self.assertEqual(1, jb.read_student_state(self.lab)["fails"]["card"])
        self.assertEqual(0, jb.read_student_state(self.lab)["fails"]["rest"])

    def test_a_kind_without_a_human_fit_never_fails_the_bar_and_an_unmeasured_round_keeps_the_count(self):
        self.rounds(legs(card=False, rest=None), legs(card=None, rest=None, route=True))
        st = jb.read_student_state(self.lab)
        self.assertEqual(1, st["fails"]["card"])  # the second round did not measure it: the count stays
        self.assertEqual({}, jb.read_frozen(self.lab))
        self.assertNotIn("reward", st["fails"])
        self.assertNotIn("select", st["fails"])

    def test_a_human_bar_that_was_not_measured_is_no_failure_of_the_student(self):
        """Lane 3 reports an unmeasured human bar as leg 2 False; the job marks it `leg2_measured` False, and no stop counts it."""
        def unmeasured():
            v = legs(card=False, rest=False, route=False)
            for k in ("card", "rest", "route"):
                v[k]["leg2_measured"] = False
            return v
        self.rounds(unmeasured(), unmeasured(), unmeasured())
        st = jb.read_student_state(self.lab)
        self.assertEqual(({}, 0, False), (jb.read_frozen(self.lab), st["all_fail"], al.alert_path(self.lab, "run-student").exists()))
        self.assertEqual({}, {k: n for k, n in st["fails"].items() if n})
        self.assertIn("| card | pass | not measured |", (self.lab / "student" / "verdicts.md").read_text(encoding="utf-8"))
        v = unmeasured()
        v["card"]["leg2_measured"] = True  # measured, and failed
        self.rounds(v)
        self.assertEqual(1, jb.read_student_state(self.lab)["fails"]["card"])

    def test_the_orchestrator_unfreezes_by_deleting_the_entry_and_the_count_starts_again(self):
        self.rounds(legs(card=False), legs(card=False))
        self.assertIn("card", jb.read_frozen(self.lab))
        jb.write_frozen(self.lab, {})
        self.rounds(legs(card=False))
        self.assertEqual({}, jb.read_frozen(self.lab))  # one failure after the unfreeze is the first, not a third
        self.assertEqual(1, jb.read_student_state(self.lab)["fails"]["card"])
        self.rounds(legs(card=False))
        self.assertIn("card", jb.read_frozen(self.lab))

    def test_a_frozen_kind_is_not_counted_again(self):
        self.rounds(legs(card=False), legs(card=False), legs(card=False))
        st = jb.read_student_state(self.lab)
        self.assertEqual(2, st["fails"]["card"])  # frozen at two; the wrapper leaves it out, a stale verdict for it does not add to it

    def test_every_kind_failing_twice_running_raises_run_student_and_the_runner_stops_picking_it(self):
        bad = legs(card=False, rest=False, route=False)
        self.rounds(bad)
        self.assertFalse(al.alert_path(self.lab, "run-student").exists())
        self.rounds(bad)
        path = al.alert_path(self.lab, "run-student")
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("ALERT run-student: every kind failed the human bar in 2 student rounds running", text)
        self.assertIn("card: leg 1 True, leg 2 False: leg 2: won +0.0100", text)
        self.assertIn("round 2 (student, ", text)
        self.shard(30000)
        self.assertIn("stopped: every kind failed the human bar twice running", self.hold())

    def test_a_round_where_one_kind_passes_resets_the_all_fail_count(self):
        bad = legs(card=False, rest=False, route=False)
        self.rounds(bad, legs(card=True, rest=False, route=False), bad)
        self.assertFalse(al.alert_path(self.lab, "run-student").exists())
        self.assertEqual(1, jb.read_student_state(self.lab)["all_fail"])

    def test_removing_the_alert_by_hand_lets_the_student_run_and_starts_the_count_over(self):
        self.config(enabled=True)
        bad = legs(card=False, rest=False, route=False)
        self.rounds(bad, bad)
        self.assertEqual(2, jb.read_student_state(self.lab)["all_fail"])
        al.alert_path(self.lab, "run-student").unlink()
        msgs = jb.run_loop_tick(self.lab, NOW + 10)
        self.assertTrue(any("run-student.txt removed by hand" in m for m in msgs))
        st = jb.read_student_state(self.lab)
        self.assertEqual((0, False), (st["all_fail"], st["run_student_raised"]))
        self.rounds(bad)  # one failing round after it is the first, not a third
        self.assertFalse(al.alert_path(self.lab, "run-student").exists())

    def test_only_student_candidates_are_the_loops_to_judge(self):
        """A train job of another recipe is none of the loop's business: its finish writes nothing under LAB/student."""
        (self.lab / "train").mkdir()
        (self.lab / "train" / "ready-11-g14-stakes").write_text("x")
        jb.finish("train", self.lab, self.tmp, {"marker": "ready-11-g14-stakes", "tag": "stakes", "candidate_generation": 14})
        self.assertTrue((self.lab / "train" / "done-11-g14-stakes").exists())
        self.assertFalse((self.lab / "student").exists())


class JudgedTests(Base):
    def judged(self, promote: bool):
        d, params = FinishTests.job(self, legs())
        jb.finish("student", self.lab, d, params)
        cdir = self.agents / "gen14" / "candidate-student"
        (cdir / "promotion.json").write_text(json.dumps({"decision": {"promote": promote, "rule": "keep-v2"}, "decision_text": "promoted" if promote else "rejected"}),
                                             encoding="utf-8")
        arms = cdir / "arms" / "E14-a3"
        arms.mkdir(parents=True)
        (arms / "runs.jsonl").write_text("".join(json.dumps({"floor": 25, "win": False, "boss_damage_acts": [[0, 1.0]]}) + "\n" for _ in range(30)),
                                         encoding="utf-8")
        (self.lab / "train").mkdir(exist_ok=True)
        t = jb.params_for("train", self.lab, jb.load_config(self.lab)["train"])
        return jb.finish("train", self.lab, self.tmp, t)

    def test_a_promotion_is_the_loops_clock_and_the_verdict_is_written_beside_the_figures(self):
        self.assertIsNone(self.judged(True))
        st = jb.read_student_state(self.lab)
        self.assertEqual("promoted", st["rounds"][-1]["judged"])
        self.assertAlmostEqual(time.time(), st["last_promotion"], delta=60)
        text = (self.lab / "student" / "verdicts.md").read_text(encoding="utf-8")
        self.assertIn("judgement of ready-13-g14-student: PROMOTED (rule keep-v2)", text)
        self.assertIn("| figure | humans | candidate |", text)
        self.assertTrue((self.lab / "train" / "done-13-g14-student").exists())

    def test_a_rejection_does_not_move_the_clock(self):
        self.assertIsNone(self.judged(False))
        st = jb.read_student_state(self.lab)
        self.assertEqual(("rejected", None), (st["rounds"][-1]["judged"], st["last_promotion"]))
        self.assertIn("not promoted", (self.lab / "student" / "verdicts.md").read_text(encoding="utf-8"))

    def test_a_round_that_ended_without_a_verdict_file_is_said_so_and_is_not_a_rejection(self):
        d, params = FinishTests.job(self, legs())
        jb.finish("student", self.lab, d, params)
        (self.lab / "train").mkdir(exist_ok=True)
        t = jb.params_for("train", self.lab, jb.load_config(self.lab)["train"])
        self.assertIsNone(jb.finish("train", self.lab, self.tmp, t))  # no promotion.json in the candidate's folder
        st = jb.read_student_state(self.lab)
        self.assertEqual(("no verdict", None), (st["rounds"][-1]["judged"], st["last_promotion"]))
        self.assertIn("no verdict (the round ended without a promotion.json)", (self.lab / "student" / "verdicts.md").read_text(encoding="utf-8"))


class PlateauAndDailyTests(Base):
    def on(self):
        self.config(enabled=True)

    def test_the_loop_is_inert_while_neither_the_teacher_nor_the_student_is_on(self):
        (self.lab / "jobs.json").write_text(json.dumps({"student": {"enabled": False}}), encoding="utf-8")
        self.assertEqual([], jb.run_loop_tick(self.lab, NOW))
        self.assertFalse((self.lab / "student").exists())
        self.assertFalse((self.lab / "run-loop.md").exists())
        self.assertEqual("", jb.run_loop_status(self.lab))

    def test_ten_days_without_a_run_loop_promotion_write_run_plateau_once(self):
        self.on()
        jb.run_loop_tick(self.lab, NOW)  # the loop's clock starts
        self.assertEqual(NOW, jb.read_student_state(self.lab)["loop_start"])
        self.assertEqual([], [m for m in jb.run_loop_tick(self.lab, NOW + 9.9 * DAY) if "ALERT" in m])
        self.assertFalse(al.alert_path(self.lab, "run-plateau").exists())
        msgs = jb.run_loop_tick(self.lab, NOW + 10.1 * DAY)
        self.assertTrue(any(m.startswith("ALERT run-plateau") for m in msgs))
        text = al.alert_path(self.lab, "run-plateau").read_text(encoding="utf-8")
        self.assertIn("ALERT run-plateau: no run-loop promotion in 10 days", text)
        self.assertIn("(no student round yet)", text)
        self.assertFalse(any("ALERT" in m for m in jb.run_loop_tick(self.lab, NOW + 11 * DAY)))  # raised once, not at every pass

    def test_the_last_verdicts_are_in_the_plateau_note(self):
        self.on()
        d, params = FinishTests.job(self, legs(card=False))
        jb.finish("student", self.lab, d, params)
        jb.run_loop_tick(self.lab, NOW)
        jb.run_loop_tick(self.lab, NOW + 11 * DAY)
        text = al.alert_path(self.lab, "run-plateau").read_text(encoding="utf-8")
        self.assertIn("round 1 (student", text)
        self.assertIn("card: leg 1 True, leg 2 False", text)

    def test_a_promotion_resets_the_ten_days(self):
        self.on()
        jb.run_loop_tick(self.lab, NOW)
        self.state(last_promotion=NOW + 8 * DAY)
        jb.run_loop_tick(self.lab, NOW + 15 * DAY)
        self.assertFalse(al.alert_path(self.lab, "run-plateau").exists())  # seven days since the promotion
        jb.run_loop_tick(self.lab, NOW + 18.5 * DAY)
        self.assertTrue(al.alert_path(self.lab, "run-plateau").exists())

    def test_removing_the_plateau_note_by_hand_counts_ten_days_again_from_then(self):
        self.on()
        jb.run_loop_tick(self.lab, NOW)
        jb.run_loop_tick(self.lab, NOW + 10.5 * DAY)
        al.alert_path(self.lab, "run-plateau").unlink()
        msgs = jb.run_loop_tick(self.lab, NOW + 11 * DAY)
        self.assertTrue(any("run-plateau.txt removed by hand" in m for m in msgs))
        self.assertFalse(al.alert_path(self.lab, "run-plateau").exists())
        jb.run_loop_tick(self.lab, NOW + 20 * DAY)
        self.assertFalse(al.alert_path(self.lab, "run-plateau").exists())  # nine days after the removal
        jb.run_loop_tick(self.lab, NOW + 21.5 * DAY)
        self.assertTrue(al.alert_path(self.lab, "run-plateau").exists())

    def test_the_banner_and_status_prefix_show_the_new_alerts_like_any_other(self):
        al.write_run_student(self.lab, [], now=NOW)
        al.write_run_plateau(self.lab, NOW - 11 * DAY, 10, [], now=NOW)
        self.assertEqual("ALERT run-plateau, run-student (LAB/alerts)", al.status_prefix(self.lab))
        self.assertIn("run-student.txt", al.banner(self.lab))

    # ---- the daily line

    def ledger(self, store, *rows):
        for r in rows:
            jb.append_ledger(jb.teacher_ledger(store), r)

    def test_a_days_figures_are_the_runs_screens_and_core_hours_that_ended_that_day_in_every_ledger(self):
        day = jb._iso(NOW - DAY)[:10]
        led = jb.teacher_ledger(self.store)
        led.parent.mkdir(parents=True, exist_ok=True)
        old = lambda ev, cpu=0.0, **k: json.dumps({"t": jb._iso(NOW - DAY), "ev": ev, "job": "j", "seed": "s", "cpu": cpu, **k}) + "\n"  # noqa: E731
        led.write_text(old("end", 3600.0, screens=40) + old("end", 7200.0, screens=60) + old("start") + old("end", 0.0, retracted=True)
                       + json.dumps({"t": jb._iso(NOW - 5 * DAY), "ev": "end", "cpu": 99999.0, "screens": 9}) + "\n", encoding="utf-8")
        r0 = self.store / "gen13-r0" / "ledger.jsonl"
        r0.parent.mkdir(parents=True)
        r0.write_text(old("end", 1800.0, screens=5), encoding="utf-8")
        self.assertEqual({"runs": 3, "screens": 105, "core_hours": 3.5}, jb.teacher_day_figures(self.store, day))
        self.assertEqual({"runs": 0, "screens": 0, "core_hours": 0.0}, jb.teacher_day_figures(self.store / "none", day))

    def test_the_tick_writes_the_closed_days_line_once_a_day_and_the_status_text(self):
        self.on()
        led = jb.teacher_ledger(self.store)
        led.parent.mkdir(parents=True, exist_ok=True)
        led.write_text(json.dumps({"t": jb._iso(NOW - DAY), "ev": "end", "job": "j", "seed": "s", "cpu": 36000.0, "screens": 1500}) + "\n", encoding="utf-8")
        msgs = jb.run_loop_tick(self.lab, NOW)
        yesterday = jb._iso(NOW - DAY)[:10]
        self.assertTrue(any(m.startswith(f"run-loop: {yesterday} teacher: 1 runs, 1,500 screens, 10.0 core-hours") for m in msgs))
        text = (self.lab / "run-loop.md").read_text(encoding="utf-8")
        self.assertEqual(1, text.count(yesterday))
        self.assertIn("student: 0 rounds, no fit yet, frozen -; last run-loop promotion none", text)
        self.assertEqual([], [m for m in jb.run_loop_tick(self.lab, NOW + 60) if m.startswith("run-loop:")])  # the same day: no second line
        self.assertEqual(1, (self.lab / "run-loop.md").read_text(encoding="utf-8").count(yesterday))
        jb.run_loop_tick(self.lab, NOW + DAY)  # the next day closes the one after
        self.assertEqual(2, len([ln for ln in (self.lab / "run-loop.md").read_text(encoding="utf-8").splitlines() if ln.startswith("20")]))
        self.assertIn("run-loop today: teacher", jb.run_loop_status(self.lab))
        self.assertIn(yesterday, jb.run_loop_section(self.lab))

    def test_the_line_names_the_last_fit_the_frozen_kinds_and_the_last_promotion(self):
        self.on()
        st = self.state(round_count=2, last_fit={"t": NOW, "utc": "2026-11-20T01:02:03Z", "screens": 12345, "round": 2},
                        last_promotion=NOW - 3 * DAY)
        jb.write_frozen(self.lab, {"card": {}, "rest": {}})
        line = jb.run_loop_line("2026-11-21", jb.config_from(json.loads((self.lab / "jobs.json").read_text(encoding="utf-8"))), st, jb.read_frozen(self.lab))
        self.assertIn("student: 2 rounds, last fit 2026-11-20 on 12,345 screens, frozen card,rest; last run-loop promotion " + jb._iso(NOW - 3 * DAY)[:10], line)


class ChainTests(Base):
    """teacher -> student -> marker -> train -> promotion -> the next teacher block, through the runner's real step machinery."""

    def setUp(self):
        super().setUp()
        self.lab_dir = self.lab
        saved = dict(pl.PLANS)
        self.addCleanup(lambda: (pl.PLANS.clear(), pl.PLANS.update(saved)))

        def unavailable(c):
            raise pl.Unavailable("nothing to do")
        for j in ("fights", "runs", "probes", "mutation", "invariants"):
            pl.PLANS[j] = unavailable
        self.read = self.tmp / "teacher-read.txt"

    def runner(self) -> rn.Runner:
        off = ",".join(ALL_GUARDS)
        a = rn.parse_args(["--rotation", "all", "--reports", "all", "--lab", str(self.lab), "--tree", str(self.tmp / "tree"), "--game", str(self.tmp / "game" / "g"),
                           "--backlog-dir", str(self.tmp / "bl"), "--poll", "0.05", "--wait-poll", "0.05", "--grace", "0", "--no-guard", off])
        r = rn.Runner(a)
        r.lab.runs.mkdir(parents=True, exist_ok=True)
        r.state["last_presorted_sha"] = r.state["tree_sha"] = "abc"
        return r

    def write_jobs(self):
        agents = str(self.agents)
        teacher = ("import sys, json, pathlib; cur = json.loads(pathlib.Path(sys.argv[1], 'current.json').read_text()); "
                   "open(sys.argv[2], 'a').write(str(cur['generation']) + '\\n'); "
                   "d = pathlib.Path(sys.argv[3], str(cur['generation']), 'shards', pathlib.Path(sys.argv[4]).name); d.mkdir(parents=True, exist_ok=True); "
                   "(d / 'decisions.jsonl').write_text('{}\\n' * 6000); (d / 'manifest.json').write_text(json.dumps({'counts': {'decisions': 6000}}))")
        student = ("import sys, json, pathlib; job = pathlib.Path(sys.argv[1]); root = pathlib.Path(sys.argv[2]); tag = sys.argv[3]; "
                   "c = root / 'gen14' / ('candidate-' + tag); c.mkdir(parents=True); (c / 'run_student.npz').write_bytes(b'student'); "
                   "(c / 'agent.json').write_text(json.dumps({'student': {'file': 'run_student.npz'}})); "
                   "v = {k: {'pass': True, 'leg1': True, 'leg2': True if k in ('card', 'rest') else None, 'why': ''} for k in ('card', 'rest', 'reward')}; "
                   "(job / 'result.json').write_text(json.dumps({'built': True, 'candidate': 'gen14/candidate-' + tag, 'decisions': 12000, 'verdict': v, "
                   "'kinds_passing': {k: [0.1] for k in v}}))")
        train = ("import sys, json, pathlib; root = pathlib.Path(sys.argv[1]); c = root / 'gen14' / 'candidate-student'; "
                 "(c / 'promotion.json').write_text(json.dumps({'decision': {'promote': True, 'rule': 'stand-in'}})); "
                 "(root / 'current.json').write_text(json.dumps({'generation': 14, 'dir': str(c), 'promoted': 'now'}))")
        doc = {
            "teacher": {"enabled": True, "store": str(self.store), "cwd": "{lab}", "needs": [],
                        "cmd": ["{python}", "-c", teacher, agents, str(self.read), str(self.store), "{job}"]},
            "student": {"enabled": True, "shards_root": str(self.store), "agent_root": agents, "cwd": "{lab}", "needs": [],
                        "check_cmd": [*CHECK[:-1], str(self.calls)],
                        "cmd": ["{python}", "-c", student, "{job}", agents, "{student_tag}"]},
            "train": {"enabled": True, "cwd": "{lab}", "needs": [], "cmd": ["{python}", "-c", train, agents]},
        }
        (self.lab / "jobs.json").write_text(json.dumps(doc), encoding="utf-8")

    def run_next(self, r: rn.Runner, want: str) -> rn.Runner:
        job, run, why = r.choose("abc")
        self.assertEqual(want, job, why)
        self.assertIn("ladder", why)
        run = r.new_run(job, "abc")
        self.assertIsNone(r.execute(run))
        self.assertEqual("0", (run.dir / "done.rc").read_text().strip(), job)
        return r

    def test_the_whole_chain_runs_by_itself_and_a_promotion_reaches_the_next_teacher_block(self):
        self.write_jobs()
        r = self.runner()
        # nothing is labelled yet: the teacher's slot, the last thing the lab does
        self.assertIn("0 new screens recorded", jb.unready("student", self.lab, jb.load_config(self.lab)["student"]))
        self.run_next(r, "teacher")
        self.assertEqual("13", self.read.read_text().split()[-1])
        # 6,000 screens: under the threshold, the teacher again
        self.run_next(r, "teacher")  # the stand-in writes the same job folder's shard: 6,000 labelled screens in all
        # a second shard (round 0's layout) takes it to 12,000: the student is due
        self.shard(6000, "j9", round0=True)
        self.run_next(r, "student")
        marker = self.lab / "train" / "ready-13-g14-student"
        self.assertTrue(marker.exists())
        self.assertEqual(12000, jb.read_student_state(self.lab)["last_fit"]["screens"])
        # judgement first: with the marker up the next pick is the train job, never a second student or the teacher
        job, _, _ = r.choose("abc")
        self.assertEqual("train", job)
        t = r.default_params("train")
        self.assertEqual(("13", 14, "student"), (t["generation"], t["candidate_generation"], t["tag"]))
        self.run_next(r, "train")
        self.assertTrue((self.lab / "train" / "done-13-g14-student").exists())
        self.assertFalse(marker.exists())
        st = jb.read_student_state(self.lab)
        self.assertEqual("promoted", st["rounds"][-1]["judged"])
        self.assertIsNotNone(st["last_promotion"])
        self.assertEqual(14, json.loads((self.agents / "current.json").read_text())["generation"])
        # the screens were counted at the fit: no second student until 10,000 more; the teacher takes the slot and reads the promoted agent
        self.assertIn("0 new screens recorded", jb.unready("student", self.lab, jb.load_config(self.lab)["student"]))
        self.run_next(r, "teacher")
        self.assertEqual(["13", "13", "14"], self.read.read_text().split())  # the third block played the generation the loop promoted
        text = (self.lab / "student" / "verdicts.md").read_text(encoding="utf-8")
        self.assertIn("student round 1", text)
        self.assertIn("judgement of ready-13-g14-student: PROMOTED", text)

    def test_a_ready_bench_comes_before_the_teacher_and_a_waiting_train_marker_before_the_student(self):
        self.write_jobs()
        doc = json.loads((self.lab / "jobs.json").read_text(encoding="utf-8"))
        doc["bench-emu"] = {"enabled": True, "cwd": "{lab}", "cmd": ["{python}", "-c", "pass"], "needs": []}
        (self.lab / "jobs.json").write_text(json.dumps(doc), encoding="utf-8")
        r = self.runner()
        self.assertEqual("bench-emu", r.choose("abc")[0])
        jb._atomic(self.lab / "bench-emu" / "last-ok", "now\n")
        self.assertEqual("teacher", r.choose("abc")[0])
        self.shard(12000)
        self.assertEqual("student", r.choose("abc")[0])  # outranks the teacher's slot
        (self.lab / "train").mkdir()
        (self.lab / "train" / "ready-13-g14-imit").write_text("x")
        self.assertEqual("train", r.choose("abc")[0])  # ... and is outranked by the judgement

    def test_a_student_round_that_builds_nothing_is_a_finished_job_not_a_failure(self):
        self.write_jobs()
        doc = json.loads((self.lab / "jobs.json").read_text(encoding="utf-8"))
        doc["student"]["cmd"] = ["{python}", "-c", "import sys, json, pathlib; pathlib.Path(sys.argv[1], 'result.json').write_text(json.dumps({'built': False, 'decisions': 12000, "
                                 "'verdict': {}})); sys.exit(3)", "{job}"]
        (self.lab / "jobs.json").write_text(json.dumps(doc), encoding="utf-8")
        self.shard(12000)
        r = self.runner()
        run = r.new_run("student", "abc")
        self.assertEqual([0, 3], run.data["params"]["ok_rcs"])
        self.assertIsNone(r.execute(run))
        self.assertEqual("0", (run.dir / "done.rc").read_text().strip())
        self.assertEqual(0, r.state["fail_streak"].get("student", 0))
        self.assertEqual(12000, jb.read_student_state(self.lab)["last_fit"]["screens"])
        self.assertFalse((self.lab / "train").exists())

    def test_any_other_exit_code_is_a_failure_and_two_in_a_row_stop_the_kind_like_any_other(self):
        self.write_jobs()
        doc = json.loads((self.lab / "jobs.json").read_text(encoding="utf-8"))
        doc["student"]["cmd"] = ["{python}", "-c", "import sys; sys.exit(1)"]
        (self.lab / "jobs.json").write_text(json.dumps(doc), encoding="utf-8")
        self.shard(12000)
        r = self.runner()
        for _ in range(2):
            run = r.new_run("student", "abc")
            r.execute(run)
            self.assertEqual("1", (run.dir / "done.rc").read_text().strip())
        self.assertTrue(al.alert_path(self.lab, "student").exists())
        self.assertIn("stopped after repeated failures", r.unavailable("student", "abc"))
        self.assertEqual({}, jb.read_student_state(self.lab)["fails"])  # a crash is not a failed human bar

    def test_the_runner_writes_the_loops_status_and_ticks_after_its_jobs(self):
        self.write_jobs()
        r = self.runner()
        self.run_next(r, "teacher")
        st = jb.read_student_state(self.lab)
        self.assertIsNotNone(st["loop_start"])  # the tick after the job started the clock
        r.status("idle")
        self.assertIn("| run-loop today: teacher", r.lab.status_path.read_text(encoding="utf-8"))

    def test_a_one_shot_runner_does_not_tick(self):
        self.write_jobs()
        a = rn.parse_args(["--lab", str(self.lab), "--tree", str(self.tmp / "tree"), "--once", "teacher", "--no-guard", ",".join(ALL_GUARDS)])
        r = rn.Runner(a)
        r.run_loop_tick()
        self.assertFalse((self.lab / "student").exists())


class FragmentTests(unittest.TestCase):
    def test_the_documented_fragments_parse_are_disabled_and_are_the_shipped_defaults(self):
        doc = (ROOT / "docs" / "lab" / "RUN_LOOP.md").read_text(encoding="utf-8")
        import re  # noqa: PLC0415

        blocks = re.findall(r"```json\n(\{.*?\n\})\n```", doc, re.S)
        self.assertGreaterEqual(len(blocks), 1)
        frag = json.loads(blocks[0])
        self.assertEqual({"teacher", "student"}, set(frag))
        for kind in ("teacher", "student"):
            self.assertFalse(frag[kind]["enabled"], kind)
            cfg = jb.config_from(frag)[kind]
            self.assertEqual(jb.DEFAULTS[kind]["cmd"], cfg["cmd"], kind)
            self.assertEqual(jb.DEFAULTS[kind]["needs"], cfg["needs"], kind)
        self.assertEqual(10000, frag["student"]["new_screens"])
        self.assertNotIn("selfplay", frag)


if __name__ == "__main__":
    unittest.main()
