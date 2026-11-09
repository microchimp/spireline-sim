"""tools/lab: the agent-era job kinds (selfplay, train, score-real, bench-env, bench-emu), the priority ladder and the yield columns.

Trivial `python -c` commands only: no game, no emulator, no long or multi-process work.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.lab import jobs as jb  # noqa: E402
from tools.lab import mix  # noqa: E402
from tools.lab import plans as pl  # noqa: E402
from tools.lab import runner as rn  # noqa: E402

KINDS = ("selfplay", "train", "score-real", "bench-env", "bench-emu")
ALL_GUARDS = ("halt", "user", "pause", "lock", "load", "disk", "parent")
# a trivial command that writes the result a real job would: the {job} token is expanded by the runner
WRITE_RESULT = ["{python}", "-c", "import sys, json; open(sys.argv[1] + '/result.json', 'w').write(sys.argv[2])", "{job}"]
RESULTS = {
    "selfplay": {"runs": 12, "decisions": 3400, "decisions_s": 410.5, "generation": "7"},
    "train": {"generation": "7", "promoted": True, "loss": 0.123},
    "score-real": {"runs": 40, "wins": 3, "mean_floor": 31.5, "crashes": 1, "decisions_s": 55.0, "divergences": 2},
    "bench-env": {"dec_s": 900.5, "crashes": 0, "mem_mb_per_run": 812},
    "bench-emu": {"steps_s": 20000, "clones_s": 5000, "sims_s": 300},
}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        saved = dict(pl.PLANS)
        self.addCleanup(lambda: (pl.PLANS.clear(), pl.PLANS.update(saved)))
        self.lab_dir = self.tmp / "ref" / "LAB"  # readiness.json lives next to LAB
        self.lab_dir.mkdir(parents=True)

    def runner(self, guards_on: tuple[str, ...] = (), *extra: str) -> rn.Runner:
        off = ",".join(g for g in ALL_GUARDS if g not in guards_on)
        a = rn.parse_args(["--lab", str(self.lab_dir), "--tree", str(self.tmp / "tree"), "--game", str(self.tmp / "game" / "g"),
                           "--backlog-dir", str(self.tmp / "bl"), "--poll", "0.05", "--wait-poll", "0.05", "--grace", "0",
                           "--no-guard", off, *extra])
        r = rn.Runner(a)
        r.lab.runs.mkdir(parents=True, exist_ok=True)
        r.state["last_presorted_sha"] = r.state["tree_sha"] = "abc"
        return r

    def config(self, **kinds):
        """LAB/jobs.json: every kind runs its trivial result-writing command in the lab folder, on unless said otherwise."""
        doc = {}
        for k in KINDS:
            doc[k] = {"enabled": k not in jb.BENCHES, "cwd": "{lab}", "cmd": WRITE_RESULT + [json.dumps(RESULTS[k])], "needs": []}
            doc[k].update(kinds.get(k, {}))
        (self.lab_dir / "jobs.json").write_text(json.dumps(doc), encoding="utf-8")

    def nothing_else(self):
        """The rotation's job types cannot run (nothing on this fake tree), so only the ladder's kinds can be chosen."""
        def unavailable(c):
            raise pl.Unavailable("nothing to do")
        for j in ("fights", "runs", "probes", "mutation", "invariants"):
            pl.PLANS[j] = unavailable

    def marker(self, rel: str):
        p = self.lab_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("1\n")
        return p


class LadderTest(Base):
    def test_score_real_beats_selfplay_when_its_marker_exists(self):
        self.config()
        self.nothing_else()
        self.marker("score/pending")
        job, run, why = self.runner().choose("abc")
        self.assertEqual("score-real", job)
        self.assertIsNone(run)
        self.assertIn("ladder", why)

    def test_the_ladder_order_is_score_real_then_train_then_bench_then_the_rotation_then_selfplay(self):
        self.config(**{"bench-emu": {"enabled": True}})
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [pl.Cmd("x", ["{py}", "-c", "pass"], core=True)])]
        r = self.runner()
        self.marker("score/pending")
        self.marker("train/ready-3")
        self.assertEqual("score-real", r.choose("abc")[0])
        (self.lab_dir / "score" / "pending").unlink()
        self.assertEqual("train", r.choose("abc")[0])
        (self.lab_dir / "train" / "ready-3").unlink()
        self.assertEqual("bench-emu", r.choose("abc")[0])  # nightly and the lab otherwise idle
        jb._atomic(self.lab_dir / "bench-emu" / "last-ok", "now\n")
        self.assertEqual("fights", r.choose("abc")[0])  # the rotation, with the mix's own weights

    def test_a_retrain_marker_sorts_after_its_generation_and_before_the_next(self):
        self.config()
        self.nothing_else()
        r = self.runner()
        self.marker("train/ready-12")
        self.marker("train/ready-4-r1")
        self.marker("train/ready-4")
        self.assertEqual("4", r.default_params("train")["generation"])
        (self.lab_dir / "train" / "ready-4").unlink()
        self.assertEqual("4-r1", r.default_params("train")["generation"])
        (self.lab_dir / "train" / "ready-4-r1").unlink()
        self.assertEqual("12", r.default_params("train")["generation"])

    def test_a_target_marker_passes_the_candidate_generation_and_is_consumed_like_the_others(self):
        self.config()
        self.nothing_else()
        r = self.runner()
        self.marker("train/ready-3-g5")
        self.marker("train/ready-12")
        p = r.default_params("train")
        self.assertEqual((p["marker"], p["generation"], p["candidate_generation"]), ("ready-3-g5", "3", 5))
        run = r.new_run("train", "abc")
        r.execute(run)
        self.assertFalse((self.lab_dir / "train" / "ready-3-g5").exists())
        self.assertTrue((self.lab_dir / "train" / "done-3-g5").exists())
        p = r.default_params("train")  # the next marker is an old form: no candidate generation is passed
        self.assertEqual((p["marker"], p["generation"]), ("ready-12", "12"))
        self.assertNotIn("candidate_generation", p)

    def test_train_only_when_its_marker_exists(self):
        self.config()
        self.nothing_else()
        r = self.runner()
        self.assertEqual("selfplay", r.choose("abc")[0])  # train has no marker
        self.marker("train/ready-4")
        self.marker("train/ready-12")
        job, _, _ = r.choose("abc")
        self.assertEqual("train", job)
        self.assertEqual("4", r.default_params("train")["generation"])  # the oldest generation first

    def test_selfplay_when_nothing_else_is_ready(self):
        self.config()
        self.nothing_else()
        job, _, why = self.runner().choose("abc")
        self.assertEqual("selfplay", job)
        self.assertIn("filler", why)

    def test_selfplay_is_off_in_the_shipped_defaults_so_the_live_lab_does_not_change(self):
        self.nothing_else()  # no jobs.json at all
        r = self.runner()
        self.assertEqual("", r.choose("abc")[0])
        self.assertFalse(jb.DEFAULTS["selfplay"]["enabled"])
        self.assertFalse(any(jb.DEFAULTS[k]["enabled"] for k in jb.BENCHES))
        self.assertFalse(jb.DEFAULTS["score-real"]["weekly"])
        self.assertFalse((self.lab_dir / "jobs.json").exists())  # reading the config writes nothing
        self.assertFalse((self.lab_dir / "mix.json").exists() and "selfplay" in (self.lab_dir / "mix.json").read_text())

    def test_the_weekly_score_block_makes_score_real_ready(self):
        self.config(**{"score-real": {"weekly": True}})
        self.nothing_else()
        r = self.runner()
        self.assertEqual("score-real", r.choose("abc")[0])  # never run: due
        jb._atomic(self.lab_dir / "score" / "last-ok", "now\n")
        self.assertEqual("selfplay", r.choose("abc")[0])

    def window(self, rows: dict[str, float], days_ago: float = 1.0):
        """Rows in LAB/yield.tsv that finished `days_ago` days back: strategy -> CPU minutes."""
        t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - days_ago * 86400))
        for strategy, cpu_min in rows.items():
            mix.append_yield(self.lab_dir, {"utc": t, "job": "j-" + strategy, "strategy": strategy, "rc": 0, "cpu_min": cpu_min})

    def fights_only(self):
        self.config(selfplay={"share": 0.5})
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [pl.Cmd("x", ["{py}", "-c", "pass"], core=True)])]
        return self.runner()

    def test_selfplay_share_runs_it_ahead_of_the_rotation_until_its_share_is_spent(self):
        r = self.fights_only()
        self.window({"aimed-bugs": 100.0, "selfplay": 10.0})
        self.assertEqual("selfplay", r.choose("abc")[0])
        self.window({"selfplay": 90.0}, days_ago=0.5)
        self.assertEqual("fights", r.choose("abc")[0])

    def test_the_share_is_the_loop_of_selfplay_plus_train_plus_score_real_not_selfplay_alone(self):
        r = self.fights_only()
        r.state["strategy_cpu"] = {"aimed-bugs": 100.0, "selfplay": 10.0}  # all-time, a statistic: it does not decide
        self.window({"aimed-bugs": 40.0, "selfplay": 30.0, "train": 30.0})  # 60% of the window is the loop
        job, _, why = r.choose("abc")
        self.assertEqual("fights", job)
        self.assertIn("rotation", why)

    def test_selfplay_gets_the_slot_while_the_loop_is_under_its_share(self):
        r = self.fights_only()
        r.state["strategy_cpu"] = {"aimed-bugs": 100.0, "selfplay": 100.0}  # all-time says spent; the window says 30%
        self.window({"aimed-bugs": 70.0, "selfplay": 20.0, "train": 10.0})
        self.assertEqual("selfplay", r.choose("abc")[0])

    def test_score_real_cpu_counts_for_the_loop_and_the_benches_do_not(self):
        r = self.fights_only()
        self.window({"aimed-bugs": 40.0, "selfplay": 20.0, "score-real": 20.0, "bench-emu": 20.0})
        self.assertEqual("selfplay", r.choose("abc")[0])  # 40 of 100: under 50%
        self.window({"score-real": 20.0}, days_ago=0.5)
        self.assertEqual("fights", r.choose("abc")[0])  # 60 of 120 is at the share, no longer under it

    def test_cpu_older_than_seven_days_does_not_count(self):
        r = self.fights_only()
        self.window({"selfplay": 600.0, "train": 300.0}, days_ago=8.0)  # long ago: past the window
        self.window({"aimed-bugs": 90.0, "selfplay": 10.0}, days_ago=2.0)
        r.state["strategy_cpu"] = {"aimed-bugs": 90.0, "selfplay": 610.0, "train": 300.0}
        self.assertEqual("selfplay", r.choose("abc")[0])  # 10 of 100 in the window
        self.window({"selfplay": 300.0}, days_ago=6.5)  # inside the window it counts again
        self.assertEqual("fights", r.choose("abc")[0])

    def test_share_days_sets_the_window_so_one_burst_does_not_lock_the_loop_out_for_a_week(self):
        r = self.fights_only()
        self.config(selfplay={"share": 0.5, "share_days": 2.0})
        self.window({"selfplay": 600.0, "train": 300.0}, days_ago=3.0)  # a burst three days back: outside a two-day window
        self.window({"aimed-bugs": 90.0, "selfplay": 10.0}, days_ago=1.0)
        self.assertEqual("selfplay", r.choose("abc")[0])  # 10 of 100 in the window
        self.config(selfplay={"share": 0.5, "share_days": 7.0})
        self.assertEqual("fights", r.choose("abc")[0])  # 910 of 1,000 over seven days: the burst counts
        self.config(selfplay={"share": 0.5})
        self.assertEqual("fights", r.choose("abc")[0])  # the default window is still seven days

    def test_no_cpu_in_the_window_means_selfplay_first_and_the_log_says_so(self):
        r = self.fights_only()
        self.window({"selfplay": 600.0, "aimed-bugs": 400.0}, days_ago=9.0)  # only old rows: an empty window
        self.assertEqual("selfplay", r.choose("abc")[0])
        self.assertIn("self-play first", (self.lab_dir / "runner.log").read_text(encoding="utf-8"))
        (self.lab_dir / "yield.tsv").unlink()  # no file at all is the same
        self.assertEqual("selfplay", r.choose("abc")[0])

    def test_fights_wait_while_a_score_window_is_open_and_a_stale_marker_does_not_block(self):
        self.config()
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [pl.Cmd("x", ["{py}", "-c", "pass"], core=True)])]
        r = self.runner()
        act = self.marker("score/active")
        self.assertIn("score-real window", r.unavailable("fights", "abc", "dense"))
        self.assertIn("score-real window", r.unavailable("runs", "abc", "whole-runs"))
        old = time.time() - 6 * 3600
        os.utime(act, (old, old))
        self.assertNotIn("score-real window", r.unavailable("fights", "abc", "dense") or "")

    def test_benches_need_their_night_and_a_configured_tool(self):
        self.config(**{"bench-env": {"enabled": True}, "bench-emu": {"enabled": True, "needs": ["{tree}/rl/bench_emu.py"]}})
        r = self.runner()
        self.assertIn("bench_emu.py", r.unavailable("bench-emu", "abc"))  # missing on this tree
        self.assertIsNone(r.unavailable("bench-env", "abc"))
        jb._atomic(self.lab_dir / "bench-env" / "last-ok", "now\n")
        self.assertIn("within the last night", r.unavailable("bench-env", "abc"))


class GuardTest(Base):
    """Every guard blocks every new kind: the command does not start while the guard holds. The one exception is main.lock (the
    landing lock): it guards git state only a presort touches, so a new kind's job, once running, carries on under it (N4);
    the pick gate at the top of the runner's pass still holds every kind (test_lab_runner)."""

    def hold(self, r: rn.Runner, guard: str):
        """Make the guard fire; returns (release function or None for the guards that make the runner leave)."""
        lab = self.lab_dir
        if guard == "pause":
            self.marker("PAUSE")
            return lambda: (lab / "PAUSE").unlink()
        if guard == "lock":
            lock = Path(r.a.backlog_dir) / "main.lock"
            lock.parent.mkdir(parents=True, exist_ok=True)
            lock.write_text("x")
            return lock.unlink
        if guard == "load":
            state = {"busy": True}

            class Busy:
                def sample(self):
                    return None

                def mean(self):
                    return (0.9 if state["busy"] else 0.0, 120.0)

                def sustained(self, th):
                    return state["busy"]
            r.guards.load = Busy()
            return lambda: state.update(busy=False)
        if guard == "disk":
            r.guards.free_gb, r.guards.disk_every = 10**9, 0
            return lambda: setattr(r.guards, "free_gb", 0)
        if guard == "halt":
            self.marker("HALT")
        elif guard == "user":
            r.guards.idle_fn = lambda: 0.0
        elif guard == "parent":
            p = subprocess.Popen([sys.executable, "-c", "pass"])
            p.wait()
            r.guards.parent_pid = p.pid
        return None

    def check(self, kind: str, guard: str):
        self.config()
        if kind == "train":
            self.marker("train/ready-1")
        if kind == "score-real":
            self.marker("score/pending")
        r = self.runner((guard,))
        release = self.hold(r, guard)
        run = r.new_run(kind, "abc")
        log = run.dir / "logs" / f"{kind}.log"
        if guard == "load" and kind == "train":  # the judgement holds the machine (the lab-train line): load does not pause it (test_lab_cuts.py)
            v = r.execute(run)
            self.assertIsNone(v, f"{kind} under {guard}")
            self.assertTrue(log.exists(), f"{kind} did not start under {guard}")
            self.assertTrue((run.dir / "done.rc").exists())
            self.assertNotIn("load", r.state["counters"]["interrupts"])
            release()
            return
        if guard == "lock":  # not a stop for this kind: it runs to its end with the lock up, and nothing is counted as an interrupt
            v = r.execute(run)
            self.assertIsNone(v, f"{kind} under {guard}")
            self.assertTrue(log.exists(), f"{kind} did not start under {guard}")
            self.assertTrue((run.dir / "done.rc").exists())
            self.assertNotIn("lock", r.state["counters"]["interrupts"])
            release()
            return
        slept = []
        real_sleep = time.sleep

        def fake_sleep(s):
            if not slept and release:  # first wait inside the guard's hold: the command must not have started
                slept.append(log.exists())
                release()
            real_sleep(min(s, 0.02))
        with unittest.mock.patch.object(rn.time, "sleep", fake_sleep):
            v = r.execute(run)
        if release:
            self.assertEqual([False], slept, f"{kind} under {guard}")  # held, then released
            self.assertIsNone(v)
            self.assertTrue((run.dir / "done.rc").exists())
        else:
            self.assertIsNotNone(v, f"{kind} under {guard}")
            self.assertEqual(guard, v.kind)
            self.assertFalse(log.exists(), f"{kind} started under {guard}")
            self.assertFalse((run.dir / "done.rc").exists())  # left open for the next start

    def test_every_guard_blocks_every_new_kind(self):
        for kind in KINDS:
            for guard in ALL_GUARDS:
                with self.subTest(kind=kind, guard=guard):
                    self.check(kind, guard)
                    shutil.rmtree(self.lab_dir, True)
                    self.lab_dir.mkdir(parents=True)

    def test_a_guard_that_fires_mid_command_stops_a_new_kind_too(self):
        self.config(selfplay={"cmd": ["{python}", "-c", "import time; time.sleep(30)"]})
        r = self.runner(("halt",))
        run = r.new_run("selfplay", "abc")
        threading_timer = unittest.mock.patch.object(rn.Runner, "poll", autospec=True)
        calls = []

        def poll(self_):
            calls.append(1)
            return rn.guards.Verdict("halt", "test", leave=True) if len(calls) > 3 else rn.guards.Verdict()
        t0 = time.time()
        with threading_timer as m:
            m.side_effect = poll
            v = r.execute(run)
        self.assertEqual("halt", v.kind)
        self.assertLess(time.time() - t0, 20)


class KindRunTest(Base):
    """A dry run of each kind on a trivial command through the runner's real step machinery."""

    def test_each_kind_writes_a_job_json_and_a_yield_row(self):
        self.config()
        self.marker("train/ready-7")
        self.marker("score/pending")
        for kind in KINDS:
            with self.subTest(kind=kind):
                r = self.runner()
                run = r.new_run(kind, "abc")
                self.assertIsNone(r.execute(run))
                d = json.loads((run.dir / "job.json").read_text(encoding="utf-8"))
                self.assertEqual((kind, 0), (d["job"], d["rc"]))
                self.assertEqual("done", d["steps"][0]["cmds"][kind]["status"])
                self.assertEqual({"result": ["result.json"]}, d["outputs"])
                self.assertEqual("0", (run.dir / "done.rc").read_text().strip())
                rows = [x for x in mix.read_yield(self.lab_dir) if x["strategy"] == kind]
                self.assertEqual(1, len(rows))
                for col, val in RESULTS[kind].items():
                    want = "true" if val is True else str(val)
                    self.assertEqual(want, rows[0][col], f"{kind}.{col}")
                self.assertEqual("NA", rows[0]["fights"])
                self.assertGreater(r.state["strategy_cpu"].get(kind, 0) + 1, 0)

    def test_consume_marker_renames_each_marker_form_and_a_job_that_renamed_its_own_marker_finishes_clean(self):
        for name in ("ready-7", "ready-7-r2", "ready-7-g9"):
            m = self.marker("train/" + name)
            done = jb.consume_marker(m)
            self.assertEqual((done, m.exists(), done.is_file()), (m.with_name("done-" + name.removeprefix("ready-")), False, True))
            self.assertIsNone(jb.consume_marker(m))  # nothing left to rename
        # `gen0.py train` stops a round the review guard could not judge: it renames the marker and exits 0 (a finished job)
        self.config(train={"cmd": ["{python}", "-c", "import os, sys; os.replace(sys.argv[1], sys.argv[2])",
                                   str(self.lab_dir / "train" / "ready-8-r2"), str(self.lab_dir / "train" / "done-8-r2")]})
        self.marker("train/ready-8-r2")
        r = self.runner()
        run = r.new_run("train", "abc")
        self.assertIsNone(r.execute(run))
        self.assertEqual("0", (run.dir / "done.rc").read_text().strip())
        self.assertEqual(sorted(p.name for p in (self.lab_dir / "train").iterdir() if p.name.endswith("-8-r2")), ["done-8-r2"])

    def test_markers_are_consumed_on_success_and_kept_on_failure(self):
        self.config()
        self.marker("train/ready-7")
        self.marker("score/pending")
        r = self.runner()
        run = r.new_run("train", "abc")
        self.assertEqual("7", run.data["params"]["generation"])
        r.execute(run)
        self.assertFalse((self.lab_dir / "train" / "ready-7").exists())
        self.assertTrue((self.lab_dir / "train" / "done-7").exists())
        run = r.new_run("score-real", "abc")
        r.execute(run)
        self.assertFalse((self.lab_dir / "score" / "pending").exists())
        self.assertFalse((self.lab_dir / "score" / "active").exists())  # the window closed
        self.assertTrue((self.lab_dir / "score" / "last-ok").exists())
        self.config(train={"cmd": ["{python}", "-c", "import sys; sys.exit(3)"]})
        self.marker("train/ready-8")
        run = r.new_run("train", "abc")
        r.execute(run)
        self.assertTrue((self.lab_dir / "train" / "ready-8").exists())
        self.assertEqual(1, json.loads((run.dir / "job.json").read_text(encoding="utf-8"))["rc"])

    def test_the_score_window_marker_is_open_while_the_command_runs(self):
        seen = self.tmp / "seen.txt"
        code = f"import os; open(r'{seen}', 'w').write(str(os.path.exists(r'{self.lab_dir / 'score' / 'active'}')))"
        self.config(**{"score-real": {"cmd": ["{python}", "-c", code]}})
        self.marker("score/pending")
        r = self.runner()
        r.execute(r.new_run("score-real", "abc"))
        self.assertEqual("True", seen.read_text())
        self.assertFalse((self.lab_dir / "score" / "active").exists())

    def test_benches_write_readiness_json_and_keep_its_other_keys(self):
        self.config(**{"bench-env": {"enabled": True}, "bench-emu": {"enabled": True}})
        ready = self.lab_dir.parent / "readiness.json"
        ready.write_text(json.dumps({"hypotheses": {"open": 2}, "bench": {"emu": {"old": 1}, "other": 5}}), encoding="utf-8")
        r = self.runner()
        r.execute(r.new_run("bench-env", "abc"))
        doc = json.loads(ready.read_text(encoding="utf-8"))
        self.assertEqual({"open": 2}, doc["hypotheses"])
        self.assertEqual(RESULTS["bench-env"], doc["bench"]["env"])
        self.assertEqual({"old": 1}, doc["bench"]["emu"])  # untouched until its own bench runs
        self.assertEqual(5, doc["bench"]["other"])
        r.execute(r.new_run("bench-emu", "abc"))
        doc = json.loads(ready.read_text(encoding="utf-8"))
        self.assertEqual(RESULTS["bench-emu"], doc["bench"]["emu"])
        self.assertEqual(RESULTS["bench-env"], doc["bench"]["env"])
        self.assertEqual({"open": 2}, doc["hypotheses"])

    def test_a_missing_readiness_file_is_created_with_the_bench_only(self):
        self.config(**{"bench-emu": {"enabled": True}})
        r = self.runner()
        r.execute(r.new_run("bench-emu", "abc"))
        doc = json.loads((self.lab_dir.parent / "readiness.json").read_text(encoding="utf-8"))
        self.assertEqual({"bench": {"emu": RESULTS["bench-emu"]}}, doc)

    def test_the_default_placeholder_commands_run_and_name_their_entry_points(self):
        # the shipped defaults, with the kind switched on; the placeholder prints what it stands for
        (self.lab_dir / "jobs.json").write_text(json.dumps({"selfplay": {"enabled": True, "cwd": "{lab}"}}), encoding="utf-8")
        r = self.runner()
        run = r.new_run("selfplay", "abc")
        r.execute(run)
        self.assertIn("rl/run_train/cli.py", (run.dir / "logs" / "selfplay.log").read_text())

    def test_every_new_kind_plans_and_has_a_priority(self):
        for k in KINDS:
            self.assertIn(k, pl.PLANS)
            self.assertIn(k, rn.JOB_TYPES)
        self.assertLess(jb.PRIORITY["presort"], jb.PRIORITY["nightly-real"])  # the real-game checks come first (2026-11-06)
        self.assertLess(jb.PRIORITY["nightly-real"], jb.PRIORITY["score-real"])
        self.assertLess(jb.PRIORITY["score-real"], jb.PRIORITY["train"])
        self.assertLess(jb.PRIORITY["train"], jb.PRIORITY["selfplay"])
        self.assertEqual(7, jb.PRIORITY["bench-env"])


class YieldColumnsTest(Base):
    def test_old_rows_keep_parsing_and_new_rows_round_trip(self):
        old_cols = mix.YIELD_COLUMNS[: mix.YIELD_COLUMNS.index("runs")]
        self.assertEqual("screen_groups", old_cols[-1])
        old_row = ["2026-10-20T00:00:00Z", "20261020-fights", "dense", "abc1234", "0", "30.0", "28.0", "300", "12", "3", "5", "2", "1"]
        (self.lab_dir / "yield.tsv").write_text("\t".join(old_cols) + "\n" + "\t".join(old_row) + "\n", encoding="utf-8")
        before = mix.read_yield(self.lab_dir)  # an old file read as it is
        self.assertEqual(("dense", "1"), (before[0]["strategy"], before[0]["screen_groups"]))
        self.assertNotIn("decisions", before[0])
        new = dict.fromkeys(mix.YIELD_COLUMNS, "NA")
        new.update({"utc": "2026-10-23T00:00:00Z", "job": "j", "strategy": "train", "generation": "7", "promoted": "true", "loss": "0.1"})
        mix.append_yield(self.lab_dir, new)
        rows = mix.read_yield(self.lab_dir)
        self.assertEqual(2, len(rows))
        self.assertEqual(list(mix.YIELD_COLUMNS), list(rows[0]))
        self.assertEqual(("dense", "300", "1", "NA", "NA"), (rows[0]["strategy"], rows[0]["fights"], rows[0]["screen_groups"],
                                                           rows[0]["decisions_s"], rows[0]["sims_s"]))
        self.assertEqual(("train", "7", "true", "0.1"), (rows[1]["strategy"], rows[1]["generation"], rows[1]["promoted"], rows[1]["loss"]))
        # an old reader keeps working: the old columns are still the first ones, in order
        self.assertEqual(list(old_cols), list(mix.YIELD_COLUMNS[: len(old_cols)]))
        self.assertEqual(len(mix.YIELD_COLUMNS), len((self.lab_dir / "yield.tsv").read_text().splitlines()[1].split("\t")))

    def test_a_fights_row_carries_na_in_the_agent_columns(self):
        job = self.tmp / "job"
        job.mkdir()
        row = mix.yield_row(job, {"name": "j", "job": "fights", "params": {"strategy": "dense"}, "rc": 0}, None, False, "t")
        self.assertTrue(all(row[c] == "NA" for c in mix.AGENT_COLUMNS))

    def test_a_result_with_odd_values_cannot_break_the_row(self):
        job = self.tmp / "job"
        job.mkdir()
        (job / "result.json").write_text(json.dumps({"loss": "a\tb\nc", "promoted": False, "extra": 1}), encoding="utf-8")
        row = mix.yield_row(job, {"name": "j", "job": "train", "params": {"generation": "9"}, "rc": 0}, None, False, "t")
        self.assertEqual(("a b c", "false", "9"), (row["loss"], row["promoted"], row["generation"]))
        self.assertNotIn("extra", row)


class NightlyKindTest(Base):
    """The nightly-real kind (D8): due once per night, held with a reason, marked while it runs, stamped and stored when it ends."""

    RESULT = {"date": "2026-10-25", "engine": "core2", "runs_per_agent": 20,
              "agents": {"g3": {"date": "2026-10-25", "runs": 20, "win_rate": 0.0, "mean_floor": 12.5, "boss_kills": 1,
                                "divergence_rate": 0.003, "crashes": 1}}}
    SEES_ACTIVE = ["{python}", "-c", "import sys, json, os; r = json.loads(sys.argv[3]); r['saw_active'] = os.path.exists(sys.argv[2]); "
                   "open(sys.argv[1] + '/result.json', 'w').write(json.dumps(r))", "{job}", "{lab}/nightly/active"]

    def nightly(self, result=None, **kw):
        self.config()
        self.nothing_else()
        doc = json.loads((self.lab_dir / "jobs.json").read_text(encoding="utf-8"))
        doc["nightly-real"] = {"enabled": True, "cwd": "{lab}", "cmd": self.SEES_ACTIVE + [json.dumps(result or self.RESULT)], "needs": [],
                               "every_h": 20, "gpu_ok_file": str(self.tmp / "GPU_OK"), **kw}
        (self.lab_dir / "jobs.json").write_text(json.dumps(doc), encoding="utf-8")
        return self.runner()

    def unready(self):
        return jb.unready("nightly-real", self.lab_dir, jb.load_config(self.lab_dir)["nightly-real"])

    def test_off_until_jobs_json_turns_it_on(self):
        self.assertIn("is off", jb.unready("nightly-real", self.lab_dir, jb.load_config(self.lab_dir)["nightly-real"]))

    def test_due_once_per_night(self):
        self.nightly()
        self.assertIsNone(self.unready())  # never ran
        last = self.marker("nightly/last-ok")
        self.assertEqual("ran within the last night", self.unready())
        old = time.time() - 21 * 3600
        os.utime(last, (old, old))
        self.assertIsNone(self.unready())

    def test_held_with_its_reason_while_halt_pause_or_a_window_is_up(self):
        self.nightly()
        for rel, word in (("HALT", "HALT"), ("PAUSE", "PAUSE"), ("nightly/active", "nightly block has not finished"),
                          ("score/active", "score-real window")):
            m = self.marker(rel)
            self.assertIn(word, self.unready(), rel)
            m.unlink()
        self.assertIsNone(self.unready())

    def test_a_stale_active_marker_is_a_crashs_leftover(self):
        self.nightly()
        m = self.marker("nightly/active")
        old = time.time() - (jb.NIGHTLY_ACTIVE_STALE_H + 1) * 3600
        os.utime(m, (old, old))
        self.assertIsNone(self.unready())

    def test_a_long_block_keeps_its_window_open_for_the_whole_nine_hundred_minute_cap(self):
        self.nightly()
        m = self.marker("nightly/active")
        for hours, open_ in ((1, True), (7.6, True), (14.5, True), (jb.NIGHTLY_ACTIVE_STALE_H + 1, False)):
            t = time.time() - hours * 3600
            os.utime(m, (t, t))
            self.assertEqual(open_, jb.nightly_active(self.lab_dir), hours)  # the filler and the fights runners stay off while it plays
            self.assertEqual(open_, "nightly block has not finished" in (self.unready() or ""), hours)
        self.assertGreaterEqual(jb.NIGHTLY_WINDOW_H, 15.0)
        self.assertEqual(900, jb.DEFAULTS["nightly-real"]["timeout_min"])

    def test_the_ladder_picks_it_before_score_real_and_train_and_says_why_when_it_is_held(self):
        r = self.nightly()
        self.marker("PAUSE")
        self.assertNotEqual("nightly-real", r.choose("abc")[0])
        self.assertTrue(any("nightly-real: skipped, LAB/PAUSE is up" in x for x in (self.lab_dir / "runner.log").read_text().splitlines()))
        (self.lab_dir / "PAUSE").unlink()
        self.marker("train/ready-3")
        self.assertEqual("nightly-real", r.choose("abc")[0])  # the weekly clock is due (never ran): the real-game check first
        self.marker("nightly/last-ok")
        self.assertEqual("train", r.choose("abc")[0])  # ran within the week: the rest of the ladder as ever

    # ---- 2026-11-06: a promotion makes the block ready at once, at the top of the ladder

    def pending(self, age_s: float = 0.0):
        m = self.marker("nightly/pending")
        t = time.time() - age_s
        os.utime(m, (t, t))
        return m

    def test_a_pending_marker_makes_it_ready_though_the_clock_says_it_ran(self):
        self.nightly(every_h=168)
        self.marker("nightly/last-ok")
        self.assertEqual("ran within the last night", self.unready())
        self.pending()
        self.assertIsNone(self.unready())
        self.assertTrue(jb.nightly_wanted(self.lab_dir, 168))
        (self.lab_dir / "nightly" / "pending").unlink()
        self.assertFalse(jb.nightly_wanted(self.lab_dir, 168))

    def test_a_pending_marker_does_not_override_the_other_wait_reasons(self):
        self.nightly()
        self.marker("nightly/last-ok")
        self.pending()
        for rel, word in (("HALT", "HALT"), ("PAUSE", "PAUSE"), ("nightly/active", "nightly block has not finished"),
                          ("score/active", "score-real window")):  # a running real-game job of either kind is never pre-empted
            m = self.marker(rel)
            self.assertIn(word, self.unready(), rel)
            m.unlink()
        self.marker("nightly/last-skip")  # the last attempt was skipped a moment ago: the retry waits an hour, the marker stays
        self.assertIn("skipped", self.unready())
        (self.lab_dir / "nightly" / "last-skip").unlink()
        self.assertIsNone(self.unready())

    def test_the_ladder_puts_a_pending_block_above_score_real_and_train_and_runs_them_after_it(self):
        r = self.nightly(every_h=168)
        self.marker("nightly/last-ok")
        self.pending()
        self.marker("score/pending")
        self.marker("train/ready-3")
        self.assertEqual("nightly-real", r.choose("abc")[0])
        r.execute(r.new_run("nightly-real", "abc"))
        self.assertFalse((self.lab_dir / "nightly" / "pending").exists())  # consumed when the block ended
        self.assertEqual("score-real", r.choose("abc")[0])
        (self.lab_dir / "score" / "pending").unlink()
        self.assertEqual("train", r.choose("abc")[0])

    def test_a_running_score_window_holds_a_pending_block_and_it_goes_first_when_the_window_closes(self):
        r = self.nightly(every_h=168)
        self.marker("nightly/last-ok")
        self.pending()
        self.marker("score/pending")
        a = self.marker("score/active")
        self.assertEqual("score-real", r.choose("abc")[0])  # no pre-emption: the window that is open finishes first
        a.unlink()
        self.assertEqual("nightly-real", r.choose("abc")[0])

    def test_the_marker_is_kept_while_the_block_plays_and_consumed_when_it_ends(self):
        saw = ["{python}", "-c", "import sys, json, os; r = json.loads(sys.argv[3]); r['saw_pending'] = os.path.exists(sys.argv[2]); "
               "open(sys.argv[1] + '/result.json', 'w').write(json.dumps(r))", "{job}", "{lab}/nightly/pending"]
        r = self.nightly(cmd=saw + [json.dumps(self.RESULT)])
        self.marker("nightly/last-ok")
        self.pending()
        r.execute(r.new_run("nightly-real", "abc"))
        res = json.loads(next(self.lab_dir.glob("runs/*-nightly-real/result.json")).read_text(encoding="utf-8"))
        self.assertTrue(res["saw_pending"])  # an interrupted block is resumed on the marker
        self.assertFalse((self.lab_dir / "nightly" / "pending").exists())

    def finish_with(self, result: dict, pending_age_s: float | None):
        """jb.finish for a nightly-real job whose driver wrote `result`; the pending marker `pending_age_s` old (None: none)."""
        if pending_age_s is not None:
            self.pending(pending_age_s)
        job = self.tmp / "job"
        job.mkdir(exist_ok=True)
        (job / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return jb.finish("nightly-real", self.lab_dir, job, {})

    def test_a_promotion_that_landed_after_the_block_took_its_list_keeps_the_marker_for_the_next_block(self):
        t0 = time.time() - 3600  # the block took its list an hour ago and played since
        self.assertIsNone(self.finish_with({"date": "d", "agents": {"g3": {"runs": 1}}, "started_epoch": t0}, pending_age_s=1800))
        self.assertTrue((self.lab_dir / "nightly" / "pending").exists())  # appended 30 min ago: after t0
        last = self.lab_dir / "nightly" / "last-ok"
        self.assertAlmostEqual(t0, last.stat().st_mtime, delta=2)  # stamped with the block's START, so that promotion is "since the last block"
        self.assertIsNone(jb.unready("nightly-real", self.lab_dir, {**jb.load_config(self.lab_dir)["nightly-real"], "enabled": True}))
        # a block that started after the marker's last line consumes it
        self.assertIsNone(self.finish_with({"date": "d", "agents": {"g3": {"runs": 1}}, "started_epoch": time.time() - 600}, pending_age_s=1800))
        self.assertFalse((self.lab_dir / "nightly" / "pending").exists())

    def test_a_result_without_a_start_time_consumes_the_marker_and_stamps_the_end(self):
        self.assertIsNone(self.finish_with({"date": "d", "agents": {"g3": {"runs": 1}}}, pending_age_s=10))
        self.assertFalse((self.lab_dir / "nightly" / "pending").exists())
        self.assertLess(time.time() - (self.lab_dir / "nightly" / "last-ok").stat().st_mtime, 60)
        self.assertFalse(jb.consume_nightly_pending(self.lab_dir, None))  # nothing to consume now

    def test_an_idle_promotion_block_consumes_the_marker_and_stamps_nothing(self):
        self.marker("nightly/last-skip")
        self.assertIsNone(self.finish_with({"idle": "no generation promoted since the last block", "agents": {}, "started_epoch": time.time() - 5},
                                           pending_age_s=60))
        self.assertFalse((self.lab_dir / "nightly" / "pending").exists())
        self.assertFalse((self.lab_dir / "nightly" / "last-ok").exists())
        self.assertTrue((self.lab_dir / "nightly" / "last-skip").exists())
        self.assertFalse((self.lab_dir.parent / "readiness.json").exists())

    def test_a_skipped_or_failed_block_keeps_the_marker(self):
        self.assertIsNone(self.finish_with({"skipped": "QUIET.lock stayed taken", "agents": {}}, pending_age_s=60))
        self.assertTrue((self.lab_dir / "nightly" / "pending").exists())
        r = self.nightly(cmd=["{python}", "-c", "import sys; sys.exit(3)"])
        run = r.new_run("nightly-real", "abc")
        r.execute(run)
        self.assertTrue((self.lab_dir / "nightly" / "pending").exists())  # rc 3: finish never ran
        self.assertFalse((self.lab_dir / "nightly" / "last-ok").exists())

    def test_the_runner_says_why_a_pending_block_is_held(self):
        r = self.nightly(every_h=168)
        self.marker("nightly/last-ok")
        self.pending()
        self.marker("PAUSE")
        self.assertNotEqual("nightly-real", r.choose("abc")[0])
        self.assertTrue(any("nightly-real: skipped, LAB/PAUSE is up" in x for x in (self.lab_dir / "runner.log").read_text().splitlines()))

    def test_a_run_marks_its_window_then_stamps_the_night_and_stores_readiness(self):
        r = self.nightly()
        ready = self.lab_dir.parent / "readiness.json"
        ready.write_text(json.dumps({"bench": {"env": {"dec_s": 1}}, "other": 1}), encoding="utf-8")
        r.execute(r.new_run("nightly-real", "abc"))
        self.assertTrue((self.lab_dir / "nightly" / "last-ok").exists())
        self.assertFalse((self.lab_dir / "nightly" / "active").exists())
        res = json.loads(next(self.lab_dir.glob("runs/*-nightly-real/result.json")).read_text(encoding="utf-8"))
        self.assertTrue(res["saw_active"])  # the marker was up while the command ran
        doc = json.loads(ready.read_text(encoding="utf-8"))
        self.assertEqual(("2026-10-25", 20, 1), (doc["nightly"]["date"], doc["nightly"]["agents"]["g3"]["runs"], doc["nightly"]["agents"]["g3"]["boss_kills"]))
        self.assertEqual((1, {"env": {"dec_s": 1}}), (doc["other"], doc["bench"]))
        self.assertEqual("ran within the last night", self.unready())

    def test_a_skipped_block_is_not_a_night(self):
        r = self.nightly({"skipped": "QUIET.lock stayed taken", "agents": {}})
        r.execute(r.new_run("nightly-real", "abc"))
        self.assertFalse((self.lab_dir / "nightly" / "last-ok").exists())
        self.assertIn("skipped", self.unready())  # tried again after NIGHTLY_RETRY_H, not at once
        self.assertFalse((self.lab_dir.parent / "readiness.json").exists())

    # ---- L3: the consent line is removed whenever a nightly-real job stops

    def gpu(self, text="lab-nightly\nother-lane\n"):
        p = self.tmp / "GPU_OK"
        p.write_text(text, encoding="utf-8")
        return p

    def test_a_finished_job_removes_the_line_and_keeps_the_others(self):
        gpu = self.gpu()
        r = self.nightly()
        r.execute(r.new_run("nightly-real", "abc"))
        self.assertEqual("other-lane\n", gpu.read_text(encoding="utf-8"))

    def test_the_file_is_deleted_when_the_line_was_alone_in_it(self):
        gpu = self.gpu("lab-nightly\n")
        r = self.nightly()
        r.execute(r.new_run("nightly-real", "abc"))
        self.assertFalse(gpu.exists())

    def test_a_job_that_fails_removes_it_too(self):
        gpu = self.gpu()
        r = self.nightly(cmd=["{python}", "-c", "import sys; sys.exit(3)"])
        run = r.new_run("nightly-real", "abc")
        r.execute(run)
        self.assertEqual("other-lane\n", gpu.read_text(encoding="utf-8"))

    def test_a_job_stopped_by_an_interrupt_removes_it_and_stays_open(self):
        gpu = self.gpu()
        r = self.nightly()
        run = r.new_run("nightly-real", "abc")
        with unittest.mock.patch.object(r, "_execute", return_value="a verdict"):  # the guards stopped it: no finally in the job ran
            self.assertEqual("a verdict", r.execute(run))
        self.assertFalse((run.dir / "done.rc").exists())
        self.assertEqual("other-lane\n", gpu.read_text(encoding="utf-8"))

    def test_a_job_that_raises_removes_it(self):
        gpu = self.gpu()
        r = self.nightly()
        run = r.new_run("nightly-real", "abc")
        with unittest.mock.patch.object(r, "_execute", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                r.execute(run)
        self.assertEqual("other-lane\n", gpu.read_text(encoding="utf-8"))

    def test_a_superseded_job_removes_it(self):
        gpu = self.gpu()
        r = self.nightly()
        run = r.new_run("nightly-real", "abc")  # opened, never finished: the runner was killed with its job object
        r.choose("zzz")  # main moved: the open job is closed with rc 3
        self.assertEqual(3, rn.read_rc(run.dir / "done.rc"))
        self.assertEqual("other-lane\n", gpu.read_text(encoding="utf-8"))

    def test_another_kind_stopping_leaves_the_file_alone(self):
        gpu = self.gpu()
        self.config()
        self.nothing_else()
        r = self.runner()
        r.execute(r.new_run("selfplay", "abc"))
        self.assertEqual("lab-nightly\nother-lane\n", gpu.read_text(encoding="utf-8"))

    def test_the_path_is_the_jobs_configs_else_the_default(self):
        self.assertEqual("D:/spireline/referee/GPU_OK", jb.GPU_OK_FILE)
        self.config()
        self.assertEqual(jb.GPU_OK_FILE, self.runner().gpu_ok_path())
        r = self.nightly()
        self.assertEqual(str(self.tmp / "GPU_OK"), r.gpu_ok_path())

    def test_a_gpu_ok_file_in_the_commands_arguments_is_the_one_cleared(self):
        other = self.gpu()  # tmp/GPU_OK: the entry's gpu_ok_file key
        custom = self.tmp / "custom_GPU_OK"
        custom.write_text("lab-nightly\nother-lane\n", encoding="utf-8")
        r = self.nightly(cmd=self.SEES_ACTIVE + [json.dumps(self.RESULT), "--gpu-ok-file", str(custom)])
        self.assertEqual(str(custom), r.gpu_ok_path())
        r.execute(r.new_run("nightly-real", "abc"))
        self.assertEqual("other-lane\n", custom.read_text(encoding="utf-8"))
        self.assertEqual("lab-nightly\nother-lane\n", other.read_text(encoding="utf-8"))  # the command named the other path

    def test_the_loop_share_counts_it(self):
        self.assertNotIn("nightly-real", jb.LOOP_KINDS)  # outside the self-play loop share (D29 accounting)


class ClearConsentTest(Base):
    """L3: jobs.clear_consent is the rule of gen0.gpu_ok_remove for the runner (which does not import rl/gen0.py)."""

    CASES = ["lab-nightly\n", "lab-nightly\nother\n", "other\nlab-nightly\nlab-nightly\n", "other\n", "", "lab-nightly\n\n  \nx\n"]

    def test_the_same_files_as_the_drivers_helper(self):
        sys.path.insert(0, str(ROOT / "rl"))
        import gen0

        for text in self.CASES:
            a, b = self.tmp / "a", self.tmp / "b"
            a.write_text(text, encoding="utf-8")
            b.write_text(text, encoding="utf-8")
            removed = jb.clear_consent(a)
            gen0.gpu_ok_remove(b, "lab-nightly")
            self.assertEqual(b.exists(), a.exists(), repr(text)) if text.strip() else None
            if a.exists() and b.exists():
                self.assertEqual(b.read_text(encoding="utf-8"), a.read_text(encoding="utf-8"), repr(text))
            self.assertEqual("lab-nightly" in [x.strip() for x in text.splitlines()], removed, repr(text))
            a.unlink(missing_ok=True)
            b.unlink(missing_ok=True)

    def test_an_absent_file_is_nothing_to_do(self):
        self.assertFalse(jb.clear_consent(self.tmp / "none"))


class ConsentPathTest(unittest.TestCase):
    def test_the_command_flag_then_the_key_then_the_default(self):
        cp = jb.consent_path
        self.assertEqual("X/G", cp({"cmd": ["{python}", "rl/nightly_real.py", "--gpu-ok-file", "X/G"], "gpu_ok_file": "K"}))
        self.assertEqual("X/G", cp({"cmd": ["a", "--gpu-ok-file=X/G"]}))
        self.assertEqual("X/G", cp({"cmd": "python rl/nightly_real.py --gpu-ok-file X/G"}))
        self.assertEqual("K", cp({"cmd": ["a"], "gpu_ok_file": "K"}))
        self.assertEqual(jb.GPU_OK_FILE, cp({"cmd": ["a", "--gpu-ok-file"]}))  # a flag with no value
        self.assertEqual(jb.GPU_OK_FILE, cp({"cmd": ["a", "--gpu-ok-file", "{lab}/G"]}))  # a token the runner has not expanded: not a path
        self.assertEqual(jb.GPU_OK_FILE, cp(None))


if __name__ == "__main__":
    unittest.main()
