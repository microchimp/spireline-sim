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
    """Every guard blocks every new kind: the command does not start while the guard holds."""

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


if __name__ == "__main__":
    unittest.main()
