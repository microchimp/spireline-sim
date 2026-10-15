"""tools/lab/mix.py and the runner's use of it: the strategy mix, whole-runs and random jobs, and LAB/yield.tsv.

The mix is checked as a scheduler (shares of CPU time follow the weights, an unavailable or disabled strategy is
skipped, a bad file falls back), the plans per strategy on a fake tree (the scenarios.py call, the budgets, the reasons a
strategy cannot run), and the runner on trivial commands (the strategy of each job, the yield row, per-strategy backoff,
the account of CPU time across stretches of a job, always-on).
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.lab import mix  # noqa: E402
from tools.lab import plans as pl  # noqa: E402
from tools.lab import runner as rn  # noqa: E402


def write(p: Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")


class MixFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_the_default_mix_is_the_owners(self):
        self.assertEqual({"aimed-bugs": 30, "aimed-gaps": 25, "choice-edges": 10, "whole-runs": 10, "dense": 10, "under-tested": 10, "random": 5},
                         mix.DEFAULT_MIX)
        self.assertEqual(set(mix.STRATEGIES), set(mix.DEFAULT_MIX))

    def test_a_missing_file_is_written_with_the_defaults_unless_asked_not_to(self):
        lab = self.tmp / "lab"
        self.assertEqual((mix.DEFAULT_MIX, []), mix.load_mix(lab, create=False))
        self.assertFalse((lab / "mix.json").exists())
        self.assertEqual((mix.DEFAULT_MIX, []), mix.load_mix(lab))
        text = (lab / "mix.json").read_text(encoding="utf-8")
        self.assertEqual(mix.DEFAULT_MIX, mix.parse_mix(text)[0])
        self.assertIn("_note", json.loads(text))  # the owner can read what it is

    def test_the_owners_edit_is_read_whoever_saved_it(self):
        lab = self.tmp / "lab"
        (lab).mkdir()
        for raw in (b'\xef\xbb\xbf{"aimed-bugs": 1, "random": 3}', b'{"aimed-bugs": 1, "random": 3}\x00\x00', b'{"aimed-bugs": 1, "random": 3, "_note": "x"}'):
            (lab / "mix.json").write_bytes(raw)
            self.assertEqual(({"aimed-bugs": 1.0, "random": 3.0}, []), mix.load_mix(lab), raw)

    def test_a_file_that_cannot_be_used_falls_back_with_a_note(self):
        for text, fragment in (("{", "not JSON"), ("[1, 2]", "not an object"), ('{"aimed-bugs": 0}', "no strategy a weight"), ("{}", "no strategy a weight")):
            weights, notes = mix.parse_mix(text)
            self.assertEqual(mix.DEFAULT_MIX, weights)
            self.assertTrue(any(fragment in n for n in notes), (text, notes))
        weights, notes = mix.parse_mix('{"aimed-bugs": 2, "nonsense": 5, "random": -1, "dense": "x", "choice-edges": true, "under-tested": 0}')
        self.assertEqual({"aimed-bugs": 2.0, "under-tested": 0.0}, weights)
        self.assertEqual(4, len(notes))
        self.assertTrue(any("unknown strategy 'nonsense'" in n for n in notes))


class PickTest(unittest.TestCase):
    def test_the_heaviest_goes_first_and_ties_go_to_the_name(self):
        none = lambda s: None  # noqa: E731
        self.assertEqual("aimed-bugs", mix.pick(mix.DEFAULT_MIX, {}, none)[0])
        w = {"dense": 5, "random": 5, "aimed-gaps": 5}
        self.assertEqual("aimed-gaps", mix.pick(w, {}, none)[0])

    def test_cpu_shares_follow_the_weights(self):
        weights = dict(mix.DEFAULT_MIX)
        cost = {"aimed-bugs": 300, "aimed-gaps": 300, "choice-edges": 200, "whole-runs": 1800, "dense": 250, "under-tested": 250, "random": 120}
        spent: dict[str, float] = {}
        jobs = {s: 0 for s in weights}
        for _ in range(400):
            s, _ = mix.pick(weights, spent, lambda s: None)
            spent[s] = spent.get(s, 0.0) + cost[s]
            jobs[s] += 1
        total = sum(spent.values())
        for s, w in weights.items():
            self.assertAlmostEqual(w / 100, spent[s] / total, delta=0.02, msg=s)
        self.assertGreater(jobs["aimed-bugs"], jobs["whole-runs"])  # many short jobs of one, few long jobs of the other

    def test_an_unavailable_or_disabled_strategy_is_skipped_and_the_reason_kept(self):
        w = {"aimed-bugs": 30, "aimed-gaps": 25, "random": 5, "dense": 0}
        got, skipped = mix.pick(w, {}, lambda s: "no backlog" if s == "aimed-bugs" else None)
        self.assertEqual(("aimed-gaps", {"aimed-bugs": "no backlog"}), (got, skipped))
        self.assertEqual((None, {"aimed-bugs": "x", "aimed-gaps": "x", "random": "x"}), mix.pick(w, {}, lambda s: "x"))
        self.assertEqual("random", mix.pick({"random": 1}, {"random": 1e9}, lambda s: None)[0])

    def test_a_new_weight_catches_up_from_zero_spent(self):
        spent = {"aimed-bugs": 1000.0, "aimed-gaps": 800.0}
        w = {"aimed-bugs": 30, "aimed-gaps": 25, "random": 5}
        self.assertEqual("random", mix.pick(w, spent, lambda s: None)[0])
        self.assertIn("random 8%/0%", mix.share_table(w, spent))


class YieldTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def job(self):
        job = self.tmp / "lab" / "runs" / "20261015T0100-fights"
        write(job / "fights" / "a.verdicts.json", [
            {"seed": "S1", "identical": True}, {"seed": "S2", "identical": False, "step": 3},
            {"seed": "S3", "identical": False, "step": 1, "mirror_gap": [4]}, {"seed": "S4", "identical": True, "mirror_gap": [2]},
            {"seed": "S5", "error": "no result"}])
        write(job / "fights" / "b.verdicts.json", [{"seed": "S6", "identical": True}])
        return job

    def test_counts_from_the_verdicts(self):
        self.assertEqual({"fights": 6, "mismatches": 2, "unbuildable": 3}, mix.job_counts(self.job()))

    def test_counts_for_whole_runs(self):
        job = self.tmp / "lab" / "runs" / "20261015T0100-runs"
        write(job / "runs" / "r.runs.json", {"runs": [{"seed": "A", "recorded": True, "first": None}, {"seed": "B", "recorded": True, "first": {"decision": 3}},
                                                       {"seed": "C", "recorded": False}]})
        self.assertEqual({"fights": 3, "mismatches": 1, "unbuildable": 1}, mix.job_counts(job))
        self.assertEqual({"fights": 0, "mismatches": 0, "unbuildable": 0}, mix.job_counts(self.tmp / "nothing"))

    def test_groups_and_fix_ready_come_from_by_job(self):
        doc = {"groups": [
            {"id": "G1", "host_suspect": False, "by_job": {"J1": {"n": 2, "ready": 1, "gap": 0}, "J2": {"n": 1, "ready": 1}}},
            {"id": "G2", "host_suspect": True, "by_job": {"J1": {"n": 1, "ready": 1}}},   # host-flagged: a group, not fix-ready
            {"id": "G3", "host_suspect": False, "by_job": {"J1": {"n": 4, "ready": 0}}},  # nothing traced
            {"id": "G5", "host_suspect": False, "by_job": {"J1": {"n": 2, "ready": 0, "gap": 2}}},  # a card screen the mirror missed
            {"id": "G4", "host_suspect": False, "by_job": {"J2": {"n": 1, "ready": 1}}}]}
        self.assertEqual((4, 1, 1), mix.job_groups(doc, "J1"))
        self.assertEqual((2, 2, 0), mix.job_groups(doc, "J2"))
        self.assertEqual((0, 0, 0), mix.job_groups(doc, "J9"))
        self.assertEqual((0, 0, 0), mix.job_groups({"groups": []}, "J1"))
        self.assertIsNone(mix.job_groups({"groups": [{"id": "G1"}]}, "J1"))  # an older lab_groups: cannot say
        self.assertIsNone(mix.job_groups(None, "J1"))

    def test_the_row_and_the_file(self):
        job = self.job()
        data = {"name": job.name, "job": "fights", "main_sha": "abcdef0123", "rc": 0, "cpu_secs": 150, "wall_secs": 90,
                "params": {"strategy": "aimed-bugs"}}
        doc = {"groups": [{"host_suspect": False, "by_job": {job.name: {"n": 1, "ready": 1}}},
                          {"host_suspect": False, "by_job": {job.name: {"n": 1, "ready": 0, "gap": 1}}}]}
        row = mix.yield_row(job, data, doc, True, "2026-10-15T01:00:00Z")
        self.assertEqual(("aimed-bugs", "abcdef0", 2.5, 1.5, 6, 2, 3, 2, 1, 1),
                         (row["strategy"], row["main"], row["cpu_min"], row["wall_min"], row["fights"], row["mismatches"], row["unbuildable"],
                          row["groups"], row["fix_ready"], row["screen_groups"]))
        na = mix.yield_row(job, data, doc, False, "t")  # the groups step did not finish: no count is better than a stale one
        self.assertEqual(("NA", "NA", "NA"), (na["groups"], na["fix_ready"], na["screen_groups"]))
        lab = self.tmp / "lab"
        mix.append_yield(lab, row)
        mix.append_yield(lab, na)
        lines = (lab / "yield.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual("\t".join(mix.YIELD_COLUMNS), lines[0])
        self.assertEqual(3, len(lines))
        self.assertEqual(len(mix.YIELD_COLUMNS), len(lines[1].split("\t")))
        self.assertEqual(["aimed-bugs", "aimed-bugs"], [r["strategy"] for r in mix.read_yield(lab)])
        self.assertEqual("NA", mix.read_yield(lab)[1]["groups"])

    def test_an_older_file_with_fewer_columns_is_widened_and_a_foreign_one_is_kept_aside(self):
        lab = self.tmp / "lab"
        lab.mkdir()
        old_cols = mix.YIELD_COLUMNS[:-1]
        (lab / "yield.tsv").write_text("\t".join(old_cols) + "\n" + "\t".join(str(i) for i in range(len(old_cols))) + "\n", encoding="utf-8")
        mix.append_yield(lab, {c: "x" for c in mix.YIELD_COLUMNS})
        rows = mix.read_yield(lab)
        self.assertEqual(["NA", "x"], [r["screen_groups"] for r in rows])
        self.assertEqual("0", rows[0]["utc"])
        self.assertNotIn("\r", (lab / "yield.tsv").read_bytes().decode("utf-8"))
        (lab / "yield.tsv").write_text("a\tb\n1\t2\n", encoding="utf-8")  # not ours
        mix.append_yield(lab, {c: "y" for c in mix.YIELD_COLUMNS})
        self.assertEqual(["y"], [r["utc"] for r in mix.read_yield(lab)])
        self.assertTrue((lab / "yield.tsv.old").exists())

    def test_a_job_without_a_strategy_is_named_by_its_type(self):
        job = self.job()
        row = mix.yield_row(job, {"name": job.name, "job": "mutation", "params": {}}, None, False, "t")
        self.assertEqual("mutation", row["strategy"])


def fake_tree(root: Path, **have) -> Path:
    """A tree whose tools carry (or lack) the markers the lab reads: scenarios.py options, the driver's key, batch.py, runs.py."""
    tree = root / "tree"
    scen = ['"targets"', '"--jobs"'] + (['"--aim"'] if have.get("aim", True) else []) + (['"choices"'] if have.get("choices", True) else []) \
        + (['"analyze"'] if have.get("analyze", True) else []) + (['"--shrink-cap"'] if have.get("budgets", True) else [])
    write(tree / "referee" / "scenarios.py", " ".join(scen))
    write(tree / "referee" / "driver" / "Combat.cs", "choice_edges" if have.get("edges", True) else "")
    write(tree / "tools" / "fidelity" / "lab_targets.py", "")
    if have.get("batch", True):
        write(tree / "referee" / "batch.py", "")
    write(tree / "referee" / "runs.py", '"--jobs"' if have.get("runs", True) else "")
    for rel in ("tools/emudump/bin/Release/net9.0/EmuDump.dll", "referee/driver/bin/Release/net9.0/SpirelineReferee.dll"):
        write(tree / rel, "")
    return tree


class StrategyPlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def ctx(self, strategy, tree=None, backlog=True, codex=True, **params):
        tree = tree or fake_tree(self.tmp)
        if codex:
            write(self.tmp / "codex.gz", "")
        else:
            (self.tmp / "codex.gz").unlink(missing_ok=True)
        bl = self.tmp / "bl" / "backlog-c1.json"
        return pl.Ctx(lab=self.tmp / "lab", tree=tree, game=self.tmp / "g" / "game", fork=self.tmp / "fork", py="py", sha="a" * 40,
                      job_dir=self.tmp / "lab" / "runs" / "20261015T0100-fights", job_name="20261015T0100-fights",
                      params={"batches": 2, "batch": 50, "start": 7, "servers": 8, "shrink": 1000, "shrink_cap": 160, "shrink_budget": 6000,
                              "shrink_minutes": 45, "trace": 1000, "trace_minutes": 20, "strategy": strategy, "run_batch": 10, "procs": 4, **params},
                      backlog_dir=self.tmp / "bl", backlog=bl if backlog else None, codex=self.tmp / "codex.gz", decompiled=self.tmp / "dec",
                      records=self.tmp / "rec", game_src=self.tmp / "gsrc", nuget=self.tmp / "nuget")

    def argv(self, steps, step="batch-00", cmd=0):
        return [s for s in steps if s.name == step][0].cmds[cmd].argv

    def opt(self, argv, name):
        return argv[argv.index(name) + 1]

    def test_each_scenarios_strategy_calls_its_mode(self):
        for strategy, mode, extra in (("aimed-bugs", "targets", ("--aim", "bugs")), ("aimed-gaps", "targets", ("--aim", "gaps")),
                                      ("choice-edges", "choices", ("--features", "{lab}/cache/model_features.json")),
                                      ("dense", "dense", None), ("under-tested", "coverage", None)):
            steps = pl.plan_fights(self.ctx(strategy))
            argv = self.argv(steps)
            self.assertEqual(["{py}", "referee/scenarios.py"], argv[:2], strategy)
            self.assertEqual(mode, self.opt(argv, "--mode"), strategy)
            if extra:
                self.assertEqual(extra[1], self.opt(argv, extra[0]), strategy)
            self.assertEqual(("8", "{job}/fights", "8"), (self.opt(argv, "--servers"), self.opt(self.argv(steps, "batch-01"), "--jobs"),
                                                         self.opt(self.argv(steps, "batch-01"), "--start")))
            self.assertEqual("targets" in [s.name for s in steps], mode == "targets")
            self.assertEqual("--targets" in argv, mode == "targets")

    def test_every_divergence_is_shrunk_and_traced_within_the_jobs_budget(self):
        argv = self.argv(pl.plan_fights(self.ctx("aimed-gaps")))
        self.assertEqual(["1000", "160", "3000", "23", "1000", "10", "8"],
                         [self.opt(argv, o) for o in ("--shrink", "--shrink-cap", "--shrink-budget", "--shrink-minutes", "--trace", "--trace-minutes", "--trace-servers")])
        argv = self.argv(pl.plan_fights(self.ctx("dense", batches=6, servers=3)))  # the job's budget is split over its batches
        self.assertEqual(["8", "4", "3"], [self.opt(argv, o) for o in ("--shrink-minutes", "--trace-minutes", "--trace-servers")])
        old = fake_tree(self.tmp / "old", budgets=False)  # a tree whose scenarios.py has no budgets gets none
        self.assertNotIn("--shrink-cap", self.argv(pl.plan_fights(self.ctx("dense", tree=old))))

    def test_random_is_a_batch_py_batch_then_an_analysis_of_it(self):
        steps = pl.plan_fights(self.ctx("random"))
        batch, analyze = [s for s in steps if s.name == "batch-00"][0].cmds
        self.assertEqual(["{py}", "referee/batch.py"], batch.argv[:2])
        self.assertEqual(("50", "8", "{job}/fights", "7"), tuple(self.opt(batch.argv, o) for o in ("--fights", "--parallel", "--jobs", "--start")))
        self.assertEqual((0, 1), batch.ok_rcs[:2])
        self.assertEqual("analyze", self.opt(analyze.argv, "--mode"))
        self.assertEqual(self.opt(batch.argv, "--name"), self.opt(analyze.argv, "--name"))
        self.assertEqual("{job}/fights", self.opt(analyze.argv, "--jobs"))
        self.assertEqual("1000", self.opt(analyze.argv, "--shrink"))
        self.assertEqual("fights/lab20261015T0100-b00.verdicts.json", analyze.outputs["fights"])
        self.assertTrue(all(c.core for c in (batch, analyze)))
        self.assertIn("{job}/fights/lab20261015T0100-b00.packs", analyze.cleanup)
        self.assertNotIn("{job}/fights/lab20261015T0100-b00.*", analyze.cleanup)  # batch.py's own outputs stay

    def test_whole_runs_are_runs_py_calls_in_safe_points(self):
        steps = pl.plan_runs(self.ctx("whole-runs", batches=4))
        calls = [s for s in steps if s.name.startswith("runs-")]
        self.assertEqual(4, len(calls))
        argv = calls[2].cmds[0].argv
        self.assertEqual(["{py}", "referee/runs.py"], argv[:2])
        self.assertEqual(("lab20261015T0100-w02", "10", "9", "4", "{job}/runs"),
                         tuple(self.opt(argv, o) for o in ("--name", "--runs", "--start", "--procs", "--jobs")))
        cmd = calls[0].cmds[0]
        self.assertEqual((0, 1, 3, 4), cmd.ok_rcs)
        self.assertEqual("runs/lab20261015T0100-w00.runs.json", cmd.outputs["runs"])
        self.assertEqual("report", "report" if [s.name for s in steps][-2:] == ["groups", "kits"] else "")

    def test_reasons_a_strategy_cannot_run(self):
        no = lambda strategy, tree=None, **kw: pl.strategy_unavailable(strategy, self.ctx(strategy, tree=tree, **kw))  # noqa: E731
        self.assertIsNone(no("aimed-bugs"))
        self.assertIn("no --aim", no("aimed-bugs", fake_tree(self.tmp / "t1", aim=False)))
        self.assertIn("no backlog", no("aimed-bugs", backlog=False))
        self.assertIsNone(no("aimed-gaps", backlog=False))  # gaps do not need a backlog
        self.assertIn("no Codex dataset", no("aimed-gaps", codex=False, tree=fake_tree(self.tmp / "t2")))
        self.assertIn("no choices mode", no("choice-edges", fake_tree(self.tmp / "t3", choices=False)))
        self.assertIn("no choice_edges", no("choice-edges", fake_tree(self.tmp / "t4", edges=False)))
        self.assertIn("batch.py", no("random", fake_tree(self.tmp / "t5", batch=False)))
        self.assertIn("analyze", no("random", fake_tree(self.tmp / "t6", analyze=False)))
        self.assertIn("no --jobs", no("whole-runs", fake_tree(self.tmp / "t7", runs=False)))
        self.assertIsNone(no("dense"))
        with self.assertRaises(pl.Unavailable):
            pl.plan_fights(self.ctx("choice-edges", tree=fake_tree(self.tmp / "t8", edges=False)))
        with self.assertRaises(pl.Unavailable):
            pl.plan_runs(self.ctx("whole-runs", tree=fake_tree(self.tmp / "t9", runs=False)))

    def test_a_job_from_before_strategies_plans_as_it_did(self):
        steps = pl.plan_fights(self.ctx(None))
        argv = self.argv(steps)
        self.assertEqual(("targets", "{lab}/targets.json"), (self.opt(argv, "--mode"), self.opt(argv, "--targets")))
        self.assertNotIn("--shrink-cap", argv)
        self.assertNotIn("--aim", argv)

    def test_the_safety_checks_read_batch_pys_second_line_and_runs_pys_sessions(self):
        c = self.ctx("random")
        write(c.job_dir / "fights" / "n.summary.txt", "SPEED 1 verified\nbatch n: 5 fights, session exit [0], 3 s, network [], real profile touched False\n")
        self.assertIsNone(pl._halt_on_unsafe("n")(c))
        write(c.job_dir / "fights" / "n.summary.txt", "SPEED 1\nbatch n: 5 fights, session exit [0], 3 s, network ['1.2.3.4'], real profile touched False\n")
        self.assertIn("HALT", pl._halt_on_unsafe("n")(c))
        write(c.job_dir / "fights" / "n.summary.txt", "SPEED 1\nbatch n: 5 fights, session exit [0], 3 s, network [], real profile touched True\n")
        self.assertIn("HALT", pl._halt_on_unsafe("n")(c))
        check = pl._halt_on_unsafe_runs("w")
        ok = "session 0: {'exit': 0, 'network_connections': [], 'real_profile_touched': False}\n"
        write(c.job_dir / "runs" / "w.summary.txt", ok + "40 runs\n")
        self.assertIsNone(check(c))
        write(c.job_dir / "runs" / "w.summary.txt", ok + "session 1 FAILED, the batch goes on: SESSION FAILED: exit 3221225477\n")
        self.assertIn("1 game session(s) crashed", check(c))
        self.assertFalse(check(c).startswith("HALT"))
        write(c.job_dir / "runs" / "w.summary.txt", "session 0: {'exit': 0, 'network_connections': ['x'], 'real_profile_touched': False}\n")
        self.assertTrue(check(c).startswith("HALT"))
        write(c.job_dir / "runs" / "w.summary.txt", ok + "session 1: SAFETY CHECK FAILED (network_connections missing)\n")
        self.assertTrue(check(c).startswith("HALT"))
        write(c.job_dir / "runs" / "w.summary.txt", "session 0: {'exit': 0, 'network_connections': [], 'real_profile_touched': True}\n")
        self.assertTrue(check(c).startswith("HALT"))


class RunnerMixTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        saved = dict(pl.PLANS)
        self.addCleanup(lambda: (pl.PLANS.clear(), pl.PLANS.update(saved)))
        self.tree = fake_tree(self.tmp)
        write(self.tmp / "codex.gz", "")
        write(self.tmp / "bl" / "backlog-c1.json", {"items": []})

    def runner(self, *extra: str) -> rn.Runner:
        a = rn.parse_args(["--lab", str(self.tmp / "lab"), "--tree", str(self.tree), "--game", str(self.tmp / "game" / "g"),
                           "--backlog-dir", str(self.tmp / "bl"), "--codex", str(self.tmp / "codex.gz"), "--poll", "0.05", "--wait-poll", "0.05",
                           "--grace", "0", "--no-guard", "user,load,disk", *extra])
        r = rn.Runner(a)
        r.lab.runs.mkdir(parents=True, exist_ok=True)
        r.state["last_presorted_sha"] = r.state["tree_sha"] = "abc"
        self.silence(r)
        return r

    @staticmethod
    def silence(r):
        r.log = lambda msg, _l=[]: _l.append(msg)  # noqa: B006
        r.logged = r.log.__defaults__[0]

    def trivial_plans(self):
        fake = lambda c: [pl.Step("a", [pl.Cmd("x", ["{py}", "-c", "pass"], cwd="{lab}", core=True, counts={"fights": 1})], mode="serial")]  # noqa: E731
        pl.PLANS["fights"], pl.PLANS["runs"] = fake, fake

    def run_one(self, r, cpu=100.0):
        job, run, why = r.choose("abc")
        self.assertTrue(job, why)
        run = r.new_run(job, "abc")
        mark = [getattr(r, "_fake_cpu", 0.0)]
        r.cpu_secs = lambda: mark[0]
        orig_execute = r._execute

        def execute(run_):
            mark[0] += cpu  # the job's children used this much CPU
            return orig_execute(run_)
        r._execute = execute
        try:
            r.execute(run)
        finally:
            r._execute = orig_execute
            r._fake_cpu = mark[0]
        return run

    def test_jobs_follow_the_mix_and_each_leaves_a_yield_row(self):
        self.trivial_plans()
        write(self.tmp / "lab" / "mix.json", {"aimed-bugs": 2, "aimed-gaps": 1, "whole-runs": 1})
        r = self.runner("--fights", "50", "--batch", "50")
        seen = []
        for _ in range(8):
            run = self.run_one(r)
            seen.append((run.job, run.data["params"]["strategy"]))
        self.assertEqual({"aimed-bugs": 4, "aimed-gaps": 2, "whole-runs": 2}, {s: sum(1 for _, x in seen if x == s) for s in ("aimed-bugs", "aimed-gaps", "whole-runs")})
        self.assertEqual("runs", next(j for j, s in seen if s == "whole-runs"))
        self.assertEqual("fights", next(j for j, s in seen if s == "aimed-bugs"))
        rows = mix.read_yield(r.lab.root)
        self.assertEqual(8, len(rows))
        self.assertEqual(seen[0][1], rows[0]["strategy"])
        self.assertEqual(["0"] * 8, [x["rc"] for x in rows])
        self.assertEqual(["1.7"] * 8, [x["cpu_min"] for x in rows])
        self.assertEqual({"aimed-bugs": 400.0, "aimed-gaps": 200.0, "whole-runs": 200.0}, r.state["strategy_cpu"])
        self.assertEqual(["NA"] * 8, [x["groups"] for x in rows])  # no groups step ran in these plans

    def test_the_owners_edit_of_the_mix_takes_effect_at_the_next_job_and_a_bad_file_is_logged_once(self):
        self.trivial_plans()
        r = self.runner()
        write(self.tmp / "lab" / "mix.json", {"dense": 1})
        self.assertEqual("dense", self.run_one(r).data["params"]["strategy"])
        write(self.tmp / "lab" / "mix.json", {"random": 1})
        self.assertEqual("random", self.run_one(r).data["params"]["strategy"])
        write(self.tmp / "lab" / "mix.json", "{not json")
        for _ in range(3):
            r.choose("abc")
        self.assertEqual(1, sum("not JSON" in m for m in r.logged))  # the note is made once, not every pass
        self.assertTrue(any("is next" in m for m in r.logged))

    def test_a_strategy_that_keeps_failing_backs_off_alone(self):
        bad = lambda c: [pl.Step("a", [pl.Cmd("x", ["{py}", "-c", "import sys; sys.exit(2)"], cwd="{lab}", core=True)])]  # noqa: E731
        good = lambda c: [pl.Step("a", [pl.Cmd("x", ["{py}", "-c", "pass"], cwd="{lab}", core=True)])]  # noqa: E731
        pl.PLANS["fights"] = lambda c: bad(c) if c.params.get("strategy") == "dense" else good(c)
        write(self.tmp / "lab" / "mix.json", {"dense": 10, "random": 1})
        r = self.runner()
        r.state["strategy_cpu"]["random"] = 1e6  # random has had far more than its share, so dense is picked until it backs off
        for _ in range(2):
            self.assertEqual(("fights", "dense"), (lambda run: (run.job, run.data["params"]["strategy"]))(self.run_one(r)))
        self.assertIn("dense", r.state["backoff"])
        self.assertNotIn("random", r.state["backoff"])
        self.assertEqual("random", self.run_one(r).data["params"]["strategy"])  # dense waits, the rest goes on
        self.assertEqual("1", mix.read_yield(r.lab.root)[0]["rc"])

    def test_a_strategy_that_cannot_run_is_passed_over(self):
        self.trivial_plans()
        write(self.tmp / "lab" / "mix.json", {"whole-runs": 5, "dense": 1})
        pl.PLANS["runs"] = lambda c: (_ for _ in ()).throw(pl.Unavailable("runs.py on this tree has no --jobs"))
        r = self.runner()
        self.assertEqual("dense", self.run_one(r).data["params"]["strategy"])
        self.assertTrue(any("whole-runs: not runnable" in m for m in r.logged))

    def test_once_runs_and_once_fights_pick_a_strategy(self):
        self.trivial_plans()
        write(self.tmp / "lab" / "mix.json", {"random": 1})
        r = self.runner("--once", "fights")
        self.assertEqual(("fights", "random"), (lambda x: (x[0], r.next_strategy))(r.choose("abc")))
        r = self.runner("--once", "runs")
        self.assertEqual("runs", r.choose("abc")[0])
        run = r.new_run("runs", "abc")
        self.assertEqual(("whole-runs", 4, 10, 1000), (run.data["params"]["strategy"], run.data["params"]["procs"],
                                                       run.data["params"]["run_batch"], run.data["params"]["start"]))
        self.assertEqual(1004, r.state["next_run_start"])  # four calls of ten runs make forty

    def test_once_fights_returns_the_runs_job_when_the_mix_picks_whole_runs(self):
        self.trivial_plans()
        write(self.tmp / "lab" / "mix.json", {"whole-runs": 1})
        r = self.runner("--once", "fights")
        job, _, why = r.choose("abc")
        self.assertEqual(("runs", None), (job, r.next_strategy), why)
        self.assertEqual("whole-runs", r.new_run(job, "abc").data["params"]["strategy"])

    def test_new_jobs_get_the_budgets_and_the_defaults_of_the_task(self):
        r = self.runner()
        p = r.default_params("fights")
        self.assertEqual((8, 1000, 160, 6, 50), (p["servers"], p["shrink"], p["shrink_cap"], p["batches"], p["batch"]))
        self.assertEqual((6000, 45.0, 1000, 20.0), (p["shrink_budget"], p["shrink_minutes"], p["trace"], p["trace_minutes"]))
        p = r.default_params("runs")
        self.assertEqual((4, 10, 4, "whole-runs"), (p["batches"], p["run_batch"], p["procs"], p["strategy"]))

    def test_cpu_is_accounted_across_stretches_of_one_job(self):
        self.trivial_plans()
        r = self.runner()
        cpu = [0.0]
        r.cpu_secs = lambda: cpu[0]
        run = r.new_run("fights", "abc")
        r._mark = (0.0, r.clock())
        cpu[0] = 30.0
        r.account(run)
        cpu[0] = 30.0  # nothing used since: nothing added twice
        r.account(run)
        self.assertEqual(30.0, run.data["cpu_secs"])
        cpu[0] = 100.0
        r.account(run)
        self.assertEqual(100.0, run.data["cpu_secs"])
        self.assertGreaterEqual(run.data["wall_secs"], 0)

    def test_a_job_closed_as_superseded_still_leaves_its_row_and_its_cpu(self):
        self.trivial_plans()
        r = self.runner()
        r.next_strategy = "dense"
        run = r.new_run("fights", "old")
        run.data["cpu_secs"] = 600.0
        run.save()
        job, resumed, why = r.choose("new")  # main moved: the old job is closed
        self.assertEqual("3", (run.dir / "done.rc").read_text().strip())
        rows = mix.read_yield(r.lab.root)
        self.assertEqual(("dense", "3", "10.0"), (rows[0]["strategy"], rows[0]["rc"], rows[0]["cpu_min"]))
        self.assertEqual(600.0, r.state["strategy_cpu"]["dense"])

    def test_the_dry_run_shows_the_mix_and_creates_nothing(self):
        r = self.runner("--dry-run")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r.main_sha = lambda: "abc"
            r.dry_run()
        self.assertIn("mix (wanted share / CPU share had): aimed-bugs 30%/0%", out.getvalue())
        self.assertFalse((r.lab.root / "mix.json").exists())
        self.assertFalse((r.lab.root / "yield.tsv").exists())

    def test_always_on_waits_for_the_owner_to_stop_typing_instead_of_leaving(self):
        r = self.runner("--always-on")
        r.guards.disabled.discard("user")
        idle = [3.0]
        r.guards.idle_fn = lambda: idle[0]
        threading.Timer(0.4, lambda: idle.__setitem__(0, 500.0)).start()
        t0 = time.time()
        self.assertIsNone(r.gate())  # no leave verdict: it paused, then resumed when the input stopped
        self.assertGreaterEqual(time.time() - t0, 0.3)
        r2 = self.runner()  # without --always-on the same input ends the runner
        r2.guards.disabled.discard("user")
        r2.guards.idle_fn = lambda: 3.0
        self.assertTrue(r2.gate().leave)

    def test_always_on_waits_out_a_halt_instead_of_exiting_and_a_plain_runner_exits(self):
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [pl.Cmd("g", ["{py}", "-c", "print('x')"], cwd="{lab}", core=True,
                                                             post=lambda c: "HALT: the game opened a connection")], mode="serial")]
        plain = self.runner()
        plain.setup, plain.main_sha = (lambda: None), (lambda: "abc")
        self.assertEqual(0, plain.run())  # the plain runner ends at once; the owner restarts it after reading the logs
        self.assertTrue((plain.lab.root / "HALT").exists())
        (plain.lab.root / "HALT").unlink()
        r = self.runner("--always-on")
        r.lab = rn.Lab(self.tmp / "lab2")
        r.guards.lab = r.lab.root
        r.lab.root.mkdir(parents=True)
        r.lab.runs.mkdir()
        r.setup, r.main_sha = (lambda: None), (lambda: "abc")
        r.state["last_presorted_sha"] = r.state["tree_sha"] = "abc"
        result: list = []
        t = threading.Thread(target=lambda: result.append(r.run()), daemon=True)
        t.start()
        end = time.time() + 30
        while time.time() < end and not (r.lab.root / "HALT").exists():
            time.sleep(0.05)
        self.assertTrue((r.lab.root / "HALT").exists())
        time.sleep(0.6)
        self.assertTrue(t.is_alive())  # waiting at the gate for the owner, not exited
        self.assertIn("paused(halt)", (r.lab.root / "status.txt").read_text(encoding="utf-8"))
        (r.lab.root / "HALT").unlink()
        r.stop_signal = "signal 15"  # the test ends the runner once it has seen it wait
        t.join(30)
        self.assertFalse(t.is_alive())
        self.assertEqual([0], result)


if __name__ == "__main__":
    unittest.main()
