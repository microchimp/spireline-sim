"""The lab's clock: the runner's per-command accounting line (wall, CPU, I/O, whether the output changed), the stamp's `changed`
field it reads, and tools/lab/clock_report.py with its three flags and its place in readiness.json.

The runner runs trivial `python -c` commands; the accounting needs a real Windows Job Object, so those tests are skipped elsewhere.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import readiness as rd  # noqa: E402
from tools.lab import clock_report as cr  # noqa: E402
from tools.lab import plans as pl  # noqa: E402
from tools.lab import runner as rn  # noqa: E402
from tools.lab import stepcache as sc  # noqa: E402
from tools.lab import winsys  # noqa: E402

WIN = os.name == "nt"


class RunnerClockCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        saved = dict(pl.PLANS)
        self.addCleanup(lambda: (pl.PLANS.clear(), pl.PLANS.update(saved)))

    def runner(self) -> rn.Runner:
        a = rn.parse_args(["--rotation", "all", "--reports", "all", "--lab", str(self.tmp / "lab"), "--tree", str(self.tmp / "tree"), "--game", str(self.tmp / "game" / "g"),
                           "--backlog-dir", str(self.tmp / "bl"), "--poll", "0.05", "--wait-poll", "0.05", "--grace", "0",
                           "--no-guard", "user,load,disk"])
        r = rn.Runner(a)
        r.lab.runs.mkdir(parents=True, exist_ok=True)
        return r

    @staticmethod
    def py(name: str, code: str, **kw) -> pl.Cmd:
        return pl.Cmd(name, ["{py}", "-c", code], cwd="{lab}", **kw)

    def run_plan(self, r: rn.Runner, steps) -> rn.Run:
        pl.PLANS["fights"] = lambda c: steps
        run = r.new_run("fights", "abc")
        self.assertIsNone(r.execute(run))
        return run

    def lines(self, r: rn.Runner) -> dict[str, dict]:
        """The parsed accounting line of each command in the log (the last one when a command ran more than once)."""
        out = {}
        for x in r.lab.log_path.read_text(encoding="utf-8").splitlines():
            p = cr.parse_line(x)
            if p and p["out"] != "not started":
                out[p["key"]] = p
        return out


@unittest.skipUnless(WIN, "needs a Windows Job Object")
class AccountingTests(RunnerClockCase):
    def test_a_commands_cpu_reads_and_writes_are_on_its_log_line_and_in_job_json(self):
        big = self.tmp / "lab" / "big.bin"
        big.parent.mkdir(parents=True, exist_ok=True)
        big.write_bytes(b"x" * (8 << 20))
        code = ("import os; sum(range(2 * 10**7)); d = open('big.bin', 'rb').read(); open('out.bin', 'wb').write(d[:4 << 20]); print(len(d))")
        r = self.runner()
        run = self.run_plan(r, [pl.Step("a", [self.py("burn", code)], mode="serial")])
        log = [x for x in r.lab.log_path.read_text(encoding="utf-8").splitlines() if "a/burn:" in x][0]
        p = cr.parse_line(log)
        self.assertEqual(0, p["rc"])
        self.assertGreater(p["cpu"], 0.2)  # twenty million additions
        self.assertGreaterEqual(p["read_mb"], 8.0)
        self.assertGreaterEqual(p["write_mb"], 4.0)
        self.assertGreater(p["reads"], 0)
        self.assertGreaterEqual(p["wall"], p["cpu"] * 0.5)
        self.assertEqual("?", p["out"])  # no stamp of its own
        e = run.rec("a")["cmds"]["burn"]
        self.assertEqual((p["cpu"], p["read_mb"], p["write_mb"]), (e["cpu_secs"], e["read_mb"], e["write_mb"]))
        self.assertTrue(log.split(": ", 1)[1].startswith("rc=0 in "))  # the old part of the line stands

    def test_a_childs_cpu_counts_for_the_command_that_started_it(self):
        code = "import subprocess, sys; subprocess.run([sys.executable, '-c', 'sum(range(3 * 10**7))'], check=True)"
        r = self.runner()
        self.run_plan(r, [pl.Step("a", [self.py("parent", code)], mode="serial")])
        self.assertGreater(self.lines(r)["a/parent"]["cpu"], 0.3)  # the parent itself only waits

    def test_commands_run_at_once_have_figures_of_their_own(self):
        r = self.runner()
        self.run_plan(r, [pl.Step("a", [self.py("busy", "sum(range(4 * 10**7))"), self.py("idle", "import time; time.sleep(1.0)")])])
        got = self.lines(r)
        self.assertGreater(got["a/busy"]["cpu"], 0.4)
        self.assertLess(got["a/idle"]["cpu"], 0.4)
        self.assertGreaterEqual(got["a/idle"]["wall"], 1.0)

    def test_a_command_the_job_cannot_nest_has_a_line_without_figures(self):
        r = self.runner()
        r.account_job = lambda proc: None  # type: ignore[method-assign]
        self.run_plan(r, [pl.Step("a", [self.py("one", "print(1)")], mode="serial")])
        text = [x for x in r.lab.log_path.read_text(encoding="utf-8").splitlines() if "a/one:" in x][0]
        self.assertIn("cpu=? reads=? read_mb=? write_mb=? out=?", text)
        p = cr.parse_line(text)
        self.assertEqual((p["rc"], p["cpu"], p["reads"]), (0, None, None))

    def test_the_job_object_reports_cpu_and_io_for_a_finished_process_and_its_child(self):
        code = textwrap.dedent(f"""
            import subprocess, sys
            sys.path.insert(0, {str(ROOT)!r})
            from tools.lab import winsys
            outer = winsys.JobObject(); outer.assign_self()
            step = winsys.JobObject()
            p = winsys.suspended_popen([sys.executable, "-c",
                "import subprocess, sys; open(sys.argv[0] + 'x', 'wb').write(b'y' * 3000000); "
                "subprocess.run([sys.executable, '-c', 'sum(range(2*10**7))'])"], step)
            mine = winsys.JobObject(kill_on_close=False)
            mine.assign_handle(int(p._handle))
            p.wait()
            a = mine.accounting()
            print(a is not None and a["cpu_secs"] > 0.2, a["write_bytes"] >= 3000000, a["read_ops"] > 0, mine.accounting() == a)
        """)
        import subprocess
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False, cwd=self.tmp)
        self.assertEqual(out.stdout.split(), ["True"] * 4, out.stdout + out.stderr)

    def test_accounting_is_none_off_windows_or_for_a_closed_job(self):
        j = winsys.JobObject()
        j.close()
        self.assertIsNone(j.accounting())


@unittest.skipUnless(WIN, "needs a Windows Job Object")
class MemoryCapTests(RunnerClockCase):
    """A command's `mem_share` (the mutation job's runaway mutant, 2026-11-05: one engine process held tens of GB until a timer killed
    it): its processes, children included, may commit that share of the RAM together, and the line says what they peaked at."""

    ALLOC = "import sys\ntry:\n    b = bytearray(int(sys.argv[1]) << 20)\n    print('allocated')\nexcept MemoryError:\n    print('refused')\n"

    @staticmethod
    def share_of(mb: int) -> float:
        """The mem_share that makes `mb` MB of this machine's RAM."""
        return (mb << 20) / winsys.total_ram_bytes()

    def alloc_cmd(self, name: str, ask_mb: int, cap_mb: int | None) -> pl.Cmd:
        kw = {"mem_share": self.share_of(cap_mb)} if cap_mb else {}
        return pl.Cmd(name, ["{py}", "-c", self.ALLOC, str(ask_mb)], cwd="{lab}", **kw)

    def test_ram_share_is_a_fraction_of_the_ram_the_os_reports(self):
        self.assertEqual(winsys.ram_share(0.5, ram=32 << 30), 16 << 30)
        self.assertIsNone(winsys.ram_share(0, ram=32 << 30))
        self.assertIsNone(winsys.ram_share(0.5, ram=0))
        self.assertGreater(winsys.total_ram_bytes(), 1 << 30)

    def test_a_command_cannot_commit_more_than_its_share_and_one_under_it_runs_as_before(self):
        r = self.runner()
        run = self.run_plan(r, [pl.Step("a", [self.alloc_cmd("under", 32, 400), self.alloc_cmd("over", 700, 400),
                                              self.alloc_cmd("free", 32, None)])])
        said = {n: (run.dir / "logs" / f"{n}.log").read_text(encoding="utf-8").split()[-1] for n in ("under", "over", "free")}
        self.assertEqual({"under": "allocated", "over": "refused", "free": "allocated"}, said)
        got = self.lines(r)
        for n in ("under", "over"):  # the line stays what clock_report reads, and gains the peak
            text = [x for x in r.lab.log_path.read_text(encoding="utf-8").splitlines() if f"a/{n}:" in x][0]
            self.assertRegex(text, r"peak_mb=\d+$")
            self.assertIsNotNone(got[f"a/{n}"]["cpu"])
            self.assertEqual("?", got[f"a/{n}"]["out"])
        entries = run.rec("a")["cmds"]
        self.assertLessEqual(entries["under"]["peak_mb"], 400)
        self.assertGreater(entries["under"]["peak_mb"], 30)  # an interpreter and 32 MB
        self.assertGreater(entries["over"]["peak_mb"], 30)  # (Windows counts the request it refused in the job's peak)
        self.assertNotIn("peak_mb", entries["free"])
        self.assertNotIn("peak_mb", [x for x in r.lab.log_path.read_text(encoding="utf-8").splitlines() if "a/free:" in x][0])

    def test_a_childs_memory_counts_against_its_parents_share(self):
        code = ("import subprocess, sys; "
                f"print(subprocess.run([sys.executable, '-c', {self.ALLOC!r}, '700'], capture_output=True, text=True).stdout.split()[-1])")
        r = self.runner()
        run = self.run_plan(r, [pl.Step("a", [pl.Cmd("parent", ["{py}", "-c", code], cwd="{lab}", mem_share=self.share_of(400))],
                                        mode="serial")])
        self.assertEqual("refused", (run.dir / "logs" / "parent.log").read_text(encoding="utf-8").split()[-1])

    def test_a_host_in_its_own_job_nests_inside_the_commands_cap(self):
        # What mut-schemata does: the command (capped) starts a host with a limit of its own, under the command's.
        code = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(ROOT)!r})
            from tools.fidelity import mutate as mu
            for ask in (32, 700):
                p = mu.popen([sys.executable, '-c', {self.ALLOC!r}, str(ask)], 400 << 20)
                print(p.communicate()[0].decode().split()[-1], p.returncode)
        """)
        r = self.runner()
        run = self.run_plan(r, [pl.Step("a", [pl.Cmd("parent", ["{py}", "-c", code], cwd="{lab}", mem_share=self.share_of(2000))],
                                        mode="serial")])
        self.assertEqual(["allocated", "0", "refused", "0"], (run.dir / "logs" / "parent.log").read_text(encoding="utf-8").split()[-4:])

    def test_a_command_whose_cap_cannot_be_set_is_not_started(self):
        r = self.runner()

        def refuse(cmd):
            raise OSError("no job")

        r.memory_cap = refuse  # type: ignore[method-assign]
        pl.PLANS["fights"] = lambda c: [pl.Step("a", [pl.Cmd("x", ["{py}", "-c", "open('ran.txt', 'w').write('1')"], cwd="{lab}",
                                                          mem_share=0.5),
                                                      self.py("y", "print(1)")], mode="serial")]
        run = r.new_run("fights", "abc")
        r.execute(run)
        x = run.rec("a")["cmds"]["x"]
        self.assertEqual(("failed", 127), (x["status"], x["rc"]))
        self.assertIn("could not start", x["note"])
        self.assertFalse((r.lab.root / "ran.txt").exists())
        self.assertNotIn("y", run.rec("a")["cmds"])  # a serial step stops at its first failure

    def test_a_command_with_no_share_gets_no_cap(self):
        self.assertIsNone(rn.Runner.memory_cap(pl.Cmd("x", ["a"])))
        j = rn.Runner.memory_cap(pl.Cmd("x", ["a"], mem_share=0.25))
        self.addCleanup(j.close)
        self.assertEqual(j.peak_memory()["job"], 0)  # a limited job nobody has run in yet


class OutputVerdictTests(RunnerClockCase):
    """`out=` on the line: from the command's skip stamp, which the command (or its post hook) writes."""

    def stamp_cmd(self, write_stamp: bool, text: str, name: str = "stepx") -> pl.Cmd:
        code = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(ROOT)!r})
            from pathlib import Path
            from tools.lab import stepcache as sc
            Path('out.txt').write_text({text!r})
            if {write_stamp!r}:
                sc.write_stamp(Path('.'), {name!r}, 'digest', [Path('out.txt')])
        """)
        return self.py(name, code)

    def out_of(self, r: rn.Runner, key: str = "a/stepx") -> str:
        return self.lines(r)[key]["out"]

    def test_the_output_is_changed_then_same_then_changed_and_skipped_when_the_stamp_stands(self):
        r = self.runner()
        seen = []
        for write, text in ((True, "A"), (True, "A"), (True, "B"), (False, "B")):
            self.run_plan(r, [pl.Step("a", [self.stamp_cmd(write, text)], mode="serial")])
            seen.append(self.out_of(r))
        self.assertEqual(["changed", "same", "changed", "skipped"], seen)

    def test_a_command_with_no_stamp_or_a_failed_one_says_a_question_mark(self):
        r = self.runner()
        self.run_plan(r, [pl.Step("a", [self.py("plain", "print(1)"), self.py("bad", "import sys; sys.exit(1)", ok_rcs=(0, 1))], mode="serial")])
        got = self.lines(r)
        self.assertEqual(("?", "?"), (got["a/plain"]["out"], got["a/bad"]["out"]))

    def test_a_failure_with_a_stamp_written_is_still_a_question_mark(self):
        r = self.runner()
        code = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(ROOT)!r})
            from pathlib import Path
            from tools.lab import stepcache as sc
            Path('out.txt').write_text('A')
            sc.write_stamp(Path('.'), 'stepy', 'd', [Path('out.txt')])
            sys.exit(1)
        """)
        self.run_plan(r, [pl.Step("a", [self.py("stepy", code)], mode="serial")])
        self.assertEqual("?", self.out_of(r, "a/stepy"))


class DailyClockTests(RunnerClockCase):
    def stale_log(self, r: rn.Runner) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 3600))
        with r.lab.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line(stamp, "x/waits", 1000, 100) + "\n" + line(stamp, "x/waits", 1000, 100) + "\n")

    def test_the_table_is_written_once_a_day_and_a_flagged_step_is_logged(self):
        r = self.runner()
        self.stale_log(r)
        clock = r.lab.root / "clock.md"
        r.daily_clock()
        self.assertIn("x/waits", clock.read_text(encoding="utf-8"))
        self.assertIn("clock: 1 step(s) flagged over the last 7 days (clock.md): x/waits IDLE", r.lab.log_path.read_text(encoding="utf-8"))
        clock.write_text("kept", encoding="utf-8")
        r.daily_clock()
        self.assertEqual("kept", clock.read_text(encoding="utf-8"))  # less than a day old: left alone
        old = time.time() - rn.CLOCK_EVERY_S - 60
        os.utime(clock, (old, old))
        r.daily_clock()
        self.assertIn("x/waits", clock.read_text(encoding="utf-8"))

    def test_a_fault_is_a_log_line_and_a_runner_that_does_not_stay_writes_nothing(self):
        r = self.runner()
        self.stale_log(r)
        orig = cr.report_for
        cr.report_for = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        self.addCleanup(setattr, cr, "report_for", orig)
        r.daily_clock()
        self.assertIn("clock: report not written (RuntimeError)", r.lab.log_path.read_text(encoding="utf-8"))
        cr.report_for = orig
        r.a.once = "presort"
        r.daily_clock()
        self.assertFalse((r.lab.root / "clock.md").exists())

    def test_one_pass_asks_for_it_between_jobs(self):
        r = self.runner()
        self.stale_log(r)
        r.a.trial = None
        r.a.once = None
        r.a.dry_run = False
        calls = []
        r.daily_clock = lambda: calls.append(1)  # type: ignore[method-assign]
        r.sleep_idle = lambda *a, **k: True  # type: ignore[method-assign]
        r.main_sha = lambda: None  # type: ignore[method-assign]
        r.one_pass()
        self.assertEqual([1], calls)


class StampChangedTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.lab = Path(self._td.name) / "lab"
        self.out = self.lab / "runs" / "j" / "out.json"
        self.out.parent.mkdir(parents=True)

    def stamp(self, text: str, **kw) -> dict:
        self.out.write_text(text, encoding="utf-8")
        sc.write_stamp(self.lab, "s", "d", [self.out], **kw)
        return sc.read_stamp(self.lab, "s")

    def test_first_stamp_changed_then_same_then_changed(self):
        self.assertTrue(self.stamp("A")["changed"])
        again = self.stamp("A")
        self.assertIs(again["changed"], False)
        self.assertEqual(again["content"], {"runs/j/out.json": sc.content_hash(self.out)})
        self.assertTrue(self.stamp("B")["changed"])

    def test_the_skip_decision_does_not_look_at_content(self):
        self.stamp("A")
        self.assertIsNotNone(sc.skip_reason(self.lab, "s", "d", [self.out]))
        self.assertIsNone(sc.skip_reason(self.lab, "s", "other", [self.out]))

    def test_content_given_by_the_writer_replaces_the_files_own_hash(self):
        label = "runs/j/out.json"
        self.stamp("time 1", content={label: "same groups"})
        self.assertIs(self.stamp("time 2", content={label: "same groups"})["changed"], False)  # a file with a clock inside
        self.assertTrue(self.stamp("time 3", content={label: "other groups"})["changed"])

    def test_a_stamp_from_before_the_field_cannot_say_and_an_output_too_big_to_hash_cannot_either(self):
        self.stamp("A")
        p = sc.stamp_path(self.lab, "s")
        doc = json.loads(p.read_text(encoding="utf-8"))
        del doc["content"], doc["changed"]
        p.write_text(json.dumps(doc), encoding="utf-8")
        self.assertIsNone(self.stamp("A")["changed"])
        old = sc.CONTENT_CAP
        sc.CONTENT_CAP = 1
        self.addCleanup(setattr, sc, "CONTENT_CAP", old)
        self.assertIsNone(self.stamp("AA")["changed"])

    def test_a_folder_or_a_missing_output_has_no_content(self):
        self.assertIsNone(sc.content_hash(self.out.parent))
        self.assertIsNone(sc.content_hash(self.out.parent / "nothing.json"))


def line(t: str, key: str, wall: float, cpu: float | None = None, out: str = "?", rc: int = 0, reads: int = 100) -> str:
    base = f"{t} {key}: rc={rc} in {wall}s"
    if cpu is None:
        return base
    return f"{base} cpu={cpu}s reads={reads} read_mb=5.0 write_mb=1.0 out={out}" + ("" if rc == 0 else " (FAILED)")


NOW = cr._ts("2026-11-02T12:00:00Z")


def day(n: int) -> str:
    """A log time n hours after 100 hours before NOW: inside the 7-day window for n from 0 to 260 (a day or two of lines), outside for -1000."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(NOW + (n - 100) * 3600))


class ClockReportTests(unittest.TestCase):
    def analyse(self, *lines: str, days: int = 7) -> dict:
        return cr.analyse("\n".join(lines) + "\n", NOW, days)

    def test_the_new_line_the_old_line_the_skip_line_and_the_failed_line_parse(self):
        p = cr.parse_line(line("2026-11-02T04:22:04Z", "groups/groups", 1727.2, 412.3, "same"))
        self.assertEqual((p["key"], p["rc"], p["wall"], p["cpu"], p["reads"], p["read_mb"], p["write_mb"], p["out"]),
                         ("groups/groups", 0, 1727.2, 412.3, 100.0, 5.0, 1.0, "same"))
        o = cr.parse_line("2026-10-20T04:22:04Z checks/refset: rc=0 in 69.4s")
        self.assertEqual((o["wall"], o["cpu"], o["out"]), (69.4, None, "?"))
        q = cr.parse_line("2026-11-02T03:19:31Z game/game-copy: skipped (game copy exists)")
        self.assertEqual((q["out"], q["rc"]), ("not started", None))
        f = cr.parse_line(line("2026-11-02T04:22:04Z", "kits/kits", 5.0, 1.0, "?", rc=1))
        self.assertEqual(f["rc"], 1)
        u = cr.parse_line("2026-11-02T04:22:04Z batch-03/scen-03: rc=0 in 5.0s cpu=? reads=? read_mb=? write_mb=? out=?")
        self.assertEqual((u["cpu"], u["reads"], u["key"]), (None, None, "batch-N/scen-N"))
        for other in ("2026-11-02T03:19:31Z resuming (was held by: load)", "step score-real stopped: load", ""):
            self.assertIsNone(cr.parse_line(other))

    def test_wall_more_than_three_times_cpu_is_flagged_idle_and_three_times_is_not(self):
        r = self.analyse(line(day(10), "x/waits", 1000, 100), line(day(11), "x/waits", 1000, 100),
                         line(day(10), "x/exact", 900, 300), line(day(11), "x/exact", 900, 300),
                         line(day(10), "x/busy", 1000, 2000))
        self.assertEqual(["IDLE"], r["by_step"]["x/waits"]["flags"])
        self.assertEqual(10.0, r["by_step"]["x/waits"]["wall_over_cpu"])
        self.assertEqual([], r["by_step"]["x/exact"]["flags"])  # exactly 3x is not more than 3x
        self.assertEqual([], r["by_step"]["x/busy"]["flags"])  # more CPU than wall: several cores
        self.assertEqual(["x/waits"], r["flagged"])

    def test_a_short_step_that_waits_is_not_flagged_and_a_step_without_cpu_figures_cannot_be(self):
        r = self.analyse(line(day(10), "x/tiny", 60, 1), line(day(10), "x/oldline", 3000))
        self.assertEqual([], r["by_step"]["x/tiny"]["flags"])  # under five minutes of wall in all
        self.assertEqual(["LONG"], r["by_step"]["x/oldline"]["flags"])  # wall alone can still say LONG
        self.assertIsNone(r["by_step"]["x/oldline"]["wall_over_cpu"])
        self.assertEqual(1, r["steps_with_cpu"])

    def test_the_output_the_same_five_runs_running_is_flagged_and_four_or_a_change_is_not(self):
        same = [line(day(i), "g/groups", 100, 90, "same") for i in range(1, 6)]
        r = self.analyse(*same)
        self.assertEqual(["SAME5"], r["by_step"]["g/groups"]["flags"])
        self.assertEqual(["same"] * 5, r["by_step"]["g/groups"]["last_outputs"])
        self.assertEqual([], self.analyse(*same[1:])["by_step"]["g/groups"]["flags"])  # four
        broken = same[:2] + [line(day(3), "g/groups", 100, 90, "changed")] + same[3:]
        self.assertEqual([], self.analyse(*broken)["by_step"]["g/groups"]["flags"])  # a change inside the last five
        old_change = [line(day(0), "g/groups", 100, 90, "changed")] + same  # a change before them
        self.assertEqual(["SAME5"], self.analyse(*old_change)["by_step"]["g/groups"]["flags"])

    def test_runs_that_ended_on_their_stamp_or_could_not_tell_do_not_count_as_the_same_or_as_wall(self):
        mixed = [line(day(i), "g/groups", 100, 90, "same") for i in range(1, 5)]
        mixed += [line(day(5), "g/groups", 3, 1, "skipped"), line(day(6), "g/groups", 100, 90, "?"), line(day(7), "g/groups", 100, 90, "same")]
        r = self.analyse(*mixed)["by_step"]["g/groups"]
        self.assertEqual(["SAME5"], r["flags"])  # five that could tell, the skipped one and the '?' between them ignored
        self.assertEqual((r["runs"], r["work_runs"], r["self_skipped"]), (7, 6, 1))
        self.assertEqual(100.0, r["wall_mean_s"])  # the 3-second skipped run is not in the mean

    def test_a_mean_wall_above_twenty_minutes_is_flagged_and_twenty_exactly_is_not(self):
        r = self.analyse(line(day(10), "x/long", 1500, 1400), line(day(11), "x/long", 1000, 900),
                         line(day(10), "x/edge", 1200, 1100), line(day(10), "x/short", 900, 800), line(day(11), "x/short", 1400, 1300))
        self.assertEqual(["LONG"], r["by_step"]["x/long"]["flags"])  # (1500 + 1000) / 2 = 1250
        self.assertEqual([], r["by_step"]["x/edge"]["flags"])  # twenty minutes exactly
        self.assertEqual([], r["by_step"]["x/short"]["flags"])  # (900 + 1400) / 2 = 1150

    def test_the_jobs_own_long_work_is_marked_and_not_an_alarm_unless_it_also_waits(self):
        r = self.analyse(line(day(10), "train/train", 4000, 3900), line(day(10), "selfplay/selfplay", 4000, 500), line(day(10), "x/long", 4000, 3900))
        self.assertEqual((r["by_step"]["train/train"]["flags"], r["by_step"]["train/train"]["alarm"]), (["LONG"], False))
        self.assertEqual((r["by_step"]["selfplay/selfplay"]["flags"], r["by_step"]["selfplay/selfplay"]["alarm"]), (["IDLE", "LONG"], True))
        self.assertTrue(r["by_step"]["x/long"]["alarm"])
        self.assertEqual(2, r["alarms"])
        self.assertEqual(["selfplay/selfplay", "x/long", "train/train"], r["flagged"])  # the alarms first, by wall

    def test_the_window_cuts_the_log_and_numbered_batches_are_one_step(self):
        r = self.analyse(line(day(-1000), "x/old", 3000, 100), line(day(10), "batch-00/scen-00", 100, 90), line(day(10), "batch-05/scen-05", 200, 190),
                         "2026-11-02T13:00:00Z x/future: rc=0 in 3000.0s", days=7)
        self.assertNotIn("x/old", r["by_step"])
        self.assertNotIn("x/future", r["by_step"])
        self.assertEqual(2, r["by_step"]["batch-N/scen-N"]["runs"])
        self.assertEqual(150.0, r["by_step"]["batch-N/scen-N"]["wall_mean_s"])
        wide = self.analyse(line(day(-500), "x/old", 3000, 100), line(day(-499), "x/old", 3000, 100), days=30)
        self.assertEqual(2, wide["by_step"]["x/old"]["runs"])

    def test_the_table_lists_the_flagged_steps_worst_first_and_all_with_the_option(self):
        lines = [line(day(10), "x/waits", 1000, 100), line(day(11), "x/waits", 1000, 100), line(day(10), "x/fine", 100, 90)]
        r = self.analyse(*lines)
        text = cr.render(r)
        self.assertIn("1 flagged", text)
        self.assertIn("x/waits", text)
        self.assertNotIn("x/fine", text)
        self.assertIn("IDLE", text)
        self.assertIn("x/fine", cr.render(r, show_all=True))
        self.assertIn("nothing flagged", cr.render(self.analyse(line(day(10), "x/fine", 100, 90))))
        self.assertIn("no CPU figures yet", cr.render(self.analyse(line(day(10), "x/old", 100))))

    def test_the_command_line_prints_the_table_and_writes_the_json(self):
        td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, td, True)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 3600))
        (td / "runner.log").write_text(line(stamp, "x/waits", 1000, 100) + "\n" + line(stamp, "x/waits", 1000, 100) + "\n", encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cr.main(["--lab", str(td), "--json", str(td / "clock.json")])
        self.assertEqual(0, rc)
        self.assertIn("x/waits", buf.getvalue())
        self.assertEqual(["x/waits"], json.loads((td / "clock.json").read_text(encoding="utf-8"))["flagged"])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(2, cr.main(["--lab", str(td / "nowhere")]))


class ReadinessClockTests(unittest.TestCase):
    def test_the_flagged_steps_ride_in_readiness_under_lab_clock_and_in_its_text(self):
        td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, td, True)
        lab = td / "lab"
        lab.mkdir()
        (lab / "groups.json").write_text(json.dumps({"groups": [], "main_sha": "a" * 40}), encoding="utf-8")
        (lab / "runner.log").write_text(line(day(10), "x/waits", 1000, 100) + "\n" + line(day(11), "x/waits", 1000, 100) + "\n", encoding="utf-8")
        r = rd.build(td, now=NOW)
        ck = r["lab"]["clock"]
        self.assertEqual((ck["alarms"], list(ck["flagged"])), (1, ["x/waits"]))
        self.assertEqual(ck["flagged"]["x/waits"]["flags"], ["IDLE"])
        self.assertIn("lab clock", rd.render(r))
        self.assertIn("x/waits IDLE", rd.render(r))

    def test_without_a_runner_log_readiness_has_no_clock(self):
        td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, td, True)
        (td / "lab").mkdir()
        (td / "lab" / "groups.json").write_text(json.dumps({"groups": [], "main_sha": "a" * 40}), encoding="utf-8")
        self.assertNotIn("clock", rd.build(td, now=NOW)["lab"])


if __name__ == "__main__":
    unittest.main()
