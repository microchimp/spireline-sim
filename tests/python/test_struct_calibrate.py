"""struct_calibrate's switch set and its verdict against an earlier run (docs/rebuild/COMPARER.md, "Core2 on the traced set").

No emulator runs here: the pool is replaced by a recorder, so these tests check what the command hands the workers (the scratch
build with the switch set, the migrated families the comparer judges with) and how two runs of one set are compared."""

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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from tools.fidelity import struct_calibrate as cal  # noqa: E402
from tools.fidelity import struct_compare as sc  # noqa: E402
from test_struct_compare import ROWS, fight, snap  # noqa: E402


def run_main(argv: list[str], fake_results=None) -> tuple[int, str, list]:
    """main(argv) with the pool replaced: (exit code, stderr, the jobs the workers would have got)."""
    seen: list = []

    def fake(jobs, procs):
        seen.extend(jobs)
        return [dict(r) for r in (fake_results or [])]

    err = io.StringIO()
    with mock.patch.object(cal, "run_jobs", fake), contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc = cal.main(argv)
    return rc, err.getvalue(), seen


class Switches(unittest.TestCase):
    def test_no_flag_leaves_the_build_to_its_own_file(self):
        for engine in ("old", "core2"):
            self.assertIsNone(cal.resolve_switches(engine, None))

    def test_a_core2_set_is_parsed_as_the_fuzzer_parses_it(self):
        self.assertEqual(cal.resolve_switches("core2", "F10, F08"), ["F08", "F10"])
        self.assertEqual(cal.resolve_switches("core2", "none"), [])
        self.assertEqual(cal.resolve_switches("core2", "all"), [f"F{n:02d}" for n in range(1, 18)])

    def test_an_unknown_family_is_refused(self):
        with self.assertRaises(ValueError):
            cal.resolve_switches("core2", "F08,F99")

    def test_old_has_no_switches_so_the_flag_is_refused_not_ignored(self):
        with self.assertRaises(ValueError):
            cal.resolve_switches("old", "F08,F10")
        with self.assertRaises(ValueError):
            cal.resolve_switches("old", "none")


class Command(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = Path(self.tmp.name)
        self.bin = t / "bin"
        self.bin.mkdir()
        (self.bin / "EmuDump.dll").write_bytes(b"dll")
        (self.bin / "migrated_families.json").write_text('{"migrated": []}', encoding="utf-8")
        self.traced = t / "traced.out.json"
        self.traced.write_text(json.dumps({"results": [{"seed": "S", "ascension": 10, "steps": [{"net": [{}]}]}]}), encoding="utf-8")
        self.out = t / "o" / "calib.json"

    def argv(self, *extra: str) -> list[str]:
        return ["traced", "--emudump", str(self.bin / "EmuDump.dll"), "--out", str(self.out), str(self.traced), *extra]

    def test_a_switch_set_runs_on_a_scratch_build_that_is_removed_afterwards(self):
        seen_dirs: list = []
        real = cal.run_jobs

        def spy(jobs, procs):
            seen_dirs.extend(Path(j[2]).parent for j in jobs)
            return []

        with mock.patch.object(cal, "run_jobs", spy), contextlib.redirect_stdout(io.StringIO()):
            rc = cal.main(self.argv("--engine", "core2", "--switches", "F08,F10"))
        self.assertEqual(rc, 0)
        self.assertEqual(len(seen_dirs), 1)
        scratch = seen_dirs[0]
        self.assertNotEqual(scratch.resolve(), self.bin.resolve())
        self.assertTrue(scratch.name.startswith("bin-F08-F10-"))
        self.assertFalse(scratch.exists(), "the scratch build is dropped when the command ends")
        # the build itself was never touched
        self.assertEqual(json.loads((self.bin / "migrated_families.json").read_text()), {"migrated": []})
        self.assertIs(cal.run_jobs, real)

    def test_the_workers_get_the_set_beside_the_dll_and_the_comparer_gets_it_as_migrated(self):
        beside: list = []

        def spy(jobs, procs):
            for j in jobs:
                beside.append(json.loads((Path(j[2]).parent / "migrated_families.json").read_text()))
            seen.extend(jobs)
            return []

        seen: list = []
        with mock.patch.object(cal, "run_jobs", spy), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cal.main(self.argv("--engine", "core2", "--switches", "F10,F08")), 0)
        self.assertEqual(beside, [{"migrated": ["F08", "F10"]}])
        job = seen[0]
        self.assertEqual((job[6], job[7]), ("core2", ["F08", "F10"]))
        doc = json.loads(self.out.read_text())
        self.assertEqual((doc["engine"], doc["switches"]), ("core2", ["F08", "F10"]))

    def test_without_the_flag_the_build_itself_runs_and_the_comparer_gets_the_builds_own_set(self):
        # The build's own migrated_families.json is the set: the comparer judges core2 with it as migrated and also with the
        # shared rows (like with like against old), as with --switches; no scratch build is made.
        (self.bin / "migrated_families.json").write_text('{"migrated": ["F10"]}')
        rc, _, jobs = run_main(self.argv("--engine", "core2"))
        self.assertEqual(rc, 0)
        self.assertEqual(Path(jobs[0][2]), self.bin / "EmuDump.dll")
        self.assertEqual(jobs[0][7], ["F10"])
        self.assertEqual(json.loads(self.out.read_text())["switches"], ["F10"])

    def test_old_without_the_flag_judges_with_no_migrated_set(self):
        (self.bin / "migrated_families.json").write_text('{"migrated": ["F10"]}')
        rc, _, jobs = run_main(self.argv("--engine", "old"))
        self.assertEqual(rc, 0)
        self.assertIsNone(jobs[0][7])

    def test_the_set_equal_to_the_builds_own_runs_on_the_build(self):
        (self.bin / "migrated_families.json").write_text('{"migrated": ["F08", "F10"]}', encoding="utf-8")
        _, _, jobs = run_main(self.argv("--engine", "core2", "--switches", "F08,F10"))
        self.assertEqual(Path(jobs[0][2]), self.bin / "EmuDump.dll")
        self.assertTrue((self.bin / "EmuDump.dll").exists(), "the build the user named is never removed")

    def scratch_dirs(self) -> list:
        return sorted(p.name for p in self.out.parent.glob("bin-*"))

    def test_a_leftover_scratch_directory_is_never_reused(self):
        # an earlier run with this pid died: its directory (a stale DLL, a stale file, another switch set) must not be the run's build
        stale = self.out.parent / f"bin-F08-F10-{os.getpid()}"
        stale.mkdir(parents=True)
        (stale / "EmuDump.dll").write_bytes(b"stale")
        (stale / "stale.txt").write_text("x")
        (stale / "migrated_families.json").write_text('{"migrated": ["F01"]}', encoding="utf-8")
        seen: list = []

        def spy(jobs, procs):
            d = Path(jobs[0][2]).parent
            seen.append(((d / "EmuDump.dll").read_bytes(), (d / "stale.txt").exists(),
                         json.loads((d / "migrated_families.json").read_text())))
            return []

        with mock.patch.object(cal, "run_jobs", spy), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cal.main(self.argv("--engine", "core2", "--switches", "F08,F10")), 0)
        self.assertEqual(seen, [(b"dll", False, {"migrated": ["F08", "F10"]})])
        self.assertEqual(self.scratch_dirs(), [])

    def test_the_scratch_build_is_dropped_when_the_run_fails(self):
        def boom(jobs, procs):
            raise RuntimeError("worker died")

        with mock.patch.object(cal, "run_jobs", boom), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                cal.main(self.argv("--engine", "core2", "--switches", "F08,F10"))
        self.assertEqual(self.scratch_dirs(), [])

    def test_a_copy_that_fails_halfway_leaves_nothing_behind_and_the_error_surfaces(self):
        def half(src, dest, *a, **kw):
            Path(dest).mkdir(parents=True)
            (Path(dest) / "EmuDump.dll").write_bytes(b"half")
            raise OSError("disk full")

        ran: list = []
        with mock.patch.object(cal.FZ.shutil, "copytree", half), mock.patch.object(cal, "run_jobs", lambda j, p: ran.append(j) or []), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError):
                cal.main(self.argv("--engine", "core2", "--switches", "F08,F10"))
        self.assertEqual((self.scratch_dirs(), ran), ([], []))
        self.assertEqual((self.bin / "EmuDump.dll").read_bytes(), b"dll")  # the build itself is untouched

    def test_dll_for_removes_a_stale_directory_and_a_failed_switch_file_write(self):
        work = Path(self.tmp.name) / "w"
        stale = work / f"bin-F08-{os.getpid()}"
        stale.mkdir(parents=True)
        (stale / "old.txt").write_text("x")
        dll = cal.FZ.dll_for(self.bin / "EmuDump.dll", ["F08"], work)
        self.assertEqual(dll.parent, stale)
        self.assertFalse((stale / "old.txt").exists())
        self.assertEqual(json.loads((stale / "migrated_families.json").read_text()), {"migrated": ["F08"]})
        cal.FZ.drop_build(dll, self.bin / "EmuDump.dll")
        with mock.patch.object(Path, "write_text", side_effect=OSError("no space")):
            with self.assertRaises(OSError):
                cal.FZ.dll_for(self.bin / "EmuDump.dll", ["F08"], work)
        self.assertEqual(sorted(p.name for p in work.glob("bin-*")), [])

    def test_the_flag_with_old_exits_2_and_runs_nothing(self):
        rc, err, jobs = run_main(self.argv("--switches", "F08,F10"))
        self.assertEqual(rc, 2)
        self.assertIn("--engine core2", err)
        self.assertEqual(jobs, [])
        self.assertFalse(self.out.exists())

    def test_an_unknown_family_exits_2(self):
        rc, err, jobs = run_main(self.argv("--engine", "core2", "--switches", "F08,F99"))
        self.assertEqual((rc, jobs), (2, []))
        self.assertIn("F99", err)


def lost_fight():
    """A lost fight: old's row waives all but in_combat and the player's HP; compared in full once F10 is migrated."""
    rec = snap(in_combat=False, player={**snap()["player"], "hp": 0}, hand=[])
    eng = snap(in_combat=False, player={**snap()["player"], "hp": 0, "block": 9})
    return fight([rec], [eng])


class Judge(unittest.TestCase):
    def judge(self, migrated, engine="core2") -> dict:
        g, e = lost_fight()
        with mock.patch.object(cal.fc, "first_difference", return_value=None):
            return cal._judge("f", g, e, {}, ROWS, False, False, engine, migrated)

    def test_a_switch_set_judges_twice_with_the_migrated_rows_and_with_the_shared_ones(self):
        r = self.judge(["F10"])
        self.assertEqual(r["sc"], 0)           # F10's fight-lost row is no longer inherited: the block is compared
        self.assertGreater(r["sc_n"], 0)
        self.assertIsNone(r["sc_shared"])      # the rows old is judged by: nothing to flag
        self.assertEqual(r["sc_shared_n"], 0)

    def test_no_set_is_one_judgement(self):
        # None reads the repository's migrated_families.json (F08+F10 since the switch-on): held empty here, so the case is
        # the comparer with no migrated family whatever the file says.
        for migrated in (None, []):
            with mock.patch.object(cal.sc, "load_migrated", return_value=[]):
                r = self.judge(migrated)
            self.assertIsNone(r["sc"])
            self.assertNotIn("sc_shared", r)

    def test_old_never_loses_its_rows(self):
        self.assertIsNone(self.judge(["F10"], "old")["sc"])


def res(sc_step, sc_n=None, nd=None, **more) -> dict:
    return {"census": ["x"], "sc": sc_step, "sc_n": sc_n, **({"nd": nd} if nd is not None or "nd" in more else {}), **more}


class Verdicts(unittest.TestCase):
    def test_a_later_first_difference_is_better_an_earlier_one_worse(self):
        self.assertEqual(cal.verdict(res(3), res(5), "sc"), "better")
        self.assertEqual(cal.verdict(res(3), res(1), "sc"), "worse")
        self.assertEqual(cal.verdict(res(3), res(3), "sc"), "same")

    def test_no_difference_is_past_every_step(self):
        self.assertEqual(cal.verdict(res(30), res(None), "sc"), "better")
        self.assertEqual(cal.verdict(res(None), res(30), "sc"), "worse")
        self.assertEqual(cal.verdict(res(None), res(None), "sc"), "same")

    def test_at_one_step_fewer_differing_fields_is_better(self):
        self.assertEqual(cal.verdict(res(2, 5), res(2, 3), "sc", "sc_n"), "better")
        self.assertEqual(cal.verdict(res(2, 3), res(2, 5), "sc", "sc_n"), "worse")
        self.assertEqual(cal.verdict(res(2, 3), res(2, 3), "sc", "sc_n"), "same")

    def test_an_output_without_field_counts_is_judged_by_the_step_alone(self):
        self.assertEqual(cal.verdict(res(2, None), res(2, 9), "sc", "sc_n"), "same")
        self.assertEqual(cal.verdict(res(2, None), res(4, 9), "sc", "sc_n"), "better")

    def test_the_comparison_reads_the_shared_rows_of_the_new_run(self):
        base = {"a": res(4, 2), "b": res(4, 2)}
        mine = {"a": res(0, 7, sc_shared=4, sc_shared_n=2), "b": res(9, 1, sc_shared=2, sc_shared_n=1)}
        out = cal.compare_runs(base, mine)
        self.assertEqual(out["fights"]["a"]["sc"], [4, 4, "same"])    # the stricter rows flag step 0, the shared ones do not
        self.assertEqual(out["fights"]["b"]["sc"], [4, 2, "worse"])
        self.assertEqual(out["counts"]["sc"], {"better": 0, "same": 1, "worse": 1})
        self.assertEqual(cal.exposed(mine), 1)                         # "a": the migrated rows flag earlier than the shared ones

    def test_the_net_state_measure_counts_separately_and_a_missing_fight_is_reported(self):
        base = {"a": res(3, 1, nd=0), "b": res(3, 1, nd=0), "c": res(3, 1, nd=0)}
        mine = {"a": res(3, 1, nd=2), "b": res(3, 1, nd=0)}
        out = cal.compare_runs(base, mine)
        self.assertEqual(out["counts"]["nd"], {"better": 1, "same": 1, "worse": 0})
        self.assertEqual(out["missing"], ["c"])
        self.assertEqual(sum(out["counts"]["sc"].values()), 2)

    def test_a_result_without_a_census_is_not_compared(self):
        out = cal.compare_runs({"a": {"sc": None}}, {"a": res(1)})
        self.assertEqual(out["fights"], {})


class Gate(unittest.TestCase):
    """main(--gate-against): exit 1 for a worse fight or a missing one, 0 when none is worse."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = Path(self.tmp.name)
        (t / "bin").mkdir()
        (t / "bin" / "EmuDump.dll").write_bytes(b"dll")
        (t / "bin" / "migrated_families.json").write_text('{"migrated": []}', encoding="utf-8")
        self.traced = t / "traced.out.json"
        self.traced.write_text(json.dumps({"results": [{"seed": "S", "ascension": 10, "steps": [{"net": [{}]}]}]}), encoding="utf-8")
        self.base = t / "old.json"
        self.out = t / "new.json"
        self.fid = cal.traced_fight_id(str(self.traced), 0)
        self.t = t

    def run_gate(self, base_res: dict, new_res: dict) -> tuple[int, str]:
        self.base.write_text(json.dumps({"results": [{"id": self.fid, **base_res}]}), encoding="utf-8")
        argv = ["traced", "--engine", "core2", "--switches", "F08,F10", "--emudump", str(self.t / "bin" / "EmuDump.dll"),
                "--out", str(self.out), "--gate-against", str(self.base), str(self.traced)]
        out = io.StringIO()
        with mock.patch.object(cal, "run_jobs", lambda jobs, procs: [{"id": self.fid, **new_res}]), contextlib.redirect_stdout(out):
            rc = cal.main(argv)
        return rc, out.getvalue()

    def test_no_worse_fight_and_no_fewer_fields_passes_and_prints_the_counts(self):
        rc, text = self.run_gate(res(3, 1, nd=0), res(5, 1, nd=0, sc_shared=5, sc_shared_n=1))
        self.assertEqual(rc, 0)
        self.assertIn("against the game, struct_compare: 1 better, 0 same, 0 worse of 1 fights", text)
        self.assertIn("against the game, netstate_diff: 0 better, 1 same, 0 worse of 1 fights", text)
        doc = json.loads(self.out.read_text())
        self.assertEqual(doc["against_game"]["counts"]["sc"], {"better": 1, "same": 0, "worse": 0})
        self.assertEqual(doc["verdicts"][self.fid]["sc"], [3, 5, "better"])

    def test_a_worse_fight_fails_the_gate(self):
        rc, text = self.run_gate(res(5, 1, nd=0), res(2, 1, nd=0, sc_shared=2, sc_shared_n=1))
        self.assertEqual(rc, 1)
        self.assertIn("(worse)", text)

    def test_a_worse_net_state_alone_fails_the_gate(self):
        rc, _ = self.run_gate(res(5, 1, nd=7), res(5, 1, nd=1, sc_shared=5, sc_shared_n=1))
        self.assertEqual(rc, 1)

    def test_a_fight_compared_on_fewer_fields_fails_the_gate(self):
        rc, text = self.run_gate({**res(5, 1, nd=0), "census": ["x", "y"]}, res(5, 1, nd=0, sc_shared=5, sc_shared_n=1))
        self.assertEqual(rc, 1)
        self.assertIn("census gate: 1 of 1", text)


if __name__ == "__main__":
    unittest.main()
