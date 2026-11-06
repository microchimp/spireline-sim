"""rl/nightly_real.py and the `nightly-real` lab kind (D8, D33 addendum 2). No game, no emulator: `gen0.py score` is replaced."""

from __future__ import annotations

import calendar
import json
import os
import re
import shutil
import sys
import tempfile
import time
import types
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "rl"):
    sys.path.insert(0, str(p))

import gen0  # noqa: E402
import nightly_real as nr  # noqa: E402
from tools.lab import jobs as jb  # noqa: E402


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)


class QuietLockTest(Tmp):
    def lock(self, line: str | None, now: str = "2026-10-25T12:00:00Z", **kw):
        f = self.tmp / "QUIET.lock"
        if line is not None:
            f.write_text(line + "\n")
        t = [nr.parse_utc(now)]

        def sleep(s):
            t[0] += s

        return nr.QuietLock(f, clock=lambda: t[0], sleep=sleep, poll_s=600, **kw), f

    def test_waits_while_another_is_under_two_hours_old_and_skips_after_the_bounded_wait(self):
        lk, f = self.lock("w4-readout 2026-10-25T11:30:00Z", wait_h=0.5)
        self.assertFalse(lk.acquire())
        self.assertTrue(f.read_text().startswith("w4-readout"))  # the other's line is untouched

    def test_takes_the_lock_when_the_others_line_ages_out_during_the_wait(self):
        lk, f = self.lock("w4-readout 2026-10-25T10:30:00Z", wait_h=2.0)  # 1.5 h old: stale in 30 min
        self.assertTrue(lk.acquire())
        self.assertTrue(f.read_text().startswith("lab-nightly "))

    def test_takes_a_free_stale_or_own_lock_and_releases_only_its_own_line(self):
        for line in (None, "w4-x 2026-10-25T08:00:00Z", "lab-nightly 2026-10-25T11:59:00Z"):
            lk, f = self.lock(line)
            self.assertTrue(lk.acquire(), line)
            lk.release()
            self.assertFalse(f.exists())
        lk, f = self.lock("lab-nightly 2026-10-25T11:59:00Z")
        f.write_text("w4-y 2026-10-25T11:59:00Z\n")  # someone else took it meanwhile
        lk.release()
        self.assertTrue(f.exists())


class AgentsTest(Tmp):
    def root(self, rows: list[tuple[str, int, str, str]]) -> Path:
        r = self.tmp / "agent"
        for d in ("gen3/candidate-r1", "gen4/candidate", "gen4/candidate-r1"):
            (r / d).mkdir(parents=True)
        (r / "current.json").write_text(json.dumps({"generation": 3, "dir": str(r / "gen3" / "candidate-r1")}))
        head = "written\tcand_gen\tbase_gen\tseed_block\tseeds\tdecision\tcand_sha\tcand_dir\n"
        (r / "promotions.tsv").write_text(head + "".join(f"{w}\t{g}\t{g - 1}\tE\t2000\t{dec}\tsha\t{r / d}\n" for w, g, dec, d in rows))
        return r

    def test_incumbent_alone_without_a_last_nightly(self):
        r = self.root([("2026-10-25T01:00:00Z", 4, "promoted", "gen4/candidate")])
        got = nr.agents_to_score(r, None)
        self.assertEqual([("g3", "incumbent")], [(a["label"], a["role"]) for a in got])

    def test_promotions_since_the_last_nightly_are_scored_too_once_per_directory(self):
        r = self.root([("2026-10-24T01:00:00Z", 4, "promoted", "gen4/candidate"),   # before the last nightly
                       ("2026-10-25T01:00:00Z", 4, "rejected", "gen4/candidate"),   # not a promotion
                       ("2026-10-25T02:00:00Z", 4, "promoted", "gen4/candidate-r1"),
                       ("2026-10-25T03:00:00Z", 3, "promoted", "gen3/candidate-r1")])  # the incumbent's own directory
        since = nr.parse_utc("2026-10-24T12:00:00Z")
        got = nr.agents_to_score(r, since)
        self.assertEqual(["g3", "g4"], [a["label"] for a in got])
        self.assertEqual("promoted", got[1]["role"])
        self.assertEqual("N-2026-10-25", nr.seed_block_name("2026-10-25", got[0]))
        self.assertEqual("N-2026-10-25-g4", nr.seed_block_name("2026-10-25", got[1]))

    def rows_as_promote_writes_them(self, rows: list[tuple[str, int, str, str]]) -> Path:
        """promotions.tsv as gen_loop.promote writes it: cand_dir is the candidate folder's name, not a path."""
        r = self.root([])
        head = "written\tcand_gen\tbase_gen\tseed_block\tseeds\tdecision\tcand_sha\tcand_dir\n"
        (r / "promotions.tsv").write_text(head + "".join(f"{w}\t{g}\t{g - 1}\tE\t2000\t{dec}\tsha\t{d}\n" for w, g, dec, d in rows))
        return r

    def test_a_folder_name_in_cand_dir_is_the_folder_under_its_generation(self):
        r = self.rows_as_promote_writes_them([("2026-10-25T02:00:00Z", 4, "promoted", "candidate-r1")])
        got = nr.agents_to_score(r, nr.parse_utc("2026-10-24T12:00:00Z"))
        self.assertEqual([("g3", "incumbent"), ("g4", "promoted")], [(a["label"], a["role"]) for a in got])
        self.assertEqual((r / "gen4" / "candidate-r1").resolve(), Path(got[1]["dir"]).resolve())
        self.assertEqual(2000, nr.night_runs(got, 500, 2000))

    def test_an_incumbent_promoted_since_the_last_nightly_makes_a_promotion_night(self):
        # the usual promotion day: the promoted generation is the incumbent, so it is scored once, with the promotion night's runs
        r = self.rows_as_promote_writes_them([("2026-10-25T03:00:00Z", 3, "promoted", "candidate-r1")])
        got = nr.agents_to_score(r, nr.parse_utc("2026-10-24T12:00:00Z"))
        self.assertEqual([("g3", "incumbent")], [(a["label"], a["role"]) for a in got])
        self.assertTrue(got[0].get("promoted_since"))
        self.assertEqual(["g3"], nr.promoted_labels(got))
        self.assertEqual(2000, nr.night_runs(got, 500, 2000))
        self.assertEqual("N-2026-10-25", nr.seed_block_name("2026-10-25", got[0]))  # the incumbent's block name is unchanged

    def test_an_incumbent_promoted_before_the_last_nightly_is_an_ordinary_night(self):
        r = self.rows_as_promote_writes_them([("2026-10-24T03:00:00Z", 3, "promoted", "candidate-r1"),
                                              ("2026-10-25T03:00:00Z", 4, "rejected", "candidate")])
        got = nr.agents_to_score(r, nr.parse_utc("2026-10-24T12:00:00Z"))
        self.assertFalse(got[0].get("promoted_since"))
        self.assertEqual([], nr.promoted_labels(got))
        self.assertEqual(500, nr.night_runs(got, 500, 2000))

    def test_the_full_night_only_when_a_generation_was_promoted(self):
        inc, prom = {"label": "g3", "role": "incumbent"}, {"label": "g4", "role": "promoted"}
        self.assertEqual(500, nr.night_runs([inc], 500, 2000))
        self.assertEqual(2000, nr.night_runs([inc, prom], 500, 2000))
        self.assertEqual(500, nr.night_runs([inc, prom], 500, None))  # without --runs-promoted every night plays --runs


class BlockTest(Tmp):
    def test_a_fresh_block_is_the_sha256_rule_written_under_lab_seeds(self):
        path, seeds = nr.make_block(self.tmp, "N-2026-10-25", 50, [])
        self.assertEqual(self.tmp / "seeds" / "N-2026-10-25.json", path)
        self.assertEqual(gen0.e_block("N-2026-10-25", 50), seeds)
        self.assertEqual(seeds, gen0.load_seeds(path))

    def test_a_shared_seed_with_any_other_block_refuses_it(self):
        mine = gen0.e_block("N-2026-10-25", 50)
        other = self.tmp / "other"
        other.mkdir()
        (other / "S9-2026-10-25.json").write_text(json.dumps({"seeds": mine[:1] + ["ZZZZZZZZZZ"]}))
        with self.assertRaises(ValueError):
            nr.make_block(self.tmp, "N-2026-10-25", 50, [other])
        self.assertFalse((self.tmp / "seeds" / "N-2026-10-25.json").exists())  # nothing is written for a refused block

    def test_a_name_that_is_not_an_n_block_is_refused(self):
        with self.assertRaises(ValueError):
            nr.make_block(self.tmp, "E9-2026-10-25", 5, [])

    def test_last_nights_block_is_not_reused_tonight(self):
        nr.make_block(self.tmp, "N-2026-10-24", 50, [])
        nr.make_block(self.tmp, "N-2026-10-25", 50, [])  # disjoint from the earlier N block, which sits in the same folder


def end(seed: str, floor: int, kind: str = "death") -> dict:
    return {"ev": "end", "seed": seed, "end": kind, "floor": floor, "s": 1.0, "stats": {"decisions": 10}}


class BlockNameTest(Tmp):
    def test_the_name_is_the_utc_start_to_the_minute(self):
        t = calendar.timegm((2026, 11, 6, 3, 12, 59))
        self.assertEqual("2026-11-06T0312", nr.block_name(t))
        self.assertRegex(nr.block_name(), r"^\d{4}-\d{2}-\d{2}T\d{4}$")
        self.assertNotEqual(nr.block_name(t), nr.block_name(t + 60))  # two blocks of a day do not collide

    def test_dates_and_start_times_are_both_block_folders_and_nothing_else_is(self):
        for ok in ("2026-11-04", "2026-11-06T0312", "2026-11-06T2359"):
            self.assertTrue(nr.is_block_folder(ok), ok)
        for bad in ("last-ok", "active", "pending", "2026-11-06T03", "2026-11-06T031", "2026-11-06T03:12", "2026-11-06-g4", "scratch", "x2026-11-06"):
            self.assertFalse(nr.is_block_folder(bad), bad)

    def test_folder_names_sort_in_time_order_across_both_forms(self):
        names = ["2026-11-06T0415", "2026-11-05", "2026-11-06", "2026-11-06T0300", "2026-11-07"]
        self.assertEqual(["2026-11-05", "2026-11-06", "2026-11-06T0300", "2026-11-06T0415", "2026-11-07"], sorted(names))

    def test_trigger_is_the_marker_unless_forced(self):
        lab = self.tmp / "lab"
        self.assertEqual("clock", nr.trigger_of(lab))
        (lab / "nightly").mkdir(parents=True)
        (lab / "nightly" / "pending").write_text("x\n")
        self.assertEqual("promotion", nr.trigger_of(lab))
        self.assertEqual("clock", nr.trigger_of(lab, "clock"))
        (lab / "nightly" / "pending").unlink()
        self.assertEqual("promotion", nr.trigger_of(lab, "promotion"))

    def test_promotion_agents_drop_an_incumbent_that_was_not_promoted_since(self):
        inc = {"label": "g3", "role": "incumbent"}
        prom = {"label": "g4", "role": "promoted"}
        self.assertEqual(["g4"], [a["label"] for a in nr.promotion_agents([inc, prom])])
        self.assertEqual(["g3", "g4"], [a["label"] for a in nr.promotion_agents([{**inc, "promoted_since": True}, prom])])
        self.assertEqual([], nr.promotion_agents([inc]))


class UnfinishedNightTest(Tmp):
    def night(self, date: str, *folders: str, finished: bool = False, age_h: float = 1.0) -> Path:
        d = self.tmp / "nightly" / date
        for f in folders:
            (d / f).mkdir(parents=True)
            (d / f / "runs.jsonl").write_text(json.dumps(end("A", 5)) + "\n")
            os.utime(d / f / "runs.jsonl", (time.time() - age_h * 3600,) * 2)
        if finished:
            (d / folders[0].removesuffix("-emu")).mkdir(parents=True, exist_ok=True)
            (d / folders[0].removesuffix("-emu") / "summary.json").write_text(json.dumps({"nightly": {"runs": 1}}))
        return d

    def test_the_newest_night_whose_block_has_no_finished_mark_is_resumed(self):
        self.night("2026-10-26", "g5", "g5-emu", finished=True)
        self.night("2026-10-27", "g7-emu")  # the arm has rows, the real block never ran (a job the runner superseded)
        (self.tmp / "nightly" / "last-ok").write_text("x")
        (self.tmp / "nightly" / "scratch").mkdir()
        self.assertEqual("2026-10-27", nr.unfinished_night(self.tmp))

    def test_blocks_named_by_their_start_are_found_and_the_newest_unfinished_one_wins(self):
        self.night("2026-11-06", "g12", "g12-emu", finished=True)
        self.night("2026-11-06T0300", "g13", "g13-emu", finished=True)
        self.night("2026-11-06T0415", "g14-emu")  # a block of the same day that a stopped job left unfinished
        self.assertEqual("2026-11-06T0415", nr.unfinished_night(self.tmp))
        self.night("2026-11-07", "g14", "g14-emu", finished=True)
        self.assertEqual("2026-11-06T0415", nr.unfinished_night(self.tmp))  # the newer finished block does not hide it (sorted newest first, then finished)

    def test_a_finished_or_old_night_starts_a_fresh_one(self):
        self.night("2026-10-26", "g5", "g5-emu", finished=True)
        self.assertIsNone(nr.unfinished_night(self.tmp))
        self.night("2026-10-27", "g7-emu", age_h=40.0)  # abandoned longer than RESUME_NIGHT_H ago
        self.assertIsNone(nr.unfinished_night(self.tmp))
        self.assertEqual("2026-10-27", nr.unfinished_night(self.tmp, max_age_h=48.0))
        self.assertIsNone(nr.unfinished_night(self.tmp / "none"))  # no nightly folder yet


class NumbersTest(Tmp):
    def test_a_crash_is_a_loss_at_floor_zero_with_no_win_and_no_boss_kill(self):
        rows = [end("A", 20), end("B", 10), {"ev": "crash", "seed": "C", "error": "GameDied", "s": 2.0}]
        (self.tmp / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        (self.tmp / "summary.json").write_text(json.dumps({"divergence_rate": 0.003, "reanchors": 4, "wall_s": 99.0, "diverged_decisions": 6}))
        n = nr.nightly_numbers(self.tmp, "2026-10-25", {"label": "g3", "generation": 3, "role": "incumbent"}, "N-2026-10-25", 3, "core2")
        self.assertEqual((3, 1, 0.0), (n["runs"], n["crashes"], n["win_rate"]))
        self.assertEqual(10.0, n["mean_floor"])  # (20 + 10 + 0) / 3: the crash counts at floor 0
        self.assertEqual(1, n["boss_kills"])  # floor 20 is past the Act 1 boss; the crash is not a kill
        self.assertEqual((0.003, 4, 99.0), (n["divergence_rate"], n["reanchors"], n["wall_s"]))
        self.assertEqual((None, None), (n["boot_retries"], n["boot_gave_up"]))  # a summary from before the retry of a failed start

    def test_the_second_comparisons_records_and_decisions_are_the_summarys_and_none_for_a_summary_from_before(self):
        (self.tmp / "runs.jsonl").write_text(json.dumps(end("A", 20)))
        (self.tmp / "summary.json").write_text(json.dumps({"hidden": {"records": 7, "diverged_decisions": 31, "checked": 900}}))
        agent = {"label": "g3", "generation": 3, "role": "incumbent"}
        n = nr.nightly_numbers(self.tmp, "2026-10-25", agent, "N-2026-10-25", 1, "core2")
        self.assertEqual((7, 31), (n["hidden_records"], n["hidden_diverged_decisions"]))
        (self.tmp / "summary.json").write_text(json.dumps({}))
        n = nr.nightly_numbers(self.tmp, "2026-10-25", agent, "N-2026-10-25", 1, "core2")
        self.assertEqual((None, None), (n["hidden_records"], n["hidden_diverged_decisions"]))

    def test_the_starts_retried_and_given_up_are_the_summarys(self):
        (self.tmp / "runs.jsonl").write_text(json.dumps(end("A", 20)))
        (self.tmp / "summary.json").write_text(json.dumps({"boot_retries": 5, "boot_gave_up": 1}))
        n = nr.nightly_numbers(self.tmp, "2026-10-25", {"label": "g3", "generation": 3, "role": "incumbent"}, "N-2026-10-25", 1, "core2")
        self.assertEqual((5, 1), (n["boot_retries"], n["boot_gave_up"]))


class DriverTest(Tmp):
    """The whole driver with `gen0.py score` replaced by a function that leaves the files a real block leaves."""

    def run_driver(self, lock_line: str | None = None, *extra: str, date: str | None = "2026-10-25", prepare=None):
        lab, agent = self.tmp / "lab", self.tmp / "agent"
        if prepare:  # what a stopped job left in the lab
            prepare(lab / "nightly" / (date or "2026-10-25") / "g3")
        (agent / "gen3" / "c").mkdir(parents=True)
        (agent / "current.json").write_text(json.dumps({"generation": 3, "dir": str(agent / "gen3" / "c")}))
        lock = self.tmp / "QUIET.lock"
        if lock_line:
            lock.write_text(lock_line + "\n")
        seen: list[list[str]] = []

        def fake_run(cmd, **kw):
            seen.append(cmd)
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            self.at_start = {p.name: p.read_text(encoding="utf-8") for p in out.iterdir() if p.is_file()}  # the folder as the block found it
            gpu = self.tmp / "GPU_OK"
            self.gpu_at_play = gpu.read_text(encoding="utf-8") if gpu.is_file() else None  # the file as the block's play would find it
            (out / "scratch").mkdir(exist_ok=True)
            (out / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in [end("A", 20), {"ev": "crash", "seed": "B", "s": 1.0}]))
            (out / "divergences.jsonl").write_text("")
            (out / "summary.json").write_text(json.dumps({"divergence_rate": 0.002, "reanchors": 1, "wall_s": 5.0, "emu_arm": {"dir": "g3-emu"}}))
            return types.SimpleNamespace(returncode=0)

        job = self.tmp / "job"
        argv = ["--lab", str(lab), "--agent-root", str(agent), "--game", "G", "--lib", "L", "--out", str(job), "--runs", "20",
                *(["--date", date] if date else []), "--quiet-lock", str(lock), "--gpu-ok-file", str(self.tmp / "GPU_OK"), *extra]
        with unittest.mock.patch.object(nr.subprocess, "run", side_effect=fake_run), \
                unittest.mock.patch.object(nr.shutil, "rmtree", wraps=nr.shutil.rmtree):
            rc = nr.main(argv)
        return rc, seen, lab, job, lock

    def test_one_block_per_agent_with_core2_recycle_and_twelve_processes(self):
        rc, seen, lab, job, lock = self.run_driver()
        self.assertEqual(0, rc)
        cmd = seen[0]
        for flag, val in (("--engine", "core2"), ("--recycle", "60"), ("--procs", "12"), ("--runs", "20"), ("--config", "a3-v1")):
            self.assertEqual(val, cmd[cmd.index(flag) + 1], flag)
        self.assertEqual(str(lab / "seeds" / "N-2026-10-25.json"), cmd[cmd.index("--seeds-file") + 1])
        folder = lab / "nightly" / "2026-10-25" / "g3"
        self.assertEqual(str(folder), cmd[cmd.index("--out") + 1])
        summ = json.loads((folder / "summary.json").read_text())
        self.assertEqual((2, 1, 10.0, 1), (summ["nightly"]["runs"], summ["nightly"]["crashes"], summ["nightly"]["mean_floor"], summ["nightly"]["boss_kills"]))
        self.assertTrue((folder / "ranked.json").is_file())  # ranked in place for the by-family table
        self.assertFalse((folder / "scratch").exists())
        res = json.loads((job / "result.json").read_text())
        self.assertEqual(["g3"], list(res["agents"]))
        self.assertEqual("2026-10-25", res["date"])
        self.assertFalse(lock.exists())  # taken and removed

    def test_runs_promoted_is_not_used_without_a_promotion(self):
        rc, seen, lab, job, lock = self.run_driver(None, "--runs-promoted", "40")
        self.assertEqual(0, rc)
        self.assertEqual(["20"], [c[c.index("--runs") + 1] for c in seen])
        self.assertEqual(20, json.loads((job / "result.json").read_text())["runs_per_agent"])

    def test_runs_promoted_for_every_agent_on_a_promotion_night(self):
        def promote(_folder):
            agent = self.tmp / "agent"
            (agent / "gen4" / "c").mkdir(parents=True)
            (agent / "promotions.tsv").write_text("written\tcand_gen\tbase_gen\tseed_block\tseeds\tdecision\tcand_sha\tcand_dir\n"
                                                 f"2026-10-25T01:00:00Z\t4\t3\tE\t2000\tpromoted\tsha\t{agent / 'gen4' / 'c'}\n")
        rc, seen, lab, job, lock = self.run_driver(None, "--runs-promoted", "40", "--since", "2026-10-24T00:00:00Z", prepare=promote)
        self.assertEqual(0, rc)
        self.assertEqual(["40", "40"], [c[c.index("--runs") + 1] for c in seen])  # the incumbent and the promoted generation alike
        self.assertEqual(40, json.loads((job / "result.json").read_text())["runs_per_agent"])

    def test_a_half_finished_real_block_is_resumed_in_its_folder_not_replayed(self):
        row = json.dumps(end("A", 20))

        def half(folder):
            (folder / "scratch" / "p00r0").mkdir(parents=True)
            (folder / "runs.jsonl").write_text(row + "\n", encoding="utf-8")
            (folder / "resume.json").write_text('{"agent_seed": 1}', encoding="utf-8")
            (folder / "divergences.jsonl").write_text("", encoding="utf-8")

        rc, seen, lab, job, lock = self.run_driver(prepare=half)
        self.assertEqual(0, rc)
        self.assertIn("--resume", seen[0])
        self.assertEqual((row + "\n", '{"agent_seed": 1}'), (self.at_start["runs.jsonl"], self.at_start["resume.json"]))  # not cleared
        self.assertFalse((lab / "nightly" / "2026-10-25" / "g3" / "scratch").exists())  # the game copies still go after the run

    def test_the_command_carries_the_gpu_tag_and_a_stale_line_is_removed_at_start(self):
        gpu = self.tmp / "GPU_OK"
        gpu.write_text("lab-nightly\nw3-blocks 2026-10-28\nlab-nightly\n", encoding="utf-8")  # a killed job's line, and another lane's
        rc, seen, lab, job, lock = self.run_driver()
        self.assertEqual("lab-nightly", seen[0][seen[0].index("--gpu-ok-tag") + 1])
        self.assertEqual(str(gpu), seen[0][seen[0].index("--gpu-ok-file") + 1])
        self.assertEqual("w3-blocks 2026-10-28\n", self.gpu_at_play)  # gone before the first play (and so before any emulator arm)
        self.assertEqual("w3-blocks 2026-10-28\n", gpu.read_text(encoding="utf-8"))

    def test_the_real_arm_gets_the_machine_sampler_at_thirty_seconds_and_zero_switches_it_off(self):
        rc, seen, lab, job, lock = self.run_driver()
        self.assertEqual("30", seen[0][seen[0].index("--sample-machine") + 1])

    def test_sample_machine_zero_leaves_the_flag_out(self):
        rc, seen, lab, job, lock = self.run_driver(None, "--sample-machine", "0")
        self.assertNotIn("--sample-machine", seen[0])

    def test_a_stale_line_is_removed_even_when_the_night_is_skipped(self):
        gpu = self.tmp / "GPU_OK"
        gpu.write_text("lab-nightly\n", encoding="utf-8")
        rc, seen, lab, job, lock = self.run_driver(f"w4-readout {nr.utc_now()}", "--wait-h", "0")
        self.assertEqual((0, []), (rc, seen))
        self.assertFalse(gpu.exists())  # only that line was in it

    def test_a_finished_block_is_not_played_again(self):
        def finished(folder):
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text(json.dumps({"nightly": {"date": "2026-10-25", "agent": "g3", "generation": 3, "role": "incumbent",
                                                                          "runs": 2, "crashes": 1}}), encoding="utf-8")

        rc, seen, lab, job, lock = self.run_driver(prepare=finished)
        self.assertEqual((0, []), (rc, seen))

    def test_without_a_date_the_newest_unfinished_block_is_resumed_else_a_fresh_one_named_by_its_start(self):
        lab = self.tmp / "lab"
        emu = lab / "nightly" / "2026-10-24" / "g3-emu"
        emu.mkdir(parents=True)
        (emu / "runs.jsonl").write_text(json.dumps(end("A", 9)) + "\n")  # a stopped night: the arm has rows, the real block never ran
        rc, seen, lab, job, lock = self.run_driver(date=None)
        self.assertEqual(0, rc)
        self.assertEqual(str(lab / "seeds" / "N-2026-10-24.json"), seen[0][seen[0].index("--seeds-file") + 1])
        self.assertEqual("2026-10-24", json.loads((job / "result.json").read_text())["date"])
        self.assertTrue((emu / "runs.jsonl").is_file())  # the arm's rows are left for gen0.run_block's resume
        shutil.rmtree(job)
        shutil.rmtree(self.tmp / "agent")
        rc, seen, lab, job, lock = self.run_driver(date=None)  # that night is finished now: a fresh one under its UTC start time
        fresh = json.loads((job / "result.json").read_text())["date"]
        self.assertRegex(fresh, r"^\d{4}-\d{2}-\d{2}T\d{4}$")
        self.assertEqual(time.strftime("%Y-%m-%d", time.gmtime()), fresh[:10])
        self.assertTrue((lab / "nightly" / fresh / "g3").is_dir())

    def test_two_blocks_on_one_day_have_folders_and_seed_blocks_of_their_own(self):
        with unittest.mock.patch.object(nr, "block_name", return_value="2026-11-06T0300"):
            rc, seen, lab, job, lock = self.run_driver(date=None)
        first = seen[0]
        shutil.rmtree(job)
        shutil.rmtree(self.tmp / "agent")
        with unittest.mock.patch.object(nr, "block_name", return_value="2026-11-06T0415"):
            rc, seen, lab, job, lock = self.run_driver(date=None)  # the first is finished: this one is a block of its own, not "already played"
        self.assertEqual(1, len(seen))
        self.assertEqual(str(lab / "seeds" / "N-2026-11-06T0300.json"), first[first.index("--seeds-file") + 1])
        self.assertEqual(str(lab / "seeds" / "N-2026-11-06T0415.json"), seen[0][seen[0].index("--seeds-file") + 1])
        for name in ("2026-11-06T0300", "2026-11-06T0415"):
            summ = json.loads((lab / "nightly" / name / "g3" / "summary.json").read_text())
            self.assertEqual(name, summ["nightly"]["date"])
        self.assertEqual("2026-11-06T0415", json.loads((job / "result.json").read_text())["date"])

    # ---- 2026-11-06: the promotion trigger

    def promoted_g4(self, marker: bool = True, written: str = "2026-10-25T02:00:00Z", last_ok: str | None = "2026-10-24T12:00:00Z"):
        """A prepare for run_driver: generation 4 promoted at `written`, the pending marker up, last-ok at `last_ok` (None: no stamp)."""
        def prepare(_folder):
            lab, agent = self.tmp / "lab", self.tmp / "agent"
            (agent / "gen4" / "c").mkdir(parents=True)
            (agent / "promotions.tsv").write_text("written\tcand_gen\tbase_gen\tseed_block\tseeds\tdecision\tcand_sha\tcand_dir\n"
                                                 f"{written}\t4\t3\tE\t2000\tpromoted\tsha\t{agent / 'gen4' / 'c'}\n")
            (lab / "nightly").mkdir(parents=True, exist_ok=True)
            if marker:
                (lab / "nightly" / "pending").write_text(f"{written} gen4/c\n", encoding="utf-8")
            if last_ok:
                (lab / "nightly" / "last-ok").write_text("x\n")
                t = nr.parse_utc(last_ok)
                os.utime(lab / "nightly" / "last-ok", (t, t))
        return prepare

    def test_a_promotion_block_plays_the_promoted_generation_alone_at_runs_promoted_with_the_copy_sample(self):
        rc, seen, lab, job, lock = self.run_driver(None, "--runs-promoted", "40", "--emu-runs", "7", prepare=self.promoted_g4())
        self.assertEqual(0, rc)
        self.assertEqual(1, len(seen))  # g3, the incumbent, is not played at --runs: g4 is the promoted player here
        cmd = seen[0]
        self.assertEqual(str(self.tmp / "agent" / "gen4" / "c"), cmd[cmd.index("--gen-dir") + 1])
        self.assertEqual(("40", "7"), (cmd[cmd.index("--runs") + 1], cmd[cmd.index("--emu-runs") + 1]))
        res = json.loads((job / "result.json").read_text())
        self.assertEqual(["g4"], list(res["agents"]))
        self.assertEqual(("promotion", 40), (res["trigger"], res["runs_per_agent"]))
        self.assertEqual("promotion", res["agents"]["g4"]["trigger"])
        self.assertAlmostEqual(time.time(), res["started_epoch"], delta=60)

    def test_the_incumbent_promoted_since_is_the_promotion_blocks_player(self):
        def prepare(folder):
            self.promoted_g4(written="2026-10-25T02:00:00Z")(folder)
            agent = self.tmp / "agent"
            (agent / "promotions.tsv").write_text("written\tcand_gen\tbase_gen\tseed_block\tseeds\tdecision\tcand_sha\tcand_dir\n"
                                                 f"2026-10-25T02:00:00Z\t3\t2\tE\t2000\tpromoted\tsha\t{agent / 'gen3' / 'c'}\n")
        rc, seen, lab, job, lock = self.run_driver(None, "--runs-promoted", "40", prepare=prepare)
        self.assertEqual([("40", str(self.tmp / "agent" / "gen3" / "c"))], [(c[c.index("--runs") + 1], c[c.index("--gen-dir") + 1]) for c in seen])
        self.assertEqual("N-2026-10-25", Path(seen[0][seen[0].index("--seeds-file") + 1]).stem)  # the incumbent's block name as before

    def test_a_clock_block_without_the_marker_keeps_the_old_night(self):
        rc, seen, lab, job, lock = self.run_driver(None, "--runs-promoted", "40", prepare=self.promoted_g4(marker=False))
        self.assertEqual(0, rc)
        self.assertEqual([("40", "g3"), ("40", "g4")],
                         [(c[c.index("--runs") + 1], Path(c[c.index("--out") + 1]).name) for c in seen])  # the incumbent too, at the promotion night's runs
        self.assertEqual("clock", json.loads((job / "result.json").read_text())["trigger"])

    def test_trigger_clock_forces_the_old_night_though_the_marker_is_up(self):
        rc, seen, lab, job, lock = self.run_driver(None, "--runs-promoted", "40", "--trigger", "clock", prepare=self.promoted_g4())
        self.assertEqual(2, len(seen))
        self.assertEqual("clock", json.loads((job / "result.json").read_text())["trigger"])

    def test_a_marker_whose_promotion_the_last_block_already_played_is_an_idle_block(self):
        # the promotion (01:00) is before last-ok (stamped with the last block's start): nothing to play, nothing stamped, no incumbent block
        rc, seen, lab, job, lock = self.run_driver(None, prepare=self.promoted_g4(written="2026-10-25T01:00:00Z", last_ok="2026-10-25T05:00:00Z"))
        self.assertEqual((0, []), (rc, seen))
        res = json.loads((job / "result.json").read_text())
        self.assertEqual(({}, "promotion"), (res["agents"], res["trigger"]))
        self.assertIn("no generation promoted", res["idle"])
        self.assertFalse(lock.exists())  # the QUIET.lock line was taken and released

    def test_the_block_names_its_start_in_the_result_for_the_lab_to_stamp(self):
        before = time.time()
        rc, seen, lab, job, lock = self.run_driver()
        res = json.loads((job / "result.json").read_text())
        self.assertGreaterEqual(res["started_epoch"], before - 1)
        self.assertLessEqual(res["started_epoch"], time.time())
        self.assertEqual("clock", res["trigger"])

    def test_skips_with_a_reason_when_the_lock_stays_taken(self):
        rc, seen, lab, job, lock = self.run_driver(f"w4-readout {nr.utc_now()}", "--wait-h", "0")
        self.assertEqual((0, []), (rc, seen))
        self.assertIn("w4-readout", json.loads((job / "result.json").read_text())["skipped"])
        self.assertTrue(lock.read_text().startswith("w4-readout"))


class StallNumbersTest(Tmp):
    def test_the_nights_numbers_list_the_stalled_runs(self):
        rows = [end("A", 20), {**end("B", 4), "streak": {"n": 700, "screen": "event", "action": 0}},
                {**end("C", 9), "end": "timeout"}, {"ev": "net", "w": 0, "conns": []}]
        (self.tmp / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        n = nr.nightly_numbers(self.tmp, "2026-10-25", {"label": "g3", "generation": 3, "role": "incumbent"}, "N-2026-10-25", 3, "core2")
        self.assertEqual(["B", "C"], [s["seed"] for s in n["stalls"]])
        self.assertEqual({"n": 700, "screen": "event", "action": 0}, n["stalls"][0]["streak"])

    def test_the_sibling_emulator_arm_is_read_too(self):
        (self.tmp / "runs.jsonl").write_text(json.dumps({**end("B", 4), "streak": {"n": 700, "screen": "event", "action": 0}}))
        emu = self.tmp.with_name(self.tmp.name + "-emu")  # <agent>-emu beside <agent>
        emu.mkdir()
        self.addCleanup(shutil.rmtree, emu, True)
        (emu / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"seed": "E1", "win": False, "floor": 4, "end": "timeout"}, {"seed": "E2", "win": False, "floor": 12, "end": "death"}]))
        n = nr.nightly_numbers(self.tmp, "2026-10-25", {"label": "g3", "generation": 3, "role": "incumbent"}, "N-2026-10-25", 1, "core2")
        self.assertEqual([("real", "B", 4), ("emu", "E1", 4)], [(s["arm"], s["seed"], s["floor"]) for s in n["stalls"]])

    def test_a_clean_night_has_an_empty_list(self):
        (self.tmp / "runs.jsonl").write_text(json.dumps(end("A", 20)))
        n = nr.nightly_numbers(self.tmp, "2026-10-25", {"label": "g3", "generation": 3, "role": "incumbent"}, "N-2026-10-25", 1, "core2")
        self.assertEqual([], n["stalls"])



class PartialCopyArmNumbersTest(Tmp):
    """The nightly numbers of a night whose copy arm played a prefix of the block (`--emu-runs`)."""

    AGENT = {"label": "g3", "generation": 3, "role": "incumbent"}

    def arms(self, real_seeds: list[str], emu_seeds: list[str]) -> Path:
        folder = self.tmp / "g3"
        folder.mkdir()
        (folder / "runs.jsonl").write_text("\n".join(json.dumps(end(s, 10 + i)) for i, s in enumerate(real_seeds)))
        emu = self.tmp / "g3-emu"
        emu.mkdir()
        rows = [{"seed": s, "win": False, "floor": 9, "end": "death", "decisions": 5, "wall": 1.0} for s in emu_seeds]
        (emu / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        return folder

    def test_the_summary_says_how_many_seeds_the_copy_arm_was_to_play_and_how_many_are_paired(self):
        folder = self.arms(["A", "B", "C", "D", "E"], ["A", "B"])
        n = nr.nightly_numbers(folder, "2026-11-04", self.AGENT, "N-2026-11-04", 5, "core2", 2)
        self.assertEqual((5, 2, 2, 2), (n["runs_planned"], n["emu_runs_planned"], n["emu_rows"], n["paired_seeds"]))
        self.assertEqual(5, n["runs"])  # the real arm's own numbers are over all its seeds

    def test_the_pairs_are_the_seeds_with_a_row_in_both_arms(self):
        folder = self.arms(["A", "B", "C"], ["B", "C", "Z"])  # a copy-arm row the real arm lacks pairs with nothing
        (folder / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in [end("A", 5), end("B", 7), {"ev": "crash", "seed": "C", "s": 1.0}, {"ev": "net", "w": 0}]))
        n = nr.nightly_numbers(folder, "2026-11-04", self.AGENT, "N-2026-11-04", 3, "core2", 3)
        self.assertEqual((3, 2), (n["emu_rows"], n["paired_seeds"]))  # B (ended) and C (a crash row is a row)

    def test_without_the_option_the_copy_arm_was_to_play_every_seed(self):
        folder = self.arms(["A", "B", "C"], ["A", "B", "C"])
        n = nr.nightly_numbers(folder, "2026-11-04", self.AGENT, "N-2026-11-04", 3, "core2")
        self.assertEqual((3, 3, 3), (n["emu_runs_planned"], n["emu_rows"], n["paired_seeds"]))

    def test_a_night_with_no_copy_arm_rows_has_no_pairs(self):
        folder = self.tmp / "g3"
        folder.mkdir()
        (folder / "runs.jsonl").write_text(json.dumps(end("A", 5)))
        n = nr.nightly_numbers(folder, "2026-11-04", self.AGENT, "N-2026-11-04", 1, "core2", 1)
        self.assertEqual((0, 0), (n["emu_rows"], n["paired_seeds"]))

    def test_the_copy_arms_seed_count_is_cut_to_the_block(self):
        self.assertEqual(200, nr.emu_runs_tonight(200, 500))
        self.assertEqual(500, nr.emu_runs_tonight(200_000, 500))  # a promotion night's 2,000 or a small block: the whole block
        self.assertIsNone(nr.emu_runs_tonight(None, 500))

    def test_the_ranking_joins_the_copy_arms_outcome_on_the_seeds_it_has(self):
        from tools.lab import rank

        folder = self.arms(["A", "B", "C"], ["A", "B"])
        (folder / "divergences.jsonl").write_text("")
        doc = rank.rank(folder, self.tmp / "g3-emu")
        self.assertEqual(2, doc["summary"]["emu_runs_joined"])  # C has no partner and uses the proxy rule


class PartialCopyArmDriverTest(Tmp):
    run_driver = DriverTest.run_driver  # the whole driver with `gen0.py score` replaced; this class adds the copy-arm tests only

    def cmd_value(self, cmd, flag):
        return cmd[cmd.index(flag) + 1] if flag in cmd else None

    def test_the_command_has_no_flag_without_the_option(self):
        rc, seen, lab, job, lock = self.run_driver()
        self.assertNotIn("--emu-runs", seen[0])
        res = json.loads((job / "result.json").read_text())
        self.assertEqual(20, res["emu_runs_per_agent"])
        nightly = json.loads((lab / "nightly" / "2026-10-25" / "g3" / "summary.json").read_text())["nightly"]
        self.assertEqual(20, nightly["emu_runs_planned"])

    def test_the_copy_arm_gets_the_prefix_and_the_real_arm_all_the_runs(self):
        rc, seen, lab, job, lock = self.run_driver(None, "--emu-runs", "7")
        self.assertEqual(0, rc)
        self.assertEqual(("7", "20"), (self.cmd_value(seen[0], "--emu-runs"), self.cmd_value(seen[0], "--runs")))
        res = json.loads((job / "result.json").read_text())
        self.assertEqual((20, 7), (res["runs_per_agent"], res["emu_runs_per_agent"]))
        nightly = json.loads((lab / "nightly" / "2026-10-25" / "g3" / "summary.json").read_text())["nightly"]
        self.assertEqual((20, 7), (nightly["runs_planned"], nightly["emu_runs_planned"]))

    def test_a_count_past_the_block_is_the_block(self):
        rc, seen, lab, job, lock = self.run_driver(None, "--emu-runs", "50")
        self.assertEqual("20", self.cmd_value(seen[0], "--emu-runs"))

    def test_a_promotion_night_cuts_to_its_own_block(self):
        def promote(_folder):
            agent = self.tmp / "agent"
            (agent / "gen4" / "c").mkdir(parents=True)
            (agent / "promotions.tsv").write_text("written\tcand_gen\tbase_gen\tseed_block\tseeds\tdecision\tcand_sha\tcand_dir\n"
                                                 f"2026-10-25T01:00:00Z\t4\t3\tE\t2000\tpromoted\tsha\t{agent / 'gen4' / 'c'}\n")
        rc, seen, lab, job, lock = self.run_driver(None, "--runs-promoted", "40", "--emu-runs", "30", "--since", "2026-10-24T00:00:00Z", prepare=promote)
        self.assertEqual(["30", "30"], [self.cmd_value(c, "--emu-runs") for c in seen])  # 30 of 40 on both agents' blocks
        self.assertEqual(30, json.loads((job / "result.json").read_text())["emu_runs_per_agent"])

    def test_a_count_under_one_is_refused(self):
        with self.assertRaises(SystemExit):
            self.run_driver(None, "--emu-runs", "0")

    def test_a_half_finished_block_resumes_both_arms_on_the_same_prefix(self):
        row = json.dumps(end("A", 20))

        def half(folder):  # the real arm has one row; the copy arm had played two of its seven when the job was stopped
            (folder / "scratch").mkdir(parents=True)
            (folder / "runs.jsonl").write_text(row + "\n", encoding="utf-8")
            (folder / "resume.json").write_text('{"agent_seed": 1}', encoding="utf-8")
            (folder / "divergences.jsonl").write_text("", encoding="utf-8")
            emu = folder.with_name(folder.name + "-emu")
            emu.mkdir()
            (emu / "runs.jsonl").write_text("".join(json.dumps({"seed": s, "win": False, "floor": 5, "end": "death", "decisions": 3, "wall": 1.0}) + "\n"
                                                    for s in ("A", "B")), encoding="utf-8")

        rc, seen, lab, job, lock = self.run_driver(None, "--emu-runs", "7", prepare=half)
        self.assertEqual(0, rc)
        self.assertIn("--resume", seen[0])  # gen0.py score resumes the real arm and the copy arm
        self.assertEqual("7", self.cmd_value(seen[0], "--emu-runs"))  # on the prefix the first start used
        self.assertEqual((row + "\n", '{"agent_seed": 1}'), (self.at_start["runs.jsonl"], self.at_start["resume.json"]))  # not cleared
        emu_rows = (lab / "nightly" / "2026-10-25" / "g3-emu" / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(2, len(emu_rows))  # the copy arm's rows are left for gen0.run_block's resume
        nightly = json.loads((lab / "nightly" / "2026-10-25" / "g3" / "summary.json").read_text())["nightly"]
        self.assertEqual((7, 2, 2), (nightly["emu_runs_planned"], nightly["emu_rows"], nightly["paired_seeds"]))  # the fake block ends on A and B, as the copy arm has


if __name__ == "__main__":
    unittest.main()
