"""Generation 0's pure parts (rl/gen0.py): the seed block, the Wilson and paired maths, the result writer."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT))

import gen0  # noqa: E402
from tools.lab import mix  # noqa: E402


class SeedTests(unittest.TestCase):
    def test_block_is_a_pure_function_of_its_name(self):
        a = gen0.e_block("E0-2026-10-23", 500)
        self.assertEqual(a, gen0.e_block("E0-2026-10-23", 500))
        self.assertEqual(len(set(a)), 500)
        self.assertNotEqual(a[:5], gen0.e_block("E0-other", 5))
        self.assertTrue(all(len(s) == 10 and set(s) <= set(gen0.ALPHABET) for s in a))

    def test_committed_block_matches_its_name(self):
        f = ROOT / "referee" / "seeds" / "E0-2026-10-23.json"
        self.assertEqual(gen0.load_seeds(f), gen0.e_block("E0-2026-10-23", 500))

    def test_self_play_stream_never_overlaps_the_e_block(self):
        self.assertFalse(set(gen0.stream_seeds("job1", 2000)) & set(gen0.e_block("E0-2026-10-23", 500)))

    def test_duplicate_seeds_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"
            p.write_text(json.dumps({"seeds": ["A", "A"]}))
            with self.assertRaises(ValueError):
                gen0.load_seeds(p)


class MathTests(unittest.TestCase):
    def test_wilson_known_values(self):
        lo, hi = gen0.wilson(0, 10)
        self.assertAlmostEqual(lo, 0.0, places=6)
        self.assertAlmostEqual(hi, 0.2775, places=3)
        lo, hi = gen0.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        self.assertEqual(gen0.wilson(0, 0), (0.0, 1.0))
        self.assertEqual(gen0.wilson(0, 500)[0], 0.0)  # exactly 0, not 4e-19
        self.assertEqual(gen0.wilson(500, 500)[1], 1.0)

    def test_paired_uses_only_shared_seeds_and_differences(self):
        a = {"x": 10.0, "y": 12.0, "z": 5.0}
        b = {"x": 8.0, "y": 8.0, "w": 99.0}
        p = gen0.paired(a, b)
        self.assertEqual(p["seeds"], 2)
        self.assertAlmostEqual(p["mean"], 3.0)
        sd = math.sqrt(2.0)  # differences 2 and 4
        self.assertAlmostEqual(p["hi"] - p["mean"], gen0.Z * sd / math.sqrt(2))

    def test_pairing_beats_unpaired_when_seeds_drive_the_outcome(self):
        a = {str(i): float(i) for i in range(50)}
        b = {str(i): float(i) - 1 for i in range(50)}
        p = gen0.paired(a, b)
        self.assertAlmostEqual(p["mean"], 1.0)
        self.assertAlmostEqual(p["lo"], 1.0)  # no spread in the differences: the interval is a point

    def test_compare_computes_G_and_floor_gap(self):
        def run(s, win, floor, crash=False):
            return {"seed": s, "win": win, "floor": floor, "decisions": 10, "wall": 1.0, "crash": crash}
        emu = [run("a", True, 50), run("b", False, 10), run("c", True, 50), run("d", False, 8)]
        real = [run("a", True, 50), run("b", False, 6), run("c", False, 20), run("e", False, 1)]
        c = gen0.compare(emu, real)
        self.assertEqual(c["paired_seeds"], 3)
        self.assertAlmostEqual(c["G"]["mean"], 1 / 3)
        self.assertAlmostEqual(c["floor_gap"]["mean"], (0 + 4 + 30) / 3)
        self.assertEqual(c["emu"]["wins"], 2)  # only the shared seeds a, b, c count in the table


class ResultTests(unittest.TestCase):
    def test_result_json_keys_are_yield_columns(self):
        rows = [{"seed": "a", "win": True, "floor": 50, "decisions": 100, "wall": 10.0, "crash": False},
                {"seed": "b", "win": False, "floor": 0, "decisions": 0, "wall": 1.0, "crash": True}]
        r = gen0.result_json(rows, 20.0)
        self.assertTrue(set(r) <= set(mix.AGENT_COLUMNS), set(r) - set(mix.AGENT_COLUMNS))
        self.assertEqual((r["runs"], r["wins"], r["crashes"], r["decisions"]), (2, 1, 1, 100))
        self.assertEqual(r["decisions_s"], 5.0)
        self.assertEqual(r["mean_floor"], 25.0)

    def test_real_runs_counts_a_crash_as_a_loss(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "runs.jsonl"
            p.write_text("\n".join(json.dumps(e) for e in [
                {"ev": "end", "seed": "a", "end": "win", "floor": 51, "s": 9.0, "stats": {"decisions": 30}, "assumed": 2},
                {"ev": "end", "seed": "b", "end": "death", "floor": 7, "s": 3.0, "stats": {"decisions": 12}},
                {"ev": "crash", "seed": "c", "error": "x", "s": 1.0}]))
            rows = gen0.real_runs(p)
        self.assertEqual([r["win"] for r in rows], [True, False, False])
        self.assertEqual([r["crash"] for r in rows], [False, False, True])
        self.assertEqual(rows[2]["floor"], 0)
        self.assertEqual(rows[0]["ai_state_assumed"], 2)


class BenchResultTests(unittest.TestCase):
    def test_env_bench_summary_maps_to_yield_keys(self):
        r = gen0.env_bench_result({"runs_ended": 40, "decisions": 9000, "decisions_per_s": 108.5, "decisions_per_s_per_proc": 9.04,
                                   "mem_mb_mean": 900, "crashes": [{"ev": "crash"}], "hangs": []})
        self.assertTrue(set(r) <= set(mix.AGENT_COLUMNS))
        self.assertEqual((r["decisions_s"], r["dec_s"], r["mem_mb_per_run"], r["crashes"]), (108.5, 9.04, 900, 1))

    def test_emu_bench_takes_the_last_worker_count(self):
        doc = {"runs": [{"workers": 1, "steps_per_s": {"total": 5}}, {"workers": 16, "steps_per_s": {"total": 80.0},
                                                                    "clones_per_s": {"total": 1.5}, "sims_per_s": {"total": 9000.0}}]}
        r = gen0.emu_bench_result(doc)
        self.assertEqual(r, {"steps_s": 80.0, "clones_s": 1.5, "sims_s": 9000.0})
        self.assertTrue(set(r) <= set(mix.AGENT_COLUMNS))


class ScoreOnlyTest(unittest.TestCase):
    """B7: `score --only` hands the seed subset to envproto play and plays no emulator arm."""

    def test_only_goes_to_the_game_side_and_no_emulator_arm_is_played(self):
        from types import SimpleNamespace
        from unittest import mock

        out = Path(tempfile.mkdtemp()) / "probe"
        a = SimpleNamespace(out=str(out), game="G", seeds_file="F.json", runs=500, procs=4, sims=60, config="a3-v1", ascension=10,
                            character="IRONCLAD", stop_after_h=1.0, lib=None, emu_dir=None, gen_dir=None, only="S3,S5")
        cmds = []
        with mock.patch("subprocess.run", lambda cmd, **kw: cmds.append(cmd) or SimpleNamespace(returncode=0)),                 mock.patch.object(gen0, "play_emu_arm", side_effect=AssertionError("an emulator arm was played")):
            gen0.score_real(a)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0][cmds[0].index("--only") + 1], "S3,S5")

    def test_the_parser_takes_only(self):
        a = gen0.build_parser().parse_args(["score", "--game", "G", "--seeds-file", "F", "--out", "O", "--only", "A,B"])
        self.assertEqual(a.only, "A,B")


class ScoreResumeTest(unittest.TestCase):
    """N3: `score --resume` leaves the real block's folder alone, hands envproto play `--resume` with the agent's identity, and
    computes summary.json and compare.json over every row of the block (the kept ones and the new)."""

    def block(self, resume):
        import shutil
        from types import SimpleNamespace
        from unittest import mock

        gen0._setup()
        import gen_loop

        agent = gen_loop.load_agent(None).describe()
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        emu, out = tmp / "emu", tmp / "real"
        emu.mkdir()
        out.mkdir()
        (emu / "summary.json").write_text(json.dumps({"agent": agent}), encoding="utf-8")
        (emu / "runs.jsonl").write_text("".join(json.dumps({"seed": s, "win": s == "S0", "floor": 9, "decisions": 50, "wall": 1.0}) + "\n"
                                                for s in ("S0", "S1")), encoding="utf-8")
        end = lambda s, floor: {"ev": "end", "seed": s, "end": "death", "floor": floor, "s": 2.0, "stats": {"decisions": 80}}  # noqa: E731
        (out / "runs.jsonl").write_text(json.dumps(end("S1", 6)) + "\n", encoding="utf-8")  # the stopped session's row
        (out / "keep.txt").write_text("x", encoding="utf-8")
        a = SimpleNamespace(out=str(out), game="G", seeds_file="F.json", runs=2, procs=4, sims=60, config="a3-v1", ascension=10,
                            character="IRONCLAD", stop_after_h=1.0, lib=None, emu_dir=str(emu), gen_dir=None, only=None, resume=resume)
        cmds, seen = [], []

        def fake_run(cmd, **kw):  # what envproto play leaves: the kept row and the new one, and the summary over both
            cmds.append(cmd)
            seen.append((out / "keep.txt").is_file() and (out / "runs.jsonl").is_file())
            (out / "runs.jsonl").write_text(json.dumps(end("S1", 6)) + "\n" + json.dumps(end("S0", 12)), encoding="utf-8")
            (out / "summary.json").write_text(json.dumps({"runs": 2, "wall_s": 30.0, "resumed_rows": 1, "diverged_decisions": 0}), encoding="utf-8")
            return SimpleNamespace(returncode=0)

        with mock.patch("subprocess.run", fake_run), mock.patch.object(gen0, "play_emu_arm", side_effect=AssertionError("arm played")):
            gen0.score_real(a)
        return agent, cmds[0], seen, out

    def test_resume_passes_the_flag_with_the_agent_and_keeps_the_folder(self):
        agent, cmd, seen, out = self.block(True)
        self.assertIn("--resume", cmd)
        self.assertEqual(agent, json.loads(cmd[cmd.index("--agent-identity") + 1]))
        self.assertEqual([True], seen)  # the folder and its rows were still there when envproto started
        summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual((1, 30.0), (summ["resumed_rows"], summ["wall_s"]))  # envproto's keys stay
        self.assertFalse(summ["emu_arm"]["played"])
        cmp = json.loads((out / "compare.json").read_text(encoding="utf-8"))
        self.assertEqual((2, 2, 2), (cmp["paired_seeds"], cmp["real"]["runs"], cmp["emu"]["runs"]))  # every row of the block
        self.assertEqual((2, 30.0), (json.loads((out / "result.json").read_text(encoding="utf-8"))["runs"], summ["wall_s"]))

    def test_without_resume_the_command_is_what_it_was(self):
        _agent, cmd, _seen, _out = self.block(False)
        self.assertNotIn("--resume", cmd)
        self.assertNotIn("--agent-identity", cmd)

    def test_the_parser_takes_resume(self):
        base = ["score", "--game", "G", "--seeds-file", "F", "--out", "O"]
        self.assertTrue(gen0.build_parser().parse_args([*base, "--resume"]).resume)
        self.assertFalse(gen0.build_parser().parse_args(base).resume)


class GpuOkTest(unittest.TestCase):
    """The GPU lane's window: `score --gpu-ok-tag TAG` keeps the line TAG first in the GPU_OK file exactly while the game's side
    plays (never during the emulator arm) and removes it however the play ends; without the option nothing is touched."""

    def run_score(self, tag, initial, fail=False, **extra):
        import shutil
        from types import SimpleNamespace
        from unittest import mock

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        gpu = tmp / "GPU_OK"
        if initial is not None:
            gpu.write_text(initial, encoding="utf-8")
        read = lambda: gpu.read_text(encoding="utf-8") if gpu.is_file() else None  # noqa: E731
        self.gpu, self.during_arm, self.during_play = gpu, [], []

        def fake_arm(*args, **kw):
            self.during_arm.append(read())

        def fake_run(cmd, **kw):
            self.during_play.append(read())
            if fail:
                raise RuntimeError("the play died")
            return SimpleNamespace(returncode=0)

        a = SimpleNamespace(out=str(tmp / "real"), game="G", seeds_file="F.json", runs=2, procs=4, sims=60, config="a3-v1", ascension=10,
                            character="IRONCLAD", stop_after_h=1.0, lib=None, emu_dir=None, gen_dir=None, only=None, gpu_ok_file=str(gpu),
                            **({"gpu_ok_tag": tag} if tag else {}), **extra)
        with mock.patch("subprocess.run", fake_run), mock.patch.object(gen0, "play_emu_arm", fake_arm):
            if fail:
                with self.assertRaises(RuntimeError):
                    gen0.score_real(a)
            else:
                gen0.score_real(a)
        return read()

    def test_the_line_is_first_exactly_while_the_play_runs(self):
        after = self.run_score("lab-nightly", "other-lane\nlab-nightly\n")
        self.assertEqual(["other-lane\nlab-nightly\n"], self.during_arm)  # not written before the emulator arm has returned
        self.assertEqual(["lab-nightly\nother-lane\n"], self.during_play)  # first; the older equal line is gone, the other kept
        self.assertEqual("other-lane\n", after)

    def test_the_line_is_removed_when_the_play_raises(self):
        after = self.run_score("lab-nightly", "other-lane\n", fail=True)
        self.assertEqual(["lab-nightly\nother-lane\n"], self.during_play)
        self.assertEqual("other-lane\n", after)

    def test_an_absent_file_is_created_for_the_play_and_deleted_when_nothing_else_is_in_it(self):
        after = self.run_score("lab-nightly", None)
        self.assertEqual(([None], ["lab-nightly\n"], None), (self.during_arm, self.during_play, after))
        self.assertFalse(self.gpu.exists())

    def test_without_the_option_nothing_is_written(self):
        after = self.run_score(None, "other-lane\n")
        self.assertEqual((["other-lane\n"], ["other-lane\n"], "other-lane\n"), (self.during_arm, self.during_play, after))
        self.assertIsNone(self.run_score(None, None))
        self.assertEqual(([None], [None]), (self.during_arm, self.during_play))

    def test_the_parser_takes_the_options(self):
        base = ["score", "--game", "G", "--seeds-file", "F", "--out", "O"]
        a = gen0.build_parser().parse_args([*base, "--gpu-ok-tag", "lab-nightly", "--gpu-ok-file", "X"])
        self.assertEqual(("lab-nightly", "X"), (a.gpu_ok_tag, a.gpu_ok_file))
        b = gen0.build_parser().parse_args(base)
        self.assertEqual((None, gen0.GPU_OK_FILE), (b.gpu_ok_tag, b.gpu_ok_file))


class SampleMachineTest(unittest.TestCase):
    """L3: `score --sample-machine SECONDS` is handed to envproto play (the real arm); off unless given."""

    def play_cmd(self, **extra):
        import shutil
        from types import SimpleNamespace
        from unittest import mock

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        seen = []

        def fake_run(cmd, **kw):
            seen.append(cmd)
            return SimpleNamespace(returncode=0)

        a = SimpleNamespace(out=str(tmp / "real"), game="G", seeds_file="F.json", runs=2, procs=4, sims=60, config="a3-v1", ascension=10,
                            character="IRONCLAD", stop_after_h=1.0, lib=None, emu_dir=None, gen_dir=None, only=None, **extra)
        with mock.patch("subprocess.run", fake_run), mock.patch.object(gen0, "play_emu_arm", lambda *a, **k: None):
            gen0.score_real(a)
        return seen[0]

    def test_the_play_command_carries_the_interval_only_when_given(self):
        self.assertNotIn("--sample-machine", self.play_cmd())
        cmd = self.play_cmd(sample_machine=30.0)
        self.assertEqual("30.0", cmd[cmd.index("--sample-machine") + 1])

    def test_the_parser_takes_the_option_off_by_default(self):
        base = ["score", "--game", "G", "--seeds-file", "F", "--out", "O"]
        self.assertEqual(0.0, gen0.build_parser().parse_args(base).sample_machine)
        self.assertEqual(30.0, gen0.build_parser().parse_args([*base, "--sample-machine", "30"]).sample_machine)



class PartialCopyArmTest(unittest.TestCase):
    """`score --emu-runs N` (2026-11-04): the copy arm plays the first N seeds of the block; everything that pairs the arms works on
    the seeds both hold, and a resumed arm keeps its rows on the prefix."""

    SEEDS = [f"CP{i:08d}" for i in range(10)]

    def test_the_copy_arms_seeds_are_a_fixed_prefix_of_the_block(self):
        self.assertEqual(self.SEEDS[:4], gen0.emu_arm_seeds(self.SEEDS, 10, 4))
        self.assertEqual(gen0.emu_arm_seeds(self.SEEDS, 10, 4), gen0.emu_arm_seeds(self.SEEDS, 10, 4))  # the same on every call
        self.assertEqual(gen0.emu_arm_seeds(self.SEEDS, 10, 6)[:4], gen0.emu_arm_seeds(self.SEEDS, 10, 4))  # a larger N extends it
        self.assertEqual(self.SEEDS, gen0.emu_arm_seeds(self.SEEDS, 10))  # no option: every seed the real arm plays
        self.assertEqual(self.SEEDS[:7], gen0.emu_arm_seeds(self.SEEDS, 7))  # the real arm's own --runs cuts the block first
        self.assertEqual(self.SEEDS[:7], gen0.emu_arm_seeds(self.SEEDS, 7, 50))  # N past the block is the whole block
        self.assertEqual(self.SEEDS[:3], gen0.emu_arm_seeds(self.SEEDS, 7, 3))

    def test_a_copy_arm_of_no_seeds_is_refused(self):
        for n in (0, -5):
            with self.assertRaises(ValueError):
                gen0.emu_arm_seeds(self.SEEDS, 10, n)

    def test_the_parser_takes_emu_runs_and_defaults_to_all(self):
        base = ["score", "--game", "G", "--seeds-file", "F", "--out", "O"]
        self.assertIsNone(gen0.build_parser().parse_args(base).emu_runs)
        self.assertEqual(200, gen0.build_parser().parse_args([*base, "--emu-runs", "200"]).emu_runs)

    def play_arm(self, emu_runs, runs=10):
        import shutil
        from types import SimpleNamespace
        from unittest import mock

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        f = tmp / "seeds.json"
        f.write_text(json.dumps({"seeds": self.SEEDS}), encoding="utf-8")
        a = SimpleNamespace(lib=None, engine="core2", seeds_file=str(f), runs=runs, config="a3-v1", sims=60, procs=2, ascension=10,
                            character="IRONCLAD", **({} if emu_runs == "absent" else {"emu_runs": emu_runs}))
        got = {}

        def fake_block(seeds, sims, procs, out, **kw):
            out.mkdir(parents=True, exist_ok=True)
            got.update(seeds=list(seeds), kw=kw)
            return {"runs": len(seeds)}

        with mock.patch.object(gen0, "run_block", fake_block), mock.patch.object(gen0, "select_engine", lambda e: None):
            gen0.play_emu_arm(a, None, tmp / "emu")
        return got

    def test_play_emu_arm_hands_run_block_the_prefix_and_resumes(self):
        got = self.play_arm(4)
        self.assertEqual(self.SEEDS[:4], got["seeds"])
        self.assertTrue(got["kw"]["resume"])  # an unfinished arm resumes in its folder, on the same prefix
        self.assertEqual(self.SEEDS, self.play_arm(None)["seeds"])
        self.assertEqual(self.SEEDS, self.play_arm("absent")["seeds"])  # a caller that never heard of the option plays the whole block
        self.assertEqual(self.SEEDS[:6], self.play_arm(None, runs=6)["seeds"])

    def test_compare_pairs_on_the_seeds_both_arms_hold_and_counts_the_unpaired(self):
        def run(s, win, floor):
            return {"seed": s, "win": win, "floor": floor, "decisions": 10, "wall": 1.0, "crash": False}
        emu = [run("a", True, 50), run("b", False, 10)]  # the copy arm played a prefix of two
        real = [run("a", True, 50), run("b", False, 6), run("c", False, 20), run("d", False, 1), run("e", True, 51)]
        c = gen0.compare(emu, real)
        self.assertEqual(2, c["paired_seeds"])
        self.assertEqual((2, 2), (c["emu"]["runs"], c["real"]["runs"]))  # both tables over the pairs
        self.assertAlmostEqual(2.0, c["floor_gap"]["mean"])  # (0 + 4) / 2
        self.assertEqual(2, c["floor_gap"]["seeds"])
        self.assertEqual({"emu": 2, "real": 5, "emu_unpaired": 0, "real_unpaired": 3}, c["rows"])

    def score(self, emu_runs, runs=6):
        import shutil
        from types import SimpleNamespace
        from unittest import mock

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        out = tmp / "real"
        end = lambda s, floor: {"ev": "end", "seed": s, "end": "death", "floor": floor, "s": 2.0, "stats": {"decisions": 80}}  # noqa: E731

        def fake_arm(a, agent, emu_out):  # what run_block leaves for the prefix
            emu_out.mkdir(parents=True, exist_ok=True)
            (emu_out / "runs.jsonl").write_text("".join(json.dumps({"seed": s, "win": False, "floor": 9, "decisions": 50, "wall": 1.0}) + "\n"
                                                        for s in gen0.emu_arm_seeds(self.SEEDS, a.runs, a.emu_runs)), encoding="utf-8")

        def fake_run(cmd, **kw):  # envproto play over the whole block
            out.mkdir(parents=True, exist_ok=True)
            (out / "runs.jsonl").write_text("".join(json.dumps(end(s, 6 + i)) + "\n" for i, s in enumerate(self.SEEDS[:runs])), encoding="utf-8")
            (out / "summary.json").write_text(json.dumps({"runs": runs, "wall_s": 30.0, "diverged_decisions": 0, "divergence_rate": 0.0}),
                                              encoding="utf-8")
            return SimpleNamespace(returncode=0)

        a = SimpleNamespace(out=str(out), game="G", seeds_file="F.json", runs=runs, procs=4, sims=60, config="a3-v1", ascension=10,
                            character="IRONCLAD", stop_after_h=1.0, lib=None, emu_dir=None, gen_dir=None, only=None, emu_runs=emu_runs)
        with mock.patch("subprocess.run", fake_run), mock.patch.object(gen0, "play_emu_arm", fake_arm):
            gen0.score_real(a)
        return (json.loads((out / "summary.json").read_text(encoding="utf-8")), json.loads((out / "compare.json").read_text(encoding="utf-8")))

    def test_the_score_records_the_prefix_and_compares_the_shared_seeds(self):
        summ, cmp = self.score(4)
        self.assertEqual((4, True), (summ["emu_arm"]["emu_runs"], summ["emu_arm"]["partial"]))
        self.assertEqual((4, 4, 4), (cmp["paired_seeds"], cmp["real"]["runs"], cmp["emu"]["runs"]))
        self.assertEqual({"emu": 4, "real": 6, "emu_unpaired": 0, "real_unpaired": 2}, cmp["rows"])
        self.assertEqual(4, cmp["floor_gap"]["seeds"])
        self.assertEqual((True, 4), (cmp["emu_arm"]["partial"], cmp["emu_arm"]["emu_runs"]))

    def test_the_whole_block_is_not_marked_partial(self):
        summ, cmp = self.score(50)  # N past the block: every seed
        self.assertEqual((6, False), (cmp["emu_arm"]["emu_runs"], cmp["emu_arm"]["partial"]))
        self.assertEqual((6, 0), (cmp["paired_seeds"], cmp["rows"]["real_unpaired"]))
        _summ, plain = self.score(None)
        self.assertNotIn("partial", plain["emu_arm"])  # no option: the record is what it was

    def test_a_resumed_copy_arm_plays_only_the_prefix_seeds_it_has_no_row_for(self):
        import shutil
        from unittest import mock

        played: list[tuple[str, int]] = []

        def fake_run_safe(seed, ix, sims, ascension, agent=None, record=False, character="IRONCLAD"):
            played.append((seed, ix))
            return {"seed": seed, "ix": ix, "win": False, "floor": 5, "end": "death", "decisions": 3, "wall": 0.1, "ai_state_assumed": 0}

        class InlinePool:  # gen0.run_block's SupervisedPool with every task run in this process, at once
            stats = {"started": 0, "recycled": 0, "died": 0, "timed_out": 0, "raised": 0}

            def __init__(self, ctx, size, **kw):
                self.size, self.ended = size, []

            def free(self):
                return self.size - len(self.ended)

            def inflight(self):
                return len(self.ended)

            def submit(self, key, fn, args=()):
                self.ended.append(gen0.spawn_pool.Done(key, "ok", fn(*args)))

            def collect(self):
                out, self.ended = self.ended, []
                return out

            def wait(self, timeout):
                pass

            def terminate(self):
                pass

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        prefix = gen0.emu_arm_seeds(self.SEEDS, 10, 5)
        key = {"engine": "core2"}
        with mock.patch.object(gen0, "_run_safe", fake_run_safe), mock.patch.object(gen0.spawn_pool, "SupervisedPool", InlinePool):
            gen0.run_block(prefix[:2], "a3-v1", 2, tmp, resume=True, resume_key=key)  # the stopped session: two of five
            self.assertEqual([(s, i) for i, s in enumerate(prefix[:2])], played)
            played.clear()
            summ = gen0.run_block(prefix, "a3-v1", 2, tmp, resume=True, resume_key=key)  # the resumed job: the same prefix
            self.assertEqual([(s, i) for i, s in enumerate(prefix) if i >= 2], played)  # the other three, at their places in the block
            self.assertEqual((5, 2), (summ["runs"], summ["resumed_rows"]))
            rows = [json.loads(x) for x in (tmp / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sorted(prefix), sorted(r["seed"] for r in rows))  # one row per prefix seed, none beyond it
            played.clear()
            gen0.run_block(prefix, "a3-v1", 2, tmp, resume=True, resume_key=key)  # a third start finds nothing left
            self.assertEqual([], played)


if __name__ == "__main__":
    unittest.main()
