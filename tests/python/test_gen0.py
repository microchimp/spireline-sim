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


if __name__ == "__main__":
    unittest.main()
