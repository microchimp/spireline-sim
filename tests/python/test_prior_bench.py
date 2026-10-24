"""The pure parts of rl/prior_bench.py (section 9.1 and 9.3 arithmetic) and rl/prior_tolerance.py (section 9.2): the ABAB order,
the CPU-time arithmetic with the server's CPU, the exact two-sided McNemar, the equivalence verdict, the quality statistics,
the first divergence of two runs, and the rebuild of the tokens from a part's concatenated arrays. No emulator, no GPU."""

import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT))

import gen_loop  # noqa: E402
import prior_bench as pb  # noqa: E402
import prior_tolerance as pt  # noqa: E402


@dataclass
class FakeTokens:
    types: np.ndarray
    ids: np.ndarray
    owners: np.ndarray
    feats: np.ndarray

    def __len__(self):
        return len(self.types)


def fake_tokens(rng, n):
    return FakeTokens(rng.integers(1, 17, n).astype(np.int64), rng.integers(0, 90, n).astype(np.int64),
                      rng.integers(0, 3, n).astype(np.int64), rng.random((n, 18), dtype=np.float32))


def fake_part_runs(rng, shapes):
    """[(seed, arrays)] like ShardSink's pending list: one entry per run, `shapes` the token counts of its samples."""
    runs = []
    for r, lens in enumerate(shapes):
        samples = gen_loop.CombatSamples()
        for j, n in enumerate(lens):
            visits = np.zeros(32, dtype=np.float32)
            visits[:3] = rng.integers(1, 30, 3)
            legal = np.zeros(32, dtype=np.int8)
            legal[:3] = 1
            samples.add(fake_tokens(rng, n), visits, np.zeros(32, dtype=np.float32), legal, 1, 10, 1, 60 + j)
        samples.end_fight(40, 80)
        runs.append((f"SEED{r:04d}", samples.arrays() if lens else None))
    return runs


OK = (1.0, 1.0, 0)  # the verdict's path and pair conditions met: served share, CPU share, dropped pairs


def qrow(seed, floor, path, dec=100, comb=40, sims=2400, end="death", win=False):
    return {"seed": seed, "floor": floor, "decisions": dec, "combat_decisions": comb, "sims": sims, "end": end, "win": win,
            "prior_path": path}


class RowsAndOrder(unittest.TestCase):
    def test_rows_parse(self):
        self.assertEqual(pb.parse_rows("noprior:16, cpu:16,server:24"), [("noprior", 16), ("cpu", 16), ("server", 24)])
        for bad in ("", "gpu:4", "cpu", "cpu:0", "cpu:x"):
            with self.assertRaises(ValueError):
                pb.parse_rows(bad)

    def test_abab_order_runs_every_row_in_turn_each_repeat(self):
        rows = [("noprior", 16), ("cpu", 16), ("server", 16)]
        got = pb.abab_order(rows, 2)
        self.assertEqual([(a, p) for _, a, p in got], rows + rows)
        self.assertEqual([rep for rep, _, _ in got], [0, 0, 0, 1, 1, 1])
        self.assertEqual(pb.abab_order(rows, 1), [(0, "noprior", 16), (0, "cpu", 16), (0, "server", 16)])


class CpuArithmetic(unittest.TestCase):
    tasks = [{"dec": 100, "sims": 4000, "cpu": 10.0}, {"dec": 140, "sims": 6000, "cpu": 14.0}]

    def test_server_cpu_joins_the_denominator(self):
        m = pb.row_metrics(self.tasks, 6.0, {"cpu_s": 6.0, "calls": 600, "batches": 150})
        self.assertAlmostEqual(m["dec_s"], 240 / 6.0)
        self.assertAlmostEqual(m["sims_s"], 10000 / 6.0)
        self.assertAlmostEqual(m["worker_cpu_s"], 24.0)
        self.assertAlmostEqual(m["dec_per_cpu_s"], 240 / 30.0)  # 24 s of workers + 6 s of server
        self.assertAlmostEqual(m["sims_per_cpu_s"], 10000 / 30.0)
        self.assertAlmostEqual(m["server_cpu_share"], 0.2)
        self.assertAlmostEqual(m["server_calls_s"], 100.0)
        self.assertAlmostEqual(m["mean_batch"], 4.0)

    def test_no_server_means_no_server_figures(self):
        m = pb.row_metrics(self.tasks, 6.0)
        self.assertAlmostEqual(m["dec_per_cpu_s"], 10.0)
        self.assertIsNone(m["server_cpu_share"])
        self.assertIsNone(m["server_calls_s"])
        self.assertIsNone(m["mean_batch"])

    def test_mean_over_repeats(self):
        a = pb.row_metrics(self.tasks, 6.0)
        b = pb.row_metrics(self.tasks, 8.0)
        m = pb.mean_metrics([a, b])
        self.assertAlmostEqual(m["dec_s"], (240 / 6.0 + 240 / 8.0) / 2)
        self.assertIsNone(m["server_cpu_share"])


class Statistics(unittest.TestCase):
    def test_mcnemar_two_sided_exact(self):
        self.assertEqual(pb.mcnemar_two_sided(0, 0), 1.0)
        self.assertEqual(pb.mcnemar_two_sided(5, 5), 1.0)
        self.assertAlmostEqual(pb.mcnemar_two_sided(0, 10), 2 / 1024)
        self.assertAlmostEqual(pb.mcnemar_two_sided(10, 0), 2 / 1024)
        self.assertAlmostEqual(pb.mcnemar_two_sided(3, 12), 2 * (1 + 15 + 105 + 455) / 2**15)
        # the same number from the one-sided helper the judge uses
        self.assertAlmostEqual(pb.mcnemar_two_sided(3, 12), min(1.0, 2 * gen_loop.mcnemar_exact_one_sided(12, 3)))

    def test_equivalence_verdict_is_the_preregistered_rule(self):
        ci90 = {"n": 1000, "mean": 0.05, "lo": -0.12, "hi": 0.22}
        v = pb.equivalence_verdict(ci90, 0.8, 1.004, 0.999, *OK)
        self.assertTrue(v["equivalent"])
        self.assertEqual((v["floor_ok"], v["boss_ok"], v["ratios_ok"], v["paths_ok"], v["pairs_ok"]), (True,) * 5)
        # the old rule's conditions no longer bind: a point estimate over 0.1 and a thin interval still pass inside +-0.3
        self.assertTrue(pb.equivalence_verdict({"n": 1000, "mean": 0.2, "lo": 0.15, "hi": 0.25}, 1.0, 1.0, 1.0, *OK)["equivalent"])
        self.assertTrue(pb.equivalence_verdict({"n": 10, "mean": 0.0, "lo": -0.30, "hi": 0.30}, 1.0, 0.98, 1.02, *OK)["equivalent"])
        # (a) an interval that crosses +0.3 fails, whatever the rest says
        v = pb.equivalence_verdict({**ci90, "hi": 0.31}, 1.0, 1.0, 1.0, *OK)
        self.assertFalse(v["equivalent"])
        self.assertFalse(v["floor_ok"])
        self.assertFalse(pb.equivalence_verdict({**ci90, "lo": -0.31}, 1.0, 1.0, 1.0, *OK)["floor_ok"])
        # (b) boss kills significant at 0.05
        v = pb.equivalence_verdict(ci90, 0.04, 1.0, 1.0, *OK)
        self.assertEqual((v["equivalent"], v["boss_ok"], v["floor_ok"]), (False, False, True))
        # (c) a 3% sims ratio, or a 3% decisions ratio, fails; so does a missing ratio
        v = pb.equivalence_verdict(ci90, 1.0, 1.03, 1.0, *OK)
        self.assertEqual((v["equivalent"], v["ratios_ok"]), (False, False))
        self.assertFalse(pb.equivalence_verdict(ci90, 1.0, 1.0, 0.97, *OK)["equivalent"])
        self.assertFalse(pb.equivalence_verdict(ci90, 1.0, None, 1.0, *OK)["equivalent"])
        self.assertFalse(pb.equivalence_verdict({"n": 0}, 1.0, 1.0, 1.0, *OK)["equivalent"])
        # (d) the arms must have run on the paths they are named for: 99% server, 100% cpu
        self.assertTrue(pb.equivalence_verdict(ci90, 1.0, 1.0, 1.0, 0.99, 1.0, 0)["equivalent"])
        for served, cpu in ((0.98, 1.0), (0.0, 1.0), (1.0, 0.99), (None, 1.0), (1.0, None)):
            v = pb.equivalence_verdict(ci90, 1.0, 1.0, 1.0, served, cpu, 0)
            self.assertEqual((v["equivalent"], v["paths_ok"], v["floor_ok"]), (False, False, True), (served, cpu))
        # (e) a pair lost to a crash is a finding
        v = pb.equivalence_verdict(ci90, 1.0, 1.0, 1.0, 1.0, 1.0, 1)
        self.assertEqual((v["equivalent"], v["pairs_ok"]), (False, False))
        self.assertFalse(pb.equivalence_verdict(ci90, 1.0, 1.0, 1.0, 1.0, 1.0, None)["equivalent"])

    def test_boss_kill_is_the_judges_rule(self):
        rows = [{"win": False, "floor": 17}, {"win": False, "floor": 18}, {"win": True, "floor": 17}, {"win": False, "floor": 0},
                {"win": True, "floor": 53}, {}]
        for row in rows:
            self.assertEqual(pb.boss_killed(row), gen_loop.boss_kill(row), row)
        self.assertEqual([pb.boss_killed(r) for r in rows[:3]], [False, True, True])

    def test_bootstrap_interval_and_se(self):
        zeros = pb.bootstrap_ci([0.0] * 50, draws=500)
        self.assertEqual((zeros["lo"], zeros["hi"], zeros["se"]), (0.0, 0.0, 0.0))
        self.assertTrue(pb.equivalence_verdict(zeros, 1.0, 1.0, 1.0, *OK)["equivalent"])
        d = [1.0, -1.0] * 500
        ci = pb.bootstrap_ci(d, draws=2000)
        self.assertAlmostEqual(ci["mean"], 0.0)
        self.assertAlmostEqual(ci["se"], 1.0 / np.sqrt(1000), places=3)
        self.assertLess(ci["lo"], 0.0)
        self.assertGreater(ci["hi"], 0.0)
        self.assertLess(ci["hi"] - ci["lo"], 5 * ci["se"])
        self.assertEqual(ci, pb.bootstrap_ci(d, draws=2000))  # the seed is fixed
        self.assertAlmostEqual(ci["hi"], 1.96 * ci["se"], delta=0.01)  # a 95% interval is about +-1.96 SE
        ci90 = pb.bootstrap_ci(d, draws=2000, level=pb.EQUIV_LEVEL)
        self.assertAlmostEqual(ci90["hi"], 1.645 * ci90["se"], delta=0.01)
        self.assertLess(ci90["hi"], ci["hi"])
        shifted = pb.bootstrap_ci([0.5] * 30 + [0.4] * 30, draws=500, level=pb.EQUIV_LEVEL)
        self.assertFalse(pb.equivalence_verdict(shifted, 1.0, 1.0, 1.0, *OK)["floor_ok"])

    def test_quality_stats(self):
        cpu = [qrow("A", 10, "cpu"), qrow("B", 12, "cpu"), qrow("C", 18, "cpu"), qrow("D", 9, "cpu"), {"seed": "E", "crash": True}]
        srv = [qrow("A", 10, "server"), qrow("B", 11, "server", dec=101), qrow("C", 17, "server"), qrow("D", 9, "server"),
               qrow("E", 5, "server")]
        div = {"A": None, "B": {"level": "search", "index": 3, "fight": 2, "floor": 6}, "C": {"level": "decision", "floor": 12, "index": 9,
                                                                                               "fight": None}, "D": None}
        st = pb.quality_stats(cpu, srv, div, draws=200)
        self.assertEqual(st["paired_seeds"], 4)
        self.assertEqual((st["crashed_pairs_dropped"], st["unpaired_seeds"]), (1, 0))
        self.assertEqual(st["identical_rows"], 2)  # A and D
        self.assertEqual(st["identical_decision_for_decision"], 2)
        self.assertEqual(st["first_diverging_floor_histogram"], {"6": 1, "12": 1})
        self.assertAlmostEqual(st["floor"]["diff_ci"]["mean"], -0.5)
        self.assertEqual((st["boss_kill"]["cpu"], st["boss_kill"]["server"]), (1, 0))  # C ended past floor 17 on the CPU path
        self.assertEqual((st["boss_kill"]["server_only"], st["boss_kill"]["cpu_only"]), (0, 1))
        self.assertAlmostEqual(st["boss_kill"]["p_two_sided"], 1.0)
        self.assertAlmostEqual(st["sims_per_decision"]["ratio"], 1.0)
        self.assertAlmostEqual(st["decisions_per_run"]["ratio"], 401 / 400)
        self.assertTrue(st["verdict"]["ratios_ok"])
        self.assertNotIn("ratios_within_2_percent", st)  # one copy of the rule: the verdict's
        self.assertFalse(st["verdict"]["equivalent"])  # four pairs: the 90% interval is far wider than +-0.3
        self.assertFalse(st["verdict"]["floor_ok"])
        self.assertFalse(st["verdict"]["pairs_ok"])  # E crashed on the CPU side

    def test_quality_verdict_end_to_end(self):
        def arm(path, floors=None, sims=2400, n=60):
            return [qrow(f"S{i:03d}", (floors or (lambda i: 8 + i % 7))(i), path, sims=sims) for i in range(n)]

        cpu = arm("cpu")
        same = pb.quality_stats(cpu, arm("server"), draws=300)
        self.assertTrue(same["verdict"]["equivalent"])
        self.assertEqual((same["floor"]["diff_ci90"]["lo"], same["floor"]["diff_ci90"]["hi"]), (0.0, 0.0))
        # half the seeds one floor higher on the server: mean +0.5, the 90% interval leaves +0.3
        st = pb.quality_stats(cpu, arm("server", lambda i: 8 + i % 7 + (1 if i % 2 else 0)), draws=300)
        self.assertGreater(st["floor"]["diff_ci90"]["hi"], 0.3)
        self.assertEqual((st["verdict"]["floor_ok"], st["verdict"]["equivalent"]), (False, False))
        # same floors, 3% more sims per decision on the server
        st = pb.quality_stats(cpu, arm("server", sims=2472), draws=300)
        self.assertTrue(st["verdict"]["floor_ok"])
        self.assertAlmostEqual(st["sims_per_decision"]["ratio"], 1.03)
        self.assertEqual((st["verdict"]["ratios_ok"], st["verdict"]["equivalent"]), (False, False))

    def test_a_dead_server_cannot_make_two_cpu_arms_equivalent(self):
        cpu = [qrow(f"S{i:03d}", 8 + i % 7, "cpu") for i in range(100)]
        dead = [qrow(f"S{i:03d}", 8 + i % 7, "cpu") for i in range(100)]  # the "server" arm that fell back every call
        st = pb.quality_stats(cpu, dead, draws=100)
        v = st["verdict"]
        self.assertEqual((v["floor_ok"], v["boss_ok"], v["ratios_ok"], v["pairs_ok"]), (True, True, True, True))
        self.assertEqual((v["paths_ok"], v["equivalent"]), (False, False))
        self.assertEqual(st["arm_paths"], {"server_arm_server_share": 0.0, "cpu_arm_cpu_share": 1.0})

    def test_served_share_gate_is_99_percent_and_the_cpu_arm_must_be_all_cpu(self):
        cpu = [qrow(f"S{i:03d}", 8 + i % 7, "cpu") for i in range(100)]

        def srv(bad):
            return [qrow(f"S{i:03d}", 8 + i % 7, "mixed" if i < bad else "server") for i in range(100)]

        self.assertTrue(pb.quality_stats(cpu, srv(1), draws=100)["verdict"]["equivalent"])  # 99 of 100
        v = pb.quality_stats(cpu, srv(2), draws=100)["verdict"]  # 98 of 100
        self.assertEqual((v["paths_ok"], v["equivalent"]), (False, False))
        dirty = list(cpu)
        dirty[0] = qrow("S000", 8, "server")
        v = pb.quality_stats(dirty, srv(0), draws=100)["verdict"]
        self.assertEqual((v["cpu_share"], v["paths_ok"], v["equivalent"]), (0.99, False, False))

    def test_a_missing_run_is_not_a_clean_pair(self):
        cpu = [qrow(f"S{i:03d}", 8, "cpu") for i in range(10)]
        srv = [qrow(f"S{i:03d}", 8, "server") for i in range(9)]
        st = pb.quality_stats(cpu, srv, draws=100)
        self.assertEqual((st["paired_seeds"], st["unpaired_seeds"], st["crashed_pairs_dropped"]), (9, 1, 0))
        self.assertFalse(st["verdict"]["pairs_ok"])


class ThroughputReport(unittest.TestCase):
    @staticmethod
    def row(served, fallback=0, exact=True):
        base = pb.row_metrics([{"dec": 100, "sims": 4000, "cpu": 10.0}], 5.0, {"cpu_s": 1.0, "calls": 500, "batches": 100})
        return {"arm": "server", "procs": 2, "repeat": 0, **base, "served_share": served, "fallback_calls": fallback,
                "latency_ms": None, "server": {"baseline_exact": exact}, "load_before": 50, "load_after": 60}

    def test_a_clean_server_row_has_no_warning(self):
        self.assertEqual(pb.throughput_warnings([self.row(1.0)]), [])
        self.assertEqual(pb.throughput_warnings([{**self.row(None), "arm": "cpu"}]), [])

    def test_fallbacks_and_an_inexact_baseline_warn(self):
        w = pb.throughput_warnings([self.row(0.75, fallback=12), self.row(1.0, exact=False)])
        self.assertEqual(len(w), 2)
        self.assertIn("75.0% of tasks tagged server, 12 calls fell back", w[0])
        self.assertIn("baseline", w[1])
        self.assertEqual(len(pb.throughput_warnings([self.row(1.0, fallback=3)])), 1)  # calls fell back although every task was tagged

    def test_the_table_has_a_served_column(self):
        table = pb.table_throughput([self.row(0.5)])
        self.assertIn("tasks tagged server |", table.splitlines()[0])
        self.assertIn("| 50.0% |", table)


class DivergenceAndParts(unittest.TestCase):
    def test_tokens_rebuild_from_concatenated_arrays(self):
        rng = np.random.default_rng(3)
        runs = fake_part_runs(rng, [[5, 9, 7], [], [12, 4]])  # the middle run has no samples
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "combat-g3-0000.npz"
            n = gen_loop.write_part(path, runs, 3, ["server", "cpu", "server"])
            part = gen_loop.read_part(path)
        self.assertEqual(n, 5)
        self.assertEqual(part["lengths"].tolist(), [5, 9, 7, 12, 4])
        self.assertEqual(part["feats"].shape, (37, 18))
        flat = []
        for _, arrays in runs:
            if arrays is None:
                continue
            offsets = np.concatenate([[0], np.cumsum(arrays["lengths"])])
            for i in range(arrays["lengths"].size):
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                flat.append(FakeTokens(*(arrays[k][lo:hi] for k in gen_loop.TOKEN_KEYS)))
        self.assertEqual(len(flat), 5)
        order = [3, 0, 4, 2, 1]  # any rows, in any order
        got = pt.rebuild_tokens(part, order, make=FakeTokens)
        for g, i in zip(got, order, strict=True):
            self.assertEqual(len(g), part["lengths"][i])
            for key in gen_loop.TOKEN_KEYS:
                np.testing.assert_array_equal(getattr(g, key), getattr(flat[i], key))
        self.assertEqual(pt.TOKEN_KEYS, gen_loop.TOKEN_KEYS)

    def test_sample_digests_follow_the_run_that_took_them(self):
        rng = np.random.default_rng(5)
        runs = fake_part_runs(rng, [[5, 9], [7], [6, 6, 6]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "combat-g3-0000.npz"
            gen_loop.write_part(path, runs, 3)
            part = gen_loop.read_part(path)
        digests = pb.sample_digests(part)
        self.assertEqual({s: len(v) for s, v in digests.items()}, {"SEED0000": 2, "SEED0001": 1, "SEED0002": 3})
        self.assertEqual(digests, pb.sample_digests(part))
        self.assertEqual(len({d for v in digests.values() for d, _ in v}), 6)
        # one changed visit count changes that sample's digest alone
        part2 = dict(part)
        part2["visits"] = part["visits"].copy()
        part2["visits"][1, 0] += 1
        other = pb.sample_digests(part2)
        self.assertEqual([a[0] == b[0] for a, b in zip(digests["SEED0000"], other["SEED0000"], strict=True)], [True, False])
        self.assertEqual(digests["SEED0002"], other["SEED0002"])

    def test_first_divergence(self):
        rec = {"decisions": [{"floor": 1, "chosen": 0}, {"floor": 2, "chosen": 1}], "fights": [{"floor": 1}, {"floor": 3}]}
        same = [("a", 0), ("b", 0), ("c", 1)]
        self.assertIsNone(pb.first_divergence(rec, same, dict(rec), list(same)))
        d = pb.first_divergence(rec, same, rec, [("a", 0), ("b", 0), ("X", 1)])
        self.assertEqual((d["level"], d["index"], d["fight"], d["floor"]), ("search", 2, 1, 3))
        short = pb.first_divergence(rec, same, rec, same[:2])
        self.assertEqual((short["level"], short["index"]), ("search-count", 2))
        rec2 = {**rec, "decisions": [{"floor": 1, "chosen": 0}, {"floor": 2, "chosen": 0}]}
        d = pb.first_divergence(rec, same, rec2, list(same))
        self.assertEqual((d["level"], d["index"], d["floor"]), ("decision", 1, 2))
        timed = {**rec, "fights": [{"floor": 1, "seconds": 12.0}, {"floor": 3, "seconds": 0.3}]}
        timed2 = {**rec, "fights": [{"floor": 1, "seconds": 0.2}, {"floor": 3, "seconds": 9.9}]}
        self.assertIsNone(pb.first_divergence(timed, same, timed2, list(same)))  # wall-clock fields are not identity
        rec3 = {**rec, "fights": [{"floor": 1}, {"floor": 3, "hp": 1}]}
        d = pb.first_divergence(rec, same, rec3, list(same))
        self.assertEqual((d["level"], d["index"]), ("fight", 1))


class Tolerance(unittest.TestCase):
    def rows(self, n=6):
        rng = np.random.default_rng(1)
        cpu = np.full((n, 32), -np.inf, dtype=np.float32)
        cpu[:, :5] = rng.normal(size=(n, 5)).astype(np.float32)
        legal = np.zeros((n, 32), dtype=np.int8)
        legal[:, :4] = 1
        return cpu, legal

    def test_equal_logits_give_no_difference(self):
        cpu, legal = self.rows()
        r = pt.compare(cpu, cpu.copy(), legal)
        self.assertEqual(r["logit"]["max"], 0.0)
        self.assertEqual(r["logit"]["n"], 6 * 5)
        self.assertEqual((r["mismatch_finite"], r["flips_action_slots"]["flips"], r["prob"]["flips"]), (0, 0, 0))
        self.assertEqual(r["flips_action_slots"]["positions"], 6)

    def test_small_noise_is_measured_and_a_flip_is_counted_above_the_gap(self):
        cpu, legal = self.rows()
        gpu = cpu.copy()
        gpu[:, :5] += np.float32(1e-6)
        r = pt.compare(cpu, gpu, legal)
        self.assertLess(r["logit"]["max"], 1.5e-6)
        self.assertEqual(r["flips_action_slots"]["flips"], 0)
        top = int(np.argmax(cpu[0]))
        other = (top + 1) % 5
        flipped = cpu.copy()
        flipped[0, other], flipped[0, top] = cpu[0, top] + 1e-3, cpu[0, top] - 1e-3  # a clear change of the best action
        r = pt.compare(cpu, flipped, legal)
        self.assertEqual(r["flips_action_slots"]["flips"], 1)
        self.assertGreater(r["logit"]["max"], 1e-4)

    def test_a_tie_is_not_a_flip_and_finiteness_is_checked(self):
        cpu, legal = self.rows(2)
        cpu[0, :5] = [1.0, 1.0, 0.0, -1.0, -2.0]  # identical cards: a tie at the top
        gpu = cpu.copy()
        gpu[0, 0], gpu[0, 1] = 1.0 - 1e-7, 1.0 + 1e-7
        r = pt.compare(cpu, gpu, legal)
        self.assertEqual(r["flips_action_slots"]["flips"], 0)
        self.assertEqual(r["flips_action_slots"]["below_gap_positions"], 1)
        self.assertEqual(r["flips_action_slots"]["below_gap_flips"], 1)  # listed apart, not counted
        gpu[1, 4] = -np.inf
        self.assertEqual(pt.compare(cpu, gpu, legal)["mismatch_finite"], 1)
        self.assertFalse(pt.gate([pt.compare(cpu, gpu, legal)])["pass"])


class ToleranceInputs(unittest.TestCase):
    def test_newest_parts_skips_temporary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("combat-g3-0000.npz", "combat-g3-0001.tmp.npz"):  # the temporary one is written last, so the newest
                np.savez(Path(tmp) / name, lengths=np.array([3, 4]))
            got = pt.newest_parts(str(Path(tmp) / "combat-g3-*.npz"), 1)
        self.assertEqual([Path(g).name for g in got], ["combat-g3-0000.npz"])

    def test_the_digest_names_the_positions(self):
        rng = np.random.default_rng(4)
        toks = [fake_tokens(rng, 5), fake_tokens(rng, 7)]
        legal = np.ones((2, 32), dtype=np.int8)
        d = pt.positions_digest(toks, legal)
        self.assertEqual(d, pt.positions_digest(list(toks), legal.copy()))
        self.assertNotEqual(d, pt.positions_digest(toks[::-1], legal))
        other = legal.copy()
        other[0, 3] = 0
        self.assertNotEqual(d, pt.positions_digest(toks, other))
        moved = FakeTokens(toks[0].types, toks[0].ids, toks[0].owners, toks[0].feats + np.float32(1e-6))
        self.assertNotEqual(d, pt.positions_digest([moved, toks[1]], legal))


if __name__ == "__main__":
    unittest.main()
