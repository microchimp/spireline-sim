"""tools/solver/coverage.py (S6): the readings from one deepening run, the seeded stratified sample, the curve tables."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "solver"))
import coverage as cov  # noqa: E402


def hz(*marks):
    """Horizon marks: (horizon, end, cumulative nodes, seconds, exact, estimate)."""
    return [{"horizon": h, "end": e, "nodes": n, "steps": n * 3, "seconds": s, "exact": x, "estimate_log10": est}
            for h, e, n, s, x, est in marks]


class ReadingTests(unittest.TestCase):
    def test_exact_needs_the_exact_horizon_within_the_budget(self):
        marks = hz((0, "done", 50, 0.001, False, None), (1, "done", 4000, 0.05, False, 3.1), (2, "done", 90_000, 2.0, True, 4.5))
        self.assertFalse(cov.exact_within_nodes(marks, 10**3))
        self.assertFalse(cov.exact_within_nodes(marks, 10**4))
        self.assertTrue(cov.exact_within_nodes(marks, 10**5))
        self.assertTrue(cov.exact_within_nodes(marks, 10**7))
        self.assertEqual(cov.deepest_within_nodes(marks, 10**4), 1)
        self.assertEqual(cov.deepest_within_nodes(marks, 100), 0)

    def test_the_size_estimate_skips_a_horizon_at_a_small_budget(self):
        # Horizon 2's estimate is 10^5.5: a budget of 10^4 skips it (5.5 > 4 + 1) even though it finished in 9,000 nodes.
        marks = hz((0, "done", 50, 0.001, False, None), (1, "done", 900, 0.01, False, 2.0), (2, "done", 9000, 0.1, True, 5.5))
        self.assertFalse(cov.exact_within_nodes(marks, 10**4))
        self.assertTrue(cov.exact_within_nodes(marks, 10**5))
        self.assertEqual(cov.deepest_within_nodes(marks, 10**4), 1)

    def test_a_cut_or_refused_horizon_ends_the_reading(self):
        self.assertFalse(cov.exact_within_nodes(hz((0, "done", 10, 0.0, False, None), (1, "budget", 10**6, 9.0, False, 3.0)), 10**7))
        self.assertFalse(cov.exact_within_nodes(hz((0, "refused", 10, 0.0, False, None)), 10**7))
        self.assertTrue(cov.exact_within_nodes(hz((0, "done", 10, 0.0, True, None)), 10**3))

    def test_which_solves_a_larger_cap_can_change(self):
        cut = {"status": 1, "stats": {"horizons": hz((0, "done", 50, 0.0, False, None), (1, "budget", 10**6, 9.0, False, 5.0))}}
        near = {"status": 1, "stats": {"horizons": hz((0, "done", 50, 0.0, False, None), (1, "estimate", 50, 0.0, False, 7.6))}}
        far = {"status": 1, "stats": {"horizons": hz((0, "done", 50, 0.0, False, None), (1, "estimate", 50, 0.0, False, 8.4))}}
        done = {"status": 0, "stats": {"horizons": hz((0, "done", 50, 0.0, True, None))}}
        refused = {"status": -3, "stats": {"horizons": []}}
        self.assertEqual([cov.needs_larger_cap(x, 10**7) for x in (cut, near, far, done, refused)], [True, True, False, False, False])

    def test_time_reading_uses_cumulative_seconds_and_the_cap_estimate(self):
        marks = hz((0, "done", 50, 0.004, False, None), (1, "done", 4000, 0.3, True, 3.0))
        self.assertFalse(cov.exact_within_seconds(marks, 0.01, 10**7))
        self.assertFalse(cov.exact_within_seconds(marks, 0.1, 10**7))
        self.assertTrue(cov.exact_within_seconds(marks, 1.0, 10**7))
        self.assertFalse(cov.exact_within_seconds(hz((0, "done", 50, 0.004, False, None), (1, "done", 4000, 0.3, True, 9.5)), 10.0, 10**7))


def decision(src, seed, k, floor, node_type, act, turn, run_floor, run_end):
    return {"source": src, "seed": seed, "ix": 1, "k": k, "floor": floor, "node_type": node_type, "act": act, "turn": turn,
            "run_floor": run_floor, "run_end": run_end}


class SampleTests(unittest.TestCase):
    def rows(self):
        rows = []
        for r in range(6):
            seed = f"S{r}"
            k = 0
            for floor, nt in ((1, 1), (5, 2), (17, 6)):
                for turn in (1, 1, 2, 3):
                    rows.append(decision("night/a/g7-emu", seed, k, floor, nt, 1, turn, 17, "death" if r % 2 else "cut"))
                    k += 1
        return rows

    def test_fights_carry_kind_act_and_outcome(self):
        fights = cov.fights_of(self.rows())
        self.assertEqual(len(fights), 18)
        boss = [f for f in fights if f["kind"] == "boss"]
        self.assertEqual({f["won"] for f in boss}, {False, None})  # a death on the boss floor; a cut run's last fight unknown
        self.assertTrue(all(f["won"] for f in fights if f["kind"] != "boss"))
        self.assertEqual({f["decisions"] for f in fights}, {4})
        self.assertEqual({f["turns"] for f in fights}, {3})

    def test_sample_is_seeded_stratified_and_leaves_unknown_outcomes_out(self):
        fights = cov.fights_of(self.rows())
        a = cov.sample_fights(fights, 5, 7)
        self.assertEqual(a, cov.sample_fights(fights, 5, 7))
        self.assertEqual({cov.stratum(f) for f in a}, {"act1/hallway/won", "act1/elite/won", "act1/boss/lost"})
        self.assertTrue(all(f["won"] is not None for f in cov.sample_fights(fights, 100, 7)))
        self.assertEqual(len(cov.sample_fights(fights, 100, 7)), 15)

    def test_source_label_has_no_drive_user_or_host(self):
        self.assertEqual(cov.source_label("D:/x/referee/lab/nightly/2026-10-27/g7-emu"), "nightly/2026-10-27/g7-emu")
        self.assertEqual(cov.source_label(r"C:\Users\someone\lab\nightly\2026-10-26\g5-emu"), "nightly/2026-10-26/g5-emu")


class CurveTests(unittest.TestCase):
    def test_curve_tables(self):
        dec = [{"id": f"d{i}", "fight": f"f{i // 2}", "kind": "boss" if i < 2 else "hallway", "act": 1, "won": i >= 2, "turn": 1 + i % 2,
                "turns_left": 1 - i % 2} for i in range(4)]
        solves = [
            {"id": "d0", "status": 1, "cap": 10**6, "stats": {"horizons": hz((0, "done", 10**6, 30.0, False, None))}},
            {"id": "d1", "status": 0, "cap": 10**6, "stats": {"horizons": hz((0, "done", 200, 0.005, True, None))}},
            {"id": "d2", "status": 0, "cap": 10**6, "stats": {"horizons": hz((0, "done", 20, 0.001, False, None), (1, "done", 5000, 0.5, True, 3.0))}},
            {"id": "d3", "status": -3, "reason": "draw-peek: x", "cap": 10**6, "stats": {"horizons": []}},
        ]
        c = cov.curve(dec, solves, 10**6)
        self.assertEqual(c["decisions"], 4)
        self.assertEqual(c["overall"]["all"]["nodes_1e3"], 0.25)
        self.assertEqual(c["overall"]["all"]["nodes_1e4"], 0.5)
        self.assertNotIn("nodes_1e7", c["overall"]["all"])
        self.assertEqual(c["overall"]["all"]["time_0.01s"], 0.25)
        self.assertEqual(c["overall"]["all"]["time_1s"], 0.5)
        self.assertEqual(c["by_kind"]["boss"]["nodes_1e6"], 0.5)
        self.assertEqual(c["refused"], {"draw-peek": 1})
        self.assertEqual(c["fights_fully_exact"]["10s"], 0.0)
        self.assertEqual(c["fights_exact_from_turns_left_at_10s"], {"0": 1, "never": 1})
        json.dumps(c)

    def test_results_hold_no_absolute_path(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.jsonl"
            cov.write_jsonl(p, [{"source": cov.source_label(r"D:\a\b\nightly\2026-10-27\g7-emu")}])
            text = p.read_text(encoding="utf-8")
            self.assertNotIn(":\\", text)
            self.assertNotIn(":/", text)
            self.assertEqual(cov.read_jsonl(p)[0]["source"], "nightly/2026-10-27/g7-emu")

    def test_audit_report(self):
        rows = [{"stats": {"merge_audit": {"distinct_full": 10, "distinct_masked": 5, "distinct_masked_keep_block": 8}}},
                {"stats": {"merge_audit": {"distinct_full": 30, "distinct_masked": 30, "distinct_masked_keep_block": 30}}}]
        r = cov.audit_report(rows)
        self.assertEqual(r["ratio_pooled"], 0.875)
        self.assertEqual(r["ratio_keep_block_pooled"], 0.95)
        self.assertEqual(r["ratio_median"], 0.75)


def card(def_id, upgraded=False):
    return {"def_id": def_id, "upgraded": upgraded}


def enemy(slot, def_id, hp):
    return {"def_id": def_id, "slot": slot, "hp": hp}


DOC = {"player": {"max_hp": 80}, "pending_selection": None, "piles": {"hand": [card(1), card(2), card(1), card(3)]},
       "potions": {"slots": [7, 7, 0]}, "enemies": [enemy(0, 9, 10), enemy(1, 9, 10), enemy(2, 4, 30)]}
# The solver's keys at DOC: card 0 (an attack: targets 0 and 2; enemy 1 equals enemy 0), card 1 (no target), card 2 merged
# into card 0, card 3 (no target), end turn (4), the first potion (5, no target); the second potion (6) merged into it.
Q = {(0, 0): 0.70, (0, 2): 0.75, (1, -1): 0.60, (3, -1): 0.65, (4, -1): 0.55, (5, -1): 0.72}


class RegretTests(unittest.TestCase):
    def test_q_rows_old_and_new(self):
        self.assertEqual(cov.q_map({"id": "x", "status": 0}), {})
        self.assertEqual(cov.q_map({"q": [[0, -1, 0.5], [1, 2, 0.25]]}), {(0, -1): 0.5, (1, 2): 0.25})

    def test_the_curve_reads_rows_with_and_without_q(self):
        dec = [{"id": f"d{i}", "fight": "f", "kind": "hallway", "act": 1, "won": True, "turn": 1, "turns_left": 0} for i in range(2)]
        old = {"id": "d0", "status": 0, "cap": 10**6, "stats": {"horizons": hz((0, "done", 200, 0.005, True, None))}}
        new = dict(old, id="d1", q=[[0, -1, 0.9]], value=0.9, exact=True, exact_mass=1.0, best=[0, -1])
        c = cov.curve(dec, [old, new], 10**6)
        self.assertEqual(c["overall"]["all"]["time_0.01s"], 1.0)

    def test_the_played_move_maps_onto_the_solver_keys(self):
        self.assertEqual(cov.map_played(Q, 0, 2, DOC), ((0, 2), "direct"))
        self.assertEqual(cov.map_played(Q, 1, 0, DOC), ((1, -1), "untargeted"))  # a target given to a move without one
        self.assertEqual(cov.map_played(Q, 0, 1, DOC), ((0, 0), "merged"))  # enemy 1 equals enemy 0
        self.assertEqual(cov.map_played(Q, 2, 2, DOC), ((0, 2), "merged"))  # card 2 equals card 0
        self.assertEqual(cov.map_played(Q, 2, 1, DOC), ((0, 0), "merged"))  # both merges
        self.assertEqual(cov.map_played(Q, 6, -1, DOC), ((5, -1), "merged"))  # the second potion equals the first
        self.assertEqual(cov.map_played(Q, 3, -1, DOC), ((3, -1), "direct"))
        self.assertEqual(cov.map_played({(0, 1): 0.5}, 0, -1, None), ((0, 1), "untargeted"))  # one target, none named
        self.assertEqual(cov.map_played(Q, 9, -1, DOC), (None, "unmapped"))
        other = dict(DOC, piles={"hand": [card(1), card(2), card(1, True), card(3)]})
        self.assertEqual(cov.map_played(Q, 2, 2, other), (None, "unmapped"))  # an upgraded copy is a different card

    def test_regret_and_the_checks_read_exact_solves_only(self):
        d = {"id": "s|1|k0001", "fight": "s|1|3", "source": "nightly/2026-10-27/g7-emu", "kind": "elite", "won": True, "turns_left": 2}
        solve = {"status": 0, "exact": True, "q": [[a, t, v] for (a, t), v in Q.items()]}
        played = {"action": 2, "target": 2, "replay_same": True}  # card 2 into enemy 2: card 0's value 0.75, the best
        r = cov.regret_row(d, solve, played, None, None, DOC)
        self.assertEqual((r["source"], r["how"], r["regret"]), ("g7", "merged", 0.0))
        r = cov.regret_row(d, solve, dict(played, action=4, target=-1), None, None, DOC)  # end turn: 0.75 - 0.55
        self.assertAlmostEqual(r["regret"], 0.20)
        self.assertAlmostEqual(r["regret_hp"], 0.20 * 160)
        self.assertTrue(r["decisive"])
        cut = dict(solve, status=1, exact=False)
        self.assertFalse(cov.regret_row(d, cut, played, None, None, DOC)["l0_exact"])
        self.assertFalse(cov.regret_row(d, solve, dict(played, replay_same=False), None, None, DOC)["replay_same"])
        # Lethal: a move with chance 1 exists and end turn's is 0: missed. A cut check has no reading.
        lethal = {"status": 0, "exact": True, "value": 1.0, "q": [[0, 2, 1.0], [4, -1, 0.0]]}
        r = cov.regret_row(d, solve, dict(played, action=4, target=-1), lethal, dict(lethal, exact=False), DOC)
        self.assertEqual((r["lethal"]["done"], r["lethal"]["avail"], r["lethal"]["missed"]), (True, True, True))
        self.assertEqual(r["survival"], {"done": False})
        self.assertFalse(cov.regret_row(d, solve, played, lethal, None, DOC)["lethal"]["missed"])
        near = dict(lethal, value=0.9999, q=[[0, 2, 0.9999], [4, -1, 0.0]])
        self.assertFalse(cov.regret_row(d, solve, dict(played, action=4, target=-1), near, None, DOC)["lethal"]["missed"])

    def test_the_report(self):
        base = {"fight": "f1", "source": "g7", "kind": "boss", "won": False, "turns_left": 0, "played": True, "replay_same": True}
        rows = [dict(base, id="a", l0_exact=True, how="direct", q_best=0.8, q_worst=0.8, decisive=True, regret=0.0, regret_hp=0.0,
                     lethal={"done": True, "avail": True, "played": 0.5, "how": "direct", "missed": True}, survival={"done": False}),
                dict(base, id="b", fight="f2", source="g5", kind="hallway", turns_left=3, l0_exact=True, how="direct", q_best=0.7,
                     q_worst=0.6, decisive=True, regret=0.05, regret_hp=8.0, lethal={"done": False},
                     survival={"done": True, "avail": True, "played": 1.0, "how": "direct", "missed": False}),
                dict(base, id="c", fight="f2", source="g5", l0_exact=False, lethal={"done": False}, survival={"done": False})]
        rep = cov.regret_report(rows, 4)
        o = rep["overall"]
        self.assertEqual((o["l0_exact"], o["l0_valued"], o["loss_over_0.01"], o["decisive"]), (2, 2, 1, 2))
        self.assertAlmostEqual(o["mean_regret_l0"], 0.025)
        self.assertAlmostEqual(o["regret_hp_per_fight"], 2.0)
        self.assertEqual((o["lethal"]["checked"], o["lethal"]["missed_decisions"], o["lethal"]["missed_per_100_fights"]), (1, 1, 25.0))
        self.assertEqual((o["survival"]["checked"], o["survival"]["missed_per_100_fights"]), (1, 0.0))
        self.assertEqual(sorted(rep["by_source"]), ["g5", "g7"])
        self.assertEqual(rep["by_turns_left"]["0"]["lethal"]["fights_with_miss"], 1)
        self.assertEqual(rep["mapping"], {"direct": 2})
        json.dumps(rep)


class MemoryTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "GetProcessMemoryInfo is Windows only")
    def test_the_peak_working_set_is_read_and_never_falls(self):
        a = cov.peak_working_set_mb()
        block = bytearray(64 * 2**20)  # touch 64 MB
        block[:: 4096] = b"x" * len(block[:: 4096])
        b = cov.peak_working_set_mb()
        self.assertGreater(a, 0)
        self.assertGreaterEqual(b, a + 32)
        del block
        self.assertGreaterEqual(cov.peak_working_set_mb(), b)

    def test_the_new_subcommands_parse(self):
        for cmd in ("memory", "macro-ab", "played", "regret"):
            with self.assertRaises(SystemExit):
                cov.main([cmd, "--help"])


if __name__ == "__main__":
    unittest.main()
