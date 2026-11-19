"""tools/lab/ledger_strict.py and tools/lab/act3_boss_table.py: the strict boss reading's re-score of the ledger, and the boss table. Constructed
data only (a tiny agent folder in a temp dir); the lab's own folder is not read here."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.lab import act3_boss_table as bt  # noqa: E402
from tools.lab import keep_v2 as k2  # noqa: E402
from tools.lab import ledger_strict as ls  # noqa: E402

HEADER = ("written", "cand_gen", "base_gen", "seed_block", "seeds", "decision", "cand_dir", "rule", "route", "guard")


def nodes(*, second: bool = False, act3_boss: bool = True) -> str:
    return "M" * 15 + "B" + "A" + "M" * 14 + "B" + "A" + "M" * 13 + ("B" if act3_boss else "") + ("B" if second else "")


def run(seed: str, kind: str) -> dict:
    """One arm row. kinds: early (died on floor 20), full48 (the first Act 3 boss's bar read 1.0, died on floor 48), past (went past, died on
    floor 49), win."""
    if kind == "early":
        return {"seed": seed, "floor": 20, "win": False, "end": "death", "nodes": "M" * 15 + "BAMM", "boss_damage_acts": [[0, 1.0]]}
    if kind == "full48":
        return {"seed": seed, "floor": 48, "win": False, "end": "death", "nodes": nodes(), "boss_damage_acts": [[0, 1.0], [1, 1.0], [2, 1.0]]}
    if kind == "past":
        return {"seed": seed, "floor": 49, "win": False, "end": "death", "nodes": nodes(second=True),
                "boss_damage_acts": [[0, 1.0], [1, 1.0], [2, 1.0], [2, 0.3]]}
    return {"seed": seed, "floor": 49, "win": True, "end": "win", "nodes": nodes(second=True),
            "boss_damage_acts": [[0, 1.0], [1, 1.0], [2, 1.0], [2, 1.0]]}


def write_arm(agent: Path, gen: int, cand_dir: str, block: str, rows: list[dict]) -> None:
    d = agent / f"gen{gen}" / cand_dir / "arms" / f"{block}-a3-v1"
    d.mkdir(parents=True)
    (d / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def write_ledger(agent: Path, rows: list[dict]) -> None:
    lines = ["\t".join(HEADER)] + ["\t".join(str(r.get(c, "")) for c in HEADER) for r in rows]
    (agent / "promotions.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


class LedgerStrictTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.agent = Path(self.tmp.name)
        seeds = [f"S{i}" for i in range(10)]
        # generation 1 (candidate) against generation 0 on block E1: the candidate's bar reads 1.0 on three runs that died on floor 48 and one
        # that went past; the incumbent's on two that died there
        write_arm(self.agent, 1, "candidate", "E1", [run(s, k) for s, k in zip(seeds, ["full48"] * 3 + ["past"] + ["early"] * 6, strict=True)])
        d = self.agent / "gen0" / "arms" / "E1-a3-v1"
        d.mkdir(parents=True)
        (d / "runs.jsonl").write_text("".join(json.dumps(run(s, k)) + "\n" for s, k in zip(seeds, ["full48"] * 2 + ["win"] + ["early"] * 7, strict=True)),
                                      encoding="utf-8")
        # generation 2 (a named candidate) against generation 1 on blocks E2 and E3 (pooled): the incumbent's arm is under generation 1's promoted folder
        for block, c_kinds in (("E2", ["past", "early"]), ("E3", ["full48", "past"])):
            write_arm(self.agent, 2, "candidate-x", block, [run(f"{block}a", c_kinds[0]), run(f"{block}b", c_kinds[1])])
            write_arm(self.agent, 1, "candidate", block, [run(f"{block}a", "early"), run(f"{block}b", "full48")])
        write_ledger(self.agent, [
            {"written": "2026-11-01T00:00:00Z", "cand_gen": 1, "base_gen": 0, "seed_block": "E1", "seeds": 10, "decision": "promoted"},
            {"written": "2026-11-02T00:00:00Z", "cand_gen": 2, "base_gen": 1, "seed_block": "E3", "seeds": 4, "decision": "rejected", "cand_dir": "candidate-x",
             "rule": "v3", "guard": "v3 route none | pooled E3 (2) + E2 (2) = 4 pairs"},
            {"written": "2026-11-03T00:00:00Z", "cand_gen": 3, "base_gen": 2, "seed_block": "E9", "seeds": 10, "decision": "rejected"},
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def test_each_judgement_is_counted_under_both_readings(self):
        rows = {r["block"]: r for r in ls.rescore(self.agent)}
        r = rows["E1"]
        self.assertEqual((r["pairs"], r["arms"]), (10, "ok"))
        self.assertEqual((r["b3a_loose_cand"], r["b3a_loose_base"]), (4, 3))  # a full bar on floor 48 counts: three of them and one past; the win and two bars
        self.assertEqual((r["b3a_strict_cand"], r["b3a_strict_base"]), (1, 1))  # only the run past the boss; only the win
        self.assertEqual((r["wins_cand"], r["wins_base"]), (0, 1))
        self.assertEqual((r["b1_loose_cand"], r["b1_strict_cand"]), (10, 10))  # Act 1 reads the same both ways
        self.assertEqual(r["b3a_strict_diff_per100"], 0.0)

    def test_a_pooled_judgement_counts_both_blocks_and_a_missing_arm_is_said(self):
        rows = ls.rescore(self.agent)
        pooled = rows[1]
        self.assertEqual((pooled["pairs"], pooled["arms"]), (4, "ok"))
        self.assertEqual((pooled["b3a_loose_cand"], pooled["b3a_strict_cand"]), (3, 2))
        self.assertEqual((pooled["b3a_loose_base"], pooled["b3a_strict_base"]), (2, 0))  # the incumbent's full bar on floor 48, once in each block
        missing = rows[2]
        self.assertTrue(missing["arms"].startswith("missing"))
        self.assertNotIn("pairs", missing)

    def test_the_generation_table_pools_every_arm_of_a_promoted_agent(self):
        g = {r["gen"]: r for r in ls.generations(self.agent)}
        self.assertEqual(sorted(g), [1])  # generation 2 was never promoted
        self.assertEqual((g[1]["arms"], g[1]["runs"]), (3, 14))  # E1 (10) + E2 (2) + E3 (2)
        self.assertEqual((g[1]["b3a_loose"], g[1]["b3a_strict"]), (6, 1))  # E1's 4 and the full bars of E2 and E3 under loose; E1's one past under strict

    def test_the_tsv_has_one_row_per_judgement_and_the_ledger_is_untouched(self):
        before = (self.agent / "promotions.tsv").read_text(encoding="utf-8")
        out = self.agent / "out.tsv"
        self.assertEqual(ls.main(["--agent", str(self.agent), "--out", str(out), "--generations", str(self.agent / "gen.tsv")]), 0)
        lines = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0].split("\t"), list(ls.LEDGER_COLUMNS))
        self.assertEqual((self.agent / "promotions.tsv").read_text(encoding="utf-8"), before)


class BossTableTests(unittest.TestCase):
    BOSSES = {"A": ("TEST_SUBJECT_BOSS", "QUEEN_BOSS"), "B": ("QUEEN_BOSS", "TEST_SUBJECT_BOSS"), "C": ("AEONGLASS_BOSS", "QUEEN_BOSS")}

    def identity(self, seed: str):
        return self.BOSSES[seed[0]]

    def test_the_table_says_which_boss_reads_full_without_dying(self):
        rows = [run("A1", "full48"), run("A2", "full48"), run("A3", "past"),  # the Test Subject: two full bars that died on 48, one past
                run("B1", "past"), run("B2", "win"), run("B3", "early"), run("C1", "past"),
                {**run("C2", "full48"), "boss_damage_acts": [[0, 1.0], [1, 1.0], [2, 0.4]]}]  # a part bar, died there
        t = bt.tabulate(rows, self.identity)
        ts = t["TEST_SUBJECT_BOSS"]
        self.assertEqual((ts["arrivals"], ts["past"], ts["bar_full"], ts["bar_full_died_there"]), (3, 1, 3, 2))
        self.assertEqual(ts["second_QUEEN_BOSS_arrived"], 1)
        q = t["QUEEN_BOSS"]
        self.assertEqual((q["arrivals"], q["past"], q["bar_full_died_there"], q["second_TEST_SUBJECT_BOSS_wins"]), (2, 2, 0, 1))
        a = t["AEONGLASS_BOSS"]
        self.assertEqual((a["arrivals"], a["past"], a["bar_part"], a["bar_full"]), (2, 1, 1, 1))
        self.assertNotIn("B3", json.dumps(t))  # a run that never began the fight is not an arrival (and no seed is printed)

    def test_a_crash_row_is_no_arrival(self):
        t = bt.tabulate([{"seed": "A1", "floor": 0, "win": False, "crash": True}, run("A2", "full48")], self.identity)
        self.assertEqual(t["TEST_SUBJECT_BOSS"]["arrivals"], 1)

    def test_the_sample_lists_runs_that_read_full_and_died_on_48(self):
        rows = [("gen17/c/arms/E-a3-v1", run("A1", "full48")), ("gen17/c/arms/E-a3-v1", run("A2", "past")), ("gen18/c/arms/E-a3-v1", run("C1", "full48")),
                ("gen9/c/arms/E-a3-v1", run("A3", "full48"))]
        s = bt.sample_died_full(rows, self.identity, 5, [17, 18])
        self.assertEqual([(x["boss"], x["floor"]) for x in s], [("TEST_SUBJECT_BOSS", 48), ("AEONGLASS_BOSS", 48)])
        self.assertNotIn("seed", s[0])

    def test_the_strict_reading_of_these_rows(self):
        self.assertEqual(k2.bosses_beaten(run("x", "full48"))[:3], (True, True, False))
        self.assertEqual(k2.bosses_beaten(run("x", "past"))[:3], (True, True, True))


class BarTraceToolTests(unittest.TestCase):
    def test_the_replay_tool_loads_without_the_emulator_and_says_what_it_needs(self):
        from tools.lab import act3_bar_trace as tr  # noqa: PLC0415 -- stdlib only at import time; the emulator is loaded by main()

        with self.assertRaises(SystemExit) as cm:
            tr.main(["--help"])
        self.assertEqual(cm.exception.code, 0)
        with self.assertRaises(SystemExit):
            tr.main([])  # --gen-dir, --seed and --ix are required


if __name__ == "__main__":
    unittest.main()
