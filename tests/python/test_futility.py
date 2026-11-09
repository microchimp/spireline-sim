"""The futility stop of the sequential judgement (D55 addendum 18; rl/gen_loop.py `futility_check`, `funnel(futility=)`, `promote`;
tools/lab/keep_v2.py `futility_routes`; docs/rl/KEEP_TEST_V2.md).

From a number of paired seeds on, a look that would play on asks whether any route's 95% upper bound of the candidate's gain reaches the gain the
route needs to promote on the whole block; when none does the block ends as a rejection with the rule `futility@<seeds>`. These tests build paired
arms from exact differences (no emulator), run the funnel and `promote` on them, check the ledger, re-run the rule on its normal model, and replay
the lab's recorded blocks where they exist (those skip where the lab's agent folder is absent: SPIRELINE_REFEREE names it)."""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import fight_exam as fe  # noqa: E402
import gen0  # noqa: E402
import gen_loop  # noqa: E402
import keep_v2_fixtures as fx  # noqa: E402
from tools.lab import keep_v2 as k2  # noqa: E402
import round1  # noqa: E402
import round1_data as rd  # noqa: E402
from test_funnel import StubArms, stub_candidate  # noqa: E402

PINNED_COLUMNS = ("written", "cand_gen", "base_gen", "seed_block", "seeds", "d_floor", "floor_lower", "p_floor", "boss_cand", "boss_base", "p_boss",
                  "wins_cand", "wins_base", "decision", "cand_sha", "cand_dir", "attempt", "drift", "guard", "run_head_sha", "macro", "rule", "frontier",
                  "route", "k2_cand", "k2_base", "d_deep", "deep_lower", "p_deep", "floor_lower95")


def pattern(n: int, hi: float, lo: float) -> np.ndarray:
    """Differences in adjacent pairs (hi, lo, hi, lo ...): every even prefix has the mean (hi + lo) / 2 exactly."""
    d = np.empty(n)
    d[0::2], d[1::2] = hi, lo
    return d


def block(n: int, floor_diffs: np.ndarray, deep_every: int = 0, seed: int = 5, prefix: str = "F"):
    """(seeds, candidate rows, incumbent rows): both arms reach the Act 2 (level 1: the Act 1 boss beaten, the Act 2 boss not), the candidate's floor
    is the incumbent's plus `floor_diffs[i]`, and every `deep_every`-th seed the candidate also beats the Act 2 boss (deep_B2 +100 for that seed)."""
    assert np.all(np.asarray(floor_diffs) == np.rint(floor_diffs)), "floors are whole numbers"
    rng = np.random.default_rng(seed)
    base_floor = np.clip(np.rint(rng.normal(24, 3, n)), 18, 30).astype(int)
    seeds = [f"{prefix}{i:09d}" for i in range(n)]
    cand, base = [], []
    for i, s in enumerate(seeds):
        b = fx.run_row(s, 1, np.random.default_rng(i))
        c = fx.run_row(s, 2 if deep_every and i % deep_every == 0 else 1, np.random.default_rng(i + 10**6))
        b["floor"], c["floor"] = int(base_floor[i]), int(base_floor[i] + int(floor_diffs[i]))
        cand.append(c)
        base.append(b)
    return seeds, cand, base


def walk(seeds, cand, base, *, keep_rule: str, futility: int | None, smoke: int = 200, batch: int = 200, frontier: int = 1):
    """`gen_loop.funnel` on whole arms, with a judgement that never promotes (the early promotion is not under test), and what it said."""
    said: list[str] = []

    def judge(c, b):
        return {"decision": {"promote": False, "checks": {}}, "keep_v2": {"route": "", "review": False}}

    out = gen_loop.funnel(seeds, lambda n: (cand, base), judge, smoke=smoke, batch=batch, keep_rule=keep_rule, frontier=frontier,
                          futility=futility, frontier_name="deep_B2", say=said.append)
    return out, said


class RoutesTests(unittest.TestCase):
    """`keep_v2.futility_routes` on summaries: which route is alive against which gain."""

    def summary(self, mean: float, se: float, upper95: float | None = None) -> dict:
        return {"mean": mean, "se": se, "upper95": mean + 1.645 * se if upper95 is None else upper95}

    def test_the_critical_gain_is_the_gain_the_final_test_needs(self):
        z = NormalDist().inv_cdf(0.995)
        self.assertAlmostEqual(k2.critical_gain(0.2, 1000, 3000, 0.005), z * 0.2 * math.sqrt(1 / 3), places=12)
        self.assertAlmostEqual(k2.critical_gain(0.2, 3000, 3000, 0.005), z * 0.2, places=12)  # at the last seed it is the test's own bound
        self.assertLess(k2.critical_gain(0.2, 1000, 3000, 0.005), k2.critical_gain(0.2, 1000, 1000, 0.005))

    def test_a_route_is_alive_while_its_upper_bound_reaches_its_gain(self):
        # 1,000 of 3,000 seeds: floors SE 0.25 (SD 7.9), deep_f SE 1.30 (SD 41)
        need_f, need_a = k2.critical_gain(0.25, 1000, 3000, 0.005), max(k2.MIN_GAIN, k2.critical_gain(1.3, 1000, 3000, 0.005))
        self.assertGreater(need_a, k2.MIN_GAIN)  # the 0.005 critical gain of deep_f (1.93) is above the rule's +1.0 at this spread
        r = k2.futility_routes(self.summary(0.0, 0.25, need_f * 0.99), self.summary(0.0, 1.3, need_a * 0.99), 1000, 3000)
        self.assertTrue(r["futile"])
        self.assertEqual({k: v["alive"] for k, v in r["routes"].items()}, {"A": False, "F": False})
        self.assertIn("falls short of", r["text"])
        r = k2.futility_routes(self.summary(0.0, 0.25, need_f * 1.01), self.summary(0.0, 1.3, need_a * 0.99), 1000, 3000)
        self.assertFalse(r["futile"])  # route F alone keeps the judgement going
        self.assertEqual({k: v["alive"] for k, v in r["routes"].items()}, {"A": False, "F": True})
        r = k2.futility_routes(self.summary(0.0, 0.25, need_f * 0.99), self.summary(0.0, 1.3, need_a * 1.01), 1000, 3000)
        self.assertFalse(r["futile"])  # and route A alone
        self.assertEqual({k: v["alive"] for k, v in r["routes"].items()}, {"A": True, "F": False})

    def test_route_a_never_asks_less_than_the_rules_own_gain_of_one_per_hundred(self):
        # a deep_f spread so small that its critical gain is 0.1: the rule still asks +1.0 per 100
        r = k2.futility_routes(self.summary(0, 0.05), self.summary(0.0, 0.04, upper95=0.9), 1000, 3000)
        self.assertEqual(r["routes"]["A"]["needs"], k2.MIN_GAIN)
        self.assertFalse(r["routes"]["A"]["alive"])
        self.assertTrue(k2.futility_routes(self.summary(0, 0.05, upper95=-1.0), self.summary(0.0, 0.04, upper95=1.0), 1000, 3000)["routes"]["A"]["alive"])

    def test_no_route_promotes_on_a_gain_that_is_not_positive(self):
        r = k2.futility_routes(self.summary(0.0, 0.0, upper95=0.0), self.summary(0.0, 0.0, upper95=0.0), 1000, 3000)
        self.assertTrue(r["futile"])  # identical arms: nothing to promote on, whatever the (zero) critical gain
        self.assertFalse(r["routes"]["F"]["alive"])

    def test_a_bound_that_cannot_be_worked_out_keeps_the_route_alive(self):
        r = k2.futility_routes({"mean": 0.0, "se": math.inf, "upper95": math.nan}, {"mean": 0.0, "se": math.inf, "upper95": math.nan}, 1, 3000)
        self.assertFalse(r["futile"])

    def test_the_v1_rule_has_the_floor_route_alone_at_its_own_alpha(self):
        r = k2.futility_routes(self.summary(0.0, 0.25), None, 1000, 3000, floor_alpha=0.01, routes=("F",))
        self.assertEqual(list(r["routes"]), ["F"])
        self.assertAlmostEqual(r["routes"]["F"]["needs"], NormalDist().inv_cdf(0.99) * 0.25 * math.sqrt(1 / 3), places=12)


class FunnelFutilityTests(unittest.TestCase):
    """`funnel(futility=)` on exact differences: a level candidate stops at 1,000, a better one never does."""

    N = 2000

    def test_a_level_candidate_stops_at_one_thousand_seeds_and_says_which_bounds_fell_short(self):
        seeds, cand, base = block(self.N, pattern(self.N, 6.0, -6.0))
        out, said = walk(seeds, cand, base, keep_rule="v2", futility=1000)
        self.assertEqual((out["stop"]["action"], out["stop"]["n"], out["stop"]["rule"]), ("futility", 1000, "futility@1000"))
        self.assertEqual([x["action"] for x in out["looks"]], ["continue"] * 4 + ["futility"])  # 200 ... 800: not looked at; 1000: stopped
        self.assertNotIn("futility", out["looks"][3])  # nothing was asked before 1,000 seeds
        last = out["looks"][-1]
        self.assertTrue(last["futility"]["futile"])
        self.assertFalse(any(r["alive"] for r in last["futility"]["routes"].values()))
        self.assertEqual(len(said), 1)
        self.assertIn("futility look at 1000 of 2000 seeds", said[0])
        self.assertIn("route A (deep_B2 per 100 runs): 95% upper bound", said[0])
        self.assertIn("route F (floors): 95% upper bound", said[0])
        self.assertEqual(said[0].count("falls short of"), 2)
        self.assertEqual([len(out["cand_rows"]), len(out["base_rows"])], [1000, 1000])  # both arms' rows to the stop, no more

    def test_a_level_candidate_under_the_v1_rule_stops_the_same_way(self):
        seeds, cand, base = block(1200, pattern(1200, 6.0, -6.0))  # a block of 1,200: the floor route alone, at D24's alpha
        out, said = walk(seeds, cand, base, keep_rule="v1", futility=1000)
        self.assertEqual((out["stop"]["action"], out["stop"]["n"]), ("futility", 1000))
        self.assertEqual(list(out["looks"][-1]["futility"]["routes"]), ["F"])

    def test_nothing_is_asked_before_the_first_futility_seed_and_the_switch_is_off_by_default(self):
        seeds, cand, base = block(self.N, pattern(self.N, 6.0, -6.0))
        out, said = walk(seeds, cand, base, keep_rule="v2", futility=1200)
        self.assertEqual((out["stop"]["action"], out["stop"]["n"]), ("futility", 1200))
        out, said = walk(seeds, cand, base, keep_rule="v2", futility=None)
        self.assertEqual((out["stop"]["action"], out["stop"]["n"]), ("final", self.N))  # the funnel itself: no switch, no futility look
        self.assertEqual(said, [])
        self.assertTrue(all("futility" not in x for x in out["looks"]))

    def test_a_clearly_better_candidate_is_never_stopped(self):
        seeds, cand, base = block(self.N, pattern(self.N, 7.0, -4.0))  # +1.5 floors: the early promotion is for the judgement, not for us
        out, said = walk(seeds, cand, base, keep_rule="v2", futility=1000)
        self.assertNotEqual(out["stop"]["action"], "futility")
        self.assertTrue(all(x["action"] != "futility" for x in out["looks"]))
        self.assertTrue(all("falls short" not in s.split("; no route")[0] or "reaches" in s for s in said))

    def test_better_on_floors_only_goes_on_while_the_floor_route_can_reach_its_gain(self):
        seeds, cand, base = block(self.N, pattern(self.N, 6.0, -5.0))  # +0.5 floors (z 2.7 at 1,000, under the early z); deep_f level
        out, said = walk(seeds, cand, base, keep_rule="v2", futility=1000)
        self.assertEqual((out["stop"]["action"], out["stop"]["n"]), ("final", self.N))  # played to the last look, never stopped
        asked = [x for x in out["looks"] if "futility" in x]
        self.assertEqual([x["n"] for x in asked], [1000, 1200, 1400, 1600])  # at 1,800 the floor route clears the early z (eligible: no futility look)
        self.assertEqual(out["looks"][8]["action"], "eligible")  # the judgement stub did not promote it ("held"): the walk went on
        self.assertIn("held", out["looks"][8])
        for x in asked:
            routes = x["futility"]["routes"]
            self.assertTrue(routes["F"]["alive"], x["n"])  # the floors' bound reaches the gain the final test needs ...
            self.assertFalse(routes["A"]["alive"], x["n"])  # ... where deep_f's does not (nothing moved there)
            self.assertFalse(x["futility"]["futile"])
        self.assertEqual(len(said), 4)

    def test_better_on_deep_f_only_goes_on_while_route_a_can_reach_its_gain(self):
        seeds, cand, base = block(self.N, np.zeros(self.N), deep_every=100)  # +1.0 per 100 at the Act 2 boss, floors identical
        out, _ = walk(seeds, cand, base, keep_rule="v2", futility=1000)
        self.assertEqual((out["stop"]["action"], out["stop"]["n"]), ("final", self.N))
        first = out["looks"][4]["futility"]["routes"]
        self.assertTrue(first["A"]["alive"])
        self.assertFalse(first["F"]["alive"])
        self.assertGreaterEqual(first["A"]["upper95"], first["A"]["needs"])

    def test_a_deep_f_gain_under_the_rules_one_per_hundred_cannot_keep_it_alive(self):
        seeds, cand, base = block(self.N, np.zeros(self.N), deep_every=200)  # +0.5 per 100: under the +1.0 the rule asks
        out, _ = walk(seeds, cand, base, keep_rule="v2", futility=1000)
        self.assertEqual((out["stop"]["action"], out["stop"]["n"]), ("futility", 1000))
        self.assertLess(out["looks"][-1]["futility"]["routes"]["A"]["upper95"], k2.MIN_GAIN)

    def test_refusals_come_first_and_are_as_they_were(self):
        seeds, cand, base = block(self.N, pattern(self.N, -4.0, -8.0))  # -6 floors: refused at the smoke block, as without the futility stop
        for fut in (None, 1000, 200, 1):  # even where the futility stop would look at the same seeds, the refusal is the call
            out, _ = walk(seeds, cand, base, keep_rule="v2", futility=fut)
            self.assertEqual((out["stop"]["action"], out["stop"]["n"]), ("refuse", 200))
        seeds, cand, base = block(self.N, pattern(self.N, 6.0, -6.0))  # level: only the futility stop ends it, where nothing refused before
        a, _ = walk(seeds, cand, base, keep_rule="v2", futility=None)
        self.assertEqual((a["stop"]["action"], a["stop"]["n"]), ("final", self.N))  # (without it: all 2,000 seeds)

    def test_the_floor_route_of_v1_uses_the_d24_alpha(self):
        seeds, cand, base = block(self.N, pattern(self.N, 6.0, -6.0))
        out, _ = walk(seeds, cand, base, keep_rule="v1", futility=1000)
        need = out["looks"][-1]["futility"]["routes"]["F"]["needs"]
        sd = float(np.std(pattern(1000, 6.0, -6.0), ddof=1))
        self.assertAlmostEqual(need, NormalDist().inv_cdf(0.99) * sd / math.sqrt(self.N), places=9)  # z(0.99) x the whole block's SE


class PromoteFutilityTests(unittest.TestCase):
    """`promote(sequential=True)` on stub arms: the record, the ledger and the log line of a futility stop."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.agents = self.root / "agent"
        self.agents.mkdir()
        (self.root / "seeds").mkdir()
        self.seed_file = self.root / "seeds" / "Etest-9.json"
        for name in ("EXAM_DIR", "LOOP_ROOT"):
            patcher = mock.patch.object(fe, name, self.root / "no-gym")
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def run_promote(self, diffs: np.ndarray, cand: Path, **kw):
        seeds, cand_rows, base_rows = block(len(diffs), diffs, prefix="G")
        self.seed_file.write_text(json.dumps({"seeds": seeds}))
        arms = StubArms(cand_rows, base_rows)
        out = io.StringIO()
        with mock.patch.object(gen_loop, "run_arm", arms), contextlib.redirect_stdout(out):
            doc = gen_loop.promote(cand, None, self.seed_file, agent_root=self.agents, lab=self.root / "lab", boot=500, sequential=True,
                                   smoke=200, batch=200, **kw)
        return doc, arms, out.getvalue()

    def test_a_level_candidate_is_rejected_at_one_thousand_seeds_with_the_rule_in_the_decision_and_the_ledger(self):
        cand = stub_candidate(self.root, "c1", "sha-a", generation=4)
        doc, arms, printed = self.run_promote(pattern(2000, 6.0, -6.0), cand, keep_rule="v2")
        self.assertEqual((doc["decision_text"], doc["n_seeds"], doc["paired_seeds"]), ("rejected", 1000, 1000))
        self.assertEqual(arms.calls[-2:], [("cand", 1000, True), ("base", 1000, True)])  # nothing past the stop was played
        stop = doc["sequential"]["stop"]
        self.assertEqual((stop["action"], stop["n"], stop["rule"]), ("futility", 1000, "futility@1000"))
        self.assertIs(doc["decision"]["checks"]["sequential"], False)
        self.assertFalse(doc["decision"]["promote"])
        self.assertTrue(doc["decision"]["rule"].endswith("+D51-funnel+futility@1000"), doc["decision"]["rule"])
        self.assertEqual(doc["sequential"]["futility"], {"enabled": True, "min_seeds": 1000})
        self.assertIn("keep_v2", doc)  # the keep test v2 ran on the seeds played, as it does on a refusal
        self.assertIn("route", doc["keep_v2"])
        self.assertEqual(json.loads((Path(cand) / "promotion.json").read_text())["decision_text"], "rejected")
        self.assertFalse((self.agents / "current.json").exists())
        self.assertFalse((self.root / "lab" / "score" / "pending").exists())
        self.assertIn("futility look at 1000 of 2000 seeds", printed)  # the job log says which bounds fell short
        self.assertIn("falls short of", printed)
        self.assertIn("the judgement stops", printed)
        rows = gen_loop.read_promotions(self.agents / "promotions.tsv")
        self.assertEqual([(r["seed_block"], r["seeds"], r["decision"], r["attempt"]) for r in rows], [("Etest-9", "1000", "rejected", "1")])
        self.assertIn("funnel futility@1000 of 2000 seeds", rows[0]["guard"])
        self.assertEqual(rows[0]["rule"], "v2")  # the column keeps its meaning (`row_rule` reads it)

    def test_the_ledger_line_keeps_its_columns_in_number_and_order(self):
        cand = stub_candidate(self.root, "c1", "sha-a", generation=4)
        self.run_promote(pattern(2000, 6.0, -6.0), cand, keep_rule="v2")
        tsv = self.agents / "promotions.tsv"
        header = tsv.read_text().splitlines()[0].split("\t")
        self.assertEqual(tuple(header), PINNED_COLUMNS)
        self.assertEqual(tuple(gen_loop.TSV_COLUMNS), PINNED_COLUMNS)
        line = tsv.read_text().splitlines()[1].split("\t")
        self.assertEqual(len(line), len(PINNED_COLUMNS))
        self.assertEqual(len(gen_loop.read_promotions(tsv)[0]), len(PINNED_COLUMNS))

    def test_the_switch_off_plays_the_whole_block_as_before(self):
        cand = stub_candidate(self.root, "c1", "sha-a", generation=4)
        doc, arms, printed = self.run_promote(pattern(2000, 6.0, -6.0), cand, keep_rule="v2", futility=False)
        self.assertEqual((doc["sequential"]["stop"]["action"], doc["n_seeds"]), ("final", 2000))
        self.assertEqual(doc["sequential"]["futility"], {"enabled": False, "min_seeds": None})
        self.assertNotIn("futility look", printed)
        self.assertEqual(doc["decision"]["rule"], k2.RULE)

    def test_the_first_futility_seed_is_a_switch(self):
        cand = stub_candidate(self.root, "c1", "sha-a", generation=4)
        doc, arms, _ = self.run_promote(pattern(2000, 6.0, -6.0), cand, keep_rule="v2", futility_min_seeds=1400)
        self.assertEqual((doc["sequential"]["stop"]["rule"], doc["n_seeds"]), ("futility@1400", 1400))
        self.assertEqual(doc["sequential"]["futility"]["min_seeds"], 1400)

    def test_a_candidate_the_rule_promotes_is_promoted_and_the_futility_stop_is_not_in_its_record(self):
        cand = stub_candidate(self.root, "c1", "sha-a", generation=4)
        doc, arms, printed = self.run_promote(pattern(2000, 6.0, -5.0), cand, keep_rule="v2")  # +0.5 floors: route F at the last look or at 1,800
        self.assertEqual(doc["decision_text"], "promoted")
        self.assertNotEqual(doc["sequential"]["stop"]["action"], "futility")
        self.assertNotIn("futility@", doc["decision"]["rule"])
        self.assertTrue((self.agents / "current.json").exists())
        self.assertEqual(printed.count("futility look at"), 4)  # four looks asked (1,000 to 1,600) and each found a route alive
        self.assertNotIn("the judgement stops", printed)

    def test_rejudge_reads_a_futility_stop_as_a_rejection(self):
        cand = stub_candidate(self.root, "c1", "sha-a", generation=4)
        doc, _, _ = self.run_promote(pattern(2000, 6.0, -6.0), cand, keep_rule="v2")
        seeds = json.loads(self.seed_file.read_text())["seeds"]
        base_agent = gen_loop.load_agent(None, agent_root=self.agents)
        self.assertEqual(gen_loop.funnel_stop_record(doc)["action"], "futility")
        # the arms of a stopped block hold the seeds played: write them where `rejudge` reads them and judge again
        _, cand_rows, base_rows = block(2000, pattern(2000, 6.0, -6.0), prefix="G")
        arm = "Etest-9-a3-v1"
        for d, rows in ((Path(cand) / "arms" / arm, cand_rows[:1000]), (self.agents / "gen0" / "arms" / arm, base_rows[:1000])):
            d.mkdir(parents=True, exist_ok=True)
            (d / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        again = gen_loop.rejudge(cand, None, self.seed_file, agent_root=self.agents, note="again", keep_rule="v2", boot=500, out=self.root / "dry")
        self.assertEqual((again["decision_text"], again["n_seeds"]), ("rejected", 1000))
        self.assertIs(again["decision"]["checks"]["sequential"], False)
        self.assertTrue(again["decision"]["rule"].endswith("+futility@1000"))
        last = gen_loop.read_promotions(self.root / "dry" / "promotions.tsv")[-1]
        self.assertIn("funnel futility@1000 of 2000 seeds", last["guard"])
        del seeds, base_agent


class WiringTests(unittest.TestCase):
    def test_the_flags_are_on_train_and_promote_and_default_to_on(self):
        p = gen0.build_parser()
        a = p.parse_args(["promote", "--cand", "c", "--agent-root", "a", "--sequential"])
        self.assertEqual((a.no_futility, a.futility_min_seeds), (False, None))
        a = p.parse_args(["promote", "--cand", "c", "--agent-root", "a", "--sequential", "--no-futility", "--futility-min-seeds", "1500"])
        self.assertEqual((a.no_futility, a.futility_min_seeds), (True, 1500))
        t = p.parse_args(["train", "--marker-dir", "m", "--shards-root", "s", "--agent-root", "a", "--no-futility"])
        self.assertTrue(t.no_futility)

    def test_promote_cmd_hands_the_switches_to_promote(self):
        doc = {k: 0 for k in ("decision_text", "floor", "boss_kill", "wins")}
        for argv, want in ((["--sequential"], (None, None)), (["--sequential", "--no-futility", "--futility-min-seeds", "700"], (False, 700))):
            a = gen0.build_parser().parse_args(["promote", "--cand", "c", "--agent-root", "a", *argv])
            with mock.patch.object(gen_loop, "promote", return_value=doc) as pr, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(gen0.promote_cmd(a), 0)  # rc 0: the lab's marker is consumed as for any finished judgement
            self.assertEqual((pr.call_args[1]["futility"], pr.call_args[1]["futility_min_seeds"]), want)

    def test_the_subprocess_passes_a_switch_off_and_a_start_only_when_named(self):
        ran = []
        with mock.patch.object(gen_loop.subprocess, "run", lambda cmd, check=False: ran.append(cmd) or SimpleNamespace(returncode=0)):
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None, sequential=True)
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None, sequential=True, futility=False,
                                            futility_min_seeds=800)
        self.assertNotIn("--no-futility", ran[0])
        self.assertNotIn("--futility-min-seeds", ran[0])
        self.assertIn("--no-futility", ran[1])
        self.assertEqual(ran[1][ran[1].index("--futility-min-seeds") + 1], "800")

    def test_the_rounds_config_carries_the_switch(self):
        cfg = rd.load_config()
        self.assertEqual(cfg["futility"], {"enabled": True, "min_seeds": 1000})
        ctx = round1.Ctx(cfg, Path("D:/x/round"))
        a = gen0.build_parser().parse_args(round1.promote_cmd(ctx, "block", Path("c"), "s.json")[2:])
        self.assertEqual((a.no_futility, a.futility_min_seeds), (False, 1000))
        off = round1.Ctx(rd.load_config(futility={"enabled": False, "min_seeds": 1500}), Path("D:/x/round"))
        a = gen0.build_parser().parse_args(round1.promote_cmd(off, "block", Path("c"), "s.json")[2:])
        self.assertEqual((a.no_futility, a.futility_min_seeds), (True, 1500))
        with self.assertRaises(ValueError):
            rd.load_config(futility={"enabled": True, "min_seeds": 0})


class ModelTests(unittest.TestCase):
    """The rule on its normal model (known spreads, the looks of a 3,000-seed block): what the stop costs and what it saves."""

    LOOKS = np.array([200, 500, 800, 1100, 1400, 1700, 2000, 2300, 2600, 2900, 3000])
    SD_F, SD_D = 7.74, 41.0
    Z95, Z_R = NormalDist().inv_cdf(0.95), NormalDist().inv_cdf(1 - k2.ROUTE_ALPHA)

    def simulate(self, mf: float, md: float, min_seeds: int = 1000, paths: int = 40000, seed: int = 1) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        n, inc = self.LOOKS, np.diff(np.concatenate([[0], self.LOOKS]))

        def walk_(m, sd):
            return np.cumsum(rng.normal(m * inc, sd * np.sqrt(inc), size=(paths, len(n))), axis=1) / n

        mean_f, mean_d = walk_(mf, self.SD_F), walk_(md, self.SD_D)
        se_f, se_d = self.SD_F / np.sqrt(n), self.SD_D / np.sqrt(n)
        need_f = self.Z_R * se_f * np.sqrt(n / n[-1])
        need_d = np.maximum(k2.MIN_GAIN, self.Z_R * se_d * np.sqrt(n / n[-1]))
        alive = ((mean_f + self.Z95 * se_f > 0) & (mean_f + self.Z95 * se_f >= need_f)) | (
            (mean_d + self.Z95 * se_d > 0) & (mean_d + self.Z95 * se_d >= need_d))
        stop = ~alive & (n >= min_seeds) & (n < n[-1])
        stopped = stop.any(axis=1)
        would = (mean_f[:, -1] / se_f[-1] >= self.Z_R) | ((mean_d[:, -1] / se_d[-1] >= self.Z_R) & (mean_d[:, -1] >= k2.MIN_GAIN))
        at = np.where(stopped, stop.argmax(axis=1), len(n) - 1)
        return {"would": float(would.mean()), "lost": float((stopped & would).mean()), "stopped": float(stopped.mean()), "seeds": float(n[at].mean())}

    def test_a_level_candidate_is_stopped_early_most_of_the_time_and_costs_nothing(self):
        r = self.simulate(0.0, 0.0)
        self.assertGreater(r["stopped"], 0.65)  # three in four level candidates end before the last look ...
        self.assertLess(r["seeds"], 2300)  # ... saving about a third of the block's seeds on average
        self.assertLess(r["lost"], 0.0005)  # and a promotion lost to the stop is about one in ten thousand (the null promotes 0.01)

    def test_a_real_gain_is_promoted_as_often_as_without_the_stop(self):
        for mf, md in ((0.3, 0.0), (0.5, 0.0), (1.07, 0.0), (0.0, 2.0), (0.0, 3.0), (0.3, 3.0)):
            r = self.simulate(mf, md)
            self.assertLess(r["lost"], 0.004, (mf, md))  # at most four in a thousand blocks, at the worst effect (the promotion's own coin-flip)
            self.assertLess(r["lost"], 0.01 * max(r["would"], 0.01) + 0.0025, (mf, md))
        self.assertLess(self.simulate(1.07, 0.0)["stopped"], 1e-4)  # a gain the test finds for certain is never stopped
        self.assertLess(self.simulate(0.0, 5.0)["stopped"], 1e-4)

    def test_a_harmful_candidate_is_stopped_at_the_first_futility_look(self):
        r = self.simulate(-0.5, -2.0)
        self.assertGreater(r["stopped"], 0.99)
        self.assertLess(r["seeds"], 1300)


# ----------------------------------------------------------------- the lab's recorded blocks

REFEREE = Path(os.environ.get("SPIRELINE_REFEREE", "D:/spireline/referee"))
AGENT = REFEREE / "agent"


def recorded_arms(block_name: str, cand_gen: int, cand_dir: str, base_gen: int):
    """(seeds, candidate rows, incumbent rows) of a block the lab judged, or None where the lab's folders are absent."""
    seeds_f = next((p for p in (REFEREE / "LAB" / "seeds" / f"{block_name}.json", REFEREE / "seeds" / f"{block_name}.json") if p.is_file()), None)
    cand = AGENT / f"gen{cand_gen}" / cand_dir / "arms" / f"{block_name}-a3-v1" / "runs.jsonl"
    bases = [p for p in (AGENT / f"gen{base_gen}").glob(f"*/arms/{block_name}-a3-v1/runs.jsonl") if p.parent.parent.parent.name != cand_dir]
    if seeds_f is None or not cand.is_file() or not bases:
        return None
    rows = [[json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] for p in (cand, bases[0])]
    return json.loads(seeds_f.read_text())["seeds"], rows[0], rows[1]


class RecordedBlockTests(unittest.TestCase):
    """The lab's own E-blocks through the stop (nothing is played): no block that was promoted is ever stopped."""

    PROMOTED = (("E3-2026-10-23-r1", 3, "candidate-r1", 2, "v1"), ("E5-2026-10-25", 5, "candidate", 3, "v1"), ("E7-2026-10-26-r1", 7, "candidate-r1", 6, "v1"),
                ("E8-2026-10-29", 8, "candidate", 7, "v1"), ("E12-2026-11-05-r3", 12, "candidate-imit-l05", 11, "v2"))

    def replay(self, name, cg, cd, bg, rule, futility, frontier):
        got = recorded_arms(name, cg, cd, bg)
        if got is None:
            self.skipTest(f"the lab's recorded rows of {name} are not here")
        seeds, cand, base = got
        return walk(seeds, cand, base, keep_rule=rule, futility=futility, smoke=200, batch=300, frontier=frontier)[0]

    def test_a_block_the_rule_promoted_is_never_stopped_for_futility(self):
        ran = 0
        for name, cg, cd, bg, rule in self.PROMOTED:
            for futility in (1000, 500):
                if recorded_arms(name, cg, cd, bg) is None:
                    continue
                out = self.replay(name, cg, cd, bg, rule, futility, frontier=1)
                self.assertNotEqual(out["stop"]["action"], "futility", (name, futility))
                self.assertTrue(all(x["action"] != "futility" for x in out["looks"]), (name, futility))
                ran += 1
        if not ran:
            self.skipTest("none of the lab's promoted blocks is here")

    def test_the_block_the_owner_watched_is_stopped_at_the_first_look_the_rule_allows(self):
        out = self.replay("E14-2026-11-07", 14, "candidate-stakes-b8", 13, "v2", 1000, frontier=1)
        self.assertEqual((out["stop"]["action"], out["stop"]["n"]), ("futility", 1100))  # 2,000 seeds were played, and it was rejected


if __name__ == "__main__":
    unittest.main()
