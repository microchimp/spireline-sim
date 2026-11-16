"""The test funnel (D51): a 200-seed smoke block, then the paired block in looks (rl/gen_loop.py `seq_call`, `funnel`, `promote(sequential=True)`).

The rule and its error rates are in `gen_loop.seq_call`'s docstring; these tests re-run its arithmetic, play `promote` on stub arms
(no emulator), check the attempt counter and the record, and replay the lab's recorded E-blocks through the rule (those skip where the
lab's agent folder is absent: SPIRELINE_REFEREE names it, default D:/spireline/referee)."""

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
from test_fight_exam import human_gym_folder  # noqa: E402
import pool_fakes as pf  # noqa: E402
from tools.lab import pairing_check as pc  # noqa: E402
from tools.lab import keep_v2 as k2  # noqa: E402

LOOKS = [200, 500, 800, 1100, 1400, 1700, 2000]
SD = 7.74  # the per-seed SD of the paired floor difference on E11's block


def stub_candidate(root: Path, name: str, sha: str, generation: int = 5) -> Path:
    """A candidate directory the agent loader accepts (as test_gen_loop's): a policy sha, never loaded."""
    c = root / name
    c.mkdir(parents=True)
    (c / "model.json").write_text(json.dumps({"policy_sha256": sha}))
    (c / "combat_policy.pt").write_bytes(b"x")
    (c / "agent.json").write_text(json.dumps({"generation": generation, "macro": "rules"}))
    return c


def make_block(d_mean: float, n: int = 2000, seed: int = 11, sd: float = SD, prefix: str = "T") -> tuple[list[str], list[dict], list[dict]]:
    """Seeds and both arms' rows: the incumbent's floor about 17 (SD 8), the candidate's the same plus a paired difference of SD `sd`
    whose mean over the whole block is exactly `d_mean` (a few floors are nudged by one to make it so, clipping included)."""
    rng = np.random.default_rng(seed)
    seeds = [f"{prefix}{i:09d}" for i in range(n)]
    base = np.clip(np.rint(rng.normal(17, 8, n)), 0, 51)
    cand = np.clip(base + np.rint(rng.normal(d_mean, sd, n)), 0, 51)
    delta = int(round(d_mean * n)) - int((cand - base).sum())
    for i in rng.permutation(n):  # one floor at a time, until the block's total difference is the one asked for
        if delta == 0:
            break
        step = 1 if delta > 0 else -1
        if 0 <= cand[i] + step <= 51:
            cand[i] += step
            delta -= step

    def rows(floors: np.ndarray) -> list[dict]:
        return [{"seed": s, "floor": int(f), "win": False, "end": "death", "decisions": 10, "wall": 1.0, "crash": False,
                 "boss_damage": float(min(1.0, f / 18.0))} for s, f in zip(seeds, floors, strict=True)]

    return seeds, rows(cand), rows(base)


class LooksTests(unittest.TestCase):
    def test_the_looks_of_a_2000_seed_block(self):
        self.assertEqual(gen_loop.seq_looks(2000), LOOKS)
        self.assertEqual(gen_loop.seq_looks(2000, smoke=200, batch=300)[-1], 2000)

    def test_short_and_ragged_blocks(self):
        self.assertEqual(gen_loop.seq_looks(300), [200, 300])
        self.assertEqual(gen_loop.seq_looks(200), [200])
        self.assertEqual(gen_loop.seq_looks(50), [50])
        self.assertEqual(gen_loop.seq_looks(1900), [200, 500, 800, 1100, 1400, 1700, 1900])
        with self.assertRaises(ValueError):
            gen_loop.seq_looks(2000, smoke=1)
        with self.assertRaises(ValueError):
            gen_loop.seq_looks(2000, batch=0)

    def test_the_early_z_is_the_bonferroni_share_of_a_tenth_of_alpha(self):
        z = gen_loop.seq_early_z(7)
        self.assertAlmostEqual(z, NormalDist().inv_cdf(1 - 0.001 / 5), places=9)
        self.assertAlmostEqual(z, 3.54, places=2)
        self.assertEqual(gen_loop.seq_early_z(2), math.inf)  # no interim promote look
        self.assertEqual(gen_loop.seq_early_z(1), math.inf)
        self.assertLess(5 * (1 - NormalDist().cdf(z)), 0.0010001)  # the union bound: five looks spend at most 0.001


class RuleTests(unittest.TestCase):
    """`seq_call` against the rule as written: refuse on the one-sided 99% upper bound (never at the smoke look since D64), promote-eligible at the early z."""

    @staticmethod
    def diffs(mean: float, n: int, sd: float = SD) -> np.ndarray:
        x = np.random.default_rng(3).normal(0, 1, n)
        return (x - x.mean()) / x.std(ddof=1) * sd + mean  # sample mean and SD exactly as asked

    def call(self, mean: float, n: int, j: int, sd: float = SD) -> dict:
        return gen_loop.seq_call(self.diffs(mean, n, sd), j, LOOKS)

    def test_the_smoke_block_refuses_nothing_and_records_what_the_old_rule_said(self):
        c = self.call(-1.05, 200, 0)
        self.assertEqual(c["action"], "continue")  # D64: it refused below -1.0 before
        self.assertIn("smoke", c["reason"])
        self.assertIs(c["smoke_would_refuse"], True)
        c = self.call(-0.95, 200, 0)  # above the old smoke line, and its 99% upper bound is above 0
        self.assertEqual((c["action"], c["smoke_would_refuse"]), ("continue", False))
        c = self.call(-6.79, 200, 0)  # generation 12's first 200
        self.assertEqual((c["action"], c["smoke_would_refuse"]), ("continue", True))
        self.assertLess(c["upper99"], 0)  # the figures are in the call

    def test_the_old_smoke_rules_lower_bound_clause_follows_from_the_mean_clause(self):
        # whatever the SD, mean below -1 puts the 95% lower bound below 0, so the clause never changes what the old rule would have said
        for sd in (0.5, 7.7, 40.0):
            c = gen_loop.seq_call(self.diffs(-1.01, 200, sd), 0, LOOKS)
            self.assertEqual((c["action"], c["smoke_would_refuse"]), ("continue", True))
            self.assertLess(c["mean"] - 1.645 * c["se"], 0)

    def test_an_interim_look_refuses_on_the_99_percent_upper_bound(self):
        z = NormalDist().inv_cdf(0.99)
        for j in (1, 2, 3):
            se = SD / math.sqrt(LOOKS[j])
            self.assertEqual(self.call(-z * se * 1.02, LOOKS[j], j)["action"], "refuse")
            self.assertEqual(self.call(-z * se * 0.98, LOOKS[j], j)["action"], "continue")  # an upper bound just above 0: play on

    def test_the_smoke_block_never_promotes(self):
        self.assertEqual(self.call(5.0, 200, 0)["action"], "continue")  # z 9.1: the smoke look has no verdict to give

    def test_an_interim_look_is_eligible_at_the_early_z_and_not_below_it(self):
        z_e = gen_loop.seq_early_z(7)
        for j in (1, 2, 3, 4, 5):
            se = SD / math.sqrt(LOOKS[j])
            self.assertEqual(self.call(z_e * se * 1.01, LOOKS[j], j)["action"], "eligible", j)
            self.assertEqual(self.call(z_e * se * 0.99, LOOKS[j], j)["action"], "continue", j)

    def test_the_last_look_is_todays_decision(self):
        for mean in (-9.0, 0.0, 0.3, 9.0):
            self.assertEqual(self.call(mean, 2000, 6)["action"], "final")

    def test_no_spread_is_a_defined_call(self):
        self.assertEqual(gen_loop.seq_call([2.0] * 500, 1, LOOKS)["action"], "eligible")  # z infinite
        self.assertEqual(gen_loop.seq_call([-2.0] * 500, 1, LOOKS)["action"], "refuse")
        self.assertEqual(gen_loop.seq_call([0.0] * 500, 1, LOOKS)["action"], "continue")

    def test_the_call_matches_an_independent_reading_of_the_rule_on_random_paths(self):
        rng = np.random.default_rng(5)
        z99, z_e, z95 = NormalDist().inv_cdf(0.99), NormalDist().inv_cdf(1 - 0.0002), NormalDist().inv_cdf(0.95)
        seen = set()
        for d in (-3.0, -1.0, -0.3, 0.0, 0.5, 2.0):
            for _ in range(60):
                x = rng.normal(d, SD, 2000)
                for j, n in enumerate(LOOKS):
                    p = x[:n]
                    m, s = p.mean(), p.std(ddof=1) / math.sqrt(n)
                    if j == len(LOOKS) - 1:
                        want = "final"
                    elif j == 0:  # D64: the smoke look refuses nothing
                        want = "continue"
                    elif m + z99 * s < 0:
                        want = "refuse"
                    elif j >= 1 and m / s >= z_e:
                        want = "eligible"
                    else:
                        want = "continue"
                    self.assertEqual(gen_loop.seq_call(p, j, LOOKS)["action"], want)
                    seen.add(want)
        self.assertEqual(seen, {"final", "refuse", "eligible", "continue"})


class ErrorRateTests(unittest.TestCase):
    """The table in `seq_call`'s docstring, re-run on its normal model (Brownian sums at the looks, known SD), 300,000 paths per row."""

    N = 300_000

    def simulate(self, d: float) -> dict[str, float]:
        rng = np.random.default_rng(2026)
        n = np.array(LOOKS)
        inc = np.diff([0, *LOOKS])
        s = np.cumsum(rng.normal(d * inc, SD * np.sqrt(inc), size=(self.N, len(LOOKS))), axis=1)
        mean, se = s / n, SD / np.sqrt(n)
        z99, z_e = NormalDist().inv_cdf(0.99), gen_loop.seq_early_z(len(LOOKS))
        later = mean[:, 1:-1] + z99 * se[1:-1] < 0  # the refusal looks: every look but the smoke block (D64) and the last
        refuse_at = np.where(later.any(axis=1), later.argmax(axis=1) + 1, 99)
        early = mean[:, 1:-1] / se[1:-1] >= z_e
        early_at = np.where(early.any(axis=1), early.argmax(axis=1) + 1, 99)
        final_pass = mean[:, -1] / se[-1] >= z99
        promoted = (early_at < refuse_at) | ((refuse_at == 99) & (early_at == 99) & final_pass)
        stop = np.minimum(refuse_at, np.where(early_at < refuse_at, early_at, 99))
        stop = np.where(stop < 99, stop, len(LOOKS) - 1)
        return {"today": float(final_pass.mean()), "funnel": float(promoted.mean()), "early": float((early_at < 99).mean()),
                "refused_would_pass": float(((refuse_at < 99) & (early_at > refuse_at) & final_pass).mean()), "seeds": float(n[stop].mean())}

    def test_with_no_improvement_the_error_stays_within_a_tenth_of_alpha_of_todays(self):
        r = self.simulate(0.0)
        self.assertLess(abs(r["today"] - 0.01), 0.0007)  # the one-sided 0.01 of today's test
        self.assertLessEqual(r["early"], 0.0012)  # the Bonferroni budget of 0.001 (a union bound; simulated 0.0007)
        self.assertLessEqual(r["funnel"], r["today"] + 0.0010 + 0.0003)  # never more than the early budget above today's
        self.assertLess(r["funnel"], 0.0115)

    def test_a_real_improvement_is_promoted_as_often_as_by_the_last_look_alone(self):
        for d in (0.3, 0.5, 1.07):
            r = self.simulate(d)
            self.assertLess(abs(r["funnel"] - r["today"]), 0.006, d)  # 0.277 / 0.277, 0.713 / 0.713, 1.000 / 1.000 in the table
            self.assertLess(r["refused_would_pass"], 0.0015, d)  # an early refusal costs almost no power
        self.assertLess(self.simulate(1.07)["seeds"], 1000)  # generation 11's size of gain: about 860 seeds of 2,000

    def test_a_harmful_candidate_is_stopped_early(self):
        self.assertAlmostEqual(self.simulate(-5.9)["seeds"], 500, delta=1)  # D64: the smoke block plays on, the next look refuses (200 before)
        self.assertLess(self.simulate(-1.44)["seeds"], 640)
        self.assertEqual(self.simulate(-1.44)["funnel"], 0.0)


class StubArms:
    """`gen_loop.run_arm` replaced: serves the stub rows for the seeds asked and logs (who, seeds asked, resume)."""

    def __init__(self, cand_rows: list[dict], base_rows: list[dict]) -> None:
        self.cand_rows, self.base_rows, self.calls = cand_rows, base_rows, []

    def __call__(self, seeds, arm_dir, agent, config, procs, ascension=10, resume=False, **kw):
        who = "cand" if agent.prior_dir else "base"
        self.calls.append((who, len(seeds), resume))
        want = set(seeds)
        return [r for r in (self.cand_rows if who == "cand" else self.base_rows) if r["seed"] in want]


class PromoteFunnelTests(unittest.TestCase):
    """`promote(sequential=True)` on stub arms: where it stops, the record, the attempt counter."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.agents = self.root / "agent"
        self.agents.mkdir()
        self.seed_file = self.root / "seeds" / "Etest-1.json"
        self.seed_file.parent.mkdir()
        self.blocks: dict[tuple[float, str], tuple] = {}

    def tearDown(self):
        self.tmp.cleanup()

    def block(self, d_mean: float, name: str = "Etest-1"):
        """The block `name` (its own seeds, so blocks are disjoint) with a candidate `d_mean` floors ahead; its seed file is written."""
        if (d_mean, name) not in self.blocks:
            self.blocks[(d_mean, name)] = make_block(d_mean, prefix=name[-1])
        self.seed_file = self.root / "seeds" / f"{name}.json"
        self.seed_file.write_text(json.dumps({"seeds": self.blocks[(d_mean, name)][0]}))
        return self.blocks[(d_mean, name)]

    def run_promote(self, cand: Path, d_mean: float, *, sequential: bool = True, lab: Path | None = None, agents: Path | None = None,
                    review_guard: bool = True, strict: bool = False, name: str = "Etest-1"):
        seeds, cand_rows, base_rows = self.block(d_mean, name)
        arms = StubArms(cand_rows, base_rows)
        stub = arms
        if strict:  # the plain path's call has no `resume` parameter, as before
            def stub(seeds, arm_dir, agent, config, procs, ascension=10):  # noqa: ANN001
                return arms(seeds, arm_dir, agent, config, procs, ascension)
        with mock.patch.object(gen_loop, "run_arm", stub):
            doc = gen_loop.promote(cand, None, self.seed_file, agent_root=agents or self.agents, lab=lab, boot=500,
                                   review_guard=review_guard, sequential=sequential)
        return doc, arms

    def test_a_candidate_far_below_the_incumbent_plays_the_smoke_block_on_and_is_refused_at_the_second_look(self):
        cand = stub_candidate(self.root, "c1", "sha-a")
        lab = self.root / "lab"
        doc, arms = self.run_promote(cand, -4.0, lab=lab)
        self.assertEqual(arms.calls, [("cand", 200, True), ("base", 200, True), ("cand", 500, True), ("base", 500, True)])  # D64: the smoke block refuses nothing
        self.assertEqual((doc["decision_text"], doc["n_seeds"], doc["paired_seeds"]), ("rejected", 500, 500))
        seq = doc["sequential"]
        self.assertEqual((seq["block_seeds"], seq["played"], seq["stop"]["action"], seq["stop"]["n"]), (2000, 500, "refuse", 500))
        self.assertEqual((seq["looks"][0]["action"], seq["looks"][0]["smoke_would_refuse"]), ("continue", True))  # the smoke figures are in the record
        self.assertEqual(seq["schedule"], LOOKS)
        self.assertIs(doc["decision"]["checks"]["sequential"], False)
        self.assertTrue(doc["decision"]["rule"].endswith("+D51-funnel"))
        self.assertFalse((self.agents / "current.json").exists())
        self.assertFalse((lab / "score" / "pending").exists())
        self.assertEqual(json.loads((Path(cand) / "promotion.json").read_text())["decision_text"], "rejected")
        rows = gen_loop.read_promotions(self.agents / "promotions.tsv")
        self.assertEqual([(r["seed_block"], r["seeds"], r["decision"], r["cand_sha"], r["attempt"]) for r in rows],
                         [("Etest-1", "500", "rejected", "sha-a", "1")])
        self.assertIn("funnel refuse at 500 of 2000 seeds", rows[0]["guard"])
        self.assertEqual((self.agents / "promotions.tsv").read_text().splitlines()[0].split("\t"), list(gen_loop.TSV_COLUMNS))

    def test_a_refusal_spends_the_attempt_as_a_block_refusal_does(self):
        """The attempt of a generation is the number of distinct candidate shas it has in promotions.tsv (`attempts_before`): two spent, a third not played."""
        tsv = self.agents / "promotions.tsv"
        self.assertEqual(gen_loop.attempts_spent(tsv, 5, True), None)
        self.run_promote(stub_candidate(self.root, "c1", "sha-a"), -4.0)
        self.assertEqual(gen_loop.attempts_before(gen_loop.read_promotions(tsv), 5, "sha-b"), 1)
        self.assertEqual((gen_loop.rejections(tsv, 5), gen_loop.attempts_spent(tsv, 5, True)), (1, None))
        doc, _ = self.run_promote(stub_candidate(self.root, "c2", "sha-b"), -4.0, name="Etest-2")  # a fresh block, as `resolve_e_block` gives
        self.assertEqual(doc["decision_text"], "rejected")
        self.assertEqual([r["attempt"] for r in gen_loop.read_promotions(tsv)], ["1", "2"])
        self.assertEqual((gen_loop.rejections(tsv, 5), gen_loop.attempts_spent(tsv, 5, True)), (2, 2))  # the loop writes no third marker, `train` trains nothing
        arms = StubArms(*self.block(-4.0, "Etest-3")[1:])
        with mock.patch.object(gen_loop, "run_arm", arms), self.assertRaisesRegex(ValueError, "already had 2 promotion attempts"):
            gen_loop.promote(stub_candidate(self.root, "c3", "sha-c"), None, self.seed_file, agent_root=self.agents, boot=500,
                             review_guard=True, sequential=True)
        self.assertEqual(arms.calls, [])  # refused before any run was played

    def test_the_same_candidate_resumes_and_another_may_not_reuse_the_block(self):
        self.run_promote(stub_candidate(self.root, "c1", "sha-a"), -4.0)
        with self.assertRaisesRegex(ValueError, "already judged another candidate"):
            self.run_promote(stub_candidate(self.root, "c2", "sha-b"), -4.0)
        self.run_promote(self.root / "c1", -4.0)  # the same sha: a rerun of the same verdict
        self.assertEqual([r["attempt"] for r in gen_loop.read_promotions(self.agents / "promotions.tsv")], ["1", "1"])

    def test_a_large_improvement_is_promoted_early_with_every_check_holding(self):
        cand = stub_candidate(self.root, "c1", "sha-a")
        lab = self.root / "lab"
        doc, arms = self.run_promote(cand, 3.0, lab=lab)
        n = doc["n_seeds"]
        self.assertLess(n, 2000)
        self.assertGreaterEqual(n, 500)  # never at the smoke block
        self.assertEqual(arms.calls[-2:], [("cand", n, True), ("base", n, True)])
        self.assertEqual(doc["decision_text"], "promoted")
        self.assertEqual(doc["sequential"]["stop"]["action"], "promote")
        self.assertGreaterEqual(doc["sequential"]["looks"][-1]["z"], gen_loop.seq_early_z(7))
        checks = doc["decision"]["checks"]
        self.assertTrue(all(checks.values()), checks)
        self.assertIs(checks["sequential"], True)
        cur = json.loads((self.agents / "current.json").read_text())
        self.assertEqual((cur["generation"], Path(cur["dir"]).name), (5, "c1"))
        self.assertTrue((lab / "score" / "pending").is_file())
        rows = gen_loop.read_promotions(self.agents / "promotions.tsv")
        self.assertEqual((rows[0]["decision"], rows[0]["seeds"]), ("promoted", str(n)))
        self.assertIn(f"funnel promote at {n} of 2000 seeds", rows[0]["guard"])

    def test_a_close_candidate_plays_all_2000_and_its_record_is_todays(self):
        """A +0.3 candidate (the realised mean of this block is below the early z by far) plays every look; the figures and the decision
        are those of the same rows through the plain path, and the record keeps its format (one added key, `sequential`)."""
        cand = stub_candidate(self.root, "c1", "sha-a")
        doc, arms = self.run_promote(cand, 0.3)
        self.assertEqual([c[1] for c in arms.calls if c[0] == "cand"], LOOKS)
        self.assertEqual([c[1] for c in arms.calls if c[0] == "base"], LOOKS)
        self.assertEqual((doc["n_seeds"], doc["sequential"]["stop"]["action"], doc["sequential"]["stop"]["n"]), (2000, "final", 2000))
        self.assertEqual([x["action"] for x in doc["sequential"]["looks"]][:-1], ["continue"] * 6)
        self.assertNotIn("sequential", doc["decision"]["checks"])  # the last look is judged as it always was
        self.assertFalse(doc["decision"]["rule"].endswith("+D51-funnel"))
        plain_agents = self.root / "agent-plain"
        plain_agents.mkdir()
        plain, plain_arms = self.run_promote(stub_candidate(self.root, "p1", "sha-a"), 0.3, sequential=False, agents=plain_agents, strict=True)
        self.assertEqual([c[1] for c in plain_arms.calls], [2000, 2000])
        self.assertTrue(all(c[2] is False for c in plain_arms.calls))  # the plain path never asks for resume
        for key in ("floor", "boss_kill", "wins", "decision", "decision_text", "paired_seeds", "n_seeds", "base", "alpha", "review_guard"):
            self.assertEqual(doc[key], plain[key], key)
        self.assertEqual(doc["cand"]["summary"], plain["cand"]["summary"])
        self.assertEqual(set(doc) - set(plain), {"sequential"})
        self.assertEqual(set(plain) - set(doc), set())
        got = gen_loop.read_promotions(self.agents / "promotions.tsv")[0]
        want = gen_loop.read_promotions(plain_agents / "promotions.tsv")[0]
        for key in gen_loop.TSV_COLUMNS:
            if key not in ("written", "cand_dir"):
                self.assertEqual(got[key], want[key], key)

    def test_an_interim_look_that_clears_the_z_but_not_a_check_plays_on(self):
        """z clears at look 2, but the boss-kill guard does not hold on the seeds so far: no early promotion, the walk goes on."""
        seeds, cand_rows, base_rows = self.block(3.0)
        real = gen_loop.compare_arms

        def judged(c, b, alpha, boot, guard=None):
            res = real(c, b, alpha, boot, guard)
            if len(c) < 1400:
                res["decision"]["checks"]["boss_kill"] = False
                res["decision"]["promote"] = False
            return res

        out = gen_loop.funnel(seeds, lambda n: (cand_rows, base_rows), lambda c, b: judged(c, b, gen_loop.ALPHA, 500))
        self.assertEqual((out["stop"]["action"], out["stop"]["n"]), ("promote", 1400))
        self.assertEqual([x.get("held") for x in out["looks"][:3]], [None, ["boss_kill"], ["boss_kill"]])

    def test_a_flagged_drift_keeps_the_early_promotion_for_the_full_block(self):
        seeds, cand_rows, base_rows = self.block(3.0)

        def judged(c, b):
            res = gen_loop.compare_arms(c, b, gen_loop.ALPHA, 500)
            res["review_guard"] = {"review": len(c) < 2000}
            return res

        out = gen_loop.funnel(seeds, lambda n: (cand_rows, base_rows), judged)
        self.assertEqual((out["stop"]["action"], out["stop"]["n"]), ("final", 2000))
        self.assertIn("drift flagged", out["looks"][1]["held"])

    def test_both_arms_must_cover_every_seed_of_a_look(self):
        seeds, cand_rows, base_rows = self.block(0.3)
        with self.assertRaisesRegex(ValueError, "only 150 of 200 seeds are paired in both arms at look 1 of 7"):
            gen_loop.funnel(seeds, lambda n: (cand_rows[:150], base_rows), lambda c, b: {})

    # -- rejudge on a block the funnel stopped early: the arms hold only the seeds played

    def write_arms(self, cand: Path, name: str, cand_rows: list[dict], base_rows: list[dict], base: Path | None = None) -> None:
        """The arms as the funnel leaves them on disk: runs.jsonl of the candidate and of the base (generation 0 when none is given)."""
        base_arms = (base or self.agents / "gen0") / "arms" / f"{name}-a3-v1"
        for arm_dir, rows in ((cand / "arms" / f"{name}-a3-v1", cand_rows), (base_arms, base_rows)):
            arm_dir.mkdir(parents=True, exist_ok=True)
            (arm_dir / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def rejudge(self, cand: Path, base: Path | None = None, **kw):
        return gen_loop.rejudge(cand, base, self.seed_file, agent_root=self.agents, note="owner", boot=500, review_guard=True, **kw)

    def test_rejudge_reads_the_funnel_stop_from_the_record_and_only_from_a_stop_before_the_last_look(self):
        stop = {"action": "refuse", "n": 200}
        seq = {"stop": stop, "played": 200, "block_seeds": 2000}
        self.assertEqual(gen_loop.funnel_stop_record({"sequential": seq}), {"action": "refuse", "played": 200, "block_seeds": 2000, "sequential": seq})
        self.assertIsNone(gen_loop.funnel_stop_record(None))
        self.assertIsNone(gen_loop.funnel_stop_record({"n_seeds": 2000}))  # not played as the funnel
        self.assertIsNone(gen_loop.funnel_stop_record({"sequential": {**seq, "stop": {"action": "final", "n": 2000}, "played": 2000}}))
        self.assertIsNone(gen_loop.funnel_stop_record({"sequential": {**seq, "played": 2000}}))  # every seed played: the plain path
        self.assertIsNone(gen_loop.funnel_stop_record({"sequential": {"stop": stop}}))  # a record that does not say how far it played
        self.assertEqual(pc.funnel_stop({"sequential": seq}), {"action": "refuse", "played": 200, "block_seeds": 2000})  # pairing_check reads it alike

    def test_a_refused_block_is_rejudged_on_the_seeds_played_and_says_so(self):
        cand = stub_candidate(self.root, "c1", "sha-a")
        doc, _ = self.run_promote(cand, -4.0)
        self.assertEqual(doc["n_seeds"], 500)
        seeds, cand_rows, base_rows = self.block(-4.0)
        # the arms on disk: the 500 seeds played, and a few more in the candidate's folder that the rejudge must not take
        self.write_arms(cand, "Etest-1", cand_rows[:530], base_rows[:500])
        with mock.patch.object(gen0, "run_block", side_effect=AssertionError("must not play again")):
            out = self.rejudge(cand)
        self.assertEqual((out["decision_text"], out["n_seeds"], out["paired_seeds"]), ("rejected", 500, 500))
        self.assertEqual(out["rejudged"]["seeds"]["judged"], 500)
        self.assertIn("judged the 500 of 2000 seeds, where the funnel stopped on a refuse", out["rejudged"]["seeds"]["text"])
        self.assertIn("rejudged on the 500 of 2000 seeds", out["sequential"]["note"])
        self.assertEqual(out["sequential"]["stop"]["action"], "refuse")
        self.assertIs(out["decision"]["checks"]["sequential"], False)  # the block ended on a refusal and stays refused
        self.assertTrue(out["decision"]["rule"].endswith("+D51-funnel"))
        self.assertEqual((out["cand"]["summary"]["runs"], out["base"]["summary"]["runs"]), (500, 500))  # the 30 extra rows were left out
        self.assertTrue((cand / "promotion-rejected-" f"{doc['written'][:10]}.json").is_file())  # the earlier record kept beside it
        rows = gen_loop.read_promotions(self.agents / "promotions.tsv")
        self.assertEqual([(r["seed_block"], r["seeds"], r["decision"]) for r in rows], [("Etest-1", "500", "rejected")] * 2)
        self.assertIn("funnel refuse at 500 of 2000 seeds", rows[1]["guard"])
        self.assertFalse((self.agents / "current.json").exists())
        again = self.rejudge(cand)  # the record is carried over, so a second rejudge reads the stop again
        self.assertEqual((again["n_seeds"], again["rejudged"]["seeds"]["judged"]), (500, 500))
        day = doc["written"][:10]  # the promote's record and the first rejudge's both stay, the second under a name of its own
        self.assertTrue((cand / f"promotion-rejected-{day}.json").is_file() and (cand / f"promotion-rejected-{day}-r2.json").is_file())

    def test_an_early_promotion_is_rejudged_on_the_seeds_played_and_promotes_with_the_funnel_check_holding(self):
        cand = stub_candidate(self.root, "c1", "sha-a")
        lab = self.root / "lab"
        doc, _ = self.run_promote(cand, 3.0)
        n = doc["n_seeds"]
        self.assertLess(n, 2000)
        seeds, cand_rows, base_rows = self.block(3.0)
        base = stub_candidate(self.root, "base", "sha-base", generation=4)  # the promotion wrote current.json, so the incumbent is named
        self.write_arms(cand, "Etest-1", cand_rows[:n], base_rows[:n], base)
        out = self.rejudge(cand, base, lab=lab)
        self.assertEqual((out["decision_text"], out["n_seeds"], out["rejudged"]["seeds"]["judged"]), ("promoted", n, n))
        self.assertIs(out["decision"]["checks"]["sequential"], True)
        self.assertEqual(json.loads((self.agents / "current.json").read_text())["note"], "owner")
        self.assertTrue((lab / "score" / "pending").is_file())
        self.assertIn(f"funnel promote at {n} of 2000 seeds", gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]["guard"])

    def test_a_stopped_blocks_arms_must_still_hold_every_seed_played(self):
        cand = stub_candidate(self.root, "c1", "sha-a")
        self.run_promote(cand, -4.0)
        seeds, cand_rows, base_rows = self.block(-4.0)
        self.write_arms(cand, "Etest-1", cand_rows[:500], base_rows[:450])
        with self.assertRaisesRegex(ValueError, "only 450 of 500 seeds are paired in both cached arms"):
            self.rejudge(cand)
        self.assertTrue((cand / "promotion.json").is_file())  # nothing renamed or written by the refused rejudge
        self.assertEqual(len(gen_loop.read_promotions(self.agents / "promotions.tsv")), 1)

    def test_a_block_that_was_not_stopped_by_the_funnel_still_needs_every_seed(self):
        cand = stub_candidate(self.root, "c1", "sha-a")
        doc, _ = self.run_promote(cand, 0.3, sequential=False)
        self.assertEqual(doc["n_seeds"], 2000)
        seeds, cand_rows, base_rows = self.block(0.3)
        self.write_arms(cand, "Etest-1", cand_rows[:200], base_rows[:200])
        with self.assertRaisesRegex(ValueError, "only 200 of 2000 seeds are paired in both cached arms"):
            self.rejudge(cand)

    def test_the_seed_file_must_be_the_block_the_funnel_stopped(self):
        cand = stub_candidate(self.root, "c1", "sha-a")
        self.run_promote(cand, -4.0)
        seeds, cand_rows, base_rows = self.block(-4.0)
        self.write_arms(cand, "Etest-1", cand_rows[:500], base_rows[:500])
        self.seed_file.write_text(json.dumps({"seeds": seeds[:1500]}))  # not the 2,000-seed block the record names
        with self.assertRaisesRegex(ValueError, "stopped block Etest-1 at 500 of 2000 seeds, but the seed file is Etest-1 with 1500"):
            self.rejudge(cand)

    def test_the_command_says_which_seeds_were_judged(self):
        cand = stub_candidate(self.root, "c1", "sha-a")
        self.run_promote(cand, -4.0)
        seeds, cand_rows, base_rows = self.block(-4.0)
        self.write_arms(cand, "Etest-1", cand_rows[:500], base_rows[:500])
        args = SimpleNamespace(cand=str(cand), base=None, seeds_file=str(self.seed_file), agent_root=str(self.agents), config="a3-v1",
                               note="owner", lab=None, review_guard=True)
        buf = io.StringIO()
        with mock.patch.object(gen0, "_setup"), contextlib.redirect_stdout(buf):
            self.assertEqual(gen0.rejudge_cmd(args), 0)
        printed = json.loads(buf.getvalue())
        self.assertEqual(printed["n_seeds"], 500)
        self.assertIn("judged the 500 of 2000 seeds", printed["seeds_judged"])


class RunArmResumeTests(unittest.TestCase):
    """`run_arm(resume=True)`: an arm grows from a part of the block to the whole of it, playing only the seeds it has no row for."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.seeds = [f"S{i:09d}" for i in range(2000)]
        self.played: list[tuple[int, str]] = []
        self.calls: list[tuple[int, bool | None]] = []

    def tearDown(self):
        self.tmp.cleanup()

    def fake_run_block(self, seeds, sims, procs, out, *, ascension=10, agent=None, resume=None, **kw):
        out.mkdir(parents=True, exist_ok=True)
        self.calls.append((len(seeds), resume))
        f = out / "runs.jsonl"
        held = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if (resume and f.is_file()) else []
        have = {r["seed"] for r in held}
        new = [{"seed": s, "ix": ix, "floor": 5, "win": False} for ix, s in enumerate(seeds) if s not in have]
        self.played += [(r["ix"], r["seed"]) for r in new]
        f.write_text("".join(json.dumps(r) + "\n" for r in held + new))
        (out / "summary.json").write_text(json.dumps({"config": sims, "agent": agent.describe(), "wall_s": 10.0,
                                                      "timed_out": [{"seed": new[0]["seed"]}] if len(self.calls) in (1, 3) else []}))
        return {}

    def test_the_arm_grows_through_the_looks_and_each_seed_keeps_its_place_in_the_block(self):
        agent, arm = gen_loop.Agent(), self.root / "arm"
        with mock.patch.object(gen0, "run_block", self.fake_run_block):
            self.assertEqual(len(gen_loop.run_arm(self.seeds[:200], arm, agent, "a3-v1", 2, resume=True)), 200)
            self.assertEqual(self.calls, [(200, True)])
            self.assertEqual(len(gen_loop.run_arm(self.seeds[:200], arm, agent, "a3-v1", 2, resume=True)), 200)  # finished: not played again
            self.assertEqual(self.calls, [(200, True)])
            rows = gen_loop.run_arm(self.seeds[:500], arm, agent, "a3-v1", 2, resume=True)
            self.assertEqual((len(rows), self.calls[-1]), (500, (500, True)))
            self.assertEqual(len(self.played), 500)  # only the 300 new seeds were played
            self.assertEqual(len(gen_loop.run_arm(self.seeds[:300], arm, agent, "a3-v1", 2, resume=True)), 300)  # a part of what it holds
            self.assertEqual(len(self.calls), 2)
            gen_loop.run_arm(self.seeds[:800], arm, agent, "a3-v1", 2, resume=True)
        self.assertEqual(sorted(self.played), [(i, self.seeds[i]) for i in range(800)])  # ix is the seed's place in the block: the rng key of its run
        summary = json.loads((arm / "summary.json").read_text())
        self.assertEqual(summary["wall_s"], 30.0)  # three calls of 10 s
        self.assertEqual([x["seed"] for x in summary["timed_out"]], [self.seeds[0], self.seeds[500]])  # the first and the third call's, kept

    def test_a_plain_arm_is_played_and_reused_as_it_was(self):
        agent, arm = gen_loop.Agent(), self.root / "arm"
        with mock.patch.object(gen0, "run_block", self.fake_run_block):
            gen_loop.run_arm(self.seeds[:300], arm, agent, "a3-v1", 2)
            self.assertEqual(self.calls, [(300, None)])  # no resume argument is passed
            gen_loop.run_arm(self.seeds[:300], arm, agent, "a3-v1", 2)
            self.assertEqual(len(self.calls), 1)  # the same seeds: reused
            gen_loop.run_arm(self.seeds[:200], arm, agent, "a3-v1", 2)
            self.assertEqual(len(self.calls), 2)  # a different set: played again from nothing

    def test_another_agents_arm_is_not_reused(self):
        arm = self.root / "arm"
        with mock.patch.object(gen0, "run_block", self.fake_run_block):
            gen_loop.run_arm(self.seeds[:200], arm, gen_loop.Agent(generation=1), "a3-v1", 2, resume=True)
            gen_loop.run_arm(self.seeds[:200], arm, gen_loop.Agent(generation=2), "a3-v1", 2, resume=True)
        self.assertEqual(len(self.calls), 2)


class RealPoolResumeTests(unittest.TestCase):
    """The real `gen0.run_block` on the fake task, as `run_arm(resume=True)` calls it: a block that grows keeps its first rows and plays only
    the new seeds, each with its place in the whole block as `ix` (the rng key of a run), so the last look is the single block of today."""

    def test_a_growing_block_plays_each_seed_once_with_its_place_in_the_block(self):
        seeds = [f"S{i:03d}" for i in range(8)]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "arm"
            for part, token in ((seeds[:3], "first"), (seeds[:8], "second")):
                with mock.patch.dict(os.environ, {"POOL_FAKE_TOKEN": token}):
                    summ = gen0.run_block(part, 1, 2, out, agent=pf.FakeAgent(), task=pf.fake_task, initializer=None, resume=True,
                                          run_timeout_s=120.0)
            rows = [json.loads(x) for x in (out / "runs.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        by_seed = {r["seed"]: r for r in rows}
        self.assertEqual(sorted(by_seed), seeds)
        self.assertEqual(len(rows), 8)
        self.assertEqual([by_seed[s]["ix"] for s in seeds], list(range(8)))
        self.assertEqual([by_seed[s]["token"] for s in seeds], ["first"] * 3 + ["second"] * 5)  # the first three were not played again
        self.assertEqual((summ["resumed_rows"], summ["runs"]), (3, 8))


class WiringTests(unittest.TestCase):
    def test_the_flags_default_off_on_promote_and_train(self):
        p = gen0.build_parser()
        a = p.parse_args(["promote", "--cand", "c", "--agent-root", "a"])
        self.assertEqual((a.sequential, a.smoke_seeds, a.batch_seeds), (False, 200, 300))
        a = p.parse_args(["promote", "--cand", "c", "--agent-root", "a", "--sequential", "--smoke-seeds", "100", "--batch-seeds", "400"])
        self.assertEqual((a.sequential, a.smoke_seeds, a.batch_seeds), (True, 100, 400))
        t = ["train", "--marker-dir", "m", "--shards-root", "s", "--agent-root", "a"]
        self.assertFalse(p.parse_args(t).sequential)
        self.assertTrue(p.parse_args([*t, "--sequential"]).sequential)

    def test_the_subprocess_carries_the_flag_only_when_asked(self):
        ran = []
        with mock.patch.object(gen_loop.subprocess, "run", lambda cmd, check=False: ran.append(cmd) or SimpleNamespace(returncode=0)):
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None)
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None, sequential=True,
                                            prior_server_flags=("--prior-server", "auto"))
        self.assertNotIn("--sequential", ran[0])
        self.assertIn("--sequential", ran[1])
        self.assertEqual(ran[1][-2:], ["--prior-server", "auto"])

    def test_promote_cmd_hands_the_flag_to_promote(self):
        a = gen0.build_parser().parse_args(["promote", "--cand", "c", "--agent-root", "a", "--sequential"])
        doc = {k: 0 for k in ("decision_text", "floor", "boss_kill", "wins")}
        with mock.patch.object(gen_loop, "promote", return_value=doc) as pr:
            self.assertEqual(gen0.promote_cmd(a), 0)
        self.assertEqual((pr.call_args[1]["sequential"], pr.call_args[1]["smoke"], pr.call_args[1]["batch"]), (True, 200, 300))


# ----------------------------------------------------------------- the keep test v2's funnel (tools/lab/keep_v2.py `seq_call`)

SD_FLOOR_L05, SD_DEEP_L05, RHO_L05 = 8.77, 28.04, 0.458  # candidate-imit-l05's block: the SD of the paired floor and deep_B2 differences, their correlation


def v2_call(floor_mean: float, deep_mean: float, n: int, j: int, looks=LOOKS, sd_f: float = SD_FLOOR_L05, sd_d: float = SD_DEEP_L05) -> dict:
    """`keep_v2.seq_call` on differences whose sample mean and SD are exactly the ones given."""
    return k2.seq_call(fx.std_diffs(floor_mean, sd_f, n, 3), fx.std_diffs(deep_mean, sd_d, n, 4), j, looks)


class V2RuleTests(unittest.TestCase):
    def test_v2_smoke_refuses_nothing_and_records_the_floor_collapse_the_old_rule_refused(self):
        # D64: the smoke look gives no verdict on the floor; `smoke_would_refuse` is the rule of 2026-11-05 (a floor collapse deep_f does not offset)
        # 200 seeds: floor SE 0.62, deep_f SE 1.98
        def smoke(*a, **kw):
            c = v2_call(*a, **kw)
            self.assertEqual(c["action"], "continue")
            return c["smoke_would_refuse"]

        self.assertTrue(smoke(-1.5, -0.01, 200, 0))  # the floor mean is below -1.0 and deep_f is not above 0
        self.assertTrue(smoke(-1.5, -3.0, 200, 0))
        c = v2_call(-1.5, 3.0, 200, 0)  # the same floor with Act 2 kills up: the old rule played on too (the 95% upper bound -0.48 is above -1.0)
        self.assertFalse(c["smoke_would_refuse"])
        self.assertGreater(c["upper95"], -1.0)
        c = v2_call(-3.0, 5.0, 200, 0)  # a collapse no deep_f gain pays for: the floor's 95% upper bound is below -1.0, no route can pass
        self.assertTrue(c["smoke_would_refuse"])
        self.assertLess(c["upper95"], -1.0)
        self.assertFalse(smoke(-0.9, -0.01, 200, 0))  # above the old smoke line
        self.assertTrue(smoke(-1.01, -0.01, 200, 0))
        self.assertFalse(smoke(-1.01, 0.01, 200, 0))  # any sign above 0 kept it alive
        self.assertFalse(smoke(8.0, 30.0, 200, 0))  # the smoke look never promotes
        # v1's old smoke rule would have refused the floor below -1.0 with the 95% bound below 0; it too only records it now
        c = gen_loop.seq_call(fx.std_diffs(-1.5, SD_FLOOR_L05, 200, 3), 0, LOOKS)
        self.assertEqual((c["action"], c["smoke_would_refuse"]), ("continue", True))
        # the design's three 200-seed blocks: generation 12's second attempt, the pure imitation and generation 12
        self.assertTrue(smoke(-1.29, -1.0, 200, 0, sd_f=9.7, sd_d=20.0))
        self.assertTrue(smoke(-3.14, 2.5, 200, 0, sd_f=9.7, sd_d=29.0))
        self.assertTrue(smoke(-6.79, -1.0, 200, 0))

    def test_v2_later_look_refuses_only_when_both_routes_dead(self):
        # 800 seeds (look 3): floor SE 0.31, so its 99% upper bound is the mean + 0.72; deep_f SE 0.99, its 95% upper bound the mean + 1.63
        c = v2_call(-0.9, -2.0, 800, 2)
        self.assertEqual(c["action"], "refuse")  # route F dead (upper99 -0.18) and route A dead (deep_f's upper bound -0.37 is under +1.0)
        self.assertLess(c["upper99"], 0)
        self.assertLess(c["deep_upper95"], k2.MIN_GAIN)
        c = v2_call(-0.9, 1.0, 800, 2)
        self.assertEqual(c["action"], "continue")  # route F is dead but route A is alive: today's rule would have refused here
        self.assertLess(c["upper99"], 0)
        self.assertEqual(gen_loop.seq_call(fx.std_diffs(-0.9, SD_FLOOR_L05, 800, 3), 2, LOOKS)["action"], "refuse")
        c = v2_call(-2.0, 5.0, 800, 2)
        self.assertEqual(c["action"], "refuse")  # route A needs the floor guard (95% lower bound at least -1.0): its 99% upper bound is -1.28
        self.assertLess(c["upper99"], -k2.FLOOR_MARGIN)
        self.assertEqual(v2_call(-0.1, -5.0, 800, 2)["action"], "continue")  # route F alive (upper99 +0.62): deep_f is no reason to refuse
        self.assertEqual(v2_call(0.0, 0.0, 800, 2)["action"], "continue")
        # a refusal needs the bound below 0, not the mean: at -0.7 the upper bound is +0.02
        self.assertEqual(v2_call(-0.7, -5.0, 800, 2)["action"], "continue")

    def test_v2_early_z_splits_alpha_over_routes(self):
        z = k2.early_z(7)
        self.assertAlmostEqual(z, NormalDist().inv_cdf(1 - 0.001 / (2 * 5)), places=9)  # 0.001 over five interim looks and two routes
        self.assertAlmostEqual(z, 3.72, places=2)
        self.assertAlmostEqual(gen_loop.seq_early_z(7), 3.54, places=2)  # v1 spends the same alpha on one route
        self.assertGreater(z, gen_loop.seq_early_z(7))
        self.assertLess(2 * 5 * (1 - NormalDist().cdf(z)), 0.0010001)  # the union bound: ten (look, route) pairs spend at most 0.001
        self.assertEqual(k2.early_z(2), math.inf)
        self.assertEqual(k2.early_z(1), math.inf)
        self.assertAlmostEqual(k2.early_z(4), NormalDist().inv_cdf(1 - 0.001 / 4), places=9)
        # route F is eligible at the early z, with deep_f not worse; at 0.99 of it it is not
        n, j = 1100, 3
        se_f, se_d = SD_FLOOR_L05 / math.sqrt(n), SD_DEEP_L05 / math.sqrt(n)
        c = v2_call(z * se_f * 1.01, 0.0, n, j)
        self.assertEqual((c["action"], c["routes"]), ("eligible", ["F"]))
        self.assertEqual(v2_call(z * se_f * 0.99, 0.0, n, j)["action"], "continue")
        # route F also needs deep_f's 95% upper bound at or above 0: a clearly worse deep_f is no route F
        self.assertEqual(v2_call(z * se_f * 1.01, -3.0 * 1.645 * se_d, n, j)["action"], "continue")
        # route A at the early z needs its gain of +1.0 per 100 and the floor guard too
        c = v2_call(0.0, z * se_d * 1.01, n, j)
        self.assertEqual((c["action"], c["routes"]), ("eligible", ["A"]))
        self.assertEqual(v2_call(0.0, z * se_d * 0.99, n, j)["action"], "continue")
        self.assertEqual(v2_call(-1.3, z * se_d * 1.2, n, j)["action"], "continue")  # floor 95% lower bound -1.3 - 0.45 is under -1.0
        both = v2_call(z * se_f * 1.5, z * se_d * 1.5, n, j)
        self.assertEqual(both["routes"], ["F", "A"])
        # a z that clears 3.54 but not 3.72 is not enough for v2, though it is for v1
        c = v2_call(3.6 * se_f, 0.0, n, j)
        self.assertEqual(c["action"], "continue")
        self.assertEqual(gen_loop.seq_call(fx.std_diffs(3.6 * se_f, SD_FLOOR_L05, n, 3), j, LOOKS)["action"], "eligible")
        self.assertEqual(v2_call(8.0, 30.0, 2000, 6)["action"], "final")

    def test_seq_call_matches_an_independent_reading_of_the_rule_on_random_paths(self):
        rng = np.random.default_rng(8)
        z95, z99, z_e = NormalDist().inv_cdf(0.95), NormalDist().inv_cdf(0.99), k2.early_z(7)
        seen = set()
        for mf in (-3.0, -1.0, -0.4, 0.0, 0.6, 1.5):
            for md in (-3.0, 0.0, 2.0, 6.0):
                for _ in range(8):
                    f = rng.normal(mf, SD_FLOOR_L05, 2000)
                    d = rng.normal(md, SD_DEEP_L05, 2000)
                    for j, n in enumerate(LOOKS):
                        pf, pd_ = f[:n], d[:n]
                        m, s = pf.mean(), pf.std(ddof=1) / math.sqrt(n)
                        dm, ds = pd_.mean(), pd_.std(ddof=1) / math.sqrt(n)
                        f_dead, a_dead = m + z99 * s < 0, (m + z99 * s < -1.0) or (dm + z95 * ds < 1.0)
                        if j == len(LOOKS) - 1:
                            want = "final"
                        elif j == 0:  # D64: the smoke look refuses nothing on the floor
                            want = "continue"
                        elif f_dead and a_dead:
                            want = "refuse"
                        elif (m / s >= z_e and dm + z95 * ds >= 0) or (dm / ds >= z_e and dm >= 1.0 and m - z95 * s >= -1.0):
                            want = "eligible"
                        else:
                            want = "continue"
                        got = k2.seq_call(pf, pd_, j, LOOKS)["action"]
                        self.assertEqual(got, want, (mf, md, j))
                        seen.add(want)
        self.assertEqual(seen, {"final", "refuse", "eligible", "continue"})

    def simulate_null(self, paths: int = 20000, seed: int = 2026) -> dict[str, float]:
        """The rule on its normal model with no difference at all (both means 0), l05's moments: Brownian sums of a correlated pair at the
        looks, known SDs, the final look's decision the routes' own (z at 0.005, the gain, the floor guard, deep_f not worse)."""
        rng = np.random.default_rng(seed)
        n = np.array(LOOKS, dtype=float)
        inc = np.diff([0, *LOOKS]).astype(float)
        zf = rng.standard_normal((paths, len(LOOKS)))
        zd = RHO_L05 * zf + math.sqrt(1 - RHO_L05**2) * rng.standard_normal((paths, len(LOOKS)))
        sf = np.cumsum(SD_FLOOR_L05 * np.sqrt(inc) * zf, axis=1)
        sd = np.cumsum(SD_DEEP_L05 * np.sqrt(inc) * zd, axis=1)
        m, d = sf / n, sd / n
        se_f, se_d = SD_FLOOR_L05 / np.sqrt(n), SD_DEEP_L05 / np.sqrt(n)
        z95, z99, z_e, z_r = NormalDist().inv_cdf(0.95), NormalDist().inv_cdf(0.99), k2.early_z(len(LOOKS)), NormalDist().inv_cdf(1 - k2.ROUTE_ALPHA)
        alive = np.ones(paths, dtype=bool)
        promoted = np.zeros(paths, dtype=bool)
        early = np.zeros(paths, dtype=bool)
        stop = np.full(paths, len(LOOKS) - 1)
        for j in range(len(LOOKS)):
            f_dead = m[:, j] + z99 * se_f[j] < 0
            a_dead = (m[:, j] + z99 * se_f[j] < -k2.FLOOR_MARGIN) | (d[:, j] + z95 * se_d[j] < k2.MIN_GAIN)
            if j == len(LOOKS) - 1:
                route_f = (m[:, j] / se_f[j] >= z_r) & (d[:, j] + z95 * se_d[j] >= 0)
                route_a = (d[:, j] / se_d[j] >= z_r) & (d[:, j] >= k2.MIN_GAIN) & (m[:, j] - z95 * se_f[j] >= -k2.FLOOR_MARGIN)
                promoted |= alive & (route_f | route_a)
                break
            if j == 0:  # D64: no refusal at the smoke look
                refuse = np.zeros(paths, dtype=bool)
                eligible = np.zeros(paths, dtype=bool)
            else:
                refuse = f_dead & a_dead
                eligible = ((m[:, j] / se_f[j] >= z_e) & (d[:, j] + z95 * se_d[j] >= 0)) | (
                    (d[:, j] / se_d[j] >= z_e) & (d[:, j] >= k2.MIN_GAIN) & (m[:, j] - z95 * se_f[j] >= -k2.FLOOR_MARGIN))
            stop[alive & (refuse | eligible)] = j
            promoted |= alive & eligible
            early |= alive & eligible
            alive &= ~(refuse | eligible)
        return {"promote": float(promoted.mean()), "early": float(early.mean()), "seeds": float(n[stop].mean())}

    def test_v2_null_promotes_at_most_0p012(self):
        r = self.simulate_null()
        self.assertLessEqual(r["promote"], 0.012)  # 0.005 per route at the last look plus 0.001 early: the design's 0.011
        self.assertLessEqual(r["early"], 0.0012)  # the early budget, split over the interim looks and the two routes
        self.assertGreater(r["promote"], 0.005)  # and the model is not trivially inert
        # a real gain of the size the design wants (+3 per 100 at the Act 2 boss, floor -0.3) is promoted: the model's other end
        rng = np.random.default_rng(7)
        n = 2000
        f = rng.normal(-0.3, SD_FLOOR_L05, (500, n))
        d = RHO_L05 * (f + 0.3) / SD_FLOOR_L05 * SD_DEEP_L05 + math.sqrt(1 - RHO_L05**2) * rng.normal(0, SD_DEEP_L05, (500, n)) + 3.0
        final_a = (d.mean(axis=1) / (d.std(axis=1, ddof=1) / math.sqrt(n)) >= NormalDist().inv_cdf(1 - k2.ROUTE_ALPHA)) & (d.mean(axis=1) >= k2.MIN_GAIN) & (
            f.mean(axis=1) - 1.645 * f.std(axis=1, ddof=1) / math.sqrt(n) >= -k2.FLOOR_MARGIN)
        self.assertGreater(final_a.mean(), 0.9)  # route A at 2,000 seeds finds +3.0 per 100 nine times in ten or better


class PromoteV2Tests(unittest.TestCase):
    """`promote(sequential=True, keep_rule="v2")` on the synthetic arms of `candidate-imit-l05` (no emulator)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.agents = self.root / "agent"
        self.agents.mkdir()
        for name in ("EXAM_DIR", "LOOP_ROOT"):  # the judgement's human-bar line reads the gym's folder: not the real one in a test
            patcher = mock.patch.object(fe, name, self.root / "no-gym")
            patcher.start()
            self.addCleanup(patcher.stop)
        self.seeds, self.cand_rows, self.base_rows = fx.arms()
        (self.root / "seeds").mkdir()
        self.seed_file = self.root / "seeds" / "Etest-2.json"
        self.seed_file.write_text(json.dumps({"seeds": self.seeds}))
        # the incumbent's lineage: generation 3 promoted on Eprev with the Act 2 boss beaten 3 times in 100 (60 in 2,000)
        prev = [fx.run_row(f"P{i:09d}", 2 if i < 60 else 1 if i < 1000 else 0, np.random.default_rng(i)) for i in range(2000)]
        arm = self.agents / "gen3" / "candidate" / "arms" / "Eprev-a3-v1"
        arm.mkdir(parents=True)
        (arm / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prev), encoding="utf-8")
        gen_loop.append_promotion(self.agents / "promotions.tsv",
                                  {"cand_gen": 3, "base_gen": 2, "seed_block": "Eprev", "seeds": 2000, "boss_cand": 1000, "boss_base": 900,
                                   "decision": "promoted", "cand_sha": "sha-3", "cand_dir": "candidate", "attempt": 1, "drift": "clear"})

    def tearDown(self):
        self.tmp.cleanup()

    def run_promote(self, cand_rows, base_rows, cand: Path, **kw):
        arms = StubArms(cand_rows, base_rows)
        with mock.patch.object(gen_loop, "run_arm", arms):
            doc = gen_loop.promote(cand, None, self.seed_file, agent_root=self.agents, boot=500, sequential=True, keep_rule="v2", **kw)
        return doc, arms

    def test_the_judgement_carries_one_report_only_line_on_the_human_stratum(self):
        loop, exam, sha = human_gym_folder(self.root)
        cand = stub_candidate(self.root, "c1", sha, generation=4)  # its fight network is the bytes of the gym's player v2
        with mock.patch.object(fe, "EXAM_DIR", exam), mock.patch.object(fe, "LOOP_ROOT", loop):
            doc, _ = self.run_promote(self.cand_rows, self.base_rows, cand)
        line = doc["human_bar"]["line"]
        self.assertTrue(line.startswith(f"human bar: fight network {sha[:12]} (gym player v2): survival "), line)
        self.assertIn("on 39 of 39 starts, 156 games, engine e1, games file written ", line)
        self.assertEqual(json.loads((cand / "promotion.json").read_text())["human_bar"]["line"], line)
        self.assertEqual(doc["decision_text"], "promoted")  # the verdict is the keep test's alone
        self.assertTrue(all(doc["decision"]["checks"].values()), doc["decision"]["checks"])
        self.assertNotIn("human", json.dumps(doc["decision"]) + json.dumps(doc["keep_v2"]))
        row = gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]
        self.assertFalse(any("human" in k or "human" in v for k, v in row.items()))  # the ledger row is as it was

    def test_a_player_without_games_on_the_stratum_says_so_and_a_line_that_cannot_be_read_never_stops_the_verdict(self):
        loop, exam, sha = human_gym_folder(self.root, games=False)
        cand = stub_candidate(self.root, "c1", sha, generation=4)
        with mock.patch.object(fe, "EXAM_DIR", exam), mock.patch.object(fe, "LOOP_ROOT", loop):
            doc, _ = self.run_promote(self.cand_rows, self.base_rows, cand)
        self.assertEqual(doc["human_bar"]["line"], f"human bar: no human-stratum games for fight network {sha[:12]} (gym player v2)")
        self.assertEqual(doc["decision_text"], "promoted")

    def test_a_line_that_cannot_be_read_never_stops_the_verdict(self):
        cand = stub_candidate(self.root, "c1", "sha-a", generation=4)
        with mock.patch.object(fe, "lab_human_bar", side_effect=OSError("disk gone")):
            doc, _ = self.run_promote(self.cand_rows, self.base_rows, cand)
        self.assertEqual(doc["human_bar"]["line"], "human bar: not read (OSError)")
        self.assertEqual(doc["decision_text"], "promoted")
        self.assertTrue(all(doc["decision"]["checks"].values()), doc["decision"]["checks"])

    def test_the_line_names_the_incumbents_network_too_when_it_is_another_and_says_when_there_is_none(self):
        loop, exam, sha = human_gym_folder(self.root)
        cand = gen_loop.load_agent(stub_candidate(self.root, "c1", sha, generation=4))
        base = gen_loop.load_agent(stub_candidate(self.root, "c2", "ee" * 32, generation=3))
        with mock.patch.object(fe, "EXAM_DIR", exam), mock.patch.object(fe, "LOOP_ROOT", loop):
            same = gen_loop.human_bar_note(cand, cand)["line"]
            both = gen_loop.human_bar_note(cand, base)["line"]
            none = gen_loop.human_bar_note(gen_loop.Agent(), base)["line"]
            self.assertEqual(same, gen_loop.human_bar_note(cand)["line"])  # one network: one clause
        self.assertEqual(both.count("human bar:"), 1)
        self.assertIn(" | base fight network eeeeeeeeeeee (no gym player holds these bytes); the gym incumbent v2's reading, not this network's: ", both)
        self.assertTrue(none.startswith("human bar: the judged player has no fight network (no combat prior)"))

    def test_a_route_a_gain_is_promoted_early_with_the_record_and_the_ledger_row(self):
        cand = stub_candidate(self.root, "c1", "sha-a", generation=4)
        doc, arms = self.run_promote(self.cand_rows, self.base_rows, cand)
        n = doc["n_seeds"]
        self.assertEqual(doc["decision_text"], "promoted")
        self.assertTrue(500 <= n < 2000, n)  # the strict early z, and never at the smoke block
        self.assertEqual(arms.calls[-2:], [("cand", n, True), ("base", n, True)])
        self.assertEqual(doc["sequential"]["stop"]["action"], "promote")
        self.assertGreaterEqual(doc["sequential"]["looks"][-1]["deep_z"], k2.early_z(7))
        self.assertAlmostEqual(doc["sequential"]["z_early"], 3.72, places=2)
        self.assertEqual((doc["sequential"]["keep_rule"], doc["sequential"]["frontier"]), ("v2", "B2"))  # fixed from the incumbent's own arms
        self.assertEqual(doc["keep_v2"]["frontier"]["pooled_kills"][1], 60)
        self.assertEqual(doc["keep_v2"]["route"], "A")
        self.assertEqual(doc["decision"]["rule"], k2.RULE + "+D51-funnel")
        self.assertTrue(all(doc["decision"]["checks"].values()), doc["decision"]["checks"])
        self.assertEqual(doc["keep_rule"], "v2")
        cur = json.loads((self.agents / "current.json").read_text())
        self.assertEqual((cur["generation"], Path(cur["dir"]).name), (4, "c1"))
        rows = gen_loop.read_promotions(self.agents / "promotions.tsv")
        new = rows[-1]
        self.assertEqual((new["decision"], new["seeds"], new["rule"], new["frontier"], new["route"], new["attempt"]), ("promoted", str(n), "v2", "B2", "A", "1"))
        self.assertEqual((gen_loop.row_rule(new), gen_loop.row_rule(rows[0])), ("v2", "v1"))
        self.assertGreater(int(new["k2_cand"]), int(new["k2_base"]))
        self.assertGreater(float(new["deep_lower"]), 0)
        self.assertIn(f"funnel promote at {n} of 2000 seeds", new["guard"])
        self.assertIn("v2 route A", new["guard"])
        self.assertEqual(rows[0]["rule"], "")  # the incumbent's older row has no rule: it reads as v1
        # the Act 1 kill guard fires on these arms and blocks nothing
        self.assertIn("kill", doc["keep_v2"]["act1_review"]["fires"])

    def test_a_floor_collapse_plays_the_smoke_look_on_and_is_refused_at_the_second_look(self):
        collapse = ((600, 0, 1), (1400, 1, 1))  # 600 of 2,000 runs lose the Act 1 boss they used to beat
        seeds, cand, base = fx.arms(collapse, prefix="C")
        self.seed_file.write_text(json.dumps({"seeds": seeds}))
        cand_dir = stub_candidate(self.root, "c1", "sha-a", generation=4)
        doc, arms = self.run_promote(cand, base, cand_dir)
        self.assertEqual(arms.calls, [("cand", 200, True), ("base", 200, True), ("cand", 500, True), ("base", 500, True)])
        self.assertEqual((doc["decision_text"], doc["n_seeds"], doc["sequential"]["stop"]["action"]), ("rejected", 500, "refuse"))
        self.assertEqual((doc["sequential"]["looks"][0]["action"], doc["sequential"]["looks"][0]["smoke_would_refuse"]), ("continue", True))
        self.assertIs(doc["decision"]["checks"]["sequential"], False)
        self.assertFalse((self.agents / "current.json").exists())

    def test_the_eligible_route_must_hold_in_the_judgement(self):
        seeds, cand, base = self.seeds, self.cand_rows, self.base_rows
        real = gen_loop.compare_arms

        def judged(c, b, alpha, boot, guard=None, keep=None):
            res = real(c, b, alpha, boot, guard, keep=keep)
            if len(c) < 1400:  # the judgement on the seeds played finds no route (as if the route that cleared the z did not hold)
                res["keep_v2"] = {**res["keep_v2"], "route": ""}
            return res

        out = gen_loop.funnel(seeds, lambda n: (cand, base), lambda c, b: judged(c, b, gen_loop.ALPHA, 500, keep=gen_loop.keep_v2_verdict(
            c, b, [], cand_gen=4, cand_sha="x", block="Etest-2", frontier_info={"index": 1, "name": "B2"}, boot=500)),
            keep_rule="v2", frontier=1)
        self.assertEqual((out["stop"]["action"], out["stop"]["n"]), ("promote", 1400))
        self.assertEqual([x["action"] for x in out["looks"][:3]], ["continue"] * 3)
        self.assertEqual(out["looks"][3]["routes"], ["A"])  # route A cleared the early z at 1,100 seeds ...
        self.assertEqual(out["looks"][3]["held"], ["route A"])  # ... but did not hold in the judgement, so the walk went on
        self.assertNotIn("held", out["looks"][4])



# ----------------------------------------------------------------- the lab's recorded blocks

REFEREE = Path(os.environ.get("SPIRELINE_REFEREE", "D:/spireline/referee"))
AGENT = REFEREE / "agent"


def recorded(block: str, cand_dir: str, base_dir: str, cand_gen: int):
    """(seeds, candidate rows, incumbent rows, the promotions.tsv rows written before this block) of a block the lab judged, or None."""
    seeds_f = REFEREE / "LAB" / "seeds" / f"{block}.json"
    arms = [AGENT / d / "arms" / f"{block}-a3-v1" / "runs.jsonl" for d in (cand_dir, base_dir)]
    tsv = AGENT / "promotions.tsv"
    if not (seeds_f.is_file() and tsv.is_file() and all(p.is_file() for p in arms)):
        return None
    rows = [[json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] for p in arms]
    promos = gen_loop.read_promotions(tsv)
    mine = [r for r in promos if r["seed_block"] == block][-1]
    return (json.loads(seeds_f.read_text())["seeds"], rows[0], rows[1], [r for r in promos if r["written"] < mine["written"]], mine)


def replay(block: str, cand_dir: str, base_dir: str, cand_gen: int):
    got = recorded(block, cand_dir, base_dir, cand_gen)
    if got is None:
        raise unittest.SkipTest(f"the lab's recorded rows of {block} are not here")
    seeds, cand_rows, base_rows, history, mine = got

    def judge(c, b):
        g = gen_loop.review_guard_verdict(c, b, history, cand_gen=cand_gen, cand_sha=mine["cand_sha"], block=block, boot=4000)
        return gen_loop.compare_arms(c, b, gen_loop.ALPHA, 4000, g)

    return gen_loop.funnel(seeds, lambda n: (cand_rows, base_rows), judge), judge, mine


class RecordedBlockTests(unittest.TestCase):
    """The lab's own E-blocks, read from their per-seed rows (nothing is played)."""

    def test_generation_12s_refused_block_plays_the_smoke_block_on_and_stops_at_the_second_look(self):
        out, _, mine = replay("E12-2026-11-02", "gen12/candidate", "gen11/candidate", 12)
        self.assertEqual(mine["decision"], "rejected")  # -5.91 floors, both guards fired
        self.assertEqual(out["stop"]["action"], "refuse")  # D64: the smoke block refused it at 200 before; now the look at 500 does
        self.assertIn(out["stop"]["n"], (500, 800))  # a quarter of the block's time
        self.assertEqual((out["looks"][0]["action"], out["looks"][0]["smoke_would_refuse"]), ("continue", True))
        self.assertLess(out["looks"][0]["mean"], -1.0)
        self.assertLess(out["looks"][0]["upper99"], 0)

    def test_generation_11s_block_ends_only_where_the_rule_allows(self):
        out, judge, mine = replay("E11-2026-11-02", "gen11/candidate", "gen9/candidate", 11)
        self.assertEqual(mine["decision"], "promoted")  # +1.069 floors, lower bound +0.666
        stop = out["stop"]
        self.assertIn(stop["action"], ("final", "promote"))  # a refusal of the block that promoted would be a wrong stop
        if stop["action"] == "promote":
            self.assertGreaterEqual(stop["n"], 500)
            self.assertGreaterEqual(out["looks"][-1]["z"], gen_loop.seq_early_z(7))
            res = judge(out["cand_rows"], out["base_rows"])
            self.assertTrue(res["decision"]["promote"])  # every check holds on the seeds played
            self.assertFalse(res["review_guard"]["review"])
        else:
            self.assertEqual(stop["n"], 2000)
        self.assertTrue(all(x["action"] != "refuse" for x in out["looks"]))

    def test_generation_10s_close_second_block_plays_all_2000(self):
        out, _, mine = replay("E10-2026-10-30-r1", "gen10/candidate-r1", "gen9/candidate", 10)
        self.assertEqual(mine["decision"], "rejected")  # +0.23 floors, lower bound -0.24
        self.assertEqual((out["stop"]["action"], out["stop"]["n"]), ("final", 2000))
        self.assertEqual([x["action"] for x in out["looks"]], ["continue"] * 6 + ["final"])

    def test_generation_10s_first_block_plays_the_smoke_block_on_and_is_refused_at_a_later_look(self):
        out, _, mine = replay("E10-2026-10-29", "gen10/candidate", "gen9/candidate", 10)
        self.assertEqual(mine["decision"], "rejected")  # -1.44 floors, both guards fired
        self.assertEqual(out["stop"]["action"], "refuse")  # D64: refused at 200 before; the smoke look only records it now
        self.assertIn(out["stop"]["n"], (500, 800, 1100))
        self.assertEqual((out["looks"][0]["action"], out["looks"][0]["smoke_would_refuse"]), ("continue", True))


if __name__ == "__main__":
    unittest.main()
