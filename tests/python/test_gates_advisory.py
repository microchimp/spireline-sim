"""D64 (2026-11-16): five hard-coded gates became recorded diagnostics. A V_run fit is kept when its export exists, whatever its D33 Brier
skill, its heal and rest rates and its human agreement say (they are computed and recorded); the funnel's smoke block (200 pairs) refuses
nothing on the paired floor difference (a crashing candidate is still refused: an engineering fault, not a learning verdict). The judge's
verdict and the later looks' stops are as they were."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, ROOT / "tests" / "python"):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import gen0  # noqa: E402
import gen_loop  # noqa: E402
import keep_v2_fixtures as fx  # noqa: E402
import test_vrun_swap as tvs  # noqa: E402
from test_vrun_matched import shard_block  # noqa: E402
from tools.lab import keep_v2 as k2  # noqa: E402

LOOKS = [200, 500, 800, 1100, 1400, 1700, 2000]
SD = 7.74


def d33_straddling_zero_keep() -> dict:
    """A D33 record whose Brier skill interval straddles zero: `all_computable_pass` is false."""
    head = {"positives": 40, "positive_runs": 40, "rows": 900, "pass": False,
            "brier_skill_vs_base_rate": {"value": 0.005, "ci": [-0.03, 0.04], "runs": 900, "pass": False},
            "brier_skill_vs_v0": {"value": 0.0, "ci": [-0.02, 0.02], "pass": False},
            "log_loss_vs_base_rate": {"pass": True}, "ece_bins_ge_30_positives": {"pass": True}}
    return {"status": "ok", "all_computable_pass": False, "heads": {"act1_boss_kill": head}}


def heal_down_20_points(candidate_npz, incumbent_npz, runs_root, out_dir, **kw) -> dict:
    """A ranking block: the heal rate fell 20 points (the old limit was 10), the rest rate and the human agreement are fine."""
    checks = [{"name": "rest_rate", "candidate": 0.8, "incumbent": 0.8, "diff": 0.0, "limit": -0.05, "pass": True},
              {"name": "heal_rate", "candidate": 0.30, "incumbent": 0.50, "diff": -0.20, "limit": -0.10, "pass": False},
              {"name": "human_agreement_rest", "candidate": 0.7, "incumbent": 0.7, "diff": 0.0, "limit": -0.01, "pass": True}]
    return {"status": "ok", "pass": False, "checks": checks, "set": {"digest": "d" * 8}}


class KeepIsTheExportTests(unittest.TestCase):
    """`vrun_keep_passed` reads nothing but the export; the checks stay in the record, marked advisory."""

    def test_the_keep_rule_is_the_export_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as d:
            cand = Path(d)
            failing = {"all_computable_pass": False, "ranking": {"status": "ok", "pass": False}}
            self.assertFalse(gen_loop.vrun_keep_passed(failing, cand))  # no export: nothing to keep
            (cand / "run_value.npz").write_bytes(b"x")
            self.assertTrue(gen_loop.vrun_keep_passed(failing, cand))
            self.assertTrue(gen_loop.vrun_keep_passed({"status": "not_run", "all_computable_pass": False}, cand))
            self.assertTrue(gen_loop.vrun_keep_passed(None, cand))
            self.assertTrue(gen_loop.vrun_keep_passed({"all_computable_pass": True, "ranking": {"pass": True}}, cand))
            self.assertFalse(gen_loop.ranking_ok(failing))  # the old limit is still computed from the block ...
            self.assertEqual(gen_loop.macro_for(failing, cand, enabled=True), "vrun")  # ... and does not decide the macro
            self.assertEqual(gen_loop.macro_for(failing, cand, enabled=False), "rules")  # the opt-out switch is not a check
            (cand / "run_value.npz").unlink()
            self.assertEqual(gen_loop.macro_for({"all_computable_pass": True}, cand, enabled=True), "rules")


class AdvisoryBuildTests(unittest.TestCase):
    """A fit whose heal rate fell by 20 points and whose D33 skill interval straddles zero: kept, both failures recorded, and it builds."""

    setUp = tvs.VrunSwapTests.setUp
    build = tvs.VrunSwapTests.build
    record = tvs.VrunSwapTests.record

    def make_fit(self, *, human_leg: bool = True) -> tuple[Path, dict]:
        prior = gen_loop.load_agent(self.inc).describe()["prior_sha256"]
        runs = Path(self.tmp.name) / "store" / "jobs"
        shard_block(runs, "j6", 6, prior, ["A000000001", "A000000002"])
        incumbent = Path(self.tmp.name) / "inc-head" / "run_value.npz"
        incumbent.parent.mkdir()
        incumbent.write_bytes(b"incumbent head")
        out = Path(self.tmp.name) / "plain-fit"

        def fake_train(paths, out_dir, **kw):
            Path(out_dir, "run_value.npz").write_bytes(b"the plain-run fit")
            Path(out_dir, "run_value.pt").write_bytes(b"the plain-run fit, torch")
            return {"wall_min": 0.1, "value": {"runs": 4, "keep": d33_straddling_zero_keep()}}

        with mock.patch.object(gen_loop, "train_run_value", side_effect=fake_train):
            doc = gen_loop.fit_run_value_matched(runs, prior, out, beta=0.3, seed_dir=Path(self.tmp.name) / "no-seeds", incumbent=incumbent,
                                                 ranking_check=heal_down_20_points, human_leg=human_leg)
        return out, doc

    def test_kept_with_both_failures_recorded_and_it_builds(self):
        out, doc = self.make_fit()
        self.assertIs(doc["kept"], True)
        keep = doc["keep"]
        self.assertIs(keep["all_computable_pass"], False)  # "predicts at all": computed, recorded, not gating
        self.assertIs(keep["ranking"]["pass"], False)
        heal = [c for c in keep["ranking"]["checks"] if c["name"] == "heal_rate"][0]
        self.assertEqual((heal["pass"], heal["diff"], heal["limit"]), (False, -0.20, -0.10))  # each check's pass is written as before
        self.assertEqual((keep["advisory"], keep["gated_on"]), (True, ["export"]))
        self.assertIs(keep["old_rule_passed"], False)  # what the D33 and ranking rule would have said
        saved = json.loads((out / gen_loop.VRUN_FIT_JSON).read_text())
        self.assertEqual((saved["kept"], saved["keep"]["advisory"], saved["keep"]["gated_on"]), (True, True, ["export"]))
        self.assertEqual(saved, gen_loop.read_vrun_fit(out))
        self.fit = out
        cand = self.build()
        meta = json.loads((cand / "agent.json").read_text())
        self.assertEqual(gen_loop.file_sha256(cand / "run_value.npz"), meta["vrun_fit"]["sha256"])
        self.assertEqual([c["pass"] for c in meta["vrun_fit"]["ranking"]["checks"]], [True, False, True])  # the failure travels with the candidate
        self.assertEqual("run", meta["vrun_fit"]["human_leg"])

    def test_a_record_written_before_d64_that_says_kept_false_is_read_and_built_by_its_export(self):
        # the own-runs fit of round 3: heal rate refused it at fit time, the record says kept false, the export is on disk
        self.record(kept=False, d33=False, ranking=tvs.ranking_doc(heal=False, human={"name": "human_agreement_rest", "pass": False, "diff": -0.5}))
        doc = gen_loop.read_vrun_fit(self.fit)
        self.assertIs(doc["kept"], False)  # the record is read as written ...
        self.assertTrue(self.build().is_dir())  # ... and the export decides
        (self.fit / "run_value.npz").unlink()
        with self.assertRaises(ValueError):  # without the export there is nothing to carry
            gen_loop.read_vrun_fit(self.fit)

    def test_the_human_leg_flag_is_accepted_and_changes_nothing(self):
        out, doc = self.make_fit(human_leg=False)
        self.assertIs(doc["kept"], True)
        self.assertIs(doc["human_leg"], False)
        self.fit = out
        a = json.loads((self.build() / "agent.json").read_text())
        self.assertEqual("not run", a["vrun_fit"]["human_leg"][:7])  # the record still says the leg was left out
        self.assertEqual(doc, gen_loop.read_vrun_fit(out, without_human_leg=True))
        self.assertEqual(doc, gen_loop.read_vrun_fit(out, without_human_leg=False))

    def test_the_command_builds_without_the_flag_and_accepts_it(self):
        out, _ = self.make_fit()

        def argv(tag, *more):
            return ["build-vrun", "--base", str(self.inc), "--vrun-fit", str(out), "--agent-root", str(self.agents), "--candidate-generation", "7",
                    "--tag", tag, "--why", "test", "--no-export", "--marker-dir", str(self.lab / "train"), *more]

        with mock.patch("sys.stdout"):
            self.assertEqual(0, gen0.main(argv("cheap-v")))
            self.assertEqual(0, gen0.main(argv("cheap-w", "--without-human-leg")))
        self.assertTrue((self.lab / "train" / "ready-6-g7-cheap-v").is_file())
        self.assertTrue((self.lab / "train" / "ready-6-g7-cheap-w").is_file())


class CarryTests(unittest.TestCase):
    """`carry_vrun_fit`: a new fit that exported replaces the seed fit; the seed is carried only when the new fit made no export."""

    def test_a_new_fit_that_exported_is_carried_whatever_its_checks_said(self):
        with tempfile.TemporaryDirectory() as d:
            cand, seed = Path(d) / "cand", Path(d) / "seed.npz"
            cand.mkdir()
            (cand / "run_value.npz").write_bytes(b"NEWFIT")
            seed.write_bytes(b"SEEDFIT")
            keep = {"all_computable_pass": False, "ranking": {"status": "ok", "pass": False}}
            got = gen_loop.carry_vrun_fit(cand, keep, seed)
            self.assertEqual((got["source"], got["kept"], got["new_fit_passed_keep"]), ("new", True, True))
            self.assertEqual((cand / "run_value.npz").read_bytes(), b"NEWFIT")
            self.assertFalse((cand / "rejected-fit").exists())
            (cand / "run_value.npz").unlink()  # the fit made no export: the seed fit takes its place
            got = gen_loop.carry_vrun_fit(cand, keep, seed)
            self.assertEqual((got["source"], got["kept"]), ("seed", True))
            self.assertEqual((cand / "run_value.npz").read_bytes(), b"SEEDFIT")
            (cand / "run_value.npz").unlink()
            self.assertEqual((gen_loop.carry_vrun_fit(cand, keep, None)["source"]), "none")


def rows(seeds, floors, crashed=()) -> list[dict]:
    return [{"seed": s, "floor": int(f), "win": False, "end": "death", "decisions": 10, "wall": 1.0, "crash": s in crashed,
             "boss_damage": 0.5} for s, f in zip(seeds, floors, strict=True)]


def judge_never(cr, br):
    return {"decision": {"promote": False, "checks": {}}}


class SmokeBlockTests(unittest.TestCase):
    """The smoke block refuses nothing on the floor difference; a crash still refuses; the later looks are as they were."""

    N = 2000

    @staticmethod
    def diffs(mean: float, n: int, sd: float = SD) -> np.ndarray:
        x = np.random.default_rng(3).normal(0, 1, n)
        return (x - x.mean()) / x.std(ddof=1) * sd + mean  # sample mean and SD exactly as asked

    def arms(self, d_mean: float, crashed=()):
        rng = np.random.default_rng(4)
        seeds = [f"S{i:09d}" for i in range(self.N)]
        base = np.clip(np.rint(rng.normal(17, 8, self.N)), 1, 50)
        cand = np.clip(base + np.rint(rng.normal(d_mean, 2.0, self.N)), 0, 51)
        return seeds, rows(seeds, cand, crashed), rows(seeds, base)

    def test_a_mean_paired_floor_difference_of_minus_3_is_not_refused_at_the_smoke_block(self):
        c = gen_loop.seq_call(self.diffs(-3.0, 200), 0, LOOKS)
        self.assertEqual(c["action"], "continue")
        self.assertIs(c["smoke_would_refuse"], True)  # the figure is still computed and recorded: what the old rule said
        self.assertLess(c["upper99"], 0)  # even the 99% upper bound below 0, which refuses at every later look
        self.assertEqual(gen_loop.seq_call(self.diffs(-6.79, 200), 0, LOOKS)["action"], "continue")  # generation 12's first 200
        self.assertIs(gen_loop.seq_call(self.diffs(-0.5, 200), 0, LOOKS)["smoke_would_refuse"], False)
        self.assertEqual(gen_loop.seq_call(self.diffs(-3.0, 500), 1, LOOKS)["action"], "refuse")  # the next look's rule is unchanged

    def test_the_funnel_plays_the_smoke_block_on_and_refuses_at_the_next_look(self):
        seeds, cand, base = self.arms(-3.0)
        asked = []

        def rows_at(n):
            asked.append(n)
            return cand[:n], base[:n]

        out = gen_loop.funnel(seeds, rows_at, judge_never, smoke=200, batch=300)
        self.assertEqual(asked, [200, 500])  # it played the smoke block, kept going, and the 500 look refused
        self.assertEqual((out["stop"]["action"], out["stop"]["n"], out["stop"]["look"]), ("refuse", 500, 2))
        first = out["looks"][0]
        self.assertEqual((first["action"], first["n"], first["smoke_would_refuse"]), ("continue", 200, True))
        self.assertLess(first["mean"], -2.5)  # its figures are in the record

    def test_a_crashing_candidate_is_still_refused_at_the_smoke_block(self):
        seeds, cand, base = self.arms(0.0, crashed=set(f"S{i:09d}" for i in range(0, 60, 10)))  # 6 crashed seeds in the first 200; slack is 2
        out = gen_loop.funnel(seeds, lambda n: (cand[:n], base[:n]), judge_never, smoke=200, batch=300)
        self.assertEqual((out["stop"]["action"], out["stop"]["n"]), ("refuse", 200))
        self.assertIn("crash", out["stop"]["reason"])
        self.assertEqual(out["looks"][0]["crash"], {"holds": False, "cand": 6, "base": 0, "slack": 2, "n": 200})
        seeds, cand, base = self.arms(0.0, crashed={"S000000000", "S000000001"})  # two is within the slack: play on
        out = gen_loop.funnel(seeds, lambda n: (cand[:n], base[:n]), judge_never, smoke=200, batch=300)
        self.assertGreater(out["stop"]["n"], 200)
        self.assertIs(out["looks"][0]["crash"]["holds"], True)

    def test_the_crash_check_is_the_verdicts_guard_not_a_new_rule(self):
        seeds, cand, base = self.arms(0.0, crashed={f"S{i:09d}" for i in range(0, 60, 10)})
        guard = k2.crash_guard(cand[:200], base[:200])
        self.assertEqual((guard["holds"], guard["cand"], guard["slack"]), (False, 6, 2))

    def test_the_v2_smoke_look_refuses_nothing_on_the_floor_and_records_what_it_would_have(self):
        floor, deep = fx.std_diffs(-3.0, SD, 200, 3), fx.std_diffs(-1.0, 20.0, 200, 5)
        c = k2.seq_call(floor, deep, 0, LOOKS)
        self.assertEqual(c["action"], "continue")
        self.assertIs(c["smoke_would_refuse"], True)
        c = k2.seq_call(fx.std_diffs(-0.5, SD, 200, 3), fx.std_diffs(1.0, 20.0, 200, 5), 0, LOOKS)
        self.assertEqual((c["action"], c["smoke_would_refuse"]), ("continue", False))
        # the later looks keep their rule: both routes dead refuses
        floor, deep = fx.std_diffs(-3.0, SD, 500, 3), fx.std_diffs(-3.0, 20.0, 500, 5)
        self.assertEqual(k2.seq_call(floor, deep, 1, LOOKS)["action"], "refuse")


if __name__ == "__main__":
    unittest.main()
