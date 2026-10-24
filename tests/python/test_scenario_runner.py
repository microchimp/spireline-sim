"""tools/fidelity/scenarios.py and `EmuDump scenario`: the checked-in directory (tests/scenarios) on both engines, and broken
cases the runner must catch (a wrong expectation, an xfail that passes, an unknown key, an engine the scenario leaves out, a
`block`/`enemy_hp`/`enemy_block` setter the fight cannot hold or the format rejects).
Needs the Release EmuDump (tools/build_and_test.sh builds it before the Python suite)."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import scenarios as S  # noqa: E402

BUILT = S.DEFAULT_DLL.is_file() and shutil.which("dotnet")
REGRESSION = S.DEFAULT_DIR / "regression" / "strike-damages-the-enemy-and-costs-1.json"
SETTERS = "regression/block-absorbs-first-and-zero-hp-kills"  # the scenario that sets the enemy's HP and block (SCENARIOS.md)


class Rendering(unittest.TestCase):
    def test_unexpected_is_a_fail_an_xpass_or_an_invalid_file(self):
        rows = [{"name": n, "file": "f", "status": s, "unexpected": s in ("fail", "xpass", "invalid"), "diffs": ["d"], "reason": "r"}
                for n, s in [("a", "pass"), ("b", "fail"), ("c", "xfail"), ("d", "xpass"), ("e", "invalid"), ("f", "skip")]]
        rep = {"engine": "old", "results": rows, "summary": {"pass": 1, "fail": 1, "xfail": 1, "xpass": 1, "invalid": 1, "skip": 1}}
        self.assertEqual([r["name"] for r in S.unexpected(rep)], ["b", "d", "e"])
        text = S.render(rep)
        self.assertIn("XPASS", text)
        self.assertNotIn("PASS   f", text)  # a pass is not listed


@unittest.skipUnless(BUILT, "EmuDump is not built (dotnet build tools/emudump/EmuDump.csproj -c Release)")
class CheckedInScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = {e: S.run_dir(S.DEFAULT_DIR, e) for e in S.ENGINES}

    def test_the_directory_passes_on_both_engines_with_its_listed_xfails(self):
        for engine, rep in self.reports.items():
            self.assertEqual(S.unexpected(rep), [], engine)
            self.assertEqual(rep["exit"], 0, engine)
            self.assertGreaterEqual(rep["summary"]["pass"], 10, engine)
            self.assertGreaterEqual(rep["summary"]["xfail"], 1, engine)

    def test_every_batch3_class_is_converted_and_the_regression_cover_passes_on_both(self):
        for engine, rep in self.reports.items():
            names = {r["name"] for r in rep["results"]}
            for family in ("exhaust-order/", "victory-heal/", "rounding-order/"):
                self.assertTrue(any(n.startswith(family) for n in names), (engine, family))
            reg = [r for r in rep["results"] if r["name"].startswith("regression/")]
            self.assertGreaterEqual(len(reg), 3)
            self.assertTrue(all(r["status"] == "pass" for r in reg), engine)

    def test_what_old_fails_core2_fails_too_while_no_switch_is_on(self):
        """core2 with no family switch runs old's rules (migrated_families.json): the two engines' outcomes agree."""
        a, b = (sorted((r["name"], r["status"]) for r in self.reports[e]["results"]) for e in S.ENGINES)
        self.assertEqual(a, b)

    def test_the_scenario_that_sets_enemy_hp_and_block_passes_on_both_engines(self):
        for engine, rep in self.reports.items():
            status = {r["name"]: r["status"] for r in rep["results"]}
            self.assertEqual(status.get(SETTERS), "pass", (engine, status.get(SETTERS)))

    def test_the_runner_names_the_differing_path_of_a_known_failure(self):
        old = {r["name"]: r for r in self.reports["old"]["results"]}
        r = old["victory-heal/meat-at-threshold-then-burning-blood"]
        self.assertEqual(r["status"], "xfail")
        self.assertIn("snapshot.player.hp", r["diffs"][0])
        self.assertIn("expected 58", r["diffs"][0])


@unittest.skipUnless(BUILT, "EmuDump is not built (dotnet build tools/emudump/EmuDump.csproj -c Release)")
class BrokenCases(unittest.TestCase):
    """A scenario file that is wrong in a known way must end in the outcome that fails the run (or, for a listed xfail, not)."""

    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        base = json.loads(REGRESSION.read_text(encoding="utf-8"))

        def make(key, edit):
            sc = copy.deepcopy(base)
            sc["name"] = "broken/" + key
            edit(sc)
            (cls.dir / f"{key}.json").write_text(json.dumps(sc), encoding="utf-8")

        make("passes", lambda sc: None)
        make("wrong-value", lambda sc: sc["expect"][0].update(equals=-7))
        make("xfail-that-fails", lambda sc: (sc["expect"][0].update(equals=-7), sc.update(xfail_engines={"old": "a reason the row waits on"})))
        make("xfail-that-passes", lambda sc: sc.update(xfail_engines={"old": "a reason the row waits on"}))
        make("other-engine-only", lambda sc: sc.update(engines=["core2"]))
        make("typo-key", lambda sc: sc["setup"].update(relic=["BURNING_BLOOD"]))
        make("no-source", lambda sc: sc.pop("source"))
        make("illegal-action", lambda sc: sc["actions"][0].update(target=5))
        make("unknown-model", lambda sc: sc["setup"].update(deck=["NO_SUCH_CARD"]))
        make("bad-path", lambda sc: sc["expect"].append({"path": "snapshot.hand[x", "equals": 1}))
        make("xfail-engine-not-run", lambda sc: sc.update(engines=["core2"], xfail_engines={"old": "a reason the row waits on"}))
        # The setters: a slot the fight has no living enemy in and a current HP above the rolled max are refused at the setup item
        # (not capped), a negative value is caught by the format; one that fits passes, and the start view shows it took.
        make("enemy-slot-unknown", lambda sc: sc["setup"].update(enemy_hp=[{"slot": 7, "current": 5}]))
        make("enemy-hp-above-max", lambda sc: sc["setup"].update(enemy_hp=[{"slot": 0, "current": 9999}]))
        make("enemy-block-slot-unknown", lambda sc: sc["setup"].update(enemy_block=[{"slot": 7, "amount": 5}]))
        make("negative-block", lambda sc: sc["setup"].update(block=-1))
        make("negative-enemy-block", lambda sc: sc["setup"].update(enemy_block=[{"slot": 0, "amount": -1}]))
        make("enemy-hp-current-above-its-max", lambda sc: sc["setup"].update(enemy_hp=[{"slot": 0, "max": 5, "current": 6}]))
        make("setters-that-fit", lambda sc: (sc["setup"].update(block=3, enemy_hp=[{"slot": 0, "current": 30}], enemy_block=[{"slot": 0, "amount": 0}]),
                                             sc["expect"].extend([{"at": "start", "path": "snapshot.enemies[0].hp", "equals": 30},
                                                                  {"at": "start", "path": "snapshot.player.block", "equals": 3}])))
        cls.report = S.run_dir(cls.dir, "old")
        cls.status = {r["name"]: r for r in cls.report["results"]}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, True)

    def test_outcomes(self):
        want = {"passes": "pass", "wrong-value": "fail", "xfail-that-fails": "xfail", "xfail-that-passes": "xpass", "other-engine-only": "skip",
                "typo-key": "invalid", "no-source": "invalid", "illegal-action": "fail", "unknown-model": "fail", "bad-path": "invalid",
                "xfail-engine-not-run": "invalid",
                "enemy-slot-unknown": "fail", "enemy-hp-above-max": "fail", "enemy-block-slot-unknown": "fail", "negative-block": "invalid",
                "negative-enemy-block": "invalid", "enemy-hp-current-above-its-max": "invalid", "setters-that-fit": "pass"}
        self.assertEqual({k: self.status["broken/" + k]["status"] for k in want}, want)

    def test_the_run_fails_exactly_when_something_unexpected_happened(self):
        self.assertEqual(self.report["exit"], 1)
        self.assertEqual(sorted(r["name"] for r in S.unexpected(self.report)),
                         sorted("broken/" + k for k in ("wrong-value", "xfail-that-passes", "typo-key", "no-source", "illegal-action", "unknown-model",
                                                        "bad-path", "xfail-engine-not-run", "enemy-slot-unknown", "enemy-hp-above-max",
                                                        "enemy-block-slot-unknown", "negative-block", "negative-enemy-block",
                                                        "enemy-hp-current-above-its-max")))

    def test_diagnostics_name_what_was_wrong(self):
        def d(k):
            return " ".join(self.status["broken/" + k]["diffs"])

        self.assertIn("expected -7, got -6", d("wrong-value"))
        self.assertIn("unknown key 'relic'", d("typo-key"))
        self.assertIn("target 5 is not legal", d("illegal-action"))
        self.assertIn("NO_SUCH_CARD", d("unknown-model"))
        self.assertIn("no living enemy in slot 7", d("enemy-slot-unknown"))
        self.assertIn("enemy_hp", d("enemy-slot-unknown"))
        self.assertIn("current HP 9999 above max HP", d("enemy-hp-above-max"))
        self.assertIn("enemy_block", d("enemy-block-slot-unknown"))
        self.assertIn("setup.block: an integer, 0 or more", d("negative-block"))
        self.assertIn("setup.enemy_block[0].amount", d("negative-enemy-block"))
        self.assertIn("current 6 above max 5", d("enemy-hp-current-above-its-max"))

    def test_a_directory_of_passing_scenarios_exits_0(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        shutil.copy(REGRESSION, d / "one.json")
        self.assertEqual(S.run_dir(d, "old")["exit"], 0)


if __name__ == "__main__":
    unittest.main()
