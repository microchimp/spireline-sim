"""Build against build: the full-state hash (corpus.py identity) and the RNG draw log (rng_diff).

A refactor is byte-safe when every fight leaves every state field the same after every step and draws
the same values at the same places. Each check here has a faithful case that passes and broken ones
that must fail: a hash that differs from some step on, a run without hashes, a replay error on one side
only; a draw moved to another caller or step with every stream's count unchanged, a draw added; a draw
moved in a fight that needs a console setup (rng_diff replays those like identity does, and says so when
it leaves a fight out).
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import corpus as cp  # noqa: E402
from tools.fidelity import rng_diff as rd  # noqa: E402


def logged(*steps) -> dict:
    """A fight as `EmuDump fight --rng-log` writes it: per step, its draws."""
    return {"seed": "RNGDIFF001", "steps": [{"rng_log": [list(d) for d in s]} for s in steps]}


SHUFFLE = ("Shuffle", 1, "CardEffects.ShufflePile < CombatFactory.Reset")
SHUFFLE2 = ("Shuffle", 2, "CardEffects.ShufflePile < CombatFactory.Reset")
AI = ("MonsterAi", 1, "EnemyAI.ChooseIntents < CombatEngine.EndTurn")


class RngDiffTest(unittest.TestCase):
    def test_the_same_draws_are_the_same(self):
        self.assertIsNone(rd.diff_fight(logged([SHUFFLE, SHUFFLE2], [AI]), logged([SHUFFLE, SHUFFLE2], [AI])))

    def test_a_draw_at_another_caller_with_the_same_counts_is_moved(self):
        moved = ("Shuffle", 2, "CardEffects.NextIndex < CardEffects.ShufflePile")
        d = rd.diff_fight(logged([SHUFFLE, SHUFFLE2], [AI]), logged([SHUFFLE, moved], [AI]))
        self.assertEqual((d["kind"], d["stream"], d["draw"], d["counts"]), ("moved", "Shuffle", 2, {}))

    def test_a_draw_in_another_step_is_moved(self):
        d = rd.diff_fight(logged([SHUFFLE], [AI]), logged([], [SHUFFLE, AI]))
        self.assertEqual((d["kind"], d["stream"], d["at_step"]), ("moved", "Shuffle", 0))

    def test_a_draw_at_another_stream_position_is_moved(self):
        # The same caller and step, but the stream was reset (or advanced unlogged) before the draw.
        d = rd.diff_fight(logged([SHUFFLE, SHUFFLE2], [AI]), logged([SHUFFLE, ("Shuffle", 1, SHUFFLE[2])], [AI]))
        self.assertEqual((d["kind"], d["stream"], d["draw"]), ("moved", "Shuffle", 2))
        self.assertEqual((d["a"][2], d["b"][2]), (2, 1))

    def test_an_extra_draw_changes_a_count(self):
        d = rd.diff_fight(logged([SHUFFLE], [AI]), logged([SHUFFLE], [AI, ("MonsterAi", 2, AI[2])]))
        self.assertEqual((d["kind"], d["counts"]), ("count", {"MonsterAi": (1, 2)}))

    def test_an_error_on_one_side_differs(self):
        self.assertEqual(rd.diff_fight(logged([SHUFFLE]), dict(logged([SHUFFLE]), error="boom"))["kind"], "error")

    def test_the_command_line_fails_on_any_difference_and_on_nothing_compared(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / "a.json"
            b = Path(d) / "b.json"
            a.write_text(json.dumps({"results": [logged([SHUFFLE], [AI])]}), encoding="utf-8")
            b.write_text(json.dumps({"results": [logged([SHUFFLE], [AI])]}), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(rd.main(["--logs", str(a), str(b)]), 0)
                b.write_text(json.dumps({"results": [logged([SHUFFLE], [("MonsterAi", 1, "Other.Caller < X.Y")])]}), encoding="utf-8")
                self.assertEqual(rd.main(["--logs", str(a), str(b)]), 1)
                b.write_text(json.dumps({"results": []}), encoding="utf-8")
                self.assertEqual(rd.main(["--logs", str(a), str(b)]), 1)
                a.write_text(json.dumps({"results": []}), encoding="utf-8")
                self.assertEqual(rd.main(["--logs", str(a), str(b)]), 1)


class RngDiffFightSelectionTest(unittest.TestCase):
    """Which fights rng_diff replays. A setup fight is a fight: some draws (potions, the creation of a few
    enemies) happen only there, and a draw moved in one must fail the check, as it fails corpus.py identity."""

    PLAIN, SETUP, HPMULT = "PLAINFIGHT", "SETUPFIGHT", "HPMULTFIGHT"

    def corpus(self, d: str) -> Path:
        c = cp.Corpus(Path(d) / "corpus")
        c.fights_dir.mkdir(parents=True)
        metas = []
        for fid, needs in ((self.PLAIN, []), (self.SETUP, ["setup"]), (self.HPMULT, ["enemy_hp_mult"])):
            c.write(fid, {"seed": fid, "ascension": 10, "steps": [{}]})
            metas.append({"id": fid, "seed": fid, "needs": needs})
        c.save_index({"format": cp.INDEX_FORMAT, "fights": metas, "scanned": {}, "excluded": {}})
        return c.path

    def args(self, corpus: Path, **kw) -> SimpleNamespace:
        return SimpleNamespace(**{"fights": None, "corpus": str(corpus), "only": None, "limit": None, **kw})

    def test_setup_and_hp_mult_fights_are_replayed_when_the_emudump_source_applies_them(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(cp, "emudump_features", return_value={"setup", "enemy_hp_mult"}):
            fights, left = rd.load_fights(self.args(self.corpus(d)))
        self.assertEqual(sorted(f["seed"] for f in fights), sorted([self.PLAIN, self.SETUP, self.HPMULT]))
        self.assertEqual(left, {})

    def test_a_fight_the_emudump_source_cannot_replay_is_counted_with_its_reason(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(cp, "emudump_features", return_value=set()):
            fights, left = rd.load_fights(self.args(self.corpus(d)))
        self.assertEqual([f["seed"] for f in fights], [self.PLAIN])
        self.assertEqual(sorted((why.split(",")[0], n) for why, n in left.items()), [("needs enemy_hp_mult", 1), ("needs setup", 1)])

    def test_records_from_files_that_cannot_be_replayed_are_counted(self):
        good = {"seed": "GOOD", "ascension": 10, "steps": [{}]}
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "out.json"
            f.write_text(json.dumps({"results": [good, {"seed": "NOSTEPS", "ascension": 10, "steps": []}, {"seed": "NOASC", "steps": [{}]},
                                                  "junk", {"seed": "ERR", "ascension": 10, "steps": [{}], "error": "x"}]}), encoding="utf-8")
            fights, left = rd.load_fights(self.args(Path(d), fights=[str(f)]))
        self.assertEqual([x["seed"] for x in fights], ["GOOD", "ERR"])
        self.assertEqual(left, {"no recorded steps": 1, "no ascension": 1, "not a fight record": 1})

    def run_main(self, features: set, swap_in: str) -> tuple[int, str, list]:
        """rng_diff.main over the fixture corpus, the two builds stubbed: b draws one fight's Shuffle at another caller."""
        replayed = []

        def stub(dll, fights, work, procs, tag):
            replayed.append(sorted(f["seed"] for f in fights) if tag == "a" else None)
            other = "Other.Caller < X.Y" if tag == "b" else SHUFFLE[2]
            return [logged([("Shuffle", 1, other if f["seed"] == swap_in else SHUFFLE[2])]) | {"seed": f["seed"]} for f in fights]

        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(cp, "emudump_features", return_value=features),                 mock.patch.object(rd, "run_emudump", stub), redirect_stdout(out):
            code = rd.main(["--a", "a.dll", "--b", "b.dll", "--corpus", str(self.corpus(d)), "--work", str(Path(d) / "w")])
        return code, out.getvalue(), [r for r in replayed if r]

    def test_a_draw_moved_in_a_setup_fight_fails_the_check(self):
        code, out, replayed = self.run_main({"setup", "enemy_hp_mult"}, swap_in=self.SETUP)
        self.assertEqual(replayed, [sorted([self.PLAIN, self.SETUP, self.HPMULT])])
        self.assertEqual(code, 1, out)
        self.assertIn(f"{self.SETUP}: Shuffle draw 1", out)
        self.assertNotIn("NOT COMPARED", out)

    def test_the_same_draws_in_every_fight_pass(self):
        code, out, _ = self.run_main({"setup", "enemy_hp_mult"}, swap_in="NONE")
        self.assertEqual(code, 0, out)
        self.assertIn("rng_diff: ok: 3 fights, 3 draw for draw the same", out)

    def test_a_fight_left_out_is_said_in_the_output_and_in_the_last_line(self):
        code, out, replayed = self.run_main(set(), swap_in=self.SETUP)
        self.assertEqual(replayed, [[self.PLAIN]])
        self.assertEqual(code, 0, out)  # the fight with the moved draw was not replayed ...
        self.assertIn("NOT COMPARED: 2 fights", out)  # ... and the output says so, in full ...
        self.assertIn("NOT COMPARED 2 (", out.strip().splitlines()[-1])  # ... and in the line a reader stops at.
        self.assertIn("rng_diff: ok: 1 fights", out)


class StateIdentityTest(unittest.TestCase):
    A = {"f1": {"sh": ["aa", "bb", "cc"], "err": ""}, "f2": {"sh": ["dd", "ee"], "err": ""}}

    def test_the_same_hashes_pass(self):
        res = cp.compare_hashes(self.A, json.loads(json.dumps(self.A)))
        self.assertEqual((res["same"], res["differ"], res["errors"], res["missing"]), (2, [], [], []))

    def test_each_broken_case_is_reported(self):
        b = json.loads(json.dumps(self.A))
        b["f1"]["sh"][1] = "XX"                      # hidden state changed from step 1 on
        self.assertEqual(cp.compare_hashes(self.A, b)["differ"], [("f1", 1)])
        b = json.loads(json.dumps(self.A))
        b["f2"]["sh"] = ["dd"]                       # the replay stopped a step early
        self.assertEqual(cp.compare_hashes(self.A, b)["differ"], [("f2", 1)])
        b = json.loads(json.dumps(self.A))
        b["f2"] = {"sh": [None, None], "err": ""}    # a build without --state-hash
        self.assertEqual(cp.compare_hashes(self.A, b)["missing"], ["f2"])
        b = json.loads(json.dumps(self.A))
        del b["f1"]
        self.assertEqual(cp.compare_hashes(self.A, b)["missing"], ["f1"])
        b = json.loads(json.dumps(self.A))
        b["f1"]["err"] = "System.Exception: boom"
        self.assertEqual(len(cp.compare_hashes(self.A, b)["errors"]), 1)

    def test_identity_of_two_stored_runs_exits_1_on_a_difference(self):
        with tempfile.TemporaryDirectory() as d:
            corpus = cp.Corpus(Path(d) / "corpus")
            ra, rb = Path(d) / "a.json", Path(d) / "b.json"
            ra.write_text(json.dumps({"fights": self.A}), encoding="utf-8")
            rb.write_text(json.dumps({"fights": self.A}), encoding="utf-8")
            args = ["--corpus", str(corpus.path), "identity", "--runs", str(ra), str(rb)]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cp.main(args), 0)
                changed = json.loads(json.dumps(self.A))
                changed["f2"]["sh"][1] = "zz"
                rb.write_text(json.dumps({"fights": changed}), encoding="utf-8")
                self.assertEqual(cp.main(args), 1)
                rb.write_text(json.dumps({"fights": {}}), encoding="utf-8")
                self.assertEqual(cp.main(args), 1)


if __name__ == "__main__":
    unittest.main()
