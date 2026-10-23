"""tools.fidelity.byte_proof: the comparison of two EmuDump builds, per fight, byte for byte.

A fake EmuDump (a Python script standing in for EmuDump.dll) writes one result per input fight in the
shape the real one does; the proof's splitting, comparison, error-text rule and run-failure handling are
checked on it, and on made-up outputs.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import byte_proof as bp  # noqa: E402

FAKE = textwrap.dedent('''
    import json, sys
    games = json.load(open(sys.argv[2], encoding="utf-8"))["results"]
    flags = sys.argv[3:]
    BEHAVIOUR
    out = []
    for g in games:
        r = {"steps": [{"action": None, "state": {"in_combat": True, "hp": 80}}], "seed": g["seed"], "flags": flags}
        out.append(r)
    out = fix(out)
    text = json.dumps({"results": out}, separators=(",", ":"))
    sys.stdout.write(text)
''')


def fake(folder: Path, name: str, behaviour: str = "fix = lambda out: out") -> str:
    p = folder / f"{name}.py"
    p.write_text(FAKE.replace("BEHAVIOUR", behaviour), encoding="utf-8")
    return str(p)


def games(*seeds):
    return [{"seed": s, "steps": [{}, {}]} for s in seeds]


class SplitTests(unittest.TestCase):
    def test_each_element_is_the_text_as_written(self):
        a = '{"steps":[{"a":"x]},{"}],"e":"q\\"uote"}'
        b = '{"error":"boom","steps":[]}'
        self.assertEqual(bp.split_results(f'{{"results":[{a},{b}]}}'.encode()), [a, b])

    def test_indented_output_and_an_empty_array(self):
        raw = b'{\n  "results": [\n    {"k": 1},\n    {"k": [2, 3]}\n  ]\n}\n'
        self.assertEqual(bp.split_results(raw), ['{"k": 1}', '{"k": [2, 3]}'])
        self.assertEqual(bp.split_results(b'{"results":[]}'), [])

    def test_something_that_is_not_a_results_array_is_refused(self):
        with self.assertRaises(ValueError):
            bp.split_results(b"hello")


class CompareTests(unittest.TestCase):
    @staticmethod
    def out(*results):
        return json.dumps({"results": list(results)}, separators=(",", ":")).encode()

    def test_identical_outputs(self):
        o = self.out({"s": 1}, {"s": 2})
        r = bp.compare_outputs(["f1", "f2"], o, o)
        self.assertEqual((r["same"], r["differ"], r["problems"]), (2, [], []))

    def test_one_byte_in_one_fight_names_that_fight_and_where(self):
        a = self.out({"s": 1, "hp": 80}, {"s": 2, "hp": 80})
        b = self.out({"s": 1, "hp": 80}, {"s": 2, "hp": 81})
        r = bp.compare_outputs(["f1", "f2"], a, b)
        self.assertEqual(r["same"], 1)
        self.assertEqual([d["id"] for d in r["differ"]], ["f2"])
        d = r["differ"][0]
        self.assertEqual(d["a"][d["at"] - max(0, d["at"] - 40)], "0")  # the context starts before the first differing char
        self.assertNotEqual(d["sha_a"], d["sha_b"])

    def test_key_order_is_a_difference(self):
        r = bp.compare_outputs(["f1"], self.out({"a": 1, "b": 2}), self.out({"b": 2, "a": 1}))
        self.assertEqual(len(r["differ"]), 1)

    def test_a_stack_trace_is_compared_by_its_first_line_and_counted(self):
        a = self.out({"steps": [], "error": "System.InvalidOperationException: no\r\n   at Fight.Replay() in C:\\a\\Fight.cs:line 85"})
        b = self.out({"steps": [], "error": "System.InvalidOperationException: no\r\n   at OldEngine.StartFight() in C:\\b\\OldEngine.cs:line 9"})
        r = bp.compare_outputs(["f1"], a, b)
        self.assertEqual((r["same"], r["differ"], r["error_cut"]), (1, [], 1))

    def test_a_different_error_message_is_still_a_difference(self):
        a = self.out({"steps": [], "error": "System.InvalidOperationException: no\r\n   at x"})
        b = self.out({"steps": [], "error": "System.InvalidOperationException: nope\r\n   at x"})
        self.assertEqual(len(bp.compare_outputs(["f1"], a, b)["differ"]), 1)

    def test_what_precedes_the_error_is_still_compared(self):
        a = self.out({"steps": [{"hp": 1}], "error": "E\r\n at x"})
        b = self.out({"steps": [{"hp": 2}], "error": "E\r\n at y"})
        self.assertEqual(len(bp.compare_outputs(["f1"], a, b)["differ"]), 1)

    def test_a_missing_side_or_a_short_output_is_a_problem_not_a_pass(self):
        o = self.out({"s": 1}, {"s": 2})
        r = bp.compare_outputs(["f1", "f2"], o, None)
        self.assertEqual((r["same"], len(r["problems"])), (0, 1))
        r = bp.compare_outputs(["f1", "f2"], o, self.out({"s": 1}))
        self.assertEqual((r["same"], len(r["problems"])), (0, 1))
        r = bp.compare_outputs(["f1"], b"garbage", o[:0] + self.out({"s": 1}))
        self.assertEqual(len(r["problems"]), 1)


class ModeTableTests(unittest.TestCase):
    def test_the_four_modes_and_their_flags(self):
        self.assertEqual({m: bp.MODES[m] for m in ("fight", "netstate", "state-hash", "rng-log")}, {
            "fight": ("fight", ["--compact"]),
            "netstate": ("fight", ["--compact", "--netstate"]),
            "state-hash": ("fight", ["--compact", "--state-hash=fields"]),
            "rng-log": ("fight", ["--compact", "--rng-log"]),
        })


class FakeEmuDumpTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def run_mode(self, mode, a, b, seeds=("S1", "S2", "S3")):
        ids = [f"id-{s}" for s in seeds]
        return bp.prove_batch(mode, ids, games(*seeds), a, b, str(self.dir), timeout=60)

    def test_identical_builds_are_identical_in_every_mode(self):
        a, b = fake(self.dir, "a"), fake(self.dir, "b")
        for mode in bp.MODES:
            r = self.run_mode(mode, a, b)
            self.assertEqual((r["same"], r["differ"], r["problems"]), (3, [], []), mode)

    def test_the_mode_decides_the_flags_the_build_gets(self):
        # The fake writes its flags into every result: a build that ignored them would not differ here.
        a = fake(self.dir, "a")
        b = fake(self.dir, "b", 'fix = lambda out: [dict(o, flags=flags[:1]) for o in out]')
        self.assertEqual(self.run_mode("fight", a, b)["differ"], [])  # --compact only: the same first flag
        self.assertEqual(len(self.run_mode("netstate", a, b)["differ"]), 3)  # a has two flags, b keeps one

    def test_a_build_that_changes_one_fight_is_named(self):
        a = fake(self.dir, "a")
        b = fake(self.dir, "b", 'fix = lambda out: [dict(o, steps=[{"action": None, "state": {"in_combat": True, "hp": 79}}]) if o["seed"] == "S2" else o for o in out]')
        r = self.run_mode("fight", a, b)
        self.assertEqual((r["same"], [d["id"] for d in r["differ"]], r["problems"]), (2, ["id-S2"], []))

    def test_a_build_that_crashes_or_drops_a_fight_is_a_problem(self):
        a = fake(self.dir, "a")
        crash = fake(self.dir, "crash", "sys.exit(3)")
        r = self.run_mode("fight", a, crash)
        self.assertEqual(r["same"], 0)
        self.assertTrue(any("exit 3" in p for p in r["problems"]), r["problems"])
        short = fake(self.dir, "short", "fix = lambda out: out[:-1]")
        r = self.run_mode("fight", a, short)
        self.assertEqual(r["same"], 0)
        self.assertTrue(r["problems"])

    def test_the_report_exits_one_on_any_difference_or_problem_or_an_empty_run(self):
        def totals(**kw):
            t = {"fights": 3, "same": 3, "differ": [], "problems": [], "error_cut": 0, "seconds": 0.0}
            return {"fight": {**t, **kw}}

        def run(t):
            with redirect_stdout(io.StringIO()) as out:
                return bp.report(t, {}, 5), out.getvalue()

        self.assertEqual(run(totals())[0], 0)
        code, text = run(totals(same=2, differ=[{"id": "f", "at": 3, "a": "x", "b": "y", "len_a": 1, "len_b": 1, "sha_a": "a", "sha_b": "b"}]))
        self.assertEqual(code, 1)
        self.assertIn("FAIL", text)
        self.assertEqual(run(totals(problems=["a: no output"]))[0], 1)
        self.assertEqual(run(totals(fights=0, same=0))[0], 1)


if __name__ == "__main__":
    unittest.main()
