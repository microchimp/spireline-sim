"""The fight solver's Python binding (src/sts2_gym/native.py, docs/rebuild/FIGHT_SOLVER.md sections 1 and 4a, piece S4).

The ctypes structures mirror Interop/SolverExports.cs field for field (read from the C#, so the two cannot drift); the
expected result and the hindsight bound share no field; the wrappers decode what the library writes and pass the hindsight
flag only where asked. Run against a stand-in library, so no build is needed; the last case runs on the real library when
one is built and fresh.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
NATIVE_PY = ROOT / "src" / "sts2_gym" / "native.py"
SOLVER_CS = ROOT / "src" / "Sts2Emulator" / "Interop" / "SolverExports.cs"
_CS_TYPES = {"long": ctypes.c_int64, "int": ctypes.c_int32, "double": ctypes.c_double}


def load_native_without_a_library(lib=None):
    """native.py imported under another name against a stand-in library that answers the version checks."""
    lib = lib or mock.MagicMock()
    src = NATIVE_PY.read_text(encoding="utf-8")
    lib.Sts2_NativeApiVersion.return_value = int(re.search(r"^_REQUIRED_NATIVE_API_VERSION = (\d+)", src, re.M).group(1))
    lib.Sts2Run_NativeApiVersion.return_value = int(re.search(r"^_REQUIRED_RUN_NATIVE_API_VERSION = (\d+)", src, re.M).group(1))
    lib.Sts2Run_ObsLayout.return_value = 13
    spec = importlib.util.spec_from_file_location("native_solver_under_test", NATIVE_PY)
    mod = importlib.util.module_from_spec(spec)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "Sts2Emulator.dll").write_bytes(b"")
        with mock.patch.dict(os.environ, {"STS2_LIB_PATH": d, "STS2_ALLOW_STALE_NATIVE": "1"}), mock.patch("ctypes.CDLL", return_value=lib):
            os.environ.pop("STS2_ENGINE", None)
            spec.loader.exec_module(mod)
    return mod, lib


def cs_struct(name: str) -> list[tuple[str, object]]:
    """A C# struct's instance fields in order as (snake_case name, ctypes type); a fixed buffer as an array type."""
    src = SOLVER_CS.read_text(encoding="utf-8")
    body = re.search(r"public (?:unsafe )?struct " + name + r"\s*\{(.*?)\n\}", src, re.S).group(1)
    fields = []
    for m in re.finditer(r"^\s*public (fixed )?(long|int|double) (\w+)(?:\[(\d+)\])?;", body, re.M):
        fixed, typ, field, n = m.groups()
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", field).lower()
        ct = _CS_TYPES[typ]
        fields.append((snake, ct * int(n) if fixed else ct))
    return fields


class SolverBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native, cls.lib = load_native_without_a_library()

    def test_the_version_is_40(self):
        self.assertEqual(self.native._REQUIRED_NATIVE_API_VERSION, 40)
        self.assertEqual(self.native._REQUIRED_RUN_NATIVE_API_VERSION, 33)

    def test_every_structure_is_the_csharp_one_field_for_field(self):
        n = self.native
        for cs, py in [("SolveOptionsNative", n._SolveOptionsC), ("ExpectedResultNative", n._ExpectedResultC),
                       ("HindsightOptionsNative", n._HindsightOptionsC), ("HindsightResultNative", n._HindsightResultC),
                       ("CurveEntryNative", n._CurveEntryC)]:
            with self.subTest(struct=cs):
                want = cs_struct(cs)
                self.assertTrue(want, cs)
                got = [(f, t) for f, t in py._fields_]
                self.assertEqual([f for f, _ in got], [f for f, _ in want])
                for (f, t), (_, w) in zip(got, want):
                    self.assertEqual(ctypes.sizeof(t), ctypes.sizeof(w), f)
                    self.assertEqual(getattr(t, "_type_", t), getattr(w, "_type_", w), f)

    def test_the_two_results_share_no_field(self):
        n = self.native
        self.assertFalse(set(n.ExpectedSolve.__slots__) & set(n.HindsightBound.__slots__))
        self.assertFalse({f for f, _ in n._ExpectedResultC._fields_} & {f for f, _ in n._HindsightResultC._fields_})
        self.assertFalse(issubclass(n.ExpectedSolve, n.HindsightBound) or issubclass(n.HindsightBound, n.ExpectedSolve))

    def test_hindsight_is_not_bound_at_import(self):
        self.assertNotIn("Sts2_SolveHindsight", self.lib._mock_children)
        self.assertIn("Sts2_SolveExpected", self.lib._mock_children)

    def test_solve_expected_decodes_the_result_and_the_q_buffer(self):
        n = self.native
        lib = mock.MagicMock()

        def fake(solver, handle, opts, res, q, max_actions):
            r = res._obj
            r.status, r.horizon, r.exact, r.move_count, r.best_action, r.best_target, r.q_stride = 0, 1, 1, 2, 0, 1, 7
            r.value, r.exact_mass, r.nodes, r.allowed_potions = 0.75, 1.0, 42, -1
            for i in range(max_actions * 7):
                q[i] = float("nan")
            q[0 * 7 + 2] = 0.75  # action 0 at target 1
            q[3 * 7 + 0] = 0.5  # end turn, untargeted
            return 0

        lib.Sts2_SolveExpected.side_effect = fake
        with mock.patch.object(n, "_lib", lib):
            out = n.solve_expected(1, 2, n.SolveOptions(node_budget=10))
        self.assertEqual((out.status, out.value, out.exact, out.best_action, out.best_target, out.nodes), (0, 0.75, True, 0, 1, 42))
        self.assertEqual(out.allowed_potions, -1)
        self.assertEqual(out.q, {(0, 1): 0.75, (3, -1): 0.5})
        self.assertIsNone(out.reason)

    def test_a_refusal_carries_its_reason_and_an_error_raises(self):
        n = self.native
        lib = mock.MagicMock()

        def refused(solver, handle, opts, res, q, max_actions):
            res._obj.value, res._obj.exact_mass = float("nan"), float("nan")
            return -3

        lib.Sts2_SolveExpected.side_effect = refused
        with mock.patch.object(n, "_lib", lib), mock.patch.object(n, "_get_last_error", return_value="draw-peek: x"):
            out = n.solve_expected(1, 2)
            self.assertEqual(out.reason, "draw-peek: x")
            self.assertIsNone(out.value)  # no value, never the 0 of a certain loss
            self.assertIsNone(out.exact_mass)
            lib.Sts2_SolveExpected.side_effect = None
            lib.Sts2_SolveExpected.return_value = -1
            with self.assertRaisesRegex(RuntimeError, "Sts2_SolveExpected failed: draw-peek"):
                n.solve_expected(1, 2)

    def test_the_hindsight_flag_is_passed_only_when_asked(self):
        n = self.native
        lib = mock.MagicMock()
        seen = []

        def create(handle, opts):
            seen.append(opts._obj.allow_hindsight)
            return 4

        lib.Sts2Solver_Create.side_effect = create
        with mock.patch.object(n, "_lib", lib):
            self.assertEqual(n.solver_create(1), 4)
            n.solver_create(1, n.SolveOptions(), allow_hindsight=True)
        self.assertEqual(seen, [0, 1])

    def test_the_potion_mask_crosses_as_its_complement(self):
        # A zeroed native struct forbids nothing; Python's slot mask of allowed potions is complemented on the way.
        n = self.native
        self.assertEqual(n._SolveOptionsC().forbidden_potions, 0)
        self.assertEqual(n.SolveOptions()._c().forbidden_potions, 0)
        self.assertEqual(n.SolveOptions(allowed_potions=~0b101)._c().forbidden_potions, 0b101)

    def test_reset_raises_on_an_unknown_solver_and_open_solver_destroys_its_handle(self):
        n = self.native
        lib = mock.MagicMock()
        lib.Sts2Solver_Reset.return_value = -1
        lib.Sts2Solver_Create.return_value = 7
        with mock.patch.object(n, "_lib", lib), mock.patch.object(n, "_get_last_error", return_value="unknown solver handle"):
            with self.assertRaisesRegex(RuntimeError, "unknown solver handle"):
                n.solver_reset(9)
            with self.assertRaises(KeyError), n.open_solver(1) as s:
                self.assertEqual(s, 7)
                raise KeyError("inside")
        lib.Sts2Solver_Destroy.assert_called_once_with(7)

    def test_a_refused_curve_entry_has_no_numbers(self):
        n = self.native
        lib = mock.MagicMock()

        def fake(solver, handle, opts, buf, max_subsets):
            for i, (status, value) in enumerate([(-3, float("nan")), (0, 0.5)]):
                e = buf[i]
                e.subset, e.curve_status, e.allowed_potions = i, status, -1 if i else ~1
                e.expected_l0 = e.win_chance = e.curve_exact_mass = e.expected_win_hp = value
                for slot in range(8):
                    e.held[slot] = float("nan")
            return 2

        lib.Sts2_SolvePotionCurve.side_effect = fake
        with mock.patch.object(n, "_lib", lib):
            refused, solved = n.solve_potion_curve(1, 2, max_subsets=2)
        self.assertEqual((refused["expected_l0"], refused["win_chance"], refused["held"], refused["allowed_potions"]), (None, None, {}, ~1))
        self.assertEqual((solved["expected_l0"], solved["allowed_potions"]), (0.5, -1))

    def test_solve_hindsight_binds_on_first_call_and_decodes_the_line(self):
        n = self.native
        lib = mock.MagicMock()

        def fake(solver, handle, opts, res):
            r = res._obj
            r.outcome, r.win_line_exists, r.complete, r.line_length, r.best_utility = 0, 1, 1, 2, 0.9
            r.line[0], r.line[1], r.line[2], r.line[3] = 0, 0, 5, -1
            return 0

        lib.Sts2_SolveHindsight.side_effect = fake
        with mock.patch.object(n, "_lib", lib), mock.patch.object(n, "_hindsight_export", None):
            self.assertIsNone(n._hindsight_export)
            b = n.solve_hindsight(1, 2)
            self.assertIs(n._hindsight_export, lib.Sts2_SolveHindsight)
            self.assertEqual(len(lib.Sts2_SolveHindsight.argtypes), 4)
        self.assertEqual((b.win_line_exists, b.best_utility, b.complete, b.line), ("yes", 0.9, True, [(0, 0), (5, -1)]))

    def test_a_cut_or_refused_hindsight_has_no_bound(self):
        n = self.native
        for outcome in (1, -3):
            lib = mock.MagicMock()

            def fake(solver, handle, opts, res, outcome=outcome):
                res._obj.outcome, res._obj.best_utility = outcome, float("nan")
                return outcome

            lib.Sts2_SolveHindsight.side_effect = fake
            with mock.patch.object(n, "_lib", lib), mock.patch.object(n, "_hindsight_export", None), \
                    mock.patch.object(n, "_get_last_error", return_value="hindsight-unset-streams: x"):
                self.assertIsNone(n.solve_hindsight(1, 2).best_utility, outcome)


def _real_library():
    lib = Path(os.environ.get("STS2_LIB_PATH", ROOT / "out")) / "Sts2Emulator.dll"
    return lib if lib.is_file() else None


@unittest.skipUnless(_real_library(), "no native library built")
class SolverBindingLiveTests(unittest.TestCase):
    def test_a_small_fight_end_to_end(self):
        import sys  # noqa: PLC0415

        if str(ROOT / "src") not in sys.path:
            sys.path.insert(0, str(ROOT / "src"))
        from sts2_gym import native  # noqa: PLC0415

        obs = (ctypes.c_int * native._lib.Sts2_ObsSize())()
        h = native.create(11)
        native.reset(h, obs)
        s = native.solver_create(h, native.SolveOptions(node_budget=20_000, max_horizon=0))
        try:
            e = native.solve_expected(s, h)
            self.assertIn(e.status, (0, 1))
            self.assertTrue(0.0 <= e.value <= 1.0)
            self.assertIn((e.best_action, e.best_target), e.q)
            self.assertEqual(native.solve_hindsight(s, h).outcome, -3)  # made without the flag
            self.assertEqual(native.solver_info()["native_api_version"], 40)
            # With the flag, a handle from create() and reset() is still refused: some of its streams are unset.
            with native.open_solver(h, allow_hindsight=True) as flagged:
                b = native.solve_hindsight(flagged, h, node_budget=2_000)
            self.assertEqual(b.outcome, -3)
            self.assertTrue(b.refusal.startswith("hindsight-unset-streams"), b.refusal)
            self.assertIsNone(b.best_utility)
            native.solver_reset(s)
        finally:
            native.solver_destroy(s)
            native.destroy(h)


if __name__ == "__main__":
    unittest.main()
