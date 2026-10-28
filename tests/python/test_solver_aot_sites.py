"""The solver in the published native library against the JIT (review 1 of the solver optimisation plan, section 6 item 1).

The xunit host runs the solver under the JIT, so it cannot see a defect of the NativeAOT build alone: the old stack walks
that found the discard reshuffle and the sorted pick returned nothing there, leaving reshuffles unsteered and draw ties
unrefused. Here the AOT library (out/Sts2Emulator.dll, through sts2_gym.native) and `dotnet EmuDump.dll solve --snapshot`
(the JIT) solve the same `sts2-emu-snapshot/1` documents and must agree: status, value, refusal, node and step counts.
tests/fixtures/solver/aot_sites.jsonl holds a reshuffling position and a draw-tie position (AotSiteFixtureTests keeps it
current); with STS2_SOLVER_DOCS naming the harness documents (written by AotSiteFixtureTests.ExportTheHarnessDocumentsWhenAsked)
the whole harness set is compared too: S4's native-equals-EmuDump gate with EmuDump under the JIT.

Skipped unless the library, EmuDump.dll (STS2_EMUDUMP_DLL, or its Release build) and dotnet are all present.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "solver" / "aot_sites.jsonl"
LOAD_SEED = 5
NODE_BUDGET = 200_000
STEP_BUDGET = 2_000_000
_STATUS = {0: "exact", 1: "budget", -3: "refused"}


def _library() -> Path | None:
    lib = Path(os.environ.get("STS2_LIB_PATH", ROOT / "out")) / "Sts2Emulator.dll"
    return lib if lib.is_file() else None


def _emudump() -> Path | None:
    p = Path(os.environ.get("STS2_EMUDUMP_DLL", ROOT / "tools" / "emudump" / "bin" / "Release" / "net9.0" / "EmuDump.dll"))
    return p if p.is_file() else None


_READY = _library() is not None and _emudump() is not None and shutil.which("dotnet") is not None


def _jit(lines: list[dict], engine: str, horizon: int) -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "docs.jsonl"
        f.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
        cmd = ["dotnet", str(_emudump()), "solve", "--snapshot", str(f), "--load-seed", str(LOAD_SEED), "--horizon", str(horizon),
               "--budget-nodes", str(NODE_BUDGET), "--budget-steps", str(STEP_BUDGET), "--no-estimate"]
        if engine != "old":
            cmd += ["--engine", engine]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
    if r.returncode != 0:
        raise AssertionError(r.stderr[-2000:])
    return [json.loads(x) for x in r.stdout.splitlines() if x.strip()]


def _aot(native, doc: dict, horizon: int):
    opts = native.SolveOptions(node_budget=NODE_BUDGET, step_budget=STEP_BUDGET, max_horizon=horizon, use_estimate=False)
    h = native.load_combat_snapshot(doc, LOAD_SEED)
    s = native.solver_create(h, opts)
    try:
        return native.solve_expected(s, h, opts)
    finally:
        native.solver_destroy(s)
        native.destroy(h)


def _differences(native, lines: list[dict]) -> tuple[int, list[str]]:
    """AOT against JIT on every document, grouped by (engine, horizon) so that each JIT run is one process."""
    diffs, compared = [], 0
    groups: dict[tuple[str, int], list[dict]] = {}
    for x in lines:
        groups.setdefault((x.get("engine", "old"), int(x["horizon"])), []).append(x)
    for (engine, horizon), group in sorted(groups.items()):
        native.set_engine(engine)
        jit = _jit(group, engine, horizon)
        if len(jit) != len(group):
            diffs.append(f"{engine}/h{horizon}: {len(jit)} JIT lines for {len(group)} documents")
            continue
        for x, j in zip(group, jit):
            tag = x.get("name", x.get("id"))
            try:
                e = _aot(native, x["doc"], horizon)
            except Exception as ex:  # noqa: BLE001  a document the loader refuses (SnapshotError: a stopped step under a kernel switch)
                if "error" not in j:
                    diffs.append(f"{tag}: AOT refused the document ({ex}), the JIT solved it")
                continue
            if "error" in j:
                diffs.append(f"{tag}: JIT error {j['error']}, the AOT library solved it")
                continue
            compared += 1
            je = j["expected"]
            if _STATUS.get(e.status) != je["status"]:
                diffs.append(f"{tag}: status AOT {e.status} ({e.reason}) vs JIT {je['status']} ({je['reason']})")
            elif e.status == -3:
                if e.reason != je["reason"]:
                    diffs.append(f"{tag}: refusal AOT {e.reason!r} vs JIT {je['reason']!r}")
            elif (e.value, e.exact, e.horizon) != (je["value_f"], je["exact"], je["horizon"]):
                diffs.append(f"{tag}: AOT {e.value} {e.exact} h{e.horizon} vs JIT {je['value_f']} {je['exact']} h{je['horizon']}")
            elif (e.nodes, e.steps) != (j["counters"]["nodes"], j["counters"]["steps"]):
                diffs.append(f"{tag}: nodes/steps AOT {e.nodes}/{e.steps} vs JIT {j['counters']['nodes']}/{j['counters']['steps']}")
    native.set_engine("old")
    return compared, diffs


@unittest.skipUnless(_READY, "needs the published library, EmuDump.dll and dotnet")
class SolverAotSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "src"))
        from sts2_gym import native  # noqa: PLC0415

        cls.native = native

    def test_the_reshuffle_and_the_draw_tie_solve_as_under_the_jit(self):
        lines = [json.loads(x) for x in FIXTURE.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertEqual({x["name"] for x in lines}, {"reshuffle", "draw-tie"})
        compared, diffs = _differences(self.native, lines)
        self.assertEqual(compared, 2)
        self.assertEqual(diffs, [])
        by_name = {x["name"]: x for x in lines}
        tie = _aot(self.native, by_name["draw-tie"]["doc"], 0)
        self.assertEqual(tie.status, -3)
        self.assertTrue(tie.reason.startswith("draw-tie"), tie.reason)
        shuffled = _aot(self.native, by_name["reshuffle"]["doc"], 1)
        self.assertIn(shuffled.status, (0, 1))
        self.assertLess(shuffled.steps, 50_000)  # steered; unsteered, each Fisher-Yates call of the reshuffle branches

    @unittest.skipUnless(os.environ.get("STS2_SOLVER_DOCS"), "set STS2_SOLVER_DOCS to the harness documents")
    def test_the_harness_set_solves_as_under_the_jit(self):
        path = Path(os.environ["STS2_SOLVER_DOCS"])
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        compared, diffs = _differences(self.native, lines)
        print(f"AOT vs JIT on the harness set: {compared} documents, {len(diffs)} differences", file=sys.stderr)
        self.assertGreaterEqual(compared, 300)
        self.assertEqual(diffs[:20], [])


if __name__ == "__main__":
    unittest.main()
