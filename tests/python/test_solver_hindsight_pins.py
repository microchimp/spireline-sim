"""The hindsight bound stays away from the lab and the search (docs/rebuild/FIGHT_SOLVER.md section 4a, piece S4).

Perfect play is best in expectation at decision time; the hindsight bound (native.solve_hindsight, Sts2_SolveHindsight,
HindsightBound) is for lost-fight attribution only and never a label, a regret or a grade. Pinned three ways, in the
manner of test_native_entry_pins.py: each lab and search entry point is run in a fresh interpreter in which sts2_gym.native
is a stub recording every name read from it, and none may read a hindsight name; sts2_gym.native itself binds no hindsight
export at import; and no lab, training or search module names one in its source.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HINDSIGHT_NAMES = ("solve_hindsight", "Sts2_SolveHindsight", "HindsightBound", "_bind_hindsight", "_HindsightResultC")


def _pinned_commands() -> list[str]:
    spec = importlib.util.spec_from_file_location("entry_pins_for_hindsight", Path(__file__).with_name("test_native_entry_pins.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.PINNED_COMMANDS)


# The lab's self-play, promotion and training commands, and the search's entry points.
SEARCH_COMMANDS = ["rl.combat_mcts", "rl.live_search"]

PROBE = r"""
import importlib, importlib.abc, importlib.machinery, json, runpy, sys, types
from pathlib import Path
from unittest import mock
root, name = Path(sys.argv[1]), sys.argv[2]
sys.path[:0] = [str(root), str(root / "rl"), str(root / "src"), str(root / "referee")]
read = set()


class Stub(types.ModuleType):
    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        read.add(attr)
        return mock.MagicMock(name=attr)


class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        return importlib.machinery.ModuleSpec(fullname, self) if fullname == "sts2_gym.native" else None

    def create_module(self, spec):
        return Stub(spec.name)

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, Finder())
sys.argv = [name, "--help"]
try:
    if name.endswith(".cli"):
        importlib.import_module(name).main(["--help"])
    else:
        runpy.run_module(name, run_name="__main__", alter_sys=True)
except SystemExit:
    pass
print(json.dumps({"read": sorted(read), "loaded": "sts2_gym.native" in sys.modules}))
"""


class HindsightPinTests(unittest.TestCase):
    def _probe(self, name: str) -> dict:
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "probe.py"
            script.write_text(PROBE, encoding="utf-8")
            env = {k: v for k, v in __import__("os").environ.items() if k not in ("STS2_ENGINE", "STS2_LIB_PATH")}
            r = subprocess.run([sys.executable, "-I", str(script), str(ROOT), name], env=env, capture_output=True, text=True,
                               timeout=180, check=False)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_no_lab_or_search_entry_point_reads_a_hindsight_name(self):
        for name in _pinned_commands() + SEARCH_COMMANDS:
            with self.subTest(command=name):
                doc = self._probe(name)
                self.assertTrue(doc["loaded"], f"{name} did not load sts2_gym.native: the probe would be vacuous")
                self.assertFalse(set(doc["read"]) & set(HINDSIGHT_NAMES), f"{name} read {sorted(set(doc['read']) & set(HINDSIGHT_NAMES))}")

    def test_the_probe_sees_a_hindsight_read(self):
        # The control: a module that reads native.solve_hindsight is caught, so the cases above are not vacuous.
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d) / "hindsight_reader_probe.py"
            pkg.write_text("from sts2_gym import native\nnative.solve_hindsight\n", encoding="utf-8")
            script = Path(d) / "probe.py"
            script.write_text(PROBE.replace('sys.path[:0] = [', f'sys.path[:0] = [{str(d)!r}, '), encoding="utf-8")
            r = subprocess.run([sys.executable, "-I", str(script), str(ROOT), "hindsight_reader_probe"], capture_output=True, text=True,
                               timeout=120, check=False)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("solve_hindsight", json.loads(r.stdout.strip().splitlines()[-1])["read"])

    def test_native_binds_no_hindsight_export_at_import(self):
        spec = importlib.util.spec_from_file_location("solver_binding_for_pins", Path(__file__).with_name("test_solver_binding.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        native, lib = mod.load_native_without_a_library()
        self.assertNotIn("Sts2_SolveHindsight", lib._mock_children)
        self.assertIsNone(native._hindsight_export)

    def test_no_lab_training_or_search_module_names_the_hindsight_bound(self):
        # Only lost-fight attribution may: no such tool is in these trees yet, so the allowed list is empty.
        allowed: set[str] = set()
        roots = [ROOT / "rl", ROOT / "referee", ROOT / "src" / "sts2_gym", ROOT / "tools" / "lab"]
        hits = []
        for top in roots:
            for path in top.rglob("*.py") if top.is_dir() else []:
                rel = path.relative_to(ROOT).as_posix()
                if rel == "src/sts2_gym/native.py" or rel in allowed or "/tests/" in f"/{rel}" or path.name.startswith("test_"):
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                hits += [f"{rel}: {n}" for n in HINDSIGHT_NAMES if n in text]
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
