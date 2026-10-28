"""The hindsight bound stays away from the lab and the search (docs/rebuild/FIGHT_SOLVER.md section 4a, piece S4).

Perfect play is best in expectation at decision time; the hindsight bound (native.solve_hindsight, Sts2_SolveHindsight,
HindsightBound) is for lost-fight attribution only and never a label, a regret or a grade. The gate is the library's: the
native search refuses unless the solver was made with the hindsight flag (allow_hindsight in Python, AllowHindsight in C,
--allow-hindsight on EmuDump's command line). Pinned here, in the manner of test_native_entry_pins.py: each lab and search
entry point is run in a fresh interpreter in which sts2_gym.native is a stub recording every name read from it, and none
may read a hindsight name; the source of every module the entry point loads (the code its calls can reach) names none,
nor the flag, nor EmuDump's hindsight command; and no module under rl/, referee/, src/sts2_gym/ or tools/ (but
tools/emudump, the command itself) does either, in Python or in a shell script.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HINDSIGHT_NAMES = ("solve_hindsight", "Sts2_SolveHindsight", "HindsightBound", "_bind_hindsight", "_HindsightResultC",
                   "allow_hindsight", "allow-hindsight", "AllowHindsight")
# EmuDump's hindsight command as a command line names it: the command word as an argument, or --mode hindsight.
HINDSIGHT_COMMAND = re.compile(r"""["']hindsight["']|--mode[\s=,'"]+hindsight""")


def _hits(text: str) -> list[str]:
    return [n for n in HINDSIGHT_NAMES if n in text] + [m.group(0) for m in HINDSIGHT_COMMAND.finditer(text)]


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
files = sorted({str(Path(m.__file__).resolve()) for m in list(sys.modules.values())
                if getattr(m, "__file__", None) and m.__name__ != "sts2_gym.native"
                and Path(m.__file__).resolve().is_relative_to(root.resolve()) and Path(m.__file__).resolve().is_file()})
print(json.dumps({"read": sorted(read), "loaded": "sts2_gym.native" in sys.modules, "files": files}))
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
                # The call path: every repo module the entry point loaded, whose functions its calls reach.
                self.assertTrue(doc["files"], f"{name} loaded no module of the repo: the call-path scan would be vacuous")
                hits = [f"{Path(f).name}: {h}" for f in doc["files"] for h in _hits(Path(f).read_text(encoding="utf-8", errors="replace"))]
                self.assertEqual(hits, [], f"{name} loads code naming the hindsight bound")

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
        doc = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertIn("solve_hindsight", doc["read"])

    def test_the_call_path_scan_sees_a_hindsight_call_inside_a_function(self):
        # The control of the call-path scan: a call made only when a function runs, never at import, and a shell-out to
        # EmuDump's hindsight command, are both caught in the loaded module's source.
        text = ("def attribute(native, s, h):\n    return native.solve_hindsight(s, h)\n\n"
                "def shell():\n    return ['dotnet', 'EmuDump.dll', 'hindsight', '10', '5', '--allow-hindsight']\n")
        self.assertEqual(sorted(set(_hits(text))), sorted({"solve_hindsight", "allow-hindsight", "'hindsight'"}))
        self.assertTrue(_hits("lab.solver(handle, allow_hindsight=True)"))
        self.assertTrue(_hits('cmd = ["EmuDump", "solve", "--mode", "hindsight"]'))
        self.assertFalse(_hits("the hindsight bound is for attribution only"))

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
        roots = [ROOT / "rl", ROOT / "referee", ROOT / "src" / "sts2_gym", ROOT / "tools"]
        hits, scanned = [], 0
        for top in roots:
            for path in (p for pattern in ("*.py", "*.sh", "*.ps1") for p in top.rglob(pattern)) if top.is_dir() else []:
                rel = path.relative_to(ROOT).as_posix()
                if (rel == "src/sts2_gym/native.py" or rel.startswith("tools/emudump/") or rel in allowed or "/tests/" in f"/{rel}"
                        or path.name.startswith("test_")):
                    continue
                scanned += 1
                hits += [f"{rel}: {h}" for h in _hits(path.read_text(encoding="utf-8", errors="replace"))]
        self.assertGreater(scanned, 50)
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
