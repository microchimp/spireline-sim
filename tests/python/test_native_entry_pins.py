"""The engine option's pins (docs/rebuild/NATIVE_KERNEL.md section 3): the lab's self-play, train and benchmark commands leave the
engine old when the environment they started from says core2, and the pure parts of sts2_gym/native.py that the option touched.

sts2_gym.native reads STS2_ENGINE once, when it loads, so the entry points are run in a fresh interpreter in which native is a stub
that records STS2_ENGINE at the moment the real one would load the library: what is tested is the order of the pin and the imports,
and it needs no build.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import gen0  # noqa: E402

# Self-play, promotion and training commands (the lab's, and the tools run by hand beside them): each loads sts2_gym.native.
PINNED_COMMANDS = [
    "rl.run_train.cli",
    "rl.run_train.evaluate",
    "rl.run_train.smoke",
    "rl.run_train.probe_throughput",
    "rl.run_train.prior_check",
    "rl.train",
    "rl.train_combat",
    "rl.evaluate",
    "rl.vrun_sweep",
    "rl.prior_loop.coach_check",
]

ENTRY_PROBE = r"""
import importlib, importlib.abc, importlib.machinery, json, os, runpy, sys, types
from pathlib import Path
from unittest import mock
root, name = Path(sys.argv[1]), sys.argv[2]
sys.path[:0] = [str(root), str(root / "rl"), str(root / "src"), str(root / "referee")]
loaded = []


class Stub(types.ModuleType):
    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return mock.MagicMock(name=attr)


class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        return importlib.machinery.ModuleSpec(fullname, self) if fullname == "sts2_gym.native" else None

    def create_module(self, spec):
        loaded.append(os.environ.get("STS2_ENGINE"))
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
print(json.dumps({"loaded": loaded, "env": os.environ.get("STS2_ENGINE")}))
"""


class EntryPointPinTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get("STS2_ENGINE")
        os.environ.pop("STS2_ENGINE", None)

    def tearDown(self):
        if self._env is None:
            os.environ.pop("STS2_ENGINE", None)
        else:
            os.environ["STS2_ENGINE"] = self._env

    def test_every_command_leaves_the_engine_old_when_the_parent_says_core2(self):
        env = {k: v for k, v in os.environ.items() if k not in ("STS2_ENGINE", "STS2_LIB_PATH")}
        env["STS2_ENGINE"] = "core2"
        for name in PINNED_COMMANDS:
            with self.subTest(command=name), tempfile.TemporaryDirectory() as d:
                script = Path(d) / "probe.py"
                script.write_text(ENTRY_PROBE, encoding="utf-8")
                r = subprocess.run([sys.executable, "-I", str(script), str(ROOT), name], env=env, capture_output=True, text=True,
                                   timeout=120, check=False)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                doc = json.loads(r.stdout.strip().splitlines()[-1])
                self.assertEqual(doc["loaded"], [None], "native loaded on the inherited engine, or did not load")
                self.assertIsNone(doc["env"])

    def test_the_probe_sees_core2_when_nothing_pins(self):
        # The control: a module that loads native and pins nothing keeps the engine, so the cases above are not vacuous.
        env = {k: v for k, v in os.environ.items() if k not in ("STS2_ENGINE", "STS2_LIB_PATH")}
        env["STS2_ENGINE"] = "core2"
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "probe.py"
            script.write_text(ENTRY_PROBE, encoding="utf-8")
            r = subprocess.run([sys.executable, "-I", str(script), str(ROOT), "rl.combat_mcts"], env=env, capture_output=True,
                               text=True, timeout=120, check=False)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(json.loads(r.stdout.strip().splitlines()[-1])["loaded"], ["core2"])

    def test_the_run_train_main_clears_the_engine_for_the_processes_it_starts(self):
        probe = (
            "import json, os, sys, importlib.abc, importlib.machinery, types\n"
            "from unittest import mock\n"
            "sys.path[:0] = sys.argv[1:4]\n"
            "class Stub(types.ModuleType):\n"
            "    def __getattr__(self, a):\n"
            "        if a.startswith('__'): raise AttributeError(a)\n"
            "        return mock.MagicMock()\n"
            "class F(importlib.abc.MetaPathFinder, importlib.abc.Loader):\n"
            "    def find_spec(self, n, p, t=None): return importlib.machinery.ModuleSpec(n, self) if n == 'sts2_gym.native' else None\n"
            "    def create_module(self, s): return Stub(s.name)\n"
            "    def exec_module(self, m): pass\n"
            "sys.meta_path.insert(0, F())\n"
            "from rl.run_train import cli\n"
            "os.environ['STS2_ENGINE'] = 'core2'  # after the import: main() pins on its own\n"
            "try:\n"
            "    cli.main(['--help'])\n"
            "except SystemExit:\n"
            "    pass\n"
            "print(json.dumps({'env': os.environ.get('STS2_ENGINE')}))\n"
        )
        env = {k: v for k, v in os.environ.items() if k not in ("STS2_ENGINE", "STS2_LIB_PATH")}
        r = subprocess.run([sys.executable, "-I", "-c", probe, str(ROOT), str(ROOT / "rl"), str(ROOT / "src")], env=env,
                           capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIsNone(json.loads(r.stdout.strip().splitlines()[-1])["env"])

    def test_bench_emu_takes_its_engine_from_the_flag_alone(self):
        import bench_emu  # noqa: PLC0415

        seen = []

        def fake_bench(n, seconds, metrics):
            seen.append(os.environ.get("STS2_ENGINE"))  # what the spawned workers inherit
            return {"workers": n}

        with mock.patch.object(bench_emu, "bench", fake_bench), mock.patch("builtins.print") as shown:
            os.environ["STS2_ENGINE"] = "core2"
            bench_emu.main(["--n", "1", "--seconds", "1"])
            bench_emu.main(["--n", "1", "--seconds", "1", "--engine", "core2"])
        self.assertEqual(seen, [None, "core2"])
        self.assertEqual([json.loads(c.args[0])["engine"] for c in shown.call_args_list], ["old", "core2"])

    def test_gen0_bench_emu_passes_the_engine_and_the_other_commands_clear_it(self):
        a = gen0.build_parser().parse_args(["bench-emu", "--out", "O"])
        self.assertEqual(a.engine, "old")
        self.assertFalse(hasattr(gen0.build_parser().parse_args(["bench-env", "--out", "O"]), "engine"))
        cmds = []

        def fake_run(cmd, **kw):
            cmds.append(cmd)
            return SimpleNamespace(returncode=0, stdout="{}")

        out = Path(tempfile.mkdtemp())
        for engine in ("old", "core2"):
            b = gen0.build_parser().parse_args(["bench-emu", "--out", str(out), "--engine", engine])
            with mock.patch("subprocess.run", fake_run):
                gen0.bench_cmd(b)
        self.assertEqual([c[c.index("--engine") + 1] for c in cmds], ["old", "core2"])
        os.environ["STS2_ENGINE"] = "core2"
        with mock.patch.object(gen0, "bench_cmd", return_value=0):
            gen0.main(["bench-emu", "--out", "O"])
        self.assertNotIn("STS2_ENGINE", os.environ)  # the default is old, whatever the environment said
        with mock.patch.object(gen0, "bench_cmd", return_value=0):
            gen0.main(["bench-emu", "--out", "O", "--engine", "core2"])
        self.assertEqual(os.environ.get("STS2_ENGINE"), "core2")


NATIVE_PY = ROOT / "src" / "sts2_gym" / "native.py"


def _load_native_without_a_library():
    """sts2_gym/native.py imported under another name against a stand-in library (a mock that answers the version checks), so its
    pure parts run on a machine with no build."""
    lib = mock.MagicMock()
    lib.Sts2_NativeApiVersion.return_value = 37
    lib.Sts2Run_NativeApiVersion.return_value = 27
    lib.Sts2Run_ObsLayout.return_value = 13  # the layout block's length
    spec = importlib.util.spec_from_file_location("native_under_test", NATIVE_PY)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "Sts2Emulator.dll").write_bytes(b"")
        with mock.patch.dict(os.environ, {"STS2_LIB_PATH": d, "STS2_ALLOW_STALE_NATIVE": "1"}), mock.patch("ctypes.CDLL", return_value=lib):
            os.environ.pop("STS2_ENGINE", None)
            spec.loader.exec_module(mod)
    return mod


class NativePureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = _load_native_without_a_library()

    def test_an_edit_to_an_embedded_data_file_marks_the_library_stale(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "src" / "Sts2Emulator").mkdir(parents=True)
            (root / "tools" / "fidelity").mkdir(parents=True)
            (root / "data").mkdir()
            source = root / "src" / "Sts2Emulator" / "A.cs"
            source.write_text("//", encoding="utf-8")
            lib = root / "Sts2Emulator.dll"
            lib.write_bytes(b"")
            data = [root / n for n in self.native._EMBEDDED_DATA_FILES]
            for f in data:
                f.write_text("{}", encoding="utf-8")
            paths = self.native._native_source_paths(root)
            for f in data:
                self.assertIn(f, paths)
            for f in [source, *data]:
                os.utime(f, (1000, 1000))
            os.utime(lib, (2000, 2000))
            with mock.patch.object(self.native, "_repo_root", return_value=root), mock.patch.dict(os.environ):
                os.environ.pop("STS2_ALLOW_STALE_NATIVE", None)
                self.native._assert_native_library_is_fresh(lib)  # newer than every source and data file
                for f in data:
                    os.utime(f, (3000, 3000))  # an edit to the data file alone
                    with self.assertRaisesRegex(RuntimeError, "is older than"):
                        self.native._assert_native_library_is_fresh(lib)
                    os.utime(f, (1000, 1000))

    def test_a_missing_data_file_is_skipped_not_a_crash(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "src" / "Sts2Emulator").mkdir(parents=True)
            self.assertEqual(self.native._native_source_paths(Path(d)), [])

    def test_the_embedded_file_list_is_the_csproj_s(self):
        csproj = (ROOT / "src" / "Sts2Emulator" / "Sts2Emulator.csproj").read_text(encoding="utf-8")
        listed = [m.replace("\\", "/").removeprefix("../../") for m in re.findall(r'<EmbeddedResource Include="([^"]+)"', csproj)]
        self.assertEqual(sorted(listed), sorted(self.native._EMBEDDED_DATA_FILES))
        for name in self.native._EMBEDDED_DATA_FILES:
            self.assertTrue((ROOT / name).is_file(), name)

    def test_a_kernel_on_clone_refusal_is_not_reported_as_a_bad_handle(self):
        lib = mock.MagicMock()
        with mock.patch.object(self.native, "_lib", lib), \
                mock.patch.object(self.native, "_get_last_error", return_value="kernel_on: Sts2Run_Clone does not carry"):
            lib.Sts2Run_Clone.return_value = -3
            with self.assertRaisesRegex(RuntimeError, r"Sts2Run_Clone refused: kernel_on: Sts2Run_Clone does not carry"):
                self.native.run_clone(1, True, 7, None)
            lib.Sts2Run_Clone.return_value = -1
            with self.assertRaisesRegex(RuntimeError, "unknown handle or the pool is full"):
                self.native.run_clone(1, True, 7, None)
            lib.Sts2Run_Clone.return_value = 5
            self.assertEqual(self.native.run_clone(1, True, 7, None), 5)


if __name__ == "__main__":
    unittest.main()
