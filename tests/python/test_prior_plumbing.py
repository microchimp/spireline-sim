"""The prior server's plumbing (docs/rl/GPU_INFERENCE.md rollout item 2): the Agent's transport, the launcher, the gen0 flags,
`prior_path` in rows, records, parts and manifests, and `build_training_set(exclude_prior_paths=...)`.

The tests of the launcher run against a fake server script. The ones that need the server itself (`rl/prior_server.py`) or the
client (`rl/prior_client.py`) skip until the other worktree's commit is integrated; the ones that need the emulator skip without
the native library."""

import dataclasses
import json
import os
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import jsonschema
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (Path(__file__).resolve().parent, ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import gen0  # noqa: E402
import gen_loop  # noqa: E402
import prior_launch  # noqa: E402
from test_gen_loop import fake_record, fake_run_arrays  # noqa: E402
from test_run_train_shards import synthetic_record  # noqa: E402

from rl.run_train import shards  # noqa: E402

HAVE_NATIVE = bool(os.environ.get("STS2_LIB_PATH") and (Path(os.environ["STS2_LIB_PATH"]) / "Sts2Emulator.dll").exists())
HAVE_CLIENT = (ROOT / "rl" / "prior_client.py").is_file()
HAVE_SERVER = (ROOT / "rl" / "prior_server.py").is_file()
SEEDS = ["SJ9DHBJHPE", "RCQ4PFEAHK"]
FLAT_SIMS = 6

_FIXTURE: dict = {}


def fixture_prior() -> Path:
    """A tiny exported combat prior (tests/python/prior_fixture.py), once per test run."""
    if not _FIXTURE:
        import atexit

        import prior_fixture

        tmp = tempfile.TemporaryDirectory()
        atexit.register(tmp.cleanup)
        _FIXTURE["dir"] = prior_fixture.export_prior(Path(tmp.name) / "gen1")
    return _FIXTURE["dir"]


# ------------------------------------------------------------------ the decision


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(prior_launch.MODE_ENV, None)
        self.py = sys.executable
        self.kill = self.root / "GPU_OFF"

    def test_off_is_off_whatever_else_is_true(self):
        self.assertEqual(prior_launch.resolve_prior_server("off", self.py, self.kill), "off")
        self.assertEqual(prior_launch.resolve_prior_server("off", None, None), "off")

    def test_auto_is_on_with_a_gpu_python_and_no_kill_file(self):
        self.assertEqual(prior_launch.resolve_prior_server("auto", self.py, self.kill), "on")

    def test_auto_is_off_without_a_gpu_python_or_with_the_kill_file(self):
        self.assertEqual(prior_launch.resolve_prior_server("auto", None, self.kill), "off")
        self.assertEqual(prior_launch.resolve_prior_server("auto", str(self.root / "missing.exe"), self.kill), "off")
        self.kill.write_text("x")
        self.assertEqual(prior_launch.resolve_prior_server("auto", self.py, self.kill), "off")

    def test_the_environment_overrides_every_mode_to_off(self):
        os.environ[prior_launch.MODE_ENV] = "off"
        for mode in ("auto", "require"):
            self.assertEqual(prior_launch.resolve_prior_server(mode, self.py, self.kill), "off")

    def test_require_is_on_or_an_error(self):
        self.assertEqual(prior_launch.resolve_prior_server("require", self.py, self.kill), "on")
        with self.assertRaises(prior_launch.PriorServerError):
            prior_launch.resolve_prior_server("require", None, self.kill)
        self.kill.write_text("x")
        with self.assertRaises(prior_launch.PriorServerError):
            prior_launch.resolve_prior_server("require", self.py, self.kill)

    def test_an_unknown_mode_is_a_value_error(self):
        with self.assertRaises(ValueError):
            prior_launch.resolve_prior_server("always", self.py, self.kill)


class FlagTests(unittest.TestCase):
    def test_gen0_flags_default_to_off_on_emu_promote_and_train_but_not_score(self):
        ap = gen0.build_parser()
        emu = ap.parse_args(["emu", "--out", "o"])
        self.assertEqual((emu.prior_server, emu.gpu_python, emu.server_procs, emu.prior_server_args, emu.latency_p99_ms),
                         ("off", None, None, None, 20.0))
        self.assertEqual(prior_launch.ServerOpts.from_args(emu), prior_launch.ServerOpts())
        self.assertEqual(ap.parse_args(["promote", "--cand", "c", "--agent-root", "a"]).prior_server, "off")
        self.assertEqual(ap.parse_args(["train", "--shards-root", "s", "--agent-root", "a"]).prior_server, "off")
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            ap.parse_args(["score", "--game", "g", "--seeds-file", "f", "--out", "o", "--prior-server", "auto"])

    def test_options_round_trip_through_their_flags(self):
        ap = gen0.build_parser()
        a = ap.parse_args(["emu", "--out", "o", "--prior-server", "auto", "--gpu-python", "gpu/python.exe", "--server-procs", "20",
                           "--prior-server-args", "--max-batch 64 --deadline-us 200", "--latency-p99-ms", "9"])
        opts = prior_launch.ServerOpts.from_args(a)
        self.assertEqual((opts.mode, opts.server_procs, opts.extra_args, opts.latency_p99_ms),
                         ("auto", 20, ("--max-batch", "64", "--deadline-us", "200"), 9.0))
        again = ap.parse_args(["emu", "--out", "o", *opts.flags()])
        self.assertEqual(prior_launch.ServerOpts.from_args(again), opts)
        self.assertEqual(prior_launch.ServerOpts().flags(), [])  # off passes nothing to a subprocess

    def test_the_gpu_python_defaults_to_the_environment(self):
        a = gen0.build_parser().parse_args(["emu", "--out", "o", "--prior-server", "auto"])
        with mock.patch.dict(os.environ, {prior_launch.GPU_PYTHON_ENV: "from/env.exe"}):
            self.assertEqual(prior_launch.ServerOpts.from_args(a).gpu_python, "from/env.exe")

    def test_the_promote_subprocess_gets_server_flags_only_when_given(self):
        ran = []
        with mock.patch.object(gen_loop.subprocess, "run", lambda cmd, check=False: ran.append(cmd) or SimpleNamespace(returncode=0)):
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None)
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None,
                                            prior_server_flags=("--prior-server", "auto"))
        self.assertNotIn("--prior-server", ran[0])
        self.assertEqual(ran[1][-2:], ["--prior-server", "auto"])

    def test_exclude_prior_paths_parses_a_comma_list(self):
        self.assertEqual(gen0.exclude_prior_paths("server, mixed"), ("server", "mixed"))
        self.assertEqual(gen0.exclude_prior_paths(None), ())


# ------------------------------------------------------------------ the launcher against a fake server

FAKE_READY = textwrap.dedent('''
    import json, os, sys
    out = os.environ.get("FAKE_OUT")
    if out:
        with open(out, "w") as fh:
            json.dump({"argv": sys.argv[1:], "key": os.environ.get("SPIRELINE_PRIOR_AUTHKEY"),
                       "lib": os.environ.get("STS2_LIB_PATH")}, fh)
    print("fake: warming up", flush=True)
    print("READY " + sys.argv[sys.argv.index("--endpoint") + 1], flush=True)
    sys.stdin.read()
''')
FAKE_NEVER_READY = "import sys\nprint('fake: still loading', flush=True)\nsys.stdin.read()\n"
FAKE_EXITS = "import sys\nprint('fake: no CUDA', flush=True)\nsys.exit(2)\n"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.handles = []

    def tearDown(self):
        for h in self.handles:
            h.stop()

    def script(self, text: str) -> Path:
        p = self.root / "fake_server.py"
        p.write_text(text, encoding="utf-8")
        return p

    def start(self, text: str, **kw):
        h = prior_launch.start_prior_server(sys.executable, self.root / "combat_policy.pt", lib_path="the/lib",
                                            kill_file=self.root / "GPU_OFF", stats_path=self.root / "stats" / "s.json",
                                            server_script=self.script(text), **{"ready_timeout_s": 60, **kw})
        if h is not None:
            self.handles.append(h)
        return h

    def test_start_waits_for_ready_hands_over_the_key_by_environment_only_and_stop_closes_stdin(self):
        out = self.root / "seen.json"
        with mock.patch.dict(os.environ, {"FAKE_OUT": str(out)}):
            h = self.start(FAKE_READY, extra_args=("--device", "cpu"))
        self.assertIsNotNone(h)
        self.assertTrue(h.alive())
        self.assertEqual(len(h.authkey), 32)
        self.assertEqual((h.kind, h.restarts), ("pipe" if os.name == "nt" else "unix", 0))
        if os.name == "nt":
            self.assertTrue(h.endpoint.startswith("\\\\.\\pipe\\spireline-prior-" + str(os.getpid()) + "-"))
        seen = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(seen["key"], h.authkey.hex())
        self.assertNotIn(h.authkey.hex(), " ".join(seen["argv"]))  # never on a command line
        self.assertEqual(seen["lib"], "the/lib")
        argv = seen["argv"]
        self.assertEqual(argv[argv.index("--endpoint") + 1], h.endpoint)
        self.assertEqual(argv[argv.index("--authkey-env") + 1], "SPIRELINE_PRIOR_AUTHKEY")
        self.assertEqual(argv[argv.index("--parent-pid") + 1], str(os.getpid()))
        self.assertEqual(argv[argv.index("--kill-file") + 1], str(self.root / "GPU_OFF"))
        self.assertEqual(argv[argv.index("--stats") + 1], str(self.root / "stats" / "s.json"))
        self.assertEqual(argv[-2:], ["--device", "cpu"])  # extra arguments come last, so they win
        t0 = time.monotonic()
        h.stop()
        self.assertFalse(h.alive())
        self.assertEqual(h.proc.returncode, 0)  # it left on stdin EOF, not on terminate
        self.assertLess(time.monotonic() - t0, 4.0)
        h.stop()  # twice is fine
        self.assertNotIn("authkey", repr(h).lower())

    def test_each_server_gets_its_own_key_and_endpoint(self):
        a, b = self.start(FAKE_READY), self.start(FAKE_READY)
        self.assertNotEqual(a.authkey, b.authkey)
        self.assertNotEqual(a.endpoint, b.endpoint)

    def test_no_ready_line_is_none_after_the_timeout_and_the_child_is_gone(self):
        t0 = time.monotonic()
        self.assertIsNone(self.start(FAKE_NEVER_READY, ready_timeout_s=1.5))
        self.assertLess(time.monotonic() - t0, 15.0)

    def test_a_server_that_exits_before_ready_is_none_at_once(self):
        t0 = time.monotonic()
        self.assertIsNone(self.start(FAKE_EXITS, ready_timeout_s=60))
        self.assertLess(time.monotonic() - t0, 30.0)

    def test_a_missing_interpreter_is_none_not_an_exception(self):
        h = prior_launch.start_prior_server(self.root / "no-such-python.exe", "p.pt", lib_path=None, kill_file=self.root / "k",
                                            stats_path=self.root / "s.json", ready_timeout_s=5)
        self.assertIsNone(h)

    def test_restart_once_keeps_the_endpoint_and_key_and_only_once(self):
        h = self.start(FAKE_READY)
        endpoint, key, first = h.endpoint, h.authkey, h.proc
        first.terminate()
        first.wait(timeout=10)
        self.assertFalse(h.alive())
        self.assertTrue(h.restart_once())
        self.assertTrue(h.alive())
        self.assertIsNot(h.proc, first)
        self.assertEqual((h.endpoint, h.authkey, h.restarts), (endpoint, key, 1))
        h.proc.terminate()
        h.proc.wait(timeout=10)
        self.assertFalse(h.restart_once())  # the budget is one
        self.assertFalse(h.alive())
        self.assertEqual(h.restarts, 1)

    def test_read_stats_is_the_servers_file_or_empty(self):
        h = self.start(FAKE_READY)
        self.assertEqual(h.read_stats(), {})
        h.stats_path.write_text(json.dumps({"batches": 3}), encoding="utf-8")
        self.assertEqual(h.read_stats(), {"batches": 3})


# ------------------------------------------------------------------ the agent's transport


class _StubClient:
    """What `search_extra` needs of rl/prior_client.py, for the builds that do not have it yet."""

    @dataclasses.dataclass(frozen=True)
    class RemotePrior:
        policy: str
        probe: bool
        endpoint: str
        authkey: bytes
        policy_sha256: str
        timeout_s: float = 5.0


@unittest.skipUnless(HAVE_NATIVE, "needs the native library")
class AgentTransportTests(unittest.TestCase):
    def test_describe_and_repr_ignore_the_transport(self):
        d = fixture_prior()
        plain = gen_loop.Agent(prior_dir=str(d))
        served = dataclasses.replace(plain, prior_server="endpoint-name", prior_authkey=b"secret-key")
        self.assertEqual(plain.describe(), served.describe())
        self.assertNotIn("secret", repr(served))

    def test_the_default_agent_keeps_the_in_process_prior(self):
        import combat_prior

        extra = gen_loop.Agent(prior_dir=str(fixture_prior())).search_extra()
        self.assertIsInstance(extra["prior_fn"], combat_prior.LivePrior)

    def test_a_server_without_a_key_is_refused(self):
        with self.assertRaises(ValueError):
            gen_loop.Agent(prior_dir=str(fixture_prior()), prior_server="x").search_extra()

    def test_a_server_replaces_the_prior_fn_and_keeps_the_gate_settings(self):
        d = fixture_prior()
        base = gen_loop.Agent(prior_dir=str(d)).search_extra()
        with mock.patch.dict(sys.modules, {"prior_client": _StubClient}):
            extra = gen_loop.Agent(prior_dir=str(d), prior_server="ep", prior_authkey=b"k").search_extra()
        rp = extra["prior_fn"]
        self.assertIsInstance(rp, _StubClient.RemotePrior)
        self.assertEqual((rp.endpoint, rp.authkey, rp.probe, rp.policy), ("ep", b"k", base["prior_fn"].probe, base["prior_fn"].policy))
        self.assertEqual(rp.policy_sha256, json.loads((d / "model.json").read_text())["policy_sha256"])
        self.assertEqual({k: v for k, v in extra.items() if k != "prior_fn"}, {k: v for k, v in base.items() if k != "prior_fn"})

    def test_a_server_agent_without_a_prior_has_nothing_to_serve(self):
        self.assertEqual(gen_loop.Agent(prior_server="ep", prior_authkey=b"k").search_extra(), {})

    @unittest.skipUnless(HAVE_CLIENT, "rl/prior_client.py is not in this worktree yet (integration)")
    def test_the_real_remote_prior_carries_the_agents_fields(self):
        import prior_client

        d = fixture_prior()
        rp = gen_loop.Agent(prior_dir=str(d), prior_server="ep", prior_authkey=b"k").search_extra()["prior_fn"]
        self.assertIsInstance(rp, prior_client.RemotePrior)
        self.assertEqual((rp.endpoint, rp.authkey, rp.policy), ("ep", b"k", str(d / "combat_policy.pt")))

    def test_the_lockstep_and_the_score_never_receive_a_remote_prior(self):
        import combat_prior
        import search_configs

        d = fixture_prior()
        with mock.patch.dict(os.environ, {gen_loop.ENV_GEN_DIR: str(d)}):
            with mock.patch.dict(sys.modules, {"prior_client": _StubClient}):  # even with a client importable
                self.assertIsInstance(gen_loop.active_agent_extra()["prior_fn"], combat_prior.LivePrior)
                self.assertIsInstance(search_configs.get("a3-v1").search_kwargs(env_prior=True)["prior_fn"], combat_prior.LivePrior)
            self.assertNotIn("prior_fn", search_configs.get("a3-v1").search_kwargs(env_prior=False))
        # the guard itself: even an agent that arrives with a transport is stripped of it on this path
        served = gen_loop.Agent(prior_dir=str(d), prior_server="ep", prior_authkey=b"k")
        with mock.patch.object(gen_loop, "load_agent", return_value=served),                 mock.patch.dict(os.environ, {gen_loop.ENV_GEN_DIR: str(d)}),                 mock.patch.dict(sys.modules, {"prior_client": _StubClient}):
            self.assertIsInstance(gen_loop.active_agent_extra()["prior_fn"], combat_prior.LivePrior)


# ------------------------------------------------------------------ prior_path in rows, records, parts and manifests


class PriorPathOfTests(unittest.TestCase):
    A = SimpleNamespace(prior_dir="d")

    def counts(self, server, cpu, reason=None):
        return {"server": server, "cpu": cpu, "fallback_reason": reason, "latency_ms": {"p50": 1.0, "p99": 2.0, "n": server + cpu}}

    def test_no_prior_is_none_and_the_in_process_prior_is_cpu(self):
        self.assertEqual(gen0.prior_path_of(SimpleNamespace(prior_dir=None), None, None), ("none", {"server": 0, "cpu": 0}, None))
        self.assertEqual(gen0.prior_path_of(self.A, None, None), ("cpu", {"server": 0, "cpu": None}, None))

    def test_the_counter_delta_decides_server_cpu_or_mixed(self):
        before = self.counts(10, 4)
        self.assertEqual(gen0.prior_path_of(self.A, before, self.counts(25, 4))[:2], ("server", {"server": 15, "cpu": 0}))
        self.assertEqual(gen0.prior_path_of(self.A, before, self.counts(10, 9, "eof"))[:2], ("cpu", {"server": 0, "cpu": 5}))
        self.assertEqual(gen0.prior_path_of(self.A, before, self.counts(12, 9, "eof"))[:2], ("mixed", {"server": 2, "cpu": 5}))

    def test_any_uniform_call_makes_the_run_uniform_and_the_counts_show_the_split(self):
        def counts(server, cpu, uniform, reason=None):
            return {**self.counts(server, cpu, reason), "uniform": uniform}

        before = counts(10, 4, 0)
        self.assertEqual(gen0.prior_path_of(self.A, before, counts(12, 4, 5, "memory floor"))[:2],
                         ("uniform", {"server": 2, "cpu": 0, "uniform": 5}))  # served, then fell under the floor: still uniform
        self.assertEqual(gen0.prior_path_of(self.A, before, counts(10, 4, 3, "memory floor"))[:2],
                         ("uniform", {"server": 0, "cpu": 0, "uniform": 3}))
        self.assertEqual(gen0.prior_path_of(self.A, counts(0, 0, 3, "x"), counts(0, 0, 3, "x"))[0], "uniform")  # no calls: the state
        self.assertEqual(gen0.prior_path_of(self.A, counts(0, 0, 0), counts(5, 0, 0))[:2], ("server", {"server": 5, "cpu": 0}))

    def test_a_run_with_no_calls_follows_the_process_fallback_state(self):
        self.assertEqual(gen0.prior_path_of(self.A, self.counts(3, 0), self.counts(3, 0))[0], "server")
        self.assertEqual(gen0.prior_path_of(self.A, self.counts(0, 3, "x"), self.counts(0, 3, "x"))[0], "cpu")

    def test_the_latency_carries_the_processs_pid(self):
        lat = gen0.prior_path_of(self.A, self.counts(0, 0), self.counts(2, 0))[2]
        self.assertEqual((lat["p50"], lat["p99"], lat["n"], lat["pid"]), (1.0, 2.0, 2, os.getpid()))

    def test_block_latency_takes_each_processs_fullest_reservoir(self):
        rows = [{"prior_latency_ms": {"p50": 1.0, "p99": 10.0, "n": 100, "pid": 1}},
                {"prior_latency_ms": {"p50": 2.0, "p99": 12.0, "n": 300, "pid": 1}},  # the same process, later
                {"prior_latency_ms": {"p50": 4.0, "p99": 30.0, "n": 100, "pid": 2}}, {"seed": "crash"}]
        lat = gen0.aggregate_latency(rows)
        self.assertEqual((lat["n"], lat["processes"]), (400, 2))
        self.assertAlmostEqual(lat["p50"], (2.0 * 300 + 4.0 * 100) / 400)
        self.assertAlmostEqual(lat["p99"], (12.0 * 300 + 30.0 * 100) / 400)
        self.assertEqual(lat["p99_max"], 30.0)
        self.assertEqual(gen0.aggregate_latency([{}])["n"], 0)


class SchemaTests(unittest.TestCase):
    def test_the_schema_takes_the_new_fields_and_still_takes_old_records_and_refuses_unknown_keys(self):
        validator = jsonschema.Draft202012Validator(shards.RUN_RECORD_SCHEMA)
        validator.validate(synthetic_record())  # a record from before the fields
        for path in ("none", "cpu", "server", "mixed"):
            validator.validate(synthetic_record(prior_path=path, prior_calls={"server": 3, "cpu": 0}))
        validator.validate(synthetic_record(prior_path="uniform", prior_calls={"server": 2, "cpu": 0, "uniform": 7}))
        self.assertFalse(validator.is_valid(synthetic_record(prior_path="uniform", prior_calls={"server": 2, "cpu": 0, "uniform": -1})))
        validator.validate(synthetic_record(prior_path="cpu", prior_calls={"server": 0, "cpu": None}))
        self.assertFalse(validator.is_valid(synthetic_record(prior_path="gpu")))
        self.assertFalse(validator.is_valid(synthetic_record(prior_calls={"server": 1})))
        self.assertFalse(validator.is_valid(synthetic_record(prior_calls={"server": -1, "cpu": 0})))
        self.assertFalse(validator.is_valid(synthetic_record(surprise=1)))


@unittest.skipUnless(HAVE_NATIVE, "needs the native library")
class RecorderFieldTests(unittest.TestCase):
    def test_finish_writes_the_prior_fields_only_when_given(self):
        rec = gen_loop.RunRecorder("AAAAAAAAAA", 0, {}, 30, 0.2, 10)
        sess = SimpleNamespace(info={"floor": 3, "hp": 5, "max_hp": 80}, stats=SimpleNamespace(
            unknown_decisions={}, forced_skips={}, potion_left_full_belt=0), budget="30", acts_cleared=0, died=True, steps=7)
        plain = rec.finish(sess, 4, won=False)
        self.assertNotIn("prior_path", plain)
        self.assertNotIn("prior_calls", plain)
        full = rec.finish(sess, 4, won=False, prior_path="server", prior_calls={"server": 9, "cpu": 0})
        self.assertEqual((full["prior_path"], full["prior_calls"]), ("server", {"server": 9, "cpu": 0}))


class PartTests(unittest.TestCase):
    def test_a_part_round_trips_the_prior_path_codes_and_an_old_part_has_none(self):
        rng = np.random.default_rng(1)
        runs = [("AAAAAAAAAA", fake_run_arrays(rng)), ("BBBBBBBBBB", None), ("CCCCCCCCCC", fake_run_arrays(rng, 2, 3)),
                ("DDDDDDDDDD", None)]
        with tempfile.TemporaryDirectory() as d:
            gen_loop.write_part(Path(d) / "new.npz", runs, 2, ["server", "none", "mixed", None])
            gen_loop.write_part(Path(d) / "old.npz", runs, 2)
            new, old = gen_loop.read_part(Path(d) / "new.npz"), gen_loop.read_part(Path(d) / "old.npz")
        self.assertEqual(new["prior_path"].dtype, np.int8)
        self.assertEqual(new["prior_path"].tolist(), [2, 0, 3, -1])  # aligned with `seeds`, -1 for a run without the field
        self.assertEqual(new["seeds"].shape, new["prior_path"].shape)
        self.assertNotIn("prior_path", old)
        np.testing.assert_array_equal(new["visits"], old["visits"])
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            gen_loop.write_part(Path(d) / "x.npz", runs, 2, ["cpu"])

    def test_a_part_round_trips_the_uniform_code(self):
        rng = np.random.default_rng(3)
        runs = [("AAAAAAAAAA", fake_run_arrays(rng)), ("BBBBBBBBBB", fake_run_arrays(rng, 2, 3))]
        self.assertEqual(gen_loop.PRIOR_PATH_CODES["uniform"], 4)
        self.assertIn("uniform", gen_loop.PRIOR_PATHS)
        with tempfile.TemporaryDirectory() as d:
            gen_loop.write_part(Path(d) / "u.npz", runs, 2, ["uniform", "server"])
            part = gen_loop.read_part(Path(d) / "u.npz")
            sink = gen_loop.ShardSink(Path(d) / "job", 0, config="a3-v1", base_sims=30, ascension=10, agent={}, part_runs=2)
            for i, tag in enumerate(["uniform", "uniform"]):
                sink.add({**fake_record(f"U{i:09d}", 0), "prior_path": tag}, fake_run_arrays(rng))
            man = sink.close()
        self.assertEqual(part["prior_path"].tolist(), [4, 2])
        self.assertEqual(man["prior_paths"], {"uniform": 2})

    def test_the_sink_counts_runs_by_path_in_the_manifest(self):
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as d:
            sink = gen_loop.ShardSink(Path(d), 0, config="a3-v1", base_sims=30, ascension=10, agent={}, part_runs=2)
            for i, tag in enumerate(["server", "server", "cpu", None]):
                sink.add({**fake_record(f"Q{i:09d}", 0), **({"prior_path": tag} if tag else {})}, fake_run_arrays(rng))
            man = sink.close()
            parts = [gen_loop.read_part(Path(d) / "shards" / n) for n in man["combat_parts"]]
        self.assertEqual(man["prior_paths"], {"cpu": 1, "server": 2, "unknown": 1})
        self.assertEqual([p["prior_path"].tolist() for p in parts], [[2, 2], [1, -1]])


class ExclusionTests(unittest.TestCase):
    SIZES = [(3, 2), (2, 4), (1, 5)]  # fights x decisions: 6, 8 and 5 samples

    def build(self, root: Path, tags, part_runs=2):
        rng = np.random.default_rng(8)
        sink = gen_loop.ShardSink(root / "job", 0, config="a3-v1", base_sims=30, ascension=10, agent={}, part_runs=part_runs)
        for i, ((nf, nd), tag) in enumerate(zip(self.SIZES, tags, strict=True)):
            sink.add({**fake_record(f"Q{i:09d}", 0), "prior_path": tag}, fake_run_arrays(rng, nf, nd))
        sink.close()
        return gen_loop.window_manifests(root, 0)

    def test_excluded_paths_drop_whole_runs_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as d:
            mans = self.build(Path(d), ["server", "cpu", "mixed"])
            everything = gen_loop.build_training_set(mans, max_fights=1000)
            data = gen_loop.build_training_set(mans, max_fights=1000, exclude_prior_paths=("server", "mixed"))
            nothing = gen_loop.build_training_set(mans, max_fights=1000, exclude_prior_paths=("none",))
        self.assertEqual(everything["lengths"].size, 19)
        self.assertEqual(int(everything["_samples_excluded"][0]), 0)
        self.assertEqual(nothing["lengths"].size, 19)
        self.assertEqual(data["lengths"].size, 8)  # the cpu run's samples
        self.assertEqual(int(data["_samples_excluded"][0]), 11)
        self.assertEqual(len(np.unique(data["fights"])), 1)
        self.assertEqual(data["types"].size, int(data["lengths"].sum()))  # the tokens followed their rows
        self.assertEqual(data["qvals"].shape[0], 8)
        self.assertEqual(data["_samples_by_generation"].tolist(), [[0, 8]])
        keep = np.concatenate([np.full(n, i) for i, n in enumerate((6, 8, 5))]) == 1
        np.testing.assert_array_equal(data["visits"], everything["visits"][keep])
        np.testing.assert_array_equal(data["qvals"], everything["qvals"][keep])

    def test_uniform_runs_can_be_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            mans = self.build(Path(d), ["uniform", "server", "uniform"])
            everything = gen_loop.build_training_set(mans, max_fights=1000)
            data = gen_loop.build_training_set(mans, max_fights=1000, exclude_prior_paths=("uniform",))
        self.assertEqual(everything["lengths"].size, 19)
        self.assertEqual(data["lengths"].size, 8)  # the server run's samples
        self.assertEqual(int(data["_samples_excluded"][0]), 11)

    def test_a_part_from_before_the_field_is_never_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            mans = self.build(Path(d), ["server", "server", "server"])
            for man_dir, man in mans:
                for name in man["combat_parts"]:
                    blob = gen_loop.read_part(man_dir / name)
                    blob.pop("prior_path")
                    np.savez_compressed(man_dir / name, **blob)
            data = gen_loop.build_training_set(mans, max_fights=1000, exclude_prior_paths=("server", "cpu", "mixed", "none", "uniform"))
        self.assertEqual(data["lengths"].size, 19)

    def test_dropping_every_run_or_naming_an_unknown_path_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            mans = self.build(Path(d), ["server", "server", "server"])
            with self.assertRaises(ValueError):
                gen_loop.build_training_set(mans, max_fights=1000, exclude_prior_paths=("server",))
            with self.assertRaises(ValueError):
                gen_loop.build_training_set(mans, max_fights=1000, exclude_prior_paths=("gpu",))

    def test_train_generation_passes_the_exclusion_through(self):
        seen = {}

        def fake_build(mans, **kw):
            seen.update(kw)
            raise RuntimeError("stop")

        with tempfile.TemporaryDirectory() as d, mock.patch.object(gen_loop, "build_training_set", fake_build):
            self.build(Path(d) / "runs", ["cpu", "cpu", "cpu"])
            with self.assertRaises(RuntimeError):
                gen_loop.train_generation(0, Path(d) / "runs", Path(d) / "agent", exclude_prior_paths=("mixed",))
        self.assertEqual(seen["exclude_prior_paths"], ("mixed",))


# ------------------------------------------------------------------ blocks (emulator)


class FakeHandle:
    """A started server as `run_block` sees it: it dies once after a few polls, then restarts once."""

    kind = "unix"
    endpoint = "fake-endpoint"
    authkey = b"fake-key"

    def __init__(self, die_after: int = 2):
        self.polls, self.die_after, self.restarts, self.stopped = 0, die_after, 0, 0

    def alive(self) -> bool:
        self.polls += 1
        return self.restarts > 0 or self.polls <= self.die_after

    def restart_once(self) -> bool:
        if self.restarts:
            return False
        self.restarts += 1
        return True

    def stop(self) -> None:
        self.stopped += 1

    def read_stats(self) -> dict:
        return {"batches": 7}


class DeadTwiceHandle(FakeHandle):
    """A server that dies and cannot be restarted: the second death."""

    def restart_once(self) -> bool:
        return False


def stable(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in ("wall", "prior_latency_ms")}


def read_rows(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / "runs.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


@unittest.skipUnless(HAVE_NATIVE, "needs the native library")
class BlockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_prior_server_off_plays_the_default_path_unchanged(self):
        agent = gen_loop.load_agent(fixture_prior())
        base = gen0.run_block(SEEDS, FLAT_SIMS, 2, self.root / "base", agent=agent)
        (self.root / "seeds.json").write_text(json.dumps({"seeds": SEEDS}), encoding="utf-8")
        off = gen0.main(["emu", "--seeds-file", str(self.root / "seeds.json"), "--sims", str(FLAT_SIMS), "--procs", "2",
                         "--out", str(self.root / "off"), "--gen-dir", str(fixture_prior()), "--prior-server", "off"])
        self.assertEqual(off, 0)
        a, b = read_rows(self.root / "base"), read_rows(self.root / "off")
        self.assertEqual(sorted(map(json.dumps, map(stable, a))), sorted(map(json.dumps, map(stable, b))))
        self.assertEqual({r["prior_path"] for r in a}, {"cpu"})
        self.assertEqual(base["prior_paths"], {"cpu": 2})
        self.assertNotIn("prior_server", base)
        self.assertNotIn("prior_server_degraded", base)

    def test_a_block_with_a_handle_restarts_a_dead_server_once_stops_it_and_reports_it(self):
        h = FakeHandle(die_after=1)
        summ = gen0.run_block(SEEDS[:1], FLAT_SIMS, 1, self.root / "fake", prior_server=h, server_procs=2, latency_p99_ms=5.0)
        self.assertEqual((h.restarts, h.stopped), (1, 1))
        self.assertEqual(summ["prior_server"], {"kind": "unix", "restarts": 1, "stats": {"batches": 7}})
        self.assertEqual(summ["procs"], 2)  # server_procs, not procs
        self.assertNotIn("fake-endpoint", json.dumps(summ))  # never the endpoint name or the key
        self.assertNotIn("fake-key", json.dumps(summ))
        self.assertEqual(summ["prior_paths"], {"none": 1})
        self.assertTrue(summ["prior_server_degraded"])  # no run was served

    def shrunk_block(self, server_procs: int, name: str):
        """The two-seed block on a prior that points at a server that is not there and is dead twice, with the headroom faked
        to 1 GiB: the workers in flight fall back to a uniform prior (no torch), the workers of the shrunk pool run the CPU prior."""
        h = DeadTwiceHandle(die_after=1)
        agent = gen_loop.load_agent(fixture_prior())
        with mock.patch.dict(os.environ, {"SPIRELINE_FAKE_MEM_HEADROOM_GB": "1"}):  # the spawn workers inherit it
            summ = gen0.run_block(SEEDS, FLAT_SIMS, 2, self.root / name, agent=agent, prior_server=h, server_procs=server_procs)
        return h, summ, read_rows(self.root / name)

    def test_a_server_dead_twice_drains_and_shrinks_the_pool_to_procs(self):
        h, summ, rows = self.shrunk_block(3, "shrunk3")  # both seeds are in flight when the second death is seen
        self.assertEqual(sorted(r["seed"] for r in rows), sorted(SEEDS))  # every seed once
        self.assertEqual(sorted(r["ix"] for r in rows), [0, 1])
        self.assertEqual(h.stopped, 1)
        shrunk = summ["pool_shrunk"]
        self.assertEqual((shrunk["from"], shrunk["to"]), (3, 2))
        self.assertEqual(set(shrunk), {"from", "to", "at_run", "drained_s"})
        self.assertTrue(0 <= shrunk["at_run"] <= len(rows) and shrunk["drained_s"] >= 0)
        self.assertEqual(summ["procs"], 3)  # the width it started with
        self.assertEqual(summ["runs"], 2)
        self.assertEqual(summ["prior_paths"], {"uniform": 2})  # the drained runs fell back under the floor
        self.assertEqual(sorted(r["prior_path"] for r in rows), ["uniform", "uniform"])
        self.assertTrue(all(r["prior_calls"]["uniform"] > 0 and r["prior_calls"]["server"] == 0 for r in rows))
        self.assertTrue(summ["prior_server_degraded"])
        self.assertNotIn("fake-endpoint", json.dumps(summ))

    def test_the_seed_stream_goes_on_after_the_shrink_on_plain_cpu_path_workers(self):
        _, summ, rows = self.shrunk_block(1, "shrunk1")  # one in flight; the second seed starts after the shrink
        self.assertEqual([r["seed"] for r in rows], SEEDS)  # in stream order, none replayed, none skipped
        self.assertEqual([r["ix"] for r in rows], [0, 1])
        self.assertEqual(summ["pool_shrunk"]["from"], 1)
        self.assertEqual(summ["pool_shrunk"]["to"], 2)
        self.assertEqual(summ["pool_shrunk"]["at_run"], 0)
        self.assertEqual([r["prior_path"] for r in rows], ["uniform", "cpu"])  # the new pool's agent has no server
        self.assertEqual(summ["prior_paths"], {"cpu": 1, "uniform": 1})
        self.assertEqual(rows[1]["prior_calls"], {"server": 0, "cpu": None})
        self.assertTrue(summ["prior_server_degraded"])

    @unittest.skipUnless(HAVE_SERVER and HAVE_CLIENT, "rl/prior_server.py and rl/prior_client.py are not in this worktree yet (integration)")
    def test_a_block_through_a_cpu_device_server_reproduces_the_cpu_path_and_tags_server(self):
        agent = gen_loop.load_agent(fixture_prior())
        cpu = gen0.run_block(SEEDS, FLAT_SIMS, 2, self.root / "cpu", agent=agent)
        opts = prior_launch.ServerOpts(mode="require", gpu_python=sys.executable, server_procs=2,
                                       extra_args=("--device", "cpu", "--max-batch", "8"))
        srv = gen0.run_block_served(SEEDS, FLAT_SIMS, 2, self.root / "srv", opts=opts, agent=agent)
        keys = ("seed", "win", "floor", "end", "decisions", "combat_decisions", "sims")
        a = {r["seed"]: {k: r[k] for k in keys} for r in read_rows(self.root / "cpu")}
        b = {r["seed"]: {k: r[k] for k in keys} for r in read_rows(self.root / "srv")}
        self.assertEqual(a, b)
        self.assertEqual({r["prior_path"] for r in read_rows(self.root / "srv")}, {"server"})
        self.assertEqual(cpu["prior_paths"], {"cpu": 2})
        self.assertEqual(srv["prior_paths"], {"server": 2})
        self.assertEqual(srv["prior_server"]["kind"], "pipe" if os.name == "nt" else "unix")
        self.assertEqual(srv["prior_server"]["restarts"], 0)
        self.assertGreater(srv["prior_latency_ms"]["n"], 0)

    def test_require_that_cannot_start_raises_and_auto_plays_the_cpu_path(self):
        agent = gen_loop.load_agent(fixture_prior())
        with mock.patch.object(prior_launch, "start_prior_server", lambda *a, **k: None):
            with mock.patch.object(prior_launch, "resolve_prior_server", lambda *a: "on"):
                with self.assertRaises(prior_launch.PriorServerError):
                    gen0.run_block_served(SEEDS[:1], FLAT_SIMS, 1, self.root / "req", agent=agent,
                                          opts=prior_launch.ServerOpts(mode="require", gpu_python=sys.executable))
                summ = gen0.run_block_served(SEEDS[:1], FLAT_SIMS, 1, self.root / "auto", agent=agent,
                                             opts=prior_launch.ServerOpts(mode="auto", gpu_python=sys.executable))
        self.assertEqual(summ["prior_paths"], {"cpu": 1})
        self.assertNotIn("prior_server", summ)


if __name__ == "__main__":
    unittest.main()
