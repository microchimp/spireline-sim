"""The batched prior server, its client and their wire format (docs/rl/GPU_INFERENCE.md), on the CPU device.

The server runs as a subprocess of the test's own interpreter with `--device cpu`. A batch of one is bit-exact with
`combat_prior.LivePrior`; a padded bucket is within 1e-6. The GPU class at the end runs only when this module is run
by an interpreter with CUDA (the GPU venv's python) and pins the logit tolerance of section 5 on the fixture prior.
"""

import contextlib
import hashlib
import io
import json
import os
import pickle  # noqa: S403 -- our own objects, as they travel to a worker
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from multiprocessing import connection
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "rl", ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import combat_prior  # noqa: E402
import prior_client as pc  # noqa: E402
import prior_protocol as pp  # noqa: E402
import torch  # noqa: E402
from combat_tokens import MAX_ACTIONS, N_FEATURES, encode  # noqa: E402
from prior_fixture import export_prior  # noqa: E402
from test_combat_prior import DECK, TOADPOLES, toadpoles  # noqa: E402

from sts2_gym import procutil  # noqa: E402

SERVER = ROOT / "rl" / "prior_server.py"
IS_WINDOWS = sys.platform == "win32"
VOCAB_SHA = combat_prior.vocab_sha256()
CARD_SHA = pp.card_table_sha256()
GPU_PYTHON = Path(os.environ.get("SPIRELINE_GPU_PYTHON") or ROOT.parents[1] / "venv-gpu" / "Scripts" / "python.exe")


class ServerProc:
    """A prior server subprocess on a CPU device, with its stderr kept in a file."""

    def __init__(self, policy, *, extra=(), env=None, stdin=subprocess.PIPE, wrapper=None):
        self.policy = Path(policy)
        self.sha = hashlib.sha256(self.policy.read_bytes()).hexdigest()
        self.key = secrets.token_hex(32)  # hex digits in the environment, as the launcher writes it
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.endpoint = (
            f"spireline-test-{os.getpid()}-{secrets.token_hex(4)}" if IS_WINDOWS else str(tmp / "prior.sock")
        )
        self.stats_path = tmp / "stats.json"
        self.kill_file = tmp / "KILL"
        self.err = open(tmp / "stderr.txt", "wb")  # noqa: SIM115
        cmd = [
            sys.executable, str(SERVER), "--policy", str(self.policy), "--endpoint", self.endpoint,
            "--authkey-env", "SPR_TEST_KEY", "--device", "cpu", "--stats", str(self.stats_path),
            "--kill-file", str(self.kill_file), *extra,
        ]
        if wrapper is not None:
            cmd = [sys.executable, "-c", wrapper, *cmd]
        self.proc = subprocess.Popen(  # noqa: S603
            cmd, stdin=stdin, stdout=subprocess.PIPE, stderr=self.err,
            env=dict(os.environ, SPR_TEST_KEY=self.key, **(env or {})),
        )
        watchdog = threading.Timer(180, self.proc.kill)
        watchdog.start()
        try:
            self.ready = self.proc.stdout.readline().decode().strip()
        finally:
            watchdog.cancel()

    def stderr(self) -> str:
        self.err.flush()
        return (Path(self.tmp.name) / "stderr.txt").read_text(errors="replace")

    def prior(self, **overrides):
        fields = dict(policy=str(self.policy), probe=False, endpoint=self.endpoint,
                      authkey=bytes.fromhex(self.key), policy_sha256=self.sha, timeout_s=10.0)
        return pc.RemotePrior(**{**fields, **overrides})

    def raw(self, *, key=None, policy_sha=None, vocab=VOCAB_SHA, card=CARD_SHA, version=pp.PROTO):
        family = "AF_PIPE" if IS_WINDOWS else "AF_UNIX"
        conn = connection.Client(pp.pipe_address(self.endpoint), family=family, authkey=bytes.fromhex(key or self.key))
        conn.send_bytes(pp.pack_hello(policy_sha or self.sha, vocab, card, version))
        status, reason = pp.unpack_hello_reply(conn.recv_bytes())
        return conn, status, reason

    def stop(self, timeout=15):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except (OSError, AttributeError):
                pass
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc.stdout.close()
        self.err.close()
        return self.proc.returncode

    def close(self):
        self.stop()
        self.tmp.cleanup()


def tokens_and_legal(seed):
    combat = toadpoles(seed)
    try:
        return encode(combat.handle, combat.obs, TOADPOLES, probe=False), combat.legal()
    finally:
        combat.close()


def old_weights(row, legal):
    """The per-row code `PolicyNet.weights` held before `legal_weights` was lifted out of it."""
    index = np.asarray(legal, dtype=np.int64)
    scores = np.full(len(index), -np.inf)
    inside = index < row.shape[0]
    scores[inside] = row[index[inside]]
    finite = np.isfinite(scores)
    if not finite.any():
        return np.full(len(index), 1.0 / max(1, len(index)))
    scores = np.where(finite, scores, -np.inf)
    weights = np.exp(scores - scores[finite].max())
    return weights / weights.sum()


class ProtocolTests(unittest.TestCase):
    def test_the_constants_match_the_token_module(self):
        import combat_tokens as ct

        self.assertEqual((pp.MAX_ACTIONS, pp.N_FEATURES), (ct.MAX_ACTIONS, ct.N_FEATURES))
        self.assertEqual(
            (pc._T_GLOBAL, pc._T_HAND, pc._T_DRAW, pc._T_DISCARD, pc._T_EXHAUST, pc._T_ENEMY, pc._T_BUFF),  # noqa: SLF001
            (ct.TYPE_GLOBAL, ct.TYPE_HAND, ct.TYPE_DRAW, ct.TYPE_DISCARD, ct.TYPE_EXHAUST, ct.TYPE_ENEMY, ct.TYPE_BUFF),
        )
        self.assertEqual(
            (pc._T_ENEMY_BUFF, pc._T_END_TURN, pc._T_RELIC, pc._T_INTENT, pc._T_INTENT_NEXT),  # noqa: SLF001
            (ct.TYPE_ENEMY_BUFF, ct.TYPE_END_TURN, ct.TYPE_RELIC, ct.TYPE_INTENT, ct.TYPE_INTENT_NEXT),
        )

    def test_hello_and_its_reply_round_trip(self):
        sha = ["a" * 64, "b" * 64, "c" * 64]
        self.assertEqual(pp.unpack_hello(pp.pack_hello(*sha)), (1, *sha))
        self.assertEqual(pp.unpack_hello(pp.pack_hello(*sha, proto=9))[0], 9)
        self.assertEqual(pp.unpack_hello_reply(pp.pack_hello_reply(0)), (0, ""))
        self.assertEqual(pp.unpack_hello_reply(pp.pack_hello_reply(4, "card table differs")), (4, "card table differs"))

    def test_a_prior_frame_round_trips_real_tokens(self):
        tokens, _ = tokens_and_legal(0)
        frame = pp.pack_prior(70000, tokens, flags=1)
        req_id, types, ids, owners, feats, flags = pp.unpack_prior(frame)
        self.assertEqual((req_id, flags), (70000 & 0xFFFFFFFF, 1))
        self.assertEqual((types.dtype, ids.dtype, owners.dtype, feats.dtype), (np.uint8, np.int32, np.uint8, np.float32))
        self.assertEqual(feats.shape, (len(tokens), N_FEATURES))
        for got, want in ((types, tokens.types), (ids, tokens.ids), (owners, tokens.owners), (feats, tokens.feats)):
            self.assertTrue(np.array_equal(got, want))
        self.assertEqual(len(frame), 14 + len(tokens) * (1 + 4 + 1 + 4 * N_FEATURES))

    def test_a_reply_round_trips_and_is_140_bytes(self):
        row = np.linspace(-1, 1, MAX_ACTIONS).astype(np.float32)
        row[5] = -np.inf
        frame = pp.pack_reply(0, 12, row)
        self.assertEqual(len(frame), 140)
        status, req_id, got = pp.unpack_reply(frame)
        self.assertEqual((status, req_id), (0, 12))
        self.assertTrue(np.array_equal(got, row))
        status, req_id, got = pp.unpack_reply(pp.pack_reply(1, 3))
        self.assertEqual((status, req_id, float(got.sum())), (1, 3, 0.0))

    def test_malformed_frames_are_protocol_errors(self):
        tokens, _ = tokens_and_legal(0)
        prior = pp.pack_prior(1, tokens)
        hello = pp.pack_hello("a" * 64, "b" * 64, "c" * 64)
        reply = pp.pack_reply(0, 1, np.zeros(MAX_ACTIONS, np.float32))
        zero_tokens = prior[:10] + (0).to_bytes(2, "little") + prior[12:14]
        cases = {
            "bad magic": lambda: pp.unpack_prior(b"XXXX" + prior[4:]),
            "short header": lambda: pp.peek_kind(b"SPR"),
            "wrong kind": lambda: pp.unpack_prior(hello),
            "truncated prior": lambda: pp.unpack_prior(prior[:-1]),
            "long prior": lambda: pp.unpack_prior(prior + b"\0"),
            "prior with no tokens": lambda: pp.unpack_prior(zero_tokens),
            "truncated hello": lambda: pp.unpack_hello(hello[:-1]),
            "hello wrong kind": lambda: pp.unpack_hello(prior),
            "reply wrong size": lambda: pp.unpack_reply(reply[:-4]),
            "reply wrong kind": lambda: pp.unpack_reply(hello),
            "hello reply wrong kind": lambda: pp.unpack_hello_reply(prior),
            "short sha": lambda: pp.pack_hello("a" * 63, "b" * 64, "c" * 64),
            "feats of the wrong width": lambda: pp.pack_prior(1, pc.SyntheticTokens(
                np.zeros(3), np.zeros(3), np.zeros(3), np.zeros((3, 4)))),
            "no tokens": lambda: pp.pack_prior(1, pc.SyntheticTokens(
                np.zeros(0), np.zeros(0), np.zeros(0), np.zeros((0, N_FEATURES)))),
        }
        for name, call in cases.items():
            with self.subTest(name), self.assertRaises(pp.ProtocolError):
                call()

    def test_legal_weights_equal_the_old_per_row_code(self):
        rng = np.random.default_rng(7)
        for trial in range(400):
            row = rng.normal(size=MAX_ACTIONS).astype(np.float32) * 3
            row[rng.random(MAX_ACTIONS) < 0.4] = -np.inf
            if trial % 10 == 0:
                row[:] = -np.inf  # nothing finite: uniform
            if trial % 17 == 0:
                row[3] = np.nan
            legal = [int(a) for a in rng.integers(0, MAX_ACTIONS + 8, rng.integers(1, 9))]  # some out of the row
            got, want = pp.legal_weights(row, legal), old_weights(row, legal)
            self.assertTrue(np.array_equal(got, want), (trial, got, want))
            self.assertEqual(got.dtype, np.float64)
        self.assertEqual(pp.legal_weights(np.zeros(4, np.float32), [9, 10]).tolist(), [0.5, 0.5])

    def test_the_card_table_hash_is_stable(self):
        self.assertEqual(pp.card_table_sha256(), CARD_SHA)
        self.assertRegex(CARD_SHA, "^[0-9a-f]{64}$")


class ProcutilTests(unittest.TestCase):
    def test_watch_process_sees_a_child_exit(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])  # noqa: S603
        try:
            gone = procutil.watch_process(child.pid)
            self.assertFalse(gone())
            child.kill()
            child.wait()
            self.assertTrue(gone())
        finally:
            child.kill()
            child.wait()
        self.assertTrue(procutil.watch_process(2**22 + 3)())  # a pid nobody has

    @unittest.skipUnless(IS_WINDOWS, "priority classes are Windows only")
    def test_set_priority_class_lowers_and_raises(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); from sts2_gym import procutil as p;"
            "p.set_priority_class(p.BELOW_NORMAL_PRIORITY_CLASS); a = p.priority_class();"
            "p.set_priority_class(p.NORMAL_PRIORITY_CLASS); print(a, p.priority_class())"
        )
        out = subprocess.run([sys.executable, "-c", code, str(ROOT / "src")], capture_output=True, text=True, check=True)  # noqa: S603
        self.assertEqual(out.stdout.split(), [str(procutil.BELOW_NORMAL_PRIORITY_CLASS), str(procutil.NORMAL_PRIORITY_CLASS)])


class MemoryFloorTests(unittest.TestCase):
    """The memory floor of the fallback (docs/rl/GPU_INFERENCE.md section 6): no server, and a headroom under the floor."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.policy = export_prior(Path(cls.tmp.name)) / combat_prior.POLICY

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(pc.FAKE_HEADROOM_ENV, None)
        os.environ.pop(pc.FLOOR_ENV, None)
        pc.reset_state()
        self.addCleanup(pc.reset_state)

    def endpoint(self):
        return f"spireline-test-nomem-{os.getpid()}" if IS_WINDOWS else str(Path(self.tmp.name) / "none.sock")

    def remote(self):
        return pc.RemotePrior(str(self.policy), False, self.endpoint(), b"k" * 16, "0" * 64, timeout_s=2.0)

    def call(self, prior):
        combat = toadpoles(1)
        err = io.StringIO()
        try:
            legal = combat.legal()
            with contextlib.redirect_stderr(err):
                got = prior(combat.handle, legal)
            want = combat_prior.LivePrior(str(self.policy), False)(combat.handle, legal)
        finally:
            combat.close()
        return got, want, legal, err.getvalue()

    def test_the_headroom_is_a_positive_number_and_the_override_wins(self):
        real = pc.memory_headroom_gb()
        print(f"memory_headroom_gb on this machine: {real:.1f} GiB")
        self.assertIsInstance(real, float)
        self.assertTrue(0 < real < float("inf"))
        os.environ[pc.FAKE_HEADROOM_ENV] = "1.5"
        self.assertEqual(pc.memory_headroom_gb(), 1.5)
        os.environ[pc.FAKE_HEADROOM_ENV] = "not a number"  # an unreadable override is ignored
        self.assertTrue(0 < pc.memory_headroom_gb() < float("inf"))

    def test_the_floor_is_3_gib_unless_the_environment_says_otherwise(self):
        self.assertEqual(pc.fallback_floor_gb(), 3.0)
        os.environ[pc.FLOOR_ENV] = "5"
        self.assertEqual(pc.fallback_floor_gb(), 5.0)
        os.environ[pc.FLOOR_ENV] = "junk"
        self.assertEqual(pc.fallback_floor_gb(), 3.0)

    def test_under_the_floor_the_fallback_is_uniform_and_imports_no_torch(self):
        script = (
            "import sys, json, ctypes\nrl, src, args = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])\n"
            "sys.path[:0] = [rl, src]\nfrom sts2_gym import native\nimport prior_client as pc\n"
            "h = native.create(0)\nobs = (ctypes.c_int * native.OBS_SIZE)()\n"
            "native.reset_arena(h, args['deck'], obs, encounter_id=args['enc'])\n"
            "mask = native.valid_actions(h, 32)\nlegal = [a for a in range(32) if mask[a]]\n"
            "p = pc.RemotePrior(args['policy'], False, args['ep'], bytes.fromhex(args['key']), args['sha'])\n"
            "w = p(h, legal)\nw2 = p(h, legal)\n"
            "print(json.dumps({'torch': 'torch' in sys.modules, 'w': w.tolist(), 'w2': w2.tolist(), 'n': len(legal),\n"
            "                  'stats': pc.stats()}))\n"
        )
        args = {"deck": DECK, "enc": TOADPOLES, "policy": str(self.policy), "ep": self.endpoint(), "key": "6b" * 16, "sha": "0" * 64}
        env = {**os.environ, pc.FAKE_HEADROOM_ENV: "1"}
        out = subprocess.run([sys.executable, "-c", script, str(ROOT / "rl"), str(ROOT / "src"), json.dumps(args)],  # noqa: S603
                             capture_output=True, text=True, check=True, env=env)
        got = json.loads(out.stdout.strip().splitlines()[-1])
        n = got["n"]
        self.assertGreater(n, 1)
        self.assertFalse(got["torch"])  # the process that fell back never loaded torch
        for weights in (got["w"], got["w2"]):
            self.assertEqual(len(weights), n)
            self.assertAlmostEqual(sum(weights), 1.0)
            self.assertEqual(set(weights), {1.0 / n})  # equal weight per legal action
        stats = got["stats"]
        self.assertEqual((stats["server"], stats["cpu"], stats["uniform"]), (0, 0, 2))
        self.assertIn("memory floor", stats["fallback_reason"])
        self.assertIn("1.0 GiB of headroom under the 3 GiB floor", stats["fallback_reason"])
        lines = out.stderr.strip().splitlines()
        self.assertEqual(lines, ["prior falls back to a uniform prior: 1.0 GiB of headroom under the 3 GiB floor"])

    def test_the_floor_variable_moves_the_line_and_the_counts_are_in_process(self):
        os.environ[pc.FAKE_HEADROOM_ENV] = "4"
        os.environ[pc.FLOOR_ENV] = "5"
        got, _, legal, err = self.call(self.remote())
        np.testing.assert_array_equal(got, np.full(len(legal), 1.0 / len(legal)))
        stats = pc.stats()
        self.assertEqual((stats["server"], stats["cpu"], stats["uniform"]), (0, 0, 1))
        self.assertIn("4.0 GiB of headroom under the 5 GiB floor", err)
        self.assertEqual(len(err.strip().splitlines()), 1)
        pc.reset_stats()  # the counters reset; the process stays uniform, as it stays on the CPU prior today
        self.call(self.remote())
        self.assertEqual(pc.stats()["uniform"], 1)

    def test_over_the_floor_it_falls_back_to_the_cpu_prior_as_before(self):
        os.environ[pc.FAKE_HEADROOM_ENV] = "100"
        got, want, _, err = self.call(self.remote())
        self.assertTrue(np.array_equal(got, want))
        stats = pc.stats()
        self.assertEqual((stats["server"], stats["cpu"], stats["uniform"]), (0, 1, 0))
        self.assertNotIn("memory floor", stats["fallback_reason"])
        self.assertIn("using the CPU prior", err)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = export_prior(Path(cls.tmp.name))
        cls.policy = cls.root / combat_prior.POLICY

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        pc.reset_state()
        self.servers = []

    def tearDown(self):
        pc.reset_state()
        for server in self.servers:
            server.close()

    def start(self, **kwargs):
        server = ServerProc(self.policy, **kwargs)
        self.servers.append(server)
        self.assertTrue(server.ready.startswith("READY "), server.stderr())
        return server

    def test_a_batch_of_one_is_bit_exact_with_the_cpu_prior(self):
        server = self.start()
        remote, live = server.prior(), combat_prior.LivePrior(str(self.policy), False)
        for seed in range(6):
            combat = toadpoles(seed)
            try:
                legal = combat.legal()
                got, want = remote(combat.handle, legal), live(combat.handle, legal)
            finally:
                combat.close()
            self.assertTrue(np.array_equal(got, want), (seed, got, want))
        stats = pc.stats()
        self.assertEqual((stats["server"], stats["cpu"], stats["fallback_reason"]), (6, 0, None))
        self.assertEqual(stats["latency_ms"]["n"], 6)
        self.assertGreater(stats["latency_ms"]["p50"], 0)
        pc.reset_stats()
        self.assertEqual((pc.stats()["server"], pc.stats()["latency_ms"]["n"]), (0, 0))

    def test_the_client_imports_no_torch_and_travels_as_a_pickle(self):
        server = self.start()
        prior = server.prior()
        self.assertLess(len(pickle.dumps(prior)), 600)
        self.assertEqual(pickle.loads(pickle.dumps(prior)), prior)  # noqa: S301
        script = (
            "import sys, json, ctypes\nrl, src, args = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])\n"
            "sys.path[:0] = [rl, src]\nfrom sts2_gym import native\nimport prior_client as pc\n"
            "h = native.create(0)\nobs = (ctypes.c_int * native.OBS_SIZE)()\n"
            "native.reset_arena(h, args['deck'], obs, encounter_id=args['enc'])\n"
            "mask = native.valid_actions(h, 32)\nlegal = [a for a in range(32) if mask[a]]\n"
            "p = pc.RemotePrior(args['policy'], False, args['ep'], bytes.fromhex(args['key']), args['sha'])\n"
            "w = p(h, legal)\nprint(json.dumps({'torch': 'torch' in sys.modules, 'server': pc.stats()['server'], 'n': len(w)}))\n"
        )
        args = {"deck": DECK, "enc": TOADPOLES, "policy": str(self.policy), "ep": server.endpoint,
                "key": server.key, "sha": server.sha}
        out = subprocess.run([sys.executable, "-c", script, str(ROOT / "rl"), str(ROOT / "src"), json.dumps(args)],  # noqa: S603
                             capture_output=True, text=True, check=True, env=dict(os.environ))
        got = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual((got["torch"], got["server"]), (False, 1))
        self.assertGreater(got["n"], 1)

    def test_four_concurrent_clients_are_padded_and_agree_within_1e6(self):
        server = self.start(extra=["--max-batch", "4", "--deadline-us", "500000"])
        net = combat_prior.network(self.policy)
        positions = [tokens_and_legal(seed) for seed in range(4)]
        expected = []
        for tokens, _ in positions:
            with torch.inference_mode():
                expected.append(net.model.policy_logits(**net._batch([tokens]))[0].numpy())  # noqa: SLF001
        barrier, rows, errors = threading.Barrier(4, timeout=60), {}, []

        def client(i):
            try:
                conn, status, reason = server.raw()
                assert status == 0, reason
                for round_ in range(4):
                    barrier.wait()
                    if i == 3 and round_ % 2:
                        continue  # a round of three: the spare row is a copy of a real one
                    conn.send_bytes(pp.pack_prior(round_ + 1, positions[i][0]))
                    status, req_id, row = pp.unpack_reply(conn.recv_bytes())
                    assert (status, req_id) == (0, round_ + 1)
                    rows.setdefault(i, []).append(row.copy())
                conn.send_bytes(pp.pack_bye())
            except Exception as error:  # noqa: BLE001
                errors.append(pp.describe_error(error))
                barrier.abort()

        threads = [threading.Thread(target=client, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
        self.assertEqual(errors, [])
        for i in range(4):
            self.assertEqual(len(rows[i]), 4 if i != 3 else 2)
            for row in rows[i]:
                np.testing.assert_allclose(row, expected[i], atol=1e-6, rtol=0)
                legal = positions[i][1]
                np.testing.assert_allclose(pp.legal_weights(row, legal), pp.legal_weights(expected[i], legal), atol=1e-6)
        self.assertEqual(server.stop(), 0)
        hist = json.loads(server.stats_path.read_text())["batch_hist"]
        self.assertGreater(hist.get("4", 0), 0, hist)  # padding really happened: rows or tokens beyond the requests'
        self.assertGreater(hist.get("3", 0), 0, hist)

    def test_hello_refusals_carry_the_status(self):
        server = self.start()
        cases = {
            "policy": (dict(policy_sha="0" * 64), 1),
            "vocab": (dict(vocab="1" * 64), 2),
            "proto": (dict(version=2), 3),
            "card table": (dict(card="2" * 64), 4),
        }
        for name, (kwargs, want) in cases.items():
            with self.subTest(name):
                conn, status, reason = server.raw(**kwargs)
                self.assertEqual(status, want, reason)
                self.assertIn("differs", reason)
                conn.close()
        conn, status, _ = server.raw()  # the server is still there
        self.assertEqual(status, 0)
        conn.send_bytes(b"SPR1\x02\x00junk")  # a PRIOR that cannot be parsed: status 1, then the connection closes
        self.assertEqual(pp.unpack_reply(conn.recv_bytes())[0], 1)
        with self.assertRaises((EOFError, OSError)):
            conn.recv_bytes()
        conn.close()

    def test_an_invalid_token_id_is_a_bad_request_not_a_crash(self):
        server = self.start()
        conn, status, _ = server.raw()
        self.assertEqual(status, 0)
        tokens, _ = tokens_and_legal(0)
        bad = pc.SyntheticTokens(tokens.types.copy(), tokens.ids.copy(), tokens.owners.copy(), tokens.feats)
        bad.types[0], bad.ids[0] = 1, 10**6  # the global token's encounter id, far past the table
        conn.send_bytes(pp.pack_prior(1, bad))
        self.assertEqual(pp.unpack_reply(conn.recv_bytes())[0], 1)
        conn.close()
        conn, status, _ = server.raw()
        conn.send_bytes(pp.pack_prior(2, tokens))
        self.assertEqual(pp.unpack_reply(conn.recv_bytes())[:2], (0, 2))
        self.assertIsNone(server.proc.poll())

    def test_no_server_falls_back_once_and_for_good(self):
        endpoint = f"spireline-test-none-{os.getpid()}" if IS_WINDOWS else str(Path(self.tmp.name) / "none.sock")
        remote = pc.RemotePrior(str(self.policy), False, endpoint, b"k" * 16, "0" * 64, timeout_s=2.0)
        live = combat_prior.LivePrior(str(self.policy), False)
        err = io.StringIO()
        combat = toadpoles(1)
        try:
            legal = combat.legal()
            with contextlib.redirect_stderr(err):
                first, second = remote(combat.handle, legal), remote(combat.handle, legal)
            want = live(combat.handle, legal)
        finally:
            combat.close()
        self.assertTrue(np.array_equal(first, want) and np.array_equal(second, want))
        stats = pc.stats()
        self.assertEqual((stats["server"], stats["cpu"]), (0, 2))
        self.assertTrue(stats["fallback_reason"])
        self.assertEqual(len(err.getvalue().strip().splitlines()), 1, err.getvalue())  # one line, not one per call
        self.assertNotIn("Traceback", err.getvalue())

    def _falls_back(self, server, prior, expect_in_reason):
        combat = toadpoles(2)
        try:
            legal = combat.legal()
            with contextlib.redirect_stderr(io.StringIO()):
                got = prior(combat.handle, legal)
            want = combat_prior.LivePrior(str(self.policy), False)(combat.handle, legal)
        finally:
            combat.close()
        self.assertTrue(np.array_equal(got, want))
        stats = pc.stats()
        self.assertEqual((stats["server"], stats["cpu"]), (0, 1))
        self.assertIn(expect_in_reason, stats["fallback_reason"])
        return stats

    def test_a_wrong_authkey_falls_back(self):
        server = self.start()
        self._falls_back(server, server.prior(authkey=b"not the key"), "AuthenticationError")
        pc.reset_state()
        combat = toadpoles(3)  # the server dropped that client and carries on
        try:
            server.prior()(combat.handle, combat.legal())
        finally:
            combat.close()
        self.assertEqual(pc.stats()["server"], 1)

    def test_a_policy_mismatch_falls_back_with_the_status(self):
        server = self.start()
        stats = self._falls_back(server, server.prior(policy_sha256="0" * 64), "status 1")
        self.assertIn("policy", stats["fallback_reason"])

    def test_a_server_that_dies_mid_run_makes_the_next_call_fall_back(self):
        server = self.start()
        remote, live = server.prior(timeout_s=3.0), combat_prior.LivePrior(str(self.policy), False)
        combat = toadpoles(4)
        try:
            legal = combat.legal()
            self.assertTrue(np.array_equal(remote(combat.handle, legal), live(combat.handle, legal)))
            self.assertEqual(pc.stats()["server"], 1)
            server.proc.terminate()
            server.proc.wait(15)
            with contextlib.redirect_stderr(io.StringIO()):
                after = remote(combat.handle, legal)  # no exception
            self.assertTrue(np.array_equal(after, live(combat.handle, legal)))
        finally:
            combat.close()
        stats = pc.stats()
        self.assertEqual((stats["server"], stats["cpu"]), (1, 1))
        self.assertTrue(stats["fallback_reason"])

    def test_the_kill_file_stops_the_server_and_the_client_falls_back(self):
        server = self.start()
        remote = server.prior(timeout_s=3.0)
        combat = toadpoles(5)
        try:
            legal = combat.legal()
            remote(combat.handle, legal)
            self.assertEqual(pc.stats()["server"], 1)
            started = time.perf_counter()
            server.kill_file.write_text("off")
            self.assertEqual(server.proc.wait(5.0), 0)
            self.assertLess(time.perf_counter() - started, 2.5)
            print(f"kill file to exit: {time.perf_counter() - started:.2f} s")
            with contextlib.redirect_stderr(io.StringIO()):
                remote(combat.handle, legal)
        finally:
            combat.close()
        self.assertEqual((pc.stats()["server"], pc.stats()["cpu"]), (1, 1))

    def test_the_server_exits_on_stdin_eof(self):
        server = self.start()
        self.assertIsNone(server.proc.poll())
        server.proc.stdin.close()
        self.assertEqual(server.proc.wait(10), 0)
        self.assertTrue(server.stats_path.exists())  # the exit flushes the stats

    def test_the_server_exits_when_its_parent_is_gone(self):
        # The test holds the pipe the server reads as stdin, so only the parent watch can end it.
        wrapper = (
            "import os, subprocess, sys\n"
            "p = subprocess.Popen(sys.argv[1:] + ['--parent-pid', str(os.getpid())], stdout=subprocess.PIPE,"
            " stderr=subprocess.DEVNULL)\n"
            "print(p.stdout.readline().decode().strip(), p.pid, flush=True)\n"
        )
        holder = subprocess.Popen([sys.executable, "-c", wrapper, sys.executable, str(SERVER), "--policy",  # noqa: S603
                                   str(self.policy), "--endpoint", f"spireline-test-parent-{os.getpid()}"
                                   if IS_WINDOWS else str(Path(self.tmp.name) / "parent.sock"),
                                   "--authkey-env", "SPR_TEST_KEY", "--device", "cpu"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  env=dict(os.environ, SPR_TEST_KEY="ab" * 16))
        parts = holder.stdout.readline().decode().split()
        self.assertEqual(parts[0], "READY", parts)
        pid = int(parts[-1])
        holder.wait(60)
        self.addCleanup(holder.stdin.close)
        deadline = time.time() + 15
        while procutil.pid_alive(pid) and time.time() < deadline:
            time.sleep(0.25)
        self.assertFalse(procutil.pid_alive(pid), "the server outlived its parent")

    def test_a_thread_exception_takes_the_process_down_with_code_1(self):
        server = self.start(env={"SPIRELINE_PRIOR_SERVER_FAULT": "batcher"})
        conn, status, _ = server.raw()
        self.assertEqual(status, 0)
        tokens, _ = tokens_and_legal(0)
        conn.send_bytes(pp.pack_prior(1, tokens))
        with self.assertRaises((EOFError, OSError)):
            conn.recv_bytes()
        self.assertEqual(server.proc.wait(15), 1)
        text = server.stderr()
        self.assertIn("RuntimeError: injected fault", text)
        self.assertNotIn("Traceback", text)
        self.assertTrue(server.stats_path.exists())

    def test_a_connection_thread_exception_also_exits_with_code_1(self):
        server = self.start(env={"SPIRELINE_PRIOR_SERVER_FAULT": "connection"})
        conn, status, _ = server.raw()
        tokens, _ = tokens_and_legal(0)
        conn.send_bytes(pp.pack_prior(1, tokens))
        self.assertEqual(server.proc.wait(15), 1)

    def test_without_cuda_the_default_device_refuses_to_start(self):
        if torch.cuda.is_available():
            self.skipTest("this interpreter has CUDA")
        out = subprocess.run([sys.executable, str(SERVER), "--policy", str(self.policy), "--endpoint", "x",  # noqa: S603
                              "--authkey-env", "SPR_TEST_KEY"], capture_output=True, text=True, env=dict(os.environ, SPR_TEST_KEY="ab" * 16))
        self.assertEqual(out.returncode, 2)
        self.assertEqual(len(out.stderr.strip().splitlines()), 1, out.stderr)


    def test_a_request_longer_than_max_tokens_gets_its_own_eager_forward_and_is_counted(self):
        server = self.start(extra=["--max-tokens", "8", "--priority", "below"])
        net = combat_prior.network(self.policy)
        tokens, legal = tokens_and_legal(0)
        self.assertGreater(len(tokens), 8)
        with torch.inference_mode():
            want = net.model.policy_logits(**net._batch([tokens]))[0].numpy()  # noqa: SLF001
        conn, status, _ = server.raw()
        self.assertEqual(status, 0)
        conn.send_bytes(pp.pack_prior(1, tokens))
        status, req_id, row = pp.unpack_reply(conn.recv_bytes())
        self.assertEqual((status, req_id), (0, 1))
        self.assertTrue(np.array_equal(row, want))
        conn.send_bytes(pp.pack_bye())
        self.assertEqual(server.stop(), 0)
        stats = json.loads(server.stats_path.read_text())
        self.assertEqual((stats["calls"], stats["overlong"], stats["batches"]), (1, 1, 0))
        self.assertEqual(stats["restarts"], 0)
        self.assertEqual(stats["device"], "cpu")

    def test_a_key_that_is_not_hex_is_refused(self):
        out = subprocess.run([sys.executable, str(SERVER), "--policy", str(self.policy), "--endpoint", "x",  # noqa: S603
                              "--authkey-env", "SPR_TEST_KEY", "--device", "cpu"],
                             capture_output=True, text=True, env=dict(os.environ, SPR_TEST_KEY="not hex"))
        self.assertEqual(out.returncode, 2)
        self.assertIn("not hexadecimal", out.stderr)


@unittest.skipUnless(
    GPU_PYTHON.exists() and torch.cuda.is_available(),
    "needs the GPU venv and a CUDA device in the interpreter that runs this module",
)
class GpuToleranceTests(unittest.TestCase):
    """Section 5: max |dlogit| <= 1e-4 at the action slots and no argmax flip above a 1e-5 gap, alone and in 32s."""

    def test_the_graph_logits_match_the_cpu_prior_on_500_positions(self):
        import prior_server

        # server.load() sets process-wide torch switches (fused attention off, TF32 off); put them back afterwards, or the
        # CPU priors that later tests build in this process would no longer play as the server subprocess does
        backends = (torch.backends.mha.get_fastpath_enabled(), torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)

        def restore():
            torch.backends.mha.set_fastpath_enabled(backends[0])
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = backends[1], backends[2]

        self.addCleanup(restore)
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as tmp:
            policy = export_prior(Path(tmp)) / combat_prior.POLICY
            net = combat_prior.PolicyNet(policy)
            server = prior_server.PriorServer(policy, "cuda", max_batch=32, max_tokens=64)
            server.load()
            server.capture()
        tokens = [pc.synthetic_tokens(rng, int(n)) for n in rng.integers(8, 61, 500)]
        reference = []
        for t in tokens:
            with torch.inference_mode():
                reference.append(net.model.policy_logits(**net._batch([t]))[0].numpy())  # noqa: SLF001

        def request(t):
            return prior_server.Request(t.types.astype(np.uint8), t.ids.astype(np.int32), t.owners.astype(np.uint8), t.feats)

        for rows in (1, 32):
            worst, flips, compared = 0.0, 0, 0
            for start in range(0, 500, rows):
                batch = [request(t) for t in tokens[start : start + rows]]
                server.run(batch)
                for offset, req in enumerate(batch):
                    got, want = req.logits, reference[start + offset]
                    self.assertTrue(np.array_equal(np.isfinite(got), np.isfinite(want)))
                    finite = np.isfinite(want)
                    if finite.any():
                        worst = max(worst, float(np.abs(got[finite] - want[finite]).max()))
                    if finite.sum() >= 2:
                        top = np.sort(want[finite])[-2:]
                        if top[1] - top[0] > 1e-5:
                            compared += 1
                            flips += int(np.argmax(np.where(finite, got, -np.inf)) != np.argmax(np.where(finite, want, -np.inf)))
            print(f"gpu bucket of {rows}: max |dlogit| {worst:.3g}, argmax flips {flips} of {compared} clear positions")
            self.assertLessEqual(worst, 1e-4)
            self.assertEqual(flips, 0)


class SecretsTests(unittest.TestCase):
    def test_a_repr_of_the_remote_prior_never_shows_the_key(self):
        key = bytes(range(32))
        prior = pc.RemotePrior(policy="p.pt", probe=False, endpoint="ep", authkey=key, policy_sha256="0" * 64)
        text = repr(prior) + str(prior)
        self.assertNotIn(key.hex(), text)
        self.assertNotIn(repr(key), text)
        self.assertEqual(prior.authkey, key)

    def test_a_short_key_is_refused_at_start(self):
        proc = subprocess.run([sys.executable, str(ROOT / "rl" / "prior_server.py"), "--device", "cpu", "--policy", "x.pt",
                               "--endpoint", "e", "--authkey-env", "SPR_TEST_KEY"], capture_output=True, text=True,
                              env=dict(os.environ, SPR_TEST_KEY="ab"))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("at least", proc.stderr)

    def test_describe_error_hides_drive_paths(self):
        text = pp.describe_error(FileNotFoundError("no such file: C:\\x\\y.pt or D:/a/b.pt"))
        self.assertNotIn("C:", text)
        self.assertNotIn("D:/", text)
        self.assertIn("<path>", text)


if __name__ == "__main__":
    unittest.main()
