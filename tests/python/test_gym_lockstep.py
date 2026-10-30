"""The lockstep fight gym (rl/gym_lockstep.py): K games a process, the network's calls answered in one batch.

The pinned facts: on the CPU path a game played in lockstep is the game the sequential gym plays (K = 1 against
`fight_gym.run_tasks`, and K = 4 against K = 1: the same won, end HP, turns, decisions, simulations and moves); grading with the
exact solver never changes a move; a step that never returns (F18's fixture) or a process over its memory is killed and its
other games are played again in a fresh process; the PRIORS frame carries many rows and the server answers them as it answers
one. The games are short arena fights with the fixture prior (tests/python/prior_fixture.py), played in spawned processes on
core2 as the gym plays them.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fight_gym as fg  # noqa: E402
import gym_lockstep as gl  # noqa: E402
import live_search  # noqa: E402
import prior_protocol as pp  # noqa: E402

F18 = HERE / "fixtures" / "f18_step_hang.json"
STRIKE, DEFEND, BASH = 472, 131, 30
DECK = [STRIKE] * 5 + [DEFEND] * 4 + [BASH]
GAME_FIELDS = ("won", "end_hp", "max_hp", "potions_used", "potions_held", "turns", "decisions", "sims", "capped", "L0",
               "root_value", "root_p_death", "root_best_value", "root_sims", "first_move", "moves_sha", "budget_exhausted")


# ---------------------------------------------------------------------------------------------------------- stand-ins


def _fake_steps(task, clock, game, ctx):
    """A stand-in game for the guard tests: three prior requests, then ("ok", id, 0) ends, ("sleep", id, s) never returns from
    a 'step', ("alloc", id, MB) takes private memory and then sleeps, ("raise", id, 0) raises."""
    kind, tid, x = task[:3]
    game.note({"k": 1, "turns": 1})
    for _ in range(3):
        w = yield (0, [0, 1, 2])
        assert len(w) == 3
    if kind == "sleep":
        time.sleep(x)
    if kind == "alloc":
        blob = bytearray(int(x * 2**20))
        blob[::4096] = b"x" * len(blob[::4096])  # touch every page so it is private memory
        time.sleep(120)
    if kind == "raise":
        raise RuntimeError("combat action refused")
    return {"_task": list(task), "ok": True, "seconds": round(clock(), 4)}


def _make_starts(queue, f18_doc) -> None:
    """Arena fights' first decisions as gym starts, made on core2 in a child (the gym's engine)."""
    try:
        for p in (ROOT / "rl", ROOT / "src", ROOT):
            sys.path.insert(0, str(p))
        import gen0  # noqa: PLC0415

        gen0.select_engine("core2")
        import ctypes  # noqa: PLC0415

        from sts2_gym import native  # noqa: PLC0415
        from sts2_gym.env import ENCOUNTER_IDS  # noqa: PLC0415

        starts = []
        for i, (enc, seed, hp) in enumerate((("toadpoles", 3, 60), ("nibbits-weak", 5, 50), ("toadpoles", 11, 40))):
            h = native.create(seed)
            obs = (ctypes.c_int * native.OBS_SIZE)()
            native.reset_arena(h, DECK, obs, encounter_id=ENCOUNTER_IDS[enc], player_hp=hp, player_max_hp=80)
            doc = native.get_state_json(h)
            native.destroy(h)
            starts.append({"id": f"T{i}|f05", "cat": "act1-boss", "kind": "boss", "act": 1, "floor": 5, "node_type": 0, "hp": hp,
                           "max_hp": 80, "deck_size": len(DECK), "potions": 0, "doc": doc})
        starts.append({"id": "F18|f17", "cat": "act1-boss", "kind": "boss", "act": 1, "floor": 17, "node_type": 6, "hp": 7,
                       "max_hp": 93, "deck_size": 0, "potions": 0, "doc": f18_doc})
        queue.put(("ok", starts))
    except Exception as error:  # noqa: BLE001 -- the tests skip when the native library is not there
        queue.put(("error", f"{type(error).__name__}: {error}"))


def _rows(path: Path) -> dict:
    return {(r["_task"][1], r["_task"][2]): r for r in fg.read_jsonl(path)}


def _game(row: dict) -> dict:
    return {k: row.get(k) for k in GAME_FIELDS}


# ---------------------------------------------------------------------------------------------------------- pure parts


class PureTests(unittest.TestCase):
    def test_a_fights_clock_counts_its_own_time_and_its_share(self):
        c = gl.OwnClock()
        self.assertEqual(c(), 0.0)
        c.start()
        time.sleep(0.05)
        c.stop()
        ran = c()
        time.sleep(0.05)  # stopped: another fight's turn
        self.assertEqual(c(), ran)
        self.assertGreaterEqual(ran, 0.04)
        c.acc += 0.5
        self.assertAlmostEqual(c(), ran + 0.5)

    def test_drive_raises_a_prior_error_inside_the_steps_so_their_cleanup_runs(self):
        closed = []

        def steps():
            try:
                w = yield (7, [0, 1])
                return w
            finally:
                closed.append(True)

        self.assertEqual(live_search._drive(steps(), lambda h, legal: [h, len(legal)]), [7, 2])  # noqa: SLF001

        def bad(_h, _legal):
            raise ValueError("broken prior")

        with self.assertRaisesRegex(ValueError, "broken prior"):
            live_search._drive(steps(), bad)  # noqa: SLF001
        self.assertEqual(closed, [True, True])

    def test_search_steps_refuses_the_oracle(self):
        with self.assertRaisesRegex(ValueError, "always resamples"):
            next(live_search.search_steps(0, resample=False))

    def test_a_priors_frame_round_trips_and_a_bad_one_is_refused(self):
        rng = np.random.default_rng(0)
        from prior_client import synthetic_tokens  # noqa: PLC0415

        rows = [synthetic_tokens(rng, int(n)) for n in (16, 30, 4, 57)]
        req, got, flags = pp.unpack_priors(pp.pack_priors(9, rows))
        self.assertEqual((req, flags, len(got)), (9, 0, 4))
        for t, (types, ids, owners, feats) in zip(rows, got):
            self.assertTrue(np.array_equal(types, t.types) and np.array_equal(ids, t.ids) and np.array_equal(owners, t.owners))
            self.assertTrue(np.array_equal(feats, t.feats))
        logits = rng.random((4, pp.MAX_ACTIONS), dtype=np.float32)
        status, req, back = pp.unpack_priors_reply(pp.pack_priors_reply(pp.REPLY_OK, 9, 4, logits))
        self.assertEqual((status, req), (pp.REPLY_OK, 9))
        self.assertTrue(np.array_equal(back, logits))
        frame = pp.pack_priors(1, rows)
        for broken in (frame[:-1], frame + b"\0", frame[:10]):
            with self.assertRaises(pp.ProtocolError):
                pp.unpack_priors(broken)
        with self.assertRaises(pp.ProtocolError):
            pp.pack_priors(1, [])
        with self.assertRaises(pp.ProtocolError):
            pp.unpack_prior(frame)  # a PRIORS frame is not a PRIOR frame

    def test_k_is_held_inside_the_emulators_pool(self):
        for k in (0, gl.K_MAX + 1):
            with self.assertRaisesRegex(ValueError, "pool"):
                gl.run_lockstep([], Path("unused"), 1, k, (), {"prior": "uniform"})

    def test_a_grade_summary_counts_exact_regret_only(self):
        grades = [{"k": 0, "status": 0, "exact": True, "h0": True, "value": 0.8, "q_played": 0.75, "s": 0.1},
                  {"k": 1, "status": 1, "exact": False, "h0": True, "value": 0.6, "q_played": 0.1, "s": 0.2},
                  {"k": 2, "error": "RuntimeError: x", "s": 0.0}]
        self.assertEqual(gl.grade_summary(grades), {"graded": 3, "h0": 2, "exact": 1, "regret_exact": 0.05, "seconds": 0.3})


class GuardTests(unittest.TestCase):
    def test_a_hung_step_a_memory_hog_and_a_raise_are_capped_and_the_other_games_still_finish(self):
        opts = {"prior": "uniform", "game_steps": "test_gym_lockstep:_fake_steps", "paths": [str(HERE)]}
        tasks = [("sleep", "a", 600), ("ok", "b", 0), ("alloc", "c", 1500), ("ok", "d", 0), ("raise", "e", 0), ("ok", "f", 0)]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "rows.jsonl"
            t0 = time.perf_counter()
            stopped, stats = gl.run_lockstep(tasks, target, 1, 2, (), opts, hang_s=20.0, mem_limit=1.0 * 2**30, poll_s=0.2)
            seconds = time.perf_counter() - t0
            rows = {r["_task"][1]: r for r in fg.read_jsonl(target)}
        self.assertIsNone(stopped)
        self.assertLess(seconds, 150)  # never waits on the sleeper or the allocator
        self.assertEqual(sorted(rows), ["a", "b", "c", "d", "e", "f"])
        for tid in "bdf":
            self.assertTrue(rows[tid].get("ok"), rows[tid])
        self.assertTrue(rows["a"]["cap_reason"].startswith("no step for 20 s: hung (killed)"), rows["a"])
        self.assertTrue(rows["c"]["cap_reason"].startswith("memory over 1.0 GB"), rows["c"])
        self.assertTrue(rows["e"]["cap_reason"].startswith("error: RuntimeError: combat action refused"), rows["e"])
        # Only the last process lives to report: d, e (its capped row) and f; b finished in the one killed for c.
        self.assertEqual([s["games"] for s in stats], [3])
        self.assertGreater(stats[0]["requests"], 0)


# ---------------------------------------------------------------------------------------------------------- real games


class LockstepGameTests(unittest.TestCase):
    """Short arena fights at a3-v1 with the fixture prior (prior_min_visits 0: every new tree node asks the network)."""

    @classmethod
    def setUpClass(cls):
        from prior_fixture import export_prior  # noqa: PLC0415 -- torch

        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        export_prior(root / "agents" / "fixture", prior_min_visits=0)
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        p = ctx.Process(target=_make_starts, args=(queue, json.loads(F18.read_text(encoding="utf-8"))["doc"]), daemon=True)
        p.start()
        kind, got = queue.get(timeout=120)
        p.join(30)
        if kind != "ok":
            raise unittest.SkipTest(f"native library unavailable: {got}")
        cls.out = root / "out"
        cls.out.mkdir()
        (cls.out / "starts.jsonl").write_text("".join(json.dumps(s) + "\n" for s in got), encoding="utf-8")
        (cls.out / "meta.json").write_text(json.dumps({"engine": "core2", "agent_root": str(root / "agents"), "agent": "fixture"}),
                                           encoding="utf-8")
        cls.init = fg._init_args(cls.out)  # noqa: SLF001
        cls.tasks = [("a3-v1", f"T{i}|f05", j, "act1-boss") for i in range(3) for j in range(2)]
        cls.seq = cls.out / "seq.jsonl"
        stopped = fg.run_tasks(cls.tasks, fg._game_task, cls.seq, 1, init=fg._init_worker, init_args=cls.init)  # noqa: SLF001
        assert stopped is None
        cls.k1 = cls.out / "k1.jsonl"
        stopped, cls.k1_stats = gl.run_lockstep(cls.tasks, cls.k1, 1, 1, cls.init, {"prior": "cpu"})
        assert stopped is None

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_k1_plays_the_sequential_gyms_games(self):
        seq, k1 = _rows(self.seq), _rows(self.k1)
        self.assertEqual(sorted(seq), sorted(k1))
        self.assertEqual(len(seq), 6)
        for key in seq:
            self.assertFalse(seq[key]["capped"], seq[key])
            self.assertEqual(_game(k1[key]), _game(seq[key]), key)
            self.assertEqual(k1[key].get("_leaf"), seq[key].get("_leaf"), key)  # M3's leaf, captured at the first End Turn of j = 0
            self.assertEqual((k1[key]["lockstep"], k1[key]["prior_path"], k1[key]["worlds"]), (1, "cpu", "reload"))
        self.assertGreater(sum(s["requests"] for s in self.k1_stats), 50)  # the prior was asked, through the batch

    def test_k4_plays_the_same_games_as_k1_in_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "k4.jsonl"
            stopped, stats = gl.run_lockstep(self.tasks, target, 1, 4, self.init, {"prior": "cpu"})
            k4, k1 = _rows(target), _rows(self.k1)
        self.assertIsNone(stopped)
        self.assertEqual(sorted(k4), sorted(k1))
        for key in k1:
            self.assertEqual(_game(k4[key]), _game(k1[key]), key)
        hist = {int(n): c for s in stats for n, c in s["batch_hist"].items()}
        self.assertGreater(max(hist), 1)  # some batches held several games' calls

    def test_grading_records_a_grade_per_decision_and_never_changes_a_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "graded.jsonl"
            stopped, stats = gl.run_lockstep(self.tasks, target, 1, 4, self.init, {"prior": "cpu", "grade_wall_s": 0.5})
            graded, k1 = _rows(target), _rows(self.k1)
        self.assertIsNone(stopped)
        exact = 0
        for key in k1:
            self.assertEqual(_game(graded[key]), _game(k1[key]), key)
            grades = graded[key]["grades"]
            self.assertEqual([g["k"] for g in grades], list(range(k1[key]["decisions"])))
            self.assertEqual(graded[key]["graded"]["graded"], len(grades))
            for g in grades:
                self.assertNotIn("error", g)
                if g["exact"] and g["q_played"] is not None:
                    exact += 1
                    self.assertLessEqual(g["q_played"], g["value"] + 1e-6)  # the best move's value (q is float32)
        self.assertGreater(exact, 0)  # the fights' last turns are solved exactly
        self.assertGreater(sum(s["grade_s"] for s in stats), 0)

    def test_memory_worlds_fork_one_load_and_seed_zero_is_the_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "mem.jsonl"
            stopped, _ = gl.run_lockstep(self.tasks, target, 1, 4, self.init, {"prior": "cpu", "worlds": "memory"})
            mem, k1 = _rows(target), _rows(self.k1)
        self.assertIsNone(stopped)
        self.assertEqual(sorted(mem), sorted(k1))
        for key in k1:
            self.assertFalse(mem[key]["capped"], mem[key])
            self.assertEqual(mem[key]["worlds"], "memory")
            if key[1] == 0:
                self.assertEqual(_game(mem[key]), _game(k1[key]), key)

    def test_a_hung_step_f18_is_killed_and_the_others_replayed_in_a_fresh_process(self):
        tasks = [("a3-v1", "F18|f17", 0, "act1-boss")] + self.tasks[:2]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "f18.jsonl"
            t0 = time.perf_counter()
            stopped, _ = gl.run_lockstep(tasks, target, 1, 3, self.init, {"prior": "cpu"}, hang_s=30.0, poll_s=0.2)
            seconds = time.perf_counter() - t0
            rows, k1 = _rows(target), _rows(self.k1)
        self.assertIsNone(stopped)
        self.assertLess(seconds, 240)
        hung = rows[("F18|f17", 0)]
        if not hung.get("capped"):
            self.skipTest("F18's step returned: the engine is fixed (test_kernel_f18_step_hang reports it)")
        self.assertTrue(hung["cap_reason"].startswith("no step for 30 s: hung (killed)"), hung)
        self.assertFalse(hung["won"])
        for key in (("T0|f05", 0), ("T0|f05", 1)):
            self.assertEqual(_game(rows[key]), _game(k1[key]), key)


class ServerBatchTests(unittest.TestCase):
    """One PRIORS frame to a CPU-device server (the tests' device) answers each row as the CPU prior does, within 1e-6."""

    def test_a_batch_of_positions_matches_the_cpu_prior(self):
        import combat_prior  # noqa: PLC0415
        import importlib.util  # noqa: PLC0415

        import prior_client as pc  # noqa: PLC0415
        from prior_fixture import export_prior  # noqa: PLC0415

        spec = importlib.util.spec_from_file_location("prior_server_tests_for_lockstep", HERE / "test_prior_server.py")
        tps = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tps)
        with tempfile.TemporaryDirectory() as tmp:
            policy = export_prior(Path(tmp)) / combat_prior.POLICY
            server = tps.ServerProc(policy)
            pc.reset_state()
            combats = []
            try:
                self.assertTrue(server.ready.startswith("READY "), server.stderr())
                remote, live = server.prior(), combat_prior.LivePrior(str(policy), False)
                combats = [tps.toadpoles(seed) for seed in range(5)]
                requests = [(c.handle, c.legal()) for c in combats]
                got = gl.ServerBatch(remote)(requests)
                want = [live(h, legal) for h, legal in requests]
                stats = pc.stats()
                one = gl.ServerBatch(remote)(requests[:1])
            finally:
                for c in combats:
                    c.close()
                pc.reset_state()
                server.close()
        self.assertEqual((stats["server"], stats["cpu"], stats["fallback_reason"]), (5, 0, None))
        for g, w in zip(got, want):
            self.assertEqual(g.shape, w.shape)
            self.assertLess(float(np.max(np.abs(g - w))), 1e-6)
        self.assertTrue(np.array_equal(one[0], want[0]))  # a batch of one is unpadded: bit-exact


if __name__ == "__main__":
    unittest.main()
