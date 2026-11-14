"""The generation loop (rl/gen_loop.py, rl/gen0.py train and promote): shards, training, the agent's prior, the E1 block,
the promotion maths and the lab's train markers. Tests that need the native library skip without it."""

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import fight_exam as fe  # noqa: E402
import gen0  # noqa: E402
import gen_loop  # noqa: E402
import human_prior_infer  # noqa: E402
import run_student  # noqa: E402
import keep_v2_fixtures as fx  # noqa: E402
from test_fight_exam import human_gym_folder  # noqa: E402

HAVE_NATIVE = (ROOT / "out" / "Sts2Emulator.dll").exists() or bool(os.environ.get("STS2_LIB_PATH") and (Path(os.environ["STS2_LIB_PATH"]) / "Sts2Emulator.dll").exists())


class _Tok:
    """A stand-in for combat_tokens.Tokens."""

    def __init__(self, n: int, rng: np.random.Generator) -> None:
        from combat_tokens import N_FEATURES, TYPE_BUFF, TYPE_ENEMY, TYPE_GLOBAL, TYPE_HAND, TYPE_RELIC

        pool = np.array([TYPE_ENEMY, TYPE_BUFF, TYPE_RELIC])
        # token 0 is the global token, tokens 1..4 are the four legal actions (hand cards), then the rest of the state
        self.types = np.concatenate([[TYPE_GLOBAL], [TYPE_HAND] * 4, pool[rng.integers(0, len(pool), n - 5)]]).astype(np.int64)
        self.ids = rng.integers(0, 4, n).astype(np.int64)
        self.ids[0] = 0
        self.owners = rng.integers(0, 2, n).astype(np.int64)
        self.feats = (rng.normal(size=(n, N_FEATURES)) * 0.1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.types)


def fake_run_arrays(rng: np.random.Generator, n_fights: int = 3, per_fight: int = 4) -> dict:
    s = gen_loop.CombatSamples()
    for f in range(n_fights):
        for _ in range(per_fight):
            visits = np.zeros(32, np.float32)
            legal = np.zeros(32, np.int8)
            legal[:4] = 1
            visits[:4] = rng.integers(1, 20, 4)
            q = np.full(32, np.nan, np.float32)
            q[:4] = rng.normal(size=4)
            s.add(_Tok(int(rng.integers(8, 14)), rng), visits, q, legal, encounter=f, deck=10, relics=1, sims=30)
        s.end_fight(end_hp=40 + f, max_hp=80)
    return s.arrays()


def fake_record(seed: str, generation: int) -> dict:
    return {"seed": seed, "generation": generation}


class ShardTests(unittest.TestCase):
    def test_part_round_trip_and_generation_tag(self):
        rng = np.random.default_rng(1)
        runs = [("AAAAAAAAAA", fake_run_arrays(rng)), ("BBBBBBBBBB", None), ("CCCCCCCCCC", fake_run_arrays(rng, 2, 3))]
        with tempfile.TemporaryDirectory() as d:
            n = gen_loop.write_part(Path(d) / "p.npz", runs, generation=3)
            back = gen_loop.read_part(Path(d) / "p.npz")
        self.assertEqual(n, 12 + 6)
        self.assertEqual(int(back["generation"][0]), 3)
        self.assertEqual(back["seeds"].tolist(), ["AAAAAAAAAA", "BBBBBBBBBB", "CCCCCCCCCC"])
        np.testing.assert_array_equal(back["visits"][:12], runs[0][1]["visits"])
        np.testing.assert_array_equal(back["types"], np.concatenate([runs[0][1]["types"], runs[2][1]["types"]]))
        # `fights` is the run's index in the part, so train_combat's split "by fight" is a split by run
        self.assertEqual(sorted(set(back["fights"].tolist())), [0, 2])
        self.assertEqual(int(back["lengths"].sum()), back["types"].size)
        self.assertTrue(np.isnan(back["returns"]).all())

    def test_sink_writes_manifest_tagged_with_the_generation_and_refuses_another(self):
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as d:
            sink = gen_loop.ShardSink(Path(d), 2, config="a3-v1", base_sims=30, ascension=10, agent={}, part_runs=2)
            sink.add(fake_record("AAAAAAAAAA", 2), fake_run_arrays(rng))
            sink.add(fake_record("BBBBBBBBBB", 2), fake_run_arrays(rng))
            sink.add(fake_record("CCCCCCCCCC", 2), None)
            with self.assertRaises(ValueError):  # a raise, not an assert: python -O keeps it
                sink.add(fake_record("DDDDDDDDDD", 1), None)
            man = sink.close()
            found = gen_loop.find_manifests(Path(d))
            self.assertEqual(len(found), 1)
            self.assertEqual(int(found[0][1]["generation"]), 2)
            self.assertEqual(gen_loop.runs_in_generation(Path(d), 2), 3)
            self.assertEqual(gen_loop.runs_in_generation(Path(d), 1), 0)
            self.assertEqual(man["runs"], 3)
            self.assertEqual(len(man["combat_parts"]), 2)
            self.assertEqual(man["samples"], 24)
            self.assertEqual(len(gen_loop.window_manifests(Path(d), 4)), 1)  # generations 2..4
            self.assertEqual(len(gen_loop.window_manifests(Path(d), 5)), 0)  # 3..5 has no generation 2

    def test_root_targets_share_merged_actions_and_sum_targets(self):
        ch = [SimpleNamespace(action=1, merged_actions=[2], visits=10, value_mean=-4.0),
              SimpleNamespace(action=5, merged_actions=[], visits=6, value_mean=-2.0),
              SimpleNamespace(action=5, merged_actions=[], visits=2, value_mean=-10.0)]
        v, q = gen_loop.root_targets(ch, 8)
        self.assertEqual(v.tolist(), [0, 5, 5, 0, 0, 8, 0, 0])
        self.assertAlmostEqual(float(q[1]), -4.0)
        self.assertAlmostEqual(float(q[5]), (6 * -2.0 + 2 * -10.0) / 8)
        self.assertTrue(np.isnan(q[0]))


class ResumeWriterTests(unittest.TestCase):
    """A self-play block the lab interrupts and resumes replays its seed stream from the start; the writer must not append
    a seed it already holds (a run record and its combat part) and the manifest must count the distinct seeds."""

    SEEDS = [f"RESUME{i:04d}" for i in range(7)]

    @staticmethod
    def _records(directory: Path) -> list[dict]:
        from run_train import shards

        return [r for p in shards.shard_paths(Path(directory) / "shards") for r in shards.read_shard(p)]

    def _sink(self, d: str, generation: int = 2) -> "gen_loop.ShardSink":
        return gen_loop.ShardSink(Path(d), generation, config="a3-v1", base_sims=30, ascension=10, agent={}, part_runs=2)

    def test_replayed_stream_writes_each_seed_once(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as d:
            first = self._sink(d)
            for s in self.SEEDS[:5]:  # two parts of two runs reach the disk; the fifth run is still buffered when it is killed
                first.add(fake_record(s, 2), fake_run_arrays(rng))
            for _ in range(2):  # resumed twice: the stream restarts at its first seed each time
                sink = self._sink(d)
                for s in self.SEEDS:
                    sink.add(fake_record(s, 2), fake_run_arrays(rng))
                man = sink.close()
            seeds = [r["seed"] for r in self._records(d)]
            self.assertEqual(sorted(seeds), sorted(self.SEEDS))  # one run record per seed
            part_seeds = [str(x) for n in man["combat_parts"] for x in gen_loop.read_part(Path(d) / "shards" / n)["seeds"]]
            self.assertEqual(sorted(part_seeds), sorted(self.SEEDS))  # one combat part row per seed
            self.assertEqual(man["runs"], len(self.SEEDS))
            self.assertEqual(man["samples"], sum(int(gen_loop.read_part(Path(d) / "shards" / n)["lengths"].size) for n in man["combat_parts"]))
            self.assertEqual(sum(man["prior_paths"].values()), len(self.SEEDS))

    def test_resumed_sink_names_the_seeds_it_holds_and_continues_the_parts(self):
        rng = np.random.default_rng(6)
        with tempfile.TemporaryDirectory() as d:
            first = self._sink(d)
            for s in self.SEEDS[:4]:
                first.add(fake_record(s, 2), fake_run_arrays(rng))
            first.flush()
            sink = self._sink(d)
            self.assertEqual(sink.done, set(self.SEEDS[:4]))
            self.assertEqual(sink.runs, 4)
            self.assertEqual(len(sink.parts), 2)
            sink.add(fake_record(self.SEEDS[4], 2), fake_run_arrays(rng))
            sink.add(fake_record(self.SEEDS[5], 2), fake_run_arrays(rng))
            man = sink.close()
            self.assertEqual(len(set(man["combat_parts"])), 3)  # the resume numbers on: it overwrote no part
            self.assertEqual(man["runs"], 6)

    def test_run_block_plays_only_the_seeds_not_yet_recorded(self):
        rng = np.random.default_rng(7)
        played: list[str] = []

        def fake_run_safe(seed, ix, sims, ascension, agent=None, record=False, character="IRONCLAD"):
            played.append(seed)
            return {"seed": seed, "ix": ix, "win": False, "floor": 5, "end": "death", "decisions": 3, "wall": 0.1,
                    "ai_state_assumed": 0,
                    "_shard": {"record": fake_record(seed, gen_loop.Agent().generation), "samples": fake_run_arrays(rng)}}

        class InlinePool:  # gen0.run_block's SupervisedPool (rl/spawn_pool.py) with every task run in this process, at once
            stats = {"started": 0, "recycled": 0, "died": 0, "timed_out": 0, "raised": 0}

            def __init__(self, ctx, size, **kw):
                self.size, self.ended = size, []

            def free(self):
                return self.size - len(self.ended)

            def inflight(self):
                return len(self.ended)

            def submit(self, key, fn, args=()):
                self.ended.append(gen0.spawn_pool.Done(key, "ok", fn(*args)))

            def collect(self):
                out, self.ended = self.ended, []
                return out

            def wait(self, timeout):
                pass

            def terminate(self):
                pass

        with tempfile.TemporaryDirectory() as d, mock.patch.object(gen0, "_run_safe", fake_run_safe), \
                mock.patch.object(gen0.spawn_pool, "SupervisedPool", InlinePool):
            gen0.run_block(self.SEEDS[:4], "a3-v1", 2, Path(d), record=True)  # the first attempt
            played.clear()
            summ = gen0.run_block(self.SEEDS, "a3-v1", 2, Path(d), record=True)  # the resume replays the whole stream
            self.assertEqual(sorted(played), sorted(self.SEEDS[4:]))
            self.assertEqual(summ["shards"]["runs"], len(self.SEEDS))
            self.assertEqual(sorted(r["seed"] for r in self._records(d)), sorted(self.SEEDS))
            played.clear()
            gen0.run_block(self.SEEDS, "a3-v1", 2, Path(d), record=True)  # a third attempt finds nothing left to play
            self.assertEqual(played, [])
            self.assertEqual(len(self._records(d)), len(self.SEEDS))


class DedupeReaderTests(unittest.TestCase):
    """Blocks written before the writer was idempotent hold a seed up to 8 times; the readers keep one copy per (generation, seed)."""

    SEEDS = [f"READER{i:04d}" for i in range(6)]

    @staticmethod
    def _block(root: Path, name: str, seeds: list[str], rng: np.random.Generator, generation: int = 2) -> dict:
        sink = gen_loop.ShardSink(root / name, generation, config="a3-v1", base_sims=30, ascension=10, agent={}, part_runs=2)
        arrays = {}
        for s in seeds:
            arrays[s] = fake_run_arrays(rng)  # a copy differs from the first record so the test can tell which one was read
            sink.add(fake_record(s, generation), arrays[s])
        sink.close()
        return arrays

    def _window(self, root: Path, rng: np.random.Generator) -> tuple[dict, dict]:
        a = self._block(root, "job-a", self.SEEDS[:4], rng)
        b = self._block(root, "job-b", self.SEEDS[2:6], rng)  # replays the last two seeds of a, then two new ones
        return a, b

    def test_build_training_set_keeps_the_first_copy_of_each_run(self):
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as d:
            a, b = self._window(Path(d), rng)
            data = gen_loop.build_training_set(gen_loop.window_manifests(Path(d), 2), max_fights=10**6)
        kept = [a[s] for s in self.SEEDS[:4]] + [b[s] for s in self.SEEDS[4:6]]
        self.assertEqual(int(data["lengths"].size), sum(int(x["lengths"].size) for x in kept))
        self.assertEqual(int(data["_fights_available"][0]), sum(len(np.unique(x["fight_ix"])) for x in kept))
        self.assertAlmostEqual(float(data["visits"].sum()), float(sum(x["visits"].sum() for x in kept)), places=3)
        self.assertEqual(int(data["_runs_dropped"][0]), 2)
        self.assertEqual(int(data["_samples_dedup_dropped"][0]), sum(int(b[s]["lengths"].size) for s in self.SEEDS[2:4]))
        self.assertEqual(int(data["_samples_by_generation"][0][1]), int(data["lengths"].size))
        self.assertEqual(len(np.unique(data["fights"])), 6)  # six distinct runs

    def test_a_window_without_copies_drops_nothing(self):
        rng = np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as d:
            self._block(Path(d), "job-a", self.SEEDS[:3], rng)
            self._block(Path(d), "job-b", self.SEEDS[3:], rng)
            data = gen_loop.build_training_set(gen_loop.window_manifests(Path(d), 2), max_fights=10**6)
        self.assertEqual(int(data["_runs_dropped"][0]), 0)
        self.assertEqual(int(data["_samples_dedup_dropped"][0]), 0)

    def test_the_same_seed_in_two_generations_is_two_runs(self):
        rng = np.random.default_rng(10)
        with tempfile.TemporaryDirectory() as d:
            self._block(Path(d), "job-a", self.SEEDS[:2], rng, generation=1)
            self._block(Path(d), "job-b", self.SEEDS[:2], rng, generation=2)
            data = gen_loop.build_training_set(gen_loop.window_manifests(Path(d), 2), max_fights=10**6)
        self.assertEqual(int(data["_runs_dropped"][0]), 0)
        self.assertEqual(len(np.unique(data["fights"])), 4)

    def test_train_json_reports_the_duplicates_dropped(self):
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._window(root / "runs", rng)
            cand = gen_loop.train_generation(2, root / "runs", root / "agent", device="cpu", epochs=1, max_fights=20, threads=2,
                                             vrun=False, export=False)
            doc = json.loads((cand / "train.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["dedup"]["combat_runs_dropped"], 2)
        self.assertGreater(doc["dedup"]["combat_samples_dropped"], 0)
        self.assertEqual(doc["dedup"]["combat_runs_distinct"], 6)


_FIXTURE: dict = {}


def trained_candidate() -> tuple[Path, Path, Path]:
    """Two generations of tiny synthetic shards, and generation 2's candidate trained from them (once per test run)."""
    if not _FIXTURE:
        import atexit

        tmp = tempfile.TemporaryDirectory()
        atexit.register(tmp.cleanup)
        root = Path(tmp.name)
        rng = np.random.default_rng(3)
        for g in (0, 1):
            sink = gen_loop.ShardSink(root / "runs" / f"job{g}", g, config="a3-v1", base_sims=30, ascension=10, agent={}, part_runs=3)
            for i in range(6):
                sink.add(fake_record(f"S{g}{i:08d}"[:10], g), fake_run_arrays(rng, 3, 5))
            sink.close()
        with mock.patch.dict(os.environ):
            os.environ.pop(gen_loop.MACRO_VRUN_FLAG, None)  # no variable: the switch is on by default (D33)
            cand = gen_loop.train_generation(1, root / "runs", root / "agent", device="cpu", epochs=2, max_fights=20, threads=2,
                                             vrun=False)
        _FIXTURE.update(runs=root / "runs", agents=root / "agent", cand=cand)
    return _FIXTURE["runs"], _FIXTURE["agents"], _FIXTURE["cand"]


class TrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs, cls.agents, cls.cand = trained_candidate()

    def test_train_writes_checkpoints_and_train_json(self):
        c = self.cand
        self.assertEqual(c, self.agents / "gen2" / "candidate")
        for name in ("combat_policy.pt", "combat_net.pt", "model.json", "agent.json", "train.json"):
            self.assertTrue((c / name).is_file(), name)
        t = json.loads((c / "train.json").read_text())
        self.assertEqual(t["candidate_generation"], 2)
        self.assertEqual(t["fights_kept"], 20)  # 12 runs x 3 fights = 36 available, capped
        self.assertEqual(t["fights_available"], 36)
        self.assertEqual(len(t["combat"]["curves"]), 2)
        self.assertGreater(t["combat"]["cpu_min"], 0)
        self.assertEqual(json.loads((c / "agent.json").read_text())["generation"], 2)
        self.assertEqual(json.loads((c / "agent.json").read_text())["mask_version"], 4)  # D34: new candidates carry the versioned mask (3 until the Wongos Leave was shown, 2 under D32)
        switch = t["vrun_keep"]["macro_switch"]  # D33: on without the variable (no fit here: vrun=False, so nothing passed)
        self.assertEqual((switch["passed"], switch["enabled"]), (False, True))
        self.assertIn("D33", switch["note"])

    def test_onnx_is_written_and_read_back_or_the_reason_is_recorded(self):
        c = self.cand
        onnx = json.loads((c / "train.json").read_text())["onnx"]["combat_net"]
        if onnx["ok"]:  # an environment with the onnx package: the file exists and was read back
            self.assertTrue((c / "combat_net.onnx").is_file())
            self.assertIn("checked_with", onnx)
        else:  # offline without it (this PC): torch's exporter itself stops, and train.json says why
            self.assertIn("onnx", onnx["reason"])
            self.assertFalse((c / "combat_net.onnx").exists())

    def test_onnx_reader_parses_a_model_proto(self):
        def ld(no, payload):
            return bytes([(no << 3) | 2, len(payload)]) + payload

        node = ld(4, b"Gemm")
        graph = ld(1, node) + ld(1, node) + ld(5, b"\x00") + ld(11, ld(1, b"x")) + ld(12, ld(1, b"y"))
        model = bytes([(1 << 3) | 0, 8]) + ld(7, graph)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "m.onnx").write_bytes(model)
            info = gen_loop.onnx_structure(Path(d) / "m.onnx")
        self.assertEqual((info["nodes"], info["ops"], info["inputs"], info["outputs"], info["initializers"], info["ir_version"]),
                         (2, ["Gemm"], ["x"], ["y"], 1, 8))

    def test_exported_prior_is_found_and_agent_carries_it(self):
        import combat_prior

        champ = combat_prior.find(self.cand)
        self.assertIsNotNone(champ)
        self.assertFalse(champ.probe)
        agent = gen_loop.load_agent(self.cand)
        self.assertEqual(agent.generation, 2)
        extra = agent.search_extra()
        self.assertIsInstance(extra["prior_fn"], combat_prior.LivePrior)
        self.assertEqual(extra["prior_min_visits"], 3)
        self.assertEqual(gen_loop.Agent().search_extra(), {})  # generation 0 has no prior

    def test_window_keeps_three_generations(self):
        data = gen_loop.build_training_set(gen_loop.window_manifests(self.runs, 1), max_fights=1000)
        self.assertEqual(int(data["_fights_available"][0]), 36)
        self.assertEqual(sorted(g for g, _ in data["_samples_by_generation"].tolist()), [0, 1])
        self.assertEqual(len(set(data["fights"].tolist())), 12)  # 12 distinct run ids across both generations


@unittest.skipUnless(HAVE_NATIVE, "needs the native library")
class SearchUsesThePriorTests(unittest.TestCase):
    def test_play_emu_run_hands_the_prior_to_the_search(self):
        import live_search

        cand = trained_candidate()[2]
        agent = gen_loop.load_agent(cand)
        seen = {}

        class Stop(Exception):
            pass

        def fake(handle, **kw):
            seen.update(kw)
            raise Stop

        with mock.patch.object(live_search, "search_live", fake):
            row = gen0._run_safe("SJ9DHBJHPE", 0, "a3-v1", 10, agent)
        self.assertEqual(row["end"], "crash")
        self.assertIn("Stop", row["error"])
        self.assertIsNotNone(seen.get("prior_fn"))
        self.assertEqual(seen["prior_min_visits"], 3)
        self.assertAlmostEqual(seen["c_puct"], 0.2)

    def test_a_real_search_asks_the_prior(self):
        import live_search
        from run_train import engine

        cand = trained_candidate()[2]
        extra = gen_loop.load_agent(cand).search_extra()
        real, calls = extra["prior_fn"], []

        def counted(handle, legal):
            calls.append(len(legal))
            return real(handle, legal)

        extra["prior_fn"] = counted
        s = engine.RunSession("SJ9DHBJHPE", 10, engine.CombatBudget(30, 30))
        try:
            rules = gen_loop.Agent().macro_policy()
            rng = np.random.default_rng(0)
            while s.phase != engine.rc.PHASE_COMBAT or s.decision() is not None:
                d = s.decision()
                s.apply(d.options[rules.pick(s.view(), d, rng).index])
            root = s.combat(fork_combat=True).combat_fork(None)
            try:
                res = live_search.search_live(root.handle, seed=1, max_sims=12, min_child_sims=1, c=0.2, confirm_pairs=0, **extra)
            finally:
                root.close() if hasattr(root, "close") else None
        finally:
            s.close()
        self.assertGreater(len(calls), 0)
        self.assertGreater(res.sims, 0)

    def test_search_config_adds_the_prior_when_the_environment_names_a_generation(self):
        import search_configs

        cfg = search_configs.get("a3-v1")
        self.assertNotIn("prior_fn", cfg.search_kwargs())
        with mock.patch.dict(os.environ, {"SPIRELINE_GEN_DIR": str(trained_candidate()[2])}):
            self.assertIn("prior_fn", cfg.search_kwargs())


class E1BlockTests(unittest.TestCase):
    def test_e1_file_follows_the_rule_and_overlaps_neither_e0_nor_a_stream(self):
        f = ROOT / "referee" / "seeds" / "E1-2026-10-23.json"
        seeds = gen0.load_seeds(f)
        self.assertEqual(seeds, gen0.e_block("E1-2026-10-23", 2000))
        self.assertEqual(len(seeds), 2000)
        e0 = gen0.load_seeds(ROOT / "referee" / "seeds" / "E0-2026-10-23.json")
        self.assertFalse(set(seeds) & set(e0))
        for job in ("selfplay", "selfplay-1", "selfplay-2026", "SP-selfplay"):
            self.assertFalse(set(seeds) & set(gen0.stream_seeds(job, 20000)), job)
        gen_loop.check_disjoint(seeds, {"E0": e0})
        with self.assertRaises(ValueError):
            gen_loop.check_disjoint(seeds, {"copy of E1": seeds[:5]})

    def test_resolve_e_block_reuses_a_committed_block_and_creates_a_fresh_one_per_generation(self):
        with tempfile.TemporaryDirectory() as d:
            tpl = str(Path(d) / "E{k}-{date}.json")
            (Path(d) / "E1-2026-10-23.json").write_text(json.dumps({"seeds": ["A"]}))
            self.assertEqual(gen_loop.resolve_e_block(tpl, 1).name, "E1-2026-10-23.json")
            p2 = gen_loop.resolve_e_block(tpl, 2, n=30)
            self.assertTrue(p2.name.startswith("E2-"))
            self.assertEqual(gen0.load_seeds(p2), gen0.e_block(p2.stem, 30))
            self.assertEqual(gen_loop.resolve_e_block(tpl, 2, n=30), p2)
            self.assertFalse(set(gen0.load_seeds(p2)) & set(gen0.e_block("E1-2026-10-23", 2000)))

    def test_known_seed_sets_read_the_other_blocks_and_the_lab_streams(self):
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d) / "runs" / "selfplay-1"
            runs.mkdir(parents=True)
            (runs / "runs.jsonl").write_text(json.dumps({"seed": "ZZZZZZZZZZ"}) + "\n", encoding="utf-8")
            sets = gen_loop.known_seed_sets(ROOT / "referee" / "seeds", Path(d) / "runs", "E1-2026-10-23")
            self.assertIn("E0-2026-10-23", sets)
            self.assertNotIn("E1-2026-10-23", sets)
            self.assertEqual(sets["self-play runs"], ["ZZZZZZZZZZ"])


class PromotionMathTests(unittest.TestCase):
    def test_exact_mcnemar_one_sided(self):
        self.assertAlmostEqual(gen_loop.mcnemar_exact_one_sided(10, 0), 2 ** -10)
        self.assertAlmostEqual(gen_loop.mcnemar_exact_one_sided(7, 1), 9 / 256)
        self.assertAlmostEqual(gen_loop.mcnemar_exact_one_sided(3, 3), 42 / 64)
        self.assertEqual(gen_loop.mcnemar_exact_one_sided(0, 0), 1.0)
        self.assertAlmostEqual(gen_loop.mcnemar_exact_one_sided(0, 5), 1.0)

    def test_bootstrap_lower_bound(self):
        r = gen_loop.cluster_bootstrap_lower([1.0] * 50, b=2000)
        self.assertEqual(r["lower"], 1.0)
        self.assertLess(r["p"], 0.01)
        x = [1.0] * 90 + [0.0] * 10  # mean .9, se = sqrt(.9 * .1 / 100) = .03; the 1% quantile is about .9 - 2.326 * .03
        r = gen_loop.cluster_bootstrap_lower(x, b=20000)
        self.assertAlmostEqual(r["mean"], 0.9)
        self.assertAlmostEqual(r["lower"], 0.9 - 2.326 * 0.03, delta=0.02)
        sym = [1.0, -1.0] * 100
        r = gen_loop.cluster_bootstrap_lower(sym, b=4000)
        self.assertLess(r["lower"], 0)
        self.assertGreater(r["p"], 0.3)
        self.assertTrue(math.isnan(gen_loop.cluster_bootstrap_lower([])["mean"]))

    def test_boss_kill_is_a_win_or_a_run_past_floor_17(self):
        self.assertFalse(gen_loop.boss_kill({"win": False, "floor": 17}))
        self.assertTrue(gen_loop.boss_kill({"win": False, "floor": 18}))
        self.assertTrue(gen_loop.boss_kill({"win": True, "floor": 53}))

    def test_decision_rule_and_the_ten_percent_wins_clause(self):
        good_f = {"lower": 0.2}
        bad_f = {"lower": -0.1}
        good_b = {"p": 0.001, "b": 20, "c": 2}
        bad_b = {"p": 0.2, "b": 8, "c": 5}
        wins_ok = {"p": 0.004, "b": 15, "c": 1}
        wins_bad = {"p": 0.3, "b": 5, "c": 4}
        d = gen_loop.decide(good_f, good_b, wins_bad, 0.05, 0.02)  # neither arm at 10%: wins do not count
        self.assertTrue(d["promote"])
        self.assertFalse(d["wins_co_primary"])
        d = gen_loop.decide(good_f, good_b, wins_bad, 0.12, 0.05)  # only one arm at 10%: still off
        self.assertTrue(d["promote"])
        d = gen_loop.decide(good_f, good_b, wins_bad, 0.12, 0.10)  # both arms at 10%: wins must pass
        self.assertTrue(d["wins_co_primary"])
        self.assertFalse(d["promote"])
        self.assertTrue(gen_loop.decide(good_f, good_b, wins_ok, 0.2, 0.1)["promote"])
        self.assertFalse(gen_loop.decide(bad_f, good_b, wins_ok, 0.2, 0.1)["promote"])
        # D24: under 10% boss kills in either arm the boss kill is a not-worse guard, from 10% in both a co-primary
        guard = gen_loop.decide(good_f, bad_b, wins_ok, 0.0, 0.0)
        self.assertTrue(guard["promote"])
        self.assertEqual(guard["boss_kill_mode"], "guard")
        co = gen_loop.decide(good_f, bad_b, wins_ok, 0.0, 0.0, cand_boss_rate=0.2, base_boss_rate=0.15)
        self.assertEqual(co["boss_kill_mode"], "co-primary")
        self.assertFalse(co["promote"])
        self.assertTrue(gen_loop.decide(good_f, good_b, wins_ok, 0.0, 0.0, cand_boss_rate=0.2, base_boss_rate=0.15)["promote"])
        # the guard fails only when the candidate is significantly worse
        self.assertFalse(gen_loop.decide(good_f, {"p": 0.999, "b": 2, "c": 20}, wins_ok, 0.0, 0.0)["promote"])
        self.assertTrue(gen_loop.decide(good_f, {"p": 0.9, "b": 10, "c": 14}, wins_ok, 0.0, 0.0)["promote"])
        # generation 1's numbers on E1-2026-10-23: 40 against 27 kills of 2,000, discordant 33 and 20
        g1 = {"pairs": 2000, "cand": 40, "base": 27, "b": 33, "c": 20, "p": 0.0492}
        d = gen_loop.decide(good_f, g1, wins_ok, 0.0, 0.0)
        self.assertEqual((d["boss_kill_mode"], d["promote"]), ("guard", True))
        self.assertGreater(d["boss_kill_p_worse"], 0.9)

    def test_compare_arms_pairs_on_seed_and_decides(self):
        rng = np.random.default_rng(4)
        seeds = [f"S{i:09d}" for i in range(400)]
        base = [{"seed": s, "floor": int(rng.integers(3, 17)), "win": False} for s in seeds]
        cand = [{"seed": s, "floor": min(17, b["floor"] + int(rng.integers(0, 3))), "win": False} for s, b in zip(seeds, base)]
        for r in cand[:60]:
            r["floor"] = 18  # killed the boss
        res = gen_loop.compare_arms(cand, base, boot=2000)
        self.assertEqual(res["paired_seeds"], 400)
        self.assertGreater(res["floor"]["lower"], 0)
        self.assertEqual(res["boss_kill"]["c"], 0)
        self.assertGreaterEqual(res["boss_kill"]["b"], 55)
        self.assertTrue(res["decision"]["promote"])
        self.assertFalse(res["decision"]["wins_co_primary"])
        same = gen_loop.compare_arms(base, base, boot=500)
        self.assertFalse(same["decision"]["promote"])
        self.assertEqual(same["floor"]["mean"], 0.0)


class LabMarkerTests(unittest.TestCase):
    def _shards(self, root: Path, gen: int, runs: int):
        sink = gen_loop.ShardSink(root / f"job-g{gen}-{runs}", gen, config="a3-v1", base_sims=30, ascension=10, agent={})
        for i in range(runs):
            sink.add(fake_record(f"R{gen}{i:08d}"[:10], gen), None)
        sink.close()

    def test_ready_markers_come_lowest_generation_first_and_skip_done(self):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d)
            for n in ("ready-10", "ready-2", "done-1", "ready-x", "ready-3"):
                (t / n).write_text("x")
            self.assertEqual([g for g, _ in gen_loop.ready_markers(t)], [2, 3, 10])
            self.assertEqual(gen_loop.ready_markers(t / "nope"), [])
            from tools.lab import jobs

            self.assertEqual(jobs.train_markers(Path(d).parent / "none"), [])

    def test_marker_is_written_when_the_budget_is_spent_and_only_once(self):
        with tempfile.TemporaryDirectory() as d:
            lab, runs = Path(d) / "lab", Path(d) / "runs"
            self._shards(runs, 1, 3)
            self.assertIsNone(gen_loop.mark_ready(lab, 1, runs, budget=5))
            self.assertFalse((lab / "train" / "ready-1").exists())
            self._shards(runs, 1, 2)
            self._shards(runs, 0, 9)  # another generation's runs do not count
            p = gen_loop.mark_ready(lab, 1, runs, budget=5)
            self.assertEqual(p, lab / "train" / "ready-1")
            self.assertIsNone(gen_loop.mark_ready(lab, 1, runs, budget=5))
            (lab / "train" / "ready-1").rename(lab / "train" / "done-1")  # what the lab does after the job
            self.assertIsNone(gen_loop.mark_ready(lab, 1, runs, budget=5))
            from tools.lab import jobs

            self.assertEqual(jobs.train_markers(lab), [])

    def test_train_command_takes_the_marker_and_needs_one(self):
        with tempfile.TemporaryDirectory() as d:
            marker_dir = Path(d) / "train"
            rc = gen0.main(["train", "--marker-dir", str(marker_dir), "--shards-root", d, "--agent-root", d])
            self.assertEqual(rc, 2)
            marker_dir.mkdir()
            (marker_dir / "ready-7").write_text("x")
            with mock.patch.object(gen_loop, "train_generation", return_value=Path(d) / "gen8" / "candidate") as tg:
                rc = gen0.main(["train", "--marker-dir", str(marker_dir), "--shards-root", d, "--agent-root", d])
            self.assertEqual(rc, 0)
            self.assertEqual(tg.call_args[0][0], 7)  # the marker's generation
            self.assertTrue((marker_dir / "ready-7").exists())  # the lab consumes it when the command succeeds


class BuiltCandidateTests(unittest.TestCase):
    """`build_candidate` and `gen0.py build` (D44 step A): another prior export with a generation's V_run fit and macro, nothing
    trained; `train` with the `ready-<base>-g<N>` marker judges it as it is and never trains over it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _, _, trained = trained_candidate()
        self.agents = self.root / "agent"
        self.base = self.agents / "gen5" / "candidate"  # a vrun generation: its fit and settings are what the candidate keeps
        self.base.mkdir(parents=True)
        for name in ("combat_policy.pt", "combat_net.pt", "model.json"):
            (self.base / name).write_bytes((trained / name).read_bytes())
        (self.base / "run_value.npz").write_bytes(b"the base fit")
        (self.base / "run_value.pt").write_bytes(b"the base fit, torch")
        (self.base / "agent.json").write_text(json.dumps({"generation": 5, "macro": "vrun", "mask_version": 4, "tau": 0.25}))
        (self.agents / "current.json").write_text(json.dumps({"generation": 5, "dir": str(self.base)}))
        self.prior = self.root / "players" / "v2"  # the fight loop's player layout
        self.prior.mkdir(parents=True)
        for name in ("combat_policy.pt", "combat_net.pt", "model.json"):
            (self.prior / name).write_bytes((trained / name).read_bytes())
        meta = json.loads((self.prior / "model.json").read_text())
        (self.prior / "model.json").write_text(json.dumps({**meta, "iteration": 7, "gym_loop": {"checkpoint": "c0019"}}))
        (self.prior / "train.json").write_text(json.dumps({"checkpoint": "c0019", "steps": 921}))

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, target=7, **kw):
        return gen_loop.build_candidate(self.prior, self.agents / "current.json", self.agents, target, why="test", export=False, **kw)

    def test_the_candidate_plays_the_prior_with_the_base_generations_fit_and_macro(self):
        cand = self.build()
        self.assertEqual(self.agents / "gen7" / "candidate", cand)
        agent = gen_loop.load_agent(cand)
        self.assertEqual((agent.generation, agent.macro, agent.tau, agent.mask_version), (7, "vrun", 0.25, 4))
        self.assertEqual(gen_loop.file_sha256(self.prior / "combat_policy.pt"), agent.describe()["prior_sha256"])
        self.assertEqual(gen_loop.file_sha256(self.base / "run_value.npz"), gen_loop.file_sha256(agent.macro_npz))
        self.assertEqual((self.base / "run_value.pt").read_bytes(), (cand / "run_value.pt").read_bytes())
        model = json.loads((cand / "model.json").read_text())
        self.assertEqual((model["iteration"], model["search"], model["probe"]), (7, gen_loop.PRIOR_SEARCH, False))
        self.assertNotIn("gym_loop", model)  # a candidate's model.json, as `write_prior_model_json` writes every one
        meta = json.loads((cand / "agent.json").read_text())
        self.assertEqual((meta["built_on"], meta["lineage"]["skips"], meta["vrun_fit"]["source"]), (5, [6], "base"))
        self.assertFalse(meta["built"]["trained"])
        self.assertEqual("c0019", json.loads((cand / "train.json").read_text())["prior_train"]["checkpoint"])
        self.assertEqual(cand, gen_loop.built_candidate(self.agents, 7))
        self.assertIsNone(gen_loop.built_candidate(self.agents, 5))  # a candidate the train job trained is not a built one

    def test_nothing_is_written_over_and_a_prior_played_otherwise_is_refused(self):
        self.build()
        with self.assertRaises(FileExistsError):
            self.build()
        with self.assertRaises(ValueError):
            self.build(target=5)  # not after the base
        meta = json.loads((self.prior / "model.json").read_text())
        (self.prior / "model.json").write_text(json.dumps({**meta, "probe": True}))
        with self.assertRaises(ValueError):
            self.build(target=8)
        self.assertFalse((self.agents / "gen8" / "candidate").exists())
        import combat_prior

        (self.prior / "model.json").write_text(json.dumps({**meta, "vocab_sha256": "0" * 64}))
        with self.assertRaises(combat_prior.PriorError):
            self.build(target=8)

    def train_args(self, marker_dir: Path) -> list[str]:
        return ["train", "--marker-dir", str(marker_dir), "--shards-root", str(self.root), "--agent-root", str(self.agents),
                "--promote", "--emu-python", "python", "--seeds-file", str(self.root / "E7-test.json"), "--review-guard"]

    def test_the_train_job_judges_a_built_candidate_and_trains_nothing(self):
        marker_dir = self.root / "train"
        rc = gen0.main(["build", "--prior", str(self.prior), "--agent-root", str(self.agents), "--candidate-generation", "7",
                        "--why", "test", "--no-export", "--marker-dir", str(marker_dir)])
        self.assertEqual(0, rc)
        self.assertEqual(["ready-5-g7"], [p.name for p in marker_dir.iterdir()])
        with mock.patch.object(gen_loop, "train_generation") as tg, \
                mock.patch.object(gen_loop, "run_promote_subprocess", return_value=0) as rp:
            rc = gen0.main(self.train_args(marker_dir))
        self.assertEqual(0, rc)
        tg.assert_not_called()
        self.assertEqual(self.agents / "gen7" / "candidate", rp.call_args[0][1])
        self.assertTrue(rp.call_args[1]["review_guard"])
        self.assertTrue((marker_dir / "ready-5-g7").exists())  # the lab consumes it when the command succeeds

    def test_train_passes_keep_rule_to_promote(self):
        marker_dir = self.root / "train"
        self.assertEqual(0, gen0.main(["build", "--prior", str(self.prior), "--agent-root", str(self.agents), "--candidate-generation", "7",
                                       "--why", "test", "--no-export", "--marker-dir", str(marker_dir)]))
        seeds = str(self.root / "E7-test.json")
        args = ["train", "--marker-dir", str(marker_dir), "--shards-root", str(self.root), "--agent-root", str(self.agents), "--promote",
                "--emu-python", "python", "--seeds-file", seeds]
        with mock.patch.dict(os.environ):
            os.environ.pop(gen_loop.KEEP_RULE_FLAG, None)
            with mock.patch.object(gen_loop, "train_generation") as tg, mock.patch.object(gen_loop, "run_promote_subprocess", return_value=0) as rp:
                self.assertEqual(0, gen0.main([*args, "--review-guard"]))
                self.assertIsNone(rp.call_args[1]["keep_rule"])  # no flag: nothing passed on, the child reads the variable or v1
                self.assertEqual(0, gen0.main([*args, "--review-guard", "--keep-rule", "v2"]))
                self.assertEqual(rp.call_args[1]["keep_rule"], "v2")
                self.assertEqual(0, gen0.main([*args, "--keep-rule", "v1"]))
                self.assertEqual(rp.call_args[1]["keep_rule"], "v1")
                tg.assert_not_called()
            # the cap before training: v2 has it in the rule, so two judged candidates of the generation stop the round without --review-guard
            tsv = self.agents / "promotions.tsv"
            for i in range(2):
                gen_loop.append_promotion(tsv, {"cand_gen": 7, "seed_block": f"E7-{i}", "cand_sha": f"sha-{i}", "decision": "rejected"})
            with mock.patch.object(gen_loop, "train_generation") as tg, mock.patch.object(gen_loop, "run_promote_subprocess", return_value=0) as rp:
                self.assertEqual(0, gen0.main(args))  # v1 without the review guard has no cap: the candidate goes on to be judged
                rp.assert_called_once()
                rp.reset_mock()
                self.assertEqual(0, gen0.main([*args, "--keep-rule", "v2"]))  # the marker is renamed done-, nothing is judged or trained
                rp.assert_not_called()
                tg.assert_not_called()
                self.assertTrue((marker_dir / "done-5-g7").exists())

    def test_a_judged_built_candidate_is_not_judged_again_or_trained_over(self):
        cand = self.build()
        (cand / "promotion.json").write_text("{}")
        marker_dir = self.root / "train"
        marker_dir.mkdir()
        (marker_dir / "ready-5-g7").write_text("x")
        with mock.patch.object(gen_loop, "train_generation") as tg, mock.patch.object(gen_loop, "run_promote_subprocess") as rp:
            rc = gen0.main(self.train_args(marker_dir))
        self.assertEqual(0, rc)
        tg.assert_not_called()
        rp.assert_not_called()
        self.assertEqual(["done-5-g7"], [p.name for p in marker_dir.iterdir()])

    def test_the_build_command_refuses_a_marker_that_was_written_before(self):
        marker_dir = self.root / "train"
        marker_dir.mkdir()
        (marker_dir / "done-5-g7").write_text("x")
        rc = gen0.main(["build", "--prior", str(self.prior), "--agent-root", str(self.agents), "--candidate-generation", "7",
                        "--why", "test", "--no-export", "--marker-dir", str(marker_dir)])
        self.assertEqual(2, rc)
        self.assertFalse((self.agents / "gen7").exists())


class StaleEnvironmentTests(unittest.TestCase):
    def test_generation_0_search_kwargs_are_the_base_ones_with_the_variable_unset(self):
        import search_configs

        with mock.patch.dict(os.environ):
            os.environ.pop("SPIRELINE_GEN_DIR", None)
            pos = search_configs.Position("play", 3, 1.0, False)
            self.assertEqual(search_configs.get("a3-v1").search_kwargs(pos),
                             {"max_sims": 30, "min_child_sims": 1, "c": 0.2, "confirm_pairs": 0, "time_budget_s": 120.0,
                              "resample": True})

    def test_the_variable_reaches_the_lockstep_player_and_never_the_emulator_driver(self):
        import search_configs

        cfg = search_configs.get("a3-v1")
        with mock.patch.dict(os.environ, {"SPIRELINE_GEN_DIR": str(trained_candidate()[2])}):
            self.assertIn("prior_fn", cfg.search_kwargs())
            self.assertNotIn("prior_fn", cfg.search_kwargs(env_prior=False))

    @unittest.skipUnless(HAVE_NATIVE, "needs the native library")
    def test_a_stale_variable_gives_the_generation_0_arm_no_prior(self):
        import live_search

        seen = {}

        class Stop(Exception):
            pass

        def fake(handle, **kw):
            seen.update(kw)
            raise Stop

        with mock.patch.dict(os.environ, {"SPIRELINE_GEN_DIR": str(trained_candidate()[2])}), \
                mock.patch.object(live_search, "search_live", fake):
            row = gen0._run_safe("SJ9DHBJHPE", 0, "a3-v1", 10, None)
        self.assertIn("Stop", row["error"])
        self.assertNotIn("prior_fn", seen)
        self.assertNotIn("c_puct", seen)


class GenerationTagAtReadTests(unittest.TestCase):
    def test_a_part_whose_tag_disagrees_with_its_manifest_is_refused(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as d:
            sink = gen_loop.ShardSink(Path(d) / "job", 2, config="a3-v1", base_sims=30, ascension=10, agent={}, part_runs=2)
            sink.add(fake_record("AAAAAAAAAA", 2), fake_run_arrays(rng))
            sink.close()
            mans = gen_loop.window_manifests(Path(d), 2)
            self.assertEqual(gen_loop.build_training_set(mans)["lengths"].size, 12)
            part = Path(d) / "job" / "shards" / mans[0][1]["combat_parts"][0]
            blob = gen_loop.read_part(part)
            blob["generation"] = np.array([0], dtype=np.int64)  # a part from another generation under this manifest
            np.savez_compressed(part, **blob)
            with self.assertRaisesRegex(ValueError, "tagged generation 0"):
                gen_loop.build_training_set(gen_loop.window_manifests(Path(d), 2))


def stub_candidate(root: Path, name: str, sha: str, generation: int = 5) -> Path:
    """A candidate directory the agent loader accepts, with a policy sha; its prior is never loaded (rows are stubbed)."""
    c = root / name
    c.mkdir(parents=True)
    (c / "model.json").write_text(json.dumps({"policy_sha256": sha}))
    (c / "combat_policy.pt").write_bytes(b"x")
    (c / "agent.json").write_text(json.dumps({"generation": generation, "macro": "rules"}))
    return c


def arm_rows_like(seeds, floor_of, kill_from=None):
    return [{"seed": s, "floor": floor_of(i), "win": False, "end": "death", "decisions": 10, "wall": 1.0, "crash": False}
            for i, s in enumerate(seeds)]


class PromoteEndToEndTests(unittest.TestCase):
    """`promote` with the arms stubbed: the tsv line, current.json and the marker only on a promotion, the refusal of a reused
    block, the paired-count guard."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.seeds = [f"S{i:09d}" for i in range(300)]
        self.seed_file = self.root / "seeds" / "Etest-1.json"
        self.seed_file.parent.mkdir()
        self.seed_file.write_text(json.dumps({"seeds": self.seeds}))
        self.agents = self.root / "agent"
        self.agents.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def run_promote(self, cand, better: bool, cand_rows_cut=None, lab=None):
        base_rows = arm_rows_like(self.seeds, lambda i: 3 + i % 12)
        cand_rows = arm_rows_like(self.seeds, (lambda i: 5 + i % 12) if better else (lambda i: 3 + i % 12))
        if better:
            for r in cand_rows[:80]:
                r["floor"] = 18
        if cand_rows_cut:
            cand_rows = cand_rows[:cand_rows_cut]

        def fake_run_arm(seeds, arm_dir, agent, config, procs, ascension=10):
            return cand_rows if agent.prior_dir else base_rows

        with mock.patch.object(gen_loop, "run_arm", fake_run_arm):
            return gen_loop.promote(cand, None, self.seed_file, agent_root=self.agents, lab=lab, boot=500)

    def test_a_promotion_writes_the_tsv_line_current_json_and_the_marker(self):
        cand = stub_candidate(self.root, "gen5c", "sha-a")
        lab = self.root / "lab"
        doc = self.run_promote(cand, better=True, lab=lab)
        self.assertEqual(doc["decision_text"], "promoted")
        self.assertEqual((doc["f7"], doc["matched_wall_clock"], doc["cand_sha"]), ("not run", False, "sha-a"))
        rows = gen_loop.read_promotions(self.agents / "promotions.tsv")
        self.assertEqual([(r["seed_block"], r["decision"], r["cand_sha"], r["cand_dir"]) for r in rows],
                         [("Etest-1", "promoted", "sha-a", "gen5c")])
        cur = json.loads((self.agents / "current.json").read_text())
        self.assertEqual((cur["generation"], Path(cur["dir"]).name), (5, "gen5c"))
        self.assertTrue((lab / "score" / "pending").is_file())
        self.assertTrue((cand / "promotion.json").is_file())

    def test_a_rejection_leaves_current_json_and_the_marker_alone(self):
        cand = stub_candidate(self.root, "gen5c", "sha-a")
        lab = self.root / "lab"
        doc = self.run_promote(cand, better=False, lab=lab)
        self.assertEqual(doc["decision_text"], "rejected")
        self.assertEqual(len(gen_loop.read_promotions(self.agents / "promotions.tsv")), 1)
        self.assertFalse((self.agents / "current.json").exists())
        self.assertFalse((lab / "score" / "pending").exists())

    def test_a_block_that_judged_another_candidate_is_refused_and_the_same_one_may_resume(self):
        self.run_promote(stub_candidate(self.root, "c1", "sha-a"), better=False)
        with self.assertRaisesRegex(ValueError, "already judged another candidate"):
            self.run_promote(stub_candidate(self.root, "c2", "sha-b"), better=False)
        self.run_promote(self.root / "c1", better=False)  # the same sha: a resume, allowed
        # an old tsv row (no sha column) counts as another candidate
        old = self.agents / "promotions.tsv"
        old.write_text("written\tcand_gen\tbase_gen\tseed_block\n2026\t1\t0\tEtest-1\n")
        with self.assertRaisesRegex(ValueError, "already judged"):
            self.run_promote(self.root / "c1", better=False)

    def test_rejudge_applies_the_current_rule_to_the_cached_arms_without_playing(self):
        cand = stub_candidate(self.root, "gen5c", "sha-a")
        base_rows = arm_rows_like(self.seeds, lambda i: 3 + i % 12)
        cand_rows = arm_rows_like(self.seeds, lambda i: 5 + i % 12)
        for arm_dir, rows in ((cand / "arms" / "Etest-1-a3-v1", cand_rows), (self.agents / "gen0" / "arms" / "Etest-1-a3-v1", base_rows)):
            arm_dir.mkdir(parents=True)
            (arm_dir / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (cand / "promotion.json").write_text(json.dumps({"decision_text": "rejected", "written": "2026-10-23T13:30:52Z"}))
        lab = self.root / "lab"
        with mock.patch.object(gen0, "run_block", side_effect=AssertionError("must not play again")):
            doc = gen_loop.rejudge(cand, None, self.seed_file, agent_root=self.agents, note="D24", lab=lab, boot=500)
        self.assertEqual((doc["decision_text"], doc["decision"]["rule"]), ("promoted", "D24"))
        self.assertEqual(doc["rejudged"], {"note": "D24", "earlier_decision": "rejected"})
        self.assertTrue((cand / "promotion-rejected-2026-10-23.json").is_file())
        self.assertEqual(json.loads((self.agents / "current.json").read_text())["note"], "D24")
        self.assertTrue((lab / "score" / "pending").is_file())
        self.assertEqual([r["decision"] for r in gen_loop.read_promotions(self.agents / "promotions.tsv")], ["promoted"])
        cut = arm_rows_like(self.seeds[:100], lambda i: 4)
        (cand / "arms" / "Etest-1-a3-v1" / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cut))
        with self.assertRaisesRegex(ValueError, "only 100 of 300 seeds are paired"):
            gen_loop.rejudge(cand, None, self.seed_file, agent_root=self.agents, note="D24", boot=500)

    def test_a_partial_arm_cannot_promote(self):
        cand = stub_candidate(self.root, "gen5c", "sha-a")
        with self.assertRaisesRegex(ValueError, "only 200 of 300 seeds are paired"):
            self.run_promote(cand, better=True, cand_rows_cut=200)
        self.assertFalse((self.agents / "current.json").exists())
        self.assertFalse((self.agents / "promotions.tsv").exists())

    def test_an_old_tsv_header_is_rewritten_with_the_new_columns(self):
        t = self.agents / "promotions.tsv"
        t.write_text("written\tcand_gen\tbase_gen\tseed_block\tseeds\td_floor\tfloor_lower\tp_floor\tboss_cand\tboss_base\tp_boss\t"
                     "wins_cand\twins_base\tdecision\n2026\t1\t0\tE1-x\t2000\t0.1\t0.0\t0.5\t1\t2\t0.9\t0\t0\trejected\n")
        gen_loop.append_promotion(t, {"seed_block": "E9", "cand_sha": "s"})
        rows = gen_loop.read_promotions(t)
        self.assertEqual([(r["seed_block"], r["cand_sha"]) for r in rows], [("E1-x", ""), ("E9", "s")])
        self.assertEqual(rows[0]["decision"], "rejected")

    def test_run_arm_reuses_a_finished_arm_of_the_same_agent_and_block(self):
        agent = gen_loop.Agent()
        arm = self.root / "arm"
        arm.mkdir()
        rows = arm_rows_like(self.seeds, lambda i: 4)
        (arm / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (arm / "summary.json").write_text(json.dumps({"config": "a3-v1", "agent": agent.describe()}))
        with mock.patch.object(gen0, "run_block", side_effect=AssertionError("must not play again")):
            self.assertEqual(len(gen_loop.run_arm(self.seeds, arm, agent, "a3-v1", 2)), 300)


class KeepRuleV2Tests(unittest.TestCase):
    """The keep test v2 in the generation loop (tools/lab/keep_v2.py, docs/rl/KEEP_TEST_V2.md): the ledger's nine columns, `rejudge` on saved
    arms (no attempt spent, a dry run that writes only to its own folder), and the default rule staying v1."""

    NINE = ("rule", "frontier", "route", "k2_cand", "k2_base", "d_deep", "deep_lower", "p_deep", "floor_lower95")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ("EXAM_DIR", "LOOP_ROOT"):  # the judgement's human-bar line reads the gym's folder: not the real one in a test
            patcher = mock.patch.object(fe, name, self.root / "no-gym")
            patcher.start()
            self.addCleanup(patcher.stop)
        self.agents = self.root / "agent"
        self.agents.mkdir()
        self.seeds, self.cand_rows, self.base_rows = fx.arms()
        self.seed_file = self.root / "seeds" / "Etest-2.json"
        self.seed_file.parent.mkdir()
        self.seed_file.write_text(json.dumps({"seeds": self.seeds}))

    def tearDown(self):
        self.tmp.cleanup()

    def lineage(self) -> None:
        """Generation 3 promoted on Eprev with the Act 2 boss beaten 3 times in 100: the incumbent's own arm that sets the frontier at B2."""
        prev = [fx.run_row(f"P{i:09d}", 2 if i < 60 else 1 if i < 1000 else 0, np.random.default_rng(i)) for i in range(2000)]
        self.write_arm(self.agents / "gen3" / "candidate", "Eprev", prev)
        gen_loop.append_promotion(self.agents / "promotions.tsv",
                                  {"cand_gen": 3, "base_gen": 2, "seed_block": "Eprev", "seeds": 2000, "boss_cand": 1000, "boss_base": 900,
                                   "decision": "promoted", "cand_sha": "sha-3", "cand_dir": "candidate", "attempt": 1, "drift": "clear"})

    @staticmethod
    def write_arm(folder: Path, name: str, rows: list[dict]) -> None:
        arm = folder / "arms" / f"{name}-a3-v1"
        arm.mkdir(parents=True)
        (arm / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def judged_candidate(self, others: int = 0) -> Path:
        """Candidate sha-a of generation 4, refused by the floor test on Etest-2 (its row is in the ledger, its arms are on disk), and
        `others` more candidates of generation 4 judged on other blocks."""
        self.lineage()
        cand = stub_candidate(self.root, "c1", "sha-a", generation=4)
        self.write_arm(cand, "Etest-2", self.cand_rows)
        self.write_arm(self.agents / "gen0", "Etest-2", self.base_rows)
        tsv = self.agents / "promotions.tsv"
        for i in range(others):
            gen_loop.append_promotion(tsv, {"cand_gen": 4, "base_gen": 3, "seed_block": f"Eother{i}", "seeds": 200, "boss_cand": 90, "boss_base": 100,
                                            "decision": "rejected", "cand_sha": f"sha-o{i}", "cand_dir": f"candidate-o{i}", "attempt": i + 1})
        gen_loop.append_promotion(tsv, {"cand_gen": 4, "base_gen": 3, "seed_block": "Etest-2", "seeds": 2000, "d_floor": "-0.3060",
                                        "floor_lower": "-0.7660", "p_floor": "0.93900", "boss_cand": 965, "boss_base": 1043, "p_boss": "0.99944",
                                        "decision": "rejected", "cand_sha": "sha-a", "cand_dir": "c1", "attempt": others + 1, "drift": "clear",
                                        "guard": "kill FIRES"})
        (cand / "promotion.json").write_text(json.dumps({"decision_text": "rejected", "written": "2026-11-05T17:49:35Z", "seed_block": "Etest-2"}))
        return cand

    def rejudge(self, cand: Path, **kw):
        with mock.patch.object(gen0, "run_block", side_effect=AssertionError("must not play again")):
            return gen_loop.rejudge(cand, None, self.seed_file, agent_root=self.agents, note="keep test v2", boot=500, keep_rule="v2", **kw)

    # ---------------------------------------------------------------- the ledger

    def test_ledger_v2_columns_append_and_old_rows_read_as_v1(self):
        cols = gen_loop.TSV_COLUMNS
        self.assertEqual(cols[-len(self.NINE):], self.NINE)  # appended after `macro`, nothing moved
        self.assertEqual(cols[cols.index("macro") + 1:], self.NINE)
        t = self.root / "promotions.tsv"
        old_cols = cols[:cols.index("rule")]
        old = {c: "x" for c in old_cols} | {"seed_block": "E9-old", "decision": "promoted", "seeds": "2000", "boss_cand": "928", "boss_base": "829"}
        t.write_text("\t".join(old_cols) + "\n" + "\t".join(old[c] for c in old_cols) + "\n", encoding="utf-8")
        rows = gen_loop.read_promotions(t)
        self.assertNotIn("rule", rows[0])  # the file before the columns existed
        self.assertEqual(gen_loop.row_rule(rows[0]), "v1")
        cells = {"rule": "v2", "frontier": "B2", "route": "A", "k2_cand": 107, "k2_base": 49, "d_deep": "3.0500", "deep_lower": "1.4500",
                 "p_deep": "0.00005", "floor_lower95": "-0.6300"}
        gen_loop.append_promotion(t, {"seed_block": "E12-new", "decision": "promoted", "seeds": 2000, "boss_cand": 965, "boss_base": 1043, **cells})
        lines = t.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0].split("\t"), list(cols))  # the old header was rewritten to the new columns
        self.assertTrue(all(len(x.split("\t")) == len(cols) for x in lines))  # and every row has every cell
        rows = gen_loop.read_promotions(t)
        self.assertEqual([r["seed_block"] for r in rows], ["E9-old", "E12-new"])
        self.assertEqual([r[c] for r in rows[:1] for c in self.NINE], [""] * 9)  # old rows keep their values and read empty here: v1
        self.assertEqual(rows[0]["boss_cand"], "928")
        self.assertEqual((gen_loop.row_rule(rows[0]), gen_loop.row_rule(rows[1])), ("v1", "v2"))
        self.assertEqual({c: rows[1][c] for c in self.NINE}, {c: str(v) for c, v in cells.items()})
        # the drift check's lineage reads the Act 1 columns of both kinds of row alike
        self.assertEqual(gen_loop.lineage_blocks(rows), [(928, 2000), (965, 2000)])
        # a file that already has the columns is appended to as it stands; a v1 row leaves the nine cells empty
        gen_loop.append_promotion(t, {"seed_block": "E13-v1", "decision": "rejected", "seeds": 200})
        self.assertEqual(t.read_text(encoding="utf-8").splitlines()[0].split("\t"), list(cols))
        self.assertEqual([gen_loop.row_rule(r) for r in gen_loop.read_promotions(t)], ["v1", "v2", "v1"])
        self.assertEqual(gen_loop.read_promotions(t)[2]["route"], "")

    # ---------------------------------------------------------------- rejudge

    def test_rejudge_v2_from_saved_rows_spends_no_attempt(self):
        cand = self.judged_candidate(others=2)  # a third candidate of generation 4: `promote` would refuse it before playing a seed
        tsv = self.agents / "promotions.tsv"
        before = gen_loop.read_promotions(tsv)
        self.assertEqual(gen_loop.attempts_before(before, 4, "sha-new"), 3)
        self.assertEqual(gen_loop.attempts_before(before, 4, "sha-a"), 2)
        doc = self.rejudge(cand, lab=self.root / "lab")
        k = doc["keep_v2"]
        self.assertEqual(doc["decision_text"], "promoted")
        self.assertEqual((doc["decision"]["rule"], k["route"], k["frontier"]["name"]), (gen_loop.keep_v2_module().RULE, "A", "B2"))
        self.assertEqual(k["frontier"]["pooled_kills"][1], 60)  # read from the incumbent's own arm on Eprev
        self.assertEqual(k["kills"]["B2"], [107, 49])
        self.assertEqual((k["attempt"]["attempt"], k["attempt"]["cap"], k["attempt"]["allowed"]), (3, 2, True))  # attempt 3 of 2, and a re-read is allowed
        self.assertEqual(doc["rejudged"], {"note": "keep test v2", "earlier_decision": "rejected"})
        self.assertEqual(doc["keep_rule"], "v2")
        # the ledger: a second row for the block, v2 columns filled, the same candidate: the count of attempts is as it was
        after = gen_loop.read_promotions(tsv)
        self.assertEqual(len(after), len(before) + 1)
        new = after[-1]
        self.assertEqual((new["seed_block"], new["decision"], new["rule"], new["frontier"], new["route"], new["k2_cand"], new["k2_base"]),
                         ("Etest-2", "promoted", "v2", "B2", "A", "107", "49"))
        self.assertEqual((new["boss_cand"], new["boss_base"], new["attempt"]), ("965", "1043", "3"))  # Act 1 columns as ever
        self.assertEqual(gen_loop.attempts_before(after, 4, "sha-new"), 3)  # nothing was spent
        self.assertEqual(gen_loop.attempts_before(after, 4, "sha-a"), 2)
        self.assertEqual(gen_loop.attempts_spent(tsv, 4, True), 3)  # the cap still stands for a new candidate
        self.assertEqual(json.loads((self.agents / "current.json").read_text())["note"], "keep test v2")
        self.assertTrue((self.root / "lab" / "score" / "pending").is_file())
        self.assertTrue((cand / "promotion-rejected-2026-11-05.json").is_file())  # the earlier record kept beside the new one
        self.assertEqual(json.loads((cand / "promotion.json").read_text())["keep_v2"]["route"], "A")
        # the drift check reads the rows before this block, not its own: Eprev (1000 of 2000) and the two other candidates' blocks (100 of 200 each)
        self.assertEqual((k["drift"]["pooled_kills"], k["drift"]["pooled_n"], k["drift"]["verdict"]), (1200, 2400, "clear"))
        self.assertEqual(new["drift"], "clear")

    def test_rejudge_v2_carries_the_report_only_human_bar_line_and_the_command_prints_it(self):
        loop, exam, _ = human_gym_folder(self.root)
        cand = self.judged_candidate()  # its claimed network (sha-a) is no gym player's: the gym incumbent's reading, labelled so
        out = self.root / "dry"
        with mock.patch.object(fe, "EXAM_DIR", exam), mock.patch.object(fe, "LOOP_ROOT", loop):
            doc = self.rejudge(cand, out=out)
            line = doc["human_bar"]["line"]
            self.assertTrue(line.startswith("human bar: fight network sha-a (no gym player holds these bytes); the gym incumbent v2's reading, "
                                            "not this network's: survival "), line)
            self.assertIn("on 39 of 39 starts, 156 games, engine e1, games file written ", line)
            self.assertEqual(json.loads((out / "promotion.json").read_text())["human_bar"]["line"], line)
            self.assertEqual(doc["decision_text"], "promoted")  # the keep test's verdict, as without the line
            self.assertEqual(doc["keep_v2"]["route"], "A")
            args = SimpleNamespace(cand=str(cand), base=None, seeds_file=str(self.seed_file), agent_root=str(self.agents), config="a3-v1",
                                   note="keep test v2", lab=None, review_guard=None, keep_rule="v2", out=str(self.root / "dry2"))
            buf = io.StringIO()
            with mock.patch.object(gen0, "_setup"), mock.patch.object(gen0, "run_block", side_effect=AssertionError("must not play")):
                with contextlib.redirect_stdout(buf):
                    self.assertEqual(gen0.rejudge_cmd(args), 0)
        printed = json.loads(buf.getvalue())
        self.assertEqual(printed["human_bar"], line)
        self.assertEqual(printed["keep_v2"]["route"], "A")
        # the ledger row is as it was: no cell of it names the human bar
        row = gen_loop.read_promotions(out / "promotions.tsv")[-1]
        self.assertFalse(any("human" in k or "human" in v for k, v in row.items()))

    def test_rejudge_v2_dry_run_writes_only_to_its_own_folder(self):
        cand = self.judged_candidate()
        snapshot = {p: p.read_bytes() for p in sorted(self.root.rglob("*")) if p.is_file()}
        out = self.root / "dry"
        doc = self.rejudge(cand, out=out)
        self.assertEqual(doc["decision_text"], "promoted")
        after = {p: p.read_bytes() for p in sorted(self.root.rglob("*")) if p.is_file() and out not in p.parents}
        self.assertEqual(after, snapshot)  # the agent folder and the candidate's folder are byte for byte as they were
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["current.json", "promotion.json", "promotions.tsv"])
        rows = gen_loop.read_promotions(out / "promotions.tsv")
        self.assertEqual(len(rows), len(gen_loop.read_promotions(self.agents / "promotions.tsv")) + 1)
        self.assertEqual((rows[-1]["rule"], rows[-1]["route"], rows[-1]["decision"]), ("v2", "A", "promoted"))
        self.assertEqual(json.loads((out / "promotion.json").read_text())["keep_v2"]["route"], "A")
        self.assertEqual(json.loads((out / "current.json").read_text())["generation"], 4)
        self.assertFalse((self.agents / "current.json").exists())
        with self.assertRaisesRegex(ValueError, "no lab marker"):  # a dry run names no lab folder
            self.rejudge(cand, out=out, lab=self.root / "lab")
        self.assertFalse((self.root / "lab").exists())
        # run twice: the ledger copy is made afresh, so the dry run is repeatable
        self.rejudge(cand, out=out)
        self.assertEqual(len(gen_loop.read_promotions(out / "promotions.tsv")), len(rows))

    def test_a_refused_v2_rejudge_names_the_blocking_check(self):
        cand = self.judged_candidate()
        self.cand_rows[:] = fx.with_bad(self.cand_rows, 8)  # eight crashed seeds against none: the crash guard
        (cand / "arms" / "Etest-2-a3-v1" / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.cand_rows), encoding="utf-8")
        doc = self.rejudge(cand, out=self.root / "dry")
        self.assertEqual(doc["decision_text"], "rejected")
        self.assertEqual(doc["keep_v2"]["blocking"], ["crash"])
        self.assertFalse(doc["decision"]["checks"]["crash"])
        self.assertFalse((self.root / "dry" / "current.json").exists())

    # ---------------------------------------------------------------- the default

    def test_keep_rule_default_v1_is_unchanged(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(gen_loop.KEEP_RULE_FLAG, None)
            self.assertEqual(gen_loop.KEEP_RULE_FLAG, "SPIRELINE_KEEP_RULE")
            self.assertEqual(gen_loop.keep_rule_of(), "v1")
            self.assertEqual(gen_loop.keep_rule_of(None), "v1")
            self.assertEqual((gen_loop.keep_rule_of("v1"), gen_loop.keep_rule_of(" V2 ")), ("v1", "v2"))
            os.environ[gen_loop.KEEP_RULE_FLAG] = "v2"
            self.assertEqual(gen_loop.keep_rule_of(), "v2")
            self.assertEqual(gen_loop.keep_rule_of("v1"), "v1")  # the flag beats the variable
            os.environ[gen_loop.KEEP_RULE_FLAG] = ""
            self.assertEqual(gen_loop.keep_rule_of(), "v1")
            for bad in ("v3", "on", "2"):
                os.environ[gen_loop.KEEP_RULE_FLAG] = bad
                with self.assertRaises(ValueError):
                    gen_loop.keep_rule_of()
            os.environ.pop(gen_loop.KEEP_RULE_FLAG)
            p = gen0.build_parser()
            for cmd, extra in (("promote", ["--cand", "c", "--agent-root", "a"]), ("rejudge", ["--cand", "c", "--seeds-file", "s", "--agent-root", "a", "--note", "n"]),
                               ("train", ["--marker-dir", "m", "--shards-root", "s", "--agent-root", "a"])):
                self.assertIsNone(p.parse_args([cmd, *extra]).keep_rule, cmd)
                self.assertEqual(p.parse_args([cmd, *extra, "--keep-rule", "v2"]).keep_rule, "v2", cmd)
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    p.parse_args([cmd, *extra, "--keep-rule", "v3"])
            self.assertIsNone(p.parse_args(["rejudge", "--cand", "c", "--seeds-file", "s", "--agent-root", "a", "--note", "n"]).out)
        # the child process is told only when the caller names a rule
        ran = []
        with mock.patch.object(gen_loop.subprocess, "run", lambda cmd, check=False: ran.append(cmd) or SimpleNamespace(returncode=0)):
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None)
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None, keep_rule="v2")
        self.assertNotIn("--keep-rule", ran[0])
        self.assertEqual(ran[1][ran[1].index("--keep-rule") + 1], "v2")
        # a judgement without the rule is D24's, with the columns empty and no v2 key anywhere
        seeds = [f"S{i:09d}" for i in range(300)]
        seed_file = self.root / "seeds" / "Etest-1.json"
        seed_file.write_text(json.dumps({"seeds": seeds}))
        base_rows = arm_rows_like(seeds, lambda i: 3 + i % 12)
        cand_rows = arm_rows_like(seeds, lambda i: 5 + i % 12)
        for r in cand_rows[:80]:
            r["floor"] = 18
        cand = stub_candidate(self.root, "gen5c", "sha-z")
        with mock.patch.dict(os.environ), mock.patch.object(gen_loop, "run_arm", lambda s, a, ag, c, p, ascension=10: cand_rows if ag.prior_dir else base_rows):
            os.environ.pop(gen_loop.KEEP_RULE_FLAG, None)
            os.environ.pop(gen_loop.REVIEW_GUARD_FLAG, None)
            doc = gen_loop.promote(cand, None, seed_file, agent_root=self.agents, boot=500)
            self.assertEqual(doc["decision_text"], "promoted")
            self.assertEqual(doc["decision"]["rule"], "D24")
            self.assertNotIn("keep_v2", doc)
            self.assertNotIn("keep_rule", doc)
            new = gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]
            self.assertEqual([new[c] for c in self.NINE], [""] * 9)
            self.assertEqual(gen_loop.row_rule(new), "v1")
            self.assertEqual(new["guard"], "off")
        # the comparison without a verdict of the rule is the one it always was
        res = gen_loop.compare_arms(cand_rows, base_rows, gen_loop.ALPHA, 500)
        self.assertEqual(res["decision"]["rule"], "D24")
        self.assertNotIn("keep_v2", res)
        self.assertEqual(gen_loop.attempts_spent(self.agents / "promotions.tsv", 5, None), None)  # the cap is the review guard's alone under v1

    def test_the_cap_before_training_holds_under_v2_without_the_review_guard(self):
        tsv = self.agents / "promotions.tsv"
        for i in range(2):
            gen_loop.append_promotion(tsv, {"cand_gen": 6, "seed_block": f"E{i}", "cand_sha": f"sha-{i}", "decision": "rejected"})
        with mock.patch.dict(os.environ):
            os.environ.pop(gen_loop.REVIEW_GUARD_FLAG, None)
            self.assertIsNone(gen_loop.attempts_spent(tsv, 6, None))  # v1 with the flag off: no cap
            self.assertEqual(gen_loop.attempts_spent(tsv, 6, None, keep_rule="v2"), 2)  # v2 carries the cap in the rule
            self.assertIsNone(gen_loop.attempts_spent(tsv, 7, None, keep_rule="v2"))


class FreshBlockTests(unittest.TestCase):
    def test_resolve_e_block_gives_a_fresh_r_block_to_a_second_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            tpl = str(Path(d) / "E{k}-{date}.json")
            tsv = Path(d) / "promotions.tsv"
            (Path(d) / "E1-2026-10-23.json").write_text(json.dumps({"seeds": ["A"]}))
            self.assertEqual(gen_loop.resolve_e_block(tpl, 1, tsv=tsv, cand_sha="aaa").name, "E1-2026-10-23.json")  # unused
            gen_loop.append_promotion(tsv, {"seed_block": "E1-2026-10-23", "cand_sha": "aaa", "decision": "rejected"})
            self.assertEqual(gen_loop.resolve_e_block(tpl, 1, tsv=tsv, cand_sha="aaa").name, "E1-2026-10-23.json")  # same sha
            r1 = gen_loop.resolve_e_block(tpl, 1, n=40, tsv=tsv, cand_sha="bbb")
            self.assertRegex(r1.name, r"^E1-\d{4}-\d\d-\d\d-r1\.json$")
            self.assertEqual(gen0.load_seeds(r1), gen0.e_block(r1.stem, 40))
            self.assertFalse(set(gen0.load_seeds(r1)) & set(gen0.e_block("E1-2026-10-23", 2000)))
            gen_loop.append_promotion(tsv, {"seed_block": r1.stem, "cand_sha": "bbb", "decision": "rejected"})
            self.assertEqual(gen_loop.resolve_e_block(tpl, 1, n=40, tsv=tsv, cand_sha="bbb"), r1)
            r2 = gen_loop.resolve_e_block(tpl, 1, n=40, tsv=tsv, cand_sha="ccc")
            self.assertTrue(r2.name.endswith("-r2.json"))

    def test_known_seed_sets_hold_runs_trains_reserved_seeds(self):
        sets = gen_loop.known_seed_sets(ROOT / "referee" / "seeds", None, "E1-2026-10-23")
        self.assertIn("run_train eval_seeds", sets)
        self.assertIn("run_train test_seeds", sets)
        e1 = set(gen0.load_seeds(ROOT / "referee" / "seeds" / "E1-2026-10-23.json"))
        self.assertFalse(e1 & set(sets["run_train eval_seeds"]) | e1 & set(sets["run_train test_seeds"]))


class TrainingSeedGuardTests(unittest.TestCase):
    def test_training_refuses_a_window_that_holds_a_benchmark_seed(self):
        e0 = gen0.load_seeds(ROOT / "referee" / "seeds" / "E0-2026-10-23.json")[0]
        rng = np.random.default_rng(6)
        with tempfile.TemporaryDirectory() as d:
            sink = gen_loop.ShardSink(Path(d) / "runs" / "job", 0, config="a3-v1", base_sims=30, ascension=10, agent={}, part_runs=2)
            sink.add(fake_record(e0, 0), fake_run_arrays(rng))
            sink.close()
            with self.assertRaisesRegex(ValueError, "also in E0-2026-10-23"):
                gen_loop.train_generation(0, Path(d) / "runs", Path(d) / "agent", device="cpu", epochs=1, vrun=False, export=False)
            self.assertFalse((Path(d) / "agent" / "gen1").exists())  # nothing was created or trained

    def test_a_reserved_run_train_seed_is_refused_too(self):
        reserved = gen0.load_seeds(ROOT / "rl" / "run_train" / "eval_seeds.json")[0]
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as d:
            sink = gen_loop.ShardSink(Path(d) / "runs" / "job", 0, config="a3-v1", base_sims=30, ascension=10, agent={}, part_runs=2)
            sink.add(fake_record(reserved, 0), fake_run_arrays(rng))
            sink.close()
            with self.assertRaisesRegex(ValueError, "run_train eval_seeds"):
                gen_loop.train_generation(0, Path(d) / "runs", Path(d) / "agent", device="cpu", epochs=1, vrun=False, export=False)

    def test_the_fixture_windows_pass_the_guard(self):
        self.assertTrue(trained_candidate()[2].is_dir())  # its seeds are S...: no benchmark block holds them


class RetrainAfterRejectionTests(unittest.TestCase):
    def _runs(self, root: Path, gen: int, n: int, tag: str):
        sink = gen_loop.ShardSink(root / f"job-{tag}", gen, config="a3-v1", base_sims=30, ascension=10, agent={})
        for i in range(n):
            sink.add(fake_record(f"R{gen}{tag}{i:05d}"[:10], gen), None)
        sink.close()

    def test_a_rejection_buys_the_generation_another_budget_and_a_promotion_does_not(self):
        with tempfile.TemporaryDirectory() as d:
            lab, runs, tsv = Path(d) / "lab", Path(d) / "runs", Path(d) / "promotions.tsv"
            self._runs(runs, 1, 5, "a")
            self.assertEqual(gen_loop.mark_ready(lab, 1, runs, budget=5, tsv=tsv), lab / "train" / "ready-1")
            (lab / "train" / "ready-1").rename(lab / "train" / "done-1")  # what the lab does after the job
            self._runs(runs, 1, 5, "b")  # self-play went on at generation 1 while the candidate was judged
            self.assertIsNone(gen_loop.mark_ready(lab, 1, runs, budget=5, tsv=tsv))  # no verdict yet
            gen_loop.append_promotion(tsv, {"cand_gen": 2, "decision": "promoted", "seed_block": "E1", "cand_sha": "a"})
            self.assertIsNone(gen_loop.mark_ready(lab, 1, runs, budget=5, tsv=tsv))  # promoted: nothing to retrain
            gen_loop.append_promotion(tsv, {"cand_gen": 2, "decision": "rejected", "seed_block": "E1", "cand_sha": "b"})
            self.assertEqual(gen_loop.mark_ready(lab, 1, runs, budget=5, tsv=tsv), lab / "train" / "ready-1-r1")
            self.assertIsNone(gen_loop.mark_ready(lab, 1, runs, budget=5, tsv=tsv))  # once
            (lab / "train" / "ready-1-r1").rename(lab / "train" / "done-1-r1")
            self._runs(runs, 1, 4, "c")  # 14 runs: short of the third budget of 5
            gen_loop.append_promotion(tsv, {"cand_gen": 2, "decision": "rejected", "seed_block": "E1-r1", "cand_sha": "c"})
            self.assertIsNone(gen_loop.mark_ready(lab, 1, runs, budget=5, tsv=tsv))
            self._runs(runs, 1, 1, "d")
            self.assertEqual(gen_loop.mark_ready(lab, 1, runs, budget=5, tsv=tsv), lab / "train" / "ready-1-r2")
            self.assertEqual([g for g, _ in gen_loop.ready_markers(lab / "train")], [1])
            self.assertEqual(gen_loop.marker_parts(lab / "train" / "ready-1-r2"), (1, 2))

    def test_markers_are_ordered_by_generation_then_retrain_index(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("ready-12", "ready-4-r1", "ready-4", "done-4", "ready-4-x"):
                (Path(d) / n).write_text("x")
            self.assertEqual([p.name for _, p in gen_loop.ready_markers(Path(d))], ["ready-4", "ready-4-r1", "ready-12"])

    def test_the_train_command_retrains_into_its_own_directory_with_the_larger_window(self):
        with tempfile.TemporaryDirectory() as d:
            marker_dir = Path(d) / "train"
            marker_dir.mkdir()
            (marker_dir / "ready-7-r1").write_text("x")
            with mock.patch.object(gen_loop, "train_generation", return_value=Path(d) / "gen8" / "candidate-r1") as tg:
                rc = gen0.main(["train", "--marker-dir", str(marker_dir), "--shards-root", d, "--agent-root", d])
            self.assertEqual(rc, 0)
            self.assertEqual(tg.call_args[0][0], 7)
            self.assertEqual((tg.call_args[1]["retrain"], tg.call_args[1]["max_fights"]), (1, 2 * gen_loop.MAX_FIGHTS))
            self.assertEqual(gen_loop.MAX_FIGHTS, 32000)  # the train job passes no --max-fights: this is its cap (D44 step E)
            (marker_dir / "ready-7-r1").write_text("x")
            with mock.patch.object(gen_loop, "train_generation", return_value=Path(d) / "gen8" / "candidate-r1") as tg:
                gen0.main(["train", "--marker-dir", str(marker_dir), "--shards-root", d, "--agent-root", d, "--max-fights", "100"])
            self.assertEqual(tg.call_args[1]["max_fights"], 200)  # an explicit cap still wins

    def test_a_retrain_keeps_the_rejected_candidate(self):
        runs, agents, first = trained_candidate()
        again = gen_loop.train_generation(1, runs, agents, device="cpu", epochs=1, max_fights=10, threads=2, vrun=False, export=False,
                                          retrain=1)
        self.assertEqual(again, agents / "gen2" / "candidate-r1")
        self.assertTrue((again / "combat_policy.pt").is_file())
        self.assertTrue((first / "combat_policy.pt").is_file())  # the first candidate is untouched
        self.assertEqual(json.loads((again / "agent.json").read_text())["generation"], 2)


@unittest.skipUnless(HAVE_NATIVE, "needs the native library")
class RecorderTests(unittest.TestCase):
    def test_a_forced_move_is_counted_but_not_recorded(self):
        rec = gen_loop.RunRecorder("AAAAAAAAAA", 0, {}, 30, 0.2, 10)
        rec.fight = {"decisions": 0}
        one = SimpleNamespace(action=0, merged_actions=[], visits=0, value_mean=0.0)
        rec.sample(None, SimpleNamespace(children=[one], sims=0), None)
        self.assertEqual((len(rec.samples.rows), rec.fight["decisions"]), (0, 1))

    def test_validation_is_by_run_the_fights_column_groups_each_runs_samples(self):
        rng = np.random.default_rng(8)
        sizes = [(3, 2), (2, 4), (1, 5)]  # (fights, decisions per fight) of three runs: 6, 8 and 5 samples
        with tempfile.TemporaryDirectory() as d:
            sink = gen_loop.ShardSink(Path(d) / "job", 0, config="a3-v1", base_sims=30, ascension=10, agent={}, part_runs=2)
            for i, (nf, nd) in enumerate(sizes):
                sink.add(fake_record(f"Q{i:09d}", 0), fake_run_arrays(rng, nf, nd))
            sink.close()  # two parts: runs 0-1 and run 2, so run ids repeat across parts before the merge
            data = gen_loop.build_training_set(gen_loop.window_manifests(Path(d), 0), max_fights=1000)
        ids, counts = np.unique(data["fights"], return_counts=True)
        self.assertEqual(sorted(counts.tolist()), [5, 6, 8])  # one id per run, whatever the part it was in
        self.assertEqual(len(ids), 3)
        # train_combat splits on the unique values of this column, so no run is on both sides of the split
        rng2 = np.random.default_rng(0)
        held = set(ids[rng2.permutation(len(ids))][int(0.85 * len(ids)):].tolist())
        in_val = np.isin(data["fights"], list(held))
        for run_id in ids:
            rows = data["fights"] == run_id
            self.assertTrue(in_val[rows].all() or not in_val[rows].any())



# ---------------------------------------------------------------------------------------------------------- a candidate with the run student (2026-11-11)


def student_world(root: Path, *, blended: bool = True, schedule: bool = True):
    """A base generation (a blended recipe with a stakes schedule and the hold, or a plain vrun one), a prior export to build on and a student
    file fitted for card, route and shop. Returns (agents, base, prior export, student file, human prior file)."""
    import hashlib

    import human_prior_infer as hpi
    import macro_recipe as mr
    from test_run_student import random_weights, write_weights

    _, _, trained = trained_candidate()
    agents = root / "agent"
    base = agents / "gen5" / "candidate"
    base.mkdir(parents=True)
    for name in ("combat_policy.pt", "combat_net.pt", "model.json"):
        (base / name).write_bytes((trained / name).read_bytes())
    (base / "run_value.npz").write_bytes(b"the base fit")
    (base / "run_value.pt").write_bytes(b"the base fit, torch")
    human = write_weights(root / "pi_h.npz", random_weights())
    meta = {"generation": 5, "macro": "vrun", "mask_version": 4, "tau": 0.0}
    if blended:
        sha = hashlib.sha256(human.read_bytes()).hexdigest()
        (base / "human_prior.npz").write_bytes(human.read_bytes())
        meta.update(macro=mr.macro_name(0.5, sha, acts={1: 0.0}), human_prior={"file": "human_prior.npz", "sha256": sha, "lambda": "0.5",
                                                                              "source": "pi_h/weights.npz", "lambda_acts": {"1": "0"}})
        if schedule:
            meta.update(search_schedule={"base": "a3-v1", "mult": {"hall": 1, "elite": 4, "boss": 16}, "tag": "a3-v1@h1e4b16"},
                        potion_hold={"hp_frac": 0.3, "fights": "hall"})
            meta["recipe"] = mr.full_recipe(meta["macro"], "a3-v1@h1e4b16", 30)
    (base / "agent.json").write_text(json.dumps(meta))
    (agents / "current.json").write_text(json.dumps({"generation": 5, "dir": str(base)}))
    prior = root / "players" / "v2"
    prior.mkdir(parents=True)
    for name in ("combat_policy.pt", "combat_net.pt", "model.json"):
        (prior / name).write_bytes((trained / name).read_bytes())
    student = root / "student.npz"
    hpi.write_student(student, random_weights(seed=9), kinds=("card", "route", "shop"), prior_sha256=hpi.file_sha256(human), fit={"test": True})
    return agents, base, prior, student, human


class StudentBuildTests(unittest.TestCase):
    """`build_candidate(student=, student_lams=)`: the recipe `vrun+pi_s:...`, the folder, the base's blend, search and hold kept, and the loader."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.agents, self.base, self.prior, self.student, self.human = student_world(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, **kw):
        kw.setdefault("student_lams", {"route": (0.1, 0.2, 0.2), "card": (0.3, 0.3, 0.3)})
        return gen_loop.build_candidate(self.prior, self.agents / "current.json", self.agents, 7, why="test", export=False, tag="student",
                                        student=self.student, **kw)

    def test_the_candidate_keeps_the_bases_blend_search_and_hold_and_names_the_student(self):
        import hashlib

        import screens

        cand = self.build()
        sha8 = hashlib.sha256(self.student.read_bytes()).hexdigest()[:8]
        meta = json.loads((cand / "agent.json").read_text())
        base_meta = json.loads((self.base / "agent.json").read_text())
        hp8 = base_meta["human_prior"]["sha256"][:8]
        want = f"vrun+pi_s:lambda=0.5:pi_h={hp8}:act1=0:pi_s={sha8}:card=0.3/0.3/0.3:route=0.1/0.2/0.2"
        self.assertEqual(want, meta["macro"])
        self.assertEqual(want + "+stakes:a3-v1@h1e4b16+hold:30", meta["recipe"])
        self.assertEqual(base_meta["human_prior"]["sha256"], meta["human_prior"]["sha256"])  # the prior's blend is the base's, not re-chosen
        self.assertEqual({"1": "0"}, meta["human_prior"]["lambda_acts"])
        self.assertEqual(base_meta["search_schedule"]["mult"], meta["search_schedule"]["mult"])
        self.assertEqual(0.3, meta["potion_hold"]["hp_frac"])
        self.assertEqual(self.student.read_bytes(), (cand / "run_student.npz").read_bytes())
        self.assertEqual(self.human.read_bytes(), (cand / "human_prior.npz").read_bytes())
        self.assertEqual({"card": ["0.3"] * 3, "route": ["0.1", "0.2", "0.2"]}, meta["student"]["kinds"])
        self.assertEqual(["card", "route", "shop"], meta["student"]["fitted_for"])
        self.assertIn("run student on card, route", meta["note"])
        a = gen_loop.load_agent(cand)
        self.assertEqual((want, str(cand / "run_student.npz")), (a.macro, a.student))
        self.assertEqual((("card", (0.3, 0.3, 0.3)), ("route", (0.1, 0.2, 0.2))), a.student_lams)
        self.assertEqual((0.5, ((1, 0.0),), "a3-v1@h1e4b16", 30), (a.lam, a.lam_acts, a.stakes, a.hold))
        self.assertEqual(meta["recipe"], gen_loop.candidate_recipe(a))
        self.assertEqual(cand, gen_loop.built_candidate(self.agents, 7, tag="student"))
        self.assertEqual("student", screens.kind_of(cand, self.base))  # it differs from the base in the student (and the recipe naming it) alone

    def test_a_base_that_is_already_a_student_is_replaced_by_the_new_student(self):
        first = self.build()
        again = gen_loop.build_candidate(self.prior, first, self.agents, 8, why="second", export=False, tag="second", student=self.student,
                                         student_lams={"shop": (0.4, 0.4, 0.4)})
        meta = json.loads((again / "agent.json").read_text())
        self.assertEqual({"shop": ["0.4"] * 3}, meta["student"]["kinds"])  # the new student's kinds, not the earlier student's
        self.assertTrue(meta["macro"].startswith("vrun+pi_s:lambda=0.5:pi_h="))
        self.assertEqual(json.loads((first / "agent.json").read_text())["human_prior"]["sha256"], meta["human_prior"]["sha256"])

    def test_a_plain_vrun_base_needs_the_human_prior_given_and_then_builds_the_blend_and_the_student(self):
        root = self.root / "plain"
        root.mkdir()
        agents, base, prior, student, human = student_world(root, blended=False)
        with self.assertRaisesRegex(ValueError, "blended base"):
            gen_loop.build_candidate(prior, agents / "current.json", agents, 7, why="t", export=False, student=student, student_lams={"route": (0.1,) * 3})
        self.assertFalse((agents / "gen7").exists() and any((agents / "gen7").iterdir()))  # nothing was written
        cand = gen_loop.build_candidate(prior, agents / "current.json", agents, 7, why="t", export=False, student=student, student_lams={"route": (0.1,) * 3},
                                        human_prior=human, lam=0.5)
        meta = json.loads((cand / "agent.json").read_text())
        self.assertTrue(meta["macro"].startswith("vrun+pi_s:lambda=0.5:pi_h="))
        self.assertNotIn("search_schedule", meta)  # a plain base has no schedule to keep

    def test_what_the_builder_refuses_it_refuses_before_writing(self):
        for kw, pattern in (({"student_lams": {"rest": (0.1,) * 3}}, "was fitted for"),  # a kind the file was not fitted for
                            ({"student_lams": {"bundle": (0.1,) * 3}}, "was fitted for|takes over the kinds"),
                            ({"student_lams": {"route": (0.1, 0.1)}}, "three numbers"),
                            ({"student_lams": {"route": (0.1, 0.1, -1.0)}}, "three numbers"),
                            ({"student_lams": {}}, "needs its per-kind lambdas")):
            with self.assertRaisesRegex(ValueError, pattern, msg=str(kw)):
                self.build(**kw)
        with self.assertRaisesRegex(ValueError, "needs its per-kind lambdas"):
            gen_loop.build_candidate(self.prior, self.agents / "current.json", self.agents, 7, why="t", export=False, student_lams={"route": (0.1,) * 3})
        with self.assertRaises(FileNotFoundError):
            gen_loop.build_candidate(self.prior, self.agents / "current.json", self.agents, 7, why="t", export=False, student=self.root / "nope.npz",
                                     student_lams={"route": (0.1,) * 3})
        with self.assertRaisesRegex(ValueError, "not a run student"):
            gen_loop.build_candidate(self.prior, self.agents / "current.json", self.agents, 7, why="t", export=False, student=self.human,
                                     student_lams={"route": (0.1,) * 3})
        self.assertFalse((self.agents / "gen7").exists() and any((self.agents / "gen7").iterdir()))

    def test_without_a_student_a_blended_base_is_still_refused_and_nothing_else_changed(self):
        with self.assertRaisesRegex(ValueError, "adds the prior to the vrun macro"):
            gen_loop.build_candidate(self.prior, self.agents / "current.json", self.agents, 7, why="t", export=False, human_prior=self.human, lam=0.5)
        with self.assertRaisesRegex(ValueError, "stakes schedule or a potion hold"):
            gen_loop.build_candidate(self.prior, self.agents / "current.json", self.agents, 7, why="t", export=False)

    def test_the_loader_holds_the_folder_to_the_recipe(self):
        cand = self.build()
        meta_path = cand / "agent.json"
        good = json.loads(meta_path.read_text())
        gen_loop.load_agent(cand)
        bad = json.loads(json.dumps(good))
        bad["student"]["kinds"]["route"] = ["0.1", "0.2", "0.9"]
        meta_path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "student's kinds"):
            gen_loop.load_agent(cand)
        meta_path.write_text(json.dumps(good))
        original = (cand / "run_student.npz").read_bytes()
        (cand / "run_student.npz").write_bytes(original + b"x")
        with self.assertRaisesRegex(ValueError, "is not the student"):
            gen_loop.load_agent(cand)
        (cand / "run_student.npz").write_bytes(self.human.read_bytes())  # a plain prior in the student's place (its sha is of course not the recipe's)
        with self.assertRaises(ValueError):
            gen_loop.load_agent(cand)
        (cand / "run_student.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            gen_loop.load_agent(cand)
        (cand / "run_student.npz").write_bytes(original)
        gen_loop.load_agent(cand)

    def test_a_recipe_that_names_a_kind_the_file_was_not_fitted_for_does_not_load(self):
        import hashlib

        import human_prior_infer as hpi
        from test_run_student import random_weights

        cand = self.build()
        narrow = self.root / "narrow.npz"
        hpi.write_student(narrow, random_weights(seed=9), kinds=("card",), prior_sha256="0" * 64)
        (cand / "run_student.npz").write_bytes(narrow.read_bytes())
        meta = json.loads((cand / "agent.json").read_text())
        sha = hashlib.sha256(narrow.read_bytes()).hexdigest()
        meta["macro"] = meta["macro"].replace(meta["student"]["sha256"][:8], sha[:8])
        meta["recipe"] = meta["recipe"].replace(meta["student"]["sha256"][:8], sha[:8])
        meta["student"]["sha256"] = sha
        (cand / "agent.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "was fitted for"):
            gen_loop.load_agent(cand)

    def test_the_recipe_is_its_own_in_the_ledger_and_the_exploring_agent_keeps_the_student(self):
        a = gen_loop.load_agent(self.build())
        e = gen_loop.exploring_agent(a, 0.5, 0.0)
        self.assertEqual((a.student, a.student_lams, a.macro), (e.student, e.student_lams, e.macro))
        base_a = gen_loop.load_agent(self.base)
        self.assertNotEqual(gen_loop.candidate_recipe(a), gen_loop.candidate_recipe(base_a))
        rows = [{"cand_gen": "7", "macro": gen_loop.candidate_recipe(base_a), "cand_sha": "x", "run_head_sha": "y"}]
        self.assertEqual(0, gen_loop.attempts_before(rows, 7, "x", "y", gen_loop.candidate_recipe(a)))  # the student's recipe starts at attempt 1


class StudentCandidateTests(unittest.TestCase):
    """`run_student.build_student_candidate`: the fit folder's student and legs become `vrun+pi_s:...` on the incumbent, the legs stored beside it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.agents, self.base, self.prior, self.student, self.human = student_world(self.root)
        self.out = self.root / "fit"
        self.out.mkdir()
        (self.out / run_student.STUDENT_OUT).write_bytes(self.student.read_bytes())
        self.sha = human_prior_infer.file_sha256(self.student)

    def tearDown(self):
        self.tmp.cleanup()

    def legs(self, **kw) -> dict:
        from test_screens import legs_doc

        kinds = kw.pop("kinds", ("route", "shop"))
        doc = legs_doc(self.student.read_bytes(), self.human.read_bytes(), kinds=kinds, human=kw.pop("human", kinds), **kw)
        doc["student_sha256"] = self.sha
        return doc

    def write(self, legs: dict) -> None:
        (self.out / run_student.LEGS_FILE).write_text(json.dumps(legs), encoding="utf-8")

    def build(self, **kw):
        return run_student.build_student_candidate(self.out, self.agents / "current.json", self.agents, 7, why="test", export=False, **kw)

    def test_the_passing_kinds_become_the_recipe_and_the_legs_sit_beside_the_student(self):
        self.write(self.legs(lam=0.25))
        cand = self.build()
        meta = json.loads((cand / "agent.json").read_text())
        self.assertEqual({"route": ["0.25"] * 3, "shop": ["0.25"] * 3}, meta["student"]["kinds"])
        self.assertTrue(meta["macro"].endswith(f":pi_s={self.sha[:8]}:route=0.25/0.25/0.25:shop=0.25/0.25/0.25"), meta["macro"])
        stored = json.loads((cand / "run_student_legs.json").read_text())
        self.assertEqual(self.sha, stored["student_sha256"])
        import screens

        r = screens.run_student(cand, self.base)  # the screen reads exactly what the builder stored
        self.assertFalse(r["rejected"], r.get("line"))
        self.assertNotIn("skipped", r)

    def test_only_the_kinds_that_passed_are_given_to_the_student(self):
        legs = self.legs()
        legs["leg1"]["kinds"]["shop"]["gain"]["lo95"] = -0.01  # shop fails leg 1
        self.write(legs)
        meta = json.loads((self.build() / "agent.json").read_text())
        self.assertEqual(["route"], list(meta["student"]["kinds"]))

    def test_a_subset_may_be_asked_for_and_no_passing_kind_is_no_candidate(self):
        self.write(self.legs())
        meta = json.loads((self.build(kinds=("shop",), tag="only-shop") / "agent.json").read_text())
        self.assertEqual(["shop"], list(meta["student"]["kinds"]))
        legs = self.legs()
        for k in legs["leg1"]["kinds"].values():
            k["gain"]["lo95"] = -1.0
        self.write(legs)
        with self.assertRaisesRegex(ValueError, "no kind passed"):
            self.build(tag="none")
        self.assertFalse((self.agents / "gen7" / "candidate-none").exists())

    def test_legs_measured_against_another_prior_than_the_bases_are_refused(self):
        legs = self.legs()
        legs["baseline"]["sha256"] = "0" * 64
        self.write(legs)
        with self.assertRaisesRegex(ValueError, "another prior than the base's"):
            self.build()

    def test_the_candidate_is_the_base_with_the_student_and_nothing_else(self):
        import screens

        self.write(self.legs())
        cand = self.build()
        self.assertEqual(["student", "settings"], screens.differing(cand, self.base))
        self.assertEqual("student", screens.kind_of(cand, self.base))
        for name in ("combat_policy.pt", "combat_net.pt", "run_value.npz", "human_prior.npz"):
            self.assertEqual((self.base / name).read_bytes(), (cand / name).read_bytes(), name)


    def test_a_vrun_base_takes_the_blend_by_name_and_the_legs_must_be_against_that_prior(self):
        from test_screens import legs_doc

        root = self.root / "plain"
        root.mkdir()
        agents, base, prior, student, human = student_world(root, blended=False)
        out = root / "fit"
        out.mkdir()
        (out / run_student.STUDENT_OUT).write_bytes(student.read_bytes())
        legs = legs_doc(student.read_bytes(), human.read_bytes(), kinds=("route",), lam=0.2)
        legs["student_sha256"] = human_prior_infer.file_sha256(student)
        (out / run_student.LEGS_FILE).write_text(json.dumps(legs), encoding="utf-8")
        args = (out, agents / "current.json", agents, 7)
        with self.assertRaisesRegex(ValueError, "another prior than the base's"):  # a vrun base has no prior of its own to be the baseline
            run_student.build_student_candidate(*args, why="t", export=False)
        with self.assertRaisesRegex(ValueError, "both the human prior's weights and lambda"):
            run_student.build_student_candidate(*args, why="t", export=False, human_prior=human)
        cand = run_student.build_student_candidate(*args, why="t", export=False, human_prior=human, lam=0.5)
        self.assertTrue(json.loads((cand / "agent.json").read_text())["macro"].startswith("vrun+pi_s:lambda=0.5:pi_h="))


class StudentKeepRestTests(unittest.TestCase):
    """`build_student_candidate(keep_rest=True)` (D59 lane H), the lab's student job: the candidate is the incumbent's student with the kinds the new fit took replaced
    and every other kind it serves kept, the incumbent being a merged student already (generation 15's: card, rest, shop, reward on one net, route and event on another)."""

    SIX = ("card", "rest", "shop", "reward", "route", "event")

    def setUp(self):
        from test_run_student import random_weights
        from test_screens import legs_doc

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.agents, _base5, self.prior, student, self.human = student_world(self.root)
        self.legs_doc, self.weights = legs_doc, random_weights
        sha = human_prior_infer.file_sha256
        # generation 6 serves route and event with its own net; generation 7 (the incumbent) takes card, rest, shop, reward from a new fit and keeps those two
        self.sb = self.fit_file("sb", seed=21, kinds=("route", "event"), prior=self.human)
        inc0 = gen_loop.build_candidate(self.prior, self.agents / "current.json", self.agents, 6, why="inc0", export=False, tag="inc0", student=self.sb,
                                        student_lams={"route": (0.3715, 2.0, 2.0), "event": (0.8603, 0.9014, 0.9014)})
        legs0 = legs_doc((inc0 / "run_student.npz").read_bytes(), self.human.read_bytes(), kinds=("route", "event"), human=("route", "event"), lam=0.3)
        legs0["student_sha256"] = sha(inc0 / "run_student.npz")
        legs0["lambda"]["chosen"] = {"route": [0.3715, 2.0, 2.0], "event": [0.8603, 0.9014, 0.9014]}
        (inc0 / "run_student_legs.json").write_text(json.dumps(legs0), encoding="utf-8")
        (self.agents / "current.json").write_text(json.dumps({"generation": 6, "dir": str(inc0)}))
        self.sa = self.fit_file("sa", seed=22, kinds=("card", "rest", "shop", "reward"), prior=inc0 / "run_student.npz")
        out_a = self.fit_folder("fita", self.sa, ("card", "rest", "shop", "reward"), inc0 / "run_student.npz", lam=0.2)
        self.inc = run_student.build_student_candidate(out_a, self.agents / "current.json", self.agents, 7, why="inc", export=False, tag="inc", keep=("route", "event"),
                                                       lambda_floor={"card": 0.5})
        (self.agents / "current.json").write_text(json.dumps({"generation": 7, "dir": str(self.inc)}))
        self.inc_student = self.inc / "run_student.npz"
        self.inc_meta = json.loads((self.inc / "agent.json").read_text())
        self.inc_legs = json.loads((self.inc / "run_student_legs.json").read_text())
        # the lab's next fit: one net for every kind, started from the incumbent's file (its first net), legs against that file
        self.sn = self.fit_file("sn", seed=23, kinds=self.SIX, prior=self.inc_student)

    def fit_file(self, name, *, seed, kinds, prior):
        path = self.root / f"{name}.npz"
        human_prior_infer.write_student(path, self.weights(seed=seed), kinds=kinds, prior_sha256=human_prior_infer.file_sha256(prior), fit={"test": name})
        return path

    def fit_folder(self, name, student, passing, baseline, *, lam=0.25):
        out = self.root / name
        out.mkdir()
        (out / run_student.STUDENT_OUT).write_bytes(student.read_bytes())
        human = tuple(k for k in passing if k in run_student.HUMAN_KINDS)
        legs = self.legs_doc(student.read_bytes(), Path(baseline).read_bytes(), kinds=tuple(passing), human=human, lam=lam)
        legs["student_sha256"] = human_prior_infer.file_sha256(student)
        (out / run_student.LEGS_FILE).write_text(json.dumps(legs), encoding="utf-8")
        return out

    def build(self, passing, **kw):
        out = self.fit_folder("fitn", self.sn, passing, self.inc_student, lam=0.15)
        kw.setdefault("keep_rest", True)
        return run_student.build_student_candidate(out, self.agents / "current.json", self.agents, 8, why="test", export=False, tag="third", **kw)

    def scores(self, path, kind):
        rng = np.random.default_rng(5)
        dense = rng.standard_normal((5, human_prior_infer.DENSE_WIDTH)).astype(np.float32)
        counts = rng.integers(0, 3, (5, human_prior_infer.E_SIZE + human_prior_infer.F_SIZE + human_prior_infer.G_SIZE)).astype(np.float32)
        return human_prior_infer.load_student(path).scorer_for(kind).logits(dense, counts)

    def test_the_incumbent_here_is_a_merged_student_of_two_nets_serving_six_kinds(self):
        self.assertEqual(self.SIX, human_prior_infer.student_kinds_of(self.inc_student))
        self.assertEqual(sorted(self.SIX), sorted(self.inc_meta["student"]["kinds"]))
        with np.load(self.inc_student, allow_pickle=False) as z:
            self.assertEqual([["card", "rest", "shop", "reward"], ["route", "event"]], [n["kinds"] for n in json.loads(str(z["meta"]))["student"]["nets"]])

    def test_a_fit_passing_two_of_six_leaves_a_candidate_serving_six_four_of_them_the_incumbents_own_nets(self):
        cand = self.build(("card", "route"))
        meta = json.loads((cand / "agent.json").read_text())
        s = meta["student"]
        self.assertEqual(sorted(self.SIX), sorted(s["kinds"]))  # the recipe lists every kind the incumbent served
        for kind in ("rest", "shop", "reward", "event"):
            self.assertEqual(self.inc_meta["student"]["kinds"][kind], s["kinds"][kind], kind)  # the incumbent's lambdas, number for number
            np.testing.assert_array_equal(self.scores(cand / "run_student.npz", kind), self.scores(self.inc_student, kind), err_msg=kind)  # its net, byte for byte
        for kind in ("card", "route"):
            np.testing.assert_array_equal(self.scores(cand / "run_student.npz", kind), self.scores(self.sn, kind), err_msg=kind)
        self.assertEqual({"card": ["0.15"] * 3, "route": ["0.15"] * 3}, {k: s["kinds"][k] for k in ("card", "route")})  # the new fit's own lambdas
        self.assertEqual(sorted(("rest", "shop", "reward", "event")), sorted(s["kept"]["kinds"]))
        self.assertEqual(human_prior_infer.file_sha256(self.inc_student), s["kept"]["sha256"])
        self.assertEqual(human_prior_infer.file_sha256(cand / "run_student.npz"), s["sha256"])
        import macro_recipe

        self.assertTrue(meta["macro"].endswith(":".join(f"{k}={'/'.join(s['kinds'][k])}" for k in macro_recipe.STUDENT_KINDS if k in s["kinds"])), meta["macro"])
        agent = gen_loop.load_agent(cand)
        self.assertEqual(set(self.SIX), {k for k, _ in agent.student_lams})

    def test_the_stored_legs_cover_the_six_kinds_and_say_where_each_kept_head_came_from(self):
        import screens

        cand = self.build(("card", "route"))
        stored = json.loads((cand / "run_student_legs.json").read_text())
        self.assertEqual(human_prior_infer.file_sha256(cand / "run_student.npz"), stored["student_sha256"])
        for kind in ("rest", "shop", "reward", "event"):
            self.assertEqual(self.inc_legs["leg1"]["kinds"][kind], stored["leg1"]["kinds"][kind], kind)
        self.assertEqual({"rest", "shop", "reward", "event"}, set(stored["kept"]))
        inc_sha = human_prior_infer.file_sha256(self.inc_student)
        for kind in ("rest", "shop", "reward"):  # kept from the incumbent's own net, whose legs are the incumbent's fit's
            self.assertEqual(inc_sha, stored["kept"][kind]["student_sha256"], kind)
        self.assertEqual(self.inc_legs["kept"]["event"], {k: stored["kept"]["event"][k] for k in self.inc_legs["kept"]["event"]})  # event was kept by the incumbent already: its source stays
        self.assertNotEqual(inc_sha, stored["kept"]["event"]["student_sha256"])
        r = screens.run_student(cand, self.inc)
        self.assertNotIn("skipped", r)
        self.assertFalse(r["rejected"], r.get("line"))
        self.assertEqual(set(self.SIX), set(r["kinds"]))
        self.assertEqual("student", screens.kind_of(cand, self.inc))

    def test_a_kind_the_fit_passed_that_was_frozen_out_of_it_is_kept_from_the_incumbent(self):
        cand = self.build(("card", "route", "rest"), kinds=("card",))  # the lab's frozen kinds are not in `kinds`: the fit's pass for them is not handed over
        s = json.loads((cand / "agent.json").read_text())["student"]
        self.assertEqual(sorted(self.SIX), sorted(s["kinds"]))
        self.assertEqual(sorted(("rest", "shop", "reward", "route", "event")), sorted(s["kept"]["kinds"]))
        np.testing.assert_array_equal(self.scores(cand / "run_student.npz", "route"), self.scores(self.inc_student, "route"))
        np.testing.assert_array_equal(self.scores(cand / "run_student.npz", "card"), self.scores(self.sn, "card"))

    def test_a_fit_that_takes_every_kind_the_incumbent_serves_is_a_plain_student_as_before(self):
        cand = self.build(self.SIX)
        meta = json.loads((cand / "agent.json").read_text())
        self.assertNotIn("kept", meta["student"])
        self.assertEqual(human_prior_infer.file_sha256(self.sn), meta["student"]["sha256"])

    def test_without_the_flag_nothing_is_kept_and_a_base_without_a_student_keeps_nothing_and_does_not_fail(self):
        meta = json.loads((self.build(("card", "route"), keep_rest=False) / "agent.json").read_text())
        self.assertEqual(["card", "route"], sorted(meta["student"]["kinds"]))
        self.assertNotIn("kept", meta["student"])
        plain = self.root / "plain"
        plain.mkdir()
        agents, base, prior, student, human = student_world(plain)  # a base that plays no student
        out = plain / "fit"
        out.mkdir()
        (out / run_student.STUDENT_OUT).write_bytes(student.read_bytes())
        legs = self.legs_doc(student.read_bytes(), human.read_bytes(), kinds=("route", "shop"), human=("route", "shop"))
        legs["student_sha256"] = human_prior_infer.file_sha256(student)
        (out / run_student.LEGS_FILE).write_text(json.dumps(legs), encoding="utf-8")
        cand = run_student.build_student_candidate(out, agents / "current.json", agents, 7, why="t", export=False, keep_rest=True)
        s = json.loads((cand / "agent.json").read_text())["student"]
        self.assertEqual(["route", "shop"], sorted(s["kinds"]))
        self.assertNotIn("kept", s)

    def test_the_lab_student_job_builds_a_candidate_serving_six_kinds_from_a_fit_that_passed_two(self):
        """The lab's own command (`tools/lab/student_job.py`), with the fit stood in for by the fit folder of a pass of card and route and the real builder:
        generation 15's merged incumbent, six kinds served, two passed; the first lab student after it dropped the other four."""
        from unittest import mock

        from tools.lab import student_job as sj

        src = self.fit_folder("jobfit", self.sn, ("card", "route"), self.inc_student, lam=0.15)
        legs = json.loads((src / run_student.LEGS_FILE).read_text())
        legs["kinds_passing"] = {k: [0.15] * 3 for k in ("card", "route")}
        legs["verdict"] = {k: {"pass": k in ("card", "route"), "leg1": True, "leg2": True if k == "card" else None, "lambda": [0.15] * 3, "why": ""} for k in run_student.KINDS}
        out, built = self.root / "job" / "student", run_student.build_student_candidate

        def fit(shards_root, prior, folder, **kw):
            Path(folder).mkdir(parents=True)
            (Path(folder) / run_student.STUDENT_OUT).write_bytes((src / run_student.STUDENT_OUT).read_bytes())
            (Path(folder) / run_student.LEGS_FILE).write_text(json.dumps(legs), encoding="utf-8")
            (Path(folder) / run_student.FIT_FILE).write_text(json.dumps({"data": {"decisions": 7}}), encoding="utf-8")
            return legs

        with mock.patch.object(run_student, "run_fit", side_effect=fit), mock.patch.object(run_student, "_priority"), \
                mock.patch.object(run_student, "build_student_candidate", side_effect=lambda *a, **kw: built(*a, **{**kw, "export": False})):
            with contextlib.redirect_stdout(io.StringIO()):
                rc = sj.main(["--lab", str(self.root / "LAB"), "--shards-root", str(self.root / "store"), "--prior", str(self.inc_student), "--out", str(out),
                              "--base", str(self.agents / "current.json"), "--agent-root", str(self.agents), "--target", "8", "--tag", "student",
                              "--result", str(self.root / "job" / "result.json")])
        self.assertEqual(0, rc)
        result = json.loads((self.root / "job" / "result.json").read_text(encoding="utf-8"))
        self.assertEqual((True, "gen8/candidate-student", ["event", "rest", "reward", "shop"]), (result["built"], result["candidate"], result["kept"]))
        cand = self.agents / "gen8" / "candidate-student"
        meta = json.loads((cand / "agent.json").read_text())
        self.assertEqual(sorted(self.SIX), sorted(meta["student"]["kinds"]))
        for kind in ("rest", "shop", "reward", "event"):
            self.assertEqual(self.inc_meta["student"]["kinds"][kind], meta["student"]["kinds"][kind], kind)
            np.testing.assert_array_equal(self.scores(cand / "run_student.npz", kind), self.scores(self.inc_student, kind), err_msg=kind)
        for kind in ("card", "route"):
            np.testing.assert_array_equal(self.scores(cand / "run_student.npz", kind), self.scores(self.sn, kind), err_msg=kind)

    def test_no_kind_passing_is_still_no_candidate_and_leaves_no_folder(self):
        with self.assertRaisesRegex(ValueError, "no kind passed"):
            self.build(())
        self.assertFalse((self.agents / "gen8" / "candidate-third").exists())


class StudentKeepTests(unittest.TestCase):
    """`run_student.build_student_candidate(keep=, lambda_floor=)` (D59 lane G): heads kept from the incumbent's student beside a new fit's, in one student
    file; a kind's lambda at play floored with the measured one recorded; the stored legs cover every kind the recipe names."""

    def setUp(self):
        import hashlib

        from test_run_student import random_weights
        from test_screens import legs_doc

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.agents, _base5, self.prior, self.student, self.human = student_world(self.root)
        # the incumbent: generation 6, a student serving route and shop, with the legs it passed (against the human prior)
        self.inc = gen_loop.build_candidate(self.prior, self.agents / "current.json", self.agents, 6, why="incumbent", export=False, tag="inc",
                                            student=self.student, student_lams={"route": (0.3715, 2.0, 2.0), "shop": (0.2858, 0.2311, 0.2311)})
        inc_student = self.inc / "run_student.npz"
        legs = legs_doc(inc_student.read_bytes(), self.human.read_bytes(), kinds=("route", "shop"), human=("route", "shop"), lam=0.3)
        legs["student_sha256"] = human_prior_infer.file_sha256(inc_student)
        legs["lambda"]["chosen"] = {"route": [0.3715, 2.0, 2.0], "shop": [0.2858, 0.2311, 0.2311]}
        self.inc_legs = legs
        (self.inc / "run_student_legs.json").write_text(json.dumps(legs), encoding="utf-8")
        (self.agents / "current.json").write_text(json.dumps({"generation": 6, "dir": str(self.inc)}))
        # the new fit: card, shop and reward, started from the incumbent's student, its legs against it
        self.new = self.root / "new.npz"
        human_prior_infer.write_student(self.new, random_weights(seed=11), kinds=("card", "rest", "shop", "reward"),
                                        prior_sha256=human_prior_infer.file_sha256(inc_student), fit={"test": True})
        self.out = self.root / "fit2"
        self.out.mkdir()
        (self.out / run_student.STUDENT_OUT).write_bytes(self.new.read_bytes())
        new_legs = legs_doc(self.new.read_bytes(), inc_student.read_bytes(), kinds=("card", "shop", "reward"), human=("card", "shop"), lam=0.25)
        new_legs["lambda"]["chosen"]["card"] = [0.0, 0.1149, 0.1149]
        self.new_legs = new_legs
        (self.out / run_student.LEGS_FILE).write_text(json.dumps(new_legs), encoding="utf-8")
        self.sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def build(self, **kw):
        kw.setdefault("tag", "second")
        return run_student.build_student_candidate(self.out, self.inc, self.agents, 7, why="test", export=False, **kw)

    def test_a_kept_head_and_a_new_one_are_one_student_file_each_at_the_lambdas_it_was_measured_at(self):
        cand = self.build(keep=("route",), lambda_floor={"card": 0.5})
        meta = json.loads((cand / "agent.json").read_text())
        merged = cand / "run_student.npz"
        self.assertEqual(self.sha(merged), meta["student"]["sha256"])
        self.assertTrue(meta["macro"].endswith(f":pi_s={self.sha(merged)[:8]}:card=0.5/0.5/0.5:route=0.3715/2/2:shop=0.25/0.25/0.25:reward=0.25/0.25/0.25"), meta["macro"])
        self.assertEqual({"card": ["0.5"] * 3, "route": ["0.3715", "2", "2"], "shop": ["0.25"] * 3, "reward": ["0.25"] * 3}, meta["student"]["kinds"])
        # the kept head is the incumbent's own net, the new kinds the new fit's
        s, inc, new = (human_prior_infer.load_student(p) for p in (merged, self.inc / "run_student.npz", self.new))
        rng = np.random.default_rng(2)
        dense = rng.standard_normal((5, human_prior_infer.DENSE_WIDTH)).astype(np.float32)
        counts = rng.integers(0, 3, (5, human_prior_infer.E_SIZE + human_prior_infer.F_SIZE + human_prior_infer.G_SIZE)).astype(np.float32)
        np.testing.assert_allclose(s.scorer_for("route").logits(dense, counts), inc.logits(dense, counts), atol=1e-6)
        for kind in ("card", "shop", "reward"):
            np.testing.assert_allclose(s.scorer_for(kind).logits(dense, counts), new.logits(dense, counts), atol=1e-6, err_msg=kind)
        # what the record says: where each head came from, and the measured lambda beside the floored one
        self.assertEqual({"kinds": {"route": ["0.3715", "2", "2"]}, "sha256": self.sha(self.inc / "run_student.npz")}, {
            "kinds": meta["student"]["kept"]["kinds"], "sha256": meta["student"]["kept"]["sha256"]})
        self.assertEqual({"card": {"measured": ["0", "0.1149", "0.1149"], "floor": "0.5"}}, meta["student"]["floored"])
        self.assertEqual(list(human_prior_infer.student_kinds_of(merged)), meta["student"]["fitted_for"])
        agent = gen_loop.load_agent(cand)
        self.assertEqual({"card": (0.5,) * 3, "route": (0.3715, 2.0, 2.0), "shop": (0.25,) * 3, "reward": (0.25,) * 3}, dict(agent.student_lams))

    def test_the_stored_legs_cover_every_named_kind_and_the_screen_reads_them(self):
        import screens

        cand = self.build(keep=("route",), lambda_floor={"card": 0.5})
        stored = json.loads((cand / "run_student_legs.json").read_text())
        self.assertEqual(self.sha(cand / "run_student.npz"), stored["student_sha256"])
        self.assertEqual(self.new_legs["baseline"], stored["baseline"])  # measured against the incumbent's student, as the screen checks
        self.assertEqual(self.inc_legs["leg1"]["kinds"]["route"], stored["leg1"]["kinds"]["route"])  # the kept head carries its own legs
        self.assertEqual(self.new_legs["leg1"]["kinds"]["card"], stored["leg1"]["kinds"]["card"])
        self.assertEqual({"route": {"student_sha256": self.inc_legs["student_sha256"], "baseline": self.inc_legs["baseline"]}},
                         {k: {x: v[x] for x in ("student_sha256", "baseline")} for k, v in stored["kept"].items()})
        self.assertEqual([0.3715, 2.0, 2.0], stored["lambda"]["chosen"]["route"])
        self.assertEqual([0.0, 0.1149, 0.1149], stored["lambda"]["chosen"]["card"])  # the measured value stays in the legs, the floor is in the recipe
        r = screens.run_student(cand, self.inc)
        self.assertNotIn("skipped", r)
        self.assertFalse(r["rejected"], r.get("line"))
        self.assertEqual({"card", "route", "shop", "reward"}, set(r["kinds"]))
        self.assertEqual("student", screens.kind_of(cand, self.inc))

    def test_a_floor_only_raises_a_lambda_and_only_for_a_kind_the_student_takes(self):
        meta = json.loads((self.build(lambda_floor={"card": 0.05, "shop": 0.2, "ancient": 0.5}, tag="floors") / "agent.json").read_text())
        self.assertEqual({"card": ["0.05", "0.1149", "0.1149"], "shop": ["0.25"] * 3}, {k: meta["student"]["kinds"][k] for k in ("card", "shop")})
        self.assertEqual({"card": {"measured": ["0", "0.1149", "0.1149"], "floor": "0.05"}}, meta["student"]["floored"])  # shop is over its floor, ancient is not taken

    def test_without_keep_or_floor_the_candidate_is_what_it_was(self):
        meta = json.loads((self.build(tag="plain") / "agent.json").read_text())
        self.assertNotIn("kept", meta["student"])
        self.assertNotIn("floored", meta["student"])
        self.assertEqual(self.sha(self.new), meta["student"]["sha256"])  # the new fit's own file, not a merged one
        self.assertEqual(["card", "rest", "shop", "reward"], meta["student"]["fitted_for"])

    def test_what_cannot_be_kept_is_refused_with_the_reason_and_leaves_no_folder(self):
        for kw, why in (({"keep": ("card",)}, "does not serve"),  # the incumbent's student serves route and shop only
                        ({"keep": ("shop",)}, "new student and by the kept"),  # the new fit takes shop too
                        ({"keep": ("route",), "lambda_floor": {"bundle": 0.5}}, "kind")):
            with self.assertRaisesRegex(ValueError, why, msg=str(kw)):
                self.build(tag="refused", **kw)
            self.assertFalse((self.agents / "gen7" / "candidate-refused").exists())

    def test_a_base_without_a_student_or_without_its_legs_keeps_nothing(self):
        (self.inc / "run_student_legs.json").unlink()
        with self.assertRaisesRegex(ValueError, "legs"):
            self.build(keep=("route",), tag="nolegs")
        (self.inc / "run_student_legs.json").write_text(json.dumps({**self.inc_legs, "student_sha256": "0" * 64}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "legs"):
            self.build(keep=("route",), tag="otherlegs")
        legs = json.loads((self.out / run_student.LEGS_FILE).read_text())
        (self.root / "plain").mkdir()
        agents, base, prior, student, human = student_world(self.root / "plain")  # a base that plays no student
        (self.out / run_student.LEGS_FILE).write_text(json.dumps({**legs, "baseline": {"sha256": human_prior_infer.file_sha256(human), "kind": "human prior"}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no student"):
            run_student.build_student_candidate(self.out, agents / "current.json", agents, 7, why="t", export=False, keep=("route",))
        self.assertFalse((agents / "gen7").exists())

    def test_the_command_line_takes_the_kinds_the_heads_to_keep_and_the_floors(self):
        p = run_student.build_parser()
        a = p.parse_args(["candidate", "--out", "o", "--base", "b", "--agent-root", "r", "--target", "9", "--keep", "route,event", "--kinds", "card,rest",
                          "--lambda-floor", "card=0.5", "--lambda-floor", "rest=0.25"])
        self.assertEqual((("route", "event"), ("card", "rest"), {"card": 0.5, "rest": 0.25}),
                         (tuple(a.keep), tuple(a.kinds), run_student.parse_floors(a.lambda_floor)))
        d = p.parse_args(["candidate", "--out", "o", "--base", "b", "--agent-root", "r", "--target", "9"])
        self.assertEqual(((), None, None), (tuple(d.keep), d.kinds, d.lambda_floor))
        with self.assertRaises(SystemExit):
            p.parse_args(["candidate", "--out", "o", "--base", "b", "--agent-root", "r", "--target", "9", "--lambda-floor", "card"])


if __name__ == "__main__":
    unittest.main()
