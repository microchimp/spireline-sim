"""The generation loop (rl/gen_loop.py, rl/gen0.py train and promote): shards, training, the agent's prior, the E1 block,
the promotion maths and the lab's train markers. Tests that need the native library skip without it."""

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
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import gen0  # noqa: E402
import gen_loop  # noqa: E402

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

        class Ready:
            def __init__(self, value):
                self.value = value

            def ready(self):
                return True

            def get(self):
                return self.value

        class Pool:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def apply_async(self, fn, args):
                return Ready(fn(*args))

        ctx = SimpleNamespace(Pool=Pool)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(gen0, "_run_safe", fake_run_safe), \
                mock.patch.object(gen0.mp, "get_context", return_value=ctx), mock.patch.object(gen0.time, "sleep"):
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
            self.assertEqual((tg.call_args[1]["retrain"], tg.call_args[1]["max_fights"]), (1, 12800))

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


if __name__ == "__main__":
    unittest.main()
