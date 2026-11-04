"""Generation 12's second attempt, the plumbing (docs/diagnostics/2026-11-04-gen12-attempt2.md in the Spireline repo, D47):

1. a candidate is its combat prior and its run head, so attempt 2 (same fight player, new head) gets a fresh E-block and its own row;
2. attempt 2 is placed beside attempt 1 (gen<N>/candidate-a2, ready-<base>-g<N>-a2) and the train job takes it;
3. a stopped self-play block loses at most the runs in flight (the sink's age flush, the stop file);
4. `vrun-fit --exploring-only` takes the blocks an exploring agent played.

No game and no emulator: stub arms, a fake pool, synthetic manifests.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import gen0  # noqa: E402
import gen_loop  # noqa: E402
from tests.python.test_funnel import StubArms, make_block, stub_candidate  # noqa: E402
from tests.python.test_gen_loop import fake_record, trained_candidate  # noqa: E402
from tools.lab import jobs  # noqa: E402

PRIOR_SHA = "64d057e174ac" + "0" * 52  # generation 11's fight player, as attempt 1 and attempt 2 both keep it

# promotions.tsv as it stood before the run_head_sha column: attempt 1 of generation 12, refused on E12-2026-11-02
OLD_HEADER = ("written\tcand_gen\tbase_gen\tseed_block\tseeds\td_floor\tfloor_lower\tp_floor\tboss_cand\tboss_base\tp_boss\twins_cand\t"
              "wins_base\tdecision\tcand_sha\tcand_dir\tattempt\tdrift\tguard")
OLD_ROW = ("2026-11-02T16:01:00Z\t12\t11\tE12-2026-11-02\t2000\t-5.9100\t-6.4400\t1.00000\t541\t1033\t1.00000\t0\t0\trejected\t"
           f"{PRIOR_SHA}\tcandidate\t1\tclear\toff")


def vrun_stub(root: Path, name: str, head: bytes, sha: str = PRIOR_SHA, generation: int = 12) -> Path:
    """A candidate directory the agent loader accepts: a prior sha (never loaded) and a run head file whose bytes name the head."""
    c = stub_candidate(root, name, sha, generation)
    (c / "agent.json").write_text(json.dumps({"generation": generation, "macro": "vrun"}))
    (c / "run_value.npz").write_bytes(head)
    return c


def legacy_tsv(agents: Path) -> Path:
    tsv = agents / "promotions.tsv"
    tsv.write_text(OLD_HEADER + "\n" + OLD_ROW + "\n", encoding="utf-8")
    return tsv


class CandidateIdentityTests(unittest.TestCase):
    """Item 1: the identity is (combat prior sha, run head sha); rows without the head read as their prior alone."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.agents = self.root / "agent"
        self.agents.mkdir()
        self.tsv = legacy_tsv(self.agents)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_old_rows_stay_readable_and_read_as_a_prior_with_no_head(self):
        rows = gen_loop.read_promotions(self.tsv)
        self.assertEqual(1, len(rows))
        self.assertEqual((PRIOR_SHA, ""), gen_loop.row_identity(rows[0]))
        self.assertNotIn("run_head_sha", rows[0])
        # exactly the readings the first attempt had: the same candidate again is no new attempt, a different prior is
        self.assertEqual(0, gen_loop.attempts_before(rows, 12, PRIOR_SHA))
        self.assertEqual(1, gen_loop.attempts_before(rows, 12, "other"))
        self.assertFalse(gen_loop.block_used_by_another(self.tsv, "E12-2026-11-02", PRIOR_SHA))
        self.assertTrue(gen_loop.block_used_by_another(self.tsv, "E12-2026-11-02", "other"))

    def test_the_same_prior_with_another_head_is_another_candidate_the_old_reading_could_not_tell(self):
        rows = gen_loop.read_promotions(self.tsv)
        # what the readiness check found: by the prior alone the second attempt reads as the first, on the refused block
        self.assertEqual(0, gen_loop.attempts_before(rows, 12, PRIOR_SHA, ""))
        self.assertFalse(gen_loop.block_used_by_another(self.tsv, "E12-2026-11-02", PRIOR_SHA, ""))
        # with the head in the identity it is attempt 2, and the refused block is used by another
        self.assertEqual(1, gen_loop.attempts_before(rows, 12, PRIOR_SHA, "head-b"))
        self.assertTrue(gen_loop.block_used_by_another(self.tsv, "E12-2026-11-02", PRIOR_SHA, "head-b"))
        self.assertIsNone(gen_loop.attempts_spent(self.tsv, 12, True))  # one candidate judged: one attempt left

    def test_resolve_e_block_gives_attempt_2_a_fresh_block_and_attempt_1_keeps_its_own(self):
        seeds = self.root / "seeds"
        seeds.mkdir()
        tpl = str(seeds / "E{k}-{date}.json")
        (seeds / "E12-2026-11-02.json").write_text(json.dumps({"seeds": ["A"]}))
        kw = {"tsv": self.tsv, "also_dirs": [seeds]}
        self.assertEqual("E12-2026-11-02.json", gen_loop.resolve_e_block(tpl, 12, n=30, cand_sha=PRIOR_SHA, **kw).name)  # attempt 1 re-read
        fresh = gen_loop.resolve_e_block(tpl, 12, n=30, cand_sha=PRIOR_SHA, head_sha="head-b", **kw)
        self.assertNotEqual("E12-2026-11-02.json", fresh.name)
        self.assertRegex(fresh.name, r"^E12-\d{4}-\d\d-\d\d(-r\d+)?\.json$")
        self.assertTrue(fresh.is_file())
        self.assertFalse(set(gen0.load_seeds(fresh)) & {"A"})

    def promote(self, cand: Path, name: str, d_mean: float = -4.0):
        seeds, cand_rows, base_rows = make_block(d_mean, prefix=name[-1])
        seed_file = self.root / "seeds" / f"{name}.json"
        seed_file.parent.mkdir(exist_ok=True)
        seed_file.write_text(json.dumps({"seeds": seeds}))
        with mock.patch.object(gen_loop, "run_arm", StubArms(cand_rows, base_rows)):
            return gen_loop.promote(cand, None, seed_file, agent_root=self.agents, boot=500, review_guard=True, sequential=True)

    def test_attempt_2_is_judged_as_attempt_2_with_its_own_row_and_the_cap_then_counts_both(self):
        """The whole judgement on a promotions.tsv of the old shape: the row of attempt 2 says attempt 2, names the head, the old row is
        unchanged, the block of attempt 1 is refused to it, and a third candidate is refused before a seed is played."""
        a2 = vrun_stub(self.root, "g12-a2", b"head of attempt two")
        with self.assertRaisesRegex(ValueError, "already judged another candidate"):
            self.promote(a2, "E12-2026-11-02")  # the refused block: not for the second attempt
        doc = self.promote(a2, "E12-2026-11-0X")
        self.assertEqual(gen_loop.file_sha256(a2 / "run_value.npz"), doc["run_head_sha"])
        rows = gen_loop.read_promotions(self.tsv)
        self.assertEqual(["E12-2026-11-02", "E12-2026-11-0X"], [r["seed_block"] for r in rows])
        self.assertEqual(["1", "2"], [r["attempt"] for r in rows])
        self.assertEqual(["", gen_loop.file_sha256(a2 / "run_value.npz")], [r["run_head_sha"] for r in rows])
        self.assertEqual(OLD_ROW.split("\t")[:19], [rows[0][c] for c in OLD_HEADER.split("\t")])  # attempt 1's row is as it was
        self.assertEqual(self.tsv.read_text().splitlines()[0].split("\t"), list(gen_loop.TSV_COLUMNS))
        self.assertEqual(2, gen_loop.attempts_spent(self.tsv, 12, True))  # the cap counts both attempts of the pair
        a3 = vrun_stub(self.root, "g12-a3", b"head of attempt three")
        seeds, cand_rows, base_rows = make_block(-4.0, prefix="Y")
        fresh = self.root / "seeds" / "E12-2026-11-0Y.json"  # a block no one used: only the cap can refuse it
        fresh.write_text(json.dumps({"seeds": seeds}))
        arms = StubArms(cand_rows, base_rows)
        with mock.patch.object(gen_loop, "run_arm", arms), self.assertRaisesRegex(ValueError, "already had 2 promotion attempts"):
            gen_loop.promote(a3, None, fresh, agent_root=self.agents, boot=500, review_guard=True, sequential=True)
        self.assertEqual([], arms.calls)  # refused before any seed was played

    def test_the_same_pair_judged_again_is_not_a_new_attempt(self):
        a2 = vrun_stub(self.root, "g12-a2", b"head of attempt two")
        self.promote(a2, "E12-2026-11-0X")
        rows = gen_loop.read_promotions(self.tsv)
        head = gen_loop.file_sha256(a2 / "run_value.npz")
        self.assertEqual(1, gen_loop.attempts_before(rows, 12, PRIOR_SHA, head))  # attempt 1's pair only
        self.assertFalse(gen_loop.block_used_by_another(self.tsv, "E12-2026-11-0X", PRIOR_SHA, head))

    def test_a_candidate_with_the_rules_macro_has_no_head_and_reads_as_before(self):
        rules = stub_candidate(self.root, "rules", "sha-r", 5)
        agent = gen_loop.load_agent(rules)
        self.assertEqual(("sha-r", ""), gen_loop.candidate_identity(agent))


class PlacementTests(unittest.TestCase):
    """Item 2: `gen0.py build --attempt 2` and the train job's marker for it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _, _, trained = trained_candidate()
        self.agents = self.root / "agent"
        self.base = self.agents / "gen5" / "candidate"
        self.base.mkdir(parents=True)
        for name in ("combat_policy.pt", "combat_net.pt", "model.json"):
            (self.base / name).write_bytes((trained / name).read_bytes())
        (self.base / "run_value.npz").write_bytes(b"the base fit")
        (self.base / "run_value.pt").write_bytes(b"the base fit, torch")
        (self.base / "agent.json").write_text(json.dumps({"generation": 5, "macro": "vrun", "mask_version": 4}))
        (self.agents / "current.json").write_text(json.dumps({"generation": 5, "dir": str(self.base)}))
        self.prior = self.root / "players" / "v2"
        self.prior.mkdir(parents=True)
        for name in ("combat_policy.pt", "combat_net.pt", "model.json"):
            (self.prior / name).write_bytes((trained / name).read_bytes())
        self.marker_dir = self.root / "train"

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, *extra: str) -> int:
        return gen0.main(["build", "--prior", str(self.prior), "--agent-root", str(self.agents), "--candidate-generation", "7",
                          "--why", "test", "--no-export", "--marker-dir", str(self.marker_dir), *extra])

    def snapshot(self, d: Path) -> dict[str, bytes]:
        return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}

    def test_attempt_2_is_placed_beside_attempt_1_which_stays_as_it_was(self):
        self.assertEqual(0, self.build())
        first = self.agents / "gen7" / "candidate"
        (first / "promotion.json").write_text('{"decision_text": "rejected"}')  # attempt 1 was judged
        jobs.consume_marker(self.marker_dir / "ready-5-g7")  # and the lab consumed its marker
        before = self.snapshot(first)
        # the old way refuses, as the readiness check found
        self.assertEqual(2, self.build())
        # a second head: the base's own is copied here, so only the folder, the marker and the sha of the head differ in what is checked
        self.assertEqual(0, self.build("--attempt", "2"))
        second = self.agents / "gen7" / "candidate-a2"
        self.assertTrue(second.is_dir())
        self.assertEqual(before, self.snapshot(first))  # attempt 1's folder, its promotion.json: untouched
        self.assertEqual(["done-5-g7", "ready-5-g7-a2"], sorted(p.name for p in self.marker_dir.iterdir()))
        self.assertEqual(second, gen_loop.built_candidate(self.agents, 7, 0, 2))
        self.assertEqual(first, gen_loop.built_candidate(self.agents, 7))
        meta = json.loads((second / "agent.json").read_text())
        self.assertEqual((7, 2), (meta["generation"], meta["built"]["attempt"]))
        self.assertEqual(2, json.loads((second / "train.json").read_text())["built"]["attempt"])
        self.assertNotIn("attempt", json.loads((first / "agent.json").read_text())["built"])
        self.assertEqual(5, json.loads((self.agents / "current.json").read_text())["generation"])  # current.json untouched

    def test_attempt_2_markers_are_not_written_twice_and_attempt_1_names_are_unchanged(self):
        self.assertEqual(0, self.build("--attempt", "2"))
        self.assertEqual(["ready-5-g7-a2"], [p.name for p in self.marker_dir.iterdir()])
        self.assertEqual(2, self.build("--attempt", "2"))  # the marker (or its done- form) stands: not built again
        with self.assertRaises(ValueError):
            gen_loop.build_candidate(self.prior, self.agents / "current.json", self.agents, 8, why="x", export=False, attempt=0)

    def test_the_train_job_takes_the_attempt_marker_and_judges_candidate_a2_on_a_fresh_block(self):
        self.assertEqual(0, self.build("--attempt", "2"))
        seeds = self.root / "seeds"
        seeds.mkdir()
        tsv = self.agents / "promotions.tsv"
        cand = self.agents / "gen7" / "candidate-a2"
        sha, head = gen_loop.candidate_identity(gen_loop.load_agent(cand))
        # generation 7's attempt 1 had the same fight player and the base's head: judged on E7-2026-11-02
        gen_loop.append_promotion(tsv, {"cand_gen": 7, "seed_block": "E7-2026-11-02", "cand_sha": sha, "decision": "rejected", "attempt": 1})
        (seeds / "E7-2026-11-02.json").write_text(json.dumps({"seeds": ["A"]}))
        args = ["train", "--marker-dir", str(self.marker_dir), "--shards-root", str(self.root), "--agent-root", str(self.agents),
                "--promote", "--emu-python", "python", "--seeds-file", str(seeds / "E{k}-{date}.json"), "--review-guard"]
        with mock.patch.object(gen_loop, "train_generation") as tg, \
                mock.patch.object(gen_loop, "run_promote_subprocess", return_value=0) as rp, \
                mock.patch.object(gen_loop, "COMMITTED_SEEDS", seeds):
            self.assertEqual(0, gen0.main(args))
        tg.assert_not_called()
        self.assertEqual(cand, rp.call_args[0][1])
        block = rp.call_args[0][3]
        self.assertNotEqual("E7-2026-11-02", block.stem)  # attempt 1's block is not attempt 2's
        self.assertTrue(block.stem.startswith("E7-") and block.is_file())
        self.assertEqual(head, gen_loop.file_sha256(cand / "run_value.npz"))

    def test_an_attempt_marker_without_a_built_candidate_trains_nothing(self):
        self.marker_dir.mkdir()
        (self.marker_dir / "ready-5-g7-a2").write_text("x")
        with mock.patch.object(gen_loop, "train_generation") as tg:
            rc = gen0.main(["train", "--marker-dir", str(self.marker_dir), "--shards-root", str(self.root),
                            "--agent-root", str(self.agents), "--review-guard"])
        self.assertEqual(2, rc)
        tg.assert_not_called()
        self.assertTrue((self.marker_dir / "ready-5-g7-a2").exists())

    def test_the_marker_forms_sort_and_parse(self):
        self.marker_dir.mkdir()
        for n in ("ready-12", "ready-11-g12-a10", "ready-11-g12-a2", "ready-11-g12", "ready-11-r1", "ready-11", "done-11-g12-a2",
                  "ready-11-g12-a", "ready-11-g12-b2", "ready-11-g13"):
            (self.marker_dir / n).write_text("x")
        want = ["ready-11", "ready-11-r1", "ready-11-g12", "ready-11-g12-a2", "ready-11-g12-a10", "ready-11-g13", "ready-12"]
        self.assertEqual(want, [p.name for _, p in gen_loop.ready_markers(self.marker_dir)])
        self.assertEqual(want, [p.name for p in jobs.train_markers(self.root)][:7])
        m = self.marker_dir / "ready-11-g12-a2"
        self.assertEqual(((11, 0), 12, 2), (gen_loop.marker_parts(m), gen_loop.marker_target(m), gen_loop.marker_attempt(m)))
        self.assertEqual(1, gen_loop.marker_attempt(self.marker_dir / "ready-11-g12"))
        self.assertIsNone(gen_loop.marker_parts(self.marker_dir / "done-11-g12-a2"))
        self.assertIsNone(gen_loop._MARKER.fullmatch("ready-11-g12-a2"))  # mark_ready's budget count does not see it

    def test_the_train_jobs_params_carry_the_attempt(self):
        lab = self.root / "lab"
        (lab / "train").mkdir(parents=True)
        (lab / "train" / "ready-11-g12-a2").write_text("x")
        p = jobs.params_for("train", lab, {"cmd": ["x"]})
        self.assertEqual(("ready-11-g12-a2", "11", 12, 2), (p["marker"], p["generation"], p["candidate_generation"], p["attempt"]))
        self.assertEqual(lab / "train" / "done-11-g12-a2", jobs.consume_marker(lab / "train" / "ready-11-g12-a2"))
        (lab / "train" / "ready-11-g12").write_text("x")
        p = jobs.params_for("train", lab, {"cmd": ["x"]})
        self.assertNotIn("attempt", p)
        self.assertEqual((12, "11"), (p["candidate_generation"], p["generation"]))


class SyntheticStreamTests(unittest.TestCase):
    """Item 3, measured: a stream of finished runs at the filler's pace, a kill at every second of it, and what is on disk."""

    PACE_S = 20  # a run finishes every 20 s: 180 an hour, the filler's pace on 11-04 (159 an hour reached a shard)
    RUNS = 300

    class Clock:
        t = 0.0

        def __call__(self) -> float:
            return self.t

    @staticmethod
    def on_disk(sink) -> int:
        return len(gen_loop.shard_run_seeds(sink.dir, sink.generation))

    def lost(self, flush_age_s):
        """Run the stream; the runs a kill would lose at each second of it (finished and not in a shard)."""
        clock = self.Clock()
        lost: list[int] = []
        with tempfile.TemporaryDirectory() as d:
            sink = gen_loop.ShardSink(Path(d), 11, config="a3-v1", base_sims=30, ascension=10, agent={}, flush_age_s=flush_age_s, clock=clock)
            finished = 0
            for i in range(self.RUNS):
                for _ in range(self.PACE_S):
                    clock.t += 1
                    sink.tick()
                    lost.append(finished - (sink.runs - len(sink.pending)))
                sink.add(fake_record(f"S{i:09d}", 11), None)
                finished += 1
                lost[-1] = finished - (sink.runs - len(sink.pending))
            self.assertEqual(finished - len(sink.pending), self.on_disk(sink))  # the sink's count is what the files hold
        return lost

    def test_a_hard_kill_used_to_lose_up_to_49_runs_and_loses_at_most_the_age_now(self):
        old = self.lost(None)  # only full parts of 50
        new = self.lost(gen_loop.FLUSH_AGE_S)
        per_age = int(gen_loop.FLUSH_AGE_S) // self.PACE_S + 1
        self.assertEqual(gen_loop.PART_RUNS - 1, max(old))
        self.assertGreater(sum(old) / len(old), 20)  # a kill at a random second lost 24 runs on average
        self.assertLessEqual(max(new), per_age)  # the age: 6 runs at this pace
        self.assertLess(sum(new) / len(new), 4)
        print(f"\nsynthetic stream, {self.RUNS} runs at one per {self.PACE_S} s: runs lost to a kill at a random second, "
              f"old mean {sum(old) / len(old):.1f} max {max(old)}; now mean {sum(new) / len(new):.1f} max {max(new)}")

    def test_a_flush_at_the_stop_request_loses_nothing_that_had_finished(self):
        clock = self.Clock()
        with tempfile.TemporaryDirectory() as d:
            sink = gen_loop.ShardSink(Path(d), 11, config="a3-v1", base_sims=30, ascension=10, agent={}, flush_age_s=None, clock=clock)
            for i in range(37):
                sink.add(fake_record(f"S{i:09d}", 11), None)
            self.assertEqual(0, self.on_disk(sink))  # nothing flushed yet (37 of 50)
            sink.flush()
            self.assertEqual(37, self.on_disk(sink))

    def test_a_part_is_written_when_the_oldest_run_has_waited_and_not_before(self):
        clock = self.Clock()
        with tempfile.TemporaryDirectory() as d:
            sink = gen_loop.ShardSink(Path(d), 11, config="a3-v1", base_sims=30, ascension=10, agent={}, flush_age_s=120.0, clock=clock)
            sink.add(fake_record("S000000001", 11), None)
            clock.t = 119
            self.assertFalse(sink.tick())
            sink.add(fake_record("S000000002", 11), None)  # the age is the oldest run's, not the newest's
            clock.t = 120
            self.assertTrue(sink.tick())
            self.assertEqual(2, self.on_disk(sink))
            self.assertFalse(sink.tick())  # nothing pending: nothing written, no empty part
            self.assertEqual(1, len(sink.parts))


class FakePool:
    """gen0.run_block's SupervisedPool with each task finishing 1 to 3 collections after it was submitted (by its index), so a stop
    request finds runs in flight. `on_collect(n)` is called at the start of each collection (a test's hook: the kill)."""

    stats = {"started": 0, "recycled": 0, "died": 0, "timed_out": 0, "raised": 0}
    size = 4
    on_collect = None
    instances: list = []

    def __init__(self, ctx, size, **kw):
        self.slots: list[list] = []
        self.n = 0
        type(self).instances.append(self)

    def free(self):
        return type(self).size - len(self.slots)

    def inflight(self):
        return len(self.slots)

    def submit(self, key, fn, args=()):
        self.slots.append([key, fn, args, 1 + key[0] % 3])

    def collect(self):
        self.n += 1
        if type(self).on_collect:
            type(self).on_collect(self.n)
        out = []
        for slot in list(self.slots):
            slot[3] -= 1
            if slot[3] <= 0:
                self.slots.remove(slot)
                out.append(gen0.spawn_pool.Done(slot[0], "ok", slot[1](*slot[2])))
        return out

    def wait(self, timeout):
        pass

    def terminate(self):
        pass


class Killed(BaseException):
    """The runner's Job Object terminating the block: nothing after it runs in the block."""


class RunBlockStopTests(unittest.TestCase):
    """Item 3: `run_block(stop_file=...)` and the sink's flush, in the block's own loop."""

    SEEDS = [f"STOPSEED{i:02d}" for i in range(60)]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "job"
        self.played: list[str] = []
        FakePool.on_collect, FakePool.instances = None, []
        self.addCleanup(self.tmp.cleanup)

    def fake_run_safe(self, seed, ix, sims, ascension, agent=None, record=False, character="IRONCLAD"):
        self.played.append(seed)
        if self.on_run:
            self.on_run(ix)
        return {"seed": seed, "ix": ix, "win": False, "floor": 5, "end": "death", "decisions": 3, "wall": 0.1, "ai_state_assumed": 0,
                "_shard": {"record": fake_record(seed, gen_loop.Agent().generation), "samples": None}}

    on_run = None

    def run_block(self, **kw):
        with mock.patch.object(gen0, "_run_safe", self.fake_run_safe), mock.patch.object(gen0.spawn_pool, "SupervisedPool", FakePool):
            return gen0.run_block(self.SEEDS, "a3-v1", 4, self.out, record=True, **kw)

    def held(self) -> int:
        return len(gen_loop.shard_run_seeds(self.out / "shards", 0))

    def test_a_stop_request_writes_the_finished_runs_starts_no_run_and_lets_those_in_flight_end(self):
        stop = self.out / gen0.STOP_FILE
        self.on_run = lambda ix: (self.out.mkdir(parents=True, exist_ok=True), stop.write_text("x")) if ix == 9 else None
        summ = self.run_block(stop_file=stop)
        self.assertLessEqual(summ["stopped"]["in_flight"], FakePool.size)
        rows = [json.loads(x) for x in (self.out / "runs.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), len(self.played))  # every run that was started was finished and written
        self.assertGreater(summ["stopped"]["in_flight"], 0)  # runs were in flight when it was read, and they finished
        self.assertEqual(summ["stopped"]["at_run"] + summ["stopped"]["in_flight"], len(rows))  # and nothing else was started
        self.assertLess(len(rows), len(self.SEEDS))
        self.assertEqual(self.held(), len(rows))  # all in the shards, the 50-run part never filled
        self.assertEqual(summ["shards"]["runs"], len(rows))
        self.assertEqual(len(rows), summ["stopped"]["runs_written"])
        self.assertTrue((self.out / gen0.STOP_HONORED_FILE).is_file())

    def test_a_kill_right_after_the_request_loses_none_of_the_runs_that_had_finished(self):
        stop = self.out / gen0.STOP_FILE
        self.on_run = lambda ix: (self.out.mkdir(parents=True, exist_ok=True), stop.write_text("x")) if ix == 17 else None

        def kill(n):  # the collection after the request: the block has read it at the top of its pass, the Job Object ends the process
            if stop.exists():
                raise Killed

        FakePool.on_collect = kill
        with self.assertRaises(Killed):
            self.run_block(stop_file=stop)
        rows = [json.loads(x) for x in (self.out / "runs.jsonl").read_text().splitlines()]
        self.assertGreaterEqual(len(rows), 10)  # run index 17 finished and asked
        self.assertEqual(len(rows), self.held())  # all of them reached a shard although the part (50) never filled

    def test_the_same_kill_without_a_request_loses_every_run_of_the_part_in_progress(self):

        def kill(n):
            if n == 12:
                raise Killed

        FakePool.on_collect = kill
        with self.assertRaises(Killed):
            self.run_block(stop_file=None)
        rows = (self.out / "runs.jsonl").read_text().splitlines()
        self.assertGreater(len(rows), 10)
        self.assertEqual(0, self.held())  # the control: this is what the filler lost at every stop

    def test_the_loop_ticks_the_sink_and_a_stale_stop_file_is_removed_at_the_start(self):
        stop = self.out / gen0.STOP_FILE
        self.out.mkdir(parents=True)
        stop.write_text("left by an earlier block")
        ticks = []
        with mock.patch.object(gen_loop.ShardSink, "tick", autospec=True, side_effect=lambda s: ticks.append(1) or False):
            summ = self.run_block(stop_file=stop)
        self.assertNotIn("stopped", summ)
        self.assertEqual(len(self.SEEDS), self.held())  # every seed played: the stale file did not stop the block
        self.assertGreater(len(ticks), 0)

    def test_a_block_without_a_stop_file_behaves_as_before(self):
        summ = self.run_block()
        self.assertNotIn("stopped", summ)
        self.assertFalse((self.out / gen0.STOP_HONORED_FILE).exists())
        self.assertEqual(len(self.SEEDS), self.held())

    def test_the_emu_command_gives_a_recorded_block_its_stop_file(self):
        seen = {}

        def fake_block(seeds, sims, procs, out, **kw):
            seen.update(kw)
            return {"ok": True}

        argv = ["emu", "--seeds-file", str(ROOT / "referee" / "seeds" / "E1-2026-10-23.json"), "--n", "3", "--out", str(self.out),
                "--record", "--config", "a3-v1"]
        with mock.patch.object(gen0, "run_block", fake_block):
            self.assertEqual(0, gen0.main(argv))
            self.assertEqual(self.out / gen0.STOP_FILE, seen["stop_file"])
            seen.clear()
            self.assertEqual(0, gen0.main([a for a in argv if a != "--record"]))
            self.assertIsNone(seen["stop_file"])


class ExploringWindowTests(unittest.TestCase):
    """Item 4: `vrun-fit --exploring-only` takes the blocks an exploring agent played, and reads the same window back."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "runs"

    def block(self, job: str, runs: int, *, tau=None, eps=None, sha=PRIOR_SHA, gen=11) -> Path:
        d = self.root / job / "shards"
        d.mkdir(parents=True)
        agent = {"generation": gen, "prior_sha256": sha, "macro": "vrun", **({"tau": tau} if tau else {}), **({"eps": eps} if eps else {})}
        seeds = [f"{job[9:13]}{i:06d}" for i in range(runs)]
        gen_loop.write_part(d / f"combat-g{gen}-0000.npz", [(s, None) for s in seeds], gen)
        (d / "shards.json").write_text(json.dumps({"kind": "selfplay-shards", "generation": gen, "config": "a3-v1", "base_sims": 30,
                                                  "ascension": 10, "agent": agent, "runs": runs, "samples": 0,
                                                  "combat_parts": [f"combat-g{gen}-0000.npz"], "run_shards": []}))
        return d

    def make(self):
        self.block("20261102T1050-selfplay", 776, tau=0.1, eps=0.02)
        self.block("20261102T1111-selfplay", 752, tau=0.1, eps=0.02)
        self.block("20261103T0310-selfplay", 400)  # the filler, argmax
        self.block("20261104T0701-selfplay", 50)
        self.block("20261104T0800-selfplay", 50, eps=0.02)  # explores by eps alone
        self.block("20261104T0900-selfplay", 50, tau=0.1, sha="f" * 64)  # another fighter's exploring block

    def test_exploring_manifests_are_those_naming_a_temperature_or_a_uniform_share(self):
        self.assertTrue(gen_loop.exploring_manifest({"agent": {"tau": 0.1}}))
        self.assertTrue(gen_loop.exploring_manifest({"agent": {"eps": 0.02}}))
        self.assertFalse(gen_loop.exploring_manifest({"agent": {"prior_sha256": "x", "macro": "vrun"}}))
        self.assertFalse(gen_loop.exploring_manifest({}))

    def test_the_selection_takes_the_exploring_blocks_of_the_fighter_only_when_asked(self):
        self.make()
        every = gen_loop.fighter_manifests(self.root, PRIOR_SHA)
        exploring = gen_loop.fighter_manifests(self.root, PRIOR_SHA, exploring_only=True)
        self.assertEqual(5, len(every))
        self.assertEqual(["20261102T1050-selfplay", "20261102T1111-selfplay", "20261104T0800-selfplay"],
                         sorted(d.parent.name for d, _ in exploring))
        self.assertEqual(776 + 752 + 50, sum(m["runs"] for _, m in exploring))
        self.assertEqual([], gen_loop.fighter_manifests(self.root, "", exploring_only=True))

    def test_the_fit_records_its_window_and_the_window_is_read_back_the_same(self):
        self.make()
        seen = {}

        def fake_train(paths, out, **kw):
            seen["paths"] = paths
            (Path(out) / "run_value.npz").write_bytes(b"fit")
            return {"value": {"keep": {"status": "ok", "all_computable_pass": True}}, "wall_min": 0.0}

        out = Path(self.tmp.name) / "fit"
        with mock.patch.object(gen_loop, "train_run_value", fake_train), \
                mock.patch.object(gen_loop, "ranking_block", return_value={"status": "ok", "pass": True}), \
                mock.patch.object(gen_loop, "vrun_keep_passed", return_value=True), \
                mock.patch.object(gen_loop, "known_seed_sets", return_value={}):
            doc = gen_loop.fit_run_value_matched(self.root, PRIOR_SHA, out, exploring_only=True, incumbent=out / "x.npz")
            everything = gen_loop.fit_run_value_matched(self.root, PRIOR_SHA, Path(self.tmp.name) / "fit2", incumbent=out / "x.npz")
        self.assertTrue(doc["window"]["exploring_only"])
        self.assertEqual(3, doc["window"]["blocks"])
        self.assertEqual({"11": 776 + 752 + 50}, doc["window"]["runs_by_generation"])
        self.assertEqual(sorted(doc["window"]["jobs"]), ["20261102T1050-selfplay", "20261102T1111-selfplay", "20261104T0800-selfplay"])
        self.assertFalse(everything["window"]["exploring_only"])
        self.assertEqual(5, everything["window"]["blocks"])
        self.assertEqual(776 + 752 + 50, len(gen_loop.fit_window_seeds(doc, self.root)))  # the seeds the ranking set must leave out
        self.assertEqual(776 + 752 + 400 + 50 + 50, len(gen_loop.fit_window_seeds(everything, self.root)))
        old_doc = {**everything, "window": {k: v for k, v in everything["window"].items() if k != "exploring_only"}}
        self.assertEqual(776 + 752 + 400 + 50 + 50, len(gen_loop.fit_window_seeds(old_doc, self.root)))  # a fit before the key: every block
        with self.assertRaisesRegex(ValueError, "no exploring"):
            gen_loop.fit_run_value_matched(self.root, "e" * 64, Path(self.tmp.name) / "none", exploring_only=True)

    def test_the_command_has_the_flag(self):
        seen = {}

        def fake_fit(root, sha, out, **kw):
            seen.update(kw)
            return {"window": {"runs_by_generation": {}, "blocks": 0}, "beta": 0.0, "kept": True, "keep": {},
                    "run_value": {}}

        base = ["vrun-fit", "--shards-root", str(self.root), "--prior-sha", PRIOR_SHA, "--out", str(Path(self.tmp.name) / "o"),
                "--incumbent", "x"]
        # `vrun_fit_cmd` lowers the process it runs in; in a test that is the test runner, and every
        # later test in the process would run (and start its workers) at BelowNormal.
        with mock.patch.object(gen_loop, "fit_run_value_matched", fake_fit), mock.patch("sts2_gym.procutil.lower_priority"):
            self.assertEqual(0, gen0.main(base + ["--exploring-only"]))
            self.assertTrue(seen["exploring_only"])
            self.assertEqual(0, gen0.main(base))
            self.assertFalse(seen["exploring_only"])


if __name__ == "__main__":
    unittest.main()
