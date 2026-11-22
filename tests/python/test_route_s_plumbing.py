"""Route S's plumbing (D68 addendum 6 outcome 4, judge build lane 2, step 2): rl/gen_loop.py (`fight_only_swap`, `s1_stage` and its ruler ledger,
`base_arm_reuse`, `promote(route_s=)`, `rejudge(route_s=, s1_file=, complete=)`) and rl/gen0.py (`--route-s`, `rejudge --s1 --complete-block`);
docs/rl/KEEP_TEST_V3.md "Route S".

Nothing plays a game: the replay pool and the exam are replaced by a `run` that answers as the child would (and writes what it would write), the arms
by stub rows, as `test_keep_v3` and `test_funnel` do. The synthetic arms are `test_keep_v3`'s (`block`): a flat-ish block whose deep_f gain is -0.6 per 100
(v3's futility stop ends it at 800 pairs; S2's looks do not), and a +1.45 block v3 extends."""

from __future__ import annotations

import contextlib
import io
import json
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

import act3_starts as a3  # noqa: E402
import fight_exam as fe  # noqa: E402
import fight_gym as fg  # noqa: E402
import gen0  # noqa: E402
import gen_loop  # noqa: E402
import gym_loop as gl  # noqa: E402
import keep_v2_fixtures as fx  # noqa: E402
import test_gen12_plumbing as tg12  # noqa: E402
from test_arrival_exam import nodes_row, replay_row  # noqa: E402
from test_gen_loop import stub_candidate  # noqa: E402
from test_keep_v3 import F_B3A, block  # noqa: E402
from test_keep_v3_route_s import CAND, CHAMP, RULER, result  # noqa: E402, F401

from tools.lab import keep_v2 as k2  # noqa: E402

CAND_SHA, CHAMP_SHA = "sha-c1", "sha-g3"
FLATISH = ((40, 3, 2), (52, 2, 3), (1000, 1, 1), (400, 0, 1), (508, 1, 0))  # deep_f -0.6 per 100 in 2,000 pairs, SE about 0.48
HARM = ((20, 3, 2), (140, 2, 3), (1000, 1, 1), (400, 0, 1), (440, 1, 0))  # deep_f -6.0 per 100


def holds(expect: dict, points: float = 1.8, se: float = 0.4, **kw) -> dict:
    """An S1 reading that names `expect`'s pair and holds."""
    return k2.s1_reading(result(points, se, ruler=expect["ruler_hash"], cand=expect["candidate"], champ=expect["champion"], **kw), expect)


def make_ruler(folder: Path, champion: str = "gen3-candidate", champion_dir: str = "gen3/candidate") -> dict:
    """A small valid ruler (three starts) in `folder`, for the champion `champion_dir`."""
    rows = {r["seed"]: r for r in [nodes_row("p1", "past", 1), nodes_row("d1", "died", 2)]}
    recs, counts = a3.arrival_records([replay_row(rows["p1"], "B"), replay_row(rows["d1"], "B")], "B", rows, "old")
    return a3.build_ruler(folder, champion, champion_dir, [{"label": "B", "records": recs, "counts": counts, "rows": 10, "arrivals": 2}], "old", "a3-v1")


class World(unittest.TestCase):
    """The agents' folder: the champion generation 3 (its arms folder holds the lineage's block, so the frontier is the first Act 3 boss), the candidate
    generation 4 a fight-only swap, and the block's two arms."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("EXAM_DIR", "LOOP_ROOT"):  # the judgement's human-bar line reads the gym's folder: not the real one in a test
            patcher = mock.patch.object(fe, name, self.root / "no-gym")
            patcher.start()
            self.addCleanup(patcher.stop)
        self.agents = self.root / "agent"
        self.agents.mkdir()
        self.champ = stub_candidate(self.agents / "gen3", "candidate", CHAMP_SHA, generation=3)
        prev = [fx.run_row(f"P{i:09d}", 3 if i < 60 else 2 if i < 300 else 1 if i < 1000 else 0, np.random.default_rng(i)) for i in range(2000)]
        self.write_arm(self.champ, "Eprev", prev)
        gen_loop.append_promotion(self.agents / "promotions.tsv",
                                  {"cand_gen": 3, "base_gen": 2, "seed_block": "Eprev", "seeds": 2000, "boss_cand": 1000, "boss_base": 900,
                                   "decision": "promoted", "cand_sha": CHAMP_SHA, "cand_dir": "candidate", "attempt": 1, "drift": "clear"})
        self.cand = self.swap_candidate("c1", CAND_SHA)
        (self.root / "seeds").mkdir()
        self.ruler_root = self.root / "arrival"

    def swap_candidate(self, name: str, sha: str, swap: str | None = "fight") -> Path:
        cand = stub_candidate(self.agents / "gen4", name, sha, generation=4)
        meta = json.loads((cand / "agent.json").read_text())
        if swap is not None:
            meta["built"] = {"swap": swap, "trained": False, "tag": name}
        (cand / "agent.json").write_text(json.dumps(meta))
        return cand

    @staticmethod
    def write_arm(folder: Path, name: str, rows: list[dict]) -> Path:
        arm = folder / "arms" / f"{name}-a3-v1"
        arm.mkdir(parents=True)
        (arm / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return arm

    def seed_file(self, name: str, seeds: list[str]) -> Path:
        p = self.root / "seeds" / f"{name}.json"
        p.write_text(json.dumps({"seeds": seeds}))
        return p

    def stub_arms(self, cand_rows, base_rows):
        calls = []

        def fake_run_arm(seeds, arm_dir, agent, config, procs, ascension=10, resume=False, **kw):
            calls.append((agent.generation, len(seeds), resume, kw))
            want = set(seeds)
            return [r for r in (cand_rows if agent.generation == 4 else base_rows) if r["seed"] in want]

        return fake_run_arm, calls

    def expect(self) -> dict:
        return gen_loop.s1_expect(gen_loop.load_agent(self.cand), gen_loop.load_agent(self.champ), RULER)


class FightSwapTests(World):
    def test_the_field_gym_wiring_writes_is_the_one_read(self):
        self.assertEqual(gen_loop.FIGHT_SWAP, "fight")
        self.assertTrue(gen_loop.fight_only_swap(self.cand))
        for swap in ("vrun", "chain", "prior", ""):
            self.assertFalse(gen_loop.fight_only_swap(self.swap_candidate(f"x-{swap or 'none'}", f"sha-{swap}", swap=swap)), swap)
        self.assertFalse(gen_loop.fight_only_swap(self.swap_candidate("plain", "sha-p", swap=None)))  # no built block
        self.assertFalse(gen_loop.fight_only_swap(self.champ))  # the champion's own folder
        self.assertFalse(gen_loop.fight_only_swap(self.root / "nowhere"))  # no agent.json
        (self.cand / "agent.json").write_text("{not json")
        self.assertFalse(gen_loop.fight_only_swap(self.cand))

    def test_the_flag_is_auto_or_off_and_off_by_default(self):
        self.assertEqual([gen_loop.route_s_of(v) for v in (None, "", "off", "OFF", "auto", " auto ")], ["off", "off", "off", "off", "auto", "auto"])
        self.assertEqual(gen_loop.ROUTE_S_MODES, ("auto", "off"))
        with self.assertRaisesRegex(ValueError, "unknown --route-s"):
            gen_loop.route_s_of("on")


class S1StageTests(World):
    """`s1_stage` with the two heavy children replaced by a `run` that answers as `act3_starts.py` would."""

    def setUp(self):
        super().setUp()
        self.calls: list[list[str]] = []
        self.codes = {"arrivals": 0, "arrival-exam": 0}
        self.points = (1.8, 0.4)
        self.shas = {}

    def answer(self, cmd, check=False):
        self.calls.append([str(x) for x in cmd])
        verb = cmd[2]
        if verb == "arrivals":
            if self.codes["arrivals"] == 0:
                make_ruler(Path(cmd[cmd.index("--out") + 1]))
            return SimpleNamespace(returncode=self.codes["arrivals"])
        if self.codes["arrival-exam"] == 0:
            k = int(cmd[cmd.index("--test") + 1])
            ruler = Path(cmd[cmd.index("--ruler") + 1])
            meta = json.loads((ruler / "meta.json").read_text())
            out = Path(cmd[cmd.index("--report-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "results.json").write_text(json.dumps(result(*self.points, ruler=meta["ruler_hash"], cand=CAND_SHA, champ=CHAMP_SHA, test=k)))
        return SimpleNamespace(returncode=self.codes["arrival-exam"])

    def stage(self, cand: Path | None = None, **kw):
        cand = cand or self.cand
        return gen_loop.s1_stage(cand, self.champ, gen_loop.load_agent(cand), gen_loop.load_agent(self.champ), ruler_root=self.ruler_root, run=self.answer,
                                 python="py", say=lambda m: None, **kw)

    def test_the_ruler_is_built_then_the_exam_is_played_and_the_reading_is_written(self):
        got = self.stage()
        self.assertEqual([c[2] for c in self.calls], ["arrivals", "arrival-exam"])
        ruler = self.ruler_root / "gen3-candidate"
        build, exam = self.calls
        self.assertEqual(build[build.index("--champion") + 1], str(self.champ))
        self.assertEqual(build[build.index("--out") + 1], str(ruler))
        self.assertEqual((build[build.index("--procs") + 1], exam[exam.index("--procs") + 1]), ("12", "12"))  # 12 processes
        self.assertEqual((exam[exam.index("--test") + 1], exam[exam.index("--candidate") + 1], exam[exam.index("--champion") + 1]), ("0", str(self.cand), str(self.champ)))
        self.assertEqual(exam[exam.index("--out") + 1], str(ruler / "exam"))
        self.assertEqual(Path(exam[exam.index("--report-dir") + 1]), self.cand / "s1")
        self.assertTrue(got["match"]["verified"])
        self.assertEqual((got["mean"], got["se"], got["test"], got["ruler_hash"]), (1.8, 0.4, 0, a3.load_ruler(ruler)[0]["ruler_hash"]))
        self.assertTrue(k2.route_s1(got)["holds"])
        self.assertEqual(json.loads((self.cand / "s1.json").read_text())["mean"], 1.8)  # the reading is written to the candidate's folder
        ledger = json.loads((ruler / "s1-ledger.json").read_text())
        self.assertEqual([(t["test"], t["candidate"], t["champion"], t["done"] is not None) for t in ledger["tests"]], [(0, CAND_SHA, CHAMP_SHA, True)])

    def test_a_ruler_that_names_this_champion_and_hashes_is_not_built_again(self):
        ruler = self.ruler_root / "gen3-candidate"
        make_ruler(ruler)
        self.stage()
        self.assertEqual([c[2] for c in self.calls], ["arrival-exam"])
        # a ruler of another champion, or an edited one, is built again over it
        self.calls.clear()
        other = self.ruler_root / "gen3-candidate"
        make_ruler(other, "gen9-other", "gen9/other")
        self.assertIsNone(gen_loop.usable_ruler(other, self.champ))
        (other / "starts-arrival.jsonl").write_text((other / "starts-arrival.jsonl").read_text().replace('"hp":61', '"hp":99', 1))
        self.assertIsNone(gen_loop.usable_ruler(other, self.champ))

    def test_the_exam_is_told_to_ignore_the_labs_own_quiet_line_and_then_it_does_not_stop(self):
        self.stage()
        for cmd in self.calls:  # both heavy children
            i = cmd.index("--allow-quiet")
            self.assertEqual(cmd[i + 1], gen_loop.LAB_TRAIN_TAG)
        self.assertEqual(gen_loop.LAB_TRAIN_TAG, "lab-train")
        from tools.lab import runner

        self.assertEqual(gen_loop.LAB_TRAIN_TAG, runner.TRAIN_TAG)  # the runner's own tag
        exam = self.calls[-1]
        allowed = exam[exam.index("--allow-quiet") + 1:exam.index("--lib")]
        with tempfile.TemporaryDirectory() as tmp:
            q = Path(tmp) / "QUIET.lock"
            q.write_text(f"lab-train {gl.stamp()} pid=5\n")
            with mock.patch.object(fg, "QUIET", q), mock.patch.object(gl, "in_night_window", lambda now=None: False), \
                    mock.patch.object(fg, "ACTIVE", Path(tmp) / "none"), mock.patch.object(gl, "PAUSE", Path(tmp) / "pause"), \
                    mock.patch.object(gl, "HALT", Path(tmp) / "halt"), mock.patch.object(fg, "MAIN_LOCK", Path(tmp) / "main.lock"):
                self.assertEqual(fe.blocked_reason(starting=True), "quiet:lab-train")  # without the flag the run reads the lab's line as another's and stops
                with mock.patch.object(fe, "ALLOW_QUIET", list(allowed)):  # what `cmd_arrival_exam` does with the flag
                    self.assertIsNone(fe.blocked_reason(starting=True))
                    q.write_text(f"lab-train {gl.stamp()} pid=5\nsomeone-else {gl.stamp()} pid=6\n")
                    self.assertEqual(fe.blocked_reason(starting=True), "quiet:someone-else")  # only the lab's own line is ignored

    def test_the_commands_parse_under_the_drivers_own_parser(self):
        """Both commands the stage builds are accepted by `act3_starts.main`, with the flag landing in `allow_quiet` (and the library folder in `lib`)."""
        self.stage()
        for cmd in self.calls:
            seen = {}

            def capture(args, seen=seen):
                seen["args"] = args
                return 0

            with mock.patch.object(a3, "cmd_" + cmd[2].replace("-", "_"), capture):
                self.assertEqual(a3.main(cmd[2:]), 0)
            self.assertEqual(seen["args"].allow_quiet, ["lab-train"])
            self.assertEqual(seen["args"].lib, os.environ["STS2_LIB_PATH"])
            self.assertEqual(seen["args"].procs, 12)

    def test_the_second_candidate_takes_new_draws_and_a_resumed_one_keeps_its_own(self):
        self.stage()
        c2 = self.swap_candidate("c2", "sha-c2")
        self.calls.clear()
        self.points = (1.5, 0.4)
        gen2 = gen_loop.s1_stage(c2, self.champ, gen_loop.load_agent(c2), gen_loop.load_agent(self.champ), ruler_root=self.ruler_root,
                                 run=lambda cmd, check=False: self.run_for(cmd, "sha-c2"), python="py", say=lambda m: None)
        self.assertEqual(gen2["test"], 1)  # draws 6 to 11: no draw is used by two candidates
        self.assertEqual(self.calls[-1][self.calls[-1].index("--test") + 1], "1")
        self.assertEqual(gen2["draws"], [6, 7, 8, 9, 10, 11])
        # a candidate whose games were started and stopped resumes the same test
        c3 = self.swap_candidate("c3", "sha-c3")
        ruler = self.ruler_root / "gen3-candidate"
        ledger = json.loads((ruler / "s1-ledger.json").read_text())
        ledger["tests"].append({"test": 2, "candidate": "sha-c3", "champion": CHAMP_SHA, "ruler_hash": a3.load_ruler(ruler)[0]["ruler_hash"], "started": "t", "done": None})
        (ruler / "s1-ledger.json").write_text(json.dumps(ledger))
        gen3 = gen_loop.s1_stage(c3, self.champ, gen_loop.load_agent(c3), gen_loop.load_agent(self.champ), ruler_root=self.ruler_root,
                                 run=lambda cmd, check=False: self.run_for(cmd, "sha-c3"), python="py", say=lambda m: None)
        self.assertEqual(gen3["test"], 2)
        self.assertEqual(len(json.loads((ruler / "s1-ledger.json").read_text())["tests"]), 3)

    def run_for(self, cmd, cand_sha):
        """`run`, with the results naming another candidate's network."""
        rc = self.answer(cmd)
        if cmd[2] == "arrival-exam" and rc.returncode == 0:
            report = Path(cmd[cmd.index("--report-dir") + 1]) / "results.json"
            doc = json.loads(report.read_text())
            doc["candidate"]["policy_sha256"] = cand_sha
            report.write_text(json.dumps(doc))
        return rc

    def test_a_test_played_by_hand_is_counted_from_its_games(self):
        ruler = self.ruler_root / "gen3-candidate"
        make_ruler(ruler)
        (ruler / "exam").mkdir()
        (ruler / "exam" / "games-arr-cand-abcd1234.jsonl").write_text("".join(json.dumps({"start": "s", "j": j}) + "\n" for j in range(12)))
        self.stage()
        self.assertEqual(json.loads((self.cand / "s1.json").read_text())["test"], 2)  # draws 0 to 11 were played: the next test is 2
        self.assertEqual(gen_loop.s1_used_tests(ruler, gen_loop.s1_ledger(ruler)), {0, 1, 2})

    def test_a_reading_that_stands_for_this_ruler_champion_and_candidate_is_not_played_again(self):
        self.stage()
        self.calls.clear()
        again = self.stage()
        self.assertEqual(self.calls, [])
        self.assertEqual((again["mean"], again["test"]), (1.8, 0))
        # a reading for another ruler does not match: the exam is played again
        stored = json.loads((self.cand / "s1.json").read_text())
        stored["ruler_hash"] = "e" * 64
        (self.cand / "s1.json").write_text(json.dumps(stored))
        self.stage()
        self.assertEqual([c[2] for c in self.calls], ["arrival-exam"])

    def test_a_stop_or_a_block_raises_and_a_pair_it_cannot_play_is_unavailable(self):
        self.codes["arrival-exam"] = 3
        with self.assertRaises(gen_loop.S1Blocked):
            self.stage()
        self.assertFalse((self.cand / "s1.json").exists())
        ledger = gen_loop.s1_ledger(self.ruler_root / "gen3-candidate")
        self.assertEqual([t["done"] for t in ledger["tests"]], [None])  # the test is entered before its games, so a resume plays the same draws
        self.codes["arrival-exam"] = 0
        self.stage()  # the next attempt resumes test 0
        self.assertEqual(json.loads((self.cand / "s1.json").read_text())["test"], 0)
        # the ruler's replays stopped
        c2 = self.swap_candidate("c2", "sha-c2")
        shutil_rmtree = __import__("shutil").rmtree
        shutil_rmtree(self.ruler_root)
        self.codes["arrivals"] = 3
        with self.assertRaises(gen_loop.S1Blocked):
            self.stage(c2)
        self.codes["arrivals"] = 2
        with self.assertRaises(gen_loop.S1Unavailable):
            self.stage(c2)
        self.codes["arrivals"] = 1
        with self.assertRaisesRegex(RuntimeError, "status 1"):
            self.stage(c2)
        # a champion with no arms has no ruler to build
        champ = stub_candidate(self.agents / "gen2", "candidate", "sha-g2", generation=2)
        with self.assertRaises(gen_loop.S1Unavailable):
            gen_loop.s1_stage(c2, champ, gen_loop.load_agent(c2), gen_loop.load_agent(champ), ruler_root=self.ruler_root, run=self.answer, python="py")

    def test_the_same_fight_network_is_nothing_to_compare(self):
        twin = self.swap_candidate("twin", CHAMP_SHA)
        with self.assertRaises(gen_loop.S1Unavailable):
            self.stage(twin)


class BaseReuseTests(World):
    def setUp(self):
        super().setUp()
        self.seeds, self.c, self.b = block(FLATISH, floor_mean=0.0, prefix="K", seed=7)
        self.arm = "Etest-r1-a3-v1"[: -len("-a3-v1")] + "-a3-v1"
        self.stored = self.write_arm(self.champ, "Etest-r1", self.b)

    def reuse(self, replay_rows, **kw):
        calls = []

        def play(seeds, arm_dir, agent, config, procs, ascension, **k):
            calls.append((list(seeds), Path(arm_dir), k))
            return [r for r in replay_rows(seeds)]

        rec = gen_loop.base_arm_reuse(self.seeds, self.cand, self.champ, gen_loop.load_agent(self.champ), "Etest-r1-a3-v1", "a3-v1", 16, 10,
                                      arm_kw={"server_opts": "opts"}, say=lambda m: None, play=play, **kw)
        return rec, calls

    def by_seed(self):
        return {r["seed"]: r for r in self.b}

    def test_twenty_seeds_that_replay_identically_let_the_stored_arm_be_reused(self):
        rec, calls = self.reuse(lambda seeds: [{**self.by_seed()[s], "wall": 99.0, "prior_calls": {"cpu": 5}} for s in seeds])  # timing and transport differ, outcomes do not
        self.assertEqual((rec["action"], rec["checked"], rec["same"], rec["differing"]), ("reused", 20, 20, []))
        self.assertEqual((rec["stored_rows"], rec["candidate_rows"], rec["reusable"], rec["wanted"]), (2000, 0, 2000, 20))
        self.assertEqual(calls[0][0], self.seeds[:20])  # the first 20 seeds of the block, in order
        self.assertEqual(calls[0][1], self.cand / "arms-check" / "Etest-r1-a3-v1")  # a scratch arm of the candidate's, never the champion's
        self.assertEqual(calls[0][2], {"start_ix": 0, "server_opts": "opts"})  # each keeps its index in the block; the arm's own prior transport
        self.assertTrue(self.stored.is_dir())  # the stored arm stays where it is
        self.assertEqual(rec["fields"], list(gen_loop.OUTCOME_FIELDS))

    def test_one_seed_that_differs_sets_the_stored_arm_aside_and_the_champion_plays_afresh(self):
        def replay(seeds):
            rows = [dict(self.by_seed()[s]) for s in seeds]
            rows[7]["floor"] += 1
            return rows

        rec, _ = self.reuse(replay)
        self.assertEqual((rec["action"], rec["checked"], rec["same"], rec["differing"]), ("played afresh", 20, 19, [self.seeds[7]]))
        self.assertFalse(self.stored.exists())  # the stored arm is no longer where run_arm looks
        stale = list((self.champ / "arms-stale").iterdir())
        self.assertEqual(len(stale), 1)
        self.assertTrue((stale[0] / "runs.jsonl").is_file())
        self.assertEqual(rec["stale"], f"candidate/arms-stale/{stale[0].name}")
        self.assertNotIn(str(self.root), json.dumps(rec))  # no absolute path in the record
        # a missing replay row, a replay on another outcome field, each differs
        for field, value in (("win", True), ("end", "crash"), ("nodes", "M"), ("ix", 99)):
            self.setUp()
            rec2, _ = self.reuse(lambda seeds, f=field, v=value: [{**self.by_seed()[s], f: v} for s in seeds])
            self.assertEqual(rec2["action"], "played afresh", field)
        self.setUp()
        rec3, _ = self.reuse(lambda seeds: [dict(self.by_seed()[s]) for s in seeds[1:]])
        self.assertEqual((rec3["action"], rec3["differing"]), ("played afresh", [self.seeds[0]]))

    def test_nothing_is_checked_where_the_candidate_played_the_rows_beside_the_champion(self):
        self.write_arm(self.cand, "Etest-r1", self.c)
        rec, calls = self.reuse(lambda seeds: [])
        self.assertEqual((rec["action"], rec["checked"], rec["reusable"]), ("none", 0, 0))
        self.assertEqual(calls, [])
        self.assertTrue(self.stored.is_dir())

    def test_a_block_the_candidate_played_in_part_checks_from_the_first_row_it_did_not_play(self):
        half = self.c[:600]
        self.write_arm(self.cand, "Etest-r1", half)
        rec, calls = self.reuse(lambda seeds: [dict(self.by_seed()[s]) for s in seeds])
        self.assertEqual((rec["action"], rec["reusable"], rec["checked"], rec["start_index"]), ("reused", 1400, 20, 600))
        self.assertEqual((calls[0][0], calls[0][2]["start_ix"]), (self.seeds[600:620], 600))  # the index of each seed is its place in the block
        self.assertEqual(rec["candidate_rows"], 600)

    def test_a_champion_with_no_stored_arm_has_nothing_to_reuse(self):
        import shutil

        shutil.rmtree(self.stored)
        rec, calls = self.reuse(lambda seeds: [])
        self.assertEqual((rec["action"], rec["stored_rows"], rec["checked"]), ("none", 0, 0))
        self.assertEqual(calls, [])


class PromoteRouteSTests(World):
    """`promote(route_s=)` on stub arms with the S1 stage replaced: the flag, the swap, S1 holding or not, S2's funnel, the record."""

    def setUp(self):
        super().setUp()
        self.seeds, self.c, self.b = block(FLATISH, floor_mean=0.0, prefix="K", seed=7)
        self.f1 = self.seed_file("Etest-r1", self.seeds)
        self.stage_calls: list[tuple] = []

    def stage(self, points=(1.8, 0.4), raises=None):
        def fake(cand, base_dir, cand_agent, base_agent, **kw):
            self.stage_calls.append((Path(cand).name, Path(base_dir).name, kw.get("procs")))
            if raises is not None:
                raise raises
            return holds(self.expect(), *points)

        return fake

    def promote(self, cand_rows=None, base_rows=None, *, stage=None, **kw):
        arms, calls = self.stub_arms(cand_rows or self.c, base_rows or self.b)
        self.arm_calls = calls
        with mock.patch.object(gen_loop, "run_arm", arms), mock.patch.object(gen_loop, "s1_stage", stage or self.stage()):
            return gen_loop.promote(self.cand, self.champ, self.f1, agent_root=self.agents, boot=500, keep_rule="v3", sequential=True, **kw)

    def test_off_is_the_default_and_nothing_of_s_runs(self):
        doc = self.promote()
        self.assertEqual(self.stage_calls, [])
        for key in ("s1", "route_s_mode", "state_shift", "base_reuse"):
            self.assertNotIn(key, doc)
        self.assertNotIn("route_s", doc["keep_v2"])
        self.assertEqual((doc["decision_text"], doc["sequential"]["stop"]["action"], doc["sequential"]["stop"]["rule"]), ("rejected", "futility", "futility@800"))
        self.assertTrue(doc["sequential"]["futility"]["enabled"])
        self.assertNotIn("route", doc["sequential"])
        off = self.promote(route_s="off")  # the same on a candidate asked for explicitly off
        self.assertEqual(self.stage_calls, [])

    def test_auto_leaves_a_candidate_that_is_not_a_fight_swap_to_v3(self):
        plain = self.swap_candidate("plain", "sha-p", swap=None)
        arms, _ = self.stub_arms(self.c, self.b)
        with mock.patch.object(gen_loop, "run_arm", arms), mock.patch.object(gen_loop, "s1_stage", self.stage()):
            doc = gen_loop.promote(plain, self.champ, self.f1, agent_root=self.agents, boot=500, keep_rule="v3", sequential=True, route_s="auto")
        self.assertEqual(self.stage_calls, [])
        self.assertEqual((doc["route_s_mode"]["fight_swap"], doc["route_s_mode"]["applies"]), (False, False))
        self.assertIn("not a fight-only swap", doc["route_s_mode"]["why"])
        self.assertNotIn("s1", doc)
        self.assertEqual(doc["sequential"]["stop"]["rule"], "futility@800")  # v3's futility stop, as before

    def test_auto_and_s1_holding_walk_the_funnel_with_the_looks_refusing_only_and_promote_by_s_at_2000(self):
        doc = self.promote(route_s="auto")
        self.assertEqual(self.stage_calls, [("c1", "candidate", 12)])  # S1 first: for this candidate against this champion, on 12 processes
        seq = doc["sequential"]
        self.assertEqual((seq["stop"]["action"], seq["stop"]["n"], seq["played"], doc["n_seeds"]), ("final", 2000, 2000, 2000))  # v3 stopped it at 800 on futility
        self.assertEqual((seq["route"], seq["futility"]["enabled"], seq["keep_rule"]), ("S", False, "v3"))
        self.assertTrue(all(x["action"] in ("continue", "final") for x in seq["looks"]), [x["action"] for x in seq["looks"]])
        self.assertTrue(all("futility" not in x for x in seq["looks"]))
        self.assertEqual((doc["decision_text"], doc["decision"]["promote"], doc["decision"]["route"]), ("promoted", True, "S"))
        k = doc["keep_v2"]
        self.assertEqual((k["route"], k["extend"], k["route_s"]["applies"], k["route_s"]["decided_by"]), ("S", False, True, "S"))
        self.assertTrue(k["route_s"]["s2"]["final_ok"])
        self.assertEqual(k["route_s"]["parameters"], k2.route_s_parameters())
        self.assertEqual(doc["s1"]["mean"], 1.8)  # the reading is in promotion.json
        self.assertEqual(doc["route_s_mode"], {"mode": "auto", "fight_swap": True, "applies": True})
        self.assertEqual(doc["state_shift"], k["route_s"]["state_shift"])
        self.assertEqual(doc["base_reuse"]["action"], "none")  # the champion's arm is not stored here: nothing to reuse unchecked
        cur = json.loads((self.agents / "current.json").read_text())
        self.assertEqual((cur["generation"], Path(cur["dir"]).name), (4, "c1"))
        row = gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]
        self.assertEqual((row["decision"], row["rule"], row["route"], row["seeds"]), ("promoted", "v3", "S", "2000"))
        self.assertIn("S1 +1.80 ±0.80 per 100 arrivals", row["guard"])
        self.assertIn("S2 deep 95% upper", row["guard"])
        self.assertEqual(gen_loop.row_rule(row), "v3")

    def test_s2s_looks_refuse_a_harm_and_the_block_ends_there(self):
        _, c, b = block(HARM, floor_mean=0.0, prefix="K", seed=7)
        doc = self.promote(c, b, route_s="auto")
        seq = doc["sequential"]
        self.assertEqual((seq["stop"]["action"], seq["stop"]["n"]), ("refuse", 500))
        self.assertIn("route S", seq["stop"]["reason"])
        self.assertEqual((doc["decision_text"], doc["decision"]["checks"]["sequential"]), ("rejected", False))
        self.assertTrue(doc["decision"]["rule"].endswith("+D51-funnel"))
        self.assertFalse((self.agents / "current.json").exists())

    def test_s1_not_holding_leaves_v3_exactly_as_it_was_and_records_the_reading(self):
        off = self.promote()
        doc = self.promote(route_s="auto", stage=self.stage(points=(0.6, 0.2)))
        self.assertEqual(self.stage_calls[-1][0], "c1")
        self.assertEqual((doc["decision_text"], doc["sequential"]["stop"]["rule"]), (off["decision_text"], off["sequential"]["stop"]["rule"]))
        self.assertEqual(doc["sequential"]["futility"]["enabled"], True)
        self.assertNotIn("route", doc["sequential"])
        self.assertEqual(doc["route_s_mode"]["applies"], False)
        self.assertEqual(doc["s1"]["mean"], 0.6)
        self.assertEqual((doc["keep_v2"]["route_s"]["applies"], doc["keep_v2"]["route_s"]["decided_by"]), (False, "v3"))
        self.assertNotIn("base_reuse", doc)
        row = gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]
        self.assertIn("S1 +0.60 ±0.40 per 100 arrivals", row["guard"])
        self.assertIn("does not hold, v3 decides", row["guard"])

    def test_a_pair_s1_cannot_be_played_for_is_judged_by_v3_with_the_reason_recorded(self):
        doc = self.promote(route_s="auto", stage=self.stage(raises=gen_loop.S1Unavailable("the champion has no arms to build a ruler from")))
        self.assertIn("S1 could not be played (the champion has no arms to build a ruler from): v3 decides", doc["route_s_mode"]["why"])
        self.assertEqual((doc["route_s_mode"]["applies"], doc["sequential"]["stop"]["rule"]), (False, "futility@800"))
        self.assertNotIn("s1", doc)
        self.assertNotIn("route_s", doc["keep_v2"])

    def test_a_stopped_s1_stops_the_judgement_before_an_arm_is_played(self):
        with self.assertRaises(gen_loop.S1Blocked):
            self.promote(route_s="auto", stage=self.stage(raises=gen_loop.S1Blocked("stopped")))
        self.assertEqual(self.arm_calls, [])
        self.assertFalse((self.cand / "promotion.json").exists())
        self.assertFalse((self.cand / "arms").exists())  # no arm of the candidate's was played

    def test_the_flag_is_the_keep_rule_v3s(self):
        arms, _ = self.stub_arms(self.c, self.b)
        with mock.patch.object(gen_loop, "run_arm", arms), self.assertRaisesRegex(ValueError, "keep rule v3"):
            gen_loop.promote(self.cand, self.champ, self.f1, agent_root=self.agents, boot=500, keep_rule="v2", sequential=True, route_s="auto")
        with self.assertRaisesRegex(ValueError, "unknown --route-s"):
            gen_loop.promote(self.cand, self.champ, self.f1, agent_root=self.agents, boot=500, keep_rule="v3", route_s="yes")

    def test_the_champions_stored_arm_is_checked_before_it_is_reused(self):
        self.write_arm(self.champ, "Etest-r1", self.b)  # the champion's arm of this block, stored; the candidate never played it
        calls = []

        def arms(seeds, arm_dir, agent, config, procs, ascension=10, resume=False, **kw):
            calls.append((agent.generation, Path(arm_dir).parent.parent.name, len(seeds), kw.get("start_ix", 0)))
            want = set(seeds)
            return [r for r in (self.c if agent.generation == 4 else self.b) if r["seed"] in want]

        with mock.patch.object(gen_loop, "run_arm", arms), mock.patch.object(gen_loop, "s1_stage", self.stage()):
            doc = gen_loop.promote(self.cand, self.champ, self.f1, agent_root=self.agents, boot=500, keep_rule="v3", sequential=True, route_s="auto")
        self.assertEqual(calls[0], (3, "c1", 20, 0))  # the champion replays 20 seeds of the block into the candidate's scratch arm, before any arm of the funnel
        self.assertEqual(doc["base_reuse"]["action"], "reused")
        self.assertEqual((doc["base_reuse"]["checked"], doc["base_reuse"]["same"]), (20, 20))
        self.assertEqual(doc["decision_text"], "promoted")

        # a stored arm that no longer replays is set aside and the champion plays the block afresh
        self.setUp()
        self.write_arm(self.champ, "Etest-r1", self.b)

        def changed(seeds, arm_dir, agent, config, procs, ascension=10, resume=False, **kw):
            rows = [dict(r) for r in (self.c if agent.generation == 4 else self.b) if r["seed"] in set(seeds)]
            if "arms-check" in Path(arm_dir).parts:
                rows[0]["floor"] += 1
            return rows

        with mock.patch.object(gen_loop, "run_arm", changed), mock.patch.object(gen_loop, "s1_stage", self.stage()):
            doc = gen_loop.promote(self.cand, self.champ, self.f1, agent_root=self.agents, boot=500, keep_rule="v3", sequential=True, route_s="auto")
        self.assertEqual((doc["base_reuse"]["action"], doc["base_reuse"]["same"]), ("played afresh", 19))
        self.assertFalse((self.champ / "arms" / "Etest-r1-a3-v1").exists())
        self.assertEqual(len(list((self.champ / "arms-stale").iterdir())), 1)


class RejudgeRouteSTests(World):
    """`rejudge(route_s="auto", s1_file=)`: a reading from a file is checked against the ruler's hash and the two fight networks; a block the funnel stopped
    is completed (the funnel's own resume) before S2's final bound is asked."""

    def setUp(self):
        super().setUp()
        self.seeds, self.c, self.b = block(FLATISH, floor_mean=0.0, prefix="K", seed=7)
        self.f1 = self.seed_file("Etest-r1", self.seeds)
        self.ruler = self.ruler_root / "gen3-candidate"
        self.meta = make_ruler(self.ruler)
        self.s1_path = self.root / "s1-results.json"

    def write_s1(self, **kw) -> Path:
        doc = result(1.8, 0.4, ruler=kw.pop("ruler", self.meta["ruler_hash"]), cand=kw.pop("cand", CAND_SHA), champ=kw.pop("champ", CHAMP_SHA), **kw)
        self.s1_path.write_text(json.dumps(doc))
        return self.s1_path

    def full_block(self) -> None:
        self.write_arm(self.cand, "Etest-r1", self.c)
        self.write_arm(self.champ, "Etest-r1", self.b)

    def stopped_block(self, played: int = 1100) -> None:
        """The block as the funnel left it at futility@`played`: both arms hold the first `played` pairs, promotion.json says it stopped."""
        self.write_arm(self.cand, "Etest-r1", self.c[:played])
        self.write_arm(self.champ, "Etest-r1", self.b[:played])
        gen_loop.write_json(self.cand / "promotion.json", {
            "seed_block": "Etest-r1", "decision_text": "rejected", "written": "2026-11-21T21:14:00Z",
            "sequential": {"rule": "D51-funnel", "block_seeds": 2000, "played": played, "smoke": 200, "batch": 300, "stop": {"action": "futility", "n": played}, "looks": []}})

    def rejudge(self, **kw):
        kw.setdefault("route_s", "auto")
        kw.setdefault("ruler_root", self.ruler_root)
        return gen_loop.rejudge(self.cand, self.champ, self.f1, agent_root=self.agents, note="D68", boot=500, keep_rule="v3", **kw)

    def test_a_reading_that_matches_is_applied_and_the_decision_is_s(self):
        self.full_block()
        doc = self.rejudge(s1_file=self.write_s1())
        k = doc["keep_v2"]
        self.assertEqual((doc["decision_text"], k["route"], k["route_s"]["decided_by"], k["route_s"]["s1"]["applied"], k["route_s"]["s1"]["holds"]), ("promoted", "S", "S", True, True))
        self.assertEqual(doc["s1"]["mean"], 1.8)
        self.assertEqual((doc["route_s_mode"]["mode"], doc["route_s_mode"]["applies"], doc["route_s_mode"]["source"]), ("auto", True, "--s1 s1-results.json"))
        self.assertNotIn(str(self.root), json.dumps(doc["route_s_mode"]))  # the file's name, never its path
        row = gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]
        self.assertEqual((row["decision"], row["route"], row["seeds"]), ("promoted", "S", "2000"))
        # the test the file played is entered in the ruler's ledger, so the next test takes new draws
        self.assertEqual([(t["test"], t["candidate"]) for t in gen_loop.s1_ledger(self.ruler)["tests"]], [(0, CAND_SHA)])
        self.assertEqual(gen_loop.s1_used_tests(self.ruler, gen_loop.s1_ledger(self.ruler)), {0})

    def test_a_reading_that_names_another_ruler_champion_or_candidate_is_refused_and_v3_decides(self):
        self.full_block()
        for what, kw in (("ruler_hash", {"ruler": "d" * 64}), ("candidate", {"cand": "d" * 64}), ("champion", {"champ": "d" * 64})):
            with self.subTest(what):
                doc = self.rejudge(s1_file=self.write_s1(**kw), out=self.root / f"dry-{what}")
                rs = doc["keep_v2"]["route_s"]
                self.assertEqual((rs["s1"]["applied"], rs["applies"], rs["decided_by"]), (False, False, "v3"))
                self.assertIn(f"{what} does not match", rs["s1"]["reasons"][0])
                self.assertEqual(doc["route_s_mode"]["applies"], False)
                self.assertEqual(doc["s1"]["match"]["verified"], False)
                self.assertNotEqual(doc["keep_v2"]["route"], "S")  # v3's own verdict on the block (flat: refused)
                self.assertEqual(doc["decision_text"], "rejected")
        self.assertEqual(gen_loop.s1_ledger(self.ruler)["tests"], [])  # a reading that does not match enters no ledger

    def test_a_ruler_that_cannot_be_read_leaves_the_reading_unchecked_and_unapplied(self):
        self.full_block()
        import shutil

        shutil.rmtree(self.ruler)
        doc = self.rejudge(s1_file=self.write_s1(), out=self.root / "dry")
        self.assertFalse(doc["keep_v2"]["route_s"]["s1"]["applied"])
        self.assertIn("ruler_hash: the pair's is not known", doc["keep_v2"]["route_s"]["s1"]["reasons"][0])

    def test_a_dry_run_reads_and_writes_no_ledger(self):
        self.full_block()
        doc = self.rejudge(s1_file=self.write_s1(), out=self.root / "dry")
        self.assertEqual(doc["decision_text"], "promoted")
        self.assertFalse((self.ruler / "s1-ledger.json").exists())
        self.assertFalse((self.cand / "promotion.json").exists())

    def test_the_candidates_own_reading_is_used_when_no_file_is_named_and_none_means_v3(self):
        self.full_block()
        none = self.rejudge(out=self.root / "dry0")
        self.assertEqual((none["route_s_mode"]["applies"], none["keep_v2"]["route"]), (False, ""))
        self.assertIn("no S1 reading", none["route_s_mode"]["why"])
        self.assertNotIn("route_s", none["keep_v2"])
        gen_loop.write_json(self.cand / "s1.json", k2.s1_reading(json.loads(self.write_s1().read_text()), None))  # a stored reading is checked again, here
        doc = self.rejudge(out=self.root / "dry1")
        self.assertEqual((doc["route_s_mode"]["applies"], doc["keep_v2"]["route"], doc["decision_text"]), (True, "S", "promoted"))
        self.assertEqual(doc["route_s_mode"]["source"], "the candidate's s1.json")

    def test_the_flag_without_the_rule_or_the_file_without_the_flag_is_an_error(self):
        self.full_block()
        with self.assertRaisesRegex(ValueError, "keep rule v3"):
            gen_loop.rejudge(self.cand, self.champ, self.f1, agent_root=self.agents, note="x", boot=500, keep_rule="v2", route_s="auto", out=self.root / "d")
        with self.assertRaisesRegex(ValueError, "name --route-s auto"):
            self.rejudge(route_s="off", s1_file=self.write_s1(), out=self.root / "d")
        with self.assertRaisesRegex(ValueError, "name --route-s auto"):
            self.rejudge(route_s=None, complete=True, out=self.root / "d")

    def test_a_block_the_funnel_stopped_needs_the_completion_and_a_dry_run_cannot_play(self):
        self.stopped_block()
        s1 = self.write_s1()
        with self.assertRaisesRegex(ValueError, "add --complete-block to play the other 900 pairs"):
            self.rejudge(s1_file=s1)
        with self.assertRaisesRegex(ValueError, "never a dry run"):
            self.rejudge(s1_file=s1, complete=True, out=self.root / "dry")
        # with S1 not holding the stopped block is judged as it always was: on the pairs played, refused as the funnel stopped it
        doc = self.rejudge(s1_file=self.write_s1(cand="d" * 64), out=self.root / "dry")
        self.assertEqual((doc["decision_text"], doc["n_seeds"]), ("rejected", 1100))
        self.assertIn("rejudged", doc)

    def test_the_stopped_block_is_completed_from_its_stored_rows_and_judged_by_s2(self):
        self.stopped_block()
        arms, calls = self.stub_arms(self.c, self.b)
        with mock.patch.object(gen_loop, "run_arm", arms):
            doc = self.rejudge(s1_file=self.write_s1(), complete=True, procs=8, ascension=10)
        self.assertEqual((doc["decision_text"], doc["n_seeds"], doc["paired_seeds"]), ("promoted", 2000, 2000))
        seq = doc["sequential"]
        self.assertEqual((seq["stop"]["action"], seq["played"], seq["route"], seq["futility"]["enabled"]), ("final", 2000, "S", False))
        self.assertEqual((seq["smoke"], seq["batch"]), (200, 300))  # the earlier record's schedule
        self.assertTrue(all(x["action"] in ("continue", "final") for x in seq["looks"]))
        self.assertTrue(all(resume for _g, _n, resume, _kw in calls))  # the funnel's own resume: a seed with a row is not played again
        self.assertEqual([n for g, n, _r, _kw in calls if g == 4], [200, 500, 800, 1100, 1400, 1700, 2000])  # the looks are walked again over the stored rows
        self.assertEqual(doc["rejudged"]["completed"], {"from_seeds": 1100, "to_seeds": 2000, "block": 2000, "earlier_stop": "futility",
                                                        "text": "completed from 1100 to 2000 of 2000 seeds under route S (the stored rows kept)"})
        self.assertEqual(doc["keep_v2"]["route"], "S")
        self.assertEqual(doc["keep_v2"]["route_s"]["s2"]["pairs"], 2000)
        self.assertEqual(doc["base_reuse"]["action"], "none")  # both arms played the first 1,100 pairs together: nothing of the champion's is reused unchecked
        # the earlier record is kept beside the new one, and the ledger has a second row
        kept = list(self.cand.glob("promotion-rejected-*.json"))
        self.assertEqual(len(kept), 1)
        self.assertEqual(json.loads(kept[0].read_text())["sequential"]["stop"]["action"], "futility")
        row = gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]
        self.assertEqual((row["decision"], row["route"], row["seeds"]), ("promoted", "S", "2000"))
        self.assertEqual(json.loads((self.agents / "current.json").read_text())["generation"], 4)

    def test_a_harm_shown_while_the_block_is_completed_refuses_it_at_that_look(self):
        _, c, b = block(HARM, floor_mean=0.0, prefix="K", seed=7)
        self.c, self.b = c, b
        self.stopped_block(1100)
        arms, _ = self.stub_arms(c, b)
        with mock.patch.object(gen_loop, "run_arm", arms):
            doc = self.rejudge(s1_file=self.write_s1(), complete=True)
        self.assertEqual((doc["decision_text"], doc["n_seeds"], doc["sequential"]["stop"]["action"]), ("rejected", 500, "refuse"))
        self.assertEqual(doc["rejudged"]["completed"]["to_seeds"], 500)
        self.assertEqual(doc["decision"]["checks"]["sequential"], False)

    def test_the_command_line_names_the_flags_and_prints_the_route(self):
        self.full_block()
        args = SimpleNamespace(cand=str(self.cand), base=str(self.champ), seeds_file=str(self.f1), agent_root=str(self.agents), config="a3-v1", note="D68",
                               lab=None, review_guard=None, keep_rule="v3", out=str(self.root / "dry"), pool_block=[], min_gain=1.0, route_s="auto",
                               s1=str(self.write_s1()), complete_block=False, procs=16, ascension=10, lib=None, prior_server="off")
        buf = io.StringIO()
        with mock.patch.object(gen0, "_setup"), mock.patch.object(gen_loop, "s1_ruler_dir", lambda base, root=None: self.ruler), contextlib.redirect_stdout(buf):
            self.assertEqual(gen0.rejudge_cmd(args), 0)
        printed = json.loads(buf.getvalue())
        self.assertEqual((printed["decision_text"], printed["keep_v2"]["route"]), ("promoted", "S"))
        self.assertEqual((printed["route_s"]["decided_by"], printed["route_s"]["s1"]["mean"], printed["route_s"]["s1"]["holds"]), ("S", 1.8, True))
        self.assertTrue(printed["route_s"]["s2"]["final_ok"])
        p = gen0.build_parser().parse_args(["rejudge", "--cand", "c", "--seeds-file", "s.json", "--agent-root", "a", "--note", "n", "--keep-rule", "v3",
                                            "--route-s", "auto", "--s1", "results.json", "--complete-block", "--procs", "12"])
        self.assertEqual((p.route_s, p.s1, p.complete_block, p.procs, p.prior_server), ("auto", "results.json", True, 12, "off"))
        q = gen0.build_parser().parse_args(["rejudge", "--cand", "c", "--seeds-file", "s.json", "--agent-root", "a", "--note", "n"])
        self.assertEqual((q.route_s, q.s1, q.complete_block), ("off", None, False))  # the default is off


class CommandLineTests(World):
    def test_the_flag_is_off_by_default_on_train_promote_and_rejudge(self):
        parser = gen0.build_parser()
        need = ["--shards-root", "r", "--agent-root", "a"]
        self.assertEqual(parser.parse_args(["train", "--gen", "1", *need]).route_s, "off")
        self.assertEqual(parser.parse_args(["promote", "--cand", "c", "--agent-root", "a"]).route_s, "off")
        self.assertEqual(parser.parse_args(["train", "--gen", "1", *need, "--route-s", "auto"]).route_s, "auto")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["promote", "--cand", "c", "--agent-root", "a", "--route-s", "on"])

    def test_the_child_command_carries_the_flag_only_when_it_is_auto(self):
        calls = []

        def fake_run(cmd, check=False):
            calls.append(cmd)
            return SimpleNamespace(returncode=0)

        with mock.patch.object(gen_loop.subprocess, "run", fake_run):
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None, keep_rule="v3")
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None, keep_rule="v3", route_s="off")
            gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None, keep_rule="v3", route_s="auto")
        self.assertNotIn("--route-s", calls[0])
        self.assertNotIn("--route-s", calls[1])
        self.assertEqual(calls[2][calls[2].index("--route-s") + 1], "auto")
        self.assertEqual(calls[0], calls[1])  # the default command is exactly as it was

    def test_promote_cmd_passes_the_flag_and_a_stopped_s1_exits_non_zero(self):
        args = SimpleNamespace(cand=str(self.cand), base=str(self.champ), seeds_file="s.json", agent_root=str(self.agents), config="a3-v1", ascension=10,
                               procs=4, lib=None, lab=None, lab_runs=None, review_guard=False, sequential=True, smoke_seeds=200, batch_seeds=300,
                               keep_rule="v3", keep_rule_why=None, min_gain=1.0, no_futility=False, futility_min_seeds=None, no_screen=False,
                               pool_block=[], route_s="auto", prior_server="off", gpu_python=None, server_procs=None, prior_server_args=None,
                               latency_p99_ms=20.0)
        seen = {}

        def fake_promote(*a, **kw):
            seen.update(kw)
            return {"decision_text": "rejected", "floor": {}, "boss_kill": {}, "wins": {}}

        with mock.patch.object(gen0, "_setup"), mock.patch.object(gen_loop, "promote", fake_promote), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen0.promote_cmd(args), 0)
        self.assertEqual(seen["route_s"], "auto")
        err = io.StringIO()

        def blocked(*a, **kw):
            raise gen_loop.S1Blocked("the arrival exam was stopped")

        with mock.patch.object(gen0, "_setup"), mock.patch.object(gen_loop, "promote", blocked), contextlib.redirect_stderr(err):
            self.assertEqual(gen0.promote_cmd(args), 3)
        self.assertIn("S1 stopped", err.getvalue())
        self.assertIn("run the judgement again to resume", err.getvalue())

    def test_a_default_promote_cmd_names_no_route(self):
        args = SimpleNamespace(cand=str(self.cand), base=None, seeds_file="s.json", agent_root=str(self.agents), config="a3-v1", ascension=10, procs=4, lib=None,
                               lab=None, lab_runs=None, review_guard=False, sequential=False, smoke_seeds=200, batch_seeds=300, keep_rule=None, min_gain=1.0,
                               prior_server="off", gpu_python=None, server_procs=None, prior_server_args=None, latency_p99_ms=20.0)
        seen = {}

        def fake_promote(*a, **kw):
            seen.update(kw)
            return {"decision_text": "rejected", "floor": {}, "boss_kill": {}, "wins": {}}

        with mock.patch.object(gen0, "_setup"), mock.patch.object(gen_loop, "promote", fake_promote), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen0.promote_cmd(args), 0)
        self.assertIsNone(seen["route_s"])
        self.assertEqual(gen_loop.route_s_of(seen["route_s"]), "off")



class TrainFlagTests(unittest.TestCase):
    """`gen0.py train --promote` on a built candidate, the judging child stubbed: the flag reaches it only when it is auto, and only under v3."""

    setUp, tearDown, build = tg12.PlacementTests.setUp, tg12.PlacementTests.tearDown, tg12.PlacementTests.build

    def train(self, *flags: str):
        self.assertEqual(0, self.build())
        seeds = self.root / "seeds"
        seeds.mkdir()
        calls = []

        def child(*args, **kw):
            calls.append((args, kw))
            return 0

        argv = ["train", "--marker-dir", str(self.marker_dir), "--shards-root", str(self.root), "--agent-root", str(self.agents), "--promote",
                "--emu-python", "python", "--seeds-file", str(seeds / "E{k}-{date}.json"), "--sequential", *flags]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(gen_loop, "train_generation"), mock.patch.object(gen_loop, "run_promote_subprocess", side_effect=child), \
                mock.patch.object(gen_loop, "COMMITTED_SEEDS", seeds), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = gen0.main(argv)
        return rc, calls, err.getvalue()

    def test_auto_reaches_the_child_under_v3_and_the_default_names_nothing(self):
        rc, calls, _ = self.train("--keep-rule", "v3", "--route-s", "auto")
        self.assertEqual((rc, len(calls)), (0, 1))
        self.assertEqual(calls[0][1]["route_s"], "auto")
        self.assertEqual(calls[0][1]["keep_rule"], "v3")
        self.tearDown()
        self.setUp()
        rc, calls, _ = self.train("--keep-rule", "v3")
        self.assertEqual((rc, len(calls)), (0, 1))
        self.assertNotIn("route_s", calls[0][1])  # the child's call is exactly as it was
        self.tearDown()
        self.setUp()
        rc, calls, _ = self.train("--keep-rule", "v3", "--route-s", "off")
        self.assertNotIn("route_s", calls[0][1])

    def test_auto_under_another_rule_is_refused_before_anything_is_trained_or_judged(self):
        for rule in ("v2", "v1"):
            self.tearDown()
            self.setUp()
            rc, calls, err = self.train("--keep-rule", rule, "--route-s", "auto")
            self.assertEqual((rc, calls), (2, []), rule)
            self.assertIn("--route-s auto is the keep rule v3's", err)
            self.assertIn("the marker stays", err)

    def test_an_extension_requeue_keeps_the_flag(self):
        self.assertEqual(0, self.build())
        cand = self.agents / "gen7" / "candidate"
        seeds = self.root / "seeds"
        seeds.mkdir()
        calls = []

        def child(*args, **kw):
            calls.append((args, kw))
            if len(calls) == 1:
                (cand / "promotion.json").write_text(json.dumps({"seed_block": "E7-day", "decision": {"promote": False, "extend": True, "route": "X"},
                                                                 "decision_text": "extend"}))
            return 0

        argv = ["train", "--marker-dir", str(self.marker_dir), "--shards-root", str(self.root), "--agent-root", str(self.agents), "--promote",
                "--emu-python", "python", "--seeds-file", str(seeds / "E{k}-{date}.json"), "--sequential", "--keep-rule", "v3", "--route-s", "auto"]
        with mock.patch.object(gen_loop, "train_generation"), mock.patch.object(gen_loop, "run_promote_subprocess", side_effect=child), \
                mock.patch.object(gen_loop, "COMMITTED_SEEDS", seeds), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen0.main(argv), 0)
        self.assertEqual([c[1]["route_s"] for c in calls], ["auto", "auto"])  # S1's reading stands for the second block: the stage reads s1.json


if __name__ == "__main__":
    unittest.main()
