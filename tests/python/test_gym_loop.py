"""The always-on fight-gym loop (rl/gym_loop.py, rl/gym_loop_train.py; D40, gym build item L1).

The pinned facts: the promotion rule decides as written (stage 1's futility stop on a pair of seed indices the incumbent was not
picked on, rotating one pair per promotion with the incumbent's games on it taken from its own promotion gate; stage 2's exact
one-sided McNemar test and the L0 guard; a capped game is lost); the daemon's guards read the lab's locks as the lab's jobs do (the night, main.lock only before a
step, PAUSE, HALT, another live QUIET.lock line; its own lines kept with its PID; GPU_OK taken only when absent or ours); the
buffer's compaction, retention window and disk cap; a generation fight recorded through `gen_game_steps` is the gym's game move for
move (recording changes nothing) and leaves one training sample per decision; the trainer spends its freshness budget and then idles,
writes a checkpoint the players load as an agent, and resumes from its state; the gate plays the incumbent once and keeps its games.
Real games are short arena fights with the fixture prior, on core2 in spawned processes as the gym plays them.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fight_gym as fg  # noqa: E402
import gym_loop as gl  # noqa: E402

F18 = HERE / "fixtures" / "f18_step_hang.json"


def _row(start, j, won, l0=0.5, capped=False):
    return {"start": start, "j": j, "won": won, "L0": l0 if won else 0.0, "capped": capped}


class PureTests(unittest.TestCase):
    def test_mcnemar_is_the_exact_one_sided_sign_test(self):
        self.assertEqual(gl.mcnemar_one_sided(0, 0), 1.0)
        self.assertAlmostEqual(gl.mcnemar_one_sided(5, 0), 1 / 32)
        self.assertAlmostEqual(gl.mcnemar_one_sided(3, 3), 0.65625)
        self.assertLess(gl.mcnemar_one_sided(60, 35), 0.05)
        self.assertGreater(gl.mcnemar_one_sided(55, 40), 0.05)

    def test_the_rule_stops_at_stage_one_and_promotes_only_on_a_paired_win(self):
        rule = {**gl.PROMOTION_RULE, "starts": 10, "stage1_seeds": 1, "seeds": 4}
        starts = [f"S{i}" for i in range(10)]
        inc = [_row(s, j, i % 2 == 0) for i, s in enumerate(starts) for j in range(4)]
        same = [dict(r) for r in inc]
        self.assertEqual(gl.decide(same[:5], inc, 1, rule)["verdict"], "incomplete")
        v = gl.decide(same, inc, 1, rule)
        self.assertEqual(v["verdict"], "reject")  # +0.0 points is under the +1.0 needed to go on
        better = [_row(r["start"], r["j"], True) for r in inc]  # wins every game: 20 candidate-only wins, none against
        self.assertEqual(gl.decide(better, inc, 1, rule)["verdict"], "continue")
        v = gl.decide(better, inc, 2, rule)
        self.assertEqual((v["verdict"], v["cand_only"], v["inc_only"], v["pairs"]), ("promote", 20, 0, 40))
        lower_l0 = [_row(r["start"], r["j"], True, l0=0.0001) for r in inc]  # wins more, but leaves less HP where both win
        hi = [_row(r["start"], r["j"], r["won"], l0=1.0) for r in inc]
        self.assertEqual(gl.decide(lower_l0, hi, 2, rule)["verdict"], "promote" if gl.paired(lower_l0, hi, 4)["l0_diff"] >= 0
                         else "reject")
        capped = [_row(r["start"], r["j"], True, capped=True) for r in inc]  # a capped game counts as lost
        self.assertEqual(gl.paired(capped, inc, 4)["cand_wins"], 0)

    def test_runs_give_their_fights_by_kind_and_the_split_is_by_run_seed(self):
        run = {"nodes": "M?ME" + "RMMMMMMMMMM" + "B" + "MMEMB", "floor": 20}
        fl = gl.fight_floors(run)
        self.assertEqual([f["cat"] for f in fl][:3], ["act1-hallway", "act1-hallway", "act1-elite"])
        self.assertEqual([f["floor"] for f in fl if f["cat"] == "act1-boss"], [17])
        self.assertTrue(all(f["floor"] <= 20 for f in fl))
        self.assertEqual(sum(f["act"] == 2 for f in fl), 3)  # floors 18..20
        fpr = gl.fights_per_run([run, {"nodes": "MM", "floor": 3}])
        self.assertEqual(fpr["act1-boss"], 0.5)
        seeds = [f"SEED{i:05d}" for i in range(4000)]
        share = sum(gl.is_val_seed(s) for s in seeds) / len(seeds)
        self.assertTrue(0.08 < share < 0.12, share)
        self.assertEqual([gl.is_val_seed(s) for s in seeds[:50]], [gl.is_val_seed(s) for s in seeds[:50]])

    def test_a_launcher_and_the_interpreter_it_starts_count_as_one_of_ours(self):
        parents = {10: 1, 11: 10, 12: 11, 20: 1, 21: 20, 30: 30}
        self.assertEqual(gl.descendants(parents, 10), {10, 11, 12})
        self.assertEqual(gl.descendants(parents, None), set())
        self.assertEqual(gl.descendants(parents, 30), {30})
        here = gl.process_parents()
        if here:
            self.assertIn(os.getpid(), gl.descendants(here, os.getppid()))

    def test_the_loop_reads_the_expected_solve_only(self):
        # The owner's solver rule: labels and grades never read the hindsight export (gym_lockstep.Grader, which the loop uses for
        # its exact grades, is under test_solver_hindsight_pins; this pins the loop's own sources with that test's names).
        import test_solver_hindsight_pins as pins  # noqa: PLC0415

        for f in ("rl/gym_loop.py", "rl/gym_loop_train.py"):
            self.assertEqual(pins._hits((ROOT / f).read_text(encoding="utf-8")), [], f)  # noqa: SLF001

    def test_logs_never_keep_a_home_folder(self):
        home = "C:" + "\\" + "Us" + "ers" + "\\" + "someone"  # built here, so the source holds no home folder
        self.assertEqual(gl.clean("open " + home + "\\AppData\\x.py"), "open ~\\AppData\\x.py")
        self.assertEqual(gl.clean("D:/spireline/referee/gym-loop"), "D:/spireline/referee/gym-loop")


class GatePairTests(unittest.TestCase):
    """Stage 1's rotating pair (PROMOTION_RULE v2, D43): no games, the pure parts and the gate with its games faked."""

    RULE = {**gl.PROMOTION_RULE, "starts": 4, "stage1_seeds": 2, "seeds": 6, "futility_points": 1.0}  # three pairs of j

    def _rows(self, who, js, win, starts=("S0", "S1", "S2", "S3")):
        return [{**_row(s, j, win(i, j)), "arm": who, "player": who} for i, s in enumerate(starts) for j in js]

    def test_the_rule_is_v2_and_the_pairs_rotate_one_per_promotion_and_wrap(self):
        rule = gl.PROMOTION_RULE
        self.assertEqual(rule["name"], "gym-loop-promote-v2")
        self.assertEqual([gl.stage1_js(rule, p) for p in range(6)],
                         [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (0, 1)])
        self.assertEqual(gl.stage1_pairs(rule), 5)
        walk = [0]
        for _ in range(6):
            walk.append(gl.next_gate_pair(rule, walk[-1]))
        self.assertEqual(walk, [0, 1, 2, 3, 4, 0, 1])
        self.assertEqual(gl.stage1_js(self.RULE, 2), (4, 5))  # a rule with other widths rotates the same way
        self.assertEqual(gl.next_gate_pair(self.RULE, 2), 0)

    def test_stage_one_counts_only_its_pair_and_an_incumbent_lucky_on_its_own_pair_no_longer_blocks_an_equal_candidate(self):
        # the incumbent was picked on pair 0 and won every game there; elsewhere it wins half. The candidate is as good as the
        # incumbent really is (half), plus two more wins than it on pair 1.
        inc = self._rows("v2", range(6), lambda i, j: True if j < 2 else i % 2 == 0)
        cand = self._rows("c", range(6), lambda i, j: i % 2 == 0 or (j in (2, 3) and i == 1))
        old = gl.decide(cand, inc, 1, self.RULE, pair=0)  # the v1 behaviour: the lucky pair
        self.assertEqual((old["verdict"], old["pairs"], old["inc_wins"], old["cand_wins"]), ("reject", 8, 8, 4))
        new = gl.decide(cand, inc, 1, self.RULE, pair=1)
        self.assertEqual((new["verdict"], new["pairs"], new["inc_wins"], new["cand_wins"]), ("continue", 8, 4, 6))
        same = gl.decide(inc, inc, 1, self.RULE, pair=1)
        self.assertEqual((same["verdict"], same["win_diff"]), ("reject", 0.0))  # an equal candidate is still stopped at +0.0
        self.assertEqual(gl.decide(cand[:5], inc, 1, self.RULE, pair=1)["verdict"], "incomplete")
        self.assertEqual(gl.paired(cand, inc, 6, (4, 5))["pairs"], 8)
        self.assertEqual(gl.paired(cand, inc, 6)["pairs"], 24)  # no pair given: every j below `seeds`, stage 2's way
        self.assertEqual(gl.decide(cand, inc, 1, self.RULE)["verdict"], old["verdict"])  # the default pair is pair 0

    def test_the_incumbents_reference_comes_from_its_promotion_gate_once_and_only_for_the_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = gl.Loop(Path(tmp) / "loop")
            loop.state["promotions"].append({"version": "v2", "checkpoint": "c0019", "over": "v1", "t": "x"})
            src = loop.gates / "c0019-cfg-eng.jsonl"
            inc_file = loop.gates / "inc-v2-cfg-eng.jsonl"
            rows = self._rows("c0019", range(6), lambda i, j: (i + j) % 2 == 0)
            src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            # the incumbent's own replays of pair 0 are there already (the last line cut by a stop: no newline at its end)
            mine = self._rows("v2", (0, 1), lambda i, j: True)
            inc_file.write_text("".join(json.dumps(r) + "\n" for r in mine) + '{"start": "S0", "j"', encoding="utf-8")
            self.assertEqual(gl.reuse_incumbent_games(loop, "v2", inc_file, (0, 1), "cfg", "eng"), 0)  # pair 0 is there
            self.assertEqual(gl.reuse_incumbent_games(loop, "v2", inc_file, (2, 3), "cfg", "eng"), 8)
            got = fg.read_jsonl(inc_file)
            self.assertEqual(len(got), 16)  # the cut line is dropped, nothing swallowed
            taken = [r for r in got if r.get("reused_from")]
            self.assertEqual({(r["j"], r["arm"], r["player"], r["reused_from"]) for r in taken},
                             {(2, "v2", "v2", "c0019"), (3, "v2", "v2", "c0019")})
            want = {(r["start"], r["j"]): r["won"] for r in rows if r["j"] in (2, 3)}
            self.assertEqual({(r["start"], r["j"]): r["won"] for r in taken}, want)  # the promotion gate's games, as played
            self.assertEqual({r["j"] for r in got}, {0, 1, 2, 3})  # nothing of the other pairs
            self.assertEqual(gl.reuse_incumbent_games(loop, "v2", inc_file, (2, 3), "cfg", "eng"), 0)  # once
            self.assertEqual(len(fg.read_jsonl(inc_file)), 16)
            self.assertEqual(gl.reuse_incumbent_games(loop, "v2", loop.gates / "inc-v2-cfg-new.jsonl", (2, 3), "cfg", "new"), 0)
            self.assertEqual(gl.reuse_incumbent_games(loop, "v0", loop.gates / "inc-v0-cfg-eng.jsonl", (2, 3), "cfg", "eng"), 0)
            fresh = loop.gates / "inc-v2-other-eng.jsonl"  # another config: no promotion gate file to take from
            self.assertEqual(gl.reuse_incumbent_games(loop, "v2", fresh, (2, 3), "other", "eng"), 0)
            self.assertFalse(fresh.exists())

    def test_a_promotion_moves_stage_one_to_the_next_pair_and_the_state_keeps_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "loop"
            loop = gl.Loop(root)
            self.assertEqual((loop.gate_pair(), loop.state["gate_pair"]), (0, 0))  # a new loop starts on j = 0..1
            ck = Path(tmp) / "ckpt"
            ck.mkdir()
            (ck / "model.json").write_text("{}", encoding="utf-8")
            (loop.gates / "cand.jsonl").write_text(json.dumps(_row("S0", 0, True)) + "\n", encoding="utf-8")
            for n, (before, after) in enumerate([(0, 1), (1, 2), (4, 0)], start=1):
                loop.state["gate_pair"] = before
                gl.promote(loop, f"c{n:04d}", ck, {"cand_file": "cand.jsonl", "cand_wins": 1, "inc_wins": 0, "pairs": 1}, "off")
                self.assertEqual(loop.state["incumbent"], f"v{n}")
                self.assertEqual(gl.Loop(root).gate_pair(), after)  # on disk
            self.assertIn("stage 1 now plays j [0, 1]", (root / "loop.log").read_text(encoding="utf-8"))

    def test_a_state_from_before_the_rotation_starts_on_the_second_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "loop"
            root.mkdir()
            old = {"segment": 3, "incumbent": "v2", "promotions": [{"version": "v1"}, {"version": "v2"}],
                   "versions": {"v0": {"dir": "x", "n": 0}, "v2": {"dir": "y", "n": 2}}}
            (root / "state.json").write_text(json.dumps(old), encoding="utf-8")
            loop = gl.Loop(root)
            self.assertEqual(gl.stage1_js(gl.PROMOTION_RULE, loop.gate_pair()), (2, 3))  # v2 was picked on j = 0..1
            self.assertEqual(loop.state["segment"], 3)

    def test_the_gate_plays_the_candidate_on_the_pair_and_the_incumbent_not_at_all_when_its_games_are_kept(self):
        starts = ("S0", "S1", "S2", "S3")
        played: list[tuple[str, int]] = []

        def fake_play(loop, tasks, target, player_dir, procs, k, prior, what, **kw):
            with open(target, "a", encoding="utf-8") as fh:
                for _kind, sid, j, _cat, info in tasks:
                    played.append((info["label"], j))
                    won = starts.index(sid) % 2 == 0  # every player wins the even starts: no difference
                    fh.write(json.dumps({**_row(sid, j, won), "arm": info["label"]}) + "\n")
            return None

        with tempfile.TemporaryDirectory() as tmp:
            loop = gl.Loop(Path(tmp) / "loop")
            d = Path(tmp) / "agent"
            d.mkdir()
            loop.state["versions"]["v2"] = {"dir": str(d), "n": 2}
            loop.state["incumbent"] = "v2"
            loop.state["promotions"].append({"version": "v2", "checkpoint": "c0019", "over": "v1", "t": "x"})
            loop.state["gate_pair"] = 1
            rows = [{**_row(s, j, i % 2 == 0), "arm": "c0019", "player": "c0019"} for i, s in enumerate(starts) for j in range(6)]
            (loop.gates / "c0019-cfg-eng.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            mine = [{**r, "arm": "v2"} for r in rows if r["j"] < 2]  # the incumbent's own replays of the first pair
            (loop.gates / "inc-v2-cfg-eng.jsonl").write_text("".join(json.dumps(r) + "\n" for r in mine), encoding="utf-8")
            board = [{"id": s, "cat": "act1-boss"} for s in starts]
            with mock.patch.object(gl, "PROMOTION_RULE", self.RULE), mock.patch.object(gl, "play_tasks", fake_play), \
                    mock.patch.object(gl, "engine_tag", lambda: "eng"), mock.patch.object(gl, "board_starts", lambda: board):
                v = gl.run_gate(loop, "c0020", d, "cfg", 1, 4, "cpu")
            # stage 1: the candidate plays j 2..3 only; the incumbent's 8 games on them are the promotion gate's
            self.assertEqual(sorted(played), sorted([("c0020", j) for j in (2, 3)] * 4))
            self.assertEqual((v["verdict"], v["stage"], v["pairs"], v["rule"]), ("reject", 1, 8, "gym-loop-promote-v2"))
            self.assertEqual((v["stage1_pair"], v["stage1_js"], v["win_diff"]), (1, [2, 3], 0.0))
            self.assertEqual(sorted({r["j"] for r in fg.read_jsonl(loop.gates / "inc-v2-cfg-eng.jsonl")}), [0, 1, 2, 3])
            self.assertIn("stage 1 (j [2, 3]) reject", (loop.root / "loop.log").read_text(encoding="utf-8"))


class LockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        (t / "LAB" / "nightly").mkdir(parents=True)
        (t / "backlog").mkdir()
        self.t = t
        self.patches = [mock.patch.object(fg, "QUIET", t / "QUIET.lock"), mock.patch.object(fg, "MAIN_LOCK", t / "backlog" / "main.lock"),
                        mock.patch.object(fg, "ACTIVE", t / "LAB" / "nightly" / "active"), mock.patch.object(gl, "PAUSE", t / "LAB" / "PAUSE"),
                        mock.patch.object(gl, "HALT", t / "LAB" / "HALT"), mock.patch.object(gl, "GPU_OK", t / "GPU_OK"),
                        mock.patch.object(gl, "in_night_window", lambda now=None: False)]
        for p in self.patches:
            p.start()
        self.root = t / "loop"
        self.root.mkdir()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_the_guards_read_the_labs_locks(self):
        b = lambda starting=True: gl.why_blocked(self.root, starting=starting)  # noqa: E731
        self.assertIsNone(b())
        (self.t / "backlog" / "main.lock").write_text("x 2026-10-30T10:00:00Z pid=1\n")
        self.assertEqual(b(), "main.lock")
        self.assertIsNone(b(starting=False))  # a running step carries on through a landing
        (self.t / "backlog" / "main.lock").unlink()
        now = gl.stamp()
        base = f"w3-blocks {now} pid=1\ngym-loop {now} pid=2\ngym-loop-train {now} pid=3\nold-gate 2026-01-01T00:00:00Z pid=4\n"
        fg.QUIET.write_text(base)
        self.assertIsNone(b())  # w3-blocks, our own lines and a line over 2 h old do not block
        fg.QUIET.write_text(base + f"gym-loop-gate {now} pid=6\n")
        self.assertEqual(b(), "quiet:gym-loop-gate")  # the loop's own gate or build is another heavy step: it stops the daemon
        fg.QUIET.write_text(base)
        fg.QUIET.write_text(fg.QUIET.read_text() + f"f16-gate {now} pid=5\n")
        self.assertEqual(b(), "quiet:f16-gate")
        self.assertEqual(b(starting=False), "quiet:f16-gate")  # another worker's heavy step stops ours at once
        fg.QUIET.unlink()
        for f, why in ((self.t / "LAB" / "PAUSE", "pause"), (self.t / "LAB" / "nightly" / "active", "nightly-active"),
                       (self.t / "LAB" / "HALT", "halt"), (self.root / "STOP", "stop")):
            f.write_text("x\n")
            self.assertEqual(b(), why)
        with mock.patch.object(gl, "in_night_window", lambda now=None: True):
            (self.t / "LAB" / "nightly" / "active").unlink()
            (self.t / "LAB" / "HALT").unlink()
            (self.root / "STOP").unlink()
            self.assertEqual(b(), "night-window")

    def test_the_quiet_line_carries_our_pid_and_keeps_the_others(self):
        fg.QUIET.write_text("w3-blocks 2026-10-30T10:00:00Z pid=1\n")
        gl.hold_quiet(True)
        lines = fg.QUIET.read_text().splitlines()
        self.assertEqual(lines[0], "w3-blocks 2026-10-30T10:00:00Z pid=1")
        self.assertTrue(lines[1].startswith("gym-loop ") and lines[1].endswith(f"pid={os.getpid()}"))
        gl.hold_quiet(True)
        self.assertEqual(len(fg.QUIET.read_text().splitlines()), 2)
        gl.hold_quiet(False)
        self.assertEqual(fg.QUIET.read_text().splitlines(), ["w3-blocks 2026-10-30T10:00:00Z pid=1"])
        fg.QUIET.unlink()
        gl.hold_quiet(True)
        gl.hold_quiet(False)
        self.assertFalse(fg.QUIET.exists())

    def test_gpu_ok_is_taken_only_when_absent_or_ours(self):
        self.assertTrue(gl.take_gpu(gl.TAG))
        self.assertTrue(gl.take_gpu(gl.TRAIN_TAG))
        self.assertEqual(gl.GPU_OK.read_text().splitlines(), [gl.TAG, gl.TRAIN_TAG])
        gl.drop_gpu(gl.TAG)
        gl.drop_gpu(gl.TRAIN_TAG)
        self.assertFalse(gl.GPU_OK.exists())
        gl.GPU_OK.write_text("lab-nightly\n")
        self.assertFalse(gl.take_gpu(gl.TAG))
        self.assertFalse(gl.gpu_allowed())
        gl.GPU_OK.write_text("gym-loop-train\nsome-gate\n")
        self.assertTrue(gl.take_gpu(gl.TAG))
        self.assertEqual(gl.GPU_OK.read_text().splitlines(), ["gym-loop-train", "gym-loop", "some-gate"])
        gl.drop_gpu(gl.TAG)
        self.assertEqual(gl.GPU_OK.read_text().splitlines(), ["gym-loop-train", "some-gate"])

    def test_the_trainer_trains_on_the_card_at_low_cpu_at_night_and_waits_otherwise(self):
        import gym_loop_train as tr  # noqa: PLC0415

        with mock.patch.dict(os.environ, {"GYM_LOOP_NO_GPU": ""}):
            self.assertEqual(tr.trainer_mode(self.root), "gpu")
            gl.GPU_OK.write_text("some-gate\n")
            self.assertEqual(tr.trainer_mode(self.root), "paused:gpu_ok:some-gate")
            gl.GPU_OK.write_text("lab-nightly\n")
            (self.t / "LAB" / "nightly" / "active").write_text("x\n")
            fg.QUIET.write_text(f"lab-nightly {gl.stamp()} pid=9\n")
            self.assertEqual(tr.trainer_mode(self.root), "cpu")  # the night's own lines: training goes on, on one thread
            fg.QUIET.write_text(f"lab-nightly {gl.stamp()} pid=9\nf16-gate {gl.stamp()} pid=5\n")
            self.assertEqual(tr.trainer_mode(self.root), "paused:quiet:f16-gate")
            (self.t / "LAB" / "PAUSE").write_text("x\n")
            self.assertEqual(tr.trainer_mode(self.root), "paused:pause")


def _fake_fight(rows: int, player: str, start: str, split: str, won: bool, rng) -> dict:
    lengths = rng.integers(5, 9, rows).astype(np.int32)
    t = int(lengths.sum())
    meta = {"start": start, "j": 0, "cat": "act1-hallway", "player": player, "split": split, "won": won, "label": 30 if won else 0,
            "value_ok": True}
    return {"types": rng.integers(1, 4, t).astype(np.int16), "ids": rng.integers(0, 5, t).astype(np.int32),
            "owners": np.zeros(t, np.int16), "feats": rng.random((t, 18)).astype(np.float32), "lengths": lengths,
            "visits": rng.random((rows, 32)).astype(np.float32), "qvals": np.zeros((rows, 32), np.float32),
            "legal": np.ones((rows, 32), np.int8), "sims": np.full(rows, 10, np.int32), "k": np.arange(rows, dtype=np.int16),
            "exact": np.zeros(rows, np.int8), "exact_value": np.full(rows, np.nan, np.float32),
            "q_played": np.full(rows, np.nan, np.float32), "solver_best": np.full(rows, -1, np.int16),
            "played": np.zeros(rows, np.int16), "meta": np.array(json.dumps(meta))}


class BoardTests(unittest.TestCase):
    def test_a_rebuild_keeps_every_row_on_the_page_and_adds_the_versions(self):
        html = ('const ARMS = [{"key":"a3-v1","label":"Usual thinking"},{"key":"x16","label":"Sixteen"},'
                '{"key":"judge-v0","label":"The judge"}]; const LOG = ["a", "b"];')
        self.assertEqual(gl.page_arms(html), [("a3-v1", "Usual thinking"), ("x16", "Sixteen"), ("judge-v0", "The judge")])
        self.assertEqual(gl.page_log(html), ["a", "b"])
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            loop = gl.Loop(t / "loop")
            m2, jg = t / "m2.jsonl", t / "judge.jsonl"
            m2.write_text("{}\n")
            jg.write_text("{}\n")
            (loop.board / "fp-v1.jsonl").write_text("{}\n")
            with mock.patch.object(gl, "BOARD_GAMES", {"a3-v1": m2, "x16": m2}):
                with self.assertRaisesRegex(RuntimeError, "judge-v0"):  # a row whose games are unknown is never dropped
                    gl.board_args(loop, gl.page_arms(html), ["v1"])
                gl.write_json(loop.board / "games.json", {"judge-v0": str(jg)})
                files, arms = gl.board_args(loop, gl.page_arms(html), ["v1"])
        self.assertEqual(files, [str(m2), str(jg), str(loop.board / "fp-v1.jsonl")])
        self.assertEqual(arms, ["a3-v1=Usual thinking", "x16=Sixteen", "judge-v0=The judge", "fp-v1=Fight player, version 1"])


class BufferTests(unittest.TestCase):
    def test_compaction_retention_and_the_disk_cap(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            loop = gl.Loop(Path(tmp) / "loop")
            for i in range(3):
                gl.save_npz(loop.incoming / f"f{i}.npz", _fake_fight(4 + i, "v0", f"S{i}", "val" if i == 2 else "train", i != 1, rng))
            (loop.incoming / "half.tmp99.npz").write_bytes(b"partial")
            e = gl.compact(loop, 1)
            self.assertEqual((e["rows"], e["fights"], e["val_rows"], e["players"]), (15, 3, 6, {"v0": 3}))
            self.assertEqual(list(loop.incoming.iterdir()), [])
            with np.load(loop.shards / e["shard"]) as z:
                self.assertEqual(z["fight"].tolist(), [0] * 4 + [1] * 5 + [2] * 6)
                self.assertEqual(z["label"].tolist()[:5], [30, 30, 30, 30, 0])
                self.assertEqual(int(z["lengths"].sum()), z["types"].size)
            self.assertIsNone(gl.compact(loop, 2))
            gl.save_npz(loop.incoming / "cut.npz", _fake_fight(3, "v0", "S7", "train", True, rng))
            cut = gl.compact(loop, 9, value_labels=False)  # a segment stopped early: policy targets kept, value labels off
            self.assertEqual((cut["rows"], cut["value_rows"]), (3, 0))
            (loop.shards / cut["shard"]).unlink()
            for seg in (2, 3):
                gl.save_npz(loop.incoming / "g.npz", _fake_fight(10, "c0001", "S9", "train", True, rng))
                gl.compact(loop, seg)
                time.sleep(1.1)  # the shard's name carries the second
            self.assertEqual(len(list(loop.shards.glob("s*.npz"))), 3)
            kept = gl.retain(loop, max_rows=21, max_days=30, max_gb=1.0)
            self.assertEqual((kept["rows"], kept["removed"]), (20, 1))  # the oldest shard (15 rows) goes
            names = sorted(p.name for p in loop.shards.glob("s*.npz"))
            self.assertTrue(all(not n.startswith("s000001") for n in names))
            kept = gl.retain(loop, max_rows=10**9, max_days=30, max_gb=1e-9)
            self.assertEqual(kept["rows"], 0)  # the disk cap: nothing fits in one byte

    def test_the_trainers_buffer_keeps_the_newest_window_and_its_targets(self):
        import gym_loop_train as tr  # noqa: PLC0415

        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            loop = gl.Loop(Path(tmp) / "loop")
            for seg in range(3):
                f = _fake_fight(10, "v0", f"S{seg}", "train", seg != 1, rng)
                if seg == 2:  # an exact grade where the played move was worse than the solver's
                    f["exact"][0], f["exact_value"][0], f["q_played"][0], f["solver_best"][0] = 1, 0.9, 0.5, 7
                gl.save_npz(loop.incoming / f"x{seg}.npz", f)
                gl.compact(loop, seg)
                time.sleep(1.1)
            buf = tr.Buffer(loop.shards, window=15)
            self.assertEqual(buf.refresh(), 30)
            self.assertEqual(buf.rows, 20)  # the oldest shard is out of the window
            self.assertEqual(buf.refresh(), 0)
            flat = buf.arrays()
            self.assertEqual(len(flat["train_idx"]), 20)
            import torch  # noqa: PLC0415

            inputs, tg = tr.batch_tensors(flat, np.arange(20), "cpu", width=5, n_buckets=28)
            self.assertEqual(tuple(inputs["types"].shape)[0], 20)
            self.assertTrue(bool((inputs["mask"] == (inputs["types"] == 0)).all()))
            self.assertEqual(tg["bucket"].tolist(), [0] * 10 + [6] * 10)  # death is class 0; 30 HP is 1 + 29 // 5
            self.assertEqual(int(torch.argmax(tg["policy"][10])), 7)  # the solver's move where the search's was worse
            self.assertAlmostEqual(float(tg["policy"][11].sum()), 1.0, places=5)


# ---------------------------------------------------------------------------------------------------------- real games


class LoopGameTests(unittest.TestCase):
    """Short arena fights at a3-v1 with the fixture prior, through the loop's own game steps."""

    @classmethod
    def setUpClass(cls):
        from prior_fixture import export_prior  # noqa: PLC0415 -- torch
        from test_gym_lockstep import _make_starts  # noqa: PLC0415

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
        cls.loop = gl.Loop(root / "loop")
        starts = [{**s, "seed": f"RUN{i}", "split": "val" if i == 2 else "train"} for i, s in enumerate(got) if not s["id"].startswith("F18")]
        gl.write_start_set(cls.loop.starts, starts)
        cls.agent = root / "agents" / "fixture"
        cls.init = ("core2", str(cls.agent.parent), cls.agent.name, str(cls.loop.root))
        info = {"starts": str(cls.loop.starts / "starts.jsonl"), "incoming": str(cls.loop.incoming), "player": "fixture",
                "config": "a3-v1", "record": True, "ns": "g", "exact_wall_s": 0.3}
        cls.tasks = [("a3-v1", s["id"], j, "act1-boss", info) for s in starts for j in range(2)]
        cls.rec = cls.loop.games / "rec.jsonl"
        import gym_lockstep  # noqa: PLC0415

        stopped, _ = gym_lockstep.run_lockstep(cls.tasks, cls.rec, 1, 2, cls.init,
                                               {"prior": "cpu", "game_steps": "gym_loop:gen_game_steps"})
        assert stopped is None
        cls.plain = cls.loop.games / "plain.jsonl"
        plain = [(a, s, j, c, {**i, "record": False}) for a, s, j, c, i in cls.tasks]
        stopped, _ = gym_lockstep.run_lockstep(plain, cls.plain, 1, 2, cls.init,
                                               {"prior": "cpu", "game_steps": "gym_loop:gen_game_steps"})
        assert stopped is None
        cls.seq = cls.loop.games / "seq.jsonl"
        (cls.loop.root / "starts.jsonl").write_text("".join(json.dumps(s) + "\n" for s in starts), encoding="utf-8")
        stopped = fg.run_tasks([t[:4] for t in cls.tasks], fg._game_task, cls.seq, 1, init=fg._init_worker, init_args=cls.init)  # noqa: SLF001
        assert stopped is None
        cls.entry = gl.compact(cls.loop, 1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _by(self, path):
        return {(r["start"], int(r["j"])): r for r in fg.read_jsonl(path)}

    def test_recording_changes_no_move_and_plays_the_gyms_game(self):
        rec, plain, seq = self._by(self.rec), self._by(self.plain), self._by(self.seq)
        self.assertEqual(len(rec), 6)
        self.assertEqual(sorted(rec), sorted(seq))
        for key in rec:
            for f in ("won", "end_hp", "turns", "decisions", "sims", "moves_sha", "first_move", "L0"):
                self.assertEqual(rec[key][f], plain[key][f], (key, f))
                self.assertEqual(rec[key][f], seq[key][f], (key, f))
            self.assertEqual(rec[key]["samples"], rec[key]["decisions"])
            self.assertEqual(rec[key]["graded"]["graded"], rec[key]["decisions"])
            self.assertNotIn("grades", rec[key])

    def test_every_decision_is_one_sample_with_the_root_visits_and_the_outcome(self):
        rec = self._by(self.rec)
        self.assertEqual(self.entry["rows"], sum(r["decisions"] for r in rec.values()))
        self.assertEqual(self.entry["val_rows"], sum(r["decisions"] for r in rec.values() if r["split"] == "val"))
        with np.load(self.loop.shards / self.entry["shard"]) as z:
            metas = json.loads(str(z["fights_meta"]))
            searched = z["visits"].sum(axis=1) > 0
            self.assertGreater(searched.mean(), 0.5)  # a forced move is not searched: no policy target there
            self.assertTrue(bool((z["visits"][z["legal"] == 0] == 0).all()))  # visits only on legal actions
            for i, m in enumerate(metas):
                labels = set(z["label"][z["fight"] == i].tolist())
                self.assertEqual(labels, {m["end_hp"] if m["won"] else 0})
            self.assertEqual(set(z["played"][z["k"] == 0].tolist()) <= set(range(32)), True)

    def test_the_trainer_spends_its_budget_checkpoints_a_playable_agent_and_resumes(self):
        import combat_prior  # noqa: PLC0415
        import gym_loop_train as tr  # noqa: PLC0415

        a = argparse.Namespace(root=str(self.loop.root), base=str(self.agent), parent_pid=0, every_min=0.0, max_epochs=1.0,
                               window=10_000, batch=16, lr=1e-4, policy_weight=1.0, seed=0)
        with mock.patch.dict(os.environ, {"GYM_LOOP_NO_GPU": "1"}), mock.patch.object(tr, "CHUNK_S", 0.5):
            t = tr.Trainer(a)
            t.refresh()
            train_rows = len(t.buffer.arrays()["train_idx"])
            self.assertEqual(t.st["budget"], train_rows)
            n = 0
            while t.st["budget"] >= a.batch:
                n += t.train_chunk("cpu")
            self.assertGreater(n, 0)
            self.assertLess(t.st["budget"], a.batch)
            self.assertEqual(t.train_chunk("cpu"), 0)  # fresh data spent: it idles
            name = t.checkpoint()
            ck = self.loop.checkpoints / name
            champ = combat_prior.find(ck)  # the players load it as an agent's prior
            self.assertIsNotNone(champ)
            info = json.loads((ck / "train.json").read_text())
            self.assertIn("val_top1_agree", info)
            self.assertIn("base_val_top1_agree", info)
            steps = t.st["steps"]
            t.save_state()
            t2 = tr.Trainer(a)
            self.assertEqual(t2.st["steps"], steps)
            t2.refresh()
            self.assertLess(t2.st["budget"], a.batch)  # a restart does not count the same shard again
            self.assertEqual(self.loop.newest_checkpoint()[0], name)

    def test_the_gate_plays_the_incumbent_once_and_decides_by_the_rule(self):
        rule = {**gl.PROMOTION_RULE, "starts": 2, "stage1_seeds": 1, "seeds": 2, "futility_points": -100.0}
        board = Path(self.tmp.name) / "board"
        board.mkdir(exist_ok=True)
        lite = json.loads((self.loop.starts / "starts.lite.json").read_text())
        docs = [gl.load_start(str(self.loop.starts / "starts.jsonl"), s["id"]) for s in lite]
        (board / "starts.jsonl").write_text("".join(json.dumps(s) + "\n" for s in docs), encoding="utf-8")
        self.loop.state["versions"]["v0"]["dir"] = str(self.agent)
        with mock.patch.object(gl, "PROMOTION_RULE", rule), mock.patch.object(gl, "BOARD_DIR", board), \
                mock.patch.object(gl, "why_blocked", lambda *a, **k: None), mock.patch.object(gl, "hold_quiet", lambda on: None):
            v = gl.run_gate(self.loop, "c9999", self.agent, "a3-v1", 1, 4, "cpu")
            inc_games = fg.read_jsonl(self.loop.gates / v["inc_file"])
            v2 = gl.run_gate(self.loop, "c9999", self.agent, "a3-v1", 1, 4, "cpu")
        self.assertEqual(len(inc_games), 4)
        self.assertEqual(len(fg.read_jsonl(self.loop.gates / v["inc_file"])), 4)  # played once, kept
        self.assertEqual(v["pairs"], 4)
        self.assertEqual(v["verdict"], "reject")  # the same network: no candidate-only win, McNemar p = 1
        self.assertEqual((v["rule"], v["stage1_js"]), ("gym-loop-promote-v2", [0]))  # a new loop's stage 1 plays the first pair
        self.assertEqual(v2["verdict"], v["verdict"])
        self.assertEqual({r["arm"] for r in inc_games}, {"v0"})


if __name__ == "__main__":
    unittest.main()
