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
import contextlib
import datetime as dt
import io
import json
import multiprocessing as mp
import os
import random
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

import fight_exam as fe  # noqa: E402
import fight_gym as fg  # noqa: E402
import gym_loop as gl  # noqa: E402
import test_fight_exam as fx  # noqa: E402

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
        rule = {**gl.PROMOTION_RULE_V2, "starts": 10, "stage1_seeds": 1, "seeds": 4}
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

    RULE = {**gl.PROMOTION_RULE_V2, "starts": 4, "stage1_seeds": 2, "seeds": 6, "futility_points": 1.0}  # three pairs of j (the v2 rule: v3 is in test_gym_gate)

    def _rows(self, who, js, win, starts=("S0", "S1", "S2", "S3")):
        return [{**_row(s, j, win(i, j)), "arm": who, "player": who} for i, s in enumerate(starts) for j in js]

    def test_the_rule_is_v2_and_the_pairs_rotate_one_per_promotion_and_wrap(self):
        rule = gl.PROMOTION_RULE_V2
        self.assertEqual(rule["name"], "gym-loop-promote-v2")
        self.assertEqual(gl.PROMOTION_RULE["name"], "gym-loop-promote-v3")  # v3 rotates the same pairs
        self.assertEqual([gl.stage1_js(gl.PROMOTION_RULE, p) for p in range(5)], [gl.stage1_js(rule, p) for p in range(5)])
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


# A run row of a night's emulator arm: 12 Act-1 hallway fights, an Act-1 elite and boss, two Act-2 hallway fights and an Act-2 elite.
_RUN_NODES, _RUN_FLOOR, _RUN_STARTS = "M?ME" + "RMMMMMMMMMM" + "B" + "MMEMB", 20, 17


def _night_runs(tag: str, n: int) -> list[dict]:
    return [{"seed": f"{tag}{i:03d}", "ix": i, "nodes": _RUN_NODES, "floor": _RUN_FLOOR, "end": "death"} for i in range(n)]


class StartSetTests(unittest.TestCase):
    """The start set follows the nights (D44 step E): every finished night's emulator arm, a window of the newest, each night's own
    draw of runs, only new runs replayed, only the picked runs assembled (streamed, with a byte index that reads every start back);
    the kind weights follow the set and the incumbent, keeping the incumbent's games."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.lab = t / "LAB"
        (self.lab / "nightly").mkdir(parents=True)
        self.arms = {"2026-10-26/g5-emu": _night_runs("A", 8), "2026-11-03/g11-emu": _night_runs("B", 8),
                     "2026-11-04/g11-emu": _night_runs("C", 8)}
        self.replayed: list[str] = []
        self.patches = [mock.patch.object(gl, "LAB", self.lab), mock.patch.object(fg, "QUIET", t / "QUIET.lock"),
                        mock.patch.object(gl, "board_run_seeds", lambda: {"A007"}),
                        mock.patch.object(fg, "run_tasks", self._fake_replays)]
        for p in self.patches:
            p.start()
        from tools.solver import coverage  # noqa: PLC0415

        self.patches.append(mock.patch.object(coverage, "_arm", self._fake_arm))
        self.patches[-1].start()
        self.loop = gl.Loop(t / "loop")

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _fake_arm(self, source):
        night = "/".join(Path(source).parts[-2:])
        if night not in self.arms:
            raise FileNotFoundError(source)
        return {"config": "a3-v1"}, self.arms[night], "gen11/candidate", "core2"

    def _fake_replays(self, tasks, fn, target, procs, **kw):
        with open(target, "a", encoding="utf-8") as fh:
            for _, seed, ix, _config, wanted, _row in tasks:
                self.replayed.append(seed)
                starts = [{**w, "node_type": gl.NODE_TYPE[w["kind"]], "hp": 40, "max_hp": 80, "doc": {"start": w["id"]},
                           "replay_same": True} for w in wanted]
                fh.write(json.dumps({"_task": ["replay", seed, ix], "seed": seed, "ix": ix, "starts": starts}) + "\n")
        return None

    def _finish(self, night):
        d = self.lab / "nightly" / night
        d.mkdir(parents=True, exist_ok=True)
        (d / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.arms[night]), encoding="utf-8")
        (d / "summary.json").write_text("{}", encoding="utf-8")

    def test_every_finished_nights_emulator_arm_is_found_oldest_first_and_the_window_keeps_the_newest(self):
        self._finish("2026-11-03/g11-emu")
        self._finish("2026-10-26/g5-emu")
        (self.lab / "nightly" / "2026-11-04" / "g11-emu").mkdir(parents=True)  # the night still running: no summary yet
        (self.lab / "nightly" / "2026-11-03" / "g11").mkdir()  # the real arm
        (self.lab / "nightly" / "last-ok").write_text("2026-11-03T23:58:55Z\n", encoding="utf-8")
        self.assertEqual(gl.nights_available(), ("2026-10-26/g5-emu", "2026-11-03/g11-emu"))
        self.assertEqual(gl.nights_available(max_nights=1), ("2026-11-03/g11-emu",))
        self.assertEqual(gl.nights_available(max_nights=0), ("2026-10-26/g5-emu", "2026-11-03/g11-emu"))
        runs = _night_runs("X", 40)
        one = gl.pick_runs(runs, 5, gl.night_seed(1, "2026-11-03/g11"), set())
        self.assertEqual(one, gl.pick_runs(runs, 5, gl.night_seed(1, "2026-11-03/g11"), set()))  # by its own label only
        self.assertNotEqual(one, gl.pick_runs(runs, 5, gl.night_seed(1, "2026-11-04/g11"), set()))

    def test_a_new_night_replays_only_its_own_runs_and_the_set_holds_only_the_picked_runs(self):
        first = ("2026-10-26/g5-emu",)
        self.assertEqual(gl.starts_stale(self.loop, first, 3), "no start set")
        meta = gl.build_starts(self.loop, first, runs_per_night=3)
        self.assertEqual(len(self.replayed), 3)
        self.assertNotIn("A007", self.replayed)  # a board run is never a training start
        self.assertEqual(meta["starts"], 3 * _RUN_STARTS)
        self.assertEqual(meta["by_cat"]["act1-boss"], 3)
        self.assertEqual(meta["spec"], {"nights": ["2026-10-26/g5"], "runs_per_night": 3})
        self.assertIsNone(gl.starts_stale(self.loop, first, 3))
        self.assertEqual(gl.starts_stale(self.loop, first, 4), "runs per night 3 to 4")
        both = ("2026-10-26/g5-emu", "2026-11-03/g11-emu")
        self.assertEqual(gl.starts_stale(self.loop, both, 3), "nights: 1 new, 0 gone")
        # a replay file keeps what it ever replayed: a run no longer picked, and a picked run replayed twice
        target = self.loop.starts / "replays-2026-10-26-g5.jsonl"
        lines = target.read_text(encoding="utf-8").splitlines()
        stray = json.loads(lines[0])
        stray["seed"] = stray["_task"][1] = "A_OLD"
        for s in stray["starts"]:
            s["id"] = s["id"].replace(json.loads(lines[0])["seed"], "A_OLD")
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(stray) + "\n" + lines[1] + "\n" + '{"_task": ["replay", "A_CUT", 9], "capped": true}\n')
        meta = gl.build_starts(self.loop, both, runs_per_night=3)
        self.assertEqual(len(self.replayed), 6)
        self.assertTrue(all(s.startswith("B") for s in self.replayed[3:]))  # the first night's runs are not replayed again
        self.assertEqual(meta["starts"], 6 * _RUN_STARTS)
        lite = json.loads((self.loop.starts / "starts.lite.json").read_text(encoding="utf-8"))
        ids = [s["id"] for s in lite]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(any("A_OLD" in i for i in ids))
        self.assertEqual({s["split"] for s in lite} <= {"train", "val"}, True)
        gl._START_CACHE.clear()  # noqa: SLF001
        for sid in ids:  # the byte index reads every start back, document and all
            self.assertEqual(gl.load_start(str(self.loop.starts / "starts.jsonl"), sid)["doc"], {"start": sid})
        self.assertIsNone(gl.starts_stale(self.loop, both, 3))
        # the window moves on: the oldest night leaves, the others keep their picks and nothing is replayed for them
        self.arms["2026-11-04/g11-emu"] = _night_runs("C", 8)
        newest = ("2026-11-03/g11-emu", "2026-11-04/g11-emu")
        b_ids = {i for i in ids if i.startswith("2026-11-03/")}
        self.assertEqual(gl.starts_stale(self.loop, newest, 3), "nights: 1 new, 1 gone")
        gl.build_starts(self.loop, newest, runs_per_night=3)
        self.assertTrue(all(s.startswith("C") for s in self.replayed[6:]) and len(self.replayed) == 9)
        lite = json.loads((self.loop.starts / "starts.lite.json").read_text(encoding="utf-8"))
        self.assertEqual({s["id"] for s in lite if s["id"].startswith("2026-11-03/")}, b_ids)
        self.assertFalse(any(s["id"].startswith("2026-10-26/") for s in lite))

    def test_a_set_from_before_the_night_window_is_read_by_its_nights_and_runs(self):
        (self.loop.starts / "starts.jsonl").write_text("", encoding="utf-8")
        gl.write_json(self.loop.starts / "meta.json", {"nights": {"2026-10-26/g5": {"runs": 50}, "2026-10-27/g7": {"runs": 50}},
                                                       "runs_per_night": 50, "written": "2026-10-30T16:34:23Z"})
        old = ("2026-10-26/g5-emu", "2026-10-27/g7-emu")
        self.assertIsNone(gl.starts_stale(self.loop, old, 50))
        self.assertEqual(gl.starts_stale(self.loop, old, 250), "runs per night 50 to 250")
        self.assertEqual(gl.starts_stale(self.loop, old + ("2026-11-02/g9-emu",), 50), "nights: 1 new, 0 gone")

    def test_an_unreadable_night_is_skipped_and_recorded(self):
        self._finish("2026-10-26/g5-emu")
        meta = gl.build_starts(self.loop, ("2026-10-26/g5-emu", "2026-10-28/g7-emu"), runs_per_night=2)
        self.assertEqual(meta["nights"]["2026-10-28/g7"], {"skipped": "unreadable: FileNotFoundError"})
        self.assertEqual(meta["starts"], 2 * _RUN_STARTS)
        self.assertIsNone(gl.starts_stale(self.loop, ("2026-10-26/g5-emu", "2026-10-28/g7-emu"), 2))

    def test_the_weights_follow_the_set_and_the_incumbent_and_keep_the_incumbents_games(self):
        cats = ("act1-hallway", "act1-boss", "act2-elite")
        starts = [{"id": f"N:{c}:{i}", "cat": c, "split": "val" if i == 0 else "train", "doc": {}} for c in cats for i in range(5)]
        gl.write_start_set(self.loop.starts, starts)
        gl.write_json(self.loop.starts / "meta.json", {"written": "T1", "fights_per_run": {c: 1.0 for c in cats}})
        self.loop.state["versions"]["v2"] = {"dir": str(self.loop.root / "players" / "v2")}
        self.loop.state["incumbent"] = "v2"
        with open(self.loop.starts / "weights_games.jsonl", "w", encoding="utf-8") as fh:  # the old incumbent's games do not count
            fh.write(json.dumps({"start": "N:act1-boss:1", "j": 0, "cat": "act1-boss", "player": "v0", "config": "a3-v1",
                                 "won": False, "L0": 0.0, "start_hp": 40, "start_potions": 0, "start_max_hp": 80}) + "\n")
        played: list[tuple] = []

        def fake_play(loop, tasks, target, pdir, procs, k, prior, label):
            with open(target, "a", encoding="utf-8") as fh:
                for _, sid, j, c, info in tasks:
                    played.append((info["player"], sid, j))
                    lost = {"act1-hallway": 0.1, "act1-boss": 0.5, "act2-elite": 0.3}[c]
                    fh.write(json.dumps({"start": sid, "j": j, "cat": c, "player": info["player"], "config": info["config"],
                                         "won": True, "L0": 1.0 - lost, "start_hp": 80, "start_potions": 0, "start_max_hp": 80}) + "\n")
            return None

        with mock.patch.object(gl, "play_tasks", fake_play):
            self.assertEqual(gl.weights_stale(self.loop), "no weights")
            w = gl.kind_weights(self.loop, per_kind=2, seeds=2)
            self.assertEqual(len(played), 12)
            self.assertTrue(all(p == "v2" and not sid.endswith(":0") for p, sid, _ in played))  # training starts only
            self.assertEqual(w["games"], {c: 4 for c in cats})
            self.assertAlmostEqual(w["weights"]["act1-boss"], 0.5 / 0.9)
            self.assertEqual((w["player"], w["starts_written"]), ("v2", "T1"))
            self.assertIsNone(gl.weights_stale(self.loop))
            gl.write_json(self.loop.starts / "meta.json", {"written": "T2", "fights_per_run": {c: 2.0 if c == "act1-hallway" else 1.0
                                                                                               for c in cats}})
            self.assertEqual(gl.weights_stale(self.loop), "start set rebuilt")
            w = gl.kind_weights(self.loop, per_kind=2, seeds=2)
            self.assertEqual(len(played), 12)  # a rebuilt set plays nothing: the incumbent's games are kept
            self.assertAlmostEqual(w["weights"]["act1-hallway"], 0.2 / 1.0)
            self.assertIsNone(gl.weights_stale(self.loop))
            self.loop.state["versions"]["v3"] = {"dir": str(self.loop.root / "players" / "v3")}
            self.loop.state["incumbent"] = "v3"
            self.assertEqual(gl.weights_stale(self.loop), "incumbent v2 to v3")
            gl.kind_weights(self.loop, per_kind=3, seeds=2)
            self.assertEqual([p for p, _, _ in played[12:]], ["v3"] * 18)
            gl.kind_weights(self.loop, per_kind=4, seeds=2)  # topped up: the new games only
            self.assertEqual(len(played), 30 + 6)
            self.assertEqual(len({(p, s, j) for p, s, j in played}), len(played))


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
            # the pre-filter's reference: whose figures the base ones are, on which split, and the same figures from `score_pair`
            inc_name, inc_dir = t.incumbent()
            self.assertEqual((info["base_version"], info["split"]), (inc_name, gl.val_split_id()))
            self.assertEqual(info["base_sha256"], gl.sha256_file(inc_dir / "combat_policy.pt"))
            got = tr.score_pair(self.loop.root, ck, inc_dir, window=a.window)
            self.assertEqual((got["base_sha256"], got["split"]), (info["base_sha256"], info["split"]))
            self.assertEqual(got["sha256"], gl.sha256_file(ck / "combat_policy.pt"))
            for k in ("val_rows", "val_policy_ce", "base_val_policy_ce", "val_value_nll", "base_val_value_nll"):
                self.assertAlmostEqual(got[k], info[k], places=4, msg=k)
            steps = t.st["steps"]
            t.save_state()
            t2 = tr.Trainer(a)
            self.assertEqual(t2.st["steps"], steps)
            t2.refresh()
            self.assertLess(t2.st["budget"], a.batch)  # a restart does not count the same shard again
            self.assertEqual(self.loop.newest_checkpoint()[0], name)

    def test_the_gate_plays_the_incumbent_once_and_decides_by_the_rule(self):
        rule = {**gl.PROMOTION_RULE_V2, "starts": 2, "stage1_seeds": 1, "seeds": 2, "futility_points": -100.0}
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


# ---------------------------------------------------------------------------------------------------------- the human bar at the gate


class HumanBarGateTests(unittest.TestCase):
    """The human line after a stage-2 verdict (D55 addendum 11): a tiny stratum (two starts, one draw) and a fake player. The rule is
    not read by it and does not read it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.loop = gl.Loop(t / "loop")
        gl.write_start_set(self.loop.starts, [{**s, "doc": {"x": s["id"], "hp": 3}} for s in fx.synthetic_pool()])
        self.exam = self.loop.root / "exam"
        with mock.patch.object(gl, "board_run_seeds", lambda: set()):
            fe.build_exam(self.loop.root, self.exam, per_cell=15)
        recs, outcomes = fx.human_pool(60)
        fx.write_human_files(self.loop.starts, recs, outcomes)
        fe.build_human(self.loop.root, self.exam, per_cell=1, total=2, exam_ids=[])
        self.human = fe.load_human_exam(self.exam)
        self.assertEqual(len(self.human), 2)
        self.cand_dir = t / "ckpt"
        self.cand_dir.mkdir()
        (self.cand_dir / "model.json").write_text("{}", encoding="utf-8")
        (self.loop.gates / "cand.jsonl").write_text(json.dumps(_row("S0", 0, True)) + "\n", encoding="utf-8")
        self.calls: list[dict] = []
        self.statuses: list[dict] = []

    def tearDown(self):
        self.tmp.cleanup()

    def verdict(self, stage: int, verdict: str, **kw) -> dict:
        return {"candidate": "c0190", "incumbent": "v0", "config": "cfg", "engine": "eng", "rule": gl.PROMOTION_RULE["name"], "stage1_pair": 0,
                "stage1_js": [0, 1], "verdict": verdict, "stage": stage, "why": None, "pairs": 8, "cand_wins": 4, "inc_wins": 3,
                "win_diff": 0.125, "win_se": 0.05, "l0_diff": 0.0, "p": 0.02, "cand_file": "cand.jsonl", "inc_file": "inc.jsonl", **kw}

    def rows(self, label: str, wins) -> list[dict]:
        return [{"start": sid, "j": 0, "won": bool(wins(i)), "capped": False, "L0": 0.6 if wins(i) else 0.0, "hp_lost": 10.0 if wins(i) else 50.0,
                 "config": "a3-v1", "stratum": "human", "engine": "eng", "arm": label, "player": label, "cell": s["cell"], "half": s["half"]}
                for i, (sid, s) in enumerate(sorted(self.human.items()))]

    def write_games(self, label: str, rows: list[dict]) -> Path:
        p = self.exam / f"games-{label}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return p

    def fake_play(self, stop=None, wins=lambda i: i == 0, boom: bool = False):
        def play(loop, tasks, target, player_dir, procs, k, prior, what, **kw):
            self.calls.append({"n": len(tasks), "target": Path(target), "dir": Path(player_dir), "procs": procs, "k": k, "prior": prior,
                               "what": what, "tasks": tasks, **kw})
            if boom:
                raise RuntimeError("the lockstep fell over")
            if stop:
                return stop
            order = {sid: i for i, sid in enumerate(sorted(self.human))}
            with open(target, "a", encoding="utf-8") as fh:
                for _arm, sid, j, _cat, info in tasks:
                    w = wins(order[sid])
                    fh.write(json.dumps({"start": sid, "j": j, "won": w, "capped": False, "L0": 0.6 if w else 0.0, "hp_lost": 10.0 if w else 50.0,
                                         "config": info["config"], "stratum": "human", "engine": "eng", "arm": info["label"]}) + "\n")
            return None
        return play

    def daemon(self, **kw):
        a = argparse.Namespace(player_config="cfg", procs=3, gate_k=7, prior="cpu", judge="", prefilter="off", board="off", **kw)
        d = gl.Daemon.__new__(gl.Daemon)
        d.a, d.loop = a, self.loop
        d.status = lambda phase, **k: self.statuses.append({"phase": phase, **k})
        return d

    def gate(self, v: dict, play=None, human_bar: str | None = "on", draws: int = 1) -> gl.Daemon:
        kw = {} if human_bar is None else {"human_bar": human_bar}
        d = self.daemon(**kw)
        with mock.patch.object(gl, "run_gate", lambda *a, **k: dict(v)), mock.patch.object(gl, "play_tasks", play or self.fake_play()), \
                mock.patch.dict(gl.HUMAN_BAR, {"draws": draws}):
            d.gate("c0190", self.cand_dir)
        return d

    def log(self) -> str:
        p = self.loop.root / "loop.log"
        return p.read_text(encoding="utf-8") if p.is_file() else ""

    def recorded(self) -> tuple[dict, dict]:
        state = json.loads((self.loop.root / "state.json").read_text(encoding="utf-8"))["gated"]["c0190"]
        last = [json.loads(x) for x in (self.loop.gates / "verdicts.jsonl").read_text(encoding="utf-8").splitlines()][-1]
        return state, last

    def test_a_stage_one_reject_plays_nothing_and_records_no_human_field(self):
        self.gate(self.verdict(1, "reject", why="stage 1: -3.0 points"))
        self.assertEqual(self.calls, [])
        state, last = self.recorded()
        self.assertEqual(state["verdict"], "reject")
        self.assertNotIn("human", state)
        self.assertNotIn("human", last)
        self.assertNotIn("human bar", self.log())
        self.assertFalse((self.exam / "games-c0190.jsonl").exists())

    def test_a_verdict_that_is_not_a_stage_two_promote_or_reject_plays_nothing(self):
        self.gate(self.verdict(2, "incomplete"))
        self.gate(self.verdict(1, "continue"))
        self.assertEqual(self.calls, [])

    def test_a_stage_two_verdict_plays_the_stratum_with_the_gates_processes_and_records_the_line(self):
        self.write_games("v0", self.rows("v0", lambda i: i == 1))
        d = self.gate(self.verdict(2, "reject", why="McNemar p 0.4 >= 0.05"))
        self.assertEqual(len(self.calls), 1)
        c = self.calls[0]
        self.assertEqual((c["n"], c["target"].name, c["dir"], c["procs"], c["k"], c["prior"], c["what"]),
                         (2, "games-c0190.jsonl", self.cand_dir, 3, 7, "cpu", "human-bar-c0190"))
        self.assertEqual((c["steps"], c["daemon"]), ("fight_exam:exam_game_steps", d))  # the exam's game, stoppable by the daemon's reasons
        self.assertEqual({(t[1], t[2]) for t in c["tasks"]}, {(sid, 0) for sid in self.human})
        for t in c["tasks"]:
            info = t[4]
            self.assertEqual((Path(info["starts"]).name, info["config"], info["ns"], info["label"], info["player"]),
                             ("starts-human.jsonl", "a3-v1", fe.NS, "c0190", "c0190"))
        state, last = self.recorded()
        self.assertEqual(state["verdict"], "reject")  # the rule's word stands
        self.assertEqual(state["human"], last["human"])
        h = state["human"]
        self.assertEqual((h["starts"], h["stratum_starts"], h["incumbent"], h["candidate"], h["config"]), (2, 2, "v0", "c0190", "a3-v1"))
        self.assertEqual((h["cand_surv"], h["inc_surv"]), (50.0, 50.0))
        want_human = 100 * sum(1 for s in self.human.values() if s["human_survived"]) / 2
        self.assertEqual(h["human_surv"], want_human)
        self.assertRegex(h["line"], r"^human bar: cand 50\.0 / inc 50\.0 / human \d+\.\d in 100 on 2 starts; cand-inc [+-]\d+\.\d "
                                    r"\[[+-]\d+\.\d, [+-]\d+\.\d\]; HP lost \d+\.\d / \d+\.\d / \d+\.\d$")
        self.assertIn("gate c0190: " + h["line"], self.log())
        self.assertIn({"phase": "gate", "candidate": "c0190", "part": "human bar"}, self.statuses)
        self.assertEqual(len(fg.read_jsonl(self.exam / "games-c0190.jsonl")), 2)
        self.assertFalse((self.exam / "games-v1.jsonl").exists())  # a reject names no new version

    def test_a_promote_plays_records_and_keeps_the_games_under_the_new_versions_name(self):
        self.write_games("v0", self.rows("v0", lambda i: True))
        self.gate(self.verdict(2, "promote"))
        state, last = self.recorded()
        self.assertEqual((state["verdict"], last["verdict"], state["human"]["cand_surv"], state["human"]["inc_surv"]), ("promote", "promote", 50.0, 100.0))
        self.assertEqual(json.loads((self.loop.root / "state.json").read_text())["incumbent"], "v1")
        mine, theirs = fg.read_jsonl(self.exam / "games-c0190.jsonl"), fg.read_jsonl(self.exam / "games-v1.jsonl")
        self.assertEqual(len(theirs), 2)  # found under the new incumbent's name; the candidate's file stays
        self.assertEqual({(r["start"], r["j"], r["won"]) for r in theirs}, {(r["start"], r["j"], r["won"]) for r in mine})
        self.assertEqual({(r["arm"], r["player"], r["reused_from"]) for r in theirs}, {("v1", "v1", "c0190")})
        self.assertIn("2 human-stratum games of c0190 kept as v1's (games-v1.jsonl)", self.log())
        self.assertEqual(gl.human_bar_adopt(self.loop, "c0190", "v1"), 0)  # once
        self.assertEqual(len(fg.read_jsonl(self.exam / "games-v1.jsonl")), 2)
        # the exam reads them as the new incumbent's own
        h = fe.load_human_exam(self.exam)
        self.assertEqual(len(fe.human_games(theirs, h, "a3-v1")), 2)

    def test_a_cut_play_records_stopped_and_the_verdict_goes_on_to_promote(self):
        self.gate(self.verdict(2, "promote"), play=self.fake_play(stop="night-window"))
        state, last = self.recorded()
        self.assertEqual((state["human"], last["human"], state["verdict"]), ("stopped", "stopped", "promote"))
        self.assertEqual(json.loads((self.loop.root / "state.json").read_text())["incumbent"], "v1")  # promoted all the same
        self.assertIn("human bar: stopped (night-window)", self.log())
        self.assertFalse((self.exam / "games-v1.jsonl").exists())

    def test_a_play_that_fails_never_loses_or_changes_the_verdict(self):
        self.gate(self.verdict(2, "reject"), play=self.fake_play(boom=True))
        state, _ = self.recorded()
        self.assertEqual((state["verdict"], state["human"]), ("reject", "error: RuntimeError"))
        self.assertIn("human bar: RuntimeError: the lockstep fell over", self.log())

    def test_the_games_already_played_are_not_played_again(self):
        self.write_games("c0190", self.rows("c0190", lambda i: True)[:1])
        self.gate(self.verdict(2, "reject"))
        self.assertEqual([c["n"] for c in self.calls], [1])
        self.assertEqual(len(fg.read_jsonl(self.exam / "games-c0190.jsonl")), 2)
        self.calls.clear()
        self.gate(self.verdict(2, "reject"))
        self.assertEqual(self.calls, [])  # every game has its row: read, not played
        self.assertIn("human bar: cand ", self.recorded()[0]["human"]["line"])

    def test_without_incumbent_games_the_candidates_side_is_given_alone(self):
        self.gate(self.verdict(2, "reject"))
        h = self.recorded()[0]["human"]
        self.assertNotIn("inc_surv", h)
        self.assertEqual(h["inc_games"], 0)
        self.assertIn("(no human-stratum games for the incumbent v0)", h["line"])
        self.assertNotIn("cand-inc", h["line"])

    def test_an_incumbent_on_another_engine_is_flagged(self):
        self.write_games("v0", [{**r, "engine": "old"} for r in self.rows("v0", lambda i: True)])
        self.gate(self.verdict(2, "reject"))
        h = self.recorded()[0]["human"]
        self.assertFalse(h["same_engine"])
        self.assertTrue(h["line"].endswith("; engines differ (cand eng, inc old)"))

    def test_the_flag_turns_it_off_and_the_stratum_must_exist(self):
        self.gate(self.verdict(2, "reject"), human_bar="off")
        self.assertEqual(self.calls, [])
        self.assertNotIn("human", self.recorded()[0])
        self.gate(self.verdict(2, "reject"), human_bar=None)  # a daemon from before the flag: the default, a promotion only
        self.assertEqual(self.calls, [])
        self.gate(self.verdict(2, "reject"), human_bar="promotion")
        self.assertEqual(self.calls, [])
        self.gate(self.verdict(2, "reject"), human_bar="on")  # `on` keeps its meaning: every stage-2 verdict
        self.assertEqual(len(self.calls), 1)
        self.calls.clear()
        (self.exam / "starts-human.lite.json").unlink()
        self.gate(self.verdict(2, "reject"), human_bar="on")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.recorded()[0]["human"], "no human stratum")
        self.assertIn("human bar: no human stratum in exam/", self.log())

    def test_mode_promotion_plays_only_a_promotion_once_on_the_cpu_prior(self):
        # the daemon's own prior is the server (the card); the human comparison keeps the CPU prior the weekly readings were taken with
        d = self.daemon(human_bar="promotion")
        d.a.prior = "server"
        with mock.patch.object(gl, "run_gate", lambda *a, **k: self.verdict(2, "reject")), mock.patch.object(gl, "play_tasks", self.fake_play()), \
                mock.patch.dict(gl.HUMAN_BAR, {"draws": 1}):
            d.gate("c0190", self.cand_dir)
        self.assertEqual(self.calls, [])  # a stage-2 reject: nothing played, no human field
        self.assertNotIn("human", self.recorded()[0])
        with mock.patch.object(gl, "run_gate", lambda *a, **k: self.verdict(1, "continue")), mock.patch.object(gl, "play_tasks", self.fake_play()):
            d.gate("c0190", self.cand_dir)
        self.assertEqual(self.calls, [])
        with mock.patch.object(gl, "run_gate", lambda *a, **k: self.verdict(2, "promote")), mock.patch.object(gl, "play_tasks", self.fake_play()), \
                mock.patch.dict(gl.HUMAN_BAR, {"draws": 1}):
            d.gate("c0190", self.cand_dir)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual((self.calls[0]["prior"], self.calls[0]["what"]), ("cpu", "human-bar-c0190"))
        state, last = self.recorded()
        self.assertEqual((state["verdict"], state["human"], last["human"]["candidate"]), ("promote", last["human"], "c0190"))  # the same reading in the state and the ledger
        self.assertIn("human bar: cand ", state["human"]["line"])
        self.assertIn("gate c0190: " + state["human"]["line"], self.log())  # in the log, and in the ledger line above

    def test_a_promotion_the_guard_refuses_plays_no_human_bar(self):
        d = self.daemon(human_bar="promotion")
        self.loop.state["promotions"].append({"version": "v0", "checkpoint": "c0001", "over": "v0", "t": gl.stamp()})
        with mock.patch.object(gl, "run_gate", lambda *a, **k: self.verdict(2, "promote", path="single")), mock.patch.object(gl, "play_tasks", self.fake_play()):
            d.gate("c0190", self.cand_dir)
        self.assertEqual(self.calls, [])
        state, last = self.recorded()
        self.assertEqual((state["verdict"], state["guard"] is not None, last["guarded_path"]), ("reject", True, "single"))
        self.assertEqual(json.loads((self.loop.root / "state.json").read_text())["incumbent"], "v0")

    def test_the_gates_play_is_the_exams_own_configuration(self):
        self.assertEqual((gl.HUMAN_BAR["config"], gl.HUMAN_BAR["draws"]), (fe.DEFAULT_CONFIG, fe.DRAWS))  # not the gate's --player-config

    def test_the_daemons_flag_is_promotion_by_default(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(gl, "Daemon") as daemon:
            daemon.return_value.run.return_value = 0
            gl.main(["run", "--root", tmp])
            gl.main(["run", "--root", tmp, "--human-bar", "off"])
            self.assertEqual([c.args[0].human_bar for c in daemon.call_args_list], ["promotion", "off"])

    def test_the_rule_and_decide_do_not_know_the_human_bar(self):
        import inspect  # noqa: PLC0415

        self.assertEqual(gl.PROMOTION_RULE_V2, {"name": "gym-loop-promote-v2", "fixed": "2026-11-01 (D43): stage 1's pair rotates; v1 fixed 2026-10-30",
                                                "starts": 200, "cat": "act1-boss", "stage1_seeds": 2, "stage1_rotates": True, "futility_points": 1.0,
                                                "seeds": 10, "alpha": 0.05, "l0_not_worse": True, "capped": "lost", "ns": "g"})
        for fn in (gl.decide, gl.paired, gl.mcnemar_one_sided, gl.stage1_js, gl.run_gate):
            self.assertNotIn("human", inspect.getsource(fn).lower(), fn.__name__)
        self.assertEqual(list(inspect.signature(gl.decide).parameters), ["cand", "inc", "stage", "rule", "pair", "pool"])
        rule = {**gl.PROMOTION_RULE_V2, "starts": 4, "stage1_seeds": 2, "seeds": 6}
        starts = ("S0", "S1", "S2", "S3")
        inc = [{**_row(s, j, i % 2 == 0), "arm": "v0"} for i, s in enumerate(starts) for j in range(6)]
        cand = [{**_row(s, j, i % 2 == 0 or j >= 2), "arm": "c"} for i, s in enumerate(starts) for j in range(6)]
        first = [gl.decide(cand, inc, 1, rule, pair=1), gl.decide(cand, inc, 2, rule)]
        self.write_games("c0190", self.rows("c0190", lambda i: True))  # human games present or not: the same answers
        self.write_games("v0", self.rows("v0", lambda i: False))
        self.assertEqual([gl.decide(cand, inc, 1, rule, pair=1), gl.decide(cand, inc, 2, rule)], first)
        self.assertEqual([x["verdict"] for x in first], ["continue", "promote"])
        self.assertNotIn("human", first[0])

    def test_the_ai_exams_readings_are_byte_for_byte_the_same_with_and_without_the_human_games(self):
        starts = fe.load_exam(self.exam)
        ai_files = [self.exam / n for n in ("starts.jsonl", "starts.index.json", "starts.lite.json", "meta.json")]
        rnd = random.Random(4)
        for label, p in (("v0", 0.5), ("c0190", 0.6)):
            ai = fx.games_for(starts, lambda s, p=p: p, rnd)
            self.write_games(label, [{**r, "config": "a3-v1", "engine": "eng"} for r in ai.values()])
        self.write_games("v0", fg.read_jsonl(self.exam / "games-v0.jsonl") + self.rows("v0", lambda i: i == 0))

        def readings():
            out = {}
            for what, fn, kw in (("report", fe.cmd_report, {"player": "c0190", "write_calibration": False, "boot": 30}),
                                 ("compare", fe.cmd_compare, {"candidate": "c0190", "champion": "v0", "no_play": True, "boot": 30})):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    self.assertEqual(fn(argparse.Namespace(exam_dir=str(self.exam), **kw)), 0)
                out[what] = buf.getvalue()
            return out

        before_files = [gl.sha256_file(f) for f in ai_files]
        before = readings()
        ai_lines = [x for x in (self.exam / "games-c0190.jsonl").read_text(encoding="utf-8").splitlines()]
        self.gate(self.verdict(2, "reject"))
        after = readings()
        self.assertEqual([gl.sha256_file(f) for f in ai_files], before_files)  # the AI exam's files did not move
        for what in before:
            self.assertTrue(after[what].startswith(before[what].rstrip("\n")), what)  # the exam's own text first and unchanged
            self.assertNotIn("human", before[what])
            self.assertIn("human stratum", after[what][len(before[what].rstrip("\n")):])
        self.assertEqual((self.exam / "games-c0190.jsonl").read_text(encoding="utf-8").splitlines()[: len(ai_lines)], ai_lines)  # appended only
        starts_after = {k: r for k, r in fe.index_games(fg.read_jsonl(self.exam / "games-c0190.jsonl")).items() if k[0] in starts}
        self.assertEqual(len(starts_after), 4 * len(starts))  # the human games sit beside them and are never counted in them


class HumanBarRealGameTests(unittest.TestCase):
    """The gate's human play is the exam's: real short arena fights, the fixture prior, the exam's own game steps in spawned processes."""

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
        cls.exam = cls.loop.root / "exam"
        cls.exam.mkdir()
        human = [{**s, "seed": f"RUN{i}", "run": f"RUN{i}", "kind": "hallway", "cell": "act1-hallway", "kind_cell": "act1-hallway",
                  "half": fe.half_of(s["id"]), "stratum": "human", "split": "train", "human_survived": True, "human_hp_lost": 12,
                  "human_l0_unit": 0.8, "human_turns": 4, "human_potions_used": 0}
                 for i, s in enumerate(got) if not s["id"].startswith("F18")]
        fe.write_stem_set(cls.exam, "starts-human", human)
        cls.agent = root / "agents" / "fixture"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_gates_play_of_the_stratum_is_the_exams_own_games(self):
        import gym_lockstep  # noqa: PLC0415

        with mock.patch.object(gl, "why_blocked", lambda *a, **k: None), mock.patch.object(gl, "hold_quiet", lambda on: None), \
                mock.patch.dict(gl.HUMAN_BAR, {"draws": 1}):
            reading = gl.human_bar(self.loop, "cfix", self.agent, "nobody", 1, 2, "cpu")
        rows = fg.read_jsonl(self.exam / "games-cfix.jsonl")
        self.assertEqual(len(rows), 3)
        self.assertEqual({(r["stratum"], r["config"], r["exam"], r["arm"], r["player"]) for r in rows},
                         {("human", "a3-v1", fe.EXAM_RULE, "cfix", "cfix")})
        self.assertTrue(all(len(r["engine"]) == 12 and r["hp_lost"] == r["start_hp"] - (r["end_hp"] if r["won"] else 0) for r in rows))
        self.assertEqual((reading["starts"], reading["inc_games"], reading["cand_games"], reading["incumbent"]), (3, 0, 3, "nobody"))
        self.assertIn("(no human-stratum games for the incumbent nobody)", reading["line"])
        # the same games by the exam's own route (fight_exam's tasks and game steps, a lockstep run), move for move
        human = fe.load_human_exam(self.exam)
        tasks = fe.exam_tasks({}, "direct", "direct", "a3-v1", self.exam, set(), draws=1, human=human)
        init = ("core2", str(self.agent.parent), self.agent.name, str(self.loop.root))
        stopped, _ = gym_lockstep.run_lockstep(tasks, self.exam / "direct.jsonl", 1, 2, init,
                                               {"prior": "cpu", "game_steps": "fight_exam:exam_game_steps", "paths": [str(ROOT / "rl")]})
        self.assertIsNone(stopped)
        a, b = fe.index_games(rows), fe.index_games(fg.read_jsonl(self.exam / "direct.jsonl"))
        self.assertEqual(sorted(a), sorted(b))
        for key in a:
            for f in ("won", "end_hp", "turns", "decisions", "moves_sha", "L0", "hp_lost"):
                self.assertEqual(a[key][f], b[key][f], (key, f))


# ---------------------------------------------------------------------------------------------------------- the gate's pre-filter


class PrefilterTests(unittest.TestCase):
    """D49 item 3: who enters the gate. Checkpoints are folders with fake weights and a train.json; no games, no torch."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.loop = gl.Loop(Path(self.tmp.name) / "loop")
        self.addCleanup(self.tmp.cleanup)
        self._version("v2", b"weights of v2")
        self.loop.state["incumbent"] = "v2"
        self.cfg = {**gl.PREFILTER, "window": 1000}
        self.t0 = dt.datetime(2026, 11, 3, 6, 0, 0, tzinfo=dt.timezone.utc)

    def _version(self, name, blob):
        d = self.loop.players / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "combat_policy.pt").write_bytes(blob)
        self.loop.state["versions"][name] = {"dir": str(d), "n": int(name[1:])}

    def _ck(self, name, loss, ref, base="v2", split=None, identity=True):
        """A checkpoint as the trainer writes it: its held-out policy CE and the incumbent's, naming whose and on which split."""
        d = self.loop.checkpoints / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.json").write_text("{}", encoding="utf-8")
        (d / "combat_policy.pt").write_bytes(b"weights of " + name.encode())
        info = {"checkpoint": name, "val_rows": 20000, "val_policy_ce": loss, "base_val_policy_ce": ref}
        if identity:
            info.update(base_version=base, base_sha256=gl.sha256_file(self.loop.players / base / "combat_policy.pt"),
                        split=split or gl.val_split_id())
        (d / "train.json").write_text(json.dumps(info), encoding="utf-8")
        return d

    def _log(self):
        return (self.loop.root / "loop.log").read_text(encoding="utf-8") if (self.loop.root / "loop.log").exists() else ""

    def _scorer(self, calls, loss=1.0500, ref=1.0700):
        def score(loop, cand_dir, inc_dir, window):
            calls.append((cand_dir.name, inc_dir.name, window))
            return {"val_rows": 777, "val_policy_ce": loss, "base_val_policy_ce": ref, "sha256": gl.sha256_file(cand_dir / "combat_policy.pt"),
                    "base_sha256": gl.sha256_file(inc_dir / "combat_policy.pt"), "split": gl.val_split_id()}
        return score

    def _pick(self, now=None, scorer=None):
        return gl.prefilter_pick(self.loop, self.cfg, now or self.t0, scorer)

    def _verdict(self, cand, now, verdict="reject"):
        gl.prefilter_note_verdict(self.loop, cand, {"verdict": verdict}, now)
        self.loop.state["gated"][cand] = {"verdict": verdict}

    def test_a_checkpoint_not_below_the_incumbents_loss_is_skipped_and_the_log_gives_both_figures(self):
        self._ck("c0001", 1.0800, 1.0700)  # worse
        self._ck("c0002", 1.0700, 1.0700)  # equal: not below
        self.assertIsNone(self._pick())
        self.assertEqual({k: self.loop.state["gated"][k]["verdict"] for k in ("c0001", "c0002")},
                         {"c0001": "skipped", "c0002": "skipped"})
        log = self._log()
        self.assertIn("prefilter: c0001 skipped: held-out policy CE 1.0800 against v2's 1.0700 on the same 20000 rows", log)
        self.assertIn("not below the incumbent's", log)
        self.assertIn("prefilter: c0002 skipped: held-out policy CE 1.0700 against v2's 1.0700", log)
        self.assertEqual(self.loop.state["prefilter"]["window"], ["c0001", "c0002"])
        before = self._log()
        self.assertIsNone(self._pick())  # decided once: no second line, nothing new to gate
        self.assertEqual(self._log(), before)

    def test_a_better_one_is_gated_and_the_newest_passing_one_wins(self):
        self._ck("c0001", 1.0600, 1.0700)  # gap +0.0100
        self._ck("c0002", 1.0650, 1.0700)  # gap +0.0050
        self.assertEqual(self._pick()[0], "c0002")  # both pass: the newest is gated (the gate supersedes the older)
        self.assertIn("prefilter: c0001 gated: held-out policy CE 1.0600 against v2's 1.0700", self._log())
        self.assertIn("prefilter: c0002 gated", self._log())
        self.assertEqual(self.loop.state["prefilter"]["window"], [])

    def test_a_checkpoint_no_better_than_one_the_gate_rejected_is_skipped_unless_the_mode_is_incumbent(self):
        self._ck("c0001", 1.0600, 1.0700)
        self.assertEqual(self._pick()[0], "c0001")
        self._verdict("c0001", self.t0 + dt.timedelta(minutes=8))
        self.assertEqual(self.loop.state["prefilter"]["best"]["cand"], "c0001")
        self._ck("c0002", 1.0610, 1.0700)  # gap +0.0090 against the rejected one's +0.0100
        self.assertIsNone(self._pick(self.t0 + dt.timedelta(minutes=40)))
        self.assertIn("c0002 skipped", self._log())
        self.assertIn("not past c0001's gap of +0.0100, which the gate rejected", self._log())
        self._ck("c0003", 1.0580, 1.0700)  # +0.0120: a new best
        self.assertEqual(self._pick(self.t0 + dt.timedelta(minutes=70))[0], "c0003")
        self.assertIn("c0003 gated: held-out policy CE 1.0580 against v2's 1.0700 on the same 20000 rows (gap +0.0120): "
                      "below the incumbent's and past c0001's gap of +0.0100", self._log())
        # mode incumbent: only the incumbent's figure counts
        self._verdict("c0003", self.t0 + dt.timedelta(minutes=80))
        self.cfg = {**self.cfg, "mode": "incumbent"}
        self._ck("c0004", 1.0640, 1.0700)  # +0.0060
        self.assertEqual(self._pick(self.t0 + dt.timedelta(minutes=100))[0], "c0004")
        # and a margin the gap must exceed
        self._verdict("c0004", self.t0 + dt.timedelta(minutes=110))
        self.cfg = {**self.cfg, "margin": 0.0070}
        self._ck("c0005", 1.0640, 1.0700)
        self.assertIsNone(self._pick(self.t0 + dt.timedelta(minutes=130)))
        self.assertIn("not below the incumbent's by more than 0.0070", self._log())

    def test_the_floor_gates_the_best_skipped_checkpoint_after_six_hours_and_only_then(self):
        self._ck("c0001", 1.0600, 1.0700)
        self.assertEqual(self._pick()[0], "c0001")
        self._verdict("c0001", self.t0)  # the last gate: the floor counts from here
        for i, gap in enumerate((0.0080, 0.0095, 0.0090, -0.0020), start=2):
            self._ck(f"c{i:04d}", 1.0700 - gap, 1.0700)
        self.assertIsNone(self._pick(self.t0 + dt.timedelta(hours=5, minutes=59)))
        self.assertEqual(self.loop.state["prefilter"]["window"], ["c0002", "c0003", "c0004", "c0005"])
        got = self._pick(self.t0 + dt.timedelta(hours=6, minutes=1))
        self.assertEqual(got[0], "c0003")  # the best gap of the window, not the newest
        self.assertEqual(got[1], self.loop.checkpoints / "c0003")
        self.assertRegex(self._log(), r"prefilter: floor: 6\.0 h since the last gate, gating c0003, the best of 4 skipped "
                                      r"\(held-out policy CE 1\.0605 against v2's 1\.0700, gap \+0\.0095\)")
        self.assertEqual(self._pick(self.t0 + dt.timedelta(hours=6, minutes=2))[0], "c0003")  # a stopped gate comes back
        self._verdict("c0003", self.t0 + dt.timedelta(hours=6, minutes=10))
        self.assertEqual(self.loop.state["prefilter"]["window"], [])  # the window is over
        self.assertEqual(self.loop.state["prefilter"]["best"]["cand"], "c0001")  # the bar is the best gap rejected so far
        self.assertIsNone(self._pick(self.t0 + dt.timedelta(hours=7)))  # nothing in the window: nothing to gate
        self._ck("c0006", 1.0705, 1.0700)
        self.assertIsNone(self._pick(self.t0 + dt.timedelta(hours=7, minutes=1)))
        self.assertIsNone(self._pick(self.t0 + dt.timedelta(hours=12, minutes=9)))  # six hours counted from the last verdict
        self.assertEqual(self._pick(self.t0 + dt.timedelta(hours=12, minutes=11))[0], "c0006")

    def test_the_reference_is_recomputed_when_the_incumbent_or_the_split_changes(self):
        calls: list = []
        score = self._scorer(calls)
        d = self._ck("c0001", 1.0600, 1.0700)
        fig = gl.checkpoint_figures(self.loop, d, 1000, score)
        self.assertEqual((calls, fig["source"], round(fig["gap"], 4)), ([], "train.json", 0.0100))  # the trainer's, on disk
        # a checkpoint from before the figures named whose they were: scored once, stored, read from the store after that
        old = self._ck("c0002", 1.0600, 1.0700, identity=False)
        fig = gl.checkpoint_figures(self.loop, old, 1000, score)
        self.assertEqual((calls, fig["source"], fig["rows"], round(fig["gap"], 4)), ([("c0002", "v2", 1000)], "scored", 777, 0.0200))
        stored = json.loads((old / "prefilter.json").read_text())
        self.assertEqual((stored["base_version"], stored["split"]), ("v2", gl.val_split_id()))
        self.assertEqual(gl.checkpoint_figures(self.loop, old, 1000, score)["source"], "prefilter.json")
        self.assertEqual(len(calls), 1)
        # a new incumbent: the trainer's figures were against the old one and the stored ones too: scored again for both
        self._version("v3", b"weights of v3")
        self.loop.state["incumbent"] = "v3"
        for name in ("c0001", "c0002"):
            self.assertEqual(gl.checkpoint_figures(self.loop, self.loop.checkpoints / name, 1000, score)["source"], "scored")
        self.assertEqual([c[:2] for c in calls], [("c0002", "v2"), ("c0001", "v3"), ("c0002", "v3")])
        self.assertEqual(json.loads((self.loop.checkpoints / "c0002" / "prefilter.json").read_text())["base_version"], "v3")
        self.assertEqual(gl.checkpoint_figures(self.loop, self.loop.checkpoints / "c0002", 1000, score)["source"], "prefilter.json")
        self.assertEqual(len(calls), 3)
        # the same name with other weights is another incumbent
        (self.loop.players / "v3" / "combat_policy.pt").write_bytes(b"v3 retrained")
        self.assertEqual(gl.checkpoint_figures(self.loop, self.loop.checkpoints / "c0002", 1000, score)["source"], "scored")
        self.assertEqual(len(calls), 4)
        # a new split: the held-out rows are not the same rows
        with mock.patch.object(gl, "VAL_SHARE", 0.2):
            self.assertEqual(gl.val_split_id(), "gym-loop-val:0.2")
            fig = gl.checkpoint_figures(self.loop, self.loop.checkpoints / "c0002", 1000, score)
            self.assertEqual((fig["source"], len(calls)), ("scored", 5))
            self.assertEqual(json.loads((self.loop.checkpoints / "c0002" / "prefilter.json").read_text())["split"], "gym-loop-val:0.2")

    def test_a_new_incumbent_starts_a_new_record_and_the_floor_counts_from_its_promotion_gate(self):
        self._ck("c0001", 1.0600, 1.0700)
        self.assertEqual(self._pick()[0], "c0001")
        self._verdict("c0001", self.t0)
        self.assertEqual(self.loop.state["prefilter"]["best"]["cand"], "c0001")
        (self.loop.gates / "verdicts.jsonl").write_text(json.dumps({"t": "2026-11-03T07:30:00Z", "verdict": "promote"}) + "\n", encoding="utf-8")
        self._version("v3", b"weights of v3")
        self.loop.state["incumbent"] = "v3"
        pf = gl.prefilter_state(self.loop, self.t0 + dt.timedelta(hours=2))
        self.assertEqual((pf["incumbent"], pf["best"], pf["window"], pf["figures"], pf["last_gate"]),
                         ("v3", None, [], {}, "2026-11-03T07:30:00Z"))

    def test_figures_that_cannot_be_had_send_the_checkpoint_to_the_gate(self):
        def broken(loop, cand_dir, inc_dir, window):
            raise RuntimeError("score rc=1: no held-out rows in the buffer")
        self._ck("c0001", 1.0600, 1.0700, identity=False)
        got = self._pick(scorer=broken)
        self.assertEqual(got[0], "c0001")
        self.assertIn("prefilter: c0001 gated: no held-out figures, the gate decides (RuntimeError: score rc=1", self._log())
        bad = self._scorer([])  # a scorer that scored another incumbent's weights is refused as well
        self._ck("c0002", 1.0600, 1.0700, identity=False)
        with mock.patch.object(gl, "policy_sha", lambda d: "x" if Path(d).parent.name == "players" else "y"):
            with self.assertRaises(RuntimeError):
                gl.checkpoint_figures(self.loop, self.loop.checkpoints / "c0002", 1000, bad)

    def test_a_status_write_a_reader_blocks_is_retried_and_never_ends_the_daemon(self):
        # 2026-11-03 07:18Z: os.replace of the status file was refused while the lab's runner had it open; the daemon died (rc 1)
        target = Path(self.tmp.name) / "status"
        real = os.replace
        refused = {"n": 3}

        def replace(src, dst):
            if refused["n"] > 0:
                refused["n"] -= 1
                raise PermissionError(5, "Access is denied")
            return real(src, dst)

        with mock.patch.object(gl.os, "replace", replace), mock.patch.object(gl.time, "sleep", lambda s: None):
            gl.write_json(target, {"phase": "waiting"})
        self.assertEqual(json.loads(target.read_text()), {"phase": "waiting"})
        self.assertEqual(refused["n"], 0)  # refused three times, written on the fourth
        self.assertEqual([p.name for p in target.parent.glob("status*")], ["status"])  # no temporary file left
        refused["n"] = 10**9  # a reader that never lets go: the error is raised after the patience, the temporary file removed
        with mock.patch.object(gl.os, "replace", replace), mock.patch.object(gl.time, "sleep", lambda s: None), \
                self.assertRaises(PermissionError):
            gl.write_json(target, {"phase": "x"}, patience_s=0.05)
        self.assertEqual(json.loads(target.read_text()), {"phase": "waiting"})
        self.assertEqual([p.name for p in target.parent.glob("status*")], ["status"])
        d = gl.Daemon.__new__(gl.Daemon)  # and the daemon's status line goes on without the file
        d.loop, d.commit, d.server_pid, d.trainer_pid = self.loop, "abc", None, None
        with mock.patch.object(gl, "STATUS_DIR", target.parent), mock.patch.object(gl, "write_json", side_effect=PermissionError(5, "x")):
            d.status("waiting", why="lock")
        self.assertEqual(d.phase, "waiting")
        self.assertIn("status: not written (PermissionError", self._log())

    def test_the_daemon_asks_the_filter_unless_it_is_off(self):
        self._ck("c0001", 1.0800, 1.0700)
        a = argparse.Namespace(prefilter="best", prefilter_margin=0.0, prefilter_floor_h=6.0, train_window=1000)
        d = gl.Daemon.__new__(gl.Daemon)
        d.a, d.loop = a, self.loop
        self.assertIsNone(gl.Daemon.next_gate(d))  # skipped
        self.assertEqual(self.loop.state["gated"]["c0001"]["verdict"], "skipped")
        self._ck("c0002", 1.0800, 1.0700)
        d.a = argparse.Namespace(**{**vars(a), "prefilter": "off"})
        self.assertEqual(gl.Daemon.next_gate(d)[0], "c0002")  # the old rule: the newest without a verdict
        cfg = gl.prefilter_cfg(argparse.Namespace(prefilter="incumbent", prefilter_margin=0.002, prefilter_floor_h=3.0, train_window=5))
        self.assertEqual((cfg["mode"], cfg["margin"], cfg["floor_h"], cfg["window"]), ("incumbent", 0.002, 3.0, 5))
        self.assertEqual(gl.prefilter_cfg()["floor_h"], 6.0)
        self.assertEqual(gl.PROMOTION_RULE["name"], "gym-loop-promote-v3")  # the pre-filter does not touch the promotion rule


if __name__ == "__main__":
    unittest.main()
