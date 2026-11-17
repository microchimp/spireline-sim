"""The fight loop's daemon on the plain runs' fight rows, with its own generation off (D65 addendum 4, lane T; `gym_loop.py run
--procs 0 --gate-procs N --trainer-extra-rows-root PATH --trainer-extra-window N --trainer-policy-min-grade N`).

The pinned facts: the three trainer flags reach the trainer's command line (and again on a restart of the trainer) and, absent, leave it
what it was; `--procs 0` is legal and the daemon then plays no segment, builds no start set, weights or teacher starts and runs no
retention, while the trainer, the pre-filter and the gate go on, waiting (timed as `idle`) when no checkpoint is offered; the gate's and the
human bar's games play with `--gate-procs` (default: `--procs`); a `--procs 0` that leaves the gate no process is refused; what the trainer
reads is in `state.json` and the log; the trainer trains from the plain rows alone when the loop's own shards are empty; the pre-filter
still scores a checkpoint of the plain split on the loop's own held-out rows (the trainer's `score` takes no plain root). Synthetic arrays
and a tiny network on the CPU; no game, no card, no lock file of the lab.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gym_clock as gclock  # noqa: E402
import gym_loop as gl  # noqa: E402
import gym_loop_train as tr  # noqa: E402
from test_fight_rows import write_plain_store  # noqa: E402
from test_gym_loop_train import fake_shard, make_base, namespace  # noqa: E402
from tools.lab import winsys  # noqa: E402


def parse(*argv: str):
    return gl.build_parser().parse_args(list(argv))


def run_args(root: Path, *extra: str):
    return parse("run", "--root", str(root), "--yield-s", "0", "--idle-s", "0", "--prefilter", "off", "--board", "off",
                 "--plateau-alert", "off", "--human-bar", "off", *extra)


class FlagTests(unittest.TestCase):
    def test_the_flags_default_to_what_the_daemon_always_did(self):
        a = parse("run", "--root", "R")
        self.assertEqual((a.procs, a.gate_procs), (6, None))
        self.assertEqual(gl.gate_procs(a), 6)
        self.assertTrue(gl.generating(a))
        self.assertEqual((a.trainer_extra_rows_root, a.trainer_extra_window, a.trainer_policy_min_grade), ([], None, None))

    def test_gate_procs_is_the_gates_own_count_and_procs_0_is_generation_off(self):
        a = parse("run", "--root", "R", "--procs", "0", "--gate-procs", "4")
        self.assertEqual((a.procs, gl.gate_procs(a), gl.generating(a)), (0, 4, False))
        b = parse("run", "--root", "R", "--procs", "3", "--gate-procs", "5")
        self.assertEqual((gl.gate_procs(b), gl.generating(b)), (5, True))
        self.assertEqual(gl.gate_procs(parse("gate", "--root", "R", "--candidate", "C", "--procs", "2")), 2)  # a one-shot gate: as it was
        self.assertEqual(gl.gate_procs(parse("gate", "--root", "R", "--candidate", "C", "--gate-procs", "3")), 3)

    def test_a_trainer_flag_is_repeatable_and_a_namespace_without_the_new_fields_reads_as_before(self):
        a = parse("run", "--root", "R", "--trainer-extra-rows-root", "S1", "--trainer-extra-rows-root", "S2", "--trainer-extra-window", "150000",
                  "--trainer-policy-min-grade", "4")
        self.assertEqual((a.trainer_extra_rows_root, a.trainer_extra_window, a.trainer_policy_min_grade), (["S1", "S2"], 150000, 4))
        import argparse  # noqa: PLC0415

        old = argparse.Namespace(procs=3)  # the daemon tests build their arguments by hand
        self.assertEqual(gl.gate_procs(old), 3)
        self.assertTrue(gl.generating(old))

    def refused(self, *argv: str) -> str:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), mock.patch.object(gl, "Daemon") as daemon, self.assertRaises(SystemExit) as cm:
            gl.main(list(argv))
        self.assertEqual(cm.exception.code, 2)
        daemon.assert_not_called()
        return err.getvalue()

    def test_generation_off_needs_a_gate_pool_or_no_gate_and_the_start_builds_need_processes(self):
        self.assertIn("--gate-procs", self.refused("run", "--root", "R", "--procs", "0"))
        self.assertIn("--gate-procs", self.refused("run", "--root", "R", "--procs", "0", "--gate-procs", "0"))
        self.assertIn("--procs", self.refused("run", "--root", "R", "--procs", "-1"))
        self.assertIn("--procs", self.refused("starts", "--root", "R", "--procs", "0"))
        self.assertIn("--procs", self.refused("weights", "--root", "R", "--procs", "0"))
        self.assertIn("--gate-procs", self.refused("gate", "--root", "R", "--candidate", "C", "--procs", "0"))
        with mock.patch.object(gl, "Daemon") as daemon:  # legal: no generation and a gate pool, or no gate at all
            daemon.return_value.run.return_value = 0
            self.assertEqual(gl.main(["run", "--root", "R", "--procs", "0", "--gate-procs", "4"]), 0)
            self.assertEqual(gl.main(["run", "--root", "R", "--procs", "0", "--no-gate"]), 0)
            self.assertEqual(daemon.call_args.args[0].procs, 0)

    def test_a_trainer_window_or_grade_without_a_plain_root_is_refused(self):
        self.assertIn("--trainer-extra-rows-root", self.refused("run", "--root", "R", "--trainer-extra-window", "5"))
        self.assertIn("--trainer-extra-rows-root", self.refused("run", "--root", "R", "--trainer-policy-min-grade", "4"))


class TrainerArgvTests(unittest.TestCase):
    OLD = ["--root", "ROOT", "--base", "BASE", "--parent-pid", "77", "--every-min", "30.0", "--max-epochs", "4.0", "--window", "300000"]

    def test_without_the_flags_the_trainers_command_line_is_what_it_was(self):
        a = parse("run", "--root", "R")
        argv = gl.trainer_argv(a, Path("ROOT"), Path("BASE"), 77)
        self.assertEqual(argv[0], gl.GPU_PYTHON)
        self.assertEqual(Path(argv[1]), gl.ROOT / "rl" / "gym_loop_train.py")
        self.assertEqual(argv[2:], self.OLD)
        self.assertEqual(gl.trainer_argv(parse("run", "--root", "R", "--reset-every", "24"), Path("ROOT"), Path("BASE"), 77)[2:],
                         self.OLD + ["--reset-every", "24"])

    def test_the_flags_reach_the_trainer_and_its_own_parser_reads_them(self):
        a = parse("run", "--root", "R", "--trainer-extra-rows-root", "D:/store/a", "--trainer-extra-rows-root", "D:/store/b",
                  "--trainer-extra-window", "150000", "--trainer-policy-min-grade", "4")
        argv = gl.trainer_argv(a, Path("ROOT"), Path("BASE"), 77)
        self.assertEqual(argv[2:], self.OLD + ["--extra-rows-root", "D:/store/a", "--extra-rows-root", "D:/store/b", "--extra-window", "150000",
                                              "--policy-min-grade", "4"])
        got = tr.trainer_parser().parse_args(argv[2:])  # after the interpreter and the script: the trainer's own flags
        self.assertEqual((got.extra_rows_root, got.extra_window, got.policy_min_grade), (["D:/store/a", "D:/store/b"], 150000, 4))
        self.assertEqual((got.max_epochs, got.window), (4.0, 300000))
        only = gl.trainer_argv(parse("run", "--root", "R", "--trainer-extra-rows-root", "D:/store/a"), Path("ROOT"), Path("BASE"), 77)
        self.assertEqual(only[2:], self.OLD + ["--extra-rows-root", "D:/store/a"])  # a flag not given is not passed: the trainer's defaults stand

    def test_the_trainers_score_command_takes_no_plain_root(self):
        # the pre-filter scores on the loop's own held-out rows: nobody adds a plain root to `score` unnoticed
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tr.main(["score", "--root", "R", "--candidate", "C", "--base", "B", "--extra-rows-root", "S"])

    def started(self, tmp: Path, a, polls=(None,)):
        """Start the daemon's trainer with the popen faked; returns the argv it was given."""
        d = gl.Daemon(a)
        d.loop = gl.Loop(tmp / "loop")
        d.job = object()
        calls = []

        def popen(argv, job, **kw):
            p = mock.Mock()
            p.pid = 4000 + len(calls)
            p.poll.return_value = polls[min(len(calls), len(polls) - 1)]
            p.returncode = 1
            calls.append(list(argv))
            return p

        with mock.patch.object(winsys, "suspended_popen", popen):
            d.start_trainer()
            return d, calls

    def test_the_daemon_starts_the_trainer_with_the_flags_and_again_after_it_ended(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a = run_args(tmp / "loop", "--trainer-extra-rows-root", "D:/store", "--trainer-extra-window", "150000",
                         "--trainer-policy-min-grade", "4", "--max-epochs", "8")
            d, calls = self.started(tmp, a)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][calls[0].index("--extra-rows-root"):], ["--extra-rows-root", "D:/store", "--extra-window", "150000",
                                                                              "--policy-min-grade", "4"])
            self.assertEqual(calls[0][calls[0].index("--max-epochs") + 1], "8.0")
            # it ended on its own: the next call (ten minutes on) starts it again with the same flags
            d.trainer.poll.return_value = 1
            d._trainer_t -= 700
            with mock.patch.object(winsys, "suspended_popen", lambda argv, job, **kw: (calls.append(list(argv)), mock.Mock(pid=4001))[1]):
                d.start_trainer()
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1], calls[0])
            self.assertIn("trainer: ended rc=1", (d.loop.root / "loop.log").read_text(encoding="utf-8"))

    def test_without_the_flags_the_daemons_trainer_has_none_of_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, calls = self.started(tmp, run_args(tmp / "loop"))
            for flag in ("--extra-rows-root", "--extra-window", "--policy-min-grade"):
                self.assertNotIn(flag, calls[0])


class RunRecordTests(unittest.TestCase):
    def test_the_state_and_the_log_say_what_the_trainer_reads_and_how_many_processes_play(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = run_args(Path(tmp) / "loop", "--procs", "0", "--gate-procs", "4", "--trainer-extra-rows-root", "D:/store",
                         "--trainer-extra-window", "150000", "--max-epochs", "8")
            d = gl.Daemon(a)
            rec = d.run_record()
            self.assertEqual((rec["generation"], rec["procs"], rec["gate_procs"]), ("off", 0, 4))
            self.assertEqual(rec["trainer"], {"extra_rows_root": ["D:/store"], "extra_window": 150000, "policy_min_grade": None,
                                              "window": 300000, "max_epochs": 8.0, "checkpoint_min": 30.0})
            line = d.run_line()
            self.assertIn("generation off", line)
            self.assertIn("gate 4 processes", line)
            self.assertIn("D:/store", line)
            on = gl.Daemon(run_args(Path(tmp) / "loop2")).run_record()
            self.assertEqual((on["generation"], on["procs"], on["gate_procs"], on["trainer"]["extra_rows_root"]), ("on", 6, 6, []))


class Harness:
    """`Daemon.run` with everything that plays a game, takes a lock of the lab or reads the lab's folders faked; the steps' calls are counted."""

    def __init__(self, tmp: Path, *extra: str) -> None:
        self.tmp = tmp
        self.a = run_args(tmp / "loop", *extra)
        self.d = gl.Daemon(self.a)
        self.calls: list[str] = []
        self.patches: list = []

    def log(self) -> str:
        p = self.d.loop.root / "loop.log"
        return p.read_text(encoding="utf-8") if p.is_file() else ""

    def __enter__(self):
        fake_job = mock.MagicMock()
        for p in (mock.patch.object(gl, "STATUS_DIR", self.tmp / "status"), mock.patch.object(gl, "hold_quiet", lambda on: None),
                  mock.patch.object(gl, "drop_gpu", lambda tag: None), mock.patch.object(gl, "GpuWatch", mock.MagicMock()),
                  mock.patch.object(gl, "why_blocked", lambda *a, **k: None),
                  mock.patch.object(winsys, "acquire_mutex", lambda name: True), mock.patch.object(winsys, "set_below_normal", lambda: None),
                  mock.patch.object(winsys, "JobObject", lambda **kw: fake_job)):
            p.start()
            self.patches.append(p)
        self.d.start_trainer = lambda: self.calls.append("trainer")
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()

    def count(self, step: str) -> int:
        return self.calls.count(step)


class GenerationOffTests(unittest.TestCase):
    def forbid(self, h: Harness) -> list:
        """Every step that exists for the generation: calling one is a failure."""
        def boom(name):
            def f(*a, **k):
                raise AssertionError(f"{name} ran with generation off")
            return f

        ps = [mock.patch.object(gl, name, boom(name)) for name in ("build_starts", "kind_weights", "starts_stale", "weights_stale", "sample_tasks",
                                                                    "nights_available", "retain")]
        ps.append(mock.patch.object(gl.Daemon, "segment", boom("segment")))
        ps.append(mock.patch.object(gl.Daemon, "_segment", boom("_segment")))
        h.d.teacher_step = boom("teacher_step")
        for p in ps:
            p.start()
        return ps

    def test_with_procs_0_the_daemon_idles_gates_and_trains_and_never_generates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with Harness(tmp, "--procs", "0", "--gate-procs", "4", "--teacher-store", "D:/no/store") as h:
                old = h.d.loop.shards / "s000001-1000.npz"
                old.write_bytes(b"an old shard of the loop's own")
                os.utime(old, (1_000_000_000.0, 1_000_000_000.0))  # decades past every retention age
                ps = self.forbid(h)
                offers = iter([None, None, ("c0001", h.d.loop.checkpoints / "c0001"), None, None])

                def next_gate():
                    h.calls.append("next_gate")
                    return next(offers)

                def gate(cand, cand_dir):
                    h.calls.append(f"gate:{cand}")

                real_idle = gl.Daemon.idle

                def idle():
                    h.calls.append("idle")
                    real_idle(h.d)
                    if h.count("idle") >= 3:
                        (h.d.loop.root / "STOP").write_text("stop\n", encoding="utf-8")

                h.d.next_gate, h.d.gate, h.d.idle = next_gate, gate, idle
                try:
                    rc = h.d.run()
                finally:
                    for p in ps:
                        p.stop()
                self.assertEqual(rc, 0)
                self.assertEqual([c for c in h.calls if c != "trainer"],
                                 ["next_gate", "idle", "next_gate", "idle", "next_gate", "gate:c0001", "next_gate", "idle"])
                self.assertGreaterEqual(h.count("trainer"), 4)  # asked for at every turn of the loop, as before
                self.assertTrue(old.exists())  # no retention runs: the loop's own shards stay as the older rows
                self.assertFalse((h.d.loop.games / "segments.jsonl").exists())
                self.assertEqual(h.d.loop.state["segment"], 0)
                log = h.log()
                self.assertIn("generation off", log)
                self.assertIn("gate 4 processes", log)
                self.assertIn("teacher starts: skipped", log)  # the store is named and left alone: its starts feed the generation only
                self.assertNotIn("segment 1", log)
                state = json.loads((h.d.loop.root / "state.json").read_text(encoding="utf-8"))
                self.assertEqual((state["run"]["generation"], state["run"]["gate_procs"]), ("off", 4))
                status = json.loads((h.tmp / "status" / "status").read_text(encoding="utf-8"))
                self.assertEqual((status["phase"], status["procs"], status["gate_procs"]), ("stopped", 0, 4))

    def test_the_idle_wait_is_timed_not_logged_and_ends_at_a_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with Harness(tmp, "--procs", "0", "--gate-procs", "2") as h:
                wall = [dt.datetime(2026, 11, 18, 3, 0, 0, tzinfo=dt.timezone.utc)]
                led = gclock.TimeLedger(h.d.loop.games / "clock.jsonl", h.d.loop.log, now=lambda: wall[0])
                h.d.clock, h.d.loop.clock = led, led
                h.d.a.idle_s = 600.0
                now = [1000.0]

                def sleep(s):
                    now[0] += s
                    wall[0] += dt.timedelta(seconds=s)
                    if now[0] > 1100.0:
                        (h.d.loop.root / "STOP").write_text("stop\n", encoding="utf-8")

                statuses = []
                h.d.status = lambda phase, **k: statuses.append(phase)
                with mock.patch.object(gl.time, "time", lambda: now[0]), mock.patch.object(gl.time, "sleep", sleep):
                    h.d.idle()
                self.assertLess(now[0], 1200.0)  # the stop cut the ten minutes short
                self.assertEqual(statuses, ["idle"])
                self.assertNotIn("wait start: idle", h.log())  # timed, not logged line by line (as the yield)
                self.assertIsNone(led.cur)
                self.assertEqual(led.totals(wall[0].date())["wait"], {"idle": 105.0})  # a timed wait of its own, 21 looks of 5 seconds
                self.assertEqual(led.status_fields()["waits_today"], {"idle": 105.0})

    def test_idle_keeps_the_newest_checkpoints_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Harness(Path(tmp), "--procs", "0", "--gate-procs", "2") as h:
                for n in range(1, 61):
                    (h.d.loop.checkpoints / f"c{n:04d}").mkdir()
                h.d.status = lambda *a, **k: None
                h.d.idle()
                self.assertEqual(len(list(h.d.loop.checkpoints.glob("c[0-9][0-9][0-9][0-9]"))), 48)
                self.assertTrue((h.d.loop.checkpoints / "c0060").exists())
                self.assertFalse((h.d.loop.checkpoints / "c0012").exists())

    def test_with_generation_on_the_loop_is_what_it_was(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with Harness(tmp, "--procs", "3") as h:
                steps = []
                for name, ret in (("starts_stale", None), ("weights_stale", None), ("nights_available", ())):
                    p = mock.patch.object(gl, name, lambda *a, _n=name, _r=ret, **k: (steps.append(_n), _r)[1])
                    p.start()
                    h.patches.append(p)

                def segment():
                    steps.append("segment")
                    (h.d.loop.root / "STOP").write_text("stop\n", encoding="utf-8")

                h.d.teacher_step = lambda: steps.append("teacher_step") or False
                h.d.next_gate = lambda: steps.append("next_gate")
                h.d.segment = segment
                h.d.idle = lambda: steps.append("idle")
                with mock.patch.object(gl.pmix, "config", lambda starts: {"human_share": {}, "regret": {}, "preloss": {"max": 1}, "teacher_share": {}}):
                    h.d.run()
                self.assertEqual(steps, ["nights_available", "starts_stale", "teacher_step", "weights_stale", "next_gate", "segment"])
                log = h.log()
                self.assertIn("3 processes K=2", log)
                self.assertNotIn("generation off", log)
                self.assertIn("practice mix", log)

    def test_the_gate_and_the_human_bar_play_with_the_gate_procs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Harness(Path(tmp), "--procs", "0", "--gate-procs", "4", "--human-bar", "on") as h:
                d = h.d
                d.status = lambda *a, **k: None
                seen = {}
                verdict = {"candidate": "c0001", "incumbent": "v0", "verdict": "reject", "stage": 2, "why": "x", "cand_file": "a", "inc_file": "b"}

                def run_gate(loop, cand, cand_dir, config, procs, k, prior, daemon=None, judge=None):
                    seen["gate"] = procs
                    return dict(verdict)

                def human_bar(loop, cand, cand_dir, inc, procs, k, prior, daemon=None):
                    seen["human"] = procs
                    return {"line": "human bar"}

                with mock.patch.object(gl, "run_gate", run_gate), mock.patch.object(gl, "human_bar", human_bar):
                    d.gate("c0001", d.loop.checkpoints / "c0001")
                self.assertEqual(seen, {"gate": 4, "human": 4})
            with Harness(Path(tmp) / "b", "--procs", "3", "--human-bar", "on") as h:  # no --gate-procs: the generation's count, as before
                d = h.d
                d.status = lambda *a, **k: None
                seen = {}
                with mock.patch.object(gl, "run_gate", run_gate), mock.patch.object(gl, "human_bar", human_bar):
                    d.gate("c0001", d.loop.checkpoints / "c0001")
                self.assertEqual(seen, {"gate": 3, "human": 3})

    def test_a_one_shot_gate_command_uses_the_gate_procs(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(gl, "run_gate", return_value={"verdict": "x"}) as rg, \
                contextlib.redirect_stdout(io.StringIO()):
            gl.main(["gate", "--root", tmp, "--candidate", str(Path(tmp) / "c0001"), "--procs", "0", "--gate-procs", "3"])
            self.assertEqual(rg.call_args.args[4], 3)
            gl.main(["gate", "--root", tmp, "--candidate", str(Path(tmp) / "c0001"), "--procs", "2"])
            self.assertEqual(rg.call_args.args[4], 2)


class BufferWithoutGenerationTests(unittest.TestCase):
    """The trainer with the loop's own shards empty (or old): its freshness budget comes from the plain rows alone, and the pre-filter still
    scores on the loop's own held-out rows."""

    def test_the_trainer_trains_from_the_plain_rows_alone_and_names_the_plain_split(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"GYM_LOOP_NO_GPU": "1"}):
            tmp = Path(tmp)
            root = tmp / "loop"
            (root / "buffer" / "shards").mkdir(parents=True)  # no shard of the loop's own
            make_base(tmp / "base")
            store = tmp / "store"
            write_plain_store(store, [("J1", tuple(f"S{i:09d}" for i in range(60))), ("J2", tuple(f"T{i:09d}" for i in range(60)))])
            t = tr.Trainer(namespace(root, tmp / "base", extra_rows_root=[str(store)], extra_window=150_000, max_epochs=3.0, batch=16))
            t.refresh()
            flat = t.buffer.arrays()
            self.assertEqual(set(flat["src"].tolist()), {1})
            train_rows = int((flat["val"] == 0).sum())
            self.assertGreater(train_rows, 16)
            self.assertEqual(t.st["budget"], 3.0 * train_rows)  # the budget is the plain rows' alone
            self.assertGreater(t.train_chunk("cpu"), 0)
            name = t.checkpoint()
            info = json.loads((root / "checkpoints" / name / "train.json").read_text(encoding="utf-8"))
            self.assertNotEqual(info["split"], gl.val_split_id())
            self.assertEqual(info["extra_rows"]["rows"], int(flat["lengths"].size))

    def _loop_with_incumbent(self, tmp: Path) -> gl.Loop:
        loop = gl.Loop(tmp / "loop")
        d = loop.players / "v2"
        d.mkdir(parents=True, exist_ok=True)
        (d / "combat_policy.pt").write_bytes(b"weights of v2")
        loop.state["versions"]["v2"] = {"dir": str(d), "n": 2}
        loop.state["incumbent"] = "v2"
        return loop

    def test_a_checkpoint_of_the_plain_split_is_scored_by_the_pre_filter_on_the_loops_own_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = self._loop_with_incumbent(Path(tmp))
            ck = loop.checkpoints / "c0001"
            ck.mkdir()
            (ck / "model.json").write_text("{}", encoding="utf-8")
            (ck / "combat_policy.pt").write_bytes(b"weights of c0001")
            info = {"checkpoint": "c0001", "val_rows": 9000, "val_policy_ce": 0.5, "base_val_policy_ce": 0.6, "base_version": "v2",
                    "base_sha256": gl.sha256_file(loop.players / "v2" / "combat_policy.pt"), "split": gl.val_split_id() + tr.PLAIN_SPLIT_SUFFIX}
            (ck / "train.json").write_text(json.dumps(info), encoding="utf-8")
            calls = []

            def scorer(lp, cand_dir, inc_dir, window):
                calls.append((cand_dir.name, inc_dir.name, window))
                return {"val_rows": 321, "val_policy_ce": 1.0, "base_val_policy_ce": 1.1, "sha256": gl.sha256_file(cand_dir / "combat_policy.pt"),
                        "base_sha256": gl.sha256_file(inc_dir / "combat_policy.pt"), "split": gl.val_split_id()}

            fig = gl.checkpoint_figures(loop, ck, 1000, scorer)
            self.assertEqual(calls, [("c0001", "v2", 1000)])  # the trainer's own figures, on another split, are not taken
            self.assertEqual((fig["source"], fig["rows"]), ("scored", 321))
            rec = json.loads((ck / "prefilter.json").read_text(encoding="utf-8"))
            self.assertEqual(rec["split"], gl.val_split_id())  # the gate's measure: the loop's own held-out tenth

    def test_the_pre_filters_scoring_command_names_no_plain_root_and_reads_the_loops_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = self._loop_with_incumbent(Path(tmp))
            done = mock.Mock(returncode=0, stdout=json.dumps({"val_rows": 1}) + "\n", stderr="")
            with mock.patch.object(gl.subprocess, "run", return_value=done) as run:
                gl.score_checkpoint(loop, loop.checkpoints / "c0001", loop.players / "v2", 300_000)
            argv = run.call_args.args[0]
            self.assertEqual(argv[argv.index("--root") + 1], str(loop.root))
            self.assertNotIn("--extra-rows-root", argv)

    def test_scoring_on_the_loops_shards_ignores_a_plain_store_beside_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root = tmp / "loop"
            (root / "buffer" / "shards").mkdir(parents=True)
            fake_shard(root / "buffer" / "shards" / "s000001-1000.npz", 100, 1)
            write_plain_store(tmp / "store", [("J1", tuple(f"S{i:09d}" for i in range(40)))])
            buf = tr.Buffer(root / "buffer" / "shards", 10_000)  # what `score_pair` builds
            self.assertEqual(buf.refresh(), 100)
            self.assertEqual(set(buf.arrays()["src"].tolist()), {0})


if __name__ == "__main__":
    unittest.main()
