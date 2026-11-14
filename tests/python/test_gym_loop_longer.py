"""The fight loop's longer training before the gate (D60 item 5, lane L; `gym_loop.py run --gate-every-min`).

The pinned facts: with the flag absent (or 0) the gate's choices are exactly what they were (a synthetic step log of 40 checkpoints,
the prefilter deciding each, gives one pinned list of gated checkpoints and verdicts); with `--gate-every-min 120` against the trainer's
30-minute checkpoints a quarter of the checkpoints are offered to the gate (the first, then each made 120 minutes or more after the
one the gate last played to a verdict), the ones in between are never decided by the prefilter and are superseded by the next gate;
the anchor is read from the state on disk (a restart changes nothing); the pre-filter's own rules (best gap, floor) hold on what is
offered. Nothing here plays a game or loads torch.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gym_loop as gl  # noqa: E402

T0 = dt.datetime(2026, 11, 14, 0, 0, 0, tzinfo=dt.timezone.utc)
STAMP = "%Y-%m-%dT%H:%M:%SZ"

# What the loop chose on the synthetic step log below, as the code stood before `--gate-every-min` (main 4244b89e): the checkpoints the gate
# was offered and every checkpoint's final verdict. The flag absent must give this to the letter.
GOLDEN_CHOICES = (
    "{\"offered\": [\"c0001\", \"c0002\", \"c0003\", \"c0010\", \"c0014\", \"c0017\", \"c0018\", \"c0030\", \"c0037\", \"c0038\"], \"verdi"
    "cts\": {\"c0001\": \"reject\", \"c0002\": \"reject\", \"c0003\": \"reject\", \"c0004\": \"skipped\", \"c0005\": \"skipped\", \"c0006"
    "\": \"skipped\", \"c0007\": \"skipped\", \"c0008\": \"skipped\", \"c0009\": \"skipped\", \"c0010\": \"reject\", \"c0011\": \"skipped"
    "\", \"c0012\": \"skipped\", \"c0013\": \"skipped\", \"c0014\": \"reject\", \"c0015\": \"skipped\", \"c0016\": \"skipped\", \"c0017\":"
    " \"reject\", \"c0018\": \"reject\", \"c0019\": \"skipped\", \"c0020\": \"skipped\", \"c0021\": \"skipped\", \"c0022\": \"skipped\", "
    "\"c0023\": \"skipped\", \"c0024\": \"skipped\", \"c0025\": \"skipped\", \"c0026\": \"skipped\", \"c0027\": \"skipped\", \"c0028\": \""
    "skipped\", \"c0029\": \"skipped\", \"c0030\": \"reject\", \"c0031\": \"skipped\", \"c0032\": \"skipped\", \"c0033\": \"skipped\", \""
    "c0034\": \"skipped\", \"c0035\": \"skipped\", \"c0036\": \"skipped\", \"c0037\": \"reject\", \"c0038\": \"reject\", \"c0039\": \"ski"
    "pped\", \"c0040\": \"skipped\"}}"
)


def _checkpoint(loop: "gl.Loop", n: int, loss: float, ref: float, t: dt.datetime) -> Path:
    """A checkpoint as the trainer writes it: weights, model.json and a train.json naming the incumbent and the split, stamped `t`."""
    name = f"c{n:04d}"
    d = loop.checkpoints / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.json").write_text("{}", encoding="utf-8")
    (d / "combat_policy.pt").write_bytes(b"weights of " + name.encode())
    inc, inc_dir = loop.incumbent()
    info = {"checkpoint": name, "t": t.strftime(STAMP), "val_rows": 20000, "val_policy_ce": loss, "base_val_policy_ce": ref,
            "base_version": inc, "base_sha256": gl.sha256_file(inc_dir / "combat_policy.pt"), "split": gl.val_split_id()}
    (d / "train.json").write_text(json.dumps(info), encoding="utf-8")
    return d


def _loop(tmp: str) -> "gl.Loop":
    loop = gl.Loop(Path(tmp) / "loop")
    d = loop.players / "v2"
    d.mkdir(parents=True, exist_ok=True)
    (d / "combat_policy.pt").write_bytes(b"weights of v2")
    loop.state["versions"]["v2"] = {"dir": str(d), "n": 2}
    loop.state["incumbent"] = "v2"
    return loop


def _gaps(n: int, mode: str) -> list[float]:
    """The held-out gaps of n checkpoints: 'rising' passes every look of the prefilter; 'noisy' passes some and skips the rest."""
    if mode == "rising":
        return [0.0010 + 0.0005 * i for i in range(n)]
    rnd = random.Random(11)
    return [round(0.004 + 0.003 * rnd.random() + 0.0001 * i, 5) for i in range(n)]


def simulate(tmp: str, *, n: int = 40, mode: str = "noisy", args: argparse.Namespace | None = None, restart: bool = False,
             prefilter: str = "best") -> dict:
    """A step log: every 30 minutes the trainer writes a checkpoint; the daemon asks `Daemon.next_gate` for the gate's candidate a
    few seconds later; a chosen one is gated (a reject, 12 minutes of play), the older ungated ones are superseded as `Daemon.gate`
    does. Returns the checkpoints offered, the final verdict of every checkpoint and the log's prefilter lines."""
    loop = _loop(tmp)
    ns = argparse.Namespace(root=str(loop.root), prefilter=prefilter, prefilter_margin=0.0, prefilter_floor_h=6.0, train_window=1000,
                            **(vars(args) if args else {}))
    daemon = gl.Daemon(ns)
    daemon.loop = loop
    gaps = _gaps(n, mode)
    offered: list[str] = []
    for i in range(1, n + 1):
        t = T0 + dt.timedelta(minutes=30 * i)
        _checkpoint(loop, i, 1.0700 - gaps[i - 1], 1.0700, t)
        if restart and i % 7 == 0:  # the daemon stopped and started: only what is on disk survives
            loop.save()
            loop = gl.Loop(loop.root)
            daemon.loop = loop
        with mock.patch.object(gl, "utc", lambda t=t: t + dt.timedelta(seconds=5)):
            pick = daemon.next_gate()
        if pick is None:
            continue
        name = pick[0]
        offered.append(name)
        now = t + dt.timedelta(minutes=12)
        verdict = {"verdict": "reject", "stage": 2}
        if ns.prefilter != "off":
            gl.prefilter_note_verdict(loop, name, verdict, now)
        for old in sorted(p.name for p in loop.checkpoints.glob("c[0-9][0-9][0-9][0-9]") if p.name < name):
            loop.state["gated"].setdefault(old, {"verdict": "superseded", "by": name})
        loop.state["gated"][name] = dict(verdict)
    return {"offered": offered, "verdicts": {k: v["verdict"] for k, v in sorted(loop.state["gated"].items())}}


class GateCadenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_with_the_flag_absent_the_choices_are_what_they_were(self):
        got = json.dumps(simulate(self.tmp.name), sort_keys=True)
        self.assertEqual(got, GOLDEN_CHOICES)

    def test_a_zero_interval_is_the_flag_absent(self):
        got = json.dumps(simulate(self.tmp.name, args=argparse.Namespace(gate_every_min=0.0)), sort_keys=True)
        self.assertEqual(got, GOLDEN_CHOICES)

    def test_with_the_flag_absent_and_the_prefilter_off_every_checkpoint_is_gated(self):
        got = simulate(self.tmp.name, mode="rising", prefilter="off", n=12)
        self.assertEqual(got["offered"], [f"c{i:04d}" for i in range(1, 13)])

    def test_the_default_of_the_flag_is_today(self):
        with mock.patch.object(gl, "Daemon") as daemon:
            daemon.return_value.run.return_value = 0
            gl.main(["run", "--root", str(Path(self.tmp.name) / "loop")])
            a = daemon.call_args.args[0]
        self.assertEqual(a.gate_every_min, 0.0)
        self.assertEqual((a.checkpoint_min, a.max_epochs), (30.0, 4.0))  # the trainer's own cadence and pass cap are not touched
        with mock.patch.object(gl, "Daemon") as daemon:
            daemon.return_value.run.return_value = 0
            gl.main(["run", "--root", str(Path(self.tmp.name) / "loop"), "--gate-every-min", "120", "--max-epochs", "16"])
            a = daemon.call_args.args[0]
        self.assertEqual((a.gate_every_min, a.max_epochs), (120.0, 16.0))
        self.assertEqual(gl.prefilter_cfg(a)["gate_every_min"], 120.0)

    def test_one_checkpoint_in_four_is_offered_at_120_minutes_against_30(self):
        every = argparse.Namespace(gate_every_min=120.0)
        base = simulate(self.tmp.name + "/a", mode="rising", n=40)
        got = simulate(self.tmp.name + "/b", mode="rising", n=40, args=every)
        self.assertEqual(len(base["offered"]), 40)
        self.assertEqual(len(got["offered"]), 10)  # a quarter
        self.assertEqual(got["offered"], [f"c{i:04d}" for i in range(1, 41, 4)])
        # the checkpoints in between are superseded by the next gate, never decided (skipped) by the prefilter
        self.assertEqual({v for k, v in got["verdicts"].items() if k not in got["offered"]}, {"superseded"})
        self.assertEqual({got["verdicts"][k] for k in got["offered"]}, {"reject"})

    def test_the_prefilters_own_rules_hold_on_what_is_offered(self):
        every = argparse.Namespace(gate_every_min=120.0)
        got = simulate(self.tmp.name, mode="noisy", n=40, args=every)
        base = simulate(self.tmp.name + "/base", mode="noisy", n=40)
        self.assertLess(len(got["offered"]), len(base["offered"]))
        # every offered checkpoint is at least 120 minutes after the one before it (the first is offered at once)
        times = [T0 + dt.timedelta(minutes=30 * int(c[1:])) for c in got["offered"]]
        for a, b in zip(times, times[1:]):
            self.assertGreaterEqual((b - a).total_seconds(), 120 * 60)
        # a checkpoint made inside the interval after a gated one is never decided by the prefilter (it is superseded, not skipped)
        for o in got["offered"]:
            for c, v in got["verdicts"].items():
                if 0 < int(c[1:]) - int(o[1:]) < 4:
                    self.assertNotEqual(v, "skipped", (o, c))

    def test_the_anchor_is_on_disk_so_a_restart_changes_nothing(self):
        every = argparse.Namespace(gate_every_min=120.0)
        a = simulate(self.tmp.name + "/a", mode="rising", n=40, args=every)
        b = simulate(self.tmp.name + "/b", mode="rising", n=40, args=every, restart=True)
        self.assertEqual(a, b)

    def test_the_prefilter_off_path_offers_the_newest_checkpoint_only_once_the_interval_has_passed(self):
        every = argparse.Namespace(gate_every_min=90.0)
        got = simulate(self.tmp.name, mode="rising", prefilter="off", n=12, args=every)
        self.assertEqual(got["offered"], ["c0001", "c0004", "c0007", "c0010"])

    def test_the_interval_counts_from_when_the_checkpoint_was_made_not_from_the_verdict(self):
        loop = _loop(self.tmp.name)
        cfg = {**gl.PREFILTER, "window": 1000, "gate_every_min": 120.0}
        _checkpoint(loop, 1, 1.0600, 1.0700, T0)
        self.assertEqual(gl.prefilter_pick(loop, cfg, T0 + dt.timedelta(seconds=5))[0], "c0001")
        loop.state["gated"]["c0001"] = {"verdict": "reject"}
        gl.prefilter_note_verdict(loop, "c0001", {"verdict": "reject"}, T0 + dt.timedelta(minutes=300))  # a long gate: late verdict
        _checkpoint(loop, 2, 1.0550, 1.0700, T0 + dt.timedelta(minutes=119, seconds=29))
        self.assertIsNone(gl.prefilter_pick(loop, cfg, T0 + dt.timedelta(minutes=400)))
        _checkpoint(loop, 3, 1.0540, 1.0700, T0 + dt.timedelta(minutes=120))
        self.assertEqual(gl.prefilter_pick(loop, cfg, T0 + dt.timedelta(minutes=401))[0], "c0003")


if __name__ == "__main__":
    unittest.main()
