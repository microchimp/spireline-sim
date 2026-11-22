"""The fight loop's output reaches the run player by itself, or the trainer pauses (rl/gym_wiring.py; D68, Act 3 week, lane 6c).

The pinned facts: a promoted player queues its own run candidate (the champion's folder with only its fight brain swapped, under
agent/gen<k+1>/candidate-fight-v<N>, and the lab's marker ready-<k>-g<k+1>-fight-v<N> carrying the lab's keep rule); the same player is never
built or queued twice, a champion that changed is the next base, a half-built folder is built again; a player that no champion, queued or
judged candidate uses 24 hours after its promotion writes LAB/alerts/gym-unconsumed.txt and pauses the daemon, which resumes when the player
is consumed or the file is removed (an acknowledgement); the run judge's verdict on the candidate that carried a player is joined to the gate's
books once. The bookkeeping runs on fixture folders and a fake builder; one test builds a real candidate from the repo's tiny trained prior.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _p in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import gym_clock as gc  # noqa: E402
import gym_wiring as gw  # noqa: E402

T0 = dt.datetime(2026, 11, 19, 6, 0, 0, tzinfo=dt.timezone.utc)
PROMOTED = "2026-11-19T06:00:00Z"


def put(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def promotion_doc(*, promote=True, kills=(158, 159), wins=(2, 0), pairs=2000, rule="v3") -> dict:
    return {"written": "2026-11-18T16:05:04Z", "paired_seeds": pairs, "keep_rule": rule,
            "decision": {"promote": promote, "rule": "keep-v3"},
            "keep_v2": {"kills": {"B1": [1530, 1456], "B2": [490, 472], "B3a": list(kills), "win": list(wins)}},
            "wins": {"pairs": pairs, "cand": wins[0], "base": wins[1]}}


class World:
    """A lab in a temp folder: the champion (generation 5, fight network aaa), the gym's players, the lab's train and alerts folders."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.agents, self.train, self.alerts = root / "agent", root / "LAB" / "train", root / "LAB" / "alerts"
        self.jobs = put(root / "LAB" / "jobs.json", {"train": {"cmd": ["python", "rl/gen0.py", "train", "--keep-rule", "v3", "--procs", "16"]}})
        self.logs: list[str] = []
        self.builds: list[tuple] = []
        self.fail: str | None = None
        self.champion(5, "a" * 64)
        self.state: dict = {"versions": {}, "promotions": []}
        self.w = gw.Wiring(agent_root=self.agents, train_dir=self.train, alerts_dir=self.alerts, jobs_json=self.jobs, partial_dir=root / "partial",
                           builder=self.build)

    def champion(self, generation: int, sha: str) -> Path:
        d = self.agents / f"gen{generation}" / "candidate"
        put(d / "model.json", {"policy_sha256": sha})
        put(d / "agent.json", {"generation": generation, "macro": "vrun"})
        put(self.agents / "current.json", {"generation": generation, "dir": str(d).replace("/", "\\")})
        return d

    def player(self, version: str, sha: str, promoted: str = PROMOTED, checkpoint: str = "c0001") -> Path:
        d = self.root / "players" / version
        put(d / "model.json", {"policy_sha256": sha, "gym_loop": {"checkpoint": checkpoint}})
        (d / "combat_policy.pt").write_bytes(sha.encode())
        self.state["versions"][version] = {"dir": str(d), "promoted": promoted}
        self.state["promotions"].append({"version": version, "checkpoint": checkpoint, "over": "v0", "t": promoted})
        return d

    def build(self, base, player_dir, agent_root, target, tag, why) -> Path:
        """A fake builder: the folder a real build leaves (model.json, agent.json last), recording what it was asked."""
        self.builds.append((Path(base), Path(player_dir), int(target), tag, why))
        if self.fail:
            raise gw.WiringError(self.fail)
        cand = Path(agent_root) / f"gen{target}" / f"candidate-{tag}"
        base_meta = json.loads((Path(base) / "agent.json").read_text())
        put(cand / "model.json", {"policy_sha256": gw.policy_sha(player_dir)})
        put(cand / "agent.json", {"generation": target, "built_on": base_meta["generation"],
                                  "built": {"swap": "fight", "base": {"policy_sha256": gw.policy_sha(base)}, "tag": tag}})
        return cand

    def log(self, msg: str) -> None:
        self.logs.append(msg)

    def evaluate(self, now: dt.datetime = T0, **kw) -> dict:
        return gw.evaluate(self.w, self.state, now, log=self.log, gates_dir=self.root / "gate", **kw)


class WiringCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        (Path(self.tmp.name) / "gate").mkdir()
        self.world = World(Path(self.tmp.name))


class QueueTests(WiringCase):
    def test_a_promoted_player_queues_the_champion_with_its_fight_network_swapped(self):
        w = self.world
        pdir = w.player("v4", "b" * 64)
        got = gw.queue_candidate(w.w, "v4", pdir, checkpoint="c0185", promoted=PROMOTED, now=T0, log=w.log)
        self.assertTrue(got["built"])
        self.assertEqual(w.agents / "gen6" / "candidate-fight-v4", Path(got["candidate"]))
        base, player, target, tag, why = w.builds[0]
        self.assertEqual((w.agents / "gen5" / "candidate", pdir, 6, "fight-v4"), (base, player, target, tag))
        self.assertIn("c0185", why)
        self.assertEqual("ready-5-g6-fight-v4", got["marker"])
        text = (w.train / got["marker"]).read_text()
        self.assertEqual(["2026-11-19T06:00:00Z", "keep_rule: v3"], text.splitlines())  # the stamp as ever, then the lab's rule
        self.assertEqual("v3", got["keep_rule"])
        self.assertEqual(["ready-5-g6-fight-v4"], sorted(p.name for p in w.train.iterdir()))  # no temporary file left behind
        self.assertEqual("queued", gw.consumption(w.w, "v4", pdir)["state"])

    def test_the_candidate_it_queues_is_a_fight_only_swap_as_route_s_reads_the_field(self):
        import gen_loop

        w = self.world
        pdir = w.player("v4", "b" * 64)
        got = gw.queue_candidate(w.w, "v4", pdir, checkpoint="c0185", promoted=PROMOTED, now=T0, log=w.log)
        cand = Path(got["candidate"])
        meta = json.loads((cand / "agent.json").read_text())
        self.assertEqual(meta["built"]["swap"], gen_loop.FIGHT_SWAP)  # the field the wiring writes is the one route S reads
        self.assertTrue(gen_loop.fight_only_swap(cand))
        self.assertFalse(gen_loop.fight_only_swap(w.agents / "gen5" / "candidate"))  # the champion's own folder is not a swap

    def test_the_marker_has_the_lab_rule_from_the_field_or_the_command_and_none_when_unknown(self):
        w = self.world
        self.assertEqual("v3", gw.lab_keep_rule(w.jobs))
        put(w.jobs, {"train": {"keep_rule": "v4", "cmd": ["--keep-rule", "v3"]}})
        self.assertEqual("v4", gw.lab_keep_rule(w.jobs))  # lane 6a's field wins over the command line
        put(w.jobs, {"train": {"cmd": ["--keep-rule=v2"]}})
        self.assertEqual("v2", gw.lab_keep_rule(w.jobs))
        put(w.jobs, {"train": {"cmd": ["rl/gen0.py"]}})
        self.assertIsNone(gw.lab_keep_rule(w.jobs))
        self.assertEqual("2026-11-19T06:00:00Z\n", gw.marker_text(T0, None))
        self.assertIsNone(gw.lab_keep_rule(w.root / "absent.json"))

    def test_it_is_idempotent_and_never_queues_a_player_twice(self):
        w = self.world
        pdir = w.player("v4", "b" * 64)
        gw.queue_candidate(w.w, "v4", pdir, now=T0, log=w.log)
        again = gw.queue_candidate(w.w, "v4", pdir, now=T0 + dt.timedelta(hours=1), log=w.log)
        self.assertEqual(1, len(w.builds))
        self.assertFalse(again["built"])
        self.assertEqual("2026-11-19T06:00:00Z", (w.train / "ready-5-g6-fight-v4").read_text().splitlines()[0])  # not rewritten
        (w.train / "ready-5-g6-fight-v4").rename(w.train / "done-5-g6-fight-v4")  # the lab judged it and consumed the marker
        gw.queue_candidate(w.w, "v4", pdir, now=T0 + dt.timedelta(hours=2), log=w.log)
        self.assertEqual(["done-5-g6-fight-v4"], sorted(p.name for p in w.train.iterdir()))
        self.assertEqual(1, len(w.builds))

    def test_a_built_folder_whose_marker_was_never_written_gets_its_marker_without_a_second_build(self):
        w = self.world
        pdir = w.player("v4", "b" * 64)
        w.build(w.agents / "gen5" / "candidate", pdir, w.agents, 6, "fight-v4", "by hand")  # a crash after the build, before the marker
        self.assertEqual("none", gw.consumption(w.w, "v4", pdir)["state"])  # built and not queued is not consumed
        got = gw.queue_candidate(w.w, "v4", pdir, now=T0, log=w.log)
        self.assertFalse(got["built"])
        self.assertEqual(1, len(w.builds))
        self.assertTrue((w.train / "ready-5-g6-fight-v4").is_file())

    def test_a_champion_that_already_has_the_fight_network_needs_no_candidate(self):
        w = self.world
        pdir = w.player("v4", "a" * 64)
        got = gw.queue_candidate(w.w, "v4", pdir, now=T0, log=w.log)
        self.assertIsNone(got["candidate"])
        self.assertEqual([], w.builds)
        self.assertEqual("champion", gw.consumption(w.w, "v4", pdir)["state"])

    def test_the_next_promotion_builds_on_the_champion_of_that_day(self):
        w = self.world
        gw.queue_candidate(w.w, "v4", w.player("v4", "b" * 64), now=T0, log=w.log)
        w.champion(6, "b" * 64)  # the run judge promoted it: generation 6 is the champion now
        gw.queue_candidate(w.w, "v5", w.player("v5", "c" * 64), now=T0 + dt.timedelta(days=1), log=w.log)
        base, _, target, tag, _ = w.builds[1]
        self.assertEqual((w.agents / "gen6" / "candidate", 7, "fight-v5"), (base, target, tag))
        self.assertTrue((w.train / "ready-6-g7-fight-v5").is_file())

    def test_a_failed_build_writes_no_marker_and_a_half_built_folder_is_moved_aside_and_built_again(self):
        w = self.world
        pdir = w.player("v4", "b" * 64)
        w.fail = "torch could not export"
        with self.assertRaisesRegex(gw.WiringError, "could not export"):
            gw.queue_candidate(w.w, "v4", pdir, now=T0, log=w.log)
        self.assertFalse(any(w.train.glob("*")) if w.train.exists() else False)
        half = w.agents / "gen6" / "candidate-fight-v4"
        (half / "combat_policy.pt").parent.mkdir(parents=True, exist_ok=True)
        (half / "combat_policy.pt").write_bytes(b"cut short")  # no agent.json: the build died
        w.fail = None
        got = gw.queue_candidate(w.w, "v4", pdir, now=T0, log=w.log)
        self.assertTrue(got["built"])
        self.assertTrue((half / "agent.json").is_file())
        moved = list((w.root / "partial").iterdir())
        self.assertEqual(1, len(moved))
        self.assertEqual(b"cut short", (moved[0] / "combat_policy.pt").read_bytes())
        self.assertTrue(any("moved aside" in m for m in w.logs))

    def test_a_builder_that_leaves_nothing_where_it_was_asked_is_an_error(self):
        w = self.world
        pdir = w.player("v4", "b" * 64)
        w.w.builder = lambda *a: w.root / "elsewhere"
        with self.assertRaises(gw.WiringError):
            gw.queue_candidate(w.w, "v4", pdir, now=T0, log=w.log)


class ConsumptionTests(WiringCase):
    def test_the_champions_fight_network_a_queued_marker_and_a_judged_candidate_each_consume_a_player(self):
        w = self.world
        sha = "b" * 64
        pdir = w.player("v4", sha)
        self.assertEqual("none", gw.consumption(w.w, "v4", pdir)["state"])
        put(w.train / "ready-5-g6-fight-v4", {})
        self.assertEqual("queued", gw.consumption(w.w, "v4", pdir)["state"])
        (w.train / "ready-5-g6-fight-v4").unlink()
        w.build(w.agents / "gen5" / "candidate", pdir, w.agents, 6, "fight-v4", "x")
        put(w.agents / "gen6" / "candidate-fight-v4" / "promotion.json", promotion_doc())
        self.assertEqual(("judged", "gen6/candidate-fight-v4"), tuple(gw.consumption(w.w, "v4", pdir)[k] for k in ("state", "via")))
        w.champion(6, sha)
        self.assertEqual("champion", gw.consumption(w.w, "v4", pdir)["state"])

    def test_another_players_marker_does_not_consume_this_one(self):
        w = self.world
        put(w.train / "ready-5-g6-fight-v40", {})
        put(w.train / "ready-5-g6-fight-v3", {})
        self.assertEqual("none", gw.consumption(w.w, "v4", w.player("v4", "b" * 64))["state"])


class PauseTests(WiringCase):
    def test_an_unconsumed_player_pauses_the_loop_only_after_a_day_and_the_alert_names_why(self):
        w = self.world
        w.player("v4", "b" * 64)
        w.fail = "no GPU venv"
        first = w.evaluate(T0 + dt.timedelta(hours=1))
        self.assertIsNone(first["paused"])  # an hour after the promotion: still trying
        self.assertFalse((w.alerts / gw.ALERT_FILE).exists())
        self.assertEqual(1, sum("not queued" in m for m in w.logs))
        w.evaluate(T0 + dt.timedelta(hours=2))
        self.assertEqual(1, sum("not queued" in m for m in w.logs))  # the same failure is said once
        late = w.evaluate(T0 + dt.timedelta(hours=24, minutes=1))
        self.assertIn("v4 unconsumed", late["paused"])
        text = (w.alerts / gw.ALERT_FILE).read_text()
        self.assertTrue(text.startswith("ALERT gym-unconsumed: the fight loop's player v4"))
        self.assertIn("no GPU venv", text)
        self.assertEqual("v4", w.state["wiring"]["alert"]["version"])
        again = w.evaluate(T0 + dt.timedelta(hours=25))
        self.assertIsNotNone(again["paused"])  # it stays paused while nothing changes
        self.assertEqual(1, sum("wrote" in m and gw.ALERT_FILE in m for m in w.logs))

    def test_the_loop_resumes_when_the_player_is_consumed(self):
        w = self.world
        w.player("v4", "b" * 64)
        w.fail = "no GPU venv"
        self.assertIsNotNone(w.evaluate(T0 + dt.timedelta(hours=25))["paused"])
        w.fail = None  # the cause is fixed; the next look queues it
        res = w.evaluate(T0 + dt.timedelta(hours=26))
        self.assertIsNone(res["paused"])
        self.assertEqual("queued", res["consumption"]["state"])
        self.assertFalse((w.alerts / gw.ALERT_FILE).exists())
        self.assertNotIn("alert", w.state["wiring"])

    def test_the_orchestrator_removing_the_file_resumes_the_loop_and_it_is_not_alerted_again(self):
        w = self.world
        w.player("v4", "b" * 64)
        w.fail = "no GPU venv"
        self.assertIsNotNone(w.evaluate(T0 + dt.timedelta(hours=25))["paused"])
        (w.alerts / gw.ALERT_FILE).unlink()
        res = w.evaluate(T0 + dt.timedelta(hours=26))
        self.assertIsNone(res["paused"])
        self.assertEqual(["v4"], w.state["wiring"]["acked"])
        self.assertIsNone(w.evaluate(T0 + dt.timedelta(days=5))["paused"])
        self.assertFalse((w.alerts / gw.ALERT_FILE).exists())
        # a newer promotion is a new question
        w.player("v5", "c" * 64, promoted="2026-11-25T00:00:00Z")
        self.assertIsNone(w.evaluate(dt.datetime(2026, 11, 25, 1, tzinfo=dt.timezone.utc))["paused"])
        self.assertIsNotNone(w.evaluate(dt.datetime(2026, 11, 26, 1, tzinfo=dt.timezone.utc))["paused"])

    def test_a_player_used_by_the_champion_never_pauses_and_clears_a_standing_alert(self):
        w = self.world
        w.player("v4", "b" * 64)
        w.fail = "no GPU venv"
        w.evaluate(T0 + dt.timedelta(hours=25))
        self.assertTrue((w.alerts / gw.ALERT_FILE).is_file())
        w.champion(6, "b" * 64)
        self.assertIsNone(w.evaluate(T0 + dt.timedelta(hours=26))["paused"])
        self.assertFalse((w.alerts / gw.ALERT_FILE).exists())

    def test_no_promotion_yet_is_nothing_to_look_at(self):
        w = self.world
        self.assertEqual({"version": None, "consumption": None, "paused": None, "queued": None, "error": None}, w.evaluate())

    def test_a_look_that_is_not_allowed_to_build_only_reads(self):
        w = self.world
        w.player("v4", "b" * 64)
        res = w.evaluate(T0 + dt.timedelta(hours=1), queue=False)
        self.assertEqual([], w.builds)
        self.assertEqual("none", res["consumption"]["state"])
        self.assertIsNone(res["paused"])
        self.assertIsNotNone(w.evaluate(T0 + dt.timedelta(hours=30), queue=False)["paused"])  # but a day without use still pauses


class JoinTests(WiringCase):
    def test_the_run_judges_verdict_is_joined_once_to_the_player_the_candidate_carried(self):
        w = self.world
        pdir = w.player("v4", "b" * 64, checkpoint="c0185")
        cand = w.build(w.agents / "gen5" / "candidate", pdir, w.agents, 6, "fight-v4", "x")
        put(cand / "promotion.json", promotion_doc(promote=False, kills=(158, 159), wins=(2, 0), pairs=2000))
        res = w.evaluate()
        self.assertEqual("judged", res["consumption"]["state"])
        rows = [json.loads(x) for x in (w.root / "gate" / "run-judge.jsonl").read_text().splitlines()]
        self.assertEqual(1, len(rows))
        r = rows[0]
        self.assertEqual(("v4", "c0185", "gen6/candidate-fight-v4", False, 2000), (r["version"], r["checkpoint"], r["candidate"], r["promoted"], r["pairs"]))
        self.assertEqual(([158, 159], [2, 0], "v3"), (r["act3_boss_kills"], r["wins"], r["keep_rule"]))
        line = [m for m in w.logs if m.startswith("run judge:")]
        self.assertEqual(1, len(line))
        self.assertIn("v4 (gate c0185) carried by gen6/candidate-fight-v4: not promoted under keep rule v3 on 2000 pairs", line[0])
        self.assertIn("first Act 3 boss kills 158 against 159, wins 2 against 0", line[0])
        w.evaluate()
        self.assertEqual(1, len((w.root / "gate" / "run-judge.jsonl").read_text().splitlines()))  # once per judgement
        put(cand / "promotion.json", {**promotion_doc(), "written": "2026-11-20T01:00:00Z"})  # a re-judgement is a new verdict
        w.evaluate()
        self.assertEqual(2, len((w.root / "gate" / "run-judge.jsonl").read_text().splitlines()))

    def test_a_candidate_that_only_inherited_the_network_is_not_the_one_that_carried_it(self):
        w = self.world
        pdir = w.player("v4", "b" * 64)
        cand = w.agents / "gen7" / "candidate-cheap"  # a later candidate of the same line: built on a base that already had the network
        put(cand / "model.json", {"policy_sha256": "b" * 64})
        put(cand / "agent.json", {"generation": 7, "built": {"base": {"policy_sha256": "b" * 64}}})
        put(cand / "promotion.json", promotion_doc())
        w.champion(7, "b" * 64)
        w.evaluate()
        self.assertFalse((w.root / "gate" / "run-judge.jsonl").exists())
        self.assertEqual([("gen7/candidate", False), ("gen7/candidate-cheap", False)],
                         [(c["rel"], c["introduced"]) for c in gw.carrying(w.agents, gw.policy_sha(pdir))])  # the champion's own folder too: neither introduced it

    def test_the_summary_survives_a_document_without_the_fields(self):
        s = gw.run_judge_summary({"decision": {}})
        self.assertEqual((None, None, None), (s["promoted"], s["act3_boss_kills"], s["wins"]))
        self.assertIn("no verdict read", gw.judge_line("v4", "gen6/x", s))

    def test_the_daily_line_names_the_days_run_judge_verdicts(self):
        d = Path(self.tmp.name)
        put(d / "x.json", {})
        (d / "run-judge.jsonl").write_text(json.dumps({"t": "2026-11-19T07:00:00Z", "version": "v4", "promoted": True, "act3_boss_kills": [158, 159],
                                                       "wins": [2, 0], "pairs": 2000}) + "\n" + json.dumps({"t": "2026-11-18T07:00:00Z", "version": "v3"}) + "\n")
        clock = [dt.datetime(2026, 11, 20, 1, tzinfo=dt.timezone.utc)]
        led = gc.TimeLedger(d / "clock.jsonl", lambda m: None, now=lambda: clock[0], run_judge=d / "run-judge.jsonl")
        line = led.daily_line(dt.date(2026, 11, 19))
        self.assertTrue(line.endswith("run judge: v4 promoted (first Act 3 boss kills 158 against 159, wins 2 against 0 of 2000 pairs)"), line)
        self.assertNotIn("run judge", led.daily_line(dt.date(2026, 11, 17)))


class DaemonTests(unittest.TestCase):
    """The daemon's side: the options, the tick, the pause (the loop itself needs the engine and is not run here)."""

    def setUp(self):
        import gym_loop as gl

        self.gl = gl
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.world = World(Path(self.tmp.name))
        (Path(self.tmp.name) / "gate").mkdir(exist_ok=True)
        argv = ["run", "--root", str(Path(self.tmp.name) / "loop"), "--procs", "0", "--gate-procs", "1",
                "--agent-root", str(self.world.agents), "--train-dir", str(self.world.train), "--alerts-dir", str(self.world.alerts),
                "--jobs-json", str(self.world.jobs)]
        self.args = gl.build_parser().parse_args(argv)
        self.d = gl.Daemon(self.args)
        self.d.loop.state.update(self.world.state)
        self.world.state = self.d.loop.state
        self.d.wiring = self.d.make_wiring()
        self.d.wiring.builder = self.world.build
        status = mock.patch.object(gl, "STATUS_DIR", Path(self.tmp.name) / "status")
        status.start()
        self.addCleanup(status.stop)

    def test_the_options_default_to_wiring_on_with_a_day(self):
        self.assertEqual(("on", gw.UNCONSUMED_H), (self.args.wiring, self.args.unconsumed_h))
        off = self.gl.build_parser().parse_args(["run", "--root", str(Path(self.tmp.name) / "x"), "--procs", "0", "--gate-procs", "1", "--wiring", "off"])
        self.assertIsNone(self.gl.Daemon(off).make_wiring())

    def test_a_promotion_is_queued_at_once_and_an_unconsumed_player_pauses_after_a_day(self):
        w = self.world
        w.player("v4", "b" * 64, promoted="2026-11-01T00:00:00Z")
        self.d.wiring_tick(queue=True, force=True)
        self.assertIsNone(self.d.paused)
        self.assertTrue((w.train / "ready-5-g6-fight-v4").is_file())
        self.assertIn("wiring", json.loads((self.d.loop.root / "state.json").read_text()))  # the daemon's look is saved with its state
        (w.train / "ready-5-g6-fight-v4").unlink()
        shutil.rmtree(w.agents / "gen6")  # the candidate is lost, and the day is over (the promotion was weeks ago)
        w.fail = "no GPU venv"
        self.d.wiring_tick(queue=True, force=True)
        self.assertIn("v4 unconsumed", self.d.paused)
        self.d.refresh_status()
        status = json.loads((Path(self.tmp.name) / "status" / "status").read_text())
        self.assertEqual(("off", self.d.paused), (status["generation"], status["paused"]))
        self.assertTrue((w.alerts / gw.ALERT_FILE).is_file())

    def test_pausing_stops_the_trainer_and_waits_as_a_logged_wait(self):
        self.d.paused = "v4 unconsumed 25 h after its promotion"
        self.d.clock = gc.TimeLedger(self.d.loop.games / "clock.jsonl", self.d.loop.log)
        stopped = []
        with mock.patch.object(self.d, "stop_trainer", side_effect=lambda: stopped.append(1)), mock.patch.object(self.gl.time, "sleep"):
            self.d.pause_unconsumed()
        self.assertEqual([1], stopped)
        self.assertIn("wait start: unconsumed", (self.d.loop.root / "loop.log").read_text())

    def test_a_wiring_failure_is_logged_once_and_never_raised(self):
        self.d.loop.state["promotions"] = [{"version": "v9", "t": PROMOTED}]
        self.d.loop.state["versions"] = {"v9": "not a dict"}
        with mock.patch.object(gw, "evaluate", side_effect=RuntimeError("boom")):
            self.d.wiring_tick(force=True)
            self.d.wiring_tick(force=True)
        self.assertEqual(1, (self.d.loop.root / "loop.log").read_text().count("wiring: RuntimeError: boom"))


class BuildTests(unittest.TestCase):
    """The real builder on the repo's tiny trained prior: the champion with only its fight network swapped."""

    @classmethod
    def setUpClass(cls):
        import gen_loop
        import torch
        from test_gen_loop import student_world

        cls.gen_loop, cls.torch = gen_loop, torch
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.agents, cls.base, prior, cls.student, _ = student_world(root)
        cls.root = root
        cls.player = root / "players" / "v9"  # a second network: the fixture prior with its weights nudged
        cls.player.mkdir(parents=True)
        for name in ("combat_policy.pt", "combat_net.pt"):
            obj = torch.load(prior / name, map_location="cpu", weights_only=True)
            gen = torch.Generator().manual_seed(11)
            for k, v in obj["model"].items():
                if v.is_floating_point():
                    v.add_(torch.randn(v.shape, generator=gen) * 1e-3)
            torch.save(obj, cls.player / name)
        gen_loop.write_prior_model_json(cls.player, 9)
        doc = json.loads((cls.player / "model.json").read_text())
        doc["gym_loop"] = {"checkpoint": "c0777"}
        (cls.player / "model.json").write_text(json.dumps(doc))
        (cls.base / "run_value.onnx").write_bytes(b"the base's V export")
        (cls.base / "fight_net.onnx").write_bytes(b"the base's fight net export")
        (cls.base / "run_student_legs.json").write_text("{}")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        shutil.rmtree(self.agents / "gen6", ignore_errors=True)  # each test builds its own generation 6

    def build(self, **kw):
        kw.setdefault("export", False)
        return gw.build_fight_candidate(self.agents / "current.json", self.player, self.agents, 6, why="test", tag="fight-v9", **kw)

    def test_the_candidate_is_the_champion_with_only_its_fight_network_replaced(self):
        import screens

        gen_loop = self.gen_loop
        cand = self.build()
        self.assertEqual(self.agents / "gen6" / "candidate-fight-v9", cand)
        self.assertEqual(cand, gen_loop.built_candidate(self.agents, 6, tag="fight-v9"))
        base_sha = gen_loop.file_sha256(self.base / "combat_policy.pt")
        player_sha = gen_loop.file_sha256(self.player / "combat_policy.pt")
        self.assertNotEqual(base_sha, player_sha)
        self.assertEqual(player_sha, gw.policy_sha(cand))
        self.assertEqual(player_sha, gen_loop.load_agent(cand).describe()["prior_sha256"])
        for name in ("combat_policy.pt", "combat_net.pt"):
            self.assertEqual((self.player / name).read_bytes(), (cand / name).read_bytes(), name)
        for name in ("run_value.npz", "run_value.pt", "human_prior.npz", "run_student_legs.json", "run_value.onnx", "fight_net.onnx"):
            if (self.base / name).is_file():
                self.assertEqual((self.base / name).read_bytes(), (cand / name).read_bytes(), name)
        self.assertEqual(["fight"], screens.differing(cand, self.base))
        self.assertEqual("fight", screens.kind_of(cand, self.base))
        meta = json.loads((cand / "agent.json").read_text())
        base_meta = json.loads((self.base / "agent.json").read_text())
        for key in ("macro", "recipe", "human_prior", "search_schedule", "potion_hold", "mask_version"):
            self.assertEqual(base_meta.get(key), meta.get(key), key)
        self.assertEqual((6, 5), (meta["generation"], meta["built_on"]))
        built = meta[gen_loop.BUILT_KEY]
        self.assertEqual(("fight", "fight-v9", False, player_sha, base_sha),
                         (built["swap"], built["tag"], built["trained"], built["prior"]["policy_sha256"], built["base"]["policy_sha256"]))
        self.assertIn("c0777", meta["lineage"]["difference"])
        self.assertIn("fight network alone", meta["note"])
        self.assertFalse((cand / "promotion.json").exists())
        self.assertEqual(self.agents / "gen5" / "candidate", Path(json.loads((self.agents / "current.json").read_text())["dir"]))  # the champion is untouched

    def test_the_real_builders_candidate_is_a_fight_only_swap_for_route_s_and_the_base_is_not(self):
        gen_loop = self.gen_loop
        cand = self.build()
        field = json.loads((cand / "agent.json").read_text())[gen_loop.BUILT_KEY]["swap"]
        self.assertEqual((field, gen_loop.FIGHT_SWAP), ("fight", "fight"))  # what `build_fight_candidate` writes is what route S reads
        self.assertTrue(gen_loop.fight_only_swap(cand))
        self.assertFalse(gen_loop.fight_only_swap(self.base))
        meta = json.loads((cand / "agent.json").read_text())
        meta[gen_loop.BUILT_KEY]["swap"] = "vrun"  # any other builder's value is not a fight swap
        (cand / "agent.json").write_text(json.dumps(meta))
        self.assertFalse(gen_loop.fight_only_swap(cand))

    def test_a_second_build_never_writes_over_the_first_and_a_failure_leaves_nothing(self):
        cand = self.build()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertTrue((cand / "agent.json").is_file())
        with self.assertRaises(ValueError):  # not above the champion's generation
            gw.build_fight_candidate(self.agents / "current.json", self.player, self.agents, 5, why="t", tag="fight-v8", export=False)
        self.assertFalse((self.agents / "gen5" / "candidate-fight-v8").exists())
        broken = self.root / "players" / "v8"
        broken.mkdir(parents=True)
        (broken / "model.json").write_text(json.dumps({"policy_sha256": "0" * 64}))
        (broken / "combat_policy.pt").write_bytes(b"not a network")
        with self.assertRaises(Exception):
            gw.build_fight_candidate(self.agents / "current.json", broken, self.agents, 6, why="t", tag="fight-v8", export=False)
        self.assertFalse((self.agents / "gen6" / "candidate-fight-v8").exists())


if __name__ == "__main__":
    unittest.main()
