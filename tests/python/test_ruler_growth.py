"""The boss-door ruler grows with the champion (judge build 3, step 1; D68 addendum 6 outcome 5): `rl/act3_starts.py` (`arrivals` extends the ruler in `--out`
by the arms it lacks, `ruler_gap`, `build_ruler`'s history, `retire_exam` and `exam_games`: the exam's games belong to one hash) and `rl/gen_loop.py`
(`s1_stage` extends the ruler before the exam, a reading that stands is not played again on a bigger ruler, `s1_used_tests` counts the current hash only, a champion
with nothing to read makes no folder, the ledger row says `S1 unavailable`).

Nothing plays a game: the replay pool is a function that writes the rows a matching replay would (as `test_arrival_exam` does), the exam is the fake of
`test_route_s_plumbing`."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import act3_starts as a3  # noqa: E402
import fight_exam as fe  # noqa: E402
import fight_gym as fg  # noqa: E402
import gen_loop  # noqa: E402
import gym_loop as gl  # noqa: E402
import test_route_s_plumbing as trs  # noqa: E402
from test_arrival_exam import nodes_row, player, replay_row  # noqa: E402
from test_keep_v3 import block  # noqa: E402

from tools.lab import keep_v2 as k2  # noqa: E402

CHAMP_DIR = "gen7/cand"
A_ROWS = [nodes_row("a0", "died", 1), nodes_row("a1", "past", 2), nodes_row("a2", "early", 3)]  # a0 died at the first boss, a1 went past it: 3 starts
B_ROWS = [nodes_row("b0", "past", 1), nodes_row("b1", "died", 2)]  # 3 starts
C_ROWS = [nodes_row("c0", "died", 1), nodes_row("c1", "died", 2), nodes_row("c2", "won", 3)]  # 4 starts


def write_arm(champ: Path, label: str, rows: list[dict], agent: str = CHAMP_DIR) -> Path:
    arm = champ / "arms" / f"{label}-a3-v1"
    arm.mkdir(parents=True, exist_ok=True)
    (arm / "summary.json").write_text(json.dumps({"config": "a3-v1", "agent": {"dir": agent}}), encoding="utf-8")
    (arm / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return arm


class Pool:
    """The replay pool replaced: what `fight_gym.run_tasks` would write for each task, and a record of what it was asked."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.bad: set[str] = set()  # seeds whose replay does not match its row
        self.stop_after: int | None = None  # play this many tasks, then report a stop
        self.blocked: str | None = None

    def run_tasks(self, tasks, fn, target, procs, *, init=None, init_args=(), **kw):
        label = Path(target).stem.removeprefix("replays-")
        self.calls.append((label, [t[1] for t in tasks]))
        n = len(tasks) if self.stop_after is None else min(self.stop_after, len(tasks))
        with open(target, "a", encoding="utf-8", newline="\n") as fh:
            for t in tasks[:n]:
                fh.write(json.dumps(replay_row(t[5], label, same=t[1] not in self.bad)) + "\n")
        return "night-window" if n < len(tasks) else None

    def patches(self):
        # `cmd_arrivals` sets the exam module's list of QUIET.lock tags to ignore (the lab's own, from the stage's command line): a list of this test's own, so
        # that no later test in the process reads the lab's line as allowed
        return (mock.patch.object(fg, "run_tasks", self.run_tasks), mock.patch.object(fe, "blocked_reason", lambda **kw: self.blocked),
                mock.patch.object(fe, "prepare_process", lambda: None), mock.patch.object(fe, "ALLOW_QUIET", []))

    @property
    def replayed(self) -> list[str]:
        return [label for label, seeds in self.calls if seeds]


def arrivals(champ: Path, out: Path, agents: Path, *extra: str) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = a3.main(["arrivals", "--champion", str(champ), "--out", str(out), "--agent-root", str(agents), "--procs", "3", *extra])
    return rc, buf.getvalue()


class ExtendTests(unittest.TestCase):
    """`act3_starts.py arrivals` on a folder that already holds this champion's ruler."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.agents = self.root / "agent"
        self.champ = self.agents / "gen7" / "cand"
        self.champ.mkdir(parents=True)
        (self.champ / "promotion.json").write_text(json.dumps({"seed_block": "Esel"}), encoding="utf-8")
        write_arm(self.champ, "Esel", [nodes_row("sel1", "died", 1)])  # the block it was selected on: never a ruler's
        self.arm_a = write_arm(self.champ, "A", A_ROWS)
        self.out = self.root / "ruler"
        self.pool = Pool()
        for p in self.pool.patches():
            p.start()
            self.addCleanup(p.stop)

    def run_main(self, *extra: str) -> tuple[int, str]:
        return arrivals(self.champ, self.out, self.agents, *extra)

    def first(self) -> dict:
        rc, text = self.run_main()
        self.assertEqual(rc, 0, text)
        return a3.load_ruler(self.out)[0]

    def test_a_second_block_is_added_and_the_first_is_not_replayed_again(self):
        m1 = self.first()
        self.assertEqual((m1["starts"], [x["block"] for x in m1["arms"]]), (3, ["A"]))
        before = {r["id"]: r for r in a3.read_ruler_records(self.out)}
        write_arm(self.champ, "B", B_ROWS)
        n = len(self.pool.calls)
        rc, text = self.run_main()
        self.assertEqual(rc, 0, text)
        self.assertEqual(self.pool.calls[n:], [("B", ["b0", "b1"])])  # only the arm the ruler lacked
        m2, lite = a3.load_ruler(self.out)  # the starts hash to the meta's
        self.assertEqual((m2["starts"], m2["first"], m2["second"], [x["block"] for x in m2["arms"]]), (6, 4, 2, ["A", "B"]))
        self.assertNotEqual(m2["ruler_hash"], m1["ruler_hash"])
        after = {r["id"]: r for r in a3.read_ruler_records(self.out)}
        self.assertEqual({i: after[i] for i in before}, before)  # every start the ruler had is kept, document and all
        self.assertEqual(sorted(after), sorted([*before, "B:b0|f48", "B:b0|f49", "B:b1|f48"]))
        self.assertEqual((m2["arrivals"], m2["matched"]), (4, 4))  # the totals cover both arms
        ex = json.loads((self.out / "exclude_runs.json").read_text(encoding="utf-8"))
        self.assertEqual(ex["run_seeds"], ["a0", "a1", "b0", "b1"])
        self.assertEqual(len(ex["start_ids"]), 6)
        self.assertIn('"action": "extended"', text)
        # the same set built in one go has the same hash: the extension is no different from a fresh build of all the arms
        other = self.root / "fresh"
        self.assertEqual(arrivals(self.champ, other, self.agents)[0], 0)
        self.assertEqual(a3.load_ruler(other)[0]["ruler_hash"], m2["ruler_hash"])

    def test_the_meta_keeps_the_history_of_the_builds(self):
        m1 = self.first()
        self.assertEqual([(h["action"], h["ruler_hash"], h["starts"]) for h in m1["history"]], [("built", m1["ruler_hash"], 3)])
        self.assertEqual([(x["block"], x["starts"], x["first"], x["second"]) for x in m1["history"][0]["added"]], [("A", 3, 2, 1)])
        write_arm(self.champ, "B", B_ROWS)
        self.run_main()
        m2 = a3.load_ruler(self.out)[0]
        self.assertEqual([h["action"] for h in m2["history"]], ["built", "extended"])
        last = m2["history"][-1]
        self.assertEqual((last["previous_hash"], last["ruler_hash"], last["starts"]), (m1["ruler_hash"], m2["ruler_hash"], 6))
        self.assertEqual([(x["block"], x["rows"], x["arrivals"], x["starts"]) for x in last["added"]], [("B", 2, 2, 3)])  # the arm it added and what it brought
        self.assertRegex(last["written"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        write_arm(self.champ, "C", C_ROWS)
        self.run_main()
        m3 = a3.load_ruler(self.out)[0]
        self.assertEqual([h["action"] for h in m3["history"]], ["built", "extended", "extended"])
        self.assertEqual(m3["history"][:2], m2["history"])  # the earlier entries are carried unchanged
        self.assertEqual([x["block"] for x in m3["arms"]], ["A", "B", "C"])

    def test_a_ruler_written_before_there_was_a_history_gets_the_one_it_implies(self):
        m1 = self.first()
        old = {k: v for k, v in m1.items() if k != "history"}
        (self.out / "meta.json").write_text(json.dumps(old), encoding="utf-8")  # a meta as the first build lane wrote it
        write_arm(self.champ, "B", B_ROWS)
        self.run_main()
        m2 = a3.load_ruler(self.out)[0]
        self.assertEqual([h["action"] for h in m2["history"]], ["built", "extended"])
        self.assertEqual((m2["history"][0]["ruler_hash"], m2["history"][0]["starts"]), (m1["ruler_hash"], 3))
        self.assertEqual([(x["block"], x["starts"]) for x in m2["history"][0]["added"]], [("A", 3)])

    def test_a_ruler_with_nothing_to_add_is_left_exactly_as_it_is(self):
        m1 = self.first()
        files = {f.name: f.read_bytes() for f in self.out.glob("*") if f.is_file()}
        n = len(self.pool.calls)
        rc, text = self.run_main()
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.pool.calls), n)
        self.assertIn("up to date", text)
        self.assertEqual({f.name: f.read_bytes() for f in self.out.glob("*") if f.is_file()}, files)
        self.assertEqual(a3.load_ruler(self.out)[0]["ruler_hash"], m1["ruler_hash"])
        # up to date needs no machine: a block or a stop does not matter
        self.pool.blocked = "pause"
        self.assertEqual(self.run_main()[0], 0)

    def test_a_block_named_to_leave_out_is_not_added(self):
        self.first()
        write_arm(self.champ, "B", B_ROWS)
        rc, text = self.run_main("--exclude-block", "B")
        self.assertEqual(rc, 0, text)
        self.assertEqual([x["block"] for x in a3.load_ruler(self.out)[0]["arms"]], ["A"])
        self.assertEqual(self.pool.replayed, ["A"])

    def test_an_arm_that_grew_replays_only_its_new_rows(self):
        m1 = self.first()
        before = {r["id"]: r for r in a3.read_ruler_records(self.out)}
        with open(self.arm_a / "runs.jsonl", "a", encoding="utf-8") as fh:  # the block was completed: two more runs, both arrived
            fh.write(json.dumps(nodes_row("a3", "died", 4)) + "\n" + json.dumps(nodes_row("a4", "past", 5)) + "\n")
        n = len(self.pool.calls)
        rc, text = self.run_main()
        self.assertEqual(rc, 0, text)
        self.assertEqual(self.pool.calls[n:], [("A", ["a3", "a4"])])  # a0 and a1 have rows on disk
        m2 = a3.load_ruler(self.out)[0]
        self.assertEqual((m2["starts"], [(x["block"], x["rows"], x["arrivals"]) for x in m2["arms"]]), (6, [("A", 5, 4)]))
        after = {r["id"]: r for r in a3.read_ruler_records(self.out)}
        self.assertEqual({i: after[i] for i in before}, before)
        self.assertEqual([x["block"] for x in m2["history"][-1]["added"]], ["A"])
        self.assertEqual(m2["history"][-1]["added"][0]["starts"], 6)  # the arm's starts as they stand now (an arm is re-read whole when it grew)
        self.assertNotEqual(m2["ruler_hash"], m1["ruler_hash"])

    def test_an_arm_that_grew_with_no_new_arrival_keeps_the_hash(self):
        m1 = self.first()
        with open(self.arm_a / "runs.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(nodes_row("a9", "early", 9)) + "\n")  # died on floor 30: no arrival
        exam = self.out / "exam"
        exam.mkdir()
        (exam / "games-arr-cand-aaaaaaaa.jsonl").write_text(json.dumps({"start": "s", "j": 0}) + "\n", encoding="utf-8")
        n = len(self.pool.calls)
        self.assertEqual(self.run_main()[0], 0)
        self.assertEqual(len(self.pool.calls), n)  # the tasks are the same as before: nothing to replay
        m2 = a3.load_ruler(self.out)[0]
        self.assertEqual((m2["ruler_hash"], m2["arms"][0]["rows"], len(m2["history"])), (m1["ruler_hash"], 4, 2))
        self.assertEqual(m2["history"][-1]["added"][0]["starts"], 3)
        self.assertTrue((exam / "games-arr-cand-aaaaaaaa.jsonl").exists())  # the hash is the same: the games stay

    def test_the_exam_games_of_the_old_hash_are_put_away_and_the_new_hash_is_written_down(self):
        m1 = self.first()
        exam = self.out / "exam"
        exam.mkdir()
        for name in ("games-arr-cand-aaaaaaaa.jsonl", "games-arr-champ-bbbbbbbb.jsonl", "report.txt", "results.json", "log.txt", "STOP"):
            (exam / name).write_text(name, encoding="utf-8")
        write_arm(self.champ, "B", B_ROWS)
        rc, text = self.run_main()
        self.assertEqual(rc, 0, text)
        m2 = a3.load_ruler(self.out)[0]
        shelf = exam / m1["ruler_hash"][:8]
        self.assertEqual(sorted(f.name for f in shelf.iterdir()), ["games-arr-cand-aaaaaaaa.jsonl", "games-arr-champ-bbbbbbbb.jsonl", "report.txt", "results.json"])
        self.assertEqual(sorted(f.name for f in exam.iterdir() if f.is_file()), ["STOP", "log.txt", a3.EXAM_MARK])  # the log and the stop file are no hash's
        self.assertEqual(a3.exam_owner(exam), m2["ruler_hash"])
        self.assertEqual((shelf / "report.txt").read_text(encoding="utf-8"), "report.txt")  # moved whole, not copied over
        self.assertEqual(a3.exam_games(exam, m2["ruler_hash"]), [])  # no game of the new hash yet

    def test_rebuild_ignores_the_ruler_in_the_folder(self):
        m1 = self.first()
        write_arm(self.champ, "B", B_ROWS)
        n = len(self.pool.calls)
        self.assertEqual(self.run_main("--rebuild")[0], 0)
        m2 = a3.load_ruler(self.out)[0]
        self.assertEqual([h["action"] for h in m2["history"]], ["built"])  # a build, not an extension: no earlier history
        self.assertEqual(m2["starts"], 6)
        self.assertEqual(self.pool.calls[n:], [("B", ["b0", "b1"])])  # A's replays are on disk
        self.assertNotEqual(m2["ruler_hash"], m1["ruler_hash"])

    def test_a_ruler_that_does_not_hash_is_built_again_from_the_replays_on_disk(self):
        self.first()
        write_arm(self.champ, "B", B_ROWS)
        self.run_main()
        want = a3.load_ruler(self.out)[0]["ruler_hash"]
        path = self.out / f"{a3.ARRIVAL_STEM}.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace('"floor":48,"hp":61}', '"floor":48,"hp":99}', 1), encoding="utf-8")  # a write that did not finish, or an edit
        self.assertIsNone(a3.prior_ruler(self.out, "gen7-cand", CHAMP_DIR))
        n = len(self.pool.calls)
        rc, text = self.run_main()
        self.assertEqual(rc, 0, text)
        self.assertEqual(len(self.pool.calls), n)  # every replay has its row: nothing is played
        m = a3.load_ruler(self.out)[0]
        self.assertEqual(m["ruler_hash"], want)
        self.assertEqual([h["action"] for h in m["history"]], ["built"])

    def test_a_ruler_of_another_champion_is_not_extended(self):
        self.first()
        meta = json.loads((self.out / "meta.json").read_text(encoding="utf-8"))
        meta["champion"], meta["champion_dir"] = "gen6-other", "gen6/other"
        (self.out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        self.assertIsNone(a3.prior_ruler(self.out, "gen7-cand", CHAMP_DIR))

    def test_a_stop_in_an_extension_leaves_the_ruler_as_it_was_and_the_next_run_finishes_it(self):
        m1 = self.first()
        files = {f.name: f.read_bytes() for f in self.out.glob("*") if f.is_file() and not f.name.startswith("replays-")}
        write_arm(self.champ, "B", B_ROWS)
        self.pool.stop_after = 1  # one of B's two replays is played, then the night window
        rc, text = self.run_main()
        self.assertEqual(rc, 3)
        self.assertIn("stopped (night-window)", text)
        self.assertEqual({f.name: f.read_bytes() for f in self.out.glob("*") if f.is_file() and not f.name.startswith("replays-")}, files)
        self.assertEqual(a3.load_ruler(self.out)[0]["ruler_hash"], m1["ruler_hash"])  # a ruler a test could still be played on
        self.pool.stop_after = None
        n = len(self.pool.calls)
        self.assertEqual(self.run_main()[0], 0)
        self.assertEqual(self.pool.calls[n:], [("B", ["b1"])])  # the replay that was played is kept
        self.assertEqual(a3.load_ruler(self.out)[0]["starts"], 6)
        # a block before anything is replayed
        write_arm(self.champ, "C", C_ROWS)
        self.pool.blocked = "pause"
        n = len(self.pool.calls)
        rc, text = self.run_main()
        self.assertEqual((rc, len(self.pool.calls)), (3, n))
        self.assertIn("not started: pause", text)

    def test_no_arm_to_read_makes_no_folder(self):
        lone = self.agents / "gen8" / "cand"
        lone.mkdir(parents=True)
        (lone / "promotion.json").write_text(json.dumps({"seed_block": "Esel"}), encoding="utf-8")
        write_arm(lone, "Esel", [nodes_row("s", "died", 1)], agent="gen8/cand")  # the block it was selected on is its only arm
        out = self.root / "nowhere" / "ruler"
        rc, text = arrivals(lone, out, self.agents)
        self.assertEqual(rc, 2)
        self.assertIn("no arm to read", text)
        self.assertFalse(out.exists())
        self.assertFalse(out.parent.exists())
        none = self.agents / "gen9" / "cand"  # no arms folder at all
        none.mkdir(parents=True)
        self.assertEqual(arrivals(none, out, self.agents)[0], 2)
        self.assertFalse(out.exists())
        self.assertEqual(self.pool.calls, [])

    def test_arms_another_agent_played_leave_a_ruler_alone(self):
        self.first()
        write_arm(self.champ, "B", B_ROWS, agent="gen9/other")
        files = {f.name: f.read_bytes() for f in self.out.glob("*") if f.is_file()}
        rc, text = self.run_main()
        self.assertEqual(rc, 2)
        self.assertIn("were not played by gen7/cand", text)
        self.assertEqual({f.name: f.read_bytes() for f in self.out.glob("*") if f.is_file()}, files)

    def test_the_gap_names_what_a_ruler_lacks(self):
        self.assertEqual([(g["label"], g["why"], g["rows"]) for g in a3.ruler_gap(None, self.champ)], [("A", "new", 3)])  # no ruler: every eligible arm
        meta = self.first()
        self.assertEqual(a3.ruler_gap(meta, self.champ), [])
        write_arm(self.champ, "B", B_ROWS)
        with open(self.arm_a / "runs.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(nodes_row("a7", "died", 7)) + "\n")
        self.assertEqual([(g["label"], g["why"], g["rows"]) for g in a3.ruler_gap(meta, self.champ)], [("A", "grown", 4), ("B", "new", 2)])
        self.assertEqual([g["label"] for g in a3.ruler_gap(meta, self.champ, ["B"])], ["A"])
        self.assertEqual(a3.ruler_gap(meta, self.agents / "gen9" / "nothing"), [])  # a folder with no arms


class ExamFolderTests(unittest.TestCase):
    """`exam/ruler_hash.txt`: the games in an exam folder belong to one hash."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.exam = Path(self.tmp.name) / "exam"
        self.exam.mkdir()

    def games(self, name: str = "games-arr-cand-aaaaaaaa.jsonl", js=range(6)) -> Path:
        f = self.exam / name
        f.write_text("".join(json.dumps({"start": "s", "j": j}) + "\n" for j in js), encoding="utf-8")
        return f

    def test_a_folder_that_names_no_owner_counts_for_any_hash_and_one_that_names_another_counts_for_none(self):
        f = self.games()
        self.assertEqual(a3.exam_games(self.exam, "h" * 64), [f])
        self.assertEqual(a3.exam_games(self.exam, None), [f])
        (self.exam / a3.EXAM_MARK).write_text("a" * 64 + "\n", encoding="utf-8")
        self.assertEqual(a3.exam_games(self.exam, "a" * 64), [f])
        self.assertEqual(a3.exam_games(self.exam, "b" * 64), [])  # played on another ruler
        self.assertEqual(a3.exam_games(self.exam, None), [f])  # a caller that does not know the hash counts everything: never fewer draws used
        self.assertEqual(a3.exam_games(self.exam / "missing", "a" * 64), [])

    def test_retiring_an_exam_folder(self):
        old, new = "1" * 64, "2" * 64
        f = self.games()
        self.assertEqual(a3.retire_exam(self.exam, old), [])  # a legacy folder: it is this ruler's, the owner is written down and the games stay
        self.assertEqual((a3.exam_owner(self.exam), f.exists()), (old, True))
        self.assertEqual(a3.retire_exam(self.exam, old), [])  # the same hash again changes nothing
        self.assertEqual(a3.retire_exam(self.exam, new), [f.name])  # another hash: the games go to <old first 8>/
        self.assertEqual((a3.exam_owner(self.exam), f.exists(), (self.exam / old[:8] / f.name).exists()), (new, False, True))
        self.assertEqual(a3.retire_exam(self.exam / "missing", new), [])  # no folder: nothing is made
        self.assertFalse((self.exam / "missing").exists())

    def test_an_old_hash_the_caller_names_is_the_owner_of_a_folder_that_names_none(self):
        self.games()
        self.assertEqual(a3.retire_exam(self.exam, "2" * 64, old_hash="1" * 64), ["games-arr-cand-aaaaaaaa.jsonl"])
        self.assertTrue((self.exam / ("1" * 8) / "games-arr-cand-aaaaaaaa.jsonl").exists())

    def test_nothing_is_ever_overwritten_when_the_shelf_is_taken(self):
        old = "1" * 64
        (self.exam / old[:8]).mkdir()
        (self.exam / old[:8] / "games-arr-cand-aaaaaaaa.jsonl").write_text("earlier", encoding="utf-8")
        self.games()
        (self.exam / a3.EXAM_MARK).write_text(old, encoding="utf-8")
        a3.retire_exam(self.exam, "2" * 64)
        self.assertEqual((self.exam / old[:8] / "games-arr-cand-aaaaaaaa.jsonl").read_text(encoding="utf-8"), "earlier")
        self.assertEqual(sorted(f.name for f in (self.exam / old[:8]).iterdir()), ["games-arr-cand-aaaaaaaa.jsonl", "games-arr-cand-aaaaaaaa.jsonl.1"])


class ExamDriverTests(unittest.TestCase):
    """`arrival-exam` plays no game on the games of another hash: they are put away first, then every draw is played."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ruler = self.root / "ruler"
        self.meta = trs.make_ruler(self.ruler, "gen7-cand", CHAMP_DIR, label="B")
        self.cand = player(self.root, "candidate", b"candidate-net")
        self.champ = player(self.root, "champion", b"champion-net")
        self.asked: list[tuple[str, int]] = []
        for p in (mock.patch.object(fe, "play_exam_tasks", self.fake_lockstep), mock.patch.object(fe, "blocked_reason", lambda **kw: None),
                  mock.patch.object(fe, "prepare_process", lambda: None), mock.patch.object(gl, "engine_tag", lambda: "E2"),
                  mock.patch.object(fe, "ALLOW_QUIET", [])):
            p.start()
            self.addCleanup(p.stop)

    def fake_lockstep(self, exam_dir, lock, tasks, target, player_dir, what, *, procs, k, prior="server", judge=None):
        self.asked.append((Path(player_dir).name, len(tasks)))
        with open(target, "a", encoding="utf-8", newline="\n") as fh:
            for _a, sid, j, _cat, _info in tasks:
                fh.write(json.dumps({"start": sid, "j": j, "won": False, "capped": False, "engine": "E2", "config": "a3-v1"}) + "\n")
        return None

    def run_exam(self, exam: Path | None = None) -> int:
        argv = ["arrival-exam", "--ruler", str(self.ruler), "--candidate", str(self.cand), "--champion", str(self.champ), "--procs", "2",
                "--report-dir", str(self.root / "report")]
        if exam is not None:
            argv += ["--out", str(exam)]
        with contextlib.redirect_stdout(io.StringIO()):
            return a3.main(argv)

    def test_games_of_another_hash_in_the_folder_are_put_away_and_not_read_as_played(self):
        exam = self.ruler / "exam"
        exam.mkdir()
        (exam / a3.EXAM_MARK).write_text("9" * 64 + "\n", encoding="utf-8")
        champ_name = f"games-arr-champ-{a3.policy_sha(self.champ)[:8]}.jsonl"
        (exam / champ_name).write_text("".join(json.dumps({"start": sid, "j": j, "won": True, "engine": "E2", "config": "a3-v1"}) + "\n"
                                               for sid in ("B:p1|f48", "B:p1|f49", "B:d1|f48") for j in range(6)), encoding="utf-8")
        self.assertEqual(self.run_exam(), 0)
        self.assertEqual(self.asked, [("champion", 18), ("candidate", 18)])  # all 18 games of the champion are played: its old rows are another ruler's
        self.assertTrue((exam / ("9" * 8) / champ_name).exists())
        self.assertEqual(a3.exam_owner(exam), self.meta["ruler_hash"])
        self.assertEqual(gen_loop.s1_used_tests(self.ruler, {"tests": []}), {0})  # the games just played are this hash's test 0

    def test_a_folder_of_this_hash_is_resumed(self):
        exam = self.ruler / "exam"
        self.assertEqual(self.run_exam(), 0)
        self.assertEqual(a3.exam_owner(exam), self.meta["ruler_hash"])  # the first exam writes the owner down
        n = len(self.asked)
        self.assertEqual(self.run_exam(), 0)
        self.assertEqual(len(self.asked), n)  # every game has its row


class UsedTestsTests(trs.World):
    """`gen_loop.s1_used_tests`: the tests played on the ruler's current hash, and no others."""

    def setUp(self):
        super().setUp()
        self.ruler = self.ruler_root / "gen3-candidate"
        self.meta = trs.make_ruler(self.ruler)
        self.h = self.meta["ruler_hash"]

    def ledger(self, *tests) -> dict:
        return {"rule": gen_loop.S1_LEDGER_RULE, "tests": [{"test": k, "candidate": "c", "champion": "g", "ruler_hash": h, "done": "t"} for k, h in tests]}

    def test_the_ledgers_entries_of_another_hash_are_not_counted(self):
        old = "e" * 64
        self.assertEqual(gen_loop.s1_used_tests(self.ruler, self.ledger((0, old), (1, old), (2, old))), set())  # a new hash starts at test 0
        self.assertEqual(gen_loop.s1_used_tests(self.ruler, self.ledger((0, old), (0, self.h), (1, self.h))), {0, 1})
        self.assertEqual(gen_loop.s1_used_tests(self.ruler, self.ledger((3, None))), {3})  # an entry that names no hash cannot be told apart: it counts

    def test_games_of_another_hash_are_not_counted_and_games_of_this_one_are(self):
        exam = self.ruler / "exam"
        exam.mkdir()
        (exam / "games-arr-cand-abcd1234.jsonl").write_text("".join(json.dumps({"start": "s", "j": j}) + "\n" for j in range(12)), encoding="utf-8")
        self.assertEqual(gen_loop.s1_used_tests(self.ruler, {"tests": []}), {0, 1})  # no owner written: a folder from before, counted whole
        (exam / a3.EXAM_MARK).write_text("e" * 64, encoding="utf-8")
        self.assertEqual(gen_loop.s1_used_tests(self.ruler, {"tests": []}), set())  # played on another hash
        (exam / a3.EXAM_MARK).write_text(self.h, encoding="utf-8")
        self.assertEqual(gen_loop.s1_used_tests(self.ruler, {"tests": []}), {0, 1})
        shelf = exam / ("e" * 8)  # the old hash's games, put away: a record, never counted
        shelf.mkdir()
        (shelf / "games-arr-cand-abcd1234.jsonl").write_text(json.dumps({"start": "s", "j": 40}) + "\n", encoding="utf-8")
        self.assertEqual(gen_loop.s1_used_tests(self.ruler, {"tests": []}), {0, 1})

    def test_a_ruler_with_no_readable_meta_counts_everything(self):
        (self.ruler / "meta.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(gen_loop.s1_used_tests(self.ruler, self.ledger((0, "e" * 64), (4, self.h))), {0, 4})
        self.assertIsNone(gen_loop.ruler_hash_of(self.ruler))
        self.assertEqual(gen_loop.ruler_hash_of(self.root / "nowhere"), None)


class GrowthBase(trs.World):
    """`s1_stage` over a champion that plays more blocks: the ruler is built by the first challenger's judgement and extended by the later ones'. The ruler's builder
    is `act3_starts.py arrivals` itself, run in this process on the fake pool; the exam is the fake of `test_route_s_plumbing`."""

    def setUp(self):
        super().setUp()
        shutil.rmtree(self.champ / "arms")  # World's own arm has no node strings: this champion's arms are the ones below
        write_arm(self.champ, "A", A_ROWS, agent="gen3/candidate")
        self.pool = Pool()
        for p in self.pool.patches():
            p.start()
            self.addCleanup(p.stop)
        self.calls: list[list[str]] = []
        self.codes = {"arrivals": 0, "arrival-exam": 0}
        self.points = (1.8, 0.4)
        self.ruler = self.ruler_root / "gen3-candidate"

    def child(self, cmd, check=False):
        if cmd[2] == "arrivals":
            self.calls.append([str(x) for x in cmd])
            with contextlib.redirect_stdout(io.StringIO()):
                return SimpleNamespace(returncode=a3.main(cmd[2:]))
        if cmd[2] == "arrival-exam":  # what the command does to its folder before it plays, and the games it would leave
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            h = json.loads((Path(cmd[cmd.index("--ruler") + 1]) / "meta.json").read_text(encoding="utf-8"))["ruler_hash"]
            a3.retire_exam(out, h)
            k = int(cmd[cmd.index("--test") + 1])
            (out / f"games-arr-cand-{k:08d}.jsonl").write_text("".join(json.dumps({"start": "s", "j": 6 * k + j}) + "\n" for j in range(6)), encoding="utf-8")
        return trs.S1StageTests.answer(self, cmd)

    def stage(self, cand: Path | None = None, **kw):
        cand = cand or self.cand
        return gen_loop.s1_stage(cand, self.champ, gen_loop.load_agent(cand), gen_loop.load_agent(self.champ), ruler_root=self.ruler_root, run=self.child,
                                 python="py", say=lambda m: None, **kw)

    def sha(self, name: str, sha: str) -> Path:
        return self.swap_candidate(name, sha)

    def run_for(self, cmd, cand_sha):
        rc = self.child(cmd)
        if cmd[2] == "arrival-exam" and rc.returncode == 0:
            report = Path(cmd[cmd.index("--report-dir") + 1]) / "results.json"
            doc = json.loads(report.read_text(encoding="utf-8"))
            doc["candidate"]["policy_sha256"] = cand_sha
            report.write_text(json.dumps(doc), encoding="utf-8")
        return rc


class GrowthStageTests(GrowthBase):
    """The ruler's growth through `s1_stage`."""

    def test_the_first_challenger_builds_the_ruler_and_the_next_one_judged_after_a_new_block_extends_it(self):
        r1 = self.stage()
        self.assertEqual([c[2] for c in self.calls], ["arrivals", "arrival-exam"])
        self.assertEqual(self.pool.replayed, ["A"])
        h1 = r1["ruler_hash"]
        self.assertEqual((r1["test"], a3.load_ruler(self.ruler)[0]["first"]), (0, 2))  # two first-boss starts
        # the champion plays a block as the base of the next judgement
        write_arm(self.champ, "B", B_ROWS, agent="gen3/candidate")
        c2 = self.sha("c2", "sha-c2")
        self.calls.clear()
        r2 = gen_loop.s1_stage(c2, self.champ, gen_loop.load_agent(c2), gen_loop.load_agent(self.champ), ruler_root=self.ruler_root,
                               run=lambda cmd, check=False: self.run_for(cmd, "sha-c2"), python="py", say=lambda m: None, block="Ecur")
        self.assertEqual([c[2] for c in self.calls], ["arrivals", "arrival-exam"])
        build = self.calls[0]
        self.assertEqual(build[build.index("--exclude-block") + 1], "Ecur")  # the block being judged is left out
        self.assertEqual(build[build.index("--out") + 1], str(self.ruler))
        self.assertEqual(self.pool.replayed, ["A", "B"])  # A was not replayed again
        meta = a3.load_ruler(self.ruler)[0]
        self.assertEqual((meta["starts"], [x["block"] for x in meta["arms"]]), (6, ["A", "B"]))
        h2 = meta["ruler_hash"]
        self.assertNotEqual(h2, h1)
        self.assertEqual((r2["ruler_hash"], meta["first"]), (h2, 4))  # read on the whole ruler: the first-boss starts of both arms
        self.assertEqual(r2["test"], 0)  # a new hash starts at test 0 with draws of its own
        ledger = gen_loop.s1_ledger(self.ruler)
        self.assertEqual([(t["test"], t["ruler_hash"] == h1) for t in ledger["tests"]], [(0, True), (0, False)])  # the old hash's test stays in the ledger
        self.assertEqual(gen_loop.s1_used_tests(self.ruler, ledger), {0})  # and is not counted on the new one
        exam = self.ruler / "exam"
        self.assertEqual([f.name for f in (exam / h1[:8]).iterdir()], ["games-arr-cand-00000000.jsonl"])  # the old hash's games are put away
        self.assertEqual(a3.exam_owner(exam), h2)
        # a third challenger on the same ruler takes the next test
        c3 = self.sha("c3", "sha-c3")
        self.calls.clear()
        r3 = gen_loop.s1_stage(c3, self.champ, gen_loop.load_agent(c3), gen_loop.load_agent(self.champ), ruler_root=self.ruler_root,
                               run=lambda cmd, check=False: self.run_for(cmd, "sha-c3"), python="py", say=lambda m: None)
        self.assertEqual([c[2] for c in self.calls], ["arrival-exam"])  # nothing to add
        self.assertEqual((r3["test"], r3["ruler_hash"]), (1, h2))

    def test_a_reading_that_stands_is_not_played_again_and_the_ruler_is_not_grown_under_it(self):
        first = self.stage()
        write_arm(self.champ, "B", B_ROWS, agent="gen3/candidate")  # a v3 extension block of the same judgement finds the champion's first block now
        self.calls.clear()
        again = self.stage(block="Eext")
        self.assertEqual(self.calls, [])  # no ruler work and no exam
        self.assertEqual((again["mean"], again["ruler_hash"], again["test"]), (first["mean"], first["ruler_hash"], 0))
        self.assertEqual([x["block"] for x in a3.load_ruler(self.ruler)[0]["arms"]], ["A"])

    def test_a_reading_on_the_old_hash_is_not_applied_to_the_new_one(self):
        first = self.stage()
        write_arm(self.champ, "B", B_ROWS, agent="gen3/candidate")
        c2 = self.sha("c2", "sha-c2")
        gen_loop.s1_stage(c2, self.champ, gen_loop.load_agent(c2), gen_loop.load_agent(self.champ), ruler_root=self.ruler_root,
                          run=lambda cmd, check=False: self.run_for(cmd, "sha-c2"), python="py", say=lambda m: None)
        h2 = a3.load_ruler(self.ruler)[0]["ruler_hash"]
        self.assertNotEqual(first["ruler_hash"], h2)
        old = json.loads((self.cand / "s1.json").read_text(encoding="utf-8"))
        agents = (gen_loop.load_agent(self.cand), gen_loop.load_agent(self.champ))
        got = gen_loop.s1_expect(*agents, h2)
        self.assertFalse(k2.s1_match(old, got)["verified"])  # a reading names its ruler's hash, and this is not it
        self.assertIsNone(gen_loop.standing_reading(self.cand / "s1.json", got))
        # and the keep rule does not apply it
        path = self.root / "old-results.json"
        path.write_text(json.dumps(trs.result(1.8, 0.4, ruler=old["ruler_hash"], cand=trs.CAND_SHA, champ=trs.CHAMP_SHA)), encoding="utf-8")
        reading = gen_loop.s1_from_file(path, self.cand, self.champ, *agents, ruler_root=self.ruler_root, register=False)
        self.assertFalse(reading["match"]["verified"])
        self.assertFalse(k2.route_s1(reading, fight_swap=True)["holds"])

    def test_the_block_being_judged_is_not_in_the_ruler(self):
        write_arm(self.champ, "B", B_ROWS, agent="gen3/candidate")
        self.stage(block="B")
        self.assertEqual([x["block"] for x in a3.load_ruler(self.ruler)[0]["arms"]], ["A"])
        self.assertEqual(self.calls[0][self.calls[0].index("--exclude-block") + 1], "B")

    def test_the_command_without_a_block_names_none_and_parses_under_the_drivers_own_parser(self):
        self.stage()
        self.assertNotIn("--exclude-block", self.calls[0])
        self.calls.clear()
        write_arm(self.champ, "B", B_ROWS, agent="gen3/candidate")
        self.stage(self.sha("c2", "sha-c2"), block="Ecur")
        build = next(c for c in self.calls if c[2] == "arrivals")
        seen = {}

        def capture(args):
            seen["args"] = args
            return 0

        with mock.patch.object(a3, "cmd_arrivals", capture):
            self.assertEqual(a3.main(build[2:]), 0)
        self.assertEqual((seen["args"].exclude_block, seen["args"].allow_quiet, seen["args"].rebuild, seen["args"].procs), (["Ecur"], ["lab-train"], False, 12))

    def test_a_champion_with_nothing_to_read_makes_no_folder_and_plays_no_child(self):
        shutil.rmtree(self.champ / "arms")
        write_arm(self.champ, "Esel", A_ROWS, agent="gen3/candidate")
        (self.champ / "promotion.json").write_text(json.dumps({"seed_block": "Esel"}), encoding="utf-8")  # its only arm is the block it was selected on
        with self.assertRaisesRegex(gen_loop.S1Unavailable, "no arm outside the blocks it was selected on"):
            self.stage()
        self.assertEqual(self.calls, [])
        self.assertFalse(self.ruler.exists())
        self.assertFalse(self.ruler_root.exists())
        shutil.rmtree(self.champ / "arms")  # and no arms at all
        with self.assertRaises(gen_loop.S1Unavailable):
            self.stage()
        self.assertFalse(self.ruler_root.exists())

    def test_the_only_arm_being_the_block_judged_now_is_nothing_to_read(self):
        with self.assertRaises(gen_loop.S1Unavailable):
            self.stage(block="A")
        self.assertEqual(self.calls, [])
        self.assertFalse(self.ruler_root.exists())

    def test_the_same_network_is_refused_before_any_ruler_is_built(self):
        twin = self.swap_candidate("twin", trs.CHAMP_SHA)
        with self.assertRaises(gen_loop.S1Unavailable):
            self.stage(twin)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.ruler_root.exists())

    def test_a_ruler_that_cannot_be_extended_is_unavailable_not_silently_thin(self):
        self.stage()
        write_arm(self.champ, "B", B_ROWS, agent="gen3/candidate")
        c2 = self.sha("c2", "sha-c2")
        self.codes["arrivals"] = 2
        with mock.patch.object(a3, "main", lambda argv: 2), self.assertRaisesRegex(gen_loop.S1Unavailable, "cannot be extended"):
            self.stage(c2)
        with mock.patch.object(a3, "main", lambda argv: 3), self.assertRaises(gen_loop.S1Blocked):
            self.stage(c2)


class GuardCellTests(trs.World):
    """The ledger row says why route S did not apply when it was asked and S1 had no reading."""

    def setUp(self):
        super().setUp()
        self.seeds, self.c, self.b = block(trs.FLATISH, floor_mean=0.0, prefix="K", seed=7)
        self.f1 = self.seed_file("Etest-r1", self.seeds)

    def promote(self, cand=None, *, stage=None, **kw):
        arms, _ = self.stub_arms(self.c, self.b)

        def played(cand_, base_dir, cand_agent, base_agent, **k):
            return trs.holds(self.expect(), 1.8, 0.4)

        with mock.patch.object(gen_loop, "run_arm", arms), mock.patch.object(gen_loop, "s1_stage", stage or played):
            return gen_loop.promote(cand or self.cand, self.champ, self.f1, agent_root=self.agents, boot=500, keep_rule="v3", sequential=True, **kw)

    def guard(self) -> str:
        return gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]["guard"]

    def test_s1_unavailable_is_in_the_row_with_its_reason(self):
        def none(*a, **k):
            raise gen_loop.S1Unavailable("the champion gen3/candidate has no arm outside the blocks it was selected on to build a ruler from")

        doc = self.promote(route_s="auto", stage=none)
        self.assertIn("S1 unavailable: the champion gen3/candidate has no arm outside the blocks it was selected on to build a ruler from; v3 decides", self.guard())
        self.assertIn("no arm outside the blocks", doc["route_s_mode"]["reason"])
        self.assertIn("S1 could not be played", doc["route_s_mode"]["why"])  # the older wording is kept
        self.assertNotIn("\t", self.guard())

    def test_a_candidate_that_is_not_a_fight_swap_says_so(self):
        plain = self.swap_candidate("plain", "sha-p", swap=None)
        doc = self.promote(plain, route_s="auto")
        self.assertIn("S1 unavailable: not a fight-only swap; v3 decides", self.guard())
        self.assertEqual(doc["route_s_mode"]["reason"], "not a fight-only swap")

    def test_a_reading_prints_its_own_words_and_the_flag_off_prints_none(self):
        self.promote(route_s="auto")  # S1 played and held
        self.assertNotIn("unavailable", self.guard())
        self.assertIn("S1 +1.80", self.guard())
        self.setUp()
        self.promote()  # route S not asked
        self.assertNotIn("S1", self.guard())
        self.setUp()
        self.promote(route_s="off")
        self.assertNotIn("S1", self.guard())

    def test_the_note_is_only_for_a_judgement_that_asked_and_had_no_reading(self):
        self.assertIsNone(gen_loop.route_s_note(None, None))
        self.assertIsNone(gen_loop.route_s_note({"mode": "auto", "applies": True}, {"mean": 1.0}))
        self.assertIsNone(gen_loop.route_s_note({"mode": "auto", "applies": False}, {"mean": 0.2}))  # read, did not hold: keep_v2's own words
        self.assertEqual(gen_loop.route_s_note({"mode": "auto", "applies": False, "why": "w", "reason": "r"}, None), "S1 unavailable: r; v3 decides")
        self.assertEqual(gen_loop.route_s_note({"mode": "auto", "applies": False, "why": "w"}, None), "S1 unavailable: w; v3 decides")

    def test_a_rejudgement_with_no_reading_says_so_too(self):
        self.write_arm(self.cand, "Etest-r1", self.c)
        self.write_arm(self.champ, "Etest-r1", self.b)
        doc = gen_loop.rejudge(self.cand, self.champ, self.f1, agent_root=self.agents, note="D68", boot=500, keep_rule="v3", route_s="auto", out=self.root / "dry",
                               ruler_root=self.ruler_root)
        row = gen_loop.read_promotions(self.root / "dry" / "promotions.tsv")[-1]
        self.assertIn("S1 unavailable: no S1 reading (no --s1 file, no s1.json in the candidate's folder); v3 decides", row["guard"])
        self.assertEqual(doc["route_s_mode"]["applies"], False)


# --------------------------------------------------------------------------------------------------------------------------- step 2: the exclusions


def numbered_ruler(folder: Path, name: str, n: int, label: str = "L") -> dict:
    """A ruler of `n` runs that each went past the first boss (two starts a run), for the champion `gen20/candidate-chain`'s folder name."""
    rows = {f"{label}{i}": nodes_row(f"{label}{i}", "past", i) for i in range(n)}
    recs, counts = a3.arrival_records([replay_row(r, label) for r in rows.values()], label, rows, "old")
    return a3.build_ruler(folder, name, "gen20/candidate-chain", [{"label": label, "records": recs, "counts": counts, "rows": n, "arrivals": n}], "old", "a3-v1")


def grow_ruler(folder: Path, n: int, label: str = "M") -> dict:
    """The ruler in `folder` extended by an arm of `n` runs that each went past the first boss: what `arrivals` leaves after an extension."""
    old = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    records = a3.read_ruler_records(folder)
    rows = {f"{label}{i}": nodes_row(f"{label}{i}", "past", i) for i in range(n)}
    recs, counts = a3.arrival_records([replay_row(r, label) for r in rows.values()], label, rows, "old")
    arms = [a3.carried_arm(e, records) for e in old["arms"]] + [{"label": label, "records": recs, "counts": counts, "rows": n, "arrivals": n, "added": True}]
    return a3.build_ruler(folder, old["champion"], old["champion_dir"], arms, "old", "a3-v1", previous=old)


def exam_exclusions(run_seeds, start_ids) -> dict:
    """The fight loop's `exam/exclude_runs.json` in the shape the exam writes it."""
    return {"rule": "fight-exam-v1", "run_seeds": sorted(run_seeds), "start_ids": sorted(start_ids), "starts": 400, "by_stratum": {"own": 300, "human": 100},
            "human_stratum": {"rule": "fight-exam-human-v1", "written": "2026-11-10T00:00:00Z", "starts": 100, "runs": 100}}


def write_hand(path: Path, doc: dict) -> bytes:
    """A file as the hand script of 2026-11-22 wrote it: indent 1, no ASCII escaping, no final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(doc, indent=1, ensure_ascii=False).encode("utf-8")
    path.write_bytes(raw)
    return raw


def hand_merge(e: dict, ruler: Path) -> dict:
    """What `judge-design/merge_arrival_exclusions.py` did to the exam's list (its algorithm, for the fixture of its result)."""
    r = json.loads((ruler / "exclude_runs.json").read_text(encoding="utf-8"))
    meta = json.loads((ruler / "meta.json").read_text(encoding="utf-8"))
    es, ei = set(e["run_seeds"]), set(e["start_ids"])
    new_s, new_i = sorted(set(r["run_seeds"]) - es), sorted(set(r["start_ids"]) - ei)
    e["run_seeds"], e["start_ids"] = sorted(es | set(new_s)), sorted(ei | set(new_i))
    e["by_stratum"]["arrival-own"] = int(e["by_stratum"].get("arrival-own", 0)) + len(new_i)
    e.setdefault("arrival", {})[ruler.name] = {"rule": r["rule"], "written": "2026-11-22T09:56:00Z", "starts": len(r["start_ids"]), "runs": len(r["run_seeds"]),
                                               "added_starts": len(new_i), "added_runs": len(new_s), "ruler_hash": meta["ruler_hash"],
                                               "why": "the Act 3 arrival exam (route S, S1) plays these starts; the practice must not draw them"}
    return e


class MergeTests(unittest.TestCase):
    """`act3_starts.merge_exclusions`: the ruler's runs and starts into the fight loop's list."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.loop = Path(self.tmp.name) / "gym-loop"
        self.ruler = self.loop / "arrival" / "gen20-candidate-chain"
        self.exam = self.loop / "exam" / "exclude_runs.json"
        self.backup = self.exam.with_name("exclude_runs.json.pre-arrival.bak")
        self.meta = numbered_ruler(self.ruler, "gen20-candidate-chain", 30)  # 30 runs, 60 starts
        self.before = exam_exclusions([f"x{i}" for i in range(50)], [f"x{i}|f40" for i in range(80)])
        self.raw = write_hand(self.exam, self.before)

    def doc(self) -> dict:
        return json.loads(self.exam.read_text(encoding="utf-8"))

    def test_the_union_is_sorted_and_the_entry_and_the_count_say_what_was_added(self):
        rec = a3.merge_exclusions(self.ruler)
        self.assertEqual((rec["status"], rec["wrote"], rec["added_starts"], rec["added_runs"]), ("merged", True, 60, 30))
        e = self.doc()
        self.assertEqual(len(e["run_seeds"]), 80)
        self.assertEqual(len(e["start_ids"]), 140)
        self.assertEqual(e["run_seeds"], sorted(e["run_seeds"]))
        self.assertEqual(e["start_ids"], sorted(e["start_ids"]))
        self.assertTrue({f"L{i}" for i in range(30)} <= set(e["run_seeds"]))
        self.assertTrue({"L:L0|f48", "L:L0|f49"} <= set(e["start_ids"]))
        self.assertEqual(e["by_stratum"], {"own": 300, "human": 100, "arrival-own": 60})
        entry = e["arrival"]["gen20-candidate-chain"]
        self.assertEqual({k: entry[k] for k in ("rule", "starts", "runs", "added_starts", "added_runs", "ruler_hash")},
                         {"rule": "act3-arrival-v1", "starts": 60, "runs": 30, "added_starts": 60, "added_runs": 30, "ruler_hash": self.meta["ruler_hash"]})
        self.assertRegex(entry["written"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        for key in ("rule", "starts", "human_stratum"):  # nothing else of the exam's file is touched
            self.assertEqual(e[key], self.before[key])

    def test_the_old_file_is_kept_once_and_no_temporary_file_is_left(self):
        a3.merge_exclusions(self.ruler)
        self.assertEqual(self.backup.read_bytes(), self.raw)
        self.assertEqual(sorted(f.name for f in self.exam.parent.iterdir()), ["exclude_runs.json", "exclude_runs.json.pre-arrival.bak"])
        grow_ruler(self.ruler, 5)  # a later merge does not replace the backup
        self.assertEqual(a3.merge_exclusions(self.ruler)["status"], "merged")
        self.assertEqual(self.backup.read_bytes(), self.raw)

    def test_a_ruler_already_merged_changes_nothing_and_writes_nothing(self):
        a3.merge_exclusions(self.ruler)
        once = self.exam.read_bytes()
        stamp = self.exam.stat().st_mtime_ns
        rec = a3.merge_exclusions(self.ruler)
        self.assertEqual((rec["status"], rec["wrote"], rec["added_starts"], rec["added_runs"]), ("already merged", False, 0, 0))
        self.assertEqual((self.exam.read_bytes(), self.exam.stat().st_mtime_ns), (once, stamp))  # idempotent: not even rewritten

    def test_what_the_hand_script_left_on_the_22nd_reads_as_already_merged(self):
        """The gym's list as the hand merge left it for generation 20's ruler: 3,712 run seeds and 4,229 start ids, `arrival.gen20-candidate-chain`."""
        base = exam_exclusions([f"x{i}" for i in range(3712 - 30)], [f"x{i}|f40" for i in range(4229 - 60)])
        merged = hand_merge(base, self.ruler)
        self.assertEqual((len(merged["run_seeds"]), len(merged["start_ids"])), (3712, 4229))
        raw = write_hand(self.exam, merged)
        self.backup.unlink(missing_ok=True)
        rec = a3.merge_exclusions(self.ruler)
        self.assertEqual((rec["status"], rec["wrote"]), ("already merged", False))
        self.assertEqual(self.exam.read_bytes(), raw)  # not a byte changed
        e = self.doc()
        self.assertEqual((len(e["run_seeds"]), len(e["start_ids"]), len(set(e["run_seeds"])), len(set(e["start_ids"]))), (3712, 4229, 3712, 4229))  # no duplicates
        self.assertEqual(e["by_stratum"]["arrival-own"], 60)
        self.assertFalse(self.backup.exists())  # this wrote nothing: the hand run's own backup is the one that stands

    def test_a_ruler_that_grew_adds_only_its_new_ids_and_updates_its_entry(self):
        write_hand(self.exam, hand_merge(self.before, self.ruler))
        old = json.loads((self.ruler / "meta.json").read_text(encoding="utf-8"))
        new = grow_ruler(self.ruler, 10)
        self.assertNotEqual(new["ruler_hash"], old["ruler_hash"])
        rec = a3.merge_exclusions(self.ruler)
        self.assertEqual((rec["status"], rec["added_starts"], rec["added_runs"]), ("merged", 20, 10))
        e = self.doc()
        self.assertEqual((len(e["run_seeds"]), len(e["start_ids"])), (50 + 30 + 10, 80 + 60 + 20))
        self.assertEqual(e["by_stratum"]["arrival-own"], 80)  # the starts merged, each once
        entry = e["arrival"]["gen20-candidate-chain"]
        self.assertEqual((entry["starts"], entry["runs"], entry["added_starts"], entry["added_runs"], entry["ruler_hash"]), (80, 40, 20, 10, new["ruler_hash"]))
        self.assertEqual(a3.merge_exclusions(self.ruler)["status"], "already merged")

    def test_the_practice_reads_the_merged_list(self):
        import practice_mix as pmix  # noqa: PLC0415

        start = {"id": "L:L0|f48", "seed": "L0", "run": "L0"}
        self.assertFalse(pmix.excluded(start, pmix.read_exclusion(self.loop)))
        a3.merge_exclusions(self.ruler)
        ex = pmix.read_exclusion(self.loop)
        self.assertTrue(pmix.excluded(start, ex))
        self.assertTrue(pmix.excluded({"id": "other|f48", "seed": "L7", "run": "L7"}, ex))  # a start of the same run, by its run
        self.assertTrue({"x0", "x49"} <= set(ex))  # the old ids are still there

    def test_a_loop_with_no_exclusion_file_gets_none_made(self):
        self.exam.unlink()
        rec = a3.merge_exclusions(self.ruler)
        self.assertEqual((rec["status"], rec["wrote"]), ("no exclusion file", False))
        self.assertFalse(self.exam.exists())

    def test_a_file_that_is_not_the_rules_own_is_never_rewritten(self):
        raw = write_hand(self.exam, {**self.before, "rule": "something-else"})
        with self.assertRaisesRegex(ValueError, "not the fight exam's file"):
            a3.merge_exclusions(self.ruler)
        self.assertEqual(self.exam.read_bytes(), raw)
        self.exam.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            a3.merge_exclusions(self.ruler)
        self.assertEqual(self.exam.read_text(encoding="utf-8"), "{not json")
        self.assertFalse(self.backup.exists())

    def test_a_ruler_list_that_does_not_match_its_meta_is_not_merged(self):
        r = json.loads((self.ruler / "exclude_runs.json").read_text(encoding="utf-8"))
        r["start_ids"] = r["start_ids"][:-1]
        (self.ruler / "exclude_runs.json").write_text(json.dumps(r), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "start ids and its meta"):
            a3.merge_exclusions(self.ruler)
        r["rule"] = "other"
        (self.ruler / "exclude_runs.json").write_text(json.dumps(r), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "names rule 'other'"):
            a3.merge_exclusions(self.ruler)
        self.assertEqual(self.exam.read_bytes(), self.raw)

    def test_a_write_that_fails_leaves_the_file_as_it_was(self):
        with mock.patch("os.replace", side_effect=OSError("disk full")), self.assertRaises(OSError):
            a3.merge_exclusions(self.ruler)
        self.assertEqual(self.exam.read_bytes(), self.raw)  # the list is replaced whole or not at all
        for f in self.exam.parent.glob("*.tmp*"):
            f.unlink()
        wrote = []
        real = os.replace

        def spy(src, dst, *a, **k):
            wrote.append((Path(src).parent == Path(dst).parent, Path(src).name != Path(dst).name))
            return real(src, dst, *a, **k)

        with mock.patch("os.replace", spy):
            a3.merge_exclusions(self.ruler)
        self.assertEqual(wrote, [(True, True)])  # a temporary file beside it, then replaced over it

    def test_a_dry_run_says_what_it_would_do_and_writes_nothing(self):
        rec = a3.merge_exclusions(self.ruler, dry=True)
        self.assertEqual((rec["status"], rec["wrote"], rec["added_starts"]), ("merged", False, 60))
        self.assertEqual(self.exam.read_bytes(), self.raw)
        self.assertFalse(self.backup.exists())

    def test_the_command(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = a3.main(["merge-exclusions", "--ruler", str(self.ruler), "--exam-file", str(self.exam), "--dry"])
        self.assertEqual((rc, json.loads(buf.getvalue())["added_starts"]), (0, 60))
        self.assertEqual(self.exam.read_bytes(), self.raw)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(a3.main(["merge-exclusions", "--ruler", str(self.ruler)]), 0)  # the default file: the loop's, beside the ruler's folder
        self.assertEqual(self.doc()["by_stratum"]["arrival-own"], 60)
        self.exam.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(a3.main(["merge-exclusions", "--ruler", str(self.ruler)]), 2)
        write_hand(self.exam, {**self.before, "rule": "x"})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(a3.main(["merge-exclusions", "--ruler", str(self.ruler)]), 2)
        self.assertIn("not merged", out.getvalue())

    def test_the_fight_exams_rule_is_the_one_named_here(self):
        self.assertEqual(a3.EXAM_EXCLUDE_RULE, fe.EXAM_RULE)

    def test_the_loops_file_is_found_beside_the_rulers_folder(self):
        self.assertEqual(a3.exam_exclusion_file(self.ruler), self.exam)
        self.assertEqual(a3.exam_exclusion_file(a3.ARRIVAL_ROOT / "gen21-candidate-fight-v5"), a3.ARRIVAL_ROOT.parent / "exam" / "exclude_runs.json")


class ExclusionStageTests(GrowthBase):
    """The S1 stage and the readers of a ruler merge its exclusions by themselves."""

    def setUp(self):
        super().setUp()
        self.exam = self.root / "exam" / "exclude_runs.json"  # beside the ruler root's folder, as the loop's is
        self.backup = self.exam.with_name("exclude_runs.json.pre-arrival.bak")
        self.before = exam_exclusions([f"x{i}" for i in range(50)], [f"x{i}|f40" for i in range(80)])
        self.raw = write_hand(self.exam, self.before)
        self.said: list[str] = []

    def stage_say(self, cand=None, **kw):
        cand = cand or self.cand
        return gen_loop.s1_stage(cand, self.champ, gen_loop.load_agent(cand), gen_loop.load_agent(self.champ), ruler_root=self.ruler_root, run=self.child,
                                 python="py", say=self.said.append, **kw)

    def doc(self) -> dict:
        return json.loads(self.exam.read_text(encoding="utf-8"))

    def test_a_ruler_built_in_the_stage_is_on_the_list_before_the_exam_plays(self):
        listed = []
        real = trs.S1StageTests.answer

        def watch(self_, cmd, check=False):
            if cmd[2] == "arrival-exam":
                listed.append("arrival" in self.doc())
            return real(self_, cmd)

        with mock.patch.object(trs.S1StageTests, "answer", watch):
            self.stage_say()
        self.assertEqual(listed, [True])  # the practice's list held the ruler when the exam was asked for its first game
        e = self.doc()
        meta = a3.load_ruler(self.ruler)[0]
        entry = e["arrival"]["gen3-candidate"]
        self.assertEqual((entry["ruler_hash"], entry["starts"], entry["runs"]), (meta["ruler_hash"], 3, 2))
        self.assertEqual((len(e["run_seeds"]), len(e["start_ids"]), e["by_stratum"]["arrival-own"]), (52, 83, 3))
        self.assertEqual(self.backup.read_bytes(), self.raw)
        self.assertTrue(any("on the practice's exclusion list now" in m for m in self.said))

    def test_a_ruler_that_grew_is_merged_again_and_one_that_did_not_is_left_alone(self):
        self.stage_say()
        write_arm(self.champ, "B", B_ROWS, agent="gen3/candidate")
        c2 = self.sha("c2", "sha-c2")
        gen_loop.s1_stage(c2, self.champ, gen_loop.load_agent(c2), gen_loop.load_agent(self.champ), ruler_root=self.ruler_root,
                          run=lambda cmd, check=False: self.run_for(cmd, "sha-c2"), python="py", say=self.said.append)
        e = self.doc()
        meta = a3.load_ruler(self.ruler)[0]
        self.assertEqual((e["arrival"]["gen3-candidate"]["ruler_hash"], e["arrival"]["gen3-candidate"]["starts"]), (meta["ruler_hash"], 6))
        self.assertEqual((len(e["run_seeds"]), len(e["start_ids"]), e["by_stratum"]["arrival-own"]), (50 + 4, 80 + 6, 6))
        once = self.exam.read_bytes()
        c3 = self.sha("c3", "sha-c3")
        gen_loop.s1_stage(c3, self.champ, gen_loop.load_agent(c3), gen_loop.load_agent(self.champ), ruler_root=self.ruler_root,
                          run=lambda cmd, check=False: self.run_for(cmd, "sha-c3"), python="py", say=self.said.append)
        self.assertEqual(self.exam.read_bytes(), once)  # nothing to add: not rewritten
        self.assertEqual(self.backup.read_bytes(), self.raw)

    def test_a_ruler_built_by_hand_is_merged_the_first_time_a_stage_reads_it(self):
        self.assertEqual(arrivals(self.champ, self.ruler, self.agents)[0], 0)  # outside the train job
        self.assertNotIn("arrival", self.doc())
        self.stage_say()
        self.assertIn("gen3-candidate", self.doc()["arrival"])
        # also when the candidate's reading stands and nothing is played
        self.exam.write_bytes(self.raw)
        self.calls.clear()
        self.stage_say()
        self.assertEqual(self.calls, [])
        self.assertIn("gen3-candidate", self.doc()["arrival"])

    def test_a_list_that_cannot_be_merged_stops_the_stage_before_the_exam_plays(self):
        raw = write_hand(self.exam, {**self.before, "rule": "something-else"})
        with self.assertRaisesRegex(ValueError, "not the fight exam's file"):
            self.stage_say()
        self.assertEqual([c[2] for c in self.calls], ["arrivals"])  # the ruler was built; no game was asked for
        self.assertEqual(self.exam.read_bytes(), raw)

    def test_a_loop_with_no_list_is_said_and_the_stage_goes_on(self):
        self.exam.unlink()
        self.exam.parent.rmdir()
        self.stage_say()
        self.assertFalse(self.exam.parent.exists())  # no file or folder is made for it
        self.assertTrue(any("no exclusion file" in m for m in self.said))
        self.assertEqual([c[2] for c in self.calls], ["arrivals", "arrival-exam"])

    def test_a_reading_read_from_a_file_merges_a_hand_built_ruler_and_never_stops_for_a_bad_list(self):
        self.assertEqual(arrivals(self.champ, self.ruler, self.agents)[0], 0)
        meta = a3.load_ruler(self.ruler)[0]
        path = self.root / "s1-results.json"
        path.write_text(json.dumps(trs.result(1.8, 0.4, ruler=meta["ruler_hash"], cand=trs.CAND_SHA, champ=trs.CHAMP_SHA)), encoding="utf-8")
        agents = (gen_loop.load_agent(self.cand), gen_loop.load_agent(self.champ))
        gen_loop.s1_from_file(path, self.cand, self.champ, *agents, ruler_root=self.ruler_root, register=False)  # a dry run writes nothing anywhere
        self.assertNotIn("arrival", self.doc())
        reading = gen_loop.s1_from_file(path, self.cand, self.champ, *agents, ruler_root=self.ruler_root)
        self.assertTrue(reading["match"]["verified"])
        self.assertEqual(self.doc()["arrival"]["gen3-candidate"]["ruler_hash"], meta["ruler_hash"])
        raw = write_hand(self.exam, {**self.before, "rule": "something-else"})  # a list that is not the exam's: said, and the reading is still read
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            reading = gen_loop.s1_from_file(path, self.cand, self.champ, *agents, ruler_root=self.ruler_root)
        self.assertTrue(reading["match"]["verified"])
        self.assertIn("could not be put on the practice's exclusion list", buf.getvalue())
        self.assertEqual(self.exam.read_bytes(), raw)

    def test_the_sync_says_nothing_when_nothing_changed(self):
        self.stage_say()
        self.said.clear()
        rec = gen_loop.sync_ruler_exclusions(self.ruler, self.said.append)
        self.assertEqual((rec["status"], self.said), ("already merged", []))


if __name__ == "__main__":
    unittest.main()
