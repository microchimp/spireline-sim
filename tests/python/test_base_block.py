"""The base block (D68 addendum 7 item 2, lane 25): a champion with no judged arm outside the blocks it was selected on plays one before S1 is read.
`rl/gen_loop.py` (`base_block_of`, `BaseBlock`, `base_block_name`, `base_block_seeds`, `play_base_block`, `base_block_record`, `base_block_note`,
`s1_stage(base_block=, play=)`, `promote(base_block=)`, `run_promote_subprocess(base_block=)`, `resolve_e_block`) and rl/gen0.py (`--base-block`);
docs/rl/KEEP_TEST_V3.md "A fresh champion".

Nothing plays a game: the block's player (`run_arm`) is a function that writes the rows a played block would, the ruler's builder is `act3_starts.py arrivals`
itself on the fake replay pool and the exam is the fake of `test_route_s_plumbing`, as in `test_ruler_growth`."""

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
import gen0  # noqa: E402
import gen_loop  # noqa: E402
import test_ruler_growth as trg  # noqa: E402
import test_route_s_plumbing as trs  # noqa: E402
from test_arrival_exam import nodes_row  # noqa: E402
from test_keep_v3 import block  # noqa: E402

from tools.lab import keep_v2 as k2  # noqa: E402

JUDGED = "E4-2026-11-25"  # the block being judged: its base block is E4-2026-11-25-base
NAME = "E4-2026-11-25-base"
FATES = ("died", "past", "early")  # six runs: four arrive at the first Act 3 boss (two of them also at the second), two do not


class NamingTests(unittest.TestCase):
    def test_the_flag_is_auto_or_off_and_auto_by_default(self):
        self.assertEqual([gen_loop.base_block_of(v) for v in (None, "", "auto", "AUTO", " auto ", "off", "OFF")], ["auto", "auto", "auto", "auto", "auto", "off", "off"])
        self.assertEqual(gen_loop.BASE_BLOCK_MODES, ("auto", "off"))
        with self.assertRaisesRegex(ValueError, "unknown --base-block"):
            gen_loop.base_block_of("on")

    def test_a_base_block_is_two_thousand_runs_named_for_the_judged_block(self):
        self.assertEqual(gen_loop.BASE_BLOCK_RUNS, 2000)
        self.assertEqual(gen_loop.BaseBlock(name=NAME, seeds_dir=Path("s")).n, 2000)
        self.assertEqual(gen_loop.base_block_name(4, JUDGED), NAME)
        self.assertEqual(gen_loop.base_block_name(4, JUDGED + "-r3"), NAME)  # a retry or an extension block of the same day: the same base block
        self.assertEqual(gen_loop.base_block_name(22, None, "2026-12-01"), "E22-2026-12-01-base")
        self.assertEqual(gen_loop.base_block_name(4, "Etest-r1"), "Etest-base")

    def test_resolve_e_block_never_takes_a_base_block_for_a_judged_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            seeds = Path(tmp) / "seeds"
            day = gen_loop.datetime.now(gen_loop.UTC).strftime("%Y-%m-%d")
            gen0.write_seed_file(f"E5-{day}-base", 10, seeds / f"E5-{day}-base.json")
            got = gen_loop.resolve_e_block(str(seeds / "E{k}-{date}.json"), 5, 10, also_dirs=[])
            self.assertEqual(got.stem, f"E5-{day}")  # a block of its own, never the champion's base block
            self.assertNotEqual(gen0.load_seeds(got), gen0.load_seeds(seeds / f"E5-{day}-base.json"))
            again = gen_loop.resolve_e_block(str(seeds / "E{k}-{date}.json"), 5, 10, also_dirs=[])
            self.assertEqual(again, got)  # and the block it wrote is found next time as before

    def test_the_seeds_come_from_the_name_are_checked_and_written_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            seeds = Path(tmp) / "seeds"
            plan = gen_loop.BaseBlock(name=NAME, seeds_dir=seeds)
            got = gen_loop.base_block_seeds(plan)
            self.assertEqual((len(got), got == gen0.e_block(NAME, 2000), len(set(got))), (2000, True, 2000))
            self.assertEqual(gen0.load_seeds(seeds / f"{NAME}.json"), got)
            stamp = (seeds / f"{NAME}.json").read_bytes()
            self.assertEqual(gen_loop.base_block_seeds(plan), got)  # a second try reads the file and does not rewrite it
            self.assertEqual((seeds / f"{NAME}.json").read_bytes(), stamp)
            (seeds / f"{NAME}.json").write_text(json.dumps({"seeds": got[:-1] + ["x"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not hold the seeds its name gives"):
                gen_loop.base_block_seeds(plan)
            (seeds / f"{NAME}.json").unlink()
            (seeds / "E4-other.json").write_text(json.dumps({"seeds": got[:5]}), encoding="utf-8")  # a block that shares seeds with it
            with self.assertRaisesRegex(ValueError, "5 seeds also in E4-other"):
                gen_loop.base_block_seeds(plan)
            self.assertFalse((seeds / f"{NAME}.json").exists())  # nothing is written for seeds that are not fresh


class BaseBlockWorld(trg.GrowthBase):
    """The champion generation 3 with no arm at all; `play` writes the arm a played base block would leave."""

    def setUp(self):
        super().setUp()
        shutil.rmtree(self.champ / "arms")
        self.plan = gen_loop.BaseBlock(name=NAME, seeds_dir=self.root / "seeds", n=6, arm_kw={"server_opts": "opts"})
        self.said: list[str] = []
        self.plays: list[dict] = []

    def player(self, *, short: int = 0, summary: bool = True):
        """A block player (`run_arm`'s signature): the rows of a played block, written into the champion's arms folder as the real one writes them."""
        def play(seeds, arm_dir, agent, config, procs, ascension=10, resume=False, **kw):
            self.plays.append({"seeds": list(seeds), "arm": Path(arm_dir), "agent": agent.generation, "config": config, "procs": procs, "ascension": ascension,
                               "resume": resume, "kw": kw})
            rows = [nodes_row(s, FATES[i % len(FATES)], i) for i, s in enumerate(seeds)][: len(seeds) - short]
            arm = trg.write_arm(self.champ, Path(arm_dir).name.removesuffix("-a3-v1"), rows, agent="gen3/candidate")
            if not summary:
                (arm / "summary.json").unlink()
            return rows

        return play

    def run_stage(self, cand: Path | None = None, **kw):
        cand = cand or self.cand
        kw.setdefault("block", JUDGED)
        return gen_loop.s1_stage(cand, self.champ, gen_loop.load_agent(cand), gen_loop.load_agent(self.champ), ruler_root=self.ruler_root, run=self.child,
                                 python="py", say=self.said.append, **kw)

    def seed_path(self) -> Path:
        return self.root / "seeds" / f"{NAME}.json"

    def marker(self) -> dict:
        return json.loads((self.champ / "arms" / f"{NAME}-a3-v1" / gen_loop.BASE_BLOCK_FILE).read_text(encoding="utf-8"))


class StageTests(BaseBlockWorld):
    def test_a_champion_with_no_arm_plays_a_base_block_and_the_exam_goes_on_from_the_ruler_it_gives(self):
        reading = self.run_stage(base_block=self.plan, play=self.player())
        seeds = gen0.e_block(NAME, 6)
        arm = self.champ / "arms" / f"{NAME}-a3-v1"
        self.assertEqual(self.plays, [{"seeds": seeds, "arm": arm, "agent": 3, "config": "a3-v1", "procs": 12, "ascension": 10, "resume": True,
                                       "kw": {"server_opts": "opts"}}])  # the champion alone, the judgement's own settings, resumable, 12 processes
        self.assertEqual([c[2] for c in self.calls], ["arrivals", "arrival-exam"])  # the ruler is built from the block, then the exam is played as ever
        self.assertEqual(self.pool.replayed, [NAME])
        meta, _ = a3.load_ruler(self.ruler)
        self.assertEqual([(x["block"], x["rows"], x["arrivals"]) for x in meta["arms"]], [(NAME, 6, 4)])
        self.assertEqual((meta["starts"], meta["first"], meta["second"]), (6, 4, 2))
        self.assertTrue(reading["match"]["verified"])
        self.assertTrue(k2.route_s1(reading)["holds"])
        self.assertEqual(json.loads((self.cand / "s1.json").read_text(encoding="utf-8"))["mean"], 1.8)
        self.assertTrue(any("playing a base block of 6 runs" in m and NAME in m for m in self.said))
        self.assertEqual(gen0.load_seeds(self.seed_path()), seeds)  # the seeds are on file beside the E-blocks'

    def test_the_block_is_registered_where_arms_are_read_and_its_runs_are_excluded_from_practice(self):
        self.run_stage(base_block=self.plan, play=self.player())
        self.assertEqual([d.name for d in a3.champion_arms(self.champ, [JUDGED])], [f"{NAME}-a3-v1"])  # `champion_arms` finds it: an arm with the base arm only
        self.assertEqual(a3.ruler_gap(a3.load_ruler(self.ruler)[0], self.champ, [JUDGED]), [])  # and the ruler holds all of it
        ex = json.loads((self.ruler / "exclude_runs.json").read_text(encoding="utf-8"))
        arrived = [s for i, s in enumerate(gen0.e_block(NAME, 6)) if FATES[i % len(FATES)] != "early"]
        self.assertEqual(ex["run_seeds"], sorted(arrived))  # `exclusion_of` on a ruler of one block
        self.assertEqual((ex["starts"], ex["by_stratum"]), (6, {"own": 6}))

    def test_the_runs_reach_the_practices_exclusion_list_before_the_exam_plays(self):
        exam = self.root / "exam" / "exclude_runs.json"
        trg.write_hand(exam, trg.exam_exclusions([f"x{i}" for i in range(5)], [f"x{i}|f40" for i in range(8)]))
        self.run_stage(base_block=self.plan, play=self.player())
        e = json.loads(exam.read_text(encoding="utf-8"))
        self.assertEqual((len(e["run_seeds"]), len(e["start_ids"]), e["by_stratum"]["arrival-own"]), (5 + 4, 8 + 6, 6))
        self.assertEqual((e["arrival"]["gen3-candidate"]["starts"], e["arrival"]["gen3-candidate"]["runs"]), (6, 4))
        self.assertTrue(any("on the practice's exclusion list now" in m for m in self.said))

    def test_off_raises_as_before_and_plays_and_writes_nothing(self):
        for kw in ({}, {"base_block": None}):
            with self.assertRaisesRegex(gen_loop.S1Unavailable, "has no arm outside the blocks it was selected on to build a ruler from"):
                self.run_stage(play=self.player(), **kw)
        self.assertEqual((self.plays, self.calls), ([], []))
        self.assertFalse(self.ruler_root.exists())  # no ruler folder is made
        self.assertFalse(self.seed_path().exists())
        self.assertFalse((self.champ / "arms").exists())

    def test_the_same_fight_network_is_refused_before_a_block_is_played(self):
        twin = self.swap_candidate("twin", trs.CHAMP_SHA)
        with self.assertRaises(gen_loop.S1Unavailable):
            self.run_stage(twin, base_block=self.plan, play=self.player())
        self.assertEqual((self.plays, self.calls), ([], []))
        self.assertFalse(self.seed_path().exists())

    def test_a_champion_that_has_arms_plays_no_base_block(self):
        trg.write_arm(self.champ, "A", trg.A_ROWS, agent="gen3/candidate")
        reading = self.run_stage(base_block=self.plan, play=self.player())
        self.assertEqual(self.plays, [])
        self.assertEqual([x["block"] for x in a3.load_ruler(self.ruler)[0]["arms"]], ["A"])  # the ruler is the arms', as today
        self.assertTrue(reading["match"]["verified"])
        self.assertFalse(self.seed_path().exists())
        self.assertIsNone(gen_loop.base_block_record(self.champ, self.plan, JUDGED, self.ruler_root))

    def test_a_ruler_that_exists_is_used_and_no_block_is_played(self):
        trs.make_ruler(self.ruler, "gen3-candidate", "gen3/candidate", label="Eprev")
        self.run_stage(base_block=self.plan, play=self.player())
        self.assertEqual((self.plays, [c[2] for c in self.calls]), ([], ["arrival-exam"]))

    def test_a_second_try_does_not_play_the_block_or_build_the_ruler_again_and_keeps_its_time(self):
        self.codes["arrival-exam"] = 3  # the exam is stopped (the night window, a landing)
        with self.assertRaises(gen_loop.S1Blocked):
            self.run_stage(base_block=self.plan, play=self.player())
        first = self.marker()
        h = a3.load_ruler(self.ruler)[0]["ruler_hash"]
        self.codes["arrival-exam"] = 0
        reading = self.run_stage(base_block=self.plan, play=self.player())
        self.assertEqual(len(self.plays), 1)
        self.assertEqual([c[2] for c in self.calls], ["arrivals", "arrival-exam", "arrival-exam"])
        self.assertEqual((self.marker(), reading["ruler_hash"], reading["test"]), (first, h, 0))
        # the ruler does not change between two tries: the champion's arm on the judged block (played since) is not read
        self.codes["arrival-exam"] = 3
        c2 = self.swap_candidate("c2", "sha-c2")
        with self.assertRaises(gen_loop.S1Blocked):
            self.run_stage(c2, base_block=self.plan, play=self.player())
        trg.write_arm(self.champ, JUDGED, trg.B_ROWS, agent="gen3/candidate")
        self.codes["arrival-exam"] = 0
        self.run_stage(c2, base_block=self.plan, play=self.player())
        self.assertEqual(a3.load_ruler(self.ruler)[0]["ruler_hash"], h)
        self.assertEqual(self.pool.replayed, [NAME])
        self.assertEqual(len(self.plays), 1)

    def test_a_block_that_was_played_but_whose_ruler_was_stopped_is_not_played_again(self):
        self.pool.stop_after = 1  # the ruler's replays are stopped (the night window, a landing) after one of four
        with self.assertRaises(gen_loop.S1Blocked):
            self.run_stage(base_block=self.plan, play=self.player())
        self.assertEqual(len(self.plays), 1)
        self.assertFalse((self.ruler / "meta.json").exists())  # no ruler a test could be played on
        self.pool.stop_after = None
        self.run_stage(base_block=self.plan, play=self.player())
        self.assertEqual(len(self.plays), 1)  # the next try builds the ruler from the arm on disk
        self.assertEqual([x["block"] for x in a3.load_ruler(self.ruler)[0]["arms"]], [NAME])

    def test_a_block_that_was_stopped_is_finished_before_a_ruler_is_built_from_it(self):
        arm = trg.write_arm(self.champ, NAME, [nodes_row("z0", "died", 0)], agent="gen3/candidate")  # a job killed after one row: no summary yet
        (arm / "summary.json").unlink()
        self.assertTrue(gen_loop.base_block_unfinished(self.champ, self.plan))
        self.run_stage(base_block=self.plan, play=self.player())
        self.assertEqual(len(self.plays), 1)  # `run_arm(resume=True)` plays the seeds it has no row for
        self.assertTrue(self.plays[0]["resume"])
        self.assertTrue(any("1 rows on disk" in m for m in self.said))
        self.assertEqual(a3.load_ruler(self.ruler)[0]["arms"][0]["rows"], 6)
        self.assertFalse(gen_loop.base_block_unfinished(self.champ, self.plan))

    def test_a_block_that_ends_short_is_blocked_and_finished_by_the_next_try(self):
        with self.assertRaisesRegex(gen_loop.S1Blocked, "4 of 6 runs"):
            self.run_stage(base_block=self.plan, play=self.player(short=2))
        self.assertEqual(self.calls, [])  # no ruler from half a block
        self.assertFalse((self.champ / "arms" / f"{NAME}-a3-v1" / gen_loop.BASE_BLOCK_FILE).exists())
        self.assertTrue(gen_loop.base_block_unfinished(self.champ, self.plan))  # its summary is there: a ruler would have read it as an arm
        self.run_stage(base_block=self.plan, play=self.player())
        self.assertEqual(len(self.plays), 2)
        self.assertEqual(a3.load_ruler(self.ruler)[0]["arms"][0]["rows"], 6)

    def test_the_record_and_the_words_of_the_ledger_cell(self):
        self.run_stage(base_block=self.plan, play=self.player())
        rec = gen_loop.base_block_record(self.champ, self.plan, JUDGED, self.ruler_root)
        self.assertEqual((rec["block"], rec["for_block"], rec["runs"], rec["arrivals"], rec["starts"], rec["matched"]), (NAME, JUDGED, 6, 4, 6, 4))
        self.assertEqual(rec["rule"], gen_loop.BASE_BLOCK_RULE)
        self.assertEqual(gen_loop.base_block_note(rec), f"S1 on a base block played {rec['written']}: 6 runs, 4 arrivals")
        self.assertRegex(rec["written"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual(gen_loop.base_block_note(rec, s1_read=False), f"a base block was played {rec['written']}: 6 runs, no S1 reading")
        self.assertIsNone(gen_loop.base_block_record(self.champ, self.plan, JUDGED + "-r1", self.ruler_root))  # another block's judgement: no note
        self.assertIsNone(gen_loop.base_block_note(None))
        self.assertNotIn(str(self.root), json.dumps(rec))  # no absolute path in the record


class PromoteTests(BaseBlockWorld):
    """`promote(route_s="auto")` with the real S1 stage on the fakes above: the flag, the block, the ledger's `guard` cell."""

    def setUp(self):
        super().setUp()
        self.seeds, self.c, self.b = block(trs.FLATISH, floor_mean=0.0, prefix="K", seed=7)
        self.f1 = self.seed_file("Etest-r1", self.seeds)
        self.arm_calls: list[tuple] = []
        for p in (mock.patch.object(a3, "ARRIVAL_ROOT", self.ruler_root), mock.patch.object(gen_loop, "BASE_BLOCK_RUNS", 6)):
            p.start()
            self.addCleanup(p.stop)

    def arms(self):
        played = self.player()

        def run_arm(seeds, arm_dir, agent, config, procs, ascension=10, resume=False, **kw):
            self.arm_calls.append((agent.generation, Path(arm_dir).name, len(seeds)))
            if Path(arm_dir).name.endswith("-base-a3-v1"):
                return played(seeds, arm_dir, agent, config, procs, ascension, resume, **kw)
            want = set(seeds)
            return [r for r in (self.c if agent.generation == 4 else self.b) if r["seed"] in want]

        return run_arm

    def promote(self, **kw):
        real = gen_loop.s1_stage

        def stage(cand, base_dir, cand_agent, base_agent, **k):
            return real(cand, base_dir, cand_agent, base_agent, run=self.child, python="py", **k)

        with mock.patch.object(gen_loop, "run_arm", self.arms()), mock.patch.object(gen_loop, "s1_stage", stage):
            return gen_loop.promote(self.cand, self.champ, self.f1, agent_root=self.agents, boot=500, keep_rule="v3", sequential=True, **kw)

    def guard(self) -> str:
        return gen_loop.read_promotions(self.agents / "promotions.tsv")[-1]["guard"]

    def test_auto_plays_the_block_and_the_row_says_so(self):
        doc = self.promote(route_s="auto")
        rec = doc["route_s_mode"]["base_block"]
        self.assertEqual((rec["block"], rec["for_block"], rec["runs"], rec["arrivals"]), ("Etest-base", "Etest-r1", 6, 4))
        self.assertEqual([c for c in self.arm_calls if c[1].endswith("-base-a3-v1")], [(3, "Etest-base-a3-v1", 6)])  # one block, the champion alone
        self.assertEqual(sorted({c[0] for c in self.arm_calls[1:]}), [3, 4])  # then the judged block's two arms, as ever
        self.assertTrue((self.root / "seeds" / "Etest-base.json").is_file())
        self.assertIn(f"S1 on a base block played {rec['written']}: 6 runs, 4 arrivals", self.guard())
        self.assertIn("S1 +1.80", self.guard())  # beside the reading's own words
        self.assertEqual((doc["route_s_mode"]["applies"], doc["decision_text"]), (True, "promoted"))
        self.assertEqual(doc["s1"]["mean"], 1.8)
        self.assertNotIn("\t", self.guard())

    def test_off_is_the_old_way_s1_unavailable_and_v3_decides(self):
        doc = self.promote(route_s="auto", base_block="off")
        self.assertEqual([c for c in self.arm_calls if c[1].endswith("-base-a3-v1")], [])
        self.assertIn("S1 unavailable: the champion gen3/candidate has no arm outside the blocks it was selected on to build a ruler from; v3 decides", self.guard())
        self.assertNotIn("base block", self.guard())
        self.assertNotIn("base_block", doc["route_s_mode"])
        self.assertEqual((doc["route_s_mode"]["applies"], doc["sequential"]["futility"]["enabled"]), (False, True))  # v3 judges, futility stop on, as before
        self.assertNotIn("route", doc["sequential"])
        self.assertNotIn("route_s", doc["keep_v2"])
        self.assertNotIn("s1", doc)
        self.assertFalse((self.root / "seeds" / "Etest-base.json").exists())

    def test_the_default_under_route_s_auto_is_auto_and_route_s_off_plays_no_block(self):
        doc = self.promote(route_s="auto", base_block=None)
        self.assertIn("base_block", doc["route_s_mode"])
        self.setUp()
        doc = self.promote()  # route S not asked: S1 is not read, so no block
        self.assertEqual(self.arm_calls[0][1], "Etest-r1-a3-v1")
        self.assertNotIn("route_s_mode", doc)
        self.assertNotIn("base block", self.guard())
        self.setUp()
        self.promote(base_block="auto")
        self.assertFalse(any(c[1].endswith("-base-a3-v1") for c in self.arm_calls))

    def test_a_champion_with_arms_gets_the_rows_as_before(self):
        trg.write_arm(self.champ, "A", trg.A_ROWS, agent="gen3/candidate")
        doc = self.promote(route_s="auto")
        self.assertNotIn("base_block", doc["route_s_mode"])
        self.assertNotIn("base block", self.guard())
        self.assertIn("S1 +1.80", self.guard())

    def test_a_flag_that_is_not_a_mode_is_an_error_before_anything_is_played(self):
        with mock.patch.object(gen_loop, "run_arm", self.arms()), self.assertRaisesRegex(ValueError, "unknown --base-block"):
            gen_loop.promote(self.cand, self.champ, self.f1, agent_root=self.agents, boot=500, keep_rule="v3", route_s="auto", base_block="yes")
        self.assertEqual(self.arm_calls, [])

    def test_a_block_that_ends_short_stops_the_judgement_before_the_blocks_arms(self):
        real = self.arms()

        def short(seeds, arm_dir, agent, config, procs, ascension=10, resume=False, **kw):
            rows = real(seeds, arm_dir, agent, config, procs, ascension, resume, **kw)
            return rows[:-2] if Path(arm_dir).name.endswith("-base-a3-v1") else rows

        stage = gen_loop.s1_stage
        with mock.patch.object(gen_loop, "run_arm", short), \
                mock.patch.object(gen_loop, "s1_stage", lambda *a, **k: stage(*a, run=self.child, python="py", **k)), \
                self.assertRaises(gen_loop.S1Blocked):
            gen_loop.promote(self.cand, self.champ, self.f1, agent_root=self.agents, boot=500, keep_rule="v3", sequential=True, route_s="auto")
        self.assertFalse((self.cand / "promotion.json").exists())
        self.assertFalse((self.cand / "arms").exists())


class CommandLineTests(trs.World):
    def test_the_flag_is_auto_by_default_on_train_and_promote(self):
        parser = gen0.build_parser()
        need = ["--shards-root", "r", "--agent-root", "a"]
        self.assertEqual(parser.parse_args(["train", "--gen", "1", *need]).base_block, "auto")
        self.assertEqual(parser.parse_args(["promote", "--cand", "c", "--agent-root", "a"]).base_block, "auto")
        self.assertEqual(parser.parse_args(["train", "--gen", "1", *need, "--base-block", "off"]).base_block, "off")
        self.assertEqual(parser.parse_args(["promote", "--cand", "c", "--agent-root", "a", "--base-block", "off"]).base_block, "off")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["promote", "--cand", "c", "--agent-root", "a", "--base-block", "on"])

    def test_the_child_command_carries_the_flag_only_when_it_is_off(self):
        calls = []

        def fake_run(cmd, check=False):
            calls.append(cmd)
            return SimpleNamespace(returncode=0)

        with mock.patch.object(gen_loop.subprocess, "run", fake_run):
            for kw in ({}, {"base_block": None}, {"base_block": "auto"}, {"base_block": "off"}, {"route_s": "auto", "base_block": "off"}):
                gen_loop.run_promote_subprocess("py", Path("c"), None, Path("s.json"), Path("a"), 4, None, None, keep_rule="v3", **kw)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[0], calls[2])  # the default command is exactly as it was
        self.assertNotIn("--base-block", calls[0])
        self.assertEqual(calls[3][calls[3].index("--base-block") + 1], "off")
        self.assertEqual(calls[4][calls[4].index("--route-s") + 1], "auto")
        self.assertEqual(calls[4][calls[4].index("--base-block") + 1], "off")

    def test_promote_cmd_passes_the_flag_on(self):
        args = SimpleNamespace(cand=str(self.cand), base=str(self.champ), seeds_file="s.json", agent_root=str(self.agents), config="a3-v1", ascension=10,
                               procs=4, lib=None, lab=None, lab_runs=None, review_guard=False, sequential=True, smoke_seeds=200, batch_seeds=300,
                               keep_rule="v3", keep_rule_why=None, min_gain=1.0, no_futility=False, futility_min_seeds=None, no_screen=False,
                               pool_block=[], route_s="auto", base_block="off", prior_server="off", gpu_python=None, server_procs=None,
                               prior_server_args=None, latency_p99_ms=20.0)
        seen = {}

        def fake_promote(*a, **kw):
            seen.update(kw)
            return {"decision_text": "rejected", "floor": {}, "boss_kill": {}, "wins": {}}

        with mock.patch.object(gen0, "_setup"), mock.patch.object(gen_loop, "promote", fake_promote), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen0.promote_cmd(args), 0)
        self.assertEqual((seen["route_s"], seen["base_block"]), ("auto", "off"))
        del args.base_block  # a caller that names no flag reaches promote with none: the default
        with mock.patch.object(gen0, "_setup"), mock.patch.object(gen_loop, "promote", fake_promote), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen0.promote_cmd(args), 0)
        self.assertIsNone(seen["base_block"])
        self.assertEqual(gen_loop.base_block_of(seen["base_block"]), "auto")


class TrainFlagTests(unittest.TestCase):
    """`gen0.py train --promote` on a built candidate, the judging child stubbed: `--base-block off` reaches the child, the default names nothing."""

    setUp, tearDown, build, train = trs.TrainFlagTests.setUp, trs.TrainFlagTests.tearDown, trs.TrainFlagTests.build, trs.TrainFlagTests.train

    def test_off_reaches_the_child_and_the_default_names_nothing(self):
        rc, calls, _ = self.train("--keep-rule", "v3", "--route-s", "auto", "--base-block", "off")
        self.assertEqual((rc, len(calls)), (0, 1))
        self.assertEqual((calls[0][1]["route_s"], calls[0][1]["base_block"]), ("auto", "off"))
        self.tearDown()
        self.setUp()
        rc, calls, _ = self.train("--keep-rule", "v3", "--route-s", "auto")
        self.assertEqual((rc, len(calls)), (0, 1))
        self.assertNotIn("base_block", calls[0][1])  # the child's call is exactly as it was: its default is auto
        self.tearDown()
        self.setUp()
        rc, calls, _ = self.train("--keep-rule", "v3", "--route-s", "auto", "--base-block", "auto")
        self.assertNotIn("base_block", calls[0][1])


if __name__ == "__main__":
    unittest.main()
