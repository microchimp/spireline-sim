"""rl/run_seed_exam.py: the seed-matched run exam (D58 addendum 14): our player on the seeds of human runs, against the humans' own outcomes.

A synthetic corpus and a fake player stand in for the real ones (no corpus is read, no game is played). The pinned facts: the exam fifth is refused without
`--exam --milestone` and each use is written to the use log; the pick is half won / half lost by the human, by the smallest seed key, from the asked fifth
only; no seed (and no run id, no player id) reaches a file, a log line or an error text, and the per-seed rows carry the seed's key alone; the won and lost
splits and the aggregates are the means of the rows; a cell under 20 runs is hidden; a rerun keeps its finished rows; a worker that died is run again once.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import human_starts as hs  # noqa: E402
import run_human_bar as hb  # noqa: E402
import run_seed_exam as rse  # noqa: E402


def find_ids(bucket: int, n: int, prefix: str) -> list[str]:
    out, i = [], 0
    while len(out) < n:
        rid = f"{prefix}{i:012x}"
        if hs.exam_bucket(rid) == bucket:
            out.append(rid)
        i += 1
    return out


def pt(kind: str, hp: int, max_hp: int = 80) -> dict:
    return {"map_point_type": kind, "player_stats": [{"current_hp": hp, "max_hp": max_hp}], "rooms": [{"room_type": kind}]}


def human_record(rid: str, seed: str, won: bool, door1: int = 50) -> dict:
    """A won run (both doors, 50 points) or a lost one (died in the Act 1 boss: 17 points)."""
    act1 = [pt("monster", 60) for _ in range(15)] + [pt("rest_site", door1), pt("boss", 0 if not won else 20)]
    mph = [act1]
    if won:
        mph.append([pt("monster", 60) for _ in range(15)] + [pt("rest_site", 55), pt("boss", 10)])
        mph.append([pt("monster", 40) for _ in range(16)])
    return {"acts": ["ACT.OVERGROWTH", "ACT.HIVE", "ACT.GLORY"], "ascension": 10, "game_mode": "standard", "was_abandoned": False, "modifiers": [],
            "players": [{"character": "CHARACTER.IRONCLAD", "id": "PLAYER-SECRET-ID"}], "win": won, "run_hash": rid, "seed": seed, "start_time": 1,
            "map_point_history": mph}


class Fixture:
    """A corpus of fifth 4 (won and lost), fifth 2, and the exam fifth (listed in the ids file), with a seed per run that no output may carry."""

    def __init__(self, root: Path, n_won: int = 30, n_lost: int = 30) -> None:
        self.root = root
        self.records: list[dict] = []
        self.fifth4_won: list[str] = []
        self.fifth4_lost: list[str] = []
        self.other: list[str] = []
        self.exam: list[str] = []
        ids4 = find_ids(4, n_won + n_lost, "a")
        for i, rid in enumerate(ids4):
            won = i < n_won
            seed = f"S4{'W' if won else 'L'}{i:03d}XQ"
            self.records.append(human_record(rid, seed, won, door1=40 + i % 20))
            (self.fifth4_won if won else self.fifth4_lost).append(seed)
        for i, rid in enumerate(find_ids(2, 8, "b")):
            seed = f"S2OTH{i:03d}"
            self.records.append(human_record(rid, seed, i % 2 == 0))
            self.other.append(seed)
        exam_ids = find_ids(0, 8, "c")
        for i, rid in enumerate(exam_ids):
            seed = f"S0EXM{i:03d}"
            self.records.append(human_record(rid, seed, i % 2 == 0))
            self.exam.append(seed)
        self.corpus = root / "runs.jsonl"
        self.corpus.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in self.records), encoding="utf-8")
        self.ids = root / "exam-ids.txt"
        self.ids.write_text("\n".join(exam_ids) + "\n", encoding="utf-8")

    @property
    def all_seeds(self) -> list[str]:
        return self.fifth4_won + self.fifth4_lost + self.other + self.exam


def fake_task(seed, key, draw, ix, config, ascension, agent, human_won):  # noqa: ANN001 -- the shape of `play_task`
    """A deterministic outcome from the key and the draw: the seeds the human won go a little further; one draw in six of those wins."""
    h = int(key[:6], 16) + 3 * draw
    floor = 17 + h % 9 + (12 if human_won else 0)
    win = bool(human_won) and h % 6 == 0
    acts = [[0, 1.0 if floor >= 20 else 0.5]]
    door = {"1": [40 + h % 30, 80]}
    if floor >= 29:
        acts.append([1, 1.0 if win else 0.3])
        door["2"] = [50 + h % 20, 80]
    return {"key": key, "draw": draw, "human_won": human_won, "win": win, "floor": 51 if win else floor, "end": "win" if win else "death",
            "boss_damage_acts": acts, "door": door, "decisions": 100, "wall": 0.5, "cpu": 0.4}


class StubAgent:
    generation = 13

    def describe(self) -> dict:
        return {"generation": 13, "dir": "gen13/candidate-stakes", "prior": True, "prior_sha256": "abcd", "macro": "vrun", "stakes": "a3-v1@h1e4b16"}

    def wall_factor(self) -> float:
        return 4.0


def play(fx: Fixture, out: Path, *extra: str, procs: int = 0) -> tuple[int, str]:
    """`run_seed_exam.py play` on the fixture with the fake player. Returns (exit code, what it printed)."""
    agent_file = fx.root / "current.json"
    agent_file.write_text(json.dumps({"generation": 13}), encoding="utf-8")
    argv = ["play", "--corpus", str(fx.corpus), "--exam-ids", str(fx.ids), "--agent", str(agent_file), "--out", str(out), "--procs", str(procs),
            "--exam-use-log", str(fx.root / "exam-use.log"), *extra]
    import gen0  # noqa: PLC0415
    import gen_loop  # noqa: PLC0415

    buf = io.StringIO()
    with mock.patch.object(gen_loop, "load_agent", return_value=StubAgent()), mock.patch.object(gen0, "select_engine"), \
            mock.patch.object(gen0, "_setup"), mock.patch.object(rse, "play_task", fake_task), mock.patch.dict("os.environ", {}), \
            contextlib.redirect_stdout(buf):
        rc = rse.main(argv)
    return rc, buf.getvalue()


def rows_of(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / "runs.jsonl").read_text(encoding="utf-8").splitlines() if x]


def all_text(*roots: Path) -> str:
    text = []
    for root in roots:
        for p in ([root] if root.is_file() else sorted(root.rglob("*"))):
            if p.is_file():
                text.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(text)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


class ExamFifthTests(Base):
    def test_the_exam_fifth_is_refused_without_the_flag_and_without_a_milestone(self):
        with self.assertRaises(SystemExit):
            rse.check_fifth(0, False, None)  # fifth 0 without --exam
        with self.assertRaises(SystemExit):
            rse.check_fifth(0, True, None)  # --exam without a milestone
        with self.assertRaises(SystemExit):
            rse.check_fifth(0, True, "  ")
        with self.assertRaises(SystemExit):
            rse.check_fifth(4, True, "m7")  # --exam is the exam fifth and no other
        with self.assertRaises(SystemExit):
            rse.check_fifth(5, False, None)
        self.assertEqual(4, rse.check_fifth(4, False, None))
        self.assertEqual(0, rse.check_fifth(0, True, "m7"))
        self.assertEqual(0, rse.check_fifth(None, True, "m7"))

    def test_the_command_refuses_before_it_reads_anything(self):
        fx = Fixture(self.tmp)
        for extra in (["--fifth", "0"], ["--exam"], ["--exam", "--fifth", "4", "--milestone", "m"]):
            with self.assertRaises(SystemExit):
                play(fx, self.tmp / "o", *extra)
        self.assertFalse((self.tmp / "o").exists())
        self.assertFalse((self.tmp / "exam-use.log").exists())

    def test_a_training_fifth_picks_only_that_fifth(self):
        fx = Fixture(self.tmp)
        play(fx, self.tmp / "o", "--seeds", "40", "--draws", "1")
        keys = {r["key"] for r in rows_of(self.tmp / "o")}
        self.assertFalse(keys & {rse.seed_key(s) for s in fx.exam + fx.other})
        self.assertTrue(keys <= {rse.seed_key(s) for s in fx.fifth4_won + fx.fifth4_lost})

    def test_an_exam_use_is_one_line_in_the_use_log_and_reads_only_the_exam_fifth(self):
        fx = Fixture(self.tmp)
        rc, text = play(fx, self.tmp / "ex", "--exam", "--milestone", "milestone-9", "--seeds", "8", "--draws", "1")
        self.assertEqual(0, rc)
        lines = (self.tmp / "exam-use.log").read_text(encoding="utf-8").splitlines()
        self.assertEqual(1, len(lines))
        self.assertRegex(lines[0], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ exam-fifth milestone=milestone-9 seeds=8 draws=1 agent=gen13$")
        keys = {r["key"] for r in rows_of(self.tmp / "ex")}
        self.assertEqual(8, len(keys))
        self.assertEqual(keys, {rse.seed_key(s) for s in fx.exam})
        self.assertNotIn("PLAYER-SECRET-ID", all_text(self.tmp / "ex", self.tmp / "exam-use.log") + text)


class PickTests(Base):
    def test_half_won_half_lost_by_the_smallest_key_one_run_per_seed(self):
        entries = [(f"seed{i}", {"win": i % 3 == 0, "floor": 10, "doors": {}}) for i in range(90)] + [("seed3", {"win": True, "floor": 99, "doors": {}})]
        picks, counts = rse.pick(entries, 20)
        self.assertEqual((10, 10), (sum(p["won"] for p in picks), sum(not p["won"] for p in picks)))
        self.assertEqual(1, counts["duplicate_seed"])
        for won in (True, False):
            chosen = [p["key"] for p in picks if p["won"] == won]
            pool = sorted(rse.seed_key(f"seed{i}") for i in range(90) if (i % 3 == 0) == won)
            self.assertEqual(pool[:10], sorted(chosen))
        everyone, _ = rse.pick(entries, 200)
        self.assertEqual(10, [p for p in everyone if p["seed"] == "seed3"][0]["row"]["floor"])  # the first occurrence of a repeated seed

    def test_an_odd_count_gives_the_extra_run_to_the_lost_and_a_short_class_is_counted(self):
        entries = [(f"w{i}", {"win": True, "floor": 1, "doors": {}}) for i in range(3)] + [(f"l{i}", {"win": False, "floor": 1, "doors": {}}) for i in range(30)]
        picks, counts = rse.pick(entries, 11)
        self.assertEqual((3, 6), (counts["picked_won"], counts["picked_lost"]))
        self.assertEqual(3, counts["available_won"])

    def test_the_key_is_a_salted_hash_and_the_draw_index_is_a_function_of_it(self):
        k = rse.seed_key("ABCDEFGHJK")
        self.assertRegex(k, r"^[0-9a-f]{16}$")
        self.assertNotIn("ABCDEFGHJK", k)
        self.assertEqual(k, rse.seed_key("ABCDEFGHJK"))
        self.assertNotEqual(k, rse.seed_key("ABCDEFGHJL"))
        self.assertEqual(len({rse.draw_index(k, d) for d in range(rse.MAX_DRAWS)}), rse.MAX_DRAWS)
        self.assertEqual(rse.draw_index(k, 1), rse.draw_index(k, 1))


class NoSeedTests(Base):
    def test_no_seed_reaches_a_file_a_log_line_or_an_error_text(self):
        fx = Fixture(self.tmp)
        rc, text = play(fx, self.tmp / "o", "--seeds", "20", "--draws", "2")
        self.assertEqual(0, rc)
        text += all_text(self.tmp / "o")
        for seed in fx.all_seeds:
            self.assertNotIn(seed, text)
        for rid in (r["run_hash"] for r in fx.records):
            self.assertNotIn(rid, text)
        self.assertNotIn("PLAYER-SECRET-ID", text)

    def test_a_crash_text_that_names_the_seed_names_its_key_instead(self):
        seed = "SECRETSEED1"
        import gen0  # noqa: PLC0415

        with mock.patch.object(gen0, "_setup"), mock.patch.object(gen0, "play_emu_run", side_effect=RuntimeError(f"no such seed {seed} on this engine")):
            row = rse.play_task(seed, rse.seed_key(seed), 0, 1, "a3-v1", 10, StubAgent(), True)
        self.assertTrue(row["crash"])
        self.assertNotIn(seed, json.dumps(row))
        self.assertIn(rse.seed_key(seed), row["error"])
        self.assertEqual({"win", "floor", "end", "decisions", "crash", "error", "key", "draw", "human_won", "door", "wall", "cpu"}, set(row))

    def test_scrub_replaces_every_seed(self):
        self.assertEqual(f"a {rse.seed_key('AA')} b {rse.seed_key('BB')} {rse.seed_key('AA')}", rse.scrub("a AA b BB AA", ["AA", "BB", "CC"]))

    def test_the_source_never_reads_or_writes_a_player_id(self):
        text = (ROOT / "rl" / "run_seed_exam.py").read_text(encoding="utf-8")
        for bad in ('players"][0]', "players[0]", "player_id", '"id"'):
            self.assertNotIn(bad, text)
        self.assertNotIn("data/spire-codex", text)  # the corpus path is the readers' (human_starts.CORPUS), not copied here

    def test_the_rows_carry_the_key_alone(self):
        fx = Fixture(self.tmp)
        play(fx, self.tmp / "o", "--seeds", "20", "--draws", "2")
        rows = rows_of(self.tmp / "o")
        self.assertEqual(40, len(rows))
        for r in rows:
            self.assertRegex(r["key"], r"^[0-9a-f]{16}$")
            self.assertTrue({"seed", "ix", "run", "run_hash", "run_id", "id"}.isdisjoint(r))
        self.assertEqual({0, 1}, {r["draw"] for r in rows})
        self.assertEqual(20, len({r["key"] for r in rows}))


class AggregateTests(Base):
    def setUp(self):
        super().setUp()
        self.fx = Fixture(self.tmp)
        play(self.fx, self.tmp / "o", "--seeds", "40", "--draws", "2")
        self.rows = rows_of(self.tmp / "o")
        self.doc = json.loads((self.tmp / "o" / "summary.json").read_text(encoding="utf-8"))

    def test_the_pick_is_twenty_won_and_twenty_lost(self):
        self.assertEqual((20, 20), (self.doc["picked"]["picked_won"], self.doc["picked"]["picked_lost"]))
        self.assertEqual((40, 2), (self.doc["seeds"], self.doc["draws"]))
        self.assertEqual(80, len(self.rows))

    def test_the_human_side_is_split_by_whether_the_human_won(self):
        h = self.doc["human"]
        self.assertEqual((40, 20, 20), (h["all"]["runs"], h["won"]["runs"], h["lost"]["runs"]))
        self.assertEqual((0.5, 1.0, 0.0), (h["all"]["win_rate"], h["won"]["win_rate"], h["lost"]["win_rate"]))
        self.assertEqual(50, h["won"]["floor"]["p50"])  # 17 + 17 + 16 points
        self.assertEqual(17, h["lost"]["floor"]["p50"])
        self.assertEqual(20, h["won"]["acts"]["2"]["reached"])
        self.assertIsNone(h["lost"]["acts"]["2"])  # nobody lost reached the Act 2 door
        self.assertEqual(20, h["lost"]["acts"]["1"]["reached"])
        self.assertEqual(0.0, h["lost"]["acts"]["1"]["kill_rate"])
        self.assertEqual(1.0, h["won"]["acts"]["1"]["kill_rate"])

    def test_our_figures_are_the_means_of_the_rows_in_each_split(self):
        for name, want in (("all", None), ("won", True), ("lost", False)):
            sel = [r for r in self.rows if want is None or r["human_won"] == want]
            b = self.doc["ours"][name]
            self.assertEqual(len(sel), b["runs"])
            self.assertAlmostEqual(sum(r["floor"] for r in sel) / len(sel), b["floor"]["mean"], places=3)
            self.assertAlmostEqual(sum(r["win"] for r in sel) / len(sel), b["win_rate"], places=4)
            reached = [r for r in sel if any(a == 0 for a, _ in r["boss_damage_acts"])]
            self.assertEqual(len(reached), b["acts"]["1"]["reached"])
            self.assertEqual(sum(1 for r in reached if dict(map(tuple, r["boss_damage_acts"]))[0] >= 1.0), b["acts"]["1"]["killed"])
            self.assertAlmostEqual(sum(r["door"]["1"][0] for r in reached) / len(reached), b["acts"]["1"]["door_hp"]["mean"], places=2)
        self.assertEqual(0.0, self.doc["ours"]["lost"]["win_rate"])  # the fake player wins only on seeds the human won

    def test_the_seed_clustered_interval_brackets_the_mean(self):
        w = self.doc["ours"]["all"]
        ci = w["floor_ci"]
        self.assertEqual(40, ci["n"])
        self.assertLessEqual(ci["lo"], w["floor"]["mean"])
        self.assertGreaterEqual(ci["hi"], w["floor"]["mean"])

    def test_the_paired_block_compares_each_seed_with_its_own_human(self):
        p = self.doc["paired"]
        self.assertEqual((40, 20, 20), (p["all"]["seeds"], p["won"]["seeds"], p["lost"]["seeds"]))
        by = {}
        for r in self.rows:
            by.setdefault(r["key"], []).append(r["floor"])
        humans = {rse.seed_key(s): 50 for s in self.fx.fifth4_won} | {rse.seed_key(s): 17 for s in self.fx.fifth4_lost}
        won_keys = [k for k in by if humans[k] == 50]
        want = sum(sum(by[k]) / len(by[k]) - 50 for k in won_keys) / len(won_keys)
        self.assertAlmostEqual(want, p["won"]["floor_diff"]["mean"], places=3)
        self.assertEqual(sum(1 for k in won_keys if any(r["win"] for r in self.rows if r["key"] == k)), p["won"]["some_draw_won"])
        self.assertEqual(0, p["lost"]["some_draw_won"])

    def test_the_table_prints_every_column_and_the_rows_agree_with_report(self):
        text = (self.tmp / "o" / "table.md").read_text(encoding="utf-8")
        for head in ("our player", "humans, same seeds", "human won", "humans, won", "human lost", "humans, lost", "copy on its own seeds"):
            self.assertIn(head, text)
        for row in ("win rate", "mean floor reached", "Act 1 boss reached", "Act 1 boss killed", "Act 2 HP at the door", "On the same seed"):
            self.assertIn(row, text)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rse.main(["report", "--out", str(self.tmp / "o")])
        self.assertEqual(text.strip(), buf.getvalue().strip())


class HiddenCellTests(Base):
    def test_a_cell_under_twenty_runs_is_hidden_in_the_table_and_the_files(self):
        fx = Fixture(self.tmp)
        play(fx, self.tmp / "o", "--seeds", "30", "--draws", "1")  # 15 won + 15 lost: the splits are under 20
        doc = json.loads((self.tmp / "o" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(30, doc["ours"]["all"]["runs"])
        for side in ("ours", "human"):
            for split in ("won", "lost"):
                self.assertIsNone(doc[side][split]["runs"])
                self.assertIsNone(doc[side][split]["floor"])
                self.assertIsNone(doc[side][split]["win_rate"])
        self.assertIsNone(doc["paired"]["won"]["seeds"])
        self.assertNotIn("floor_diff", doc["paired"]["won"])
        text = (self.tmp / "o" / "table.md").read_text(encoding="utf-8")
        won_col = [ln.split("|")[4].strip() for ln in text.splitlines() if ln.startswith("| runs |")][0]
        self.assertEqual("-", won_col)

    def test_an_act_cell_under_twenty_reaching_runs_is_hidden_but_the_run_cell_stays(self):
        rows = [{"key": f"{i:016x}", "draw": 0, "human_won": False, "win": False, "floor": 17, "end": "death", "boss_damage_acts": [[0, 0.5]],
                 "door": {"1": [40, 80]}} for i in range(30)]
        rows += [{"key": f"{i + 100:016x}", "draw": 0, "human_won": False, "win": False, "floor": 33, "end": "death",
                  "boss_damage_acts": [[0, 1.0], [1, 0.2]], "door": {"1": [50, 80], "2": [30, 80]}} for i in range(12)]
        b = rse.summarise_rows(rows)
        self.assertEqual(42, b["runs"])
        self.assertEqual(42, b["acts"]["1"]["reached"])
        self.assertIsNone(b["acts"]["2"])
        self.assertIn("act2", b["hidden"])
        self.assertIsNone(rse.mean_ci([1.0] * 19))
        self.assertIsNotNone(rse.mean_ci([1.0, 2.0] * 10))

    def test_runs_that_did_not_end_are_counted_and_left_out_of_the_means(self):
        rows = [{"key": f"{i:016x}", "draw": 0, "human_won": True, "win": False, "floor": 20, "end": "death", "boss_damage_acts": []} for i in range(25)]
        rows += [{"key": "f" * 16, "draw": 0, "human_won": True, "win": False, "floor": 0, "end": "crash", "crash": True},
                 {"key": "e" * 16, "draw": 0, "human_won": True, "win": False, "floor": 0, "end": "lost", "lost": True},
                 {"key": "d" * 16, "draw": 0, "human_won": True, "win": False, "floor": 5, "end": "timeout"}]
        b = rse.summarise_rows(rows)
        self.assertEqual((25, 3), (b["runs"], b["excluded"]))
        self.assertEqual(20.0, b["floor"]["mean"])


class OwnSeedTests(Base):
    def make_agent(self, arm_agent: dict) -> Path:
        d = self.tmp / "agent" / "gen13" / "candidate-stakes"
        (d / "arms" / "E1-a3-v1").mkdir(parents=True)
        (d / "agent.json").write_text(json.dumps({"generation": 13, "macro": "vrun", "search_schedule": {"tag": "a3-v1@h1e4b16"}}), encoding="utf-8")
        (d / "arms" / "E1-a3-v1" / "summary.json").write_text(json.dumps({"agent": arm_agent}), encoding="utf-8")
        rows = [{"seed": "LABSEED", "floor": 24, "win": False, "end": "death", "boss_damage_acts": [[0, 1.0]]} for _ in range(25)]
        rows.append({"seed": "LABSEED", "floor": 0, "win": False, "end": "crash", "crash": True})
        (d / "arms" / "E1-a3-v1" / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        cur = self.tmp / "agent" / "current.json"
        cur.write_text(json.dumps({"generation": 13, "dir": str(d)}), encoding="utf-8")
        return cur

    def test_the_copys_own_seed_rows_are_the_judgement_arms_of_the_same_agent(self):
        cur = self.make_agent({"generation": 13, "macro": "vrun", "stakes": "a3-v1@h1e4b16"})
        rows, info = rse.own_seed_rows(cur)
        self.assertEqual((26, 1, 13), (len(rows), info["arms"], info["generation"]))
        b = rse.summarise_rows(rows)
        self.assertEqual((25, 1), (b["runs"], b["excluded"]))
        self.assertEqual(24.0, b["floor"]["mean"])
        self.assertEqual(1.0, b["acts"]["1"]["kill_rate"])
        self.assertNotIn("seed", rows[0])  # the lab's own seed is not carried either

    def test_an_arm_of_another_agent_is_not_the_copys_own_seeds(self):
        cur = self.make_agent({"generation": 13, "macro": "vrun", "stakes": "a3-v1@h1e4b8"})
        self.assertEqual(([], {"generation": 13, "arms": 0, "config": "a3-v1"}), rse.own_seed_rows(cur))
        self.assertEqual(([], {}), rse.own_seed_rows(self.tmp / "no" / "current.json"))


class FakePool:
    """A pool that runs each task in this process, but lets a script decide how a (key, attempt) ends."""
    script: dict = {}
    seen: list = []

    def __init__(self, ctx, size, **kw):
        self.q: list = []
        self.n: dict = {}

    def free(self):
        return 2 - len(self.q)

    def inflight(self):
        return len(self.q)

    def submit(self, key, fn, args=()):
        self.q.append((key, fn, args))
        FakePool.seen.append(key)

    def collect(self):
        from spawn_pool import Done  # noqa: PLC0415

        out = []
        q, self.q = self.q, []
        for key, fn, args in q:
            n = self.n[key] = self.n.get(key, 0) + 1
            how = FakePool.script.get((key[0], n))
            if how == "died":
                out.append(Done(key, "died", f"worker gone while playing {args[0]}", 1.0))
            elif how == "timeout":
                out.append(Done(key, "timeout", f"too long on {args[0]}", 9.0))
            else:
                out.append(Done(key, "ok", fn(*args), 0.1))
        return out

    def wait(self, t):
        pass

    def terminate(self):
        pass


class DriverTests(Base):
    def picks(self, n: int = 6) -> list[dict]:
        return [{"key": rse.seed_key(f"SD{i}"), "seed": f"SD{i}", "won": i % 2 == 0, "row": {"floor": 20, "win": False, "doors": {}}} for i in range(n)]

    def test_a_rerun_keeps_its_rows_and_plays_only_the_missing_draws(self):
        picks, out = self.picks(), self.tmp / "o"
        ident = {"agent": {"generation": 13}, "draws": 2}
        with mock.patch.object(rse, "play_task", fake_task):
            first = rse.run_block(picks, 2, procs=0, out=out, config="a3-v1", ascension=10, agent=StubAgent(), identity=ident, log=lambda s: None)
            self.assertEqual(12, first["played"])
            (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_of(out)[:7]) + '{"key": "cut', encoding="utf-8")  # a stop cut the last line
            again = rse.run_block(picks, 2, procs=0, out=out, config="a3-v1", ascension=10, agent=StubAgent(), identity=ident, log=lambda s: None)
        self.assertEqual(5, again["played"])
        rows = rows_of(out)
        self.assertEqual(12, len({(r["key"], r["draw"]) for r in rows}))
        self.assertEqual(12, len(rows))

    def test_the_first_draw_of_every_seed_comes_before_any_second_draw(self):
        picks, out = self.picks(), self.tmp / "o"
        with mock.patch.object(rse, "play_task", fake_task):
            rse.run_block(picks, 2, procs=0, out=out, config="a3-v1", ascension=10, agent=StubAgent(), log=lambda s: None)
        self.assertEqual([0] * 6 + [1] * 6, [r["draw"] for r in rows_of(out)])

    def test_another_exams_rows_are_refused_unless_forced(self):
        picks, out = self.picks(), self.tmp / "o"
        with mock.patch.object(rse, "play_task", fake_task):
            rse.run_block(picks, 1, procs=0, out=out, config="a3-v1", ascension=10, agent=StubAgent(), identity={"a": 1}, log=lambda s: None)
            with self.assertRaises(SystemExit):
                rse.run_block(picks, 1, procs=0, out=out, config="a3-v1", ascension=10, agent=StubAgent(), identity={"a": 2}, log=lambda s: None)
            forced = rse.run_block(picks, 1, procs=0, out=out, config="a3-v1", ascension=10, agent=StubAgent(), identity={"a": 2}, force=True, log=lambda s: None)
        self.assertEqual(6, forced["played"])

    def test_a_worker_that_died_is_run_again_once_a_timeout_is_a_lost_row_and_no_seed_is_logged(self):
        import spawn_pool  # noqa: PLC0415

        picks, out = self.picks(4), self.tmp / "o"
        died, slow = picks[0]["key"], picks[1]["key"]
        FakePool.script, FakePool.seen = {(died, 1): "died", (slow, 1): "timeout"}, []
        lines: list[str] = []
        with mock.patch.object(rse, "play_task", fake_task), mock.patch.object(spawn_pool, "SupervisedPool", FakePool):
            summ = rse.run_block(picks, 1, procs=2, out=out, config="a3-v1", ascension=10, agent=StubAgent(), log=lines.append)
        rows = {r["key"]: r for r in rows_of(out)}
        self.assertEqual(4, len(rows))
        self.assertIn(rows[died]["end"], ("win", "death"))  # run again, and it finished
        self.assertTrue(rows[slow]["lost"])
        self.assertEqual("lost", rows[slow]["end"])
        self.assertEqual(1, summ["retried"])
        self.assertEqual(2, FakePool.seen.count((died, 0)))
        text = "\n".join(lines) + all_text(out)
        for p in picks:
            self.assertNotIn(p["seed"], text)

    def test_a_worker_that_dies_twice_is_written_as_lost(self):
        import spawn_pool  # noqa: PLC0415

        picks, out = self.picks(2), self.tmp / "o"
        k = picks[0]["key"]
        FakePool.script, FakePool.seen = {(k, 1): "died", (k, 2): "died"}, []
        with mock.patch.object(rse, "play_task", fake_task), mock.patch.object(spawn_pool, "SupervisedPool", FakePool):
            rse.run_block(picks, 1, procs=1, out=out, config="a3-v1", ascension=10, agent=StubAgent(), log=lambda s: None)
        row = {r["key"]: r for r in rows_of(out)}[k]
        self.assertEqual(("lost", True), (row["end"], row["lost"]))


class HookTests(Base):
    def test_the_hook_reads_the_door_at_the_first_decision_of_each_boss_fight_and_never_ends_the_run(self):
        import gen0  # noqa: PLC0415

        class Sess:
            def __init__(self, floor, node, hp, act):
                self.info = {"floor": floor, "node_type": node, "hp": hp, "max_hp": 80}
                self.act_index = act

        def fake_play(seed, ix, config, ascension, agent=None, hook=None, **kw):
            assert hook(0, None, Sess(3, 1, 70, 0), None) is False  # a hallway: no door
            assert hook(1, None, Sess(17, hb.BOSS_NODE, 55, 0), None) is False  # the Act 1 boss, first decision
            assert hook(2, None, Sess(17, hb.BOSS_NODE, 30, 0), None) is False  # later in the same fight: not a door
            assert hook(3, None, Sess(33, hb.BOSS_NODE, 61, 1), None) is False
            return {"seed": seed, "ix": ix, "win": False, "floor": 40, "end": "death", "decisions": 9, "wall": 1.0, "crash": False,
                    "boss_damage_acts": [[0, 1.0], [1, 0.4]], "config": config, "generation": 13}

        with mock.patch.object(gen0, "_setup"), mock.patch.object(gen0, "play_emu_run", fake_play):
            row = rse.play_task("S1", rse.seed_key("S1"), 1, 7, "a3-v1", 10, StubAgent(), False)
        self.assertEqual({"1": [55, 80], "2": [61, 80]}, row["door"])
        self.assertNotIn("seed", row)
        self.assertEqual((rse.seed_key("S1"), 1, False), (row["key"], row["draw"], row["human_won"]))
        run = rse.ours_run(row)
        self.assertEqual({1: {"hp": 55, "max_hp": 80, "killed": True}, 2: {"hp": 61, "max_hp": 80, "killed": False}}, run["doors"])

    def test_the_lab_boss_node_is_the_one_the_human_bar_uses(self):
        text = (ROOT / "rl" / "run_train" / "vocab.py").read_text(encoding="utf-8")
        self.assertEqual(hb.BOSS_NODE, int(re.search(r"^NODE_BOSS\s*=\s*(\d+)", text, re.M).group(1)))


if __name__ == "__main__":
    unittest.main()
