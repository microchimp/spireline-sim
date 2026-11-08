"""Judge v1's data and scoring (rl/judge_data.py `buffer`, rl/judge.py `eval`; training design 1.3's fallback, 2026-11-08).

The pinned facts: a buffer fight is a label only when it finished and kept its value (`value_ok`); its targets are its outcome (the
potions it held at the end where the shard carries them, else those it began with less those it used); the exam's and the boards'
runs and starts are left out whole; the held-out fifth is chosen by RUN, so every start of a run sits on one side; the rows of a
shard come out as the same positions `build` writes (tokens flat, one label row a position, a start index); `Positions` takes the
validation tenth by run too; `evaluate` reports by category and by source; and `eval` scores several judges on the same positions
and on the starts a manifest's judge never trained on.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import judge_data as jd  # noqa: E402
import practice_mix  # noqa: E402


def fight(start, seed, cat="act1-boss", won=True, end_hp=30, max_hp=80, j=0, value_ok=True, capped=False, **more):
    return {"start": start, "j": j, "cat": cat, "player": "c0001", "config": "x16", "split": "train", "seed": seed, "won": won,
            "end_hp": end_hp if won else 0, "max_hp": max_hp, "L0": 0.6 if won else 0.0, "capped": capped, "potions_used": 1,
            "turns": 5, "decisions": 3, "start_hp": 60, "start_potions": 2, "encounter": 9, "deck": 20, "relics": 3,
            "label": 0, "value_ok": value_ok, **more}


def write_shard(folder: Path, name: str, metas: list[dict], rows_per_fight: list[int], segment: int = 1, seed: int = 0) -> dict:
    """A shard in the loop's layout: per row a token block of 4 to 9 tokens, root visits and values, the fight it belongs to."""
    from prior_client import synthetic_tokens

    rng = np.random.default_rng(seed)
    fight_of = np.concatenate([np.full(n, i, np.int32) for i, n in enumerate(rows_per_fight)])
    toks = [synthetic_tokens(rng, int(rng.integers(4, 10))) for _ in fight_of]
    n = len(fight_of)
    visits = np.zeros((n, 32), np.float32)
    qvals = np.zeros((n, 32), np.float32)
    visits[:, 0], visits[:, 1] = 3.0, 1.0
    qvals[:, 0], qvals[:, 1] = -10.0, 6.0
    visits[0] = 0.0  # a forced move: no root value
    arrays = {
        "types": np.concatenate([t.types for t in toks]).astype(np.int16), "ids": np.concatenate([t.ids for t in toks]).astype(np.int32),
        "owners": np.concatenate([t.owners for t in toks]).astype(np.int16), "feats": np.concatenate([t.feats for t in toks]).astype(np.float32),
        "lengths": np.array([len(t.types) for t in toks], np.int32), "visits": visits, "qvals": qvals, "legal": np.ones((n, 32), np.int8),
        "sims": np.full(n, 480, np.int32), "k": np.arange(n, dtype=np.int16), "exact": np.zeros(n, np.int8),
        "exact_value": np.zeros(n, np.float32), "q_played": np.zeros(n, np.float32), "solver_best": np.zeros(n, np.int16),
        "played": np.zeros(n, np.int16), "fight": fight_of, "label": np.zeros(n, np.int16), "value_ok": np.ones(n, np.int8),
        "val": np.zeros(n, np.int8), "fights_meta": np.array(json.dumps(metas)),
    }
    (folder / "buffer" / "shards").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "buffer" / "shards" / name, **arrays)
    return {"shard": name, "segment": segment, "rows": n, "fights": len(metas), "t": f"2026-11-0{segment}T10:00:00Z"}


def make_loop(folder: Path) -> list[dict]:
    """Two shards: runs R1 (two starts, a loss among them), R2 (a human start; shard 1 has no `potions_held`), the capped fight, the
    fight of a segment that lost its value labels, a fight of the excluded run X1, and a pre-loss start of R1."""
    m1 = [fight("2026-11-01/g9:R1|f17", "R1", won=True, end_hp=40),
          fight("2026-11-01/g9:R1|f33", "R1", cat="act2-boss", won=False, j=1),
          fight("2026-11-01/g9:R1|f17", "R1", won=True, end_hp=20, j=1, capped=True),
          fight("2026-11-01/g9:R3|f05", "R3", cat="act1-hallway", won=True, value_ok=False),
          fight("2026-11-01/g9:X1|f17", "X1", won=True)]
    m2 = [fight("human:R2|f17", "R2", won=True, end_hp=16, potions_held=1),
          fight("preloss:2026-11-01/g9:R1|f33|j1|k7", "R1", cat="act2-boss", won=True, end_hp=8, potions_held=0)]
    index = [write_shard(folder, "s000001-1.npz", m1, [4, 3, 2, 2, 5], 1, 1), write_shard(folder, "s000002-2.npz", m2, [3, 2], 2, 2)]
    (folder / "buffer" / "index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in index), encoding="utf-8")
    return index


class PureTests(unittest.TestCase):
    def test_a_start_ids_source_and_run(self):
        self.assertEqual([jd.start_source(s) for s in ("2026-11-01/g9:R1|f17", "human:R2|f17", "preloss:human:R2|f33|j0|k24")],
                         ["night", "human", "preloss"])
        self.assertEqual([jd.bare_run_of(s) for s in ("R1|f17", "2026-11-01/g9:R1|f17", "human:R2|f17", "preloss:human:R2|f33|j0|k24",
                                                      "preloss:2026-11-01/g9:R1|f33|j1|k7")], ["R1", "R1", "R2", "R2", "R1"])

    def test_labels_are_the_fights_outcome_with_the_potions_it_held(self):
        won = fight("s|f17", "R", end_hp=40, potions_held=2)
        self.assertEqual(jd.buffer_labels(won), {"won": True, "hp_frac": 0.5, "potions": 2, "L0": 0.6})
        old = fight("s|f17", "R", end_hp=40)  # a shard from before potions_held: began with 2, used 1
        self.assertEqual(jd.buffer_labels(old)["potions"], 1)
        self.assertEqual(jd.buffer_labels({**old, "potions_used": 5})["potions"], 0)  # never negative
        lost = fight("s|f17", "R", won=False, potions_held=3)
        self.assertEqual(jd.buffer_labels(lost), {"won": False, "hp_frac": 0.0, "potions": 0, "L0": 0.0})

    def test_only_a_finished_fight_with_its_value_is_a_label(self):
        self.assertTrue(jd.buffer_usable(fight("s|f1", "R")))
        self.assertFalse(jd.buffer_usable(fight("s|f1", "R", capped=True)))
        self.assertFalse(jd.buffer_usable(fight("s|f1", "R", value_ok=False)))

    def test_the_search_value_of_a_row_is_the_visit_weighted_mean(self):
        rng = np.random.default_rng(0)
        visits = rng.integers(0, 5, (20, 32)).astype(np.float32)
        visits[3] = 0
        q = rng.normal(0, 30, (20, 32)).astype(np.float32)
        q[5, 4] = np.nan
        got = jd.root_values(visits, q)
        for i in range(20):
            want = practice_mix.root_value(visits[i], q[i])
            if want is None:
                self.assertTrue(np.isnan(got[i]))
            else:
                self.assertAlmostEqual(float(got[i]), want, places=3)

    def test_exclusions_read_the_exam_file_and_a_boards_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            exam = Path(tmp) / "exclude_runs.json"
            exam.write_text(json.dumps({"rule": "x", "run_seeds": ["R9"], "start_ids": ["a:R8|f05"]}), encoding="utf-8")
            board = Path(tmp) / "meta.json"
            board.write_text(json.dumps({"act1_boss_starts": ["B1|f17"], "timing_starts": {"act1-elite": ["2026-11-01/g9:T1|f05"]},
                                         "procs": 12}), encoding="utf-8")
            self.assertEqual(jd.excluded_from(exam), ({"R9"}, {"a:R8|f05"}))
            runs, ids = jd.excluded_from(board)
            self.assertEqual((runs, ids), ({"B1", "T1"}, {"B1|f17", "2026-11-01/g9:T1|f05"}))


class BufferTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.loop = Path(self._tmp.name)
        self.index = make_loop(self.loop)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_usable_fights_come_out_as_positions_with_their_outcome(self):
        arrays, rep = jd.read_buffer(self.loop, skip_runs={"X1"})
        # shard 1: fights 0 (4 rows) and 1 (3 rows) are usable; the capped, the unlabelled and X1's are not. shard 2: both (3 + 2 rows).
        self.assertEqual((rep["fights"], rep["positions"], rep["starts"], rep["runs"]), (4, 12, 4, 2))
        self.assertEqual(rep["skipped_fights"], {"capped": 1, "no value label": 1, "excluded run or start": 1})
        self.assertEqual(rep["window"], ["2026-11-01T10:00:00Z", "2026-11-02T10:00:00Z"])
        self.assertEqual(rep["by_src"], {"human": 1, "night": 2, "preloss": 1})
        self.assertEqual(rep["by_cat"], {"act1-boss": 2, "act2-boss": 2})
        lens = arrays["lens"]
        self.assertEqual(len(lens), 12)
        self.assertEqual(int(lens.astype(np.int64).sum()), len(arrays["types"]))
        self.assertEqual(arrays["feats"].shape, (len(arrays["types"]), 18))
        lab = arrays["labels"]
        self.assertEqual(lab.shape, (12, 4))
        won = lab[:, 0]
        self.assertEqual(won.tolist(), [1] * 4 + [0] * 3 + [1] * 5)
        self.assertAlmostEqual(float(lab[0, 1]), 40 / 80)
        self.assertEqual(lab[0, 2], 1)  # shard 1 has no potions_held: began with 2, used 1
        self.assertEqual(lab[7, 2], 1)  # the human fight carried potions_held = 1
        self.assertEqual(lab[7, 1], np.float32(16 / 80))
        meta = arrays["meta"]
        self.assertTrue(np.isnan(meta[0, jd.META.index("root_value")]))  # the forced move
        self.assertAlmostEqual(float(meta[1, jd.META.index("root_value")]), (3 * -10 + 6) / 4, places=4)
        self.assertEqual(meta[:, jd.META.index("max_hp")].tolist(), [80.0] * 12)
        # one start index a start, and a (start, j, rows) row a fight
        self.assertEqual(arrays["start_ids"].tolist(), ["2026-11-01/g9:R1|f17", "2026-11-01/g9:R1|f33", "human:R2|f17",
                                                        "preloss:2026-11-01/g9:R1|f33|j1|k7"])
        self.assertEqual(arrays["group"].tolist(), [0] * 4 + [1] * 3 + [2] * 3 + [3] * 2)
        self.assertEqual(arrays["games"].tolist(), [[0, 0, 4], [1, 1, 3], [2, 0, 3], [3, 0, 2]])
        self.assertEqual(arrays["start_run"].tolist(), ["R1", "R1", "R2", "R1"])
        self.assertEqual([str(arrays["srcs"][i]) for i in arrays["start_src"]], ["night", "night", "human", "preloss"])

    def test_a_run_is_held_out_whole(self):
        arrays, _ = jd.read_buffer(self.loop, skip_runs={"X1"})
        by_run = {}
        for run, held in zip(arrays["start_run"].tolist(), arrays["start_heldout"].tolist()):
            by_run.setdefault(run, set()).add(held)
        self.assertTrue(all(len(v) == 1 for v in by_run.values()), by_run)  # R1's three starts agree
        for run, v in by_run.items():
            self.assertEqual(v, {jd.split_of(run) == "heldout"})

    def test_the_exam_and_the_boards_are_left_out_by_run_and_by_start(self):
        arrays, rep = jd.read_buffer(self.loop, skip_runs={"R2", "X1"}, skip_ids={"2026-11-01/g9:R1|f33"})
        self.assertEqual(sorted(arrays["start_ids"].tolist()), ["2026-11-01/g9:R1|f17", "preloss:2026-11-01/g9:R1|f33|j1|k7"])
        self.assertEqual(rep["skipped_fights"]["excluded run or start"], 3)  # X1's fight, R2's human fight, R1|f33's lost fight

    def test_the_command_writes_the_npz_and_a_manifest_and_finds_no_contamination(self):
        exam = self.loop / "exam"
        exam.mkdir()
        (exam / "exclude_runs.json").write_text(json.dumps({"run_seeds": ["X1"], "start_ids": []}), encoding="utf-8")
        board = self.loop / "board.jsonl"
        board.write_text(json.dumps({"id": "Z9|f17", "cat": "act1-boss"}) + "\n", encoding="utf-8")
        out = self.loop / "out"
        rc = jd.main(["buffer", "--loop", str(self.loop), "--out", str(out), "--board", str(board)])
        self.assertEqual(rc, 0)
        man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(man["contamination"], {"exam_starts": 1, "same_id": 0, "same_run": 0})
        self.assertEqual((man["fights"], man["positions"]), (4, 12))  # X1's fight is out; so are the capped and the unlabelled
        z = np.load(out / "positions.npz")
        self.assertEqual(int(z["lens"].shape[0]), man["positions"])
        self.assertEqual(man["heldout_positions"], int(z["start_heldout"][z["group"]].sum()))
        # a start of the excluded board run, if it were in the file, would be seen by the check
        bad = jd.run_contamination({"start_ids": np.array(["R1|f17"]), "start_run": np.array(["R1"])}, ["night:R1|f05"])
        self.assertEqual((bad["same_id"], bad["same_run"]), (0, 1))


class ScoringTests(unittest.TestCase):
    """`Positions`, `evaluate` and `eval` on a buffer-made npz with a seeded random judge (not one that judges well)."""

    @classmethod
    def setUpClass(cls):
        import torch
        from test_judge import export_judge

        cls.torch = torch
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        rng = np.random.default_rng(3)
        metas, rows, runs = [], [], []
        for r in range(40):  # 40 runs, 3 fights each, mixed kinds and sources
            run = f"RUN{r:03d}"
            for f, (cat, src) in enumerate((("act1-boss", "night"), ("act2-boss", "human"), ("act1-hallway", "night"))):
                sid = (f"human:{run}|f{17 + f}" if src == "human" else f"2026-11-01/g9:{run}|f{17 + f}")
                metas.append(fight(sid, run, cat=cat, won=bool(rng.random() < 0.5), end_hp=int(rng.integers(5, 60)), max_hp=int(rng.integers(60, 100))))
                rows.append(int(rng.integers(20, 30)))
        index = [write_shard(tmp, "s000001-1.npz", metas, rows, 1, 7)]
        (tmp / "buffer" / "index.jsonl").write_text(json.dumps(index[0]) + "\n", encoding="utf-8")
        arrays, cls.report = jd.read_buffer(tmp)
        cls.npz = tmp / "positions.npz"
        np.savez_compressed(cls.npz, **arrays)
        cls.judge_a = export_judge(tmp / "a.pt", 0)
        cls.judge_b = export_judge(tmp / "b.pt", 1)
        cls.tmp = tmp

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_validation_tenth_is_by_run_and_the_splits_partition_the_positions(self):
        import judge

        data = judge.Positions(self.npz, self.torch.device("cpu"))
        self.assertEqual(data.start_runs[:3], ["RUN000"] * 3)
        for run in set(data.start_runs):
            groups = [i for i, r in enumerate(data.start_runs) if r == run]
            self.assertEqual(len({int(data.split[np.flatnonzero(data.group == g)[0]]) for g in groups}), 1)  # one side for a run
        counts = np.bincount(data.split, minlength=3)
        self.assertEqual(int(counts.sum()), len(data.split))
        self.assertTrue(counts[2] > 0 and counts[1] > 0 and counts[0] > counts[1])  # train, validation and held out all hold something
        self.assertEqual(data.srcs, ["night", "human", "preloss"])

    def test_evaluate_reports_by_category_and_source(self):
        import judge

        data = judge.Positions(self.npz, self.torch.device("cpu"))
        model, _ = judge.load(self.judge_a)
        rep = judge.evaluate(model, data, 0)
        self.assertEqual(set(rep["by_cat"]), {"act1-boss", "act2-boss", "act1-hallway"})
        self.assertEqual(set(rep["by_src"]), {"night", "human"})
        self.assertEqual(sum(v["positions"] for v in rep["by_cat"].values()), rep["positions"])
        self.assertIn("win_brier", rep["by_cat"]["act1-boss"]["judge"])
        self.assertNotIn("search_best_p_win", rep)  # the buffer has no best-move death chance
        self.assertIn("search_root_value", rep)

    def test_eval_scores_both_judges_on_the_same_positions_and_the_unseen_runs(self):
        import judge

        manifest = self.tmp / "v0_manifest.json"
        manifest.write_text(json.dumps({"train": ["RUN000|f17", "RUN001|f17"], "heldout": ["RUN002|f05"]}), encoding="utf-8")
        out = self.tmp / "eval.json"
        rc = judge.main(["eval", "--data", str(self.npz), "--judge", str(self.judge_a), "--judge", str(self.judge_b),
                         "--unseen-by", str(manifest), "--out", str(out), "--cpu"])
        self.assertEqual(rc, 0)
        rep = json.loads(out.read_text(encoding="utf-8"))
        a, b = (rep["judges"][str(p)] for p in (self.judge_a, self.judge_b))
        self.assertEqual(a["all"]["positions"], b["all"]["positions"])  # the same positions
        self.assertNotEqual(a["sha256"], b["sha256"])
        self.assertLessEqual(a["unseen_by"]["positions"], a["all"]["positions"])
        # the manifest's three runs are seen: no position of them is in the unseen subset
        data = judge.Positions(self.npz, self.torch.device("cpu"))
        seen = np.array([r in {"RUN000", "RUN001", "RUN002"} for r in data.start_runs])
        held = data.split == 2
        want = int((held & ~seen[data.group]).sum())
        self.assertEqual(a["unseen_by"]["positions"], want)

    def test_train_keeps_the_version_and_the_datas_window(self):
        import judge

        out = self.tmp / "v1" / "judge.pt"
        man = self.tmp / "manifest.json"
        man.write_text(json.dumps({**self.report, "sha256": "ab" * 32}), encoding="utf-8")
        rc = judge.main(["train", "--data", str(self.npz), "--out", str(out), "--epochs", "1", "--d-model", "32", "--layers", "1",
                         "--batch", "64", "--pause", "0", "--cpu", "--version", "judge-v1", "--manifest", str(man)])
        self.assertEqual(rc, 0)
        blob = self.torch.load(out, map_location="cpu", weights_only=True)
        self.assertEqual(blob["version"], "judge-v1")
        self.assertEqual(blob["trained"]["buffer"]["window"], self.report["window"])
        self.assertEqual(blob["trained"]["buffer"]["fights"], self.report["fights"])
        judge.load(out)  # a v1 checkpoint loads as any judge
        self.assertTrue(out.with_suffix(".report.json").is_file())


if __name__ == "__main__":
    unittest.main()
