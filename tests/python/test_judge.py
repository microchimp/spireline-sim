"""The fight judge v0 (rl/judge.py, rl/judge_client.py, rl/judge_data.py) and its leaf in the search (D41).

The pinned facts: the training starts never touch the exam (the fight board's 200 Act-1 boss starts: no shared start, run seed or
position, and not the board's source night), and the held-out fifth is split by start id; a game's recorded positions round-trip;
the judge's row reads as the search's value in its units; the prior server in judge mode answers a batch as the CPU judge does;
the search with JUDGE_V0's settings asks the judge at its leaves, never rolls out there, and returns a legal move; the lockstep gym
plays a judge arm's games; and the trained judge clears its accuracy floor on held-out positions (when the lab's files are here).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
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

import judge_client as jc  # noqa: E402
import judge_data as jd  # noqa: E402
import search_configs as sc  # noqa: E402

MANIFEST = HERE / "fixtures" / "judge_v0_manifest.json"
HELDOUT = HERE / "fixtures" / "judge_v0_heldout.npz"
LAB_JUDGE = Path(os.environ.get("SPIRELINE_JUDGE", "D:/spireline/referee/agent/judge/v0/judge.pt"))
LAB_DATA = Path(os.environ.get("SPIRELINE_JUDGE_DATA", "D:/spireline/referee/lab/results/judge/2026-10-30-v0"))
# The floors the v0 judge must clear on its held-out positions (set from its training report, docs/rl/JUDGE.md): the L0
# correlation a little under what it measured, and a calibration error a little over.
PEARSON_FLOOR = 0.80  # v0 measured 0.880 on the fixture (0.872 on all 28,326 held-out positions); the untrained net 0.467
ECE_CEILING = 0.06  # v0 measured 0.034 on the fixture (0.027 on all held-out positions); the untrained net 0.254


def export_judge(path: Path, seed: int = 0) -> Path:
    """A small judge checkpoint with seeded random weights: a network the real loader plays, not one that judges well."""
    import judge
    import torch
    from combat_tokens import VOCAB

    torch.manual_seed(seed)
    shape = {"n_buckets": 19, "d_model": 32, "layers": 1, "heads": 2}
    model = judge.JudgeNet(VOCAB, **shape).eval()
    torch.save({"kind": judge.KIND, "version": "test", "model": model.state_dict(), **shape}, path)
    return path


# ---------------------------------------------------------------------------------------------------------- data


class DataTests(unittest.TestCase):
    RUNS = [
        {"seed": "AAA", "ix": 0, "end": "death", "floor": 20, "nodes": "M?MERSTMEBMMEBRRMB", "crash": None},
        {"seed": "BBB", "ix": 1, "end": "death", "floor": 5, "nodes": "MEMMMMMMMMMMMMMB", "crash": None},
        {"seed": "CCC", "ix": 2, "end": "death", "floor": 30, "nodes": "MMMMMMMMMMMMMMMB", "crash": None},
        {"seed": "DDD", "ix": 3, "end": None, "floor": 9, "nodes": "MMM", "crash": "boom"},
    ]

    def test_every_fight_on_the_path_up_to_the_floor_reached_and_never_an_excluded_run(self):
        got = jd.select_fight_starts(self.RUNS, 10, 0, exclude_seeds={"CCC"})
        ids = {s["id"] for s in got}
        self.assertFalse(any(i.startswith(("CCC|", "DDD|")) for i in ids))  # excluded, crashed
        self.assertEqual(sorted(i for i in ids if i.startswith("BBB|")), ["BBB|f02", "BBB|f03", "BBB|f04", "BBB|f05"])
        a = {s["id"]: s for s in got if s["seed"] == "AAA"}
        self.assertEqual(sorted(a), ["AAA|f02", "AAA|f04", "AAA|f05", "AAA|f09", "AAA|f10", "AAA|f11", "AAA|f12", "AAA|f13",
                                     "AAA|f14", "AAA|f15", "AAA|f18", "AAA|f19"])  # ? R S T are not fights
        self.assertEqual(a["AAA|f05"]["cat"], "act1-elite")
        self.assertEqual(a["AAA|f11"]["cat"], "act1-boss")
        self.assertEqual(a["AAA|f19"]["cat"], "act2-boss")
        self.assertEqual(jd.select_fight_starts(self.RUNS, 10, 0, set()), jd.select_fight_starts(self.RUNS, 10, 0, set()))

    def test_a_fifth_is_held_out_by_start_id(self):
        ids = [f"S{i:04d}|f{f:02d}" for i in range(400) for f in (5, 17)]
        held = [i for i in ids if jd.split_of(i) == "heldout"]
        self.assertTrue(0.15 < len(held) / len(ids) < 0.25, len(held))
        self.assertEqual(held, [i for i in ids if jd.split_of(i) == "heldout"])  # a function of the id alone

    def test_contamination_counts_shared_starts_run_seeds_and_positions(self):
        rep = jd.contamination(["A|f02", "B|f17", "C|f05"], ["B|f17", "C|f17"], {"A|f02": "x", "C|f05": "y"}, {"B|f17": "y"})
        self.assertEqual(rep, {"train_starts": 3, "exam_starts": 2, "same_id": 1, "same_run_seed": 2, "same_position": 1})

    def test_positions_round_trip(self):
        rng = np.random.default_rng(0)
        recs = []
        for k in range(3):
            n = int(rng.integers(5, 30))
            tok = jc_tokens(rng, n)
            recs.append({"tokens": tok, "k": k, "turn": 1 + k // 2, "hp": 50 - k, "max_hp": 80, "root_value": None if k == 1 else 3.5,
                         "best_value": 4.0, "best_p_death": 0.1, "sims": 30})
        z = jd.unpack_positions(jd.pack_positions(recs))
        self.assertEqual(z["lens"].tolist(), [len(r["tokens"]) for r in recs])
        self.assertTrue(np.array_equal(z["types"], np.concatenate([r["tokens"].types for r in recs])))
        self.assertTrue(np.array_equal(z["feats"], np.concatenate([r["tokens"].feats for r in recs])))
        self.assertTrue(np.isnan(z["meta"][1, jd.META.index("root_value")]))
        self.assertEqual(z["meta"][2, jd.META.index("hp")], 48)
        self.assertEqual(jd.unpack_positions(jd.pack_positions([]))["lens"].shape, (0,))

    def test_labels_are_the_games_outcome(self):
        self.assertEqual(jd.labels_of({"won": True, "end_hp": 40, "max_hp": 80, "potions_held": 2, "L0": 0.95}),
                         {"won": True, "hp_frac": 0.5, "potions": 2, "L0": 0.95})
        self.assertEqual(jd.labels_of({"won": False, "end_hp": 0, "max_hp": 80, "potions_held": 3, "L0": 0.0}),
                         {"won": False, "hp_frac": 0.0, "potions": 0, "L0": 0.0})


def jc_tokens(rng, n):
    from prior_client import synthetic_tokens

    return synthetic_tokens(rng, n)


class ContaminationTests(unittest.TestCase):
    """The judge v0's training starts (the committed manifest) against the exam: nothing shared."""

    def test_no_training_or_heldout_start_shares_a_run_with_the_board(self):
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        exam = set(m["exam"])
        self.assertEqual(len(exam), 200)
        exam_seeds = {jd.run_seed_of(i) for i in exam}
        for side in ("train", "heldout"):
            self.assertGreater(len(m[side]), 100)
            self.assertFalse(set(m[side]) & exam, side)
            self.assertFalse({jd.run_seed_of(i) for i in m[side]} & exam_seeds, side)
        self.assertFalse(set(m["train"]) & set(m["heldout"]))
        self.assertTrue(all(jd.split_of(i) == "heldout" for i in m["heldout"]))
        self.assertTrue(all(jd.split_of(i) == "train" for i in m["train"]))
        self.assertEqual({k: v for k, v in m["contamination"].items() if k.startswith(("same_", "board_"))},
                         {"same_id": 0, "same_run_seed": 0, "same_position": 0, "board_source_starts": 0})

    def test_the_board_source_night_is_refused(self):
        from tools.solver import coverage

        self.assertEqual(coverage.source_label("X:/a/LAB/nightly/2026-10-27/g7-emu"), jd.BOARD_SOURCE)

    def test_the_lab_files_match_the_manifest(self):
        if not (LAB_DATA / "starts.jsonl").is_file() or not jd.BOARD_STARTS.is_file():
            self.skipTest("the lab's judge data is not on this machine")
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        now = jd.manifest(LAB_DATA)
        self.assertEqual((now["train"], now["heldout"], now["exam"]), (m["train"], m["heldout"], m["exam"]))
        self.assertEqual(now["contamination"]["same_position"], 0)
        z = np.load(LAB_DATA / "positions.npz") if (LAB_DATA / "positions.npz").is_file() else None
        if z is not None:  # every position trained on comes from a manifest start, on the side the manifest says
            ids = [str(s) for s in z["start_ids"]]
            held = z["start_heldout"].astype(bool)
            self.assertTrue(set(ids) <= set(m["train"]) | set(m["heldout"]))
            self.assertEqual({i for i, h in zip(ids, held) if h}, {i for i in ids if i in set(m["heldout"])})


# ---------------------------------------------------------------------------------------------------------- the judge's numbers


class ClientTests(unittest.TestCase):
    def test_a_row_reads_as_the_searchs_value(self):
        sure_win = jc.estimate([20.0, 0.0, 1.0], 80)  # P(win) ~ 1, half HP, one potion
        self.assertAlmostEqual(sure_win.value, 40 + 10, places=4)
        self.assertAlmostEqual(sure_win.p_death, 0.0, places=6)
        sure_death = jc.estimate([-20.0, 3.0, 2.0], 80)
        self.assertAlmostEqual(sure_death.value, jc.DEATH_VALUE, places=4)
        even = jc.estimate([0.0, 0.0, 0.0], 60)
        self.assertAlmostEqual(even.value, 0.5 * 30 + 0.5 * jc.DEATH_VALUE)
        self.assertAlmostEqual(even.hp_if_win, 30.0)

    def test_l0_is_the_gyms_l0_in_expectation(self):
        import fight_gym as fg

        self.assertAlmostEqual(float(jc.l0_of(1.0, 0.5, 1, 80)), fg.l0(True, 40, 1, 80))
        self.assertAlmostEqual(float(jc.l0_of(1.0, 0.95, 2, 80)), fg.l0(True, 76, 2, 80))
        self.assertAlmostEqual(float(jc.l0_of(0.25, 0.5, 0, 80)), 0.25 * fg.l0(True, 40, 0, 80))


class ModelTests(unittest.TestCase):
    def test_the_cpu_judge_and_a_cpu_device_server_in_judge_mode_agree(self):
        import combat_prior
        import judge
        import prior_client as pc

        spec = importlib.util.spec_from_file_location("prior_server_tests_for_judge", HERE / "test_prior_server.py")
        tps = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tps)
        with tempfile.TemporaryDirectory() as tmp:
            path = export_judge(Path(tmp) / "judge.pt")
            local = judge.LocalJudge(path)
            server = tps.ServerProc(path)
            combats = []
            jc.reset_state()
            try:
                self.assertTrue(server.ready.startswith("READY "), server.stderr())
                remote = jc.RemoteJudge(str(path), server.endpoint, bytes.fromhex(server.key), server.sha)
                combats = [tps.toadpoles(seed) for seed in range(5)]
                handles = [c.handle for c in combats]
                got = remote.rows(handles)
                want = local.rows([jc.encode_leaf(h) for h in handles])
                one = remote.rows(handles[:1])
                st = jc.stats()
                wrong = jc.RemoteJudge(str(path), server.endpoint, bytes.fromhex(server.key), "0" * 64)  # another judge's sha
                jc.reset_state()
                fell = wrong.rows(handles[:2])
                st_wrong = jc.stats()
            finally:
                for c in combats:
                    c.close()
                jc.reset_state()
                server.close()
        self.assertEqual(got.shape, (5, jc.JUDGE_OUT))
        self.assertLess(float(np.max(np.abs(got - want))), 1e-5)
        self.assertLess(float(np.max(np.abs(one[0] - want[0]))), 1e-6)
        self.assertEqual((st["server"], st["cpu"], st["fallback"]), (6, 0, None))
        self.assertIsNotNone(st_wrong["fallback"])  # HELLO refused: the CPU judge answers, the search goes on
        self.assertLess(float(np.max(np.abs(fell - want[:2]))), 1e-6)
        self.assertIsNone(pc.stats()["fallback_reason"])  # the prior's fallback is untouched
        self.assertTrue(combat_prior.vocab_sha256())

    def test_a_policy_checkpoint_is_not_a_judge(self):
        import combat_prior
        import judge
        from prior_fixture import export_prior

        with tempfile.TemporaryDirectory() as tmp:
            policy = export_prior(Path(tmp)) / combat_prior.POLICY
            with self.assertRaisesRegex(ValueError, "not a judge"):
                judge.load(policy)


# ---------------------------------------------------------------------------------------------------------- the search


class SearchTests(unittest.TestCase):
    def test_judge_v0_is_a3_v1_with_judged_leaves_and_more_simulations(self):
        a3, jv = sc.get("a3-v1"), sc.get("judge-v0")
        self.assertEqual(jv.leaf, "judge")
        self.assertNotIn("leaf", a3.search_kwargs(None))  # a rollout config's call is unchanged
        self.assertEqual(jv.search_kwargs(None)["leaf"], "judge")
        for pos in (sc.Position("play", 5, 0.5, True), sc.Position("target", 5, 0.5, True), sc.Position("select", 3, 0.2, False)):
            self.assertEqual(jv.sims_for(pos), a3.sims_for(pos) * sc.JUDGE_SIMS_X)
        self.assertEqual((jv.min_child_sims, jv.c, jv.confirm_pairs), (a3.min_child_sims, a3.c, a3.confirm_pairs))

    def test_the_search_asks_the_judge_at_its_leaves_and_returns_a_legal_move(self):
        import judge
        import live_search
        from test_combat_prior import toadpoles

        with tempfile.TemporaryDirectory() as tmp:
            local = judge.LocalJudge(export_judge(Path(tmp) / "judge.pt"))
            asked = []

            def leaf_fn(handle):
                asked.append(handle)
                return local.rows([jc.encode_leaf(handle)])[0]

            kw = sc.get("judge-v0").search_kwargs(sc.Position("play", 6, 1.0, True), env_prior=False)
            kw["max_sims"] = 40
            for seed in range(3):
                combat = toadpoles(seed)
                try:
                    legal = set(combat.legal())
                    before = len(asked)
                    res = live_search.search_live(combat.handle, seed=seed, leaf_fn=leaf_fn, **kw)
                finally:
                    combat.close()
                self.assertIn(res.best.action, legal)
                self.assertGreater(res.sims, 0)
                self.assertEqual(res.truncated_rollouts, 0)
                self.assertGreater(len(asked) - before, 0)
                self.assertLessEqual(len(asked) - before, res.sims)  # one judge call per simulation at most (a fight's end needs none)
                for c in res.children:
                    if c.visits:
                        self.assertTrue(np.isfinite(c.value_mean))
                        self.assertTrue(0.0 <= c.p_death <= 1.0)
            with self.assertRaisesRegex(ValueError, "leaf_fn"):
                combat = toadpoles(0)
                try:
                    live_search.search_live(combat.handle, seed=0, **kw)
                finally:
                    combat.close()

    def test_the_steps_yield_leaf_requests_the_driver_answers(self):
        import live_search
        from test_combat_prior import toadpoles

        combat = toadpoles(1)
        try:
            kw = {"max_sims": 5, "min_child_sims": 1, "confirm_pairs": 0, "leaf": "judge"}
            steps = live_search.search_steps(combat.handle, **kw)
            self.assertIsInstance(next(steps), live_search.LeafAsk)
            steps.close()
            with self.assertRaisesRegex(ValueError, "no leaf_fn"):
                live_search._drive(live_search.search_steps(combat.handle, **kw), None)
            with self.assertRaisesRegex(ValueError, "unknown leaf"):
                next(live_search.search_steps(combat.handle, leaf="oracle"))
        finally:
            combat.close()


def _make_arena_starts(queue) -> None:
    try:
        for p in (ROOT / "rl", ROOT / "src", ROOT, HERE):
            sys.path.insert(0, str(p))
        import gen0

        gen0.select_engine("core2")
        import ctypes

        from test_gym_lockstep import DECK

        from sts2_gym import native
        from sts2_gym.env import ENCOUNTER_IDS

        starts = []
        for i, (enc, seed, hp) in enumerate((("toadpoles", 3, 60), ("nibbits-weak", 5, 50))):
            h = native.create(seed)
            obs = (ctypes.c_int * native.OBS_SIZE)()
            native.reset_arena(h, DECK, obs, encounter_id=ENCOUNTER_IDS[enc], player_hp=hp, player_max_hp=80)
            doc = native.get_state_json(h)
            native.destroy(h)
            starts.append({"id": f"T{i}|f05", "cat": "act1-hall", "kind": "hall", "act": 1, "floor": 5, "node_type": 1, "hp": hp,
                           "max_hp": 80, "deck_size": len(DECK), "potions": 0, "doc": doc})
        queue.put(("ok", starts))
    except Exception as error:
        queue.put(("error", f"{type(error).__name__}: {error}"))


class LockstepJudgeTests(unittest.TestCase):
    """A judge arm's games in the lockstep gym (the CPU judge path), and a3-v1's recorded positions."""

    def test_a_judge_arm_plays_its_games_and_a3_v1_records_every_position(self):
        import fight_gym as fg
        import gym_lockstep as gl
        from prior_fixture import export_prior

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_prior(root / "agents" / "fixture", prior_min_visits=0)
            judge_path = export_judge(root / "judge.pt")
            ctx = mp.get_context("spawn")
            queue = ctx.Queue()
            p = ctx.Process(target=_make_arena_starts, args=(queue,), daemon=True)
            p.start()
            kind, got = queue.get(timeout=120)
            p.join(30)
            if kind != "ok":
                self.skipTest(f"native library unavailable: {got}")
            out = root / "out"
            out.mkdir()
            (out / "starts.jsonl").write_text("".join(json.dumps(s) + "\n" for s in got), encoding="utf-8")
            (out / "meta.json").write_text(json.dumps({"engine": "core2", "agent_root": str(root / "agents"), "agent": "fixture"}),
                                           encoding="utf-8")
            init = fg._init_args(out)
            sha = hashlib.sha256(judge_path.read_bytes()).hexdigest()
            tasks = [("judge-v0", f"T{i}|f05", 0, "act1-hall") for i in range(2)]
            stopped, stats = gl.run_lockstep(tasks, out / "judge.jsonl", 1, 2, init,
                                             {"prior": "cpu", "judge": {"path": str(judge_path), "sha256": sha}})
            self.assertIsNone(stopped)
            rows = fg.read_jsonl(out / "judge.jsonl")
            rec_tasks = [("a3-v1", f"T{i}|f05", 0, "act1-hall") for i in range(2)]
            stopped, _ = gl.run_lockstep(rec_tasks, out / "rec.jsonl", 1, 2, init, {"prior": "cpu", "record": True})
            self.assertIsNone(stopped)
            rec = fg.read_jsonl(out / "rec.jsonl")
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertFalse(r.get("capped"), r)
            self.assertEqual(r["judge_path"], "cpu")
            self.assertGreater(r["sims"], 0)
        self.assertGreater(stats[0]["leaves"], 0)
        self.assertEqual(len(rec), 2)
        for r in rec:
            z = jd.unpack_positions(r["_pos"])
            self.assertEqual(len(z["lens"]), r["decisions"])  # one position per decision
            self.assertEqual(z["meta"][:, 0].tolist(), list(range(r["decisions"])))


# ---------------------------------------------------------------------------------------------------------- accuracy


class AccuracyTests(unittest.TestCase):
    """The trained v0 judge on held-out positions (a fixed sample of the held-out fifth, committed) clears its floors."""

    def test_the_v0_judge_clears_its_floor_on_held_out_positions(self):
        if not LAB_JUDGE.is_file():
            self.skipTest("the v0 judge checkpoint is not on this machine")
        import judge

        z = np.load(HELDOUT)
        held = set(json.loads(MANIFEST.read_text(encoding="utf-8"))["heldout"])
        self.assertTrue({str(s) for s in z["start_ids"]} <= held)  # held-out starts only
        local = judge.LocalJudge(LAB_JUDGE)
        offs = np.concatenate([[0], np.cumsum(z["lens"].astype(np.int64))]).astype(np.int64)
        rows = []
        for at in range(0, len(z["lens"]), 512):
            toks = [_Tok(z, offs, i) for i in range(at, min(at + 512, len(z["lens"])))]
            rows.append(local.rows(toks))
        rows = np.concatenate(rows)
        p, l0 = judge.rows_to_win_l0(rows, np.maximum(1.0, z["max_hp"]))
        sc_ = judge.score(l0, z["labels"][:, 3], p, z["labels"][:, 0])
        self.assertGreaterEqual(sc_["pearson"], PEARSON_FLOOR, sc_)
        self.assertLessEqual(sc_["win"]["ece"], ECE_CEILING, sc_)


class _Tok:
    def __init__(self, z, offs, i):
        a, b = offs[i], offs[i + 1]
        self.types, self.ids, self.owners, self.feats = z["types"][a:b], z["ids"][a:b], z["owners"][a:b], z["feats"][a:b]


if __name__ == "__main__":
    unittest.main()
