"""Strong sets by achievement (rl/strong_sets.py; D55 addendum 15, 2026-11-08).

Pinned: the filter's rules (standard mode, not abandoned, one player, no modifiers, build, vanilla acts, one line per run hash, the exam
fifth out by the ids file and by the hash bucket) for Ascension 10 (all-player pool; S-win = its wins) and Ascension 11 to 20 (S-high);
a written run carries no token, badge or player id; the build's Ascension option; the shard loader's parts (a won-only part keeps the
winning runs' decisions, parts join with run indices offset, the held-out fifth is disjoint from the training fifths); the per-decision
measures equal `human_prior.metrics_from_scores`; the paired bootstrap; the selection rule. Synthetic records stand in for the corpus.
"""

from __future__ import annotations

import gzip
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import human_prior as hp  # noqa: E402
import human_starts as hs  # noqa: E402
import strong_sets as ss  # noqa: E402

ACTS = ["ACT.OVERGROWTH", "ACT.HIVE", "ACT.GLORY"]


def _ids(want_exam: bool, n: int, prefix: str = "s") -> list[str]:
    out, i = [], 0
    while len(out) < n:
        rid = f"{prefix}{i:014x}"
        if (hs.exam_bucket(rid) == hs.EXAM_BUCKET) == want_exam:
            out.append(rid)
        i += 1
    return out


def _run(rid: str, asc: int = 10, win: bool = False, **kw) -> dict:
    r = {"run_hash": rid, "ascension": asc, "game_mode": "standard", "was_abandoned": False, "win": win, "modifiers": [], "build_id": "v0.107.1",
         "acts": list(ACTS), "player_token": "tok", "map_point_history": [[{"map_point_type": "monster"}], [{"map_point_type": "monster"}]],
         "players": [{"character": "CHARACTER.IRONCLAD", "id": 1234, "badges": ["x"], "deck": []}]}
    r.update(kw)
    return r


def _line(r: dict) -> str:
    return json.dumps(r, separators=(",", ":")) + "\n"


class FilterRules(unittest.TestCase):
    def setUp(self):
        self.ids = _ids(False, 40)
        self.exam = frozenset()

    def why(self, r, seen=None):
        return ss.reject_reason(r, self.exam, set() if seen is None else seen)

    def test_levels_and_sets(self):
        win, lose = _run(self.ids[0], 10, True), _run(self.ids[1], 10, False)
        self.assertIsNone(self.why(win))
        self.assertEqual(("a10", "s-win"), ss.sets_of(win))
        self.assertEqual(("a10",), ss.sets_of(lose))
        for asc in (11, 15, 20):
            r = _run(self.ids[2], asc, win=asc == 15)
            self.assertIsNone(self.why(r))
            self.assertEqual(("s-high",), ss.sets_of(r))  # any outcome
        for asc in (0, 9, 21, 30):
            self.assertEqual("ascension", self.why(_run(self.ids[3], asc)))
            self.assertEqual((), ss.sets_of(_run(self.ids[3], asc)))

    def test_each_rule_refuses(self):
        cases = {"game_mode": {"game_mode": "daily"}, "abandoned": {"was_abandoned": True}, "modifiers": {"modifiers": ["x"]},
                 "build": {"build_id": "v0.106.0"}, "modded_acts": {"acts": ["ACT.OVERGROWTH", "ACT.MODDED", "ACT.GLORY"]}}
        for reason, patch in cases.items():
            self.assertEqual(reason, self.why(_run(self.ids[5], 10, **patch)), reason)
        two = _run(self.ids[6], 10, players=[{"character": "CHARACTER.IRONCLAD"}, {"character": "CHARACTER.IRONCLAD"}])
        self.assertEqual("multiplayer", self.why(two))
        silent = _run(self.ids[7], 10, players=[{"character": "CHARACTER.SILENT"}])
        self.assertEqual("character", self.why(silent))

    def test_one_line_per_run_hash(self):
        seen: set = set()
        self.assertIsNone(self.why(_run(self.ids[8]), seen))
        self.assertEqual("duplicate", self.why(_run(self.ids[8]), seen))

    def test_exam_fifth_is_out_by_hash_and_by_list(self):
        bucket0 = _ids(True, 1)[0]
        self.assertEqual("exam_fifth", self.why(_run(bucket0, 10, True)))
        self.assertEqual("exam_fifth", self.why(_run(bucket0, 15)))  # runs new to the pipeline join the fifths by the same hash rule
        listed = self.ids[9]
        self.assertEqual("exam_fifth", ss.reject_reason(_run(listed), frozenset({listed}), set()))

    def test_scrub_drops_token_badges_and_id(self):
        r = _run(self.ids[10])
        out = ss.scrub(r)
        self.assertNotIn("player_token", out)
        self.assertNotIn("id", out["players"][0])
        self.assertNotIn("badges", out["players"][0])
        self.assertEqual("CHARACTER.IRONCLAD", out["players"][0]["character"])
        self.assertIn("id", r["players"][0])  # the input is untouched


class FilterStream(unittest.TestCase):
    def test_counts_files_and_scrubbing(self):
        ids = _ids(False, 30)
        exam_id = _ids(True, 1)[0]
        runs = [_run(ids[0], 10, True), _run(ids[1], 10, True), _run(ids[2], 10, False), _run(ids[3], 12, False), _run(ids[4], 20, True),
                _run(ids[5], 9, True), _run(ids[6], 10, True, was_abandoned=True), _run(exam_id, 10, True), _run(ids[0], 10, True)]
        silent = _run(ids[7], 10, True, players=[{"character": "CHARACTER.SILENT", "id": 7}])
        lines = [_line(r) for r in runs] + [_line(silent), '{"character":"CHARACTER.IRONCLAD" broken\n']
        outs = {s: io.StringIO() for s in ss.FILES}
        doc = ss.filter_stream(lines, frozenset(), outs)
        s = doc["sets"]
        self.assertEqual({"runs": 3, "wins": 2}, {k: s["a10"][k] for k in ("runs", "wins")})
        self.assertEqual(2, s["s-win"]["runs"])
        self.assertEqual({"runs": 2, "wins": 1}, {k: s["s-high"][k] for k in ("runs", "wins")})
        self.assertEqual(1, doc["read"]["rejected|abandoned"])
        self.assertEqual(1, doc["read"]["rejected|exam_fifth"])
        self.assertEqual(1, doc["read"]["rejected|duplicate"])
        self.assertEqual(1, doc["read"]["unparsable"])
        for name, n in (("a10", 3), ("s-win", 2), ("s-high", 2)):
            written = [json.loads(x) for x in outs[name].getvalue().splitlines()]
            self.assertEqual(n, len(written), name)
            for w in written:
                self.assertNotIn("player_token", w)
                self.assertNotIn("id", w["players"][0])
        self.assertTrue(all(json.loads(x)["win"] for x in outs["s-win"].getvalue().splitlines()))
        by_fifth = sum(v for k, v in s["a10"].items() if k.startswith("fifth"))
        self.assertEqual(3, by_fifth)
        self.assertNotIn("fifth0", s["a10"])  # the exam fifth never counts
        self.assertEqual(3, s["a10"]["last_act2"])


class BuildOption(unittest.TestCase):
    def test_parse_ascensions(self):
        self.assertEqual((10,), hp.parse_ascensions("10"))
        self.assertEqual(tuple(range(11, 21)), hp.parse_ascensions("11-20"))
        self.assertEqual((11, 13), hp.parse_ascensions("11,13"))
        with self.assertRaises(ValueError):
            hp.parse_ascensions("")

    def test_human_ok_levels(self):
        self.assertTrue(hs.human_ok(_run("a", 10)))
        self.assertFalse(hs.human_ok(_run("a", 12)))  # the default is Ascension 10 only
        self.assertTrue(hs.human_ok(_run("a", 12), hp.parse_ascensions("11-20")))
        self.assertFalse(hs.human_ok(_run("a", 10), hp.parse_ascensions("11-20")))

    def test_iter_run_lines_reads_the_levels_asked(self):
        ids = _ids(False, 6, "b")
        runs = [_run(ids[0], 10), _run(ids[1], 12), _run(ids[2], 18, True), _run(ids[3], 9)]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.jsonl.gz"
            with gzip.open(p, "wt", encoding="utf-8") as fh:
                for r in runs:
                    fh.write(_line(r))
            for asc, want in (((10,), {ids[0]}), (hp.parse_ascensions("11-20"), {ids[1], ids[2]})):
                got = {json.loads(x)["run_hash"] for x in hp.iter_run_lines(p, frozenset(), Counter(), 0, asc)}
                self.assertEqual(want, got)
            default = {json.loads(x)["run_hash"] for x in hp.iter_run_lines(p, frozenset(), Counter())}
            self.assertEqual({ids[0]}, default)


def _shard(path: Path, runs: list[tuple[str, int, int]], per_run: int = 3) -> None:
    """A shard of card decisions (kind 0): `runs` = (run id, fifth, won); each run has `per_run` decisions of 4 options. Features are blank."""
    n_dec = len(runs) * per_run
    dec = np.zeros((n_dec, len(hp.COLS)), dtype=np.int32)
    d = 0
    for ix, (_rid, fifth, won) in enumerate(runs):
        for j in range(per_run):
            dec[d, hp.C["kind"]] = hp.KIND["card"]
            dec[d, hp.C["fifth"]] = fifth
            dec[d, hp.C["run"]] = ix
            dec[d, hp.C["won"]] = won
            dec[d, hp.C["n"]] = 4
            dec[d, hp.C["start"]] = 4 * d
            dec[d, hp.C["chosen"]] = j % 4
            d += 1
    rows = 4 * n_dec
    np.savez_compressed(path, dec=dec, dense=np.zeros((rows, hp.N_DENSE), np.float32), counts=np.zeros((rows, 4), np.uint8),
                        role=np.zeros(rows, np.int8), ident=np.zeros(rows, np.int32), run_ids=np.array([r[0] for r in runs], dtype="U32"),
                        cols=np.array(hp.COLS))


class Loader(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.work = Path(self.td.name)
        (self.work / "prior-a10" / "decisions").mkdir(parents=True)
        (self.work / "prior-high" / "decisions").mkdir(parents=True)
        # A10 pool: 8 runs over fifths 1 to 4, half won; the high set: 4 runs, all outcomes
        a10 = [(f"a{i}", 1 + i % 4, i % 2) for i in range(8)]
        high = [(f"h{i}", 1 + i % 4, 0) for i in range(4)]
        _shard(self.work / "prior-a10" / "decisions" / "shard-00000.npz", a10[:4])
        _shard(self.work / "prior-a10" / "decisions" / "shard-00001.npz", a10[4:])
        _shard(self.work / "prior-high" / "decisions" / "shard-00000.npz", high)
        self.a10, self.high = a10, high

    def tearDown(self):
        self.td.cleanup()

    def runs(self, name, fifths):
        ds = ss.load_parts(self.work, name, fifths, None, kinds=["card"])
        return sorted({ds.run_ids[i] for i in ds.dec[:, hp.C["run"]]}), ds

    def test_baseline_is_every_run_of_the_fifths(self):
        got, _ = self.runs("baseline", hp.TRAIN_FIFTHS)
        self.assertEqual(sorted(r for r, f, _w in self.a10 if f in hp.TRAIN_FIFTHS), got)

    def test_s_win_keeps_only_winning_runs(self):
        got, ds = self.runs("s-win", hp.TRAIN_FIFTHS)
        self.assertEqual(sorted(r for r, f, w in self.a10 if f in hp.TRAIN_FIFTHS and w), got)
        self.assertTrue((ds.dec[:, hp.C["won"]] == 1).all())

    def test_union_joins_the_parts_with_run_indices_kept_apart(self):
        got, ds = self.runs("s-win+s-high", hp.TRAIN_FIFTHS)
        want = sorted([r for r, f, w in self.a10 if f in hp.TRAIN_FIFTHS and w] + [r for r, f, _w in self.high if f in hp.TRAIN_FIFTHS])
        self.assertEqual(want, got)
        self.assertEqual(len(want), len(set(ds.dec[:, hp.C["run"]])))  # one index per run: the parts do not collide
        self.assertEqual(len(ds.dense), 4 * len(ds.dec))
        self.assertEqual(np.arange(0, len(ds.dense), 4).tolist(), ds.dec[:, hp.C["start"]].tolist())  # rows stay decision-contiguous

    def test_heldout_fifth_is_disjoint_from_every_fit(self):
        _, held = self.runs("baseline", [hp.VAL_FIFTH])
        for name in ss.FITS:
            train, _ = self.runs(name, hp.TRAIN_FIFTHS)
            self.assertEqual(len(held.run_ids) > 0, True)
            self.assertEqual(0, len(set(train) & {held.run_ids[i] for i in held.dec[:, hp.C["run"]]}), name)
            self.assertEqual({hp.VAL_FIFTH}, set(held.dec[:, hp.C["fifth"]].tolist()))
        ds = ss.heldout_sets(self.work, None)
        self.assertEqual({hp.VAL_FIFTH}, set(ds.dec[:, hp.C["fifth"]].tolist()))

    def test_runs_of_counts_only_the_runs_the_decisions_come_from(self):
        ds = ss.load_parts(self.work, "baseline", [hp.VAL_FIFTH], None, kinds=["card"])
        self.assertGreater(len(ds.run_ids), len(ss.runs_of(ds)))  # run_ids lists every run of the shards read, whatever its fifth
        self.assertEqual({r for r, f, _w in self.a10 if f == hp.VAL_FIFTH}, ss.runs_of(ds))

    def test_cells_under_the_minimum_are_not_shown(self):
        ds = ss.load_parts(self.work, "baseline", hp.TRAIN_FIFTHS, None, kinds=["card"])
        c = ss.set_cells(ds)
        self.assertEqual({"card|all", "card|act1", "card|act2", "card|act3"}, set(c))
        self.assertTrue(all(not v["shown"] for v in c.values()))  # a handful of runs: every cell is under the minimum
        self.assertEqual(sum(v["runs"] for k, v in c.items() if k == "card|all"), len(ss.runs_of(ds)))

    def test_no_overlap_guard(self):
        self.assertEqual(2, ss.no_overlap(["a", "b"], ["c", "d"]))
        with self.assertRaises(AssertionError):
            ss.no_overlap(["a", "b"], ["b", "c"])


def _scored_dataset(seed: int = 3):
    """Card (4 options), ancient (3) and rest (3 options with roles heal, smith, smith) decisions, random scores."""
    rng = np.random.default_rng(seed)
    decs, roles = [], []
    start = 0
    n_run = 6
    for kind, n_opt, count in (("card", 4, 40), ("ancient", 3, 25), ("rest", 3, 40)):
        for i in range(count):
            d = np.zeros(len(hp.COLS), dtype=np.int32)
            d[hp.C["kind"]] = hp.KIND[kind]
            d[hp.C["act"]] = i % 3
            d[hp.C["run"]] = i % n_run
            d[hp.C["n"]] = n_opt
            d[hp.C["start"]] = start
            if kind == "card":
                d[hp.C["chosen"]] = int(rng.integers(0, 4))
                d[hp.C["aux"]] = int(rng.integers(0, 2))
                d[hp.C["flags"]] = int(rng.integers(0, 2)) * hp.F_SPLIT
                roles += [0] * 4
            elif kind == "ancient":
                d[hp.C["chosen"]] = int(rng.integers(0, 3))
                roles += [0] * 3
            else:
                smith = i % 4 != 0
                d[hp.C["aux"]] = hp.ROLE_SMITH if smith else hp.ROLE_HEAL
                d[hp.C["chosen"]] = int(rng.integers(1, 3)) if smith else -1
                roles += [hp.ROLE_HEAL, hp.ROLE_SMITH, hp.ROLE_SMITH]
            decs.append(d)
            start += n_opt
    dec = np.stack(decs)
    ds = hp.Dataset(dec, np.zeros((start, hp.N_DENSE), np.float32), np.zeros((start, 4), np.uint8), np.array(roles, np.int8),
                    np.zeros(start, np.int32), [f"r{i}" for i in range(n_run)])
    ds._index(None)
    return ds, rng.normal(size=start).astype(np.float32)


class Measures(unittest.TestCase):
    def test_per_decision_means_equal_the_priors_own_metrics(self):
        ds, scores = _scored_dataset()
        valid = np.ones(len(ds.dec), dtype=bool)
        m = hp.metrics_from_scores(ds, scores, valid)
        pd = ss.per_decision(ds, scores, valid)
        pairs = {"card": "card", "ancient": "ancient", "rest_kind": "rest_kind", "rest_smith_target": "rest_smith_target"}
        for mine, theirs in pairs.items():
            self.assertAlmostEqual(m[theirs]["nll"], float(pd[mine]["nll"].mean()), places=9, msg=mine)
            self.assertAlmostEqual(m[theirs]["top1"], float(pd[mine]["hit"].mean()), places=9, msg=mine)
            self.assertEqual(m[theirs]["n"], len(pd[mine]["nll"]), mine)
        self.assertAlmostEqual(m["rest_smith_target_top3"]["top1"], float(pd["rest_smith_target"]["top3"].mean()), places=9)
        # the rank of the human's card: 1 when the prior's top option is the human's
        self.assertTrue((pd["card"]["rank"][pd["card"]["hit"] == 1] == 1).all())
        self.assertTrue((pd["card"]["rank"] >= 1).all() and (pd["card"]["rank"] <= 4).all())

    def test_summary_pools_the_three_softmaxes_and_masks_by_run(self):
        ds, scores = _scored_dataset()
        pd = ss.per_decision(ds, scores, np.ones(len(ds.dec), dtype=bool))
        allr = ss.summary(pd, None)
        n = sum(len(pd[k]["nll"]) for k in ss.POOLED)
        self.assertEqual(n, allr["decisions"])
        self.assertAlmostEqual(np.mean(np.concatenate([pd[k]["nll"] for k in ss.POOLED])), allr["log_loss"], places=9)
        mask = np.zeros(6, dtype=bool)
        mask[:2] = True
        some = ss.summary(pd, mask)
        self.assertLess(some["decisions"], allr["decisions"])
        keep = sum(int(mask[pd[k]["run"]].sum()) for k in ss.POOLED)
        self.assertEqual(keep, some["decisions"])
        self.assertIsNotNone(some["ranking"]["card_mean_rank"])

    def test_cells_count_decisions_and_runs(self):
        ds, scores = _scored_dataset()
        pd = ss.per_decision(ds, scores, np.ones(len(ds.dec), dtype=bool))
        c = ss.cells(pd, None)
        self.assertEqual(40, c["card|all"]["decisions"])
        self.assertEqual(6, c["card|all"]["runs"])
        self.assertEqual(sum(c[f"card|act{a}"]["decisions"] for a in (1, 2, 3)), 40)

    def test_paired_bootstrap_interval_contains_the_mean_and_is_seeded(self):
        rng = np.random.default_rng(1)
        run = np.repeat(np.arange(200), 5)
        diff = rng.normal(-0.05, 0.3, size=len(run))
        a = ss.paired_bootstrap(run, diff, 200)
        b = ss.paired_bootstrap(run, diff, 200)
        self.assertEqual(a, b)
        self.assertLess(a["lo95"], a["mean"])
        self.assertGreater(a["hi95"], a["mean"])
        self.assertLess(a["hi95"] - a["lo95"], 0.2)
        zero = ss.paired_bootstrap(run, np.zeros(len(run)), 200)
        self.assertEqual((0.0, 0.0, 0.0), (zero["mean"], zero["lo95"], zero["hi95"]))


class Selection(unittest.TestCase):
    def rows(self, base=(1.20, 1.10), swin=(1.15, 1.115), shigh=(1.30, 1.40), union=(1.14, 1.12)):
        return {"baseline": {"a": base[0], "b": base[1], "size": 3000}, "s-win": {"a": swin[0], "b": swin[1], "size": 330},
                "s-high": {"a": shigh[0], "b": shigh[1], "size": 20}, "s-win+s-high": {"a": union[0], "b": union[1], "size": 350}}

    def test_best_a_wins_when_it_passes_b(self):
        sel = ss.select_fit(self.rows())
        self.assertEqual("s-win+s-high", sel["selected"])
        self.assertEqual(["s-win", "s-win+s-high"], sel["eligible"])
        self.assertEqual(["s-high"], sel["refused_on_b"])

    def test_b_limit_refuses_a_better_a(self):
        sel = ss.select_fit(self.rows(union=(1.10, 1.14)))  # best (a), but 0.04 worse on all runs
        self.assertEqual("s-win", sel["selected"])
        self.assertIn("s-win+s-high", sel["refused_on_b"])

    def test_tie_goes_to_the_larger_set(self):
        sel = ss.select_fit(self.rows(swin=(1.1500, 1.11), union=(1.1510, 1.11)))
        self.assertEqual("s-win+s-high", sel["selected"])  # within the tie margin: the larger set
        sel = ss.select_fit(self.rows(swin=(1.1500, 1.11), union=(1.1600, 1.11)))
        self.assertEqual("s-win", sel["selected"])

    def test_nothing_beats_the_baseline(self):
        sel = ss.select_fit(self.rows(swin=(1.21, 1.10), union=(1.20, 1.10), shigh=(1.5, 1.5)))
        self.assertIsNone(sel["selected"])
        self.assertIn("no strong fit", sel["why"])

    def test_a_tie_with_the_baseline_is_not_a_win(self):
        sel = ss.select_fit(self.rows(swin=(1.1990, 1.10), union=(1.1995, 1.10), shigh=(1.5, 1.5)))
        self.assertIsNone(sel["selected"])


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class WarmStart(unittest.TestCase):
    def test_a_fitted_priors_weights_load_into_a_fresh_net(self):
        a, b = hp.make_net(), hp.make_net()
        with torch.no_grad():
            for p_ in a.parameters():
                p_.add_(torch.randn_like(p_) * 0.1)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "weights.npz"
            hp.export_net(a, path, {"kinds": ["card"]})
            hp.init_from_prior(b, path)
            for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
                self.assertEqual(ka, kb)
                self.assertTrue(torch.equal(va, vb), ka)

    def test_the_informational_arm_is_outside_the_selection(self):
        self.assertIn("s-win-ft", ss.ARMS)
        self.assertNotIn("s-win-ft", ss.FITS)
        self.assertEqual(ss.ARMS["s-win"], ss.ARMS["s-win-ft"])


class CandidateCopy(unittest.TestCase):
    OLD = "2ebfe6d7" + "f" * 56

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        root = Path(self.td.name)
        self.base, self.out = root / "gen13" / "candidate-stakes", root / "gen14" / "candidate-best-players"
        self.base.mkdir(parents=True)
        (self.base / "arms").mkdir()
        (self.base / "arms" / "x.json").write_text("{}", encoding="utf-8")
        (self.base / "promotion.json").write_text("{}", encoding="utf-8")
        for name, payload in (("run_value.npz", b"rv"), ("combat_policy.pt", b"cp"), ("fight_net.onnx", b"fn"), ("human_prior.npz", b"old prior")):
            (self.base / name).write_bytes(payload)
        macro = "vrun+pi_h:lambda=0.5:pi_h=2ebfe6d7"
        agent = {"generation": 13, "macro": macro, "written": "x", "lineage": {"skips": [12], "difference": "d (" + macro + ")", "decision": "old"},
                 "human_prior": {"file": "human_prior.npz", "sha256": self.OLD, "lambda": "0.5", "source": "pi_h-v0/weights.npz"},
                 "recipe": macro + "+stakes:a3-v1@h1e4b16", "built": {"why": "old", "tag": "stakes", "chooser": {"sha256": self.OLD}}, "note": "n"}
        train = {"candidate_generation": 13, "built": {"why": "old", "written": "x", "tag": "stakes",
                                                       "chooser": {"file": "human_prior.npz", "sha256": self.OLD, "source": "pi_h-v0/weights.npz",
                                                                   "macro": macro}},
                 "lineage": {"skips": [12], "difference": "d", "decision": "old"}}
        (self.base / "agent.json").write_text(json.dumps(agent), encoding="utf-8")
        (self.base / "train.json").write_text(json.dumps(train), encoding="utf-8")
        (self.base / "model.json").write_text(json.dumps({"kind": "combat_prior", "iteration": 13}), encoding="utf-8")
        self.new = root / "weights.npz"
        self.new.write_bytes(b"a different prior")
        self.info = {"set": "s-win", "source": "strong-sets/s-win/weights.npz", "why": "because", "counts": {"runs_train": 7}}

    def tearDown(self):
        self.td.cleanup()

    def test_only_the_prior_changes(self):
        agent = ss.build_candidate(self.base, self.out, self.new, self.info, now="2026-11-09T00:00:00Z")
        new = ss.file_sha256(self.new)
        self.assertEqual(new, agent["human_prior"]["sha256"])
        self.assertEqual(f"vrun+pi_h:lambda=0.5:pi_h={new[:8]}", agent["macro"])
        self.assertEqual(f"vrun+pi_h:lambda=0.5:pi_h={new[:8]}+stakes:a3-v1@h1e4b16", agent["recipe"])
        self.assertEqual(14, agent["generation"])
        self.assertEqual([12, 13], agent["lineage"]["skips"])
        self.assertEqual(b"a different prior", (self.out / "human_prior.npz").read_bytes())
        for name in ("run_value.npz", "combat_policy.pt", "fight_net.onnx"):
            self.assertEqual((self.base / name).read_bytes(), (self.out / name).read_bytes(), name)
        self.assertEqual({"agent.json", "train.json", "model.json", "human_prior.npz", "run_value.npz", "combat_policy.pt", "fight_net.onnx"},
                         {p.name for p in self.out.iterdir()})  # no arms, no promotion.json, no marker
        train = json.loads((self.out / "train.json").read_text(encoding="utf-8"))
        self.assertEqual(14, train["candidate_generation"])
        self.assertEqual(new, train["built"]["chooser"]["sha256"])
        self.assertEqual({"runs_train": 7}, train["strong_set"]["counts"])
        self.assertEqual(self.OLD, train["strong_set"]["replaces_sha256"])
        self.assertEqual(14, json.loads((self.out / "model.json").read_text(encoding="utf-8"))["iteration"])
        self.assertNotIn("2ebfe6d7", agent["macro"] + agent["recipe"] + json.dumps(agent["human_prior"]))

    def _work(self, selected):
        work = Path(self.td.name) / "work"
        (work / "fits" / "s-win").mkdir(parents=True, exist_ok=True)
        (work / "fits" / "s-win" / "weights.npz").write_bytes(b"a different prior")
        (work / "fits" / "s-win" / "fit.json").write_text(json.dumps({"runs": {"train": 7, "validation": 2}, "trained_on_decisions": {"card": 5},
                                                                    "best_epoch": 3, "init": "the generation 11 judgement's trunk"}), encoding="utf-8")
        (work / "heldout.json").write_text(json.dumps({"selection": {"selected": selected}}), encoding="utf-8")
        return work

    def test_candidate_command_builds_only_what_the_rule_picked(self):
        import types as _types  # noqa: PLC0415

        args = _types.SimpleNamespace(work=str(self._work(None)), base=str(self.base), out=str(self.out), set="")
        self.assertEqual(3, ss.cmd_candidate(args))  # the rule picked none: no folder
        self.assertFalse(self.out.exists())
        args = _types.SimpleNamespace(work=str(self._work("s-win")), base=str(self.base), out=str(self.out), set="")
        self.assertEqual(0, ss.cmd_candidate(args))
        agent = json.loads((self.out / "agent.json").read_text(encoding="utf-8"))
        self.assertIn("selected by the held-out rule", agent["built"]["why"])

    def test_forcing_an_arm_is_recorded(self):
        import types as _types  # noqa: PLC0415

        args = _types.SimpleNamespace(work=str(self._work(None)), base=str(self.base), out=str(self.out), set="s-win")
        self.assertEqual(0, ss.cmd_candidate(args))
        agent = json.loads((self.out / "agent.json").read_text(encoding="utf-8"))
        self.assertIn("NOT the rule's pick", agent["built"]["why"])

    def test_refuses_the_same_prior_and_an_existing_folder(self):
        same = self.base.parent / "same.npz"
        same.write_bytes(b"x")
        agent = json.loads((self.base / "agent.json").read_text(encoding="utf-8"))
        agent["human_prior"]["sha256"] = ss.file_sha256(same)
        (self.base / "agent.json").write_text(json.dumps(agent), encoding="utf-8")
        with self.assertRaises(ValueError):
            ss.build_candidate(self.base, self.out, same, self.info)
        self.out.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            ss.build_candidate(self.base, self.out, self.new, self.info)


if __name__ == "__main__":
    unittest.main()
