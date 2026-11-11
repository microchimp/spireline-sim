"""The run student and its gate (rl/run_student.py; lane 3 of the learning-system build plan, D56 and D57 part 3).

Pinned: the teacher's soft targets and their temperature (the target is the softmax of Q plus lambda log pi_h over the played options at
the noise sd of K draws; an option off the shortlist has no target; lambda is applied only where the player blends the prior); the shard
loader and each thing it refuses; the split of runs by a hash of the seed (a fifth held out, no run in two parts); the human term never reads
the exam fifth (a planted exam run is refused, bucket 0 cannot be asked for); the net before any step is the prior, logit for logit; a fit
learns a planted teacher on synthetic shards and the KL term holds it to the prior; leg 1's arithmetic, leg 2's arithmetic and the
verdict (won runs within +0.002, all runs within +0.02, a kind with no human fit on leg 1 alone); lambda's regression recovers a planted
value and its fall-backs; the candidate builder and the recipe it writes.

No game, no emulator and no real fit on real data: synthetic shards (their rows random, the teacher's values a planted function of the
rows), the measurement's own records with synthetic rows where the measurement is on this machine (skipped where it is not).
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import human_prior as hp  # noqa: E402
import human_prior_infer as hpi  # noqa: E402
import human_starts as hs  # noqa: E402
import macro_recipe as mr  # noqa: E402
import run_student as rs  # noqa: E402
from run_train import models  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

MEASURE = Path("D:/spireline/referee/backlog/run-self-play/measure/decisions.d")


# ------------------------------------------------------------------------------------------------------------------ synthetic data


def random_weights(seed: int = 3, scale: float = 0.05) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shapes = {"embed.deck.weight": (64, 1154), "embed.relics.weight": (32, 296), "embed.potions.weight": (8, 63), "hidden1.weight": (256, 454),
              "hidden1.bias": (256,), "hidden2.weight": (256, 256), "hidden2.bias": (256,), "score.weight": (1, 256), "score.bias": (1,)}
    return {k: (rng.standard_normal(s) * scale).astype(np.float32) for k, s in shapes.items()}


def write_weights(path: Path, weights: dict[str, np.ndarray]) -> Path:
    np.savez(path, meta=np.array(json.dumps({"variant": "test"})), **weights)
    return path


def student_file(path: Path, weights: dict[str, np.ndarray], kinds=("card", "route", "shop", "reward"), prior_sha: str = "0" * 64) -> Path:
    hpi.write_student(path, weights, kinds=kinds, prior_sha256=prior_sha, fit={"test": True})
    return path


KIND_SHAPE = {"card": ("card_reward", 4, "card"), "ancient": ("neow", 3, "ancient"), "route": ("map", 3, None), "shop": ("shop", 6, None),
              "reward": ("rewards", 2, None), "rest": ("rest", 5, "rest"), "event": ("event", 2, None), "select": ("card_select", 4, None)}


class Synth:
    """A planted teacher: rows are random; the teacher's value of an option is a fixed linear function of its dense input (standardised),
    plus noise per draw; V is a noisier copy of the same signal."""

    def __init__(self, prior_weights: dict[str, np.ndarray], seed: int = 7, signal: float = 1.0, v_share: float = 0.6, noise: float = 2.0):
        self.rng = np.random.default_rng(seed)
        self.prior = hpi.PriorScorer(prior_weights)
        self.u = self.rng.standard_normal(rs.DENSE_WIDTH) / math.sqrt(rs.DENSE_WIDTH)
        self.signal, self.v_share, self.noise = signal, v_share, noise
        probe = self.rng.normal(0, 0.4, (4000, rs.ROWS_WIDTH)).astype(np.float32)
        lin = models.value_inputs(probe)[0] @ self.u
        self.mu, self.sd = float(lin.mean()), float(lin.std())

    def rows(self, n: int) -> np.ndarray:
        return self.rng.normal(0, 0.4, (n, rs.ROWS_WIDTH)).astype(np.float32)

    def signal_of(self, rows: np.ndarray) -> np.ndarray:
        return self.signal * (models.value_inputs(rows)[0] @ self.u - self.mu) / self.sd

    def record(self, seed: str, ordinal: int, kind: str, act: int = 0, k: int = 4, lam: float = 0.5, n_options: int | None = None,
               keep_rows: bool = True) -> tuple[dict, np.ndarray]:
        dtype, n_default, screen = KIND_SHAPE[kind]
        n = n_options or n_default
        rows = self.rows(n)
        sig = self.signal_of(rows)
        played = list(range(n)) if n <= 4 else sorted({0} | set(int(x) for x in self.rng.choice(np.arange(1, n), 3, replace=False)))
        g = {str(o): [float(sig[o] + self.rng.normal(0, self.noise)) for _ in range(k)] for o in played}
        q = {o: round(float(np.mean(v)), 4) for o, v in g.items()}
        v = (self.v_share * sig + self.rng.normal(0, 0.8, n)).tolist()
        dense, counts = models.value_inputs(rows)
        logp = None
        if screen is not None:
            logits = self.prior.logits(dense, counts)
            logp = hpi.log_softmax(logits).tolist()
        player = int(max(q, key=lambda o: q[o]))
        rec = {"v": 1, "seed": seed, "ordinal": ordinal, "type": dtype, "kind": kind, "act": act, "floor": 3 + 20 * act, "hp": 50, "max_hp": 80,
               "n_options": n, "played": played, "player": player, "k": k, "screen": screen, "lam": lam, "v_values": v, "logp": logp, "g": g,
               "ends": {"door": 4}, "options": [f"{kind}.x"] * n, "teacher": player, "teacher_raw": player,
               "q": {str(o): x for o, x in q.items()}, "gain": 0.0, "gain_se": None, "cpu": 1.0}
        return rec, rows


def write_shard(root: Path, name: str, records: list[dict], rows_by_decision: list[np.ndarray | None], manifest: dict | None = None,
                options: list[list[int]] | None = None) -> Path:
    """A shard folder: decisions.jsonl with `row0`, rows.npz, manifest.json. `rows_by_decision[i]` None: the decision has no rows; `options[i]`
    names the options its rows are of (default: 0, 1, ...)."""
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    rows, dec, opt = [], [], []
    at = 0
    for i, (r, rr) in enumerate(zip(records, rows_by_decision, strict=True)):
        r["row0"] = at if rr is not None else None
        if rr is not None:
            rows.append(rr)
            dec += [i] * len(rr)
            opt += list(range(len(rr))) if options is None else list(options[i])
            at += len(rr)
    (d / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    np.savez_compressed(d / "rows.npz", rows=np.concatenate(rows).astype(np.float32) if rows else np.zeros((0, rs.ROWS_WIDTH), np.float32),
                        decision=np.array(dec, dtype=np.int32), option=np.array(opt, dtype=np.int16))
    man = {"version": 1, "agent": "stub", "judge": "stub", "k": 4, "max_options": 4, "counts": {"decisions": len(records), "rows": at}}
    (d / "manifest.json").write_text(json.dumps(man if manifest is None else manifest), encoding="utf-8")
    return d


def synth_shards(root: Path, synth: Synth, runs: int = 40, kinds=("card", "route", "shop", "reward"), per_kind: int = 2, name: str = "job0") -> Path:
    records, rows = [], []
    for i in range(runs):
        seed = f"S{i:08d}"
        n = 0
        for kind in kinds:
            for _ in range(per_kind):
                r, rr = synth.record(seed, n, kind, act=int(n % 2))
                records.append(r)
                rows.append(rr)
                n += 1
    return write_shard(root, name, records, rows)


def seed_in(part: str, start: int = 0) -> str:
    i = start
    while True:
        s = f"T{i:08d}"
        if rs.split_of(s) == part:
            return s
        i += 1


def run_ids_by_bucket(per: int = 25) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {b: [] for b in range(5)}
    i = 0
    while min(len(v) for v in out.values()) < per:
        rid = f"h{i:014x}"
        b = hs.exam_bucket(rid)
        if len(out[b]) < per:
            out[b].append(rid)
        i += 1
    return out


def human_dataset(runs_by_fifth: dict[int, list[str]], seed: int = 5, kinds=("card", "ancient", "rest", "shop", "route", "event"),
                  per_run: int = 2, won_every: int = 2) -> "hp.Dataset":
    """A v1 dataset: `per_run` decisions of each kind for each run; random rows, random choices (a shop: row 0 leave, row 1 buy)."""
    rng = np.random.default_rng(seed)
    run_ids = [r for f in sorted(runs_by_fifth) for r in runs_by_fifth[f]]
    fifth_of = {r: f for f in runs_by_fifth for r in runs_by_fifth[f]}
    dec, dense, counts, role = [], [], [], []
    n_opts = {"card": 4, "ancient": 3, "rest": 4, "shop": 2, "route": 3, "event": 2}
    start = 0
    for ri, rid in enumerate(run_ids):
        for kind in kinds:
            for _ in range(per_run):
                n = n_opts[kind]
                row = np.zeros(len(hp.COLS), dtype=np.int32)
                row[hp.C["kind"]] = hp.KIND[kind]
                row[hp.C["fifth"]] = fifth_of[rid]
                row[hp.C["act"]] = int(rng.integers(0, 3))
                row[hp.C["run"]] = ri
                row[hp.C["won"]] = int(ri % won_every == 0)
                row[hp.C["start"]], row[hp.C["n"]] = start, n
                role_k = np.full(n, -1, dtype=np.int8)
                if kind == "rest":
                    role_k = np.array([hp.ROLE_HEAL, hp.ROLE_SMITH, hp.ROLE_SMITH, hp.ROLE_OTHER], dtype=np.int8)
                    row[hp.C["aux"]] = hp.ROLE_SMITH
                    row[hp.C["chosen"]] = int(rng.integers(1, 3))
                elif kind == "shop":
                    row[hp.C["chosen"]] = int(rng.integers(0, 2))
                else:
                    row[hp.C["chosen"]] = int(rng.integers(0, n))
                dec.append(row)
                dense.append(rng.normal(0, 0.3, (n, hp.N_DENSE_V1)).astype(np.float32))
                c = np.zeros((n, rs.COUNTS_WIDTH), dtype=np.uint8)
                c[:, :5] = rng.integers(0, 3, (n, 5))
                counts.append(c)
                role.append(role_k)
                start += n
    ds = hp.Dataset(np.stack(dec), np.concatenate(dense), np.concatenate(counts), np.concatenate(role), np.zeros(start, dtype=np.int32), run_ids)
    ds._index(None)  # noqa: SLF001
    return ds


# ------------------------------------------------------------------------------------------------------------------ the targets


class TargetTests(unittest.TestCase):
    def test_the_target_is_the_softmax_of_q_plus_lambda_log_prior_at_the_temperature(self):
        q = [10.0, 9.0, 12.0]
        logp = [math.log(0.5), math.log(0.3), math.log(0.2)]
        t = rs.teacher_target(q, logp, 0.5, 1.5)
        score = np.array([10 + 0.5 * logp[0], 9 + 0.5 * logp[1], 12 + 0.5 * logp[2]]) / 1.5
        want = np.exp(score - score.max())
        np.testing.assert_allclose(t, want / want.sum(), atol=1e-12)
        self.assertAlmostEqual(1.0, float(t.sum()))
        np.testing.assert_allclose(rs.teacher_target(q, None, 0.0, 1.5), rs.teacher_target(q, logp, 0.0, 1.5))  # lambda 0 never reads the prior

    def test_the_temperature_flattens_or_sharpens_the_target(self):
        q = [3.0, 1.0, 0.0]
        sharp, mid, flat = (rs.teacher_target(q, None, 0.0, tau) for tau in (0.1, 1.0, 100.0))
        self.assertGreater(sharp[0], 0.999)
        self.assertGreater(mid[0], mid[1])
        np.testing.assert_allclose(flat, [1 / 3] * 3, atol=0.03)
        self.assertGreater(float(-(sharp * np.log(sharp + 1e-300)).sum()), -1e-9)
        with self.assertRaises(ValueError):
            rs.teacher_target(q, None, 0.0, 0.0)

    def test_tau_is_the_noise_sd_of_one_draw_over_the_root_of_k(self):
        # two options, draws [0, 2, 0, 2] and [0, 0, 0, 0]: the centred values are +-(0, 1, 0, 1), a sample variance of 1/3 per option
        g = [[[0.0, 2.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0]]] * 25
        table = rs.tau_table({("card", 0): g})
        self.assertAlmostEqual(math.sqrt(1 / 3), table["cells"]["card/a1"], places=3)
        self.assertAlmostEqual(table["cells"]["card/a1"], table["kinds"]["card"])
        self.assertEqual(table["kinds"]["card"], table["all"])

    def test_a_small_cell_takes_its_kinds_noise_then_everyones(self):
        big = [[[0.0, 2.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0]]] * 30
        small = [[[0.0, 4.0, 0.0, 4.0], [0.0, 0.0, 0.0, 0.0]]] * 3
        table = rs.tau_table({("card", 0): big, ("card", 1): small})
        self.assertNotIn("card/a2", table["cells"])
        self.assertAlmostEqual(table["kinds"]["card"], rs.noise_sd_for(table, "card", 1))
        self.assertAlmostEqual(table["all"], rs.noise_sd_for(table, "route", 0))  # a kind never seen: everyone's

    def test_the_set_gives_each_decision_tau_at_its_own_k_and_no_target_off_the_shortlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = random_weights()
            synth = Synth(weights)
            recs, rows = [], []
            for i in range(30):
                r, rr = synth.record(f"S{i:08d}", 0, "rest", n_options=8, k=4)  # 8 options, 4 played
                recs.append(r)
                rows.append(rr)
            r2, rr2 = synth.record("S99999999", 0, "rest", n_options=8, k=2)
            recs.append(r2)
            rows.append(rr2)
            ts = rs.load_shards(write_shard(Path(tmp), "a", recs, rows), prior=hpi.PriorScorer(weights))
            self.assertEqual(31, len(ts))
            self.assertGreater(float(ts.tau[-1]), float(ts.tau[0]))  # fewer draws, a noisier mean, a flatter target
            self.assertAlmostEqual(float(ts.tau[-1]) / float(ts.tau[0]), math.sqrt(4 / 2), places=4)
            t = ts.targets()
            for i in (0, 30):
                a, b = int(ts.start[i]), int(ts.start[i] + ts.count[i])
                played = ~np.isnan(ts.q[a:b])
                self.assertEqual(4, int(played.sum()))
                np.testing.assert_array_equal(t[a:b][~played], 0.0)  # an option off the shortlist carries no target
                self.assertAlmostEqual(1.0, float(t[a:b].sum()), places=5)

    def test_lambda_is_applied_only_where_the_player_blends_the_prior(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = random_weights()
            synth = Synth(weights)
            recs, rows = [], []
            for kind in ("card", "route"):  # both records carry lam 0.5; only the card screen is blended (screen "card" against None)
                r, rr = synth.record("S00000001", len(recs), kind, lam=0.5)
                recs.append(r)
                rows.append(rr)
            ts = rs.load_shards(write_shard(Path(tmp), "a", recs, rows), prior=hpi.PriorScorer(weights))
            self.assertEqual([0.5, 0.0], [float(x) for x in ts.lam])
            t = ts.targets()
            for i in (0, 1):
                a, b = int(ts.start[i]), int(ts.start[i] + ts.count[i])
                lp = hpi.log_softmax(ts.zh[a:b])
                want = rs.teacher_target(ts.q[a:b], lp, float(ts.lam[i]), float(ts.tau[i]))
                np.testing.assert_allclose(t[a:b], want, atol=1e-6)
            a, b = int(ts.start[1]), int(ts.start[1] + ts.count[1])
            np.testing.assert_allclose(t[a:b], rs.teacher_target(ts.q[a:b], None, 0.0, float(ts.tau[1])), atol=1e-6)
            # a lambda override per (kind, act) moves only the cell it names
            t2 = ts.targets({("route", 0): 1.0})
            self.assertFalse(np.allclose(t2[a:b], t[a:b]))
            np.testing.assert_allclose(t2[: int(ts.count[0])], t[: int(ts.count[0])], atol=1e-6)


# ------------------------------------------------------------------------------------------------------------------ the loader


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.weights = random_weights()
        self.prior = hpi.PriorScorer(self.weights)
        self.synth = Synth(self.weights)

    def tearDown(self):
        self.tmp.cleanup()

    def shard(self, n: int = 6, name: str = "a", kinds=("card", "route", "shop")):
        recs, rows = [], []
        for i in range(n):
            r, rr = self.synth.record(f"S{i:08d}", i, kinds[i % len(kinds)])
            recs.append(r)
            rows.append(rr)
        return recs, rows

    def test_a_shard_loads_with_its_counts_and_the_prior_check(self):
        recs, rows = self.shard(9)
        d = write_shard(self.root, "a", recs, rows)
        ts = rs.load_shards(d, prior=self.prior)
        self.assertEqual((9, 9, 0.0 <= ts.stats["logp_check"]["max_abs_difference"] < 1e-4), (len(ts), len(ts.seeds), True))
        self.assertEqual(3, ts.stats["logp_check"]["decisions_checked"])  # the three card screens carry the player's log-probabilities
        self.assertEqual(int(ts.count.sum()), len(ts.q))
        stats = rs.public_stats(ts)
        self.assertEqual({"card": 3, "route": 3, "shop": 3}, stats["kept_by_kind"])
        self.assertNotIn("S00000000", json.dumps(stats))  # the report names no run

    def test_shards_are_found_under_a_root_and_joined(self):
        for name in ("job1", "job0"):
            recs, rows = self.shard(3, kinds=("card",))
            write_shard(self.root / "gen13", name, recs, rows)
        ts = rs.load_shards(self.root)
        self.assertEqual((6, ["job0", "job1"]), (len(ts), [m["shard"] for m in ts.manifests]))

    def test_decisions_the_student_cannot_learn_from_are_dropped_and_counted(self):
        recs, rows = self.shard(4, kinds=("card",))
        refused, rr = self.synth.record("S10000000", 0, "card")
        refused.pop("q"), refused.update(refused="no_return")
        bundle, br = self.synth.record("S10000001", 0, "card")
        bundle["kind"] = "bundle"
        one, orr = self.synth.record("S10000002", 0, "card")
        one["q"] = {"0": 1.0}
        partial, pr = self.synth.record("S10000003", 0, "route", n_options=3)
        inf, ir = self.synth.record("S10000004", 0, "card")
        inf["lam"] = float("inf")
        no_rows, _ = self.synth.record("S10000005", 0, "card")
        recs += [refused, bundle, one, partial, inf, no_rows]
        rows += [rr, br, orr, pr[:2], ir, None]  # the partial decision lost an option's row, the last has none
        ts = rs.load_shards(write_shard(self.root, "a", recs, rows), prior=self.prior)
        self.assertEqual(4, len(ts))
        self.assertEqual({"refused": 1, "kind_not_taken": 1, "fewer_than_two_priced": 1, "incomplete_rows": 2, "lambda_infinite": 1},
                         {k: v for k, v in ts.stats["dropped"].items() if v})

    def test_a_cut_last_line_is_dropped_and_any_other_bad_line_refuses(self):
        recs, rows = self.shard(3)
        d = write_shard(self.root, "a", recs, rows)
        with open(d / "decisions.jsonl", "a", encoding="utf-8") as fh:
            fh.write('{"v": 1, "seed": "S0000')
        man = json.loads((d / "manifest.json").read_text())
        man["counts"]["decisions"] = 4  # the cut line counts: the writer had started it
        (d / "manifest.json").write_text(json.dumps(man))
        ts = rs.load_shards(d)
        self.assertEqual((3, 1), (len(ts), ts.stats["dropped"]["truncated_tail"]))
        lines = (d / "decisions.jsonl").read_text().splitlines()
        lines.insert(1, "not json")
        (d / "decisions.jsonl").write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(rs.ShardError, "not JSON"):
            rs.load_shards(d)

    def refuse(self, why: str, edit, *, pattern: str | None = None):
        recs, rows = self.shard(4, kinds=("card",))
        d = write_shard(self.root, why.replace(" ", "-"), recs, rows)
        edit(d, recs)
        with self.assertRaisesRegex(rs.ShardError, pattern or why):
            rs.load_shards(d)

    def rewrite_rows(self, d: Path, **changes):
        with np.load(d / "rows.npz") as z:
            data = {k: z[k] for k in z.files}
        data.update(changes)
        np.savez_compressed(d / "rows.npz", **data)

    def test_the_loader_refuses_what_it_cannot_trust(self):
        self.refuse("no rows.npz", lambda d, r: (d / "rows.npz").unlink())
        self.refuse("no manifest.json", lambda d, r: (d / "manifest.json").unlink())
        self.refuse("manifest not an object", lambda d, r: (d / "manifest.json").write_text("[1]"), pattern="not an object")
        self.refuse("record version", lambda d, r: (d / "decisions.jsonl").write_text(
            "\n".join(json.dumps({**x, "v": 2}) for x in r) + "\n"), pattern="not a version 1")
        self.refuse("rows of another width", lambda d, r: self.rewrite_rows(d, rows=np.zeros((16, 1862), np.float32)), pattern="not \\(n, 1863\\)")
        self.refuse("rows that are not finite", lambda d, r: self.rewrite_rows(
            d, rows=np.where(np.arange(1863)[None, :] == 3, np.nan, np.zeros((16, 1863), np.float32)).astype(np.float32)), pattern="not finite")
        self.refuse("a row of a decision that is not there", lambda d, r: self.rewrite_rows(d, decision=np.full(16, 99, np.int32)),
                    pattern="outside the 4 records")
        self.refuse("an option past n_options", lambda d, r: self.rewrite_rows(d, option=np.full(16, 9, np.int16)), pattern="past its decision")
        self.refuse("an option with two rows", lambda d, r: self.rewrite_rows(d, option=np.zeros(16, np.int16)), pattern="two rows")
        self.refuse("rows and decision differ", lambda d, r: self.rewrite_rows(d, decision=np.zeros(3, np.int32)), pattern="differ in length")
        self.refuse("row0 that points elsewhere", lambda d, r: (d / "decisions.jsonl").write_text(
            "\n".join(json.dumps({**x, "row0": x["row0"] + 1}) for x in r) + "\n"), pattern="row0")
        self.refuse("manifest counts", lambda d, r: (d / "manifest.json").write_text(json.dumps({"counts": {"decisions": 99}})),
                    pattern="manifest counts 99 decisions")
        self.refuse("a played option out of range", lambda d, r: (d / "decisions.jsonl").write_text(
            "\n".join(json.dumps({**x, "played": [0, 1, 7]}) for x in r) + "\n"), pattern="outside its 4 options")
        self.refuse("q that is not finite", lambda d, r: (d / "decisions.jsonl").write_text(
            "\n".join(json.dumps({**x, "q": {"0": float("nan"), "1": 1.0}}) for x in r) + "\n"), pattern="not finite")
        with self.assertRaisesRegex(rs.ShardError, "no decision shard"):
            rs.load_shards(self.root / "empty")

    def test_shards_played_with_another_prior_are_refused(self):
        recs, rows = self.shard(4, kinds=("card",))
        d = write_shard(self.root, "a", recs, rows)
        other = hpi.PriorScorer(random_weights(seed=99))
        with self.assertRaisesRegex(rs.ShardError, "another human prior"):
            rs.load_shards(d, prior=other)

    @unittest.skipUnless(MEASURE.is_dir(), "the 50-run measurement is not on this machine")
    def test_the_measurements_own_records_load_with_synthetic_rows(self):
        recs, rows, options, rng = [], [], [], np.random.default_rng(11)
        for f in sorted(MEASURE.glob("*.jsonl"))[:12]:
            for line in f.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                have = [i for i, v in enumerate(r["v_values"]) if v is not None]  # an option the judgement cannot score has no row
                recs.append(r)
                rows.append(rng.normal(0, 0.4, (len(have), rs.ROWS_WIDTH)).astype(np.float32))
                options.append(have)
        d = write_shard(self.root, "measure", recs, rows, options=options)
        ts = rs.load_shards(d)
        st = ts.stats
        self.assertEqual(len(recs), sum(st["kept"].values()) + sum(st["dropped"].values()))
        self.assertLessEqual(set(k for k, v in st["dropped"].items() if v), {"kind_not_taken", "incomplete_rows"})
        self.assertTrue(set(st["kept"]) <= set(rs.KINDS))
        ts.attach_prior(hpi.PriorScorer(random_weights()), tol=1e9)
        t = ts.targets()
        for i in range(len(ts)):
            a, b = int(ts.start[i]), int(ts.start[i] + ts.count[i])
            played = ~np.isnan(ts.q[a:b])
            self.assertAlmostEqual(1.0, float(t[a:b].sum()), places=4)
            self.assertTrue((t[a:b][~played] == 0).all())
            self.assertGreaterEqual(int(played.sum()), 2)
        self.assertTrue(0.25 <= float(ts.tau.min()) and float(ts.tau.max()) < 6.0, (float(ts.tau.min()), float(ts.tau.max())))


class SplitTests(unittest.TestCase):
    def test_runs_split_by_a_hash_of_the_seed_into_a_fifth_a_tenth_and_the_rest(self):
        parts = [rs.split_of(f"S{i:08d}") for i in range(10000)]
        share = {p: parts.count(p) / len(parts) for p in ("test", "val", "train")}
        self.assertAlmostEqual(0.2, share["test"], delta=0.02)
        self.assertAlmostEqual(0.1, share["val"], delta=0.02)
        self.assertAlmostEqual(0.7, share["train"], delta=0.03)
        self.assertEqual(parts, [rs.split_of(f"S{i:08d}") for i in range(10000)])  # a fixed rule

    def test_no_run_is_in_two_parts_and_every_decision_is_in_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            synth = Synth(random_weights())
            ts = rs.load_shards(synth_shards(Path(tmp), synth, runs=50, kinds=("card", "route")))
            idx = ts.split_idx()
            self.assertEqual(len(ts), sum(len(v) for v in idx.values()))
            runs = {k: set(ts.run[v].tolist()) for k, v in idx.items()}
            self.assertFalse(runs["train"] & runs["test"] or runs["train"] & runs["val"] or runs["val"] & runs["test"])
            self.assertTrue(all(rs.split_of(ts.seeds[r]) == "test" for r in runs["test"]))


# ------------------------------------------------------------------------------------------------------------------ the human term


class HumanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids = run_ids_by_bucket(25)

    def test_the_exam_fifth_is_never_read(self):
        clean = human_dataset({1: self.ids[1][:5], 4: self.ids[4][:5]})
        self.assertEqual(10, rs.check_no_exam(clean, frozenset()))
        planted = human_dataset({1: self.ids[1][:5], 4: self.ids[0][:1]})  # a run of bucket 0 written into the shards
        with self.assertRaisesRegex(AssertionError, "exam fifth"):
            rs.check_no_exam(planted, frozenset())
        listed = human_dataset({1: self.ids[1][:5]})
        with self.assertRaisesRegex(AssertionError, "exam fifth"):
            rs.check_no_exam(listed, frozenset({self.ids[1][0]}))  # a run the ids file names, whatever its bucket
        bucket0 = human_dataset({1: self.ids[1][:3]})
        bucket0.dec[0, hp.C["fifth"]] = hs.EXAM_BUCKET
        with self.assertRaisesRegex(AssertionError, "bucket 0"):
            rs.check_no_exam(bucket0, frozenset())

    def test_the_exam_bucket_cannot_be_asked_for_and_a_planted_shard_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ((0,), (1, 2, 0), (5,)):
                with self.assertRaisesRegex(AssertionError, "only 1 to 4"):
                    rs.load_human(tmp, bad, exam_ids=frozenset())
            ds = human_dataset({1: self.ids[1][:4], 2: self.ids[0][:1]})  # fifth 2's run is really bucket 0: a planted exam run
            ds.dec[:, hp.C["fifth"]] = np.where(ds.dec[:, hp.C["run"]] == 4, 2, 1)
            np.savez(Path(tmp) / "shard-00000.npz", dec=ds.dec, dense=ds.dense, counts=ds.counts, role=ds.role, ident=ds.ident,
                     run_ids=np.array(ds.run_ids))
            with self.assertRaisesRegex(AssertionError, "exam fifth"):
                rs.load_human(tmp, (1, 2, 3), exam_ids=frozenset())
            clean = human_dataset({1: self.ids[1][:4], 3: self.ids[3][:3]})
            clean_dir = Path(tmp) / "clean"
            clean_dir.mkdir()
            np.savez(clean_dir / "shard-00000.npz", dec=clean.dec, dense=clean.dense, counts=clean.counts, role=clean.role, ident=clean.ident,
                     run_ids=np.array(clean.run_ids))
            got = rs.load_human(clean_dir, (1, 2, 3), exam_ids=frozenset())
            self.assertEqual(len(clean.dec), len(got.dec))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_the_human_loss_given_the_logits_is_the_priors_own_loss(self):
        ds = human_dataset({1: self.ids[1][:12], 2: self.ids[2][:12]})
        net = rs.make_net(random_weights(), 0.0).eval()
        gpu = hp.Gpu(ds, None, torch.device("cpu"), 1.0)
        for kind in ("card", "ancient", "rest", "shop", "route"):
            didx = torch.from_numpy(ds.kind_idx[kind][:16])
            z, m, valid = gpu.logits(net, didx, False)
            self.assertAlmostEqual(float(gpu.loss(net, kind, didx).detach()), float(rs.human_loss_from_z(gpu, kind, didx, z, m, valid).detach()),
                                   places=5, msg=kind)
        # an event page is a softmax over its options, as a route
        didx = torch.from_numpy(ds.kind_idx["event"][:16])
        z, m, valid = gpu.logits(net, didx, False)
        want = float((torch.logsumexp(z, 1) - z.gather(1, gpu.chosen[didx][:, None])[:, 0]).mean().detach())
        self.assertAlmostEqual(want, float(rs.human_loss_from_z(gpu, "event", didx, z, m, valid).detach()), places=5)


# ------------------------------------------------------------------------------------------------------------------ the net


@unittest.skipIf(torch is None, "torch is not installed")
class NetTests(unittest.TestCase):
    def test_the_student_before_any_step_is_the_prior_logit_for_logit(self):
        weights = random_weights()
        rows = np.random.default_rng(1).normal(0, 0.4, (40, rs.ROWS_WIDTH)).astype(np.float32)
        dense, counts = models.value_inputs(rows)
        net = rs.make_net(weights, 0.0).eval()
        with torch.no_grad():
            got = net(torch.from_numpy(dense), torch.from_numpy(counts)).numpy()
        np.testing.assert_allclose(got, hpi.PriorScorer(weights).logits(dense, counts), atol=2e-5)
        self.assertEqual(set(weights), set(rs.net_params(net)))

    def test_a_student_file_names_its_kinds_and_a_plain_prior_is_not_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = random_weights()
            plain = write_weights(Path(tmp) / "pi_h.npz", weights)
            with self.assertRaisesRegex(ValueError, "not a run student"):
                hpi.StudentScorer.load(plain)
            with self.assertRaisesRegex(ValueError, "not a run student"):
                hpi.student_kinds_of(plain)
            st = student_file(Path(tmp) / "s.npz", weights, kinds=("route", "shop"), prior_sha=hpi.file_sha256(plain))
            s = hpi.StudentScorer.load(st)
            self.assertEqual((("route", "shop"), hpi.file_sha256(plain), True, False), (s.kinds, s.prior_sha256, s.has("shop"), s.has("card")))
            self.assertEqual(("route", "shop"), hpi.student_kinds_of(st))
            self.assertEqual(s.sha256, hpi.load_student(st).sha256)
            rows = np.random.default_rng(2).normal(0, 0.4, (5, rs.ROWS_WIDTH)).astype(np.float32)
            np.testing.assert_allclose(s.row_logits(rows), hpi.PriorScorer(weights).row_logits(rows), atol=1e-6)  # it scores as a prior does
            with self.assertRaisesRegex(ValueError, "at least one kind"):
                hpi.write_student(Path(tmp) / "x.npz", weights, kinds=(), prior_sha256="")


# ------------------------------------------------------------------------------------------------------------------ the fit and the legs


@unittest.skipIf(torch is None, "torch is not installed")
class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.weights = random_weights()
        cls.prior_path = write_weights(cls.root / "pi_h.npz", cls.weights)
        cls.prior = hpi.PriorScorer.load(cls.prior_path)
        cls.synth = Synth(cls.weights, seed=21)
        cls.shards = synth_shards(cls.root, cls.synth, runs=260, kinds=("card", "route", "shop", "reward"), per_kind=2)
        cls.ts = rs.load_shards(cls.shards, prior=cls.prior)
        cls.cfg = rs.FitConfig(epochs=14, patience=14, batch_teacher=64, lr=2e-3, beta=0.0, kl=0.0, dropout=0.0)
        cls.res = rs.fit(cls.ts, cls.weights, cfg=cls.cfg, device="cpu", log=lambda *_: None)
        cls.student_path = student_file(cls.root / "student.npz", cls.res["params"], kinds=("card", "route", "shop", "reward"),
                                        prior_sha=hpi.file_sha256(cls.prior_path))
        cls.student = hpi.StudentScorer.load(cls.student_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_fit_learns_a_planted_teacher_and_stops_on_the_validation_runs(self):
        h = self.res["history"]
        self.assertLess(h[-1]["train_ce"], h[0]["train_ce"] - 0.02)
        self.assertLessEqual(self.res["best_epoch"], len(h))
        self.assertGreater(self.res["val_decisions"], 0)
        self.assertEqual(set(self.weights), set(self.res["params"]))

    def test_leg_1_the_student_beats_the_prior_on_held_out_runs_with_an_interval_above_zero(self):
        test = self.ts.split_idx()["test"]
        with mock.patch.object(rs, "BOOT_DRAWS", 300):
            got = rs.leg1(self.ts, test, self.student, self.prior)
        self.assertEqual(len(test), got["all"]["n"])
        self.assertEqual({"card", "route", "shop", "reward"}, set(got["kinds"]))
        for kind, c in got["kinds"].items():
            self.assertGreater(c["gain"]["mean"], 0.0, kind)
            self.assertAlmostEqual(c["ce_prior"] - c["ce_student"], c["gain"]["mean"], places=4, msg=kind)  # the gain is the paired difference
            self.assertLessEqual(c["gain"]["lo95"], c["gain"]["hi95"])
            self.assertGreaterEqual(c["ce_student"], c["ce_floor"] - 1e-6)  # no cross-entropy is below the target's own entropy
        self.assertGreater(got["all"]["gain"]["lo95"], 0.0)

    def test_the_student_is_no_better_than_the_prior_where_it_is_the_prior(self):
        test = self.ts.split_idx()["test"]
        same = rs.leg1(self.ts, test, hpi.StudentScorer.load(student_file(self.root / "same.npz", self.weights)), self.prior)
        for c in same["kinds"].values():
            self.assertEqual(0.0, c["gain"]["mean"])
            self.assertEqual((0.0, 0.0), (c["gain"]["lo95"], c["gain"]["hi95"]))

    def test_the_kl_term_holds_the_student_to_the_prior(self):
        free = self.res["params"]
        held = rs.fit(self.ts, self.weights, cfg=rs.FitConfig(epochs=14, patience=14, batch_teacher=64, lr=2e-3, beta=0.0, kl=30.0, dropout=0.0),
                      device="cpu", log=lambda *_: None)["params"]
        idx = np.arange(300)

        def drift(params):
            """Mean absolute change of the decisions' log-probabilities (a softmax cannot see a shift of all of a decision's logits)."""
            scorer = hpi.PriorScorer(params)
            out = []
            for i in idx:
                a, b = int(self.ts.start[i]), int(self.ts.start[i] + self.ts.count[i])
                x, c = self.ts.dense[a:b], self.ts.counts[a:b]
                out.append(np.abs(hpi.log_softmax(scorer.logits(x, c)) - hpi.log_softmax(self.prior.logits(x, c))).mean())
            return float(np.mean(out))

        self.assertLess(drift(held), 0.5 * drift(free))

    def test_the_human_term_runs_beside_the_teachers_and_leaves_the_exam_alone(self):
        ids = run_ids_by_bucket(25)
        human = human_dataset({1: ids[1][:12], 2: ids[2][:12], 3: ids[3][:12]})
        rs.check_no_exam(human, frozenset())
        res = rs.fit(self.ts, self.weights, human=human, cfg=rs.FitConfig(epochs=2, patience=2, batch_teacher=128, batch_human=32, beta=1.0, kl=0.1,
                                                                        lr=1e-3, dropout=0.0), device="cpu", log=lambda *_: None)
        self.assertIsNotNone(res["history"][0]["train_human"])
        self.assertTrue(math.isfinite(res["history"][-1]["train_human"]))

    def test_lambda_per_kind_and_act_comes_from_the_held_out_runs_and_falls_back_as_stated(self):
        test = self.ts.split_idx()["test"]
        got = rs.measure_lambda(self.ts, test, self.student)
        self.assertEqual(set(rs.KINDS), set(got["kinds"]))
        for kind, triple in got["chosen"].items():
            self.assertEqual(3, len(triple))
            self.assertEqual(triple[1], triple[2], kind)  # Act 3 takes Act 2's
            self.assertTrue(all(0.0 <= x <= rs.LAMBDA_CAP for x in triple), (kind, triple))
        self.assertNotIn("rest", got["chosen"])  # no rest decisions: no lambda

    def test_the_candidate_is_refused_when_the_legs_are_for_another_student(self):
        out = self.root / "round"
        out.mkdir(exist_ok=True)
        (out / rs.LEGS_FILE).write_text(json.dumps({"student_sha256": "0" * 64}))
        (out / rs.STUDENT_OUT).write_bytes(Path(self.student_path).read_bytes())
        with self.assertRaisesRegex(ValueError, "not for this student"):
            rs.build_student_candidate(out, self.root, self.root, 3, why="t")


# ------------------------------------------------------------------------------------------------------------------ leg 2


class Leg2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids = run_ids_by_bucket(40)
        cls.weights = random_weights()
        cls.prior = hpi.PriorScorer(cls.weights)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.same = hpi.StudentScorer.load(student_file(Path(cls.tmp.name) / "same.npz", cls.weights))
        worse = dict(cls.weights)
        worse["score.weight"] = (cls.weights["score.weight"] * 300.0 + np.random.default_rng(4).normal(0, 0.1, (1, 256))).astype(np.float32)
        cls.worse = hpi.StudentScorer.load(student_file(Path(cls.tmp.name) / "worse.npz", worse))
        cls.ds = human_dataset({4: cls.ids[4][:40]}, per_run=3)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_a_student_equal_to_the_prior_has_zero_difference_and_every_human_kind_is_measured(self):
        with mock.patch.object(rs, "BOOT_DRAWS", 200):
            got = rs.leg2(self.ds, self.same, self.prior, exam_ids=frozenset())
        self.assertEqual(set(rs.HUMAN_KINDS), set(got["kinds"]))
        for kind, c in got["kinds"].items():
            self.assertEqual(0.0, c["all"]["diff"], kind)
            self.assertEqual(0.0, c["won"]["diff"], kind)
            self.assertEqual(c["all"]["ll_student"], c["all"]["ll_prior"])
        self.assertEqual({"reward", "select"}, set(got["no_human_fit"]))  # no human choice exists for these: leg 1 alone decides them
        self.assertTrue(got["measured"])

    def test_a_worse_student_shows_a_positive_difference_with_the_runs_resampled(self):
        with mock.patch.object(rs, "BOOT_DRAWS", 200):
            got = rs.leg2(self.ds, self.worse, self.prior, exam_ids=frozenset())
        self.assertTrue(any(c["all"]["diff"] > 0.02 for c in got["kinds"].values()))
        for c in got["kinds"].values():
            self.assertLessEqual(c["all"]["lo95"], c["all"]["diff"] + 1e-9)
            self.assertGreaterEqual(c["all"]["hi95"], c["all"]["diff"] - 1e-9)
            self.assertEqual(40, c["all"]["runs"])

    def test_a_cell_over_fewer_than_twenty_runs_is_hidden_and_its_kind_has_no_human_fit(self):
        ds = human_dataset({4: self.ids[4][:12]}, per_run=3)
        got = rs.leg2(ds, self.same, self.prior, exam_ids=frozenset())
        self.assertEqual([], list(got["kinds"]))
        self.assertEqual(set(rs.HUMAN_KINDS), set(got["hidden"]))

    def test_the_won_runs_cell_is_hidden_when_under_twenty_won_runs(self):
        ds = human_dataset({4: self.ids[4][:40]}, per_run=3, won_every=4)  # a quarter won: 10 runs
        got = rs.leg2(ds, self.same, self.prior, exam_ids=frozenset())
        self.assertTrue(all(c["won"] is None and c["all"] is not None for c in got["kinds"].values()))
        legs = {"leg2": got, "leg1": {"kinds": {"card": {"n": 500, "gain": {"mean": 0.1, "lo95": 0.05, "hi95": 0.15}}}}, "lambda": {"chosen": {"card": [0.1] * 3}}}
        v = rs.kind_verdict(legs, "card")
        self.assertFalse(v["pass"])
        self.assertIn("not measured on the won runs", v["why"])

    def test_leg_2_reads_no_exam_run(self):
        planted = human_dataset({4: self.ids[4][:25] + self.ids[0][:2]}, per_run=1)
        with self.assertRaisesRegex(AssertionError, "exam fifth"):
            rs.leg2(planted, self.same, self.prior, exam_ids=frozenset())


class VerdictTests(unittest.TestCase):
    def legs(self, *, gain_lo=0.01, n=500, won=0.0015, allr=0.01, lam=(0.2, 0.2, 0.2), kind="card", human=True, measured=True):
        l2 = {"measured": measured, "kinds": {}, "no_human_fit": [] if human else [kind]}
        if human and measured:
            l2["kinds"][kind] = {"won": {"n": 900, "runs": 200, "diff": won, "lo95": won - 0.01, "hi95": won + 0.01},
                                 "all": {"n": 9000, "runs": 2000, "diff": allr, "lo95": allr - 0.01, "hi95": allr + 0.01}}
        return {"leg1": {"kinds": {kind: {"n": n, "gain": {"mean": gain_lo + 0.02, "lo95": gain_lo, "hi95": gain_lo + 0.04}}}}, "leg2": l2,
                "lambda": {"chosen": {kind: list(lam)} if lam else {}}}

    def test_a_kind_passes_with_leg_1_above_zero_and_the_human_bar_met(self):
        v = rs.kind_verdict(self.legs(), "card")
        self.assertEqual((True, True, True, ""), (v["leg1"], v["leg2"], v["pass"], v["why"]))

    def test_the_human_bar_is_two_thousandths_on_won_runs_and_two_hundredths_on_all(self):
        self.assertTrue(rs.kind_verdict(self.legs(won=0.0019), "card")["pass"])
        self.assertTrue(rs.kind_verdict(self.legs(won=0.002), "card")["pass"])  # at the margin is within it
        self.assertFalse(rs.kind_verdict(self.legs(won=0.0021), "card")["pass"])
        self.assertTrue(rs.kind_verdict(self.legs(allr=0.02), "card")["pass"])
        self.assertFalse(rs.kind_verdict(self.legs(allr=0.0201), "card")["pass"])
        self.assertTrue(rs.kind_verdict(self.legs(won=-0.05, allr=-0.05), "card")["pass"])  # better than the prior passes
        why = rs.kind_verdict(self.legs(won=0.01), "card")["why"]
        self.assertIn("leg 2", why)
        self.assertEqual((rs.TIE_NATS, rs.LIMIT_NATS), (0.002, 0.02))

    def test_the_margins_are_the_strong_sets_rules(self):
        import strong_sets as ss

        self.assertEqual((ss.TIE_NATS, ss.LIMIT_B_NATS), (rs.TIE_NATS, rs.LIMIT_NATS))
        self.assertEqual(ss.MIN_CELL, rs.MIN_CELL)

    def test_leg_1_needs_a_gain_interval_above_zero_and_enough_decisions(self):
        self.assertFalse(rs.kind_verdict(self.legs(gain_lo=0.0), "card")["pass"])
        self.assertFalse(rs.kind_verdict(self.legs(gain_lo=-0.01), "card")["pass"])
        v = rs.kind_verdict(self.legs(n=rs.LEG1_MIN_DECISIONS - 1), "card")
        self.assertFalse(v["pass"])
        self.assertIn("needs 100", v["why"])
        self.assertTrue(rs.kind_verdict(self.legs(n=rs.LEG1_MIN_DECISIONS), "card")["pass"])
        self.assertFalse(rs.kind_verdict({"leg1": {"kinds": {}}, "leg2": {}}, "card")["pass"])

    def test_a_kind_with_no_human_fit_hands_over_on_leg_1_alone(self):
        for kind in ("reward", "select"):
            v = rs.kind_verdict(self.legs(kind=kind, human=False, won=9.9, allr=9.9), kind)
            self.assertEqual((True, None, True), (v["leg1"], v["leg2"], v["pass"]), kind)
        # a kind the record does have, when the held-out human data had no decision for it (a build without events), is the same
        v = rs.kind_verdict(self.legs(kind="event", human=False), "event")
        self.assertEqual((True, None, True), (v["leg1"], v["leg2"], v["pass"]))
        # ... but not when the human bar simply was not run
        v = rs.kind_verdict(self.legs(kind="card", measured=False), "card")
        self.assertFalse(v["pass"])
        self.assertIn("not measured", v["why"])
        v = rs.kind_verdict(self.legs(kind="reward", measured=False), "reward")
        self.assertTrue(v["pass"])  # reward has no human choices to measure against

    def test_a_kind_needs_a_measured_lambda(self):
        v = rs.kind_verdict(self.legs(lam=None), "card")
        self.assertFalse(v["pass"])
        self.assertIn("no measured lambda", v["why"])

    def test_the_candidate_takes_passing_kinds_and_leaves_a_no_op_student_out(self):
        legs = {"leg1": {"kinds": {}}, "leg2": {"measured": True, "kinds": {}, "no_human_fit": ["reward", "select"]}, "lambda": {"chosen": {}}}
        for kind, lam in (("reward", (0.3, 0.3, 0.3)), ("select", (0.0, 0.0, 0.0)), ("card", (0.0, 0.0, 0.0))):
            legs["leg1"]["kinds"][kind] = {"n": 400, "gain": {"mean": 0.05, "lo95": 0.01, "hi95": 0.09}}
            legs["lambda"]["chosen"][kind] = list(lam)
        legs["leg2"]["kinds"]["card"] = {"won": {"n": 1, "runs": 99, "diff": 0.0, "lo95": 0, "hi95": 0}, "all": {"n": 1, "runs": 99, "diff": 0.0, "lo95": 0, "hi95": 0}}
        got = rs.passing_kinds(legs)
        # select's lambdas are all 0 and the base decides it by vrun alone: the student would be vrun again, so it is not named; card's lambda 0
        # replaces the prior's blend by vrun, which is a change, so it is
        self.assertEqual({"reward": [0.3, 0.3, 0.3], "card": [0.0, 0.0, 0.0]}, got)


class LambdaTests(unittest.TestCase):
    def planted(self, lam_true: float, n: int = 300, seed: int = 1):
        rng = np.random.default_rng(seed)
        weights = random_weights()
        scorer = hpi.PriorScorer(weights)
        out_q, out_v, rows_all = [], [], []
        decisions = []
        for _ in range(n):
            rows = rng.normal(0, 0.4, (4, rs.ROWS_WIDTH)).astype(np.float32)
            dense, counts = models.value_inputs(rows)
            logp = hpi.log_softmax(scorer.logits(dense, counts) * 40.0)  # a spread the regression can see
            v = rng.normal(0, 1.5, 4)
            q = v + lam_true * logp + rng.normal(0, 0.3, 4)
            decisions.append([(float(a), float(b), float(c)) for a, b, c in zip(q, v, logp, strict=True)])
        return decisions

    def test_the_regression_recovers_a_planted_lambda_with_a_standard_error_that_covers_it(self):
        for lam in (0.0, 0.25, 0.8):
            fit = rs.teacher_fn("lambda_fit")(self.planted(lam))
            self.assertLess(abs(fit["lambda"] - lam), 3 * fit["se"] + 1e-3, (lam, fit))
            self.assertEqual(300, fit["decisions"])

    def test_lambda_and_the_noise_are_run_teachers_own_functions(self):
        import run_teacher as rt

        self.assertIs(rt.lambda_fit, rs.teacher_fn("lambda_fit"))
        self.assertIs(rt.noise_and_spread, rs.teacher_fn("noise_and_spread"))

    def test_the_regression_is_exact_on_a_hand_computation(self):
        # one decision, three options: q = v + 0.5 l exactly, so lambda is 0.5 and the standard error 0
        l = [-1.0, 0.0, 1.0]
        v = [3.0, 1.0, 2.0]
        data = [[(v[i] + 0.5 * l[i], v[i], l[i]) for i in range(3)]] * 4
        fit = rs.teacher_fn("lambda_fit")(data)
        self.assertEqual((0.5, 0.0, 4), (fit["lambda"], fit["se"], fit["decisions"]))
        self.assertEqual({"decisions": 0, "lambda": None, "se": None}, rs.teacher_fn("lambda_fit")([[(1.0, 1.0, 0.0)]]))


# ------------------------------------------------------------------------------------------------------------------ the whole round


@unittest.skipIf(torch is None, "torch is not installed")
class RoundTests(unittest.TestCase):
    def test_a_round_on_synthetic_shards_writes_the_student_the_legs_and_the_verdict(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(rs, "BOOT_DRAWS", 200), mock.patch.object(rs, "LEG1_MIN_DECISIONS", 20), \
                mock.patch.object(rs, "MIN_LAMBDA_DECISIONS", 10):
            root = Path(tmp)
            weights = random_weights()
            prior_path = write_weights(root / "pi_h.npz", weights)
            synth_shards(root, Synth(weights, seed=3), runs=260, kinds=("card", "route", "reward"))
            ids = run_ids_by_bucket(30)
            human = {k: human_dataset({f: ids[f][:30]}, per_run=1) for k, f in ((1, 1), (4, 4))}
            with mock.patch.object(rs, "load_human", side_effect=lambda d, fifths, **kw: human[1 if fifths == rs.HUMAN_TRAIN_FIFTHS else 4]):
                legs = rs.run_fit(root, prior_path, root / "out", human_dir=root, exam_ids=frozenset(), cfg=rs.FitConfig(epochs=6, patience=6, lr=2e-3, beta=0.5, kl=0.05,
                                                                                                   batch_human=16, dropout=0.0),
                                  device="cpu", log=lambda *_: None)
            for name in (rs.STUDENT_OUT, rs.FIT_FILE, rs.LEGS_FILE):
                self.assertTrue((root / "out" / name).is_file(), name)
            self.assertEqual(hpi.file_sha256(root / "out" / rs.STUDENT_OUT), legs["student_sha256"])
            self.assertEqual(hpi.file_sha256(prior_path), legs["baseline"]["sha256"])
            self.assertEqual(set(rs.KINDS), set(legs["verdict"]))
            self.assertEqual({"reward", "select"}, {"reward", "select"} & set(legs["leg2"]["no_human_fit"]))
            text = (root / "out" / rs.LEGS_FILE).read_text() + (root / "out" / rs.FIT_FILE).read_text()
            self.assertNotIn("S00000000", text)  # no run is named
            self.assertNotIn(str(root).replace("\\", "/"), text.replace("\\\\", "/"))  # and no absolute path
            self.assertTrue(legs["leg1"]["kinds"]["card"]["gain"]["mean"] > 0)


class CommandLineTests(unittest.TestCase):
    def test_check_prints_the_counts_and_names_no_run(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            weights = random_weights()
            write_weights(Path(tmp) / "pi_h.npz", weights)
            synth_shards(Path(tmp), Synth(weights, seed=2), runs=20, kinds=("card", "route"))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = rs.main(["check", "--shards-root", tmp, "--prior", str(Path(tmp) / "pi_h.npz")])
            doc = json.loads(out.getvalue())
            self.assertEqual(0, rc)
            self.assertEqual((80, 20, {"card": 40, "route": 40}), (doc["decisions"], doc["runs"], doc["kept_by_kind"]))
            self.assertNotIn("S00000000", out.getvalue())

    def test_the_commands_and_their_defaults(self):
        p = rs.build_parser()
        a = p.parse_args(["round", "--shards-root", "s", "--prior", "p", "--out", "o", "--base", "b", "--agent-root", "r", "--target", "9"])
        self.assertEqual((rs.FitConfig.beta, rs.FitConfig.kl, a.tag, a.no_build, a.lam), (a.beta, a.kl, "student", False, None))
        self.assertEqual(9, a.target)
        with self.assertRaises(SystemExit):
            p.parse_args(["fit", "--shards-root", "s"])  # the prior and the output are required


if __name__ == "__main__":
    unittest.main()
