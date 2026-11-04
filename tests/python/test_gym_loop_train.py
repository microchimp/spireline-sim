"""The loop's trainer: the pass cap, the heads' resets and the offline replay (rl/gym_loop_train.py; training design 1b, section 1.4).

The pinned facts: after every `reset_every`-th checkpoint the last linear layers of the value head and of the policy head are
re-initialised (and their Adam moments dropped) while the encoder and every other weight stay as they were, and the checkpoint
written at that point keeps the heads it was scored with; the default flags (no reset, 4 passes) train exactly as the trainer did
before the change (a seed-fixed chunk gives the same losses and weights); the budget takes `--max-epochs 16`; the replay trains
arms from one base over the stored shards in arrival order, scores every checkpoint on the same held-out batches (the figures
`evaluate_model` gives), writes only to its own folder, and its summary adopts an arm by the design's rule. Synthetic shards and a
tiny network on the CPU; no game, no card.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT / "rl", ROOT / "src", ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gym_loop as gl  # noqa: E402
import gym_loop_train as tr  # noqa: E402


def fake_shard(path: Path, rows: int, seed: int, val_every: int = 7) -> None:
    """A shard as the buffer stores one: tokens with an action block (tokens 1 to 3), root visits on the legal actions, an outcome."""
    from combat_tokens import TYPE_BUFF, TYPE_END_TURN, TYPE_ENEMY, TYPE_GLOBAL, TYPE_HAND  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    lengths = rng.integers(5, 9, rows).astype(np.int32)
    head = [TYPE_GLOBAL, TYPE_HAND, TYPE_HAND, TYPE_END_TURN, TYPE_ENEMY]
    types = np.concatenate([head + [TYPE_BUFF] * (int(n) - 5) for n in lengths])
    t = int(lengths.sum())
    legal = np.zeros((rows, 32), np.int8)
    legal[:, :3] = 1
    visits = (rng.random((rows, 32)) * legal).astype(np.float32)  # the draws in this order: the pinned numbers below depend on it
    label = rng.integers(0, 40, rows).astype(np.int32)
    np.savez(path, types=types.astype(np.int16), ids=rng.integers(0, 5, t).astype(np.int32), owners=np.zeros(t, np.int16),
             feats=rng.random((t, 18)).astype(np.float32), lengths=lengths, visits=visits,
             legal=legal, label=label, value_ok=np.ones(rows, np.int8),
             val=(np.arange(rows) % val_every == 0).astype(np.int8), exact=np.zeros(rows, np.int8),
             exact_value=np.full(rows, np.nan, np.float32), q_played=np.full(rows, np.nan, np.float32),
             solver_best=np.full(rows, -1, np.int16), played=np.zeros(rows, np.int16))


def make_base(d: Path, seed: int = 0) -> None:
    import torch  # noqa: PLC0415
    from combat_net import CombatNet  # noqa: PLC0415
    from combat_tokens import VOCAB  # noqa: PLC0415

    d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    m = CombatNet(VOCAB, 8, d_model=16, layers=1, heads=2)
    torch.save({"model": m.state_dict(), "n_buckets": 8, "width": 5, "max_hp": 40, "d_model": 16, "layers": 1, "heads": 2},
               d / "combat_policy.pt")
    (d / "model.json").write_text(json.dumps({"kind": "combat_prior"}), encoding="utf-8")


def make_loop(tmp: str, shards: int = 3, rows: int = 300) -> tuple[Path, Path]:
    root = Path(tmp) / "loop"
    (root / "buffer" / "shards").mkdir(parents=True)
    for i in range(shards):
        fake_shard(root / "buffer" / "shards" / f"s{i + 1:06d}-{1000 + i}.npz", rows, i)
    make_base(Path(tmp) / "base")
    return root, Path(tmp) / "base"


def namespace(root: Path, base: Path, **kw) -> argparse.Namespace:
    d = dict(root=str(root), base=str(base), parent_pid=0, every_min=0.0, max_epochs=2.0, window=10_000, batch=32, lr=1e-4,
             policy_weight=1.0, seed=0)
    d.update(kw)
    return argparse.Namespace(**d)


def head_state(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(("head.3.", "policy.3."))}


class ResetTests(unittest.TestCase):
    def test_reset_heads_reinitialises_the_last_linear_layers_and_not_the_encoder(self):
        import torch  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            make_base(Path(tmp) / "base")
            model, shape = tr.load_net(Path(tmp) / "base" / "combat_policy.pt")
            names = [n for n, _ in model.named_parameters()]
            self.assertIn("head.3.weight", names)
            self.assertIn("policy.3.weight", names)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            flat = self._flat(tmp)
            rng = np.random.default_rng(0)
            for _ in range(2):
                tr.train_step(model, opt, flat, flat["train_idx"], rng, "cpu", shape, 32, 1.0)
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            reset = tr.reset_heads(model, opt, seed=5)
            self.assertEqual(sorted(reset), ["head.bias", "head.weight", "policy.bias", "policy.weight"])
            after = model.state_dict()
            for k, v in before.items():
                if k in ("head.3.weight", "head.3.bias", "policy.3.weight", "policy.3.bias"):
                    self.assertFalse(torch.equal(v, after[k]), k)  # re-initialised
                else:
                    self.assertTrue(torch.equal(v, after[k]), k)  # the encoder, the first layers of the heads, the norms: untouched
            bound = 1.0 / np.sqrt(model.head[3].in_features)
            self.assertLessEqual(float(after["head.3.weight"].abs().max()), bound)
            self.assertLessEqual(float(after["policy.3.bias"].abs().max()), bound)
            params = dict(model.named_parameters())
            for k in ("head.3.weight", "head.3.bias", "policy.3.weight", "policy.3.bias"):
                self.assertNotIn(params[k], opt.state)  # Adam starts again on them
            self.assertIn(params["head.1.weight"], opt.state)  # and keeps its moments everywhere else
            again, shape = tr.load_net(Path(tmp) / "base" / "combat_policy.pt")
            tr.reset_heads(again, None, seed=5)
            self.assertTrue(torch.equal(again.state_dict()["policy.3.weight"], after["policy.3.weight"]))  # seeded: the same draw
            tr.reset_heads(again, None, seed=6)
            self.assertFalse(torch.equal(again.state_dict()["policy.3.weight"], after["policy.3.weight"]))

    @staticmethod
    def _flat(tmp: str) -> dict:
        shards = Path(tmp) / "shards"
        shards.mkdir(exist_ok=True)
        fake_shard(shards / "s000001-1000.npz", 200, 3)
        buf = tr.Buffer(shards, 10_000)
        buf.refresh()
        return buf.arrays()

    def test_the_trainer_resets_after_every_nth_checkpoint_and_the_checkpoint_keeps_the_heads_it_was_scored_with(self):
        import torch  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"GYM_LOOP_NO_GPU": "1"}):
            root, base = make_loop(tmp)
            t = tr.Trainer(namespace(root, base, reset_every=2, max_epochs=1.0))
            t.refresh()

            def train_some():
                t.st["budget"] += 4 * t.a.batch
                t.train_chunk("cpu")

            seen = []
            for _ in range(3):
                train_some()
                trained = head_state(t.model)
                encoder = {k: v.detach().clone() for k, v in t.model.state_dict().items() if k not in trained}
                name = t.checkpoint()
                seen.append(name)
                saved = torch.load(root / "checkpoints" / name / "combat_policy.pt", map_location="cpu", weights_only=True)["model"]
                for k, v in trained.items():
                    self.assertTrue(torch.equal(saved[k], v), k)  # the checkpoint holds what it was scored with
                now = t.model.state_dict()
                for k, v in encoder.items():
                    self.assertTrue(torch.equal(now[k], v), k)  # the encoder is never reset
                changed = any(not torch.equal(now[k], v) for k, v in trained.items())
                self.assertEqual(changed, len(seen) == 2, (len(seen), changed))  # only after the 2nd checkpoint
            self.assertEqual(t.st["resets"], 1)
            state = torch.load(root / "trainer" / "state.pt", map_location="cpu", weights_only=False)
            self.assertEqual(state["st"]["resets"], 1)
            self.assertIn("reset", (root / "trainer" / "trainer.log").read_text(encoding="utf-8"))

    def test_without_the_flag_nothing_is_ever_reset(self):
        import torch  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"GYM_LOOP_NO_GPU": "1"}):
            root, base = make_loop(tmp)
            a = namespace(root, base, max_epochs=1.0)
            self.assertFalse(hasattr(a, "reset_every"))  # the Namespace the trainer had before the flag existed
            t = tr.Trainer(a)
            t.refresh()
            for _ in range(4):
                t.st["budget"] += 2 * a.batch
                t.train_chunk("cpu")
                before = head_state(t.model)
                t.checkpoint()
                for k, v in before.items():
                    self.assertTrue(torch.equal(t.model.state_dict()[k], v), k)
            self.assertNotIn("resets", t.st)


class DefaultsTests(unittest.TestCase):
    def test_default_flags_train_exactly_as_the_trainer_did(self):
        """A seed-fixed chunk gives the same losses and weights as the body `train_chunk` had before `train_step` was taken out of it."""
        import torch  # noqa: PLC0415

        def old_chunk(t):  # the pre-change loop body, kept here as the reference
            flat = t.buffer.arrays()
            idx = flat["train_idx"]
            t.model.train(True)
            while t.st["budget"] >= t.a.batch:
                rows = idx[t.rng.integers(0, len(idx), t.a.batch)]
                inputs, tg = tr.batch_tensors(flat, rows, "cpu", t.shape["width"], t.shape["n_buckets"])
                vl, pl, _ = tr.losses(t.model, inputs, tg)
                loss = vl + t.a.policy_weight * pl
                t.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(t.model.parameters(), 1.0)
                t.opt.step()
                t.st["budget"] -= t.a.batch
                t.st["steps"] += 1
                v, p = float(vl.detach()), float(pl.detach())
                t.st["ema_value"] = 0.98 * t.st.get("ema_value", v) + 0.02 * v
                t.st["ema_policy"] = 0.98 * t.st.get("ema_policy", p) + 0.02 * p

        runs = []
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"GYM_LOOP_NO_GPU": "1"}):
            root, base = make_loop(tmp)
            for which in ("new", "old"):
                t = tr.Trainer(namespace(root, base))
                t.refresh()
                if which == "new":
                    while t.st["budget"] >= t.a.batch:
                        t.train_chunk("cpu")
                else:
                    old_chunk(t)
                runs.append((t.st["steps"], t.st["ema_value"], t.st["ema_policy"],
                             sum(float(p.detach().double().sum()) for p in t.model.parameters())))
        self.assertEqual(runs[0], runs[1])
        steps, ema_v, ema_p, psum = runs[0]
        # the same chunk on the code before this change (measured there on 2026-11-04)
        self.assertEqual(steps, 48)
        self.assertAlmostEqual(ema_v, 2.1019219547845753, places=5)
        self.assertAlmostEqual(ema_p, 1.097120296298004, places=5)
        self.assertAlmostEqual(psum, -106.74800938595331, places=3)

    def test_the_budget_takes_sixteen_passes_and_the_flags_reach_the_trainer(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"GYM_LOOP_NO_GPU": "1"}):
            root, base = make_loop(tmp)
            t = tr.Trainer(namespace(root, base, max_epochs=16.0))
            t.refresh()
            train_rows = len(t.buffer.arrays()["train_idx"])
            self.assertEqual(t.st["budget"], 16 * train_rows)
            n = 0
            while t.st["budget"] >= t.a.batch:
                n += t.train_chunk("cpu")
            self.assertEqual(n, 16 * train_rows // t.a.batch)
        with mock.patch.object(tr, "Trainer") as trainer:
            trainer.return_value.run.return_value = 0
            tr.main(["--root", "r", "--base", "b", "--max-epochs", "16", "--reset-every", "24"])
            a = trainer.call_args.args[0]
            self.assertEqual((a.max_epochs, a.reset_every), (16.0, 24))
            tr.main(["--root", "r", "--base", "b"])
            a = trainer.call_args.args[0]
            self.assertEqual((a.max_epochs, a.reset_every), (4.0, 0))  # today's defaults


class ReplayTests(unittest.TestCase):
    def test_the_prepared_scoring_gives_the_figures_evaluate_model_gives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, base = make_loop(tmp, shards=2)
            buf = tr.Buffer(root / "buffer" / "shards", 10_000)
            buf.refresh()
            flat = buf.arrays()
            model, shape = tr.load_net(base / "combat_policy.pt")
            want = tr.evaluate_model(flat, model, shape, "cpu", limit=70)
            got = tr.eval_prepared(model, tr.prepare_eval(flat, shape, "cpu", limit=70))
            self.assertEqual(want["val_rows"], 70)
            for k, v in want.items():
                self.assertAlmostEqual(got[k], v, places=9, msg=k)

    def test_arms_train_in_arrival_order_from_one_base_and_are_scored_on_the_same_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, base = make_loop(tmp, shards=5, rows=140)
            before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
            out = Path(tmp) / "out"
            arms = [tr.parse_arm("a:2"), tr.parse_arm("b:4:2:3"), tr.parse_arm("c:16")]
            said: list[str] = []
            summary = tr.replay(root, base, out, arms, window=10_000, limit=60, batch=16, device="cpu", say=said.append)
            self.assertEqual(sorted(str(p.relative_to(root)) for p in root.rglob("*")), before)  # the loop's folder is only read
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["a.jsonl", "b.jsonl", "c.jsonl", "summary.json"])
            rows = {a["name"]: [json.loads(x) for x in (out / f"{a['name']}.jsonl").read_text().splitlines()] for a in arms}
            for name, r in rows.items():
                self.assertEqual([x["ck"] for x in r], list(range(6)))  # the base, then one checkpoint a shard
                self.assertEqual([x["shard"] for x in r[1:]], [f"s{i:06d}-{999 + i}.npz" for i in range(1, 6)])
                self.assertEqual([x["rows"] for x in r], [0, 140, 280, 420, 560, 700])
                self.assertTrue(all(x["val_rows"] == 60 for x in r))  # the same held-out rows for every arm and checkpoint
            # the starting point is the base: the same figures for every arm
            self.assertEqual({round(r[0]["val_policy_ce"], 9) for r in rows.values()}, {round(rows["a"][0]["val_policy_ce"], 9)})
            # budget: `passes` times the training rows that have arrived, spent in steps of the batch
            per_shard = 120  # 140 rows, every 7th held out
            for name, passes in (("a", 2), ("b", 4), ("c", 16)):
                for x in rows[name][1:]:
                    self.assertLessEqual(x["samples"], passes * per_shard * x["ck"])
                    self.assertGreater(x["samples"], passes * per_shard * x["ck"] - 16)
            self.assertEqual(rows["c"][-1]["samples"], 16 * per_shard * 5)  # the budget takes 16 passes
            self.assertEqual([x["reset_after"] for x in rows["b"][1:]], [False, True, False, True, False])
            self.assertEqual([x["resets"] for x in rows["b"][1:]], [0, 0, 1, 1, 2])
            self.assertFalse(any(x["reset_after"] for x in rows["a"]))
            self.assertEqual(sorted(summary["arms"]), ["a", "b", "c"])
            self.assertEqual(summary["arms"]["c"]["end"]["val_policy_ce"], rows["c"][-1]["val_policy_ce"])
            self.assertEqual(json.loads((out / "summary.json").read_text())["split"], gl.val_split_id())
            # a rerun keeps the arms that finished
            said.clear()
            tr.replay(root, base, out, arms, window=10_000, limit=60, batch=16, device="cpu", say=said.append)
            self.assertEqual(sum("kept from an earlier run" in m for m in said), 3)

    def test_the_replay_will_not_write_inside_the_loop_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, base = make_loop(tmp, shards=1)
            with self.assertRaises(SystemExit):
                tr.main(["replay", "--root", str(root), "--base", str(base), "--out", str(root / "replay")])
            self.assertFalse((root / "replay").exists())

    def test_the_card_is_free_only_while_gpu_ok_is_absent_or_the_loops_and_the_live_trainer_is_off_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "GPU_OK"
            with mock.patch.object(gl, "GPU_OK", f):
                self.assertTrue(tr.card_free())  # absent
                f.write_text("gym-loop\n", encoding="utf-8")
                self.assertTrue(tr.card_free())
                f.write_text("gym-loop\ngym-loop-train\n", encoding="utf-8")
                self.assertFalse(tr.card_free())  # the live trainer keeps priority
                f.write_text("somebody-else\ngym-loop\n", encoding="utf-8")
                self.assertFalse(tr.card_free())

    def test_the_arm_spec_and_the_summary_rule(self):
        self.assertEqual(tr.parse_arm("x:8:24:2"), {"name": "x", "epochs": 8.0, "reset_every": 24, "seed": 2})
        self.assertEqual(tr.parse_arm("cap16:16"), {"name": "cap16", "epochs": 16.0, "reset_every": 0, "seed": 0})
        with self.assertRaises(ValueError):
            tr.parse_arm("cap16")
        self.assertEqual([tr.parse_arm(s)["name"] for s in tr.DEFAULT_ARMS],
                         ["cap4a", "cap4b", "cap8", "cap16", "cap8r24", "cap16r24"])

        def arm(name, ce, **kw):
            rows = [{"arm": name, "ck": k, "rows": 100 * k, "steps": k, "card_s": float(k), "val_policy_ce": c, "val_value_nll": 1.0,
                     "val_top1_agree": 0.7, "reset_after": False} for k, c in enumerate([1.2, *ce])]
            return rows, {"name": name, "epochs": kw.get("epochs", 4.0), "reset_every": kw.get("reset", 0), "seed": kw.get("seed", 0)}

        def run(**ends):
            spec, rows = {}, {}
            for name, (ce, kw) in ends.items():
                rows[name], spec[name] = arm(name, ce, **kw)
            return tr.summarize(rows, spec)["rule"]

        base = {"cap4a": ([1.06, 1.050], {}), "cap4b": ([1.06, 1.052], {"seed": 1})}
        r = run(**base, cap8=([1.05, 1.040], {"epochs": 8.0}), cap16=([1.05, 1.030], {"epochs": 16.0}))
        self.assertAlmostEqual(r["cap4_spread"], 0.002)
        self.assertEqual((r["best_at_end"], r["adopt"]), ("cap16", "cap16"))
        r = run(**base, cap8=([1.05, 1.0505], {"epochs": 8.0}), cap16=([1.05, 1.0495], {"epochs": 16.0}))
        self.assertEqual(r["adopt"], "cap4")  # no arm beats cap 4 by more than the two cap-4 seeds differ
        r = run(**base, cap16=([1.05, 1.040], {"epochs": 16.0}), cap16r24=([1.05, 1.039], {"epochs": 16.0, "reset": 24}))
        self.assertEqual((r["best_at_end"], r["adopt"]), ("cap16r24", "cap16"))  # resets only if they beat the same cap without
        r = run(**base, cap16=([1.05, 1.040], {"epochs": 16.0}), cap16r24=([1.05, 1.030], {"epochs": 16.0, "reset": 24}))
        self.assertEqual(r["adopt"], "cap16r24")


if __name__ == "__main__":
    unittest.main()
