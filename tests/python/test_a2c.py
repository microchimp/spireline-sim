"""A2c: the paired score plays the scored agent's own emulator arm (gen0.score_real), the prior's search cost (bit-identical
results), and the V_run keep conditions with the agent.json macro switch. Synthetic data only."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import gen0  # noqa: E402
import gen_loop  # noqa: E402

SEEDS = ["AAAAAAAAAA", "BBBBBBBBBB", "CCCCCCCCCC", "DDDDDDDDDD"]


def _row(seed: str, win: bool = False, floor: int = 10) -> dict:
    return {"seed": seed, "ix": 0, "win": win, "floor": floor, "end": "death", "decisions": 50, "wall": 10.0, "crash": False}


def _gen_dir(root: Path, generation: int, sha: str) -> Path:
    d = root / f"gen{generation}" / "candidate"
    d.mkdir(parents=True, exist_ok=True)
    (d / "agent.json").write_text(json.dumps({"generation": generation, "macro": "rules"}), encoding="utf-8")
    (d / "model.json").write_text(json.dumps({"policy_sha256": sha}), encoding="utf-8")
    (d / "combat_policy.pt").write_bytes(b"x")
    return d


def _emu_arm(d: Path, agent: dict | None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "runs.jsonl").write_text("\n".join(json.dumps(_row(s)) for s in SEEDS) + "\n", encoding="utf-8")
    doc = {"sims": "a3-v1", "procs": 4}
    if agent is not None:
        doc["agent"] = agent
    (d / "summary.json").write_text(json.dumps(doc), encoding="utf-8")
    return d


def _args(tmp: Path, gen_dir: Path | None, emu_dir: Path | None) -> SimpleNamespace:
    return SimpleNamespace(game="G", seeds_file=str(tmp / "seeds.json"), runs=3, procs=4, sims=60, config="a3-v1", ascension=10,
                           character="IRONCLAD", stop_after_h=1.0, lib=None, emu_dir=str(emu_dir) if emu_dir else None,
                           gen_dir=str(gen_dir) if gen_dir else None, out=str(tmp / "job"))


class ScorePairsWithTheAgentsOwnEmulatorArmTests(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = Path(self._t.name)
        (self.tmp / "seeds.json").write_text(json.dumps({"seeds": SEEDS}), encoding="utf-8")
        self.gen3 = _gen_dir(self.tmp / "agent", 3, "sha-gen3")
        self.played: list[tuple] = []

    def tearDown(self):
        self._t.cleanup()

    def _score(self, a):
        def fake_play_emu_arm(args, agent, out):  # the emulator arm, mocked: writes what run_block would
            self.played.append((list(gen0.load_seeds(args.seeds_file)[: args.runs]), args.config, args.procs, agent.generation, out))
            _emu_arm(out, agent.describe())

        def fake_envproto(cmd, **kw):  # the real-game block, mocked
            out = Path(cmd[cmd.index("--out") + 1])
            (out / "runs.jsonl").write_text("{}\n", encoding="utf-8")
            (out / "summary.json").write_text(json.dumps({"wall_s": 5.0, "diverged_decisions": 0}), encoding="utf-8")
            return SimpleNamespace(returncode=0)

        real = [_row(s, floor=9) for s in SEEDS[:3]]
        with mock.patch.object(gen0, "play_emu_arm", fake_play_emu_arm), mock.patch("subprocess.run", fake_envproto), \
                mock.patch.object(gen0, "real_runs", return_value=real), mock.patch.object(gen0, "_setup"):
            rc = gen0.score_real(a)
        out = Path(a.out)
        return rc, json.loads((out / "summary.json").read_text()), json.loads((out / "compare.json").read_text())

    def test_a_missing_emulator_dir_is_played_on_the_same_seeds_then_paired(self):
        rc, summ, comp = self._score(_args(self.tmp, self.gen3, None))
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.played), 1)
        seeds, config, procs, generation, out = self.played[0]
        self.assertEqual((seeds, config, procs, generation), (SEEDS[:3], "a3-v1", 4, 3))
        self.assertEqual(out, Path(self.tmp / "job-emu"))
        self.assertEqual(summ["emu_arm"]["played"], True)
        self.assertEqual(summ["emu_arm"]["dir"], "job-emu")
        self.assertEqual(summ["emu_arm"]["agent"]["prior_sha256"], "sha-gen3")
        self.assertEqual(comp["paired_seeds"], 3)
        self.assertEqual(comp["emu_arm"]["played"], True)

    def test_a_stale_emulator_dir_of_another_agent_is_played_again(self):
        for stale in (_emu_arm(self.tmp / "e0", {"generation": 0, "prior_sha256": None}),
                      _emu_arm(self.tmp / "e3x", {"generation": 3, "prior_sha256": "other-sha"}),
                      _emu_arm(self.tmp / "e-legacy", None)):
            self.played.clear()
            _, summ, _ = self._score(_args(self.tmp, self.gen3, stale))
            self.assertEqual(len(self.played), 1, stale.name)
            self.assertTrue(summ["emu_arm"]["played"])
            self.assertIn("played:", summ["emu_arm"]["reason"])

    def test_a_matching_emulator_dir_is_used_and_not_played(self):
        good = _emu_arm(self.tmp / "e3", {"generation": 3, "prior_sha256": "sha-gen3", "prior": True, "macro": "rules"})
        _, summ, comp = self._score(_args(self.tmp, self.gen3, good))
        self.assertEqual(self.played, [])
        self.assertFalse(summ["emu_arm"]["played"])
        self.assertTrue(summ["emu_arm"]["dir"].endswith("e3"))
        self.assertEqual(comp["paired_seeds"], 3)

    def test_without_a_gen_dir_the_scored_agent_is_generation_0(self):
        g0 = _emu_arm(self.tmp / "e0", {"generation": 0, "prior_sha256": None})
        self._score(_args(self.tmp, None, g0))
        self.assertEqual(self.played, [])
        self._score(_args(self.tmp, None, None))
        self.assertEqual(len(self.played), 1)
        self.assertEqual(self.played[0][3], 0)

    def test_emu_arm_match_reports_why(self):
        ok, why = gen0.emu_arm_match(self.tmp / "nope", {"generation": 1})
        self.assertFalse(ok)
        self.assertIn("no finished", why)
        self.assertEqual(gen0.emu_arm_match(None, {})[0], False)


# ------------------------------------------------------------------ the prior's search cost: bit-identical answers

try:
    import numpy as np
    import torch

    import combat_prior
    from combat_net import CombatNet, collate
    from combat_tokens import TYPE_BUFF, TYPE_DISCARD, TYPE_DRAW, TYPE_ENEMY, TYPE_GLOBAL, TYPE_HAND, TYPE_INTENT, TYPE_RELIC, VOCAB, N_FEATURES
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prior_fixture import export_prior

    HAVE_TORCH = True
except Exception as _e:  # noqa: BLE001 -- no torch or native library here: those tests skip
    HAVE_TORCH = False
    print('torch tests skipped:', repr(_e), file=sys.stderr)


class _Tok:
    def __init__(self, n: int, rng) -> None:
        pool = np.array([TYPE_ENEMY, TYPE_BUFF, TYPE_RELIC, TYPE_DRAW, TYPE_DISCARD, TYPE_INTENT])
        self.types = np.concatenate([[TYPE_GLOBAL], [TYPE_HAND] * 6, pool[rng.integers(0, len(pool), n - 7)]]).astype(np.int64)
        self.ids = rng.integers(0, 4, n).astype(np.int64)
        self.ids[0] = 0
        self.owners = rng.integers(0, 2, n).astype(np.int64)
        self.feats = (rng.normal(size=(n, N_FEATURES)) * 0.3).astype(np.float32)

    def __len__(self) -> int:
        return len(self.types)


def _reference_weights(net, tokens, legals):
    """combat_prior.PolicyNet.weights as it was before A2c: the padded collate batch through the whole forward."""
    with torch.inference_mode():
        _, logits = net.model(**collate(list(tokens), torch.device("cpu")))
    out = []
    for row, legal in zip(logits.numpy(), legals, strict=True):
        index = np.asarray(legal, dtype=np.int64)
        scores = np.full(len(index), -np.inf)
        inside = index < row.shape[0]
        scores[inside] = row[index[inside]]
        finite = np.isfinite(scores)
        if not finite.any():
            out.append(np.full(len(index), 1.0 / max(1, len(index))))
            continue
        scores = np.where(finite, scores, -np.inf)
        weights = np.exp(scores - scores[finite].max())
        out.append(weights / weights.sum())
    return out


@unittest.skipUnless(HAVE_TORCH, "needs torch and the native library")
class PriorForwardIsBitIdenticalTests(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        root = export_prior(Path(self._t.name))
        self.net = combat_prior.PolicyNet(root / combat_prior.POLICY)

    def tearDown(self):
        self._t.cleanup()

    def test_policy_logits_are_the_forward_s_second_output_bit_for_bit(self):
        rng = np.random.default_rng(3)
        for n in (8, 15, 24, 40, 61):
            batch = collate([_Tok(n, rng)], torch.device("cpu"))
            with torch.inference_mode():
                self.assertTrue(torch.equal(self.net.model(**batch)[1], self.net.model.policy_logits(**batch)))

    def test_weights_equal_the_padded_batch_path_exactly_for_one_position_and_for_a_batch(self):
        rng = np.random.default_rng(5)
        toks = [_Tok(int(n), rng) for n in rng.integers(8, 70, 12)]
        legals = [[int(a) for a in np.arange(1, 7)[rng.random(6) < 0.7]] or [1] for _ in toks]
        for i in range(len(toks)):  # the search's case: one position per call
            got = self.net.weights([toks[i]], [legals[i]])[0]
            self.assertTrue(np.array_equal(got, _reference_weights(self.net, [toks[i]], [legals[i]])[0]), i)
        got = self.net.weights(toks[:5], legals[:5])  # a batch still takes the padded path
        for g, w in zip(got, _reference_weights(self.net, toks[:5], legals[:5]), strict=True):
            self.assertTrue(np.array_equal(g, w))


@unittest.skipUnless(HAVE_TORCH and (ROOT / "out" / "Sts2Emulator.dll").exists() or os.path.exists(os.path.join(os.environ["STS2_LIB_PATH"], "Sts2Emulator.dll")),
                     "needs the native library")
class SearchIsUnchangedByTheLeanPriorTests(unittest.TestCase):
    def test_same_visits_and_chosen_actions_as_the_padded_full_forward_on_fixed_seeds(self):
        import live_search
        from run_train import engine

        with tempfile.TemporaryDirectory() as tmp:
            root_dir = export_prior(Path(tmp), prior_min_visits=1)
            settings = combat_prior.find(root_dir).live_settings()
            lean = settings["prior_fn"]
            ref_net = combat_prior.PolicyNet(root_dir / combat_prior.POLICY)

            def reference(handle, legal):  # the same LivePrior with the pre-A2c weights()
                with mock.patch.object(combat_prior, "network", lambda _p: type("N", (), {"weights": staticmethod(
                        lambda t, l: _reference_weights(ref_net, t, l))})()):
                    return lenient(handle, legal)

            lenient = combat_prior.LivePrior(str(root_dir / combat_prior.POLICY))
            checked = 0
            for seed in ("SJ9DHBJHPE", "A2CSEARCH1"):
                s = engine.RunSession(seed, 10, engine.CombatBudget(30, 30))
                try:
                    rules = gen_loop.Agent().macro_policy()
                    rng = np.random.default_rng(0)
                    while s.phase != engine.rc.PHASE_COMBAT or s.decision() is not None:
                        d = s.decision()
                        s.apply(d.options[rules.pick(s.view(), d, rng).index])
                    root = s.combat(fork_combat=True).combat_fork(None)
                    try:
                        kw = dict(seed=7, max_sims=48, time_budget_s=120.0, min_child_sims=1, c=0.2, confirm_pairs=0,
                                  c_puct=settings["c_puct"], prior_min_visits=1)
                        a = live_search.search_live(root.handle, prior_fn=lean, **kw)
                        b = live_search.search_live(root.handle, prior_fn=reference, **kw)
                    finally:
                        root.close() if hasattr(root, "close") else None
                finally:
                    s.close()
                self.assertEqual(a.best, b.best)
                self.assertEqual(a.children, b.children)
                self.assertEqual((a.sims, a.line), (b.sims, b.line))
                checked += len(a.children)
            self.assertGreater(checked, 2)


if __name__ == "__main__":
    unittest.main()
