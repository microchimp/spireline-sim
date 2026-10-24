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


if __name__ == "__main__":
    unittest.main()
