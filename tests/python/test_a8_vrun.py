"""A8 (D33): the promotion arms play each agent's own macro; the out-of-fold keep test; the shipped seed fit."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))

import gen_loop  # noqa: E402
from run_train import keep  # noqa: E402


def write_agent(d: Path, generation: int, **meta) -> Path:
    d.mkdir(parents=True)
    (d / "model.json").write_text(json.dumps({"policy_sha256": f"sha{generation}"}))
    (d / "combat_policy.pt").write_bytes(b"x")
    (d / "agent.json").write_text(json.dumps({"generation": generation, **meta}))
    return d


def fresh_npz(path: Path) -> Path:
    import torch  # noqa: PLC0415

    from run_train import models  # noqa: PLC0415

    torch.manual_seed(3)
    path.write_bytes(Path(models.save(path.parent / f"{path.stem}.pt", models.FightNet(), models.RunValueNet(), {"generation": 4})).read_bytes())
    return path


class PromotionArmsTests(unittest.TestCase):
    """`promote` (what `gen0.py promote` runs) builds the candidate arm from its agent.json and the base arm from generation 3's."""

    def test_the_candidate_arm_plays_lookahead_and_the_base_arm_the_rules_in_the_same_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agents = root / "agent"
            cand = write_agent(agents / "gen4" / "candidate", 4, macro="vrun")
            fresh_npz(cand / "run_value.npz")
            gen3 = write_agent(agents / "gen3", 3)  # the incumbent: no macro key
            (agents / "current.json").write_text(json.dumps({"generation": 3, "dir": str(gen3)}))
            seeds = [f"S{i:09d}" for i in range(20)]
            seed_file = root / "seeds" / "Etest-1.json"
            seed_file.parent.mkdir()
            seed_file.write_text(json.dumps({"seeds": seeds}))
            played = []

            def fake_run_arm(seeds_, arm_dir, agent, config, procs, ascension=10):
                played.append((Path(arm_dir).parts[-3], agent))
                return [{"seed": s, "floor": 5 + i % 3, "win": False, "end": "death", "decisions": 10, "wall": 1.0, "crash": False}
                        for i, s in enumerate(seeds_)]

            with mock.patch.object(gen_loop, "run_arm", fake_run_arm):
                gen_loop.promote(cand, None, seed_file, agent_root=agents, boot=200)
            self.assertEqual([name for name, _ in played], ["candidate", "gen3"])
            (_, cand_agent), (_, base_agent) = played
            self.assertEqual((cand_agent.generation, cand_agent.macro, base_agent.generation, base_agent.macro), (4, "vrun", 3, "rules"))
            self.assertEqual(type(cand_agent.macro_policy()).__name__, "LookaheadPolicy")
            self.assertEqual(type(base_agent.macro_policy()).__name__, "RulesPlayer")


if __name__ == "__main__":
    unittest.main()
