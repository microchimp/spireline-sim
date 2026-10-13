"""Record and replay combat traces: (config, actions) in, one canonical hash per step out.

A trace is the unit every fidelity check shares. The same (config, actions) can be replayed on
another emulator build (golden master for speed patches) or, later, on the referee running the
game's own code, and the first step whose hash differs is the bug.

Trace file: JSON Lines. Line 1 is the header {"kind": "trace", "config": {...}}; each further
line is one step {"i", "action", "legal", "neutral", "full", "rng"}; step 0 is the opening.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .canon import canonical, digest


@dataclass
class Config:
    seed: int
    encounter: int
    ascension: int = 10
    character: str = "Ironclad"
    extra_cards: tuple = (2, 8)  # (low, high) as play_net samples them
    relics: tuple = (0, 5)
    chooser_seed: int = 0
    max_steps: int = 300
    floor: int = 0
    extra: dict = field(default_factory=dict)


class Emu:
    """The emulator modules, imported from one checkout (STS2_EMU)."""

    def __init__(self, root: str) -> None:
        import sys  # noqa: PLC0415

        base = Path(root).resolve()
        for sub in ("src", "rl"):
            p = str(base / sub)
            if p not in sys.path:
                sys.path.insert(0, p)
        import combat_mcts as cm  # noqa: PLC0415
        from sts2_gym import native  # noqa: PLC0415
        from sts2_gym.env import ENCOUNTER_NAMES  # noqa: PLC0415

        self.root = base
        self.cm = cm
        self.native = native
        self.encounter_names = dict(ENCOUNTER_NAMES)
        self.id_map = str(base / "data" / "id_map.json")

    def open(self, cfg: Config):
        cm = self.cm
        extra = cm.sample_extra_cards(cfg.seed, cfg.extra_cards[0], cfg.extra_cards[1], cfg.character)
        relics = cm.sample_relics(cfg.seed, cfg.relics[0], cfg.relics[1], cfg.character)
        return cm.Combat.opening(cfg.seed, cfg.encounter, cfg.ascension, cfg.floor, extra, relics)

    def state(self, combat) -> dict:
        won = bool(combat.won()) if combat.done else None
        return canonical(combat.handle, self.native, self.id_map, done=combat.done, won=won)

    def step_record(self, i: int, action, legal, combat) -> dict:
        doc = self.state(combat)
        return {"i": i, "action": action, "legal": legal, "neutral": digest(doc["neutral"]),
                "full": digest(doc["full"]), "rng": doc["neutral"]["rng"]}


def choose(rng: random.Random, legal: list) -> int:
    return legal[rng.randrange(len(legal))]


def record(emu: Emu, cfg: Config, *, keep_states: bool = False) -> dict:
    """Play `cfg` with a seeded uniform chooser; return {"config", "steps", "states"?}."""
    rng = random.Random(cfg.chooser_seed)
    combat = emu.open(cfg)
    steps, states = [], []
    try:
        legal = combat.legal()
        steps.append(emu.step_record(0, None, legal, combat))
        if keep_states:
            states.append(emu.state(combat))
        for i in range(1, cfg.max_steps + 1):
            if combat.done or not legal:
                break
            action = choose(rng, legal)
            combat.step(action)
            legal = combat.legal() if not combat.done else []
            steps.append(emu.step_record(i, action, legal, combat))
            if keep_states:
                states.append(emu.state(combat))
    finally:
        combat.close()
    out = {"config": asdict(cfg), "steps": steps}
    if keep_states:
        out["states"] = states
    return out


def replay(emu: Emu, trace: dict, *, keep_states: bool = False) -> dict:
    """Play the recorded actions again on `emu`; same output shape as `record`."""
    cfg = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in trace["config"].items()})
    combat = emu.open(cfg)
    steps, states = [], []
    try:
        steps.append(emu.step_record(0, None, combat.legal(), combat))
        if keep_states:
            states.append(emu.state(combat))
        for rec in trace["steps"][1:]:
            if combat.done:
                break
            combat.step(rec["action"])
            legal = combat.legal() if not combat.done else []
            steps.append(emu.step_record(rec["i"], rec["action"], legal, combat))
            if keep_states:
                states.append(emu.state(combat))
    finally:
        combat.close()
    out = {"config": trace["config"], "steps": steps}
    if keep_states:
        out["states"] = states
    return out


def write(path: Path, trace: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "trace", "config": trace["config"]}) + "\n")
        for s in trace["steps"]:
            f.write(json.dumps(s) + "\n")


def read(path: Path) -> dict:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    head = json.loads(lines[0])
    return {"config": head["config"], "steps": [json.loads(x) for x in lines[1:] if x.strip()]}
