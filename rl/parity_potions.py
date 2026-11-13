"""Block P4 of the information-parity side blocks: what each potion does (D59 lane D, `docs/rl/HUMAN_PRIOR.md` "v4").

A human reads a potion's text at a reward, a shop and in the belt; the run encoder holds the belt as potion ids (block G of the base vector counts them by id). This module gives every potion
id an effect-tag vector in the vocabulary of `rl/parity_relics.py` (energy, cards, block, Strength, ... as the amount one drink does), flags (can be drunk out of a fight, fires by itself,
needs a target, opens a card choice, rolls something) and the potion's rarity. `rl/parity_data/potion_effects.txt` is the table, read off the engine's potion arms; the immediate energy, block,
Strength, Dexterity and HP of a drink are checked against the engine by drinking each potion in a fight (`measure_drink`, `tests/python/test_parity_potions.py`).

The drop odds the brief also names ("the potion odds of this act") are **not** here: the engine keeps them in a run-state field (`PotionRewardOdds`: 40 in 100, a step of 10 a fight, an elite's bonus)
that no state list exports, so a Python-only change cannot read them; the one additive state list that would is described in `PARITY.md` section 4.
"""

from __future__ import annotations

import re
import sys
from functools import cache
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT / "rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from parity_relics import EFFECTS, SCALES  # noqa: E402

FLAGS = ("any_time", "automatic", "targets_enemy", "chooses", "random")
RARITIES = ("Common", "Uncommon", "Rare", "None")
POTION_TAG_WIDTH = len(EFFECTS) + len(FLAGS) + len(RARITIES)
TABLE_FILE = Path(__file__).resolve().parent / "parity_data" / "potion_effects.txt"
POTIONS_G = _ROOT / "src" / "Sts2Emulator" / "Generated" / "Potions.g.cs"
# potions the immediate-drink check leaves out, with the reason: the table counts what the potion gives over its turns (or needs something the test fight lacks), a drink shows the first part
DRINK_CHECK_EXEMPT = {
    "RadiantTincture": "one energy now and one at each of the next three turns' starts",
    "Fortifier": "triples the block held: nothing to triple at the first turn",
    "HeartOfIron": "plating: the block arrives at each turn's end",
    "MazalethsGift": "a ritual: the Strength arrives at each turn's start",
    "RegenPotion": "heals at each turn's end",
    "ShipInABottle": "ten block now and ten at the next turn",
    "StarPotion": "stars, a Regent resource",
    "FlexPotion": "temporary Strength: taken back at the turn's end",
    "SpeedPotion": "temporary Dexterity: taken back at the turn's end",
}


@cache
def _name_to_id() -> dict[str, int]:
    from sts2_gym import names  # noqa: PLC0415

    return {name: pid for pid, name in names.potions().items()}


@cache
def potion_rows() -> dict[int, tuple[dict[str, float], tuple[str, ...]]]:
    """potion id -> ({effect: amount}, (flag, ...)) from `rl/parity_data/potion_effects.txt`; a name the engine's table lacks is an error."""
    by_name = _name_to_id()
    rows: dict[int, tuple[dict[str, float], tuple[str, ...]]] = {}
    for n, line in enumerate(TABLE_FILE.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            raise ValueError(f"{TABLE_FILE.name}:{n}: expected `Name | effects | flags`")
        name, eff, flg = parts
        if name not in by_name:
            raise ValueError(f"{TABLE_FILE.name}:{n}: {name} is not a potion of the engine's table")
        effects: dict[str, float] = {}
        for item in eff.split():
            key, _, amount = item.partition("=")
            if key not in SCALES:
                raise ValueError(f"{TABLE_FILE.name}:{n}: unknown effect {key!r}")
            effects[key] = float(amount) if "." in amount else int(amount)
        flags = tuple(flg.split())
        bad = [f for f in flags if f not in FLAGS]
        if bad:
            raise ValueError(f"{TABLE_FILE.name}:{n}: unknown flag {bad[0]!r}")
        if by_name[name] in rows:
            raise ValueError(f"{TABLE_FILE.name}:{n}: {name} twice")
        rows[by_name[name]] = (effects, flags)
    return rows


@cache
def potion_rarities() -> dict[int, str]:
    text = POTIONS_G.read_text(encoding="utf-8")
    out = {int(i): r for i, r in re.findall(r"new PotionDef\(Id: (\d+), Name: \"\w+\", Rarity: PotionRarity\.(\w+)", text)}
    if len(out) != len(_name_to_id()):
        raise ValueError("Potions.g.cs: a potion without a rarity")
    return out


@cache
def potion_tag_matrix() -> np.ndarray:
    """float32 (64, POTION_TAG_WIDTH): row p is `potion_tags(p)`; row 0 (no potion) is zero."""
    top = max(_name_to_id().values())
    m = np.zeros((top + 1, POTION_TAG_WIDTH), dtype=np.float32)
    rar = potion_rarities()
    for pid, (effects, flags) in potion_rows().items():
        for k, v in effects.items():
            m[pid, EFFECTS.index(k)] = max(-1.0, min(1.0, v / SCALES[k]))
        for f in flags:
            m[pid, len(EFFECTS) + FLAGS.index(f)] = 1.0
        r = rar.get(pid, "None")
        m[pid, len(EFFECTS) + len(FLAGS) + RARITIES.index(r if r in RARITIES else "None")] = 1.0
    m.setflags(write=False)
    return m


def potion_tags(potion_id: int) -> np.ndarray:
    m = potion_tag_matrix()
    if not 0 <= int(potion_id) < len(m):
        return np.zeros(POTION_TAG_WIDTH, dtype=np.float32)
    return m[int(potion_id)].copy()


def held_tags(potions) -> np.ndarray:  # noqa: ANN001
    """The sum of the tag vectors of the potions held (ids; 0 is an empty slot)."""
    out = np.zeros(POTION_TAG_WIDTH, dtype=np.float32)
    m = potion_tag_matrix()
    for p in potions:
        if 0 < int(p) < len(m):
            out += m[int(p)]
    return out


# ---------------------------------------------------------------------------------------------------------- the engine's view of a drink


def measure_drink(potion_id: int, *, seed: str = "PARPOT01", encounter: int = 62) -> dict[str, int] | None:
    """Drink the potion in a fight's first turn (a run's fight with the starter deck, one enemy, the potion in slot 0) and return the change in energy, block, Strength, Dexterity, HP and max HP;
    None when the engine does not offer the drink (no legal potion action)."""
    import ctypes  # noqa: PLC0415

    import combat_tokens as ct  # noqa: PLC0415
    from run_train.engine import RunSession  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    def read(obs) -> dict[str, int]:  # noqa: ANN001
        o = list(obs)
        buffs = {}
        for slot in range(ct.MAX_PLAYER_BUFFS):
            b = ct.PLAYER_BUFF_OFFSET + slot * 2
            if o[b] or o[b + 1]:
                buffs[o[b]] = o[b + 1]
        hand = sum(1 for i in range(native._lib.Sts2_ObsMaxHand()) if o[native._lib.Sts2_ObsHandOffset() + i * native._lib.Sts2_ObsCardSlotSize()])  # noqa: SLF001
        return {"hp": o[0], "max_hp": o[1], "block": o[2], "energy": o[3], "strength": buffs.get(0, 0), "dexterity": buffs.get(1, 0), "hand": hand}

    s = RunSession(seed, 10)
    try:
        obs = s.env._run_obs_buf  # noqa: SLF001
        native.run_start_combat(s.handle, native.starting_deck(10), encounter, [], 64, 80, [int(potion_id)], 99, 0, obs)
        before = read(obs)
        action = before["hand"] + 1  # card i, End Turn at the hand's size, a potion slot s at size + 1 + s
        mask = native.run_action_mask(s.handle, 300)
        if not mask[action]:
            return None
        rew, term, trunc = (ctypes.c_float * 1)(), (ctypes.c_int * 1)(), (ctypes.c_int * 1)()
        if native.run_step(s.handle, action, 0, obs, rew, term, trunc) != 0:
            return None
        after = read(obs)
        return {k: after[k] - before[k] for k in before}
    finally:
        s.close()
