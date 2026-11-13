"""Block P1 of the information-parity side blocks: what each relic does (D59 lane D, `docs/rl/HUMAN_PRIOR.md` "v4").

A human reads a relic's text; the run encoder holds it as one id (a 296-wide one-hot, learned from self-play where a rare relic is seen a few times). This
module gives every relic id an **effect-tag vector** so the net can read what a relic does and generalise across relics: the amounts of energy, cards, block,
Strength, Dexterity, healing, max HP, gold, potion slots, card rewards and the rest (`EFFECTS`), when it acts (`TRIGGERS`) and its rarity (`RARITIES`).
`rl/parity_data/relic_effects.txt` is the table, one row per relic of the fork engine's table, read off the engine's own relic hooks and pickup arms; the effects
an engine fight shows at its first turn are checked against the engine by playing (`measure_opening`, `tests/python/test_parity_relics.py`).

A side block like M and G: `BASE_SIZE` 1855 and the feature version are untouched. The block's two halves are `held_tags` (the sum over the relics held after the
option) and the option's own relic (`relic_tags`); `rl/parity_blocks.py` writes them.
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

EFFECTS = (
    "energy", "draw", "block", "strength", "dexterity", "heal", "max_hp", "gold", "potion_slot", "potion", "card_reward", "rest", "shop", "curse", "damage",
    "debuff", "upgrade", "remove", "add_card", "hp_cost", "relic", "mitigate", "junk", "drawback", "map",
)
TRIGGERS = ("pickup", "combat_start", "turn_start", "turn_end", "card_play", "damage", "kill", "rest_site", "shop", "reward", "passive", "conditional")
RARITIES = ("Starter", "Common", "Uncommon", "Rare", "Shop", "Event", "Ancient", "None")
# the unit an effect's amount is divided by (then clipped to [-1, 1]): a relic's number in the table is the game's own where it states one
SCALES = {
    "energy": 3.0, "draw": 5.0, "block": 20.0, "strength": 3.0, "dexterity": 3.0, "heal": 25.0, "max_hp": 20.0, "gold": 300.0, "potion_slot": 3.0, "potion": 3.0,
    "card_reward": 3.0, "rest": 1.0, "shop": 1.0, "curse": 2.0, "damage": 10.0, "debuff": 4.0, "upgrade": 4.0, "remove": 4.0, "add_card": 3.0, "hp_cost": 15.0,
    "relic": 2.0, "mitigate": 1.0, "junk": 3.0, "drawback": 1.0, "map": 1.0,
}
RELIC_TAG_WIDTH = len(EFFECTS) + len(TRIGGERS) + len(RARITIES)
TABLE_FILE = Path(__file__).resolve().parent / "parity_data" / "relic_effects.txt"
RELICS_G = _ROOT / "src" / "Sts2Emulator" / "Generated" / "Relics.g.cs"
# relics the first-turn measurement test skips (the engine's first turn does not show what the table says: a character the test fight does not play, an elite-only
# or rest-armed effect, a count the opening does not reach). A relic is listed with the reason.
FIRST_TURN_CHECKED: dict[str, str] = {
    "DivineDestiny": "stars, a Regent resource the Ironclad test fight does not show as energy",
    "DivineRight": "stars, a Regent resource the Ironclad test fight does not show as energy",
    "Gorget": "plating: the block arrives at the turn's end",
    "WhisperingEarring": "its energy is spent at once by the thirteen cards it plays on the first turn",
}


@cache
def _name_to_id() -> dict[str, int]:
    from sts2_gym import names  # noqa: PLC0415

    return {name: rid for rid, name in names.relics().items()}


@cache
def relic_rows() -> dict[int, tuple[dict[str, float], tuple[str, ...]]]:
    """relic id -> ({effect: amount}, (trigger, ...)), read from `rl/parity_data/relic_effects.txt`. A name the engine's table lacks is an error."""
    by_name = _name_to_id()
    rows: dict[int, tuple[dict[str, float], tuple[str, ...]]] = {}
    for n, line in enumerate(TABLE_FILE.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            raise ValueError(f"{TABLE_FILE.name}:{n}: expected `Name | effects | triggers`")
        name, eff, trig = parts
        if name not in by_name:
            raise ValueError(f"{TABLE_FILE.name}:{n}: {name} is not a relic of the engine's table")
        effects: dict[str, float] = {}
        for item in eff.split():
            key, _, amount = item.partition("=")
            if key not in SCALES:
                raise ValueError(f"{TABLE_FILE.name}:{n}: unknown effect {key!r}")
            effects[key] = float(amount) if "." in amount else int(amount)
        triggers = tuple(trig.split())
        if by_name[name] in rows:
            raise ValueError(f"{TABLE_FILE.name}:{n}: {name} twice")
        rows[by_name[name]] = (effects, triggers)
    return rows


@cache
def relic_rarities() -> dict[int, str]:
    """relic id -> rarity name, from the engine's generated relic table (`Relics.g.cs`)."""
    text = RELICS_G.read_text(encoding="utf-8")
    out = {int(i): r for i, r in re.findall(r"new RelicDef\(Id: (\d+), Name: \"\w+\", Entry: \"\w+\", Rarity: RelicRarity\.(\w+)", text)}
    if len(out) != len(_name_to_id()):
        raise ValueError("Relics.g.cs: a relic without a rarity")
    return out


@cache
def relic_tag_matrix() -> np.ndarray:
    """float32 (297, RELIC_TAG_WIDTH): row r is `relic_tags(r)`; row 0 (no relic) is zero."""
    top = max(_name_to_id().values())
    m = np.zeros((top + 1, RELIC_TAG_WIDTH), dtype=np.float32)
    rar = relic_rarities()
    for rid, (effects, triggers) in relic_rows().items():
        for k, v in effects.items():
            m[rid, EFFECTS.index(k)] = max(-1.0, min(1.0, v / SCALES[k]))
        for t in triggers:
            m[rid, len(EFFECTS) + TRIGGERS.index(t)] = 1.0
        r = rar.get(rid, "None")
        m[rid, len(EFFECTS) + len(TRIGGERS) + RARITIES.index(r if r in RARITIES else "None")] = 1.0
    m.setflags(write=False)
    return m


def relic_tags(relic_id: int) -> np.ndarray:
    """The tag vector of one relic (float32, RELIC_TAG_WIDTH); an id the table does not hold is zero."""
    m = relic_tag_matrix()
    if not 0 <= int(relic_id) < len(m):
        return np.zeros(RELIC_TAG_WIDTH, dtype=np.float32)
    return m[int(relic_id)].copy()


def held_tags(relics) -> np.ndarray:  # noqa: ANN001
    """The sum of the tag vectors of the relics held (a multiset of ids)."""
    out = np.zeros(RELIC_TAG_WIDTH, dtype=np.float32)
    m = relic_tag_matrix()
    for r in relics:
        if 0 <= int(r) < len(m):
            out += m[int(r)]
    return out


# ---------------------------------------------------------------------------------------------------------- the engine's view of a relic's first turn


def measure_opening(relic_id: int | None, *, seed: int = 1, encounter: int = 62, ascension: int = 10) -> dict[str, int]:
    """A fight at its first turn with `relic_id` held (None: no relic): the player's energy, hand size, block, Strength and Dexterity read off the engine. The test fight is
    the card-tag probe's (one enemy, no block, a plain starter deck), so the difference to the same fight with no relic is what the relic does at the opening."""
    import combat_tokens as ct  # noqa: PLC0415
    from combat_mcts import Combat  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    c = Combat.opening(seed, encounter, ascension, 1, relics=[] if relic_id is None else [int(relic_id)])
    try:
        obs = list(c.obs)
        buffs = {}
        for slot in range(ct.MAX_PLAYER_BUFFS):
            base = ct.PLAYER_BUFF_OFFSET + slot * 2
            if obs[base] == 0 and obs[base + 1] == 0:
                continue
            buffs[obs[base]] = obs[base + 1]
        return {"energy": int(obs[3]), "hand": len(native.get_pile(c.handle, "hand")), "block": int(obs[2]), "hp": int(obs[0]), "max_hp": int(obs[1]),
                "strength": int(buffs.get(0, 0)), "dexterity": int(buffs.get(1, 0))}
    finally:
        c.close()
