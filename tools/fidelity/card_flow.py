"""Which game models can put a card into a fight, or take one out: the table behind card conservation.

A card appears mid-fight only through a few game commands (a generated card added to a pile, a
status card added by a monster). `build` reads the local decompile and writes, per model class
(card, power, relic, monster, potion), the card classes it names and whether it can create
"open" cards (a clone or a pool draw). The table holds short ids only; never game code. The
decompile stays outside the repo.

    python -m tools.fidelity.card_flow build --decompiled D:/spireline/referee/decompiled \
        --out data/card_flow.json

`card_types` reads the card type of every game card id from the emulator's generated table
(src/Sts2Emulator/Generated/Cards.g.cs), so the conservation check can tell a Power (which leaves
the piles when played) from the rest.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .fight_compare import snake

KINDS = {
    "MegaCrit.Sts2.Core.Models.Cards": "card",
    "MegaCrit.Sts2.Core.Models.Powers": "power",
    "MegaCrit.Sts2.Core.Models.Relics": "relic",
    "MegaCrit.Sts2.Core.Models.Monsters": "monster",
    "MegaCrit.Sts2.Core.Models.Potions": "potion",
    "MegaCrit.Sts2.Core.Models.Enchantments": "enchantment",
    "MegaCrit.Sts2.Core.Models.Afflictions": "affliction",
}
# The commands that make a card that did not exist in the fight a moment ago.
CREATE = re.compile(r"AddGeneratedCardsToCombat|AddGeneratedCardToCombat|AddToCombatAndPreview|CreateInHand")
OPEN = re.compile(r"CreateClone|CardFactory|CardGeneration|GetRandom")
# Forging makes a Sovereign Blade when the player has none (ForgeCmd.Forge), so whoever forges can create one.
FORGE = re.compile(r"ForgeCmd\.Forge|Forge\(")
# Powers a creature can hold several of (one per source), and powers whose amount may fall below zero.
INSTANCED = re.compile(r"InstanceType\s*=>\s*PowerInstanceType\.Instanced")
ALLOW_NEGATIVE = re.compile(r"AllowNegative\s*=>\s*true")
# What takes a card out of the fight without it being played.
REMOVE = re.compile(r"RemoveFromCombat")


def _model_id(kind: str, cls: str) -> str:
    n = snake(cls)
    return n[: -len("_POWER")] if kind == "power" and n.endswith("_POWER") else n


def build(decompiled: Path) -> dict:
    classes = {}
    for sub, kind in KINDS.items():
        d = decompiled / sub
        if d.is_dir():
            for f in d.glob("*.cs"):
                classes.setdefault(kind, {})[f.stem] = f.read_text(encoding="utf-8", errors="replace")
    card_names = set(classes.get("card", {}))
    table: dict = {"creators": {}, "removers": {}, "powers": {"instanced": [], "allow_negative": []}}
    for kind, files in classes.items():
        for cls, text in files.items():
            body = re.sub(r"//.*", "", text)
            mid = _model_id(kind, cls)
            if kind == "power":
                if INSTANCED.search(body):
                    table["powers"]["instanced"].append(mid)
                if ALLOW_NEGATIVE.search(body):
                    table["powers"]["allow_negative"].append(mid)
            if REMOVE.search(body):
                table["removers"].setdefault(kind, []).append(mid)
            forges = kind != "card" or cls != "SovereignBlade"
            forges = forges and bool(FORGE.search(body))
            if not CREATE.search(body) and not forges:
                continue
            named = sorted({_model_id("card", c) for c in re.findall(r"\b([A-Z][A-Za-z0-9]+)\b", body)
                            if c in card_names and c != cls})
            if forges:
                named = sorted(set(named) | {"SOVEREIGN_BLADE"})
            table["creators"].setdefault(kind, {})[mid] = {"named": named, "open": bool(OPEN.search(body))}
    for kind in table["removers"]:
        table["removers"][kind].sort()
    for k in table["powers"]:
        table["powers"][k].sort()
    return table


def card_types(cards_g_cs: Path) -> dict:
    """Game card id (UPPER_SNAKE entry) -> {"type", "rarity"} from the emulator's generated table."""
    out = {}
    pat = re.compile(r'Entry: "([A-Z0-9_]+)".*?Type: CardType\.(\w+), Rarity: CardRarity\.(\w+)')
    for line in cards_g_cs.read_text(encoding="utf-8").splitlines():
        m = pat.search(line)
        if m:
            out[m.group(1)] = {"type": m.group(2), "rarity": m.group(3)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--decompiled", required=True)
    ap.add_argument("--out", default="data/card_flow.json")
    a = ap.parse_args()
    table = build(Path(a.decompiled))
    Path(a.out).write_text(json.dumps(table, indent=1, sort_keys=True), encoding="utf-8")
    print({k: {kk: len(vv) for kk, vv in v.items()} for k, v in table.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
