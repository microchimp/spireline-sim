"""The equal-value harness's position list (docs/rebuild/FIGHT_SOLVER.md section 6, piece S3).

Writes tests/fixtures/solver/positions.jsonl: one seeded recipe per line, from which
EqualValueHarnessTests builds a small position and solves it with the reference solver and
with the pruned solver under every pruning set, holding them equal to the last bit (both
compute exact rationals). A recipe names its source fight (a corpus fixture's deck, relics
and potions from tests/fixtures/combat, or self-play's default fight), the engine (old, or
core2 with F08 and F10), the seeded random-policy steps played into the fight, the cut
(enemy HP capped, draw and discard piles trimmed), the cards mixed in (order readers,
screens, draws and rolls, so each kind is met), the steps played after the cut (which
reach open screens and mid-turn positions) and the turn horizon.

    python tools/solver/harness.py          # write the list
    python tools/solver/harness.py --check  # fail if the list on disk differs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "solver" / "positions.jsonl"

# Cards that draw, open a screen, read or place on the draw pile, or roll.
MIX = [
    [],
    ["PommelStrike"],
    ["BurningPact"],
    ["TrueGrit"],
    ["Havoc"],
    ["Headbutt"],
    ["Armaments"],
    ["SwordBoomerang"],
    ["InfernalBlade"],
    ["Warcry"],
    ["ShrugItOff", "Anger"],
    ["Cleave", "Thunderclap"],
]

COUNT = 400


def recipes() -> list[dict]:
    out = []
    for i in range(COUNT):
        out.append(
            {
                "id": i,
                "source": "corpus" if i % 2 == 0 else "selfplay",
                "engine": "old" if (i // 2) % 2 == 0 else "core2",
                "seed": 5000 + i,
                "setup": i // 4,
                "steps": (i * 7) % 9,
                "enemy_hp": [2, 5, 9, 14][i % 4],
                "draw_keep": (i * 3 + i // 7) % 5,
                "discard_keep": (i + i // 2) % 3,
                "mix": MIX[(i // 3) % len(MIX)],
                "after": (i // 5) % 3,
                "horizon": 1 if i % 8 == 3 else 0,
            }
        )
    return out


def render() -> str:
    return "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in recipes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = render()
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT.name} is stale: run tools/solver/harness.py", file=sys.stderr)
            return 1
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(recipes())} recipes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
