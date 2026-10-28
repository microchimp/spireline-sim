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

The horizon is drawn from its own hash of the recipe's index, apart from every other field,
so horizon-1 positions fall on both engines, both sources and every enemy HP. Three groups
follow the 400 mixed recipes:

- screens: a card that opens a screen is mixed in and played last (`play`), so the position
  stands on the open screen, draw-pile choices (Seeker Strike's sample, Cleanse, Wish) among
  them, which is what runs the solver's keep path;
- order-dependence (the owner's check): Hunter Killer once its TENDERIZING_GOOP has put Tender
  on the player (each play costs Strength and Dexterity until the turn ends, so Strike then
  Defend and Defend then Strike reach different states), and the corpus's Toadpoles fight once
  a Toadpole has Thorns (each attack into it hits back; the corpus files of that fight record no
  deck, so it is played with the starter deck);
  `encounter` names the fight, `until` the buff the player waits for, ending turns (`player:`
  or `enemy:`), `fixture` the corpus file whose deck it uses; enemy HP is capped at 30, so the
  turn cannot end the fight. L0 reads the player's HP only, so Tender's order shows in the
  value at horizon 1 (the enemy's attack meets the block), Thorns' within the turn;
- macro draw (FIGHT_SOLVER.md section 17): a five-card draw over six distinct values (mid-turn through Expertise, and
  the turn start's draw at horizon 1 with cards too dear to play twice), one crossing a reshuffle, and one per draw
  observer and per reason the macro draw switches off (`macro` names it), each on both engines: the position is set
  after the cut (`hand`, `draw_pile`, `discard_pile`, the player's `buff` as Name:amount, `energy`).

    python tools/solver/harness.py          # write the list
    python tools/solver/harness.py --check  # fail if the list on disk differs
"""

from __future__ import annotations

import argparse
import hashlib
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

# Cards played last to stand on their screen: draw-pile choices first, then hand and discard choices.
SCREEN_PLAYS = ["SeekerStrike", "Cleanse", "Wish", "BurningPact", "Headbutt", "Armaments", "TrueGrit"]

COUNT = 400
SCREEN_COUNT = 8  # recipes per screen card

# The owner's order-dependence positions: (encounter, until, corpus fixture or "" for the starter deck).
ORDER = [
    ("HunterKiller", "player:Tender", ""),
    ("Toadpoles", "enemy:Thorns", ""),
]
ORDER_COUNT = 6  # recipes per fight: both engines, three seeds

# The macro draw's positions: (macro, hand, draw pile, discard pile, buff, energy, horizon[, encounter]).
# off:rest-taken: Thieving Hopper's first move steals a card of the pile's best rarity tier (two Uncommons here) at the
# enemy turn, by its place in the pile; the next turn's five-card draw is then sequential (g7-emu N8ST89LTSW k0106-k0112).
MACRO_PILE = ["Anger", "Whirlwind", "Bash", "Uppercut", "Impervious", "Thunderclap"]
TURN_PILE = ["Bash", "Bludgeon", "Impervious", "Uppercut", "Barricade", "DemonForm"]
EXPERTISE = ["Expertise", "DefendIronclad"]
MACRO = [
    ("draw5-mid", EXPERTISE, MACRO_PILE, [], "", 1, 0),
    ("draw5-turn", ["DefendIronclad"], TURN_PILE, [], "", 0, 1),
    ("reshuffle", EXPERTISE, MACRO_PILE[:3], MACRO_PILE[3:], "", 1, 0),
    ("hand-ordered", EXPERTISE, MACRO_PILE, ["CalculatedGamble"], "", 1, 0),
    ("observer:Hellraiser", EXPERTISE, ["StrikeIronclad", *MACRO_PILE[:5]], [], "Hellraiser:1", 1, 0),
    ("observer:ChainsOfBinding", EXPERTISE, MACRO_PILE, [], "ChainsOfBinding:2", 1, 0),
    ("observer:Speedster", EXPERTISE, MACRO_PILE, [], "Speedster:3", 1, 0),
    ("observer:Iteration", EXPERTISE, ["Wound", *MACRO_PILE[:5]], [], "Iteration:1", 1, 0),
    ("observer:Pagestorm", EXPERTISE, ["Dazed", *MACRO_PILE[:5]], [], "Pagestorm:1", 1, 0),
    ("observer:Stratagem", EXPERTISE, MACRO_PILE, [], "StratagemPower:1", 1, 0),
    ("observer:Pillage", ["Pillage", "DefendIronclad"], MACRO_PILE, [], "", 1, 0),
    ("observer:Scrape", ["Scrape", "DefendIronclad"], MACRO_PILE, [], "", 1, 0),
    ("off:rest-taken", ["DefendIronclad"], ["Bash", "Bludgeon", "Impervious", "Uppercut", "Whirlwind", "Barricade", "DemonForm"], [], "", 0, 1,
     "ThievingHopper"),
]


def horizon(i: int) -> int:
    """1 for about one recipe in six, by a sha256 of the index alone."""
    return 1 if hashlib.sha256(f"horizon {i}".encode()).digest()[0] % 6 == 0 else 0


def base(i: int, **kw) -> dict:
    r = {
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
        "horizon": horizon(i),
        "play": "",
        "encounter": "",
        "until": "",
        "fixture": "",
        "macro": "",
        "hand": [],
        "draw_pile": [],
        "discard_pile": [],
        "buff": "",
        "energy": -1,
    }
    r.update(kw)
    return r


def recipes() -> list[dict]:
    out = [base(i) for i in range(COUNT)]
    i = COUNT
    for card in SCREEN_PLAYS:
        for j in range(SCREEN_COUNT):
            out.append(
                base(
                    i,
                    steps=j % 3,
                    enemy_hp=[9, 14][j % 2],
                    draw_keep=3 + j % 2,
                    discard_keep=1 + j % 2,
                    mix=[card],
                    after=0,
                    horizon=0,
                    play=card,
                )
            )
            i += 1
    for encounter, until, fixture in ORDER:
        for j in range(ORDER_COUNT):
            out.append(
                base(
                    i,
                    source="corpus" if fixture else "selfplay",
                    engine="old" if j % 2 == 0 else "core2",
                    steps=0,
                    enemy_hp=30,
                    draw_keep=1,
                    discard_keep=1,
                    mix=["StrikeIronclad", "DefendIronclad"],
                    after=0,
                    horizon=1 if encounter == "HunterKiller" and j >= ORDER_COUNT - 2 else 0,
                    encounter=encounter,
                    until=until,
                    fixture=fixture,
                )
            )
            i += 1
    for macro, hand, draw, discard, buff, energy, h, *enc in MACRO:
        for engine in ("old", "core2"):
            out.append(
                base(
                    i,
                    source="selfplay",
                    engine=engine,
                    steps=0,
                    enemy_hp=9,
                    draw_keep=0,
                    discard_keep=0,
                    mix=[],
                    after=0,
                    horizon=h,
                    macro=macro,
                    encounter=enc[0] if enc else "",
                    hand=hand,
                    draw_pile=draw,
                    discard_pile=discard,
                    buff=buff,
                    energy=energy,
                )
            )
            i += 1
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
