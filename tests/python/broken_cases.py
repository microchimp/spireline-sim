"""Shared fixtures of the checkers' broken-case tests (tests/python/test_*_broken.py).

A checker is trusted only while it is seen to fail: each test file gives one checker a faithful case,
which must pass, and deliberately broken ones, which must fail (by the exit status where it has one).
Here: a fight's game snapshot with something in every field the comparers read, the emulator's dump of
the same position, one perturbation per compared field and the emulator state left out on purpose.
"""

from __future__ import annotations

import copy
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import fight_compare as fc  # noqa: E402

ID_MAP = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))
NM = fc.names(ID_MAP)
CID, EID = ID_MAP["cards"], ID_MAP["enemies"]
STREAMS = ("Shuffle", "CombatTargets", "MonsterAi", "CombatCardSelection", "CombatCardGeneration",
           "CombatEnergyCosts", "CombatPotionGeneration", "CombatOrbGeneration", "Niche")


def run_main(fn, argv: list) -> tuple[int, str]:
    """A checker's main() with this command line; (exit status, what it printed)."""
    out = io.StringIO()
    with mock.patch.object(sys, "argv", ["checker", *argv]), redirect_stdout(out):
        code = fn()
    return code, out.getvalue()


def write_json(path: Path, doc) -> str:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------------------ fight states

def game_state() -> dict:
    """A game snapshot with something in every field the comparers read."""
    return {"in_combat": True, "round": 2,
            "player": {"hp": 61, "max_hp": 80, "block": 5, "energy": 2, "powers": [["STRENGTH_POWER", 2]]},
            "hand": [["STRIKE_IRONCLAD", False], ["BASH", True]], "draw": [["DEFEND_IRONCLAD", False], ["ANGER", False]],
            "discard": [["DEFEND_IRONCLAD", False]], "exhaust": [["BASH", False]],
            "enemies": [{"name": "CORPSE_SLUG", "hp": 20, "max_hp": 27, "block": 3, "powers": [["WEAK_POWER", 1]],
                         "intents": [["Attack", 6, 1]]}],
            "rng": {s: {"seed": 7, "counter": 3 + i} for i, s in enumerate(STREAMS)},
            "potions": ["FIRE_POTION", None]}


def emu_state() -> dict:
    """The emulator's dump of the same position (EmuDump fight)."""
    def card(name, up):
        return {"def_id": CID[name], "upgraded": up}
    return {"in_combat": True, "run_phase": "Combat",
            "snapshot": {"player": {"hp": 61, "max_hp": 80, "block": 5, "energy": 2,
                                    "buffs": [{"name": "Strength", "amount": 2}, {"name": "Stunned", "amount": 1}]},
                         "piles": {"hand": [card("StrikeIronclad", False), card("Bash", True)],
                                   "discard": [card("DefendIronclad", False)], "exhaust": [card("Bash", False)]},
                         "enemies": [{"def_id": EID["CorpseSlug"], "hp": 20, "block": 3, "buffs": [{"name": "Weak", "amount": 1}],
                                      "intent": {"type": 0, "magnitude": 6, "hits": 1, "carries_damage": False,
                                                 "secondary": None}}]},
            "draw_order": [[CID["DefendIronclad"], False], [CID["Anger"], False]],
            "rng": {s: 3 + i for i, s in enumerate(STREAMS)},
            "potions": ["FIRE_POTION", None]}


def fight_pair(n: int = 3, seed: str = "BROKENCASE") -> tuple[dict, dict]:
    g = {"seed": seed, "ascension": 10, "steps": [{"action": None if j == 0 else {"kind": "end_turn"}, "state": game_state()}
                                                   for j in range(n)]}
    e = {"seed": seed, "steps": [{"action": s["action"], "state": emu_state()} for s in g["steps"]]}
    return g, e


def snap(e: dict) -> dict:
    return e["state"]["snapshot"]


# One perturbation per field the fight comparison covers: the emulator's dump with that field wrong.
PERTURB = {
    "player.hp": lambda st: st["snapshot"]["player"].__setitem__("hp", 60),
    "player.max_hp": lambda st: st["snapshot"]["player"].__setitem__("max_hp", 81),
    "player.block": lambda st: st["snapshot"]["player"].__setitem__("block", 0),
    "player.energy": lambda st: st["snapshot"]["player"].__setitem__("energy", 3),
    "player.power amount": lambda st: st["snapshot"]["player"]["buffs"][0].__setitem__("amount", 3),
    "player.power extra": lambda st: st["snapshot"]["player"]["buffs"].append({"name": "Vulnerable", "amount": 1}),
    "player.power gone": lambda st: st["snapshot"]["player"]["buffs"].pop(0),
    "hand card": lambda st: st["snapshot"]["piles"]["hand"][0].__setitem__("def_id", CID["Anger"]),
    "hand upgrade": lambda st: st["snapshot"]["piles"]["hand"][1].__setitem__("upgraded", False),
    "hand order": lambda st: st["snapshot"]["piles"]["hand"].reverse(),
    "hand extra": lambda st: st["snapshot"]["piles"]["hand"].append({"def_id": CID["Anger"], "upgraded": False}),
    "draw order": lambda st: st["draw_order"].reverse(),
    "draw upgrade": lambda st: st["draw_order"][0].__setitem__(1, True),
    "discard": lambda st: st["snapshot"]["piles"]["discard"].clear(),
    "exhaust": lambda st: st["snapshot"]["piles"]["exhaust"][0].__setitem__("upgraded", True),
    "enemy hp": lambda st: st["snapshot"]["enemies"][0].__setitem__("hp", 19),
    "enemy block": lambda st: st["snapshot"]["enemies"][0].__setitem__("block", 0),
    "enemy power": lambda st: st["snapshot"]["enemies"][0]["buffs"][0].__setitem__("amount", 2),
    "enemy kind": lambda st: st["snapshot"]["enemies"][0].__setitem__("def_id", EID["Seapunk"]),
    "enemy extra": lambda st: st["snapshot"]["enemies"].append(copy.deepcopy(st["snapshot"]["enemies"][0])),
    "enemy dead": lambda st: st["snapshot"]["enemies"][0].__setitem__("hp", 0),
    "intent kind": lambda st: st["snapshot"]["enemies"][0]["intent"].__setitem__("type", 1),
    "intent damage": lambda st: st["snapshot"]["enemies"][0]["intent"].__setitem__("magnitude", 7),
    "intent hits": lambda st: st["snapshot"]["enemies"][0]["intent"].__setitem__("hits", 2),
    "potion belt": lambda st: st["potions"].reverse(),
    "fight over": lambda st: st.__setitem__("in_combat", False),
    **{f"rng.{s}": (lambda s_: lambda st: st["rng"].__setitem__(s_, st["rng"][s_] + 1))(s) for s in STREAMS},
}
# Emulator state the comparison leaves out on purpose (fight_compare.NOT_POWERS, the dump's explanation
# fields): changing it must NOT make a difference, so the list of what is skipped stays explicit.
NOT_COMPARED = {
    "stun buff (a move in the game)": lambda st: st["snapshot"]["player"]["buffs"].pop(1),
    "move position": lambda st: st["snapshot"]["enemies"][0]["intent"].__setitem__("move_index", 9),
    "run phase": lambda st: st.__setitem__("run_phase", "Over"),
}
