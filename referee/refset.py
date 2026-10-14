#!/usr/bin/env python3
"""Ironclad A10 reference set: one versioned set of game recordings that every accuracy number is quoted from.

    python -I referee/refset.py build   [--version v2] [--realistic 600] [--codex <runs.jsonl.gz>]
    python -I referee/refset.py record  [--version v2] [--subset all|pilot] [--parts realistic,coverage,regression]
                                        [--game D:/spireline/referee/game-refset] [--servers 3] [--label <name>]
                                        [--nodeless-pass none|a|b]   (one pass on the stripped-down game)
    python -I referee/refset.py rejudge [--version v2] --record <record dir>     (judge saved passes again)
    python -I referee/refset.py status  [--version v2]
    python -m tools.fidelity.refset_score --version v2 [--subset pilot] --emu <emulator checkout>   (the numbers)

Run from the fork's root, with -I: the Spire Codex history is third-party data, and -I keeps the script's
and the current directory off sys.path while it is read. Everything this writes is local, under
D:/spireline/referee/refset/<version>/ (manifest, recordings, scores), never in the repo: realistic specs
are derived from per-run data. referee/REFSET.md has the whole method.

The set (manifest.json, one entry per spec with the reason it is in the set, its chooser and its realism flags):
  realistic   fights from real Ironclad A10 runs (Spire Codex, v0.107.1): the run's own seed and Neow pick,
              then its deck (upgrades, enchantments, saved properties), relics (with their saved state rebuilt
              for that floor, STATE_RULES), potions, slots, HP and gold as they stood before that floor's fight,
              then that floor's encounter in its own act. Stratified by act, room type (normal/elite/boss) and
              the floor's place in the act; one fight per run; the heuristic chooser. The headline numbers.
  coverage    the fewest plausible setups that put every Ironclad and colourless card (base and upgraded) in
              a deck often enough to be played 5 times, meet every act 1-3 encounter 3 times, and hold every
              Ironclad-reachable relic and potion the emulator models 3 times (planned; refset_score counts
              what the recordings really reached). Random play.
  regression  one minimal setup per known divergence class (FINDINGS.md, triage files, the setup fights'
              shrunk cases), plus the original fight that found it, re-recorded with today's driver; each class
              names the trigger its fights must show.
  runs        whole runs: every Ironclad A10 codex run (the codex replay, run-level offers) and 30 seeded
              runs for the run referee (driver/RunPlay.cs).

Setup goes through referee/driver/Setup.cs and RefsetSetup.cs: the game's own paths only (the act command, the
save-load path for a run's holdings, the console's fight, relic, card, potion and power commands); never godmode,
kill, win, damage or heal shortcuts.

Recording (record): only a clean build of the checkout's committed driver with every REQUIRED_CAPS capability
records (checked on the DLL first and on every server's job-start line after). Every fight is recorded twice
(the second pass in reverse order, on fresh servers); a fight whose two recordings differ is excluded as an
unstable reference, as is any fight whose game side logged an error or whose setup the game could not build.
Each kept recording is stamped with the set version, the game build (sts2.dll sha256), the driver commit, DLL
hash and capabilities, and the tools commit. The safety checks are serve.py's (known sts2.dll, no network,
scratch profile, real profile untouched); a breach aborts the record.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

SET_NAME = "Ironclad A10 reference set"
REFSET = Path("D:/spireline/referee/refset")
CODEX = Path("D:/spireline/data/spire-codex/runs-v0.107.1.jsonl.gz")
EXPORT = Path("D:/spireline/referee/exports/v0.107.1.json")
JOBS = Path("D:/spireline/referee/jobs")
JOBS_SETUP = Path("D:/spireline/referee/jobs-setup")
GAME = "D:/spireline/referee/game-refset"
EMU_DLL = ROOT / "tools" / "emudump" / "bin" / "Release" / "net9.0" / "EmuDump.dll"
DRIVER_BUILD = HERE / "driver" / "bin" / "Release" / "net9.0"
ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
ACTS = ("OVERGROWTH", "UNDERDOCKS", "HIVE", "GLORY")
ACT_NUMBER = {"OVERGROWTH": 1, "UNDERDOCKS": 1, "HIVE": 2, "GLORY": 3}
KINDS = ("weak", "normal", "elite", "boss")
ROOM_OF_KIND = {"weak": "monster", "normal": "monster", "elite": "elite", "boss": "boss"}
MIRRORED_FOLLOW_UPS = {"None", "TransformSelect"}
PARTS = ("realistic", "coverage", "regression")
MAX_ACTIONS = 150

# Realistic quotas per act (x3 acts = 600): (room type, fights, floor buckets within the act).
REALISTIC_CELLS = (("monster", 110, 3), ("elite", 55, 2), ("boss", 35, 1))


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).upper()


def bare(x: str | None) -> str:
    return (x or "").split(".", 1)[1] if x and "." in x else (x or "")


def h(*parts) -> str:
    return hashlib.sha1(":".join(str(p) for p in parts).encode()).hexdigest()


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            d.update(b)
    return d.hexdigest()


def git_commit(path: Path) -> dict:
    def g(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            # Untracked files count: a build compiles every source file in the tree, tracked or not.
            "dirty": bool(g("status", "--porcelain", "--untracked-files=all"))}


def vdir(version: str) -> Path:
    return REFSET / version


# ---------------------------------------------------------------- tables: what the game has and the emulator models

class Tables:
    """The game's own content tables (the export) cut to what EmuDump can set up and mirror."""

    def __init__(self, export: dict, setupinfo: dict, id_map: dict, enchantments: set[str]):
        self.export = export
        self.follow = setupinfo.get("relics", {})
        self.cards_emu = {snake(k) for k in id_map["cards"]}
        self.potions_emu = {snake(k) for k in id_map["potions"]}
        self.enchant_emu = enchantments
        self.encounters = {}  # id -> (act, kind, index)
        for act in ACTS:
            for kind in KINDS:
                for i, e in enumerate(export["acts"][act][kind]):
                    self.encounters.setdefault(e, (act, kind, i))
        ci = export["character_info"]["IRONCLAD"]
        cp, rp, pp = export["card_pools"], export["relic_pools"], export["potion_pools"]
        legal = lambda c: c.get("multiplayer_constraint", "None") == "None"  # noqa: E731
        self.card_info = {c["id"]: c for pool in (ci["card_pool"], "ColorlessCardPool") for c in cp[pool]["all"]}
        self.cards = sorted(c for c, v in self.card_info.items()
                            if legal(v) and v["rarity"] in ("Basic", "Common", "Uncommon", "Rare", "Ancient") and c in self.cards_emu)
        self.upgradable = {c for c, v in self.card_info.items() if int(v.get("max_upgrade_level", "0") or 0) > 0}
        others = [c for c in export["character_info"] if c != "IRONCLAD"]
        self.other_cards = {c["id"] for o in others for c in cp[export["character_info"][o]["card_pool"]]["all"]}
        self.other_relics = {r["id"] for o in others for r in rp[export["character_info"][o]["relic_pool"]]["all"]}
        self.other_potions = {p["id"] for o in others for p in pp[export["character_info"][o]["potion_pool"]]["all"]}
        self.pool_relics = {r["id"]: r["rarity"] for pool in (ci["relic_pool"], "SharedRelicPool") for r in rp[pool]["all"]}
        self.event_relics = {r["id"] for r in rp["EventRelicPool"]["all"]}
        self.pool_potions = {p["id"] for pool in (ci["potion_pool"], "SharedPotionPool", "EventPotionPool") for p in pp[pool]["all"]}

    def relic_ok(self, r: str) -> bool:
        return self.follow.get(r) in MIRRORED_FOLLOW_UPS

    def card_ok(self, cid: str, upgrades: int, ench: str) -> bool:
        return cid in self.cards_emu and upgrades <= 1 and (not ench or ench in self.enchant_emu)


def emulator_enchantments() -> set[str]:
    """The emulator's Enchantment enum (src/Sts2Emulator/Core/Card.cs), in the game's naming."""
    src = (ROOT / "src" / "Sts2Emulator" / "Core" / "Card.cs").read_text(encoding="utf-8")
    body = re.search(r"public enum Enchantment\s*\{(.*?)\}", src, re.S).group(1)
    body = re.sub(r"//.*", "", body)
    return {snake(n) for n in re.findall(r"^\s*([A-Z]\w*)\s*(?:=\s*\d+)?\s*,", body, re.M)} - {"NONE"}


def load_tables(cache: Path) -> Tables:
    info_path = cache / "setupinfo.json"
    if not info_path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        out = subprocess.run(["dotnet", str(EMU_DLL), "setupinfo"], capture_output=True, text=True, check=True).stdout
        info_path.write_text(out, encoding="utf-8")
    export = json.loads(EXPORT.read_text(encoding="utf-8"))["results"][0]
    id_map = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))
    return Tables(export, json.loads(info_path.read_text(encoding="utf-8")), id_map, emulator_enchantments())


# ---------------------------------------------------------------- realistic: states from real runs (Spire Codex)

def codex_runs(path: Path, stats: collections.Counter, max_runs: int = 0):
    """Ironclad A10 single-player standard unmodded v0.107.1 runs; the record is data only."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if max_runs and stats["ironclad_a10_runs"] >= max_runs:
                break
            stats["lines"] += 1
            if '"CHARACTER.IRONCLAD"' not in line:
                continue
            if "BaseLib" in line or "save_dict_" in line or '"CardModel_' in line:
                stats["skip_modded"] += 1
                continue
            try:
                d = json.loads(line)
            except ValueError:
                stats["skip_unreadable"] += 1
                continue
            p = d.get("players") or []
            if (d.get("ascension") != 10 or d.get("game_mode") != "standard" or d.get("modifiers") or len(p) != 1
                    or p[0].get("character") != "CHARACTER.IRONCLAD" or d.get("build_id") != "v0.107.1"
                    or any(bare(a) not in ACTS for a in d.get("acts", []))):
                continue
            stats["ironclad_a10_runs"] += 1
            yield d


def _ck(c: dict, final: bool = False) -> list:
    """[id, upgrade level, enchantment, amount, floor added, saved properties, whether they are the run's end
    values] of a history card (the final deck's cards carry their [SavedProperty] values at the run's end)."""
    e = c.get("enchantment") or {}
    return [bare(c.get("id")), int(c.get("current_upgrade_level", 0) or 0), bare(e.get("id")), int(e.get("amount", 0) or 0),
            int(c.get("floor_added_to_deck") or 1), c.get("props") or {}, final]


# ---------------------------------------------------------------- saved state: the [SavedProperty] values before a floor

# A relic or card with saved properties ([SavedProperty]: a counter, a flag, or what its event picked) is put in
# place with them (the player_load item, the game's save-load path). The codex keeps each held item's values at the
# run's END ("props"); the value BEFORE floor F is had by one of these rules, from the floors between the item's own
# floor O and F (the item's effects on floor O came before it was held, or are part of the record):
#   fixed        set when the item was made (its event's pick, an act index, a skin) and never changed: the end value.
#   reset        reset as every room is entered, so a fight starts from this value.
#   monotone     only ever moves away from its fresh value: fresh at the end means fresh before F; else unknown.
#   inc / dec    +1 / -1 (floored at 0) per combat; inc_monster: per monster (not elite, boss) combat; elite:
#                +1 per elite won; turns: + the combat's turns, mod N;
#   rest         + n per rest-site choice C; cards: + each card gained or transformed into; tea_set: true after
#                a rest, false after a fight;
#   kindle       Pumpkin Candle: 5 at pickup, -1 per combat, +5 per KINDLE.
#   card_rewards + 1 per floor with a card reward, up to N (1: a flag); treasures: + 1 per treasure room;
#   act1_boss    true once the act 1 boss is won (Lava Rock's extra relics).
# A rebuilt counter is trusted ("verified") only when the same rule, run on to the run's end, gives the end value.
# Anything else (no rule, a failed check, an item that left the run before its end) FLAGS the spec: it is kept, and
# counted apart (state_unknown / state_unverified), never silently given a fresh value.
FIXED, RESET, MONOTONE = "fixed", "reset", "monotone"
STATE_RULES: dict[tuple[str, str], tuple] = {
    ("ARCHAIC_TOOTH", "StarterCard"): (FIXED,), ("ARCHAIC_TOOTH", "AncientCard"): (FIXED,),
    ("DUSTY_TOME", "AncientCard"): (FIXED,),
    ("TOUCH_OF_OROBAS", "StarterRelic"): (FIXED,), ("TOUCH_OF_OROBAS", "UpgradedRelic"): (FIXED,),
    ("SEA_GLASS", "CharacterId"): (FIXED,), ("BYRDPIP", "Skin"): (FIXED,), ("PAELS_LEGION", "Skin"): (FIXED,),
    ("GOLDEN_COMPASS", "GoldenPathAct"): (FIXED,),
    ("FUR_COAT", "FurCoatActIndex"): (FIXED,), ("FUR_COAT", "FurCoatCoordsSet"): (FIXED,),
    ("FUR_COAT", "FurCoatCoordCols"): (FIXED,), ("FUR_COAT", "FurCoatCoordRows"): (FIXED,),
    ("SPOILS_MAP", "SpoilsActIndex"): (FIXED,),
    ("MAD_SCIENCE", "TinkerTimeType"): (FIXED,), ("MAD_SCIENCE", "TinkerTimeRider"): (FIXED,),
    ("*", "IsWax"): (FIXED,), ("*", "IsMelted"): (MONOTONE,),
    ("LAVA_LAMP", "TookDamageThisCombat"): (RESET, False),
    ("BONE_TEA", "CombatsLeft"): ("dec", 1), ("TEA_OF_DISCOURTESY", "CombatsLeft"): ("dec", 1), ("EMBER_TEA", "CombatsLeft"): ("dec", 5),
    ("FISHING_ROD", "CombatsSeen"): ("inc_monster",), ("LASTING_CANDY", "CombatsSeen"): ("inc", None),
    ("TOY_BOX", "CombatsSeen"): ("inc", None), ("WONGOS_MYSTERY_TICKET", "CombatsFinished"): ("inc", None),
    ("GUILTY", "CombatsSeen"): ("inc", None), ("SWORD_OF_STONE", "ElitesDefeated"): ("elite",),
    ("HAPPY_FLOWER", "TurnsSeen"): ("turns", 3), ("FAKE_HAPPY_FLOWER", "TurnsSeen"): ("turns", 5),
    ("PENDULUM", "TurnsSeen"): ("turns", 3), ("POLLINOUS_CORE", "TurnsSeen"): ("turns", 4),
    ("GIRYA", "TimesLifted"): ("rest", "LIFT", 1), ("BOOK_OF_FIVE_RINGS", "CardsAdded"): ("cards",),
    ("VENERABLE_TEA_SET", "GainEnergyInNextCombat"): ("tea_set",), ("FAKE_VENERABLE_TEA_SET", "GainEnergyInNextCombat"): ("tea_set",),
    ("PUMPKIN_CANDLE", "KindleCount"): ("kindle",),
    ("LIZARD_TAIL", "WasUsed"): (MONOTONE,), ("LAVA_ROCK", "HasTriggered"): ("act1_boss",), ("MAW_BANK", "HasItemBeenBought"): (MONOTONE,),
    ("WONGOS_MYSTERY_TICKET", "GaveRelic"): (MONOTONE,),
    ("PEN_NIB", "AttacksPlayed"): (MONOTONE,), ("NUNCHAKU", "AttacksPlayed"): (MONOTONE,), ("IRON_CLUB", "CardsPlayed"): (MONOTONE,),
    ("JOSS_PAPER", "CardsExhausted"): (MONOTONE,), ("PAELS_WING", "RewardsSacrificed"): (MONOTONE,),
    ("WINGED_BOOTS", "TimesUsed"): (MONOTONE,),
    ("SILVER_CRUCIBLE", "TimesUsed"): ("card_rewards", 3), ("SILVER_CRUCIBLE", "TreasureRoomsEntered"): ("treasures",),
    ("SILKEN_TRESS", "IsUsed"): ("card_rewards", 1),
}
# Items with saved state: one held before F that left the run before its end has no end value to rebuild from.
STATEFUL = {item for item, _ in STATE_RULES if item != "*"} | {"TUNING_FORK", "PAELS_TOOTH", "GENETIC_ALGORITHM", "THE_SCYTHE"}
# The fresh value of a property saved under each kind.
PROP_FRESH = {"bools": False, "ints": 0}


def floor_events(pts: list, act_of: list[int] | None = None) -> list[dict]:
    """Per floor of a run (index k-1 for floor k): what the saved counters count; act_of the act index per floor."""
    out = []
    for k, pt in enumerate(pts):
        st = (pt.get("player_stats") or [{}])[0]
        combats = [r for r in pt.get("rooms") or [] if r.get("room_type") in ("monster", "elite", "boss")]
        out.append({"combats": len(combats), "elites": sum(r["room_type"] == "elite" for r in combats),
                    "monsters": sum(r["room_type"] == "monster" for r in combats),
                    "turns": sum(int(r.get("turns_taken") or 0) for r in combats),
                    "rests": [str(x) for x in st.get("rest_site_choices", [])], "rest_site": pt.get("map_point_type") == "rest_site",
                    "cards": len(st.get("cards_gained", [])) + len(st.get("cards_transformed", [])),
                    "card_reward": bool(st.get("card_choices")),
                    "treasure": any(r.get("room_type") == "treasure" for r in pt.get("rooms") or []),
                    "act1_boss": (act_of[k] if act_of else 9) == 0 and any(r["room_type"] == "boss" for r in combats)})
    return out


def rule_value(rule: tuple, ev: list[dict], own_floor: int, upto: int, lost_last: int = 0):
    """A counter rule's value after floors own_floor+1..upto (1-based; the item's own floor came before it was
    held). A combat on lost_last (the floor a lost run ended on) never ended: counters of ends and wins skip it."""
    kind = rule[0]
    v = {"dec": rule[1] if kind == "dec" else 0, "kindle": 5, "tea_set": False, "act1_boss": False}.get(kind, 0)
    # Cards join a deck after the relic on its own floor too (an ancient's or a shop's floor): the codex agrees
    # with this for 482 of 505 Books of Five Rings (6000 runs), against 248 when the own floor is left out.
    for k in range(own_floor + (kind != "cards"), upto + 1):
        e = ev[k - 1]
        ended = 0 if k == lost_last else e["combats"]
        if kind == "inc":
            v = v + ended if rule[1] is None else min(rule[1], v + ended)
        elif kind == "dec":
            v = max(0, v - e["combats"])  # spent as the fight starts
        elif kind == "elite":
            v += 0 if k == lost_last else e["elites"]
        elif kind == "inc_monster":
            v += 0 if k == lost_last else e["monsters"]
        elif kind == "turns":
            v = (v + e["turns"]) % rule[1]
        elif kind == "rest":
            v += rule[2] * e["rests"].count(rule[1])
        elif kind == "cards":
            v += e["cards"]
        elif kind == "kindle":
            v = max(0, v - ended) + 5 * e["rests"].count("KINDLE")
        elif kind == "tea_set":
            v = False if e["combats"] else (True if e["rest_site"] else v)
        elif kind == "card_rewards":  # a card reward screen modified, up to N (N 1: a flag, saved as a bool)
            v = min(rule[1], v + e["card_reward"])
        elif kind == "treasures":
            v += e["treasure"]
        elif kind == "act1_boss":  # Lava Rock: the act 1 boss's rewards, once won
            v = v or (e["act1_boss"] and k != lost_last)
    return (v > 0) if kind == "card_rewards" and rule[1] == 1 else v


def state_before(item: str, props: dict, at_end: bool, own_floor: int, F: int, ev: list[dict], lost_last: int,
                 flags: list[str]) -> dict:
    """The item's saved properties as they stood before floor F, in the save's shape ({"ints": [{name, value}], ...}),
    by STATE_RULES; adds to flags what cannot be had. props are the item's values at the run's end when at_end."""
    out: dict[str, list] = {}
    seen = set()
    for kind, name, value in [(k, p.get("name"), p.get("value")) for k, lst in (props or {}).items() for p in lst or []]:
        rule = STATE_RULES.get((item, name)) or STATE_RULES.get(("*", name))
        seen.add(name)
        if rule is None:
            if value not in (0, False, None, [], ""):
                flags.append(f"state_unknown:{item}.{name}")
            continue
        if rule[0] == FIXED:
            out.setdefault(kind, []).append({"name": name, "value": value})
        elif rule[0] == RESET:
            out.setdefault(kind, []).append({"name": name, "value": rule[1]})
        elif rule[0] == MONOTONE:
            fresh = PROP_FRESH.get(kind)
            if value != fresh or not at_end:
                flags.append(f"state_unknown:{item}.{name}")  # moved by the end: when, the record does not say
            out.setdefault(kind, []).append({"name": name, "value": fresh})
        else:
            if not at_end or rule_value(rule, ev, own_floor, len(ev), lost_last) != value:
                flags.append(f"state_unverified:{item}.{name}")
            out.setdefault(kind, []).append({"name": name, "value": rule_value(rule, ev, own_floor, F - 1, lost_last)})
    # A counter the end record leaves out (a save omits a fresh value) is rebuilt too, and checked against fresh.
    for (it, name), rule in STATE_RULES.items():
        if it != item or name in seen or rule[0] in (FIXED, RESET, MONOTONE):
            continue
        kind = "bools" if rule[0] in ("tea_set", "act1_boss") or rule[:2] == ("card_rewards", 1) else "ints"
        if not at_end or rule_value(rule, ev, own_floor, len(ev), lost_last) != PROP_FRESH[kind]:
            flags.append(f"state_unverified:{item}.{name}")
        v = rule_value(rule, ev, own_floor, F - 1, lost_last)
        if v != PROP_FRESH[kind]:
            out.setdefault(kind, []).append({"name": name, "value": v})
    if not at_end and not props and item in STATEFUL:
        flags.append(f"state_unknown:{item}")
    return out


def deck_before(d: dict, pts: list, F: int) -> tuple[list, bool]:
    """The deck as it stood before floor F's fight, rebuilt back from the final deck; and whether every step
    of the rebuild was unambiguous."""
    exact = True
    p = d["players"][0]
    cards = [_ck(c, True) for c in p["deck"] if int(c.get("floor_added_to_deck") or 1) < F]
    later_added = [_ck(c, True) for c in p["deck"] if int(c.get("floor_added_to_deck") or 1) >= F]
    for pt in pts[F - 1:]:
        st = (pt.get("player_stats") or [{}])[0]
        for r in st.get("cards_removed", []):
            (cards if int(r.get("floor_added_to_deck") or 1) < F else later_added).append(_ck(r))
        for t in st.get("cards_transformed", []):
            o = t.get("original_card") or {}
            (cards if int(o.get("floor_added_to_deck") or 1) < F else later_added).append(_ck(o))
    for j, pt in enumerate(pts[F - 1:], start=F):
        st = (pt.get("player_stats") or [{}])[0]
        for cid in st.get("upgraded_cards", []):
            cid = bare(cid)
            later = [c for c in later_added if c[0] == cid and c[1] > 0 and c[4] <= j]
            if later:  # the upgrade went to a card added after F
                later[0][1] -= 1
                continue
            mine = [c for c in cards if c[0] == cid and c[1] > 0]
            if mine:
                max(mine, key=lambda c: c[4])[1] -= 1
            else:
                exact = False
            if len({tuple(c[:4]) for c in cards if c[0] == cid}) > 1:
                exact = False
        for cid in st.get("downgraded_cards", []):
            mine = [c for c in cards if c[0] == bare(cid) and c[1] == 0]
            if mine:
                mine[0][1] += 1
            else:
                exact = False
        for e in st.get("cards_enchanted", []):
            card = e.get("card") or {}
            if int(card.get("floor_added_to_deck") or 1) >= F:
                continue
            ench = bare(e.get("enchantment"))
            mine = [c for c in cards if c[0] == bare(card.get("id")) and c[2] == ench]
            if mine:
                mine[0][2], mine[0][3] = "", 0
            else:
                exact = False
    return cards, exact


def relics_before(d: dict, pts: list, F: int) -> list | None:
    """Relic ids held before floor F's fight, in the order they were obtained; None if a removed relic's
    acquisition cannot be placed."""
    full = relics_before_full(d, pts, F)
    return None if full is None else [r[1] for r in full]


def relics_before_full(d: dict, pts: list, F: int) -> list | None:
    """(floor obtained, id, saved properties, whether they are the run's end values) of each relic held before
    floor F's fight, in the order they were obtained; None if a removed relic's acquisition cannot be placed."""
    p = d["players"][0]
    held = [(int(r.get("floor_added_to_deck") or 1), bare(r["id"]), r.get("props") or {}, True)
            for r in p["relics"] if int(r.get("floor_added_to_deck") or 1) < F]
    acq = collections.defaultdict(list)
    acq["BURNING_BLOOD"].append(1)
    for k, pt in enumerate(pts, 1):
        st = (pt.get("player_stats") or [{}])[0]
        for rc in st.get("relic_choices", []):
            if rc.get("was_picked"):
                acq[bare(rc.get("choice"))].append(k)
        for b in st.get("bought_relics", []):
            acq[bare(b)].append(k)
        for a in st.get("ancient_choice", []):
            if a.get("was_chosen"):
                acq[a.get("TextKey", "")].append(k)
    for k, pt in enumerate(pts[F - 1:], start=F):
        st = (pt.get("player_stats") or [{}])[0]
        for rid in st.get("relics_removed", []):
            rid = bare(rid)
            got = [x for x in acq.get(rid, []) if x <= k]
            if not got:
                return None
            if max(got) < F:
                held.append((max(got), rid, {}, False))
    return sorted(held, key=lambda x: x[0])


def potions_before(pts: list, F: int, final: list) -> tuple[list, bool]:
    """The potion belt before floor F's fight, walked forward from the history; and whether the same walk
    to the end gives the run's final belt (else potions came or went unrecorded)."""
    def walk(upto: int):
        belt, ok = [], True
        for pt in pts[:upto]:
            st = (pt.get("player_stats") or [{}])[0]
            for x in st.get("potion_used", []) + st.get("potion_discarded", []):
                if bare(x) in belt:
                    belt.remove(bare(x))
                else:
                    ok = False
            for pc in st.get("potion_choices", []):
                if pc.get("was_picked"):
                    belt.append(bare(pc.get("choice")))
            for b in st.get("bought_potions", []):
                belt.append(bare(b))
        return belt, ok
    at, ok1 = walk(F - 1)
    end, ok2 = walk(len(pts))
    return at, ok1 and ok2 and sorted(end) == sorted(bare(x.get("id")) for x in final)


def realistic_candidates(d: dict, t: Tables, stats: collections.Counter):
    """(cell, bucket, hash, candidate) for every fight of one run whose state the emulator can set up."""
    seed = d.get("seed", "")
    if not (len(seed) == 10 and all(ch in ALPHABET for ch in seed)):
        stats["skip_seed_format"] += 1
        return
    acts = d["map_point_history"]
    pts, act_of, pos_of = [], [], []
    for ai, act in enumerate(acts):
        for pi, pt in enumerate(act):
            pts.append(pt)
            act_of.append(ai)
            pos_of.append(pi / max(1, len(act) - 1))
    first = (pts[0].get("player_stats") or [{}])[0] if pts else {}
    neow = [a.get("TextKey") for a in first.get("ancient_choice", [])]
    neow_pick = next((i for i, a in enumerate(first.get("ancient_choice", [])) if a.get("was_chosen")), 0)
    final_potions = d["players"][0].get("potions", [])
    for F in range(2, len(pts) + 1):
        pt = pts[F - 1]
        rooms = pt.get("rooms") or []
        if len(rooms) != 1 or rooms[0].get("room_type") not in ("monster", "elite", "boss"):
            continue
        enc = bare(rooms[0].get("model_id"))
        if enc not in t.encounters or act_of[F - 1] > 2:
            stats["cand_skip_encounter"] += 1
            continue
        prev = (pts[F - 2].get("player_stats") or [{}])[0]
        hp, max_hp = int(prev.get("current_hp", 0)), int(prev.get("max_hp", 0))
        if hp < 1 or max_hp < 1:
            stats["cand_skip_hp"] += 1
            continue
        deck, deck_exact = deck_before(d, pts, F)
        relics = relics_before(d, pts, F)
        if relics is None:
            stats["cand_skip_relic_history"] += 1
            continue
        if not all(t.card_ok(c[0], c[1], c[2]) for c in deck):
            stats["cand_skip_card_unmodelled"] += 1
            continue
        neow_relic = neow[neow_pick] if neow else None
        if not all(r in t.follow and (t.relic_ok(r) or r == neow_relic or r == "BURNING_BLOOD") for r in relics):
            stats["cand_skip_relic_unmodelled"] += 1
            continue
        potions, potions_exact = potions_before(pts, F, final_potions)
        if not all(p in t.potions_emu for p in potions):
            stats["cand_skip_potion_unmodelled"] += 1
            continue
        room = rooms[0]["room_type"]
        n_buckets = next(b for r, _, b in REALISTIC_CELLS if r == room)
        bucket = min(n_buckets - 1, int(pos_of[F - 1] * n_buckets))
        stats["candidates"] += 1
        cand = {"run": d.get("run_hash"), "seed": seed, "floor": F, "act": act_of[F - 1] + 1, "room": room,
                "act_name": bare(d["acts"][act_of[F - 1]]) if act_of[F - 1] < len(d.get("acts", [])) else "",
                "encounter": enc, "monsters": [bare(m) for m in rooms[0].get("monster_ids", [])],
                "hp": hp, "max_hp": max_hp, "deck": [c[:4] for c in deck], "relics": relics, "potions": potions,
                "neow": neow_pick, "neow_relic": neow_relic,
                "exact": {"deck": deck_exact, "potions": potions_exact}, "win": d.get("win")}
        # Lazily: the saved state and the rest of the holdings, only for the candidates picked (realistic_state).
        cand["_state"] = (d, pts, F, deck)
        yield (cand["act"], room), bucket, h("refset-v1", d.get("run_hash"), F), cand


# Potion slots: the A10 belt (the codex: 4996 of the first 5000 Ironclad A10 runs end with 2 slots plus their
# slot relics' own), and what each slot relic adds (PotionBelt, PhialHolster, AlchemicalCoffer: "PotionSlots").
A10_POTION_SLOTS = 2
POTION_SLOT_RELICS = {"POTION_BELT": 2, "PHIAL_HOLSTER": 1, "ALCHEMICAL_COFFER": 4}


def save_card(c: list, props: dict) -> dict:
    """A deck card in the game's save shape (SerializableCard), as player_load reads it."""
    out: dict = {"id": "CARD." + c[0]}
    if c[1]:
        out["current_upgrade_level"] = c[1]
    if c[2]:
        out["enchantment"] = {"id": "ENCHANTMENT." + c[2], "amount": c[3]}
    if props:
        out["props"] = props
    out["floor_added_to_deck"] = c[4]
    return out


def realistic_state(cand: dict) -> dict:
    """The holdings player_load puts in place for one picked candidate: the deck and relics in the save's shape
    with their saved state before the floor (state_before), gold, potion slots; and the spec's realism flags."""
    d, pts, F, deck = cand.pop("_state")
    flags: list[str] = []
    ev = floor_events(pts, [ai for ai, act in enumerate(d["map_point_history"]) for _ in act])
    lost_last = 0 if d.get("win") else len(pts)
    cards = []
    for c in deck:
        props = state_before(c[0], c[5], c[6], c[4], F, ev, lost_last, flags) if (c[5] or c[0] in STATEFUL) else {}
        cards.append(save_card(c, props))
    relics = []
    for floor, rid, props, at_end in relics_before_full(d, pts, F) or []:
        r: dict = {"id": "RELIC." + rid}
        state = state_before(rid, props, at_end, floor, F, ev, lost_last, flags) if (props or rid in STATEFUL) else {}
        if state:
            r["props"] = state
        r["floor_added_to_deck"] = floor
        relics.append(r)
    prev = (pts[F - 2].get("player_stats") or [{}])[0]
    p = d["players"][0]
    final_slots = int(p.get("max_potion_slot_count") or 0)
    if A10_POTION_SLOTS + sum(POTION_SLOT_RELICS.get(bare(r["id"]), 0) for r in p["relics"]) != final_slots:
        flags.append("potion_slots_unverified")
    slots = A10_POTION_SLOTS + sum(POTION_SLOT_RELICS.get(r, 0) for r in cand["relics"])
    if not cand["exact"]["deck"]:
        flags.append("deck_inexact")
    if not cand["exact"]["potions"]:
        flags.append("potions_inexact")
    if len(cand["potions"]) > slots:
        flags.append("potions_over_slots")
    if "FUR_COAT" in cand["relics"]:
        flags.append("map_coordinate:FUR_COAT")  # its marked rooms are map coordinates; the fight is met at row 1
    return {"deck": cards, "relics": relics, "gold": int(prev.get("current_gold", 0) or 0), "slots": slots,
            "flags": sorted(set(flags))}


def realistic_specs(codex: Path, t: Tables, total: int, stats: collections.Counter, reach: dict, max_runs: int = 0) -> list[dict]:
    """Stratified pick of real fights: act x room type quotas, floor buckets within each, one fight per run."""
    scale = total / 600
    quota = {(a, r): max(1, round(n * scale)) for a in (1, 2, 3) for r, n, _ in REALISTIC_CELLS}
    nb = {r: b for r, _, b in REALISTIC_CELLS}
    keep = {k: {b: [] for b in range(nb[k[1]])} for k in quota}  # cell -> bucket -> [(rank, hash, cand)]
    for d in codex_runs(codex, stats, max_runs):
        p = d["players"][0]
        reach["cards"].update(bare(c["id"]) for c in p["deck"])
        reach["relics"].update(bare(r["id"]) for r in p["relics"])
        for r in p["relics"]:  # the first real pick seen of each event-picked relic, for coverage (coverage_holdings)
            if bare(r["id"]) not in reach.setdefault("fixed_props", {}) and fixed_props_of(r):
                reach["fixed_props"][bare(r["id"])] = fixed_props_of(r)
        for pt in (x for a in d["map_point_history"] for x in a):
            st = (pt.get("player_stats") or [{}])[0]
            reach["potions"].update(bare(pc.get("choice")) for pc in st.get("potion_choices", []))
            reach["relics"].update(bare(r) for r in st.get("relics_removed", []))
        best = {}
        for cell, bucket, hh, cand in realistic_candidates(d, t, stats):
            k = (cell, bucket)
            rank = (not cand["exact"]["deck"]) + (not cand["exact"]["potions"])
            if k not in best or (rank, hh) < best[k][:2]:
                best[k] = (rank, hh, cand)
        for (cell, bucket), item in best.items():
            item[2]["_st"] = realistic_state(item[2])  # now, so the run's record is not kept
            lst = keep[cell][bucket]
            lst.append(item)
            if len(lst) > 8 * quota[cell]:
                lst.sort(key=lambda x: x[:2])
                del lst[4 * quota[cell]:]
    used, out = set(), []
    for cell in sorted(quota):
        n, buckets = quota[cell], keep[cell]
        per = [n // len(buckets) + (1 if i < n % len(buckets) else 0) for i in range(len(buckets))]
        for b, want in zip(sorted(buckets), per):
            got = 0
            for _, _, cand in sorted(buckets[b], key=lambda x: x[:2]):
                if got >= want:
                    break
                if cand["run"] in used:
                    continue
                used.add(cand["run"])
                out.append(cand)
                got += 1
            if got < want:
                stats[f"short_act{cell[0]}_{cell[1]}_b{b}"] += want - got
    return [realistic_entry(c, t) for c in out]


def player_load_items(act: int, deck: list, relics: list, potions: list, slots: int, hp: int, max_hp: int, gold: int | None,
                      encounter: str) -> list:
    """The fight's own act (from act 2 on), the holdings through the game's save-load path, then the fight."""
    load = {"cmd": "player_load", "deck": deck, "relics": relics, "potions": list(potions), "max_potion_slots": int(slots),
            "hp": int(min(hp, max_hp)), "max_hp": int(max_hp)}
    if gold is not None:
        load["gold"] = int(gold)
    return ([{"cmd": "act", "index": act}] if act > 1 else []) + [load, {"cmd": "fight", "id": encounter}]


def setup_items(relics: list, deck: list, hp: int, max_hp: int, potions: list, encounter: str) -> list:
    """The deck goes in first, while only the starter and Neow relics are held (so eggs, Fresnel Lens or Bing Bong
    obtained later do not act on cards the run had before them), then the relics in the run's order, then the
    deck again to undo what their pickups changed (a transform, a removal, an added card)."""
    deck_set = {"cmd": "deck_set", "cards": [{"id": c[0], "upgraded": c[1] > 0, **({"enchantment": c[2], "amount": c[3]} if c[2] else {})}
                                             for c in deck]}
    return [
        deck_set,
        {"cmd": "relics_set", "ids": list(relics)},
        dict(deck_set),
        {"cmd": "hp", "max": int(max_hp), "current": int(min(hp, max_hp))},
        {"cmd": "potions_set", "ids": list(potions)},
        {"cmd": "fight", "id": encounter},
    ]


def realistic_entry(c: dict, t: Tables) -> dict:
    hh = h("refset-v1", c["run"], c["floor"])
    st = c.get("_st") or (realistic_state(c) if "_state" in c else None) or {"deck": [save_card([*x, 1], {}) for x in c["deck"]],
                                                     "relics": [{"id": "RELIC." + r} for r in c["relics"]],
                                                     "gold": None, "slots": A10_POTION_SLOTS, "flags": []}
    act_name, kind, _ = t.encounters[c["encounter"]]
    spec = {"kind": "combat", "seed": c["seed"], "character": "IRONCLAD", "ascension": 10, "neow": c["neow"],
            "chooser_seed": int(hh[:8], 16) % 1000003, "max_actions": MAX_ACTIONS, "use_potions": True,
            "chooser": "heuristic", "relic_flashes": True,
            "setup": player_load_items(c["act"], st["deck"], st["relics"], c["potions"], st["slots"], c["hp"], c["max_hp"],
                                       st["gold"], c["encounter"]),
            "tag": f"realistic:act{c['act']}:{c['room']}"}
    return {"id": f"R{c['act']}{c['room'][0].upper()}-{c['seed']}-F{c['floor']:02d}", "part": "realistic",
            "strata": {"act": c["act"], "room": c["room"], "kind": kind, "floor": c["floor"], "act_name": c["act_name"]},
            "reason": (f"real Ironclad A10 run (codex {str(c['run'])[:8]}), floor {c['floor']} (act {c['act']} {c['room']}, "
                       f"{c['encounter']}): {len(c['deck'])} cards, {len(c['relics'])} relics, {len(c['potions'])} potions, "
                       f"HP {c['hp']}/{c['max_hp']}"),
            "exact": c["exact"], "neow_relic": c["neow_relic"], "codex_monsters": c["monsters"],
            "chooser": "heuristic", "flags": st["flags"], "expect_act": act_name if c["act"] > 1 else None, "spec": spec}


# ---------------------------------------------------------------- coverage: the fewest plausible setups that reach every item

def _seed(rng: random.Random) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(10))


def coverage_targets(t: Tables, reach: dict) -> dict:
    cards = [c for c in t.cards if c not in t.other_cards]
    variants = [(c, False) for c in cards] + [(c, True) for c in cards if c in t.upgradable]
    relics = sorted(r for r in (set(t.pool_relics) | (t.event_relics & reach.get("relics", set())))
                    if r in t.follow and t.relic_ok(r) and r not in t.other_relics and t.pool_relics.get(r) != "Starter")
    potions = sorted(p for p in (t.pool_potions | (reach.get("potions", set()) & t.pool_potions))
                     if p in t.potions_emu and p not in t.other_potions)
    encounters = sorted(t.encounters, key=lambda e: (ACT_NUMBER[t.encounters[e][0]], t.encounters[e][0], KINDS.index(t.encounters[e][1]), e))
    return {"variants": variants, "relics": relics, "potions": potions, "encounters": encounters}


# Plans, per item, how often it should appear: cards 7 decks (about 5 plays: a copy in a 15-20 card deck is
# played about 0.8 times a fight under random play), relics 3 fights, potions 5 belts (about 3 uses), encounters 3.
PLAN = {"card": 7, "relic": 3, "potion": 5, "encounter": 3}
BASICS = {1: (4, 4), 2: (3, 3), 3: (2, 2)}  # Strikes, Defends by act (plus Bash and Ascender's Bane)
TARGETS_PER_DECK = {1: 9, 2: 11, 3: 13}
RELICS_PER_FIGHT = {1: 2, 2: 3, 3: 4}


def coverage_holdings(relics: list, deck: list, fixed_props: dict) -> tuple[list, list]:
    """A coverage fight's deck and relics in the save's shape, as a run that picked those relics up holds them: a
    relic whose event picked something (Archaic Tooth, Dusty Tome, Touch of Orobas, ...) carries the pick a real
    run's record shows (fixed_props), with what its pickup left: Archaic Tooth's starter card transformed, Dusty
    Tome's card added upgraded, Touch of Orobas' starter relic upgraded. Counters start fresh, as just obtained."""
    deck = [list(c) for c in deck]
    relics = list(relics)
    props = {r: fixed_props.get(r) for r in relics if fixed_props.get(r)}
    val = {r: {x["name"]: x["value"] for lst in pr.values() for x in lst} for r, pr in props.items()}
    if "ARCHAIC_TOOTH" in val:
        start, anc = (bare((val["ARCHAIC_TOOTH"].get(k) or {}).get("id")) for k in ("StarterCard", "AncientCard"))
        at = next((i for i, c in enumerate(deck) if c[0] == start), None)
        if at is not None and anc:
            deck[at] = [anc, deck[at][1], deck[at][2], deck[at][3]]
    if "DUSTY_TOME" in val and val["DUSTY_TOME"].get("AncientCard"):
        deck.append([bare(val["DUSTY_TOME"]["AncientCard"]), 1, "", 0])
    if "TOUCH_OF_OROBAS" in val:
        old, new = (bare(val["TOUCH_OF_OROBAS"].get(k)) for k in ("StarterRelic", "UpgradedRelic"))
        if old in relics and new:
            relics[relics.index(old)] = new
    out_relics = []
    for r in relics:
        e: dict = {"id": "RELIC." + r}
        if props.get(r):
            e["props"] = props[r]
        e["floor_added_to_deck"] = 1
        out_relics.append(e)
    return [save_card([*c[:4], 1], {}) for c in deck], out_relics


def fixed_props_of(relic: dict) -> dict:
    """The relic's saved properties that STATE_RULES marks fixed (its event's pick), in the save's shape."""
    rid = bare(relic.get("id"))
    out: dict = {}
    for kind, lst in (relic.get("props") or {}).items():
        for x in lst or []:
            rule = STATE_RULES.get((rid, x.get("name")))
            if rule and rule[0] == FIXED:
                out.setdefault(kind, []).append({"name": x["name"], "value": x["value"]})
    return out


def coverage_specs(t: Tables, reach: dict) -> list[dict]:
    tg = coverage_targets(t, reach)
    rng = random.Random("refset-v1-coverage")
    need = collections.Counter({("card", v): PLAN["card"] for v in tg["variants"]})
    need.update({("relic", r): PLAN["relic"] for r in tg["relics"]})
    need.update({("potion", p): PLAN["potion"] for p in tg["potions"]})
    order = []
    for rep in range(PLAN["encounter"]):
        es = list(tg["encounters"])
        random.Random(f"refset-v1-enc-{rep}").shuffle(es)
        order += es
    out, n = [], 0
    while order or any(v > 0 for v in need.values()):
        enc = order.pop(0) if order else tg["encounters"][n % len(tg["encounters"])]
        act_name, kind, _ = t.encounters[enc]
        act = ACT_NUMBER[act_name]

        def take(kind_: str, pool: list, k: int, avoid=()) -> list:
            ranked = sorted(pool, key=lambda x: (-need[(kind_, x)], rng.random()))
            got = [x for x in ranked if x not in avoid][:k]
            for x in got:
                need[(kind_, x)] -= 1
            return got
        variants = take("card", tg["variants"], TARGETS_PER_DECK[act])
        relics = take("relic", tg["relics"], RELICS_PER_FIGHT[act])
        slots = 2 + (2 if "POTION_BELT" in relics else 0)
        potions = [] if "SOZU" in relics else take("potion", tg["potions"], slots)
        s, dfd = BASICS[act]
        deck = ([["STRIKE_IRONCLAD", 0, "", 0]] * s + [["DEFEND_IRONCLAD", 0, "", 0]] * dfd
                + [["BASH", 0, "", 0], ["ASCENDERS_BANE", 0, "", 0]] + [[c, int(u), "", 0] for c, u in variants])
        starter = [] if "BLACK_BLOOD" in relics else ["BURNING_BLOOD"]
        max_hp = 80 + 4 * (act - 1) + rng.randint(0, 10)
        hp = max(1, int(max_hp * rng.uniform(0.55, 1.0)))
        load_deck, load_relics = coverage_holdings(starter + relics, deck, reach.get("fixed_props", {}))
        load_slots = A10_POTION_SLOTS + sum(POTION_SLOT_RELICS.get(r, 0) for r in relics)
        spec = {"kind": "combat", "seed": _seed(rng), "character": "IRONCLAD", "ascension": 10, "neow": n % 3,
                "chooser_seed": 9000 + n, "max_actions": MAX_ACTIONS, "use_potions": True, "chooser": "random",
                "relic_flashes": True,
                "setup": player_load_items(act, load_deck, load_relics, potions, load_slots, hp, max_hp, None, enc),
                "tag": f"coverage:act{act}:{kind}"}
        out.append({"id": f"C{n + 1:04d}-{spec['seed']}", "part": "coverage", "chooser": "random", "flags": [],
                    "expect_act": act_name if act > 1 else None,
                    "strata": {"act": act, "room": ROOM_OF_KIND[kind], "kind": kind, "act_name": act_name},
                    "reason": (f"coverage: {enc} ({act_name} {kind}); deck targets {', '.join(c + ('+' if u else '') for c, u in variants)}; "
                               f"relics {', '.join(relics) or '-'}; potions {', '.join(potions) or '-'}"),
                    "targets": {"cards": [c + ("+" if u else "") for c, u in variants], "relics": relics, "potions": potions,
                                "encounter": enc}, "spec": spec})
        n += 1
        if n > 2000:
            break
    return out


# ---------------------------------------------------------------- regression: one minimal setup per known divergence class

# Each class: (name, where it was found, cards, relics, potions, encounter, fight-start powers, original seeds).
# Cards go in a small deck (3 copies each with 2 Strikes and 2 Defends) so random play reaches them early.
REGRESSION_CLASSES = [
    ("pool order: Splash's other-character pools", "FINDINGS 2026-10-13", ["SPLASH"], [], [], None, [], ["YDT1E1R92A"]),
    ("combat generation pool filter: Jack of All Trades", "FINDINGS / triage generation", ["JACK_OF_ALL_TRADES"], [], [], None, [], ["MVNNHX131P", "2ZM2F3EGWU"]),
    ("combat generation pool filter: Discovery", "FINDINGS / triage generation", ["DISCOVERY"], [], [], None, [], ["J1VPNKFZ00", "SEJU9UUPR0"]),
    ("combat generation pool filter: Jackpot", "FINDINGS / triage generation", ["JACKPOT"], [], [], None, [], ["PNTTYRBQGR", "GYQN75150X"]),
    ("combat generation pool filter: Stoke", "FINDINGS / triage generation", ["STOKE"], [], [], None, [], ["F8SYM8H1A2", "SZRRC58F5Z"]),
    ("factory potion roll: Alchemize", "FINDINGS / triage generation", ["ALCHEMIZE"], [], [], None, [], ["7LMBLSSF3M", "MX1GNK7KGX"]),
    ("random pile placement: Capture Spirit's Souls", "FINDINGS / triage generation", ["CAPTURE_SPIRIT"], [], [], None, [], ["Q91FAC7EJB"]),
    ("hook order: Thorns after the damage is fixed (Rupture)", "FINDINGS / triage enemy-effects", ["STRIKE_IRONCLAD", "RUPTURE"], [], [], "TOADPOLES_WEAK", [], ["P70F1RFA2Y"]),
    # Thorns 15 against Toadpoles (22-26 HP at A10, Spike Spit 3 hits): the game's ThornsPower strikes back before
    # each blow and AttackCommand stops the hits once the attacker is dead, so a Toadpole dies after its first or
    # second hit; the emulator's DealAttack lands every hit and its Thorns after each. A Nibbit's single hit lands
    # either way, so the old Nibbit setup never reached the bug.
    ("hook order: the player's Thorns before an enemy hit lands (open)", "FINDINGS 2026-10-13", ["DEFEND_IRONCLAD"], [], [], "TOADPOLES_WEAK", [("THORNS_POWER", 15, "player")], []),
    ("hidden counter: Inferno's self damage", "triage powers-a", ["INFERNO"], [], [], None, [], ["J1XUUAW07A"]),
    ("hidden counter: Crimson Mantle", "triage powers-a", ["CRIMSON_MANTLE"], [], [], None, [], ["S37M8GW3NE"]),
    ("hidden counter: The Bomb", "triage powers-a / night04", ["THE_BOMB"], [], [], None, [], ["2YCT30W7AY", "CDE3SPM32Z"]),
    ("power on play: Beacon of Hope", "triage powers-a", ["BEACON_OF_HOPE"], [], [], None, [], ["25MD4CTEXL"]),
    ("power on play: Tank", "triage powers-a", [], [], [], None, [], ["L0382734W2", "P61G9Z6JUJ"]),
    ("comparer: Panache", "triage powers-b", ["PANACHE"], [], [], None, [], []),
    ("comparer: Setup Strike", "triage powers-b", ["SETUP_STRIKE"], [], [], None, [], []),
    ("end of turn: block and HP (Defend)", "triage powers-b", ["DEFEND_IRONCLAD"], [], [], None, [], ["PMJAMTN7YX"]),
    ("end of turn: Stampede's target stream", "triage end-turn", ["STAMPEDE"], [], [], None, [], ["X10CFTPU9L", "DLADDVSPCS"]),
    ("end of turn: Stratagem", "triage end-turn", ["STRATAGEM"], [], [], None, [], ["0Q5B2PVQ2T", "RZY8A5KYY0"]),
    ("end of turn: Entropy", "triage end-turn", ["ENTROPY"], [], [], None, [], ["3409CE6CNA", "LU2SE4M7WF"]),
    ("driver: turn-start auto-play (Mayhem, Hellraiser, Rolling Boulder)", "triage end-turn / night04 freezes",
     ["MAYHEM", "HELLRAISER", "ROLLING_BOULDER"], [], [], None, [], ["6V7YRDLB2L", "VC4GU7G4FN"]),
    ("end of turn: Beat Down", "triage end-turn", ["BEAT_DOWN"], [], [], None, [], ["EF4DGFCG8B"]),
    ("target stream: Havoc", "triage generation / night04", ["HAVOC"], [], [], None, [], ["04CR6MK7U5", "SDN432DTY7"]),
    ("enemy powers: Dark Shackles", "triage enemy-effects", ["DARK_SHACKLES"], [], [], None, [], []),
    ("enemy powers: Mangle", "triage enemy-effects / night04", ["MANGLE"], [], [], None, [], ["B51QSC208N"]),
    ("enemy powers: Knockdown", "triage enemy-effects / night04", ["KNOCKDOWN"], [], [], None, [], ["PBLPGKQ618", "3S1UX1BPU3"]),
    ("enemy powers: Tag Team", "triage enemy-effects / night04", ["TAG_TEAM"], [], [], None, [], ["NAJP4BQKVR", "AAVMJZUDPK"]),
    ("pile order: Purity's exhaust", "triage multihit-piles", ["PURITY"], [], [], None, [], ["B31RULS23F"]),
    ("pile order: Catastrophe", "triage multihit-piles / night04", ["CATASTROPHE"], [], [], None, [], ["YGQDZTK4BU", "0VN5TMAGE2"]),
    ("selection: Brand", "triage multihit-piles", ["BRAND"], [], [], None, [], ["KEHPPTBGE1"]),
    ("selection: Armaments", "triage multihit-piles", ["ARMAMENTS"], [], [], None, [], ["KQT9NZ4JLH"]),
    ("selection: Secret Weapon", "triage multihit-piles", ["SECRET_WEAPON"], [], [], None, [], ["ULVHYPSR3R"]),
    ("selection: Seeker Strike", "triage multihit-piles", ["SEEKER_STRIKE"], [], [], None, [], ["YCEFGNM668"]),
    ("fight start: opening draw", "triage multihit-piles / night04", [], [], [], None, [], ["672CRJ8TJ9", "GQRZXY0HR0"]),
    ("multi-hit: Volley", "triage multihit-piles", ["VOLLEY"], [], [], None, [], []),
    ("multi-hit: Sword Boomerang", "triage multihit-piles", ["SWORD_BOOMERANG"], [], [], None, [], []),
    ("generated cards: Dirge", "night04", ["DIRGE"], [], [], None, [], ["CZ6FCBJ4RG"]),
    ("block: Cascade", "night04", ["CASCADE"], [], [], None, [], ["QK66593PE0"]),
    ("enemy powers: Uppercut", "night04", ["UPPERCUT"], [], [], None, [], ["C9H67S1JG2"]),
    ("end of turn: discard pile order", "night04", [], [], [], None, [], ["W3LQBXMEG1"]),
    ("end of turn: enemy moves", "night04", [], [], [], None, [], ["EHQTVLJCWS"]),
    ("end of turn: CombatTargets stream", "night04", [], [], [], None, [], ["0MSJJVRJTD"]),
    ("end of turn: hand after the draw", "night04", [], [], [], None, [], ["V3LDXLDHW2"]),
    ("end of turn: MonsterAi stream", "night04", [], [], [], None, [], ["GVCYB1LRQJ"]),
    ("run start: Neow's Bones nested rewards", "FINDINGS 2026-10-13", [], [], [], None, [], ["0CERYNNVH0", "LT3K8D3K44"]),
    ("room type: elites outside the hand-written list (Soul Nexus)", "FINDINGS setup fights", [], [], [], "SOUL_NEXUS_ELITE", [], ["HM1ZS1TZRZ"]),
    ("monster start: Turret Operator's Rampart", "FINDINGS setup fights", [], [], [], "TURRET_OPERATOR_WEAK", [], ["XKSB75348Q", "HVL23QXSWR"]),
    ("monster classes: Decimillipede segments", "FINDINGS setup fights", [], [], [], "DECIMILLIPEDE_ELITE", [], ["C7KQXGB0QA"]),
    ("potion roll without repeats: Phial Holster", "FINDINGS setup fights", [], ["PHIAL_HOLSTER"], [], None, [], ["46KVVMX437"]),
    ("fight start: Stone Cracker upgrades a draw-pile card", "FINDINGS setup fights", [], ["STONE_CRACKER"], [], None, [], ["FSW6LVSPDQ"]),
    ("fight start: Mercury Hourglass", "FINDINGS setup fights", [], ["MERCURY_HOURGLASS"], [], None, [], ["U0TNXQZM08"]),
    ("fight start: Toolbox's pick", "FINDINGS setup fights", [], ["TOOLBOX"], [], None, [], ["8UAGRV20G8"]),
    ("fight start: Gambling Chip with Vexing Puzzlebox", "FINDINGS setup fights", [], ["GAMBLING_CHIP", "VEXING_PUZZLEBOX"], [], None, [], ["6K928H3WXL"]),
    ("cost: Snecko Oil with Dismantle+", "FINDINGS setup fights", ["DISMANTLE"], [], ["SNECKO_OIL"], None, [], ["4FQ6NTF72R"]),
    ("turn start: Aggression returns an Attack", "FINDINGS setup fights", ["AGGRESSION"], [], [], None, [], ["F00EAE08LE"]),
    ("power timing: Self-Forming Clay after Bloodletting", "FINDINGS setup fights", ["BLOODLETTING"], ["SELF_FORMING_CLAY"], [], None, [], ["MM7V9M80NV"]),
    ("selection count: Kifuda / Gnarled Hammer take 0 to 3 cards", "FINDINGS setup fights", [], ["KIFUDA"], [], None, [], []),
]


def find_original(seed: str, cache: dict) -> dict | None:
    """The spec a fight was first recorded from, in the local job files (jobs/*.json, jobs-setup/*.packs/minimal)."""
    if not cache:
        for p in sorted(JOBS.glob("*.json")) + sorted(JOBS_SETUP.glob("*.packs/minimal/*.json")) + sorted(JOBS_SETUP.glob("*.json")):
            if p.name.endswith((".out.json", ".emu.json", ".reanchor.json", ".verdicts.json")) or p.stat().st_size > 50_000_000:
                continue
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            runs = doc.get("runs") if isinstance(doc, dict) else doc if isinstance(doc, list) else None
            for s in runs or []:
                if isinstance(s, dict) and s.get("kind") == "combat" and s.get("seed"):
                    # A shrunk minimal setup (packs/minimal) wins over the fight it came from.
                    if s["seed"] not in cache or "/minimal/" in p.as_posix():
                        cache[s["seed"]] = (s, p.name)
        cache.setdefault("", None)
    return cache.get(seed)


def regression_specs(t: Tables) -> list[dict]:
    rng = random.Random("refset-v1-regression")
    cache: dict = {}
    out = []
    plain = "NIBBITS_NORMAL"
    for n, (name, source, cards, relics, potions, enc, powers, seeds) in enumerate(REGRESSION_CLASSES):
        cross = [c for c in cards if c not in t.card_info] + [r for r in relics if r in t.other_relics]
        missing = [c for c in cards if c not in t.cards_emu] + [r for r in relics if r not in t.follow] + \
                  [p for p in potions if p not in t.potions_emu]
        if (cards or relics or potions or enc or powers) and not missing:
            up = name.startswith("cost: Snecko")
            deck = ([["STRIKE_IRONCLAD", 0, "", 0]] * 2 + [["DEFEND_IRONCLAD", 0, "", 0]] * 2
                    + [[c, int(up), "", 0] for c in cards for _ in range(3)])
            items = setup_items(["BURNING_BLOOD"] + relics, deck, 70, 80, potions, enc or plain)
            items += [{"cmd": "power", "id": p, "amount": a, "target": tg} for p, a, tg in powers]
            act = ACT_NUMBER[t.encounters[enc or plain][0]] if (enc or plain) in t.encounters else 1
            if act > 1:  # met in its own act, as a run meets it
                items = [{"cmd": "act", "index": act}] + items
            spec = {"kind": "combat", "seed": _seed(rng), "character": "IRONCLAD", "ascension": 10, "neow": n % 3,
                    "chooser_seed": 7700 + n, "max_actions": MAX_ACTIONS, "use_potions": True, "setup": items,
                    "chooser": "random", "relic_flashes": True, "tag": f"regression:{name}"}
            out.append({"id": f"G{n + 1:03d}-{spec['seed']}", "part": "regression", "class": name, "source": source,
                        "class_kind": class_kind(name), "trigger": class_trigger(name),
                        "reason": f"minimal setup for the divergence class '{name}' ({source})", "chooser": "random",
                        "expect_act": t.encounters[enc or plain][0] if act > 1 else None,
                        "flags": ["cross_character"] if cross else [], "spec": spec})
            if name in KILL_CLASSES:
                # More seeds of the same minimal setup, from their own stream (the other specs' seeds stay put):
                # whether a kill partway through an attack shows depends on the player's block that turn and on it
                # not ending the fight (after the last enemy dies the comparer compares only in_combat).
                xr, base = random.Random(f"refset-v2-kill-extra:{name}"), out[-1]
                for k in range(KILL_EXTRA_FIGHTS):
                    extra = dict(spec, seed=_seed(xr), chooser_seed=7700 + n + 100 * (k + 1), setup=json.loads(json.dumps(items)))
                    out.append({**base, "id": f"G{n + 1:03d}{chr(ord('b') + k)}-{extra['seed']}", "spec": extra,
                                "reason": f"minimal setup for the divergence class '{name}' ({source}), seed {k + 2}"})
        for s in seeds:
            got = find_original(s, cache)
            if not got:
                continue
            spec, src = got
            spec = {k: v for k, v in spec.items() if k not in ("out",)}
            spec.setdefault("max_actions", MAX_ACTIONS)
            spec["tag"] = f"regression-original:{name}"
            spec["relic_flashes"] = True
            out.append({"id": f"G{n + 1:03d}o-{s}", "part": "regression", "class": name, "source": f"{source}; original {src}",
                        "class_kind": class_kind(name), "trigger": class_trigger(name),
                        "reason": f"the fight that found '{name}', re-recorded with the fixed driver",
                        "chooser": "script" if spec.get("script") else spec.get("chooser", "random"), "flags": [], "spec": spec})
    return out


FILLER_CARDS = {"STRIKE_IRONCLAD", "DEFEND_IRONCLAD"}
# Classes about a relic's pickup itself (what it rolls or asks for as it is obtained), not about it in a fight: their
# trigger is the relic obtained by the setup, held at the fight's start.
PICKUP_CLASSES = {"potion roll without repeats: Phial Holster", "selection count: Kifuda / Gnarled Hammer take 0 to 3 cards"}
# Classes about a player power that strikes back per hit as an enemy attacks: reached only when a multi-hit attacker
# with HP low enough for the power alone to kill it before its last hit died in that enemy turn (the game stops the
# attack there; the hits after are the case the open bug is about), not merely when it attacked into the power or died.
KILL_CLASSES = {"hook order: the player's Thorns before an enemy hit lands (open)"}
KILL_EXTRA_FIGHTS = 2  # extra seeds of a kill class's minimal setup (regression_specs)
# Classes whose trigger is not one of their setup's cards, relics, potions or powers: what one of their recorded game
# fights must show for a pass to count (refset_score.evidence). No class is reached by a plain replay.
CLASS_TRIGGERS = {
    "power on play: Tank": {"plays": ["TANK"]},
    "fight start: opening draw": {"opening_hand": True},
    "end of turn: discard pile order": {"end_turn_discard": 2},
    "end of turn: enemy moves": {"enemy_turn": True},
    "end of turn: CombatTargets stream": {"rng_end_turn": ["CombatTargets"]},
    "end of turn: hand after the draw": {"end_turn_draw": True},
    "end of turn: MonsterAi stream": {"rng_end_turn": ["MonsterAi"]},
    "run start: Neow's Bones nested rewards": {"neow": ["NEOWS_BONES"]},
    "room type: elites outside the hand-written list (Soul Nexus)": {"encounter": ["SOUL_NEXUS_ELITE"]},
    "monster start: Turret Operator's Rampart": {"encounter": ["TURRET_OPERATOR_WEAK"], "enemy_powers_start": ["RAMPART_POWER"]},
    "monster classes: Decimillipede segments": {"encounter": ["DECIMILLIPEDE_ELITE"]},
}


def class_trigger(name: str) -> dict | None:
    """The trigger of a regression class by name: CLASS_TRIGGERS, else from its REGRESSION_CLASSES entry (the manifest
    keeps a copy)."""
    if name in CLASS_TRIGGERS:
        return dict(CLASS_TRIGGERS[name])
    for cname, _, cards, relics, potions, _, powers, _ in REGRESSION_CLASSES:
        if cname == name:
            return trigger_of(cards, relics, potions, powers, pickup=name in PICKUP_CLASSES, kill=name in KILL_CLASSES)
    return None


def class_kind(name: str) -> str:
    """'driver' for a class about the referee driver itself (counted apart), else 'emulator'."""
    return "driver" if name.startswith("driver:") else "emulator"


def trigger_of(cards: list, relics: list, potions: list, powers: list, pickup: bool = False, kill: bool = False) -> dict:
    """What a fight of the class must show for its pass to count (refset_score): one of the class's own cards
    played (the fillers count only when they are all the class has), its relic fired (a flash; for a pickup class,
    obtained and held at the start), its potion used, its power up when an enemy's attack lands (for a kill class:
    and a multi-hit attacker the power alone kills before its last hit died in that turn). A class with none of these
    has its trigger in CLASS_TRIGGERS; {"replay": True} is never evidence (refset_score.trigger_reached)."""
    own = [c for c in cards if c not in FILLER_CARDS] or list(cards)
    out = {}
    if own:
        out["plays"] = sorted(set(own))
    if relics:
        out["pickup" if pickup else "relic_flashes"] = sorted(set(relics))
    if potions:
        out["potions"] = sorted(set(potions))
    if powers:
        out["powers_hit"] = sorted({p for p, _, tg in powers if tg == "player"})
        if kill:
            out["attacker_killed_midattack"] = out["powers_hit"]
    return out or {"replay": True}


# ---------------------------------------------------------------- whole runs

def runs_part(stats: collections.Counter) -> dict:
    rng = random.Random("refset-v1-runs")
    # The run referee's own spec (referee/runs.py, driver/RunPlay.cs): seeded random legal choices on every screen.
    seeded = [{"kind": "run", "seed": _seed(rng), "character": "IRONCLAD", "ascension": 10, "chooser_seed": 500 + i,
               "buff_max_hp": 2000, "avoid": [], "stop_after_act": 3, "max_decisions": 1500, "potions": False}
              for i in range(30)]
    return {"codex": {"filter": {"character": "IRONCLAD", "ascension": 10, "build": "v0.107.1"},
                      "runs_in_dataset": stats.get("ironclad_a10_runs", 0),
                      "how": "EmuDump codex <dataset> <out.jsonl> --decompiled <root> --character IRONCLAD --ascension 10 "
                             "(feat/codex-replay); every run, every recorded offer"},
            "seeded": [{"id": f"W{i + 1:02d}-{s['seed']}", "part": "runs", "reason": "seeded whole A10 run for the run referee",
                        "spec": s} for i, s in enumerate(seeded)]}


# ---------------------------------------------------------------- build

def manifest_flags(entries: list) -> dict:
    """The realism flags the set's own specs carry before any recording, per part: specs, flagged, specs per flag
    cause (a spec counted once per cause), per act x room cell (realistic), and per flag."""
    def block(rows):
        return {"specs": len(rows), "flagged": sum(1 for e in rows if e.get("flags")),
                "by_cause": dict(collections.Counter(c for e in rows for c in sorted({f.split(":")[0] for f in e.get("flags", [])})))}
    out: dict = {}
    for part in dict.fromkeys(e["part"] for e in entries):
        es = [e for e in entries if e["part"] == part]
        out[part] = block(es)
        if part == "realistic":
            cells = collections.defaultdict(list)
            for e in es:
                cells[f"act{e['strata']['act']} {e['strata']['room']}"].append(e)
            out[part]["cells"] = {k: block(v) for k, v in sorted(cells.items())}
        out[part]["by_flag"] = dict(collections.Counter(f for e in es for f in e.get("flags", [])).most_common())
    return out


def print_manifest_flags(flags: dict) -> None:
    for part, b in flags.items():
        print(f"  {part}: {b['specs']} specs, {b['flagged']} flagged {b['by_cause'] or ''}")
        for cell, c in (b.get("cells") or {}).items():
            print(f"    {cell:<14} {c['specs']:>4} specs  {c['flagged']:>3} flagged {c['by_cause'] or ''}")


def pilot_ids(entries: list, n_real: int, n_cov: int, regression: bool = True) -> list[str]:
    real = [e for e in entries if e["part"] == "realistic"]
    cells = collections.defaultdict(list)
    for e in real:
        cells[(e["strata"]["act"], e["strata"]["room"])].append(e)
    pick = []
    total = len(real)
    for k in sorted(cells):
        share = max(1, round(len(cells[k]) * n_real / max(1, total)))
        pick += [e["id"] for e in cells[k][:share]]
    pick = pick[:n_real]
    pick += [e["id"] for e in entries if e["part"] == "coverage"][:n_cov]
    if regression:
        pick += [e["id"] for e in entries if e["part"] == "regression"]
    return pick


def cmd_build(a) -> int:
    out = vdir(a.version)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    t = load_tables(out)
    stats: collections.Counter = collections.Counter()
    reach = {"cards": set(), "relics": set(), "potions": set()}
    realistic = realistic_specs(Path(a.codex), t, a.realistic, stats, reach, a.max_runs)
    coverage = coverage_specs(t, reach)
    regression = regression_specs(t)
    runs = runs_part(stats)
    entries = realistic + coverage + regression
    seen = collections.Counter(e["spec"]["seed"] for e in entries)
    dup = {s for s, k in seen.items() if k > 1}
    if dup:
        # A seed names a result; a codex run seed met again (a coverage seed, an original fight) keeps the first.
        kept, have = [], set()
        for e in entries:
            if e["spec"]["seed"] in have:
                stats["dropped_duplicate_seed"] += 1
                continue
            have.add(e["spec"]["seed"])
            kept.append(e)
        entries = kept
    tg = coverage_targets(t, reach)
    manifest = {
        "set": SET_NAME, "version": a.version, "created": time.strftime("%Y-%m-%d %H:%M"),
        "generator": git_commit(ROOT), "max_actions": MAX_ACTIONS,
        "sources": {"codex": {"file": CODEX.name, "sha256": _codex_sha(Path(a.codex))},
                    "export_sha256": sha256_file(EXPORT), "setupinfo_sha256": sha256_file(out / "setupinfo.json")},
        "counts": dict(collections.Counter(e["part"] for e in entries)) | {"runs_seeded": len(runs["seeded"])},
        "coverage_targets": {"card_variants": len(tg["variants"]), "relics": len(tg["relics"]), "potions": len(tg["potions"]),
                             "encounters": len(tg["encounters"]), "plan": PLAN,
                             "lists": {"cards": [c + ("+" if u else "") for c, u in tg["variants"]], "relics": tg["relics"],
                                       "potions": tg["potions"], "encounters": tg["encounters"]}},
        "codex_stats": dict(stats), "runs": runs, "specs": entries,
        "pilot": pilot_ids(entries, a.pilot_realistic, a.pilot_coverage),
        "flags": manifest_flags(entries),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{SET_NAME} {a.version}: {manifest['counts']} in {time.time() - t0:.0f} s -> {out / 'manifest.json'}")
    print(f"  codex: {stats.get('ironclad_a10_runs', 0)} Ironclad A10 runs, {stats.get('candidates', 0)} usable fights; "
          + ", ".join(f"{k} {v}" for k, v in sorted(stats.items()) if k.startswith(("short_", "cand_skip", "skip_"))))
    print(f"  coverage targets: {manifest['coverage_targets']['card_variants']} card variants, {len(tg['relics'])} relics, "
          f"{len(tg['potions'])} potions, {len(tg['encounters'])} encounters; pilot {len(manifest['pilot'])} specs")
    print("  realism flags of the specs (before recording):")
    print_manifest_flags(manifest["flags"])
    return 0


def _codex_sha(p: Path) -> str:
    side = p.with_name(p.name + ".sha256")
    if side.exists():
        return side.read_text(encoding="utf-8").split()[0]
    return sha256_file(p)


# ---------------------------------------------------------------- record

def load_manifest(version: str) -> dict:
    return json.loads((vdir(version) / "manifest.json").read_text(encoding="utf-8"))


def select(man: dict, subset: str, parts: list[str]) -> list[dict]:
    ids = set(man["pilot"]) if subset == "pilot" else None
    return [e for e in man["specs"] if e["part"] in parts and (ids is None or e["id"] in ids)]


def host_error(r: dict | None) -> bool:
    """A full-host replay's error_log verdict (Nodeless.PlayFull "full host: N game error(s)", any error or first-chance
    exception), which a full pass does not write: the fight's errors are judged from the game log on both hosts."""
    return bool(r) and str(r.get("error") or "").startswith("full host: ") and bool(r.get("steps"))


def without_host_error(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "error"} if host_error(r) else r


def comparable(r: dict) -> dict:
    r = without_host_error(r)
    try:
        import serve  # noqa: PLC0415  (referee/serve.py)
        return serve.comparable(r)
    except (ImportError, AttributeError):
        skip = {"timing_ms", "phase_ms", "cost", "cost_setup", "_id", "_pid", "_t", "settle_audit", "idle_probe"}
        out = {k: v for k, v in r.items() if k not in skip}
        if isinstance(out.get("steps"), list):
            out["steps"] = [{k: v for k, v in s.items() if k not in ("cost", "trace")} for s in out["steps"]]
        return out


def first_difference(a: dict, b: dict) -> str:
    ca, cb = comparable(a), comparable(b)
    for k in sorted(set(ca) | set(cb)):
        if k == "steps":
            continue
        if ca.get(k) != cb.get(k):
            return f"field {k}"
    sa, sb = ca.get("steps", []), cb.get("steps", [])
    for i, (x, y) in enumerate(zip(sa, sb)):
        if x != y:
            keys = sorted(k for k in set(x.get("state", {})) | set(y.get("state", {}))
                          if x.get("state", {}).get(k) != y.get("state", {}).get(k))
            return f"step {i} ({'action' if x.get('action') != y.get('action') else ','.join(keys[:3])})"
    return f"step count {len(sa)} vs {len(sb)}" if len(sa) != len(sb) else "same"


def exclusion(entry: dict, r: dict | None) -> str | None:
    """Why a recording cannot be a reference (game-side trouble), or None."""
    if not r:
        return "no result (server produced none)"
    # A nodeless pass's full-host replays run with error_log (Nodeless.PlayFull): "full host: N game error(s)" for any
    # error or first-chance exception. A full pass judges a fight's errors from the game log (log_errors: a visual
    # fault is kept, the game's own code throwing excludes), so a replay is judged the same way, not by that line.
    if r.get("error") and not host_error(r):
        return "game error: " + str(r["error"]).splitlines()[0][:160]
    if r.get("seed") != entry["spec"]["seed"]:
        return f"result seed {r.get('seed')} is not the spec's"
    if not r.get("steps"):
        return "no steps"
    for it in r.get("setup_log", []):
        # (A residual deck logged ok by a driver older than deck_residual_fails fails here too.)
        if it.get("i", -1) >= 0 and (not it.get("ok") or str(it.get("msg", "")).startswith("residual:")):
            # A setup the game could not build (a deck residual, a load that does not hold what was asked, a
            # refused command) is not the fight the spec describes.
            return f"setup: item {it['i']} {it.get('cmd')} {it.get('id') or ''} failed: {it.get('msg', '')}"[:200]
        if it.get("idle_timeout"):
            return f"setup: item {it['i']} {it.get('cmd')} timed out"
    if any(it.get("cmd") == "fight" for it in entry["spec"].get("setup", [])) and not r.get("fight"):
        return "setup: the fight was not entered"
    want_act = entry.get("expect_act")
    if want_act and (r.get("fight") or {}).get("current_act") != want_act:
        return f"setup: the fight was met in {(r.get('fight') or {}).get('current_act')}, not {want_act}"
    if r.get("script_illegal"):
        return "scripted action illegal"
    if not r["steps"][0].get("state", {}).get("in_combat"):
        return "not in combat at step 0"
    return None


# Why a fight is not a reference, as counted in the record and the score: the first word of its exclusion. A fight
# that killed or stalled two servers (serve.py "stalled": a crash, a run-start exception or the driver's watchdog
# seeing no progress) is not a logged game error: its own cause.
CAUSES = (("setup:", "setup"), ("game error: stalled", "stalled"), ("game error: nodeless", "nodeless_host"),
          ("game error", "game_error"), ("the game logged an error", "game_error"),
          ("unstable", "unstable"), ("host difference", "host_difference"), ("after a faulted fight", "neighbour_fault"),
          ("no result", "no_result"), ("result seed", "no_result"), ("no steps", "no_result"), ("game log", "log_uncut"))
# The job keys of a pass on the stripped-down game (driver/Nodeless.cs: no NRun scene; a fight whose models need
# nodes, or whose nodeless attempt logged an error, is replayed on the full host; 5% of the others are audited there).
NODELESS_JOB = {"nodeless": True, "nodeless_on_error": "full", "nodeless_audit": 0.05}


def host_of(r: dict | None) -> str:
    """Which host a recording came from: "nodeless", "full" (a nodeless pass's fallback or a full pass), "" unknown."""
    return str(((r or {}).get("cost") or {}).get("host") or "")


def cause_of(excluded: str) -> str:
    if "idle_timeout" in excluded:
        return "driver_wait_limit"
    return next((c for prefix, c in CAUSES if excluded.startswith(prefix)), "other")


def realism(entry: dict, r: dict) -> dict:
    """How the recorded start compares with the spec: the setup's own check (player_load fails when the game does
    not hold what was asked, so a kept fight's holdings are the record's), the act the fight was met in, and
    whether the seed's Neow offered the codex run's pick (with player_load the pick only moves RNG and gold)."""
    log = [it for it in r.get("setup_log", []) if it.get("i", -1) >= 0]
    return {"setup_ok": all(it.get("ok") for it in log),
            "items": [it.get("cmd") for it in log],
            "fight_act": (r.get("fight") or {}).get("act"), "current_act": (r.get("fight") or {}).get("current_act"),
            "neow_match": (r.get("neow_options") or [None] * 9)[r.get("neow_pick", 0)] == entry.get("neow_relic")
            if entry.get("neow_relic") else None}


def _fight_of(r: dict) -> tuple[int | None, str] | None:
    """(the pass's spec number, role) of one server result line whose fight printed the driver's Neow line (its
    result holds "neow_options", Combat.cs prints both together), or None for a line with no Neow line in the log: a
    server's "_retire" line (Server.cs), a nodeless line sent to the full host before it played (Nodeless.Play: a
    fallback model in the spec) or cut short before Neow, a spec the server could not read. Roles: "plain" (the
    result), "attempt" (a nodeless line the pool replays on the full host: "_full"), "replay" (that full-host replay,
    id "<n>f")."""
    if "neow_options" not in r:
        return None
    rid = r.get("_id")
    if rid is None:
        return None, "plain"
    rid = str(rid)
    if rid.endswith("f"):
        return int(rid[:-1]), "replay"
    return int(rid), ("attempt" if r.get("_full") else "plain")


def log_errors(run_dir: Path, diag: dict | None = None) -> dict[int, list]:
    """The game's own [ERROR] lines per fight of one pass, by the fight's place in the pass (0-based, as serve.py
    numbers the specs: a result line's "_id"): each server's log is cut at the driver's Neow line, one per fight, in
    the order of the server's result lines. On a nodeless pass (driver/Nodeless.cs, the pool's full-host lane) a
    fight's errors are those of the recording that stands: a fallback's full-host replay ("<n>f"), an audited fight's
    nodeless one. The others (a fallback's nodeless attempt, an audit's full-host replay) go to diag
    {"nodeless_attempt" | "audit_replay": {n: errors}} when it is given."""
    out: dict[int, list] = {}
    servers = []
    why: dict[int, str] = {}  # spec number -> "fallback" | "audit", from the nodeless line the pool replayed
    for res in sorted((run_dir / "out").glob("s*.jsonl")):
        rows = [json.loads(x) for x in res.read_text(encoding="utf-8").splitlines() if x.strip()]
        for r in rows:
            if r.get("_full") and r.get("_id") is not None:
                why[int(str(r["_id"]).rstrip("f"))] = (r["_full"] or {}).get("why", "fallback")
        servers.append((res, [f for f in map(_fight_of, rows) if f is not None]))

    def put(i: int | None, role: str, errs: list) -> None:
        if i is None:
            return
        stands = role == "plain" or (role == "attempt" and why.get(i) == "audit") or (role == "replay" and why.get(i) != "audit")
        if stands:
            out[i] = errs
        elif diag is not None:
            diag.setdefault("nodeless_attempt" if role == "attempt" else "audit_replay", {})[i] = errs
    for res, fights in servers:
        log = run_dir / "scratch" / res.stem / "game-stdout.log"
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
        starts = [k for k, x in enumerate(lines) if x.startswith("[referee] neow options")]
        if len(starts) != len(fights):
            for i, role in fights:
                put(i, role, [["the game log cannot be cut into this server's fights", "", "uncut"]])
            continue
        before: set = set()
        for n, (i, role) in enumerate(fights):
            seg = lines[starts[n]:starts[n + 1] if n + 1 < len(starts) else len(lines)]
            errs = []
            for k, x in enumerate(seg):
                if x.startswith("[ERROR]"):
                    frame = next((y.strip()[3:] for y in seg[k + 1:k + 14] if y.strip().startswith("at MegaCrit.")), "")
                    # Visual-layer faults (a disposed node or material in a VFX or transition, headless) are kept
                    # and listed; anything thrown by the game's models, commands or combat code excludes the fight.
                    # The same fault as the previous fight's on this server is that fight's, leaking on (a faulted
                    # turn task is not cleared by the fast reset): this fight is retried, not excluded.
                    sig = (x[8:140], frame[:160])
                    # (The game spells one Nodes namespace "MegaCrit.sts2.Core.Nodes.TopBar": compare without case.)
                    errs.append([*sig, frame.lower().startswith("megacrit.sts2.core.nodes.") or ("leaked" if sig in before else False)])
            before = {(x[0], x[1]) for x in errs} | (before if any(x[2] == "leaked" for x in errs) else set())
            if errs:
                put(i, role, errs)
    return out


def steady_flashes(ra: dict, rb: dict) -> tuple[dict, list] | None:
    """Two recordings that differ only in their relic flashes ([step, relic], driver evidence, not game state; a
    flash raised from a pickup's deferred animation lands on whatever step the driver has reached): the recording
    with only the flashes both hold, and the ones dropped. None when anything else differs."""
    fa, fb = ra.get("relic_flashes") or [], rb.get("relic_flashes") or []
    if comparable({k: v for k, v in ra.items() if k != "relic_flashes"}) != comparable({k: v for k, v in rb.items() if k != "relic_flashes"}):
        return None
    left = [tuple(x) for x in fb]
    both, dropped = [], []
    for x in fa:
        if tuple(x) in left:
            left.remove(tuple(x))
            both.append(x)
        else:
            dropped.append(x)
    return dict(ra, relic_flashes=both), dropped + [list(x) for x in left]


def judge_pair(entries: list[dict], pa: dict, pb: dict) -> dict:
    """Per spec id: the kept recording, or why the fight is excluded (and whether a retry may clear it)."""
    out = {}
    for e in entries:
        (ra, ea), (rb, eb) = pa.get(e["id"], (None, None)), pb.get(e["id"], (None, None))
        logic = [x for x in (ea or []) + (eb or []) if x[2] is False]
        leaked = [x for x in (ea or []) + (eb or []) if x[2] == "leaked"]
        uncut = [x for x in (ea or []) + (eb or []) if x[2] == "uncut"]
        visual = sorted({x[1] or x[0] for x in (ea or []) + (eb or []) if x[2] is True})
        why = exclusion(e, ra) or exclusion(e, rb)
        j = {"visual_errors": visual} if visual else {}
        if logic:
            j["excluded"] = f"the game logged an error: {logic[0][0]} at {logic[0][1]}"[:240]
        elif leaked:
            j["excluded"] = f"after a faulted fight on its server: {leaked[0][0]} at {leaked[0][1]}"[:240]
            j["retry"] = True
        elif uncut:
            j["excluded"] = "game log: " + uncut[0][0]  # its errors cannot be told apart; a fresh server may
            j["retry"] = True
        elif why:
            j["excluded"] = why
            j["retry"] = "idle_timeout" in why
        elif comparable(ra) != comparable(rb):
            kept = steady_flashes(ra, rb)
            if kept is None and "nodeless" in (host_of(ra), host_of(rb)):
                # The full game and the stripped-down one disagree: a host difference, not the game's instability
                # until two more full-host recordings say so (passes c and d).
                j["excluded"] = "host difference: the full and the nodeless recordings differ at " + first_difference(ra, rb)
                j["retry"] = True
            elif kept is None:
                j["excluded"] = "unstable: the two recordings differ at " + first_difference(ra, rb)
            else:
                j["result"], j["unsteady_flashes"] = without_host_error(kept[0]), kept[1]
        else:
            j["result"] = without_host_error(ra)
        out[e["id"]] = j
    return out


def play_pass(specs: list[dict], game: str, servers: int, run_dir: Path, out: Path, timeout: float, build: Path,
              jopts: dict | None = None) -> tuple[list, dict]:
    """One pass through serve.py; jopts are job keys for every server (NODELESS_JOB: the stripped-down game)."""
    job = run_dir.with_suffix(".job.json")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    job.write_text(json.dumps({"runs": [e["spec"] for e in specs], **(jopts or {})}), encoding="utf-8")
    cmd = [sys.executable, str(HERE / "serve.py"), "--game", game, "--specs", str(job), "--servers", str(servers),
           "--spares", "0", "--run-dir", str(run_dir), "--out", str(out), "--timeout", str(int(timeout)), "--build", str(build)]
    if (jopts or {}).get("nodeless"):
        # A nodeless attempt that meets a fallback model inside a turn leaves the turn flags set, and its server
        # retires after the result (Driver.StaleState, exit 5): a planned exit, nothing re-queued, so at most one per
        # spec and one per full-host replay. serve.py counts retirements against its crash budget (3 x servers + 3),
        # which a full set's ~20% fallbacks exceed (22 retirements ended the first v2 record's nodeless pass at 791
        # of 1000), so the budget grows by those planned exits.
        cmd += ["--max-restarts", str(3 * servers + 3 + 2 * len(specs))]
    print(f"[refset] {run_dir.name}: {len(specs)} fights on {servers} servers", flush=True)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    (run_dir.with_suffix(".log")).write_text(p.stdout, encoding="utf-8")
    try:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        summary = {"tail": p.stdout[-1500:]}
    summary["exit"] = p.returncode
    got = json.loads(out.read_text(encoding="utf-8"))["results"] if out.exists() else []
    return [got[i] if i < len(got) else None for i in range(len(specs))], summary


def run_sessions(specs: list[dict], game: str, procs: int, rdir: Path, name: str, timeout: float, build: Path) -> tuple[list, list]:
    """Whole runs through run_referee.py sessions (the run referee plays one run after another in a session)."""
    procs = max(1, min(procs, len(specs)))
    running = []
    for p in range(procs):
        job = rdir / f"{name}.p{p}.json"
        job.write_text(json.dumps({"out": str(rdir / f"{name}.p{p}.out.json"), "fast_reset": True, "no_preload": True,
                                   "fast_patches": True, "stall_s": 90, "runs": [e["spec"] for e in specs[p::procs]]}), encoding="utf-8")
        running.append(subprocess.Popen([sys.executable, str(HERE / "run_referee.py"), "--game", game, "--job", str(job),
                                         "--timeout", str(int(timeout)), "--below-normal", "--build", str(build)],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
        time.sleep(3)  # the first session installs the mod
    reports = []
    for proc in running:
        text = proc.communicate()[0]
        try:
            reports.append(json.loads(text[text.rindex("{\n"):]))
        except ValueError:
            reports.append({"exit": proc.returncode, "tail": text[-800:]})
    merged: list = [None] * len(specs)
    for p in range(procs):
        path = rdir / f"{name}.p{p}.out.json"
        res = json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else []
        for k, r in enumerate(res):
            merged[p + k * procs] = r
    return merged, reports


def record_runs(man: dict, a, rdir: Path, stamp: dict, build: Path) -> int:
    specs = man["runs"]["seeded"][: a.limit or None]
    passes = []
    for name in ("runs-a", "runs-b"):
        res, reports = run_sessions(specs, a.game, a.servers, rdir, name, a.timeout, build)
        for r in reports:
            if r.get("network_connections") or r.get("real_profile_touched") is not False:
                print(f"[refset] SAFETY: {name}: {r}; no run references written", file=sys.stderr)
                return 1
        passes.append(res)
        stamp.setdefault("sts2_dll_sha256", (reports[0] if reports else {}).get("sts2_dll_sha256"))
    kept, index = [], []
    for e, ra, rb in zip(specs, *passes):
        row = {"id": e["id"], "part": "runs", "seed": e["spec"]["seed"]}
        why = None if ra and not ra.get("error") else "game error: " + str((ra or {}).get("error", "no result"))[:160]
        if not why and comparable(ra) != comparable(rb or {}):
            why = "unstable: the two recordings differ at " + first_difference(ra, rb or {})
        if why:
            row["excluded"] = why
        else:
            ra = dict(ra)
            ra["refset"] = {"id": e["id"], "part": "runs", "stamp": stamp}
            kept.append(ra)
        index.append(row)
    (rdir / "runs.refs.json").write_text(json.dumps({"refset": stamp, "results": kept}), encoding="utf-8")
    (rdir / "runs.record.json").write_text(json.dumps({"stamp": stamp, "index": index}, indent=1), encoding="utf-8")
    print(f"[refset] {len(specs)} whole runs recorded twice: {len(kept)} kept")
    return 0


# ---------------------------------------------------------------- the driver a reference may be recorded with

# What a recording needs from the driver (driver/Capabilities.cs names them; each is present only when its code is
# compiled in): the fixed settle and its Play-phase gate, the frame-counting watchdog, the console and reference
# setup items with the save-load path and the act entry, a deck residual failing its item, relic flashes (coverage
# and regression evidence) raised by the rules only (a display-raised flash lands on any step, full host only), the
# heuristic chooser of the realistic part, and the nodeless host's room-end rewards (generated at CombatWon, as the
# full host's NCombatUi does: without them the two hosts' last steps differ in the reward draws).
REQUIRED_CAPS = ("settle_flags", "play_phase_gate", "watchdog_v2", "console_setup", "refset_setup", "player_load",
                 "act_entry", "deck_residual_fails", "relic_flashes", "heuristic_chooser", "rule_flashes", "nodeless_room_end")
# The members each capability is read off (Capabilities.cs), for the check before any game starts.
CAP_MEMBERS = {"settle_flags": ("CheckSettleFields", "SettleMode"), "play_phase_gate": ("InPlayPhase",),
               "watchdog_v2": ("BudgetFrames", "Budget", "ArmWatchdog"), "console_setup": ("SetupItem", "EnterFight"),
               "refset_setup": ("RefsetItem",), "player_load": ("PlayerLoad",), "act_entry": ("EnterActItem",),
               "deck_residual_fails": ("DeckResidualFails",), "relic_flashes": ("RecordRelicFlashes",),
               "heuristic_chooser": ("ChooseHeuristic",), "rule_flashes": ("TakeDisplayFlashes", "FlashFrames"),
               "nodeless_room_end": ("OfferRoomEndRewardsNodeless",)}


def dll_stamp(dll: Path) -> dict:
    """The commit and the changed-sources list SpirelineReferee.csproj (target SpirelineStamp) wrote into the DLL as
    AssemblyMetadata attributes, read off the attribute blobs (a length-prefixed key, then the value)."""
    b = dll.read_bytes()

    def value(key: str) -> str | None:
        k = key.encode()
        i = b.find(bytes([len(k)]) + k)
        if i < 0:
            return None
        j = i + 1 + len(k)
        n = b[j]
        if n == 0xFF:
            return ""
        if n & 0x80 == 0:
            ln, j = n, j + 1
        elif n & 0xC0 == 0x80:
            ln, j = ((n & 0x3F) << 8) | b[j + 1], j + 2
        else:
            ln, j = ((n & 0x1F) << 24) | (b[j + 1] << 16) | (b[j + 2] << 8) | b[j + 3], j + 4
        return b[j:j + ln].decode("utf-8", "replace")
    return {"commit": value("spireline.commit"), "dirty": value("spireline.dirty")}


def static_caps(dll: Path) -> list[str]:
    """The capabilities whose members the DLL's metadata names (a preflight; the job-start line is the proof)."""
    b = dll.read_bytes()
    return [c for c, members in CAP_MEMBERS.items() if all(b"\x00" + m.encode() + b"\x00" in b for m in members)]


def driver_checkout(build: Path) -> Path:
    """The checkout whose referee/driver the build directory belongs to (<checkout>/referee/driver/bin/Release/net9.0)."""
    return build.parents[3] if len(build.parents) > 3 else ROOT


def driver_preflight(build: Path) -> tuple[list[str], dict]:
    """Why this driver build may not record references (empty: it may), and what it is. A reference is recorded
    only by a DLL built from the checkout's committed driver, with every REQUIRED_CAPS capability."""
    dll = build / "SpirelineReferee.dll"
    if not dll.exists():
        return [f"no driver DLL in {build}"], {}
    src = driver_checkout(build)

    def g(*args):
        return subprocess.run(["git", "-C", str(src), *args], capture_output=True, text=True, check=False).stdout.strip()
    head = g("rev-parse", "HEAD")
    changed = g("status", "--porcelain", "--untracked-files=all", "--", "driver/*.cs", "driver/*.csproj")
    st = dll_stamp(dll)
    caps = static_caps(dll)
    problems = []
    if not st["commit"]:
        problems.append("the DLL carries no commit stamp (built before SpirelineStamp): rebuild it")
    elif st["commit"] != head:
        problems.append(f"the DLL was built from {st['commit'][:12]}, the checkout is at {head[:12]}: rebuild it")
    if st["dirty"]:
        problems.append("the DLL was built from changed driver sources: " + st["dirty"][:200])
    if changed:
        problems.append("the driver's sources differ from the commit: " + " ".join(changed.split())[:200])
    missing = [c for c in REQUIRED_CAPS if c not in caps]
    if missing:
        problems.append("the driver lacks " + ", ".join(missing))
    return problems, {"commit": head, "dll_commit": st["commit"], "dll_dirty": bool(st["dirty"]), "static_capabilities": caps}


def driver_lines(run_dir: Path) -> list[dict]:
    """Every server's job-start line ("[referee] driver {...}", Capabilities.cs) in one pass's game logs."""
    out = []
    for log in sorted((run_dir / "scratch").glob("*/game-stdout.log")):
        for x in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if x.startswith("[referee] driver {"):
                try:
                    out.append(json.loads(x[len("[referee] driver "):]))
                except ValueError:
                    out.append({"unreadable": x[:200]})
    return out


def driver_runtime_problems(lines: list[dict], head: str) -> list[str]:
    """What the servers' own job-start lines say against recording: none seen, a different build, a missing
    capability, the old settle."""
    if not lines:
        return ["no server reported its driver at job start (a driver older than Capabilities.cs)"]
    problems = []
    for d in {json.dumps(x, sort_keys=True) for x in lines}:
        x = json.loads(d)
        if x.get("commit") != head or x.get("dirty"):
            problems.append(f"a server ran driver {str(x.get('commit'))[:12]} (dirty {x.get('dirty')}), not {head[:12]}")
        missing = [c for c in REQUIRED_CAPS if c not in (x.get("capabilities") or [])]
        if missing:
            problems.append("a server's driver lacks " + ", ".join(missing))
        if x.get("settle") != "flags":
            problems.append(f"a server settled with {x.get('settle')!r}, not the flag settle")
    return problems


# ---------------------------------------------------------------- record and judge

def pass_results(rdir: Path, name: str, order: list[dict], diag: dict | None = None) -> dict:
    """{spec id: (result, the game's errors)} of one recorded pass, results by their place in the pass; diag gets the
    errors of recordings that do not stand (log_errors), by spec id."""
    out = rdir / f"pass-{name}.out.json"
    got = json.loads(out.read_text(encoding="utf-8"))["results"] if out.exists() else []
    # serve.py's merged out file drops "cost"; its arrival file keeps each result's cost (the host it came from, a
    # fallback's reason, an audit's verdict), by the pass's spec number.
    arrival = rdir / f"pass-{name}" / "results.jsonl"
    costs = {}
    if arrival.exists():
        for line in arrival.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x = json.loads(line)
                if x.get("_id") is not None and x.get("cost") is not None:
                    costs[int(str(x["_id"]).rstrip("f"))] = x["cost"]
    got = [dict(r, cost=costs[i]) if isinstance(r, dict) and "cost" not in r and i in costs else r for i, r in enumerate(got)]
    d: dict = {}
    errs = log_errors(rdir / f"pass-{name}", d)
    if diag is not None:
        for kind, by_n in d.items():
            diag.setdefault(kind, {}).update({order[n]["id"]: v for n, v in by_n.items() if n < len(order)})
    return {e["id"]: (got[i] if i < len(got) else None, errs.get(i)) for i, e in enumerate(order)}


def host_report(entries: list[dict], passes: dict, judged: dict, hosts: dict, diag: dict) -> dict:
    """The full game against the stripped-down one, fight by fight (pass a against the nodeless pass): which host
    each nodeless-pass recording came from (fallbacks by reason, audits), how many fights are identical between the
    hosts, and every difference with its first differing step and what the full-host retries (c, d) found."""
    nl = next((n for n, h in hosts.items() if h == "nodeless"), None)
    full = next((n for n, h in hosts.items() if h == "full" and n in ("a", "b")), None)
    if not nl or not full:
        return {}
    by_host, fallbacks, audits = collections.Counter(), collections.Counter(), collections.Counter()
    same, same_nodeless, flashes_only, diffs, missing = 0, 0, 0, [], []
    for e in entries:
        rf, _ = passes[full].get(e["id"], (None, None))
        rn, _ = passes[nl].get(e["id"], (None, None))
        cost = (rn or {}).get("cost") or {}
        host = host_of(rn) or "none"
        by_host[host] += 1
        if cost.get("fallback"):
            fallbacks[str(cost["fallback"]).split(":")[0][:60]] += 1
        if isinstance(cost.get("audit"), dict):
            a = cost["audit"]
            audits["same" if a.get("same") else ("not replayed" if a.get("same") is None else "differ")] += 1
        if not rf or not rn or not rf.get("steps") or not rn.get("steps"):
            missing.append(e["id"])
            continue
        if comparable(rf) == comparable(rn):
            same += 1
            same_nodeless += host == "nodeless"
            continue
        if steady_flashes(rf, rn) is not None:
            flashes_only += 1
        j = judged.get(e["id"], {})
        rc = passes.get("c", {}).get(e["id"], (None, None))[0]
        diffs.append({"id": e["id"], "nodeless_host": host, "at": first_difference(rf, rn),
                      "flashes_only": steady_flashes(rf, rn) is not None,
                      "full_retry_same_as_a": (comparable(rc) == comparable(rf)) if rc and rc.get("steps") else None,
                      "verdict": j.get("excluded") or "kept"})
    compared = len(entries) - len(missing)
    return {"full_pass": full, "nodeless_pass": nl, "fights": len(entries), "nodeless_pass_hosts": dict(by_host),
            "fallbacks": dict(fallbacks), "audits": dict(audits), "compared": compared, "identical": same,
            "identical_on_nodeless": same_nodeless,
            "differ": len(diffs), "differ_flashes_only": flashes_only, "differences": diffs, "not_compared": missing,
            "nodeless_attempt_errors": {k: v[:3] for k, v in (diag.get("nodeless_attempt") or {}).items()},
            "audit_replay_errors": {k: v[:3] for k, v in (diag.get("audit_replay") or {}).items()}}


def judge_passes(entries: list[dict], passes: dict) -> tuple[dict, list[str]]:
    """The two passes a and b judged; a fight that hit a wait limit or a neighbour's fault is judged again on the
    retry passes c and d when they hold it (else it stays excluded, marked "retry not recorded")."""
    first = judge_pair(entries, passes["a"], passes["b"])
    retry = [e for e in entries if first[e["id"]].get("retry")]
    if retry:
        have = [e for e in retry if "c" in passes and e["id"] in passes["c"] and e["id"] in passes.get("d", {})]
        second = judge_pair(have, passes["c"], passes["d"]) if have else {}
        for e in retry:
            if e["id"] in second:
                second[e["id"]]["retried"] = first[e["id"]].get("excluded")
                first[e["id"]] = second[e["id"]]
            else:
                first[e["id"]]["excluded"] += " (retry not recorded)"
    return first, [e["id"] for e in retry]


def summarize(index: list[dict]) -> dict:
    """Per part and per act x room cell: specs, kept, excluded by cause, flagged (kept with a realism flag)."""
    def block(rows):
        kept = [r for r in rows if "excluded" not in r]
        return {"specs": len(rows), "kept": len(kept),
                "excluded": dict(collections.Counter(cause_of(r["excluded"]) for r in rows if "excluded" in r)),
                "flagged": sum(1 for r in kept if r.get("flags")),
                "flags": dict(collections.Counter(f.split(":")[0] for r in kept for f in r.get("flags", []))),
                "unsteady_flash_evidence": sum(1 for r in kept if r.get("unsteady_flashes"))}
    out = {"parts": {p: block([r for r in index if r["part"] == p]) for p in dict.fromkeys(r["part"] for r in index)}}
    cells = collections.defaultdict(list)
    for r in index:
        st = r.get("strata") or {}
        if r["part"] == "realistic":
            cells[f"act{st.get('act')} {st.get('room')}"].append(r)
    out["realistic_cells"] = {k: block(v) for k, v in sorted(cells.items())}
    out["flags_by_item"] = dict(collections.Counter(f for r in index if "excluded" not in r for f in r.get("flags", [])).most_common())
    return out


def build_index(entries: list[dict], judged: dict, stamp: dict) -> tuple[list[dict], dict]:
    index, refs = [], collections.defaultdict(list)
    for e in entries:
        j = judged[e["id"]]
        row = {"id": e["id"], "part": e["part"], "seed": e["spec"]["seed"], "chooser": e.get("chooser", e["spec"].get("chooser", "random")),
               "strata": e.get("strata", {}), "flags": list(e.get("flags", []))}
        for k in ("excluded", "retried", "visual_errors", "unsteady_flashes"):
            if j.get(k):
                row[k] = j[k]
        if not j.get("excluded"):
            ra = dict(j["result"])
            cost = ra.pop("cost", None) or {}  # (pass_results' diagnostics; a reference carries none, as before)
            if cost.get("host"):
                row["host"] = cost["host"]
            row["realism"] = realism(e, ra)
            ra["refset"] = {"id": e["id"], "part": e["part"], "stamp": stamp, "flags": row["flags"], "chooser": row["chooser"]}
            refs[e["part"]].append(ra)
        index.append(row)
    return index, refs


def write_record(rdir: Path, a, parts: list[str], entries: list[dict], judged: dict, retry: list, stamp: dict,
                 minutes: float, passes_summary: dict, latest_name: str | None, hosts: dict | None = None) -> dict:
    index, refs = build_index(entries, judged, stamp)
    for part in parts:
        (rdir / f"{part}.refs.json").write_text(json.dumps({"refset": stamp, "results": refs.get(part, [])}), encoding="utf-8")
    kept = sum(1 for r in index if "excluded" not in r)
    report = {"stamp": stamp, "minutes": round(minutes, 1), "fights": len(entries), "kept": kept,
              "summary": summarize(index), "excluded": [r for r in index if "excluded" in r], "index": index,
              "retried": len(retry), "passes": passes_summary, "hosts": hosts or {}}
    (rdir / "record.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if latest_name:
        latest = vdir(a.version) / latest_name
        latest.write_text(json.dumps({"record": str(rdir), "stamp": stamp, "parts": parts}, indent=1), encoding="utf-8")
    return report


def print_summary(report: dict) -> None:
    for part, b in report["summary"]["parts"].items():
        print(f"[refset] {part}: {b['specs']} specs, {b['kept']} kept, excluded {b['excluded'] or 0}, "
              f"{b['flagged']} kept with realism flags {b['flags'] or ''}")
    for cell, b in report["summary"]["realistic_cells"].items():
        print(f"[refset]   {cell:<14} {b['specs']:>4} specs {b['kept']:>4} kept  excluded {b['excluded'] or 0}  flagged {b['flagged']}")


def print_hosts(rep: dict) -> None:
    if not rep:
        return
    print(f"[refset] full vs nodeless: {rep['identical']}/{rep['compared']} fights identical between the hosts "
          f"({rep['identical_on_nodeless']} of them played nodeless; nodeless pass hosts {rep['nodeless_pass_hosts']}, "
          f"fallbacks {rep['fallbacks']}, audits {rep['audits']}); {rep['differ']} differ ({rep['differ_flashes_only']} only in "
          f"relic flashes); not compared {len(rep['not_compared'])}")
    for d in rep["differences"][:60]:
        print(f"  {d['id']} ({d['nodeless_host']}): {d['at']}; full retry same as a: {d['full_retry_same_as_a']}; {d['verdict']}"[:300])


def record_pointer(subset: str, ids: str, limit: int) -> str | None:
    """The file naming a subset's current record (refs-<subset>.json, what refset_score reads), or None for a
    trial (only some specs: --ids, --limit), which never replaces it."""
    return None if (ids or limit) else f"refs-{subset}.json"


def cmd_record(a) -> int:
    man = load_manifest(a.version)
    parts = a.parts.split(",")
    build = Path(a.build)
    problems, info = driver_preflight(build)
    if problems:
        print("[refset] REFUSED: this driver build may not record references:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    base_stamp = {"set": f"{SET_NAME} {a.version}", "driver": git_commit(driver_checkout(build)) | info,
                  "driver_dll_sha256": sha256_file(build / "SpirelineReferee.dll"), "tools": git_commit(ROOT),
                  "recorded": time.strftime("%Y-%m-%d %H:%M")}
    if "runs" in parts:
        label = a.label or time.strftime("%Y%m%d-%H%M%S")
        rdir = vdir(a.version) / "record" / f"runs-{label}"
        rdir.mkdir(parents=True, exist_ok=True)
        stamp = dict(base_stamp)
        rc = record_runs(man, a, rdir, stamp, build)
        if rc or parts == ["runs"]:
            (vdir(a.version) / "refs-runs.json").write_text(json.dumps({"record": str(rdir), "stamp": stamp}, indent=1), encoding="utf-8")
            return rc
        parts = [p for p in parts if p != "runs"]
    entries = select(man, a.subset, parts)
    if a.limit:
        entries = [e for part in parts for e in [x for x in entries if x["part"] == part][:a.limit]]
    if a.ids:
        wanted = set(a.ids.split(","))
        entries = [e for e in entries if e["id"] in wanted]
    label = a.label or time.strftime("%Y%m%d-%H%M%S")
    rdir = vdir(a.version) / "record" / f"{a.subset}-{label}"
    rdir.mkdir(parents=True, exist_ok=True)
    stamp = base_stamp | {"subset": a.subset, "label": label}
    # A trial (--ids, --limit) is not the subset's record: it never replaces refs-<subset>.json, which the score
    # reads; refset_score --record <dir> scores it by its own name.
    trial = record_pointer(a.subset, a.ids, a.limit) is None
    if trial:
        stamp["trial"] = {"specs": len(entries), "by_ids": bool(a.ids), "limit": a.limit or None}
    t0 = time.time()
    orders = {"a": entries, "b": list(reversed(entries))}
    summaries = {}
    # Which game each pass runs: the full game, or (--nodeless-pass) the stripped-down one for one of the two, so the
    # record also checks the stripped-down game against the full one fight by fight. Retries are on the full game.
    hosts = {n: ("nodeless" if a.nodeless_pass == n else "full") for n in ("a", "b", "c", "d")}
    stamp["hosts"] = {n: hosts[n] for n in ("a", "b")} | ({"nodeless_job": NODELESS_JOB} if a.nodeless_pass != "none" else {})
    # Two passes: in spec order, then reversed (other neighbours on other servers).
    for name in ("a", "b"):
        _, summaries[name] = play_pass(orders[name], a.game, a.servers, rdir / f"pass-{name}", rdir / f"pass-{name}.out.json",
                                       a.timeout, build, NODELESS_JOB if hosts[name] == "nodeless" else None)
    diag: dict = {}
    passes = {n: pass_results(rdir, n, orders[n], diag) for n in ("a", "b")}
    first = judge_pair(entries, passes["a"], passes["b"])
    # A fight that only hit a driver wait limit (idle_timeout), with no error of its own in the game's log, is
    # recorded twice more on fresh servers: the timeout is often a neighbour's doing (a fight whose game code threw
    # leaves its server unable to settle the next rooms). A fight whose own game code threw is not retried.
    retry = [e for e in entries if first[e["id"]].get("retry")]
    if retry:
        orders["c"], orders["d"] = retry, list(reversed(retry))
        for name in ("c", "d"):
            _, summaries[name] = play_pass(orders[name], a.game, a.servers, rdir / f"pass-{name}", rdir / f"pass-{name}.out.json",
                                           a.timeout, build)
            passes[name] = pass_results(rdir, name, orders[name])
    (rdir / "orders.json").write_text(json.dumps({n: [e["id"] for e in o] for n, o in orders.items()}), encoding="utf-8")
    judged, retried = judge_passes(entries, passes)
    hosts_rep = host_report(entries, passes, first, hosts, diag)
    minutes = (time.time() - t0) / 60
    for name, summ in summaries.items():
        if summ.get("network_connections") or summ.get("real_profile_touched") is not False:
            print(f"[refset] SAFETY: pass {name}: network {summ.get('network_connections')} real profile touched "
                  f"{summ.get('real_profile_touched')}; no references written", file=sys.stderr)
            return 1
    builds = {summ.get("sts2_dll_sha256") for summ in summaries.values()}
    if len(builds) != 1:
        print(f"[refset] the passes ran different game builds {builds}", file=sys.stderr)
        return 1
    stamp["sts2_dll_sha256"] = builds.pop()
    lines = [x for n in summaries for x in driver_lines(rdir / f"pass-{n}")]
    bad = driver_runtime_problems(lines, info["commit"])
    if bad:
        print("[refset] REFUSED after recording: the servers' drivers may not record references; none written:\n  "
              + "\n  ".join(bad), file=sys.stderr)
        return 2
    stamp["driver_build"] = lines[0]
    report = write_record(rdir, a, parts, entries, judged, retried, stamp, minutes,
                          {n: {k: summ.get(k) for k in ("fights", "wall_s", "s_per_fight_per_server", "fights_per_min", "restarts",
                                                        "retired", "errors", "full_lane", "full_replays", "audit_diffs", "exit")}
                           for n, summ in summaries.items()},
                          record_pointer(a.subset, a.ids, a.limit), hosts_rep)
    print(f"[refset] {len(entries)} fights recorded twice in {minutes:.1f} min ({len(retried)} retried after a wait limit), "
          f"driver {info['commit'][:12]} {','.join(lines[0].get('capabilities', []))}")
    print_summary(report)
    print_hosts(hosts_rep)
    print(f"[refset] references: {rdir}" + (" (a trial: refset_score --record <dir> scores it)" if trial else ""))
    return 0


def cmd_rejudge(a) -> int:
    """Judge a recorded record directory again from its saved passes (no game is started), e.g. after a fix to the
    judging; writes record.rejudged.json and lists every fight whose verdict changed."""
    man = load_manifest(a.version)
    rdir = Path(a.record)
    old = json.loads((rdir / "record.json").read_text(encoding="utf-8"))
    by_seed = {e["spec"]["seed"]: e for e in man["specs"]}
    orders = {}
    for n in ("a", "b", "c", "d"):
        job = rdir / f"pass-{n}.job.json"
        if job.exists():
            orders[n] = [by_seed[s["seed"]] for s in json.loads(job.read_text(encoding="utf-8"))["runs"]]
    entries = [by_seed[r["seed"]] for r in old["index"]]
    diag: dict = {}
    passes = {n: pass_results(rdir, n, o, diag) for n, o in orders.items()}
    judged, retried = judge_passes(entries, passes)
    hosts = {n: "full" for n in ("a", "b", "c", "d")} | (old["stamp"].get("hosts") or {})
    hosts_rep = host_report(entries, passes, judge_pair(entries, passes["a"], passes["b"]), hosts, diag)
    stamp = dict(old["stamp"]) | {"rejudged": time.strftime("%Y-%m-%d %H:%M"), "rejudged_by": git_commit(ROOT)}
    index, _ = build_index(entries, judged, stamp)
    before = {r["id"]: r.get("excluded") for r in old["index"]}
    changed = [{"id": r["id"], "was": before.get(r["id"]), "now": r.get("excluded")} for r in index
               if (before.get(r["id"]) or None) != (r.get("excluded") or None)]
    report = {"stamp": stamp, "fights": len(entries), "kept": sum(1 for r in index if "excluded" not in r),
              "summary": summarize(index), "excluded": [r for r in index if "excluded" in r], "index": index,
              "changed": changed, "hosts": hosts_rep}
    (rdir / "record.rejudged.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if a.write:
        # The record itself, judged again: references and record.json rewritten from the same passes (the first
        # judgement kept as record.first.json), the stamp naming the rejudging commit.
        first = rdir / "record.first.json"
        if not first.exists():
            first.write_text(json.dumps(old, indent=1), encoding="utf-8")
        parts = list(dict.fromkeys(e["part"] for e in entries))
        write_record(rdir, a, parts, entries, judged, retried, stamp, old.get("minutes", 0), old.get("passes", {}), None, hosts_rep)
        print(f"[refset] {rdir / 'record.json'} and its references rewritten")
    print(f"[refset] {rdir.name} judged again: {old['kept']} -> {report['kept']} kept of {len(entries)}; {len(changed)} verdicts changed")
    print_hosts(hosts_rep)
    for c in changed:
        print(f"  {c['id']}: was {c['was'] or 'kept'} | now {c['now'] or 'kept'}"[:300])
    print_summary(report)
    return 0


def cmd_status(a) -> int:
    man = load_manifest(a.version)
    print(f"{man['set']} {man['version']} ({man['created']}, generator {man['generator']['commit'][:12]}): {man['counts']}")
    print("  realism flags of the specs (before recording):")
    print_manifest_flags(manifest_flags(man["specs"]))
    for sub in ("all", "pilot"):
        p = vdir(a.version) / f"refs-{sub}.json"
        if p.exists():
            rec = json.loads((Path(json.loads(p.read_text(encoding='utf-8'))["record"]) / "record.json").read_text(encoding="utf-8"))
            print(f"  {sub}: {rec['kept']} of {rec['fights']} kept ({rec['stamp']['recorded']}, driver {rec['stamp']['driver']['commit'][:9]})")
            if rec.get("summary"):
                print_summary(rec)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--version", default="v2")
    b.add_argument("--codex", default=str(CODEX))
    b.add_argument("--realistic", type=int, default=600)
    b.add_argument("--max-runs", type=int, default=0, help="stop reading the codex after this many runs (a quick trial)")
    b.add_argument("--pilot-realistic", type=int, default=60)
    b.add_argument("--pilot-coverage", type=int, default=60)
    r = sub.add_parser("record")
    r.add_argument("--version", default="v2")
    r.add_argument("--subset", choices=("all", "pilot"), default="all")
    r.add_argument("--parts", default=",".join(PARTS), help="realistic,coverage,regression and/or runs (whole runs: needs the run referee driver)")
    r.add_argument("--game", default=GAME)
    r.add_argument("--servers", type=int, default=3, help="game processes (at most 6 on this PC)")
    r.add_argument("--nodeless-pass", choices=("none", "a", "b"), default="none",
                   help="record this pass on the stripped-down game (NODELESS_JOB), the other on the full game")
    r.add_argument("--build", default=str(DRIVER_BUILD))
    r.add_argument("--timeout", type=float, default=6 * 3600)
    r.add_argument("--label", default=None)
    r.add_argument("--limit", type=int, default=0, help="only the first N specs of each part (a trial)")
    r.add_argument("--ids", default="", help="only these spec ids, comma-separated (a trial)")
    s = sub.add_parser("status")
    s.add_argument("--version", default="v2")
    j = sub.add_parser("rejudge", help="judge a record directory again from its saved passes (no game)")
    j.add_argument("--version", default="v2")
    j.add_argument("--record", required=True, help="the record directory (record/<subset>-<label>)")
    j.add_argument("--write", action="store_true", help="rewrite the record's references and record.json from the new judgement")
    a = ap.parse_args()
    if a.cmd == "record" and a.servers > 6:
        ap.error("at most 6 game processes")
    return {"build": cmd_build, "record": cmd_record, "status": cmd_status, "rejudge": cmd_rejudge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
