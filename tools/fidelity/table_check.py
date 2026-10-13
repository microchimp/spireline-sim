"""Compare the game's content tables with the emulator's.

    python -m tools.fidelity.table_check [--game-export <export.json>] [--emu-json <tables.json>]

The game side comes from the referee's "export" job (job kind "export", see referee/RUNBOOK.md):
pool contents in the order the game enumerates them, with rarity, type, cost and target. The
emulator side is `EmuDump tables` (build tools/emudump first) plus the per-act event pools, which
are local arrays in Core/Run/RunMapGenerator.cs and are read from that file. Game data stays
local: the export is never committed.

Reports, per section: entries missing from or extra in the emulator, the first index where the
common entries diverge in order, and field mismatches (type, rarity, cost, target, upgradable).
Differences are reported, never fixed. Exit status 1 if there are any.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = Path("D:/spireline/referee/exports/v0.107.1.json")
EMUDUMP = ROOT / "tools/emudump/bin/Release/net9.0/EmuDump.dll"
MAP_GEN = ROOT / "src/Sts2Emulator/Core/Run/RunMapGenerator.cs"

# Emulator encounter names that differ from the game's model names only in spelling. Matching is
# on the name minus the _WEAK/_NORMAL/_ELITE/_BOSS suffix, ignoring underscores and case; these
# map the normalised emulator key onto the game's.
ENCOUNTER_ALIASES = {
    "CHOMPER": "CHOMPERS",
    "OBSCURA": "THEOBSCURA",
    "AXEBOT": "AXEBOTS",
    "EXOSKELETON": "EXOSKELETONS",
    "SCROLLS": "SCROLLSOFBITING",
    "LOSTANDFORGOTTEN": "THELOSTANDFORGOTTEN",
}

lines: list[str] = []
counts: dict[str, int] = {}


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).upper()


def report(section: str, msg: str) -> None:
    counts[section] = counts.get(section, 0) + 1
    lines.append(f"  [{section}] {msg}")


def compare_lists(section: str, label: str, game: list[str], emu: list[str], limit: int = 6) -> None:
    """Missing, extra, duplicates, and the first order divergence among the common entries."""
    gs, es = set(game), set(emu)
    missing = [x for x in game if x not in es]
    extra = [x for x in emu if x not in gs]
    if missing:
        report(section, f"{label}: {len(missing)} in game, not in emulator: {', '.join(missing[:limit])}"
               + (" ..." if len(missing) > limit else ""))
    if extra:
        report(section, f"{label}: {len(extra)} in emulator, not in game: {', '.join(extra[:limit])}"
               + (" ..." if len(extra) > limit else ""))
    gc = [x for x in game if x in es]
    ec = [x for x in emu if x in gs]
    if gc != ec:
        i = next((k for k, (a, b) in enumerate(zip(gc, ec)) if a != b), min(len(gc), len(ec)))
        moved = sum(1 for a, b in zip(gc, ec) if a != b)
        report(section, f"{label}: ORDER differs, first at common index {i} (game {gc[i:i+3]} vs "
               f"emulator {ec[i:i+3]}); {moved} of {len(gc)} positions differ")
    if len(emu) != len(set(emu)):
        report(section, f"{label}: emulator list has duplicates")


def norm_enc(name: str, emulator: bool = False) -> str:
    n = re.sub(r"_(WEAK|NORMAL|ELITE|BOSS)$", "", name).replace("_", "")
    return ENCOUNTER_ALIASES.get(n, n) if emulator else n


def parse_event_pools(src: str) -> dict[str, list[str]]:
    """The emulator's per-act event pools (before the shuffle), from RunMapGenerator.cs."""
    shared_m = re.search(r"SharedEventPool\s*=\s*\[(.*?)\];", src, re.S)
    shared = [snake(m) for m in re.findall(r"RunConstants\.Event(\w+)", shared_m.group(1))] if shared_m else []
    start = src.index("int[] GenerateEventSequence")
    end = re.search(r"\n    (?:private|public|internal) static", src[start + 10:])
    body = src[start: start + 10 + (end.start() if end else 16000)]
    pools = []
    for m in re.finditer(r"\[\s*((?:RunConstants\.Event\w+,\s*|\.\. SharedEventPool,?\s*)+)\]", body):
        items: list[str] = []
        for tok in re.findall(r"RunConstants\.Event\w+|\.\. SharedEventPool", m.group(1)):
            items += shared if tok.startswith("..") else [snake(tok[len("RunConstants.Event"):])]
        pools.append(items)
    out = {"SHARED": shared}
    # In source order: Hive, Glory, Underdocks, Overgrowth.
    if len(pools) == 4:
        out.update(HIVE=pools[0], GLORY=pools[1], UNDERDOCKS=pools[2], OVERGROWTH=pools[3])
    else:
        report("events", f"could not parse the emulator's event pools (found {len(pools)} lists)")
    return out


def check_field(section: str, what: str, name: str, g, e) -> None:
    if g != e:
        report(section, f"{what} {name}: game {g!r} vs emulator {e!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--game-export", type=Path, default=DEFAULT_EXPORT)
    ap.add_argument("--emu-json", type=Path, help="EmuDump tables output; default: run EmuDump")
    a = ap.parse_args()

    game = json.loads(a.game_export.read_text(encoding="utf-8"))["results"][0]
    if a.emu_json:
        emu = json.loads(a.emu_json.read_text(encoding="utf-8"))
    else:
        emu = json.loads(subprocess.run(["dotnet", str(EMUDUMP), "tables"], capture_output=True, text=True,
                                        check=True).stdout)
    for err in game.get("errors", []):
        print("game export error:", err)

    # Card pools, in pool order, then fields of every card the game has in a compared pool.
    gcards: dict[str, dict] = {}
    for pool, emu_ids in emu["card_pools"].items():
        gp = game["card_pools"][pool]
        compare_lists("card_pools", pool, [c["id"] for c in gp["all"]], emu_ids)
        # A single-player run draws GetUnlockedCards(all, SingleplayerOnly): AllCards minus the
        # multiplayer-only cards. Check the emulator drops the same ones (by its flag).
        dropped = [c["id"] for c in gp["all"] if c["id"] not in set(gp["unlocked"])]
        emu_mp = [i for i in emu_ids if emu["cards"][i]["multiplayer_only"]]
        if dropped != emu_mp:
            report("card_pools", f"{pool}: single-player filter drops {len(dropped)} in game "
                   f"({', '.join(dropped[:4])}) vs {len(emu_mp)} flagged multiplayer_only in emulator "
                   f"({', '.join(emu_mp[:4])})")
        for c in gp["all"]:
            gcards[c["id"]] = c
    skipped = [p for p in game["card_pools"] if p not in emu["card_pools"]]
    skipped += [p for k in ("relic", "potion") for p in game[f"{k}_pools"] if p not in emu[f"{k}_pools"]]
    for cid, g in gcards.items():
        e = emu["cards"].get(cid)
        if e is None:
            continue
        check_field("card_fields", "type", cid, g["type"], e["type"])
        check_field("card_fields", "rarity", cid, g["rarity"], e["rarity"])
        check_field("card_fields", "target", cid, g["target"], e["target"])
        check_field("card_fields", "cost_x", cid, g["cost_x"] == "True", e["cost_x"])
        if g["cost_x"] != "True":
            check_field("card_fields", "cost", cid, int(g["cost"]), e["cost"])
        check_field("card_fields", "star_cost", cid, int(g["star_cost"]), e["star_cost"])
        check_field("card_fields", "upgradable", cid, int(g["max_upgrade_level"]) > 0, e["upgradable"])
        check_field("card_fields", "multiplayer_only", cid, g["multiplayer_constraint"] == "MultiplayerOnly",
                    e["multiplayer_only"])

    # Relic and potion pools, with rarity.
    for kind in ("relic", "potion"):
        grefs: dict[str, dict] = {}
        for pool, emu_ids in emu[f"{kind}_pools"].items():
            gp = game[f"{kind}_pools"][pool]
            compare_lists(f"{kind}_pools", pool, [r["id"] for r in gp["all"]], emu_ids)
            for r in gp["all"]:
                grefs[r["id"]] = r
        for rid, g in grefs.items():
            e = emu[f"{kind}s"].get(rid)
            if e is None:
                continue
            check_field(f"{kind}_fields", "rarity", rid, g["rarity"], e["rarity"])
            if kind == "potion":
                check_field("potion_fields", "target", rid, g["target"], e["target"])

    # Acts: index grouping and the four encounter lists.
    if game["acts_by_index"] != emu["acts_by_index"]:
        report("acts", f"ActsByIndex: game {game['acts_by_index']} vs emulator {emu['acts_by_index']}")
    for act, ea in emu["acts"].items():
        ga = game["acts"][act]
        for kind in ("weak", "normal", "elite", "boss"):
            compare_lists("encounters", f"{act}.{kind}", [norm_enc(x) for x in ga[kind]],
                          [norm_enc(snake(x), True) for x in ea[kind]])

    # Events: each act's own list followed by ModelDb.AllSharedEvents, as ActModel.GenerateRooms builds it.
    pools = parse_event_pools(MAP_GEN.read_text(encoding="utf-8"))
    compare_lists("events", "AllSharedEvents", game["shared_events"], pools.get("SHARED", []))
    for act, ga in game["acts"].items():
        if act in pools:
            compare_lists("events", f"{act} pool (AllEvents + AllSharedEvents)",
                          ga["events"] + game["shared_events"], pools[act])

    # Neow's AllPossibleOptions, by relic.
    compare_lists("neow", "AllPossibleOptions relics", [o["relic"] for o in game["neow_options"]],
                  emu["neow_options"])

    print("Table check: game export vs emulator tables")
    print(f"  game export: {a.game_export.name}; cards in compared pools: {len(gcards)}")
    if skipped:
        print(f"  pools with no emulator table (not compared): {', '.join(skipped)}")
    for section in ("card_pools", "card_fields", "relic_pools", "relic_fields", "potion_pools", "potion_fields",
                    "acts", "encounters", "events", "neow"):
        print(f"{section}: {counts.get(section, 0)} difference(s)")
        for ln in lines:
            if ln.startswith(f"  [{section}]"):
                print(ln)
    total = sum(counts.values())
    print(f"TOTAL: {total} difference(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
