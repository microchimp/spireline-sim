#!/usr/bin/env python3
"""Ironclad A10 reference set: one versioned set of game recordings that every accuracy number is quoted from.

    python -I referee/refset.py build  [--version v1] [--realistic 600] [--codex <runs.jsonl.gz>]
    python -I referee/refset.py record [--version v1] [--subset all|pilot] [--parts realistic,coverage,regression]
                                       [--game D:/spireline/referee/game-refset] [--servers 3] [--label <name>]
    python -I referee/refset.py status [--version v1]
    python -m tools.fidelity.refset_score --version v1 [--subset pilot] --emu <emulator checkout>   (the numbers)

Run from the fork's root, with -I: the Spire Codex history is third-party data, and -I keeps the script's
and the current directory off sys.path while it is read. Everything this writes is local, under
D:/spireline/referee/refset/<version>/ (manifest, recordings, scores), never in the repo: realistic specs
are derived from per-run data.

The set (manifest.json, one entry per spec with the reason it is in the set):
  realistic   fights from real Ironclad A10 runs (Spire Codex, v0.107.1): the run's own seed and Neow pick,
              then its deck (upgrades, enchantments), relics, potions and HP as they stood before that floor's
              fight, then that floor's encounter. Stratified by act, room type (normal/elite/boss) and the
              floor's place in the act; one fight per run. The headline numbers.
  coverage    the fewest plausible setups that put every Ironclad and colourless card (base and upgraded) in
              a deck often enough to be played 5 times, meet every act 1-3 encounter 3 times, and hold every
              Ironclad-reachable relic and potion the emulator models 3 times (planned; refset_score counts
              what the recordings really reached).
  regression  one minimal setup per known divergence class (FINDINGS.md, triage files, the setup fights'
              shrunk cases), plus the original fight that found it, re-recorded with today's driver.
  runs        whole runs: every Ironclad A10 codex run (the codex replay, run-level offers) and 30 seeded
              runs for the run referee (driver/RunPlay.cs; recorded once that driver is integrated).

Setup goes through referee/driver/Setup.cs and RefsetSetup.cs: the game's own commands only (fight, relics,
deck cards incl. upgrades and enchantments, potions, HP); never godmode, kill, win, damage or heal shortcuts.

Recording (record): every fight is recorded twice (the second pass in reverse order, on fresh servers); a fight
whose two recordings differ is excluded as an unstable reference, as is any fight whose game side logged an
error (driver error or idle timeout, a setup item the game refused, no fight). Each kept recording is stamped
with the set version, the game build (sts2.dll sha256) and the driver commit and DLL hash. The safety checks are
serve.py's (known sts2.dll, no network, scratch profile, real profile untouched); a breach aborts the record.
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
            "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}


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


def _ck(c: dict) -> list:
    """[id, upgrade level, enchantment, amount, floor added] of a history card."""
    e = c.get("enchantment") or {}
    return [bare(c.get("id")), int(c.get("current_upgrade_level", 0) or 0), bare(e.get("id")), int(e.get("amount", 0) or 0),
            int(c.get("floor_added_to_deck") or 1)]


def deck_before(d: dict, pts: list, F: int) -> tuple[list, bool]:
    """The deck as it stood before floor F's fight, rebuilt back from the final deck; and whether every step
    of the rebuild was unambiguous."""
    exact = True
    p = d["players"][0]
    cards = [_ck(c) for c in p["deck"] if int(c.get("floor_added_to_deck") or 1) < F]
    later_added = [_ck(c) for c in p["deck"] if int(c.get("floor_added_to_deck") or 1) >= F]
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
    p = d["players"][0]
    held = [(int(r.get("floor_added_to_deck") or 1), bare(r["id"])) for r in p["relics"] if int(r.get("floor_added_to_deck") or 1) < F]
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
                held.append((max(got), rid))
    return [r for _, r in sorted(held, key=lambda x: x[0])]


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
        yield (cand["act"], room), bucket, h("refset-v1", d.get("run_hash"), F), cand


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
    spec = {"kind": "combat", "seed": c["seed"], "character": "IRONCLAD", "ascension": 10, "neow": c["neow"],
            "chooser_seed": int(hh[:8], 16) % 1000003, "max_actions": MAX_ACTIONS, "use_potions": True,
            "setup": setup_items(c["relics"], c["deck"], c["hp"], c["max_hp"], c["potions"], c["encounter"]),
            "tag": f"realistic:act{c['act']}:{c['room']}"}
    _, kind, _ = t.encounters[c["encounter"]]
    return {"id": f"R{c['act']}{c['room'][0].upper()}-{c['seed']}-F{c['floor']:02d}", "part": "realistic",
            "strata": {"act": c["act"], "room": c["room"], "kind": kind, "floor": c["floor"], "act_name": c["act_name"]},
            "reason": (f"real Ironclad A10 run (codex {str(c['run'])[:8]}), floor {c['floor']} (act {c['act']} {c['room']}, "
                       f"{c['encounter']}): {len(c['deck'])} cards, {len(c['relics'])} relics, {len(c['potions'])} potions, "
                       f"HP {c['hp']}/{c['max_hp']}"),
            "exact": c["exact"], "neow_relic": c["neow_relic"], "codex_monsters": c["monsters"], "spec": spec}


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
        spec = {"kind": "combat", "seed": _seed(rng), "character": "IRONCLAD", "ascension": 10, "neow": n % 3,
                "chooser_seed": 9000 + n, "max_actions": MAX_ACTIONS, "use_potions": True,
                "setup": setup_items(starter + relics, deck, hp, max_hp, potions, enc), "tag": f"coverage:act{act}:{kind}"}
        out.append({"id": f"C{n + 1:04d}-{spec['seed']}", "part": "coverage",
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
    ("hook order: the player's Thorns before an enemy hit lands (open)", "FINDINGS 2026-10-13", ["DEFEND_IRONCLAD"], [], [], "NIBBITS_NORMAL", [("THORNS_POWER", 3, "player")], []),
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
            spec = {"kind": "combat", "seed": _seed(rng), "character": "IRONCLAD", "ascension": 10, "neow": n % 3,
                    "chooser_seed": 7700 + n, "max_actions": MAX_ACTIONS, "use_potions": True, "setup": items,
                    "tag": f"regression:{name}"}
            out.append({"id": f"G{n + 1:03d}-{spec['seed']}", "part": "regression", "class": name, "source": source,
                        "reason": f"minimal setup for the divergence class '{name}' ({source})",
                        "flags": ["cross_character"] if cross else [], "spec": spec})
        for s in seeds:
            got = find_original(s, cache)
            if not got:
                continue
            spec, src = got
            spec = {k: v for k, v in spec.items() if k not in ("out",)}
            spec.setdefault("max_actions", MAX_ACTIONS)
            spec["tag"] = f"regression-original:{name}"
            out.append({"id": f"G{n + 1:03d}o-{s}", "part": "regression", "class": name, "source": f"{source}; original {src}",
                        "reason": f"the fight that found '{name}', re-recorded with the fixed driver", "flags": [], "spec": spec})
    return out


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
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{SET_NAME} {a.version}: {manifest['counts']} in {time.time() - t0:.0f} s -> {out / 'manifest.json'}")
    print(f"  codex: {stats.get('ironclad_a10_runs', 0)} Ironclad A10 runs, {stats.get('candidates', 0)} usable fights; "
          + ", ".join(f"{k} {v}" for k, v in sorted(stats.items()) if k.startswith(("short_", "cand_skip", "skip_"))))
    print(f"  coverage targets: {manifest['coverage_targets']['card_variants']} card variants, {len(tg['relics'])} relics, "
          f"{len(tg['potions'])} potions, {len(tg['encounters'])} encounters; pilot {len(manifest['pilot'])} specs")
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


def comparable(r: dict) -> dict:
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
    if r.get("error"):
        return "game error: " + str(r["error"]).splitlines()[0][:160]
    if r.get("seed") != entry["spec"]["seed"]:
        return f"result seed {r.get('seed')} is not the spec's"
    if not r.get("steps"):
        return "no steps"
    for it in r.get("setup_log", []):
        if it.get("i", -1) >= 0 and not it.get("ok"):
            return f"setup item {it['i']} {it.get('cmd')} {it.get('id') or ''} refused: {it.get('msg', '')}"[:200]
        if it.get("idle_timeout"):
            return f"setup item {it['i']} {it.get('cmd')} timed out"
    if any(it.get("cmd") == "fight" for it in entry["spec"].get("setup", [])) and not r.get("fight"):
        return "the setup's fight was not entered"
    if r.get("script_illegal"):
        return "scripted action illegal"
    if not r["steps"][0].get("state", {}).get("in_combat"):
        return "not in combat at step 0"
    return None


def realism(entry: dict, r: dict) -> dict:
    """How closely the recorded fight start matches the state the spec asked for (deck, relics, potions, HP)."""
    setup = {it["cmd"]: it for it in entry["spec"].get("setup", [])}
    st = r["steps"][0]["state"]
    piles = collections.Counter(tuple(c[:2]) for k in ("hand", "draw", "discard", "exhaust") for c in st.get(k, []))
    want = collections.Counter((c["id"], bool(c.get("upgraded"))) for c in setup.get("deck_set", {}).get("cards", []))
    relics_want = collections.Counter(setup.get("relics_set", {}).get("ids", []))
    out = {"deck": piles == want, "deck_extra": sum((piles - want).values()), "deck_missing": sum((want - piles).values()),
           "relics": collections.Counter(st.get("relics", [])) == relics_want if "relics" in st else None,
           "potions": sorted(p for p in st.get("potions", []) if p) == sorted(setup.get("potions_set", {}).get("ids", []))
           if "potions" in st else None,
           "max_hp": st.get("player", {}).get("max_hp") == setup.get("hp", {}).get("max"),
           "neow_match": (r.get("neow_options") or [None] * 9)[r.get("neow_pick", 0)] == entry.get("neow_relic")
           if entry.get("neow_relic") else None,
           "fight_act": (r.get("fight") or {}).get("act"), "current_act": (r.get("fight") or {}).get("current_act")}
    return out


def log_errors(run_dir: Path) -> dict[int, list]:
    """The game's own [ERROR] lines per fight of one pass, by the fight's place in the pass (0-based): each server's
    log is cut at the driver's Neow line, one per fight, in the order of the server's result lines."""
    out: dict[int, list] = {}
    for res in sorted((run_dir / "out").glob("s*.jsonl")):
        ids = []
        for line in res.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.append(int(json.loads(line).get("_id", "0")) - 1)
        log = run_dir / "scratch" / res.stem / "game-stdout.log"
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
        starts = [k for k, x in enumerate(lines) if x.startswith("[referee] neow options")]
        if len(starts) != len(ids):
            for i in ids:
                out[i] = [["the game log cannot be cut into this server's fights", "", False]]
            continue
        before: set = set()
        for n, i in enumerate(ids):
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
                    errs.append([*sig, frame.startswith("MegaCrit.Sts2.Core.Nodes.") or ("leaked" if sig in before else False)])
            before = {(x[0], x[1]) for x in errs} | (before if any(x[2] == "leaked" for x in errs) else set())
            if errs:
                out[i] = errs
    return out


def judge_pair(entries: list[dict], pa: dict, pb: dict) -> dict:
    """Per spec id: the kept recording, or why the fight is excluded (and whether a retry may clear it)."""
    out = {}
    for e in entries:
        (ra, ea), (rb, eb) = pa.get(e["id"], (None, None)), pb.get(e["id"], (None, None))
        logic = [x for x in (ea or []) + (eb or []) if x[2] is False]
        leaked = [x for x in (ea or []) + (eb or []) if x[2] == "leaked"]
        visual = sorted({x[1] or x[0] for x in (ea or []) + (eb or []) if x[2] is True})
        why = exclusion(e, ra) or exclusion(e, rb)
        j = {"visual_errors": visual} if visual else {}
        if logic:
            j["excluded"] = f"the game logged an error: {logic[0][0]} at {logic[0][1]}"[:240]
        elif leaked:
            j["excluded"] = f"after a faulted fight on its server: {leaked[0][0]} at {leaked[0][1]}"[:240]
            j["retry"] = True
        elif why:
            j["excluded"] = why
            j["retry"] = "idle_timeout" in why
        elif comparable(ra) != comparable(rb):
            j["excluded"] = "unstable: the two recordings differ at " + first_difference(ra, rb)
        else:
            j["result"] = ra
        out[e["id"]] = j
    return out


def play_pass(specs: list[dict], game: str, servers: int, run_dir: Path, out: Path, timeout: float, build: Path) -> tuple[list, dict]:
    job = run_dir.with_suffix(".job.json")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    job.write_text(json.dumps({"runs": [e["spec"] for e in specs]}), encoding="utf-8")
    cmd = [sys.executable, str(HERE / "serve.py"), "--game", game, "--specs", str(job), "--servers", str(servers),
           "--spares", "0", "--run-dir", str(run_dir), "--out", str(out), "--timeout", str(int(timeout)), "--build", str(build)]
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


def cmd_record(a) -> int:
    man = load_manifest(a.version)
    parts = a.parts.split(",")
    if "runs" in parts:
        label = a.label or time.strftime("%Y%m%d-%H%M%S")
        rdir = vdir(a.version) / "record" / f"runs-{label}"
        rdir.mkdir(parents=True, exist_ok=True)
        build = Path(a.build)
        stamp = {"set": f"{SET_NAME} {a.version}", "driver": git_commit(build.parents[3] if len(build.parents) > 3 else ROOT),
                 "driver_dll_sha256": sha256_file(build / "SpirelineReferee.dll"), "recorded": time.strftime("%Y-%m-%d %H:%M")}
        rc = record_runs(man, a, rdir, stamp, build)
        if rc or parts == ["runs"]:
            (vdir(a.version) / "refs-runs.json").write_text(json.dumps({"record": str(rdir), "stamp": stamp}, indent=1), encoding="utf-8")
            return rc
        parts = [p for p in parts if p != "runs"]
    entries = select(man, a.subset, parts)
    if a.limit:
        entries = [e for part in parts for e in [x for x in entries if x["part"] == part][:a.limit]]
    label = a.label or time.strftime("%Y%m%d-%H%M%S")
    rdir = vdir(a.version) / "record" / f"{a.subset}-{label}"
    rdir.mkdir(parents=True, exist_ok=True)
    build = Path(a.build)
    driver_src = build.parents[3] if len(build.parents) > 3 else ROOT
    stamp = {"set": f"{SET_NAME} {a.version}", "driver": git_commit(driver_src),
             "driver_dll_sha256": sha256_file(build / "SpirelineReferee.dll"), "recorded": time.strftime("%Y-%m-%d %H:%M"),
             "subset": a.subset, "label": label}
    t0 = time.time()
    passes = {}
    # Two passes: in spec order, then reversed (other neighbours on other servers).
    for name, order in (("a", entries), ("b", list(reversed(entries)))):
        res, summ = play_pass(order, a.game, a.servers, rdir / f"pass-{name}", rdir / f"pass-{name}.out.json", a.timeout, build)
        errs = log_errors(rdir / f"pass-{name}")
        by_id = {e["id"]: (r, errs.get(i)) for i, (e, r) in enumerate(zip(order, res))}
        passes[name] = (by_id, summ)
    first = judge_pair(entries, passes["a"][0], passes["b"][0])
    # A fight that only hit a driver wait limit (idle_timeout), with no error of its own in the game's log, is
    # recorded twice more on fresh servers: the timeout is often a neighbour's doing (a fight whose game code threw
    # leaves its server unable to settle the next rooms). A fight whose own game code threw is not retried.
    retry = [e for e in entries if first[e["id"]].get("retry")]
    if retry:
        for name, order in (("c", retry), ("d", list(reversed(retry)))):
            res, summ = play_pass(order, a.game, a.servers, rdir / f"pass-{name}", rdir / f"pass-{name}.out.json", a.timeout, build)
            errs = log_errors(rdir / f"pass-{name}")
            passes[name] = ({e["id"]: (r, errs.get(i)) for i, (e, r) in enumerate(zip(order, res))}, summ)
        second = judge_pair(retry, passes["c"][0], passes["d"][0])
        for e in retry:
            second[e["id"]]["retried"] = first[e["id"]].get("excluded")
            first[e["id"]] = second[e["id"]]
    minutes = (time.time() - t0) / 60
    for name, (_, summ) in passes.items():
        if summ.get("network_connections") or summ.get("real_profile_touched") is not False:
            print(f"[refset] SAFETY: pass {name}: network {summ.get('network_connections')} real profile touched "
                  f"{summ.get('real_profile_touched')}; no references written", file=sys.stderr)
            return 1
    builds = {summ.get("sts2_dll_sha256") for _, summ in passes.values()}
    if len(builds) != 1:
        print(f"[refset] the passes ran different game builds {builds}", file=sys.stderr)
        return 1
    stamp["sts2_dll_sha256"] = builds.pop()
    index, refs = [], collections.defaultdict(list)
    for e in entries:
        j = first[e["id"]]
        row = {"id": e["id"], "part": e["part"], "seed": e["spec"]["seed"]}
        for k in ("excluded", "retried", "visual_errors"):
            if j.get(k):
                row[k] = j[k]
        if not j.get("excluded"):
            ra = dict(j["result"])
            if e["part"] == "realistic":
                row["realism"] = realism(e, ra)
            ra["refset"] = {"id": e["id"], "part": e["part"], "stamp": stamp}
            refs[e["part"]].append(ra)
        index.append(row)
    for part in parts:
        (rdir / f"{part}.refs.json").write_text(json.dumps({"refset": stamp, "results": refs.get(part, [])}), encoding="utf-8")
    kept = sum(1 for r in index if "excluded" not in r)
    report = {"stamp": stamp, "minutes": round(minutes, 1), "fights": len(entries), "kept": kept,
              "excluded": [r for r in index if "excluded" in r], "index": index,
              "retried": len(retry),
              "passes": {n: {k: summ.get(k) for k in ("fights", "wall_s", "s_per_fight_per_server", "fights_per_min", "restarts", "errors", "exit")}
                         for n, (_, summ) in passes.items()}}
    (rdir / "record.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    latest = vdir(a.version) / f"refs-{a.subset}.json"
    latest.write_text(json.dumps({"record": str(rdir), "stamp": stamp, "parts": parts}, indent=1), encoding="utf-8")
    why = collections.Counter(r["excluded"].split(":")[0] for r in report["excluded"])
    print(f"[refset] {len(entries)} fights recorded twice in {minutes:.1f} min ({len(retry)} retried after a wait limit): "
          f"{kept} kept, {len(entries) - kept} excluded {dict(why)}")
    print(f"[refset] references: {rdir}  (latest: {latest})")
    return 0


def cmd_status(a) -> int:
    man = load_manifest(a.version)
    print(f"{man['set']} {man['version']} ({man['created']}): {man['counts']}")
    for sub in ("all", "pilot"):
        p = vdir(a.version) / f"refs-{sub}.json"
        if p.exists():
            rec = json.loads((Path(json.loads(p.read_text(encoding='utf-8'))["record"]) / "record.json").read_text(encoding="utf-8"))
            print(f"  {sub}: {rec['kept']} of {rec['fights']} kept ({rec['stamp']['recorded']}, driver {rec['stamp']['driver']['commit'][:9]})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--version", default="v1")
    b.add_argument("--codex", default=str(CODEX))
    b.add_argument("--realistic", type=int, default=600)
    b.add_argument("--max-runs", type=int, default=0, help="stop reading the codex after this many runs (a quick trial)")
    b.add_argument("--pilot-realistic", type=int, default=60)
    b.add_argument("--pilot-coverage", type=int, default=60)
    r = sub.add_parser("record")
    r.add_argument("--version", default="v1")
    r.add_argument("--subset", choices=("all", "pilot"), default="all")
    r.add_argument("--parts", default=",".join(PARTS), help="realistic,coverage,regression and/or runs (whole runs: needs the run referee driver)")
    r.add_argument("--game", default=GAME)
    r.add_argument("--servers", type=int, default=3, help="game processes (at most 3 on this PC)")
    r.add_argument("--build", default=str(DRIVER_BUILD))
    r.add_argument("--timeout", type=float, default=6 * 3600)
    r.add_argument("--label", default=None)
    r.add_argument("--limit", type=int, default=0, help="only the first N specs of each part (a trial)")
    s = sub.add_parser("status")
    s.add_argument("--version", default="v1")
    a = ap.parse_args()
    if a.cmd == "record" and a.servers > 3:
        ap.error("at most 3 game processes")
    return {"build": cmd_build, "record": cmd_record, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
