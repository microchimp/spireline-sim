#!/usr/bin/env python3
"""Fight starts from human runs (D38, 2026-10-30): the practice item of the 2026-11-04 memo, section 3.2.

    python rl/human_starts.py build  [--corpus FILE] [--out-dir DIR] [--procs 4] [--hallway-cap 5000] [--limit-runs N]
    python rl/human_starts.py report [--out-dir DIR]
    python rl/human_starts.py outcomes [--corpus FILE] [--out-dir DIR] [--limit-runs N]

A human run in the public corpus (D13) records, floor by floor, the HP and gold after the floor, the cards gained, removed,
transformed, upgraded and enchanted, the relics and potions picked up, and the encounter met. From that, the opening of each
fight is rebuilt on the copy: the deck, relics, potions, HP and max HP the human walked in with, the same encounter, at the
floor's number. The copy deals the opening hand from its own seed and every practice game draws afresh from it (`fight_gym.
_fresh_world`, as for the nights' starts). The records go to `<out-dir>/human-starts.jsonl` in the gym's `starts.jsonl` schema with
`source: "human"`, one start a line with its snapshot document, and a byte index (`human-starts.index.json`, the file
`gym_loop.load_start` reads through) and a list without documents (`human-starts.lite.json`, which the sampler reads).

What is used and what is never used:
  - Ironclad, Ascension 10, standard mode, one player, not abandoned, no modifiers, vanilla acts and content (a run with a card,
    relic or encounter the copy does not know is dropped by reason, not patched).
  - The exam fifth (D38 addendum 1) is never read as a start. A run is excluded when its id is in `human-exam-fifth-ids.txt` or
    when the fixed hash rule of D38 puts it in the fifth (`is_exam`: the union, so a run the ids file left out because it is not
    vanilla is still out). `assert_no_exam` stops the build if a written record belongs to a run of the fifth.
  - Acts 1 and 2 only (Act 3 is barely verified on the copy). Elites and bosses in full; hallway fights capped
    (`--hallway-cap`, split over the two acts, picked by a fixed hash so the same corpus gives the same set).
  - Human data is reported as aggregates only; a cell over fewer than `MIN_CELL` runs is not shown.

The human's own result (D55 addendum 10). The same record gives each fight floor's HP after it, damage taken, HP healed, potions used
and turns, and the run says whether it went on past the floor. `fight_outcome` reads them in the gym's terms (end HP before the run
engine's post-fight heals, HP lost, survival, potions, turns, value in the search's currency, the gym's L0); `outcomes` makes
`starts/human-outcomes.json` from one pass over the corpus (the start file is not rewritten), and `start_record` carries the same
fields for a future build. The reading, the check against the record's own figures and the coverage are in HUMAN_STARTS.md. The record
has no card plays: nothing here says what move the human made.

What the record cannot say, and the build does about it (every one counted in the report):
  - Enchantments are rebuilt at the amount they were applied with; the ones that grow in play (Goopy) come out low.
  - Relic counters, a card's permanent combat bonuses, and potions found by events are not in the record.
  - A fight inside an event (the event page first, then the fight) is skipped: the HP going in is not known.
  - The deck replay is checked against the run's final deck; a run where they disagree is dropped, by reason.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import practice_mix  # noqa: E402
from practice_mix import doc_cards  # noqa: E402

CORPUS = Path("D:/spireline/data/spire-codex/runs-v0.107.1.jsonl.gz")
EXAM_IDS = CORPUS.with_name("human-exam-fifth-ids.txt")
OUT_DIR = Path("D:/spireline/referee/gym-loop/starts")
STEM = "human-starts"
DOC_PATH = ROOT / "docs" / "rl" / "HUMAN_STARTS.md"
EXAM_SALT = "spireline-d38-exam-fifth|"  # tools/boss_entry.py's rule (Spireline repo): the fifth is by a hash of the run id
EXAM_BUCKET = 0
MIN_CELL = 20  # D13: a cell over fewer runs is not shown
HALLWAY_CAP = 5000
ENGINE = "core2"
A10_POTION_SLOTS = 2  # TightBelt from Ascension 4; Potion Belt adds two
NODE_TYPE = {"hallway": 1, "elite": 2, "boss": 6}  # run_train.vocab's node types, as gym_loop.NODE_TYPE
KINDS = ("hallway", "elite", "boss")
STARTER_RELIC, BLACK_BLOOD = "BURNING_BLOOD", "BLACK_BLOOD"
BOSS_FLOOR = {1: 17, 2: 33}  # the floors of the Act 1 and Act 2 bosses on the copy (rl/run_fields.ACT_BOSS_FLOORS)
VANILLA_ACTS = ("ACT.OVERGROWTH", "ACT.UNDERDOCKS"), "ACT.HIVE", "ACT.GLORY"
ROSTER_TRIES = 32  # seeds tried for a fight whose roster the copy rolls (slimes, bowlbugs, raiders) until it is the human's
WEAK_ROOMS = 0  # `reset_arena`'s completed_combat_rooms for a weak Corpse Slugs or Seapunk (-1 deals their normal twin)


# ---------------------------------------------------------------------------------------------------------------- the exam fifth


def exam_bucket(run_id: str) -> int:
    """The fifth of the corpus a run belongs to, by its run id: a fixed hash, so the split never moves (D38 addendum 1)."""
    return int(hashlib.sha256((EXAM_SALT + str(run_id)).encode()).hexdigest()[:8], 16) % 5


def read_exam_ids(path: Path | str = EXAM_IDS) -> frozenset[str]:
    """The ids file beside the corpus. It must exist: a build without it could not keep the exam out."""
    ids = frozenset(x.strip() for x in Path(path).read_text(encoding="utf-8").split() if x.strip())
    if not ids:
        raise SystemExit(f"human_starts: {path} lists no run: the exam fifth cannot be kept out")
    return ids


def is_exam(run_id: str, ids: Iterable[str]) -> bool:
    """A run of the exam fifth: listed in the ids file, or put in the fifth by the fixed hash rule (the union)."""
    return run_id in ids or exam_bucket(run_id) == EXAM_BUCKET


def assert_no_exam(run_ids: Iterable[str], ids: Iterable[str]) -> int:
    """Raise when any of `run_ids` belongs to the exam fifth; returns how many were checked."""
    ids = ids if isinstance(ids, (set, frozenset)) else frozenset(ids)
    n = 0
    for rid in run_ids:
        n += 1
        if is_exam(rid, ids):
            raise AssertionError(f"human_starts: run {rid} belongs to the exam fifth and may not be a start")
    return n


# ------------------------------------------------------------------------------------------------------------------ the corpus


def human_ok(r: dict) -> bool:
    """Ironclad, Ascension 10, standard mode, one player, finished (not abandoned), no modifiers."""
    pl = r.get("players") or []
    return (r.get("ascension") == 10 and r.get("game_mode") == "standard" and not r.get("was_abandoned") and not r.get("modifiers")
            and len(pl) == 1 and pl[0].get("character") == "CHARACTER.IRONCLAD")


def vanilla_acts(r: dict) -> bool:
    acts = r.get("acts") or []
    return len(acts) == 3 and acts[0] in VANILLA_ACTS[0] and acts[1] == VANILLA_ACTS[1] and acts[2] == VANILLA_ACTS[2]


def run_id(r: dict) -> str:
    return str(r.get("run_hash") or f"{r.get('seed')}:{r.get('start_time')}")


def read_runs(path: Path | str, exam: frozenset[str], stats: Counter) -> Iterator[dict]:
    """The corpus's usable runs, one at a time (a gzip of 650,000 lines: nothing is held). The raw line is screened for the
    character and the Ascension before it is parsed, so the pass is about three minutes. `stats` counts what was read, what the
    exam fifth took and what was left."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            stats["lines"] += 1
            if '"ascension":10,' not in line or '"character":"CHARACTER.IRONCLAD"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                stats["unparsable"] += 1
                continue
            if not human_ok(r):
                continue
            stats["ironclad_a10"] += 1
            rid = run_id(r)
            if is_exam(rid, exam):
                stats["exam_fifth_left_out"] += 1
                continue
            if not vanilla_acts(r):
                stats["modded_acts"] += 1
                continue
            stats["runs_usable"] += 1
            yield r


# --------------------------------------------------------------------------------------------------------------------- tables


def slug(name: str) -> str:
    """PascalCase class name -> the game's UPPER_SNAKE id (`DoorsOfLight` -> `DOORS_OF_LIGHT`)."""
    text = re.sub(r"(?<=[A-Za-z0-9])(?=[A-Z])", "_", name.strip())
    return re.sub(r"[^A-Z0-9_]", "", re.sub(r"\s+", "_", text.upper()))


def _norm(name: str) -> str:
    """A monster's name for comparing rosters: lower case letters and digits. The three Decimillipede segments are one monster
    in the copy, so front, middle and back collapse."""
    flat = re.sub(r"[^a-z0-9]", "", name.lower())
    return re.sub(r"(front|middle|back)$", "", flat) if flat.startswith("decimillipedesegment") else flat


def sub_multiset(small: list[str], big: list[str]) -> bool:
    """Every monster of `small` is in `big` (a human's list of monsters also holds what the fight summoned later)."""
    need = Counter(small)
    need.subtract(Counter(big))
    return all(v <= 0 for v in need.values())


def _cid(x) -> str:
    """`CARD.STRIKE_IRONCLAD` -> `STRIKE_IRONCLAD`."""
    return str(x).split(".", 1)[-1]


class Tables:
    """The copy's ids by the corpus's names: cards, relics, potions, enchantments, encounters, and each enemy's name."""

    def __init__(self, cards: dict[str, int], relics: dict[str, int], potions: dict[str, int], enchants: dict[str, int],
                 enemies: dict[int, str], encounter_names: dict[str, int] | None = None, encounter_of=None) -> None:
        self.cards, self.relics, self.potions, self.enchants, self.enemies = cards, relics, potions, enchants, enemies
        self._encounter_names = encounter_names
        self._encounter_of = encounter_of
        self._enc: dict[str, int | None] = {}

    @classmethod
    def load(cls, repo: Path = ROOT) -> Tables:
        im = json.loads((repo / "data" / "id_map.json").read_text(encoding="utf-8"))
        from sts2_gym import names  # noqa: PLC0415 -- reads the copy's C# tables as text; the library is not loaded

        enemies = {int(i): _norm(n) for i, n in names.enemies().items()}
        enchants = {slug(n): int(i) for i, n in names.enchantments().items() if int(i) != 0}
        spec = importlib.util.spec_from_file_location("generate_encounter_tags", repo / "scripts" / "generate_encounter_tags.py")
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        return cls({slug(k): v for k, v in im["cards"].items()}, {slug(k): v for k, v in im["relics"].items()},
                   {slug(k): v for k, v in im["potions"].items()}, enchants, enemies, gen.encounter_ids(), gen.emulator_name)

    def encounter_id(self, model: str) -> int | None:
        """The copy's id of a corpus encounter (`ENCOUNTER.SLIMES_WEAK` or `SLIMES_WEAK`), None when the copy has no such entry
        (a modded encounter, or an event's fight)."""
        key = _cid(model)
        if key not in self._enc:
            pascal = "".join(p[:1] + p[1:].lower() for p in key.split("_"))
            try:
                name = self._encounter_of(pascal)
                self._enc[key] = self._encounter_names.get(name)
            except SystemExit:
                self._enc[key] = None
        return self._enc[key]

    def monsters(self, ids: Iterable[str]) -> list[str]:
        return sorted(_norm(_cid(m)) for m in ids)


# --------------------------------------------------------------------------------------------------------------- the rebuild
# A card in the replayed deck: (id slug, upgrade level, enchantment slug or None, enchantment amount).


def _card(entry: dict) -> tuple:
    ench = entry.get("enchantment") or {}
    return (_cid(entry["id"]), int(entry.get("current_upgrade_level") or 0), _cid(ench["id"]) if ench.get("id") else None,
            int(ench.get("amount") or 0) if ench.get("id") else 0)


def _take(deck: list, slug_: str, level: int | None = None, ench: str | None = None):
    """Remove (and return) the copy of a card that best matches: id, level and enchantment, then id and level, then the id."""
    for want in ((slug_, level, ench), (slug_, level, None), (slug_, None, None)):
        for i, c in enumerate(deck):
            if c[0] == slug_ and (want[1] is None or c[1] == want[1]) and (want[2] is None or c[2] == want[2]):
                return deck.pop(i)
    return None


def apply_point(deck: list, ps: dict) -> None:
    """One floor's changes to the deck, in the order the game makes them: removals, transformations, new cards, enchantments,
    upgrades and downgrades; a finished quest takes its card out."""
    for c in ps.get("cards_removed") or []:
        e = _card(c)
        _take(deck, e[0], e[1], e[2])
    for t in ps.get("cards_transformed") or []:
        o, f = _card(t["original_card"]), _card(t["final_card"])
        _take(deck, o[0], o[1], o[2])
        deck.append(f)
    gained = [_card(c) for c in ps.get("cards_gained") or []]
    deck.extend(gained)
    for c in ps.get("bought_colorless") or []:  # a bought colorless card is named here; whether the gain list also names it is checked
        name = _cid(c)
        if not any(g[0] == name for g in gained):
            deck.append((name, 0, None, 0))
    for ce in ps.get("cards_enchanted") or []:
        e = _card(ce["card"])  # the card as it stands once enchanted: the enchantment and the amount it was applied with
        hit = next((i for i, c in enumerate(deck) if c[0] == e[0] and c[1] == e[1] and c[2] is None), None)
        if hit is None:
            hit = next((i for i, c in enumerate(deck) if c[0] == e[0] and c[1] == e[1]), None)
        if hit is None:
            hit = next((i for i, c in enumerate(deck) if c[0] == e[0]), None)
        if hit is not None:
            deck[hit] = (e[0], deck[hit][1], e[2], e[3])
        else:
            deck.append(e)
    for cid in ps.get("upgraded_cards") or []:
        name = _cid(cid)
        for i, c in enumerate(deck):
            if c[0] == name and c[1] == 0:
                deck[i] = (c[0], 1, c[2], c[3])
                break
    for cid in ps.get("downgraded_cards") or []:
        name = _cid(cid)
        for i, c in enumerate(deck):
            if c[0] == name and c[1] >= 1:
                deck[i] = (c[0], c[1] - 1, c[2], c[3])
                break
    for cid in ps.get("completed_quests") or []:
        _take(deck, _cid(cid))


def apply_potions(held: list[str], pt: dict, ps: dict) -> None:
    """Potions held after a floor: picked from a reward (a shop lists its offers there, so a shop counts only what was bought),
    bought, less used and discarded. Potions an event gives are not in the record."""
    if pt.get("map_point_type") == "shop":
        held.extend(_cid(p) for p in ps.get("bought_potions") or [])
    else:
        held.extend(_cid(p["choice"]) for p in ps.get("potion_choices") or [] if p.get("was_picked"))
        held.extend(_cid(p) for p in ps.get("bought_potions") or [])
    for p in list(ps.get("potion_used") or []) + list(ps.get("potion_discarded") or []):
        name = _cid(p)
        if name in held:
            held.remove(name)


def final_deck_matches(deck: list, r: dict) -> bool:
    """The replayed deck against the run's last deck (card, upgrade level and enchantment, not the enchantment's amount)."""
    final = Counter(_card(c)[:3] for c in r["players"][0].get("deck") or [])
    return Counter(c[:3] for c in deck) == final


def point_fight(pt: dict) -> tuple[str, str] | None:
    """(kind, corpus encounter id) of the fight a floor was, or None for any other room. A floor with an event page before the
    fight returns ("event", ...): the HP going into the fight is not known."""
    rooms = pt.get("rooms") or []
    if not rooms:
        return None
    last = rooms[-1]
    kind = {"monster": "hallway", "elite": "elite", "boss": "boss"}.get(last.get("room_type"))
    model = last.get("model_id") or ""
    if kind is None or not model.startswith("ENCOUNTER."):
        return None
    return ("event" if len(rooms) > 1 else kind), model


def relics_at(r: dict, floor: int, gains: dict[str, int], removed: dict[str, int]) -> list[str]:
    """The relics held walking into floor `floor`: the run's final relics gained before it, and those lost later that were
    picked up before it. The starter relic is held until Black Blood replaces it."""
    held: list[str] = []
    for x in r["players"][0].get("relics") or []:
        if int(x.get("floor_added_to_deck") or 0) <= floor - 1:
            held.append(_cid(x["id"]))
    for name, gone in removed.items():
        if gone >= floor and gains.get(name, floor) <= floor - 1 and name not in held:
            held.append(name)
    if BLACK_BLOOD not in held and STARTER_RELIC not in held:
        held.insert(0, STARTER_RELIC)
    return held


def relic_timeline(flat: list) -> tuple[dict[str, int], dict[str, int]]:
    """The floor each relic was first gained on, and the floor each lost one went on (from the floors' choices, purchases and removals)."""
    gains: dict[str, int] = {}
    removed: dict[str, int] = {}
    for f, (_, pt) in enumerate(flat, 1):
        ps = (pt.get("player_stats") or [{}])[0]
        for c in ps.get("relic_choices") or []:
            if c.get("was_picked"):
                gains.setdefault(_cid(c["choice"]), f)
        for c in ps.get("bought_relics") or []:
            gains.setdefault(_cid(c), f)
        for c in ps.get("relics_removed") or []:
            removed[_cid(c)] = f
    return gains, removed


def rebuild_run(r: dict, tables: Tables, fail: Counter, seen: Counter | None = None) -> list[dict]:
    """The fight specs of one run: for every Act 1 and Act 2 fight, the state the human walked in with. `fail` counts the fights
    and runs that could not be rebuilt, by reason; `seen` counts every fight the run entered by act and kind (rebuilt or not)."""
    flat = [(a, pt) for a, act in enumerate(r.get("map_point_history") or []) for pt in act]
    rid = run_id(r)
    deck: list = [("ASCENDERS_BANE", 0, None, 0)] + [("STRIKE_IRONCLAD", 0, None, 0)] * 5 + [("DEFEND_IRONCLAD", 0, None, 0)] * 4 \
        + [("BASH", 0, None, 0)]
    potions: list[str] = []
    gains, removed = relic_timeline(flat)
    specs: list[dict] = []
    hp = max_hp = gold = None
    for f, (a, pt) in enumerate(flat, 1):
        ps = (pt.get("player_stats") or [{}])[0]
        fight = point_fight(pt)
        if fight is not None and f >= 2:
            kind, model = fight
            act = a + 1
            cell = f"act{act}-{'hallway' if kind == 'event' else kind}"
            if seen is not None and kind != "event":
                seen[cell] += 1
            if act > 2:
                fail["act 3 or later: not built"] += 1
            elif kind == "event":
                fail["an event page before the fight: HP going in not known"] += 1
            elif hp is None:
                fail["no HP before the fight"] += 1
            else:
                spec = _spec(r, rid, tables, f, act, kind, model, pt, deck, potions, gains, removed, hp, max_hp, gold, fail, last=f == len(flat))
                if spec is not None:
                    specs.append(spec)
        apply_point(deck, ps)
        apply_potions(potions, pt, ps)
        hp, max_hp, gold = ps.get("current_hp"), ps.get("max_hp"), ps.get("current_gold")
    if specs and not final_deck_matches(deck, r):
        fail["deck replay disagrees with the run's final deck: run dropped"] += len(specs)
        return []
    return specs


def _spec(r, rid, tables, f, act, kind, model, pt, deck, potions, gains, removed, hp, max_hp, gold, fail, last: bool = False) -> dict | None:
    eid = tables.encounter_id(model)
    if eid is None:
        fail["encounter unknown to the copy"] += 1
        return None
    if kind == "boss" and f != BOSS_FLOOR[act]:
        fail["boss at an unusual floor: the route is not the copy's"] += 1
        return None
    if not 0 < int(hp or 0) <= int(max_hp or 0):
        fail["HP before the fight not credible"] += 1
        return None
    ids, ench, level2 = [], [], 0
    for c in deck:
        cid = tables.cards.get(c[0])
        if cid is None:
            fail["card unknown to the copy"] += 1
            return None
        eng = 0
        if c[2] is not None:
            eng = tables.enchants.get(c[2])
            if eng is None:
                fail["enchantment unknown to the copy"] += 1
                return None
        ids.append(-cid if c[1] >= 1 else cid)
        ench.append((eng, c[3] if eng else 0))
        level2 += c[1] > 1
    held = relics_at(r, f, gains, removed)
    rel = []
    for name in held:
        rid_ = tables.relics.get(name)
        if rid_ is None:
            fail["relic unknown to the copy"] += 1
            return None
        rel.append(rid_)
    pots = [tables.potions[p] for p in potions if p in tables.potions]
    room = (pt.get("rooms") or [{}])[-1]
    slots = A10_POTION_SLOTS + (2 if "POTION_BELT" in held else 0)
    outcome = fight_outcome(hp_in=int(hp), max_hp_in=int(max_hp), survived=not last, ps=(pt.get("player_stats") or [{}])[0], room=room,
                            held=held, potions_in=len(pots))
    return {"id": f"human:{rid}|f{f:02d}", "run": rid, "seed": rid, "act": act, "floor": f, "kind": kind, "cat": f"act{act}-{kind}",
            "node_type": NODE_TYPE[kind], "enemy": _cid(room["model_id"]), "encounter_id": eid,
            "monsters": tables.monsters(room.get("monster_ids") or []), "weak": model.endswith("_WEAK"), "deck": ids, "enchantments": ench, "relics": rel,
            "potions": pots, "potion_slots": max(slots, len(pots)), "hp": int(hp), "max_hp": int(max_hp), "gold": int(gold or 0),
            "approx": {"enchanted": any(e[0] for e in ench), "upgrade_level_above_1": bool(level2),
                       "potions_unknown": len(pots) != len(potions)},
            "outcome": outcome if isinstance(outcome, dict) else None, "outcome_missing": outcome if isinstance(outcome, str) else None}


# ------------------------------------------------------------------------------------------------- the human's own result
# A fight floor of the record gives the HP after the floor (`current_hp`), the damage taken, the HP healed on the floor
# (`hp_healed`), the potions used and the room's turns; the run itself says whether it went on past the floor. A fight start can
# carry the human's result so the player is held to it (D55 addendum 10). The record has no card plays: nothing here imitates a move.
#
# The reading. The gym ends a fight when the last enemy dies, and its end HP is the HP at that moment: with what the fight healed
# on the way (a potion, a card), without what the run engine heals once the combat is over (Burning Blood's six, Black Blood's
# twelve, Meat on the Bone's twelve at half HP or less, Chosen Cheese's point). The record's `current_hp` is after those heals and
# `hp_healed` is the floor's total, so the human's end HP is `current_hp` less the post-fight heals the relics held explain:
# less min(hp_healed, what they give). The heal is capped by the HP room left, which is why the minimum. Taking `hp_in -
# current_hp` as the HP lost would understate it by that heal (six with Burning Blood); taking `damage_taken` would overstate it
# by the fight's own heals (about one fight in ten). A death is the whole HP going in lost (the exam's rule), HP 0, and the
# search's DEATH_VALUE.

OUTCOME_RULE = "human-outcomes-v1"
OUTCOME_FILE = "human-outcomes.json"
END_HEALS = {"BURNING_BLOOD": 6, "BLACK_BLOOD": 12, "CHOSEN_CHEESE": 1}
MEAT_ON_THE_BONE, MEAT_HEAL = "MEAT_ON_THE_BONE", 12
DEATH_VALUE = practice_mix.DEATH_VALUE
POTION_VALUE = practice_mix.POTION_VALUE
OUTCOME_KEYS = ("human_survived", "human_end_hp", "human_hp_lost", "human_damage_taken", "human_post_heal", "human_turns",
                "human_potions_used", "human_potions_held_end", "human_l0", "human_l0_unit")


def l0_unit(won: bool, hp: int, potions: int, max_hp: int) -> float:
    """The gym's L0 (`fight_gym.l0`; a test pins the two): 0 for a death, else 0.5 + 0.5 min(1, (HP + 10 potions) / max HP)."""
    if not won:
        return 0.0
    return 0.5 + 0.5 * min(1.0, (hp + POTION_VALUE * potions) / max(1, max_hp))


def post_fight_heal(held: Iterable[str], hp_healed: int) -> tuple[int, bool]:
    """What the floor's `hp_healed` owes to the relics that heal once the combat is over, and whether that is a guess (Meat on the
    Bone heals only at half HP or less: it counts when the floor healed the relics' sure heals and its twelve besides)."""
    held = set(held)
    base = sum(v for k, v in END_HEALS.items() if k in held)
    guess = MEAT_ON_THE_BONE in held
    if guess and hp_healed >= base + MEAT_HEAL:
        base += MEAT_HEAL
    return max(0, min(int(hp_healed), base)), guess


def fight_outcome(*, hp_in, max_hp_in, survived: bool, ps: dict, room: dict, held: Iterable[str], potions_in: int) -> dict | str:
    """The human's result of one fight floor in the gym's terms, or the reason it cannot be read (a string). `survived`: the run went
    on past the floor (for a boss, the boss died); `held`: the relics held going in; `potions_in`: the potions held going in."""
    if hp_in is None or max_hp_in is None:
        return "HP going in not known"
    for key in ("current_hp", "damage_taken", "hp_healed"):
        if not isinstance(ps.get(key), (int, float)):
            return f"the floor lacks {key}"
    hp_in, max_hp_in = int(hp_in), int(max_hp_in)
    cur, dmg, heal = int(ps["current_hp"]), int(ps["damage_taken"]), int(ps["hp_healed"])
    used = len(ps.get("potion_used") or [])
    held_end = max(0, int(potions_in) - used)
    turns = room.get("turns_taken")
    flags: list[str] = []
    if survived:
        post, guess = post_fight_heal(held, heal)
        end = cur - post
        top = max(int(ps.get("max_hp") or 0), max_hp_in)
        if end < 1:
            end = 1
            flags.append("end_hp_raised_to_1")
        elif end > top:
            end = top
            flags.append("end_hp_cut_to_max")
        if guess:
            flags.append("heal_guess")
        lost, value, unit = hp_in - end, float(end + POTION_VALUE * held_end), l0_unit(True, end, held_end, max_hp_in)
    else:
        post, end, lost, value, unit = 0, 0, hp_in, DEATH_VALUE, 0.0
        if cur != 0:
            flags.append("death_hp_not_zero")
    out = {"human_survived": bool(survived), "human_end_hp": end, "human_hp_lost": lost, "human_damage_taken": dmg, "human_post_heal": post,
           "human_turns": int(turns) if isinstance(turns, (int, float)) else None, "human_potions_used": used,
           "human_potions_held_end": held_end, "human_l0": value, "human_l0_unit": round(unit, 4)}
    if flags:
        out["human_flags"] = flags
    return out


def outcomes_of_run(r: dict) -> tuple[dict[str, dict], dict[str, str], dict[str, dict]]:
    """The results of a run's Act 1 and 2 fight floors, by the start ids `rebuild_run` gives them (an event page before the fight
    leaves no start): ({id: outcome}, {id: why there is none}, {id: the floor's numbers for the checks}). Needs no copy and no
    tables, only the record's own HP, relics and potions."""
    flat = [(a, pt) for a, act in enumerate(r.get("map_point_history") or []) for pt in act]
    rid = run_id(r)
    gains, removed = relic_timeline(flat)
    potions: list[str] = []
    outs: dict[str, dict] = {}
    why: dict[str, str] = {}
    checks: dict[str, dict] = {}
    hp = max_hp = None
    for f, (a, pt) in enumerate(flat, 1):
        ps = (pt.get("player_stats") or [{}])[0]
        fight = point_fight(pt)
        if fight is not None and f >= 2 and a + 1 <= 2 and fight[0] != "event":
            sid = f"human:{rid}|f{f:02d}"
            held = relics_at(r, f, gains, removed)
            res = fight_outcome(hp_in=hp, max_hp_in=max_hp, survived=f < len(flat), ps=ps, room=(pt.get("rooms") or [{}])[-1], held=held,
                                potions_in=len(potions))
            if isinstance(res, dict):
                outs[sid] = res
                checks[sid] = {"hp_in": hp, "cur": ps["current_hp"], "dmg": ps["damage_taken"], "heal": ps["hp_healed"],
                               "act": a + 1, "kind": fight[0], "run": rid}
            else:
                why[sid] = res
        apply_potions(potions, pt, ps)
        hp, max_hp = ps.get("current_hp"), ps.get("max_hp")
    return outs, why, checks


# ------------------------------------------------------------------------------------------------------- the copy (child side)

_W: dict = {}


def below_normal() -> None:
    """BelowNormal priority through the Windows priority class (no POSIX call)."""
    try:
        import ctypes  # noqa: PLC0415

        k = ctypes.windll.kernel32  # typed: the untyped call truncates the pseudo handle and fails without an error
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)
    except (AttributeError, OSError):
        pass


def _init_worker() -> None:
    """A pool worker: BelowNormal, the engine of the gym (core2), the library loaded once."""
    below_normal()
    os.environ.setdefault("STS2_ENGINE", ENGINE)
    from sts2_gym import names, native  # noqa: PLC0415

    _W["native"] = native
    _W["enemy_names"] = {int(i): _norm(n) for i, n in names.enemies().items()}


def game_seed(start: str, tries: int) -> int:
    """The seed the opening is dealt from: fixed by the start's id (and the try), so a rebuild gives the same file."""
    h = hashlib.sha256(f"{start}|hbuild|{tries}".encode()).digest()
    return int.from_bytes(h[:4], "little") % (1 << 30)


def build_doc(spec: dict, native=None, enemy_names: dict[int, str] | None = None, tries: int = ROSTER_TRIES) -> tuple[dict | None, str]:
    """The fight's opening as an `sts2-emu-snapshot/1` document: the copy's arena entry (deck, enchantments, relics, potions, HP,
    gold, encounter, floor, Ascension 10), then its exported state. The copy's opening roster must be among the human's monsters
    (their list also holds what the fight summoned later); where the copy rolls the roster from the seed (slimes, bowlbugs,
    raiders) other seeds are tried until it is.
    Returns (document, "") or (None, reason)."""
    import ctypes  # noqa: PLC0415

    native = native or _W["native"]
    enemy_names = enemy_names or _W["enemy_names"]
    reason = "roster differs"
    for t in range(max(1, tries)):
        h = None
        try:
            h = native.create(game_seed(spec["id"], t))
            obs = (ctypes.c_int * native.OBS_SIZE)()
            native.reset_arena(h, list(spec["deck"]), obs, enchantments=[tuple(e) for e in spec["enchantments"]],
                               encounter_id=spec["encounter_id"], relic_ids=list(spec["relics"]), potion_ids=list(spec["potions"]),
                               player_hp=spec["hp"], player_max_hp=spec["max_hp"], player_gold=spec["gold"], ascension=10,
                               total_floor=spec["floor"], max_potion_slots=spec["potion_slots"],
                               completed_combat_rooms=WEAK_ROOMS if spec.get("weak") else -1)
            doc = native.get_state_json(h)
        except Exception as error:  # noqa: BLE001 -- the copy refused the position: counted by its message
            return None, f"copy refused: {type(error).__name__}: {str(error)[:80]}"
        finally:
            if h is not None:
                native.destroy(h)
        mine = sorted(enemy_names.get(int(e.get("def_id", -1)), f"?{e.get('def_id')}") for e in doc.get("enemies") or [])
        if not spec["monsters"] or sub_multiset(mine, spec["monsters"]):
            return doc, ""
        reason = "roster differs"
    return None, reason


def build_chunk(specs: list[dict]) -> list[tuple[str, str | None, str]]:
    """A chunk of specs as (id, the record's JSON line or None, reason). The record is built here so the parent only writes."""
    out = []
    for spec in specs:
        doc, why = build_doc(spec)
        out.append((spec["id"], None if doc is None else json.dumps(start_record(spec, doc), separators=(",", ":")), why))
    return out


def start_record(spec: dict, doc: dict) -> dict:
    """A start in the gym's `starts.jsonl` schema (the nights' replays write the same fields), marked `source: "human"`."""
    from gym_loop import is_val_seed  # noqa: PLC0415 -- the trainer's split, by run

    pots = doc.get("potions") or {}
    held = sum(1 for x in (pots.get("slots") or [])[: int(pots.get("max_slots", 0))] if x)
    deck_size = len(doc_cards(doc))
    return {"id": spec["id"], "cat": spec["cat"], "kind": spec["kind"], "act": spec["act"], "floor": spec["floor"], "seed": spec["seed"],
            "ix": -1, "night": "human", "source": "human", "run": spec["run"], "enemy": spec["enemy"], "k": 0,
            "node_type": spec["node_type"], "hp": spec["hp"], "max_hp": spec["max_hp"], "deck_size": deck_size, "potions": held,
            "relics": len(spec["relics"]), "encounter_id": doc.get("encounter_id"), "turn": int(doc.get("turn", 0)),
            "approx": spec["approx"], "kind_same": True, "replay_same": None,
            "split": "val" if is_val_seed(spec["seed"]) else "train", **(spec.get("outcome") or {}),
            **({"human_outcome_missing": spec["outcome_missing"]} if spec.get("outcome_missing") else {}), "doc": doc}


# ------------------------------------------------------------------------------------------------------------------ the build


def pick_specs(specs: list[dict], hallway_cap: int) -> list[dict]:
    """Every elite and boss spec, and the hallway ones up to `hallway_cap` (half to each act, by a fixed hash of the id)."""
    keep = [s for s in specs if s["kind"] != "hallway"]
    hall = [s for s in specs if s["kind"] == "hallway"]
    by_act: dict[int, list[dict]] = defaultdict(list)
    for s in sorted(hall, key=lambda s: hashlib.sha256(s["id"].encode()).hexdigest()):
        by_act[s["act"]].append(s)
    per = hallway_cap // max(1, len(by_act) or 1)
    left = hallway_cap
    for act in sorted(by_act, reverse=True):  # Act 2 first
        take = by_act[act][:min(per, left)]
        keep.extend(take)
        left -= len(take)
    for act in sorted(by_act, reverse=True):  # what one act could not fill goes to the other
        extra = by_act[act][per:per + left]
        keep.extend(extra)
        left -= len(extra)
    return sorted(keep, key=lambda s: s["id"])


class SetWriter:
    """human-starts.jsonl written as the openings arrive (one start a line), with its byte index and the list without documents.
    Nothing is kept in memory but the lite records; the files replace the old set only when `close` is called."""

    def __init__(self, out_dir: Path) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{STEM}.jsonl"
        self.tmp = self.path.with_suffix(".tmp")
        self.fh = open(self.tmp, "wb")  # noqa: SIM115 -- closed in `close`
        self.index: dict[str, int] = {}
        self.lite: list[dict] = []

    def add(self, line: str) -> dict:
        rec = json.loads(line)
        rec.pop("doc", None)
        sid = rec["id"]
        if sid in self.index:
            raise ValueError(f"human_starts: start {sid} written twice")
        self.index[sid] = self.fh.tell()
        self.fh.write((line + "\n").encode("utf-8"))
        self.lite.append(rec)
        return rec

    def close(self) -> int:
        self.fh.close()
        os.replace(self.tmp, self.path)
        self.lite.sort(key=lambda r: r["id"])
        for name, obj in ((f"{STEM}.index.json", self.index), (f"{STEM}.lite.json", self.lite)):
            t = self.dir / (name + ".tmp")
            t.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
            os.replace(t, self.dir / name)
        return len(self.lite)


def scan(corpus: Path, exam: frozenset[str], limit_runs: int, cache: Path | None) -> tuple[list[dict], Counter, Counter, Counter]:
    """The corpus read and every fight rebuilt in state (no copy needed): (specs, stats, failures, fights seen). With a cache file
    the result is read from it when it exists and written to it when it does not, so the three-minute read happens once."""
    if cache is not None and Path(cache).is_file():
        d = json.loads(Path(cache).read_text(encoding="utf-8"))
        return d["specs"], Counter(d["stats"]), Counter(d["fail"]), Counter(d["seen"])
    tables = Tables.load()
    stats: Counter = Counter()
    fail: Counter = Counter()
    seen: Counter = Counter()
    specs: list[dict] = []
    for r in read_runs(corpus, exam, stats):
        assert not is_exam(run_id(r), exam)
        got = rebuild_run(r, tables, fail, seen)
        if got:
            stats["runs_with_fights"] += 1
        specs.extend(got)
        if limit_runs and stats["runs_usable"] >= limit_runs:
            break
    stats["floors_rebuilt_in_state"] = len(specs)
    if cache is not None:
        Path(cache).write_text(json.dumps({"specs": specs, "stats": stats, "fail": fail, "seen": seen}, separators=(",", ":")),
                               encoding="utf-8")
    return specs, stats, fail, seen


def build(corpus: Path, out_dir: Path, procs: int, hallway_cap: int, limit_runs: int = 0, exam_path: Path = EXAM_IDS,
          doc_path: Path | None = DOC_PATH, cache: Path | None = None, specs_only: bool = False) -> dict | None:
    """Read the corpus, rebuild the fights, deal their openings on the copy in `procs` processes, write the set and the report.
    `cache` keeps the read corpus's fight states between runs; `specs_only` stops after writing it (the read is a single
    process and plays nothing; the dealing is the part that uses the copy)."""
    t0 = time.time()
    exam = read_exam_ids(exam_path)
    specs, stats, fail, seen = scan(corpus, exam, limit_runs, cache)
    assert_no_exam((s["run"] for s in specs), exam)
    chosen = pick_specs(specs, hallway_cap)
    print(f"human_starts: {stats['runs_usable']} runs usable of {stats['ironclad_a10']} Ironclad A10; {len(specs)} fight states, "
          f"{len(chosen)} chosen ({time.time() - t0:.0f} s)", flush=True)
    if specs_only:
        return None
    print(f"human_starts: dealing the openings on {procs} processes", flush=True)
    writer = SetWriter(out_dir)
    try:
        doc_fail = deal(chosen, procs, writer.add)
        fail.update(doc_fail)
        assert_no_exam((r["run"] for r in writer.lite), exam)
        recs = list(writer.lite)
    except BaseException:
        writer.fh.close()
        writer.tmp.unlink(missing_ok=True)
        raise
    n = writer.close()
    report = coverage(recs, stats, fail, seen, len(specs), len(chosen))
    report["hallway_cap"] = hallway_cap
    text = coverage_text(report)
    print(text)
    if doc_path is not None:
        Path(doc_path).parent.mkdir(parents=True, exist_ok=True)
        old = Path(doc_path).read_text(encoding="utf-8") if Path(doc_path).is_file() else ""
        i = old.find("\n" + OUTCOME_HEAD)  # the outcomes section belongs to the sidecar's pass, not to this build: it is kept
        Path(doc_path).write_text(text + ("\n" + old[i + 1:] if i >= 0 else ""), encoding="utf-8", newline="\n")
    print(f"human_starts: {n} starts written to {out_dir} ({time.time() - t0:.0f} s)")
    return report


def deal(specs: list[dict], procs: int, sink, chunk: int = 200) -> Counter:
    """Deal the openings: chunks of specs over a supervised spawn pool (a worker that dies loses one chunk, which is run again
    once and then counted as a failure by reason). Each start's JSON line goes to `sink(line)`; returns the failures by reason."""
    import multiprocessing as mp  # noqa: PLC0415

    from spawn_pool import SupervisedPool  # noqa: PLC0415

    chunks = [specs[i:i + chunk] for i in range(0, len(specs), chunk)]
    fail: Counter = Counter()
    pool = SupervisedPool(mp.get_context("spawn"), max(1, procs), initializer=_init_worker, run_timeout_s=600.0)
    todo = list(enumerate(chunks))
    retried: set[int] = set()
    try:
        while todo or pool.inflight():
            while todo and pool.free():
                i, c = todo.pop()
                pool.submit(i, build_chunk, (c,))
            for d in pool.collect():
                if d.status == "ok":
                    for _sid, line, why in d.value:
                        if line is None:
                            fail[why or "no document"] += 1
                        else:
                            sink(line)
                elif d.key not in retried:
                    retried.add(d.key)
                    todo.append((d.key, chunks[d.key]))
                else:
                    fail["copy process died on the chunk"] += len(chunks[d.key])
            pool.wait(0.5)
    finally:
        pool.close()
    return fail


# ------------------------------------------------------------------------------------------------------------------- the report


def coverage(recs: list[dict], stats: Counter, fail: Counter, seen: Counter, states: int, chosen: int) -> dict:
    """Aggregates of the written set. Cells over fewer than MIN_CELL distinct runs are suppressed."""
    cells: dict[tuple, set] = defaultdict(set)
    by_ak: dict[tuple, set] = defaultdict(set)
    for r in recs:
        cells[(r["act"], r["kind"], r["enemy"])].add(r["run"])
        by_ak[(r["act"], r["kind"])].add(r["run"])
    counts = Counter((r["act"], r["kind"]) for r in recs)
    return {"runs_read": stats["lines"], "ironclad_a10": stats["ironclad_a10"], "exam_fifth_left_out": stats["exam_fifth_left_out"],
            "modded_acts": stats["modded_acts"], "runs_usable": stats["runs_usable"], "runs_with_fights": stats["runs_with_fights"],
            "floors_in_state": states, "floors_chosen": chosen, "records": len(recs), "seen": dict(seen),
            "by_act_kind": {f"act{a}-{k}": {"records": counts[(a, k)], "runs": len(by_ak[(a, k)])} for (a, k) in sorted(counts)},
            "cells": {f"act{a} {k} {e}": len(runs) for (a, k, e), runs in sorted(cells.items()) if len(runs) >= MIN_CELL},
            "cells_suppressed": sum(1 for runs in cells.values() if len(runs) < MIN_CELL),
            "failures": dict(fail.most_common()),
            "enchanted_records": sum(1 for r in recs if r["approx"]["enchanted"]),
            "potions_unknown_records": sum(1 for r in recs if r["approx"]["potions_unknown"])}


def coverage_text(rep: dict) -> str:
    """The coverage report (aggregates only; no run is named, no cell under MIN_CELL runs is shown)."""
    out = ["# Human fight starts: coverage", "",
           "Written by `rl/human_starts.py build` (aggregates only; a cell over fewer than %d runs is not shown)." % MIN_CELL, "",
           f"- runs read (every run of the corpus, any character and Ascension): {rep['runs_read']:,}",
           f"- of those Ironclad, Ascension 10, standard mode, finished, no modifiers: {rep['ironclad_a10']:,}",
           f"- left out as the exam fifth (ids file or hash rule): {rep['exam_fifth_left_out']:,}",
           f"- left out for modded acts: {rep['modded_acts']:,}",
           f"- runs usable: {rep['runs_usable']:,}; runs that gave at least one fight: {rep['runs_with_fights']:,}",
           f"- floors usable (Act 1 and 2 fights whose state could be rebuilt): {rep['floors_in_state']:,}; chosen (every elite and "
           f"boss, hallway capped at {rep.get('hallway_cap', HALLWAY_CAP):,}): {rep['floors_chosen']:,}",
           f"- records written: {rep['records']:,} (with an enchanted card: {rep['enchanted_records']:,}; potions only partly known: "
           f"{rep['potions_unknown_records']:,})", "", "## Records by act and kind", "",
           "| act and kind | fights the usable runs entered | records | runs |", "|---|---|---|---|"]
    for k, v in rep["by_act_kind"].items():
        n_runs = v["runs"] if v["runs"] >= MIN_CELL else f"< {MIN_CELL}"
        out.append(f"| {k} | {rep['seen'].get(k, 0):,} | {v['records']:,} | {n_runs} |")
    out += ["", "## Records by act, kind and enemy (runs per cell)", ""]
    for k, n in rep["cells"].items():
        out.append(f"- {k}: {n}")
    out += [f"- {rep['cells_suppressed']} further cells are under {MIN_CELL} runs and not shown", "", "## Rebuild failures by reason", ""]
    for k, n in rep["failures"].items():
        out.append(f"- {k}: {n:,}")
    out += ["", "A fight state can fail for one reason only (the first found). Records marked `approx` are rebuilt without the "
            "growth of in-play enchantments, relic counters, or potions an event gave.", ""]
    return "\n".join(out)


def cmd_report(a) -> int:
    lite = json.loads((Path(a.out_dir) / f"{STEM}.lite.json").read_text(encoding="utf-8"))
    counts = Counter((r["act"], r["kind"]) for r in lite)
    runs: dict = defaultdict(set)
    for r in lite:
        runs[(r["act"], r["kind"])].add(r["run"])
    for k in sorted(counts):
        n = len(runs[k])
        print(f"act{k[0]}-{k[1]}: {counts[k]} records, {n if n >= MIN_CELL else '< ' + str(MIN_CELL)} runs")
    return 0


# --------------------------------------------------------------------------------------------------- the outcomes sidecar


OUTCOME_HEAD = "## Human outcomes"


def collect_outcomes(corpus: Path, exam: frozenset[str], lite: dict[str, dict], limit_runs: int = 0, log=None) -> tuple[dict, dict]:
    """One pass over the corpus's usable runs (the exam fifth is never read as a run): the outcome of every start in `lite`. Returns
    ({start id: outcome}, a report of what was found and what was not)."""
    stats: Counter = Counter()
    outcomes: dict[str, dict] = {}
    why: Counter = Counter()
    seen_ids: set[str] = set()
    cell = defaultdict(lambda: {"starts": 0, "survived": 0, "lost": 0.0, "turns": 0.0, "turn_n": 0, "potions": 0, "runs": set(), "value": 0.0})
    chk: Counter = Counter()
    heal_sums: dict[str, float] = defaultdict(float)
    t0 = time.time()
    for r in read_runs(corpus, exam, stats):
        assert not is_exam(run_id(r), exam)
        outs, miss, checks = outcomes_of_run(r)
        for sid, o in outs.items():
            rec = lite.get(sid)
            if rec is None:
                continue
            seen_ids.add(sid)
            c = checks[sid]
            if c["hp_in"] != rec["hp"]:
                chk["hp_going_in_differs_from_the_start"] += 1
                continue
            outcomes[sid] = o
            k = cell[(rec["act"], rec["kind"])]
            k["starts"] += 1
            k["runs"].add(rec["run"])
            k["survived"] += o["human_survived"]
            k["value"] += o["human_l0"]
            k["potions"] += o["human_potions_used"]
            if o["human_survived"]:
                k["lost"] += o["human_hp_lost"]
            if o["human_turns"] is not None:
                k["turns"] += o["human_turns"]
                k["turn_n"] += 1
            for fl in o.get("human_flags") or []:
                chk["flag: " + fl] += 1
            if o["human_survived"]:
                exact = c["hp_in"] - c["dmg"] + c["heal"] == c["cur"]
                chk["survivor floors"] += 1
                chk["identity holds"] += exact
                heal_sums["post"] += o["human_post_heal"]
                heal_sums["in_fight"] += max(0, c["heal"] - o["human_post_heal"])
                chk["floors with a heal beyond the relics own"] += c["heal"] - o["human_post_heal"] > 0
                heal_sums["hp_in_minus_after"] += c["hp_in"] - c["cur"]
                heal_sums["damage_taken"] += c["dmg"]
                heal_sums["hp_lost_read"] += o["human_hp_lost"]
        for sid, reason in miss.items():
            if sid in lite:
                why[reason] += 1
                seen_ids.add(sid)
        if limit_runs and stats["runs_usable"] >= limit_runs:
            break
        if log and stats["runs_usable"] % 2000 == 0:
            log(f"human_starts outcomes: {stats['runs_usable']} runs, {len(outcomes)} outcomes ({time.time() - t0:.0f} s)")
    if not limit_runs:
        for sid in lite:
            if sid not in seen_ids:
                why["start not found again in the corpus pass"] += 1
    n = max(1, chk["survivor floors"])
    rep = {"runs_usable": stats["runs_usable"], "starts_in_set": len(lite), "outcomes": len(outcomes), "missing": dict(why.most_common()),
           "checks": dict(chk), "heals": {k: round(v / n, 3) for k, v in heal_sums.items()},
           "cells": {f"act{a}-{k}": {"starts": v["starts"], "runs": len(v["runs"]), "survived": v["survived"],
                                     "mean_hp_lost_survivors": v["lost"] / max(1, v["survived"]), "mean_turns": v["turns"] / max(1, v["turn_n"]),
                                     "mean_potions_used": v["potions"] / max(1, v["starts"]), "mean_value": v["value"] / max(1, v["starts"])}
                     for (a, k), v in sorted(cell.items())}}
    return outcomes, rep


def outcomes_text(rep: dict) -> str:
    """The doc section for the outcomes (aggregates only; a cell over fewer than MIN_CELL runs is not shown)."""
    ch, hl = rep["checks"], rep["heals"]
    n = max(1, ch.get("survivor floors", 0))
    out = [OUTCOME_HEAD, "",
           "Written by `rl/human_starts.py outcomes` into `starts/human-outcomes.json` (one entry a start id; the start file itself is not "
           "rewritten). Source: Spire Codex public run export for v0.107.1, the four training fifths. Aggregates only; a cell over "
           f"fewer than {MIN_CELL} runs is not shown.", "",
           f"- starts in the set: {rep['starts_in_set']:,}; with an outcome: {rep['outcomes']:,} ({100 * rep['outcomes'] / max(1, rep['starts_in_set']):.2f}%)",
           "- fields: `human_survived` (the run went on past the floor; a boss floor: the boss died), `human_end_hp` (HP when the fight "
           "ended, before the run engine's post-fight heals; 0 for a death), `human_hp_lost` (HP going in less end HP; the whole HP going "
           "in for a death), `human_damage_taken` (the floor's raw figure), `human_post_heal`, `human_turns`, `human_potions_used`, "
           "`human_potions_held_end`, `human_l0` (the search's currency: end HP plus 10 a potion held, a death -100; what practice "
           "regret compares with), `human_l0_unit` (the gym's L0 of 0 to 1, `fight_gym.l0`), `human_flags`.", "",
           "### The reading of end HP and what post-fight heals do to it", "",
           "The gym ends a fight when the last enemy dies and reads HP then: it has what the fight healed on the way and not what the run "
           "engine heals afterwards (Burning Blood 6, Black Blood 12, Meat on the Bone 12 at half HP or less, Chosen Cheese 1). The "
           "record's `current_hp` is after those and `hp_healed` is the floor's total, so the human's end HP is `current_hp` less "
           "min(`hp_healed`, the relics' post-fight heal).", "",
           f"- survivor floors checked: {ch.get('survivor floors', 0):,}; HP in - damage taken + healed = HP after holds on "
           f"{100 * ch.get('identity holds', 0) / n:.2f}%",
           f"- mean per survivor floor: HP in - HP after {hl.get('hp_in_minus_after', 0):.2f}, damage taken {hl.get('damage_taken', 0):.2f}, "
           f"post-fight heal {hl.get('post', 0):.2f}, heal inside the fight {hl.get('in_fight', 0):.2f}, HP lost as read {hl.get('hp_lost_read', 0):.2f}",
           f"- so `HP in - current HP` would understate the loss by the post-fight heal ({hl.get('post', 0):.2f} HP on average) and "
           f"`damage taken` would overstate it by the heal inside the fight ({hl.get('in_fight', 0):.2f} on average; "
           f"{100 * ch.get('floors with a heal beyond the relics own', 0) / n:.1f}% of fights heal beyond the relics)",
           "- a heal inside the fight and a potion drunk on the reward screen are not told apart in the record; the Meat on the Bone "
           "reading is a guess (`heal_guess`)", "",
           "### Coverage by act and kind", "", "| act and kind | starts with an outcome | runs | survived % | HP lost (survivors) | turns | potions used |",
           "|---|---|---|---|---|---|---|"]
    for k, v in rep["cells"].items():
        if v["runs"] < MIN_CELL:
            out.append(f"| {k} | {v['starts']:,} | < {MIN_CELL} | - | - | - | - |")
            continue
        out.append(f"| {k} | {v['starts']:,} | {v['runs']:,} | {100 * v['survived'] / max(1, v['starts']):.1f} | {v['mean_hp_lost_survivors']:.1f} | "
                   f"{v['mean_turns']:.1f} | {v['mean_potions_used']:.2f} |")
    out += ["", "### Starts without an outcome, by reason", ""]
    if rep["missing"]:
        out += [f"- {k}: {m:,}" for k, m in rep["missing"].items()]
    else:
        out.append("- none")
    out += ["", "### Flags and checks", ""]
    out += [f"- {k}: {v:,}" for k, v in sorted(ch.items()) if k.startswith("flag: ") or k == "hp_going_in_differs_from_the_start"] or ["- none"]
    out.append("")
    return "\n".join(out)


def write_doc_section(doc_path: Path, text: str, head: str = OUTCOME_HEAD) -> None:
    """Put `text` (a section that begins with `head`) into the doc: it replaces the old section of that name, else it is appended."""
    p = Path(doc_path)
    old = p.read_text(encoding="utf-8") if p.is_file() else ""
    i = old.find("\n" + head)
    body = old if i < 0 else old[:i + 1]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body.rstrip("\n") + "\n\n" + text, encoding="utf-8", newline="\n")


def atomic_json(path: Path, obj, patience_s: float = 10.0) -> None:
    """Written beside, then renamed over (the daemon reads `starts/` every segment); Windows refuses the rename while a reader has the
    target open, so it is tried again for `patience_s` seconds."""
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8", newline="\n")
    end = time.monotonic() + patience_s
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if time.monotonic() >= end:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.2)


def cmd_outcomes(a) -> int:
    d = Path(a.out_dir)
    lite = {r["id"]: r for r in json.loads((d / f"{STEM}.lite.json").read_text(encoding="utf-8"))}
    exam = read_exam_ids(a.exam_ids)
    assert_no_exam((r["run"] for r in lite.values()), exam)
    outcomes, rep = collect_outcomes(Path(a.corpus), exam, lite, a.limit_runs, log=lambda m: print(m, flush=True))
    doc = {"rule": OUTCOME_RULE, "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "source": "Spire Codex public run export for v0.107.1 (the four training fifths)", "starts": len(lite), "n": len(outcomes),
           "currency": "human_l0 is end HP plus 10 a potion held, a death -100 (the search's value, practice_mix's played value); "
                       "human_l0_unit is the gym's L0 of 0 to 1", "outcomes": outcomes}
    if not a.limit_runs or a.force_partial:
        atomic_json(d / OUTCOME_FILE, doc)
    else:
        print("human_starts outcomes: --limit-runs run: the sidecar is not written (--force-partial writes it)")
    text = outcomes_text(rep)
    print(text)
    if a.doc:
        write_doc_section(Path(a.doc), text)
    print(f"human_starts outcomes: {len(outcomes)} of {len(lite)} starts have an outcome")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--corpus", default=str(CORPUS))
    b.add_argument("--exam-ids", default=str(EXAM_IDS))
    b.add_argument("--out-dir", default=str(OUT_DIR))
    b.add_argument("--procs", type=int, default=4)
    b.add_argument("--hallway-cap", type=int, default=HALLWAY_CAP)
    b.add_argument("--limit-runs", type=int, default=0)
    b.add_argument("--doc", default=str(DOC_PATH), help="where the coverage report is written ('' for nowhere)")
    b.add_argument("--cache", default="", help="a file keeping the read corpus's fight states between runs (made when absent)")
    b.add_argument("--specs-only", action="store_true", help="read the corpus and write the cache only; deal nothing on the copy")
    r = sub.add_parser("report")
    r.add_argument("--out-dir", default=str(OUT_DIR))
    o = sub.add_parser("outcomes", help="the human's own result of every start, from the corpus's per-floor record (one pass)")
    o.add_argument("--corpus", default=str(CORPUS))
    o.add_argument("--exam-ids", default=str(EXAM_IDS))
    o.add_argument("--out-dir", default=str(OUT_DIR))
    o.add_argument("--limit-runs", type=int, default=0)
    o.add_argument("--force-partial", action="store_true", help="write the sidecar after a --limit-runs pass too")
    o.add_argument("--doc", default=str(DOC_PATH), help="the doc whose outcomes section is written ('' for nowhere)")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        below_normal()
        if a.procs > 4:
            raise SystemExit("human_starts: at most 4 processes")
        build(Path(a.corpus), Path(a.out_dir), a.procs, a.hallway_cap, a.limit_runs, Path(a.exam_ids), Path(a.doc) if a.doc else None,
              Path(a.cache) if a.cache else None, a.specs_only)
        return 0
    if a.cmd == "outcomes":
        below_normal()
        return cmd_outcomes(a)
    return cmd_report(a)


if __name__ == "__main__":
    raise SystemExit(main())
