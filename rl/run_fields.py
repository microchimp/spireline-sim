"""Per-run record fields the emulator arms and the real-game block share (runs.jsonl rows, D32 addendum item 2).

`RunFields` is fed by the two run drivers, `gen0.play_emu_run` (the emulator, from the run session's combat observation) and
`referee/lockstep.py` (the real game, from the referee's messages), and returns the same keys for both:

  boss_damage       the damage dealt to the Act 1 boss as a fraction of the HP its fight began with (1.0 when the fight was won,
                    0.0 when the run never reached that boss). This is the field `tools/lab/promotion_guard.py` reads (DAMAGE_FIELD).
  boss_damage_acts  [[act, fraction], ...], one entry per boss fight reached, in order (act is 0-based): the later bosses too.
  cards             [[floor, "NAME" | "NAME+" | "skip"], ...]: each card reward decided, by the floor it was decided on, the card
                    taken (+ upgraded) or skipped; a reroll or a relic alternative is not recorded.
  nodes             the map node types taken in order, one letter each: M monster, E elite, R rest, S shop, T treasure, B boss,
                    ? unknown, A ancient (x: a type this table lacks).
  rests             [[floor, "heal" | "smith" | ...], ...]: each rest site choice (leaving is not recorded).
Shop buys are not recorded: the referee's shop label carries a slot and a price, not the item, so a buy needs new plumbing.

A row without these keys is an older row: absent means unknown, never 0 (the guard stays inert on such rows).
The pure helpers here need neither the native library nor the game.
"""

from __future__ import annotations

import functools
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BOSS_KEYS = ("boss_damage", "boss_damage_acts")
MACRO_KEYS = ("cards", "nodes", "rests")
ROW_KEYS = BOSS_KEYS + MACRO_KEYS
SKIP = "skip"
# The map node types, one letter each, as the referee names them (MapPointType) and as the emulator numbers them (vocab.NODE_*).
NODE_CODES = {"Monster": "M", "Elite": "E", "RestSite": "R", "Shop": "S", "Treasure": "T", "Boss": "B", "Unknown": "?", "Ancient": "A"}
# The floor of each act's boss, counted across acts (obs.floor, the game's TotalFloor): ActModel.GetNumberOfFloors is rooms + 2 (the ancient
# floor and the boss), 17 / 16 / 15 floors for the three acts (tools/fidelity/readiness.ACT_ENDS holds the same numbers, a test checks it).
ACT_BOSS_FLOORS = (17, 33, 48)
# Words the referee uses for leaving a screen: not a choice of the screen.
LEAVE_LABELS = ("leave", "proceed")
EMU_NODE_CODES = {1: "M", 2: "E", 3: "R", 4: "S", 5: "T", 6: "B", 7: "?", 8: "A"}


@functools.lru_cache(maxsize=1)
def card_slugs() -> dict[int, str]:
    """Emulator card id -> the referee's card name (the lockstep's slug), so both sides write one id scheme."""
    from run_train import vocab  # noqa: PLC0415

    ids = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))["cards"]
    return {v: vocab.game_slug(n) for n, v in ids.items()}


def obs_enemies(obs: Sequence[int], offset: int, slot_size: int, slots: int) -> list[tuple[int, int, int]]:
    """(slot, hp, max_hp) of every enemy slot of a combat observation (a dead enemy keeps its slot at 0 HP; an empty slot has no max)."""
    out = []
    for slot in range(slots):
        base = offset + slot * slot_size
        if int(obs[base + 1]) > 0:
            out.append((slot, int(obs[base]), int(obs[base + 1])))
    return out


def is_boss_floor(act: Any, floor: Any) -> bool:
    """Whether `floor` of act `act` (0-based; the referee's obs.act and obs.floor) is the boss floor of that act or past it (act 3's
    second boss sits one floor on). The real stream asks no map decision when the boss is the only node left, so the last map label
    cannot name the boss fight; the observation does."""
    if not isinstance(act, int) or not isinstance(floor, int) or not 0 <= act < len(ACT_BOSS_FLOORS):
        return False
    return ACT_BOSS_FLOORS[act] <= floor <= ACT_BOSS_FLOORS[act] + (1 if act == len(ACT_BOSS_FLOORS) - 1 else 0)


def real_enemies(combat: dict) -> list[tuple[Any, int, int]]:
    """(key, hp, max_hp) of the enemies in the referee's `obs.combat`; the key is the enemy's name and its place among the
    enemies of that name, so a minion that spawns later does not move the keys of the ones the fight began with."""
    seen: dict[str, int] = {}
    out = []
    for e in combat.get("enemies") or []:
        if not isinstance(e, dict):
            continue
        hp, mx = e.get("hp", e.get("current_hp")), e.get("max_hp")
        name = str(e.get("name"))
        k = seen[name] = seen.get(name, -1) + 1
        if isinstance(hp, int) and isinstance(mx, int):
            out.append(((name, k), hp, mx))
    return out


class RunFields:
    """One run's record fields, filled as the run is played."""

    def __init__(self) -> None:
        self.bosses: list[list] = []  # [act, damage fraction] per boss fight, in order
        self._fight: dict | None = None  # the open boss fight
        self.cards: list[list] = []  # [floor, card name | name+ (upgraded) | "skip"] per card reward decided
        self.nodes: list[str] = []  # the map node types taken, one letter each, in order
        self.rests: list[list] = []  # [floor, "heal" | "smith" | ...] per rest site choice

    @property
    def in_boss_fight(self) -> bool:
        return self._fight is not None

    def boss_enemies(self, act: int, enemies: Iterable[tuple[Any, int, int]]) -> None:
        """The enemies at one decision of a boss fight, (key, hp, max_hp) each; the first call opens the fight. Damage is
        measured on the creatures the fight began with (a minion it summons later is neither counted nor allowed to cancel
        damage), at the lowest HP they reached."""
        now = {k: max(0, int(hp)) for k, hp, _mx in enemies}
        f = self._fight
        if f is None:
            start = {k: hp for k, hp in now.items() if hp > 0}
            if not start:
                return
            self._fight = {"act": int(act), "start": start, "total": sum(start.values()), "left": sum(start.values())}
            return
        f["left"] = min(f["left"], sum(min(now.get(k, 0), hp) for k, hp in f["start"].items()))

    def boss_over(self, won: bool) -> None:
        """The boss fight ended: won (every creature it began with is dead) or the run ended in it."""
        f, self._fight = self._fight, None
        if f is not None:
            self.bosses.append([f["act"], 1.0 if won else round((f["total"] - f["left"]) / f["total"], 4)])

    def finish(self, won_run: bool) -> None:
        """The run ended: a boss fight still open was the last thing played (won with the run, else lost in it)."""
        self.boss_over(won=won_run)

    def card(self, floor: int, name: str, upgraded: bool = False) -> None:
        self.cards.append([int(floor), name + ("+" if upgraded else "")])

    def skip(self, floor: int) -> None:
        self.cards.append([int(floor), SKIP])

    def emu_macro(self, floor: int, option: Any, slugs: dict[int, str]) -> None:
        """The option the macro policy took on the emulator side (run_train.features.Option)."""
        kind = option.kind
        if kind == "card.take":
            self.card(floor, slugs.get(option.card.card_id, f"?{option.card.card_id}"), option.card.upgraded)
        elif kind == "card.skip":
            self.skip(floor)
        elif kind == "map.node":
            self.nodes.append(EMU_NODE_CODES.get(option.node[2], "x"))
        elif kind.startswith("rest.") and kind != "rest.leave":
            self.rests.append([int(floor), kind[5:]])

    def real_macro(self, kind: str, label: Any, floor: Any) -> None:
        """The referee action the lockstep took (its label), by decision kind. A card reward's other alternatives (a reroll, a
        relic) are neither a card nor a skip and are not recorded."""
        floor = floor if isinstance(floor, int) else -1
        if kind == "card_reward":
            if isinstance(label, list) and label:
                self.card(floor, str(label[0]), bool(label[1]) if len(label) > 1 else False)
            elif label in ("skip", "Skip"):
                self.skip(floor)
        elif kind == "map" and isinstance(label, list) and len(label) > 2:
            self.nodes.append(NODE_CODES.get(str(label[2]), "x"))
        elif kind == "rest" and isinstance(label, str) and label.lower() not in LEAVE_LABELS:
            # The referee's label (HEAL, SMITH, LIFT, DIG, TOKE, ...) in lower case is the emulator's kind (lockstep.EmuLock.REST); a
            # label this table lacks is recorded as its own text, never as a heal.
            self.rests.append([floor, label.lower()])

    def row(self) -> dict:
        """The keys for the run's row (an open boss fight is closed as lost: the run stopped in it)."""
        self.boss_over(won=False)
        first = next((frac for act, frac in self.bosses if act == 0), 0.0)
        return {"boss_damage": first, "boss_damage_acts": [list(b) for b in self.bosses],
                "cards": [list(c) for c in self.cards], "nodes": "".join(self.nodes), "rests": [list(r) for r in self.rests]}


def aggregate(rows: Iterable[dict]) -> dict:
    """The tables' per-arm numbers from runs.jsonl rows: mean boss damage over the rows that carry it, and the card reward
    choices (taken per run, taken and skipped by floor, skip rate) over the rows that carry `cards`. A key is absent when no
    row has its field (an older block): absent is unknown, never 0."""
    rows = list(rows)
    out: dict = {}
    dmg = [float(r["boss_damage"]) for r in rows if r.get("boss_damage") is not None]
    unknown = sum(1 for r in rows if "boss_damage" in r and r["boss_damage"] is None)  # a backfilled lost fight: its damage was not logged
    if dmg or unknown:
        out["boss_damage"] = {"n": len(dmg), "mean": round(sum(dmg) / len(dmg), 4) if dmg and not unknown else None}
        if unknown:
            out["boss_damage"]["unknown"] = unknown
    acts = [r["boss_damage_acts"] for r in rows if isinstance(r.get("boss_damage_acts"), list)]
    if acts:
        fights = [next((f for a, f in b if a == 0), "none") for b in acts]
        reached = [f for f in fights if f != "none"]
        won = sum(1 for f in reached if f == 1.0)
        out["boss_act1"] = {"runs": len(acts), "reached": len(reached), "won": won,
                            "kill_rate": round(won / len(reached), 4) if reached else None}
    lists = [r["cards"] for r in rows if isinstance(r.get("cards"), list)]
    if lists:
        taken: Counter = Counter()
        skipped: Counter = Counter()
        for cards in lists:
            for floor, what in cards:
                (skipped if what == SKIP else taken)[int(floor)] += 1
        t, s = sum(taken.values()), sum(skipped.values())
        out["cards"] = {"runs": len(lists), "decisions": t + s, "taken": t, "skipped": s,
                        "skip_rate": round(s / (t + s), 4) if t + s else None, "taken_per_run": round(t / len(lists), 3),
                        "taken_by_floor": {str(f): taken[f] for f in sorted(taken)},
                        "skipped_by_floor": {str(f): skipped[f] for f in sorted(skipped)}}
    return out
