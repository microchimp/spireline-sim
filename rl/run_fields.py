"""Per-run record fields the emulator arms and the real-game block share (runs.jsonl rows, D32 addendum item 2).

`RunFields` is fed by the two run drivers, `gen0.play_emu_run` (the emulator, from the run session's combat observation) and
`referee/lockstep.py` (the real game, from the referee's messages), and returns the same keys for both:

  boss_damage       the damage dealt to the Act 1 boss as a fraction of the HP its fight began with (1.0 when the fight was won,
                    0.0 when the run never reached that boss). This is the field `tools/lab/promotion_guard.py` reads (DAMAGE_FIELD).
  boss_damage_acts  [[act, fraction], ...], one entry per boss fight reached, in order (act is 0-based): the later bosses too.

A row without these keys is an older row: absent means unknown, never 0 (the guard stays inert on such rows).
The pure helpers here need neither the native library nor the game.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

BOSS_KEYS = ("boss_damage", "boss_damage_acts")


def obs_enemies(obs: Sequence[int], offset: int, slot_size: int, slots: int) -> list[tuple[int, int, int]]:
    """(slot, hp, max_hp) of every enemy slot of a combat observation (a dead enemy keeps its slot at 0 HP; an empty slot has no max)."""
    out = []
    for slot in range(slots):
        base = offset + slot * slot_size
        if int(obs[base + 1]) > 0:
            out.append((slot, int(obs[base]), int(obs[base + 1])))
    return out


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

    def row(self) -> dict:
        """The keys for the run's row (an open boss fight is closed as lost: the run stopped in it)."""
        self.boss_over(won=False)
        first = next((frac for act, frac in self.bosses if act == 0), 0.0)
        return {"boss_damage": first, "boss_damage_acts": [list(b) for b in self.bosses]}
