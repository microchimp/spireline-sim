"""Per-run measurement records for the seed exam (D59, the Act 2 measure): what the player built, fought and chose on a seed.

The exam's row (`run_seed_exam.play_task`) carries outcomes, the route, the card picks and the rest choices. This module adds, behind
`run_seed_exam.py play --records DIR`, one derived file a run: for every fight the deck, relics, potions, gold and HP the fight began with, its
enemies, its turns (the enemies' HP at the start and the end of each of our turns, our HP at the start of each, the cards and potions played) and
whether it was won; for every decision its kind, the option taken, and the relics, cards and potions offered, with HP, max HP, gold and deck size.

The recorder is the lab's own `gen_loop.RunRecorder` with accounting on top; it only reads the position and the search's answer, so the run is the
run the exam played without it (`gen0.play_emu_run(record=True)` is the lab's recording path). The files carry the seed's key, never the seed.
Nothing here needs the game: the pure parts are `TurnTracker` and `compact_decision`; `recorder_class()` imports `gen_loop` when a run starts.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

SCHEMA = 1
# A search result's `best.kind` (live_search._Move): the move the player makes now.
KIND_CARD, KIND_POTION, KIND_END_TURN = "card", "potion", "end_turn"
# Decisions whose options are not worth keeping (the map's own choices are in the row's `nodes`).
SKIP_OFFERS_PREFIX = "map."


class TurnTracker:
    """The turns of one fight from the player's searched decisions. A turn opens at the first decision after the last `end_turn` (or the first of the
    fight) and closes at its `end_turn`; the last turn of a won fight closes when the fight ends, with the enemies at 0 HP.

    A turn is `[enemy HP at its start, enemy HP at its end, our HP at its start, our HP at its end, cards played, potions drunk]`. The enemies' HP is
    the sum over every enemy slot the observation shows (minions that appear later are in the sum from the decision that sees them), so a turn's
    `start - end` is the HP the enemies lost during it, including by kills; a heal or a split can make it negative."""

    def __init__(self, enemies: list[list[int]]) -> None:
        self.enemies0 = [list(e) for e in enemies]  # [slot, hp, max_hp] at the fight's first observation
        self.turns: list[list[int]] = []
        self._open: list[int] | None = None
        self.potions_used = 0
        self.last_enemy_hp = sum(e[1] for e in enemies)
        self.last_hp = 0

    def decision(self, enemy_hp: int, hp: int, kind: str) -> None:
        if self._open is None:
            self._open = [int(enemy_hp), int(enemy_hp), int(hp), int(hp), 0, 0]
        t = self._open
        if kind == KIND_CARD:
            t[4] += 1
        elif kind == KIND_POTION:
            t[5] += 1
            self.potions_used += 1
        self.last_enemy_hp, self.last_hp = int(enemy_hp), int(hp)
        if kind == KIND_END_TURN:
            t[1], t[3] = int(enemy_hp), int(hp)
            self.turns.append(t)
            self._open = None

    def finish(self, won: bool) -> list[list[int]]:
        """The turns, with the turn that was open when the fight ended closed: at 0 enemy HP when the fight was won, at the last HP seen when it was not."""
        if self._open is not None:
            t = self._open
            t[1] = 0 if won else self.last_enemy_hp
            t[3] = self.last_hp
            self.turns.append(t)
            self._open = None
        return self.turns


def compact_decision(view: Any, decision: Any, pick_index: int) -> list[Any]:
    """One decision as `[floor, act, type, kind, card, upgraded, relic, potion, price, hp, max_hp, gold, deck size, offers]`. `offers` is
    `[[kind, id], ...]` of the screen's options (the id is the card, relic or potion the option carries, else -1), left out for a map choice."""
    opts = decision.options
    o = opts[pick_index]
    card = o.card
    offers: list[list[Any]] | None = None
    if not str(o.kind).startswith(SKIP_OFFERS_PREFIX):
        offers = []
        for x in opts:
            ident = x.card.card_id if x.card is not None else (x.relic if x.relic is not None else (x.potion if x.potion is not None else -1))
            offers.append([x.kind, ident])
    return [int(view.floor), int(view.act_index), decision.type, o.kind, None if card is None else int(card.card_id), None if card is None else int(card.upgraded),
            o.relic, o.potion, int(o.price or 0), int(view.hp), int(view.max_hp), int(view.gold), len(view.deck), offers]


class MeasureMixin:
    """Accounting on top of `gen_loop.RunRecorder` (listed after it in the bases): the lab's record is built exactly as before."""

    def _measure_init(self) -> None:
        if getattr(self, "_m_ready", False):
            return
        self._m_ready = True
        self.m_fights: list[dict[str, Any]] = []
        self.m_decisions: list[list[Any]] = []
        self.m_decks: list[list[list[int]]] = []
        self._m_cur: dict[str, Any] | None = None
        self._m_tracker: TurnTracker | None = None
        self.m_error: str | None = None

    def _deck_index(self, deck: list[list[int]]) -> int:
        try:
            return self.m_decks.index(deck)
        except ValueError:
            self.m_decks.append(deck)
            return len(self.m_decks) - 1

    def decision(self, view: Any, decision: Any, pick: Any) -> None:
        self._measure_init()
        super().decision(view, decision, pick)
        try:
            self.m_decisions.append(compact_decision(view, decision, pick.index))
        except Exception as e:  # noqa: BLE001 -- the measurement must never end a run
            self.m_error = f"decision: {type(e).__name__}"

    def open_fight(self, session: Any, combat: Any) -> None:
        self._measure_init()
        super().open_fight(session, combat)
        try:
            view = session.view()
            deck = sorted([int(c.card_id), int(c.upgraded)] for c in view.deck)
            self._m_cur = {"floor": int(view.floor), "act": int(view.act_index), "node": int(combat.node_type), "enc": int(combat.encounter_id),
                           "hp0": int(view.hp), "max_hp": int(view.max_hp), "gold": int(view.gold), "deck": self._deck_index(deck),
                           "relics": [int(r) for r in view.relics], "potions": [int(p) for p in view.potions]}
            self._m_tracker = TurnTracker(self.enemies_of(session))
        except Exception as e:  # noqa: BLE001
            self.m_error = f"open_fight: {type(e).__name__}"
            self._m_cur = self._m_tracker = None

    def enemies_of(self, session: Any) -> list[list[int]]:
        """`[slot, hp, max_hp]` of every enemy in the run observation the fight opened with."""
        import run_fields  # noqa: PLC0415
        from sts2_gym import native  # noqa: PLC0415

        return [list(e) for e in run_fields.obs_enemies(session.env._run_obs_buf, native.OBS_ENEMY_OFFSET, native.OBS_ENEMY_SLOT_SIZE, native.MAX_ENEMIES)]

    def enemy_hp_of(self, obs: Any) -> int:
        import run_fields  # noqa: PLC0415
        from sts2_gym import native  # noqa: PLC0415

        return sum(hp for _, hp, _ in run_fields.obs_enemies(obs, native.OBS_ENEMY_OFFSET, native.OBS_ENEMY_SLOT_SIZE, native.MAX_ENEMIES))

    def sample(self, root: Any, res: Any, tokens: Any) -> None:
        self._measure_init()
        super().sample(root, res, tokens)
        if self._m_tracker is None:
            return
        try:
            self._m_tracker.decision(self.enemy_hp_of(root.obs), int(root.obs[0]), str(res.best.kind))
        except Exception as e:  # noqa: BLE001
            self.m_error = f"sample: {type(e).__name__}"

    def close_fight(self, session: Any, won: bool) -> None:
        self._measure_init()
        f = self.fight
        hp_after = None if f is None else (max(0, int(session.info["hp"])) if won else 0)
        super().close_fight(session, won)
        cur, tr = self._m_cur, self._m_tracker
        self._m_cur = self._m_tracker = None
        if cur is None or tr is None:
            return
        try:
            cur.update(won=bool(won), hp1=hp_after, enemies=tr.enemies0, turns=tr.finish(bool(won)), potions_used=tr.potions_used)
            self.m_fights.append(cur)
        except Exception as e:  # noqa: BLE001
            self.m_error = f"close_fight: {type(e).__name__}"

    def finish(self, session: Any, moves: int, won: bool, **kw: Any) -> dict[str, Any]:
        self._measure_init()
        record = super().finish(session, moves, won, **kw)
        record["measure"] = {"schema": SCHEMA, "fights": self.m_fights, "decisions": self.m_decisions, "decks": self.m_decks, "error": self.m_error}
        return record


def recorder_class() -> type:
    """The recorder `gen0.play_emu_run(record=True)` should build: the lab's `RunRecorder` with the accounting above."""
    import gen_loop  # noqa: PLC0415

    return type("MeasureRecorder", (MeasureMixin, gen_loop.RunRecorder), {"__doc__": MeasureMixin.__doc__})


def write_record(directory: Path | str, key: str, draw: int, row: dict[str, Any], record: dict[str, Any]) -> Path:
    """The run's derived file `<key>-d<draw>.json.gz`: the exam row's outcome fields and the measure block, atomically. No seed, no run id."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    keep = {k: row.get(k) for k in ("end", "win", "floor", "nodes", "rests", "cards", "boss_damage_acts", "door", "decisions", "combat_decisions", "generation")}
    doc = {"key": key, "draw": int(draw), **keep, "measure": record.get("measure"), "outcome": record.get("outcome")}
    path = d / f"{key}-d{int(draw)}.json.gz"
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    os.replace(tmp, path)
    return path


def read_record(path: Path | str) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)
