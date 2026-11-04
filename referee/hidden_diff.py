"""The lockstep's second comparison: the fields the ranked comparison does not look at.

`lockstep.project` compares the run level, the hand and the enemies' HP. An enemy's block, its powers, the player's block
and powers and a relic's counter can differ for several decisions and only then move an HP number, which is why a record
whose first visible difference is an HP looked causeless (docs/diagnostics/2026-11-05-enemy-health.md, section 8). This
module reads those fields on both sides and names the ones that differ. The result is NOT ranked: it goes into the
divergence record under `hidden_diff` (never into `diff`, `n_diff`, the hashes, the reanchor plan or `agent_weighted`) and
into the run's stats, so the ranked groups, the by-family table and the agent-weighted figures are what they were.

What each side carries:

  * The game's per-decision stream (`obs.combat`, referee/driver/Combat.cs Snapshot): the player's block and powers and, per
    enemy, its HP, block and powers, at every combat decision. An enemy's next move and intents are carried too and are not
    compared here (the two sides name moves in different vocabularies; tools/fidelity/fight_compare.py maps them).
  * The game's relic counters and melted flag are NOT in the per-decision stream (`obs.relics` is the ids only). They come
    in the driver's `anchor_state` (referee/driver/AnchorHoldings.cs), which is sent at the re-anchor points only: a map
    decision and the first decision of a fight. So relic counters and the melted flag are compared at those decisions and
    at no other. `lockstep.EmuLock` makes that comparison (it holds the anchor state) and returns only the differences.
  * The copy's side is its observation buffer (the agent's own view: enemy block at slot +2, powers in the buff slots, the
    player's block at 2 and powers in theirs) and two run state lists (24: melted relics, 25: relic counters of the open
    fight). The buffer holds at most 5 powers an enemy and 10 the player, instanced powers summed in one slot; a creature
    whose slots are full may hold more than it shows, so a power the game has and the copy's list lacks is not a difference
    there.

Powers go through tools/fidelity/fight_compare.py (`game_powers`, `emu_powers`): the same folds, renames and internal
counters the lab's fight comparer uses, so the two comparers name a power the same way.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]

# CombatObservation.ValueBound: the copy's buffer holds no value above it. Both sides are compared through the same bound.
OBS_BOUND = 2 ** 15

# The hidden diffs of a record are capped like the ranked `diff` (referee/lockstep.py keeps 12).
RECORD_CAP = 12


def _fight_compare():
    try:
        from tools.fidelity import fight_compare  # noqa: PLC0415
    except ImportError:
        sys.path.insert(0, str(ROOT))
        from tools.fidelity import fight_compare  # noqa: PLC0415
    return fight_compare


def bound(v: Any) -> Any:
    return min(v, OBS_BOUND) if isinstance(v, int) and not isinstance(v, bool) else v


def _powers(pairs: list) -> dict:
    """{power: amount} from a (name, amount) list already in the game's vocabulary; a repeated name adds up."""
    out: dict = {}
    for name, amount in pairs:
        out[name] = bound(out.get(name, 0) + amount) if isinstance(amount, int) else amount
    return out


def game_powers(raw: Any) -> dict | None:
    """The game's `[[power id, amount], ...]` as {power: amount}, folds and renames applied; None when the stream has none."""
    if not isinstance(raw, list):
        return None
    pairs = [(p[0], p[1]) for p in raw if isinstance(p, (list, tuple)) and len(p) == 2]
    return _powers(_fight_compare().game_powers([[n, a] for n, a in pairs]))


def emu_powers(pairs: list) -> dict:
    """The copy's (BuffId name, amount) pairs as {power: amount} in the game's vocabulary."""
    return _powers(_fight_compare().emu_powers(pairs))


def game_view(obs: dict) -> dict | None:
    """The referee's fields at a combat decision, or None outside a fight. Only what the stream carries is present: a key the
    message lacks (a driver from before block or powers were recorded) is left out and compares as nothing."""
    c = obs.get("combat") if isinstance(obs, dict) else None
    if not isinstance(c, dict):
        return None
    out: dict = {}
    p = c.get("player")
    if isinstance(p, dict):
        out["player"] = {k: v for k, v in (("block", bound(p.get("block"))), ("powers", game_powers(p.get("powers"))))
                         if v is not None}
    enemies = []
    for e in c.get("enemies") or []:
        if not isinstance(e, dict):
            continue
        enemies.append({k: v for k, v in (("hp", bound(e.get("hp", e.get("current_hp")))), ("block", bound(e.get("block"))),
                                          ("powers", game_powers(e.get("powers")))) if v is not None})
    out["enemies"] = enemies
    return out


def _slots(obs, at: int, slots: int, names: dict) -> tuple[list, bool]:
    """The (BuffId name, amount) pairs of one creature's buff slots, and whether every slot is taken. An empty slot reads
    (0, 0); id 0 with an amount is Strength (rl/combat_tokens.py)."""
    pairs = []
    for k in range(slots):
        ident, amount = int(obs[at + 2 * k]), int(obs[at + 2 * k + 1])
        if ident == 0 and amount == 0:
            continue
        pairs.append((names.get(ident, f"?{ident}"), amount))
    return pairs, len(pairs) >= slots


def emu_view(obs, *, player_buffs_at: int, player_buff_slots: int, enemies_at: int, enemy_size: int, enemy_slots: int,
             enemy_buff_slots: int, names: dict) -> dict:
    """The copy's fields from its observation buffer (the layout is the native library's: rl/live_search.py's `_ENEMIES`,
    `_ENEMY`, `_ENEMY_BUFFS`; `OBS_PLAYER_BUFF_OFFSET`, `OBS_MAX_PLAYER_BUFFS`). `names` maps a BuffId to its name. Each
    creature carries `full`: its buff slots are all taken, so the list may be a part of what it holds."""
    pairs, full = _slots(obs, player_buffs_at, player_buff_slots, names)
    out: dict = {"player": {"block": bound(int(obs[2])), "powers": emu_powers(pairs), "full": full}, "enemies": []}
    for k in range(enemy_slots):
        at = enemies_at + k * enemy_size
        hp, mx = int(obs[at]), int(obs[at + 1])
        if hp <= 0 and mx <= 0:
            continue  # a slot past the roster; a corpse the roster still holds has max HP and lists at 0
        pairs, full = _slots(obs, at + 5, enemy_buff_slots, names)
        out["enemies"].append({"hp": bound(hp), "block": bound(int(obs[at + 2])), "powers": emu_powers(pairs), "full": full})
    return out


def _power_diff(path: str, game: dict | None, emu: dict | None, emu_full: bool) -> list:
    if game is None or emu is None:
        return []  # the stream did not carry the powers
    out = []
    for name in sorted(set(game) | set(emu)):
        g, e = game.get(name), emu.get(name)
        if g == e or (g is not None and e is None and emu_full):
            continue  # equal, or a power the copy's full slots may not show
        out.append([f"{path}.{name}", g, e])
    return out


def diff(game: dict | None, emu: dict | None) -> list:
    """[path, game, copy] for every field the two sides carry and hold differently; paths below `.combat`. Per-enemy fields
    are compared by roster position, for the enemies both sides hold alive; a roster of another size is one entry (the
    ranked comparison reports it as the enemies' HP list)."""
    if game is None or emu is None:
        return []
    out: list = []
    gp, ep = game.get("player"), emu.get("player")
    if gp and ep:
        if gp.get("block") is not None and gp["block"] != ep["block"]:
            out.append([".combat.player.block", gp["block"], ep["block"]])
        out += _power_diff(".combat.player.powers", gp.get("powers"), ep["powers"], ep["full"])
    ge, ee = game.get("enemies", []), emu.get("enemies", [])
    if len(ge) != len(ee):
        out.append([".combat.enemies.count", len(ge), len(ee)])
        return out
    for k, (g, e) in enumerate(zip(ge, ee)):
        if g.get("hp", 1) <= 0 or e["hp"] <= 0:
            continue  # a corpse keeps no block and no powers worth comparing
        if g.get("block") is not None and g["block"] != e["block"]:
            out.append([f".combat.enemies[{k}].block", g["block"], e["block"]])
        out += _power_diff(f".combat.enemies[{k}].powers", g.get("powers"), e["powers"], e["full"])
    return out


def relic_diff(game_relics: Any, emu_counters: dict, emu_melted: list | None, writes: Callable[[list], list]) -> list:
    """[path, game, copy] for the relic counters and the melted flag, from the driver's anchor-state relics (`[{"id", "ints",
    "bools"}, ...]`). `writes` is refset.relic_counter_writes: the game's counters in the copy's own units, for the relics it
    covers. `emu_counters` is {relic: counter} of the relics in play; a relic the copy does not hold is the ranked relic
    comparison's, not this one. `emu_melted` is the copy's melted relics, or None when its library cannot say."""
    if not isinstance(game_relics, list):
        return []
    melted = sorted(str(r["id"]) for r in game_relics if isinstance(r, dict) and (r.get("bools") or {}).get("IsMelted"))
    held = [r for r in game_relics if isinstance(r, dict) and str(r.get("id")) not in melted]
    out = []
    for relic, counter in writes(held):
        got = emu_counters.get(relic)
        if got is not None and got != counter:
            out.append([f".relics.{relic}.counter", counter, got])
    if emu_melted is not None and melted != sorted(emu_melted):
        out.append([".relics.melted", melted, sorted(emu_melted)])
    return out


_KIND = re.compile(r"^\.(combat\.(?P<who>player|enemies)(\[\d+\])?\.(?P<what>block|powers|count)(\.(?P<name>.+))?"
                   r"|relics\.(?P<relic>melted)|relics\.(?P<rid>[^.]+)\.counter)$")


def kind(path: str) -> str:
    """The summary's name for a difference's path: enemy_block, enemy_power:<POWER>, player_block, player_power:<POWER>,
    relic_counter:<RELIC>, relic_melted, roster (the enemies' count)."""
    m = _KIND.match(path)
    if not m:
        return "other"
    if m["relic"]:
        return "relic_melted"
    if m["rid"]:
        return f"relic_counter:{m['rid']}"
    who = "enemy" if m["who"] == "enemies" else "player"
    if m["what"] == "count":
        return "roster"
    return f"{who}_block" if m["what"] == "block" else f"{who}_power:{m['name']}"
