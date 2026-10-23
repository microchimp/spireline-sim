"""Align the game's and the emulator's trace of one step and name the first event that differs.

    python -m tools.fidelity.trace_diff <traced referee out.json> <EmuDump reanchor --trace output>
        [--seed SEED] [--step J] [--json out.json]

Both sides record, inside each step, the same marks (play / played a card, turn_end, enemy_turn, move /
moved a monster, turn_start) with the visible state at each, and every draw from a combat RNG stream
with its caller (referee/driver/Trace.cs, tools/emudump/Engines/Old/OldTrace.cs). For the first step whose result
differs (or --step), the marks are aligned by their sequence; the first aligned mark at which the two
states differ closes the segment where the divergence happened. For that segment it prints what changed
on each side (as events: HP, block, energy, powers, cards into and out of hand and the piles), the
game's own combat log, actions and hooks there, and both sides' draws. The draws of the whole step are
aligned too: the first draw that one side makes and the other does not (or makes from another stream)
names the caller on each side. Hidden state the step started from (card costs and enchantments, relic
counters) is compared where both sides can say it.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import sys
from pathlib import Path

from tools.fidelity.fight_compare import (
    EMU_RENAME,
    NOT_POWERS,
    diff,
    emu_view,
    game_powers,
    game_view,
    names,
    power,
)

ROOT = Path(__file__).resolve().parents[2]
COMBAT_STREAMS = {"Shuffle", "CombatTargets", "MonsterAi", "CombatCardSelection", "CombatCardGeneration",
                  "CombatEnergyCosts", "CombatPotionGeneration", "CombatOrbGeneration", "Niche"}


# ---------------------------------------------------------------- normalised marks

def _emu_powers(raw) -> list:
    return sorted([EMU_RENAME.get(power(n), power(n)), a] for n, a in raw if power(n) not in NOT_POWERS)


def view(mini: dict, side: str) -> dict:
    """A mark's state in one vocabulary (fight_compare's power names)."""
    pw = game_powers if side == "game" else _emu_powers
    hp, block, energy = mini["player"]
    return {"player": {"hp": hp, "block": block, "energy": energy, "powers": pw(mini["powers"])},
            "enemies": [{"name": n, "hp": h, "block": b, "powers": pw(p)} for n, h, b, p in mini["enemies"]],
            "hand": list(mini["hand"]), "n": {"draw": mini["n"][0], "discard": mini["n"][1], "exhaust": mini["n"][2]}}


def state_view(gv: dict) -> dict:
    """fight_compare's game view of a whole step state, cut to what a mark shows."""
    if not gv.get("in_combat"):
        return {"over": True}
    p = gv["player"]
    return {"player": {"hp": p["hp"], "block": p["block"], "energy": p["energy"], "powers": p["powers"]},
            "enemies": [{"name": e["name"], "hp": e["hp"], "block": e["block"], "powers": e["powers"]} for e in gv["enemies"]],
            "hand": [c[0] for c in gv["hand"]],
            "n": {"draw": len(gv["draw"]), "discard": len(gv["discard"]), "exhaust": len(gv["exhaust"])}}


def deltas(a: dict, b: dict) -> list[str]:
    """What changed from mark state a to b, as short events."""
    if a.get("over") or b.get("over"):
        return ["fight over"] if b.get("over") and not a.get("over") else []
    out = []
    for k in ("hp", "block", "energy"):
        if a["player"][k] != b["player"][k]:
            out.append(f"player {k} {b['player'][k] - a['player'][k]:+d}")
    out += _power_deltas("player", a["player"]["powers"], b["player"]["powers"])
    ea = collections.defaultdict(list)
    for e in a["enemies"]:
        ea[e["name"]].append(e)
    seen = collections.Counter()
    for e in b["enemies"]:
        k = seen[e["name"]]
        seen[e["name"]] += 1
        prev = ea[e["name"]][k] if k < len(ea[e["name"]]) else None
        who = f"{e['name']}#{k}"
        if prev is None:
            out.append(f"{who} appears")
            continue
        for f in ("hp", "block"):
            if prev[f] != e[f]:
                out.append(f"{who} {f} {e[f] - prev[f]:+d}")
        out += _power_deltas(who, prev["powers"], e["powers"])
    for name, lst in ea.items():
        for k in range(seen[name], len(lst)):
            out.append(f"{name}#{k} gone")
    ha, hb = collections.Counter(a["hand"]), collections.Counter(b["hand"])
    out += [f"hand +{c}" for c in sorted((hb - ha).elements())] + [f"hand -{c}" for c in sorted((ha - hb).elements())]
    for k in ("draw", "discard", "exhaust"):
        if a["n"][k] != b["n"][k]:
            out.append(f"{k} {b['n'][k] - a['n'][k]:+d}")
    return out


def _power_deltas(who: str, a: list, b: list) -> list[str]:
    da, db = dict(map(tuple, a)), dict(map(tuple, b))
    return [f"{who} {n} {da.get(n, 0)}->{db.get(n, 0)}" for n in sorted(set(da) | set(db)) if da.get(n) != db.get(n)]


def marks(trace: list, side: str) -> list[dict]:
    """The marks of one step with the events between each mark and the one before it."""
    out, pending = [], []
    for ev in trace:
        if ev[0] == "M":
            out.append({"label": f"{ev[1]} {ev[2]}" if ev[2] else ev[1], "view": view(ev[3], side), "before": pending})
            pending = []
        else:
            pending.append(ev)
    out.append({"label": "end", "view": None, "before": pending})
    return out


# ---------------------------------------------------------------- RNG alignment

def draws(trace: list, side: str) -> list[tuple]:
    """(stream, counter, caller) of every combat-stream draw, in order, each draw once. A game draw
    made while a monster rolls its move names the monster in the caller (MonsterModel.RollMove)."""
    out, last, rolling = [], {}, None
    for ev in trace:
        if ev[0] == "A" and ev[1].startswith("RollMove "):
            rolling = ev[1][len("RollMove "):]
        elif ev[0] == "E" and ev[1] == "MoveRolled":
            rolling = None
        if ev[0] == "R" and (ev[1] in COMBAT_STREAMS or (side == "emu" and ev[1] == "fallback")):
            if last.get(ev[1]) == ev[2]:
                continue  # a nested Next* call logging the same draw
            last[ev[1]] = ev[2]
            out.append((ev[1], ev[2], ev[3] + (f" rolling {rolling}" if rolling else "")))
    return out


def first_draw_difference(g: list, e: list):
    """Per stream, the first draw one side makes and the other does not.

    Streams are compared on their own: the two sides may interleave streams differently (the game
    rolls monster moves before the turn's draw, the emulator after) without drawing differently.
    The stream reported is the one whose first extra draw comes earliest in the game's order (or
    the emulator's, for a draw only it makes).
    """
    found = []
    for stream in dict.fromkeys([x[0] for x in g] + [x[0] for x in e]):
        gs, es = [x for x in g if x[0] == stream], [x for x in e if x[0] == stream]
        k = 0
        while k < min(len(gs), len(es)) and gs[k][1] == es[k][1]:
            k += 1
        if k == len(gs) == len(es):
            continue
        order = g.index(gs[k]) if k < len(gs) else len(g) + e.index(es[k])
        found.append((order, {"stream": stream, "after_same": k, "game": gs[k:k + 3], "emu": es[k:k + 3],
                              "game_total": len(gs), "emu_total": len(es),
                              "same_callers": [(a[2], b[2]) for a, b in zip(gs[:k], es[:k])][:1]}))
    return min(found, key=lambda t: t[0])[1] if found else None


# ---------------------------------------------------------------- hidden state

def hidden_cards(h: dict | None, side: str) -> dict:
    """Cards by pile as (id, cost, enchantment+amount): what both sides can say about a card underneath."""
    out = collections.defaultdict(list)
    for c in (h or {}).get("cards", []):
        pile, cid, cost = c[0], c[1], c[2]
        ench = c[3]
        if ench and side == "emu":
            ench = power(ench)  # Enchantment enum names -> UPPER_SNAKE like the game's ids
        out[pile].append((cid, cost, f"{ench}:{c[4]}" if ench else ""))
    return out


def hidden_diffs(gh: dict | None, eh: dict | None) -> list[str]:
    if gh is None or eh is None:
        return []
    g, e = hidden_cards(gh, "game"), hidden_cards(eh, "emu")
    out = []
    for pile in ("hand", "draw", "discard", "exhaust"):
        a, b = sorted(g.get(pile, [])), sorted(e.get(pile, []))
        if [x[0] for x in a] != [x[0] for x in b]:
            continue  # different cards: the visible state already says so
        for x, y in zip(a, b):
            if x[1] == -2:  # an X cost (the game's marker); the emulator's number is the energy it would spend
                x, y = (x[0], 0, x[2]), (y[0], 0, y[2])
            if x != y:
                out.append(f"{pile} {x[0]}: game cost {x[1]} {x[2] or '-'}, emulator cost {y[1]} {y[2] or '-'}")
    grel = {r[0]: r[1] for r in gh.get("relics", []) if r[1] is not None}
    erel = {r[0]: r[1] for r in eh.get("relics", [])}
    for rid, amount in sorted(grel.items()):
        if rid in erel and erel[rid] != amount:
            out.append(f"relic {rid}: game counter {amount}, emulator {erel[rid]}")
    return out


# ---------------------------------------------------------------- one step

def dedupe(evs: list) -> list:
    """Each draw once (a nested Next* call logs the draw its caller logs again)."""
    out, last = [], {}
    for ev in evs:
        if ev[0] == "R":
            if last.get(ev[1]) == ev[2]:
                continue
            last[ev[1]] = ev[2]
        out.append(ev)
    return out


def compress(items: list[str]) -> list[str]:
    """Runs of one line as "line x N"; draws in a run from one caller as "draw S#a..b by caller"."""
    out = []
    for it in items:
        if out and out[-1][0] == it:
            out[-1][1] += 1
            continue
        if out and it.startswith("draw ") and out[-1][0].startswith("draw "):
            a, b = out[-1][0].split(" by ", 1), it.split(" by ", 1)
            sa, sb = a[0].split("#"), b[0].split("#")
            if len(a) == 2 and len(b) == 2 and a[1] == b[1] and sa[0] == sb[0]:
                first = sa[1].split("..")[0]
                out[-1][0] = f"{sa[0]}#{first}..{sb[1]} by {a[1]}"
                continue
        out.append([it, 1])
    return [f"{t} x{n}" if n > 1 else t for t, n in out]


def fmt_ev(ev) -> str:
    if ev[0] == "R":
        return f"draw {ev[1]}#{ev[2]} by {ev[3]}"
    if ev[0] == "E":
        return "log " + " ".join(str(x) for x in ev[1:] if x is not None)
    if ev[0] == "A":
        return "action " + ev[1]
    if ev[0] == "H":
        return "hook " + " ".join(ev[1:])
    return str(ev)


def mask(v: dict, keys) -> dict:
    """Powers only one side can name (reanchor_compare._mask): a game power the emulator models in pieces,
    and the emulator's own bookkeeping buffs. ``keys`` = (game power keys the emulator has, its internal keys)."""
    if keys is None or v.get("over"):
        return v
    buff_keys, internal = keys
    buff_keys = {EMU_RENAME.get(k, k) for k in buff_keys} | {EMU_RENAME[k] for k in internal if k in EMU_RENAME}
    internal = set(internal) - set(EMU_RENAME)
    keep = lambda ps: [p for p in ps if p[0] in buff_keys and p[0] not in internal]  # noqa: E731
    out = json.loads(json.dumps(v))
    out["player"]["powers"] = keep(out["player"]["powers"])
    for e in out["enemies"]:
        e["powers"] = keep(e["powers"])
    return out


def involved(changes: list[str]) -> set:
    """Holders named in a list of delta events ("player", "CORPSE_SLUG#0")."""
    return {c.split(" ")[0] for c in changes if c.split(" ")[0] == "player" or "#" in c.split(" ")[0]}


def context(gh: dict | None, who: set) -> list[str]:
    """The game's powers and relics of the holders involved, with the fields the snapshot does not show."""
    if not gh:
        return []
    out = []
    for holder, pid, amount, fields in gh.get("powers", []):
        name = "player" if holder == "player" else holder
        base = holder.split("#")[0] + "#"
        if name in who or any(w.startswith(base) for w in who):
            f = ", ".join(f"{k}={v}" for k, v in fields.items() if v not in (None, 0, False, ""))
            out.append(f"{holder} {pid} {amount}" + (f" ({f})" if f else ""))
    if "player" in who:
        rel = [f"{rid}={amt}" for rid, amt, _ in gh.get("relics", []) if amt is not None]
        if rel:
            out.append("relic counters: " + ", ".join(rel))
    return out[:8]


def step_diff(game: dict, emu: dict, j: int, nm: dict, keys=None) -> dict:
    """Everything trace_diff knows about step j of one fight, as a dict (text() prints it).

    Each side's marks are measured from that side's own state at the step start (the emulator's is its
    re-anchored state, the "start" mark), so a power both name differently from the start cancels out;
    the first aligned mark at which the changes so far differ closes the segment that diverged.
    """
    gs, es = game["steps"][j], emu["steps"][j]
    out = {"seed": game.get("seed"), "step": j, "action": gs.get("action")}
    if "trace" not in gs or "trace" not in es:
        out["error"] = "no trace on the " + ("game" if "trace" not in gs else "emulator") + " side for this step"
        return out
    gv, ev = game_view(gs["state"]), emu_view(es["state"], nm)
    out["result_diffs"] = [p for p, _, _ in diff(gv, ev)]
    gm, em = marks(gs["trace"], "game"), marks(es["trace"], "emu")
    g_start = state_view(game_view(game["steps"][j - 1]["state"]))
    e_start = g_start
    if em and em[0]["label"] == "start":
        e_start = em[0]["view"]
        em[1]["before"] = em[0]["before"] + em[1]["before"]
        em = em[1:]
    gm[-1]["view"], em[-1]["view"] = state_view(gv), state_view(ev)
    g_start, e_start = mask(g_start, keys), mask(e_start, keys)
    for m in gm + em:
        m["view"] = mask(m["view"], keys)
    # The re-anchored start the emulator was given against the game's: a difference here is the
    # write-back's (Reanchor.cs), there before the step did anything.
    out["start_diffs"] = [f"{p_.strip('.')}: game {x} emulator {y}" for p_, x, y in diff(g_start, e_start)][:4]
    sm = difflib.SequenceMatcher(a=[m["label"] for m in gm], b=[m["label"] for m in em], autojunk=False)
    pairs = [(i, k) for blk in sm.get_matching_blocks() for i, k in zip(range(blk.a, blk.a + blk.size), range(blk.b, blk.b + blk.size))]
    if not pairs or pairs[-1] != (len(gm) - 1, len(em) - 1):
        pairs.append((len(gm) - 1, len(em) - 1))  # "end" closes both
    prev_g, prev_e = -1, -1
    for i, k in pairs:
        a, b = gm[i]["view"], em[k]["view"]
        if collections.Counter(deltas(g_start, a)) != collections.Counter(deltas(e_start, b)) or (a != b and g_start == e_start):
            pa = gm[prev_g]["view"] if prev_g >= 0 else g_start
            pb = em[prev_e]["view"] if prev_e >= 0 else e_start
            seg_g = [ev_ for m in gm[prev_g + 1:i + 1] for ev_ in m["before"]]
            seg_e = [ev_ for m in em[prev_e + 1:k + 1] for ev_ in m["before"]]
            ga, eb = deltas(pa, a), deltas(pb, b)
            only_g = [x for x in ga if x not in eb]
            only_e = [x for x in eb if x not in ga]
            if not only_g and not only_e:  # the same changes, landing differently (an order, a stack)
                only_g = [f"{p_.strip('.')} {x}" for p_, x, _ in diff(a, b)][:3]
                only_e = [f"{p_.strip('.')} {y}" for p_, _, y in diff(a, b)][:3]
            evs = [x for x in dedupe(seg_g) if x[0] in "EAH" or x[1] in COMBAT_STREAMS]
            out["segment"] = {
                "from": gm[prev_g]["label"] if prev_g >= 0 else "step start", "to": gm[i]["label"],
                "unmatched_game_marks": [m["label"] for m in gm[prev_g + 1:i]],
                "unmatched_emu_marks": [m["label"] for m in em[prev_e + 1:k]],
                "game": ga, "emu": eb,
                "game_log": compress([fmt_ev(x) for x in evs if x[0] in "EA"])[:20],
                "game_hooks": compress([fmt_ev(x) for x in evs if x[0] == "H"])[:20],
                "game_draws": compress([fmt_ev(x) for x in evs if x[0] == "R"])[:10],
                "emu_draws": compress([fmt_ev(x) for x in dedupe(seg_e) if x[0] == "R"])[:10],
            }
            out["first_difference"] = (f"between {out['segment']['from']} and {out['segment']['to']}: "
                                       f"game {', '.join(only_g) or 'nothing'}; emulator {', '.join(only_e) or 'nothing'}")
            out["context"] = context(game["steps"][j - 1].get("hidden"), involved(only_g + only_e) or {"player"})
            break
        prev_g, prev_e = i, k
    out["draw_difference"] = first_draw_difference(draws(gs["trace"], "game"), draws(es["trace"], "emu"))
    out["hidden"] = hidden_diffs(game["steps"][j - 1].get("hidden"), es.get("hidden_pre"))
    out["unsettled"] = unsettled(gs, es)
    return out


def unsettled(gs: dict, es: dict) -> str | None:
    """Why the game's record of an end of turn may have been taken before the game finished it.

    The game's player turn start ends with Hook.AfterSideTurnStart (CombatManager.StartTurn) and then
    enters its auto pre-play phase, where cards like Mayhem's play themselves. A record without the
    first, or one that ends in that phase while the emulator plays a card there, is the driver's
    snapshot (referee/driver/Combat.cs Idle), not the emulator.
    """
    if (gs.get("action") or {}).get("kind") != "end_turn" or not gs["state"].get("in_combat"):
        return None
    t = gs.get("trace", [])
    hooks = [(e[1], e[2] if len(e) > 2 else None) for e in t if e[0] == "H"]
    tail = ", ".join(fmt_ev(e) for e in t[-3:])
    if ("AfterSideTurnStart", "Player") not in hooks:
        return f"the game's turn start had not finished when it was recorded (no AfterSideTurnStart; ends with {tail})"
    after = t[max(i for i, e in enumerate(t) if e[0] == "H" and e[1] == "AfterSideTurnStart"):]
    tstart = next((i for i, e in enumerate(es.get("trace", [])) if e[0] == "M" and e[1] == "turn_start"), None)
    emu_auto = [e[2] for e in es.get("trace", [])[tstart:] if e[0] == "M" and e[1] == "play"] if tstart is not None else []
    played = any(e[0] == "M" and e[1] == "play" for e in after)
    pre = [i for i, e in enumerate(after) if e[0] == "H" and e[1] == "AfterAutoPrePlayPhaseEntered"]
    if pre and not played and len(after) > pre[0] + 1:
        return f"the game was recorded in the middle of its auto pre-play phase, a card not yet played (ends with {tail})"
    if emu_auto and not played:
        return (f"the game was recorded as its auto pre-play phase began, before any card played itself; "
                f"the emulator auto-plays {', '.join(emu_auto)} at turn start (ends with {tail})")
    return None


def text(d: dict) -> str:
    act = d.get("action") or {}
    head = f"{d['seed']} step {d['step']}: {act.get('kind', 'start')} {act.get('card') or ''}".rstrip()
    lines = [head]
    if d.get("error"):
        return "\n".join(lines + ["  " + d["error"]]) + "\n"
    lines.append("  result differs in: " + (", ".join(d.get("result_diffs", [])) or "nothing"))
    if d.get("unsettled"):
        lines.append("  GAME RECORD UNSETTLED: " + d["unsettled"])
    for sd in d.get("start_diffs", []):
        lines.append("  STARTS DIFFERENTLY (re-anchored state) " + sd)
    if d.get("first_difference"):
        lines.append("  FIRST DIFFERENCE " + d["first_difference"])
        seg = d["segment"]
        for side in ("game", "emu"):
            if seg[f"unmatched_{side}_marks"]:
                lines.append(f"  marks only the {'game' if side == 'game' else 'emulator'} has here: {', '.join(seg[f'unmatched_{side}_marks'])}")
        lines.append("  game log there: " + ("; ".join(seg["game_log"]) or "-"))
        lines.append("  game hooks there: " + ("; ".join(seg["game_hooks"]) or "-"))
        lines.append("  draws there: game " + ("; ".join(seg["game_draws"]) or "-") + " | emulator " + ("; ".join(seg["emu_draws"]) or "-"))
        for c in d.get("context", []):
            lines.append("  at step start (game): " + c)
    else:
        lines.append("  every mark agrees (the difference is in what no mark shows: draw order, the RNG streams, moves)")
    dd = d.get("draw_difference")
    if dd:
        fmt = lambda xs: "; ".join(f"#{c} by {who}" for _, c, who in xs) or "none"  # noqa: E731
        lines.append(f"  FIRST DRAW DIFFERENCE on {dd['stream']} after {dd['after_same']} equal draws "
                     f"({dd['game_total']} in the game, {dd['emu_total']} in the emulator): "
                     f"game {fmt(dd['game'])} | emulator {fmt(dd['emu'])}")
    else:
        lines.append("  draws: the same on every stream")
    for h in d.get("hidden", [])[:6]:
        lines.append("  hidden at step start: " + h)
    return "\n".join(lines) + "\n"


def first_diverging_step(game: dict, emu: dict, nm: dict):
    for j in range(1, min(len(game.get("steps", [])), len(emu.get("steps", [])))):
        if diff(game_view(game["steps"][j]["state"]), emu_view(emu["steps"][j]["state"], nm)):
            return j
    return None


def keys_of(doc: dict):
    """The power keys EmuDump reanchor writes beside its results (see reanchor_compare.keys_of)."""
    return (set(doc["buff_keys"]), set(doc["internal_keys"])) if "buff_keys" in doc else None


def run(game_path, emu_path, seed=None, step=None, id_map=None) -> list[dict]:
    nm = names(json.loads(Path(id_map or ROOT / "data" / "id_map.json").read_text(encoding="utf-8")))
    games = json.loads(Path(game_path).read_text(encoding="utf-8"))["results"]
    edoc = json.loads(Path(emu_path).read_text(encoding="utf-8"))
    emus, keys = edoc["results"], keys_of(edoc)
    out = []
    for g, e in zip(games, emus):
        if seed and g.get("seed") != seed:
            continue
        j = step if step is not None else first_diverging_step(g, e, nm)
        if j is None:
            out.append({"seed": g.get("seed"), "step": None, "error": "no step differs"})
            continue
        if j >= len(e.get("steps", [])) or j >= len(g.get("steps", [])):
            out.append({"seed": g.get("seed"), "step": j, "error": "the step is not in both records"})
            continue
        out.append(step_diff(g, e, j, nm, keys))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("referee")
    ap.add_argument("reanchor")
    ap.add_argument("--seed")
    ap.add_argument("--step", type=int)
    ap.add_argument("--id-map")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = run(a.referee, a.reanchor, a.seed, a.step, a.id_map)
    for d in res:
        sys.stdout.write(text(d))
    if a.json:
        Path(a.json).write_text(json.dumps(res), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
