"""The first game action whose full combat state the emulator (or another game run) does not reproduce.

The referee's ChecksumTap (referee/driver/ChecksumTap.cs) records the game's own desync checksums
after every game action and turn phase; with "trace": true also the full state behind each one
(NetFullCombatState). `EmuDump fight <out.json> --netstate` writes the emulator's state after each
step in the same shape (tools/emudump/NetState.cs).

    python -m tools.fidelity.netstate_diff <game trace out.json> <emudump --netstate json> [--seed S] [-v]
    python -m tools.fidelity.netstate_diff <game out.json> <game out.json>   game against game
    python -m tools.fidelity.netstate_diff --check <game out.json>           driver integrity only

Game against emulator: per fight, the first step whose state differs on a field both sides model,
the differing fields, and the game sub-action (checksum context: a card play, or one of the end of
turn's phases) after which the game's value is one the emulator no longer reproduces. Fields only
the game has (choice/reward ids, phase, card floors, saved props, seeds) are listed once as not
modelled. Power names go through fight_compare's rules.

Game against game: the first checksum that differs between two runs of the same specs, and which
fields differ there when both runs were traced.

--check (also done first in the other modes): a step whose action left no checksum of its own
(a play without its PlayCardAction, an end of turn without "After player turn start" while the
fight goes on) was recorded before the game finished it; such a fight is a driver artifact from
that step on, not an emulator divergence. A fight recorded without the tap (no step has checksums)
cannot be checked and FAILS the check: it is never reported OK.

Exit status: 0 only when every fight was checked (or compared) and nothing differs; 1 otherwise,
including a fight the other side has no result for, a driver artifact, a fight with no trace and an
emulator replay that stops before the game's last step or reports an error.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from tools.fidelity import fight_compare as fc

PILES = ("Hand", "Draw", "Discard", "Exhaust", "Play")
END_OF_TURN = "After player turn start"


def _card_core(c: dict) -> list:
    return [c.get("id"), (c.get("upgrades") or 0) > 0]


def _card_detail(c: dict, with_cost: bool = True) -> dict:
    d = {}
    if c.get("enchantment"):
        d["enchantment"] = [c["enchantment"].get("id"), c["enchantment"].get("amount")]
    if with_cost and "cost" in c:
        d["cost"] = c["cost"]
    if c.get("keywords"):
        d["keywords"] = sorted(c["keywords"])
    if c.get("affliction"):
        d["affliction"] = c["affliction"][0]
    return d


def _powers(raw: list, side: str) -> list:
    if side == "game":
        return fc.game_powers(raw)
    return fc.emu_powers(raw)


def flat(state: dict | None, side: str) -> dict:
    """A state as path -> comparable value, in the comparer's (fight_compare's) terms."""
    if state is None:
        return {"in_combat": False}
    out: dict = {"in_combat": bool(state.get("creatures"))}
    enemies = [c for c in state.get("creatures", []) if c.get("monster")]
    for c in state.get("creatures", []):
        if c.get("player") is not None:
            for k in ("hp", "max_hp", "block"):
                out[f"player.{k}"] = c[k]
            out["player.powers"] = _powers(c["powers"], side)
    for i, c in enumerate(enemies):
        out[f"enemies[{i}]"] = [c["monster"], c["hp"], c["max_hp"], c["block"]]
        out[f"enemies[{i}].powers"] = _powers(c["powers"], side)
    out["enemies.count"] = len(enemies)
    for k, v in (state.get("rng", {}).get("counters") or {}).items():
        out[f"rng.{k}"] = v
    if "seed" in state.get("rng", {}):
        out["rng.seed"] = state["rng"]["seed"]
    for k in ("choice_ids", "reward_ids", "last_action", "last_hook"):
        if k in state:
            out[k] = state[k]
    for p in state.get("players", [])[:1]:
        for k in ("turn", "phase", "energy", "stars", "max_potions", "gold", "potions", "orbs", "character"):
            if k in p:
                out[k] = p[k]
        out["relics"] = [r.get("id") for r in p.get("relics", [])]
        if side == "game":
            out["relics.props"] = [r.get("props") for r in p.get("relics", [])]
            out["relics.floor"] = [r.get("floor") for r in p.get("relics", [])]
        else:
            out["relics.counter"] = [r.get("counter") for r in p.get("relics", [])]
        piles = p.get("piles", {})
        if piles:
            for name in PILES:
                cards = piles.get(name, [])
                out[f"piles.{name}"] = [_card_core(c) for c in cards]
                # Cost only in hand: NetState.cs prices the hand (pile-dependent cost hooks).
                out[f"piles.{name}.detail"] = [_card_detail(c, name == "Hand") for c in cards]
                if side == "game":
                    out[f"piles.{name}.props"] = [c.get("props") for c in cards]
                    out[f"piles.{name}.floor"] = [c.get("floor") for c in cards]
        for k, v in (p.get("rng", {}).get("counters") or {}).items():
            out[f"player_rng.{k}"] = v
        if "seed" in p.get("rng", {}):
            out["player_rng.seed"] = p["rng"]["seed"]
        for k, v in (p.get("grab_bag") or {}).items():
            out[f"grab_bag.{k}"] = v
    return out


RNG_NAMES = {"CombatOrbGeneration": "CombatOrbs"}  # the snapshot names RunRngSet properties


def snapshot_flat(st: dict) -> dict:
    """The driver's own idle snapshot of a step (Combat.cs Snapshot) under flat()'s paths."""
    if not st.get("in_combat"):
        return {}
    p = st["player"]
    out = {"player.hp": p["hp"], "player.max_hp": p["max_hp"], "player.block": p["block"],
           "player.powers": fc.game_powers(p["powers"]), "energy": p["energy"]}
    for name, key in (("Hand", "hand"), ("Draw", "draw"), ("Discard", "discard"), ("Exhaust", "exhaust")):
        out[f"piles.{name}"] = [[c[0], bool(c[1])] for c in st[key]]
    alive = [e for e in st["enemies"] if e["hp"] > 0]
    out["enemies.count"] = len(alive)
    for i, e in enumerate(alive):
        out[f"enemies[{i}]"] = [e["name"], e["hp"], e["max_hp"], e["block"]]
        out[f"enemies[{i}].powers"] = fc.game_powers(e["powers"])
    for k, v in (st.get("rng") or {}).items():
        out[f"rng.{RNG_NAMES.get(k, k)}"] = v["counter"]
    return out


def settle(last: dict, snapshot: dict) -> tuple[dict, list]:
    """The step's end state: its last checkpoint, with the fields the driver's later snapshot shows
    otherwise (work the game did after its last checksum, such as the auto-play phase after
    "After player turn start"). Returns (state, the fields that moved).
    """
    moved = [k for k in snapshot if k in last and last[k] != snapshot[k]]
    return ({**last, **{k: snapshot[k] for k in moved}}, moved)


def compare(g: dict, e: dict) -> list:
    """(path, game, emulator) for every path both sides have and that differs."""
    return [(k, g[k], e[k]) for k in sorted(set(g) & set(e)) if g[k] != e[k]]


# ── driver integrity ───────────────────────────────────────────────────────────────────────

def tapped(fight: dict) -> bool:
    """Whether the fight was recorded with the checksum tap (some step carries checksums)."""
    return any("checksums" in s for s in fight.get("steps", []))


def late_steps(fight: dict) -> list:
    """Steps recorded before the game finished their action (see the module doc).

    Empty for a fight recorded without the tap, which says nothing either way (``tapped``); in a tapped
    fight a step without checksums is a step that left none.
    """
    out = []
    if not tapped(fight):
        return out
    for k, s in enumerate(fight.get("steps", [])[1:], 1):
        a, kinds = s["action"] or {}, [c[1] or c[0] for c in s.get("checksums", [])]
        if (a.get("kind") == "play" and a.get("played") is not False and "PlayCardAction" not in kinds) or (a.get("kind") == "end_turn" and s["state"].get("in_combat") and END_OF_TURN not in kinds):
            out.append(k)
    return out


def _act(a: dict | None) -> str:
    if not a:
        return "start"
    return a.get("card") or a.get("kind", "?")


# ── game against emulator ──────────────────────────────────────────────────────────────────

def checkpoints(step: dict) -> list:
    """(context, flat game state) after each checksum the game generated during the step."""
    return [(n["ctx"], flat(n["state"], "game")) for n in step.get("net", [])]


def blame(path: str, start: dict, points: list, emu_value) -> tuple[int | None, str]:
    """The game sub-action (index into points) a field's divergence is first visible at, judged from
    the game's values at each checkpoint of the step (start = the previous step's last state):
    - the emulator shows a value the game had earlier in the step: the game's next change of it,
      which the emulator did not make;
    - the emulator shows a value the game never had: the game's last change of it (where the
      game's final value was made), when there is one;
    - the game never changes it in the step: not localisable from the game side (None).
    """
    seq = [start.get(path)] + [st.get(path) for _, st in points]
    changes = [i for i in range(1, len(seq)) if seq[i] != seq[i - 1]]
    matched = [i for i, v in enumerate(seq) if v == emu_value]
    if matched:
        i = matched[-1]
        later = [c for c in changes if c > i]
        where = "the step's start" if i == 0 else points[i - 1][0][:40]
        return (later[0] - 1 if later else None), f"emulator stops at the game's value from {where}"
    if changes:
        return changes[-1] - 1, "game's final value made here; the emulator never had it"
    return None, "game leaves it unchanged in this step; the emulator changed it"


def diff_fight(g: dict, e: dict, verbose: bool, ignore_late: bool = False) -> dict:
    seed = g.get("seed")
    res = {"seed": seed, "late": late_steps(g)}
    gsteps, esteps = g.get("steps", []), e.get("steps", [])
    prev = None
    not_modelled: set = set()
    for k, (gs, es) in enumerate(zip(gsteps, esteps)):
        if res["late"] and k >= res["late"][0] - 1 and not ignore_late:
            first = res["late"][0]
            # The game was not ready for the late step's action, so the step before it (an end of
            # turn whose turn-start work, such as an auto-play, was still running) is suspect too.
            suspect = first - 1 if (gsteps[first - 1]["action"] or {}).get("kind") == "end_turn" else first
            if k >= suspect:
                res["verdict"] = (f"driver: step {first} ({_act(gsteps[first]['action'])}) recorded before the game "
                                  f"finished it; steps from {suspect} on are not comparable")
                return res
        points = checkpoints(gs)
        if not points and ignore_late and prev is not None and k > 0:
            points = [("(no checksum in this step)", prev)]
        if not points or "net" not in es:
            res["verdict"] = f"step {k}: no trace (game {len(points)} checkpoints, emulator {'net' in es})"
            return res
        if not gs["state"].get("in_combat"):
            # The fight is over; its last checksum was taken while the killing action finished.
            gfin, moved = {"in_combat": False}, []
        else:
            gfin, moved = settle(points[-1][1], snapshot_flat(gs["state"]))
        if moved:
            points = points + [("(after the last checksum: driver snapshot)", gfin)]
            res.setdefault("moved", []).append((k, moved))
        efin = flat(es["net"][-1]["state"], "emu")
        if gfin.get("in_combat") and efin.get("in_combat"):
            not_modelled |= {p for p in set(gfin) - set(efin) if not p.startswith("enemies")}
        d = compare(gfin, efin)
        if d:
            start = prev if prev is not None else {}
            blamed = []
            for path, gv, ev in d:
                i, why = blame(path, start, points, ev)
                blamed.append((i, path, gv, ev, why))
            known = [b[0] for b in blamed if b[0] is not None]
            at = points[min(known)][0] if known else "not localisable from the game side"
            res.update(step=k, action=gs["action"], subaction=at, n_checkpoints=len(points),
                       fields=[(p, gv, ev, why) for i, p, gv, ev, why in
                               sorted(blamed, key=lambda b: (b[0] is None, b[0] or 0, b[1]))])
            if verbose:
                res["timeline"] = {p: [("start", start.get(p))] + [(c, st.get(p)) for c, st in points]
                                   for _, p, _, _, _ in blamed}
            res["not_modelled"] = sorted(not_modelled)
            return res
        prev = gfin
    n_g, n_e = len(gsteps), len(esteps)
    if n_e != n_g or e.get("error"):
        # Identical as far as both go is not identical: a replay that stopped early (or went on, or threw
        # after its last step) did not reproduce the fight.
        res["verdict"] = (f"{min(n_g, n_e)} steps identical, then the emulator's replay "
                          + (f"stops after {n_e} of the game's {n_g} steps" if n_e < n_g else
                             f"has {n_e} steps to the game's {n_g}" if n_e > n_g else "reports an error")
                          + (f": {_short(e['error'])}" if e.get("error") else ""))
    else:
        res["verdict"] = f"{n_g} steps identical on every modelled field"
    res["not_modelled"] = sorted(not_modelled)
    return res


# ── game against game ──────────────────────────────────────────────────────────────────────

def _sums(fight: dict) -> list:
    out = []
    for k, s in enumerate(fight.get("steps", [])):
        nets = s.get("net", [])
        for j, c in enumerate(s.get("checksums", [])):
            out.append((k, c, nets[j]["state"] if j < len(nets) else None))
    return out


def diff_games(a: dict, b: dict) -> dict:
    sa, sb = _sums(a), _sums(b)
    res = {"seed": a.get("seed"), "late": (late_steps(a), late_steps(b))}
    moved = next(((i, x[0], y[0]) for i, (x, y) in enumerate(zip(sa, sb)) if x[1] == y[1] and x[0] != y[0]), None)
    for i, (x, y) in enumerate(zip(sa, sb)):
        if x[1] != y[1]:
            res.update(index=i, steps=(x[0], y[0]), checksums=(x[1], y[1]))
            if x[2] is not None and y[2] is not None:
                fa, fb = flat(x[2], "game"), flat(y[2], "game")
                res["fields"] = [(p, fa.get(p), fb.get(p)) for p in sorted(set(fa) | set(fb)) if fa.get(p) != fb.get(p)]
            break
    else:
        res["equal_prefix"] = min(len(sa), len(sb))
        res["lengths"] = (len(sa), len(sb))
    if moved:
        res["first_step_shift"] = {"checksum": moved[0], "steps": moved[1:]}
    return res


def _short(v, n=160) -> str:
    s = json.dumps(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("other", nargs="?")
    ap.add_argument("--check", action="store_true", help="driver integrity only")
    ap.add_argument("--seed", action="append", default=[])
    ap.add_argument("--ignore-late", action="store_true", help="compare past steps the driver recorded too early")
    ap.add_argument("-v", action="store_true", help="each differing field's value after every game checkpoint")
    a = ap.parse_args()
    games = [r for r in json.loads(pathlib.Path(a.game).read_text(encoding="utf-8"))["results"]
             if not a.seed or r.get("seed") in a.seed]
    if a.check or not a.other:
        bad = unchecked = 0
        for g in games:
            late = late_steps(g)
            bad += bool(late)
            what = "OK"
            if not tapped(g):
                unchecked += 1
                what = "UNCHECKED: recorded without the checksum tap"
            elif late:
                what = f"{len(late)} late steps, first {late[0]} ({_act(g['steps'][late[0]]['action'])})"
            print(f"{g.get('seed')}: {what}")
        print(f"{bad} of {len(games)} fights have steps recorded before the game finished them; "
              f"{unchecked} could not be checked (no checksums)")
        return 1 if bad or unchecked or not games else 0
    others = {r.get("seed"): r for r in json.loads(pathlib.Path(a.other).read_text(encoding="utf-8"))["results"]}
    vs_game = any("checksums" in s for r in others.values() for s in r.get("steps", []))
    nm_all: set = set()
    bad = compared = 0
    for g in games:
        o = others.get(g.get("seed"))
        if o is None:
            print(f"{g.get('seed')}: no result on the other side")
            bad += 1
            continue
        compared += 1
        if vs_game:
            r = diff_games(g, o)
            if "index" in r or r.get("lengths", (0, 0))[0] != r.get("lengths", (0, 0))[1]:
                bad += 1
            if "index" in r:
                print(f"{r['seed']}: checksum #{r['index']} differs (steps {r['steps']}): {r['checksums'][0]} vs {r['checksums'][1]}")
                for p, x, y in r.get("fields", [])[:10]:
                    print(f"    {p}: {_short(x, 70)}  vs  {_short(y, 70)}")
            else:
                print(f"{r['seed']}: {r['equal_prefix']} checksums equal (lengths {r['lengths']})")
            if "first_step_shift" in r:
                sh = r["first_step_shift"]
                print(f"    same checksum #{sh['checksum']} recorded in step {sh['steps'][0]} vs {sh['steps'][1]} (driver step boundary)")
            continue
        r = diff_fight(g, o, a.v, a.ignore_late)
        nm_all |= set(r.get("not_modelled", []))
        # Identical only when every step was compared: a divergence, a driver artifact and a missing
        # trace all fail.
        bad += "step" in r or not r.get("verdict", "").endswith("identical on every modelled field")
        for k, mv in r.get("moved", [])[:3]:
            print(f"{r['seed']}: note: step {k} changed after its last checksum: {', '.join(mv[:6])}")
        if "step" in r:
            print(f"{r['seed']}: step {r['step']} ({_act(r['action'])}), {r['n_checkpoints']} game checkpoints; "
                  f"first divergent sub-action: {r['subaction']}")
            for p, gv, ev, why in r["fields"][:12]:
                print(f"    {p}: game {_short(gv, 70)}  emulator {_short(ev, 70)}  [{why}]")
            for p, tl in (r.get("timeline") or {}).items():
                print(f"      {p}: " + " -> ".join(f"{c[:28]}={_short(v, 40)}" for c, v in tl))
        else:
            print(f"{r['seed']}: {r['verdict']}")
    if nm_all:
        print("not modelled by the emulator (game only):", ", ".join(sorted({p.split("[")[0] for p in nm_all})))
    print(f"{compared} fights compared, {bad} not identical or not comparable")
    return 1 if bad or not compared else 0


if __name__ == "__main__":
    sys.exit(main())
