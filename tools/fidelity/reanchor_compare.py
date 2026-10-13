"""Judge a re-anchored fight: every recorded step tried on its own, not only those before the first divergence.

    EmuDump reanchor <referee out.json> > reanchor.json
    python -m tools.fidelity.reanchor_compare <referee out.json> <reanchor.json> [--emu <emudump fight.json>]
        [--coverage cov.json --commit <sha>] [--json summary.json] [-v]

For step j the emulator is put into the game's state S(j-1) (tools/emudump/Reanchor.cs), the game's
action is applied, and the result is compared with the game's S(j) with the same views as
fight_compare. A step that matches verifies its card; one that differs is a divergence of that
card (or of the end of turn), whatever happened before it.

A fight stops being re-anchorable when the emulator's enemy move state can no longer be trusted. The
game's snapshot does not say what an enemy intends, so it is carried over from the emulator's own
state, and the first end-of-turn step whose result differs in anything an enemy's move touches
(the enemies, the player's HP, block or powers, the monster AI and target streams) ends the fight's
re-anchoring there. So does an enemy or card the emulator has no id for, or an enemy the game has
and the emulator lacks. Steps after the stop are reported as not re-anchored.

A diverging step is "primary" when the step before it matched, and "knock-on" otherwise: the
hidden state it started from (relic counters, card modifiers, AI state) may already be wrong.
Coverage suspects are taken from primary divergences only.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

from .fight_compare import EMU_RENAME, diff, emu_view, game_view, names

# What an enemy's move can change that the snapshot shows. The move itself (its id and intent) is not
# recorded, so a difference here is the only evidence that the emulator's enemy chose another move.
ENEMY_PATHS = (".enemies", ".rng.MonsterAi", ".rng.CombatTargets")
# Powers the monsters put on the player. A difference in any other player power at the end of a turn is
# a player-side hook (Beacon of Hope, Crimson Mantle, ...), and then HP or block differences go with it.
ENEMY_APPLIED = {"WEAK", "FRAIL", "VULNERABLE", "POISON", "SHRINK", "ENTANGLED", "TANGLED", "CONSTRICT", "HEX",
                 "NO_DRAW", "DRAW_REDUCTION", "SLOW"}


def _differing_powers(gv: dict, ev: dict) -> set:
    a, b = dict(map(tuple, gv["player"]["powers"])), dict(map(tuple, ev["player"]["powers"]))
    return {n for n in set(a) | set(b) if a.get(n) != b.get(n)}


def enemy_suspect(d: list, gv: dict, ev: dict) -> bool:
    """Whether an end-of-turn difference could come from the enemies' moves (their intent), not the player's own hooks."""
    paths = [p for p, _, _ in d]
    if any(p.startswith(ENEMY_PATHS) for p in paths):
        return True
    powers = _differing_powers(gv, ev) if ".player.powers" in paths else set()
    if powers & ENEMY_APPLIED:
        return True
    # HP and block change when an enemy hits; a missing player-side power of its own explains them too.
    return any(p in (".player.hp", ".player.block") for p in paths) and not powers


def group(path: str) -> str:
    """Field group of a diff path: the top level, and the stream or stat under player and rng."""
    parts = path.strip(".").split(".")
    return "." + ".".join(parts[:2] if parts[0] in ("player", "rng") and len(parts) > 1 else parts[:1])


def _names(view: dict) -> list:
    return [e["name"] for e in view.get("enemies", [])]


def _mask(gv: dict, ev: dict, buff_keys, internal) -> None:
    """Leave out what the two sides cannot both say about powers.

    A game power the emulator has no buff for (Crimson Mantle, Tank, Beacon of Hope, ...) lives in its
    own pieces, so such a power is dropped from the game's list and the emulator's buffs that are no
    game power are dropped from its list, for the holders that have one. Otherwise the same difference
    would show up in every step after the power appears.
    """
    if buff_keys is None:
        return
    # Keys in the names emu_view shows: a renamed buff (CRIMSON_MANTLE_BLOCK -> CRIMSON_MANTLE) is that
    # game power, so the game's power is compared, not dropped.
    buff_keys = {EMU_RENAME.get(k, k) for k in buff_keys} | {EMU_RENAME[k] for k in internal if k in EMU_RENAME}
    internal = set(internal) - set(EMU_RENAME)
    pairs = [(gv["player"], ev["player"])] + list(zip(gv["enemies"], ev["enemies"]))
    for g, e in pairs:
        unmapped = {n for n, _ in g["powers"] if n not in buff_keys}
        if unmapped:
            g["powers"] = [p for p in g["powers"] if p[0] not in unmapped]
            e["powers"] = [p for p in e["powers"] if p[0] not in internal]


def analyse(game: dict, anchored: dict, nm: dict, after_desync: str = "stop", keys=None) -> dict:
    """Per-step verdicts for one fight. Steps are indexed as in the game's record (0 is the start).

    ``after_desync``: what happens once an end of turn has differed in a way the enemies could explain.
    "stop" ends the fight's re-anchoring there; "plays" goes on with card plays (which an enemy's move
    state cannot change) and leaves later ends of turn unjudged.
    """
    steps = []
    out = {"seed": game.get("seed"), "steps": steps, "stop": None, "first_divergence": None,
           "transitions": max(0, len(game.get("steps", [])) - 1), "error": game.get("error") or anchored.get("error")}
    gsteps, rsteps = game.get("steps", []), anchored.get("steps", [])
    prev_matched = True
    desync = False
    for j, gs in enumerate(gsteps):
        if j >= len(rsteps):
            out["stop"] = out["stop"] or {"at": j, "why": "the emulator's fight had ended or could not start"}
            break
        rs = rsteps[j]
        action = gs.get("action") or {}
        kind = "start" if j == 0 else action.get("kind", "?")
        card = "END_TURN" if kind == "end_turn" else ("POTION:" + action.get("potion", "?") if kind == "potion" else action.get("card"))
        if rs.get("anchor_error"):
            out["stop"] = {"at": j, "why": "cannot rebuild the game's state: " + rs["anchor_error"]}
            break
        gv, ev = game_view(gs["state"]), emu_view(rs["state"], nm)
        if out["first_divergence"] is None and diff(gv, ev):  # the strict test the plain replay uses
            out["first_divergence"] = j
        if gv.get("in_combat") and ev.get("in_combat"):
            _mask(gv, ev, *(keys or (None, None)))
        d = diff(gv, ev)
        problems = [c for c in rs.get("choices", []) if c.get("error") or c.get("unexpected_selection")]
        verdict = "error" if rs.get("error") else ("match" if not d and not problems else "diverge")
        if desync and kind == "end_turn":
            verdict = "unjudged"  # the enemies' move state is no longer the game's
        step = {"j": j, "kind": kind, "card": card, "verdict": verdict, "paths": [p for p, _, _ in d],
                "primary": verdict != "match" and prev_matched, "patched": rs.get("patched", []),
                "error": rs.get("error"), "choice_problems": problems}
        steps.append(step)
        prev_matched = verdict in ("match", "unjudged")
        if kind == "start":
            # The start state is the emulator's own; only an enemy it cannot have is fatal. Hit points
            # that differ (a different HP roll) are patched and the fight carries on.
            if _names(gv) != _names(ev) or any(p == ".rng.MonsterAi" for p, _, _ in d):
                out["stop"] = {"at": 1, "why": "the fight starts with other enemies or another monster AI position"}
                break
        elif kind == "end_turn" and not desync and d and enemy_suspect(d, gv, ev):
            if after_desync == "stop":
                out["stop"] = {"at": j + 1, "why": f"enemy move/intent not reproducible after the end of turn at step {j}"}
                break
            desync = True
            out["desync"] = j
    return out


def summarise(analyses: list) -> dict:
    tot = collections.Counter()
    verified_cards, diverged = collections.Counter(), collections.Counter()
    stops = collections.Counter()
    for a in analyses:
        if a["error"] and not a["steps"]:
            tot["fights_unusable"] += 1
            continue
        tot["fights"] += 1
        tot["transitions"] += a["transitions"]
        tried = [s for s in a["steps"] if s["j"] > 0 and s["verdict"] != "unjudged"]
        tot["unjudged_end_turns"] += len([s for s in a["steps"] if s["verdict"] == "unjudged"])
        tot["reanchored"] += len(tried)
        # First-divergence method: steps before the first difference (the start counts as step 0).
        before = a["first_divergence"] if a["first_divergence"] is not None else len(a["steps"])
        tot["verified_before"] += max(0, before - 1)
        for s in tried:
            if s["verdict"] == "match":
                tot["verified"] += 1
                if s["card"] != "END_TURN":
                    verified_cards[s["card"]] += 1
            else:
                tot["diverging"] += 1
                tot["diverging_primary" if s["primary"] else "diverging_knock_on"] += 1
                for g in sorted({group(p) for p in s["paths"]} or {"(error)"}):
                    diverged[s["card"], g, s["primary"]] += 1
        if a["stop"]:
            stops[a["stop"]["why"].split(" at step")[0]] += 1
            tot["fights_stopped"] += 1
        elif "desync" in a:
            tot["fights_desynced"] += 1
        tot["not_reanchored"] += a["transitions"] - len(tried)
        if a["first_divergence"] is None:
            tot["fights_identical"] += 1
    return {"totals": dict(tot), "stops": dict(stops),
            "verified_cards": dict(verified_cards),
            "diverged": [{"card": c, "field": f, "primary": p, "steps": n} for (c, f, p), n in diverged.most_common()]}


def pair(games: list, res: list) -> list:
    if len(games) == len(res):
        return list(zip(games, res))
    by_seed = {r.get("seed"): r for r in res}
    return [(g, by_seed[g.get("seed")]) for g in games if g.get("seed") in by_seed]


def keys_of(doc: dict):
    """The power keys EmuDump reanchor wrote beside the results, or None for older output."""
    return (set(doc["buff_keys"]), set(doc["internal_keys"])) if "buff_keys" in doc else None


def learn(cov: dict, games: list, doc: dict, nm: dict, commit: str, after_desync: str = "stop") -> dict:
    """Fold a re-anchored batch into coverage (replaces ``coverage.update`` on the plain replay)."""
    from . import coverage  # noqa: PLC0415

    keys = keys_of(doc)
    return coverage.update_reanchored(cov, [analyse(g, r, nm, after_desync, keys) for g, r in pair(games, doc["results"])], commit)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("referee")
    ap.add_argument("reanchor")
    ap.add_argument("--id-map", default="data/id_map.json")
    ap.add_argument("--after-enemy-desync", choices=("stop", "plays"), default="stop",
                    help="stop re-anchoring a fight where the enemies' move state stops being reproducible (default), "
                         "or go on with card plays only and leave later ends of turn unjudged")
    ap.add_argument("--emu", help="the plain EmuDump fight output; checks that both methods agree on the first divergence")
    ap.add_argument("--coverage", help="coverage JSON to update (verified plays, primary divergences as suspects)")
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--json", help="write the per-fight verdicts and the summary here")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every diverging step")
    a = ap.parse_args()
    nm = names(json.load(pathlib.Path(a.id_map).open(encoding="utf-8")))
    games = json.load(pathlib.Path(a.referee).open(encoding="utf-8"))["results"]
    doc = json.load(pathlib.Path(a.reanchor).open(encoding="utf-8"))
    pairs = pair(games, doc["results"])
    keys = keys_of(doc)
    analyses = [analyse(g, r, nm, a.after_enemy_desync, keys) for g, r in pairs]
    for (g, _), an in zip(pairs, analyses):
        if an["error"] and not an["steps"]:
            print(f"{an['seed']}: not usable: {str(an['error'])[:120]}")
            continue
        bad = [s for s in an["steps"] if s["verdict"] not in ("match", "unjudged")]
        done = sum(1 for s in an["steps"] if s["j"] > 0 and s["verdict"] != "unjudged")
        line = f"{an['seed']}: {done}/{an['transitions']} steps re-anchored, {done - len([s for s in bad if s['j'] > 0])} verified"
        if an["stop"]:
            line += f"; stopped: {an['stop']['why']}"
        if bad or an["stop"] or "desync" in an or a.verbose:
            print(line)
        for s in bad if a.verbose else []:
            note = f" error {s['error'][:80]}" if s["error"] else ""
            print(f"    step {s['j']} {s['card']} {'primary' if s['primary'] else 'knock-on'}: {', '.join(s['paths'][:6])}{note}")
    sm = summarise(analyses)
    t = sm["totals"]
    print()
    print(f"fights {t.get('fights', 0)} ({t.get('fights_identical', 0)} identical end to end, {t.get('fights_stopped', 0)} stopped re-anchoring, "
          f"{t.get('fights_desynced', 0)} with unjudged ends of turn, {t.get('fights_unusable', 0)} without a game result)")
    print(f"steps in those fights {t.get('transitions', 0)}: re-anchored {t.get('reanchored', 0)}, not re-anchored {t.get('not_reanchored', 0)}")
    print(f"verified {t.get('verified', 0)} (first-divergence method: {t.get('verified_before', 0)}), "
          f"diverging {t.get('diverging', 0)} ({t.get('diverging_primary', 0)} primary, {t.get('diverging_knock_on', 0)} knock-on)")
    for why, n in sm["stops"].items():
        print(f"  stopped, {why}: {n} fights")
    print("diverging by card and field (primary only):")
    for row in [r for r in sm["diverged"] if r["primary"]][:25]:
        print(f"  {row['steps']:4d}  {row['card']:<28} {row['field']}")
    if a.emu:
        emus = json.load(pathlib.Path(a.emu).open(encoding="utf-8"))["results"]
        from .coverage import first_divergence
        bad = 0
        for (g, _), an in zip(pairs, analyses):
            e = next((x for x in emus if x.get("seed") == g.get("seed")), None)
            if e is None or (an["error"] and not an["steps"]):
                continue
            old = first_divergence(g, e, nm)
            if old != an["first_divergence"]:
                bad += 1
                print(f"{g['seed']}: first divergence {old} by the plain replay, {an['first_divergence']} re-anchored")
        print(f"first divergence agrees with the plain replay in all but {bad} fights")
    if a.coverage:
        from pathlib import Path

        from . import coverage
        path = Path(a.coverage)
        cov = coverage.load(path)
        print("coverage:", coverage.update_reanchored(cov, analyses, a.commit))
        coverage.save(path, cov)
    if a.json:
        json.dump({"summary": sm, "fights": analyses}, Path(a.json).open("w", encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
