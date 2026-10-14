"""Compare whole runs: the game's run snapshots (referee) with the emulator's, decision by decision.

    python -m tools.fidelity.run_compare2 start <referee out.json> <EmuDump runstart json>
    python -m tools.fidelity.run_compare2 run <referee out.json> <EmuDump run json> [-v]

`start`: the run-start save of each run (any referee result that carries "save") against
`EmuDump runstart`: all three acts' room sets, both relic grab bags, the odds, every RNG counter.
`run`: each recorded run decision (referee/driver/RunPlay.cs) against the emulator's replay of the same
choices (tools/emudump/RunReplay.cs). Per run it prints the first decision that differs, fights
included: a fight step that differs, a fight that ends at another step, or a fight identical at
every step whose outcome (the run snapshot right after it, before any anchoring, and the reward
screen it leaves) differs from the game's. The replay then anchors each fight to the game's outcome,
so it also prints the first divergence outside fights. A run counts as identical only when nothing
compared differs, and what is not compared is printed with every comparison
(run_snapshot.NOT_COMPARED, run_snapshot.not_compared). Exit 0 when everything compared matches.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

from tools.fidelity import run_snapshot as rs


def start(ref_path: str, emu_path: str) -> int:
    refs = [r for r in json.load(open(ref_path, encoding="utf-8"))["results"] if "save" in r]
    emus = {e["seed"]: e for e in json.load(open(emu_path, encoding="utf-8"))["results"]}
    bad = 0
    unchecked: collections.Counter = collections.Counter()
    for r in refs:
        g = rs.from_save(r["save"])
        e = emus.get(r["seed"])
        if e is None:
            print(f"{r['seed']}: no emulator result")
            bad += 1
            continue
        unchecked.update(set(rs.not_compared(g, e["snap"])))
        diffs = rs.compare(g, e["snap"])
        if diffs:
            bad += 1
            print(f"{r['seed']}: {len(diffs)} differences")
            for path, gv, ev in diffs[:8]:
                print(f"  {path}: game {rs.short(gv)} emulator {rs.short(ev)}")
    print(f"{len(refs)} run starts compared, {bad} with differences")
    coverage(unchecked, len(refs), "run starts")
    return 0 if not bad and refs else 1


def coverage(unchecked: collections.Counter, n: int, what: str) -> None:
    """What the comparison checks and what it does not: an unchecked field is never taken as equal."""
    print("compared: run snapshot " + ", ".join(rs.ORDER) + "; each screen's offer, the room entered (type and id), "
          "event options enabled, new acts' maps, every fight step and each fight's outcome")
    print("not compared: " + "; ".join(f"{k} ({v})" for k, v in rs.NOT_COMPARED.items()))
    if unchecked:
        print(f"not compared, present in the game's saves (in how many of {n} {what}): "
              + ", ".join(f"{k} {v}" for k, v in sorted(unchecked.items())))


def norm_offer(screen: str, offer, emulator: bool):
    """An offer as a comparable value: what the screen lets the player take, in a fixed order."""
    if offer is None:
        return None
    if screen in ("neow", "treasure"):
        return sorted(rs.bare(x[0] if isinstance(x, list) else x) for x in offer)
    if screen == "map":
        return sorted((x[0], x[1], x[2]) for x in offer)
    if screen == "rewards":
        # A SpecialCardReward (a Thieving Hopper's stolen card, the Lantern Key) is a row on both
        # screens, compared like the others.
        return sorted(json.dumps(norm_reward(r), sort_keys=True) for r in offer)
    if screen == "rest":
        return sorted(x[0] for x in offer if x[1])
    if screen == "fake_merchant":
        return [[rs.bare(x[0]), x[1]] if x else None for x in offer]
    if screen == "shop" and isinstance(offer, dict):
        return {k: ([[rs.bare(x[0])] + list(x[1:]) if x else None for x in v] if isinstance(v, list) else v)
                for k, v in offer.items()}
    return offer


def norm_reward(r: dict) -> dict:
    r = dict(r)
    if "id" in r:
        r["id"] = rs.bare(r["id"])
    if "cards" in r:
        r["cards"] = [[rs.bare(c[0]), int(c[1])] for c in r["cards"]]
    if isinstance(r.get("card"), list):
        r["card"] = [rs.bare(r["card"][0]), int(r["card"][1])]
    return r


def room_of(screen: str, game: dict, emu: dict) -> list:
    """The room a map move entered: its type (the game's RoomType) and its encounter or event id,
    an encounter compared as the emulator's entry (table_check.norm_enc: a weak fight and its normal
    twin are two). Rest sites, shops and chests have no id, so the type is what tells them apart."""
    out = []
    groom, eroom = game.get("room"), emu.get("room")
    if screen != "map" or not groom:
        return out
    if not eroom:
        return [] if emu.get("error") else [("room", groom, None)]
    if len(eroom) < 3 or groom[0] != eroom[2]:
        out.append(("room type", groom[0], eroom[2] if len(eroom) >= 3 else eroom[0]))
    gid, eid = groom[1], eroom[1]
    if groom[0] in ("Monster", "Elite", "Boss"):
        gid, eid = rs.enc(gid), rs.enc(eid, emulator=True)
    else:
        gid, eid = rs.bare(gid), rs.bare(eid)
    if (gid or eid) and gid != eid:
        out.append(("room", groom, eroom))
    if game.get("treasure_relic") or emu.get("treasure_relic"):
        if rs.bare(game.get("treasure_relic")) != rs.bare(emu.get("treasure_relic")):
            out.append(("treasure_relic", game.get("treasure_relic"), emu.get("treasure_relic")))
    return out


NOT_COMPARED_FIGHT = "fight steps (no --id-map)"


def fight_diff(game: dict, emu: dict, nm: dict | None):
    """The first fight step whose state differs (fight_compare's views), or None. A replay that ends
    before the game's fight (or goes on after it) differs at the first step one side lacks, and a game
    fight the replay has no record of differs at step 0. Without an id map the steps cannot be compared:
    NOT_COMPARED_FIGHT, which the caller counts as not compared, never as identical."""
    if "fight" not in game:
        return None
    if "fight" not in emu:
        return 0, None, [("steps", len(game["fight"].get("steps", [])), "no fight in the replay")]
    if nm is None:
        return NOT_COMPARED_FIGHT
    from tools.fidelity.fight_compare import diff, emu_view, game_view
    gsteps, esteps = game["fight"]["steps"], emu["fight"]["steps"]
    for i, (gs, es) in enumerate(zip(gsteps, esteps)):
        d = diff(game_view(gs["state"]), emu_view(es["state"], nm))
        if d:
            return i, gs["action"], d
    if len(gsteps) != len(esteps):
        i = min(len(gsteps), len(esteps))
        act = gsteps[i]["action"] if i < len(gsteps) else None
        why = emu.get("emulator_fight_ended")
        return i, act, [("steps", len(gsteps), f"{len(esteps)}" + (f" (replay: {why})" if why else ""))]
    return None


def fight_outcome(d: dict, ed: dict, nd: dict | None) -> list:
    """A fight identical at every step can still end elsewhere: the emulator's own state right after
    it (RunReplay's outcome_snap, before anchoring) against the game's post-fight save, and the
    reward screen it leaves against the game's next rewards offer."""
    result = d.get("result")
    end = ed.get("fight_end")
    if result not in ("win", "death"):
        # The game's run stopped inside the fight (the driver's max_actions): it has no outcome
        # to compare, and run() reports the fight as not compared.
        return []
    if result == "death":
        return [] if "Complete" in (end, ed.get("phase")) else [("fight end", "death", end or ed.get("phase"))]
    if end is None:
        return [("fight end", result, "not recorded by the replay")]
    if end == "Combat":
        return [("fight end", result, "emulator still fighting")]
    if end == "Complete":
        return [("fight end", result, "emulator died")]
    out = []
    if d.get("snap") is not None:
        out += rs.compare(rs.from_save(d["snap"]), ed.get("outcome_snap") or {})
    if nd is not None and nd.get("screen") == "rewards" and nd.get("offer") is not None:
        go = norm_offer("rewards", nd["offer"], False)
        eo = norm_offer("rewards", ed.get("outcome_offer"), True)
        if nd.get("depth"):
            if eo is None or any(x not in eo for x in go):
                out.append(("rewards offer", go, eo))
        elif go != eo:
            out.append(("rewards offer", go, eo))
    return out


def compare_map(game_map: list, emu_map: list) -> list:
    """A new act's map: every point's type and children (run_compare's act-1 comparison)."""
    from tools.fidelity.run_compare import map_set
    gm, em = map_set(game_map), map_set(emu_map)
    return [(f"map {c}", gm.get(c), em.get(c)) for c in sorted(set(gm) | set(em), key=lambda c: (c[1], c[0]))
            if gm.get(c) != em.get(c)]


def event_options(d: dict, ed: dict) -> tuple[list, list]:
    """(differences, not compared): the game's enabled options against the emulator's action mask
    at every index (an index past the game's options is the emulator offering one the game lacks),
    and the first page's option count. The emulator's skip action is not an event option; where the
    game has an option at that index the two cannot be told apart, and that index is not compared."""
    offer = d["offer"]
    n = len(offer)
    skip = ed.get("skip_action", 3)
    g_en = [i for i, o in enumerate(offer) if not o[1]]
    unchecked = []
    if n > skip:
        unchecked.append(f"event option index {skip} (the emulator's skip action)")
        g_en = [i for i in g_en if i != skip]
    e_en = [i for i in ed["enabled"] if i != skip]
    diffs = []
    if g_en != e_en:
        diffs.append(("event options enabled", g_en, e_en))
    count = ed.get("option_count")
    first = offer[0][0] if offer else ""
    if count and ".pages.INITIAL." in first and count != n:
        diffs.append(("event option count", n, count))
    return diffs, unchecked


def compare_decision(d: dict, ed: dict, nm, game_maps: list | None = None) -> list:
    """Everything a decision shows outside a fight's steps and outcome: replay errors, the offer, the
    room entered, event options, a new act's map and the run snapshot after the decision (after a
    fight that is after anchoring, so a difference there is something the anchoring did not reset)."""
    if d.get("screen") == "combat" and d.get("result") == "death":
        # The run ends here; the fight's end is its outcome (fight_outcome).
        return []
    diffs = []
    if ed.get("error"):
        diffs.append(("replay", d.get("choice"), ed["error"]))
    screen = d.get("screen", "")
    go, eo = norm_offer(screen, d.get("offer"), False), norm_offer(screen, ed.get("offer"), True)
    if screen == "event":
        # RunReplay.Apply exports an event's options as its action mask ("enabled", "option_count"), never
        # as an offer (every other screen with an offer gets "offer"): event_options compares them, and a
        # mask the replay did not export is a difference.
        if d.get("offer") is not None and ed.get("enabled") is not None:
            diffs += event_options(d, ed)[0]
        elif d.get("offer") is not None and not ed.get("error"):
            diffs.append(("event options enabled", [i for i, o in enumerate(d["offer"]) if not o[1]], None))
    elif go is not None and eo is None:
        # An offer the game showed and the replay did not export is a difference, not a pass (a replay
        # that failed there has its error as the difference already).
        if not ed.get("error"):
            diffs.append(("offer", go, None))
    elif screen == "rewards" and d.get("depth") and go is not None:
        # A screen a reward opened on top of another (Lost Coffer off Neow's Bones): the emulator
        # keeps one screen and adds the inner rewards to it, so the inner screen's offer is a
        # part of the emulator's.
        if any(x not in eo for x in go):
            diffs.append(("offer", go, eo))
    elif go is not None and go != eo:
        diffs.append(("offer", go, eo))
    diffs += room_of(screen, d, ed)
    if screen == "act_change" and ed.get("map") is not None and game_maps:
        act = (d.get("snap") or {}).get("current_act_index")
        if act is not None and act < len(game_maps):
            diffs += compare_map(game_maps[act], ed["map"])[:6]
    if not diffs and d.get("snap") is not None and ed.get("snap") is None and not ed.get("error"):
        diffs = [("run snapshot", "recorded", None)]
    elif not diffs and d.get("snap") is not None and ed.get("snap") is not None:
        diffs = rs.compare(rs.from_save(d["snap"]), ed["snap"])
        if screen == "act_change":
            # Representation, not rules: on a new act the emulator stands on the act's start point
            # and has counted its ancient already (EnterNextAct); the game does both when the
            # ancient is entered, the very next decision, where both are compared again.
            diffs = [x for x in diffs if x[0] != "coord" and not x[0].endswith(".events_visited")]
    return diffs


def field_of(path: str) -> str:
    return path.split(".")[0].split("[")[0].split(" (")[0]


def component(decision: dict, diffs: list, fight: str | None = None) -> str:
    """Which part of the run a divergence belongs to: the screen, refined by the first field."""
    screen = decision.get("screen", "?")
    path = diffs[0][0]
    if fight == "in fight":
        return "combat: in fight"
    if fight == "outcome":
        return f"combat outcome -> {field_of(path)}"
    if screen == "combat":
        return f"combat: not anchored -> {field_of(path)}"
    if path in ("offer", "room", "room type", "treasure_relic", "replay") or path.startswith("event option"):
        return f"{screen} {path}"
    return f"{screen} -> {field_of(path)}"


def run(ref_path: str, emu_path: str, verbose: bool, id_map: str | None) -> int:
    refs = json.load(open(ref_path, encoding="utf-8"))["results"]
    emus = json.load(open(emu_path, encoding="utf-8"))["results"]
    nm = None
    if id_map:
        from tools.fidelity.fight_compare import names
        nm = names(json.load(open(id_map, encoding="utf-8")))
    else:
        print("fight steps NOT compared (no --id-map)")
    by_first: collections.Counter = collections.Counter()
    by_outside: collections.Counter = collections.Counter()
    in_fight: collections.Counter = collections.Counter()
    outcome_fields: collections.Counter = collections.Counter()
    unchecked: collections.Counter = collections.Counter()
    totals: collections.Counter = collections.Counter()
    clean = 0
    for ref, emu in zip(refs, emus):
        seed = ref.get("seed")
        if "decisions" not in ref:
            print(f"{seed}: no run recorded ({(ref.get('error') or '')[:200]})")
            totals["not recorded"] += 1
            continue
        decisions = ref["decisions"]
        if emu.get("seed") not in (None, seed):
            # Results are paired by position: one of another run is never a match.
            print(f"{seed}: paired with the emulator's replay of {emu.get('seed')}")
            totals["mispaired"] += 1
            continue
        edec = emu.get("decisions") or []
        totals["decisions"] += len(decisions)
        first = outside = None
        run_unchecked: set = set()
        nf = same_steps = same_outcome = 0
        notes = []
        for i, d in enumerate(decisions):
            if i >= len(edec):
                div = (i, d, "replay stopped", [("replay", "decision recorded", emu.get("error", "emulator stopped"))])
                first, outside = first or div, div
                break
            ed = edec[i]
            if d.get("snap") is not None and ed.get("snap") is not None:
                run_unchecked.update(rs.not_compared(rs.from_save(d["snap"]), ed["snap"]))
            if d.get("screen") == "event" and d.get("offer") is not None and ed.get("enabled") is not None:
                run_unchecked.update(event_options(d, ed)[1])
            if d.get("screen") == "combat":
                nf += 1
                enc = rs.enc(d.get("encounter")) or "?"
                fd = fight_diff(d, ed, nm)
                steps_compared = fd != NOT_COMPARED_FIGHT
                if not steps_compared:
                    # The steps are not compared (and never counted identical); the outcome still is.
                    run_unchecked.add(NOT_COMPARED_FIGHT)
                if fd and steps_compared:
                    in_fight[enc] += 1
                    step, act, fdiffs = fd
                    fdiv = [(f"step {step} after {rs.short(act, 60)}: {p}", g, e) for p, g, e in fdiffs[:3]]
                    kind = "in fight"
                else:
                    same_steps += steps_compared
                    fdiv = fight_outcome(d, ed, decisions[i + 1] if i + 1 < len(decisions) else None)
                    kind = "outcome"
                    if d.get("result") not in ("win", "death"):
                        run_unchecked.add(f"fight outcome (the game's run stopped inside the fight: {d.get('result')})")
                    elif fdiv:
                        outcome_fields[field_of(fdiv[0][0])] += 1
                    else:
                        same_outcome += steps_compared
                if fdiv:
                    notes.append(f"  fight at decision {i} ({enc}), {kind}: "
                                 + "; ".join(f"{p}: game {rs.short(g, 70)} emulator {rs.short(e, 70)}"
                                             for p, g, e in fdiv[:2]))
                    if first is None:
                        first = (i, d, component(d, fdiv, kind), fdiv)
            diffs = compare_decision(d, ed, nm, ref.get("maps"))
            if diffs:
                div = (i, d, component(d, diffs), diffs)
                first, outside = first or div, div
                break
        totals["fights"] += nf
        totals["same_steps"] += same_steps
        totals["same_outcome"] += same_outcome
        unchecked.update(run_unchecked)
        n = len(decisions)
        end = f"{ref.get('end')} at floor {ref.get('floor')}, {(ref.get('timing_ms') or {}).get('play', 0) / 1000:.0f} s"
        fnote = f"fights {nf}: identical steps {same_steps}, and outcome {same_outcome}"
        if first is None:
            clean += 1
            print(f"{seed}: no divergence in {n} decisions (game: {end}); {fnote}")
        else:
            i, d, comp, diffs = first
            by_first[comp] += 1
            print(f"{seed}: first divergence at decision {i}/{n} [{comp}] floor {d.get('floor')}, "
                  f"choice {rs.short(d.get('choice'), 70)} (game: {end}); {fnote}")
            for path, gv, ev in diffs[: (20 if verbose else 4)]:
                print(f"  {path}: game {rs.short(gv)} emulator {rs.short(ev)}")
        if verbose:
            for line in notes:
                print(line)
        if outside is not None:
            by_outside[outside[2]] += 1
            if outside is not first:
                i, d, comp, diffs = outside
                print(f"  first outside fights: decision {i} [{comp}] floor {d.get('floor')}: "
                      + "; ".join(f"{p}: game {rs.short(g, 80)} emulator {rs.short(e, 80)}" for p, g, e in diffs[:2]))
        if run_unchecked:
            print("  not compared here: " + ", ".join(sorted(run_unchecked)))
    nruns = len(refs)
    if len(emus) < nruns:
        print(f"{nruns - len(emus)} runs have no emulator replay")
    print(f"{nruns} runs, {totals['decisions']} decisions; {clean} with no divergence in what is compared")
    print("first divergence per run, fights included: "
          + (", ".join(f"{k} {v}" for k, v in by_first.most_common()) or "none"))
    print("first divergence outside fights (each fight anchored to the game's outcome): "
          + (", ".join(f"{k} {v}" for k, v in by_outside.most_common()) or "none"))
    print(f"fights: {totals['fights']} played, {totals['same_steps']} identical at every step, "
          f"{totals['same_outcome']} of those also ending identically (state and reward screen)")
    print("  in-fight divergences by encounter: " + (", ".join(f"{k} {v}" for k, v in in_fight.most_common()) or "none"))
    print("  outcome divergences by first field: "
          + (", ".join(f"{k} {v}" for k, v in outcome_fields.most_common()) or "none"))
    coverage(unchecked, nruns, "runs")
    return 0 if clean == nruns and not totals["not recorded"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("mode", choices=["start", "run"])
    ap.add_argument("referee")
    ap.add_argument("emulator")
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--id-map", default="data/id_map.json", help="for in-fight steps (fight_compare); '' to skip")
    a = ap.parse_args()
    if a.mode == "start":
        return start(a.referee, a.emulator)
    return run(a.referee, a.emulator, a.v, a.id_map or None)


if __name__ == "__main__":
    sys.exit(main())
