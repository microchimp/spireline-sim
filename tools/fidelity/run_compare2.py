"""Compare whole runs: the game's run snapshots (referee) with the emulator's, decision by decision.

    python -m tools.fidelity.run_compare2 start <referee out.json> <EmuDump runstart json>
    python -m tools.fidelity.run_compare2 run <referee out.json> <EmuDump run json> [-v]

`start`: the run-start save of each run (any referee result that carries "save") against
`EmuDump runstart`: all three acts' room sets, both relic grab bags, the odds, every RNG counter.
`run`: each recorded run decision (referee/driver/RunPlay.cs) against the emulator's replay of the same
choices (tools/emudump/RunReplay.cs); prints, per run, the first decision whose snapshot or offer differs
and the fields that differ there, then a summary by component. Exit 0 when everything matches.
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
    for r in refs:
        g = rs.from_save(r["save"])
        e = emus.get(r["seed"])
        if e is None:
            print(f"{r['seed']}: no emulator result")
            bad += 1
            continue
        diffs = rs.compare(g, e["snap"])
        if diffs:
            bad += 1
            print(f"{r['seed']}: {len(diffs)} differences")
            for path, gv, ev in diffs[:8]:
                print(f"  {path}: game {rs.short(gv)} emulator {rs.short(ev)}")
    print(f"{len(refs)} run starts compared, {bad} with differences")
    return 0 if not bad and refs else 1


def norm_offer(screen: str, offer, emulator: bool):
    """An offer as a comparable value: what the screen lets the player take, in a fixed order."""
    if offer is None:
        return None
    if screen in ("neow", "treasure"):
        return sorted(rs.bare(x[0] if isinstance(x, list) else x) for x in offer)
    if screen == "map":
        return sorted((x[0], x[1], x[2]) for x in offer)
    if screen == "rewards":
        # Known gap, not hidden: a Thieving Hopper's stolen card comes back in the game as a
        # SpecialCardReward the player claims; the emulator puts it straight back into the deck
        # (RunRewardGenerator.ReturnStolenCards). The deck is compared (and anchored) instead.
        return sorted(json.dumps(norm_reward(r), sort_keys=True) for r in offer
                      if r.get("type") not in ("special_card", "SpecialCardReward"))
    if screen == "rest":
        return sorted(x[0] for x in offer if x[1])
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
    return r


def room_of(screen: str, game: dict, emu: dict) -> list:
    """The room a map move entered: encounter or event id, compared without variant suffixes."""
    out = []
    groom, eroom = game.get("room"), emu.get("room")
    if screen != "map" or not groom or not eroom:
        return out
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


def fight_diff(game: dict, emu: dict, nm: dict | None):
    """The first fight step whose state differs (fight_compare's views), or None."""
    if nm is None or "fight" not in game or "fight" not in emu:
        return None
    from tools.fidelity.fight_compare import diff, emu_view, game_view
    for i, (gs, es) in enumerate(zip(game["fight"]["steps"], emu["fight"]["steps"])):
        d = diff(game_view(gs["state"]), emu_view(es["state"], nm))
        if d:
            return i, gs["action"], d
    return None


def compare_decision(d: dict, ed: dict, nm) -> list:
    diffs = []
    if ed.get("error"):
        diffs.append(("replay", d.get("choice"), ed["error"]))
    screen = d.get("screen", "")
    go, eo = norm_offer(screen, d.get("offer"), False), norm_offer(screen, ed.get("offer"), True)
    if go is not None and eo is not None and go != eo:
        diffs.append(("offer", go, eo))
    if screen == "event" and d.get("offer") is not None and ed.get("enabled") is not None:
        g_en = [i for i, o in enumerate(d["offer"]) if not o[1]]
        e_en = [i for i in ed["enabled"] if i < len(d["offer"])]
        if g_en != e_en:
            diffs.append(("event options enabled", g_en, e_en))
    diffs += room_of(screen, d, ed)
    if not diffs and d.get("snap") is not None and ed.get("snap") is not None:
        diffs = rs.compare(rs.from_save(d["snap"]), ed["snap"])
        if screen == "act_change":
            # Representation, not rules: on a new act the emulator stands on the act's start point
            # and has counted its ancient already (EnterNextAct); the game does both when the
            # ancient is entered, the very next decision, where both are compared again.
            diffs = [x for x in diffs if x[0] != "coord" and not x[0].endswith(".events_visited")]
    return diffs


def component(decision: dict, diffs: list) -> str:
    """Which part of the run a divergence belongs to: the screen, refined by the first field."""
    screen = decision.get("screen", "?")
    path = diffs[0][0]
    if screen == "combat":
        return "combat (after fight)"
    if path in ("offer", "room", "treasure_relic", "replay"):
        return f"{screen} {path}"
    return f"{screen} -> {path.split('.')[0].split('[')[0]}"


def run(ref_path: str, emu_path: str, verbose: bool, id_map: str | None) -> int:
    refs = json.load(open(ref_path, encoding="utf-8"))["results"]
    emus = json.load(open(emu_path, encoding="utf-8"))["results"]
    nm = None
    if id_map:
        from tools.fidelity.fight_compare import names
        nm = names(json.load(open(id_map, encoding="utf-8")))
    by_comp: collections.Counter = collections.Counter()
    fight_firsts: collections.Counter = collections.Counter()
    anchor_counts: collections.Counter = collections.Counter()
    total_fights = [0, 0]
    bad = 0
    total_dec = 0
    for ref, emu in zip(refs, emus):
        seed = ref.get("seed")
        if "decisions" not in ref:
            print(f"{seed}: no run recorded ({(ref.get('error') or '')[:200]})")
            continue
        decisions = ref["decisions"]
        edec = emu.get("decisions") or []
        total_dec += len(decisions)
        first = None
        anchored_fights = []
        fights = []  # (decision, encounter, first in-fight step that differs or None)
        for i, d in enumerate(decisions):
            if i >= len(edec):
                first = (i, d, [("replay", "decision recorded", emu.get("error", "emulator stopped"))])
                break
            if d.get("screen") == "combat":
                fd = fight_diff(d, edec[i], nm)
                fights.append((i, d.get("encounter"), fd))
                if fd:
                    fight_firsts[rs.enc(d.get("encounter")) or "?"] += 1
            diffs = compare_decision(d, edec[i], nm)
            if edec[i].get("anchored"):
                # The replay put the emulator where the game came out of this fight (RunReplay's
                # anchoring): a combat divergence, counted, and the run goes on being compared.
                anchored_fights.append((i, d.get("encounter"), edec[i]["anchored"]))
                anchor_counts[rs.enc(d.get("encounter")) or "?"] += 1
                if not [x for x in diffs if x[0] != "replay"]:
                    diffs = []
            if diffs:
                if d.get("screen") == "combat" and fights and fights[-1][2]:
                    _, act, fdiffs = fights[-1][2]
                    diffs += [(f"in fight, step {fights[-1][2][0]} after {rs.short(act, 60)}: {p}", g, e)
                              for p, g, e in fdiffs[:3]]
                first = (i, d, diffs)
                break
        nf = len(fights)
        fights_ok = sum(1 for f in fights if not f[2])
        total_fights[0] += nf
        total_fights[1] += fights_ok
        n = len(decisions)
        end = f"{ref.get('end')} at floor {ref.get('floor')}, {(ref.get('timing_ms') or {}).get('play', 0) / 1000:.0f} s"
        fight_note = f"; fights identical {fights_ok}/{nf}, outcome anchored {len(anchored_fights)}"
        if verbose:
            for i, enc, fd in fights:
                if fd:
                    print(f"  fight at decision {i} ({enc}): step {fd[0]} after {rs.short(fd[1], 70)}: "
                          + "; ".join(f"{p}: game {rs.short(g, 70)} emulator {rs.short(e, 70)}" for p, g, e in fd[2][:2]))
        if first is None:
            print(f"{seed}: {n} decisions identical (game: {end}){fight_note}")
            continue
        bad += 1
        i, d, diffs = first
        comp = component(d, diffs)
        by_comp[comp] += 1
        print(f"{seed}: first divergence at decision {i}/{n} [{comp}] floor {d.get('floor')}, "
              f"choice {rs.short(d.get('choice'), 70)} (game: {end}){fight_note}")
        for path, gv, ev in diffs[: (20 if verbose else 4)]:
            print(f"  {path}: game {rs.short(gv)} emulator {rs.short(ev)}")
    print(f"{len(refs)} runs, {total_dec} decisions; {bad} runs diverge; first run-level divergence by component: "
          + (", ".join(f"{k} {v}" for k, v in by_comp.most_common()) or "none"))
    print(f"fights: {total_fights[0]} played, {total_fights[1]} identical at every step; encounters with an in-fight "
          "divergence: " + (", ".join(f"{k} {v}" for k, v in fight_firsts.most_common()) or "none"))
    print("fights whose outcome had to be anchored (HP, deck, RNG...): "
          + (", ".join(f"{k} {v}" for k, v in anchor_counts.most_common()) or "none"))
    return 0 if not bad else 1


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
