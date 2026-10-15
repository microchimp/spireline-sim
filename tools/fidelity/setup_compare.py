"""Line up a setup fight's items on both sides: which item first leaves the emulator's run somewhere else.

    python -m tools.fidelity.setup_compare <referee out.json> <emudump fight.json>

The referee (referee/driver/Setup.cs) and EmuDump (tools/emudump/Setup.cs) log every setup item with
whether it took and the run's RNG counters after it. Map items are compared on both (combat items move
the fight's own streams, which step 0 of the fight compare covers). An item the emulator logged as
unsupported is a mirror gap: the fight's later divergences say nothing about the emulator then.
"""

from __future__ import annotations

import argparse
import json
import sys


def _streams(entry: dict) -> dict:
    rng = entry.get("rng") or {}
    out = {f"rng.{k}": (v["counter"] if isinstance(v, dict) else v) for k, v in rng.items()}
    prng = entry.get("player_rng") or {}
    out.update({f"player_rng.{k}": (v["counter"] if isinstance(v, dict) else v) for k, v in prng.items()})
    return out


def compare(game: dict, emu: dict) -> dict:
    """{"items": n, "unsupported": [i...], "first": {i, cmd, id, why, diff} or None} for one fight."""
    glog = {e["i"]: e for e in game.get("setup_log", [])}
    elog = {e["i"]: e for e in emu.get("setup_log", [])}
    fight_at = next((i for i, it in enumerate(game.get("setup", [])) if it.get("cmd") == "fight"), None)
    res = {"items": len(game.get("setup", [])), "unsupported": sorted(i for i, e in elog.items() if e.get("unsupported")),
           "first": None}
    res["gap_why"] = {i: str(elog[i].get("msg") or "")[:140] for i in res["unsupported"]}  # what the mirror said it could not do
    for i in sorted(glog):
        g, e = glog[i], elog.get(i)
        if e is None:
            res["first"] = {"i": i, "cmd": g["cmd"], "id": g.get("id"), "why": "the emulator did not run it"}
            break
        if e.get("unsupported"):
            continue
        if bool(g.get("ok")) != bool(e.get("ok")):
            res["first"] = {"i": i, "cmd": g["cmd"], "id": g.get("id"),
                            "why": f"game ok={g.get('ok')} ({g.get('msg')}), emulator ok={e.get('ok')} ({e.get('msg')})"}
            break
        if fight_at is not None and i > fight_at and i >= 0:
            continue  # combat items: the fight's own streams are compared at step 0
        gs, es = _streams(g), _streams(e)
        d = {k: (gs[k], es[k]) for k in sorted(set(gs) & set(es)) if gs[k] != es[k]}
        if d:
            why = "RNG counters already differ after Neow, before any item" if i < 0 else "RNG counters differ after it"
            res["first"] = {"i": i, "cmd": g["cmd"], "id": g.get("id"), "why": why, "diff": d}
            break
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("referee")
    ap.add_argument("emulator")
    a = ap.parse_args()
    games = json.load(open(a.referee, encoding="utf-8"))["results"]
    emus = json.load(open(a.emulator, encoding="utf-8"))["results"]
    bad = 0
    for g, e in zip(games, emus):
        if not g.get("setup"):
            continue
        r = compare(g, e)
        if r["first"] or r["unsupported"]:
            bad += 1
            print(f"{g.get('seed')}: unsupported {r['unsupported']} first {r['first']}")
    print(f"{bad} setup fights with a mirror gap or an item that differs")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
