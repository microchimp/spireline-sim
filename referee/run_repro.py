"""Whole-run mismatches as standalone setup fights: the fix lab's minimal repro for a run.

A runs.py batch plays seeded runs in the game and replays them in the emulator; tools/fidelity/run_compare2.py
names each run's first divergence (<name>.runs.json). Most of them are in a fight (decision i, "combat: in
fight"). The game's record holds the run's save (RunSnap, SerializableRun) after every decision, so the state
just before that fight is the snap of decision i-1: deck (the save's own cards: upgrades, enchantments, saved
properties), relics (with their saved state), potions, potion slots, HP, max HP and gold, and the act.

fight_spec turns that into a setup fight spec (referee/scenarios.py, the --mode runs batch): the run's own seed and
Neow pick, its act (from act 2 on), the holdings through RefsetSetup.cs's player_load (the game's save-load path),
then the fight's encounter. The batch is played, judged, shrunk (referee/shrink.py splits player_load into its
cards, relics and potions) and traced like any setup batch, so tools/fidelity/lab_groups.py reads its minimal setups
as primary repros.

What a setup fight cannot put back (STATE_GAPS, written into each spec's "run_repro" note): the run's RNG stream
positions (Shuffle, MonsterAi, Niche, ... are at the counters Neow and the setup leave, not the run's), the floor and
map coordinate (the fight item enters at row 1, so the encounter's own Rng, seeded from seed, floor and id, differs),
potion slot positions (player_load fills slots from 0), and the run-level history (events seen, odds, grab bags,
visited rooms). A divergence that needs one of them does not reproduce: the batch's verdict then says identical or
diverges elsewhere, and the run's field is kept beside it (repro.json) to tell the two apart.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

STATE_GAPS = ("rng_counters", "floor_and_map_coord", "potion_slot_positions", "run_history")
FIGHT = "combat: in fight"
STEP = re.compile(r"^step (\d+) after .*?: (\S+)$")


def bare(x: str | None) -> str:
    return (x or "").split(".", 1)[-1]


def first_step(first: dict) -> tuple[int | None, str | None]:
    """The fight step and field run_compare2 names ("step 17 after {...}: .enemies")."""
    for d in first.get("diffs") or []:
        m = STEP.match(str(d[0]))
        if m:
            return int(m.group(1)), m.group(2)
    return None, None


def neow_pick(decisions: list) -> int:
    d = next((d for d in decisions if d.get("screen") == "neow"), None)
    return int(d.get("choice_index") or 0) if d else 0


def holdings(snap: dict) -> dict:
    """The player's holdings in a run save, as player_load takes them."""
    p = snap["players"][0]
    pots = sorted(p.get("potions") or [], key=lambda x: x.get("slot_index", 0))
    return {"deck": [dict(c) for c in p.get("deck") or []], "relics": [dict(r) for r in p.get("relics") or []],
            "potions": [bare(x.get("id")) for x in pots], "max_potion_slots": int(p.get("max_potion_slot_count") or 0),
            "hp": int(min(p["current_hp"], p["max_hp"])), "max_hp": int(p["max_hp"]), "gold": int(p.get("gold") or 0)}


def fight_spec(run: dict, verdict: dict, max_actions: int = 150) -> tuple[dict | None, str]:
    """(spec, "") for a run whose first divergence is in a fight, else (None, why not)."""
    first = verdict.get("first") or {}
    if first.get("component") != FIGHT:
        return None, f"first divergence not in a fight ({first.get('component')})"
    i = first.get("decision")
    ds = run.get("decisions") or []
    if not isinstance(i, int) or not 0 < i < len(ds) or ds[i].get("screen") != "combat":
        return None, f"decision {i} is not a recorded fight"
    before = next((ds[j].get("snap") for j in range(i - 1, -1, -1) if ds[j].get("snap")), None)
    enc = ds[i].get("encounter")
    if not before or not enc:
        return None, "no save before the fight" if not before else "no encounter"
    hold = holdings(before)
    if hold["hp"] < 1:
        return None, "hp below 1 before the fight"
    act = int(before.get("current_act_index") or 0) + 1
    step, field = first_step(first)
    load = {"cmd": "player_load", **hold}
    setup = ([{"cmd": "act", "index": act}] if act > 1 else []) + [load, {"cmd": "fight", "id": enc}]
    spec = {"kind": "combat", "seed": run["seed"], "character": run.get("character", "IRONCLAD"),
            "ascension": run.get("ascension", 10), "neow": neow_pick(ds), "chooser_seed": int(run.get("chooser_seed") or 0),
            "max_actions": max_actions, "use_potions": True, "chooser": "heuristic", "setup": setup,
            "tag": f"run:act{act}:{enc}",
            "run_repro": {"decision": i, "floor": ds[i].get("floor"), "encounter": enc, "step": step, "field": field,
                          "gaps": list(STATE_GAPS)}}
    return spec, ""


def specs_from(out_doc: dict, runs_doc: dict, limit: int = 0) -> tuple[list[dict], list[dict]]:
    """Specs for a runs.py batch (<name>.out.json, <name>.runs.json): one per diverging run whose first divergence
    is in a fight, and one note per run left out (with why)."""
    by_seed = {r.get("seed"): r for r in out_doc.get("results") or [] if isinstance(r, dict)}
    specs, skipped = [], []
    for v in runs_doc.get("runs") or []:
        if not v.get("first"):
            continue
        run = by_seed.get(v.get("seed"))
        spec, why = fight_spec(run, v) if run and run.get("decisions") else (None, "no game record")
        if spec is None:
            skipped.append({"seed": v.get("seed"), "why": why})
        elif not limit or len(specs) < limit:
            specs.append(spec)
    return specs, skipped


def load(prefix: Path) -> tuple[dict, dict]:
    """A runs.py batch by its prefix (jobs/<name>): the game's runs and the comparison's per-run records."""
    out = json.loads(prefix.with_name(prefix.name + ".out.json").read_text(encoding="utf-8"))
    runs = json.loads(prefix.with_name(prefix.name + ".runs.json").read_text(encoding="utf-8"))
    return out, runs


def matched(spec: dict, verdict: dict) -> str:
    """How a repro's verdict stands to the run's divergence: same (step and field), field (same field, other
    step), other (diverges elsewhere), identical, or error."""
    if verdict.get("error"):
        return "error"
    if verdict.get("identical"):
        return "identical"
    want = spec.get("run_repro") or {}
    from tools.fidelity.reanchor_compare import group  # noqa: PLC0415
    field = group(want["field"]) if str(want.get("field") or "").startswith(".") else want.get("field")
    if verdict.get("field") == field:
        return "same" if verdict.get("step") == want.get("step") else "field"
    return "other"
