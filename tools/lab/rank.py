#!/usr/bin/env python3
"""The agent-weighted ranking of emulator divergence groups (docs/PLAN.md section 8.2, Lane B).

    python tools/lab/rank.py <score-real out dir> --out <dir>

Input: the lockstep records of a score-real / `envproto --agent play` out dir: divergences.jsonl (one record per diverging
decision: decision index, floor, room, type, hashes, structural diff, not_ranked_span, outcome, ai_state_assumed, gap),
runs.jsonl (the run ends) and summary.json. Older folders lack the span, outcome and ai_state_assumed keys; they count as
span 0, the outcome of the run's end event in runs.jsonl, and not assumed. No game, no emulator, no network.

Group key: the lab's own, tools/fidelity/lab_groups.group_key over base_key (the key groups.json uses). A small adapter
(symptom_of) turns the lockstep diff into the symptom shape: phase = decision type (no acted card), field = the differing
field paths with digits removed, sorted and comma-joined, ingredients = the differing entities (`kind:ID` via entity_of;
cards that differ between hands/piles by multiset difference, "enemy"/"player" for an HP field; at most 3). Absent in a
lockstep record: acted card, first trace event, hooks, encounter, shrunk setup. The group id is the lab's stable_id.
Each sample ref (run seed, decision, floor) is what the lab's `runs` repro step takes to shrink a first divergence.
A mapping gap (gap: true: an unmapped screen, an unmirrorable step) is not a state divergence; it is counted apart, not ranked.

Weight of a divergence = affected * floor_factor * outcome_factor
  affected       = 1 + not_ranked_span.count: the decision plus the decisions scored but not ranked after it (the lockstep was
                   lost there, so the agent's choices up to the next re-anchor were not checked against the emulator)
  floor_factor   = 1 + (50 - min(floor, 50)) / 50, from 2.0 at floor 0 to 1.0 at floor 50: an earlier divergence lets the
                   emulator drift over more of the run
  outcome_factor = 1.5 when outcome-changing CONFIRMED (both engines' outcomes recorded and different), else 1; the proxy
                   never weighs
Group weight = sum over its divergences. Decisions flagged ai_state_assumed (the search assumed enemy moves it could not
re-roll) carry no fidelity weight: they are counted apart (`ai_state_assumed`) and left out of every sum, group and rate.

Outcome-changing: the run's outcome differed between the engines. The emulator only plays until the lockstep is lost, so
usually only the referee finishes the run. Rule: (1) if the record (or the run's end event) has `emu_outcome`, it is
outcome-changing when its class or floor differs from the referee's (confirmed). (2) Otherwise a proxy: the divergence is a
state difference (not a gap), was not repaired by a re-anchor, and its diff touches state that carries into every later fight
(player hp, max hp, gold, relics, deck, potions outside the combat block) (proxy). Both are counted apart per group (outcome_changing_confirmed, outcome_changing_proxy) and never mixed.

The emulator's outcome comes from the paired arm: the same seeds played on the emulator alone (an `emu-*` folder with its own runs.jsonl:
seed, end, floor). `--emu-dir` (rank.write(emu_dir=...), rank.paired_emu_dir(folder) reads `emu_arm.dir` from the records folder's
summary.json / compare.json) joins it by seed into the run ends as `emu_outcome` {class: its end, floor}, so a divergence is confirmed
outcome-changing when the two engines' ends differ. With no arm the proxy rule above applies, and the report (summary `outcome_rule`,
top20.md, by_family.md) says which one was used; a run the arm lacks falls back to the proxy.

Tag: architectural when the diff matches a class of the audits (referee/design/game-shaped-dispatch.md, architecture-survey.md):
  rng_stream             a pile (hand, draw, discard, exhaust) holds different cards at equal size; or card reward / potion
                         contents differ in a reward screen
  action_queue           a pile differs in size (a card created, exhausted, drawn or played in a different order of actions)
  listener_order         powers, buffs, relics or energy of the combat block differ in a combat decision (no pile differs); the
                         run's relic holding (`.relics`) is carried state like `.deck`, not an order of triggers: single_site
  damage_block_pipeline  enemy hp / hp / block differ and nothing above does
A combat decision is any type in COMBAT_TYPES (combat, select_cards_combat, card_reward_combat), the one list by_family reads too.
otherwise single_site (run layer, selections outside a fight, events, shop, unmapped). First match in that order. A group takes the
most frequent class among its divergences (ties: the order above).

Summary (8.2): arch_share = architectural weight / total weight; state_divergence_rate = (non-gap, non-assumed divergences)
/ all decisions.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fidelity.lab_groups import base_key, entity_of, group_key as lab_group_key, phase_of, stable_id  # the lab's own key

OUTCOME_FACTOR = 1.5
CARRIED = (".hp", ".max_hp", ".gold", ".relics", ".deck", ".potions")
PILES = ("hand", "draw", "discard", "exhaust")
HP_FIELDS = ("enemy_hp", "block", "hp")
LISTENERS = ("power", "buff", "relic", "energy")
CLASSES = ("rng_stream", "action_queue", "listener_order", "damage_block_pipeline")
# The decision types at which a fight is on the table (referee/lockstep.py COMBAT_KINDS): a selection screen or a card reward opened
# by a card is compared with the fight's state, so it is a combat decision. The one list rank.classify and by_family both read.
COMBAT_TYPES = ("combat", "select_cards_combat", "card_reward_combat")


def floor_factor(floor: Any) -> float:
    f = float(floor) if isinstance(floor, (int, float)) else 50.0
    return 1.0 + (50.0 - max(0.0, min(f, 50.0))) / 50.0


def _read_jsonl(p: Path) -> list[dict]:
    out: list[dict] = []
    try:
        text = Path(p).read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                out.append(d)
    return out


def _kind(path: Any) -> str:
    return "".join(ch for ch in str(path) if not ch.isdigit())


def _ids(v: Any) -> list[str]:
    ids: list[str] = []
    for x in v if isinstance(v, list) else []:
        try:
            y = json.loads(x) if isinstance(x, str) and x[:1] in ('[', '"') else x
        except ValueError:
            y = x
        ids.append(str(y[0] if isinstance(y, list) and y else y))
    return ids


def _entries(diff: Any) -> list[tuple[str, Any, Any]]:
    return [(str(d[0]), d[1], d[2]) for d in diff or [] if isinstance(d, (list, tuple)) and len(d) >= 3]


def entities(diff: list) -> list[str]:
    out: set[str] = set()
    for path, a, b in _entries(diff):
        if "enemy_hp" in path:
            out.add("enemy")
        elif isinstance(a, list) and isinstance(b, list):
            pre = "relic:" if "relics" in path else "potion:" if "potions" in path else "card:"
            ca, cb = collections.Counter(_ids(a)), collections.Counter(_ids(b))
            for k in list((ca - cb).elements()) + list((cb - ca).elements()):
                out.add(entity_of(pre + k))
        elif path.endswith("hp") or path.endswith("block"):
            out.add("player")
    return sorted(out)[:3]


def classify(rec: dict) -> str:
    """One of CLASSES, or single_site."""
    ents = _entries(rec.get("diff"))
    paths = [p for p, _, _ in ents]
    piles = [(p, a, b) for p, a, b in ents if any(f".{w}" in p for w in PILES)]
    if piles:
        same = all(isinstance(a, list) and isinstance(b, list) and len(a) == len(b) for _, a, b in piles)
        return "rng_stream" if same else "action_queue"
    if rec.get("type") in ("card_reward", "card_reward_combat", "reward") and not rec.get("gap") and \
            any("card" in p or "potion" in p for p in paths):
        return "rng_stream"
    if rec.get("type") in COMBAT_TYPES:
        # listener words count on the combat block only: `.relics` is the run's holding (like `.deck`), not an order of triggers
        if any(p.startswith(".combat.") and w in p for p in paths for w in LISTENERS):
            return "listener_order"
        if any(w in p for p in paths for w in HP_FIELDS):
            return "damage_block_pipeline"
    return "single_site"


def symptom_of(rec: dict) -> dict:
    """The lab symptom (the shape lab_groups.base_key reads) of one lockstep divergence. Fields the lockstep record lacks:
    the acted card (the phase is then the decision type, else lab_groups.phase_of of the card), the first trace event, hooks,
    encounter and a shrunk setup; the differing entities stand in as the ingredients (a shrunk-style key)."""
    kinds = sorted({_kind(p) for p, _, _ in _entries(rec.get("diff"))})
    if not kinds and rec.get("reason"):  # a loss with no state diff (a phase mismatch): the reason is its kind
        kinds = ["reason:" + ":".join(str(rec["reason"]).split(":")[:2])]
    phase = rec.get("phase") or (phase_of(rec["card"]) if rec.get("card") else rec.get("type"))
    return {"source": "lockstep", "phase": phase, "field": ",".join(kinds), "first_event": rec.get("first_event"),
            "ingredients": entities(rec.get("diff") or []), "implicated": []}


def group_key(rec: dict) -> dict:
    """The lab's group key (lab_groups.group_key over lab_groups.base_key) of the divergence, as a one-key group."""
    return lab_group_key({"keys": [base_key(symptom_of(rec))], "mechanism": []})


def group_id(key: dict) -> str:
    return stable_id(key)


def outcome_changing(rec: dict, run_end: dict | None) -> str | None:
    """'confirmed', 'proxy' or None (see the module doc)."""
    emu = rec.get("emu_outcome") or (run_end or {}).get("emu_outcome")
    if emu:
        e = emu if isinstance(emu, dict) else {"class": emu}
        r = rec.get("outcome") or {"class": (run_end or {}).get("end"), "floor": (run_end or {}).get("floor")}
        differs = e.get("class") != r.get("class") or (
            e.get("floor") is not None and r.get("floor") is not None and e["floor"] != r["floor"])
        return "confirmed" if differs else None
    if rec.get("gap") or rec.get("reanchored"):
        return None
    for p, _, _ in _entries(rec.get("diff")):
        if not p.startswith(".combat.") and p.startswith(CARRIED):
            return "proxy"
    return None


def emu_outcomes(emu_dir: Path) -> dict[str, dict]:
    """The paired emulator arm's outcome by seed ({class: its end, floor}) from <emu_dir>/runs.jsonl; crashed runs have none."""
    out: dict[str, dict] = {}
    for r in _read_jsonl(Path(emu_dir) / "runs.jsonl"):
        if r.get("seed") is None or r.get("crash") or (r.get("end") is None and r.get("floor") is None):
            continue
        out[str(r["seed"])] = {"class": r.get("end"), "floor": r.get("floor")}
    return out


def join_emu(runs: list[dict], emu: dict[str, dict]) -> int:
    """Set emu_outcome on the run ends that lack one and have a seed in emu; how many run ends carry one afterwards."""
    for r in runs:
        if r.get("ev") == "end" and not r.get("emu_outcome") and str(r.get("seed")) in emu:
            r["emu_outcome"] = emu[str(r["seed"])]
    return sum(1 for r in runs if r.get("ev") == "end" and r.get("emu_outcome"))


def paired_emu_dir(folder: Path) -> Path | None:
    """The paired emulator arm a records folder names (`emu_arm.dir` in its summary.json or compare.json), when that folder exists
    and holds runs.jsonl; relative paths are tried against the records folder too."""
    folder = Path(folder)
    for name in ("summary.json", "compare.json"):
        try:
            d = json.loads((folder / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        arm = d.get("emu_arm") if isinstance(d, dict) else None
        raw = arm.get("dir") if isinstance(arm, dict) else None
        for c in ([Path(str(raw)), folder / str(raw)] if raw else []):
            if (c / "runs.jsonl").is_file():
                return c
    return None


def load(folder: Path, emu_dir: Path | None = None) -> tuple[list[dict], list[dict], dict]:
    folder = Path(folder)
    divs = _read_jsonl(folder / "divergences.jsonl")
    runs = _read_jsonl(folder / "runs.jsonl")
    if emu_dir is not None:
        join_emu(runs, emu_outcomes(Path(emu_dir)))
    try:
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = {}
    return divs, runs, summary if isinstance(summary, dict) else {}


def run_ends(runs: list[dict]) -> dict[str, dict]:
    """The run-end events of runs.jsonl by seed."""
    return {str(r.get("seed")): r for r in runs if r.get("ev") == "end"}


def weigh(rec: dict, ends: dict[str, dict]) -> tuple:
    """(record with the run's outcome filled in, group key, group id, decisions affected, outcome-changing flag, weight) of one
    state divergence (not a gap, not ai_state_assumed); the one place the weight formula of the module doc is computed."""
    run = str(rec.get("run") or rec.get("seed"))
    end = ends.get(run) or ends.get(str(rec.get("seed")))
    if not rec.get("outcome") and end:
        rec = dict(rec, outcome={"class": end.get("end"), "floor": end.get("floor")})
    key = group_key(rec)
    affected = 1 + int((rec.get("not_ranked_span") or {}).get("count") or 0)
    oc = outcome_changing(rec, end)
    w = affected * floor_factor(rec.get("floor")) * (OUTCOME_FACTOR if oc == "confirmed" else 1.0)
    return rec, key, group_id(key), affected, oc, w


def rank(folder: Path, emu_dir: Path | None = None) -> dict:
    divs, runs, summary = load(folder, emu_dir)
    ends = run_ends(runs)
    total = int(summary.get("decisions") or sum(int(e.get("asked") or 0) for e in ends.values()))
    groups: dict[str, dict] = {}
    classes: dict[str, collections.Counter] = {}
    gaps = assumed = 0
    gap_reasons: collections.Counter = collections.Counter()
    for rec in divs:
        if rec.get("gap"):
            gaps += 1
            gap_reasons[":".join(str(rec.get("reason") or "gap").split(":")[:2])] += 1
            continue
        if rec.get("ai_state_assumed"):
            assumed += 1
            continue
        run = str(rec.get("run") or rec.get("seed"))
        rec, key, gid, affected, oc, w = weigh(rec, ends)
        g = groups.setdefault(gid, {"id": gid, "key": key, "tag": "single_site", "class": "single_site", "weight": 0.0,
                                    "display": {"type": rec.get("type"), "kinds": symptom_of(rec)["field"].split(","),
                                    "entities": symptom_of(rec)["ingredients"]}, "decisions_affected": 0, "divergences": 0, "first_floor": None, "outcome_changing_confirmed": 0,
                                    "outcome_changing_proxy": 0, "samples": []})
        classes.setdefault(gid, collections.Counter())[classify(rec)] += 1
        g["weight"] += w
        g["decisions_affected"] += affected
        g["divergences"] += 1
        fl = rec.get("floor")
        if isinstance(fl, (int, float)):
            g["first_floor"] = fl if g["first_floor"] is None else min(g["first_floor"], fl)
        g["outcome_changing_confirmed"] += oc == "confirmed"
        g["outcome_changing_proxy"] += oc == "proxy"
        if len(g["samples"]) < 5:
            g["samples"].append({"run": run, "decision": rec.get("decision"), "floor": fl, "type": rec.get("type"),
                                 "diff": (rec.get("diff") or [])[:3]})
    order = (*CLASSES, "single_site")
    for gid, g in groups.items():
        c = classes[gid]
        g["class"] = max(order, key=lambda k: (c[k], -order.index(k)))
        g["tag"] = "single_site" if g["class"] == "single_site" else "architectural"
    out = sorted(groups.values(), key=lambda g: (-g["weight"], g["id"]))
    tw = sum(g["weight"] for g in out)
    arch = sum(g["weight"] for g in out if g["tag"] == "architectural")
    fid = sum(g["divergences"] for g in out)
    for i, g in enumerate(out, 1):
        g["rank"] = i
        g["weight"] = round(g["weight"], 3)
    doc = {"summary": {
        "decisions": total, "divergences_total": len(divs), "state_divergences": fid, "mapping_gaps": gaps,
        "ai_state_assumed": assumed, "groups": len(out), "total_weight": round(tw, 3), "architectural_weight": round(arch, 3),
        "arch_share": round(arch / tw, 4) if tw else 0.0,
        "state_divergence_rate": round(fid / total, 5) if total else None,
        "outcome_confirmed": sum(g["outcome_changing_confirmed"] for g in out), "outcome_proxy": sum(g["outcome_changing_proxy"] for g in out),
        "gap_reasons": dict(gap_reasons),
        "weights": "affected * (1 + (50 - min(floor, 50)) / 50) * (1.5 if outcome-changing confirmed else 1)"},
        "groups": out}
    if emu_dir is not None:
        n = sum(1 for e in ends.values() if e.get("emu_outcome"))
        doc["summary"].update(
            emu_dir=Path(emu_dir).as_posix(), emu_runs_joined=n,
            outcome_rule=f"confirmed: the paired emulator arm's end joined by seed ({n} of {len(ends)} run ends); a run the arm lacks uses the proxy")
    return doc


def top_md(doc: dict, n: int = 20) -> str:
    s = doc["summary"]
    lines = ["# Top divergence groups by agent weight", "",
             f"decisions {s['decisions']}; state divergences {s['state_divergences']} (per-decision rate "
             f"{s['state_divergence_rate']}); mapping gaps {s['mapping_gaps']}; ai_state_assumed (excluded) "
             f"{s['ai_state_assumed']}; groups {s['groups']}; architectural share of weight {s['arch_share']:.1%}; "
             f"outcome-changing confirmed {s['outcome_confirmed']}, proxy {s['outcome_proxy']}", "",
             "| rank | group | weight | decisions affected | first floor | outcome-changing confirmed | outcome-changing (proxy) | tag |",
             "|---|---|---|---|---|---|---|---|"]
    if s.get("outcome_rule"):
        lines[4:4] = [f"Outcome weighting: {s['outcome_rule']}.", ""]
    for g in doc["groups"][:n]:
        k = g["display"]
        name = f"{k['type']}: {', '.join(k['kinds']) or '-'} [{', '.join(k['entities']) or '-'}] ({g['id']})"
        tag = g["tag"] + (f" ({g['class']})" if g["tag"] == "architectural" else "")
        lines.append(f"| {g['rank']} | {name} | {g['weight']} | {g['decisions_affected']} | {g['first_floor']} | "
                     f"{g['outcome_changing_confirmed']}/{g['divergences']} | {g['outcome_changing_proxy']}/{g['divergences']} | {tag} |")
    return "\n".join(lines) + "\n"


def write(folder: Path, out: Path, emu_dir: Path | None = None) -> dict:
    doc = rank(folder, emu_dir)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ranked.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    (out / "top20.md").write_text(top_md(doc), encoding="utf-8")
    try:  # the hook-family view of the same ranking (tools/lab/by_family.py); a failure there never costs the files above
        from tools.lab import by_family  # noqa: PLC0415
        by_family.write(folder, out, doc)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, AttributeError, IndexError) as e:
        (out / "by_family.error.txt").write_text(type(e).__name__, encoding="utf-8")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--emu-dir", type=Path, default=None, help="the paired emulator arm (same seeds, emulator alone): its runs.jsonl gives emu_outcome")
    a = ap.parse_args(argv)
    print(json.dumps(write(a.folder, a.out, a.emu_dir)["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
