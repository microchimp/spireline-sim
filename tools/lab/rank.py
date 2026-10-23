#!/usr/bin/env python3
"""The agent-weighted ranking of emulator divergence groups (docs/PLAN.md section 8.2, Lane B).

    python tools/lab/rank.py <score-real out dir> --out <dir>

Input: the lockstep records of a score-real / `envproto --agent play` out dir: divergences.jsonl (one record per diverging
decision: decision index, floor, room, type, hashes, structural diff, not_ranked_span, outcome, ai_state_assumed, gap),
runs.jsonl (the run ends) and summary.json. Older folders lack the span, outcome and ai_state_assumed keys; they count as
span 0, the outcome of the run's end event in runs.jsonl, and not assumed. No game, no emulator, no network.

Group key (the lab's: tools/fidelity/lab_groups.py keys a symptom by phase, field group, first event and ingredient
entities; a divergence record has no trace, so the first event is absent): (decision type, state-diff kinds, entities).
  - kinds: the differing field paths with digits removed, sorted (".combat.enemy_hp", ".combat.hand", ".hp").
  - entities: `kind:ID` strings as lab_groups.entity_of writes them: the cards that differ between the two hands/piles
    (multiset difference), "enemy" / "player" for an HP field. At most 3, sorted.
Each sample ref (run seed, decision, floor) is what the lab's `runs` repro step takes to shrink a first divergence to a setup
fight; the group id is a hash of the key and is NOT merged into lab groups.json ids (no trace to match on).
A mapping gap (gap: true: an unmapped screen, an unmirrorable step) is not a state divergence; it is counted apart, not ranked.

Weight of a divergence = affected * floor_factor * outcome_factor
  affected       = 1 + not_ranked_span.count: the decision plus the decisions scored but not ranked after it (the lockstep was
                   lost there, so the agent's choices up to the next re-anchor were not checked against the emulator)
  floor_factor   = 1 + (50 - min(floor, 50)) / 50, from 2.0 at floor 0 to 1.0 at floor 50: an earlier divergence lets the
                   emulator drift over more of the run
  outcome_factor = 1.5 when outcome-changing, else 1
Group weight = sum over its divergences. Decisions flagged ai_state_assumed (the search assumed enemy moves it could not
re-roll) carry no fidelity weight: they are counted apart (`ai_state_assumed`) and left out of every sum, group and rate.

Outcome-changing: the run's outcome differed between the engines. The emulator only plays until the lockstep is lost, so
usually only the referee finishes the run. Rule: (1) if the record (or the run's end event) has `emu_outcome`, it is
outcome-changing when its class or floor differs from the referee's (confirmed). (2) Otherwise a proxy: the divergence is a
state difference (not a gap), was not repaired by a re-anchor, and its diff touches state that carries into every later fight
(player hp, max hp, gold, relics, deck, potions outside the combat block) (proxy). Both counts are reported.

Tag: architectural when the diff matches a class of the audits (referee/design/game-shaped-dispatch.md, architecture-survey.md):
  rng_stream             a pile (hand, draw, discard, exhaust) holds different cards at equal size; or card reward / potion
                         contents differ in a reward screen
  action_queue           a pile differs in size (a card created, exhausted, drawn or played in a different order of actions)
  listener_order         powers, buffs, relics or energy differ in a combat (no pile differs)
  damage_block_pipeline  enemy hp / hp / block differ and nothing above does
otherwise single_site (run layer, selections, events, shop, unmapped). First match in that order. A group takes the
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

try:  # the lab's own entity naming
    from tools.fidelity.lab_groups import entity_of
except Exception:  # noqa: BLE001 -- keep the ranking usable without the lab's heavy imports
    def entity_of(key: str) -> str:
        k = key.split("@")[0]
        return k[:-1] if k.endswith("+") else k

OUTCOME_FACTOR = 1.5
CARRIED = (".hp", ".max_hp", ".gold", ".relics", ".deck", ".potions")
PILES = ("hand", "draw", "discard", "exhaust")
HP_FIELDS = ("enemy_hp", "block", "hp")
LISTENERS = ("power", "buff", "relic", "energy")
CLASSES = ("rng_stream", "action_queue", "listener_order", "damage_block_pipeline")


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
    if rec.get("type") == "combat":
        if any(w in p for p in paths for w in LISTENERS):
            return "listener_order"
        if any(w in p for p in paths for w in HP_FIELDS):
            return "damage_block_pipeline"
    return "single_site"


def group_key(rec: dict) -> dict:
    kinds = sorted({_kind(p) for p, _, _ in _entries(rec.get("diff"))})
    key = {"type": rec.get("type"), "kinds": kinds, "entities": entities(rec.get("diff") or [])}
    if not kinds and rec.get("reason"):  # a loss with no state diff (a phase mismatch): the reason is its kind
        key["kinds"] = ["reason:" + ":".join(str(rec["reason"]).split(":")[:2])]
    return key


def group_id(key: dict) -> str:
    return "R" + hashlib.sha1(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:10]


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


def load(folder: Path) -> tuple[list[dict], list[dict], dict]:
    folder = Path(folder)
    divs = _read_jsonl(folder / "divergences.jsonl")
    runs = _read_jsonl(folder / "runs.jsonl")
    try:
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = {}
    return divs, runs, summary if isinstance(summary, dict) else {}


def rank(folder: Path) -> dict:
    divs, runs, summary = load(folder)
    ends = {str(r.get("seed")): r for r in runs if r.get("ev") == "end"}
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
        end = ends.get(run) or ends.get(str(rec.get("seed")))
        if not rec.get("outcome") and end:
            rec = dict(rec, outcome={"class": end.get("end"), "floor": end.get("floor")})
        key = group_key(rec)
        gid = group_id(key)
        affected = 1 + int((rec.get("not_ranked_span") or {}).get("count") or 0)
        oc = outcome_changing(rec, end)
        w = affected * floor_factor(rec.get("floor")) * (OUTCOME_FACTOR if oc else 1.0)
        g = groups.setdefault(gid, {"id": gid, "key": key, "tag": "single_site", "class": "single_site", "weight": 0.0,
                                    "decisions_affected": 0, "divergences": 0, "first_floor": None, "outcome_confirmed": 0,
                                    "outcome_proxy": 0, "samples": []})
        classes.setdefault(gid, collections.Counter())[classify(rec)] += 1
        g["weight"] += w
        g["decisions_affected"] += affected
        g["divergences"] += 1
        fl = rec.get("floor")
        if isinstance(fl, (int, float)):
            g["first_floor"] = fl if g["first_floor"] is None else min(g["first_floor"], fl)
        g["outcome_confirmed"] += oc == "confirmed"
        g["outcome_proxy"] += oc == "proxy"
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
        g["outcome_changing_share"] = round((g["outcome_confirmed"] + g["outcome_proxy"]) / g["divergences"], 3)
    return {"summary": {
        "decisions": total, "divergences_total": len(divs), "state_divergences": fid, "mapping_gaps": gaps,
        "ai_state_assumed": assumed, "groups": len(out), "total_weight": round(tw, 3), "architectural_weight": round(arch, 3),
        "arch_share": round(arch / tw, 4) if tw else 0.0,
        "state_divergence_rate": round(fid / total, 5) if total else None,
        "outcome_confirmed": sum(g["outcome_confirmed"] for g in out), "outcome_proxy": sum(g["outcome_proxy"] for g in out),
        "gap_reasons": dict(gap_reasons),
        "weights": "affected * (1 + (50 - min(floor, 50)) / 50) * (1.5 if outcome-changing else 1)"},
        "groups": out}


def top_md(doc: dict, n: int = 20) -> str:
    s = doc["summary"]
    lines = ["# Top divergence groups by agent weight", "",
             f"decisions {s['decisions']}; state divergences {s['state_divergences']} (per-decision rate "
             f"{s['state_divergence_rate']}); mapping gaps {s['mapping_gaps']}; ai_state_assumed (excluded) "
             f"{s['ai_state_assumed']}; groups {s['groups']}; architectural share of weight {s['arch_share']:.1%}; "
             f"outcome-changing confirmed {s['outcome_confirmed']}, proxy {s['outcome_proxy']}", "",
             "| rank | group | weight | decisions affected | first floor | outcome-changing share | tag |",
             "|---|---|---|---|---|---|---|"]
    for g in doc["groups"][:n]:
        k = g["key"]
        name = f"{k['type']}: {', '.join(k['kinds']) or '-'} [{', '.join(k['entities']) or '-'}] ({g['id']})"
        tag = g["tag"] + (f" ({g['class']})" if g["tag"] == "architectural" else "")
        lines.append(f"| {g['rank']} | {name} | {g['weight']} | {g['decisions_affected']} | {g['first_floor']} | "
                     f"{g['outcome_changing_share']:.0%} | {tag} |")
    return "\n".join(lines) + "\n"


def write(folder: Path, out: Path) -> dict:
    doc = rank(folder)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ranked.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    (out / "top20.md").write_text(top_md(doc), encoding="utf-8")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    print(json.dumps(write(a.folder, a.out)["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
