#!/usr/bin/env python3
"""Divergence by hook family (docs/PLAN.md section 8.2, Lane B): the agent weight of the ranked divergence groups per game hook
family, written beside top20.md as by_family.md / by_family.json by every score-real analysis (tools/lab/rank.py write()).

    python tools/lab/by_family.py <score-real out dir> --out <dir>

Families. F01-F17 are the hook families of referee/design/game-shaped-dispatch.md section 2 (the audits' hook families, as that
file lists them); the coarse family is the audits' grouping of PLAN 8.2 (exhaust and discard triggers, damage and block pipeline,
start- and end-of-turn hooks, card-play hooks, power stacking and removal, relic triggers, RNG order) plus two this module adds
because the lockstep sees them: "death and victory" (F10, F17) and "run flow" (the run layer: rewards, shop, rest, map, events, deck).

What a lockstep divergence carries: the decision's type, the differing fields with both values (hand, enemy_hp, hp, deck, gold,
potions, relics ...), a reason when there is no state diff, and no acted card, no trace event and no hook. So a rule can read the
SHAPE of the diff (which fields, sizes, order, direction), not the hook that ran. Each rule says how strong its evidence is:
  diff    the differing field is the hook family's own state (an enemy listed on one side only; hp and enemy hp alone)
  shape   the field set and sizes fit one hook moment best (a fresh hand of equal size means the draw), but another moment could
          produce the same diff; a trace-mode L patch (the game's hook calls per step) would turn this into evidence
  reason  the record's own reason names the moment (the referee is at the rewards screen while the emulator is in combat)
  type    only the decision type and field name place it (run layer)
A divergence no rule places stays in the `unassigned` bucket, never a guess. Rules are tried in order; the first match wins.
READINGS lists the other readings of a boundary (the draw as F03, a listed corpse as an F10 death hook); the top-five share is
reported under the primary reading and under each of them.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.lab import rank  # noqa: E402

UNASSIGNED = "unassigned"
FAMILIES = {
    "F01": "combat start", "F02": "side turn start", "F03": "player turn start", "F04": "auto-play", "F05": "turn end / flush",
    "F06": "card play", "F07": "costs", "F08": "damage / HP", "F09": "block", "F10": "death", "F11": "card piles",
    "F12": "powers", "F13": "energy / orbs", "F14": "potions", "F15": "run rewards", "F16": "run shop / rest / map / deck", "F17": "combat end"}
COARSE = ("exhaust and discard triggers", "damage and block pipeline", "start- and end-of-turn hooks", "card-play hooks",
          "power stacking and removal", "relic triggers", "RNG order", "death and victory", "run flow")
EVIDENCE = ("diff", "reason", "type", "shape", "none")  # strongest first
SENTINEL_HP = 10000  # an enemy hp delta this large is a stand-in value (999999999 against 32768), not a damage difference
COMBAT_TYPES = rank.COMBAT_TYPES  # one list with rank.classify
REWARD_TYPES = ("reward", "card_reward", "card_reward_combat")
HP_KINDS = {".combat.enemy_hp", ".hp"}

# id, matcher, family, coarse family, evidence, reason. First match wins.
RULES = [
    ("R01", "phase_mismatch", "F10", "death and victory", "reason", "the referee is at the rewards screen while the emulator is still in combat: the combat-end win check at the action boundary (design 2: victory is not checked there); cause unverified"),
    ("R02", "screen_mismatch", "F16", "run flow", "reason", "the run layer shows another screen or event page (rest, map, event): flow, not a combat hook"),
    ("R03", "reward_hp", "F17", "death and victory", "diff", "hp differs at the reward screen: what runs at the victory (AfterCombatVictory heals) differs"),
    ("R04", "enemy_count", UNASSIGNED, "", "none", "an enemy with 0 hp is listed on the referee side only: the emulator's lockstep view keeps hp above 0 only, the referee's keeps every enemy the game lists (referee/lockstep.py), so this is a comparer asymmetry unless the corpse matters to play; reading C puts it in F10 (AfterDeath, removal after death)"),
    ("R05", "sentinel_hp", UNASSIGNED, "", "none", "an enemy hp stand-in value (999999999 against 32768) is a representation difference of the encounter, not a hook; needs the encounter"),
    ("R06", "hp_only", "F08", "damage and block pipeline", "diff", "only enemy hp and player hp differ: a damage amount (modifier folds, rounding) or an hp change"),
    ("R07", "block_only", "F09", "damage and block pipeline", "diff", "only block differs: the block gain passes"),
    ("R08", "hand_order_only", "F11", "exhaust and discard triggers", "shape", "the same cards in another order: pile insertion order (CardPileCmd position), not draw content"),
    ("R09", "hand_upgrade_only", UNASSIGNED, "", "none", "the same cards at another upgrade level (a curse upgraded on one side): an in-combat upgrade effect picked a card the game cannot upgrade; which effect needs the trace"),
    ("R10", "hand_same_size", "F11", "RNG order", "shape", "equal-size hands of other cards: a fresh draw out of a different draw-pile order (shuffle stream and pile order); reading B if read as the turn-start draw"),
    ("R11", "hand_off_by_one", "F11", "exhaust and discard triggers", "shape", "one card more on one side only: a card drawn, discarded or exhausted on one side (CardPileCmd); reading B if the turn-start draw count"),
    ("R12", "hand_size_gap", UNASSIGNED, "", "none", "hands differ by two or more cards or both ways: a turn-boundary flush, a draw count and several plays all fit; needs the trace"),
    ("R13", "combat_multi_field", UNASSIGNED, "", "none", "hand with hp, enemy hp or another field in one diff: each is a consequence of an upstream action or draw the diff does not name; needs the trace"),
    ("R14", "gold_reward", "F15", "RNG order", "type", "gold differs at the reward screen: the gold roll and the Rewards stream draws (kit G2877adf2ff)"),
    ("R15", "potion_reward", "F15", "RNG order", "type", "potion or card reward contents differ at a reward screen: reward generation (rng stream)"),
    ("R16", "deck_or_relics_carried", "F15", "run flow", "type", "deck, gold or relics differ at a combat start or reward screen: a card taken from a reward on one side only (kit G05884084d1 lists the reward flow first; an ancient or event card is the other candidate, F16)"),
    ("R17", "run_screen", "F16", "run flow", "type", "a run-layer screen (select_cards, treasure, map, event, shop, rest) differs in a carried field"),
    ("R18", "no_rule", UNASSIGNED, "", "none", "no rule places this diff"),
]
# Other readings of a boundary: rule id -> the family it would join. Each is reported beside the primary top-five share.
READINGS = {
    "B: the draw is F03": ({"R10": "F03", "R11": "F03"}, "the dispatch file lists ModifyHandDraw / BeforeHandDraw under F03 (player turn start) and the draw, shuffle and pile moves under F11 (card piles); a hand drawn at turn start can be read either way"),
    "C: a listed corpse is a death hook": ({"R04": "F10"}, "if the game keeps a dead enemy in its roster (ShouldCreatureBeRemovedFromCombatAfterDeath refuses) and the emulator removes it, R04 is a death-removal difference in F10; as it stands the comparer's asymmetry alone produces the diff"),
}
RULE_IDS = {r[0]: r for r in RULES}
# unassigned sub-reasons shown in the report are the rule ids above (R05, R11, R12, R17)


# ---------------------------------------------------------------- features of one divergence

def _kinds(rec: dict) -> set[str]:
    return {rank._kind(p) for p, _, _ in rank._entries(rec.get("diff"))}


def _entry(rec: dict, kind: str) -> tuple[Any, Any] | None:
    for p, a, b in rank._entries(rec.get("diff")):
        if rank._kind(p) == kind:
            return a, b
    return None


def hand_shape(a: Any, b: Any) -> dict:
    """How the referee hand a and the emulator hand b differ: sizes, cards on one side only, order-only, who holds a superset."""
    la = [str(x) for x in a] if isinstance(a, list) else []
    lb = [str(x) for x in b] if isinstance(b, list) else []
    ca, cb = collections.Counter(la), collections.Counter(lb)
    ref_only, emu_only = ca - cb, cb - ca
    return {"ref": len(la), "emu": len(lb), "ref_only": sum(ref_only.values()), "emu_only": sum(emu_only.values()),
            "order_only": not ref_only and not emu_only and la != lb,
            "names_equal": collections.Counter(rank._ids(la)) == collections.Counter(rank._ids(lb)),
            "ref_names": dict(collections.Counter(rank._ids(list(ref_only.elements())))),
            "emu_names": dict(collections.Counter(rank._ids(list(emu_only.elements()))))}


def enemy_shape(a: Any, b: Any) -> dict:
    la = list(a) if isinstance(a, list) else []
    lb = list(b) if isinstance(b, list) else []
    deltas = [y - x for x, y in zip(la, lb) if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x != y]
    return {"ref": len(la), "emu": len(lb), "max_abs_delta": max((abs(d) for d in deltas), default=0)}


def _match(name: str, rec: dict) -> bool:
    typ, kinds, reason = rec.get("type"), _kinds(rec), str(rec.get("reason") or "")
    hand = _entry(rec, ".combat.hand")
    enemy = _entry(rec, ".combat.enemy_hp")
    combat = typ in COMBAT_TYPES
    if name == "phase_mismatch":
        return not kinds and reason.startswith("phase:")
    if name == "screen_mismatch":
        return not kinds and reason.startswith(("screen:", "event:"))
    if name == "reward_hp":
        return typ == "reward" and kinds == {".hp"}
    if name == "enemy_count":
        return combat and kinds == {".combat.enemy_hp"} and enemy is not None and len(enemy[0]) != len(enemy[1])
    if name == "sentinel_hp":
        return combat and enemy is not None and enemy_shape(*enemy)["max_abs_delta"] >= SENTINEL_HP
    if name == "hp_only":
        return combat and bool(kinds) and kinds <= HP_KINDS
    if name == "block_only":
        return combat and kinds == {".combat.block"}
    if name.startswith("hand_"):
        if not combat or kinds != {".combat.hand"} or hand is None:
            return False
        h = hand_shape(*hand)
        if name == "hand_order_only":
            return h["order_only"]
        if name == "hand_upgrade_only":
            return h["names_equal"] and not h["order_only"] and h["ref"] == h["emu"]
        if name == "hand_same_size":
            return h["ref"] == h["emu"] and (h["ref_only"] > 0 or h["emu_only"] > 0)
        if name == "hand_off_by_one":
            return abs(h["ref"] - h["emu"]) == 1 and (h["ref_only"] == 0 or h["emu_only"] == 0)
        return name == "hand_size_gap" and h["ref"] != h["emu"]  # every other unequal pair
    if name == "combat_multi_field":
        return combat and hand is not None and len(kinds) > 1
    if name == "gold_reward":
        return typ == "reward" and kinds == {".gold"}
    if name == "potion_reward":
        return typ in REWARD_TYPES and ".potions" in kinds
    if name == "deck_or_relics_carried":
        return bool(kinds) and kinds <= {".deck", ".gold", ".relics"} and (combat or typ in REWARD_TYPES)
    if name == "run_screen":
        return typ in ("select_cards", "treasure", "map", "event", "shop", "rest") and bool(kinds)
    return name == "no_rule"


def assign(rec: dict) -> dict:
    """The first matching rule of RULES for one divergence: {rule, family, coarse, evidence}."""
    for rid, matcher, family, coarse, evidence, _ in RULES:
        if _match(matcher, rec):
            return {"rule": rid, "family": family, "coarse": coarse or UNASSIGNED, "evidence": evidence}
    return {"rule": "R18", "family": UNASSIGNED, "coarse": UNASSIGNED, "evidence": "none"}


def sample_record(g: dict, s: dict) -> dict:
    """A divergence-shaped record from a ranked group's sample (ranked.json has the first three diff entries of 5 samples)."""
    kinds = (g.get("display") or {}).get("kinds") or []
    reason = kinds[0][len("reason:"):] if len(kinds) == 1 and kinds[0].startswith("reason:") and not s.get("diff") else None
    return {"type": s.get("type") or (g.get("display") or {}).get("type"), "diff": s.get("diff") or [], "reason": reason}


# ---------------------------------------------------------------- the parts (one per group x assignment) and the table

def parts_from_records(folder: Path, doc: dict) -> list[dict]:
    """One part per divergence, weighted as rank.rank weighs it (rank.weigh), with its group's tag from doc."""
    divs, runs, _ = rank.load(Path(folder))
    ends = rank.run_ends(runs)
    groups = {g["id"]: g for g in doc.get("groups", [])}
    parts = []
    for rec in divs:
        if rec.get("gap") or rec.get("ai_state_assumed"):
            continue
        rec2, _, gid, affected, oc, w = rank.weigh(rec, ends)
        a = assign(rec2)
        parts.append({**a, "group": gid, "tag": (groups.get(gid) or {}).get("tag", "single_site"), "weight": w,
                      "affected": affected, "confirmed": oc == "confirmed", "divergences": 1})
    return parts


def parts_from_ranked(doc: dict) -> list[dict]:
    """One part per ranked group, placed by the plurality of its samples (ranked.json only: no per-divergence records)."""
    parts = []
    for g in doc.get("groups", []):
        votes = collections.Counter()
        first: dict[tuple, dict] = {}
        for s in g.get("samples") or []:
            a = assign(sample_record(g, s))
            votes[(a["rule"], a["family"])] += 1
            first.setdefault((a["rule"], a["family"]), a)
        a = first[max(votes, key=lambda k: (votes[k], k))] if votes else assign(
            {"type": (g.get("display") or {}).get("type"), "diff": [], "reason": None})
        parts.append({**a, "group": g["id"], "tag": g.get("tag", "single_site"), "weight": float(g.get("weight") or 0.0),
                      "affected": int(g.get("decisions_affected") or 0), "confirmed_n": int(g.get("outcome_changing_confirmed") or 0),
                      "divergences": int(g.get("divergences") or 0), "from_samples": True})
    return parts


def _row(name: str, ps: list[dict], total: float, arch_total: float, key: str = "family") -> dict:
    w = sum(p["weight"] for p in ps)
    aw = sum(p["weight"] for p in ps if p["tag"] == "architectural")
    return {key: name, "weight": round(w, 3), "share": round(w / total, 4) if total else 0.0,
            "arch_weight": round(aw, 3), "arch_share_within": round(aw / w, 4) if w else 0.0,
            "arch_share_of_arch": round(aw / arch_total, 4) if arch_total else 0.0,
            "groups": len({p["group"] for p in ps}), "divergences": sum(p["divergences"] for p in ps),
            "decisions_affected": sum(p["affected"] for p in ps),
            "outcome_changing_confirmed": sum(p.get("confirmed_n", 1 if p.get("confirmed") else 0) for p in ps)}


def _top5(parts: list[dict], family_of, arch_total: float) -> dict:
    by = collections.defaultdict(float)
    tot = collections.defaultdict(float)
    for p in parts:
        f = family_of(p)
        if f == UNASSIGNED:
            continue
        tot[f] += p["weight"]
        if p["tag"] == "architectural":
            by[f] += p["weight"]
    top = sorted(by, key=lambda f: (-by[f], f))[:5]
    top_w = sorted(tot, key=lambda f: (-tot[f], f))[:5]
    return {"families": top, "arch_weight": round(sum(by[f] for f in top), 3),
            "share_of_arch": round(sum(by[f] for f in top) / arch_total, 4) if arch_total else 0.0,
            "by_total_weight": {"families": top_w, "share_of_arch": round(sum(by.get(f, 0.0) for f in top_w) / arch_total, 4) if arch_total else 0.0}}


def verdict(share: float) -> str:
    if share >= 0.70:
        return "Tier 1 (top five hold at least 70% of the architectural weight)"
    if share < 0.50:
        return "Tier 2 (top five under 50%)"
    return "between the PLAN 8.3 thresholds (50% to 70%): neither rule fires"


def table(parts: list[dict], groups: list[dict] | None = None, summary: dict | None = None) -> dict:
    total = sum(p["weight"] for p in parts)
    arch = sum(p["weight"] for p in parts if p["tag"] == "architectural")
    fams = sorted({p["family"] for p in parts if p["family"] != UNASSIGNED})
    rows = [_row(f, [p for p in parts if p["family"] == f], total, arch) for f in fams]
    rows.sort(key=lambda r: (-r["weight"], r["family"]))
    cum = 0.0
    for r in rows:
        r["name"] = FAMILIES.get(r["family"], r["family"])
        cw = collections.Counter()
        for p in parts:
            if p["family"] == r["family"]:
                cw[p["coarse"]] += p["weight"]
        r["coarse"] = max(cw, key=lambda c: (cw[c], c)) if cw else ""
        r["coarse_split"] = {c: round(v, 3) for c, v in sorted(cw.items(), key=lambda kv: -kv[1])}
        cum += r["weight"]
        r["cum_share"] = round(cum / total, 4) if total else 0.0
    coarse = [_row(c, [p for p in parts if p["coarse"] == c], total, arch, "coarse") for c in sorted({p["coarse"] for p in parts if p["family"] != UNASSIGNED})]
    coarse.sort(key=lambda r: (-r["weight"], r["coarse"]))
    un = [p for p in parts if p["family"] == UNASSIGNED]
    unrow = _row(UNASSIGNED, un, total, arch)
    unrow["by_rule"] = {rid: round(sum(p["weight"] for p in un if p["rule"] == rid), 3) for rid in sorted({p["rule"] for p in un})}
    t5 = _top5(parts, lambda p: p["family"], arch)
    readings = {}
    for name, (moves, why) in READINGS.items():
        r = _top5(parts, lambda p, m=moves: m.get(p["rule"], p["family"]), arch)
        readings[name] = {**r, "why": why, "verdict": verdict(r["share_of_arch"])}
    both = {k: v for m, _ in READINGS.values() for k, v in m.items()}
    t5_all = _top5(parts, lambda p: both.get(p["rule"], p["family"]), arch)
    readings["B and C together"] = {**t5_all, "why": "both boundary readings at once", "verdict": verdict(t5_all["share_of_arch"])}
    ev = collections.Counter()
    rules = collections.Counter()
    rule_n = collections.Counter()
    for p in parts:
        ev[p["evidence"]] += p["weight"]
        rules[p["rule"]] += p["weight"]
        rule_n[p["rule"]] += p["divergences"]
    out = {"summary": {
        "total_weight": round(total, 3), "architectural_weight": round(arch, 3),
        "arch_share": round(arch / total, 4) if total else 0.0,
        "assigned_share": round(1 - unrow["weight"] / total, 4) if total else 0.0,
        "unassigned_weight": unrow["weight"], "unassigned_share": unrow["share"],
        "unassigned_arch_share_of_arch": unrow["arch_share_of_arch"],
        "top5_arch": t5, "verdict": verdict(t5["share_of_arch"]), "readings": readings,
        "top5_arch_upper_bound": round(min(1.0, (t5["arch_weight"] + unrow["arch_weight"]) / arch), 4) if arch else 0.0,
        "evidence_weight_share": {e: round(ev[e] / total, 4) for e in EVIDENCE if total and ev.get(e)},
        "families_with_weight": len(rows)},
        "families": rows, "coarse": coarse, "unassigned": unrow,
        "rules": [{"rule": r[0], "matcher": r[1], "family": r[2], "coarse": r[3], "evidence": r[4], "reason": r[5],
                   "weight": round(rules[r[0]], 3), "divergences": rule_n[r[0]]} for r in RULES]}
    if summary:
        out["summary"]["ranked_total_weight"] = summary.get("total_weight")
        out["summary"]["ranked_arch_share"] = summary.get("arch_share")
    if groups is not None:
        out["single_site"] = single_site(parts, groups)
        out["groups"] = group_assignments(parts, groups)
    return out


def group_assignments(parts: list[dict], groups: list[dict]) -> list[dict]:
    by = collections.defaultdict(list)
    for p in parts:
        by[p["group"]].append(p)
    res = []
    for g in groups:
        ps = by.get(g["id"], [])
        w = collections.Counter()
        for p in ps:
            w[(p["family"], p["rule"], p["evidence"])] += p["weight"]
        if not w:
            continue
        (fam, rule, ev), _ = max(w.items(), key=lambda kv: (kv[1], kv[0]))
        fams = sorted({k[0] for k in w})
        res.append({"id": g["id"], "rank": g.get("rank"), "tag": g.get("tag"), "class": g.get("class"), "weight": g.get("weight"),
                    "family": fam, "rule": rule, "evidence": ev, "mixed": len(fams) > 1, "families": fams,
                    "split": {f"{k[0]}:{k[1]}": round(v, 3) for k, v in sorted(w.items(), key=lambda kv: -kv[1])}})
    return res


def single_site(parts: list[dict], groups: list[dict]) -> list[dict]:
    ga = {g["id"]: g for g in group_assignments(parts, groups)}
    out = []
    for g in groups:
        a = ga.get(g["id"])
        if g.get("tag") == "single_site" and a:
            k = g.get("display") or {}
            out.append({"rank": g.get("rank"), "id": g["id"], "weight": g.get("weight"), "family": a["family"], "rule": a["rule"],
                        "evidence": a["evidence"], "type": k.get("type"), "kinds": k.get("kinds"), "divergences": g.get("divergences"),
                        "first_floor": g.get("first_floor"), "outcome_changing_confirmed": g.get("outcome_changing_confirmed")})
    out.sort(key=lambda r: (-(r["weight"] or 0), r["id"]))
    return out


def build(folder: Path | None, doc: dict) -> dict:
    """The by-family table of a ranked doc: per divergence from the records in folder when it has them, else per group from the
    samples in doc."""
    groups = doc.get("groups", [])
    parts = parts_from_records(folder, doc) if folder is not None and (Path(folder) / "divergences.jsonl").exists() else []
    basis = "divergences"
    if not parts:
        parts, basis = parts_from_ranked(doc), "group samples"
    out = table(parts, groups, doc.get("summary"))
    out["summary"]["basis"] = basis
    gw = collections.defaultdict(float)
    for p in parts:
        gw[p["group"]] += p["weight"]
    out["summary"]["weight_check_max_abs_error"] = round(max((abs(gw.get(g["id"], 0.0) - float(g.get("weight") or 0)) for g in groups), default=0.0), 3)
    return out


# ---------------------------------------------------------------- the report

def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def md(doc: dict) -> str:
    s = doc["summary"]
    t5 = s["top5_arch"]
    L = ["# Divergence by hook family", "",
         f"Basis: {s['basis']}. Total agent weight {s['total_weight']}; architectural share {_pct(s['arch_share'])}; unassigned "
         f"{_pct(s['unassigned_share'])} of the weight ({_pct(s['unassigned_arch_share_of_arch'])} of the architectural weight); "
         f"families with weight {s['families_with_weight']}. Evidence by weight: " +
         ", ".join(f"{k} {_pct(v)}" for k, v in s["evidence_weight_share"].items()) + ".", "",
         f"Top five families by architectural weight: {', '.join(t5['families'])} hold {_pct(t5['share_of_arch'])} of the "
         f"architectural weight: {s['verdict']} (at most {_pct(s['top5_arch_upper_bound'])} if every unassigned diff belonged to them). "
         f"Top five by total weight: {', '.join(t5['by_total_weight']['families'])} {_pct(t5['by_total_weight']['share_of_arch'])} "
         f"of the architectural weight. Only {s['families_with_weight']} families carry weight at all, so top five is every family "
         f"the lockstep diff can show.", "", "Other readings of a family boundary:", ""] + [
         f"- {k}: {', '.join(v['families'])} hold {_pct(v['share_of_arch'])}: {v['verdict']}. {v['why']}." for k, v in s["readings"].items()] + ["",
         "The lockstep diff holds hand, enemy hp, hp, deck, gold, potions and relics only, so the families that act through powers, "
         "relic counters, energy, orbs or potions in combat cannot show in this table, whatever their share of the real error.", "",
         "## Families by weight (cumulative curve)", "",
         "| # | family | name | coarse family | weight | share | cumulative | architectural weight | architectural share in family | share of architectural | groups | divergences | decisions affected | outcome-changing confirmed |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(doc["families"], 1):
        L.append(f"| {i} | {r['family']} | {r['name']} | {r['coarse']} | {r['weight']} | {_pct(r['share'])} | {_pct(r['cum_share'])} | "
                 f"{r['arch_weight']} | {_pct(r['arch_share_within'])} | {_pct(r['arch_share_of_arch'])} | {r['groups']} | "
                 f"{r['divergences']} | {r['decisions_affected']} | {r['outcome_changing_confirmed']} |")
    u = doc["unassigned"]
    L += [f"| - | unassigned | no rule places it | - | {u['weight']} | {_pct(u['share'])} | - | {u['arch_weight']} | "
          f"{_pct(u['arch_share_within'])} | {_pct(u['arch_share_of_arch'])} | {u['groups']} | {u['divergences']} | "
          f"{u['decisions_affected']} | {u['outcome_changing_confirmed']} |", "",
          "Unassigned by rule (weight): " + ", ".join(f"{k} {v}" for k, v in u["by_rule"].items()), "",
          "## Coarse families (PLAN 8.2 names; the last two added)", "",
          "| coarse family | weight | share | architectural weight | architectural share in family | groups | divergences |",
          "|---|---|---|---|---|---|---|"]
    for r in doc["coarse"]:
        L.append(f"| {r['coarse']} | {r['weight']} | {_pct(r['share'])} | {r['arch_weight']} | {_pct(r['arch_share_within'])} | "
                 f"{r['groups']} | {r['divergences']} |")
    L += ["", "## Single-site groups (Lane B's fix queue, ranked)", "",
          "| rank | group | weight | family | rule | evidence | type: fields | divergences | first floor | outcome-changing confirmed |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in doc.get("single_site", [])[:30]:
        L.append(f"| {r['rank']} | {r['id']} | {r['weight']} | {r['family']} | {r['rule']} | {r['evidence']} | "
                 f"{r['type']}: {', '.join(r['kinds'] or []) or '-'} | {r['divergences']} | {r['first_floor']} | {r['outcome_changing_confirmed']} |")
    L += ["", "## Rules (first match wins)", "", "| rule | family | coarse | evidence | weight | divergences | reason |", "|---|---|---|---|---|---|---|"]
    for r in doc["rules"]:
        L.append(f"| {r['rule']} | {r['family']} | {r['coarse'] or '-'} | {r['evidence']} | {r['weight']} | {r['divergences']} | {r['reason']} |")
    mixed = [g for g in doc.get("groups", []) if g["mixed"]]
    L += ["", f"Groups whose divergences fall into more than one family: {len(mixed)} (listed with their split in by_family.json). "
          f"Weight check against ranked.json: largest per-group difference {s['weight_check_max_abs_error']}.", ""]
    return "\n".join(L)


def write(folder: Path | None, out: Path, doc: dict) -> dict:
    res = build(folder, doc)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "by_family.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "by_family.md").write_text(md(res), encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = write(a.folder, a.out, rank.rank(a.folder))
    print(json.dumps(res["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
