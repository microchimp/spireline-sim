"""The game's listener order against ``hook_table.game_order_key`` (rebuild-plan P0 D2).

    python -m tools.kernel.listener_check RECORDING [RECORDING ...] [--json]

A recording is a referee result file (``{"results": [...]}``, a list of results, one result, or JSONL of
results) of fights run with ``"trace": true, "trace_listeners": true`` (referee/driver/ListenerTrace.cs). Each step's
``trace`` holds, per listener enumeration the game made, ``["LB", id, hook, level]``, one
``["L", id, model, kind, owner, index]`` per listener in the order the game yielded it, and ``["LE", id]``;
enumerations nest, ``id`` tells them apart (it counts per fight, so a result is checked on its own).

For every enumeration the check builds the listener dict ``game_order_key`` takes from the L record's metadata and
tests that the game's yielded order is non-decreasing under the key. Reported per hook family
(``hook_families.json``, by the dispatcher the enumeration is attributed to): dispatches (enumerations),
agreeing, disagreeing with the first disagreeing pair, and listeners the rule cannot place.

Placing: the creature is ``slot`` for a player-side owner and ``1000 + slot`` for an enemy (allies come first, as in the
game), the index the record's. A listener the rule has no key for is *unplaced* and left out of the comparison:
a kind outside the rule, a card in a pile it has no rank for, an owner-less creature kind. The deck cards a run
enumeration yields first (pile ``deck``) are unplaced for the same reason (``game_order_key`` has no deck
pile), but the check still holds them to the rule's documented head position: all of them must come before every placed
listener, in deck order, else the enumeration disagrees.

Attribution: the dispatcher is the enumeration's ``hook`` (the innermost Hook method on the call stack when it was
created). A dispatcher that is not itself a model hook takes the family of the model hooks it fronts (``family_of``). An
enumeration the game makes outside ``Hook`` (``?Caller.Method``) goes to the family in ``CALLER_FAMILY`` by its caller, else to ``"-"``. A record without a hook (older format) takes the last ``H`` event before it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.fidelity import hook_table as ht  # noqa: E402

FAMILIES_JSON = Path(__file__).resolve().parents[1] / "fidelity" / "hook_families.json"
# Direct IterateHookListeners callers in the game that are not Hook methods, and the family their hooks sit in.
CALLER_FAMILY = {"CardCostHelper.": "F07"}
ENEMY_BASE = 1000
CREATURE_KINDS = ("power", "monster", "relic", "potion", "orb", "card", "affliction", "enchantment")


def load_families(path: Path = FAMILIES_JSON) -> tuple[dict[str, str], dict[str, str]]:
    """(hook name -> family id, family id -> name)."""
    d = json.loads(path.read_text(encoding="utf-8"))
    return dict(d["hooks"]), {k: v["name"] for k, v in d["families"].items()}


def iter_results(obj):
    """The fight results in a loaded recording (see the module doc)."""
    if isinstance(obj, dict) and "results" in obj:
        yield from obj["results"]
    elif isinstance(obj, list):
        for x in obj:
            yield from iter_results(x)
    elif isinstance(obj, dict) and "steps" in obj:
        yield obj


def load(path: Path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        return list(iter_results(json.loads(text)))
    except json.JSONDecodeError:
        return [r for line in text.splitlines() if line.strip() for r in iter_results(json.loads(line))]


def enumerations(events):
    """Yield ``{"id", "hook", "level", "listeners": [L records], "closed"}`` in creation order.

    Nested enumerations interleave in ``events``; the id sorts the records out. ``hook`` is the LB record's, else the
    last H event before it (None when there was none).
    """
    open_: dict[int, dict] = {}
    order: list[dict] = []
    last_h = None
    for e in events:
        if not e:
            continue
        tag = e[0]
        if tag == "H":
            last_h = e[1]
        elif tag == "LB":
            hook = e[2] if len(e) > 2 and e[2] is not None else last_h
            en = {"id": e[1], "hook": hook, "level": e[3] if len(e) > 3 else "combat", "listeners": [], "closed": False}
            open_[e[1]] = en
            order.append(en)
        elif tag == "L" and e[1] in open_:
            open_[e[1]]["listeners"].append(e)
        elif tag == "LE" and e[1] in open_:
            open_[e[1]]["closed"] = True
    yield from order


def listener_dict(rec) -> dict | None:
    """The dict ``game_order_key`` takes for an L record, or None when the rule cannot place it."""
    _, _, _model, kind, owner, index = rec[:6]
    if kind not in ht.KIND_RANK:
        return None
    d: dict = {"kind": kind}
    if kind in CREATURE_KINDS:
        if not owner:
            return None
        side, slot = owner
        d["creature"] = slot if side == "Player" else ENEMY_BASE + slot
    if kind in ("card", "affliction", "enchantment"):
        if not isinstance(index, list) or index[0] not in ht.PILE_RANK:
            return None
        d["pile"], d["index"] = index[0], index[1]
    elif kind in CREATURE_KINDS:
        if not isinstance(index, int):
            return None
        d["index"] = index
    return d


def is_deck(rec) -> bool:
    return rec[3] in ("card", "enchantment", "affliction") and isinstance(rec[5], list) and rec[5][0] == "deck"


def deck_key(rec) -> tuple:
    """The run enumeration's head: the deck's cards in deck order, each followed by its enchantment."""
    owner = rec[4] or ["Player", 0]
    return (owner[1], rec[5][1], 0 if rec[3] == "card" else 1)


def family_of(hook, hooks: dict[str, str]) -> str:
    """The family of a dispatcher. ``hook_families.json`` is keyed by the model hooks (``ModifyDamageAdditive``,
    ``TryModifyEnergyCostInCombat``); a ``Hook`` dispatcher that is not itself one of them (``ModifyDamage``,
    ``ModifyEnergyCostInCombat``) takes the family its ``Try``-prefixed name or the model hooks it is the prefix of
    share, and ``"-"`` when those disagree or there are none."""
    if hook is None:
        return "-"
    if hook.startswith("?"):
        caller = hook[1:]
        for prefix, fam in CALLER_FAMILY.items():
            if caller.startswith(prefix):
                return fam
        return "-"
    if hook in hooks:
        return hooks[hook]
    fams = {hooks[k] for k in hooks if k == "Try" + hook or k.startswith(hook)}
    return fams.pop() if len(fams) == 1 else "-"


def check_enumeration(en) -> dict:
    """``{"agree": bool, "placed": n, "unplaced": [(kind, pile)], "first": None | (a, b, why)}``."""
    keyed = []  # (key, rec)
    head = []   # deck listeners of a run enumeration, in yielded order
    unplaced = []
    for rec in en["listeners"]:
        d = listener_dict(rec)
        if d is not None:
            keyed.append((ht.game_order_key(d, 0), rec))
        else:
            unplaced.append((rec[3], rec[5][0] if isinstance(rec[5], list) else None))
            if is_deck(rec):
                head.append((len(keyed), rec))
    first = None
    for (ka, ra), (kb, rb) in zip(keyed, keyed[1:]):
        if ka > kb:
            first = (ra, rb, "key %r before %r" % (ka, kb))
            break
    if first is None and head:
        if any(pos > 0 for pos, _ in head):
            pos, rec = next((p, r) for p, r in head if p > 0)
            first = (keyed[pos - 1][1], rec, "deck card after a placed listener")
        else:
            for (_, ra), (_, rb) in zip(head, head[1:]):
                if deck_key(ra) > deck_key(rb):
                    first = (ra, rb, "deck order")
                    break
    return {"agree": first is None, "placed": len(keyed), "unplaced": unplaced, "first": first}


def check(results, hooks: dict[str, str] | None = None) -> dict:
    """Per family: ``{"dispatches", "agree", "disagree", "unplaced", "unplaced_kinds", "first"}``; ``"first"`` is
    ``{"result", "enum", "hook", "a", "b", "why"}`` for the first disagreeing enumeration of the family."""
    if hooks is None:
        hooks, _ = load_families()
    fam: dict[str, dict] = defaultdict(lambda: {"dispatches": 0, "agree": 0, "disagree": 0, "unplaced": 0,
                                                "unplaced_kinds": defaultdict(int), "first": None, "hooks": defaultdict(int)})
    for ri, res in enumerate(results):
        events = [e for s in res.get("steps", []) for e in (s.get("trace") or [])]
        for en in enumerations(events):
            f = fam[family_of(en["hook"], hooks)]
            r = check_enumeration(en)
            f["dispatches"] += 1
            f["hooks"][en["hook"] or "-"] += 1
            f["unplaced"] += len(r["unplaced"])
            for k in r["unplaced"]:
                f["unplaced_kinds"]["%s/%s" % k if k[1] else k[0]] += 1
            if r["agree"]:
                f["agree"] += 1
            else:
                f["disagree"] += 1
                if f["first"] is None:
                    a, b, why = r["first"]
                    f["first"] = {"result": res.get("seed", ri), "enum": en["id"], "hook": en["hook"], "a": a[2:], "b": b[2:], "why": why}
    return {k: {**v, "unplaced_kinds": dict(v["unplaced_kinds"]), "hooks": dict(v["hooks"])} for k, v in sorted(fam.items())}


def table(report: dict, names: dict[str, str]) -> str:
    rows = ["%-4s %-26s %10s %10s %9s %9s" % ("fam", "name", "dispatches", "agree", "disagree", "unplaced")]
    tot = defaultdict(int)
    for k, v in report.items():
        rows.append("%-4s %-26s %10d %10d %9d %9d" % (k, names.get(k, "(no family)")[:26], v["dispatches"], v["agree"], v["disagree"], v["unplaced"]))
        for c in ("dispatches", "agree", "disagree", "unplaced"):
            tot[c] += v[c]
    rows.append("%-4s %-26s %10d %10d %9d %9d" % ("", "total", tot["dispatches"], tot["agree"], tot["disagree"], tot["unplaced"]))
    for k, v in report.items():
        if v["first"]:
            f = v["first"]
            rows.append("first disagreement %s (%s, enum %s of %s): %s then %s -- %s" % (k, f["hook"], f["enum"], f["result"], f["a"], f["b"], f["why"]))
    kinds = defaultdict(int)
    for v in report.values():
        for k, n in v["unplaced_kinds"].items():
            kinds[k] += n
    if kinds:
        rows.append("unplaced by the rule: " + ", ".join("%s x%d" % kv for kv in sorted(kinds.items())))
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("recordings", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args(argv)
    results = [r for p in a.recordings for r in load(p)]
    hooks, names = load_families()
    report = check(results, hooks)
    if a.json:
        print(json.dumps(report, indent=1))
    else:
        print("%d fight(s)" % len(results))
        print(table(report, names))
    return 1 if any(v["disagree"] for v in report.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
