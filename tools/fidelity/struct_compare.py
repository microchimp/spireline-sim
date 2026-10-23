"""One structural comparer: a fight's record (the game's own shapes) against an engine's game view, step by step.

    python -m tools.fidelity.struct_compare <referee out.json> <EmuDump fight --game-view json> [--seed S] [--json OUT]

A record holds two shapes of the game's state: the referee's per-step snapshot (referee/driver/Combat.cs Snapshot) and,
in a traced fight, the game's NetFullCombatState after each checksum (step["net"]). `EmuDump fight --game-view` writes
the engine's step in the same two shapes with the game's names (IEngine.GameView: step["game_view"] = {snapshot, net,
private}). Nothing is renamed or projected here: both sides are flattened by the same structural rules and compared leaf
by leaf. The only exceptions are the reviewed representation-only rows of allowed_diff.json (KINDS below), each with
its reason and the legacy rule it stands for. Scope: combat (corpus fights and traced fights); whole runs
(run_compare2) are the run layer and not judged here.

Structural rules (facts of the game's schema, the same for every engine):
- dicts by key; lists by index with ``path#len``, a list of scalars also as one leaf; an empty list is its ``#len``;
- ``powers`` ([[id, amount], ...]) keyed by id: ``path{STRENGTH_POWER}`` (a creature's powers are a set by id);
- ``keywords`` a set; a card object of the net shape omits default fields (CARD_DEFAULTS);
- a path one side lacks: an entry of a collection the other side has (an index past its end, a power id it lacks) is a
  difference; a field the record has and the engine does not write is a difference unless a row lists it as
  unmodelled; a field only the engine writes is unjudged (the record does not say).

Per fight: the first differing step and its paths, the allowed rows that fired, and the field census: the schema paths
(indices [*], keys {*}) compared anywhere in the fight. census_gate() fails an engine compared on fewer fields than
another on the same fight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ALLOWED_FILE = Path(__file__).with_name("allowed_diff.json")

KEYED = {"powers"}
SETS = {"keywords"}
CARD_DEFAULTS = {"upgrades": 0, "keywords": [], "cost": None, "enchantment": None, "affliction": None}
_CARD = re.compile(r"^net\.players\[\d+\]\.piles\.\w+\[\d+\]$")
_ELEM = re.compile(r"\[\d+\]|\{[^}]*\}")
# The row kinds allowed_diff.json may use (docs/rebuild/COMPARER.md).
KINDS = {"drop_dead_enemies", "lost_outcome_only", "power_fold", "intent_coarse", "unmodelled", "ignore", "same_first",
         "both_positive", "settle", "late_steps"}
END_OF_TURN = "After player turn start"


def schema(path: str) -> str:
    return re.sub(r"\{[^}]*\}", "{*}", re.sub(r"\[\d+\]", "[*]", path))


def _glob(p: str):
    """A row pattern: matches the schema path and everything under it (its fields, entries and #len)."""
    return re.compile(re.escape(p).replace(r"\*\*", ".+").replace(r"\*", "[^.]*") + r"(?:[.\[{#].*)?\Z")


def load_rows(path: Path = ALLOWED_FILE) -> list:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["rows"]
    for r in rows:
        if r["kind"] not in KINDS:
            raise ValueError(f"allowed row {r['id']}: unknown kind {r['kind']}")
        if not r.get("reason") or not r.get("legacy"):
            raise ValueError(f"allowed row {r['id']}: a row needs its reason and its legacy rule")
        r["_re"] = [_glob(p) for p in r.get("patterns", [])]
    return rows


def rows_sha(path: Path = ALLOWED_FILE) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---- flattening -----------------------------------------------------------------------------------------------------

def flatten(x, path: str = "") -> tuple[dict, set]:
    """(leaves path -> value, node paths: every collection entry and keyed collection present)."""
    out: dict = {}
    nodes: set = set()

    def walk(v, p):
        if isinstance(v, dict):
            if _CARD.match(p):
                v = {**CARD_DEFAULTS, **v}
            for k in v:
                walk(v[k], f"{p}.{k}" if p else k)
            return
        if not isinstance(v, list):
            out[p] = v
            return
        last = p.rsplit(".", 1)[-1]
        if last in KEYED:
            nodes.add(p)
            by: dict = {}
            for name, amount in v:
                by.setdefault(name, []).append(amount)
            for name, amounts in by.items():
                q = f"{p}{{{name}}}"
                nodes.add(q)
                out[q] = amounts[0] if len(amounts) == 1 else sorted(amounts, key=json.dumps)
            return
        if last in SETS:
            out[p] = sorted(v)
            return
        out[f"{p}#len"] = len(v)
        if v and all(not isinstance(e, (dict, list)) for e in v):
            out[p] = list(v)
            return
        for i, e in enumerate(v):
            q = f"{p}[{i}]"
            nodes.add(q)
            if isinstance(e, list) and all(not isinstance(y, (dict, list)) for y in e):
                out[q] = list(e)
            else:
                walk(e, q)

    walk(x, path)
    return out, nodes


# ---- structural rows applied before flattening ----------------------------------------------------------------------

def _fold(powers: list, folds: dict) -> tuple[list, bool]:
    member = {m: base for base, ms in folds.items() for m in ms}
    out, summed, hit = [], {}, False
    for name, amount in powers:
        base = member.get(name, name if name in folds else None)
        if base is None:
            out.append([name, amount])
        else:
            hit = hit or name in member  # the base power alone folds nothing
            summed[base] = summed.get(base, 0) + amount
    return out + [[k, v] for k, v in summed.items() if v], hit


def _prefilter(state: dict | None, shape: str, rows: list, fired: dict, record: bool, rec_snap: dict) -> dict | None:
    if state is None:
        return None
    st = json.loads(json.dumps(state))
    for r in rows:
        if r.get("shape", "any") not in ("any", shape) or not _when(r, rec_snap):
            continue
        if r["kind"] == "drop_dead_enemies" and isinstance(st.get("enemies"), list):
            alive = [e for e in st["enemies"] if not (isinstance(e, dict) and e.get("hp", 1) <= 0)]
            if len(alive) != len(st["enemies"]):
                fired[r["id"]] = fired.get(r["id"], 0) + 1
            st["enemies"] = alive
        elif r["kind"] == "lost_outcome_only" and not rec_snap.get("in_combat") and (rec_snap.get("player") or {}).get("hp", 1) <= 0:
            fired[r["id"]] = fired.get(r["id"], 0) + record
            st = {"in_combat": st.get("in_combat"), "player": {"hp": (st.get("player") or {}).get("hp")}}
        elif r["kind"] == "power_fold":
            holders = [st.get("player")] + list(st.get("enemies") or []) + list(st.get("creatures") or [])
            for h in holders:
                if isinstance(h, dict) and isinstance(h.get("powers"), list):
                    h["powers"], hit = _fold(h["powers"], r["folds"])
                    if hit:
                        fired[r["id"]] = fired.get(r["id"], 0) + 1
    return st


def _when(row: dict, rec_snap: dict) -> bool:
    return all(rec_snap.get(k) == v for k, v in (row.get("when") or {}).items())


# ---- the game's own record: integrity and settling ------------------------------------------------------------------

def late_steps(fight: dict) -> list:
    """Steps whose action left no checksum of its own (recorded before the game finished them): netstate_diff.late_steps."""
    steps = fight.get("steps", [])
    if not any("checksums" in s for s in steps):
        return []
    out = []
    for k, s in enumerate(steps[1:], 1):
        a, kinds = s["action"] or {}, [c[1] or c[0] for c in s.get("checksums", [])]
        if (a.get("kind") == "play" and a.get("played") is not False and "PlayCardAction" not in kinds) or \
                (a.get("kind") == "end_turn" and s["state"].get("in_combat") and END_OF_TURN not in kinds):
            out.append(k)
    return out


def net_limit(fight: dict) -> int:
    """The first step whose net shape is not comparable (record integrity), or the step count."""
    late = late_steps(fight)
    if not late:
        return len(fight.get("steps", []))
    first = late[0]
    return first - 1 if (fight["steps"][first - 1]["action"] or {}).get("kind") == "end_turn" else first


_SNAP_PILES = {"Hand": "hand", "Draw": "draw", "Discard": "discard", "Exhaust": "exhaust"}
_SNAP_RNG = {"CombatOrbs": "CombatOrbGeneration"}


def moved(snap: dict, net: dict) -> set:
    """Net path prefixes whose value the record's own snapshot shows changed after the step's last checkpoint."""
    if not snap.get("in_combat"):
        return set()
    pairs = []
    alive = [e for e in snap.get("enemies", []) if e.get("hp", 0) > 0]
    k = 0
    for i, c in enumerate(net.get("creatures") or []):
        if c.get("player") is not None:
            s = snap.get("player") or {}
        elif c.get("monster"):
            s = alive[k] if k < len(alive) else {}
            k += 1
        else:
            continue
        for f in ("hp", "max_hp", "block"):
            pairs.append((f"net.creatures[{i}].{f}", c.get(f), s.get(f)))
        pairs.append((f"net.creatures[{i}].powers", sorted(map(list, c.get("powers") or [])), sorted(map(list, s.get("powers") or []))))
    if k != len(alive):
        pairs.append(("net.creatures", k, len(alive)))
    for p in (net.get("players") or [])[:1]:
        pairs.append(("net.players[0].energy", p.get("energy"), (snap.get("player") or {}).get("energy")))
        for name, key in _SNAP_PILES.items():
            cards = [[c.get("id"), (c.get("upgrades") or 0) > 0] for c in (p.get("piles") or {}).get(name, [])]
            pairs.append((f"net.players[0].piles.{name}", cards, [[c[0], bool(c[1])] for c in snap.get(key, [])]))
    for name, v in ((net.get("rng") or {}).get("counters") or {}).items():
        s = (snap.get("rng") or {}).get(_SNAP_RNG.get(name, name))
        if isinstance(s, dict):
            pairs.append((f"net.rng.counters.{name}", v, s.get("counter")))
    return {p for p, a, b in pairs if a != b}


# ---- one step -------------------------------------------------------------------------------------------------------

def _coarse(intents: list, classes: dict):
    if not intents:
        return ("NONE", None)
    kind = classes.get(intents[0][0], "OTHER") if intents[0] else "NONE"
    attack = next(([i[1], i[2]] for i in intents if isinstance(i, list) and len(i) >= 3), None)
    return (kind, attack)


def _intents(flat: dict, base: str) -> list:
    n = flat.get(f"{base}.intents#len", 0)
    return [flat.get(f"{base}.intents[{j}]") for j in range(n)]


def compare_shape(rec, eng, shape: str, rows: list, rec_snap: dict, fired: dict, moved_paths: set = frozenset()) -> dict:
    """One shape of one step: {diffs: [(path, record, engine)], compared, unmodelled, unjudged} (schema-path sets)."""
    g, gnodes = flatten(_prefilter(rec, shape, rows, fired, True, rec_snap), shape)
    e, enodes = flatten(_prefilter(eng, shape, rows, {}, False, rec_snap), shape)
    active = [r for r in rows if r.get("shape", "any") in ("any", shape) and _when(r, rec_snap)]
    unmod_rows = [r for r in active if r["kind"] == "unmodelled"]
    raw, compared, unmodelled, unjudged = [], set(), set(), set()
    for p in sorted(set(g) | set(e)):
        if p in g and p in e:
            compared.add(p)
            if g[p] != e[p]:
                raw.append((p, g[p], e[p]))
            continue
        have, other_nodes, other = (g, enodes, e) if p in g else (e, gnodes, g)
        missing_entry = False
        for m in _ELEM.finditer(p):
            node = p[:m.end()]
            container = p[:m.start()]
            if node not in other_nodes and (f"{container}#len" in other or container in other_nodes):
                missing_entry = True
                break
            if node not in other_nodes:
                break
        if missing_entry:
            compared.add(p)
            raw.append((p, g.get(p), e.get(p)))
        elif p in g:
            sp = schema(p)
            if any(rx.match(sp) for r in unmod_rows for rx in r["_re"]):
                fired_id = next(r["id"] for r in unmod_rows if any(rx.match(sp) for rx in r["_re"]))
                fired[fired_id] = fired.get(fired_id, 0) + 1
                unmodelled.add(sp)
            else:
                compared.add(p)
                raw.append((p, g[p], None))
        else:
            unjudged.add(schema(p))
    diffs = []
    coarse_ok: dict = {}
    for p, gv, ev in raw:
        sp = schema(p)
        why = None
        for r in active:
            k = r["kind"]
            if k in ("ignore", "same_first", "both_positive") and not any(rx.match(sp) for rx in r["_re"]):
                continue
            if k == "ignore" or (k == "same_first" and isinstance(gv, list) and isinstance(ev, list) and gv[:1] == ev[:1]) or \
                    (k == "both_positive" and isinstance(gv, int) and isinstance(ev, int) and gv > 0 and ev > 0):
                why = r["id"]
            elif k == "intent_coarse" and ".intents" in p:
                base = p[:p.index(".intents")]
                if base not in coarse_ok:
                    coarse_ok[base] = _coarse(_intents(g, base), r["classes"]) == _coarse(_intents(e, base), r["classes"])
                why = r["id"] if coarse_ok[base] else None
            elif k == "settle" and any(p == m or p.startswith(m + c) for m in moved_paths for c in ".[{#"):
                why = r["id"]
            if why:
                break
        if why:
            fired[why] = fired.get(why, 0) + 1
        else:
            diffs.append((p, gv, ev))
    return {"diffs": diffs, "compared": {schema(p) for p in compared}, "n": len(compared), "unmodelled": unmodelled,
            "unjudged": unjudged}


def compare_fight(g: dict, e: dict, rows: list | None = None) -> dict:
    """Every step of one fight in both shapes; see the module doc for the result."""
    rows = load_rows() if rows is None else rows
    gsteps, esteps = g.get("steps", []), e.get("steps", [])
    late = next((r for r in rows if r["kind"] == "late_steps"), None)
    limit = net_limit(g) if late else len(gsteps)  # the limit is the row: no row, no limit
    first, at, fired = None, [], {}
    census, unmodelled, unjudged, leaves = set(), set(), set(), 0
    for k, (gs, es) in enumerate(zip(gsteps, esteps)):
        view = es.get("game_view")
        if view is None:
            raise ValueError("an engine step without game_view: run EmuDump fight with --game-view")
        rec_snap = gs["state"]
        parts = [compare_shape(rec_snap, view.get("snapshot"), "snap", rows, rec_snap, fired)]
        nets = gs.get("net") or []
        if nets and rec_snap.get("in_combat") and k >= limit:
            fired[late["id"]] = fired.get(late["id"], 0) + 1  # record integrity: the net shape of this step is not judged
        elif nets and rec_snap.get("in_combat"):
            last = nets[-1]["state"]
            if view.get("net") is None:
                parts.append({"diffs": [("net", "(state)", None)], "compared": {"net"}, "n": 1, "unmodelled": set(), "unjudged": set()})
            else:
                parts.append(compare_shape(last, view["net"], "net", rows, rec_snap, fired, moved(rec_snap, last)))
        for part in parts:
            census |= part["compared"]
            unmodelled |= part["unmodelled"]
            unjudged |= part["unjudged"]
            leaves += part["n"]
        diffs = [d for part in parts for d in part["diffs"]]
        if diffs and first is None:
            first, at = k, diffs
    if first is None and len(esteps) < len(gsteps):
        first = len(esteps)
        at = [("(error)" if e.get("error") else "(length)", f"{len(gsteps)} steps", f"{len(esteps)} steps")]
    return {"seed": g.get("seed"), "first": first, "at": at, "allowed": dict(sorted(fired.items())), "census": sorted(census),
            "leaves": leaves, "unmodelled": sorted(unmodelled), "unjudged": sorted(unjudged - census),
            "steps": [len(gsteps), len(esteps)], **({"error": str(e["error"]).splitlines()[0][:200]} if e.get("error") else {})}


def judge(g: dict, e: dict, rows: list | None = None) -> dict:
    """One fight's entry in the corpus results format (corpus.judge_fight: fd, path, card, ferr) plus the census."""
    r = compare_fight(g, e, rows)
    out = {"fd": r["first"]}
    if r["first"] is not None:
        out["path"] = r["at"][0][0]
        out["fields"] = sorted({schema(p) for p, _, _ in r["at"]})
    if r.get("error"):
        out["ferr"] = r["error"]
    out.update(census=r["census"], leaves=r["leaves"], allowed=r["allowed"])
    return out


def census_gate(a: dict, b: dict) -> dict:
    """{item: schema paths engine A was compared on and engine B was not}, for every item where B has fewer: the gate
    fails when it is not empty. ``a`` and ``b`` map item ids to results of judge() or compare_fight()."""
    bad = {}
    for item, ra in a.items():
        rb = b.get(item)
        missing = sorted(set(ra["census"]) - set(rb["census"])) if rb is not None else ["(no result)"]
        if missing:
            bad[item] = missing
    return bad


def _short(v, n=70) -> str:
    s = json.dumps(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("referee")
    ap.add_argument("engine")
    ap.add_argument("--seed", action="append", default=[])
    ap.add_argument("--json", type=Path, help="write the per-fight results here")
    ap.add_argument("--allowed", type=Path, default=ALLOWED_FILE)
    a = ap.parse_args()
    rows = load_rows(a.allowed)
    games = [r for r in json.loads(Path(a.referee).read_text(encoding="utf-8"))["results"] if not a.seed or r.get("seed") in a.seed]
    engines = json.loads(Path(a.engine).read_text(encoding="utf-8"))["results"]
    out, bad = [], 0
    for k, g in enumerate(games):
        e = engines[k] if k < len(engines) else None
        if e is None or e.get("seed") not in (None, g.get("seed")):
            print(f"{g.get('seed')}: no engine result")
            bad += 1
            continue
        r = compare_fight(g, e, rows)
        out.append(r)
        bad += r["first"] is not None or "error" in r
        if r["first"] is None:
            print(f"{r['seed']}: {r['steps'][0]} steps identical on {len(r['census'])} fields ({r['leaves']} leaves)")
        else:
            print(f"{r['seed']}: first difference at step {r['first']}")
            for p, x, y in r["at"][:12]:
                print(f"    {p}: game {_short(x)}  engine {_short(y)}")
    if a.json:
        a.json.write_text(json.dumps({"comparer": "struct_compare", "allowed_sha": rows_sha(a.allowed), "results": out}), encoding="utf-8")
    print(f"{len(games)} fights, {bad} not reproduced")
    return 0 if games and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
