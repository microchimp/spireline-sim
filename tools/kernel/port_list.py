"""Handler port list for a family migration (docs/rebuild/MIGRATION_F08.md).

    python -m tools.kernel.port_list --families F08 F10 [--exclude-hooks H ...] [--size 12] [--md out.md] [--json out.json]
                                     [--gate] [--emudump EmuDump.dll]

Every model the port manifest says is reached by records (corpus, agent, lab or Codex) that overrides a hook of the
given families, with: the family hooks it overrides, how the old engine stands for each (``hook_table.classify``:
registered at a dispatch site, not modelled with its reason, open, or missing), the old code locations (the dispatch
site and line where the old engine names the model for that hook, else the methods that name it elsewhere), its reach,
and the batch it is ported in. Batches are per kind (power, relic, card, monster, potion, other), ordered by reach,
``--size`` models each, so a worker ports one batch per round. Identifiers only: no game text.

Pseudo-hooks (``PSEUDO_HOOKS``) are the kill sequence's model calls that are not AbstractModel hooks (a MonsterModel
or PowerModel virtual, or the creature's ``Died`` event a monster subscribes to); they are listed like hooks of their
family so their models are ported in the same batches. ``--gate`` exits 1 while any listed (model, hook) pair has
no Core2 handler and is not ``not_modelled``: a registered pair (the old engine has a location for it) is unported until
its handler is in the HandlerTable. A handler counts when ``EmuDump kernel-handlers`` (the built EmuDump.dll, ``--emudump``)
lists it: the table with every switch on, not the source text; a handler marked ``[StubHandler]`` does not count. That is
the switch-on gate of the migration.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from tools.fidelity import hook_table as ht  # noqa: E402

FAMILIES = ROOT / "tools" / "fidelity" / "hook_families.json"
MANIFEST = Path(os.environ.get("SPIRELINE_REBUILD", r"D:\spireline\referee\rebuild")) / "manifest.json"
BATCH_KINDS = ("power", "relic", "card", "monster", "potion")
SOURCES = ("corpus_fights", "agent_runs", "lab_fights", "codex_runs")
DEFAULT_DLL = ROOT / "tools" / "emudump" / "bin" / "Release" / "net9.0" / "EmuDump.dll"
# The kill sequence's non-hook model calls: name -> (family, kind of the declaring model, how a model takes part).
PSEUDO_HOOKS = {
    "BeforeRemovedFromRoom": ("F10", "monster", "override"),        # MonsterModel virtual, CombatManager.RemoveCreature
    "ShouldPowerBeRemovedAfterOwnerDeath": ("F10", "power", "override"),  # PowerModel virtual, the kill's power sweep
    "AfterRemoved": ("F10", "power", "override"),                  # PowerModel virtual, after each removed power
    "Died": ("F10", "monster", "event"),                           # Creature.Died, raised before the removal query
}
# The one status that needs no Core2 handler: no state change to port, or multiplayer only (hook_unmodelled.json).
NO_HANDLER = "not_modelled"
# The statuses for which the old engine names the model at the hook: such a model is reachable whatever the manifest counts.
MODELLED = ("registered", "open")
# A value hook (Modify*, Should*) only reads its model; a location whose statement writes the model's buff is where the
# old engine applies the power (EnemyAI.TriggerSuck applying Strength), not where it folds it in.
BUFF_WRITE = re.compile(r"BuffSystem\.(?:Apply|Remove)\(|\bBuffs\[[^\]]*\]\s*[-+*]?=(?!=)")


def family_hooks(fam_doc: dict, families: list[str], exclude=()) -> dict[str, str]:
    """hook -> family, for the hooks of the given families (pseudo-hooks included), less ``exclude``."""
    out = {}
    for hook, v in fam_doc["hooks"].items():
        fam = v if isinstance(v, str) else v.get("family")
        if fam in families:
            out[hook] = fam
    for hook, (fam, _kind, _how) in PSEUDO_HOOKS.items():
        if fam in families:
            out[hook] = fam
    for hook in exclude:
        out.pop(hook, None)
    return out


EVENT_SUB = re.compile(r"\bCreature\.(\w+)\s*\+=")


def pseudo_overrides(decompiled: str) -> list[tuple[str, str, str]]:
    """(model key, kind, pseudo-hook) for every concrete model taking part in a pseudo-hook, inherited ones included."""
    classes = ht.model_classes(decompiled)
    root = Path(decompiled)
    own: dict[str, list[str]] = {}
    for key, c in classes.items():
        text = ht._read(root / c.rel)
        got = [h for h in dict.fromkeys(ht.OVERRIDE.findall(text)) if PSEUDO_HOOKS.get(h, ("", "", ""))[2] == "override"]
        got += [h for h in dict.fromkeys(EVENT_SUB.findall(text)) if PSEUDO_HOOKS.get(h, ("", "", ""))[2] == "event"]
        own[c.name] = got
    out = []
    for key, c in sorted(classes.items()):
        if c.abstract or c.name in ht.KIND_OF_BASE:
            continue
        kind = ht.kind_of(classes, c.base)
        cur, seen, done = c, set(), []
        while cur is not None and cur.name not in ht.KIND_OF_BASE and cur.name not in seen:
            seen.add(cur.name)
            done += [h for h in own.get(cur.name, []) if h not in done]
            cur = classes.get(cur.base)
        out += [(key, kind, h) for h in done if PSEUDO_HOOKS[h][1] == kind]
    return out


def pseudo_rows(decompiled: str, engine) -> list:
    """Rows for the pseudo-hooks: no old dispatch site exists, so a pair is open (the old engine names the model in
    the methods listed under elsewhere: each is read in its batch) or missing (named nowhere)."""
    rows = []
    for key, kind, hook in pseudo_overrides(decompiled):
        name = ht.short(key)
        refs = ht.references(engine, kind, name)
        elsewhere = sorted({ht.qual(mt) for r in refs if (mt := engine.method_at(r.file, r.offset)) is not None})
        rows.append(ht.Row(key, kind, hook, "open" if elsewhere else "missing", [], "", elsewhere))
    return rows


def drop_buff_writes(rows: list, engine) -> list:
    """A value hook's location whose statement writes the model's buff is an application, not the hook: moved from
    where to elsewhere (a row left with no location becomes open)."""
    out = []
    for r in rows:
        if not (r.hook.startswith("Modify") or r.hook.startswith("Should")) or not r.where or not r.sites:
            out.append(r)
            continue
        writes = set()
        for rel, off in r.sites:
            text = engine.masked[rel]
            if BUFF_WRITE.search(text[text.rfind(";", 0, off) + 1:text.find(";", off)]):
                mt = engine.method_at(rel, off)
                writes.add(f"{ht.qual(mt) if mt else ''}:{engine.line_of(rel, off)}")
        keep = [w for w in r.where if re.sub(r"\[[^\]]*\]", "", w) not in writes]
        moved = [w for w in r.where if w not in keep]
        if not moved:
            out.append(r)
            continue
        where = list(dict.fromkeys(keep))
        status = r.status if where else "open"
        out.append(ht.Row(r.model, r.kind, r.hook, status, where, r.reason,
                          sorted(set(r.elsewhere) | {m.rsplit(":", 1)[0] for m in moved}), r.sites))
    return out


class HandlerListError(RuntimeError):
    """The kernel's handler list could not be read. The gate fails on it: no list is not 'no handler ported'."""


def handler_pairs(doc: dict) -> set[tuple[str, str]]:
    """(game class, hook) of every handler in an ``EmuDump kernel-handlers`` document that is not a stub."""
    return {(h["model"], h["hook"]) for h in doc["handlers"] if not h.get("stub")}


def ported(dll: Path = DEFAULT_DLL, run=subprocess.run) -> set[tuple[str, str]]:
    """(game class, hook) pairs with a Core2 handler: the registrations of the kernel's HandlerTable with every switch on,
    from ``EmuDump kernel-handlers`` (Engines/Core2/KernelHandlers.cs), stubs left out. The source is never read, so a comment
    or a model file that does not register does not count."""
    dll = Path(dll)
    if not dll.is_file():
        raise HandlerListError(f"{dll} is not built (dotnet build tools/emudump/EmuDump.csproj -c Release)")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "handlers.json"
        try:
            r = run(["dotnet", str(dll), "kernel-handlers", str(out)], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise HandlerListError(f"EmuDump kernel-handlers did not run: {e}") from e
        if r.returncode != 0:
            raise HandlerListError(f"EmuDump kernel-handlers exited {r.returncode}: {(r.stderr or '').strip()[-300:]}")
        try:
            return handler_pairs(json.loads(out.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise HandlerListError(f"EmuDump kernel-handlers wrote no usable list: {e!r}") from e


def unported_pairs(items: list[dict], done: set[tuple[str, str]] = frozenset()) -> list[tuple[str, str, str]]:
    """(model, hook, status) of every listed pair that has no Core2 handler and is not ``not_modelled``. A registered pair
    counts: the old engine has a location for it, but the switch needs the handler in Core2's HandlerTable."""
    return [(i["model"], h["hook"], h["status"]) for i in items for h in i["hooks"]
            if h["status"] != NO_HANDLER and (i["class"], h["hook"]) not in done]


def reach(entry: dict) -> tuple[int, ...]:
    return tuple(int(entry.get(s, 0) or 0) for s in SOURCES)


def batch_kind(kind: str) -> str:
    return kind if kind in BATCH_KINDS else "other"


def build(rows: list, models: dict, hooks: dict[str, str], unreached: bool = False) -> dict:
    """The port list from classify rows, the manifest's models (key -> entry) and the families' hooks.

    A model is listed when its manifest entry overrides one of the hooks (combat_hooks), or a pseudo-hook row names it,
    and any source reaches it, or (``unreached``, ``--unreached``) the old engine names it at one of those hooks: the
    manifest counts a model's own records, so a power another model applies (Diamond Diadem's) or a relic no record holds
    yet reads 0 there while the old engine plays it. Rows
    are matched to manifest entries by (kind, class); a reached model with a family hook and no row (the decompile and
    the manifest disagree) is listed under ``unmatched``, never dropped."""
    by_class: dict[tuple[str, str], list] = {}
    for r in rows:
        if r.hook in hooks:
            by_class.setdefault((r.kind, ht.short(r.model)), []).append(r)
    items, unmatched = [], []
    for key, m in sorted(models.items()):
        pseudo = {r.hook for r in by_class.get((m["kind"], m["class"]), []) if r.hook in PSEUDO_HOOKS}
        mine = sorted({h for h in m.get("combat_hooks", []) if h in hooks} | pseudo)
        got = {r.hook: r for r in by_class.get((m["kind"], m["class"]), [])}
        if not mine or not (any(reach(m)) or unreached and any(got[h].status in MODELLED for h in mine if h in got)):
            continue
        if not got:
            unmatched.append(key)
        per_hook = []
        for h in mine:
            r = got.get(h)
            per_hook.append({
                "hook": h, "family": hooks[h], "pseudo": h in PSEUDO_HOOKS,
                "status": r.status if r else "unmatched",
                "where": list(r.where) if r else [],
                "elsewhere": list(r.elsewhere) if r else [],
                "reason": r.reason if r else "",
            })
        items.append({
            "model": key, "kind": m["kind"], "class": m["class"], "batch_kind": batch_kind(m["kind"]),
            "families": sorted({p["family"] for p in per_hook}),
            "reach": dict(zip(SOURCES, reach(m))), "hooks": per_hook,
        })
    return {"items": items, "unmatched": unmatched}


def weight(item: dict) -> tuple:
    r = item["reach"]
    return (-(r["corpus_fights"] + r["agent_runs"]), -r["lab_fights"], -r["codex_runs"], item["model"])


def batches(items: list[dict], size: int) -> list[dict]:
    """Per batch kind (BATCH_KINDS order, then other), models by reach, ``size`` per batch."""
    if size < 1:
        raise ValueError("batch size must be at least 1")
    out = []
    for kind in BATCH_KINDS + ("other",):
        mine = sorted((i for i in items if i["batch_kind"] == kind), key=weight)
        for n in range(0, len(mine), size):
            out.append({"id": f"{kind}-{n // size + 1}", "kind": kind, "models": [i["model"] for i in mine[n:n + size]]})
    return out


def summary(items: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i["batch_kind"]] = out.get(i["batch_kind"], 0) + 1
    return out


def render_md(doc: dict) -> str:
    items = {i["model"]: i for i in doc["items"]}
    L = [f"# Port list: {'+'.join(doc['families'])}", "",
         f"{len(items)} models reached by records override a hook of these families: "
         + ", ".join(f"{k} {v}" for k, v in doc["summary"].items()) + "."]
    if doc.get("excluded"):
        L.append(f"Hooks left out: {', '.join(doc['excluded'])}.")
    pairs = [h for i in doc["items"] for h in i["hooks"]]
    st: dict[str, int] = {}
    for h in pairs:
        st[h["status"]] = st.get(h["status"], 0) + 1
    L += [f"{len(pairs)} (model, hook) pairs ({sum(1 for h in pairs if h.get('pseudo'))} pseudo-hook): "
          + ", ".join(f"{k} {v}" for k, v in sorted(st.items())) + f"; gate: {len(doc.get('unported', []))} pairs without a Core2 handler.", ""]
    if doc["unmatched"]:
        L += [f"Unmatched (manifest model with no decompile row): {', '.join(doc['unmatched'])}", ""]
    for b in doc["batches"]:
        L += [f"## {b['id']}", "", "| model | reach c/a/l/x | hook | status | old location |", "|---|---|---|---|---|"]
        for key in b["models"]:
            i = items[key]
            r = i["reach"]
            rr = f"{r['corpus_fights']}/{r['agent_runs']}/{r['lab_fights']}/{r['codex_runs']}"
            for n, h in enumerate(i["hooks"]):
                loc = "; ".join(h["where"]) or ("elsewhere: " + ", ".join(h["elsewhere"][:4]) if h["elsewhere"] else "")
                if h["status"] == "not_modelled" and h["reason"]:
                    loc = (loc + "; " if loc else "") + "not modelled: " + h["reason"][:80]
                name = h["hook"] + (" (pseudo)" if h.get("pseudo") else "")
                L.append(f"| {key if n == 0 else ''} | {rr if n == 0 else ''} | {name} | {h['status']} | {loc} |")
        L.append("")
    return "\n".join(L) + "\n"


def make(families: list[str], size: int, manifest_path: Path, decompiled: str, exclude=(), dll: Path = DEFAULT_DLL,
         unreached: bool = False) -> dict:
    fam_doc = json.loads(FAMILIES.read_text(encoding="utf-8"))
    hooks = family_hooks(fam_doc, families, exclude)
    engine = ht.Engine.load(ht.ENGINE)
    open_keys = {k.split(" ", 1)[1] for k in ht.read_open() if k.startswith("missing ")}
    rows = ht.classify(decompiled, engine, ht.load_json(ht.DISPATCH), ht.load_json(ht.UNMODELLED), open_keys)
    rows = drop_buff_writes(rows, engine) + pseudo_rows(decompiled, engine)
    models = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["models"]
    got = build(rows, models, hooks, unreached=unreached)
    got["families"] = list(families)
    got["excluded"] = sorted(exclude)
    got["unported"] = [list(p) for p in unported_pairs(got["items"], ported(dll))]
    got["summary"] = summary(got["items"])
    got["batches"] = batches(got["items"], size)
    return got


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--size", type=int, default=12, help="models per batch")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--decompiled", default=ht.DEFAULT_DECOMPILED)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--exclude-hooks", nargs="*", default=[], help="hooks to leave out of the list")
    ap.add_argument("--emudump", type=Path, default=DEFAULT_DLL, help="the built EmuDump.dll whose kernel-handlers lists the handlers")
    ap.add_argument("--unreached", action="store_true", help="also list a model no record reaches that the old engine names "
                    "at a family hook (registered or open)")
    ap.add_argument("--gate", action="store_true", help="exit 1 while any pair has no Core2 handler and is not not_modelled")
    a = ap.parse_args(argv)
    try:
        doc = make(a.families, a.size, a.manifest, a.decompiled, a.exclude_hooks, a.emudump, unreached=a.unreached)
    except HandlerListError as e:
        sys.stderr.write(f"port_list: {e}\n")
        return 2
    if a.json:
        a.json.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    md = render_md(doc)
    if a.md:
        a.md.write_text(md, encoding="utf-8")
    else:
        sys.stdout.write(md)
    if a.gate and doc["unported"]:
        sys.stderr.write(f"gate: {len(doc['unported'])} (model, hook) pairs without a Core2 handler\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
