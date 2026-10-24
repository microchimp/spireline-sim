"""Handler port list for a family migration (docs/rebuild/MIGRATION_F08.md).

    python -m tools.kernel.port_list --families F08 F10 [--size 12] [--md out.md] [--json out.json]

Every model the port manifest says is reached by records (corpus, agent, lab or Codex) that overrides a hook of the
given families, with: the family hooks it overrides, how the old engine stands for each (``hook_table.classify``:
registered at a dispatch site, not modelled with its reason, open, or missing), the old code locations (the dispatch
site and line where the old engine names the model for that hook, else the methods that name it elsewhere), its reach,
and the batch it is ported in. Batches are per kind (power, relic, card, monster, potion, other), ordered by reach,
``--size`` models each, so a worker ports one batch per round. Identifiers only: no game text.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from tools.fidelity import hook_table as ht  # noqa: E402

FAMILIES = ROOT / "tools" / "fidelity" / "hook_families.json"
MANIFEST = Path(os.environ.get("SPIRELINE_REBUILD", r"D:\spireline\referee\rebuild")) / "manifest.json"
BATCH_KINDS = ("power", "relic", "card", "monster", "potion")
SOURCES = ("corpus_fights", "agent_runs", "lab_fights", "codex_runs")


def family_hooks(fam_doc: dict, families: list[str]) -> dict[str, str]:
    """hook -> family, for the hooks of the given families."""
    out = {}
    for hook, v in fam_doc["hooks"].items():
        fam = v if isinstance(v, str) else v.get("family")
        if fam in families:
            out[hook] = fam
    return out


def reach(entry: dict) -> tuple[int, ...]:
    return tuple(int(entry.get(s, 0) or 0) for s in SOURCES)


def batch_kind(kind: str) -> str:
    return kind if kind in BATCH_KINDS else "other"


def build(rows: list, models: dict, hooks: dict[str, str]) -> dict:
    """The port list from classify rows, the manifest's models (key -> entry) and the families' hooks.

    A model is listed when its manifest entry overrides one of the hooks (combat_hooks) and any source reaches it. Rows
    are matched to manifest entries by (kind, class); a reached model with a family hook and no row (the decompile and
    the manifest disagree) is listed under ``unmatched``, never dropped."""
    by_class: dict[tuple[str, str], list] = {}
    for r in rows:
        if r.hook in hooks:
            by_class.setdefault((r.kind, ht.short(r.model)), []).append(r)
    items, unmatched = [], []
    for key, m in sorted(models.items()):
        mine = sorted(h for h in m.get("combat_hooks", []) if h in hooks)
        if not mine or not any(reach(m)):
            continue
        got = {r.hook: r for r in by_class.get((m["kind"], m["class"]), [])}
        if not got:
            unmatched.append(key)
        per_hook = []
        for h in mine:
            r = got.get(h)
            per_hook.append({
                "hook": h, "family": hooks[h],
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
         + ", ".join(f"{k} {v}" for k, v in doc["summary"].items()) + ".", ""]
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
                L.append(f"| {key if n == 0 else ''} | {rr if n == 0 else ''} | {h['hook']} | {h['status']} | {loc} |")
        L.append("")
    return "\n".join(L) + "\n"


def make(families: list[str], size: int, manifest_path: Path, decompiled: str) -> dict:
    fam_doc = json.loads(FAMILIES.read_text(encoding="utf-8"))
    hooks = family_hooks(fam_doc, families)
    engine = ht.Engine.load(ht.ENGINE)
    open_keys = {k.split(" ", 1)[1] for k in ht.read_open() if k.startswith("missing ")}
    rows = ht.classify(decompiled, engine, ht.load_json(ht.DISPATCH), ht.load_json(ht.UNMODELLED), open_keys)
    models = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["models"]
    got = build(rows, models, hooks)
    got["families"] = list(families)
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
    a = ap.parse_args(argv)
    doc = make(a.families, a.size, a.manifest, a.decompiled)
    if a.json:
        a.json.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    md = render_md(doc)
    if a.md:
        a.md.write_text(md, encoding="utf-8")
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
