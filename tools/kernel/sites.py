"""Site inventory and switch groups (docs/rebuild/TIER1_KERNEL.md section 1).

    python -m tools.kernel.sites build    # writes tools/kernel/sites.json and src/Sts2Emulator/Core2/Switches.g.cs
    python -m tools.kernel.sites check    # both equal a fresh build, every site resolves, the groups hold (exit 1 if not)
    python -m tools.kernel.sites groups   # prints the switch groups and the sites that join them

An old dispatch site is an engine method (or a region of a long one) that hand-dispatches some hook moments:
``tools/fidelity/hook_dispatch.json`` lists them per hook virtual ("methods", plus the methods its "generic" entries
name). The inventory maps each site to the hook virtuals it serves, the game dispatchers that call those virtuals
(``tools/kernel/hooks.json`` passes) and their families (``tools/fidelity/hook_families.json``).

A switch group is the smallest set of families such that no site, no dispatcher and no flow straddles two groups:
families that share an old site switch together (a site serving two families cannot run one family's part while the
kernel runs the other's, unless it is declared split below), a dispatcher's passes switch together, and each FLOW is
one switch. A coarser group is always sound; a SPLITS entry (a site whose per-family parts are separate statements, read
and recorded with the anchor of each part) is what makes groups finer. Identifiers only.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from tools.fidelity import hook_table as ht  # noqa: E402

DISPATCH = ROOT / "tools" / "fidelity" / "hook_dispatch.json"
FAMILIES = ROOT / "tools" / "fidelity" / "hook_families.json"
HOOKS = ROOT / "tools" / "kernel" / "hooks.json"
OUT = ROOT / "tools" / "kernel" / "sites.json"
CS = ROOT / "src" / "Sts2Emulator" / "Core2" / "Switches.g.cs"
FORMAT = 1

# Flows that are one switch whatever their sites say: the turn-start flow's pause timing (CombatManager's setup,
# AfterSideTurnStart and the orb queue's turn start; TIER1_KERNEL.md sections 1 and 5).
FLOWS = [{"families": ["F02", "F03", "F13"], "why": "turn-start flow: one pause timing (TIER1_KERNEL.md 1, 5)"}]

# Sites read and found to serve their families in separate statements, so each family's part can be gated on its own
# switch: site -> {family: the anchor text that starts its part}. Empty until a round reads them (K3 lists the bridges).
SPLITS: dict[str, dict[str, str]] = {}

_SITE = re.compile(r"^[A-Z]\w*\.[A-Z]\w*(?:\[\w+\])?$")


def generic_sites(reason: str) -> list[str]:
    """The methods a generic entry's reason names before its colon ("A.M + N[r]: ..." -> A.M, A.N[r])."""
    head = reason.split(":", 1)[0]
    out, cls = [], None
    for part in (p.strip() for p in head.split("+")):
        if _SITE.match(part):
            cls = part.split(".", 1)[0]
            out.append(part)
        elif cls and re.match(r"^[A-Z]\w*(?:\[\w+\])?$", part):
            out.append(f"{cls}.{part}")
    return out


def build(dispatch: dict, families: dict, hooks: dict, engine: ht.Engine | None,
          flows: list = FLOWS, splits: dict = SPLITS) -> dict:
    fam_of = families["hooks"]
    h2d = collections.defaultdict(list)
    for d, info in sorted(hooks["dispatchers"].items()):
        for p in info["passes"]:
            if d not in h2d[p["hook"]]:
                h2d[p["hook"]].append(d)
    sites: dict[str, dict] = {}

    def add(site: str, hook: str, generic: str | None) -> None:
        s = sites.setdefault(site, {"hooks": [], "generic": []})
        if hook not in s["hooks"]:
            s["hooks"].append(hook)
        if generic and generic not in s["generic"]:
            s["generic"].append(generic)

    for hook, spec in dispatch["hooks"].items():
        for m in spec.get("methods", []):
            add(m, hook, None)
        for model, reason in spec.get("generic", {}).items():
            for m in generic_sites(reason):
                add(m, hook, model)
    spans = ht.region_spans(engine, dispatch) if engine else {}
    where = {}
    if engine:
        for rel, ms in engine.methods.items():
            for m in ms:
                where.setdefault(ht.qual(m), (rel, m.line))
    problems: list[str] = []
    for name, s in sites.items():
        s["hooks"].sort()
        s["generic"].sort()
        s["dispatchers"] = sorted({d for h in s["hooks"] for d in h2d.get(h, [])})
        s["families"] = sorted({fam_of[h] for h in s["hooks"] if h in fam_of})
        if engine:
            if "[" in name:
                sp = spans.get(name)
                loc = (sp[0], engine.line_of(sp[0], sp[1])) if sp else None
            else:
                loc = where.get(name)
            if loc is None:
                problems.append(f"STALE site {name}: not a method or region of the engine")
            else:
                s["file"], s["line"] = loc
    for h in sorted({h for s in sites.values() for h in s["hooks"]}):
        if h not in fam_of:
            problems.append(f"hook {h} has an old site but no family (hook_families.json)")
    # Union-find over families.
    parent: dict[str, str] = {f: f for f in families["families"]}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    joined = collections.defaultdict(list)

    def join(fs: list[str], why: str) -> None:
        fs = sorted(set(fs))
        for f in fs[1:]:
            a, b = find(fs[0]), find(f)
            if a != b:
                parent[max(a, b)] = min(a, b)
        if len(fs) > 1:
            joined[tuple(fs)].append(why)

    for name, s in sorted(sites.items()):
        if name in splits:
            missing = [f for f in s["families"] if f not in splits[name]]
            if missing:
                problems.append(f"split site {name} names no part for {missing}")
            continue
        join(s["families"], f"site {name}")
    disp_fam = {}
    for d, info in sorted(hooks["dispatchers"].items()):
        fs = sorted({fam_of[p["hook"]] for p in info["passes"] if p["hook"] in fam_of})
        disp_fam[d] = fs
        join(fs, f"dispatcher {d}")
    for fl in flows:
        join(fl["families"], f"flow: {fl['why']}")
    groups = collections.defaultdict(list)
    for f in sorted(parent):
        groups[find(f)].append(f)
    glist = []
    for lead in sorted(groups):
        fs = groups[lead]
        why = sorted({w for k, ws in joined.items() if set(k) <= set(fs) for w in ws})
        glist.append({"name": "+".join(fs), "families": fs, "joined_by": why})
    with_site = sorted({d for s in sites.values() for d in s["dispatchers"]})
    no_dispatcher = sorted({h for s in sites.values() for h in s["hooks"] if h not in h2d})
    bridges = {n: s["families"] for n, s in sorted(sites.items()) if len(s["families"]) > 1}
    return {
        "_about": "Generated by tools/kernel/sites.py from hook_dispatch.json, hooks.json, hook_families.json and the "
                  "engine source; do not edit. Per old dispatch site: the hook virtuals it hand-dispatches, the game "
                  "dispatchers that call them and their families. groups: the switch groups (TIER1_KERNEL.md 1).",
        "format": FORMAT,
        "flows": flows,
        "splits": splits,
        "sites": {n: sites[n] for n in sorted(sites)},
        "dispatchers": {d: {"families": disp_fam[d],
                            "sites": sorted(n for n, s in sites.items() if d in s["dispatchers"])}
                        for d in with_site},
        "dispatchers_without_site": sorted(set(hooks["dispatchers"]) - set(with_site)),
        "hooks_without_dispatcher": no_dispatcher,
        "bridges": bridges,
        "groups": glist,
        "problems": problems,
    }


def check_groups(inv: dict) -> list[str]:
    """Every dispatcher with an old site is in exactly one group, every family in one group, no site straddles two."""
    problems = []
    group_of = collections.defaultdict(list)
    for i, g in enumerate(inv["groups"]):
        for f in g["families"]:
            group_of[f].append(i)
    for f, gs in group_of.items():
        if len(gs) != 1:
            problems.append(f"family {f} is in {len(gs)} groups")
    for d, info in inv["dispatchers"].items():
        gs = {g for f in info["families"] for g in group_of.get(f, [])}
        if len(gs) != 1:
            problems.append(f"dispatcher {d} ({info['families']}) is in {len(gs)} groups")
    for n, s in inv["sites"].items():
        if n in inv.get("splits", {}):
            continue
        gs = {g for f in s["families"] for g in group_of.get(f, [])}
        if len(gs) > 1:
            problems.append(f"site {n} ({s['families']}) straddles {len(gs)} groups")
    for fl in inv.get("flows", []):
        gs = {g for f in fl["families"] for g in group_of.get(f, [])}
        if len(gs) != 1 or any(f not in group_of for f in fl["families"]):
            problems.append(f"flow {fl['families']} is not in exactly one group ({len(gs)})")
    return problems


def gen_cs(inv: dict, families: dict, hookids: list[str], hooks: dict) -> str:
    fam_of = families["hooks"]
    fams = sorted(families["families"])
    lines = [
        "// <auto-generated>",
        "// Generated from tools/kernel/sites.json and tools/fidelity/hook_families.json by tools/kernel/sites.py; do not edit.",
        "// Regenerate: python -m tools.kernel.sites build. Checked by: python -m tools.kernel.sites check.",
        "// </auto-generated>",
        "namespace Sts2Emulator.Core2;",
        "",
        "/// <summary>A hook family (tools/fidelity/hook_families.json). <c>k.On(Switch.X)</c> is true when X's group is on.</summary>",
        "public enum Switch : byte",
        "{",
        "    None = 0,",
    ]
    for f in fams:
        lines.append(f"    /// <summary>{families['families'][f]['name']}</summary>")
        lines.append(f"    {f},")
    lines += ["}", "", "/// <summary>The switch groups of the site inventory (TIER1_KERNEL.md section 1) and each hook's family.</summary>",
              "public static class SwitchGroups", "{", "    /// <summary>Families that switch together, one array per group.</summary>",
              "    public static readonly Switch[][] Groups =", "    ["]
    for g in inv["groups"]:
        lines.append("        [" + ", ".join(f"Switch.{f}" for f in g["families"]) + "],")
    lines += ["    ];", "", "    /// <summary>The family of a hook virtual or dispatcher; None for one no family holds.</summary>",
              "    public static Switch FamilyOf(HookId hook) => hook switch", "    {"]
    disp = {d: sorted({fam_of[p["hook"]] for p in i["passes"] if p["hook"] in fam_of}) for d, i in hooks["dispatchers"].items()}
    for h in hookids:
        if h.startswith("Dispatch_"):
            fs = disp.get(h[len("Dispatch_"):], [])
            f = fs[0] if len(fs) == 1 else None
        else:
            f = fam_of.get(h)
        if f:
            lines.append(f"        HookId.{h} => Switch.{f},")
    lines += ["        _ => Switch.None,", "    };", "}", ""]
    return "\n".join(lines)


def hook_ids() -> list[str]:
    text = (ROOT / "src" / "Sts2Emulator" / "Core2" / "HookId.g.cs").read_text(encoding="utf-8")
    body = text.split("{", 1)[1].split("}", 1)[0]
    return [m.group(1) for m in re.finditer(r"^\s*(\w+)\s*(?:=\s*\d+)?,", body, re.M) if m.group(1) != "None"]


def fresh() -> tuple[dict, str]:
    dispatch, families, hooks = (ht.load_json(p) for p in (DISPATCH, FAMILIES, HOOKS))
    engine = ht.Engine.load(ht.ENGINE)
    inv = build(dispatch, families, hooks, engine)
    return inv, gen_cs(inv, families, hook_ids(), hooks)


def dump(inv: dict) -> str:
    return json.dumps(inv, indent=1, sort_keys=False) + "\n"


def show(inv: dict) -> None:
    for g in inv["groups"]:
        n = sum(1 for d in inv["dispatchers"].values() if set(d["families"]) <= set(g["families"]))
        print(f"group {g['name']}: {n} dispatchers with an old site; joined by {len(g['joined_by'])}")
    print(f"{len(inv['sites'])} sites, {len(inv['bridges'])} bridge sites (more than one family), "
          f"{len(inv['dispatchers'])} dispatchers with a site, {len(inv['dispatchers_without_site'])} without, "
          f"hooks with a site and no dispatcher: {inv['hooks_without_dispatcher']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["build", "check", "groups"])
    a = ap.parse_args(argv)
    inv, cs = fresh()
    if a.cmd == "build":
        OUT.write_text(dump(inv), encoding="utf-8", newline="\n")
        CS.write_text(cs, encoding="utf-8", newline="\n")
        show(inv)
        return 1 if inv["problems"] else 0
    if a.cmd == "groups":
        show(inv)
        for g in inv["groups"]:
            print(f"  {g['name']}")
            for w in g["joined_by"]:
                print(f"    {w}")
        # What the bridges cost: the groups if every bridge site were read and split.
        dispatch, families, hooks = (ht.load_json(x) for x in (DISPATCH, FAMILIES, HOOKS))
        every = {n: {f: "?" for f in fs} for n, fs in inv["bridges"].items()}
        finest = build(dispatch, families, hooks, None, splits=every)["groups"]
        print("with every bridge split: " + ", ".join(g["name"] for g in finest))
        for n, fs in inv["bridges"].items():
            print(f"  bridge {n}: {'+'.join(fs)}")
        return 0
    problems = list(inv["problems"]) + check_groups(inv)
    if not OUT.exists() or OUT.read_text(encoding="utf-8") != dump(inv):
        problems.append(f"{OUT.name} differs from a fresh build (python -m tools.kernel.sites build)")
    else:
        problems += check_groups(json.loads(OUT.read_text(encoding="utf-8")))
    if not CS.exists() or CS.read_text(encoding="utf-8") != cs:
        problems.append(f"{CS.name} differs from a fresh build")
    for p in problems:
        print(p)
    print(f"sites check: {'FAIL' if problems else 'ok'}: {len(inv['sites'])} sites, {len(inv['groups'])} groups, "
          f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
