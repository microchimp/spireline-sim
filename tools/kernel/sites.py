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
one switch. A coarser group is always sound; a SPLITS record is what makes groups finer, and each kind is checked against
the source it was read from (below). Identifiers only.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from tools.fidelity import hook_table as ht  # noqa: E402
from tools.kernel.port_list import PSEUDO_HOOKS  # noqa: E402

DISPATCH = ROOT / "tools" / "fidelity" / "hook_dispatch.json"
FAMILIES = ROOT / "tools" / "fidelity" / "hook_families.json"
HOOKS = ROOT / "tools" / "kernel" / "hooks.json"
OUT = ROOT / "tools" / "kernel" / "sites.json"
CS = ROOT / "src" / "Sts2Emulator" / "Core2" / "Switches.g.cs"
FORMAT = 1

# Flows that are one switch whatever their sites say: the turn-start flow's pause timing (CombatManager's setup,
# AfterSideTurnStart and the orb queue's turn start; TIER1_KERNEL.md sections 1 and 5).
FLOWS = [{"families": ["F02", "F03", "F13"], "why": "turn-start flow: one pause timing (TIER1_KERNEL.md 1, 5)"}]

# Sites read and found not to bind their families together. site -> a record of one of three kinds, each with a reason:
#   split         the families' parts are separate statements of the site, so each part gates on its own switch:
#                 {"kind": "split", "file": "Core/X.cs", "sha256": <file hash when read>, "reason": ...,
#                  "parts": [{"family": "F06", "anchor": "text that starts the part"}, ...]}  (parts in source order)
#   by_caller     the site is a shared helper that serves one family per caller; the gate sits at the callers:
#                 {"kind": "by_caller", "file": ..., "sha256": ..., "reason": ...,
#                  "callers": [{"family": "F06", "caller": "Class.Method", "anchor": "text of the call"}, ...]}
#   false_listing a hook_dispatch.json entry that is not a dispatch there (the method names the hook, but does not run its
#                 moment): {"kind": "false_listing", "hooks": ["HookName"], "reason": ...}; the hooks leave the site, and a
#                 site with none left leaves the inventory.
# A split's anchors must be present and unique in the file, in the order of its parts; a by_caller's anchors present; each
# caller a method of the engine; the file's hash (text with \n line ends) the one recorded, so any edit of the file sends
# the record back to be read again. Verified when build is given a source reader (check does, via fresh()).
# Read for the F08 migration (docs/rebuild/MIGRATION_F08.md section 2, round R1): the two bridges that joined F08 and F10 to
# other families. The third one, RunRewardGenerator.GenerateCombatRewards (F10+F14+F15), needs no record: its F10 hooks
# (AfterCombatVictory, AfterCombatVictoryEarly) are family F17 now.
SPLITS: dict[str, dict] = {
    "CombatEngine.DoomThePlayerAtTurnEnd": {
        "kind": "split",
        "file": "Core/CombatEngine.cs",
        "sha256": "76ba7a633fb21f7d6a372baffeed2e32eba1fe1b53d3f093262188276f2c0065",
        "reason": "Doom's turn-end body is two statements: the threshold test on the Doom stack and the open gate "
                  "(the BeforeSideTurnEnd moment, F05), then the first writes of the player's death (HP to zero and the "
                  "HP-changed dispatch, F10; the kill and AfterDiedToDoom are not in this body); the F10 part gates on "
                  "its own switch (R4: the old two statements are the else branch of the F08 gate, whose switch-on "
                  "branch is CreatureCmd.Kill on the player)",
        "parts": [
            {"family": "F05", "anchor": "int doom = BuffSystem.Get(state.PlayerBuffs, BuffId.Doom);"},
            {"family": "F10", "anchor": "state.PlayerHp = 0;\n            Effects.RelicEffects.ApplyAfterPlayerHpChanged(state);"},
        ],
    },
    "RelicEffects.BiteForFurCoat": {
        "kind": "split",
        "file": "Core/Effects/RelicEffects.cs",
        "sha256": "a47f3b108ce11da4066f02415eda7f3fe86a04d6879cc4943772dbd4fe50eea3",
        "reason": "Fur Coat's AfterCreatureAddedToCombat body is the method as entered (F01: the bite of whatever "
                  "joins); inside it the hittable test and the one-HP write are F08 (ShouldAllowHitting, "
                  "SetCurrentHp), a separate statement that gates on its own switch",
        "parts": [
            {"family": "F01", "anchor": "public static void BiteForFurCoat(CombatState state, EnemyState enemy)"},
            {"family": "F08", "anchor": "state.FurCoatBites && EnemyAI.IsTargetable(enemy)"},
        ],
    },
}

SPLIT_KINDS = ("split", "by_caller", "false_listing")
PART_KINDS = ("split", "by_caller")  # the kinds under which a site no longer binds its families

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


def source_hash(text: str) -> str:
    """The hash a record keeps of its file: sha256 of the text with \\n line ends."""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def engine_source(rel: str) -> str | None:
    """The text of an engine file by the path the inventory names it with (relative to src/Sts2Emulator)."""
    p = ht.ENGINE / rel
    return p.read_text(encoding="utf-8") if p.is_file() else None


def record_parts(rec: dict) -> list[tuple[str | None, str | None]]:
    """(family, anchor) of each part of a split or by_caller record."""
    if rec.get("kind") == "split":
        return [(p.get("family"), p.get("anchor")) for p in rec.get("parts", [])]
    return [(c.get("family"), c.get("anchor")) for c in rec.get("callers", [])]


def check_record(site: str, rec: dict, listed: set[str], site_file: str | None,
                 read: Callable[[str], str | None] | None, known: set[str] | None) -> list[str]:
    """The problems of one SPLITS record. listed: the hooks hook_dispatch.json lists the site for (before false listings);
    site_file: where the engine has the site; read: the source text of a file (None: no source check); known: the engine's
    method names (None: not checked)."""
    kind = rec.get("kind")
    if kind not in SPLIT_KINDS:
        return [f"{site}: unknown record kind {kind!r} (one of {', '.join(SPLIT_KINDS)})"]
    out = []
    if not str(rec.get("reason", "")).strip():
        out.append(f"{site}: {kind} record has no reason")
    if kind == "false_listing":
        if not rec.get("hooks"):
            out.append(f"false listing {site}: names no hook")
        for h in rec.get("hooks", []):
            if h not in listed:
                out.append(f"false listing {site}: hook_dispatch.json does not list {site} for {h}")
        return out
    parts = record_parts(rec)
    fams = [f for f, _ in parts]
    if kind == "split" and len(parts) < 2:
        out.append(f"split {site}: a split has at least two parts")
    if len(set(fams)) != len(fams) or any(not f for f in fams):
        out.append(f"{kind} {site}: a family is missing or named twice")
    for fam, anchor in parts:
        if not anchor:
            out.append(f"{kind} {site}: no anchor for {fam}")
    if kind == "by_caller":
        for c in rec.get("callers", []):
            if not c.get("caller"):
                out.append(f"by_caller {site}: a caller of {c.get('family')} is not named")
            elif known is not None and c["caller"] not in known:
                out.append(f"STALE caller {c['caller']} of {site}: not a method of the engine")
    if read is None:
        return out
    file = rec.get("file")
    text = read(file) if file else None
    if text is None:
        return out + [f"{kind} {site}: file {file!r} cannot be read"]
    text = text.replace("\r\n", "\n")
    if kind == "split" and site_file and file != site_file:
        out.append(f"split {site}: the record is for {file}, the site is in {site_file}")
    if rec.get("sha256") != source_hash(text):
        out.append(f"{kind} {site}: {file} changed since it was read (hash {source_hash(text)[:12]}, recorded "
                   f"{str(rec.get('sha256'))[:12]}): read the parts again and record the new hash")
    last = -1
    for fam, anchor in parts:
        if not anchor:
            continue
        n = text.count(anchor)
        if n == 0:
            out.append(f"{kind} {site}: anchor of {fam} is not in {file}: {anchor!r}")
        elif kind == "split" and n > 1:
            out.append(f"split {site}: anchor of {fam} is not unique in {file} ({n} places): {anchor!r}")
        elif kind == "split":
            at = text.index(anchor)
            if at < last:
                out.append(f"split {site}: anchor of {fam} comes before the part before it in {file}: {anchor!r}")
            last = max(last, at)
    return out


def build(dispatch: dict, families: dict, hooks: dict, engine: ht.Engine | None,
          flows: list = FLOWS, splits: dict = SPLITS, read: Callable[[str], str | None] | None = None) -> dict:
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
    problems: list[str] = []
    listed = {n: set(s["hooks"]) for n, s in sites.items()}
    for name, rec in sorted(splits.items()):
        if rec.get("kind") != "false_listing":
            continue
        if name not in sites:
            problems.append(f"false listing {name}: not a site of the inventory")
            continue
        s = sites[name]
        s["hooks"] = [h for h in s["hooks"] if h not in rec.get("hooks", [])]
        if not s["hooks"]:
            del sites[name]
    spans = ht.region_spans(engine, dispatch) if engine else {}
    where = {}
    if engine:
        for rel, ms in engine.methods.items():
            for m in ms:
                where.setdefault(ht.qual(m), (rel, m.line))
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
        rec = splits.get(name)
        if rec and rec.get("kind") in PART_KINDS:
            named = [f for f, _ in record_parts(rec)]
            missing = [f for f in s["families"] if f not in named]
            if missing:
                problems.append(f"split site {name} names no part for {missing}")
            continue
        join(s["families"], f"site {name}")
    known = set(where) if engine else None
    for name, rec in sorted(splits.items()):
        if rec.get("kind") in PART_KINDS and name not in sites:
            problems.append(f"{rec.get('kind')} {name}: not a site of the inventory")
        problems += check_record(name, rec, listed.get(name, set()), sites.get(name, {}).get("file"), read, known)
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
        if inv.get("splits", {}).get(n, {}).get("kind") in PART_KINDS:
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
    fam_of = {**families["hooks"], **{h: v[0] for h, v in PSEUDO_HOOKS.items()}}
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
    inv = build(dispatch, families, hooks, engine, read=engine_source)
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
        every = {n: {"kind": "split", "reason": "probe", "parts": [{"family": f, "anchor": "?"} for f in fs]}
                 for n, fs in inv["bridges"].items()}
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
