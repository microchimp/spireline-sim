"""Construction checks of the Tier 1 kernel (docs/rebuild/TIER1_KERNEL.md sections 2 and 7).

    python -m tools.kernel.construct shape [--decompiled DIR]
    python -m tools.kernel.construct commands [--decompiled DIR] [--list-writes]

``shape``: tools/kernel/hooks.json against a fresh extraction from the local decompile, both ways (a dispatcher,
iterator, pass, op, changed criterion, other part of a try condition, run-time flag, context, wait point, named step or partner on one side only
fails, printed per dispatcher), and the generated Core2/HookId.g.cs and Hooks.g.cs against a fresh generation from
hooks.json. Exit 0 only when both hold.

``commands`` (tools/kernel/commands.py): each migrated Core2 command's declared call sequence against the game method's
(hook calls and named callees with branch labels) and against the Core2 method's calls, both ways; and the HP/block
write scan: a write in Core2 outside Core2/Commands and the OldState adapter fails; the old engine's writes that run with
the F08+F10 switch on are listed (R4 reroutes them; ``--list-writes`` prints each). Exit 0 only when every command
matches and Core2 has no stray write.

Modes are a table (``MODES``); later rounds add ``pauses`` and ``rng``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.fidelity import hook_table as ht
from tools.kernel import gen_hooks, hooks_extract as hx

FIELDS = ("iterator", "context", "wait_all", "pre", "post", "partner")
PASS_FIELDS = ("hook", "op", "changed", "when", "also")


def _fmt(v) -> str:
    if isinstance(v, list) and v and isinstance(v[0], dict):
        return "[" + ", ".join(x["step"] + (f" when {x['when']}" if "when" in x else "") for x in v) + "]"
    return repr(v)


def diff(desc: dict, fresh: dict) -> list[str]:
    """Every structural difference between the checked-in descriptor and a fresh extraction (args are ignored)."""
    out: list[str] = []
    a, b = desc["dispatchers"], fresh["dispatchers"]
    if desc.get("format") != fresh.get("format"):
        out.append(f"descriptor format {desc.get('format')!r}, extractor format {fresh.get('format')!r}")
    for n in sorted(set(a) | set(b)):
        if n not in b:
            out.append(f"{n}: dispatcher only in the descriptor")
            continue
        if n not in a:
            out.append(f"{n}: dispatcher only in the decompile")
            continue
        x, y = a[n], b[n]
        for f in FIELDS:
            if x.get(f) != y.get(f):
                out.append(f"{n}: {f}: descriptor {_fmt(x.get(f))}, decompile {_fmt(y.get(f))}")
        px, py = x.get("passes", []), y.get("passes", [])
        for i in range(max(len(px), len(py))):
            if i >= len(px):
                out.append(f"{n}: pass {i} ({py[i]['hook']}) only in the decompile")
            elif i >= len(py):
                out.append(f"{n}: pass {i} ({px[i]['hook']}) only in the descriptor")
            else:
                for f in PASS_FIELDS:
                    if px[i].get(f) != py[i].get(f):
                        out.append(f"{n}: pass {i} {f}: descriptor {px[i].get(f)!r}, decompile {py[i].get(f)!r}")
    return out


def generated(desc: dict, root: Path = gen_hooks.ENGINE) -> list[str]:
    """The generated files that do not equal a fresh generation from the descriptor."""
    try:
        return gen_hooks.stale(desc, root)
    except gen_hooks.GenError as e:
        return [f"generator: {e}"]


def shape(a) -> int:
    desc = hx.load(Path(a.descriptor))
    try:
        fresh = hx.extract(a.decompiled)
    except hx.ExtractError as e:
        print(f"shape: FAIL: the extractor cannot classify {e}")
        return 1
    problems = diff(desc, fresh) + generated(desc, Path(a.engine))
    for p in problems:
        print(f"  {p}")
    n = len(desc["dispatchers"])
    print(f"shape: {'FAIL' if problems else 'ok'}: {n} dispatchers in {len(hx.shape_counts(desc))} shapes, "
          f"{len(problems)} differences")
    return 1 if problems else 0


def commands(a) -> int:
    from tools.kernel import commands as cm

    problems: list[str] = []
    for name, cmd in cm.COMMANDS.items():
        p = cm.check_command(name, cmd, Path(a.decompiled), Path(a.engine))
        print(f"  {name}: {'ok' if not p else f'{len(p)} differences'}")
        problems += p
    for p in problems:
        print(f"  {p}")
    stray = cm.core2_writes(Path(a.engine))
    for w in stray:
        print(f"  Core2 writes {w.field} outside Core2/Commands and OldState: {w.file}:{w.line}")
    old = cm.old_writes(Path(a.engine))
    if getattr(a, "list_writes", False):
        for w in old:
            print(f"  R4 {w.family} {w.file}:{w.line} {w.method} {w.field}: {w.text}")
    fam = {f: sum(1 for w in old if w.family == f) for f in ("F08", "F09")}
    files = len({w.file for w in old})
    bad = bool(problems or stray)
    print(f"commands: {'FAIL' if bad else 'ok'}: {len(cm.COMMANDS)} commands, {len(problems)} differences; "
          f"{len(stray)} Core2 writes outside Commands; old HP/block writes with the switch on (R4's list): {len(old)} "
          f"({fam['F08']} F08 HP and absorbed block, {fam['F09']} F09 block) in {files} files")
    return 1 if bad else 0


# mode -> (handler, help). Later rounds: pauses, rng.
MODES = {
    "shape": (shape, "hooks.json against the decompile both ways, and the generated hook code against hooks.json"),
    "commands": (commands, "the migrated commands' call sequences against the game's and Core2's, and the write scan"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name, (_, help_) in MODES.items():
        p = sub.add_parser(name, help=help_)
        p.add_argument("--decompiled", default=ht.DEFAULT_DECOMPILED)
        p.add_argument("--descriptor", default=str(hx.DESCRIPTOR))
        p.add_argument("--engine", default=str(gen_hooks.ENGINE), help="src/Sts2Emulator (where the .g.cs files are)")
        if name == "commands":
            p.add_argument("--list-writes", action="store_true", help="print every old write R4 must reroute")
    a = ap.parse_args(argv)
    return MODES[a.mode][0](a)


if __name__ == "__main__":
    sys.exit(main())
