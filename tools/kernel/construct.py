"""Construction checks of the Tier 1 kernel (docs/rebuild/TIER1_KERNEL.md sections 2 and 7).

    python -m tools.kernel.construct shape [--decompiled DIR]

``shape``: tools/kernel/hooks.json against a fresh extraction from the local decompile, both ways (a dispatcher,
iterator, pass, op, changed criterion, run-time flag, context, wait point, named step or partner on one side only
fails, printed per dispatcher), and the generated Core2/HookId.g.cs and Hooks.g.cs against a fresh generation from
hooks.json. Exit 0 only when both hold.

Modes are a table (``MODES``); later rounds add ``commands``, ``pauses`` and ``rng``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.fidelity import hook_table as ht
from tools.kernel import gen_hooks, hooks_extract as hx

FIELDS = ("iterator", "context", "wait_all", "pre", "post", "partner")
PASS_FIELDS = ("hook", "op", "changed", "when")


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


# mode -> (handler, help). Later rounds: commands, pauses, rng.
MODES = {
    "shape": (shape, "hooks.json against the decompile both ways, and the generated hook code against hooks.json"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name, (_, help_) in MODES.items():
        p = sub.add_parser(name, help=help_)
        p.add_argument("--decompiled", default=ht.DEFAULT_DECOMPILED)
        p.add_argument("--descriptor", default=str(hx.DESCRIPTOR))
        p.add_argument("--engine", default=str(gen_hooks.ENGINE), help="src/Sts2Emulator (where the .g.cs files are)")
    a = ap.parse_args(argv)
    return MODES[a.mode][0](a)


if __name__ == "__main__":
    sys.exit(main())
