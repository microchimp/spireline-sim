"""Hook descriptor extractor: the structure of every public dispatcher of the game's ``Hook`` class.

    python -m tools.kernel.hooks_extract [--decompiled DIR] [--write]

Reads the local decompile (``MegaCrit.Sts2.Core.Hooks/Hook.cs`` and, for ``partner``, the callers of the
receiving dispatchers) and returns, per dispatcher, every field of docs/rebuild/TIER1_KERNEL.md section 2 except
``args``: ``iterator``, ``passes`` (hook virtual, ``op``, ``changed``, ``when``, ``also``), ``context``, ``wait_all``,
``pre`` / ``post`` named steps and ``partner``. It extends ``tools.fidelity.hook_table.dispatchers`` (the pass
list, which it cross-checks) with the per-pass classification.

Identifiers and our own structural words only: no game text is copied into the result. A dispatcher whose
shape is not one of the known ones raises ``ExtractError`` naming it; nothing is defaulted.

``--write`` rewrites ``tools/kernel/hooks.json`` with a fresh extraction, keeping each dispatcher's hand-written
``args``; it refuses when a dispatcher has none. Without it, the shape counts are printed.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from functools import cache
from pathlib import Path

from tools.fidelity import hook_table as ht

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DESCRIPTOR = HERE / "hooks.json"
HOOK_FILE = Path("MegaCrit.Sts2.Core.Hooks") / "Hook.cs"
FORMAT = 1

OPS = ("event", "fold-add", "fold-mul", "fold-cap", "fold-set", "try", "predicate-all", "predicate-any",
       "filter-set", "preventer-only", "mutate")
CHANGED = ("nonzero", "not-one", "ne", "int-ne", "trunc-ne", "try-true", "lowers", "none")
CONTEXTS = ("none", "passed", "passed-unpushed", "caller-hook", "per-listener")
ITERATORS = ("combat", "combat-direct", "run+combat", "run")
PRE_STEPS = ("netid-guard", "negative-cost-passthrough", "owner-in-combat", "enchant-add", "enchant-mul",
             "preview-hittable")
POST_STEPS = ("clamp-min0",)          # plus "dispatch:<Dispatcher>" (a tail call into another dispatcher)
EVENT_OPS = ("event", "filter-set", "preventer-only")
# The other parts of a try pass's condition, beside the listener call ("also"): out-not-null = the value the
# listener gave back through its out parameter must be present. Any other part is an ExtractError.
ALSO = ("out-not-null",)


class ExtractError(Exception):
    """A dispatcher the extractor cannot classify (the message names it)."""


# ----------------------------------------------------------------------------------------------- text helpers

def _close(text: str, i: int, o: str, c: str) -> int:
    """Index of the bracket closing the one at ``i`` (``text[i] == o``)."""
    depth = 0
    for j in range(i, len(text)):
        ch = text[j]
        if ch == o:
            depth += 1
        elif ch == c:
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced")


def _block(text: str, pos: int) -> tuple[int, int]:
    """The brace block that starts at the first '{' after ``pos``: (index of '{', index of '}')."""
    i = text.index("{", pos)
    return i, _close(text, i, "{", "}")


def _ifs(text: str) -> list[tuple[int, str, int, int]]:
    """Every ``if (cond)`` with a braced body: (position, condition, body start, body end)."""
    out = []
    for m in re.finditer(r"\bif \(", text):
        p = m.end() - 1
        q = _close(text, p, "(", ")")
        rest = text[q + 1:]
        if not rest.lstrip().startswith("{"):
            continue
        s, e = _block(text, q + 1)
        out.append((m.start(), text[p + 1:q], s, e))
    return out


def _enclosing_when(text: str, pos: int, name: str) -> str | None:
    whens = [(s, e, cond) for _, cond, s, e in _ifs(text) if s < pos < e
             and re.fullmatch(r"\w+\.HasFlag\([\w.]+\)", cond.strip())]
    if len(whens) > 1:
        raise ExtractError(f"{name}: nested run-time pass flags")
    if not whens:
        return None
    return re.fullmatch(r"\w+\.HasFlag\(([\w.]+)\)", whens[0][2].strip()).group(1)


# ------------------------------------------------------------------------------------------------ the methods

@cache
def _methods(decompiled: str) -> dict[str, dict]:
    """Hook.cs's static methods: header text (signature), body text and whether public."""
    text = ht._read(Path(decompiled) / HOOK_FILE)
    out: dict[str, dict] = {}
    for m in ht.STATIC_METHOD.finditer(text):
        name = m.group(1)
        if name in out:
            raise ExtractError(f"{name}: overloaded static method")
        s, e = _block(text, m.start())
        out[name] = {"header": text[m.start():s], "body": text[s:e + 1],
                     "public": m.group(0).lstrip().startswith("public")}
    return out


PARAM = re.compile(r"(?:(out|ref|in) )?([\w.]+(?:<[^()]*?>)?(?:\?)?(?:\[\])?) (\w+)$")


def params(header: str) -> list[tuple[str, str, str]]:
    """The parameters of a method header: (modifier, type, name)."""
    p = re.search(r"\w\(", header).end() - 1      # the method's own '(' (a tuple return type has a space)
    inner = header[p + 1:_close(header, p, "(", ")")]
    parts, depth, cur = [], 0, ""
    for ch in inner:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    out = []
    for part in parts:
        m = PARAM.match(part.strip())
        if not m:
            raise ExtractError(f"unreadable parameter {part.strip()!r}")
        out.append((m.group(1) or "", m.group(2), m.group(3)))
    return out


# ------------------------------------------------------------------------------------------------ the passes

FOREACH = re.compile(r"\bforeach \([\w<>?]+ (\w+) in (.*)\)\s*$", re.M)
CONTAINS_GATE = re.compile(r"\bif \((.*Iterate\w*HookListeners\(.*\))\.Contains\((\w+)\)\)\s*$", re.M)
HOOK_CALL = re.compile(r"\b(\w+)\.(\w+)\(")


def _iterator(source: str, name: str) -> str:
    s = source.strip()
    if s.startswith("IterateCombatHookListeners("):
        return "combat"
    if re.fullmatch(r"(?:\w+\.)*[Cc]ombatState\??\.IterateHookListeners\(\)", s):
        return "combat-direct"
    if re.fullmatch(r"runState\.IterateHookListeners\(null\)", s):
        return "run"
    if re.fullmatch(r"runState\.IterateHookListeners\(\w+\)", s):
        return "run+combat"
    raise ExtractError(f"{name}: unknown listener iterator {s!r}")


def _conjuncts(cond: str) -> list[str]:
    """A condition split at its top-level ``&&`` (parts stripped); one part when there is none."""
    parts, depth, start, i = [], 0, 0, 0
    while i < len(cond):
        ch = cond[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and cond.startswith("&&", i):
            parts.append(cond[start:i].strip())
            start = i + 2
            i += 1
        i += 1
    return parts + [cond[start:].strip()]


def _try_parts(name: str, cond: str, var: str, hook: str) -> list[str]:
    """The parts of a try condition beside the listener call, in our words (``ALSO``); the call must come first."""
    parts = _conjuncts(cond)
    call = parts[0]
    if not re.fullmatch(rf"{var}\.{hook}\(.*\)", call) or any(f"{var}.{hook}(" in x for x in parts[1:]) or any(
            "||" in x for x in parts):
        raise ExtractError(f"{name}: {hook}: try condition is not the listener call followed by known parts")
    outs = set(re.findall(r"\bout (?:[\w.<>?]+ )?(\w+)", call))
    also = []
    for x in parts[1:]:
        m = re.fullmatch(r"(\w+) != null", x)
        if m and m.group(1) in outs:
            also.append("out-not-null")
        else:
            raise ExtractError(f"{name}: {hook}: unknown part of the try condition {x!r}")
    return also


def _changed(name: str, op: str, block: str, var: str, hook: str) -> tuple[str, list[str]]:
    """The pass's changed criterion and, for try-true, the other parts of its condition (``ALSO``)."""
    adds = [m.start() for m in re.finditer(rf"\.Add\({var}\)", block)]
    if not adds:
        return "none", []
    guards = [(s, cond.strip()) for _, cond, s, e in _ifs(block) if s < adds[0] < e]
    if not guards:
        raise ExtractError(f"{name}: {hook}: changed set added to without a condition")
    cond = max(guards)[1]
    if f"{var}.{hook}(" in cond:
        return "try-true", _try_parts(name, cond, var, hook)
    if re.fullmatch(r"\w+", cond) and re.search(rf"\b{cond} = {var}\.{hook}\(", block):
        return "try-true", []
    for criterion, pattern in (("nonzero", r"\w+ != 0m"), ("not-one", r"\w+ != 1m"),
                               ("int-ne", r"\(int\)\w+ != \(int\)\w+"),
                               ("trunc-ne", r"decimal\.Truncate\(\w+\) != decimal\.Truncate\(\w+\)"),
                               ("ne", r"\w+ != \w+(?: \|\| \w+ != \w+)*")):
        if re.fullmatch(pattern, cond):
            return criterion, []
    if op == "fold-cap" and re.fullmatch(r"\w+ > \w+", cond):
        return "lowers", []
    raise ExtractError(f"{name}: {hook}: unknown changed criterion {cond!r}")


def _op(name: str, block: str, var: str, hook: str) -> str:
    call = rf"{var}\.{hook}\("
    if re.search(rf"\bif \(\w+\.Contains\({var}\)\)", block):
        if not re.search(rf"await {call}", block):
            raise ExtractError(f"{name}: {hook}: filtered pass that is not an event")
        return "filter-set"
    if re.search(rf"await {call}|\bTask \w+ = {call}", block):
        return "event"
    tmp = re.search(rf"\b(?:decimal|int) (\w+) = {call}", block)
    if tmp:
        t = tmp.group(1)
        if re.search(rf"\w+ \+= {t};", block):
            return "fold-add"
        if re.search(rf"\w+ \*= {t};", block):
            return "fold-mul"
        if re.search(rf"\bif \({t} < \w+\)", block):
            return "fold-cap"
        raise ExtractError(f"{name}: {hook}: listener result kept but not folded")
    if re.search(rf"\bif \(!{call}", block):
        return "predicate-all"
    if hook.startswith("Try") and re.search(rf"\bif \({call}|\bbool \w+ = {call}|^\s*{call}[^;]*\bout\b[^;]*\);",
                                            block, re.M):
        return "try"
    if re.search(rf"\bif \({call}[^\n]*\)\s*\{{\s*return true;", block) or re.search(
            rf"\b(\w+) = \1 \|\| {call}", block):
        return "predicate-any"
    if re.search(rf"^\s*{call}[^;]*\);", block, re.M):
        return "mutate"
    if re.search(rf"(?:\b\w+|\)) = {call}", block):
        return "fold-set"
    raise ExtractError(f"{name}: {hook}: unknown pass op")


def _context(name: str, header: str, block: str, var: str, hook: str, call_args: str) -> str:
    if "new HookPlayerChoiceContext(" in block:
        return "per-listener"
    ps = params(header)
    if re.search(rf"\.PushModel\({var}\)", block):
        if any(t == "HookPlayerChoiceContext" for _, t, _ in ps):
            return "caller-hook"
        if any(t == "PlayerChoiceContext" for _, t, _ in ps):
            return "passed"
        raise ExtractError(f"{name}: {hook}: pushes into a context it was not passed")
    ctx = [n for _, t, n in ps if t in ("PlayerChoiceContext", "HookPlayerChoiceContext")]
    if any(re.search(rf"\b{n}\b", call_args) for n in ctx):
        return "passed-unpushed"
    return "none"


def _passes_in(name: str, header: str, body: str, hooks: set[str]) -> list[dict]:
    """The passes written in one method body, in source order (position, record)."""
    loops = []
    for m in FOREACH.finditer(body):
        s, e = _block(body, m.end())
        loops.append((m.start(), s, e, m.group(1), m.group(2)))
    gates = []
    for m in CONTAINS_GATE.finditer(body):
        s, e = _block(body, m.end())
        gates.append((m.start(), s, e, m.group(2), m.group(1)))
    out: list[dict] = []
    seen: set[str] = set()
    for m in HOOK_CALL.finditer(body):
        obj, hook = m.group(1), m.group(2)
        if hook not in hooks or obj in ("Hook", "base"):
            continue
        if hook in seen:
            continue
        seen.add(hook)
        pos = m.start()
        loop = [x for x in loops if x[1] < pos < x[2]]
        gate = [x for x in gates if x[1] < pos < x[2]]
        if loop:
            start, s, e, var, source = max(loop)
            kind = None
        elif gate:
            start, s, e, var, source = max(gate)
            kind = "preventer-only"
        else:
            raise ExtractError(f"{name}: {hook} is called outside a listener loop")
        if obj != var:
            raise ExtractError(f"{name}: {hook} is called on {obj}, not the listener {var}")
        block = body[s:e + 1]
        op = kind or _op(name, block, var, hook)
        p = m.end() - 1
        call_args = body[p:_close(body, p, "(", ")") + 1]
        changed, also = _changed(name, op, block, var, hook)
        rec = {"hook": hook, "op": op, "changed": changed,
               "iterator": _iterator(source, name),
               "context": _context(name, header, block, var, hook, call_args) if op in EVENT_OPS else None}
        when = _enclosing_when(body, start, name)
        if when:
            rec["when"] = when
        if also:
            rec["also"] = also
        out.append({"pos": start, "end": e, "rec": rec})
    return out


# ------------------------------------------------------------------------------------------------ the steps

PRE_PATTERNS = (
    ("netid-guard", re.compile(r"LocalContext\.NetId[\s\S]*?return;\s*\}")),
    ("negative-cost-passthrough", re.compile(r"\bif \(\w+ < 0m\)\s*\{\s*return \w+;\s*\}")),
    ("owner-in-combat", re.compile(r"\bif \(\w+\.Owner\.CombatState == null\)\s*\{\s*return true;\s*\}")),
    ("enchant-add", re.compile(r"\.Enchant\w*Additive\(")),
    ("enchant-mul", re.compile(r"\.Enchant\w*Multiplicative\(")),
    ("preview-hittable", re.compile(r"\.HittableEnemies\b")),
)
RETURN = re.compile(r"\breturn ([^;]*);")


def _steps(name: str, body: str, first_pass: int, publics: set[str]) -> tuple[list[dict], list[dict]]:
    pre: list[tuple[int, dict]] = []
    covered: list[tuple[int, int]] = []
    for step, pat in PRE_PATTERNS:
        for m in pat.finditer(body):
            if m.start() > first_pass:
                raise ExtractError(f"{name}: step {step} after the first pass")
            rec = {"step": step}
            when = _enclosing_when(body, m.start(), name)
            if when:
                rec["when"] = when
            pre.append((m.start(), rec))
            covered.append((m.start(), m.end()))
    for m in re.finditer(r"\breturn\b", body[:first_pass]):
        if not any(a <= m.start() < b for a, b in covered):
            raise ExtractError(f"{name}: early return not explained by a named step")
    post: list[dict] = []
    for m in RETURN.finditer(body, first_pass):
        expr = m.group(1).strip()
        call = re.fullmatch(r"(\w+)\(.*\)", expr)
        if re.fullmatch(r"Math\.Max\(0m, \w+\)", expr):
            step = "clamp-min0"
        elif call and call.group(1) in publics:
            step = f"dispatch:{call.group(1)}"
        elif re.fullmatch(r"\w+|\(\w+, \w+\)", expr):
            continue
        else:
            raise ExtractError(f"{name}: unknown result expression {expr!r}")
        if {"step": step} not in post:
            post.append({"step": step})
    return [r for _, r in sorted(pre, key=lambda x: x[0])], post


# ------------------------------------------------------------------------------------------------ partners

RECEIVER_OPS = ("filter-set", "preventer-only")
MEMBER_HEAD = re.compile(r"^\t+(?:(?:public|private|protected|internal|static|override|virtual|async|sealed)\s+)+"
                         r"[^\n;=]*\)\s*$", re.M)


@cache
def _sources(decompiled: str) -> tuple[tuple[str, str], ...]:
    out = []
    hook_file = (Path(decompiled) / HOOK_FILE).resolve()
    for dirpath, _, files in os.walk(decompiled):
        for f in files:
            if f.endswith(".cs"):
                p = Path(dirpath) / f
                if p.resolve() == hook_file:
                    continue
                t = ht._read(p)
                if "Hook.AfterModifying" in t or "Hook.AfterPreventing" in t:
                    out.append((str(p.relative_to(decompiled)), t))
    return tuple(sorted(out))


def partners(decompiled: str, receivers: set[str], producers: set[str]) -> dict[str, set[str]]:
    """Producer dispatcher -> the receivers its callers pass its changed set or preventer to."""
    out: dict[str, set[str]] = collections.defaultdict(set)
    for _, text in _sources(decompiled):
        for m in re.finditer(r"\bHook\.(\w+)\(", text):
            recv = m.group(1)
            if recv not in receivers:
                continue
            p = m.end() - 1
            args = set(re.findall(r"\b\w+\b", text[p:_close(text, p, "(", ")")]))
            heads = [h.start() for h in MEMBER_HEAD.finditer(text, 0, m.start())]
            scope = text[heads[-1] if heads else 0:m.start()]
            last: dict[str, str] = {}          # variable -> the dispatcher whose call bound it last
            for c in re.finditer(r"\bHook\.(\w+)\(", scope):
                q = c.end() - 1
                try:
                    call = scope[q:_close(scope, q, "(", ")") + 1]
                except ValueError:
                    continue
                outs = set(re.findall(r"\bout (?:[\w<>?]+ )?(\w+)\b", call))
                lhs = re.search(r"(?:^|[\s(])(\w+) = $", scope[max(0, c.start() - 200):c.start()])
                for v in outs | ({lhs.group(1)} if lhs else set()):
                    last[v] = c.group(1)
            for v in args:
                if last.get(v) in producers:
                    out[last[v]].add(recv)
    return out


# ------------------------------------------------------------------------------------------------ extraction

def extract(decompiled: str) -> dict:
    """The descriptor without ``args``: {"format", "dispatchers": {name: {...}}}."""
    hooks = set(ht.hook_names(decompiled))
    methods = _methods(decompiled)
    publics = {n for n, m in methods.items() if m["public"]}
    privates = {n for n, m in methods.items() if not m["public"]}
    table = ht.dispatchers(decompiled)
    out: dict[str, dict] = {}
    for name in sorted(publics):
        meth = methods[name]
        body, header = meth["body"], meth["header"]
        found = _passes_in(name, header, body, hooks)
        for h in privates:
            m = re.search(rf"\b{h}\(", body)
            if m:
                sub = _passes_in(name, methods[h]["header"], methods[h]["body"], hooks)
                for x in sub:
                    x["pos"] = m.start()
                    x["end"] = m.start()
                found += sub
        found.sort(key=lambda x: x["pos"])
        if not found:
            raise ExtractError(f"{name}: public dispatcher with no hook pass")
        names = [x["rec"]["hook"] for x in found]
        if name in table and table[name]["passes"] != names:
            raise ExtractError(f"{name}: passes {names} disagree with hook_table {table[name]['passes']}")
        recs = [x["rec"] for x in found]
        iters = {r.pop("iterator") for r in recs}
        if len(iters) != 1:
            raise ExtractError(f"{name}: passes over different iterators {sorted(iters)}")
        ctxs = {r.pop("context") for r in recs} - {None}
        if len(ctxs) > 1:
            raise ExtractError(f"{name}: passes with different contexts {sorted(ctxs)}")
        ops = {r["op"] for r in recs}
        if ops & set(EVENT_OPS) and ops - set(EVENT_OPS):
            raise ExtractError(f"{name}: event and value passes mixed")
        first = min(x["pos"] for x in found)
        pre, post = _steps(name, body, first, publics)
        waits = []
        for w in re.finditer(r"\bTask\.WhenAll\(", body):
            done = [i for i, x in enumerate(found) if x["end"] < w.start()]
            if not done:
                raise ExtractError(f"{name}: wait before any pass")
            waits.append(max(done))
        out[name] = {"iterator": iters.pop(), "passes": recs, "context": ctxs.pop() if ctxs else "none",
                     "wait_all": sorted(set(waits)), "pre": pre, "post": post, "partner": []}
    receivers = {n for n, d in out.items() if d["passes"][0]["op"] in RECEIVER_OPS}
    for prod, recvs in partners(decompiled, receivers, set(out) - receivers).items():
        out[prod]["partner"] = sorted(recvs)
    return {"format": FORMAT, "dispatchers": out}


def shape_key(d: dict) -> str:
    """One dispatcher's shape: its ops and changed criteria, context, waits and steps (not hooks or iterator)."""
    ops = "+".join(dict.fromkeys(f"{p['op']}/{p['changed']}" + ("?" if "when" in p else "") for p in d["passes"]))
    steps = ",".join(s["step"].split(":")[0] for s in d["pre"] + d["post"])
    return f"{ops} | {d['context']}" + (" | wait" if d["wait_all"] else "") + (f" | {steps}" if steps else "")


def shape_counts(desc: dict) -> dict[str, int]:
    return dict(sorted(collections.Counter(shape_key(d) for d in desc["dispatchers"].values()).items()))


# ------------------------------------------------------------------------------------------------ the file

def _scalar(v) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=True)


def dumps(desc: dict) -> str:
    """Stable formatting: sorted keys, one dispatcher field per line, one list item per line."""
    lines = ["{", f'  "format": {desc["format"]},', '  "dispatchers": {']
    names = sorted(desc["dispatchers"])
    for i, name in enumerate(names):
        d = desc["dispatchers"][name]
        lines.append(f"    {_scalar(name)}: {{")
        keys = sorted(d)
        for j, k in enumerate(keys):
            v = d[k]
            sep = "," if j < len(keys) - 1 else ""
            if isinstance(v, list) and v and isinstance(v[0], dict):
                lines.append(f"      {_scalar(k)}: [")
                lines += [f"        {_scalar(x)}" + ("," if n < len(v) - 1 else "") for n, x in enumerate(v)]
                lines.append(f"      ]{sep}")
            else:
                lines.append(f"      {_scalar(k)}: {_scalar(v)}{sep}")
        lines.append("    }" + ("," if i < len(names) - 1 else ""))
    lines += ["  }", "}", ""]
    return "\n".join(lines)


def load(path: Path = DESCRIPTOR) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decompiled", default=ht.DEFAULT_DECOMPILED)
    ap.add_argument("--write", action="store_true", help="rewrite hooks.json, keeping the hand-written args")
    a = ap.parse_args(argv)
    try:
        desc = extract(a.decompiled)
    except ExtractError as e:
        print(f"hooks_extract: cannot classify: {e}", file=sys.stderr)
        return 1
    if a.write:
        old = load()["dispatchers"] if DESCRIPTOR.exists() else {}
        missing = sorted(n for n in desc["dispatchers"] if "args" not in old.get(n, {}))
        if missing:
            print(f"hooks_extract: no hand-written args for {', '.join(missing)}", file=sys.stderr)
            return 1
        for n, d in desc["dispatchers"].items():
            d["args"] = old[n]["args"]
        DESCRIPTOR.write_text(dumps(desc), encoding="utf-8", newline="\n")
    counts = shape_counts(desc)
    print(f"{len(desc['dispatchers'])} dispatchers in {len(counts)} shapes")
    for k, v in counts.items():
        print(f"  {v:3d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
