"""Mutation testing of the emulator against recorded fights: how many seeded bugs do our fight checks catch?

    python -m tools.fidelity.mutate prepare --out-json <referee out.json> --emu <emu.json> --dir <run dir>
    python -m tools.fidelity.mutate run --dir <run dir> [--mutants 150] [--workers 2] [--shards 2] [--minutes 90]
    python -m tools.fidelity.mutate report --dir <run dir> [--json summary.json]

`prepare` picks the fights the current emulator replays identically (the same test as referee/batch.py:
no error, no first divergence), copies the engine into a scratch tree, adds a probe at the start of every
block and switch section of the target files, and replays those fights once (EmuDump digest, both the plain
replay and the re-anchored one) to learn which fights ran which block and each step's hash. `run` samples
mutants (weighted by sqrt of the number of fights that ran their block), builds EmuDump for each in a
scratch copy (incremental), replays only the fights that ran the mutated block (a fight that never runs it
is unchanged by construction) in a random order split across processes, and stops at the first fight the
checks flag. `report` gives kill rates per file and operator, the survivors, and the kill rate a random
subset of n fights would reach.

A mutant is killed when, in some fight, the plain replay now has an error or a first divergence from the
game (fight_compare's views), or a re-anchored step that matched the game no longer does (reanchor_compare),
or the replay hangs or crashes. A step whose hash changed but that both checks still pass is counted as a
silent difference: the checks do not look at what changed.

Operators: flip a comparison (boundary and negation), +/-1 on an integer literal on a damage, block, energy,
draw, HP or amount line, swap two adjacent call statements, delete a call statement, use another combat RNG
stream, drop a Math.Max/Min/Clamp or a Floor/Ceiling/Round. Sources are edited as text with comments and
strings masked; anything that does not compile is recorded as build-failed.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = "src/Sts2Emulator"
TARGETS = ["Core/CombatEngine.cs", "Core/BuffSystem.cs", "Core/EnemyAI.cs",
           "Core/Effects/CardEffects.cs", "Core/Effects/PotionEffects.cs", "Core/Effects/RelicEffects.cs"]
PROBE = "global::Sts2Emulator.MutCov.Hit({});"
MUTCOV = """// Written by tools/fidelity/mutate.py into its scratch copy only: which probed blocks ran.
namespace Sts2Emulator;

public static class MutCov
{{
    static readonly bool[] Seen = new bool[{n}];

    public static void Hit(int i) => Seen[i] = true;

    public static int[] Take()
    {{
        var got = new List<int>();
        for (int i = 0; i < Seen.Length; i++)
        {{
            if (Seen[i]) {{ got.Add(i); Seen[i] = false; }}
        }}

        return [.. got];
    }}
}}
"""
BELOW_NORMAL = 0x00004000  # Windows priority class for every emulator process this tool starts

# ---------------------------------------------------------------------------------------------- source text


def mask(text: str) -> str:
    """The text with comments and string/char literal contents blanked (same length, newlines kept)."""
    out = list(text)
    i, n = 0, len(text)

    def blank(a: int, b: int) -> None:
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
        elif text.startswith('"""', i):
            j = text.find('"""', i + 3)
            j = n if j < 0 else j + 3
            blank(i, j)
            i = j
        elif c == '"' or (c in "@$" and i + 1 < n and text[i + 1] in '"@$'):
            k = i
            verbatim = False
            while k < n and text[k] in "@$":
                verbatim |= text[k] == "@"
                k += 1
            if k >= n or text[k] != '"':
                i += 1
                continue
            j = k + 1
            depth = 0  # interpolation holes may hold strings of their own; blank them all
            while j < n:
                ch = text[j]
                if verbatim and ch == '"' and j + 1 < n and text[j + 1] == '"':
                    j += 2
                    continue
                if not verbatim and ch == "\\":
                    j += 2
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth = max(0, depth - 1)
                elif (ch == '"' and depth == 0) or (ch == "\n" and not verbatim and depth == 0):
                    break
                j += 1
            blank(i, j + 1)
            i = j + 1
        elif c == "'":
            j = i + 1
            while j < n and text[j] != "'" and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            blank(i, j + 1)
            i = j + 1
        else:
            i += 1
    return "".join(out)


TYPE_KW = re.compile(r"\b(class|struct|interface|enum|namespace|record)\b")
STMT_KW = re.compile(r"^(if|else|for|foreach|while|do|try|catch|finally|using|lock|fixed|unchecked|checked|unsafe)\b")
ACCESSOR = re.compile(r"^((public|private|protected|internal)\s+)*(get|set|init|add|remove)$")


def _header(mlines: list, i: int) -> str:
    """The statement text before the `{` on line i (masked lines, joined back to the last terminator)."""
    parts = []
    for k in range(i - 1, max(-1, i - 12), -1):
        s = mlines[k].strip()
        if not s or s.endswith((";", "{", "}")):
            break
        parts.append(s)
        if s.endswith(":") and not parts[1:]:  # a case label: that is the whole header
            break
    return " ".join(reversed(parts))


def classify(header: str, parent: str) -> str:
    """What a `{` standing on its own line opens: method, stmt (a statement block), switch, type, property, init."""
    h = re.sub(r"^(\[[^\]]*\]\s*)+", "", header).strip()
    if not h:
        return "stmt" if parent in ("method", "stmt", "switch") else "init"
    if re.match(r"^switch\s*\(", h):
        return "switch"
    if h.endswith(":"):
        return "stmt"  # a block right after a case label
    if h.count("(") > h.count(")") or h.count("[") > h.count("]"):
        return "init"  # still inside an expression: `foreach (var p in new[]` + `{`
    if h.endswith(("switch", "=", ",", "(", "[", "?", "??", "&&", "||")) or re.search(r"\b(is|and|or|not|when|in)$", h):
        return "init"  # an initializer, a switch expression or a property pattern (even inside an `if (`)
    if STMT_KW.match(h) or h.endswith("=>") or ACCESSOR.match(h):
        return "stmt"
    if re.search(r"\bnew\b", h) or h.startswith("return"):
        return "init"
    pre = h.split("(")[0]
    if TYPE_KW.search(pre):
        return "type"
    if "(" in h and re.search(r"\)\s*(where\b.*|:\s*(base|this)\s*\(.*\))?$", h):
        if "=" in pre:
            return "init"
        return "method" if parent == "type" else "stmt"
    if parent == "type":
        return "property"
    return "unknown"


LABEL = re.compile(r"^\s*(case\b.*|default)\s*:\s*$")


@dataclass
class Probed:
    text: str            # the source with probes appended (line numbers unchanged)
    probes: dict         # probe id -> line number (1-based) it sits on
    unit: list           # per line (0-based): the probe whose block holds that line, or None


def instrument(text: str, first_id: int = 0, skip_lines: frozenset = frozenset()) -> Probed:
    """Append a probe to every `{` line that opens a method or statement block, and to the last label of
    each switch section. The unit of a line is the innermost such block (or section) it sits in.
    """
    lines = text.split("\n")
    mlines = mask(text).split("\n")
    probes, unit = {}, []
    nid = first_id
    stack = [["type", None]]  # [kind, current unit]
    for i, (line, m) in enumerate(zip(lines, mlines)):
        s = m.strip()
        top = stack[-1]
        if s == "{":
            kind = classify(_header(mlines, i), top[0])
            u = top[1]
            if kind in ("method", "stmt") and (i + 1) not in skip_lines:
                u = nid
                probes[nid] = i + 1
                at = m.index("{") + 1  # right after the brace: a trailing comment would swallow it
                lines[i] = line[:at] + " " + PROBE.format(nid) + line[at:]
                nid += 1
            stack.append([kind, u])
            unit.append(top[1])
            continue
        unit.append(top[1])
        if top[0] == "switch" and LABEL.match(m):
            nxt = next((mlines[k] for k in range(i + 1, len(mlines)) if mlines[k].strip()), "")
            if not LABEL.match(nxt) and (i + 1) not in skip_lines:
                probes[nid] = i + 1
                at = m.rindex(":") + 1
                lines[i] = line[:at] + " " + PROBE.format(nid) + line[at:]
                top[1] = nid
                nid += 1
            continue
        for ch in m:
            if ch == "{":
                stack.append(["inline", stack[-1][1]])
            elif ch == "}" and len(stack) > 1:
                stack.pop()  # a switch's sections end with it: later lines are the parent's again
    return Probed("\n".join(lines), probes, unit)


# ----------------------------------------------------------------------------------------------- mutants


@dataclass
class Mutant:
    id: str
    file: str            # relative to src/Sts2Emulator
    line: int            # 1-based; a swap also changes line + 1
    op: str
    old: str             # the whole line(s) before
    new: str             # the whole line(s) after
    unit: int | None = None
    fights: int = 0      # how many of the recorded fights ran its block
    extra: dict = field(default_factory=dict)


CMP = re.compile(r"(?<=\s)(<=|>=|==|!=|<|>)(?=\s)")
BOUNDARY = {"<": "<=", "<=": "<", ">": ">=", ">=": ">"}
NEGATE = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}
INT = re.compile(r"(?<![\w.])\d+(?![\w.])")
NUM_CONTEXT = re.compile(r"damage|block|energy|draw|hp\b|heal|strength|dexterity|amount|magnitude|"
                         r"vulnerable|weak|frail|poison|hits|cost|stack", re.IGNORECASE)
CALL = re.compile(r"^(\s*)(?:await\s+)?([A-Za-z_][\w.]*(?:<[\w.,\s]*>)?)\s*\((.*)\)\s*;\s*$")
RNG_SWAP = {"ShuffleRng": "CardSelectionRng", "CardSelectionRng": "ShuffleRng", "TargetRng": "ShuffleRng",
            "CardGenerationRng": "CardSelectionRng", "EnergyCostRng": "CardGenerationRng",
            "PotionGenerationRng": "CardGenerationRng", "OrbGenerationRng": "CardGenerationRng",
            "NicheHpRng": "TargetRng"}
RNG = re.compile(r"\.(" + "|".join(RNG_SWAP) + r")\b")
MATHF = re.compile(r"\bMath(?:F)?\.(Max|Min|Clamp|Floor|Ceiling|Round)\s*\(")


def _balanced(m: str, start: int) -> int | None:
    """Index just past the `)` matching the `(` at start, on one masked line."""
    depth = 0
    for k in range(start, len(m)):
        if m[k] in "([{":
            depth += 1
        elif m[k] in ")]}":
            depth -= 1
            if depth == 0:
                return k + 1
    return None


def _split_args(m: str, a: int, b: int) -> list:
    """Top-level argument spans (start, end) inside m[a:b]."""
    spans, depth, s = [], 0, a
    for k in range(a, b):
        if m[k] in "([{<" and not (m[k] == "<" and m[k - 1] == " "):
            depth += m[k] != "<"
        elif m[k] in ")]}":
            depth -= 1
        elif m[k] == "," and depth == 0:
            spans.append((s, k))
            s = k + 1
    spans.append((s, b))
    return spans


def _is_literal(s: str) -> bool:
    return re.fullmatch(r"-?\d+(\.\d+)?[fFdDmM]?", s.strip()) is not None


def call_statement(m: str) -> str | None:
    """The called name if this masked line is one whole call statement, else None."""
    c = CALL.match(m)
    if c and re.search(r"\bout\s+[\w<>,\[\]?]+\s+\w+", m):
        return None  # it declares a variable (`out int x`): the call cannot go without it
    if not c or c.group(2).split(".")[0] in ("return", "throw", "yield", "await", "new", "nameof", "if", "while"):
        return None
    start = m.index("(", c.start(2))
    end = _balanced(m, start)
    if end is None or m[end:].strip() != ";":
        return None
    return c.group(2).split(".")[-1]


def sites(rel: str, text: str) -> list:
    """Every mutant of one file (rel is its path under src/Sts2Emulator)."""
    lines = text.split("\n")
    mlines = mask(text).split("\n")
    out = []
    stem = Path(rel).stem

    def add(i, op, new, extra=None, span=1):
        old = "\n".join(lines[i:i + span])
        if new != old:
            out.append(Mutant(f"{stem}:{i + 1}:{op}:{len(out)}", rel, i + 1, op, old, new, extra=extra or {}))

    for i, (line, m) in enumerate(zip(lines, mlines)):
        s = m.strip()
        if not s or s.startswith(("#", "[", "using ")) or " const " in f" {s} ":
            continue
        for c in CMP.finditer(m):
            op = c.group(1)
            if op in BOUNDARY:
                add(i, "cmp-boundary", line[:c.start()] + BOUNDARY[op] + line[c.end():], {"from": op, "to": BOUNDARY[op], "col": c.start(), "end": c.end(), "rep": BOUNDARY[op]})
            add(i, "cmp-negate", line[:c.start()] + NEGATE[op] + line[c.end():], {"from": op, "to": NEGATE[op], "col": c.start(), "end": c.end(), "rep": NEGATE[op]})
        if NUM_CONTEXT.search(m) and not LABEL.match(m):
            for c in INT.finditer(m):
                v = int(c.group())
                for d in (1, -1):
                    add(i, "literal+1" if d > 0 else "literal-1", line[:c.start()] + str(v + d) + line[c.end():], {"from": v, "col": c.start(), "end": c.end(), "rep": str(v + d)})
        for c in RNG.finditer(m):
            add(i, "rng-stream", line[:c.start(1)] + RNG_SWAP[c.group(1)] + line[c.end(1):], {"from": c.group(1), "col": c.start(1), "end": c.end(1), "rep": RNG_SWAP[c.group(1)]})
        for c in MATHF.finditer(m):
            open_ = c.end() - 1
            close = _balanced(m, open_)
            if close is None:
                continue
            args = _split_args(m, open_ + 1, close - 1)
            fn = c.group(1)
            if fn in ("Max", "Min") and len(args) == 2:
                keep = [k for k, (a, b) in enumerate(args) if not _is_literal(m[a:b])] or [0]
                for k in keep:
                    a, b = args[k]
                    add(i, "drop-clamp", line[:c.start()] + "(" + line[a:b].strip() + ")" + line[close:],
                        {"fn": fn, "col": c.start(), "end": close, "rep": "(" + line[a:b].strip() + ")"})
            elif fn == "Clamp" and len(args) == 3:
                a, b = args[0]
                add(i, "drop-clamp", line[:c.start()] + "(" + line[a:b].strip() + ")" + line[close:],
                        {"fn": fn, "col": c.start(), "end": close, "rep": "(" + line[a:b].strip() + ")"})
            elif fn in ("Floor", "Ceiling", "Round") and len(args) == 1:
                a, b = args[0]
                add(i, "drop-rounding", line[:c.start()] + "(" + line[a:b].strip() + ")" + line[close:],
                        {"fn": fn, "col": c.start(), "end": close, "rep": "(" + line[a:b].strip() + ")"})
        name = call_statement(m)
        if name:
            indent = line[: len(line) - len(line.lstrip())]
            add(i, "delete-call", indent + "{ }" + ("\r" if line.endswith("\r") else ""), {"call": name})
            if i + 1 < len(lines):
                n2 = call_statement(mlines[i + 1])
                ind2 = lines[i + 1][: len(lines[i + 1]) - len(lines[i + 1].lstrip())]
                if n2 and ind2 == indent and lines[i + 1].strip() != line.strip():
                    add(i, "swap-calls", lines[i + 1] + "\n" + line, {"calls": [name, n2]}, span=2)
    return out


def apply(text: str, m: Mutant) -> str:
    lines = text.split("\n")
    span = m.old.count("\n") + 1
    assert "\n".join(lines[m.line - 1:m.line - 1 + span]) == m.old, f"{m.id}: source moved"
    lines[m.line - 1:m.line - 1 + span] = m.new.split("\n")
    return "\n".join(lines)


def sample(cands: list, k: int, rng: random.Random) -> list:
    """K mutants without replacement, weighted by sqrt(fights that ran the block): hot code is drawn
    more often, but a block one fight in fifty runs is still drawn.
    """
    pool = [c for c in cands if c.fights > 0]
    keys = sorted(pool, key=lambda c: -math.log(1.0 - rng.random() + 1e-300) / math.sqrt(c.fights))
    return keys[:k]


# -------------------------------------------------------------------------------------------- scratch copies


COPIED_TREES = (ENGINE, "tools/emudump", "src/Sts2Emulator.Tests/Support/Invariants")  # EmuDump.csproj compiles the last one in
COPIED_FILES = ("data/id_map.json", "tools/fidelity/power_aliases.json")  # and links these two


def make_copy(src: Path, dst: Path) -> None:
    """The engine, EmuDump, the invariant sources it compiles in and the data files it links, in dst (sources only;
    dst keeps its own bin/obj between runs)."""
    for rel in COPIED_TREES:
        if not (src / rel).is_dir():
            continue
        for p in (src / rel).rglob("*"):
            r = p.relative_to(src)
            if any(part in ("bin", "obj") for part in r.parts) or p.is_dir():
                continue
            t = dst / r
            if t.exists() and t.stat().st_size == p.stat().st_size and t.read_bytes() == p.read_bytes():
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)
    for rel in COPIED_FILES:
        if (src / rel).is_file():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, dst / rel)


def dotnet_env() -> dict:
    return dict(os.environ, DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_NOLOGO="1", MSBUILDDISABLENODEREUSE="1")


def build(copy: Path, restore: bool = False) -> tuple:
    proj = str(copy / "tools/emudump")
    if restore:
        r = subprocess.run(["dotnet", "restore", proj, "--source", str(Path.home() / ".nuget/packages"), "-v", "q"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dotnet_env())
        if r.returncode:
            return False, r.stdout[-2000:]
    r = subprocess.run(["dotnet", "build", proj, "-c", "Release", "--no-restore", "-v", "q", "-nologo", "-clp:NoSummary"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dotnet_env())
    return r.returncode == 0, r.stdout[-200000:]


def emudump(copy: Path) -> str:
    return str(copy / "tools/emudump/bin/Release/net9.0/EmuDump.dll")


def popen(args: list) -> subprocess.Popen:
    kw = {"creationflags": BELOW_NORMAL} if os.name == "nt" else {}
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=dotnet_env(), **kw)


def digest_all(copy: Path, out_json: str, fights: list, shards: int, baseline: str | None = None) -> dict:
    """Run `EmuDump digest` for both modes over fights, split into shards; {mode: {i: line}}."""
    res = {"fight": {}, "reanchor": {}, "keys": None}
    procs = []
    for mode in ("fight", "reanchor"):
        for s in range(shards):
            part = fights[s::shards]
            if not part:
                continue
            args = ["dotnet", emudump(copy), "digest", mode, out_json, "--only", ",".join(map(str, part))]
            if baseline:
                args += ["--baseline", f"{baseline}.{mode}.jsonl"]
            procs.append((mode, popen(args)))

    def drain(mode, p):
        for raw in p.stdout:
            o = json.loads(raw)
            if "keys" in o:
                res["keys"] = o["keys"]
            else:
                res[mode][o["i"]] = o

    ts = [threading.Thread(target=drain, args=mp) for mp in procs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for _, p in procs:
        p.wait()
    return res


# ---------------------------------------------------------------------------------------------- judging


class Judge:
    """The checks batch.py and reanchor_compare apply, against a mutant's changed fights."""

    def __init__(self, games: list, nm: dict, keys, base_verdicts: dict):
        self.games, self.nm, self.keys, self.base = games, nm, keys, base_verdicts

    def fight(self, i: int, emu: dict) -> str | None:
        from .coverage import first_divergence  # noqa: PLC0415

        if emu.get("error"):
            return "error"
        return "diverge" if first_divergence(self.games[i], emu, self.nm) is not None else None

    def reanchor(self, i: int, emu: dict) -> str | None:
        from .reanchor_compare import analyse  # noqa: PLC0415

        if emu.get("error"):
            return "error"
        now = {s["j"]: s["verdict"] for s in analyse(self.games[i], emu, self.nm, "stop", self.keys)["steps"]}
        lost = [j for j, v in self.base[i].items() if v == "match" and now.get(j) != "match"]
        return "reanchor" if lost else None


def base_verdicts(games: list, reanchored: list, nm: dict, keys, fights: list) -> dict:
    from .reanchor_compare import analyse  # noqa: PLC0415

    return {i: {s["j"]: s["verdict"] for s in analyse(games[i], reanchored[i], nm, "stop", keys)["steps"]} for i in fights}


# ---------------------------------------------------------------------------------------------- one mutant


def run_mutant(m: Mutant, copy: Path, ctx: dict, shards: int, stall: float = 120.0) -> dict:
    """Build and replay one mutant; its result line."""
    t0 = time.time()
    path = copy / ENGINE / m.file
    original = path.read_bytes()
    rec = {**asdict(m), "status": None}
    try:
        path.write_bytes(apply(original.decode("utf-8"), m).encode("utf-8"))
        ok, log = build(copy)
        rec["build_s"] = round(time.time() - t0, 1)
        if not ok:
            rec["status"] = "build_failed"
            rec["build_error"] = next((ln.strip()[-240:] for ln in log.splitlines() if " error " in ln), log[-240:])
            return rec
        order = ctx["order"](m)  # this mutant's fights, in its random order, with their positions
        pos = {i: p for p, i in order}
        fights = [i for _, i in order]
        best = [math.inf]  # first position (1-based, among all recorded fights) a check flagged
        kills, silent, seen = [], [], collections.Counter()
        lock = threading.Lock()
        procs = []
        for mode in ("fight", "reanchor"):
            for s in range(shards):
                part = fights[s::shards]
                if part:
                    args = ["dotnet", emudump(copy), "digest", mode, ctx["out_json"], "--only", ",".join(map(str, part)),
                            "--baseline", f"{ctx['baseline']}.{mode}.jsonl"]
                    procs.append({"mode": mode, "part": part, "p": popen(args), "last": 0, "done": 0, "t": time.time()})

        def drain(pr):
            for raw in pr["p"].stdout:
                o = json.loads(raw)
                if "i" not in o:
                    continue
                i, p = o["i"], pos[o["i"]]
                verdict = None
                if "full" in o:
                    judge = ctx["judge"]
                    verdict = judge.fight(i, o["full"]) if pr["mode"] == "fight" else judge.reanchor(i, o["full"])
                with lock:
                    pr["last"], pr["t"] = p, time.time()
                    pr["done"] += 1
                    seen[pr["mode"]] += 1
                    if verdict:
                        kills.append({"i": i, "pos": p, "mode": pr["mode"], "why": verdict})
                        best[0] = min(best[0], p)
                    elif "full" in o:
                        silent.append({"i": i, "mode": pr["mode"]})
                    if p >= best[0]:
                        break
            try:
                pr["p"].kill()
            except OSError:
                pass

        ts = [threading.Thread(target=drain, args=(pr,), daemon=True) for pr in procs]
        for t in ts:
            t.start()
        while any(t.is_alive() for t in ts):
            time.sleep(0.2)
            with lock:
                for pr in procs:
                    alive = pr["p"].poll() is None
                    if alive and pr["last"] > best[0]:
                        pr["p"].kill()  # every fight it has left comes after the first kill
                    elif alive and time.time() - pr["t"] > stall:
                        pr["p"].kill()
                        nxt = pr["part"][pr["done"]] if pr["done"] < len(pr["part"]) else pr["part"][-1]
                        kills.append({"i": nxt, "pos": pos[nxt], "mode": pr["mode"], "why": "timeout"})
                        best[0] = min(best[0], pos[nxt])
        for pr in procs:
            pr["p"].wait()
            if pr["p"].returncode not in (0, None) and pr["done"] < len(pr["part"]) and pr["last"] < best[0] \
                    and not any(k["why"] == "timeout" and k["mode"] == pr["mode"] for k in kills):
                nxt = pr["part"][pr["done"]]
                if pos[nxt] < best[0] or best[0] == math.inf:
                    kills.append({"i": nxt, "pos": pos[nxt], "mode": pr["mode"], "why": "crash"})
                    best[0] = min(best[0], pos[nxt])
        first = [k for k in kills if k["pos"] == best[0]]
        rec.update(status="killed" if kills else "survived", first_kill_pos=None if best[0] == math.inf else best[0],
                   first_kill=first, fights_selected=len(fights), fights_run=dict(seen), silent=len(silent))
        return rec
    finally:
        path.write_bytes(original)
        rec["secs"] = round(time.time() - t0, 1)


# ------------------------------------------------------------------------------------------------- commands


def _load_games(out_json: str) -> list:
    return json.loads(Path(out_json).read_text(encoding="utf-8"))["results"]


def cmd_prepare(a) -> int:
    from .coverage import first_divergence  # noqa: PLC0415
    from .fight_compare import names  # noqa: PLC0415

    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    nm = names(json.loads((ROOT / "data/id_map.json").read_text(encoding="utf-8")))
    games = _load_games(a.out_json)
    emus = json.loads(Path(a.emu).read_text(encoding="utf-8"))["results"]
    fights = [i for i, (g, e) in enumerate(zip(games, emus))
              if g.get("steps") and not g.get("error") and not e.get("error") and first_divergence(g, e, nm) is None]
    del emus
    print(f"{len(fights)} of {len(games)} fights identical", flush=True)
    # Probes go into a copy of the engine; line numbers stay put, so mutants found on the clean source apply to it.
    cov = d / "cov"
    make_copy(ROOT, cov)
    skip: dict = collections.defaultdict(set)
    for attempt in range(15):
        nid, units, probe_lines = 0, {}, {}
        for rel in TARGETS:
            text = (ROOT / ENGINE / rel).read_bytes().decode("utf-8")
            pr = instrument(text, nid, frozenset(skip[rel]))
            (cov / ENGINE / rel).write_bytes(pr.text.encode("utf-8"))
            units[rel] = pr.unit
            probe_lines[rel] = {pid: ln for pid, ln in pr.probes.items()}
            nid += len(pr.probes)
        (cov / ENGINE / "Core/MutCov.cs").write_text(MUTCOV.format(n=max(1, nid)), encoding="utf-8")
        ok, log = build(cov, restore=attempt == 0)
        if ok:
            break
        bad = re.findall(r"[\\/]src[\\/]Sts2Emulator[\\/](Core[\\/][\w\\/]+\.cs)\((\d+),\d+\): error", log)
        if not bad:
            print(log[-3000:])
            return 1
        for f, ln in bad:
            # The error is reported on or after the probe that caused it: drop the nearest one above.
            rel = f.replace("\\", "/")
            above = [pl for pl in probe_lines[rel].values() if pl <= int(ln)]
            if above:
                skip[rel].add(max(above))
        print(f"probe build attempt {attempt}: dropping probes on {len(bad)} lines", flush=True)
    else:
        return 1
    print(f"{nid} probes; replaying {len(fights)} fights for coverage and hashes", flush=True)
    t0 = time.time()
    res = digest_all(cov, a.out_json, fights, a.shards)
    print(f"coverage run {time.time() - t0:.0f} s", flush=True)
    for mode in ("fight", "reanchor"):
        with Path(d / f"baseline.{mode}.jsonl").open("w", encoding="utf-8") as f:
            for i in fights:
                o = dict(res[mode][i])
                o.pop("cov", None)
                f.write(json.dumps(o) + "\n")
    fight_cov = {i: sorted(set(res["fight"][i].get("cov", [])) | set(res["reanchor"][i].get("cov", []))) for i in fights}
    # Mutants of the clean source, each tied to the probe whose block holds it.
    unit_fights = collections.Counter(u for us in fight_cov.values() for u in us)
    cands = []
    for rel in TARGETS:
        text = (ROOT / ENGINE / rel).read_bytes().decode("utf-8")
        for m in sites(rel, text):
            m.unit = units[rel][m.line - 1]
            m.fights = unit_fights.get(m.unit, 0) if m.unit is not None else -1
            cands.append(m)
    keys = res["keys"]
    (d / "prepared.json").write_text(json.dumps({
        "out_json": str(Path(a.out_json).resolve()), "fights": fights, "fight_cov": fight_cov, "keys": keys,
        "probes": {rel: pl for rel, pl in probe_lines.items()}, "candidates": [asdict(c) for c in cands],
        "commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
    }), encoding="utf-8")
    by = collections.Counter((c.file, "run" if c.fights > 0 else "never" if c.fights == 0 else "no probe") for c in cands)
    for rel in TARGETS:
        print(f"  {rel}: {by[rel, 'run']} mutants in blocks the fights ran, {by[rel, 'never']} never ran, "
              f"{by[rel, 'no probe']} outside probed blocks")
    return 0


def cmd_run(a) -> int:
    from .fight_compare import names  # noqa: PLC0415
    from .reanchor_compare import keys_of  # noqa: PLC0415

    d = Path(a.dir)
    prep = json.loads((d / "prepared.json").read_text(encoding="utf-8"))
    nm = names(json.loads((ROOT / "data/id_map.json").read_text(encoding="utf-8")))
    games = _load_games(prep["out_json"])
    fights = prep["fights"]
    keys = keys_of(prep["keys"])
    reanchored = json.loads(Path(a.reanchor).read_text(encoding="utf-8"))["results"]
    judge = Judge(games, nm, keys, base_verdicts(games, reanchored, nm, keys, fights))
    del reanchored
    cands = [Mutant(**c) for c in prep["candidates"]]
    done_path = d / "results.jsonl"
    done = {json.loads(ln)["id"] for ln in done_path.open(encoding="utf-8")} if done_path.exists() else set()
    rng = random.Random(a.seed)
    picked = [c for c in cands if c.id in set(a.ids.split(","))] if a.ids else sample(cands, a.mutants, rng)
    todo = [m for m in picked if m.id not in done]
    cov_sets = {int(i): set(us) for i, us in prep["fight_cov"].items()}

    def order(m: Mutant) -> list:
        perm = list(fights)
        random.Random(f"{a.seed}:{m.id}").shuffle(perm)
        return [(p + 1, i) for p, i in enumerate(perm) if m.unit is None or m.unit in cov_sets[i]]

    ctx = {"out_json": prep["out_json"], "baseline": str(d / "baseline"), "judge": judge, "order": order}
    copies = [d / f"w{k}" for k in range(a.workers)]
    for c in copies:
        make_copy(ROOT, c)
        ok, log = build(c, restore=True)
        if not ok:
            print(log[-2000:])
            return 1
    # The unmutated engine must reproduce the baseline (taken with probes in) exactly, or no kill means anything.
    check = digest_all(copies[0], prep["out_json"], fights[:: max(1, len(fights) // 12)], 2, ctx["baseline"])
    differ = sorted({i for mode in ("fight", "reanchor") for i, o in check[mode].items() if "full" in o})
    if differ:
        print(f"the unmutated build differs from the baseline on fights {differ}: not deterministic, stopping")
        return 1
    print(f"{len(todo)} mutants to run on {a.workers} workers x {a.shards} shards per mode", flush=True)
    deadline = time.time() + a.minutes * 60
    q = list(todo)
    qlock, wlock = threading.Lock(), threading.Lock()
    counts = collections.Counter()

    def worker(copy):
        while time.time() < deadline:
            with qlock:
                if not q:
                    return
                m = q.pop(0)
            rec = run_mutant(m, copy, ctx, a.shards)
            with wlock:
                with done_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                counts[rec["status"]] += 1
                print(f"{sum(counts.values())}/{len(todo)} {rec['status']:<12} {m.id:<40} fights {rec.get('fights_selected', '-')}"
                      f" first {rec.get('first_kill_pos')} {rec['secs']} s  {dict(counts)}", flush=True)

    ts = [threading.Thread(target=worker, args=(c,)) for c in copies]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return 0


def kill_curve(recs: list, total: int, ns=(50, 100, 250)) -> dict:
    """Share of built mutants a random n of the total fights would kill: the first flagged fight's position
    in each mutant's own random order is at most n.
    """
    built = [r for r in recs if r["status"] in ("killed", "survived")]
    out = {}
    for n in [*ns, total]:
        out[n] = sum(1 for r in built if r["status"] == "killed" and r["first_kill_pos"] <= n) / max(1, len(built))
    return out


def summarize(prep: dict, recs: list) -> dict:
    cands = prep["candidates"]
    total = len(prep["fights"])
    built = [r for r in recs if r["status"] in ("killed", "survived")]

    def rate(rs):
        b = [r for r in rs if r["status"] in ("killed", "survived")]
        return {"built": len(b), "killed": sum(r["status"] == "killed" for r in b),
                "rate": round(sum(r["status"] == "killed" for r in b) / max(1, len(b)), 3)}

    # Uniform over every covered candidate (undo the sqrt(fights) sampling weight), then over all of them.
    w = [(1 / math.sqrt(r["fights"]), r["status"] == "killed") for r in built]
    uni = sum(x for x, k in w if k) / max(1e-9, sum(x for x, _ in w))
    reach = collections.Counter("run" if c["fights"] > 0 else "never" if c["fights"] == 0 else "no probe" for c in cands)
    return {
        "fights": total, "mutants": len(recs), "status": dict(collections.Counter(r["status"] for r in recs)),
        "kill_rate": rate(recs),
        "kill_rate_uniform_over_run_code": round(uni, 3),
        "candidates": dict(reach),
        "kill_rate_all_candidates_est": round(uni * reach["run"] / max(1, reach["run"] + reach["never"]), 3),
        "by_file": {f: rate([r for r in recs if r["file"] == f]) for f in sorted({r["file"] for r in recs})},
        "by_op": {o: rate([r for r in recs if r["op"] == o]) for o in sorted({r["op"] for r in recs})},
        "first_kill_by": dict(collections.Counter(
            "+".join(sorted({k["mode"] for k in r["first_kill"]})) + ":" + "/".join(sorted({k["why"] for k in r["first_kill"]}))
            for r in recs if r["status"] == "killed")),
        "curve": {str(n): round(v, 3) for n, v in kill_curve(recs, total).items()},
        # A survivor whose replay changed somewhere passed both checks on state they do not compare; one whose
        # replay never changed was not exercised by these fights (or is equivalent).
        "survivors_silent": sum(1 for r in recs if r["status"] == "survived" and r.get("silent")),
        "survivors_unchanged": sum(1 for r in recs if r["status"] == "survived" and not r.get("silent")),
        "survivors": [{"id": r["id"], "op": r["op"], "fights": r["fights"], "old": r["old"].strip()[:160],
                       "new": r["new"].strip()[:160], "silent": r.get("silent", 0)} for r in recs if r["status"] == "survived"],
    }


def cmd_report(a) -> int:
    d = Path(a.dir)
    prep = json.loads((d / "prepared.json").read_text(encoding="utf-8"))
    recs = [json.loads(ln) for ln in Path(a.results or d / "results.jsonl").open(encoding="utf-8")]
    s = summarize(prep, recs)
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k != "survivors"}, indent=1))
    print(f"survivors ({len(s['survivors'])}):")
    for r in sorted(s["survivors"], key=lambda r: r["id"]):
        print(f"  {r['id']:<36} fights {r['fights']:>3}  {r['old'][:90]}  ->  {r['new'][:90]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out-json", required=True, help="referee out.json (the recorded fights)")
    p.add_argument("--emu", required=True, help="EmuDump fight output for it from this commit (picks the identical fights)")
    p.add_argument("--dir", required=True)
    p.add_argument("--shards", type=int, default=4)
    r = sub.add_parser("run")
    r.add_argument("--dir", required=True)
    r.add_argument("--reanchor", required=True, help="EmuDump reanchor output for the same fights (baseline verdicts)")
    r.add_argument("--mutants", type=int, default=150)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--shards", type=int, default=2)
    r.add_argument("--minutes", type=float, default=90)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--ids", help="these mutant ids (comma-separated) instead of a sample, e.g. to cross-check schemata")
    c = sub.add_parser("schemata", help="all mutants in one build, long-lived hosts (tools/fidelity/schemata.py)")
    c.add_argument("--dir", required=True)
    c.add_argument("--reanchor", required=True, help="EmuDump reanchor output for the same fights (baseline verdicts)")
    c.add_argument("--mutants", type=int, default=150)
    c.add_argument("--shards", type=int, default=6)
    c.add_argument("--minutes", type=float, default=90)
    c.add_argument("--stall", type=float, default=120, help="seconds without output before a host counts as hung")
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--tag", default="", help="suffix of the schemata<tag> folder in --dir")
    c.add_argument("--max-fights", type=int, default=0, help="only the first n fights of the order (a trial)")
    s = sub.add_parser("report")
    s.add_argument("--dir", required=True)
    s.add_argument("--results", help="results.jsonl to report (default <dir>/results.jsonl)")
    s.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "schemata":
        from .schemata import cmd_schemata  # noqa: PLC0415

        return cmd_schemata(a)
    return {"prepare": cmd_prepare, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
