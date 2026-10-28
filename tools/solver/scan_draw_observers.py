"""Scan the engine for everything that acts on drawn cards, for the fight solver's macro draw.

    python tools/solver/scan_draw_observers.py           # write tools/solver/draw_observers.json
    python tools/solver/scan_draw_observers.py --check   # exit 1 when the scan is stale or an entry is unclassed

docs/rebuild/FIGHT_SOLVER.md section 17. The macro draw takes a run of plain draws from the unknown middle as one chance
node over the sub-multisets drawn. That is exact only when nothing acts per drawn card in a way that depends on the order
the cards came in, and the count drawn does not depend on their values. This scan lists every place that could:

- ``draw-loop``: every method the draw loop of ``CardEffects.DrawCards`` calls per card (the per-card hooks);
- ``draw-result``: every caller that reads a draw's result list (what it drew, in order), by its ``case`` label or method;
- ``kernel-hook``: the kernel's own hook ids for a drawn card (any listener a switched-on family registers there);
- ``hook``: every game hook ``X.AfterCardDrawn`` / ``X.AfterCardDrawnEarly`` the engine sources name (the old engine's
  rules cite the game's listener they carry), and every Core2 model that registers a listener on either hook.

draw_observer_classes.json classes each entry: ``observer`` (with the name of the solver's test in
``Solver/MacroDraw.cs`` ``DrawObservers.All`` that turns the macro draw off while it can act) or ``safe``, each with why.
``--check`` fails on an entry with no class, a class with no entry, an observer name the C# table does not hold, and a scan
older than the sources (their sha256 is in draw_observers.json). DrawObserverTests holds the C# table to the observers here.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "Sts2Emulator"
CARD_EFFECTS = SRC / "Core" / "Effects" / "CardEffects.cs"
OUT = ROOT / "tools" / "solver" / "draw_observers.json"
CLASSES = ROOT / "tools" / "solver" / "draw_observer_classes.json"
MACRO_CS = SRC / "Solver" / "MacroDraw.cs"
CALL = re.compile(r"\b([A-Za-z_]\w*)\s*(?:<[^<>()]*>)?\s*\(")
HOOK_IDS = SRC / "Core2" / "HookId.g.cs"
HOOK_ID = re.compile(r"^\s*(AfterCardDrawn(?:Early)?),\s*$")
NEW = re.compile(r"\bnew\s+[\w.<>]+\s*\(")
HOOK_NAME = re.compile(r"\b(\w+)\.(AfterCardDrawn(?:Early)?)\b")
HOOK_ON = re.compile(r"\.On\(\s*HookId\.(AfterCardDrawn(?:Early)?)\b")
RESULT_CALL = re.compile(r"\bDrawCards\(\s*state\s*,[^;]*?,\s*rng\s*,\s*(?!drawn:\s*null)(\w+)\s*[,)]")
CASE = re.compile(r"^\s*case\s+(?:\w+\.)?(\"?)(\w+)\1\s*(?:when[^:]*)?:")
DECL = re.compile(
    r"^\s*(?:(?:public|private|internal|protected|static|override|virtual|async|unsafe|extern|partial|new|sealed|readonly)\s+)+"
    r"(?!class\b|struct\b|record\b|enum\b|interface\b|const\b|event\b|delegate\b)[\w<>\[\],.?() ]+?\s+(\w+)\s*(?:<[^>()]*>)?\s*\("
)
KEYWORDS = {"if", "for", "foreach", "while", "switch", "return", "catch", "using", "lock", "nameof", "typeof", "sizeof", "when"}


def norm(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n")


def strip_comment(line: str) -> str:
    i = line.find("//")
    return line if i < 0 else line[:i]


def sources() -> list[Path]:
    core = sorted((SRC / "Core").rglob("*.cs"))
    models = sorted((SRC / "Core2" / "Models").rglob("*.cs"))
    return core + models


def draw_loop_calls(text: str) -> list[str]:
    """The methods the draw loop calls per card: the `for` loop of the DrawCards overload that takes the result list."""
    lines = text.split("\n")
    start = next(i for i, ln in enumerate(lines) if re.match(r"\s*internal static void DrawCards\($", ln))
    loop = next(i for i in range(start, len(lines)) if re.match(r"\s*for \(int i = 0; i < count; i\+\+\)", lines[i]))
    depth, body, i = 0, [], loop + 1
    while True:
        code = strip_comment(lines[i])
        depth += code.count("{") - code.count("}")
        body.append(code)
        if depth == 0 and "}" in code:
            break
        i += 1
    names = []
    for code in body:
        code = NEW.sub("(", code)
        for m in CALL.finditer(code):
            if m.group(1) not in KEYWORDS and m.group(1) not in names:
                names.append(m.group(1))
    return names


def holder(lines: list[str], at: int) -> str:
    """The `case` label the line sits under, else the method it is in."""
    for i in range(at, -1, -1):
        m = CASE.match(lines[i])
        if m:
            return m.group(2)
        m = DECL.match(lines[i])
        if m:
            return m.group(1)
    return "?"


def scan() -> dict:
    entries: dict[str, dict] = {}
    text = norm(CARD_EFFECTS.read_bytes()).decode("utf-8")
    for name in draw_loop_calls(text):
        entries[f"draw-loop:{name}"] = {"kind": "draw-loop", "holder": name, "file": rel(CARD_EFFECTS)}
    for path in sources():
        src = norm(path.read_bytes()).decode("utf-8")
        lines = src.split("\n")
        core2 = "Core2" in path.parts
        for n, line in enumerate(lines):
            if core2:
                m = HOOK_ON.search(line)
                if m:
                    entries.setdefault(f"hook:{path.stem}", {"kind": "hook", "holder": path.stem, "file": rel(path), "hooks": []})
                    hooks = entries[f"hook:{path.stem}"]["hooks"]
                    if m.group(1) not in hooks:
                        hooks.append(m.group(1))
                continue
            for m in HOOK_NAME.finditer(line):
                key = f"hook:{m.group(1)}"
                e = entries.setdefault(key, {"kind": "hook", "holder": m.group(1), "file": rel(path), "hooks": []})
                if m.group(2) not in e["hooks"]:
                    e["hooks"].append(m.group(2))
            code = strip_comment(line)
            if RESULT_CALL.search(code) and not DECL.match(line):
                h = holder(lines, n)
                entries[f"draw-result:{h}"] = {"kind": "draw-result", "holder": h, "file": rel(path)}
    for line in norm(HOOK_IDS.read_bytes()).decode("utf-8").split("\n"):
        m = HOOK_ID.match(line)
        if m:
            entries[f"kernel-hook:{m.group(1)}"] = {"kind": "kernel-hook", "holder": m.group(1), "file": rel(HOOK_IDS)}
    for e in entries.values():
        if "hooks" in e:
            e["hooks"].sort()
    sha = hashlib.sha256()
    for path in [*sources(), HOOK_IDS]:
        sha.update(rel(path).encode())
        sha.update(norm(path.read_bytes()))
    return {"entries": dict(sorted(entries.items())), "sources_sha256": sha.hexdigest()}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def csharp_observers() -> list[str]:
    text = norm(MACRO_CS.read_bytes()).decode("utf-8")
    block = text[text.index("public static readonly (string Name, Func<CombatState, bool> On)[] All"):]
    block = block[: block.index("];")]
    return re.findall(r'\(\s*"([^"]+)"\s*,', block)


def build() -> tuple[dict, list[str]]:
    s = scan()
    classes = json.loads(CLASSES.read_text(encoding="utf-8"))["classes"]
    problems = []
    out = {}
    for key, e in s["entries"].items():
        c = classes.get(key)
        if c is None:
            problems.append(f"unclassed: {key} ({e['file']}): class it in {CLASSES.name} as observer (with its test's name) or safe, with why")
            continue
        if c.get("class") not in ("observer", "safe") or not c.get("why"):
            problems.append(f"bad class for {key}: needs class observer|safe and a why")
        if c.get("class") == "observer" and not c.get("observer"):
            problems.append(f"observer {key} names no test (observer: <DrawObservers.All name>)")
        out[key] = {**e, **c}
    for key in classes:
        if key not in s["entries"]:
            problems.append(f"stale class: {key} is no longer found by the scan")
    named = sorted({c["observer"] for c in out.values() if c.get("class") == "observer" and c.get("observer")})
    cs = csharp_observers()
    for n in named:
        if n not in cs:
            problems.append(f"observer {n} has no test in DrawObservers.All")
    for n in cs:
        if n not in named:
            problems.append(f"DrawObservers.All holds {n}, which no entry here names")
    doc = {
        "about": "Written by tools/solver/scan_draw_observers.py; do not edit. Everything that acts on drawn cards (the draw loop's per-card calls, the draw-result readers, the AfterCardDrawn listeners), each classed in draw_observer_classes.json: an observer turns the macro draw off while its test in DrawObservers.All holds (FIGHT_SOLVER.md section 17).",
        "sources_sha256": s["sources_sha256"],
        "observers": named,
        "entries": out,
    }
    return doc, problems


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1, sort_keys=False) + "\n"


def main() -> int:
    check = "--check" in sys.argv[1:]
    doc, problems = build()
    if check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != render(doc):
            problems.append(f"{OUT.name} is stale: run tools/solver/scan_draw_observers.py")
        for p in problems:
            print(p, file=sys.stderr)
        return 1 if problems else 0
    for p in problems:
        print(p, file=sys.stderr)
    OUT.write_text(render(doc), encoding="utf-8", newline="\n")
    print(f"wrote {len(doc['entries'])} entries, observers {doc['observers']}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
