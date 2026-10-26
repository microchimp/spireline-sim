"""Scan the engine for every read of a pile's index or position, for the fight solver's state key.

    python tools/solver/scan_order_reads.py           # write tools/solver/order_reads.json
    python tools/solver/scan_order_reads.py --check   # exit 1 when the scan is stale or a read is unclassed

docs/rebuild/FIGHT_SOLVER.md section 2. The key writes the hand, discard and exhaust piles and the potion slots as
multisets, so it may merge two positions only when nothing reads those piles' order. This scan finds every place that
could: in CardEffects, RelicEffects, PotionEffects, BuffSystem, CombatEngine and EnemyAI, every index or position read
of a pile (``[0]``, ``[^1]``, ``[i]``, ``First``, ``Last``, ``Find``, ``FindIndex``, ``IndexOf``, ``ElementAt``,
``Take``, ``Skip``, ``GetRange``), every walk (``for`` over a pile's count, ``foreach`` over a pile, with or without an
early stop), every index set (``Enumerable.Range`` over a pile's count), every shuffle and every order-keeping copy
(``AddRange``) of a pile, also through a local alias of the pile (one pile, one of two by a condition, or a loop variable over
an array of piles). Each read gets the pile, a kind, the method it is in
and its holders: the card, relic, potion, buff or enemy whose ``case`` label the read sits under, else the holders of
every place the method is named (a call graph over the six files), else ``engine`` (always reachable).

order_readers.json classes the reads: by kind through ``rules`` where the syntax decides (a screen's candidates, a
roll), else per read in ``sites``. KeyFieldCoverageTests fails on an unclassed read, on a classed read the scan no
longer finds, and on a scan older than the engine sources (their sha256 is in order_reads.json). A potion's ``case`` label is its id: the
holder is the name Generated/Potions.g.cs gives that id, so a helper a potion calls is classed by that potion. Every other engine
file whose code names a pile is in ``scope``; order_readers.json ``files`` classes each one the scan does not read out, with why,
and ``--check`` fails on a file it does not know or whose count of such lines changed. Reads of piles the key
writes in order (key_fields.json class ``plain`` or ``regions``) are counted, not classed: the key keeps their order.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "src" / "Sts2Emulator" / "Core"
FILES = {
    "CardEffects": CORE / "Effects" / "CardEffects.cs",
    "RelicEffects": CORE / "Effects" / "RelicEffects.cs",
    "PotionEffects": CORE / "Effects" / "PotionEffects.cs",
    "BuffSystem": CORE / "BuffSystem.cs",
    "CombatEngine": CORE / "CombatEngine.cs",
    "EnemyAI": CORE / "EnemyAI.cs",
}
OUT = ROOT / "tools" / "solver" / "order_reads.json"
READERS = ROOT / "tools" / "solver" / "order_readers.json"
FIELDS = ROOT / "tools" / "solver" / "key_fields.json"
SCOPE = ROOT / "src" / "Sts2Emulator"
POTION_DEFS = SCOPE / "Generated" / "Potions.g.cs"
PILE_ACCESS = re.compile(r"\b(?:Hand|DrawPile|DiscardPile|ExhaustPile|PotionSlots)\b")
HOLDER_KIND = {"CardEffects": "card", "RelicEffects": "relic", "PotionEffects": "potion", "BuffSystem": "buff", "EnemyAI": "enemy", "CombatEngine": "card"}
POS_OPS = r"First|FirstOrDefault|Last|LastOrDefault|Find|FindLast|FindIndex|FindLastIndex|IndexOf|LastIndexOf|ElementAt|ElementAtOrDefault|Take|TakeLast|TakeWhile|Skip|SkipLast|SkipWhile|GetRange"
ROLL = re.compile(r"\.Next\w*\(|\bNextItem\b|\bNextInt\b|\bShuffle\w*\(|\bRoll\w*\(")
DECL = re.compile(
    r"^\s*(?:(?:public|private|internal|protected|static|override|virtual|async|unsafe|extern|partial|new|sealed|readonly)\s+)+"
    r"(?!class\b|struct\b|record\b|enum\b|interface\b|const\b|event\b|delegate\b)[\w<>\[\],.?() ]+?\s+(\w+)\s*(?:<[^>()]*>)?\s*\("
)


def norm(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n")


def blank(src: str) -> tuple[str, str]:
    """(code, bare): comments blanked; comments and string contents blanked. Newlines kept, so lines line up."""
    code, bare = [], []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            code.append(" " * (j - i)); bare.append(" " * (j - i)); i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            seg = re.sub(r"[^\n]", " ", src[i:j]); code.append(seg); bare.append(seg); i = j
        elif c == '"' or (c in "@$" and i + 1 < n and src[i + 1] in '"@$'):
            k = i
            verbatim = False
            while src[k] in "@$":
                verbatim |= src[k] == "@"; k += 1
            j = k + 1
            while j < n:
                if src[j] == "\\" and not verbatim: j += 2; continue
                if src[j] == '"':
                    if verbatim and j + 1 < n and src[j + 1] == '"': j += 2; continue
                    break
                j += 1
            j += 1
            code.append(src[i:j]); bare.append(src[i:k + 1] + re.sub(r"[^\n]", " ", src[k + 1:j - 1]) + '"'); i = j
        elif c == "'" and re.match(r"'(\\.|[^\\'])'", src[i:i + 4]):
            m = re.match(r"'(\\.|[^\\'])'", src[i:i + 4]); code.append(m.group(0)); bare.append("' '"[: len(m.group(0))].ljust(len(m.group(0)))); i += len(m.group(0))
        else:
            code.append(c); bare.append(c); i += 1
    return "".join(code), "".join(bare)


def methods(lines: list[str]) -> list[tuple[str, int, int]]:
    """(name, first line, last line) of each outermost member body, by brace depth."""
    out, depth, cur = [], 0, None
    for ln, text in enumerate(lines):
        if cur is None:
            m = DECL.match(text)
            if m and " = " not in text.split("(")[0]:
                cur = [m.group(1), ln, depth, False]
        if cur is not None:
            for ch in text:
                if ch == "{": depth += 1; cur[3] = True
                elif ch == "}": depth -= 1
            if (cur[3] and depth == cur[2]) or (not cur[3] and text.rstrip().endswith(";")):
                out.append((cur[0], cur[1], ln)); cur = None
        else:
            depth += text.count("{") - text.count("}")
    return out


CASE = re.compile(r"\bcase\s+([^:]+?)\s*:(?!:)")


_POTIONS: dict[int, str] = {}


def potion_names() -> dict[int, str]:
    """The potions' ids and names (Generated/Potions.g.cs): PotionEffects labels its cases by id."""
    if not _POTIONS:
        for m in re.finditer(r'new PotionDef\(Id: (\d+), Name: "(\w+)"', norm(POTION_DEFS.read_bytes()).decode("utf-8")):
            _POTIONS[int(m.group(1))] = m.group(2)
    return _POTIONS


def holder_labels(label: str, kind: str) -> list[str]:
    """The holders a case label names: card, relic, potion, buff or enemy names; [] for any other switch."""
    out = []
    label = re.split(r"\bwhen\b", label)[0]
    for part in re.split(r"\bor\b", label):
        p = part.strip().strip("()")
        if m := re.fullmatch(r'"(\w+)"', p):
            out.append(f"{kind}:{m.group(1)}")
        elif kind == "potion" and re.fullmatch(r"\d+", p) and int(p) in potion_names():
            out.append(f"potion:{potion_names()[int(p)]}")
        elif m := re.fullmatch(r"(?:\w+\.)*IC\.(\w+)", p):
            out.append(f"card:{m.group(1)}")
        elif m := re.fullmatch(r"(?:\w+\.)*BuffId\.(\w+)", p):
            out.append(f"buff:{m.group(1)}")
        elif m := re.fullmatch(r"(?:\w+\.)*(?:RelicEffects|Relics?)\.(\w+)", p):
            out.append(f"relic:{m.group(1)}")
        elif m := re.fullmatch(r"(?:\w+\.)*(?:PotionEffects|Potions?)\.(\w+)", p):
            out.append(f"potion:{m.group(1)}")
    return out


class File:
    def __init__(self, name: str, path: Path):
        self.name = name
        self.src = norm(path.read_bytes()).decode("utf-8")
        code, bare = blank(self.src)
        self.code, self.bare = code.split("\n"), bare.split("\n")
        self.methods = methods(self.bare)
        self.line_method = {}
        for i, (mname, a, b) in enumerate(self.methods):
            for ln in range(a, b + 1):
                self.line_method[ln] = i

    def label_at(self, ln: int) -> list[str] | None:
        """The holders of the nearest enclosing holder case above line ln in its method (None when there is none)."""
        mi = self.line_method.get(ln)
        if mi is None: return None
        start = self.methods[mi][1]
        indent = len(self.bare[ln]) - len(self.bare[ln].lstrip())
        for k in range(ln - 1, start - 1, -1):
            t = self.code[k]
            if not t.strip(): continue
            ind = len(t) - len(t.lstrip())
            if ind < indent and CASE.search(t):
                hs = [h for lab in CASE.findall(t) for h in holder_labels(lab, HOLDER_KIND[self.name])]
                # stacked case labels above share the section
                j = k - 1
                while j >= start and CASE.search(self.code[j]) and len(self.code[j]) - len(self.code[j].lstrip()) == ind:
                    hs += [h for lab in CASE.findall(self.code[j]) for h in holder_labels(lab, HOLDER_KIND[self.name])]
                    j -= 1
                if hs: return hs
                indent = ind
            elif ind < indent:
                indent = ind
        return None


def scan() -> dict:
    fields = json.loads(FIELDS.read_text(encoding="utf-8"))["types"]["CombatState"]["fields"]
    piles = {n: f["class"] for n, f in fields.items() if f["class"] != "excluded" and (re.match(r"(List|IReadOnlyList)<", f["type"]) or f["type"].endswith("[]"))}
    files = {n: File(n, p) for n, p in FILES.items()}
    others = [norm(p.read_bytes()).decode("utf-8") for p in (ROOT / "src" / "Sts2Emulator").rglob("*.cs") if p not in FILES.values() and "obj" not in p.parts and "Solver" not in p.parts]
    by_name: dict[str, list[tuple[str, int]]] = {}
    for f in files.values():
        for i, (m, _, _) in enumerate(f.methods): by_name.setdefault(m, []).append((f.name, i))
    memo: dict[tuple[str, int], frozenset] = {}

    def holders_of(fname: str, mi: int, stack=()) -> frozenset:
        key = (fname, mi)
        if key in memo: return memo[key]
        if key in stack: return frozenset()
        mname, a, b = files[fname].methods[mi]
        hs: set[str] = set()
        pat = re.compile(rf"\b{re.escape(mname)}\b")
        for g in files.values():
            for ln, t in enumerate(g.bare):
                if not pat.search(t): continue
                cm = g.line_method.get(ln)
                if cm is None or (g.name == fname and cm == mi) or (g.methods[cm][1] == ln and g.methods[cm][0] == mname): continue
                lab = g.label_at(ln)
                hs |= set(lab) if lab else holders_of(g.name, cm, stack + (key,))
        if not hs or any(pat.search(o) for o in others): hs.add("engine")
        memo[key] = frozenset(hs)
        return memo[key]

    pn = "|".join(sorted(piles, key=len, reverse=True))
    ref = re.compile(rf"\.({pn})\b")
    sites = []
    for f in files.values():
        for mi, (mname, a, b) in enumerate(f.methods):
            body = "\n".join(f.bare[a:b + 1])
            aliases = {}
            for m in re.finditer(rf"\b(?:var|List<CardInstance>|int\[\])\s+(\w+)\s*=\s*([^;]*?)\.({pn})\b([^;]*);", body):
                rest = m.group(4)
                if re.fullmatch(r"\s*(\.ToList\(\)|\.ToArray\(\)|\.Where\(.*\)(\.ToList\(\)|\.ToArray\(\))?)?\s*\]?\s*(:.*)?", rest, re.S):
                    aliases.setdefault(m.group(1), []).append(m.group(3))
            # a local that is one of two piles by a condition, and a loop variable over an array of piles
            for m in re.finditer(rf"\b(?:var|List<CardInstance>|int\[\])\s+(\w+)\s*=\s*[^;?]*\?\s*[\w.]*\.({pn})\s*:\s*[\w.]*\.({pn})\s*;", body):
                aliases.setdefault(m.group(1), []).extend([m.group(2), m.group(3)])
            for m in re.finditer(r"\bforeach\s*\(\s*(?:var|List<CardInstance>)\s+(\w+)\s+in\s+new\s*(?:List<CardInstance>)?\s*\[\s*\]\s*\{([^}]*)\}", body):
                aliases.setdefault(m.group(1), []).extend(re.findall(rf"\.({pn})\b", m.group(2)))
            loops = []  # (pile, var, first line, last line)
            for ln in range(a, b + 1):
                t = f.bare[ln]
                found = []
                for m in ref.finditer(t): found.append((m.group(1), m.end()))
                for al, ps in aliases.items():
                    for m in re.finditer(rf"(?<![.\w]){al}\b", t):
                        if not re.match(r"\s*=[^=]", t[m.end():]) and not re.search(r"\b(var|List<CardInstance>|int\[\])\s*$", t[:m.start()]):
                            found.extend((p, m.end()) for p in dict.fromkeys(ps))
                for pile, end in found:
                    rest = t[end:]
                    kind = read = None
                    head = t.lstrip()
                    if re.match(r"for\s*\(", head) and re.search(r"\bCount\b|\bLength\b", rest[:20]):
                        var = re.search(r"for\s*\(\s*(?:int|var)\s+(\w+)", head)
                        cond = head.split(";")[1] if head.count(";") >= 2 else ""
                        last = block_end(f.bare, ln)
                        inner = "\n".join(f.bare[ln + 1:last + 1])
                        stop = "&&" in cond or "||" in cond or re.search(r"\b(break|return|goto)\b", inner)
                        kind, read = ("stop" if stop else "walk"), f"for {pile}"
                        if var: loops.append((pile, var.group(1), ln, last))
                    elif re.match(r"foreach\s*\(.*\bin\b", head):
                        last = block_end(f.bare, ln)
                        inner = "\n".join(f.bare[ln + 1:last + 1])
                        kind, read = ("stop" if re.search(r"\b(break|return|goto)\b", inner) else "walk"), f"foreach {pile}"
                    elif m := re.match(r"\s*\[([^\]]*)\]", rest):
                        idx = m.group(1).strip()
                        if re.fullmatch(r"0|\^\d+|[\w.]*Count\s*-\s*\d+|[\w.]*Length\s*-\s*\d+", idx):
                            kind, read = "fixed", f"{pile}[{re.sub(r'[\w.]*(Count|Length)', 'n', idx).replace(' ', '')}]"
                        elif any(v == idx and p == pile and l0 <= ln <= l1 for p, v, l0, l1 in loops):
                            continue  # the walk's own element
                        else:
                            rolled = ROLL.search(idx) or re.search(rf"\b{re.escape(idx)}\s*=[^=;][^;]*?(\.Next\w*\(|\bNextItem\b|\bNextInt\b)", body) if re.fullmatch(r"\w+", idx) else ROLL.search(idx)
                            kind, read = ("roll" if rolled else "index"), f"{pile}[{re.sub(r'\s+', '', idx)}]"
                    elif m := re.match(rf"\s*\.({POS_OPS})\s*\(", rest):
                        kind, read = "fixed", f"{pile}.{m.group(1)}"
                    elif m := re.match(rf"\s*\.(?:Where|Select|OfType|Cast|Distinct|Reverse|ToList|ToArray)\b[^;]*?\.({POS_OPS})\s*\(", rest):
                        kind, read = "fixed", f"{pile}...{m.group(1)}"
                    elif re.search(r"Enumerable\.Range\([^;]*$", t[: end]) and re.match(r"\s*\.(Count|Length)\b", rest):
                        kind, read = ("screen" if re.search(r"\bCandidates\s*=", t) else "range"), f"Range({pile})"
                    elif ROLL.search(t[: end].split("(")[-2] + "(") if t[:end].count("(") >= 1 and re.search(r"\b\w*Shuffle\w*\s*\([^;()]*$", t[:end]) else False:
                        kind, read = "roll", f"Shuffle({pile})"
                    elif re.search(r"\.(AddRange|InsertRange)\s*\(\s*(?:\[\.\.\s*)?[\w.]*$", t[:end]) and not rest.lstrip().startswith((".Where", ".Select", ".OrderBy")):
                        kind, read = "flow", f"AddRange({pile})"
                    if kind is None: continue
                    sites.append({"file": f.name, "method": mname, "pile": pile, "kind": kind, "read": read, "line": ln + 1,
                                  "holders": sorted(holders_of(f.name, mi) if (lab := f.label_at(ln)) is None else lab)})
    seen: dict[str, int] = {}
    for s in sorted(sites, key=lambda s: (s["file"], s["line"])):
        base = f'{s["file"]}:{s["method"]}:{s["read"]}'
        seen[base] = seen.get(base, 0) + 1
        s["id"] = base if seen[base] == 1 else f"{base}#{seen[base]}"
    digest = hashlib.sha256(b"".join(norm(p.read_bytes()) for p in FILES.values())).hexdigest()
    scope = {}  # every engine file whose code (comments and strings blanked) names a pile, with the number of such lines
    for path in sorted(SCOPE.rglob("*.cs")):
        if {"obj", "bin", "Solver"} & set(path.parts) or path.name.endswith(".g.cs"): continue
        n = sum(1 for t in blank(norm(path.read_bytes()).decode("utf-8"))[1].split("\n") if PILE_ACCESS.search(t))
        if n: scope[path.relative_to(ROOT).as_posix()] = n
    ordered = {p: c for p, c in piles.items() if c != "multiset"}
    return {"about": "Written by tools/solver/scan_order_reads.py; do not edit. Every index or position read of a pile in the six engine files; order_readers.json classes those on multiset piles.",
            "sources_sha256": digest, "scope": scope, "multiset_piles": sorted(p for p, c in piles.items() if c == "multiset"), "ordered_piles": ordered,
            "sites": sorted(sites, key=lambda s: s["id"])}


def block_end(lines: list[str], ln: int) -> int:
    """The last line of the statement or block a loop header on line ln governs."""
    depth, opened = 0, False
    for k in range(ln, min(len(lines), ln + 400)):
        t = lines[k] if k > ln else lines[k][lines[k].find(")") + 1 if ")" in lines[k] else 0:]
        for ch in t:
            if ch == "{": depth += 1; opened = True
            elif ch == "}": depth -= 1
        if opened and depth <= 0: return k
        if not opened and k > ln and t.strip().endswith(";"): return k
        if not opened and k == ln and t.strip().endswith(";"): return k
    return ln


def classify(result: dict) -> list[str]:
    """The problems: multiset-pile reads with no class, classes for reads the scan no longer finds."""
    readers = json.loads(READERS.read_text(encoding="utf-8"))
    rules, classed = readers.get("rules", {}), readers.get("sites", {})
    probs = []
    ids = set()
    for s in result["sites"]:
        if s["pile"] not in result["multiset_piles"]: continue
        ids.add(s["id"])
        if s["id"] not in classed and s["kind"] not in rules:
            probs.append(f'unclassed: {s["id"]} ({s["kind"]}, line {s["line"]}, holders {",".join(s["holders"])})')
    for i in classed:
        if i not in ids: probs.append(f"stale: {i}")
    files, scanned = readers.get("files", {}), {f.relative_to(ROOT).as_posix() for f in FILES.values()}
    for rel, n in result["scope"].items():
        if rel in scanned: continue
        if rel not in files: probs.append(f"unclassed file: {rel} ({n} lines name a pile): scan it or class it out in order_readers.json files")
        elif files[rel].get("accesses") != n or not files[rel].get("reason"): probs.append(f"stale file class: {rel} ({n} lines now, classed at {files[rel].get('accesses')})")
    for rel in files:
        if rel not in result["scope"] or rel in scanned: probs.append(f"stale file class: {rel}")
    return probs


def main(argv: list[str]) -> int:
    result = scan()
    text = json.dumps(result, indent=1) + "\n"
    if "--check" in argv:
        have = OUT.read_text(encoding="utf-8").replace("\r\n", "\n") if OUT.exists() else ""
        probs = classify(result) + ([] if have == text else ["order_reads.json is stale: run python tools/solver/scan_order_reads.py"])
        for p in probs: print(p, file=sys.stderr)
        return 1 if probs else 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    ms = [s for s in result["sites"] if s["pile"] in result["multiset_piles"]]
    from collections import Counter
    print(f"scan: {len(result['sites'])} reads, {len(ms)} on multiset piles")
    print(" ", dict(Counter((s["pile"], s["kind"]) for s in ms)))
    print(" ", dict(Counter(s["pile"] for s in result["sites"] if s["pile"] not in result["multiset_piles"])))
    if "--list" in argv:
        for s in ms: print(f'{s["id"]} | {s["kind"]} | {s["line"]} | {",".join(s["holders"])}')
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
