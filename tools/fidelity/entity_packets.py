"""Entity packets for the logic audit: one small markdown file per entity, so an auditor reads it instead of exploring.

    python -m tools.fidelity.entity_packets ID [ID ...] [--out DIR] [--decompiled DIR] [--src DIR] [--no-mechanisms]

For each id (CARD, RELIC, POWER, MONSTER or potion id in the game's upper-snake form) the packet holds

  1. the game class body from the local decompile;
  2. the bodies it calls that are not generic engine calls, one level deep, capped at CALLEE_CAP lines in total:
     other model classes it names (the power a card applies, the monsters a boss spawns), Hook.* methods, helper
     methods of the class or its base class (bodies <= 40 lines). Calls through *Cmd classes, collections and
     value types are the generic engine and are skipped;
  3. the emulator code that handles the id: every hit of the class name (and its Power-less form, and the id) in the
     emulator source outside Generated, tests and build output, +-30 lines, merged, capped at EMU_CAP lines.

A packet starts with TEMPLATE if the class only does damage, block or one stock power apply (Vulnerable, Weak, Frail, Strength, Dexterity, Poison) and overrides no hook: the
data checks cover its numbers and the audit can skip it. A shared _mechanisms.md (hook listener order, the combat
victory passes, the damage pipeline) is written once per run, not per entity.

The packets hold decompiled game code: the default output is D:/spireline/referee/audit/packets, outside every repo.
Nothing is committed from there. No game process is started.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_DECOMPILED = Path("D:/spireline/referee/decompiled")
DEFAULT_OUT = Path("D:/spireline/referee/audit/packets")
ROOT = Path(__file__).resolve().parents[2]
MODELS = "MegaCrit.Sts2.Core.Models"
KINDS = ("Cards", "Relics", "Powers", "Monsters", "Potions", "Enchantments", "Afflictions", "Orbs")
CALLEE_CAP = 250
EMU_CAP = 250
CTX = 30
GENERIC = set("""ToString Equals GetType GetHashCode Add Remove Contains Count Any All Select Where First Last ToList
ToArray OrderBy Max Min Sum Concat Distinct Cast OfType Math Clamp Floor Ceiling Round Abs Task WhenAll Delay Result
FromResult Empty Format Join Append Insert Clear Sort Reverse Dequeue Enqueue Push Pop Peek TryGetValue Execute
Execute_ Create Clone ToMutable FromModel Static Keyword""".split())

_DECL = r"^\s*(?:\[[^\]]*\]\s*)*(?:public|private|protected|internal)\b[^;=]*?\b{name}\s*(?:<[^>]*>)?\s*\("


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).upper()


def block_end(lines: list[str], start: int) -> int:
    """Index of the last line of the member starting at lines[start] (brace matching, or ';' for expression bodies)."""
    depth, seen = 0, False
    for i in range(start, len(lines)):
        code = re.sub(r'"(?:\\.|[^"\\])*"', '""', lines[i])
        code = code.split("//")[0]
        depth += code.count("{") - code.count("}")
        seen = seen or "{" in code
        if seen and depth <= 0:
            return i
        if not seen and code.rstrip().endswith(";"):
            return i
    return len(lines) - 1


def find_method(lines: list[str], name: str, after: int = 0) -> tuple[int, int] | None:
    rx = re.compile(_DECL.format(name=re.escape(name)))
    for i in range(after, len(lines)):
        if rx.search(lines[i]):
            s = i
            while s > after and lines[s - 1].lstrip().startswith("["):
                s -= 1
            return s, block_end(lines, i)
    return None


def squeeze(lines: list[str]) -> list[str]:
    return [l.rstrip() for l in lines if l.strip() and not l.lstrip().startswith(("using ", "#"))]


class Decomp:
    def __init__(self, root: Path):
        self.root = root
        self.files: dict[str, Path] = {}  # class stem -> file (model kinds first)
        self.kind: dict[str, str] = {}
        for k in KINDS:
            d = root / f"{MODELS}.{k}"
            if d.is_dir():
                for f in d.glob("*.cs"):
                    self.files.setdefault(f.stem, f)
                    self.kind.setdefault(f.stem, k)
        self.by_id: dict[str, list[str]] = {}
        for stem in self.files:
            self.by_id.setdefault(snake(stem), []).append(stem)
        self._cache: dict[Path, list[str]] = {}

    def read(self, p: Path) -> list[str]:
        if p not in self._cache:
            self._cache[p] = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return self._cache[p]

    def locate(self, stem: str) -> Path | None:
        if stem in self.files:
            return self.files[stem]
        for f in self.root.rglob(stem + ".cs"):
            return f
        return None


def class_body(lines: list[str]) -> list[str]:
    for i, l in enumerate(lines):
        if re.search(r"\bclass\s+\w+", l):
            return squeeze(lines[i:block_end(lines, i) + 1]) if not lines[i].lstrip().startswith("//") else []
    return squeeze(lines)


def base_of(lines: list[str]) -> str | None:
    for l in lines:
        m = re.search(r"\bclass\s+\w+\s*:\s*(\w+)", l)
        if m:
            return m.group(1)
    return None


def hook_overrides(body: list[str]) -> list[str]:
    return re.findall(r"public override (?:async )?\w[\w<>?,\s]*?\s(\w+)\(", "\n".join(body))


STOCK_POWERS = {"VulnerablePower", "WeakPower", "FrailPower", "StrengthPower", "DexterityPower", "PoisonPower"}


def is_template(body: list[str], kind: str) -> bool:
    text = "\n".join(body)
    hooks = [h for h in hook_overrides(body) if h not in ("OnPlay", "OnUpgrade")]
    if hooks or re.search(r"\b(if|for|foreach|while|switch)\s*\(", text):
        return False
    if kind == "Cards":
        awaits = re.findall(r"await\s+(\w+Cmd\.\w+|\w+\.\w+)", text)
        applied = set(re.findall(r"<(\w+Power)>", text))
        return (len(awaits) <= 1 and applied <= STOCK_POWERS
                and all(a.split(".")[0] in ("DamageCmd", "CreatureCmd", "PowerCmd") for a in awaits))
    return False


def callees(dec: Decomp, stem: str, path: Path, lines: list[str], body: list[str]) -> list[tuple[str, list[str]]]:
    text = "\n".join(body)
    out: list[tuple[str, list[str]]] = []
    seen = {stem}
    # 1. other model classes named in the body
    for m in re.finditer(r"\b([A-Z]\w+)\b", text):
        n = m.group(1)
        if n in dec.files and n not in seen:
            seen.add(n)
            out.append((f"class {n}", class_body(dec.read(dec.files[n]))))
    # 2. Hook.* methods
    hook = dec.root / "MegaCrit.Sts2.Core.Hooks" / "Hook.cs"
    if hook.exists():
        hl = dec.read(hook)
        for n in dict.fromkeys(re.findall(r"\bHook\.(\w+)\(", text)):
            r = find_method(hl, n)
            if r:
                out.append((f"Hook.{n}", squeeze(hl[r[0]:r[1] + 1])))
    # 3. helper methods of this class or its base (<= 40 lines)
    base = base_of(lines)
    homes = [(path.stem, lines)]  # first entry is the entity itself
    if base and (bp := dec.locate(base)):
        homes.append((base, dec.read(bp)))
    called = [n for n in dict.fromkeys(re.findall(r"(?<![\w.])(?:this\.|base\.)?([A-Z]\w+)\s*\(", text)) if n not in GENERIC]
    for n in called:
        if find_method(lines, n):
            continue  # declared in the entity's own class: already in the class body
        for home, hl in homes[1:]:
            r = find_method(hl, n)
            if r and r[1] - r[0] + 1 <= 40:
                out.append((f"{home}.{n}", squeeze(hl[r[0]:r[1] + 1])))
                break
    return out


def emu_hits(src: Path, terms: list[str]) -> list[tuple[str, int, int, int]]:
    """(file, start, end, hits) windows of +-CTX lines around every term hit, merged per file."""
    rx = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b")
    res = []
    for f in sorted(src.rglob("*.cs")):
        parts = set(f.parts)
        if parts & {"bin", "obj", "Generated"} or "Tests" in "".join(f.parts[-3:]) or f.name.endswith(".g.cs"):
            continue
        if any("Tests" in p for p in f.relative_to(src).parts):
            continue
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        idx = [i for i, l in enumerate(lines) if rx.search(l) and not l.lstrip().startswith("using ")]
        win: list[list[int]] = []
        for i in idx:
            s, e = max(0, i - CTX), min(len(lines) - 1, i + CTX)
            if win and s <= win[-1][1] + 1:
                win[-1][1] = e
                win[-1][2] += 1
            else:
                win.append([s, e, 1])
        for s, e, h in win:
            res.append((str(f.relative_to(src)).replace("\\", "/"), s, e, h))
    return res


def emu_section(src: Path, terms: list[str]) -> tuple[list[str], int]:
    wins = sorted(emu_hits(src, terms), key=lambda w: (-w[3] / (w[2] - w[1] + 1), w[0], w[1]))
    out, used, dropped = [], 0, 0
    for f, s, e, h in wins:
        n = e - s + 1
        if used + n > EMU_CAP:
            room = EMU_CAP - used
            if room < 20:
                dropped += n
                continue
            e = s + room - 1
            n = room
        lines = (src / f).read_text(encoding="utf-8", errors="replace").splitlines()
        out.append(f"// {f}:{s + 1}-{e + 1} ({h} hit{'s' if h != 1 else ''})")
        out += [l.rstrip() for l in lines[s:e + 1]]
        used += n
    if dropped:
        out.append(f"// ... {dropped} more lines of hits not shown (cap {EMU_CAP})")
    return out, len(wins)


def build_packet(dec: Decomp, src: Path, eid: str) -> tuple[str, dict]:
    stems = dec.by_id.get(eid, [])
    if not stems:
        return f"# {eid}\n\nNo game class found for this id in the decompile.\n", {"lines": 0, "template": False, "missing": True}
    stem = stems[0]
    kind = dec.kind[stem]
    path = dec.files[stem]
    lines = dec.read(path)
    body = class_body(lines)
    tmpl = is_template(body, kind)
    cal = callees(dec, stem, path, lines, body)
    parts = [f"# {eid} ({kind[:-1].lower()}, class {stem})"]
    if len(stems) > 1:
        parts.append(f"Other classes share this id: {', '.join(stems[1:])}")
    if tmpl:
        parts.append("TEMPLATE: only damage, block or one power apply, no hook overrides. Numbers are covered by the "
                     "data checks; skip unless the card text says otherwise.")
    parts += ["", "## Game class", "```csharp"] + body + ["```", "", "## Called game code (one level, not generic engine)"]
    used = 0
    for title, b in cal:
        room = CALLEE_CAP - used
        if room <= 5:
            parts.append(f"// {title}: not shown (cap {CALLEE_CAP} lines)")
            continue
        cut = b[:room]
        parts += [f"### {title}", "```csharp"] + cut + (["// ... cut"] if len(b) > room else []) + ["```"]
        used += len(cut)
    short = stem[:-5] if stem.endswith("Power") and len(stem) > 5 else stem
    terms = list(dict.fromkeys([stem, short, eid]))
    emu, nwin = emu_section(src, terms)
    parts += ["", f"## Emulator code naming {', '.join(terms)} (+-{CTX} lines)", "```csharp"] + (emu or ["// no hit"]) + ["```"]
    text = "\n".join(parts) + "\n"
    return text, {"lines": text.count("\n"), "template": tmpl, "missing": False, "emu_windows": nwin}


MECH = [
    ("Hook listener order: CombatState.IterateHookListeners (state machine MoveNext)", "MegaCrit.Sts2.Core.Combat/CombatState.cs",
     "_003CIterateHookListeners_003Ed__69", "MoveNext"),
    ("Hook.IterateCombatHookListeners", "MegaCrit.Sts2.Core.Hooks/Hook.cs", None, "IterateCombatHookListeners"),
    ("Hook.AfterCombatVictory (Early pass, then normal pass)", "MegaCrit.Sts2.Core.Hooks/Hook.cs", None, "AfterCombatVictory"),
    ("Hook.AfterCardExhausted (example of an After hook walk)", "MegaCrit.Sts2.Core.Hooks/Hook.cs", None, "AfterCardExhausted"),
    ("Hook.ModifyDamage", "MegaCrit.Sts2.Core.Hooks/Hook.cs", None, "ModifyDamage"),
    ("Hook.ModifyDamageInternal", "MegaCrit.Sts2.Core.Hooks/Hook.cs", None, "ModifyDamageInternal"),
    ("Hook.ModifyHpLost", "MegaCrit.Sts2.Core.Hooks/Hook.cs", None, "ModifyHpLost"),
    ("Hook.BeforeDamageReceived", "MegaCrit.Sts2.Core.Hooks/Hook.cs", None, "BeforeDamageReceived"),
    ("Hook.AfterDamageGiven", "MegaCrit.Sts2.Core.Hooks/Hook.cs", None, "AfterDamageGiven"),
    ("Hook.AfterDamageReceived", "MegaCrit.Sts2.Core.Hooks/Hook.cs", None, "AfterDamageReceived"),
    ("CreatureCmd.Damage (card/dealer overload)", "MegaCrit.Sts2.Core.Commands/CreatureCmd.cs", None, "Damage"),
]


def mechanisms(dec: Decomp) -> str:
    out = ["# Mechanisms shared by all entities (hook order, victory passes, damage pipeline)",
           "Game code, included once per batch. Callers: read this instead of re-searching.", ""]
    for title, rel, anchor, name in MECH:
        p = dec.root / rel
        if not p.exists():
            out.append(f"## {title}\n(file not found: {rel})\n")
            continue
        lines = dec.read(p)
        after = 0
        if anchor:
            after = next((i for i, l in enumerate(lines) if anchor in l and "class" in l), 0)
        # the first matching overload that has a body; for Damage take the dealer+card overload with CardModel? card
        rx = re.compile(_DECL.format(name=re.escape(name)))
        pick = None
        for i in range(after, len(lines)):
            if rx.search(lines[i]) and (name != "Damage" or "CardModel? card" in lines[i] and "decimal amount" in lines[i]):
                pick = i
                break
        if pick is None:
            out.append(f"## {title}\n(method not found)\n")
            continue
        end = block_end(lines, pick)
        out += [f"## {title}  ({rel}:{pick + 1}-{end + 1})", "```csharp"] + squeeze(lines[pick:end + 1]) + ["```", ""]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--decompiled", type=Path, default=DEFAULT_DECOMPILED)
    ap.add_argument("--src", type=Path, default=ROOT / "src" / "Sts2Emulator")
    ap.add_argument("--no-mechanisms", action="store_true")
    a = ap.parse_args()
    dec = Decomp(a.decompiled)
    a.out.mkdir(parents=True, exist_ok=True)
    skipped = 0
    for eid in a.ids:
        text, info = build_packet(dec, a.src, eid.upper())
        (a.out / f"{eid.upper()}.md").write_text(text, encoding="utf-8")
        skipped += info["template"]
        print(f"{eid.upper()}: {info['lines']} lines" + (" TEMPLATE" if info["template"] else "") + (" NOT FOUND" if info["missing"] else ""))
    if not a.no_mechanisms:
        text = mechanisms(dec)
        (a.out / "_mechanisms.md").write_text(text, encoding="utf-8")
        print(f"_mechanisms.md: {text.count(chr(10))} lines")
    print(f"{skipped} of {len(a.ids)} plain-template (skippable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
