"""The sweep table for one game mechanism: every game caller, and where the emulator does the same.

    python -m tools.fidelity.sweep_table CardCmd.AutoPlay [--depth N] [--out sweep.md] [--keep old_sweep.md] [--check]
    python -m tools.fidelity.sweep_table "CardSelectCmd.From*"       a family of methods
    python -m tools.fidelity.sweep_table PowerInstanceType.Instanced  any identifier

A fix covers a mechanism and all of its callers (the bug-process study: 8 of 10 rejected branches had
siblings half swept). This lists every call of the command (or use of the identifier) in the local
decompile (D:/spireline/referee/decompiled, read only, never committed): the class-qualified name anywhere,
and the bare name inside the command's own class, where the decompile writes it without the class
(CardCmd.DiscardAndDraw calls AutoPlay so). It follows the game's own wrappers: a Commands or Hooks
method that calls it is followed to its callers, and theirs, until no new wrapper turns up (--depth caps
the levels). A call made through a delegate, an interface or a virtual hook (Hook.AfterX calling each
model's AfterX) is not a textual call and is not found. For each caller it lists the emulator sites that
look like the same behaviour:

- the mapping file (data/sweep_map.json): per mechanism, the emulator's implementing methods and the
  terms its sites use, and per caller any site an owner has pinned;
- grep heuristics over src/Sts2Emulator: lines naming the caller (its class name, without a Power,
  Relic, Potion or Card suffix, or its UPPER_SNAKE id) ranked by whether they also use a mechanism term.

Each row also says how good the grep hit is (match: strong when a site names the caller and uses a
mechanism term, weak when it only names the caller, none, or pinned) and which emulator tests name the
caller. The status column is the owner's: converted (the emulator site follows the game's rule), tested
(a test pins it), unreachable (the note says why). --keep carries the statuses and notes of an earlier
table over by caller and call site. --check makes the table a gate: exit 1 while any game caller has no
valid status (or an unreachable one no reason), naming them, and when a kept row's call site is gone.
Only identifiers and line numbers are printed, never game code.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECOMPILED = Path("D:/spireline/referee/decompiled")
EMULATOR = ROOT / "src" / "Sts2Emulator"
TESTS = ROOT / "src" / "Sts2Emulator.Tests"
STATUSES = ("converted", "tested", "unreachable")
COLUMNS = ("#", "game caller", "via", "decompile", "kind", "emulator sites", "match", "tests", "status", "note")
MAP = ROOT / "data" / "sweep_map.json"
WRAPPER_NAMESPACES = ("MegaCrit.Sts2.Core.Commands", "MegaCrit.Sts2.Core.Hooks")
SUFFIXES = ("Power", "Relic", "Potion", "Card", "Enchantment", "Affliction", "Event", "Encounter")
KINDS = {"Models.Cards": "card", "Models.Relics": "relic", "Models.Powers": "power", "Models.Potions": "potion",
         "Models.Monsters": "monster", "Models.Events": "event", "Models.Enchantments": "enchantment",
         "Models.Afflictions": "affliction", "Models.Encounters": "encounter", "Commands": "command",
         "Hooks": "hook", "AutoSlay": "autoslay (test harness)"}
METHOD = re.compile(r"^\s*(?:(?:public|private|protected|internal|static|async|override|virtual|sealed|new|unsafe|extern|partial)\s+)+"
                    r"[\w<>\[\],.? ]+?\s+(\w+)\s*(?:<[^>]*>)?\s*\(")
CLASS = re.compile(r"^\s*(?:(?:public|private|protected|internal|static|sealed|abstract|partial)\s+)*(?:class|struct|record)\s+(\w+)")


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).upper()


def pattern_for(target: str) -> re.Pattern:
    """A call of Class.Method (with generic arguments), a family Class.Prefix*, or a bare identifier."""
    if target.endswith("*"):
        return re.compile(r"\b" + re.escape(target[:-1]) + r"\w*\s*(?:<[^>()]*>)?\s*\(")
    if "." in target and target.split(".")[-1][:1].isupper() and "(" not in target:
        return re.compile(r"\b" + re.escape(target) + r"\b\s*(?:<[^>()]*>)?\s*\(|\b" + re.escape(target) + r"\b")
    return re.compile(r"\b" + re.escape(target) + r"\b")


def own_class_pattern(target: str) -> tuple[str, re.Pattern] | None:
    """(class, pattern) for the calls of Class.Method (or a family Class.Prefix*) inside Class itself, which the
    decompile writes without the class name (``AutoPlay(...)``, ``this.AutoPlay(...)``); None for an identifier.
    The pattern is only applied to lines whose enclosing class is that class (decompile_callers)."""
    if "." not in target or "(" in target:
        return None
    cls, name = target.rsplit(".", 1)
    cls = cls.split(".")[-1]
    if not name[:1].isupper() or not cls[:1].isupper():
        return None
    word = re.escape(name[:-1]) + r"\w*" if name.endswith("*") else re.escape(name) + r"\b"
    return cls, re.compile(r"(?:(?<![\w.])|\bthis\.)" + word + r"\s*(?:<[^>()]*>)?\s*\(")


def is_target(target: str, cls: str, method: str) -> bool:
    """Whether Class.method is the target itself (an overload calling another) or, for a family, one of it."""
    tc, _, name = target.rpartition(".")
    if tc.split(".")[-1] != cls:
        return False
    return method.startswith(name[:-1]) if name.endswith("*") else method == name


def enclosing(lines: list, i: int) -> tuple[str, str]:
    """(class, method) around line i: the nearest method and class declarations above it."""
    method = cls = "?"
    for j in range(i, -1, -1):
        if method == "?":
            m = METHOD.match(lines[j])
            if m and not lines[j].strip().startswith(("if", "while", "for", "return", "await")):
                method = m.group(1)
        c = CLASS.match(lines[j])
        if c:
            cls = c.group(1)
            break
    return cls, method


def kind_of(rel: str) -> str:
    ns = rel.split("/")[0]
    for key, kind in KINDS.items():
        if key in ns:
            return kind
    return ns.replace("MegaCrit.Sts2.Core.", "")


def decompile_callers(decompiled: Path, target: str, depth: int | None = None) -> list:
    """Every call site, followed through wrappers: (depth, via, file, line, namespace kind, class, method).

    A call is the class-qualified name anywhere, or the bare name inside the declaring class (the decompile
    drops the class there: CardCmd.DiscardAndDraw calls AutoPlay). A caller in the game's Commands or Hooks
    namespaces is a wrapper, and its own callers are found the same way, each wrapper once, until no new
    wrapper turns up (or --depth levels). The target's own overloads (a family's own members) calling each
    other are not callers.
    """
    files = sorted(decompiled.rglob("*.cs"))
    texts: dict = {}
    split: dict = {}
    owners: dict = {}

    def full_of(f):
        if f not in texts:
            texts[f] = f.read_text(encoding="utf-8", errors="replace")
        return texts[f]

    def lines_of(f):
        if f not in split:
            split[f] = full_of(f).splitlines()
        return split[f]

    def own_files(cls):
        """The files that declare the class (a partial class has several)."""
        if cls not in owners:
            decl = re.compile(r"\b(?:class|struct|record|interface)\s+" + re.escape(cls) + r"\b")
            owners[cls] = [f for f in files if f.stem == cls or (cls in full_of(f) and decl.search(full_of(f)))]
        return owners[cls]

    rows, seen, frontier, followed = [], set(), [(target, None)], {target}
    level = 0
    while frontier and (depth is None or level < depth):
        level += 1
        nxt = []
        for tgt, via in frontier:
            pat, own = pattern_for(tgt), own_class_pattern(tgt)
            scan = [(f, False) for f in files] + ([(f, True) for f in own_files(own[0])] if own else [])
            needle = tgt.rsplit(".", 1)[-1].rstrip("*")  # in every match: a file without it is skipped
            for f, inside in scan:
                if needle not in full_of(f):
                    continue
                ls = lines_of(f)
                rel = f.relative_to(decompiled).as_posix()
                for i, line in enumerate(ls):
                    if not (own[1] if inside else pat).search(line):
                        continue
                    stripped = line.strip()
                    if stripped.startswith(("//", "///", "using ")):
                        continue
                    cls, method = enclosing(ls, i)
                    if cls != f.stem and f.stem[:1].isalpha():
                        cls = f.stem  # a nested helper class: the model is the file's
                    if inside and cls != own[0]:
                        continue  # the bare name in another class is some other method
                    if is_target(tgt, cls, method) or METHOD.match(line):
                        continue  # the definition itself, its overloads, or a declaration with the same name
                    key = (rel, i + 1)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({"depth": level, "via": via, "file": rel, "line": i + 1, "kind": kind_of(rel),
                                 "caller": f"{cls}.{method}", "class": cls})
                    wrapper = f"{cls}.{method}"
                    if rel.split("/")[0] in WRAPPER_NAMESPACES and method != "?" and wrapper not in followed:
                        followed.add(wrapper)
                        nxt.append((wrapper, wrapper))
        frontier = nxt
    return rows


def emulator_index(root: Path) -> list:
    """(relative path, line number, text) of every engine source line, generated tables excluded."""
    out = []
    for f in sorted(root.rglob("*.cs")):
        if "Generated" in f.parts or "obj" in f.parts or "bin" in f.parts:
            continue
        try:
            rel = f.relative_to(ROOT).as_posix()
        except ValueError:  # a tree outside the checkout (the tests' fixtures)
            rel = f.relative_to(root.parent).as_posix()
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines()):
            out.append((rel, i + 1, line))
    return out


def names_for(cls: str) -> list:
    base = cls
    for suf in SUFFIXES:
        if base.endswith(suf) and len(base) > len(suf):
            base = base[: -len(suf)]
            break
    return list(dict.fromkeys([cls, base, snake(base)]))


def caller_pattern(cls: str) -> re.Pattern | None:
    """The caller's names as the emulator writes them: a PascalCase name (also as a segment of a longer
    identifier, Bombardment in AutoPlayBombardmentsFromExhaust) or the UPPER_SNAKE id as a word."""
    names = [n for n in names_for(cls) if len(n) >= 4]
    if not names:
        return None
    return re.compile("|".join(r"\b" + re.escape(n) + r"\b" if n.isupper() else re.escape(n) for n in names))


def emulator_sites(index: list, cls: str, terms: list, pinned: list, limit: int = 3) -> tuple[list, str]:
    """(sites, match): pinned sites first, then lines naming the caller, those that also use a mechanism term
    first. match is pinned, strong (a site names the caller and uses a term), weak (names it only) or none."""
    if pinned:
        return [f"{p} (pinned)" for p in pinned], "pinned"
    pat = caller_pattern(cls)
    if pat is None:
        return [], "none"
    term = re.compile("|".join(re.escape(t) for t in terms)) if terms else None
    hits = []
    for rel, ln, text in index:
        if pat.search(text):
            strong = bool(term and term.search(text))
            hits.append((not strong, rel, ln))
    hits.sort()
    match = "none" if not hits else ("weak" if hits[0][0] else "strong")
    return [f"{rel}:{ln}" + ("" if not weak else " ~") for weak, rel, ln in hits[:limit]], match


def test_sites(tests: list, cls: str, limit: int = 2) -> list:
    """Emulator test files that name the caller (file:line of the first mention in each)."""
    pat = caller_pattern(cls)
    if pat is None:
        return []
    seen = {}
    for rel, ln, text in tests:
        if rel not in seen and pat.search(text):
            seen[rel] = ln
    return [f"{rel.rsplit('/', 1)[-1]}:{ln}" for rel, ln in list(seen.items())[:limit]]


def keep_statuses(path: Path | None) -> dict:
    """(caller, call site) -> (status, note) from an earlier table, by its header's column names."""
    out = {}
    if not path or not path.exists():
        return out
    cols = None
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == "#":
            cols = {name: i for i, name in enumerate(cells)}
            continue
        if cols and len(cells) == len(cols) and cells[0].isdigit():
            out[cells[cols["game caller"]], cells[cols["decompile"]]] = (
                cells[cols["status"]] if "status" in cols else "", cells[cols["note"]] if "note" in cols else "")
    return out


def site_of(r: dict) -> str:
    return r["file"].split("/")[-1] + f":{r['line']}"


def build_rows(rows: list, index: list, tests: list, mapping: dict, target: str, kept: dict) -> list:
    entry = mapping.get(target, {})
    terms = entry.get("terms", [target.split(".")[-1].rstrip("*")])
    pins = entry.get("callers", {})
    out = []
    for n, r in enumerate(sorted(rows, key=lambda r: (r["depth"], r["kind"], r["caller"], r["file"], r["line"])), 1):
        sites, match = emulator_sites(index, r["class"], terms, pins.get(r["caller"], []))
        status, note = kept.get((r["caller"], site_of(r)), ("", ""))
        out.append({"#": str(n), "game caller": r["caller"], "via": r["via"] or "", "decompile": site_of(r), "kind": r["kind"],
                    "emulator sites": "<br>".join(sites) or "**none found**", "match": match,
                    "tests": "<br>".join(test_sites(tests, r["class"])) or "-", "status": status, "note": note})
    return out


def table(target: str, rows: list, built: list, mapping: dict) -> str:
    entry = mapping.get(target, {})
    terms = entry.get("terms", [target.split(".")[-1].rstrip("*")])
    out = [f"# Sweep: {target}", ""]
    if entry.get("emulator"):
        out.append("Emulator implementation (data/sweep_map.json): " + ", ".join(f"`{x}`" for x in entry["emulator"]))
    out.append(f"Game call sites: {len(rows)} in the decompile (depth up to {max((r['depth'] for r in rows), default=0)}); "
               f"emulator sites by grep over src/Sts2Emulator for the caller's name, those using {', '.join(terms)} first "
               "(`~`: names the caller only). Status: converted / tested / unreachable (the note says why).")
    by_kind = collections.Counter(r["kind"] for r in rows)
    out.append("By kind: " + ", ".join(f"{k} {v}" for k, v in by_kind.most_common()))
    by_match = collections.Counter(b["match"] for b in built)
    done = sum(1 for b in built if b["status"] in STATUSES)
    out.append("Match: " + ", ".join(f"{k} {v}" for k, v in by_match.most_common()) + f"; status filled {done} of {len(built)}")
    out.append("")
    out.append("| " + " | ".join(COLUMNS) + " |")
    out.append("|" + "---|" * len(COLUMNS))
    for b in built:
        out.append("| " + " | ".join(b[c] for c in COLUMNS) + " |")
    return "\n".join(out) + "\n"


def check(built: list, kept: dict) -> list:
    """What keeps the sweep from being complete: callers without a valid status, unreachable without a reason,
    and kept rows whose call site the decompile no longer has."""
    problems = []
    for b in built:
        st = b["status"]
        if st not in STATUSES:
            problems.append(f"{b['game caller']} ({b['decompile']}): "
                            + (f"status '{st}' is not one of {', '.join(STATUSES)}" if st else "no status"))
        elif st == "unreachable" and not b["note"]:
            problems.append(f"{b['game caller']} ({b['decompile']}): unreachable needs a reason in the note")
    now = {(b["game caller"], b["decompile"]) for b in built}
    problems += [f"{c} ({s}): in the kept table, no such call site now" for c, s in kept if (c, s) not in now]
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("target", help="Class.Method, Class.Prefix* or an identifier")
    ap.add_argument("--decompiled", type=Path, default=DECOMPILED)
    ap.add_argument("--emulator", type=Path, default=EMULATOR, help="the engine sources to grep")
    ap.add_argument("--tests", type=Path, default=TESTS, help="the engine's tests to grep")
    ap.add_argument("--map", type=Path, default=MAP)
    ap.add_argument("--depth", type=int, help="follow callers through Commands and Hooks wrappers at most this deep "
                                               "(default: until no new wrapper turns up)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--keep", type=Path, help="an earlier table whose statuses and notes carry over")
    ap.add_argument("--check", action="store_true", help="exit 1 unless every game caller has a valid status (from --keep)")
    a = ap.parse_args(argv)
    if not a.decompiled.exists():
        print(f"no decompile at {a.decompiled}", file=sys.stderr)
        return 2
    mapping = json.loads(a.map.read_text(encoding="utf-8")) if a.map.exists() else {}
    rows = decompile_callers(a.decompiled, a.target, a.depth)
    kept = keep_statuses(a.keep)
    built = build_rows(rows, emulator_index(a.emulator), emulator_index(a.tests) if a.tests.exists() else [], mapping,
                       a.target, kept)
    text = table(a.target, rows, built, mapping)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
        print(f"{len(rows)} game call sites -> {a.out}")
    elif not a.check:
        sys.stdout.write(text)
    if not rows:
        print(f"no game call site of {a.target} in {a.decompiled}", file=sys.stderr)
        return 1
    if a.check:
        problems = check(built, kept)
        for p in problems:
            print("  " + p)
        print(f"sweep check {a.target}: {'FAIL' if problems else 'ok'}: {len(built)} game callers, {len(problems)} problems")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
