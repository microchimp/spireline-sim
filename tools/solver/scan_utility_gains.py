"""Scan the engine for everything that can raise the fight solver's utility within a combat, for its tight bound.

    python tools/solver/scan_utility_gains.py           # write utility_gains.json and Solver/UtilityGains.g.cs
    python tools/solver/scan_utility_gains.py --check   # exit 1 when the scan is stale or a site is unclassed
    python tools/solver/scan_utility_gains.py --list    # print every site with its holders
    python tools/solver/scan_utility_gains.py --check --classes PATH  # the check against another classes file

docs/rebuild/FIGHT_SOLVER.md section 19. The utility L0 (section 10) is 0 for a death, else
0.5 + 0.5 * min(1, (HP + 10 * potions) / max HP). With no heal, no max-HP change and no potion gain able to happen in the
rest of the combat, HP and the potions held only fall and max HP stays, so a node's value is at most L0 of its own HP
(plus what the winning step heals at the combat's end before the terminal is scored) and potions: the solver's tight
bound. This scan lists every place that could break that, in the old engine (every file under Core outside the run's own
code) and in Core2 (the commands, the OldState adapters and every model):

- ``heal``: ``HealPlayer``, Core2's ``CreatureCmd.Heal``;
- ``hp``: every write of ``PlayerHp`` and every call of an HP setter (``SetHp``, ``SetCurrentHp``, ``SetMaxAndCurrentHp``);
- ``max-hp``: every write of ``PlayerMaxHp`` and every max-HP command (``GainMaxHp``, ``LoseMaxHp``, ``SetMaxHp``); a max
  lowered raises L0 as surely as a heal (the ratio's denominator);
- ``potion``: every write into a potion slot, every procure (``TryToProcure``), every change of the slot count;
- ``generate``: every new card (``NewCardUid``) and every screen's generated candidates: a card made in the combat can
  bring a holder the combat did not hold.

Each site gets its holders: the card, relic, potion, buff or enemy whose ``case`` label it sits under, else the holders
of every place its method is named (``scan_order_reads.py``'s call graph over the six engine files); a Core2 model's
holder is the model (``Models/Cards/X.cs`` is card X, ``Powers/XPower.cs`` buff X, and so on). A buff that is a holder
brings in every holder whose code names it (``BuffId.X``, ``XPower``): the cards, potions, relics and enemies that apply
it are holders too (``applies`` entries, written by the scan, no class needed).

utility_gain_classes.json classes each site, with why: ``gain`` (raises the utility; its holders, or the ``holders`` the
class names where the scan's are the engine's, turn the bound off while present in the combat), ``generator`` (makes
cards or potions from a pool that may hold a gain: its holders turn the bound off), ``fixed`` (makes the named cards,
``makes``; none may be a gain holder), ``copy-card`` (copies a card the combat holds: no new card), ``end`` (the
combat's end, with the ``constant`` it heals at most and ``applied``: ``before-terminal`` adds it to the bound,
``after-terminal`` does not), ``loss``, ``primitive`` (a helper's own body: its callers are sites of their own), ``copy``
(a state copy or a re-anchor, outside a step), ``setup`` (building a combat), ``run`` (the run's state, between combats), ``other`` (another creature's HP) or
``safe``. Files outside the scan that name a needle (the run's own code) are classed per file in ``files`` with why and
their count of such lines.

``--check`` fails on a site with no class, a class with no site, a gain or generator with no holder or an engine
holder, a holder that names nothing, a fixed card that is a gain holder, an unknown or changed file, a stale
utility_gains.json or UtilityGains.g.cs, and a scan older than the sources (their sha256 is in utility_gains.json).
UtilityGainTests holds the C# table to the file.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_order_reads as sor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "Sts2Emulator"
OUT = ROOT / "tools" / "solver" / "utility_gains.json"
CLASSES = ROOT / "tools" / "solver" / "utility_gain_classes.json"
CS_OUT = SRC / "Solver" / "UtilityGains.g.cs"
GEN = SRC / "Generated"
BUFFS = SRC / "Core" / "BuffState.cs"

NEEDLES = {
    "heal": re.compile(r"\bHealPlayer\s*\(|\bCreatureCmd\.Heal\s*\(|(?<![\w.])Heal\s*\(\s*k\s*,"),
    "hp": re.compile(r"\bPlayerHp\s*(?:=(?!=)|\+=|-=|\+\+|--)|(?:\+\+|--)\s*[\w.]*\bPlayerHp\b|\b(?:SetHp|SetCurrentHp|SetMaxAndCurrentHp)\s*\("),
    "max-hp": re.compile(r"\bPlayerMaxHp\s*(?:=(?!=)|\+=|-=|\+\+|--)|\b(?:GainMaxHp|LoseMaxHp|SetMaxHp)\s*\("),
    "potion": re.compile(r"\bPotionSlots\s*\[[^\]]*\]\s*=(?!=)|\bTryToProcure\s*\(|\bMaxPotionSlots\s*(?:=(?!=)|\+=|-=|\+\+|--)"),
    "generate": re.compile(r"\bNewCardUid\s*\(|\bGeneratedCandidates\s*=(?!=)"),
}
CLASS_NAMES = ("gain", "generator", "fixed", "copy-card", "end", "loss", "primitive", "copy", "setup", "run", "other", "safe")
HOLDS = ("gain", "generator")
MODEL_KIND = {"Cards": "card", "Relics": "relic", "Potions": "potion", "Powers": "buff", "Monsters": "enemy"}
_labels = sor.holder_labels


def holder_labels(label: str, kind: str) -> list[str]:
    """scan_order_reads' case labels, plus the card-id classes CardEffects labels its cases with (IC, CL, SI, ST, ...)."""
    out = _labels(label, kind)
    if kind == "card":
        for part in re.split(r"or", re.split(r"when", label)[0]):
            m = re.fullmatch(r"(?:\w+\.)*[A-Z]{2,3}\.(\w+)", part.strip().strip("()"))
            if m and f"card:{m.group(1)}" not in out:
                out.append(f"card:{m.group(1)}")
    return out


sor.holder_labels = holder_labels


def label_at(self, ln: int) -> list[str] | None:
    """scan_order_reads' label_at, also when the case's block opens on its own line (``case X:`` then ``{`` at one indent)."""
    mi = self.line_method.get(ln)
    if mi is None: return None
    start = self.methods[mi][1]
    indent = len(self.bare[ln]) - len(self.bare[ln].lstrip())
    for k in range(ln - 1, start - 1, -1):
        t = self.code[k]
        if not t.strip(): continue
        ind = len(t) - len(t.lstrip())
        if ind >= indent: continue
        j = k
        if t.strip() == "{":
            j = k - 1
            while j >= start and not self.code[j].strip(): j -= 1
        hs = []
        while j >= start and sor.CASE.search(self.code[j]) and len(self.code[j]) - len(self.code[j].lstrip()) == ind:
            hs += [h for lab in sor.CASE.findall(self.code[j]) for h in holder_labels(lab, sor.HOLDER_KIND[self.name])]
            j -= 1
        if hs: return hs
        indent = ind
    return None


sor.File.label_at = label_at


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def norm(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n")


def in_scope(p: Path) -> bool:
    """The files whose code runs inside a combat step: Core outside the run's own code (KernelGate is the step's gate), and Core2."""
    parts = p.relative_to(SRC).parts
    if parts[0] == "Core2":
        return True
    if parts[0] != "Core":
        return False
    # KernelGate is the step's gate; RunRewardGenerator holds the combat's-end heals (classed end, with their order).
    return parts[1] != "Run" or p.name in ("KernelGate.cs", "RunRewardGenerator.cs")


def all_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.cs") if not ({"obj", "bin", "Solver"} & set(p.parts)) and not p.name.endswith(".g.cs"))


def names() -> dict[str, set[str]]:
    """The names a holder may take, by kind, from the generated data and the buff enum."""
    out = {}
    for kind, f, pat in (("card", "Cards.g.cs", r"new CardDef\(Id: -?\d+, Name: \"(\w+)\""), ("relic", "Relics.g.cs", r"new RelicDef\(Id: \d+, Name: \"(\w+)\""),
                         ("potion", "Potions.g.cs", r"new PotionDef\(Id: \d+, Name: \"(\w+)\""), ("enemy", "Enemies.g.cs", r"new EnemyDef\(Id: \d+, Name: \"(\w+)\"")):
        out[kind] = set(re.findall(pat, norm((GEN / f).read_bytes()).decode("utf-8")))
    text = norm(BUFFS.read_bytes()).decode("utf-8")
    body = text[text.index("public enum BuffId"):]
    body = body[body.index("{") + 1: body.index("}")]
    out["buff"] = set(re.findall(r"^\s*(\w+)\s*(?:=\s*[^,]+)?,", sor.blank(body)[1], re.M))
    return out


def model_holder(p: Path) -> str | None:
    parts = p.relative_to(SRC).parts
    if len(parts) >= 4 and parts[0] == "Core2" and parts[1] == "Models" and parts[2] in MODEL_KIND:
        kind, stem = MODEL_KIND[parts[2]], p.stem
        if kind == "buff" and stem.endswith("Power"):
            stem = stem[: -len("Power")]
        return f"{kind}:{stem}"
    return None


def scan() -> dict:
    six = {n: sor.File(n, p) for n, p in sor.FILES.items()}
    six_paths = {p: n for n, p in sor.FILES.items()}
    # holders_of: the call graph of scan_order_reads over the six files (a method's holders are its callers' labels).
    # Other files that name a method: a Core2 model is its own holder, any other file the engine's.
    others = [(model_holder(p) or "engine", sor.blank(norm(p.read_bytes()).decode("utf-8"))[1]) for p in SRC.rglob("*.cs")
              if p not in sor.FILES.values() and "obj" not in p.parts and "Solver" not in p.parts]
    memo: dict[tuple[str, int], frozenset] = {}

    def holders_of(fname: str, mi: int, stack=()) -> frozenset:
        key = (fname, mi)
        if key in memo: return memo[key]
        if key in stack: return frozenset()
        mname = six[fname].methods[mi][0]
        hs: set[str] = set()
        pat = re.compile(rf"\b{re.escape(mname)}\b")
        for g in six.values():
            for ln, t in enumerate(g.bare):
                if not pat.search(t): continue
                cm = g.line_method.get(ln)
                if cm is None or (g.name == fname and cm == mi) or (g.methods[cm][1] == ln and g.methods[cm][0] == mname): continue
                lab = g.label_at(ln)
                hs |= set(lab) if lab else holders_of(g.name, cm, stack + (key,))
        hs |= {h for h, o in others if pat.search(o)}
        if not hs: hs.add("engine")
        memo[key] = frozenset(hs)
        return memo[key]

    sites, files, sha = [], {}, hashlib.sha256()
    for path in all_files():
        src = norm(path.read_bytes()).decode("utf-8")
        code, bare = sor.blank(src)
        bl = bare.split("\n")
        hits = [(ln, kind) for ln, t in enumerate(bl) for kind, rx in NEEDLES.items() if rx.search(t) and not sor.DECL.match(t)]
        if not in_scope(path):
            if hits:
                files[rel(path)] = len({ln for ln, _ in hits})
            continue
        sha.update(rel(path).encode()); sha.update(norm(path.read_bytes()))
        f = six.get(six_paths.get(path, ""))
        meths = f.methods if f else sor.methods(bl)
        line_method = {ln: i for i, (_, a, b) in enumerate(meths) for ln in range(a, b + 1)}
        mh = model_holder(path)
        for ln, kind in hits:
            mi = line_method.get(ln)
            mname = meths[mi][0] if mi is not None else "?"
            if f is not None:
                lab = f.label_at(ln)
                holders = sorted(lab if lab else (holders_of(f.name, mi) if mi is not None else {"engine"}))
            else:
                holders = [mh] if mh else ["engine"]
            sites.append({"file": rel(path), "method": mname, "kind": kind, "line": ln + 1, "holders": holders,
                          "text": re.sub(r"\s+", " ", bl[ln].strip())[:160]})
    seen: dict[str, int] = {}
    for s in sorted(sites, key=lambda s: (s["file"], s["line"], s["kind"])):
        base = f'{Path(s["file"]).stem}:{s["method"]}:{s["kind"]}'
        seen[base] = seen.get(base, 0) + 1
        s["id"] = base if seen[base] == 1 else f"{base}#{seen[base]}"
    return {"sources_sha256": sha.hexdigest(), "files": files, "sites": sorted(sites, key=lambda s: s["id"])}


def applies(buffs: set[str]) -> dict[str, list[str]]:
    """For each buff holder, the holders of every line that names it (BuffId.X in the six files, XPower in Core2's models)."""
    six = {n: sor.File(n, p) for n, p in sor.FILES.items()}
    out = {}
    unlabeled: dict[str, list[str]] = {}
    for b in sorted(buffs):
        hs: set[str] = set()
        pat = re.compile(rf"\bBuffId\.{re.escape(b)}\b")
        for f in six.values():
            for ln, t in enumerate(f.bare):
                if pat.search(t):
                    lab = f.label_at(ln)
                    if lab:
                        hs |= set(lab)
                    else:
                        # no case label: the line acts where the buff is held (its turn-end or hit hook) or at the
                        # combat's setup; listed, so a reviewer sees each one
                        unlabeled.setdefault(f"applies:buff:{b}", []).append(f"{f.name}:{ln + 1}")
        mp = re.compile(rf"\b{re.escape(b)}Power\b")
        for p in sorted((SRC / "Core2" / "Models").rglob("*.cs")):
            if mp.search(sor.blank(norm(p.read_bytes()).decode("utf-8"))[0]) and (h := model_holder(p)):
                hs.add(h)
        hs.discard(f"buff:{b}")
        hs.discard("engine")
        out[f"applies:buff:{b}"] = {"holders": sorted(hs), "unlabeled_lines": unlabeled.get(f"applies:buff:{b}", [])}
    return out


def build(classes_path: Path | None = None) -> tuple[dict, list[str]]:
    s = scan()
    doc_classes = json.loads((classes_path or CLASSES).read_text(encoding="utf-8"))
    classes, fclasses = doc_classes["sites"], doc_classes.get("files", {})
    known = names()
    probs: list[str] = []
    out = {}

    def resolves(h: str) -> bool:
        k, _, n = h.partition(":")
        return k in known and n in known[k]

    for e in s["sites"]:
        c = classes.get(e["id"])
        if c is None:
            probs.append(f'unclassed: {e["id"]} ({e["file"]}:{e["line"]}, holders {",".join(e["holders"])}): {e["text"]}')
            continue
        cls = c.get("class")
        if cls not in CLASS_NAMES or not c.get("why"):
            probs.append(f"bad class for {e['id']}: needs a class ({'|'.join(CLASS_NAMES)}) and a why")
            continue
        holders = c.get("holders", e["holders"]) if cls in HOLDS else []
        if cls in HOLDS:
            if not holders or "engine" in holders:
                probs.append(f"{cls} {e['id']} has holders {holders}: name them in the class (holders), with why")
            for h in holders:
                if h != "engine" and not resolves(h):
                    probs.append(f"{cls} {e['id']}: holder {h} names nothing")
        if cls == "end" and (not isinstance(c.get("constant"), int) or c.get("applied") not in ("before-terminal", "after-terminal") or not c.get("holder")):
            probs.append(f"end {e['id']}: needs holder, an int constant and applied (before-terminal|after-terminal)")
        if cls == "fixed" and not c.get("makes"):
            probs.append(f"fixed {e['id']} names no card it makes (makes)")
        out[e["id"]] = {**{k: v for k, v in e.items() if k != "text"}, **c, **({"holders": holders} if cls in HOLDS else {})}
    for key in classes:
        if key not in {e["id"] for e in s["sites"]}:
            probs.append(f"stale class: {key} is no longer found by the scan")
    for f, n in s["files"].items():
        fc = fclasses.get(f)
        if fc is None:
            probs.append(f"unclassed file: {f} ({n} lines name a needle) is outside the scan: class it in files with why")
        elif fc.get("lines") != n or not fc.get("why"):
            probs.append(f"stale file class: {f} ({n} lines now, classed at {fc.get('lines')})")
    for f in fclasses:
        if f not in s["files"]:
            probs.append(f"stale file class: {f}")
    gain = sorted({h for v in out.values() if v["class"] == "gain" for h in v["holders"]})
    gen = sorted({h for v in out.values() if v["class"] == "generator" for h in v["holders"]})
    app = applies({h.split(":", 1)[1] for h in gain + gen if h.startswith("buff:")})
    for k, a in app.items():
        for h in a["holders"]:
            if not resolves(h):
                probs.append(f"{k}: holder {h} names nothing")
    holders = sorted(set(gain) | set(gen) | {h for a in app.values() for h in a["holders"]})
    for k, v in out.items():
        if v["class"] == "fixed":
            for card in v["makes"]:
                if not resolves(f"card:{card}"):
                    probs.append(f"fixed {k}: card {card} names nothing")
                if f"card:{card}" in holders:
                    probs.append(f"fixed {k} makes {card}, a gain holder: class it generator")
    ends = sorted({(v["holder"], v["constant"], v["applied"]) for v in out.values() if v["class"] == "end"})
    doc = {
        "about": "Written by tools/solver/scan_utility_gains.py; do not edit. Every site that can raise the solver's utility within a combat (heal, HP, max HP, potion, generation), classed in utility_gain_classes.json; the holders turn the tight bound off while present (FIGHT_SOLVER.md section 19).",
        "sources_sha256": s["sources_sha256"],
        "holders": holders,
        "end": [{"holder": h, "constant": c, "applied": a} for h, c, a in ends],
        "applies": app,
        "files": s["files"],
        "sites": out,
    }
    return doc, probs


def render_cs(doc: dict) -> str:
    by = {k: [h.split(":", 1)[1] for h in doc["holders"] if h.startswith(k + ":")] for k in ("card", "relic", "potion", "buff", "enemy")}

    def arr(xs: list[str]) -> str:
        return "[" + ", ".join(f'"{x}"' for x in xs) + "]"

    ends = ", ".join(f'("{e["holder"].split(":", 1)[1]}", {e["constant"]}, {"true" if e["applied"] == "before-terminal" else "false"})' for e in doc["end"])
    return (
        "// <auto-generated> Written by tools/solver/scan_utility_gains.py from utility_gains.json; do not edit. </auto-generated>\n"
        "namespace Sts2Emulator.Solver;\n\n"
        "/// <summary>The utility-gain holders by kind (FIGHT_SOLVER.md section 19): with none of them in the combat, no heal, max-HP\n"
        "/// change or potion gain can happen in it.</summary>\n"
        "internal static partial class UtilityGains\n{\n"
        f"    internal static readonly string[] CardNames = {arr(by['card'])};\n\n"
        f"    internal static readonly string[] RelicNames = {arr(by['relic'])};\n\n"
        f"    internal static readonly string[] PotionNames = {arr(by['potion'])};\n\n"
        f"    internal static readonly string[] BuffNames = {arr(by['buff'])};\n\n"
        f"    internal static readonly string[] EnemyNames = {arr(by['enemy'])};\n\n"
        "    /// <summary>The combat's-end heals: the relic, the most it heals, and whether the winning step applies it before the\n"
        "    /// terminal is scored (only those are added to the bound).</summary>\n"
        f"    internal static readonly (string Relic, int Constant, bool BeforeTerminal)[] EndHeals = [{ends}];\n"
        "}\n"
    )


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1) + "\n"


def main(argv: list[str]) -> int:
    # --classes PATH: check against another classes file (the registry test removes a class and expects the check to fail)
    alt = Path(argv[argv.index("--classes") + 1]) if "--classes" in argv else None
    doc, probs = build(alt)
    if "--list" in argv:
        for k, v in doc["sites"].items():
            print(f'{k} | {v["class"]} | {",".join(v.get("holders", []))}')
    if "--check" in argv:
        if not OUT.exists() or OUT.read_text(encoding="utf-8").replace("\r\n", "\n") != render(doc):
            probs.append(f"{OUT.name} is stale: run python tools/solver/scan_utility_gains.py")
        if not CS_OUT.exists() or CS_OUT.read_text(encoding="utf-8").replace("\r\n", "\n") != render_cs(doc):
            probs.append(f"{CS_OUT.name} is stale: run python tools/solver/scan_utility_gains.py")
        for p in probs:
            print(p, file=sys.stderr)
        return 1 if probs else 0
    for p in probs:
        print(p, file=sys.stderr)
    OUT.write_text(render(doc), encoding="utf-8", newline="\n")
    CS_OUT.write_text(render_cs(doc), encoding="utf-8", newline="\n")
    print(f"wrote {len(doc['sites'])} sites, {len(doc['holders'])} holders, end heals {doc['end']}")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
