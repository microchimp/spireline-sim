"""Hook table: every (model, hook) override in the game, checked against the emulator's dispatch points.

    python -m tools.fidelity.hook_table game  --decompiled <dir> [--json game_hooks.json]
    python -m tools.fidelity.hook_table check --decompiled <dir> [--json out.json] [--src <engine dir>]
    python -m tools.fidelity.hook_table order --decompiled <dir> --work <scratch> [--json out.json]
    python -m tools.fidelity.hook_table mutants --decompiled <dir> --work <scratch> --run <mutation run dir>

The game dispatches every gameplay hook through ``Hook.cs``: one loop over the combat's (or the run's)
listeners per pass, in a fixed order. ``game`` lists every concrete model that overrides one of
``AbstractModel``'s hook methods (inherited overrides included), with the pass each hook runs in and the
listener order that applies to it. Names and file paths only: no game code is copied.

``check`` maps each (model, hook) to the emulator. The emulator has no listener list; each hook is a
hand-written dispatch point that names the models it serves. ``hook_dispatch.json`` lists, per hook, the
emulator methods that ARE that hook's dispatch point. A pair is

  registered    the model's emulator id is named inside one of those methods (where: file, method, line);
  not modelled  listed in ``hook_unmodelled.json`` with a reason, or matched by one of its rules;
  open          a known candidate bug, listed in ``hook_open.txt`` (a missing override);
  MISSING       none of these. The check exits 1 on any, and on stale list entries.

``order`` instruments a scratch copy of the engine (a probe before every statement of a dispatch method
that names a listener), builds the harness in ``tools/hookprobe`` against it, runs one synthetic combat per
trigger holding every registered listener of the hooks that trigger fires, and compares the order the
probes fire in with the game's listener order (see ``game_order_key``). Mismatches not in the open list
exit 1. ``mutants`` runs both checks on the surviving mutants of a mutation run (tools/fidelity/mutate.py)
and reports how many they kill.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ENGINE = ROOT / "src" / "Sts2Emulator"
DISPATCH = HERE / "hook_dispatch.json"
UNMODELLED = HERE / "hook_unmodelled.json"
OPEN = HERE / "hook_open.txt"
DEFAULT_DECOMPILED = os.environ.get("SPIRELINE_DECOMPILED", r"D:\spireline\referee\decompiled")
FORMAT = 1

# ============================================================================================ the game side

MODELS_NS = "MegaCrit.Sts2.Core.Models"
# The framework base each kind of listener derives from, and the kind it makes.
KIND_OF_BASE = {
    "CardModel": "card", "PowerModel": "power", "RelicModel": "relic", "PotionModel": "potion",
    "OrbModel": "orb", "MonsterModel": "monster", "EnchantmentModel": "enchantment",
    "AfflictionModel": "affliction", "ModifierModel": "modifier", "BadgeModel": "badge",
    "AchievementModel": "achievement", "AbstractModel": "singleton",
}
NOT_HOOKS = frozenset({"CompareTo"})

VIRTUAL = re.compile(r"^\s*public virtual (?!event)[\w<>\[\]?, ()]+?\s+(\w+)\(", re.M)
OVERRIDE = re.compile(
    r"^\s*(?:public|protected|internal)\s+(?:sealed\s+)?override\s+(?:sealed\s+)?(?:async\s+)?"
    r"[\w<>\[\]?, ()]+?\s+(\w+)\(", re.M)
CLASS = re.compile(r"^\s*public\s+(?:sealed\s+|abstract\s+|partial\s+)*class\s+(\w+)(?:<[^>]*>)?\s*(?::\s*([\w.]+))?",
                   re.M)
STATIC_METHOD = re.compile(r"^\t(?:public|private|internal) static (?:async )?[\w<>\[\]?, ().]+?\s+(\w+)\(", re.M)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


@dataclass
class ModelClass:
    name: str
    base: str
    abstract: bool
    rel: str                     # file path relative to the decompiled root
    own: list[str]               # hooks this class overrides itself, in source order


def _body_spans(text: str, starts: list[tuple[int, str]]) -> dict[str, list[str]]:
    """Each named method's body text (braces balanced from the first '{' after its header)."""
    out: dict[str, list[str]] = collections.defaultdict(list)
    for pos, name in starts:
        i = text.find("{", pos)
        depth, j = 0, i
        while j < len(text):
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out[name].append(text[i:j + 1])
    return out


@cache
def hook_names(decompiled: str) -> tuple[str, ...]:
    text = _read(Path(decompiled) / MODELS_NS / "AbstractModel.cs")
    return tuple(h for h in dict.fromkeys(VIRTUAL.findall(text)) if h not in NOT_HOOKS)


@cache
def dispatchers(decompiled: str) -> dict[str, dict]:
    """Hook.cs's static methods: which listener iterator each loops over and the hook passes, in order.

    A public method that calls a private helper (ModifyDamage -> ModifyDamageInternal) gets the helper's
    passes in its own place. ``iterator``: ``combat`` (the combat's listeners, empty once combat is
    ending), ``combat-direct`` (the same list without that guard), ``run+combat`` (the run's deck cards and
    mod subscribers, then the combat's listeners; outside combat the run's relics and potions too) or
    ``run`` (outside combat).
    """
    hooks = set(hook_names(decompiled))
    text = _read(Path(decompiled) / "MegaCrit.Sts2.Core.Hooks" / "Hook.cs")
    starts = [(m.start(), m.group(1)) for m in STATIC_METHOD.finditer(text)]
    publics = {m.group(1) for m in STATIC_METHOD.finditer(text) if m.group(0).lstrip().startswith("public")}
    bodies = _body_spans(text, starts)
    raw: dict[str, list[tuple[str, str]]] = {}
    iters: dict[str, set[str]] = {}
    call = re.compile(r"\b(\w+)\.(\w+)\(|\b(\w+)\(")
    for name, parts in bodies.items():
        seq: list[tuple[str, str]] = []
        it: set[str] = set()
        for body in parts:
            if "IterateCombatHookListeners(" in body:
                it.add("combat")
            if re.search(r"combatState\??\.IterateHookListeners\(\)", body):
                it.add("combat-direct")
            if re.search(r"runState\.IterateHookListeners\((?!null)", body):
                it.add("run+combat")
            if "runState.IterateHookListeners(null)" in body:
                it.add("run")
            for m in call.finditer(body):
                if m.group(2) in hooks and m.group(1) not in ("Hook", "base"):
                    seq.append(("hook", m.group(2)))
                elif m.group(3) in bodies and m.group(3) != name:
                    seq.append(("call", m.group(3)))
        raw[name] = seq
        iters[name] = it

    def flatten(name: str, seen: frozenset) -> tuple[list[str], set[str]]:
        passes: list[str] = []
        it = set(iters.get(name, ()))
        for kind, what in raw.get(name, []):
            if kind == "hook":
                if what not in passes:
                    passes.append(what)
            elif what not in seen and what not in publics:
                sub, sit = flatten(what, seen | {what})
                passes += [p for p in sub if p not in passes]
                it |= sit
        return passes, it

    out = {}
    for name in sorted(publics):
        passes, it = flatten(name, frozenset({name}))
        if passes:
            out[name] = {"iterator": sorted(it), "passes": passes}
    return out


def hook_dispatch(decompiled: str) -> dict[str, dict]:
    """Per hook method: the dispatcher(s) that call it, and its pass index in each."""
    out: dict[str, dict] = {}
    for disp, info in dispatchers(decompiled).items():
        for i, h in enumerate(info["passes"]):
            e = out.setdefault(h, {"dispatchers": [], "iterator": set()})
            e["dispatchers"].append({"dispatcher": disp, "pass": i, "of": len(info["passes"])})
            e["iterator"] |= set(info["iterator"])
    for e in out.values():
        e["iterator"] = sorted(e["iterator"])
    return out


@cache
def model_classes(decompiled: str) -> dict[str, ModelClass]:
    root = Path(decompiled)
    hooks = set(hook_names(decompiled))
    out: dict[str, ModelClass] = {}
    for d in sorted(root.glob(MODELS_NS + "*")):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.cs")):
            text = _read(p)
            m = CLASS.search(text)
            if not m:
                continue
            name, base = m.group(1), (m.group(2) or "").split(".")[-1]
            own = [h for h in dict.fromkeys(OVERRIDE.findall(text)) if h in hooks]
            abstract = bool(re.search(rf"\babstract\s+class\s+{name}\b", text))
            key = name if name not in out else f"{d.name.split('.')[-1]}.{name}"
            out[key] = ModelClass(name, base, abstract, p.relative_to(root).as_posix(), own)
    return out


def kind_of(classes: dict[str, ModelClass], name: str) -> str | None:
    seen = set()
    while name and name not in seen:
        seen.add(name)
        if name in KIND_OF_BASE:
            return KIND_OF_BASE[name]
        c = classes.get(name)
        if c is None:
            return None
        name = c.base
    return None


@dataclass
class Override:
    model: str
    kind: str
    hook: str
    file: str
    declared_in: str             # the class whose source holds the override (a base class when inherited)


def overrides(decompiled: str) -> list[Override]:
    """Every (concrete model, hook) the game dispatches to, inherited overrides included."""
    classes = model_classes(decompiled)
    out = []
    for key, c in sorted(classes.items()):
        if c.abstract or c.name in KIND_OF_BASE:
            continue
        kind = kind_of(classes, c.base)
        if kind is None:
            continue
        chain, cur, seen = [], c, set()
        while cur is not None and cur.name not in KIND_OF_BASE and cur.name not in seen:
            seen.add(cur.name)
            chain.append(cur)
            cur = classes.get(cur.base)
        done = set()
        for cls in chain:
            for h in cls.own:
                if h not in done:
                    done.add(h)
                    out.append(Override(key, kind, h, c.rel, cls.name))
    return out


# The game's listener order (CombatState.IterateHookListeners; RunState.IterateHookListeners wraps it):
# creatures in order, allies then enemies; per creature its powers in application order, then for a
# monster the monster itself, for a player (when active for hooks) its relics in obtain order (melted
# ones skipped), potions by slot, orbs in queue order, then every card of Hand, Draw, Discard, Exhaust and
# Play in pile order, each followed by its affliction and its enchantment; then the combat's modifiers,
# badges and multiplayer scaling model. The run iterator (run+combat) puts the deck's cards and their
# enchantments first. One pass runs the whole list before the next pass (Early, normal, Late) starts.
KIND_RANK = {"power": 0, "monster": 1, "relic": 1, "potion": 2, "orb": 3, "card": 4, "affliction": 4,
             "enchantment": 4, "modifier": 5, "badge": 6, "singleton": 7, "achievement": 8}
PILE_RANK = {"hand": 0, "draw": 1, "discard": 2, "exhaust": 3, "play": 4}


def game_order_key(listener: dict, pass_index: int = 0) -> tuple:
    """The game's position of one listener in one dispatch: (pass, creature, kind, index within kind).

    ``listener``: {"kind", "creature" (0 = the player, then pets, then enemies in roster order),
    "index" (application order for a power, obtain order for a relic, slot for a potion, queue position
    for an orb), "pile" and "index" for a card, with its affliction and enchantment right after it}.
    """
    kind = listener["kind"]
    creature = listener.get("creature", 0)
    if kind in ("modifier", "badge", "singleton", "achievement"):
        creature = 10 ** 6
    sub = listener.get("index", 0)
    if kind in ("card", "affliction", "enchantment"):
        sub = (PILE_RANK.get(listener.get("pile", "hand"), 0), listener.get("index", 0),
               {"card": 0, "affliction": 1, "enchantment": 2}[kind])
    return (pass_index, creature, KIND_RANK.get(kind, 9), sub)


UI_STATEMENT = re.compile(
    r"^(?:await\s+)?(?:base\.)?(?:Flash\w*\([^;]*\)|Cmd\.Wait\([^;]*\)|TaskHelper\.RunSafely\(\w*Visual\w*\(\)\)|"
    r"(?:VfxCmd|SfxCmd|TalkCmd|NGame|NCombatRoom|NRun|NAudio|ThoughtBubble\w*)\b[^;]*|"
    r"InvokeExecutionFinished\(\)|return(?:\s+Task\.CompletedTask)?|return\s+base\.\w+\([^;]*\))\s*;?$")


def override_body(decompiled: str, rel: str, hook: str) -> str:
    """The body of ``hook``'s override in one model file (for classifying it; never copied anywhere)."""
    text = _read(Path(decompiled) / rel)
    m = re.search(rf"override\s+(?:async\s+)?[\w<>\[\]?, ()]+?\s+{hook}\(", text)
    if not m:
        return ""
    p = _close_text(text, text.find("(", m.end() - 1), "(", ")")
    k = p + 1
    while k < len(text) and text[k] in " \t\r\n":
        k += 1
    if text.startswith("=>", k):
        return text[k + 2:text.find(";", k) + 1]
    end = _close_text(text, k, "{", "}")
    return text[k + 1:end]


def _close_text(t: str, i: int, o: str, c: str) -> int:
    depth = 0
    for j in range(i, len(t)):
        if t[j] == o:
            depth += 1
        elif t[j] == c:
            depth -= 1
            if depth == 0:
                return j
    return len(t) - 1


def ui_only(body: str) -> bool:
    """True when the override only gives feedback (a flash, a sound, a wait) and changes no game state.

    Statements are split on ';' and braces after comments are dropped; a guard ``if (...) return ...;`` that
    leaves only feedback is still feedback only.
    """
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", body, flags=re.S)
    if not body.strip():
        return True
    stmts = [s.strip() for s in re.split(r"[;{}]", body) if s.strip()]
    for s in stmts:
        s2 = re.sub(r"^if\s*\(.*\)\s*", "", s, flags=re.S).strip()
        if not s2 or s2 == "else":
            continue
        if not UI_STATEMENT.match(s2 + ";"):
            return False
    return True


RESET_STATEMENT = re.compile(
    r"^(?:base\.)?(\w+)\s*=\s*(?:0m?|false|null|default|RelicStatus\.\w+|new\(\)|\w+\.Normal)$|"
    r"^(?:base\.)?(\w+)\.Clear\(\)$|^(?:InvokeDisplayAmountChanged|UpdateDisplay|RefreshStatus)\(\)$|"
    r"^return(?:\s+Task\.CompletedTask)?$")


def saved_fields(decompiled: str, rel: str) -> set[str]:
    """The model's [SavedProperty] members: state the game keeps from one combat to the next."""
    text = _read(Path(decompiled) / rel)
    return set(re.findall(r"\[SavedProperty[^\]]*\]\s*(?:\[[^\]]*\]\s*)*(?:public|private|protected|internal)\s+"
                          r"[\w<>\[\]?, ]+?\s+(\w+)\s*[{;=]", text))


def reset_only(body: str, saved: set[str]) -> bool:
    """True when the override only puts combat-local fields back to their start values (no game command)
    and none of them is saved between combats. The emulator builds every combat's state fresh, so such an
    end-of-combat reset has nothing to do there."""
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", body, flags=re.S)
    stmts = [s.strip() for s in re.split(r"[;{}]", body) if s.strip()]
    if not stmts:
        return False
    for s in stmts:
        m = RESET_STATEMENT.match(s)
        if not m:
            return False
        name = m.group(1) or m.group(2)
        if name and name in saved:
            return False
    return True


REGISTRY_FILES = frozenset({"AbstractModelSubtypes.cs", "ModelDb.cs"})


@cache
def _mentions(decompiled: str) -> dict[str, set[str]]:
    """For each model class name, the model classes whose source names it (registries left out)."""
    classes = model_classes(decompiled)
    names = {c.name for c in classes.values()}
    word = re.compile(r"\b[A-Z]\w+\b")
    out: dict[str, set[str]] = collections.defaultdict(set)
    root = Path(decompiled)
    for p in root.rglob("*.cs"):
        if p.name in REGISTRY_FILES:
            continue
        text = _read(p)
        owner = p.stem
        for w in set(word.findall(text)) & names:
            if w != owner:
                out[w].add(owner)
    return out


@cache
def multiplayer_cards(decompiled: str) -> frozenset[str]:
    d = Path(decompiled) / f"{MODELS_NS}.Cards"
    return frozenset(p.stem for p in d.glob("*.cs")
                     if "CardMultiplayerConstraint.MultiplayerOnly" in _read(p))


def reach(decompiled: str, name: str) -> str:
    """How game content reaches a power: ``unreachable`` when nothing but the registries names it,
    ``multiplayer`` when every path to it starts at a multiplayer-only card, else ``reachable``."""
    ment = _mentions(decompiled)
    mp = multiplayer_cards(decompiled)
    classes = {c.name: c for c in model_classes(decompiled).values()}
    seen, todo, sources = {name}, [name], set()
    while todo:
        cur = todo.pop()
        for user in ment.get(cur, ()):
            if user in seen:
                continue
            seen.add(user)
            if user in mp:
                sources.add("multiplayer")
            elif user in classes and kind_of(classes, classes[user].base) == "power":
                todo.append(user)              # a power that applies it: follow it to what applies that
            else:
                sources.add("content")
    if not sources:
        return "unreachable"
    return "multiplayer" if sources == {"multiplayer"} else "reachable"


def game_table(decompiled: str) -> dict:
    ov = overrides(decompiled)
    hd = hook_dispatch(decompiled)
    kinds = collections.defaultdict(set)
    for o in ov:
        kinds[o.hook].add(o.kind)
    hooks = {}
    for h in hook_names(decompiled):
        n = sum(1 for o in ov if o.hook == h)
        hooks[h] = {"overrides": n, "kinds": sorted(kinds.get(h, ())),
                    "dispatch": hd.get(h, {"dispatchers": [], "iterator": []})}
    return {"format": FORMAT, "overrides": [asdict(o) for o in ov], "hooks": hooks,
            "dispatchers": dispatchers(decompiled),
            "counts": {"overrides": len(ov), "hooks_overridden": len({o.hook for o in ov}),
                       "hook_methods": len(hook_names(decompiled)), "dispatchers": len(dispatchers(decompiled)),
                       "models": len({o.model for o in ov}),
                       "by_kind": dict(collections.Counter(o.kind for o in ov))}}


# ======================================================================================== the emulator side

CARD_CLASSES = ("IC", "SI", "CL", "ST", "AN", "NB", "RG", "DF")
METHOD_HEAD = re.compile(
    r"^[ \t]*(?:\[[^\]\n]*\][ \t]*)*(?:(?:public|private|internal|protected|static|override|virtual|async|unsafe|"
    r"sealed|extern|new|partial)\s+)+(?!class\b|struct\b|record\b|enum\b|interface\b|delegate\b|const\b|event\b)"
    r"[\w<>\[\]?,.() ]+?\s+(\w+)\s*(?:<[^>()]*>)?\s*\(", re.M)
LOCAL_FUNC = re.compile(r"^[ \t]+(?:static\s+)?(?:void|bool|int|string|var|float|double|decimal|[A-Z][\w<>\[\]?,.]*)"
                        r"\s+(\w+)\s*\([^;{}]*\)\s*(?:\{|=>)", re.M)


@dataclass
class Method:
    file: str                    # relative to the engine root, forward slashes
    name: str
    start: int                   # offset of the header
    body: tuple[int, int]        # offsets of the body, braces (or => ... ;) included
    line: int                    # 1-based line of the header
    field: bool = False          # a field initializer rather than a method


def _masked(text: str) -> str:
    from .mutate import mask
    return mask(text)


def _close(m: str, i: int, open_: str = "{", close: str = "}") -> int:
    depth = 0
    for j in range(i, len(m)):
        c = m[j]
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return j
    return len(m) - 1


KEYWORDS = frozenset({"if", "for", "foreach", "while", "switch", "using", "lock", "catch", "return", "new", "nameof",
                      "typeof", "sizeof", "default", "when", "else", "do", "throw", "await", "base", "this"})


def methods_of(rel: str, text: str, masked: str | None = None) -> list[Method]:
    """Every method (and local function) of one C# file, with its body span."""
    m = masked if masked is not None else _masked(text)
    out = []
    seen = set()
    for rx in (METHOD_HEAD, LOCAL_FUNC):
        for h in rx.finditer(m):
            name = h.group(1)
            if name in KEYWORDS or h.start() in seen:
                continue
            open_paren = m.find("(", h.end(1))
            p = _close(m, open_paren, "(", ")")
            k = p + 1
            while k < len(m) and m[k] in " \t\r\n":
                k += 1
            if m.startswith("where", k):
                cands = [x for x in (m.find("{", k), m.find("=>", k)) if x >= 0]
                k = min(cands) if cands else k
            if m.startswith("=>", k):
                body = (k, m.find(";", k) + 1)
            elif k < len(m) and m[k] == "{":
                body = (k, _close(m, k) + 1)
            else:
                continue                   # an abstract or interface declaration, or not a method at all
            seen.add(h.start())
            out.append(Method(rel, name, h.start(), body, m.count("\n", 0, h.start()) + 1))
    # Field initializers (`static readonly int[] MaxEnergyRelics = [...]`) outside every method: a list of
    # listeners kept in a field is read where the field is, so the field is a member a reference sits in.
    inside = sorted(x.body for x in out)
    for f in FIELD_HEAD.finditer(m):
        pos = f.start()
        i = bisect.bisect_right(inside, (pos, 10 ** 12)) - 1
        if i >= 0 and inside[i][0] <= pos < inside[i][1]:
            continue
        end = _statement_end(m, f.end())
        out.append(Method(rel, f.group(1), pos, (f.end(), end), m.count("\n", 0, pos) + 1, field=True))
    out.sort(key=lambda x: x.start)
    return out


FIELD_HEAD = re.compile(r"^[ \t]+(?:(?:public|private|internal|protected|static|readonly)\s+)+"
                        r"(?!class\b|struct\b|record\b|enum\b|const\b)[\w<>\[\],.? ]+?\s+(\w+)\s*=(?![=>])", re.M)


def _statement_end(m: str, i: int) -> int:
    depth = 0
    for j in range(i, len(m)):
        c = m[j]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ";" and depth <= 0:
            return j + 1
    return len(m)


@dataclass
class Engine:
    root: Path
    files: dict[str, str]                         # rel -> source text
    masked: dict[str, str]
    methods: dict[str, list[Method]]              # rel -> methods in source order
    index: dict[str, list[tuple[str, int]]] = field(default_factory=dict)   # token -> (rel, offset)
    _starts: dict[str, list[int]] = field(default_factory=dict)
    _lines: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path, texts: dict[str, str] | None = None) -> Engine:
        root = Path(root)
        if texts is None:
            texts = {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
                     for p in sorted((root / "Core").rglob("*.cs"))}
        files = dict(texts)
        masked = {r: _masked(t) for r, t in files.items()}
        meths = {r: methods_of(r, files[r], masked[r]) for r in files}
        e = cls(root, files, masked, meths)
        e._build_index()
        return e

    def _build_index(self) -> None:
        """Code tokens (``A.B`` and bare names, over the masked source) and string literals (``"x"``, in code)."""
        idx: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
        chain = re.compile(r"(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
        ident = re.compile(r"[A-Za-z_]\w*")
        bare = re.compile(r"(?<!\w)[A-Za-z_]\w*\b")
        string = re.compile(r'"(\w+)"')
        for rel, m in self.masked.items():
            for x in chain.finditer(m):
                parts = [(p.group(0), x.start() + p.start()) for p in ident.finditer(x.group(0))]
                for (a, _), (b, ob) in zip(parts, parts[1:]):
                    idx[f"{a}.{b}"].append((rel, ob))
            for x in bare.finditer(m):
                idx["@" + x.group(0)].append((rel, x.start()))
                if x.start() == 0 or m[x.start() - 1] != ".":
                    idx["!" + x.group(0)].append((rel, x.start()))
            raw = self.files[rel]
            for x in string.finditer(raw):
                if m[x.start()] == " " and _string_in_code(raw, x.start()):
                    idx['"' + x.group(1)].append((rel, x.start()))
            self._starts[rel] = [mt.body[0] for mt in self.methods[rel]]
            self._lines[rel] = [i for i, c in enumerate(m) if c == "\n"]
        self.index = dict(idx)

    def method_at(self, rel: str, offset: int) -> Method | None:
        """The innermost method whose body holds ``offset`` (methods are in source order, so the latest
        one that starts before it and still holds it is the innermost)."""
        ms = self.methods.get(rel, [])
        starts = self._starts.get(rel)
        if starts is None or len(starts) != len(ms):
            starts = self._starts[rel] = [mt.body[0] for mt in ms]
        i = bisect.bisect_right(starts, offset) - 1
        while i >= 0:
            mt = ms[i]
            if mt.body[0] <= offset < mt.body[1]:
                return mt
            i -= 1
        return None

    def line_of(self, rel: str, offset: int) -> int:
        lines = self._lines.get(rel)
        if lines is None:
            return self.masked[rel].count("\n", 0, offset) + 1
        return bisect.bisect_left(lines, offset) + 1


@cache
def _buff_resolver(engine_root: str):
    """Map a game power class name to the BuffId member that models it (folded powers included)."""
    sys.path.insert(0, str(ROOT / "src"))
    from sts2_gym.live import ids
    from sts2_gym.live import powers as live_powers
    text = _masked((Path(engine_root) / "Core" / "BuffState.cs").read_text(encoding="utf-8"))
    body = text[text.index("enum BuffId"):]
    body = body[body.index("{") + 1:body.index("}")]
    members = [x for x in re.findall(r"^\s*(\w+)\s*(?:=\s*\d+)?\s*,?\s*$", body, re.M)]
    by_entry = {entry: members[o] for o, entry in live_powers.BUFF_ID_TO_LIVE.items() if o < len(members)}
    member_set = set(members)

    def resolve(cls: str) -> str | None:
        live = live_powers.canonical_power(ids.game_entry(cls))
        if live in by_entry:
            return by_entry[live]
        bare = cls[:-5] if cls.endswith("Power") else cls
        return bare if bare in member_set else None

    return resolve


def buff_member(engine_root: Path, power_class: str) -> str | None:
    return _buff_resolver(str(engine_root))(power_class)


def model_tokens(engine: Engine, kind: str, name: str) -> list[str]:
    """The index keys that name this model's emulator id: ``A.B`` code tokens, ``@Name`` bare names and
    ``"Name`` string literals. A bare relic name counts only inside RelicEffects.cs, where it is a const."""
    s = '"' + name
    if kind == "relic":
        return [f"RelicEffects.{name}", f"@Relic{name}", s, f"!{name}"]
    if kind == "power":
        member = buff_member(engine.root, name)
        return ([f"BuffId.{member}"] if member else []) + [s]
    if kind == "card":
        # ``RunConstants.SpoilsMapCard``, ``LanternKeyCard``: the run side's id for a card (quest cards) is ``<Name>Card``.
        return [f"{c}.{name}" for c in CARD_CLASSES] + [s, f"@{name}Card"]
    if kind == "monster":
        return [f"KE.{name}", s]
    if kind == "enchantment":
        return [f"Enchantment.{name}", s]
    if kind == "affliction":
        return [f"Affliction.{name}", f"@Is{name}", s]
    return [s]


@dataclass(frozen=True)
class Ref:
    file: str
    method: str
    line: int
    offset: int


def _relic_consts(engine: Engine, rel: str) -> set[str]:
    key = "__relic_consts__" + rel
    got = engine.index.get(key)
    if got is None:
        names = set(re.findall(r"public const int (\w+) = \d+;", engine.masked[rel]))
        engine.index[key] = [(n, 0) for n in names]
        return names
    return {n for n, _ in got}


def references(engine: Engine, kind: str, name: str) -> list[Ref]:
    """Every place the engine's code names this model (comments and doc text never count)."""
    out = set()
    keys = model_tokens(engine, kind, name)
    for key in keys:
        for rel, off in engine.index.get(key, []):
            if kind == "relic" and key == f"!{name}":
                if not rel.endswith("RelicEffects.cs") or name not in _relic_consts(engine, rel):
                    continue
            masked = engine.masked[rel]
            line_start = masked.rfind("\n", 0, off) + 1
            line = masked[line_start:masked.find("\n", off)]
            if re.match(r"\s*(?:public|private|internal)\s+const\b", line):
                continue
            mt = engine.method_at(rel, off)
            out.add(Ref(rel, mt.name if mt else "", engine.line_of(rel, off), off))
    # A field that holds the model's id (or a list of ids with it) stands for the model wherever it is read.
    for r in list(out):
        mt = engine.method_at(r.file, r.offset)
        if mt is None or not mt.field:
            continue
        for key, same_file in ((f"!{mt.name}", True), (f"{Path(r.file).stem}.{mt.name}", False)):
            for rel, off in engine.index.get(key, []):
                if (same_file and rel != r.file) or (rel == r.file and mt.start <= off < mt.body[1]):
                    continue
                use = engine.method_at(rel, off)
                out.add(Ref(rel, use.name if use else "", engine.line_of(rel, off), off))
    return sorted(out, key=lambda r: (r.file, r.offset))


def _string_in_code(raw: str, pos: int) -> bool:
    """A string literal (masked to blanks) rather than text inside a comment: no comment opens before it."""
    line_start = raw.rfind("\n", 0, pos) + 1
    before = raw[line_start:pos]
    return "//" not in before and not before.lstrip().startswith(("*", "///"))


# ============================================================================================ the mapping

def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def short(model: str) -> str:
    """The class name of a model key (keys of same-named classes carry their namespace: ``Relics.X``)."""
    return model.rsplit(".", maxsplit=1)[-1]


def qual(m: Method) -> str:
    return f"{Path(m.file).stem}.{m.name}"


def citations(engine: Engine, hooks: set[str], models: set[str]) -> dict[tuple[str, str], list[tuple[str, int]]]:
    """Where the engine's comments name a game listener as ``Model.Hook`` (file, offset)."""
    out: dict[tuple[str, str], list[tuple[str, int]]] = collections.defaultdict(list)
    rx = re.compile(r"\b([A-Z]\w+)\.([A-Z]\w+)\b")
    for rel, raw in engine.files.items():
        masked = engine.masked[rel]
        for m in rx.finditer(raw):
            if masked[m.start()] != " ":
                continue
            if m.group(2) in hooks and m.group(1) in models:
                out[m.group(1), m.group(2)].append((rel, m.start()))
    return out


def _owner(engine: Engine, rel: str, offset: int) -> Method | None:
    """The method a comment belongs to: the one it sits in, else the next one when it is that one's doc."""
    mt = engine.method_at(rel, offset)
    if mt is not None:
        return mt
    following = [x for x in engine.methods.get(rel, []) if x.start >= offset]
    if not following:
        return None
    nxt = min(following, key=lambda x: x.start)
    gap = engine.masked[rel][offset:nxt.start]
    return nxt if gap.count("\n") <= 40 and not re.search(r"[;}]\s*$", gap.split("\n")[0]) else None


PRIVATE_HEAD = re.compile(r"\bprivate\s+static\b")


@cache
def _callees(engine_id: int, qualname: str) -> tuple[str, ...]:
    engine = _ENGINES[engine_id]
    stem, _, name = qualname.partition(".")
    out = []
    for rel, ms in engine.methods.items():
        if Path(rel).stem != stem:
            continue
        privates = {m.name for m in ms if not m.field and PRIVATE_HEAD.search(engine.masked[rel][m.start:m.body[0]])}
        for m in ms:
            if m.name != name or m.field:
                continue
            body = engine.masked[rel][m.body[0]:m.body[1]]
            for c in re.findall(r"(?<![\w.])(\w+)\s*\(", body):
                if c in privates and c != name:
                    out.append(f"{stem}.{c}")
    return tuple(dict.fromkeys(out))


_ENGINES: dict[int, Engine] = {}


def callees(engine: Engine, qualname: str) -> tuple[str, ...]:
    """The private helpers of the same class that a dispatch method calls by bare name: the listener's code
    moved out of line (``ApplyAfterCardPlayed`` -> ``MakeRandomHandCardFree``). One level only."""
    _ENGINES[id(engine)] = engine
    return _callees(id(engine), qualname)


def region_spans(engine: Engine, manifest: dict) -> dict[str, tuple[str, int, int] | None]:
    """``File.Method[name]`` -> (file, start, end): the part of that method's body between the region's
    ``from`` and ``to`` anchors (raw source text; a missing ``to`` runs to the end of the body). None when
    the method or an anchor is gone (the check reports it stale)."""
    out: dict[str, tuple[str, int, int] | None] = {}
    for entry, spec in manifest.get("regions", {}).items():
        base = entry.split("[", 1)[0]
        stem, _, name = base.partition(".")
        out[entry] = None
        for rel, ms in engine.methods.items():
            if Path(rel).stem != stem:
                continue
            for m in ms:
                if m.name != name:
                    continue
                raw = engine.files[rel]
                a = raw.find(spec["from"], m.body[0], m.body[1]) if spec.get("from") else m.body[0]
                if a < 0:
                    continue
                b = raw.find(spec["to"], a, m.body[1]) if spec.get("to") else m.body[1]
                if b < 0:
                    continue
                out[entry] = (rel, a, b)
    return out


@dataclass
class Row:
    model: str
    kind: str
    hook: str
    status: str                          # registered | not_modelled | open | missing
    where: list[str] = field(default_factory=list)       # File.Method:line (evidence)
    reason: str = ""
    elsewhere: list[str] = field(default_factory=list)   # methods that name the model at no dispatch point
    sites: list[list] = field(default_factory=list)      # [file, offset] of each naming at a dispatch point


def classify(decompiled: str, engine: Engine, manifest: dict, unmodelled: dict, open_keys: set[str]) -> list[Row]:
    ov = overrides(decompiled)
    hooks = set(hook_names(decompiled))
    classes = model_classes(decompiled)
    by_name = {c.name: c for c in classes.values()}
    cites = citations(engine, hooks, {short(o.model) for o in ov})
    mhooks = manifest.get("hooks", {})
    spans = region_spans(engine, manifest)
    ref_cache: dict[tuple[str, str], list[Ref]] = {}
    rows = []
    for o in ov:
        name = short(o.model)
        key = f"{name}.{o.hook}"
        refs = ref_cache.setdefault((o.kind, name), references(engine, o.kind, name))
        spec = mhooks.get(o.hook, {})
        methods = set(spec.get("methods", []))
        methods |= {c for m in spec.get("methods", []) if "[" not in m for c in callees(engine, m)}
        where, sites = [], []
        for r in refs:
            mt = engine.method_at(r.file, r.offset)
            hit = None
            if mt is not None:
                for entry in methods:
                    if entry == qual(mt):
                        hit = entry
                    elif entry.startswith(qual(mt) + "[") and spans.get(entry) is not None:
                        f, a, b = spans[entry]
                        if f == r.file and a <= r.offset < b:
                            hit = entry
                    if hit:
                        break
            if hit:
                where.append(f"{hit}:{r.line}")
                sites.append([r.file, r.offset])
                continue
            for (cf, co) in cites.get((name, o.hook), []):
                if cf != r.file:
                    continue
                cm = _owner(engine, cf, co)
                if cm is not None and mt is not None and cm.start == mt.start:
                    where.append(f"{qual(mt)}:{r.line}")
                    sites.append([r.file, r.offset])
                    break
        generic = spec.get("generic", {}).get(name)
        if generic:
            where.append(generic)
        where = list(dict.fromkeys(where))
        reason = _unmodelled_reason(decompiled, o, by_name, unmodelled)
        if where:
            status = "registered"
        elif reason:
            status = "not_modelled"
        elif key in open_keys:
            status = "open"
        else:
            status = "missing"
        elsewhere = sorted({f"{qual(mt)}" for r in refs if (mt := engine.method_at(r.file, r.offset)) is not None}) \
            if status in ("open", "missing") else []
        rows.append(Row(o.model, o.kind, o.hook, status, where, reason if status == "not_modelled" else "", elsewhere,
                        sites if status == "registered" else []))
    return rows


def _unmodelled_reason(decompiled: str, o: Override, by_name: dict, um: dict) -> str:
    name = short(o.model)
    pair = um.get("pairs", {}).get(f"{name}.{o.hook}")
    if pair:
        return pair
    model = um.get("models", {}).get(name)
    if model:
        return model
    if o.kind in um.get("kinds", {}):
        return um["kinds"][o.kind]
    if o.hook in um.get("hooks", {}):
        return um["hooks"][o.hook]
    for ns, why in um.get("namespaces", {}).items():
        if o.file.startswith(ns + "/"):
            return why
    if o.kind == "power" and um.get("reach"):
        r = reach(decompiled, name)
        if r in um["reach"]:
            return um["reach"][r]
    decl = by_name.get(o.declared_in)
    if decl is None:
        return ""
    body = override_body(decompiled, decl.rel, o.hook)
    if um.get("ui_only") and ui_only(body):
        return um["ui_only"]
    reset = um.get("combat_reset", {})
    if o.hook in reset.get("hooks", []) and reset_only(body, saved_fields(decompiled, decl.rel)):
        return reset["reason"]
    return ""


def read_open(path: Path = OPEN) -> dict[str, str]:
    """hook_open.txt: ``missing Model.Hook  # note`` and ``order Hook A<B  # note`` lines."""
    out = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        body, _, note = line.partition("#")
        parts = body.split()
        if len(parts) >= 2:
            out[" ".join(parts)] = note.strip()
    return out


def check(decompiled: str, engine: Engine, manifest: dict, unmodelled: dict, open_list: dict[str, str]) -> dict:
    """The mapping, its counts and its problems: unlisted (missing) overrides, stale list entries."""
    open_missing = {k.split(" ", 1)[1] for k in open_list if k.startswith("missing ")}
    rows = classify(decompiled, engine, manifest, unmodelled, open_missing)
    keys = {f"{short(r.model)}.{r.hook}": r for r in rows}
    problems = []
    for r in rows:
        if r.status == "missing":
            problems.append(f"MISSING {short(r.model)}.{r.hook} ({r.kind}): no dispatch point names it"
                            + (f"; named only in {', '.join(r.elsewhere[:4])}" if r.elsewhere else "; named nowhere"))
    for k in sorted(open_missing):
        r = keys.get(k)
        if r is None:
            problems.append(f"STALE open entry {k}: the game has no such override")
        elif r.status != "open":
            problems.append(f"STALE open entry {k}: now {r.status}")
    for k in sorted(unmodelled.get("pairs", {})):
        r = keys.get(k)
        if r is None:
            problems.append(f"STALE not-modelled entry {k}: the game has no such override")
        elif r.status == "registered":
            problems.append(f"STALE not-modelled entry {k}: it is registered at {r.where[0]}")
    names = {qual(m) for ms in engine.methods.values() for m in ms}
    spans = region_spans(engine, manifest)
    for h, spec in manifest.get("hooks", {}).items():
        for m in spec.get("methods", []):
            if "[" in m:
                if spans.get(m) is None:
                    problems.append(f"STALE dispatch point {h}: region {m} is not found (method or anchor gone)")
            elif m not in names:
                problems.append(f"STALE dispatch point {h}: {m} is not a method of the engine")
        if h not in hook_names(decompiled):
            problems.append(f"STALE dispatch hook {h}: not a hook of the game")
    for h, spec in manifest.get("hooks", {}).items():
        for model, why in spec.get("generic", {}).items():
            m = re.match(r"(\w+\.\w+):", why)      # a generic entry names the method that serves it
            if m and m.group(1) not in names:
                problems.append(f"STALE generic entry {h}/{model}: {m.group(1)} is not a method of the engine")
    counts = collections.Counter(r.status for r in rows)
    return {"format": FORMAT, "counts": {"overrides": len(rows), **{k: counts.get(k, 0) for k in
                                                                    ("registered", "not_modelled", "open", "missing")}},
            "problems": problems, "rows": [asdict(r) for r in rows]}


# ================================================================================================ order

@dataclass
class OrderRig:
    """An instrumented engine build and the scenarios to run on it."""
    copy: Path
    probes: dict
    skipped: list
    dropped: set
    scenarios: list
    positions: dict
    listener_of: dict
    present: dict


def prepare_order(engine: Engine, rows: list[Row], work: Path, src_root: Path = ROOT, log=print,
                  build=None) -> OrderRig:
    """Instrument and build the scratch engine; write each trigger's scenario (hook_order.py)."""
    from . import hook_order as ho
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    copy = work / "engine"
    points, skipped, dropped = (build or ho.build_instrumented)(engine, rows, src_root, copy, log=log)
    probes = {p.id: p for p in points.values() if p.id not in dropped}
    scenarios, positions, listener_of, present = [], {}, {}, {}
    for trig, spec in ho.TRIGGERS.items():
        kinds = {a["do"] for a in spec["actions"]}
        avoid = frozenset().union(*(ho.AVOID.get(k, frozenset()) for k in kinds))
        ls = ho.listeners_for(engine, rows, set(spec["hooks"]) - avoid, avoid)
        for x in ls:
            listener_of[x.model, x.kind] = ho._lkey(x)
        for rev in (False, True):
            sc, pos = ho.scenario(engine, trig, ls, rev)
            scenarios.append(sc)
            positions[sc["name"]] = pos
            present[sc["name"]] = len(pos)
    return OrderRig(copy, probes, skipped, dropped, scenarios, positions, listener_of, present)


def judge_order(decompiled: str, rig: OrderRig, results: list[dict]) -> dict:
    """Per mutant (-1: none): the inversion summary, its gate keys, a digest of every pair, harness errors."""
    import hashlib

    from . import hook_order as ho
    out_by_mutant: dict[int, dict] = {}
    for res in results:
        mid = res.get("mutant", -1)
        acc = out_by_mutant.setdefault(mid, {"inversions": [], "errors": {}, "coverage": {}})
        inv, cov = ho.analyse(decompiled, res["log"], rig.probes, rig.positions[res["name"]], rig.listener_of)
        acc["inversions"] += inv
        if res.get("error"):
            acc["errors"][res["name"]] = res["error"]
        acc["coverage"][res["name"]] = {"listeners": rig.present[res["name"]], "seen": len(cov["listeners_seen"]),
                                        "probes_hit": len(cov["probes_hit"])}
    out = {}
    for mid, acc in out_by_mutant.items():
        summary = ho.summarise(acc["inversions"])
        pairs = sorted(map(repr, ho.pair_set(acc["inversions"])))
        out[mid] = {"summary": summary, "keys": sorted(ho.order_keys(summary)), "errors": acc["errors"],
                    "coverage": acc["coverage"], "pairs": len(pairs),
                    "pair_digest": hashlib.sha256(chr(10).join(pairs).encode()).hexdigest()}
    return out


def order(decompiled: str, engine: Engine, rows: list[Row], work: Path, src_root: Path = ROOT, log=print) -> dict:
    """Build the instrumented engine, run every trigger's scenario and judge the order (hook_order.py)."""
    from . import hook_order as ho
    rig = prepare_order(engine, rows, work, src_root, log)
    results = ho.run_harness(rig.copy, rig.scenarios, Path(work))
    return {"probes": len(rig.probes), "dropped": len(rig.dropped), "skipped": rig.skipped,
            "by_mutant": judge_order(decompiled, rig, results)}


# ================================================================================================== CLI

def _engine_root(a) -> Path:
    return Path(a.src) if getattr(a, "src", None) else ENGINE


def cmd_game(a) -> int:
    t = game_table(a.decompiled)
    if a.json:
        Path(a.json).write_text(json.dumps(t, indent=1), encoding="utf-8")
    c = t["counts"]
    print(f"{c['overrides']} (model, hook) overrides of {c['hooks_overridden']} hooks ({c['hook_methods']} hook "
          f"methods, {c['dispatchers']} Hook.cs dispatchers), {c['models']} models; by kind {c['by_kind']}")
    return 0


def _check(a) -> tuple[Engine, dict]:
    engine = Engine.load(_engine_root(a))
    res = check(a.decompiled, engine, load_json(a.dispatch), load_json(a.unmodelled), read_open(a.open))
    return engine, res


def cmd_check(a) -> int:
    _, res = _check(a)
    c = res["counts"]
    print(f"hook table: {c['overrides']} overrides: {c['registered']} registered, {c['not_modelled']} not modelled, "
          f"{c['open']} open (known candidate bugs), {c['missing']} MISSING")
    for p in res["problems"]:
        print("  " + p)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 1 if res["problems"] else 0


def cmd_order(a) -> int:
    engine, res = _check(a)
    rows = [Row(**r) for r in res["rows"]]
    out = order(a.decompiled, engine, rows, Path(a.work))
    base = out["by_mutant"][-1]
    print(f"order: {out['probes']} probes ({out['dropped']} dropped by the compiler, {len(out['skipped'])} sites "
          f"not probeable)")
    for name, cov in base["coverage"].items():
        print(f"  {name}: {cov['seen']} of {cov['listeners']} listeners fired a probe ({cov['probes_hit']} probes)")
    for name, err in base["errors"].items():
        print(f"  {name}: harness error {err}")
    open_order = {k for k in read_open(a.open) if k.startswith("order ")}
    new = [k for k in base["keys"] if k not in open_order]
    stale = sorted(open_order - set(base["keys"]))
    for r in base["summary"]:
        tag = "index" if r["within_kind"] else "commutes" if r["commutes"] else "ORDER"
        print(f"  {tag} {r['segment']} {r['check']}/{r['level']} {r['group']}: {r['emulator_first']} before "
              f"{r['game_first']} ({r['pairs']} pairs), e.g. {'; '.join(r['examples'])}")
    for k in new:
        print(f"  NEW {k}")
    for k in stale:
        print(f"  STALE open entry {k}: no longer seen")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return 1 if new or stale or base["errors"] else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hook_table", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("game", "check", "order", "mutants"):
        p = sub.add_parser(name)
        p.add_argument("--decompiled", default=DEFAULT_DECOMPILED)
        p.add_argument("--json")
        if name != "game":
            p.add_argument("--src", help="engine root (src/Sts2Emulator) to check instead of this checkout's")
            p.add_argument("--dispatch", default=str(DISPATCH))
            p.add_argument("--unmodelled", default=str(UNMODELLED))
            p.add_argument("--open", default=str(OPEN))
        if name in ("order", "mutants"):
            p.add_argument("--work", required=True, help="scratch directory for the instrumented engine")
        if name == "mutants":
            p.add_argument("--run", required=True, help="mutation run directory (tools/fidelity/mutate.py)")
            p.add_argument("--results", help="results.jsonl to read survivors from (default <run>/schemataB)")
            p.add_argument("--no-order", action="store_true", help="the static check only")
    a = ap.parse_args(argv)
    if not Path(a.decompiled).exists():
        print(f"no decompiled game at {a.decompiled} (--decompiled or SPIRELINE_DECOMPILED)")
        return 2
    if a.cmd == "mutants":
        from . import hook_mutants
        return hook_mutants.cmd_mutants(a)
    return {"game": cmd_game, "check": cmd_check, "order": cmd_order}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
