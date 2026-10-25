"""``construct commands``: the migrated Core2 commands against the game's methods, and the HP/block write scan.

Per command (``COMMANDS``): the game method's call sequence, extracted from the local decompile (every ``Hook.X`` call and
the command's named callees, in source order, each with its branch label: the enclosing ``if``/``else``/``foreach``/...
blocks, ``<kind>-cond`` for a call inside a block header, ``cond`` for one after ``&&``, ``||`` or ``?:``; listed members
inlined), equals the sequence the Core2 command declares in its source::

    // construct: game CreatureCmd.Kill
    //   foreach            KillWithoutCheckingWinCondition
    //   if/if              LoseCombat
    //   else/foreach/if/if PlayerCmd.EndTurn ! multiplayer only: needs a second player
    // end construct

both ways (label and name), and the declared names that are not marked ``!`` (not modelled, with the reason) equal the
calls extracted from the Core2 method (``Hooks.X`` but the ``Resume`` continuations, and the command's mapped callees),
both ways. A declaration missing, a call only on one side, a label or an order that differs fails.

The write scan lists every write of an HP or block field of the old state (``PlayerHp``, ``PlayerMaxHp``, ``PlayerBlock``,
``OstyHp``, ``OstyMaxHp``, ``OstyBlock``, an enemy's ``.Hp``, ``.MaxHp``, ``.Block``) in ``Core/`` that runs with the F08+F10
switch on: outside the else branch of a site gated with ``Kernel.If(Switch.F08/F10 ...)`` and outside the setup allow-list
(``SETUP``, each with its reason). Those are the sites R4 reroutes (a block write is reported under F09, but the damage's
own block absorption, ``ABSORBED``, under F08: it moves with the HP write, or the block is taken twice). A gated ``if``
whose block ends in ``return`` or ``throw`` gates the rest of its enclosing block too (the old code after a switch-on
early return). A write in
``Core2/`` outside ``Core2/Commands`` and the OldState adapter (``Core2/OldState*.cs``) fails.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "src" / "Sts2Emulator"

HOOK_CALL = re.compile(r"\bHook\.(\w+)\s*\(")
CORE2_HOOK_CALL = re.compile(r"\bHooks\.(\w+)\s*\(")
STRING = re.compile(r'\$?@?"(?:\\.|[^"\\])*"')
HEADER = re.compile(r"^(else\s+if|if|foreach|for|while|switch|else|do|catch|finally|try|using|lock)\b")
SKIPPED_KINDS = {"try", "using", "lock", "block"}


@dataclass(frozen=True)
class Inline:
    """A member whose body is inlined where the pattern matches (a property read or a helper call)."""

    member: str


@dataclass(frozen=True)
class Side:
    file: str  # relative to the decompile (game) or to src/Sts2Emulator (Core2)
    member: str  # regex matching the member's signature line
    calls: tuple = ()  # (regex, name | tuple of names | Inline)


@dataclass(frozen=True)
class Command:
    game: Side
    core2: Side


_CMD = "MegaCrit.Sts2.Core.Commands/CreatureCmd.cs"
_HP = "Core2/Commands/CreatureCmd.Hp.cs"
_KILL_CALLS = (
    (r"\bInvokeDiedEvent\s*\(", "Died"),
    (r"CombatManager\.Instance\.RemoveCreature\s*\(", "RemoveCreature"),
    (r"\bcombatState\.RemoveCreature\s*\(", "RemoveFromRoster"),
    (r"\bRemoveAllPowersAfterDeath\s*\(", "RemoveAllPowersAfterDeath"),
    (r"\.AfterRemoved\s*\(", "AfterRemoved"),
    (r"\bOrbQueue\.Clear\s*\(", "ClearOrbs"),
    (r"\bDeactivateHooks\s*\(", "DeactivateHooks"),
    (r"\bHandlePlayerDeath\s*\(", "HandlePlayerDeath"),
    (r"\bKillWithoutCheckingWinCondition\s*\(", "KillWithoutCheckingWinCondition"),
    (r"\bawait Kill\s*\(", "Kill"),
)

COMMANDS: dict[str, Command] = {
    "CreatureCmd.Damage": Command(
        Side(_CMD, r"Task<IEnumerable<DamageResult>> Damage\(PlayerChoiceContext choiceContext, IEnumerable<Creature> "
                   r"targets, decimal amount, ValueProp props, Creature\? dealer, CardModel\? cardSource\)",
             ((r"\bawait Kill\s*\(", "Kill"),)),
        Side("Core2/Commands/CreatureCmd.cs", r"public static IReadOnlyList<DamageResult> Damage\(KernelState k",
             ((r"\bAfterHpLost\(k, \w+, beforeOsty: true\)", "AfterModifyingHpLostBeforeOsty"),
              (r"\bAfterHpLost\(k, \w+, beforeOsty: false\)", "AfterModifyingHpLostAfterOsty"),
              (r"(?<![\w.])Kill\(k\b", "Kill"))),
    ),
    "AttackCommand.Execute": Command(
        Side("MegaCrit.Sts2.Core.Commands.Builders/AttackCommand.cs", r"public async Task<AttackCommand> Execute\(",
             ((r"\bawait _beforeDamage\s*\(", "BeforeDamage"), (r"\bCreatureCmd\.Damage\s*\(", "CreatureCmd.Damage"))),
        Side("Core2/Commands/AttackCmd.cs", r"public AttackCmd Execute\(KernelState k",
             ((r"\bBeforeDamage\?\.Invoke\(", "BeforeDamage"), (r"\bCreatureCmd\.Damage\(", "CreatureCmd.Damage"))),
    ),
    "CreatureCmd.Kill": Command(
        Side(_CMD, r"public static async Task Kill\(IReadOnlyCollection<Creature> creatures",
             ((r"\bKillWithoutCheckingWinCondition\s*\(", "KillWithoutCheckingWinCondition"),
              (r"\.LoseCombat\s*\(", "LoseCombat"), (r"\bPlayerCmd\.EndTurn\s*\(", "PlayerCmd.EndTurn"))),
        Side("Core2/Commands/CreatureCmd.Kill.cs", r"public static void Kill\(KernelState k",
             ((r"\bKillWithoutCheckingWinCondition\(k\b", "KillWithoutCheckingWinCondition"),
              (r"\bLoseCombat\(k\)", "LoseCombat"))),
    ),
    "CreatureCmd.KillWithoutCheckingWinCondition": Command(
        Side(_CMD, r"private static async Task KillWithoutCheckingWinCondition\(", _KILL_CALLS),
        Side("Core2/Commands/CreatureCmd.Kill.cs", r"internal static void KillWithoutCheckingWinCondition\(KernelState k",
             ((r"\bDied\(k\b", "Died"),
              (r"(?<![\w.])RemoveCreature\(k\b", "RemoveCreature"),
              (r"\bOldRoster\.RemoveFromRoster\(", "RemoveFromRoster"),
              (r"\bRemoveAllPowersAfterDeath\(k\b", ("RemoveAllPowersAfterDeath", "AfterRemoved")),
              (r"\bs\.Orbs\.Clear\(", "ClearOrbs"),
              (r"\bDeactivatePlayerHooks\(", "DeactivateHooks"),
              (r"\bKillWithoutCheckingWinCondition\(k\b", "KillWithoutCheckingWinCondition"),
              (r"(?<![\w.])Kill\(k\b", "Kill"))),
    ),
    "CreatureCmd.Escape": Command(
        Side(_CMD, r"public static Task Escape\(Creature creature",
             ((r"\bRemoveAllPowersInternalExcept\s*\(", "RemoveAllPowersInternalExcept"),
              (r"CombatManager\.Instance\.RemoveCreature\s*\(", "RemoveCreature"),
              (r"\bCreatureEscaped\s*\(", "CreatureEscaped"))),
        Side("Core2/Commands/CreatureCmd.Kill.cs", r"public static void Escape\(KernelState k",
             ((r"\benemy\.Buffs\.Clear\(", "RemoveAllPowersInternalExcept"),
              (r"(?<![\w.])RemoveCreature\(k\b", "RemoveCreature"),
              (r"\bOldRoster\.Escaped\(", "CreatureEscaped"))),
    ),
    "CombatManager.CheckWinCondition": Command(
        Side("MegaCrit.Sts2.Core.Combat/CombatManager.cs", r"public async Task<bool> CheckWinCondition\(\)",
             ((r"\bProcessPendingLoss\s*\(", "ProcessPendingLoss"), (r"\bIsEnding\b", Inline(r"public bool IsEnding\b")),
              (r"\bEndCombatInternal\s*\(", "EndCombatInternal"))),
        Side("Core2/Commands/CombatEnd.cs", r"public static bool CheckWinCondition\(KernelState k",
             ((r"\bProcessPendingLoss\(k\b", "ProcessPendingLoss"),
              (r"\bIsEnding\(k\)", Inline(r"public static bool IsEnding\(KernelState k")),
              (r"\bEndCombatInternal\(k\b", "EndCombatInternal"))),
    ),
    # Item 9, the HP commands (batch L1): each game method's internal setter and the commands it awaits, in order.
    "CreatureCmd.Heal": Command(
        Side(_CMD, r"public static async Task Heal\(Creature creature, decimal amount",
             ((r"\.HealInternal\s*\(", "HealInternal"),)),
        Side(_HP, r"public static int Heal\(KernelState k", ((r"\bHealInternal\(k\b", "HealInternal"),)),
    ),
    "CreatureCmd.SetCurrentHp": Command(
        Side(_CMD, r"public static async Task SetCurrentHp\(Creature creature, decimal amount\)",
             ((r"\.SetCurrentHpInternal\s*\(", "SetCurrentHpInternal"), (r"\bawait Kill\s*\(", "Kill"))),
        Side(_HP, r"public static void SetCurrentHp\(KernelState k",
             ((r"\bSetCurrentHpInternal\(s\b", "SetCurrentHpInternal"), (r"(?<![\w.])Kill\(k\b", "Kill"))),
    ),
    "CreatureCmd.SetMaxHp": Command(
        Side(_CMD, r"public static async Task<decimal> SetMaxHp\(Creature creature, decimal amount\)",
             ((r"\.SetMaxHpInternal\s*\(", "SetMaxHpInternal"), (r"\bawait Kill\s*\(", "Kill"))),
        Side(_HP, r"public static int SetMaxHp\(KernelState k",
             ((r"\bSetMaxHpInternal\(s\b", "SetMaxHpInternal"), (r"(?<![\w.])Kill\(k\b", "Kill"))),
    ),
    "CreatureCmd.SetMaxAndCurrentHp": Command(
        Side(_CMD, r"public static async Task SetMaxAndCurrentHp\(Creature creature, decimal amount\)",
             ((r"\bawait SetMaxHp\s*\(", "SetMaxHp"), (r"\bawait SetCurrentHp\s*\(", "SetCurrentHp"))),
        Side(_HP, r"public static void SetMaxAndCurrentHp\(KernelState k",
             ((r"(?<![\w.])SetMaxHp\(k\b", "SetMaxHp"), (r"(?<![\w.])SetCurrentHp\(k\b", "SetCurrentHp"))),
    ),
    "CreatureCmd.GainMaxHp": Command(
        Side(_CMD, r"public static async Task GainMaxHp\(Creature creature, decimal amount\)",
             ((r"\bawait SetMaxHp\s*\(", "SetMaxHp"), (r"\bawait Heal\s*\(", "Heal"))),
        Side(_HP, r"public static void GainMaxHp\(KernelState k",
             ((r"(?<![\w.])SetMaxHp\(k\b", "SetMaxHp"), (r"(?<![\w.])Heal\(k\b", "Heal"))),
    ),
    "CreatureCmd.LoseMaxHp": Command(
        Side(_CMD, r"public static async Task LoseMaxHp\(PlayerChoiceContext choiceContext, Creature creature",
             ((r"\bawait Damage\s*\(", "Damage"), (r"\bawait SetMaxHp\s*\(", "SetMaxHp"))),
        Side(_HP, r"public static void LoseMaxHp\(KernelState k",
             ((r"(?<![\w.])Damage\(k\b", "Damage"), (r"(?<![\w.])SetMaxHp\(k\b", "SetMaxHp"))),
    ),
}


class CommandsError(Exception):
    pass


# ----------------------------------------------------------------------------------------------- source reading


def _clean(line: str) -> str:
    """The line without its string literals and its // comment."""
    line = STRING.sub('""', line)
    i = line.find("//")
    return line[:i] if i >= 0 else line


def member_body(text: str, signature: str) -> list[str]:
    """The lines of the member's body (between its outer braces) whose signature line matches ``signature``."""
    lines = text.splitlines()
    rx = re.compile(signature)
    hits = [i for i, ln in enumerate(lines) if rx.search(ln) and not ln.lstrip().startswith("//")]
    if len(hits) != 1:
        raise CommandsError(f"{signature!r}: {len(hits)} matching members")
    depth, body, started = 0, [], False
    for ln in lines[hits[0]:]:
        c = _clean(ln)
        opens, closes = c.count("{"), c.count("}")
        if not started and opens:
            started = True
            depth += opens - closes
            continue
        if started:
            depth += opens - closes
            if depth <= 0:
                return body
            body.append(ln)
        elif ln.rstrip().endswith(";") and "=>" in c:
            return [c.split("=>", 1)[1]]  # an expression-bodied member
    raise CommandsError(f"{signature!r}: no body")


def _matches(c: str, calls) -> list[tuple[int, object]]:
    found = []
    for rx, what in calls:
        for m in re.finditer(rx, c):
            found.append((m.start(), what))
    return sorted(found, key=lambda t: t[0])


def sequence(text: str, signature: str, calls, hook: re.Pattern = HOOK_CALL, prefix: tuple = (),
             skip_resume: bool = False, _depth: int = 0) -> list[tuple[str, str]]:
    """The (label, name) calls of the member in source order."""
    if _depth > 4:
        raise CommandsError(f"{signature!r}: inlining too deep")
    out: list[tuple[str, str]] = []
    stack: list[str] = []
    pending: str | None = None
    single: str | None = None  # a header without braces governs the next statement
    for raw in member_body(text, signature):
        c = _clean(raw).strip()
        if not c:
            continue
        kind_here = None
        m = HEADER.match(c)
        if m:
            kind_here = "elif" if m.group(1).startswith("else") and m.group(1) != "else" else m.group(1)
        if c.startswith("{") and c.count("{") > c.count("}"):
            stack.append(pending or "block")
            pending = None
            continue
        if c.startswith("}"):
            if stack:
                stack.pop()
            continue
        path = [k for k in (*prefix, *stack) if k not in SKIPPED_KINDS]
        if single is not None:
            path.append(single)
            single = None
        found = [(p, ("hook", h.group(1))) for h in hook.finditer(c) for p in [h.start()]
                 if not (skip_resume and h.group(1).endswith("Resume"))]
        found += [(p, w) for p, w in _matches(c, calls)]
        found.sort(key=lambda t: t[0])
        for pos, what in found:
            label = list(path)
            if kind_here:
                label.append(f"{kind_here}-cond")
            elif re.search(r"&&|\|\||(?<![?.])\?(?![?.\[])", c[:pos]):
                label.append("cond")
            if isinstance(what, tuple) and what and what[0] == "hook":
                name = f"Hook.{what[1]}"
                out.append(("/".join(label) or "-", name))
            elif isinstance(what, Inline):
                out += sequence(text, what.member, calls, hook, tuple(label), skip_resume, _depth + 1)
            elif isinstance(what, tuple):
                out += [("/".join(label) or "-", n) for n in what]
            else:
                out.append(("/".join(label) or "-", what))
        if kind_here:
            if c.endswith("{"):
                stack.append(kind_here)
            elif not c.endswith(";") and not c.endswith("}"):
                pending = kind_here
                single = None
            # a header line that ends the statement itself (if (x) return;) governs nothing after it
        elif pending is not None:
            # the header's single statement (no braces)
            pending = None
    return out


def core2_names(text: str, side: Side) -> list[str]:
    return [n.removeprefix("Hook.") for _, n in sequence(text, side.member, side.calls, CORE2_HOOK_CALL,
                                                         skip_resume=True)]


DECL = re.compile(r"^\s*//\s*construct:\s*game\s+(\S+)\s*$")
DECL_END = re.compile(r"^\s*//\s*end construct\s*$")
DECL_LINE = re.compile(r"^\s*//\s+(\S+)\s+(\S+)(?:\s+!\s*(.+))?\s*$")


def declarations(text: str) -> dict[str, list[tuple[str, str, str | None]]]:
    """Every ``// construct: game X`` block of a source: X -> [(label, name, not-modelled reason)]."""
    out: dict[str, list] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = DECL.match(lines[i])
        if not m:
            i += 1
            continue
        name, rows = m.group(1), []
        i += 1
        while i < len(lines) and not DECL_END.match(lines[i]):
            d = DECL_LINE.match(lines[i])
            if not d:
                raise CommandsError(f"{name}: bad declaration line {lines[i].strip()!r}")
            rows.append((d.group(1), d.group(2), d.group(3)))
            i += 1
        if i >= len(lines):
            raise CommandsError(f"{name}: no '// end construct'")
        if name in out:
            raise CommandsError(f"{name}: declared twice")
        out[name] = rows
        i += 1
    return out


def _diff(a: list, b: list, an: str, bn: str, show) -> list[str]:
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        for x in a[i1:i2]:
            out.append(f"{show(x)} only in {an}")
        for y in b[j1:j2]:
            out.append(f"{show(y)} only in {bn}")
    return out


def check_command(name: str, cmd: Command, decompiled: Path, engine: Path = ENGINE) -> list[str]:
    try:
        game = sequence((decompiled / cmd.game.file).read_text(encoding="utf-8-sig"), cmd.game.member, cmd.game.calls)
        src = (engine / cmd.core2.file).read_text(encoding="utf-8-sig")
        decl = declarations(src).get(name)
        if decl is None:
            return [f"{name}: no '// construct: game {name}' declaration in {cmd.core2.file}"]
        core2 = core2_names(src, cmd.core2)
    except (CommandsError, OSError) as e:
        return [f"{name}: {e}"]
    show = lambda t: f"{t[1]} [{t[0]}]"  # noqa: E731
    out = [f"{name}: {p}" for p in _diff(game, [(l, n) for l, n, _ in decl], "the game", "the declaration", show)]
    modelled = [n.removeprefix("Hook.") for _, n, why in decl if why is None]
    out += [f"{name}: {p}" for p in _diff(modelled, core2, "the declaration", "Core2's calls", str)]
    return out


# ----------------------------------------------------------------------------------------------- the write scan

FIELDS = ("PlayerHp", "PlayerMaxHp", "PlayerBlock", "OstyHp", "OstyMaxHp", "OstyBlock")
MEMBER_FIELDS = ("Hp", "MaxHp", "Block")
_F = "|".join(FIELDS)
_M = "|".join(MEMBER_FIELDS)
_ASSIGN = r"\s*(?:=(?!=)|\+=|-=|\*=|/=|\?\?=|\+\+|--)"
WRITE = re.compile(rf"(?:(?<![\w.])(?:\w+\.)*?\b(?P<f>{_F})|\.(?P<m>{_M}))\b{_ASSIGN}"
                   rf"|(?:\+\+|--)\s*[\w.\[\]]*?(?:\b(?P<f2>{_F})|\.(?P<m2>{_M}))\b")
METHOD = re.compile(r"^\s*(?:(?:public|private|internal|protected|static|override|virtual|sealed|async|unsafe|partial|"
                    r"readonly|extern|new)\s+)+[\w<>\[\],.? ]+?\s+(\w+)\s*(?:<[^>]*>)?\s*\(")
GATE = re.compile(r"Kernel\.If\(\s*(?:Core2\.)?Switch\.F(?:08|10)\b")

# Setup and the run layer: writes that build a combat or carry the run, not a combat's HP change (doc section 4).
SETUP: dict[str, str] = {
    "CombatFactory.cs": "combat setup (the game's Add/creation, not an HP change)",
    "StateCloning.cs": "a state copy",
    "FightAnchor*.cs": "anchoring a fight to a capture (setup)",
    "Run*.cs": "the run layer, outside a combat",
    "method:CreateEnemy": "creature setup",
    "method:SetUpHatchling": "a spawned hatchling's HP: creature setup (the game's Add), not an HP change",
    "method:ApplyAfterRoomEntered": "run-level heals on entering a room (RunState, outside a combat; F01's)",
    "method:ApplyBeforeBossCombat": "a run-level heal before the boss combat starts (RunState, outside a combat; F01's)",
}

# Lines the write pattern matches that write no combat HP or block (file name, the stripped line): each with its reason.
NOT_WRITES: dict[tuple[str, str], str] = {
    ("CombatEngine.cs", "PlayerHp = state.PlayerHp;"): "Baseline's constructor copying the HP into its own field (a read)",
}


# The damage's block absorption (game: CreatureCmd.Damage's DamageBlockInternal): a block write inside the old damage
# entry points that moves with their HP write, so F08's (doc section 4; the three sites R3b's review named).
ABSORBED: dict[tuple[str, str], str] = {
    ("CardEffects.cs", "DealDamageToPlayer"): "the player's block absorbing a hit",
    ("CardEffects.cs", "LandDamageOnEnemy"): "an enemy's block absorbing a card's hit",
    ("EnemyAI.cs", "DealAttackDamage"): "the player's block absorbing an enemy's attack",
}
TERMINAL = re.compile(r"^(?:return|throw)\b")


def family(rel: str, method: str, f: str) -> str:
    """F08 for an HP field and for the damage's block absorption (ABSORBED), else F09 (block)."""
    if "Block" not in f or (Path(rel).name, method) in ABSORBED:
        return "F08"
    return "F09"


@dataclass
class Write:
    file: str
    line: int
    method: str
    field: str
    family: str
    text: str = field(compare=False, default="")


def _allowed(rel: str, method: str) -> str | None:
    name = Path(rel).name
    for key, why in SETUP.items():
        if key.startswith("method:"):
            if method == key[len("method:"):]:
                return why
        elif Path(name).match(key):
            return why
    return None


INLINE_BRACES = re.compile(r"\{[^{}]*\}")


def _block_braces(s: str) -> str:
    """The braces of a line that open or close blocks: balanced groups inside it (property patterns such as
    ``is { } k``, initializers) removed."""
    prev = None
    while prev != s:
        prev, s = s, INLINE_BRACES.sub("", s)
    return "".join(ch for ch in s if ch in "{}")


def scan_writes(text: str, rel: str) -> list[Write]:
    """The HP/block writes of one old-engine file that run with the switch on: not inside the old (else) branch of an
    ``if (Kernel.If(Switch.F08/F10 ...))`` gate, and not in the setup allow-list."""
    out: list[Write] = []
    method = "?"
    stack: list[tuple[str, bool, bool]] = []  # (kind, an old branch of a gate, the if chain is gated)
    closed: dict[int, bool] = {}  # depth -> the if chain whose block just closed there was gated
    pending: tuple[str, bool, bool] | None = None
    last_return: dict[int, bool] = {}  # depth -> the block's last statement there returns or throws
    for no, raw in enumerate(text.splitlines(), 1):
        c = _clean(raw)
        s = c.strip()
        if not s:
            continue
        mm = METHOD.match(c)
        if mm and not s.endswith(";"):
            method = mm.group(1)
        hm = HEADER.match(s.lstrip("}").strip())
        depth = len(stack) - (1 if s.startswith("}") else 0)
        if hm:
            kind = hm.group(1)
            if kind == "else":
                pending = ("else", closed.get(depth, False), False)
            elif kind.startswith("else"):
                chain = closed.get(depth, False)
                pending = ("if", chain, chain or bool(GATE.search(s)))
            elif kind == "if":
                pending = ("if", False, bool(GATE.search(s)))
            else:
                pending = (kind, False, False)
        elif not s.startswith("}") and not s.startswith("{"):
            closed[len(stack)] = False
            last_return[len(stack)] = bool(TERMINAL.match(s))
        old = any(o for _, o, _ in stack) or (pending is not None and pending[1] and not s.endswith("{") and hm is None)
        if not old and not _allowed(rel, method) and (Path(rel).name, s) not in NOT_WRITES:
            for m in WRITE.finditer(c):
                f = m.group("f") or m.group("m") or m.group("f2") or m.group("m2")
                out.append(Write(rel, no, method, f, family(rel, method, f), s))
        for ch in _block_braces(s):
            if ch == "{":
                stack.append(pending or ("block", False, False))
                pending = None
            elif ch == "}" and stack:
                kind, _, chain = stack.pop()
                closed[len(stack)] = chain if kind == "if" else False
                if kind == "if" and chain and last_return.pop(len(stack) + 1, False) and stack:
                    # a switch-on early return: the rest of the enclosing block is the old path
                    k0, _, c0 = stack[-1]
                    stack[-1] = (k0, True, c0)
                last_return.pop(len(stack) + 1, None)
    return out


def old_writes(engine: Path = ENGINE) -> list[Write]:
    out: list[Write] = []
    for p in sorted((engine / "Core").rglob("*.cs")):
        rel = p.relative_to(engine).as_posix()
        out += scan_writes(p.read_text(encoding="utf-8-sig"), rel)
    return out


def core2_writes(engine: Path = ENGINE) -> list[Write]:
    """HP/block writes in Core2 outside Core2/Commands and the OldState adapter: construction errors."""
    out: list[Write] = []
    for p in sorted((engine / "Core2").rglob("*.cs")):
        rel = p.relative_to(engine).as_posix()
        if rel.startswith("Core2/Commands/") or Path(rel).name.startswith("OldState"):
            continue
        for no, raw in enumerate(p.read_text(encoding="utf-8-sig").splitlines(), 1):
            c = _clean(raw)
            for m in WRITE.finditer(c):
                f = m.group("f") or m.group("m") or m.group("f2") or m.group("m2")
                out.append(Write(rel, no, "?", f, "F09" if "Block" in f else "F08", c.strip()))
    return out
