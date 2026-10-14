"""Listener order: does the emulator fire a hook's listeners in the game's order?

Used by ``python -m tools.fidelity.hook_table order`` (and ``mutants``); see hook_table.py for the table.

1. ``instrument``: a scratch copy of the engine gets a probe at every dispatch-point statement that names a
   registered listener (``HookProbe.Hit(n)`` before it, ``HookProbe.Cond(n) &&`` inside an ``if``/``while``
   condition, after the last label of a ``case`` section). Line numbers never move. A probe that does not
   compile is dropped (the build is retried without the probes on the lines the compiler names).
2. ``scenarios``: one synthetic combat per trigger (combat start, an attack, a skill, a power, a potion, an
   exhaust, a draw, the end of a turn round) holding every registered listener of the hooks that trigger
   fires: relics in obtain order, potions by slot, powers in application order on the player or on one
   dummy enemy (the side the dispatch point reads), cards in the hand. Each runs twice, with the relics and
   the powers in the opposite order the second time, so an order that ignores obtain order shows.
3. ``analyse``: the probe log is cut at the engine's CombatTrace seams into segments (``turn_end``,
   ``enemy_turn_start``, ``turn_start``, a play's ``before``/``after`` ...). Within a segment

   * listeners of one dispatcher (a ``Hook.X`` method and its Early/Late passes) must fire in the game's
     listener order (``hook_table.game_order_key``): pass, creature, kind, then obtain/application order;
   * listeners of different phase hooks (``PHASES``) must fire in the game's phase order (CombatManager:
     BeforeSideTurnStart, AfterBlockCleared, AfterEnergyReset, BeforeHandDraw, AfterPlayerTurnStart,
     AfterSideTurnStart, ...), except across a pause: when a screen holds the setup of a turn start
     (``SetupPlayerTurn``), the game runs on to AfterSideTurnStart before the screen is asked, so the setup
     listeners after the screen's own come after AfterSideTurnStart (``pause_floors``). The harness marks each
     screen it answers; an AfterSideTurnStart listener before a mark and a setup listener after it, of at least
     the phase the setup had reached by then, are not an inversion.

   A listener whose probe sits in a method listed for several hooks may stand for any of them; a pair counts
   as out of order only when no choice puts it in order. Query hooks (Modify*, Should*, TryModify*) are read
   many times and only take part in the first check.
"""

from __future__ import annotations

import collections
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import hook_table as ht

PROBE_CS = """// Written by tools/fidelity/hook_order.py into its scratch copy of the engine only.
namespace Sts2Emulator;

public static class HookProbe
{
    public static readonly List<int> Log = new();
    public static bool On;

    public static void Hit(int id)
    {
        if (On) Log.Add(id);
    }

    public static bool Cond(int id)
    {
        if (On) Log.Add(id);
        return true;
    }

    public static TValue Pass<TValue>(int id, TValue value)
    {
        if (On) Log.Add(id);
        return value;
    }
}
"""
HIT = "global::Sts2Emulator.HookProbe.Hit({});"
COND = "global::Sts2Emulator.HookProbe.Cond({}) && "
PASS_THROUGH = "global::Sts2Emulator.HookProbe.Pass({}, "
BELOW_NORMAL = 0x00004000

# ------------------------------------------------------------------------------------------- statements


@dataclass
class Stmt:
    start: int            # first character of the statement
    kind: str             # "plain" | "if" | "while" | "case" | "embedded"
    label_end: int = -1   # for "case": where the section's statements start


BLOCK_BEFORE = re.compile(r"(?:\)|=>|\belse|\bdo|\btry|\bfinally|\bget|\bset|\binit|\badd|\bremove|[;{}:])\s*$")
EMBED_HEAD = re.compile(r"\b(if|while|for|foreach|using|lock|fixed)\s*\($")


def _matching_open(m: str, close: int) -> int:
    depth = 0
    for i in range(close, -1, -1):
        if m[i] == ")":
            depth += 1
        elif m[i] == "(":
            depth -= 1
            if depth == 0:
                return i
    return -1


def statements(m: str, body: tuple[int, int]) -> list[Stmt]:
    """The statement starts inside one method body (masked text), in order.

    A '{' opens a block when what precedes it ends a statement head (``)`` of an if/for/..., ``=>``, ``else``,
    ``;``...) and the ``)`` is not a constructor call's (``new X() { ... }`` is an initializer). Inside
    parentheses, brackets and initializers nothing starts a statement.
    """
    a, b = body
    if m[a] != "{":
        return []
    out: list[Stmt] = []
    stack: list[str] = []
    i = a

    def begin(pos: int) -> None:
        j = pos
        while j < b and m[j] in " \t\r\n":
            j += 1
        if j >= b or m[j] in "};":
            return
        word = re.match(r"[A-Za-z_]\w*", m[j:j + 12])
        w = word.group(0) if word else ""
        if w in ("case", "default") and re.match(r"(?:case\b[^;{}]*?|default\s*):(?!:)", m[j:b]):
            # the section's statements start after its last label
            k = j
            while True:
                lab = re.match(r"\s*(?:case\b[^;{}]*?|default\s*):(?!:)", m[k:b])
                if not lab:
                    break
                k += lab.end()
            out.append(Stmt(j, "case", k))
            begin(k)
            return
        kind = "embedded" if stack and stack[-1] == "embed" else ("if" if w == "if" else "while" if w == "while" else "plain")
        if w == "else":
            k = j + re.match(r"else\s*", m[j:b]).end()
            if m.startswith("if", k) and not re.match(r"\w", m[k + 2:k + 3]):
                out.append(Stmt(k, "if"))
            elif k < b and m[k] != "{":
                out.append(Stmt(k, "embedded"))
            return
        out.append(Stmt(j, kind))

    while i < b:
        c = m[i]
        top = stack[-1] if stack else None
        if c == "{":
            before = m[max(a, i - 200):i]
            is_block = i == a or (top in (None, "block", "embed") and BLOCK_BEFORE.search(before))
            if is_block and before.rstrip().endswith(")"):
                op = _matching_open(m, a + len(m[a:i].rstrip()) - 1)
                head = m[max(a, op - 80):op]
                if re.search(r"\bnew\b[\w.<>\[\], ]*$", head):
                    is_block = False
            if i == a:
                stack.append("block")
                begin(i + 1)
            elif is_block:
                if top == "embed":
                    stack.pop()
                stack.append("block")
                begin(i + 1)
            else:
                stack.append("init")
        elif c == "}":
            if stack:
                stack.pop()
            if stack and stack[-1] in ("block",):
                begin(i + 1)
        elif c in "([":
            stack.append("paren")
        elif c in ")]":
            if stack and stack[-1] == "paren":
                stack.pop()
            if c == ")" and (not stack or stack[-1] in ("block", "embed")):
                op = _matching_open(m, i)
                if op >= 0 and EMBED_HEAD.search(m[max(a, op - 12):op + 1]):
                    j = i + 1
                    while j < b and m[j] in " \t\r\n":
                        j += 1
                    if j < b and m[j] != "{":
                        stack.append("embed")
                        begin(i + 1)
                        stack.pop()
        elif c == ";":
            if not stack or stack[-1] == "block":
                begin(i + 1)
        i += 1
    out.sort(key=lambda s: s.start)
    return out


def enclosing_statement(stmts: list[Stmt], pos: int) -> Stmt | None:
    best = None
    for s in stmts:
        if s.start <= pos:
            best = s
        else:
            break
    return best


# ---------------------------------------------------------------------------------------- instrumenting


@dataclass
class Probe:
    id: int
    file: str
    line: int
    pairs: list[tuple[str, str, str]] = field(default_factory=list)   # (model, kind, hook)


DECLARATION = re.compile(r"(?:var|int|bool|long|float|double|decimal|string|[A-Z][\w.]*(?:<[^;=()]*>)?\??)\s+"
                         r"(\w+)\s*=(?![=>])")


def _past_declaration(m: str, stmts: list[Stmt], st: Stmt) -> Stmt:
    """A listener read into a local (``int ritual = Get(..., BuffId.Ritual);``) acts where the local is used,
    so the probe goes to the next statement that names it. The read itself stays when none does."""
    if st.kind != "plain":
        return st
    d = DECLARATION.match(m, st.start)
    if not d:
        return st
    name = re.compile(rf"\b{re.escape(d.group(1))}\b")
    later = [s for s in stmts if s.start > st.start]
    for s, nxt in zip(later, later[1:] + [None]):
        end = nxt.start if nxt else s.start + 400
        if name.search(m, s.start, end):
            return s
    return st


def plan_probes(engine: ht.Engine, rows: list[ht.Row]) -> tuple[dict[tuple[str, int, str], Probe], list[str]]:
    """Insertion points for every registered site: (file, offset, form) -> probe. Skipped sites, with why."""
    stmts_cache: dict[tuple[str, int], list[Stmt]] = {}
    points: dict[tuple[str, int, str], Probe] = {}
    skipped = []
    # A site that serves several hooks of one model (a turn-start method listed for six hooks) stands for the
    # hooks its comment cites, when it cites any: `BlurPower.ShouldClearBlock` above the block clear.
    hooks_of = collections.defaultdict(set)
    for r in rows:
        if r.status == "registered":
            hooks_of[ht.short(r.model)].add(r.hook)
    cited: dict[tuple[str, str, int], set] = collections.defaultdict(set)
    word = re.compile(r"\b([A-Z]\w+)\.([A-Z]\w+)\b")
    for rel, raw in engine.files.items():
        masked = engine.masked[rel]
        for x in word.finditer(raw):
            if masked[x.start()] == " " and x.group(2) in hooks_of.get(x.group(1), ()):
                owner = ht._owner(engine, rel, x.start())
                if owner is not None:
                    cited[rel, x.group(1), owner.start].add((x.group(2), engine.line_of(rel, x.start())))
    for r in rows:
        for rel, off in r.sites:
            mt = engine.method_at(rel, off)
            if mt is None or mt.field:
                skipped.append(f"{rel}:{engine.line_of(rel, off)} field initializer")
                continue
            line = engine.line_of(rel, off)
            near = {h for h, ln in cited.get((rel, ht.short(r.model), mt.start), ()) if -40 <= line - ln <= 3}
            if near and r.hook not in near:
                continue
            m = engine.masked[rel]
            if m[mt.body[0]] != "{":
                header = m[mt.start:mt.body[0]]
                if re.search(r"\bvoid\b", header):
                    skipped.append(f"{rel}:{engine.line_of(rel, off)} void expression body")
                    continue
                key = (rel, mt.body[0] + 2, "pass")
            else:
                key_c = (rel, mt.start)
                if key_c not in stmts_cache:
                    stmts_cache[key_c] = statements(m, mt.body)
                st = enclosing_statement(stmts_cache[key_c], off)
                if st is None:
                    skipped.append(f"{rel}:{engine.line_of(rel, off)} no statement")
                    continue
                st = _past_declaration(m, stmts_cache[key_c], st)
                if st.kind in ("if", "while"):
                    par = m.find("(", st.start)
                    key = (rel, par + 1, "cond")
                elif st.kind == "case":
                    key = (rel, st.label_end, "hit")
                elif st.kind == "embedded":
                    key = (rel, st.start, "wrap")
                else:
                    key = (rel, st.start, "hit")
            p = points.get(key)
            if p is None:
                p = points[key] = Probe(len(points), rel, engine.line_of(rel, key[1]))
            pair = (r.model, r.kind, r.hook)
            if pair not in p.pairs:
                p.pairs.append(pair)
    return points, skipped


def apply_probes(texts: dict[str, str], masked: dict[str, str], points: dict, drop: set[int]) -> dict[str, str]:
    """The engine's files with the probes in, inserted back to front so offsets hold; lines never move."""
    by_file = collections.defaultdict(list)
    for (rel, pos, form), p in points.items():
        if p.id not in drop:
            by_file[rel].append((pos, form, p.id))
    out = dict(texts)
    for rel, items in by_file.items():
        t = texts[rel]
        m = masked[rel]
        edits: list[tuple[int, int, str]] = []          # (position, 0 = closing text first, text)
        for pos, form, pid in items:
            if form == "hit":
                edits.append((pos, 1, HIT.format(pid) + " "))
            elif form == "cond":
                edits.append((pos, 1, COND.format(pid)))
            elif form == "pass":
                edits.append((m.find(";", pos), 0, ")"))
                edits.append((pos, 1, PASS_THROUGH.format(pid)))
            elif form == "wrap":
                edits.append((ht._statement_end(m, pos), 0, " }"))
                edits.append((pos, 1, "{ " + HIT.format(pid) + " "))
        for pos, _, s in sorted(edits, key=lambda e: (-e[0], e[1])):
            t = t[:pos] + s + t[pos:]
        out[rel] = t
    return out


def make_copy(src_root: Path, dst: Path) -> None:
    """The engine, EmuDump and the two data files it links, in dst (sources only; bin/obj kept)."""
    for rel in ("src/Sts2Emulator", "tools/emudump"):
        for p in (src_root / rel).rglob("*"):
            r = p.relative_to(src_root)
            if any(part in ("bin", "obj") for part in r.parts) or p.is_dir():
                continue
            t = dst / r
            if t.exists() and t.stat().st_size == p.stat().st_size and t.read_bytes() == p.read_bytes():
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)
    for rel in ("data/id_map.json", "tools/fidelity/power_aliases.json"):
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_root / rel, dst / rel)
    # The invariant sources EmuDump compiles in.
    inv = src_root / "src/Sts2Emulator.Tests/Support/Invariants"
    if inv.exists():
        for p in inv.glob("*.cs"):
            t = dst / p.relative_to(src_root)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)


def dotnet_env() -> dict:
    return dict(os.environ, DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_NOLOGO="1", MSBUILDDISABLENODEREUSE="1")


def build(copy: Path, log=print) -> tuple[bool, str]:
    proj = str(copy / "tools/emudump")
    kw = {"creationflags": BELOW_NORMAL} if os.name == "nt" else {}
    r = subprocess.run(["dotnet", "build", proj, "-c", "Release", "-v", "q", "-nologo", "-clp:NoSummary"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dotnet_env(), **kw)
    return r.returncode == 0, r.stdout


ERROR_LINE = re.compile(r"([^\s(]+\.cs)\((\d+),\d+\): error")


class BuildFailed(RuntimeError):
    """The instrumented build failed on lines that hold no probe; ``output`` is the compiler's."""

    def __init__(self, output: str):
        super().__init__("instrumented build failed outside the probes:\n" + output[-4000:])
        self.output = output


def build_instrumented(engine: ht.Engine, rows: list[ht.Row], src_root: Path, copy: Path, log=print,
                       texts: dict[str, str] | None = None) -> tuple[dict, list[str], set[int]]:
    """Instrument, write and build the scratch copy; drop probes the compiler rejects. -> (points, skipped,
    dropped ids). ``texts`` (rel -> source) replaces the engine's own sources, for a mutant."""
    points, skipped = plan_probes(engine, rows)
    make_copy(src_root, copy)
    base = texts if texts is not None else engine.files
    drop: set[int] = set()
    eng_dir = copy / "src" / "Sts2Emulator"
    (eng_dir / "HookProbe.cs").write_text(PROBE_CS, encoding="utf-8")
    for attempt in range(6):
        out = apply_probes(base, engine.masked, points, drop)
        for rel, t in out.items():
            (eng_dir / rel).write_text(t, encoding="utf-8")
        ok, text = build(copy)
        if ok:
            return points, skipped, drop
        bad = set()
        for f, line in ERROR_LINE.findall(text):
            fp = Path(f).as_posix()
            for (rel, pos, form), p in points.items():
                if fp.endswith(rel) and p.line == int(line) and p.id not in drop:
                    bad.add(p.id)
        if not bad:
            raise BuildFailed(text)
        log(f"  build {attempt + 1}: dropping {len(bad)} probes the compiler rejected")
        drop |= bad
    raise RuntimeError("instrumented build still failing after dropping probes")


# -------------------------------------------------------------------------------------------- scenarios

STRIKE, DEFEND = "StrikeIronclad", "DefendIronclad"
TRIGGERS: dict[str, dict] = {
    "combat_start": {"actions": [], "hooks": [
        "AfterRoomEntered", "BeforeCombatStart", "BeforeCombatStartLate", "AfterCreatureAddedToCombat",
        "AfterCardEnteredCombat", "BeforeSideTurnStart", "ShouldClearBlock", "AfterPreventingBlockClear",
        "AfterBlockCleared", "ShouldPlayerResetEnergy", "AfterEnergyReset", "AfterEnergyResetLate", "ModifyMaxEnergy",
        "BeforeHandDraw", "BeforeHandDrawLate", "ModifyHandDraw", "ModifyHandDrawLate", "AfterModifyingHandDraw",
        "AfterPlayerTurnStartEarly", "AfterPlayerTurnStart", "AfterPlayerTurnStartLate", "AfterSideTurnStart",
        "AfterSideTurnStartLate", "AfterAutoPrePlayPhaseEnteredEarly", "AfterAutoPrePlayPhaseEntered",
        "AfterAutoPrePlayPhaseEnteredLate"]},
    "play_attack": {"actions": [{"do": "play", "card": STRIKE}], "hooks": [
        "ModifyCardPlayCount", "AfterModifyingCardPlayCount", "BeforeCardPlayed", "BeforeAttack", "ModifyDamageAdditive",
        "ModifyDamageMultiplicative", "ModifyDamageCap", "AfterModifyingDamageAmount", "BeforeDamageReceived",
        "ModifyHpLostBeforeOsty", "ModifyHpLostBeforeOstyLate", "ModifyHpLostAfterOsty", "ModifyHpLostAfterOstyLate",
        "AfterDamageGiven", "AfterDamageReceived", "AfterDamageReceivedLate", "AfterCurrentHpChanged", "AfterBlockBroken",
        "AfterAttack", "AfterCardPlayed", "AfterCardPlayedLate", "ModifyCardPlayResultPileTypeAndPosition",
        "AfterModifyingCardPlayResultPileOrPosition", "TryModifyEnergyCostInCombat", "TryModifyEnergyCostInCombatLate",
        "AfterEnergySpent", "ShouldPlay"]},
    "play_skill": {"actions": [{"do": "play", "card": DEFEND}], "hooks": [
        "BeforeCardPlayed", "ModifyBlockAdditive", "ModifyBlockMultiplicative", "AfterModifyingBlockAmount",
        "BeforeBlockGained", "AfterBlockGained", "AfterCardPlayed", "AfterCardPlayedLate"]},
    "play_power": {"actions": [{"do": "play", "card": "Inflame"}], "hooks": [
        "BeforeCardPlayed", "BeforePowerAmountChanged", "AfterPowerAmountChanged", "TryModifyPowerAmountReceived",
        "AfterModifyingPowerAmountReceived", "ModifyPowerAmountGivenAdditive", "ModifyPowerAmountGivenMultiplicative",
        "AfterModifyingPowerAmountGiven", "AfterCardPlayed", "AfterCardPlayedLate"]},
    "exhaust": {"actions": [{"do": "play", "card": "TrueGrit"}], "hooks": [
        "AfterCardExhausted", "BeforeCardPlayed", "AfterCardPlayed", "ModifyBlockAdditive", "ModifyBlockMultiplicative"]},
    "draw": {"actions": [{"do": "play", "card": "PommelStrike"}], "draw_empty": True, "hooks": [
        "ShouldDraw", "AfterCardDrawnEarly", "AfterCardDrawn", "AfterPreventingDraw", "AfterShuffle",
        "ModifyShuffleOrder", "AfterCardPlayed"]},
    "potion": {"actions": [{"do": "potion", "slot": 0}], "potion": "BlockPotion", "hooks": [
        "BeforePotionUsed", "AfterPotionUsed", "ModifyBlockAdditive", "ModifyBlockMultiplicative", "AfterBlockGained"]},
    "end_turn": {"actions": [{"do": "end"}], "hooks": [
        "AfterAutoPostPlayPhaseEntered", "BeforeSideTurnEndVeryEarly", "BeforeSideTurnEndEarly", "BeforeSideTurnEnd",
        "BeforeFlush", "BeforeFlushLate", "ShouldFlush", "AfterFlush", "AfterSideTurnEnd", "AfterSideTurnEndLate",
        "ShouldTakeExtraTurn", "AfterTakingExtraTurn", "ShouldEtherealTrigger", "BeforeSideTurnStart", "ShouldClearBlock",
        "AfterPreventingBlockClear", "AfterBlockCleared", "AfterSideTurnStart", "AfterSideTurnStartLate",
        "ShouldPlayerResetEnergy", "AfterEnergyReset", "AfterEnergyResetLate", "BeforeHandDraw", "BeforeHandDrawLate",
        "ModifyHandDraw", "ModifyHandDrawLate", "AfterModifyingHandDraw", "AfterPlayerTurnStartEarly",
        "AfterPlayerTurnStart", "AfterPlayerTurnStartLate", "AfterAutoPrePlayPhaseEnteredEarly",
        "AfterAutoPrePlayPhaseEntered", "AfterAutoPrePlayPhaseEnteredLate", "ModifyDamageAdditive",
        "ModifyDamageMultiplicative", "BeforeDamageReceived", "ModifyHpLostAfterOsty", "AfterDamageReceived",
        "AfterCurrentHpChanged", "AfterAttack", "BeforeAttack", "AfterDamageGiven"]},
}
_TURN_START = ["BeforeSideTurnStart", "AfterBlockCleared", "AfterEnergyReset", "AfterEnergyResetLate", "BeforeHandDraw",
               "BeforeHandDrawLate", "AfterPlayerTurnStartEarly", "AfterPlayerTurnStart", "AfterPlayerTurnStartLate",
               "AfterSideTurnStart", "AfterSideTurnStartLate", "AfterAutoPrePlayPhaseEnteredEarly",
               "AfterAutoPrePlayPhaseEntered", "AfterAutoPrePlayPhaseEnteredLate"]
_SIDE_END = ["BeforeSideTurnEndVeryEarly", "BeforeSideTurnEndEarly", "BeforeSideTurnEnd"]
# The game's phase order within each segment (CombatManager.StartCombatInternal, StartTurn, SetupPlayerTurn,
# EndPlayerTurnPhaseOne/Two, DoTurnEnd, FlushPlayerHand; CombatRoom.Enter for the room hook).
PHASES: dict[str, list[str]] = {
    "combat_start": ["AfterRoomEntered", "BeforeCombatStart", "BeforeCombatStartLate"] + _TURN_START,
    "turn_start": _TURN_START,
    "turn_end": ["AfterAutoPostPlayPhaseEntered"] + _SIDE_END + ["BeforeFlush", "BeforeFlushLate", "AfterFlush",
                                                                 "AfterSideTurnEnd", "AfterSideTurnEndLate"],
    "enemy_turn_start": ["BeforeSideTurnStart", "AfterBlockCleared", "AfterSideTurnStart", "AfterSideTurnStartLate"],
    "enemy_turn_end": _SIDE_END + ["AfterSideTurnEnd", "AfterSideTurnEndLate"],
    "play_before": ["BeforeCardPlayed"],
    "play_after": ["AfterCardPlayed", "AfterCardPlayedLate"],
}
QUERY = re.compile(r"^(Modify|Should|TryModify|AfterModifying)")
# Every power goes on at this amount: high enough that a countdown (Sandpit) does not end the fight on the
# first turn, low enough that nothing kills a 999 HP player.
AMOUNT = 5
# Hooks whose listeners would stop the trigger itself: a refused play, an extra turn.
AVOID = {"play": frozenset({"ShouldPlay"}), "end": frozenset({"ShouldTakeExtraTurn"})}
# The Hook.cs dispatchers of the phase hooks: one dispatch per side and turn (or per play), so every listener
# in a segment answers the same dispatch. The rest fire once per event.
PHASE_DISPATCHERS = frozenset({
    "BeforeCombatStart", "AfterRoomEntered", "BeforeSideTurnStart", "AfterEnergyReset",
    "BeforeHandDraw", "AfterPlayerTurnStart", "AfterSideTurnStart", "AfterAutoPrePlayPhaseEntered",
    "AfterAutoPostPlayPhaseEntered", "BeforeTurnEnd", "BeforeFlush", "AfterFlush", "AfterTurnEnd",
    "BeforeCardPlayed", "AfterCardPlayed", "ModifyHandDraw", "ShouldFlush",
})


def _table(engine_root: Path, name: str, ctor: str) -> dict[str, int]:
    p = Path(engine_root) / "Generated" / name
    return {n: int(i) for i, n in re.findall(rf'new {ctor}\(Id: (\d+), Name: "(\w+)"', p.read_text(encoding="utf-8"))}


def power_side(engine: ht.Engine, sites: list[list]) -> str:
    """Which creature a power listener sits on, read off its dispatch-point statements: ``enemy`` when they
    read an enemy's buffs, else ``player``."""
    enemy = player = 0
    for rel, off in sites:
        m = engine.masked[rel]
        a = m.rfind("\n", 0, off)
        b = m.find("\n", off)
        line = m[a:b]
        if "PlayerBuffs" in line:
            player += 1
        elif re.search(r"\b\w+\.Buffs\b|Enemies", line):
            enemy += 1
    return "enemy" if enemy > player else "player"


@dataclass
class Listener:
    model: str
    kind: str
    emu: str | int                  # the id the harness takes
    side: str = "player"
    hooks: set[str] = field(default_factory=set)


def listeners_for(engine: ht.Engine, rows: list[ht.Row], hooks: set[str], avoid: frozenset = frozenset()) -> list[Listener]:
    """Every registered listener of these hooks that the harness can put in a combat, bar the models that
    also override a hook in ``avoid`` (a card that refuses plays would stop the play the trigger makes)."""
    refuses = {r.model for r in rows if r.hook in avoid}
    relics = _table(engine.root, "Relics.g.cs", "RelicDef")
    potions = _table(engine.root, "Potions.g.cs", "PotionDef")
    cards = _table(engine.root, "Cards.g.cs", "CardDef")
    out: dict[str, Listener] = {}
    for r in rows:
        if r.status != "registered" or r.hook not in hooks or not r.sites or r.model in refuses:
            continue
        name = ht.short(r.model)
        emu: str | int | None = None
        side = "player"
        if r.kind == "relic":
            emu = relics.get(name)
        elif r.kind == "potion":
            emu = potions.get(name)
        elif r.kind == "card":
            emu = cards.get(name)
        elif r.kind == "enchantment":
            emu = name
        elif r.kind == "power":
            emu = ht.buff_member(engine.root, name)
            side = power_side(engine, r.sites)
        if emu is None:
            continue
        key = f"{r.kind}:{emu}:{side}"
        lst = out.setdefault(key, Listener(r.model, r.kind, emu, side))
        lst.hooks.add(r.hook)
    return list(out.values())


def scenario(engine: ht.Engine, trigger: str, listeners: list[Listener], reverse: bool) -> tuple[dict, dict]:
    """The harness scenario for one trigger, and the game positions of its listeners."""
    spec = TRIGGERS[trigger]
    cards = _table(engine.root, "Cards.g.cs", "CardDef")
    potions = _table(engine.root, "Potions.g.cs", "PotionDef")
    order = (lambda xs: list(reversed(xs))) if reverse else (lambda xs: list(xs))
    relics = order([x for x in listeners if x.kind == "relic"])
    ppow = order([x for x in listeners if x.kind == "power" and x.side == "player"])
    epow = order([x for x in listeners if x.kind == "power" and x.side == "enemy"])
    pots = [x for x in listeners if x.kind == "potion"]
    hand_cards = [x for x in listeners if x.kind in ("card", "enchantment")]
    slots = []
    if spec.get("potion"):
        slots.append(potions[spec["potion"]])
    slots += [x.emu for x in pots]
    strike, defend = cards[STRIKE], cards[DEFEND]
    hand = [[x.emu] if x.kind == "card" else [defend, x.emu] for x in hand_cards]
    sc = {"name": f"{trigger}{'/rev' if reverse else ''}", "relics": [x.emu for x in relics], "potions": slots[:3],
          "player_buffs": [[x.emu, AMOUNT] for x in ppow], "enemy_buffs": [[x.emu, AMOUNT] for x in epow],
          "deck": [[strike]] * 5 + [[defend]] * 5 + ([[x.emu] for x in hand_cards if x.kind == "card"]
                                                     if trigger == "combat_start" else []),
          "hand": hand + [[strike]] * 2, "draw": [] if spec.get("draw_empty") else [[strike], [defend]] * 10,
          "discard": [[strike], [defend]] * 4,
          "actions": [dict(a, card=[cards[a["card"]]]) if "card" in a else a for a in spec["actions"]]}
    positions = {}
    for i, x in enumerate(relics):
        positions[_lkey(x)] = {"kind": "relic", "creature": 0, "index": i}
    for i, x in enumerate(ppow):
        positions[_lkey(x)] = {"kind": "power", "creature": 0, "index": i}
    for i, x in enumerate(epow):
        positions[_lkey(x)] = {"kind": "power", "creature": 1, "index": i}
    for i, x in enumerate(pots):
        positions[_lkey(x)] = {"kind": "potion", "creature": 0, "index": i + (1 if spec.get("potion") else 0)}
    for i, x in enumerate(hand_cards):
        positions[_lkey(x)] = {"kind": "card" if x.kind == "card" else "enchantment", "creature": 0,
                               "pile": "hand", "index": i}
    return sc, positions


def _lkey(x: Listener) -> str:
    return f"{x.kind}:{x.emu}:{x.side}"


# ---------------------------------------------------------------------------------------------- analysis

MARKS = {-1: "turn_end", -2: "enemy_turn", -3: "turn_start", -4: "play", -5: "played", -6: "move", -7: "moved"}
# The harness answers an open screen (a relic's pick, Entropy's transform ...) here: the point the turn start's
# setup was paused at. It cuts no segment.
SCREEN = -8


def segments(log: list[int]) -> list[tuple[str, list[tuple[int, int]]]]:
    """Cut a probe log into (segment name, [(time, probe id)]) at its marks."""
    out: list[tuple[str, list]] = []
    cur, items = "combat_start", []
    in_move = False
    for t, x in enumerate(log):
        if x >= 0 or x == SCREEN:
            if not in_move:
                items.append((t, x))
            continue
        name = MARKS.get(x)
        if x <= -100:
            out.append((cur, items))
            cur, items = ("combat_start" if x == -100 else "action"), []
            in_move = False
            continue
        if name == "move":
            out.append((cur, items))
            cur, items, in_move = "enemy_move", [], True
            continue
        if name == "moved":
            in_move = False
            cur, items = "enemy_turn_end", []
            continue
        out.append((cur, items))
        items = []
        cur = {"turn_end": "turn_end", "enemy_turn": "enemy_turn_start", "turn_start": "turn_start",
               "play": "play_effect", "played": "play_after"}[name]
    out.append((cur, items))
    fixed = []
    for i, (name, its) in enumerate(out):
        if name == "action":
            # the stretch of a play before its "play" mark is its BeforeCardPlayed side
            nxt = out[i + 1][0] if i + 1 < len(out) else ""
            name = "play_before" if nxt == "play_effect" else "action"
        fixed.append((name, its))
    return [(n, its) for n, its in fixed if its]


@dataclass
class Inversion:
    segment: str
    check: str            # "listener" | "phase"
    group: str            # dispatcher, or "a > b" for phases
    first: tuple[str, str, str]    # (kind, model, hook) of the listener the emulator ran first
    second: tuple[str, str, str]
    # Where the game's order and the emulator's part: "pass" (an Early/Late or additive/multiplicative pass),
    # "creature", "kind" (powers, relics, potions, orbs, cards) or "index" (obtain or application order);
    # "phase" for the phase check.
    level: str = "phase"
    # A value hook (Modify*, Should*, TryModify*) inside one pass: a sum or a product, so the order does not
    # change the value unless a listener reads the running value.
    commutes: bool = False


def _dispatcher_of(decompiled: str) -> dict[str, tuple[str, int, list[str]]]:
    out = {}
    for disp, info in ht.dispatchers(decompiled).items():
        for i, h in enumerate(info["passes"]):
            out.setdefault(h, (disp, i, info["iterator"]))
    return out


def analyse(decompiled: str, log: list[int], probes: dict[int, Probe], positions: dict,
            listener_of: dict[tuple[str, str], str]) -> tuple[list[Inversion], dict]:
    """Order inversions in one scenario's log, and coverage (probes hit, listeners seen)."""
    disp = _dispatcher_of(decompiled)
    inversions: list[Inversion] = []
    seen_listeners: set[str] = set()
    hit_probes: set[int] = set()
    for seg, items in segments(log):
        # events: (time, listener key, model, kind, candidate hooks)
        firsts: dict[tuple[str, str], int] = {}
        events = []
        screens = [t for t, pid in items if pid == SCREEN]
        for t, pid in items:
            p = probes.get(pid)
            if p is None:
                continue
            hit_probes.add(pid)
            by_listener = collections.defaultdict(set)
            for model, kind, hook in p.pairs:
                lk = listener_of.get((model, kind))
                if lk is None or lk not in positions:
                    continue
                by_listener[lk, model, kind].add(hook)
            for (lk, model, kind), hooks in by_listener.items():
                seen_listeners.add(lk)
                events.append((t, lk, model, kind, hooks))
        # 1. listener order within each dispatcher (first time each listener shows in the segment)
        by_disp = collections.defaultdict(list)
        for t, lk, model, kind, hooks in events:
            cands = collections.defaultdict(set)
            for h in hooks:
                if h in disp:
                    cands[disp[h][0]].add(disp[h][1])
            if len(cands) > 1:
                continue        # the probe stands for hooks of several dispatchers: which one ran is unknown
            for d, passes in cands.items():
                if (d, lk) in firsts:
                    continue
                firsts[d, lk] = t
                iterator = next(disp[h][2] for h in hooks if h in disp and disp[h][0] == d)
                pos = dict(positions[lk])
                if "run+combat" in iterator and pos["kind"] in ("card", "enchantment"):
                    pos["creature"] = -1
                lo = ht.game_order_key(pos, min(passes))
                hi = ht.game_order_key(pos, max(passes))
                query = all(QUERY.match(h) for h in hooks if h in disp and disp[h][0] == d)
                by_disp[d].append((t, lo, hi, (kind, ht.short(model), "/".join(sorted(hooks))), query))
        for d, evs in by_disp.items():
            evs.sort()
            # A dispatcher that fires once per event (a hit, a draw, an exhaust, a power change) can fire several
            # times in a segment, for different creatures; only listeners on one creature are compared there.
            per_event = d not in PHASE_DISPATCHERS
            for i in range(len(evs)):
                for j in range(i + 1, len(evs)):
                    if per_event and evs[i][1][1] != evs[j][1][1]:
                        continue
                    if evs[i][0] < evs[j][0] and evs[i][1] > evs[j][2]:
                        lo, hi = evs[i][1], evs[j][2]
                        k = next(n for n in range(len(lo)) if lo[n] != hi[n])
                        level = ("pass", "creature", "kind", "index")[min(k, 3)]
                        commutes = evs[i][4] and evs[j][4] and level != "pass"
                        inversions.append(Inversion(seg, "listener", d, evs[i][3], evs[j][3], level, commutes))
        # 2. phase order within the segment
        phases = PHASES.get(seg)
        if not phases:
            continue
        pe = []
        seen_phase: set[tuple[str, int]] = set()
        for t, lk, model, kind, hooks in events:
            if any(QUERY.match(h) for h in hooks):
                continue        # it may be the listener's value hook (ShouldClearBlock) rather than its phase hook
            idx = sorted({phases.index(h) for h in hooks if h in phases})
            if not idx:
                continue
            key = (lk, idx[0])
            if key in seen_phase:
                continue
            seen_phase.add(key)
            pe.append((t, idx[0], idx[-1], (kind, ht.short(model), "/".join(sorted(h for h in hooks if h in phases)))))
        pe.sort()
        side, floors = pause_floors(phases, pe, screens)
        for i in range(len(pe)):
            for j in range(i + 1, len(pe)):
                if pe[i][0] < pe[j][0] and pe[i][1] > pe[j][2]:
                    if pe[i][1] in side:
                        pause = next(((mark, floor) for mark, floor in floors if mark > pe[i][0]), None)
                        if pause is not None and pe[j][0] > pause[0] and pe[j][2] >= pause[1]:
                            continue        # the game ran on to AfterSideTurnStart while the screen waited
                    a, b = phases[pe[i][1]], phases[pe[j][2]]
                    inversions.append(Inversion(seg, "phase", f"{b} < {a}", pe[i][3], pe[j][3]))
    return inversions, {"probes_hit": sorted(hit_probes), "listeners_seen": sorted(seen_listeners)}


SIDE_START = ("AfterSideTurnStart", "AfterSideTurnStartLate")
SETUP_FIRST = "AfterEnergyReset"


def pause_floors(phases: list[str], pe: list[tuple], screens: list[int]) -> tuple[set[int], list[tuple[int, int]]]:
    """The phase indices of AfterSideTurnStart, and ``(time, floor)`` for each screen the harness answered.

    ``CombatManager.StartTurn`` awaits ``SetupPlayerTurn`` (energy reset, BeforeHandDraw, the draw,
    AfterPlayerTurnStart) only until it completes or pauses on a player's choice, then runs AfterSideTurnStart,
    and the choice is asked after that. So when a screen paused the setup, an AfterSideTurnStart listener that ran
    before the screen is answered and a setup listener that ran after it are in the game's order, as long as that
    listener is no earlier in the game's phases than where the setup had reached: ``floor``, the latest setup
    phase a listener had fired in by then (the first setup phase when none had). A listener of an earlier phase
    that ran after the screen was owed before it, and stays out of order."""
    if SETUP_FIRST not in phases:
        return set(), []
    side = {phases.index(h) for h in SIDE_START if h in phases}
    first = phases.index(SETUP_FIRST)
    out = []
    for mark in screens:
        reached = [e[2] for e in pe if e[0] < mark and first <= e[2] < min(side)]
        out.append((mark, max(reached, default=first)))
    return side, out


def summarise(inversions: list[Inversion]) -> list[dict]:
    """One row per (segment kind, check, group, first kind, second kind), with up to three example pairs.

    ``within`` marks a pair of the same kind (relic after relic): obtain order, not the kind order."""
    agg: dict[tuple, dict] = {}
    for inv in inversions:
        within = inv.level == "index"
        key = (inv.segment, inv.check, inv.group, inv.level, inv.first[0], inv.second[0], inv.commutes)
        row = agg.setdefault(key, {"segment": inv.segment, "check": inv.check, "group": inv.group, "level": inv.level,
                                   "emulator_first": inv.first[0], "game_first": inv.second[0], "within_kind": within,
                                   "commutes": inv.commutes, "pairs": 0, "examples": []})
        row["pairs"] += 1
        ex = f"{inv.first[1]}.{inv.first[2]} ran before {inv.second[1]}.{inv.second[2]}"
        if len(row["examples"]) < 3 and ex not in row["examples"]:
            row["examples"].append(ex)
    return sorted(agg.values(), key=lambda r: (r["within_kind"] or r["commutes"], r["check"], r["group"], -r["pairs"]))


def order_keys(summary: list[dict]) -> set[str]:
    """The open-list keys of the rows that can change a result: phase rows and listener rows at the pass,
    creature or kind level that do not commute. ``order <segment> <group> <level> <first>><second>``."""
    return {f"order {r['segment']} {r['group'].replace(' ', '')} {r['level']} {r['emulator_first']}>{r['game_first']}"
            for r in summary if not r["within_kind"] and not r["commutes"]}


def pair_set(inversions: list[Inversion]) -> set[tuple]:
    """Every inversion as a hashable pair, for telling two runs apart (a mutant from its base)."""
    return {(i.segment, i.check, i.group, i.first, i.second) for i in inversions}


def run_harness(copy: Path, scenarios: list[dict], work: Path, mutants: list[int] | None = None,
                timeout: float = 600.0) -> list[dict]:
    spec = {"scenarios": scenarios}
    if mutants is not None:
        spec["mutants"] = mutants
    sp = work / "scenarios.json"
    op = work / "probe_out.json"
    sp.write_text(json.dumps(spec), encoding="utf-8")
    dll = copy / "tools/emudump/bin/Release/net9.0/EmuDump.dll"
    kw = {"creationflags": BELOW_NORMAL} if os.name == "nt" else {}
    r = subprocess.run(["dotnet", str(dll), "hookprobe", str(sp), str(op)], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, env=dotnet_env(), timeout=timeout, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"EmuDump hookprobe failed ({r.returncode}): {r.stdout[-2000:]}")
    return json.loads(op.read_text(encoding="utf-8"))["scenarios"]
