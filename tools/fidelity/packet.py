"""Diagnosis packets: everything a fixer needs about one open item (or one failing fight), in one markdown file.

    python -m tools.fidelity.packet --item C3-F02 [--backlog backlog-c3.json] [--out DIR]
    python -m tools.fidelity.packet --top 10 [--backlog backlog-c3.json] [--out DIR]     the first 10 open items
    python -m tools.fidelity.packet --fight <corpus fight id | corpus seed | refset id> [--out DIR]
    python -m tools.fidelity.packet --item-file item.json [--out DIR]                     an item written by hand
    python -m tools.fidelity.packet <item id>... --groups LAB/groups.json [--lab LAB]     with the fix lab's fix kit (grades.json)
    python -m tools.fidelity.packet --top N --groups LAB/groups.json --backlog <newest>   kits for the top N open items

A packet answers, in this order, what a fixer would otherwise dig out of ten tools by hand:

 1. the smallest failing case: a stored shrink of the item's setup when there is one, else the shortest
    recorded fight (corpus, or reference set) that still diverges on the current emulator and is tied to the
    item (a seed the item cites, or the cards, enemies, powers and relics it names, weighed by how rare they
    are in the corpus), with the one-step case re-anchored from the game's own state;
 2. the first differing event inside that step: tools/fidelity/trace_diff.py on a trace re-run when a traced
    game record of the fight exists, else the re-anchored step's differing fields;
 3. the game's per-action checksum state at that step (tools/fidelity/netstate_diff.py) when the fight was
    recorded with the checksum tap and its full state;
 4. the game classes and methods involved (the item's evidence, the trace, the probes) and the emulator sites
    that implement them (tools/fidelity/sweep_table.py, the entities' hook overrides);
 5. the RNG draw difference of the step (the trace's, else the stream counters and the emulator's draws with
    their callers);
 6. the rule probe that settled the rule (referee/probes.py reports), when there is one;
 7. related ledger lines (referee/backlog/ledger.md, FINDINGS.md, earlier fix attempts).

No game process is ever started. The emulator side is replayed with EmuDump (one process, the emulator only)
on the recordings that exist. What only a game run can give (a trace of a fight that has none, the checksum
tap, a shrink of a setup) is not run: the packet says so, names the command, and counts the section partial
or missing. Each section carries a status (full, partial, missing, n/a) and a one-line reason; the head of the
packet and index.md tabulate them, so a thin packet is visible as thin.

Only identifiers, paths and line numbers are written, never game code. The game side's state and event logs
come from recordings the referee made; they stay under the referee folder, which is local and not committed.
Packets are written to the referee backlog folder (default D:/spireline/referee/backlog/packets).

With --groups (the fix lab's LAB/groups.json) each packet starts with a fix kit (tools/fidelity/kit.py): the matching
groups' minimal repros replayed on this checkout, which ingredient causes them, a generated test verified to fail here,
the game's rule pointer and the sibling sites, graded A/B/C in <out>/grades.json. Without --groups nothing changes.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import dataclasses
import datetime
import gzip
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":  # run as a script: import the package so pool workers can find this module
    sys.path.insert(0, str(ROOT))

from tools.fidelity import fight_compare as fc  # noqa: E402
from tools.fidelity import kit as kt  # noqa: E402
from tools.fidelity import netstate_diff as nd  # noqa: E402
from tools.fidelity import reanchor_compare as rc  # noqa: E402
from tools.fidelity import sweep_table as sw  # noqa: E402
from tools.fidelity import trace_diff as td  # noqa: E402
from tools.fidelity.context_pack import (  # noqa: E402
    fmt_card,
    fmt_list,
    fmt_powers,
    power_diffs,
)
from tools.fidelity.corpus import (  # noqa: E402
    Corpus,
    emudump_command,
    emulator_id,
    judge_fight,
    judge_reanchor,
)
from tools.fidelity.rng_diff import LABELLED_FLAGS, NO_INLINING  # noqa: E402
from tools.lab import stepcache as sc  # noqa: E402

DEFAULT_REFEREE = Path(os.environ.get("SPIRELINE_REFEREE", r"D:\spireline\referee"))
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
STATUS = ("full", "partial", "missing", "n/a")
SECTIONS = (  # key, title (the numbering of the module doc)
    ("case", "Smallest failing case"),
    ("event", "First differing event inside the step"),
    ("checksum", "The game's per-action checksum state at the step"),
    ("classes", "Game classes and methods, and the emulator sites that implement them"),
    ("rng", "RNG draw difference of the step"),
    ("probe", "Rule probe"),
    ("ledger", "Related ledger lines"),
)
MIN_SCORE = 4.0  # rare-name weight that ties an uncited fight to an item (rank_fights)
REFSET_ID = r"\b(?:R\d[A-Z]-[A-Z0-9]{10}-F\d+|C\d{4}-[A-Z0-9]{10}|G\d{3}o?-[A-Z0-9]{10})\b"
GAME_COMMANDS = ("Commands", "Hooks")  # decompile namespaces whose members are the game's shared commands


# ================================================================== small helpers

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def clip(s, n: int = 300) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def bare(cid) -> str:
    """A model id as the states write it: no CARD./RELIC./... prefix."""
    s = str(cid or "")
    return s.split(".", 1)[1] if re.match(r"^[A-Z]+\.", s) else s


REFEREE_BASES = [DEFAULT_REFEREE]


def rel(path: Path | str, base: Path | None = None) -> str:
    """A path without the machine's folders: relative to `base`, else to the referee folder or the fork, else the bare name."""
    p = Path(path).resolve()
    if base is not None:
        with contextlib.suppress(ValueError):
            return p.relative_to(Path(base).resolve()).as_posix()
    for b, tag in [*((x, "referee") for x in REFEREE_BASES), (ROOT, "fork")]:
        with contextlib.suppress(ValueError):
            return f"{tag}/{p.relative_to(Path(b).resolve()).as_posix()}"
    return p.name


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False).stdout.strip()


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


# ================================================================== sources

@dataclasses.dataclass
class Sources:
    """Where each kind of evidence lives. Every path may be missing: the section that needs it says so."""

    referee: Path = DEFAULT_REFEREE
    corpus: Path | None = None
    run: Path | None = None  # a corpus run's results.json; default: the newest full run of this checkout
    ledger: Path | None = None
    findings: Path | None = None
    attempts: Path | None = None
    probes: Path | None = None  # folder of <name>.report.json
    probes_more: tuple = ()  # more such folders: the lab's own probe jobs write theirs under lab/runs/<job>/probes
    trace_dirs: tuple = ()  # searched recursively for traced game records
    minimal_dirs: tuple = ()  # searched recursively for minimal/<seed>.json (referee/shrink.py)
    refset: Path | None = None  # referee/refset/<version>
    decompiled: Path | None = None
    emudump: Path | None = None
    codex_walk: Path | None = None  # folder of EmuDump codex outputs (*.jsonl), or a folder of such folders (newest wins)
    codex_data: Path | None = None  # the run histories the walk read (runs-v0.107.1.jsonl.gz)
    walker: Path = ROOT / "tools" / "emudump" / "Codex" / "Walker.cs"
    id_map: Path = ROOT / "data" / "id_map.json"
    engine: Path = ROOT / "src" / "Sts2Emulator"
    tests: Path = ROOT / "src" / "Sts2Emulator.Tests"
    sweep_map: Path = ROOT / "data" / "sweep_map.json"

    @classmethod
    def default(cls, referee: Path = DEFAULT_REFEREE, **kw) -> Sources:
        r = Path(referee)
        found = [p for p in (kw.get("emudump"), os.environ.get("SPIRELINE_EMUDUMP"),
                             ROOT / "tools/emudump/bin/Release/net9.0/EmuDump.dll") if p]
        emu = next((Path(p) for p in found if Path(p).is_file()), None)
        base = dict(referee=r, corpus=Path(os.environ.get("SPIRELINE_CORPUS", r / "corpus")),
                    ledger=r / "backlog" / "ledger.md", findings=ROOT / "referee" / "FINDINGS.md",
                    attempts=r / "backlog" / "fix-attempts.csv", probes=r / "probes",
                    probes_more=tuple(sorted((r / "lab" / "runs").glob("*-probes/probes"))) if (r / "lab" / "runs").is_dir() else (),
                    trace_dirs=(r / "jobs-trace", r / "jobs-checksum"), minimal_dirs=(r / "jobs-setup",),
                    refset=r / "refset" / "v2", decompiled=r / "decompiled", emudump=emu,
                    codex_walk=r / "codex-walk", codex_data=r.parent / "data" / "spire-codex" / "runs-v0.107.1.jsonl.gz")
        base.update({k: v for k, v in kw.items() if k != "emudump" or v is not None})
        if kw.get("emudump") is None:
            base["emudump"] = emu
        return cls(**base)


# ================================================================== names, entities, identifiers

class Names:
    """The id_map names in both vocabularies: English (Hardened Shell), Pascal (HardenedShellPower), game id."""

    KINDS = (("cards", "card"), ("enemies", "enemy"), ("powers", "power"), ("relics", "relic"), ("potions", "potion"))

    def __init__(self, id_map: dict):
        self.by_norm: dict = collections.defaultdict(list)  # norm(name) -> [(kind, key, pascal)]
        self.keys: dict = collections.defaultdict(set)  # key -> kinds
        self.pascal: dict = {}  # PascalName -> (kind, key)
        self.pascals: dict = collections.defaultdict(list)  # key -> [PascalName]
        for table, kind in self.KINDS:
            for pas in id_map.get(table, {}):
                key = fc.power(pas) if kind == "power" else fc.snake(pas)
                base = pas[: -len("Power")] if kind == "power" and pas.endswith("Power") and len(pas) > 5 else pas
                self.by_norm[norm(base)].append((kind, key, pas))
                self.keys[key].add(kind)
                self.pascal[pas] = (kind, key)
                self.pascals[key].append((kind, pas))


WORD = re.compile(r"[A-Za-z][A-Za-z0-9']*")
SNAKE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
PASCAL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+\b")
CALL = re.compile(r"\b([A-Z][A-Za-z0-9]+)\.([A-Z][A-Za-z0-9]+)\b")
# Words that are also model names but read as plain English in a backlog item.
STOP = {"Block", "Draw", "Power", "Weak", "Frail", "Poison", "Strength", "Focus", "Energy", "Retain", "Exhaust",
        "Attack", "Skill", "Unplayable", "Ethereal", "Innate", "Debuff", "Buff", "Main", "Run", "Fight", "Map",
        "Shop", "Rest", "Event", "Gold", "Max", "Card", "Cards", "Relic", "Potion", "Hook", "Order", "Hand", "Play",
        "Shuffle", "Summon", "Stun", "Heal", "Thorns", "Enemy", "Elite", "Boss", "Normal"}


def entities_in(text: str, names: Names) -> dict:
    """{key: (kind, label)} for the model names a text mentions: English runs of up to four words ("Hardened
    Shell"), game ids (SELF_FORMING_CLAY) and Pascal names (HardenedShellPower). A lone English word counts only
    when capitalised, at least four letters, and not an everyday word that is also a model name.
    """
    out: dict = {}
    for m in SNAKE.finditer(text):
        k = fc.power(m.group()) if m.group().endswith("_POWER") else m.group()
        if k in names.keys:
            out.setdefault(k, (sorted(names.keys[k])[0], m.group()))
    for m in PASCAL.finditer(text):
        hit = names.pascal.get(m.group())
        if hit:
            out.setdefault(hit[1], (hit[0], m.group()))
    words = [(m.group(), m.start(), m.end()) for m in WORD.finditer(text)]
    i = 0
    while i < len(words):
        step = 1
        for n in (4, 3, 2, 1):
            if i + n > len(words):
                continue
            gram = words[i:i + n]
            if any(text[gram[j][2]:gram[j + 1][1]].strip(" -'") for j in range(n - 1)):
                continue  # something other than spaces between the words
            hits = names.by_norm.get(norm("".join(w for w, _, _ in gram)))
            if not hits:
                continue
            if n == 1 and (len(gram[0][0]) < 4 or not gram[0][0][0].isupper() or gram[0][0] in STOP):
                continue
            for kind, key, _ in hits:
                out.setdefault(key, (kind, " ".join(w for w, _, _ in gram)))
            step = n
            break
        i += step
    return out


class Decompile:
    """Class and method lookup in the local decompile (names and paths only)."""

    def __init__(self, root: Path | None):
        self.root = Path(root) if root and Path(root).is_dir() else None
        self._files: dict | None = None
        self._text: dict = {}

    @property
    def files(self) -> dict:
        if self._files is None:
            self._files = collections.defaultdict(list)
            if self.root:
                for p in self.root.rglob("*.cs"):
                    self._files[p.stem].append(p)
        return self._files

    def text(self, p: Path) -> str:
        if p not in self._text:
            self._text[p] = p.read_text(encoding="utf-8", errors="replace")
        return self._text[p]

    def method_file(self, cls: str, method: str) -> Path | None:
        for p in self.files.get(cls, []):
            if re.search(r"\b" + re.escape(method) + r"\s*(?:<[^>()]*>)?\s*\(", self.text(p)):
                return p
        return None

    def declared(self, p: Path, method: str) -> bool:
        return bool(re.search(r"\b(?:override|static|virtual|public|private|internal|protected)\b[^;{=\n]*\b"
                              + re.escape(method) + r"\s*(?:<[^>()]*>)?\s*\(", self.text(p)))

    def overrides(self, p: Path) -> list:
        """Names of the methods a model class overrides (the hooks it listens to)."""
        return list(dict.fromkeys(re.findall(r"\boverride\s+(?:async\s+)?[\w<>\[\],.? ]+?\s+(\w+)\s*\(", self.text(p))))

    def kind_dir(self, p: Path) -> str:
        return p.relative_to(self.root).parts[0].replace("MegaCrit.Sts2.Core.", "") if self.root else ""


# ================================================================== the item and what it refers to

@dataclasses.dataclass
class Refs:
    seeds: list = dataclasses.field(default_factory=list)  # corpus seeds the item cites
    fight_ids: list = dataclasses.field(default_factory=list)  # corpus fight ids (16 hex) it cites
    refset_ids: list = dataclasses.field(default_factory=list)  # R1E-...-F08, C0139-U0U7LZ8DU2, G056o-MM7V9M80NV
    refset_groups: list = dataclasses.field(default_factory=list)  # G056o: a regression class, when the item says "refset"
    probe_ids: list = dataclasses.field(default_factory=list)  # P09
    old_ids: list = dataclasses.field(default_factory=list)  # C1-F10, C2-F04
    entities: dict = dataclasses.field(default_factory=dict)  # key -> (kind, label)
    game_calls: list = dataclasses.field(default_factory=list)  # [(Class, Method, path)] found in the decompile
    hooks: list = dataclasses.field(default_factory=list)  # Hook members the text names
    emu_names: list = dataclasses.field(default_factory=list)  # Class.Method the decompile does not have
    idents: set = dataclasses.field(default_factory=set)
    phrases: list = dataclasses.field(default_factory=list)  # 'quoted text' of the item, to find in stored stop reasons


def quoted_phrases(text: str) -> list:
    """Phrases the item puts in single or double quotes (18 to 120 characters; an apostrophe inside a word does not open one)."""
    out = re.findall(r"(?<![A-Za-z])'([^'\n]{18,120})'(?![A-Za-z])", text) + re.findall(r'"([^"\n]{18,120})"', text)
    return [norm(x) for x in out]


def stop_step(res: dict) -> int | None:
    m = re.search(r"step (\d+)", (res or {}).get("stop", ""))
    return int(m.group(1)) if m else None


def item_text(item: dict) -> str:
    return " ".join(str(item.get(k) or "") for k in ("title", "evidence", "impact_note", "subsystem", "notes"))


def refs_of(item: dict, names: Names, corpus_seeds: set, decomp: Decompile) -> Refs:
    text = item_text(item)
    r = Refs()
    toks = set(re.findall(r"\b[A-Z0-9]{10}\b", text))
    r.seeds = sorted(t for t in toks if t in corpus_seeds) + [s for s in item.get("seeds", []) if s not in toks]
    r.fight_ids = sorted(set(re.findall(r"\b[0-9a-f]{16}\b", text)) | set(item.get("fight_ids", [])))
    r.refset_ids = sorted(set(re.findall(REFSET_ID, text)) | set(item.get("refset_ids", [])))
    if re.search(r"refset|reference set", text, re.IGNORECASE):
        r.refset_groups = sorted(set(re.findall(r"\bG\d{3}o?\b", text)) | set(item.get("refset_groups", [])))
    r.probe_ids = sorted(set(re.findall(r"\bP\d\d\b", text)) | set(item.get("probe_ids", [])))
    r.old_ids = sorted(set(re.findall(r"\bC\d-[A-Z]\d\d\b", text)) - {item.get("id")})
    r.entities = entities_in(text, names)
    r.idents = set(PASCAL.findall(text))
    r.phrases = quoted_phrases(text)
    seen = set()
    for cls, meth in CALL.findall(text):
        if (cls, meth) in seen:
            continue
        seen.add((cls, meth))
        p = decomp.method_file(cls, meth) if decomp.root else None
        if p is not None:
            r.game_calls.append((cls, meth, p))
        elif cls[:1].isupper() and not cls.startswith(("Models", "Core")) and meth[:1].isupper():
            r.emu_names.append(f"{cls}.{meth}")
    emu_tokens = {t for n in r.emu_names for t in n.split(".")}
    sources = [("AbstractModel", p) for p in decomp.files.get("AbstractModel", [])] + [("Hook", p) for p in decomp.files.get("Hook", [])] if decomp.root else []  # a name declared on both is the model-level virtual that the dispatcher calls: sweep its overrides
    called = {m for _, m, _ in r.game_calls}
    for ident in sorted(r.idents - emu_tokens - called):
        where = next((w for w, p in sources if decomp.declared(p, ident)), None)
        if where:
            r.hooks.append((ident, where))
    return r


# ================================================================== emulator runs (one process, no game)

class EmuError(RuntimeError):
    pass


class Emu:
    """EmuDump over one or a few recorded fights. The only process a packet starts."""

    def __init__(self, dll: Path | None, work: Path, timeout: float = 600):
        self.dll, self.work, self.timeout = dll, Path(work), timeout
        self.calls: list = []

    @property
    def available(self) -> bool:
        return self.dll is not None

    def run(self, mode: str, fights: list, *extra: str) -> dict:
        if not self.available:
            raise EmuError("no EmuDump build")
        self.work.mkdir(parents=True, exist_ok=True)
        src = self.work / f"in-{len(self.calls)}.json"
        src.write_text(json.dumps({"results": fights}), encoding="utf-8")
        self.calls.append((mode, extra))
        env = {**os.environ, **NO_INLINING} if any(x in LABELLED_FLAGS for x in extra) else None
        proc = subprocess.run([*emudump_command(self.dll), mode, str(src), *extra], capture_output=True, check=False,
                              creationflags=BELOW_NORMAL, timeout=self.timeout, env=env)
        src.unlink(missing_ok=True)
        try:
            doc = json.loads(proc.stdout.decode("utf-8"))
        except ValueError as ex:
            raise EmuError(f"EmuDump {mode} exit {proc.returncode}: {clip(proc.stderr.decode('utf-8', 'replace'), 200)}") from ex
        if not doc.get("results") or doc["results"][0] is None:
            raise EmuError(f"EmuDump {mode} gave no result")
        return doc


# ================================================================== the reference set

class Refset:
    """referee/refset/<version>: the manifest (the regression part holds one minimal setup per known divergence class) and
    the game recordings of the newest full record (pass a), loaded only when a fight is wanted.
    """

    def __init__(self, root: Path | None):
        self.root = Path(root) if root else None
        self._order: list | None = None
        self._results: list | None = None
        self._regression: list | None = None

    @property
    def ok(self) -> bool:
        return bool(self.root and (self.root / "refs-all.json").is_file() and (self.root / "manifest.json").is_file())

    def record_dir(self) -> Path | None:
        ptr = read_json(self.root / "refs-all.json", {}) if self.root else {}
        d = Path(ptr["record"]) if ptr.get("record") else None
        return d if d and d.is_dir() else None

    def order(self) -> list:
        if self._order is None:
            d = self.record_dir()
            self._order = (read_json(d / "orders.json", {}).get("a", []) if d else [])
        return self._order

    def regression(self) -> list:
        """[{id, class, trigger, source, flags, steps_in_setup}] of the regression specs."""
        if self._regression is None:
            doc = read_json(self.root / "manifest.json", {}) if self.ok else {}
            self._regression = [{"id": x["id"], "class": x.get("class", ""), "trigger": x.get("trigger") or {}, "source": x.get("source", ""),
                                 "flags": x.get("flags", []), "reason": x.get("reason", "")}
                                for x in doc.get("specs", []) if x.get("part") == "regression"]
        return self._regression

    def load(self, ref: str) -> tuple[dict, dict] | None:
        d = self.record_dir()
        if not d or ref not in self.order():
            return None
        if self._results is None:
            self._results = read_json(d / "pass-a.out.json", {}).get("results", [])
        i = self.order().index(ref)
        rec = self._results[i] if i < len(self._results) else None
        if not rec:
            return None
        spec = next((x for x in self.regression() if x["id"] == ref), {})
        return rec, {"id": ref, "seed": rec.get("seed"), "steps": len(rec["steps"]), "needs": ["setup"], "sources": [], "flags": spec.get("flags", []),
                     "tags": [], "ref": True, "class": spec.get("class", "")}


def refset_candidates(refs: Refs, cx: Context, limit: int = 4) -> list:
    """Reference-set regression fights tied to the item, replayed now: a regression class the item names (its id, or its group
    when the item says "refset"), or whose class text and trigger (the cards played, the relics that flash) share two rare
    model names with the item. Each is run through the emulator here, so the list says which still diverge.
    """
    if not (cx.refset.ok and cx.emu.available):
        return []
    want = set(refs.entities)
    groups = set(refs.refset_groups)  # G056o names the re-recorded originals of a class, G056 its minimal setup: not the same fights
    picks = []
    for sp in cx.refset.regression():
        grp = sp["id"].split("-")[0]
        cited = sp["id"] in refs.refset_ids or grp in groups
        trig = {x for k in ("plays", "relic_flashes", "potions") for x in (sp["trigger"].get(k) or [])}
        hit = sorted(((set(entities_in(sp["class"], cx.names)) | trig) & want), key=lambda k: -cx.idf_of(k))
        score = sum(cx.idf_of(k) for k in hit)
        rare = [k for k in hit if cx.idf_of(k) >= 2.0]
        if cited or (len(rare) >= 2 and score >= MIN_SCORE):
            picks.append((not cited, grp.endswith("o"), -score, sp, hit, cited))
    picks.sort(key=lambda t: t[:3])
    out = []
    for _, _, _, sp, hit, cited in picks[:limit]:
        got = cx.refset.load(sp["id"])
        if not got:
            continue
        rec, meta = got
        case = analyse_fight(rec, cx.emu, cx.nm, None)
        step = case.step
        out.append({"id": sp["id"], "seed": rec.get("seed"), "cited": cited, "quoted": False, "live": step is not None, "score": round(sum(cx.idf_of(k) for k in hit), 1),
                    "hit": hit[:4], "step": step, "n": len(rec["steps"]), "iso": case.anchored_verdict == "DIVERGES",
                    "fd": case.first_step, "dp": (case.anchored.get("dp") or {}).get(str(step)),
                    "field": case.plain.get("field"), "card": case.plain.get("card"), "setup": True, "meta": meta, "case": case, "rec": rec})
    return out


# ================================================================== corpus, runs, ranking

@dataclasses.dataclass
class RunInfo:
    label: str
    path: Path
    commit: str
    fights: dict
    totals: dict


def find_run(src: Sources) -> RunInfo | None:
    """The corpus run to take first divergences from: the given one, else the newest full run of this checkout's
    commit (not dirty), else the newest full run.
    """
    if src.run:
        paths = [Path(src.run)]
    else:
        runs = Path(src.corpus or "") / "runs"
        paths = sorted(runs.glob("*/results.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:40] if runs.is_dir() else []
    head, best = git("rev-parse", "HEAD"), None
    for p in paths:
        doc = read_json(p)
        if not doc or doc.get("subset") or "fights" not in doc:
            continue
        em = doc.get("emulator") or {}
        info = RunInfo(doc.get("label") or p.parent.name, p, em.get("commit", ""), doc["fights"], doc.get("totals", {}))
        if src.run or (em.get("commit") == head and not em.get("dirty")):
            return info
        best = best or info
    return best


def diverges(res: dict | None) -> bool:
    return bool(res) and (res.get("fd") is not None or "D" in res.get("rv", "") or "E" in res.get("rv", "")
                          or bool(res.get("ferr") or res.get("rerr")))


def result_step(res: dict) -> int | None:
    """The step a stored result puts the first divergence at: the plain replay's, else the first re-anchored one."""
    if res.get("fd") is not None:
        return res["fd"]
    rv = res.get("rv", "")
    for k, ch in enumerate(rv):
        if ch in "DE":
            return k + 1
    return None


def fight_keys(rec: dict) -> set:
    """The model names a recording involves: enemies, powers seen, cards and potions played, setup items."""
    keys: set = set()
    for s in rec.get("steps", []):
        st = s.get("state") or {}
        for e in st.get("enemies", []):
            keys.add(e["name"])
            keys.update(fc.power(p) for p, _ in e.get("powers", []))
        keys.update(fc.power(p) for p, _ in (st.get("player") or {}).get("powers", []))
        keys.update(bare(x) for x in (st.get("relics") or []) + [x for x in (st.get("potions") or []) if x])
        a = s.get("action") or {}
        keys.update(bare(a[k]) for k in ("card", "potion") if a.get(k))
    for it in rec.get("setup") or []:
        if it.get("id"):
            keys.add(fc.power(bare(it["id"])) if it.get("cmd") == "power" else bare(it["id"]))
    keys.update(bare(c if isinstance(c, str) else c.get("id")) for c in rec.get("deck_add") or [])
    return keys


def step_keys(rec: dict, j: int | None) -> dict:
    """The model names at step j in three strengths. strong: the action's card or potion (what the step does).
    medium: the enemies in the fight and the powers on either side before and after the step. weak: relics and
    potions held, and every other name the fight involves (a held relic is in every step, so it says little about this one).
    """
    out = {"strong": set(), "medium": set(), "weak": fight_keys(rec)}
    steps = rec.get("steps", [])
    if j is None or not steps:
        return out
    for k in {max(0, j - 1), min(j, len(steps) - 1)}:
        st = steps[k].get("state") or {}
        for e in st.get("enemies", []):
            out["medium"].add(e["name"])
            out["medium"].update(fc.power(p) for p, _ in e.get("powers", []))
        out["medium"].update(fc.power(p) for p, _ in (st.get("player") or {}).get("powers", []))
    if 0 < j < len(steps):
        a = steps[j].get("action") or {}
        out["strong"].update(bare(a[k]) for k in ("card", "potion") if a.get(k))
    out["medium"] -= out["strong"]
    out["weak"] -= out["strong"] | out["medium"]
    return out


class Idf:
    """How rare each model name is across the corpus (log of fights over fights holding it)."""

    def __init__(self, corpus: Corpus, metas: list):
        self.n = max(1, len(metas))
        self.df: collections.Counter = collections.Counter()
        for m in metas:
            with contextlib.suppress(OSError, ValueError):
                self.df.update(fight_keys(corpus.read(m["id"])))

    def __call__(self, key: str) -> float:
        return math.log(self.n / (1 + self.df.get(key, 0)))


def isolated(res: dict | None, step: int | None) -> bool:
    """Whether the stored re-anchored verdict at the step is a divergence: the rule at that step is wrong from the game's own
    state, so the case is self-contained (a plain divergence whose step re-anchors clean has its cause in earlier hidden state).
    """
    rv = (res or {}).get("rv", "")
    return step is not None and 0 < step <= len(rv) and rv[step - 1] == "D"


def rank_fights(refs: Refs, corpus: Corpus, metas: list, run: RunInfo | None, idf, opening_ok: bool = False) -> list:
    """Candidate fights for an item, the smallest failing case first.

    A fight is a candidate when the current emulator still diverges on it (the newest run's result) and it is tied
    to the item. Tier 1: the item cites it (by seed or fight id): its author named it as the evidence. Tier 2: not cited,
    but the diverging step involves model names the item also names. A name is worth log(fights / fights holding it):
    5.7 for one in 10 fights of 3000, 0.5 for Strike. Names count by how the step involves them (step_keys): the step's
    card in full, enemies and powers on the field half, held relics and the rest a quarter. A tier 2 fight needs the
    step's card among the names, or two names on the field, and MIN_SCORE together; it is a guess (the name may
    be in the fight for another reason) and is only chosen when no cited fight diverges. A divergence at step 0 (the
    opening state: comparer and driver gaps) ties only to an item about it (opening_ok) or by citation. A cited
    fight that no longer diverges is listed as fixed, never chosen.
    Order: tier, then divergences that re-anchor as divergences (self-contained), then the earliest step, then the
    shortest fight.
    """
    want = set(refs.entities)
    out = []
    for m in metas:
        cited = m["seed"] in refs.seeds or m["id"] in refs.fight_ids
        res = run.fights.get(m["id"]) if run else None
        quoted = bool(res and res.get("stop") and any(ph in norm(res["stop"]) for ph in refs.phrases))
        cited = cited or quoted
        live = diverges(res) or quoted
        if not live and not cited:
            continue
        step = (stop_step(res) if quoted else None)
        if step is None:
            step = result_step(res) if res else None
        if live and step == 0 and not (cited or opening_ok):
            continue
        hit, n_strong, n_medium, score = [], 0, 0, 0.0
        if want:
            with contextlib.suppress(OSError, ValueError):
                keys = step_keys(corpus.read(m["id"]), step)
                for tier, w in (("strong", 1.0), ("medium", 0.5), ("weak", 0.25)):
                    got = sorted(want & keys[tier], key=lambda k: -idf(k))
                    score += w * sum(idf(k) for k in got)
                    n_strong += len([k for k in got if idf(k) >= 1.5]) if tier == "strong" else 0
                    n_medium += len([k for k in got if idf(k) >= 1.0]) if tier == "medium" else 0
                    hit += [k if tier != "weak" else f"({k})" for k in got[:3]]
        if not cited and not ((n_strong >= 1 or n_medium >= 2) and score >= MIN_SCORE):
            continue
        out.append({"id": m["id"], "seed": m["seed"], "cited": cited, "quoted": quoted, "live": live, "score": round(score, 1),
                    "hit": hit[:6], "step": step, "n": m["steps"], "iso": isolated(res, step),
                    "fd": (res or {}).get("fd"), "dp": ((res or {}).get("dp") or {}).get(str(step)),
                    "field": (res or {}).get("field"), "card": (res or {}).get("card"),
                    "setup": "setup" in (m.get("needs") or []), "meta": m})
    out.sort(key=lambda c: (not c["live"], not c["cited"], not c["iso"], c["step"] if c["step"] is not None else 10**6,
                            c["n"], -c["score"], c["id"]))
    return out


# ================================================================== analysing one recorded fight against the emulator

def action_text(a: dict | None) -> str:
    if a is None:
        return "start"
    k = a.get("kind")
    if k == "play":
        t = a.get("target_slot", -1)
        return f"play {a.get('card')}" + (f" -> slot {t}" if t is not None and t >= 0 else "")
    if k == "potion":
        return f"potion {a.get('potion')}"
    return k or "?"


def view_lines(v: dict, indent: str = "  ") -> list:
    """A compact view of one state: the player, the hand, the enemies."""
    if not v.get("in_combat", True):
        return [indent + "(not in combat)"]
    p = v["player"]
    out = [f"{indent}player hp {p['hp']}/{p['max_hp']} block {p['block']} energy {p['energy']} powers {fmt_powers(p['powers'])}",
           f"{indent}hand [{' '.join(fmt_card(c) for c in v.get('hand', []))}] draw {len(v.get('draw', []))} "
           f"discard {len(v.get('discard', []))} exhaust {len(v.get('exhaust', []))}"]
    for i, e in enumerate(v.get("enemies", [])):
        mv = e.get("intent")
        out.append(f"{indent}enemy[{i}] {e['name']} hp {e['hp']} block {e['block']} powers {fmt_powers(e['powers'])}"
                   + (f" intent {mv}" if mv else ""))
    return out


def masked_powers(gv_raw: dict, gv: dict, ev_raw: dict, ev: dict) -> list:
    """[(holder, side, powers)] that reanchor_compare._mask left out of the comparison: the views as they were, against the
    views the comparer judged. The mask drops a game power the emulator has no buff for, and the emulator's own helper buffs.
    """
    if not all(v.get("in_combat") for v in (gv_raw, gv, ev_raw, ev)):
        return []
    holders = [("player", gv_raw["player"], gv["player"], ev_raw["player"], ev["player"])]
    holders += [(f"enemy[{i}] {a['name']}", a, b, c, d)
                for i, (a, b, c, d) in enumerate(zip(gv_raw["enemies"], gv["enemies"], ev_raw["enemies"], ev["enemies"]))]
    out = []
    for who, g0, g1, e0, e1 in holders:
        for side, raw, kept in (("game", g0, g1), ("emulator", e0, e1)):
            gone = [p for p in raw["powers"] if p not in kept["powers"]]
            if gone:
                out.append((who, side, gone))
    return out


def mask_lines(masked: list) -> list:
    """Say what the comparer's power mask left out, so a power list shown whole is not read as the list that was compared."""
    if not masked:
        return []
    return ["- left out of the comparison by the comparer's power mask (reanchor_compare._mask drops a game power the emulator has no buff for, "
            "and the emulator's own helper buffs, for that holder): " + "; ".join(f"{who}: {side} {fmt_powers(ps)}" for who, side, ps in masked)
            + ". The power lists above are shown whole; `differ` names what is left after the mask."]


def _at(view: dict | None, path: str, default):
    """The value of a diff path (.player.powers) in a view, else `default`."""
    cur = view
    for part in path.strip(".").split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _enemy_powers(view: dict | None, i: int, default):
    """The powers of enemy i in a view, else `default`."""
    es = (view or {}).get("enemies")
    return es[i].get("powers", default) if isinstance(es, list) and i < len(es) and isinstance(es[i], dict) else default


def diff_lines(gv: dict, ev: dict, d: list, limit: int = 14) -> list:
    """The differing fields in words: piles trimmed to the differing region, powers by name, the rest by value. `gv` and `ev` are
    the views as they are; the diff `d` may be of masked copies (mask_lines says so), so power lists are shown from the views.
    """
    out = []
    for path, x, y in d[:limit]:
        top = path.strip(".").split(".")[0]
        if top in ("hand", "draw", "discard", "exhaust") and isinstance(x, list) and isinstance(y, list):
            out.append(f"- {top}:")
            out += [f"    {t}" for t in fmt_list(x, y, fmt_card)]
        elif top == "enemies" and isinstance(x, list) and isinstance(y, list):
            for i in range(max(len(x), len(y))):
                a, b = (x[i] if i < len(x) else None), (y[i] if i < len(y) else None)
                if a == b:
                    continue
                if a is None or b is None:
                    out.append(f"- enemy[{i}] {(a or b)['name']}: only the {'game' if a else 'emulator'} has it")
                    continue
                for k in ("name", "hp", "block", "powers", "intent"):
                    if a.get(k) != b.get(k):
                        if k == "powers":
                            out.append(f"- enemy[{i}] {a['name']} powers: game {fmt_powers(_enemy_powers(gv, i, a[k]))}  "
                                       f"emulator {fmt_powers(_enemy_powers(ev, i, b[k]))}  "
                                       f"(differ: {', '.join(f'{n} {p}/{q}' for n, p, q in power_diffs(a[k], b[k]))})")
                        else:
                            out.append(f"- enemy[{i}] {a['name']} {k}: game {a.get(k)}  emulator {b.get(k)}")
        elif path.endswith(".powers"):
            out.append(f"- {path.strip('.')}: game {fmt_powers(_at(gv, path, x))}  emulator {fmt_powers(_at(ev, path, y))}  "
                       f"(differ: {', '.join(f'{n} {p}/{q}' for n, p, q in power_diffs(x, y))})")
        else:
            out.append(f"- {path.strip('.')}: game {clip(x, 160)}  emulator {clip(y, 160)}")
    if len(d) > limit:
        out.append(f"- ...(+{len(d) - limit} more differing fields)")
    return out


@dataclasses.dataclass
class Case:
    """What running the emulator on one recorded fight showed."""

    rec: dict
    step: int | None = None  # the step the case is about: the first divergence, or the step a quoted stop reason names
    first_step: int | None = None  # the plain replay's own first difference (None: it matches), whatever step the case is about
    first_fields: list = dataclasses.field(default_factory=list)  # field groups differing at first_step
    plain: dict = dataclasses.field(default_factory=dict)  # corpus.judge_fight
    anchored: dict = dataclasses.field(default_factory=dict)  # corpus.judge_reanchor
    plain_diff: list = dataclasses.field(default_factory=list)  # [(path, game, emu)] at `step`, plain replay
    anchored_diff: list = dataclasses.field(default_factory=list)  # same, re-anchored step
    anchored_verdict: str = "-"
    gv_pre: dict = dataclasses.field(default_factory=dict)
    gv_post: dict = dataclasses.field(default_factory=dict)
    ev_plain: dict = dataclasses.field(default_factory=dict)
    ev_anchored: dict = dataclasses.field(default_factory=dict)  # the emulator's view as it is, before the comparer's power mask
    masked: list = dataclasses.field(default_factory=list)  # [(holder, side, powers)] the comparer's power mask left out at the step
    emu_plain: dict = dataclasses.field(default_factory=dict)  # EmuDump fight result (with rng_log)
    emu_anchored: dict = dataclasses.field(default_factory=dict)
    emu_trace: dict = dataclasses.field(default_factory=dict)  # EmuDump reanchor --trace result, emulator side only
    choice_problems: list = dataclasses.field(default_factory=list)  # selection screens at the step the replay did not match
    keys: tuple | None = None
    live: bool = False  # the emulator really ran
    note: str = ""


def analyse_fight(rec: dict, emu: Emu | None, nm: dict, stored: dict | None = None, step: int | None = None) -> Case:
    """Replay one recorded fight in the emulator (plain, and re-anchored at every step) and find its first
    divergence. Without an EmuDump the stored run result gives the step and the fields, not the values.
    """
    c = Case(rec=rec)
    steps = rec.get("steps", [])
    if emu is not None and emu.available:
        try:
            fo = emu.run("fight", [rec], "--rng-log")
            ra = emu.run("reanchor", [rec])
        except (EmuError, subprocess.TimeoutExpired) as ex:
            c.note = f"EmuDump failed: {clip(ex, 200)}"
            fo = ra = None
        if fo is not None:
            c.live = True
            e, r = fo["results"][0], ra["results"][0]
            c.emu_plain, c.emu_anchored, c.keys = e, r, rc.keys_of(ra)
            c.plain = judge_fight(rec, e, nm)
            c.anchored = judge_reanchor(rec, r, nm, c.keys)
            fd = fc.first_difference(rec, e, nm)
            # a step handed in (the one a quoted stop reason names) is the step the case is about; the plain replay's own first
            # difference is kept apart, so the packet never calls the quoted step "the first divergence"
            c.first_step, c.first_fields = (fd[0], sorted({fc_group(p) for p, _, _ in fd[1]})) if fd else (None, [])
            c.step = step if step is not None else (fd[0] if fd else result_step({**c.plain, **c.anchored}))
    if not c.live:
        c.plain, c.anchored = {"fd": (stored or {}).get("fd"), "field": (stored or {}).get("field"),
                               "card": (stored or {}).get("card")}, {"rv": (stored or {}).get("rv", "")}
        c.first_step, c.first_fields = c.plain["fd"], [c.plain["field"]] if c.plain["field"] else []
        c.step = step if step is not None else (result_step(stored) if stored else None)
        c.note = c.note or "no EmuDump build: step and fields come from the stored corpus run, no emulator values"
    if c.step is not None and 0 <= c.step < len(steps):
        c.gv_pre = fc.game_view(steps[c.step - 1]["state"]) if c.step > 0 else {}
        c.gv_post = fc.game_view(steps[c.step]["state"])
        if c.live:
            e, r = c.emu_plain, c.emu_anchored
            if c.step < len(e.get("steps", [])):
                c.ev_plain = fc.emu_view(e["steps"][c.step]["state"], nm)
                c.plain_diff = fc.diff(c.gv_post, c.ev_plain)
            if c.step < len(r.get("steps", [])):
                c.choice_problems = [x for x in r["steps"][c.step].get("choices", []) if x.get("error") or x.get("unexpected_selection")]
            if c.step < len(r.get("steps", [])) and "state" in r["steps"][c.step]:
                c.ev_anchored = fc.emu_view(r["steps"][c.step]["state"], nm)
                # the comparer judges the step on masked copies; both views stay whole here, and what the mask left out is kept
                gv, ev = json.loads(json.dumps(c.gv_post)), json.loads(json.dumps(c.ev_anchored))
                if gv.get("in_combat") and ev.get("in_combat"):
                    rc._mask(gv, ev, *(c.keys or (None, None)))
                c.anchored_diff = fc.diff(gv, ev)
                c.masked = masked_powers(c.gv_post, gv, c.ev_anchored, ev)
            rv = c.anchored.get("rv", "")
            c.anchored_verdict = "n/a (step 0 is the opening state, the emulator's own)" if c.step == 0 else {"M": "matches", "D": "DIVERGES", "E": "ERROR", "U": "unjudged", "-": "not re-anchored (stopped earlier)"}.get(
                rv[c.step - 1] if 0 < c.step <= len(rv) else "-", "-")
    return c


def case_file(rec: dict, path: Path) -> Path:
    """The fight as a referee out.json, so EmuDump (fight, reanchor) and every comparer read it as it is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": [rec]}), encoding="utf-8")
    return path


def rng_step(c: Case) -> dict:
    """Per stream at the case's step: draws the game made, draws the emulator made when put on the game's state,
    and the emulator's own draws (stream, index after, caller) in the plain replay.
    """
    out: dict = {"streams": {}, "emu_draws": []}
    j, rec = c.step, c.rec
    if j is None or not c.live or not (0 < j < len(rec.get("steps", []))):
        return out
    gb, ga = (rec["steps"][k]["state"].get("rng") or {} for k in (j - 1, j))
    rea = c.emu_anchored.get("steps", [])
    ea = (rea[j]["state"].get("rng") if j < len(rea) and "state" in rea[j] else None) or {}
    pl = c.emu_plain.get("steps", [])
    ep = (pl[j]["state"].get("rng") if j < len(pl) else None) or {}
    for s in fc.RNG_STREAMS:
        if s not in ga or s not in gb:
            continue
        g_draws = ga[s]["counter"] - gb[s]["counter"]
        a_draws = (ea[s] - gb[s]["counter"]) if s in ea else None
        p_draws = (ep[s] - gb[s]["counter"]) if s in ep else None  # meaningful while the plain replay was in step
        out["streams"][s] = {"game": g_draws, "anchored": a_draws, "plain": p_draws, "game_after": ga[s]["counter"],
                             "emu_after": ea.get(s)}
    for stream, idx, caller in (pl[j].get("rng_log", []) if j < len(pl) else []):
        out["emu_draws"].append((stream, idx, caller))
    return out


# ================================================================== traced game records

class TraceStore:
    """Traced game records on disk (referee/trace_divergent.py's traced.out.json, the checksum tap's traced runs)."""

    PATTERNS = ("traced.out.json", "*.trace.out.json")

    def __init__(self, dirs, extra: list | None = None):
        self.paths = []
        for d in dirs:
            if Path(d).is_dir():
                for pat in self.PATTERNS:
                    self.paths += sorted(Path(d).rglob(pat))
        self.paths += [Path(p) for p in (extra or [])]
        self._index: dict | None = None

    @property
    def index(self) -> dict:
        if self._index is None:
            self._index = collections.defaultdict(list)
            for p in self.paths:
                doc = read_json(p, {})
                for k, r in enumerate(doc.get("results", [])):
                    if r and r.get("seed") and any("trace" in s for s in r.get("steps", [])):
                        self._index[r["seed"]].append((p, k, len(r["steps"]), any("net" in s for s in r["steps"])))
        return self._index

    def find(self, seed: str) -> list:
        """[(path, index in file, steps, has the full checksum state)], the ones with the full state and most steps first."""
        return sorted(self.index.get(seed, []), key=lambda t: (not t[3], -t[2], str(t[0])))

    def load(self, entry) -> dict:
        return read_json(entry[0])["results"][entry[1]]


def same_recording(a: dict, b: dict, upto: int) -> int | None:
    """None when the two game records agree on actions and states up to `upto`; else the first step they do not."""
    for j in range(1, min(upto, len(a["steps"]) - 1, len(b["steps"]) - 1) + 1):
        if a["steps"][j].get("action") != b["steps"][j].get("action") or \
                fc.game_view(a["steps"][j]["state"]) != fc.game_view(b["steps"][j]["state"]):
            return j
    return None


@dataclasses.dataclass
class Traced:
    d: dict | None = None  # trace_diff.step_diff
    text: str = ""
    step: int | None = None
    path: str = ""
    net: dict | None = None  # netstate_diff.diff_fight
    has_net: bool = False
    note: str = ""
    traced_len: int = 0
    reproduces: bool = True
    emu_doc: dict | None = None


def trace_case(c: Case, store: TraceStore, emu: Emu | None, nm: dict, want_net: bool = True) -> Traced:
    """Align a traced game record of this fight with the emulator re-anchored under --trace, at the case's step."""
    t = Traced()
    seed = c.rec.get("seed")
    entries = store.find(seed)
    if not entries:
        t.note = "no traced game record of this seed on disk"
        return t
    if emu is None or not emu.available:
        t.note = f"traced game record exists ({rel(entries[0][0])}) but there is no EmuDump to replay the emulator side"
        return t
    why = []
    for entry in entries:
        g = store.load(entry)
        bad = same_recording(c.rec, g, c.step if c.step is not None else 10**6)
        if bad is not None:
            why.append(f"{rel(entry[0])} differs from the corpus recording at step {bad}")
            continue
        if bad is None and len(g["steps"]) < 2:
            continue
        try:
            doc = emu.run("reanchor", [g], "--trace-to", str(len(g["steps"])))
        except (EmuError, subprocess.TimeoutExpired) as ex:
            why.append(f"EmuDump failed on {rel(entry[0])}: {clip(ex, 120)}")
            continue
        r = doc["results"][0]
        j = c.step
        if j is None or j >= len(g["steps"]) or j >= len(r.get("steps", [])):
            j = td.first_diverging_step(g, r, nm)  # the traced step of an older divergence
            if j is not None:
                t.note = f"the traced record ends before step {c.step}; showing its own first differing step {j}"
        if j is None:
            t.note = "the traced record and the emulator agree on every step it covers"
            t.path, t.traced_len, t.has_net = rel(entry[0]), len(g["steps"]), entry[3]
            return t
        t.d = td.step_diff(g, r, j, nm, td.keys_of(doc))
        t.text, t.step, t.path, t.traced_len, t.has_net = td.text(t.d), j, rel(entry[0]), len(g["steps"]), entry[3]
        t.emu_doc = r
        if want_net and entry[3]:
            try:
                en = emu.run("fight", [g], "--netstate")["results"][0]
                t.net = nd.diff_fight(g, en, False)
            except (EmuError, subprocess.TimeoutExpired) as ex:
                t.note += f" netstate failed: {clip(ex, 120)}"
        return t
    t.note = "; ".join(why) or "traced record unusable"
    t.reproduces = False
    return t


# ================================================================== classes, methods, emulator sites, sweeps

class EngineIndex:
    """Grep index of the emulator's sources (sweep_table's), built once."""

    def __init__(self, engine: Path, tests: Path):
        self.engine, self.tests = engine, tests
        self._idx = self._tests = None

    @property
    def idx(self) -> list:
        if self._idx is None:
            self._idx = sw.emulator_index(self.engine) if self.engine.is_dir() else []
        return self._idx

    @property
    def test_idx(self) -> list:
        if self._tests is None:
            self._tests = sw.emulator_index(self.tests) if self.tests.is_dir() else []
        return self._tests

    def members(self, name: str, limit: int = 3) -> list:
        """file:line where the engine declares or assigns a field or property (assignments first)."""
        assign = re.compile(r"\b" + re.escape(name) + r"\b\s*(?:\[[^\]]*\])?\s*=(?!=)")
        decl = re.compile(r"\b(?:public|internal|private|protected)\b[^;(=]*\b" + re.escape(name) + r"\b\s*(?:\{|;|=)")
        a = [f"{r}:{ln}" for r, ln, text in self.idx if assign.search(text)]
        d = [f"{r}:{ln}" for r, ln, text in self.idx if decl.search(text)]
        return list(dict.fromkeys(d + a))[:limit]

    def defs(self, name: str, limit: int = 3) -> list:
        """file:line of the engine's declarations of a method or class."""
        pat = re.compile(r"\b" + re.escape(name) + r"\s*(?:<[^>()]*>)?\s*\(")
        cls = re.compile(r"\b(?:class|struct|record|enum)\s+" + re.escape(name) + r"\b")
        out = []
        for rel_, ln, text in self.idx:
            if (sw.METHOD.match(text) and pat.search(text)) or cls.search(text):
                out.append(f"{rel_}:{ln}")
                if len(out) >= limit:
                    break
        return out


def sweep_stamp() -> str:
    dirty = bool(git("status", "--porcelain", "--", "src/Sts2Emulator", "data/sweep_map.json"))
    return f"<!-- sweep made at fork {git('rev-parse', '--short', 'HEAD')}{' + uncommitted changes' if dirty else ''} -->"


def sweep_fresh(path: Path) -> bool:
    """A cached table is used while the fork is at the commit it was made at (the emulator sites it lists move with the code)."""
    return path.is_file() and sweep_stamp() in path.read_text(encoding="utf-8")[-300:]


def sweep_cache_name(target: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", target)


def override_rows(decompiled: Path, hook: str) -> list:
    """The model classes that override a hook (sweep_table's row shape): the game's listeners for it, whichever call dispatches them."""
    pat = re.compile(r"\boverride\s+(?:async\s+)?[\w<>\[\],.? ]+?\s+" + re.escape(hook) + r"\s*\(")
    rows = []
    for f in sorted(Path(decompiled).rglob("*.cs")):
        if "Models" not in str(f.relative_to(decompiled).parts[0]) or f.stem == "AbstractModel":
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        m = pat.search(text)
        if m:
            rel_ = f.relative_to(decompiled).as_posix()
            rows.append({"depth": 0, "via": "", "caller": f"{f.stem}.{hook}", "class": f.stem, "method": hook, "file": rel_,
                         "line": text.count("\n", 0, m.start()) + 1, "kind": sw.kind_of(rel_)})
    return rows


def make_sweep(target: str, decompiled: Path, idx: EngineIndex, mapping: dict, depth: int | None, out: Path,
               by_overrides: bool | None = None) -> dict:
    """sweep_table for one target, written in full to `out`; returns the rows and the counts. A bare hook name (no class) is swept
    by its overrides; by_overrides=False sweeps a bare name as an identifier (every use of it) instead.
    """
    if by_overrides is None:
        by_overrides = "." not in target
    rows = override_rows(decompiled, target) if by_overrides else sw.decompile_callers(decompiled, target, depth)
    built = sw.build_rows(rows, idx.idx, idx.test_idx, mapping, target, {})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sw.table(target, rows, built, mapping) + "\n" + sweep_stamp() + "\n", encoding="utf-8")
    return {"target": target, "rows": built, "n": len(rows)}


def _sweep_worker(args) -> dict:
    target, decompiled, engine, tests, map_path, depth, out = args
    mapping = read_json(map_path, {}) if map_path else {}
    return make_sweep(target, Path(decompiled), EngineIndex(Path(engine), Path(tests)), mapping, depth, Path(out))


def parse_sweep(path: Path, target: str) -> dict | None:
    """Rows of a sweep table written earlier (the cache)."""
    if not path.is_file():
        return None
    cols, rows = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == "#":
            cols = {n: i for i, n in enumerate(cells)}
        elif cols and len(cells) == len(cols) and cells[0].isdigit():
            rows.append({n: cells[i] for n, i in cols.items()})
    return {"target": target, "rows": rows, "n": len(rows)}


def sweep_summary(sw_res: dict, want: set, names: Names, link: str, limit: int = 12) -> list:
    """The part of a sweep table a fixer reads first: callers that are the item's entities, then callers with no
    emulator site at all. The whole table is the linked file.
    """
    rows = sw_res["rows"]
    by_match = collections.Counter(r["match"] for r in rows)
    what = "classes that override it" if "." not in sw_res["target"] else "game call sites"
    out = [f"- sweep `{sw_res['target']}`: {sw_res['n']} {what} ({', '.join(f'{k} {v}' for k, v in by_match.most_common())}); "
           f"full table: {link}"]

    def own(r):  # the caller's class names one of the item's entities
        cls = r["game caller"].split(".")[0]
        return any(len(norm(k)) >= 4 and norm(k) in norm(cls) for k in want)

    rows = [r for r in rows if not r["game caller"].startswith("Mock") and "autoslay" not in r["kind"]]
    if "." not in sw_res["target"]:  # an override sweep: every listener, the ones with no emulator site first
        picked = sorted(rows, key=lambda r: (r["match"] != "none", r["match"] != "weak", r["game caller"]))
        limit = max(limit, 16)
    else:
        picked = [r for r in rows if own(r)] + [r for r in rows if not own(r) and r["match"] == "none"]
    for r in picked[:limit]:
        out.append(f"  - {r['game caller']} ({r['decompile']}, {r['kind']}) -> "
                   f"{r['emulator sites'].replace('<br>', ' ; ')} [{r['match']}]" + ("  <- the item names it" if own(r) else ""))
    if len(picked) > limit:
        out.append(f"  - ...(+{len(picked) - limit} more in the full table)")
    return out


# ================================================================== probes, ledger, findings, earlier fix attempts

def load_probes(src: Sources) -> dict:
    """Every probe verdict on disk (newest report wins per probe id; each keeps its report's name, time and emulator
    commit), and the catalog's game_class by id.
    """
    out = {"verdicts": [], "names": [], "path": None, "name": None, "classes": {}, "catalog": []}
    seen = set()
    dirs = [Path(d) for d in (src.probes, *src.probes_more) if d and Path(d).is_dir()]
    reports = sorted((p for d in dirs for p in d.glob("*.report.json")), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in reports:
        doc = read_json(p)
        if not doc or not doc.get("verdicts"):
            continue
        when = datetime.datetime.fromtimestamp(p.stat().st_mtime, tz=datetime.UTC).strftime("%Y-%m-%d %H:%M")
        for v in doc["verdicts"]:
            if v["id"] not in seen:
                seen.add(v["id"])
                out["verdicts"].append(dict(v, _report=doc.get("name") or p.stem, _when=when, _commit=doc.get("commit"),
                                            _path=str(p)))
        out["names"].append(doc.get("name") or p.stem)
    counts = collections.Counter(v["_report"] for v in out["verdicts"])
    out["name"] = counts.most_common(1)[0][0] if counts else None
    out["path"] = next((Path(v["_path"]) for v in out["verdicts"] if v["_report"] == out["name"]), None)
    catalog = ROOT / "referee" / "probes.py"
    if catalog.is_file():
        with contextlib.suppress(Exception):
            spec = importlib.util.spec_from_file_location("_packet_probe_catalog", catalog)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_packet_probe_catalog"] = mod
            spec.loader.exec_module(mod)
            out["classes"] = {p.id: p.game_class for p in mod.PROBES}
            out["catalog"] = [{"id": p.id, "question": p.question, "ledger": list(p.ledger), "answer": ""} for p in mod.PROBES]
    return out


def refresh_probes(probes: dict, work: Path, emudump_ok: bool) -> dict:
    """Judge the recorded probe games again with this checkout's EmuDump (referee/probe.py run --no-game: the
    emulator only, never the game); {probe id: status now}.
    """
    if not emudump_ok or not probes.get("path"):
        return {}
    name, d = probes["name"], Path(probes["path"]).parent
    work.mkdir(parents=True, exist_ok=True)
    for ext in (".json", ".out.json"):
        if not (d / f"{name}{ext}").is_file():
            return {}
        shutil.copy2(d / f"{name}{ext}", work / f"{name}{ext}")
    proc = subprocess.run([sys.executable, str(ROOT / "referee" / "probe.py"), "run", "--name", name, "--no-game", "--dir", str(work)],
                          capture_output=True, text=True, check=False, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    doc = read_json(work / f"{name}.report.json")
    if proc.returncode != 0 or not doc:
        return {}
    return {v["id"]: v for v in doc["verdicts"]}


def probe_matches(item: dict, refs: Refs, probes: dict, names: Names, limit: int = 3) -> list:
    """Probes tied to the item: named by id (100), carrying the item's own id in their ledger (100), by its old ledger tag (60), or by sharing identifiers
    (4 each, a class or hook name) and model names (3 each) with the question and the game class.
    """
    return _probe_scored(refs, probes, names, probes.get("verdicts", []), limit, item.get("id", ""))


def probe_requests(item: dict, refs: Refs, probes: dict, names: Names, limit: int = 3) -> list[str]:
    """Ids of catalog probes that match the item by the same rule as probe_matches but have no verdict on disk yet:
    the questions idle game time should answer for this item (grades.json "probes"; the lab's probes job runs them).
    """
    done = {v["id"] for v in probes.get("verdicts", [])}
    pool = [c for c in probes.get("catalog", []) if c["id"] not in done]
    return [v["id"] for _, v, _, _ in _probe_scored(refs, probes, names, pool, limit, item.get("id", ""))]


def _probe_scored(refs: Refs, probes: dict, names: Names, pool: list, limit: int, item_id: str = "") -> list:
    out = []
    for v in pool:
        gc = probes.get("classes", {}).get(v["id"], "")
        text = f"{v.get('question', '')} {v.get('answer') or ''} {gc}"
        score, why = 0, []
        if v["id"] in refs.probe_ids:
            score, why = score + 100, why + ["named in the item"]
        if item_id and item_id in v.get("ledger", []):
            score, why = score + 100, why + ["written for this item"]
        if set(v.get("ledger", [])) & set(refs.old_ids):
            score, why = score + 60, why + ["its ledger tag is in the item"]
        known = refs.idents | {m for _, m, _ in refs.game_calls} | {c for c, _, _ in refs.game_calls} | {h for h, _ in refs.hooks}
        shared = (set(PASCAL.findall(text)) | {x for c, m in CALL.findall(text) for x in (c, m)}) & known
        if shared:
            score, why = score + 4 * len(shared), why + [f"shares {', '.join(sorted(shared))}"]
        ents = set(entities_in(text, names)) & set(refs.entities)
        if ents:
            score, why = score + 3 * len(ents), why + [f"names {', '.join(sorted(ents))}"]
        if score >= 6:
            out.append((score, v, gc, why))
    out.sort(key=lambda t: (-t[0], t[1]["id"]))
    return out[:limit]


BULLET = re.compile(r"^(\s*)[-*] +(\S.*)$")


def bullet_entries(path: Path | None, heading: str | None = None) -> list:
    """[{heading, line, text, parent}] for every bullet of a markdown file at any depth, under the `##` heading it stands in (or the
    fixed `heading` given). A nested bullet is an entry of its own (the notes under a finding hold its root cause, its siblings
    and the seeds it was checked on) matched on its own text; `parent` is the line of the top-level bullet it stands under (0 for
    a top-level one). Table rows and wrapped text are not bullets.
    """
    if not path or not Path(path).is_file():
        return []
    out, head, top = [], "", 0
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("#"):
            head, top = line.lstrip("#").strip(), 0
            continue
        m = BULLET.match(line)
        if not m:
            if line.strip() and not line[0].isspace():
                top = 0  # plain text at the left margin ends the list
            continue
        if not m.group(1):
            top = ln
        out.append({"heading": heading if heading is not None else head, "line": ln, "text": m.group(2).strip(), "parent": top if m.group(1) else 0})
    return out


def ledger_entries(path: Path | None) -> list:
    """The bullets of ledger.md (bullet_entries), each under the heading it stands in."""
    return bullet_entries(path)


class IdentWeights:
    """How telling an identifier is across the ledger and the findings: log(entries / entries naming it). CardEffects is in a
    third of them and weighs under 1; LoseHp in two weighs over 4.
    """

    def __init__(self, texts: list):
        self.n = max(1, len(texts))
        self.df: collections.Counter = collections.Counter()
        for t in texts:
            self.df.update(set(PASCAL.findall(t)))

    def __call__(self, ident: str) -> float:
        return math.log(self.n / (1 + self.df.get(ident, 0)))


def text_matches(entries: list, refs: Refs, names: Names, idf, iw=None, min_score: float = 5.0, limit: int = 6) -> list:
    """Entries (ledger bullets, findings lines) about the same thing as the item: 8 per cited seed, 4 per old item tag, the
    rarity weight of each shared model name and identifier. A model name alone does not make an entry about the item (examples
    reuse names), so an entry also needs a cited seed, an old item tag, a shared identifier of weight 4 (a method or hook:
    LoseHp, not CardEffects) or two shared model names of weight 3, and the score must reach min_score.
    """
    idents = refs.idents | {m for _, m, _ in refs.game_calls} | {c for c, _, _ in refs.game_calls} | {h for h, _ in refs.hooks}
    iw = iw or (lambda _i: 1.0)
    out = []
    for e in entries:
        t = e["text"]
        ents = set(entities_in(t, names)) & set(refs.entities)
        shared = {i for i in PASCAL.findall(t) if i in idents}
        seeds = [x for x in refs.seeds if x in t]
        tags = [o for o in refs.old_ids if o in t]
        score = sum(idf(k) for k in ents) + 8 * len(seeds) + sum(iw(i) for i in shared) + 4 * len(tags)
        about = seeds or tags or any(iw(i) >= 4.0 for i in shared) or len([k for k in ents if idf(k) >= 3.0]) >= 2
        if about and score >= min_score:
            out.append((score, e))
    out.sort(key=lambda t: (-t[0], t[1]["line"]))
    return out[:limit]


def attempts_matches(path: Path | None, refs: Refs, names: Names, idf, limit: int = 4) -> list:
    if not path or not Path(path).is_file():
        return []
    out = []
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            blob = " ".join(row.get(k) or "" for k in ("attempt_id", "branch", "bug_class", "lineage", "notes", "measured_effect"))
            ents = set(entities_in(blob, names)) & set(refs.entities)
            snake = {k for k in refs.entities if len(k) >= 5 and k in blob.upper()}
            score = sum(idf(k) for k in ents | snake) + 8 * len([s for s in refs.seeds if s in blob])
            if score >= 4.0 and (len([k for k in ents | snake if idf(k) >= 3.0]) >= 2 or any(x in blob for x in refs.seeds)):
                out.append((score, row))
    out.sort(key=lambda t: (-t[0], t[1].get("attempt_id", "")))
    return out[:limit]


# ================================================================== run-layer cases: codex walk stops

@dataclasses.dataclass
class Walk:
    """EmuDump codex output: one row per run walked, with the first difference (stop) when there is one."""

    label: str
    path: Path
    total: int
    rows: list  # the rows that stopped
    by_seed: dict


def load_walk(src: Sources) -> Walk | None:
    root = Path(src.codex_walk) if src.codex_walk else None
    if not root or not root.is_dir():
        return None
    dirs = [root] if list(root.glob("*.jsonl")) else sorted((d for d in root.iterdir() if d.is_dir() and list(d.glob("*.jsonl"))),
                                                           key=lambda d: d.stat().st_mtime, reverse=True)[:1]
    if not dirs:
        return None
    d = dirs[0]
    rows, total = [], 0
    for f in sorted(d.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            total += 1
            r = json.loads(line)
            if r.get("stop"):
                rows.append(r)
    return Walk(d.name, d, total, rows, {r["seed"]: r for r in rows})


def _words(text: str) -> set:
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in re.findall(r"[a-z]+", text.lower())}


def signature(row: dict) -> tuple:
    st = row["stop"]
    return (st.get("class", "?"), st.get("component", "?"), st.get("field", "?"))


def run_signatures(item: dict, refs: Refs, walk: Walk, limit: int = 3) -> list:
    """The stop signatures (class, component, field) whose words the item uses, best first: [(score, sig, rows, why)].
    Score: 3 for the class named (PROFILE-DEPENDENT), up to 2 for the component's words, up to 1 for the field's, 2 per RNG
    stream the stops name that the item names, 3 when a model or identifier the item names is in the stops' game/emulator values.
    """
    text = item_text(item)
    W = _words(text)
    phrases = {norm(m) for m in re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}\b", text)} | {norm(i) for i in refs.idents}
    phrases |= {norm(lab) for _, lab in refs.entities.values() if len(norm(lab)) >= 6}
    phrases = {x for x in phrases if len(x) >= 7}
    groups: dict = collections.defaultdict(list)
    for r in walk.rows:
        groups[signature(r)].append(r)
    out = []
    for sig, rows in groups.items():
        cls, comp, fld = sig
        cw, fw = _words(comp.replace("_", " ")), _words(fld)
        score, why = 0.0, []
        if re.search(r"(?<![A-Za-z-])" + re.escape(cls) + r"(?![A-Za-z-])", text):
            score, why = score + 3, why + [f"class {cls} is named"]
        cs = 2 * len(cw & W) / max(1, len(cw))
        fs = len(fw & W) / max(1, len(fw))
        if cs:
            score, why = score + cs, why + [f"component '{comp}'"]
        if fs:
            score, why = score + fs, why + [f"field '{fld}'"]
        streams = {x for r in rows for x in r["stop"].get("streams", [])} & set(re.findall(r"\b[A-Z][A-Za-z]+\b", text))
        if streams:
            score, why = score + 2 * len(streams), why + [f"stream {', '.join(sorted(streams))}"]
        hit = [r for r in rows if any(ph in norm(str(r["stop"].get("expected", "")) + str(r["stop"].get("emulator", ""))) for ph in phrases)]
        if hit:
            score, why = score + 3, why + [f"{len(hit)} stops carry a model or identifier the item names"]
            rows = hit + [r for r in rows if r not in hit]
        if score >= 4.0:
            out.append((score, sig, rows, why, len(hit)))
    out.sort(key=lambda t: (-t[0], -len(t[2]), t[1]))
    return out[:limit]


def read_codex_lines(data: Path, indices: list) -> dict:
    """{line index: record} from the run histories, in one pass (the file is gzip: no seeking)."""
    want = sorted(set(indices))
    out: dict = {}
    if not want or not data or not Path(data).is_file():
        return out
    i, k = -1, 0
    opener = gzip.open if str(data).endswith(".gz") else open
    with opener(data, "rt", encoding="utf-8") as f:
        for line in f:
            i += 1
            if i == want[k]:
                with contextlib.suppress(ValueError):
                    out[i] = json.loads(line)
                k += 1
                if k >= len(want):
                    break
    return out


def run_floor_lines(rec: dict, stop: dict, around: int = 3) -> list:
    """The run's floors up to the stop, one line each (the stop's floor and the three before it, and the first floor, where Neow is)."""
    flat = []
    for a, act in enumerate(rec.get("map_point_history", [])):
        for k, mp in enumerate(act):
            flat.append((a + 1, k + 1, mp))
    out = []
    f_stop = stop.get("floor", 0)
    for n, (a, k, mp) in enumerate(flat, 1):
        if n != 1 and not (f_stop - around <= n <= f_stop):
            continue
        ps = (mp.get("player_stats") or [{}])[0]
        room = (mp.get("rooms") or [{}])[0]
        what = bare(room.get("model_id") or "") or ",".join(bare(m) for m in room.get("monster_ids", [])) or room.get("room_type", "")
        bits = [f"hp {ps.get('current_hp')}/{ps.get('max_hp')}", f"gold {ps.get('current_gold')}"]
        if ps.get("ancient_choice"):
            bits.append("Neow/ancient: " + ", ".join(("*" if c.get("was_chosen") else "") + c.get("TextKey", "?") for c in ps["ancient_choice"]))
        if ps.get("card_choices"):
            bits.append("cards offered: " + ", ".join(("*" if c.get("was_picked") else "") + bare(c["card"]["id"]) for c in ps["card_choices"]))
        if ps.get("relic_choices"):
            bits.append("relics: " + ", ".join(("*" if c.get("was_picked") else "") + bare(c["choice"]) for c in ps["relic_choices"]))
        if ps.get("potion_choices"):
            bits.append("potions: " + ", ".join(("*" if c.get("was_picked") else "") + bare(c["choice"]) for c in ps["potion_choices"]))
        if ps.get("event_choices"):
            bits.append("event option: " + "; ".join(str(c.get("title", {}).get("key", "?")).split(".")[-2:-1][0] if "." in str(c.get("title", {}).get("key", "")) else "?" for c in ps["event_choices"]))
        out.append(f"F{n} (act {a}.{k}) {mp.get('map_point_type')} {what}: " + "; ".join(bits) + ("   <- the stop" if n == f_stop else ""))
    return out


def walker_call(component: str, field: str):
    """The pattern of a walker Check/Fail call for exactly this component and field. A stop's field is the literal the walker passed
    ("cards (fight floor)" and "cards" are two different checks, in two different branches), so it is never shortened.
    """
    return re.compile(r'\b(?:Check|Fail)\(\s*"' + re.escape(component) + r'"\s*,\s*"' + re.escape(field) + '"')


def walker_sites(walker: Path, component: str, field: str, limit: int = 3) -> list:
    """Lines of the walker that check or fail this component and field (identifiers and line numbers). A field the walker builds at
    run time has no literal call: no site is named for it.
    """
    if not Path(walker).is_file():
        return []
    pat = walker_call(component, field)
    out = [f"tools/emudump/Codex/Walker.cs:{n}" for n, line in enumerate(Path(walker).read_text(encoding="utf-8").splitlines(), 1) if pat.search(line)]
    return out[:limit]


CS_WORDS = frozenset("""var int string bool long double float decimal object new null true false this base return if else foreach for while in is as
    not and or out ref default switch case typeof nameof await async static void char byte short uint ulong""".split())


def _cs_statement(lines: list, at: int, limit: int = 8) -> str:
    """The text from line `at` to the first line holding a `;` (a statement, or an expression-bodied member)."""
    out = []
    for line in lines[at:at + limit]:
        out.append(line.strip())
        if ";" in line:
            break
    return " ".join(out)


def _cs_args(text: str) -> list:
    """The top-level arguments of the first call in `text` (quotes and brackets respected); [] when the call is not closed in it."""
    start = text.find("(")
    if start < 0:
        return []
    out, cur, depth, quote, esc = [], "", 0, False, False
    for ch in text[start + 1:]:
        if quote:
            cur += ch
            esc, quote = (False, True) if esc else (ch == "\\", ch != '"')
            continue
        if ch == '"':
            quote, cur = True, cur + ch
        elif ch in "([{":
            depth, cur = depth + 1, cur + ch
        elif ch in ")]}":
            if depth == 0:
                return out + [cur]
            depth, cur = depth - 1, cur + ch
        elif ch == "," and depth == 0:
            out, cur = out + [cur], ""
        else:
            cur += ch
    return []


def _cs_definition(lines: list, name: str, before: int, back: int) -> tuple[str, int] | None:
    """(text, line index) of what defines `name` as used at line `before`: its nearest earlier assignment inside the same member, else a
    member of the walker declared with an expression body (`List<string> DeckKeys() => ...;`). Anything else is not resolved.
    """
    assign = re.compile(r"\b" + re.escape(name) + r"\s*=(?![=>])")
    for n in range(before - 1, max(-1, before - back - 1), -1):
        if re.match(r"^    [A-Za-z].*\(.*\)\s*(=>.*)?$", lines[n]) and not lines[n].lstrip().startswith(("if", "for", "while", "switch", "foreach", "else")):
            break  # the declaration of the member the call is in: a local is not defined above it
        if assign.search(lines[n]):
            return _cs_statement(lines, n), n
    member = re.compile(r"^\s*(?:[\w<>\[\],.?]+\s+)+" + re.escape(name) + r"\s*\([^;]*\)\s*=>")
    return next(((_cs_statement(lines, n), n) for n, line in enumerate(lines) if member.search(line)), None)


def walker_members(walker: Path, component: str, field: str, back: int = 120) -> list:
    """Run-state members (S.Name) the emulator side of the walker's check of this component and field reads: in its emulator-value
    argument, and through the locals and expression-bodied helpers that argument names (the nearest earlier assignment of each).
    What the walker reads near the check for other purposes is not the state it compares, and a value that cannot be traced to S.* names nothing.
    """
    if not Path(walker).is_file():
        return []
    lines = Path(walker).read_text(encoding="utf-8").splitlines()
    pat = walker_call(component, field)
    at = next((n for n, line in enumerate(lines) if pat.search(line)), None)
    if at is None:
        return []
    args = _cs_args(_cs_statement(lines, at, limit=10))
    if len(args) < 4:
        return []
    found, seen, todo = [], set(), [(args[3], at, 0)]
    while todo:
        text, pos, depth = todo.pop(0)
        found += re.findall(r"\bS\.([A-Z]\w+)", text)
        if depth >= 3:
            continue
        bare_text = re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)
        params = set(re.findall(r"\b(\w+)\s*=>", bare_text)) | {p.strip() for g in re.findall(r"\(([\w\s,]*)\)\s*=>", bare_text) for p in g.split(",")}
        for tok in dict.fromkeys(re.findall(r"(?<![\w.])([A-Za-z_]\w*)", bare_text)):
            if tok in seen or tok in CS_WORDS or tok in params or tok == "S":
                continue
            seen.add(tok)
            got = _cs_definition(lines, tok, pos, back)
            if got:
                todo.append((got[0], got[1], depth + 1))
    return list(dict.fromkeys(found))


STRONG_SIGNATURE = 6.0  # run_signatures score from which a stop signature is taken for the item's case (class + component + field, or a model it names)


def run_case(item: dict, refs: Refs, cx: Context) -> dict | None:
    """The codex-walk case for an item: a run it cites, else the earliest run of the best-matching stop signature when that
    matches strongly (STRONG_SIGNATURE); a weaker match is only a hint.
    """
    walk = cx.walk
    if not walk:
        return None
    cited = [walk.by_seed[x] for x in sorted(set(re.findall(r"\b[A-Z0-9]{10}\b", item_text(item)))) if x in walk.by_seed]
    sigs = run_signatures(item, refs, walk)
    pool = cited or (sigs[0][2] if sigs and sigs[0][0] >= STRONG_SIGNATURE else [])
    row = min(pool, key=lambda r: (r["stop"].get("act", 9), r["stop"].get("floor", 99), r.get("floors_recorded", 0), r["seed"]), default=None)
    return {"walk": walk, "sigs": sigs, "row": row, "cited": bool(cited), "weak": bool(sigs) and not row}


def run_lines(rc_: dict, cx: Context) -> list:
    walk, L = rc_["walk"], []
    by_class = collections.Counter(r["stop"].get("class") for r in walk.rows)
    L.append(f"- codex walk {walk.label}: {walk.total} runs walked with the current emulator, {len(walk.rows)} stopped ("
             + ", ".join(f"{k} {v}" for k, v in by_class.most_common()) + ")")
    for _score, sig, rows, why, _nhit in rc_["sigs"]:
        L.append(f"  - stops of {sig[0]} {sig[1]}/{sig[2]}: {len(rows)} runs ({'; '.join(why)}); earliest: act "
                 f"{min(r['stop'].get('act', 9) for r in rows)} floor {min(r['stop'].get('floor', 99) for r in rows)}")
    row = rc_["row"]
    if not row:
        L.append("- none of the stop signatures matches the item's words well enough (score 4), and the item cites no run in the walk")
        return L
    st = row["stop"]
    L += [f"- smallest case: run `{row['seed']}` (line {row.get('index')} of the run histories), {row.get('character')} A{row.get('ascension')}, "
          f"{'won' if row.get('win') else 'lost' if not row.get('abandoned') else 'abandoned'}, {row.get('floors_recorded')} floors recorded; "
          f"{'cited by the item' if rc_.get('cited') else 'the earliest stop of the best-matching signature'}",
          f"  - stops at act {st.get('act')} floor {st.get('floor')}: {st.get('class')} {st.get('component')}/{st.get('field')}",
          f"  - the game (history) had: {clip(st.get('expected'), 300)}",
          f"  - the emulator had: {clip(st.get('emulator'), 300)}",
          f"  - RNG streams that component draws on: {', '.join(st.get('streams', [])) or 'none named'}"]
    rec = cx.codex_runs.get(row.get("index"))
    if rec:
        L.append("  - the run up to the stop (* = chosen):")
        L += ["    " + x for x in run_floor_lines(rec, st)]
    else:
        L.append("  - the run's own history is not read (no run histories on disk, or the line is out of range)")
    sites = walker_sites(cx.src.walker, st.get("component", ""), st.get("field", ""))
    if sites:
        L.append(f"  - the walker compares this at: {', '.join(sites)}")
    L.append("  - walk this run again (emulator only; the history line goes in a file of its own): "
             f"`zcat <run histories> | sed -n '{(row.get('index') or 0) + 1}p' > one.jsonl; dotnet tools/emudump/bin/Release/net9.0/EmuDump.dll "
             f"codex one.jsonl out.jsonl --decompiled <decompiled> --character {row.get('character', 'IRONCLAD')} --ascension {row.get('ascension')}`")
    return L


# ================================================================== the packet

class Section:
    def __init__(self, key: str, title: str):
        self.key, self.title = key, title
        self.status, self.why, self.lines, self.game_needed = "missing", "", [], []

    def done(self, status: str, why: str) -> Section:
        assert status in STATUS, status
        self.status, self.why = status, why
        return self


@dataclasses.dataclass
class Context:
    """Everything one run shares across packets."""

    src: Sources
    names: Names
    nm: dict
    decomp: Decompile
    corpus: Corpus | None
    metas: list
    run: RunInfo | None
    idf: Idf | None
    emu: Emu
    engine: EngineIndex
    traces: TraceStore
    probes: dict
    probes_now: dict
    ledger: list
    findings: list
    sweep_dir: Path
    do_sweep: bool = True
    sweeps: dict = dataclasses.field(default_factory=dict)  # target -> result (precomputed or cached)
    ident_w: object = None  # IdentWeights over the ledger and the findings
    refset: Refset = None
    walk: Walk | None = None
    codex_runs: dict = dataclasses.field(default_factory=dict)  # line index -> run history, filled before the packets
    run_cases: dict = dataclasses.field(default_factory=dict)  # item id -> run_case
    id_map: dict = dataclasses.field(default_factory=dict)
    kit_cfg: dict | None = None  # --groups: {"groups": doc, "lab": Path, "verify": bool}; None keeps the packet as it was
    kits: dict = dataclasses.field(default_factory=dict)  # item id -> fix kit (precompute_kits)

    @classmethod
    def build(cls, src: Sources, work: Path, sweep_dir: Path, do_sweep: bool = True, refresh: bool = False) -> Context:
        id_map = read_json(src.id_map, {})
        if Path(src.referee) not in REFEREE_BASES:
            REFEREE_BASES.append(Path(src.referee))
        corpus = Corpus(src.corpus) if src.corpus and (Path(src.corpus) / "index.json").is_file() else None
        metas = corpus.load_index()["fights"] if corpus else []
        run = find_run(src) if corpus else None
        probes = load_probes(src)
        emu = Emu(src.emudump, work / "emu")
        findings = bullet_entries(src.findings, "FINDINGS.md")
        return cls(src=src, names=Names(id_map), nm=fc.names(id_map), decomp=Decompile(src.decompiled), corpus=corpus,
                   metas=metas, run=run, idf=Idf(corpus, metas) if corpus else None, emu=emu,
                   engine=EngineIndex(src.engine, src.tests), traces=TraceStore(src.trace_dirs), probes=probes,
                   probes_now=refresh_probes(probes, work / "probes", emu.available) if refresh else {},
                   ledger=ledger_entries(src.ledger), findings=findings, sweep_dir=sweep_dir, do_sweep=do_sweep, id_map=id_map,
                   walk=load_walk(src), refset=Refset(src.refset), ident_w=IdentWeights([e["text"] for e in ledger_entries(src.ledger)] + [e["text"] for e in findings]))

    def idf_of(self, key: str) -> float:
        return self.idf(key) if self.idf else 1.0

    def load_fight(self, ref: str) -> tuple[dict, dict] | None:
        """(recording, meta) for a corpus id, a corpus seed or a reference-set id."""
        if self.corpus:
            for m in self.metas:
                if m["id"] == ref or m["seed"] == ref:
                    return self.corpus.read(m["id"]), m
        if re.match(r"^(?:R\d[A-Z]|C\d{4}|G\d{3})", ref):
            return self.refset.load(ref)
        return None


def packet_title(item: dict) -> str:
    return f"{item['id']} ({item.get('lane', '?')}, {item.get('kind', '?')}, {item.get('status', '?')}, impact {item.get('impact_runs', '?')} runs)"


def build_packet(item: dict, cx: Context, out_dir: Path, forced: list | None = None) -> dict:
    """Assemble one item's packet: returns {"item", "text", "sections": {key: (status, why)}, "files": [...]}."""
    iid = item["id"]
    files_dir = out_dir / iid
    refs = refs_of(item, cx.names, {m["seed"] for m in cx.metas}, cx.decomp)
    forced = list(forced or [])
    secs = {k: Section(k, t) for k, t in SECTIONS}
    files: list = []

    # ---- candidates, the chosen fight
    opening = about_opening(item)
    cands = rank_fights(refs, cx.corpus, cx.metas, cx.run, cx.idf, opening) if cx.corpus and cx.idf else []
    for rc_ in refset_candidates(refs, cx):
        cands.append(rc_)
    cands.sort(key=lambda c: (not c["live"], not c["cited"], not c["iso"], c["step"] if c["step"] is not None else 10**6,
                              c["n"], -c["score"], str(c["id"])))
    chosen, rec, meta = None, None, None
    for ref in forced:
        got = cx.load_fight(ref)
        if got:
            rec, meta = got
            chosen = {"id": meta["id"], "seed": meta["seed"], "cited": True, "live": True, "score": 0, "hit": [], "step": None,
                      "n": meta["steps"], "field": None, "card": None, "setup": "setup" in (meta.get("needs") or []), "meta": meta,
                      "forced": ref}
            break
    if chosen is None:
        chosen = next((c for c in cands if c["live"]), None)
        if chosen:
            rec = chosen["rec"] if chosen.get("rec") else cx.corpus.read(chosen["id"])
            meta = chosen["meta"]
    if chosen and not chosen.get("forced") and not chosen["meta"].get("ref"):
        better = prefer_traced(chosen, cands, cx)
        if better is not chosen:
            chosen, rec, meta = better, cx.corpus.read(better["id"]), better["meta"]
    stored = cx.run.fights.get(chosen["id"]) if (chosen and cx.run) else None
    case = (chosen["case"] if chosen and chosen.get("case") else
            analyse_fight(rec, cx.emu, cx.nm, stored, chosen["step"] if (chosen and chosen.get("quoted")) else None)) if rec else None
    if case and not case.live and not stored:
        case.note = case.note or "no stored run result for this fight"
    if case and case.step is None and case.live:
        case.note = "the emulator reproduces this fight step for step now (the item may be fixed, or the fight is not the cause)"
    if rec and chosen.get("forced") and case and case.step is not None:
        refs.entities.update(item_entities_from_case(case, cx.names))
    traced = trace_case(case, cx.traces, cx.emu, cx.nm) if case and rec else None
    rcase = cx.run_cases[iid] if iid in cx.run_cases else (run_case(item, refs, cx) if cx.walk else None)
    run_primary = bool(rcase and rcase.get("row") and (item.get("lane") == "run" or not rec))

    # ---- 1. smallest failing case
    s = secs["case"]
    weak_hint = []
    if rcase and rcase.get("weak") and (item.get("lane") == "run" or not rec):
        weak_hint = ["", "- the codex walk's closest stop signatures match the item's words only weakly (score under "
                     f"{STRONG_SIGNATURE:g}), so none is taken as its case: "
                     + "; ".join(f"{sig[0]} {sig[1]}/{sig[2]} ({len(rows)} runs; {', '.join(why)})" for _, sig, rows, why, _ in rcase["sigs"])]
    if run_primary:
        st = rcase["row"]["stop"]
        s.lines += ["- a run-layer case: the first difference of a codex walk (EmuDump codex: the emulator walks a real run's recorded decisions, no game)"]
        s.lines += run_lines(rcase, cx)
        if rec and case:
            s.lines += ["", "- a recorded combat fight also ties to this item:"] + case_section(item, refs, cands, chosen, meta, rec, case, cx, files_dir, files)
        s.done("full", f"codex run {rcase['row']['seed']} stops at act {st.get('act')} floor {st.get('floor')} ({st.get('class')} {st.get('component')}/{st.get('field')})"
               + ("; cited by the item" if rcase.get("cited") else "; the earliest of the best-matching stop signature"))
    else:
        s.lines += case_section(item, refs, cands, chosen, meta, rec, case, cx, files_dir, files) + weak_hint
    if run_primary:
        pass
    elif rec and case and case.step is not None:
        s.done("full" if case.live else "partial",
               f"fight {chosen['id'][:8]} seed {chosen['seed']}, divergence at step {case.step}"
               + (f" (the plain replay first differs at step {case.first_step})" if case.first_step is not None and case.first_step != case.step else "")
               + ("" if case.live else "; no emulator values (no EmuDump)"))
    elif rec and case:
        s.done("partial", f"fight {chosen['id'][:8]}: " + (case.note or "no divergence found"))
    elif item.get("setup"):
        s.done("partial", "the item's setup spec is given; no recorded fight diverges on it, shrink needs the game")
    elif not cx.corpus:
        s.done("missing", "no corpus on disk")
    else:
        s.done("missing", "no recorded fight that diverges now is tied to this item" + (
            f" ({len(refs.seeds)} cited seed{'s' if len(refs.seeds) != 1 else ''}, all identical on the current emulator: fixed since the item was written, "
            "or the divergence is above the fight, in the run layer)" if refs.seeds else "")
               + ("; the codex walk has no stop signature that matches the item's words" if cx.walk and item.get("lane") == "run" else ""))
    if item.get("lane") in ("tests",) and s.status == "missing":
        s.done("n/a", "a tests/coverage item: no fight shows it; see the mutation report and the ledger")

    # ---- 2. first differing event
    s = secs["event"]
    if run_primary:
        st = rcase["row"]["stop"]
        s.lines += [f"- the walk stops at the first difference between the run's history and what the emulator offered: act {st.get('act')} floor "
                    f"{st.get('floor')}, {st.get('component')}/{st.get('field')}: game {clip(st.get('expected'), 200)} | emulator {clip(st.get('emulator'), 200)} "
                    f"(section 1 has the run up to it). A run-layer step has no hook trace; the nearest is the stream list in section 5."]
        s.done("full", "the codex walk's stop: what the history had against what the emulator offered")
    else:
        s.lines += event_section(case, traced, chosen, cx, files_dir, files, s)
    # ---- 3. checksum state
    s = secs["checksum"]
    if run_primary:
        s.lines += ["- a run-layer case: the game's checksum tap covers combat fights only"]
        s.done("n/a", "run-layer case; the checksum tap covers combat only")
    else:
        s.lines += checksum_section(case, traced, chosen, cx, s)
    # ---- 4. classes, methods, sites
    s = secs["classes"]
    if run_primary:
        st = rcase["row"]["stop"]
        sites = walker_sites(cx.src.walker, st.get("component", ""), st.get("field", ""))
        if sites:
            s.lines += [f"- the codex walker checks {st.get('component')}/{st.get('field')} at: {', '.join(sites)}"]
        for name in walker_members(cx.src.walker, st.get("component", ""), st.get("field", ""))[:4]:
            ms = cx.engine.members(name)
            if ms:
                s.lines += [f"- the emulator state the walker compares there (S.{name}): {' ; '.join(ms)}"]
    s.lines += classes_section(item, refs, case if not run_primary else None, traced if not run_primary else None, cx, out_dir, s)
    # ---- 5. RNG
    s = secs["rng"]
    if run_primary:
        st = rcase["row"]["stop"]
        s.lines += [f"- RNG streams this stop's component draws on: {', '.join(st.get('streams', [])) or 'none named'}. The walk row records no per-stream "
                    "counters, so which stream's draws differ is not known from it; the walker's class (" + str(st.get("class")) + ") says where the cause lies "
                    "(EMULATOR: a table, an order or a draw of the emulator's own; FIGHT-DEPENDENT: a stream a resolved fight leaves behind; PROFILE-DEPENDENT: the "
                    "player's unlocks; WALKER: a screen the history cannot drive)."]
        s.done("partial", "the streams the stopped component draws on; no counters in the walk row")
    else:
        s.lines += rng_section(case, traced, s)
    # ---- 6. probe
    s = secs["probe"]
    s.lines += probe_section(item, refs, cx, s)
    # ---- 7. ledger
    s = secs["ledger"]
    s.lines += ledger_section(refs, cx, s)

    if item.get("lane") == "tests" and secs["case"].status == "n/a":
        for k in ("event", "checksum", "rng", "probe"):
            secs[k].lines = ["- not applicable: this item is about what the checks cover, not about one step of a fight"]
            secs[k].done("n/a", "tests item: no fight step")

    kit = cx.kits.get(iid)
    if kit is not None:
        try:
            finish_kit(kit, secs)
        except Exception as e:  # noqa: BLE001 - the packet is still written, its kit says why it is empty
            kit.setdefault("errors", []).append(f"the kit's text: {kit_error(e)}")
            kit["grade"], kit["missing"] = "C", list(kt.PARTS)
            kit["lines"] = ["## Fix kit [grade C]", "", f"- grade C: the kit could not be written: {'; '.join(kit['errors'][:6])}"]
    text = render(item, secs, cx, chosen, case, kit)
    pk_ = {"item": item, "text": text, "sections": {k: (v.status, v.why) for k, v in secs.items()},
           "files": [rel(f, out_dir) for f in files], "chosen": chosen["id"] if chosen else None, "refs": refs}
    if kit is not None:
        pk_["kit"] = {"grade": kit["grade"], "missing": kit["missing"], **({"errors": kit["errors"]} if kit.get("errors") else {})}
    return pk_


def about_opening(item: dict) -> bool:
    """Whether the item is about the opening state of a fight (a harness item, or one that says so): only then does a divergence at step 0 tie to it."""
    return item.get("lane") == "harness" or bool(re.search(r"\b(opening|start of the fight|at start)\b", item_text(item), re.IGNORECASE))


def prefer_traced(chosen: dict, cands: list, cx: Context) -> dict:
    """A seed is often recorded more than once (other drivers, other days), and a traced game record only matches the recordings it
    reproduces. When the chosen recording has no matching traced record but another live recording of the same seed has, take that one:
    the trace (the game's events, hooks and draws) is the most telling thing a packet can hold.
    """
    entries = cx.traces.find(chosen["seed"])
    if not entries:
        return chosen
    traced = [cx.traces.load(e) for e in entries]

    def fits(c):
        rec = cx.corpus.read(c["id"])
        upto = c["step"] if c["step"] is not None else 10**6
        return any(same_recording(rec, g, upto) is None and len(g["steps"]) > 1 for g in traced)
    if fits(chosen):
        return chosen
    pick = next((c for c in cands if c["seed"] == chosen["seed"] and c["live"] and not c["meta"].get("ref") and fits(c)), None)
    if pick is None:
        return chosen
    pick["via_trace"] = True
    return pick


def item_entities_from_case(case: Case, names: Names) -> dict:
    """For a packet about a fight alone: the model names at its divergence (the action's card, the enemies, the powers that differ)."""
    out: dict = {}
    a = (case.rec["steps"][case.step].get("action") or {}) if case.step is not None else {}
    keys = [bare(a.get("card"))] if a.get("card") else []
    keys += [e["name"] for e in case.gv_post.get("enemies", [])]
    for path, x, y in case.anchored_diff or case.plain_diff:
        if path.endswith(".powers") and isinstance(x, list) and isinstance(y, list):
            keys += [n for n, _, _ in power_diffs(x, y)]
        elif path == ".enemies" and isinstance(x, list) and isinstance(y, list):
            for ge, ee in zip(x, y):
                keys += [n for n, _, _ in power_diffs(ge.get("powers", []), ee.get("powers", []))]
    for k in keys:
        if k in names.keys:
            out[k] = (sorted(names.keys[k])[0], k)
    return out


# ------------------------------------------------------------------ section bodies

def case_section(item, refs, cands, chosen, meta, rec, case, cx, files_dir, files) -> list:
    L: list = []
    if item.get("setup"):
        L += ["- the item's own setup spec (use as is; the game decides whether it still fails):",
              "  " + clip(json.dumps(item["setup"]), 600)]
        L.append("- shrink: " + shrink_status(item, rec, cx, item.get("setup")))
    if not rec:
        cited = ", ".join(refs.seeds + refs.fight_ids + refs.refset_ids) or "none"
        L += [f"- no recorded fight is chosen. Cited by the item: {cited}.",
              ("- cited fights found in the corpus are all identical on the current emulator (see below)" if refs.seeds else
               "- the item cites no seed, and none of the corpus fights that diverge now involves enough of the model names it mentions "
               f"(weight {MIN_SCORE:g} needed; entities read from the item: "
               + (", ".join(f"{lab}" for _, lab in refs.entities.values()) or "none") + ")")]
        L += cand_table(cands, chosen)
        L += ["- to get a case: run `python referee/batch.py` / `referee/scenarios.py` for the item's setup on a game copy of your own, "
              "or add a cited seed; the packet does not start the game."]
        return L
    n_setup = len([x for x in (rec.get("setup") or []) if x.get("cmd") != "fight"])
    why = (("the item quotes this fight's stop reason" if chosen.get("quoted") else
            "the item names this reference-set fight or its group" if meta.get("ref") else "the item cites its seed") if chosen["cited"] else
           f"NOT cited by the item: no cited fight diverges now; it involves {', '.join(chosen['hit'])} (weight {chosen['score']}), "
           "which the item names - check that it is the same cause before using it")
    if chosen.get("via_trace"):
        why += "; of this seed's recordings, the one a traced game record reproduces"
    L += [f"- fight `{chosen['id']}` ({('reference set, regression class: ' + meta.get('class', '')) if meta.get('ref') else 'corpus'}), seed {rec.get('seed')}, "
          f"{rec.get('character')} A{rec.get('ascension')}, Neow pick {rec.get('neow_pick')}, room {rec.get('room')}, "
          f"{len(rec.get('steps', []))} steps" + (f", a setup fight ({n_setup} setup items)" if n_setup else ""),
          f"- why this one: {why}" + ("; the smallest of the candidates that diverge now (self-contained cases first, then earliest step, then shortest)" if len(cands) > 1 else "")]
    if case.step is None:
        L += [f"- {case.note}"]
        return L + cand_table(cands, chosen)
    j = case.step
    steps = rec["steps"]
    act = steps[j].get("action")
    first = case.first_step  # the plain replay's own first difference; j may be another step (the one a quoted stop reason names)
    if first is None:
        plain = f"none (the plain replay matches the game; step {j} is the first the re-anchored replay or a stop flags)"
    else:
        a0 = steps[first].get("action") if first < len(steps) else None
        plain = (f"step {first}" + (f" after {action_text(a0)}" if first else " (the opening state)")
                 + (f", fields {', '.join(f for f in case.first_fields if f)}" if any(case.first_fields) else "")
                 + (f"; the case below is step {j}, the step the item's quoted stop reason names" if first != j else ""))
    L += ["- first divergence, plain replay: " + plain,
          f"- the same step re-anchored on the game's state before it: {case.anchored_verdict}"
          + (" (so the rule at this step is wrong, not only the state it started from)" if case.anchored_verdict == "DIVERGES" else
             " (so the emulator's rule at this step is right and its state before the step already differed: look earlier, at hidden state)"
             if case.anchored_verdict == "matches" else "")]
    prefix = [action_text(steps[k].get("action")) for k in range(1, j + 1)]
    if j == 0:
        L.append("- the fight's opening state (before any action) differs; nothing to re-anchor")
    else:
        L.append(f"- actions to reach it ({j}): " + "; ".join(prefix[-8:]) + (f"  (the first {len(prefix) - 8} not shown)" if len(prefix) > 8 else ""))
        L.append(f"- the one-step case: from the game's state before step {j}, apply `{action_text(act)}`")
        L.append("  - state before (game):")
        L += view_lines(case.gv_pre, "    ")
    if case.live:
        diff = case.anchored_diff or case.plain_diff
        which = "re-anchored" if case.anchored_diff else "plain replay"
        L.append(f"  - result differs ({which}), game vs emulator:")
        L += ["    " + t for t in diff_lines(case.gv_post, case.ev_anchored if case.anchored_diff else case.ev_plain, diff)]
        L += ["  " + t for t in mask_lines(case.masked)]
    else:
        L.append("  - state after (game):")
        L += view_lines(case.gv_post, "    ")
        dp = ((cx.run.fights.get(chosen["id"], {}) if cx.run else {}).get("dp") or {}).get(str(j), [])
        L.append(f"  - differing field groups (stored run): {', '.join(dp) or (case.plain.get('field') if first == j else 'not stored for this step')}")
    if case.choice_problems:
        L.append("  - selection screens at the step: " + "; ".join(
            (f"the emulator opened an unexpected selection ({x['unexpected_selection']})" if x.get("unexpected_selection") else f"error: {clip(x.get('error'), 120)}")
            for x in case.choice_problems))
    if case.anchored.get("stop"):
        L.append(f"  - re-anchoring stops: {case.anchored['stop']}")
    if (meta or {}).get("flags") or (meta or {}).get("tags"):
        L.append(f"  - recording flags: {', '.join((meta.get('flags') or []) + (meta.get('tags') or []))}")
    if n_setup:
        items = [x for x in rec["setup"] if x.get("cmd") != "fight"]
        fight = next((x.get("id") for x in rec["setup"] if x.get("cmd") == "fight"), None)

        def label(x):
            return f"{x.get('cmd')} {x.get('id') or x.get('amount')}" + (f" {x.get('amount')}@{x.get('target')}" if x.get("cmd") == "power" else "")

        def named(x):
            k = bare(x.get("id") or "")
            return (fc.power(k) if x.get("cmd") == "power" else k) in refs.entities
        mine, others = [x for x in items if named(x)], [x for x in items if not named(x)]
        L.append(f"- setup ({n_setup} items, fight {fight}); the items the packet's item names: "
                 + (clip("; ".join(label(x) for x in mine), 400) or "none") + (f"; {len(others)} others" if others else ""))
        L.append("- shrink: " + shrink_status(item, rec, cx, rec["setup"]))
    cf = case_file(rec, files_dir / f"{rec.get('seed')}.case.out.json")
    files.append(cf)
    L += [f"- replay it: `dotnet tools/emudump/bin/Release/net9.0/EmuDump.dll fight {rel(cf, files_dir.parent)}` "
          f"(or `reanchor`); compare with `python -m tools.fidelity.fight_compare` / `reanchor_compare`",
          "- the recording's sources (relative to the referee folder): " + (", ".join(chosen["meta"].get("sources", [])[:3]) or "(reference set / given)")]
    L += cand_table(cands, chosen)
    return L


def fc_group(path: str) -> str:
    return rc.group(path) if path.startswith(".") else path


def cand_step(c: dict) -> str:
    """Where a candidate fails, said once for each of the two things a result holds: the plain replay's first difference
    (fd, with the field and card there) and the step the candidate is about (the first re-anchored divergence, or the step a
    quoted stop reason names). They are the same step for most fights, and not for a fight tied by a quoted stop.
    """
    step, fd = c["step"], c.get("fd")
    if fd is not None and fd == step:
        return f"step {step} {c['field']} after {c['card']}"
    if fd is not None:
        return (f"plain replay first differs at step {fd} ({c['field']} after {c['card']}); "
                f"the quoted stop reason is at step {step}")
    if c.get("quoted"):
        return f"plain replay matches; the quoted stop reason is at step {step}"
    return f"plain replay matches; the re-anchored replay first diverges at step {step}" + (f" ({', '.join(c['dp'])})" if c.get("dp") else "")


def cand_table(cands: list, chosen) -> list:
    """The candidates, recordings of the same seed that fail the same way folded into one line."""
    if len(cands) <= 1:
        return [f"- candidates: {len(cands)}"] if not cands else []
    groups: dict = {}
    for c in cands:
        groups.setdefault((c["seed"], c["live"], c["step"], c.get("fd"), c["field"], c["card"]), []).append(c)
    L = [f"- other fights tied to the item ({len(cands)} recordings, {len(groups)} distinct failure{'s' if len(groups) != 1 else ''}; "
         "id, seed, step, field, card, shape, steps, why; * = chosen):"]
    for n, (_, grp) in enumerate(groups.items()):
        if n >= 8:
            L.append(f"  ...(+{len(groups) - 8} more)")
            break
        c = grp[0]
        mark = "*" if chosen and any(x["id"] == chosen["id"] for x in grp) else " "
        state = (f"{cand_step(c)}; "
                 + ("re-anchors as a divergence" if c["iso"] else "plain only (earlier hidden state)") if c["live"]
                 else "identical now (fixed, or masked by a comparer change)")
        why = ("the item quotes its stop reason" if c.get("quoted") else "cited") if c["cited"] else f"{', '.join(c['hit'][:3])} ({c['score']})"
        ident = c["id"] if c["meta"].get("ref") else c["id"][:8]
        why += f"; reference set: {c['meta'].get('class')}" if c["meta"].get("ref") else ""
        L.append(f"  {mark} {ident} {c['seed']}" + (f" x{len(grp)}" if len(grp) > 1 else "") + f" {state}; {c['n']} steps; {why}")
    return L


def shrink_status(item, rec, cx, setup) -> str:
    n = len([x for x in setup if x.get("cmd") != "fight"]) if setup else 0
    seed = (rec or {}).get("seed") or item.get("seed")
    found = []
    for d in cx.src.minimal_dirs:
        if Path(d).is_dir() and seed:
            found += sorted(Path(d).rglob(f"minimal/{seed}.json"))
    if found:
        spec = read_json(found[0], {}).get("runs", [{}])[0]
        items = [x for x in spec.get("setup", []) if x.get("cmd") != "fight"]
        return (f"a stored shrink exists ({rel(found[0])}): {len(items)} of {n} items: "
                + clip("; ".join(f"{x.get('cmd')} {x.get('id') or x.get('amount')}" for x in items), 300) + " (it runs as a job as is)")
    if n <= 2:
        return f"the setup has {n} items, nothing to shrink"
    return (f"none stored for seed {seed}; the setup has {n} items. Shrinking replays the scripted fight in the GAME "
            "(`python referee/scenarios.py ... --shrink K`, referee/shrink.py): not run by this tool, needs a game copy of your own")


def event_section(case, traced, chosen, cx, files_dir, files, s) -> list:
    L: list = []
    if not case or case.step is None:
        s.done("missing", "no diverging step to look into")
        return ["- no diverging step (see section 1)"]
    j = case.step
    if traced and traced.d and not traced.d.get("error"):
        d = traced.d
        L += [f"- trace re-run: traced game record {traced.path} ({traced.traced_len} steps) with the emulator re-anchored under --trace "
              f"(EmuDump, this checkout), at step {traced.step}" + ("" if traced.step == j else f" (the case's step is {j}; the record ends before it)"),
              "```", *traced.text.rstrip().splitlines()[:30], "```"]
        tf = files_dir / f"{case.rec.get('seed')}.trace.txt"
        tf.parent.mkdir(parents=True, exist_ok=True)
        tf.write_text(traced.text, encoding="utf-8")
        files.append(tf)
        ok = traced.step == j and traced.reproduces and not d.get("unsettled")
        s.done("full" if ok else "partial",
               f"trace_diff at step {traced.step} of {traced.path}" + ("" if traced.step == j else f"; case step is {j}")
               + ("; the game's record of the step is UNSETTLED (a driver artifact, not an emulator difference)" if d.get("unsettled") else ""))
        if d.get("unsettled"):
            L.append("- WARNING: the game's record of this step is unsettled: the driver snapshotted it before the game finished the turn "
                     "(referee/driver/Combat.cs Idle). What differs above is not evidence against the emulator; the step needs a record taken after the game settles.")
        return L
    # fallback: the re-anchored step's differing fields and selection problems
    note = (traced.note if traced else "no trace store")
    if traced and traced.d and traced.d.get("error"):
        note = traced.d["error"]
    diff = case.anchored_diff or case.plain_diff
    if case.live and (diff or case.choice_problems):
        L += [f"- no usable game trace for this fight ({note}); what differs at the {'re-anchored' if case.anchored_diff else 'plain'} step {j} "
              "is the nearest thing (section 1 has it in full):"]
        L += ["  " + t for t in diff_lines(case.gv_post, case.ev_anchored if case.anchored_diff else case.ev_plain, diff, limit=8)]
        L += ["  " + t for t in mask_lines(case.masked)]
        if case.choice_problems:
            L.append("  - selection screens: " + "; ".join(
                (f"the emulator opened an unexpected selection ({x['unexpected_selection']})" if x.get("unexpected_selection")
                 else f"error: {clip(x.get('error'), 120)}") for x in case.choice_problems))
        L += emu_side_trace(case, cx)
        spec_src = [(cx.src.referee / x).as_posix() for x in (chosen or {}).get("meta", {}).get("sources", [])]
        L += ["- NEEDS A GAME RUN (not run here): a traced re-run records the game's event order, hooks and RNG draws in the step:",
              f"  `python referee/trace_divergent.py <source out.json of the fight: {', '.join(spec_src[:2]) or 'see section 1'}> "
              f"--game <your own game copy> --seed {case.rec.get('seed')} --max 1`  then  "
              f"`python -m tools.fidelity.trace_diff <traced out.json> <traced reanchor.json> --seed {case.rec.get('seed')} --step {j}`"]
        s.done("partial", "no usable traced game record; what differs at the re-anchored step only" + (f" ({note})" if note else ""))
        return L
    if not case.live:
        s.done("partial", "no EmuDump build: only the stored first-divergence fields (section 1)")
        return ["- no EmuDump build to replay the emulator, and no traced record: see section 1 for the stored fields"]
    s.done("partial", "the replay shows no differing field at this step" + (f" ({note})" if note else ""))
    return ["- the plain and re-anchored replays show no differing field at this step"]


def emu_side_trace(case: Case, cx: Context) -> list:
    """The emulator's own marks and draws (with callers) in the re-anchored step: EmuDump reanchor --trace, no game."""
    if not (cx.emu.available and case.step is not None):
        return []
    try:
        doc = cx.emu.run("reanchor", [case.rec], "--trace-to", str(case.step + 1))
    except (EmuError, subprocess.TimeoutExpired):
        return []
    steps = doc["results"][0].get("steps", [])
    if case.step >= len(steps) or "trace" not in steps[case.step]:
        return []
    case.emu_trace = doc["results"][0]
    tr = steps[case.step]["trace"]
    marks = [f"{e[1]} {e[2]}" if len(e) > 2 and e[2] else str(e[1]) for e in tr if e and e[0] == "M"]
    draws = td.compress([td.fmt_ev(e) for e in td.dedupe(tr) if e[0] == "R"])
    return [f"- the emulator's own side of the step (EmuDump reanchor --trace; the game's side needs a game run): marks {', '.join(marks[:8]) or '-'}; "
            f"draws {'; '.join(draws[:8]) or 'none'}"]


def checksum_section(case, traced, chosen, cx, s) -> list:
    if not case or case.step is None:
        s.done("missing", "no diverging step")
        return ["- no diverging step"]
    if traced and traced.net is not None:
        r = traced.net
        L = [f"- recorded with the checksum tap and the full state ({traced.path}); EmuDump fight --netstate against it:"]
        if "step" in r:
            L.append(f"  - first differing step {r['step']} ({nd._act(r['action'])}), {r['n_checkpoints']} game checkpoints; "
                     f"first divergent sub-action: {r['subaction']}")
            for p, gv, ev, why in r["fields"][:10]:
                L.append(f"    - {p}: game {nd._short(gv, 70)}  emulator {nd._short(ev, 70)}  [{why}]")
        else:
            L.append(f"  - {r.get('verdict')}")
        if r.get("not_modelled"):
            L.append("  - fields only the game has (not modelled): " + ", ".join(sorted({p.split("[")[0] for p in r["not_modelled"]})[:10]))
        s.done("full" if r.get("step") == case.step or "step" not in r else "partial",
               f"netstate_diff over {traced.path}" + ("" if r.get("step") == case.step else f" (its first differing step is {r.get('step')}, the case's is {case.step})"))
        return L
    has_tap = any("checksums" in st for st in case.rec.get("steps", []))
    if has_tap:
        s.done("partial", "the recording has checksums but no full state to compare")
        return ["- the recording holds the game's checksums (no full state): they identify the step, they cannot say which field"]
    note = "traced record exists but without the full state" if traced and traced.path else "recorded without the checksum tap"
    s.done("missing", f"{note}; a traced re-run records it (needs the game)")
    return [f'- {note}. NEEDS A GAME RUN (not run here): re-run the fight with `"trace": true` (trace mode also turns the checksum tap on; '
            "`python referee/trace_divergent.py ...` does it), then `python -m tools.fidelity.netstate_diff <traced out.json> <EmuDump fight --netstate>` "
            "names the game sub-action after which the emulator's value stops matching."]


def classes_section(item, refs, case, traced, cx, out_dir, s) -> list:
    L: list = []
    got = 0
    # entities the item names: decompile file, hooks it overrides, emulator sites
    shown = 0
    order = {"relic": 0, "power": 0, "potion": 0, "card": 1, "enemy": 2}
    for key, (kind, label) in sorted(refs.entities.items(), key=lambda kv: (order.get(kv[1][0], 3), -cx.idf_of(kv[0]))):
        if shown >= 8:
            break
        folder = {"card": "Cards", "enemy": "Monsters", "power": "Powers", "relic": "Relics", "potion": "Potions"}[kind]
        stem = None
        for _pkind, pas in cx.names.pascals.get(key, []):
            hit = [q for q in cx.decomp.files.get(pas, []) if f"Models.{folder}" in str(q)] if cx.decomp.root else []
            if hit:
                stem = hit[0]
                break
        sites, match = (sw.emulator_sites(cx.engine.idx, stem.stem if stem else label.replace(" ", ""),
                                          [key, label.replace(" ", "")], [], 3) if cx.engine.idx else ([], "none"))
        if not stem and not sites:
            continue
        shown += 1
        line = f"- {kind} {key}:"
        if stem:
            ov = cx.decomp.overrides(stem)
            line += f" game {rel_decomp(stem, cx)}" + (f" (overrides {', '.join(ov[:10])}{' ...' if len(ov) > 10 else ''})" if ov else "")
        line += f"; emulator {' ; '.join(sites) or 'no site found'} [{match}]"
        L.append(line)
        got += 1
    for cls, meth, p in refs.game_calls[:8]:
        L.append(f"- game {cls}.{meth}: {rel_decomp(p, cx)}" + (f"; emulator implementing: {' ; '.join(cx.engine.defs(meth))}" if cx.engine.defs(meth) else ""))
        got += 1
    for h, where in refs.hooks[:8]:
        L.append(f"- game hook {h} (declared in {where}; models override it, the dispatcher calls it)")
        got += 1
    for n in refs.emu_names[:8]:
        cls, meth = n.split(".")
        sites = cx.engine.defs(meth)
        if sites:
            L.append(f"- emulator {n} (named by the item): {' ; '.join(sites)}")
            got += 1
    # from the trace: hooks and draw callers
    if traced and traced.d and traced.d.get("segment"):
        seg = traced.d["segment"]
        hooks = list(seg.get("game_hooks", []))[:12]
        if hooks:
            L.append("- game hooks fired where the trace first differs: " + "; ".join(hooks))
            got += 1
        callers = set()
        callers.update(dd[2] for dd in (traced.d.get("draw_difference") or {}).get("game", []))
        if callers:
            L.append("- game draw callers at the first differing draw: " + "; ".join(sorted(callers)))
    # probes name their game classes too
    for _, v, gc, _ in probe_matches(item, refs, cx.probes, cx.names)[:2]:
        if gc:
            L.append(f"- probe {v['id']} names: {gc}")
            got += 1
    # sweeps
    targets = sweep_targets(refs)
    mapping = read_json(cx.src.sweep_map, {})
    for t in targets:
        if mapping.get(t, {}).get("emulator"):
            L.append(f"- emulator implementation of {t} (data/sweep_map.json): " + ", ".join(f"`{x}`" for x in mapping[t]["emulator"]))
    swept = 0
    if cx.do_sweep and cx.decomp.root:
        for t in targets:
            cached = cx.sweep_dir / f"{sweep_cache_name(t)}.md"
            res = cx.sweeps.get(t) or (parse_sweep(cached, t) if sweep_fresh(cached) else None)
            if res is None and cx.do_sweep:
                with contextlib.suppress(Exception):
                    res = make_sweep(t, cx.decomp.root, cx.engine, mapping, 1, cx.sweep_dir / f"{sweep_cache_name(t)}.md")
            if res:
                cx.sweeps[t] = res
                L += sweep_summary(res, set(refs.entities), cx.names, f"sweeps/{sweep_cache_name(t)}.md")
                swept += 1
    elif not cx.do_sweep and targets:
        L.append(f"- sweeps not run (--no-sweep): {', '.join(targets)}")
    ptr = repo_pointers(item)
    L += ptr
    got += len(ptr)
    if refs.emu_names or item.get("subsystem"):
        L.append(f"- the item's subsystem: {clip(item.get('subsystem'), 220)}")
    if not got and not swept:
        s.done("missing", "no game class or method could be read from the item or the trace")
    elif targets and not swept:
        s.done("partial", "classes and sites listed; the sweep table was not built" + ("" if cx.do_sweep else " (--no-sweep)"))
    elif not targets:
        s.done("partial", "classes and sites listed; the item names no game command to sweep")
    else:
        s.done("full", f"{got} classes/methods with emulator sites, {swept} sweep tables")
    return L


_TRACKED: list = []


def tracked_files() -> list:
    if not _TRACKED:
        _TRACKED.extend(git("ls-files").splitlines())
    return _TRACKED


def repo_pointers(item: dict, limit: int = 6) -> list:
    """Files of the fork the item names (path or bare file name) and the RUNBOOK sections that share its words."""
    text = item_text(item)
    out, seen = [], set()
    for tok in re.findall(r"[A-Za-z0-9_./-]+\.(?:py|cs|md|json|sh)\b", text):
        hits = [f for f in tracked_files() if f == tok or f.endswith("/" + tok)]
        for f in hits[:2]:
            if f not in seen and (ROOT / f).is_file():
                seen.add(f)
                out.append(f"- the code the item is about: {f} ({len((ROOT / f).read_text(encoding='utf-8', errors='replace').splitlines())} lines)")
    rb = ROOT / "referee" / "RUNBOOK.md"
    if rb.is_file():
        W = {w for w in _words(text) if len(w) >= 5}
        heads = [(n, ln[3:].strip()) for n, ln in enumerate(rb.read_text(encoding="utf-8").splitlines(), 1) if ln.startswith("## ")]
        scored = sorted(((len(W & {w for w in _words(h) if len(w) >= 5}), n, h) for n, h in heads), reverse=True)
        for sc, n, h in scored[:2]:
            if sc >= 2:
                out.append(f"- RUNBOOK section: '{h}' (referee/RUNBOOK.md line {n})")
    return out[:limit]


def sweep_targets(refs: Refs) -> list:
    """What to sweep: the game commands the item names (Class.Method in the Commands or Hooks namespaces), then the model-level
    hooks it names (every override and every dispatch of the hook), at most three.
    """
    ts = [f"{c}.{m}" for c, m, p in refs.game_calls if any(ns in str(p) for ns in GAME_COMMANDS)]
    ts += [f"Hook.{h}" if where == "Hook" else h for h, where in refs.hooks]
    return list(dict.fromkeys(ts))[:3]


def rel_decomp(p: Path, cx: Context) -> str:
    return p.relative_to(cx.decomp.root).as_posix().replace("MegaCrit.Sts2.Core.", "") if cx.decomp.root else p.name


def rng_section(case, traced, s) -> list:
    L: list = []
    if not case or case.step is None:
        s.done("missing", "no diverging step")
        return ["- no diverging step"]
    if traced and traced.d and traced.d.get("draw_difference"):
        dd = traced.d["draw_difference"]
        fmt = lambda xs: "; ".join(f"#{c} by {who}" for _, c, who in xs) or "none"  # noqa: E731
        s.done("full", f"trace_diff: first differing draw on {dd['stream']}")
        return [f"- FIRST DRAW DIFFERENCE on {dd['stream']} after {dd['after_same']} equal draws ({dd['game_total']} in the game, "
                f"{dd['emu_total']} in the emulator): game {fmt(dd['game'])} | emulator {fmt(dd['emu'])} (step {traced.step})"]
    r = rng_step(case)
    differing = {k: v for k, v in r["streams"].items() if (v["anchored"] is not None and v["anchored"] != v["game"])
                 or (v["anchored"] is None and v["plain"] is not None and v["plain"] != v["game"])}
    if not case.live:
        s.done("partial", "no EmuDump build: stored fields only")
        paths = [p for p in (cx_paths(case)) if p.startswith(".rng")]
        return [f"- stored differing fields at the step: {', '.join(paths) or 'no .rng stream'}"]
    if traced and traced.d and not traced.d.get("draw_difference"):
        if not differing:
            s.done("n/a", "no RNG stream differs at the step (the trace's draws agree on every stream)")
            return ["- the game's and the emulator's draws agree on every stream at this step (trace_diff)"]
    if not differing:
        s.done("n/a", "no RNG stream's draw count differs at the step")
        return ["- no combat RNG stream's draw count differs at this step (game vs the re-anchored emulator)"]
    L.append("- streams whose draw count differs at the step (game draws / emulator draws when put on the game's state):")
    for k, v in differing.items():
        L.append(f"  - {k}: game {v['game']}, emulator {v['anchored'] if v['anchored'] is not None else v['plain']} (counter after: game {v['game_after']}, emulator {v['emu_after']})")
    mine = [(s_, i, c) for s_, i, c in r["emu_draws"] if s_ in differing]
    if mine:
        L.append("  - the emulator's draws on those streams in the plain replay of the step (stream, index after, caller): "
                 + "; ".join(f"{a} #{b} by {c}" for a, b, c in mine[:10]))
    s.done("partial", "draw counts per stream and the emulator's callers; the game's callers need a traced game record (needs the game)")
    L.append("- the game's callers of those draws need a traced re-run (NEEDS A GAME RUN, see section 2); rng_diff compares two emulator "
             "builds, so it is for checking a fix does not move draws, not for game vs emulator")
    return L


def cx_paths(case: Case) -> list:
    return ["." + (case.plain.get("field") or "")] if case.plain.get("field") else []


def probe_section(item, refs, cx, s) -> list:
    ms = probe_matches(item, refs, cx.probes, cx.names)
    if not cx.probes.get("verdicts"):
        s.done("missing", "no probe report on disk")
        return ["- no probe report on disk"]
    if not ms:
        s.done("missing", "no probe covers this rule; a new probe needs a game run (referee/probes.py, `probe.py run`)")
        return [f"- not probed: none of the {len(cx.probes['verdicts'])} probes on disk ({', '.join(cx.probes['names'])}) "
                "shares an id, a ledger tag, a game class or a model name with this item. A probe is a setup fight plus a script that "
                "measures one number in the real game (referee/probes.py); writing and running one needs a game copy."]
    L = []
    for _score, v, gc, why in ms:
        now = cx.probes_now.get(v["id"])
        L.append(f"- {v['id']} [{', '.join(v.get('ledger', []))}] {v['question']}  (matched: {'; '.join(why)})")
        L.append(f"  - game {v.get('game')} = {'/'.join(v.get('game_rule') or []) or '-'}; emulator when probed {v.get('emulator')} = "
                 f"{'/'.join(v.get('emulator_rule') or []) or '-'} -> {v['status']} (report {v['_report']}, {v['_when']}, emulator {v['_commit']})")
        if now:
            L.append(f"  - emulator now (probe.py run --no-game on this checkout's EmuDump): {now.get('emulator')} -> {now['status']}")
        if v.get("answer"):
            L.append(f"  - answer: {v['answer']}")
        if v.get("game_trace"):
            L.append(f"  - game did: {clip(v['game_trace'], 260)}")
        if gc:
            L.append(f"  - game class: {gc}")
        if v.get("note"):
            L.append(f"  - note: {clip(v['note'], 240)}")
    direct = [v for _, v, _, why in ms if any(w.startswith("named in the item") for w in why)]
    s.done("full" if direct else "partial", f"{len(ms)} probe(s): " + ", ".join(v["id"] for _, v, _, _ in ms)
           + ("" if direct else " (related by ledger tag, shared class or model names; the item names none, so none is known to settle its rule)"))
    return L


def nested(e: dict) -> str:
    """Where a nested bullet stands: under which top-level line."""
    return f", a note under line {e['parent']}" if e.get("parent") else ""


def ledger_section(refs, cx, s) -> list:
    L = []
    idf = cx.idf_of
    led = text_matches(cx.ledger, refs, cx.names, idf, cx.ident_w)
    fnd = text_matches(cx.findings, refs, cx.names, idf, cx.ident_w, limit=5)
    att = attempts_matches(cx.src.attempts, refs, cx.names, idf)
    for _, e in led:
        L.append(f"- ledger ({e['heading'][:48]}, line {e['line']}{nested(e)}): {clip(e['text'], 420)}")
    for _, e in fnd:
        L.append(f"- FINDINGS.md line {e['line']}{nested(e)}: {clip(e['text'], 300)}")
    for _, r in att:
        L.append(f"- earlier fix attempt {r.get('attempt_id')} on {r.get('branch')}: {r.get('outcome')}"
                 + (f" ({r.get('rejection_causes')})" if r.get("rejection_causes") else "")
                 + (f"; {clip(r.get('notes'), 160)}" if r.get("notes") else ""))
    if not L:
        s.done("missing", "no ledger, FINDINGS or attempt line shares a seed, class or rare model name with the item")
        return ["- none found"]
    s.done("full", f"{len(led)} ledger, {len(fnd)} FINDINGS, {len(att)} earlier attempts")
    return L


# ------------------------------------------------------------------ rendering

def render(item: dict, secs: dict, cx: Context, chosen, case, kit: dict | None = None) -> str:
    em = cx.emu.dll
    info = {}
    if em:
        with contextlib.suppress(Exception):
            info = emulator_id(Path(em))
    head = [f"# Diagnosis packet {item['id']}", "", f"**{clip(item.get('title'), 400)}**", "",
            f"- item: {packet_title(item)}; subsystem: {clip(item.get('subsystem'), 200)}",
            f"- made {datetime.date.today().isoformat()} by tools/fidelity/packet.py at fork {git('rev-parse', '--short', 'HEAD')}"
            + (" (uncommitted changes)" if info.get("dirty") else "")
            + f"; EmuDump {'sha ' + info['emudump_sha256'][:8] if info.get('emudump_sha256') else 'not available'}"
            + (" STALE (a source is newer than the build)" if info.get("stale_build") else "")
            + (f"; first divergences from corpus run {cx.run.label}" if cx.run else "; no corpus run"),
            "- no game process was started for this packet", ""]
    if kit is not None:  # --groups: the fix kit comes first, it is what the fixer reads before anything else
        head += kit["lines"] + [""]
    head += ["## Completeness", "", "| # | section | status | reason |", "|---|---|---|---|"]
    for n, (k, t) in enumerate(SECTIONS, 1):
        head.append(f"| {n} | {t} | {secs[k].status} | {secs[k].why or '-'} |")
    head += ["", "## The item", "", f"> {clip(item.get('evidence'), 1800)}", "",
             f"> impact: {clip(item.get('impact_note'), 500)}", ""]
    body = []
    for n, (k, t) in enumerate(SECTIONS, 1):
        body += [f"## {n}. {t} [{secs[k].status}]", ""] + (secs[k].lines or ["- (nothing)"]) + [""]
    return "\n".join(head + body)


def completeness(sections: dict) -> str:
    c = collections.Counter(v[0] for v in sections.values())
    return ", ".join(f"{c[s]} {s}" for s in STATUS if c[s])


# ================================================================== fix kits (--groups; tools/fidelity/kit.py has the parts)

def kit_repros(group: dict, cx: Context, index: kt.RecordIndex) -> list:
    """The group's examples (smallest first, at most kit.MAX_EXAMPLES), each with its game record from the lab and, when
    EmuDump is there, the first step this checkout gets wrong on it."""
    reps = []
    for ex in group.get("examples", [])[: kt.MAX_EXAMPLES]:
        ref = ex.get("ref")
        seed = ex.get("seed") or (ref if isinstance(ref, str) and kt.SEED.match(ref) else None)
        rec, where = index.find(ref, ex.get("minimal_setup"), seed)
        r = {"ex": ex, "rec": rec, "where": where, "now": None, "status": "no record"}
        if rec is not None and cx.emu.available:
            try:
                fw = kt.first_wrong(rec, cx.emu.run("fight", [rec])["results"][0], cx.nm)
                r["now"], r["status"] = fw, ("reproduces" if fw else "matches")
            except (EmuError, subprocess.TimeoutExpired) as e:
                r["status"], r["err"] = "emulator error", kt.scrub(str(e))
            except Exception as e:  # noqa: BLE001 - a record the comparison cannot read is no repro, not the end of the run
                r["status"], r["err"] = "not comparable", kit_error(e)
        elif rec is not None:
            r["status"] = "not replayed (no EmuDump)"
        reps.append(r)
    return reps


def kit_implicated(group: dict, prim: dict | None) -> list:
    """(entity key, why) pairs a rule pointer starts from: the shrink's culprits, the group's entities, the card played at the
    wrong step, the fight, and the Neow pick when the divergence is at the fight's start or the shrink left the fight alone."""
    culp = kt.culprits(group)
    out = [(k, "the shrink's culprit") for k in culp] + [(e, "named by the group") for e in group.get("entities", []) if e not in culp]
    rec = (prim or {}).get("rec") or {}
    step, _d, act = prim["now"] if prim and prim.get("now") else (None, None, None)
    if act and act.get("card"):
        out.append((act["card"], "played at the wrong step"))
    if act and act.get("potion"):
        out.append((act["potion"], "used at the wrong step"))
    fight = (rec.get("fight") or {}).get("id")
    if fight:
        out.append((fight, "the fight"))
    opts, pick = rec.get("neow_options") or [], rec.get("neow_pick")
    if isinstance(pick, int) and 0 <= pick < len(opts) and (step == 0 or not culp):
        out.append((opts[pick], "the Neow pick"))
    seen, uniq = set(), []
    for k, why in out:
        if k and str(k) not in seen:
            seen.add(str(k))
            uniq.append((str(k), why))
    return uniq


def kit_sweep(target: str, cx: Context, hook_names: set) -> dict | None:
    """A sweep for the kit: a hook by its overrides, a bare method name that is no hook by its call sites, Class.Member by its uses."""
    by_overrides = "." not in target and target in hook_names
    ident = "." not in target and not by_overrides  # a bare game method: its uses, in a table of its own name
    cached = cx.sweep_dir / f"{sweep_cache_name(target)}{'.uses' if ident else ''}.md"
    key = target + (" uses" if ident else "")
    res = cx.sweeps.get(key) or (parse_sweep(cached, target) if sweep_fresh(cached) else None)
    if res is None and cx.do_sweep and cx.decomp.root:
        with contextlib.suppress(Exception):
            res = make_sweep(target, cx.decomp.root, cx.engine, read_json(cx.src.sweep_map, {}), 1, cached, by_overrides=by_overrides)
    if res:
        res["file"], res["by_overrides"] = cached.name, by_overrides
        cx.sweeps[key] = res
    return res


def precompute_kits(items: list, cx: Context, out: Path) -> None:
    """Every item's fix kit, before the packets: the matching groups' repros replayed, one scratch build for all the generated
    tests (only those that fail as stated are kept, in <packets>/<id>/kit-test/), rule pointers and sibling tables."""
    cfg = cx.kit_cfg
    index = kt.RecordIndex(cfg["lab"], {"referee/": cx.src.referee, "fork/": ROOT})
    seeds = {m["seed"] for m in cx.metas}
    hook_check = kt.load_hook_check(cx.sweep_dir / "hook_check.json", sweep_stamp(), cx.decomp.root) if cx.do_sweep else None
    hook_names = {r["hook"] for r in (hook_check or {}).get("rows", [])}
    for p in cx.decomp.files.get("AbstractModel", []) if not hook_names and cx.decomp.root else []:  # no hook table: the virtuals
        hook_names |= set(re.findall(r"\bvirtual\s+(?:async\s+)?[\w<>\[\],.? ]+?\s+(\w+)\s*\(", cx.decomp.text(p)))
    sha = git("rev-parse", "HEAD")
    tests: dict = {}
    # Each item, and each group within it, is built on its own: a part that fails (an odd record, a decompile file it cannot
    # read) costs that group or that kit its parts and leaves a reason in the kit, never the other kits or the run.
    for it in items:
        iid = it["id"]
        try:
            refs = refs_of(it, cx.names, seeds, cx.decomp)
            matches = kt.matching_groups(it, set(refs.entities), cfg["groups"])
            probes = probe_matches(it, refs, cx.probes, cx.names)
            kit = {"item": it, "sha": sha, "groups": [], "more": len(matches[kt.MAX_GROUPS:]), "hook_check": bool(hook_check),
                   "probes": [(v["id"], v.get("status"), gc) for _, v, gc, _ in probes],
                   "probe_requests": probe_requests(it, refs, cx.probes, cx.names),
                   "pointers": kt.rule_pointers([], "", None, refs.game_calls, [gc for _, _, gc, _ in probes], cx.names, cx.decomp,
                                                lambda p: rel_decomp(p, cx)),
                   "cmd_sweeps": list(sweep_targets(refs)), "errors": []}
        except Exception as e:  # noqa: BLE001
            cx.kits[iid] = {"item": it, "sha": sha, "groups": [], "more": 0, "hook_check": bool(hook_check), "probes": [], "probe_requests": [],
                            "pointers": [], "targets": [], "cmd_sweeps": [], "errors": [f"the item's kit: {kit_error(e)}"]}
            continue
        tdir = out / iid / "kit-test"
        if tdir.is_dir():
            shutil.rmtree(tdir)  # a kit's test is only ever the one verified on this checkout
        for g, how in matches[: kt.MAX_GROUPS]:
            try:
                e = kit_group(it, g, how, cx, index, hook_names, sha, tdir, tests)
            except Exception as ex:  # noqa: BLE001
                e = {"group": g, "how": how, "reps": [], "primary": None, "test": None, "implicated": [], "field": "",
                     "pointers": [], "targets": [], "hook_names": hook_names, "error": kit_error(ex)}
            if e.get("error") or e.get("test_error"):
                kit["errors"].append(f"group {g.get('id')}: {e.get('error') or e.get('test_error')}")
            kit["groups"].append(e)
        cx.kits[iid] = kit
    if tests:
        try:
            with tempfile.TemporaryDirectory(prefix="kit-") as tmp:
                res = kt.verify_tests({c: t[3] for c, t in tests.items()}, Path(cx.emu.dll), Path(tmp))
        except Exception as ex:  # noqa: BLE001
            res = {c: {"outcome": "error", "message": f"the scratch build could not run: {kit_error(ex)}"} for c in tests}
        for cls, (e, step, path, src, tdir) in tests.items():
            r = res.get(cls, {"outcome": "error", "message": "not run"})
            e["test"] = {"cls": cls, "result": r, "step": step, "path": path, "kept": kt.fails_as_stated(r, step, path), "file": None,
                         "rel": f"{tdir.parent.name}/kit-test/{cls}.cs"}
            if e["test"]["kept"]:
                tdir.mkdir(parents=True, exist_ok=True)
                (tdir / f"{cls}.cs").write_text(src, encoding="utf-8")
                e["test"]["file"] = tdir / f"{cls}.cs"
    for kit in cx.kits.values():
        for e in kit["groups"]:
            try:
                e["sweeps"] = [s for s in (kit_sweep(t, cx, e["hook_names"]) for t in e["targets"]) if s]
                models = set()
                for key, why in e["implicated"]:
                    if why in kt.MAIN:
                        models.update(p.stem for p in kt.class_files(key, cx.names, cx.decomp)[:1])
                e["hook_rows"] = kt.hook_rows(hook_check, models, {t for t in e["targets"] if t in hook_names})
            except Exception as ex:  # noqa: BLE001
                e["sweeps"], e["hook_rows"] = [], []
                kit["errors"].append(f"group {e['group'].get('id')}'s siblings: {kit_error(ex)}")
        try:  # the item's own pointers (its evidence, its probes) have siblings too
            kit["targets"] = kt.sibling_targets(kit["pointers"], "", None, cx.decomp, hook_names)
            kit["sweeps"] = [s for s in (kit_sweep(t, cx, hook_names) for t in kit["targets"]) if s]
        except Exception as ex:  # noqa: BLE001
            kit["targets"], kit["sweeps"] = [], []
            kit["errors"].append(f"the item's siblings: {kit_error(ex)}")
        try:
            kit["cmd_sweeps"] = [s for s in (kit_sweep(t, cx, hook_names) for t in kit["cmd_sweeps"]) if s]
        except Exception as ex:  # noqa: BLE001
            kit["cmd_sweeps"] = []
            kit["errors"].append(f"the item's command sweeps: {kit_error(ex)}")


def kit_error(e: BaseException) -> str:
    """One line for a kit part that failed: the exception's type and message, with no machine folders."""
    return clip(kt.scrub(f"{type(e).__name__}: {e}"), 240)


def kit_group(it: dict, g: dict, how: str, cx: Context, index: kt.RecordIndex, hook_names: set, sha: str, tdir: Path,
              tests: dict) -> dict:
    """One matching group's part of a kit: its repros replayed, the implicated entities, rule pointers, sibling targets, and
    the generated test queued in `tests` (a test that cannot be generated leaves test_error and the rest of the part)."""
    iid = it["id"]
    reps = kit_repros(g, cx, index)
    prim = next((r for r in reps if r["status"] == "reproduces"), None)
    field = prim["now"][1][0][0] if prim else str((g.get("key") or {}).get("field") or (g["examples"] or [{}])[0].get("field") or "")
    hook = kt.event_hook(g, prim["ex"] if prim else (g["examples"] or [None])[0])
    impl = kit_implicated(g, prim)
    parts = (g.get("key") or {}).get("parts") or []
    system = kt.system_rules(field or (str(parts[0]) if parts else ""))
    pointers = kt.rule_pointers(impl, field, hook, [], [], cx.names, cx.decomp, lambda p: rel_decomp(p, cx), system=system)
    e = {"group": g, "how": how, "reps": reps, "primary": prim, "test": None, "implicated": impl, "field": field,
         "pointers": pointers, "targets": kt.sibling_targets(pointers, field, hook, cx.decomp, hook_names), "hook_names": hook_names}
    if prim and cx.kit_cfg["verify"] and cx.emu.available:
        step, d, _ = prim["now"]
        cls = kt.cs_name("Kit", iid, prim["rec"].get("seed"), str(g["id"])[:8])
        meta = {"item": iid, "title": it.get("title"), "group": g["id"], "where": kt.where_text(prim["where"]), "sha": sha[:8],
                "setup": ", ".join(f"{c} {i}" for c, i in kt.setup_items(prim["rec"].get("setup")))}
        try:
            tests[cls] = (e, step, d[0][0], kt.test_source(cls, meta, prim["rec"], step, d[0][0]), tdir)
        except Exception as ex:  # noqa: BLE001
            e["test_error"] = f"the test could not be generated from {kt.where_text(prim['where'])}: {kit_error(ex)}"
    return e


def kit_sweep_lines(s: dict, want: set, limit: int = 6) -> list:
    """A sweep table in a few lines: its counts, the implicated models' rows, then rows with no or a weak emulator site."""
    rows = [r for r in s["rows"] if not r["game caller"].startswith("Mock") and "autoslay" not in r["kind"]]
    by = collections.Counter(r["match"] for r in rows)
    what = "classes that override it" if s.get("by_overrides", "." not in s["target"]) else "game sites"
    own = [r for r in rows if any(len(norm(k)) >= 4 and norm(k) in norm(r["game caller"].split(".")[0]) for k in want)]
    rest = sorted((r for r in rows if r not in own), key=lambda r: (r["match"] != "none", r["match"] != "weak", r["game caller"]))
    out = [f"- sweep `{s['target']}`: {s['n']} {what} ({', '.join(f'{k} {v}' for k, v in by.most_common())}); full table: "
           f"sweeps/{s.get('file') or sweep_cache_name(s['target']) + '.md'}"]
    for r in (own + rest)[:limit]:
        out.append(f"  - {r['game caller']} ({r['decompile']}, {r['kind']}) -> {clip(r['emulator sites'].replace('<br>', ' ; '), 160)} [{r['match']}]"
                   + ("  <- implicated" if r in own else ""))
    if len(rows) > limit:
        out.append(f"  - ...(+{len(rows) - limit} more in the full table)")
    return out


def _vals(d: list, n: int = 240) -> str:
    path, g, e = d[0]
    more = f" (+{len(d) - 1} more: {', '.join(x[0] for x in d[1:5])})" if len(d) > 1 else ""
    return f"{path}: game {clip(json.dumps(g), n)} | emulator {clip(json.dumps(e), n)}{more}"


def sibling_sweep(s: dict) -> bool:
    """A sweep that is a sibling list: built from a rule pointer (kit.sibling_targets) and specific to its mechanism."""
    return kt.specific(s.get("n") or 0)


def finish_kit(kit: dict, secs: dict) -> None:
    """Grade the kit (now that the packet's sections say what they found) and write its lines."""
    have = set()
    G = kit["groups"]
    if any(e["primary"] for e in G):
        have.add("minimal repro")
    elif any(r["ex"].get("minimal_setup") and r["ex"].get("step") is not None and r["status"] in ("no record", "not replayed (no EmuDump)")
             for e in G for r in e["reps"]):
        have.add("minimal repro")  # the lab's stored values, not checked again here
    if any(e["test"] and e["test"]["kept"] for e in G):
        have.add("failing test")
    if any("." in p["what"] for e in G for p in e["pointers"]) or any("." in p["what"] for p in kit["pointers"]):
        have.add("rule pointer")
    if any(sibling_sweep(s) for e in G for s in e.get("sweeps", [])) or any(sibling_sweep(s) for s in kit.get("sweeps", [])):
        have.add("sibling list")  # the generic sweeps of the item's hooks (kit["cmd_sweeps"]: AfterCardPlayed, 64 classes) are no sibling list
    if any(r["ex"].get("first_event") for e in G for r in e["reps"]) or secs["event"].status == "full":
        have.add("first differing event")
    if any(e["group"].get("ingredient_effect") for e in G):
        have.add("ingredient effect")
    if kit["probes"] or secs["probe"].status in ("full", "partial"):
        have.add("probe answer")
    kit["grade"], kit["missing"] = kt.grade(have)
    L = [f"## Fix kit [grade {kit['grade']}]", "",
         f"- grade {kit['grade']} (A: minimal repro + failing test verified + rule pointer + sibling list; B: minimal repro + rule pointer; "
         f"C: the rest); has {', '.join(p for p in kt.PARTS if p in have) or 'nothing'}; missing {', '.join(kit['missing']) or 'nothing'}",
         f"- {len(G)} lab group(s) match this item" + (f" (+{kit['more']} more, not expanded)" if kit["more"] else "")]
    if kit.get("errors"):
        L.append(f"- kit parts that failed (their parts are missing above; the rest stands): {'; '.join(kit['errors'][:6])}"
                 + (f" (+{len(kit['errors']) - 6} more)" if len(kit["errors"]) > 6 else ""))
    kept = [e["test"] for e in G if e["test"] and e["test"]["kept"]]
    if kept:
        t = kept[0]
        L.append(f"- start here: run the failing test, `python -m tools.fidelity.kit verify <packets>/{t['rel']}` (<packets> is this packet's folder; on fork {kit['sha'][:8]} it fails with "
                 f"\"KIT step {t['step']} {t['path']}: ...\"); read the game's rule below; fix the mechanism at every sibling site; the test "
                 "then passes, and the group's other repros should match too")
    for e in G:
        g = e["group"]
        key = g.get("key") or {}
        L += ["", f"### Group {g['id']}: {clip(g.get('title') or '-', 160)} (matched by {e['how']})", "",
              f"- key: {clip('; '.join(f'{k} ' + (', '.join(map(str, v)) if isinstance(v, list) else str(v)) for k, v in key.items()), 300) or '-'}; {g.get('count', '?')} divergences, impact ~{g.get('impact_runs_est', '?')} runs, "
              f"first seen {g.get('first_seen_utc', '?')}, relation {g.get('relation', '?')}, sources {', '.join(map(str, g.get('sources') or [])) or '-'}"]
        if e.get("error"):
            L.append(f"- this group's kit part failed: {e['error']}")
            continue
        if g.get("stale"):
            L.append(f"- STALE: {clip(g.get('host_reason') or 'a retest matches the game now', 240)} (likely nothing left to fix)")
        if g.get("host_suspect"):
            L.append(f"- HOST SUSPECT: {clip(g.get('host_reason') or 'no reason given', 240)} (the game host may be the cause, not the emulator)")
        L.append("- minimal repros (smallest first):")
        for r in e["reps"]:
            ex = r["ex"]
            setup = ", ".join(f"{c} {i}" for c, i in kt.setup_items(ex.get("minimal_setup"))) or "-"
            line = f"  - {(r['rec'] or {}).get('seed') or ex.get('ref') or '?'}: minimal setup [{setup}]; record " + (kt.where_text(r["where"]) if r["rec"] else r["where"])
            if r["status"] == "reproduces":
                step, d, act = r["now"]
                line += f"; on this checkout first wrong at step {step} (after {kt.action_label(act)}), {_vals(d)}"
            elif r["status"] == "matches":
                line += "; MATCHES the game on this checkout (fixed since the lab saw it, or the record is not the group's)"
            else:
                line += f"; {r['status']}" + (f" ({clip(r.get('err'), 160)})" if r.get("err") else "")
            L.append(line)
            if ex.get("step") is not None or ex.get("field"):
                L.append(f"    - the lab saw: step {ex.get('step')} {ex.get('field')}: game {clip(ex.get('game'), 160)} | emulator {clip(ex.get('emu'), 160)} (from {ex.get('source', '?')})")
            if ex.get("first_event"):
                L.append(f"    - first differing event: {kt.event_text(ex['first_event'])}")
        eff = g.get("ingredient_effect") or []
        if eff:
            cause = [str(x.get("item")) for x in eff if x.get("removal_vanishes")]
            inert = [str(x.get("item")) for x in eff if not x.get("removal_vanishes")]
            L.append(f"- which ingredient causes it: {', '.join(cause) or 'none alone (the fight itself, or the Neow pick)'}"
                     + (f"; removing any of {', '.join(inert[:12])}{' ...' if len(inert) > 12 else ''} leaves it" if inert else ""))
        t = e["test"]
        if t and t["kept"]:
            L.append(f"- failing test: {t['rel']} ({t['cls']}); verified on fork {kit['sha'][:8]}: FAILS with \"{clip(t['result']['message'], 300)}\"")
        elif t:
            L.append(f"- failing test: not kept, {t['result']['outcome']}: {clip(t['result']['message'], 240) or '-'} (expected \"KIT step {t['step']} {t['path']}:\")")
        elif e.get("test_error"):
            L.append(f"- failing test: not generated, {e['test_error']}")
        elif e["primary"]:
            L.append("- failing test: not generated (--no-kit-test or no EmuDump)")
        else:
            L.append("- failing test: none (no repro of this group diverges on this checkout)")
        L.append(f"- implicated: {'; '.join(f'{k} ({why})' for k, why in e['implicated']) or '-'}")
        L.append("- the game's rule (decompile names, paths and lines; no code):" if e["pointers"] else "- the game's rule: no decompiled class found for the implicated entities")
        L += [f"  - {p['what']} at {p['where']}: {p['why']}" for p in e["pointers"]]
        if e["targets"] or e.get("hook_rows"):
            L.append(f"- siblings (every game site of {', '.join(e['targets']) or 'the implicated models'}; fix the mechanism, not one model):")
            for s in e.get("sweeps", []):
                L += ["  " + x for x in kit_sweep_lines(s, {k for k, _ in e["implicated"]})]
            if e.get("hook_rows"):
                L.append("  - hook table (tools/fidelity/hook_table.py check):")
                L += ["  " + x for x in e["hook_rows"]]
        else:
            L.append("- siblings: no hook, command, stream or enum member found to sweep" + ("" if kit["hook_check"] else " (no hook table: --no-sweep or no decompile)"))
    if kit["pointers"]:
        L += ["", "- the item's own pointers (its evidence and probes):"] + [f"  - {p['what']} at {p['where']}: {p['why']}" for p in kit["pointers"]]
    if kit.get("sweeps"):
        L.append(f"- siblings of the item's pointers (every game site of {', '.join(kit.get('targets') or [])}; fix the mechanism, not one model):")
        for s in kit["sweeps"]:
            L += ["  " + x for x in kit_sweep_lines(s, set())]
    if kit["cmd_sweeps"]:
        L.append("- the item's hooks and commands, every use (context, not a sibling list):")
    for s in kit["cmd_sweeps"]:
        L += kit_sweep_lines(s, set())
    if kit["probes"]:
        L.append("- probe answers (section 6 has them in full): " + "; ".join(f"{pid} {st}" + (f" ({clip(gc, 60)})" if gc else "") for pid, st, gc in kit["probes"]))
    if kit.get("probe_requests") and not kit["probes"]:
        L.append("- probe answer pending: no probe has answered yet; the lab's probes job will run " + ", ".join(kit["probe_requests"])
                 + " in the real game (grades.json lists them); rebuild the packet after it")
    if not G and not any(x.startswith("the item's kit:") for x in kit.get("errors", [])):
        L.append("- no lab group names this item or shares its implicated entities: nothing the lab measured is tied to it yet")
    kit["lines"] = L


# ================================================================== running

def load_backlog(path: Path) -> dict:
    doc = read_json(path)
    if not doc or "items" not in doc:
        raise SystemExit(f"{path}: not a backlog file")
    return doc


def open_items(doc: dict, n: int) -> list:
    return [i for i in doc["items"] if i.get("status") == "open"][:n]


def latest_backlog(referee: Path) -> Path | None:
    files = sorted(Path(referee).joinpath("backlog").glob("backlog-c*.json"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    return files[-1] if files else None


def precompute_sweeps(items: list, cx: Context, jobs: int) -> None:
    """Build the sweep tables the items need, in parallel, before the packets (each reads the whole decompile)."""
    if not (cx.do_sweep and cx.decomp.root):
        return
    names = cx.names
    todo = {}
    for it in items:
        r = refs_of(it, names, {m["seed"] for m in cx.metas}, cx.decomp)
        ts = sweep_targets(r)
        for t in ts:
            out = cx.sweep_dir / f"{sweep_cache_name(t)}.md"
            if not sweep_fresh(out):
                todo[t] = out
    if not todo:
        return
    args = [(t, str(cx.decomp.root), str(cx.src.engine), str(cx.src.tests), str(cx.src.sweep_map), 1, str(o)) for t, o in todo.items()]
    if jobs > 1 and len(args) > 1:
        with ProcessPoolExecutor(min(jobs, len(args))) as ex:
            list(ex.map(_sweep_worker, args))
    else:
        for a in args:
            _sweep_worker(a)


def precompute_runs(items: list, cx: Context) -> None:
    """The codex-walk case of every item, and the run histories they need (read from the gzip in one pass)."""
    if not cx.walk:
        return
    seeds = {m["seed"] for m in cx.metas}
    for it in items:
        cx.run_cases[it["id"]] = run_case(it, refs_of(it, cx.names, seeds, cx.decomp), cx)
    idx = [c["row"].get("index") for c in cx.run_cases.values() if c and c.get("row") and c["row"].get("index") is not None]
    cx.codex_runs.update(read_codex_lines(cx.src.codex_data, idx))


def write_packets(items: list, cx: Context, out: Path, forced: dict | None = None, jobs: int = 4, quiet: bool = False) -> list:
    out.mkdir(parents=True, exist_ok=True)
    precompute_sweeps(items, cx, jobs)
    precompute_runs(items, cx)
    if cx.kit_cfg is not None:
        precompute_kits(items, cx, out)
    results = []
    for it in items:
        pk = build_packet(it, cx, out, (forced or {}).get(it["id"]))
        (out / f"{it['id']}.md").write_text(pk["text"], encoding="utf-8")
        results.append(pk)
        if not quiet:
            print(f"{it['id']}: {completeness(pk['sections'])}" + (f"; kit grade {pk['kit']['grade']}" if "kit" in pk else ""))
    if cx.kit_cfg is not None:  # packets/grades.json: every kit in the folder (a run of one item keeps the others')
        sha = git("rev-parse", "HEAD")
        grades = read_json(out / "grades.json", {}) or {}
        grades.update({pk_["item"]["id"]: {"grade": pk_["kit"]["grade"], "missing": pk_["kit"]["missing"], "main_sha": sha,
                                           **({"errors": pk_["kit"]["errors"]} if pk_["kit"].get("errors") else {}),
                                           **({"probes": pk_["kit"]["probe_requests"]} if pk_["kit"].get("probe_requests") else {})}
                       for pk_ in results if "kit" in pk_})
        (out / "grades.json").write_text(json.dumps(grades, indent=1, sort_keys=True), encoding="utf-8")
    # the index lists every packet in the folder: a run of one item keeps the others' rows
    comp = read_json(out / "completeness.json", {}) or {}
    comp.update({pk["item"]["id"]: {k: list(v) for k, v in pk["sections"].items()} for pk in results})
    (out / "completeness.json").write_text(json.dumps(comp, indent=1), encoding="utf-8")
    index = ["# Diagnosis packets", "",
             f"made {datetime.date.today().isoformat()} by tools/fidelity/packet.py; each fixer is given the path of its packet.", "",
             "| item | " + " | ".join(t for _, t in SECTIONS) + " |", "|---|" + "---|" * len(SECTIONS)]
    index += [f"| [{iid}]({iid}.md) | " + " | ".join(secs.get(k, ["-"])[0] for k, _ in SECTIONS) + " |"
              for iid, secs in sorted(comp.items()) if (out / f"{iid}.md").is_file()]
    index += ["", "Statuses: full (the section answers what it is for), partial (something is there, the reason says what is not), "
              "missing (nothing, the reason says what would produce it), n/a (does not apply to this item or step)."]
    (out / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return results


# ================================================================== skip the kits when their inputs have not changed

def shallow_sig(path: Path | None) -> str:
    """A folder as its entries' names, sizes and times (one level): it moves when a file is added, removed or written in it."""
    if path is None or not Path(path).is_dir():
        return sc.MISSING
    try:
        return sc.digest({"e": [[e.name, e.is_dir(), *(lambda st: (st.st_size, st.st_mtime_ns))(e.stat())] for e in sorted(os.scandir(path), key=lambda e: e.name)]})
    except OSError:
        return sc.MISSING


def kit_inputs(src: Sources, items: list, groups_doc: dict, a: argparse.Namespace) -> dict:
    """The named inputs of a run of `--groups` kits, each a short digest: the items; for each item the groups the kit would take
    (the first MAX_GROUPS matches, whole, and how many more there are: the groups no item matches do not matter); the engine
    and the Python tools the packet runs; the evidence files by content and the evidence folders by their entries."""
    names = Names(read_json(src.id_map, {}) or {})
    slices = []
    for it in items:
        matches = kt.matching_groups(it, set(entities_in(item_text(it), names)), groups_doc)
        slices.append([it["id"], [g for g, _ in matches[: kt.MAX_GROUPS]], len(matches)])
    ref = Path(src.referee)
    folders = [src.probes, *src.probes_more, *src.trace_dirs, *src.minimal_dirs, src.refset, src.codex_walk,
               Path(src.corpus) / "runs" if src.corpus else None]
    newest = sorted((d for d in Path(src.codex_walk).iterdir() if d.is_dir()), key=lambda d: d.name)[-1:] if src.codex_walk and Path(src.codex_walk).is_dir() else []
    return {"items": sc.digest({"items": items}), "groups": sc.digest({"slices": slices}),
            "engine": sc.engine_digest(ROOT), "tests": sc.tree_hash(src.tests, (".cs",)) if Path(src.tests).is_dir() else sc.MISSING,
            "tools": sc.tools_digest(ROOT),
            "files": sc.files_hash([p for p in (src.ledger, src.findings, src.attempts, src.id_map, src.sweep_map, src.walker,
                                                Path(src.corpus) / "index.json" if src.corpus else None) if p], None),
            "stats": sc.digest({"run": sc.stat_sig(src.run) if src.run else "", "emudump": sc.stat_sig(src.emudump) if src.emudump else "",
                                "codex": sc.stat_sig(src.codex_data) if src.codex_data else "", "decompiled": sc.decompiled_sig(src.decompiled) if src.decompiled else ""}),
            "folders": sc.digest({"d": [shallow_sig(f) for f in folders] + [shallow_sig(d) for d in newest]}),
            "flags": sc.digest({"kit_test": not a.no_kit_test, "emudump": not a.no_emudump, "sweep": not a.no_sweep, "probes": a.refresh_probes}),
            "refs": str(ref.name)}


def kit_outputs(items: list, out: Path) -> dict:
    """What a kits run leaves in the packets folder: its index files and one packet per item."""
    names = ["grades.json", "completeness.json", "index.md"] + [f"{it['id']}.md" for it in items]
    return {n: Path(out) / n for n in names}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("ids", nargs="*", help="backlog item ids (as --item)")
    ap.add_argument("--item", action="append", default=[], help="a backlog item id (repeatable)")
    ap.add_argument("--top", type=int, help="the first N open items of the backlog (in its order: by impact)")
    ap.add_argument("--fight", action="append", default=[], help="a corpus fight id or seed, or a reference-set id")
    ap.add_argument("--item-file", type=Path, help="a JSON file holding one item (id, title, evidence, ...; optional setup, seeds)")
    ap.add_argument("--backlog", type=Path, help="default: the newest backlog-c*.json in the referee backlog folder")
    ap.add_argument("--referee", type=Path, default=DEFAULT_REFEREE)
    ap.add_argument("--out", type=Path, help="default: <referee>/backlog/packets")
    ap.add_argument("--run", type=Path, help="a corpus run's results.json (default: the newest full run of this checkout)")
    ap.add_argument("--emudump", type=Path, help="EmuDump.dll (default: this checkout's build, $SPIRELINE_EMUDUMP)")
    ap.add_argument("--no-emudump", action="store_true", help="stored results only: no emulator values, no trace alignment")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--refresh-probes", action="store_true", help="judge the recorded probe games again with this EmuDump (no game)")
    ap.add_argument("--jobs", type=int, default=4, help="processes for the sweep tables")
    ap.add_argument("--codex-walk", type=Path, help="a folder of EmuDump codex outputs (default: <referee>/codex-walk, the newest folder in it)")
    ap.add_argument("--codex-data", type=Path, help="the run histories (default: <referee>/../data/spire-codex/runs-v0.107.1.jsonl.gz)")
    ap.add_argument("--groups", type=Path, help="the fix lab's groups.json: add a fix kit to each packet and write grades.json (tools/fidelity/kit.py)")
    ap.add_argument("--lab", type=Path, help="the lab the groups' refs are relative to (default: the folder of --groups)")
    ap.add_argument("--no-kit-test", action="store_true", help="with --groups: do not generate, compile or run the kits' failing tests")
    ap.add_argument("--force", action="store_true", help="with --groups: build the kits again even when no input changed since the last run (LAB/stamps/kits.json)")
    a = ap.parse_args(argv)
    a.item = list(a.ids) + [i for i in a.item if i not in a.ids]

    kw = {"run": a.run} if a.run else {}
    if a.codex_walk:
        kw["codex_walk"] = a.codex_walk
    if a.codex_data:
        kw["codex_data"] = a.codex_data
    src = Sources.default(a.referee, emudump=a.emudump, **kw)
    if a.no_emudump:
        src.emudump = None
    out = a.out or (a.referee / "backlog" / "packets")
    items, forced = [], {}
    if a.item or a.top:
        bl = a.backlog or latest_backlog(a.referee)
        if not bl:
            print("no backlog file", file=sys.stderr)
            return 2
        doc = load_backlog(bl)
        by_id = {i["id"]: i for i in doc["items"]}
        for iid in a.item:
            if iid not in by_id:
                print(f"{iid}: not in {bl.name}", file=sys.stderr)
                return 2
            items.append(by_id[iid])
        if a.top:
            items += [i for i in open_items(doc, a.top) if i not in items]
    if a.item_file:
        it = json.loads(a.item_file.read_text(encoding="utf-8"))
        items += it if isinstance(it, list) else [it]
    for ref in a.fight:
        fid = "fight-" + re.sub(r"[^A-Za-z0-9]+", "", ref)
        items.append({"id": fid, "lane": "fight", "kind": "failing fight", "status": "open", "title": f"failing fight {ref}",
                      "evidence": "", "impact_note": "", "subsystem": ""})
        forced[fid] = [ref]
    if not items:
        ap.error("give --item, --top, --fight or --item-file")
    kit_doc = kt.load_groups(a.groups) if a.groups else None
    stamp_lab, want, parts, outs = None, None, None, None
    if kit_doc is not None:  # the kits are a function of the files below: when none changed, the folder's last run stands
        stamp_lab = a.lab or a.groups.resolve().parent
        parts = kit_inputs(src, items, kit_doc, a)
        want, outs = sc.digest(parts), kit_outputs(items, out)
        why = None if a.force else sc.skip_reason(stamp_lab, "kits", want, outs)
        if why:
            print(f"step skipped: kits: {why}")
            return 0
    with tempfile.TemporaryDirectory(prefix="packet-") as tmp:
        cx = Context.build(src, Path(tmp), out / "sweeps", do_sweep=not a.no_sweep, refresh=a.refresh_probes)
        if a.groups:
            cx.kit_cfg = {"groups": kit_doc, "lab": a.lab or a.groups.resolve().parent, "verify": not a.no_kit_test}
        results = write_packets(items, cx, out, forced, a.jobs)
    if want is not None:
        sc.write_stamp(stamp_lab, "kits", want, outs, parts={k: str(v)[:16] for k, v in parts.items()})
    for pk_ in results:  # what the loop hands to each fixer: the path of its packet
        print(f"packet {pk_['item']['id']}: {(out / (pk_['item']['id'] + '.md')).resolve().as_posix()}"
              + (f" (kit grade {pk_['kit']['grade']})" if "kit" in pk_ else ""))
    print(f"packets: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
