"""The old-against-Core2 fuzzer (docs/rebuild/TIER1_KERNEL.md section 6, referee/design/rebuild-plan.md section 3).

Seeded setup fights built from models whose hook families are switched on, each played to the end by a random legal
player (EmuDump fuzz, tools/emudump/Fuzz.cs) on `--engine old` and on `--engine core2` with a switch set, and compared by
full-state hash after every step. Where the engines agree the player makes the same moves; the first step whose action
or hash differs is the first difference.

    python -m tools.fidelity.fuzz_core2 [--fights N] [--minutes M] [--seed S] [--switches F08,F10 | --switches none]
        [--emudump DLL] [--procs P] [--batch B] [--max-steps K] [--out DIR] [--manifest PATH] [--explained PATH]
        [--replay-check N] [--catalog] [--check-explained]

Switch set: tools/fidelity/migrated_families.json by default, `--switches F01,F02,...` for a test set (written next to a
scratch copy of the EmuDump build, which reads its switches from the file beside the DLL; the parent makes that copy once
and hands the workers its DLL path), `--switches none` for no switch, `--switches all` for F01-F16 (today's one switch group).
`--fights N` and `--minutes M` each end the run; given both, whichever comes first; `--minutes` alone runs until the time is
up, `--fights` alone (default 200) until N fights are done.
With no switch the models are every Ironclad, colourless and shared card, relic and potion the old engine builds and every
encounter of every act (the sanity run: core2 is the old engine, so it must find nothing). With a switch set a model is
drawn only when every combat-hook family it overrides (the port manifest's `families`) is in the set; an encounter only
when each of its monsters is.

A difference is a first differing step. It is re-run through `EmuDump fight --state-hash=fields` (the fields that differ
name it) and its minimal re-run spec, the fight cut at that step, is saved under <out>/<run>/ as a record `EmuDump fight`
replays on either engine. Its signature (`state:<fields>`, `action`, `length` or `error`) is looked up in
tools/fidelity/fuzz_explained.json: each entry names the hook_open.txt row or the migration doc item that explains it.
Exit 0 when every difference is explained (or none), 1 when any is not (or a run failed), 2 on a bad argument.

The spec of fight i of seed S is a function of (S, i) alone, so a run is repeatable whatever the batching.
"""
from __future__ import annotations

import argparse
import collections
import copy
import fnmatch
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pool workers import this module by path
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
MIGRATED = HERE / "migrated_families.json"
EXPLAINED = HERE / "fuzz_explained.json"
HOOK_OPEN = HERE / "hook_open.txt"
DEFAULT_DLL = ROOT / "tools" / "emudump" / "bin" / "Release" / "net9.0" / "EmuDump.dll"
REBUILD = Path(os.environ.get("SPIRELINE_REBUILD", r"D:\spireline\referee\rebuild"))
DEFAULT_OUT = REBUILD / "fuzz"
DEFAULT_MANIFEST = REBUILD / "manifest.json"
ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
ACTS = ("OVERGROWTH", "UNDERDOCKS", "HIVE", "GLORY")
ACT_INDEX = {"OVERGROWTH": 1, "UNDERDOCKS": 1, "HIVE": 2, "GLORY": 3}
KINDS = ("weak", "normal", "elite", "boss")
CARD_POOLS = ("IroncladCardPool", "ColorlessCardPool")
CARD_RARITIES = ("Basic", "Common", "Uncommon", "Rare")
RELIC_POOLS = ("IroncladRelicPool", "SharedRelicPool")
POTION_POOLS = ("IroncladPotionPool", "SharedPotionPool")
STARTER_CARDS = ("STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH")
EXTRA_CARDS = ("ASCENDERS_BANE",)  # no pool holds it; every A10 deck does
FORMAT = 1
RERUN_CHUNK = 50  # records per field re-run process
MAX_FAILED_BATCHES = 10  # a run with no fight limit stops starting batches after this many fail (no dotnet, a broken build)
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
SWITCHABLE = [f"F{n:02d}" for n in range(1, 17)]


# ================================================================================================ the switch set

def migrated(path: Path = MIGRATED) -> list[str]:
    return list(json.loads(Path(path).read_text(encoding="utf-8")).get("migrated", []))


def parse_switches(text: str | None, path: Path = MIGRATED) -> list[str]:
    """The families of a test set (`F08,F10`), `none` for the empty set, the migrated file's by default."""
    if text is None:
        return sorted(migrated(path))
    if text.strip().lower() in ("", "none"):
        return []
    if text.strip().lower() == "all":
        return list(SWITCHABLE)
    fams = sorted({f.strip().upper() for f in text.split(",") if f.strip()})
    bad = [f for f in fams if f not in SWITCHABLE]
    if bad:
        raise ValueError(f"unknown family {bad[0]} (F01-F16)")
    return fams


# ================================================================================================ the catalog

@dataclass
class Catalog:
    """What fights are drawn from. Each entry carries the combat families its model needs (the manifest's), so a switch
    set filters it. `unrestricted`: no switch is on, every entry is drawn (the sanity run)."""
    cards: list = field(default_factory=list)       # {"id", "upgradable", "families"}
    relics: list = field(default_factory=list)      # {"id", "families"}
    potions: list = field(default_factory=list)     # {"id", "families", "target"}
    encounters: list = field(default_factory=list)  # {"id", "act", "kind", "index", "monsters", "families"}
    switches: tuple = ()

    @property
    def unrestricted(self) -> bool:
        return not self.switches

    def allowed(self, entry: dict) -> bool:
        return self.unrestricted or set(entry.get("families", ())) <= set(self.switches)

    def pick(self, kind: str) -> list:
        return [e for e in getattr(self, kind) if self.allowed(e)]

    def coverage(self) -> dict:
        """Per kind: how many models a fight can place under the switch set, and how many the catalog holds."""
        return {k: {"placeable": len(self.pick(k)), "known": len(getattr(self, k))}
                for k in ("cards", "relics", "potions", "encounters")}


def model_families(manifest: dict | None, key: str) -> list[str]:
    """Combat-hook families of `kind:ID` in the port manifest (a model it does not list needs none)."""
    return sorted(((manifest or {}).get("models", {}).get(key) or {}).get("families", ()))


def build_catalog(tables: dict, info: dict, manifest: dict | None, switches, rosters: dict | None = None) -> Catalog:
    """The catalog from `EmuDump tables` and `EmuDump setupinfo` (the engine's own tables, no game files) and the manifest.
    `rosters`: encounter name -> monster ids (an encounter without one needs no monster family)."""
    switches = tuple(sorted(switches))
    fam = lambda kind, ident: model_families(manifest, f"{kind}:{ident}")  # noqa: E731
    cat = Catalog(switches=switches)
    cards = tables["cards"]
    ids = [c for pool in CARD_POOLS for c in tables["card_pools"][pool] if cards[c]["rarity"] in CARD_RARITIES
           and not cards[c].get("multiplayer_only")]
    for c in sorted(set(ids) | {x for x in EXTRA_CARDS if x in cards}):
        cat.cards.append({"id": c, "upgradable": bool(cards[c].get("upgradable")), "families": fam("card", c)})
    follow = info.get("relics", {})
    for r in sorted({r for pool in RELIC_POOLS for r in tables["relic_pools"][pool]}):
        if follow.get(r, "None") == "None" and r in tables["relics"]:  # a pickup that opens a screen needs the game's answers
            cat.relics.append({"id": r, "families": fam("relic", r)})
    modelled = tables["potions"]
    for p in sorted({p for pool in POTION_POOLS for p in tables["potion_pools"][pool]}):
        if p in modelled:
            cat.potions.append({"id": p, "families": fam("potion", p), "target": modelled[p].get("target")})
    for act in ACTS:
        for kind in KINDS:
            for i, name in enumerate(tables["acts"][act].get(kind, [])):
                monsters = (rosters or {}).get(name, [])
                fams = sorted({f for m in monsters for f in fam("monster", m)})
                cat.encounters.append({"id": name, "act": act, "kind": kind, "index": i, "monsters": monsters, "families": fams})
    return cat


# ================================================================================================ spec generation

def spec_rng(seed: int, index: int) -> random.Random:
    return random.Random(f"fuzz-core2:{seed}:{index}")  # a str seed: hashed with SHA-512, the same on every run


def generate(cat: Catalog, seed: int, index: int, max_steps: int = 80) -> dict:
    """Fight `index` of run `seed`: a record in the referee's shape without its steps (EmuDump fuzz plays it)."""
    rng = spec_rng(seed, index)
    encounters, cards = cat.pick("encounters"), cat.pick("cards")
    relics, potions = cat.pick("relics"), cat.pick("potions")
    if not encounters:
        raise ValueError("no encounter has all its monsters' families switched on")
    enc = rng.choice(encounters)
    by_id = {c["id"]: c for c in cards}
    held = rng.sample(relics, min(len(relics), rng.randint(0, 6)))
    held_ids = [r["id"] for r in held]
    plain = [c for c in cards if c["id"] not in STARTER_CARDS and c["id"] not in EXTRA_CARDS]
    deck = [{"id": c, "upgraded": False} for c in STARTER_CARDS if c in by_id for _ in range(rng.randint(0, 5 if c != "BASH" else 1))]
    deck += [{"id": "ASCENDERS_BANE", "upgraded": False}] if "ASCENDERS_BANE" in by_id and rng.random() < 0.5 else []
    for _ in range(rng.randint(3, 14) if plain else 0):
        c = rng.choice(plain)
        deck.append({"id": c["id"], "upgraded": bool(c["upgradable"]) and rng.random() < 0.3})
    if not deck:
        deck = [{"id": c["id"], "upgraded": False} for c in rng.sample(cards, min(5, len(cards)))]
    rng.shuffle(deck)
    slots = 2 + (2 if "POTION_BELT" in held_ids else 0)
    belt = [p["id"] for p in rng.sample(potions, min(len(potions), rng.randint(0, slots)))]
    hp_max = rng.randint(60, 120)
    items = []
    if ACT_INDEX[enc["act"]] > 1:
        items.append({"cmd": "act", "index": ACT_INDEX[enc["act"]]})
    pair = [{"cmd": "relics_set", "ids": held_ids}, {"cmd": "deck_set", "cards": deck}]
    if rng.random() < 0.5:
        pair.reverse()  # relics that act on the deck meet its cards on either side of the set
    items += pair + [{"cmd": "potions_set", "ids": belt}]
    items.append({"cmd": "hp", "max": hp_max, "current": hp_max if rng.random() < 0.7 else rng.randint(max(1, hp_max // 3), hp_max)})
    items.append({"cmd": "fight", "id": enc["id"]})
    return {
        "seed": "".join(rng.choice(ALPHABET) for _ in range(10)), "character": "IRONCLAD", "ascension": 10, "neow_pick": 0,
        "use_potions": True, "setup": items,
        "fight": {"id": enc["id"], "act": enc["act"], "kind": enc["kind"], "index": enc["index"]},
        "tag": f"fuzz:{seed}:{index}", "fuzz": {"seed": rng.getrandbits(48), "max_steps": max_steps},
    }


# ================================================================================================ running EmuDump

def emudump(dll: Path, args: list[str], timeout: float = 1800) -> str:
    """One EmuDump process (one emulator per process) at BELOW_NORMAL priority; its stdout."""
    p = subprocess.run(["dotnet", str(dll), *args], capture_output=True, timeout=timeout, creationflags=BELOW_NORMAL,
                       env={**os.environ, "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_NOLOGO": "1"})
    if p.returncode != 0:
        raise RuntimeError(f"EmuDump {args[0]} exited {p.returncode}: " + p.stderr.decode("utf-8", "replace")[:400])
    return p.stdout.decode("utf-8")


def dll_for(dll: Path, switches: list[str] | None, work: Path) -> Path:
    """The DLL to run for a switch set: the build itself for the migrated file's set, else a scratch copy of the build whose
    migrated_families.json (read from beside the DLL) holds the set. The parent calls this once, before any worker starts, and
    hands the result to the workers (Options.dll): a worker never copies or rewrites a build. The copy is this process's own
    (its pid is in the name), so two runs over one --out do not meet in it; `drop_build` removes it."""
    if switches is None or sorted(switches) == sorted(migrated(Path(dll).parent / "migrated_families.json")):
        return Path(dll)
    dest = work / f"bin-{'none' if not switches else '-'.join(switches)}-{os.getpid()}"
    if not (dest / "EmuDump.dll").is_file():
        shutil.copytree(Path(dll).parent, dest, dirs_exist_ok=True)
    (dest / "migrated_families.json").write_text(json.dumps({"migrated": sorted(switches)}), encoding="utf-8")
    return dest / "EmuDump.dll"


def drop_build(built: Path, original: Path) -> None:
    """Remove the scratch build `dll_for` made (never the build the user named)."""
    if Path(built).parent.resolve() != Path(original).parent.resolve() and Path(built).parent.name.startswith("bin-"):
        shutil.rmtree(Path(built).parent, ignore_errors=True)


def run_specs(dll: Path, engine: str, specs: list[dict], work: Path, fields: bool = False) -> list[dict]:
    """EmuDump fuzz on the specs with this engine: one result per spec."""
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=work, delete=False, encoding="utf-8") as f:
        json.dump({"results": specs}, f)
    try:
        out = emudump(dll, ["fuzz", f.name, "--engine", engine] + (["--fields"] if fields else []))
    finally:
        os.unlink(f.name)
    res = json.loads(out)["results"]
    if len(res) != len(specs):
        raise RuntimeError(f"EmuDump fuzz answered {len(res)} fights for {len(specs)} specs")
    return res


def run_records(dll: Path, engine: str, records: list[dict], work: Path, fields: bool = True) -> list[dict]:
    """EmuDump fight (--state-hash) on referee-shaped records: the replay path."""
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=work, delete=False, encoding="utf-8") as f:
        json.dump({"results": records}, f)
    try:
        out = emudump(dll, ["fight", f.name, "--compact", "--engine", engine, "--state-hash=fields" if fields else "--state-hash"])
    finally:
        os.unlink(f.name)
    return json.loads(out)["results"]


def load_catalog(dll: Path, switches: list[str], manifest_path: Path | None, work: Path) -> Catalog:
    tables = json.loads(emudump(dll, ["tables"]))
    info = json.loads(emudump(dll, ["setupinfo"]))
    manifest = None
    if manifest_path and Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    elif switches:
        raise FileNotFoundError("a switch set needs the port manifest (tools/fidelity/manifest.py build): " + str(manifest_path))
    probe = build_catalog(tables, info, manifest, switches)
    specs = [{"seed": "ROSTERPROB", "ascension": 10, "neow_pick": 0, "setup": ([{"cmd": "act", "index": ACT_INDEX[e["act"]]}]
              if ACT_INDEX[e["act"]] > 1 else []) + [{"cmd": "fight", "id": e["id"]}],
              "fight": {"id": e["id"], "act": e["act"], "kind": e["kind"], "index": e["index"]}, "fuzz": {"max_steps": 0, "roster": True}}
             for e in probe.encounters]
    rosters = {}
    for e, r in zip(probe.encounters, run_specs(dll, "old", specs, work)):
        rosters[e["id"]] = r.get("roster", [])
    return build_catalog(tables, info, manifest, switches, rosters)


# ================================================================================================ comparing

def error_head(result: dict) -> str | None:
    err = result.get("error")
    return str(err).splitlines()[0] if err else None


def first_difference(a: dict, b: dict) -> dict | None:
    """The first step where two fuzz results (old, core2) differ, or None: {"step", "kind", ...}. `kind`: "action" (the
    player's move differs: the states before agreed, so the legal moves differ), "state" (the hash after the step),
    "error" (one engine threw, or the two threw differently) or "length" (one fight ended first)."""
    sa, sb = a.get("steps") or [], b.get("steps") or []
    for i in range(min(len(sa), len(sb))):
        if sa[i].get("action") != sb[i].get("action"):
            return {"step": i, "kind": "action", "old": sa[i].get("action"), "core2": sb[i].get("action")}
        if sa[i].get("state_hash") != sb[i].get("state_hash"):
            return {"step": i, "kind": "state"}
    ea, eb = error_head(a), error_head(b)
    if ea != eb:
        return {"step": min(len(sa), len(sb)), "kind": "error", "old": ea, "core2": eb}
    if len(sa) != len(sb):
        return {"step": min(len(sa), len(sb)), "kind": "length", "old": len(sa), "core2": len(sb)}
    return None


def differing_fields(a: dict, b: dict) -> list[str]:
    fa, fb = a.get("state_fields") or {}, b.get("state_fields") or {}
    return sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))


def signature(kind: str, fields: list[str]) -> str:
    """What a triage entry matches: `state:<up to six differing fields, sorted>` for a hash difference, else the kind."""
    return f"state:{','.join(fields[:6])}" if kind == "state" and fields else kind


def replay_record(spec: dict, result: dict, step: int) -> dict:
    """A referee-shaped record of the fight cut at `step` (what `EmuDump fight` replays): the spec, the room the engine
    took, the actions through that step and the screen answers up to it."""
    rec = {k: copy.deepcopy(v) for k, v in spec.items() if k != "fuzz"}
    if result.get("room") is not None:
        rec["room"] = result["room"]
    steps = (result.get("steps") or [])[: step + 1]
    rec["steps"] = [{"action": None}] + [{"action": s["action"]} for s in steps[1:]]
    rec["choices"] = [c for c in result.get("choices") or [] if c["step"] <= step]
    if any(c["step"] == 0 for c in rec["choices"]):
        rec["room_choices_from"] = 0
    return rec


# ================================================================================================ triage

def load_explained(path: Path = EXPLAINED) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    return list(json.loads(p.read_text(encoding="utf-8")).get("explained", []))


def hook_open_rows(path: Path = HOOK_OPEN) -> set[str]:
    """The rows of hook_open.txt (the text before its `#` note)."""
    rows = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            rows.add(line)
    return rows


def validate_explained(entries: list[dict], rows: set[str] | None) -> list[str]:
    """Problems with a triage file: an entry needs a `signature` (fnmatch pattern over `state:<fields>`, `action`,
    `length` or `error`) and exactly one of `row` (a row of hook_open.txt, which must still be there) or `doc` (a migration
    doc item: `<path>: <item>`), and a `note`."""
    bad = []
    for i, e in enumerate(entries):
        where = f"entry {i} ({e.get('signature')!r})"
        if not e.get("signature"):
            bad.append(f"{where}: no signature")
        if bool(e.get("row")) == bool(e.get("doc")):
            bad.append(f"{where}: needs exactly one of row and doc")
        if e.get("row") and rows is not None and e["row"].strip() not in rows:
            bad.append(f"{where}: row {e['row']!r} is not in hook_open.txt")
        if not e.get("note"):
            bad.append(f"{where}: no note")
    return bad


def explanation(sig: str, entries: list[dict]) -> dict | None:
    return next((e for e in entries if fnmatch.fnmatchcase(sig, e["signature"])), None)


# ================================================================================================ the run

@dataclass
class Options:
    dll: str  # the build to run as it is (dll_for's answer: workers never resolve it again)
    seed: int = 1
    fights: int | None = 200  # None: no limit, the run ends at `minutes`
    minutes: float | None = None
    switches: list = field(default_factory=list)
    procs: int = 3
    batch: int = 50
    max_steps: int = 80
    out: str = str(DEFAULT_OUT)
    manifest: str | None = str(DEFAULT_MANIFEST)
    explained: str = str(EXPLAINED)
    replay_check: int = 0


def process_batch(opts: Options, cat: Catalog, indices: list[int], work: Path, runner=None, rerunner=None) -> dict:
    """Generate, run on both engines, compare; the differences with their re-run specs and fields. `runner(engine, specs)` and
    `rerunner(engine, records)` default to EmuDump on the build `opts.dll` names (tests pass fakes). The build is the parent's:
    nothing here copies or rewrites one."""
    t0 = time.time()
    if runner is None or rerunner is None:
        dll = Path(opts.dll)
        runner = runner or (lambda engine, specs: run_specs(dll, engine, specs, work))
        rerunner = rerunner or (lambda engine, recs: run_records(dll, engine, recs, work))
    specs = [generate(cat, opts.seed, i, opts.max_steps) for i in indices]
    old, new = runner("old", specs), runner("core2", specs)
    stats = {"fights": len(specs), "steps": sum(len(r.get("steps") or []) for r in old), "errors": sum(1 for r in old if r.get("error")),
             "seconds": 0.0}
    diffs = []
    for i, spec, a, b in zip(indices, specs, old, new):
        d = first_difference(a, b)
        if d is None:
            continue
        d.update({"index": i, "tag": spec["tag"], "seed": spec["seed"], "fields": [],
                  "record": replay_record(spec, a, d["step"])})  # old's moves through the step: core2 refuses or answers it differently
        diffs.append(d)
    states = [d for d in diffs if d["kind"] == "state"]
    for lo in range(0, len(states), RERUN_CHUNK):  # every hash difference gets its fields, a few EmuDump processes for all
        chunk = states[lo: lo + RERUN_CHUNK]
        ra, rb = (rerunner(e, [d["record"] for d in chunk]) for e in ("old", "core2"))
        if len(ra) != len(chunk) or len(rb) != len(chunk):
            raise RuntimeError(f"the field re-run answered {len(ra)} and {len(rb)} fights for {len(chunk)} records")
        for d, a, b in zip(chunk, ra, rb):
            sa, sb = a.get("steps") or [], b.get("steps") or []
            if len(sa) > d["step"] and len(sb) > d["step"]:
                d["fields"] = differing_fields(sa[d["step"]], sb[d["step"]])
    for d in diffs:
        d["signature"] = signature(d["kind"], d["fields"])
    stats["seconds"] = round(time.time() - t0, 2)
    return {"stats": stats, "diffs": diffs}


def _pool_task(args):
    opts, cat, indices, work = args
    return process_batch(opts, cat, indices, Path(work))


def fights_limit(fights: int | None, minutes: float | None) -> int | None:
    """How many fights a run plays: `--fights` as given; with neither flag 200; with `--minutes` alone no limit (it ends on time)."""
    if fights is not None:
        return fights
    return None if minutes else 200


def batches_of(opts: Options):
    """The batches of fight indices, lazily: to `opts.fights`, or without end when it is None (`minutes` stops the run)."""
    start = 0
    while opts.fights is None or start < opts.fights:
        stop = start + opts.batch if opts.fights is None else min(start + opts.batch, opts.fights)
        yield list(range(start, stop))
        start = stop





def run_fuzz(opts: Options, cat: Catalog, runner=None, rerunner=None, now=time.time) -> dict:
    """The run: batches over a spawn pool (inline with a runner), the summary. Never raises on a difference. `opts.dll` is the
    build to run as it is (the parent resolved it once, `dll_for`)."""
    out = Path(opts.out) / f"run-{opts.seed}-{'-'.join(opts.switches) or 'none'}"
    work = out / "work"
    out.mkdir(parents=True, exist_ok=True)
    batches = batches_of(opts)
    deadline = now() + opts.minutes * 60 if opts.minutes else None
    t0, results, errors = now(), [], []
    if runner is not None or opts.procs <= 1:
        for b in batches:
            if (deadline and now() > deadline) or len(errors) >= MAX_FAILED_BATCHES:
                break
            try:
                results.append(process_batch(opts, cat, b, work, runner, rerunner))
            except Exception as ex:  # noqa: BLE001 - a failed batch is reported, the run goes on
                errors.append(repr(ex)[:300])
    else:
        with ProcessPoolExecutor(max_workers=opts.procs, mp_context=get_context("spawn")) as pool:
            pending = collections.deque()
            it = batches
            for _ in range(opts.procs):
                if (b := next(it, None)) is not None:
                    pending.append(pool.submit(_pool_task, (opts, cat, b, str(work))))
            while pending:
                fut = pending.popleft()
                try:
                    results.append(fut.result())
                except Exception as ex:  # noqa: BLE001 - a failed batch is reported, the run goes on
                    errors.append(repr(ex)[:300])
                if not (deadline and now() > deadline) and len(errors) < MAX_FAILED_BATCHES and (b := next(it, None)) is not None:
                    pending.append(pool.submit(_pool_task, (opts, cat, b, str(work))))
    entries = load_explained(Path(opts.explained))
    diffs = [d for r in results for d in r["diffs"]]
    for d in diffs:
        d["explained"] = (explanation(d["signature"], entries) or {}).get("row") or (explanation(d["signature"], entries) or {}).get("doc")
    saved = []
    for d in diffs:
        p = out / f"diff-{d['index']}.json"
        meta = {k: d[k] for k in ("index", "tag", "step", "kind", "signature", "fields", "explained") if k in d}
        p.write_text(json.dumps({"results": [dict(d["record"], fuzz_diff=meta)]}), encoding="utf-8")
        saved.append(p.name)
    seconds = max(now() - t0, 1e-9)
    fields = collections.Counter(f for d in diffs for f in d["fields"])
    summary = {
        "format": FORMAT, "seed": opts.seed, "switches": opts.switches, "fights": sum(r["stats"]["fights"] for r in results),
        "steps": sum(r["stats"]["steps"] for r in results), "error_fights": sum(r["stats"]["errors"] for r in results),
        "differing_fights": len(diffs), "explained": sum(1 for d in diffs if d["explained"]),
        "untriaged": sum(1 for d in diffs if not d["explained"]), "failed_batches": errors,
        "top_fields": fields.most_common(10), "signatures": collections.Counter(d["signature"] for d in diffs).most_common(10),
        "seconds": round(seconds, 1), "fights_per_second": round(sum(r["stats"]["fights"] for r in results) / seconds, 2),
        "procs": 1 if runner is not None else opts.procs, "coverage": cat.coverage(),
        "first": [{k: d[k] for k in ("index", "step", "kind", "signature", "fields", "explained")} for d in diffs[:10]],
        "specs": saved[:50],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def exit_code(summary: dict) -> int:
    return 1 if summary["untriaged"] or summary["failed_batches"] else 0


def replay_check(opts: Options, cat: Catalog, n: int, work: Path) -> dict:
    """The round trip the minimal specs rely on: the first `n` fights of the run, replayed from their records by
    `EmuDump fight` on the old engine, hash for hash the same as EmuDump fuzz played them."""
    dll = Path(opts.dll)
    specs = [generate(cat, opts.seed, i, opts.max_steps) for i in range(n)]
    played = run_specs(dll, "old", specs, work)
    recs = [replay_record(s, r, len(r.get("steps") or []) - 1) for s, r in zip(specs, played)]
    replayed = run_records(dll, "old", recs, work, fields=False)
    bad = [i for i, (p, r) in enumerate(zip(played, replayed))
           if [s.get("state_hash") for s in p.get("steps") or []] != [s.get("state_hash") for s in r.get("steps") or []]]
    return {"fights": n, "mismatched": bad}


def preflight(opts: Options, cat: Catalog, work: Path) -> str | None:
    """Why core2 cannot run the switch set at all (a partial switch group is refused when the engine is made), or None: one
    fight with no steps on each engine; an error on core2 alone is the set's, not a difference to triage."""
    dll = Path(opts.dll)
    spec = generate(cat, opts.seed, 0, 0)
    old, new = (run_specs(dll, e, [spec], work)[0] for e in ("old", "core2"))
    return error_head(new) if error_head(new) and not error_head(old) else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fights", type=int, help="fights to play (default 200, or no limit with --minutes alone)")
    ap.add_argument("--minutes", type=float, help="stop starting batches after M minutes (alone: run until then)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--switches", help="F01,F02,... for a test set, `all`, or `none` for no switch; default: migrated_families.json")
    ap.add_argument("--emudump", type=Path, default=DEFAULT_DLL)
    ap.add_argument("--procs", type=int, default=3, help="worker processes, each running one EmuDump at a time (max 3)")
    ap.add_argument("--batch", type=int, default=50, help="fights per EmuDump process")
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--explained", type=Path, default=EXPLAINED)
    ap.add_argument("--replay-check", type=int, default=0, metavar="N")
    ap.add_argument("--catalog", action="store_true", help="print what a fight can place under the switch set, and stop")
    ap.add_argument("--check-explained", action="store_true", help="validate the triage file against hook_open.txt, and stop")
    a = ap.parse_args(argv)
    if a.check_explained:
        bad = validate_explained(load_explained(a.explained), hook_open_rows())
        print("\n".join(bad) or "triage file ok")
        return 1 if bad else 0
    if a.procs < 1 or a.procs > 3:
        print("--procs 1..3 (the run keeps to three EmuDump processes at a time)", file=sys.stderr)
        return 2
    try:
        switches = parse_switches(a.switches)
    except ValueError as ex:
        print(ex, file=sys.stderr)
        return 2
    work = Path(a.out) / "work"
    built = dll_for(a.emudump, switches, work)  # once, here: the workers get this path and never copy a build
    try:
        opts = Options(dll=str(built), seed=a.seed, fights=fights_limit(a.fights, a.minutes), minutes=a.minutes, switches=switches,
                       procs=a.procs, batch=a.batch, max_steps=a.max_steps, out=str(a.out), manifest=str(a.manifest),
                       explained=str(a.explained), replay_check=a.replay_check)
        cat = load_catalog(built, switches, a.manifest, work)
        if a.catalog:
            print(json.dumps({"switches": switches, "coverage": cat.coverage()}, indent=1))
            return 0
        if why := preflight(opts, cat, work):
            print(f"core2 cannot run the switch set {switches}: {why}", file=sys.stderr)
            return 2
        if a.replay_check:
            rc = replay_check(opts, cat, a.replay_check, work)
            print("replay check:", json.dumps(rc))
            if rc["mismatched"]:
                return 1
        summary = run_fuzz(opts, cat)
    finally:
        drop_build(built, a.emudump)
    print(json.dumps({k: v for k, v in summary.items() if k != "coverage"}, indent=1))
    print("coverage:", json.dumps(summary["coverage"]))
    return exit_code(summary)


if __name__ == "__main__":
    sys.exit(main())
