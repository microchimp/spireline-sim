#!/usr/bin/env python3
"""Fix lab groups: every symptom the lab has seen, merged into root mechanisms and matched to the backlog.

    python tools/fidelity/lab_groups.py --lab LAB [--backlog <newest backlog-*.json>] --out LAB/groups.json
        --summary LAB/summary.md [--batch <jobs dir>/<name>]... [--corpus-results FILE] [--codex-walk DIR]
        [--refset-score FILE] [--invariants FILE] [--mutation FILE] [--decompiled DIR] [--host-gaps FILE]
        [--force] [--no-cache] [--cache-dir DIR]

Reads every finished job under LAB/runs (a folder with done.rc): aimed and setup batches (<name>.verdicts.json
with their specs, <name>.shrink.json or the summary's shrink lines, <name>.packs/minimal and traces), and the
presort outputs it finds there by shape (a corpus check's results.json, codex walk *.jsonl, a reference-set
score, an invariants report, a mutation report) or that its job.json names under "outputs" (for those kept
elsewhere, such as the corpus check's own run folder). --batch and the other flags add sources from outside.
No game, no emulator, no network: only files already written.

A finished job's files are never written again, so each is read once in its life: the sources a job holds and the symptoms the
readers made of each source are kept under LAB/cache/groups (tools/lab/groupcache.py), keyed by the job's directory listing (names,
sizes, times), the code the readers run and the maps they use. A changed file or changed code reads again; nothing newer than two
seconds is trusted; a damaged cache file is deleted and rebuilt. --no-cache neither reads nor writes it, --force clears it.

One symptom = one fight (or run) with its first wrong step. Its signature: the phase of that step (start,
play, end_turn, potion), the field group that differs, the first differing event from its trace (what one
side did that the other did not inside the step, numbers left out) and its implicated ingredients: the setup
items its shrunk minimal setup kept (each one needed: without it the divergence vanishes), or, unshrunk, the
card or potion it acted with. Groups are made of equal signatures and then merged into root mechanisms:
two groups with the same phase, field and event merge when their minimal setups differ only in ingredients
that share a primitive or a hook (the game's own model classes in the decompile: the hooks each overrides,
tools/fidelity/hook_table.py, and the game commands it calls), and an unshrunk symptom joins the shrunk group
whose ingredients hold its card (or its encounter, for a start or end of turn) when that is not a starter card
and the fight is known to hold every other ingredient the group needs (join_target). A feature shared by more than a
tenth of a kind's models (DamageCmd.Attack) does not count, unless the step's trace names that hook; a command
data/sweep_map.json names (a swept mechanism) counts up to three tenths. A fight-alone minimal setup has its
encounter as the ingredient (an encounter's features are its monsters'). Codex-walk stops group by component,
field and class; invariant violations by side and check; mutation survivors by file.

Each group gets a stable id (a hash of its key), its examples (smallest minimal setup first, at most 5, with
the game's and the emulator's value at the first wrong step), the ingredient effect, host_suspect (it matches
a known gap of the headless host, referee/host_gaps.json, or a generic rule: setups no real run has, a game
side that stands still, a trace that does not reproduce), the backlog items it matches (by entities, seeds
the evidence cites for the fight the seed names, see cited_seeds, and the field) with the relation: confirms
(one item, one group), splits (the item matches several groups), merges (the group matches several items) or
new, and impact_runs_est (real Ironclad A10 runs that meet its ingredients together, from the lab's Codex
profile; codex-walk groups count their runs).
summary.md: totals and the top 20 groups by impact, one line each. Paths are relative to LAB.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import datetime
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "referee"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tools.lab import stepcache as sc  # noqa: E402
from tools.lab.groupcache import GroupCache, Listing  # noqa: E402
from tools.fidelity.lab_targets import (  # noqa: E402
    BASIC_CARDS,
    REFEREE,
    Profile,
    newest_backlog,
    newest_refset_score,
    profile_path,
    read_json,
    snake,
    stop_tokens,
    utc_now,
    write_json,
)

DECOMPILED = REFEREE / "decompiled"
HOST_GAPS = ROOT / "referee" / "host_gaps.json"
SEED = re.compile(r"\b[0-9A-HJ-NP-Z]{10}\b")
FIELD = re.compile(r"(?<![\w])\.(hand|draw|discard|exhaust|enemies|potions|relic_flashes|player(?:\.\w+)?|rng(?:\.\w+)?)\b")
FEATURE_SHARE = 0.10
MAX_EXAMPLES = 5
GROUP_KINDS = ("card", "relic", "potion", "power", "monster")
NUM = re.compile(r"[-+]?\d+")
MIN_POOL = 300  # a rate from a small batch counts as if over this many fights
JOB_REF = re.compile(r"^runs/([^/#]+)/")  # a symptom's ref names the lab job it came from


def git_sha(tree: Path = ROOT) -> str:
    import subprocess  # noqa: PLC0415
    return subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()


def rel(path: Path | str, lab: Path) -> str:
    p = Path(path).resolve()
    for b, tag in ((lab, ""), (REFEREE, "referee/"), (ROOT, "fork/")):
        with contextlib.suppress(ValueError, OSError):
            return tag + p.relative_to(Path(b).resolve()).as_posix()
    return p.name


def clip(x, n: int = 160) -> str | None:
    if x is None:
        return None
    s = " ".join(str(x).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def stamp_utc(path: Path) -> str:
    """A job folder's UTC time from its name (yyyymmddThhmm-job), else the file's mtime."""
    m = re.match(r"(\d{8}T\d{4})", path.name)
    if m:
        return datetime.datetime.strptime(m.group(1), "%Y%m%dT%H%M").strftime("%Y-%m-%dT%H:%M:00Z")
    with contextlib.suppress(OSError):
        return datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return utc_now()


def phase_of(card) -> str:
    c = str(card or "")
    if c in ("start", ""):
        return "start"
    if c.upper() == "END_TURN":
        return "end_turn"
    if c.upper().startswith("POTION"):
        return "potion"
    return "play"


def item_key(it: dict) -> str | None:
    """A setup item as an ingredient key (kind:ID), None for the fight itself."""
    cmd, ident = it.get("cmd"), it.get("id")
    if cmd == "fight":
        return None
    if cmd == "card":
        return f"card:{ident}" + ("+" if it.get("upgraded") else "")
    if cmd in ("relic", "potion"):
        return f"{cmd}:{ident}"
    if cmd == "power":
        return f"power:{ident}@{'player' if it.get('target') == 'player' else 'enemy'}"
    if cmd == "act":
        return f"act:{it.get('index')}"
    if cmd == "deck_set":
        return "deck:set"
    if "deck_add" in it:
        return f"card:{it['deck_add'] if isinstance(it['deck_add'], str) else it['deck_add'].get('id', '?')}"
    return f"{cmd}:{ident}" if cmd else None


def item_keys(it: dict) -> list[str]:
    """A setup item's ingredient keys: player_load (a run's holdings, referee/run_repro.py) is one key per card,
    relic and potion it loads; any other item is item_key's one key (none for the fight)."""
    if it.get("cmd") == "player_load":
        bare = lambda x: str(x or "?").split(".", 1)[-1]  # noqa: E731
        return ([f"card:{bare(c.get('id'))}" + ("+" if c.get("current_upgrade_level") else "") for c in it.get("deck") or []]
                + [f"relic:{bare(r.get('id'))}" for r in it.get("relics") or []] + [f"potion:{bare(x)}" for x in it.get("potions") or []])
    k = item_key(it)
    return [k] if k else []


def entity_of(key: str) -> str:
    """kind:ID without the upgrade mark or the power's side: the model the ingredient is."""
    k = key.split("@")[0]
    return k[:-1] if k.endswith("+") else k


# ================================================================== the game's models: hooks and primitives

CMD_CALL = re.compile(r"\b([A-Z]\w*Cmd)\.([A-Z]\w*)\b")


def model_features(decompiled: Path, cache: Path | None) -> dict:
    """{"kind:ID": [features]}: hook:<Hook> for each hook the model overrides (inherited ones included) and
    cmd:<Class.Method> for each game command its class calls. Names only, from the local decompile."""
    root = Path(decompiled)
    if not root.is_dir():
        return {}
    files = sorted(root.glob("MegaCrit.Sts2.Core.Models*/*.cs"))
    stamp = [len(files), max((f.stat().st_mtime for f in files), default=0)]
    if cache:
        got = read_json(cache, {}) or {}
        if got.get("stamp") == stamp:
            return got.get("features", {})
    from tools.fidelity import hook_table as ht  # noqa: PLC0415
    classes = ht.model_classes(str(root))
    feats = collections.defaultdict(set)
    kind_name = {"monster": "enemy"}
    for o in ht.overrides(str(root)):
        if o.kind in GROUP_KINDS:
            feats[f"{kind_name.get(o.kind, o.kind)}:{snake(o.model.split('.')[-1])}"].add(f"hook:{o.hook}")
    for c in classes.values():
        kind = ht.kind_of(classes, c.base) if not c.abstract else None
        if kind not in GROUP_KINDS:
            continue
        with contextlib.suppress(OSError):
            text = (root / c.rel).read_text(encoding="utf-8", errors="replace")
            ent = f"{kind_name.get(kind, kind)}:{snake(c.name)}"
            feats[ent] |= {f"cmd:{a}.{b}" for a, b in CMD_CALL.findall(text)}
    out = {k: sorted(v) for k, v in feats.items()}
    if cache:
        write_json(cache, {"stamp": stamp, "features": out}, indent=None)
    return out


class Features:
    """Which features two ingredients share that say "same mechanism"."""

    def __init__(self, feats: dict, sweep_map: dict | None = None, encounter_monsters: dict | None = None):
        self.feats = {k: set(v) for k, v in feats.items()}
        self.encounter_monsters = encounter_monsters or {}
        self.curated = {f"cmd:{m}" for m in (sweep_map or {}) if not m.startswith("_") and "*" not in m}
        self.curated_prefix = [f"cmd:{m[:-1]}" for m in (sweep_map or {}) if m.endswith("*")]
        by_kind = collections.defaultdict(collections.Counter)
        n_kind = collections.Counter()
        for k, fs in self.feats.items():
            kind = k.split(":", 1)[0]
            n_kind[kind] += 1
            by_kind[kind].update(fs)
        self.common = {kind: {f for f, n in c.items() if n > max(3, FEATURE_SHARE * n_kind[kind])} for kind, c in by_kind.items()}
        # a command data/sweep_map.json names (a swept mechanism) counts up to three times as widely used
        self.ubiquitous = {kind: {f for f, n in c.items() if n > max(3, 3 * FEATURE_SHARE * n_kind[kind])} for kind, c in by_kind.items()}

    def of(self, key: str) -> set:
        e = entity_of(key)
        if e.startswith("encounter:"):  # its monsters' features (the profile names who is in it)
            mons = self.encounter_monsters.get(e.split(":", 1)[1], [])
            return set().union(*(self.feats.get(f"enemy:{m}", set()) for m in mons)) if mons else set()
        if e.startswith("power:"):
            base = e.split(":", 1)[1]
            return self.feats.get(e, set()) | self.feats.get(f"power:{base.removesuffix('_POWER')}_POWER", set())
        return self.feats.get(e, set())

    def curated_feature(self, f: str) -> bool:
        return f in self.curated or any(f.startswith(p) for p in self.curated_prefix)

    def shared(self, a: str, b: str, hooks: set | None = None) -> set:
        """Specific features a and b share: a hook the trace names, a sweep-map command, or one rare in its kind."""
        fa, fb = self.of(a), self.of(b)
        both = fa & fb
        if not both:
            return set()
        ka, kb = (("enemy" if k == "encounter" else k) for k in (entity_of(a).split(":", 1)[0], entity_of(b).split(":", 1)[0]))
        common = self.common.get(ka, set()) | self.common.get(kb, set())
        ubiq = self.ubiquitous.get(ka, set()) | self.ubiquitous.get(kb, set())
        return {f for f in both if (hooks and f in hooks) or f not in common or (self.curated_feature(f) and f not in ubiq)}


# ================================================================== symptoms from each source

def trace_event(d: dict | None) -> tuple[str | None, list, str | None, bool | None]:
    """(first differing event without numbers, the segment's game hooks, its first-difference text, reproducible)."""
    if not d or d.get("error"):
        return None, [], (d or {}).get("error"), (d or {}).get("reproducible")
    seg = d.get("segment") or {}
    norm = lambda evs: [" ".join(NUM.sub("", e).replace("#", "").replace("->", "").split()) for e in evs or []]  # noqa: E731
    g, e = norm(seg.get("game")), norm(seg.get("emu"))
    cg, ce = collections.Counter(g), collections.Counter(e)
    g_only = [x for x in g if cg[x] > ce[x]]
    e_only = [x for x in e if ce[x] > cg[x]]
    event = None
    if g_only or e_only:
        event = f"{seg.get('from', '?').split(' ')[0]}: game {g_only[0] if g_only else 'nothing'}; emulator {e_only[0] if e_only else 'nothing'}"
    dd = d.get("draw_difference")
    if not event and dd:
        side = "game" if dd.get("game") else "emulator"
        draws = dd.get("game") or dd.get("emu") or [[dd.get("stream"), 0, "?"]]
        event = f"draw {dd.get('stream')}: only the {side} draws ({str(draws[0][2]).split(' < ')[0]})"
    hooks = [h.split(" x")[0].removeprefix("hook ").strip() for h in seg.get("game_hooks", []) or []]
    text = d.get("first_difference")
    if d.get("unsettled"):
        text = f"GAME RECORD UNSETTLED: {d['unsettled']}; {text}"
    return event, hooks, text, d.get("reproducible")


def stale_after(game: dict | None, step: int, plays: int = 3) -> bool:
    """The game's state stands still across the next `plays` card plays after the step (a stale snapshot)."""
    if not game or not game.get("steps"):
        return False
    steps = game["steps"]
    view = lambda s: json.dumps({k: (s.get("state") or {}).get(k) for k in ("hand", "draw", "discard", "player")},  # noqa: E731
                                sort_keys=True)
    seen = 0
    base = view(steps[step]) if step < len(steps) else None
    for s in steps[step + 1:]:
        if (s.get("action") or {}).get("kind") != "play":
            return False
        if view(s) != base:
            return False
        seen += 1
        if seen >= plays:
            return True
    return False


SHRINK_LINE = re.compile(r"^\s*([0-9A-Z]{10}): (minimal|capped|flaky|unresolved|reproducing|shrinking), (\d+) of (\d+) items in (\d+) game fights")


def shrink_cases(prefix: Path) -> dict:
    """seed -> {status, items, of, fights} from <name>.shrink.json, else from the summary's shrink lines."""
    doc = read_json(prefix.with_name(prefix.name + ".shrink.json"))
    if doc:
        return {c["seed"]: c for c in doc.get("cases", [])}
    out = {}
    with contextlib.suppress(OSError):
        for line in prefix.with_name(prefix.name + ".summary.txt").read_text(encoding="utf-8").splitlines():
            m = SHRINK_LINE.match(line)
            if m:
                out[m.group(1)] = {"seed": m.group(1), "status": m.group(2), "of": int(m.group(4)), "fights": int(m.group(5))}
    return out


def minimal_spec(prefix: Path, seed: str) -> dict | None:
    doc = read_json(prefix.with_name(prefix.name + ".packs") / "minimal" / f"{seed}.json")
    runs = (doc or {}).get("runs") or []
    return runs[0] if runs else None


SELECTION_GAP = re.compile(r"selection|selectable", re.I)  # the setup mirror's words for a card screen it answers differently


def selection_gaps(v: dict) -> list[tuple[int, str]]:
    """(setup item, message) for each item whose card screen the emulator's setup mirror could not answer as the game's
    was answered: a count or eligibility difference (scenarios.judge keeps the mirror's message as mirror_gap_why)."""
    return sorted((int(k), str(m)) for k, m in (v.get("mirror_gap_why") or {}).items() if SELECTION_GAP.search(str(m)))


def derived_verdicts(prefix: Path) -> list | None:
    """Verdicts for a batch.py batch, which writes none: scenarios.judge over its game, emulator and re-anchor records
    (no game, no emulator is run; the records are the files batch.py wrote). None when they are not all there."""
    try:
        import scenarios  # noqa: PLC0415  (referee/scenarios.py)
        from tools.fidelity.fight_compare import names  # noqa: PLC0415
        load = lambda ext: read_json(prefix.with_name(prefix.name + ext))  # noqa: E731
        specs, games, emus, rea = load(".json"), load(".out.json"), load(".emu.json"), load(".reanchor.json")
        if not all(isinstance(d, dict) for d in (specs, games, emus, rea)):
            return None
        sizes = {len(specs.get("runs", [])), len(games.get("results", [])), len(emus.get("results", [])), len(rea.get("results", []))}
        if len(sizes) != 1 or 0 in sizes:
            return None
        return scenarios.judge(specs["runs"], games["results"], emus["results"], rea,
                               names(read_json(ROOT / "data" / "id_map.json", {}) or {}), scenarios.NoContent())
    except Exception:  # noqa: BLE001  (a batch that cannot be judged contributes no symptoms, it does not stop the grouping)
        return None


def selection_symptom(prefix: Path, lab: Path, when: str, src: str, v: dict, spec: dict, aim: dict, gap: tuple[int, str],
                      shared: bool) -> dict:
    """A fight that matched but whose setup mirror could not answer a card screen as the game did (a relic that allows up
    to N picks answered with none, a card the player holds but the emulator does not offer ...): not a divergence of the
    fight, but the count or eligibility difference the choice-edge answers look for."""
    seed, (at, msg) = v["seed"], gap
    items = spec.get("setup", [])
    item = items[at] if 0 <= at < len(items) else {}
    kind, ident = item.get("cmd"), item.get("id")
    enc = next((it.get("id") for it in items if it.get("cmd") == "fight"), None)
    full = [k for it in items for k in item_keys(it)]
    own = not shared and (src != "aimed" or aim.get("seed_from") == "fresh")
    what = [f"{kind}:{ident}"] if kind in ("relic", "card", "potion") and ident else ([f"encounter:{enc}"] if enc else [])
    held = sorted({entity_of(k) for k in full if not k.startswith(("deck:", "act:"))} | ({f"encounter:{enc}"} if enc else set()))
    return {"source": src, "ref": f"{rel(prefix, lab)}#{seed}", "seed": seed, "when": when,
            "batch": [prefix.name] + ([prefix.parent.name] if prefix.parent.parent.name == "runs" else []), "seed_own": own,
            "held": held, "step": at, "field": "setup.selection", "phase": "setup", "card": ident,
            "game": clip(f"{kind} {ident}: a card screen the game answered"), "emu": clip(msg),
            "first_event": clip(re.sub(r"\d+", "#", msg), 160), "event_text": clip(msg, 240), "hooks": [], "reproducible": None,
            "stale": False, "flags": v.get("flags", []), "mirror_gap": True, "encounter": enc, "shrink": None,
            "minimal_setup": None, "ingredients": None, "dropped": [], "implicated": what, "aim": aim.get("target")}


def batch_symptoms(prefix: Path, lab: Path, when: str, source: str | None = None) -> list[dict]:
    """The diverging fights of one setup or aimed batch (<prefix>.verdicts.json and its neighbours); a batch.py batch has
    no verdicts file, so its verdicts are derived. A fight whose setup mirror could not answer a card screen as the game
    did is one symptom, the screen (selection_symptom), whether or not the fight then diverges: what follows a gap says
    nothing about the emulator."""
    verdicts = read_json(prefix.with_name(prefix.name + ".verdicts.json"))
    if verdicts is None:
        verdicts = derived_verdicts(prefix)
    verdicts = verdicts or []
    specs = (read_json(prefix.with_name(prefix.name + ".json"), {}) or {}).get("runs", [])
    aim_list = (read_json(prefix.with_name(prefix.name + ".aims.json"), {}) or {}).get("aims", [])
    src = source or ("aimed" if aim_list or prefix.with_name(prefix.name + ".aims.json").exists() else "setup")
    # A seed on two specs of one batch (an older targets batch drew cached seeds with replacement): the spec, aim and
    # game record are taken by index. Shrink and trace_divergent key their outputs by seed, so the shrink case, the
    # minimal pack and the trace are used only when one of the seed's fights diverges; the trace also only when its
    # step is past the start or its fight is the seed's last (a start divergence is traced off the reanchor record
    # trace_divergent takes by seed, the last one).
    dup = {s for s, n in collections.Counter(x.get("seed") for x in (specs or verdicts)).items() if n > 1}
    dup_div = {s for s, n in collections.Counter(x.get("seed") for x in verdicts if not x.get("error")
                                                 and not x.get("identical") and "step" in x).items() if n > 1}
    last = {x.get("seed"): i for i, x in enumerate(specs or verdicts)}
    cases = shrink_cases(prefix)
    tdir = prefix.with_name(prefix.name + ".packs") / "traces"
    games = None
    out = []
    for i, v in enumerate(verdicts):
        if v.get("error"):
            continue
        seed = v["seed"]
        gaps = selection_gaps(v)
        if gaps:  # the mirror could not answer a card screen: the fight's later differences say nothing about the emulator
            out.append(selection_symptom(prefix, lab, when, src, v, by_index(specs, i, seed, seed in dup) or {},
                                         by_index(aim_list, i, seed, seed in dup) or {}, gaps[0], seed in dup))
            continue
        if v.get("identical") or "step" not in v:
            continue
        spec = by_index(specs, i, seed, seed in dup) or {}
        aim = by_index(aim_list, i, seed, seed in dup) or {}
        case = {} if seed in dup_div else cases.get(seed) or {}
        mini = None
        if seed not in dup_div and (case.get("status") in ("minimal", "capped") or not case):
            mini = minimal_spec(prefix, seed)
        if mini is None and case.get("status") in ("minimal", "capped"):
            mini = {"setup": case.get("items", [])}
        enc = next((it.get("id") for it in spec.get("setup", []) if it.get("cmd") == "fight"), None)
        full = [k for it in spec.get("setup", []) for k in item_keys(it)]
        kept = [k for it in (mini or {}).get("setup", []) for k in item_keys(it)] if mini else None
        if kept is not None and enc and not [k for k in kept if not k.startswith(("deck:", "act:"))]:
            kept.append(f"encounter:{enc}")  # the fight alone diverges: the encounter is the ingredient
        traced_alone = seed not in dup or (seed not in dup_div and (v.get("step", 0) > 0 or last.get(seed) == i))
        tr = read_json(tdir / f"{seed}.trace.json") if traced_alone else None
        event, hooks, text, repro = trace_event(tr)
        stale = False
        if v.get("step", 0) > 0:
            if games is None:
                games = (read_json(prefix.with_name(prefix.name + ".out.json"), {}) or {}).get("results", [])
            stale = stale_after(by_index(games, i, seed, seed in dup), v["step"])
        diff = (v.get("diff") or [[None, None, None]])[0]
        # a seed names this fight only when the batch made it fresh: a borrowed one (targets mode reuses seeds of
        # recorded fights for their act-1 variant) also names the recorded fight a backlog item may cite
        own = seed not in dup and (src != "aimed" or aim.get("seed_from") == "fresh")
        held = sorted({entity_of(k) for k in full if not k.startswith(("deck:", "act:"))} | ({f"encounter:{enc}"} if enc else set()))
        out.append({
            "source": src, "ref": f"{rel(prefix, lab)}#{seed}", "seed": seed, "when": when,
            "batch": [prefix.name] + ([prefix.parent.name] if prefix.parent.parent.name == "runs" else []), "seed_own": own,
            "held": held,
            "step": v.get("step"), "field": v.get("field"), "phase": phase_of(v.get("card")), "card": v.get("card"),
            "game": clip(diff[1]), "emu": clip(diff[2]), "first_event": event, "event_text": clip(text, 240),
            "hooks": hooks, "reproducible": repro, "stale": stale, "flags": v.get("flags", []),
            "mirror_gap": bool(v.get("mirror_gap")), "encounter": enc,
            "shrink": case.get("status") if case else None, "minimal_setup": (mini or {}).get("setup") if mini else None,
            "ingredients": sorted(set(kept)) if kept is not None else None,
            "dropped": sorted(set(full) - set(kept)) if kept is not None else [],
            "implicated": implicated(v.get("card"), enc, kept), "aim": aim.get("target"),
            "retest_spec": retest_spec(spec, mini, seed) if seed not in dup else None,
        })
    return out


def retest_spec(spec: dict, mini: dict | None, seed: str) -> dict | None:
    """The fight spec a retest plays: the example's own spec with its setup cut to the shrunk one when there is one."""
    setup = (mini or {}).get("setup") or spec.get("setup")
    if not setup:
        return None
    base = {k: v for k, v in spec.items() if k not in ("setup", "tag", "script", "choice_edges")} or {
        "kind": "combat", "character": "IRONCLAD", "ascension": 10, "neow": 0, "chooser_seed": 7000, "max_actions": 150, "use_potions": True}
    return {**base, "seed": seed, "setup": setup}


def by_index(rows: list, i: int, seed: str, shared: bool) -> dict | None:
    """Row i of a batch file when it is this seed's, else the seed's own row (none when two fights share the seed)."""
    if i < len(rows) and (rows[i] or {}).get("seed") == seed:
        return rows[i]
    return None if shared else next((r for r in rows if (r or {}).get("seed") == seed), None)


def implicated(card, enc: str | None, kept: list | None) -> list:
    """What a symptom points at: its kept ingredients; with none kept, the encounter; unshrunk, its action's model."""
    if kept is not None:
        got = sorted({entity_of(k) for k in kept if not k.startswith(("deck:", "act:"))})
        return got or ([f"encounter:{enc}"] if enc else [])
    c = str(card or "")
    if phase_of(c) == "play":
        return [f"card:{c.rstrip('+')}"]
    if phase_of(c) == "potion" and ":" in c:
        return [f"potion:{c.split(':', 1)[1]}"]
    return [f"encounter:{enc}"] if enc else []


def corpus_symptoms(path: Path, lab: Path, when: str) -> list[dict]:
    doc = read_json(path, {}) or {}
    out = []
    for fid, f in (doc.get("fights") or {}).items():
        if f.get("fd") is None:
            continue
        card = f.get("card")
        out.append({"source": "corpus", "ref": f"corpus:{fid}", "seed": None, "when": when, "step": f.get("fd"),
                    "field": f.get("field"), "phase": phase_of(card), "card": card, "game": None, "emu": None,
                    "first_event": None, "event_text": clip(f.get("stop"), 200), "hooks": [], "reproducible": None,
                    "stale": False, "flags": [], "mirror_gap": False, "encounter": None, "shrink": None,
                    "minimal_setup": None, "ingredients": None, "dropped": [], "implicated": implicated(card, None, None),
                    "aim": None})
    return out


def refset_symptoms(path: Path, lab: Path, when: str) -> list[dict]:
    score = read_json(path, {}) or {}
    rec = Path(str(score.get("record_dir") or ""))
    man = read_json(rec.parent.parent / "manifest.json") if rec.name else None
    enc_of = {}
    for e in (man or {}).get("specs", []):
        for it in (e.get("spec") or {}).get("setup", []):
            if it.get("cmd") == "fight":
                enc_of[e.get("id")] = it.get("id")
    out = []
    for v in score.get("verdicts", []) or []:
        if v.get("error") or v.get("identical") or "step" not in v or v.get("step") is None:
            continue
        diff = (v.get("diff") or [[None, None, None]])[0]
        enc = enc_of.get(v.get("id"))
        out.append({"source": "refset", "ref": f"refset:{v.get('id')}", "seed": v.get("seed"), "when": when,
                    "step": v.get("step"), "field": v.get("field"), "phase": phase_of(v.get("card")), "card": v.get("card"),
                    "game": clip(diff[1]), "emu": clip(diff[2]), "first_event": None, "event_text": None, "hooks": [],
                    "reproducible": None, "stale": False, "flags": v.get("flags", []), "mirror_gap": bool(v.get("mirror_gap")),
                    "encounter": enc, "shrink": None, "minimal_setup": None, "ingredients": None, "dropped": [],
                    "implicated": implicated(v.get("card"), enc, None), "aim": None, "part": v.get("part")})
    return out


def walk_symptoms(folder: Path, lab: Path, when: str) -> list[dict]:
    out = []
    for f in sorted(folder.glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                st = d.get("stop")
                if d.get("status") != "mismatch" or not st:
                    continue
                ents = [snake(tok) if not tok.isupper() else tok for tok in stop_tokens(st)]
                out.append({"source": "codex", "ref": f"{rel(f, lab)}:{n}", "seed": None, "when": when,
                            "step": st.get("floor"), "field": f"{st.get('component')}.{st.get('field')}", "phase": "run",
                            "card": None, "game": clip(st.get("expected")), "emu": clip(st.get("emulator")),
                            "first_event": st.get("class"), "event_text": clip(st.get("note"), 200), "hooks": [],
                            "reproducible": None, "stale": False, "flags": [], "mirror_gap": False, "encounter": None,
                            "shrink": None, "minimal_setup": None, "ingredients": None, "dropped": [],
                            "implicated": [], "walk_tokens": sorted(set(ents)), "aim": None,
                            "act": st.get("act")})
    return out


def invariant_symptoms(path: Path, lab: Path, when: str) -> list[dict]:
    doc = read_json(path, {}) or {}
    out = []
    for side in ("game", "emulator"):
        for check, kinds in ((doc.get(side) or {}).get("kinds") or {}).items():
            for kind_text, n in kinds:
                norm = NUM.sub("#", str(kind_text))
                ents = sorted({f"card:{m}" for m in re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", str(kind_text))})
                for _ in range(min(int(n), 1)):
                    out.append({"source": "invariants", "ref": f"{rel(path, lab)}#{side}:{check}", "seed": None,
                                "when": when, "step": None, "field": check, "phase": side, "card": None, "game": None,
                                "emu": None, "first_event": clip(norm, 120), "event_text": None, "hooks": [],
                                "reproducible": None, "stale": False, "flags": [], "mirror_gap": False, "encounter": None,
                                "shrink": None, "minimal_setup": None, "ingredients": None, "dropped": [],
                                "implicated": ents, "aim": None, "weight": int(n)})
    return out


def mutation_symptoms(path: Path, lab: Path, when: str) -> list[dict]:
    doc = read_json(path, {}) or {}
    out = []
    for s in doc.get("survivors", []) or []:
        stem = str(s.get("id", "")).split(":")[0]
        out.append({"source": "mutation", "ref": f"{rel(path, lab)}#{s.get('id')}", "seed": None, "when": when,
                    "step": None, "field": stem, "phase": "test-gap", "card": None, "game": clip(s.get("old"), 120),
                    "emu": clip(s.get("new"), 120), "first_event": s.get("op"), "event_text": None, "hooks": [],
                    "reproducible": None, "stale": False, "flags": [], "mirror_gap": False, "encounter": None,
                    "shrink": None, "minimal_setup": None, "ingredients": None, "dropped": [], "implicated": [],
                    "aim": None})
    return out


def run_symptoms(path: Path, lab: Path, when: str) -> list[dict]:
    """The whole runs of one runs.py batch that diverge: <name>.runs.json (run_compare2 --json), each run at its first
    divergence. Grouped by where it is (component and field) like the codex walk's stops, which are whole runs too.
    A run whose fight was played again as a setup fight (scenarios.py --mode runs, <name>-repro beside it) names that
    fight's symptom ("repro": its ref and whether it diverged as the run did); the setup batch is its own symptom."""
    out = []
    rbase = path.with_name(path.name.removesuffix(".runs.json") + "-repro")
    repros = {x.get("seed"): x for x in (read_json(rbase.with_name(rbase.name + ".repro.json"), {}) or {}).get("runs", []) or []}
    for r in (read_json(path, {}) or {}).get("runs", []) or []:
        first = r.get("first")
        if not r.get("recorded") or not first:
            continue
        diffs = first.get("diffs") or [["", None, None]]
        stem = NUM.sub("#", str(diffs[0][0]).split(": ")[-1]).strip()[:60]
        ents = [snake(tok) if not tok.isupper() else tok for tok in stop_tokens({"expected": diffs[0][1], "emulator": diffs[0][2]})]
        out.append({"source": "run", "ref": f"{rel(path, lab)}#{r.get('seed')}", "seed": r.get("seed"), "when": when,
                    "step": first.get("decision"), "field": f"{first.get('component')}.{stem}", "phase": "run", "card": None,
                    "game": clip(diffs[0][1]), "emu": clip(diffs[0][2]), "first_event": str(first.get("component")),
                    "event_text": clip(f"floor {first.get('floor')}, choice {first.get('choice')}: {diffs[0][0]}", 240), "hooks": [],
                    "reproducible": None, "stale": False, "flags": [], "mirror_gap": False, "encounter": None, "shrink": None,
                    "minimal_setup": None, "ingredients": None, "dropped": [], "implicated": [], "walk_tokens": sorted(set(ents)),
                    "aim": None,
                    "repro": ({"ref": f"{rel(rbase, lab)}#{r.get('seed')}", "match": repros[r.get("seed")].get("repro")}
                              if r.get("seed") in repros else None)})
    return out


def source_kind(p: Path) -> str | None:
    """Which presort output a JSON file is, by its opening keys: a corpus check's results.json, a reference-set
    score, an invariants report (tools/fidelity/invariants.py --json) or a mutation report (mutate.py report --json)."""
    try:
        size = p.stat().st_size
        with p.open(encoding="utf-8") as fh:
            head = fh.read(4096)
    except (OSError, UnicodeDecodeError):
        return None
    if '"format"' in head and '"totals"' in head and '"fights"' in head:
        return "corpus"
    if '"set": "Ironclad A10 reference set' in head and '"headline"' in head:
        return "refset"
    if '"label": "game recordings"' in head:
        return "invariants"
    if size < (32 << 20) and '"kill' in head:
        with contextlib.suppress(OSError, UnicodeDecodeError):
            if '"survivors"' in p.read_text(encoding="utf-8"):
                return "mutation"
    return None


def pool_size(kind: str, path: Path, found: int) -> int:
    """How many fights (or runs) a source judged, the denominator of its symptoms' rate."""
    if kind == "batch":
        return (len(read_json(path.with_name(path.name + ".verdicts.json"), []) or [])
                or len((read_json(path.with_name(path.name + ".out.json"), {}) or {}).get("results", []))) or found
    if kind == "run":
        return len((read_json(path, {}) or {}).get("runs", [])) or found
    if kind == "corpus":
        return int(((read_json(path, {}) or {}).get("totals") or {}).get("fights") or found)
    if kind == "refset":
        return len((read_json(path, {}) or {}).get("verdicts") or []) or found
    if kind == "codex":
        n = 0
        for f in path.glob("*.jsonl"):
            with f.open(encoding="utf-8") as fh:
                n += sum(1 for _ in fh)
        return n or found
    return found


def job_sources(job: Path, when: str) -> list[tuple[str, Path, str]]:
    """(kind, path, when) for every output of one job folder, recognised by its shape (the sniffing: it opens the head of every
    json file that is not a batch's own record). The order is the order the grouping reads them in."""
    out = []
    for v in sorted(job.rglob("*.verdicts.json")):
        out.append(("batch", v.with_name(v.name.removesuffix(".verdicts.json")), when))
    for r in sorted(job.rglob("*.reanchor.json")):  # a batch.py batch has records but no verdicts: they are derived
        base = r.with_name(r.name.removesuffix(".reanchor.json"))
        if not any(x.endswith(".packs") for x in r.parts) and not base.with_name(base.name + ".verdicts.json").exists()                 and all(base.with_name(base.name + e).exists() for e in (".json", ".out.json", ".emu.json")):
            out.append(("batch", base, when))
    for p in sorted(job.rglob("*.json")):
        if p.name.endswith(".runs.json") and not any(x.endswith(".packs") for x in p.parts):
            out.append(("run", p, when))
            continue
        if p.name.endswith((".verdicts.json", ".out.json", ".emu.json", ".reanchor.json", ".trace.json")) or any(
                x.endswith(".packs") for x in p.parts):
            continue
        kind = source_kind(p)
        if kind:
            out.append((kind, p, when))
    walks = {f.parent for f in job.rglob("*.jsonl")}
    for w in sorted(walks):
        with contextlib.suppress(OSError):
            with next(iter(sorted(w.glob("*.jsonl")))).open(encoding="utf-8") as fh:
                first = fh.readline()
            if '"checks"' in first and '"status"' in first:
                out.append(("codex", w, when))
    out += job_outputs(job, when, {str(p) for _, p, _ in out})
    return out


def lab_sources(lab: Path, jobs: tuple[Path, ...] = (), cache=None) -> list[tuple[str, Path, str]]:
    """(kind, path, when) for every output of a finished job under LAB/runs, recognised by its shape, and of each
    job in `jobs` though unfinished (the runner's own job, whose groups step runs before its done.rc is written).
    With a `cache` (tools/lab/groupcache.py) a finished job whose json files are as they were is not sniffed again."""
    out = []
    runs = Path(lab) / "runs"
    done = [j for j in (sorted(runs.iterdir()) if runs.is_dir() else []) if j.is_dir() and (j / "done.rc").exists()]
    seen = {j.resolve() for j in done}
    finished = set(done)
    for job in done + [j for j in jobs if j.is_dir() and j.resolve() not in seen]:
        when = stamp_utc(job)
        if cache is not None and job in finished:
            out += cache.job_sources(job, when, lambda job=job, when=when: job_sources(job, when))
        else:
            out += job_sources(job, when)
    return out


OUTPUT_KINDS = {"batch": "batch", "run": "run", "runs": "run", "corpus": "corpus", "corpus_results": "corpus", "codex": "codex", "codex_walk": "codex",
                "refset": "refset", "refset_score": "refset", "invariants": "invariants", "mutation": "mutation"}


def job_outputs(job: Path, when: str, have: set) -> list[tuple[str, Path, str]]:
    """Outputs a job keeps outside its folder (a corpus check's results.json under the referee's corpus runs),
    named in its job.json: {"outputs": {"corpus": PATH, "codex_walk": DIR, "refset_score": PATH, ...}}, a path
    absolute or relative to the job folder, or a list of them. A codex walk named by its shard files
    (the runner's codex/s0.jsonl ...) is read as their folder, once."""
    doc = read_json(job / "job.json", {}) or {}
    out = []
    have = set(have)
    for key, val in (doc.get("outputs") or {}).items():
        kind = OUTPUT_KINDS.get(key)
        for v in (val if isinstance(val, list) else [val]):
            if not kind or not v:
                continue
            p = Path(v) if Path(v).is_absolute() else job / v
            if kind == "codex" and p.suffix == ".jsonl":
                p = p.parent
            if str(p) not in have and (p.exists() or kind == "batch" and p.with_name(p.name + ".verdicts.json").exists()):
                out.append((kind, p, when))
                have.add(str(p))
    return out


# ================================================================== skip the step when its inputs have not changed

GROUPS_CODE = ("tools/fidelity/lab_groups.py", "tools/fidelity/lab_targets.py", "tools/fidelity/hook_table.py", "tools/lab/stepcache.py")


def source_files(kind: str, path: Path, listing: dict, referee: Path) -> list[Path]:
    """The files the reader of one source looks at: a batch's <name>.* files and its .packs folder, a runs.json and the
    <name>-repro.* files beside it, a codex walk's *.jsonl, a reference-set score and its manifest, else the file itself.
    `listing` caches folder listings (a dict, or a groupcache.Listing)."""
    names = getattr(listing, "names", None)  # a groupcache.Listing: the folders are listed once for the whole run

    def ls(d: Path) -> list[str]:
        if names is not None:
            return names(d)
        if d not in listing:
            try:
                listing[d] = sorted(os.listdir(d))
            except OSError:
                listing[d] = []
        return listing[d]

    path = Path(path)
    if kind == "batch":
        files = [path.parent / n for n in ls(path.parent) if n.startswith(path.name + ".") and not n.endswith(".packs")]
        packs = path.parent / (path.name + ".packs")
        if packs.is_dir():
            files += listing.walk_files(packs) if names is not None else sc.walk_files(packs)
        return files
    if kind == "run":
        stem = path.name.removesuffix(".runs.json")
        return [path] + [path.parent / n for n in ls(path.parent) if n.startswith(stem + "-repro.")]
    if kind == "codex":
        return [path / n for n in ls(path) if n.endswith(".jsonl")]
    if kind == "refset":
        return [path, referee / "refset" / "v2" / "manifest.json"]
    return [path]


def inputs_parts(lab: Path, sources: list, backlog_path: Path | None, host_gaps: Path, decompiled: Path, referee: Path,
                 listing=None, rows_of=None) -> dict:
    """The named inputs of the grouping, each a short digest: every file a source is read from (size and time: a finished job's
    recordings are written once), the fix list and the other small files by content, the Codex profile, the decompile's model
    files, and the code that does the grouping. The tree's sha is not one of them: only the code is."""
    listing = listing if listing is not None else Listing()  # the folders are listed once, not each file asked for
    rows_of = rows_of or (lambda kind, path: listing.stat_files(source_files(kind, Path(path), listing, referee)))  # (name, sig, time)
    rows: list = []
    for kind, path, _when in sources:
        rows += rows_of(kind, path)
    models = sorted(Path(decompiled).glob("MegaCrit.Sts2.Core.Models*/*.cs")) if Path(decompiled).is_dir() else []
    return {"sources": listing.rows_hash(rows, lab), "n_sources": len(sources),
            "backlog": sc.file_hash(backlog_path) if backlog_path else sc.MISSING,
            "small": sc.files_hash([ROOT / "data" / "id_map.json", ROOT / "data" / "sweep_map.json", ROOT / "referee" / "FINDINGS.md",
                                    Path(host_gaps), Path(lab) / "retests.json"], None),
            "profile": sc.stat_sig(profile_path(lab)),
            "models": f"{len(models)}:{max((m.stat().st_mtime_ns for m in models), default=0)}",
            "code": sc.files_hash([ROOT / c for c in GROUPS_CODE], ROOT)}


# ================================================================== grouping

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


FIGHT_SOURCES = ("aimed", "setup", "corpus", "refset")


def base_key(s: dict) -> tuple:
    """The preliminary signature of one symptom."""
    src = s["source"]
    if src in ("codex", "run"):
        return ("run", s["field"], s.get("first_event"))
    if src == "invariants":
        return ("invariant", s["phase"], s["field"], s.get("first_event"))
    if src == "mutation":
        return ("test-gap", s["field"], tuple(s["implicated"]))
    if s.get("ingredients") is not None:
        return ("fight", s["phase"], s["field"], s.get("first_event"), "shrunk", tuple(s["ingredients"]))
    if specific_event(s.get("first_event")):  # traced: the event names the mechanism better than the acted card
        return ("fight", s["phase"], s["field"], s["first_event"], "traced", ())
    return ("fight", s["phase"], s["field"], None, "unshrunk", tuple(s["implicated"]))


SPECIFIC = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)*\b")


def specific_event(event: str | None) -> bool:
    """An event that names a model (an upper-case id), so two symptoms with it share more than a field."""
    return bool(event and SPECIFIC.search(event))


def pair_explained(a: tuple, b: tuple, feats: Features, hooks: set) -> set | None:
    """The shared features that pair every ingredient a has and b lacks with one b has and a lacks, or None."""
    da, db = [x for x in a if x not in b], [x for x in b if x not in a]
    if len(da) != len(db) or not da:
        return None
    used, why = set(), set()
    for x in da:
        best = None
        for y in db:
            if y in used:
                continue
            sh = feats.shared(x, y, hooks)
            if sh and (best is None or len(sh) > len(best[1])):
                best = (y, sh)
        if best is None:
            return None
        used.add(best[0])
        why |= best[1]
    return why


def group_symptoms(symptoms: list[dict], feats: Features) -> list[dict]:
    """Merge symptoms into root-mechanism groups; each group: its key, member indices, why it merged."""
    prelim: dict = collections.OrderedDict()
    keys_of = [base_key(s) for s in symptoms]
    shrunk_sizes = collections.Counter(k for k in keys_of if k[0] == "fight" and k[4] == "shrunk")
    for i, s in enumerate(symptoms):
        k = keys_of[i]
        if k[0] == "fight" and k[4] == "unshrunk":
            to = join_target(s, k, shrunk_sizes)
            k = k + (to,) if to else k
        prelim.setdefault(k, []).append(i)
    keys = list(prelim)
    index = {k: j for j, k in enumerate(keys)}
    dsu = DSU(len(keys))
    why: dict = collections.defaultdict(set)
    hooks_of = {k: {f"hook:{h}" for i in prelim[k] for h in symptoms[i].get("hooks", [])} for k in keys}
    buckets = collections.defaultdict(list)
    for j, k in enumerate(keys):
        if k[0] == "fight":
            buckets[(k[1], k[2])].append(j)
    for _, js in buckets.items():
        shrunk = [j for j in js if keys[j][4] == "shrunk"]
        for x, y in ((x, y) for n, x in enumerate(shrunk) for y in shrunk[n + 1:]):
            kx, ky = keys[x], keys[y]
            if kx[3] and ky[3] and kx[3] != ky[3]:
                continue
            got = pair_explained(kx[5], ky[5], feats, hooks_of[kx] | hooks_of[ky])
            if not got and kx[3] and kx[3] == ky[3] and specific_event(kx[3]):
                got = {f"event:{kx[3]}"}  # the same first event, naming a model: one mechanism
            if got:
                why[dsu.find(x)] |= got
                dsu.union(x, y)
                why[dsu.find(x)] |= got
        for t in (j for j in js if keys[j][4] == "traced"):  # a traced symptom joins the shrunk ones with its event
            same = [s_ for s_ in shrunk if keys[s_][3] == keys[t][3]]
            if same:
                dsu.union(t, max(same, key=lambda s_: len(prelim[keys[s_]])))
        for u in (j for j in js if keys[j][4] == "unshrunk" and len(keys[j]) > 6):
            dsu.union(u, index[keys[u][6]])
    comps = collections.defaultdict(list)
    for j in range(len(keys)):
        comps[dsu.find(j)].append(j)
    groups = []
    for js in comps.values():
        members = [i for j in js for i in prelim[keys[j]]]
        ks = [keys[j] for j in js]
        reasons = set()
        for j in js:
            reasons |= why.get(j, set())
        groups.append({"keys": ks, "members": members, "mechanism": sorted(reasons)})
    return groups


STARTER = frozenset(f"card:{c}" for c in BASIC_CARDS)  # every deck holds them: no hint of a mechanism


def join_target(s: dict, k: tuple, shrunk_sizes: dict) -> tuple | None:
    """The shrunk key (same phase and field) an unshrunk symptom joins, or None.

    The acted card or the encounter is a weak hint, so it must be one no deck holds by default (not a starter
    card), and the minimal setup's every needed ingredient (starter cards aside) must be one the unshrunk fight
    is known to hold: its setup's items for a setup or aimed batch, else only what it acted with and its
    encounter. The most common such shrunk key wins (the first on a tie)."""
    hints = set(k[5]) - STARTER
    if not hints:
        return None
    known = set(k[5]) | set(s.get("held") or ()) | ({f"encounter:{s['encounter']}"} if s.get("encounter") else set())
    best = None
    for sk, n in shrunk_sizes.items():
        if (sk[1], sk[2]) != (k[1], k[2]):
            continue
        need = {entity_of(z) for z in sk[5] if not z.startswith(("deck:", "act:"))} - STARTER
        if need & hints and need <= known and (best is None or n > best[1]):
            best = (sk, n)
    return best[0] if best else None


def group_key(g: dict) -> dict:
    ks = g["keys"]
    k0 = ks[0]
    if k0[0] != "fight":
        return {"family": k0[0], "parts": [str(x) for x in k0[1:]]}
    shrunk = [k for k in ks if k[4] == "shrunk"]
    ing_sets = [set(k[5]) for k in shrunk]
    common = sorted(set.intersection(*ing_sets)) if ing_sets else []
    imp = sorted({z for k in ks if k[4] == "unshrunk" for z in k[5]})
    events = sorted({k[3] for k in ks if k[3] and k[4] in ("shrunk", "traced")})
    key = {"family": "fight", "phase": k0[1], "field": k0[2], "event": events[0] if len(events) == 1 else (events or None),
           "ingredients": common, "mechanism": g["mechanism"]}
    if not shrunk and not events:
        key["implicated"] = imp
    return key


def stable_id(key: dict) -> str:
    return "G" + hashlib.sha1(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:10]


# ================================================================== host suspicion, backlog, impact

class HostGaps:
    def __init__(self, doc: dict | None, findings_text: str | None):
        self.entries = (doc or {}).get("entries", [])
        self.flags = set((doc or {}).get("plausibility_flags", ["cross_act", "cross_character", "duplicate_relic"]))
        self.stale_entries = []
        if findings_text is not None:
            for e in self.entries:
                quotes = [e.get("finding", {})] + e.get("also", [])
                if not all(q.get("quote", "") in findings_text for q in quotes):
                    self.stale_entries.append(e["id"])

    def judge(self, members: list[dict], entities: set) -> tuple[bool, str]:
        reasons = self.reasons(members, entities)
        return bool(reasons), "; ".join(r for r, _ in reasons)

    def retest_only(self, members: list[dict], entities: set) -> bool:
        """True when the only reasons are host gaps since fixed: a retest on the current tree can settle the group."""
        reasons = self.reasons(members, entities)
        return bool(reasons) and all(fixed for _, fixed in reasons)

    def reasons(self, members: list[dict], entities: set) -> list[tuple[str, bool]]:
        """[(reason, from a gap since fixed)] of every rule the group meets."""
        reasons = []
        phases = {s["phase"] for s in members}
        fields = {s["field"] for s in members}
        texts = " ".join(str(s.get("event_text") or "") + " " + str(s.get("first_event") or "") for s in members)
        for e in self.entries:
            m = e.get("match", {})
            ents = set(m.get("entities", []))
            hit_ent = bool(ents & entities)
            hit_text = any(t in texts for t in m.get("text", []) + m.get("errors", []))
            if not (hit_ent or hit_text):
                continue
            if m.get("phases") and not phases & set(m["phases"]) and not hit_text:
                continue
            if m.get("fields") and not any(f and any(f.startswith(x) for x in m["fields"]) for f in fields) and not hit_text:
                continue
            reasons.append((f"{e['id']} ({e.get('status', 'open')}; FINDINGS {e.get('finding', {}).get('date')}): {e.get('why', '')}",
                            e.get("status") == "fixed"))
        fight = [s for s in members if s["source"] in FIGHT_SOURCES]
        if fight and all(set(s.get("flags", [])) & self.flags for s in fight):
            fl = sorted({f for s in fight for f in s.get("flags", []) if f in self.flags})
            reasons.append((f"every example's setup is implausible ({', '.join(fl)}): no real run holds it", False))
        if any(s.get("stale") for s in members):
            reasons.append(("the game's state stands still across the next plays after the step (stale snapshot)", False))
        if any(s.get("reproducible") is False for s in members):
            reasons.append(("a trace re-run did not reproduce the batch before the step (game-side flake?)", False))
        if any(s["source"] == "invariants" and s["phase"] == "game" for s in members):
            reasons.append(("an invariant fails on the game's own recording (recording or driver issue)", False))
        return reasons


def backlog_index(backlog: dict | None, id_map: dict) -> list[dict]:
    """Open items with the entities, seeds and fields their title and evidence name."""
    if not backlog:
        return []
    try:
        from tools.fidelity.packet import Names, entities_in  # noqa: PLC0415
        names = Names(id_map)
    except Exception:  # noqa: BLE001
        names = entities_in = None
    out = []
    for it in backlog.get("items", []):
        if it.get("status", "open") != "open":
            continue
        text = f"{it.get('title', '')} {it.get('evidence', '')}"
        ents = set()
        if names:
            for key, (kind, _) in entities_in(text, names).items():
                ents.add(f"{kind}:{key}_POWER" if kind == "power" else f"{kind}:{key}")
        ents |= {f"?:{m}" for m in re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", text)}
        out.append({"id": it["id"], "lane": it.get("lane"), "title": it.get("title", ""), "entities": ents,
                    "seeds": set(SEED.findall(it.get("evidence", ""))), "fields": {"." + f for f in FIELD.findall(text)},
                    "subsystem": it.get("subsystem", ""), "text": text.lower()})
    return out


def seed_refs(g: dict) -> set:
    """(seed, batch names, the seed names this fight alone) of a group's members; _seeds alone: own seeds."""
    if "_seedrefs" in g:
        return set(g["_seedrefs"])
    return {(s, (), True) for s in g.get("_seeds", ()) if s}


def names_batch(name: str, text: str) -> bool:
    return bool(name) and re.search(rf"(?<![\w-]){re.escape(name.lower())}(?![\w-])", text) is not None


def cited_seeds(refs: set, it: dict, batches_of: dict) -> set:
    """The seeds an item's evidence cites for these symptoms' own fights. A seed may name several fights (targets
    mode borrows recorded fights' seeds): when the evidence names one of the batches a seed was seen in, it cites
    those batches' fights; when it names none, it cites the fight the seed is its own, never a borrowed copy."""
    got = set()
    for sd, names, own in refs:
        if sd not in it["seeds"]:
            continue
        named = {b for b in batches_of.get(sd, ()) if names_batch(b, it["text"])}
        if (named & set(names)) or (own and not named):
            got.add(sd)
    return got


def match_backlog(groups: list[dict], items: list[dict], min_score: float = 2.5) -> None:
    """backlog_ids and relation on each group (see the module doc)."""
    df = collections.Counter(e.split(":", 1)[1] for it in items for e in {x for x in it["entities"]})
    n = max(1, len(items))
    # 3 for an id one item names, falling to 1 for one every item names: a lone id matches when few items name it
    idf = lambda e: 3.0 if df.get(e, 0) <= 1 else min(3.0, 1.0 + 2.0 * max(0.0, math.log((n + 1) / (1 + df[e]))) / math.log(max(3.0, (n + 1) / 2)))  # noqa: E731
    refs = [seed_refs(g) for g in groups]
    batches_of = collections.defaultdict(set)  # seed -> every batch name a symptom with it came from
    for rs in refs:
        for sd, names, _ in rs:
            batches_of[sd] |= set(names)
    hits = collections.defaultdict(list)
    for gi, g in enumerate(groups):
        ents = {e.split(":", 1)[1] for e in g["_entities"]}
        for it in items:
            item_ids = {e.split(":", 1)[1] for e in it["entities"]}
            common = ents & item_ids
            seeds = cited_seeds(refs[gi], it, batches_of)
            score = sum(idf(e) for e in common) + 4.0 * len(seeds)
            if score and g["_fields"] & it["fields"]:
                score += 0.5
            if g["key"].get("family") == "run" and it.get("lane") == "run":
                comp = (g["key"].get("parts") or [""])[0].split(".")[0]
                if comp and comp.lower() in it["text"]:
                    score += 1.0
            if (common or seeds) and score >= min_score:
                hits[gi].append((score, it["id"]))
    by_item = collections.defaultdict(list)
    for gi, hs in hits.items():
        hs.sort(reverse=True)
        top = hs[0][0]
        keep = [i for s, i in hs if s >= 0.5 * top][:5]
        groups[gi]["backlog_ids"] = keep
        for i in keep:
            by_item[i].append(gi)
    for g in groups:
        ids = g.setdefault("backlog_ids", [])
        if not ids:
            g["relation"] = "new"
        elif len(ids) > 1:
            g["relation"] = "merges"
        elif len(by_item[ids[0]]) > 1:
            g["relation"] = "splits"
        else:
            g["relation"] = "confirms"


COMMON_SHARE = 0.9  # an ingredient more than this share of runs meet says nothing about which runs a group reaches


def common_dropped(prof: Profile, meets: list) -> list:
    """The (runs, entity) pairs without those every run has (the starter deck and relic, anything held in over
    COMMON_SHARE of runs): the estimate comes from the distinctive ingredients. All common: they stay."""
    if not prof.doc.get("runs"):
        return meets
    rare = [(m, e) for m, e in meets if m <= COMMON_SHARE * prof.runs]
    return rare or meets


def impact(prof: Profile, entities: list[str]) -> int:
    """Runs that meet every entity: the rarest one's runs times how often each other is held with it."""
    if not entities:
        return 0
    meets = []
    for e in entities:
        kind, _, ident = e.partition(":")
        if kind == "power" and prof.meet("card", ident.removesuffix("_POWER")):
            e, kind, ident = f"card:{ident.removesuffix('_POWER')}", "card", ident.removesuffix("_POWER")  # its power card
        if kind not in ("enemy", "encounter", "card", "relic", "potion"):
            continue
        meets.append((prof.meet(kind, ident), e))
    if not meets:
        return 0
    meets = common_dropped(prof, meets)
    meets.sort()
    m, rarest = meets[0]
    est = float(m)
    for _, e in meets[1:]:
        kind = e.split(":", 1)[0]
        if kind in ("card", "relic", "potion") and rarest.split(":", 1)[0] in ("card", "relic", "potion"):
            held = sum(prof.held(*rarest.split(":", 1), a) for a in (1, 2, 3))
            pair = sum(prof.pair(a, rarest, e) for a in (1, 2, 3))
            est *= min(1.0, pair / held) if held else 0.0
        else:
            est *= min(1.0, prof.meet(kind, e.split(":", 1)[1]) / prof.runs)
    return int(round(est))


def observed_runs(prof: Profile, members: list[dict], pools: dict) -> int:
    """Runs a group of unshrunk fights reaches, from how often its sources saw it: the worst source's rate per
    fight, over a real run's fights (the profile's fights per run)."""
    per_run = sum(prof.act(a).get("fights", 0) for a in (1, 2, 3)) / prof.runs if prof.doc.get("runs") else 10.0
    by = collections.Counter(s.get("pool") for s in members)
    rate = max((n / max(MIN_POOL, pools.get(k, n)) for k, n in by.items()), default=0.0)
    return int(round(prof.runs * (1 - (1 - min(1.0, rate)) ** max(1.0, per_run))))


def union_runs(runs: int, parts: list[int]) -> int:
    """Runs reached by at least one of several variants, each reaching parts[i] of `runs`, taken as independent: a sum
    over many variants would saturate at every run, the union does not."""
    miss = 1.0
    for n in parts:
        miss *= 1.0 - min(1.0, max(0, n) / max(1, runs))
    return int(round(runs * (1.0 - miss)))


def retest_of(members: list[dict], result: dict | None) -> dict:
    """The retest mark of a group judged only by fixed host gaps: the spec of its smallest example, the fields its
    symptoms differ in (tools/lab/retest.py judges a divergence by them), and the verdict so far. State: pending,
    decided (diverges: the symptom is still there), stale (identical: the fight matches now) or exhausted (unclear
    MAX_ATTEMPTS times: stays a host suspect)."""
    from tools.lab import retest as rt  # noqa: PLC0415

    mine = [s for s in members if s.get("retest_spec")]
    out = {"state": "pending", "spec": mine[0]["retest_spec"] if mine else None, "ref": mine[0]["ref"] if mine else None,
           "fields": sorted({str(s["field"]) for s in members if s.get("field")})}
    if result:
        out.update(verdict=result.get("verdict"), reason=result.get("reason"), utc=result.get("utc"), main_sha=result.get("main_sha"),
                   attempts=result.get("attempts"))
        if result.get("verdict") == "diverges":
            out["state"] = "decided"
        elif result.get("verdict") == "identical":
            out["state"] = "stale"
        elif (result.get("attempts") or 0) >= rt.MAX_ATTEMPTS:
            out["state"] = "exhausted"
    return out


def build(symptoms: list[dict], feats: Features, prof: Profile, hosts: HostGaps, items: list[dict], old: dict | None,
          pools: dict | None = None, retests: dict | None = None) -> list[dict]:
    pools = pools or {}
    retests = retests or {}
    raw = group_symptoms(symptoms, feats)
    old_first = {g["id"]: g.get("first_seen_utc") for g in (old or {}).get("groups", [])}
    old_refs = {e["ref"]: g.get("first_seen_utc") for g in (old or {}).get("groups", []) for e in g.get("examples", [])}
    groups = []
    for g in raw:
        members = [symptoms[i] for i in g["members"]]
        key = group_key(g)
        gid = stable_id(key)
        ents = set()
        for s in members:
            ents |= set(s.get("implicated") or [])
            ents |= {entity_of(k) for k in s.get("ingredients") or [] if not k.startswith(("deck:", "act:"))}
            ents |= {f"?:{t}" for t in s.get("walk_tokens", [])}
            ents |= {f"?:{t}" for t in SPECIFIC.findall(str(s.get("first_event") or "")) if "_" in t or len(t) > 5}
        members.sort(key=lambda s: (len(s["minimal_setup"]) if s.get("minimal_setup") is not None else 99,
                                    s.get("step") if isinstance(s.get("step"), int) else 999, s["ref"]))
        examples = [{"source": s["source"], "ref": s["ref"], "step": s.get("step"), "field": s.get("field"),
                     "game": s.get("game"), "emu": s.get("emu"), "first_event": s.get("first_event") or s.get("event_text"),
                     "minimal_setup": s.get("minimal_setup") if s.get("minimal_setup") is not None else [],
                     **({"shrink": s["shrink"]} if s.get("shrink") else {}), **({"flags": s["flags"]} if s.get("flags") else {})}
                    for s in members[:MAX_EXAMPLES]]
        effect = {}
        for s in members:
            if s.get("shrink") == "minimal":
                for k in s.get("ingredients") or []:
                    effect.setdefault(k, True)
            for k in s.get("dropped") or []:
                effect.setdefault(k, False)
        host_ents = {e for e in ents if not e.startswith("?:")}
        host, host_reason = hosts.judge(members, host_ents)
        retest = None
        stale = False
        if host and hosts.retest_only(members, host_ents):
            retest = retest_of(members, retests.get(gid))
            if retest["state"] == "decided":  # still differs in its own field on the current tree: a real divergence
                host_reason = f"retested ({retest['verdict']}): {retest['reason']}; was: {host_reason}"
                host = False
            elif retest["state"] == "stale":  # matches on the current tree: it came from the fixed gap, nothing to fix
                host_reason = f"retested ({retest['verdict']}): {retest['reason']}; stale, was: {host_reason}"
                host, stale = False, True
            elif retest["state"] == "exhausted":
                host_reason = f"retest unsettled after {retest.get('attempts')} tries ({retest.get('reason')}): {host_reason}"
            elif retest["spec"] is None:
                retest = None  # no example to replay: stays a host suspect
            else:
                host_reason = "retest pending: " + host_reason
        shrunk = [s for s in members if s.get("ingredients") is not None]
        unshrunk = [s for s in members if s.get("ingredients") is None]
        variants = {tuple(s.get("implicated") or []) for s in shrunk}
        if key.get("family") == "run":
            est = len(members)
        elif not prof.doc.get("runs"):
            est = 0
        else:  # shrunk: real runs holding the ingredients together; unshrunk: how often the sources saw it
            parts = [impact(prof, list(v)) for v in variants]
            if unshrunk:
                parts.append(min(observed_runs(prof, unshrunk, pools), max(impact(prof, s.get("implicated") or []) for s in unshrunk)))
            est = union_runs(prof.runs, parts)
        if stale:
            est = 0
        whens = [s["when"] for s in members if s.get("when")]
        first = old_first.get(gid) or min([old_refs[s["ref"]] for s in members if old_refs.get(s["ref"])] + whens or [utc_now()])
        groups.append({
            "id": gid, "key": key, "title": title(key, members), "count": len(members), "first_seen_utc": first,
            "examples": examples, "entities": sorted(e for e in ents if not e.startswith("?:")),
            "ingredient_effect": [{"item": k, "removal_vanishes": v} for k, v in sorted(effect.items(), key=lambda kv: (not kv[1], kv[0]))][:24],
            "host_suspect": host, "host_reason": host_reason, **({"retest": retest} if retest else {}), **({"stale": True} if stale else {}), "impact_runs_est": int(est),
            "sources": sorted({s["source"] for s in members}), "by_job": by_job(members),
            "_entities": ents, "_fields": {s.get("field") for s in members},
            "_seedrefs": {(s["seed"], tuple(s.get("batch") or ()), bool(s.get("seed_own", True))) for s in members if s.get("seed")},
        })
    match_backlog(groups, items)
    for g in groups:
        for k in ("_entities", "_seedrefs", "_fields"):
            g.pop(k, None)
    groups.sort(key=lambda g: (-g["impact_runs_est"], -g["count"], g["id"]))
    return groups


def by_job(members: list[dict]) -> dict:
    """{job folder: {"n": symptoms of it in the group, "ready": those a fixer can start from, "gap": those that are a card
    screen the setup mirror could not answer}}. Ready: the first differing event is traced and the setup was fully
    mirrored, in a fight (a whole run's divergence is located, not traced; the group's host suspicion is judged on the
    group). The runner's yield.tsv counts a job's groups, its fix-ready ones and its card-screen ones from this."""
    out: dict = {}
    for s in members:
        m = JOB_REF.match(str(s.get("ref")))
        if m:
            j = out.setdefault(m.group(1), {"n": 0, "ready": 0, "gap": 0})
            j["n"] += 1
            j["ready"] += 1 if s.get("first_event") and not s.get("mirror_gap") and s.get("source") != "run" else 0  # a whole run has no trace
            j["gap"] += 1 if s.get("field") == "setup.selection" else 0
    return out


def title(key: dict, members: list[dict]) -> str:
    if key.get("family") == "run":
        return clip(f"run: {key['parts'][0]} ({key['parts'][1]})", 120)
    if key.get("family") == "invariant":
        return clip(f"invariant {key['parts'][1]} on the {key['parts'][0]} side: {key['parts'][2]}", 120)
    if key.get("family") == "test-gap":
        return clip(f"mutation survivors in {key['parts'][0]}", 120)
    what = key.get("ingredients") or [f"{'in' if x.startswith('encounter:') else 'after'} {x.split(':', 1)[1]} (not shrunk)"
                                      for x in key.get("implicated") or []]
    if not what:
        sets = sorted({"+".join(s["ingredients"]) for s in members if s.get("ingredients")})
        cards = sorted({str(s.get("card")) for s in members if s.get("ingredients") is None and s.get("card")})
        what = [" | ".join(sets[:3]) + (" ..." if len(sets) > 3 else "")] if sets else [f"after {', '.join(cards[:3])} (traced)"]
    mech = [m.split(":", 1)[1] for m in key.get("mechanism", [])][:2]
    ev = key.get("event") if isinstance(key.get("event"), str) else None
    parts = [f"{key.get('field')} at {key.get('phase')}", ", ".join(what) or "the encounter alone"]
    if mech:
        parts.append("via " + ", ".join(mech))
    if ev:
        parts.append(ev)
    return clip(": ".join(parts[:2]) + ("; " + "; ".join(parts[2:]) if parts[2:] else ""), 160)


def summary_md(doc: dict, symptoms: list[dict], limit: int = 4000) -> str:
    groups = doc["groups"]
    src = collections.Counter(s["source"] for s in symptoms)
    rel_ = collections.Counter(g["relation"] for g in groups)
    head = ["# Fix lab groups", "",
            f"main {doc['main_sha'][:10]}, {doc['generated_utc']}. {len(symptoms)} symptoms "
            f"({', '.join(f'{k} {v}' for k, v in sorted(src.items()))}) in {len(groups)} groups; "
            f"backlog: {rel_.get('confirms', 0)} confirm, {rel_.get('splits', 0)} split, {rel_.get('merges', 0)} merge, "
            f"{rel_.get('new', 0)} new; {sum(g['host_suspect'] for g in groups)} host-suspect, "
            f"{sum(bool(g.get('stale')) for g in groups)} stale (a retest matches the game now).", "",
            "Top 20 by impact (runs est, symptoms, backlog, smallest example):", ""]
    lines = []
    for n, g in enumerate(groups[:20], 1):
        ex = g["examples"][0]["ref"] if g["examples"] else ""
        bl = ",".join(g["backlog_ids"][:3]) or "-"
        lines.append(f"{n}. {g['id']} {g['title'][:110]} | ~{g['impact_runs_est']} runs, {g['count']} | {g['relation']} {bl}"
                     f"{' | HOST?' if g['host_suspect'] else ''}{' | STALE' if g.get('stale') else ''} | {ex[:70]}")
    text = "\n".join(head)
    for line in lines:
        if len(text) + len(line) + 2 > limit:
            break
        text += "\n" + line
    return text + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--lab", type=Path, required=True)
    ap.add_argument("--backlog", type=Path, help="default: the newest referee/backlog/backlog-*.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--job", type=Path, action="append", default=[],
                    help="a LAB/runs job read though it has no done.rc yet (the runner's job, from its own groups step)")
    ap.add_argument("--batch", type=Path, action="append", default=[], help="a setup batch outside the lab: <dir>/<name>")
    ap.add_argument("--corpus-results", type=Path, action="append", default=[])
    ap.add_argument("--codex-walk", type=Path, action="append", default=[])
    ap.add_argument("--refset-score", type=Path, action="append", default=[], help="'newest' for the newest score")
    ap.add_argument("--invariants", type=Path, action="append", default=[])
    ap.add_argument("--mutation", type=Path, action="append", default=[])
    ap.add_argument("--decompiled", type=Path, default=DECOMPILED)
    ap.add_argument("--host-gaps", type=Path, default=HOST_GAPS)
    ap.add_argument("--referee", type=Path, default=REFEREE)
    ap.add_argument("--force", action="store_true",
                    help="group again even when no input changed since the last run (LAB/stamps/groups.json), and read every source "
                         "again (clears LAB/cache/groups)")
    ap.add_argument("--no-cache", action="store_true", help="read every source, and neither use nor write LAB/cache/groups")
    ap.add_argument("--cache-dir", type=Path, help="default: LAB/cache/groups (tools/lab/groupcache.py)")
    a = ap.parse_args(argv)

    lab = a.lab.resolve()
    symptoms = []
    cache = GroupCache(lab, ROOT, directory=a.cache_dir, enabled=not a.no_cache)
    cache.files_of = lambda kind, path: source_files(kind, path, cache.listing, a.referee)
    if a.force:
        cache.clear()
    sources = lab_sources(lab, tuple(a.job), cache)
    sources += [("batch", p, stamp_utc(p.with_name(p.name + ".verdicts.json"))) for p in a.batch]
    sources += [("corpus", p, stamp_utc(p)) for p in a.corpus_results]
    sources += [("codex", p, stamp_utc(next(iter(p.glob("*.jsonl")), p))) for p in a.codex_walk]
    for p in a.refset_score:
        p = newest_refset_score(a.referee) if str(p) == "newest" else p
        if p:
            sources.append(("refset", p, stamp_utc(p)))
    sources += [("invariants", p, stamp_utc(p)) for p in a.invariants]
    sources += [("mutation", p, stamp_utc(p)) for p in a.mutation]
    read = {"batch": batch_symptoms, "run": run_symptoms, "corpus": corpus_symptoms, "refset": refset_symptoms, "codex": walk_symptoms,
            "invariants": invariant_symptoms, "mutation": mutation_symptoms}
    backlog_path = a.backlog or newest_backlog(a.referee)
    parts = inputs_parts(lab, sources, backlog_path, a.host_gaps, a.decompiled, a.referee, cache.listing, cache.rows_of)
    want = sc.digest(parts)
    outputs = sc.outputs_of(lab, [a.out, a.summary])
    why = None if a.force else sc.skip_reason(lab, "groups", want, outputs)
    if why:  # nothing the grouping reads has changed: the file it wrote stands
        print(f"step skipped: groups: {why}")
        cache.finish(complete=False)
        return 0
    per_source, pools = collections.Counter(), {}
    for kind, path, when in sources:
        tok = cache.lookup(kind, Path(path), when)
        if tok.hit:  # read once in its life: the symptoms of this source's files as they are now, by the code as it is now
            got, pool, label = tok.symptoms, tok.pool, tok.label
        else:
            got = read[kind](Path(path), lab, when)
            label = f"{kind}:{rel(path, lab)}"
            pool = pool_size(kind, Path(path), len(got))
            cache.store(tok, got, pool, label)
        per_source[label] = len(got)
        pools[label] = pool
        for s in got:
            s["pool"] = label
        symptoms += got
    cache.finish()
    id_map = read_json(ROOT / "data" / "id_map.json", {}) or {}
    prof = Profile(read_json(profile_path(lab)))
    feats = Features(model_features(a.decompiled, lab / "cache" / "model_features.json"),
                     read_json(ROOT / "data" / "sweep_map.json", {}), prof.doc.get("encounter_monsters", {}))
    findings = None
    with contextlib.suppress(OSError):
        findings = (ROOT / "referee" / "FINDINGS.md").read_text(encoding="utf-8")
    hosts = HostGaps(read_json(a.host_gaps), findings)
    items = backlog_index(read_json(backlog_path) if backlog_path else None, id_map)
    groups = build(symptoms, feats, prof, hosts, items, read_json(a.out), pools, read_json(lab / "retests.json", {}) or {})
    doc = {"main_sha": git_sha(), "generated_utc": utc_now(),
           "inputs": {"sources": dict(per_source), "backlog": rel(backlog_path, lab) if backlog_path else None,
                      "backlog_open_items": len(items), "profile_runs": prof.doc.get("runs", 0),
                      "host_gaps": rel(a.host_gaps, lab), "host_gap_entries_stale": hosts.stale_entries},
           "groups": groups}
    write_json(a.out, doc)
    text = summary_md(doc, symptoms)
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(text, encoding="utf-8")
    # what the output says is its groups; its time, sha and the summary's header are new every run
    sc.write_stamp(lab, "groups", want, outputs, job=a.job[0].name if a.job else "", parts={k: str(v)[:16] for k, v in parts.items()},
                   content={sc.label_of(lab, a.out): sc.digest({"groups": groups}), sc.label_of(lab, a.summary): "-"})
    rel_ = collections.Counter(g["relation"] for g in groups)
    print(f"groups: {len(groups)} from {len(symptoms)} symptoms; {dict(rel_)}; host-suspect {sum(g['host_suspect'] for g in groups)}; stale {sum(bool(g.get('stale')) for g in groups)}"
          + (f"; stale host-gap entries {hosts.stale_entries}" if hosts.stale_entries else ""))
    if cache.enabled:
        print(cache.stats.line(cache.listing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
