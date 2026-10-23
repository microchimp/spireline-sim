"""The port manifest: which game models the records reach, which hooks and families each one overrides, and which
items become eligible as the hook families migrate to the game-shaped dispatcher (referee/design/rebuild-plan.md
sections 3 and 4, referee/design/game-shaped-dispatch.md section 2).

    python -m tools.fidelity.manifest build  [--out <dir>] [--no-codex] [--codex-rescan]
    python -m tools.fidelity.manifest status [--out <dir>]
    python -m tools.fidelity.manifest codex  [--out <dir>]        (only the Codex counts, a slow read of the dataset)

Inputs (all outside the repo, read only):
  hook_families.json     every gameplay hook -> one family F01-F16 (or `unassigned`), checked in
  migrated_families.json {"migrated": [...]}: the families the new dispatcher already runs, filled in as they land
  the decompile          class and hook names only, through hook_table (no game text is copied)
  the corpus             the game's recorded fights: per fight, the models in its piles, on every creature, in its
                         relic and potion lists and in its actions
  the agent's real runs  gen0/real: runs.jsonl and divergences.jsonl record decisions and the entities of DIVERGING
                         decisions only, so a run's model set here is a lower bound (`partial`)
  the lab's recordings   lab/runs/*/fights/*.out.json, one fight per setup; the lab prunes a recording once the corpus
                         holds it, so this is whatever is still on disk
  the Codex dataset      counts per model only, read by manifest_codex.py under `python -I`; never per run

Output: <out>/manifest.json and manifest.md (default: the rebuild folder next to the referee data). Both name sources
by role, not by path.

`eligible(item, migrated_families)` is the rule the shadow comparison uses: an item is eligible when every model it
touches has all its COMBAT-hook families migrated. A model with no combat hook never blocks an item. Run hooks (the
run families F15 and F16, and hooks the game dispatches only outside combat) are not part of a fight's eligibility. An item whose relic list is unknown (`relics_known: false`) is not approved
unless the caller asks (`allow_unknown_relics=True`); the manifest reports both counts.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
FAMILIES = HERE / "hook_families.json"
MIGRATED = HERE / "migrated_families.json"
DISPATCH = HERE / "hook_dispatch.json"
REFEREE = Path(os.environ.get("SPIRELINE_REFEREE", r"D:\spireline\referee"))
OUT = Path(os.environ.get("SPIRELINE_REBUILD", str(REFEREE / "rebuild")))
AGENT_REAL = Path(os.environ.get("SPIRELINE_AGENT_REAL", str(REFEREE / "agent" / "gen0" / "real")))
LAB = Path(os.environ.get("SPIRELINE_LAB", str(REFEREE / "lab")))
CODEX = Path(os.environ.get("SPIRELINE_CODEX", r"D:\spireline\data\spire-codex\runs-v0.107.1.jsonl.gz"))
CODEX_READER = HERE / "manifest_codex.py"
FORMAT = 1
UNASSIGNED = "UNASSIGNED"
PILES = ("hand", "draw", "discard", "exhaust")
SETUP_KINDS = ("relic", "power", "card", "potion")        # the setup commands that name a model (cmd == kind)
CODEX_COUNTS = "codex-counts.json"


# ================================================================================================ families

def load_families(path: Path = FAMILIES) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def migrated_families(path: Path = MIGRATED) -> list:
    return list(json.loads(Path(path).read_text(encoding="utf-8")).get("migrated", []))


def snake(name: str) -> str:
    """A model class name as the game's id: StrikeIronclad -> STRIKE_IRONCLAD."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).upper()


def hook_is_run(hook: str, fam_doc: dict, iterators: dict | None = None) -> bool:
    """A run hook: in a run family, or one the game dispatches only outside combat."""
    fam = fam_doc["hooks"].get(hook)
    if fam is not None and fam_doc["families"][fam].get("run"):
        return True
    return (iterators or {}).get(hook) == ["run"]


def hook_family(hook: str, fam_doc: dict) -> str:
    return fam_doc["hooks"].get(hook, UNASSIGNED)


# ================================================================================================ the game's models

def catalog(decompiled: str, fam_doc: dict) -> dict:
    """Per model key ``kind:ID``: class identifier, combat and run hooks (inherited included), combat families."""
    from tools.fidelity import hook_table as ht

    iters = {h: e["iterator"] for h, e in ht.hook_dispatch(decompiled).items()}
    classes = ht.model_classes(decompiled)
    hooks_of: dict = collections.defaultdict(list)
    for o in ht.overrides(decompiled):
        hooks_of[o.model].append(o.hook)
    out: dict = {}
    for key, c in sorted(classes.items()):
        if c.abstract or c.name in ht.KIND_OF_BASE:
            continue
        kind = ht.kind_of(classes, c.base)
        if kind is None:
            continue
        hs = hooks_of.get(key, [])
        run = sorted(h for h in hs if hook_is_run(h, fam_doc, iters))
        combat = sorted(h for h in hs if h not in run)
        out[f"{kind}:{snake(c.name)}"] = {
            "kind": kind, "class": c.name, "combat_hooks": combat, "run_hooks": run,
            "families": sorted({hook_family(h, fam_doc) for h in combat}),
            "run_families": sorted({hook_family(h, fam_doc) for h in run})}
    return out


# ================================================================================================ record readers

def _id(x):
    """A card cell is [id, upgraded]; a relic or potion slot is an id or None."""
    if isinstance(x, (list, tuple)):
        x = x[0] if x else None
    return x if isinstance(x, str) and x else None


def corpus_models(rec: dict) -> tuple[set, bool]:
    """(model keys a recorded fight touches, whether its steps list relics). Cards created mid-fight appear in
    some step's piles, so the union over the steps covers them. The record's ``setup`` commands (relic, power, card
    and potion, by ``id``) name models too; they do not make the relic list known."""
    out: set = set()
    relics_known = False

    def add(kind, v):
        v = _id(v)
        if v:
            out.add(f"{kind}:{v}")

    for cmd in rec.get("setup") or []:
        if isinstance(cmd, dict) and cmd.get("cmd") in SETUP_KINDS:
            add(cmd["cmd"], cmd.get("id"))
    for step in rec.get("steps") or []:
        st = step.get("state") if isinstance(step, dict) else None
        if not isinstance(st, dict):
            continue
        for pile in PILES:
            for c in st.get(pile) or []:
                add("card", c)
        for e in st.get("enemies") or []:
            add("monster", e.get("name"))
            for p in e.get("powers") or []:
                add("power", p)
        for p in (st.get("player") or {}).get("powers") or []:
            add("power", p)
        if "relics" in st:
            relics_known = True
            for r in st.get("relics") or []:
                add("relic", r)
        for p in st.get("potions") or []:
            add("potion", p)
        a = step.get("action")
        if isinstance(a, dict):
            if a.get("kind") == "play":
                add("card", a.get("card"))
            elif a.get("kind") == "potion":
                add("potion", a.get("potion") or a.get("id") or a.get("name"))
    return out, relics_known


def starter_relics(records) -> dict:
    """Per character, the most common first relic of the fights that list their relics: the run's own starter."""
    seen: dict = collections.defaultdict(collections.Counter)
    for rec in records:
        for step in rec.get("steps") or []:
            st = step.get("state") if isinstance(step, dict) else None
            if isinstance(st, dict) and st.get("relics"):
                first = _id(st["relics"][0])
                if first:
                    seen[rec.get("character")][first] += 1
                break
    return {c: n.most_common(1)[0][0] for c, n in seen.items() if c}


def neow_relic(rec: dict):
    opts, pick = rec.get("neow_options"), rec.get("neow_pick")
    if isinstance(opts, list) and isinstance(pick, int) and 0 <= pick < len(opts):
        return _id(opts[pick])
    return None


def _diff_ids(vals) -> list:
    out = []
    for v in vals if isinstance(vals, list) else []:
        if isinstance(v, str) and v[:1] in ('[', '"'):     # a card cell as a JSON list, a relic or potion id JSON-quoted
            try:
                v = json.loads(v)
            except ValueError:
                pass
        v = _id(v)
        if v:
            out.append(v)
    return out


DIFF_KIND = (("hand", "card"), (".deck", "card"), ("relics", "relic"), ("potions", "potion"))


def agent_models(run: dict) -> set:
    """The models named in a run's diverging decisions' diffs: a lower bound of what the run touched."""
    out: set = set()
    for d in run.get("divergences") or []:
        for path, *sides in d.get("diff") or []:
            kind = next((k for frag, k in DIFF_KIND if frag in path), None)
            if kind:
                for s in sides:
                    out.update(f"{kind}:{i}" for i in _diff_ids(s))
    return out


def read_agent_runs(real: Path) -> list:
    f = Path(real) / "runs.jsonl"
    if not f.is_file():
        return []
    runs = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("ev") == "end":
            runs.append({"id": f"{r.get('seed')}/{r.get('agent_seed')}", "models": sorted(agent_models(r)),
                         "decisions": int((r.get("stats") or {}).get("decisions") or r.get("asked") or 0),
                         "partial": True})
    return runs


def _lab_group(files: list, spec_id) -> tuple[list, int]:
    """One run folder's recordings, in the given (sorted) order: ([(spec id, sorted models, relics known)] with each
    spec once, number of files read). A spec another folder holds too is dropped by the caller, in folder order."""
    items, seen, nfiles = [], set(), 0
    for f in files:
        try:
            doc = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        nfiles += 1
        for r in (doc.get("results") if isinstance(doc, dict) else doc) or []:
            if not isinstance(r, dict) or not r.get("steps") or r.get("error"):
                continue
            sid = spec_id(r)
            if sid in seen:
                continue
            seen.add(sid)
            models, known = corpus_models(r)
            items.append((sid, sorted(models), known))
    return items, nfiles


def _lab_task(args) -> tuple[list, int]:
    return _lab_group(*args)


def lab_groups(lab: Path) -> list:
    """The lab's recording files in sorted path order, grouped by run folder (a folder's files are contiguous in
    that order, so the groups in order concatenate to the sorted list)."""
    groups: list = []
    for f in sorted(glob.glob(str(Path(lab) / "runs" / "*" / "fights" / "*.out.json"))):
        folder = os.path.dirname(os.path.dirname(f))
        if groups and groups[-1][0] == folder:
            groups[-1][1].append(f)
        else:
            groups.append((folder, [f]))
    return [g[1] for g in groups]


def read_lab_fights(lab: Path, spec_id, procs: int = 1) -> tuple[list, int]:
    """(one item per setup of the lab recordings still on disk, number of files read). With ``procs`` > 1 the run
    folders are read in a spawn pool (``spec_id`` must then be a module-level function); the parent dedupes by spec
    id in folder order, so the result equals the serial one."""
    tasks = [(g, spec_id) for g in lab_groups(lab)]
    if procs > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
        from multiprocessing import get_context  # noqa: PLC0415

        from tools.fidelity.corpus import _below_normal  # noqa: PLC0415
        with ProcessPoolExecutor(max_workers=min(procs, len(tasks)), mp_context=get_context("spawn"),
                                 initializer=_below_normal) as ex:
            results = list(ex.map(_lab_task, tasks, chunksize=1))
    else:
        results = map(_lab_task, tasks)
    items, seen, files = [], set(), 0
    for group_items, n in results:
        files += n
        for sid, models, known in group_items:
            if sid not in seen:
                seen.add(sid)
                items.append({"id": sid, "models": models, "relics_known": known})
    return items, files


# ================================================================================================ eligibility and cover

def item_families(models, cat: dict) -> list:
    """The combat-hook families an item needs: the union over the models it touches (unknown models need none)."""
    need: set = set()
    for m in models:
        need.update((cat.get(m) or {}).get("families", ()))
    return sorted(need)


def eligible(item: dict, migrated, model_families: dict | None = None, allow_unknown_relics: bool = False) -> bool:
    """Every model the item touches has all its combat-hook families migrated.

    ``item`` carries ``families`` (the union of its models' combat families, as the manifest stores it), or
    ``models`` with ``model_families`` (model key -> its combat families) to work them out.

    An item with ``relics_known: false`` is never eligible: relics won in earlier rooms are missing from its model
    set and could carry a combat hook of a family not yet migrated. ``allow_unknown_relics=True`` judges the
    models it does list (the count the manifest reports as "eligible", before the relics condition).
    """
    if item.get("relics_known") is False and not allow_unknown_relics:
        return False
    if "families" in item:
        need = set(item["families"])
    else:
        need = set()
        for m in item.get("models", ()):
            need.update((model_families or {}).get(m, ()))
    return need <= set(migrated)


def greedy_cover(needs: list, limit: int = 100) -> list:
    """Order the things to port (models or families) by the items they unlock. ``needs``: per item, the set of
    things it needs ported. Each round picks the thing that completes the most items (ties: the larger share of
    the progress of items it is part of, then the more items, then the name); returns
    [{"thing", "unlocks", "cumulative"}] where cumulative counts the items that need nothing too.

    Incremental: per thing, a count of its live items by how many things each still needs; a round rescans only
    the things whose items changed. The share of progress is summed exactly (fractions)."""
    need = [set(n) for n in needs]
    remaining = {i: len(n) for i, n in enumerate(need) if n}
    free = sum(1 for n in need if not n)
    by: dict = collections.defaultdict(set)
    for i in remaining:
        for t in need[i]:
            by[t].add(i)
    by = dict(by)
    live: dict = {t: collections.Counter(remaining[i] for i in idxs) for t, idxs in by.items()}
    keys: dict = {}

    def key_of(t):
        c = live[t]
        return (c.get(1, 0), sum(Fraction(n, r) for r, n in c.items()), sum(c.values()))

    dirty = set(by)
    done, order = free, []
    while by and len(order) < limit:
        for t in dirty:
            keys[t] = key_of(t)
        dirty.clear()
        best, best_key = None, None
        for t in sorted(by):
            if best_key is None or keys[t] > best_key:
                best, best_key = t, keys[t]
        if best_key[2] == 0:
            break
        idxs = by.pop(best)
        del live[best]
        newly = 0
        for i in idxs:
            r = remaining[i]
            if r > 0:
                remaining[i] = r - 1
                for t in need[i]:
                    if t in by:
                        live[t][r] -= 1
                        if live[t][r] == 0:
                            del live[t][r]
                        if r > 1:
                            live[t][r - 1] += 1
                        dirty.add(t)
                if r == 1:
                    newly += 1
        done += newly
        order.append({"thing": best, "unlocks": newly, "cumulative": done})
    return order


class Stages:
    """Wall time per build stage, one line each on stderr."""

    def __init__(self, enabled: bool = True):
        self.enabled, self.t = enabled, time.perf_counter()
        self.times: dict = {}

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.times[name] = round(now - self.t, 2)
        if self.enabled:
            print(f"manifest: {name:<8} {now - self.t:8.1f} s", file=sys.stderr, flush=True)
        self.t = now


def share(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


# ================================================================================================ the build

def codex_counts(out: Path, rescan: bool) -> dict | None:
    """The Codex aggregate (counts per model), from the cache in the out folder or a child `python -I` run."""
    cache = Path(out) / CODEX_COUNTS
    if rescan or not cache.is_file():
        if not CODEX.is_file():
            return None
        Path(out).mkdir(parents=True, exist_ok=True)
        from tools.fidelity.corpus import BELOW_NORMAL  # noqa: PLC0415  (the helper every replay child uses)
        subprocess.run([sys.executable, "-I", str(CODEX_READER), str(CODEX), str(cache)], check=True,
                       creationflags=BELOW_NORMAL)
    doc = json.loads(cache.read_text(encoding="utf-8"))
    return {"runs": int(doc.get("runs", 0)), "by_model": {k: int(v) for k, v in doc.get("by_model", {}).items()}}


def assemble(fam_doc: dict, cat: dict, corpus_items: list, agent_items: list, lab_items: list, codex: dict | None,
             migrated: list, notes: list | None = None, stages: Stages | None = None) -> dict:
    """The manifest document from already-read inputs (the part the tests exercise)."""
    stages = stages or Stages(False)
    for it in corpus_items + agent_items + lab_items:
        it["families"] = item_families(it["models"], cat)
    models = {k: dict(v, corpus_fights=0, agent_runs=0, lab_fights=0, codex_runs=0) for k, v in cat.items()}
    unresolved = collections.defaultdict(collections.Counter)

    def touch(source, key, n=1):
        if key in models:
            models[key][source] += n
        else:
            kind = key.split(":", 1)[0]
            unresolved[source][kind] += 1
            models.setdefault(key, {"kind": kind, "class": None, "combat_hooks": [], "run_hooks": [], "families": [],
                                    "run_families": [], "unresolved": True, "corpus_fights": 0, "agent_runs": 0,
                                    "lab_fights": 0, "codex_runs": 0})[source] += n

    for src, items in (("corpus_fights", corpus_items), ("agent_runs", agent_items), ("lab_fights", lab_items)):
        for it in items:
            for m in it["models"]:
                touch(src, m)
    for key, n in (codex or {}).get("by_model", {}).items():
        touch("codex_runs", key, n)

    reached = {src: {k for k, v in models.items() if v[src]} for src in
               ("corpus_fights", "agent_runs", "lab_fights", "codex_runs")}
    reached_any = set().union(*reached.values())
    kinds = {}
    for kind in sorted({v["kind"] for v in models.values()}):
        ks = [k for k, v in models.items() if v["kind"] == kind]
        kinds[kind] = {"models": len(ks), "with_combat_hooks": sum(1 for k in ks if models[k]["combat_hooks"]),
                       "with_run_hooks": sum(1 for k in ks if models[k]["run_hooks"]),
                       "reached_by_corpus": sum(1 for k in ks if k in reached["corpus_fights"]),
                       "reached_by_agent": sum(1 for k in ks if k in reached["agent_runs"]),
                       "reached_by_lab": sum(1 for k in ks if k in reached["lab_fights"]),
                       "reached_by_codex": sum(1 for k in ks if k in reached["codex_runs"]),
                       "reached_by_any": sum(1 for k in ks if k in reached_any),
                       "unresolved": sum(1 for k in ks if models[k].get("unresolved"))}

    n_fights = len(corpus_items)
    n_dec = sum(a["decisions"] for a in agent_items)
    fam_stats = {}
    for fid, meta in fam_doc["families"].items():
        in_fam = [k for k, v in models.items() if fid in v["families"]]
        fights = sum(1 for it in corpus_items if fid in it["families"])
        runs = [a for a in agent_items if fid in a["families"]]
        fam_stats[fid] = {
            "name": meta["name"], "run": bool(meta.get("run")), "hooks": sum(1 for f in fam_doc["hooks"].values() if f == fid),
            "models_overriding": len(in_fam), "models_reached": sum(1 for k in in_fam if k in reached_any),
            "corpus_fights": fights, "corpus_fight_share_pct": share(fights, n_fights),
            "agent_runs": len(runs), "agent_decision_share_pct": share(sum(a["decisions"] for a in runs), n_dec),
            "alone_unlocks_fights": sum(1 for it in corpus_items if set(it["families"]) <= {fid})}
    stages.mark("join")
    greedy_models = greedy_cover([[m for m in it["models"] if models.get(m, {}).get("combat_hooks")]
                                  for it in corpus_items], 100)
    for g in greedy_models:
        g["model"] = g.pop("thing")
        g["kind"], g["families"] = models[g["model"]]["kind"], models[g["model"]]["families"]
    greedy_fams = [{"family": g["thing"], "unlocks": g["unlocks"], "cumulative": g["cumulative"],
                    "cumulative_share_pct": share(g["cumulative"], n_fights)}
                   for g in greedy_cover([it["families"] for it in corpus_items], 20)]
    stages.mark("cover")
    eligible_now = {"migrated": sorted(migrated)}
    for src, items in (("corpus_fights", corpus_items), ("agent_runs", agent_items), ("lab_fights", lab_items)):
        eligible_now[src] = sum(1 for it in items if eligible(it, migrated, allow_unknown_relics=True))
        eligible_now[f"{src}_known_relics"] = sum(1 for it in items if eligible(it, migrated))
    return {
        "format": FORMAT, "built": datetime.date.today().isoformat(),
        "counts": {"models": len(models), "models_reached_by_any": len(reached_any),
                   "corpus_fights": n_fights, "corpus_fights_relics_unknown": sum(1 for it in corpus_items if not it.get("relics_known")),
                   "agent_runs": len(agent_items), "agent_decisions": n_dec, "lab_fights": len(lab_items),
                   "codex_runs": (codex or {}).get("runs"),
                   "hooks_mapped": len(fam_doc["hooks"]), "hooks_unassigned": len(fam_doc.get("unassigned", {}))},
        "unassigned_hooks": fam_doc.get("unassigned", {}),
        "kinds": kinds, "families": fam_stats,
        "unresolved_ids": {src: dict(c) for src, c in unresolved.items()},
        "eligible_now": eligible_now,
        "greedy_models": greedy_models, "greedy_families": greedy_fams,
        "models": models, "corpus_fights": corpus_items, "agent_runs": agent_items,
        "lab_fights": [{"id": it["id"], "families": it["families"], "models": it["models"],
                        "relics_known": it.get("relics_known")} for it in lab_items],
        "notes": notes or []}


def render_md(m: dict) -> str:
    c = m["counts"]
    L = ["# Port manifest", "", f"Built {m['built']}. Models are `kind:ID`; ids are the game's.", ""]
    L += ["## Counts", ""] + [f"- {k}: {v}" for k, v in c.items()] + [""]
    L += ["## Per kind", "", "| kind | models | combat hooks | run hooks | corpus | agent | lab | codex | any | unresolved |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in m["kinds"].items():
        L.append(f"| {k} | {v['models']} | {v['with_combat_hooks']} | {v['with_run_hooks']} | {v['reached_by_corpus']} | "
                 f"{v['reached_by_agent']} | {v['reached_by_lab']} | {v['reached_by_codex']} | {v['reached_by_any']} | {v['unresolved']} |")
    L += ["", "## Per family", "",
          "| family | name | hooks | models overriding | reached | corpus fights | share % | agent decisions % | alone unlocks |",
          "|---|---|---|---|---|---|---|---|---|"]
    for f, v in m["families"].items():
        L.append(f"| {f} | {v['name']}{' (run)' if v['run'] else ''} | {v['hooks']} | {v['models_overriding']} | {v['models_reached']} | "
                 f"{v['corpus_fights']} | {v['corpus_fight_share_pct']} | {v['agent_decision_share_pct']} | {v['alone_unlocks_fights']} |")
    L += ["", "Share = corpus fights (agent decisions, over the runs' lower-bound model sets) that touch a model overriding a "
          "COMBAT hook of the family. Run families F15/F16 never gate a fight.", "",
          "## Family cover (greedy: the family that completes the most fights first)", "",
          "| step | family | unlocks | cumulative fights | share % |", "|---|---|---|---|---|"]
    for i, g in enumerate(m["greedy_families"], 1):
        L.append(f"| {i} | {g['family']} | {g['unlocks']} | {g['cumulative']} | {g['cumulative_share_pct']} |")
    L += ["", "## Unassigned hooks", ""] + [f"- {h}: {w}" for h, w in m["unassigned_hooks"].items()] + [""]
    L += ["## Eligible now", "", f"Migrated families: {m['eligible_now']['migrated'] or 'none'}. "
          f"Eligible by the models listed: corpus fights {m['eligible_now']['corpus_fights']}, agent runs "
          f"{m['eligible_now']['agent_runs']}, lab fights {m['eligible_now']['lab_fights']}. Of those, with the relic list "
          f"known (the only ones `eligible` approves): corpus fights {m['eligible_now']['corpus_fights_known_relics']}, "
          f"agent runs {m['eligible_now']['agent_runs_known_relics']} (runs carry no `relics_known`; their model sets are "
          f"partial), lab fights {m['eligible_now']['lab_fights_known_relics']}.", "",
          "## Greedy model cover (models with combat hooks, by corpus fights unlocked)", "",
          "| rank | model | families | unlocks | cumulative fights |", "|---|---|---|---|---|"]
    for i, g in enumerate(m["greedy_models"][:40], 1):
        L.append(f"| {i} | {g['model']} | {','.join(g['families'])} | {g['unlocks']} | {g['cumulative']} |")
    L += ["", f"The full order (first {len(m['greedy_models'])}) is in manifest.json.", ""]
    if m["unresolved_ids"]:
        L += ["## Ids the decompile did not resolve (kept as models with no hooks)", ""]
        L += [f"- {s}: {d}" for s, d in m["unresolved_ids"].items()] + [""]
    L += ["## Notes", ""] + [f"- {n}" for n in m["notes"]] + [""]
    return "\n".join(L)


def cmd_build(a) -> int:
    from tools.fidelity import corpus as cp
    from tools.fidelity import hook_table as ht

    out = Path(a.out)
    stages = Stages()
    fam_doc = load_families()
    cat = catalog(ht.DEFAULT_DECOMPILED, fam_doc)
    stages.mark("catalog")
    corpus = cp.Corpus(cp.DEFAULT_CORPUS)
    idx = corpus.load_index()
    recs = [(m, corpus.read(m["id"])) for m in idx["fights"]]
    starters = starter_relics(r for _, r in recs)
    notes = []
    corpus_items = []
    for meta, rec in recs:
        models, known = corpus_models(rec)
        if not known:
            s, n = starters.get(rec.get("character")), neow_relic(rec)
            models.update(f"relic:{x}" for x in (s, n) if x)
        corpus_items.append({"id": meta["id"], "models": sorted(models), "relics_known": known})
    stages.mark("corpus")
    unknown = sum(1 for it in corpus_items if not it["relics_known"])
    notes.append(f"{unknown} of {len(corpus_items)} corpus fights do not list relics in their steps; their model set holds the "
                 "character's starter relic and the Neow pick when it is a relic, so relics won in earlier rooms are missing "
                 "(`relics_known` is false on them).")
    agent_items = read_agent_runs(AGENT_REAL)
    stages.mark("agent")
    notes.append("Agent runs: the lockstep records keep only the entities of diverging decisions, so a run's model set "
                 "is a lower bound (every agent run is `partial`); decision shares use the runs' recorded decision counts.")
    lab_items, files = read_lab_fights(LAB, cp.spec_id, a.procs)
    stages.mark("lab")
    notes.append(f"Lab fights: {len(lab_items)} setups from {files} recording files still on disk under the lab's runs "
                 "(the lab prunes a recording once the corpus holds it).")
    codex = None if a.no_codex else codex_counts(out, a.codex_rescan)
    stages.mark("codex")
    notes.append("Codex: counts per model only, over the runs the Codex walk replays (Ironclad, ascension 10); "
                 + ("not read in this build." if codex is None else f"{codex['runs']} runs matched.")
                 + " The walk's own outputs hold no model lists, so the dataset is read directly.")
    notes.append("Encounters, enchantments and afflictions are not recorded in the corpus steps (cards are [id, upgraded]); "
                 "monsters stand for the encounter. Models without any hook override never block an item, so card, "
                 "monster and power effects themselves are the per-model ports of rebuild-plan phase P2, not part of eligibility.")
    m = assemble(fam_doc, cat, corpus_items, agent_items, lab_items, codex, migrated_families(), notes, stages)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(m, indent=0), encoding="utf-8")
    (out / "manifest.md").write_text(render_md(m), encoding="utf-8")
    stages.mark("write")
    print(f"manifest: {m['counts']['models']} models ({m['counts']['models_reached_by_any']} reached), "
          f"{m['counts']['corpus_fights']} corpus fights, {m['counts']['agent_runs']} agent runs, "
          f"{m['counts']['lab_fights']} lab fights, codex {m['counts']['codex_runs']}")
    return 0


def cmd_status(a) -> int:
    f = Path(a.out) / "manifest.json"
    if not f.is_file():
        print("no manifest yet: run `manifest.py build`")
        return 1
    m = json.loads(f.read_text(encoding="utf-8"))
    migrated = migrated_families()
    n = len(m["corpus_fights"])
    e = sum(1 for it in m["corpus_fights"] if eligible(it, migrated, allow_unknown_relics=True))
    ek = sum(1 for it in m["corpus_fights"] if eligible(it, migrated))
    ea = sum(1 for it in m["agent_runs"] if eligible(it, migrated, allow_unknown_relics=True))
    eak = sum(1 for it in m["agent_runs"] if eligible(it, migrated))
    print(f"manifest built {m['built']}: {m['counts']['models']} models, {m['counts']['models_reached_by_any']} reached")
    print(f"migrated families: {migrated or 'none'}")
    print(f"eligible corpus fights {e}/{n} ({share(e, n)}%), agent runs {ea}/{len(m['agent_runs'])}")
    print(f"eligible with known relics: corpus fights {ek}/{n} ({share(ek, n)}%), agent runs {eak}/{len(m['agent_runs'])}")
    for fid, v in m["families"].items():
        print(f"  {fid} {v['name']:<28} fights {v['corpus_fight_share_pct']:>6}%  decisions {v['agent_decision_share_pct']:>6}%"
              f"{'  migrated' if fid in migrated else ''}")
    if m["unassigned_hooks"]:
        print("unassigned hooks:", ", ".join(m["unassigned_hooks"]))
    return 0


def cmd_codex(a) -> int:
    d = codex_counts(Path(a.out), True)
    print("codex: not found" if d is None else f"codex: {d['runs']} runs, {len(d['by_model'])} models")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("build", cmd_build), ("status", cmd_status), ("codex", cmd_codex)):
        p = sub.add_parser(name)
        p.add_argument("--out", default=str(OUT))
        if name == "build":
            p.add_argument("--no-codex", action="store_true")
            p.add_argument("--codex-rescan", action="store_true")
            p.add_argument("--procs", type=int, default=12, help="processes reading the lab's run folders")
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
