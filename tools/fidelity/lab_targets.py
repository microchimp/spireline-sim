#!/usr/bin/env python3
"""Fix lab targets: what aimed setup fights should reach, weighted by how often real runs meet it and how little we know.

    python tools/fidelity/lab_targets.py --lab LAB --out LAB/targets.json [--backlog <newest backlog-*.json>]
        [--codex <runs-v0.107.1.jsonl.gz>] [--corpus DIR] [--refset-score FILE] [--reach FILE] [--codex-walk DIR]
        [--codex-max-runs N] [--rebuild-profile]

Writes {main_sha, generated_utc, profile, totals, targets: [{kind, id, weight, reasons}]}, highest weight first,
kind one of card, relic, potion, enemy, encounter, event, power, act. No game, no emulator, no network.

weight = meet x gap. meet is the share of real Ironclad A10 runs (Spire Codex, v0.107.1) that meet the id (a card
or relic held, a potion picked, an encounter, enemy or event met, an act reached; a power through the power cards
that grant it or the enemies seen holding it). gap multiplies one factor per reason, each written in `reasons`:

  uncovered  no recorded fight (the corpus: a card played, an enemy or power on the board, a set-up item), setup
             fight (jobs-setup coverage, the lab's own batches) or reference-set fight reaches it: x3; fewer than 3: x1.5
  reach      engine blocks naming it that no recorded fight runs (tools/fidelity/mutate.py prepare's coverage,
             the newest prepared.json under LAB/runs, or named by a job.json "outputs": {"reach": ...}, or --reach): x1.5
  refset     it sits in a low-accuracy cell of the newest reference-set score (act x room; for a held item the
             acts it is held in): x(2 - identical share)
  backlog    an open item of the newest backlog names it (title or evidence): x(1 + 0.5 per item, at most 3)
  walk       the newest codex walk stops at it (its expected or emulator value): x1.5

The Codex history is third-party data. It is read only by this script's profile step, in an isolated interpreter
(python -I, started by this script when it is not one already), and only aggregate counts leave it: per act the
fights, rooms, encounters, and for one snapshot per run and act (its middle fight, referee/refset.py's
deck_before/relics_before/potions_before) how many held each card, relic and potion, deck and belt sizes, and
how often two held items met (pairs seen at least 3 times). The profile is LAB/cache/codex_profile.json, outside
every repo; scenarios.py --mode targets draws realistic setups from it. No per-run value is written.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import datetime
import gzip
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "referee"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

REFEREE = Path(os.environ.get("SPIRELINE_REFEREE", r"D:\spireline\referee"))
CODEX = Path(r"D:\spireline\data\spire-codex\runs-v0.107.1.jsonl.gz")
EXPORT = REFEREE / "exports" / "v0.107.1.json"
KINDS = ("card", "relic", "potion", "enemy", "encounter", "event", "power", "act")
ACT_NAMES = ("OVERGROWTH", "UNDERDOCKS", "HIVE", "GLORY")
ACT_NUMBER = {"OVERGROWTH": 1, "UNDERDOCKS": 1, "HIVE": 2, "GLORY": 3}
ROOMS = ("monster", "elite", "boss")
BASIC_CARDS = ("STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH", "ASCENDERS_BANE")
STARTER_RELICS = ("BURNING_BLOOD",)
PAIR_MIN = 3
PROFILE_VERSION = 1


def bare(x) -> str:
    s = str(x or "")
    return s.split(".", 1)[1] if "." in s else s


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).upper()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def write_json(path: Path, doc, indent: int | None = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=indent, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def git_sha(tree: Path = ROOT) -> str:
    r = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return r.stdout.strip()


def rel_to(path: Path, base: Path) -> str:
    """A path without the machine's folders: relative to base, else the referee folder, else its name."""
    p = Path(path).resolve()
    for b, tag in ((base, ""), (REFEREE, "referee/"), (ROOT, "fork/")):
        with contextlib.suppress(ValueError, OSError):
            return tag + p.relative_to(Path(b).resolve()).as_posix()
    return p.name


# ================================================================== the Codex profile (aggregate counts only)

def _act_doc() -> dict:
    return {"fights": 0, "rooms": collections.Counter(), "encounters": {r: collections.Counter() for r in ROOMS},
            "act_names": collections.Counter(), "snapshots": 0, "cards": collections.Counter(),
            "copies": collections.Counter(), "upgraded": collections.Counter(), "relics": collections.Counter(),
            "potions": collections.Counter(), "deck_size": collections.Counter(), "relic_count": collections.Counter(),
            "potion_count": collections.Counter(), "basics": {c: collections.Counter() for c in BASIC_CARDS},
            "pairs": collections.Counter()}


def snapshot_items(deck: list, relics: list, potions: list) -> list[str]:
    """The held items one snapshot counts pairs over: non-basic cards, non-starter relics, potions."""
    items = {"card:" + c[0] for c in deck if c[0] not in BASIC_CARDS}
    items |= {"relic:" + r for r in relics if r not in STARTER_RELICS}
    items |= {"potion:" + p for p in potions}
    return sorted(items)


def build_profile(codex: Path, max_runs: int = 0) -> dict:
    """Aggregate counts over the Ironclad A10 runs (referee/refset.py's filter). Nothing per run is kept."""
    import refset  # noqa: PLC0415  (referee/refset.py: the run filter and the holdings before a floor)

    stats: collections.Counter = collections.Counter()
    meet = {k: collections.Counter() for k in ("card", "relic", "potion", "encounter", "enemy", "event", "act")}
    acts = {a: _act_doc() for a in (1, 2, 3)}
    enc_monsters: dict = collections.defaultdict(collections.Counter)
    enc_room: dict = collections.defaultdict(collections.Counter)
    runs = 0
    for d in refset.codex_runs(codex, stats, max_runs):
        runs += 1
        p = d["players"][0]
        seen = {k: set() for k in meet}
        seen["card"] |= {bare(c.get("id")) for c in p.get("deck", [])}
        seen["relic"] |= {bare(r.get("id")) for r in p.get("relics", [])}
        seen["potion"] |= {bare(x.get("id")) for x in p.get("potions", []) if isinstance(x, dict)}
        history = d.get("map_point_history") or []
        seen["act"] |= {bare(a) for a in (d.get("acts") or [])[:len(history)]}
        pts, act_of = [], []
        for ai, act in enumerate(history):
            for pt in act:
                pts.append(pt)
                act_of.append(ai)
        for ai, name in enumerate((d.get("acts") or [])[:min(3, len(history))]):
            acts[ai + 1]["act_names"][bare(name)] += 1
        fights_by_act = collections.defaultdict(list)
        for f, pt in enumerate(pts, 1):
            st = (pt.get("player_stats") or [{}])[0]
            seen["card"] |= {bare(c.get("id")) for c in st.get("cards_gained", []) if isinstance(c, dict)}
            seen["card"] |= {bare(c.get("id")) for c in st.get("cards_removed", []) if isinstance(c, dict)}
            seen["relic"] |= {bare(rc.get("choice")) for rc in st.get("relic_choices", []) if rc.get("was_picked")}
            seen["relic"] |= {bare(b) for b in st.get("bought_relics", [])}
            seen["relic"] |= {bare(b) for b in st.get("relics_removed", [])}
            seen["potion"] |= {bare(pc.get("choice")) for pc in st.get("potion_choices", []) if pc.get("was_picked")}
            seen["potion"] |= {bare(b) for b in st.get("bought_potions", [])}
            for room in pt.get("rooms") or []:
                rt, mid = room.get("room_type"), bare(room.get("model_id"))
                if rt in ROOMS and mid:
                    mons = [bare(m) for m in room.get("monster_ids") or []]
                    seen["encounter"].add(mid)
                    seen["enemy"] |= set(mons)
                    enc_monsters[mid].update(set(mons))
                    enc_room[mid][rt] += 1
                    a = act_of[f - 1] + 1
                    if a <= 3:
                        acts[a]["fights"] += 1
                        acts[a]["rooms"][rt] += 1
                        acts[a]["encounters"][rt][mid] += 1
                        fights_by_act[a].append(f)
                elif rt == "event" and mid:
                    seen["event"].add(mid)
        final_potions = p.get("potions", [])
        for a, floors in fights_by_act.items():
            f = floors[len(floors) // 2]
            if f < 2:
                continue
            deck, _ = refset.deck_before(d, pts, f)
            relics = refset.relics_before(d, pts, f) or []
            potions, _ = refset.potions_before(pts, f, final_potions)
            doc = acts[a]
            doc["snapshots"] += 1
            ids = collections.Counter(c[0] for c in deck)
            ups = collections.Counter(c[0] for c in deck if c[1] > 0)
            doc["cards"].update(set(ids))
            doc["copies"].update(ids)
            doc["upgraded"].update(ups)
            doc["relics"].update(set(relics))
            doc["potions"].update(set(potions))
            doc["deck_size"][len(deck)] += 1
            doc["relic_count"][len([r for r in relics if r not in STARTER_RELICS])] += 1
            doc["potion_count"][len(potions)] += 1
            for b in BASIC_CARDS:
                doc["basics"][b][ids.get(b, 0)] += 1
            items = snapshot_items(deck, relics, potions)
            doc["pairs"].update(f"{x}|{y}" for x, y in itertools.combinations(items, 2))
        for k, s in seen.items():
            meet[k].update(s - {""})
    out_acts = {}
    for a, doc in acts.items():
        o = {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in doc.items()
             if k not in ("encounters", "basics", "pairs")}
        o["encounters"] = {r: dict(c) for r, c in doc["encounters"].items()}
        o["basics"] = {b: {str(k): v for k, v in c.items()} for b, c in doc["basics"].items()}
        for k in ("deck_size", "relic_count", "potion_count"):
            o[k] = {str(n): v for n, v in doc[k].items()}
        o["pairs"] = {k: v for k, v in doc["pairs"].items() if v >= PAIR_MIN}
        out_acts[str(a)] = o
    return {"version": PROFILE_VERSION, "runs": runs, "stats": {k: v for k, v in stats.items() if not k.startswith("cand")},
            "meet": {k: dict(c) for k, c in meet.items()}, "acts": out_acts,
            "encounter_monsters": {e: sorted(c) for e, c in enc_monsters.items()},
            "encounter_rooms": {e: c.most_common(1)[0][0] for e, c in enc_room.items()}}


def source_stamp(codex: Path, max_runs: int) -> dict:
    st = codex.stat()
    return {"name": codex.name, "size": st.st_size, "mtime": int(st.st_mtime), "max_runs": max_runs}


def profile_path(lab: Path) -> Path:
    return Path(lab) / "cache" / "codex_profile.json"


def ensure_profile(lab: Path, codex: Path, max_runs: int = 0, rebuild: bool = False, log=print) -> dict | None:
    """The cached profile, rebuilt in an isolated interpreter when it is missing or the dataset changed."""
    path = profile_path(lab)
    prof = read_json(path)
    if not codex.is_file():
        if prof is None:
            log(f"no Codex dataset at {codex.name} and no cached profile: meet counts are all zero")
        return prof
    stamp = source_stamp(codex, max_runs)
    if prof and not rebuild and prof.get("source") == stamp and prof.get("version") == PROFILE_VERSION:
        return prof
    log(f"building the Codex profile from {codex.name} (aggregate counts; isolated interpreter)")
    cmd = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--profile-only", "--lab", str(lab), "--codex", str(codex),
           "--codex-max-runs", str(max_runs)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False,
                       creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
    if r.returncode != 0:
        log("profile build failed: " + (r.stderr or r.stdout)[-600:])
        return prof
    return read_json(path)


def write_profile(lab: Path, codex: Path, max_runs: int) -> int:
    if not sys.flags.isolated:
        print("the profile step reads third-party data and must run under python -I", file=sys.stderr)
        return 2
    prof = build_profile(codex, max_runs)
    prof["source"] = source_stamp(codex, max_runs)
    prof["generated_utc"] = utc_now()
    write_json(profile_path(lab), prof, indent=None)
    print(f"profile: {prof['runs']} Ironclad A10 runs")
    return 0


class Profile:
    """Read side of the profile: shares and conditional draws for realistic setups."""

    def __init__(self, doc: dict | None):
        self.doc = doc or {"runs": 0, "meet": {}, "acts": {}}
        self.runs = max(1, int(self.doc.get("runs") or 0))
        self.acts = self.doc.get("acts", {})

    def meet(self, kind: str, ident: str) -> int:
        return int(self.doc.get("meet", {}).get(kind, {}).get(ident, 0))

    def act(self, a: int) -> dict:
        return self.acts.get(str(a), {})

    def held(self, kind: str, ident: str, a: int) -> int:
        table = {"card": "cards", "relic": "relics", "potion": "potions"}[kind]
        return int(self.act(a).get(table, {}).get(ident, 0))

    def held_acts(self, kind: str, ident: str) -> dict:
        """Act -> share of that act's snapshots holding the item."""
        out = {}
        for a in (1, 2, 3):
            n = max(1, int(self.act(a).get("snapshots", 0)))
            out[a] = self.held(kind, ident, a) / n
        return out

    def pair(self, a: int, x: str, y: str) -> int:
        k = f"{x}|{y}" if x < y else f"{y}|{x}"
        return int(self.act(a).get("pairs", {}).get(k, 0))

    def encounter_monsters(self, enc: str) -> list:
        return self.doc.get("encounter_monsters", {}).get(enc, [])


# ================================================================== what is already covered

def corpus_entities(corpus: Path, cache: Path, log=print) -> dict:
    """{fight id: {card: [...played], enemy, power, relic, potion}} for every corpus fight, cached by id."""
    index = read_json(corpus / "index.json", {}) or {}
    have = read_json(cache, {}) or {}
    fresh = 0
    for f in index.get("fights", []):
        fid = f.get("id")
        if not fid or fid in have:
            continue
        rec = None
        with contextlib.suppress(OSError, ValueError, EOFError):
            with gzip.open(corpus / "fights" / f"{fid}.json.gz", "rt", encoding="utf-8") as fh:
                rec = json.load(fh)
        if rec is None:
            continue
        ent = {k: set() for k in ("card", "enemy", "power", "relic", "potion", "encounter")}
        for it in rec.get("setup") or []:
            cmd, ident = it.get("cmd"), str(it.get("id") or "")
            if cmd in ("relic", "potion"):
                ent[cmd].add(ident)
            elif cmd == "power":
                ent["power"].add(ident)
            elif cmd == "fight":
                ent["encounter"].add(ident)
        if (rec.get("fight") or {}).get("id"):
            ent["encounter"].add(rec["fight"]["id"])
        for s in rec.get("steps", []):
            a = s.get("action") or {}
            if a.get("kind") == "play" and a.get("card"):
                ent["card"].add(str(a["card"]))
            elif a.get("kind") == "potion" and a.get("potion"):
                ent["potion"].add(str(a["potion"]))
            st = s.get("state") or {}
            for e in st.get("enemies", []):
                ent["enemy"].add(str(e.get("name")))
                ent["power"] |= {str(p[0]) for p in e.get("powers", []) if p}
            ent["power"] |= {str(p[0]) for p in (st.get("player") or {}).get("powers", []) if p}
            ent["relic"] |= {str(r if isinstance(r, str) else (r or {}).get("id")) for r in st.get("relics", []) or []}
        have[fid] = {k: sorted(v - {"None", ""}) for k, v in ent.items() if v}
        fresh += 1
    if fresh:
        write_json(cache, have, indent=None)
        log(f"corpus entities: {fresh} new fights read, {len(have)} cached")
    return have


def coverage_counts(lab: Path, corpus: Path | None, setup_cov: list[Path], refset_score: dict | None, log=print) -> dict:
    """{"kind:ID": fights reaching it} over the corpus, setup-fight coverage files and the lab's batches."""
    cov: collections.Counter = collections.Counter()
    if corpus and (corpus / "index.json").is_file():
        for ent in corpus_entities(corpus, Path(lab) / "cache" / "corpus_entities.json", log).values():
            for kind, ids in ent.items():
                cov.update(f"{kind}:{i}" for i in ids)
    for path in setup_cov:
        for key, c in ((read_json(path, {}) or {}).get("items") or {}).items():
            kind, _, ident = key.partition(":")
            kind = {"card+": "card"}.get(kind, kind)
            ident = ident.split("@")[0]
            cov[f"{kind}:{ident}"] += int(c.get("seen", 0))
    for vpath in sorted(Path(lab, "runs").glob("*/**/*.verdicts.json")) if Path(lab, "runs").is_dir() else []:
        spath = vpath.with_name(vpath.name.removesuffix(".verdicts.json") + ".json")
        for spec in (read_json(spath, {}) or {}).get("runs", []):
            for it in spec.get("setup", []):
                kind = {"fight": "encounter"}.get(it.get("cmd"), it.get("cmd"))
                if kind in ("card", "relic", "potion", "encounter", "power"):
                    cov[f"{kind}:{it.get('id')}"] += 1
    if refset_score:
        not_reached = {f"{k}:{i.rstrip('+')}" for k, ids in ((refset_score.get("coverage") or {}).get("not_reached") or {}).items()
                       for i in ids}
        for v in refset_score.get("verdicts", []) or []:
            for c, _ in v.get("plays", []) or []:
                if f"card:{c}" not in not_reached:
                    cov[f"card:{str(c).rstrip('+')}"] += 1
    return dict(cov)


# ================================================================== where we know least

def refset_cells(score: dict | None) -> dict:
    """(act, room) -> identical share of the kept fights, from the newest score's realistic table."""
    out = {}
    for cell, v in ((score or {}).get("realistic_table") or {}).items():
        m = re.match(r"act(\d) (monster|elite|boss)$", cell)
        if m and v.get("kept"):
            out[(int(m.group(1)), m.group(2))] = (v.get("identical_pct_of_kept") or 0) / 100.0
    return out


def newest(paths) -> Path | None:
    paths = [p for p in paths if p.is_file()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def newest_refset_score(referee: Path) -> Path | None:
    scores = referee / "refset" / "v2" / "scores"
    return newest(p for p in scores.glob("all-*.json")) if scores.is_dir() else None


CASE = re.compile(r"\bcase\s+(?:[A-Za-z_][\w.]*\.)?([A-Z][A-Za-z0-9]+)\b")


def reach_entities(prepared: dict | None, engine: Path, known: dict) -> dict:
    """{"kind:ID": engine blocks naming it that no recorded fight ran}: a mutation candidate whose block ran in no
    fight (fights == 0), tied to the nearest `case X` label above it in its file, X an id_map name."""
    out: collections.Counter = collections.Counter()
    if not prepared:
        return {}
    by_file = collections.defaultdict(set)
    for c in prepared.get("candidates", []):
        if c.get("fights") == 0 and c.get("file") and c.get("line"):
            by_file[c["file"]].add(int(c["line"]))
    for rel, lines in by_file.items():
        try:
            text = (engine / rel).read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        label_at: list = [None] * (len(text) + 1)
        cur = None
        for i, line in enumerate(text, 1):
            m = CASE.search(line)
            if m:
                cur = m.group(1)
            label_at[i] = cur
        blocks = set()
        for ln in lines:
            name = label_at[ln] if ln < len(label_at) else None
            key = known.get(name or "")
            if key:
                blocks.add((key, ln))
        out.update(k for k, _ in blocks)
    return dict(out)


def known_names(id_map: dict) -> dict:
    """PascalName -> "kind:ID" for the id_map's cards, relics, potions, enemies and powers."""
    out = {}
    for table, kind in (("cards", "card"), ("relics", "relic"), ("potions", "potion"), ("enemies", "enemy"), ("powers", "power")):
        for pas in id_map.get(table, {}):
            out.setdefault(pas, f"{kind}:{snake(pas)}")
            if kind == "power" and pas.endswith("Power"):
                out.setdefault(pas[:-5], f"power:{snake(pas)}")
    return out


def backlog_mentions(backlog: dict | None, id_map: dict) -> dict:
    """{"kind:ID": [open item ids whose title or evidence name it]}."""
    if not backlog:
        return {}
    try:
        from tools.fidelity.packet import Names, entities_in  # noqa: PLC0415
    except Exception:  # noqa: BLE001  (packet.py is optional here: fall back to game ids alone)
        Names = entities_in = None  # noqa: N806
    names = Names(id_map) if Names else None
    out = collections.defaultdict(list)
    for it in backlog.get("items", []):
        if it.get("status", "open") != "open":
            continue
        text = f"{it.get('title', '')} {it.get('evidence', '')}"
        if names:
            ents = {f"{kind}:{key if kind != 'power' else key + '_POWER'}" for key, (kind, _) in entities_in(text, names).items()}
        else:
            ents = set()
        for m in re.finditer(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", text):
            ents.add(m.group())  # bare game ids, resolved below
        for e in ents:
            out[e].append(it["id"])
    return dict(out)


def walk_mentions(walk_dir: Path | None, known_ids: dict) -> dict:
    """{"kind:ID": runs whose codex-walk stop names it as the expected or the emulator's value}."""
    out: collections.Counter = collections.Counter()
    if not walk_dir or not walk_dir.is_dir():
        return {}
    for f in sorted(walk_dir.glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                with contextlib.suppress(ValueError):
                    stop = json.loads(line).get("stop") or {}
                    for tok in stop_tokens(stop):
                        key = known_ids.get(tok) or known_ids.get(snake(tok))
                        if key:
                            out[key] += 1
    return dict(out)


TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def stop_tokens(stop: dict) -> list[str]:
    """The names one side of a codex-walk stop has and the other lacks (a deck list names every card it holds,
    so only what differs points at the cause)."""
    a = collections.Counter(TOKEN.findall(str(stop.get("expected") or "")))
    b = collections.Counter(TOKEN.findall(str(stop.get("emulator") or "")))
    return sorted(set((a - b) + (b - a)))


def newest_walk(referee: Path) -> Path | None:
    root = referee / "codex-walk"
    dirs = [d for d in root.glob("*") if d.is_dir() and any(d.glob("*.jsonl"))] if root.is_dir() else []
    return max(dirs, key=lambda d: max(f.stat().st_mtime for f in d.glob("*.jsonl"))) if dirs else None


def newest_backlog(referee: Path) -> Path | None:
    return newest((referee / "backlog").glob("backlog-*.json")) if (referee / "backlog").is_dir() else None


def newest_reach(lab: Path) -> Path | None:
    """The newest mutate.py prepared.json of a lab job: in the job's folder, or named by a finished job's job.json
    ({"outputs": {"reach": PATH}}, absolute or relative to the job folder)."""
    runs = Path(lab) / "runs"
    if not runs.is_dir():
        return None
    found = list(runs.glob("*/**/prepared.json"))
    for job in runs.iterdir():
        if (job / "done.rc").exists():
            v = ((read_json(job / "job.json", {}) or {}).get("outputs") or {}).get("reach")
            if v:
                found.append(Path(v) if Path(v).is_absolute() else job / v)
    return newest(found)


# ================================================================== the universe of ids and their meet counts

def universe(export: dict | None, id_map: dict) -> dict:
    """kind -> set of ids the game has (export tables; id_map for enemies and powers)."""
    u = {k: set() for k in KINDS}
    if export:
        ci = export.get("character_info", {}).get("IRONCLAD", {})
        for pool in (ci.get("card_pool"), "ColorlessCardPool"):
            u["card"] |= {c["id"] for c in export.get("card_pools", {}).get(pool, {}).get("all", [])}
        for pool in (ci.get("relic_pool"), "SharedRelicPool", "EventRelicPool"):
            u["relic"] |= {r["id"] for r in export.get("relic_pools", {}).get(pool, {}).get("all", [])}
        for pool in (ci.get("potion_pool"), "SharedPotionPool", "EventPotionPool"):
            u["potion"] |= {p["id"] for p in export.get("potion_pools", {}).get(pool, {}).get("all", [])}
        for act in ACT_NAMES:
            a = export.get("acts", {}).get(act, {})
            for kind in ("weak", "normal", "elite", "boss"):
                u["encounter"] |= set(a.get(kind, []))
            u["event"] |= set(a.get("events", []))
        u["event"] |= set(export.get("shared_events", []) or [])
    u["enemy"] |= {snake(k) for k in id_map.get("enemies", {})}
    u["power"] |= {snake(k) for k in id_map.get("powers", {})}
    u["act"] |= set(ACT_NAMES)
    return u


def power_sources(export: dict | None, corpus_ent: dict) -> dict:
    """power id -> [("card", id) | ("enemy", id)] that grant or hold it: the Ironclad's power cards (X -> X_POWER)
    and the enemies the corpus saw holding it."""
    out = collections.defaultdict(set)
    if export:
        ci = export.get("character_info", {}).get("IRONCLAD", {})
        for c in export.get("card_pools", {}).get(ci.get("card_pool"), {}).get("all", []):
            if c.get("type") == "Power":
                out[c["id"] + "_POWER"].add(("card", c["id"]))
    return out


def compute_targets(prof: Profile, uni: dict, cov: dict, reach: dict, cells: dict, backlog: dict, walk: dict,
                    psrc: dict, refset_enc: dict | None = None) -> list[dict]:
    """The weighted target list (see the module doc)."""
    total = prof.runs
    targets = []
    act_acc = {}
    for a in (1, 2, 3):
        vals = [(acc, prof.act(a).get("rooms", {}).get(r, 0)) for (aa, r), acc in cells.items() if aa == a]
        n = sum(w for _, w in vals)
        act_acc[a] = sum(acc * w for acc, w in vals) / n if n else None

    def held_acc(kind, ident):
        shares = prof.held_acts(kind, ident)
        n = sum(shares.values())
        acc = [(act_acc[a], s) for a, s in shares.items() if act_acc.get(a) is not None]
        w = sum(s for _, s in acc)
        return (sum(x * s for x, s in acc) / w) if (n and w) else None

    for kind in KINDS:
        for ident in sorted(uni.get(kind, ())):
            reasons = []
            if kind == "power":
                srcs = psrc.get(ident, set())
                m = max([prof.meet(k, i) for k, i in srcs] or [0])
                if m:
                    reasons.append(f"met through {', '.join(sorted(i for _, i in srcs))[:80]}")
            else:
                m = prof.meet(kind, ident)
            if m <= 0:
                continue
            share = m / total
            reasons.insert(0, f"{m} of {total} Ironclad A10 runs meet it")
            gap = 1.0
            key = f"{kind}:{ident}"
            c = cov.get(key, 0)
            if kind in ("event", "act"):
                pass  # run-level: setup fights never cover them
            elif c == 0:
                gap *= 3.0
                reasons.append("no recorded or setup fight reaches it yet")
            elif c < 3:
                gap *= 1.5
                reasons.append(f"only {c} recorded or setup fights reach it")
            if reach.get(key):
                gap *= 1.5
                reasons.append(f"{reach[key]} engine blocks naming it run in no recorded fight")
            acc = None
            if kind == "encounter":
                room = prof.doc.get("encounter_rooms", {}).get(ident)
                a = next((ACT_NUMBER.get(n) for n in uni.get("_enc_act", {}).get(ident, [])), None)
                own = (refset_enc or {}).get(ident)
                if own and own[1] >= 3:
                    acc = own[0] / own[1]
                    why = f"reference set: {own[0]} of {own[1]} of its fights identical"
                elif a and room and (a, room) in cells:
                    acc = cells[(a, room)]
                    why = f"reference set act {a} {room}: {acc:.0%} identical"
                else:
                    why = ""
            elif kind == "enemy":
                encs = [e for e, mons in prof.doc.get("encounter_monsters", {}).items() if ident in mons]
                vals = []
                for e in encs:
                    room = prof.doc.get("encounter_rooms", {}).get(e)
                    a = next((ACT_NUMBER.get(n) for n in uni.get("_enc_act", {}).get(e, [])), None)
                    if a and room and (a, room) in cells:
                        vals.append(cells[(a, room)])
                acc = min(vals) if vals else None
                why = f"reference set: its worst cell {acc:.0%} identical" if acc is not None else ""
            elif kind == "act":
                acc = act_acc.get(ACT_NUMBER.get(ident))
                why = f"reference set act {ACT_NUMBER.get(ident)}: {acc:.0%} identical" if acc is not None else ""
            elif kind in ("card", "relic", "potion"):
                acc = held_acc(kind, ident)
                why = f"held mostly where the reference set is {acc:.0%} identical" if acc is not None else ""
            else:
                why = ""
            if acc is not None and acc < 0.95:
                gap *= 2.0 - acc
                reasons.append(why)
            items = sorted(set(backlog.get(key, []) + backlog.get(ident, [])))
            if items:
                gap *= min(3.0, 1.0 + 0.5 * len(items))
                reasons.append(f"open backlog items name it: {', '.join(items[:6])}")
            if walk.get(key):
                gap *= 1.5
                reasons.append(f"the codex walk stops at it in {walk[key]} runs")
            targets.append({"kind": kind, "id": ident, "weight": round(share * gap, 6), "reasons": reasons})
    targets.sort(key=lambda t: (-t["weight"], t["kind"], t["id"]))
    return targets


def refset_encounters(score: dict | None) -> dict:
    """encounter -> (identical, fights) over the score's verdicts, through its manifest's specs."""
    if not score:
        return {}
    rec = Path(str(score.get("record_dir") or ""))
    man = read_json(rec.parent.parent / "manifest.json") if rec.name else None
    if not man:
        return {}
    enc_of = {}
    for e in man.get("specs", []):
        for it in (e.get("spec") or {}).get("setup", []):
            if it.get("cmd") == "fight":
                enc_of[e.get("id")] = it.get("id")
    out = collections.defaultdict(lambda: [0, 0])
    for v in score.get("verdicts", []) or []:
        enc = enc_of.get(v.get("id"))
        if enc and not v.get("error"):
            out[enc][1] += 1
            out[enc][0] += 1 if v.get("identical") else 0
    return {k: tuple(v) for k, v in out.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--lab", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--backlog", type=Path, help="default: the newest referee/backlog/backlog-*.json")
    ap.add_argument("--codex", type=Path, default=CODEX)
    ap.add_argument("--codex-max-runs", type=int, default=0, help="profile only the first N Ironclad A10 runs (tests)")
    ap.add_argument("--rebuild-profile", action="store_true")
    ap.add_argument("--profile-only", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--referee", type=Path, default=REFEREE)
    ap.add_argument("--corpus", type=Path, help="default: <referee>/corpus")
    ap.add_argument("--refset-score", type=Path, help="default: the newest <referee>/refset/v2/scores/all-*.json")
    ap.add_argument("--reach", type=Path, help="a mutate.py prepared.json (default: the newest under LAB/runs)")
    ap.add_argument("--codex-walk", type=Path, help="a folder of EmuDump codex *.jsonl (default: the newest under <referee>/codex-walk)")
    ap.add_argument("--export", type=Path, default=EXPORT)
    a = ap.parse_args(argv)

    if a.profile_only:
        return write_profile(a.lab, a.codex, a.codex_max_runs)
    if not a.out:
        ap.error("--out is required")
    a.lab.mkdir(parents=True, exist_ok=True)
    doc = ensure_profile(a.lab, a.codex, a.codex_max_runs, a.rebuild_profile)
    prof = Profile(doc)
    id_map = read_json(ROOT / "data" / "id_map.json", {}) or {}
    export = (read_json(a.export, {}) or {}).get("results", [{}])[0] if a.export.is_file() else None
    uni = universe(export, id_map)
    uni["_enc_act"] = collections.defaultdict(list)
    for act in ACT_NAMES:
        for kind in ("weak", "normal", "elite", "boss"):
            for e in ((export or {}).get("acts", {}).get(act, {}).get(kind, [])):
                uni["_enc_act"][e].append(act)
    score_path = a.refset_score or newest_refset_score(a.referee)
    score = read_json(score_path) if score_path else None
    corpus = a.corpus or a.referee / "corpus"
    setup_cov = [p for p in (a.referee / "jobs-setup" / "coverage-setup.json",) if p.is_file()]
    setup_cov += sorted(Path(a.lab, "runs").glob("*/**/coverage-setup.json")) if Path(a.lab, "runs").is_dir() else []
    cov = coverage_counts(a.lab, corpus, setup_cov, score)
    known = known_names(id_map)
    reach_path = a.reach or newest_reach(a.lab)
    reach = reach_entities(read_json(reach_path) if reach_path else None, ROOT / "src" / "Sts2Emulator", known)
    backlog_path = a.backlog or newest_backlog(a.referee)
    raw = backlog_mentions(read_json(backlog_path) if backlog_path else None, id_map)
    kind_of = {i: k for k in KINDS for i in uni.get(k, ())}
    backlog = collections.defaultdict(list)
    for key, items in raw.items():
        kk = key if ":" in key else (f"{kind_of[key]}:{key}" if key in kind_of else None)
        if kk:
            backlog[kk] += items
    ids = {i: f"{kind_of[i]}:{i}" for i in kind_of}
    ids.update(known)
    walk_dir = a.codex_walk or newest_walk(a.referee)
    walk = walk_mentions(walk_dir, ids)
    corpus_ent = read_json(Path(a.lab) / "cache" / "corpus_entities.json", {}) or {}
    targets = compute_targets(prof, uni, cov, reach, refset_cells(score), dict(backlog), walk,
                              power_sources(export, corpus_ent), refset_encounters(score))
    out = {"main_sha": git_sha(), "generated_utc": utc_now(), "profile": rel_to(profile_path(a.lab), a.lab),
           "inputs": {"runs": prof.runs, "backlog": rel_to(backlog_path, a.lab) if backlog_path else None,
                      "refset_score": rel_to(score_path, a.lab) if score_path else None,
                      "reach": rel_to(reach_path, a.lab) if reach_path else None,
                      "codex_walk": rel_to(walk_dir, a.lab) if walk_dir else None,
                      "covered_ids": len(cov)},
           "totals": dict(collections.Counter(t["kind"] for t in targets)), "targets": targets}
    write_json(a.out, out)
    top = ", ".join(f"{t['kind']}:{t['id']} {t['weight']:.3f}" for t in targets[:5])
    print(f"targets: {len(targets)} ({out['totals']}); top: {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
