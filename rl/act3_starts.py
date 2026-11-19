#!/usr/bin/env python3
"""Act 3 boss positions for the fight exam and the practice (the Act 3 week, lane 5: the Test Subject).

    python rl/act3_starts.py scan   --out DIR [--corpus FILE] [--limit-runs N] [--cache FILE]   read the corpus once: the human bar and the positions
    python rl/act3_starts.py table  --out DIR [--stores S ...]                                  the per-boss table: humans against our runs
    python rl/act3_starts.py human  --out DIR [--procs 4] [--engine old|core2] [--only-ids FILE]  deal the human positions' openings on the copy
    python rl/act3_starts.py own    --out DIR [--stores S ...] [--n-first N] [--n-second N]       our own Act 3 boss arrivals, from the plain-run stores

Why. The two Act 3 bosses (floors 48 and 49 at Ascension 10; the second is met only by a run that beat the first) hold nine tenths of the gap
between the loop and the average human, and one boss in particular, the Test Subject, passes in 2 of 100 for us and 70 of 100 for humans. The
fight exam (`fight_exam.py`) and the practice (`practice_mix.py`) held Acts 1 and 2 only (`human_starts.py`). This module builds the Act 3
positions the same way: a human run in the public corpus records, floor by floor, what the player carried into each fight, and the copy
deals that opening (`human_starts.rebuild_run(..., acts=(3,))`, `build_doc`); our own arrivals come from the plain runs' saved fight states
(`act3_replay.position_spec`).

What is written. `human-starts-act3.jsonl` (+ index, lite list) and `human-outcomes-act3.json` (the human's result of each start, the same
reading as `human-outcomes.json`); `own-starts-act3.*` for our arrivals. Both in the gym's start schema (`cat: "act3-boss"`, `boss_no` 1 or 2).
The exam takes its cells from them (`fight_exam.py add-act3`), the practice mix reads them when its Act 3 share is above zero
(`mix.json` `act3.share`, off by default).

The rules of the human corpus (D13, D38): the exam fifth is never read (a run listed in the ids file or put in the fifth by the hash rule is
left out before anything is taken from it); only aggregates leave this module (a table cell over fewer than `MIN_CELL` runs is not shown); a
human run's id never reaches a file written here: a start's run is a salted hash of it (`run_tag`), and the one place the real id is used
is the check against the runs the loop has played, made while scanning.

Lab code: nothing here changes the engine, the search or the solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import human_starts as hs  # noqa: E402

RULE = "act3-starts-v1"
CAT = "act3-boss"
HUMAN_STEM = "human-starts-act3"
OWN_STEM = "own-starts-act3"
OUTCOME_FILE = "human-outcomes-act3.json"
SPECS_FILE = "act3-specs.json"
MIN_CELL = hs.MIN_CELL
TAG_SALT = "spireline-act3-run|"
# The copy's encounter ids of the Glory act's bosses (`run_train.vocab.BOSSES`) and the corpus's names for them.
BOSS_ID = {"ENCOUNTER.AEONGLASS_BOSS": 73, "ENCOUNTER.QUEEN_BOSS": 78, "ENCOUNTER.TEST_SUBJECT_BOSS": 80}
BOSS_SLUG = {73: "aeonglass", 78: "queen", 80: "test-subject"}
BOSS_ORDER = (80, 78, 73)  # the Test Subject first
OWN_STORES = ("D:/spireline/referee/run-cheap/17", "D:/spireline/referee/run-cheap/18")
ACT3_INDEX = 2

# The exam's cells and their quotas. Weighted to where our runs die: of 3,100 deaths at the two Act 3 bosses in generations 16 to 18 about 39
# in 100 were at the Test Subject (971 as the first boss, 246 as the second), and the Test Subject is where the champion's fight is weakest,
# so it takes the largest cells (a cell is read as a rate, so the one with the most to learn is the one with the most starts). A cell under
# `MIN_CELL` runs is never shown, so no quota is below it.
HUMAN_QUOTA = {"act3-boss1-80": 80, "act3-boss2-80": 40, "act3-boss1-78": 32, "act3-boss1-73": 32, "act3-boss2-78": 20, "act3-boss2-73": 20}
OWN_QUOTA = {"act3-boss1-80": 60, "act3-boss2-80": 30, "act3-boss1-78": 30, "act3-boss1-73": 30, "act3-boss2-78": 20, "act3-boss2-73": 20}
PICK_SEED = 20261119


# ---------------------------------------------------------------------------------------------------------- pure parts


def exam_ids(*paths: str) -> frozenset[str]:
    """The exam fifth's ids: the ids files named, and the 2026-11-06 file beside the corpus when it is there (the hash rule `hs.is_exam`
    covers the same fifth either way)."""
    files = [p for p in paths if p]
    extra = hs.EXAM_IDS.with_name("human-exam-fifth-ids-2026-11-06.txt")
    if extra.is_file():
        files.append(str(extra))
    out: frozenset[str] = frozenset()
    for f in files:
        out |= hs.read_exam_ids(f)
    return out


def run_tag(run: str) -> str:
    """A human run's stand-in: a salted hash of its id, twelve hex digits. The id itself is never written."""
    return hashlib.sha256((TAG_SALT + str(run)).encode()).hexdigest()[:12]


def anonymise(spec: dict) -> dict:
    """The spec of a human fight with its run's id replaced by the tag (id, run and seed); the start's floor keeps its place in the id."""
    tag = run_tag(spec["run"])
    spec["id"] = f"human3:{tag}|f{int(spec['floor']):02d}"
    spec["run"] = tag
    spec["seed"] = tag
    return spec


def cell_name(rec: dict) -> str:
    """`act3-boss1-80`: the boss's turn (1 first, 2 second) and the copy's encounter id."""
    return f"act3-boss{int(rec['boss_no'])}-{int(rec['encounter_id'])}"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def order_key(seed: int, sid: str) -> str:
    return hashlib.sha256(f"act3-pick:{seed}:{sid}".encode()).hexdigest()


def pick_quota(records: list[dict], quota: dict[str, int], *, seed: int = PICK_SEED, taken: set | None = None, skip=None,
               prefer=None) -> tuple[list[dict], dict]:
    """Per cell, up to `quota[cell]` starts by a seeded hash, at most one start a run (across the cells, `taken` carrying the runs already
    used), a record `skip` names left out (and counted). `prefer(record)`: the records it holds for come first (each group in hash order), the
    others fill what the preferred cannot; a chosen record carries `clean` (preferred, or no preference asked). Cells the quota does not name
    are not taken. Returns (the chosen records, a report by cell: pool, runs, quota, chosen, clean, short)."""
    taken = set() if taken is None else taken
    cells: dict[str, list[dict]] = defaultdict(list)
    left_out: Counter = Counter()
    for r in records:
        c = cell_name(r)
        if c not in quota:
            continue
        why = skip(r) if skip else None
        if why:
            left_out[why] += 1
            continue
        cells[c].append(r)
    chosen: list[dict] = []
    report: dict[str, dict] = {}
    for c in sorted(quota):
        pool = sorted(cells.get(c, []), key=lambda r: (0 if prefer is None or prefer(r) else 1, order_key(seed, r["id"])))
        took = clean = 0
        for r in pool:
            if took >= quota[c]:
                break
            if r["run"] in taken:
                continue
            taken.add(r["run"])
            ok = prefer is None or bool(prefer(r))
            chosen.append({**r, "cell": c, "clean": ok})
            took += 1
            clean += ok
        report[c] = {"pool": len(pool), "runs": len({r["run"] for r in pool}), "clean_pool": sum(1 for r in pool if prefer is None or prefer(r)),
                     "quota": quota[c], "chosen": took, "clean": clean, "short": took < quota[c]}
    report["_left_out"] = dict(left_out)
    return chosen, report


# ---------------------------------------------------------------------------------------------------------- the corpus pass


def _raw_arrivals(r: dict) -> list[dict]:
    """One run's Act 3 boss fights as raw numbers (no deck rebuild): boss turn, the boss, whether the run went past it, the HP going in and
    out, damage taken, turns, potions used. Read off the per-floor record only."""
    flat = [(a, pt) for a, act in enumerate(r.get("map_point_history") or []) for pt in act]
    out = []
    boss_no = 0
    won = bool(r.get("win"))
    for f, (a, pt) in enumerate(flat, 1):
        if a != ACT3_INDEX:
            continue
        fight = hs.point_fight(pt)
        if fight is None or fight[0] != "boss":
            continue
        boss_no += 1
        ps = (pt.get("player_stats") or [{}])[0]
        prev = (flat[f - 2][1].get("player_stats") or [{}])[0] if f >= 2 else {}
        room = (pt.get("rooms") or [{}])[-1]
        out.append({"boss_no": boss_no, "boss": fight[1], "floor": f, "past": f < len(flat) or won, "hp_in": prev.get("current_hp"),
                    "max_hp": prev.get("max_hp"), "damage": ps.get("damage_taken"), "turns": room.get("turns_taken"),
                    "potions_used": len(ps.get("potion_used") or [])})
    return out


def _bump(agg: dict, key: str, x: dict) -> None:
    c = agg.setdefault(key, {"n": 0, "past": 0, "hp_n": 0, "hp_frac": 0.0, "dmg_n": 0, "dmg": 0.0, "turn_n": 0, "turns": 0.0, "potion_n": 0,
                             "potions_used": 0.0, "past_hp_frac": 0.0, "past_hp_n": 0})
    c["n"] += 1
    c["past"] += bool(x["past"])
    if isinstance(x.get("hp_in"), (int, float)) and isinstance(x.get("max_hp"), (int, float)) and x["max_hp"] > 0:
        frac = x["hp_in"] / x["max_hp"]
        c["hp_n"] += 1
        c["hp_frac"] += frac
        if x["past"]:
            c["past_hp_n"] += 1
            c["past_hp_frac"] += frac
    if isinstance(x.get("damage"), (int, float)):
        c["dmg_n"] += 1
        c["dmg"] += x["damage"]
    if isinstance(x.get("turns"), (int, float)):
        c["turn_n"] += 1
        c["turns"] += x["turns"]
    c["potion_n"] += 1
    c["potions_used"] += x["potions_used"]


def scan(corpus: Path, exam: frozenset[str], limit_runs: int = 0, played: set | frozenset = frozenset(), cache: Path | None = None) -> dict:
    """The corpus read once (the exam fifth is never an input): the human bar by Act 3 boss (`raw`: counts, never a run) and every Act 3 boss
    fight whose opening could be rebuilt (`specs`, the run's id replaced by its tag; `loop_played`: the loop has played a start of that
    run). With a cache file the result is read from it when it exists and written to it when it does not."""
    if cache is not None and Path(cache).is_file():
        return json.loads(Path(cache).read_text(encoding="utf-8"))
    tables = hs.Tables.load()
    stats: Counter = Counter()
    fail: Counter = Counter()
    seen: Counter = Counter()
    raw: dict[str, dict] = {}
    specs: list[dict] = []
    for r in hs.read_runs(corpus, exam, stats):
        rid = hs.run_id(r)
        assert not hs.is_exam(rid, exam)
        for x in _raw_arrivals(r):
            b = BOSS_ID.get(x["boss"])
            name = BOSS_SLUG.get(b, "other")
            _bump(raw, f"boss{x['boss_no']}|{name}", x)
        got = hs.rebuild_run(r, tables, fail, seen, acts=(3,), kinds=("boss",))
        if got:
            stats["runs_with_act3_fights"] += 1
        for spec in got:
            spec["loop_played"] = rid in played
            specs.append(anonymise(spec))
        if limit_runs and stats["runs_usable"] >= limit_runs:
            break
    out = {"rule": RULE, "stats": dict(stats), "fail": dict(fail.most_common()), "raw": raw, "specs": specs}
    if cache is not None:
        Path(cache).write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return out


def human_bar_rows(raw: dict) -> list[dict]:
    """The human bar by Act 3 boss: for each boss turn and boss, the runs that met it, the share that went past it (95% interval), the mean
    health going in, the mean damage taken in the fight and its turns. A cell under `MIN_CELL` runs is returned with `n` only."""
    rows = []
    for key in sorted(raw, key=lambda k: (k.split("|")[0], [BOSS_SLUG.get(b) for b in BOSS_ORDER].index(k.split("|")[1]) if k.split("|")[1] in BOSS_SLUG.values() else 9)):
        c = raw[key]
        no, name = key.split("|")
        if c["n"] < MIN_CELL:
            rows.append({"turn": no, "boss": name, "n": c["n"], "suppressed": True})
            continue
        lo, hi = wilson(c["past"], c["n"])
        rows.append({"turn": no, "boss": name, "n": c["n"], "past": c["past"], "rate": c["past"] / c["n"], "lo": lo, "hi": hi,
                     "hp_frac": c["hp_frac"] / max(1, c["hp_n"]), "damage": c["dmg"] / max(1, c["dmg_n"]),
                     "turns": c["turns"] / max(1, c["turn_n"]), "potions_used": c["potions_used"] / max(1, c["potion_n"]),
                     "past_hp_frac": c["past_hp_frac"] / c["past_hp_n"] if c["past_hp_n"] else None})
    return rows


def spec_aggregates(specs: list[dict]) -> dict:
    """What the rebuilt openings say: by boss turn and boss, and by whether the human went past, the mean deck size, relics held, potions
    held, health going in. Cells under `MIN_CELL` are left out."""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for s in specs:
        o = s.get("outcome") or {}
        cells[(int(s["boss_no"]), BOSS_SLUG.get(int(s["encounter_id"]), "other"), bool(o.get("human_survived")))].append(s)
    out = {}
    for (no, name, past), v in sorted(cells.items()):
        if len(v) < MIN_CELL:
            continue
        out[f"boss{no}|{name}|{'past' if past else 'died'}"] = {
            "n": len(v), "deck": sum(len(s["deck"]) for s in v) / len(v), "relics": sum(len(s["relics"]) for s in v) / len(v),
            "potions": sum(len(s["potions"]) for s in v) / len(v), "hp_frac": sum(s["hp"] / s["max_hp"] for s in v) / len(v),
            "max_hp": sum(s["max_hp"] for s in v) / len(v)}
    return out


# ---------------------------------------------------------------------------------------------------------- our own arrivals


def own_table(stores: list[Path]) -> dict:
    """Our Act 3 boss fights in the plain-run stores' saved records: by boss turn and boss, the fights, the kills (the fight won), the mean
    health going in, deck size, relics, potions held, and the same split by whether the fight was won. Read off the record files; no run is
    played."""
    import act3_replay as ar  # noqa: PLC0415

    cells: dict[str, dict] = {}
    for store in stores:
        seen: set[str] = set()
        for row in ar.iter_jsonl(Path(store) / "runs.jsonl"):
            seed = row.get("seed")
            if not seed or row.get("capped") or seed in seen:
                continue
            if not any(int(b.get("act", -1)) == ACT3_INDEX for b in row.get("bosses") or []):
                continue
            seen.add(seed)
            try:
                rec = json.loads(ar.record_path(str(store), seed).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            fights = [f for f in rec.get("fights") or [] if int(f.get("node_type", 0)) == ar.NODE_BOSS
                      and int((f.get("state") or {}).get("act_index", -1)) == ACT3_INDEX]
            for no, f in enumerate(fights[:2], 1):
                st = f.get("state") or {}
                if not st.get("max_hp"):
                    continue
                name = BOSS_SLUG.get(int(f.get("encounter", -1)), "other")
                for key in (f"boss{no}|{name}|all", f"boss{no}|{name}|{'won' if f.get('won') else 'lost'}"):
                    c = cells.setdefault(key, {"n": 0, "hp_frac": 0.0, "deck": 0, "relics": 0, "potions": 0, "max_hp": 0, "hp_after": 0.0})
                    c["n"] += 1
                    c["hp_frac"] += int(st.get("hp", 0)) / int(st["max_hp"])
                    c["deck"] += len(st.get("deck") or [])
                    c["relics"] += len(st.get("relics") or [])
                    c["potions"] += len(st.get("potions") or [])
                    c["max_hp"] += int(st["max_hp"])
    out = {}
    for key, c in sorted(cells.items()):
        if c["n"] < MIN_CELL:
            continue
        n = c["n"]
        out[key] = {"n": n, "hp_frac": c["hp_frac"] / n, "deck": c["deck"] / n, "relics": c["relics"] / n, "potions": c["potions"] / n,
                    "max_hp": c["max_hp"] / n}
    return out


def own_record(pos: dict) -> dict:
    """One of our Act 3 boss arrivals (a position of `act3_replay.build_positions`, with its opening document) as a start in the gym's schema."""
    from gym_loop import is_val_seed  # noqa: PLC0415

    doc = pos["doc"]
    pots = doc.get("potions") or {}
    held = sum(1 for x in (pots.get("slots") or [])[: int(pots.get("max_slots", 0))] if x)
    gen, seed = str(pos["gen"]), str(pos["seed"])
    return {"id": f"own3:{gen}:{seed}|f{int(pos['floor']):02d}", "cat": CAT, "kind": "boss", "act": 3, "floor": int(pos["floor"]), "seed": seed, "ix": -1,
            "night": "own", "source": "own", "run": seed, "enemy": pos["boss"], "k": 0, "node_type": 6, "hp": int(pos["hp"]),
            "max_hp": int(pos["max_hp"]), "deck_size": len(hs.doc_cards(doc)), "potions": held, "relics": len(pos["relics"]),
            "encounter_id": int(pos["encounter_id"]), "turn": int(doc.get("turn", 0)), "approx": {"counters_written": int(pos.get("counters_written") or 0)},
            "kind_same": True, "replay_same": None, "split": "val" if is_val_seed(seed) else "train", "boss_no": int(pos["ordinal"]),
            "gen": gen, "own_run_won": bool(pos.get("run_won")), "own_hp_after": int(pos.get("run_hp_after", 0)), "stratum": "own", "doc": doc}


def write_set(out_dir: Path, stem: str, records: list[dict]) -> int:
    """`<stem>.jsonl` with its byte index and lite list (the shape `gym_loop.load_start` reads); the old files are replaced whole."""
    w = hs.SetWriter(out_dir, stem)
    try:
        for r in sorted(records, key=lambda r: r["id"]):
            w.add(json.dumps(r, separators=(",", ":")))
    except BaseException:
        w.fh.close()
        w.tmp.unlink(missing_ok=True)
        raise
    return w.close()


# ---------------------------------------------------------------------------------------------------------- the commands


def _tables_text(rows: list[dict], own: dict, agg: dict) -> str:
    L = ["Humans against our runs at the Act 3 bosses (humans: Ironclad, Ascension 10, the corpus's training fifths; ours: the plain runs of "
         "generations 17 and 18 as the stores hold them; a cell under %d runs is not shown)" % MIN_CELL, "",
         "boss turn  boss           human runs  went past (95% interval)   HP in (share of max)  damage taken in the fight  turns  potions used"]
    for r in rows:
        if r.get("suppressed"):
            L.append(f"{r['turn']:<9}  {r['boss']:<13}  under {MIN_CELL}")
            continue
        L.append(f"{r['turn']:<9}  {r['boss']:<13}  {r['n']:>9}  {100 * r['rate']:>5.1f} [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]"
                 f"{'':>7}{100 * r['hp_frac']:>5.1f}{'':>13}{r['damage']:>6.1f}{'':>20}{r['turns']:>5.1f}{r['potions_used']:>9.2f}")
    L += ["", "Our runs (the fight won / the fight lost), by the same cells"]
    for k, c in own.items():
        L.append(f"  {k:<32} n={c['n']:>5}  HP in {100 * c['hp_frac']:.1f}% of {c['max_hp']:.0f}  deck {c['deck']:.1f}  relics {c['relics']:.1f}  potions {c['potions']:.2f}")
    L += ["", "Humans' rebuilt openings (went past / died)"]
    for k, c in agg.items():
        L.append(f"  {k:<32} n={c['n']:>5}  HP in {100 * c['hp_frac']:.1f}% of {c['max_hp']:.0f}  deck {c['deck']:.1f}  relics {c['relics']:.1f}  potions {c['potions']:.2f}")
    return "\n".join(L)


def cmd_scan(a: argparse.Namespace) -> int:
    import fight_exam as fe  # noqa: PLC0415

    hs.below_normal()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exam = exam_ids(a.exam_ids)
    played = fe.played_runs()
    t0 = time.time()
    res = scan(Path(a.corpus), exam, a.limit_runs, played, Path(a.cache) if a.cache else out / SPECS_FILE)
    rows = human_bar_rows(res["raw"])
    print(f"act3_starts scan: {res['stats'].get('runs_usable', '?')} usable runs, {len(res['specs'])} Act 3 boss openings rebuilt in state "
          f"({time.time() - t0:.0f} s)")
    for k, v in res["fail"].items():
        print(f"  not built: {k}: {v}")
    print(_tables_text(rows, {}, spec_aggregates(res["specs"])))
    return 0


def cmd_table(a: argparse.Namespace) -> int:
    res = json.loads((Path(a.out) / SPECS_FILE).read_text(encoding="utf-8"))
    text = _tables_text(human_bar_rows(res["raw"]), own_table([Path(s) for s in a.stores]), spec_aggregates(res["specs"]))
    print(text)
    return 0


def cmd_human(a: argparse.Namespace) -> int:
    """Deal the openings of the human positions on the copy and write the set and the outcomes sidecar."""
    hs.below_normal()
    if a.procs > 4:
        raise SystemExit("act3_starts: at most 4 processes")
    os.environ["HUMAN_STARTS_ENGINE"] = a.engine
    out = Path(a.out)
    res = json.loads((out / SPECS_FILE).read_text(encoding="utf-8"))
    specs = res["specs"]
    if a.only_ids:
        want = set(Path(a.only_ids).read_text(encoding="utf-8").split())
        specs = [s for s in specs if s["id"] in want]
    writer = hs.SetWriter(out, HUMAN_STEM)
    by_id = {s["id"]: s for s in specs}
    outcomes: dict[str, dict] = {}

    def sink(line: str) -> None:
        rec = json.loads(line)
        spec = by_id[rec["id"]]
        rec.update({"boss_no": spec["boss_no"], "stratum": "human", "loop_played": bool(spec.get("loop_played")), "engine": a.engine})
        if spec.get("outcome"):
            outcomes[rec["id"]] = spec["outcome"]
        writer.add(json.dumps(rec, separators=(",", ":")))

    try:
        fail = hs.deal(specs, a.procs, sink)
    except BaseException:
        writer.fh.close()
        writer.tmp.unlink(missing_ok=True)
        raise
    n = writer.close()
    doc = {"rule": hs.OUTCOME_RULE, "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": "the corpus's training fifths",
           "starts": n, "n": len(outcomes), "outcomes": outcomes}
    hs.atomic_json(out / OUTCOME_FILE, doc)
    print(f"act3_starts human: {n} openings dealt on the {a.engine} engine of {len(specs)} positions; {len(outcomes)} outcomes")
    for k, v in fail.most_common():
        print(f"  not dealt: {k}: {v}")
    return 0


def cmd_own(a: argparse.Namespace) -> int:
    """Our own Act 3 boss arrivals: the plain runs' saved fight states, rebuilt on the copy by `act3_replay` and written as starts."""
    import act3_replay as ar  # noqa: PLC0415
    import gen0  # noqa: PLC0415

    hs.below_normal()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.lib:
        os.environ["STS2_LIB_PATH"] = a.lib
    gen0.select_engine(a.engine)
    tmp = out / "own-positions"
    meta = ar.build_positions(tmp, [Path(s) for s in a.stores], a.n_first, a.n_second, a.salt)
    positions = ar.read_jsonl(tmp / ar.FILES["positions"])
    n = write_set(out, OWN_STEM, [own_record(p) | {"engine": a.engine} for p in positions])
    (tmp / ar.FILES["positions"]).unlink(missing_ok=True)  # the starts hold the documents; the positions file would be a second copy
    print(f"act3_starts own: {n} starts; {json.dumps({k: meta[k] for k in ('by_ordinal', 'by_boss', 'by_generation', 'skipped')})}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--out", required=True)
    p.add_argument("--corpus", default=str(hs.CORPUS))
    p.add_argument("--exam-ids", default=str(hs.EXAM_IDS))
    p.add_argument("--limit-runs", type=int, default=0)
    p.add_argument("--cache", default="")
    p.set_defaults(fn=cmd_scan)
    p = sub.add_parser("table")
    p.add_argument("--out", required=True)
    p.add_argument("--stores", nargs="+", default=list(OWN_STORES))
    p.set_defaults(fn=cmd_table)
    p = sub.add_parser("human")
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--engine", choices=("old", "core2"), default="old")
    p.add_argument("--only-ids", default="", help="a file of start ids: deal only those (the exam's picks)")
    p.set_defaults(fn=cmd_human)
    p = sub.add_parser("own")
    p.add_argument("--out", required=True)
    p.add_argument("--stores", nargs="+", default=list(OWN_STORES))
    p.add_argument("--n-first", type=int, default=100000)
    p.add_argument("--n-second", type=int, default=100000)
    p.add_argument("--salt", type=int, default=20261119)
    p.add_argument("--engine", choices=("old", "core2"), default="old")
    p.add_argument("--lib", default="")
    p.set_defaults(fn=cmd_own)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
