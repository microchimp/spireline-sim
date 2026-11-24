#!/usr/bin/env python3
"""Act 3 boss positions for the fight exam and the practice (the Act 3 week, lane 5: the Test Subject).

    python rl/act3_starts.py scan   --out DIR [--corpus FILE] [--limit-runs N] [--cache FILE]   read the corpus once: the human bar and the positions
    python rl/act3_starts.py table  --out DIR [--stores S ...]                                  the per-boss table: humans against our runs
    python rl/act3_starts.py decks  --out DIR                                                   the decks walked in with: skills, attacks, powers by boss and result
    python rl/act3_starts.py human  --out DIR [--procs 4] [--engine old|core2] [--only-ids FILE]  deal the human positions' openings on the copy
    python rl/act3_starts.py own    --out DIR [--stores S ...] [--n-first N] [--n-second N]       our own Act 3 boss arrivals, from the plain-run stores
    python rl/act3_starts.py arrivals --champion DIR --out DIR [--limit N] [--rebuild]   the champion's first-Act-3-boss arrivals from its judged arms (S1's ruler);
                                                                                          a ruler already in --out is extended by the arms it lacks
    python rl/act3_starts.py arrival-exam --ruler DIR --candidate DIR --champion DIR   S1: the candidate's fight network on that ruler (a report)
    python rl/act3_starts.py merge-exclusions --ruler DIR [--exam-file F] [--dry]       the ruler's runs and starts into the fight loop's exclusion list

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
from collections.abc import Sequence
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


def deck_mix(card_ids, types: dict[int, str]) -> dict[str, int]:
    """The deck as counts: its size, Skills, Attacks, Powers, and the starter Defends (a card's id may be negative: upgraded)."""
    c: Counter = Counter()
    for i in card_ids:
        t = types.get(abs(int(i)), "?")
        c["size"] += 1
        c[t.lower()] += 1
    return dict(c)


def decks_table(human_specs: list[dict], own_docs, types: dict[int, str]) -> dict[str, dict]:
    """Aggregates of the decks walked into the first and second Act 3 boss: by side (human, own), boss turn, boss and the result (went past or
    died for a human; fight won or lost for ours), the number of fights and the mean count of Skills, Attacks, Powers and the deck's size.
    `own_docs`: (boss turn, encounter id, fight won, the opening's card list `[def_id, ...]`) for each of our arrivals. A cell under `MIN_CELL` is left out."""
    cells: dict[str, dict] = {}

    def add(key: str, ids) -> None:
        m = deck_mix(ids, types)
        c = cells.setdefault(key, {"n": 0, "size": 0, "skill": 0, "attack": 0, "power": 0})
        c["n"] += 1
        for k in ("size", "skill", "attack", "power"):
            c[k] += m.get(k, 0)

    for s in human_specs:
        name = BOSS_SLUG.get(int(s["encounter_id"]), "other")
        res = "past" if (s.get("outcome") or {}).get("human_survived") else "died"
        for k in (f"human|boss{s['boss_no']}|{name}|all", f"human|boss{s['boss_no']}|{name}|{res}"):
            add(k, s["deck"])
    for no, eid, won, ids in own_docs:
        name = BOSS_SLUG.get(int(eid), "other")
        for k in (f"own|boss{no}|{name}|all", f"own|boss{no}|{name}|{'won' if won else 'lost'}"):
            add(k, ids)
    return {k: {"n": c["n"], **{m: c[m] / c["n"] for m in ("size", "skill", "attack", "power")}} for k, c in sorted(cells.items()) if c["n"] >= MIN_CELL}


def cmd_decks(a: argparse.Namespace) -> int:
    from sts2_gym import names  # noqa: PLC0415 -- reads the copy's C# tables as text

    out = Path(a.out)
    types = {int(i): c.card_type for i, c in names.cards().items()}
    res = json.loads((out / SPECS_FILE).read_text(encoding="utf-8"))

    def own_docs():
        for line in open(out / f"{OWN_STEM}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            yield int(r["boss_no"]), int(r["encounter_id"]), bool(r.get("own_run_won")), [c["def_id"] for c in hs.doc_cards(r["doc"])]

    t = decks_table(res["specs"], own_docs(), types)
    print("Decks walked into the Act 3 bosses (mean cards a deck; a cell under %d fights is not shown)" % MIN_CELL)
    print(f"  {'side | turn | boss | result':<44}{'fights':>7}{'cards':>7}{'skills':>8}{'attacks':>9}{'powers':>8}")
    for k, v in t.items():
        print(f"  {k.replace('|', ' | '):<44}{v['n']:>7}{v['size']:>7.1f}{v['skill']:>8.1f}{v['attack']:>9.1f}{v['power']:>8.1f}")
    return 0


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


# ---------------------------------------------------------------------------------------------------------- the arrival ruler (judge build 1)
#
# S1's ruler (judge-design/DESIGN.md section 3; D68 addendum 6 outcome 3): the champion's own first-Act-3-boss arrivals, rebuilt from the arms the
# run judge played it in on blocks it was not selected on. A judged run's row holds the seed, the run's index, the node string and the choices
# (`decided`), no fight position; the gym's replay step (`fight_gym._replay_task`: the run played again exactly, with the champion's own agent on
# the engine the arm was played on, ending at the boss's first decision) deals the position, and a replay is kept only when it matches its row (the
# map path, the cards taken and the rests up to the boss). The champion's judged arms were played with the prior on the GPU server and the
# replay uses the in-process prior: a replay that does not match is dropped, never repaired, and the match rate is in the ruler's meta.
#
#   python rl/act3_starts.py arrivals     --champion DIR --out DIR [--limit N] [--rebuild]   build the ruler (with --limit N: only check that N rows replay)
#   python rl/act3_starts.py arrival-exam --ruler DIR --candidate DIR --champion DIR [--test K]   play it and read S1 (a report, no rule)
#
# The ruler grows with the champion (judge build 3, D68 addendum 6 outcome 5). The champion plays one more judged block each time it is the base of a
# challenger's judgement, and a ruler built after the first block would stay that thin for the champion's whole tenure. `arrivals` on a folder that
# already holds this champion's ruler (its starts hash to its meta) therefore replays only the arms the ruler lacks (an arm it has no entry for, or one
# that holds more rows than the ruler's entry: a block completed since), keeps every start it has (no second replay of an arm in the ruler), and writes
# the whole set again with a new hash. The hash names the ruler: a test played on the old hash stays valid for it, and the exam's games of the old hash
# are put away under `exam/<old hash first 8>/` so that the new hash's tests start at 0 with draws of their own (`retire_exam`; `exam/ruler_hash.txt`
# says which hash the games in `exam/` were played on). `meta.json` keeps the history: when, which arms, how many starts each added.

ARRIVAL_RULE = "act3-arrival-v1"
ARRIVAL_STEM = "starts-arrival"
ARRIVAL_NS = "ar"  # the arrival exam's draw namespace (the gym's "g", the exam's "x", the Act 3 replay's "r3", the practice's "L")
ARRIVAL_DRAWS = 6  # draws a start in one test: test k plays draws 6k to 6k + 5
ARRIVAL_CONFIG = "a3-v1"
AGENT_ROOT = Path("D:/spireline/referee/agent")
ARRIVAL_ROOT = Path("D:/spireline/referee/gym-loop/arrival")


def arm_label(folder: str | Path, config: str = ARRIVAL_CONFIG) -> str:
    """An arm folder's block name: `E21-2026-11-21-r1-a3-v1` -> `E21-2026-11-21-r1`."""
    name = Path(folder).name
    return name[: -len("-" + config)] if config and name.endswith("-" + config) else name


def selected_blocks(champion_dir: str | Path) -> set[str]:
    """The blocks the champion was selected on: its promotion.json's `seed_block` and any block pooled into that judgement. Its judged runs there
    are the runs the keep rule picked it on, so they are not an independent ruler."""
    p = Path(champion_dir) / "promotion.json"
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    out = {str(doc["seed_block"])} if doc.get("seed_block") else set()
    for b in (doc.get("pooled") or {}).get("blocks") or []:
        if isinstance(b, dict) and b.get("block"):
            out.add(str(b["block"]))
    return out


def champion_arms(champion_dir: str | Path, exclude: Sequence[str] = ()) -> list[Path]:
    """The champion's own arm folders (`<champion>/arms/*`, each with summary.json and runs.jsonl), oldest name first, without a block it was selected
    on (`selected_blocks`) and without the blocks named in `exclude`: the blocks it played as the base of its successors' judgements."""
    skip = selected_blocks(champion_dir) | set(exclude)
    out = []
    for d in sorted((Path(champion_dir) / "arms").glob("*")):
        if (d / "summary.json").is_file() and (d / "runs.jsonl").is_file() and arm_label(d, arm_config(d)) not in skip:
            out.append(d)
    return out


def arm_config(folder: str | Path) -> str:
    """The config an arm folder was played under (its summary's `config`, `a3-v1` when it names none)."""
    return json.loads((Path(folder) / "summary.json").read_text(encoding="utf-8")).get("config") or ARRIVAL_CONFIG


def count_rows(folder: str | Path) -> int:
    """The judged runs in an arm folder: the non-blank lines of its runs.jsonl."""
    with open(Path(folder) / "runs.jsonl", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def ruler_arms(meta: dict | None) -> dict[str, dict]:
    """The arms a ruler was built from, by block label: the entries of its meta's `arms` ({block, rows, arrivals, replays, matched, ...}). Empty for no ruler."""
    return {str(x["block"]): x for x in (meta or {}).get("arms") or []}


def ruler_gap(meta: dict | None, champion_dir: str | Path, exclude: Sequence[str] = ()) -> list[dict]:
    """The champion's eligible arms (`champion_arms`) that a ruler does not hold, oldest name first: [{label, path, rows, why}]. `why` is `new` for an arm the
    ruler has no entry for and `grown` for one that has more rows now than its entry's (a block completed since the ruler was built: its extra runs may
    hold arrivals). With no ruler (`meta` None) every eligible arm is new: an empty answer then means there is nothing to build a ruler from."""
    have = ruler_arms(meta)
    out = []
    for d in champion_arms(champion_dir, exclude):
        label, rows = arm_label(d, arm_config(d)), count_rows(d)
        got = have.get(label)
        if got is None:
            out.append({"label": label, "path": d, "rows": rows, "why": "new"})
        elif rows > int(got.get("rows") or 0):
            out.append({"label": label, "path": d, "rows": rows, "why": "grown"})
    return out


def arrival_floors(row: dict) -> list[tuple[int, int]]:
    """[(boss turn, floor)] of the Act 3 boss rooms the judged run entered, from its complete node string: turn 1 the first Act 3 boss, turn 2 the
    second (a run that beat the first). [] for a run that never arrived, a crashed or lost seed, or a row whose node string is not complete (the
    string, not the usual floors 48 and 49, says where the bosses are: one run in 400 has a longer Act 2)."""
    from tools.lab import keep_v2 as k2  # noqa: PLC0415 -- pure

    if not k2.arrived_b3a(row) or not k2.complete_nodes(row):
        return []
    return [(i + 1, f) for i, f in enumerate(k2.boss_floors(row["nodes"])[2:4])]


def arrival_tasks(arm_rows: list[dict], label: str, config: str, taken: set | None = None) -> list[tuple]:
    """`fight_gym._replay_task`'s tasks for one arm's rows: one per run that arrived at the first Act 3 boss, wanting the first boss's opening (and the
    second's when the run entered it). A seed already in `taken` (another arm played it) is left to the arm that has it."""
    taken = set() if taken is None else taken
    tasks = []
    for r in arm_rows:
        fl = arrival_floors(r)
        if not fl or r["seed"] in taken:
            continue
        taken.add(r["seed"])
        wanted = [{"id": f"{label}:{r['seed']}|f{f:02d}", "cat": CAT, "kind": "boss", "act": 3, "floor": f, "seed": r["seed"], "ix": int(r["ix"]),
                   "night": label} for _no, f in fl]
        tasks.append(("replay", r["seed"], int(r["ix"]), config, wanted, r))
    return tasks


def pick_order(seed: str, salt: int = PICK_SEED) -> str:
    """The hash order a limited check (`--limit`) takes tasks in: the same rows each time, spread over the blocks."""
    return hashlib.sha256(f"arrival-pick:{salt}:{seed}".encode()).hexdigest()


def mismatch_reasons(rep: dict) -> list[str]:
    """Why a replay did not match its row (the gym's checks: `path_same`, `cards_same`, `rests_same`, every wanted start reached, kinds as wanted);
    [] when it matched, `capped` for a replay the guard ended."""
    if rep.get("same"):
        return []
    if rep.get("cap_reason"):
        return ["capped"]
    out = [name for name, key in (("path", "path_same"), ("cards", "cards_same"), ("rests", "rests_same")) if not rep.get(key)]
    if rep.get("captured") != rep.get("wanted"):
        out.append("start not reached")
    if not out:
        out.append("kind")
    return out


def arrival_records(replays: list[dict], label: str, arm_rows: dict[str, dict], engine: str) -> tuple[list[dict], dict]:
    """The ruler's starts from one arm's replay rows, and the counts. A replay that matched its row (`same`) gives its captured starts, in the gym's
    start schema (`cat` act3-boss, `boss_no` 1 or 2, `cell` act3-boss<turn>-<encounter>, `stratum` own, `source` arrival); one that did not gives
    nothing. A run's own result at the fight (`arm_run_passed`) is kept for the split. Counts: replays, matched, captured (a replay the guard ended),
    mismatched by reason, the starts of each turn."""
    from gym_loop import is_val_seed  # noqa: PLC0415

    records: list[dict] = []
    why: Counter = Counter()
    n = matched = capped = 0
    for rep in replays:
        n += 1
        reasons = mismatch_reasons(rep)
        if reasons:
            capped += reasons == ["capped"]
            why.update(reasons)
            continue
        matched += 1
        seed = str(rep["seed"])
        row = arm_rows.get(seed) or {}
        turns = {f: no for no, f in arrival_floors(row)}
        for s in rep.get("starts") or []:
            no = turns.get(int(s["floor"]))
            if no is None or not s.get("doc") or not s.get("replay_same") or not s.get("kind_same"):
                continue
            enc = int(s["encounter_id"])
            passed = bool(row.get("win")) if no == 2 else k2_went_past(row)
            records.append({**s, "cat": CAT, "kind": "boss", "act": 3, "boss_no": no, "cell": f"act3-boss{no}-{enc}", "stratum": "own", "source": "arrival",
                            "run": seed, "block": label, "engine": engine, "arm_run_passed": passed, "ix": int(rep["ix"]),
                            "split": "val" if is_val_seed(seed) else "train"})
    by_turn = Counter(r["boss_no"] for r in records)
    return records, {"replays": n, "matched": matched, "capped": capped, "mismatched": n - matched, "why": dict(why),
                     "first": by_turn.get(1, 0), "second": by_turn.get(2, 0)}


def k2_went_past(row: dict) -> bool:
    """Whether the judged run went past the first Act 3 boss (the strict reading)."""
    from tools.lab import keep_v2 as k2  # noqa: PLC0415

    return bool(k2.went_past(row, 2))


def ruler_hash(records: list[dict]) -> str:
    """The ruler's hash: the rule's name and, for each start in id order, its id and the sha256 of its document (canonical JSON). The files' layout,
    the order they were written in and the annotations on a start do not change it; a different position does."""
    h = hashlib.sha256(ARRIVAL_RULE.encode())
    for r in sorted(records, key=lambda r: r["id"]):
        doc = json.dumps(r.get("doc"), sort_keys=True, separators=(",", ":"))
        h.update(f"\n{r['id']}\t{hashlib.sha256(doc.encode()).hexdigest()}".encode())
    return h.hexdigest()


def exclusion_of(records: list[dict]) -> dict:
    """The practice's exclusion list for the ruler (`fight_exam.exclusion`'s shape): every run behind a start and every start id. It is written beside the
    ruler (`exclude_runs.json`); `merge_exclusions` puts it into the fight loop's `exam/exclude_runs.json`, which the practice reads at each segment draw."""
    return {"rule": ARRIVAL_RULE, "run_seeds": sorted({str(r["seed"]) for r in records}), "start_ids": sorted(r["id"] for r in records),
            "starts": len(records), "by_stratum": dict(Counter(r.get("stratum", "own") for r in records))}


# The ruler's runs and starts must never be practised on: the exam plays them, and a fight the practice has trained on is no exam. The fight loop reads
# `<loop>/exam/exclude_runs.json` at each segment draw (`practice_mix.read_exclusion`), so the ruler's list is merged into it (judge build 3, step 2; it was
# a hand step, `judge-design/merge_arrival_exclusions.py`, on 2026-11-22).
EXAM_EXCLUDE_RULE = "fight-exam-v1"  # the rule the fight exam's own exclusion file names (`fight_exam.EXAM_RULE`)
EXCLUDE_FILE = "exclude_runs.json"
EXCLUDE_BACKUP = EXCLUDE_FILE + ".pre-arrival.bak"
EXCLUDE_STRATUM = "arrival-own"  # `by_stratum`'s count of the arrival starts merged


def exam_exclusion_file(ruler: str | Path) -> Path:
    """The fight loop's exclusion file for a ruler in `<loop>/arrival/<name>/`: `<loop>/exam/exclude_runs.json`."""
    return Path(ruler).parent.parent / "exam" / EXCLUDE_FILE


def merge_exclusions(ruler: str | Path, exam_file: str | Path | None = None, *, dry: bool = False) -> dict:
    """Merge the ruler's exclusion list (`<ruler>/exclude_runs.json`: `run_seeds`, `start_ids`) into the fight loop's (`exam_file`, by default the loop's
    `exam/exclude_runs.json` beside the ruler's `arrival/` folder), so the practice never draws those fights.

    The union is sorted; `by_stratum["arrival-own"]` counts the arrival starts merged; `arrival[<ruler folder name>]` records the merge (rule, when, the ruler's
    starts and runs, what this merge added, the ruler's hash). The first time the old file is kept as `exclude_runs.json.pre-arrival.bak`; every write is a
    temporary file replaced over the file (`gym_loop.write_json`, which is patient with a reader that has the file open). It is idempotent: a ruler whose
    entry names its present hash and whose ids are all in already changes nothing (no write). A ruler that grew (a new hash) adds its new ids and updates its entry.

    Returns {status, ruler, ruler_hash, starts, runs, added_starts, added_runs, wrote}; `status` is `merged`, `already merged`, or `no exclusion file` (the
    loop has none: nothing to merge into, nothing written). A ruler list or a loop file that is not the rule's own, unreadable, or a ruler list that does not
    match its meta is an error (ValueError, OSError): a file somebody else's is never rewritten on a guess. `dry` reads everything and writes nothing."""
    from gym_loop import write_json  # noqa: PLC0415

    ruler = Path(ruler)
    exam_file = Path(exam_file) if exam_file is not None else exam_exclusion_file(ruler)
    if not exam_file.is_file():
        return {"status": "no exclusion file", "ruler": ruler.name, "wrote": False}
    r = json.loads((ruler / EXCLUDE_FILE).read_text(encoding="utf-8"))
    meta = json.loads((ruler / "meta.json").read_text(encoding="utf-8"))
    raw = exam_file.read_bytes()
    e = json.loads(raw.decode("utf-8"))
    if r.get("rule") != ARRIVAL_RULE:
        raise ValueError(f"{ruler.name}: the ruler's exclusion list names rule {r.get('rule')!r}, not {ARRIVAL_RULE!r}")
    if e.get("rule") != EXAM_EXCLUDE_RULE:
        raise ValueError(f"{exam_file.parent.name}/{exam_file.name} names rule {e.get('rule')!r}, not {EXAM_EXCLUDE_RULE!r}: not the fight exam's file, left as it is")
    rs, ri = [str(x) for x in r.get("run_seeds") or []], [str(x) for x in r.get("start_ids") or []]
    if len(ri) != int(meta.get("starts", -1)) or len(set(ri)) != len(ri):
        raise ValueError(f"{ruler.name}: the ruler's exclusion list holds {len(ri)} start ids and its meta {meta.get('starts')} starts: not merged")
    if not isinstance(e.get("by_stratum", {}), dict) or not isinstance(e.get("arrival", {}), dict):
        raise ValueError(f"{exam_file.name}: `by_stratum` or `arrival` is not a map: left as it is")
    have_s, have_i = {str(x) for x in e.get("run_seeds") or []}, {str(x) for x in e.get("start_ids") or []}
    new_s, new_i = sorted(set(rs) - have_s), sorted(set(ri) - have_i)
    entry = (e.get("arrival") or {}).get(ruler.name)
    summary = {"ruler": ruler.name, "ruler_hash": meta.get("ruler_hash"), "starts": len(ri), "runs": len(rs), "added_starts": len(new_i), "added_runs": len(new_s)}
    if entry is not None and entry.get("ruler_hash") == meta.get("ruler_hash") and not new_s and not new_i:
        return {"status": "already merged", **summary, "wrote": False}
    e["run_seeds"] = sorted(have_s | set(new_s))
    e["start_ids"] = sorted(have_i | set(new_i))
    by = e.setdefault("by_stratum", {})
    by[EXCLUDE_STRATUM] = int(by.get(EXCLUDE_STRATUM, 0)) + len(new_i)
    e.setdefault("arrival", {})[ruler.name] = {"rule": ARRIVAL_RULE, "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "starts": len(ri), "runs": len(rs),
                                               "added_starts": len(new_i), "added_runs": len(new_s), "ruler_hash": meta.get("ruler_hash"),
                                               "why": "the Act 3 arrival exam (route S, S1) plays these starts; the practice must not draw them"}
    if dry:
        return {"status": "merged", **summary, "wrote": False}
    backup = exam_file.with_name(EXCLUDE_BACKUP)
    if not backup.exists():
        backup.write_bytes(raw)
    write_json(exam_file, e)
    return {"status": "merged", **summary, "wrote": True}


def load_ruler(folder: str | Path) -> tuple[dict, dict[str, dict]]:
    """(meta, {start id: lite record}) of a written ruler; the starts' hash is recomputed from the file and must be the meta's (a ruler somebody
    edited is not the ruler the figures name)."""
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    lite = {s["id"]: s for s in json.loads((folder / f"{ARRIVAL_STEM}.lite.json").read_text(encoding="utf-8"))}
    recs = []
    with open(folder / f"{ARRIVAL_STEM}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                recs.append(json.loads(line))
    if ruler_hash(recs) != meta.get("ruler_hash"):
        raise ValueError(f"{folder}: the starts file does not hash to the meta's ruler_hash")
    if set(lite) != {r["id"] for r in recs}:
        raise ValueError(f"{folder}: the lite list and the starts file name different starts")
    return meta, lite


def read_ruler_records(folder: str | Path) -> list[dict]:
    """The full start records of a written ruler (documents included), in file order. `load_ruler` is the check that they are the ruler's own."""
    recs = []
    with open(Path(folder) / f"{ARRIVAL_STEM}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def history_of(meta: dict) -> list[dict]:
    """A ruler's build history: the meta's `history`, or for a ruler written before there was one (one build, no extensions) the single entry it implies."""
    if meta.get("history"):
        return [dict(x) for x in meta["history"]]
    return [{"written": meta.get("written"), "action": "built", "ruler_hash": meta.get("ruler_hash"), "starts": meta.get("starts"), "first": meta.get("first"),
             "second": meta.get("second"), "added": [{"block": x["block"], "rows": x.get("rows"), "arrivals": x.get("arrivals"),
                                                      "starts": int(x.get("first") or 0) + int(x.get("second") or 0), "first": x.get("first"), "second": x.get("second")}
                                                     for x in meta.get("arms") or []]}]


def build_ruler(folder: str | Path, champion: str, champion_dir: str, arms: list[dict], engine: str, config: str, previous: dict | None = None) -> dict:
    """Write the ruler from the arms' kept starts: `arms` [{label, path, records, counts, rows, arrivals}] (`arrival_records` of each; a kept arm of an earlier
    ruler brings its own records and counts back). The starts file, its byte index and lite list (the shape `gym_loop.load_start` reads), `exclude_runs.json`
    and, last, `meta.json` with the ruler's hash, the counts and the history. Returns the meta. A start id that two arms made is an error (the seeds were
    taken once, `arrival_tasks`).

    The meta is written last because it is what makes the ruler usable: `load_ruler` reads a starts file that does not hash to the meta's as no ruler, so a
    write stopped between two files leaves nothing a test could be played on, and the next `arrivals` builds it again from the replays on disk.

    `previous` is the meta of the ruler this one extends (`cmd_arrivals`): its history is carried and an entry added, naming the arms with `added` set (the
    new and the grown ones) and how many starts each brought. A first build adds every arm."""
    folder = Path(folder)
    records = [r for a in arms for r in a["records"]]
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("two arms made the same start")
    n = write_set(folder, ARRIVAL_STEM, records)
    h = ruler_hash(records)
    first = [r for r in records if r["boss_no"] == 1]
    second = [r for r in records if r["boss_no"] == 2]
    by_cell = Counter(r["cell"] for r in records)
    written = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry = {"written": written, "action": "extended" if previous else "built", "ruler_hash": h, "starts": n, "first": len(first), "second": len(second),
             "added": [{"block": a["label"], "rows": a["rows"], "arrivals": a["arrivals"], "starts": len(a["records"]),
                        "first": sum(1 for r in a["records"] if r["boss_no"] == 1), "second": sum(1 for r in a["records"] if r["boss_no"] == 2)}
                       for a in arms if a.get("added", previous is None)]}
    if previous:
        entry["previous_hash"] = previous.get("ruler_hash")
    meta = {"rule": ARRIVAL_RULE, "champion": champion, "champion_dir": champion_dir, "engine": engine, "config": config, "ruler_hash": h,
            "written": written, "starts": n, "first": len(first), "second": len(second),
            "by_cell": dict(sorted(by_cell.items())), "arms": [{"block": a["label"], "rows": a["rows"], "arrivals": a["arrivals"], **a["counts"]} for a in arms],
            "arrivals": sum(a["arrivals"] for a in arms), "matched": sum(a["counts"]["matched"] for a in arms),
            "history": [*(history_of(previous) if previous else []), entry],
            "note": "first-boss starts are the exam's statistic; second-boss starts are played for the record. Replays are kept only when they match "
                    "their row (map path, cards, rests); the arms were played with the prior on the server and replay on the CPU prior."}
    hs.atomic_json(folder / "exclude_runs.json", exclusion_of(records))
    hs.atomic_json(folder / "meta.json", meta)
    return meta


# The exam's games are played on one hash. A ruler that grows has a new hash, and a test index means draws of one hash only, so the games of the old hash are
# put away (under `exam/<old hash first 8>/`) when the hash changes and the folder says which hash the games in it belong to.
EXAM_MARK = "ruler_hash.txt"
EXAM_PER_HASH = ("games-arr-*.jsonl", "report.txt", "results.json")  # the exam's files that belong to one hash (its log and STOP file do not)


def exam_owner(exam_dir: str | Path) -> str | None:
    """The hash the games in an exam folder were played on (`exam/ruler_hash.txt`); None when the folder says nothing (a folder from before the ruler grew:
    its games are taken to belong to the ruler it sits beside)."""
    try:
        return (Path(exam_dir) / EXAM_MARK).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def exam_games(exam_dir: str | Path, ruler_hash_: str | None) -> list[Path]:
    """The games files in an exam folder that count for the ruler with this hash: all of them when the folder names no owner (or `ruler_hash_` is None), none
    when it names another hash. `gen_loop.s1_used_tests` reads the folder through this, and `retire_exam` puts away what it would not count."""
    owner = exam_owner(exam_dir)
    if ruler_hash_ is not None and owner is not None and owner != ruler_hash_:
        return []
    return sorted(Path(exam_dir).glob(EXAM_PER_HASH[0]))


def retire_exam(exam_dir: str | Path, ruler_hash_: str, old_hash: str | None = None) -> list[str]:
    """Make an exam folder the games of `ruler_hash_`'s: files of the exam that belong to one hash (`EXAM_PER_HASH`) move to `<exam_dir>/<owner first 8>/` when
    the owner (the folder's `ruler_hash.txt`, else `old_hash`, the hash the ruler had before it was extended) is another hash, and the owner is written down. A
    folder with no owner and no `old_hash` is a legacy folder of this ruler: it gets the owner file and keeps its games. Returns the moved file names.
    A name already taken in the destination gets a numbered suffix: nothing is ever overwritten."""
    exam_dir = Path(exam_dir)
    if not exam_dir.is_dir():
        return []
    owner = exam_owner(exam_dir) or old_hash
    moved: list[str] = []
    if owner is not None and owner != ruler_hash_:
        dest = exam_dir / owner[:8]
        for pat in EXAM_PER_HASH:
            for f in sorted(exam_dir.glob(pat)):
                dest.mkdir(exist_ok=True)
                target, i = dest / f.name, 1
                while target.exists():
                    target, i = dest / f"{f.name}.{i}", i + 1
                os.replace(f, target)
                moved.append(f.name)
    if exam_owner(exam_dir) != ruler_hash_:
        (exam_dir / EXAM_MARK).write_text(ruler_hash_ + "\n", encoding="utf-8", newline="\n")
    return moved


def replay_arms(arms: list[dict], out: Path, *, agent_root: Path, engine: str | None, procs: int, limit: int, until: str | None, stop_file: Path,
                say=print, taken: set | frozenset = frozenset()) -> tuple[list[dict], str | None]:
    """Replay each arm's arrivals (`fight_gym._replay_task` in `procs` spawned workers; a replay with a row on disk is not played again, so a stop
    loses only the replays in flight). Returns (the arms with `replays` read back, the stop reason). `limit`: only the first `limit` tasks of the
    hash order over all the arms (a check that the arms load, not a ruler). `taken`: seeds an arm already in the ruler holds (an extension): they are
    left to that arm, as a seed two blocks hold is left to the first."""
    import fight_exam as fe  # noqa: PLC0415
    import fight_gym as fg  # noqa: PLC0415

    fg.MAX_PROCS = max(fg.MAX_PROCS, procs)
    tasks_by_arm: list[list[tuple]] = []
    taken = set(taken)
    for a in arms:
        tasks_by_arm.append(arrival_tasks(a["all_rows"], a["label"], a["config"], taken))
        a["arrivals"] = len(tasks_by_arm[-1])
    if limit:
        order = sorted(((pick_order(t[1]), i, t) for i, ts in enumerate(tasks_by_arm) for t in ts), key=lambda x: x[0])[:limit]
        tasks_by_arm = [[t for _k, j, t in order if j == i] for i in range(len(arms))]
    stopped = None
    for a, tasks in zip(arms, tasks_by_arm, strict=True):
        target = out / f"replays-{a['label']}.jsonl"
        done = {r.get("seed") for r in fg.read_jsonl(target)}
        left = [t for t in tasks if t[1] not in done]
        a["tasks"] = tasks
        if left and not stopped:
            say(f"{a['label']}: replaying {len(left)} of {len(tasks)} arrivals on engine {engine or a['engine']} ({procs} processes)")
            stopped = fg.run_tasks(left, fg._replay_task, target, procs, init=fg._init_worker,  # noqa: SLF001
                                   init_args=(engine or a["engine"], str(agent_root), a["agent_dir"], str(out)), mem_limit=fg.MEM_LIMIT_GB * 2**30 * 2,
                                   stop=lambda: fe.blocked_reason(starting=False, until=until, stop_file=stop_file))
        mine = {t[1] for t in tasks}
        a["replays"] = [r for r in fg.read_jsonl(target) if r.get("seed") in mine]
    return arms, stopped


def load_arm(folder: Path) -> dict:
    """An arm folder as the replay step reads it: its summary (the agent folder, the config and the engine it was played on: none named is `old`),
    its rows by seed and its block's label."""
    summ = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (folder / "runs.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    config = summ.get("config") or ARRIVAL_CONFIG
    return {"path": folder, "label": arm_label(folder, config), "config": config, "engine": summ.get("engine") or "old",
            "agent_dir": summ["agent"]["dir"], "all_rows": rows, "rows": len(rows), "by_seed": {r["seed"]: r for r in rows}}


def prior_ruler(out: Path, champion: str, champion_dir: str) -> tuple[dict, list[dict]] | None:
    """(meta, full records) of the ruler already in `out` when it is this champion's and its starts hash to its meta's (`load_ruler`), else None: a folder with
    no ruler, a ruler of another champion, one somebody edited and one whose writing was stopped half way are all built (again) from the arms."""
    try:
        meta, _lite = load_ruler(out)
    except (OSError, ValueError, KeyError):
        return None
    if meta.get("champion") != champion or meta.get("champion_dir") != champion_dir:
        return None
    return meta, read_ruler_records(out)


def carried_arm(entry: dict, records: list[dict]) -> dict:
    """An arm of an earlier ruler kept as it was: its starts (the records of its block) and its counts, from the meta's entry. Nothing is replayed for it."""
    counts = {k: v for k, v in entry.items() if k not in ("block", "rows", "arrivals")}
    return {"label": entry["block"], "records": [r for r in records if r.get("block") == entry["block"]], "counts": counts, "rows": entry["rows"],
            "arrivals": entry["arrivals"]}


def cmd_arrivals(a: argparse.Namespace) -> int:
    """Build the champion's arrival ruler, or extend the one in `--out` (with `--limit`, only check that the judge's arms replay).

    A ruler already in `--out` that is this champion's (and `--rebuild` is not given) is extended: the arms it has an entry for, with no more rows than the
    entry says, are carried over with their own starts; the arms it lacks, and those that hold more rows now, are replayed (a replay with a row on disk is not
    played again); the starts are written again with the hash of the whole set. The exam's games of the old hash are put away (`retire_exam`). With nothing to
    add the folder is left as it is. Status: 0 built, extended or up to date; 2 no arm to read (no folder is made), or arms another agent played; 3 stopped
    or blocked (what was replayed is kept; run again)."""
    import fight_exam as fe  # noqa: PLC0415

    champion_dir = Path(a.champion)
    out = Path(a.out) if a.out else ARRIVAL_ROOT / f"{champion_dir.parent.name}-{champion_dir.name}"
    name, rel = f"{champion_dir.parent.name}-{champion_dir.name}", f"{champion_dir.parent.name}/{champion_dir.name}"
    folders = [Path(x) for x in a.arms] if a.arms else champion_arms(champion_dir, a.exclude_block)
    if not folders:
        print(f"no arm to read under {champion_dir / 'arms'}")
        return 2
    if a.lib:
        os.environ["STS2_LIB_PATH"] = a.lib
    arms = [load_arm(f) for f in folders]
    foreign = [x["label"] for x in arms if x["agent_dir"].replace("\\", "/") != rel]
    if foreign:
        print(f"these arms were not played by {rel}: {', '.join(foreign)}")
        return 2
    existing = prior_ruler(out, name, rel)
    prior = None if a.rebuild or a.limit else existing
    todo, carried = arms, []
    if prior is not None:
        pmeta, precs = prior
        have = ruler_arms(pmeta)
        todo = [x for x in arms if x["label"] not in have or x["rows"] > int(have[x["label"]].get("rows") or 0)]
        mine = {x["label"] for x in todo}
        carried = [carried_arm(e, precs) for lab, e in have.items() if lab not in mine]
        if not todo:
            print(json.dumps({"action": "up to date", **{k: pmeta[k] for k in ("starts", "first", "second", "ruler_hash", "arrivals", "matched")}}, indent=1))
            return 0
    fe.ALLOW_QUIET[:] = list(a.allow_quiet or [])
    until = a.until or None
    stop_file = out / "STOP"
    why = fe.blocked_reason(starting=True, until=until, stop_file=stop_file)
    if why:
        print(f"not started: {why}")
        return 3
    try:
        fe.prepare_process()
    except fe.Blocked as why2:
        print(f"not started: {why2}")
        return 3
    out.mkdir(parents=True, exist_ok=True)  # the folder is made once there is something to write in it
    taken: set = set()
    replaying = {x["label"] for x in todo}
    for x in arms:  # a seed an arm the ruler keeps holds is left to that arm
        if x["label"] not in replaying:
            arrival_tasks(x["all_rows"], x["label"], x["config"], taken)
    taken |= {str(r["run"]) for c in carried for r in c["records"]}
    t0 = time.time()
    todo, stopped = replay_arms(todo, out, agent_root=Path(a.agent_root), engine=a.engine or None, procs=a.procs, limit=a.limit, until=until,
                                stop_file=stop_file, say=lambda m: print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {m}", flush=True), taken=taken)
    for arm in todo:
        recs, c = arrival_records(arm["replays"], arm["label"], arm["by_seed"], a.engine or arm["engine"])
        secs = [float(r["seconds"]) for r in arm["replays"] if r.get("seconds") is not None]
        c["seconds_mean"] = round(sum(secs) / len(secs), 1) if secs else None
        arm["records"], arm["counts"] = recs, c
    tried = {arm["label"]: {"arrivals": arm["arrivals"], "rows": arm["rows"], "tried": len(arm["tasks"]), **arm["counts"]} for arm in todo}
    if a.limit:
        summary = {"rule": ARRIVAL_RULE, "mode": "check", "limit": a.limit, "engine": a.engine or arms[0]["engine"], "blocks": tried,
                   "stopped": stopped, "wall_s": round(time.time() - t0, 1)}
        hs.atomic_json(out / "verify.json", summary)
        print(json.dumps(summary, indent=1))
        return 3 if stopped else 0
    if stopped or any(len(arm["replays"]) < len(arm["tasks"]) for arm in todo):
        print(f"stopped ({stopped}); the replays on disk are kept, run again to resume")
        return 3
    for arm in todo:
        arm["added"] = True
    final = sorted([*carried, *todo], key=lambda x: x["label"])
    engine = a.engine or (prior[0]["engine"] if prior else arms[0]["engine"])
    config = prior[0]["config"] if prior else arms[0]["config"]
    meta = build_ruler(out, name, rel, final, engine, config, previous=prior[0] if prior else None)
    moved: list[str] = []
    if existing is not None and existing[0]["ruler_hash"] != meta["ruler_hash"]:  # a new hash: the games of the old one are put away
        moved = retire_exam(out / "exam", meta["ruler_hash"], existing[0]["ruler_hash"])
    last = meta["history"][-1]
    print(json.dumps({**{k: meta[k] for k in ("starts", "first", "second", "ruler_hash", "arrivals", "matched", "by_cell")}, "action": last["action"],
                      "added": [f"{x['block']}: {x['starts']} starts" for x in last["added"]],
                      **({"previous_hash": last["previous_hash"], "exam_files_put_away": moved} if "previous_hash" in last else {})}, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------------------- the arrival exam


def policy_sha(player_dir: str | Path) -> str:
    """The sha256 of the player's combat policy: the fight network's identity (a gym player and a run candidate that hold the same one are the same player)."""
    h = hashlib.sha256()
    with open(Path(player_dir) / "combat_policy.pt", "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arrival_game_tasks(starts_file: Path, lite: dict[str, dict], who: str, config: str, done: set, js: tuple[int, ...]) -> list:
    """The exam games `who` still has to play (`fight_exam.exam_game_steps`' tasks in the arrival namespace), draw by draw, long fights first (the
    boss fights are all long: the order is the draw, then the start id)."""
    info = {"starts": str(starts_file), "player": who, "config": config, "ns": ARRIVAL_NS, "label": who}
    tasks = [("a3-v1", sid, j, s["cat"], info) for sid, s in lite.items() for j in js if (sid, j) not in done]
    tasks.sort(key=lambda t: (t[2], t[1]))
    return tasks


def cmd_merge_exclusions(a: argparse.Namespace) -> int:
    """Merge a ruler's exclusion list into the fight loop's (what the train job's S1 stage does by itself; this is the hand way). Status 0 merged or already
    merged, 2 nothing to merge into or a file that is not the rule's own."""
    try:
        rec = merge_exclusions(a.ruler, a.exam_file or None, dry=a.dry)
    except (OSError, ValueError) as why:
        print(f"not merged: {why}")
        return 2
    print(json.dumps(rec, indent=1))
    return 2 if rec["status"] == "no exclusion file" else 0


def cmd_arrival_exam(a: argparse.Namespace) -> int:
    """Play the ruler under the candidate's and the champion's fight network (config a3-v1, CPU prior, 6 draws a start, test k on draws 6k..6k+5) and
    read S1: a report; no rule reads it."""
    import fight_exam as fe  # noqa: PLC0415
    import fight_gym as fg  # noqa: PLC0415
    import gym_gate as gg  # noqa: PLC0415
    import gym_loop as gl  # noqa: PLC0415

    ruler = Path(a.ruler)
    meta, lite = load_ruler(ruler)
    cand_dir, champ_dir = Path(a.candidate), Path(a.champion)
    shas = {"candidate": policy_sha(cand_dir), "champion": policy_sha(champ_dir)}
    if shas["candidate"] == shas["champion"]:
        print("the candidate and the champion hold the same fight network: nothing to compare")
        return 2
    names = {"candidate": f"arr-cand-{shas['candidate'][:8]}", "champion": f"arr-champ-{shas['champion'][:8]}"}
    out = Path(a.out) if a.out else ruler / "exam"
    out.mkdir(parents=True, exist_ok=True)
    moved = retire_exam(out, meta["ruler_hash"])  # games of another hash in this folder are not this ruler's: put away, never read as done
    if moved:
        print(f"the games in {out.name} were played on another ruler hash: {len(moved)} files put away")
    js = gg.arrival_js(a.test, a.draws)
    if a.lib:
        os.environ["STS2_LIB_PATH"] = a.lib
    fe.ALLOW_QUIET[:] = list(a.allow_quiet or [])
    until = a.until or None
    stop_file = out / "STOP"
    why = fe.blocked_reason(starting=True, until=until, stop_file=stop_file)
    if why:
        print(f"not started: {why}")
        return 3
    try:
        fe.prepare_process()
    except fe.Blocked as why2:
        print(f"not started: {why2}")
        return 3
    eng = gl.engine_tag()
    starts_file = ruler / f"{ARRIVAL_STEM}.jsonl"
    games = {w: out / f"games-{names[w]}.jsonl" for w in names}
    stop = SimpleStop(lambda: fe.blocked_reason(starting=False, until=until, stop_file=stop_file))
    stopped = None
    for w in ("champion", "candidate"):  # the champion first: its games are the reference of every later candidate
        done = {(r["start"], int(r["j"])) for r in fg.read_jsonl(games[w]) if r.get("engine") == eng and r.get("config", ARRIVAL_CONFIG) == ARRIVAL_CONFIG}
        tasks = arrival_game_tasks(starts_file, lite, names[w], ARRIVAL_CONFIG, done, js)
        if not tasks:
            continue
        fe.say(out, f"arrival exam: {names[w]} plays {len(tasks)} games ({len(lite)} starts, draws {js[0]}-{js[-1]}, {ARRIVAL_CONFIG}, CPU prior, {a.procs} processes)")
        stopped = fe.play_exam_tasks(out, stop, tasks, games[w], champ_dir if w == "champion" else cand_dir, names[w], procs=a.procs, k=a.k, prior="cpu")
        n = gl.stamp_capped(games[w], tasks, lite, eng, ARRIVAL_CONFIG)
        if n:
            fe.say(out, f"arrival exam: {n} capped games of {names[w]} counted as lost")
        if stopped:
            break
    idx = {w: gg.indexed([r for r in fg.read_jsonl(games[w]) if r.get("start") in lite and r.get("engine") == eng and r.get("config", ARRIVAL_CONFIG) == ARRIVAL_CONFIG],
                         lambda j: j in set(js)) for w in names}
    first = {sid for sid, s in lite.items() if int(s["boss_no"]) == 1}
    second = {sid for sid, s in lite.items() if int(s["boss_no"]) == 2}
    cells = {sid: s["cell"] for sid, s in lite.items()}
    reading = gg.arrival_exam(idx["candidate"], idx["champion"], ids=first, test=a.test, draws=a.draws, ruler_hash=meta["ruler_hash"], cells=cells)
    record = gg.arrival_exam(idx["candidate"], idx["champion"], ids=second, test=a.test, draws=a.draws, ruler_hash=meta["ruler_hash"], cells=cells)
    result = {"rule": ARRIVAL_RULE, "config": ARRIVAL_CONFIG, "prior": "cpu", "engine": eng, "ruler": str(ruler.name), "ruler_hash": meta["ruler_hash"],
              "candidate": {"dir": cand_dir.name, "policy_sha256": shas["candidate"]}, "champion": {"dir": champ_dir.name, "policy_sha256": shas["champion"]},
              "test": a.test, "draws": list(js), "stopped": stopped, "first_boss": reading, "second_boss_record": record}
    text = gg.arrival_report(result)
    report_dir = Path(a.report_dir) if a.report_dir else out
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
    hs.atomic_json(report_dir / "results.json", result)
    print(text)
    return 3 if stopped or not reading["complete"] else 0


class SimpleStop:
    """What `fight_exam.play_exam_tasks` asks of its lock: a `stop_reason()` (the run is not under an `ExamLock`: its caller holds the QUIET line)."""

    def __init__(self, fn) -> None:
        self.stop_reason = fn


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
    p = sub.add_parser("decks")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_decks)
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
    p = sub.add_parser("arrivals")
    p.add_argument("--champion", required=True, help="the champion's agent folder (the arms are <champion>/arms/*; the arms were played by it)")
    p.add_argument("--out", default="", help="the ruler's folder (default: gym-loop/arrival/<generation>-<folder>)")
    p.add_argument("--arms", nargs="*", default=[], help="arm folders to read (default: the champion's arms except the blocks it was selected on)")
    p.add_argument("--exclude-block", nargs="*", default=[], help="more blocks to leave out")
    p.add_argument("--agent-root", default=str(AGENT_ROOT))
    p.add_argument("--engine", default="", help="the engine to replay on (default: the one each arm was played on: `old` when its summary names none)")
    p.add_argument("--procs", type=int, default=12)
    p.add_argument("--limit", type=int, default=0, help="only check that this many arrivals replay (no ruler is written; verify.json)")
    p.add_argument("--rebuild", action="store_true", help="ignore the ruler already in --out (default: it is extended by the arms it lacks)")
    p.add_argument("--until", default="", help="a UTC stamp after which no replay starts")
    p.add_argument("--allow-quiet", nargs="*", default=[], help="QUIET.lock tags to ignore besides the loop's (the caller's own line)")
    p.add_argument("--lib", default="")
    p.set_defaults(fn=cmd_arrivals)
    p = sub.add_parser("merge-exclusions")
    p.add_argument("--ruler", required=True, help="the ruler's folder (its exclude_runs.json and meta.json)")
    p.add_argument("--exam-file", default="", help="the fight loop's exclusion file (default: exam/exclude_runs.json beside the ruler's arrival/ folder)")
    p.add_argument("--dry", action="store_true", help="read both and say what would be added; write nothing")
    p.set_defaults(fn=cmd_merge_exclusions)
    p = sub.add_parser("arrival-exam")
    p.add_argument("--ruler", required=True)
    p.add_argument("--candidate", required=True, help="a folder with the candidate's combat_policy.pt and model.json (a gym player or a run candidate)")
    p.add_argument("--champion", required=True, help="the champion's folder, the same way")
    p.add_argument("--test", type=int, default=0, help="test k plays draws 6k to 6k + 5")
    p.add_argument("--draws", type=int, default=ARRIVAL_DRAWS)
    p.add_argument("--procs", type=int, default=12)
    p.add_argument("--k", type=int, default=32, help="games a process plays at once")
    p.add_argument("--out", default="", help="where the games are kept (default: <ruler>/exam)")
    p.add_argument("--report-dir", default="", help="where report.txt and results.json go (default: --out)")
    p.add_argument("--until", default="")
    p.add_argument("--allow-quiet", nargs="*", default=[])
    p.add_argument("--lib", default="")
    p.set_defaults(fn=cmd_arrival_exam)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
