"""The fight judge's data (D41, judge v0): fight starts from other nights, played out by a3-v1 with every position recorded.

    python rl/judge_data.py starts --out DATA --source NIGHT_ARM [--source ...] --runs 100 [--procs 6]
    python rl/fight_gym.py play --out DATA --arm a3-v1 --cat all --seeds 3 --lockstep 32 --prior server --record \
                                --games-name judge_games.jsonl
    python rl/judge_data.py check --out DATA            # the contamination check; writes DATA/manifest.json
    python rl/judge_data.py build --out DATA            # positions and labels as one npz for rl/judge.py
    python rl/judge_data.py buffer --out DATA [--exclude META.json ...]   # judge v1: the gym loop's buffer as one npz (see below)

What it is. `starts` replays a night arm's runs exactly (`fight_gym._replay_task`, the M2 gym's replay with that night's agent
and engine) and keeps every fight of Acts 1 and 2 on the run's path: hallway fights (map letter M), elites (E) and bosses (B),
each as the position at its first combat decision. `fight_gym play --record` then plays each start under a3-v1 with the exam's
agent and writes, beside each game's row, the tokens of every decision position (`pack_positions`), so a position's label is the
outcome of the game it was in: won, end HP as a fraction of max HP, potions still held. `build` joins the two.

Contamination. The 200 Act-1 boss starts of the fight board (2026-10-29 M2, replayed from the 2026-10-27 g7 night) are the exam.
No run of that night is replayed here (`starts` refuses the board's source arm), no start whose run seed is a board start's seed
is kept, and `check` proves both on the files and records the ids. A fifth of the training starts, by a hash of the start id, is
held out for the judge's own accuracy number (`split_of`); a start's every seed and position sits on one side.

Judge v1 (`buffer`, 2026-11-08, training design 1.3's fallback). The always-on loop's replay buffer already holds the x16 teacher's
fights with a tokens row per decision and the fight's outcome in the shard's `fights_meta` (won, end HP, max HP, potions, L0):
`read_buffer` turns every usable fight (finished, `value_ok`) into the same npz `build` writes, minus the exam's and the boards'
runs (`exam/exclude_runs.json`, the M2 board, and any board meta given by `--exclude`). The held-out fifth is chosen by RUN
(`split_of` on the run seed), not by start: a run's fights share a deck, and a start-level split puts the same deck on both sides.
`judge.Positions` takes its validation tenth by run too when the npz has `start_run`.

Lab code: nothing here changes the engine, the search or the solver.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BOARD_STARTS = Path("D:/spireline/referee/lab/results/fight-gym/2026-10-29-m2/starts.jsonl")
BOARD_CAT = "act1-boss"
BOARD_SOURCE = "nightly/2026-10-27/g7-emu"  # the night the board's starts were replayed from
EXAM_AGENT = "gen7/candidate-r1"  # the agent the board's games were played with: the judge learns that player's outcomes
EXAM_ENGINE = "core2"
HELD_OUT = 5  # one start in five, by the hash of its id
FIGHT_LETTERS = {"M": "hall", "E": "elite", "B": "boss"}
ACT_FLOORS = {1: range(2, 18), 2: range(18, 34)}  # map floors of each act (the run row's `nodes`: index i is floor i + 2)
META = ("k", "turn", "hp", "max_hp", "root_value", "best_value", "best_p_death", "sims")


# ---------------------------------------------------------------------------------------------------------- pure parts


def split_of(start_id: str) -> str:
    """"heldout" for one start in HELD_OUT (by the sha256 of its id), else "train"."""
    return "heldout" if int(hashlib.sha256(start_id.encode()).hexdigest()[:8], 16) % HELD_OUT == 0 else "train"


def run_seed_of(start_id: str) -> str:
    return start_id.split("|")[0]


def act_of_floor(floor: int) -> int | None:
    return next((a for a, fl in ACT_FLOORS.items() if floor in fl), None)


def select_fight_starts(runs: list[dict], n_runs: int, seed: int, exclude_seeds: set[str], acts=(1, 2)) -> list[dict]:
    """Every hallway, elite and boss fight of Acts `acts` on the path of `n_runs` runs drawn (seeded) from the eligible runs whose
    seed is not excluded. A fight counts when the run reached its floor. Each start: {id, cat, kind, act, floor, seed, ix}, cat
    "act<a>-<hall|elite|boss>"; runs in the draw's order, fights by floor."""
    ok = sorted((r for r in runs if not r.get("crash") and r.get("end") in ("death", "win") and r.get("nodes")
                 and str(r["seed"]) not in exclude_seeds), key=lambda r: int(r["ix"]))
    picked = random.Random(seed).sample(ok, min(n_runs, len(ok)))
    out = []
    for r in picked:
        for i, letter in enumerate(r["nodes"]):
            floor = i + 2
            act = act_of_floor(floor)
            kind = FIGHT_LETTERS.get(letter)
            if kind is None or act not in acts or floor > int(r["floor"]):
                continue
            out.append({"id": f"{r['seed']}|f{floor:02d}", "cat": f"act{act}-{kind}", "kind": kind, "act": act, "floor": floor,
                        "seed": r["seed"], "ix": int(r["ix"])})
    return out


def board_ids(path: Path = BOARD_STARTS) -> list[str]:
    return [s["id"] for s in _jsonl(path) if s.get("cat") == BOARD_CAT]


def doc_sha(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def contamination(train_ids: list[str], exam_ids: list[str], train_docs: dict | None = None, exam_docs: dict | None = None) -> dict:
    """What the training starts share with the exam: start ids, run seeds and (when given, id -> sha256 of the position document)
    identical positions. Every count must be 0."""
    exam_seeds = {run_seed_of(i) for i in exam_ids}
    out = {"train_starts": len(train_ids), "exam_starts": len(exam_ids),
           "same_id": len(set(train_ids) & set(exam_ids)),
           "same_run_seed": sum(run_seed_of(i) in exam_seeds for i in train_ids)}
    if train_docs is not None and exam_docs is not None:
        out["same_position"] = len(set(train_docs.values()) & set(exam_docs.values()))
    return out


def pack_positions(records: list[dict]) -> str:
    """A game's decision positions as one compact string: each record is {"tokens": combat_tokens.Tokens, and META's numbers}."""
    lens = np.array([len(r["tokens"]) for r in records], dtype=np.uint16)
    cat = (lambda f, dt: np.concatenate([getattr(r["tokens"], f) for r in records]).astype(dt)) if records else (lambda f, dt: np.zeros(0, dt))
    buf = io.BytesIO()
    np.savez_compressed(buf, lens=lens, types=cat("types", np.uint8), ids=cat("ids", np.int32), owners=cat("owners", np.uint8),
                        feats=(np.concatenate([r["tokens"].feats for r in records]).astype(np.float32) if records else np.zeros((0, 18), np.float32)),
                        meta=np.array([[float(r.get(m) if r.get(m) is not None else np.nan) for m in META] for r in records], dtype=np.float32).reshape(-1, len(META)))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def unpack_positions(blob: str) -> dict:
    """`pack_positions`'s arrays: lens [n], types/ids/owners [T], feats [T, 18], meta [n, len(META)]."""
    with np.load(io.BytesIO(base64.b64decode(blob))) as z:
        return {k: z[k] for k in z.files}


def position_record(world, k: int, turn: int, res) -> dict:
    """One decision's record for `pack_positions`: the tokens the judge reads (no end-turn probe) and what the search said."""
    from prior_client import encode_position  # noqa: PLC0415 -- the emulator library, which a gym process has

    vis = sum(c.visits for c in res.children)
    best = res.best
    return {"tokens": encode_position(world.handle, False), "k": k, "turn": turn, "hp": world.hp(), "max_hp": world.max_hp(),
            "root_value": (sum(c.value_sum for c in res.children) / vis) if vis else None,
            "best_value": None if best.value_mean != best.value_mean else best.value_mean,
            "best_p_death": None if best.p_death != best.p_death else best.p_death, "sims": res.sims}


def labels_of(row: dict) -> dict:
    """A game's outcome as the judge's targets: won, end HP / max HP (0 when lost), potions held at the end (0 when lost)."""
    won = bool(row.get("won"))
    max_hp = max(1, int(row.get("max_hp") or 1))
    return {"won": won, "hp_frac": (int(row.get("end_hp") or 0) / max_hp) if won else 0.0,
            "potions": int(row.get("potions_held") or 0) if won else 0, "L0": float(row.get("L0") or 0.0)}


def _jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    out = []
    with open(p, encoding="utf-8") as fh:
        for x in fh:
            if x.strip():
                try:
                    out.append(json.loads(x))
                except ValueError:
                    pass
    return out


# ---------------------------------------------------------------------------------------------------------- the loop's buffer (v1)

LOOP = Path("D:/spireline/referee/gym-loop")
SRCS = ("night", "human", "preloss")


def start_source(start_id: str) -> str:
    """Where a buffer fight's start came from: a "preloss:" id is a pre-loss start, "human:" a rebuilt human start, anything else
    a night's replay."""
    return "preloss" if start_id.startswith("preloss:") else "human" if start_id.startswith("human:") else "night"


def bare_run_of(start_id: str) -> str:
    """The run seed inside any start id: "SEED|f17", "night/gN:SEED|f17", "human:RUN|f17" and "preloss:human:RUN|f33|j0|k24"."""
    return start_id.split("|")[0].split(":")[-1]


def buffer_labels(meta: dict) -> dict:
    """A buffer fight's outcome as the judge's targets (`labels_of`'s), from the fight's meta in its shard. Shards from before
    2026-11-04 carry no `potions_held`: it is then the potions the fight began with less those it used (the same in 97.5 of 100
    fights that have both); a lost fight holds none."""
    won = bool(meta.get("won"))
    max_hp = max(1, int(meta.get("max_hp") or 1))
    held = meta.get("potions_held")
    if held is None:
        held = max(0, int(meta.get("start_potions") or 0) - int(meta.get("potions_used") or 0))
    return {"won": won, "hp_frac": (int(meta.get("end_hp") or 0) / max_hp) if won else 0.0, "potions": int(held) if won else 0,
            "L0": float(meta.get("L0") or 0.0)}


def buffer_usable(meta: dict) -> bool:
    """A fight whose outcome is a label: it finished (not capped) and its segment was whole (`value_ok`: a cut segment's fights lean
    towards quick deaths and keep no value label)."""
    return bool(meta.get("value_ok")) and not meta.get("capped")


def excluded_from(path: Path) -> tuple[set[str], set[str]]:
    """(run seeds, start ids) a JSON file names as out of bounds: the fight exam's `exclude_runs.json` (`run_seeds`, `start_ids`) or a
    board's meta (any key ending `_starts`: a list of start ids or a dict of such lists; a start id's run is the part before `|`)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    runs, ids = {str(x) for x in doc.get("run_seeds", [])}, {str(x) for x in doc.get("start_ids", [])}
    for key, val in doc.items():
        if not key.endswith("_starts"):
            continue
        for lst in (val.values() if isinstance(val, dict) else [val]):
            for sid in lst:
                ids.add(str(sid))
                runs.add(bare_run_of(str(sid)))
    return runs, ids


def root_values(visits: np.ndarray, qvals: np.ndarray) -> np.ndarray:
    """The search's value at each row: the visit-weighted mean of the root children's values (NaN for a forced move)."""
    v = np.where(np.isfinite(qvals) & (visits > 0), visits, 0.0).astype(np.float64)
    tot = v.sum(1)
    acc = (v * np.where(v > 0, qvals, 0.0)).sum(1)
    return np.where(tot > 0, acc / np.maximum(tot, 1e-12), np.nan)


def read_buffer(loop: Path, skip_runs: set[str] = frozenset(), skip_ids: set[str] = frozenset()) -> tuple[dict, dict]:
    """Every usable fight of the loop's replay buffer as the judge's positions (the npz layout `build` writes, plus `start_run` and
    `start_src`) and a report of what was kept and left out. A position is a decision of a fight, labelled with the fight's outcome
    (`buffer_labels`); the tokens are the root prior's (`probe=False`: what the judge reads). A run is held out whole (`split_of`
    on the run seed): a run's fights share a deck, so a start-level split would leak it into the held-out fifth."""
    index = _jsonl(loop / "buffer" / "index.jsonl")
    parts = {k: [] for k in ("lens", "types", "ids", "owners", "feats", "meta", "lab", "grp")}
    start_ix: dict[str, int] = {}
    start_info: dict[str, dict] = {}
    games: list[tuple[int, int, int]] = []
    skipped = Counter()
    for entry in index:
        path = loop / "buffer" / "shards" / entry["shard"]
        if not path.is_file():
            skipped["shard missing"] += 1
            continue
        with np.load(path) as z:
            metas = json.loads(str(z["fights_meta"]))
            fight, lengths = z["fight"], z["lengths"]
            ok = np.zeros(len(metas), dtype=bool)
            for i, f in enumerate(metas):
                if not buffer_usable(f):
                    skipped["capped" if f.get("capped") else "no value label"] += 1
                elif str(f["seed"]) in skip_runs or f["start"] in skip_ids:
                    skipped["excluded run or start"] += 1
                else:
                    ok[i] = True
            rows = ok[fight]
            if not rows.any():
                continue
            fi = fight[rows]
            toks = np.repeat(rows, lengths)
            lab = np.array([[float(x) for x in (lb["won"], lb["hp_frac"], lb["potions"], lb["L0"])]
                            for lb in (buffer_labels(f) for f in metas)], np.float32)
            max_hp = np.array([max(1, int(f.get("max_hp") or 1)) for f in metas], np.float32)
            gi_of = {}
            for i, f in enumerate(metas):
                if ok[i]:
                    sid = f["start"]
                    gi_of[i] = start_ix.setdefault(sid, len(start_ix))
                    start_info.setdefault(sid, {"cat": f["cat"], "run": str(f["seed"]), "src": start_source(sid)})
                    games.append((gi_of[i], int(f["j"]), int((fi == i).sum())))
            meta = np.full((int(rows.sum()), len(META)), np.nan, np.float32)
            meta[:, 0] = z["k"][rows]
            meta[:, 3] = max_hp[fi]
            meta[:, 4] = root_values(z["visits"][rows], z["qvals"][rows])
            meta[:, 7] = z["sims"][rows]
            parts["lens"].append(lengths[rows].astype(np.uint16))
            parts["types"].append(z["types"][toks].astype(np.uint8))
            parts["ids"].append(z["ids"][toks].astype(np.int32))
            parts["owners"].append(z["owners"][toks].astype(np.uint8))
            parts["feats"].append(z["feats"][toks].astype(np.float32))
            parts["meta"].append(meta)
            parts["lab"].append(lab[fi])
            parts["grp"].append(np.array([gi_of[i] for i in fi], np.int32))
    names = sorted(start_ix, key=start_ix.get)
    cats = sorted({start_info[s]["cat"] for s in names})
    cat = lambda k: np.concatenate(parts[k]) if parts[k] else None  # noqa: E731
    out = {"lens": cat("lens"), "types": cat("types"), "ids": cat("ids"), "owners": cat("owners"), "feats": cat("feats"),
           "meta": cat("meta"), "labels": cat("lab"), "group": cat("grp"),
           "start_heldout": np.array([split_of(start_info[s]["run"]) == "heldout" for s in names], dtype=bool),
           "start_ids": np.array(names), "start_cat": np.array([cats.index(start_info[s]["cat"]) for s in names], dtype=np.int16),
           "cats": np.array(cats), "games": np.array(games, dtype=np.int32).reshape(-1, 3),
           "start_run": np.array([start_info[s]["run"] for s in names]),
           "start_src": np.array([SRCS.index(start_info[s]["src"]) for s in names], dtype=np.int8), "srcs": np.array(SRCS)}
    runs = {start_info[s]["run"] for s in names}
    report = {"shards": len(index), "window": [index[0]["t"], index[-1]["t"]] if index else None,
              "segments": [index[0]["segment"], index[-1]["segment"]] if index else None, "skipped_fights": dict(skipped),
              "fights": len(games), "starts": len(names), "runs": len(runs), "heldout_runs": sum(split_of(r) == "heldout" for r in runs),
              "positions": 0 if out["labels"] is None else int(len(out["labels"])),
              "by_cat": dict(sorted(Counter(start_info[s]["cat"] for s in names).items())),
              "by_src": dict(sorted(Counter(start_info[s]["src"] for s in names).items()))}
    return out, report


def run_contamination(positions: dict, exam_ids: list[str]) -> dict:
    """What the buffer's starts share with an exam's start ids: an id, or a run (the id's part before `|`). Both must be 0."""
    exam_runs = {bare_run_of(i) for i in exam_ids}
    return {"exam_starts": len(set(exam_ids)), "same_id": len(set(positions["start_ids"].tolist()) & set(exam_ids)),
            "same_run": len(set(positions["start_run"].tolist()) & exam_runs)}


# ---------------------------------------------------------------------------------------------------------- commands


def cmd_buffer(a) -> int:
    """The loop's buffer as one npz for `judge.py train` (v1), with a manifest: the window, the counts, the exclusions and the
    contamination check against the boards' start lists."""
    from sts2_gym import procutil  # noqa: PLC0415

    procutil.set_priority_class(procutil.BELOW_NORMAL_PRIORITY_CLASS)
    loop = Path(a.loop)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs: set[str] = set()
    ids: set[str] = set()
    exam_ids: list[str] = []
    files = [Path(p) for p in a.exclude] + ([loop / "exam" / "exclude_runs.json"] if (loop / "exam" / "exclude_runs.json").is_file() else [])
    for p in files:
        r, i = excluded_from(p)
        runs |= r
        ids |= i
    board = Path(a.board)
    if board.is_file():
        exam_ids = board_ids(board)
        runs |= {bare_run_of(i) for i in exam_ids}
        ids |= set(exam_ids)
    arrays, rep = read_buffer(loop, runs, ids)
    if arrays["labels"] is None:
        raise SystemExit("the buffer has no usable fight")
    rep["contamination"] = run_contamination(arrays, sorted(ids))
    rep["excluded"] = {"runs": len(runs), "start_ids": len(ids), "files": [p.name for p in files] + ([board.name] if board.is_file() else [])}
    np.savez_compressed(out / a.name, **arrays)
    rep["file"] = a.name
    rep["sha256"] = hashlib.sha256((out / a.name).read_bytes()).hexdigest()
    rep["heldout_positions"] = int(arrays["start_heldout"][arrays["group"]].sum())
    (out / "manifest.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep))
    bad = {k: v for k, v in rep["contamination"].items() if k in ("same_id", "same_run") and v}
    return 1 if bad else 0


def cmd_starts(a) -> int:
    import fight_gym as fg  # noqa: PLC0415
    from tools.solver import coverage  # noqa: PLC0415

    fg.TAG = a.tag  # this job's own QUIET.lock line (fight_gym.guarded_run holds it while computing)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exam = board_ids(Path(a.board))
    exclude = {run_seed_of(i) for i in exam}
    every: list[dict] = []
    for source in a.source:
        label = coverage.source_label(source)
        if label == BOARD_SOURCE:
            raise SystemExit(f"{label} is the board's source night: its runs are the exam's, never training")
        summ, runs, agent_dir, engine = coverage._arm(source)  # noqa: SLF001
        sub = out / ("src-" + label.replace("nightly/", "").replace("/", "-"))
        sub.mkdir(parents=True, exist_ok=True)
        picks = select_fight_starts(runs, a.runs, a.seed, exclude)
        meta = {"source": label, "engine": engine, "agent": agent_dir, "agent_root": a.agent_root, "config": summ.get("config") or "a3-v1",
                "seed": a.seed, "runs": a.runs, "picked": len(picks)}
        (sub / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        by_run: dict[str, list[dict]] = {}
        for p in picks:
            by_run.setdefault(p["seed"], []).append(p)
        rows = {r["seed"]: r for r in runs}
        done = {r["seed"] for r in fg.read_jsonl(sub / "replays.jsonl")}
        tasks = [("replay", sd, int(ws[0]["ix"]), meta["config"], ws, rows[sd]) for sd, ws in by_run.items() if sd not in done]
        fg.guarded_run(sub, "starts", tasks, fg._replay_task, sub / "replays.jsonl", a.procs, fg._init_args(sub))  # noqa: SLF001
        reps = fg.read_jsonl(sub / "replays.jsonl")
        got = [{**s, "source": label} for r in reps for s in r.get("starts", [])]
        every += got
        print(json.dumps({"source": label, "agent": agent_dir, "engine": engine, "runs": len(by_run), "replays": len(reps),
                          "replays_same": sum(bool(r.get("same")) for r in reps), "starts": len(got),
                          "by_cat": dict(Counter(s["cat"] for s in got))}))
    every = [s for s in every if run_seed_of(s["id"]) not in exclude]
    with open(out / "starts.jsonl", "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(s) + "\n" for s in every))
    meta = {"what": "judge v0 training starts (D41): every fight of Acts 1-2 on replayed runs of other nights",
            "engine": EXAM_ENGINE, "agent": EXAM_AGENT, "agent_root": a.agent_root, "config": "a3-v1",
            "sources": [coverage.source_label(s) for s in a.source], "runs_per_source": a.runs, "seed": a.seed,
            "starts": len(every), "starts_by_cat": dict(sorted(Counter(s["cat"] for s in every).items())),
            "heldout_starts": sum(split_of(s["id"]) == "heldout" for s in every)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "agent_root"}, indent=1))
    return 0


def manifest(out: Path, board: Path = BOARD_STARTS) -> dict:
    starts = _jsonl(out / "starts.jsonl")
    exam = [s for s in _jsonl(board) if s.get("cat") == BOARD_CAT]
    ids = [s["id"] for s in starts]
    rep = contamination(ids, [s["id"] for s in exam], {s["id"]: doc_sha(s["doc"]) for s in starts if s.get("doc")},
                        {s["id"]: doc_sha(s["doc"]) for s in exam if s.get("doc")})
    rep["board_source_starts"] = sum(s.get("source") == BOARD_SOURCE for s in starts)
    return {"contamination": rep, "train": sorted(i for i in ids if split_of(i) == "train"),
            "heldout": sorted(i for i in ids if split_of(i) == "heldout"), "exam": sorted(s["id"] for s in exam)}


def cmd_check(a) -> int:
    out = Path(a.out)
    m = manifest(out, Path(a.board))
    (out / "manifest.json").write_text(json.dumps(m, indent=0), encoding="utf-8")
    rep = m["contamination"]
    print(json.dumps({**rep, "train": len(m["train"]), "heldout": len(m["heldout"])}))
    bad = {k: v for k, v in rep.items() if k in ("same_id", "same_run_seed", "same_position", "board_source_starts") and v}
    return 1 if bad else 0


def cmd_build(a) -> int:
    """Join the recorded positions with their games' outcomes into one npz (tokens flat, per-position labels, start, split)."""
    out = Path(a.out)
    exam_seeds = {run_seed_of(i) for i in board_ids(Path(a.board))}
    starts = {s["id"]: s for s in _jsonl(out / "starts.jsonl")}
    lens, types, ids, owners, feats, meta, lab, grp, games = [], [], [], [], [], [], [], [], []
    start_ix: dict[str, int] = {}
    skipped = Counter()
    for name in a.games:
        for row in _jsonl(out / name):
            if row.get("arm") != a.arm or not row.get("_pos"):
                skipped["no positions"] += 1
                continue
            if row.get("capped"):
                skipped["capped"] += 1
                continue
            sid = row["start"]
            if sid not in starts or run_seed_of(sid) in exam_seeds:
                skipped["not a training start"] += 1
                continue
            z = unpack_positions(row["_pos"])
            n = len(z["lens"])
            if n == 0:
                continue
            gi = start_ix.setdefault(sid, len(start_ix))
            lb = labels_of(row)
            lens.append(z["lens"])
            types.append(z["types"])
            ids.append(z["ids"])
            owners.append(z["owners"])
            feats.append(z["feats"])
            meta.append(z["meta"])
            lab.append(np.tile([float(lb["won"]), lb["hp_frac"], lb["potions"], lb["L0"]], (n, 1)))
            grp.append(np.full(n, gi, dtype=np.int32))
            games.append((gi, int(row["j"]), n))
    names = sorted(start_ix, key=start_ix.get)
    split = np.array([split_of(s) == "heldout" for s in names], dtype=bool)
    cats = sorted({starts[s]["cat"] for s in names})
    np.savez_compressed(out / a.name, lens=np.concatenate(lens), types=np.concatenate(types), ids=np.concatenate(ids),
                        owners=np.concatenate(owners), feats=np.concatenate(feats), meta=np.concatenate(meta),
                        labels=np.concatenate(lab).astype(np.float32), group=np.concatenate(grp), start_heldout=split,
                        start_ids=np.array(names), start_cat=np.array([cats.index(starts[s]["cat"]) for s in names], dtype=np.int16),
                        cats=np.array(cats), games=np.array(games, dtype=np.int32))
    n_pos = int(sum(len(x) for x in lens))
    held_pos = int(sum(len(g) for g in grp if split[g[0]]))
    print(json.dumps({"positions": n_pos, "heldout_positions": held_pos, "games": len(games), "starts": len(names),
                      "heldout_starts": int(split.sum()), "skipped": dict(skipped), "file": a.name}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("starts", "check", "build", "buffer"))
    ap.add_argument("--loop", default=str(LOOP), help="buffer: the gym loop's folder")
    ap.add_argument("--exclude", action="append", default=[], help="buffer: a JSON of runs and starts out of bounds (exclude_runs.json, a board's meta); repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--agent-root", default="D:/spireline/referee/agent")
    ap.add_argument("--board", default=str(BOARD_STARTS))
    ap.add_argument("--tag", default="judge-v0")
    ap.add_argument("--games", action="append", default=[])
    ap.add_argument("--arm", default="a3-v1")
    ap.add_argument("--name", default="positions.npz")
    a = ap.parse_args(argv)
    if a.cmd == "build" and not a.games:
        a.games = ["judge_games.jsonl"]
    return {"starts": cmd_starts, "check": cmd_check, "build": cmd_build, "buffer": cmd_buffer}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
