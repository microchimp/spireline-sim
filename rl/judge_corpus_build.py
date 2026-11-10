"""The judge corpus's builder (D58 C1; 2026-11-11): the human pool and its sampling order, the teacher's fights and their turns on the ladder,
the loop's buffer as self-play shards, the compaction into the store, coverage and the manifest. The store, its loader and its record layout
are `judge_corpus.py` (read its docstring first); this file is the part that plays the game.

    python rl/judge_corpus_build.py plan      [--store DIR]
    python rl/judge_corpus_build.py work      [--store DIR] [--shard LABEL] [--player v3] [--limit N] [--cells act2-boss,...]
    python rl/judge_corpus_build.py run       [--store DIR] [--procs 8] [--player v3]     # a bounded supervisor over `work` processes
    python rl/judge_corpus_build.py compact   [--store DIR]
    python rl/judge_corpus_build.py selfplay  [--store DIR]
    python rl/judge_corpus_build.py coverage  [--store DIR]
    python rl/judge_corpus_build.py status    [--store DIR]
    python rl/judge_corpus_build.py stop      [--store DIR]       # a STOP file: the work processes end after their fight

`plan` reads the human starts (`human_starts.py`'s `starts/human-starts.*`), leaves out the fight exam's runs and starts (`exam/exclude_runs.json`)
and, as a guard, any run of the exam fifth (`human_starts.assert_no_exam`), reads each start's deck and relics, and writes `work/plan.json` (the
order of the fights to build: `judge_corpus.plan_order`) and `work/pool.jsonl` (each start's cell, boost and priority; the inclusion weights
are made from it). `work` plays one fight at a time (claim files, so processes share the order; BelowNormal; paused at fight and rung
boundaries while another job's QUIET.lock line stands or a night block runs; the loop's own line does not hold it): the teacher plays the fight,
three turns and the door are kept with their tokens and labels (`build_fight`), and the result is one file under `work/fights/`. `compact` turns
the files into shards (it can run while `work` runs). A fight that raises is recorded with its traceback and not retried by itself.

Nothing here takes a QUIET.lock line (the brief for this lane keeps the practice loop running beside it); it waits on the others' lines.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import judge_corpus as jc  # noqa: E402
from judge_corpus import STORE, TAG  # noqa: E402

PLAYER_DEFAULT = "v3"
MINIMUMS = {"cell": 3000, "enemy": 400, "arch": 300, "relic": 150, "card": 150}  # positions: the cell minimums the build aims at (coverage.md)
STOP_NAME = "STOP"
MAX_ATTEMPTS = 3  # a fight whose worker died this many times is recorded as failed
RESTARTS = 40  # a work process that dies is started again this many times in all
FAIL_STREAK = 3  # a work process that meets this many failed fights in a row stops (its exit code 3 is not restarted)


# ---------------------------------------------------------------------------------------------------------- paths and small readers


def work_dir(store: Path) -> Path:
    return Path(store) / "work"


def fight_path(store: Path, start_id: str) -> Path:
    return work_dir(store) / "fights" / (hashlib.sha1(start_id.encode()).hexdigest()[:16] + ".json")  # noqa: S324 -- a file name


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if path.is_file():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    out.append(json.loads(line))
    return out


def utc() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def excluded_from(path: Path) -> tuple[set[str], set[str]]:
    """(run ids, start ids) a JSON file puts out of bounds: the fight exam's `exclude_runs.json` (`run_seeds`, `start_ids`) or a board's meta
    (any key ending `_starts` that holds a list of start ids or a dict of such lists; a start id's run is its part before `|`, after any
    `night:` or `human:` prefix)."""
    doc = read_json(path)
    runs, ids = {str(x) for x in doc.get("run_seeds", [])}, {str(x) for x in doc.get("start_ids", [])}
    for key, val in doc.items():
        if not key.endswith("_starts"):
            continue
        for lst in (val.values() if isinstance(val, dict) else [val]):
            if not isinstance(lst, list):
                continue
            for sid in lst:
                ids.add(str(sid))
                runs.add(bare_run_of(str(sid)))
    return runs, ids


def bare_run_of(start_id: str) -> str:
    """The run inside any start id: "SEED|f17", "night/gN:SEED|f17", "human:RUN|f17" and "preloss:human:RUN|f33|j0|k24"."""
    return start_id.split("|")[0].split(":")[-1]


def exclusions(loop: Path = jc.LOOP, extra_meta: Iterable[Path] = ()) -> tuple[set[str], set[str], list[str]]:
    """Every run and start id the corpus must not hold: the fight exam's list, the boards' start lists (every `meta-main.json` under the lab's
    fight-gym results) and the M2 board's. Returns (runs, ids, the files' names)."""
    runs: set[str] = set()
    ids: set[str] = set()
    names: list[str] = []
    files = [loop / "exam" / "exclude_runs.json", *extra_meta]
    results = jc.REFEREE / "lab" / "results" / "fight-gym"
    if results.is_dir():
        files += sorted(results.glob("*/meta-main.json"))
    for f in files:
        if Path(f).is_file():
            r, i = excluded_from(Path(f))
            runs |= r
            ids |= i
            names.append(Path(f).parent.name + "/" + Path(f).name)
    board = results / "2026-10-29-m2" / "starts.jsonl"
    if board.is_file():
        for s in read_jsonl(board):
            ids.add(s["id"])
            runs.add(bare_run_of(s["id"]))
        names.append("2026-10-29-m2/starts.jsonl")
    return runs, ids, names


def git_commit(path: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h", "--", path], capture_output=True, text=True, timeout=30, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------------------------------------------------- plan


def start_features(rec: dict, cards: dict[int, Any]) -> dict:
    """A start record's strata from its document: relics, deck (card id -> copies) and its tag, enemy name (the encounter's), HP and potions."""
    import practice_mix  # noqa: PLC0415
    from sts2_gym.env import ENCOUNTER_LABELS  # noqa: PLC0415

    doc = rec["doc"]
    deck: Counter = Counter(int(c["def_id"]) for c in practice_mix.doc_cards(doc))
    pots = doc.get("potions") or {}
    held = sum(1 for x in (pots.get("slots") or [])[: int(pots.get("max_slots", 0))] if x)
    return {"relics": sorted({int(r["id"]) for r in doc.get("relics") or []}), "deck": dict(deck), "arch": jc.deck_tag(deck, cards),
            "enemy": ENCOUNTER_LABELS.get(int(rec["encounter_id"]), str(rec["encounter_id"])), "hp": int(rec["hp"]), "max_hp": int(rec["max_hp"]),
            "potions": held}


def cmd_plan(a: argparse.Namespace) -> int:
    import human_starts as hs  # noqa: PLC0415
    from sts2_gym import names  # noqa: PLC0415

    store, loop = Path(a.store), Path(a.loop)
    starts = loop / "starts"
    lite = read_json(starts / "human-starts.lite.json")
    runs, ids, files = exclusions(loop)
    exam_ids = hs.read_exam_ids()
    pool = [s for s in lite if s["id"] not in ids and s["run"] not in runs and s["kind"] in jc.KINDS]
    left_out = len(lite) - len(pool)
    pool = [s for s in pool if not hs.is_exam(s["run"], exam_ids)]
    hs.assert_no_exam((s["run"] for s in pool), exam_ids)
    index = read_json(starts / "human-starts.index.json")
    cards = names.cards()
    feats: dict[str, dict] = {}
    t0 = time.time()
    with open(starts / "human-starts.jsonl", "rb") as fh:
        for s in sorted(pool, key=lambda s: index[s["id"]]):
            fh.seek(index[s["id"]])
            feats[s["id"]] = start_features(json.loads(fh.readline()), cards)
    print(f"read {len(feats)} start documents in {time.time() - t0:.0f}s", flush=True)
    boost: dict[str, float] = {}
    for kind in jc.KINDS:
        sub = [s for s in pool if s["kind"] == kind]
        fk = [jc.feature_keys(feats[s["id"]]["relics"], feats[s["id"]]["deck"], feats[s["id"]]["enemy"], feats[s["id"]]["arch"]) for s in sub]
        for s, b in zip(sub, jc.boost_weights(fk, a.support), strict=True):
            boost[s["id"]] = b
    cells: dict[str, list[str]] = defaultdict(list)
    rows = []
    for s in pool:
        f = feats[s["id"]]
        cell = f"act{s['act']}-{s['kind']}"
        key = jc.order_key(s["id"], boost[s["id"]])
        rows.append({"id": s["id"], "run": s["run"], "cell": cell, "kind": s["kind"], "act": int(s["act"]), "enemy": f["enemy"], "floor": int(s["floor"]),
                     "boost": round(boost[s["id"]], 4), "key": key, "arch": f["arch"], "hp": f["hp"], "max_hp": f["max_hp"], "potions": f["potions"],
                     "n_relics": len(f["relics"]), "deck_n": sum(f["deck"].values())})
    for r in sorted(rows, key=lambda r: r["key"]):
        cells[r["cell"]].append(r["id"])
    order = jc.plan_order(cells, jc.QUOTA)
    plan = {"schema": jc.SCHEMA, "created": utc(), "quota": jc.QUOTA, "support": a.support, "block": jc.ORDER_BLOCK, "cells": {c: len(v) for c, v in cells.items()},
            "pool": len(pool), "left_out_by_exclusion": left_out, "exclusion_files": files, "exam_runs_excluded": len(runs), "order": order}
    jc.write_atomic(work_dir(store) / "plan.json", plan)
    pp = work_dir(store) / "pool.jsonl"
    pp.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in plan.items() if k != "order"}, indent=1))
    print(f"order: {len(order)} fights")
    return 0


# ---------------------------------------------------------------------------------------------------------- building a fight (native)


def make_ctx(player_dir: Path, starts_path: str) -> dict:
    """What a work process holds: the teacher's prior (the CPU one the player's agent loads), the vocabulary inverses and the card table."""
    import converge  # noqa: PLC0415
    import gen_loop  # noqa: PLC0415
    from sts2_gym import names  # noqa: PLC0415

    converge.setup_native()
    extra = gen_loop.load_agent(player_dir).search_extra()
    inv_card, inv_relic = jc.vocab_inverse()
    return {"extra": extra, "starts_path": starts_path, "inv_card": inv_card, "inv_relic": inv_relic, "cards": names.cards(), "pairs": dict(jc.DUEL_PAIRS)}


def position_row(start: dict, meta: dict, k: int, turn: int, stage: str, stage_n: int, decisions: int, world: Any, toks: Any, ctx: dict) -> dict:
    """The strata and identity of one position (the label and the replay are added by the caller)."""
    f = jc.token_features(toks.types, toks.ids, toks.feats, ctx["inv_card"], ctx["inv_relic"], ctx["cards"])
    hp, max_hp = int(world.hp()), int(world.max_hp())
    return {"pid": f"{start['id']}#{k}", "src": "human", "sub": "human-start", "start": start["id"], "run": meta["run"], "split": jc.split_of(meta["run"]),
            "kind": meta["kind"], "act": int(meta["act"]), "enemy": meta["enemy"], "floor": int(start["floor"]), "k": k, "turn": turn, "stage": stage,
            "stage_n": stage_n, "decisions": decisions, "door": k == 0, "hp": hp, "max_hp": max_hp, "door_max_hp": int(start["max_hp"]),
            "hp_frac": f["hp_frac"] if f["hp_frac"] is not None else hp / max(1, max_hp), "potions": f["potions"], "relics": f["relics"], "deck": f["deck"],
            "deck_n": f["deck_n"], "arch": f["arch"]}


def teacher_result(game: dict) -> dict | None:
    """The teacher's own result of a played game (None for a game cut by a cap: it has no outcome)."""
    if game.get("capped"):
        return None
    return {"won": bool(game["won"]), "end_hp": int(game["end_hp"]), "max_hp": int(game["max_hp"]), "potions_held": int(game.get("potions_held") or 0),
            "L0": float(game.get("L0") or 0.0), "decisions": int(game["decisions"]), "turns": int(game["turns"])}


def build_fight(item: dict, ctx: dict, log: Callable[[str], None] = print, pause: Callable[[str], Any] = lambda label: 0.0) -> dict:
    """Play one human start with the teacher, keep the door and three turns with their tokens and labels. `item` is the plan's row for the start
    (`id`, `run`, `kind`, `act`, `cell`, `enemy`). Returns the fight's result (positions with their tokens as base64 text under `tok`)."""
    import converge  # noqa: PLC0415
    import gym_loop as gl  # noqa: PLC0415
    from prior_client import encode_position  # noqa: PLC0415

    c0 = time.process_time()
    start = gl.load_start(ctx["starts_path"], item["id"])
    kind = item["kind"]
    pause(f"{item['cell']} game")
    game, dec = converge.play_teacher(start, ctx["extra"])
    moves = [d["move"] for d in dec]
    eligible = [d["k"] for d in dec if d["n_moves"] >= converge.MIN_ROOT_MOVES]
    picks = {p["k"]: p for p in jc.pick_stage_turns(item["id"], eligible)}
    if 0 not in picks and dec:
        picks[0] = {"k": 0, "stage": "door", "stage_n": 0}
    teacher = teacher_result(game)
    positions: list[dict] = []
    problems: list[str] = []
    for k in sorted(picks):
        p = picks[k]
        world = converge.replay_to(start, moves, k)
        try:
            oh = converge.obs_hash(world)
            if oh != dec[k]["obs"]:
                problems.append(f"k{k}: the replay's observation differs")
                continue
            toks = encode_position(world.handle, False)
            turn = 1 + sum(1 for d in dec[:k] if d["search"]["best"]["kind"] == "end_turn")
            row = position_row(start, item, k, turn, p["stage"], p["stage_n"], len(dec), world, toks, ctx)
            game_rec = dec[k]["search"]
            if kind in jc.LADDER_KINDS and dec[k]["n_moves"] >= converge.MIN_ROOT_MOVES:  # the door too, when it has a choice
                def search(rel: float, world=world, k=k) -> dict:
                    pause(f"{item['cell']} rung {rel:g}x")
                    return converge.search_rung(world, start, kind, rel, k, ctx["extra"], pairs=ctx["pairs"][rel])

                res = jc.settle({**game_rec, "budget": converge.TEACHER_MULT[kind]}, search)
                row["label"] = jc.label_of(res["rec"], res["tier"], res["rel"], res["rungs"])
                row["cpu_s"] = res["cpu_s"]
            else:  # the teacher's own search at its budget: a hallway turn, or a position with one move only
                row["label"] = jc.label_of({**game_rec, "budget": converge.TEACHER_MULT[kind]}, "played", 1)
                row["cpu_s"] = 0.0
            row["game"] = {"sims": game_rec.get("sims"), "move": [dec[k]["move"][0], dec[k]["move"][1]]}
            row["played"] = [int(dec[k]["move"][0]), int(dec[k]["move"][1])]
            row["teacher"] = teacher
            row["oh"] = oh
            row["replay"] = {"j": converge.GAME_J, "ns": converge.GAME_NS, "moves": jc.moves_string(moves[:k])}
            row["tok"] = jc.b64_tokens({"types": toks.types, "ids": toks.ids, "owners": toks.owners, "feats": toks.feats})
            positions.append(row)
        finally:
            world.close()
    log(f"{item['cell']} {item['id']}: {len(positions)} positions, game cpu {game['cpu_s']:.1f}s, total {time.process_time() - c0:.1f}s")
    return {"start": item["id"], "cell": item["cell"], "kind": kind, "act": item["act"], "run": item["run"], "enemy": item["enemy"], "game": game,
            "decisions": len(dec), "eligible": len(eligible), "cpu_s": round(time.process_time() - c0, 2), "done": utc(), "problems": problems,
            "positions": positions}


# ---------------------------------------------------------------------------------------------------------- work, run, stop


def attempts_file(store: Path, start_id: str) -> Path:
    return work_dir(store) / "attempts" / (hashlib.sha1(start_id.encode()).hexdigest()[:16] + ".txt")  # noqa: S324


def note_attempt(store: Path, start_id: str) -> int:
    """Count this claim of the fight (a line a try, in a file); returns how many there have been, this one included."""
    p = attempts_file(store, start_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(utc() + "\n")
    return len(p.read_text(encoding="utf-8").splitlines())


def pending(store: Path, plan: dict, cells: Sequence[str] | None = None) -> list[str]:
    """The plan's fights without a result file, in order."""
    by_id = None
    out = []
    for sid in plan["order"]:
        if cells:
            if by_id is None:
                by_id = {r["id"]: r["cell"] for r in read_jsonl(work_dir(store) / "pool.jsonl")}
            if by_id.get(sid) not in cells:
                continue
        if not fight_path(store, sid).is_file():
            out.append(sid)
    return out


def cmd_work(a: argparse.Namespace) -> int:
    import converge  # noqa: PLC0415

    store, loop = Path(a.store), Path(a.loop)
    plan = read_json(work_dir(store) / "plan.json")
    pool = {r["id"]: r for r in read_jsonl(work_dir(store) / "pool.jsonl")}
    ctx = make_ctx(loop / "players" / a.player, str(loop / "starts" / "human-starts.jsonl"))
    cells = set(a.cells.split(",")) if a.cells else None
    wd = work_dir(store)
    done = 0
    failed_in_a_row = 0

    def say(m: str) -> None:
        print(f"[{a.shard} {utc()}] {m}", flush=True)

    def pause(label: str) -> float:
        return converge.pause_point(label, say, out=wd, ignore=(TAG,))

    for sid in plan["order"]:
        item = pool.get(sid)
        if item is None or (cells and item["cell"] not in cells):
            continue
        if fight_path(store, sid).is_file():
            continue
        if (wd / STOP_NAME).exists():
            say("STOP file: ending")
            break
        if a.limit and done >= a.limit:
            break
        pause("between fights")
        if not converge.claim(wd, sid):
            continue
        n = note_attempt(store, sid)
        if n > MAX_ATTEMPTS:
            jc.write_atomic(fight_path(store, sid), {"start": sid, "cell": item["cell"], "error": f"the worker died {n - 1} times on this fight", "positions": []})
            continue
        t0 = time.perf_counter()
        try:
            res = build_fight(item, ctx, say, pause)
        except Exception:  # noqa: BLE001 -- a fight that raises is recorded, the process goes on
            res = {"start": sid, "cell": item["cell"], "error": traceback.format_exc()[-3000:], "positions": []}
            say(f"{item['cell']} {sid} FAILED")
        jc.write_atomic(fight_path(store, sid), res)
        done += 1
        failed_in_a_row = failed_in_a_row + 1 if res.get("error") else 0
        say(f"done {sid} wall {time.perf_counter() - t0:.0f}s")
        if failed_in_a_row >= FAIL_STREAK:  # a bug, not a bad fight: stop before the plan is spent
            say(f"{failed_in_a_row} fights failed in a row: stopping")
            return 3
    say(f"finished, {done} fights")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    """A bounded supervisor: `--procs` work processes at BelowNormal, restarted when one dies (up to RESTARTS in all), until the plan's fights
    all have a result or the STOP file appears. It holds no lock line and takes no machine from anyone: the work processes wait on others'."""
    from sts2_gym import procutil  # noqa: PLC0415

    store = Path(a.store)
    wd = work_dir(store)
    plan = read_json(wd / "plan.json")
    cells = a.cells.split(",") if a.cells else None
    (wd / STOP_NAME).unlink(missing_ok=True)
    script = str(Path(__file__).resolve())
    logs = wd / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def launch(i: int) -> subprocess.Popen:
        cmd = [sys.executable, "-I", script, "work", "--store", str(store), "--loop", a.loop, "--shard", f"w{i}", "--player", a.player]
        if cells:
            cmd += ["--cells", ",".join(cells)]
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}  # one thread a process: no thread spins at Normal priority
        with open(logs / f"work-w{i}.log", "a", encoding="utf-8") as fh:  # the child has its own copy of the handle
            return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, **procutil.popen_priority_kwargs())

    procs = {i: launch(i) for i in range(a.procs)}
    restarts = 0
    rounds = 0
    bug = False
    print(f"{utc()} supervisor: {a.procs} work processes, {len(pending(store, plan, cells))} fights to build", flush=True)
    while procs:
        time.sleep(20)
        for i, p in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            del procs[i]
            if rc == 3:
                bug = True
                print(f"{utc()} work process w{i} stopped after failed fights in a row: not restarted", flush=True)
            elif rc != 0 and not (wd / STOP_NAME).exists() and restarts < RESTARTS:
                restarts += 1
                print(f"{utc()} work process w{i} ended with {rc}: restarted ({restarts})", flush=True)
                procs[i] = launch(i)
        if not procs and not bug and not (wd / STOP_NAME).exists() and rounds < 3 and pending(store, plan, cells):
            rounds += 1  # a pass that left fights behind (a dead claim met late): one more pass
            procs = {i: launch(i) for i in range(a.procs)}
    left = pending(store, plan, cells)
    print(f"{utc()} supervisor: ended, {len(left)} fights without a result", flush=True)
    return 0


def cmd_stop(a: argparse.Namespace) -> int:
    (work_dir(Path(a.store)) / STOP_NAME).write_text(utc(), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------------------------------------- status


def read_fights(store: Path) -> list[dict]:
    d = work_dir(store) / "fights"
    out = []
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        with contextlib.suppress(ValueError, OSError):
            out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def cmd_status(a: argparse.Namespace) -> int:
    store = Path(a.store)
    plan = read_json(work_dir(store) / "plan.json")
    fights = read_fights(store)
    ok = [f for f in fights if not f.get("error")]
    bad = [f for f in fights if f.get("error")]
    by: Counter = Counter(f["cell"] for f in ok)
    pos: Counter = Counter()
    for f in ok:
        for p in f["positions"]:
            pos[(p["kind"], p["act"], p["label"]["tier"])] += 1
    cpu = sum(f.get("cpu_s", 0.0) for f in ok)
    print(json.dumps({"plan_fights": len(plan["order"]), "done": len(ok), "failed": len(bad), "cpu_hours": round(cpu / 3600, 2),
                      "by_cell": dict(by), "positions": {f"act{k[1]}-{k[0]}:{k[2]}": v for k, v in sorted(pos.items())}}, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------------------- compaction


def cell_inclusion(pool: Sequence[dict], done_ids: set[str]) -> dict[str, float]:
    """Per start of the pool, its inclusion probability given that `done_ids` were the fights built (per cell: `jc.inclusion` over the cell's
    boosts with n = the cell's built fights)."""
    out: dict[str, float] = {}
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_cell[r["cell"]].append(r)
    for cell, rows in by_cell.items():
        n = sum(1 for r in rows if r["id"] in done_ids)
        for r, pi in zip(rows, jc.inclusion([r["boost"] for r in rows], n), strict=True):
            out[r["id"]] = pi
    return out


def human_outcome(entry: dict | None) -> dict | None:
    if entry is None:
        return None
    return {"survived": bool(entry["human_survived"]), "end_hp": int(entry["human_end_hp"]), "hp_lost": int(entry["human_hp_lost"]),
            "turns": int(entry["human_turns"]), "potions_used": int(entry["human_potions_used"]), "potions_held_end": int(entry.get("human_potions_held_end") or 0),
            "l0": float(entry["human_l0"]), "l0_unit": float(entry["human_l0_unit"])}


def cmd_compact(a: argparse.Namespace) -> int:
    import human_starts as hs  # noqa: PLC0415
    from sts2_gym import procutil  # noqa: PLC0415

    procutil.lower_priority()
    store, loop = Path(a.store), Path(a.loop)
    plan = read_json(work_dir(store) / "plan.json")
    pool = read_jsonl(work_dir(store) / "pool.jsonl")
    outcomes = read_json(loop / "starts" / "human-outcomes.json")["outcomes"]
    fights = [f for f in read_fights(store) if not f.get("error") and f.get("positions")]
    done = {f["start"] for f in fights}
    pi = cell_inclusion(pool, done)
    exam_ids = hs.read_exam_ids()
    hs.assert_no_exam({f["run"] for f in fights}, exam_ids)
    groups: dict[tuple[str, int], list[tuple[dict, dict]]] = defaultdict(list)
    cell_w: dict[str, list[float]] = defaultdict(list)
    for f in fights:
        for p in f["positions"]:
            tok = jc.tokens_of_b64(p.pop("tok"))
            p["pi"] = round(pi[f["start"]], 6)
            p["w"] = round(1.0 / max(pi[f["start"]], 1e-9), 4)
            p["human"] = human_outcome(outcomes.get(f["start"]))
            p["cpu_fight_s"] = f["cpu_s"] if p["k"] == min(q["k"] for q in f["positions"]) else 0.0
            groups[(p["kind"], p["act"])].append((p, tok))
            cell_w[f"act{p['act']}-{p['kind']}"].append(p["w"])
    written: set[str] = set()
    for (kind, act), items in sorted(groups.items()):
        items.sort(key=lambda it: (it[0]["start"], it[0]["k"]))
        mean_w = float(np.mean(cell_w[f"act{act}-{kind}"]))
        for p, _ in items:
            p["w_cell"] = round(p["w"] / mean_w, 4)
        for n, at in enumerate(range(0, len(items), jc.SHARD_ROWS)):
            chunk = items[at : at + jc.SHARD_ROWS]
            stem = store / "shards" / f"human-{kind}-act{act}-{n:04d}"
            jc.write_shard(stem, [c[0] for c in chunk], [c[1] for c in chunk])
            written.add(stem.name)
    shards = store / "shards"
    for p in list(shards.glob("human-*")) if shards.is_dir() else []:
        if p.stem.split(".")[0] not in written and p.suffix in (".jsonl", ".npz"):
            p.unlink()
    print(f"compacted {len(fights)} fights into {len(written)} shards: " + ", ".join(f"act{a_}-{k}: {len(v)}" for (k, a_), v in sorted(groups.items())))
    write_manifest(store, loop, plan)
    return 0


# ---------------------------------------------------------------------------------------------------------- self-play (the loop's buffer)


def start_source(start_id: str) -> str:
    """Where a buffer fight's start came from: "preloss", "human" or a night's replay."""
    return "preloss" if start_id.startswith("preloss:") else "human" if start_id.startswith("human:") else "night"


def buffer_labels(meta: dict) -> dict | None:
    """The fight's outcome from its buffer meta (None for a capped fight or one with no value label: a cut segment's fights lean to quick
    deaths and keep none). A shard from before `potions_held` gives the potions the fight began with less those it used."""
    if meta.get("capped") or not meta.get("value_ok"):
        return None
    won = bool(meta.get("won"))
    held = meta.get("potions_held")
    if held is None:
        held = max(0, int(meta.get("start_potions") or 0) - int(meta.get("potions_used") or 0))
    return {"won": won, "end_hp": int(meta.get("end_hp") or 0), "max_hp": max(1, int(meta.get("max_hp") or 1)), "potions_held": int(held) if won else 0,
            "L0": float(meta.get("L0") or 0.0), "decisions": int(meta.get("decisions") or 0), "turns": int(meta.get("turns") or 0)}


ACTION_TYPES = {2: "card", 10: "end_turn", 8: "potion", 11: "select", 12: "skip"}  # the token at 1 + action: hand card, end turn, potion, choice, decline


def selfplay_rows(z: Mapping[str, Any], metas: list[dict], skip_runs: set[str], skip_ids: set[str], ctx: dict, enc_names: Mapping[int, str],
                  skipped: Counter) -> Iterable[tuple[dict, dict]]:
    """Every usable position of one buffer shard as (row, tokens): per decision of an included fight, the search as played (the visit
    distribution with each move's mean value), the fight's outcome and the strata read off the tokens."""
    fight, lengths = z["fight"], z["lengths"]
    toff = np.concatenate([[0], np.cumsum(lengths.astype(np.int64))])
    by_fight: dict[int, list[int]] = defaultdict(list)
    for r, fi in enumerate(fight.tolist()):
        by_fight[int(fi)].append(r)
    for fi, f in enumerate(metas):
        out = buffer_labels(f)
        if out is None:
            skipped["capped or no value label"] += 1
            continue
        if str(f["seed"]) in skip_runs or f["start"] in skip_ids or bare_run_of(f["start"]) in skip_runs:
            skipped["excluded run or start"] += 1
            continue
        cat = f.get("cat") or ""
        if not cat.startswith("act") or cat.split("-")[1] not in jc.KINDS:
            skipped["not an act 1 or 2 fight"] += 1
            continue
        act, kind = int(cat[3]), cat.split("-")[1]
        rows = by_fight.get(fi, [])
        if not rows:
            continue
        enemy = enc_names.get(int(f.get("encounter", -1)), str(f.get("encounter")))
        floor = int(next(iter(re.findall(r"\|f(\d+)", f["start"])), 0))
        n = len(rows)
        turn = 1
        for j, r in enumerate(rows):
            sl = slice(toff[r], toff[r + 1])
            types, ids, owners, feats = (np.array(z[name][sl]) for name in ("types", "ids", "owners", "feats"))  # copies: a shard's arrays are freed once read
            tf = jc.token_features(types, ids, feats, ctx["inv_card"], ctx["inv_relic"], ctx["cards"])
            vis, q, legal = z["visits"][r], z["qvals"][r], z["legal"][r]
            children = []
            for slot in np.nonzero(vis > 0)[0].tolist():
                tt = int(types[1 + slot]) if 1 + slot < len(types) else 0
                card = ctx["inv_card"].get(int(ids[1 + slot])) if tt == jc.T_HAND and 1 + slot < len(ids) else None
                children.append({"a": int(slot), "t": -1, "kind": ACTION_TYPES.get(tt, "other"), "card": card, "n": int(vis[slot]),
                                 "v": None if not np.isfinite(q[slot]) else round(float(q[slot]), 4), "se": None, "pd": None})
            children.sort(key=lambda c: (-c["n"], c["a"]))
            played = max(0, int(z["played"][r]))
            tot = sum(c["n"] for c in children)
            rv = (sum(c["n"] * c["v"] for c in children if c["v"] is not None) / tot) if tot and any(c["v"] is not None for c in children) else None
            max_hp = int(out["max_hp"])
            hp_frac = tf["hp_frac"] if tf["hp_frac"] is not None else 0.0
            best = children[0] if children else None
            row = {"pid": f"{f['start']}~{f['j']}~{f.get('player')}#{int(z['k'][r])}", "src": "selfplay", "sub": start_source(f["start"]), "start": f["start"],
                   "run": str(f["seed"]), "split": jc.split_of(str(f["seed"])), "kind": kind, "act": act, "enemy": enemy, "floor": floor, "k": int(z["k"][r]),
                   "turn": turn, "stage": jc.STAGES[min(2, 3 * j // n)], "stage_n": 0, "decisions": n, "door": j == 0, "hp": int(round(hp_frac * max_hp)),
                   "max_hp": max_hp, "door_max_hp": max_hp, "hp_frac": hp_frac, "potions": tf["potions"], "relics": tf["relics"], "deck": tf["deck"],
                   "deck_n": tf["deck_n"], "arch": tf["arch"], "player": f.get("player"), "config": f.get("config"), "j": int(f["j"]), "teacher": out,
                   "played": [played, -1], "w": 1.0, "w_cell": 1.0, "human": None,
                   "label": {"tier": "played", "rel": 1, "budget": f.get("config"), "sims": int(z["sims"][r]), "visits": tot, "move": None if best is None else
                             {"a": played, "t": -1, "kind": ACTION_TYPES.get(int(types[1 + played]) if 1 + played < len(types) else 0, "other"), "card": None},
                             "root_value": rv, "best_value": None if not np.isfinite(z["q_played"][r]) else round(float(z["q_played"][r]), 4), "p_death": None,
                             "children": children}}
            if 1 + played < len(types) and int(types[1 + played]) == 10:
                turn += 1
            yield row, {"types": types, "ids": ids, "owners": owners, "feats": feats}


def cmd_selfplay(a: argparse.Namespace) -> int:
    """The loop's buffer shards not yet in the store, as self-play shards (a ledger keeps the shards already taken, so a later run adds the newer
    ones and the older ones the loop's retention has deleted stay). Excluded: the exam's and the boards' runs and starts, capped fights and fights
    of cut segments."""
    from sts2_gym import names, procutil  # noqa: PLC0415
    from sts2_gym.env import ENCOUNTER_LABELS  # noqa: PLC0415

    procutil.lower_priority()
    store, loop = Path(a.store), Path(a.loop)
    skip_runs, skip_ids, files = exclusions(loop)
    ledger_path = work_dir(store) / "selfplay-ledger.json"
    ledger = read_json(ledger_path) if ledger_path.is_file() else {"shards": {}, "files": []}
    ctx = {"inv_card": jc.vocab_inverse()[0], "inv_relic": jc.vocab_inverse()[1], "cards": names.cards()}
    index = read_jsonl(loop / "buffer" / "index.jsonl")
    skipped: Counter = Counter()
    buffers: dict[tuple[str, int], list[tuple[dict, dict]]] = defaultdict(list)
    counters: dict[tuple[str, int], int] = {}
    shards = store / "shards"
    for p in shards.glob("selfplay-*.jsonl") if shards.is_dir() else []:
        kind, act = p.stem.split("-")[1], int(p.stem.split("-")[2][3:])
        counters[(kind, act)] = max(counters.get((kind, act), -1), int(p.stem.split("-")[3]))

    def flush(key: tuple[str, int], force: bool = False) -> None:
        buf = buffers[key]
        while len(buf) >= jc.SHARD_ROWS or (force and buf):
            chunk, buffers[key] = buf[: jc.SHARD_ROWS], buf[jc.SHARD_ROWS :]
            buf = buffers[key]
            counters[key] = counters.get(key, -1) + 1
            jc.write_shard(shards / f"selfplay-{key[0]}-act{key[1]}-{counters[key]:04d}", [c[0] for c in chunk], [c[1] for c in chunk])

    added = 0
    for entry in index:
        name = entry["shard"]
        path = loop / "buffer" / "shards" / name
        if name in ledger["shards"] or not path.is_file():
            continue
        with np.load(path) as zf:
            z = {k: zf[k] for k in zf.files if k != "fights_meta"}
            metas = json.loads(str(zf["fights_meta"]))
        n = 0
        for row, tok in selfplay_rows(z, metas, skip_runs, skip_ids, ctx, ENCOUNTER_LABELS, skipped):
            buffers[(row["kind"], row["act"])].append((row, tok))
            n += 1
        ledger["shards"][name] = {"rows": n, "t": entry.get("t"), "segment": entry.get("segment")}
        added += n
        for key in list(buffers):
            flush(key)
    for key in list(buffers):
        flush(key, force=True)
    ledger["files"] = files
    ledger["skipped"] = {k: ledger.get("skipped", {}).get(k, 0) + v for k, v in skipped.items()}
    ledger["updated"] = utc()
    jc.write_atomic(ledger_path, ledger)
    print(f"self-play: {added} positions added from {sum(1 for e in index if e['shard'] in ledger['shards'])} buffer shards; skipped {dict(skipped)}")
    write_manifest(store, loop, read_json(work_dir(store) / "plan.json") if (work_dir(store) / "plan.json").is_file() else {})
    return 0


# ---------------------------------------------------------------------------------------------------------- manifest and coverage


def sha_cached(store: Path, path: Path) -> str | None:
    """sha256 of a source file, remembered by (size, mtime) so a compaction does not read 640 MB again."""
    if not path.is_file():
        return None
    cache_path = work_dir(store) / "sha-cache.json"
    cache = read_json(cache_path) if cache_path.is_file() else {}
    st = path.stat()
    key = f"{path.name}:{st.st_size}:{int(st.st_mtime)}"
    if key not in cache:
        cache[key] = jc.sha256_file(path)
        jc.write_atomic(cache_path, cache)
    return cache[key]


def count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def write_manifest(store: Path, loop: Path, plan: dict) -> dict:
    """`manifest.json`: what the store holds and how it was made (the engine, the player, the label recipe and caps, counts per stratum, the
    source files with their sha256, the exam exclusions, the commits of the code that made it, the CPU spent)."""
    import hashlib as _h  # noqa: PLC0415

    corpus = jc.Corpus(store)
    rows = [r for r, _s, _i in corpus.iter_rows()]
    shard_files = sorted((store / "shards").glob("*.jsonl")) if (store / "shards").is_dir() else []

    def n_by(key: Callable[[dict], Any], sel: Iterable[dict] | None = None) -> dict[str, int]:
        c: Counter = Counter(str(key(r)) for r in (rows if sel is None else sel))
        return dict(sorted(c.items()))

    player_dir = loop / "players" / PLAYER_DEFAULT
    pol = player_dir / "combat_policy.pt"
    engine = None
    with contextlib.suppress(Exception):
        engine = _h.sha256((ROOT / "out" / "Sts2Emulator.dll").read_bytes()).hexdigest()[:12]
    fights = read_fights(store)
    ok = [f for f in fights if not f.get("error")]
    runs, ids, excl_files = exclusions(loop)
    human = [r for r in rows if r["src"] == "human"]
    man = {
        "schema": jc.SCHEMA, "written": utc(), "attribution": jc.ATTRIBUTION, "engine_tag": engine, "engine_commit": git_commit("src"),
        "player": {"version": PLAYER_DEFAULT, "policy_sha256": jc.sha256_file(pol) if pol.is_file() else None, "config": "a3-v1@h1e4b16", "prior": "cpu"},
        "label_recipe": {"human_labels": "the human's own result of the fight (starts/human-outcomes.json), the teacher's result of the game, and the search",
                         "ladder": {"game rung": "the teacher's own search at its budget is the x1 rung (boss x16, elite x4, hallway x1)",
                                    "rungs": list(jc.LADDER_RELS), "boss": "x32 then x64", "elite": "x8 then x16", "cap": "x64 at a boss, x16 at an elite",
                                    "duel_pairs": {str(k): v for k, v in jc.DUEL_PAIRS.items()}, "rule": "converge.rule_holds, gap 2 SE, se_mode paired",
                                    "soft": "a turn the rule never held for keeps the last uncapped rung's values", "hallway": "the teacher's own search at its budget (tier played)",
                                    "cpu_cap_s": 300},
                         "turns": "the door (first decision) and one turn in each third of the fight's decisions that have two or more distinct moves (a hash coin)",
                         "selfplay": "the buffer's search as played (visits and mean values per move) and the fight's outcome; stage = third of the fight's decisions"},
        "strata": {"minimums": MINIMUMS, "support_starts": plan.get("support"), "quota": plan.get("quota"), "cells_pool": plan.get("cells")},
        "counts": {"positions": len(rows), "by_source_kind_act": n_by(lambda r: f"{r['src']}-{r['kind']}-act{r['act']}"),
                   "by_tier": n_by(lambda r: f"{r['src']}-{r['kind']}-{r['label']['tier']}"), "by_stage": n_by(lambda r: f"{r['src']}-{r['stage']}"),
                   "by_split": n_by(lambda r: f"{r['src']}-{r['split']}"), "human_fights": len({r["start"] for r in human}),
                   "human_runs": len({r["run"] for r in human}), "selfplay_runs": len({r["run"] for r in rows if r["src"] == "selfplay"})},
        "shards": {p.stem: {"rows": count_lines(p), "jsonl_sha256": jc.sha256_file(p),
                            "npz_sha256": jc.sha256_file(p.with_suffix(".npz"))} for p in shard_files},
        "sources": {"human_starts_jsonl": sha_cached(store, loop / "starts" / "human-starts.jsonl"),
                    "human_starts_lite": sha_cached(store, loop / "starts" / "human-starts.lite.json"),
                    "human_outcomes": sha_cached(store, loop / "starts" / "human-outcomes.json"),
                    "buffer_index": sha_cached(store, loop / "buffer" / "index.jsonl")},
        "exam_exclusions": {"runs": len(runs), "start_ids": len(ids), "files": excl_files,
                            "exam_fifth": "human_starts.assert_no_exam: the ids file and the hash rule (bucket 0); the starts were built from the other four fifths",
                            "exam_fifth_ids_sha256": sha_cached(store, jc_exam_ids_path())},
        "commits": {"converge": git_commit("rl/converge.py"), "human_starts": git_commit("rl/human_starts.py"), "judge_corpus": git_commit("rl/judge_corpus.py"),
                    "judge_corpus_build": git_commit("rl/judge_corpus_build.py")},
        "cpu_hours": {"human_fights": round(sum(f.get("cpu_s", 0.0) for f in ok) / 3600, 2),
                      "fights_built": len(ok), "fights_failed": len(fights) - len(ok)},
    }
    jc.write_atomic(store / "manifest.json", man)
    return man


def jc_exam_ids_path() -> Path:
    import human_starts as hs  # noqa: PLC0415

    return Path(hs.EXAM_IDS)


def cmd_coverage(a: argparse.Namespace) -> int:
    from sts2_gym import names  # noqa: PLC0415

    store = Path(a.store)
    corpus = jc.Corpus(store)
    rows = [r for r, _s, _i in corpus.iter_rows()]
    names_of = {"relic": dict(names.relics()), "card": {cid: i.name for cid, i in names.cards().items()}}
    head = ("# Judge corpus: coverage\n\nWritten by `rl/judge_corpus_build.py coverage`. Aggregates only: positions and runs per cell; a cell under "
            f"{jc.MIN_CELL} positions or {jc.MIN_CELL} runs is not shown. Source of the human positions: {jc.ATTRIBUTION}. The minimums the build aims at: "
            + ", ".join(f"{k} {v}" for k, v in MINIMUMS.items()) + " positions.\n")
    md = head + jc.coverage_md(rows, MINIMUMS, names_of)
    (store / "coverage.md").write_text(md, encoding="utf-8", newline="\n")
    print(f"coverage.md written for {len(rows)} positions")
    return 0


# ---------------------------------------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("plan", "work", "run", "compact", "selfplay", "coverage", "status", "stop"))
    ap.add_argument("--store", default=str(STORE))
    ap.add_argument("--loop", default=str(jc.LOOP))
    ap.add_argument("--player", default=PLAYER_DEFAULT)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--shard", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cells", default="")
    ap.add_argument("--support", type=int, default=jc.MIN_SUPPORT_STARTS)
    a = ap.parse_args(argv)
    if a.procs > 8:
        raise SystemExit("at most 8 processes")
    return {"plan": cmd_plan, "work": cmd_work, "run": cmd_run, "compact": cmd_compact, "selfplay": cmd_selfplay, "coverage": cmd_coverage,
            "status": cmd_status, "stop": cmd_stop}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
