"""The run currency's fast tests (training design 3.5 and the 2026-11-06 big-step memo, step 6): the audit, the board, the exam's two cells and
a copy batch, `a3-v1-r` against `a3-v1`, each arm on the same draws, and the three numbers the stop rule reads.

    python rl/currency_fast.py audit  --out DIR                   the head's marginal table and its stop rule (seconds)
    python rl/currency_fast.py board  --out DIR [--games 400]     the board's 200 Act 1 boss starts, both arms, paired by (start, draw)
    python rl/currency_fast.py cells  --out DIR                   the exam's Act 1 hallway and elite cells (100 starts each, four draws), both arms
    python rl/currency_fast.py batch  --out DIR [--runs 200]      a copy batch of whole runs, both arms, the same seeds (gen0.py emu)
    python rl/currency_fast.py report --out DIR                   the readings and the stop rule's verdict
    python rl/currency_fast.py all    --out DIR                   audit, board, cells, batch, report in that order

The plays hold a `b-fast` QUIET.lock line (rewritten every ten minutes, removed at the end) and start only when the machine is free: no lab
night or score block active, no real-game job, no QUIET line but `gym-loop`, `lab-score` and this one, no main.lock. They stop at once
when that stops being true and go on when it is true again, each game or run played once (rows are appended, finished ones skipped).
Processes are BelowNormal. Everything is lab code: the engine, the search and the exam's files are only read.

The stop rule (the audit's, and the memo's): the audit passes (design 3.4); the potions held at the Act 1 boss door are up and the Act 1
boss kill is not down beyond its standard error in the copy batch; the board's 95% lower bound of the win difference is above -2 points;
in the hallway cell the win rate is not under 99% and the share of won fights that began with a potion in which one was drunk is down.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fight_gym as fg  # noqa: E402

TAG = "b-fast"
REF = fg.REF
QUIET = REF / "QUIET.lock"
MAIN_LOCK = REF / "backlog" / "main.lock"
LAB = REF / "LAB"
TOLERATED = ("gym-loop", "lab-score", TAG)  # QUIET lines that do not hold this job (the memo's rule: the loop pauses its step for ours)
BEAT_S = 600.0  # the line is rewritten this often (the lab's runner reads a line older than 45 minutes as dead)
ARMS = ("a3-v1", "a3-v1-r")
AGENT_ROOT = REF / "agent"
PLAYER = "gen11/candidate"  # the combat prior every generation since 11 searches with (v2)
BATCH_AGENT = "gen12/candidate-imit-l05"  # generation 12: no stakes schedule, so the lab's --config is the fight config
EXAM_STARTS = REF / "gym-loop" / "exam" / "starts.jsonl"
CELLS = ("act1-hallway", "act1-elite")
ENGINE = "core2"
Z95 = 1.96


# ----------------------------------------------------------------------------------------------------------------- the machine


def utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def stamp() -> str:
    return utc().strftime("%Y-%m-%dT%H:%M:%SZ")


def quiet_lines() -> list[str]:
    try:
        return [x for x in QUIET.read_text(encoding="utf-8").splitlines() if x.strip()]
    except FileNotFoundError:
        return []


def blocked_reason() -> str | None:
    """Why the fast tests must not compute now (None: the machine is free)."""
    for flag in (LAB / "nightly" / "active", LAB / "score" / "active"):
        if flag.exists():
            return f"{flag.parent.name}-active"
    if MAIN_LOCK.exists():
        return "main.lock"
    for line in quiet_lines():
        if not line.split()[0] in TOLERATED:
            return "quiet:" + line.split()[0]
    try:
        status = (LAB / "status.txt").read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        status = ""
    if " running " in f" {status} " and any(w in status.split("job=")[-1].split()[0] for w in ("real", "nightly")):
        return "a real-game job is running"
    return None


def set_line(on: bool) -> None:
    keep = [x for x in quiet_lines() if x.split()[0] != TAG]
    if on:
        keep.append(f"{TAG} {stamp()} pid={os.getpid()}")
    if keep:
        QUIET.write_text("".join(x + "\n" for x in keep), encoding="utf-8", newline="\n")
    else:
        QUIET.unlink(missing_ok=True)


def wait_free(what: str) -> None:
    while (why := blocked_reason()) is not None:
        print(f"{stamp()} {what}: waiting ({why})", flush=True)
        time.sleep(60)


def guarded_tasks(what: str, tasks: list, fn, target: Path, procs: int, init_args: tuple) -> None:
    """`fight_gym.run_tasks` under this job's line and the free-machine rule: stops when the machine is taken, waits, goes on."""
    left = list(tasks)
    while left:
        wait_free(what)
        set_line(True)
        last = [time.time()]

        def refresh() -> None:
            if time.time() - last[0] > BEAT_S:
                set_line(True)
                last[0] = time.time()

        try:
            stopped = fg.run_tasks(left, fg._game_task, target, procs, init=fg._init_worker, init_args=init_args,  # noqa: SLF001
                                   mem_limit=fg.MEM_LIMIT_GB * 2**30, stop=blocked_reason, refresh=refresh)
        finally:
            set_line(False)
        gone = {json.dumps(r.get("_task")) for r in fg.read_jsonl(target)}
        left = [t for t in left if json.dumps(fg._task_id(t)) not in gone]  # noqa: SLF001
        print(f"{stamp()} {what}: {len(left)} left" + (f" (stopped: {stopped})" if stopped else ""), flush=True)


# --------------------------------------------------------------------------------------------------------------------- the plays


def prepare(out: Path, starts: list[dict], agent: str = PLAYER) -> tuple:
    """A gym folder: the starts and a meta.json naming the engine and the agent (its combat prior searches), as `fight_gym` reads them."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "starts.jsonl", "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(s) + "\n" for s in starts))
    meta = {"source": "currency fast test", "engine": ENGINE, "agent": agent, "agent_root": str(AGENT_ROOT), "config": "a3-v1", "starts": len(starts)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return (ENGINE, str(AGENT_ROOT), agent, str(out))


def play_arms(out: Path, starts: list[dict], draws: int, procs: int, what: str) -> None:
    init_args = prepare(out, starts)
    done = fg.done_keys(fg.read_jsonl(out / "games.jsonl"), fg.game_key)
    cats = {s["id"]: s["cat"] for s in starts}
    # both arms' game j of a start before game j + 1 of any: a stop leaves matched pairs, and the stage can end at any draw count
    tasks = [(arm, sid, j, cats[sid]) for j in range(draws) for sid in cats for arm in ARMS if (arm, sid, j) not in done]
    guarded_tasks(what, tasks, fg._game_task, out / "games.jsonl", procs, init_args)  # noqa: SLF001


def cmd_audit(a: argparse.Namespace) -> int:
    import run_currency as rc  # noqa: PLC0415

    starts = fg.read_jsonl(EXAM_STARTS)
    rep = rc.audit(rc.default_head(), starts, per_cell=a.audit_cell)
    out = Path(a.out) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    (out / "audit.md").write_text(rc.audit_markdown(rep), encoding="utf-8")
    print(rc.audit_markdown(rep))
    return 0 if rep["passed"] else 2


def board_starts() -> Path:
    import gym_loop as gl  # noqa: PLC0415

    return gl.BOARD_DIR / "starts.jsonl"


def cmd_board(a: argparse.Namespace) -> int:
    starts = [s for s in fg.read_jsonl(board_starts()) if s["cat"] == "act1-boss"]
    play_arms(Path(a.out) / "board", starts, a.games // max(1, len(starts)) or 1, a.procs, "board")
    return 0


def cmd_cells(a: argparse.Namespace) -> int:
    pool = fg.read_jsonl(EXAM_STARTS)
    starts = [s for c in CELLS for s in [x for x in pool if x["cat"] == c][: a.per_cell]]
    play_arms(Path(a.out) / "cells", starts, a.draws, a.procs, "cells")
    return 0


def batch_seeds(n: int) -> list[str]:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return [("".join(alphabet[b % 32] for b in hashlib.sha256(f"b-fast-currency:{i}".encode()).digest()[:10])) for i in range(n)]


def cmd_batch(a: argparse.Namespace) -> int:
    out = Path(a.out) / "batch"
    out.mkdir(parents=True, exist_ok=True)
    seeds_file = out / "seeds.json"
    seeds_file.write_text(json.dumps({"name": "b-fast-currency", "seeds": batch_seeds(a.runs)}), encoding="utf-8")
    for arm in ARMS:
        d = out / arm
        while not (d / "summary.json").is_file():
            wait_free(f"batch {arm}")
            if d.exists():
                shutil.rmtree(d)  # an arm stopped half way is played again from its first seed (the block has no resume of its own here)
            set_line(True)
            cmd = [sys.executable, str(ROOT / "rl" / "gen0.py"), "emu", "--seeds-file", str(seeds_file), "--n", str(a.runs), "--config", arm,
                   "--procs", str(a.procs), "--out", str(d), "--gen-dir", str(AGENT_ROOT / BATCH_AGENT), "--lib", str(ROOT / "out")]
            proc = subprocess.Popen(cmd, creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))  # noqa: S603
            stopped = False
            try:
                beat = time.time()
                while proc.poll() is None:
                    time.sleep(5)
                    if time.time() - beat > BEAT_S:
                        set_line(True)
                        beat = time.time()
                    if (why := blocked_reason()) is not None:
                        print(f"{stamp()} batch {arm}: stopped ({why})", flush=True)
                        proc.terminate()
                        proc.wait(60)
                        stopped = True
                        break
            finally:
                set_line(False)
            print(f"{stamp()} batch {arm}: rc {proc.returncode}{' (stopped)' if stopped else ''}", flush=True)
            if not stopped and (proc.returncode != 0 or not (d / "summary.json").is_file()):
                raise SystemExit(f"batch {arm}: the block failed (rc {proc.returncode})")
    return 0


# ------------------------------------------------------------------------------------------------------------------- the readings


def paired(pairs: list[tuple[float, float]]) -> dict:
    """Mean difference (second minus first) of paired values, its standard error, n."""
    d = [b - a for a, b in pairs]
    n = len(d)
    if n < 2:
        return {"diff": (d[0] if d else None), "se": None, "n": n}
    m = sum(d) / n
    return {"diff": m, "se": math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n), "n": n}


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def board_reading(rows: list[dict]) -> dict:
    """a3-v1-r minus a3-v1 on the games both played: win rate in points with the 95% lower bound, and what else moved."""
    d = fg.paired_diff(rows, "a3-v1-r", "a3-v1", "won")
    out: dict = {"pairs": d["pairs"], "starts": d["starts"]}
    if d["diff"] is not None and d["se_pairs"] is not None:
        out.update(win_diff_points=100 * d["diff"], se_points=100 * d["se_pairs"], lower95_points=100 * (d["diff"] - Z95 * d["se_pairs"]))
    for arm in ARMS:
        g = [r for r in rows if r["arm"] == arm]
        w = [r for r in g if r["won"]]
        out[arm] = {"games": len(g), "win_rate": mean([float(r["won"]) for r in g]), "end_hp_wins": mean([r["end_hp"] for r in w]),
                    "potions_used": mean([r["potions_used"] for r in g]), "potions_held_wins": mean([r["potions_held"] for r in w]),
                    "capped": sum(bool(r.get("capped")) for r in g)}
    return out


def cell_reading(rows: list[dict], starts: dict[str, dict]) -> dict:
    """Per cell and arm: win rate, HP lost a fight (the whole start HP in a loss), potions drunk, and the share of won fights that began
    with a potion in which one was drunk."""
    out: dict = {}
    for cell in CELLS:
        out[cell] = {}
        for arm in ARMS:
            g = [r for r in rows if r["arm"] == arm and starts[r["start"]]["cat"] == cell]
            lost = [(starts[r["start"]]["hp"] - r["end_hp"]) if r["won"] else starts[r["start"]]["hp"] for r in g]
            began = [r for r in g if r["won"] and starts[r["start"]]["potions"] >= 1]
            out[cell][arm] = {"games": len(g), "win_rate": mean([float(r["won"]) for r in g]), "hp_lost": mean(lost),
                              "potions_used": mean([r["potions_used"] for r in g]), "won_with_potion": len(began),
                              "drank_share": mean([float(r["potions_used"] >= 1) for r in began])}
    return out


def run_reading(rows: list[dict]) -> dict:
    """One copy batch arm: runs, how many reached the Act 1 boss, the hand at its door (potions, relics, HP share), how many beat it."""
    door = [r for r in rows if any(d[0] == 0 for d in r.get("boss_door") or [])]

    def at_door(r: dict) -> list:
        return next(d for d in r["boss_door"] if d[0] == 0)

    def beat(r: dict) -> bool:
        return any(a == 0 and f >= 1.0 for a, f in r.get("boss_damage_acts") or [])

    return {"runs": len(rows), "reached": len(door), "beat": sum(beat(r) for r in rows),
            "potions_at_door": mean([at_door(r)[1] for r in door]), "relics_at_door": mean([at_door(r)[2] for r in door]),
            "hp_share_at_door": mean([at_door(r)[3] / max(1, at_door(r)[4]) for r in door]), "floor": mean([float(r.get("floor") or 0) for r in rows]),
            "wall_s": mean([float(r.get("wall") or 0) for r in rows]),
            "kill_of_reached": (sum(beat(r) for r in door) / len(door)) if door else None}


def batch_reading(a: list[dict], r: list[dict]) -> dict:
    """The batch: each arm's reading and the paired differences by seed (a3-v1-r minus a3-v1) with their standard errors."""
    by_a, by_r = {x["seed"]: x for x in a}, {x["seed"]: x for x in r}
    common = sorted(set(by_a) & set(by_r))

    def door(x: dict):
        return next((d for d in x.get("boss_door") or [] if d[0] == 0), None)

    def beat(x: dict) -> float:
        return float(any(act == 0 and f >= 1.0 for act, f in x.get("boss_damage_acts") or []))

    both = [s for s in common if door(by_a[s]) and door(by_r[s])]
    out = {"a3-v1": run_reading(a), "a3-v1-r": run_reading(r), "seeds_paired": len(common), "seeds_at_both_doors": len(both),
           "potions_at_door_diff": paired([(door(by_a[s])[1], door(by_r[s])[1]) for s in both]),
           "relics_at_door_diff": paired([(door(by_a[s])[2], door(by_r[s])[2]) for s in both]),
           "hp_share_at_door_diff": paired([(door(by_a[s])[3] / max(1, door(by_a[s])[4]), door(by_r[s])[3] / max(1, door(by_r[s])[4])) for s in both]),
           "act1_boss_kill_per_100_runs_diff": paired([(100 * beat(by_a[s]), 100 * beat(by_r[s])) for s in common]),
           "floor_diff": paired([(float(by_a[s].get("floor") or 0), float(by_r[s].get("floor") or 0)) for s in common])}
    return out


def verdict(audit: dict | None, board: dict | None, cells: dict | None, batch: dict | None) -> dict:
    """The stop rule on the readings: each clause true, false, or None where its test has not been played."""
    out: dict = {"audit_passed": None if audit is None else bool(audit["passed"])}
    if batch is not None:
        p, k, rel = batch["potions_at_door_diff"], batch["act1_boss_kill_per_100_runs_diff"], batch["relics_at_door_diff"]
        out["potions_at_the_door_up"] = None if p["diff"] is None else p["diff"] > 0
        out["boss_kill_not_down"] = None if k["diff"] is None or k["se"] is None else k["diff"] >= -k["se"]
        out["relics_at_the_door_not_down"] = None if rel["diff"] is None or rel["se"] is None else rel["diff"] >= -rel["se"]
    if board is not None and "lower95_points" in board:
        out["board_lower95_above_minus_2"] = board["lower95_points"] > -2.0
    if cells is not None:
        h = cells["act1-hallway"]
        out["hallway_win_rate_at_least_99"] = None if h["a3-v1-r"]["win_rate"] is None else h["a3-v1-r"]["win_rate"] >= 0.99
        x, y = h["a3-v1"]["drank_share"], h["a3-v1-r"]["drank_share"]
        out["hallway_drinking_down"] = None if x is None or y is None else y < x
    known = [v for v in out.values() if v is not None]
    out["passes"] = bool(known) and all(known) and out["audit_passed"] is not None and batch is not None
    return out


def _rows(path: Path) -> list[dict]:
    return fg.read_jsonl(path)


def cmd_report(a: argparse.Namespace) -> int:
    out = Path(a.out)
    audit = json.loads((out / "audit" / "audit.json").read_text(encoding="utf-8")) if (out / "audit" / "audit.json").is_file() else None
    board = board_reading(_rows(out / "board" / "games.jsonl")) if (out / "board" / "games.jsonl").is_file() else None
    cells = None
    if (out / "cells" / "games.jsonl").is_file():
        starts = {s["id"]: s for s in _rows(out / "cells" / "starts.jsonl")}
        cells = cell_reading(_rows(out / "cells" / "games.jsonl"), starts)
    batch = None
    if all((out / "batch" / arm / "runs.jsonl").is_file() for arm in ARMS):
        batch = batch_reading(_rows(out / "batch" / "a3-v1" / "runs.jsonl"), _rows(out / "batch" / "a3-v1-r" / "runs.jsonl"))
    rep = {"audit": None if audit is None else {k: audit[k] for k in ("passed", "stop", "head_sha256")}, "board": board, "cells": cells, "batch": batch,
           "verdict": verdict(audit, board, cells, batch), "written": stamp()}
    (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


def cmd_all(a: argparse.Namespace) -> int:
    for step in (cmd_audit, cmd_board, cmd_cells, cmd_batch):
        rc = step(a)
        if rc not in (0, 2):
            return rc
        if rc == 2 and step is cmd_audit:
            print("the audit stopped the run", flush=True)
            cmd_report(a)
            return 2
    return cmd_report(a)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("audit", "board", "cells", "batch", "report", "all"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--games", type=int, default=400, help="board: games a arm (200 starts, so 400 is two draws a start, 2000 ten)")
    ap.add_argument("--per-cell", type=int, default=100, help="cells: starts a cell")
    ap.add_argument("--audit-cell", type=int, default=60, help="audit: starts a cell")
    ap.add_argument("--draws", type=int, default=4, help="cells: draws a start")
    ap.add_argument("--runs", type=int, default=200, help="batch: runs a arm")
    a = ap.parse_args(argv)
    return {"audit": cmd_audit, "board": cmd_board, "cells": cmd_cells, "batch": cmd_batch, "report": cmd_report, "all": cmd_all}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
