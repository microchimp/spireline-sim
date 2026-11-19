#!/usr/bin/env python3
"""The Act 3 stratum of the fight exam, played (the Act 3 week, lane 5: the Test Subject).

    python rl/act3_exam.py play   --exam-dir DIR [--agent-dir D] [--draws 4] [--procs 12] [--take CELL=N,...] [--wall-hours 3] [--order draw]
    python rl/act3_exam.py report --exam-dir DIR [--json]

What it is. `fight_exam.py add-act3` puts the Act 3 boss cells in an exam folder: human openings rebuilt from the corpus's training fifths (each
with the human's own result) and our own arrivals from the plain runs, by cell `act3-boss<turn>-<boss id>`. This module plays them: the
champion's fight network on the copy at the champion's own boss budget (`agent.json` `search_schedule`, today `a3-v1@h1e4b16`: the budget the
plain runs play at a boss, `act3_replay.arm_config(.., 1)`), on the old engine (the plain runs' and the judge's), four draws a start (the shuffles
and the per-decision search seeds are fixed by (start, draw), never by the player: the common random numbers the exam's other strata use), the
prior in-process on the CPU (exact, no server). One JSONL row a game, with a trace of the decisions (the boss's forms, the player's health turn by
turn) for the Test Subject; a game with a row is never played twice, so a stop only loses the games in flight.

The report. Kills by cell and by side (the humans' positions against ours) with a 95% interval; on the human positions, the champion's kill
rate against what the human did on the same start (the human bar: the share of those humans that went past the boss), the paired difference
by start with a bootstrap interval; and the Test Subject's dynamics from the traces: turns spent in each form, the health lost a turn in each
form, and the move that killed.

Lab code: nothing here changes the engine, the search or the solver. Heavy work belongs to the caller's QUIET.lock line (`TEST-SUBJECT-exam.sh`
holds it); this module only stops when the machine is wanted (`fight_exam.blocked_reason`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Nothing here loads the emulator library at import: the pure parts run anywhere, the games in spawned processes.

RULE = "act3-exam-play-v1"
NS = "x3"  # the draws' seed namespace (the exam's AI strata use "x", the gym's "g", the practice's "L")
TAG = "act3-exam"  # the QUIET.lock line TEST-SUBJECT-exam.sh holds
DRAWS = 4
BOOT = 2000
BOOT_SEED = 20261121
DEADLINE_S = 1800.0  # a game's soft deadline (the replay's longest 1x game took 43 s)
TS_DEF = 10007  # the copy's Test Subject enemy id
GAMES = "games-act3.jsonl"
LOG = "act3-exam.log"
BOSS_NAMES = {73: "aeonglass", 78: "queen", 80: "test-subject"}


# ---------------------------------------------------------------------------------------------------------- pure parts


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def won_of(row: dict) -> bool:
    return bool(row.get("won")) and not row.get("capped")


def order_key(sid: str) -> str:
    return hashlib.sha256(f"act3-exam-order:{sid}".encode()).hexdigest()


def parse_take(text: str | None) -> dict[str, int]:
    """`act3-boss1-80=40,act3-boss1-78=20` as {cell: starts}; the cells not named are not played. Empty text: every start."""
    out: dict[str, int] = {}
    for part in (text or "").split(","):
        if part.strip():
            cell, n = part.split("=")
            out[cell.strip()] = int(n)
    return out


def select_starts(starts: dict[str, dict], take: dict[str, int] | None = None, strata: tuple = ("human", "own")) -> list[dict]:
    """The starts to play: with `take`, per (stratum, cell) the first `n` by a fixed hash of the start id (a prefix of that order, so a bigger
    take plays the same starts and more); without, every start of the strata. Sorted by hash order."""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for s in starts.values():
        if s.get("stratum") in strata:
            cells[(s["stratum"], s["cell"])].append(s)
    out = []
    for (stratum, cell), v in sorted(cells.items()):
        v = sorted(v, key=lambda s: order_key(s["id"]))
        n = len(v) if not take else take.get(cell, 0)
        out += v[:n]
    return sorted(out, key=lambda s: order_key(s["id"]))


def tasks_for(chosen: list[dict], draws: int, done: set, order: str = "draw") -> list[tuple]:
    """The games still to play as (arm, start id, j). `order` "draw": every start's draw 0 first, then draw 1 (a stop leaves every start
    with the same number of draws, give or take one); "start": all draws of a start together."""
    tasks = [("1x", s["id"], j) for rank, s in enumerate(chosen) for j in range(draws) if ("1x", s["id"], j) not in done]
    rank = {s["id"]: i for i, s in enumerate(chosen)}
    if order == "draw":
        tasks.sort(key=lambda t: (t[2], rank[t[1]]))
    else:
        tasks.sort(key=lambda t: (rank[t[1]], t[2]))
    return tasks


def done_keys(rows: list[dict]) -> set:
    return {(str(r["_task"][0]), str(r["_task"][1]), int(r["_task"][2])) for r in rows if r.get("_task")}


class TraceGrader:
    """`fight_gym.play_steps`'s grader hook: before each decision it notes the turn, the player's health, block and energy, and every enemy's
    health, announced move and Strength. It returns an empty grade and never changes a move; a position the engine will not write out (a card's
    choice screen) is skipped. A compact list of lists, so a fight is a few kilobytes."""

    def __init__(self) -> None:
        self.rows: list[list] = []
        self.misses = 0

    @staticmethod
    def read(doc: dict) -> list:
        p = doc.get("player") or {}
        pb = {int(b.get("id", -1)): int(b.get("amount", 0)) for b in p.get("buffs") or []}
        es = []
        for e in doc.get("enemies") or []:
            it = e.get("intent") or {}
            eb = {int(b.get("id", -1)): int(b.get("amount", 0)) for b in e.get("buffs") or []}
            es.append([int(e.get("def_id", -1)), int(e.get("hp", 0)), int(e.get("max_hp", 0)), int(e.get("block", 0)), int(it.get("type", -1)),
                       int(it.get("magnitude") or 0), int(it.get("hits") or 1), eb.get(0, 0)])
        return [int(doc.get("turn", 0)), int(p.get("hp", 0)), int(p.get("max_hp", 0)), int(p.get("block", 0)), int(p.get("energy", 0)),
                pb.get(0, 0), pb.get(2, 0), pb.get(3, 0), es]

    def __call__(self, world, k: int, best) -> dict:
        from sts2_gym import native  # noqa: PLC0415

        try:
            row = self.read(native.get_state_json(world.handle))
        except Exception:  # noqa: BLE001 -- a position the engine will not write out: no reading
            self.misses += 1
            return {}
        self.rows.append([k, str(best.kind)] + row)
        return {}


def forms_of(trace: list[list], def_id: int = TS_DEF) -> dict:
    """The Test Subject's story of one traced game: for each form (the boss's distinct maximum health in order of appearance), the player turns
    that began in it, the player's health lost over the turns that began in it (health at the turn's first decision less health at the next
    turn's first, the last turn's health to 0 when the player died), the boss's Strength when the form's first turn began, and the form and move
    at the last decision. A trace row is
    [k, kind, turn, hp, max_hp, block, energy, strength, vulnerable, weak, [[def, hp, max_hp, block, intent type, magnitude, hits, strength]]]."""
    form_of_hp: dict[int, int] = {}
    turns: list[dict] = []
    cur = None
    for r in trace:
        k, kind, turn, hp, _mx, _bl, _en, _st, _vu, _wk, enemies = r
        boss = next((e for e in enemies if e[0] == def_id), None)
        if boss is None:
            continue
        if boss[2] not in form_of_hp:
            form_of_hp[boss[2]] = len(form_of_hp) + 1
        form = form_of_hp[boss[2]]
        if cur is None or cur["turn"] != turn:
            cur = {"turn": turn, "form": form, "hp0": hp, "last_form": form, "last_move": (boss[4], boss[5], boss[6]), "boss_hp": boss[1],
                   "boss_max": boss[2], "boss_strength": boss[7], "strength0": boss[7], "last_hp": hp}
            turns.append(cur)
        cur["last_form"] = form
        cur["last_move"] = (boss[4], boss[5], boss[6])
        cur["last_hp"] = hp
        cur["boss_hp"], cur["boss_max"], cur["boss_strength"] = boss[1], boss[2], boss[7]
    per_form: dict[int, dict] = {}
    for i, t in enumerate(turns):
        nxt = turns[i + 1]["hp0"] if i + 1 < len(turns) else None
        pf = per_form.setdefault(t["form"], {"turns": 0, "hp_lost": 0.0, "lost_turns": 0, "strength_in": t["strength0"]})
        pf["turns"] += 1
        if nxt is not None:
            pf["hp_lost"] += max(0, t["hp0"] - nxt)
            pf["lost_turns"] += 1
    last = turns[-1] if turns else None
    return {"forms_reached": len(form_of_hp), "per_form": per_form, "turns": len(turns),
            "last": None if last is None else {"form": last["last_form"], "move": list(last["last_move"]), "hp": last["last_hp"],
                                               "boss_strength": last["boss_strength"], "boss_hp": last["boss_hp"], "boss_max": last["boss_max"]}}


def trace_summary(rows: list[dict]) -> dict:
    """The Test Subject's dynamics over the traced games (rows with a `trace`): by side (human or own positions) and overall, the share of
    games that reached each form, the mean turns spent in each form, the mean health lost a turn in each, and, over the games the player lost,
    the form and move at the last decision."""
    out: dict[str, dict] = {}
    for side in ("human", "own", "all"):
        games = [r for r in rows if r.get("trace") and (side == "all" or r.get("stratum") == side)]
        if not games:
            continue
        agg: dict[int, dict] = defaultdict(lambda: {"games": 0, "turns": 0, "hp_lost": 0.0, "lost_turns": 0, "strength_in": 0.0})
        deaths: Counter = Counter()
        death_forms: Counter = Counter()
        strengths: list = []
        boss_left: list = []
        turn_counts: list = []
        reached: Counter = Counter()
        lost = 0
        for r in games:
            f = forms_of(r["trace"])
            for form, v in f["per_form"].items():
                a = agg[form]
                a["games"] += 1
                a["turns"] += v["turns"]
                a["hp_lost"] += v["hp_lost"]
                a["lost_turns"] += v["lost_turns"]
                a["strength_in"] += v["strength_in"]
            reached[f["forms_reached"]] += 1
            turn_counts.append(f["turns"])
            if not won_of(r) and f["last"] is not None:
                lost += 1
                strengths.append(f["last"]["boss_strength"])
                boss_left.append(f["last"]["boss_hp"] / max(1, f["last"]["boss_max"]))
                death_forms[f["last"]["form"]] += 1
                mv = f["last"]["move"]
                deaths[(f["last"]["form"], mv[0], mv[1], mv[2])] += 1
        out[side] = {"games": len(games), "lost": lost, "turns_mean": sum(turn_counts) / len(turn_counts),
                     "death_boss_strength_mean": sum(strengths) / len(strengths) if strengths else None,
                     "death_boss_hp_share_mean": sum(boss_left) / len(boss_left) if boss_left else None,
                     "forms": {form: {"games": v["games"], "share": v["games"] / len(games), "turns_mean": v["turns"] / v["games"],
                                      "hp_lost_per_turn": v["hp_lost"] / v["lost_turns"] if v["lost_turns"] else None,
                                      "strength_in_mean": v["strength_in"] / v["games"]}
                               for form, v in sorted(agg.items())},
                     "death_forms": {k: n for k, n in sorted(death_forms.items())},
                     "death_moves": [{"form": k[0], "intent_type": k[1], "magnitude": k[2], "hits": k[3], "games": n} for k, n in deaths.most_common(6)]}
    return out


def start_means(rows: list[dict]) -> dict[str, dict]:
    """{start id: {kills over its games, games, the cell, the stratum, the human's result when there is one}}."""
    by: dict[str, dict] = {}
    for r in rows:
        s = by.setdefault(r["start"], {"kills": 0, "games": 0, "cell": r.get("cell"), "stratum": r.get("stratum"),
                                       "human": r.get("human_survived")})
        s["games"] += 1
        s["kills"] += won_of(r)
    return by


def bootstrap_mean(values: list[float], boot: int = BOOT, seed: int = BOOT_SEED) -> tuple[float, float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(boot))
    return means[int(0.025 * boot)], means[min(boot - 1, int(0.975 * boot))]


def report_data(rows: list[dict], boot: int = BOOT) -> dict:
    """Kills by side and cell, by side and boss, the human bar on the human positions, and the traces' summary."""
    idx: dict[tuple, dict] = {}
    for r in rows:
        if r.get("_task"):
            idx[(r["start"], int(r["j"]))] = r
    rows = list(idx.values())
    data: dict = {"games": len(rows), "capped": sum(1 for r in rows if r.get("capped")), "cells": {}, "bosses": {}, "human_bar": {}}

    def line(rs: list[dict]) -> dict:
        n, k = len(rs), sum(1 for r in rs if won_of(r))
        lo, hi = wilson(k, n)
        return {"games": n, "starts": len({r["start"] for r in rs}), "kills": k, "rate": k / n if n else None, "lo": lo, "hi": hi,
                "capped": sum(1 for r in rs if r.get("capped")), "hp_in": (sum(float(r.get("start_hp") or 0) / max(1, float(r.get("start_max_hp") or 1))
                                                                              for r in rs) / n) if n else None,
                "wall_mean": sum(float(r.get("seconds") or 0) for r in rs) / n if n else None}

    for side in ("human", "own"):
        for r_cell in sorted({r.get("cell") for r in rows if r.get("stratum") == side}, key=str):
            data["cells"][f"{side}|{r_cell}"] = line([r for r in rows if r.get("stratum") == side and r.get("cell") == r_cell])
        for no in (1, 2):
            for b, name in BOSS_NAMES.items():
                rs = [r for r in rows if r.get("stratum") == side and r.get("cell") == f"act3-boss{no}-{b}"]
                if rs:
                    data["bosses"][f"{side}|boss{no}|{name}"] = line(rs)
    for key, group in (("human", [r for r in rows if r.get("stratum") == "human"]),):
        by_start = start_means(group)
        for scope, pred in (("all", lambda s: True), *((c, (lambda s, c=c: s["cell"] == c)) for c in sorted({s["cell"] for s in by_start.values()}))):
            ss = [s for s in by_start.values() if pred(s) and s["human"] is not None]
            if not ss:
                continue
            diffs = [s["kills"] / s["games"] - (1.0 if s["human"] else 0.0) for s in ss]
            ci = bootstrap_mean(diffs, boot)
            hum = sum(1 for s in ss if s["human"])
            data["human_bar"][scope] = {"starts": len(ss), "champion": sum(s["kills"] / s["games"] for s in ss) / len(ss), "human": hum / len(ss),
                                        "diff": sum(diffs) / len(diffs), "lo": ci[0] if ci else None, "hi": ci[1] if ci else None}
    data["trace"] = trace_summary(rows)
    return data


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def cell_name(c: str) -> str:
    tail = c.rsplit("-", 1)[-1]
    return f"{c} ({BOSS_NAMES[int(tail)]})" if tail.isdigit() and int(tail) in BOSS_NAMES else c


def report_text(d: dict) -> str:
    L = [f"Act 3 exam: {d['games']} games, {d['capped']} capped (a capped game is a loss)", "",
         "Kills by side and cell (per 100 games, 95% interval; HP in = health going in as a share of max)",
         f"  {'side | cell':<44}{'starts':>7}{'games':>7}{'kills':>7}   {'per 100':>8}  {'interval':<14}{'HP in':>7}{'wall s':>8}"]
    for key, c in d["cells"].items():
        side, cell = key.split("|")
        L.append(f"  {side + ' | ' + cell_name(cell):<44}{c['starts']:>7}{c['games']:>7}{c['kills']:>7}   {pct(c['rate']):>8}  "
                 f"[{pct(c['lo'])}, {pct(c['hi'])}]".ljust(88) + f"{pct(c['hp_in']):>7}{c['wall_mean']:>8.0f}")
    L += ["", "By boss turn and boss (the cells above, named)"]
    for key, c in d["bosses"].items():
        L.append(f"  {key:<34} {c['kills']:>4} of {c['games']:<5} {pct(c['rate']):>6} per 100  [{pct(c['lo'])}, {pct(c['hi'])}]")
    if d["human_bar"]:
        L += ["", "The champion against the human on the same starts (per 100: the champion's kill rate, the share of those humans that went past the "
                  "boss, the paired difference by start with a 95% bootstrap interval over starts)"]
        for scope, h in d["human_bar"].items():
            L.append(f"  {cell_name(scope):<30} {h['starts']:>4} starts   champion {pct(h['champion']):>5}   human {pct(h['human']):>5}   "
                     f"diff {pct(h['diff']):>6} [{pct(h['lo'])}, {pct(h['hi'])}]")
    for side, t in d["trace"].items():
        L += ["", f"The Test Subject, {side} positions: {t['games']} traced games, {t['lost']} lost"]
        for form, v in t["forms"].items():
            hpl = "-" if v["hp_lost_per_turn"] is None else f"{v['hp_lost_per_turn']:.1f}"
            L.append(f"  form {form}: reached in {pct(v['share'])}% of games, {v['turns_mean']:.1f} turns when reached, health lost a turn {hpl}, "
                     f"the boss's Strength when it began {v['strength_in_mean']:.1f}")
        bs, bl = t["death_boss_strength_mean"], t["death_boss_hp_share_mean"]
        L.append(f"  fights last {t['turns_mean']:.1f} turns; lost in form: {t['death_forms']}"
                 + ("" if bs is None else f"; at the end of lost games the boss had {bs:.1f} Strength and {100 * bl:.0f}% of its last form's health"))
        for m in t["death_moves"]:
            L.append(f"    killed by form {m['form']} move type {m['intent_type']} {m['magnitude']} x {m['hits']}: {m['games']} games")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------- files


def read_jsonl(p: Path) -> list[dict]:
    out = []
    if Path(p).is_file():
        with open(p, encoding="utf-8") as fh:
            for x in fh:
                if x.strip():
                    try:
                        out.append(json.loads(x))
                    except ValueError:
                        pass  # a line cut by a stop: its game is played again
    return out


def say(d: Path, msg: str) -> None:
    import datetime as dt  # noqa: PLC0415

    with open(Path(d) / LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}\n")


def load_starts(exam_dir: Path) -> dict[str, dict]:
    """The Act 3 stratum's starts with their opening documents (the lite list says which; the documents are read by the byte index)."""
    import fight_exam as fe  # noqa: PLC0415
    import gym_loop as gl  # noqa: PLC0415

    p = fe.exam_paths(exam_dir)
    lite = fe.load_act3_exam(exam_dir)
    out = {}
    for sid, s in lite.items():
        out[sid] = {**s, "doc": gl.load_start(str(p["act3_starts"]), sid)["doc"]}
    return out


# ---------------------------------------------------------------------------------------------------------- the games (child side)


def _init_worker(engine: str, agent_dir: str, exam_dir: str, lib: str, cfg_name: str, trace_all: bool) -> None:
    """A worker's one-off: BelowNormal, the engine and its library, the champion, the starts (documents) and the champion's search config."""
    sys.path.insert(0, str(ROOT))
    from tools.solver import coverage  # noqa: PLC0415

    coverage._below_normal()  # noqa: SLF001 -- Windows priority class
    if lib:
        os.environ["STS2_LIB_PATH"] = lib
    import gen0  # noqa: PLC0415

    gen0.select_engine(engine)
    gen0._setup()  # noqa: SLF001
    import act3_replay as ar  # noqa: PLC0415
    import fight_gym as fg  # noqa: PLC0415
    import gen_loop  # noqa: PLC0415

    agent = gen_loop.load_agent(agent_dir)
    fg._W["agent"] = agent  # noqa: SLF001
    fg._W["a3x_starts"] = load_starts(Path(exam_dir))  # noqa: SLF001
    fg._W["a3x_cfg"] = ar.arm_config(agent.stakes, agent.hold, 1, cfg_name)  # noqa: SLF001
    fg._W["a3x_engine"] = engine  # noqa: SLF001
    fg._W["a3x_trace_all"] = trace_all  # noqa: SLF001


def game_task(task: tuple) -> dict:
    """One game: start `sid`, draw j, from the opening document, with the trace grader on."""
    import fight_gym as fg  # noqa: PLC0415
    import live_search  # noqa: PLC0415

    arm, sid, j = task[:3]
    W = fg._W  # noqa: SLF001
    start = W["a3x_starts"][sid]
    cfg, extra = W["a3x_cfg"], W["agent"].search_extra()
    world = fg._fresh_world(start["doc"], sid, j, NS)  # noqa: SLF001
    tracer = TraceGrader() if (int(start["encounter_id"]) == 80 or W.get("a3x_trace_all")) else None
    try:
        row = live_search._drive(fg.play_steps(world, start, j, cfg, extra, NS, deadline=DEADLINE_S, progress=fg._progress,  # noqa: SLF001
                                               grader=tracer), extra.get("prior_fn"))
    finally:
        world.close()
    row.pop("grades", None)
    mem = fg.private_bytes(os.getpid())
    out = {"_task": [arm, sid, j], "arm": arm, "start": sid, "j": j, "cell": start["cell"], "stratum": start["stratum"],
           "encounter_id": int(start["encounter_id"]), "boss_no": int(start["boss_no"]), "start_hp": int(start["hp"]),
           "start_max_hp": int(start["max_hp"]), "human_survived": start.get("human_survived"), "own_run_won": start.get("own_run_won"),
           "config": cfg.name, "engine": W.get("a3x_engine"), "rule": RULE, "mem_mb": round(mem / 2**20) if mem else None, **row}
    if tracer is not None:
        out["trace"] = tracer.rows
        out["trace_misses"] = tracer.misses
    return out


def hard_deadline(task: tuple) -> float:
    return DEADLINE_S * 1.5 + 60.0


def capped_row(task: tuple, reason: str, seconds: float, prog: dict) -> dict:
    arm, sid, j = task[:3]
    return {"_task": [arm, sid, j], "arm": arm, "start": sid, "j": j, "capped": True, "cap_reason": reason, "seconds": round(seconds, 2),
            "won": False, "end_hp": 0, "turns": prog.get("turns", 1), "decisions": prog.get("k", 0), "sims": prog.get("sims", 0), "rule": RULE}


# ---------------------------------------------------------------------------------------------------------- the commands


def cmd_play(a: argparse.Namespace) -> int:
    import act3_replay as ar  # noqa: PLC0415
    import fight_exam as fe  # noqa: PLC0415
    import fight_gym as fg  # noqa: PLC0415
    import gym_loop as gl  # noqa: PLC0415

    exam = Path(a.exam_dir)
    starts = fe.load_act3_exam(exam)
    if not starts:
        print(f"no Act 3 stratum in {exam}: run `fight_exam.py add-act3`")
        return 2
    chosen = select_starts(starts, parse_take(a.take), tuple(a.strata.split(",")))
    agent_dir, _agent = ar.champion(a.agent_dir)
    games = exam / GAMES
    done = done_keys(read_jsonl(games))
    tasks = tasks_for(chosen, a.draws, done, a.order)
    if not tasks:
        print("all games are played")
        return 0
    fg.MAX_PROCS = max(fg.MAX_PROCS, a.procs)
    say(exam, f"playing {len(tasks)} games of {len(chosen)} starts ({a.draws} draws) on {a.procs} processes, champion {Path(agent_dir).name}")
    gl_ctx = {"engine": a.engine, "agent": str(agent_dir), "procs": a.procs, "draws": a.draws, "take": a.take, "started": gl.stamp()}
    (exam / "act3-exam-context.json").write_text(json.dumps(gl_ctx, indent=1), encoding="utf-8")
    fe.ALLOW_QUIET.append(TAG)
    until = a.until or None
    if a.wall_hours:
        import datetime as dt  # noqa: PLC0415

        cut = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=a.wall_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        until = min(filter(None, [until, cut]))

    def stop() -> str | None:
        return fe.blocked_reason(starting=False, until=until, stop_file=exam / "STOP")

    why = fe.blocked_reason(starting=True, until=until, stop_file=exam / "STOP")
    if why:
        print(f"not started: {why}")
        say(exam, f"not started: {why}")
        return 3
    try:
        fe.prepare_process()
    except fe.Blocked as why2:
        print(f"not started: {why2}")
        return 3
    t0 = time.time()
    stopped = fg.run_tasks(tasks, game_task, games, a.procs, init=_init_worker,
                           init_args=(a.engine, str(agent_dir), str(exam), a.lib or "", a.base_config, bool(a.trace_all)), hard_of=hard_deadline,
                           mem_limit=a.mem_gb * 2**30, capped_of=capped_row, stop=stop)
    left = len(tasks_for(chosen, a.draws, done_keys(read_jsonl(games)), a.order))
    say(exam, f"{'stopped (' + stopped + ')' if stopped else 'finished'} after {time.time() - t0:.0f} s; {left} games left")
    print(f"stopped ({stopped}); {left} games left" if stopped else f"played; {left} left")
    return 3 if stopped or left else 0


def cmd_report(a: argparse.Namespace) -> int:
    exam = Path(a.exam_dir)
    rows = read_jsonl(exam / GAMES)
    if not rows:
        print(f"no games in {exam / GAMES}")
        return 1
    d = report_data(rows, a.boot)
    print(json.dumps(d, indent=1, default=str) if a.json else report_text(d))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play", help="play the Act 3 stratum (resumable)")
    p.add_argument("--exam-dir", required=True)
    p.add_argument("--agent-dir", default="", help="the champion's folder (default: referee/agent/current.json)")
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--take", default="", help="starts per cell and side, e.g. act3-boss1-80=40,act3-boss1-78=20 (a smoke); empty: every start")
    p.add_argument("--strata", default="human,own")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--engine", choices=("old", "core2"), default="old")
    p.add_argument("--lib", default="")
    p.add_argument("--base-config", default="a3-v1")
    p.add_argument("--order", choices=("draw", "start"), default="draw")
    p.add_argument("--trace-all", action="store_true", help="trace every boss, not the Test Subject only")
    p.add_argument("--mem-gb", type=float, default=2.5)
    p.add_argument("--until", default="", help="a UTC stamp YYYY-MM-DDTHH:MM:SSZ after which no game starts")
    p.add_argument("--wall-hours", type=float, default=0.0)
    p.set_defaults(fn=cmd_play)
    p = sub.add_parser("report", help="the table of a run")
    p.add_argument("--exam-dir", required=True)
    p.add_argument("--json", action="store_true")
    p.add_argument("--boot", type=int, default=BOOT)
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
